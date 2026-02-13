//! Energy Savings Task for gNB - Cell sleep modes and energy efficiency metrics
//!
//! Implements Rel-18 energy saving features:
//! - Cell sleep/dormant mode support
//! - Wake-up from sleep on paging or traffic arrival
//! - Per-cell energy efficiency metrics
//! - Energy consumption tracking

use std::collections::HashMap;
use tokio::sync::mpsc;
use tracing::{debug, info};

use crate::tasks::{EnergyMessage, GnbTaskBase, Task, TaskMessage};

/// Cell power state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellPowerState {
    /// Cell is fully active
    Active,
    /// Cell is in light sleep (reduced PDCCH monitoring)
    LightSleep,
    /// Cell is in deep sleep (carrier off, only wake-up signal)
    DeepSleep,
    /// Cell is in dormant mode (no transmission)
    Dormant,
}

/// Per-cell energy metrics.
#[derive(Debug, Clone)]
struct CellEnergyMetrics {
    /// Cell identifier
    cell_id: i32,
    /// Current power state
    power_state: CellPowerState,
    /// Estimated power consumption in watts
    power_consumption_w: f64,
    /// Total data volume in bits
    total_data_bits: u64,
    /// Total energy consumed in joules
    total_energy_j: f64,
    /// Number of connected UEs
    connected_ues: u32,
    /// Time spent in each state (seconds)
    time_in_active_s: f64,
    time_in_sleep_s: f64,
    /// Last state change timestamp (ms)
    last_state_change_ms: u64,
}

impl CellEnergyMetrics {
    fn new(cell_id: i32) -> Self {
        Self {
            cell_id,
            power_state: CellPowerState::Active,
            power_consumption_w: 100.0, // default active power
            total_data_bits: 0,
            total_energy_j: 0.0,
            connected_ues: 0,
            time_in_active_s: 0.0,
            time_in_sleep_s: 0.0,
            last_state_change_ms: 0,
        }
    }

    /// Get energy efficiency in bits per joule.
    fn bits_per_joule(&self) -> f64 {
        if self.total_energy_j > 0.0 {
            self.total_data_bits as f64 / self.total_energy_j
        } else {
            0.0
        }
    }

    /// Get power consumption for current state.
    fn state_power_w(&self) -> f64 {
        match self.power_state {
            CellPowerState::Active => 100.0,
            CellPowerState::LightSleep => 30.0,
            CellPowerState::DeepSleep => 5.0,
            CellPowerState::Dormant => 1.0,
        }
    }

    /// Update energy accounting when state changes.
    fn transition_to(&mut self, new_state: CellPowerState, timestamp_ms: u64) {
        // Account for energy consumed in previous state
        let elapsed_s = (timestamp_ms.saturating_sub(self.last_state_change_ms)) as f64 / 1000.0;
        let energy = self.state_power_w() * elapsed_s;
        self.total_energy_j += energy;

        match self.power_state {
            CellPowerState::Active => self.time_in_active_s += elapsed_s,
            _ => self.time_in_sleep_s += elapsed_s,
        }

        self.power_state = new_state;
        self.power_consumption_w = self.state_power_w();
        self.last_state_change_ms = timestamp_ms;
    }
}

pub struct EnergyTask {
    _task_base: GnbTaskBase,
    /// Per-cell energy metrics
    cells: HashMap<i32, CellEnergyMetrics>,
}

impl EnergyTask {
    pub fn new(task_base: GnbTaskBase) -> Self {
        Self {
            _task_base: task_base,
            cells: HashMap::new(),
        }
    }
}

#[async_trait::async_trait]
impl Task for EnergyTask {
    type Message = EnergyMessage;

    async fn run(&mut self, mut rx: mpsc::Receiver<TaskMessage<Self::Message>>) {
        info!("Energy Savings task started (Rel-18)");
        loop {
            match rx.recv().await {
                Some(TaskMessage::Message(msg)) => match msg {
                    EnergyMessage::CellSleep {
                        cell_id,
                        sleep_mode,
                        timestamp_ms,
                    } => {
                        let cell = self
                            .cells
                            .entry(cell_id)
                            .or_insert_with(|| CellEnergyMetrics::new(cell_id));
                        let new_state = match sleep_mode.as_str() {
                            "light" => CellPowerState::LightSleep,
                            "deep" => CellPowerState::DeepSleep,
                            "dormant" => CellPowerState::Dormant,
                            _ => CellPowerState::LightSleep,
                        };
                        cell.transition_to(new_state, timestamp_ms);
                        debug!(
                            "Energy: Cell {} entering {:?} (power={:.1}W)",
                            cell_id, new_state, cell.power_consumption_w
                        );
                    }
                    EnergyMessage::CellWakeUp {
                        cell_id,
                        reason,
                        timestamp_ms,
                    } => {
                        let cell = self
                            .cells
                            .entry(cell_id)
                            .or_insert_with(|| CellEnergyMetrics::new(cell_id));
                        cell.transition_to(CellPowerState::Active, timestamp_ms);
                        debug!("Energy: Cell {} waking up reason={}", cell_id, reason);
                    }
                    EnergyMessage::TrafficReport {
                        cell_id,
                        data_bits,
                        connected_ues,
                    } => {
                        let cell = self
                            .cells
                            .entry(cell_id)
                            .or_insert_with(|| CellEnergyMetrics::new(cell_id));
                        cell.total_data_bits += data_bits;
                        cell.connected_ues = connected_ues;
                        debug!(
                            "Energy: Cell {} traffic report bits={} ues={} efficiency={:.1} bits/J",
                            cell_id,
                            data_bits,
                            connected_ues,
                            cell.bits_per_joule()
                        );
                    }
                    EnergyMessage::GetMetrics { response_tx } => {
                        let metrics: Vec<CellEnergyReport> = self
                            .cells
                            .values()
                            .map(|c| CellEnergyReport {
                                cell_id: c.cell_id,
                                power_state: format!("{:?}", c.power_state),
                                power_consumption_w: c.power_consumption_w,
                                bits_per_joule: c.bits_per_joule(),
                                total_energy_j: c.total_energy_j,
                                time_active_pct: if (c.time_in_active_s + c.time_in_sleep_s) > 0.0
                                {
                                    c.time_in_active_s
                                        / (c.time_in_active_s + c.time_in_sleep_s)
                                        * 100.0
                                } else {
                                    100.0
                                },
                                connected_ues: c.connected_ues,
                            })
                            .collect();
                        debug!("Energy: Reporting metrics for {} cells", metrics.len());
                        if let Some(tx) = response_tx {
                            let _ = tx.send(metrics);
                        }
                    }
                },
                Some(TaskMessage::Shutdown) => break,
                None => break,
            }
        }
        info!(
            "Energy Savings task stopped, {} cells tracked",
            self.cells.len()
        );
    }
}

/// Cell energy report for metrics collection.
#[derive(Debug, Clone)]
pub struct CellEnergyReport {
    pub cell_id: i32,
    pub power_state: String,
    pub power_consumption_w: f64,
    pub bits_per_joule: f64,
    pub total_energy_j: f64,
    pub time_active_pct: f64,
    pub connected_ues: u32,
}
