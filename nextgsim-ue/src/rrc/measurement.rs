//! Measurement Framework for Handover
//!
//! Implements measurement configuration and reporting per 3GPP TS 38.331.
//!
//! # Overview
//!
//! In RRC_CONNECTED state, the UE performs measurements and reports them
//! to the network based on measurement configuration provided by the gNB.
//! The network uses these reports to make handover decisions.
//!
//! # Measurement Types
//!
//! - **SS-RSRP**: Synchronization Signal Reference Signal Received Power
//! - **SS-RSRQ**: Synchronization Signal Reference Signal Received Quality
//! - **SS-SINR**: Synchronization Signal Signal-to-Interference-plus-Noise Ratio
//!
//! # Event Types (for event-triggered reporting)
//!
//! - **A1**: Serving becomes better than threshold
//! - **A2**: Serving becomes worse than threshold
//! - **A3**: Neighbor becomes amount better than serving
//! - **A4**: Neighbor becomes better than threshold
//! - **A5**: Serving becomes worse than threshold1 AND neighbor becomes better than threshold2
//!
//! # Reference
//! - 3GPP TS 38.331: NR; RRC protocol specification
//! - 3GPP TS 38.215: NR; Physical layer measurements

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Measurement quantity types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MeasQuantity {
    /// SS-RSRP (Reference Signal Received Power)
    SsRsrp,
    /// SS-RSRQ (Reference Signal Received Quality)
    SsRsrq,
    /// SS-SINR (Signal-to-Interference-plus-Noise Ratio)
    SsSinr,
}

impl Default for MeasQuantity {
    fn default() -> Self {
        Self::SsRsrp
    }
}

/// Measurement event types for event-triggered reporting
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasEventType {
    /// A1: Serving becomes better than threshold
    A1,
    /// A2: Serving becomes worse than threshold
    A2,
    /// A3: Neighbor becomes offset better than serving
    A3,
    /// A4: Neighbor becomes better than threshold
    A4,
    /// A5: Serving < threshold1 AND neighbor > threshold2
    A5,
}

/// Report trigger type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportTriggerType {
    /// Event-triggered reporting
    Event(MeasEventType),
    /// Periodic reporting
    Periodic,
}

/// Measurement report trigger configuration
#[derive(Debug, Clone)]
pub struct ReportTriggerConfig {
    /// Type of trigger
    pub trigger_type: ReportTriggerType,
    /// Threshold for A1/A2/A4 events (dBm)
    pub threshold: Option<i32>,
    /// Threshold1 for A5 event (serving cell, dBm)
    pub threshold1: Option<i32>,
    /// Threshold2 for A5 event (neighbor cell, dBm)
    pub threshold2: Option<i32>,
    /// Offset for A3 event (dB)
    pub a3_offset: Option<i32>,
    /// Hysteresis (dB * 2, e.g., 4 = 2 dB)
    pub hysteresis: i32,
    /// Time-to-trigger (ms)
    pub time_to_trigger: u64,
}

impl Default for ReportTriggerConfig {
    fn default() -> Self {
        Self {
            trigger_type: ReportTriggerType::Event(MeasEventType::A3),
            threshold: None,
            threshold1: None,
            threshold2: None,
            a3_offset: Some(3), // 3 dB offset
            hysteresis: 2,      // 1 dB
            time_to_trigger: 640, // 640 ms
        }
    }
}

/// Measurement configuration for a single measurement ID
#[derive(Debug, Clone)]
pub struct MeasConfig {
    /// Measurement ID
    pub meas_id: u8,
    /// Measurement object ID (identifies frequency/cell to measure)
    pub meas_object_id: u8,
    /// Report configuration ID
    pub report_config_id: u8,
    /// Measurement quantity
    pub quantity: MeasQuantity,
    /// Report trigger configuration
    pub trigger_config: ReportTriggerConfig,
    /// Report amount (0 = infinite)
    pub report_amount: u32,
    /// Report interval (ms)
    pub report_interval: u64,
    /// Max report cells
    pub max_report_cells: u8,
}

impl Default for MeasConfig {
    fn default() -> Self {
        Self {
            meas_id: 1,
            meas_object_id: 1,
            report_config_id: 1,
            quantity: MeasQuantity::SsRsrp,
            trigger_config: ReportTriggerConfig::default(),
            report_amount: 8,
            report_interval: 480,
            max_report_cells: 4,
        }
    }
}

/// Measurement result for a single cell
#[derive(Debug, Clone, Default)]
pub struct CellMeasResult {
    /// Physical cell ID
    pub pci: u32,
    /// NR Cell Global Identity (if available)
    pub nci: Option<i64>,
    /// SS-RSRP measurement (dBm)
    pub rsrp: Option<i32>,
    /// SS-RSRQ measurement (dB)
    pub rsrq: Option<i32>,
    /// SS-SINR measurement (dB)
    pub sinr: Option<i32>,
}

/// Measurement report to be sent to network
#[derive(Debug, Clone)]
pub struct MeasurementReport {
    /// Measurement ID
    pub meas_id: u8,
    /// Serving cell results
    pub serving_cell: CellMeasResult,
    /// Neighbor cell results
    pub neighbor_cells: Vec<CellMeasResult>,
    /// Timestamp of the report
    pub timestamp: Instant,
}

/// Event state tracking for a single event
#[derive(Debug, Clone, Default)]
struct EventState {
    /// Whether the event condition is currently met
    condition_met: bool,
    /// Cell that triggered the condition (for neighbor events)
    triggering_cell: Option<i32>,
    /// Time when condition was first met
    condition_met_since: Option<Instant>,
    /// Number of reports sent for this event
    reports_sent: u32,
    /// Last report time
    last_report: Option<Instant>,
}

/// Measurement manager for handling RRC measurements
pub struct MeasurementManager {
    /// Active measurement configurations
    configs: HashMap<u8, MeasConfig>,
    /// Current measurements per cell (cell_id -> measurement)
    measurements: HashMap<i32, CellMeasResult>,
    /// Event states per measurement ID
    event_states: HashMap<u8, EventState>,
    /// Serving cell ID
    serving_cell_id: Option<i32>,
    /// Pending reports to be sent
    pending_reports: Vec<MeasurementReport>,
}

impl MeasurementManager {
    pub fn new() -> Self {
        Self {
            configs: HashMap::new(),
            measurements: HashMap::new(),
            event_states: HashMap::new(),
            serving_cell_id: None,
            pending_reports: Vec::new(),
        }
    }

    /// Set the serving cell
    pub fn set_serving_cell(&mut self, cell_id: Option<i32>) {
        self.serving_cell_id = cell_id;
        // Reset event states on serving cell change
        for state in self.event_states.values_mut() {
            state.condition_met = false;
            state.condition_met_since = None;
            state.triggering_cell = None;
        }
    }

    /// Add a measurement configuration
    pub fn add_config(&mut self, config: MeasConfig) {
        let meas_id = config.meas_id;
        self.configs.insert(meas_id, config);
        self.event_states.insert(meas_id, EventState::default());
    }

    /// Remove a measurement configuration
    pub fn remove_config(&mut self, meas_id: u8) {
        self.configs.remove(&meas_id);
        self.event_states.remove(&meas_id);
    }

    /// Clear all measurement configurations
    pub fn clear_configs(&mut self) {
        self.configs.clear();
        self.event_states.clear();
    }

    /// Update measurement for a cell
    pub fn update_measurement(&mut self, cell_id: i32, rsrp: i32) {
        let result = self.measurements.entry(cell_id).or_insert_with(|| CellMeasResult {
            pci: cell_id as u32,
            ..Default::default()
        });
        result.rsrp = Some(rsrp);
    }

    /// Remove measurement for a cell
    pub fn remove_measurement(&mut self, cell_id: i32) {
        self.measurements.remove(&cell_id);
    }

    /// Get pending reports and clear them
    pub fn take_pending_reports(&mut self) -> Vec<MeasurementReport> {
        std::mem::take(&mut self.pending_reports)
    }

    /// Evaluate measurement events and generate reports if needed
    pub fn evaluate_events(&mut self) {
        let serving_cell_id = match self.serving_cell_id {
            Some(id) => id,
            None => return,
        };

        let serving_rsrp = self.measurements.get(&serving_cell_id)
            .and_then(|m| m.rsrp)
            .unwrap_or(i32::MIN);

        // Evaluate each measurement configuration
        let configs: Vec<_> = self.configs.values().cloned().collect();
        for config in configs {
            self.evaluate_event(&config, serving_cell_id, serving_rsrp);
        }
    }

    fn evaluate_event(&mut self, config: &MeasConfig, serving_cell_id: i32, serving_rsrp: i32) {
        let meas_id = config.meas_id;
        let trigger = &config.trigger_config;

        let (condition_met, triggering_cell) = match trigger.trigger_type {
            ReportTriggerType::Event(event_type) => {
                self.check_event_condition(
                    event_type,
                    trigger,
                    serving_cell_id,
                    serving_rsrp,
                )
            }
            ReportTriggerType::Periodic => (true, None),
        };

        // Check if we need to generate a report
        let mut generate_report = false;
        let mut report_params: Option<(u8, i32, Option<i32>, u8)> = None;

        {
            let state = self.event_states.entry(meas_id).or_default();

            if condition_met {
                if !state.condition_met {
                    // Condition just became true
                    state.condition_met = true;
                    state.triggering_cell = triggering_cell;
                    state.condition_met_since = Some(Instant::now());
                }

                // Check time-to-trigger
                if let Some(since) = state.condition_met_since {
                    if since.elapsed() >= Duration::from_millis(trigger.time_to_trigger) {
                        // Check if we should send a report
                        let should_report = state.last_report.map_or(true, |last| {
                            last.elapsed() >= Duration::from_millis(config.report_interval)
                        });

                        let reports_remaining = config.report_amount == 0 ||
                            state.reports_sent < config.report_amount;

                        if should_report && reports_remaining {
                            generate_report = true;
                            report_params = Some((meas_id, serving_cell_id, triggering_cell, config.max_report_cells));
                            state.reports_sent += 1;
                            state.last_report = Some(Instant::now());

                            tracing::info!(
                                "Measurement report generated: meas_id={}, event={:?}, reports_sent={}",
                                meas_id, trigger.trigger_type, state.reports_sent
                            );
                        }
                    }
                }
            } else {
                // Condition no longer met - reset
                state.condition_met = false;
                state.condition_met_since = None;
                state.triggering_cell = None;
            }
        }

        // Generate report outside the borrow scope
        if generate_report {
            if let Some((meas_id, serving_cell_id, triggering_cell, max_cells)) = report_params {
                let report = self.generate_report(meas_id, serving_cell_id, triggering_cell, max_cells);
                self.pending_reports.push(report);
            }
        }
    }

    fn check_event_condition(
        &self,
        event_type: MeasEventType,
        trigger: &ReportTriggerConfig,
        serving_cell_id: i32,
        serving_rsrp: i32,
    ) -> (bool, Option<i32>) {
        let hyst = trigger.hysteresis / 2; // Convert to dB

        match event_type {
            MeasEventType::A1 => {
                // Serving > threshold
                if let Some(thresh) = trigger.threshold {
                    (serving_rsrp > thresh + hyst, None)
                } else {
                    (false, None)
                }
            }
            MeasEventType::A2 => {
                // Serving < threshold
                if let Some(thresh) = trigger.threshold {
                    (serving_rsrp < thresh - hyst, None)
                } else {
                    (false, None)
                }
            }
            MeasEventType::A3 => {
                // Neighbor > serving + offset
                let offset = trigger.a3_offset.unwrap_or(0);
                let mut best_neighbor: Option<(i32, i32)> = None;

                for (&cell_id, meas) in &self.measurements {
                    if cell_id == serving_cell_id {
                        continue;
                    }
                    if let Some(rsrp) = meas.rsrp {
                        if rsrp > serving_rsrp + offset + hyst {
                            if best_neighbor.map_or(true, |(_, best_rsrp)| rsrp > best_rsrp) {
                                best_neighbor = Some((cell_id, rsrp));
                            }
                        }
                    }
                }

                best_neighbor.map_or((false, None), |(cell_id, _)| (true, Some(cell_id)))
            }
            MeasEventType::A4 => {
                // Neighbor > threshold
                if let Some(thresh) = trigger.threshold {
                    let mut best_neighbor: Option<(i32, i32)> = None;

                    for (&cell_id, meas) in &self.measurements {
                        if cell_id == serving_cell_id {
                            continue;
                        }
                        if let Some(rsrp) = meas.rsrp {
                            if rsrp > thresh + hyst {
                                if best_neighbor.map_or(true, |(_, best_rsrp)| rsrp > best_rsrp) {
                                    best_neighbor = Some((cell_id, rsrp));
                                }
                            }
                        }
                    }

                    best_neighbor.map_or((false, None), |(cell_id, _)| (true, Some(cell_id)))
                } else {
                    (false, None)
                }
            }
            MeasEventType::A5 => {
                // Serving < threshold1 AND neighbor > threshold2
                let thresh1 = trigger.threshold1.unwrap_or(i32::MIN);
                let thresh2 = trigger.threshold2.unwrap_or(i32::MAX);

                if serving_rsrp >= thresh1 - hyst {
                    return (false, None);
                }

                let mut best_neighbor: Option<(i32, i32)> = None;
                for (&cell_id, meas) in &self.measurements {
                    if cell_id == serving_cell_id {
                        continue;
                    }
                    if let Some(rsrp) = meas.rsrp {
                        if rsrp > thresh2 + hyst {
                            if best_neighbor.map_or(true, |(_, best_rsrp)| rsrp > best_rsrp) {
                                best_neighbor = Some((cell_id, rsrp));
                            }
                        }
                    }
                }

                best_neighbor.map_or((false, None), |(cell_id, _)| (true, Some(cell_id)))
            }
        }
    }

    fn generate_report(
        &self,
        meas_id: u8,
        serving_cell_id: i32,
        triggering_cell: Option<i32>,
        max_cells: u8,
    ) -> MeasurementReport {
        // Get serving cell measurement
        let serving_cell = self.measurements.get(&serving_cell_id)
            .cloned()
            .unwrap_or_default();

        // Get neighbor cell measurements, sorted by RSRP
        let mut neighbors: Vec<_> = self.measurements.iter()
            .filter(|(&id, _)| id != serving_cell_id)
            .map(|(_, m)| m.clone())
            .collect();

        neighbors.sort_by(|a, b| {
            b.rsrp.unwrap_or(i32::MIN).cmp(&a.rsrp.unwrap_or(i32::MIN))
        });

        // If there's a triggering cell, put it first
        if let Some(trig_id) = triggering_cell {
            if let Some(pos) = neighbors.iter().position(|m| m.pci == trig_id as u32) {
                let trig = neighbors.remove(pos);
                neighbors.insert(0, trig);
            }
        }

        // Limit to max cells
        neighbors.truncate(max_cells as usize);

        MeasurementReport {
            meas_id,
            serving_cell,
            neighbor_cells: neighbors,
            timestamp: Instant::now(),
        }
    }

    /// Get the number of active measurement configurations
    pub fn config_count(&self) -> usize {
        self.configs.len()
    }

    /// Get all cell measurements
    pub fn measurements(&self) -> &HashMap<i32, CellMeasResult> {
        &self.measurements
    }
}

impl Default for MeasurementManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_measurement_manager_creation() {
        let manager = MeasurementManager::new();
        assert!(manager.serving_cell_id.is_none());
        assert_eq!(manager.config_count(), 0);
    }

    #[test]
    fn test_add_measurement_config() {
        let mut manager = MeasurementManager::new();
        let config = MeasConfig::default();
        manager.add_config(config);
        assert_eq!(manager.config_count(), 1);
    }

    #[test]
    fn test_update_measurement() {
        let mut manager = MeasurementManager::new();
        manager.update_measurement(1, -80);
        assert_eq!(manager.measurements().len(), 1);
        assert_eq!(manager.measurements()[&1].rsrp, Some(-80));
    }

    #[test]
    fn test_a3_event_detection() {
        let mut manager = MeasurementManager::new();

        // Set up serving cell
        manager.set_serving_cell(Some(1));
        manager.update_measurement(1, -90);

        // Set up neighbor cell that is better
        manager.update_measurement(2, -80);

        // Add A3 configuration
        let mut config = MeasConfig::default();
        config.trigger_config.trigger_type = ReportTriggerType::Event(MeasEventType::A3);
        config.trigger_config.a3_offset = Some(3);
        config.trigger_config.time_to_trigger = 0; // Immediate for test
        manager.add_config(config);

        // Evaluate - should detect A3 event
        manager.evaluate_events();

        // Check event state
        let state = manager.event_states.get(&1).unwrap();
        assert!(state.condition_met);
        assert_eq!(state.triggering_cell, Some(2));
    }

    #[test]
    fn test_a2_event_detection() {
        let mut manager = MeasurementManager::new();

        // Set up serving cell with weak signal
        manager.set_serving_cell(Some(1));
        manager.update_measurement(1, -100);

        // Add A2 configuration (serving worse than threshold)
        let mut config = MeasConfig::default();
        config.meas_id = 2;
        config.trigger_config.trigger_type = ReportTriggerType::Event(MeasEventType::A2);
        config.trigger_config.threshold = Some(-90);
        config.trigger_config.time_to_trigger = 0;
        manager.add_config(config);

        // Evaluate - should detect A2 event
        manager.evaluate_events();

        let state = manager.event_states.get(&2).unwrap();
        assert!(state.condition_met);
    }
}
