//! Energy Saving Management (Rel-18 5G-Advanced)
//!
//! Implements cell sleep modes, energy efficiency KPIs, and load-based
//! energy saving decisions per 3GPP TS 38.300 and TS 28.310.
//!
//! # Features
//!
//! - Cell sleep mode state machine (Active, LightSleep, DeepSleep, Shutdown)
//! - Load-based cell sleep decisions
//! - Energy efficiency metrics (energy_efficiency_ratio, sleep_ratio, wake_up_latency)
//! - Inter-cell coordination for coverage compensation
//! - Time-based and load-based sleep policies

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Cell sleep mode states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellSleepMode {
    /// Cell is fully active, serving UEs
    Active,
    /// Light sleep: reduced power, quick wake-up (<10ms)
    /// Symbol-level sleep, some channels remain active
    LightSleep,
    /// Deep sleep: minimal power, slower wake-up (<100ms)
    /// RF chain off, only control plane monitoring
    DeepSleep,
    /// Full shutdown: no power, manual activation required
    Shutdown,
}

impl CellSleepMode {
    /// Returns the power consumption factor relative to Active mode
    pub fn power_factor(&self) -> f32 {
        match self {
            CellSleepMode::Active => 1.0,
            CellSleepMode::LightSleep => 0.6,   // 40% power savings
            CellSleepMode::DeepSleep => 0.2,    // 80% power savings
            CellSleepMode::Shutdown => 0.05,    // 95% power savings (residual only)
        }
    }

    /// Returns typical wake-up latency in milliseconds
    pub fn wake_up_latency_ms(&self) -> u32 {
        match self {
            CellSleepMode::Active => 0,
            CellSleepMode::LightSleep => 5,
            CellSleepMode::DeepSleep => 80,
            CellSleepMode::Shutdown => 500,
        }
    }
}

/// Energy saving policy type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnergySavingPolicy {
    /// Load-based: sleep when load is below threshold
    LoadBased,
    /// Time-based: sleep during configured time windows
    TimeBased,
    /// Hybrid: combination of load and time policies
    Hybrid,
    /// Coverage-based: sleep when neighbor cells can compensate
    CoverageBased,
}

/// Energy saving configuration for a cell
#[derive(Debug, Clone)]
pub struct EnergySavingConfig {
    /// Energy saving policy
    pub policy: EnergySavingPolicy,
    /// Load threshold for entering LightSleep (PRB utilization 0.0-1.0)
    pub light_sleep_load_threshold: f32,
    /// Load threshold for entering DeepSleep
    pub deep_sleep_load_threshold: f32,
    /// Minimum time in sleep before waking up (seconds)
    pub min_sleep_duration_s: u32,
    /// Neighbor cell IDs that can provide coverage compensation
    pub neighbor_cells: Vec<i32>,
    /// Enable inter-cell coordination
    pub inter_cell_coordination: bool,
    /// Time windows for time-based sleep (hour pairs: start, end)
    pub sleep_time_windows: Vec<(u8, u8)>,
}

impl Default for EnergySavingConfig {
    fn default() -> Self {
        Self {
            policy: EnergySavingPolicy::LoadBased,
            light_sleep_load_threshold: 0.2,  // <20% PRB usage
            deep_sleep_load_threshold: 0.05,  // <5% PRB usage
            min_sleep_duration_s: 60,
            neighbor_cells: Vec::new(),
            inter_cell_coordination: true,
            sleep_time_windows: vec![(0, 6)], // Default: 00:00-06:00
        }
    }
}

/// Cell state for energy saving management
#[derive(Debug, Clone)]
pub struct CellEnergyState {
    /// Cell identifier
    pub cell_id: i32,
    /// Current sleep mode
    pub sleep_mode: CellSleepMode,
    /// Energy saving configuration
    pub config: EnergySavingConfig,
    /// Number of connected UEs
    pub connected_ues: u32,
    /// PRB utilization (0.0 - 1.0)
    pub prb_usage: f32,
    /// Data volume transmitted (bits)
    pub data_bits_tx: u64,
    /// Data volume received (bits)
    pub data_bits_rx: u64,
    /// Total energy consumed (joules)
    pub energy_consumed_j: f64,
    /// Time when entered current sleep mode
    pub mode_entry_time: Instant,
    /// Total time spent in each mode (for statistics)
    pub time_in_active: Duration,
    pub time_in_light_sleep: Duration,
    pub time_in_deep_sleep: Duration,
    pub time_in_shutdown: Duration,
    /// Number of wake-up events
    pub wake_up_count: u32,
}

impl CellEnergyState {
    /// Creates a new cell energy state
    pub fn new(cell_id: i32, config: EnergySavingConfig) -> Self {
        Self {
            cell_id,
            sleep_mode: CellSleepMode::Active,
            config,
            connected_ues: 0,
            prb_usage: 0.0,
            data_bits_tx: 0,
            data_bits_rx: 0,
            energy_consumed_j: 0.0,
            mode_entry_time: Instant::now(),
            time_in_active: Duration::ZERO,
            time_in_light_sleep: Duration::ZERO,
            time_in_deep_sleep: Duration::ZERO,
            time_in_shutdown: Duration::ZERO,
            wake_up_count: 0,
        }
    }

    /// Transitions to a new sleep mode
    pub fn transition_to(&mut self, new_mode: CellSleepMode) {
        if new_mode == self.sleep_mode {
            return;
        }

        // Update time statistics for previous mode
        let elapsed = self.mode_entry_time.elapsed();
        match self.sleep_mode {
            CellSleepMode::Active => self.time_in_active += elapsed,
            CellSleepMode::LightSleep => self.time_in_light_sleep += elapsed,
            CellSleepMode::DeepSleep => self.time_in_deep_sleep += elapsed,
            CellSleepMode::Shutdown => self.time_in_shutdown += elapsed,
        }

        // Track wake-ups
        if self.sleep_mode != CellSleepMode::Active && new_mode == CellSleepMode::Active {
            self.wake_up_count += 1;
        }

        self.sleep_mode = new_mode;
        self.mode_entry_time = Instant::now();
    }

    /// Calculates energy efficiency ratio (bits per joule)
    pub fn energy_efficiency_ratio(&self) -> f64 {
        if self.energy_consumed_j > 0.0 {
            (self.data_bits_tx + self.data_bits_rx) as f64 / self.energy_consumed_j
        } else {
            0.0
        }
    }

    /// Calculates sleep ratio (time in sleep / total time)
    pub fn sleep_ratio(&self) -> f32 {
        let total_time = self.time_in_active
            + self.time_in_light_sleep
            + self.time_in_deep_sleep
            + self.time_in_shutdown;

        if total_time.as_secs() == 0 {
            return 0.0;
        }

        let sleep_time = self.time_in_light_sleep + self.time_in_deep_sleep + self.time_in_shutdown;
        sleep_time.as_secs_f32() / total_time.as_secs_f32()
    }

    /// Gets average wake-up latency based on mode transitions
    pub fn avg_wake_up_latency_ms(&self) -> u32 {
        if self.wake_up_count == 0 {
            return 0;
        }
        // Weighted average based on time spent in each sleep mode
        let total_sleep = self.time_in_light_sleep + self.time_in_deep_sleep + self.time_in_shutdown;
        if total_sleep.as_secs() == 0 {
            return 0;
        }

        let weighted_latency =
            (self.time_in_light_sleep.as_secs_f32() * CellSleepMode::LightSleep.wake_up_latency_ms() as f32 +
             self.time_in_deep_sleep.as_secs_f32() * CellSleepMode::DeepSleep.wake_up_latency_ms() as f32 +
             self.time_in_shutdown.as_secs_f32() * CellSleepMode::Shutdown.wake_up_latency_ms() as f32) /
            total_sleep.as_secs_f32();

        weighted_latency as u32
    }

    /// Updates energy consumption based on current mode and elapsed time
    pub fn update_energy_consumption(&mut self, base_power_w: f64, elapsed_s: f64) {
        let power_w = base_power_w * self.sleep_mode.power_factor() as f64;
        self.energy_consumed_j += power_w * elapsed_s;
    }
}

/// Energy Saving Manager
///
/// Manages energy saving decisions across multiple cells
pub struct EnergySavingManager {
    /// Cell states indexed by cell ID
    cells: HashMap<i32, CellEnergyState>,
    /// Base power consumption per cell in watts (Active mode)
    base_power_w: f64,
}

impl EnergySavingManager {
    /// Creates a new Energy Saving Manager
    pub fn new(base_power_w: f64) -> Self {
        Self {
            cells: HashMap::new(),
            base_power_w,
        }
    }

    /// Adds a cell to be managed
    pub fn add_cell(&mut self, cell_id: i32, config: EnergySavingConfig) {
        let state = CellEnergyState::new(cell_id, config);
        self.cells.insert(cell_id, state);
    }

    /// Removes a cell from management
    pub fn remove_cell(&mut self, cell_id: i32) {
        self.cells.remove(&cell_id);
    }

    /// Updates cell load metrics
    pub fn update_cell_load(&mut self, cell_id: i32, connected_ues: u32, prb_usage: f32) {
        if let Some(state) = self.cells.get_mut(&cell_id) {
            state.connected_ues = connected_ues;
            state.prb_usage = prb_usage;
        }
    }

    /// Updates cell data volume
    pub fn update_cell_data(&mut self, cell_id: i32, data_bits_tx: u64, data_bits_rx: u64) {
        if let Some(state) = self.cells.get_mut(&cell_id) {
            state.data_bits_tx += data_bits_tx;
            state.data_bits_rx += data_bits_rx;
        }
    }

    /// Makes energy saving decision for a cell
    pub fn make_sleep_decision(&mut self, cell_id: i32) -> Option<CellSleepMode> {
        let state = self.cells.get(&cell_id)?;

        // If there are connected UEs, stay active
        if state.connected_ues > 0 {
            return Some(CellSleepMode::Active);
        }

        let recommended_mode = match state.config.policy {
            EnergySavingPolicy::LoadBased => self.load_based_decision(state),
            EnergySavingPolicy::TimeBased => self.time_based_decision(state),
            EnergySavingPolicy::Hybrid => {
                let load_mode = self.load_based_decision(state);
                let time_mode = self.time_based_decision(state);
                // Take the deeper sleep mode
                if load_mode.power_factor() < time_mode.power_factor() {
                    load_mode
                } else {
                    time_mode
                }
            }
            EnergySavingPolicy::CoverageBased => {
                self.coverage_based_decision(state)
            }
        };

        // Check minimum sleep duration before allowing mode change
        if state.sleep_mode != CellSleepMode::Active {
            let time_in_mode = state.mode_entry_time.elapsed().as_secs();
            if time_in_mode < state.config.min_sleep_duration_s as u64 {
                return Some(state.sleep_mode); // Stay in current sleep mode
            }
        }

        Some(recommended_mode)
    }

    /// Load-based sleep decision
    fn load_based_decision(&self, state: &CellEnergyState) -> CellSleepMode {
        if state.prb_usage < state.config.deep_sleep_load_threshold {
            CellSleepMode::DeepSleep
        } else if state.prb_usage < state.config.light_sleep_load_threshold {
            CellSleepMode::LightSleep
        } else {
            CellSleepMode::Active
        }
    }

    /// Time-based sleep decision
    fn time_based_decision(&self, state: &CellEnergyState) -> CellSleepMode {
        // Get current time from system
        use std::time::{SystemTime, UNIX_EPOCH};
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Calculate current hour (UTC)
        let current_hour = ((now / 3600) % 24) as u8;

        for (start, end) in &state.config.sleep_time_windows {
            let in_window = if start < end {
                current_hour >= *start && current_hour < *end
            } else {
                // Wraps around midnight
                current_hour >= *start || current_hour < *end
            };

            if in_window {
                return CellSleepMode::DeepSleep;
            }
        }

        CellSleepMode::Active
    }

    /// Coverage-based sleep decision (checks if neighbors can compensate)
    fn coverage_based_decision(&self, state: &CellEnergyState) -> CellSleepMode {
        if !state.config.inter_cell_coordination {
            return CellSleepMode::Active;
        }

        // Check if at least one neighbor is active
        let has_active_neighbor = state.config.neighbor_cells.iter().any(|neighbor_id| {
            if let Some(neighbor) = self.cells.get(neighbor_id) {
                neighbor.sleep_mode == CellSleepMode::Active
            } else {
                false
            }
        });

        if has_active_neighbor {
            CellSleepMode::DeepSleep
        } else {
            CellSleepMode::Active // Must stay active for coverage
        }
    }

    /// Applies the sleep decision to a cell
    pub fn apply_sleep_decision(&mut self, cell_id: i32, new_mode: CellSleepMode) {
        if let Some(state) = self.cells.get_mut(&cell_id) {
            state.transition_to(new_mode);
        }
    }

    /// Wakes up a cell (transitions to Active mode)
    pub fn wake_up_cell(&mut self, cell_id: i32, _reason: &str) {
        if let Some(state) = self.cells.get_mut(&cell_id) {
            // Logging would happen here in production
            state.transition_to(CellSleepMode::Active);
        }
    }

    /// Updates energy consumption for all cells
    pub fn update_energy_consumption(&mut self, elapsed_s: f64) {
        for state in self.cells.values_mut() {
            state.update_energy_consumption(self.base_power_w, elapsed_s);
        }
    }

    /// Gets energy efficiency KPIs for a cell
    pub fn get_cell_kpis(&self, cell_id: i32) -> Option<CellEnergyKpis> {
        let state = self.cells.get(&cell_id)?;
        Some(CellEnergyKpis {
            cell_id,
            energy_efficiency_ratio: state.energy_efficiency_ratio(),
            sleep_ratio: state.sleep_ratio(),
            avg_wake_up_latency_ms: state.avg_wake_up_latency_ms(),
            total_energy_j: state.energy_consumed_j,
            wake_up_count: state.wake_up_count,
            current_mode: state.sleep_mode,
        })
    }

    /// Gets KPIs for all cells
    pub fn get_all_kpis(&self) -> Vec<CellEnergyKpis> {
        self.cells.keys()
            .filter_map(|cell_id| self.get_cell_kpis(*cell_id))
            .collect()
    }
}

/// Energy efficiency KPIs for a cell
#[derive(Debug, Clone)]
pub struct CellEnergyKpis {
    /// Cell identifier
    pub cell_id: i32,
    /// Energy efficiency ratio (bits per joule)
    pub energy_efficiency_ratio: f64,
    /// Sleep ratio (time in sleep / total time)
    pub sleep_ratio: f32,
    /// Average wake-up latency in milliseconds
    pub avg_wake_up_latency_ms: u32,
    /// Total energy consumed in joules
    pub total_energy_j: f64,
    /// Number of wake-up events
    pub wake_up_count: u32,
    /// Current sleep mode
    pub current_mode: CellSleepMode,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cell_sleep_mode_power_factor() {
        assert_eq!(CellSleepMode::Active.power_factor(), 1.0);
        assert_eq!(CellSleepMode::LightSleep.power_factor(), 0.6);
        assert_eq!(CellSleepMode::DeepSleep.power_factor(), 0.2);
        assert_eq!(CellSleepMode::Shutdown.power_factor(), 0.05);
    }

    #[test]
    fn test_cell_sleep_mode_wake_up_latency() {
        assert_eq!(CellSleepMode::Active.wake_up_latency_ms(), 0);
        assert_eq!(CellSleepMode::LightSleep.wake_up_latency_ms(), 5);
        assert_eq!(CellSleepMode::DeepSleep.wake_up_latency_ms(), 80);
        assert_eq!(CellSleepMode::Shutdown.wake_up_latency_ms(), 500);
    }

    #[test]
    fn test_energy_state_creation() {
        let config = EnergySavingConfig::default();
        let state = CellEnergyState::new(1, config);
        assert_eq!(state.cell_id, 1);
        assert_eq!(state.sleep_mode, CellSleepMode::Active);
        assert_eq!(state.connected_ues, 0);
        assert_eq!(state.wake_up_count, 0);
    }

    #[test]
    fn test_energy_state_transition() {
        let config = EnergySavingConfig::default();
        let mut state = CellEnergyState::new(1, config);

        state.transition_to(CellSleepMode::LightSleep);
        assert_eq!(state.sleep_mode, CellSleepMode::LightSleep);
        assert_eq!(state.wake_up_count, 0);

        state.transition_to(CellSleepMode::Active);
        assert_eq!(state.sleep_mode, CellSleepMode::Active);
        assert_eq!(state.wake_up_count, 1);
    }

    #[test]
    fn test_energy_consumption_calculation() {
        let config = EnergySavingConfig::default();
        let mut state = CellEnergyState::new(1, config);

        // Active mode: 100W for 1 second = 100J
        state.update_energy_consumption(100.0, 1.0);
        assert!((state.energy_consumed_j - 100.0).abs() < 0.1);

        // LightSleep: 100W * 0.6 for 1 second = 60J
        state.transition_to(CellSleepMode::LightSleep);
        state.update_energy_consumption(100.0, 1.0);
        assert!((state.energy_consumed_j - 160.0).abs() < 0.1);
    }

    #[test]
    fn test_energy_efficiency_ratio() {
        let config = EnergySavingConfig::default();
        let mut state = CellEnergyState::new(1, config);

        state.data_bits_tx = 1_000_000;
        state.data_bits_rx = 500_000;
        state.energy_consumed_j = 100.0;

        let ratio = state.energy_efficiency_ratio();
        assert_eq!(ratio, 15_000.0); // 1.5M bits / 100J
    }

    #[test]
    fn test_energy_saving_manager() {
        let mut manager = EnergySavingManager::new(100.0);
        let config = EnergySavingConfig::default();

        manager.add_cell(1, config);
        assert!(manager.cells.contains_key(&1));

        manager.remove_cell(1);
        assert!(!manager.cells.contains_key(&1));
    }

    #[test]
    fn test_load_based_decision() {
        let mut manager = EnergySavingManager::new(100.0);
        let config = EnergySavingConfig {
            policy: EnergySavingPolicy::LoadBased,
            light_sleep_load_threshold: 0.2,
            deep_sleep_load_threshold: 0.05,
            ..Default::default()
        };

        manager.add_cell(1, config);

        // High load -> Active
        manager.update_cell_load(1, 0, 0.5);
        let decision = manager.make_sleep_decision(1);
        assert_eq!(decision, Some(CellSleepMode::Active));

        // Medium load -> LightSleep
        manager.update_cell_load(1, 0, 0.1);
        let decision = manager.make_sleep_decision(1);
        assert_eq!(decision, Some(CellSleepMode::LightSleep));

        // Low load -> DeepSleep
        manager.update_cell_load(1, 0, 0.02);
        let decision = manager.make_sleep_decision(1);
        assert_eq!(decision, Some(CellSleepMode::DeepSleep));
    }

    #[test]
    fn test_connected_ues_prevents_sleep() {
        let mut manager = EnergySavingManager::new(100.0);
        let config = EnergySavingConfig::default();

        manager.add_cell(1, config);

        // Low load but UEs connected -> Active
        manager.update_cell_load(1, 5, 0.01);
        let decision = manager.make_sleep_decision(1);
        assert_eq!(decision, Some(CellSleepMode::Active));
    }

    #[test]
    fn test_wake_up_cell() {
        let mut manager = EnergySavingManager::new(100.0);
        let config = EnergySavingConfig::default();

        manager.add_cell(1, config);
        manager.apply_sleep_decision(1, CellSleepMode::DeepSleep);

        manager.wake_up_cell(1, "paging");
        let state = manager.cells.get(&1).unwrap();
        assert_eq!(state.sleep_mode, CellSleepMode::Active);
        assert_eq!(state.wake_up_count, 1);
    }
}
