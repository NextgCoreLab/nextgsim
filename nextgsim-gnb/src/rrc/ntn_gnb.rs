//! NTN-specific gNB Behavior
//!
//! Implements gNB-side NTN enhancements including:
//! - Timing advance adjustment for satellite delay
//! - HARQ process adaptation for long RTT
//! - Beam management for satellite movement
//! - Feeder link and service link coordination

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// NTN gNB configuration
#[derive(Debug, Clone)]
pub struct NtnGnbConfig {
    /// Satellite ID this gNB is serving from
    pub satellite_id: u32,
    /// Orbital altitude in meters
    pub altitude_m: f64,
    /// Whether this is a transparent satellite (bent-pipe) or regenerative
    pub is_transparent: bool,
    /// Feeder link delay in microseconds (gateway to satellite)
    pub feeder_link_delay_us: u64,
    /// Maximum service link delay in microseconds (satellite to UE)
    pub max_service_link_delay_us: u64,
    /// HARQ RTT compensation enabled
    pub harq_rtt_compensation: bool,
    /// Number of HARQ processes (increased for NTN)
    pub num_harq_processes: u8,
    /// K-offset for HARQ timing
    pub k_offset: u16,
    /// Enable Doppler pre-compensation on DL
    pub dl_doppler_precomp: bool,
    /// Beam management mode
    pub beam_mgmt_mode: BeamManagementMode,
}

/// Beam management mode for NTN
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeamManagementMode {
    /// Fixed beams (Earth-fixed cells)
    EarthFixed,
    /// Steerable beams (satellite-fixed cells)
    SatelliteFixed,
    /// Hybrid beam management
    Hybrid,
}

impl NtnGnbConfig {
    /// Create configuration for LEO transparent satellite
    pub fn leo_transparent(satellite_id: u32, altitude_m: f64) -> Self {
        Self {
            satellite_id,
            altitude_m,
            is_transparent: true,
            feeder_link_delay_us: 2000, // ~2ms typical for LEO
            max_service_link_delay_us: 10000, // ~10ms max for LEO
            harq_rtt_compensation: true,
            num_harq_processes: 16,
            k_offset: 8,
            dl_doppler_precomp: true,
            beam_mgmt_mode: BeamManagementMode::EarthFixed,
        }
    }

    /// Create configuration for GEO regenerative satellite
    pub fn geo_regenerative(satellite_id: u32) -> Self {
        Self {
            satellite_id,
            altitude_m: 35_786_000.0,
            is_transparent: false,
            feeder_link_delay_us: 120_000, // ~120ms for GEO
            max_service_link_delay_us: 140_000,
            harq_rtt_compensation: true,
            num_harq_processes: 32,
            k_offset: 64,
            dl_doppler_precomp: false,
            beam_mgmt_mode: BeamManagementMode::SatelliteFixed,
        }
    }

    /// Calculate total RTT in microseconds
    pub fn total_rtt_us(&self) -> u64 {
        if self.is_transparent {
            // Transparent: UE -> Sat -> Gateway -> Sat -> UE
            2 * (self.feeder_link_delay_us + self.max_service_link_delay_us)
        } else {
            // Regenerative: only service link delay matters for air interface
            2 * self.max_service_link_delay_us
        }
    }

    /// Get recommended TA validity timer in seconds
    pub fn ta_validity_timer_s(&self) -> u32 {
        if self.altitude_m < 2_000_000.0 {
            // LEO: shorter validity due to fast movement
            30
        } else if self.altitude_m < 20_000_000.0 {
            // MEO: medium validity
            300
        } else {
            // GEO: long validity
            3600
        }
    }
}

/// NTN beam cell information
#[derive(Debug, Clone)]
pub struct NtnBeamCell {
    /// Cell ID
    pub cell_id: u32,
    /// Beam center latitude (degrees)
    pub center_lat_deg: f64,
    /// Beam center longitude (degrees)
    pub center_lon_deg: f64,
    /// Beam radius in km (3dB beamwidth)
    pub beam_radius_km: f64,
    /// Current beam pointing direction (for steerable beams)
    pub azimuth_deg: f64,
    /// Elevation angle (for steerable beams)
    pub elevation_deg: f64,
    /// Active UEs in this beam
    pub active_ues: Vec<u64>,
}

impl NtnBeamCell {
    /// Check if a UE position is within beam coverage
    pub fn is_ue_in_beam(&self, ue_lat_deg: f64, ue_lon_deg: f64) -> bool {
        // Simplified distance calculation (Haversine approximation)
        let dlat = (ue_lat_deg - self.center_lat_deg).to_radians();
        let dlon = (ue_lon_deg - self.center_lon_deg).to_radians();

        let a = (dlat / 2.0).sin().powi(2) +
                self.center_lat_deg.to_radians().cos() *
                ue_lat_deg.to_radians().cos() *
                (dlon / 2.0).sin().powi(2);

        let distance_km = 6371.0 * 2.0 * a.sqrt().asin();

        distance_km <= self.beam_radius_km
    }

    /// Update beam pointing for satellite movement
    pub fn update_beam_pointing(&mut self, satellite_velocity_ms: f64, delta_time_s: f64) {
        // For Earth-fixed beams, need to adjust pointing as satellite moves
        // Simplified: rotate beam center based on orbital motion
        let rotation_deg = satellite_velocity_ms * delta_time_s / 111_000.0; // rough conversion

        self.center_lon_deg += rotation_deg;

        // Normalize longitude
        if self.center_lon_deg > 180.0 {
            self.center_lon_deg -= 360.0;
        } else if self.center_lon_deg < -180.0 {
            self.center_lon_deg += 360.0;
        }
    }
}

/// NTN HARQ process manager
#[derive(Debug)]
pub struct NtnHarqManager {
    /// Number of HARQ processes
    num_processes: u8,
    /// K-offset for timing
    k_offset: u16,
    /// RTT in slots
    rtt_slots: u16,
    /// HARQ process states
    process_states: HashMap<u8, HarqProcessState>,
}

#[derive(Debug, Clone)]
struct HarqProcessState {
    /// Is process busy
    busy: bool,
    /// Time when process becomes available
    available_at: Instant,
}

impl NtnHarqManager {
    /// Create new HARQ manager for NTN
    pub fn new(num_processes: u8, k_offset: u16, rtt_us: u64) -> Self {
        // Convert RTT to slots (assuming 1 slot = 1ms for simplicity)
        let rtt_slots = (rtt_us / 1000) as u16;

        let mut process_states = HashMap::new();
        for i in 0..num_processes {
            process_states.insert(i, HarqProcessState {
                busy: false,
                available_at: Instant::now(),
            });
        }

        Self {
            num_processes,
            k_offset,
            rtt_slots,
            process_states,
        }
    }

    /// Allocate HARQ process for new transmission
    pub fn allocate_process(&mut self) -> Option<u8> {
        let now = Instant::now();

        for i in 0..self.num_processes {
            if let Some(state) = self.process_states.get_mut(&i) {
                if !state.busy || now >= state.available_at {
                    state.busy = true;
                    state.available_at = now + Duration::from_millis(self.rtt_slots as u64);
                    return Some(i);
                }
            }
        }

        None // All processes busy
    }

    /// Release HARQ process
    pub fn release_process(&mut self, process_id: u8) {
        if let Some(state) = self.process_states.get_mut(&process_id) {
            state.busy = false;
        }
    }

    /// Get number of available processes
    pub fn available_processes(&self) -> usize {
        let now = Instant::now();
        self.process_states.values()
            .filter(|s| !s.busy || now >= s.available_at)
            .count()
    }

    /// Get K-offset
    pub fn k_offset(&self) -> u16 {
        self.k_offset
    }
}

/// NTN timing advance manager
#[derive(Debug)]
pub struct NtnTimingAdvanceManager {
    /// Common TA for all UEs in cell (microseconds)
    common_ta_us: u64,
    /// UE-specific TA offsets (RNTI -> offset in us)
    ue_ta_offsets: HashMap<u16, i32>,
    /// Last TA update time
    last_update: Instant,
    /// Update period
    update_period: Duration,
}

impl NtnTimingAdvanceManager {
    /// Create new TA manager
    pub fn new(initial_common_ta_us: u64, update_period_s: u32) -> Self {
        Self {
            common_ta_us: initial_common_ta_us,
            ue_ta_offsets: HashMap::new(),
            last_update: Instant::now(),
            update_period: Duration::from_secs(update_period_s as u64),
        }
    }

    /// Update common TA based on satellite movement
    pub fn update_common_ta(&mut self, new_ta_us: u64) {
        self.common_ta_us = new_ta_us;
        self.last_update = Instant::now();
    }

    /// Set UE-specific TA offset
    pub fn set_ue_ta_offset(&mut self, rnti: u16, offset_us: i32) {
        self.ue_ta_offsets.insert(rnti, offset_us);
    }

    /// Get total TA for a UE
    pub fn get_ue_ta(&self, rnti: u16) -> i64 {
        let offset = self.ue_ta_offsets.get(&rnti).copied().unwrap_or(0);
        self.common_ta_us as i64 + offset as i64
    }

    /// Check if TA update is needed
    pub fn needs_update(&self) -> bool {
        Instant::now().duration_since(self.last_update) >= self.update_period
    }

    /// Remove UE from tracking
    pub fn remove_ue(&mut self, rnti: u16) {
        self.ue_ta_offsets.remove(&rnti);
    }
}

/// NTN gNB state manager
pub struct NtnGnbManager {
    /// Configuration
    config: NtnGnbConfig,
    /// Beam cells
    beam_cells: HashMap<u32, NtnBeamCell>,
    /// HARQ manager
    harq_manager: NtnHarqManager,
    /// TA manager
    ta_manager: NtnTimingAdvanceManager,
    /// Current satellite position (lat, lon, alt in deg/deg/m)
    satellite_position: (f64, f64, f64),
}

impl NtnGnbManager {
    /// Create new NTN gNB manager
    pub fn new(config: NtnGnbConfig) -> Self {
        let harq_manager = NtnHarqManager::new(
            config.num_harq_processes,
            config.k_offset,
            config.total_rtt_us(),
        );

        let ta_manager = NtnTimingAdvanceManager::new(
            config.max_service_link_delay_us,
            config.ta_validity_timer_s(),
        );

        Self {
            config,
            beam_cells: HashMap::new(),
            harq_manager,
            ta_manager,
            satellite_position: (0.0, 0.0, 0.0),
        }
    }

    /// Add beam cell
    pub fn add_beam_cell(&mut self, cell: NtnBeamCell) {
        self.beam_cells.insert(cell.cell_id, cell);
    }

    /// Update satellite position
    pub fn update_satellite_position(&mut self, lat_deg: f64, lon_deg: f64, alt_m: f64) {
        self.satellite_position = (lat_deg, lon_deg, alt_m);
    }

    /// Update beam pointing for all cells
    pub fn update_beam_pointing(&mut self, delta_time_s: f64) {
        if self.config.beam_mgmt_mode == BeamManagementMode::EarthFixed {
            // Assume LEO orbital velocity ~7.5 km/s
            let velocity = 7500.0;

            for cell in self.beam_cells.values_mut() {
                cell.update_beam_pointing(velocity, delta_time_s);
            }
        }
    }

    /// Find best beam cell for UE
    pub fn find_best_beam(&self, ue_lat_deg: f64, ue_lon_deg: f64) -> Option<u32> {
        self.beam_cells.iter()
            .filter(|(_, cell)| cell.is_ue_in_beam(ue_lat_deg, ue_lon_deg))
            .min_by(|(_, a), (_, b)| {
                let dist_a = ((a.center_lat_deg - ue_lat_deg).powi(2) +
                             (a.center_lon_deg - ue_lon_deg).powi(2)).sqrt();
                let dist_b = ((b.center_lat_deg - ue_lat_deg).powi(2) +
                             (b.center_lon_deg - ue_lon_deg).powi(2)).sqrt();
                dist_a.partial_cmp(&dist_b).unwrap()
            })
            .map(|(id, _)| *id)
    }

    /// Get HARQ manager
    pub fn harq_manager_mut(&mut self) -> &mut NtnHarqManager {
        &mut self.harq_manager
    }

    /// Get TA manager
    pub fn ta_manager_mut(&mut self) -> &mut NtnTimingAdvanceManager {
        &mut self.ta_manager
    }

    /// Get configuration
    pub fn config(&self) -> &NtnGnbConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ntn_gnb_config() {
        let config = NtnGnbConfig::leo_transparent(1, 550_000.0);

        assert_eq!(config.satellite_id, 1);
        assert!(config.is_transparent);
        assert!(config.total_rtt_us() > 0);
    }

    #[test]
    fn test_geo_config() {
        let config = NtnGnbConfig::geo_regenerative(100);

        assert!(!config.is_transparent);
        assert!(config.total_rtt_us() > 100_000); // > 100ms RTT
    }

    #[test]
    fn test_beam_cell_coverage() {
        let beam = NtnBeamCell {
            cell_id: 1,
            center_lat_deg: 40.0,
            center_lon_deg: -74.0,
            beam_radius_km: 50.0,
            azimuth_deg: 0.0,
            elevation_deg: 45.0,
            active_ues: Vec::new(),
        };

        // UE at beam center
        assert!(beam.is_ue_in_beam(40.0, -74.0));

        // UE far away
        assert!(!beam.is_ue_in_beam(50.0, -74.0));
    }

    #[test]
    fn test_harq_manager() {
        let mut harq = NtnHarqManager::new(16, 8, 20_000);

        // Allocate processes
        let p1 = harq.allocate_process();
        assert!(p1.is_some());

        let p2 = harq.allocate_process();
        assert!(p2.is_some());
        assert_ne!(p1, p2);

        // Release process
        harq.release_process(p1.unwrap());
        assert!(harq.available_processes() > 0);
    }

    #[test]
    fn test_ta_manager() {
        let mut ta_mgr = NtnTimingAdvanceManager::new(5000, 30);

        // Set UE-specific offset
        ta_mgr.set_ue_ta_offset(100, -200);

        // Get total TA
        let ta = ta_mgr.get_ue_ta(100);
        assert_eq!(ta, 4800);

        // UE without offset
        let ta_default = ta_mgr.get_ue_ta(200);
        assert_eq!(ta_default, 5000);
    }

    #[test]
    fn test_ntn_gnb_manager() {
        let config = NtnGnbConfig::leo_transparent(1, 550_000.0);
        let mut manager = NtnGnbManager::new(config);

        // Add beam cell
        let beam = NtnBeamCell {
            cell_id: 1,
            center_lat_deg: 40.0,
            center_lon_deg: -74.0,
            beam_radius_km: 100.0,
            azimuth_deg: 0.0,
            elevation_deg: 45.0,
            active_ues: Vec::new(),
        };

        manager.add_beam_cell(beam);

        // Find best beam
        let best = manager.find_best_beam(40.0, -74.0);
        assert_eq!(best, Some(1));
    }
}
