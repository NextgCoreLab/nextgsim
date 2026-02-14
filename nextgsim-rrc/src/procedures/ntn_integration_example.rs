//! NTN Integration Example
//!
//! This module demonstrates how to use the NTN satellite simulation features together.
//! It shows realistic scenarios for LEO/MEO/GEO satellite constellations with:
//! - Satellite position calculation and tracking
//! - Link budget and propagation modeling
//! - Handover prediction and execution
//! - Timing advance management
//! - Doppler compensation
//!
//! This is primarily for documentation and testing purposes.

#![allow(dead_code)]

use super::ntn_constellation::{Constellation, ConstellationConfig, GeodeticPosition};
use super::ntn_link_sim::{NtnLinkSimulator, GroundPosition};
use super::isl_handover::{IslHandoverManager, SatellitePosition as IslSatPosition};
use super::ntn_timing::{NtnTimingAdvance, SatelliteOrbitType, NtnEphemerisData, KeplerianElements, EphemerisSource};

/// Complete NTN simulation scenario
pub struct NtnScenario {
    /// Satellite constellation
    constellation: Constellation,
    /// Link simulator for active satellite
    link_sim: NtnLinkSimulator,
    /// Handover manager
    handover_mgr: IslHandoverManager,
    /// Current serving satellite
    serving_satellite: u32,
    /// UE ground position
    ue_position: GeodeticPosition,
    /// Simulation time in seconds
    time_s: f64,
}

impl NtnScenario {
    /// Create a new LEO constellation scenario
    ///
    /// Returns None if the constellation cannot be created or no satellite is visible
    pub fn new_leo_scenario(ue_lat_deg: f64, ue_lon_deg: f64) -> Option<Self> {
        let config = ConstellationConfig::starlink_like();
        let mut constellation = Constellation::new(config).ok()?;

        constellation.set_time(0.0);

        let ue_position = GeodeticPosition::from_degrees(ue_lat_deg, ue_lon_deg, 0.0);

        // Find initial serving satellite
        let serving_satellite = constellation.find_best_satellite(&ue_position).ok()?;

        // Create link simulator for the serving satellite
        let link_sim = NtnLinkSimulator::new_leo(550.0);

        let handover_mgr = IslHandoverManager::new();

        Some(Self {
            constellation,
            link_sim,
            handover_mgr,
            serving_satellite,
            ue_position,
            time_s: 0.0,
        })
    }

    /// Create a GEO satellite scenario
    ///
    /// Returns None if the constellation cannot be created
    pub fn new_geo_scenario(ue_lat_deg: f64, ue_lon_deg: f64) -> Option<Self> {
        let config = ConstellationConfig::geo();
        let mut constellation = Constellation::new(config).ok()?;

        constellation.set_time(0.0);

        let ue_position = GeodeticPosition::from_degrees(ue_lat_deg, ue_lon_deg, 0.0);

        let serving_satellite = 1; // GEO has only one satellite

        let link_sim = NtnLinkSimulator::new_geo();

        let handover_mgr = IslHandoverManager::new();

        Some(Self {
            constellation,
            link_sim,
            handover_mgr,
            serving_satellite,
            ue_position,
            time_s: 0.0,
        })
    }

    /// Advance simulation by time_step seconds
    pub fn step(&mut self, time_step_s: f64) {
        self.time_s += time_step_s;
        self.constellation.set_time(self.time_s);

        // Check if handover is needed
        if let Some((new_sat, handover_time)) = self.constellation.predict_handover(
            &self.ue_position,
            self.serving_satellite,
            1.0, // 1 second time step
            60.0, // 60 second lookahead
        ) {
            if handover_time < 5.0 {
                // Handover imminent (within 5 seconds)
                self.initiate_handover(new_sat);
            }
        }

        // Update satellite positions in handover manager
        if let Ok(state) = self.constellation.calculate_satellite_state(self.serving_satellite) {
            let isl_pos = IslSatPosition {
                x_km: state.position.x / 1000.0,
                y_km: state.position.y / 1000.0,
                z_km: state.position.z / 1000.0,
                timestamp_ms: (self.time_s * 1000.0) as u64,
            };
            self.handover_mgr.update_satellite_position(self.serving_satellite, isl_pos);
        }
    }

    /// Initiate handover to a new satellite
    fn initiate_handover(&mut self, new_satellite: u32) {
        if new_satellite == self.serving_satellite {
            return;
        }

        // Initiate ISL handover
        let _ = self.handover_mgr.initiate_isl_handover(
            self.serving_satellite,
            new_satellite,
            1, // service link ID
            None,
        );

        // Prepare and execute handover
        let _ = self.handover_mgr.prepare_target_satellite();
        let _ = self.handover_mgr.execute_isl_switch();

        self.serving_satellite = new_satellite;
    }

    /// Get current link quality
    ///
    /// Returns None if satellite state cannot be calculated
    pub fn get_link_quality(&mut self) -> Option<LinkQuality> {
        let ground = GroundPosition {
            latitude_deg: self.ue_position.latitude_deg(),
            longitude_deg: self.ue_position.longitude_deg(),
            altitude_km: self.ue_position.altitude_m / 1000.0,
        };

        let link_result = self.link_sim.simulate(&ground);

        // Get satellite state for additional info
        let sat_state = self.constellation.calculate_satellite_state(self.serving_satellite).ok()?;

        let doppler = sat_state.calculate_doppler(&self.ue_position, 2e9);
        let propagation_delay = sat_state.propagation_delay_ms(&self.ue_position);
        let visibility = sat_state.calculate_visibility(&self.ue_position);

        Some(LinkQuality {
            snr_db: link_result.snr_db,
            delay_ms: propagation_delay,
            doppler_hz: doppler,
            elevation_deg: visibility.elevation_deg,
            azimuth_deg: visibility.azimuth_deg,
            link_margin_db: link_result.link_margin_db,
            serving_satellite: self.serving_satellite,
        })
    }

    /// Get timing advance configuration for UE
    ///
    /// Returns None if satellite state cannot be calculated
    pub fn get_timing_advance(&self) -> Option<NtnTimingAdvance> {
        let sat_state = self.constellation.calculate_satellite_state(self.serving_satellite).ok()?;

        let delay_ms = sat_state.propagation_delay_ms(&self.ue_position);
        let delay_us = (delay_ms * 1000.0) as u64;

        // Get satellite position
        let sat_pos_geo = sat_state.position.to_geodetic();

        // Create ephemeris data
        let ephemeris = NtnEphemerisData {
            satellite_id: self.serving_satellite,
            orbital_elements: KeplerianElements {
                semi_major_axis_m: 6_371_000.0 + 550_000.0,
                eccentricity: 0.001,
                inclination_deg: 53.0,
                raan_deg: 0.0,
                argument_of_periapsis_deg: 0.0,
                mean_anomaly_deg: 0.0,
            },
            epoch_time_ms: (self.time_s * 1000.0) as u64,
            source: EphemerisSource::BroadcastSib,
            validity_duration_s: 3600,
            orbit_type: SatelliteOrbitType::Leo,
        };

        Some(NtnTimingAdvance {
            config_id: 1,
            orbit_type: SatelliteOrbitType::Leo,
            satellite_id: self.serving_satellite,
            common_ta_us: delay_us,
            ue_specific_ta_offset_us: 0,
            ta_update_periodicity_ms: 1000,
            ephemeris: Some(ephemeris),
            satellite_position: Some(super::ntn_timing::SatellitePositionGeodetic {
                latitude_deg: sat_pos_geo.latitude_deg(),
                longitude_deg: sat_pos_geo.longitude_deg(),
                altitude_m: sat_pos_geo.altitude_m,
            }),
            k_offset: 16,
            max_doppler_shift_hz: 40000.0,
            dl_doppler_precompensated: true,
            ul_doppler_compensation: true,
            ta_validity_timer_s: 30,
            autonomous_ta_enabled: true,
            guard_time_us: 50,
        })
    }

    /// Get handover statistics
    pub fn get_handover_stats(&self) -> super::isl_handover::HandoverStats {
        self.handover_mgr.get_handover_stats()
    }
}

/// Link quality snapshot
#[derive(Debug, Clone)]
pub struct LinkQuality {
    /// SNR in dB
    pub snr_db: f64,
    /// Propagation delay in ms
    pub delay_ms: f64,
    /// Doppler shift in Hz
    pub doppler_hz: f64,
    /// Elevation angle in degrees
    pub elevation_deg: f64,
    /// Azimuth angle in degrees
    pub azimuth_deg: f64,
    /// Link margin in dB
    pub link_margin_db: f64,
    /// Current serving satellite ID
    pub serving_satellite: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leo_scenario() {
        let mut scenario = NtnScenario::new_leo_scenario(40.0, -74.0).unwrap();

        // Get initial link quality
        let lq = scenario.get_link_quality().unwrap();
        assert!(lq.snr_db > -100.0 && lq.snr_db < 100.0);
        assert!(lq.delay_ms > 0.0);
        assert!(lq.elevation_deg >= 5.0);

        // Step simulation
        scenario.step(1.0);

        // Get updated link quality
        let lq2 = scenario.get_link_quality().unwrap();
        assert!(lq2.snr_db > -100.0 && lq2.snr_db < 100.0);
    }

    #[test]
    fn test_geo_scenario() {
        let mut scenario = NtnScenario::new_geo_scenario(0.0, 0.0).unwrap();

        let lq = scenario.get_link_quality().unwrap();
        assert!(lq.delay_ms > 100.0); // GEO should have > 100ms delay
        assert!(lq.doppler_hz.abs() < 1000.0); // GEO has minimal Doppler
    }

    #[test]
    fn test_timing_advance() {
        let scenario = NtnScenario::new_leo_scenario(40.0, -74.0).unwrap();

        let ta = scenario.get_timing_advance().unwrap();
        assert!(ta.validate().is_ok());
        assert!(ta.common_ta_us > 0);
        assert!(ta.max_doppler_shift_hz > 0.0);
    }

    #[test]
    fn test_simulation_step() {
        let mut scenario = NtnScenario::new_leo_scenario(40.0, -74.0).unwrap();

        let _initial_sat = scenario.serving_satellite;

        // Step through several seconds
        for _ in 0..10 {
            scenario.step(1.0);
        }

        // Satellite might still be the same (depends on constellation)
        assert!(scenario.serving_satellite > 0);

        // Time should have advanced
        assert!(scenario.time_s > 0.0);
    }

    #[test]
    fn test_handover_stats() {
        let scenario = NtnScenario::new_leo_scenario(40.0, -74.0).unwrap();

        let stats = scenario.get_handover_stats();
        // total_handovers is unsigned, so it's always >= 0 - just verify it exists
        let _ = stats.total_handovers;
    }
}
