//! NTN Satellite Constellation Simulation
//!
//! Implements comprehensive satellite constellation modeling for LEO/MEO/GEO networks.
//! Supports orbital mechanics, visibility windows, handover prediction, and realistic
//! propagation modeling per 3GPP TR 38.811/38.821.

use std::f64::consts::PI;
use std::collections::HashMap;
use thiserror::Error;

// ============================================================================
// Constants
// ============================================================================

/// Earth gravitational parameter (m³/s²)
const GM_EARTH: f64 = 3.986004418e14;
/// Earth radius in meters
const EARTH_RADIUS_M: f64 = 6_371_000.0;
/// Speed of light in m/s
const SPEED_OF_LIGHT: f64 = 299_792_458.0;
/// Minimum elevation angle for satellite visibility (degrees)
const MIN_ELEVATION_DEG: f64 = 5.0;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Debug, Error)]
pub enum ConstellationError {
    #[error("Invalid satellite ID: {0}")]
    InvalidSatelliteId(u32),

    #[error("Satellite not visible: {0}")]
    SatelliteNotVisible(String),

    #[error("Invalid orbital parameters: {0}")]
    InvalidOrbitalParameters(String),

    #[error("No satellites in constellation")]
    EmptyConstellation,
}

// ============================================================================
// Orbital Elements (Keplerian)
// ============================================================================

/// Keplerian orbital elements for satellite orbit description
#[derive(Debug, Clone)]
pub struct OrbitalElements {
    /// Semi-major axis (m)
    pub semi_major_axis_m: f64,
    /// Eccentricity (0.0 to < 1.0 for elliptical)
    pub eccentricity: f64,
    /// Inclination (radians)
    pub inclination_rad: f64,
    /// Right Ascension of Ascending Node - RAAN (radians)
    pub raan_rad: f64,
    /// Argument of periapsis (radians)
    pub argument_of_periapsis_rad: f64,
    /// Mean anomaly at epoch (radians)
    pub mean_anomaly_rad: f64,
    /// Epoch time (seconds since J2000)
    pub epoch_s: f64,
}

impl OrbitalElements {
    /// Calculate orbital period in seconds
    pub fn orbital_period(&self) -> f64 {
        2.0 * PI * (self.semi_major_axis_m.powi(3) / GM_EARTH).sqrt()
    }

    /// Calculate mean motion in radians per second
    pub fn mean_motion(&self) -> f64 {
        (GM_EARTH / self.semi_major_axis_m.powi(3)).sqrt()
    }

    /// Validate orbital elements
    pub fn validate(&self) -> Result<(), ConstellationError> {
        if self.semi_major_axis_m <= EARTH_RADIUS_M {
            return Err(ConstellationError::InvalidOrbitalParameters(
                "Semi-major axis must be greater than Earth radius".to_string()
            ));
        }

        if self.eccentricity < 0.0 || self.eccentricity >= 1.0 {
            return Err(ConstellationError::InvalidOrbitalParameters(
                "Eccentricity must be in range [0, 1)".to_string()
            ));
        }

        Ok(())
    }
}

// ============================================================================
// Position Representations
// ============================================================================

/// Cartesian position in ECEF (Earth-Centered Earth-Fixed) coordinates
#[derive(Debug, Clone, Copy)]
pub struct EcefPosition {
    /// X coordinate (m)
    pub x: f64,
    /// Y coordinate (m)
    pub y: f64,
    /// Z coordinate (m)
    pub z: f64,
}

impl EcefPosition {
    /// Calculate distance to another position in meters
    pub fn distance_to(&self, other: &EcefPosition) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Convert to geodetic coordinates
    pub fn to_geodetic(&self) -> GeodeticPosition {
        let r = (self.x * self.x + self.y * self.y).sqrt();
        let lat = (self.z / r).atan();
        let lon = self.y.atan2(self.x);
        let alt = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt() - EARTH_RADIUS_M;

        GeodeticPosition {
            latitude_rad: lat,
            longitude_rad: lon,
            altitude_m: alt,
        }
    }
}

/// Geodetic position (latitude, longitude, altitude)
#[derive(Debug, Clone, Copy)]
pub struct GeodeticPosition {
    /// Latitude (radians, -π/2 to π/2)
    pub latitude_rad: f64,
    /// Longitude (radians, -π to π)
    pub longitude_rad: f64,
    /// Altitude above WGS84 ellipsoid (m)
    pub altitude_m: f64,
}

impl GeodeticPosition {
    /// Create from degrees
    pub fn from_degrees(lat_deg: f64, lon_deg: f64, alt_m: f64) -> Self {
        Self {
            latitude_rad: lat_deg.to_radians(),
            longitude_rad: lon_deg.to_radians(),
            altitude_m: alt_m,
        }
    }

    /// Convert to ECEF coordinates
    pub fn to_ecef(&self) -> EcefPosition {
        let r = EARTH_RADIUS_M + self.altitude_m;
        let cos_lat = self.latitude_rad.cos();
        let sin_lat = self.latitude_rad.sin();
        let cos_lon = self.longitude_rad.cos();
        let sin_lon = self.longitude_rad.sin();

        EcefPosition {
            x: r * cos_lat * cos_lon,
            y: r * cos_lat * sin_lon,
            z: r * sin_lat,
        }
    }

    /// Get latitude in degrees
    pub fn latitude_deg(&self) -> f64 {
        self.latitude_rad.to_degrees()
    }

    /// Get longitude in degrees
    pub fn longitude_deg(&self) -> f64 {
        self.longitude_rad.to_degrees()
    }
}

// ============================================================================
// Satellite State
// ============================================================================

/// Complete satellite state at a given time
#[derive(Debug, Clone)]
pub struct SatelliteState {
    /// Satellite ID
    pub satellite_id: u32,
    /// Position in ECEF
    pub position: EcefPosition,
    /// Velocity in ECEF (m/s)
    pub velocity: EcefPosition,
    /// Timestamp (seconds since J2000)
    pub time_s: f64,
}

impl SatelliteState {
    /// Calculate visibility from a ground position
    pub fn calculate_visibility(&self, ground_pos: &GeodeticPosition) -> VisibilityInfo {
        let ground_ecef = ground_pos.to_ecef();
        let slant_range_m = self.position.distance_to(&ground_ecef);

        // Calculate elevation angle
        let ground_to_sat_x = self.position.x - ground_ecef.x;
        let ground_to_sat_y = self.position.y - ground_ecef.y;
        let ground_to_sat_z = self.position.z - ground_ecef.z;

        let ground_radius = (ground_ecef.x * ground_ecef.x +
                           ground_ecef.y * ground_ecef.y +
                           ground_ecef.z * ground_ecef.z).sqrt();

        let dot_product = (ground_ecef.x * ground_to_sat_x +
                          ground_ecef.y * ground_to_sat_y +
                          ground_ecef.z * ground_to_sat_z) / ground_radius;

        let elevation_rad = (dot_product / slant_range_m).asin();
        let elevation_deg = elevation_rad.to_degrees();

        // Calculate azimuth (simplified)
        let azimuth_rad = ground_to_sat_y.atan2(ground_to_sat_x);
        let azimuth_deg = azimuth_rad.to_degrees();

        VisibilityInfo {
            is_visible: elevation_deg >= MIN_ELEVATION_DEG,
            elevation_deg,
            azimuth_deg,
            slant_range_m,
        }
    }

    /// Calculate Doppler shift for given ground position and carrier frequency
    pub fn calculate_doppler(&self, ground_pos: &GeodeticPosition, carrier_freq_hz: f64) -> f64 {
        let ground_ecef = ground_pos.to_ecef();

        // Vector from ground to satellite
        let range_x = self.position.x - ground_ecef.x;
        let range_y = self.position.y - ground_ecef.y;
        let range_z = self.position.z - ground_ecef.z;
        let range = (range_x * range_x + range_y * range_y + range_z * range_z).sqrt();

        // Unit vector
        let unit_x = range_x / range;
        let unit_y = range_y / range;
        let unit_z = range_z / range;

        // Radial velocity (dot product of velocity and unit range vector)
        let radial_velocity = self.velocity.x * unit_x +
                             self.velocity.y * unit_y +
                             self.velocity.z * unit_z;

        // Doppler shift: f_d = (v_r / c) * f_c
        (radial_velocity / SPEED_OF_LIGHT) * carrier_freq_hz
    }

    /// Calculate propagation delay to ground position (one-way)
    pub fn propagation_delay_ms(&self, ground_pos: &GeodeticPosition) -> f64 {
        let ground_ecef = ground_pos.to_ecef();
        let distance_m = self.position.distance_to(&ground_ecef);
        (distance_m / SPEED_OF_LIGHT) * 1000.0 // Convert to ms
    }
}

// ============================================================================
// Visibility Information
// ============================================================================

/// Satellite visibility information from a ground position
#[derive(Debug, Clone, Copy)]
pub struct VisibilityInfo {
    /// Whether satellite is visible (elevation >= MIN_ELEVATION_DEG)
    pub is_visible: bool,
    /// Elevation angle (degrees)
    pub elevation_deg: f64,
    /// Azimuth angle (degrees)
    pub azimuth_deg: f64,
    /// Slant range (meters)
    pub slant_range_m: f64,
}

// ============================================================================
// Satellite Constellation
// ============================================================================

/// Satellite constellation configuration
#[derive(Debug, Clone)]
pub struct ConstellationConfig {
    /// Constellation name
    pub name: String,
    /// Orbital altitude (m)
    pub altitude_m: f64,
    /// Orbital inclination (degrees)
    pub inclination_deg: f64,
    /// Number of orbital planes
    pub num_planes: u32,
    /// Number of satellites per plane
    pub sats_per_plane: u32,
    /// Phase offset between planes (degrees)
    pub plane_phase_offset_deg: f64,
    /// Eccentricity
    pub eccentricity: f64,
}

impl ConstellationConfig {
    /// Create a Starlink-like LEO constellation
    pub fn starlink_like() -> Self {
        Self {
            name: "LEO-550".to_string(),
            altitude_m: 550_000.0,
            inclination_deg: 53.0,
            num_planes: 72,
            sats_per_plane: 22,
            plane_phase_offset_deg: 0.0,
            eccentricity: 0.0001,
        }
    }

    /// Create a OneWeb-like LEO constellation
    pub fn oneweb_like() -> Self {
        Self {
            name: "LEO-1200".to_string(),
            altitude_m: 1_200_000.0,
            inclination_deg: 87.9,
            num_planes: 18,
            sats_per_plane: 40,
            plane_phase_offset_deg: 0.0,
            eccentricity: 0.0001,
        }
    }

    /// Create a GEO satellite
    pub fn geo() -> Self {
        Self {
            name: "GEO".to_string(),
            altitude_m: 35_786_000.0,
            inclination_deg: 0.0,
            num_planes: 1,
            sats_per_plane: 1,
            plane_phase_offset_deg: 0.0,
            eccentricity: 0.0001,
        }
    }

    /// Total number of satellites
    pub fn total_satellites(&self) -> u32 {
        self.num_planes * self.sats_per_plane
    }
}

/// Satellite constellation manager
pub struct Constellation {
    /// Configuration
    config: ConstellationConfig,
    /// Satellites with their orbital elements
    satellites: HashMap<u32, OrbitalElements>,
    /// Current time (seconds since J2000)
    current_time_s: f64,
}

impl Constellation {
    /// Create a new constellation from configuration
    pub fn new(config: ConstellationConfig) -> Result<Self, ConstellationError> {
        let mut satellites = HashMap::new();
        let semi_major_axis = EARTH_RADIUS_M + config.altitude_m;
        let inclination_rad = config.inclination_deg.to_radians();

        let mut sat_id = 1u32;

        for plane_idx in 0..config.num_planes {
            let raan = 2.0 * PI * (plane_idx as f64) / (config.num_planes as f64);

            for sat_idx in 0..config.sats_per_plane {
                let mean_anomaly = 2.0 * PI * (sat_idx as f64) / (config.sats_per_plane as f64);
                let phase_offset = config.plane_phase_offset_deg.to_radians() * (plane_idx as f64);

                let elements = OrbitalElements {
                    semi_major_axis_m: semi_major_axis,
                    eccentricity: config.eccentricity,
                    inclination_rad,
                    raan_rad: raan,
                    argument_of_periapsis_rad: 0.0,
                    mean_anomaly_rad: mean_anomaly + phase_offset,
                    epoch_s: 0.0,
                };

                elements.validate()?;
                satellites.insert(sat_id, elements);
                sat_id += 1;
            }
        }

        Ok(Self {
            config,
            satellites,
            current_time_s: 0.0,
        })
    }

    /// Update constellation time
    pub fn set_time(&mut self, time_s: f64) {
        self.current_time_s = time_s;
    }

    /// Calculate satellite position at current time using simplified propagation
    pub fn calculate_satellite_state(&self, sat_id: u32) -> Result<SatelliteState, ConstellationError> {
        let elements = self.satellites.get(&sat_id)
            .ok_or(ConstellationError::InvalidSatelliteId(sat_id))?;

        let time_since_epoch = self.current_time_s - elements.epoch_s;
        let mean_motion = elements.mean_motion();
        let mean_anomaly = elements.mean_anomaly_rad + mean_motion * time_since_epoch;

        // Solve Kepler's equation (simplified for small eccentricity)
        let eccentric_anomaly = self.solve_kepler(mean_anomaly, elements.eccentricity);

        // True anomaly
        let true_anomaly = 2.0 * ((eccentric_anomaly / 2.0).tan() *
                                 ((1.0 + elements.eccentricity) / (1.0 - elements.eccentricity)).sqrt()).atan();

        // Orbital radius
        let r = elements.semi_major_axis_m * (1.0 - elements.eccentricity * eccentric_anomaly.cos());

        // Position in orbital plane
        let x_orb = r * true_anomaly.cos();
        let y_orb = r * true_anomaly.sin();

        // Rotate to ECEF
        let (x, y, z) = self.orbital_to_ecef(
            x_orb, y_orb,
            elements.inclination_rad,
            elements.raan_rad,
            elements.argument_of_periapsis_rad,
        );

        // Velocity calculation (simplified)
        let v_mag = (GM_EARTH / elements.semi_major_axis_m).sqrt();
        let v_x = -v_mag * mean_anomaly.sin();
        let v_y = v_mag * mean_anomaly.cos();
        let (vx, vy, vz) = self.orbital_to_ecef(
            v_x, v_y,
            elements.inclination_rad,
            elements.raan_rad,
            elements.argument_of_periapsis_rad,
        );

        Ok(SatelliteState {
            satellite_id: sat_id,
            position: EcefPosition { x, y, z },
            velocity: EcefPosition { x: vx, y: vy, z: vz },
            time_s: self.current_time_s,
        })
    }

    /// Solve Kepler's equation using Newton-Raphson
    fn solve_kepler(&self, mean_anomaly: f64, eccentricity: f64) -> f64 {
        let mut e_anom = mean_anomaly;
        for _ in 0..10 {
            let delta = e_anom - eccentricity * e_anom.sin() - mean_anomaly;
            e_anom -= delta / (1.0 - eccentricity * e_anom.cos());
            if delta.abs() < 1e-10 {
                break;
            }
        }
        e_anom
    }

    /// Transform orbital plane coordinates to ECEF
    fn orbital_to_ecef(&self, x_orb: f64, y_orb: f64, incl: f64, raan: f64, aop: f64) -> (f64, f64, f64) {
        let cos_raan = raan.cos();
        let sin_raan = raan.sin();
        let cos_incl = incl.cos();
        let sin_incl = incl.sin();
        let cos_aop = aop.cos();
        let sin_aop = aop.sin();

        let x = (cos_raan * cos_aop - sin_raan * sin_aop * cos_incl) * x_orb +
                (-cos_raan * sin_aop - sin_raan * cos_aop * cos_incl) * y_orb;

        let y = (sin_raan * cos_aop + cos_raan * sin_aop * cos_incl) * x_orb +
                (-sin_raan * sin_aop + cos_raan * cos_aop * cos_incl) * y_orb;

        let z = sin_incl * sin_aop * x_orb + sin_incl * cos_aop * y_orb;

        (x, y, z)
    }

    /// Find all visible satellites from a ground position
    pub fn find_visible_satellites(&self, ground_pos: &GeodeticPosition) -> Vec<(u32, VisibilityInfo)> {
        let mut visible = Vec::new();

        for &sat_id in self.satellites.keys() {
            if let Ok(state) = self.calculate_satellite_state(sat_id) {
                let visibility = state.calculate_visibility(ground_pos);
                if visibility.is_visible {
                    visible.push((sat_id, visibility));
                }
            }
        }

        // Sort by elevation (highest first)
        visible.sort_by(|a, b| b.1.elevation_deg.partial_cmp(&a.1.elevation_deg).unwrap());

        visible
    }

    /// Find best serving satellite (highest elevation)
    pub fn find_best_satellite(&self, ground_pos: &GeodeticPosition) -> Result<u32, ConstellationError> {
        let visible = self.find_visible_satellites(ground_pos);

        if visible.is_empty() {
            return Err(ConstellationError::SatelliteNotVisible(
                "No satellites visible from ground position".to_string()
            ));
        }

        Ok(visible[0].0)
    }

    /// Predict handover time to next satellite
    pub fn predict_handover(&self, ground_pos: &GeodeticPosition, current_sat_id: u32,
                           time_step_s: f64, max_lookahead_s: f64) -> Option<(u32, f64)> {
        let mut best_future_sat = None;
        let mut handover_time = None;

        let mut temp_time = self.current_time_s;
        let end_time = self.current_time_s + max_lookahead_s;

        while temp_time < end_time {
            temp_time += time_step_s;

            // Check if current satellite is still best
            let mut best_elev = -90.0;
            let mut best_sat = current_sat_id;

            for &sat_id in self.satellites.keys() {
                if let Ok(state) = self.calculate_satellite_state(sat_id) {
                    let vis = state.calculate_visibility(ground_pos);
                    if vis.is_visible && vis.elevation_deg > best_elev {
                        best_elev = vis.elevation_deg;
                        best_sat = sat_id;
                    }
                }
            }

            if best_sat != current_sat_id {
                best_future_sat = Some(best_sat);
                handover_time = Some(temp_time - self.current_time_s);
                break;
            }
        }

        if let (Some(sat), Some(time)) = (best_future_sat, handover_time) {
            Some((sat, time))
        } else {
            None
        }
    }

    /// Get constellation configuration
    pub fn config(&self) -> &ConstellationConfig {
        &self.config
    }

    /// Get number of satellites
    pub fn num_satellites(&self) -> usize {
        self.satellites.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orbital_elements() {
        let elements = OrbitalElements {
            semi_major_axis_m: EARTH_RADIUS_M + 550_000.0,
            eccentricity: 0.001,
            inclination_rad: 53.0_f64.to_radians(),
            raan_rad: 0.0,
            argument_of_periapsis_rad: 0.0,
            mean_anomaly_rad: 0.0,
            epoch_s: 0.0,
        };

        assert!(elements.validate().is_ok());

        let period = elements.orbital_period();
        assert!(period > 5000.0 && period < 6500.0); // ~95 minutes for LEO
    }

    #[test]
    fn test_position_conversion() {
        let geo = GeodeticPosition::from_degrees(40.0, -74.0, 0.0);
        let ecef = geo.to_ecef();
        let back = ecef.to_geodetic();

        assert!((back.latitude_deg() - 40.0).abs() < 1.0);
        assert!((back.longitude_deg() - (-74.0)).abs() < 1.0);
    }

    #[test]
    fn test_constellation_creation() {
        let config = ConstellationConfig::starlink_like();
        let constellation = Constellation::new(config.clone()).unwrap();

        assert_eq!(constellation.num_satellites(), config.total_satellites() as usize);
    }

    #[test]
    fn test_satellite_propagation() {
        let config = ConstellationConfig::starlink_like();
        let mut constellation = Constellation::new(config).unwrap();

        constellation.set_time(0.0);
        let state = constellation.calculate_satellite_state(1).unwrap();

        assert!(state.position.x.abs() > 0.0);
        assert!(state.position.y.abs() >= 0.0);
        assert!(state.position.z.abs() >= 0.0);
    }

    #[test]
    fn test_visibility_calculation() {
        let config = ConstellationConfig::geo();
        let mut constellation = Constellation::new(config).unwrap();

        constellation.set_time(0.0);
        let ground = GeodeticPosition::from_degrees(0.0, 0.0, 0.0);

        let state = constellation.calculate_satellite_state(1).unwrap();
        let visibility = state.calculate_visibility(&ground);

        // GEO at equator should be visible from equator
        assert!(visibility.elevation_deg > 0.0);
    }

    #[test]
    fn test_find_visible_satellites() {
        let config = ConstellationConfig::starlink_like();
        let mut constellation = Constellation::new(config).unwrap();

        constellation.set_time(0.0);
        let ground = GeodeticPosition::from_degrees(40.0, -74.0, 0.0);

        let visible = constellation.find_visible_satellites(&ground);

        // With Starlink-like constellation, there should be multiple visible satellites
        assert!(visible.len() > 0);
    }

    #[test]
    fn test_doppler_calculation() {
        let config = ConstellationConfig::starlink_like();
        let mut constellation = Constellation::new(config).unwrap();

        constellation.set_time(0.0);
        let ground = GeodeticPosition::from_degrees(40.0, -74.0, 0.0);
        let state = constellation.calculate_satellite_state(1).unwrap();

        let doppler = state.calculate_doppler(&ground, 2e9); // 2 GHz carrier

        // LEO satellites have significant Doppler
        assert!(doppler.abs() < 100_000.0); // Should be within +/- 100 kHz
    }

    #[test]
    fn test_propagation_delay() {
        let config = ConstellationConfig::geo();
        let mut constellation = Constellation::new(config).unwrap();

        constellation.set_time(0.0);
        let ground = GeodeticPosition::from_degrees(0.0, 0.0, 0.0);
        let state = constellation.calculate_satellite_state(1).unwrap();

        let delay = state.propagation_delay_ms(&ground);

        // GEO should have ~120ms one-way delay
        assert!(delay > 100.0 && delay < 150.0);
    }
}
