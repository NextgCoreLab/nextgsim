//! NTN (Non-Terrestrial Network) Timing Advance
//!
//! 6G extension: RRC configuration for NTN timing advance management.
//! Handles satellite-specific timing synchronization, ephemeris-based
//! timing advance calculation, and Doppler compensation at the RRC level.
//!
//! This module implements:
//! - `NtnTimingAdvance` - NTN timing advance configuration with ephemeris data
//! - Timing offset calculation and common TA management
//! - Satellite orbit type classification and position tracking

use thiserror::Error;

/// Errors that can occur during NTN timing procedures
#[derive(Debug, Error)]
pub enum NtnTimingError {
    /// Invalid NTN timing configuration
    #[error("Invalid NTN timing configuration: {0}")]
    InvalidConfig(String),

    /// Missing mandatory field
    #[error("Missing mandatory field: {0}")]
    MissingMandatoryField(String),

    /// Encoding/decoding error
    #[error("Codec error: {0}")]
    CodecError(String),

    /// Timing calculation error
    #[error("Timing calculation error: {0}")]
    CalculationError(String),
}

/// Satellite orbit type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SatelliteOrbitType {
    /// Low Earth Orbit (LEO) - altitude 300-2000 km
    Leo,
    /// Medium Earth Orbit (MEO) - altitude 2000-35786 km
    Meo,
    /// Geostationary Earth Orbit (GEO) - altitude ~35786 km
    Geo,
    /// Highly Elliptical Orbit (HEO)
    Heo,
    /// High Altitude Platform Station (HAPS) - altitude 20-50 km
    Haps,
}

impl SatelliteOrbitType {
    /// Get the typical altitude range in kilometers
    pub fn altitude_range_km(&self) -> (f64, f64) {
        match self {
            SatelliteOrbitType::Leo => (300.0, 2000.0),
            SatelliteOrbitType::Meo => (2000.0, 35786.0),
            SatelliteOrbitType::Geo => (35786.0, 35786.0),
            SatelliteOrbitType::Heo => (200.0, 50000.0),
            SatelliteOrbitType::Haps => (20.0, 50.0),
        }
    }

    /// Get the typical one-way propagation delay range in milliseconds
    pub fn typical_delay_range_ms(&self) -> (f64, f64) {
        match self {
            SatelliteOrbitType::Leo => (1.0, 13.0),
            SatelliteOrbitType::Meo => (13.0, 120.0),
            SatelliteOrbitType::Geo => (120.0, 140.0),
            SatelliteOrbitType::Heo => (1.0, 170.0),
            SatelliteOrbitType::Haps => (0.07, 0.17),
        }
    }
}

/// Ephemeris data source type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EphemerisSource {
    /// Broadcast by gNB in SIB
    BroadcastSib,
    /// Provided via dedicated RRC signalling
    DedicatedRrc,
    /// Pre-provisioned in UE (e.g., from almanac)
    PreProvisioned,
    /// Obtained from GNSS
    Gnss,
}

/// Keplerian orbital elements
#[derive(Debug, Clone)]
pub struct KeplerianElements {
    /// Semi-major axis in meters
    pub semi_major_axis_m: f64,
    /// Eccentricity (0.0 to < 1.0 for elliptical orbits)
    pub eccentricity: f64,
    /// Inclination in degrees (0-180)
    pub inclination_deg: f64,
    /// Right ascension of ascending node (RAAN) in degrees (0-360)
    pub raan_deg: f64,
    /// Argument of periapsis in degrees (0-360)
    pub argument_of_periapsis_deg: f64,
    /// Mean anomaly at epoch in degrees (0-360)
    pub mean_anomaly_deg: f64,
}

/// Ephemeris data for satellite position prediction
#[derive(Debug, Clone)]
pub struct NtnEphemerisData {
    /// Satellite ID
    pub satellite_id: u32,
    /// Orbital elements
    pub orbital_elements: KeplerianElements,
    /// Epoch time in milliseconds since Unix epoch
    pub epoch_time_ms: u64,
    /// Data source
    pub source: EphemerisSource,
    /// Validity duration in seconds
    pub validity_duration_s: u32,
    /// Satellite orbit type
    pub orbit_type: SatelliteOrbitType,
}

/// Satellite position in geodetic coordinates
#[derive(Debug, Clone)]
pub struct SatellitePositionGeodetic {
    /// Latitude in degrees (-90 to 90)
    pub latitude_deg: f64,
    /// Longitude in degrees (-180 to 180)
    pub longitude_deg: f64,
    /// Altitude above WGS84 ellipsoid in meters
    pub altitude_m: f64,
}

/// NTN timing advance configuration
///
/// Contains timing advance parameters for NTN operation, including
/// common TA, UE-specific differential TA, and ephemeris data for
/// autonomous TA calculation.
#[derive(Debug, Clone)]
pub struct NtnTimingAdvance {
    /// Configuration ID
    pub config_id: u16,
    /// Satellite orbit type
    pub orbit_type: SatelliteOrbitType,
    /// Satellite ID
    pub satellite_id: u32,
    /// Common timing advance in microseconds (broadcast to all UEs in the cell)
    pub common_ta_us: u64,
    /// UE-specific timing advance offset in microseconds (signed)
    pub ue_specific_ta_offset_us: i32,
    /// Timing advance update periodicity in milliseconds
    pub ta_update_periodicity_ms: u32,
    /// Ephemeris data for autonomous TA calculation
    pub ephemeris: Option<NtnEphemerisData>,
    /// Current satellite position (if available)
    pub satellite_position: Option<SatellitePositionGeodetic>,
    /// K-offset for HARQ timing (as per 3GPP TS 38.213 for NTN)
    pub k_offset: u16,
    /// Maximum Doppler shift in Hz
    pub max_doppler_shift_hz: f64,
    /// Pre-compensation of Doppler at gNB (true = DL Doppler pre-compensated)
    pub dl_doppler_precompensated: bool,
    /// UE should apply Doppler pre-compensation for UL
    pub ul_doppler_compensation: bool,
    /// Timing advance validity timer in seconds
    pub ta_validity_timer_s: u32,
    /// Whether UE should autonomously calculate TA from ephemeris
    pub autonomous_ta_enabled: bool,
    /// Guard time in microseconds (additional margin for TA uncertainty)
    pub guard_time_us: u32,
}

/// NTN timing advance report from UE
#[derive(Debug, Clone)]
pub struct NtnTaReport {
    /// Configuration ID
    pub config_id: u16,
    /// Measured one-way propagation delay in microseconds
    pub measured_propagation_delay_us: u64,
    /// Estimated Doppler shift in Hz
    pub estimated_doppler_hz: f64,
    /// UE position (if available from GNSS)
    pub ue_position: Option<SatellitePositionGeodetic>,
    /// Timestamp of measurement in ms since epoch
    pub measurement_time_ms: u64,
    /// TA calculation method used
    pub ta_method: TaCalculationMethod,
}

/// Timing advance calculation method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaCalculationMethod {
    /// TA from GNSS-based position and ephemeris
    GnssEphemeris,
    /// TA from RACH timing advance
    RachBased,
    /// TA from network command
    NetworkCommand,
    /// TA from pre-provisioned almanac
    AlmanacBased,
    /// TA from previous measurement extrapolation
    Extrapolated,
}

/// NTN HARQ configuration
#[derive(Debug, Clone)]
pub struct NtnHarqConfig {
    /// K-offset value for HARQ feedback timing
    pub k_offset: u16,
    /// Enable HARQ-ACK disabling for NTN (to handle long RTT)
    pub harq_ack_disabled: bool,
    /// Number of HARQ processes (increased for NTN)
    pub num_harq_processes: u8,
    /// HARQ RTT timer in slots
    pub harq_rtt_timer_slots: u16,
}

impl NtnTimingAdvance {
    /// Validate the NTN timing advance configuration
    pub fn validate(&self) -> Result<(), NtnTimingError> {
        if self.ta_update_periodicity_ms == 0 {
            return Err(NtnTimingError::InvalidConfig(
                "TA update periodicity must be > 0".to_string(),
            ));
        }

        if self.ta_validity_timer_s == 0 {
            return Err(NtnTimingError::InvalidConfig(
                "TA validity timer must be > 0".to_string(),
            ));
        }

        if self.max_doppler_shift_hz < 0.0 {
            return Err(NtnTimingError::InvalidConfig(
                "Maximum Doppler shift must be >= 0".to_string(),
            ));
        }

        // Validate ephemeris if present
        if let Some(ref eph) = self.ephemeris {
            eph.validate()?;
        }

        // Validate satellite position if present
        if let Some(ref pos) = self.satellite_position {
            if pos.latitude_deg < -90.0 || pos.latitude_deg > 90.0 {
                return Err(NtnTimingError::InvalidConfig(
                    "Satellite latitude must be in range [-90, 90]".to_string(),
                ));
            }
            if pos.longitude_deg < -180.0 || pos.longitude_deg > 180.0 {
                return Err(NtnTimingError::InvalidConfig(
                    "Satellite longitude must be in range [-180, 180]".to_string(),
                ));
            }
            if pos.altitude_m < 0.0 {
                return Err(NtnTimingError::InvalidConfig(
                    "Satellite altitude must be >= 0".to_string(),
                ));
            }
        }

        // Validate HARQ process count for NTN
        if self.k_offset == 0 && !matches!(self.orbit_type, SatelliteOrbitType::Haps) {
            return Err(NtnTimingError::InvalidConfig(
                "K-offset should be > 0 for non-HAPS satellite orbits".to_string(),
            ));
        }

        // Validate autonomous TA requires ephemeris
        if self.autonomous_ta_enabled && self.ephemeris.is_none() {
            return Err(NtnTimingError::InvalidConfig(
                "Autonomous TA calculation requires ephemeris data".to_string(),
            ));
        }

        Ok(())
    }

    /// Calculate the total timing advance in microseconds (common + UE-specific)
    pub fn total_ta_us(&self) -> i64 {
        self.common_ta_us as i64 + self.ue_specific_ta_offset_us as i64
    }

    /// Calculate the round-trip delay in microseconds
    pub fn round_trip_delay_us(&self) -> u128 {
        (self.common_ta_us as u128 + self.ue_specific_ta_offset_us.unsigned_abs() as u128) * 2
    }

    /// Estimate the one-way propagation delay in microseconds
    pub fn estimated_one_way_delay_us(&self) -> u64 {
        self.total_ta_us().unsigned_abs() / 2
    }
}

impl NtnEphemerisData {
    /// Validate the ephemeris data
    pub fn validate(&self) -> Result<(), NtnTimingError> {
        let oe = &self.orbital_elements;

        if oe.semi_major_axis_m <= 0.0 {
            return Err(NtnTimingError::InvalidConfig(
                "Semi-major axis must be > 0".to_string(),
            ));
        }

        if oe.eccentricity < 0.0 || oe.eccentricity >= 1.0 {
            return Err(NtnTimingError::InvalidConfig(
                "Eccentricity must be in range [0, 1) for bound orbits".to_string(),
            ));
        }

        if oe.inclination_deg < 0.0 || oe.inclination_deg > 180.0 {
            return Err(NtnTimingError::InvalidConfig(
                "Inclination must be in range [0, 180] degrees".to_string(),
            ));
        }

        if self.validity_duration_s == 0 {
            return Err(NtnTimingError::InvalidConfig(
                "Ephemeris validity duration must be > 0".to_string(),
            ));
        }

        Ok(())
    }

    /// Calculate the orbital period in seconds
    pub fn orbital_period_s(&self) -> f64 {
        // T = 2*pi * sqrt(a^3 / GM)
        // GM (Earth) = 3.986004418e14 m^3/s^2
        const GM_EARTH: f64 = 3.986004418e14;
        let a = self.orbital_elements.semi_major_axis_m;
        2.0 * std::f64::consts::PI * (a * a * a / GM_EARTH).sqrt()
    }

    /// Calculate the altitude at periapsis in meters
    pub fn periapsis_altitude_m(&self) -> f64 {
        const EARTH_RADIUS_M: f64 = 6_371_000.0;
        let a = self.orbital_elements.semi_major_axis_m;
        let e = self.orbital_elements.eccentricity;
        a * (1.0 - e) - EARTH_RADIUS_M
    }

    /// Calculate the altitude at apoapsis in meters
    pub fn apoapsis_altitude_m(&self) -> f64 {
        const EARTH_RADIUS_M: f64 = 6_371_000.0;
        let a = self.orbital_elements.semi_major_axis_m;
        let e = self.orbital_elements.eccentricity;
        a * (1.0 + e) - EARTH_RADIUS_M
    }
}

impl NtnHarqConfig {
    /// Validate the HARQ configuration
    pub fn validate(&self) -> Result<(), NtnTimingError> {
        if self.num_harq_processes == 0 || self.num_harq_processes > 32 {
            return Err(NtnTimingError::InvalidConfig(
                "Number of HARQ processes must be 1-32".to_string(),
            ));
        }
        Ok(())
    }
}

/// Encode NTN timing advance to bytes (simplified serialization)
pub fn encode_ntn_timing_advance(
    ta: &NtnTimingAdvance,
) -> Result<Vec<u8>, NtnTimingError> {
    ta.validate()?;
    let mut bytes = Vec::with_capacity(48);

    // config_id (2 bytes)
    bytes.extend_from_slice(&ta.config_id.to_be_bytes());
    // orbit_type (1 byte)
    bytes.push(match ta.orbit_type {
        SatelliteOrbitType::Leo => 0,
        SatelliteOrbitType::Meo => 1,
        SatelliteOrbitType::Geo => 2,
        SatelliteOrbitType::Heo => 3,
        SatelliteOrbitType::Haps => 4,
    });
    // satellite_id (4 bytes)
    bytes.extend_from_slice(&ta.satellite_id.to_be_bytes());
    // common_ta_us (8 bytes)
    bytes.extend_from_slice(&ta.common_ta_us.to_be_bytes());
    // ue_specific_ta_offset_us (4 bytes, signed)
    bytes.extend_from_slice(&ta.ue_specific_ta_offset_us.to_be_bytes());
    // k_offset (2 bytes)
    bytes.extend_from_slice(&ta.k_offset.to_be_bytes());
    // ta_update_periodicity_ms (4 bytes)
    bytes.extend_from_slice(&ta.ta_update_periodicity_ms.to_be_bytes());
    // flags (1 byte): bit 0 = dl_doppler_precompensated, bit 1 = ul_doppler_compensation,
    //                 bit 2 = autonomous_ta_enabled
    let mut flags: u8 = 0;
    if ta.dl_doppler_precompensated {
        flags |= 0x01;
    }
    if ta.ul_doppler_compensation {
        flags |= 0x02;
    }
    if ta.autonomous_ta_enabled {
        flags |= 0x04;
    }
    bytes.push(flags);

    Ok(bytes)
}

/// Decode NTN timing advance header from bytes (simplified deserialization)
pub fn decode_ntn_timing_advance_header(
    bytes: &[u8],
) -> Result<(u16, SatelliteOrbitType, u32, u64), NtnTimingError> {
    if bytes.len() < 15 {
        return Err(NtnTimingError::CodecError(
            "Insufficient bytes for NTN timing advance header".to_string(),
        ));
    }

    let config_id = u16::from_be_bytes(bytes[0..2].try_into().unwrap());
    let orbit_type = match bytes[2] {
        0 => SatelliteOrbitType::Leo,
        1 => SatelliteOrbitType::Meo,
        2 => SatelliteOrbitType::Geo,
        3 => SatelliteOrbitType::Heo,
        4 => SatelliteOrbitType::Haps,
        _ => return Err(NtnTimingError::CodecError("Unknown orbit type".to_string())),
    };
    let satellite_id = u32::from_be_bytes(bytes[3..7].try_into().unwrap());
    let common_ta_us = u64::from_be_bytes(bytes[7..15].try_into().unwrap());

    Ok((config_id, orbit_type, satellite_id, common_ta_us))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_ephemeris() -> NtnEphemerisData {
        NtnEphemerisData {
            satellite_id: 12345,
            orbital_elements: KeplerianElements {
                semi_major_axis_m: 6_921_000.0, // ~550 km altitude
                eccentricity: 0.001,
                inclination_deg: 53.0,
                raan_deg: 120.0,
                argument_of_periapsis_deg: 90.0,
                mean_anomaly_deg: 45.0,
            },
            epoch_time_ms: 1700000000000,
            source: EphemerisSource::DedicatedRrc,
            validity_duration_s: 3600,
            orbit_type: SatelliteOrbitType::Leo,
        }
    }

    fn create_test_ta() -> NtnTimingAdvance {
        NtnTimingAdvance {
            config_id: 1,
            orbit_type: SatelliteOrbitType::Leo,
            satellite_id: 12345,
            common_ta_us: 10000,
            ue_specific_ta_offset_us: -200,
            ta_update_periodicity_ms: 1000,
            ephemeris: Some(create_test_ephemeris()),
            satellite_position: Some(SatellitePositionGeodetic {
                latitude_deg: 40.0,
                longitude_deg: -74.0,
                altitude_m: 550_000.0,
            }),
            k_offset: 16,
            max_doppler_shift_hz: 25000.0,
            dl_doppler_precompensated: true,
            ul_doppler_compensation: true,
            ta_validity_timer_s: 30,
            autonomous_ta_enabled: true,
            guard_time_us: 50,
        }
    }

    #[test]
    fn test_ntn_ta_validate() {
        let ta = create_test_ta();
        assert!(ta.validate().is_ok());
    }

    #[test]
    fn test_ntn_ta_invalid_update_periodicity() {
        let mut ta = create_test_ta();
        ta.ta_update_periodicity_ms = 0;
        assert!(ta.validate().is_err());
    }

    #[test]
    fn test_ntn_ta_invalid_validity_timer() {
        let mut ta = create_test_ta();
        ta.ta_validity_timer_s = 0;
        assert!(ta.validate().is_err());
    }

    #[test]
    fn test_ntn_ta_invalid_doppler() {
        let mut ta = create_test_ta();
        ta.max_doppler_shift_hz = -1.0;
        assert!(ta.validate().is_err());
    }

    #[test]
    fn test_ntn_ta_autonomous_without_ephemeris() {
        let mut ta = create_test_ta();
        ta.ephemeris = None;
        ta.autonomous_ta_enabled = true;
        assert!(ta.validate().is_err());
    }

    #[test]
    fn test_ntn_ta_autonomous_disabled_without_ephemeris() {
        let mut ta = create_test_ta();
        ta.ephemeris = None;
        ta.autonomous_ta_enabled = false;
        assert!(ta.validate().is_ok());
    }

    #[test]
    fn test_ntn_ta_k_offset_zero_non_haps() {
        let mut ta = create_test_ta();
        ta.k_offset = 0;
        assert!(ta.validate().is_err());
    }

    #[test]
    fn test_ntn_ta_k_offset_zero_haps() {
        let mut ta = create_test_ta();
        ta.orbit_type = SatelliteOrbitType::Haps;
        ta.k_offset = 0;
        ta.ephemeris.as_mut().unwrap().orbit_type = SatelliteOrbitType::Haps;
        assert!(ta.validate().is_ok());
    }

    #[test]
    fn test_ntn_ta_invalid_satellite_position() {
        let mut ta = create_test_ta();
        ta.satellite_position = Some(SatellitePositionGeodetic {
            latitude_deg: 91.0,
            longitude_deg: 0.0,
            altitude_m: 550_000.0,
        });
        assert!(ta.validate().is_err());
    }

    #[test]
    fn test_ntn_ta_total_ta() {
        let ta = create_test_ta();
        assert_eq!(ta.total_ta_us(), 9800); // 10000 - 200
    }

    #[test]
    fn test_ntn_ta_estimated_delay() {
        let ta = create_test_ta();
        // total_ta = 9800, one-way = 9800/2 = 4900
        assert_eq!(ta.estimated_one_way_delay_us(), 4900);
    }

    #[test]
    fn test_ephemeris_validate() {
        let eph = create_test_ephemeris();
        assert!(eph.validate().is_ok());
    }

    #[test]
    fn test_ephemeris_invalid_semi_major_axis() {
        let mut eph = create_test_ephemeris();
        eph.orbital_elements.semi_major_axis_m = 0.0;
        assert!(eph.validate().is_err());
    }

    #[test]
    fn test_ephemeris_invalid_eccentricity() {
        let mut eph = create_test_ephemeris();
        eph.orbital_elements.eccentricity = 1.0;
        assert!(eph.validate().is_err());

        eph.orbital_elements.eccentricity = -0.1;
        assert!(eph.validate().is_err());
    }

    #[test]
    fn test_ephemeris_invalid_inclination() {
        let mut eph = create_test_ephemeris();
        eph.orbital_elements.inclination_deg = 181.0;
        assert!(eph.validate().is_err());
    }

    #[test]
    fn test_ephemeris_invalid_validity() {
        let mut eph = create_test_ephemeris();
        eph.validity_duration_s = 0;
        assert!(eph.validate().is_err());
    }

    #[test]
    fn test_ephemeris_orbital_period() {
        let eph = create_test_ephemeris();
        let period = eph.orbital_period_s();
        // LEO at ~550km altitude: period ~95 minutes = ~5700 seconds
        assert!(period > 5000.0 && period < 6500.0, "Period was {period}");
    }

    #[test]
    fn test_ephemeris_periapsis_altitude() {
        let eph = create_test_ephemeris();
        let alt = eph.periapsis_altitude_m();
        // a = 6_921_000, e = 0.001, Earth_R = 6_371_000
        // periapsis = a * (1-e) - R = 6_921_000 * 0.999 - 6_371_000 ≈ 543_079
        assert!(alt > 500_000.0 && alt < 600_000.0, "Periapsis altitude was {alt}");
    }

    #[test]
    fn test_ephemeris_apoapsis_altitude() {
        let eph = create_test_ephemeris();
        let alt = eph.apoapsis_altitude_m();
        assert!(alt > 540_000.0 && alt < 560_000.0, "Apoapsis altitude was {alt}");
    }

    #[test]
    fn test_orbit_type_altitude_ranges() {
        let (min, max) = SatelliteOrbitType::Leo.altitude_range_km();
        assert_eq!(min, 300.0);
        assert_eq!(max, 2000.0);

        let (min, max) = SatelliteOrbitType::Geo.altitude_range_km();
        assert_eq!(min, 35786.0);
        assert_eq!(max, 35786.0);
    }

    #[test]
    fn test_orbit_type_delay_ranges() {
        let (min, max) = SatelliteOrbitType::Leo.typical_delay_range_ms();
        assert_eq!(min, 1.0);
        assert_eq!(max, 13.0);

        let (min, max) = SatelliteOrbitType::Geo.typical_delay_range_ms();
        assert_eq!(min, 120.0);
        assert_eq!(max, 140.0);
    }

    #[test]
    fn test_ntn_harq_config_validate() {
        let harq = NtnHarqConfig {
            k_offset: 16,
            harq_ack_disabled: false,
            num_harq_processes: 16,
            harq_rtt_timer_slots: 100,
        };
        assert!(harq.validate().is_ok());
    }

    #[test]
    fn test_ntn_harq_config_invalid_processes() {
        let harq = NtnHarqConfig {
            k_offset: 16,
            harq_ack_disabled: false,
            num_harq_processes: 0,
            harq_rtt_timer_slots: 100,
        };
        assert!(harq.validate().is_err());

        let harq = NtnHarqConfig {
            k_offset: 16,
            harq_ack_disabled: false,
            num_harq_processes: 33,
            harq_rtt_timer_slots: 100,
        };
        assert!(harq.validate().is_err());
    }

    #[test]
    fn test_encode_decode_ntn_ta() {
        let ta = create_test_ta();
        let encoded = encode_ntn_timing_advance(&ta).expect("Failed to encode");
        assert!(!encoded.is_empty());

        let (config_id, orbit_type, satellite_id, common_ta_us) =
            decode_ntn_timing_advance_header(&encoded).expect("Failed to decode");
        assert_eq!(config_id, 1);
        assert_eq!(orbit_type, SatelliteOrbitType::Leo);
        assert_eq!(satellite_id, 12345);
        assert_eq!(common_ta_us, 10000);
    }

    #[test]
    fn test_all_orbit_types() {
        let types = [
            SatelliteOrbitType::Leo,
            SatelliteOrbitType::Meo,
            SatelliteOrbitType::Geo,
            SatelliteOrbitType::Heo,
            SatelliteOrbitType::Haps,
        ];

        for orbit_type in types {
            let mut ta = create_test_ta();
            ta.orbit_type = orbit_type;
            if orbit_type == SatelliteOrbitType::Haps {
                ta.k_offset = 0; // HAPS can have k_offset = 0
            }
            let encoded = encode_ntn_timing_advance(&ta).expect("Failed to encode");
            let (_, decoded_type, _, _) =
                decode_ntn_timing_advance_header(&encoded).expect("Failed to decode");
            assert_eq!(decoded_type, orbit_type);
        }
    }

    #[test]
    fn test_ta_report() {
        let report = NtnTaReport {
            config_id: 1,
            measured_propagation_delay_us: 5000,
            estimated_doppler_hz: 15000.0,
            ue_position: Some(SatellitePositionGeodetic {
                latitude_deg: 40.7128,
                longitude_deg: -74.0060,
                altitude_m: 10.0,
            }),
            measurement_time_ms: 1700000000000,
            ta_method: TaCalculationMethod::GnssEphemeris,
        };
        assert_eq!(report.ta_method, TaCalculationMethod::GnssEphemeris);
    }

    #[test]
    fn test_all_ephemeris_sources() {
        let sources = [
            EphemerisSource::BroadcastSib,
            EphemerisSource::DedicatedRrc,
            EphemerisSource::PreProvisioned,
            EphemerisSource::Gnss,
        ];
        for source in sources {
            let mut eph = create_test_ephemeris();
            eph.source = source;
            assert!(eph.validate().is_ok());
        }
    }

    #[test]
    fn test_geo_satellite_ta() {
        let ta = NtnTimingAdvance {
            config_id: 2,
            orbit_type: SatelliteOrbitType::Geo,
            satellite_id: 99999,
            common_ta_us: 270_000, // ~270 ms for GEO
            ue_specific_ta_offset_us: 500,
            ta_update_periodicity_ms: 60000,
            ephemeris: Some(NtnEphemerisData {
                satellite_id: 99999,
                orbital_elements: KeplerianElements {
                    semi_major_axis_m: 42_164_000.0,
                    eccentricity: 0.0001,
                    inclination_deg: 0.1,
                    raan_deg: 0.0,
                    argument_of_periapsis_deg: 0.0,
                    mean_anomaly_deg: 0.0,
                },
                epoch_time_ms: 1700000000000,
                source: EphemerisSource::BroadcastSib,
                validity_duration_s: 86400,
                orbit_type: SatelliteOrbitType::Geo,
            }),
            satellite_position: None,
            k_offset: 64,
            max_doppler_shift_hz: 100.0,
            dl_doppler_precompensated: true,
            ul_doppler_compensation: false,
            ta_validity_timer_s: 3600,
            autonomous_ta_enabled: true,
            guard_time_us: 200,
        };
        assert!(ta.validate().is_ok());
        assert_eq!(ta.total_ta_us(), 270_500);
    }
}
