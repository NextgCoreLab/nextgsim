//! NTN (Non-Terrestrial Network) Support IEs
//!
//! Defines NTN-specific Information Elements for use as extension IEs
//! in existing NGAP procedures, supporting satellite-based 5G/6G networks.
//!
//! This module implements:
//! - `NtnTimingInfo` - Satellite timing information
//! - `NtnCellInfo` - NTN cell-specific information

use thiserror::Error;

/// Errors that can occur with NTN support IEs
#[derive(Debug, Error)]
pub enum NtnSupportError {
    /// Invalid NTN configuration
    #[error("Invalid NTN configuration: {0}")]
    InvalidConfig(String),

    /// Missing mandatory field
    #[error("Missing mandatory field: {0}")]
    MissingMandatoryField(String),

    /// Encoding/decoding error
    #[error("Codec error: {0}")]
    CodecError(String),
}

/// NTN satellite type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SatelliteType {
    /// Low Earth Orbit (LEO) - altitude 300-2000 km
    Leo,
    /// Medium Earth Orbit (MEO) - altitude 2000-35786 km
    Meo,
    /// Geostationary Earth Orbit (GEO) - altitude ~35786 km
    Geo,
    /// High Altitude Platform Station (HAPS)
    Haps,
}

/// Ephemeris data for satellite position
#[derive(Debug, Clone)]
pub struct EphemerisData {
    /// Semi-major axis in meters
    pub semi_major_axis_m: f64,
    /// Eccentricity (0.0 to 1.0)
    pub eccentricity: f64,
    /// Inclination in degrees
    pub inclination_deg: f64,
    /// Right ascension of ascending node in degrees
    pub raan_deg: f64,
    /// Argument of periapsis in degrees
    pub argument_of_periapsis_deg: f64,
    /// Mean anomaly in degrees
    pub mean_anomaly_deg: f64,
    /// Epoch timestamp in ms since epoch
    pub epoch_ms: u64,
}

/// Satellite position in ECEF (Earth-Centered, Earth-Fixed) coordinates
#[derive(Debug, Clone)]
pub struct SatellitePositionEcef {
    /// X coordinate in meters
    pub x_m: f64,
    /// Y coordinate in meters
    pub y_m: f64,
    /// Z coordinate in meters
    pub z_m: f64,
}

/// Satellite velocity in ECEF coordinates
#[derive(Debug, Clone)]
pub struct SatelliteVelocityEcef {
    /// X velocity in m/s
    pub vx_ms: f64,
    /// Y velocity in m/s
    pub vy_ms: f64,
    /// Z velocity in m/s
    pub vz_ms: f64,
}

/// NTN timing information
///
/// Contains satellite-specific timing data needed for proper timing
/// advance calculation and synchronization in NTN scenarios.
#[derive(Debug, Clone)]
pub struct NtnTimingInfo {
    /// Satellite type
    pub satellite_type: SatelliteType,
    /// Satellite ID
    pub satellite_id: u32,
    /// Ephemeris data (orbital elements)
    pub ephemeris: Option<EphemerisData>,
    /// Current satellite position (ECEF)
    pub position: Option<SatellitePositionEcef>,
    /// Current satellite velocity (ECEF)
    pub velocity: Option<SatelliteVelocityEcef>,
    /// One-way propagation delay in microseconds (service link)
    pub propagation_delay_us: u64,
    /// Common timing advance in microseconds
    pub common_ta_us: u64,
    /// Differential timing advance in microseconds (UE-specific offset)
    pub differential_ta_us: Option<i32>,
    /// Validity duration of timing info in milliseconds
    pub validity_duration_ms: u32,
    /// Reference timestamp in ms since epoch
    pub reference_time_ms: u64,
    /// Doppler shift in Hz (carrier frequency shift due to satellite motion)
    pub doppler_shift_hz: Option<f64>,
    /// K-offset for scheduling timing (as per 3GPP TS 38.213)
    pub k_offset: Option<u16>,
}

/// NTN cell-specific information
///
/// Contains cell-level information specific to NTN operation,
/// used as extension IEs in cell-related NGAP procedures.
#[derive(Debug, Clone)]
pub struct NtnCellInfo {
    /// NR Cell Identity (36 bits)
    pub nr_cell_identity: u64,
    /// PLMN Identity (3 bytes)
    pub plmn_identity: [u8; 3],
    /// Satellite type serving this cell
    pub satellite_type: SatelliteType,
    /// Satellite ID
    pub satellite_id: u32,
    /// Cell center latitude in degrees (-90 to 90)
    pub cell_center_latitude_deg: f64,
    /// Cell center longitude in degrees (-180 to 180)
    pub cell_center_longitude_deg: f64,
    /// Cell radius in km (earth-fixed or earth-moving cell)
    pub cell_radius_km: f64,
    /// Whether the cell footprint is earth-fixed
    pub is_earth_fixed: bool,
    /// Tracking area code (3 bytes)
    pub tac: [u8; 3],
    /// Feeder link delay in microseconds
    pub feeder_link_delay_us: Option<u64>,
    /// Service link delay in microseconds
    pub service_link_delay_us: Option<u64>,
    /// Maximum Doppler shift in Hz for this cell
    pub max_doppler_shift_hz: Option<f64>,
    /// Estimated handover time to next satellite (ms)
    pub estimated_handover_time_ms: Option<u64>,
}

impl NtnTimingInfo {
    /// Validate the NTN timing information
    pub fn validate(&self) -> Result<(), NtnSupportError> {
        if self.validity_duration_ms == 0 {
            return Err(NtnSupportError::InvalidConfig(
                "Validity duration must be > 0".to_string(),
            ));
        }
        if let Some(ref ephemeris) = self.ephemeris {
            if ephemeris.eccentricity < 0.0 || ephemeris.eccentricity >= 1.0 {
                return Err(NtnSupportError::InvalidConfig(
                    "Eccentricity must be in range [0, 1)".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// Calculate the total timing advance (common + differential) in microseconds
    pub fn total_ta_us(&self) -> i64 {
        self.common_ta_us as i64 + self.differential_ta_us.unwrap_or(0) as i64
    }

    /// Calculate round-trip propagation delay in microseconds
    pub fn round_trip_delay_us(&self) -> u128 {
        self.propagation_delay_us as u128 * 2
    }
}

impl NtnCellInfo {
    /// Validate the NTN cell information
    pub fn validate(&self) -> Result<(), NtnSupportError> {
        if self.cell_center_latitude_deg < -90.0 || self.cell_center_latitude_deg > 90.0 {
            return Err(NtnSupportError::InvalidConfig(
                "Latitude must be in range [-90, 90]".to_string(),
            ));
        }
        if self.cell_center_longitude_deg < -180.0 || self.cell_center_longitude_deg > 180.0 {
            return Err(NtnSupportError::InvalidConfig(
                "Longitude must be in range [-180, 180]".to_string(),
            ));
        }
        if self.cell_radius_km <= 0.0 {
            return Err(NtnSupportError::InvalidConfig(
                "Cell radius must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Encode NTN timing info to bytes (simplified serialization)
pub fn encode_ntn_timing_info(info: &NtnTimingInfo) -> Result<Vec<u8>, NtnSupportError> {
    info.validate()?;
    let mut bytes = Vec::with_capacity(32);
    // satellite_type (1 byte)
    bytes.push(match info.satellite_type {
        SatelliteType::Leo => 0,
        SatelliteType::Meo => 1,
        SatelliteType::Geo => 2,
        SatelliteType::Haps => 3,
    });
    // satellite_id (4 bytes)
    bytes.extend_from_slice(&info.satellite_id.to_be_bytes());
    // propagation_delay_us (8 bytes)
    bytes.extend_from_slice(&info.propagation_delay_us.to_be_bytes());
    // common_ta_us (8 bytes)
    bytes.extend_from_slice(&info.common_ta_us.to_be_bytes());
    // validity_duration_ms (4 bytes)
    bytes.extend_from_slice(&info.validity_duration_ms.to_be_bytes());
    // reference_time_ms (8 bytes)
    bytes.extend_from_slice(&info.reference_time_ms.to_be_bytes());
    Ok(bytes)
}

/// Decode NTN timing info from bytes (simplified deserialization)
pub fn decode_ntn_timing_info(bytes: &[u8]) -> Result<NtnTimingInfo, NtnSupportError> {
    if bytes.len() < 33 {
        return Err(NtnSupportError::CodecError(
            "Insufficient bytes for NTN timing info".to_string(),
        ));
    }

    let satellite_type = match bytes[0] {
        0 => SatelliteType::Leo,
        1 => SatelliteType::Meo,
        2 => SatelliteType::Geo,
        3 => SatelliteType::Haps,
        _ => return Err(NtnSupportError::CodecError("Unknown satellite type".to_string())),
    };
    let satellite_id = u32::from_be_bytes(
        bytes[1..5]
            .try_into()
            .map_err(|_| NtnSupportError::CodecError("Invalid satellite_id bytes".to_string()))?,
    );
    let propagation_delay_us = u64::from_be_bytes(
        bytes[5..13]
            .try_into()
            .map_err(|_| NtnSupportError::CodecError("Invalid propagation_delay_us bytes".to_string()))?,
    );
    let common_ta_us = u64::from_be_bytes(
        bytes[13..21]
            .try_into()
            .map_err(|_| NtnSupportError::CodecError("Invalid common_ta_us bytes".to_string()))?,
    );
    let validity_duration_ms = u32::from_be_bytes(
        bytes[21..25]
            .try_into()
            .map_err(|_| NtnSupportError::CodecError("Invalid validity_duration_ms bytes".to_string()))?,
    );
    let reference_time_ms = u64::from_be_bytes(
        bytes[25..33]
            .try_into()
            .map_err(|_| NtnSupportError::CodecError("Invalid reference_time_ms bytes".to_string()))?,
    );

    Ok(NtnTimingInfo {
        satellite_type,
        satellite_id,
        ephemeris: None,
        position: None,
        velocity: None,
        propagation_delay_us,
        common_ta_us,
        differential_ta_us: None,
        validity_duration_ms,
        reference_time_ms,
        doppler_shift_hz: None,
        k_offset: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_timing_info() -> NtnTimingInfo {
        NtnTimingInfo {
            satellite_type: SatelliteType::Leo,
            satellite_id: 12345,
            ephemeris: Some(EphemerisData {
                semi_major_axis_m: 6_921_000.0,
                eccentricity: 0.001,
                inclination_deg: 53.0,
                raan_deg: 120.0,
                argument_of_periapsis_deg: 90.0,
                mean_anomaly_deg: 45.0,
                epoch_ms: 1700000000000,
            }),
            position: Some(SatellitePositionEcef {
                x_m: 4_000_000.0,
                y_m: 3_000_000.0,
                z_m: 5_000_000.0,
            }),
            velocity: Some(SatelliteVelocityEcef {
                vx_ms: 7500.0,
                vy_ms: -1000.0,
                vz_ms: 500.0,
            }),
            propagation_delay_us: 5000,
            common_ta_us: 10000,
            differential_ta_us: Some(-100),
            validity_duration_ms: 30000,
            reference_time_ms: 1700000000000,
            doppler_shift_hz: Some(15000.0),
            k_offset: Some(16),
        }
    }

    fn create_test_cell_info() -> NtnCellInfo {
        NtnCellInfo {
            nr_cell_identity: 0x123456789,
            plmn_identity: [0x00, 0xF1, 0x10],
            satellite_type: SatelliteType::Leo,
            satellite_id: 12345,
            cell_center_latitude_deg: 40.0,
            cell_center_longitude_deg: -74.0,
            cell_radius_km: 100.0,
            is_earth_fixed: true,
            tac: [0x00, 0x00, 0x01],
            feeder_link_delay_us: Some(10000),
            service_link_delay_us: Some(5000),
            max_doppler_shift_hz: Some(30000.0),
            estimated_handover_time_ms: Some(120000),
        }
    }

    #[test]
    fn test_ntn_timing_info_validate() {
        let info = create_test_timing_info();
        assert!(info.validate().is_ok());
    }

    #[test]
    fn test_ntn_timing_info_invalid_validity() {
        let mut info = create_test_timing_info();
        info.validity_duration_ms = 0;
        assert!(info.validate().is_err());
    }

    #[test]
    fn test_ntn_timing_info_invalid_eccentricity() {
        let mut info = create_test_timing_info();
        if let Some(ref mut ephemeris) = info.ephemeris {
            ephemeris.eccentricity = 1.5;
        }
        assert!(info.validate().is_err());
    }

    #[test]
    fn test_ntn_timing_info_total_ta() {
        let info = create_test_timing_info();
        assert_eq!(info.total_ta_us(), 10000 - 100);
    }

    #[test]
    fn test_ntn_timing_info_round_trip_delay() {
        let info = create_test_timing_info();
        assert_eq!(info.round_trip_delay_us(), 10000);
    }

    #[test]
    fn test_ntn_cell_info_validate() {
        let info = create_test_cell_info();
        assert!(info.validate().is_ok());
    }

    #[test]
    fn test_ntn_cell_info_invalid_latitude() {
        let mut info = create_test_cell_info();
        info.cell_center_latitude_deg = 91.0;
        assert!(info.validate().is_err());
    }

    #[test]
    fn test_ntn_cell_info_invalid_longitude() {
        let mut info = create_test_cell_info();
        info.cell_center_longitude_deg = -181.0;
        assert!(info.validate().is_err());
    }

    #[test]
    fn test_ntn_cell_info_invalid_radius() {
        let mut info = create_test_cell_info();
        info.cell_radius_km = 0.0;
        assert!(info.validate().is_err());
    }

    #[test]
    fn test_ntn_timing_encode_decode() {
        let info = create_test_timing_info();
        let encoded = encode_ntn_timing_info(&info).expect("Failed to encode");
        let decoded = decode_ntn_timing_info(&encoded).expect("Failed to decode");

        assert_eq!(decoded.satellite_type, info.satellite_type);
        assert_eq!(decoded.satellite_id, info.satellite_id);
        assert_eq!(decoded.propagation_delay_us, info.propagation_delay_us);
        assert_eq!(decoded.common_ta_us, info.common_ta_us);
        assert_eq!(decoded.validity_duration_ms, info.validity_duration_ms);
        assert_eq!(decoded.reference_time_ms, info.reference_time_ms);
    }

    #[test]
    fn test_all_satellite_types() {
        let types = [
            SatelliteType::Leo,
            SatelliteType::Meo,
            SatelliteType::Geo,
            SatelliteType::Haps,
        ];

        for sat_type in types {
            let mut info = create_test_timing_info();
            info.satellite_type = sat_type;

            let encoded = encode_ntn_timing_info(&info).expect("Failed to encode");
            let decoded = decode_ntn_timing_info(&encoded).expect("Failed to decode");
            assert_eq!(decoded.satellite_type, sat_type);
        }
    }
}
