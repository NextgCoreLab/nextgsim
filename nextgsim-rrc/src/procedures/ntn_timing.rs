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
pub fn encode_ntn_timing_advance(ta: &NtnTimingAdvance) -> Result<Vec<u8>, NtnTimingError> {
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

    let config_id = u16::from_be_bytes(
        bytes[0..2]
            .try_into()
            .map_err(|_| NtnTimingError::CodecError("Invalid config_id bytes".to_string()))?,
    );
    let orbit_type = match bytes[2] {
        0 => SatelliteOrbitType::Leo,
        1 => SatelliteOrbitType::Meo,
        2 => SatelliteOrbitType::Geo,
        3 => SatelliteOrbitType::Heo,
        4 => SatelliteOrbitType::Haps,
        _ => return Err(NtnTimingError::CodecError("Unknown orbit type".to_string())),
    };
    let satellite_id = u32::from_be_bytes(
        bytes[3..7]
            .try_into()
            .map_err(|_| NtnTimingError::CodecError("Invalid satellite_id bytes".to_string()))?,
    );
    let common_ta_us = u64::from_be_bytes(
        bytes[7..15]
            .try_into()
            .map_err(|_| NtnTimingError::CodecError("Invalid common_ta_us bytes".to_string()))?,
    );

    Ok((config_id, orbit_type, satellite_id, common_ta_us))
}

// ============================================================================
// SIB19 autonomous TA and Doppler pre-compensation (issue #56)
// ============================================================================
//
// TS 38.300 §16.14.2.2: "the UE shall compute the RTT between UE and the RP
// based on the GNSS position, the ephemeris, and the Common TA parameters ...
// and autonomously pre-compensate the T_TA for the RTT between the UE and the
// RP", and "shall compute the frequency Doppler shift of the service link, and
// autonomously pre-compensate for it in the uplink transmissions, by
// considering UE position and the ephemeris."
//
// The types below are in the units SIB19 actually carries, because a
// pre-compensation derived from a differently-scaled copy of the ephemeris is
// wrong in a way no round-trip test would catch. The scale factors are quoted
// from the TS 38.331 `EphemerisInfo` and `TA-Info` field descriptions.
//
// What this deliberately does NOT do: propagate the orbit. Deriving the
// satellite's position at an arbitrary instant from Keplerian elements needs an
// orbit propagator, and `EphemerisInfo-r17`'s `positionVelocity` arm already
// gives the state vector at `epochTime` directly. So the state vector is the
// supported arm here, and the derivation is the geometry at epoch -- which is
// what the UE has, and what a simulator with no independent notion of satellite
// motion can honestly claim.

/// Speed of light in vacuum, m/s (the constant TS 38.211 §4.1 timing rests on).
pub const SPEED_OF_LIGHT_M_S: f64 = 299_792_458.0;

/// `positionX/Y/Z` granularity: "Step of 1.3 m" (TS 38.331 `EphemerisInfo`).
pub const EPHEMERIS_POSITION_STEP_M: f64 = 1.3;

/// `velocityVX/VY/VZ` granularity: "Step of 0.06 m/s" (TS 38.331
/// `EphemerisInfo`).
pub const EPHEMERIS_VELOCITY_STEP_M_S: f64 = 0.06;

/// `ta-Common` granularity: "4.072 x 10^-3 us" (TS 38.331 `TA-Info`).
pub const TA_COMMON_STEP_US: f64 = 4.072e-3;

/// `ta-Common-r17` is `INTEGER(0..66485757)` (TS 38.331 `TA-Info-r17`).
pub const TA_COMMON_MAX: u32 = 66_485_757;

/// `ta-CommonDrift` granularity: "0.2 x 10^-3 us/s" (TS 38.331 `TA-Info`).
pub const TA_COMMON_DRIFT_STEP_US_PER_S: f64 = 0.2e-3;

/// `ta-CommonDrift-r17` is `INTEGER(-257303..257303)` (TS 38.331 `TA-Info-r17`).
pub const TA_COMMON_DRIFT_ABS_MAX: i32 = 257_303;

/// `cellSpecificKoffset-r17` is `INTEGER(1..1023)` (TS 38.331 `NTN-Config-r17`).
pub const CELL_SPECIFIC_K_OFFSET_MIN: u16 = 1;
/// See [`CELL_SPECIFIC_K_OFFSET_MIN`].
pub const CELL_SPECIFIC_K_OFFSET_MAX: u16 = 1023;

/// `PositionStateVector-r17 ::= INTEGER (-33554432..33554431)`.
pub const POSITION_STATE_VECTOR_MIN: i32 = -33_554_432;
/// See [`POSITION_STATE_VECTOR_MIN`].
pub const POSITION_STATE_VECTOR_MAX: i32 = 33_554_431;
/// `VelocityStateVector-r17 ::= INTEGER (-131072..131071)`.
pub const VELOCITY_STATE_VECTOR_MIN: i32 = -131_072;
/// See [`VELOCITY_STATE_VECTOR_MIN`].
pub const VELOCITY_STATE_VECTOR_MAX: i32 = 131_071;

/// The `ntn-UlSyncValidityDuration-r17` ENUMERATED, in seconds, by index.
///
/// `{ s5, s10, s15, s20, s25, s30, s35, s40, s45, s50, s55, s60, s120, s180,
/// s240, s900 }` (TS 38.331 `NTN-Config-r17`). A table rather than an arithmetic
/// rule because the sequence stops being uniform past index 11.
pub const UL_SYNC_VALIDITY_DURATION_S: [u16; 16] = [
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 120, 180, 240, 900,
];

/// A satellite ECEF position/velocity state vector in the raw units of
/// `EphemerisInfo-r17.positionVelocity-r17`.
///
/// Held in the WIRE units rather than in metres so the value the UE computes from
/// is bit-identical to the value the gNB broadcast. Converting to metres on the
/// gNB and back on the UE would quantise twice, and the two ends would then
/// disagree about the derived TA by up to one step with nothing to say why.
/// [`Self::position_m`] and [`Self::velocity_m_s`] apply the spec's scale factors
/// at the point of use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EphemerisStateVector {
    /// `positionX-r17`, `INTEGER (-33554432..33554431)`, step 1.3 m.
    pub position_x: i32,
    /// `positionY-r17`, same encoding.
    pub position_y: i32,
    /// `positionZ-r17`, same encoding.
    pub position_z: i32,
    /// `velocityVX-r17`, `INTEGER (-131072..131071)`, step 0.06 m/s.
    pub velocity_vx: i32,
    /// `velocityVY-r17`, same encoding.
    pub velocity_vy: i32,
    /// `velocityVZ-r17`, same encoding.
    pub velocity_vz: i32,
}

impl EphemerisStateVector {
    /// The ECEF position in metres, `[x, y, z]`.
    pub fn position_m(&self) -> [f64; 3] {
        [
            f64::from(self.position_x) * EPHEMERIS_POSITION_STEP_M,
            f64::from(self.position_y) * EPHEMERIS_POSITION_STEP_M,
            f64::from(self.position_z) * EPHEMERIS_POSITION_STEP_M,
        ]
    }

    /// The ECEF velocity in m/s, `[vx, vy, vz]`.
    pub fn velocity_m_s(&self) -> [f64; 3] {
        [
            f64::from(self.velocity_vx) * EPHEMERIS_VELOCITY_STEP_M_S,
            f64::from(self.velocity_vy) * EPHEMERIS_VELOCITY_STEP_M_S,
            f64::from(self.velocity_vz) * EPHEMERIS_VELOCITY_STEP_M_S,
        ]
    }

    /// Whether every component fits its ASN.1 range, so the value is encodable.
    ///
    /// Checked rather than clamped: a clamped ephemeris describes a satellite that
    /// is not there, and the UE would then pre-compensate for the wrong orbit with
    /// nothing in the log to say the configuration was out of range.
    pub fn is_encodable(&self) -> bool {
        let pos_ok = |v: i32| (POSITION_STATE_VECTOR_MIN..=POSITION_STATE_VECTOR_MAX).contains(&v);
        let vel_ok = |v: i32| (VELOCITY_STATE_VECTOR_MIN..=VELOCITY_STATE_VECTOR_MAX).contains(&v);
        pos_ok(self.position_x)
            && pos_ok(self.position_y)
            && pos_ok(self.position_z)
            && vel_ok(self.velocity_vx)
            && vel_ok(self.velocity_vy)
            && vel_ok(self.velocity_vz)
    }

    /// Builds a state vector from ECEF metres and m/s, rounding to the spec steps.
    ///
    /// Returns `None` when the result would not fit the ASN.1 range -- see
    /// [`Self::is_encodable`].
    pub fn from_ecef(position_m: [f64; 3], velocity_m_s: [f64; 3]) -> Option<Self> {
        fn quantise(v: f64, step: f64) -> Option<i32> {
            let scaled = (v / step).round();
            if scaled.is_finite() && scaled >= f64::from(i32::MIN) && scaled <= f64::from(i32::MAX)
            {
                Some(scaled as i32)
            } else {
                None
            }
        }
        let v = Self {
            position_x: quantise(position_m[0], EPHEMERIS_POSITION_STEP_M)?,
            position_y: quantise(position_m[1], EPHEMERIS_POSITION_STEP_M)?,
            position_z: quantise(position_m[2], EPHEMERIS_POSITION_STEP_M)?,
            velocity_vx: quantise(velocity_m_s[0], EPHEMERIS_VELOCITY_STEP_M_S)?,
            velocity_vy: quantise(velocity_m_s[1], EPHEMERIS_VELOCITY_STEP_M_S)?,
            velocity_vz: quantise(velocity_m_s[2], EPHEMERIS_VELOCITY_STEP_M_S)?,
        };
        v.is_encodable().then_some(v)
    }
}

/// The serving cell's NTN parameters as SIB19 carries them, in wire units.
///
/// This is the value that crosses the air interface: the gNB builds it from its
/// configuration and encodes it into `NTN-Config-r17`
/// (`system_information::build_sib19`), and the UE recovers it by decoding SIB19
/// and derives its pre-compensation from it ([`Self::autonomous_ta_us`],
/// [`Self::uplink_doppler_shift_hz`]). One type for both directions is what makes
/// the two ends agree by construction rather than by two parallel conversions
/// that can drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NtnServingCellConfig {
    /// `epochTime-r17.sfn-r17`: the SFN the ephemeris is referenced to.
    pub epoch_sfn: u16,
    /// `epochTime-r17.subFrameNR-r17` (0..9).
    pub epoch_subframe: u8,
    /// `ntn-UlSyncValidityDuration-r17` as its ENUMERATED index (0..15): how long
    /// the UE may keep applying this configuration. See
    /// [`Self::ul_sync_validity_duration_s`].
    pub ul_sync_validity_index: u8,
    /// `cellSpecificKoffset-r17`, `INTEGER(1..1023)`, in slots.
    pub cell_specific_k_offset: u16,
    /// `ta-Info-r17.ta-Common-r17`, in units of [`TA_COMMON_STEP_US`].
    pub ta_common: u32,
    /// `ta-Info-r17.ta-CommonDrift-r17`, in units of
    /// [`TA_COMMON_DRIFT_STEP_US_PER_S`], when the network broadcasts one.
    pub ta_common_drift: Option<i32>,
    /// `ephemerisInfo-r17.positionVelocity-r17`.
    pub ephemeris: EphemerisStateVector,
}

impl NtnServingCellConfig {
    /// `ta-Common` converted to microseconds.
    pub fn ta_common_us(&self) -> f64 {
        f64::from(self.ta_common) * TA_COMMON_STEP_US
    }

    /// `ntn-UlSyncValidityDuration` in seconds, or `None` when the index is not
    /// one the ENUMERATED defines.
    pub fn ul_sync_validity_duration_s(&self) -> Option<u16> {
        UL_SYNC_VALIDITY_DURATION_S
            .get(usize::from(self.ul_sync_validity_index))
            .copied()
    }

    /// Whether every field fits its ASN.1 range, so this configuration is
    /// encodable into `NTN-Config-r17`.
    pub fn is_encodable(&self) -> bool {
        self.epoch_sfn <= 1023
            && self.epoch_subframe <= 9
            && usize::from(self.ul_sync_validity_index) < UL_SYNC_VALIDITY_DURATION_S.len()
            && (CELL_SPECIFIC_K_OFFSET_MIN..=CELL_SPECIFIC_K_OFFSET_MAX)
                .contains(&self.cell_specific_k_offset)
            && self.ta_common <= TA_COMMON_MAX
            && self
                .ta_common_drift
                .is_none_or(|d| d.abs() <= TA_COMMON_DRIFT_ABS_MAX)
            && self.ephemeris.is_encodable()
    }

    /// The service-link slant range in metres between `ue_position_m` (ECEF) and
    /// the satellite at epoch.
    pub fn slant_range_m(&self, ue_position_m: [f64; 3]) -> f64 {
        let sat = self.ephemeris.position_m();
        let dx = sat[0] - ue_position_m[0];
        let dy = sat[1] - ue_position_m[1];
        let dz = sat[2] - ue_position_m[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// The autonomous timing advance the UE shall apply, in microseconds
    /// (TS 38.300 §16.14.2.2).
    ///
    /// `T_TA = 2 x (service-link one-way delay) + ta_common`: the service-link RTT
    /// the UE derives from its own GNSS position and the broadcast ephemeris, plus
    /// the network-signalled Common TA that covers the feeder link and any offset
    /// the network chose to include (§16.14.2.1: "Common TA is a configured timing
    /// offset that is equal to the RTT between the RP and the NTN payload").
    ///
    /// The service-link term is DOUBLED because the ephemeris gives a *one-way*
    /// geometric range while a timing advance pre-compensates a *round trip*;
    /// `ta_common` is already an RTT and is therefore added as it stands.
    pub fn autonomous_ta_us(&self, ue_position_m: [f64; 3]) -> f64 {
        let one_way_s = self.slant_range_m(ue_position_m) / SPEED_OF_LIGHT_M_S;
        one_way_s * 2.0 * 1e6 + self.ta_common_us()
    }

    /// The range rate (m/s) of the service link at epoch: the satellite's velocity
    /// projected onto the UE→satellite line of sight. Positive when receding.
    pub fn range_rate_m_s(&self, ue_position_m: [f64; 3]) -> f64 {
        let sat = self.ephemeris.position_m();
        let vel = self.ephemeris.velocity_m_s();
        let los = [
            sat[0] - ue_position_m[0],
            sat[1] - ue_position_m[1],
            sat[2] - ue_position_m[2],
        ];
        let range = (los[0] * los[0] + los[1] * los[1] + los[2] * los[2]).sqrt();
        if range == 0.0 {
            // A UE co-located with the satellite: there is no line of sight to
            // project the velocity onto. The geometry is degenerate rather than
            // Doppler-free, and zero is the only answer that does not invent a
            // direction.
            return 0.0;
        }
        (los[0] * vel[0] + los[1] * vel[1] + los[2] * vel[2]) / range
    }

    /// The service-link Doppler shift in Hz the UE shall pre-compensate in its
    /// uplink (TS 38.300 §16.14.2.2), for an uplink carrier of `carrier_freq_hz`.
    ///
    /// `f_d = -(range rate / c) x f_c`. This is the NEGATIVE of the shift the link
    /// imposes, because the value is a *pre*-compensation applied to the UE's
    /// transmitter so the signal arrives at the satellite on the nominal
    /// frequency. A receding satellite (positive range rate) red-shifts the
    /// uplink, so the UE must transmit HIGHER -- which is why the sign is
    /// inverted, and a UE that applied the raw observed shift would double the
    /// error instead of cancelling it.
    pub fn uplink_doppler_shift_hz(&self, ue_position_m: [f64; 3], carrier_freq_hz: f64) -> f64 {
        -(self.range_rate_m_s(ue_position_m) / SPEED_OF_LIGHT_M_S) * carrier_freq_hz
    }
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
        assert!(
            alt > 500_000.0 && alt < 600_000.0,
            "Periapsis altitude was {alt}"
        );
    }

    #[test]
    fn test_ephemeris_apoapsis_altitude() {
        let eph = create_test_ephemeris();
        let alt = eph.apoapsis_altitude_m();
        assert!(
            alt > 540_000.0 && alt < 560_000.0,
            "Apoapsis altitude was {alt}"
        );
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

    // ========================================================================
    // SIB19 autonomous TA and Doppler pre-compensation (issue #56)
    // ========================================================================

    /// Earth's equatorial radius, WGS84. Only used to place the test UE and
    /// satellite at plausible ECEF coordinates.
    const EARTH_RADIUS_M: f64 = 6_378_137.0;

    /// A 600 km LEO satellite directly overhead a UE on the equator at longitude
    /// 0, receding at 1000 m/s radially.
    ///
    /// Directly overhead so the slant range is exactly the altitude and the TA can
    /// be computed by hand; radial velocity so the whole 1000 m/s projects onto the
    /// line of sight and the Doppler can be too. A geometry where either had to be
    /// decomposed would make a sign or axis error invisible.
    fn overhead_leo() -> NtnServingCellConfig {
        let altitude_m = 600_000.0;
        NtnServingCellConfig {
            epoch_sfn: 512,
            epoch_subframe: 3,
            // Index 5 == s30.
            ul_sync_validity_index: 5,
            cell_specific_k_offset: 478,
            // 1 000 000 steps x 4.072e-3 us = 4072 us.
            ta_common: 1_000_000,
            ta_common_drift: Some(-500),
            ephemeris: EphemerisStateVector::from_ecef(
                [EARTH_RADIUS_M + altitude_m, 0.0, 0.0],
                [1000.0, 0.0, 0.0],
            )
            .expect("an overhead LEO is within the ASN.1 ranges"),
        }
    }

    /// The UE, on the equator at longitude 0, directly under [`overhead_leo`].
    fn ue_under_overhead_leo() -> [f64; 3] {
        [EARTH_RADIUS_M, 0.0, 0.0]
    }

    #[test]
    fn the_state_vector_applies_the_spec_scale_factors() {
        let v = EphemerisStateVector {
            position_x: 1000,
            position_y: -2000,
            position_z: 3000,
            velocity_vx: 100,
            velocity_vy: -200,
            velocity_vz: 300,
        };
        // 1.3 m and 0.06 m/s per step (TS 38.331 `EphemerisInfo`).
        assert_eq!(v.position_m(), [1300.0, -2600.0, 3900.0]);
        let vel = v.velocity_m_s();
        assert!((vel[0] - 6.0).abs() < 1e-9);
        assert!((vel[1] + 12.0).abs() < 1e-9);
        assert!((vel[2] - 18.0).abs() < 1e-9);
    }

    /// A round trip through the quantiser must land within one step, and a value
    /// outside the ASN.1 range must be REFUSED rather than clamped.
    #[test]
    fn from_ecef_quantises_within_one_step_and_refuses_the_unencodable() {
        let position = [6_978_137.0, -1_234_567.0, 987_654.0];
        let velocity = [-1234.5, 678.9, -42.0];
        let v = EphemerisStateVector::from_ecef(position, velocity).expect("within range");
        for (got, want) in v.position_m().iter().zip(position.iter()) {
            assert!(
                (got - want).abs() <= EPHEMERIS_POSITION_STEP_M,
                "position {got} must be within one 1.3 m step of {want}"
            );
        }
        for (got, want) in v.velocity_m_s().iter().zip(velocity.iter()) {
            assert!(
                (got - want).abs() <= EPHEMERIS_VELOCITY_STEP_M_S,
                "velocity {got} must be within one 0.06 m/s step of {want}"
            );
        }

        // `positionX` tops out at 33554431 steps == ~43 618 km. A satellite beyond
        // that cannot be broadcast, and saying so is the point.
        let too_far = (f64::from(POSITION_STATE_VECTOR_MAX) + 10.0) * EPHEMERIS_POSITION_STEP_M;
        assert!(
            EphemerisStateVector::from_ecef([too_far, 0.0, 0.0], [0.0, 0.0, 0.0]).is_none(),
            "a position past the ASN.1 range must be refused, not clamped into a \
             satellite that is somewhere else"
        );
    }

    /// The headline number of criterion 3: a non-zero TA derived from the
    /// ephemeris, checked against the hand-computed value rather than against
    /// whatever the code produces.
    ///
    /// 600 km overhead: one-way 600000/299792458 s = 2001.38 us, doubled = 4002.77
    /// us, plus ta-Common 1 000 000 x 4.072e-3 = 4072 us => 8074.77 us.
    #[test]
    fn the_autonomous_ta_is_the_service_link_rtt_plus_common_ta() {
        let cfg = overhead_leo();
        let ue = ue_under_overhead_leo();

        // The satellite is directly overhead, so the slant range is the altitude
        // (to within the 1.3 m quantisation step).
        assert!(
            (cfg.slant_range_m(ue) - 600_000.0).abs() <= EPHEMERIS_POSITION_STEP_M,
            "slant range {} must be the 600 km altitude",
            cfg.slant_range_m(ue)
        );

        let expected_service_link_rtt_us = 2.0 * 600_000.0 / SPEED_OF_LIGHT_M_S * 1e6;
        let expected = expected_service_link_rtt_us + 4072.0;
        let got = cfg.autonomous_ta_us(ue);
        assert!(
            (got - expected).abs() < 0.1,
            "autonomous TA {got} us must be the hand-computed {expected} us \
             (2 x 600 km one-way + ta-Common 4072 us)"
        );
        // And it is emphatically non-zero -- the defect this issue is about.
        assert!(got > 8000.0, "TA {got} us must be non-zero and ~8075 us");

        // ta-Common alone is NOT the answer: a UE that applied only the broadcast
        // Common TA and skipped the ephemeris would land here, so the two must
        // differ by the service-link RTT.
        assert!(
            (got - cfg.ta_common_us()) > 4000.0,
            "the ephemeris must contribute ~4003 us on top of ta-Common; a TA equal \
             to ta-Common means the ephemeris was never read"
        );
    }

    /// A UE further from the satellite must advance MORE. This is what makes the TA
    /// a function of the ephemeris rather than a constant.
    #[test]
    fn a_longer_slant_range_yields_a_larger_autonomous_ta() {
        let cfg = overhead_leo();
        let overhead = cfg.autonomous_ta_us(ue_under_overhead_leo());
        // Same satellite, UE displaced 1000 km along Y: the slant range grows, so
        // the TA must too.
        let oblique = cfg.autonomous_ta_us([EARTH_RADIUS_M, 1_000_000.0, 0.0]);
        assert!(
            oblique > overhead,
            "an oblique UE ({oblique} us) must advance more than one directly under \
             the satellite ({overhead} us)"
        );
    }

    /// `ta-Common` is in 4.072 ns units, and reading it as microseconds directly
    /// would be wrong by a factor of ~245.
    #[test]
    fn ta_common_is_read_in_its_4_072_ns_granularity() {
        let cfg = overhead_leo();
        assert!(
            (cfg.ta_common_us() - 4072.0).abs() < 1e-6,
            "1 000 000 steps of 4.072e-3 us is 4072 us, not {}",
            cfg.ta_common_us()
        );
    }

    /// The Doppler pre-compensation must oppose the observed shift, or the UE
    /// doubles the error instead of cancelling it.
    #[test]
    fn the_uplink_doppler_precompensation_opposes_the_range_rate() {
        let cfg = overhead_leo();
        let ue = ue_under_overhead_leo();
        let carrier_hz = 2e9;

        // Receding at 1000 m/s straight up: the whole velocity is along the line of
        // sight.
        let rate = cfg.range_rate_m_s(ue);
        assert!(
            (rate - 1000.0).abs() < 1.0,
            "range rate {rate} m/s must be the full 1000 m/s radial velocity"
        );

        // f_d = -(1000 / c) * 2e9 = -6671 Hz.
        let expected = -(1000.0 / SPEED_OF_LIGHT_M_S) * carrier_hz;
        let got = cfg.uplink_doppler_shift_hz(ue, carrier_hz);
        assert!(
            (got - expected).abs() < 1.0,
            "uplink Doppler pre-compensation {got} Hz must be the hand-computed \
             {expected} Hz"
        );
        assert!(
            got < 0.0,
            "a RECEDING satellite red-shifts the uplink, so the pre-compensation \
             must be negative-signed relative to the range rate; {got} Hz has the \
             wrong sign and would double the error"
        );

        // An APPROACHING satellite must flip the sign.
        let approaching = NtnServingCellConfig {
            ephemeris: EphemerisStateVector {
                velocity_vx: -cfg.ephemeris.velocity_vx,
                ..cfg.ephemeris
            },
            ..cfg
        };
        assert!(
            approaching.uplink_doppler_shift_hz(ue, carrier_hz) > 0.0,
            "an approaching satellite blue-shifts the uplink, so the \
             pre-compensation must be positive"
        );
    }

    /// The validity ENUMERATED is not uniform past index 11, so a computed
    /// `5 * (i + 1)` would be wrong for the last four values.
    #[test]
    fn the_ul_sync_validity_duration_table_matches_the_enumerated() {
        let cfg = overhead_leo();
        assert_eq!(
            cfg.ul_sync_validity_duration_s(),
            Some(30),
            "index 5 is s30"
        );
        assert_eq!(UL_SYNC_VALIDITY_DURATION_S[11], 60, "index 11 is s60");
        assert_eq!(
            UL_SYNC_VALIDITY_DURATION_S[12], 120,
            "index 12 jumps to s120, not s65"
        );
        assert_eq!(UL_SYNC_VALIDITY_DURATION_S[15], 900, "index 15 is s900");
        let out_of_range = NtnServingCellConfig {
            ul_sync_validity_index: 16,
            ..cfg
        };
        assert_eq!(
            out_of_range.ul_sync_validity_duration_s(),
            None,
            "an index the ENUMERATED does not define must not be invented"
        );
    }

    #[test]
    fn an_out_of_range_field_makes_the_config_unencodable() {
        let cfg = overhead_leo();
        assert!(cfg.is_encodable());
        assert!(
            !NtnServingCellConfig {
                ta_common: TA_COMMON_MAX + 1,
                ..cfg
            }
            .is_encodable(),
            "ta-Common past INTEGER(0..66485757) is not encodable"
        );
        assert!(
            !NtnServingCellConfig {
                cell_specific_k_offset: 0,
                ..cfg
            }
            .is_encodable(),
            "cellSpecificKoffset is INTEGER(1..1023): 0 is not a value"
        );
        assert!(
            !NtnServingCellConfig {
                epoch_subframe: 10,
                ..cfg
            }
            .is_encodable(),
            "subFrameNR is INTEGER(0..9)"
        );
        assert!(
            !NtnServingCellConfig {
                ta_common_drift: Some(TA_COMMON_DRIFT_ABS_MAX + 1),
                ..cfg
            }
            .is_encodable(),
            "ta-CommonDrift is INTEGER(-257303..257303)"
        );
    }

    /// A degenerate geometry must not produce a NaN that then propagates into a
    /// transmit timing.
    #[test]
    fn a_co_located_ue_and_satellite_yield_zero_range_rate_rather_than_nan() {
        let cfg = overhead_leo();
        let at_the_satellite = cfg.ephemeris.position_m();
        let rate = cfg.range_rate_m_s(at_the_satellite);
        assert_eq!(
            rate, 0.0,
            "a zero line of sight has no direction to project onto"
        );
        assert!(cfg
            .uplink_doppler_shift_hz(at_the_satellite, 2e9)
            .is_finite());
    }
}
