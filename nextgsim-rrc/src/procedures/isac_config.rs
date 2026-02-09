//! ISAC (Integrated Sensing and Communication) Measurement Configuration
//!
//! 6G extension: RRC configuration for ISAC sensing measurements at the UE
//! and gNB level. Supports configuring sensing waveforms, measurement
//! parameters, and reporting for joint sensing and communication.
//!
//! This module implements:
//! - `IsacMeasConfig` - ISAC measurement configuration with sensing periodicity
//! - Waveform configuration for sensing signals
//! - Angle, range, and velocity measurement parameters

use thiserror::Error;

/// Errors that can occur during ISAC configuration procedures
#[derive(Debug, Error)]
pub enum IsacConfigError {
    /// Invalid ISAC configuration
    #[error("Invalid ISAC configuration: {0}")]
    InvalidConfig(String),

    /// Missing mandatory field
    #[error("Missing mandatory field: {0}")]
    MissingMandatoryField(String),

    /// Encoding/decoding error
    #[error("Codec error: {0}")]
    CodecError(String),
}

/// Sensing waveform type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsacWaveformType {
    /// OFDM-based sensing (reusing communication waveform)
    Ofdm,
    /// FMCW (Frequency Modulated Continuous Wave)
    Fmcw,
    /// PMCW (Phase Modulated Continuous Wave)
    Pmcw,
    /// Pulsed radar waveform
    Pulsed,
    /// Joint communication-sensing waveform
    JointComSens,
    /// OTFS (Orthogonal Time Frequency Space) for high-mobility sensing
    Otfs,
}

/// Sensing mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SensingMode {
    /// Monostatic: transmit and receive at same node
    Monostatic,
    /// Bistatic: transmit and receive at different nodes
    Bistatic,
    /// Multistatic: multiple transmitters and/or receivers
    Multistatic,
    /// UE-assisted sensing (UE acts as passive receiver)
    UeAssisted,
    /// Network-assisted sensing (network provides sensing info to UE)
    NetworkAssisted,
}

/// Sensing signal resource type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SensingResourceType {
    /// Dedicated sensing signals
    Dedicated,
    /// Shared with communication (dual-purpose reference signals)
    SharedWithComm,
    /// Opportunistic sensing using existing signals
    Opportunistic,
}

/// Sensing periodicity values
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SensingPeriodicity {
    /// 1 ms
    Ms1,
    /// 2 ms
    Ms2,
    /// 5 ms
    Ms5,
    /// 10 ms
    Ms10,
    /// 20 ms
    Ms20,
    /// 40 ms
    Ms40,
    /// 80 ms
    Ms80,
    /// 160 ms
    Ms160,
    /// 320 ms
    Ms320,
    /// 640 ms
    Ms640,
    /// 1280 ms
    Ms1280,
}

impl SensingPeriodicity {
    /// Get the periodicity value in milliseconds
    pub fn to_ms(&self) -> u32 {
        match self {
            SensingPeriodicity::Ms1 => 1,
            SensingPeriodicity::Ms2 => 2,
            SensingPeriodicity::Ms5 => 5,
            SensingPeriodicity::Ms10 => 10,
            SensingPeriodicity::Ms20 => 20,
            SensingPeriodicity::Ms40 => 40,
            SensingPeriodicity::Ms80 => 80,
            SensingPeriodicity::Ms160 => 160,
            SensingPeriodicity::Ms320 => 320,
            SensingPeriodicity::Ms640 => 640,
            SensingPeriodicity::Ms1280 => 1280,
        }
    }
}

/// Waveform configuration for sensing
#[derive(Debug, Clone)]
pub struct IsacWaveformConfig {
    /// Waveform type
    pub waveform_type: IsacWaveformType,
    /// Center frequency in MHz
    pub center_frequency_mhz: f64,
    /// Bandwidth in MHz
    pub bandwidth_mhz: f64,
    /// Subcarrier spacing in kHz (for OFDM-based waveforms)
    pub subcarrier_spacing_khz: Option<u16>,
    /// Number of OFDM symbols per sensing burst
    pub symbols_per_burst: Option<u16>,
    /// Number of subcarriers (for OFDM/OTFS)
    pub num_subcarriers: Option<u16>,
    /// Chirp duration in microseconds (for FMCW)
    pub chirp_duration_us: Option<f64>,
    /// Number of chirps per burst (for FMCW)
    pub chirps_per_burst: Option<u16>,
    /// Pulse width in nanoseconds (for pulsed)
    pub pulse_width_ns: Option<f64>,
    /// Pulse repetition interval in microseconds (for pulsed)
    pub pri_us: Option<f64>,
    /// Transmit power in dBm
    pub tx_power_dbm: Option<f64>,
}

/// Range measurement parameters
#[derive(Debug, Clone)]
pub struct RangeMeasParams {
    /// Maximum unambiguous range in meters
    pub max_range_m: f64,
    /// Range resolution in meters
    pub range_resolution_m: f64,
    /// Minimum detectable range in meters
    pub min_range_m: f64,
    /// Number of range bins
    pub num_range_bins: u16,
}

/// Velocity measurement parameters
#[derive(Debug, Clone)]
pub struct VelocityMeasParams {
    /// Maximum unambiguous velocity in m/s
    pub max_velocity_ms: f64,
    /// Velocity resolution in m/s
    pub velocity_resolution_ms: f64,
    /// Number of Doppler bins
    pub num_doppler_bins: u16,
}

/// Angle measurement parameters
#[derive(Debug, Clone)]
pub struct AngleMeasParams {
    /// Azimuth field of view in degrees (-180 to 180)
    pub azimuth_fov_deg: (f64, f64),
    /// Elevation field of view in degrees (-90 to 90)
    pub elevation_fov_deg: (f64, f64),
    /// Azimuth angular resolution in degrees
    pub azimuth_resolution_deg: f64,
    /// Elevation angular resolution in degrees
    pub elevation_resolution_deg: f64,
    /// Number of antenna elements (for beamforming)
    pub num_antenna_elements: u16,
    /// Antenna array type
    pub array_type: AntennaArrayType,
}

/// Antenna array type for sensing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AntennaArrayType {
    /// Uniform Linear Array
    Ula,
    /// Uniform Planar Array
    Upa,
    /// Uniform Circular Array
    Uca,
    /// Conformal Array
    Conformal,
}

/// ISAC reporting configuration
#[derive(Debug, Clone)]
pub struct IsacReportingConfig {
    /// Reporting periodicity
    pub periodicity: SensingPeriodicity,
    /// Maximum number of detected targets per report
    pub max_targets_per_report: u8,
    /// Minimum detection threshold (dB above noise floor)
    pub detection_threshold_db: f64,
    /// Report range measurements
    pub report_range: bool,
    /// Report velocity measurements
    pub report_velocity: bool,
    /// Report angle measurements
    pub report_angle: bool,
    /// Report radar cross section (RCS) estimates
    pub report_rcs: bool,
    /// Report signal-to-noise ratio
    pub report_snr: bool,
    /// Report raw IQ data (for advanced processing)
    pub report_raw_iq: bool,
}

/// ISAC measurement configuration
///
/// Contains the full configuration for ISAC sensing measurements,
/// including waveform, measurement parameters, and reporting.
#[derive(Debug, Clone)]
pub struct IsacMeasConfig {
    /// Configuration ID
    pub config_id: u16,
    /// Sensing mode
    pub sensing_mode: SensingMode,
    /// Sensing resource type
    pub resource_type: SensingResourceType,
    /// Sensing periodicity
    pub periodicity: SensingPeriodicity,
    /// Waveform configuration
    pub waveform_config: IsacWaveformConfig,
    /// Range measurement parameters
    pub range_params: Option<RangeMeasParams>,
    /// Velocity measurement parameters
    pub velocity_params: Option<VelocityMeasParams>,
    /// Angle measurement parameters
    pub angle_params: Option<AngleMeasParams>,
    /// Reporting configuration
    pub reporting_config: IsacReportingConfig,
    /// Maximum sensing duty cycle (0.0 to 1.0)
    pub max_sensing_duty_cycle: f64,
    /// Communication protection: minimum guaranteed communication throughput ratio (0.0 to 1.0)
    pub min_comm_throughput_ratio: f64,
    /// Validity timer in seconds (0 = indefinite)
    pub validity_timer_s: u32,
}

impl IsacMeasConfig {
    /// Validate the ISAC measurement configuration
    pub fn validate(&self) -> Result<(), IsacConfigError> {
        // Validate waveform config
        if self.waveform_config.bandwidth_mhz <= 0.0 {
            return Err(IsacConfigError::InvalidConfig(
                "Bandwidth must be > 0 MHz".to_string(),
            ));
        }
        if self.waveform_config.center_frequency_mhz <= 0.0 {
            return Err(IsacConfigError::InvalidConfig(
                "Center frequency must be > 0 MHz".to_string(),
            ));
        }

        // Validate SCS for OFDM-based waveforms
        if matches!(
            self.waveform_config.waveform_type,
            IsacWaveformType::Ofdm | IsacWaveformType::JointComSens | IsacWaveformType::Otfs
        ) {
            if let Some(scs) = self.waveform_config.subcarrier_spacing_khz {
                match scs {
                    15 | 30 | 60 | 120 | 240 | 480 | 960 => {}
                    _ => {
                        return Err(IsacConfigError::InvalidConfig(format!(
                            "Invalid subcarrier spacing: {scs} kHz"
                        )))
                    }
                }
            }
        }

        // Validate range parameters
        if let Some(ref range) = self.range_params {
            if range.max_range_m <= 0.0 {
                return Err(IsacConfigError::InvalidConfig(
                    "Maximum range must be > 0".to_string(),
                ));
            }
            if range.range_resolution_m <= 0.0 {
                return Err(IsacConfigError::InvalidConfig(
                    "Range resolution must be > 0".to_string(),
                ));
            }
            if range.min_range_m < 0.0 {
                return Err(IsacConfigError::InvalidConfig(
                    "Minimum range must be >= 0".to_string(),
                ));
            }
            if range.min_range_m >= range.max_range_m {
                return Err(IsacConfigError::InvalidConfig(
                    "Minimum range must be < maximum range".to_string(),
                ));
            }
        }

        // Validate velocity parameters
        if let Some(ref vel) = self.velocity_params {
            if vel.max_velocity_ms <= 0.0 {
                return Err(IsacConfigError::InvalidConfig(
                    "Maximum velocity must be > 0".to_string(),
                ));
            }
            if vel.velocity_resolution_ms <= 0.0 {
                return Err(IsacConfigError::InvalidConfig(
                    "Velocity resolution must be > 0".to_string(),
                ));
            }
        }

        // Validate angle parameters
        if let Some(ref angle) = self.angle_params {
            if angle.azimuth_resolution_deg <= 0.0 || angle.elevation_resolution_deg <= 0.0 {
                return Err(IsacConfigError::InvalidConfig(
                    "Angular resolution must be > 0".to_string(),
                ));
            }
            if angle.azimuth_fov_deg.0 >= angle.azimuth_fov_deg.1 {
                return Err(IsacConfigError::InvalidConfig(
                    "Azimuth FOV min must be < max".to_string(),
                ));
            }
            if angle.elevation_fov_deg.0 >= angle.elevation_fov_deg.1 {
                return Err(IsacConfigError::InvalidConfig(
                    "Elevation FOV min must be < max".to_string(),
                ));
            }
        }

        // Validate duty cycle
        if self.max_sensing_duty_cycle <= 0.0 || self.max_sensing_duty_cycle > 1.0 {
            return Err(IsacConfigError::InvalidConfig(
                "Max sensing duty cycle must be in range (0.0, 1.0]".to_string(),
            ));
        }

        // Validate communication protection
        if self.min_comm_throughput_ratio < 0.0 || self.min_comm_throughput_ratio > 1.0 {
            return Err(IsacConfigError::InvalidConfig(
                "Min communication throughput ratio must be in range [0.0, 1.0]".to_string(),
            ));
        }

        // Validate that sensing + communication don't exceed 100%
        if self.max_sensing_duty_cycle + self.min_comm_throughput_ratio > 1.0 {
            return Err(IsacConfigError::InvalidConfig(
                "Sum of sensing duty cycle and communication throughput ratio exceeds 1.0"
                    .to_string(),
            ));
        }

        // Validate reporting config
        if self.reporting_config.detection_threshold_db < 0.0 {
            return Err(IsacConfigError::InvalidConfig(
                "Detection threshold must be >= 0 dB".to_string(),
            ));
        }

        Ok(())
    }
}

/// Encode ISAC measurement config to bytes (simplified serialization)
pub fn encode_isac_meas_config(config: &IsacMeasConfig) -> Result<Vec<u8>, IsacConfigError> {
    config.validate()?;
    let mut bytes = Vec::with_capacity(32);

    // config_id (2 bytes)
    bytes.extend_from_slice(&config.config_id.to_be_bytes());
    // sensing_mode (1 byte)
    bytes.push(match config.sensing_mode {
        SensingMode::Monostatic => 0,
        SensingMode::Bistatic => 1,
        SensingMode::Multistatic => 2,
        SensingMode::UeAssisted => 3,
        SensingMode::NetworkAssisted => 4,
    });
    // resource_type (1 byte)
    bytes.push(match config.resource_type {
        SensingResourceType::Dedicated => 0,
        SensingResourceType::SharedWithComm => 1,
        SensingResourceType::Opportunistic => 2,
    });
    // periodicity in ms (4 bytes)
    bytes.extend_from_slice(&config.periodicity.to_ms().to_be_bytes());
    // waveform_type (1 byte)
    bytes.push(match config.waveform_config.waveform_type {
        IsacWaveformType::Ofdm => 0,
        IsacWaveformType::Fmcw => 1,
        IsacWaveformType::Pmcw => 2,
        IsacWaveformType::Pulsed => 3,
        IsacWaveformType::JointComSens => 4,
        IsacWaveformType::Otfs => 5,
    });
    // bandwidth in MHz as u32 (4 bytes)
    bytes.extend_from_slice(&(config.waveform_config.bandwidth_mhz as u32).to_be_bytes());
    // validity_timer_s (4 bytes)
    bytes.extend_from_slice(&config.validity_timer_s.to_be_bytes());

    Ok(bytes)
}

/// Decode ISAC measurement config header from bytes (simplified deserialization)
pub fn decode_isac_meas_config_header(
    bytes: &[u8],
) -> Result<(u16, SensingMode, SensingResourceType), IsacConfigError> {
    if bytes.len() < 4 {
        return Err(IsacConfigError::CodecError(
            "Insufficient bytes for ISAC config header".to_string(),
        ));
    }

    let config_id = u16::from_be_bytes(bytes[0..2].try_into().unwrap());
    let sensing_mode = match bytes[2] {
        0 => SensingMode::Monostatic,
        1 => SensingMode::Bistatic,
        2 => SensingMode::Multistatic,
        3 => SensingMode::UeAssisted,
        4 => SensingMode::NetworkAssisted,
        _ => {
            return Err(IsacConfigError::CodecError(
                "Unknown sensing mode".to_string(),
            ))
        }
    };
    let resource_type = match bytes[3] {
        0 => SensingResourceType::Dedicated,
        1 => SensingResourceType::SharedWithComm,
        2 => SensingResourceType::Opportunistic,
        _ => {
            return Err(IsacConfigError::CodecError(
                "Unknown resource type".to_string(),
            ))
        }
    };

    Ok((config_id, sensing_mode, resource_type))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_waveform() -> IsacWaveformConfig {
        IsacWaveformConfig {
            waveform_type: IsacWaveformType::Ofdm,
            center_frequency_mhz: 28000.0,
            bandwidth_mhz: 400.0,
            subcarrier_spacing_khz: Some(120),
            symbols_per_burst: Some(14),
            num_subcarriers: Some(3300),
            chirp_duration_us: None,
            chirps_per_burst: None,
            pulse_width_ns: None,
            pri_us: None,
            tx_power_dbm: Some(23.0),
        }
    }

    fn create_test_config() -> IsacMeasConfig {
        IsacMeasConfig {
            config_id: 1,
            sensing_mode: SensingMode::Monostatic,
            resource_type: SensingResourceType::SharedWithComm,
            periodicity: SensingPeriodicity::Ms20,
            waveform_config: create_test_waveform(),
            range_params: Some(RangeMeasParams {
                max_range_m: 300.0,
                range_resolution_m: 0.5,
                min_range_m: 1.0,
                num_range_bins: 600,
            }),
            velocity_params: Some(VelocityMeasParams {
                max_velocity_ms: 100.0,
                velocity_resolution_ms: 0.5,
                num_doppler_bins: 400,
            }),
            angle_params: Some(AngleMeasParams {
                azimuth_fov_deg: (-60.0, 60.0),
                elevation_fov_deg: (-30.0, 30.0),
                azimuth_resolution_deg: 1.0,
                elevation_resolution_deg: 2.0,
                num_antenna_elements: 64,
                array_type: AntennaArrayType::Upa,
            }),
            reporting_config: IsacReportingConfig {
                periodicity: SensingPeriodicity::Ms80,
                max_targets_per_report: 16,
                detection_threshold_db: 10.0,
                report_range: true,
                report_velocity: true,
                report_angle: true,
                report_rcs: true,
                report_snr: true,
                report_raw_iq: false,
            },
            max_sensing_duty_cycle: 0.3,
            min_comm_throughput_ratio: 0.6,
            validity_timer_s: 3600,
        }
    }

    #[test]
    fn test_isac_config_validate() {
        let config = create_test_config();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_isac_config_invalid_bandwidth() {
        let mut config = create_test_config();
        config.waveform_config.bandwidth_mhz = 0.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_isac_config_invalid_frequency() {
        let mut config = create_test_config();
        config.waveform_config.center_frequency_mhz = -1.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_isac_config_invalid_scs() {
        let mut config = create_test_config();
        config.waveform_config.subcarrier_spacing_khz = Some(100);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_isac_config_valid_scs_values() {
        let valid_scs = [15, 30, 60, 120, 240, 480, 960];
        for scs in valid_scs {
            let mut config = create_test_config();
            config.waveform_config.subcarrier_spacing_khz = Some(scs);
            assert!(config.validate().is_ok(), "SCS {scs} should be valid");
        }
    }

    #[test]
    fn test_isac_config_invalid_range_params() {
        let mut config = create_test_config();
        config.range_params = Some(RangeMeasParams {
            max_range_m: 0.0,
            range_resolution_m: 0.5,
            min_range_m: 1.0,
            num_range_bins: 600,
        });
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_isac_config_min_range_exceeds_max() {
        let mut config = create_test_config();
        config.range_params = Some(RangeMeasParams {
            max_range_m: 100.0,
            range_resolution_m: 0.5,
            min_range_m: 200.0,
            num_range_bins: 600,
        });
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_isac_config_invalid_velocity_params() {
        let mut config = create_test_config();
        config.velocity_params = Some(VelocityMeasParams {
            max_velocity_ms: -1.0,
            velocity_resolution_ms: 0.5,
            num_doppler_bins: 400,
        });
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_isac_config_invalid_angle_params() {
        let mut config = create_test_config();
        config.angle_params = Some(AngleMeasParams {
            azimuth_fov_deg: (60.0, -60.0), // min > max
            elevation_fov_deg: (-30.0, 30.0),
            azimuth_resolution_deg: 1.0,
            elevation_resolution_deg: 2.0,
            num_antenna_elements: 64,
            array_type: AntennaArrayType::Upa,
        });
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_isac_config_invalid_duty_cycle() {
        let mut config = create_test_config();
        config.max_sensing_duty_cycle = 0.0;
        assert!(config.validate().is_err());

        config.max_sensing_duty_cycle = 1.1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_isac_config_duty_cycle_plus_comm_exceeds_one() {
        let mut config = create_test_config();
        config.max_sensing_duty_cycle = 0.6;
        config.min_comm_throughput_ratio = 0.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_isac_config_invalid_detection_threshold() {
        let mut config = create_test_config();
        config.reporting_config.detection_threshold_db = -5.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_sensing_periodicity_values() {
        assert_eq!(SensingPeriodicity::Ms1.to_ms(), 1);
        assert_eq!(SensingPeriodicity::Ms10.to_ms(), 10);
        assert_eq!(SensingPeriodicity::Ms80.to_ms(), 80);
        assert_eq!(SensingPeriodicity::Ms1280.to_ms(), 1280);
    }

    #[test]
    fn test_fmcw_waveform() {
        let mut config = create_test_config();
        config.waveform_config = IsacWaveformConfig {
            waveform_type: IsacWaveformType::Fmcw,
            center_frequency_mhz: 77000.0,
            bandwidth_mhz: 4000.0,
            subcarrier_spacing_khz: None,
            symbols_per_burst: None,
            num_subcarriers: None,
            chirp_duration_us: Some(10.0),
            chirps_per_burst: Some(128),
            pulse_width_ns: None,
            pri_us: None,
            tx_power_dbm: Some(10.0),
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_encode_decode_isac_config() {
        let config = create_test_config();
        let encoded = encode_isac_meas_config(&config).expect("Failed to encode");
        assert!(!encoded.is_empty());

        let (config_id, sensing_mode, resource_type) =
            decode_isac_meas_config_header(&encoded).expect("Failed to decode");
        assert_eq!(config_id, 1);
        assert_eq!(sensing_mode, SensingMode::Monostatic);
        assert_eq!(resource_type, SensingResourceType::SharedWithComm);
    }

    #[test]
    fn test_all_sensing_modes() {
        let modes = [
            SensingMode::Monostatic,
            SensingMode::Bistatic,
            SensingMode::Multistatic,
            SensingMode::UeAssisted,
            SensingMode::NetworkAssisted,
        ];
        for mode in modes {
            let mut config = create_test_config();
            config.sensing_mode = mode;
            let encoded = encode_isac_meas_config(&config).expect("Failed to encode");
            let (_, decoded_mode, _) =
                decode_isac_meas_config_header(&encoded).expect("Failed to decode");
            assert_eq!(decoded_mode, mode);
        }
    }

    #[test]
    fn test_all_waveform_types() {
        let waveforms = [
            IsacWaveformType::Ofdm,
            IsacWaveformType::Fmcw,
            IsacWaveformType::Pmcw,
            IsacWaveformType::Pulsed,
            IsacWaveformType::JointComSens,
            IsacWaveformType::Otfs,
        ];
        for wf in waveforms {
            let mut config = create_test_config();
            config.waveform_config.waveform_type = wf;
            // Remove SCS validation for non-OFDM types
            if !matches!(
                wf,
                IsacWaveformType::Ofdm | IsacWaveformType::JointComSens | IsacWaveformType::Otfs
            ) {
                config.waveform_config.subcarrier_spacing_khz = None;
            }
            assert!(config.validate().is_ok(), "Waveform type {wf:?} should be valid");
        }
    }

    #[test]
    fn test_all_antenna_array_types() {
        let types = [
            AntennaArrayType::Ula,
            AntennaArrayType::Upa,
            AntennaArrayType::Uca,
            AntennaArrayType::Conformal,
        ];
        for array_type in types {
            let mut config = create_test_config();
            if let Some(ref mut angle) = config.angle_params {
                angle.array_type = array_type;
            }
            assert!(config.validate().is_ok());
        }
    }

    #[test]
    fn test_config_without_optional_params() {
        let config = IsacMeasConfig {
            config_id: 2,
            sensing_mode: SensingMode::UeAssisted,
            resource_type: SensingResourceType::Opportunistic,
            periodicity: SensingPeriodicity::Ms160,
            waveform_config: IsacWaveformConfig {
                waveform_type: IsacWaveformType::Fmcw,
                center_frequency_mhz: 60000.0,
                bandwidth_mhz: 1000.0,
                subcarrier_spacing_khz: None,
                symbols_per_burst: None,
                num_subcarriers: None,
                chirp_duration_us: Some(5.0),
                chirps_per_burst: Some(64),
                pulse_width_ns: None,
                pri_us: None,
                tx_power_dbm: None,
            },
            range_params: None,
            velocity_params: None,
            angle_params: None,
            reporting_config: IsacReportingConfig {
                periodicity: SensingPeriodicity::Ms640,
                max_targets_per_report: 4,
                detection_threshold_db: 15.0,
                report_range: true,
                report_velocity: false,
                report_angle: false,
                report_rcs: false,
                report_snr: true,
                report_raw_iq: false,
            },
            max_sensing_duty_cycle: 0.1,
            min_comm_throughput_ratio: 0.8,
            validity_timer_s: 0,
        };
        assert!(config.validate().is_ok());
    }
}
