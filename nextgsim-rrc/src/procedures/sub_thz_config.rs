//! Sub-THz Band Configuration
//!
//! 6G extension: RRC configuration for sub-terahertz (sub-THz) frequency bands
//! in the range of 100 GHz to 300 GHz. Supports beam management, bandwidth part
//! configuration, and channel-specific parameters for ultra-high-frequency operation.
//!
//! This module implements:
//! - `SubThzBandConfig` - Sub-THz band configuration with frequency range and BWP
//! - Beam management parameters for sub-THz
//! - Channel model and propagation parameters

use thiserror::Error;

/// Errors that can occur during sub-THz configuration procedures
#[derive(Debug, Error)]
pub enum SubThzConfigError {
    /// Invalid sub-THz configuration
    #[error("Invalid sub-THz configuration: {0}")]
    InvalidConfig(String),

    /// Missing mandatory field
    #[error("Missing mandatory field: {0}")]
    MissingMandatoryField(String),

    /// Encoding/decoding error
    #[error("Codec error: {0}")]
    CodecError(String),
}

/// Sub-THz frequency band designation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubThzBand {
    /// D-band: 110-170 GHz
    DBand,
    /// H-band: 220-330 GHz (WR-3.4)
    HBand,
    /// Custom frequency range in GHz
    Custom { min_ghz: u16, max_ghz: u16 },
}

impl SubThzBand {
    /// Get the frequency range in GHz
    pub fn frequency_range_ghz(&self) -> (u16, u16) {
        match self {
            SubThzBand::DBand => (110, 170),
            SubThzBand::HBand => (220, 330),
            SubThzBand::Custom { min_ghz, max_ghz } => (*min_ghz, *max_ghz),
        }
    }
}

/// Subcarrier spacing for sub-THz (extended NR numerology)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubThzSubcarrierSpacing {
    /// 480 kHz (mu=5)
    Khz480,
    /// 960 kHz (mu=6)
    Khz960,
    /// 1920 kHz (mu=7, sub-THz extension)
    Khz1920,
    /// 3840 kHz (mu=8, sub-THz extension)
    Khz3840,
}

impl SubThzSubcarrierSpacing {
    /// Get the subcarrier spacing value in kHz
    pub fn to_khz(&self) -> u32 {
        match self {
            SubThzSubcarrierSpacing::Khz480 => 480,
            SubThzSubcarrierSpacing::Khz960 => 960,
            SubThzSubcarrierSpacing::Khz1920 => 1920,
            SubThzSubcarrierSpacing::Khz3840 => 3840,
        }
    }

    /// Get the NR numerology index (mu)
    pub fn numerology_index(&self) -> u8 {
        match self {
            SubThzSubcarrierSpacing::Khz480 => 5,
            SubThzSubcarrierSpacing::Khz960 => 6,
            SubThzSubcarrierSpacing::Khz1920 => 7,
            SubThzSubcarrierSpacing::Khz3840 => 8,
        }
    }
}

/// Bandwidth part (BWP) configuration for sub-THz
#[derive(Debug, Clone)]
pub struct SubThzBwpConfig {
    /// BWP ID (0-3)
    pub bwp_id: u8,
    /// Center frequency in GHz
    pub center_frequency_ghz: f64,
    /// Bandwidth in GHz
    pub bandwidth_ghz: f64,
    /// Subcarrier spacing
    pub subcarrier_spacing: SubThzSubcarrierSpacing,
    /// Number of resource blocks
    pub num_resource_blocks: u16,
    /// Whether this is the active BWP
    pub is_active: bool,
    /// Cyclic prefix type
    pub cyclic_prefix: CyclicPrefixType,
}

/// Cyclic prefix type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CyclicPrefixType {
    /// Normal cyclic prefix
    Normal,
    /// Extended cyclic prefix (for high delay spread)
    Extended,
    /// Ultra-short cyclic prefix (for sub-THz short delay spread)
    UltraShort,
}

/// Beam management configuration for sub-THz
///
/// Sub-THz requires very narrow beams due to high path loss,
/// which necessitates more sophisticated beam management.
#[derive(Debug, Clone)]
pub struct SubThzBeamConfig {
    /// Maximum number of simultaneous beams
    pub max_beams: u8,
    /// Beam width in degrees (typically very narrow at sub-THz)
    pub beam_width_deg: f64,
    /// Beam sweeping periodicity in milliseconds
    pub beam_sweep_periodicity_ms: u32,
    /// Number of SSB beams in sweep
    pub num_ssb_beams: u8,
    /// Enable beam tracking (continuous beam refinement)
    pub beam_tracking_enabled: bool,
    /// Beam tracking update interval in milliseconds
    pub beam_tracking_interval_ms: Option<u32>,
    /// Maximum beam misalignment tolerance in degrees
    pub max_misalignment_deg: f64,
    /// Enable multi-beam operation (simultaneous TX/RX on multiple beams)
    pub multi_beam_enabled: bool,
    /// Beam failure detection configuration
    pub beam_failure_config: Option<BeamFailureConfig>,
    /// Enable AI-assisted beam prediction
    pub ai_beam_prediction: bool,
}

/// Beam failure detection and recovery configuration
#[derive(Debug, Clone)]
pub struct BeamFailureConfig {
    /// Number of consecutive beam failure indications before declaring failure
    pub failure_instance_count: u8,
    /// Beam failure detection timer in milliseconds
    pub detection_timer_ms: u32,
    /// Maximum number of candidate beams for recovery
    pub max_candidate_beams: u8,
    /// Beam recovery timer in milliseconds
    pub recovery_timer_ms: u32,
    /// RSRP threshold for beam failure in dBm
    pub rsrp_threshold_dbm: i16,
}

/// Channel propagation model for sub-THz
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubThzChannelModel {
    /// Line-of-sight (`LoS`) dominant
    LineOfSight,
    /// Non-line-of-sight (`NLoS`) via reflection
    NonLineOfSight,
    /// Reconfigurable Intelligent Surface (RIS) assisted
    RisAssisted,
    /// Indoor short-range
    IndoorShortRange,
    /// Outdoor urban
    OutdoorUrban,
    /// Backhaul/fronthaul link
    BackhaulFronthaul,
}

/// Atmospheric absorption parameters for sub-THz
#[derive(Debug, Clone)]
pub struct AtmosphericConfig {
    /// Expected path loss exponent
    pub path_loss_exponent: f64,
    /// Additional atmospheric absorption in dB/km
    pub atmospheric_absorption_db_per_km: f64,
    /// Rain attenuation in dB/km (at reference rain rate)
    pub rain_attenuation_db_per_km: Option<f64>,
    /// Humidity attenuation factor
    pub humidity_factor: f64,
    /// Maximum supported link distance in meters
    pub max_link_distance_m: f64,
    /// Enable weather-adaptive modulation
    pub weather_adaptive: bool,
}

/// Power configuration for sub-THz
#[derive(Debug, Clone)]
pub struct SubThzPowerConfig {
    /// Maximum transmit power in dBm
    pub max_tx_power_dbm: f64,
    /// Minimum transmit power in dBm
    pub min_tx_power_dbm: f64,
    /// Power amplifier efficiency (0.0 to 1.0)
    pub pa_efficiency: f64,
    /// Enable power boosting for sub-THz
    pub power_boosting_enabled: bool,
    /// Power back-off in dB (for PAPR management)
    pub power_backoff_db: f64,
}

/// Modulation scheme support for sub-THz
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubThzModulation {
    /// BPSK
    Bpsk,
    /// QPSK
    Qpsk,
    /// 16-QAM
    Qam16,
    /// 64-QAM
    Qam64,
    /// 256-QAM
    Qam256,
    /// 1024-QAM
    Qam1024,
    /// OOK (On-Off Keying) - simpler modulation for sub-THz
    Ook,
}

/// Sub-THz band configuration
///
/// Contains the full configuration for sub-THz band operation,
/// including frequency parameters, beam management, channel model,
/// and power configuration.
#[derive(Debug, Clone)]
pub struct SubThzBandConfig {
    /// Configuration ID
    pub config_id: u16,
    /// Sub-THz frequency band
    pub band: SubThzBand,
    /// Center frequency in GHz
    pub center_frequency_ghz: f64,
    /// Total system bandwidth in GHz
    pub total_bandwidth_ghz: f64,
    /// Bandwidth parts configuration
    pub bwp_configs: Vec<SubThzBwpConfig>,
    /// Beam management configuration
    pub beam_config: SubThzBeamConfig,
    /// Channel propagation model
    pub channel_model: SubThzChannelModel,
    /// Atmospheric absorption parameters
    pub atmospheric_config: AtmosphericConfig,
    /// Power configuration
    pub power_config: SubThzPowerConfig,
    /// Supported modulation schemes (ordered by preference)
    pub supported_modulations: Vec<SubThzModulation>,
    /// Maximum supported MCS index
    pub max_mcs_index: u8,
    /// Enable RIS-assisted communication
    pub ris_enabled: bool,
    /// RIS element count (if RIS is enabled)
    pub ris_element_count: Option<u32>,
    /// Enable molecular absorption aware scheduling
    pub molecular_absorption_aware: bool,
}

impl SubThzBandConfig {
    /// Validate the sub-THz band configuration
    pub fn validate(&self) -> Result<(), SubThzConfigError> {
        // Validate frequency range (sub-THz: 100-300 GHz)
        if self.center_frequency_ghz < 100.0 || self.center_frequency_ghz > 300.0 {
            return Err(SubThzConfigError::InvalidConfig(
                "Center frequency must be in sub-THz range (100-300 GHz)".to_string(),
            ));
        }

        // Validate bandwidth
        if self.total_bandwidth_ghz <= 0.0 {
            return Err(SubThzConfigError::InvalidConfig(
                "Total bandwidth must be > 0 GHz".to_string(),
            ));
        }

        // Validate bandwidth doesn't exceed band range
        let (band_min, band_max) = self.band.frequency_range_ghz();
        let half_bw = self.total_bandwidth_ghz / 2.0;
        if (self.center_frequency_ghz - half_bw) < band_min as f64
            || (self.center_frequency_ghz + half_bw) > band_max as f64
        {
            return Err(SubThzConfigError::InvalidConfig(format!(
                "Bandwidth extends beyond band limits ({band_min}-{band_max} GHz)"
            )));
        }

        // Validate BWP configs
        if self.bwp_configs.is_empty() {
            return Err(SubThzConfigError::MissingMandatoryField(
                "bwp_configs (at least one BWP required)".to_string(),
            ));
        }
        for bwp in &self.bwp_configs {
            if bwp.bwp_id > 3 {
                return Err(SubThzConfigError::InvalidConfig(
                    "BWP ID must be 0-3".to_string(),
                ));
            }
            if bwp.bandwidth_ghz <= 0.0 {
                return Err(SubThzConfigError::InvalidConfig(
                    "BWP bandwidth must be > 0".to_string(),
                ));
            }
        }

        // Validate beam config
        if self.beam_config.beam_width_deg <= 0.0 || self.beam_config.beam_width_deg > 90.0 {
            return Err(SubThzConfigError::InvalidConfig(
                "Beam width must be in range (0, 90] degrees".to_string(),
            ));
        }
        if self.beam_config.max_misalignment_deg < 0.0 {
            return Err(SubThzConfigError::InvalidConfig(
                "Max beam misalignment must be >= 0".to_string(),
            ));
        }
        if self.beam_config.max_misalignment_deg >= self.beam_config.beam_width_deg {
            return Err(SubThzConfigError::InvalidConfig(
                "Max misalignment must be < beam width".to_string(),
            ));
        }

        // Validate atmospheric config
        if self.atmospheric_config.path_loss_exponent < 1.0 {
            return Err(SubThzConfigError::InvalidConfig(
                "Path loss exponent must be >= 1.0".to_string(),
            ));
        }
        if self.atmospheric_config.max_link_distance_m <= 0.0 {
            return Err(SubThzConfigError::InvalidConfig(
                "Maximum link distance must be > 0".to_string(),
            ));
        }
        if self.atmospheric_config.humidity_factor < 0.0 || self.atmospheric_config.humidity_factor > 1.0 {
            return Err(SubThzConfigError::InvalidConfig(
                "Humidity factor must be in range [0.0, 1.0]".to_string(),
            ));
        }

        // Validate power config
        if self.power_config.max_tx_power_dbm < self.power_config.min_tx_power_dbm {
            return Err(SubThzConfigError::InvalidConfig(
                "Max TX power must be >= min TX power".to_string(),
            ));
        }
        if self.power_config.pa_efficiency <= 0.0 || self.power_config.pa_efficiency > 1.0 {
            return Err(SubThzConfigError::InvalidConfig(
                "PA efficiency must be in range (0.0, 1.0]".to_string(),
            ));
        }
        if self.power_config.power_backoff_db < 0.0 {
            return Err(SubThzConfigError::InvalidConfig(
                "Power back-off must be >= 0 dB".to_string(),
            ));
        }

        // Validate supported modulations
        if self.supported_modulations.is_empty() {
            return Err(SubThzConfigError::MissingMandatoryField(
                "supported_modulations".to_string(),
            ));
        }

        // Validate RIS config consistency
        if self.ris_enabled && self.ris_element_count.is_none() {
            return Err(SubThzConfigError::InvalidConfig(
                "RIS element count required when RIS is enabled".to_string(),
            ));
        }

        // Validate beam failure config if present
        if let Some(ref bf) = self.beam_config.beam_failure_config {
            if bf.failure_instance_count == 0 {
                return Err(SubThzConfigError::InvalidConfig(
                    "Beam failure instance count must be > 0".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Calculate the approximate number of subcarriers for a given BWP
    pub fn approximate_subcarriers(&self, bwp_index: usize) -> Option<u32> {
        self.bwp_configs.get(bwp_index).map(|bwp| {
            let bw_hz = bwp.bandwidth_ghz * 1e9;
            let scs_hz = bwp.subcarrier_spacing.to_khz() as f64 * 1e3;
            (bw_hz / scs_hz) as u32
        })
    }
}

/// Encode sub-THz band config to bytes (simplified serialization)
pub fn encode_sub_thz_config(config: &SubThzBandConfig) -> Result<Vec<u8>, SubThzConfigError> {
    config.validate()?;
    let mut bytes = Vec::with_capacity(32);

    // config_id (2 bytes)
    bytes.extend_from_slice(&config.config_id.to_be_bytes());
    // band type (1 byte)
    bytes.push(match config.band {
        SubThzBand::DBand => 0,
        SubThzBand::HBand => 1,
        SubThzBand::Custom { .. } => 2,
    });
    // center_frequency_ghz as u16 (2 bytes)
    bytes.extend_from_slice(&(config.center_frequency_ghz as u16).to_be_bytes());
    // total_bandwidth_ghz as u16 in units of 0.1 GHz (2 bytes)
    bytes.extend_from_slice(&((config.total_bandwidth_ghz * 10.0) as u16).to_be_bytes());
    // channel_model (1 byte)
    bytes.push(match config.channel_model {
        SubThzChannelModel::LineOfSight => 0,
        SubThzChannelModel::NonLineOfSight => 1,
        SubThzChannelModel::RisAssisted => 2,
        SubThzChannelModel::IndoorShortRange => 3,
        SubThzChannelModel::OutdoorUrban => 4,
        SubThzChannelModel::BackhaulFronthaul => 5,
    });
    // num BWPs (1 byte)
    bytes.push(config.bwp_configs.len() as u8);
    // flags (1 byte): bit 0 = ris_enabled, bit 1 = molecular_absorption_aware,
    //                 bit 2 = ai_beam_prediction, bit 3 = multi_beam_enabled
    let mut flags: u8 = 0;
    if config.ris_enabled {
        flags |= 0x01;
    }
    if config.molecular_absorption_aware {
        flags |= 0x02;
    }
    if config.beam_config.ai_beam_prediction {
        flags |= 0x04;
    }
    if config.beam_config.multi_beam_enabled {
        flags |= 0x08;
    }
    bytes.push(flags);

    Ok(bytes)
}

/// Decode sub-THz band config header from bytes (simplified deserialization)
pub fn decode_sub_thz_config_header(
    bytes: &[u8],
) -> Result<(u16, SubThzBand, u16, SubThzChannelModel), SubThzConfigError> {
    if bytes.len() < 8 {
        return Err(SubThzConfigError::CodecError(
            "Insufficient bytes for sub-THz config header".to_string(),
        ));
    }

    let config_id = u16::from_be_bytes(bytes[0..2].try_into().unwrap());
    let band = match bytes[2] {
        0 => SubThzBand::DBand,
        1 => SubThzBand::HBand,
        2 => SubThzBand::Custom { min_ghz: 100, max_ghz: 300 },
        _ => return Err(SubThzConfigError::CodecError("Unknown band type".to_string())),
    };
    let center_freq_ghz = u16::from_be_bytes(bytes[3..5].try_into().unwrap());
    let channel_model = match bytes[7] {
        0 => SubThzChannelModel::LineOfSight,
        1 => SubThzChannelModel::NonLineOfSight,
        2 => SubThzChannelModel::RisAssisted,
        3 => SubThzChannelModel::IndoorShortRange,
        4 => SubThzChannelModel::OutdoorUrban,
        5 => SubThzChannelModel::BackhaulFronthaul,
        _ => return Err(SubThzConfigError::CodecError("Unknown channel model".to_string())),
    };

    Ok((config_id, band, center_freq_ghz, channel_model))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_bwp() -> SubThzBwpConfig {
        SubThzBwpConfig {
            bwp_id: 0,
            center_frequency_ghz: 140.0,
            bandwidth_ghz: 2.0,
            subcarrier_spacing: SubThzSubcarrierSpacing::Khz960,
            num_resource_blocks: 275,
            is_active: true,
            cyclic_prefix: CyclicPrefixType::Normal,
        }
    }

    fn create_test_beam_config() -> SubThzBeamConfig {
        SubThzBeamConfig {
            max_beams: 8,
            beam_width_deg: 2.0,
            beam_sweep_periodicity_ms: 20,
            num_ssb_beams: 64,
            beam_tracking_enabled: true,
            beam_tracking_interval_ms: Some(5),
            max_misalignment_deg: 0.5,
            multi_beam_enabled: true,
            beam_failure_config: Some(BeamFailureConfig {
                failure_instance_count: 4,
                detection_timer_ms: 100,
                max_candidate_beams: 8,
                recovery_timer_ms: 200,
                rsrp_threshold_dbm: -120,
            }),
            ai_beam_prediction: true,
        }
    }

    fn create_test_config() -> SubThzBandConfig {
        SubThzBandConfig {
            config_id: 1,
            band: SubThzBand::DBand,
            center_frequency_ghz: 140.0,
            total_bandwidth_ghz: 10.0,
            bwp_configs: vec![create_test_bwp()],
            beam_config: create_test_beam_config(),
            channel_model: SubThzChannelModel::LineOfSight,
            atmospheric_config: AtmosphericConfig {
                path_loss_exponent: 2.0,
                atmospheric_absorption_db_per_km: 1.5,
                rain_attenuation_db_per_km: Some(5.0),
                humidity_factor: 0.5,
                max_link_distance_m: 200.0,
                weather_adaptive: true,
            },
            power_config: SubThzPowerConfig {
                max_tx_power_dbm: 20.0,
                min_tx_power_dbm: -10.0,
                pa_efficiency: 0.15,
                power_boosting_enabled: true,
                power_backoff_db: 3.0,
            },
            supported_modulations: vec![
                SubThzModulation::Qpsk,
                SubThzModulation::Qam16,
                SubThzModulation::Qam64,
                SubThzModulation::Qam256,
            ],
            max_mcs_index: 28,
            ris_enabled: false,
            ris_element_count: None,
            molecular_absorption_aware: true,
        }
    }

    #[test]
    fn test_sub_thz_config_validate() {
        let config = create_test_config();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_sub_thz_config_invalid_frequency() {
        let mut config = create_test_config();
        config.center_frequency_ghz = 50.0;
        assert!(config.validate().is_err());

        config.center_frequency_ghz = 350.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_sub_thz_config_invalid_bandwidth() {
        let mut config = create_test_config();
        config.total_bandwidth_ghz = 0.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_sub_thz_config_bandwidth_exceeds_band() {
        let mut config = create_test_config();
        config.total_bandwidth_ghz = 100.0; // exceeds D-band range
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_sub_thz_config_empty_bwp() {
        let mut config = create_test_config();
        config.bwp_configs = vec![];
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_sub_thz_config_invalid_bwp_id() {
        let mut config = create_test_config();
        config.bwp_configs[0].bwp_id = 4;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_sub_thz_config_invalid_beam_width() {
        let mut config = create_test_config();
        config.beam_config.beam_width_deg = 0.0;
        assert!(config.validate().is_err());

        config.beam_config.beam_width_deg = 91.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_sub_thz_config_misalignment_exceeds_beam_width() {
        let mut config = create_test_config();
        config.beam_config.max_misalignment_deg = 2.0; // == beam_width
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_sub_thz_config_invalid_path_loss() {
        let mut config = create_test_config();
        config.atmospheric_config.path_loss_exponent = 0.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_sub_thz_config_invalid_link_distance() {
        let mut config = create_test_config();
        config.atmospheric_config.max_link_distance_m = 0.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_sub_thz_config_invalid_humidity() {
        let mut config = create_test_config();
        config.atmospheric_config.humidity_factor = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_sub_thz_config_invalid_power() {
        let mut config = create_test_config();
        config.power_config.max_tx_power_dbm = -20.0; // less than min
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_sub_thz_config_invalid_pa_efficiency() {
        let mut config = create_test_config();
        config.power_config.pa_efficiency = 0.0;
        assert!(config.validate().is_err());

        config.power_config.pa_efficiency = 1.1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_sub_thz_config_invalid_power_backoff() {
        let mut config = create_test_config();
        config.power_config.power_backoff_db = -1.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_sub_thz_config_empty_modulations() {
        let mut config = create_test_config();
        config.supported_modulations = vec![];
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_sub_thz_config_ris_without_count() {
        let mut config = create_test_config();
        config.ris_enabled = true;
        config.ris_element_count = None;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_sub_thz_config_ris_with_count() {
        let mut config = create_test_config();
        config.ris_enabled = true;
        config.ris_element_count = Some(1024);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_sub_thz_config_beam_failure_invalid() {
        let mut config = create_test_config();
        if let Some(ref mut bf) = config.beam_config.beam_failure_config {
            bf.failure_instance_count = 0;
        }
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_subcarrier_spacing_values() {
        assert_eq!(SubThzSubcarrierSpacing::Khz480.to_khz(), 480);
        assert_eq!(SubThzSubcarrierSpacing::Khz960.to_khz(), 960);
        assert_eq!(SubThzSubcarrierSpacing::Khz1920.to_khz(), 1920);
        assert_eq!(SubThzSubcarrierSpacing::Khz3840.to_khz(), 3840);
    }

    #[test]
    fn test_subcarrier_spacing_numerology() {
        assert_eq!(SubThzSubcarrierSpacing::Khz480.numerology_index(), 5);
        assert_eq!(SubThzSubcarrierSpacing::Khz960.numerology_index(), 6);
        assert_eq!(SubThzSubcarrierSpacing::Khz1920.numerology_index(), 7);
        assert_eq!(SubThzSubcarrierSpacing::Khz3840.numerology_index(), 8);
    }

    #[test]
    fn test_band_frequency_ranges() {
        let (min, max) = SubThzBand::DBand.frequency_range_ghz();
        assert_eq!(min, 110);
        assert_eq!(max, 170);

        let (min, max) = SubThzBand::HBand.frequency_range_ghz();
        assert_eq!(min, 220);
        assert_eq!(max, 330);

        let custom = SubThzBand::Custom { min_ghz: 150, max_ghz: 200 };
        let (min, max) = custom.frequency_range_ghz();
        assert_eq!(min, 150);
        assert_eq!(max, 200);
    }

    #[test]
    fn test_approximate_subcarriers() {
        let config = create_test_config();
        let subcarriers = config.approximate_subcarriers(0).unwrap();
        // 2.0 GHz / 960 kHz = ~2083 subcarriers
        assert!(subcarriers > 2_000 && subcarriers < 3_000,
            "Expected ~2083 subcarriers, got {subcarriers}");
    }

    #[test]
    fn test_approximate_subcarriers_invalid_index() {
        let config = create_test_config();
        assert!(config.approximate_subcarriers(5).is_none());
    }

    #[test]
    fn test_encode_decode_sub_thz_config() {
        let config = create_test_config();
        let encoded = encode_sub_thz_config(&config).expect("Failed to encode");
        assert!(!encoded.is_empty());

        let (config_id, band, center_freq, channel_model) =
            decode_sub_thz_config_header(&encoded).expect("Failed to decode");
        assert_eq!(config_id, 1);
        assert_eq!(band, SubThzBand::DBand);
        assert_eq!(center_freq, 140);
        assert_eq!(channel_model, SubThzChannelModel::LineOfSight);
    }

    #[test]
    fn test_all_channel_models() {
        let models = [
            SubThzChannelModel::LineOfSight,
            SubThzChannelModel::NonLineOfSight,
            SubThzChannelModel::RisAssisted,
            SubThzChannelModel::IndoorShortRange,
            SubThzChannelModel::OutdoorUrban,
            SubThzChannelModel::BackhaulFronthaul,
        ];
        for model in models {
            let mut config = create_test_config();
            config.channel_model = model;
            let encoded = encode_sub_thz_config(&config).expect("Failed to encode");
            let (_, _, _, decoded_model) =
                decode_sub_thz_config_header(&encoded).expect("Failed to decode");
            assert_eq!(decoded_model, model);
        }
    }

    #[test]
    fn test_all_cyclic_prefix_types() {
        let cp_types = [
            CyclicPrefixType::Normal,
            CyclicPrefixType::Extended,
            CyclicPrefixType::UltraShort,
        ];
        for cp in cp_types {
            let mut bwp = create_test_bwp();
            bwp.cyclic_prefix = cp;
            assert_eq!(bwp.cyclic_prefix, cp);
        }
    }

    #[test]
    fn test_all_modulation_schemes() {
        let mods = [
            SubThzModulation::Bpsk,
            SubThzModulation::Qpsk,
            SubThzModulation::Qam16,
            SubThzModulation::Qam64,
            SubThzModulation::Qam256,
            SubThzModulation::Qam1024,
            SubThzModulation::Ook,
        ];
        let mut config = create_test_config();
        config.supported_modulations = mods.to_vec();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_h_band_config() {
        let mut config = create_test_config();
        config.band = SubThzBand::HBand;
        config.center_frequency_ghz = 275.0;
        config.total_bandwidth_ghz = 20.0;
        config.bwp_configs[0].center_frequency_ghz = 275.0;
        config.bwp_configs[0].subcarrier_spacing = SubThzSubcarrierSpacing::Khz3840;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_multiple_bwps() {
        let mut config = create_test_config();
        config.bwp_configs = vec![
            SubThzBwpConfig {
                bwp_id: 0,
                center_frequency_ghz: 138.0,
                bandwidth_ghz: 2.0,
                subcarrier_spacing: SubThzSubcarrierSpacing::Khz960,
                num_resource_blocks: 275,
                is_active: true,
                cyclic_prefix: CyclicPrefixType::Normal,
            },
            SubThzBwpConfig {
                bwp_id: 1,
                center_frequency_ghz: 142.0,
                bandwidth_ghz: 4.0,
                subcarrier_spacing: SubThzSubcarrierSpacing::Khz1920,
                num_resource_blocks: 550,
                is_active: false,
                cyclic_prefix: CyclicPrefixType::Extended,
            },
        ];
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_indoor_short_range_config() {
        let mut config = create_test_config();
        config.channel_model = SubThzChannelModel::IndoorShortRange;
        config.atmospheric_config.max_link_distance_m = 10.0;
        config.power_config.max_tx_power_dbm = 10.0;
        config.power_config.min_tx_power_dbm = -15.0;
        assert!(config.validate().is_ok());
    }
}
