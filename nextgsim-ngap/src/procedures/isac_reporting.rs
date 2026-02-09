//! ISAC (Integrated Sensing and Communication) Reporting Extension
//!
//! 6G extension: NGAP procedure for ISAC measurement reporting.
//! Defines structures for sensing data exchange between gNB and AMF.
//!
//! This module implements:
//! - `IsacMeasurementReport` - Report sensing measurement results
//! - `IsacMeasurementConfig` - Configure sensing measurements

use thiserror::Error;

/// Errors that can occur during ISAC reporting procedures
#[derive(Debug, Error)]
pub enum IsacReportingError {
    /// Invalid configuration
    #[error("Invalid ISAC configuration: {0}")]
    InvalidConfig(String),

    /// Missing mandatory field
    #[error("Missing mandatory field: {0}")]
    MissingMandatoryField(String),
}

/// Sensing measurement type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SensingMeasurementType {
    /// Range-based sensing
    Range,
    /// Velocity-based sensing
    Velocity,
    /// Angle of arrival sensing
    AngleOfArrival,
    /// Angle of departure sensing
    AngleOfDeparture,
    /// Combined range-velocity
    RangeVelocity,
    /// Radar cross-section
    RadarCrossSection,
}

/// Sensing waveform type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SensingWaveformType {
    /// OFDM-based sensing
    Ofdm,
    /// FMCW (Frequency Modulated Continuous Wave)
    Fmcw,
    /// Pulsed radar waveform
    Pulsed,
    /// Joint communication-sensing waveform
    JointComSensing,
}

/// ISAC measurement configuration
#[derive(Debug, Clone)]
pub struct IsacMeasurementConfig {
    /// Configuration ID
    pub config_id: u32,
    /// Sensing measurement type
    pub measurement_type: SensingMeasurementType,
    /// Sensing waveform type
    pub waveform_type: SensingWaveformType,
    /// Sensing periodicity in milliseconds
    pub periodicity_ms: u32,
    /// Center frequency in MHz
    pub center_frequency_mhz: u64,
    /// Bandwidth in MHz
    pub bandwidth_mhz: u32,
    /// Maximum range in meters
    pub max_range_m: Option<u32>,
    /// Maximum velocity in m/s
    pub max_velocity_ms: Option<f64>,
    /// Angular resolution in degrees
    pub angular_resolution_deg: Option<f64>,
    /// Number of antenna elements for sensing
    pub num_antenna_elements: Option<u16>,
    /// Whether the sensing is monostatic or bistatic
    pub is_monostatic: bool,
}

/// ISAC measurement result - single target
#[derive(Debug, Clone)]
pub struct IsacTargetResult {
    /// Target ID
    pub target_id: u32,
    /// Range in meters (if applicable)
    pub range_m: Option<f64>,
    /// Velocity in m/s (if applicable)
    pub velocity_ms: Option<f64>,
    /// Azimuth angle in degrees (if applicable)
    pub azimuth_deg: Option<f64>,
    /// Elevation angle in degrees (if applicable)
    pub elevation_deg: Option<f64>,
    /// Radar cross-section in dBsm (if applicable)
    pub rcs_dbsm: Option<f64>,
    /// Signal-to-noise ratio in dB
    pub snr_db: f64,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
}

/// ISAC measurement report
#[derive(Debug, Clone)]
pub struct IsacMeasurementReport {
    /// Report ID
    pub report_id: u64,
    /// Configuration ID this report corresponds to
    pub config_id: u32,
    /// AMF UE NGAP ID (optional, for UE-specific sensing)
    pub amf_ue_ngap_id: Option<u64>,
    /// RAN UE NGAP ID (optional, for UE-specific sensing)
    pub ran_ue_ngap_id: Option<u32>,
    /// gNB ID performing the sensing
    pub gnb_id: u32,
    /// Timestamp in milliseconds since epoch
    pub timestamp_ms: u64,
    /// Measurement type
    pub measurement_type: SensingMeasurementType,
    /// List of detected targets
    pub targets: Vec<IsacTargetResult>,
    /// Raw sensing data (optional, opaque bytes)
    pub raw_sensing_data: Option<Vec<u8>>,
}

/// ISAC measurement report builder
impl IsacMeasurementReport {
    /// Create a new ISAC measurement report
    pub fn new(
        report_id: u64,
        config_id: u32,
        gnb_id: u32,
        timestamp_ms: u64,
        measurement_type: SensingMeasurementType,
    ) -> Self {
        Self {
            report_id,
            config_id,
            amf_ue_ngap_id: None,
            ran_ue_ngap_id: None,
            gnb_id,
            timestamp_ms,
            measurement_type,
            targets: Vec::new(),
            raw_sensing_data: None,
        }
    }

    /// Add a target result to the report
    pub fn add_target(&mut self, target: IsacTargetResult) {
        self.targets.push(target);
    }

    /// Set UE context for UE-specific sensing
    pub fn set_ue_context(&mut self, amf_ue_ngap_id: u64, ran_ue_ngap_id: u32) {
        self.amf_ue_ngap_id = Some(amf_ue_ngap_id);
        self.ran_ue_ngap_id = Some(ran_ue_ngap_id);
    }

    /// Set raw sensing data
    pub fn set_raw_data(&mut self, data: Vec<u8>) {
        self.raw_sensing_data = Some(data);
    }

    /// Validate the measurement report
    pub fn validate(&self) -> Result<(), IsacReportingError> {
        if self.targets.is_empty() && self.raw_sensing_data.is_none() {
            return Err(IsacReportingError::InvalidConfig(
                "Report must contain at least one target or raw sensing data".to_string(),
            ));
        }

        for target in &self.targets {
            if target.confidence < 0.0 || target.confidence > 1.0 {
                return Err(IsacReportingError::InvalidConfig(format!(
                    "Target {} confidence must be between 0.0 and 1.0, got {}",
                    target.target_id, target.confidence
                )));
            }
        }

        Ok(())
    }
}

impl IsacMeasurementConfig {
    /// Validate the measurement configuration
    pub fn validate(&self) -> Result<(), IsacReportingError> {
        if self.periodicity_ms == 0 {
            return Err(IsacReportingError::InvalidConfig(
                "Sensing periodicity must be > 0".to_string(),
            ));
        }
        if self.bandwidth_mhz == 0 {
            return Err(IsacReportingError::InvalidConfig(
                "Sensing bandwidth must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Encode an ISAC measurement config to bytes (simplified serialization)
pub fn encode_isac_config(config: &IsacMeasurementConfig) -> Result<Vec<u8>, IsacReportingError> {
    config.validate()?;
    // Simplified encoding: config_id (4 bytes) + measurement_type (1 byte) +
    // waveform_type (1 byte) + periodicity (4 bytes) + center_freq (8 bytes) + bandwidth (4 bytes)
    let mut bytes = Vec::with_capacity(22);
    bytes.extend_from_slice(&config.config_id.to_be_bytes());
    bytes.push(config.measurement_type as u8);
    bytes.push(config.waveform_type as u8);
    bytes.extend_from_slice(&config.periodicity_ms.to_be_bytes());
    bytes.extend_from_slice(&config.center_frequency_mhz.to_be_bytes());
    bytes.extend_from_slice(&config.bandwidth_mhz.to_be_bytes());
    bytes.push(if config.is_monostatic { 1 } else { 0 });
    Ok(bytes)
}

/// Decode an ISAC measurement config from bytes (simplified deserialization)
pub fn decode_isac_config(bytes: &[u8]) -> Result<IsacMeasurementConfig, IsacReportingError> {
    if bytes.len() < 23 {
        return Err(IsacReportingError::InvalidConfig(
            "Insufficient bytes for ISAC config".to_string(),
        ));
    }

    let config_id = u32::from_be_bytes(bytes[0..4].try_into().unwrap());
    let measurement_type = match bytes[4] {
        0 => SensingMeasurementType::Range,
        1 => SensingMeasurementType::Velocity,
        2 => SensingMeasurementType::AngleOfArrival,
        3 => SensingMeasurementType::AngleOfDeparture,
        4 => SensingMeasurementType::RangeVelocity,
        5 => SensingMeasurementType::RadarCrossSection,
        _ => {
            return Err(IsacReportingError::InvalidConfig(
                "Unknown measurement type".to_string(),
            ))
        }
    };
    let waveform_type = match bytes[5] {
        0 => SensingWaveformType::Ofdm,
        1 => SensingWaveformType::Fmcw,
        2 => SensingWaveformType::Pulsed,
        3 => SensingWaveformType::JointComSensing,
        _ => {
            return Err(IsacReportingError::InvalidConfig(
                "Unknown waveform type".to_string(),
            ))
        }
    };
    let periodicity_ms = u32::from_be_bytes(bytes[6..10].try_into().unwrap());
    let center_frequency_mhz = u64::from_be_bytes(bytes[10..18].try_into().unwrap());
    let bandwidth_mhz = u32::from_be_bytes(bytes[18..22].try_into().unwrap());
    let is_monostatic = bytes[22] == 1;

    Ok(IsacMeasurementConfig {
        config_id,
        measurement_type,
        waveform_type,
        periodicity_ms,
        center_frequency_mhz,
        bandwidth_mhz,
        max_range_m: None,
        max_velocity_ms: None,
        angular_resolution_deg: None,
        num_antenna_elements: None,
        is_monostatic,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> IsacMeasurementConfig {
        IsacMeasurementConfig {
            config_id: 1,
            measurement_type: SensingMeasurementType::Range,
            waveform_type: SensingWaveformType::Ofdm,
            periodicity_ms: 100,
            center_frequency_mhz: 28000,
            bandwidth_mhz: 400,
            max_range_m: Some(200),
            max_velocity_ms: Some(100.0),
            angular_resolution_deg: Some(1.0),
            num_antenna_elements: Some(64),
            is_monostatic: true,
        }
    }

    #[test]
    fn test_isac_config_validate() {
        let config = create_test_config();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_isac_config_invalid_periodicity() {
        let mut config = create_test_config();
        config.periodicity_ms = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_isac_config_encode_decode() {
        let config = create_test_config();
        let encoded = encode_isac_config(&config).expect("Failed to encode");
        let decoded = decode_isac_config(&encoded).expect("Failed to decode");

        assert_eq!(decoded.config_id, config.config_id);
        assert_eq!(decoded.measurement_type, config.measurement_type);
        assert_eq!(decoded.waveform_type, config.waveform_type);
        assert_eq!(decoded.periodicity_ms, config.periodicity_ms);
        assert_eq!(decoded.center_frequency_mhz, config.center_frequency_mhz);
        assert_eq!(decoded.bandwidth_mhz, config.bandwidth_mhz);
        assert_eq!(decoded.is_monostatic, config.is_monostatic);
    }

    #[test]
    fn test_isac_report_creation() {
        let mut report = IsacMeasurementReport::new(
            1,
            1,
            100,
            1700000000000,
            SensingMeasurementType::Range,
        );

        report.add_target(IsacTargetResult {
            target_id: 1,
            range_m: Some(50.0),
            velocity_ms: None,
            azimuth_deg: Some(45.0),
            elevation_deg: Some(10.0),
            rcs_dbsm: Some(-5.0),
            snr_db: 15.0,
            confidence: 0.95,
        });

        assert!(report.validate().is_ok());
        assert_eq!(report.targets.len(), 1);
    }

    #[test]
    fn test_isac_report_empty_invalid() {
        let report = IsacMeasurementReport::new(
            1,
            1,
            100,
            1700000000000,
            SensingMeasurementType::Range,
        );

        assert!(report.validate().is_err());
    }

    #[test]
    fn test_isac_report_with_ue_context() {
        let mut report = IsacMeasurementReport::new(
            2,
            1,
            100,
            1700000000000,
            SensingMeasurementType::Velocity,
        );
        report.set_ue_context(12345, 67890);
        report.set_raw_data(vec![0x01, 0x02, 0x03]);

        assert!(report.validate().is_ok());
        assert_eq!(report.amf_ue_ngap_id, Some(12345));
        assert_eq!(report.ran_ue_ngap_id, Some(67890));
    }

    #[test]
    fn test_isac_report_invalid_confidence() {
        let mut report = IsacMeasurementReport::new(
            1,
            1,
            100,
            1700000000000,
            SensingMeasurementType::Range,
        );

        report.add_target(IsacTargetResult {
            target_id: 1,
            range_m: Some(50.0),
            velocity_ms: None,
            azimuth_deg: None,
            elevation_deg: None,
            rcs_dbsm: None,
            snr_db: 10.0,
            confidence: 1.5, // Invalid: > 1.0
        });

        assert!(report.validate().is_err());
    }

    #[test]
    fn test_sensing_measurement_types() {
        let types = [
            SensingMeasurementType::Range,
            SensingMeasurementType::Velocity,
            SensingMeasurementType::AngleOfArrival,
            SensingMeasurementType::AngleOfDeparture,
            SensingMeasurementType::RangeVelocity,
            SensingMeasurementType::RadarCrossSection,
        ];

        for t in types {
            let config = IsacMeasurementConfig {
                config_id: 1,
                measurement_type: t,
                waveform_type: SensingWaveformType::Ofdm,
                periodicity_ms: 100,
                center_frequency_mhz: 28000,
                bandwidth_mhz: 400,
                max_range_m: None,
                max_velocity_ms: None,
                angular_resolution_deg: None,
                num_antenna_elements: None,
                is_monostatic: true,
            };
            assert!(config.validate().is_ok());
        }
    }
}
