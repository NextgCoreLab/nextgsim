//! Configuration Loading for gNB Application
//!
//! This module provides configuration loading and validation for the gNB.
//! It wraps the `GnbConfig` from `nextgsim-common` with additional validation
//! and error handling specific to the gNB application.
//!
//! # Example
//!
//! ```rust,ignore
//! use nextgsim_gnb::app::{load_gnb_config, validate_gnb_config};
//!
//! // Load and validate configuration
//! let config = load_gnb_config("config/gnb.yaml")?;
//! validate_gnb_config(&config)?;
//! ```

use std::path::Path;

use nextgsim_common::config::GnbConfig;
use thiserror::Error;

/// Errors that can occur during configuration loading.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// File I/O error
    #[error("Failed to read configuration file: {0}")]
    IoError(#[from] std::io::Error),

    /// YAML parsing error
    #[error("Failed to parse configuration: {0}")]
    ParseError(String),

    /// Configuration validation error
    #[error("Configuration validation failed: {0}")]
    ValidationError(#[from] ConfigValidationError),
}

/// Errors that can occur during configuration validation.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ConfigValidationError {
    /// Invalid NCI value
    #[error("Invalid NCI: {0}")]
    InvalidNci(String),

    /// Invalid gNB ID length
    #[error("Invalid gNB ID length: {0}")]
    InvalidGnbIdLength(String),

    /// Invalid TAC value
    #[error("Invalid TAC: {0}")]
    InvalidTac(String),

    /// Invalid PLMN
    #[error("Invalid PLMN: {0}")]
    InvalidPlmn(String),

    /// No AMF configured
    #[error("No AMF configured: at least one AMF must be specified")]
    NoAmfConfigured,

    /// Invalid AMF configuration
    #[error("Invalid AMF configuration: {0}")]
    InvalidAmfConfig(String),

    /// Invalid IP address configuration
    #[error("Invalid IP configuration: {0}")]
    InvalidIpConfig(String),
}

/// Loads a gNB configuration from a YAML file.
///
/// This function reads the configuration file, parses it, and performs
/// basic validation. For comprehensive validation, call `validate_gnb_config`
/// after loading.
///
/// # Arguments
///
/// * `path` - Path to the YAML configuration file
///
/// # Returns
///
/// * `Ok(GnbConfig)` - Successfully loaded and parsed configuration
/// * `Err(ConfigError)` - Loading or parsing failed
///
/// # Example
///
/// ```rust,ignore
/// use nextgsim_gnb::app::load_gnb_config;
///
/// let config = load_gnb_config("config/gnb.yaml")?;
/// println!("Loaded gNB with NCI: {}", config.nci);
/// ```
pub fn load_gnb_config<P: AsRef<Path>>(path: P) -> Result<GnbConfig, ConfigError> {
    let path = path.as_ref();

    // Read the file
    let contents = std::fs::read_to_string(path)?;

    // Parse YAML
    let config: GnbConfig =
        serde_yaml::from_str(&contents).map_err(|e| ConfigError::ParseError(e.to_string()))?;

    Ok(config)
}

/// Loads a gNB configuration from a YAML string.
///
/// # Arguments
///
/// * `yaml` - YAML string containing the configuration
///
/// # Returns
///
/// * `Ok(GnbConfig)` - Successfully parsed configuration
/// * `Err(ConfigError)` - Parsing failed
pub fn load_gnb_config_from_str(yaml: &str) -> Result<GnbConfig, ConfigError> {
    let config: GnbConfig =
        serde_yaml::from_str(yaml).map_err(|e| ConfigError::ParseError(e.to_string()))?;
    Ok(config)
}

/// Validates a gNB configuration.
///
/// This function performs comprehensive validation of the configuration,
/// checking that all values are within valid ranges and that required
/// fields are present.
///
/// # Arguments
///
/// * `config` - The configuration to validate
///
/// # Returns
///
/// * `Ok(())` - Configuration is valid
/// * `Err(ConfigValidationError)` - Validation failed
///
/// # Validation Rules
///
/// - NCI must be a valid 36-bit value (0 to 0xFFFFFFFFF)
/// - gNB ID length must be between 22 and 32 bits
/// - TAC must be a valid 24-bit value (1 to 0xFFFFFF, 0 is reserved)
/// - PLMN MCC must be 3 digits (001-999)
/// - PLMN MNC must be 2-3 digits (00-999)
/// - At least one AMF must be configured
/// - AMF port must be non-zero
///
/// # Example
///
/// ```rust,ignore
/// use nextgsim_gnb::app::{load_gnb_config, validate_gnb_config};
///
/// let config = load_gnb_config("config/gnb.yaml")?;
/// validate_gnb_config(&config)?;
/// ```
pub fn validate_gnb_config(config: &GnbConfig) -> Result<(), ConfigValidationError> {
    // Validate NCI (36-bit value)
    const MAX_NCI: u64 = 0xFFFFFFFFF; // 36 bits
    if config.nci > MAX_NCI {
        return Err(ConfigValidationError::InvalidNci(format!(
            "NCI {} exceeds maximum 36-bit value ({})",
            config.nci, MAX_NCI
        )));
    }

    // Validate gNB ID length (22-32 bits per 3GPP TS 38.413)
    if config.gnb_id_length < 22 || config.gnb_id_length > 32 {
        return Err(ConfigValidationError::InvalidGnbIdLength(format!(
            "gNB ID length {} must be between 22 and 32 bits",
            config.gnb_id_length
        )));
    }

    // Validate TAC (24-bit value, 0 is reserved)
    const MAX_TAC: u32 = 0xFFFFFF; // 24 bits
    if config.tac == 0 {
        return Err(ConfigValidationError::InvalidTac(
            "TAC value 0 is reserved and cannot be used".to_string(),
        ));
    }
    if config.tac > MAX_TAC {
        return Err(ConfigValidationError::InvalidTac(format!(
            "TAC {} exceeds maximum 24-bit value ({})",
            config.tac, MAX_TAC
        )));
    }

    // Validate PLMN
    validate_plmn(&config.plmn)?;

    // Validate AMF configurations
    if config.amf_configs.is_empty() {
        return Err(ConfigValidationError::NoAmfConfigured);
    }

    for (i, amf) in config.amf_configs.iter().enumerate() {
        if amf.port == 0 {
            return Err(ConfigValidationError::InvalidAmfConfig(format!(
                "AMF {} has invalid port 0",
                i
            )));
        }
    }

    // Validate IP addresses are not unspecified (0.0.0.0 or ::)
    if config.link_ip.is_unspecified() {
        return Err(ConfigValidationError::InvalidIpConfig(
            "link_ip cannot be unspecified (0.0.0.0 or ::)".to_string(),
        ));
    }
    if config.ngap_ip.is_unspecified() {
        return Err(ConfigValidationError::InvalidIpConfig(
            "ngap_ip cannot be unspecified (0.0.0.0 or ::)".to_string(),
        ));
    }
    if config.gtp_ip.is_unspecified() {
        return Err(ConfigValidationError::InvalidIpConfig(
            "gtp_ip cannot be unspecified (0.0.0.0 or ::)".to_string(),
        ));
    }

    Ok(())
}

/// Validates a PLMN configuration.
fn validate_plmn(plmn: &nextgsim_common::Plmn) -> Result<(), ConfigValidationError> {
    // MCC must be 3 digits (001-999)
    if plmn.mcc == 0 || plmn.mcc > 999 {
        return Err(ConfigValidationError::InvalidPlmn(format!(
            "MCC {} must be between 001 and 999",
            plmn.mcc
        )));
    }

    // MNC must be 2-3 digits (00-999)
    if plmn.mnc > 999 {
        return Err(ConfigValidationError::InvalidPlmn(format!(
            "MNC {} must be between 00 and 999",
            plmn.mnc
        )));
    }

    Ok(())
}

/// Loads and validates a gNB configuration in one step.
///
/// This is a convenience function that combines `load_gnb_config` and
/// `validate_gnb_config`.
///
/// # Arguments
///
/// * `path` - Path to the YAML configuration file
///
/// # Returns
///
/// * `Ok(GnbConfig)` - Successfully loaded and validated configuration
/// * `Err(ConfigError)` - Loading, parsing, or validation failed
pub fn load_and_validate_gnb_config<P: AsRef<Path>>(path: P) -> Result<GnbConfig, ConfigError> {
    let config = load_gnb_config(path)?;
    validate_gnb_config(&config)?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nextgsim_common::config::AmfConfig;
    use nextgsim_common::Plmn;
    use std::net::{IpAddr, Ipv4Addr};

    /// Creates a valid test configuration.
    fn valid_config() -> GnbConfig {
        GnbConfig {
            nci: 0x000000010,
            gnb_id_length: 24,
            plmn: Plmn::new(310, 410, false),
            tac: 1,
            nssai: vec![],
            amf_configs: vec![AmfConfig::new(
                IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
                38412,
            )],
            link_ip: IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
            ngap_ip: IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
            gtp_ip: IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
            gtp_advertise_ip: None,
            ignore_stream_ids: false,
            upf_addr: None,
            upf_port: 2152,
            pqc_config: nextgsim_common::config::PqcConfig::default(),
        }
    }

    #[test]
    fn test_validate_valid_config() {
        let config = valid_config();
        assert!(validate_gnb_config(&config).is_ok());
    }

    #[test]
    fn test_validate_invalid_nci() {
        let mut config = valid_config();
        config.nci = 0x1000000000; // 37 bits - too large
        let result = validate_gnb_config(&config);
        assert!(matches!(result, Err(ConfigValidationError::InvalidNci(_))));
    }

    #[test]
    fn test_validate_invalid_gnb_id_length_too_small() {
        let mut config = valid_config();
        config.gnb_id_length = 21; // Too small
        let result = validate_gnb_config(&config);
        assert!(matches!(
            result,
            Err(ConfigValidationError::InvalidGnbIdLength(_))
        ));
    }

    #[test]
    fn test_validate_invalid_gnb_id_length_too_large() {
        let mut config = valid_config();
        config.gnb_id_length = 33; // Too large
        let result = validate_gnb_config(&config);
        assert!(matches!(
            result,
            Err(ConfigValidationError::InvalidGnbIdLength(_))
        ));
    }

    #[test]
    fn test_validate_invalid_tac_zero() {
        let mut config = valid_config();
        config.tac = 0; // Reserved
        let result = validate_gnb_config(&config);
        assert!(matches!(result, Err(ConfigValidationError::InvalidTac(_))));
    }

    #[test]
    fn test_validate_invalid_tac_too_large() {
        let mut config = valid_config();
        config.tac = 0x1000000; // 25 bits - too large
        let result = validate_gnb_config(&config);
        assert!(matches!(result, Err(ConfigValidationError::InvalidTac(_))));
    }

    #[test]
    fn test_validate_invalid_plmn_mcc_zero() {
        let mut config = valid_config();
        config.plmn = Plmn::new(0, 410, false);
        let result = validate_gnb_config(&config);
        assert!(matches!(result, Err(ConfigValidationError::InvalidPlmn(_))));
    }

    #[test]
    fn test_validate_invalid_plmn_mcc_too_large() {
        let mut config = valid_config();
        config.plmn = Plmn::new(1000, 410, false);
        let result = validate_gnb_config(&config);
        assert!(matches!(result, Err(ConfigValidationError::InvalidPlmn(_))));
    }

    #[test]
    fn test_validate_invalid_plmn_mnc_too_large() {
        let mut config = valid_config();
        config.plmn = Plmn::new(310, 1000, false);
        let result = validate_gnb_config(&config);
        assert!(matches!(result, Err(ConfigValidationError::InvalidPlmn(_))));
    }

    #[test]
    fn test_validate_no_amf_configured() {
        let mut config = valid_config();
        config.amf_configs = vec![];
        let result = validate_gnb_config(&config);
        assert!(matches!(
            result,
            Err(ConfigValidationError::NoAmfConfigured)
        ));
    }

    #[test]
    fn test_validate_invalid_amf_port() {
        let mut config = valid_config();
        config.amf_configs = vec![AmfConfig::new(
            IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
            0, // Invalid port
        )];
        let result = validate_gnb_config(&config);
        assert!(matches!(
            result,
            Err(ConfigValidationError::InvalidAmfConfig(_))
        ));
    }

    #[test]
    fn test_validate_unspecified_link_ip() {
        let mut config = valid_config();
        config.link_ip = IpAddr::V4(Ipv4Addr::UNSPECIFIED);
        let result = validate_gnb_config(&config);
        assert!(matches!(
            result,
            Err(ConfigValidationError::InvalidIpConfig(_))
        ));
    }

    #[test]
    fn test_validate_unspecified_ngap_ip() {
        let mut config = valid_config();
        config.ngap_ip = IpAddr::V4(Ipv4Addr::UNSPECIFIED);
        let result = validate_gnb_config(&config);
        assert!(matches!(
            result,
            Err(ConfigValidationError::InvalidIpConfig(_))
        ));
    }

    #[test]
    fn test_validate_unspecified_gtp_ip() {
        let mut config = valid_config();
        config.gtp_ip = IpAddr::V4(Ipv4Addr::UNSPECIFIED);
        let result = validate_gnb_config(&config);
        assert!(matches!(
            result,
            Err(ConfigValidationError::InvalidIpConfig(_))
        ));
    }

    #[test]
    fn test_load_config_from_str() {
        let yaml = r#"
nci: 16
gnb_id_length: 24
plmn:
  mcc: 310
  mnc: 410
  long_mnc: false
tac: 1
nssai: []
amf_configs:
  - address: 127.0.0.1
    port: 38412
link_ip: 127.0.0.1
ngap_ip: 127.0.0.1
gtp_ip: 127.0.0.1
ignore_stream_ids: false
upf_addr: null
upf_port: 2152
"#;
        let config = load_gnb_config_from_str(yaml).unwrap();
        assert_eq!(config.nci, 16);
        assert_eq!(config.gnb_id_length, 24);
        assert_eq!(config.tac, 1);
        assert_eq!(config.amf_configs.len(), 1);
    }

    #[test]
    fn test_load_config_from_str_invalid_yaml() {
        let yaml = "invalid: yaml: content: [";
        let result = load_gnb_config_from_str(yaml);
        assert!(matches!(result, Err(ConfigError::ParseError(_))));
    }

    #[test]
    fn test_load_config_file_not_found() {
        let result = load_gnb_config("/nonexistent/path/config.yaml");
        assert!(matches!(result, Err(ConfigError::IoError(_))));
    }

    #[test]
    fn test_config_error_display() {
        let err = ConfigError::ParseError("test error".to_string());
        assert!(err.to_string().contains("test error"));

        let err = ConfigValidationError::InvalidNci("test".to_string());
        assert!(err.to_string().contains("Invalid NCI"));
    }

    #[test]
    fn test_validate_boundary_values() {
        // Test valid boundary values
        let mut config = valid_config();

        // Maximum valid NCI
        config.nci = 0xFFFFFFFFF;
        assert!(validate_gnb_config(&config).is_ok());

        // Minimum valid gNB ID length
        config.gnb_id_length = 22;
        assert!(validate_gnb_config(&config).is_ok());

        // Maximum valid gNB ID length
        config.gnb_id_length = 32;
        assert!(validate_gnb_config(&config).is_ok());

        // Maximum valid TAC
        config.tac = 0xFFFFFF;
        assert!(validate_gnb_config(&config).is_ok());

        // Minimum valid TAC
        config.tac = 1;
        assert!(validate_gnb_config(&config).is_ok());
    }
}
