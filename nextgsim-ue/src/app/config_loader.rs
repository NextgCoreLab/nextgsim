//! Configuration Loading for UE Application
//!
//! This module provides configuration loading and validation for the UE.
//! It wraps the `UeConfig` from `nextgsim-common` with additional validation
//! and error handling specific to the UE application.
//!
//! # Example
//!
//! ```rust,ignore
//! use nextgsim_ue::app::{load_ue_config, validate_ue_config};
//!
//! // Load and validate configuration
//! let config = load_ue_config("config/ue.yaml")?;
//! validate_ue_config(&config)?;
//! ```

use std::path::Path;

use nextgsim_common::config::UeConfig;
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
    /// Invalid HPLMN
    #[error("Invalid HPLMN: {0}")]
    InvalidHplmn(String),

    /// Invalid SUPI
    #[error("Invalid SUPI: {0}")]
    InvalidSupi(String),

    /// Invalid subscriber key
    #[error("Invalid subscriber key: {0}")]
    InvalidKey(String),

    /// Invalid operator key
    #[error("Invalid operator key: {0}")]
    InvalidOp(String),

    /// No gNB configured
    #[error("No gNB configured: at least one gNB address must be specified in gnb_search_list")]
    NoGnbConfigured,

    /// Invalid gNB address
    #[error("Invalid gNB address: {0}")]
    InvalidGnbAddress(String),

    /// Invalid protection scheme
    #[error("Invalid protection scheme: {0}")]
    InvalidProtectionScheme(String),

    /// Invalid session configuration
    #[error("Invalid session configuration: {0}")]
    InvalidSessionConfig(String),
}

/// Loads a UE configuration from a YAML file.
///
/// This function reads the configuration file, parses it, and performs
/// basic validation. For comprehensive validation, call `validate_ue_config`
/// after loading.
///
/// # Arguments
///
/// * `path` - Path to the YAML configuration file
///
/// # Returns
///
/// * `Ok(UeConfig)` - Successfully loaded and parsed configuration
/// * `Err(ConfigError)` - Loading or parsing failed
///
/// # Example
///
/// ```rust,ignore
/// use nextgsim_ue::app::load_ue_config;
///
/// let config = load_ue_config("config/ue.yaml")?;
/// println!("Loaded UE with HPLMN: {}-{}", config.hplmn.mcc, config.hplmn.mnc);
/// ```
pub fn load_ue_config<P: AsRef<Path>>(path: P) -> Result<UeConfig, ConfigError> {
    let path = path.as_ref();

    // Read the file
    let contents = std::fs::read_to_string(path)?;

    // Parse YAML
    let config: UeConfig =
        serde_yaml::from_str(&contents).map_err(|e| ConfigError::ParseError(e.to_string()))?;

    Ok(config)
}

/// Loads a UE configuration from a YAML string.
///
/// # Arguments
///
/// * `yaml` - YAML string containing the configuration
///
/// # Returns
///
/// * `Ok(UeConfig)` - Successfully parsed configuration
/// * `Err(ConfigError)` - Parsing failed
pub fn load_ue_config_from_str(yaml: &str) -> Result<UeConfig, ConfigError> {
    let config: UeConfig =
        serde_yaml::from_str(yaml).map_err(|e| ConfigError::ParseError(e.to_string()))?;
    Ok(config)
}


/// Validates a UE configuration.
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
/// - HPLMN MCC must be 3 digits (001-999)
/// - HPLMN MNC must be 2-3 digits (00-999)
/// - At least one gNB address must be specified
/// - Subscriber key K must not be all zeros
/// - Protection scheme must be valid (0, 1, or 2)
///
/// # Example
///
/// ```rust,ignore
/// use nextgsim_ue::app::{load_ue_config, validate_ue_config};
///
/// let config = load_ue_config("config/ue.yaml")?;
/// validate_ue_config(&config)?;
/// ```
pub fn validate_ue_config(config: &UeConfig) -> Result<(), ConfigValidationError> {
    // Validate HPLMN
    validate_hplmn(&config.hplmn)?;

    // Validate gNB search list
    if config.gnb_search_list.is_empty() {
        return Err(ConfigValidationError::NoGnbConfigured);
    }

    // Validate each gNB address
    for (i, addr) in config.gnb_search_list.iter().enumerate() {
        if addr.is_empty() {
            return Err(ConfigValidationError::InvalidGnbAddress(format!(
                "gNB address {i} is empty"
            )));
        }
        // Try to parse as IP address or hostname
        if addr.parse::<std::net::IpAddr>().is_err() && !is_valid_hostname(addr) {
            return Err(ConfigValidationError::InvalidGnbAddress(format!(
                "gNB address '{addr}' is not a valid IP address or hostname"
            )));
        }
    }

    // Validate subscriber key (must not be all zeros)
    if config.key.iter().all(|&b| b == 0) {
        return Err(ConfigValidationError::InvalidKey(
            "Subscriber key K cannot be all zeros".to_string(),
        ));
    }

    // Validate protection scheme (0: null, 1: Profile A, 2: Profile B)
    if config.protection_scheme > 2 {
        return Err(ConfigValidationError::InvalidProtectionScheme(format!(
            "Protection scheme {} is invalid (must be 0, 1, or 2)",
            config.protection_scheme
        )));
    }

    // Validate SUPI format if present
    if let Some(ref supi) = config.supi {
        let supi_str = supi.to_string();
        // SUPI should be in format "imsi-XXXXXXXXXXXXXXX" (15 digits)
        if let Some(digits) = supi_str.strip_prefix("imsi-") {
            if digits.len() != 15 || !digits.chars().all(|c| c.is_ascii_digit()) {
                return Err(ConfigValidationError::InvalidSupi(format!(
                    "IMSI '{digits}' must be exactly 15 digits"
                )));
            }
        }
    }

    Ok(())
}

/// Validates a HPLMN configuration.
fn validate_hplmn(plmn: &nextgsim_common::Plmn) -> Result<(), ConfigValidationError> {
    // MCC must be 3 digits (001-999)
    if plmn.mcc == 0 || plmn.mcc > 999 {
        return Err(ConfigValidationError::InvalidHplmn(format!(
            "MCC {} must be between 001 and 999",
            plmn.mcc
        )));
    }

    // MNC must be 2-3 digits (00-999)
    if plmn.mnc > 999 {
        return Err(ConfigValidationError::InvalidHplmn(format!(
            "MNC {} must be between 00 and 999",
            plmn.mnc
        )));
    }

    Ok(())
}

/// Checks if a string is a valid hostname.
fn is_valid_hostname(hostname: &str) -> bool {
    if hostname.is_empty() || hostname.len() > 253 {
        return false;
    }

    // Split by dots and validate each label
    for label in hostname.split('.') {
        if label.is_empty() || label.len() > 63 {
            return false;
        }
        // Labels must start with alphanumeric
        if !label.chars().next().map(|c| c.is_ascii_alphanumeric()).unwrap_or(false) {
            return false;
        }
        // Labels must end with alphanumeric
        if !label.chars().last().map(|c| c.is_ascii_alphanumeric()).unwrap_or(false) {
            return false;
        }
        // Labels can only contain alphanumeric and hyphens
        if !label.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
            return false;
        }
    }

    true
}

/// Loads and validates a UE configuration in one step.
///
/// This is a convenience function that combines `load_ue_config` and
/// `validate_ue_config`.
///
/// # Arguments
///
/// * `path` - Path to the YAML configuration file
///
/// # Returns
///
/// * `Ok(UeConfig)` - Successfully loaded and validated configuration
/// * `Err(ConfigError)` - Loading, parsing, or validation failed
pub fn load_and_validate_ue_config<P: AsRef<Path>>(path: P) -> Result<UeConfig, ConfigError> {
    let config = load_ue_config(path)?;
    validate_ue_config(&config)?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nextgsim_common::config::{OpType, SupportedAlgs};
    use nextgsim_common::types::{NetworkSlice, Plmn};

    /// Creates a valid test configuration.
    fn valid_config() -> UeConfig {
        UeConfig {
            supi: None,
            protection_scheme: 0,
            home_network_public_key_id: 0,
            home_network_public_key: Vec::new(),
            routing_indicator: None,
            hplmn: Plmn::new(310, 410, false),
            key: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            op: [0u8; 16],
            op_type: OpType::Opc,
            amf: [0x80, 0x00],
            imei: None,
            imei_sv: None,
            supported_algs: SupportedAlgs::default(),
            gnb_search_list: vec!["127.0.0.1".to_string()],
            sessions: Vec::new(),
            configured_nssai: NetworkSlice::new(),
            tun_name: None,
            pqc_config: nextgsim_common::config::PqcConfig::default(),
            redcap: false,
            snpn_config: None,
            prose_enabled: false,
            ursp_rules: vec![],
            pin_role: None,
            ..Default::default()
        }
    }

    #[test]
    fn test_validate_valid_config() {
        let config = valid_config();
        assert!(validate_ue_config(&config).is_ok());
    }

    #[test]
    fn test_validate_invalid_hplmn_mcc_zero() {
        let mut config = valid_config();
        config.hplmn = Plmn::new(0, 410, false);
        let result = validate_ue_config(&config);
        assert!(matches!(
            result,
            Err(ConfigValidationError::InvalidHplmn(_))
        ));
    }

    #[test]
    fn test_validate_invalid_hplmn_mcc_too_large() {
        let mut config = valid_config();
        config.hplmn = Plmn::new(1000, 410, false);
        let result = validate_ue_config(&config);
        assert!(matches!(
            result,
            Err(ConfigValidationError::InvalidHplmn(_))
        ));
    }

    #[test]
    fn test_validate_invalid_hplmn_mnc_too_large() {
        let mut config = valid_config();
        config.hplmn = Plmn::new(310, 1000, false);
        let result = validate_ue_config(&config);
        assert!(matches!(
            result,
            Err(ConfigValidationError::InvalidHplmn(_))
        ));
    }

    #[test]
    fn test_validate_no_gnb_configured() {
        let mut config = valid_config();
        config.gnb_search_list = vec![];
        let result = validate_ue_config(&config);
        assert!(matches!(result, Err(ConfigValidationError::NoGnbConfigured)));
    }

    #[test]
    fn test_validate_empty_gnb_address() {
        let mut config = valid_config();
        config.gnb_search_list = vec!["".to_string()];
        let result = validate_ue_config(&config);
        assert!(matches!(
            result,
            Err(ConfigValidationError::InvalidGnbAddress(_))
        ));
    }

    #[test]
    fn test_validate_invalid_key_all_zeros() {
        let mut config = valid_config();
        config.key = [0u8; 16];
        let result = validate_ue_config(&config);
        assert!(matches!(result, Err(ConfigValidationError::InvalidKey(_))));
    }

    #[test]
    fn test_validate_invalid_protection_scheme() {
        let mut config = valid_config();
        config.protection_scheme = 3;
        let result = validate_ue_config(&config);
        assert!(matches!(
            result,
            Err(ConfigValidationError::InvalidProtectionScheme(_))
        ));
    }

    #[test]
    fn test_validate_valid_protection_schemes() {
        let mut config = valid_config();

        // Test all valid protection schemes
        for scheme in 0..=2 {
            config.protection_scheme = scheme;
            assert!(validate_ue_config(&config).is_ok());
        }
    }

    #[test]
    fn test_validate_valid_gnb_addresses() {
        let mut config = valid_config();

        // Test various valid addresses
        config.gnb_search_list = vec![
            "127.0.0.1".to_string(),
            "192.168.1.1".to_string(),
            "::1".to_string(),
            "gnb.example.com".to_string(),
            "gnb-1.local".to_string(),
        ];
        assert!(validate_ue_config(&config).is_ok());
    }

    #[test]
    fn test_load_config_from_str() {
        let yaml = r#"
protection_scheme: 0
home_network_public_key_id: 0
home_network_public_key: []
hplmn:
  mcc: 310
  mnc: 410
  long_mnc: false
key: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
op: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
op_type: Opc
amf: [128, 0]
supported_algs:
  nia1: true
  nia2: true
  nia3: true
  nea1: true
  nea2: true
  nea3: true
gnb_search_list:
  - 127.0.0.1
sessions: []
configured_nssai:
  slices: []
"#;
        let config = load_ue_config_from_str(yaml).unwrap();
        assert_eq!(config.hplmn.mcc, 310);
        assert_eq!(config.hplmn.mnc, 410);
        assert_eq!(config.gnb_search_list.len(), 1);
    }

    #[test]
    fn test_load_config_from_str_invalid_yaml() {
        let yaml = "invalid: yaml: content: [";
        let result = load_ue_config_from_str(yaml);
        assert!(matches!(result, Err(ConfigError::ParseError(_))));
    }

    #[test]
    fn test_load_config_file_not_found() {
        let result = load_ue_config("/nonexistent/path/config.yaml");
        assert!(matches!(result, Err(ConfigError::IoError(_))));
    }

    #[test]
    fn test_config_error_display() {
        let err = ConfigError::ParseError("test error".to_string());
        assert!(err.to_string().contains("test error"));

        let err = ConfigValidationError::InvalidHplmn("test".to_string());
        assert!(err.to_string().contains("Invalid HPLMN"));
    }

    #[test]
    fn test_is_valid_hostname() {
        assert!(is_valid_hostname("example.com"));
        assert!(is_valid_hostname("gnb-1.local"));
        assert!(is_valid_hostname("a"));
        assert!(is_valid_hostname("gnb"));

        assert!(!is_valid_hostname(""));
        assert!(!is_valid_hostname("-invalid.com"));
        assert!(!is_valid_hostname("invalid-.com"));
        assert!(!is_valid_hostname(".invalid.com"));
    }

    #[test]
    fn test_validate_boundary_values() {
        let mut config = valid_config();

        // Minimum valid MCC
        config.hplmn = Plmn::new(1, 0, false);
        assert!(validate_ue_config(&config).is_ok());

        // Maximum valid MCC
        config.hplmn = Plmn::new(999, 0, false);
        assert!(validate_ue_config(&config).is_ok());

        // Maximum valid MNC
        config.hplmn = Plmn::new(310, 999, false);
        assert!(validate_ue_config(&config).is_ok());
    }
}
