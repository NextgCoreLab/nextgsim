//! RRC Security Mode Procedure
//!
//! Implements the RRC Security Mode procedure as defined in 3GPP TS 38.331 Section 5.3.4.
//! This procedure is used to activate AS security between the UE and the network.
//!
//! The procedure consists of two messages:
//! 1. `SecurityModeCommand` - gNB → UE: Network command to activate security
//! 2. `SecurityModeComplete` - UE → gNB: Confirmation of security activation

use crate::codec::generated::*;
use crate::codec::{decode_rrc, encode_rrc, RrcCodecError};
use thiserror::Error;

/// Errors that can occur during RRC Security Mode procedures
#[derive(Debug, Error)]
pub enum RrcSecurityModeError {
    /// Codec error during encoding/decoding
    #[error("Codec error: {0}")]
    CodecError(#[from] RrcCodecError),

    /// Invalid message type received
    #[error("Invalid message type: expected {expected}, got {actual}")]
    InvalidMessageType {
        /// Expected message type
        expected: String,
        /// Actual message type received
        actual: String,
    },

    /// Invalid field value
    #[error("Invalid field value: {0}")]
    InvalidFieldValue(String),
}

/// Ciphering algorithm for RRC security
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CipheringAlgorithmType {
    /// No encryption (NEA0)
    Nea0,
    /// 128-bit SNOW3G (NEA1)
    Nea1,
    /// 128-bit AES (NEA2)
    Nea2,
    /// 128-bit ZUC (NEA3)
    Nea3,
}

impl From<CipheringAlgorithmType> for CipheringAlgorithm {
    fn from(alg: CipheringAlgorithmType) -> Self {
        let value = match alg {
            CipheringAlgorithmType::Nea0 => CipheringAlgorithm::NEA0,
            CipheringAlgorithmType::Nea1 => CipheringAlgorithm::NEA1,
            CipheringAlgorithmType::Nea2 => CipheringAlgorithm::NEA2,
            CipheringAlgorithmType::Nea3 => CipheringAlgorithm::NEA3,
        };
        CipheringAlgorithm(value)
    }
}

impl TryFrom<CipheringAlgorithm> for CipheringAlgorithmType {
    type Error = RrcSecurityModeError;

    fn try_from(alg: CipheringAlgorithm) -> Result<Self, Self::Error> {
        match alg.0 {
            CipheringAlgorithm::NEA0 => Ok(CipheringAlgorithmType::Nea0),
            CipheringAlgorithm::NEA1 => Ok(CipheringAlgorithmType::Nea1),
            CipheringAlgorithm::NEA2 => Ok(CipheringAlgorithmType::Nea2),
            CipheringAlgorithm::NEA3 => Ok(CipheringAlgorithmType::Nea3),
            _ => Err(RrcSecurityModeError::InvalidFieldValue(format!(
                "Unknown ciphering algorithm: {}",
                alg.0
            ))),
        }
    }
}

/// Integrity protection algorithm for RRC security
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegrityAlgorithmType {
    /// No integrity protection (NIA0)
    Nia0,
    /// 128-bit SNOW3G (NIA1)
    Nia1,
    /// 128-bit AES (NIA2)
    Nia2,
    /// 128-bit ZUC (NIA3)
    Nia3,
}

impl From<IntegrityAlgorithmType> for IntegrityProtAlgorithm {
    fn from(alg: IntegrityAlgorithmType) -> Self {
        let value = match alg {
            IntegrityAlgorithmType::Nia0 => IntegrityProtAlgorithm::NIA0,
            IntegrityAlgorithmType::Nia1 => IntegrityProtAlgorithm::NIA1,
            IntegrityAlgorithmType::Nia2 => IntegrityProtAlgorithm::NIA2,
            IntegrityAlgorithmType::Nia3 => IntegrityProtAlgorithm::NIA3,
        };
        IntegrityProtAlgorithm(value)
    }
}

impl TryFrom<IntegrityProtAlgorithm> for IntegrityAlgorithmType {
    type Error = RrcSecurityModeError;

    fn try_from(alg: IntegrityProtAlgorithm) -> Result<Self, Self::Error> {
        match alg.0 {
            IntegrityProtAlgorithm::NIA0 => Ok(IntegrityAlgorithmType::Nia0),
            IntegrityProtAlgorithm::NIA1 => Ok(IntegrityAlgorithmType::Nia1),
            IntegrityProtAlgorithm::NIA2 => Ok(IntegrityAlgorithmType::Nia2),
            IntegrityProtAlgorithm::NIA3 => Ok(IntegrityAlgorithmType::Nia3),
            _ => Err(RrcSecurityModeError::InvalidFieldValue(format!(
                "Unknown integrity algorithm: {}",
                alg.0
            ))),
        }
    }
}

/// Security algorithm configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SecurityAlgorithms {
    /// Ciphering algorithm
    pub ciphering_algorithm: CipheringAlgorithmType,
    /// Integrity protection algorithm (optional)
    pub integrity_algorithm: Option<IntegrityAlgorithmType>,
}

// ============================================================================
// Security Mode Command
// ============================================================================

/// Parameters for building a Security Mode Command message
#[derive(Debug, Clone)]
pub struct SecurityModeCommandParams {
    /// RRC Transaction Identifier (0-3)
    pub rrc_transaction_id: u8,
    /// Security algorithm configuration
    pub security_algorithms: SecurityAlgorithms,
}

/// Parsed Security Mode Command data
#[derive(Debug, Clone)]
pub struct SecurityModeCommandData {
    /// RRC Transaction Identifier
    pub rrc_transaction_id: u8,
    /// Security algorithm configuration
    pub security_algorithms: SecurityAlgorithms,
}

/// Build a Security Mode Command message
pub fn build_security_mode_command(
    params: &SecurityModeCommandParams,
) -> Result<DL_DCCH_Message, RrcSecurityModeError> {
    if params.rrc_transaction_id > 3 {
        return Err(RrcSecurityModeError::InvalidFieldValue(
            "RRC Transaction ID must be 0-3".to_string(),
        ));
    }

    // Build security algorithm config
    let security_algorithm_config = SecurityAlgorithmConfig {
        ciphering_algorithm: params.security_algorithms.ciphering_algorithm.into(),
        integrity_prot_algorithm: params.security_algorithms.integrity_algorithm.map(std::convert::Into::into),
    };

    // Build security config SMC
    let security_config_smc = SecurityConfigSMC {
        security_algorithm_config,
    };

    let security_mode_command_ies = SecurityModeCommand_IEs {
        security_config_smc,
        late_non_critical_extension: None,
        non_critical_extension: None,
    };

    let security_mode_command = SecurityModeCommand {
        rrc_transaction_identifier: RRC_TransactionIdentifier(params.rrc_transaction_id),
        critical_extensions: SecurityModeCommandCriticalExtensions::SecurityModeCommand(
            security_mode_command_ies,
        ),
    };

    let message_type =
        DL_DCCH_MessageType::C1(DL_DCCH_MessageType_c1::SecurityModeCommand(security_mode_command));

    Ok(DL_DCCH_Message { message: message_type })
}

/// Parse a Security Mode Command from a DL-DCCH message
pub fn parse_security_mode_command(
    msg: &DL_DCCH_Message,
) -> Result<SecurityModeCommandData, RrcSecurityModeError> {
    let security_mode_command = match &msg.message {
        DL_DCCH_MessageType::C1(c1) => match c1 {
            DL_DCCH_MessageType_c1::SecurityModeCommand(cmd) => cmd,
            _ => {
                return Err(RrcSecurityModeError::InvalidMessageType {
                    expected: "SecurityModeCommand".to_string(),
                    actual: "other c1 message".to_string(),
                })
            }
        },
        _ => {
            return Err(RrcSecurityModeError::InvalidMessageType {
                expected: "c1".to_string(),
                actual: "messageClassExtension".to_string(),
            })
        }
    };

    let ies = match &security_mode_command.critical_extensions {
        SecurityModeCommandCriticalExtensions::SecurityModeCommand(ies) => ies,
        SecurityModeCommandCriticalExtensions::CriticalExtensionsFuture(_) => {
            return Err(RrcSecurityModeError::InvalidMessageType {
                expected: "securityModeCommand".to_string(),
                actual: "criticalExtensionsFuture".to_string(),
            })
        }
    };

    // Parse security algorithms
    let ciphering_algorithm =
        CipheringAlgorithmType::try_from(ies.security_config_smc.security_algorithm_config.ciphering_algorithm.clone())?;

    let integrity_algorithm = ies
        .security_config_smc
        .security_algorithm_config
        .integrity_prot_algorithm
        .as_ref()
        .map(|a| IntegrityAlgorithmType::try_from(a.clone()))
        .transpose()?;

    Ok(SecurityModeCommandData {
        rrc_transaction_id: security_mode_command.rrc_transaction_identifier.0,
        security_algorithms: SecurityAlgorithms {
            ciphering_algorithm,
            integrity_algorithm,
        },
    })
}

// ============================================================================
// Security Mode Complete
// ============================================================================

/// Parameters for building a Security Mode Complete message
#[derive(Debug, Clone)]
pub struct SecurityModeCompleteParams {
    /// RRC Transaction Identifier (0-3)
    pub rrc_transaction_id: u8,
}

/// Parsed Security Mode Complete data
#[derive(Debug, Clone)]
pub struct SecurityModeCompleteData {
    /// RRC Transaction Identifier
    pub rrc_transaction_id: u8,
}

/// Build a Security Mode Complete message
pub fn build_security_mode_complete(
    params: &SecurityModeCompleteParams,
) -> Result<UL_DCCH_Message, RrcSecurityModeError> {
    if params.rrc_transaction_id > 3 {
        return Err(RrcSecurityModeError::InvalidFieldValue(
            "RRC Transaction ID must be 0-3".to_string(),
        ));
    }

    let security_mode_complete_ies = SecurityModeComplete_IEs {
        late_non_critical_extension: None,
        non_critical_extension: None,
    };

    let security_mode_complete = SecurityModeComplete {
        rrc_transaction_identifier: RRC_TransactionIdentifier(params.rrc_transaction_id),
        critical_extensions: SecurityModeCompleteCriticalExtensions::SecurityModeComplete(
            security_mode_complete_ies,
        ),
    };

    let message_type = UL_DCCH_MessageType::C1(UL_DCCH_MessageType_c1::SecurityModeComplete(
        security_mode_complete,
    ));

    Ok(UL_DCCH_Message { message: message_type })
}

/// Parse a Security Mode Complete from a UL-DCCH message
pub fn parse_security_mode_complete(
    msg: &UL_DCCH_Message,
) -> Result<SecurityModeCompleteData, RrcSecurityModeError> {
    let security_mode_complete = match &msg.message {
        UL_DCCH_MessageType::C1(c1) => match c1 {
            UL_DCCH_MessageType_c1::SecurityModeComplete(complete) => complete,
            _ => {
                return Err(RrcSecurityModeError::InvalidMessageType {
                    expected: "SecurityModeComplete".to_string(),
                    actual: "other c1 message".to_string(),
                })
            }
        },
        _ => {
            return Err(RrcSecurityModeError::InvalidMessageType {
                expected: "c1".to_string(),
                actual: "messageClassExtension".to_string(),
            })
        }
    };

    // Verify we have the expected critical extensions variant
    match &security_mode_complete.critical_extensions {
        SecurityModeCompleteCriticalExtensions::SecurityModeComplete(_) => {}
        SecurityModeCompleteCriticalExtensions::CriticalExtensionsFuture(_) => {
            return Err(RrcSecurityModeError::InvalidMessageType {
                expected: "securityModeComplete".to_string(),
                actual: "criticalExtensionsFuture".to_string(),
            })
        }
    };

    Ok(SecurityModeCompleteData {
        rrc_transaction_id: security_mode_complete.rrc_transaction_identifier.0,
    })
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Build and encode a Security Mode Command to bytes
pub fn encode_security_mode_command(
    params: &SecurityModeCommandParams,
) -> Result<Vec<u8>, RrcSecurityModeError> {
    let msg = build_security_mode_command(params)?;
    Ok(encode_rrc(&msg)?)
}

/// Decode and parse a Security Mode Command from bytes
pub fn decode_security_mode_command(bytes: &[u8]) -> Result<SecurityModeCommandData, RrcSecurityModeError> {
    let msg: DL_DCCH_Message = decode_rrc(bytes)?;
    parse_security_mode_command(&msg)
}

/// Build and encode a Security Mode Complete to bytes
pub fn encode_security_mode_complete(
    params: &SecurityModeCompleteParams,
) -> Result<Vec<u8>, RrcSecurityModeError> {
    let msg = build_security_mode_complete(params)?;
    Ok(encode_rrc(&msg)?)
}

/// Decode and parse a Security Mode Complete from bytes
pub fn decode_security_mode_complete(
    bytes: &[u8],
) -> Result<SecurityModeCompleteData, RrcSecurityModeError> {
    let msg: UL_DCCH_Message = decode_rrc(bytes)?;
    parse_security_mode_complete(&msg)
}

/// Check if a DL-DCCH message is a Security Mode Command
pub fn is_security_mode_command(msg: &DL_DCCH_Message) -> bool {
    matches!(
        &msg.message,
        DL_DCCH_MessageType::C1(DL_DCCH_MessageType_c1::SecurityModeCommand(_))
    )
}

/// Check if a UL-DCCH message is a Security Mode Complete
pub fn is_security_mode_complete(msg: &UL_DCCH_Message) -> bool {
    matches!(
        &msg.message,
        UL_DCCH_MessageType::C1(UL_DCCH_MessageType_c1::SecurityModeComplete(_))
    )
}


#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Security Mode Command Tests
    // ========================================================================

    fn create_test_security_mode_command_params() -> SecurityModeCommandParams {
        SecurityModeCommandParams {
            rrc_transaction_id: 0,
            security_algorithms: SecurityAlgorithms {
                ciphering_algorithm: CipheringAlgorithmType::Nea2,
                integrity_algorithm: Some(IntegrityAlgorithmType::Nia2),
            },
        }
    }

    #[test]
    fn test_build_security_mode_command() {
        let params = create_test_security_mode_command_params();
        let result = build_security_mode_command(&params);
        assert!(result.is_ok());

        let msg = result.unwrap();
        assert!(is_security_mode_command(&msg));
    }

    #[test]
    fn test_parse_security_mode_command() {
        let params = create_test_security_mode_command_params();
        let msg = build_security_mode_command(&params).unwrap();
        let result = parse_security_mode_command(&msg);
        assert!(result.is_ok());

        let data = result.unwrap();
        assert_eq!(data.rrc_transaction_id, params.rrc_transaction_id);
        assert_eq!(
            data.security_algorithms.ciphering_algorithm,
            params.security_algorithms.ciphering_algorithm
        );
        assert_eq!(
            data.security_algorithms.integrity_algorithm,
            params.security_algorithms.integrity_algorithm
        );
    }

    #[test]
    fn test_security_mode_command_without_integrity() {
        let params = SecurityModeCommandParams {
            rrc_transaction_id: 1,
            security_algorithms: SecurityAlgorithms {
                ciphering_algorithm: CipheringAlgorithmType::Nea0,
                integrity_algorithm: None,
            },
        };

        let msg = build_security_mode_command(&params).unwrap();
        let data = parse_security_mode_command(&msg).unwrap();

        assert_eq!(data.rrc_transaction_id, 1);
        assert_eq!(data.security_algorithms.ciphering_algorithm, CipheringAlgorithmType::Nea0);
        assert_eq!(data.security_algorithms.integrity_algorithm, None);
    }

    #[test]
    fn test_encode_decode_security_mode_command() {
        let params = create_test_security_mode_command_params();
        let encoded = encode_security_mode_command(&params);
        assert!(encoded.is_ok());

        let bytes = encoded.unwrap();
        assert!(!bytes.is_empty());

        let decoded = decode_security_mode_command(&bytes);
        assert!(decoded.is_ok());

        let data = decoded.unwrap();
        assert_eq!(data.rrc_transaction_id, params.rrc_transaction_id);
        assert_eq!(
            data.security_algorithms.ciphering_algorithm,
            params.security_algorithms.ciphering_algorithm
        );
    }

    #[test]
    fn test_invalid_rrc_transaction_id_command() {
        let params = SecurityModeCommandParams {
            rrc_transaction_id: 5, // Invalid: must be 0-3
            security_algorithms: SecurityAlgorithms {
                ciphering_algorithm: CipheringAlgorithmType::Nea2,
                integrity_algorithm: Some(IntegrityAlgorithmType::Nia2),
            },
        };

        let result = build_security_mode_command(&params);
        assert!(result.is_err());
    }

    #[test]
    fn test_all_ciphering_algorithms() {
        let algorithms = [
            CipheringAlgorithmType::Nea0,
            CipheringAlgorithmType::Nea1,
            CipheringAlgorithmType::Nea2,
            CipheringAlgorithmType::Nea3,
        ];

        for alg in algorithms {
            let params = SecurityModeCommandParams {
                rrc_transaction_id: 0,
                security_algorithms: SecurityAlgorithms {
                    ciphering_algorithm: alg,
                    integrity_algorithm: None,
                },
            };

            let msg = build_security_mode_command(&params).unwrap();
            let data = parse_security_mode_command(&msg).unwrap();
            assert_eq!(data.security_algorithms.ciphering_algorithm, alg);
        }
    }

    #[test]
    fn test_all_integrity_algorithms() {
        let algorithms = [
            IntegrityAlgorithmType::Nia0,
            IntegrityAlgorithmType::Nia1,
            IntegrityAlgorithmType::Nia2,
            IntegrityAlgorithmType::Nia3,
        ];

        for alg in algorithms {
            let params = SecurityModeCommandParams {
                rrc_transaction_id: 0,
                security_algorithms: SecurityAlgorithms {
                    ciphering_algorithm: CipheringAlgorithmType::Nea0,
                    integrity_algorithm: Some(alg),
                },
            };

            let msg = build_security_mode_command(&params).unwrap();
            let data = parse_security_mode_command(&msg).unwrap();
            assert_eq!(data.security_algorithms.integrity_algorithm, Some(alg));
        }
    }

    // ========================================================================
    // Security Mode Complete Tests
    // ========================================================================

    fn create_test_security_mode_complete_params() -> SecurityModeCompleteParams {
        SecurityModeCompleteParams {
            rrc_transaction_id: 0,
        }
    }

    #[test]
    fn test_build_security_mode_complete() {
        let params = create_test_security_mode_complete_params();
        let result = build_security_mode_complete(&params);
        assert!(result.is_ok());

        let msg = result.unwrap();
        assert!(is_security_mode_complete(&msg));
    }

    #[test]
    fn test_parse_security_mode_complete() {
        let params = create_test_security_mode_complete_params();
        let msg = build_security_mode_complete(&params).unwrap();
        let result = parse_security_mode_complete(&msg);
        assert!(result.is_ok());

        let data = result.unwrap();
        assert_eq!(data.rrc_transaction_id, params.rrc_transaction_id);
    }

    #[test]
    fn test_encode_decode_security_mode_complete() {
        let params = create_test_security_mode_complete_params();
        let encoded = encode_security_mode_complete(&params);
        assert!(encoded.is_ok());

        let bytes = encoded.unwrap();
        assert!(!bytes.is_empty());

        let decoded = decode_security_mode_complete(&bytes);
        assert!(decoded.is_ok());

        let data = decoded.unwrap();
        assert_eq!(data.rrc_transaction_id, params.rrc_transaction_id);
    }

    #[test]
    fn test_invalid_rrc_transaction_id_complete() {
        let params = SecurityModeCompleteParams {
            rrc_transaction_id: 4, // Invalid: must be 0-3
        };

        let result = build_security_mode_complete(&params);
        assert!(result.is_err());
    }

    #[test]
    fn test_security_mode_complete_all_transaction_ids() {
        // Test all valid transaction IDs (0-3)
        for id in 0..=3 {
            let params = SecurityModeCompleteParams {
                rrc_transaction_id: id,
            };
            let msg = build_security_mode_complete(&params).unwrap();
            let data = parse_security_mode_complete(&msg).unwrap();
            assert_eq!(data.rrc_transaction_id, id);
        }
    }
}
