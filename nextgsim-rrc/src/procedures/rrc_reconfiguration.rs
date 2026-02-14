//! RRC Reconfiguration Procedure
//!
//! Implements the RRC Reconfiguration procedure as defined in 3GPP TS 38.331 Section 5.3.5.
//! This procedure is used to modify an RRC connection, including radio bearer configuration,
//! measurement configuration, and cell group configuration.
//!
//! The procedure consists of two messages:
//! 1. `RRCReconfiguration` - gNB → UE: Network request to modify RRC connection
//! 2. `RRCReconfigurationComplete` - UE → gNB: Confirmation of reconfiguration

use crate::codec::generated::*;
use crate::codec::{decode_rrc, encode_rrc, RrcCodecError};
use thiserror::Error;

/// Errors that can occur during RRC Reconfiguration procedures
#[derive(Debug, Error)]
pub enum RrcReconfigurationError {
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

// ============================================================================
// RRC Reconfiguration
// ============================================================================

/// Parameters for building an RRC Reconfiguration message
#[derive(Debug, Clone)]
pub struct RrcReconfigurationParams {
    /// RRC Transaction Identifier (0-3)
    pub rrc_transaction_id: u8,
    /// Radio Bearer Configuration (encoded as bytes, optional)
    pub radio_bearer_config: Option<Vec<u8>>,
    /// Secondary Cell Group Configuration (encoded as bytes, optional)
    pub secondary_cell_group: Option<Vec<u8>>,
    /// Master Cell Group Configuration (encoded as bytes, optional)
    pub master_cell_group: Option<Vec<u8>>,
    /// Full configuration indicator
    pub full_config: bool,
}

/// Parsed RRC Reconfiguration data
#[derive(Debug, Clone)]
pub struct RrcReconfigurationData {
    /// RRC Transaction Identifier
    pub rrc_transaction_id: u8,
    /// Radio Bearer Configuration (raw bytes)
    pub radio_bearer_config: Option<Vec<u8>>,
    /// Secondary Cell Group Configuration (raw bytes)
    pub secondary_cell_group: Option<Vec<u8>>,
    /// Master Cell Group Configuration (raw bytes)
    pub master_cell_group: Option<Vec<u8>>,
    /// Full configuration indicator
    pub full_config: bool,
}


/// Build an RRC Reconfiguration message
pub fn build_rrc_reconfiguration(
    params: &RrcReconfigurationParams,
) -> Result<DL_DCCH_Message, RrcReconfigurationError> {
    if params.rrc_transaction_id > 3 {
        return Err(RrcReconfigurationError::InvalidFieldValue(
            "RRC Transaction ID must be 0-3".to_string(),
        ));
    }

    // Decode radio bearer config if provided
    let radio_bearer_config = if let Some(ref bytes) = params.radio_bearer_config {
        Some(decode_rrc::<RadioBearerConfig>(bytes)?)
    } else {
        None
    };

    // Build the IEs
    let rrc_reconfiguration_ies = RRCReconfiguration_IEs {
        radio_bearer_config,
        secondary_cell_group: params
            .secondary_cell_group
            .as_ref()
            .map(|b| RRCReconfiguration_IEsSecondaryCellGroup(b.clone())),
        meas_config: None, // Simplified - not including MeasConfig for now
        late_non_critical_extension: None,
        non_critical_extension: if params.master_cell_group.is_some() || params.full_config {
            Some(build_v1530_extension(params))
        } else {
            None
        },
    };

    let rrc_reconfiguration = RRCReconfiguration {
        rrc_transaction_identifier: RRC_TransactionIdentifier(params.rrc_transaction_id),
        critical_extensions: RRCReconfigurationCriticalExtensions::RrcReconfiguration(
            rrc_reconfiguration_ies,
        ),
    };

    let message_type =
        DL_DCCH_MessageType::C1(DL_DCCH_MessageType_c1::RrcReconfiguration(rrc_reconfiguration));

    Ok(DL_DCCH_Message { message: message_type })
}

/// Build the v1530 extension for RRC Reconfiguration
fn build_v1530_extension(params: &RrcReconfigurationParams) -> RRCReconfiguration_v1530_IEs {
    RRCReconfiguration_v1530_IEs {
        master_cell_group: params
            .master_cell_group
            .as_ref()
            .map(|b| RRCReconfiguration_v1530_IEsMasterCellGroup(b.clone())),
        full_config: if params.full_config {
            Some(RRCReconfiguration_v1530_IEsFullConfig(
                RRCReconfiguration_v1530_IEsFullConfig::TRUE,
            ))
        } else {
            None
        },
        dedicated_nas_message_list: None,
        master_key_update: None,
        dedicated_sib1_delivery: None,
        dedicated_system_information_delivery: None,
        other_config: None,
        non_critical_extension: None,
    }
}

/// Parse an RRC Reconfiguration from a DL-DCCH message
pub fn parse_rrc_reconfiguration(
    msg: &DL_DCCH_Message,
) -> Result<RrcReconfigurationData, RrcReconfigurationError> {
    let rrc_reconfiguration = match &msg.message {
        DL_DCCH_MessageType::C1(c1) => match c1 {
            DL_DCCH_MessageType_c1::RrcReconfiguration(reconfig) => reconfig,
            _ => {
                return Err(RrcReconfigurationError::InvalidMessageType {
                    expected: "RRCReconfiguration".to_string(),
                    actual: "other c1 message".to_string(),
                })
            }
        },
        _ => {
            return Err(RrcReconfigurationError::InvalidMessageType {
                expected: "c1".to_string(),
                actual: "messageClassExtension".to_string(),
            })
        }
    };

    let ies = match &rrc_reconfiguration.critical_extensions {
        RRCReconfigurationCriticalExtensions::RrcReconfiguration(ies) => ies,
        RRCReconfigurationCriticalExtensions::CriticalExtensionsFuture(_) => {
            return Err(RrcReconfigurationError::InvalidMessageType {
                expected: "rrcReconfiguration".to_string(),
                actual: "criticalExtensionsFuture".to_string(),
            })
        }
    };

    // Encode radio bearer config back to bytes if present
    let radio_bearer_config = if let Some(ref config) = ies.radio_bearer_config {
        Some(encode_rrc(config)?)
    } else {
        None
    };

    // Extract secondary cell group
    let secondary_cell_group = ies.secondary_cell_group.as_ref().map(|scg| scg.0.clone());

    // Extract master cell group and full_config from v1530 extension
    let (master_cell_group, full_config) = if let Some(ref ext) = ies.non_critical_extension {
        let mcg = ext.master_cell_group.as_ref().map(|m| m.0.clone());
        let fc = ext.full_config.is_some();
        (mcg, fc)
    } else {
        (None, false)
    };

    Ok(RrcReconfigurationData {
        rrc_transaction_id: rrc_reconfiguration.rrc_transaction_identifier.0,
        radio_bearer_config,
        secondary_cell_group,
        master_cell_group,
        full_config,
    })
}


// ============================================================================
// RRC Reconfiguration Complete
// ============================================================================

/// Parameters for building an RRC Reconfiguration Complete message
#[derive(Debug, Clone)]
pub struct RrcReconfigurationCompleteParams {
    /// RRC Transaction Identifier (0-3)
    pub rrc_transaction_id: u8,
}

/// Parsed RRC Reconfiguration Complete data
#[derive(Debug, Clone)]
pub struct RrcReconfigurationCompleteData {
    /// RRC Transaction Identifier
    pub rrc_transaction_id: u8,
}

/// Build an RRC Reconfiguration Complete message
pub fn build_rrc_reconfiguration_complete(
    params: &RrcReconfigurationCompleteParams,
) -> Result<UL_DCCH_Message, RrcReconfigurationError> {
    if params.rrc_transaction_id > 3 {
        return Err(RrcReconfigurationError::InvalidFieldValue(
            "RRC Transaction ID must be 0-3".to_string(),
        ));
    }

    let rrc_reconfiguration_complete_ies = RRCReconfigurationComplete_IEs {
        late_non_critical_extension: None,
        non_critical_extension: None,
    };

    let rrc_reconfiguration_complete = RRCReconfigurationComplete {
        rrc_transaction_identifier: RRC_TransactionIdentifier(params.rrc_transaction_id),
        critical_extensions: RRCReconfigurationCompleteCriticalExtensions::RrcReconfigurationComplete(
            rrc_reconfiguration_complete_ies,
        ),
    };

    let message_type = UL_DCCH_MessageType::C1(UL_DCCH_MessageType_c1::RrcReconfigurationComplete(
        rrc_reconfiguration_complete,
    ));

    Ok(UL_DCCH_Message { message: message_type })
}

/// Parse an RRC Reconfiguration Complete from a UL-DCCH message
pub fn parse_rrc_reconfiguration_complete(
    msg: &UL_DCCH_Message,
) -> Result<RrcReconfigurationCompleteData, RrcReconfigurationError> {
    let rrc_reconfiguration_complete = match &msg.message {
        UL_DCCH_MessageType::C1(c1) => match c1 {
            UL_DCCH_MessageType_c1::RrcReconfigurationComplete(complete) => complete,
            _ => {
                return Err(RrcReconfigurationError::InvalidMessageType {
                    expected: "RRCReconfigurationComplete".to_string(),
                    actual: "other c1 message".to_string(),
                })
            }
        },
        _ => {
            return Err(RrcReconfigurationError::InvalidMessageType {
                expected: "c1".to_string(),
                actual: "messageClassExtension".to_string(),
            })
        }
    };

    // Verify we have the expected critical extensions variant
    match &rrc_reconfiguration_complete.critical_extensions {
        RRCReconfigurationCompleteCriticalExtensions::RrcReconfigurationComplete(_) => {}
        RRCReconfigurationCompleteCriticalExtensions::CriticalExtensionsFuture(_) => {
            return Err(RrcReconfigurationError::InvalidMessageType {
                expected: "rrcReconfigurationComplete".to_string(),
                actual: "criticalExtensionsFuture".to_string(),
            })
        }
    };

    Ok(RrcReconfigurationCompleteData {
        rrc_transaction_id: rrc_reconfiguration_complete.rrc_transaction_identifier.0,
    })
}


// ============================================================================
// Convenience Functions
// ============================================================================

/// Build and encode an RRC Reconfiguration to bytes
pub fn encode_rrc_reconfiguration(
    params: &RrcReconfigurationParams,
) -> Result<Vec<u8>, RrcReconfigurationError> {
    let msg = build_rrc_reconfiguration(params)?;
    Ok(encode_rrc(&msg)?)
}

/// Decode and parse an RRC Reconfiguration from bytes
pub fn decode_rrc_reconfiguration(bytes: &[u8]) -> Result<RrcReconfigurationData, RrcReconfigurationError> {
    let msg: DL_DCCH_Message = decode_rrc(bytes)?;
    parse_rrc_reconfiguration(&msg)
}

/// Build and encode an RRC Reconfiguration Complete to bytes
pub fn encode_rrc_reconfiguration_complete(
    params: &RrcReconfigurationCompleteParams,
) -> Result<Vec<u8>, RrcReconfigurationError> {
    let msg = build_rrc_reconfiguration_complete(params)?;
    Ok(encode_rrc(&msg)?)
}

/// Decode and parse an RRC Reconfiguration Complete from bytes
pub fn decode_rrc_reconfiguration_complete(
    bytes: &[u8],
) -> Result<RrcReconfigurationCompleteData, RrcReconfigurationError> {
    let msg: UL_DCCH_Message = decode_rrc(bytes)?;
    parse_rrc_reconfiguration_complete(&msg)
}

/// Check if a DL-DCCH message is an RRC Reconfiguration
pub fn is_rrc_reconfiguration(msg: &DL_DCCH_Message) -> bool {
    matches!(
        &msg.message,
        DL_DCCH_MessageType::C1(DL_DCCH_MessageType_c1::RrcReconfiguration(_))
    )
}

/// Check if a UL-DCCH message is an RRC Reconfiguration Complete
pub fn is_rrc_reconfiguration_complete(msg: &UL_DCCH_Message) -> bool {
    matches!(
        &msg.message,
        UL_DCCH_MessageType::C1(UL_DCCH_MessageType_c1::RrcReconfigurationComplete(_))
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // RRC Reconfiguration Tests
    // ========================================================================

    fn create_test_reconfiguration_params() -> RrcReconfigurationParams {
        RrcReconfigurationParams {
            rrc_transaction_id: 0,
            radio_bearer_config: None,
            secondary_cell_group: None,
            master_cell_group: Some(vec![0x00, 0x01, 0x02]), // Sample cell group config
            full_config: false,
        }
    }

    #[test]
    fn test_build_rrc_reconfiguration() {
        let params = create_test_reconfiguration_params();
        let result = build_rrc_reconfiguration(&params);
        assert!(result.is_ok());

        let msg = result.unwrap();
        assert!(is_rrc_reconfiguration(&msg));
    }

    #[test]
    fn test_parse_rrc_reconfiguration() {
        let params = create_test_reconfiguration_params();
        let msg = build_rrc_reconfiguration(&params).unwrap();
        let result = parse_rrc_reconfiguration(&msg);
        assert!(result.is_ok());

        let data = result.unwrap();
        assert_eq!(data.rrc_transaction_id, params.rrc_transaction_id);
        assert_eq!(data.master_cell_group, params.master_cell_group);
        assert_eq!(data.full_config, params.full_config);
    }

    #[test]
    fn test_rrc_reconfiguration_with_full_config() {
        let params = RrcReconfigurationParams {
            rrc_transaction_id: 2,
            radio_bearer_config: None,
            secondary_cell_group: None,
            master_cell_group: Some(vec![0xAA, 0xBB]),
            full_config: true,
        };

        let msg = build_rrc_reconfiguration(&params).unwrap();
        let data = parse_rrc_reconfiguration(&msg).unwrap();

        assert_eq!(data.rrc_transaction_id, 2);
        assert!(data.full_config);
        assert_eq!(data.master_cell_group, Some(vec![0xAA, 0xBB]));
    }

    #[test]
    fn test_encode_decode_rrc_reconfiguration() {
        let params = create_test_reconfiguration_params();
        let encoded = encode_rrc_reconfiguration(&params);
        assert!(encoded.is_ok());

        let bytes = encoded.unwrap();
        assert!(!bytes.is_empty());

        let decoded = decode_rrc_reconfiguration(&bytes);
        assert!(decoded.is_ok());

        let data = decoded.unwrap();
        assert_eq!(data.rrc_transaction_id, params.rrc_transaction_id);
    }

    #[test]
    fn test_invalid_rrc_transaction_id_reconfiguration() {
        let params = RrcReconfigurationParams {
            rrc_transaction_id: 5, // Invalid: must be 0-3
            radio_bearer_config: None,
            secondary_cell_group: None,
            master_cell_group: None,
            full_config: false,
        };

        let result = build_rrc_reconfiguration(&params);
        assert!(result.is_err());
    }

    // ========================================================================
    // RRC Reconfiguration Complete Tests
    // ========================================================================

    fn create_test_reconfiguration_complete_params() -> RrcReconfigurationCompleteParams {
        RrcReconfigurationCompleteParams {
            rrc_transaction_id: 0,
        }
    }

    #[test]
    fn test_build_rrc_reconfiguration_complete() {
        let params = create_test_reconfiguration_complete_params();
        let result = build_rrc_reconfiguration_complete(&params);
        assert!(result.is_ok());

        let msg = result.unwrap();
        assert!(is_rrc_reconfiguration_complete(&msg));
    }

    #[test]
    fn test_parse_rrc_reconfiguration_complete() {
        let params = create_test_reconfiguration_complete_params();
        let msg = build_rrc_reconfiguration_complete(&params).unwrap();
        let result = parse_rrc_reconfiguration_complete(&msg);
        assert!(result.is_ok());

        let data = result.unwrap();
        assert_eq!(data.rrc_transaction_id, params.rrc_transaction_id);
    }

    #[test]
    fn test_encode_decode_rrc_reconfiguration_complete() {
        let params = create_test_reconfiguration_complete_params();
        let encoded = encode_rrc_reconfiguration_complete(&params);
        assert!(encoded.is_ok());

        let bytes = encoded.unwrap();
        assert!(!bytes.is_empty());

        let decoded = decode_rrc_reconfiguration_complete(&bytes);
        assert!(decoded.is_ok());

        let data = decoded.unwrap();
        assert_eq!(data.rrc_transaction_id, params.rrc_transaction_id);
    }

    #[test]
    fn test_invalid_rrc_transaction_id_complete() {
        let params = RrcReconfigurationCompleteParams {
            rrc_transaction_id: 4, // Invalid: must be 0-3
        };

        let result = build_rrc_reconfiguration_complete(&params);
        assert!(result.is_err());
    }

    #[test]
    fn test_rrc_reconfiguration_complete_all_transaction_ids() {
        // Test all valid transaction IDs (0-3)
        for id in 0..=3 {
            let params = RrcReconfigurationCompleteParams {
                rrc_transaction_id: id,
            };
            let msg = build_rrc_reconfiguration_complete(&params).unwrap();
            let data = parse_rrc_reconfiguration_complete(&msg).unwrap();
            assert_eq!(data.rrc_transaction_id, id);
        }
    }
}
