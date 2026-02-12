//! RRC Information Transfer Procedures
//!
//! Implements the UL/DL Information Transfer procedures as defined in 3GPP TS 38.331 Section 5.7.2.
//! These procedures are used to transfer NAS messages between the UE and the network.
//!
//! The procedures consist of two messages:
//! 1. `DLInformationTransfer` - gNB → UE: Downlink NAS message transfer
//! 2. `ULInformationTransfer` - UE → gNB: Uplink NAS message transfer

use crate::codec::generated::*;
use crate::codec::{decode_rrc, encode_rrc, RrcCodecError};
use thiserror::Error;

/// Errors that can occur during RRC Information Transfer procedures
#[derive(Debug, Error)]
pub enum RrcInformationTransferError {
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
// DL Information Transfer
// ============================================================================

/// Parameters for building a DL Information Transfer message
#[derive(Debug, Clone)]
pub struct DlInformationTransferParams {
    /// RRC Transaction Identifier (0-3)
    pub rrc_transaction_id: u8,
    /// Dedicated NAS message (optional)
    pub dedicated_nas_message: Option<Vec<u8>>,
}

/// Parsed DL Information Transfer data
#[derive(Debug, Clone)]
pub struct DlInformationTransferData {
    /// RRC Transaction Identifier
    pub rrc_transaction_id: u8,
    /// Dedicated NAS message (optional)
    pub dedicated_nas_message: Option<Vec<u8>>,
}


/// Build a DL Information Transfer message
pub fn build_dl_information_transfer(
    params: &DlInformationTransferParams,
) -> Result<DL_DCCH_Message, RrcInformationTransferError> {
    if params.rrc_transaction_id > 3 {
        return Err(RrcInformationTransferError::InvalidFieldValue(
            "RRC Transaction ID must be 0-3".to_string(),
        ));
    }

    let dl_information_transfer_ies = DLInformationTransfer_IEs {
        dedicated_nas_message: params
            .dedicated_nas_message
            .as_ref()
            .map(|msg| DedicatedNAS_Message(msg.clone())),
        late_non_critical_extension: None,
        non_critical_extension: None,
    };

    let dl_information_transfer = DLInformationTransfer {
        rrc_transaction_identifier: RRC_TransactionIdentifier(params.rrc_transaction_id),
        critical_extensions: DLInformationTransferCriticalExtensions::DlInformationTransfer(
            dl_information_transfer_ies,
        ),
    };

    let message_type = DL_DCCH_MessageType::C1(DL_DCCH_MessageType_c1::DlInformationTransfer(
        dl_information_transfer,
    ));

    Ok(DL_DCCH_Message { message: message_type })
}

/// Parse a DL Information Transfer from a DL-DCCH message
pub fn parse_dl_information_transfer(
    msg: &DL_DCCH_Message,
) -> Result<DlInformationTransferData, RrcInformationTransferError> {
    let dl_information_transfer = match &msg.message {
        DL_DCCH_MessageType::C1(c1) => match c1 {
            DL_DCCH_MessageType_c1::DlInformationTransfer(transfer) => transfer,
            _ => {
                return Err(RrcInformationTransferError::InvalidMessageType {
                    expected: "DLInformationTransfer".to_string(),
                    actual: "other c1 message".to_string(),
                })
            }
        },
        _ => {
            return Err(RrcInformationTransferError::InvalidMessageType {
                expected: "c1".to_string(),
                actual: "messageClassExtension".to_string(),
            })
        }
    };

    let ies = match &dl_information_transfer.critical_extensions {
        DLInformationTransferCriticalExtensions::DlInformationTransfer(ies) => ies,
        DLInformationTransferCriticalExtensions::CriticalExtensionsFuture(_) => {
            return Err(RrcInformationTransferError::InvalidMessageType {
                expected: "dlInformationTransfer".to_string(),
                actual: "criticalExtensionsFuture".to_string(),
            })
        }
    };

    Ok(DlInformationTransferData {
        rrc_transaction_id: dl_information_transfer.rrc_transaction_identifier.0,
        dedicated_nas_message: ies.dedicated_nas_message.as_ref().map(|msg| msg.0.clone()),
    })
}


// ============================================================================
// UL Information Transfer
// ============================================================================

/// Parameters for building a UL Information Transfer message
#[derive(Debug, Clone)]
pub struct UlInformationTransferParams {
    /// Dedicated NAS message (optional)
    pub dedicated_nas_message: Option<Vec<u8>>,
}

/// Parsed UL Information Transfer data
#[derive(Debug, Clone)]
pub struct UlInformationTransferData {
    /// Dedicated NAS message (optional)
    pub dedicated_nas_message: Option<Vec<u8>>,
}

/// Build a UL Information Transfer message
pub fn build_ul_information_transfer(
    params: &UlInformationTransferParams,
) -> Result<UL_DCCH_Message, RrcInformationTransferError> {
    let ul_information_transfer_ies = ULInformationTransfer_IEs {
        dedicated_nas_message: params
            .dedicated_nas_message
            .as_ref()
            .map(|msg| DedicatedNAS_Message(msg.clone())),
        late_non_critical_extension: None,
        non_critical_extension: None,
    };

    let ul_information_transfer = ULInformationTransfer {
        critical_extensions: ULInformationTransferCriticalExtensions::UlInformationTransfer(
            ul_information_transfer_ies,
        ),
    };

    let message_type = UL_DCCH_MessageType::C1(UL_DCCH_MessageType_c1::UlInformationTransfer(
        ul_information_transfer,
    ));

    Ok(UL_DCCH_Message { message: message_type })
}

/// Parse a UL Information Transfer from a UL-DCCH message
pub fn parse_ul_information_transfer(
    msg: &UL_DCCH_Message,
) -> Result<UlInformationTransferData, RrcInformationTransferError> {
    let ul_information_transfer = match &msg.message {
        UL_DCCH_MessageType::C1(c1) => match c1 {
            UL_DCCH_MessageType_c1::UlInformationTransfer(transfer) => transfer,
            _ => {
                return Err(RrcInformationTransferError::InvalidMessageType {
                    expected: "ULInformationTransfer".to_string(),
                    actual: "other c1 message".to_string(),
                })
            }
        },
        _ => {
            return Err(RrcInformationTransferError::InvalidMessageType {
                expected: "c1".to_string(),
                actual: "messageClassExtension".to_string(),
            })
        }
    };

    let ies = match &ul_information_transfer.critical_extensions {
        ULInformationTransferCriticalExtensions::UlInformationTransfer(ies) => ies,
        ULInformationTransferCriticalExtensions::CriticalExtensionsFuture(_) => {
            return Err(RrcInformationTransferError::InvalidMessageType {
                expected: "ulInformationTransfer".to_string(),
                actual: "criticalExtensionsFuture".to_string(),
            })
        }
    };

    Ok(UlInformationTransferData {
        dedicated_nas_message: ies.dedicated_nas_message.as_ref().map(|msg| msg.0.clone()),
    })
}


// ============================================================================
// Convenience Functions
// ============================================================================

/// Build and encode a DL Information Transfer to bytes
pub fn encode_dl_information_transfer(
    params: &DlInformationTransferParams,
) -> Result<Vec<u8>, RrcInformationTransferError> {
    let msg = build_dl_information_transfer(params)?;
    Ok(encode_rrc(&msg)?)
}

/// Decode and parse a DL Information Transfer from bytes
pub fn decode_dl_information_transfer(
    bytes: &[u8],
) -> Result<DlInformationTransferData, RrcInformationTransferError> {
    let msg: DL_DCCH_Message = decode_rrc(bytes)?;
    parse_dl_information_transfer(&msg)
}

/// Build and encode a UL Information Transfer to bytes
pub fn encode_ul_information_transfer(
    params: &UlInformationTransferParams,
) -> Result<Vec<u8>, RrcInformationTransferError> {
    let msg = build_ul_information_transfer(params)?;
    Ok(encode_rrc(&msg)?)
}

/// Decode and parse a UL Information Transfer from bytes
pub fn decode_ul_information_transfer(
    bytes: &[u8],
) -> Result<UlInformationTransferData, RrcInformationTransferError> {
    let msg: UL_DCCH_Message = decode_rrc(bytes)?;
    parse_ul_information_transfer(&msg)
}

/// Check if a DL-DCCH message is a DL Information Transfer
pub fn is_dl_information_transfer(msg: &DL_DCCH_Message) -> bool {
    matches!(
        &msg.message,
        DL_DCCH_MessageType::C1(DL_DCCH_MessageType_c1::DlInformationTransfer(_))
    )
}

/// Check if a UL-DCCH message is a UL Information Transfer
pub fn is_ul_information_transfer(msg: &UL_DCCH_Message) -> bool {
    matches!(
        &msg.message,
        UL_DCCH_MessageType::C1(UL_DCCH_MessageType_c1::UlInformationTransfer(_))
    )
}


#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // DL Information Transfer Tests
    // ========================================================================

    fn create_test_dl_information_transfer_params() -> DlInformationTransferParams {
        DlInformationTransferParams {
            rrc_transaction_id: 0,
            dedicated_nas_message: Some(vec![0x7e, 0x00, 0x41, 0x01, 0x02, 0x03]),
        }
    }

    #[test]
    fn test_build_dl_information_transfer() {
        let params = create_test_dl_information_transfer_params();
        let result = build_dl_information_transfer(&params);
        assert!(result.is_ok());

        let msg = result.unwrap();
        assert!(is_dl_information_transfer(&msg));
    }

    #[test]
    fn test_parse_dl_information_transfer() {
        let params = create_test_dl_information_transfer_params();
        let msg = build_dl_information_transfer(&params).unwrap();
        let result = parse_dl_information_transfer(&msg);
        assert!(result.is_ok());

        let data = result.unwrap();
        assert_eq!(data.rrc_transaction_id, params.rrc_transaction_id);
        assert_eq!(data.dedicated_nas_message, params.dedicated_nas_message);
    }

    #[test]
    fn test_dl_information_transfer_without_nas_message() {
        let params = DlInformationTransferParams {
            rrc_transaction_id: 1,
            dedicated_nas_message: None,
        };

        let msg = build_dl_information_transfer(&params).unwrap();
        let data = parse_dl_information_transfer(&msg).unwrap();

        assert_eq!(data.rrc_transaction_id, 1);
        assert_eq!(data.dedicated_nas_message, None);
    }

    #[test]
    fn test_encode_decode_dl_information_transfer() {
        let params = create_test_dl_information_transfer_params();
        let encoded = encode_dl_information_transfer(&params);
        assert!(encoded.is_ok());

        let bytes = encoded.unwrap();
        assert!(!bytes.is_empty());

        let decoded = decode_dl_information_transfer(&bytes);
        assert!(decoded.is_ok());

        let data = decoded.unwrap();
        assert_eq!(data.rrc_transaction_id, params.rrc_transaction_id);
        assert_eq!(data.dedicated_nas_message, params.dedicated_nas_message);
    }

    #[test]
    fn test_invalid_rrc_transaction_id_dl() {
        let params = DlInformationTransferParams {
            rrc_transaction_id: 5, // Invalid: must be 0-3
            dedicated_nas_message: None,
        };

        let result = build_dl_information_transfer(&params);
        assert!(result.is_err());
    }

    #[test]
    fn test_dl_information_transfer_all_transaction_ids() {
        for id in 0..=3 {
            let params = DlInformationTransferParams {
                rrc_transaction_id: id,
                dedicated_nas_message: None,
            };
            let msg = build_dl_information_transfer(&params).unwrap();
            let data = parse_dl_information_transfer(&msg).unwrap();
            assert_eq!(data.rrc_transaction_id, id);
        }
    }

    // ========================================================================
    // UL Information Transfer Tests
    // ========================================================================

    fn create_test_ul_information_transfer_params() -> UlInformationTransferParams {
        UlInformationTransferParams {
            dedicated_nas_message: Some(vec![0x7e, 0x00, 0x41, 0x01, 0x02, 0x03]),
        }
    }

    #[test]
    fn test_build_ul_information_transfer() {
        let params = create_test_ul_information_transfer_params();
        let result = build_ul_information_transfer(&params);
        assert!(result.is_ok());

        let msg = result.unwrap();
        assert!(is_ul_information_transfer(&msg));
    }

    #[test]
    fn test_parse_ul_information_transfer() {
        let params = create_test_ul_information_transfer_params();
        let msg = build_ul_information_transfer(&params).unwrap();
        let result = parse_ul_information_transfer(&msg);
        assert!(result.is_ok());

        let data = result.unwrap();
        assert_eq!(data.dedicated_nas_message, params.dedicated_nas_message);
    }

    #[test]
    fn test_ul_information_transfer_without_nas_message() {
        let params = UlInformationTransferParams {
            dedicated_nas_message: None,
        };

        let msg = build_ul_information_transfer(&params).unwrap();
        let data = parse_ul_information_transfer(&msg).unwrap();

        assert_eq!(data.dedicated_nas_message, None);
    }

    #[test]
    fn test_encode_decode_ul_information_transfer() {
        let params = create_test_ul_information_transfer_params();
        let encoded = encode_ul_information_transfer(&params);
        assert!(encoded.is_ok());

        let bytes = encoded.unwrap();
        assert!(!bytes.is_empty());

        let decoded = decode_ul_information_transfer(&bytes);
        assert!(decoded.is_ok());

        let data = decoded.unwrap();
        assert_eq!(data.dedicated_nas_message, params.dedicated_nas_message);
    }

    #[test]
    fn test_ul_information_transfer_large_nas_message() {
        let large_nas_message = vec![0xAB; 1000]; // 1KB NAS message
        let params = UlInformationTransferParams {
            dedicated_nas_message: Some(large_nas_message.clone()),
        };

        let msg = build_ul_information_transfer(&params).unwrap();
        let data = parse_ul_information_transfer(&msg).unwrap();

        assert_eq!(data.dedicated_nas_message, Some(large_nas_message));
    }

    #[test]
    fn test_dl_information_transfer_large_nas_message() {
        let large_nas_message = vec![0xCD; 1000]; // 1KB NAS message
        let params = DlInformationTransferParams {
            rrc_transaction_id: 2,
            dedicated_nas_message: Some(large_nas_message.clone()),
        };

        let msg = build_dl_information_transfer(&params).unwrap();
        let data = parse_dl_information_transfer(&msg).unwrap();

        assert_eq!(data.rrc_transaction_id, 2);
        assert_eq!(data.dedicated_nas_message, Some(large_nas_message));
    }
}
