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

    Ok(DL_DCCH_Message {
        message: message_type,
    })
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

    Ok(UL_DCCH_Message {
        message: message_type,
    })
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

    // ========================================================================
    // Wave-6 C5 — hand-derived golden byte vectors (TS 38.331 §6.2.1/§5.7.2,
    // UPER per X.691). Derived BY HAND from tools/rrc-15.6.0.asn1, NOT produced
    // by the encoder. Uses a fixed 3-octet NAS payload [0x7E,0x00,0x42].
    // ========================================================================

    /// Fixed NAS payload for the golden vectors (5GMM EPD 0x7E, arbitrary body).
    const GOLDEN_NAS: [u8; 3] = [0x7E, 0x00, 0x42];

    /// DLInformationTransfer on DL-DCCH, tid 0, carrying GOLDEN_NAS, 43 bits:
    ///
    /// ```text
    /// DL-DCCH-MessageType CHOICE {c1, mce} — 1 bit, c1                       0
    ///   c1 CHOICE (16 alts) — 4 bits, dlInformationTransfer = index 5     0101
    /// DLInformationTransfer ::= SEQUENCE { tid, criticalExtensions } (no ext)
    ///   rrc-TransactionIdentifier — 2 bits, 0                               00
    ///   criticalExtensions CHOICE {dlInformationTransfer, future} — 1 bit    0
    /// DLInformationTransfer-IEs ::= SEQUENCE { 3 OPTIONAL fields } (no ext)
    ///   presence (dedicatedNAS-Message|lateNonCritical|nonCritical)        100
    ///   dedicatedNAS-Message ::= OCTET STRING (unconstrained):
    ///     length determinant, 1 octet (<128), value 3               0000 0011
    ///     content 0x7E 0x00 0x42          0111 1110 0000 0000 0100 0010
    /// = 0 0101 00 0 100 00000011 011111100000000001000010  (43 bits) + 5 pad
    /// = 0010 1000|1000 0000|0110 1111|1100 0000|0000 1000|0100 0000
    /// = 0x28 0x80 0x6F 0xC0 0x08 0x40
    /// ```
    const GOLDEN_DL_INFO_TRANSFER_TID0: [u8; 6] = [0x28, 0x80, 0x6F, 0xC0, 0x08, 0x40];

    /// ULInformationTransfer on UL-DCCH, carrying GOLDEN_NAS, 41 bits.
    /// Note: ULInformationTransfer has NO rrc-TransactionIdentifier.
    ///
    /// ```text
    /// UL-DCCH-MessageType CHOICE {c1, mce} — 1 bit, c1                       0
    ///   c1 CHOICE (16 alts) — 4 bits, ulInformationTransfer = index 7     0111
    /// ULInformationTransfer ::= SEQUENCE { criticalExtensions } (no ext)
    ///   criticalExtensions CHOICE {ulInformationTransfer, future} — 1 bit    0
    /// ULInformationTransfer-IEs ::= SEQUENCE { 3 OPTIONAL fields } (no ext)
    ///   presence (dedicatedNAS-Message|lateNonCritical|nonCritical)        100
    ///   dedicatedNAS-Message OCTET STRING:
    ///     length octet, value 3                                     0000 0011
    ///     content 0x7E 0x00 0x42          0111 1110 0000 0000 0100 0010
    /// = 0 0111 0 100 00000011 011111100000000001000010  (41 bits) + 7 pad
    /// = 0011 1010|0000 0001|1011 1111|0000 0000|0010 0001|0000 0000
    /// = 0x3A 0x01 0xBF 0x00 0x21 0x00
    /// ```
    const GOLDEN_UL_INFO_TRANSFER: [u8; 6] = [0x3A, 0x01, 0xBF, 0x00, 0x21, 0x00];

    #[test]
    fn golden_dl_information_transfer_bytes() {
        let bytes = encode_dl_information_transfer(&DlInformationTransferParams {
            rrc_transaction_id: 0,
            dedicated_nas_message: Some(GOLDEN_NAS.to_vec()),
        })
        .expect("encode DLInformationTransfer");
        assert_eq!(
            bytes,
            GOLDEN_DL_INFO_TRANSFER_TID0.to_vec(),
            "DLInformationTransfer(tid 0, NAS) must match the hand-derived UPER bytes"
        );
        // Cross-decode: the NAS survives byte-for-byte.
        let data = decode_dl_information_transfer(&GOLDEN_DL_INFO_TRANSFER_TID0).expect("decode");
        assert_eq!(data.rrc_transaction_id, 0);
        assert_eq!(data.dedicated_nas_message, Some(GOLDEN_NAS.to_vec()));
    }

    #[test]
    fn golden_ul_information_transfer_bytes() {
        let bytes = encode_ul_information_transfer(&UlInformationTransferParams {
            dedicated_nas_message: Some(GOLDEN_NAS.to_vec()),
        })
        .expect("encode ULInformationTransfer");
        assert_eq!(
            bytes,
            GOLDEN_UL_INFO_TRANSFER.to_vec(),
            "ULInformationTransfer(NAS) must match the hand-derived UPER bytes"
        );
        let data = decode_ul_information_transfer(&GOLDEN_UL_INFO_TRANSFER).expect("decode");
        assert_eq!(data.dedicated_nas_message, Some(GOLDEN_NAS.to_vec()));
    }
}
