//! RRC Reestablishment Procedure
//!
//! Implements the RRC Reestablishment procedure as defined in 3GPP TS 38.331 Section 5.3.7.
//! This procedure is used when the UE needs to recover from radio link failure,
//! handover failure, integrity check failure, or RRC reconfiguration failure.
//!
//! The procedure consists of three messages:
//! 1. RRCReestablishmentRequest - UE -> gNB: Request to reestablish RRC connection
//! 2. RRCReestablishment - gNB -> UE: Network response with configuration
//! 3. RRCReestablishmentComplete - UE -> gNB: Confirmation of reestablishment

use crate::codec::generated::*;
use crate::codec::{decode_rrc, encode_rrc, RrcCodecError};
use bitvec::prelude::*;
use thiserror::Error;

/// Errors that can occur during RRC Reestablishment procedures
#[derive(Debug, Error)]
pub enum RrcReestablishmentError {
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

    /// Missing mandatory field
    #[error("Missing mandatory field: {0}")]
    MissingMandatoryField(String),

    /// Invalid field value
    #[error("Invalid field value: {0}")]
    InvalidFieldValue(String),
}

/// Reestablishment cause
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReestablishmentCauseValue {
    /// Reconfiguration failure
    ReconfigurationFailure,
    /// Handover failure
    HandoverFailure,
    /// Other failure
    OtherFailure,
}

/// UE Identity for Reestablishment
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReestablishmentUeIdentity {
    /// C-RNTI value (16 bits)
    pub c_rnti: u16,
    /// Physical Cell ID (0-1007)
    pub phys_cell_id: u16,
    /// ShortMAC-I (16 bits)
    pub short_mac_i: u16,
}

// ============================================================================
// RRC Reestablishment Request
// ============================================================================

/// Parameters for building an RRC Reestablishment Request
#[derive(Debug, Clone)]
pub struct RrcReestablishmentRequestParams {
    /// UE Identity for reestablishment
    pub ue_identity: ReestablishmentUeIdentity,
    /// Reestablishment cause
    pub reestablishment_cause: ReestablishmentCauseValue,
}

/// Parsed RRC Reestablishment Request data
#[derive(Debug, Clone)]
pub struct RrcReestablishmentRequestData {
    /// UE Identity for reestablishment
    pub ue_identity: ReestablishmentUeIdentity,
    /// Reestablishment cause
    pub reestablishment_cause: ReestablishmentCauseValue,
}

/// Build an RRC Reestablishment Request message
pub fn build_rrc_reestablishment_request(
    params: &RrcReestablishmentRequestParams,
) -> Result<UL_CCCH_Message, RrcReestablishmentError> {
    // Build C-RNTI (u16)
    let c_rnti = RNTI_Value(params.ue_identity.c_rnti);

    // Build PhysCellId (0-1007)
    let phys_cell_id = PhysCellId(params.ue_identity.phys_cell_id);

    // Build ShortMAC-I (16 bits)
    let mut short_mac_i_bv: BitVec<u8, Msb0> = BitVec::with_capacity(16);
    for i in (0..16).rev() {
        short_mac_i_bv.push((params.ue_identity.short_mac_i >> i) & 1 == 1);
    }

    let reestab_ue_identity = ReestabUE_Identity {
        c_rnti,
        phys_cell_id,
        short_mac_i: ShortMAC_I(short_mac_i_bv),
    };

    let reestablishment_cause = match params.reestablishment_cause {
        ReestablishmentCauseValue::ReconfigurationFailure => {
            ReestablishmentCause(ReestablishmentCause::RECONFIGURATION_FAILURE)
        }
        ReestablishmentCauseValue::HandoverFailure => {
            ReestablishmentCause(ReestablishmentCause::HANDOVER_FAILURE)
        }
        ReestablishmentCauseValue::OtherFailure => {
            ReestablishmentCause(ReestablishmentCause::OTHER_FAILURE)
        }
    };

    // Build spare bit (1 bit)
    let mut spare_bv: BitVec<u8, Msb0> = BitVec::new();
    spare_bv.push(false);

    let rrc_reestablishment_request_ies = RRCReestablishmentRequest_IEs {
        ue_identity: reestab_ue_identity,
        reestablishment_cause,
        spare: RRCReestablishmentRequest_IEsSpare(spare_bv),
    };

    let rrc_reestablishment_request = RRCReestablishmentRequest {
        rrc_reestablishment_request: rrc_reestablishment_request_ies,
    };

    let message_type = UL_CCCH_MessageType::C1(
        UL_CCCH_MessageType_c1::RrcReestablishmentRequest(rrc_reestablishment_request),
    );

    Ok(UL_CCCH_Message { message: message_type })
}

/// Parse an RRC Reestablishment Request from a UL-CCCH message
pub fn parse_rrc_reestablishment_request(
    msg: &UL_CCCH_Message,
) -> Result<RrcReestablishmentRequestData, RrcReestablishmentError> {
    let request = match &msg.message {
        UL_CCCH_MessageType::C1(c1) => match c1 {
            UL_CCCH_MessageType_c1::RrcReestablishmentRequest(req) => req,
            _ => {
                return Err(RrcReestablishmentError::InvalidMessageType {
                    expected: "RRCReestablishmentRequest".to_string(),
                    actual: "other c1 message".to_string(),
                })
            }
        },
        _ => {
            return Err(RrcReestablishmentError::InvalidMessageType {
                expected: "c1".to_string(),
                actual: "messageClassExtension".to_string(),
            })
        }
    };

    let ies = &request.rrc_reestablishment_request;

    // Parse C-RNTI
    let c_rnti = ies.ue_identity.c_rnti.0;

    // Parse PhysCellId
    let phys_cell_id = ies.ue_identity.phys_cell_id.0;

    // Parse ShortMAC-I
    let short_mac_i = bitvec_to_u16(&ies.ue_identity.short_mac_i.0);

    // Parse reestablishment cause
    let reestablishment_cause = match ies.reestablishment_cause.0 {
        ReestablishmentCause::RECONFIGURATION_FAILURE => {
            ReestablishmentCauseValue::ReconfigurationFailure
        }
        ReestablishmentCause::HANDOVER_FAILURE => ReestablishmentCauseValue::HandoverFailure,
        ReestablishmentCause::OTHER_FAILURE => ReestablishmentCauseValue::OtherFailure,
        _ => ReestablishmentCauseValue::OtherFailure,
    };

    Ok(RrcReestablishmentRequestData {
        ue_identity: ReestablishmentUeIdentity {
            c_rnti,
            phys_cell_id,
            short_mac_i,
        },
        reestablishment_cause,
    })
}

fn bitvec_to_u16(bv: &BitVec<u8, Msb0>) -> u16 {
    let mut value: u16 = 0;
    for bit in bv.iter() {
        value = (value << 1) | (*bit as u16);
    }
    value
}

// ============================================================================
// RRC Reestablishment Complete
// ============================================================================

/// Parameters for building an RRC Reestablishment Complete message
#[derive(Debug, Clone)]
pub struct RrcReestablishmentCompleteParams {
    /// RRC Transaction Identifier (0-3)
    pub rrc_transaction_id: u8,
}

/// Parsed RRC Reestablishment Complete data
#[derive(Debug, Clone)]
pub struct RrcReestablishmentCompleteData {
    /// RRC Transaction Identifier
    pub rrc_transaction_id: u8,
}

/// Build an RRC Reestablishment Complete message
pub fn build_rrc_reestablishment_complete(
    params: &RrcReestablishmentCompleteParams,
) -> Result<UL_DCCH_Message, RrcReestablishmentError> {
    if params.rrc_transaction_id > 3 {
        return Err(RrcReestablishmentError::InvalidFieldValue(
            "RRC Transaction ID must be 0-3".to_string(),
        ));
    }

    let ies = RRCReestablishmentComplete_IEs {
        late_non_critical_extension: None,
        non_critical_extension: None,
    };

    let rrc_reestablishment_complete = RRCReestablishmentComplete {
        rrc_transaction_identifier: RRC_TransactionIdentifier(params.rrc_transaction_id),
        critical_extensions:
            RRCReestablishmentCompleteCriticalExtensions::RrcReestablishmentComplete(ies),
    };

    let message_type = UL_DCCH_MessageType::C1(
        UL_DCCH_MessageType_c1::RrcReestablishmentComplete(rrc_reestablishment_complete),
    );

    Ok(UL_DCCH_Message { message: message_type })
}

/// Parse an RRC Reestablishment Complete from a UL-DCCH message
pub fn parse_rrc_reestablishment_complete(
    msg: &UL_DCCH_Message,
) -> Result<RrcReestablishmentCompleteData, RrcReestablishmentError> {
    let complete = match &msg.message {
        UL_DCCH_MessageType::C1(c1) => match c1 {
            UL_DCCH_MessageType_c1::RrcReestablishmentComplete(c) => c,
            _ => {
                return Err(RrcReestablishmentError::InvalidMessageType {
                    expected: "RRCReestablishmentComplete".to_string(),
                    actual: "other c1 message".to_string(),
                })
            }
        },
        _ => {
            return Err(RrcReestablishmentError::InvalidMessageType {
                expected: "c1".to_string(),
                actual: "messageClassExtension".to_string(),
            })
        }
    };

    match &complete.critical_extensions {
        RRCReestablishmentCompleteCriticalExtensions::RrcReestablishmentComplete(_) => {}
        RRCReestablishmentCompleteCriticalExtensions::CriticalExtensionsFuture(_) => {
            return Err(RrcReestablishmentError::InvalidMessageType {
                expected: "rrcReestablishmentComplete".to_string(),
                actual: "criticalExtensionsFuture".to_string(),
            })
        }
    };

    Ok(RrcReestablishmentCompleteData {
        rrc_transaction_id: complete.rrc_transaction_identifier.0,
    })
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Build and encode an RRC Reestablishment Request to bytes
pub fn encode_rrc_reestablishment_request(
    params: &RrcReestablishmentRequestParams,
) -> Result<Vec<u8>, RrcReestablishmentError> {
    let msg = build_rrc_reestablishment_request(params)?;
    Ok(encode_rrc(&msg)?)
}

/// Decode and parse an RRC Reestablishment Request from bytes
pub fn decode_rrc_reestablishment_request(
    bytes: &[u8],
) -> Result<RrcReestablishmentRequestData, RrcReestablishmentError> {
    let msg: UL_CCCH_Message = decode_rrc(bytes)?;
    parse_rrc_reestablishment_request(&msg)
}

/// Build and encode an RRC Reestablishment Complete to bytes
pub fn encode_rrc_reestablishment_complete(
    params: &RrcReestablishmentCompleteParams,
) -> Result<Vec<u8>, RrcReestablishmentError> {
    let msg = build_rrc_reestablishment_complete(params)?;
    Ok(encode_rrc(&msg)?)
}

/// Decode and parse an RRC Reestablishment Complete from bytes
pub fn decode_rrc_reestablishment_complete(
    bytes: &[u8],
) -> Result<RrcReestablishmentCompleteData, RrcReestablishmentError> {
    let msg: UL_DCCH_Message = decode_rrc(bytes)?;
    parse_rrc_reestablishment_complete(&msg)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_request_params() -> RrcReestablishmentRequestParams {
        RrcReestablishmentRequestParams {
            ue_identity: ReestablishmentUeIdentity {
                c_rnti: 0x1234,
                phys_cell_id: 100,
                short_mac_i: 0xABCD,
            },
            reestablishment_cause: ReestablishmentCauseValue::HandoverFailure,
        }
    }

    #[test]
    fn test_build_rrc_reestablishment_request() {
        let params = create_test_request_params();
        let result = build_rrc_reestablishment_request(&params);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_rrc_reestablishment_request() {
        let params = create_test_request_params();
        let msg = build_rrc_reestablishment_request(&params).unwrap();
        let data = parse_rrc_reestablishment_request(&msg).unwrap();

        assert_eq!(data.ue_identity.c_rnti, params.ue_identity.c_rnti);
        assert_eq!(data.ue_identity.phys_cell_id, params.ue_identity.phys_cell_id);
        assert_eq!(data.ue_identity.short_mac_i, params.ue_identity.short_mac_i);
        assert_eq!(data.reestablishment_cause, params.reestablishment_cause);
    }

    #[test]
    fn test_encode_decode_rrc_reestablishment_request() {
        let params = create_test_request_params();
        let encoded = encode_rrc_reestablishment_request(&params).expect("Failed to encode");
        assert!(!encoded.is_empty());

        let decoded = decode_rrc_reestablishment_request(&encoded).expect("Failed to decode");
        assert_eq!(decoded.ue_identity.c_rnti, params.ue_identity.c_rnti);
        assert_eq!(decoded.reestablishment_cause, params.reestablishment_cause);
    }

    #[test]
    fn test_build_rrc_reestablishment_complete() {
        let params = RrcReestablishmentCompleteParams {
            rrc_transaction_id: 1,
        };
        let result = build_rrc_reestablishment_complete(&params);
        assert!(result.is_ok());
    }

    #[test]
    fn test_encode_decode_rrc_reestablishment_complete() {
        let params = RrcReestablishmentCompleteParams {
            rrc_transaction_id: 2,
        };
        let encoded = encode_rrc_reestablishment_complete(&params).expect("Failed to encode");
        let decoded = decode_rrc_reestablishment_complete(&encoded).expect("Failed to decode");
        assert_eq!(decoded.rrc_transaction_id, 2);
    }

    #[test]
    fn test_invalid_transaction_id() {
        let params = RrcReestablishmentCompleteParams {
            rrc_transaction_id: 5,
        };
        assert!(build_rrc_reestablishment_complete(&params).is_err());
    }

    #[test]
    fn test_all_reestablishment_causes() {
        let causes = [
            ReestablishmentCauseValue::ReconfigurationFailure,
            ReestablishmentCauseValue::HandoverFailure,
            ReestablishmentCauseValue::OtherFailure,
        ];

        for cause in causes {
            let params = RrcReestablishmentRequestParams {
                ue_identity: ReestablishmentUeIdentity {
                    c_rnti: 0x1000,
                    phys_cell_id: 50,
                    short_mac_i: 0x0001,
                },
                reestablishment_cause: cause,
            };
            let msg = build_rrc_reestablishment_request(&params).unwrap();
            let data = parse_rrc_reestablishment_request(&msg).unwrap();
            assert_eq!(data.reestablishment_cause, cause);
        }
    }
}
