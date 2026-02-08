//! RRC Resume Procedure
//!
//! Implements the RRC Resume procedure as defined in 3GPP TS 38.331 Section 5.3.13.
//! This procedure is used when UE transitions from RRC_INACTIVE to RRC_CONNECTED,
//! reusing the previously established UE context.
//!
//! The procedure consists of three messages:
//! 1. RRCResumeRequest - UE -> gNB: Request to resume RRC connection
//! 2. RRCResume - gNB -> UE: Network response with updated configuration
//! 3. RRCResumeComplete - UE -> gNB: Confirmation of resume

use crate::codec::generated::*;
use crate::codec::{decode_rrc, encode_rrc, RrcCodecError};
use bitvec::prelude::*;
use thiserror::Error;

/// Errors that can occur during RRC Resume procedures
#[derive(Debug, Error)]
pub enum RrcResumeError {
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

/// Resume cause
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResumeCauseValue {
    /// Emergency
    Emergency,
    /// High priority access
    HighPriorityAccess,
    /// MT access
    MtAccess,
    /// MO signalling
    MoSignalling,
    /// MO data
    MoData,
    /// MO voice call
    MoVoiceCall,
    /// MO video call
    MoVideoCall,
    /// MO SMS
    MoSms,
    /// RNA update
    RnaUpdate,
    /// MPS priority access
    MpsPriorityAccess,
    /// MCS priority access
    McsPriorityAccess,
}

// ============================================================================
// RRC Resume Request
// ============================================================================

/// Parameters for building an RRC Resume Request
#[derive(Debug, Clone)]
pub struct RrcResumeRequestParams {
    /// Resume Identity (ShortI-RNTI, 24 bits, max value 0xFFFFFF)
    pub resume_identity: u32,
    /// Resume MAC-I (16 bits)
    pub resume_mac_i: u16,
    /// Resume cause
    pub resume_cause: ResumeCauseValue,
}

/// Parsed RRC Resume Request data
#[derive(Debug, Clone)]
pub struct RrcResumeRequestData {
    /// Resume Identity (ShortI-RNTI, 24 bits)
    pub resume_identity: u32,
    /// Resume MAC-I
    pub resume_mac_i: u16,
    /// Resume cause
    pub resume_cause: ResumeCauseValue,
}

/// Build an RRC Resume Request message
pub fn build_rrc_resume_request(
    params: &RrcResumeRequestParams,
) -> Result<UL_CCCH_Message, RrcResumeError> {
    // Build Resume Identity (ShortI-RNTI, 24 bits)
    let mut resume_id_bv: BitVec<u8, Msb0> = BitVec::with_capacity(24);
    for i in (0..24).rev() {
        resume_id_bv.push((params.resume_identity >> i) & 1 == 1);
    }

    // Build Resume MAC-I (16 bits)
    let mut resume_mac_i_bv: BitVec<u8, Msb0> = BitVec::with_capacity(16);
    for i in (0..16).rev() {
        resume_mac_i_bv.push((params.resume_mac_i >> i) & 1 == 1);
    }

    // Build resume cause
    let resume_cause = match params.resume_cause {
        ResumeCauseValue::Emergency => ResumeCause(ResumeCause::EMERGENCY),
        ResumeCauseValue::HighPriorityAccess => ResumeCause(ResumeCause::HIGH_PRIORITY_ACCESS),
        ResumeCauseValue::MtAccess => ResumeCause(ResumeCause::MT_ACCESS),
        ResumeCauseValue::MoSignalling => ResumeCause(ResumeCause::MO_SIGNALLING),
        ResumeCauseValue::MoData => ResumeCause(ResumeCause::MO_DATA),
        ResumeCauseValue::MoVoiceCall => ResumeCause(ResumeCause::MO_VOICE_CALL),
        ResumeCauseValue::MoVideoCall => ResumeCause(ResumeCause::MO_VIDEO_CALL),
        ResumeCauseValue::MoSms => ResumeCause(ResumeCause::MO_SMS),
        ResumeCauseValue::RnaUpdate => ResumeCause(ResumeCause::RNA_UPDATE),
        ResumeCauseValue::MpsPriorityAccess => ResumeCause(ResumeCause::MPS_PRIORITY_ACCESS),
        ResumeCauseValue::McsPriorityAccess => ResumeCause(ResumeCause::MCS_PRIORITY_ACCESS),
    };

    // Build spare bit (1 bit)
    let mut spare_bv: BitVec<u8, Msb0> = BitVec::new();
    spare_bv.push(false);

    let rrc_resume_request_ies = RRCResumeRequest_IEs {
        resume_identity: ShortI_RNTI_Value(resume_id_bv),
        resume_mac_i: RRCResumeRequest_IEsResumeMAC_I(resume_mac_i_bv),
        resume_cause,
        spare: RRCResumeRequest_IEsSpare(spare_bv),
    };

    let rrc_resume_request = RRCResumeRequest {
        rrc_resume_request: rrc_resume_request_ies,
    };

    let message_type = UL_CCCH_MessageType::C1(
        UL_CCCH_MessageType_c1::RrcResumeRequest(rrc_resume_request),
    );

    Ok(UL_CCCH_Message { message: message_type })
}

/// Parse an RRC Resume Request from a UL-CCCH message
pub fn parse_rrc_resume_request(
    msg: &UL_CCCH_Message,
) -> Result<RrcResumeRequestData, RrcResumeError> {
    let request = match &msg.message {
        UL_CCCH_MessageType::C1(c1) => match c1 {
            UL_CCCH_MessageType_c1::RrcResumeRequest(req) => req,
            _ => {
                return Err(RrcResumeError::InvalidMessageType {
                    expected: "RRCResumeRequest".to_string(),
                    actual: "other c1 message".to_string(),
                })
            }
        },
        _ => {
            return Err(RrcResumeError::InvalidMessageType {
                expected: "c1".to_string(),
                actual: "messageClassExtension".to_string(),
            })
        }
    };

    let ies = &request.rrc_resume_request;

    // Parse Resume Identity (24 bits)
    let resume_identity = bitvec_to_u32(&ies.resume_identity.0);

    // Parse Resume MAC-I
    let resume_mac_i = bitvec_to_u16(&ies.resume_mac_i.0);

    // Parse resume cause
    let resume_cause = match ies.resume_cause.0 {
        ResumeCause::EMERGENCY => ResumeCauseValue::Emergency,
        ResumeCause::HIGH_PRIORITY_ACCESS => ResumeCauseValue::HighPriorityAccess,
        ResumeCause::MT_ACCESS => ResumeCauseValue::MtAccess,
        ResumeCause::MO_SIGNALLING => ResumeCauseValue::MoSignalling,
        ResumeCause::MO_DATA => ResumeCauseValue::MoData,
        ResumeCause::MO_VOICE_CALL => ResumeCauseValue::MoVoiceCall,
        ResumeCause::MO_VIDEO_CALL => ResumeCauseValue::MoVideoCall,
        ResumeCause::MO_SMS => ResumeCauseValue::MoSms,
        ResumeCause::RNA_UPDATE => ResumeCauseValue::RnaUpdate,
        ResumeCause::MPS_PRIORITY_ACCESS => ResumeCauseValue::MpsPriorityAccess,
        ResumeCause::MCS_PRIORITY_ACCESS => ResumeCauseValue::McsPriorityAccess,
        _ => ResumeCauseValue::MoData,
    };

    Ok(RrcResumeRequestData {
        resume_identity,
        resume_mac_i,
        resume_cause,
    })
}

fn bitvec_to_u32(bv: &BitVec<u8, Msb0>) -> u32 {
    let mut value: u32 = 0;
    for bit in bv.iter() {
        value = (value << 1) | (*bit as u32);
    }
    value
}

fn bitvec_to_u16(bv: &BitVec<u8, Msb0>) -> u16 {
    let mut value: u16 = 0;
    for bit in bv.iter() {
        value = (value << 1) | (*bit as u16);
    }
    value
}

// ============================================================================
// RRC Resume Complete
// ============================================================================

/// Parameters for building an RRC Resume Complete message
#[derive(Debug, Clone)]
pub struct RrcResumeCompleteParams {
    /// RRC Transaction Identifier (0-3)
    pub rrc_transaction_id: u8,
    /// Dedicated NAS Message (optional)
    pub dedicated_nas_message: Option<Vec<u8>>,
    /// Selected PLMN Identity (optional, 1-12)
    pub selected_plmn_identity: Option<u8>,
}

/// Parsed RRC Resume Complete data
#[derive(Debug, Clone)]
pub struct RrcResumeCompleteData {
    /// RRC Transaction Identifier
    pub rrc_transaction_id: u8,
}

/// Build an RRC Resume Complete message
pub fn build_rrc_resume_complete(
    params: &RrcResumeCompleteParams,
) -> Result<UL_DCCH_Message, RrcResumeError> {
    if params.rrc_transaction_id > 3 {
        return Err(RrcResumeError::InvalidFieldValue(
            "RRC Transaction ID must be 0-3".to_string(),
        ));
    }

    let dedicated_nas_message = params
        .dedicated_nas_message
        .as_ref()
        .map(|m| DedicatedNAS_Message(m.clone()));

    let selected_plmn_identity = params
        .selected_plmn_identity
        .map(RRCResumeComplete_IEsSelectedPLMN_Identity);

    let ies = RRCResumeComplete_IEs {
        dedicated_nas_message,
        selected_plmn_identity,
        uplink_tx_direct_current_list: None,
        late_non_critical_extension: None,
        non_critical_extension: None,
    };

    let rrc_resume_complete = RRCResumeComplete {
        rrc_transaction_identifier: RRC_TransactionIdentifier(params.rrc_transaction_id),
        critical_extensions: RRCResumeCompleteCriticalExtensions::RrcResumeComplete(ies),
    };

    let message_type = UL_DCCH_MessageType::C1(UL_DCCH_MessageType_c1::RrcResumeComplete(
        rrc_resume_complete,
    ));

    Ok(UL_DCCH_Message { message: message_type })
}

/// Parse an RRC Resume Complete from a UL-DCCH message
pub fn parse_rrc_resume_complete(
    msg: &UL_DCCH_Message,
) -> Result<RrcResumeCompleteData, RrcResumeError> {
    let complete = match &msg.message {
        UL_DCCH_MessageType::C1(c1) => match c1 {
            UL_DCCH_MessageType_c1::RrcResumeComplete(c) => c,
            _ => {
                return Err(RrcResumeError::InvalidMessageType {
                    expected: "RRCResumeComplete".to_string(),
                    actual: "other c1 message".to_string(),
                })
            }
        },
        _ => {
            return Err(RrcResumeError::InvalidMessageType {
                expected: "c1".to_string(),
                actual: "messageClassExtension".to_string(),
            })
        }
    };

    match &complete.critical_extensions {
        RRCResumeCompleteCriticalExtensions::RrcResumeComplete(_) => {}
        RRCResumeCompleteCriticalExtensions::CriticalExtensionsFuture(_) => {
            return Err(RrcResumeError::InvalidMessageType {
                expected: "rrcResumeComplete".to_string(),
                actual: "criticalExtensionsFuture".to_string(),
            })
        }
    };

    Ok(RrcResumeCompleteData {
        rrc_transaction_id: complete.rrc_transaction_identifier.0,
    })
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Build and encode an RRC Resume Request to bytes
pub fn encode_rrc_resume_request(
    params: &RrcResumeRequestParams,
) -> Result<Vec<u8>, RrcResumeError> {
    let msg = build_rrc_resume_request(params)?;
    Ok(encode_rrc(&msg)?)
}

/// Decode and parse an RRC Resume Request from bytes
pub fn decode_rrc_resume_request(
    bytes: &[u8],
) -> Result<RrcResumeRequestData, RrcResumeError> {
    let msg: UL_CCCH_Message = decode_rrc(bytes)?;
    parse_rrc_resume_request(&msg)
}

/// Build and encode an RRC Resume Complete to bytes
pub fn encode_rrc_resume_complete(
    params: &RrcResumeCompleteParams,
) -> Result<Vec<u8>, RrcResumeError> {
    let msg = build_rrc_resume_complete(params)?;
    Ok(encode_rrc(&msg)?)
}

/// Decode and parse an RRC Resume Complete from bytes
pub fn decode_rrc_resume_complete(
    bytes: &[u8],
) -> Result<RrcResumeCompleteData, RrcResumeError> {
    let msg: UL_DCCH_Message = decode_rrc(bytes)?;
    parse_rrc_resume_complete(&msg)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_resume_request_params() -> RrcResumeRequestParams {
        RrcResumeRequestParams {
            resume_identity: 0x123456, // 24 bits max
            resume_mac_i: 0xABCD,
            resume_cause: ResumeCauseValue::MoData,
        }
    }

    #[test]
    fn test_build_rrc_resume_request() {
        let params = create_test_resume_request_params();
        let result = build_rrc_resume_request(&params);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_rrc_resume_request() {
        let params = create_test_resume_request_params();
        let msg = build_rrc_resume_request(&params).unwrap();
        let data = parse_rrc_resume_request(&msg).unwrap();

        assert_eq!(data.resume_identity, params.resume_identity);
        assert_eq!(data.resume_mac_i, params.resume_mac_i);
        assert_eq!(data.resume_cause, params.resume_cause);
    }

    #[test]
    fn test_encode_decode_rrc_resume_request() {
        let params = create_test_resume_request_params();
        let encoded = encode_rrc_resume_request(&params).expect("Failed to encode");
        assert!(!encoded.is_empty());

        let decoded = decode_rrc_resume_request(&encoded).expect("Failed to decode");
        assert_eq!(decoded.resume_identity, params.resume_identity);
        assert_eq!(decoded.resume_cause, params.resume_cause);
    }

    #[test]
    fn test_build_rrc_resume_complete() {
        let params = RrcResumeCompleteParams {
            rrc_transaction_id: 1,
            dedicated_nas_message: Some(vec![0x7E, 0x00]),
            selected_plmn_identity: Some(1),
        };
        let result = build_rrc_resume_complete(&params);
        assert!(result.is_ok());
    }

    #[test]
    fn test_encode_decode_rrc_resume_complete() {
        let params = RrcResumeCompleteParams {
            rrc_transaction_id: 2,
            dedicated_nas_message: None,
            selected_plmn_identity: None,
        };
        let encoded = encode_rrc_resume_complete(&params).expect("Failed to encode");
        let decoded = decode_rrc_resume_complete(&encoded).expect("Failed to decode");
        assert_eq!(decoded.rrc_transaction_id, 2);
    }

    #[test]
    fn test_invalid_transaction_id() {
        let params = RrcResumeCompleteParams {
            rrc_transaction_id: 5,
            dedicated_nas_message: None,
            selected_plmn_identity: None,
        };
        assert!(build_rrc_resume_complete(&params).is_err());
    }

    #[test]
    fn test_all_resume_causes() {
        let causes = [
            ResumeCauseValue::Emergency,
            ResumeCauseValue::HighPriorityAccess,
            ResumeCauseValue::MtAccess,
            ResumeCauseValue::MoSignalling,
            ResumeCauseValue::MoData,
            ResumeCauseValue::MoVoiceCall,
            ResumeCauseValue::MoVideoCall,
            ResumeCauseValue::MoSms,
            ResumeCauseValue::RnaUpdate,
            ResumeCauseValue::MpsPriorityAccess,
            ResumeCauseValue::McsPriorityAccess,
        ];

        for cause in causes {
            let params = RrcResumeRequestParams {
                resume_identity: 0x000001,
                resume_mac_i: 0x0001,
                resume_cause: cause,
            };
            let msg = build_rrc_resume_request(&params).unwrap();
            let data = parse_rrc_resume_request(&msg).unwrap();
            assert_eq!(data.resume_cause, cause);
        }
    }
}
