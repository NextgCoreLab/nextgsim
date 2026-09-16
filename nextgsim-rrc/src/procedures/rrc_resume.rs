//! RRC Resume Procedure
//!
//! Implements the RRC Resume procedure as defined in 3GPP TS 38.331 Section 5.3.13.
//! This procedure is used when UE transitions from `RRC_INACTIVE` to `RRC_CONNECTED`,
//! reusing the previously established UE context.
//!
//! The procedure consists of three messages:
//! 1. `RRCResumeRequest` - UE -> gNB: Request to resume RRC connection
//! 2. `RRCResume` - gNB -> UE: Network response with updated configuration
//! 3. `RRCResumeComplete` - UE -> gNB: Confirmation of resume

use crate::codec::generated::*;
use crate::codec::{decode_rrc, encode_rrc, RrcCodecError};
use crate::procedures::rrc_reestablishment::mac_i_lsb16;
use crate::procedures::rrc_setup::srb1_rrc_setup_params;
use crate::procedures::suspend_config::CELL_IDENTITY_BITS;
use bitvec::prelude::*;
use thiserror::Error;

/// The `resumeMAC-I` a UE puts in its `RRCResumeRequest` and the gNB verifies
/// (TS 38.331 §5.3.13.3).
///
/// The 16 least significant bits of a MAC-I over the UPER-encoded
/// `VarResumeMAC-Input` — source PCell PCI, the 36-bit identity of the cell being
/// resumed on, and the source C-RNTI — computed with `K_RRCint` and the source
/// PCell's NIA, with COUNT, BEARER and DIRECTION all binary ones.
///
/// # Why this is here and not in either endpoint
///
/// The UE computes it and the **gNB verifies it**, so the two must agree byte for
/// byte over the same `VarResumeMAC-Input` encoding (issue #38). Before this, the UE
/// had its own copy and the gNB had none — which is exactly how a gNB comes to accept
/// a resume it never authenticated. Same reasoning that put `compute_short_mac_i`
/// beside it.
///
/// The gNB must supply the values from the **stored** context — the C-RNTI and PCI
/// the UE had when it was suspended — not from the resuming cell, or every MAC fails.
pub fn compute_resume_mac_i(
    k_rrc_int: &[u8; 16],
    integrity_alg_id: u8,
    source_c_rnti: u16,
    source_phys_cell_id: u16,
    target_cell_identity: u64,
) -> Result<u16, RrcResumeError> {
    let mut cell_id_bv: BitVec<u8, Msb0> = BitVec::with_capacity(CELL_IDENTITY_BITS);
    for i in (0..CELL_IDENTITY_BITS).rev() {
        cell_id_bv.push((target_cell_identity >> i) & 1 == 1);
    }

    let input = VarResumeMAC_Input {
        source_phys_cell_id: PhysCellId(source_phys_cell_id),
        target_cell_identity: CellIdentity(cell_id_bv),
        source_c_rnti: RNTI_Value(source_c_rnti),
    };
    let encoded = encode_rrc(&input)?;

    mac_i_lsb16(k_rrc_int, integrity_alg_id, &encoded).map_err(|e| {
        RrcResumeError::InvalidFieldValue(format!("resumeMAC-I could not be computed: {e}"))
    })
}

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

    let message_type =
        UL_CCCH_MessageType::C1(UL_CCCH_MessageType_c1::RrcResumeRequest(rrc_resume_request));

    Ok(UL_CCCH_Message {
        message: message_type,
    })
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

fn bitvec_to_u64(bv: &BitVec<u8, Msb0>) -> u64 {
    let mut value: u64 = 0;
    for bit in bv.iter() {
        value = (value << 1) | (*bit as u64);
    }
    value
}

// ============================================================================
// RRC Resume Request 1 (full 40-bit I-RNTI, TS 38.331 §6.2.2)
// ============================================================================

/// Parameters for building an RRC Resume Request 1 (UL-CCCH1)
///
/// Used when SIB1 signals `useFullResumeID`: the UE identifies itself with
/// the full 40-bit I-RNTI instead of the 24-bit short form.
#[derive(Debug, Clone)]
pub struct RrcResumeRequest1Params {
    /// Resume Identity (full I-RNTI, 40 bits, max value 0xFF_FFFF_FFFF)
    pub resume_identity: u64,
    /// Resume MAC-I (16 bits)
    pub resume_mac_i: u16,
    /// Resume cause
    pub resume_cause: ResumeCauseValue,
}

/// Parsed RRC Resume Request 1 data
#[derive(Debug, Clone)]
pub struct RrcResumeRequest1Data {
    /// Resume Identity (full I-RNTI, 40 bits)
    pub resume_identity: u64,
    /// Resume MAC-I
    pub resume_mac_i: u16,
    /// Resume cause
    pub resume_cause: ResumeCauseValue,
}

fn build_resume_cause(value: ResumeCauseValue) -> ResumeCause {
    match value {
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
    }
}

fn parse_resume_cause(cause: &ResumeCause) -> ResumeCauseValue {
    match cause.0 {
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
    }
}

/// Build an RRC Resume Request 1 message (UL-CCCH1)
pub fn build_rrc_resume_request1(
    params: &RrcResumeRequest1Params,
) -> Result<UL_CCCH1_Message, RrcResumeError> {
    if params.resume_identity > 0xFF_FFFF_FFFF {
        return Err(RrcResumeError::InvalidFieldValue(
            "Full I-RNTI must fit in 40 bits".to_string(),
        ));
    }

    // Build Resume Identity (full I-RNTI, 40 bits)
    let mut resume_id_bv: BitVec<u8, Msb0> = BitVec::with_capacity(40);
    for i in (0..40).rev() {
        resume_id_bv.push((params.resume_identity >> i) & 1 == 1);
    }

    // Build Resume MAC-I (16 bits)
    let mut resume_mac_i_bv: BitVec<u8, Msb0> = BitVec::with_capacity(16);
    for i in (0..16).rev() {
        resume_mac_i_bv.push((params.resume_mac_i >> i) & 1 == 1);
    }

    // Build spare bit (1 bit)
    let mut spare_bv: BitVec<u8, Msb0> = BitVec::new();
    spare_bv.push(false);

    let ies = RRCResumeRequest1_IEs {
        resume_identity: I_RNTI_Value(resume_id_bv),
        resume_mac_i: RRCResumeRequest1_IEsResumeMAC_I(resume_mac_i_bv),
        resume_cause: build_resume_cause(params.resume_cause),
        spare: RRCResumeRequest1_IEsSpare(spare_bv),
    };

    Ok(UL_CCCH1_Message {
        message: UL_CCCH1_MessageType::C1(UL_CCCH1_MessageType_c1::RrcResumeRequest1(
            RRCResumeRequest1 {
                rrc_resume_request1: ies,
            },
        )),
    })
}

/// Parse an RRC Resume Request 1 from a UL-CCCH1 message
pub fn parse_rrc_resume_request1(
    msg: &UL_CCCH1_Message,
) -> Result<RrcResumeRequest1Data, RrcResumeError> {
    let request = match &msg.message {
        UL_CCCH1_MessageType::C1(UL_CCCH1_MessageType_c1::RrcResumeRequest1(req)) => req,
        UL_CCCH1_MessageType::C1(_) => {
            return Err(RrcResumeError::InvalidMessageType {
                expected: "RRCResumeRequest1".to_string(),
                actual: "spare c1 message".to_string(),
            })
        }
        _ => {
            return Err(RrcResumeError::InvalidMessageType {
                expected: "c1".to_string(),
                actual: "messageClassExtension".to_string(),
            })
        }
    };

    let ies = &request.rrc_resume_request1;

    Ok(RrcResumeRequest1Data {
        resume_identity: bitvec_to_u64(&ies.resume_identity.0),
        resume_mac_i: bitvec_to_u16(&ies.resume_mac_i.0),
        resume_cause: parse_resume_cause(&ies.resume_cause),
    })
}

/// Build and encode an RRC Resume Request 1 to bytes
pub fn encode_rrc_resume_request1(
    params: &RrcResumeRequest1Params,
) -> Result<Vec<u8>, RrcResumeError> {
    let msg = build_rrc_resume_request1(params)?;
    Ok(encode_rrc(&msg)?)
}

/// Decode and parse an RRC Resume Request 1 from bytes
pub fn decode_rrc_resume_request1(bytes: &[u8]) -> Result<RrcResumeRequest1Data, RrcResumeError> {
    let msg: UL_CCCH1_Message = decode_rrc(bytes)?;
    parse_rrc_resume_request1(&msg)
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
    /// `dedicatedNAS-Message`, when the UE piggybacked one (TS 38.331 §6.2.2)
    pub dedicated_nas_message: Option<Vec<u8>>,
    /// `selectedPLMN-Identity` (1..12), when present
    pub selected_plmn_identity: Option<u8>,
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

    Ok(UL_DCCH_Message {
        message: message_type,
    })
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

    let ies = match &complete.critical_extensions {
        RRCResumeCompleteCriticalExtensions::RrcResumeComplete(ies) => ies,
        RRCResumeCompleteCriticalExtensions::CriticalExtensionsFuture(_) => {
            return Err(RrcResumeError::InvalidMessageType {
                expected: "rrcResumeComplete".to_string(),
                actual: "criticalExtensionsFuture".to_string(),
            })
        }
    };

    Ok(RrcResumeCompleteData {
        rrc_transaction_id: complete.rrc_transaction_identifier.0,
        // The NAS the UE piggybacks on the Complete when the resume was triggered
        // by something it had to send (TS 38.331 §5.3.13.4, issue #38). It was
        // built by the encoder and DROPPED here, which only went unnoticed while
        // the peer recovered it by slicing the bespoke framing (issue #151).
        dedicated_nas_message: ies.dedicated_nas_message.as_ref().map(|m| m.0.clone()),
        selected_plmn_identity: ies.selected_plmn_identity.as_ref().map(|p| p.0),
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
pub fn decode_rrc_resume_request(bytes: &[u8]) -> Result<RrcResumeRequestData, RrcResumeError> {
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
pub fn decode_rrc_resume_complete(bytes: &[u8]) -> Result<RrcResumeCompleteData, RrcResumeError> {
    let msg: UL_DCCH_Message = decode_rrc(bytes)?;
    parse_rrc_resume_complete(&msg)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // RRCResume golden bytes (issue #107, criterion 6)
    // ========================================================================

    /// Hand-derived `RRCResume` on DL-DCCH, tid 0, carrying the SRB1
    /// configuration with `fullConfig` set.
    ///
    /// Derivation of the framing from `tools/rrc-15.6.0.asn1`:
    ///
    /// ```text
    /// bit 0      DL-DCCH-MessageType CHOICE, 2 alternatives, no extension
    ///            marker -> 1 bit. c1 = 0
    /// bits 1..4  c1 CHOICE, 16 alternatives -> 4 bits. rrcResume = 1 -> 0001
    /// bits 5..6  RRC-TransactionIdentifier, INTEGER (0..3) -> 2 bits. tid 0 -> 00
    /// bit 7      RRCResume criticalExtensions CHOICE, 2 alternatives -> 1 bit.
    ///            rrcResume = 0
    ///            => byte 0 = 0b0_0001_00_0 = 0x08
    /// bits 8..13 RRCResume-IEs preamble, 6 OPTIONAL fields -> 6 presence bits:
    ///            radioBearerConfig=1, masterCellGroup=1, measConfig=0,
    ///            fullConfig=1, lateNonCriticalExtension=0,
    ///            nonCriticalExtension=0  -> 110100
    ///            byte 1 = 0b110100_01 = 0xD1, the trailing 01 being the first two
    ///            bits of the embedded RadioBearerConfig
    /// ```
    ///
    /// The tail is the SRB1 `RadioBearerConfig` and `CellGroupConfig` whose own
    /// bit-by-bit derivation is in `rrc_setup.rs`'s `golden_rrc_setup_srb1_bytes`
    /// — this builder reuses `srb1_rrc_setup_params`, so quoting that derivation
    /// rather than repeating it is deliberate.
    const GOLDEN_RRC_RESUME_TID0: [u8; 9] = [0x08, 0xD1, 0x00, 0x00, 0x88, 0x00, 0x10, 0x00, 0x00];

    #[test]
    fn golden_rrc_resume_bytes() {
        // The encoder output must equal the hand-derived literal. NOT a round trip:
        // a round trip through one codec passes however wrong the layout is.
        let params = fresh_rrc_resume_params(0).expect("resume params");
        let bytes = encode_rrc_resume(&params).expect("encode RRCResume");
        assert_eq!(
            bytes,
            GOLDEN_RRC_RESUME_TID0.to_vec(),
            "RRCResume(SRB1, fullConfig, tid 0) must match the hand-derived UPER bytes"
        );

        // The tid occupies bits 5..6 of byte 0 ONLY. tid 2 -> 0b0_0001_10_0 = 0x0C,
        // every other byte identical. This is what makes the derivation above a
        // derivation rather than a capture: it predicts the change.
        let params_tid2 = fresh_rrc_resume_params(2).expect("resume params tid 2");
        let bytes_tid2 = encode_rrc_resume(&params_tid2).expect("encode tid 2");
        let mut expected = GOLDEN_RRC_RESUME_TID0;
        expected[0] = 0x0C;
        assert_eq!(bytes_tid2, expected.to_vec());
    }

    #[test]
    fn golden_rrc_resume_cross_decode() {
        let data = decode_rrc_resume(&GOLDEN_RRC_RESUME_TID0).expect("decode RRCResume");
        assert_eq!(data.rrc_transaction_id, 0);
        assert!(
            data.full_config,
            "fullConfig must be set: this gNB holds no suspended context, so the \
             resumed UE applies the configuration whole rather than as a delta"
        );
        let rbc = data
            .radio_bearer_config
            .expect("radioBearerConfig must be carried");
        let srbs = rbc
            .srb_to_add_mod_list
            .expect("the resumed UE gets SRB1 back");
        assert_eq!(srbs.0.len(), 1);
        assert_eq!(srbs.0[0].srb_identity.0, 1);
        assert!(
            data.master_cell_group.is_some(),
            "masterCellGroup must be carried"
        );
    }

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

    #[test]
    fn test_resume_request1_full_irnti_roundtrip() {
        // Full 40-bit I-RNTI exercising all bytes
        let params = RrcResumeRequest1Params {
            resume_identity: 0xAB_CDEF_0123,
            resume_mac_i: 0x5A5A,
            resume_cause: ResumeCauseValue::RnaUpdate,
        };
        let bytes = encode_rrc_resume_request1(&params).unwrap();
        let data = decode_rrc_resume_request1(&bytes).unwrap();
        assert_eq!(data.resume_identity, 0xAB_CDEF_0123);
        assert_eq!(data.resume_mac_i, 0x5A5A);
        assert_eq!(data.resume_cause, ResumeCauseValue::RnaUpdate);
    }

    #[test]
    fn test_resume_request1_rejects_oversized_irnti() {
        let params = RrcResumeRequest1Params {
            resume_identity: 0x100_0000_0000, // 41 bits
            resume_mac_i: 0,
            resume_cause: ResumeCauseValue::MoData,
        };
        assert!(build_rrc_resume_request1(&params).is_err());
    }

    #[test]
    fn test_resume_request1_rejects_truncated() {
        let params = RrcResumeRequest1Params {
            resume_identity: 0xFF_FFFF_FFFF,
            resume_mac_i: 0xFFFF,
            resume_cause: ResumeCauseValue::MoData,
        };
        let bytes = encode_rrc_resume_request1(&params).unwrap();
        assert!(decode_rrc_resume_request1(&bytes[..3]).is_err());
    }
}

// ============================================================================
// RRCResume (gNB -> UE, DL-DCCH) — TS 38.331 §6.2.2
// ============================================================================

/// Parameters for the gNB-side `RRCResume` (issue #107, criterion 2).
#[derive(Debug, Clone)]
pub struct RrcResumeParams {
    /// RRC-TransactionIdentifier (0..3)
    pub rrc_transaction_id: u8,
    /// The `radioBearerConfig` the resumed UE is to apply, already UPER encoded.
    pub radio_bearer_config: Vec<u8>,
    /// The `masterCellGroup`, already UPER encoded. Carried as an OCTET STRING
    /// CONTAINING a `CellGroupConfig`, the same shape RRCSetup uses.
    pub master_cell_group: Vec<u8>,
    /// `fullConfig`: the UE is to apply a FULL configuration rather than a delta
    /// on top of a stored one (TS 38.331 §5.3.13.4).
    pub full_config: bool,
}

/// Parsed `RRCResume`.
#[derive(Debug, Clone)]
pub struct RrcResumeData {
    /// RRC-TransactionIdentifier
    pub rrc_transaction_id: u8,
    /// The decoded `radioBearerConfig`, when present
    pub radio_bearer_config: Option<RadioBearerConfig>,
    /// The raw `masterCellGroup` octets, when present
    pub master_cell_group: Option<Vec<u8>>,
    /// Whether `fullConfig` was set
    pub full_config: bool,
}

/// Builds an `RRCResume` on **DL-DCCH** (TS 38.331 §6.2.2, issue #107).
///
/// # DL-DCCH, not DL-CCCH
///
/// The hand-rolled builder this replaces produced what its own comment called a
/// "DL-CCCH-Message with RRCResume". `RRCResume` is a **DL-DCCH** message: it
/// rides SRB1, which the suspended UE already had. `RRCResumeRequest` is the
/// CCCH half (UL-CCCH), and mixing the two up would put the resume on a channel
/// the UE does not read it on.
///
/// # The bearer configuration is mandatory in practice
///
/// `radioBearerConfig` and `masterCellGroup` are both OPTIONAL in the ASN.1
/// (they are a *delta* on the stored configuration in a normal resume). This
/// builder requires them, because this gNB stores **no** suspended context — see
/// the note on `full_config`.
pub fn build_rrc_resume(params: &RrcResumeParams) -> Result<DL_DCCH_Message, RrcResumeError> {
    if params.rrc_transaction_id > 3 {
        return Err(RrcResumeError::InvalidFieldValue(
            "RRC-TransactionIdentifier is INTEGER(0..3)".to_string(),
        ));
    }
    if params.radio_bearer_config.is_empty() || params.master_cell_group.is_empty() {
        return Err(RrcResumeError::InvalidFieldValue(
            "RRCResume must carry a radioBearerConfig and a masterCellGroup: this \
             gNB stores no suspended context, so a resumed UE has nothing to apply \
             a delta to"
                .to_string(),
        ));
    }

    let radio_bearer_config: RadioBearerConfig = decode_rrc(&params.radio_bearer_config)?;

    let ies = RRCResume_IEs {
        radio_bearer_config: Some(radio_bearer_config),
        master_cell_group: Some(RRCResume_IEsMasterCellGroup(
            params.master_cell_group.clone(),
        )),
        meas_config: None,
        full_config: if params.full_config {
            Some(RRCResume_IEsFullConfig(RRCResume_IEsFullConfig::TRUE))
        } else {
            None
        },
        late_non_critical_extension: None,
        non_critical_extension: None,
    };

    Ok(DL_DCCH_Message {
        message: DL_DCCH_MessageType::C1(DL_DCCH_MessageType_c1::RrcResume(RRCResume {
            rrc_transaction_identifier: RRC_TransactionIdentifier(params.rrc_transaction_id),
            critical_extensions: RRCResumeCriticalExtensions::RrcResume(ies),
        })),
    })
}

/// Builds and encodes an `RRCResume` to UPER bytes.
pub fn encode_rrc_resume(params: &RrcResumeParams) -> Result<Vec<u8>, RrcResumeError> {
    let msg = build_rrc_resume(params)?;
    Ok(encode_rrc(&msg)?)
}

/// Parses an `RRCResume` from a DL-DCCH message.
pub fn parse_rrc_resume(msg: &DL_DCCH_Message) -> Result<RrcResumeData, RrcResumeError> {
    let DL_DCCH_MessageType::C1(DL_DCCH_MessageType_c1::RrcResume(resume)) = &msg.message else {
        return Err(RrcResumeError::InvalidMessageType {
            expected: "RRCResume".to_string(),
            actual: format!("{:?}", msg.message),
        });
    };
    let RRCResumeCriticalExtensions::RrcResume(ies) = &resume.critical_extensions else {
        return Err(RrcResumeError::InvalidMessageType {
            expected: "rrcResume critical extension".to_string(),
            actual: "criticalExtensionsFuture".to_string(),
        });
    };
    Ok(RrcResumeData {
        rrc_transaction_id: resume.rrc_transaction_identifier.0,
        radio_bearer_config: ies.radio_bearer_config.clone(),
        master_cell_group: ies.master_cell_group.as_ref().map(|m| m.0.clone()),
        full_config: ies.full_config.is_some(),
    })
}

/// Decodes and parses an `RRCResume` from UPER bytes.
pub fn decode_rrc_resume(bytes: &[u8]) -> Result<RrcResumeData, RrcResumeError> {
    let msg: DL_DCCH_Message = decode_rrc(bytes)?;
    parse_rrc_resume(&msg)
}

/// The `RRCResume` a resumed UE gets (issue #107, criterion 2 — "the 'what does a
/// resumed UE get' decision"; revisited by issue #38).
///
/// # The decision: a FRESH configuration with `fullConfig` set
///
/// TS 38.331 §5.3.13.4 lets `RRCResume` carry a delta on the configuration the UE
/// stored when it was suspended. This gNB does not, and since issue #38 that is a
/// **choice** rather than a consequence: RRC_INACTIVE is now reachable and a
/// suspended context IS stored (`SuspendedUeContext`), but what it stores is the AS
/// security context and the RAN Notification Area — not the radio bearer
/// configuration, because the gNB's RRC layer does not track DRBs at all.
///
/// So the resumed UE is given the **same SRB1 configuration RRCSetup builds**,
/// with `fullConfig` **set**, which is precisely what that IE is for: it tells the
/// UE to release its stored configuration and apply this one whole. The
/// alternative — omitting `fullConfig` and sending the same bearers as a "delta" —
/// would ask the UE to merge them into a stored configuration that does not exist,
/// and a UE that had a real one would end up with a mixture neither side intended.
///
/// This is honest rather than complete: a UE resumed this way loses its bearers, so
/// the user plane needs re-establishing after a resume. Storing the configuration
/// alongside the security context is what a delta would need, and that is the next
/// step here — not a different `fullConfig` decision.
pub fn fresh_rrc_resume_params(rrc_transaction_id: u8) -> Result<RrcResumeParams, RrcResumeError> {
    let setup = srb1_rrc_setup_params(rrc_transaction_id)
        .map_err(|e| RrcResumeError::InvalidFieldValue(e.to_string()))?;
    Ok(RrcResumeParams {
        rrc_transaction_id,
        radio_bearer_config: setup.radio_bearer_config,
        master_cell_group: setup.master_cell_group,
        full_config: true,
    })
}
