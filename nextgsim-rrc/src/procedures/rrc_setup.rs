//! RRC Setup Procedure
//!
//! Implements the RRC Setup procedure as defined in 3GPP TS 38.331 Section 5.3.3.
//! This procedure is used to establish an RRC connection between the UE and the network.
//!
//! The procedure consists of three messages:
//! 1. `RRCSetupRequest` - UE → gNB: Initial request to establish RRC connection
//! 2. `RRCSetup` - gNB → UE: Network response with radio bearer configuration
//! 3. `RRCSetupComplete` - UE → gNB: Confirmation with NAS message

use crate::codec::generated::*;
use crate::codec::{decode_rrc, encode_rrc, RrcCodecError};
use bitvec::prelude::*;
use thiserror::Error;

/// Errors that can occur during RRC Setup procedures
#[derive(Debug, Error)]
pub enum RrcSetupError {
    /// Codec error during encoding/decoding
    #[error("Codec error: {0}")]
    CodecError(#[from] RrcCodecError),

    /// Invalid message type received
    #[error("Invalid message type: expected {expected}, got {actual}")]
    InvalidMessageType { expected: String, actual: String },

    /// Missing mandatory field
    #[error("Missing mandatory field: {0}")]
    MissingMandatoryField(String),

    /// Invalid field value
    #[error("Invalid field value: {0}")]
    InvalidFieldValue(String),
}

/// Establishment cause for RRC Setup Request
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RrcEstablishmentCause {
    Emergency,
    HighPriorityAccess,
    MtAccess,
    MoSignalling,
    MoData,
    MoVoiceCall,
    MoVideoCall,
    MoSms,
    MpsPriorityAccess,
    McsPriorityAccess,
}

impl From<RrcEstablishmentCause> for EstablishmentCause {
    fn from(cause: RrcEstablishmentCause) -> Self {
        let value = match cause {
            RrcEstablishmentCause::Emergency => EstablishmentCause::EMERGENCY,
            RrcEstablishmentCause::HighPriorityAccess => EstablishmentCause::HIGH_PRIORITY_ACCESS,
            RrcEstablishmentCause::MtAccess => EstablishmentCause::MT_ACCESS,
            RrcEstablishmentCause::MoSignalling => EstablishmentCause::MO_SIGNALLING,
            RrcEstablishmentCause::MoData => EstablishmentCause::MO_DATA,
            RrcEstablishmentCause::MoVoiceCall => EstablishmentCause::MO_VOICE_CALL,
            RrcEstablishmentCause::MoVideoCall => EstablishmentCause::MO_VIDEO_CALL,
            RrcEstablishmentCause::MoSms => EstablishmentCause::MO_SMS,
            RrcEstablishmentCause::MpsPriorityAccess => EstablishmentCause::MPS_PRIORITY_ACCESS,
            RrcEstablishmentCause::McsPriorityAccess => EstablishmentCause::MCS_PRIORITY_ACCESS,
        };
        EstablishmentCause(value)
    }
}

impl TryFrom<EstablishmentCause> for RrcEstablishmentCause {
    type Error = RrcSetupError;

    fn try_from(cause: EstablishmentCause) -> Result<Self, Self::Error> {
        match cause.0 {
            EstablishmentCause::EMERGENCY => Ok(RrcEstablishmentCause::Emergency),
            EstablishmentCause::HIGH_PRIORITY_ACCESS => Ok(RrcEstablishmentCause::HighPriorityAccess),
            EstablishmentCause::MT_ACCESS => Ok(RrcEstablishmentCause::MtAccess),
            EstablishmentCause::MO_SIGNALLING => Ok(RrcEstablishmentCause::MoSignalling),
            EstablishmentCause::MO_DATA => Ok(RrcEstablishmentCause::MoData),
            EstablishmentCause::MO_VOICE_CALL => Ok(RrcEstablishmentCause::MoVoiceCall),
            EstablishmentCause::MO_VIDEO_CALL => Ok(RrcEstablishmentCause::MoVideoCall),
            EstablishmentCause::MO_SMS => Ok(RrcEstablishmentCause::MoSms),
            EstablishmentCause::MPS_PRIORITY_ACCESS => Ok(RrcEstablishmentCause::MpsPriorityAccess),
            EstablishmentCause::MCS_PRIORITY_ACCESS => Ok(RrcEstablishmentCause::McsPriorityAccess),
            _ => Err(RrcSetupError::InvalidFieldValue(format!(
                "Unknown establishment cause: {}",
                cause.0
            ))),
        }
    }
}

/// UE Identity for RRC Setup Request
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UeIdentity {
    /// 5G-S-TMSI Part 1 (39 bits)
    Ng5gSTmsiPart1(u64),
    /// Random value (39 bits)
    RandomValue(u64),
}

// ============================================================================
// RRC Setup Request
// ============================================================================

/// Parameters for building an RRC Setup Request
#[derive(Debug, Clone)]
pub struct RrcSetupRequestParams {
    /// UE Identity (5G-S-TMSI Part 1 or random value)
    pub ue_identity: UeIdentity,
    /// Establishment cause
    pub establishment_cause: RrcEstablishmentCause,
}

/// Parsed RRC Setup Request data
#[derive(Debug, Clone)]
pub struct RrcSetupRequestData {
    /// UE Identity
    pub ue_identity: UeIdentity,
    /// Establishment cause
    pub establishment_cause: RrcEstablishmentCause,
}

/// Build an RRC Setup Request message
pub fn build_rrc_setup_request(params: &RrcSetupRequestParams) -> Result<UL_CCCH_Message, RrcSetupError> {
    // Build UE Identity
    let ue_identity = match &params.ue_identity {
        UeIdentity::Ng5gSTmsiPart1(value) => {
            let mut bv: BitVec<u8, Msb0> = BitVec::new();
            for i in (0..39).rev() {
                bv.push((value >> i) & 1 == 1);
            }
            InitialUE_Identity::Ng_5G_S_TMSI_Part1(InitialUE_Identity_ng_5G_S_TMSI_Part1(bv))
        }
        UeIdentity::RandomValue(value) => {
            let mut bv: BitVec<u8, Msb0> = BitVec::new();
            for i in (0..39).rev() {
                bv.push((value >> i) & 1 == 1);
            }
            InitialUE_Identity::RandomValue(InitialUE_Identity_randomValue(bv))
        }
    };

    // Build spare bit (1 bit, set to 0)
    let mut spare_bv: BitVec<u8, Msb0> = BitVec::new();
    spare_bv.push(false);

    let rrc_setup_request_ies = RRCSetupRequest_IEs {
        ue_identity,
        establishment_cause: params.establishment_cause.into(),
        spare: RRCSetupRequest_IEsSpare(spare_bv),
    };

    let rrc_setup_request = RRCSetupRequest {
        rrc_setup_request: rrc_setup_request_ies,
    };

    let message_type = UL_CCCH_MessageType::C1(UL_CCCH_MessageType_c1::RrcSetupRequest(rrc_setup_request));

    Ok(UL_CCCH_Message { message: message_type })
}

/// Parse an RRC Setup Request from a UL-CCCH message
pub fn parse_rrc_setup_request(msg: &UL_CCCH_Message) -> Result<RrcSetupRequestData, RrcSetupError> {
    let rrc_setup_request = match &msg.message {
        UL_CCCH_MessageType::C1(c1) => match c1 {
            UL_CCCH_MessageType_c1::RrcSetupRequest(req) => req,
            _ => {
                return Err(RrcSetupError::InvalidMessageType {
                    expected: "RRCSetupRequest".to_string(),
                    actual: "other c1 message".to_string(),
                })
            }
        },
        _ => {
            return Err(RrcSetupError::InvalidMessageType {
                expected: "c1".to_string(),
                actual: "messageClassExtension".to_string(),
            })
        }
    };

    let ies = &rrc_setup_request.rrc_setup_request;

    // Parse UE Identity
    let ue_identity = match &ies.ue_identity {
        InitialUE_Identity::Ng_5G_S_TMSI_Part1(bv) => {
            let value = bitvec_to_u64(&bv.0);
            UeIdentity::Ng5gSTmsiPart1(value)
        }
        InitialUE_Identity::RandomValue(bv) => {
            let value = bitvec_to_u64(&bv.0);
            UeIdentity::RandomValue(value)
        }
    };

    // Parse establishment cause
    let establishment_cause = RrcEstablishmentCause::try_from(ies.establishment_cause.clone())?;

    Ok(RrcSetupRequestData { ue_identity, establishment_cause })
}

/// Helper function to convert `BitVec` to u64
fn bitvec_to_u64(bv: &BitVec<u8, Msb0>) -> u64 {
    let mut value: u64 = 0;
    for bit in bv.iter() {
        value = (value << 1) | (*bit as u64);
    }
    value
}

// ============================================================================
// RRC Setup
// ============================================================================

/// Parameters for building an RRC Setup message
#[derive(Debug, Clone)]
pub struct RrcSetupParams {
    /// RRC Transaction Identifier (0-3)
    pub rrc_transaction_id: u8,
    /// Radio Bearer Configuration (encoded as bytes)
    pub radio_bearer_config: Vec<u8>,
    /// Master Cell Group Configuration (encoded as bytes)
    pub master_cell_group: Vec<u8>,
}

/// Parsed RRC Setup data
#[derive(Debug, Clone)]
pub struct RrcSetupData {
    /// RRC Transaction Identifier
    pub rrc_transaction_id: u8,
    /// Radio Bearer Configuration (raw bytes)
    pub radio_bearer_config: Vec<u8>,
    /// Master Cell Group Configuration (raw bytes)
    pub master_cell_group: Vec<u8>,
}

/// Build an RRC Setup message
pub fn build_rrc_setup(params: &RrcSetupParams) -> Result<DL_CCCH_Message, RrcSetupError> {
    if params.rrc_transaction_id > 3 {
        return Err(RrcSetupError::InvalidFieldValue(
            "RRC Transaction ID must be 0-3".to_string(),
        ));
    }

    // Decode the radio bearer config from bytes
    let radio_bearer_config: RadioBearerConfig = decode_rrc(&params.radio_bearer_config)?;

    let rrc_setup_ies = RRCSetup_IEs {
        radio_bearer_config,
        master_cell_group: RRCSetup_IEsMasterCellGroup(params.master_cell_group.clone()),
        late_non_critical_extension: None,
        non_critical_extension: None,
    };

    let rrc_setup = RRCSetup {
        rrc_transaction_identifier: RRC_TransactionIdentifier(params.rrc_transaction_id),
        critical_extensions: RRCSetupCriticalExtensions::RrcSetup(rrc_setup_ies),
    };

    let message_type = DL_CCCH_MessageType::C1(DL_CCCH_MessageType_c1::RrcSetup(rrc_setup));

    Ok(DL_CCCH_Message { message: message_type })
}

/// Parse an RRC Setup from a DL-CCCH message
pub fn parse_rrc_setup(msg: &DL_CCCH_Message) -> Result<RrcSetupData, RrcSetupError> {
    let rrc_setup = match &msg.message {
        DL_CCCH_MessageType::C1(c1) => match c1 {
            DL_CCCH_MessageType_c1::RrcSetup(setup) => setup,
            _ => {
                return Err(RrcSetupError::InvalidMessageType {
                    expected: "RRCSetup".to_string(),
                    actual: "other c1 message".to_string(),
                })
            }
        },
        _ => {
            return Err(RrcSetupError::InvalidMessageType {
                expected: "c1".to_string(),
                actual: "messageClassExtension".to_string(),
            })
        }
    };

    let ies = match &rrc_setup.critical_extensions {
        RRCSetupCriticalExtensions::RrcSetup(ies) => ies,
        RRCSetupCriticalExtensions::CriticalExtensionsFuture(_) => {
            return Err(RrcSetupError::InvalidMessageType {
                expected: "rrcSetup".to_string(),
                actual: "criticalExtensionsFuture".to_string(),
            })
        }
    };

    // Encode radio bearer config back to bytes for storage
    let radio_bearer_config = encode_rrc(&ies.radio_bearer_config)?;

    Ok(RrcSetupData {
        rrc_transaction_id: rrc_setup.rrc_transaction_identifier.0,
        radio_bearer_config,
        master_cell_group: ies.master_cell_group.0.clone(),
    })
}

// ============================================================================
// RRC Setup Complete
// ============================================================================

/// GUAMI type for RRC Setup Complete
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuamiType {
    Native,
    Mapped,
}

/// S-NSSAI (Single Network Slice Selection Assistance Information)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SNssai {
    /// Slice/Service Type (SST) - 8 bits
    pub sst: u8,
    /// Slice Differentiator (SD) - 24 bits, optional
    pub sd: Option<u32>,
}

/// 5G-S-TMSI value for RRC Setup Complete
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ng5gSTmsiValue {
    /// Full 5G-S-TMSI (48 bits)
    Full(u64),
    /// 5G-S-TMSI Part 2 (9 bits)
    Part2(u16),
}

/// Parameters for building an RRC Setup Complete message
#[derive(Debug, Clone)]
pub struct RrcSetupCompleteParams {
    /// RRC Transaction Identifier (0-3)
    pub rrc_transaction_id: u8,
    /// Selected PLMN Identity index (1-12)
    pub selected_plmn_identity: u8,
    /// GUAMI type (optional)
    pub guami_type: Option<GuamiType>,
    /// S-NSSAI list (optional)
    pub s_nssai_list: Option<Vec<SNssai>>,
    /// Dedicated NAS Message
    pub dedicated_nas_message: Vec<u8>,
    /// 5G-S-TMSI value (optional)
    pub ng_5g_s_tmsi_value: Option<Ng5gSTmsiValue>,
}

/// Parsed RRC Setup Complete data
#[derive(Debug, Clone)]
pub struct RrcSetupCompleteData {
    /// RRC Transaction Identifier
    pub rrc_transaction_id: u8,
    /// Selected PLMN Identity index
    pub selected_plmn_identity: u8,
    /// GUAMI type
    pub guami_type: Option<GuamiType>,
    /// S-NSSAI list
    pub s_nssai_list: Option<Vec<SNssai>>,
    /// Dedicated NAS Message
    pub dedicated_nas_message: Vec<u8>,
    /// 5G-S-TMSI value
    pub ng_5g_s_tmsi_value: Option<Ng5gSTmsiValue>,
}

/// Build an RRC Setup Complete message
pub fn build_rrc_setup_complete(params: &RrcSetupCompleteParams) -> Result<UL_DCCH_Message, RrcSetupError> {
    if params.rrc_transaction_id > 3 {
        return Err(RrcSetupError::InvalidFieldValue(
            "RRC Transaction ID must be 0-3".to_string(),
        ));
    }

    if params.selected_plmn_identity < 1 || params.selected_plmn_identity > 12 {
        return Err(RrcSetupError::InvalidFieldValue(
            "Selected PLMN Identity must be 1-12".to_string(),
        ));
    }

    // Build GUAMI type if present
    let guami_type = params.guami_type.map(|gt| match gt {
        GuamiType::Native => RRCSetupComplete_IEsGuami_Type(RRCSetupComplete_IEsGuami_Type::NATIVE),
        GuamiType::Mapped => RRCSetupComplete_IEsGuami_Type(RRCSetupComplete_IEsGuami_Type::MAPPED),
    });

    // Build S-NSSAI list if present
    let s_nssai_list = params.s_nssai_list.as_ref().map(|list| {
        let items: Vec<S_NSSAI> = list
            .iter()
            .map(|snssai| {
                // SST is 8 bits
                let mut sst_bv: BitVec<u8, Msb0> = BitVec::new();
                for i in (0..8).rev() {
                    sst_bv.push((snssai.sst >> i) & 1 == 1);
                }
                
                if let Some(sd) = snssai.sd {
                    // SST + SD combined (32 bits total)
                    let mut sst_sd_bv: BitVec<u8, Msb0> = BitVec::new();
                    // SST (8 bits)
                    for i in (0..8).rev() {
                        sst_sd_bv.push((snssai.sst >> i) & 1 == 1);
                    }
                    // SD (24 bits)
                    for i in (0..24).rev() {
                        sst_sd_bv.push((sd >> i) & 1 == 1);
                    }
                    S_NSSAI::Sst_SD(S_NSSAI_sst_SD(sst_sd_bv))
                } else {
                    S_NSSAI::Sst(S_NSSAI_sst(sst_bv))
                }
            })
            .collect();
        RRCSetupComplete_IEsS_NSSAI_List(items)
    });

    // Build 5G-S-TMSI value if present
    let ng_5g_s_tmsi_value = params.ng_5g_s_tmsi_value.as_ref().map(|value| match value {
        Ng5gSTmsiValue::Full(v) => {
            let mut bv: BitVec<u8, Msb0> = BitVec::new();
            for i in (0..48).rev() {
                bv.push((v >> i) & 1 == 1);
            }
            RRCSetupComplete_IEsNg_5G_S_TMSI_Value::Ng_5G_S_TMSI(NG_5G_S_TMSI(bv))
        }
        Ng5gSTmsiValue::Part2(v) => {
            let mut bv: BitVec<u8, Msb0> = BitVec::new();
            for i in (0..9).rev() {
                bv.push((v >> i) & 1 == 1);
            }
            RRCSetupComplete_IEsNg_5G_S_TMSI_Value::Ng_5G_S_TMSI_Part2(
                RRCSetupComplete_IEsNg_5G_S_TMSI_Value_ng_5G_S_TMSI_Part2(bv),
            )
        }
    });

    let rrc_setup_complete_ies = RRCSetupComplete_IEs {
        selected_plmn_identity: RRCSetupComplete_IEsSelectedPLMN_Identity(params.selected_plmn_identity),
        registered_amf: None, // Simplified - not including RegisteredAMF for now
        guami_type,
        s_nssai_list,
        dedicated_nas_message: DedicatedNAS_Message(params.dedicated_nas_message.clone()),
        ng_5g_s_tmsi_value,
        late_non_critical_extension: None,
        non_critical_extension: None,
    };

    let rrc_setup_complete = RRCSetupComplete {
        rrc_transaction_identifier: RRC_TransactionIdentifier(params.rrc_transaction_id),
        critical_extensions: RRCSetupCompleteCriticalExtensions::RrcSetupComplete(rrc_setup_complete_ies),
    };

    let message_type = UL_DCCH_MessageType::C1(UL_DCCH_MessageType_c1::RrcSetupComplete(rrc_setup_complete));

    Ok(UL_DCCH_Message { message: message_type })
}

/// Parse an RRC Setup Complete from a UL-DCCH message
pub fn parse_rrc_setup_complete(msg: &UL_DCCH_Message) -> Result<RrcSetupCompleteData, RrcSetupError> {
    let rrc_setup_complete = match &msg.message {
        UL_DCCH_MessageType::C1(c1) => match c1 {
            UL_DCCH_MessageType_c1::RrcSetupComplete(complete) => complete,
            _ => {
                return Err(RrcSetupError::InvalidMessageType {
                    expected: "RRCSetupComplete".to_string(),
                    actual: "other c1 message".to_string(),
                })
            }
        },
        _ => {
            return Err(RrcSetupError::InvalidMessageType {
                expected: "c1".to_string(),
                actual: "messageClassExtension".to_string(),
            })
        }
    };

    let ies = match &rrc_setup_complete.critical_extensions {
        RRCSetupCompleteCriticalExtensions::RrcSetupComplete(ies) => ies,
        RRCSetupCompleteCriticalExtensions::CriticalExtensionsFuture(_) => {
            return Err(RrcSetupError::InvalidMessageType {
                expected: "rrcSetupComplete".to_string(),
                actual: "criticalExtensionsFuture".to_string(),
            })
        }
    };

    // Parse GUAMI type
    let guami_type = ies.guami_type.as_ref().map(|gt| match gt.0 {
        RRCSetupComplete_IEsGuami_Type::NATIVE => GuamiType::Native,
        RRCSetupComplete_IEsGuami_Type::MAPPED => GuamiType::Mapped,
        _ => GuamiType::Native, // Default fallback
    });

    // Parse S-NSSAI list
    let s_nssai_list = ies.s_nssai_list.as_ref().map(|list| {
        list.0
            .iter()
            .map(|snssai| match snssai {
                S_NSSAI::Sst(sst_bv) => {
                    let sst = bitvec_to_u64(&sst_bv.0) as u8;
                    SNssai { sst, sd: None }
                }
                S_NSSAI::Sst_SD(sst_sd_bv) => {
                    let combined = bitvec_to_u64(&sst_sd_bv.0);
                    let sst = (combined >> 24) as u8;
                    let sd = (combined & 0xFFFFFF) as u32;
                    SNssai { sst, sd: Some(sd) }
                }
            })
            .collect()
    });

    // Parse 5G-S-TMSI value
    let ng_5g_s_tmsi_value = ies.ng_5g_s_tmsi_value.as_ref().map(|value| match value {
        RRCSetupComplete_IEsNg_5G_S_TMSI_Value::Ng_5G_S_TMSI(bv) => {
            Ng5gSTmsiValue::Full(bitvec_to_u64(&bv.0))
        }
        RRCSetupComplete_IEsNg_5G_S_TMSI_Value::Ng_5G_S_TMSI_Part2(bv) => {
            Ng5gSTmsiValue::Part2(bitvec_to_u64(&bv.0) as u16)
        }
    });

    Ok(RrcSetupCompleteData {
        rrc_transaction_id: rrc_setup_complete.rrc_transaction_identifier.0,
        selected_plmn_identity: ies.selected_plmn_identity.0,
        guami_type,
        s_nssai_list,
        dedicated_nas_message: ies.dedicated_nas_message.0.clone(),
        ng_5g_s_tmsi_value,
    })
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Build and encode an RRC Setup Request to bytes
pub fn encode_rrc_setup_request(params: &RrcSetupRequestParams) -> Result<Vec<u8>, RrcSetupError> {
    let msg = build_rrc_setup_request(params)?;
    Ok(encode_rrc(&msg)?)
}

/// Decode and parse an RRC Setup Request from bytes
pub fn decode_rrc_setup_request(bytes: &[u8]) -> Result<RrcSetupRequestData, RrcSetupError> {
    let msg: UL_CCCH_Message = decode_rrc(bytes)?;
    parse_rrc_setup_request(&msg)
}

/// Build and encode an RRC Setup to bytes
pub fn encode_rrc_setup(params: &RrcSetupParams) -> Result<Vec<u8>, RrcSetupError> {
    let msg = build_rrc_setup(params)?;
    Ok(encode_rrc(&msg)?)
}

/// Decode and parse an RRC Setup from bytes
pub fn decode_rrc_setup(bytes: &[u8]) -> Result<RrcSetupData, RrcSetupError> {
    let msg: DL_CCCH_Message = decode_rrc(bytes)?;
    parse_rrc_setup(&msg)
}

/// Build and encode an RRC Setup Complete to bytes
pub fn encode_rrc_setup_complete(params: &RrcSetupCompleteParams) -> Result<Vec<u8>, RrcSetupError> {
    let msg = build_rrc_setup_complete(params)?;
    Ok(encode_rrc(&msg)?)
}

/// Decode and parse an RRC Setup Complete from bytes
pub fn decode_rrc_setup_complete(bytes: &[u8]) -> Result<RrcSetupCompleteData, RrcSetupError> {
    let msg: UL_DCCH_Message = decode_rrc(bytes)?;
    parse_rrc_setup_complete(&msg)
}

/// Check if a UL-CCCH message is an RRC Setup Request
pub fn is_rrc_setup_request(msg: &UL_CCCH_Message) -> bool {
    matches!(
        &msg.message,
        UL_CCCH_MessageType::C1(UL_CCCH_MessageType_c1::RrcSetupRequest(_))
    )
}

/// Check if a DL-CCCH message is an RRC Setup
pub fn is_rrc_setup(msg: &DL_CCCH_Message) -> bool {
    matches!(
        &msg.message,
        DL_CCCH_MessageType::C1(DL_CCCH_MessageType_c1::RrcSetup(_))
    )
}

/// Check if a UL-DCCH message is an RRC Setup Complete
pub fn is_rrc_setup_complete(msg: &UL_DCCH_Message) -> bool {
    matches!(
        &msg.message,
        UL_DCCH_MessageType::C1(UL_DCCH_MessageType_c1::RrcSetupComplete(_))
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // RRC Setup Request Tests
    // ========================================================================

    fn create_test_setup_request_params() -> RrcSetupRequestParams {
        RrcSetupRequestParams {
            ue_identity: UeIdentity::RandomValue(0x1234567890),
            establishment_cause: RrcEstablishmentCause::MoSignalling,
        }
    }

    #[test]
    fn test_build_rrc_setup_request() {
        let params = create_test_setup_request_params();
        let result = build_rrc_setup_request(&params);
        assert!(result.is_ok());

        let msg = result.unwrap();
        assert!(is_rrc_setup_request(&msg));
    }

    #[test]
    fn test_parse_rrc_setup_request() {
        let params = create_test_setup_request_params();
        let msg = build_rrc_setup_request(&params).unwrap();
        let result = parse_rrc_setup_request(&msg);
        assert!(result.is_ok());

        let data = result.unwrap();
        assert_eq!(data.ue_identity, params.ue_identity);
        assert_eq!(data.establishment_cause, params.establishment_cause);
    }

    #[test]
    fn test_rrc_setup_request_with_5g_s_tmsi() {
        let params = RrcSetupRequestParams {
            ue_identity: UeIdentity::Ng5gSTmsiPart1(0x7FFFFFFFFF), // Max 39-bit value
            establishment_cause: RrcEstablishmentCause::MtAccess,
        };
        let msg = build_rrc_setup_request(&params).unwrap();
        let data = parse_rrc_setup_request(&msg).unwrap();
        assert_eq!(data.ue_identity, params.ue_identity);
    }

    #[test]
    fn test_encode_decode_rrc_setup_request() {
        let params = create_test_setup_request_params();
        let encoded = encode_rrc_setup_request(&params);
        assert!(encoded.is_ok());

        let bytes = encoded.unwrap();
        assert!(!bytes.is_empty());

        let decoded = decode_rrc_setup_request(&bytes);
        assert!(decoded.is_ok());

        let data = decoded.unwrap();
        assert_eq!(data.establishment_cause, params.establishment_cause);
    }

    #[test]
    fn test_establishment_cause_conversion() {
        // Test all establishment causes
        let causes = [
            RrcEstablishmentCause::Emergency,
            RrcEstablishmentCause::HighPriorityAccess,
            RrcEstablishmentCause::MtAccess,
            RrcEstablishmentCause::MoSignalling,
            RrcEstablishmentCause::MoData,
            RrcEstablishmentCause::MoVoiceCall,
            RrcEstablishmentCause::MoVideoCall,
            RrcEstablishmentCause::MoSms,
            RrcEstablishmentCause::MpsPriorityAccess,
            RrcEstablishmentCause::McsPriorityAccess,
        ];

        for cause in causes {
            let asn_cause: EstablishmentCause = cause.into();
            let back: RrcEstablishmentCause = asn_cause.try_into().unwrap();
            assert_eq!(back, cause);
        }
    }

    // ========================================================================
    // RRC Setup Complete Tests
    // ========================================================================

    fn create_test_setup_complete_params() -> RrcSetupCompleteParams {
        RrcSetupCompleteParams {
            rrc_transaction_id: 0,
            selected_plmn_identity: 1,
            guami_type: None,
            s_nssai_list: None,
            dedicated_nas_message: vec![0x7E, 0x00, 0x41], // Sample NAS message
            ng_5g_s_tmsi_value: None,
        }
    }

    #[test]
    fn test_build_rrc_setup_complete() {
        let params = create_test_setup_complete_params();
        let result = build_rrc_setup_complete(&params);
        assert!(result.is_ok());

        let msg = result.unwrap();
        assert!(is_rrc_setup_complete(&msg));
    }

    #[test]
    fn test_parse_rrc_setup_complete() {
        let params = create_test_setup_complete_params();
        let msg = build_rrc_setup_complete(&params).unwrap();
        let result = parse_rrc_setup_complete(&msg);
        assert!(result.is_ok());

        let data = result.unwrap();
        assert_eq!(data.rrc_transaction_id, params.rrc_transaction_id);
        assert_eq!(data.selected_plmn_identity, params.selected_plmn_identity);
        assert_eq!(data.dedicated_nas_message, params.dedicated_nas_message);
    }

    #[test]
    fn test_rrc_setup_complete_with_optional_fields() {
        let params = RrcSetupCompleteParams {
            rrc_transaction_id: 2,
            selected_plmn_identity: 3,
            guami_type: Some(GuamiType::Native),
            s_nssai_list: Some(vec![
                SNssai { sst: 1, sd: None },
                SNssai { sst: 2, sd: Some(0x000001) },
            ]),
            dedicated_nas_message: vec![0x7E, 0x00, 0x41, 0x01, 0x02],
            ng_5g_s_tmsi_value: Some(Ng5gSTmsiValue::Full(0x123456789ABC)),
        };

        let msg = build_rrc_setup_complete(&params).unwrap();
        let data = parse_rrc_setup_complete(&msg).unwrap();

        assert_eq!(data.rrc_transaction_id, 2);
        assert_eq!(data.selected_plmn_identity, 3);
        assert_eq!(data.guami_type, Some(GuamiType::Native));
        assert!(data.s_nssai_list.is_some());
        assert_eq!(data.s_nssai_list.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_encode_decode_rrc_setup_complete() {
        let params = create_test_setup_complete_params();
        let encoded = encode_rrc_setup_complete(&params);
        assert!(encoded.is_ok());

        let bytes = encoded.unwrap();
        assert!(!bytes.is_empty());

        let decoded = decode_rrc_setup_complete(&bytes);
        assert!(decoded.is_ok());

        let data = decoded.unwrap();
        assert_eq!(data.dedicated_nas_message, params.dedicated_nas_message);
    }

    #[test]
    fn test_invalid_rrc_transaction_id() {
        let params = RrcSetupCompleteParams {
            rrc_transaction_id: 5, // Invalid: must be 0-3
            selected_plmn_identity: 1,
            guami_type: None,
            s_nssai_list: None,
            dedicated_nas_message: vec![0x7E],
            ng_5g_s_tmsi_value: None,
        };

        let result = build_rrc_setup_complete(&params);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_selected_plmn_identity() {
        let params = RrcSetupCompleteParams {
            rrc_transaction_id: 0,
            selected_plmn_identity: 0, // Invalid: must be 1-12
            guami_type: None,
            s_nssai_list: None,
            dedicated_nas_message: vec![0x7E],
            ng_5g_s_tmsi_value: None,
        };

        let result = build_rrc_setup_complete(&params);
        assert!(result.is_err());
    }

    #[test]
    fn test_5g_s_tmsi_part2() {
        let params = RrcSetupCompleteParams {
            rrc_transaction_id: 0,
            selected_plmn_identity: 1,
            guami_type: None,
            s_nssai_list: None,
            dedicated_nas_message: vec![0x7E],
            ng_5g_s_tmsi_value: Some(Ng5gSTmsiValue::Part2(0x1FF)), // Max 9-bit value
        };

        let msg = build_rrc_setup_complete(&params).unwrap();
        let data = parse_rrc_setup_complete(&msg).unwrap();
        assert_eq!(data.ng_5g_s_tmsi_value, Some(Ng5gSTmsiValue::Part2(0x1FF)));
    }
}
