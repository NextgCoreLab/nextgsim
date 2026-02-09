//! Initial UE Message Procedure
//!
//! Implements the Initial UE Message procedure as defined in 3GPP TS 38.413 Section 8.6.1.
//! This procedure is used by the NG-RAN node to transfer the initial NAS message from the UE
//! to the AMF when the UE is in RRC_CONNECTED state and no UE-associated logical NG-connection
//! exists for the UE.

use crate::codec::generated::*;
use crate::codec::{decode_ngap_pdu, encode_ngap_pdu, NgapCodecError};
use bitvec::prelude::*;
use thiserror::Error;

/// Errors that can occur during Initial UE Message procedures
#[derive(Debug, Error)]
pub enum InitialUeMessageError {
    /// Codec error during encoding/decoding
    #[error("Codec error: {0}")]
    CodecError(#[from] NgapCodecError),

    /// Invalid message type received
    #[error("Invalid message type: expected {expected}, got {actual}")]
    InvalidMessageType { expected: String, actual: String },

    /// Missing mandatory IE
    #[error("Missing mandatory IE: {0}")]
    MissingMandatoryIe(String),

    /// Invalid IE value
    #[error("Invalid IE value: {0}")]
    InvalidIeValue(String),
}

/// RRC Establishment Cause values as defined in 3GPP TS 38.413
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RrcEstablishmentCauseValue {
    Emergency,
    HighPriorityAccess,
    MtAccess,
    MoSignalling,
    MoData,
    MoVoiceCall,
    MoVideoCall,
    MoSms,
    MpsHighPriorityAccess,
    McsHighPriorityAccess,
}

impl From<RrcEstablishmentCauseValue> for RRCEstablishmentCause {
    fn from(cause: RrcEstablishmentCauseValue) -> Self {
        let value = match cause {
            RrcEstablishmentCauseValue::Emergency => RRCEstablishmentCause::EMERGENCY,
            RrcEstablishmentCauseValue::HighPriorityAccess => {
                RRCEstablishmentCause::HIGH_PRIORITY_ACCESS
            }
            RrcEstablishmentCauseValue::MtAccess => RRCEstablishmentCause::MT_ACCESS,
            RrcEstablishmentCauseValue::MoSignalling => RRCEstablishmentCause::MO_SIGNALLING,
            RrcEstablishmentCauseValue::MoData => RRCEstablishmentCause::MO_DATA,
            RrcEstablishmentCauseValue::MoVoiceCall => RRCEstablishmentCause::MO_VOICE_CALL,
            RrcEstablishmentCauseValue::MoVideoCall => RRCEstablishmentCause::MO_VIDEO_CALL,
            RrcEstablishmentCauseValue::MoSms => RRCEstablishmentCause::MO_SMS,
            RrcEstablishmentCauseValue::MpsHighPriorityAccess => {
                RRCEstablishmentCause::MPS_PRIORITY_ACCESS
            }
            RrcEstablishmentCauseValue::McsHighPriorityAccess => {
                RRCEstablishmentCause::MCS_PRIORITY_ACCESS
            }
        };
        RRCEstablishmentCause(value)
    }
}

impl TryFrom<RRCEstablishmentCause> for RrcEstablishmentCauseValue {
    type Error = InitialUeMessageError;

    fn try_from(cause: RRCEstablishmentCause) -> Result<Self, Self::Error> {
        match cause.0 {
            RRCEstablishmentCause::EMERGENCY => Ok(RrcEstablishmentCauseValue::Emergency),
            RRCEstablishmentCause::HIGH_PRIORITY_ACCESS => {
                Ok(RrcEstablishmentCauseValue::HighPriorityAccess)
            }
            RRCEstablishmentCause::MT_ACCESS => Ok(RrcEstablishmentCauseValue::MtAccess),
            RRCEstablishmentCause::MO_SIGNALLING => Ok(RrcEstablishmentCauseValue::MoSignalling),
            RRCEstablishmentCause::MO_DATA => Ok(RrcEstablishmentCauseValue::MoData),
            RRCEstablishmentCause::MO_VOICE_CALL => Ok(RrcEstablishmentCauseValue::MoVoiceCall),
            RRCEstablishmentCause::MO_VIDEO_CALL => Ok(RrcEstablishmentCauseValue::MoVideoCall),
            RRCEstablishmentCause::MO_SMS => Ok(RrcEstablishmentCauseValue::MoSms),
            RRCEstablishmentCause::MPS_PRIORITY_ACCESS => {
                Ok(RrcEstablishmentCauseValue::MpsHighPriorityAccess)
            }
            RRCEstablishmentCause::MCS_PRIORITY_ACCESS => {
                Ok(RrcEstablishmentCauseValue::McsHighPriorityAccess)
            }
            _ => Err(InitialUeMessageError::InvalidIeValue(format!(
                "Unknown RRCEstablishmentCause value: {}",
                cause.0
            ))),
        }
    }
}

/// UE Context Request values
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UeContextRequestValue {
    Requested,
}

impl From<UeContextRequestValue> for UEContextRequest {
    fn from(_: UeContextRequestValue) -> Self {
        UEContextRequest(UEContextRequest::REQUESTED)
    }
}


/// User Location Information for NR (5G)
#[derive(Debug, Clone)]
pub struct UserLocationInfoNr {
    /// NR Cell Global Identity
    pub nr_cgi: NrCgi,
    /// Tracking Area Identity
    pub tai: Tai,
    /// Time stamp (optional)
    pub time_stamp: Option<[u8; 4]>,
}

/// NR Cell Global Identity
#[derive(Debug, Clone)]
pub struct NrCgi {
    /// PLMN Identity (3 bytes)
    pub plmn_identity: [u8; 3],
    /// NR Cell Identity (36 bits)
    pub nr_cell_identity: u64,
}

/// Tracking Area Identity
#[derive(Debug, Clone)]
pub struct Tai {
    /// PLMN Identity (3 bytes)
    pub plmn_identity: [u8; 3],
    /// Tracking Area Code (3 bytes)
    pub tac: [u8; 3],
}

/// 5G-S-TMSI (5G S-Temporary Mobile Subscriber Identity)
#[derive(Debug, Clone)]
pub struct FiveGSTmsi {
    /// AMF Set ID (10 bits)
    pub amf_set_id: u16,
    /// AMF Pointer (6 bits)
    pub amf_pointer: u8,
    /// 5G-TMSI (4 bytes)
    pub five_g_tmsi: [u8; 4],
}

/// Parameters for building an Initial UE Message
#[derive(Debug, Clone)]
pub struct InitialUeMessageParams {
    /// RAN UE NGAP ID (allocated by gNB)
    pub ran_ue_ngap_id: u32,
    /// NAS PDU (contains the NAS message from UE)
    pub nas_pdu: Vec<u8>,
    /// User Location Information
    pub user_location_info: UserLocationInfoNr,
    /// RRC Establishment Cause
    pub rrc_establishment_cause: RrcEstablishmentCauseValue,
    /// 5G-S-TMSI (optional)
    pub five_g_s_tmsi: Option<FiveGSTmsi>,
    /// AMF Set ID (optional, for AMF selection)
    pub amf_set_id: Option<u16>,
    /// UE Context Request (optional)
    pub ue_context_request: Option<UeContextRequestValue>,
    /// Allowed NSSAI (optional)
    pub allowed_nssai: Option<Vec<AllowedSnssai>>,
}

/// Allowed S-NSSAI item
#[derive(Debug, Clone)]
pub struct AllowedSnssai {
    /// Slice/Service Type (SST) - 1 byte
    pub sst: u8,
    /// Slice Differentiator (SD) - 3 bytes, optional
    pub sd: Option<[u8; 3]>,
}

/// Parsed Initial UE Message data
#[derive(Debug, Clone)]
pub struct InitialUeMessageData {
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
    /// NAS PDU
    pub nas_pdu: Vec<u8>,
    /// User Location Information
    pub user_location_info: UserLocationInfoNr,
    /// RRC Establishment Cause
    pub rrc_establishment_cause: RrcEstablishmentCauseValue,
    /// 5G-S-TMSI (optional)
    pub five_g_s_tmsi: Option<FiveGSTmsi>,
    /// AMF Set ID (optional)
    pub amf_set_id: Option<u16>,
    /// UE Context Request (optional)
    pub ue_context_request: Option<UeContextRequestValue>,
}

// ============================================================================
// Initial UE Message Builder
// ============================================================================

/// Build an Initial UE Message PDU
///
/// # Arguments
/// * `params` - Parameters for the Initial UE Message
///
/// # Returns
/// * `Ok(NGAP_PDU)` - The constructed PDU
/// * `Err(InitialUeMessageError)` - If construction fails
pub fn build_initial_ue_message(
    params: &InitialUeMessageParams,
) -> Result<NGAP_PDU, InitialUeMessageError> {
    let mut protocol_ies = Vec::new();

    // IE: RAN-UE-NGAP-ID (mandatory)
    protocol_ies.push(InitialUEMessageProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_RAN_UE_NGAP_ID),
        criticality: Criticality(Criticality::REJECT),
        value: InitialUEMessageProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(RAN_UE_NGAP_ID(
            params.ran_ue_ngap_id,
        )),
    });

    // IE: NAS-PDU (mandatory)
    protocol_ies.push(InitialUEMessageProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_NAS_PDU),
        criticality: Criticality(Criticality::REJECT),
        value: InitialUEMessageProtocolIEs_EntryValue::Id_NAS_PDU(NAS_PDU(
            params.nas_pdu.clone(),
        )),
    });

    // IE: UserLocationInformation (mandatory)
    let user_location_info = build_user_location_info(&params.user_location_info);
    protocol_ies.push(InitialUEMessageProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_USER_LOCATION_INFORMATION),
        criticality: Criticality(Criticality::REJECT),
        value: InitialUEMessageProtocolIEs_EntryValue::Id_UserLocationInformation(
            user_location_info,
        ),
    });

    // IE: RRCEstablishmentCause (mandatory)
    protocol_ies.push(InitialUEMessageProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_RRC_ESTABLISHMENT_CAUSE),
        criticality: Criticality(Criticality::IGNORE),
        value: InitialUEMessageProtocolIEs_EntryValue::Id_RRCEstablishmentCause(
            params.rrc_establishment_cause.into(),
        ),
    });

    // IE: FiveG-S-TMSI (optional)
    if let Some(ref tmsi) = params.five_g_s_tmsi {
        let five_g_s_tmsi = build_five_g_s_tmsi(tmsi);
        protocol_ies.push(InitialUEMessageProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_FIVE_G_S_TMSI),
            criticality: Criticality(Criticality::REJECT),
            value: InitialUEMessageProtocolIEs_EntryValue::Id_FiveG_S_TMSI(five_g_s_tmsi),
        });
    }

    // IE: AMFSetID (optional)
    if let Some(amf_set_id) = params.amf_set_id {
        let amf_set_id_bv = build_amf_set_id(amf_set_id);
        protocol_ies.push(InitialUEMessageProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_AMF_SET_ID),
            criticality: Criticality(Criticality::IGNORE),
            value: InitialUEMessageProtocolIEs_EntryValue::Id_AMFSetID(amf_set_id_bv),
        });
    }

    // IE: UEContextRequest (optional)
    if let Some(ue_context_request) = params.ue_context_request {
        protocol_ies.push(InitialUEMessageProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_UE_CONTEXT_REQUEST),
            criticality: Criticality(Criticality::IGNORE),
            value: InitialUEMessageProtocolIEs_EntryValue::Id_UEContextRequest(
                ue_context_request.into(),
            ),
        });
    }

    // IE: AllowedNSSAI (optional)
    if let Some(ref allowed_nssai) = params.allowed_nssai {
        let allowed_nssai_list = build_allowed_nssai(allowed_nssai);
        protocol_ies.push(InitialUEMessageProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_ALLOWED_NSSAI),
            criticality: Criticality(Criticality::REJECT),
            value: InitialUEMessageProtocolIEs_EntryValue::Id_AllowedNSSAI(allowed_nssai_list),
        });
    }

    let initial_ue_message = InitialUEMessage {
        protocol_i_es: InitialUEMessageProtocolIEs(protocol_ies),
    };

    let initiating_message = InitiatingMessage {
        procedure_code: ProcedureCode(ID_INITIAL_UE_MESSAGE),
        criticality: Criticality(Criticality::IGNORE),
        value: InitiatingMessageValue::Id_InitialUEMessage(initial_ue_message),
    };

    Ok(NGAP_PDU::InitiatingMessage(initiating_message))
}


fn build_user_location_info(info: &UserLocationInfoNr) -> UserLocationInformation {
    // Build NR-CGI
    let nr_cgi = build_nr_cgi(&info.nr_cgi);

    // Build TAI
    let tai = build_tai(&info.tai);

    // Build timestamp if present
    let time_stamp = info.time_stamp.map(|ts| TimeStamp(ts.to_vec()));

    let user_location_info_nr = UserLocationInformationNR {
        nr_cgi,
        tai,
        time_stamp,
        ie_extensions: None,
    };

    UserLocationInformation::UserLocationInformationNR(user_location_info_nr)
}

/// Build an NR-CGI from the NrCgi struct
pub fn build_nr_cgi(nr_cgi: &NrCgi) -> NR_CGI {
    // NR Cell Identity is 36 bits
    let mut bv: BitVec<u8, Msb0> = BitVec::new();
    for i in (0..36).rev() {
        bv.push((nr_cgi.nr_cell_identity >> i) & 1 == 1);
    }

    NR_CGI {
        plmn_identity: PLMNIdentity(nr_cgi.plmn_identity.to_vec()),
        nr_cell_identity: NRCellIdentity(bv),
        ie_extensions: None,
    }
}

/// Build a TAI from the Tai struct
pub fn build_tai(tai: &Tai) -> TAI {
    TAI {
        plmn_identity: PLMNIdentity(tai.plmn_identity.to_vec()),
        tac: TAC(tai.tac.to_vec()),
        ie_extensions: None,
    }
}

fn build_five_g_s_tmsi(tmsi: &FiveGSTmsi) -> FiveG_S_TMSI {
    // AMF Set ID is 10 bits
    let mut amf_set_id_bv: BitVec<u8, Msb0> = BitVec::new();
    for i in (0..10).rev() {
        amf_set_id_bv.push((tmsi.amf_set_id >> i) & 1 == 1);
    }

    // AMF Pointer is 6 bits
    let mut amf_pointer_bv: BitVec<u8, Msb0> = BitVec::new();
    for i in (0..6).rev() {
        amf_pointer_bv.push((tmsi.amf_pointer >> i) & 1 == 1);
    }

    FiveG_S_TMSI {
        amf_set_id: AMFSetID(amf_set_id_bv),
        amf_pointer: AMFPointer(amf_pointer_bv),
        five_g_tmsi: FiveG_TMSI(tmsi.five_g_tmsi.to_vec()),
        ie_extensions: None,
    }
}

fn build_amf_set_id(amf_set_id: u16) -> AMFSetID {
    // AMF Set ID is 10 bits
    let mut bv: BitVec<u8, Msb0> = BitVec::new();
    for i in (0..10).rev() {
        bv.push((amf_set_id >> i) & 1 == 1);
    }
    AMFSetID(bv)
}

fn build_allowed_nssai(allowed_nssai: &[AllowedSnssai]) -> AllowedNSSAI {
    let items: Vec<AllowedNSSAI_Item> = allowed_nssai
        .iter()
        .map(|snssai| {
            let sd = snssai.sd.map(|sd_bytes| SD(sd_bytes.to_vec()));
            AllowedNSSAI_Item {
                s_nssai: S_NSSAI {
                    sst: SST(vec![snssai.sst]),
                    sd,
                    ie_extensions: None,
                },
                ie_extensions: None,
            }
        })
        .collect();

    AllowedNSSAI(items)
}

// ============================================================================
// Initial UE Message Parser
// ============================================================================

/// Parse an Initial UE Message from an NGAP PDU
///
/// # Arguments
/// * `pdu` - The NGAP PDU to parse
///
/// # Returns
/// * `Ok(InitialUeMessageData)` - The parsed message data
/// * `Err(InitialUeMessageError)` - If parsing fails
pub fn parse_initial_ue_message(pdu: &NGAP_PDU) -> Result<InitialUeMessageData, InitialUeMessageError> {
    let initiating_message = match pdu {
        NGAP_PDU::InitiatingMessage(msg) => msg,
        _ => {
            return Err(InitialUeMessageError::InvalidMessageType {
                expected: "InitiatingMessage".to_string(),
                actual: format!("{pdu:?}"),
            })
        }
    };

    let initial_ue_message = match &initiating_message.value {
        InitiatingMessageValue::Id_InitialUEMessage(msg) => msg,
        _ => {
            return Err(InitialUeMessageError::InvalidMessageType {
                expected: "InitialUEMessage".to_string(),
                actual: format!("{:?}", initiating_message.value),
            })
        }
    };

    let mut ran_ue_ngap_id: Option<u32> = None;
    let mut nas_pdu: Option<Vec<u8>> = None;
    let mut user_location_info: Option<UserLocationInfoNr> = None;
    let mut rrc_establishment_cause: Option<RrcEstablishmentCauseValue> = None;
    let mut five_g_s_tmsi: Option<FiveGSTmsi> = None;
    let mut amf_set_id: Option<u16> = None;
    let mut ue_context_request: Option<UeContextRequestValue> = None;

    for ie in &initial_ue_message.protocol_i_es.0 {
        match &ie.value {
            InitialUEMessageProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(id) => {
                ran_ue_ngap_id = Some(id.0);
            }
            InitialUEMessageProtocolIEs_EntryValue::Id_NAS_PDU(pdu) => {
                nas_pdu = Some(pdu.0.clone());
            }
            InitialUEMessageProtocolIEs_EntryValue::Id_UserLocationInformation(info) => {
                user_location_info = Some(parse_user_location_info(info)?);
            }
            InitialUEMessageProtocolIEs_EntryValue::Id_RRCEstablishmentCause(cause) => {
                rrc_establishment_cause = Some(cause.clone().try_into()?);
            }
            InitialUEMessageProtocolIEs_EntryValue::Id_FiveG_S_TMSI(tmsi) => {
                five_g_s_tmsi = Some(parse_five_g_s_tmsi(tmsi));
            }
            InitialUEMessageProtocolIEs_EntryValue::Id_AMFSetID(id) => {
                amf_set_id = Some(parse_amf_set_id(id));
            }
            InitialUEMessageProtocolIEs_EntryValue::Id_UEContextRequest(req) => {
                if req.0 == UEContextRequest::REQUESTED {
                    ue_context_request = Some(UeContextRequestValue::Requested);
                }
            }
            _ => {
                // Ignore other IEs
            }
        }
    }

    Ok(InitialUeMessageData {
        ran_ue_ngap_id: ran_ue_ngap_id
            .ok_or_else(|| InitialUeMessageError::MissingMandatoryIe("RAN-UE-NGAP-ID".to_string()))?,
        nas_pdu: nas_pdu
            .ok_or_else(|| InitialUeMessageError::MissingMandatoryIe("NAS-PDU".to_string()))?,
        user_location_info: user_location_info.ok_or_else(|| {
            InitialUeMessageError::MissingMandatoryIe("UserLocationInformation".to_string())
        })?,
        rrc_establishment_cause: rrc_establishment_cause.ok_or_else(|| {
            InitialUeMessageError::MissingMandatoryIe("RRCEstablishmentCause".to_string())
        })?,
        five_g_s_tmsi,
        amf_set_id,
        ue_context_request,
    })
}


fn parse_user_location_info(
    info: &UserLocationInformation,
) -> Result<UserLocationInfoNr, InitialUeMessageError> {
    match info {
        UserLocationInformation::UserLocationInformationNR(nr_info) => {
            let nr_cgi = parse_nr_cgi(&nr_info.nr_cgi);
            let tai = parse_tai(&nr_info.tai);
            let time_stamp = nr_info
                .time_stamp
                .as_ref()
                .and_then(|ts| ts.0.as_slice().try_into().ok());

            Ok(UserLocationInfoNr {
                nr_cgi,
                tai,
                time_stamp,
            })
        }
        _ => Err(InitialUeMessageError::InvalidIeValue(
            "Expected UserLocationInformationNR".to_string(),
        )),
    }
}

/// Parse an NR-CGI into the NrCgi struct
pub fn parse_nr_cgi(nr_cgi: &NR_CGI) -> NrCgi {
    let plmn_identity: [u8; 3] = nr_cgi
        .plmn_identity
        .0
        .as_slice()
        .try_into()
        .unwrap_or([0, 0, 0]);

    // NR Cell Identity is 36 bits stored in a BitVec
    let nr_cell_identity = if nr_cgi.nr_cell_identity.0.len() >= 36 {
        let mut value: u64 = 0;
        for (i, bit) in nr_cgi.nr_cell_identity.0.iter().take(36).enumerate() {
            if *bit {
                value |= 1 << (35 - i);
            }
        }
        value
    } else {
        0
    };

    NrCgi {
        plmn_identity,
        nr_cell_identity,
    }
}

/// Parse a TAI into the Tai struct
pub fn parse_tai(tai: &TAI) -> Tai {
    let plmn_identity: [u8; 3] = tai
        .plmn_identity
        .0
        .as_slice()
        .try_into()
        .unwrap_or([0, 0, 0]);

    let tac: [u8; 3] = tai.tac.0.as_slice().try_into().unwrap_or([0, 0, 0]);

    Tai { plmn_identity, tac }
}

fn parse_five_g_s_tmsi(tmsi: &FiveG_S_TMSI) -> FiveGSTmsi {
    // AMF Set ID is 10 bits
    let amf_set_id = if tmsi.amf_set_id.0.len() >= 10 {
        let mut value: u16 = 0;
        for (i, bit) in tmsi.amf_set_id.0.iter().take(10).enumerate() {
            if *bit {
                value |= 1 << (9 - i);
            }
        }
        value
    } else {
        0
    };

    // AMF Pointer is 6 bits
    let amf_pointer = if tmsi.amf_pointer.0.len() >= 6 {
        let mut value: u8 = 0;
        for (i, bit) in tmsi.amf_pointer.0.iter().take(6).enumerate() {
            if *bit {
                value |= 1 << (5 - i);
            }
        }
        value
    } else {
        0
    };

    let five_g_tmsi: [u8; 4] = tmsi
        .five_g_tmsi
        .0
        .as_slice()
        .try_into()
        .unwrap_or([0, 0, 0, 0]);

    FiveGSTmsi {
        amf_set_id,
        amf_pointer,
        five_g_tmsi,
    }
}

fn parse_amf_set_id(amf_set_id: &AMFSetID) -> u16 {
    if amf_set_id.0.len() >= 10 {
        let mut value: u16 = 0;
        for (i, bit) in amf_set_id.0.iter().take(10).enumerate() {
            if *bit {
                value |= 1 << (9 - i);
            }
        }
        value
    } else {
        0
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Build and encode an Initial UE Message to bytes
///
/// # Arguments
/// * `params` - Parameters for the Initial UE Message
///
/// # Returns
/// * `Ok(Vec<u8>)` - The encoded bytes
/// * `Err(InitialUeMessageError)` - If building or encoding fails
pub fn encode_initial_ue_message(
    params: &InitialUeMessageParams,
) -> Result<Vec<u8>, InitialUeMessageError> {
    let pdu = build_initial_ue_message(params)?;
    Ok(encode_ngap_pdu(&pdu)?)
}

/// Decode and parse an Initial UE Message from bytes
///
/// # Arguments
/// * `bytes` - The encoded bytes
///
/// # Returns
/// * `Ok(InitialUeMessageData)` - The parsed message data
/// * `Err(InitialUeMessageError)` - If decoding or parsing fails
pub fn decode_initial_ue_message(bytes: &[u8]) -> Result<InitialUeMessageData, InitialUeMessageError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_initial_ue_message(&pdu)
}

/// Check if an NGAP PDU is an Initial UE Message
pub fn is_initial_ue_message(pdu: &NGAP_PDU) -> bool {
    matches!(
        pdu,
        NGAP_PDU::InitiatingMessage(msg)
            if matches!(msg.value, InitiatingMessageValue::Id_InitialUEMessage(_))
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_params() -> InitialUeMessageParams {
        InitialUeMessageParams {
            ran_ue_ngap_id: 1,
            nas_pdu: vec![0x7e, 0x00, 0x41, 0x79, 0x00, 0x0d], // Sample NAS PDU
            user_location_info: UserLocationInfoNr {
                nr_cgi: NrCgi {
                    plmn_identity: [0x00, 0xF1, 0x10], // MCC=001, MNC=01
                    nr_cell_identity: 0x000000001,     // Cell ID = 1
                },
                tai: Tai {
                    plmn_identity: [0x00, 0xF1, 0x10],
                    tac: [0x00, 0x00, 0x01], // TAC = 1
                },
                time_stamp: None,
            },
            rrc_establishment_cause: RrcEstablishmentCauseValue::MoSignalling,
            five_g_s_tmsi: None,
            amf_set_id: None,
            ue_context_request: Some(UeContextRequestValue::Requested),
            allowed_nssai: None,
        }
    }

    #[test]
    fn test_build_initial_ue_message() {
        let params = create_test_params();
        let result = build_initial_ue_message(&params);
        assert!(result.is_ok());

        let pdu = result.unwrap();
        match pdu {
            NGAP_PDU::InitiatingMessage(msg) => {
                assert_eq!(msg.procedure_code.0, ID_INITIAL_UE_MESSAGE);
            }
            _ => panic!("Expected InitiatingMessage"),
        }
    }

    #[test]
    fn test_encode_initial_ue_message() {
        let params = create_test_params();
        let result = encode_initial_ue_message(&params);
        assert!(result.is_ok());

        let bytes = result.unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_initial_ue_message_roundtrip() {
        let params = create_test_params();

        // Build and encode
        let pdu = build_initial_ue_message(&params).expect("Failed to build message");
        let encoded = encode_ngap_pdu(&pdu).expect("Failed to encode");

        // Decode
        let decoded_pdu = decode_ngap_pdu(&encoded).expect("Failed to decode");

        // Verify structure
        match decoded_pdu {
            NGAP_PDU::InitiatingMessage(msg) => {
                assert_eq!(msg.procedure_code.0, ID_INITIAL_UE_MESSAGE);
                match msg.value {
                    InitiatingMessageValue::Id_InitialUEMessage(initial_msg) => {
                        // Verify we have the expected IEs (at least 4 mandatory + 1 optional)
                        assert!(initial_msg.protocol_i_es.0.len() >= 4);
                    }
                    _ => panic!("Expected InitialUEMessage"),
                }
            }
            _ => panic!("Expected InitiatingMessage"),
        }
    }

    #[test]
    fn test_initial_ue_message_parse_roundtrip() {
        let params = create_test_params();

        // Build, encode, decode, and parse
        let encoded = encode_initial_ue_message(&params).expect("Failed to encode");
        let parsed = decode_initial_ue_message(&encoded).expect("Failed to decode and parse");

        // Verify parsed data matches original params
        assert_eq!(parsed.ran_ue_ngap_id, params.ran_ue_ngap_id);
        assert_eq!(parsed.nas_pdu, params.nas_pdu);
        assert_eq!(
            parsed.user_location_info.nr_cgi.plmn_identity,
            params.user_location_info.nr_cgi.plmn_identity
        );
        assert_eq!(
            parsed.user_location_info.nr_cgi.nr_cell_identity,
            params.user_location_info.nr_cgi.nr_cell_identity
        );
        assert_eq!(
            parsed.user_location_info.tai.plmn_identity,
            params.user_location_info.tai.plmn_identity
        );
        assert_eq!(
            parsed.user_location_info.tai.tac,
            params.user_location_info.tai.tac
        );
        assert_eq!(
            parsed.rrc_establishment_cause,
            params.rrc_establishment_cause
        );
        assert_eq!(parsed.ue_context_request, params.ue_context_request);
    }

    #[test]
    fn test_initial_ue_message_with_five_g_s_tmsi() {
        let mut params = create_test_params();
        params.five_g_s_tmsi = Some(FiveGSTmsi {
            amf_set_id: 1,
            amf_pointer: 0,
            five_g_tmsi: [0x00, 0x00, 0x00, 0x01],
        });

        let encoded = encode_initial_ue_message(&params).expect("Failed to encode");
        let parsed = decode_initial_ue_message(&encoded).expect("Failed to decode and parse");

        assert!(parsed.five_g_s_tmsi.is_some());
        let tmsi = parsed.five_g_s_tmsi.unwrap();
        assert_eq!(tmsi.amf_set_id, 1);
        assert_eq!(tmsi.amf_pointer, 0);
        assert_eq!(tmsi.five_g_tmsi, [0x00, 0x00, 0x00, 0x01]);
    }

    #[test]
    fn test_initial_ue_message_with_amf_set_id() {
        let mut params = create_test_params();
        params.amf_set_id = Some(5);

        let encoded = encode_initial_ue_message(&params).expect("Failed to encode");
        let parsed = decode_initial_ue_message(&encoded).expect("Failed to decode and parse");

        assert_eq!(parsed.amf_set_id, Some(5));
    }

    #[test]
    fn test_rrc_establishment_cause_conversion() {
        // Test all cause values
        let causes = [
            RrcEstablishmentCauseValue::Emergency,
            RrcEstablishmentCauseValue::HighPriorityAccess,
            RrcEstablishmentCauseValue::MtAccess,
            RrcEstablishmentCauseValue::MoSignalling,
            RrcEstablishmentCauseValue::MoData,
            RrcEstablishmentCauseValue::MoVoiceCall,
            RrcEstablishmentCauseValue::MoVideoCall,
            RrcEstablishmentCauseValue::MoSms,
            RrcEstablishmentCauseValue::MpsHighPriorityAccess,
            RrcEstablishmentCauseValue::McsHighPriorityAccess,
        ];

        for cause in causes {
            let rrc_cause: RRCEstablishmentCause = cause.into();
            let converted_back: RrcEstablishmentCauseValue = rrc_cause.try_into().unwrap();
            assert_eq!(cause, converted_back);
        }
    }

    #[test]
    fn test_is_initial_ue_message() {
        let params = create_test_params();
        let pdu = build_initial_ue_message(&params).expect("Failed to build message");
        assert!(is_initial_ue_message(&pdu));
    }
}
