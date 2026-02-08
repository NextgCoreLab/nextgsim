//! PDU Session Resource Procedures
//!
//! Implements the PDU Session Resource procedures as defined in 3GPP TS 38.413:
//! - Section 8.2.1: PDU Session Resource Setup
//! - Section 8.2.2: PDU Session Resource Release
//! - Section 8.2.3: PDU Session Resource Modify
//!
//! These procedures are used to manage PDU session resources between the AMF and NG-RAN.

use crate::codec::generated::*;
use crate::codec::{decode_ngap_pdu, encode_ngap_pdu, NgapCodecError};
use crate::procedures::ng_setup::NgSetupFailureCause;
use thiserror::Error;

/// Errors that can occur during PDU Session Resource procedures
#[derive(Debug, Error)]
pub enum PduSessionResourceError {
    /// Codec error during encoding/decoding
    #[error("Codec error: {0}")]
    CodecError(#[from] NgapCodecError),

    /// Invalid message type received
    #[error("Invalid message type: expected {expected}, got {actual}")]
    InvalidMessageType {
        /// Expected message type
        expected: String,
        /// Actual message type received
        actual: String,
    },

    /// Missing mandatory IE
    #[error("Missing mandatory IE: {0}")]
    MissingMandatoryIe(String),

    /// Invalid IE value
    #[error("Invalid IE value: {0}")]
    InvalidIeValue(String),
}

/// Cause values for PDU Session Resource procedures
pub type PduSessionResourceCause = NgSetupFailureCause;

// ============================================================================
// PDU Session Resource Setup Request
// ============================================================================

/// PDU Session Resource item for setup request
#[derive(Debug, Clone)]
pub struct PduSessionResourceSetupItem {
    /// PDU Session ID (0-255)
    pub pdu_session_id: u8,
    /// NAS PDU (optional)
    pub nas_pdu: Option<Vec<u8>>,
    /// S-NSSAI
    pub s_nssai: SnssaiValue,
    /// PDU Session Resource Setup Request Transfer (opaque bytes)
    pub transfer: Vec<u8>,
}

/// S-NSSAI value
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnssaiValue {
    /// Slice/Service Type (SST) - 1 byte
    pub sst: u8,
    /// Slice Differentiator (SD) - 3 bytes, optional
    pub sd: Option<[u8; 3]>,
}

/// Parsed PDU Session Resource Setup Request data
#[derive(Debug, Clone)]
pub struct PduSessionResourceSetupRequestData {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
    /// NAS PDU (optional)
    pub nas_pdu: Option<Vec<u8>>,
    /// PDU Session Resource Setup List
    pub pdu_session_resource_setup_list: Vec<PduSessionResourceSetupItem>,
}

/// Parse a PDU Session Resource Setup Request from an NGAP PDU
pub fn parse_pdu_session_resource_setup_request(
    pdu: &NGAP_PDU,
) -> Result<PduSessionResourceSetupRequestData, PduSessionResourceError> {
    let initiating_message = match pdu {
        NGAP_PDU::InitiatingMessage(msg) => msg,
        _ => {
            return Err(PduSessionResourceError::InvalidMessageType {
                expected: "InitiatingMessage".to_string(),
                actual: format!("{pdu:?}"),
            })
        }
    };

    let request = match &initiating_message.value {
        InitiatingMessageValue::Id_PDUSessionResourceSetup(req) => req,
        _ => {
            return Err(PduSessionResourceError::InvalidMessageType {
                expected: "PDUSessionResourceSetupRequest".to_string(),
                actual: format!("{:?}", initiating_message.value),
            })
        }
    };

    let mut amf_ue_ngap_id: Option<u64> = None;
    let mut ran_ue_ngap_id: Option<u32> = None;
    let mut nas_pdu: Option<Vec<u8>> = None;
    let mut pdu_session_resource_setup_list: Option<Vec<PduSessionResourceSetupItem>> = None;

    for ie in &request.protocol_i_es.0 {
        match &ie.value {
            PDUSessionResourceSetupRequestProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(id) => {
                amf_ue_ngap_id = Some(id.0);
            }
            PDUSessionResourceSetupRequestProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(id) => {
                ran_ue_ngap_id = Some(id.0);
            }
            PDUSessionResourceSetupRequestProtocolIEs_EntryValue::Id_NAS_PDU(pdu) => {
                nas_pdu = Some(pdu.0.clone());
            }
            PDUSessionResourceSetupRequestProtocolIEs_EntryValue::Id_PDUSessionResourceSetupListSUReq(list) => {
                pdu_session_resource_setup_list = Some(parse_setup_list_su_req(list));
            }
            _ => {}
        }
    }

    Ok(PduSessionResourceSetupRequestData {
        amf_ue_ngap_id: amf_ue_ngap_id.ok_or_else(|| {
            PduSessionResourceError::MissingMandatoryIe("AMF-UE-NGAP-ID".to_string())
        })?,
        ran_ue_ngap_id: ran_ue_ngap_id.ok_or_else(|| {
            PduSessionResourceError::MissingMandatoryIe("RAN-UE-NGAP-ID".to_string())
        })?,
        nas_pdu,
        pdu_session_resource_setup_list: pdu_session_resource_setup_list.ok_or_else(|| {
            PduSessionResourceError::MissingMandatoryIe("PDUSessionResourceSetupListSUReq".to_string())
        })?,
    })
}

fn parse_setup_list_su_req(list: &PDUSessionResourceSetupListSUReq) -> Vec<PduSessionResourceSetupItem> {
    list.0
        .iter()
        .map(|item| {
            let sst = item.s_nssai.sst.0.first().copied().unwrap_or(0);
            let sd = item.s_nssai.sd.as_ref().and_then(|sd| sd.0.as_slice().try_into().ok());
            PduSessionResourceSetupItem {
                pdu_session_id: item.pdu_session_id.0,
                nas_pdu: item.pdu_session_nas_pdu.as_ref().map(|p| p.0.clone()),
                s_nssai: SnssaiValue { sst, sd },
                transfer: item.pdu_session_resource_setup_request_transfer.0.clone(),
            }
        })
        .collect()
}

// ============================================================================
// PDU Session Resource Setup Response
// ============================================================================

/// PDU Session Resource item for setup response (successful)
#[derive(Debug, Clone)]
pub struct PduSessionResourceSetupResponseItem {
    /// PDU Session ID (0-255)
    pub pdu_session_id: u8,
    /// PDU Session Resource Setup Response Transfer (opaque bytes)
    pub transfer: Vec<u8>,
}

/// PDU Session Resource item for setup response (failed)
#[derive(Debug, Clone)]
pub struct PduSessionResourceFailedToSetupItem {
    /// PDU Session ID (0-255)
    pub pdu_session_id: u8,
    /// PDU Session Resource Setup Unsuccessful Transfer (opaque bytes)
    pub transfer: Vec<u8>,
}

/// Parameters for building a PDU Session Resource Setup Response
#[derive(Debug, Clone)]
pub struct PduSessionResourceSetupResponseParams {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
    /// Successfully setup PDU sessions (optional)
    pub setup_list: Option<Vec<PduSessionResourceSetupResponseItem>>,
    /// Failed to setup PDU sessions (optional)
    pub failed_list: Option<Vec<PduSessionResourceFailedToSetupItem>>,
}

/// Parsed PDU Session Resource Setup Response data
#[derive(Debug, Clone)]
pub struct PduSessionResourceSetupResponseData {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
    /// Successfully setup PDU sessions
    pub setup_list: Option<Vec<PduSessionResourceSetupResponseItem>>,
    /// Failed to setup PDU sessions
    pub failed_list: Option<Vec<PduSessionResourceFailedToSetupItem>>,
}

/// Build a PDU Session Resource Setup Response PDU
pub fn build_pdu_session_resource_setup_response(
    params: &PduSessionResourceSetupResponseParams,
) -> Result<NGAP_PDU, PduSessionResourceError> {
    let mut protocol_ies = Vec::new();

    // IE: AMF-UE-NGAP-ID (mandatory)
    protocol_ies.push(PDUSessionResourceSetupResponseProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_AMF_UE_NGAP_ID),
        criticality: Criticality(Criticality::IGNORE),
        value: PDUSessionResourceSetupResponseProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(
            AMF_UE_NGAP_ID(params.amf_ue_ngap_id),
        ),
    });

    // IE: RAN-UE-NGAP-ID (mandatory)
    protocol_ies.push(PDUSessionResourceSetupResponseProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_RAN_UE_NGAP_ID),
        criticality: Criticality(Criticality::IGNORE),
        value: PDUSessionResourceSetupResponseProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(
            RAN_UE_NGAP_ID(params.ran_ue_ngap_id),
        ),
    });

    // IE: PDUSessionResourceSetupListSURes (optional)
    if let Some(ref setup_list) = params.setup_list {
        let list = build_setup_list_su_res(setup_list);
        protocol_ies.push(PDUSessionResourceSetupResponseProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_PDU_SESSION_RESOURCE_SETUP_LIST_SU_RES),
            criticality: Criticality(Criticality::IGNORE),
            value: PDUSessionResourceSetupResponseProtocolIEs_EntryValue::Id_PDUSessionResourceSetupListSURes(list),
        });
    }

    // IE: PDUSessionResourceFailedToSetupListSURes (optional)
    if let Some(ref failed_list) = params.failed_list {
        let list = build_failed_to_setup_list_su_res(failed_list);
        protocol_ies.push(PDUSessionResourceSetupResponseProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_PDU_SESSION_RESOURCE_FAILED_TO_SETUP_LIST_SU_RES),
            criticality: Criticality(Criticality::IGNORE),
            value: PDUSessionResourceSetupResponseProtocolIEs_EntryValue::Id_PDUSessionResourceFailedToSetupListSURes(list),
        });
    }

    let response = PDUSessionResourceSetupResponse {
        protocol_i_es: PDUSessionResourceSetupResponseProtocolIEs(protocol_ies),
    };

    let successful_outcome = SuccessfulOutcome {
        procedure_code: ProcedureCode(ID_PDU_SESSION_RESOURCE_SETUP),
        criticality: Criticality(Criticality::REJECT),
        value: SuccessfulOutcomeValue::Id_PDUSessionResourceSetup(response),
    };

    Ok(NGAP_PDU::SuccessfulOutcome(successful_outcome))
}

fn build_setup_list_su_res(items: &[PduSessionResourceSetupResponseItem]) -> PDUSessionResourceSetupListSURes {
    let list: Vec<PDUSessionResourceSetupItemSURes> = items
        .iter()
        .map(|item| PDUSessionResourceSetupItemSURes {
            pdu_session_id: PDUSessionID(item.pdu_session_id),
            pdu_session_resource_setup_response_transfer: PDUSessionResourceSetupItemSUResPDUSessionResourceSetupResponseTransfer(item.transfer.clone()),
            ie_extensions: None,
        })
        .collect();
    PDUSessionResourceSetupListSURes(list)
}

fn build_failed_to_setup_list_su_res(items: &[PduSessionResourceFailedToSetupItem]) -> PDUSessionResourceFailedToSetupListSURes {
    let list: Vec<PDUSessionResourceFailedToSetupItemSURes> = items
        .iter()
        .map(|item| PDUSessionResourceFailedToSetupItemSURes {
            pdu_session_id: PDUSessionID(item.pdu_session_id),
            pdu_session_resource_setup_unsuccessful_transfer: PDUSessionResourceFailedToSetupItemSUResPDUSessionResourceSetupUnsuccessfulTransfer(item.transfer.clone()),
            ie_extensions: None,
        })
        .collect();
    PDUSessionResourceFailedToSetupListSURes(list)
}

/// Parse a PDU Session Resource Setup Response from an NGAP PDU
pub fn parse_pdu_session_resource_setup_response(
    pdu: &NGAP_PDU,
) -> Result<PduSessionResourceSetupResponseData, PduSessionResourceError> {
    let successful_outcome = match pdu {
        NGAP_PDU::SuccessfulOutcome(outcome) => outcome,
        _ => {
            return Err(PduSessionResourceError::InvalidMessageType {
                expected: "SuccessfulOutcome".to_string(),
                actual: format!("{pdu:?}"),
            })
        }
    };

    let response = match &successful_outcome.value {
        SuccessfulOutcomeValue::Id_PDUSessionResourceSetup(resp) => resp,
        _ => {
            return Err(PduSessionResourceError::InvalidMessageType {
                expected: "PDUSessionResourceSetupResponse".to_string(),
                actual: format!("{:?}", successful_outcome.value),
            })
        }
    };

    let mut amf_ue_ngap_id: Option<u64> = None;
    let mut ran_ue_ngap_id: Option<u32> = None;
    let mut setup_list: Option<Vec<PduSessionResourceSetupResponseItem>> = None;
    let mut failed_list: Option<Vec<PduSessionResourceFailedToSetupItem>> = None;

    for ie in &response.protocol_i_es.0 {
        match &ie.value {
            PDUSessionResourceSetupResponseProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(id) => {
                amf_ue_ngap_id = Some(id.0);
            }
            PDUSessionResourceSetupResponseProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(id) => {
                ran_ue_ngap_id = Some(id.0);
            }
            PDUSessionResourceSetupResponseProtocolIEs_EntryValue::Id_PDUSessionResourceSetupListSURes(list) => {
                setup_list = Some(parse_setup_list_su_res(list));
            }
            PDUSessionResourceSetupResponseProtocolIEs_EntryValue::Id_PDUSessionResourceFailedToSetupListSURes(list) => {
                failed_list = Some(parse_failed_to_setup_list_su_res(list));
            }
            _ => {}
        }
    }

    Ok(PduSessionResourceSetupResponseData {
        amf_ue_ngap_id: amf_ue_ngap_id.ok_or_else(|| {
            PduSessionResourceError::MissingMandatoryIe("AMF-UE-NGAP-ID".to_string())
        })?,
        ran_ue_ngap_id: ran_ue_ngap_id.ok_or_else(|| {
            PduSessionResourceError::MissingMandatoryIe("RAN-UE-NGAP-ID".to_string())
        })?,
        setup_list,
        failed_list,
    })
}

fn parse_setup_list_su_res(list: &PDUSessionResourceSetupListSURes) -> Vec<PduSessionResourceSetupResponseItem> {
    list.0
        .iter()
        .map(|item| PduSessionResourceSetupResponseItem {
            pdu_session_id: item.pdu_session_id.0,
            transfer: item.pdu_session_resource_setup_response_transfer.0.clone(),
        })
        .collect()
}

fn parse_failed_to_setup_list_su_res(list: &PDUSessionResourceFailedToSetupListSURes) -> Vec<PduSessionResourceFailedToSetupItem> {
    list.0
        .iter()
        .map(|item| PduSessionResourceFailedToSetupItem {
            pdu_session_id: item.pdu_session_id.0,
            transfer: item.pdu_session_resource_setup_unsuccessful_transfer.0.clone(),
        })
        .collect()
}


// ============================================================================
// PDU Session Resource Release Command
// ============================================================================

/// PDU Session Resource item for release command
#[derive(Debug, Clone)]
pub struct PduSessionResourceToReleaseItem {
    /// PDU Session ID (0-255)
    pub pdu_session_id: u8,
    /// PDU Session Resource Release Command Transfer (opaque bytes)
    pub transfer: Vec<u8>,
}

/// Parsed PDU Session Resource Release Command data
#[derive(Debug, Clone)]
pub struct PduSessionResourceReleaseCommandData {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
    /// NAS PDU (optional)
    pub nas_pdu: Option<Vec<u8>>,
    /// PDU Session Resource To Release List
    pub pdu_session_resource_to_release_list: Vec<PduSessionResourceToReleaseItem>,
}

/// Parse a PDU Session Resource Release Command from an NGAP PDU
pub fn parse_pdu_session_resource_release_command(
    pdu: &NGAP_PDU,
) -> Result<PduSessionResourceReleaseCommandData, PduSessionResourceError> {
    let initiating_message = match pdu {
        NGAP_PDU::InitiatingMessage(msg) => msg,
        _ => {
            return Err(PduSessionResourceError::InvalidMessageType {
                expected: "InitiatingMessage".to_string(),
                actual: format!("{pdu:?}"),
            })
        }
    };

    let command = match &initiating_message.value {
        InitiatingMessageValue::Id_PDUSessionResourceRelease(cmd) => cmd,
        _ => {
            return Err(PduSessionResourceError::InvalidMessageType {
                expected: "PDUSessionResourceReleaseCommand".to_string(),
                actual: format!("{:?}", initiating_message.value),
            })
        }
    };

    let mut amf_ue_ngap_id: Option<u64> = None;
    let mut ran_ue_ngap_id: Option<u32> = None;
    let mut nas_pdu: Option<Vec<u8>> = None;
    let mut pdu_session_resource_to_release_list: Option<Vec<PduSessionResourceToReleaseItem>> = None;

    for ie in &command.protocol_i_es.0 {
        match &ie.value {
            PDUSessionResourceReleaseCommandProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(id) => {
                amf_ue_ngap_id = Some(id.0);
            }
            PDUSessionResourceReleaseCommandProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(id) => {
                ran_ue_ngap_id = Some(id.0);
            }
            PDUSessionResourceReleaseCommandProtocolIEs_EntryValue::Id_NAS_PDU(pdu) => {
                nas_pdu = Some(pdu.0.clone());
            }
            PDUSessionResourceReleaseCommandProtocolIEs_EntryValue::Id_PDUSessionResourceToReleaseListRelCmd(list) => {
                pdu_session_resource_to_release_list = Some(parse_to_release_list_rel_cmd(list));
            }
            _ => {}
        }
    }

    Ok(PduSessionResourceReleaseCommandData {
        amf_ue_ngap_id: amf_ue_ngap_id.ok_or_else(|| {
            PduSessionResourceError::MissingMandatoryIe("AMF-UE-NGAP-ID".to_string())
        })?,
        ran_ue_ngap_id: ran_ue_ngap_id.ok_or_else(|| {
            PduSessionResourceError::MissingMandatoryIe("RAN-UE-NGAP-ID".to_string())
        })?,
        nas_pdu,
        pdu_session_resource_to_release_list: pdu_session_resource_to_release_list.ok_or_else(|| {
            PduSessionResourceError::MissingMandatoryIe("PDUSessionResourceToReleaseListRelCmd".to_string())
        })?,
    })
}

fn parse_to_release_list_rel_cmd(list: &PDUSessionResourceToReleaseListRelCmd) -> Vec<PduSessionResourceToReleaseItem> {
    list.0
        .iter()
        .map(|item| PduSessionResourceToReleaseItem {
            pdu_session_id: item.pdu_session_id.0,
            transfer: item.pdu_session_resource_release_command_transfer.0.clone(),
        })
        .collect()
}

// ============================================================================
// PDU Session Resource Release Response
// ============================================================================

/// PDU Session Resource item for release response
#[derive(Debug, Clone)]
pub struct PduSessionResourceReleasedItem {
    /// PDU Session ID (0-255)
    pub pdu_session_id: u8,
    /// PDU Session Resource Release Response Transfer (opaque bytes)
    pub transfer: Vec<u8>,
}

/// Parameters for building a PDU Session Resource Release Response
#[derive(Debug, Clone)]
pub struct PduSessionResourceReleaseResponseParams {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
    /// Released PDU sessions
    pub released_list: Vec<PduSessionResourceReleasedItem>,
}

/// Parsed PDU Session Resource Release Response data
#[derive(Debug, Clone)]
pub struct PduSessionResourceReleaseResponseData {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
    /// Released PDU sessions
    pub released_list: Vec<PduSessionResourceReleasedItem>,
}

/// Build a PDU Session Resource Release Response PDU
pub fn build_pdu_session_resource_release_response(
    params: &PduSessionResourceReleaseResponseParams,
) -> Result<NGAP_PDU, PduSessionResourceError> {
    let mut protocol_ies = Vec::new();

    // IE: AMF-UE-NGAP-ID (mandatory)
    protocol_ies.push(PDUSessionResourceReleaseResponseProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_AMF_UE_NGAP_ID),
        criticality: Criticality(Criticality::IGNORE),
        value: PDUSessionResourceReleaseResponseProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(
            AMF_UE_NGAP_ID(params.amf_ue_ngap_id),
        ),
    });

    // IE: RAN-UE-NGAP-ID (mandatory)
    protocol_ies.push(PDUSessionResourceReleaseResponseProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_RAN_UE_NGAP_ID),
        criticality: Criticality(Criticality::IGNORE),
        value: PDUSessionResourceReleaseResponseProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(
            RAN_UE_NGAP_ID(params.ran_ue_ngap_id),
        ),
    });

    // IE: PDUSessionResourceReleasedListRelRes (mandatory)
    let list = build_released_list_rel_res(&params.released_list);
    protocol_ies.push(PDUSessionResourceReleaseResponseProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_PDU_SESSION_RESOURCE_RELEASED_LIST_REL_RES),
        criticality: Criticality(Criticality::IGNORE),
        value: PDUSessionResourceReleaseResponseProtocolIEs_EntryValue::Id_PDUSessionResourceReleasedListRelRes(list),
    });

    let response = PDUSessionResourceReleaseResponse {
        protocol_i_es: PDUSessionResourceReleaseResponseProtocolIEs(protocol_ies),
    };

    let successful_outcome = SuccessfulOutcome {
        procedure_code: ProcedureCode(ID_PDU_SESSION_RESOURCE_RELEASE),
        criticality: Criticality(Criticality::REJECT),
        value: SuccessfulOutcomeValue::Id_PDUSessionResourceRelease(response),
    };

    Ok(NGAP_PDU::SuccessfulOutcome(successful_outcome))
}

fn build_released_list_rel_res(items: &[PduSessionResourceReleasedItem]) -> PDUSessionResourceReleasedListRelRes {
    let list: Vec<PDUSessionResourceReleasedItemRelRes> = items
        .iter()
        .map(|item| PDUSessionResourceReleasedItemRelRes {
            pdu_session_id: PDUSessionID(item.pdu_session_id),
            pdu_session_resource_release_response_transfer: PDUSessionResourceReleasedItemRelResPDUSessionResourceReleaseResponseTransfer(item.transfer.clone()),
            ie_extensions: None,
        })
        .collect();
    PDUSessionResourceReleasedListRelRes(list)
}

/// Parse a PDU Session Resource Release Response from an NGAP PDU
pub fn parse_pdu_session_resource_release_response(
    pdu: &NGAP_PDU,
) -> Result<PduSessionResourceReleaseResponseData, PduSessionResourceError> {
    let successful_outcome = match pdu {
        NGAP_PDU::SuccessfulOutcome(outcome) => outcome,
        _ => {
            return Err(PduSessionResourceError::InvalidMessageType {
                expected: "SuccessfulOutcome".to_string(),
                actual: format!("{pdu:?}"),
            })
        }
    };

    let response = match &successful_outcome.value {
        SuccessfulOutcomeValue::Id_PDUSessionResourceRelease(resp) => resp,
        _ => {
            return Err(PduSessionResourceError::InvalidMessageType {
                expected: "PDUSessionResourceReleaseResponse".to_string(),
                actual: format!("{:?}", successful_outcome.value),
            })
        }
    };

    let mut amf_ue_ngap_id: Option<u64> = None;
    let mut ran_ue_ngap_id: Option<u32> = None;
    let mut released_list: Option<Vec<PduSessionResourceReleasedItem>> = None;

    for ie in &response.protocol_i_es.0 {
        match &ie.value {
            PDUSessionResourceReleaseResponseProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(id) => {
                amf_ue_ngap_id = Some(id.0);
            }
            PDUSessionResourceReleaseResponseProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(id) => {
                ran_ue_ngap_id = Some(id.0);
            }
            PDUSessionResourceReleaseResponseProtocolIEs_EntryValue::Id_PDUSessionResourceReleasedListRelRes(list) => {
                released_list = Some(parse_released_list_rel_res(list));
            }
            _ => {}
        }
    }

    Ok(PduSessionResourceReleaseResponseData {
        amf_ue_ngap_id: amf_ue_ngap_id.ok_or_else(|| {
            PduSessionResourceError::MissingMandatoryIe("AMF-UE-NGAP-ID".to_string())
        })?,
        ran_ue_ngap_id: ran_ue_ngap_id.ok_or_else(|| {
            PduSessionResourceError::MissingMandatoryIe("RAN-UE-NGAP-ID".to_string())
        })?,
        released_list: released_list.ok_or_else(|| {
            PduSessionResourceError::MissingMandatoryIe("PDUSessionResourceReleasedListRelRes".to_string())
        })?,
    })
}

fn parse_released_list_rel_res(list: &PDUSessionResourceReleasedListRelRes) -> Vec<PduSessionResourceReleasedItem> {
    list.0
        .iter()
        .map(|item| PduSessionResourceReleasedItem {
            pdu_session_id: item.pdu_session_id.0,
            transfer: item.pdu_session_resource_release_response_transfer.0.clone(),
        })
        .collect()
}


// ============================================================================
// PDU Session Resource Modify Request
// ============================================================================

/// PDU Session Resource item for modify request
#[derive(Debug, Clone)]
pub struct PduSessionResourceModifyRequestItem {
    /// PDU Session ID (0-255)
    pub pdu_session_id: u8,
    /// NAS PDU (optional)
    pub nas_pdu: Option<Vec<u8>>,
    /// PDU Session Resource Modify Request Transfer (opaque bytes)
    pub transfer: Vec<u8>,
}

/// Parsed PDU Session Resource Modify Request data
#[derive(Debug, Clone)]
pub struct PduSessionResourceModifyRequestData {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
    /// PDU Session Resource Modify List
    pub pdu_session_resource_modify_list: Vec<PduSessionResourceModifyRequestItem>,
}

/// Parse a PDU Session Resource Modify Request from an NGAP PDU
pub fn parse_pdu_session_resource_modify_request(
    pdu: &NGAP_PDU,
) -> Result<PduSessionResourceModifyRequestData, PduSessionResourceError> {
    let initiating_message = match pdu {
        NGAP_PDU::InitiatingMessage(msg) => msg,
        _ => {
            return Err(PduSessionResourceError::InvalidMessageType {
                expected: "InitiatingMessage".to_string(),
                actual: format!("{pdu:?}"),
            })
        }
    };

    let request = match &initiating_message.value {
        InitiatingMessageValue::Id_PDUSessionResourceModify(req) => req,
        _ => {
            return Err(PduSessionResourceError::InvalidMessageType {
                expected: "PDUSessionResourceModifyRequest".to_string(),
                actual: format!("{:?}", initiating_message.value),
            })
        }
    };

    let mut amf_ue_ngap_id: Option<u64> = None;
    let mut ran_ue_ngap_id: Option<u32> = None;
    let mut pdu_session_resource_modify_list: Option<Vec<PduSessionResourceModifyRequestItem>> = None;

    for ie in &request.protocol_i_es.0 {
        match &ie.value {
            PDUSessionResourceModifyRequestProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(id) => {
                amf_ue_ngap_id = Some(id.0);
            }
            PDUSessionResourceModifyRequestProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(id) => {
                ran_ue_ngap_id = Some(id.0);
            }
            PDUSessionResourceModifyRequestProtocolIEs_EntryValue::Id_PDUSessionResourceModifyListModReq(list) => {
                pdu_session_resource_modify_list = Some(parse_modify_list_mod_req(list));
            }
            _ => {}
        }
    }

    Ok(PduSessionResourceModifyRequestData {
        amf_ue_ngap_id: amf_ue_ngap_id.ok_or_else(|| {
            PduSessionResourceError::MissingMandatoryIe("AMF-UE-NGAP-ID".to_string())
        })?,
        ran_ue_ngap_id: ran_ue_ngap_id.ok_or_else(|| {
            PduSessionResourceError::MissingMandatoryIe("RAN-UE-NGAP-ID".to_string())
        })?,
        pdu_session_resource_modify_list: pdu_session_resource_modify_list.ok_or_else(|| {
            PduSessionResourceError::MissingMandatoryIe("PDUSessionResourceModifyListModReq".to_string())
        })?,
    })
}

fn parse_modify_list_mod_req(list: &PDUSessionResourceModifyListModReq) -> Vec<PduSessionResourceModifyRequestItem> {
    list.0
        .iter()
        .map(|item| PduSessionResourceModifyRequestItem {
            pdu_session_id: item.pdu_session_id.0,
            nas_pdu: item.nas_pdu.as_ref().map(|p| p.0.clone()),
            transfer: item.pdu_session_resource_modify_request_transfer.0.clone(),
        })
        .collect()
}

// ============================================================================
// PDU Session Resource Modify Response
// ============================================================================

/// PDU Session Resource item for modify response (successful)
#[derive(Debug, Clone)]
pub struct PduSessionResourceModifyResponseItem {
    /// PDU Session ID (0-255)
    pub pdu_session_id: u8,
    /// PDU Session Resource Modify Response Transfer (opaque bytes)
    pub transfer: Vec<u8>,
}

/// PDU Session Resource item for modify response (failed)
#[derive(Debug, Clone)]
pub struct PduSessionResourceFailedToModifyItem {
    /// PDU Session ID (0-255)
    pub pdu_session_id: u8,
    /// PDU Session Resource Modify Unsuccessful Transfer (opaque bytes)
    pub transfer: Vec<u8>,
}

/// Parameters for building a PDU Session Resource Modify Response
#[derive(Debug, Clone)]
pub struct PduSessionResourceModifyResponseParams {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
    /// Successfully modified PDU sessions (optional)
    pub modify_list: Option<Vec<PduSessionResourceModifyResponseItem>>,
    /// Failed to modify PDU sessions (optional)
    pub failed_list: Option<Vec<PduSessionResourceFailedToModifyItem>>,
}

/// Parsed PDU Session Resource Modify Response data
#[derive(Debug, Clone)]
pub struct PduSessionResourceModifyResponseData {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
    /// Successfully modified PDU sessions
    pub modify_list: Option<Vec<PduSessionResourceModifyResponseItem>>,
    /// Failed to modify PDU sessions
    pub failed_list: Option<Vec<PduSessionResourceFailedToModifyItem>>,
}

/// Build a PDU Session Resource Modify Response PDU
pub fn build_pdu_session_resource_modify_response(
    params: &PduSessionResourceModifyResponseParams,
) -> Result<NGAP_PDU, PduSessionResourceError> {
    let mut protocol_ies = Vec::new();

    // IE: AMF-UE-NGAP-ID (mandatory)
    protocol_ies.push(PDUSessionResourceModifyResponseProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_AMF_UE_NGAP_ID),
        criticality: Criticality(Criticality::IGNORE),
        value: PDUSessionResourceModifyResponseProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(
            AMF_UE_NGAP_ID(params.amf_ue_ngap_id),
        ),
    });

    // IE: RAN-UE-NGAP-ID (mandatory)
    protocol_ies.push(PDUSessionResourceModifyResponseProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_RAN_UE_NGAP_ID),
        criticality: Criticality(Criticality::IGNORE),
        value: PDUSessionResourceModifyResponseProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(
            RAN_UE_NGAP_ID(params.ran_ue_ngap_id),
        ),
    });

    // IE: PDUSessionResourceModifyListModRes (optional)
    if let Some(ref modify_list) = params.modify_list {
        let list = build_modify_list_mod_res(modify_list);
        protocol_ies.push(PDUSessionResourceModifyResponseProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_PDU_SESSION_RESOURCE_MODIFY_LIST_MOD_RES),
            criticality: Criticality(Criticality::IGNORE),
            value: PDUSessionResourceModifyResponseProtocolIEs_EntryValue::Id_PDUSessionResourceModifyListModRes(list),
        });
    }

    // IE: PDUSessionResourceFailedToModifyListModRes (optional)
    if let Some(ref failed_list) = params.failed_list {
        let list = build_failed_to_modify_list_mod_res(failed_list);
        protocol_ies.push(PDUSessionResourceModifyResponseProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_PDU_SESSION_RESOURCE_FAILED_TO_MODIFY_LIST_MOD_RES),
            criticality: Criticality(Criticality::IGNORE),
            value: PDUSessionResourceModifyResponseProtocolIEs_EntryValue::Id_PDUSessionResourceFailedToModifyListModRes(list),
        });
    }

    let response = PDUSessionResourceModifyResponse {
        protocol_i_es: PDUSessionResourceModifyResponseProtocolIEs(protocol_ies),
    };

    let successful_outcome = SuccessfulOutcome {
        procedure_code: ProcedureCode(ID_PDU_SESSION_RESOURCE_MODIFY),
        criticality: Criticality(Criticality::REJECT),
        value: SuccessfulOutcomeValue::Id_PDUSessionResourceModify(response),
    };

    Ok(NGAP_PDU::SuccessfulOutcome(successful_outcome))
}

fn build_modify_list_mod_res(items: &[PduSessionResourceModifyResponseItem]) -> PDUSessionResourceModifyListModRes {
    let list: Vec<PDUSessionResourceModifyItemModRes> = items
        .iter()
        .map(|item| PDUSessionResourceModifyItemModRes {
            pdu_session_id: PDUSessionID(item.pdu_session_id),
            pdu_session_resource_modify_response_transfer: PDUSessionResourceModifyItemModResPDUSessionResourceModifyResponseTransfer(item.transfer.clone()),
            ie_extensions: None,
        })
        .collect();
    PDUSessionResourceModifyListModRes(list)
}

fn build_failed_to_modify_list_mod_res(items: &[PduSessionResourceFailedToModifyItem]) -> PDUSessionResourceFailedToModifyListModRes {
    let list: Vec<PDUSessionResourceFailedToModifyItemModRes> = items
        .iter()
        .map(|item| PDUSessionResourceFailedToModifyItemModRes {
            pdu_session_id: PDUSessionID(item.pdu_session_id),
            pdu_session_resource_modify_unsuccessful_transfer: PDUSessionResourceFailedToModifyItemModResPDUSessionResourceModifyUnsuccessfulTransfer(item.transfer.clone()),
            ie_extensions: None,
        })
        .collect();
    PDUSessionResourceFailedToModifyListModRes(list)
}

/// Parse a PDU Session Resource Modify Response from an NGAP PDU
pub fn parse_pdu_session_resource_modify_response(
    pdu: &NGAP_PDU,
) -> Result<PduSessionResourceModifyResponseData, PduSessionResourceError> {
    let successful_outcome = match pdu {
        NGAP_PDU::SuccessfulOutcome(outcome) => outcome,
        _ => {
            return Err(PduSessionResourceError::InvalidMessageType {
                expected: "SuccessfulOutcome".to_string(),
                actual: format!("{pdu:?}"),
            })
        }
    };

    let response = match &successful_outcome.value {
        SuccessfulOutcomeValue::Id_PDUSessionResourceModify(resp) => resp,
        _ => {
            return Err(PduSessionResourceError::InvalidMessageType {
                expected: "PDUSessionResourceModifyResponse".to_string(),
                actual: format!("{:?}", successful_outcome.value),
            })
        }
    };

    let mut amf_ue_ngap_id: Option<u64> = None;
    let mut ran_ue_ngap_id: Option<u32> = None;
    let mut modify_list: Option<Vec<PduSessionResourceModifyResponseItem>> = None;
    let mut failed_list: Option<Vec<PduSessionResourceFailedToModifyItem>> = None;

    for ie in &response.protocol_i_es.0 {
        match &ie.value {
            PDUSessionResourceModifyResponseProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(id) => {
                amf_ue_ngap_id = Some(id.0);
            }
            PDUSessionResourceModifyResponseProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(id) => {
                ran_ue_ngap_id = Some(id.0);
            }
            PDUSessionResourceModifyResponseProtocolIEs_EntryValue::Id_PDUSessionResourceModifyListModRes(list) => {
                modify_list = Some(parse_modify_list_mod_res(list));
            }
            PDUSessionResourceModifyResponseProtocolIEs_EntryValue::Id_PDUSessionResourceFailedToModifyListModRes(list) => {
                failed_list = Some(parse_failed_to_modify_list_mod_res(list));
            }
            _ => {}
        }
    }

    Ok(PduSessionResourceModifyResponseData {
        amf_ue_ngap_id: amf_ue_ngap_id.ok_or_else(|| {
            PduSessionResourceError::MissingMandatoryIe("AMF-UE-NGAP-ID".to_string())
        })?,
        ran_ue_ngap_id: ran_ue_ngap_id.ok_or_else(|| {
            PduSessionResourceError::MissingMandatoryIe("RAN-UE-NGAP-ID".to_string())
        })?,
        modify_list,
        failed_list,
    })
}

fn parse_modify_list_mod_res(list: &PDUSessionResourceModifyListModRes) -> Vec<PduSessionResourceModifyResponseItem> {
    list.0
        .iter()
        .map(|item| PduSessionResourceModifyResponseItem {
            pdu_session_id: item.pdu_session_id.0,
            transfer: item.pdu_session_resource_modify_response_transfer.0.clone(),
        })
        .collect()
}

fn parse_failed_to_modify_list_mod_res(list: &PDUSessionResourceFailedToModifyListModRes) -> Vec<PduSessionResourceFailedToModifyItem> {
    list.0
        .iter()
        .map(|item| PduSessionResourceFailedToModifyItem {
            pdu_session_id: item.pdu_session_id.0,
            transfer: item.pdu_session_resource_modify_unsuccessful_transfer.0.clone(),
        })
        .collect()
}


// ============================================================================
// Convenience Functions
// ============================================================================

/// Decode and parse a PDU Session Resource Setup Request from bytes
pub fn decode_pdu_session_resource_setup_request(
    bytes: &[u8],
) -> Result<PduSessionResourceSetupRequestData, PduSessionResourceError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_pdu_session_resource_setup_request(&pdu)
}

/// Build and encode a PDU Session Resource Setup Response to bytes
pub fn encode_pdu_session_resource_setup_response(
    params: &PduSessionResourceSetupResponseParams,
) -> Result<Vec<u8>, PduSessionResourceError> {
    let pdu = build_pdu_session_resource_setup_response(params)?;
    Ok(encode_ngap_pdu(&pdu)?)
}

/// Decode and parse a PDU Session Resource Setup Response from bytes
pub fn decode_pdu_session_resource_setup_response(
    bytes: &[u8],
) -> Result<PduSessionResourceSetupResponseData, PduSessionResourceError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_pdu_session_resource_setup_response(&pdu)
}

/// Decode and parse a PDU Session Resource Release Command from bytes
pub fn decode_pdu_session_resource_release_command(
    bytes: &[u8],
) -> Result<PduSessionResourceReleaseCommandData, PduSessionResourceError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_pdu_session_resource_release_command(&pdu)
}

/// Build and encode a PDU Session Resource Release Response to bytes
pub fn encode_pdu_session_resource_release_response(
    params: &PduSessionResourceReleaseResponseParams,
) -> Result<Vec<u8>, PduSessionResourceError> {
    let pdu = build_pdu_session_resource_release_response(params)?;
    Ok(encode_ngap_pdu(&pdu)?)
}

/// Decode and parse a PDU Session Resource Release Response from bytes
pub fn decode_pdu_session_resource_release_response(
    bytes: &[u8],
) -> Result<PduSessionResourceReleaseResponseData, PduSessionResourceError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_pdu_session_resource_release_response(&pdu)
}

/// Decode and parse a PDU Session Resource Modify Request from bytes
pub fn decode_pdu_session_resource_modify_request(
    bytes: &[u8],
) -> Result<PduSessionResourceModifyRequestData, PduSessionResourceError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_pdu_session_resource_modify_request(&pdu)
}

/// Build and encode a PDU Session Resource Modify Response to bytes
pub fn encode_pdu_session_resource_modify_response(
    params: &PduSessionResourceModifyResponseParams,
) -> Result<Vec<u8>, PduSessionResourceError> {
    let pdu = build_pdu_session_resource_modify_response(params)?;
    Ok(encode_ngap_pdu(&pdu)?)
}

/// Decode and parse a PDU Session Resource Modify Response from bytes
pub fn decode_pdu_session_resource_modify_response(
    bytes: &[u8],
) -> Result<PduSessionResourceModifyResponseData, PduSessionResourceError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_pdu_session_resource_modify_response(&pdu)
}

// ============================================================================
// PDU Type Checking Functions
// ============================================================================

/// Check if an NGAP PDU is a PDU Session Resource Setup Request
pub fn is_pdu_session_resource_setup_request(pdu: &NGAP_PDU) -> bool {
    matches!(
        pdu,
        NGAP_PDU::InitiatingMessage(msg)
            if matches!(msg.value, InitiatingMessageValue::Id_PDUSessionResourceSetup(_))
    )
}

/// Check if an NGAP PDU is a PDU Session Resource Setup Response
pub fn is_pdu_session_resource_setup_response(pdu: &NGAP_PDU) -> bool {
    matches!(
        pdu,
        NGAP_PDU::SuccessfulOutcome(outcome)
            if matches!(outcome.value, SuccessfulOutcomeValue::Id_PDUSessionResourceSetup(_))
    )
}

/// Check if an NGAP PDU is a PDU Session Resource Release Command
pub fn is_pdu_session_resource_release_command(pdu: &NGAP_PDU) -> bool {
    matches!(
        pdu,
        NGAP_PDU::InitiatingMessage(msg)
            if matches!(msg.value, InitiatingMessageValue::Id_PDUSessionResourceRelease(_))
    )
}

/// Check if an NGAP PDU is a PDU Session Resource Release Response
pub fn is_pdu_session_resource_release_response(pdu: &NGAP_PDU) -> bool {
    matches!(
        pdu,
        NGAP_PDU::SuccessfulOutcome(outcome)
            if matches!(outcome.value, SuccessfulOutcomeValue::Id_PDUSessionResourceRelease(_))
    )
}

/// Check if an NGAP PDU is a PDU Session Resource Modify Request
pub fn is_pdu_session_resource_modify_request(pdu: &NGAP_PDU) -> bool {
    matches!(
        pdu,
        NGAP_PDU::InitiatingMessage(msg)
            if matches!(msg.value, InitiatingMessageValue::Id_PDUSessionResourceModify(_))
    )
}

/// Check if an NGAP PDU is a PDU Session Resource Modify Response
pub fn is_pdu_session_resource_modify_response(pdu: &NGAP_PDU) -> bool {
    matches!(
        pdu,
        NGAP_PDU::SuccessfulOutcome(outcome)
            if matches!(outcome.value, SuccessfulOutcomeValue::Id_PDUSessionResourceModify(_))
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_setup_response_params() -> PduSessionResourceSetupResponseParams {
        PduSessionResourceSetupResponseParams {
            amf_ue_ngap_id: 12345,
            ran_ue_ngap_id: 1,
            setup_list: Some(vec![PduSessionResourceSetupResponseItem {
                pdu_session_id: 1,
                transfer: vec![0x00, 0x01, 0x02, 0x03],
            }]),
            failed_list: None,
        }
    }

    fn create_test_release_response_params() -> PduSessionResourceReleaseResponseParams {
        PduSessionResourceReleaseResponseParams {
            amf_ue_ngap_id: 12345,
            ran_ue_ngap_id: 1,
            released_list: vec![PduSessionResourceReleasedItem {
                pdu_session_id: 1,
                transfer: vec![0x00, 0x01, 0x02, 0x03],
            }],
        }
    }

    fn create_test_modify_response_params() -> PduSessionResourceModifyResponseParams {
        PduSessionResourceModifyResponseParams {
            amf_ue_ngap_id: 12345,
            ran_ue_ngap_id: 1,
            modify_list: Some(vec![PduSessionResourceModifyResponseItem {
                pdu_session_id: 1,
                transfer: vec![0x00, 0x01, 0x02, 0x03],
            }]),
            failed_list: None,
        }
    }

    #[test]
    fn test_build_pdu_session_resource_setup_response() {
        let params = create_test_setup_response_params();
        let result = build_pdu_session_resource_setup_response(&params);
        assert!(result.is_ok());

        let pdu = result.unwrap();
        match pdu {
            NGAP_PDU::SuccessfulOutcome(outcome) => {
                assert_eq!(outcome.procedure_code.0, ID_PDU_SESSION_RESOURCE_SETUP);
            }
            _ => panic!("Expected SuccessfulOutcome"),
        }
    }

    #[test]
    fn test_encode_pdu_session_resource_setup_response() {
        let params = create_test_setup_response_params();
        let result = encode_pdu_session_resource_setup_response(&params);
        assert!(result.is_ok());

        let bytes = result.unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_pdu_session_resource_setup_response_roundtrip() {
        let params = create_test_setup_response_params();

        let pdu = build_pdu_session_resource_setup_response(&params).expect("Failed to build response");
        let encoded = encode_ngap_pdu(&pdu).expect("Failed to encode");
        let decoded_pdu = decode_ngap_pdu(&encoded).expect("Failed to decode");

        match decoded_pdu {
            NGAP_PDU::SuccessfulOutcome(outcome) => {
                assert_eq!(outcome.procedure_code.0, ID_PDU_SESSION_RESOURCE_SETUP);
            }
            _ => panic!("Expected SuccessfulOutcome"),
        }
    }

    #[test]
    fn test_pdu_session_resource_setup_response_parse_roundtrip() {
        let params = create_test_setup_response_params();

        let encoded = encode_pdu_session_resource_setup_response(&params).expect("Failed to encode");
        let parsed = decode_pdu_session_resource_setup_response(&encoded).expect("Failed to decode and parse");

        assert_eq!(parsed.amf_ue_ngap_id, params.amf_ue_ngap_id);
        assert_eq!(parsed.ran_ue_ngap_id, params.ran_ue_ngap_id);
        assert!(parsed.setup_list.is_some());
        let setup_list = parsed.setup_list.unwrap();
        assert_eq!(setup_list.len(), 1);
        assert_eq!(setup_list[0].pdu_session_id, 1);
    }

    #[test]
    fn test_build_pdu_session_resource_release_response() {
        let params = create_test_release_response_params();
        let result = build_pdu_session_resource_release_response(&params);
        assert!(result.is_ok());

        let pdu = result.unwrap();
        match pdu {
            NGAP_PDU::SuccessfulOutcome(outcome) => {
                assert_eq!(outcome.procedure_code.0, ID_PDU_SESSION_RESOURCE_RELEASE);
            }
            _ => panic!("Expected SuccessfulOutcome"),
        }
    }

    #[test]
    fn test_encode_pdu_session_resource_release_response() {
        let params = create_test_release_response_params();
        let result = encode_pdu_session_resource_release_response(&params);
        assert!(result.is_ok());

        let bytes = result.unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_pdu_session_resource_release_response_roundtrip() {
        let params = create_test_release_response_params();

        let pdu = build_pdu_session_resource_release_response(&params).expect("Failed to build response");
        let encoded = encode_ngap_pdu(&pdu).expect("Failed to encode");
        let decoded_pdu = decode_ngap_pdu(&encoded).expect("Failed to decode");

        match decoded_pdu {
            NGAP_PDU::SuccessfulOutcome(outcome) => {
                assert_eq!(outcome.procedure_code.0, ID_PDU_SESSION_RESOURCE_RELEASE);
            }
            _ => panic!("Expected SuccessfulOutcome"),
        }
    }

    #[test]
    fn test_pdu_session_resource_release_response_parse_roundtrip() {
        let params = create_test_release_response_params();

        let encoded = encode_pdu_session_resource_release_response(&params).expect("Failed to encode");
        let parsed = decode_pdu_session_resource_release_response(&encoded).expect("Failed to decode and parse");

        assert_eq!(parsed.amf_ue_ngap_id, params.amf_ue_ngap_id);
        assert_eq!(parsed.ran_ue_ngap_id, params.ran_ue_ngap_id);
        assert_eq!(parsed.released_list.len(), 1);
        assert_eq!(parsed.released_list[0].pdu_session_id, 1);
    }

    #[test]
    fn test_build_pdu_session_resource_modify_response() {
        let params = create_test_modify_response_params();
        let result = build_pdu_session_resource_modify_response(&params);
        assert!(result.is_ok());

        let pdu = result.unwrap();
        match pdu {
            NGAP_PDU::SuccessfulOutcome(outcome) => {
                assert_eq!(outcome.procedure_code.0, ID_PDU_SESSION_RESOURCE_MODIFY);
            }
            _ => panic!("Expected SuccessfulOutcome"),
        }
    }

    #[test]
    fn test_encode_pdu_session_resource_modify_response() {
        let params = create_test_modify_response_params();
        let result = encode_pdu_session_resource_modify_response(&params);
        assert!(result.is_ok());

        let bytes = result.unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_pdu_session_resource_modify_response_roundtrip() {
        let params = create_test_modify_response_params();

        let pdu = build_pdu_session_resource_modify_response(&params).expect("Failed to build response");
        let encoded = encode_ngap_pdu(&pdu).expect("Failed to encode");
        let decoded_pdu = decode_ngap_pdu(&encoded).expect("Failed to decode");

        match decoded_pdu {
            NGAP_PDU::SuccessfulOutcome(outcome) => {
                assert_eq!(outcome.procedure_code.0, ID_PDU_SESSION_RESOURCE_MODIFY);
            }
            _ => panic!("Expected SuccessfulOutcome"),
        }
    }

    #[test]
    fn test_pdu_session_resource_modify_response_parse_roundtrip() {
        let params = create_test_modify_response_params();

        let encoded = encode_pdu_session_resource_modify_response(&params).expect("Failed to encode");
        let parsed = decode_pdu_session_resource_modify_response(&encoded).expect("Failed to decode and parse");

        assert_eq!(parsed.amf_ue_ngap_id, params.amf_ue_ngap_id);
        assert_eq!(parsed.ran_ue_ngap_id, params.ran_ue_ngap_id);
        assert!(parsed.modify_list.is_some());
        let modify_list = parsed.modify_list.unwrap();
        assert_eq!(modify_list.len(), 1);
        assert_eq!(modify_list[0].pdu_session_id, 1);
    }

    #[test]
    fn test_is_pdu_session_resource_setup_response() {
        let params = create_test_setup_response_params();
        let pdu = build_pdu_session_resource_setup_response(&params).expect("Failed to build response");
        assert!(is_pdu_session_resource_setup_response(&pdu));
        assert!(!is_pdu_session_resource_release_response(&pdu));
        assert!(!is_pdu_session_resource_modify_response(&pdu));
    }

    #[test]
    fn test_is_pdu_session_resource_release_response() {
        let params = create_test_release_response_params();
        let pdu = build_pdu_session_resource_release_response(&params).expect("Failed to build response");
        assert!(is_pdu_session_resource_release_response(&pdu));
        assert!(!is_pdu_session_resource_setup_response(&pdu));
        assert!(!is_pdu_session_resource_modify_response(&pdu));
    }

    #[test]
    fn test_is_pdu_session_resource_modify_response() {
        let params = create_test_modify_response_params();
        let pdu = build_pdu_session_resource_modify_response(&params).expect("Failed to build response");
        assert!(is_pdu_session_resource_modify_response(&pdu));
        assert!(!is_pdu_session_resource_setup_response(&pdu));
        assert!(!is_pdu_session_resource_release_response(&pdu));
    }

    #[test]
    fn test_snssai_value() {
        let snssai = SnssaiValue {
            sst: 1,
            sd: Some([0x00, 0x00, 0x01]),
        };
        assert_eq!(snssai.sst, 1);
        assert_eq!(snssai.sd, Some([0x00, 0x00, 0x01]));

        let snssai_no_sd = SnssaiValue { sst: 1, sd: None };
        assert_eq!(snssai_no_sd.sst, 1);
        assert!(snssai_no_sd.sd.is_none());
    }

    #[test]
    fn test_setup_response_with_failed_list() {
        let params = PduSessionResourceSetupResponseParams {
            amf_ue_ngap_id: 12345,
            ran_ue_ngap_id: 1,
            setup_list: None,
            failed_list: Some(vec![PduSessionResourceFailedToSetupItem {
                pdu_session_id: 1,
                transfer: vec![0x00, 0x01],
            }]),
        };

        let encoded = encode_pdu_session_resource_setup_response(&params).expect("Failed to encode");
        let parsed = decode_pdu_session_resource_setup_response(&encoded).expect("Failed to decode and parse");

        assert!(parsed.setup_list.is_none());
        assert!(parsed.failed_list.is_some());
        let failed_list = parsed.failed_list.unwrap();
        assert_eq!(failed_list.len(), 1);
        assert_eq!(failed_list[0].pdu_session_id, 1);
    }

    #[test]
    fn test_modify_response_with_failed_list() {
        let params = PduSessionResourceModifyResponseParams {
            amf_ue_ngap_id: 12345,
            ran_ue_ngap_id: 1,
            modify_list: None,
            failed_list: Some(vec![PduSessionResourceFailedToModifyItem {
                pdu_session_id: 1,
                transfer: vec![0x00, 0x01],
            }]),
        };

        let encoded = encode_pdu_session_resource_modify_response(&params).expect("Failed to encode");
        let parsed = decode_pdu_session_resource_modify_response(&encoded).expect("Failed to decode and parse");

        assert!(parsed.modify_list.is_none());
        assert!(parsed.failed_list.is_some());
        let failed_list = parsed.failed_list.unwrap();
        assert_eq!(failed_list.len(), 1);
        assert_eq!(failed_list[0].pdu_session_id, 1);
    }
}
