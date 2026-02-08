//! Handover Procedures
//!
//! Implements the Handover procedures as defined in 3GPP TS 38.413:
//! - Section 8.4.1: Handover Preparation (Handover Required/Command/Preparation Failure)
//! - Section 8.4.2: Handover Resource Allocation (Handover Request/Request Acknowledge/Failure)
//! - Section 8.4.3: Handover Notification (Handover Notify)
//! - Section 8.4.4: Handover Cancellation (Handover Cancel/Cancel Acknowledge)
//!
//! These procedures are used to manage UE handover between NG-RAN nodes.

use crate::codec::generated::*;
use crate::codec::{decode_ngap_pdu, encode_ngap_pdu, NgapCodecError};
use crate::procedures::ng_setup::NgSetupFailureCause;
use thiserror::Error;

/// Errors that can occur during Handover procedures
#[derive(Debug, Error)]
pub enum HandoverError {
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

/// Cause values for Handover procedures
pub type HandoverCause = NgSetupFailureCause;

/// Handover type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandoverTypeValue {
    /// Intra-5GS handover
    Intra5gs,
    /// 5GS to EPS handover
    FivegsToEps,
    /// EPS to 5GS handover
    EpsTo5gs,
}

impl From<HandoverTypeValue> for HandoverType {
    fn from(value: HandoverTypeValue) -> Self {
        let v = match value {
            HandoverTypeValue::Intra5gs => HandoverType::INTRA5GS,
            HandoverTypeValue::FivegsToEps => HandoverType::FIVEGS_TO_EPS,
            HandoverTypeValue::EpsTo5gs => HandoverType::EPS_TO_5GS,
        };
        HandoverType(v)
    }
}

impl TryFrom<&HandoverType> for HandoverTypeValue {
    type Error = HandoverError;

    fn try_from(value: &HandoverType) -> Result<Self, Self::Error> {
        match value.0 {
            HandoverType::INTRA5GS => Ok(HandoverTypeValue::Intra5gs),
            HandoverType::FIVEGS_TO_EPS => Ok(HandoverTypeValue::FivegsToEps),
            HandoverType::EPS_TO_5GS => Ok(HandoverTypeValue::EpsTo5gs),
            _ => Err(HandoverError::InvalidIeValue(format!(
                "Unknown HandoverType value: {}",
                value.0
            ))),
        }
    }
}

/// Direct forwarding path availability
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectForwardingPathAvailabilityValue {
    /// Direct path available
    DirectPathAvailable,
}

// ============================================================================
// Handover Required (Source gNB -> AMF)
// ============================================================================

/// PDU Session Resource item for Handover Required
#[derive(Debug, Clone)]
pub struct PduSessionResourceHoRequiredItem {
    /// PDU Session ID (0-255)
    pub pdu_session_id: u8,
    /// Handover Required Transfer (opaque bytes)
    pub handover_required_transfer: Vec<u8>,
}

/// Parameters for building a Handover Required message
#[derive(Debug, Clone)]
pub struct HandoverRequiredParams {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
    /// Handover type
    pub handover_type: HandoverTypeValue,
    /// Cause for handover
    pub cause: HandoverCause,
    /// Target ID (opaque - contains TargetRANNodeID or TargeteNB-ID)
    pub target_id: TargetIdValue,
    /// Direct forwarding path availability (optional)
    pub direct_forwarding_path_availability: Option<DirectForwardingPathAvailabilityValue>,
    /// PDU Session Resource List
    pub pdu_session_resource_list: Vec<PduSessionResourceHoRequiredItem>,
    /// Source to Target Transparent Container (opaque bytes)
    pub source_to_target_transparent_container: Vec<u8>,
}

/// Target ID value for handover
#[derive(Debug, Clone)]
pub enum TargetIdValue {
    /// Target NG-RAN Node ID
    TargetRanNodeId {
        /// Global RAN Node ID (opaque bytes - encoded GlobalRANNodeID)
        global_ran_node_id: Vec<u8>,
        /// Selected TAI
        selected_tai: TaiValue,
    },
}

/// TAI (Tracking Area Identity) value
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaiValue {
    /// PLMN Identity (3 bytes)
    pub plmn_identity: [u8; 3],
    /// TAC (3 bytes)
    pub tac: [u8; 3],
}

/// Parsed Handover Required data
#[derive(Debug, Clone)]
pub struct HandoverRequiredData {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
    /// Handover type
    pub handover_type: HandoverTypeValue,
    /// Cause for handover
    pub cause: HandoverCause,
    /// PDU Session Resource List
    pub pdu_session_resource_list: Vec<PduSessionResourceHoRequiredItem>,
    /// Source to Target Transparent Container (opaque bytes)
    pub source_to_target_transparent_container: Vec<u8>,
}

/// Build a Handover Required PDU
pub fn build_handover_required(
    params: &HandoverRequiredParams,
) -> Result<NGAP_PDU, HandoverError> {
    let mut protocol_ies = Vec::new();

    // IE: AMF-UE-NGAP-ID (mandatory)
    protocol_ies.push(HandoverRequiredProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_AMF_UE_NGAP_ID),
        criticality: Criticality(Criticality::REJECT),
        value: HandoverRequiredProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(AMF_UE_NGAP_ID(
            params.amf_ue_ngap_id,
        )),
    });

    // IE: RAN-UE-NGAP-ID (mandatory)
    protocol_ies.push(HandoverRequiredProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_RAN_UE_NGAP_ID),
        criticality: Criticality(Criticality::REJECT),
        value: HandoverRequiredProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(RAN_UE_NGAP_ID(
            params.ran_ue_ngap_id,
        )),
    });

    // IE: HandoverType (mandatory)
    protocol_ies.push(HandoverRequiredProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_HANDOVER_TYPE),
        criticality: Criticality(Criticality::REJECT),
        value: HandoverRequiredProtocolIEs_EntryValue::Id_HandoverType(
            params.handover_type.into(),
        ),
    });

    // IE: Cause (mandatory)
    let cause = build_cause(&params.cause);
    protocol_ies.push(HandoverRequiredProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_CAUSE),
        criticality: Criticality(Criticality::IGNORE),
        value: HandoverRequiredProtocolIEs_EntryValue::Id_Cause(cause),
    });

    // IE: TargetID (mandatory)
    let target_id = build_target_id(&params.target_id);
    protocol_ies.push(HandoverRequiredProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_TARGET_ID),
        criticality: Criticality(Criticality::REJECT),
        value: HandoverRequiredProtocolIEs_EntryValue::Id_TargetID(target_id),
    });

    // IE: DirectForwardingPathAvailability (optional)
    if params.direct_forwarding_path_availability.is_some() {
        protocol_ies.push(HandoverRequiredProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_DIRECT_FORWARDING_PATH_AVAILABILITY),
            criticality: Criticality(Criticality::IGNORE),
            value: HandoverRequiredProtocolIEs_EntryValue::Id_DirectForwardingPathAvailability(
                DirectForwardingPathAvailability(
                    DirectForwardingPathAvailability::DIRECT_PATH_AVAILABLE,
                ),
            ),
        });
    }

    // IE: PDUSessionResourceListHORqd (mandatory)
    let pdu_session_list = build_pdu_session_resource_list_ho_rqd(&params.pdu_session_resource_list);
    protocol_ies.push(HandoverRequiredProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_PDU_SESSION_RESOURCE_LIST_HO_RQD),
        criticality: Criticality(Criticality::REJECT),
        value: HandoverRequiredProtocolIEs_EntryValue::Id_PDUSessionResourceListHORqd(
            pdu_session_list,
        ),
    });

    // IE: SourceToTarget-TransparentContainer (mandatory)
    protocol_ies.push(HandoverRequiredProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_SOURCE_TO_TARGET_TRANSPARENT_CONTAINER),
        criticality: Criticality(Criticality::REJECT),
        value: HandoverRequiredProtocolIEs_EntryValue::Id_SourceToTarget_TransparentContainer(
            SourceToTarget_TransparentContainer(
                params.source_to_target_transparent_container.clone(),
            ),
        ),
    });

    let handover_required = HandoverRequired {
        protocol_i_es: HandoverRequiredProtocolIEs(protocol_ies),
    };

    let initiating_message = InitiatingMessage {
        procedure_code: ProcedureCode(ID_HANDOVER_PREPARATION),
        criticality: Criticality(Criticality::REJECT),
        value: InitiatingMessageValue::Id_HandoverPreparation(handover_required),
    };

    Ok(NGAP_PDU::InitiatingMessage(initiating_message))
}

fn build_target_id(target_id: &TargetIdValue) -> TargetID {
    match target_id {
        TargetIdValue::TargetRanNodeId {
            global_ran_node_id,
            selected_tai,
        } => {
            // For simplicity, we encode the global_ran_node_id as a GlobalGNB-ID
            // In a real implementation, this would need proper parsing
            let gnb_id = GlobalGNB_ID {
                plmn_identity: PLMNIdentity(selected_tai.plmn_identity.to_vec()),
                gnb_id: GNB_ID::GNB_ID(GNB_ID_gNB_ID(bitvec::prelude::BitVec::from_slice(
                    global_ran_node_id,
                ))),
                ie_extensions: None,
            };

            let target_ran_node_id = TargetRANNodeID {
                global_ran_node_id: GlobalRANNodeID::GlobalGNB_ID(gnb_id),
                selected_tai: TAI {
                    plmn_identity: PLMNIdentity(selected_tai.plmn_identity.to_vec()),
                    tac: TAC(selected_tai.tac.to_vec()),
                    ie_extensions: None,
                },
                ie_extensions: None,
            };

            TargetID::TargetRANNodeID(target_ran_node_id)
        }
    }
}

fn build_pdu_session_resource_list_ho_rqd(
    items: &[PduSessionResourceHoRequiredItem],
) -> PDUSessionResourceListHORqd {
    let list: Vec<PDUSessionResourceItemHORqd> = items
        .iter()
        .map(|item| PDUSessionResourceItemHORqd {
            pdu_session_id: PDUSessionID(item.pdu_session_id),
            handover_required_transfer: PDUSessionResourceItemHORqdHandoverRequiredTransfer(
                item.handover_required_transfer.clone(),
            ),
            ie_extensions: None,
        })
        .collect();
    PDUSessionResourceListHORqd(list)
}

/// Parse a Handover Required from an NGAP PDU
pub fn parse_handover_required(
    pdu: &NGAP_PDU,
) -> Result<HandoverRequiredData, HandoverError> {
    let initiating_message = match pdu {
        NGAP_PDU::InitiatingMessage(msg) => msg,
        _ => {
            return Err(HandoverError::InvalidMessageType {
                expected: "InitiatingMessage".to_string(),
                actual: format!("{pdu:?}"),
            })
        }
    };

    let request = match &initiating_message.value {
        InitiatingMessageValue::Id_HandoverPreparation(req) => req,
        _ => {
            return Err(HandoverError::InvalidMessageType {
                expected: "HandoverRequired".to_string(),
                actual: format!("{:?}", initiating_message.value),
            })
        }
    };

    let mut amf_ue_ngap_id: Option<u64> = None;
    let mut ran_ue_ngap_id: Option<u32> = None;
    let mut handover_type: Option<HandoverTypeValue> = None;
    let mut cause: Option<HandoverCause> = None;
    let mut pdu_session_resource_list: Option<Vec<PduSessionResourceHoRequiredItem>> = None;
    let mut source_to_target_transparent_container: Option<Vec<u8>> = None;

    for ie in &request.protocol_i_es.0 {
        match &ie.value {
            HandoverRequiredProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(id) => {
                amf_ue_ngap_id = Some(id.0);
            }
            HandoverRequiredProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(id) => {
                ran_ue_ngap_id = Some(id.0);
            }
            HandoverRequiredProtocolIEs_EntryValue::Id_HandoverType(ht) => {
                handover_type = Some(HandoverTypeValue::try_from(ht)?);
            }
            HandoverRequiredProtocolIEs_EntryValue::Id_Cause(c) => {
                cause = Some(parse_cause(c));
            }
            HandoverRequiredProtocolIEs_EntryValue::Id_PDUSessionResourceListHORqd(list) => {
                pdu_session_resource_list = Some(parse_pdu_session_resource_list_ho_rqd(list));
            }
            HandoverRequiredProtocolIEs_EntryValue::Id_SourceToTarget_TransparentContainer(
                container,
            ) => {
                source_to_target_transparent_container = Some(container.0.clone());
            }
            _ => {}
        }
    }

    Ok(HandoverRequiredData {
        amf_ue_ngap_id: amf_ue_ngap_id.ok_or_else(|| {
            HandoverError::MissingMandatoryIe("AMF-UE-NGAP-ID".to_string())
        })?,
        ran_ue_ngap_id: ran_ue_ngap_id.ok_or_else(|| {
            HandoverError::MissingMandatoryIe("RAN-UE-NGAP-ID".to_string())
        })?,
        handover_type: handover_type.ok_or_else(|| {
            HandoverError::MissingMandatoryIe("HandoverType".to_string())
        })?,
        cause: cause.ok_or_else(|| HandoverError::MissingMandatoryIe("Cause".to_string()))?,
        pdu_session_resource_list: pdu_session_resource_list.ok_or_else(|| {
            HandoverError::MissingMandatoryIe("PDUSessionResourceListHORqd".to_string())
        })?,
        source_to_target_transparent_container: source_to_target_transparent_container
            .ok_or_else(|| {
                HandoverError::MissingMandatoryIe(
                    "SourceToTarget-TransparentContainer".to_string(),
                )
            })?,
    })
}

fn parse_pdu_session_resource_list_ho_rqd(
    list: &PDUSessionResourceListHORqd,
) -> Vec<PduSessionResourceHoRequiredItem> {
    list.0
        .iter()
        .map(|item| PduSessionResourceHoRequiredItem {
            pdu_session_id: item.pdu_session_id.0,
            handover_required_transfer: item.handover_required_transfer.0.clone(),
        })
        .collect()
}

// ============================================================================
// Handover Command (AMF -> Source gNB)
// ============================================================================

/// PDU Session Resource item for Handover Command
#[derive(Debug, Clone)]
pub struct PduSessionResourceHandoverItem {
    /// PDU Session ID (0-255)
    pub pdu_session_id: u8,
    /// Handover Command Transfer (opaque bytes)
    pub handover_command_transfer: Vec<u8>,
}

/// PDU Session Resource item to release in Handover Command
#[derive(Debug, Clone)]
pub struct PduSessionResourceToReleaseHoCmdItem {
    /// PDU Session ID (0-255)
    pub pdu_session_id: u8,
    /// Handover Preparation Unsuccessful Transfer (opaque bytes)
    pub handover_preparation_unsuccessful_transfer: Vec<u8>,
}

/// Parsed Handover Command data
#[derive(Debug, Clone)]
pub struct HandoverCommandData {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
    /// Handover type
    pub handover_type: HandoverTypeValue,
    /// PDU Session Resource Handover List (optional)
    pub pdu_session_resource_handover_list: Option<Vec<PduSessionResourceHandoverItem>>,
    /// PDU Session Resource To Release List (optional)
    pub pdu_session_resource_to_release_list: Option<Vec<PduSessionResourceToReleaseHoCmdItem>>,
    /// Target to Source Transparent Container (opaque bytes)
    pub target_to_source_transparent_container: Vec<u8>,
}

/// Parse a Handover Command from an NGAP PDU
pub fn parse_handover_command(pdu: &NGAP_PDU) -> Result<HandoverCommandData, HandoverError> {
    let successful_outcome = match pdu {
        NGAP_PDU::SuccessfulOutcome(outcome) => outcome,
        _ => {
            return Err(HandoverError::InvalidMessageType {
                expected: "SuccessfulOutcome".to_string(),
                actual: format!("{pdu:?}"),
            })
        }
    };

    let command = match &successful_outcome.value {
        SuccessfulOutcomeValue::Id_HandoverPreparation(cmd) => cmd,
        _ => {
            return Err(HandoverError::InvalidMessageType {
                expected: "HandoverCommand".to_string(),
                actual: format!("{:?}", successful_outcome.value),
            })
        }
    };

    let mut amf_ue_ngap_id: Option<u64> = None;
    let mut ran_ue_ngap_id: Option<u32> = None;
    let mut handover_type: Option<HandoverTypeValue> = None;
    let mut pdu_session_resource_handover_list: Option<Vec<PduSessionResourceHandoverItem>> = None;
    let mut pdu_session_resource_to_release_list: Option<Vec<PduSessionResourceToReleaseHoCmdItem>> =
        None;
    let mut target_to_source_transparent_container: Option<Vec<u8>> = None;

    for ie in &command.protocol_i_es.0 {
        match &ie.value {
            HandoverCommandProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(id) => {
                amf_ue_ngap_id = Some(id.0);
            }
            HandoverCommandProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(id) => {
                ran_ue_ngap_id = Some(id.0);
            }
            HandoverCommandProtocolIEs_EntryValue::Id_HandoverType(ht) => {
                handover_type = Some(HandoverTypeValue::try_from(ht)?);
            }
            HandoverCommandProtocolIEs_EntryValue::Id_PDUSessionResourceHandoverList(list) => {
                pdu_session_resource_handover_list =
                    Some(parse_pdu_session_resource_handover_list(list));
            }
            HandoverCommandProtocolIEs_EntryValue::Id_PDUSessionResourceToReleaseListHOCmd(list) => {
                pdu_session_resource_to_release_list =
                    Some(parse_pdu_session_resource_to_release_list_ho_cmd(list));
            }
            HandoverCommandProtocolIEs_EntryValue::Id_TargetToSource_TransparentContainer(
                container,
            ) => {
                target_to_source_transparent_container = Some(container.0.clone());
            }
            _ => {}
        }
    }

    Ok(HandoverCommandData {
        amf_ue_ngap_id: amf_ue_ngap_id.ok_or_else(|| {
            HandoverError::MissingMandatoryIe("AMF-UE-NGAP-ID".to_string())
        })?,
        ran_ue_ngap_id: ran_ue_ngap_id.ok_or_else(|| {
            HandoverError::MissingMandatoryIe("RAN-UE-NGAP-ID".to_string())
        })?,
        handover_type: handover_type.ok_or_else(|| {
            HandoverError::MissingMandatoryIe("HandoverType".to_string())
        })?,
        pdu_session_resource_handover_list,
        pdu_session_resource_to_release_list,
        target_to_source_transparent_container: target_to_source_transparent_container
            .ok_or_else(|| {
                HandoverError::MissingMandatoryIe(
                    "TargetToSource-TransparentContainer".to_string(),
                )
            })?,
    })
}

fn parse_pdu_session_resource_handover_list(
    list: &PDUSessionResourceHandoverList,
) -> Vec<PduSessionResourceHandoverItem> {
    list.0
        .iter()
        .map(|item| PduSessionResourceHandoverItem {
            pdu_session_id: item.pdu_session_id.0,
            handover_command_transfer: item.handover_command_transfer.0.clone(),
        })
        .collect()
}

fn parse_pdu_session_resource_to_release_list_ho_cmd(
    list: &PDUSessionResourceToReleaseListHOCmd,
) -> Vec<PduSessionResourceToReleaseHoCmdItem> {
    list.0
        .iter()
        .map(|item| PduSessionResourceToReleaseHoCmdItem {
            pdu_session_id: item.pdu_session_id.0,
            handover_preparation_unsuccessful_transfer: item
                .handover_preparation_unsuccessful_transfer
                .0
                .clone(),
        })
        .collect()
}

// ============================================================================
// Handover Preparation Failure (AMF -> Source gNB)
// ============================================================================

/// Parsed Handover Preparation Failure data
#[derive(Debug, Clone)]
pub struct HandoverPreparationFailureData {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
    /// Cause of failure
    pub cause: HandoverCause,
}

/// Parse a Handover Preparation Failure from an NGAP PDU
pub fn parse_handover_preparation_failure(
    pdu: &NGAP_PDU,
) -> Result<HandoverPreparationFailureData, HandoverError> {
    let unsuccessful_outcome = match pdu {
        NGAP_PDU::UnsuccessfulOutcome(outcome) => outcome,
        _ => {
            return Err(HandoverError::InvalidMessageType {
                expected: "UnsuccessfulOutcome".to_string(),
                actual: format!("{pdu:?}"),
            })
        }
    };

    let failure = match &unsuccessful_outcome.value {
        UnsuccessfulOutcomeValue::Id_HandoverPreparation(fail) => fail,
        _ => {
            return Err(HandoverError::InvalidMessageType {
                expected: "HandoverPreparationFailure".to_string(),
                actual: format!("{:?}", unsuccessful_outcome.value),
            })
        }
    };

    let mut amf_ue_ngap_id: Option<u64> = None;
    let mut ran_ue_ngap_id: Option<u32> = None;
    let mut cause: Option<HandoverCause> = None;

    for ie in &failure.protocol_i_es.0 {
        match &ie.value {
            HandoverPreparationFailureProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(id) => {
                amf_ue_ngap_id = Some(id.0);
            }
            HandoverPreparationFailureProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(id) => {
                ran_ue_ngap_id = Some(id.0);
            }
            HandoverPreparationFailureProtocolIEs_EntryValue::Id_Cause(c) => {
                cause = Some(parse_cause(c));
            }
            _ => {}
        }
    }

    Ok(HandoverPreparationFailureData {
        amf_ue_ngap_id: amf_ue_ngap_id.ok_or_else(|| {
            HandoverError::MissingMandatoryIe("AMF-UE-NGAP-ID".to_string())
        })?,
        ran_ue_ngap_id: ran_ue_ngap_id.ok_or_else(|| {
            HandoverError::MissingMandatoryIe("RAN-UE-NGAP-ID".to_string())
        })?,
        cause: cause.ok_or_else(|| HandoverError::MissingMandatoryIe("Cause".to_string()))?,
    })
}


// ============================================================================
// Handover Request (AMF -> Target gNB)
// ============================================================================

/// Parsed Handover Request data
#[derive(Debug, Clone)]
pub struct HandoverRequestData {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// Handover type
    pub handover_type: HandoverTypeValue,
    /// Cause for handover
    pub cause: HandoverCause,
    /// Source to Target Transparent Container (opaque bytes)
    pub source_to_target_transparent_container: Vec<u8>,
}

/// Parse a Handover Request from an NGAP PDU
pub fn parse_handover_request(pdu: &NGAP_PDU) -> Result<HandoverRequestData, HandoverError> {
    let initiating_message = match pdu {
        NGAP_PDU::InitiatingMessage(msg) => msg,
        _ => {
            return Err(HandoverError::InvalidMessageType {
                expected: "InitiatingMessage".to_string(),
                actual: format!("{pdu:?}"),
            })
        }
    };

    let request = match &initiating_message.value {
        InitiatingMessageValue::Id_HandoverResourceAllocation(req) => req,
        _ => {
            return Err(HandoverError::InvalidMessageType {
                expected: "HandoverRequest".to_string(),
                actual: format!("{:?}", initiating_message.value),
            })
        }
    };

    let mut amf_ue_ngap_id: Option<u64> = None;
    let mut handover_type: Option<HandoverTypeValue> = None;
    let mut cause: Option<HandoverCause> = None;
    let mut source_to_target_transparent_container: Option<Vec<u8>> = None;

    for ie in &request.protocol_i_es.0 {
        match &ie.value {
            HandoverRequestProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(id) => {
                amf_ue_ngap_id = Some(id.0);
            }
            HandoverRequestProtocolIEs_EntryValue::Id_HandoverType(ht) => {
                handover_type = Some(HandoverTypeValue::try_from(ht)?);
            }
            HandoverRequestProtocolIEs_EntryValue::Id_Cause(c) => {
                cause = Some(parse_cause(c));
            }
            HandoverRequestProtocolIEs_EntryValue::Id_SourceToTarget_TransparentContainer(
                container,
            ) => {
                source_to_target_transparent_container = Some(container.0.clone());
            }
            _ => {}
        }
    }

    Ok(HandoverRequestData {
        amf_ue_ngap_id: amf_ue_ngap_id.ok_or_else(|| {
            HandoverError::MissingMandatoryIe("AMF-UE-NGAP-ID".to_string())
        })?,
        handover_type: handover_type.ok_or_else(|| {
            HandoverError::MissingMandatoryIe("HandoverType".to_string())
        })?,
        cause: cause.ok_or_else(|| HandoverError::MissingMandatoryIe("Cause".to_string()))?,
        source_to_target_transparent_container: source_to_target_transparent_container
            .ok_or_else(|| {
                HandoverError::MissingMandatoryIe(
                    "SourceToTarget-TransparentContainer".to_string(),
                )
            })?,
    })
}

// ============================================================================
// Handover Request Acknowledge (Target gNB -> AMF)
// ============================================================================

/// PDU Session Resource item for Handover Request Acknowledge (admitted)
#[derive(Debug, Clone)]
pub struct PduSessionResourceAdmittedItem {
    /// PDU Session ID (0-255)
    pub pdu_session_id: u8,
    /// Handover Request Acknowledge Transfer (opaque bytes)
    pub handover_request_ack_transfer: Vec<u8>,
}

/// PDU Session Resource item for Handover Request Acknowledge (failed)
#[derive(Debug, Clone)]
pub struct PduSessionResourceFailedToSetupHoAckItem {
    /// PDU Session ID (0-255)
    pub pdu_session_id: u8,
    /// Handover Resource Allocation Unsuccessful Transfer (opaque bytes)
    pub handover_resource_allocation_unsuccessful_transfer: Vec<u8>,
}

/// Parameters for building a Handover Request Acknowledge message
#[derive(Debug, Clone)]
pub struct HandoverRequestAcknowledgeParams {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
    /// PDU Session Resource Admitted List
    pub pdu_session_resource_admitted_list: Vec<PduSessionResourceAdmittedItem>,
    /// PDU Session Resource Failed To Setup List (optional)
    pub pdu_session_resource_failed_list: Option<Vec<PduSessionResourceFailedToSetupHoAckItem>>,
    /// Target to Source Transparent Container (opaque bytes)
    pub target_to_source_transparent_container: Vec<u8>,
}

/// Parsed Handover Request Acknowledge data
#[derive(Debug, Clone)]
pub struct HandoverRequestAcknowledgeData {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
    /// PDU Session Resource Admitted List
    pub pdu_session_resource_admitted_list: Vec<PduSessionResourceAdmittedItem>,
    /// PDU Session Resource Failed To Setup List (optional)
    pub pdu_session_resource_failed_list: Option<Vec<PduSessionResourceFailedToSetupHoAckItem>>,
    /// Target to Source Transparent Container (opaque bytes)
    pub target_to_source_transparent_container: Vec<u8>,
}

/// Build a Handover Request Acknowledge PDU
pub fn build_handover_request_acknowledge(
    params: &HandoverRequestAcknowledgeParams,
) -> Result<NGAP_PDU, HandoverError> {
    let mut protocol_ies = Vec::new();

    // IE: AMF-UE-NGAP-ID (mandatory)
    protocol_ies.push(HandoverRequestAcknowledgeProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_AMF_UE_NGAP_ID),
        criticality: Criticality(Criticality::IGNORE),
        value: HandoverRequestAcknowledgeProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(AMF_UE_NGAP_ID(
            params.amf_ue_ngap_id,
        )),
    });

    // IE: RAN-UE-NGAP-ID (mandatory)
    protocol_ies.push(HandoverRequestAcknowledgeProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_RAN_UE_NGAP_ID),
        criticality: Criticality(Criticality::IGNORE),
        value: HandoverRequestAcknowledgeProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(RAN_UE_NGAP_ID(
            params.ran_ue_ngap_id,
        )),
    });

    // IE: PDUSessionResourceAdmittedList (mandatory)
    let admitted_list = build_pdu_session_resource_admitted_list(&params.pdu_session_resource_admitted_list);
    protocol_ies.push(HandoverRequestAcknowledgeProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_PDU_SESSION_RESOURCE_ADMITTED_LIST),
        criticality: Criticality(Criticality::IGNORE),
        value: HandoverRequestAcknowledgeProtocolIEs_EntryValue::Id_PDUSessionResourceAdmittedList(
            admitted_list,
        ),
    });

    // IE: PDUSessionResourceFailedToSetupListHOAck (optional)
    if let Some(ref failed_list) = params.pdu_session_resource_failed_list {
        let list = build_pdu_session_resource_failed_to_setup_list_ho_ack(failed_list);
        protocol_ies.push(HandoverRequestAcknowledgeProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_PDU_SESSION_RESOURCE_FAILED_TO_SETUP_LIST_HO_ACK),
            criticality: Criticality(Criticality::IGNORE),
            value:
                HandoverRequestAcknowledgeProtocolIEs_EntryValue::Id_PDUSessionResourceFailedToSetupListHOAck(
                    list,
                ),
        });
    }

    // IE: TargetToSource-TransparentContainer (mandatory)
    protocol_ies.push(HandoverRequestAcknowledgeProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_TARGET_TO_SOURCE_TRANSPARENT_CONTAINER),
        criticality: Criticality(Criticality::REJECT),
        value: HandoverRequestAcknowledgeProtocolIEs_EntryValue::Id_TargetToSource_TransparentContainer(
            TargetToSource_TransparentContainer(
                params.target_to_source_transparent_container.clone(),
            ),
        ),
    });

    let handover_request_ack = HandoverRequestAcknowledge {
        protocol_i_es: HandoverRequestAcknowledgeProtocolIEs(protocol_ies),
    };

    let successful_outcome = SuccessfulOutcome {
        procedure_code: ProcedureCode(ID_HANDOVER_RESOURCE_ALLOCATION),
        criticality: Criticality(Criticality::REJECT),
        value: SuccessfulOutcomeValue::Id_HandoverResourceAllocation(handover_request_ack),
    };

    Ok(NGAP_PDU::SuccessfulOutcome(successful_outcome))
}

fn build_pdu_session_resource_admitted_list(
    items: &[PduSessionResourceAdmittedItem],
) -> PDUSessionResourceAdmittedList {
    let list: Vec<PDUSessionResourceAdmittedItem> = items
        .iter()
        .map(|item| PDUSessionResourceAdmittedItem {
            pdu_session_id: PDUSessionID(item.pdu_session_id),
            handover_request_acknowledge_transfer:
                PDUSessionResourceAdmittedItemHandoverRequestAcknowledgeTransfer(
                    item.handover_request_ack_transfer.clone(),
                ),
            ie_extensions: None,
        })
        .collect();
    PDUSessionResourceAdmittedList(list)
}

fn build_pdu_session_resource_failed_to_setup_list_ho_ack(
    items: &[PduSessionResourceFailedToSetupHoAckItem],
) -> PDUSessionResourceFailedToSetupListHOAck {
    let list: Vec<PDUSessionResourceFailedToSetupItemHOAck> = items
        .iter()
        .map(|item| PDUSessionResourceFailedToSetupItemHOAck {
            pdu_session_id: PDUSessionID(item.pdu_session_id),
            handover_resource_allocation_unsuccessful_transfer:
                PDUSessionResourceFailedToSetupItemHOAckHandoverResourceAllocationUnsuccessfulTransfer(
                    item.handover_resource_allocation_unsuccessful_transfer.clone(),
                ),
            ie_extensions: None,
        })
        .collect();
    PDUSessionResourceFailedToSetupListHOAck(list)
}

/// Parse a Handover Request Acknowledge from an NGAP PDU
pub fn parse_handover_request_acknowledge(
    pdu: &NGAP_PDU,
) -> Result<HandoverRequestAcknowledgeData, HandoverError> {
    let successful_outcome = match pdu {
        NGAP_PDU::SuccessfulOutcome(outcome) => outcome,
        _ => {
            return Err(HandoverError::InvalidMessageType {
                expected: "SuccessfulOutcome".to_string(),
                actual: format!("{pdu:?}"),
            })
        }
    };

    let ack = match &successful_outcome.value {
        SuccessfulOutcomeValue::Id_HandoverResourceAllocation(a) => a,
        _ => {
            return Err(HandoverError::InvalidMessageType {
                expected: "HandoverRequestAcknowledge".to_string(),
                actual: format!("{:?}", successful_outcome.value),
            })
        }
    };

    let mut amf_ue_ngap_id: Option<u64> = None;
    let mut ran_ue_ngap_id: Option<u32> = None;
    let mut pdu_session_resource_admitted_list: Option<Vec<PduSessionResourceAdmittedItem>> = None;
    let mut pdu_session_resource_failed_list: Option<Vec<PduSessionResourceFailedToSetupHoAckItem>> =
        None;
    let mut target_to_source_transparent_container: Option<Vec<u8>> = None;

    for ie in &ack.protocol_i_es.0 {
        match &ie.value {
            HandoverRequestAcknowledgeProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(id) => {
                amf_ue_ngap_id = Some(id.0);
            }
            HandoverRequestAcknowledgeProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(id) => {
                ran_ue_ngap_id = Some(id.0);
            }
            HandoverRequestAcknowledgeProtocolIEs_EntryValue::Id_PDUSessionResourceAdmittedList(
                list,
            ) => {
                pdu_session_resource_admitted_list =
                    Some(parse_pdu_session_resource_admitted_list(list));
            }
            HandoverRequestAcknowledgeProtocolIEs_EntryValue::Id_PDUSessionResourceFailedToSetupListHOAck(
                list,
            ) => {
                pdu_session_resource_failed_list =
                    Some(parse_pdu_session_resource_failed_to_setup_list_ho_ack(list));
            }
            HandoverRequestAcknowledgeProtocolIEs_EntryValue::Id_TargetToSource_TransparentContainer(
                container,
            ) => {
                target_to_source_transparent_container = Some(container.0.clone());
            }
            _ => {}
        }
    }

    Ok(HandoverRequestAcknowledgeData {
        amf_ue_ngap_id: amf_ue_ngap_id.ok_or_else(|| {
            HandoverError::MissingMandatoryIe("AMF-UE-NGAP-ID".to_string())
        })?,
        ran_ue_ngap_id: ran_ue_ngap_id.ok_or_else(|| {
            HandoverError::MissingMandatoryIe("RAN-UE-NGAP-ID".to_string())
        })?,
        pdu_session_resource_admitted_list: pdu_session_resource_admitted_list.ok_or_else(|| {
            HandoverError::MissingMandatoryIe("PDUSessionResourceAdmittedList".to_string())
        })?,
        pdu_session_resource_failed_list,
        target_to_source_transparent_container: target_to_source_transparent_container
            .ok_or_else(|| {
                HandoverError::MissingMandatoryIe(
                    "TargetToSource-TransparentContainer".to_string(),
                )
            })?,
    })
}

fn parse_pdu_session_resource_admitted_list(
    list: &PDUSessionResourceAdmittedList,
) -> Vec<PduSessionResourceAdmittedItem> {
    list.0
        .iter()
        .map(|item| PduSessionResourceAdmittedItem {
            pdu_session_id: item.pdu_session_id.0,
            handover_request_ack_transfer: item.handover_request_acknowledge_transfer.0.clone(),
        })
        .collect()
}

fn parse_pdu_session_resource_failed_to_setup_list_ho_ack(
    list: &PDUSessionResourceFailedToSetupListHOAck,
) -> Vec<PduSessionResourceFailedToSetupHoAckItem> {
    list.0
        .iter()
        .map(|item| PduSessionResourceFailedToSetupHoAckItem {
            pdu_session_id: item.pdu_session_id.0,
            handover_resource_allocation_unsuccessful_transfer: item
                .handover_resource_allocation_unsuccessful_transfer
                .0
                .clone(),
        })
        .collect()
}


// ============================================================================
// Handover Failure (Target gNB -> AMF)
// ============================================================================

/// Parameters for building a Handover Failure message
#[derive(Debug, Clone)]
pub struct HandoverFailureParams {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// Cause of failure
    pub cause: HandoverCause,
}

/// Parsed Handover Failure data
#[derive(Debug, Clone)]
pub struct HandoverFailureData {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// Cause of failure
    pub cause: HandoverCause,
}

/// Build a Handover Failure PDU
pub fn build_handover_failure(
    params: &HandoverFailureParams,
) -> Result<NGAP_PDU, HandoverError> {
    let mut protocol_ies = Vec::new();

    // IE: AMF-UE-NGAP-ID (mandatory)
    protocol_ies.push(HandoverFailureProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_AMF_UE_NGAP_ID),
        criticality: Criticality(Criticality::IGNORE),
        value: HandoverFailureProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(AMF_UE_NGAP_ID(
            params.amf_ue_ngap_id,
        )),
    });

    // IE: Cause (mandatory)
    let cause = build_cause(&params.cause);
    protocol_ies.push(HandoverFailureProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_CAUSE),
        criticality: Criticality(Criticality::IGNORE),
        value: HandoverFailureProtocolIEs_EntryValue::Id_Cause(cause),
    });

    let handover_failure = HandoverFailure {
        protocol_i_es: HandoverFailureProtocolIEs(protocol_ies),
    };

    let unsuccessful_outcome = UnsuccessfulOutcome {
        procedure_code: ProcedureCode(ID_HANDOVER_RESOURCE_ALLOCATION),
        criticality: Criticality(Criticality::REJECT),
        value: UnsuccessfulOutcomeValue::Id_HandoverResourceAllocation(handover_failure),
    };

    Ok(NGAP_PDU::UnsuccessfulOutcome(unsuccessful_outcome))
}

/// Parse a Handover Failure from an NGAP PDU
pub fn parse_handover_failure(pdu: &NGAP_PDU) -> Result<HandoverFailureData, HandoverError> {
    let unsuccessful_outcome = match pdu {
        NGAP_PDU::UnsuccessfulOutcome(outcome) => outcome,
        _ => {
            return Err(HandoverError::InvalidMessageType {
                expected: "UnsuccessfulOutcome".to_string(),
                actual: format!("{pdu:?}"),
            })
        }
    };

    let failure = match &unsuccessful_outcome.value {
        UnsuccessfulOutcomeValue::Id_HandoverResourceAllocation(fail) => fail,
        _ => {
            return Err(HandoverError::InvalidMessageType {
                expected: "HandoverFailure".to_string(),
                actual: format!("{:?}", unsuccessful_outcome.value),
            })
        }
    };

    let mut amf_ue_ngap_id: Option<u64> = None;
    let mut cause: Option<HandoverCause> = None;

    for ie in &failure.protocol_i_es.0 {
        match &ie.value {
            HandoverFailureProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(id) => {
                amf_ue_ngap_id = Some(id.0);
            }
            HandoverFailureProtocolIEs_EntryValue::Id_Cause(c) => {
                cause = Some(parse_cause(c));
            }
            _ => {}
        }
    }

    Ok(HandoverFailureData {
        amf_ue_ngap_id: amf_ue_ngap_id.ok_or_else(|| {
            HandoverError::MissingMandatoryIe("AMF-UE-NGAP-ID".to_string())
        })?,
        cause: cause.ok_or_else(|| HandoverError::MissingMandatoryIe("Cause".to_string()))?,
    })
}

// ============================================================================
// Handover Notify (Target gNB -> AMF)
// ============================================================================

/// User Location Information for Handover Notify
#[derive(Debug, Clone)]
pub struct UserLocationInfoNr {
    /// NR CGI
    pub nr_cgi: NrCgiValue,
    /// TAI
    pub tai: TaiValue,
}

/// NR CGI (NR Cell Global Identity)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NrCgiValue {
    /// PLMN Identity (3 bytes)
    pub plmn_identity: [u8; 3],
    /// NR Cell Identity (36 bits)
    pub nr_cell_identity: u64,
}

/// Parameters for building a Handover Notify message
#[derive(Debug, Clone)]
pub struct HandoverNotifyParams {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
    /// User Location Information
    pub user_location_info: UserLocationInfoNr,
}

/// Parsed Handover Notify data
#[derive(Debug, Clone)]
pub struct HandoverNotifyData {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
}

/// Build a Handover Notify PDU
pub fn build_handover_notify(
    params: &HandoverNotifyParams,
) -> Result<NGAP_PDU, HandoverError> {
    let mut protocol_ies = Vec::new();

    // IE: AMF-UE-NGAP-ID (mandatory)
    protocol_ies.push(HandoverNotifyProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_AMF_UE_NGAP_ID),
        criticality: Criticality(Criticality::REJECT),
        value: HandoverNotifyProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(AMF_UE_NGAP_ID(
            params.amf_ue_ngap_id,
        )),
    });

    // IE: RAN-UE-NGAP-ID (mandatory)
    protocol_ies.push(HandoverNotifyProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_RAN_UE_NGAP_ID),
        criticality: Criticality(Criticality::REJECT),
        value: HandoverNotifyProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(RAN_UE_NGAP_ID(
            params.ran_ue_ngap_id,
        )),
    });

    // IE: UserLocationInformation (mandatory)
    let user_location_info = build_user_location_info_nr(&params.user_location_info);
    protocol_ies.push(HandoverNotifyProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_USER_LOCATION_INFORMATION),
        criticality: Criticality(Criticality::IGNORE),
        value: HandoverNotifyProtocolIEs_EntryValue::Id_UserLocationInformation(user_location_info),
    });

    let handover_notify = HandoverNotify {
        protocol_i_es: HandoverNotifyProtocolIEs(protocol_ies),
    };

    let initiating_message = InitiatingMessage {
        procedure_code: ProcedureCode(ID_HANDOVER_NOTIFICATION),
        criticality: Criticality(Criticality::IGNORE),
        value: InitiatingMessageValue::Id_HandoverNotification(handover_notify),
    };

    Ok(NGAP_PDU::InitiatingMessage(initiating_message))
}

fn build_user_location_info_nr(info: &UserLocationInfoNr) -> UserLocationInformation {
    use bitvec::prelude::*;

    // NR Cell Identity is 36 bits
    let mut nr_cell_identity_bv: BitVec<u8, Msb0> = BitVec::with_capacity(36);
    for i in (0..36).rev() {
        nr_cell_identity_bv.push((info.nr_cgi.nr_cell_identity >> i) & 1 == 1);
    }

    let nr_cgi = NR_CGI {
        plmn_identity: PLMNIdentity(info.nr_cgi.plmn_identity.to_vec()),
        nr_cell_identity: NRCellIdentity(nr_cell_identity_bv),
        ie_extensions: None,
    };

    let tai = TAI {
        plmn_identity: PLMNIdentity(info.tai.plmn_identity.to_vec()),
        tac: TAC(info.tai.tac.to_vec()),
        ie_extensions: None,
    };

    let user_location_info_nr = UserLocationInformationNR {
        nr_cgi,
        tai,
        time_stamp: None,
        ie_extensions: None,
    };

    UserLocationInformation::UserLocationInformationNR(user_location_info_nr)
}

/// Parse a Handover Notify from an NGAP PDU
pub fn parse_handover_notify(pdu: &NGAP_PDU) -> Result<HandoverNotifyData, HandoverError> {
    let initiating_message = match pdu {
        NGAP_PDU::InitiatingMessage(msg) => msg,
        _ => {
            return Err(HandoverError::InvalidMessageType {
                expected: "InitiatingMessage".to_string(),
                actual: format!("{pdu:?}"),
            })
        }
    };

    let notify = match &initiating_message.value {
        InitiatingMessageValue::Id_HandoverNotification(n) => n,
        _ => {
            return Err(HandoverError::InvalidMessageType {
                expected: "HandoverNotify".to_string(),
                actual: format!("{:?}", initiating_message.value),
            })
        }
    };

    let mut amf_ue_ngap_id: Option<u64> = None;
    let mut ran_ue_ngap_id: Option<u32> = None;

    for ie in &notify.protocol_i_es.0 {
        match &ie.value {
            HandoverNotifyProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(id) => {
                amf_ue_ngap_id = Some(id.0);
            }
            HandoverNotifyProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(id) => {
                ran_ue_ngap_id = Some(id.0);
            }
            _ => {}
        }
    }

    Ok(HandoverNotifyData {
        amf_ue_ngap_id: amf_ue_ngap_id.ok_or_else(|| {
            HandoverError::MissingMandatoryIe("AMF-UE-NGAP-ID".to_string())
        })?,
        ran_ue_ngap_id: ran_ue_ngap_id.ok_or_else(|| {
            HandoverError::MissingMandatoryIe("RAN-UE-NGAP-ID".to_string())
        })?,
    })
}

// ============================================================================
// Handover Cancel (Source gNB -> AMF)
// ============================================================================

/// Parameters for building a Handover Cancel message
#[derive(Debug, Clone)]
pub struct HandoverCancelParams {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
    /// Cause for cancellation
    pub cause: HandoverCause,
}

/// Parsed Handover Cancel data
#[derive(Debug, Clone)]
pub struct HandoverCancelData {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
    /// Cause for cancellation
    pub cause: HandoverCause,
}

/// Build a Handover Cancel PDU
pub fn build_handover_cancel(
    params: &HandoverCancelParams,
) -> Result<NGAP_PDU, HandoverError> {
    let mut protocol_ies = Vec::new();

    // IE: AMF-UE-NGAP-ID (mandatory)
    protocol_ies.push(HandoverCancelProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_AMF_UE_NGAP_ID),
        criticality: Criticality(Criticality::REJECT),
        value: HandoverCancelProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(AMF_UE_NGAP_ID(
            params.amf_ue_ngap_id,
        )),
    });

    // IE: RAN-UE-NGAP-ID (mandatory)
    protocol_ies.push(HandoverCancelProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_RAN_UE_NGAP_ID),
        criticality: Criticality(Criticality::REJECT),
        value: HandoverCancelProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(RAN_UE_NGAP_ID(
            params.ran_ue_ngap_id,
        )),
    });

    // IE: Cause (mandatory)
    let cause = build_cause(&params.cause);
    protocol_ies.push(HandoverCancelProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_CAUSE),
        criticality: Criticality(Criticality::IGNORE),
        value: HandoverCancelProtocolIEs_EntryValue::Id_Cause(cause),
    });

    let handover_cancel = HandoverCancel {
        protocol_i_es: HandoverCancelProtocolIEs(protocol_ies),
    };

    let initiating_message = InitiatingMessage {
        procedure_code: ProcedureCode(ID_HANDOVER_CANCEL),
        criticality: Criticality(Criticality::REJECT),
        value: InitiatingMessageValue::Id_HandoverCancel(handover_cancel),
    };

    Ok(NGAP_PDU::InitiatingMessage(initiating_message))
}

/// Parse a Handover Cancel from an NGAP PDU
pub fn parse_handover_cancel(pdu: &NGAP_PDU) -> Result<HandoverCancelData, HandoverError> {
    let initiating_message = match pdu {
        NGAP_PDU::InitiatingMessage(msg) => msg,
        _ => {
            return Err(HandoverError::InvalidMessageType {
                expected: "InitiatingMessage".to_string(),
                actual: format!("{pdu:?}"),
            })
        }
    };

    let cancel = match &initiating_message.value {
        InitiatingMessageValue::Id_HandoverCancel(c) => c,
        _ => {
            return Err(HandoverError::InvalidMessageType {
                expected: "HandoverCancel".to_string(),
                actual: format!("{:?}", initiating_message.value),
            })
        }
    };

    let mut amf_ue_ngap_id: Option<u64> = None;
    let mut ran_ue_ngap_id: Option<u32> = None;
    let mut cause: Option<HandoverCause> = None;

    for ie in &cancel.protocol_i_es.0 {
        match &ie.value {
            HandoverCancelProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(id) => {
                amf_ue_ngap_id = Some(id.0);
            }
            HandoverCancelProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(id) => {
                ran_ue_ngap_id = Some(id.0);
            }
            HandoverCancelProtocolIEs_EntryValue::Id_Cause(c) => {
                cause = Some(parse_cause(c));
            }
        }
    }

    Ok(HandoverCancelData {
        amf_ue_ngap_id: amf_ue_ngap_id.ok_or_else(|| {
            HandoverError::MissingMandatoryIe("AMF-UE-NGAP-ID".to_string())
        })?,
        ran_ue_ngap_id: ran_ue_ngap_id.ok_or_else(|| {
            HandoverError::MissingMandatoryIe("RAN-UE-NGAP-ID".to_string())
        })?,
        cause: cause.ok_or_else(|| HandoverError::MissingMandatoryIe("Cause".to_string()))?,
    })
}

// ============================================================================
// Handover Cancel Acknowledge (AMF -> Source gNB)
// ============================================================================

/// Parsed Handover Cancel Acknowledge data
#[derive(Debug, Clone)]
pub struct HandoverCancelAcknowledgeData {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
}

/// Parse a Handover Cancel Acknowledge from an NGAP PDU
pub fn parse_handover_cancel_acknowledge(
    pdu: &NGAP_PDU,
) -> Result<HandoverCancelAcknowledgeData, HandoverError> {
    let successful_outcome = match pdu {
        NGAP_PDU::SuccessfulOutcome(outcome) => outcome,
        _ => {
            return Err(HandoverError::InvalidMessageType {
                expected: "SuccessfulOutcome".to_string(),
                actual: format!("{pdu:?}"),
            })
        }
    };

    let ack = match &successful_outcome.value {
        SuccessfulOutcomeValue::Id_HandoverCancel(a) => a,
        _ => {
            return Err(HandoverError::InvalidMessageType {
                expected: "HandoverCancelAcknowledge".to_string(),
                actual: format!("{:?}", successful_outcome.value),
            })
        }
    };

    let mut amf_ue_ngap_id: Option<u64> = None;
    let mut ran_ue_ngap_id: Option<u32> = None;

    for ie in &ack.protocol_i_es.0 {
        match &ie.value {
            HandoverCancelAcknowledgeProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(id) => {
                amf_ue_ngap_id = Some(id.0);
            }
            HandoverCancelAcknowledgeProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(id) => {
                ran_ue_ngap_id = Some(id.0);
            }
            _ => {}
        }
    }

    Ok(HandoverCancelAcknowledgeData {
        amf_ue_ngap_id: amf_ue_ngap_id.ok_or_else(|| {
            HandoverError::MissingMandatoryIe("AMF-UE-NGAP-ID".to_string())
        })?,
        ran_ue_ngap_id: ran_ue_ngap_id.ok_or_else(|| {
            HandoverError::MissingMandatoryIe("RAN-UE-NGAP-ID".to_string())
        })?,
    })
}


// ============================================================================
// Helper Functions
// ============================================================================

use crate::procedures::ng_setup::{
    MiscCause, NasCause, ProtocolCause, RadioNetworkCause, TransportCause,
};

fn build_cause(cause: &HandoverCause) -> Cause {
    match cause {
        NgSetupFailureCause::RadioNetwork(rn) => Cause::RadioNetwork(build_radio_network_cause(rn)),
        NgSetupFailureCause::Transport(t) => Cause::Transport(build_transport_cause(t)),
        NgSetupFailureCause::Nas(n) => Cause::Nas(build_nas_cause(n)),
        NgSetupFailureCause::Protocol(p) => Cause::Protocol(build_protocol_cause(p)),
        NgSetupFailureCause::Misc(m) => Cause::Misc(build_misc_cause(m)),
    }
}

fn build_radio_network_cause(cause: &RadioNetworkCause) -> CauseRadioNetwork {
    let value = match cause {
        RadioNetworkCause::Unspecified => CauseRadioNetwork::UNSPECIFIED,
        RadioNetworkCause::TxnrelocoverallExpiry => CauseRadioNetwork::TXNRELOCOVERALL_EXPIRY,
        RadioNetworkCause::SuccessfulHandover => CauseRadioNetwork::SUCCESSFUL_HANDOVER,
        RadioNetworkCause::ReleaseDueToNgranGeneratedReason => {
            CauseRadioNetwork::RELEASE_DUE_TO_NGRAN_GENERATED_REASON
        }
        RadioNetworkCause::ReleaseDueTo5gcGeneratedReason => {
            CauseRadioNetwork::RELEASE_DUE_TO_5GC_GENERATED_REASON
        }
        RadioNetworkCause::HandoverCancelled => CauseRadioNetwork::HANDOVER_CANCELLED,
        RadioNetworkCause::PartialHandover => CauseRadioNetwork::PARTIAL_HANDOVER,
        RadioNetworkCause::HoFailureInTarget5gcNgranNodeOrTargetSystem => {
            CauseRadioNetwork::HO_FAILURE_IN_TARGET_5GC_NGRAN_NODE_OR_TARGET_SYSTEM
        }
        RadioNetworkCause::HoTargetNotAllowed => CauseRadioNetwork::HO_TARGET_NOT_ALLOWED,
        RadioNetworkCause::TngrelocOverallExpiry => CauseRadioNetwork::TNGRELOCOVERALL_EXPIRY,
        RadioNetworkCause::TngrelocPrepExpiry => CauseRadioNetwork::TNGRELOCPREP_EXPIRY,
        RadioNetworkCause::CellNotAvailable => CauseRadioNetwork::CELL_NOT_AVAILABLE,
        RadioNetworkCause::UnknownTargetId => CauseRadioNetwork::UNKNOWN_TARGET_ID,
        RadioNetworkCause::NoRadioResourcesAvailableInTargetCell => {
            CauseRadioNetwork::NO_RADIO_RESOURCES_AVAILABLE_IN_TARGET_CELL
        }
        RadioNetworkCause::UnknownLocalUeNgapId => CauseRadioNetwork::UNKNOWN_LOCAL_UE_NGAP_ID,
        RadioNetworkCause::InconsistentRemoteUeNgapId => {
            CauseRadioNetwork::INCONSISTENT_REMOTE_UE_NGAP_ID
        }
        RadioNetworkCause::HandoverDesirableForRadioReason => {
            CauseRadioNetwork::HANDOVER_DESIRABLE_FOR_RADIO_REASON
        }
        RadioNetworkCause::TimeCriticalHandover => CauseRadioNetwork::TIME_CRITICAL_HANDOVER,
        RadioNetworkCause::ResourceOptimisationHandover => {
            CauseRadioNetwork::RESOURCE_OPTIMISATION_HANDOVER
        }
        RadioNetworkCause::ReduceLoadInServingCell => {
            CauseRadioNetwork::REDUCE_LOAD_IN_SERVING_CELL
        }
        RadioNetworkCause::UserInactivity => CauseRadioNetwork::USER_INACTIVITY,
        RadioNetworkCause::RadioConnectionWithUeLost => {
            CauseRadioNetwork::RADIO_CONNECTION_WITH_UE_LOST
        }
        RadioNetworkCause::RadioResourcesNotAvailable => {
            CauseRadioNetwork::RADIO_RESOURCES_NOT_AVAILABLE
        }
        RadioNetworkCause::InvalidQosCombination => CauseRadioNetwork::INVALID_QOS_COMBINATION,
        RadioNetworkCause::FailureInRadioInterfaceProcedure => {
            CauseRadioNetwork::FAILURE_IN_RADIO_INTERFACE_PROCEDURE
        }
        RadioNetworkCause::InteractionWithOtherProcedure => {
            CauseRadioNetwork::INTERACTION_WITH_OTHER_PROCEDURE
        }
        RadioNetworkCause::UnknownPduSessionId => CauseRadioNetwork::UNKNOWN_PDU_SESSION_ID,
        RadioNetworkCause::UnkownQosFlowId => CauseRadioNetwork::UNKOWN_QOS_FLOW_ID,
        RadioNetworkCause::MultipleQosFlowIdInstances => {
            CauseRadioNetwork::MULTIPLE_QOS_FLOW_ID_INSTANCES
        }
        RadioNetworkCause::UnknownMappedUeNgapId => CauseRadioNetwork::UNSPECIFIED,
        RadioNetworkCause::Other(v) => *v,
    };
    CauseRadioNetwork(value)
}

fn build_transport_cause(cause: &TransportCause) -> CauseTransport {
    let value = match cause {
        TransportCause::TransportResourceUnavailable => {
            CauseTransport::TRANSPORT_RESOURCE_UNAVAILABLE
        }
        TransportCause::Unspecified => CauseTransport::UNSPECIFIED,
        TransportCause::Other(v) => *v,
    };
    CauseTransport(value)
}

fn build_nas_cause(cause: &NasCause) -> CauseNas {
    let value = match cause {
        NasCause::NormalRelease => CauseNas::NORMAL_RELEASE,
        NasCause::AuthenticationFailure => CauseNas::AUTHENTICATION_FAILURE,
        NasCause::Deregister => CauseNas::DEREGISTER,
        NasCause::Unspecified => CauseNas::UNSPECIFIED,
        NasCause::Other(v) => *v,
    };
    CauseNas(value)
}

fn build_protocol_cause(cause: &ProtocolCause) -> CauseProtocol {
    let value = match cause {
        ProtocolCause::TransferSyntaxError => CauseProtocol::TRANSFER_SYNTAX_ERROR,
        ProtocolCause::AbstractSyntaxErrorReject => CauseProtocol::ABSTRACT_SYNTAX_ERROR_REJECT,
        ProtocolCause::AbstractSyntaxErrorIgnoreAndNotify => {
            CauseProtocol::ABSTRACT_SYNTAX_ERROR_IGNORE_AND_NOTIFY
        }
        ProtocolCause::MessageNotCompatibleWithReceiverState => {
            CauseProtocol::MESSAGE_NOT_COMPATIBLE_WITH_RECEIVER_STATE
        }
        ProtocolCause::SemanticError => CauseProtocol::SEMANTIC_ERROR,
        ProtocolCause::AbstractSyntaxErrorFalselyConstructedMessage => {
            CauseProtocol::ABSTRACT_SYNTAX_ERROR_FALSELY_CONSTRUCTED_MESSAGE
        }
        ProtocolCause::Unspecified => CauseProtocol::UNSPECIFIED,
        ProtocolCause::Other(v) => *v,
    };
    CauseProtocol(value)
}

fn build_misc_cause(cause: &MiscCause) -> CauseMisc {
    let value = match cause {
        MiscCause::ControlProcessingOverload => CauseMisc::CONTROL_PROCESSING_OVERLOAD,
        MiscCause::NotEnoughUserPlaneProcessingResources => {
            CauseMisc::NOT_ENOUGH_USER_PLANE_PROCESSING_RESOURCES
        }
        MiscCause::HardwareFailure => CauseMisc::HARDWARE_FAILURE,
        MiscCause::OmIntervention => CauseMisc::OM_INTERVENTION,
        MiscCause::UnknownPlmnOrSnpn => CauseMisc::UNKNOWN_PLMN_OR_SNPN,
        MiscCause::Unspecified => CauseMisc::UNSPECIFIED,
        MiscCause::Other(v) => *v,
    };
    CauseMisc(value)
}

fn parse_cause(cause: &Cause) -> HandoverCause {
    match cause {
        Cause::RadioNetwork(rn) => NgSetupFailureCause::RadioNetwork(parse_radio_network_cause(rn)),
        Cause::Transport(t) => NgSetupFailureCause::Transport(parse_transport_cause(t)),
        Cause::Nas(n) => NgSetupFailureCause::Nas(parse_nas_cause(n)),
        Cause::Protocol(p) => NgSetupFailureCause::Protocol(parse_protocol_cause(p)),
        Cause::Misc(m) => NgSetupFailureCause::Misc(parse_misc_cause(m)),
        Cause::Choice_Extensions(_) => NgSetupFailureCause::Misc(MiscCause::Unspecified),
    }
}

fn parse_radio_network_cause(cause: &CauseRadioNetwork) -> RadioNetworkCause {
    match cause.0 {
        CauseRadioNetwork::UNSPECIFIED => RadioNetworkCause::Unspecified,
        CauseRadioNetwork::TXNRELOCOVERALL_EXPIRY => RadioNetworkCause::TxnrelocoverallExpiry,
        CauseRadioNetwork::SUCCESSFUL_HANDOVER => RadioNetworkCause::SuccessfulHandover,
        CauseRadioNetwork::RELEASE_DUE_TO_NGRAN_GENERATED_REASON => {
            RadioNetworkCause::ReleaseDueToNgranGeneratedReason
        }
        CauseRadioNetwork::RELEASE_DUE_TO_5GC_GENERATED_REASON => {
            RadioNetworkCause::ReleaseDueTo5gcGeneratedReason
        }
        CauseRadioNetwork::HANDOVER_CANCELLED => RadioNetworkCause::HandoverCancelled,
        CauseRadioNetwork::PARTIAL_HANDOVER => RadioNetworkCause::PartialHandover,
        CauseRadioNetwork::HO_FAILURE_IN_TARGET_5GC_NGRAN_NODE_OR_TARGET_SYSTEM => {
            RadioNetworkCause::HoFailureInTarget5gcNgranNodeOrTargetSystem
        }
        CauseRadioNetwork::HO_TARGET_NOT_ALLOWED => RadioNetworkCause::HoTargetNotAllowed,
        CauseRadioNetwork::TNGRELOCOVERALL_EXPIRY => RadioNetworkCause::TngrelocOverallExpiry,
        CauseRadioNetwork::TNGRELOCPREP_EXPIRY => RadioNetworkCause::TngrelocPrepExpiry,
        CauseRadioNetwork::CELL_NOT_AVAILABLE => RadioNetworkCause::CellNotAvailable,
        CauseRadioNetwork::UNKNOWN_TARGET_ID => RadioNetworkCause::UnknownTargetId,
        CauseRadioNetwork::NO_RADIO_RESOURCES_AVAILABLE_IN_TARGET_CELL => {
            RadioNetworkCause::NoRadioResourcesAvailableInTargetCell
        }
        CauseRadioNetwork::UNKNOWN_LOCAL_UE_NGAP_ID => RadioNetworkCause::UnknownLocalUeNgapId,
        CauseRadioNetwork::INCONSISTENT_REMOTE_UE_NGAP_ID => {
            RadioNetworkCause::InconsistentRemoteUeNgapId
        }
        CauseRadioNetwork::HANDOVER_DESIRABLE_FOR_RADIO_REASON => {
            RadioNetworkCause::HandoverDesirableForRadioReason
        }
        CauseRadioNetwork::TIME_CRITICAL_HANDOVER => RadioNetworkCause::TimeCriticalHandover,
        CauseRadioNetwork::RESOURCE_OPTIMISATION_HANDOVER => {
            RadioNetworkCause::ResourceOptimisationHandover
        }
        CauseRadioNetwork::REDUCE_LOAD_IN_SERVING_CELL => {
            RadioNetworkCause::ReduceLoadInServingCell
        }
        CauseRadioNetwork::USER_INACTIVITY => RadioNetworkCause::UserInactivity,
        CauseRadioNetwork::RADIO_CONNECTION_WITH_UE_LOST => {
            RadioNetworkCause::RadioConnectionWithUeLost
        }
        CauseRadioNetwork::RADIO_RESOURCES_NOT_AVAILABLE => {
            RadioNetworkCause::RadioResourcesNotAvailable
        }
        CauseRadioNetwork::INVALID_QOS_COMBINATION => RadioNetworkCause::InvalidQosCombination,
        CauseRadioNetwork::FAILURE_IN_RADIO_INTERFACE_PROCEDURE => {
            RadioNetworkCause::FailureInRadioInterfaceProcedure
        }
        CauseRadioNetwork::INTERACTION_WITH_OTHER_PROCEDURE => {
            RadioNetworkCause::InteractionWithOtherProcedure
        }
        CauseRadioNetwork::UNKNOWN_PDU_SESSION_ID => RadioNetworkCause::UnknownPduSessionId,
        CauseRadioNetwork::UNKOWN_QOS_FLOW_ID => RadioNetworkCause::UnkownQosFlowId,
        CauseRadioNetwork::MULTIPLE_QOS_FLOW_ID_INSTANCES => {
            RadioNetworkCause::MultipleQosFlowIdInstances
        }
        other => RadioNetworkCause::Other(other),
    }
}

fn parse_transport_cause(cause: &CauseTransport) -> TransportCause {
    match cause.0 {
        CauseTransport::TRANSPORT_RESOURCE_UNAVAILABLE => {
            TransportCause::TransportResourceUnavailable
        }
        CauseTransport::UNSPECIFIED => TransportCause::Unspecified,
        other => TransportCause::Other(other),
    }
}

fn parse_nas_cause(cause: &CauseNas) -> NasCause {
    match cause.0 {
        CauseNas::NORMAL_RELEASE => NasCause::NormalRelease,
        CauseNas::AUTHENTICATION_FAILURE => NasCause::AuthenticationFailure,
        CauseNas::DEREGISTER => NasCause::Deregister,
        CauseNas::UNSPECIFIED => NasCause::Unspecified,
        other => NasCause::Other(other),
    }
}

fn parse_protocol_cause(cause: &CauseProtocol) -> ProtocolCause {
    match cause.0 {
        CauseProtocol::TRANSFER_SYNTAX_ERROR => ProtocolCause::TransferSyntaxError,
        CauseProtocol::ABSTRACT_SYNTAX_ERROR_REJECT => ProtocolCause::AbstractSyntaxErrorReject,
        CauseProtocol::ABSTRACT_SYNTAX_ERROR_IGNORE_AND_NOTIFY => {
            ProtocolCause::AbstractSyntaxErrorIgnoreAndNotify
        }
        CauseProtocol::MESSAGE_NOT_COMPATIBLE_WITH_RECEIVER_STATE => {
            ProtocolCause::MessageNotCompatibleWithReceiverState
        }
        CauseProtocol::SEMANTIC_ERROR => ProtocolCause::SemanticError,
        CauseProtocol::ABSTRACT_SYNTAX_ERROR_FALSELY_CONSTRUCTED_MESSAGE => {
            ProtocolCause::AbstractSyntaxErrorFalselyConstructedMessage
        }
        CauseProtocol::UNSPECIFIED => ProtocolCause::Unspecified,
        other => ProtocolCause::Other(other),
    }
}

fn parse_misc_cause(cause: &CauseMisc) -> MiscCause {
    match cause.0 {
        CauseMisc::CONTROL_PROCESSING_OVERLOAD => MiscCause::ControlProcessingOverload,
        CauseMisc::NOT_ENOUGH_USER_PLANE_PROCESSING_RESOURCES => {
            MiscCause::NotEnoughUserPlaneProcessingResources
        }
        CauseMisc::HARDWARE_FAILURE => MiscCause::HardwareFailure,
        CauseMisc::OM_INTERVENTION => MiscCause::OmIntervention,
        CauseMisc::UNKNOWN_PLMN_OR_SNPN => MiscCause::UnknownPlmnOrSnpn,
        CauseMisc::UNSPECIFIED => MiscCause::Unspecified,
        other => MiscCause::Other(other),
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Build and encode a Handover Required to bytes
pub fn encode_handover_required(params: &HandoverRequiredParams) -> Result<Vec<u8>, HandoverError> {
    let pdu = build_handover_required(params)?;
    Ok(encode_ngap_pdu(&pdu)?)
}

/// Build and encode a Handover Request Acknowledge to bytes
pub fn encode_handover_request_acknowledge(
    params: &HandoverRequestAcknowledgeParams,
) -> Result<Vec<u8>, HandoverError> {
    let pdu = build_handover_request_acknowledge(params)?;
    Ok(encode_ngap_pdu(&pdu)?)
}

/// Build and encode a Handover Failure to bytes
pub fn encode_handover_failure(params: &HandoverFailureParams) -> Result<Vec<u8>, HandoverError> {
    let pdu = build_handover_failure(params)?;
    Ok(encode_ngap_pdu(&pdu)?)
}

/// Build and encode a Handover Notify to bytes
pub fn encode_handover_notify(params: &HandoverNotifyParams) -> Result<Vec<u8>, HandoverError> {
    let pdu = build_handover_notify(params)?;
    Ok(encode_ngap_pdu(&pdu)?)
}

/// Build and encode a Handover Cancel to bytes
pub fn encode_handover_cancel(params: &HandoverCancelParams) -> Result<Vec<u8>, HandoverError> {
    let pdu = build_handover_cancel(params)?;
    Ok(encode_ngap_pdu(&pdu)?)
}

/// Decode and parse a Handover Required from bytes
pub fn decode_handover_required(bytes: &[u8]) -> Result<HandoverRequiredData, HandoverError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_handover_required(&pdu)
}

/// Decode and parse a Handover Command from bytes
pub fn decode_handover_command(bytes: &[u8]) -> Result<HandoverCommandData, HandoverError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_handover_command(&pdu)
}

/// Decode and parse a Handover Preparation Failure from bytes
pub fn decode_handover_preparation_failure(
    bytes: &[u8],
) -> Result<HandoverPreparationFailureData, HandoverError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_handover_preparation_failure(&pdu)
}

/// Decode and parse a Handover Request from bytes
pub fn decode_handover_request(bytes: &[u8]) -> Result<HandoverRequestData, HandoverError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_handover_request(&pdu)
}

/// Decode and parse a Handover Request Acknowledge from bytes
pub fn decode_handover_request_acknowledge(
    bytes: &[u8],
) -> Result<HandoverRequestAcknowledgeData, HandoverError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_handover_request_acknowledge(&pdu)
}

/// Decode and parse a Handover Failure from bytes
pub fn decode_handover_failure(bytes: &[u8]) -> Result<HandoverFailureData, HandoverError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_handover_failure(&pdu)
}

/// Decode and parse a Handover Notify from bytes
pub fn decode_handover_notify(bytes: &[u8]) -> Result<HandoverNotifyData, HandoverError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_handover_notify(&pdu)
}

/// Decode and parse a Handover Cancel from bytes
pub fn decode_handover_cancel(bytes: &[u8]) -> Result<HandoverCancelData, HandoverError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_handover_cancel(&pdu)
}

/// Decode and parse a Handover Cancel Acknowledge from bytes
pub fn decode_handover_cancel_acknowledge(
    bytes: &[u8],
) -> Result<HandoverCancelAcknowledgeData, HandoverError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_handover_cancel_acknowledge(&pdu)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_handover_cancel_params() -> HandoverCancelParams {
        HandoverCancelParams {
            amf_ue_ngap_id: 12345,
            ran_ue_ngap_id: 67890,
            cause: NgSetupFailureCause::RadioNetwork(RadioNetworkCause::HandoverCancelled),
        }
    }

    fn create_test_handover_notify_params() -> HandoverNotifyParams {
        HandoverNotifyParams {
            amf_ue_ngap_id: 12345,
            ran_ue_ngap_id: 67890,
            user_location_info: UserLocationInfoNr {
                nr_cgi: NrCgiValue {
                    plmn_identity: [0x00, 0xF1, 0x10],
                    nr_cell_identity: 0x000000001, // 36-bit value
                },
                tai: TaiValue {
                    plmn_identity: [0x00, 0xF1, 0x10],
                    tac: [0x00, 0x00, 0x01],
                },
            },
        }
    }

    fn create_test_handover_failure_params() -> HandoverFailureParams {
        HandoverFailureParams {
            amf_ue_ngap_id: 12345,
            cause: NgSetupFailureCause::RadioNetwork(
                RadioNetworkCause::NoRadioResourcesAvailableInTargetCell,
            ),
        }
    }

    #[test]
    fn test_build_handover_cancel() {
        let params = create_test_handover_cancel_params();
        let result = build_handover_cancel(&params);
        assert!(result.is_ok());

        let pdu = result.unwrap();
        match pdu {
            NGAP_PDU::InitiatingMessage(msg) => {
                assert_eq!(msg.procedure_code.0, ID_HANDOVER_CANCEL);
            }
            _ => panic!("Expected InitiatingMessage"),
        }
    }

    #[test]
    fn test_build_handover_notify() {
        let params = create_test_handover_notify_params();
        let result = build_handover_notify(&params);
        assert!(result.is_ok());

        let pdu = result.unwrap();
        match pdu {
            NGAP_PDU::InitiatingMessage(msg) => {
                assert_eq!(msg.procedure_code.0, ID_HANDOVER_NOTIFICATION);
            }
            _ => panic!("Expected InitiatingMessage"),
        }
    }

    #[test]
    fn test_build_handover_failure() {
        let params = create_test_handover_failure_params();
        let result = build_handover_failure(&params);
        assert!(result.is_ok());

        let pdu = result.unwrap();
        match pdu {
            NGAP_PDU::UnsuccessfulOutcome(outcome) => {
                assert_eq!(outcome.procedure_code.0, ID_HANDOVER_RESOURCE_ALLOCATION);
            }
            _ => panic!("Expected UnsuccessfulOutcome"),
        }
    }

    #[test]
    fn test_encode_handover_cancel() {
        let params = create_test_handover_cancel_params();
        let result = encode_handover_cancel(&params);
        assert!(result.is_ok());

        let bytes = result.unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_encode_handover_notify() {
        let params = create_test_handover_notify_params();
        let result = encode_handover_notify(&params);
        assert!(result.is_ok());

        let bytes = result.unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_encode_handover_failure() {
        let params = create_test_handover_failure_params();
        let result = encode_handover_failure(&params);
        assert!(result.is_ok());

        let bytes = result.unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_handover_cancel_roundtrip() {
        let params = create_test_handover_cancel_params();

        // Build and encode
        let pdu = build_handover_cancel(&params).expect("Failed to build");
        let encoded = encode_ngap_pdu(&pdu).expect("Failed to encode");

        // Decode and parse
        let decoded_pdu = decode_ngap_pdu(&encoded).expect("Failed to decode");
        let parsed = parse_handover_cancel(&decoded_pdu).expect("Failed to parse");

        assert_eq!(parsed.amf_ue_ngap_id, params.amf_ue_ngap_id);
        assert_eq!(parsed.ran_ue_ngap_id, params.ran_ue_ngap_id);
    }

    #[test]
    fn test_handover_notify_roundtrip() {
        let params = create_test_handover_notify_params();

        // Build and encode
        let pdu = build_handover_notify(&params).expect("Failed to build");
        let encoded = encode_ngap_pdu(&pdu).expect("Failed to encode");

        // Decode and parse
        let decoded_pdu = decode_ngap_pdu(&encoded).expect("Failed to decode");
        let parsed = parse_handover_notify(&decoded_pdu).expect("Failed to parse");

        assert_eq!(parsed.amf_ue_ngap_id, params.amf_ue_ngap_id);
        assert_eq!(parsed.ran_ue_ngap_id, params.ran_ue_ngap_id);
    }

    #[test]
    fn test_handover_failure_roundtrip() {
        let params = create_test_handover_failure_params();

        // Build and encode
        let pdu = build_handover_failure(&params).expect("Failed to build");
        let encoded = encode_ngap_pdu(&pdu).expect("Failed to encode");

        // Decode and parse
        let decoded_pdu = decode_ngap_pdu(&encoded).expect("Failed to decode");
        let parsed = parse_handover_failure(&decoded_pdu).expect("Failed to parse");

        assert_eq!(parsed.amf_ue_ngap_id, params.amf_ue_ngap_id);
    }

    #[test]
    fn test_handover_type_conversion() {
        // Test all handover type conversions
        let types = [
            HandoverTypeValue::Intra5gs,
            HandoverTypeValue::FivegsToEps,
            HandoverTypeValue::EpsTo5gs,
        ];

        for ht in types {
            let converted: HandoverType = ht.into();
            let back: HandoverTypeValue = HandoverTypeValue::try_from(&converted).unwrap();
            assert_eq!(ht, back);
        }
    }
}
