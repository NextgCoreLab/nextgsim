//! UE Context Release Procedure
//!
//! Implements the UE Context Release procedure as defined in 3GPP TS 38.413 Section 8.3.3.
//! This procedure is used to release the UE context at the NG-RAN node.
//!
//! The procedure consists of:
//! - UE Context Release Request: Sent by NG-RAN to AMF to request release
//! - UE Context Release Command: Sent by AMF to NG-RAN to command release
//! - UE Context Release Complete: Sent by NG-RAN to AMF to confirm release

use crate::codec::generated::*;
use crate::codec::{decode_ngap_pdu, encode_ngap_pdu, NgapCodecError};
use crate::procedures::ng_setup::{
    MiscCause, NasCause, NgSetupFailureCause, ProtocolCause, RadioNetworkCause, TransportCause,
};
use thiserror::Error;

/// Errors that can occur during UE Context Release procedures
#[derive(Debug, Error)]
pub enum UeContextReleaseError {
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

/// Cause values for UE Context Release
pub type UeContextReleaseCause = NgSetupFailureCause;

// ============================================================================
// UE Context Release Request
// ============================================================================

/// Parameters for building a UE Context Release Request
#[derive(Debug, Clone)]
pub struct UeContextReleaseRequestParams {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
    /// Cause of release request
    pub cause: UeContextReleaseCause,
}

/// Parsed UE Context Release Request data
#[derive(Debug, Clone)]
pub struct UeContextReleaseRequestData {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
    /// Cause of release request
    pub cause: UeContextReleaseCause,
}

/// Build a UE Context Release Request PDU
///
/// # Arguments
/// * `params` - Parameters for the UE Context Release Request
///
/// # Returns
/// * `Ok(NGAP_PDU)` - The constructed PDU
/// * `Err(UeContextReleaseError)` - If construction fails
pub fn build_ue_context_release_request(
    params: &UeContextReleaseRequestParams,
) -> Result<NGAP_PDU, UeContextReleaseError> {
    let mut protocol_ies = Vec::new();

    // IE: AMF-UE-NGAP-ID (mandatory)
    protocol_ies.push(UEContextReleaseRequestProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_AMF_UE_NGAP_ID),
        criticality: Criticality(Criticality::REJECT),
        value: UEContextReleaseRequestProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(AMF_UE_NGAP_ID(
            params.amf_ue_ngap_id,
        )),
    });

    // IE: RAN-UE-NGAP-ID (mandatory)
    protocol_ies.push(UEContextReleaseRequestProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_RAN_UE_NGAP_ID),
        criticality: Criticality(Criticality::REJECT),
        value: UEContextReleaseRequestProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(RAN_UE_NGAP_ID(
            params.ran_ue_ngap_id,
        )),
    });

    // IE: Cause (mandatory)
    let cause = build_cause(&params.cause);
    protocol_ies.push(UEContextReleaseRequestProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_CAUSE),
        criticality: Criticality(Criticality::IGNORE),
        value: UEContextReleaseRequestProtocolIEs_EntryValue::Id_Cause(cause),
    });

    let request = UEContextReleaseRequest {
        protocol_i_es: UEContextReleaseRequestProtocolIEs(protocol_ies),
    };

    let initiating_message = InitiatingMessage {
        procedure_code: ProcedureCode(ID_UE_CONTEXT_RELEASE_REQUEST),
        criticality: Criticality(Criticality::IGNORE),
        value: InitiatingMessageValue::Id_UEContextReleaseRequest(request),
    };

    Ok(NGAP_PDU::InitiatingMessage(initiating_message))
}

/// Parse a UE Context Release Request from an NGAP PDU
///
/// # Arguments
/// * `pdu` - The NGAP PDU to parse
///
/// # Returns
/// * `Ok(UeContextReleaseRequestData)` - The parsed request data
/// * `Err(UeContextReleaseError)` - If parsing fails
pub fn parse_ue_context_release_request(
    pdu: &NGAP_PDU,
) -> Result<UeContextReleaseRequestData, UeContextReleaseError> {
    let initiating_message = match pdu {
        NGAP_PDU::InitiatingMessage(msg) => msg,
        _ => {
            return Err(UeContextReleaseError::InvalidMessageType {
                expected: "InitiatingMessage".to_string(),
                actual: format!("{:?}", pdu),
            })
        }
    };

    let request = match &initiating_message.value {
        InitiatingMessageValue::Id_UEContextReleaseRequest(req) => req,
        _ => {
            return Err(UeContextReleaseError::InvalidMessageType {
                expected: "UEContextReleaseRequest".to_string(),
                actual: format!("{:?}", initiating_message.value),
            })
        }
    };

    let mut amf_ue_ngap_id: Option<u64> = None;
    let mut ran_ue_ngap_id: Option<u32> = None;
    let mut cause: Option<UeContextReleaseCause> = None;

    for ie in &request.protocol_i_es.0 {
        match &ie.value {
            UEContextReleaseRequestProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(id) => {
                amf_ue_ngap_id = Some(id.0);
            }
            UEContextReleaseRequestProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(id) => {
                ran_ue_ngap_id = Some(id.0);
            }
            UEContextReleaseRequestProtocolIEs_EntryValue::Id_Cause(c) => {
                cause = Some(parse_cause(c));
            }
            _ => {
                // Ignore other IEs
            }
        }
    }

    Ok(UeContextReleaseRequestData {
        amf_ue_ngap_id: amf_ue_ngap_id.ok_or_else(|| {
            UeContextReleaseError::MissingMandatoryIe("AMF-UE-NGAP-ID".to_string())
        })?,
        ran_ue_ngap_id: ran_ue_ngap_id.ok_or_else(|| {
            UeContextReleaseError::MissingMandatoryIe("RAN-UE-NGAP-ID".to_string())
        })?,
        cause: cause
            .ok_or_else(|| UeContextReleaseError::MissingMandatoryIe("Cause".to_string()))?,
    })
}

// ============================================================================
// UE Context Release Command
// ============================================================================

/// UE NGAP IDs - can be either a pair or just AMF ID
#[derive(Debug, Clone)]
pub enum UeNgapIds {
    /// Both AMF and RAN UE NGAP IDs
    Pair {
        /// AMF UE NGAP ID
        amf_ue_ngap_id: u64,
        /// RAN UE NGAP ID
        ran_ue_ngap_id: u32,
    },
    /// Only AMF UE NGAP ID (when RAN ID is unknown)
    AmfOnly(u64),
}

/// Parameters for building a UE Context Release Command
#[derive(Debug, Clone)]
pub struct UeContextReleaseCommandParams {
    /// UE NGAP IDs
    pub ue_ngap_ids: UeNgapIds,
    /// Cause of release
    pub cause: UeContextReleaseCause,
}

/// Parsed UE Context Release Command data
#[derive(Debug, Clone)]
pub struct UeContextReleaseCommandData {
    /// UE NGAP IDs
    pub ue_ngap_ids: UeNgapIds,
    /// Cause of release
    pub cause: UeContextReleaseCause,
}

/// Build a UE Context Release Command PDU
///
/// # Arguments
/// * `params` - Parameters for the UE Context Release Command
///
/// # Returns
/// * `Ok(NGAP_PDU)` - The constructed PDU
/// * `Err(UeContextReleaseError)` - If construction fails
pub fn build_ue_context_release_command(
    params: &UeContextReleaseCommandParams,
) -> Result<NGAP_PDU, UeContextReleaseError> {
    let mut protocol_ies = Vec::new();

    // IE: UE-NGAP-IDs (mandatory)
    let ue_ngap_ids = build_ue_ngap_ids(&params.ue_ngap_ids);
    protocol_ies.push(UEContextReleaseCommandProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_UE_NGAP_I_DS),
        criticality: Criticality(Criticality::REJECT),
        value: UEContextReleaseCommandProtocolIEs_EntryValue::Id_UE_NGAP_IDs(ue_ngap_ids),
    });

    // IE: Cause (mandatory)
    let cause = build_cause(&params.cause);
    protocol_ies.push(UEContextReleaseCommandProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_CAUSE),
        criticality: Criticality(Criticality::IGNORE),
        value: UEContextReleaseCommandProtocolIEs_EntryValue::Id_Cause(cause),
    });

    let command = UEContextReleaseCommand {
        protocol_i_es: UEContextReleaseCommandProtocolIEs(protocol_ies),
    };

    let initiating_message = InitiatingMessage {
        procedure_code: ProcedureCode(ID_UE_CONTEXT_RELEASE),
        criticality: Criticality(Criticality::REJECT),
        value: InitiatingMessageValue::Id_UEContextRelease(command),
    };

    Ok(NGAP_PDU::InitiatingMessage(initiating_message))
}

fn build_ue_ngap_ids(ids: &UeNgapIds) -> UE_NGAP_IDs {
    match ids {
        UeNgapIds::Pair {
            amf_ue_ngap_id,
            ran_ue_ngap_id,
        } => UE_NGAP_IDs::UE_NGAP_ID_pair(UE_NGAP_ID_pair {
            amf_ue_ngap_id: AMF_UE_NGAP_ID(*amf_ue_ngap_id),
            ran_ue_ngap_id: RAN_UE_NGAP_ID(*ran_ue_ngap_id),
            ie_extensions: None,
        }),
        UeNgapIds::AmfOnly(amf_id) => UE_NGAP_IDs::AMF_UE_NGAP_ID(AMF_UE_NGAP_ID(*amf_id)),
    }
}

/// Parse a UE Context Release Command from an NGAP PDU
///
/// # Arguments
/// * `pdu` - The NGAP PDU to parse
///
/// # Returns
/// * `Ok(UeContextReleaseCommandData)` - The parsed command data
/// * `Err(UeContextReleaseError)` - If parsing fails
pub fn parse_ue_context_release_command(
    pdu: &NGAP_PDU,
) -> Result<UeContextReleaseCommandData, UeContextReleaseError> {
    let initiating_message = match pdu {
        NGAP_PDU::InitiatingMessage(msg) => msg,
        _ => {
            return Err(UeContextReleaseError::InvalidMessageType {
                expected: "InitiatingMessage".to_string(),
                actual: format!("{:?}", pdu),
            })
        }
    };

    let command = match &initiating_message.value {
        InitiatingMessageValue::Id_UEContextRelease(cmd) => cmd,
        _ => {
            return Err(UeContextReleaseError::InvalidMessageType {
                expected: "UEContextReleaseCommand".to_string(),
                actual: format!("{:?}", initiating_message.value),
            })
        }
    };

    let mut ue_ngap_ids: Option<UeNgapIds> = None;
    let mut cause: Option<UeContextReleaseCause> = None;

    for ie in &command.protocol_i_es.0 {
        match &ie.value {
            UEContextReleaseCommandProtocolIEs_EntryValue::Id_UE_NGAP_IDs(ids) => {
                ue_ngap_ids = Some(parse_ue_ngap_ids(ids));
            }
            UEContextReleaseCommandProtocolIEs_EntryValue::Id_Cause(c) => {
                cause = Some(parse_cause(c));
            }
        }
    }

    Ok(UeContextReleaseCommandData {
        ue_ngap_ids: ue_ngap_ids.ok_or_else(|| {
            UeContextReleaseError::MissingMandatoryIe("UE-NGAP-IDs".to_string())
        })?,
        cause: cause
            .ok_or_else(|| UeContextReleaseError::MissingMandatoryIe("Cause".to_string()))?,
    })
}

fn parse_ue_ngap_ids(ids: &UE_NGAP_IDs) -> UeNgapIds {
    match ids {
        UE_NGAP_IDs::UE_NGAP_ID_pair(pair) => UeNgapIds::Pair {
            amf_ue_ngap_id: pair.amf_ue_ngap_id.0,
            ran_ue_ngap_id: pair.ran_ue_ngap_id.0,
        },
        UE_NGAP_IDs::AMF_UE_NGAP_ID(amf_id) => UeNgapIds::AmfOnly(amf_id.0),
        UE_NGAP_IDs::Choice_Extensions(_) => {
            // Fallback for extensions - treat as AMF only with 0
            UeNgapIds::AmfOnly(0)
        }
    }
}

// ============================================================================
// UE Context Release Complete
// ============================================================================

/// Parameters for building a UE Context Release Complete
#[derive(Debug, Clone)]
pub struct UeContextReleaseCompleteParams {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
}

/// Parsed UE Context Release Complete data
#[derive(Debug, Clone)]
pub struct UeContextReleaseCompleteData {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
}

/// Build a UE Context Release Complete PDU
///
/// # Arguments
/// * `params` - Parameters for the UE Context Release Complete
///
/// # Returns
/// * `Ok(NGAP_PDU)` - The constructed PDU
/// * `Err(UeContextReleaseError)` - If construction fails
pub fn build_ue_context_release_complete(
    params: &UeContextReleaseCompleteParams,
) -> Result<NGAP_PDU, UeContextReleaseError> {
    let mut protocol_ies = Vec::new();

    // IE: AMF-UE-NGAP-ID (mandatory)
    protocol_ies.push(UEContextReleaseCompleteProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_AMF_UE_NGAP_ID),
        criticality: Criticality(Criticality::IGNORE),
        value: UEContextReleaseCompleteProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(AMF_UE_NGAP_ID(
            params.amf_ue_ngap_id,
        )),
    });

    // IE: RAN-UE-NGAP-ID (mandatory)
    protocol_ies.push(UEContextReleaseCompleteProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_RAN_UE_NGAP_ID),
        criticality: Criticality(Criticality::IGNORE),
        value: UEContextReleaseCompleteProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(RAN_UE_NGAP_ID(
            params.ran_ue_ngap_id,
        )),
    });

    let complete = UEContextReleaseComplete {
        protocol_i_es: UEContextReleaseCompleteProtocolIEs(protocol_ies),
    };

    let successful_outcome = SuccessfulOutcome {
        procedure_code: ProcedureCode(ID_UE_CONTEXT_RELEASE),
        criticality: Criticality(Criticality::REJECT),
        value: SuccessfulOutcomeValue::Id_UEContextRelease(complete),
    };

    Ok(NGAP_PDU::SuccessfulOutcome(successful_outcome))
}

/// Parse a UE Context Release Complete from an NGAP PDU
///
/// # Arguments
/// * `pdu` - The NGAP PDU to parse
///
/// # Returns
/// * `Ok(UeContextReleaseCompleteData)` - The parsed complete data
/// * `Err(UeContextReleaseError)` - If parsing fails
pub fn parse_ue_context_release_complete(
    pdu: &NGAP_PDU,
) -> Result<UeContextReleaseCompleteData, UeContextReleaseError> {
    let successful_outcome = match pdu {
        NGAP_PDU::SuccessfulOutcome(outcome) => outcome,
        _ => {
            return Err(UeContextReleaseError::InvalidMessageType {
                expected: "SuccessfulOutcome".to_string(),
                actual: format!("{:?}", pdu),
            })
        }
    };

    let complete = match &successful_outcome.value {
        SuccessfulOutcomeValue::Id_UEContextRelease(comp) => comp,
        _ => {
            return Err(UeContextReleaseError::InvalidMessageType {
                expected: "UEContextReleaseComplete".to_string(),
                actual: format!("{:?}", successful_outcome.value),
            })
        }
    };

    let mut amf_ue_ngap_id: Option<u64> = None;
    let mut ran_ue_ngap_id: Option<u32> = None;

    for ie in &complete.protocol_i_es.0 {
        match &ie.value {
            UEContextReleaseCompleteProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(id) => {
                amf_ue_ngap_id = Some(id.0);
            }
            UEContextReleaseCompleteProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(id) => {
                ran_ue_ngap_id = Some(id.0);
            }
            _ => {
                // Ignore other IEs
            }
        }
    }

    Ok(UeContextReleaseCompleteData {
        amf_ue_ngap_id: amf_ue_ngap_id.ok_or_else(|| {
            UeContextReleaseError::MissingMandatoryIe("AMF-UE-NGAP-ID".to_string())
        })?,
        ran_ue_ngap_id: ran_ue_ngap_id.ok_or_else(|| {
            UeContextReleaseError::MissingMandatoryIe("RAN-UE-NGAP-ID".to_string())
        })?,
    })
}

// ============================================================================
// Helper Functions
// ============================================================================

fn build_cause(cause: &UeContextReleaseCause) -> Cause {
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

fn parse_cause(cause: &Cause) -> UeContextReleaseCause {
    match cause {
        Cause::RadioNetwork(rn) => {
            NgSetupFailureCause::RadioNetwork(parse_radio_network_cause(rn))
        }
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

/// Build and encode a UE Context Release Request to bytes
pub fn encode_ue_context_release_request(
    params: &UeContextReleaseRequestParams,
) -> Result<Vec<u8>, UeContextReleaseError> {
    let pdu = build_ue_context_release_request(params)?;
    Ok(encode_ngap_pdu(&pdu)?)
}

/// Decode and parse a UE Context Release Request from bytes
pub fn decode_ue_context_release_request(
    bytes: &[u8],
) -> Result<UeContextReleaseRequestData, UeContextReleaseError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_ue_context_release_request(&pdu)
}

/// Build and encode a UE Context Release Command to bytes
pub fn encode_ue_context_release_command(
    params: &UeContextReleaseCommandParams,
) -> Result<Vec<u8>, UeContextReleaseError> {
    let pdu = build_ue_context_release_command(params)?;
    Ok(encode_ngap_pdu(&pdu)?)
}

/// Decode and parse a UE Context Release Command from bytes
pub fn decode_ue_context_release_command(
    bytes: &[u8],
) -> Result<UeContextReleaseCommandData, UeContextReleaseError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_ue_context_release_command(&pdu)
}

/// Build and encode a UE Context Release Complete to bytes
pub fn encode_ue_context_release_complete(
    params: &UeContextReleaseCompleteParams,
) -> Result<Vec<u8>, UeContextReleaseError> {
    let pdu = build_ue_context_release_complete(params)?;
    Ok(encode_ngap_pdu(&pdu)?)
}

/// Decode and parse a UE Context Release Complete from bytes
pub fn decode_ue_context_release_complete(
    bytes: &[u8],
) -> Result<UeContextReleaseCompleteData, UeContextReleaseError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_ue_context_release_complete(&pdu)
}

/// Check if an NGAP PDU is a UE Context Release Request
pub fn is_ue_context_release_request(pdu: &NGAP_PDU) -> bool {
    matches!(
        pdu,
        NGAP_PDU::InitiatingMessage(msg)
            if matches!(msg.value, InitiatingMessageValue::Id_UEContextReleaseRequest(_))
    )
}

/// Check if an NGAP PDU is a UE Context Release Command
pub fn is_ue_context_release_command(pdu: &NGAP_PDU) -> bool {
    matches!(
        pdu,
        NGAP_PDU::InitiatingMessage(msg)
            if matches!(msg.value, InitiatingMessageValue::Id_UEContextRelease(_))
    )
}

/// Check if an NGAP PDU is a UE Context Release Complete
pub fn is_ue_context_release_complete(pdu: &NGAP_PDU) -> bool {
    matches!(
        pdu,
        NGAP_PDU::SuccessfulOutcome(outcome)
            if matches!(outcome.value, SuccessfulOutcomeValue::Id_UEContextRelease(_))
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_request_params() -> UeContextReleaseRequestParams {
        UeContextReleaseRequestParams {
            amf_ue_ngap_id: 12345,
            ran_ue_ngap_id: 1,
            cause: UeContextReleaseCause::RadioNetwork(RadioNetworkCause::UserInactivity),
        }
    }

    fn create_test_command_params() -> UeContextReleaseCommandParams {
        UeContextReleaseCommandParams {
            ue_ngap_ids: UeNgapIds::Pair {
                amf_ue_ngap_id: 12345,
                ran_ue_ngap_id: 1,
            },
            cause: UeContextReleaseCause::Nas(NasCause::NormalRelease),
        }
    }

    fn create_test_complete_params() -> UeContextReleaseCompleteParams {
        UeContextReleaseCompleteParams {
            amf_ue_ngap_id: 12345,
            ran_ue_ngap_id: 1,
        }
    }

    #[test]
    fn test_build_ue_context_release_request() {
        let params = create_test_request_params();
        let result = build_ue_context_release_request(&params);
        assert!(result.is_ok());

        let pdu = result.unwrap();
        assert!(is_ue_context_release_request(&pdu));
    }

    #[test]
    fn test_encode_ue_context_release_request() {
        let params = create_test_request_params();
        let result = encode_ue_context_release_request(&params);
        assert!(result.is_ok());

        let bytes = result.unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_ue_context_release_request_roundtrip() {
        let params = create_test_request_params();

        let encoded = encode_ue_context_release_request(&params).expect("Failed to encode");
        let parsed = decode_ue_context_release_request(&encoded).expect("Failed to decode");

        assert_eq!(parsed.amf_ue_ngap_id, params.amf_ue_ngap_id);
        assert_eq!(parsed.ran_ue_ngap_id, params.ran_ue_ngap_id);
        assert_eq!(parsed.cause, params.cause);
    }

    #[test]
    fn test_build_ue_context_release_command() {
        let params = create_test_command_params();
        let result = build_ue_context_release_command(&params);
        assert!(result.is_ok());

        let pdu = result.unwrap();
        assert!(is_ue_context_release_command(&pdu));
    }

    #[test]
    fn test_encode_ue_context_release_command() {
        let params = create_test_command_params();
        let result = encode_ue_context_release_command(&params);
        assert!(result.is_ok());

        let bytes = result.unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_ue_context_release_command_roundtrip() {
        let params = create_test_command_params();

        let encoded = encode_ue_context_release_command(&params).expect("Failed to encode");
        let parsed = decode_ue_context_release_command(&encoded).expect("Failed to decode");

        match (&parsed.ue_ngap_ids, &params.ue_ngap_ids) {
            (
                UeNgapIds::Pair {
                    amf_ue_ngap_id: parsed_amf,
                    ran_ue_ngap_id: parsed_ran,
                },
                UeNgapIds::Pair {
                    amf_ue_ngap_id: params_amf,
                    ran_ue_ngap_id: params_ran,
                },
            ) => {
                assert_eq!(parsed_amf, params_amf);
                assert_eq!(parsed_ran, params_ran);
            }
            _ => panic!("Expected Pair variant"),
        }
        assert_eq!(parsed.cause, params.cause);
    }

    #[test]
    fn test_ue_context_release_command_amf_only() {
        let params = UeContextReleaseCommandParams {
            ue_ngap_ids: UeNgapIds::AmfOnly(12345),
            cause: UeContextReleaseCause::Nas(NasCause::Deregister),
        };

        let encoded = encode_ue_context_release_command(&params).expect("Failed to encode");
        let parsed = decode_ue_context_release_command(&encoded).expect("Failed to decode");

        match parsed.ue_ngap_ids {
            UeNgapIds::AmfOnly(amf_id) => assert_eq!(amf_id, 12345),
            _ => panic!("Expected AmfOnly variant"),
        }
    }

    #[test]
    fn test_build_ue_context_release_complete() {
        let params = create_test_complete_params();
        let result = build_ue_context_release_complete(&params);
        assert!(result.is_ok());

        let pdu = result.unwrap();
        assert!(is_ue_context_release_complete(&pdu));
    }

    #[test]
    fn test_encode_ue_context_release_complete() {
        let params = create_test_complete_params();
        let result = encode_ue_context_release_complete(&params);
        assert!(result.is_ok());

        let bytes = result.unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_ue_context_release_complete_roundtrip() {
        let params = create_test_complete_params();

        let encoded = encode_ue_context_release_complete(&params).expect("Failed to encode");
        let parsed = decode_ue_context_release_complete(&encoded).expect("Failed to decode");

        assert_eq!(parsed.amf_ue_ngap_id, params.amf_ue_ngap_id);
        assert_eq!(parsed.ran_ue_ngap_id, params.ran_ue_ngap_id);
    }

    #[test]
    fn test_various_causes() {
        // Test with different cause types
        let causes = vec![
            UeContextReleaseCause::RadioNetwork(RadioNetworkCause::UserInactivity),
            UeContextReleaseCause::RadioNetwork(RadioNetworkCause::RadioConnectionWithUeLost),
            UeContextReleaseCause::Nas(NasCause::NormalRelease),
            UeContextReleaseCause::Nas(NasCause::Deregister),
            UeContextReleaseCause::Transport(TransportCause::Unspecified),
            UeContextReleaseCause::Protocol(ProtocolCause::Unspecified),
            UeContextReleaseCause::Misc(MiscCause::Unspecified),
        ];

        for cause in causes {
            let params = UeContextReleaseRequestParams {
                amf_ue_ngap_id: 12345,
                ran_ue_ngap_id: 1,
                cause: cause.clone(),
            };

            let encoded = encode_ue_context_release_request(&params).expect("Failed to encode");
            let parsed = decode_ue_context_release_request(&encoded).expect("Failed to decode");

            assert_eq!(parsed.cause, cause);
        }
    }
}
