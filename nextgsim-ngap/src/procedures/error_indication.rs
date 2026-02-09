//! Error Indication Procedure
//!
//! Implements the Error Indication procedure as defined in 3GPP TS 38.413 Section 8.2.7.
//! This procedure is used by the gNB or AMF to report detected errors in one incoming message,
//! provided they cannot be reported by an appropriate failure message.
//!
//! The Error Indication is a Class 2 procedure (no response expected).

use crate::codec::generated::*;
use crate::codec::{decode_ngap_pdu, encode_ngap_pdu, NgapCodecError};
use crate::procedures::ng_setup::{
    MiscCause, NasCause, NgSetupFailureCause, ProtocolCause, RadioNetworkCause, TransportCause,
};
use thiserror::Error;

/// Errors that can occur during Error Indication procedures
#[derive(Debug, Error)]
pub enum ErrorIndicationError {
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

    /// Invalid IE value
    #[error("Invalid IE value: {0}")]
    InvalidIeValue(String),
}

/// Criticality Diagnostics information
#[derive(Debug, Clone, Default)]
pub struct CriticalityDiagnosticsInfo {
    /// Procedure code that caused the error (optional)
    pub procedure_code: Option<u8>,
    /// Triggering message type (optional)
    pub triggering_message: Option<TriggeringMessageValue>,
    /// Procedure criticality (optional)
    pub procedure_criticality: Option<CriticalityValue>,
    /// List of IEs that caused the error (optional)
    pub ies_criticality_diagnostics: Vec<IeCriticalityDiagnosticsItem>,
}

/// Triggering message type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggeringMessageValue {
    /// Initiating message
    InitiatingMessage,
    /// Successful outcome
    SuccessfulOutcome,
    /// Unsuccessful outcome
    UnsuccessfulOutcome,
}

impl From<TriggeringMessageValue> for TriggeringMessage {
    fn from(value: TriggeringMessageValue) -> Self {
        match value {
            TriggeringMessageValue::InitiatingMessage => {
                TriggeringMessage(TriggeringMessage::INITIATING_MESSAGE)
            }
            TriggeringMessageValue::SuccessfulOutcome => {
                TriggeringMessage(TriggeringMessage::SUCCESSFUL_OUTCOME)
            }
            TriggeringMessageValue::UnsuccessfulOutcome => {
                TriggeringMessage(TriggeringMessage::UNSUCCESSFUL_OUTCOME)
            }
        }
    }
}

impl TryFrom<TriggeringMessage> for TriggeringMessageValue {
    type Error = ErrorIndicationError;

    fn try_from(value: TriggeringMessage) -> Result<Self, Self::Error> {
        match value.0 {
            TriggeringMessage::INITIATING_MESSAGE => Ok(TriggeringMessageValue::InitiatingMessage),
            TriggeringMessage::SUCCESSFUL_OUTCOME => Ok(TriggeringMessageValue::SuccessfulOutcome),
            TriggeringMessage::UNSUCCESSFUL_OUTCOME => {
                Ok(TriggeringMessageValue::UnsuccessfulOutcome)
            }
            _ => Err(ErrorIndicationError::InvalidIeValue(format!(
                "Unknown TriggeringMessage value: {}",
                value.0
            ))),
        }
    }
}

/// Criticality value
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CriticalityValue {
    /// Reject - message shall be rejected
    Reject,
    /// Ignore - IE shall be ignored
    Ignore,
    /// Notify - IE shall be ignored and notified
    Notify,
}

impl From<CriticalityValue> for Criticality {
    fn from(value: CriticalityValue) -> Self {
        match value {
            CriticalityValue::Reject => Criticality(Criticality::REJECT),
            CriticalityValue::Ignore => Criticality(Criticality::IGNORE),
            CriticalityValue::Notify => Criticality(Criticality::NOTIFY),
        }
    }
}

impl TryFrom<Criticality> for CriticalityValue {
    type Error = ErrorIndicationError;

    fn try_from(value: Criticality) -> Result<Self, Self::Error> {
        match value.0 {
            Criticality::REJECT => Ok(CriticalityValue::Reject),
            Criticality::IGNORE => Ok(CriticalityValue::Ignore),
            Criticality::NOTIFY => Ok(CriticalityValue::Notify),
            _ => Err(ErrorIndicationError::InvalidIeValue(format!(
                "Unknown Criticality value: {}",
                value.0
            ))),
        }
    }
}

/// Type of IE error
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeOfErrorValue {
    /// IE was not understood
    NotUnderstood,
    /// IE was missing
    Missing,
}

impl From<TypeOfErrorValue> for TypeOfError {
    fn from(value: TypeOfErrorValue) -> Self {
        match value {
            TypeOfErrorValue::NotUnderstood => TypeOfError(TypeOfError::NOT_UNDERSTOOD),
            TypeOfErrorValue::Missing => TypeOfError(TypeOfError::MISSING),
        }
    }
}

impl TryFrom<TypeOfError> for TypeOfErrorValue {
    type Error = ErrorIndicationError;

    fn try_from(value: TypeOfError) -> Result<Self, Self::Error> {
        match value.0 {
            TypeOfError::NOT_UNDERSTOOD => Ok(TypeOfErrorValue::NotUnderstood),
            TypeOfError::MISSING => Ok(TypeOfErrorValue::Missing),
            _ => Err(ErrorIndicationError::InvalidIeValue(format!(
                "Unknown TypeOfError value: {}",
                value.0
            ))),
        }
    }
}

/// IE Criticality Diagnostics item
#[derive(Debug, Clone)]
pub struct IeCriticalityDiagnosticsItem {
    /// IE criticality
    pub ie_criticality: CriticalityValue,
    /// IE ID
    pub ie_id: u16,
    /// Type of error
    pub type_of_error: TypeOfErrorValue,
}

/// Parameters for building an Error Indication message
#[derive(Debug, Clone, Default)]
pub struct ErrorIndicationParams {
    /// AMF UE NGAP ID (optional)
    pub amf_ue_ngap_id: Option<u64>,
    /// RAN UE NGAP ID (optional)
    pub ran_ue_ngap_id: Option<u32>,
    /// Cause of the error (optional)
    pub cause: Option<NgSetupFailureCause>,
    /// Criticality diagnostics (optional)
    pub criticality_diagnostics: Option<CriticalityDiagnosticsInfo>,
}

/// Parsed Error Indication message data
#[derive(Debug, Clone, Default)]
pub struct ErrorIndicationData {
    /// AMF UE NGAP ID (optional)
    pub amf_ue_ngap_id: Option<u64>,
    /// RAN UE NGAP ID (optional)
    pub ran_ue_ngap_id: Option<u32>,
    /// Cause of the error (optional)
    pub cause: Option<NgSetupFailureCause>,
    /// Criticality diagnostics (optional)
    pub criticality_diagnostics: Option<CriticalityDiagnosticsInfo>,
}

// ============================================================================
// Error Indication Message Builder
// ============================================================================

/// Build an Error Indication PDU
///
/// # Arguments
/// * `params` - Parameters for the Error Indication message
///
/// # Returns
/// * `Ok(NGAP_PDU)` - The constructed PDU
/// * `Err(ErrorIndicationError)` - If construction fails
pub fn build_error_indication(params: &ErrorIndicationParams) -> Result<NGAP_PDU, ErrorIndicationError> {
    let mut protocol_ies = Vec::new();

    // IE: AMF-UE-NGAP-ID (optional)
    if let Some(amf_ue_ngap_id) = params.amf_ue_ngap_id {
        protocol_ies.push(ErrorIndicationProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_AMF_UE_NGAP_ID),
            criticality: Criticality(Criticality::IGNORE),
            value: ErrorIndicationProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(AMF_UE_NGAP_ID(
                amf_ue_ngap_id,
            )),
        });
    }

    // IE: RAN-UE-NGAP-ID (optional)
    if let Some(ran_ue_ngap_id) = params.ran_ue_ngap_id {
        protocol_ies.push(ErrorIndicationProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_RAN_UE_NGAP_ID),
            criticality: Criticality(Criticality::IGNORE),
            value: ErrorIndicationProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(RAN_UE_NGAP_ID(
                ran_ue_ngap_id,
            )),
        });
    }

    // IE: Cause (optional)
    if let Some(ref cause) = params.cause {
        let cause_ie = build_cause(cause);
        protocol_ies.push(ErrorIndicationProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_CAUSE),
            criticality: Criticality(Criticality::IGNORE),
            value: ErrorIndicationProtocolIEs_EntryValue::Id_Cause(cause_ie),
        });
    }

    // IE: CriticalityDiagnostics (optional)
    if let Some(ref diag) = params.criticality_diagnostics {
        let diag_ie = build_criticality_diagnostics(diag);
        protocol_ies.push(ErrorIndicationProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_CRITICALITY_DIAGNOSTICS),
            criticality: Criticality(Criticality::IGNORE),
            value: ErrorIndicationProtocolIEs_EntryValue::Id_CriticalityDiagnostics(diag_ie),
        });
    }

    let error_indication = ErrorIndication {
        protocol_i_es: ErrorIndicationProtocolIEs(protocol_ies),
    };

    let initiating_message = InitiatingMessage {
        procedure_code: ProcedureCode(ID_ERROR_INDICATION),
        criticality: Criticality(Criticality::IGNORE),
        value: InitiatingMessageValue::Id_ErrorIndication(error_indication),
    };

    Ok(NGAP_PDU::InitiatingMessage(initiating_message))
}

fn build_cause(cause: &NgSetupFailureCause) -> Cause {
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
        RadioNetworkCause::UnknownMappedUeNgapId => CauseRadioNetwork::UNSPECIFIED, // Fallback
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

fn build_criticality_diagnostics(diag: &CriticalityDiagnosticsInfo) -> CriticalityDiagnostics {
    let procedure_code = diag.procedure_code.map(ProcedureCode);
    let triggering_message = diag.triggering_message.map(std::convert::Into::into);
    let procedure_criticality = diag.procedure_criticality.map(std::convert::Into::into);

    let ies_criticality_diagnostics = if diag.ies_criticality_diagnostics.is_empty() {
        None
    } else {
        let items: Vec<CriticalityDiagnostics_IE_Item> = diag
            .ies_criticality_diagnostics
            .iter()
            .map(|item| CriticalityDiagnostics_IE_Item {
                ie_criticality: item.ie_criticality.into(),
                ie_id: ProtocolIE_ID(item.ie_id),
                type_of_error: item.type_of_error.into(),
                ie_extensions: None,
            })
            .collect();
        Some(CriticalityDiagnostics_IE_List(items))
    };

    CriticalityDiagnostics {
        procedure_code,
        triggering_message,
        procedure_criticality,
        i_es_criticality_diagnostics: ies_criticality_diagnostics,
        ie_extensions: None,
    }
}

// ============================================================================
// Error Indication Message Parser
// ============================================================================

/// Parse an Error Indication message from an NGAP PDU
///
/// # Arguments
/// * `pdu` - The NGAP PDU to parse
///
/// # Returns
/// * `Ok(ErrorIndicationData)` - The parsed message data
/// * `Err(ErrorIndicationError)` - If parsing fails
pub fn parse_error_indication(pdu: &NGAP_PDU) -> Result<ErrorIndicationData, ErrorIndicationError> {
    let initiating_message = match pdu {
        NGAP_PDU::InitiatingMessage(msg) => msg,
        _ => {
            return Err(ErrorIndicationError::InvalidMessageType {
                expected: "InitiatingMessage".to_string(),
                actual: format!("{pdu:?}"),
            })
        }
    };

    let error_indication = match &initiating_message.value {
        InitiatingMessageValue::Id_ErrorIndication(msg) => msg,
        _ => {
            return Err(ErrorIndicationError::InvalidMessageType {
                expected: "ErrorIndication".to_string(),
                actual: format!("{:?}", initiating_message.value),
            })
        }
    };

    let mut amf_ue_ngap_id: Option<u64> = None;
    let mut ran_ue_ngap_id: Option<u32> = None;
    let mut cause: Option<NgSetupFailureCause> = None;
    let mut criticality_diagnostics: Option<CriticalityDiagnosticsInfo> = None;

    for ie in &error_indication.protocol_i_es.0 {
        match &ie.value {
            ErrorIndicationProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(id) => {
                amf_ue_ngap_id = Some(id.0);
            }
            ErrorIndicationProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(id) => {
                ran_ue_ngap_id = Some(id.0);
            }
            ErrorIndicationProtocolIEs_EntryValue::Id_Cause(c) => {
                cause = Some(parse_cause(c));
            }
            ErrorIndicationProtocolIEs_EntryValue::Id_CriticalityDiagnostics(diag) => {
                criticality_diagnostics = Some(parse_criticality_diagnostics(diag)?);
            }
            _ => {
                // Ignore other IEs
            }
        }
    }

    Ok(ErrorIndicationData {
        amf_ue_ngap_id,
        ran_ue_ngap_id,
        cause,
        criticality_diagnostics,
    })
}

fn parse_cause(cause: &Cause) -> NgSetupFailureCause {
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

fn parse_criticality_diagnostics(
    diag: &CriticalityDiagnostics,
) -> Result<CriticalityDiagnosticsInfo, ErrorIndicationError> {
    let procedure_code = diag.procedure_code.as_ref().map(|pc| pc.0);
    let triggering_message = diag
        .triggering_message
        .as_ref()
        .map(|tm| tm.clone().try_into())
        .transpose()?;
    let procedure_criticality = diag
        .procedure_criticality
        .as_ref()
        .map(|c| c.clone().try_into())
        .transpose()?;

    let ies_criticality_diagnostics = diag
        .i_es_criticality_diagnostics
        .as_ref()
        .map(|list| {
            list.0
                .iter()
                .map(|item| {
                    Ok(IeCriticalityDiagnosticsItem {
                        ie_criticality: item.ie_criticality.clone().try_into()?,
                        ie_id: item.ie_id.0,
                        type_of_error: item.type_of_error.clone().try_into()?,
                    })
                })
                .collect::<Result<Vec<_>, ErrorIndicationError>>()
        })
        .transpose()?
        .unwrap_or_default();

    Ok(CriticalityDiagnosticsInfo {
        procedure_code,
        triggering_message,
        procedure_criticality,
        ies_criticality_diagnostics,
    })
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Build and encode an Error Indication message to bytes
///
/// # Arguments
/// * `params` - Parameters for the Error Indication message
///
/// # Returns
/// * `Ok(Vec<u8>)` - The encoded bytes
/// * `Err(ErrorIndicationError)` - If building or encoding fails
pub fn encode_error_indication(params: &ErrorIndicationParams) -> Result<Vec<u8>, ErrorIndicationError> {
    let pdu = build_error_indication(params)?;
    Ok(encode_ngap_pdu(&pdu)?)
}

/// Decode and parse an Error Indication message from bytes
///
/// # Arguments
/// * `bytes` - The encoded bytes
///
/// # Returns
/// * `Ok(ErrorIndicationData)` - The parsed message data
/// * `Err(ErrorIndicationError)` - If decoding or parsing fails
pub fn decode_error_indication(bytes: &[u8]) -> Result<ErrorIndicationData, ErrorIndicationError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_error_indication(&pdu)
}

/// Check if an NGAP PDU is an Error Indication message
pub fn is_error_indication(pdu: &NGAP_PDU) -> bool {
    matches!(
        pdu,
        NGAP_PDU::InitiatingMessage(msg)
            if matches!(msg.value, InitiatingMessageValue::Id_ErrorIndication(_))
    )
}

// ============================================================================
// Helper Functions for Common Error Indication Scenarios
// ============================================================================

/// Create an Error Indication for an unknown local UE NGAP ID
pub fn error_indication_unknown_local_ue_ngap_id(
    amf_ue_ngap_id: Option<u64>,
    ran_ue_ngap_id: Option<u32>,
) -> ErrorIndicationParams {
    ErrorIndicationParams {
        amf_ue_ngap_id,
        ran_ue_ngap_id,
        cause: Some(NgSetupFailureCause::RadioNetwork(
            RadioNetworkCause::UnknownLocalUeNgapId,
        )),
        criticality_diagnostics: None,
    }
}

/// Create an Error Indication for a protocol error (transfer syntax error)
pub fn error_indication_transfer_syntax_error(
    amf_ue_ngap_id: Option<u64>,
    ran_ue_ngap_id: Option<u32>,
) -> ErrorIndicationParams {
    ErrorIndicationParams {
        amf_ue_ngap_id,
        ran_ue_ngap_id,
        cause: Some(NgSetupFailureCause::Protocol(
            ProtocolCause::TransferSyntaxError,
        )),
        criticality_diagnostics: None,
    }
}

/// Create an Error Indication for a semantic error
pub fn error_indication_semantic_error(
    amf_ue_ngap_id: Option<u64>,
    ran_ue_ngap_id: Option<u32>,
) -> ErrorIndicationParams {
    ErrorIndicationParams {
        amf_ue_ngap_id,
        ran_ue_ngap_id,
        cause: Some(NgSetupFailureCause::Protocol(ProtocolCause::SemanticError)),
        criticality_diagnostics: None,
    }
}

/// Create an Error Indication for an abstract syntax error
pub fn error_indication_abstract_syntax_error(
    amf_ue_ngap_id: Option<u64>,
    ran_ue_ngap_id: Option<u32>,
    reject: bool,
) -> ErrorIndicationParams {
    let cause = if reject {
        NgSetupFailureCause::Protocol(ProtocolCause::AbstractSyntaxErrorReject)
    } else {
        NgSetupFailureCause::Protocol(ProtocolCause::AbstractSyntaxErrorIgnoreAndNotify)
    };
    ErrorIndicationParams {
        amf_ue_ngap_id,
        ran_ue_ngap_id,
        cause: Some(cause),
        criticality_diagnostics: None,
    }
}

/// Create an Error Indication with criticality diagnostics for a missing mandatory IE
pub fn error_indication_missing_ie(
    amf_ue_ngap_id: Option<u64>,
    ran_ue_ngap_id: Option<u32>,
    procedure_code: u8,
    ie_id: u16,
) -> ErrorIndicationParams {
    ErrorIndicationParams {
        amf_ue_ngap_id,
        ran_ue_ngap_id,
        cause: Some(NgSetupFailureCause::Protocol(
            ProtocolCause::AbstractSyntaxErrorReject,
        )),
        criticality_diagnostics: Some(CriticalityDiagnosticsInfo {
            procedure_code: Some(procedure_code),
            triggering_message: Some(TriggeringMessageValue::InitiatingMessage),
            procedure_criticality: Some(CriticalityValue::Reject),
            ies_criticality_diagnostics: vec![IeCriticalityDiagnosticsItem {
                ie_criticality: CriticalityValue::Reject,
                ie_id,
                type_of_error: TypeOfErrorValue::Missing,
            }],
        }),
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_error_indication_empty() {
        let params = ErrorIndicationParams::default();
        let result = build_error_indication(&params);
        assert!(result.is_ok());

        let pdu = result.unwrap();
        match pdu {
            NGAP_PDU::InitiatingMessage(msg) => {
                assert_eq!(msg.procedure_code.0, ID_ERROR_INDICATION);
            }
            _ => panic!("Expected InitiatingMessage"),
        }
    }

    #[test]
    fn test_build_error_indication_with_ue_ids() {
        let params = ErrorIndicationParams {
            amf_ue_ngap_id: Some(12345),
            ran_ue_ngap_id: Some(67890),
            cause: None,
            criticality_diagnostics: None,
        };
        let result = build_error_indication(&params);
        assert!(result.is_ok());

        let pdu = result.unwrap();
        assert!(is_error_indication(&pdu));
    }

    #[test]
    fn test_build_error_indication_with_cause() {
        let params = ErrorIndicationParams {
            amf_ue_ngap_id: Some(12345),
            ran_ue_ngap_id: Some(67890),
            cause: Some(NgSetupFailureCause::Protocol(
                ProtocolCause::TransferSyntaxError,
            )),
            criticality_diagnostics: None,
        };
        let result = build_error_indication(&params);
        assert!(result.is_ok());
    }

    #[test]
    fn test_encode_error_indication() {
        let params = ErrorIndicationParams {
            amf_ue_ngap_id: Some(12345),
            ran_ue_ngap_id: Some(67890),
            cause: Some(NgSetupFailureCause::RadioNetwork(
                RadioNetworkCause::UnknownLocalUeNgapId,
            )),
            criticality_diagnostics: None,
        };
        let result = encode_error_indication(&params);
        assert!(result.is_ok());

        let bytes = result.unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_error_indication_roundtrip() {
        let params = ErrorIndicationParams {
            amf_ue_ngap_id: Some(12345),
            ran_ue_ngap_id: Some(67890),
            cause: Some(NgSetupFailureCause::Protocol(
                ProtocolCause::SemanticError,
            )),
            criticality_diagnostics: None,
        };

        // Build and encode
        let pdu = build_error_indication(&params).expect("Failed to build message");
        let encoded = encode_ngap_pdu(&pdu).expect("Failed to encode");

        // Decode
        let decoded_pdu = decode_ngap_pdu(&encoded).expect("Failed to decode");

        // Verify structure
        match decoded_pdu {
            NGAP_PDU::InitiatingMessage(msg) => {
                assert_eq!(msg.procedure_code.0, ID_ERROR_INDICATION);
                match msg.value {
                    InitiatingMessageValue::Id_ErrorIndication(error_ind) => {
                        // Verify we have the expected IEs
                        assert!(error_ind.protocol_i_es.0.len() >= 2);
                    }
                    _ => panic!("Expected ErrorIndication"),
                }
            }
            _ => panic!("Expected InitiatingMessage"),
        }
    }

    #[test]
    fn test_error_indication_parse_roundtrip() {
        let params = ErrorIndicationParams {
            amf_ue_ngap_id: Some(12345),
            ran_ue_ngap_id: Some(67890),
            cause: Some(NgSetupFailureCause::RadioNetwork(
                RadioNetworkCause::UnknownLocalUeNgapId,
            )),
            criticality_diagnostics: None,
        };

        // Build, encode, decode, and parse
        let encoded = encode_error_indication(&params).expect("Failed to encode");
        let parsed = decode_error_indication(&encoded).expect("Failed to decode and parse");

        // Verify parsed data matches original params
        assert_eq!(parsed.amf_ue_ngap_id, params.amf_ue_ngap_id);
        assert_eq!(parsed.ran_ue_ngap_id, params.ran_ue_ngap_id);
        assert!(parsed.cause.is_some());
        match parsed.cause.unwrap() {
            NgSetupFailureCause::RadioNetwork(cause) => {
                assert_eq!(cause, RadioNetworkCause::UnknownLocalUeNgapId);
            }
            _ => panic!("Expected RadioNetwork cause"),
        }
    }

    #[test]
    fn test_error_indication_with_criticality_diagnostics() {
        let params = ErrorIndicationParams {
            amf_ue_ngap_id: Some(12345),
            ran_ue_ngap_id: None,
            cause: Some(NgSetupFailureCause::Protocol(
                ProtocolCause::AbstractSyntaxErrorReject,
            )),
            criticality_diagnostics: Some(CriticalityDiagnosticsInfo {
                procedure_code: Some(21), // NG Setup
                triggering_message: Some(TriggeringMessageValue::InitiatingMessage),
                procedure_criticality: Some(CriticalityValue::Reject),
                ies_criticality_diagnostics: vec![IeCriticalityDiagnosticsItem {
                    ie_criticality: CriticalityValue::Reject,
                    ie_id: 27, // GlobalRANNodeID
                    type_of_error: TypeOfErrorValue::Missing,
                }],
            }),
        };

        let encoded = encode_error_indication(&params).expect("Failed to encode");
        let parsed = decode_error_indication(&encoded).expect("Failed to decode and parse");

        assert!(parsed.criticality_diagnostics.is_some());
        let diag = parsed.criticality_diagnostics.unwrap();
        assert_eq!(diag.procedure_code, Some(21));
        assert_eq!(
            diag.triggering_message,
            Some(TriggeringMessageValue::InitiatingMessage)
        );
        assert_eq!(diag.procedure_criticality, Some(CriticalityValue::Reject));
        assert_eq!(diag.ies_criticality_diagnostics.len(), 1);
        assert_eq!(diag.ies_criticality_diagnostics[0].ie_id, 27);
        assert_eq!(
            diag.ies_criticality_diagnostics[0].type_of_error,
            TypeOfErrorValue::Missing
        );
    }

    #[test]
    fn test_helper_unknown_local_ue_ngap_id() {
        let params = error_indication_unknown_local_ue_ngap_id(Some(12345), Some(67890));
        let encoded = encode_error_indication(&params).expect("Failed to encode");
        let parsed = decode_error_indication(&encoded).expect("Failed to decode and parse");

        assert_eq!(parsed.amf_ue_ngap_id, Some(12345));
        assert_eq!(parsed.ran_ue_ngap_id, Some(67890));
        match parsed.cause.unwrap() {
            NgSetupFailureCause::RadioNetwork(cause) => {
                assert_eq!(cause, RadioNetworkCause::UnknownLocalUeNgapId);
            }
            _ => panic!("Expected RadioNetwork cause"),
        }
    }

    #[test]
    fn test_helper_transfer_syntax_error() {
        let params = error_indication_transfer_syntax_error(None, None);
        let encoded = encode_error_indication(&params).expect("Failed to encode");
        let parsed = decode_error_indication(&encoded).expect("Failed to decode and parse");

        match parsed.cause.unwrap() {
            NgSetupFailureCause::Protocol(cause) => {
                assert_eq!(cause, ProtocolCause::TransferSyntaxError);
            }
            _ => panic!("Expected Protocol cause"),
        }
    }

    #[test]
    fn test_helper_semantic_error() {
        let params = error_indication_semantic_error(Some(100), None);
        let encoded = encode_error_indication(&params).expect("Failed to encode");
        let parsed = decode_error_indication(&encoded).expect("Failed to decode and parse");

        assert_eq!(parsed.amf_ue_ngap_id, Some(100));
        match parsed.cause.unwrap() {
            NgSetupFailureCause::Protocol(cause) => {
                assert_eq!(cause, ProtocolCause::SemanticError);
            }
            _ => panic!("Expected Protocol cause"),
        }
    }

    #[test]
    fn test_helper_missing_ie() {
        let params = error_indication_missing_ie(Some(12345), Some(67890), 21, 27);
        let encoded = encode_error_indication(&params).expect("Failed to encode");
        let parsed = decode_error_indication(&encoded).expect("Failed to decode and parse");

        assert!(parsed.criticality_diagnostics.is_some());
        let diag = parsed.criticality_diagnostics.unwrap();
        assert_eq!(diag.procedure_code, Some(21));
        assert_eq!(diag.ies_criticality_diagnostics.len(), 1);
        assert_eq!(diag.ies_criticality_diagnostics[0].ie_id, 27);
    }

    #[test]
    fn test_is_error_indication() {
        let params = ErrorIndicationParams::default();
        let pdu = build_error_indication(&params).expect("Failed to build message");
        assert!(is_error_indication(&pdu));
    }

    #[test]
    fn test_all_cause_types() {
        // Test all cause types can be encoded and decoded
        let causes = vec![
            NgSetupFailureCause::RadioNetwork(RadioNetworkCause::Unspecified),
            NgSetupFailureCause::Transport(TransportCause::Unspecified),
            NgSetupFailureCause::Nas(NasCause::Unspecified),
            NgSetupFailureCause::Protocol(ProtocolCause::Unspecified),
            NgSetupFailureCause::Misc(MiscCause::Unspecified),
        ];

        for cause in causes {
            let params = ErrorIndicationParams {
                amf_ue_ngap_id: Some(1),
                ran_ue_ngap_id: Some(1),
                cause: Some(cause.clone()),
                criticality_diagnostics: None,
            };
            let encoded = encode_error_indication(&params).expect("Failed to encode");
            let parsed = decode_error_indication(&encoded).expect("Failed to decode");
            assert!(parsed.cause.is_some());
        }
    }
}
