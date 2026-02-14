//! Handover Preparation Failure Procedure
//!
//! Implements a standalone Handover Preparation Failure procedure as defined in
//! 3GPP TS 38.413 Section 8.4.1.
//! Sent by the target AMF to the source AMF when handover preparation fails.
//! Contains Cause IE and optional `CriticalityDiagnostics`.

use crate::codec::generated::*;
use crate::codec::{decode_ngap_pdu, encode_ngap_pdu, NgapCodecError};
use crate::procedures::error_indication::{
    CriticalityDiagnosticsInfo, CriticalityValue, IeCriticalityDiagnosticsItem,
    TriggeringMessageValue, TypeOfErrorValue,
};
use crate::procedures::ng_setup::{
    MiscCause, NasCause, NgSetupFailureCause, ProtocolCause, RadioNetworkCause, TransportCause,
};
use thiserror::Error;

/// Errors that can occur during Handover Preparation Failure procedures
#[derive(Debug, Error)]
pub enum HandoverPreparationFailureError {
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

/// Parameters for building a Handover Preparation Failure message
#[derive(Debug, Clone)]
pub struct HandoverPreparationFailureParams {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
    /// Cause of failure
    pub cause: NgSetupFailureCause,
    /// Criticality diagnostics (optional)
    pub criticality_diagnostics: Option<CriticalityDiagnosticsInfo>,
}

/// Parsed Handover Preparation Failure data
#[derive(Debug, Clone)]
pub struct HandoverPreparationFailureFullData {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
    /// Cause of failure
    pub cause: NgSetupFailureCause,
    /// Criticality diagnostics (optional)
    pub criticality_diagnostics: Option<CriticalityDiagnosticsInfo>,
}

// ============================================================================
// Handover Preparation Failure Builder
// ============================================================================

/// Build a Handover Preparation Failure PDU
pub fn build_handover_preparation_failure(
    params: &HandoverPreparationFailureParams,
) -> Result<NGAP_PDU, HandoverPreparationFailureError> {
    let mut protocol_ies = Vec::new();

    // IE: AMF-UE-NGAP-ID (mandatory)
    protocol_ies.push(HandoverPreparationFailureProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_AMF_UE_NGAP_ID),
        criticality: Criticality(Criticality::IGNORE),
        value: HandoverPreparationFailureProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(
            AMF_UE_NGAP_ID(params.amf_ue_ngap_id),
        ),
    });

    // IE: RAN-UE-NGAP-ID (mandatory)
    protocol_ies.push(HandoverPreparationFailureProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_RAN_UE_NGAP_ID),
        criticality: Criticality(Criticality::IGNORE),
        value: HandoverPreparationFailureProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(
            RAN_UE_NGAP_ID(params.ran_ue_ngap_id),
        ),
    });

    // IE: Cause (mandatory)
    let cause = build_cause(&params.cause);
    protocol_ies.push(HandoverPreparationFailureProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_CAUSE),
        criticality: Criticality(Criticality::IGNORE),
        value: HandoverPreparationFailureProtocolIEs_EntryValue::Id_Cause(cause),
    });

    // IE: CriticalityDiagnostics (optional)
    if let Some(ref diag) = params.criticality_diagnostics {
        let diag_ie = build_criticality_diagnostics(diag);
        protocol_ies.push(HandoverPreparationFailureProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_CRITICALITY_DIAGNOSTICS),
            criticality: Criticality(Criticality::IGNORE),
            value: HandoverPreparationFailureProtocolIEs_EntryValue::Id_CriticalityDiagnostics(
                diag_ie,
            ),
        });
    }

    let handover_preparation_failure = HandoverPreparationFailure {
        protocol_i_es: HandoverPreparationFailureProtocolIEs(protocol_ies),
    };

    let unsuccessful_outcome = UnsuccessfulOutcome {
        procedure_code: ProcedureCode(ID_HANDOVER_PREPARATION),
        criticality: Criticality(Criticality::REJECT),
        value: UnsuccessfulOutcomeValue::Id_HandoverPreparation(handover_preparation_failure),
    };

    Ok(NGAP_PDU::UnsuccessfulOutcome(unsuccessful_outcome))
}

/// Parse a Handover Preparation Failure from an NGAP PDU
pub fn parse_handover_preparation_failure_full(
    pdu: &NGAP_PDU,
) -> Result<HandoverPreparationFailureFullData, HandoverPreparationFailureError> {
    let unsuccessful_outcome = match pdu {
        NGAP_PDU::UnsuccessfulOutcome(outcome) => outcome,
        _ => {
            return Err(HandoverPreparationFailureError::InvalidMessageType {
                expected: "UnsuccessfulOutcome".to_string(),
                actual: format!("{pdu:?}"),
            })
        }
    };

    let failure = match &unsuccessful_outcome.value {
        UnsuccessfulOutcomeValue::Id_HandoverPreparation(fail) => fail,
        _ => {
            return Err(HandoverPreparationFailureError::InvalidMessageType {
                expected: "HandoverPreparationFailure".to_string(),
                actual: format!("{:?}", unsuccessful_outcome.value),
            })
        }
    };

    let mut amf_ue_ngap_id: Option<u64> = None;
    let mut ran_ue_ngap_id: Option<u32> = None;
    let mut cause: Option<NgSetupFailureCause> = None;
    let mut criticality_diagnostics: Option<CriticalityDiagnosticsInfo> = None;

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
            HandoverPreparationFailureProtocolIEs_EntryValue::Id_CriticalityDiagnostics(diag) => {
                criticality_diagnostics = Some(parse_criticality_diagnostics(diag));
            }
            _ => {}
        }
    }

    Ok(HandoverPreparationFailureFullData {
        amf_ue_ngap_id: amf_ue_ngap_id.ok_or_else(|| {
            HandoverPreparationFailureError::MissingMandatoryIe("AMF-UE-NGAP-ID".to_string())
        })?,
        ran_ue_ngap_id: ran_ue_ngap_id.ok_or_else(|| {
            HandoverPreparationFailureError::MissingMandatoryIe("RAN-UE-NGAP-ID".to_string())
        })?,
        cause: cause.ok_or_else(|| {
            HandoverPreparationFailureError::MissingMandatoryIe("Cause".to_string())
        })?,
        criticality_diagnostics,
    })
}

// ============================================================================
// Helper Functions
// ============================================================================

fn build_cause(cause: &NgSetupFailureCause) -> Cause {
    match cause {
        NgSetupFailureCause::RadioNetwork(rn) => {
            Cause::RadioNetwork(build_radio_network_cause(rn))
        }
        NgSetupFailureCause::Transport(t) => Cause::Transport(build_transport_cause(t)),
        NgSetupFailureCause::Nas(n) => Cause::Nas(build_nas_cause(n)),
        NgSetupFailureCause::Protocol(p) => Cause::Protocol(build_protocol_cause(p)),
        NgSetupFailureCause::Misc(m) => Cause::Misc(build_misc_cause(m)),
    }
}

fn build_radio_network_cause(cause: &RadioNetworkCause) -> CauseRadioNetwork {
    let value = match cause {
        RadioNetworkCause::Unspecified => CauseRadioNetwork::UNSPECIFIED,
        RadioNetworkCause::HoTargetNotAllowed => CauseRadioNetwork::HO_TARGET_NOT_ALLOWED,
        RadioNetworkCause::NoRadioResourcesAvailableInTargetCell => {
            CauseRadioNetwork::NO_RADIO_RESOURCES_AVAILABLE_IN_TARGET_CELL
        }
        RadioNetworkCause::UnknownTargetId => CauseRadioNetwork::UNKNOWN_TARGET_ID,
        RadioNetworkCause::Other(v) => *v,
        _ => CauseRadioNetwork::UNSPECIFIED,
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
        ProtocolCause::SemanticError => CauseProtocol::SEMANTIC_ERROR,
        ProtocolCause::Unspecified => CauseProtocol::UNSPECIFIED,
        ProtocolCause::Other(v) => *v,
        _ => CauseProtocol::UNSPECIFIED,
    };
    CauseProtocol(value)
}

fn build_misc_cause(cause: &MiscCause) -> CauseMisc {
    let value = match cause {
        MiscCause::ControlProcessingOverload => CauseMisc::CONTROL_PROCESSING_OVERLOAD,
        MiscCause::HardwareFailure => CauseMisc::HARDWARE_FAILURE,
        MiscCause::OmIntervention => CauseMisc::OM_INTERVENTION,
        MiscCause::Unspecified => CauseMisc::UNSPECIFIED,
        MiscCause::Other(v) => *v,
        _ => CauseMisc::UNSPECIFIED,
    };
    CauseMisc(value)
}

fn parse_cause(cause: &Cause) -> NgSetupFailureCause {
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
        CauseRadioNetwork::HO_TARGET_NOT_ALLOWED => RadioNetworkCause::HoTargetNotAllowed,
        CauseRadioNetwork::UNKNOWN_TARGET_ID => RadioNetworkCause::UnknownTargetId,
        CauseRadioNetwork::NO_RADIO_RESOURCES_AVAILABLE_IN_TARGET_CELL => {
            RadioNetworkCause::NoRadioResourcesAvailableInTargetCell
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
        CauseProtocol::SEMANTIC_ERROR => ProtocolCause::SemanticError,
        CauseProtocol::UNSPECIFIED => ProtocolCause::Unspecified,
        other => ProtocolCause::Other(other),
    }
}

fn parse_misc_cause(cause: &CauseMisc) -> MiscCause {
    match cause.0 {
        CauseMisc::CONTROL_PROCESSING_OVERLOAD => MiscCause::ControlProcessingOverload,
        CauseMisc::HARDWARE_FAILURE => MiscCause::HardwareFailure,
        CauseMisc::OM_INTERVENTION => MiscCause::OmIntervention,
        CauseMisc::UNSPECIFIED => MiscCause::Unspecified,
        other => MiscCause::Other(other),
    }
}

fn build_criticality_diagnostics(diag: &CriticalityDiagnosticsInfo) -> CriticalityDiagnostics {
    let procedure_code = diag.procedure_code.map(ProcedureCode);
    let triggering_message = diag.triggering_message.map(|tm| match tm {
        TriggeringMessageValue::InitiatingMessage => {
            TriggeringMessage(TriggeringMessage::INITIATING_MESSAGE)
        }
        TriggeringMessageValue::SuccessfulOutcome => {
            TriggeringMessage(TriggeringMessage::SUCCESSFUL_OUTCOME)
        }
        TriggeringMessageValue::UnsuccessfulOutcome => {
            TriggeringMessage(TriggeringMessage::UNSUCCESSFUL_OUTCOME)
        }
    });
    let procedure_criticality = diag.procedure_criticality.map(|c| match c {
        CriticalityValue::Reject => Criticality(Criticality::REJECT),
        CriticalityValue::Ignore => Criticality(Criticality::IGNORE),
        CriticalityValue::Notify => Criticality(Criticality::NOTIFY),
    });

    let ies_criticality_diagnostics = if diag.ies_criticality_diagnostics.is_empty() {
        None
    } else {
        let items: Vec<CriticalityDiagnostics_IE_Item> = diag
            .ies_criticality_diagnostics
            .iter()
            .map(|item| {
                let ie_criticality = match item.ie_criticality {
                    CriticalityValue::Reject => Criticality(Criticality::REJECT),
                    CriticalityValue::Ignore => Criticality(Criticality::IGNORE),
                    CriticalityValue::Notify => Criticality(Criticality::NOTIFY),
                };
                let type_of_error = match item.type_of_error {
                    TypeOfErrorValue::NotUnderstood => TypeOfError(TypeOfError::NOT_UNDERSTOOD),
                    TypeOfErrorValue::Missing => TypeOfError(TypeOfError::MISSING),
                };
                CriticalityDiagnostics_IE_Item {
                    ie_criticality,
                    ie_id: ProtocolIE_ID(item.ie_id),
                    type_of_error,
                    ie_extensions: None,
                }
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

fn parse_criticality_diagnostics(
    diag: &CriticalityDiagnostics,
) -> CriticalityDiagnosticsInfo {
    let procedure_code = diag.procedure_code.as_ref().map(|pc| pc.0);
    let triggering_message = diag.triggering_message.as_ref().map(|tm| match tm.0 {
        TriggeringMessage::INITIATING_MESSAGE => TriggeringMessageValue::InitiatingMessage,
        TriggeringMessage::SUCCESSFUL_OUTCOME => TriggeringMessageValue::SuccessfulOutcome,
        TriggeringMessage::UNSUCCESSFUL_OUTCOME => TriggeringMessageValue::UnsuccessfulOutcome,
        _ => TriggeringMessageValue::InitiatingMessage,
    });
    let procedure_criticality = diag.procedure_criticality.as_ref().map(|c| match c.0 {
        Criticality::REJECT => CriticalityValue::Reject,
        Criticality::IGNORE => CriticalityValue::Ignore,
        Criticality::NOTIFY => CriticalityValue::Notify,
        _ => CriticalityValue::Ignore,
    });

    let ies_criticality_diagnostics = diag
        .i_es_criticality_diagnostics
        .as_ref()
        .map(|list| {
            list.0
                .iter()
                .map(|item| {
                    let ie_criticality = match item.ie_criticality.0 {
                        Criticality::REJECT => CriticalityValue::Reject,
                        Criticality::IGNORE => CriticalityValue::Ignore,
                        Criticality::NOTIFY => CriticalityValue::Notify,
                        _ => CriticalityValue::Ignore,
                    };
                    let type_of_error = match item.type_of_error.0 {
                        TypeOfError::NOT_UNDERSTOOD => TypeOfErrorValue::NotUnderstood,
                        TypeOfError::MISSING => TypeOfErrorValue::Missing,
                        _ => TypeOfErrorValue::NotUnderstood,
                    };
                    IeCriticalityDiagnosticsItem {
                        ie_criticality,
                        ie_id: item.ie_id.0,
                        type_of_error,
                    }
                })
                .collect()
        })
        .unwrap_or_default();

    CriticalityDiagnosticsInfo {
        procedure_code,
        triggering_message,
        procedure_criticality,
        ies_criticality_diagnostics,
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Build and encode a Handover Preparation Failure to bytes
pub fn encode_handover_preparation_failure_msg(
    params: &HandoverPreparationFailureParams,
) -> Result<Vec<u8>, HandoverPreparationFailureError> {
    let pdu = build_handover_preparation_failure(params)?;
    Ok(encode_ngap_pdu(&pdu)?)
}

/// Decode and parse a Handover Preparation Failure from bytes
pub fn decode_handover_preparation_failure_msg(
    bytes: &[u8],
) -> Result<HandoverPreparationFailureFullData, HandoverPreparationFailureError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_handover_preparation_failure_full(&pdu)
}

/// Check if an NGAP PDU is a Handover Preparation Failure
pub fn is_handover_preparation_failure(pdu: &NGAP_PDU) -> bool {
    matches!(
        pdu,
        NGAP_PDU::UnsuccessfulOutcome(outcome)
            if matches!(outcome.value, UnsuccessfulOutcomeValue::Id_HandoverPreparation(_))
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_params() -> HandoverPreparationFailureParams {
        HandoverPreparationFailureParams {
            amf_ue_ngap_id: 12345,
            ran_ue_ngap_id: 67890,
            cause: NgSetupFailureCause::RadioNetwork(RadioNetworkCause::HoTargetNotAllowed),
            criticality_diagnostics: None,
        }
    }

    #[test]
    fn test_build_handover_preparation_failure() {
        let params = create_test_params();
        let result = build_handover_preparation_failure(&params);
        assert!(result.is_ok());

        let pdu = result.unwrap();
        assert!(is_handover_preparation_failure(&pdu));
    }

    #[test]
    fn test_encode_handover_preparation_failure() {
        let params = create_test_params();
        let result = encode_handover_preparation_failure_msg(&params);
        assert!(result.is_ok());

        let bytes = result.unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_handover_preparation_failure_roundtrip() {
        let params = create_test_params();

        let pdu = build_handover_preparation_failure(&params).expect("Failed to build");
        let encoded = encode_ngap_pdu(&pdu).expect("Failed to encode");
        let decoded_pdu = decode_ngap_pdu(&encoded).expect("Failed to decode");
        let parsed =
            parse_handover_preparation_failure_full(&decoded_pdu).expect("Failed to parse");

        assert_eq!(parsed.amf_ue_ngap_id, params.amf_ue_ngap_id);
        assert_eq!(parsed.ran_ue_ngap_id, params.ran_ue_ngap_id);
    }

    #[test]
    fn test_handover_preparation_failure_with_diagnostics() {
        let params = HandoverPreparationFailureParams {
            amf_ue_ngap_id: 100,
            ran_ue_ngap_id: 200,
            cause: NgSetupFailureCause::RadioNetwork(
                RadioNetworkCause::NoRadioResourcesAvailableInTargetCell,
            ),
            criticality_diagnostics: Some(CriticalityDiagnosticsInfo {
                procedure_code: Some(0),
                triggering_message: Some(TriggeringMessageValue::InitiatingMessage),
                procedure_criticality: Some(CriticalityValue::Reject),
                ies_criticality_diagnostics: vec![IeCriticalityDiagnosticsItem {
                    ie_criticality: CriticalityValue::Reject,
                    ie_id: 10,
                    type_of_error: TypeOfErrorValue::Missing,
                }],
            }),
        };

        let encoded = encode_handover_preparation_failure_msg(&params).expect("Failed to encode");
        let parsed =
            decode_handover_preparation_failure_msg(&encoded).expect("Failed to decode");

        assert_eq!(parsed.amf_ue_ngap_id, 100);
        assert_eq!(parsed.ran_ue_ngap_id, 200);
        assert!(parsed.criticality_diagnostics.is_some());
        let diag = parsed.criticality_diagnostics.unwrap();
        assert_eq!(diag.procedure_code, Some(0));
        assert_eq!(diag.ies_criticality_diagnostics.len(), 1);
    }
}
