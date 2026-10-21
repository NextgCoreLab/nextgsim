//! TS 38.413 §10.3.4 for messages the gNB receives from the AMF.
//!
//! [`decode_ngap_pdu`] refuses a whole message when one protocol IE carries an
//! id this schema does not know, so a newer-release AMF that adds an IE would
//! have every such message answered as undecodable. §10.3.4 says otherwise: an
//! unknown IE is handled by the criticality the sender gave it (§10.3.4.2).
//!
//! * **ignore**: drop the IE and process the rest of the message.
//! * **ignore and notify**: the same, and tell the sender which IEs were not
//!   understood. For a message initiating a procedure that has a response, the
//!   report goes in that response's Criticality Diagnostics IE; otherwise (a
//!   Class 2 procedure, or a response message) in an ERROR INDICATION.
//! * **reject**: do not process the message. For an initiating message whose
//!   procedure has a failure message, the procedure is rejected with that
//!   message; for one without (or when the message lacks what the failure needs),
//!   the procedure is terminated with an ERROR INDICATION. For a response the
//!   procedure is terminated with local error handling and nothing is sent.
//!
//! An ERROR INDICATION about a UE-associated message carries that message's AMF
//! and RAN UE NGAP IDs (§8.7.5.2).
//!
//! The classification is pure so it can be tested without a task: the gate in
//! `NgapTask::handle_ngap_pdu` acts on its verdict.

use nextgsim_ngap::codec::generated::{
    AMFConfigurationUpdateFailure, AMFConfigurationUpdateFailureProtocolIEs,
    AMFConfigurationUpdateFailureProtocolIEs_Entry,
    AMFConfigurationUpdateFailureProtocolIEs_EntryValue, Cause, CauseProtocol, HandoverFailure,
    HandoverFailureProtocolIEs, HandoverFailureProtocolIEs_Entry,
    HandoverFailureProtocolIEs_EntryValue, InitialContextSetupFailure,
    InitialContextSetupFailureProtocolIEs, InitialContextSetupFailureProtocolIEs_Entry,
    InitialContextSetupFailureProtocolIEs_EntryValue, ProtocolIE_ID, UEContextModificationFailure,
    UEContextModificationFailureProtocolIEs, UEContextModificationFailureProtocolIEs_Entry,
    UEContextModificationFailureProtocolIEs_EntryValue, UnsuccessfulOutcome,
    UnsuccessfulOutcomeValue, AMF_UE_NGAP_ID, ID_AMF_CONFIGURATION_UPDATE, ID_AMF_UE_NGAP_ID,
    ID_CAUSE, ID_CRITICALITY_DIAGNOSTICS, ID_HANDOVER_RESOURCE_ALLOCATION,
    ID_INITIAL_CONTEXT_SETUP, ID_RAN_UE_NGAP_ID, ID_UE_CONTEXT_MODIFICATION, RAN_UE_NGAP_ID,
};
use nextgsim_ngap::codec::{
    decode_aper, decode_ngap_pdu, decode_ngap_pdu_with_unknown_ies, encode_ngap_pdu, Criticality,
    UnknownIe, NGAP_PDU,
};
use nextgsim_ngap::procedures::error_indication::{
    build_criticality_diagnostics, error_indication_abstract_syntax_error,
    CriticalityDiagnosticsInfo, CriticalityValue, ErrorIndicationParams,
    IeCriticalityDiagnosticsItem, TriggeringMessageValue, TypeOfErrorValue,
};

/// Unknown `ignore-and-notify` IEs to report (§10.3.4.2).
#[derive(Debug, Clone)]
pub(super) struct Notify {
    /// The received message initiates a procedure: the report belongs in that
    /// procedure's response, and [`Self::error_indication`] is only the fallback
    /// for a procedure that sends none.
    pub(super) initiating: bool,
    /// The received message's procedure code: the response that carries the
    /// report is the one with the same code.
    pub(super) procedure_code: u8,
    /// The Criticality Diagnostics naming the IEs.
    pub(super) diagnostics: CriticalityDiagnosticsInfo,
    /// The same report as an ERROR INDICATION, with the message's UE NGAP IDs.
    pub(super) error_indication: ErrorIndicationParams,
    /// The received message carries a UE NGAP ID, so the report is UE-associated.
    pub(super) ue_associated: bool,
}

/// What the gNB sends for a message rejected for an unknown `reject` IE.
#[derive(Debug)]
pub(super) enum RejectAnswer {
    /// The procedure's failure message, encoded (§10.3.4.2 "the message normally
    /// used to report unsuccessful outcome").
    Failure(Vec<u8>),
    /// An ERROR INDICATION: the procedure has no failure message, or the request
    /// lacked what the failure message needs.
    ErrorIndication(ErrorIndicationParams),
    /// A response was rejected: local error handling only, nothing sent.
    Local,
}

/// What the gate does with one received PDU.
#[derive(Debug)]
pub(super) enum UnknownIeVerdict {
    /// Decodes under the default codec: route the original bytes unchanged.
    Clean,
    /// Does not decode even skipping unknown IEs: the existing undecodable
    /// handling (transfer-syntax-error Error Indication) applies.
    Undecodable,
    /// Every unknown IE is `ignore` or `notify`: route `bytes`, the message
    /// re-encoded without them, and report `notify` (if any IE was `notify`).
    Proceed {
        bytes: Vec<u8>,
        notify: Option<Notify>,
    },
    /// At least one unknown IE is `reject` (or carries no criticality, read as
    /// reject): the procedure is not executed and `answer` is sent.
    Reject {
        answer: RejectAnswer,
        ue_associated: bool,
    },
}

/// Classifies `bytes` per TS 38.413 §10.3.4.2.
pub(super) fn classify(bytes: &[u8]) -> UnknownIeVerdict {
    if decode_ngap_pdu(bytes).is_ok() {
        return UnknownIeVerdict::Clean;
    }
    let Ok((pdu, unknown)) = decode_ngap_pdu_with_unknown_ies(bytes) else {
        return UnknownIeVerdict::Undecodable;
    };
    if unknown.is_empty() {
        // The default decode failed for a reason other than an unknown IE id.
        return UnknownIeVerdict::Undecodable;
    }
    let (amf_ue_ngap_id, ran_ue_ngap_id) = ue_ngap_ids(bytes);
    let ue_associated = amf_ue_ngap_id.is_some() || ran_ue_ngap_id.is_some();
    let initiating = matches!(pdu, NGAP_PDU::InitiatingMessage(_));
    let criticality = |ie: &UnknownIe| ie.criticality.as_ref().map_or(Criticality::REJECT, |c| c.0);
    let error_indication = |reject: bool, diagnostics: CriticalityDiagnosticsInfo| {
        let mut params =
            error_indication_abstract_syntax_error(amf_ue_ngap_id, ran_ue_ngap_id, reject);
        params.criticality_diagnostics = Some(diagnostics);
        params
    };

    if unknown
        .iter()
        .any(|ie| !matches!(criticality(ie), Criticality::IGNORE | Criticality::NOTIFY))
    {
        // §10.3.4.2: the IEs reported are those marked reject or notify.
        let reported: Vec<&UnknownIe> = unknown
            .iter()
            .filter(|ie| criticality(ie) != Criticality::IGNORE)
            .collect();
        let diagnostics = diagnostics(&pdu, &reported);
        let answer = if !initiating {
            RejectAnswer::Local
        } else if let Some(failure) =
            failure_message(&pdu, amf_ue_ngap_id, ran_ue_ngap_id, &diagnostics)
        {
            RejectAnswer::Failure(failure)
        } else {
            RejectAnswer::ErrorIndication(error_indication(true, diagnostics))
        };
        return UnknownIeVerdict::Reject {
            answer,
            ue_associated,
        };
    }
    let Ok(clean) = encode_ngap_pdu(&pdu) else {
        return UnknownIeVerdict::Undecodable;
    };
    let notified: Vec<&UnknownIe> = unknown
        .iter()
        .filter(|ie| criticality(ie) == Criticality::NOTIFY)
        .collect();
    let notify = (!notified.is_empty()).then(|| {
        let diagnostics = diagnostics(&pdu, &notified);
        Notify {
            initiating,
            procedure_code: procedure_code(&pdu),
            error_indication: error_indication(false, diagnostics.clone()),
            diagnostics,
            ue_associated,
        }
    });
    UnknownIeVerdict::Proceed {
        bytes: clean,
        notify,
    }
}

fn procedure_code(pdu: &NGAP_PDU) -> u8 {
    match pdu {
        NGAP_PDU::InitiatingMessage(m) => m.procedure_code.0,
        NGAP_PDU::SuccessfulOutcome(m) => m.procedure_code.0,
        NGAP_PDU::UnsuccessfulOutcome(m) => m.procedure_code.0,
    }
}

/// Criticality Diagnostics (§9.3.1.3) naming `ies` as not understood, with the
/// procedure code, triggering message and procedure criticality of `pdu`.
fn diagnostics(pdu: &NGAP_PDU, ies: &[&UnknownIe]) -> CriticalityDiagnosticsInfo {
    let (triggering_message, procedure_criticality) = match pdu {
        NGAP_PDU::InitiatingMessage(m) => {
            (TriggeringMessageValue::InitiatingMessage, m.criticality.0)
        }
        NGAP_PDU::SuccessfulOutcome(m) => {
            (TriggeringMessageValue::SuccessfulOutcome, m.criticality.0)
        }
        NGAP_PDU::UnsuccessfulOutcome(m) => {
            (TriggeringMessageValue::UnsuccessfulOutcome, m.criticality.0)
        }
    };
    CriticalityDiagnosticsInfo {
        procedure_code: Some(procedure_code(pdu)),
        triggering_message: Some(triggering_message),
        procedure_criticality: Some(criticality_value(procedure_criticality)),
        ies_criticality_diagnostics: ies
            .iter()
            .filter_map(|ie| {
                Some(IeCriticalityDiagnosticsItem {
                    ie_criticality: criticality_value(
                        ie.criticality.as_ref().map_or(Criticality::REJECT, |c| c.0),
                    ),
                    ie_id: u16::try_from(ie.id).ok()?,
                    type_of_error: TypeOfErrorValue::NotUnderstood,
                })
            })
            .collect(),
    }
}

fn criticality_value(c: u8) -> CriticalityValue {
    match c {
        Criticality::IGNORE => CriticalityValue::Ignore,
        Criticality::NOTIFY => CriticalityValue::Notify,
        _ => CriticalityValue::Reject,
    }
}

/// The failure message rejecting `pdu`, a request of a procedure that has one
/// (§10.3.4.2), carrying cause abstract-syntax-error-reject and `diagnostics`.
/// `None` for a procedure without a failure message, or when the request lacks
/// a UE NGAP ID the failure message must echo -- an ERROR INDICATION then.
///
/// The procedures are those the gNB receives a request for whose failure
/// message it sends: Initial Context Setup (§9.2.2.3), UE Context Modification
/// (§9.2.2.9), Handover Resource Allocation (§9.2.3.6) and AMF Configuration
/// Update (§9.2.6.9). IE order and criticality as the ASN.1 lists them.
fn failure_message(
    pdu: &NGAP_PDU,
    amf_ue_ngap_id: Option<u64>,
    ran_ue_ngap_id: Option<u32>,
    diagnostics: &CriticalityDiagnosticsInfo,
) -> Option<Vec<u8>> {
    let NGAP_PDU::InitiatingMessage(request) = pdu else {
        return None;
    };
    let ignore = || Criticality(Criticality::IGNORE);
    let cause = || Cause::Protocol(CauseProtocol(CauseProtocol::ABSTRACT_SYNTAX_ERROR_REJECT));
    let cd = || build_criticality_diagnostics(diagnostics);
    let value = match request.procedure_code.0 {
        ID_INITIAL_CONTEXT_SETUP => {
            use InitialContextSetupFailureProtocolIEs_Entry as E;
            use InitialContextSetupFailureProtocolIEs_EntryValue as V;
            UnsuccessfulOutcomeValue::Id_InitialContextSetup(InitialContextSetupFailure {
                protocol_i_es: InitialContextSetupFailureProtocolIEs(vec![
                    E {
                        id: ProtocolIE_ID(ID_AMF_UE_NGAP_ID),
                        criticality: ignore(),
                        value: V::Id_AMF_UE_NGAP_ID(AMF_UE_NGAP_ID(amf_ue_ngap_id?)),
                    },
                    E {
                        id: ProtocolIE_ID(ID_RAN_UE_NGAP_ID),
                        criticality: ignore(),
                        value: V::Id_RAN_UE_NGAP_ID(RAN_UE_NGAP_ID(ran_ue_ngap_id?)),
                    },
                    E {
                        id: ProtocolIE_ID(ID_CAUSE),
                        criticality: ignore(),
                        value: V::Id_Cause(cause()),
                    },
                    E {
                        id: ProtocolIE_ID(ID_CRITICALITY_DIAGNOSTICS),
                        criticality: ignore(),
                        value: V::Id_CriticalityDiagnostics(cd()),
                    },
                ]),
            })
        }
        ID_UE_CONTEXT_MODIFICATION => {
            use UEContextModificationFailureProtocolIEs_Entry as E;
            use UEContextModificationFailureProtocolIEs_EntryValue as V;
            UnsuccessfulOutcomeValue::Id_UEContextModification(UEContextModificationFailure {
                protocol_i_es: UEContextModificationFailureProtocolIEs(vec![
                    E {
                        id: ProtocolIE_ID(ID_AMF_UE_NGAP_ID),
                        criticality: ignore(),
                        value: V::Id_AMF_UE_NGAP_ID(AMF_UE_NGAP_ID(amf_ue_ngap_id?)),
                    },
                    E {
                        id: ProtocolIE_ID(ID_RAN_UE_NGAP_ID),
                        criticality: ignore(),
                        value: V::Id_RAN_UE_NGAP_ID(RAN_UE_NGAP_ID(ran_ue_ngap_id?)),
                    },
                    E {
                        id: ProtocolIE_ID(ID_CAUSE),
                        criticality: ignore(),
                        value: V::Id_Cause(cause()),
                    },
                    E {
                        id: ProtocolIE_ID(ID_CRITICALITY_DIAGNOSTICS),
                        criticality: ignore(),
                        value: V::Id_CriticalityDiagnostics(cd()),
                    },
                ]),
            })
        }
        ID_HANDOVER_RESOURCE_ALLOCATION => {
            use HandoverFailureProtocolIEs_Entry as E;
            use HandoverFailureProtocolIEs_EntryValue as V;
            UnsuccessfulOutcomeValue::Id_HandoverResourceAllocation(HandoverFailure {
                protocol_i_es: HandoverFailureProtocolIEs(vec![
                    E {
                        id: ProtocolIE_ID(ID_AMF_UE_NGAP_ID),
                        criticality: ignore(),
                        value: V::Id_AMF_UE_NGAP_ID(AMF_UE_NGAP_ID(amf_ue_ngap_id?)),
                    },
                    E {
                        id: ProtocolIE_ID(ID_CAUSE),
                        criticality: ignore(),
                        value: V::Id_Cause(cause()),
                    },
                    E {
                        id: ProtocolIE_ID(ID_CRITICALITY_DIAGNOSTICS),
                        criticality: ignore(),
                        value: V::Id_CriticalityDiagnostics(cd()),
                    },
                ]),
            })
        }
        ID_AMF_CONFIGURATION_UPDATE => {
            use AMFConfigurationUpdateFailureProtocolIEs_Entry as E;
            use AMFConfigurationUpdateFailureProtocolIEs_EntryValue as V;
            UnsuccessfulOutcomeValue::Id_AMFConfigurationUpdate(AMFConfigurationUpdateFailure {
                protocol_i_es: AMFConfigurationUpdateFailureProtocolIEs(vec![
                    E {
                        id: ProtocolIE_ID(ID_CAUSE),
                        criticality: ignore(),
                        value: V::Id_Cause(cause()),
                    },
                    E {
                        id: ProtocolIE_ID(ID_CRITICALITY_DIAGNOSTICS),
                        criticality: ignore(),
                        value: V::Id_CriticalityDiagnostics(cd()),
                    },
                ]),
            })
        }
        _ => return None,
    };
    encode_ngap_pdu(&NGAP_PDU::UnsuccessfulOutcome(UnsuccessfulOutcome {
        procedure_code: request.procedure_code.clone(),
        criticality: request.criticality.clone(),
        value,
    }))
    .ok()
}

/// `bytes`, an encoded response to the procedure `procedure_code`, with a
/// Criticality Diagnostics IE carrying `diagnostics` inserted where the ASN.1
/// lists it (§10.3.4.2: "report in the response message of the procedure").
///
/// `None` when `bytes` is not a response of that procedure the gNB sends to an
/// AMF-initiated Class 1 procedure, or already carries the IE: the caller then
/// reports by ERROR INDICATION instead.
pub(super) fn with_criticality_diagnostics(
    bytes: &[u8],
    procedure_code: u8,
    diagnostics: &CriticalityDiagnosticsInfo,
) -> Option<Vec<u8>> {
    use nextgsim_ngap::codec::generated::{
        ID_NPN_ACCESS_INFORMATION, ID_PAGING_ASSIS_DATAFOR_C_ECAPAB_UE, ID_RED_CAP_INDICATION,
        ID_TARGETTO_SOURCE_FAILURE_TRANSPARENT_CONTAINER, ID_USER_LOCATION_INFORMATION,
    };
    /// Insert the IE before the first IE the ASN.1 lists after it (`$after`),
    /// else last; refuse a message that already has one.
    macro_rules! insert {
        ($msg:expr, $entry:ident, $value:ident, [$($after:expr),*]) => {{
            use nextgsim_ngap::codec::generated::{$entry, $value};
            let after: &[u16] = &[$($after),*];
            let ies = &mut $msg.protocol_i_es.0;
            if ies.iter().any(|e| e.id.0 == ID_CRITICALITY_DIAGNOSTICS) {
                return None;
            }
            let at = ies
                .iter()
                .position(|e| after.contains(&e.id.0))
                .unwrap_or(ies.len());
            ies.insert(
                at,
                $entry {
                    id: ProtocolIE_ID(ID_CRITICALITY_DIAGNOSTICS),
                    criticality: Criticality(Criticality::IGNORE),
                    value: $value::Id_CriticalityDiagnostics(build_criticality_diagnostics(
                        diagnostics,
                    )),
                },
            );
        }};
    }
    use nextgsim_ngap::codec::generated::SuccessfulOutcomeValue as S;
    use UnsuccessfulOutcomeValue as U;
    let mut pdu = decode_ngap_pdu(bytes).ok()?;
    match &mut pdu {
        NGAP_PDU::SuccessfulOutcome(o) if o.procedure_code.0 == procedure_code => {
            match &mut o.value {
                S::Id_AMFConfigurationUpdate(m) => insert!(
                    m,
                    AMFConfigurationUpdateAcknowledgeProtocolIEs_Entry,
                    AMFConfigurationUpdateAcknowledgeProtocolIEs_EntryValue,
                    []
                ),
                S::Id_HandoverResourceAllocation(m) => insert!(
                    m,
                    HandoverRequestAcknowledgeProtocolIEs_Entry,
                    HandoverRequestAcknowledgeProtocolIEs_EntryValue,
                    [ID_NPN_ACCESS_INFORMATION, ID_RED_CAP_INDICATION]
                ),
                S::Id_InitialContextSetup(m) => insert!(
                    m,
                    InitialContextSetupResponseProtocolIEs_Entry,
                    InitialContextSetupResponseProtocolIEs_EntryValue,
                    []
                ),
                S::Id_MulticastSessionActivation(m) => insert!(
                    m,
                    MulticastSessionActivationResponseProtocolIEs_Entry,
                    MulticastSessionActivationResponseProtocolIEs_EntryValue,
                    []
                ),
                S::Id_MulticastSessionDeactivation(m) => insert!(
                    m,
                    MulticastSessionDeactivationResponseProtocolIEs_Entry,
                    MulticastSessionDeactivationResponseProtocolIEs_EntryValue,
                    []
                ),
                S::Id_NGReset(m) => insert!(
                    m,
                    NGResetAcknowledgeProtocolIEs_Entry,
                    NGResetAcknowledgeProtocolIEs_EntryValue,
                    []
                ),
                S::Id_PDUSessionResourceModify(m) => insert!(
                    m,
                    PDUSessionResourceModifyResponseProtocolIEs_Entry,
                    PDUSessionResourceModifyResponseProtocolIEs_EntryValue,
                    []
                ),
                S::Id_PDUSessionResourceRelease(m) => insert!(
                    m,
                    PDUSessionResourceReleaseResponseProtocolIEs_Entry,
                    PDUSessionResourceReleaseResponseProtocolIEs_EntryValue,
                    []
                ),
                S::Id_PDUSessionResourceSetup(m) => insert!(
                    m,
                    PDUSessionResourceSetupResponseProtocolIEs_Entry,
                    PDUSessionResourceSetupResponseProtocolIEs_EntryValue,
                    [ID_USER_LOCATION_INFORMATION]
                ),
                S::Id_UEContextModification(m) => insert!(
                    m,
                    UEContextModificationResponseProtocolIEs_Entry,
                    UEContextModificationResponseProtocolIEs_EntryValue,
                    []
                ),
                S::Id_UEContextRelease(m) => insert!(
                    m,
                    UEContextReleaseCompleteProtocolIEs_Entry,
                    UEContextReleaseCompleteProtocolIEs_EntryValue,
                    [ID_PAGING_ASSIS_DATAFOR_C_ECAPAB_UE]
                ),
                S::Id_UERadioCapabilityCheck(m) => insert!(
                    m,
                    UERadioCapabilityCheckResponseProtocolIEs_Entry,
                    UERadioCapabilityCheckResponseProtocolIEs_EntryValue,
                    []
                ),
                _ => return None,
            }
        }
        NGAP_PDU::UnsuccessfulOutcome(o) if o.procedure_code.0 == procedure_code => {
            match &mut o.value {
                U::Id_AMFConfigurationUpdate(m) => insert!(
                    m,
                    AMFConfigurationUpdateFailureProtocolIEs_Entry,
                    AMFConfigurationUpdateFailureProtocolIEs_EntryValue,
                    []
                ),
                U::Id_HandoverResourceAllocation(m) => insert!(
                    m,
                    HandoverFailureProtocolIEs_Entry,
                    HandoverFailureProtocolIEs_EntryValue,
                    [ID_TARGETTO_SOURCE_FAILURE_TRANSPARENT_CONTAINER]
                ),
                U::Id_InitialContextSetup(m) => insert!(
                    m,
                    InitialContextSetupFailureProtocolIEs_Entry,
                    InitialContextSetupFailureProtocolIEs_EntryValue,
                    []
                ),
                U::Id_UEContextModification(m) => insert!(
                    m,
                    UEContextModificationFailureProtocolIEs_Entry,
                    UEContextModificationFailureProtocolIEs_EntryValue,
                    []
                ),
                _ => return None,
            }
        }
        _ => return None,
    }
    encode_ngap_pdu(&pdu).ok()
}

/// The AMF and RAN UE NGAP IDs (IEs 10 and 85) at the top level of the APER
/// NGAP-PDU `bytes`, read from the wire framing so it works for every message,
/// including one carrying IEs this schema does not know.
///
/// NGAP-PDU is a CHOICE of a SEQUENCE {procedureCode, criticality, value}: one
/// octet of choice index, the procedure code octet, the criticality octet, then
/// the open-type length and the message. Each message is a SEQUENCE with an
/// extension bit (one octet) and a ProtocolIE-Container whose 16-bit count is
/// followed by fields of id (two octets), criticality (one octet), open-type
/// length and value (X.691 §10.9, §11.6.1). A fragmented (≥16K) length is not
/// followed.
pub(super) fn ue_ngap_ids(bytes: &[u8]) -> (Option<u64>, Option<u32>) {
    fn open_type(bytes: &[u8], at: usize) -> Option<(&[u8], usize)> {
        let first = *bytes.get(at)?;
        let (len, start) = match first {
            b if b & 0x80 == 0 => (usize::from(b), at + 1),
            b if b & 0xC0 == 0x80 => (
                (usize::from(b & 0x3F) << 8) | usize::from(*bytes.get(at + 1)?),
                at + 2,
            ),
            _ => return None,
        };
        let value = bytes.get(start..start + len)?;
        Some((value, start + len))
    }
    let mut amf = None;
    let mut ran = None;
    let Some((message, _)) = open_type(bytes, 3) else {
        return (None, None);
    };
    let Some(count) = message.get(1..3).map(|c| u16::from_be_bytes([c[0], c[1]])) else {
        return (None, None);
    };
    let mut at = 3;
    for _ in 0..count {
        let Some(id) = message
            .get(at..at + 2)
            .map(|c| u16::from_be_bytes([c[0], c[1]]))
        else {
            break;
        };
        let Some((value, next)) = open_type(message, at + 3) else {
            break;
        };
        match id {
            ID_AMF_UE_NGAP_ID => amf = decode_aper::<AMF_UE_NGAP_ID>(value).ok().map(|v| v.0),
            ID_RAN_UE_NGAP_ID => ran = decode_aper::<RAN_UE_NGAP_ID>(value).ok().map(|v| v.0),
            _ => {}
        }
        at = next;
    }
    (amf, ran)
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use nextgsim_ngap::codec::encode_aper;
    use nextgsim_ngap::codec::generated::{
        InitiatingMessageValue, OverloadStartProtocolIEs_Entry,
        OverloadStartProtocolIEs_EntryValue, ProtocolIE_ID, TrafficLoadReductionIndication,
    };
    use nextgsim_ngap::procedures::error_indication::{
        decode_error_indication, encode_error_indication,
    };

    /// An OVERLOAD START (Class 2, TS 38.413 §8.7.6) carrying one extra IE whose
    /// id is rewritten on the wire to `unknown_id` with `criticality`, the #214
    /// technique: every entry starts octet-aligned, so the extra entry encoded on
    /// its own appears verbatim in the PDU. Returns the bytes and the message
    /// without the extra IE.
    pub(in crate::ngap) fn overload_start_with_unknown_ie(
        unknown_id: u16,
        criticality: u8,
    ) -> (Vec<u8>, NGAP_PDU) {
        use nextgsim_ngap::codec::generated::{
            InitiatingMessage, OverloadStart, OverloadStartProtocolIEs, ProcedureCode,
            ID_AMF_TRAFFIC_LOAD_REDUCTION_INDICATION, ID_OVERLOAD_START,
        };
        let marker = OverloadStartProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_AMF_TRAFFIC_LOAD_REDUCTION_INDICATION),
            criticality: Criticality(criticality),
            value: OverloadStartProtocolIEs_EntryValue::Id_AMFTrafficLoadReductionIndication(
                TrafficLoadReductionIndication(77),
            ),
        };
        let build = |ies: Vec<OverloadStartProtocolIEs_Entry>| {
            NGAP_PDU::InitiatingMessage(InitiatingMessage {
                procedure_code: ProcedureCode(ID_OVERLOAD_START),
                criticality: Criticality(Criticality::IGNORE),
                value: InitiatingMessageValue::Id_OverloadStart(OverloadStart {
                    protocol_i_es: OverloadStartProtocolIEs(ies),
                }),
            })
        };
        let original = build(vec![]);
        let mut bytes = encode_ngap_pdu(&build(vec![marker.clone()])).expect("encode");
        let entry = encode_aper(&marker).expect("entry");
        let at = bytes
            .windows(entry.len())
            .position(|w| w == entry.as_slice())
            .expect("the extra entry appears verbatim");
        bytes[at..at + 2].copy_from_slice(&unknown_id.to_be_bytes());
        (bytes, original)
    }

    /// `bytes`, an encoded NGAP PDU, with one more protocol IE appended on the
    /// wire: id `unknown_id`, `criticality`, a one-octet value. Rebuilds the IE
    /// count and the message's open-type length (X.691 §10.9), so it works on any
    /// message whose IE container is under 16K. Independent of `ue_ngap_ids`,
    /// which reads the same framing.
    pub(in crate::ngap) fn with_unknown_ie(
        bytes: &[u8],
        unknown_id: u16,
        criticality: u8,
    ) -> Vec<u8> {
        fn length(len: usize) -> Vec<u8> {
            if len < 128 {
                vec![len as u8]
            } else {
                vec![0x80 | (len >> 8) as u8, len as u8]
            }
        }
        let (len, start) = if bytes[3] & 0x80 == 0 {
            (usize::from(bytes[3]), 4)
        } else {
            (
                (usize::from(bytes[3] & 0x3F) << 8) | usize::from(bytes[4]),
                5,
            )
        };
        let mut message = bytes[start..start + len].to_vec();
        let count = u16::from_be_bytes([message[1], message[2]]) + 1;
        message[1..3].copy_from_slice(&count.to_be_bytes());
        message.extend_from_slice(&unknown_id.to_be_bytes());
        message.push(criticality << 6);
        message.extend_from_slice(&[0x01, 0x00]);
        let mut out = bytes[..3].to_vec();
        out.extend(length(message.len()));
        out.extend(message);
        out
    }

    /// A UE CONTEXT MODIFICATION REQUEST (Class 1 with a failure message,
    /// TS 38.413 §8.3.4) for AMF/RAN UE NGAP IDs `amf`/`ran`, carrying a new KgNB.
    pub(in crate::ngap) fn ucm_request(amf: u64, ran: u32) -> Vec<u8> {
        use nextgsim_ngap::procedures::ue_context_modification::{
            encode_ue_context_modification_request, UeContextModificationRequestData,
        };
        encode_ue_context_modification_request(&UeContextModificationRequestData {
            amf_ue_ngap_id: amf,
            ran_ue_ngap_id: ran,
            ue_aggregate_max_bit_rate: None,
            ue_security_capabilities: Some(
                nextgsim_ngap::procedures::initial_context_setup::UeSecurityCapabilitiesValue {
                    nr_encryption_algorithms: 0xC000,
                    nr_integrity_algorithms: 0xC000,
                    eutra_encryption_algorithms: None,
                    eutra_integrity_algorithms: None,
                },
            ),
            security_key: Some([0x22; 32]),
            ue_radio_capability_id: None,
        })
        .expect("encode UCM request")
    }

    /// A DOWNLINK NAS TRANSPORT (Class 2, TS 38.413 §8.6.2) for `amf`/`ran`.
    pub(in crate::ngap) fn dl_nas(amf: u64, ran: u32) -> Vec<u8> {
        use nextgsim_ngap::procedures::nas_transport::{
            encode_downlink_nas_transport, DownlinkNasTransportParams,
        };
        encode_downlink_nas_transport(&DownlinkNasTransportParams {
            amf_ue_ngap_id: amf,
            ran_ue_ngap_id: ran,
            nas_pdu: vec![0x7E, 0x00, 0x56],
            old_amf: None,
            ran_paging_priority: None,
            index_to_rfsp: None,
            ue_ambr: None,
            allowed_nssai: None,
            ue_radio_capability: None,
            ue_capability_info_request: false,
            ue_radio_capability_id: None,
        })
        .expect("encode DL NAS Transport")
    }

    /// The Cause and Criticality Diagnostics of an encoded UE CONTEXT
    /// MODIFICATION FAILURE, with its (AMF, RAN) UE NGAP IDs.
    pub(in crate::ngap) fn ucm_failure(
        bytes: &[u8],
    ) -> (Option<(u64, u32)>, Cause, CriticalityDiagnosticsInfo) {
        use nextgsim_ngap::procedures::error_indication::parse_criticality_diagnostics;
        use UEContextModificationFailureProtocolIEs_EntryValue as V;
        let NGAP_PDU::UnsuccessfulOutcome(o) = decode_ngap_pdu(bytes).expect("decodes") else {
            panic!("expected an UnsuccessfulOutcome");
        };
        let UnsuccessfulOutcomeValue::Id_UEContextModification(f) = o.value else {
            panic!("expected a UE Context Modification Failure");
        };
        let (mut amf, mut ran, mut cause, mut cd) = (None, None, None, None);
        for ie in f.protocol_i_es.0 {
            match ie.value {
                V::Id_AMF_UE_NGAP_ID(v) => amf = Some(v.0),
                V::Id_RAN_UE_NGAP_ID(v) => ran = Some(v.0),
                V::Id_Cause(c) => cause = Some(c),
                V::Id_CriticalityDiagnostics(d) => {
                    cd = Some(parse_criticality_diagnostics(&d).unwrap())
                }
            }
        }
        (
            amf.zip(ran),
            cause.expect("Cause"),
            cd.expect("Criticality Diagnostics"),
        )
    }

    fn reported(d: &CriticalityDiagnosticsInfo) -> Vec<(u16, CriticalityValue)> {
        d.ies_criticality_diagnostics
            .iter()
            .map(|i| (i.ie_id, i.ie_criticality))
            .collect()
    }

    #[test]
    fn the_ue_ngap_ids_are_read_from_the_wire_framing() {
        let max_amf = (1u64 << 40) - 1;
        assert_eq!(
            ue_ngap_ids(&ucm_request(max_amf, u32::MAX)),
            (Some(max_amf), Some(u32::MAX))
        );
        assert_eq!(ue_ngap_ids(&ucm_request(0, 7)), (Some(0), Some(7)));
        // Read past an IE the schema does not know.
        assert_eq!(
            ue_ngap_ids(&with_unknown_ie(&dl_nas(555, 3), 9999, Criticality::REJECT)),
            (Some(555), Some(3))
        );
        // A non-UE-associated message has neither.
        let (overload, _) = overload_start_with_unknown_ie(9999, Criticality::IGNORE);
        assert_eq!(ue_ngap_ids(&overload), (None, None));
        assert_eq!(ue_ngap_ids(&[0xFF]), (None, None));
    }

    #[test]
    fn the_injected_ie_is_unknown_to_the_schema_and_skipped() {
        let clean = ucm_request(555, 3);
        let bytes = with_unknown_ie(&clean, 9999, Criticality::NOTIFY);
        assert!(decode_ngap_pdu(&bytes).is_err());
        let (pdu, unknown) = decode_ngap_pdu_with_unknown_ies(&bytes).expect("skips");
        assert_eq!(pdu, decode_ngap_pdu(&clean).unwrap());
        assert_eq!(unknown.len(), 1);
        assert_eq!(unknown[0].id, 9999);
        assert_eq!(
            unknown[0].criticality,
            Some(Criticality(Criticality::NOTIFY))
        );
    }

    /// §10.3.4.2 Reject IE, a procedure with a failure message: the answer is
    /// that failure message (not an Error Indication), cause
    /// abstract-syntax-error-reject, echoing the UE NGAP IDs and naming the IE.
    #[test]
    fn a_reject_ie_in_a_request_with_a_failure_message_earns_that_failure() {
        let bytes = with_unknown_ie(&ucm_request(555, 3), 9999, Criticality::REJECT);
        match classify(&bytes) {
            UnknownIeVerdict::Reject {
                answer: RejectAnswer::Failure(failure),
                ue_associated: true,
            } => {
                let (ids, cause, cd) = ucm_failure(&failure);
                assert_eq!(ids, Some((555, 3)));
                assert_eq!(
                    cause,
                    Cause::Protocol(CauseProtocol(CauseProtocol::ABSTRACT_SYNTAX_ERROR_REJECT))
                );
                assert_eq!(reported(&cd), vec![(9999, CriticalityValue::Reject)]);
                assert_eq!(cd.procedure_code, Some(ID_UE_CONTEXT_MODIFICATION));
            }
            v => panic!("expected a UE Context Modification Failure, got {v:?}"),
        }
    }

    /// §10.3.4.2 Reject IE, a procedure without a failure message: an Error
    /// Indication, which for a UE-associated message carries its UE NGAP IDs
    /// (§8.7.5.2).
    #[test]
    fn a_reject_ie_in_a_ue_associated_message_without_a_failure_gets_an_error_indication_with_its_ids(
    ) {
        let bytes = with_unknown_ie(&dl_nas(555, 3), 9999, Criticality::REJECT);
        match classify(&bytes) {
            UnknownIeVerdict::Reject {
                answer: RejectAnswer::ErrorIndication(ei),
                ue_associated: true,
            } => {
                assert_eq!((ei.amf_ue_ngap_id, ei.ran_ue_ngap_id), (Some(555), Some(3)));
                assert_eq!(reported_ies(&ei), vec![(9999, CriticalityValue::Reject)]);
            }
            v => panic!("expected an Error Indication, got {v:?}"),
        }
    }

    /// §10.3.4.2 Ignore IE and Notify Sender, in a request: the verdict marks it
    /// as owed to the response, with the IE named; the Error Indication fallback
    /// carries the UE NGAP IDs.
    #[test]
    fn a_notify_ie_in_a_request_is_owed_to_its_response() {
        let bytes = with_unknown_ie(&ucm_request(555, 3), 9999, Criticality::NOTIFY);
        match classify(&bytes) {
            UnknownIeVerdict::Proceed {
                notify: Some(n), ..
            } => {
                assert!(n.initiating);
                assert_eq!(n.procedure_code, ID_UE_CONTEXT_MODIFICATION);
                assert_eq!(
                    reported(&n.diagnostics),
                    vec![(9999, CriticalityValue::Notify)]
                );
                assert_eq!(
                    (
                        n.error_indication.amf_ue_ngap_id,
                        n.error_indication.ran_ue_ngap_id
                    ),
                    (Some(555), Some(3))
                );
            }
            v => panic!("expected Proceed with a report, got {v:?}"),
        }
    }

    /// An ignore IE beside a reject one is not reported (§10.3.4.2 reports
    /// "Reject IE" and "Ignore IE and Notify Sender" IEs only).
    #[test]
    fn an_ignore_ie_beside_a_reject_one_is_not_reported() {
        let bytes = with_unknown_ie(&ucm_request(555, 3), 9998, Criticality::IGNORE);
        let bytes = with_unknown_ie(&bytes, 9999, Criticality::REJECT);
        match classify(&bytes) {
            UnknownIeVerdict::Reject {
                answer: RejectAnswer::Failure(failure),
                ..
            } => assert_eq!(
                reported(&ucm_failure(&failure).2),
                vec![(9999, CriticalityValue::Reject)]
            ),
            v => panic!("expected a failure, got {v:?}"),
        }
    }

    /// The Criticality Diagnostics IE goes where the ASN.1 lists it: before
    /// User Location Information in a PDU SESSION RESOURCE SETUP RESPONSE
    /// (TS 38.413 §9.2.1.2), last in a UE CONTEXT MODIFICATION RESPONSE; a
    /// message of another procedure, or one already carrying the IE, is refused.
    #[test]
    fn criticality_diagnostics_are_inserted_in_asn1_order() {
        use nextgsim_ngap::codec::generated::{
            PDUSessionResourceSetupResponse, PDUSessionResourceSetupResponseProtocolIEs,
            PDUSessionResourceSetupResponseProtocolIEs_Entry as E,
            PDUSessionResourceSetupResponseProtocolIEs_EntryValue as V, ProcedureCode,
            SuccessfulOutcome, SuccessfulOutcomeValue, UplinkNASTransportProtocolIEs_EntryValue,
            ID_PDU_SESSION_RESOURCE_SETUP, ID_USER_LOCATION_INFORMATION,
        };
        use nextgsim_ngap::procedures::initial_ue_message::{NrCgi, Tai, UserLocationInfoNr};
        use nextgsim_ngap::procedures::nas_transport::{
            encode_uplink_nas_transport, UplinkNasTransportParams,
        };
        use nextgsim_ngap::procedures::ue_context_modification::encode_ue_context_modification_response;
        // A User Location Information value, lifted from an Uplink NAS Transport.
        let ul = encode_uplink_nas_transport(&UplinkNasTransportParams {
            amf_ue_ngap_id: 1,
            ran_ue_ngap_id: 1,
            nas_pdu: vec![0x7E],
            user_location_info: UserLocationInfoNr {
                nr_cgi: NrCgi {
                    plmn_identity: [0x00, 0xF1, 0x10],
                    nr_cell_identity: 1,
                },
                tai: Tai {
                    plmn_identity: [0x00, 0xF1, 0x10],
                    tac: [0, 0, 1],
                },
                time_stamp: None,
            },
        })
        .unwrap();
        let NGAP_PDU::InitiatingMessage(m) = decode_ngap_pdu(&ul).unwrap() else {
            unreachable!()
        };
        let InitiatingMessageValue::Id_UplinkNASTransport(m) = m.value else {
            unreachable!()
        };
        let uli = m
            .protocol_i_es
            .0
            .into_iter()
            .find_map(|e| match e.value {
                UplinkNASTransportProtocolIEs_EntryValue::Id_UserLocationInformation(u) => Some(u),
                _ => None,
            })
            .expect("ULI");
        let entry = |id, value| E {
            id: ProtocolIE_ID(id),
            criticality: Criticality(Criticality::IGNORE),
            value,
        };
        let response = encode_ngap_pdu(&NGAP_PDU::SuccessfulOutcome(SuccessfulOutcome {
            procedure_code: ProcedureCode(ID_PDU_SESSION_RESOURCE_SETUP),
            criticality: Criticality(Criticality::REJECT),
            value: SuccessfulOutcomeValue::Id_PDUSessionResourceSetup(
                PDUSessionResourceSetupResponse {
                    protocol_i_es: PDUSessionResourceSetupResponseProtocolIEs(vec![
                        entry(ID_AMF_UE_NGAP_ID, V::Id_AMF_UE_NGAP_ID(AMF_UE_NGAP_ID(1))),
                        entry(ID_RAN_UE_NGAP_ID, V::Id_RAN_UE_NGAP_ID(RAN_UE_NGAP_ID(1))),
                        entry(
                            ID_USER_LOCATION_INFORMATION,
                            V::Id_UserLocationInformation(uli),
                        ),
                    ]),
                },
            ),
        }))
        .unwrap();
        let diag = CriticalityDiagnosticsInfo {
            ies_criticality_diagnostics: vec![IeCriticalityDiagnosticsItem {
                ie_criticality: CriticalityValue::Notify,
                ie_id: 9999,
                type_of_error: TypeOfErrorValue::NotUnderstood,
            }],
            ..Default::default()
        };
        let ids = |bytes: &[u8]| -> Vec<u16> {
            match decode_ngap_pdu(bytes).unwrap() {
                NGAP_PDU::SuccessfulOutcome(o) => match o.value {
                    SuccessfulOutcomeValue::Id_PDUSessionResourceSetup(m) => {
                        m.protocol_i_es.0.iter().map(|e| e.id.0).collect()
                    }
                    SuccessfulOutcomeValue::Id_UEContextModification(m) => {
                        m.protocol_i_es.0.iter().map(|e| e.id.0).collect()
                    }
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            }
        };
        let with = with_criticality_diagnostics(&response, ID_PDU_SESSION_RESOURCE_SETUP, &diag)
            .expect("attached");
        assert_eq!(ids(&with), vec![10, 85, 19, 121]);
        assert!(
            with_criticality_diagnostics(&with, ID_PDU_SESSION_RESOURCE_SETUP, &diag).is_none(),
            "a second Criticality Diagnostics is refused"
        );
        assert!(
            with_criticality_diagnostics(&response, ID_UE_CONTEXT_MODIFICATION, &diag).is_none(),
            "another procedure's response is not this request's"
        );
        let ucm = encode_ue_context_modification_response(1, 1).unwrap();
        let with = with_criticality_diagnostics(&ucm, ID_UE_CONTEXT_MODIFICATION, &diag).unwrap();
        assert_eq!(*ids(&with).last().unwrap(), 19);
    }

    fn reported_ies(p: &ErrorIndicationParams) -> Vec<(u16, CriticalityValue)> {
        // Round-trip through the wire form so the test checks what the AMF sees.
        let bytes = encode_error_indication(p).expect("encodes");
        let d = decode_error_indication(&bytes).expect("decodes");
        d.criticality_diagnostics
            .expect("diagnostics")
            .ies_criticality_diagnostics
            .iter()
            .map(|i| (i.ie_id, i.ie_criticality))
            .collect()
    }

    #[test]
    fn a_clean_message_is_routed_unchanged() {
        let (_, original) = overload_start_with_unknown_ie(9999, Criticality::IGNORE);
        let bytes = encode_ngap_pdu(&original).unwrap();
        assert!(matches!(classify(&bytes), UnknownIeVerdict::Clean));
    }

    #[test]
    fn an_unknown_ignore_ie_is_dropped_silently() {
        let (bytes, original) = overload_start_with_unknown_ie(9999, Criticality::IGNORE);
        match classify(&bytes) {
            UnknownIeVerdict::Proceed { bytes, notify } => {
                assert_eq!(decode_ngap_pdu(&bytes).unwrap(), original);
                assert!(notify.is_none(), "ignore is not reported (§10.3.4.1)");
            }
            v => panic!("expected Proceed, got {v:?}"),
        }
    }

    #[test]
    fn an_unknown_notify_ie_is_dropped_and_reported() {
        let (bytes, original) = overload_start_with_unknown_ie(9999, Criticality::NOTIFY);
        match classify(&bytes) {
            UnknownIeVerdict::Proceed { bytes, notify } => {
                assert_eq!(decode_ngap_pdu(&bytes).unwrap(), original);
                let notify = notify.expect("notify must be reported");
                let report = &notify.error_indication;
                assert_eq!(reported_ies(report), vec![(9999, CriticalityValue::Notify)]);
                assert!(!notify.ue_associated);
                let diag = report.criticality_diagnostics.as_ref().unwrap();
                assert_eq!(diag.procedure_code, Some(22), "Overload Start");
                assert_eq!(
                    diag.triggering_message,
                    Some(TriggeringMessageValue::InitiatingMessage)
                );
            }
            v => panic!("expected Proceed, got {v:?}"),
        }
    }

    #[test]
    fn an_unknown_reject_ie_rejects_an_initiating_message_with_a_report() {
        // Overload Start is Class 2: no failure message, so an Error Indication.
        let (bytes, _) = overload_start_with_unknown_ie(9999, Criticality::REJECT);
        match classify(&bytes) {
            UnknownIeVerdict::Reject {
                answer: RejectAnswer::ErrorIndication(r),
                ue_associated: false,
            } => {
                assert_eq!(reported_ies(&r), vec![(9999, CriticalityValue::Reject)]);
            }
            v => panic!("expected Reject with a report, got {v:?}"),
        }
    }

    #[test]
    fn garbage_is_undecodable() {
        assert!(matches!(
            classify(&[0xFF, 0x00]),
            UnknownIeVerdict::Undecodable
        ));
    }
}
