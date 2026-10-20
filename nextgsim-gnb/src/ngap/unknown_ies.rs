//! TS 38.413 §10.3.4 for messages the gNB receives from the AMF.
//!
//! [`decode_ngap_pdu`] refuses a whole message when one protocol IE carries an
//! id this schema does not know, so a newer-release AMF that adds an IE would
//! have every such message answered as undecodable. §10.3.4 says otherwise: an
//! unknown IE is handled by the criticality the sender gave it.
//!
//! * **ignore** (§10.3.4.1): drop the IE and process the rest of the message.
//! * **ignore and notify**: the same, and tell the sender which IEs were not
//!   understood (an ERROR INDICATION with Criticality Diagnostics, §10.3.4.2).
//! * **reject**: do not process the message. For an initiating message the
//!   procedure is not executed and the sender is told (ERROR INDICATION); for a
//!   response the procedure is terminated and local error handling applies, with
//!   no message back (§10.3.4.2).
//!
//! The classification is pure so it can be tested without a task: the gate in
//! `NgapTask::handle_ngap_pdu` acts on its verdict.

use nextgsim_ngap::codec::{
    decode_ngap_pdu, decode_ngap_pdu_with_unknown_ies, encode_ngap_pdu, Criticality, UnknownIe,
    NGAP_PDU,
};
use nextgsim_ngap::procedures::error_indication::{
    error_indication_abstract_syntax_error, CriticalityDiagnosticsInfo, CriticalityValue,
    ErrorIndicationParams, IeCriticalityDiagnosticsItem, TriggeringMessageValue, TypeOfErrorValue,
};

/// What the gate does with one received PDU.
#[derive(Debug)]
pub(super) enum UnknownIeVerdict {
    /// Decodes under the default codec: route the original bytes unchanged.
    Clean,
    /// Does not decode even skipping unknown IEs: the existing undecodable
    /// handling (transfer-syntax-error Error Indication) applies.
    Undecodable,
    /// Every unknown IE is `ignore` or `notify`: route `bytes`, the message
    /// re-encoded without them, and report `notify` (empty if none).
    Proceed {
        bytes: Vec<u8>,
        notify: Option<ErrorIndicationParams>,
    },
    /// At least one unknown IE is `reject` (or carries no criticality, read as
    /// reject). `report` is the ERROR INDICATION to send for an initiating
    /// message, `None` for a response (local error handling only).
    Reject {
        report: Option<ErrorIndicationParams>,
    },
}

/// Classifies `bytes` per TS 38.413 §10.3.4.
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
    let is_reject = |ie: &UnknownIe| {
        !matches!(
            ie.criticality,
            Some(Criticality(Criticality::IGNORE)) | Some(Criticality(Criticality::NOTIFY))
        )
    };
    if unknown.iter().any(is_reject) {
        let report = matches!(pdu, NGAP_PDU::InitiatingMessage(_))
            .then(|| diagnostics(&pdu, &unknown, true));
        return UnknownIeVerdict::Reject { report };
    }
    let Ok(bytes) = encode_ngap_pdu(&pdu) else {
        return UnknownIeVerdict::Undecodable;
    };
    let notified: Vec<UnknownIe> = unknown
        .into_iter()
        .filter(|ie| ie.criticality == Some(Criticality(Criticality::NOTIFY)))
        .collect();
    let notify = (!notified.is_empty()).then(|| diagnostics(&pdu, &notified, false));
    UnknownIeVerdict::Proceed { bytes, notify }
}

/// An ERROR INDICATION naming `ies` as not understood (§9.3.1.3): cause
/// abstract-syntax-error-reject or -ignore-and-notify, with the procedure code,
/// triggering message and procedure criticality of `pdu`.
fn diagnostics(pdu: &NGAP_PDU, ies: &[UnknownIe], reject: bool) -> ErrorIndicationParams {
    let (procedure_code, triggering_message, procedure_criticality) = match pdu {
        NGAP_PDU::InitiatingMessage(m) => (
            m.procedure_code.0,
            TriggeringMessageValue::InitiatingMessage,
            m.criticality.0,
        ),
        NGAP_PDU::SuccessfulOutcome(m) => (
            m.procedure_code.0,
            TriggeringMessageValue::SuccessfulOutcome,
            m.criticality.0,
        ),
        NGAP_PDU::UnsuccessfulOutcome(m) => (
            m.procedure_code.0,
            TriggeringMessageValue::UnsuccessfulOutcome,
            m.criticality.0,
        ),
    };
    let mut params = error_indication_abstract_syntax_error(None, None, reject);
    params.criticality_diagnostics = Some(CriticalityDiagnosticsInfo {
        procedure_code: Some(procedure_code),
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
    });
    params
}

fn criticality_value(c: u8) -> CriticalityValue {
    match c {
        Criticality::IGNORE => CriticalityValue::Ignore,
        Criticality::NOTIFY => CriticalityValue::Notify,
        _ => CriticalityValue::Reject,
    }
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
                let report = notify.expect("notify must be reported");
                assert_eq!(
                    reported_ies(&report),
                    vec![(9999, CriticalityValue::Notify)]
                );
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
        let (bytes, _) = overload_start_with_unknown_ie(9999, Criticality::REJECT);
        match classify(&bytes) {
            UnknownIeVerdict::Reject { report: Some(r) } => {
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
