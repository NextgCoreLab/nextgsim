//! PDU Session Resource Notify (TS 38.413 §8.3.5)
//!
//! The NG-RAN-initiated, UE-associated procedure by which the gNB tells the AMF
//! that listed PDU session resources are **released** or **not fulfilled** at the
//! RAN, without the AMF having asked. It is the conformant way to report a
//! user-plane problem scoped to *individual PDU sessions*.
//!
//! # Why it matters that the granularity is per-session
//!
//! The alternative the gNB already had is `UE CONTEXT RELEASE REQUEST`, which
//! tears down the **whole UE context**. A UE with one session on a failed UPF and
//! another on a healthy one loses both. This procedure is what lets the gNB report
//! only what actually broke (see issue #91, whose interim over-releases
//! deliberately, and issue #98, which this closes).
//!
//! # Two lists, two meanings
//!
//! - **`PDU Session Resource Notify List`** — sessions still up, whose QoS flows
//!   have changed state. Each flow carries a `NotificationCause` of `fulfilled` or
//!   `notFulfilled`, and flows the RAN gave up on entirely appear in the same
//!   item's `qosFlowReleasedList` with a `Cause`. The session survives.
//! - **`PDU Session Resource Released List Not`** — sessions the RAN has
//!   **released**, each with a `Cause`. These are gone.
//!
//! Both lists are OPTIONAL in the ASN.1 but a message carrying neither says
//! nothing, so [`build_pdu_session_resource_notify`] refuses that rather than
//! putting an empty notification on the wire.
//!
//! Note the inner transfers are `OCTET STRING (CONTAINING ...)`, so they are APER
//! encoded separately and carried as opaque octets — the same shape as every other
//! NGAP PDU-session transfer in this crate.

use crate::codec::generated::*;
use crate::codec::{decode_aper, decode_ngap_pdu, encode_aper, encode_ngap_pdu, NgapCodecError};
use thiserror::Error;

/// Errors that can occur building or parsing a PDU Session Resource Notify.
#[derive(Debug, Error)]
pub enum PduSessionResourceNotifyError {
    /// Codec error during encoding/decoding
    #[error("Codec error: {0}")]
    CodecError(#[from] NgapCodecError),

    /// The message is not a PDU Session Resource Notify
    #[error("Not a PDU Session Resource Notify: {0}")]
    NotANotify(String),

    /// A mandatory IE is missing
    #[error("Missing mandatory IE: {0}")]
    MissingIe(&'static str),

    /// A field value is out of range or the message says nothing
    #[error("Invalid field value: {0}")]
    InvalidFieldValue(String),
}

/// Whether a QoS flow's requirements are being met at the RAN
/// (TS 38.413 §9.3.1.56 `NotificationCause`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotificationCauseValue {
    /// The RAN can meet the flow's QoS again
    Fulfilled,
    /// The RAN can no longer meet the flow's QoS
    NotFulfilled,
}

impl NotificationCauseValue {
    fn to_asn(self) -> NotificationCause {
        NotificationCause(match self {
            Self::Fulfilled => NotificationCause::FULFILLED,
            Self::NotFulfilled => NotificationCause::NOT_FULFILLED,
        })
    }

    fn from_asn(value: &NotificationCause) -> Result<Self, PduSessionResourceNotifyError> {
        match value.0 {
            NotificationCause::FULFILLED => Ok(Self::Fulfilled),
            NotificationCause::NOT_FULFILLED => Ok(Self::NotFulfilled),
            other => Err(PduSessionResourceNotifyError::InvalidFieldValue(format!(
                "unknown NotificationCause {other}"
            ))),
        }
    }
}

/// The NGAP `Cause` values this procedure reports, as the radio-network causes a
/// user-plane failure actually maps to (TS 38.413 §9.3.1.2).
///
/// A closed set rather than the full `Cause` CHOICE: what a gNB reports here is a
/// transport or radio problem it observed, and offering it the whole cause space
/// would invite reporting, say, a NAS cause for a broken GTP-U path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotifyCause {
    /// The transport resource (e.g. the GTP-U path to the UPF) is unavailable
    TransportResourceUnavailable,
    /// Radio resources are not available for the flow
    RadioResourcesNotAvailable,
    /// A radio-interface procedure failed
    FailureInRadioInterfaceProcedure,
    /// Unspecified radio-network cause
    RadioNetworkUnspecified,
}

impl NotifyCause {
    fn to_asn(self) -> Cause {
        match self {
            Self::TransportResourceUnavailable => Cause::Transport(CauseTransport(
                CauseTransport::TRANSPORT_RESOURCE_UNAVAILABLE,
            )),
            Self::RadioResourcesNotAvailable => Cause::RadioNetwork(CauseRadioNetwork(
                CauseRadioNetwork::RADIO_RESOURCES_NOT_AVAILABLE,
            )),
            Self::FailureInRadioInterfaceProcedure => Cause::RadioNetwork(CauseRadioNetwork(
                CauseRadioNetwork::FAILURE_IN_RADIO_INTERFACE_PROCEDURE,
            )),
            Self::RadioNetworkUnspecified => {
                Cause::RadioNetwork(CauseRadioNetwork(CauseRadioNetwork::UNSPECIFIED))
            }
        }
    }
}

/// One QoS flow's notified state within a surviving PDU session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NotifiedQosFlow {
    /// QoS Flow Identifier (0..63)
    pub qos_flow_identifier: u8,
    /// Whether the RAN is meeting this flow's QoS
    pub notification_cause: NotificationCauseValue,
}

/// One QoS flow the RAN released within a surviving PDU session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReleasedQosFlow {
    /// QoS Flow Identifier (0..63)
    pub qos_flow_identifier: u8,
    /// Why the flow was released
    pub cause: NotifyCause,
}

/// A PDU session that survives, with QoS flow state to report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NotifiedSession {
    /// PDU Session ID (0..255)
    pub pdu_session_id: u8,
    /// Flows whose fulfilment state changed
    pub notified_flows: Vec<NotifiedQosFlow>,
    /// Flows the RAN released while keeping the session
    pub released_flows: Vec<ReleasedQosFlow>,
}

/// A PDU session the RAN has released.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReleasedSession {
    /// PDU Session ID (0..255)
    pub pdu_session_id: u8,
    /// Why the session was released
    pub cause: NotifyCause,
}

/// Parameters for building a PDU Session Resource Notify.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PduSessionResourceNotifyParams {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
    /// Sessions that survive, with changed QoS flow state
    pub notified_sessions: Vec<NotifiedSession>,
    /// Sessions the RAN released
    pub released_sessions: Vec<ReleasedSession>,
}

/// Parsed PDU Session Resource Notify.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PduSessionResourceNotifyData {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
    /// Sessions that survive, with changed QoS flow state
    pub notified_sessions: Vec<NotifiedSession>,
    /// Sessions the RAN released
    pub released_sessions: Vec<ReleasedSession>,
}

/// The largest QoS Flow Identifier the ASN.1 type allows
/// (`QosFlowIdentifier ::= INTEGER (0..63)`, TS 38.413 §9.3.1.51).
pub const QOS_FLOW_IDENTIFIER_MAX: u8 = 63;

/// Builds a PDU Session Resource Notify (TS 38.413 §8.3.5).
///
/// Refuses a message with neither list populated. Both are OPTIONAL in the ASN.1,
/// so such a message encodes fine and tells the AMF nothing — it would be a
/// notification whose only effect is to consume an SCTP send and an AMF dispatch.
pub fn build_pdu_session_resource_notify(
    params: &PduSessionResourceNotifyParams,
) -> Result<NGAP_PDU, PduSessionResourceNotifyError> {
    if params.notified_sessions.is_empty() && params.released_sessions.is_empty() {
        return Err(PduSessionResourceNotifyError::InvalidFieldValue(
            "a Notify with neither a Notify List nor a Released List reports \
             nothing; both IEs are optional so it would encode, and the AMF would \
             have no action to take"
                .to_string(),
        ));
    }

    let mut protocol_ies = Vec::new();

    protocol_ies.push(PDUSessionResourceNotifyProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_AMF_UE_NGAP_ID),
        criticality: Criticality(Criticality::REJECT),
        value: PDUSessionResourceNotifyProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(AMF_UE_NGAP_ID(
            params.amf_ue_ngap_id,
        )),
    });

    protocol_ies.push(PDUSessionResourceNotifyProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_RAN_UE_NGAP_ID),
        criticality: Criticality(Criticality::REJECT),
        value: PDUSessionResourceNotifyProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(RAN_UE_NGAP_ID(
            params.ran_ue_ngap_id,
        )),
    });

    if !params.notified_sessions.is_empty() {
        let mut items = Vec::with_capacity(params.notified_sessions.len());
        for session in &params.notified_sessions {
            items.push(build_notify_item(session)?);
        }
        protocol_ies.push(PDUSessionResourceNotifyProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_PDU_SESSION_RESOURCE_NOTIFY_LIST),
            criticality: Criticality(Criticality::REJECT),
            value: PDUSessionResourceNotifyProtocolIEs_EntryValue::Id_PDUSessionResourceNotifyList(
                PDUSessionResourceNotifyList(items),
            ),
        });
    }

    if !params.released_sessions.is_empty() {
        let mut items = Vec::with_capacity(params.released_sessions.len());
        for session in &params.released_sessions {
            items.push(build_released_item(session)?);
        }
        protocol_ies.push(PDUSessionResourceNotifyProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_PDU_SESSION_RESOURCE_RELEASED_LIST_NOT),
            // IGNORE per the ASN.1 IE list, not REJECT: an AMF that cannot read
            // this list should still act on the Notify List rather than discard
            // the whole message.
            criticality: Criticality(Criticality::IGNORE),
            value:
                PDUSessionResourceNotifyProtocolIEs_EntryValue::Id_PDUSessionResourceReleasedListNot(
                    PDUSessionResourceReleasedListNot(items),
                ),
        });
    }

    let notify = PDUSessionResourceNotify {
        protocol_i_es: PDUSessionResourceNotifyProtocolIEs(protocol_ies),
    };

    Ok(NGAP_PDU::InitiatingMessage(InitiatingMessage {
        procedure_code: ProcedureCode(ID_PDU_SESSION_RESOURCE_NOTIFY),
        // IGNORE: the Notify is a report, not a request. TS 38.413 §8.3.5 has no
        // response message, so an AMF that does not understand the procedure must
        // not answer with an Error Indication that the gNB would then have to
        // interpret.
        criticality: Criticality(Criticality::IGNORE),
        value: InitiatingMessageValue::Id_PDUSessionResourceNotify(notify),
    }))
}

fn build_notify_item(
    session: &NotifiedSession,
) -> Result<PDUSessionResourceNotifyItem, PduSessionResourceNotifyError> {
    if session.notified_flows.is_empty() && session.released_flows.is_empty() {
        return Err(PduSessionResourceNotifyError::InvalidFieldValue(format!(
            "PDU session {} appears in the Notify List with no flow state to \
             report; omit the session rather than sending an empty transfer",
            session.pdu_session_id
        )));
    }

    let mut qos_flow_notify_list = Vec::with_capacity(session.notified_flows.len());
    for flow in &session.notified_flows {
        qos_flow_notify_list.push(QosFlowNotifyItem {
            qos_flow_identifier: qos_flow_identifier(flow.qos_flow_identifier)?,
            notification_cause: flow.notification_cause.to_asn(),
            ie_extensions: None,
        });
    }

    let mut qos_flow_released_list = Vec::with_capacity(session.released_flows.len());
    for flow in &session.released_flows {
        qos_flow_released_list.push(QosFlowWithCauseItem {
            qos_flow_identifier: qos_flow_identifier(flow.qos_flow_identifier)?,
            cause: flow.cause.to_asn(),
            ie_extensions: None,
        });
    }

    let transfer = PDUSessionResourceNotifyTransfer {
        // Both lists are SIZE (1..): an empty one is not encodable, so an absent
        // list is `None` rather than a zero-length `Some`.
        qos_flow_notify_list: if qos_flow_notify_list.is_empty() {
            None
        } else {
            Some(QosFlowNotifyList(qos_flow_notify_list))
        },
        qos_flow_released_list: if qos_flow_released_list.is_empty() {
            None
        } else {
            Some(QosFlowListWithCause(qos_flow_released_list))
        },
        ie_extensions: None,
    };

    Ok(PDUSessionResourceNotifyItem {
        pdu_session_id: PDUSessionID(session.pdu_session_id),
        pdu_session_resource_notify_transfer:
            PDUSessionResourceNotifyItemPDUSessionResourceNotifyTransfer(encode_aper(&transfer)?),
        ie_extensions: None,
    })
}

fn build_released_item(
    session: &ReleasedSession,
) -> Result<PDUSessionResourceReleasedItemNot, PduSessionResourceNotifyError> {
    let transfer = PDUSessionResourceNotifyReleasedTransfer {
        cause: session.cause.to_asn(),
        ie_extensions: None,
    };
    Ok(PDUSessionResourceReleasedItemNot {
        pdu_session_id: PDUSessionID(session.pdu_session_id),
        pdu_session_resource_notify_released_transfer:
            PDUSessionResourceReleasedItemNotPDUSessionResourceNotifyReleasedTransfer(encode_aper(
                &transfer,
            )?),
        ie_extensions: None,
    })
}

fn qos_flow_identifier(qfi: u8) -> Result<QosFlowIdentifier, PduSessionResourceNotifyError> {
    if qfi > QOS_FLOW_IDENTIFIER_MAX {
        return Err(PduSessionResourceNotifyError::InvalidFieldValue(format!(
            "QoS Flow Identifier {qfi} exceeds the ASN.1 maximum {QOS_FLOW_IDENTIFIER_MAX}"
        )));
    }
    Ok(QosFlowIdentifier(qfi))
}

/// Parses a PDU Session Resource Notify from an NGAP PDU.
pub fn parse_pdu_session_resource_notify(
    pdu: &NGAP_PDU,
) -> Result<PduSessionResourceNotifyData, PduSessionResourceNotifyError> {
    let NGAP_PDU::InitiatingMessage(initiating) = pdu else {
        return Err(PduSessionResourceNotifyError::NotANotify(
            "not an initiating message".to_string(),
        ));
    };
    let InitiatingMessageValue::Id_PDUSessionResourceNotify(notify) = &initiating.value else {
        return Err(PduSessionResourceNotifyError::NotANotify(format!(
            "procedure code {}",
            initiating.procedure_code.0
        )));
    };

    let mut amf_ue_ngap_id = None;
    let mut ran_ue_ngap_id = None;
    let mut notified_sessions = Vec::new();
    let mut released_sessions = Vec::new();

    for ie in &notify.protocol_i_es.0 {
        match &ie.value {
            PDUSessionResourceNotifyProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(id) => {
                amf_ue_ngap_id = Some(id.0);
            }
            PDUSessionResourceNotifyProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(id) => {
                ran_ue_ngap_id = Some(id.0);
            }
            PDUSessionResourceNotifyProtocolIEs_EntryValue::Id_PDUSessionResourceNotifyList(
                list,
            ) => {
                for item in &list.0 {
                    notified_sessions.push(parse_notify_item(item)?);
                }
            }
            PDUSessionResourceNotifyProtocolIEs_EntryValue::Id_PDUSessionResourceReleasedListNot(
                list,
            ) => {
                for item in &list.0 {
                    released_sessions.push(parse_released_item(item)?);
                }
            }
            // UserLocationInformation is optional and not modelled here; a Notify
            // that carries it is still a valid Notify.
            PDUSessionResourceNotifyProtocolIEs_EntryValue::Id_UserLocationInformation(_) => {}
        }
    }

    Ok(PduSessionResourceNotifyData {
        amf_ue_ngap_id: amf_ue_ngap_id
            .ok_or(PduSessionResourceNotifyError::MissingIe("AMF-UE-NGAP-ID"))?,
        ran_ue_ngap_id: ran_ue_ngap_id
            .ok_or(PduSessionResourceNotifyError::MissingIe("RAN-UE-NGAP-ID"))?,
        notified_sessions,
        released_sessions,
    })
}

fn parse_notify_item(
    item: &PDUSessionResourceNotifyItem,
) -> Result<NotifiedSession, PduSessionResourceNotifyError> {
    let transfer: PDUSessionResourceNotifyTransfer =
        decode_aper(&item.pdu_session_resource_notify_transfer.0)?;

    let mut notified_flows = Vec::new();
    if let Some(list) = transfer.qos_flow_notify_list.as_ref() {
        for flow in &list.0 {
            notified_flows.push(NotifiedQosFlow {
                qos_flow_identifier: flow.qos_flow_identifier.0,
                notification_cause: NotificationCauseValue::from_asn(&flow.notification_cause)?,
            });
        }
    }

    let mut released_flows = Vec::new();
    if let Some(list) = transfer.qos_flow_released_list.as_ref() {
        for flow in &list.0 {
            released_flows.push(ReleasedQosFlow {
                qos_flow_identifier: flow.qos_flow_identifier.0,
                cause: cause_from_asn(&flow.cause),
            });
        }
    }

    Ok(NotifiedSession {
        pdu_session_id: item.pdu_session_id.0,
        notified_flows,
        released_flows,
    })
}

fn parse_released_item(
    item: &PDUSessionResourceReleasedItemNot,
) -> Result<ReleasedSession, PduSessionResourceNotifyError> {
    let transfer: PDUSessionResourceNotifyReleasedTransfer =
        decode_aper(&item.pdu_session_resource_notify_released_transfer.0)?;
    Ok(ReleasedSession {
        pdu_session_id: item.pdu_session_id.0,
        cause: cause_from_asn(&transfer.cause),
    })
}

/// Maps a decoded `Cause` back onto [`NotifyCause`].
///
/// Lossy on purpose, and it does NOT fail: a peer may legitimately send any
/// `Cause` value, and refusing to parse a Notify because its cause is outside the
/// set this gNB emits would discard a report the AMF needs to act on. An
/// unrecognised cause becomes [`NotifyCause::RadioNetworkUnspecified`], which is
/// what "something went wrong at the RAN and we cannot say more" means.
fn cause_from_asn(cause: &Cause) -> NotifyCause {
    match cause {
        Cause::Transport(t) if t.0 == CauseTransport::TRANSPORT_RESOURCE_UNAVAILABLE => {
            NotifyCause::TransportResourceUnavailable
        }
        Cause::RadioNetwork(r) if r.0 == CauseRadioNetwork::RADIO_RESOURCES_NOT_AVAILABLE => {
            NotifyCause::RadioResourcesNotAvailable
        }
        Cause::RadioNetwork(r)
            if r.0 == CauseRadioNetwork::FAILURE_IN_RADIO_INTERFACE_PROCEDURE =>
        {
            NotifyCause::FailureInRadioInterfaceProcedure
        }
        _ => NotifyCause::RadioNetworkUnspecified,
    }
}

/// Builds and encodes a PDU Session Resource Notify to bytes.
pub fn encode_pdu_session_resource_notify(
    params: &PduSessionResourceNotifyParams,
) -> Result<Vec<u8>, PduSessionResourceNotifyError> {
    let pdu = build_pdu_session_resource_notify(params)?;
    Ok(encode_ngap_pdu(&pdu)?)
}

/// Decodes and parses a PDU Session Resource Notify from bytes.
pub fn decode_pdu_session_resource_notify(
    bytes: &[u8],
) -> Result<PduSessionResourceNotifyData, PduSessionResourceNotifyError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_pdu_session_resource_notify(&pdu)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> PduSessionResourceNotifyParams {
        PduSessionResourceNotifyParams {
            amf_ue_ngap_id: 0x0102_0304_05,
            ran_ue_ngap_id: 7,
            notified_sessions: vec![NotifiedSession {
                pdu_session_id: 5,
                notified_flows: vec![NotifiedQosFlow {
                    qos_flow_identifier: 9,
                    notification_cause: NotificationCauseValue::NotFulfilled,
                }],
                released_flows: vec![ReleasedQosFlow {
                    qos_flow_identifier: 10,
                    cause: NotifyCause::TransportResourceUnavailable,
                }],
            }],
            released_sessions: vec![ReleasedSession {
                pdu_session_id: 6,
                cause: NotifyCause::TransportResourceUnavailable,
            }],
        }
    }

    #[test]
    fn a_notify_round_trips_both_lists() {
        let bytes = encode_pdu_session_resource_notify(&params()).expect("encodes");
        let decoded = decode_pdu_session_resource_notify(&bytes).expect("decodes");

        assert_eq!(decoded.amf_ue_ngap_id, 0x0102_0304_05);
        assert_eq!(decoded.ran_ue_ngap_id, 7);

        assert_eq!(decoded.notified_sessions.len(), 1);
        let notified = &decoded.notified_sessions[0];
        assert_eq!(notified.pdu_session_id, 5);
        assert_eq!(notified.notified_flows.len(), 1);
        assert_eq!(notified.notified_flows[0].qos_flow_identifier, 9);
        assert_eq!(
            notified.notified_flows[0].notification_cause,
            NotificationCauseValue::NotFulfilled
        );
        assert_eq!(notified.released_flows.len(), 1);
        assert_eq!(notified.released_flows[0].qos_flow_identifier, 10);
        assert_eq!(
            notified.released_flows[0].cause,
            NotifyCause::TransportResourceUnavailable
        );

        assert_eq!(decoded.released_sessions.len(), 1);
        assert_eq!(decoded.released_sessions[0].pdu_session_id, 6);
        assert_eq!(
            decoded.released_sessions[0].cause,
            NotifyCause::TransportResourceUnavailable
        );
    }

    /// The procedure code and criticality are what let an AMF dispatch this at
    /// all, so they are asserted on the DECODED PDU rather than assumed.
    #[test]
    fn the_notify_is_an_initiating_message_with_procedure_code_30_and_ignore_criticality() {
        let pdu = build_pdu_session_resource_notify(&params()).expect("builds");
        let NGAP_PDU::InitiatingMessage(initiating) = &pdu else {
            panic!("must be an initiating message");
        };
        assert_eq!(
            initiating.procedure_code.0, ID_PDU_SESSION_RESOURCE_NOTIFY,
            "TS 38.413 id-PDUSessionResourceNotify is 30"
        );
        assert_eq!(initiating.procedure_code.0, 30);
        assert_eq!(
            initiating.criticality.0,
            Criticality::IGNORE,
            "the Notify is a report with no response message, so an AMF that does \
             not understand it must not answer with an Error Indication"
        );
    }

    /// Either list alone is a valid Notify: a session can be released without any
    /// surviving session having changed, and vice versa.
    #[test]
    fn either_list_alone_is_a_valid_notify() {
        let released_only = PduSessionResourceNotifyParams {
            notified_sessions: Vec::new(),
            ..params()
        };
        let decoded = decode_pdu_session_resource_notify(
            &encode_pdu_session_resource_notify(&released_only).expect("encodes"),
        )
        .expect("decodes");
        assert!(decoded.notified_sessions.is_empty());
        assert_eq!(decoded.released_sessions.len(), 1);

        let notified_only = PduSessionResourceNotifyParams {
            released_sessions: Vec::new(),
            ..params()
        };
        let decoded = decode_pdu_session_resource_notify(
            &encode_pdu_session_resource_notify(&notified_only).expect("encodes"),
        )
        .expect("decodes");
        assert_eq!(decoded.notified_sessions.len(), 1);
        assert!(decoded.released_sessions.is_empty());
    }

    /// A Notify with neither list is refused. Both IEs are OPTIONAL so it would
    /// encode; it would just tell the AMF nothing.
    #[test]
    fn a_notify_with_neither_list_is_refused() {
        let empty = PduSessionResourceNotifyParams {
            notified_sessions: Vec::new(),
            released_sessions: Vec::new(),
            ..params()
        };
        assert!(encode_pdu_session_resource_notify(&empty).is_err());
    }

    /// A session in the Notify List with no flow state is refused, for the same
    /// reason: the inner transfer's two lists are both optional, so it encodes and
    /// says nothing.
    #[test]
    fn a_notified_session_with_no_flow_state_is_refused() {
        let hollow = PduSessionResourceNotifyParams {
            notified_sessions: vec![NotifiedSession {
                pdu_session_id: 5,
                notified_flows: Vec::new(),
                released_flows: Vec::new(),
            }],
            released_sessions: Vec::new(),
            ..params()
        };
        assert!(encode_pdu_session_resource_notify(&hollow).is_err());
    }

    #[test]
    fn a_qos_flow_identifier_above_63_is_refused() {
        let bad = PduSessionResourceNotifyParams {
            notified_sessions: vec![NotifiedSession {
                pdu_session_id: 5,
                notified_flows: vec![NotifiedQosFlow {
                    qos_flow_identifier: QOS_FLOW_IDENTIFIER_MAX + 1,
                    notification_cause: NotificationCauseValue::Fulfilled,
                }],
                released_flows: Vec::new(),
            }],
            released_sessions: Vec::new(),
            ..params()
        };
        assert!(encode_pdu_session_resource_notify(&bad).is_err());
    }

    /// `fulfilled` and `notFulfilled` are opposite meanings on one wire field, so
    /// a codec that dropped the value would still round trip one of them.
    #[test]
    fn both_notification_causes_survive_the_wire() {
        for cause in [
            NotificationCauseValue::Fulfilled,
            NotificationCauseValue::NotFulfilled,
        ] {
            let p = PduSessionResourceNotifyParams {
                notified_sessions: vec![NotifiedSession {
                    pdu_session_id: 1,
                    notified_flows: vec![NotifiedQosFlow {
                        qos_flow_identifier: 0,
                        notification_cause: cause,
                    }],
                    released_flows: Vec::new(),
                }],
                released_sessions: Vec::new(),
                ..params()
            };
            let decoded = decode_pdu_session_resource_notify(
                &encode_pdu_session_resource_notify(&p).expect("encodes"),
            )
            .expect("decodes");
            assert_eq!(
                decoded.notified_sessions[0].notified_flows[0].notification_cause, cause,
                "{cause:?} must survive"
            );
        }
    }

    /// Parsing something that is not a Notify is refused rather than yielding an
    /// empty Notify, which a caller would act on.
    #[test]
    fn a_non_notify_pdu_is_refused() {
        use crate::procedures::ng_setup::RadioNetworkCause;
        use crate::procedures::ue_context_release::{
            build_ue_context_release_request, UeContextReleaseCause, UeContextReleaseRequestParams,
        };
        let other = build_ue_context_release_request(&UeContextReleaseRequestParams {
            amf_ue_ngap_id: 1,
            ran_ue_ngap_id: 1,
            cause: UeContextReleaseCause::RadioNetwork(RadioNetworkCause::Unspecified),
        })
        .expect("builds");
        assert!(parse_pdu_session_resource_notify(&other).is_err());
    }

    /// A cause outside the set this gNB emits must not make the whole Notify
    /// unparseable: the report still names the sessions the AMF has to act on.
    #[test]
    fn an_unrecognised_cause_degrades_to_unspecified_rather_than_failing_the_parse() {
        assert_eq!(
            cause_from_asn(&Cause::Nas(CauseNas(CauseNas::NORMAL_RELEASE))),
            NotifyCause::RadioNetworkUnspecified
        );
    }

    /// The canonical Notify's exact bytes, pinned so a codec change cannot alter
    /// the wire silently — and, more importantly, so **nextgcore's independent
    /// hand-written NGAP parser can be fed the same octets** (see
    /// `specs/ngap-pdu-session-resource-notify.md`). A self round-trip through one
    /// codec passes however wrong both halves of it are; two independent
    /// implementations agreeing on these octets does not.
    ///
    /// The identifying structure is hand-verified against TS 38.413 rather than
    /// merely captured:
    ///
    /// ```text
    /// 00        NGAP-PDU CHOICE, initiatingMessage (index 0)
    /// 1e        procedureCode 30 = id-PDUSessionResourceNotify
    /// 40        criticality: ignore (0b01 in the top two bits)
    /// 23        open-type length: 35 octets follow
    /// ...
    /// 000a 00 02 002a    IE 10  = id-AMF-UE-NGAP-ID, reject, len 2, value 42
    /// 0055 00 02 0007    IE 85  = id-RAN-UE-NGAP-ID, reject, len 2, value 7
    /// 0042 00 07 ...     IE 66  = id-PDUSessionResourceNotifyList, reject, len 7
    ///        00 05 03 400128   1 item: PSI 5, 3-octet transfer 400128
    /// 0043 40 05 ...     IE 67  = id-PDUSessionResourceReleasedListNot, IGNORE, len 5
    ///        00 06 01 08       1 item: PSI 6, 1-octet transfer 08
    /// ```
    ///
    /// The two IE criticalities differ on purpose and are visible here: `00`
    /// (reject) for the Notify List, `40` (ignore) for the Released List, exactly
    /// as the ASN.1 IE list specifies.
    ///
    /// The innermost transfer bit packing (`400128`, `08`) is **not** hand-derived
    /// — it is cross-checked by nextgcore's parser, which is the honest evidence
    /// for it.
    const GOLDEN_NOTIFY_PDU: &str =
        "001e4023000004000a0002002a0055000200070042000700000503400128004340050000060108";

    /// The canonical Notify that [`GOLDEN_NOTIFY_PDU`] encodes.
    ///
    /// Deliberately exercises BOTH lists in one message and a `notFulfilled`
    /// cause, so a codec that dropped either list or defaulted the notification
    /// cause would move the bytes.
    fn golden_params() -> PduSessionResourceNotifyParams {
        PduSessionResourceNotifyParams {
            amf_ue_ngap_id: 42,
            ran_ue_ngap_id: 7,
            notified_sessions: vec![NotifiedSession {
                pdu_session_id: 5,
                notified_flows: vec![NotifiedQosFlow {
                    qos_flow_identifier: 9,
                    notification_cause: NotificationCauseValue::NotFulfilled,
                }],
                released_flows: Vec::new(),
            }],
            released_sessions: vec![ReleasedSession {
                pdu_session_id: 6,
                cause: NotifyCause::TransportResourceUnavailable,
            }],
        }
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn the_canonical_notify_encodes_to_the_golden_bytes() {
        let bytes = encode_pdu_session_resource_notify(&golden_params()).expect("encodes");
        assert_eq!(
            hex(&bytes),
            GOLDEN_NOTIFY_PDU,
            "the wire bytes changed; nextgcore's cross-decode test pins the SAME \
             string, so update both or the two products no longer agree"
        );
    }

    /// The golden bytes must also decode back, so the constant cannot drift into
    /// something this codec itself rejects.
    #[test]
    fn the_golden_bytes_decode_to_the_canonical_notify() {
        let bytes: Vec<u8> = (0..GOLDEN_NOTIFY_PDU.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&GOLDEN_NOTIFY_PDU[i..i + 2], 16).expect("hex"))
            .collect();
        let decoded = decode_pdu_session_resource_notify(&bytes).expect("decodes");
        let expected = golden_params();
        assert_eq!(decoded.amf_ue_ngap_id, expected.amf_ue_ngap_id);
        assert_eq!(decoded.ran_ue_ngap_id, expected.ran_ue_ngap_id);
        assert_eq!(decoded.notified_sessions, expected.notified_sessions);
        assert_eq!(decoded.released_sessions, expected.released_sessions);
    }
}
