use super::*;
use crate::config::{MmeConfig, PagingDrx, SupportedTa};
use std::net::{IpAddr, Ipv4Addr};

pub(crate) fn cfg() -> EnbConfig {
    EnbConfig {
        enb_id: 0x19B,
        enb_id_type: EnbIdType::Macro,
        plmn: Plmn::new(999, 70, false),
        tac: 1,
        supported_tas: vec![],
        enb_name: Some("nextgsim-enb".into()),
        paging_drx: PagingDrx::V128,
        mme_configs: vec![MmeConfig {
            address: IpAddr::V4(Ipv4Addr::LOCALHOST),
            port: 36412,
        }],
        s1ap_ip: IpAddr::V4(Ipv4Addr::LOCALHOST),
        s1_setup_retry_secs: 10,
        s1_setup_response_timeout_ms: 5000,
    }
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn request_ies(pdu: &S1AP_PDU) -> &[S1SetupRequestProtocolIEs_Entry] {
    match pdu {
        S1AP_PDU::InitiatingMessage(InitiatingMessage {
            value: InitiatingMessageValue::Id_S1Setup(r),
            ..
        }) => &r.protocol_i_es.0,
        other => panic!("not an S1 SETUP REQUEST: {other:?}"),
    }
}

#[test]
fn request_round_trips_and_carries_the_configured_values() {
    let pdu = build_s1_setup_request(&cfg());
    let bytes = codec::encode_s1ap_pdu(&pdu).unwrap();
    let back = codec::decode_s1ap_pdu(&bytes).unwrap();
    assert_eq!(back, pdu);

    let ies = request_ies(&back);
    let ids: Vec<(u16, u8)> = ies.iter().map(|e| (e.id.0, e.criticality.0)).collect();
    // Message-table order and the object set's criticalities (§9.1.8.4).
    assert_eq!(
        ids,
        vec![
            (ID_GLOBAL_ENB_ID, Criticality::REJECT),
            (ID_E_N_BNAME, Criticality::IGNORE),
            (ID_SUPPORTED_T_AS, Criticality::REJECT),
            (ID_DEFAULT_PAGING_DRX, Criticality::IGNORE),
        ]
    );
    match &ies[0].value {
        S1SetupRequestProtocolIEs_EntryValue::Id_Global_ENB_ID(g) => {
            assert_eq!(g.plm_nidentity.0, vec![0x99, 0xF9, 0x07]);
            match &g.enb_id {
                ENB_ID::MacroENB_ID(b) => {
                    assert_eq!(b.0.len(), 20);
                    let v = b.0.iter().fold(0u32, |a, bit| (a << 1) | u32::from(*bit));
                    assert_eq!(v, 0x19B);
                }
                other => panic!("expected macro eNB ID, got {other:?}"),
            }
        }
        other => panic!("{other:?}"),
    }
    match &ies[2].value {
        S1SetupRequestProtocolIEs_EntryValue::Id_SupportedTAs(t) => {
            assert_eq!(t.0.len(), 1);
            assert_eq!(t.0[0].tac.0, vec![0x00, 0x01]);
            assert_eq!(t.0[0].broadcast_plm_ns.0[0].0, vec![0x99, 0xF9, 0x07]);
        }
        other => panic!("{other:?}"),
    }
    match &ies[3].value {
        S1SetupRequestProtocolIEs_EntryValue::Id_DefaultPagingDRX(d) => {
            assert_eq!(d.0, PagingDRX::V128);
        }
        other => panic!("{other:?}"),
    }
}

/// The request's wire form, derived field by field from X.691 (APER) and
/// checked against this encoder's output octet by octet:
///
/// ```text
/// 00 11 00 31      S1AP-PDU: ext 0, index 0 (initiatingMessage); procedureCode 17;
///                  criticality reject; open type length 49
/// 00 00 04         S1SetupRequest: ext 0; protocolIEs count 4 (SIZE 0..65535)
/// 00 3b 00 08      id 59 Global-ENB-ID, reject, length 8:
///   00               ext 0, iE-Extensions absent
///   99 f9 07         pLMNidentity 999-70 (TBCD, filler F)
///   00               ENB-ID: ext 0, index 0 (macroENB-ID)
///   00 19 b0         0x0019B as 20 bits, octet-aligned and left-justified (§16.10)
/// 00 3c 40 0e      id 60 eNBname, ignore, length 14:
///   05 80            ext 0; length-1 = 11 in 8 bits (SIZE 1..150)
///   6e 65 .. 62      "nextgsim-enb", 8-bit characters (aligned variant)
/// 00 40 00 07      id 64 SupportedTAs, reject, length 7:
///   00               count-1 = 0 (SIZE 1..256)
///   00 00 40         item ext 0, iE-Extensions absent, TAC 0x0001 (fixed 2 octets,
///                    not aligned, §17.6), BPLMNs count-1 = 0 in 3 bits
///   99 f9 07         broadcast PLMN 999-70
/// 00 89 40 01 40   id 137 DefaultPagingDRX, ignore, length 1: ext 0, v128 (2)
/// ```
#[test]
fn request_golden_bytes() {
    let bytes = codec::encode_s1ap_pdu(&build_s1_setup_request(&cfg())).unwrap();
    let h = hex(&bytes);
    // Header: initiating, procedure 17, criticality reject.
    assert_eq!(&bytes[..3], &[0x00, 0x11, 0x00]);
    // Four IEs.
    assert_eq!(&bytes[5..7], &[0x00, 0x04]);
    // Global eNB ID IE header, PLMN and the 20-bit macro eNB ID.
    assert_eq!(&bytes[7..10], &[0x00, 0x3b, 0x00]);
    assert_eq!(&bytes[12..15], &[0x99, 0xf9, 0x07]);
    assert_eq!(&bytes[16..19], &[0x00, 0x19, 0xb0]);
    // Default Paging DRX is the last IE: id 137, ignore, length 1, v128.
    assert_eq!(&bytes[bytes.len() - 5..], &[0x00, 0x89, 0x40, 0x01, 0x40]);
    assert_eq!(usize::from(bytes[3]), bytes.len() - 4, "open type length");
    assert_eq!(h, GOLDEN_REQUEST, "full request bytes");
}

const GOLDEN_REQUEST: &str = include_str!("golden_s1_setup_request.hex");

#[test]
fn every_enb_id_alternative_round_trips_at_its_width() {
    for (ty, id) in [
        (EnbIdType::Macro, (1u32 << 20) - 1),
        (EnbIdType::Home, (1u32 << 28) - 1),
        (EnbIdType::ShortMacro, (1u32 << 18) - 1),
        (EnbIdType::LongMacro, (1u32 << 21) - 1),
        (EnbIdType::Macro, 0),
        (EnbIdType::Home, 0x0ABCDEF),
    ] {
        let mut c = cfg();
        c.enb_id_type = ty;
        c.enb_id = id;
        let bytes = codec::encode_s1ap_pdu(&build_s1_setup_request(&c)).unwrap();
        let back = codec::decode_s1ap_pdu(&bytes).unwrap();
        let S1SetupRequestProtocolIEs_EntryValue::Id_Global_ENB_ID(g) =
            &request_ies(&back)[0].value
        else {
            panic!()
        };
        let bits = match &g.enb_id {
            ENB_ID::MacroENB_ID(b) => (EnbIdType::Macro, &b.0),
            ENB_ID::HomeENB_ID(b) => (EnbIdType::Home, &b.0),
            ENB_ID::Short_macroENB_ID(b) => (EnbIdType::ShortMacro, &b.0),
            ENB_ID::Long_macroENB_ID(b) => (EnbIdType::LongMacro, &b.0),
        };
        assert_eq!(bits.0, ty);
        assert_eq!(bits.1.len() as u32, ty.bits());
        let v = bits
            .1
            .iter()
            .fold(0u32, |a, bit| (a << 1) | u32::from(*bit));
        assert_eq!(v, id, "{ty:?}");
    }
}

#[test]
fn explicit_supported_tas_and_no_name() {
    let mut c = cfg();
    c.enb_name = None;
    c.supported_tas = vec![
        SupportedTa {
            tac: 0xBEEF,
            broadcast_plmns: vec![Plmn::new(1, 1, false), Plmn::new(310, 410, true)],
        },
        SupportedTa {
            tac: 2,
            broadcast_plmns: vec![Plmn::new(999, 70, false)],
        },
    ];
    let pdu = build_s1_setup_request(&c);
    let bytes = codec::encode_s1ap_pdu(&pdu).unwrap();
    let back = codec::decode_s1ap_pdu(&bytes).unwrap();
    let ies = request_ies(&back);
    assert_eq!(ies.len(), 3, "no eNBname IE without a name");
    let S1SetupRequestProtocolIEs_EntryValue::Id_SupportedTAs(t) = &ies[1].value else {
        panic!()
    };
    assert_eq!(t.0[0].tac.0, vec![0xBE, 0xEF]);
    assert_eq!(t.0[0].broadcast_plm_ns.0.len(), 2);
    assert_eq!(
        Plmn::decode(t.0[0].broadcast_plm_ns.0[1].0.clone().try_into().unwrap()),
        Plmn::new(310, 410, true)
    );
    assert_eq!(t.0[1].tac.0, vec![0x00, 0x02]);
}

// ---------------------------------------------------------------- replies

pub(crate) fn gummeis() -> ServedGUMMEIs {
    ServedGUMMEIs(vec![ServedGUMMEIsItem {
        served_plm_ns: ServedPLMNs(vec![TBCD_STRING(vec![0x99, 0xF9, 0x07])]),
        served_group_i_ds: ServedGroupIDs(vec![MME_Group_ID(vec![0x00, 0x02])]),
        served_mme_cs: ServedMMECs(vec![MME_Code(vec![0x01])]),
        ie_extensions: None,
    }])
}

pub(crate) fn response(with_gummeis: bool, with_capacity: bool) -> S1AP_PDU {
    let mut ies = vec![S1SetupResponseProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_MM_ENAME),
        criticality: Criticality(Criticality::IGNORE),
        value: S1SetupResponseProtocolIEs_EntryValue::Id_MMEname(MMEname("nextgcore-mme0".into())),
    }];
    if with_gummeis {
        ies.push(S1SetupResponseProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_SERVED_GUMME_IS),
            criticality: Criticality(Criticality::REJECT),
            value: S1SetupResponseProtocolIEs_EntryValue::Id_ServedGUMMEIs(gummeis()),
        });
    }
    if with_capacity {
        ies.push(S1SetupResponseProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_RELATIVE_MME_CAPACITY),
            criticality: Criticality(Criticality::IGNORE),
            value: S1SetupResponseProtocolIEs_EntryValue::Id_RelativeMMECapacity(
                RelativeMMECapacity(255),
            ),
        });
    }
    S1AP_PDU::SuccessfulOutcome(SuccessfulOutcome {
        procedure_code: ProcedureCode(ID_S1_SETUP),
        criticality: Criticality(Criticality::REJECT),
        value: SuccessfulOutcomeValue::Id_S1Setup(S1SetupResponse {
            protocol_i_es: S1SetupResponseProtocolIEs(ies),
        }),
    })
}

pub(crate) fn failure(ttw: Option<u8>) -> S1AP_PDU {
    let mut ies = vec![S1SetupFailureProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_CAUSE),
        criticality: Criticality(Criticality::IGNORE),
        value: S1SetupFailureProtocolIEs_EntryValue::Id_Cause(Cause::Misc(CauseMisc(
            CauseMisc::UNKNOWN_PLMN,
        ))),
    }];
    if let Some(t) = ttw {
        ies.push(S1SetupFailureProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_TIME_TO_WAIT),
            criticality: Criticality(Criticality::IGNORE),
            value: S1SetupFailureProtocolIEs_EntryValue::Id_TimeToWait(TimeToWait(t)),
        });
    }
    S1AP_PDU::UnsuccessfulOutcome(UnsuccessfulOutcome {
        procedure_code: ProcedureCode(ID_S1_SETUP),
        criticality: Criticality(Criticality::REJECT),
        value: UnsuccessfulOutcomeValue::Id_S1Setup(S1SetupFailure {
            protocol_i_es: S1SetupFailureProtocolIEs(ies),
        }),
    })
}

fn enc(pdu: &S1AP_PDU) -> Vec<u8> {
    codec::encode_s1ap_pdu(pdu).unwrap()
}

/// Append an IE the schema does not define to an encoded response/failure.
///
/// Layout (APER): `[kind][proc][crit][len] [ext/pad][count:2] entries...`, each
/// entry `[id:2][crit<<6][len][value]`. The message's open-type length and the
/// IE count are rewritten to match. Only single-octet lengths are handled,
/// which every fixture here stays within.
pub(crate) fn splice_unknown_ie(bytes: &[u8], id: u16, crit: u8, value: &[u8]) -> Vec<u8> {
    assert!(bytes[3] < 0x80, "fixture needs a short-form outer length");
    let mut out = bytes[..4].to_vec();
    let mut body = bytes[4..].to_vec();
    let count = u16::from_be_bytes([body[1], body[2]]) + 1;
    body[1..3].copy_from_slice(&count.to_be_bytes());
    body.extend_from_slice(&id.to_be_bytes());
    body.push(crit << 6);
    body.push(u8::try_from(value.len()).unwrap());
    body.extend_from_slice(value);
    out[3] = u8::try_from(body.len()).unwrap();
    out.extend_from_slice(&body);
    out
}

fn expect_accepted(o: SetupOutcome) -> (MmeInfo, Option<Vec<u8>>) {
    match o {
        SetupOutcome::Accepted {
            mme,
            error_indication,
        } => (mme, error_indication),
        other => panic!("expected Accepted, got {other:?}"),
    }
}

#[test]
fn response_is_accepted_and_stored() {
    let (mme, ei) = expect_accepted(handle_setup_reply(&enc(&response(true, true))));
    assert!(ei.is_none());
    assert_eq!(mme.mme_name.as_deref(), Some("nextgcore-mme0"));
    assert_eq!(mme.relative_capacity, Some(255));
    assert_eq!(
        mme.served_gummeis,
        vec![ServedGummei {
            plmns: vec![Plmn::new(999, 70, false)],
            group_ids: vec![2],
            mme_codes: vec![1],
        }]
    );
}

#[test]
fn response_missing_served_gummeis_is_unsuccessfully_terminated() {
    // §10.3.5: Served GUMMEIs is mandatory, criticality reject.
    match handle_setup_reply(&enc(&response(false, true))) {
        SetupOutcome::UnsuccessfullyTerminated { reason } => {
            assert!(reason.contains("Served GUMMEIs"), "{reason}");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn response_missing_relative_capacity_is_still_accepted() {
    // §10.3.5: mandatory but criticality ignore -- ignore that it is missing.
    let (mme, ei) = expect_accepted(handle_setup_reply(&enc(&response(true, false))));
    assert_eq!(mme.relative_capacity, None);
    assert!(ei.is_none(), "an ignore-criticality IE is not reported");
}

#[test]
fn unknown_ignore_ie_in_response_is_dropped_silently() {
    let bytes = splice_unknown_ie(&enc(&response(true, true)), 9999, Criticality::IGNORE, &[0]);
    let (mme, ei) = expect_accepted(handle_setup_reply(&bytes));
    assert_eq!(mme.relative_capacity, Some(255));
    assert!(ei.is_none());
}

fn decode_error_indication(bytes: &[u8]) -> (Cause, CriticalityDiagnostics) {
    let (cause, diag) = decode_error_indication_opt(bytes);
    (cause, diag.expect("Criticality Diagnostics present"))
}

fn decode_error_indication_opt(bytes: &[u8]) -> (Cause, Option<CriticalityDiagnostics>) {
    let S1AP_PDU::InitiatingMessage(InitiatingMessage {
        procedure_code,
        criticality,
        value: InitiatingMessageValue::Id_ErrorIndication(ei),
    }) = codec::decode_s1ap_pdu(bytes).unwrap()
    else {
        panic!("not an ERROR INDICATION")
    };
    assert_eq!(procedure_code.0, ID_ERROR_INDICATION);
    assert_eq!(criticality.0, Criticality::IGNORE);
    let mut cause = None;
    let mut diag = None;
    for ie in ei.protocol_i_es.0 {
        match ie.value {
            ErrorIndicationProtocolIEs_EntryValue::Id_Cause(c) => cause = Some(c),
            ErrorIndicationProtocolIEs_EntryValue::Id_CriticalityDiagnostics(d) => diag = Some(d),
            _ => {}
        }
    }
    (cause.unwrap(), diag)
}

#[test]
fn unknown_notify_ie_in_response_is_accepted_and_reported() {
    // §10.3.4.2: "If a response message is received containing ... IEs ...
    // marked with 'Ignore IE and Notify Sender' ... continue with the procedure
    // ... and initiate the Error Indication procedure."
    let bytes = splice_unknown_ie(
        &enc(&response(true, true)),
        9999,
        Criticality::NOTIFY,
        &[0xAB],
    );
    let (mme, ei) = expect_accepted(handle_setup_reply(&bytes));
    assert_eq!(mme.served_gummeis.len(), 1);
    let (cause, diag) = decode_error_indication(&ei.expect("an Error Indication is owed"));
    assert_eq!(
        cause,
        Cause::Protocol(CauseProtocol(
            CauseProtocol::ABSTRACT_SYNTAX_ERROR_IGNORE_AND_NOTIFY
        ))
    );
    assert_eq!(diag.procedure_code, Some(ProcedureCode(ID_S1_SETUP)));
    assert_eq!(
        diag.triggering_message,
        Some(TriggeringMessage(TriggeringMessage::SUCCESSFUL_OUTCOME))
    );
    assert_eq!(
        diag.procedure_criticality,
        Some(Criticality(Criticality::REJECT))
    );
    let list = diag.i_es_criticality_diagnostics.unwrap().0;
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].ie_id.0, 9999);
    assert_eq!(list[0].ie_criticality.0, Criticality::NOTIFY);
    assert_eq!(list[0].type_of_error.0, TypeOfError::NOT_UNDERSTOOD);
}

#[test]
fn unknown_reject_ie_in_response_unsuccessfully_terminates() {
    let bytes = splice_unknown_ie(&enc(&response(true, true)), 9999, Criticality::REJECT, &[0]);
    match handle_setup_reply(&bytes) {
        SetupOutcome::UnsuccessfullyTerminated { reason } => assert!(reason.contains("9999")),
        other => panic!("{other:?}"),
    }
}

#[test]
fn failure_with_and_without_time_to_wait() {
    match handle_setup_reply(&enc(&failure(Some(TimeToWait::V10S)))) {
        SetupOutcome::Failed {
            cause,
            time_to_wait,
            error_indication,
        } => {
            assert_eq!(cause, Some(Cause::Misc(CauseMisc(CauseMisc::UNKNOWN_PLMN))));
            assert_eq!(time_to_wait, Some(Duration::from_secs(10)));
            assert!(error_indication.is_none());
        }
        other => panic!("{other:?}"),
    }
    match handle_setup_reply(&enc(&failure(None))) {
        SetupOutcome::Failed { time_to_wait, .. } => assert_eq!(time_to_wait, None),
        other => panic!("{other:?}"),
    }
}

#[test]
fn unknown_notify_ie_in_failure_is_reported_as_unsuccessful_outcome() {
    let bytes = splice_unknown_ie(
        &enc(&failure(Some(TimeToWait::V1S))),
        4242,
        Criticality::NOTIFY,
        &[1, 2],
    );
    match handle_setup_reply(&bytes) {
        SetupOutcome::Failed {
            time_to_wait,
            error_indication,
            ..
        } => {
            assert_eq!(time_to_wait, Some(Duration::from_secs(1)));
            let (_, diag) = decode_error_indication(&error_indication.unwrap());
            assert_eq!(
                diag.triggering_message,
                Some(TriggeringMessage(TriggeringMessage::UNSUCCESSFULL_OUTCOME))
            );
            assert_eq!(
                diag.i_es_criticality_diagnostics.unwrap().0[0].ie_id.0,
                4242
            );
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn time_to_wait_values() {
    let all: Vec<_> = (0..=5)
        .map(|v| time_to_wait_duration(&TimeToWait(v)).unwrap().as_secs())
        .collect();
    assert_eq!(all, vec![1, 2, 5, 10, 20, 60]);
    assert_eq!(time_to_wait_duration(&TimeToWait(6)), None);
}

#[test]
fn unknown_procedure_code_follows_its_criticality() {
    // §10.3.4.1. 96 is beyond id-S1Removal (67).
    let pdu = |crit: u8| vec![0x00, 96, crit << 6, 0x01, 0x00];
    for (crit, cause) in [
        (
            Criticality::REJECT,
            Some(CauseProtocol::ABSTRACT_SYNTAX_ERROR_REJECT),
        ),
        (
            Criticality::NOTIFY,
            Some(CauseProtocol::ABSTRACT_SYNTAX_ERROR_IGNORE_AND_NOTIFY),
        ),
        (Criticality::IGNORE, None),
    ] {
        match handle_setup_reply(&pdu(crit)) {
            SetupOutcome::NotSetupOutcome {
                error_indication, ..
            } => match (cause, error_indication) {
                (None, None) => {}
                (Some(want), Some(ei)) => {
                    let (c, diag) = decode_error_indication(&ei);
                    assert_eq!(c, Cause::Protocol(CauseProtocol(want)));
                    // "shall include the Procedure Code IE, the Triggering
                    // Message IE, and the Procedure Criticality IE".
                    assert_eq!(diag.procedure_code, Some(ProcedureCode(96)));
                    assert_eq!(
                        diag.triggering_message,
                        Some(TriggeringMessage(TriggeringMessage::INITIATING_MESSAGE))
                    );
                    assert_eq!(diag.procedure_criticality, Some(Criticality(crit)));
                }
                (want, got) => panic!("crit {crit}: want {want:?}, got {got:?}"),
            },
            other => panic!("{other:?}"),
        }
    }
}

#[test]
fn undecodable_known_procedure_is_a_transfer_syntax_error() {
    // §10.2. A truncated S1 SETUP RESPONSE.
    let bytes = enc(&response(true, true));
    match handle_setup_reply(&bytes[..bytes.len() - 3]) {
        SetupOutcome::NotSetupOutcome {
            error_indication, ..
        } => {
            let (c, diag) = decode_error_indication_opt(&error_indication.unwrap());
            assert_eq!(
                c,
                Cause::Protocol(CauseProtocol(CauseProtocol::TRANSFER_SYNTAX_ERROR))
            );
            // Nothing in an undecodable message can be diagnosed reliably.
            assert!(diag.is_none());
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn another_procedure_is_not_a_setup_outcome_and_owes_nothing() {
    let ei = build_error_indication(Cause::Misc(CauseMisc(CauseMisc::UNSPECIFIED)), None, &[]);
    match handle_setup_reply(&enc(&ei)) {
        SetupOutcome::NotSetupOutcome {
            description,
            error_indication,
        } => {
            assert!(description.contains("procedure 15"), "{description}");
            assert!(error_indication.is_none());
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn peek_header_reads_real_pdus() {
    assert_eq!(
        peek_header(&enc(&build_s1_setup_request(&cfg()))),
        Some((0, ID_S1_SETUP, Criticality::REJECT))
    );
    assert_eq!(
        peek_header(&enc(&response(true, true))),
        Some((1, ID_S1_SETUP, Criticality::REJECT))
    );
    assert_eq!(
        peek_header(&enc(&failure(None))),
        Some((2, ID_S1_SETUP, Criticality::REJECT))
    );
    assert_eq!(peek_header(&[0x80, 0, 0]), None);
    assert_eq!(peek_header(&[0x00]), None);
}

#[test]
fn max_known_procedure_code_is_the_schemas_last() {
    // If a schema bump adds procedures, this constant must move with it.
    assert_eq!(MAX_KNOWN_PROCEDURE_CODE, 67);
}
