use super::*;

fn location() -> UeLocation {
    UeLocation {
        plmn: Plmn {
            mcc: 999,
            mnc: 70,
            long_mnc: false,
        },
        tac: 1,
        cell_identity: 0x19B01,
    }
}

fn ies_of_initial_ue(pdu: &S1AP_PDU) -> Vec<(u16, u8)> {
    let S1AP_PDU::InitiatingMessage(InitiatingMessage {
        value: InitiatingMessageValue::Id_initialUEMessage(m),
        ..
    }) = pdu
    else {
        panic!("not an Initial UE Message");
    };
    m.protocol_i_es
        .0
        .iter()
        .map(|ie| (ie.id.0, ie.criticality.0))
        .collect()
}

#[test]
fn initial_ue_message_has_table_order_and_criticalities() {
    let pdu = build_initial_ue_message(7, &[0x07, 0x41], &location(), 3, Some((1, 0xC000_0001)));
    let bytes = codec::encode_s1ap_pdu(&pdu).unwrap();
    let back = codec::decode_s1ap_pdu(&bytes).unwrap();
    // §9.1.7.1 / InitialUEMessage-IEs: eNB UE S1AP ID (reject), NAS-PDU
    // (reject), TAI (reject), E-UTRAN CGI (ignore), RRC Establishment Cause
    // (ignore), S-TMSI (reject).
    assert_eq!(
        ies_of_initial_ue(&back),
        vec![
            (ID_E_NB_UE_S1AP_ID, Criticality::REJECT),
            (ID_NAS_PDU, Criticality::REJECT),
            (ID_TAI, Criticality::REJECT),
            (ID_EUTRAN_CGI, Criticality::IGNORE),
            (ID_RRC_ESTABLISHMENT_CAUSE, Criticality::IGNORE),
            (ID_S_TMSI, Criticality::REJECT),
        ]
    );
    // Procedure 12, criticality ignore (§9.3.2), initiating message.
    assert_eq!(
        crate::s1ap::peek_header(&bytes),
        Some((0, 12, Criticality::IGNORE))
    );
}

#[test]
fn initial_ue_message_carries_the_cell_identity_and_cause() {
    let pdu = build_initial_ue_message(0x00AB_CDEF, &[1, 2, 3], &location(), 4, None);
    let S1AP_PDU::InitiatingMessage(InitiatingMessage {
        value: InitiatingMessageValue::Id_initialUEMessage(m),
        ..
    }) = codec::decode_s1ap_pdu(&codec::encode_s1ap_pdu(&pdu).unwrap()).unwrap()
    else {
        panic!();
    };
    let mut seen = 0;
    for ie in m.protocol_i_es.0 {
        match ie.value {
            InitialUEMessageProtocolIEs_EntryValue::Id_eNB_UE_S1AP_ID(v) => {
                assert_eq!(v.0, 0x00AB_CDEF);
                seen += 1;
            }
            InitialUEMessageProtocolIEs_EntryValue::Id_EUTRAN_CGI(c) => {
                let v = c
                    .cell_id
                    .0
                    .iter()
                    .fold(0u32, |a, b| (a << 1) | u32::from(*b));
                assert_eq!((c.cell_id.0.len(), v), (28, 0x19B01));
                assert_eq!(c.plm_nidentity.0, vec![0x99, 0xF9, 0x07]);
                seen += 1;
            }
            InitialUEMessageProtocolIEs_EntryValue::Id_RRC_Establishment_Cause(c) => {
                assert_eq!(c.0, RRC_Establishment_Cause::MO_DATA);
                seen += 1;
            }
            InitialUEMessageProtocolIEs_EntryValue::Id_S_TMSI(_) => panic!("no S-TMSI given"),
            _ => {}
        }
    }
    assert_eq!(seen, 3);
}

#[test]
fn uplink_nas_transport_round_trips() {
    let pdu = build_uplink_nas_transport(0xFFFF_FFFF, 5, &[9, 9], &location());
    let bytes = codec::encode_s1ap_pdu(&pdu).unwrap();
    assert_eq!(
        crate::s1ap::peek_header(&bytes),
        Some((0, 13, Criticality::IGNORE))
    );
    assert_eq!(codec::decode_s1ap_pdu(&bytes).unwrap(), pdu);
}

fn downlink_nas(ies: Vec<DownlinkNASTransportProtocolIEs_Entry>) -> Vec<u8> {
    codec::encode_s1ap_pdu(&S1AP_PDU::InitiatingMessage(InitiatingMessage {
        procedure_code: ProcedureCode(ID_DOWNLINK_NAS_TRANSPORT),
        criticality: Criticality(Criticality::IGNORE),
        value: InitiatingMessageValue::Id_downlinkNASTransport(DownlinkNASTransport {
            protocol_i_es: DownlinkNASTransportProtocolIEs(ies),
        }),
    }))
    .unwrap()
}

fn dl_ies(with_nas: bool) -> Vec<DownlinkNASTransportProtocolIEs_Entry> {
    let mut v = vec![
        DownlinkNASTransportProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_MME_UE_S1AP_ID),
            criticality: Criticality(Criticality::REJECT),
            value: DownlinkNASTransportProtocolIEs_EntryValue::Id_MME_UE_S1AP_ID(MME_UE_S1AP_ID(
                42,
            )),
        },
        DownlinkNASTransportProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_E_NB_UE_S1AP_ID),
            criticality: Criticality(Criticality::REJECT),
            value: DownlinkNASTransportProtocolIEs_EntryValue::Id_eNB_UE_S1AP_ID(ENB_UE_S1AP_ID(7)),
        },
    ];
    if with_nas {
        v.push(DownlinkNASTransportProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_NAS_PDU),
            criticality: Criticality(Criticality::REJECT),
            value: DownlinkNASTransportProtocolIEs_EntryValue::Id_NAS_PDU(NAS_PDU(vec![
                0x07, 0x55,
            ])),
        });
    }
    v
}

#[test]
fn downlink_nas_transport_is_relayed() {
    assert_eq!(
        handle_s1_pdu(&downlink_nas(dl_ies(true))),
        S1Inbound::DownlinkNas {
            mme_ue_s1ap_id: 42,
            enb_ue_s1ap_id: 7,
            nas: vec![0x07, 0x55],
            error_indication: None,
        }
    );
}

fn error_indication_ies(bytes: &[u8]) -> Vec<ErrorIndicationProtocolIEs_EntryValue> {
    let S1AP_PDU::InitiatingMessage(InitiatingMessage {
        value: InitiatingMessageValue::Id_ErrorIndication(ei),
        ..
    }) = codec::decode_s1ap_pdu(bytes).unwrap()
    else {
        panic!("not an Error Indication");
    };
    ei.protocol_i_es.0.into_iter().map(|e| e.value).collect()
}

#[test]
fn downlink_nas_transport_without_nas_pdu_is_reported_missing() {
    // §10.3.5: NAS-PDU is mandatory with criticality reject.
    let S1Inbound::ErrorIndication(ei) = handle_s1_pdu(&downlink_nas(dl_ies(false))) else {
        panic!("expected an Error Indication");
    };
    let ies = error_indication_ies(&ei);
    assert!(matches!(
        ies[0],
        ErrorIndicationProtocolIEs_EntryValue::Id_MME_UE_S1AP_ID(MME_UE_S1AP_ID(42))
    ));
    assert!(matches!(
        ies[1],
        ErrorIndicationProtocolIEs_EntryValue::Id_eNB_UE_S1AP_ID(ENB_UE_S1AP_ID(7))
    ));
    let diag = ies
        .iter()
        .find_map(|v| match v {
            ErrorIndicationProtocolIEs_EntryValue::Id_CriticalityDiagnostics(d) => Some(d),
            _ => None,
        })
        .expect("criticality diagnostics");
    let item = &diag.i_es_criticality_diagnostics.as_ref().unwrap().0[0];
    assert_eq!(
        (item.ie_id.0, item.type_of_error.0),
        (ID_NAS_PDU, TypeOfError::MISSING)
    );
    assert_eq!(
        diag.procedure_code.as_ref().map(|p| p.0),
        Some(ID_DOWNLINK_NAS_TRANSPORT)
    );
}

#[test]
fn unknown_enb_ue_s1ap_id_error_carries_cause_and_ids() {
    let ies = error_indication_ies(&unknown_enb_ue_s1ap_id_error(42, 99).unwrap());
    assert!(ies.iter().any(|v| matches!(
        v,
        ErrorIndicationProtocolIEs_EntryValue::Id_Cause(Cause::RadioNetwork(CauseRadioNetwork(c)))
            if *c == CauseRadioNetwork::UNKNOWN_ENB_UE_S1AP_ID
    )));
    assert!(ies.iter().any(|v| matches!(
        v,
        ErrorIndicationProtocolIEs_EntryValue::Id_eNB_UE_S1AP_ID(ENB_UE_S1AP_ID(99))
    )));
}

#[test]
fn garbage_is_a_transfer_syntax_error() {
    let S1Inbound::ErrorIndication(ei) = handle_s1_pdu(&[0x00, 0x0B, 0x40]) else {
        panic!("expected an Error Indication");
    };
    assert!(error_indication_ies(&ei).iter().any(|v| matches!(
        v,
        ErrorIndicationProtocolIEs_EntryValue::Id_Cause(Cause::Protocol(CauseProtocol(c)))
            if *c == CauseProtocol::TRANSFER_SYNTAX_ERROR
    )));
}

#[test]
fn release_command_is_recognised_with_either_id_form() {
    for ids in [
        UE_S1AP_IDs::UE_S1AP_ID_pair(UE_S1AP_ID_pair {
            mme_ue_s1ap_id: MME_UE_S1AP_ID(42),
            enb_ue_s1ap_id: ENB_UE_S1AP_ID(7),
            ie_extensions: None,
        }),
        UE_S1AP_IDs::MME_UE_S1AP_ID(MME_UE_S1AP_ID(42)),
    ] {
        let expect = match &ids {
            UE_S1AP_IDs::UE_S1AP_ID_pair(_) => ReleaseIds::Pair { mme: 42, enb: 7 },
            UE_S1AP_IDs::MME_UE_S1AP_ID(_) => ReleaseIds::Mme(42),
        };
        let pdu = S1AP_PDU::InitiatingMessage(InitiatingMessage {
            procedure_code: ProcedureCode(ID_UE_CONTEXT_RELEASE),
            criticality: Criticality(Criticality::REJECT),
            value: InitiatingMessageValue::Id_UEContextRelease(UEContextReleaseCommand {
                protocol_i_es: UEContextReleaseCommandProtocolIEs(vec![
                    UEContextReleaseCommandProtocolIEs_Entry {
                        id: ProtocolIE_ID(ID_UE_S1AP_I_DS),
                        criticality: Criticality(Criticality::REJECT),
                        value: UEContextReleaseCommandProtocolIEs_EntryValue::Id_UE_S1AP_IDs(ids),
                    },
                    UEContextReleaseCommandProtocolIEs_Entry {
                        id: ProtocolIE_ID(ID_CAUSE),
                        criticality: Criticality(Criticality::IGNORE),
                        value: UEContextReleaseCommandProtocolIEs_EntryValue::Id_Cause(Cause::Nas(
                            CauseNas(CauseNas::NORMAL_RELEASE),
                        )),
                    },
                ]),
            }),
        });
        assert_eq!(
            handle_s1_pdu(&codec::encode_s1ap_pdu(&pdu).unwrap()),
            S1Inbound::ReleaseCommand { ids: expect }
        );
    }
}

#[test]
fn release_request_and_complete_carry_their_procedure_codes() {
    let req = codec::encode_s1ap_pdu(&build_ue_context_release_request(
        42,
        7,
        CauseRadioNetwork::RADIO_CONNECTION_WITH_UE_LOST,
    ))
    .unwrap();
    assert_eq!(
        crate::s1ap::peek_header(&req),
        Some((0, 18, Criticality::IGNORE))
    );
    let done = codec::encode_s1ap_pdu(&build_ue_context_release_complete(42, 7)).unwrap();
    assert_eq!(
        crate::s1ap::peek_header(&done),
        Some((1, 23, Criticality::REJECT))
    );
}

fn erab_release_command(with_list: bool) -> Vec<u8> {
    let ie = |id, v| E_RABReleaseCommandProtocolIEs_Entry {
        id: ProtocolIE_ID(id),
        criticality: Criticality(Criticality::REJECT),
        value: v,
    };
    let mut ies = vec![
        ie(
            ID_MME_UE_S1AP_ID,
            E_RABReleaseCommandProtocolIEs_EntryValue::Id_MME_UE_S1AP_ID(MME_UE_S1AP_ID(42)),
        ),
        ie(
            ID_E_NB_UE_S1AP_ID,
            E_RABReleaseCommandProtocolIEs_EntryValue::Id_eNB_UE_S1AP_ID(ENB_UE_S1AP_ID(7)),
        ),
    ];
    if with_list {
        ies.push(ie(
            ID_E_RAB_TO_BE_RELEASED_LIST,
            E_RABReleaseCommandProtocolIEs_EntryValue::Id_E_RABToBeReleasedList(E_RABList(
                [6u8, 7]
                    .iter()
                    .map(|id| E_RABList_Entry {
                        id: ProtocolIE_ID(ID_E_RAB_ITEM),
                        criticality: Criticality(Criticality::IGNORE),
                        value: E_RABList_EntryValue::Id_E_RABItem(E_RABItem {
                            e_rab_id: E_RAB_ID(*id),
                            cause: Cause::Nas(CauseNas(CauseNas::NORMAL_RELEASE)),
                            ie_extensions: None,
                        }),
                    })
                    .collect(),
            )),
        ));
    }
    ies.push(ie(
        ID_NAS_PDU,
        E_RABReleaseCommandProtocolIEs_EntryValue::Id_NAS_PDU(NAS_PDU(vec![0x27, 0xCD])),
    ));
    codec::encode_s1ap_pdu(&S1AP_PDU::InitiatingMessage(InitiatingMessage {
        procedure_code: ProcedureCode(ID_E_RAB_RELEASE),
        criticality: Criticality(Criticality::REJECT),
        value: InitiatingMessageValue::Id_E_RABRelease(E_RABReleaseCommand {
            protocol_i_es: E_RABReleaseCommandProtocolIEs(ies),
        }),
    }))
    .unwrap()
}

/// #283, TS 36.413 §9.1.3.5: E-RAB RELEASE COMMAND decodes to its E-RAB IDs
/// and NAS-PDU; without the mandatory E-RAB To Be Released List it is
/// reported missing (§10.3.5).
#[test]
fn an_erab_release_command_decodes_and_its_list_is_mandatory() {
    assert_eq!(
        handle_s1_pdu(&erab_release_command(true)),
        S1Inbound::ErabRelease(Box::new(ErabReleaseCommand {
            mme_ue_s1ap_id: 42,
            enb_ue_s1ap_id: 7,
            erabs: vec![6, 7],
            nas: Some(vec![0x27, 0xCD]),
        }))
    );
    let S1Inbound::ErrorIndication(ei) = handle_s1_pdu(&erab_release_command(false)) else {
        panic!("expected an Error Indication");
    };
    let diag = error_indication_ies(&ei)
        .into_iter()
        .find_map(|v| match v {
            ErrorIndicationProtocolIEs_EntryValue::Id_CriticalityDiagnostics(d) => Some(d),
            _ => None,
        })
        .expect("criticality diagnostics");
    let item = &diag.i_es_criticality_diagnostics.as_ref().unwrap().0[0];
    assert_eq!(
        (item.ie_id.0, item.type_of_error.0),
        (ID_E_RAB_TO_BE_RELEASED_LIST, TypeOfError::MISSING)
    );
    assert_eq!(diag.procedure_code.map(|p| p.0), Some(ID_E_RAB_RELEASE));
}

/// §9.1.3.2: E-RAB SETUP RESPONSE has its IEs in table order, all with
/// criticality ignore, and each list only when it has an item.
#[test]
fn an_erab_setup_response_lists_only_what_it_has() {
    let ids = |pdu: &S1AP_PDU| {
        let S1AP_PDU::SuccessfulOutcome(SuccessfulOutcome {
            value: SuccessfulOutcomeValue::Id_E_RABSetup(r),
            ..
        }) = pdu
        else {
            panic!("{pdu:?}");
        };
        r.protocol_i_es
            .0
            .iter()
            .map(|ie| (ie.id.0, ie.criticality.0))
            .collect::<Vec<_>>()
    };
    let ok = [ErabSetup {
        e_rab_id: 6,
        address: "10.0.0.1".parse().unwrap(),
        teid: 9,
    }];
    let failed = [(
        7,
        Cause::RadioNetwork(CauseRadioNetwork(CauseRadioNetwork::UNKNOWN_E_RAB_ID)),
    )];
    let i = Criticality::IGNORE;
    let both = build_erab_setup_response(42, 7, &ok, &failed);
    assert_eq!(
        ids(&both),
        vec![
            (ID_MME_UE_S1AP_ID, i),
            (ID_E_NB_UE_S1AP_ID, i),
            (ID_E_RAB_SETUP_LIST_BEARER_SU_RES, i),
            (ID_E_RAB_FAILED_TO_SETUP_LIST_BEARER_SU_RES, i),
        ]
    );
    let bytes = codec::encode_s1ap_pdu(&both).unwrap();
    assert_eq!(codec::decode_s1ap_pdu(&bytes).unwrap(), both);
    assert_eq!(
        ids(&build_erab_setup_response(42, 7, &[], &failed)),
        vec![
            (ID_MME_UE_S1AP_ID, i),
            (ID_E_NB_UE_S1AP_ID, i),
            (ID_E_RAB_FAILED_TO_SETUP_LIST_BEARER_SU_RES, i),
        ]
    );
}
