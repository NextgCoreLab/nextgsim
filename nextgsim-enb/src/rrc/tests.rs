use super::*;
use nextgsim_s1ap::codec::*;

fn cfg() -> EnbConfig {
    crate::s1ap::tests::cfg()
}

fn rrc() -> EnbRrc {
    EnbRrc::new(&cfg(), 2)
}

fn connection_request() -> Vec<u8> {
    msg::encode_rrc_connection_request(
        InitialUeIdentity::Random(0x12_3456_789A),
        msg::EstablishmentCauseValue::MoSignalling,
    )
    .unwrap()
}

/// What the UE does: wrap an UL-DCCH message in its own SRB1 PDCP entity.
fn ue_dcch(ue_pdcp: &mut LteSrbPdcp, rrc_msg: &[u8]) -> Vec<u8> {
    ue_pdcp.transmit(rrc_msg).0
}

fn only_s1(acts: &[Action]) -> (u16, S1AP_PDU) {
    let s1: Vec<_> = acts
        .iter()
        .filter_map(|a| match a {
            Action::S1 { stream, pdu } => Some((*stream, decode_s1ap_pdu(pdu).unwrap())),
            Action::Rrc { .. } | Action::UserPlane { .. } => None,
        })
        .collect();
    assert_eq!(s1.len(), 1, "{acts:?}");
    s1.into_iter().next().unwrap()
}

/// Run the request / setup / setup-complete exchange for RLS UE `ue`.
fn connect(e: &mut EnbRrc, ue: u32, ue_pdcp: &mut LteSrbPdcp, nas: &[u8]) -> Vec<Action> {
    let acts = e.on_ul_ccch(ue, &connection_request());
    let [Action::Rrc {
        channel: RrcChannel::DlCcch,
        pdu,
        ..
    }] = acts.as_slice()
    else {
        panic!("expected RRCConnectionSetup on DL-CCCH: {acts:?}");
    };
    let msg::DlCcch::ConnectionSetup { transaction_id, .. } = msg::decode_dl_ccch(pdu).unwrap()
    else {
        panic!("not RRCConnectionSetup");
    };
    let complete = msg::encode_rrc_connection_setup_complete(transaction_id, 1, nas).unwrap();
    e.on_ul_dcch(ue, LCID_SRB1, &ue_dcch(ue_pdcp, &complete))
}

#[test]
fn setup_complete_becomes_initial_ue_message_on_a_ue_stream() {
    let mut e = rrc();
    let mut pdcp = LteSrbPdcp::new();
    let acts = connect(&mut e, 5, &mut pdcp, &[0x07, 0x41, 0x71]);
    let (stream, pdu) = only_s1(&acts);
    // TS 36.412 §7: stream 0 is reserved for non UE-associated signalling.
    assert_eq!(stream, 1);
    let S1AP_PDU::InitiatingMessage(InitiatingMessage {
        value: InitiatingMessageValue::Id_initialUEMessage(m),
        ..
    }) = pdu
    else {
        panic!("not an INITIAL UE MESSAGE");
    };
    let nas = m.protocol_i_es.0.iter().find_map(|ie| match &ie.value {
        InitialUEMessageProtocolIEs_EntryValue::Id_NAS_PDU(n) => Some(n.0.clone()),
        _ => None,
    });
    assert_eq!(nas, Some(vec![0x07, 0x41, 0x71]));
    let enb_id = m.protocol_i_es.0.iter().find_map(|ie| match &ie.value {
        InitialUEMessageProtocolIEs_EntryValue::Id_eNB_UE_S1AP_ID(v) => Some(v.0),
        _ => None,
    });
    assert_eq!(enb_id, e.enb_ue_s1ap_id(5));
}

#[test]
fn setup_complete_with_the_wrong_transaction_id_is_ignored() {
    let mut e = rrc();
    e.on_ul_ccch(1, &connection_request());
    let mut pdcp = LteSrbPdcp::new();
    let complete = msg::encode_rrc_connection_setup_complete(3, 1, &[1]).unwrap();
    assert!(e
        .on_ul_dcch(1, LCID_SRB1, &ue_dcch(&mut pdcp, &complete))
        .is_empty());
}

#[test]
fn a_bare_rrc_message_on_dcch_is_not_accepted_as_a_pdcp_pdu() {
    // Without the PDCP header and MAC-I the eNB must not find a valid
    // SetupComplete: SRB1 always carries PDCP Data PDUs (TS 36.323 §6.2.2).
    let mut e = rrc();
    e.on_ul_ccch(1, &connection_request());
    let complete = msg::encode_rrc_connection_setup_complete(0, 1, &[0x07, 0x41]).unwrap();
    let acts = e.on_ul_dcch(1, LCID_SRB1, &complete);
    assert!(
        !acts.iter().any(|a| matches!(a, Action::S1 { .. })),
        "a non-PDCP PDU produced an INITIAL UE MESSAGE"
    );
}

#[test]
fn selected_plmn_outside_sib1_is_refused() {
    let mut e = rrc();
    e.on_ul_ccch(1, &connection_request());
    let mut pdcp = LteSrbPdcp::new();
    let complete = msg::encode_rrc_connection_setup_complete(0, 2, &[1]).unwrap();
    assert!(e
        .on_ul_dcch(1, LCID_SRB1, &ue_dcch(&mut pdcp, &complete))
        .is_empty());
}

fn downlink_nas(mme: u32, enb: u32, nas: &[u8]) -> Vec<u8> {
    encode_s1ap_pdu(&S1AP_PDU::InitiatingMessage(InitiatingMessage {
        procedure_code: ProcedureCode(ID_DOWNLINK_NAS_TRANSPORT),
        criticality: Criticality(Criticality::IGNORE),
        value: InitiatingMessageValue::Id_downlinkNASTransport(DownlinkNASTransport {
            protocol_i_es: DownlinkNASTransportProtocolIEs(vec![
                DownlinkNASTransportProtocolIEs_Entry {
                    id: ProtocolIE_ID(ID_MME_UE_S1AP_ID),
                    criticality: Criticality(Criticality::REJECT),
                    value: DownlinkNASTransportProtocolIEs_EntryValue::Id_MME_UE_S1AP_ID(
                        MME_UE_S1AP_ID(mme),
                    ),
                },
                DownlinkNASTransportProtocolIEs_Entry {
                    id: ProtocolIE_ID(ID_E_NB_UE_S1AP_ID),
                    criticality: Criticality(Criticality::REJECT),
                    value: DownlinkNASTransportProtocolIEs_EntryValue::Id_eNB_UE_S1AP_ID(
                        ENB_UE_S1AP_ID(enb),
                    ),
                },
                DownlinkNASTransportProtocolIEs_Entry {
                    id: ProtocolIE_ID(ID_NAS_PDU),
                    criticality: Criticality(Criticality::REJECT),
                    value: DownlinkNASTransportProtocolIEs_EntryValue::Id_NAS_PDU(NAS_PDU(
                        nas.to_vec(),
                    )),
                },
            ]),
        }),
    }))
    .unwrap()
}

#[test]
fn downlink_nas_is_relayed_in_a_pdcp_pdu_and_uplink_nas_uses_the_stored_mme_id() {
    let mut e = rrc();
    let mut ue_pdcp = LteSrbPdcp::new();
    connect(&mut e, 5, &mut ue_pdcp, &[0x07, 0x41]);
    let enb = e.enb_ue_s1ap_id(5).unwrap();

    let acts = e.on_s1(&downlink_nas(77, enb, &[0x07, 0x52, 0xAA]));
    let [Action::Rrc {
        ue: 5,
        channel: RrcChannel::DlDcch,
        lcid: LCID_SRB1,
        pdu,
    }] = acts.as_slice()
    else {
        panic!("expected one DL-DCCH PDU: {acts:?}");
    };
    // The UE side of SRB1: strip the PDCP PDU, decode DLInformationTransfer.
    let mut ue_rx = LteSrbPdcp::new();
    let r = ue_rx.receive(pdu).unwrap();
    let (sdu, count) = (r.sdu, r.count);
    assert_eq!(count, 0, "first downlink SRB1 PDU uses COUNT 0");
    assert_eq!(
        msg::decode_dl_dcch(&sdu).unwrap(),
        msg::DlDcch::InformationTransfer {
            nas: vec![0x07, 0x52, 0xAA]
        }
    );

    let ul = msg::encode_ul_information_transfer(&[0x07, 0x53]).unwrap();
    let (stream, pdu) = only_s1(&e.on_ul_dcch(5, LCID_SRB1, &ue_dcch(&mut ue_pdcp, &ul)));
    assert_eq!(stream, 1);
    let S1AP_PDU::InitiatingMessage(InitiatingMessage {
        value: InitiatingMessageValue::Id_uplinkNASTransport(u),
        ..
    }) = pdu
    else {
        panic!("not UPLINK NAS TRANSPORT");
    };
    let mme = u.protocol_i_es.0.iter().find_map(|ie| match &ie.value {
        UplinkNASTransportProtocolIEs_EntryValue::Id_MME_UE_S1AP_ID(v) => Some(v.0),
        _ => None,
    });
    assert_eq!(mme, Some(77));
}

#[test]
fn uplink_nas_before_the_mme_id_is_held_then_sent() {
    let mut e = rrc();
    let mut ue_pdcp = LteSrbPdcp::new();
    connect(&mut e, 5, &mut ue_pdcp, &[0x07, 0x41]);
    let ul = msg::encode_ul_information_transfer(&[0x07, 0x5E]).unwrap();
    assert!(e
        .on_ul_dcch(5, LCID_SRB1, &ue_dcch(&mut ue_pdcp, &ul))
        .is_empty());
    let enb = e.enb_ue_s1ap_id(5).unwrap();
    let acts = e.on_s1(&downlink_nas(9, enb, &[0x07, 0x52]));
    assert!(acts.iter().any(|a| matches!(a, Action::Rrc { .. })));
    let (_, pdu) = only_s1(&acts);
    assert!(matches!(
        pdu,
        S1AP_PDU::InitiatingMessage(InitiatingMessage {
            value: InitiatingMessageValue::Id_uplinkNASTransport(_),
            ..
        })
    ));
}

#[test]
fn downlink_nas_for_an_unknown_enb_id_is_reported() {
    let mut e = rrc();
    let (stream, pdu) = only_s1(&e.on_s1(&downlink_nas(1, 4242, &[0])));
    assert_eq!(stream, NON_UE_ASSOCIATED_STREAM);
    let S1AP_PDU::InitiatingMessage(InitiatingMessage {
        value: InitiatingMessageValue::Id_ErrorIndication(ei),
        ..
    }) = pdu
    else {
        panic!("not an ERROR INDICATION");
    };
    assert!(ei.protocol_i_es.0.iter().any(|ie| matches!(
        &ie.value,
        ErrorIndicationProtocolIEs_EntryValue::Id_Cause(Cause::RadioNetwork(CauseRadioNetwork(c)))
            if *c == CauseRadioNetwork::UNKNOWN_ENB_UE_S1AP_ID
    )));
}

fn release_command(mme: u32, enb: u32) -> Vec<u8> {
    encode_s1ap_pdu(&S1AP_PDU::InitiatingMessage(InitiatingMessage {
        procedure_code: ProcedureCode(ID_UE_CONTEXT_RELEASE),
        criticality: Criticality(Criticality::REJECT),
        value: InitiatingMessageValue::Id_UEContextRelease(UEContextReleaseCommand {
            protocol_i_es: UEContextReleaseCommandProtocolIEs(vec![
                UEContextReleaseCommandProtocolIEs_Entry {
                    id: ProtocolIE_ID(ID_UE_S1AP_I_DS),
                    criticality: Criticality(Criticality::REJECT),
                    value: UEContextReleaseCommandProtocolIEs_EntryValue::Id_UE_S1AP_IDs(
                        UE_S1AP_IDs::UE_S1AP_ID_pair(UE_S1AP_ID_pair {
                            mme_ue_s1ap_id: MME_UE_S1AP_ID(mme),
                            enb_ue_s1ap_id: ENB_UE_S1AP_ID(enb),
                            ie_extensions: None,
                        }),
                    ),
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
    }))
    .unwrap()
}

#[test]
fn release_command_releases_rrc_and_completes() {
    let mut e = rrc();
    let mut ue_pdcp = LteSrbPdcp::new();
    connect(&mut e, 5, &mut ue_pdcp, &[0x07, 0x41]);
    let enb = e.enb_ue_s1ap_id(5).unwrap();
    e.on_s1(&downlink_nas(9, enb, &[0x07, 0x44])); // Attach Reject, say
    let acts = e.on_s1(&release_command(9, enb));
    let rrc = acts.iter().find_map(|a| match a {
        Action::Rrc {
            channel: RrcChannel::DlDcch,
            pdu,
            ..
        } => Some(pdu.clone()),
        _ => None,
    });
    let mut ue_rx = LteSrbPdcp::new();
    ue_rx.receive(&[0, 0, 0, 0, 0]).unwrap(); // the DLInformationTransfer before it
    let sdu = ue_rx
        .receive(&rrc.expect("RRCConnectionRelease"))
        .unwrap()
        .sdu;
    assert_eq!(
        msg::decode_dl_dcch(&sdu).unwrap(),
        msg::DlDcch::ConnectionRelease
    );
    let (_, pdu) = only_s1(&acts);
    assert!(matches!(
        pdu,
        S1AP_PDU::SuccessfulOutcome(SuccessfulOutcome {
            value: SuccessfulOutcomeValue::Id_UEContextRelease(_),
            ..
        })
    ));
    assert_eq!(e.ue_count(), 0);
}

#[test]
fn losing_a_ue_with_an_s1_connection_requests_its_release() {
    let mut e = rrc();
    let mut ue_pdcp = LteSrbPdcp::new();
    connect(&mut e, 5, &mut ue_pdcp, &[0x07, 0x41]);
    let enb = e.enb_ue_s1ap_id(5).unwrap();
    e.on_s1(&downlink_nas(9, enb, &[0x07, 0x52]));
    let (_, pdu) = only_s1(&e.on_ue_lost(5));
    let S1AP_PDU::InitiatingMessage(InitiatingMessage {
        value: InitiatingMessageValue::Id_UEContextReleaseRequest(r),
        ..
    }) = pdu
    else {
        panic!("not UE CONTEXT RELEASE REQUEST");
    };
    assert!(r.protocol_i_es.0.iter().any(|ie| matches!(
        &ie.value,
        UEContextReleaseRequestProtocolIEs_EntryValue::Id_Cause(Cause::RadioNetwork(CauseRadioNetwork(c)))
            if *c == CauseRadioNetwork::RADIO_CONNECTION_WITH_UE_LOST
    )));
    // Without an S1 connection there is nothing to ask the MME.
    e.on_ul_ccch(6, &connection_request());
    assert!(e.on_ue_lost(6).is_empty());
}

#[test]
fn enb_ue_s1ap_ids_are_unique_and_admission_is_bounded() {
    let mut e = rrc();
    let mut ids = std::collections::HashSet::new();
    for ue in 0..MAX_UE_CONTEXTS as u32 {
        e.on_ul_ccch(ue, &connection_request());
        assert!(ids.insert(e.enb_ue_s1ap_id(ue).unwrap()));
    }
    let acts = e.on_ul_ccch(99_999, &connection_request());
    let [Action::Rrc { pdu, .. }] = acts.as_slice() else {
        panic!("{acts:?}");
    };
    assert_eq!(
        msg::decode_dl_ccch(pdu).unwrap(),
        msg::DlCcch::ConnectionReject {
            wait_time_s: REJECT_WAIT_TIME_S
        }
    );
}

#[test]
fn system_information_carries_the_configured_cell() {
    let mut e = rrc();
    let si = e.system_information(0x3FF);
    assert_eq!(si.len(), 3);
    assert_eq!(si[0].0, RrcChannel::BcchBch);
    let msg::BcchDlSch::Sib1(cell) = msg::decode_bcch_dl_sch(&si[1].1).unwrap() else {
        panic!("second PDU is not SIB1");
    };
    assert_eq!(cell.cell_identity, cfg().cell_identity());
    assert_eq!(cell.tac, 1);
    assert_eq!(
        cell.plmns,
        vec![PlmnId {
            mcc: 999,
            mnc: 70,
            long_mnc: false
        }]
    );
    let msg::BcchDlSch::SystemInformation { sib2: Some(sib2) } =
        msg::decode_bcch_dl_sch(&si[2].1).unwrap()
    else {
        panic!("third PDU is not SIB2");
    };
    assert_eq!(sib2.timers.t300_ms, cfg().radio.t300_ms);
    assert_eq!(sib2.default_paging_cycle, msg::PagingCycle::Rf128);
}

/// After Initial Context Setup's AS SMC: the reconfiguration adds SRB2 and the
/// DRB (RLC UM), its completion hands the E-RAB to the user plane, and the
/// next DOWNLINK NAS TRANSPORT goes out on SRB2 under SRB1's AS keys.
#[test]
fn reconfiguration_adds_srb2_and_nas_moves_onto_it() {
    let mut e = rrc();
    let mut ue_pdcp = LteSrbPdcp::new();
    connect(&mut e, 5, &mut ue_pdcp, &[0x07, 0x41]);
    let enb = e.enb_ue_s1ap_id(5).unwrap();
    e.on_s1(&downlink_nas(77, enb, &[0x07, 0x52]));
    let (k_int, k_enc, k_up) = ([1u8; 16], [2u8; 16], [3u8; 16]);
    {
        // The state the AS SMC of on_initial_context_setup leaves (that path is
        // run end to end in tests/lte_radio_e2e.rs).
        let ctx = e.ues.get_mut(&5).unwrap();
        ctx.security = Some(AsSecurity {
            eea: 2,
            eia: 2,
            k_rrc_int: k_int,
            k_rrc_enc: k_enc,
            k_up_enc: k_up,
        });
        ctx.ics = Some(PendingIcs {
            erabs: vec![ErabToSetup {
                e_rab_id: 5,
                qci: 9,
                sgw_address: vec![10, 0, 0, 9],
                sgw_teid: 0x1234,
                nas: Some(vec![0x27, 0, 0, 0, 0, 0, 0x07, 0x42]),
            }],
            notify: vec![],
            rx_ciphering: (2, k_enc),
            smc_transaction_id: 1,
            reconfiguration_transaction_id: None,
            teids: vec![],
        });
    }
    let acts = e.on_security_mode_complete(5, 1);
    let [Action::Rrc {
        lcid: LCID_SRB1,
        pdu,
        ..
    }] = acts.as_slice()
    else {
        panic!("{acts:?}");
    };
    // The UE's view of SRB1 (no ciphering on in this hand-built context, so
    // the reconfiguration is read from the PDU as transmitted: the eNB's SRB1
    // has no TX ciphering here).
    let mut ue_srb1 = LteSrbPdcp::new();
    ue_srb1.receive(&[0, 0, 0, 0, 0]).unwrap(); // the DLInformationTransfer before it
    let msg::DlDcch::ConnectionReconfiguration(r) =
        msg::decode_dl_dcch(&ue_srb1.receive(pdu).unwrap().sdu).unwrap()
    else {
        panic!("not a reconfiguration");
    };
    assert!(r.srb2, "SRB2 added with the DRBs");
    assert_eq!(r.drbs[0].rlc, DrbRlc::UmSn10);
    let tid = r.transaction_id;

    let acts = e.on_reconfiguration_complete(5, tid);
    let Some(Action::UserPlane { ue: 5, bearers }) = acts.first() else {
        panic!("no user plane: {acts:?}");
    };
    assert_eq!(bearers.len(), 1);
    let b = &bearers[0];
    assert_eq!(
        (b.e_rab_id, b.lcid, b.sgw, b.sgw_teid, b.eea, b.k_up_enc),
        (5, 3, "10.0.0.9:2152".parse().unwrap(), 0x1234, 2, k_up)
    );

    let acts = e.on_s1(&downlink_nas(77, enb, &[0x07, 0x61]));
    let [Action::Rrc {
        lcid: LCID_SRB2,
        pdu,
        ..
    }] = acts.as_slice()
    else {
        panic!("DL NAS not on SRB2: {acts:?}");
    };
    let mut ue_srb2 = LteSrbPdcp::with_bearer(1, nextgsim_pdcp::lte_srb::DIRECTION_UPLINK);
    ue_srb2.activate_rx_integrity(2, k_int).unwrap();
    ue_srb2.activate_rx_ciphering(2, k_enc).unwrap();
    assert_eq!(
        msg::decode_dl_dcch(&ue_srb2.receive(pdu).unwrap().sdu).unwrap(),
        msg::DlDcch::InformationTransfer {
            nas: vec![0x07, 0x61]
        }
    );
}
