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
            Action::Rrc { .. }
            | Action::UserPlane { .. }
            | Action::ReleaseUserPlane { .. }
            | Action::Broadcast { .. } => None,
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
                gbr_info: false,
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

fn rrc_with_inactivity(secs: u64) -> EnbRrc {
    let mut c = cfg();
    c.inactivity_timer_secs = Some(secs);
    EnbRrc::new(&c, 2)
}

fn release_request_cause(acts: &[Action]) -> u8 {
    let (_, pdu) = only_s1(acts);
    let S1AP_PDU::InitiatingMessage(InitiatingMessage {
        value: InitiatingMessageValue::Id_UEContextReleaseRequest(r),
        ..
    }) = pdu
    else {
        panic!("not UE CONTEXT RELEASE REQUEST: {pdu:?}");
    };
    r.protocol_i_es
        .0
        .iter()
        .find_map(|ie| match &ie.value {
            UEContextReleaseRequestProtocolIEs_EntryValue::Id_Cause(Cause::RadioNetwork(
                CauseRadioNetwork(c),
            )) => Some(*c),
            _ => None,
        })
        .expect("a Radio Network cause")
}

/// #350, TS 36.413 §8.3.2: a UE with an S1 connection that stays silent for the
/// configured time gets one UE CONTEXT RELEASE REQUEST with cause
/// user-inactivity. Not before the time, and not twice while the command is
/// awaited; the command then releases RRC as for any other release.
#[test]
fn an_inactive_ue_is_released_with_cause_user_inactivity() {
    let mut e = rrc_with_inactivity(10);
    let mut ue_pdcp = LteSrbPdcp::new();
    connect(&mut e, 5, &mut ue_pdcp, &[0x07, 0x41]);
    let enb = e.enb_ue_s1ap_id(5).unwrap();
    e.on_s1(&downlink_nas(9, enb, &[0x07, 0x52]));
    let start = Instant::now();

    assert!(e
        .check_inactivity(start + Duration::from_secs(9))
        .is_empty());
    let acts = e.check_inactivity(start + Duration::from_secs(11));
    assert_eq!(
        release_request_cause(&acts),
        CauseRadioNetwork::USER_INACTIVITY
    );
    assert!(
        e.check_inactivity(start + Duration::from_secs(20))
            .is_empty(),
        "one request while the command is awaited"
    );
    assert_eq!(e.ue_count(), 1, "the context stays until the command");

    let acts = e.on_s1(&release_command(9, enb));
    assert!(acts.iter().any(|a| matches!(
        a,
        Action::Rrc {
            channel: RrcChannel::DlDcch,
            ..
        }
    )));
    assert_eq!(e.ue_count(), 0);
}

/// #353 review: a request whose command never comes (lost, or the MME kept the
/// UE) is re-armed. With no command one inactivity period after the request, it
/// is sent again. Activity after the request means the UE stays connected, and a
/// new period of silence asks again.
#[test]
fn a_lost_or_declined_inactivity_release_is_asked_again() {
    let mut e = rrc_with_inactivity(10);
    let mut ue_pdcp = LteSrbPdcp::new();
    connect(&mut e, 5, &mut ue_pdcp, &[0x07, 0x41]);
    let enb = e.enb_ue_s1ap_id(5).unwrap();
    e.on_s1(&downlink_nas(9, enb, &[0x07, 0x52]));
    let start = Instant::now();
    assert!(!e
        .check_inactivity(start + Duration::from_secs(11))
        .is_empty());
    // Lost: no command within another 10 s.
    assert_eq!(
        release_request_cause(&e.check_inactivity(start + Duration::from_secs(22))),
        CauseRadioNetwork::USER_INACTIVITY,
        "asked again when the command did not come"
    );
    // Declined: data flows after the request, then 10 s of silence.
    e.on_user_plane_activity(5, start + Duration::from_secs(23));
    assert!(e
        .check_inactivity(start + Duration::from_secs(30))
        .is_empty());
    assert_eq!(
        release_request_cause(&e.check_inactivity(start + Duration::from_secs(34))),
        CauseRadioNetwork::USER_INACTIVITY,
        "a new silence asks again"
    );
}

/// #350: user data restarts the timer.
#[test]
fn user_plane_activity_restarts_the_inactivity_timer() {
    let mut e = rrc_with_inactivity(10);
    let mut ue_pdcp = LteSrbPdcp::new();
    connect(&mut e, 5, &mut ue_pdcp, &[0x07, 0x41]);
    let enb = e.enb_ue_s1ap_id(5).unwrap();
    e.on_s1(&downlink_nas(9, enb, &[0x07, 0x52]));
    let start = Instant::now();
    e.on_user_plane_activity(5, start + Duration::from_secs(8));
    assert!(e
        .check_inactivity(start + Duration::from_secs(15))
        .is_empty());
    assert!(!e
        .check_inactivity(start + Duration::from_secs(19))
        .is_empty());
}

/// #350: off by default, and a UE with no S1 connection yet is not asked about.
#[test]
fn the_inactivity_release_is_off_by_default_and_needs_an_s1_connection() {
    let mut e = rrc();
    let mut ue_pdcp = LteSrbPdcp::new();
    connect(&mut e, 5, &mut ue_pdcp, &[0x07, 0x41]);
    let enb = e.enb_ue_s1ap_id(5).unwrap();
    e.on_s1(&downlink_nas(9, enb, &[0x07, 0x52]));
    assert!(e
        .check_inactivity(Instant::now() + Duration::from_secs(86_400))
        .is_empty());

    let mut e = rrc_with_inactivity(1);
    let mut ue_pdcp = LteSrbPdcp::new();
    connect(&mut e, 6, &mut ue_pdcp, &[0x07, 0x41]);
    assert!(
        e.check_inactivity(Instant::now() + Duration::from_secs(60))
            .is_empty(),
        "no MME UE S1AP ID yet"
    );
}

/// An S1AP PAGING (TS 36.413 §9.1.6) for S-TMSI `(mmec, m_tmsi)` in TAC `tac`
/// of PLMN 999/70.
fn paging(mmec: u8, m_tmsi: u32, tac: u16, cs: bool) -> Vec<u8> {
    paging_for(
        UEPagingID::S_TMSI(S_TMSI {
            mmec: MME_Code(vec![mmec]),
            m_tmsi: M_TMSI(m_tmsi.to_be_bytes().to_vec()),
            ie_extensions: None,
        }),
        tac,
        cs,
    )
}

/// An S1AP PAGING for UE Paging ID `id` in TAC `tac` of PLMN 999/70.
fn paging_for(id: UEPagingID, tac: u16, cs: bool) -> Vec<u8> {
    let ies = vec![
        PagingProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_UE_IDENTITY_INDEX_VALUE),
            criticality: Criticality(Criticality::IGNORE),
            value: PagingProtocolIEs_EntryValue::Id_UEIdentityIndexValue(UEIdentityIndexValue(
                (0..10).rev().map(|i| (1u16 >> i) & 1 == 1).collect(),
            )),
        },
        PagingProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_UE_PAGING_ID),
            criticality: Criticality(Criticality::IGNORE),
            value: PagingProtocolIEs_EntryValue::Id_UEPagingID(id),
        },
        PagingProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_CN_DOMAIN),
            criticality: Criticality(Criticality::IGNORE),
            value: PagingProtocolIEs_EntryValue::Id_CNDomain(CNDomain(if cs {
                CNDomain::CS
            } else {
                CNDomain::PS
            })),
        },
        PagingProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_TAI_LIST),
            criticality: Criticality(Criticality::IGNORE),
            value: PagingProtocolIEs_EntryValue::Id_TAIList(TAIList(vec![TAIList_Entry {
                id: ProtocolIE_ID(ID_TAI_ITEM),
                criticality: Criticality(Criticality::IGNORE),
                value: TAIList_EntryValue::Id_TAIItem(TAIItem {
                    tai: TAI {
                        plm_nidentity: TBCD_STRING(Plmn::new(999, 70, false).encode().to_vec()),
                        tac: TAC(tac.to_be_bytes().to_vec()),
                        ie_extensions: None,
                    },
                    ie_extensions: None,
                }),
            }])),
        },
    ];
    encode_s1ap_pdu(&S1AP_PDU::InitiatingMessage(InitiatingMessage {
        procedure_code: ProcedureCode(ID_PAGING),
        criticality: Criticality(Criticality::IGNORE),
        value: InitiatingMessageValue::Id_Paging(Paging {
            protocol_i_es: PagingProtocolIEs(ies),
        }),
    }))
    .unwrap()
}

/// #351, TS 36.413 §8.5.2 / TS 36.331 §5.3.2.2: an S1AP PAGING listing this
/// cell's TAI is broadcast as an RRC Paging on PCCH carrying the S-TMSI and the
/// CN domain; one for another TA is not this cell's to send.
#[test]
fn an_s1_paging_for_this_tai_is_broadcast_on_pcch() {
    let mut e = rrc();
    let acts = e.on_s1(&paging(1, 0x1234_5678, 1, false));
    let [Action::Broadcast {
        channel: RrcChannel::Pcch,
        pdu,
    }] = acts.as_slice()
    else {
        panic!("expected one PCCH broadcast: {acts:?}");
    };
    assert_eq!(
        msg::decode_pcch(pdu).unwrap(),
        vec![msg::PagingRecordInfo {
            identity: msg::PagedUeIdentity::STmsi {
                mmec: 1,
                m_tmsi: 0x1234_5678
            },
            cn_domain: msg::PagingCnDomain::Ps,
        }]
    );
    let acts = e.on_s1(&paging(1, 0x1234_5678, 1, true));
    let [Action::Broadcast { pdu, .. }] = acts.as_slice() else {
        panic!("{acts:?}");
    };
    assert_eq!(
        msg::decode_pcch(pdu).unwrap()[0].cn_domain,
        msg::PagingCnDomain::Cs
    );
    assert!(
        e.on_s1(&paging(1, 0x1234_5678, 7, false)).is_empty(),
        "TAC 7 is not this cell"
    );
}

/// #351 review, TS 36.413 §9.2.3.11: the IMSI is TBCD, "bit 4 to 1 of octet n
/// encoding digit 2n-1" and "bit 8 to 5 of octet n encoding digit 2n", with the
/// 1111 filler "encoded as bit 8 to 5 of octet N" for an odd number of digits.
/// Hand-encoded: IMSI 999700000000001 (15 digits) is 99 79 00 00 00 00 00 F1,
/// and 99970000000012 (14 digits) is 99 79 00 00 00 00 21. A nibble swap would
/// read the first as 9 9 7 9 ... and is caught.
#[test]
fn an_s1_imsi_paging_decodes_the_tbcd_digits_low_nibble_first() {
    for (tbcd, digits) in [
        (
            vec![0x99, 0x79, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF1],
            vec![9, 9, 9, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ),
        (
            vec![0x99, 0x79, 0x00, 0x00, 0x00, 0x00, 0x21],
            vec![9, 9, 9, 7, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2],
        ),
    ] {
        let mut e = rrc();
        let acts = e.on_s1(&paging_for(UEPagingID::IMSI(IMSI(tbcd.clone())), 1, false));
        let [Action::Broadcast { pdu, .. }] = acts.as_slice() else {
            panic!("expected one PCCH broadcast for {tbcd:02X?}: {acts:?}");
        };
        assert_eq!(
            msg::decode_pcch(pdu).unwrap()[0].identity,
            msg::PagedUeIdentity::Imsi(digits),
            "{tbcd:02X?}"
        );
    }
}

// ---------------------------------------------------------------------------
// E-RAB management (TS 36.413 §8.2; #283)
// ---------------------------------------------------------------------------

/// A connected UE `5` with MME UE S1AP ID 77, AS security on and E-RAB 5 on
/// DRB 1 (LCID 3) -- what Initial Context Setup leaves (run end to end in
/// tests/lte_radio_e2e.rs). Returns the eNB UE S1AP ID.
fn established(e: &mut EnbRrc) -> u32 {
    let mut ue_pdcp = LteSrbPdcp::new();
    connect(e, 5, &mut ue_pdcp, &[0x07, 0x41]);
    let enb = e.enb_ue_s1ap_id(5).unwrap();
    let ctx = e.ues.get_mut(&5).unwrap();
    ctx.mme_ue_s1ap_id = Some(77);
    ctx.security = Some(AsSecurity {
        eea: 2,
        eia: 2,
        k_rrc_int: [1; 16],
        k_rrc_enc: [2; 16],
        k_up_enc: [3; 16],
    });
    ctx.drbs.push(DrbConfig {
        eps_bearer_identity: 5,
        drb_identity: 1,
        logical_channel_identity: 3,
        rlc: DrbRlc::UmSn10,
    });
    enb
}

fn to_setup(e_rab_id: u8) -> ErabToSetup {
    ErabToSetup {
        e_rab_id,
        qci: 5,
        gbr_info: false,
        sgw_address: vec![10, 0, 0, 9],
        sgw_teid: 0x5600 + u32::from(e_rab_id),
        nas: Some(vec![0x27, e_rab_id]),
    }
}

fn erab_setup(enb: u32, erabs: Vec<ErabToSetup>) -> ErabRequest {
    ErabRequest::Setup(s1ap_ue::ErabSetupRequest {
        mme_ue_s1ap_id: 77,
        enb_ue_s1ap_id: enb,
        erabs,
    })
}

fn erab_release(enb: u32, erabs: Vec<u8>, nas: Option<Vec<u8>>) -> ErabRequest {
    ErabRequest::Release(s1ap_ue::ErabReleaseCommand {
        mme_ue_s1ap_id: 77,
        enb_ue_s1ap_id: enb,
        erabs,
        nas,
    })
}

/// The RRCConnectionReconfiguration on SRB1 among `acts`, as the UE reads it
/// (no SRB1 security in the hand-built context; `count` is its PDCP COUNT).
fn erab_reconfiguration(acts: &[Action], count: u32) -> msg::Reconfiguration {
    let pdu = acts
        .iter()
        .find_map(|a| match a {
            Action::Rrc {
                lcid: LCID_SRB1,
                pdu,
                ..
            } => Some(pdu.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no reconfiguration on SRB1: {acts:?}"));
    let mut ue_srb1 = LteSrbPdcp::new();
    for _ in 0..count {
        ue_srb1.receive(&[0, 0, 0, 0, 0]).unwrap();
    }
    let msg::DlDcch::ConnectionReconfiguration(r) =
        msg::decode_dl_dcch(&ue_srb1.receive(&pdu).unwrap().sdu).unwrap()
    else {
        panic!("not a reconfiguration");
    };
    r
}

/// (E-RAB ID, radio network cause) of each item of an E-RAB List.
fn failed_items(l: &E_RABList) -> Vec<(u8, u8)> {
    l.0.iter()
        .map(|i| {
            let E_RABList_EntryValue::Id_E_RABItem(i) = &i.value;
            let Cause::RadioNetwork(c) = &i.cause else {
                panic!("{:?}", i.cause);
            };
            (i.e_rab_id.0, c.0)
        })
        .collect()
}

/// What an E-RAB response lists: the E-RABs that succeeded, and the failed
/// ones with their radio network causes.
type ErabAnswer<T> = (Vec<T>, Vec<(u8, u8)>);

/// E-RAB SETUP RESPONSE: the set-up E-RABs with their eNB TEIDs, the failed
/// ones with their causes.
fn setup_response(acts: &[Action]) -> ErabAnswer<(u8, u32)> {
    let (_, pdu) = only_s1(acts);
    let S1AP_PDU::SuccessfulOutcome(SuccessfulOutcome {
        value: SuccessfulOutcomeValue::Id_E_RABSetup(r),
        ..
    }) = pdu
    else {
        panic!("not an E-RAB SETUP RESPONSE: {pdu:?}");
    };
    let (mut ok, mut failed) = (vec![], vec![]);
    for ie in r.protocol_i_es.0 {
        match ie.value {
            E_RABSetupResponseProtocolIEs_EntryValue::Id_E_RABSetupListBearerSURes(l) => {
                ok = l
                    .0
                    .iter()
                    .map(|i| {
                        let E_RABSetupListBearerSURes_EntryValue::Id_E_RABSetupItemBearerSURes(i) =
                            &i.value;
                        (
                            i.e_rab_id.0,
                            u32::from_be_bytes(i.gtp_teid.0.as_slice().try_into().unwrap()),
                        )
                    })
                    .collect();
            }
            E_RABSetupResponseProtocolIEs_EntryValue::Id_E_RABFailedToSetupListBearerSURes(l) => {
                failed = failed_items(&l);
            }
            _ => {}
        }
    }
    (ok, failed)
}

/// E-RAB RELEASE RESPONSE: the released E-RABs, the failed ones with causes.
fn release_response(acts: &[Action]) -> ErabAnswer<u8> {
    let (_, pdu) = only_s1(acts);
    let S1AP_PDU::SuccessfulOutcome(SuccessfulOutcome {
        value: SuccessfulOutcomeValue::Id_E_RABRelease(r),
        ..
    }) = pdu
    else {
        panic!("not an E-RAB RELEASE RESPONSE: {pdu:?}");
    };
    let (mut ok, mut failed) = (vec![], vec![]);
    for ie in r.protocol_i_es.0 {
        match ie.value {
            E_RABReleaseResponseProtocolIEs_EntryValue::Id_E_RABReleaseListBearerRelComp(l) => {
                ok = l
                    .0
                    .iter()
                    .map(|i| {
                        let E_RABReleaseListBearerRelComp_EntryValue::Id_E_RABReleaseItemBearerRelComp(i) =
                            &i.value;
                        i.e_rab_id.0
                    })
                    .collect();
            }
            E_RABReleaseResponseProtocolIEs_EntryValue::Id_E_RABFailedToReleaseList(l) => {
                failed = failed_items(&l);
            }
            _ => {}
        }
    }
    (ok, failed)
}

fn pending_tid(e: &EnbRrc) -> u8 {
    e.ues[&5]
        .erab
        .as_ref()
        .expect("an E-RAB procedure runs")
        .transaction_id
}

/// §8.2.1.2: the DRB goes to the UE first, with the NAS-PDU; the user plane
/// and E-RAB SETUP RESPONSE (with the eNB's TEID) follow its completion.
#[test]
fn an_erab_setup_adds_a_drb_and_answers_once_the_ue_completes() {
    let mut e = rrc();
    let enb = established(&mut e);
    let acts = e.on_erab_request(erab_setup(enb, vec![to_setup(6)]));
    assert!(
        !acts.iter().any(|a| matches!(a, Action::S1 { .. })),
        "{acts:?}"
    );
    let r = erab_reconfiguration(&acts, 0);
    assert_eq!(
        r.drbs,
        vec![DrbConfig {
            eps_bearer_identity: 6,
            drb_identity: 2,
            logical_channel_identity: 4,
            rlc: DrbRlc::UmSn10,
        }],
        "the lowest free DRB identity"
    );
    assert!(r.drbs_released.is_empty() && !r.srb2);
    assert_eq!(r.nas, vec![vec![0x27, 6]]);

    // Another transaction id completes nothing.
    assert!(e
        .on_reconfiguration_complete(5, r.transaction_id ^ 1)
        .is_empty());
    let acts = e.on_reconfiguration_complete(5, r.transaction_id);
    let Some(Action::UserPlane { ue: 5, bearers }) = acts.first() else {
        panic!("no user plane: {acts:?}");
    };
    let b = &bearers[0];
    assert_eq!(
        (b.e_rab_id, b.lcid, b.sgw, b.sgw_teid, b.eea, b.k_up_enc),
        (6, 4, "10.0.0.9:2152".parse().unwrap(), 0x5606, 2, [3; 16])
    );
    assert_eq!(setup_response(&acts), (vec![(6, b.enb_teid)], vec![]));
    assert!(e.ues[&5].erab.is_none());
    assert_eq!(e.ues[&5].drbs.len(), 2);
}

/// §8.2.1.4: an E-RAB ID given twice, or of an E-RAB already set up, fails
/// "Multiple E-RAB ID instances" -- once -- and the rest is set up; with
/// nothing to set up, or no security yet, the answer comes at once.
#[test]
fn an_erab_setup_fails_repeated_and_active_ids_and_answers_at_once_with_nothing_to_do() {
    let mut e = rrc();
    let enb = established(&mut e);
    let acts = e.on_erab_request(erab_setup(
        enb,
        vec![to_setup(5), to_setup(7), to_setup(7), to_setup(8)],
    ));
    let r = erab_reconfiguration(&acts, 0);
    assert_eq!(
        r.drbs
            .iter()
            .map(|d| d.eps_bearer_identity)
            .collect::<Vec<_>>(),
        vec![8]
    );
    let acts = e.on_reconfiguration_complete(5, r.transaction_id);
    let (ok, failed) = setup_response(&acts);
    assert_eq!(ok.iter().map(|(id, _)| *id).collect::<Vec<_>>(), vec![8]);
    let multiple = CauseRadioNetwork::MULTIPLE_E_RAB_ID_INSTANCES;
    assert_eq!(failed, vec![(5, multiple), (7, multiple)]);

    let acts = e.on_erab_request(erab_setup(enb, vec![to_setup(8)]));
    assert_eq!(setup_response(&acts), (vec![], vec![(8, multiple)]));
    assert!(e.ues[&5].erab.is_none());

    // §8.2.1.4: a GBR QCI without GBR QoS Information fails.
    let acts = e.on_erab_request(erab_setup(
        enb,
        vec![ErabToSetup {
            qci: 1,
            ..to_setup(9)
        }],
    ));
    assert_eq!(
        setup_response(&acts),
        (
            vec![],
            vec![(9, CauseRadioNetwork::INVALID_QOS_COMBINATION)]
        )
    );

    e.ues.get_mut(&5).unwrap().security = None;
    let acts = e.on_erab_request(erab_setup(enb, vec![to_setup(9)]));
    assert_eq!(
        setup_response(&acts),
        (vec![], vec![(9, CauseRadioNetwork::UNSPECIFIED)])
    );
}

/// §8.2.3.2.1: the DRB goes in drb-ToReleaseList with the NAS-PDU; its
/// S1-U tunnel and the E-RAB go once the UE completes. An unknown E-RAB ID
/// fails; with nothing to release the NAS-PDU still goes to the UE.
#[test]
fn an_erab_release_releases_the_drb_and_forwards_the_nas_pdu() {
    let mut e = rrc();
    let enb = established(&mut e);
    // §8.2.3.3: E-RAB 5 twice releases it once.
    let acts = e.on_erab_request(erab_release(enb, vec![5, 9, 5], Some(vec![0x27, 0xCD])));
    let r = erab_reconfiguration(&acts, 0);
    assert_eq!(
        (r.drbs_released, r.drbs, r.nas),
        (vec![1], vec![], vec![vec![0x27, 0xCD]])
    );
    let acts = e.on_reconfiguration_complete(5, r.transaction_id);
    assert!(
        matches!(acts.first(), Some(Action::ReleaseUserPlane { ue: 5, lcids }) if lcids == &[3]),
        "{acts:?}"
    );
    assert_eq!(
        release_response(&acts),
        (vec![5], vec![(9, CauseRadioNetwork::UNKNOWN_E_RAB_ID)])
    );
    assert!(e.ues[&5].drbs.is_empty());

    let acts = e.on_erab_request(erab_release(enb, vec![5], Some(vec![0x27, 0xCE])));
    assert_eq!(
        release_response(&acts),
        (vec![], vec![(5, CauseRadioNetwork::UNKNOWN_E_RAB_ID)])
    );
    let r = acts
        .iter()
        .find_map(|a| match a {
            Action::Rrc { pdu, .. } => {
                let mut ue_srb1 = LteSrbPdcp::new();
                ue_srb1.receive(&[0, 0, 0, 0, 0]).unwrap();
                Some(msg::decode_dl_dcch(&ue_srb1.receive(pdu).unwrap().sdu).unwrap())
            }
            _ => None,
        })
        .expect("the NAS-PDU to the UE");
    assert_eq!(
        r,
        msg::DlDcch::InformationTransfer {
            nas: vec![0x27, 0xCE]
        }
    );
}

/// §8.2.2.4: modifying an E-RAB the UE does not have fails "Unknown E-RAB
/// ID" at once; a known one goes to the UE and is answered on completion.
#[test]
fn an_erab_modify_of_an_unknown_id_or_to_gbr_without_gbr_information_fails() {
    let mut e = rrc();
    let enb = established(&mut e);
    let modify = |ids: &[u8]| {
        ErabRequest::Modify(s1ap_ue::ErabModifyRequest {
            mme_ue_s1ap_id: 77,
            enb_ue_s1ap_id: enb,
            erabs: ids
                .iter()
                .map(|id| s1ap_ue::ErabToModify {
                    e_rab_id: *id,
                    qci: 6,
                    gbr_info: false,
                    nas: vec![0x27, 0xC9, *id],
                })
                .collect(),
        })
    };
    let modify_response = |acts: &[Action]| {
        let (_, pdu) = only_s1(acts);
        let S1AP_PDU::SuccessfulOutcome(SuccessfulOutcome {
            value: SuccessfulOutcomeValue::Id_E_RABModify(r),
            ..
        }) = pdu
        else {
            panic!("not an E-RAB MODIFY RESPONSE: {pdu:?}");
        };
        let (mut ok, mut failed) = (vec![], vec![]);
        for ie in r.protocol_i_es.0 {
            match ie.value {
                E_RABModifyResponseProtocolIEs_EntryValue::Id_E_RABModifyListBearerModRes(l) => {
                    ok = l
                        .0
                        .iter()
                        .map(|i| {
                            let E_RABModifyListBearerModRes_EntryValue::Id_E_RABModifyItemBearerModRes(i) = &i.value;
                            i.e_rab_id.0
                        })
                        .collect();
                }
                E_RABModifyResponseProtocolIEs_EntryValue::Id_E_RABFailedToModifyList(l) => {
                    failed = failed_items(&l);
                }
                _ => {}
            }
        }
        (ok, failed)
    };
    let unknown = CauseRadioNetwork::UNKNOWN_E_RAB_ID;
    let acts = e.on_erab_request(modify(&[9]));
    assert_eq!(modify_response(&acts), (vec![], vec![(9, unknown)]));

    let acts = e.on_erab_request(modify(&[5, 9]));
    let r = erab_reconfiguration(&acts, 0);
    assert_eq!(
        (r.drbs, r.drbs_released, r.nas),
        (vec![], vec![], vec![vec![0x27, 0xC9, 5]])
    );
    let acts = e.on_reconfiguration_complete(5, r.transaction_id);
    assert_eq!(modify_response(&acts), (vec![5], vec![(9, unknown)]));
    assert_eq!(e.ues[&5].erab_qci.get(&5), Some(&6), "the new QCI kept");

    // A non-GBR E-RAB turned GBR (QCI 1) needs GBR QoS Information; once GBR,
    // a modification within GBR does not.
    let to_gbr = |gbr_info| {
        ErabRequest::Modify(s1ap_ue::ErabModifyRequest {
            mme_ue_s1ap_id: 77,
            enb_ue_s1ap_id: enb,
            erabs: vec![s1ap_ue::ErabToModify {
                e_rab_id: 5,
                qci: 1,
                gbr_info,
                nas: vec![0x27, 0xC9],
            }],
        })
    };
    let acts = e.on_erab_request(to_gbr(false));
    assert_eq!(
        modify_response(&acts),
        (
            vec![],
            vec![(5, CauseRadioNetwork::INVALID_QOS_COMBINATION)]
        )
    );
    for gbr_info in [true, false] {
        let acts = e.on_erab_request(to_gbr(gbr_info));
        let tid = erab_reconfiguration(&acts, 1 + u32::from(!gbr_info)).transaction_id;
        let acts = e.on_reconfiguration_complete(5, tid);
        assert_eq!(
            modify_response(&acts),
            (vec![5], vec![]),
            "GBR info {gbr_info}"
        );
    }
}

/// One E-RAB procedure at a time per UE, none during Initial Context Setup:
/// a later request waits and starts when the running one completes.
#[test]
fn an_erab_request_waits_for_the_reconfiguration_in_progress() {
    let mut e = rrc();
    let enb = established(&mut e);
    let acts = e.on_erab_request(erab_setup(enb, vec![to_setup(6)]));
    let first = pending_tid(&e);
    assert_eq!(acts.len(), 1);
    assert!(e
        .on_erab_request(erab_release(enb, vec![5], None))
        .is_empty());

    let acts = e.on_reconfiguration_complete(5, first);
    let (ok, _) = setup_response(&acts);
    assert_eq!(ok.len(), 1);
    let r = erab_reconfiguration(&acts, 1);
    assert_eq!(r.drbs_released, vec![1], "the release follows the setup");
    assert_ne!(r.transaction_id, first);
    let acts = e.on_reconfiguration_complete(5, r.transaction_id);
    assert_eq!(release_response(&acts), (vec![5], vec![]));

    // During an Initial Context Setup nothing starts.
    e.ues.get_mut(&5).unwrap().ics = Some(PendingIcs {
        erabs: vec![],
        notify: vec![],
        rx_ciphering: (0, [0; 16]),
        smc_transaction_id: 0,
        reconfiguration_transaction_id: None,
        teids: vec![],
    });
    assert!(e
        .on_erab_request(erab_release(enb, vec![6], None))
        .is_empty());
    assert_eq!(e.ues[&5].erab_queue.len(), 1);
}

/// An E-RAB procedure for a UE this eNB does not know is answered with an
/// ERROR INDICATION "Unknown eNB UE S1AP ID" (§8.2.1.4 by way of §10.6).
#[test]
fn an_erab_request_for_an_unknown_ue_is_reported() {
    let mut e = rrc();
    let enb = established(&mut e);
    let acts = e.on_erab_request(erab_setup(enb + 1, vec![to_setup(6)]));
    let (stream, pdu) = only_s1(&acts);
    assert_eq!(stream, NON_UE_ASSOCIATED_STREAM);
    assert!(
        matches!(
            pdu,
            S1AP_PDU::InitiatingMessage(InitiatingMessage {
                value: InitiatingMessageValue::Id_ErrorIndication(_),
                ..
            })
        ),
        "{pdu:?}"
    );
}
