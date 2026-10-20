use super::*;
use std::sync::{Arc, Mutex};

const CELL: u64 = 0x19B01;
const K_ENB: [u8; 32] = [0x42; 32];

#[derive(Default)]
struct Seen {
    initial_asked: usize,
    indications: Vec<RrcIndication>,
    downlink: Vec<Vec<u8>>,
}

/// A NAS that requests a connection with a fixed message, echoes a fixed
/// reply to downlink NAS, and records what RRC tells it.
struct Scripted {
    seen: Arc<Mutex<Seen>>,
    initial: Option<Vec<u8>>,
    reply: Option<Vec<u8>>,
    k_enb: Option<[u8; 32]>,
}

impl NasLayer for Scripted {
    fn initial_nas(&mut self) -> Option<InitialNas> {
        self.seen.lock().unwrap().initial_asked += 1;
        self.initial.clone().map(|nas| InitialNas {
            nas,
            cause: msg::EstablishmentCauseValue::MoSignalling,
            s_tmsi: None,
        })
    }
    fn on_downlink_nas(&mut self, nas: &[u8]) -> Vec<Vec<u8>> {
        self.seen.lock().unwrap().downlink.push(nas.to_vec());
        self.reply.clone().into_iter().collect()
    }
    fn on_rrc_indication(&mut self, ind: RrcIndication) {
        self.seen.lock().unwrap().indications.push(ind);
    }
    fn k_enb(&mut self) -> Option<[u8; 32]> {
        self.k_enb
    }
}

fn plmn() -> Plmn {
    Plmn::new(999, 70, false)
}

fn ue(initial: Option<&[u8]>) -> (UeRrc, Arc<Mutex<Seen>>) {
    let seen = Arc::new(Mutex::new(Seen::default()));
    let nas = Scripted {
        seen: Arc::clone(&seen),
        initial: initial.map(<[u8]>::to_vec),
        reply: Some(vec![0x07, 0x53]),
        k_enb: Some(K_ENB),
    };
    (
        UeRrc::new(plmn(), Box::new(nas), Box::new(|| 0x12_3456_789A)),
        seen,
    )
}

fn cell_info() -> CellInfo {
    CellInfo {
        plmns: vec![PlmnId {
            mcc: 999,
            mnc: 70,
            long_mnc: false,
        }],
        tac: 1,
        cell_identity: 0x19B01,
        cell_barred: false,
        q_rx_lev_min_dbm: -140,
        freq_band: 7,
        system_info_value_tag: 0,
    }
}

fn sib2(t300_ms: u16) -> Vec<u8> {
    msg::encode_sib2_si(&Sib2Info {
        timers: msg::UeTimers {
            t300_ms,
            ..msg::UeTimers::default()
        },
        default_paging_cycle: msg::PagingCycle::Rf128,
    })
    .unwrap()
}

fn mib() -> Vec<u8> {
    msg::encode_mib(&MibInfo {
        dl_bandwidth: msg::DlBandwidth::N50,
        sfn: 0,
    })
    .unwrap()
}

/// Measure the cell and deliver MIB, SIB1 and (optionally) SIB2.
fn acquire(u: &mut UeRrc, cell: CellInfo, with_sib2: bool, now: Instant) -> Vec<UeAction> {
    let mut acts = u.on_cell_measurement(CELL, 0, now);
    acts.extend(u.on_downlink(CELL, RrcChannel::BcchBch, 0, &mib(), now));
    acts.extend(u.on_downlink(
        CELL,
        RrcChannel::BcchDlSch,
        0,
        &msg::encode_sib1(&cell).unwrap(),
        now,
    ));
    if with_sib2 {
        acts.extend(u.on_downlink(CELL, RrcChannel::BcchDlSch, 0, &sib2(1000), now));
    }
    acts
}

#[test]
fn camps_on_a_suitable_cell_but_waits_for_sib2_before_requesting() {
    let now = Instant::now();
    let (mut u, _) = ue(Some(&[0x07, 0x41]));
    assert!(acquire(&mut u, cell_info(), false, now).is_empty());
    assert_eq!(u.state(), UeRrcState::Camped { cell: CELL });
    let acts = u.on_downlink(CELL, RrcChannel::BcchDlSch, 0, &sib2(1000), now);
    let [UeAction::Rrc {
        channel: RrcChannel::UlCcch,
        pdu,
        ..
    }] = acts.as_slice()
    else {
        panic!("{acts:?}");
    };
    assert_eq!(
        msg::decode_ul_ccch(pdu).unwrap(),
        msg::UlCcch::ConnectionRequest {
            identity: InitialUeIdentity::Random(0x12_3456_789A),
            cause: msg::EstablishmentCauseValue::MoSignalling
        }
    );
    assert_eq!(u.state(), UeRrcState::Connecting { cell: CELL });
    assert_eq!(u.next_deadline(), Some(now + Duration::from_millis(1000)));
}

#[test]
fn unsuitable_cells_are_not_camped_on() {
    let now = Instant::now();
    // Wrong PLMN.
    let (mut u, _) = ue(None);
    let mut c = cell_info();
    c.plmns[0].mnc = 71;
    acquire(&mut u, c, true, now);
    assert_eq!(u.state(), UeRrcState::NoCell);
    // Barred.
    let (mut u, _) = ue(None);
    let mut c = cell_info();
    c.cell_barred = true;
    acquire(&mut u, c, true, now);
    assert_eq!(u.state(), UeRrcState::NoCell);
    // S criterion: Srxlev = -150 - (-140) = -10 dB, not > 0.
    let (mut u, _) = ue(None);
    acquire(&mut u, cell_info(), true, now);
    u.on_cell_measurement(CELL, -150, now);
    assert_eq!(u.state(), UeRrcState::NoCell);
    // Srxlev = 1 dB is enough.
    u.on_cell_measurement(CELL, -139, now);
    assert_eq!(u.state(), UeRrcState::Camped { cell: CELL });
}

#[test]
fn the_strongest_suitable_cell_is_selected() {
    let now = Instant::now();
    let (mut u, _) = ue(None);
    for (sti, dbm) in [(1u64, -90), (2u64, -60)] {
        u.on_cell_measurement(sti, dbm, now);
        u.on_downlink(sti, RrcChannel::BcchBch, 0, &mib(), now);
        u.on_downlink(
            sti,
            RrcChannel::BcchDlSch,
            0,
            &msg::encode_sib1(&cell_info()).unwrap(),
            now,
        );
    }
    assert_eq!(u.state(), UeRrcState::Camped { cell: 2 });
    u.on_cell_lost(2, now);
    assert_eq!(u.state(), UeRrcState::Camped { cell: 1 });
}

fn setup(tid: u8) -> Vec<u8> {
    msg::encode_rrc_connection_setup(tid).unwrap()
}

#[test]
fn setup_is_answered_with_complete_over_srb1_pdcp() {
    let now = Instant::now();
    let (mut u, seen) = ue(Some(&[0x07, 0x41, 0x01]));
    acquire(&mut u, cell_info(), true, now);
    let acts = u.on_downlink(CELL, RrcChannel::DlCcch, 0, &setup(2), now);
    let [UeAction::Rrc {
        channel: RrcChannel::UlDcch,
        pdu,
        ..
    }] = acts.as_slice()
    else {
        panic!("{acts:?}");
    };
    let mut enb_srb1 = LteSrbPdcp::new();
    let r = enb_srb1.receive(pdu).unwrap();
    let (sdu, count) = (r.sdu, r.count);
    assert_eq!(count, 0);
    assert_eq!(
        msg::decode_ul_dcch(&sdu).unwrap(),
        msg::UlDcch::ConnectionSetupComplete {
            transaction_id: 2,
            selected_plmn_index: 1,
            nas: vec![0x07, 0x41, 0x01]
        }
    );
    assert_eq!(u.state(), UeRrcState::Connected { cell: CELL });
    assert_eq!(u.next_deadline(), None, "T300 stopped");
    assert_eq!(
        seen.lock().unwrap().indications,
        vec![RrcIndication::Connected]
    );
}

#[test]
fn t300_expiry_fails_the_establishment() {
    let now = Instant::now();
    let (mut u, seen) = ue(Some(&[0x07, 0x41]));
    acquire(&mut u, cell_info(), true, now);
    assert!(u.on_timer(now + Duration::from_millis(999)).is_empty());
    assert_eq!(u.state(), UeRrcState::Connecting { cell: CELL });
    // At expiry the UE goes idle and NAS is told; this scripted NAS asks again
    // at once, so a new request follows.
    let acts = u.on_timer(now + Duration::from_millis(1000));
    assert_eq!(
        seen.lock().unwrap().indications,
        vec![RrcIndication::EstablishmentFailed { wait_time_s: None }]
    );
    assert!(matches!(
        acts.as_slice(),
        [UeAction::Rrc {
            channel: RrcChannel::UlCcch,
            ..
        }]
    ));
}

#[test]
fn reject_bars_establishment_for_the_wait_time() {
    let now = Instant::now();
    let (mut u, seen) = ue(Some(&[0x07, 0x41]));
    acquire(&mut u, cell_info(), true, now);
    let reject = msg::encode_rrc_connection_reject(5).unwrap();
    assert!(u
        .on_downlink(CELL, RrcChannel::DlCcch, 0, &reject, now)
        .is_empty());
    assert_eq!(
        seen.lock().unwrap().indications,
        vec![RrcIndication::EstablishmentFailed {
            wait_time_s: Some(5)
        }]
    );
    assert!(u.on_timer(now + Duration::from_millis(4999)).is_empty());
    assert_eq!(u.next_deadline(), Some(now + Duration::from_secs(5)));
    assert_eq!(u.on_timer(now + Duration::from_secs(5)).len(), 1);
}

#[test]
fn downlink_nas_is_delivered_and_replies_go_out_as_ul_information_transfer() {
    let now = Instant::now();
    let (mut u, seen) = ue(Some(&[0x07, 0x41]));
    acquire(&mut u, cell_info(), true, now);
    u.on_downlink(CELL, RrcChannel::DlCcch, 0, &setup(0), now);
    let mut enb_tx = LteSrbPdcp::new();
    let dl = msg::encode_dl_information_transfer(1, &[0x07, 0x52]).unwrap();
    let acts = u.on_downlink(
        CELL,
        RrcChannel::DlDcch,
        LCID_SRB1,
        &enb_tx.transmit(&dl).0,
        now,
    );
    assert_eq!(seen.lock().unwrap().downlink, vec![vec![0x07, 0x52]]);
    let [UeAction::Rrc { pdu, .. }] = acts.as_slice() else {
        panic!("{acts:?}");
    };
    let mut enb_rx = LteSrbPdcp::new();
    enb_rx.receive(&[0, 0, 0, 0, 0]).unwrap(); // the SetupComplete before it
    let r = enb_rx.receive(pdu).unwrap();
    let (sdu, count) = (r.sdu, r.count);
    assert_eq!(count, 1, "second uplink SRB1 PDU");
    assert_eq!(
        msg::decode_ul_dcch(&sdu).unwrap(),
        msg::UlDcch::InformationTransfer {
            nas: vec![0x07, 0x53]
        }
    );
}

#[test]
fn release_and_radio_link_loss_return_to_idle() {
    let now = Instant::now();
    let (mut u, seen) = ue(None);
    acquire(&mut u, cell_info(), true, now);
    // Force a connection with a NAS that asks once.
    let (mut u2, seen2) = ue(Some(&[0x07, 0x41]));
    acquire(&mut u2, cell_info(), true, now);
    u2.on_downlink(CELL, RrcChannel::DlCcch, 0, &setup(0), now);
    let mut enb_tx = LteSrbPdcp::new();
    let release = msg::encode_rrc_connection_release(0).unwrap();
    u2.on_downlink(
        CELL,
        RrcChannel::DlDcch,
        LCID_SRB1,
        &enb_tx.transmit(&release).0,
        now,
    );
    assert!(seen2
        .lock()
        .unwrap()
        .indications
        .contains(&RrcIndication::Released));

    // Radio link failure on an idle camped UE: nothing to tell NAS.
    u.on_cell_lost(CELL, now);
    assert_eq!(u.state(), UeRrcState::NoCell);
    assert!(seen.lock().unwrap().indications.is_empty());
}

#[test]
fn a_ue_with_no_nas_request_only_camps() {
    let now = Instant::now();
    let (mut u, seen) = ue(None);
    assert!(acquire(&mut u, cell_info(), true, now).is_empty());
    assert_eq!(u.state(), UeRrcState::Camped { cell: CELL });
    assert!(seen.lock().unwrap().initial_asked > 0);
}

/// After the AS SMC and the RRCConnectionReconfiguration that adds SRB2 and a
/// DRB: downlink NAS on SRB2 is answered on SRB2, and IP packets cross the
/// DRB both ways. The eNB side is built here from the PDCP/RLC entities and
/// the TS 33.401 key derivation directly.
#[test]
fn after_reconfiguration_nas_uses_srb2_and_ip_crosses_the_drb() {
    use nextgsim_pdcp::lte_drb::drb_bearer;
    let now = Instant::now();
    let (mut u, seen) = ue(Some(&[0x07, 0x41]));
    acquire(&mut u, cell_info(), true, now);
    u.on_downlink(CELL, RrcChannel::DlCcch, 0, &setup(0), now);
    let k = |t| derive_algorithm_key(&K_ENB, t, 2);
    let (k_int, k_enc, k_up) = (
        k(EpsAlgorithmType::RrcInt),
        k(EpsAlgorithmType::RrcEnc),
        k(EpsAlgorithmType::UpEnc),
    );
    // AS SMC, EEA2/EIA2, as the eNB sends it (TS 33.401 §7.2.4.5).
    let mut srb1 = LteSrbPdcp::with_bearer(SRB1_BEARER, DIRECTION_DOWNLINK);
    srb1.activate_tx_integrity(2, k_int).unwrap();
    let smc = msg::encode_security_mode_command(1, 2, 2).unwrap();
    let acts = u.on_downlink(
        CELL,
        RrcChannel::DlDcch,
        LCID_SRB1,
        &srb1.transmit(&smc).0,
        now,
    );
    assert!(matches!(
        acts.as_slice(),
        [UeAction::Rrc {
            lcid: LCID_SRB1,
            ..
        }]
    ));
    srb1.activate_tx_ciphering(2, k_enc).unwrap();
    let drb = DrbConfig {
        eps_bearer_identity: 5,
        drb_identity: 1,
        logical_channel_identity: 3,
        rlc: DrbRlc::UmSn10,
    };
    let reconf = msg::encode_rrc_connection_reconfiguration(2, true, &[drb], &[]).unwrap();
    let acts = u.on_downlink(
        CELL,
        RrcChannel::DlDcch,
        LCID_SRB1,
        &srb1.transmit(&reconf).0,
        now,
    );
    assert!(
        matches!(
            acts.as_slice(),
            [UeAction::Rrc {
                lcid: LCID_SRB1,
                ..
            }]
        ),
        "ReconfigurationComplete goes on SRB1: {acts:?}"
    );
    assert_eq!(u.drbs(), vec![drb]);

    // Downlink NAS on SRB2 (BEARER 1, SRB1's keys); the reply comes back on SRB2.
    let mut srb2 = LteSrbPdcp::with_bearer(1, DIRECTION_DOWNLINK);
    srb2.activate_tx_integrity(2, k_int).unwrap();
    srb2.activate_rx_integrity(2, k_int).unwrap();
    srb2.activate_tx_ciphering(2, k_enc).unwrap();
    srb2.activate_rx_ciphering(2, k_enc).unwrap();
    let dl = msg::encode_dl_information_transfer(3, &[0x07, 0x52]).unwrap();
    let acts = u.on_downlink(
        CELL,
        RrcChannel::DlDcch,
        LCID_SRB2,
        &srb2.transmit(&dl).0,
        now,
    );
    assert_eq!(
        seen.lock().unwrap().downlink.last().unwrap(),
        &vec![0x07, 0x52]
    );
    let [UeAction::Rrc {
        lcid: LCID_SRB2,
        pdu,
        ..
    }] = acts.as_slice()
    else {
        panic!("NAS reply not on SRB2: {acts:?}");
    };
    assert_eq!(
        msg::decode_ul_dcch(&srb2.receive(pdu).unwrap().sdu).unwrap(),
        msg::UlDcch::InformationTransfer {
            nas: vec![0x07, 0x53]
        }
    );

    // SRB1 still works with SRB2 up: its PDCP is chosen by LCID, not by
    // "SRB2 exists" (a downlink NAS on SRB1 is deciphered with SRB1's COUNT).
    let dl1 = msg::encode_dl_information_transfer(0, &[0x07, 0x61]).unwrap();
    u.on_downlink(
        CELL,
        RrcChannel::DlDcch,
        LCID_SRB1,
        &srb1.transmit(&dl1).0,
        now,
    );
    assert_eq!(
        seen.lock().unwrap().downlink.last().unwrap(),
        &vec![0x07, 0x61]
    );
    assert_eq!(u.state(), UeRrcState::Connected { cell: CELL });

    // The DRB: uplink and downlink through PDCP (EEA2, K_UPenc) and RLC UM.
    let mut enb_pdcp = LteDrbPdcp::new(drb_bearer(3), DIRECTION_DOWNLINK, 2, k_up).unwrap();
    let mut enb_rx = LteUmRx::new(DRB_T_REORDERING);
    let mut enb_tx = LteUmTx::new();
    let ip = vec![0x45, 0, 0, 20, 7, 7, 7, 7];
    let ul = u.send_ip(&ip);
    let [UeAction::Data {
        cell: CELL,
        lcid: 3,
        pdu,
    }] = ul.as_slice()
    else {
        panic!("no uplink data PDU");
    };
    let sdus = enb_rx.receive(pdu, now).unwrap();
    assert_eq!(enb_pdcp.receive(&sdus[0]).unwrap().0, ip);
    let down = enb_tx.transmit(&enb_pdcp.transmit(&ip).0);
    assert_eq!(
        u.on_data(CELL, 3, &down, now),
        vec![UeAction::DeliverIp(ip)]
    );
    // A data PDU on an LCID with no DRB is not delivered.
    assert!(u.on_data(CELL, 4, &down, now).is_empty());
}
