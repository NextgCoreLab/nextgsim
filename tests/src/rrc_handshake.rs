//! Wave-6 C3 — In-process strict-peer gNB<->UE RRC handshake test.
//!
//! Drives the REAL gNB and UE RRC task handlers end-to-end in one process
//! (our own gNB and our own UE as peers — no mocks, no Open5GS):
//!
//! ```text
//! UE start_connection_establishment (real ASN.1 RRCSetupRequest, UL-CCCH)
//!   -> gNB handle_uplink_rrc / handle_ul_ccch_message
//!   -> captured RRCSetup PDU (DL-CCCH)
//!   -> UE handle_downlink_rrc / handle_dl_ccch_message
//!   -> captured RRCSetupComplete PDU (UL-DCCH, primary ASN.1 encoding)
//!   -> gNB handle_uplink_rrc / handle_ul_dcch_message
//!   -> NgapMessage::InitialNasDelivery whose NAS bytes are byte-for-byte
//!      equal to the original registration PDU.
//! ```
//!
//! Pins the C3 discovery: the UE's primary ASN.1 UL-DCCH RRCSetupComplete
//! (TS 38.331 §6.2.2, c1 CHOICE index 2, tid 0 → leading byte 0x10) matched
//! NO arm of the gNB's legacy nibble dispatcher (`bytes[0] & 0x0F`) and was
//! silently dropped — the registration NAS never became an Initial UE
//! Message. The gNB now decodes UL-DCCH ASN.1-first (additive-accept; the
//! bespoke fallback framing is still accepted until C6).
//!
//! The SMC hop is the C5 green case (issue #151): the gNB's real ASN.1
//! SecurityModeCommand (DL-DCCH c1 index 4, tid 0 → leading byte 0x20, low
//! nibble 0x0) used to be misrouted by the UE's nibble dispatcher into its
//! RRC-reconfiguration arm, which answered with the bespoke
//! ReconfigurationComplete `[0x08, smc[1]]` that the gNB then dropped. Both ends
//! now dispatch on the decoded message, so the command REACHES the UE's AS
//! security handler and the UE's answer reaches the gNB.
//!
//! Reference: TS 38.331 §5.3.3 (RRC connection establishment), §6.2
//! (UL-DCCH-Message / DL-DCCH-Message classes).

use nextgsim_common::config::{GnbConfig, UeConfig};
use nextgsim_common::{OctetString, Plmn};
use nextgsim_gnb::tasks::{
    GnbTaskBase, NgapMessage, RlsMessage as GnbRlsMessage, TaskMessage as GnbTaskMessage,
};
use nextgsim_gnb::RrcTask as GnbRrcTask;
use nextgsim_rls::RrcChannel;
use nextgsim_rrc::procedures::security_mode::{
    decode_security_mode_failure, encode_security_mode_command, CipheringAlgorithmType,
    IntegrityAlgorithmType, SecurityAlgorithms, SecurityModeCommandParams,
};
use nextgsim_ue::{
    NasMessage, RlsMessage as UeRlsMessage, RrcTask as UeRrcTask, TaskMessage as UeTaskMessage,
    UeTaskBase,
};
use tokio::sync::mpsc;

const CELL_ID: i32 = 1;
const UE_ID: i32 = 1;

/// Registration Request stand-in NAS PDU (5GMM EPD 0x7E, Registration Request
/// message type 0x41). The exact bytes only matter for the byte-equality
/// assertion at the NGAP boundary.
fn registration_nas() -> Vec<u8> {
    vec![0x7E, 0x00, 0x41, 0x79, 0x00, 0x0D, 0x01, 0x00, 0xF1, 0x10]
}

fn gnb_config() -> GnbConfig {
    GnbConfig {
        nci: 0x000000010,
        gnb_id_length: 32,
        plmn: Plmn::new(1, 1, false),
        tac: 1,
        ..Default::default()
    }
}

/// The in-process strict-peer harness: the real UE and gNB RRC tasks plus the
/// channel receivers on which their real handlers emit messages.
struct Harness {
    ue: UeRrcTask,
    gnb: GnbRrcTask,
    ue_nas_rx: mpsc::Receiver<UeTaskMessage<NasMessage>>,
    ue_rls_rx: mpsc::Receiver<UeTaskMessage<UeRlsMessage>>,
    gnb_ngap_rx: mpsc::Receiver<GnbTaskMessage<NgapMessage>>,
    gnb_rls_rx: mpsc::Receiver<GnbTaskMessage<GnbRlsMessage>>,
}

fn harness() -> Harness {
    let (ue_base, _ue_app_rx, ue_nas_rx, _ue_rrc_rx, ue_rls_rx) =
        UeTaskBase::new(UeConfig::default(), 32);
    let ue = UeRrcTask::new(ue_base);

    let (gnb_base, _gnb_app_rx, gnb_ngap_rx, _gnb_rrc_rx, _gnb_gtp_rx, gnb_rls_rx, _gnb_sctp_rx) =
        GnbTaskBase::new(gnb_config(), 32);
    let gnb = GnbRrcTask::new(gnb_base);

    Harness {
        ue,
        gnb,
        ue_nas_rx,
        ue_rls_rx,
        gnb_ngap_rx,
        gnb_rls_rx,
    }
}

/// Pops the next uplink RRC PDU the UE handed to its RLS (skipping
/// non-PDU RLS traffic such as `AssignCurrentCell`). All handler sends have
/// completed by the time the driving `await` returns, so `try_recv` is
/// deterministic here.
fn next_ue_uplink_rrc(
    rx: &mut mpsc::Receiver<UeTaskMessage<UeRlsMessage>>,
) -> (RrcChannel, OctetString) {
    loop {
        match rx.try_recv() {
            Ok(UeTaskMessage::Message(UeRlsMessage::RrcPduDelivery { channel, pdu, .. })) => {
                return (channel, pdu)
            }
            Ok(_) => continue,
            Err(e) => panic!("expected an uplink RRC PDU from the UE, got none: {e}"),
        }
    }
}

/// Pops the next downlink RRC PDU the gNB handed to its RLS.
fn next_gnb_downlink_rrc(
    rx: &mut mpsc::Receiver<GnbTaskMessage<GnbRlsMessage>>,
) -> (i32, RrcChannel, OctetString) {
    loop {
        match rx.try_recv() {
            Ok(GnbTaskMessage::Message(GnbRlsMessage::DownlinkRrc {
                ue_id,
                rrc_channel,
                data,
                ..
            })) => return (ue_id, rrc_channel, data),
            Ok(_) => continue,
            Err(e) => panic!("expected a downlink RRC PDU from the gNB, got none: {e}"),
        }
    }
}

/// Pops the next NGAP-bound Initial UE Message, if any was emitted.
fn try_take_initial_nas(
    rx: &mut mpsc::Receiver<GnbTaskMessage<NgapMessage>>,
) -> Option<(i32, OctetString, i64)> {
    while let Ok(msg) = rx.try_recv() {
        if let GnbTaskMessage::Message(NgapMessage::InitialNasDelivery {
            ue_id,
            pdu,
            rrc_establishment_cause,
            ..
        }) = msg
        {
            return Some((ue_id, pdu, rrc_establishment_cause));
        }
    }
    None
}

/// Returns true if any NAS PDU was forwarded to the UE NAS task.
fn ue_received_nas_delivery(rx: &mut mpsc::Receiver<UeTaskMessage<NasMessage>>) -> bool {
    while let Ok(msg) = rx.try_recv() {
        if let UeTaskMessage::Message(NasMessage::NasDelivery { .. }) = msg {
            return true;
        }
    }
    false
}

/// Drives the full RRC connection establishment through the REAL handlers of
/// both peers and returns the RRCSetupComplete PDU the UE emitted plus the
/// registration NAS bytes it carried.
async fn run_setup_handshake(h: &mut Harness) -> (OctetString, Vec<u8>) {
    // gNB: NG Setup done -> radio power on (unbars the cell).
    h.gnb.handle_radio_power_on();

    // UE: camp on the simulated cell via the real cell-detection +
    // cell-selection path (first perform_cycle call is not time-gated).
    h.ue.handle_signal_changed(CELL_ID, -60).await;
    h.ue.perform_cycle().await;

    // NAS hands the initial Registration Request to RRC -> the UE encodes a
    // real ASN.1 RRCSetupRequest on UL-CCCH.
    let nas = registration_nas();
    h.ue.handle_uplink_nas_delivery(1, OctetString::from_slice(&nas))
        .await;
    let (ch, setup_req) = next_ue_uplink_rrc(&mut h.ue_rls_rx);
    assert_eq!(ch, RrcChannel::UlCcch, "RRCSetupRequest must go on UL-CCCH");

    // gNB: UL-CCCH -> RRCSetup on DL-CCCH (fresh gNB task -> tid 0).
    h.gnb
        .handle_uplink_rrc(UE_ID, RrcChannel::UlCcch, setup_req)
        .await;
    let (gnb_ue, ch, rrc_setup) = next_gnb_downlink_rrc(&mut h.gnb_rls_rx);
    assert_eq!(gnb_ue, UE_ID);
    assert_eq!(ch, RrcChannel::DlCcch, "RRCSetup must go on DL-CCCH");
    // Wave-6 C1: the live RRCSetup must be EXACTLY the hand-derived golden
    // UPER PDU carrying the SRB1 configuration (TS 38.331 §5.3.5.6.3;
    // leading byte 0x20 = DL-CCCH c1=rrcSetup, tid 0). Bit-by-bit derivation:
    // nextgsim-rrc rrc_setup.rs `golden_rrc_setup_srb1_bytes`.
    assert_eq!(
        rrc_setup.data(),
        &[0x20, 0x40, 0x00, 0x22, 0x00, 0x04, 0x00, 0x00][..],
        "RRCSetup must be the C1 golden SRB1 PDU (tid 0)"
    );

    // UE: DL-CCCH -> Connected; emits RRCSetupComplete on UL-DCCH.
    h.ue.handle_downlink_rrc(CELL_ID, RrcChannel::DlCcch, rrc_setup)
        .await;
    let (ch, setup_complete) = next_ue_uplink_rrc(&mut h.ue_rls_rx);
    assert_eq!(
        ch,
        RrcChannel::UlDcch,
        "RRCSetupComplete must go on UL-DCCH"
    );

    (setup_complete, nas)
}

/// The C3 strict-peer gate: the full RRCSetupRequest -> InitialUEMessage
/// chain through the real handlers of both stacks, with an exact NAS
/// byte-equality assertion at the NGAP boundary.
#[tokio::test]
async fn gnb_ue_rrc_setup_handshake_delivers_initial_nas_byte_equal() {
    let mut h = harness();
    let (setup_complete, nas) = run_setup_handshake(&mut h).await;

    // Wave-6 C2: the UE decoded the live RRCSetup and recorded SRB1
    // (TS 38.331 §5.3.3.4 / §5.3.5.6.3) before any DL-DCCH handling.
    let srb1 =
        h.ue.srb1_config()
            .expect("UE must record SRB1 from the live RRCSetup (C2)");
    assert_eq!(srb1.rrc_transaction_id, 0);

    // Pin the C3 finding: the UE's PRIMARY encoding is the real ASN.1
    // UL-DCCH-Message (c1 index 2 = rrcSetupComplete, tid 0 -> leading byte
    // 0x10, low nibble 0x0). This is NOT the bespoke fallback framing
    // [0x04, 0x00, 0x01, NAS...]; the legacy gNB nibble dispatcher had no
    // arm for nibble 0x0 and dropped it.
    assert_eq!(
        setup_complete.data()[0],
        0x10,
        "UE primary RRCSetupComplete must be ASN.1 UL-DCCH (c1 idx 2, tid 0)"
    );
    assert_ne!(
        setup_complete.data()[0] & 0x0F,
        0x04,
        "must not be the bespoke fallback framing"
    );

    // gNB: UL-DCCH -> the ASN.1-first dispatch must accept it and emit the
    // NGAP-bound Initial UE Message with the NAS byte-for-byte intact.
    h.gnb
        .handle_uplink_rrc(UE_ID, RrcChannel::UlDcch, setup_complete)
        .await;
    let (ngap_ue, pdu, cause) = try_take_initial_nas(&mut h.gnb_ngap_rx).expect(
        "gNB must emit an Initial UE Message from the UE's ASN.1 RRCSetupComplete \
         (C3: previously dropped by the nibble dispatcher)",
    );
    assert_eq!(ngap_ue, UE_ID);
    assert_eq!(
        pdu.data(),
        &nas[..],
        "registration NAS must be byte-for-byte equal at the NGAP boundary"
    );
    // UE default establishment cause 3 -> mo-Signalling round-trips through
    // the ASN.1 RRCSetupRequest.
    assert_eq!(cause, 3, "establishment cause must survive the chain");
}

/// SMC hop, C5 green (issue #151): the gNB's real ASN.1 SecurityModeCommand
/// reaches the UE's AS-security handler rather than its reconfiguration arm, and
/// the UE's answer reaches the gNB rather than being dropped.
///
/// This test replaces `smc_hop_current_tolerated_behavior_pinned_for_c5_baseline`,
/// which pinned the misroute: the UE answered `[0x08, smc[1]]` -- a bespoke
/// ReconfigurationComplete -- and the gNB's UL-information-transfer arm discarded
/// it as too short.
///
/// The harness UE runs with `as_security_enabled` OFF, which is the default, so the
/// conformant answer is a `SecurityModeFailure`: the UE cannot verify the command's
/// integrity without an AS context, and TS 38.331 §5.3.4.4 answers "unable to
/// comply" with a failure rather than with silence. What matters here is that the
/// hop is no longer silent in either direction; the ACTIVATION path is covered by
/// the AS-security tests that set the flag.
#[tokio::test]
async fn the_smc_hop_reaches_the_ue_security_handler_and_answers_it() {
    let mut h = harness();
    let (setup_complete, _nas) = run_setup_handshake(&mut h).await;
    h.gnb
        .handle_uplink_rrc(UE_ID, RrcChannel::UlDcch, setup_complete)
        .await;
    // Drain handshake-time traffic so the SMC-hop assertions start clean.
    let _ = try_take_initial_nas(&mut h.gnb_ngap_rx);
    while h.gnb_rls_rx.try_recv().is_ok() {}
    while h.ue_nas_rx.try_recv().is_ok() {}

    // Encode the SecurityModeCommand exactly as the gNB NGAP task does at
    // Initial Context Setup (activate_as_security, ngap/task.rs): tid 0.
    let smc = encode_security_mode_command(&SecurityModeCommandParams {
        rrc_transaction_id: 0,
        security_algorithms: SecurityAlgorithms {
            ciphering_algorithm: CipheringAlgorithmType::Nea0,
            integrity_algorithm: Some(IntegrityAlgorithmType::Nia2),
        },
    })
    .expect("encode SecurityModeCommand");
    assert_eq!(
        smc[0], 0x20,
        "SMC: DL-DCCH c1 index 4, tid 0 -> leading byte 0x20 (low nibble 0x0)"
    );

    // UE: the typed DL-DCCH dispatch resolves c1 index 4 to the SecurityModeCommand
    // arm. With no AS context it answers a real SecurityModeFailure echoing the tid.
    h.ue.handle_downlink_rrc(CELL_ID, RrcChannel::DlDcch, OctetString::from_slice(&smc))
        .await;
    let (ch, response) = next_ue_uplink_rrc(&mut h.ue_rls_rx);
    assert_eq!(ch, RrcChannel::UlDcch);
    assert_eq!(
        decode_security_mode_failure(response.data()).expect(
            "the UE must answer a decodable SecurityModeFailure, not a bespoke \
             ReconfigurationComplete"
        ),
        0,
        "and it must refuse the tid the command carried"
    );
    assert_ne!(
        response.data(),
        &[0x08, smc[1]][..],
        "the bespoke ReconfigurationComplete is exactly what the misroute produced; \
         seeing it again means the SMC is being routed as a reconfiguration"
    );
    assert!(
        !ue_received_nas_delivery(&mut h.ue_nas_rx),
        "the SMC must not fabricate NAS toward the UE NAS task"
    );

    // gNB: the failure is RECOGNISED (its own message class, its own handler) and
    // produces no NGAP traffic — it is an RRC-local outcome.
    h.gnb
        .handle_uplink_rrc(UE_ID, RrcChannel::UlDcch, response)
        .await;
    assert!(
        h.gnb_ngap_rx.try_recv().is_err(),
        "a SecurityModeFailure is an RRC-local outcome and must not reach NGAP"
    );
}

/// The KgNB both ends of the NEA2 cross-end tests hold.
const KGNB: [u8; 32] = [0x42; 32];

/// The real UE and gNB RRC tasks of the NEA2 cross-end tests, and their channels.
struct Nea2Pair {
    ue: UeRrcTask,
    gnb: GnbRrcTask,
    ue_nas_rx: mpsc::Receiver<UeTaskMessage<NasMessage>>,
    ue_rls_rx: mpsc::Receiver<UeTaskMessage<UeRlsMessage>>,
    gnb_ngap_rx: mpsc::Receiver<GnbTaskMessage<NgapMessage>>,
    gnb_rls_rx: mpsc::Receiver<GnbTaskMessage<GnbRlsMessage>>,
}

/// AS security with NEA2/NIA2 through the real handlers of both ends, then an RLF
/// and a re-establishment, up to the UE receiving the gNB's RRCReestablishment.
/// The gNB's AS context holds NCC `gnb_ncc`.
async fn nea2_up_to_the_reestablishment(gnb_ncc: u8) -> Nea2Pair {
    use nextgsim_crypto::kdf::{derive_rrc_up_key, AlgorithmTypeDistinguisher};
    use nextgsim_gnb::rrc::ue_context::ReestablishmentSecurity;
    use nextgsim_rrc::procedures::rrc_reestablishment::{phys_cell_id_from_nci, SIMULATED_C_RNTI};
    use nextgsim_ue::tasks::RlfCause;

    let (ue_base, _ue_app_rx, mut ue_nas_rx, _ue_rrc_rx, mut ue_rls_rx) = UeTaskBase::new(
        UeConfig {
            as_security_enabled: true,
            ..UeConfig::default()
        },
        32,
    );
    let mut ue = UeRrcTask::new(ue_base);
    let (gnb_base, _a, mut gnb_ngap_rx, _r, _g, mut gnb_rls_rx, _s) = GnbTaskBase::new(
        GnbConfig {
            as_security_enabled: true,
            ..gnb_config()
        },
        32,
    );
    let mut gnb = GnbRrcTask::new(gnb_base);

    // Camp with SIB1, so both ends derive the PCI from the broadcast NCI.
    gnb.handle_radio_power_on();
    gnb.broadcast_system_information(true).await;
    ue.handle_signal_changed(CELL_ID, -60).await;
    while let Ok(msg) = gnb_rls_rx.try_recv() {
        if let GnbTaskMessage::Message(GnbRlsMessage::BroadcastRrc {
            rrc_channel, data, ..
        }) = msg
        {
            ue.handle_downlink_rrc(CELL_ID, rrc_channel, data).await;
        }
    }
    ue.perform_cycle().await;

    // Connection establishment.
    ue.handle_uplink_nas_delivery(1, OctetString::from_slice(&registration_nas()))
        .await;
    let (_, setup_req) = next_ue_uplink_rrc(&mut ue_rls_rx);
    gnb.handle_uplink_rrc(UE_ID, RrcChannel::UlCcch, setup_req)
        .await;
    let (_, ch, rrc_setup) = next_gnb_downlink_rrc(&mut gnb_rls_rx);
    ue.handle_downlink_rrc(CELL_ID, ch, rrc_setup).await;
    let (_, setup_complete) = next_ue_uplink_rrc(&mut ue_rls_rx);
    gnb.handle_uplink_rrc(UE_ID, RrcChannel::UlDcch, setup_complete)
        .await;
    assert!(try_take_initial_nas(&mut gnb_ngap_rx).is_some());

    // Both ends hold the same KgNB, as the NAS layer and the NGAP task give it.
    ue.set_pending_kgnb(KGNB);
    gnb.handle_as_security_for_reestablishment(
        UE_ID,
        ReestablishmentSecurity {
            k_rrc_int: derive_rrc_up_key(&KGNB, AlgorithmTypeDistinguisher::RrcInt, 2),
            k_rrc_enc: derive_rrc_up_key(&KGNB, AlgorithmTypeDistinguisher::RrcEnc, 2),
            integrity_alg_id: 2,
            ciphering_alg_id: 2,
            c_rnti: SIMULATED_C_RNTI,
            phys_cell_id: phys_cell_id_from_nci(gnb_config().nci),
            next_hop_chaining_count: gnb_ncc,
            kgnb: KGNB,
        },
    );

    // SecurityModeCommand NEA2/NIA2 through the gNB's send path.
    let smc = encode_security_mode_command(&SecurityModeCommandParams {
        rrc_transaction_id: 0,
        security_algorithms: SecurityAlgorithms {
            ciphering_algorithm: CipheringAlgorithmType::Nea2,
            integrity_algorithm: Some(IntegrityAlgorithmType::Nia2),
        },
    })
    .expect("encode SecurityModeCommand");
    gnb.handle_security_mode_command(UE_ID, OctetString::from_slice(&smc))
        .await;
    let (_, smc_ch, protected_smc) = next_gnb_downlink_rrc(&mut gnb_rls_rx);

    // A downlink NAS PDU right behind the command, before the UE has answered --
    // the Registration Accept the AMF sends after the SMC. It must already be
    // ciphered and protected, at DL COUNT 1 (TS 33.501 §6.7.4: "RRC downlink
    // ciphering (encryption) at the gNB/ng-eNB shall start after sending the AS
    // security mode command message").
    let accept = vec![0x7E, 0x02, 0x11, 0x22, 0x33, 0x44];
    gnb.handle_nas_delivery(UE_ID, OctetString::from_slice(&accept))
        .await;
    let (_, ch, dl) = next_gnb_downlink_rrc(&mut gnb_rls_rx);
    assert!(
        !dl.data()
            .windows(accept.len())
            .any(|w| w == accept.as_slice()),
        "the DLInformationTransfer after the SMC must be ciphered"
    );

    ue.handle_downlink_rrc(CELL_ID, smc_ch, protected_smc).await;
    let (_, smc_complete) = next_ue_uplink_rrc(&mut ue_rls_rx);
    while ue_nas_rx.try_recv().is_ok() {}
    ue.handle_downlink_rrc(CELL_ID, ch, dl).await;
    assert!(
        ue_received_nas_delivery(&mut ue_nas_rx),
        "the UE must verify and decipher the first protected downlink PDU"
    );
    gnb.handle_uplink_rrc(UE_ID, RrcChannel::UlDcch, smc_complete)
        .await;

    // Radio link failure, then re-establishment on the same cell.
    ue.handle_radio_link_failure(RlfCause::RadioLinkQualityLost)
        .await;
    // T311: cell selection finds the same cell, and the request goes to it.
    ue.perform_cycle().await;
    let (ch, reest_req) = next_ue_uplink_rrc(&mut ue_rls_rx);
    assert_eq!(ch, RrcChannel::UlCcch);
    gnb.handle_uplink_rrc(UE_ID, ch, reest_req).await;
    let (_, ch, reestablishment) = next_gnb_downlink_rrc(&mut gnb_rls_rx);
    ue.handle_downlink_rrc(CELL_ID, ch, reestablishment).await;
    Nea2Pair {
        ue,
        gnb,
        ue_nas_rx,
        ue_rls_rx,
        gnb_ngap_rx,
        gnb_rls_rx,
    }
}

/// #285 review (BLOCKER): AS security with NEA2/NIA2 through the REAL handlers of
/// both ends, then an RRC re-establishment under it.
///
/// - The gNB's SecurityModeCommand goes through its own send path, which starts
///   downlink protection (TS 33.501 §6.7.4). The UE's SecurityModeComplete is
///   integrity protected at UL COUNT 0, and the next downlink PDU is ciphered and
///   protected at DL COUNT 1.
/// - Re-establishment: the RRCReestablishment is integrity protected, not ciphered,
///   at DL COUNT 0 with K_RRCint from `KgNB*` (TS 38.331 §5.3.7.4-5, TS 33.501
///   §6.11). The UE must verify it with the same re-derived key, and both ends then
///   cipher again, the RRCReestablishmentComplete first.
///
/// Each end used to be tested on its own, with NEA0 on the UE side, where "ciphered"
/// and "unciphered" are the same bytes. So the UE's full-context check of an
/// integrity-only PDU, and the missing re-key, went unseen.
#[tokio::test]
async fn nea2_security_and_re_establishment_agree_end_to_end() {
    use nextgsim_crypto::kdf::{derive_kgnb_star, derive_rrc_up_key, AlgorithmTypeDistinguisher};
    use nextgsim_rrc::procedures::rrc_reestablishment::phys_cell_id_from_nci;

    let Nea2Pair {
        mut ue,
        mut gnb,
        mut ue_nas_rx,
        mut ue_rls_rx,
        mut gnb_ngap_rx,
        mut gnb_rls_rx,
    } = nea2_up_to_the_reestablishment(0).await;
    let accept = vec![0x7E, 0x02, 0x11, 0x22, 0x33, 0x44];
    let (ch, reest_complete) = next_ue_uplink_rrc(&mut ue_rls_rx);
    assert_eq!(
        ch,
        RrcChannel::UlDcch,
        "the UE must accept the integrity-only RRCReestablishment and complete"
    );
    gnb.handle_uplink_rrc(UE_ID, ch, reest_complete).await;

    // TS 33.501 Annex A.11, computed here independently of both ends: KgNB* from
    // the KgNB, this cell's PCI and ARFCN-DL, then K_RRCint for NIA2 (A.8). Both
    // ends dropping the KgNB* step would still agree with each other; they would
    // not agree with this (#348 re-review).
    let kgnb_star = derive_kgnb_star(
        &KGNB,
        phys_cell_id_from_nci(gnb_config().nci),
        gnb_config().dl_arfcn,
    );
    assert_eq!(gnb_config().dl_arfcn, UeConfig::default().dl_arfcn);
    assert_eq!(
        ue.as_security().expect("the re-keyed context").k_rrc_int,
        derive_rrc_up_key(&kgnb_star, AlgorithmTypeDistinguisher::RrcInt, 2),
        "the UE's K_RRCint after re-establishment is the one from KgNB*"
    );

    // After re-establishment both directions run on the re-derived keys: a
    // downlink NAS PDU at DL COUNT 1 reaches the UE NAS, and an uplink one at UL
    // COUNT 1 reaches the NGAP task.
    while ue_nas_rx.try_recv().is_ok() {}
    gnb.handle_nas_delivery(UE_ID, OctetString::from_slice(&accept))
        .await;
    let (_, ch, dl) = next_gnb_downlink_rrc(&mut gnb_rls_rx);
    ue.handle_downlink_rrc(CELL_ID, ch, dl).await;
    assert!(
        ue_received_nas_delivery(&mut ue_nas_rx),
        "downlink after re-establishment must verify with the re-derived keys"
    );
    let uplink = vec![0x7E, 0x00, 0x67, 0x01];
    while gnb_ngap_rx.try_recv().is_ok() {}
    ue.handle_uplink_nas_delivery(2, OctetString::from_slice(&uplink))
        .await;
    let (ch, ul) = next_ue_uplink_rrc(&mut ue_rls_rx);
    gnb.handle_uplink_rrc(UE_ID, ch, ul).await;
    let mut delivered = None;
    while let Ok(msg) = gnb_ngap_rx.try_recv() {
        if let GnbTaskMessage::Message(NgapMessage::UplinkNasDelivery { pdu, .. }) = msg {
            delivered = Some(pdu.data().to_vec());
        }
    }
    assert_eq!(
        delivered,
        Some(uplink),
        "uplink after re-establishment must verify with the re-derived keys"
    );
}

/// #348 re-review, TS 33.501 §6.11: an RRCReestablishment whose NCC differs from
/// the UE's needs the NH chain (Annex A.10/A.11), which this UE does not keep
/// yet (#325). The UE refuses it and goes to RRC_IDLE, rather than complete on a
/// key the network does not hold.
#[tokio::test]
async fn a_re_establishment_with_another_ncc_is_refused() {
    let Nea2Pair {
        ue, mut ue_rls_rx, ..
    } = nea2_up_to_the_reestablishment(1).await;
    let mut completed = false;
    while let Ok(msg) = ue_rls_rx.try_recv() {
        if let UeTaskMessage::Message(UeRlsMessage::RrcPduDelivery { channel, .. }) = msg {
            completed |= channel == RrcChannel::UlDcch;
        }
    }
    assert!(!completed, "no RRCReestablishmentComplete");
    assert!(
        ue.serving_cell_id().is_none() || ue.as_security().is_none(),
        "the UE went to RRC_IDLE"
    );
}

/// #264 (TS 38.331 §5.3.10.3, §5.3.11; TS 38.304 §5.2): after a radio link
/// failure with no AS security to re-establish with, the real UE RRC goes to
/// RRC_IDLE and tells NAS, selects the cell that is there now (the restarted gNB
/// shows up as a new cell), hands it to NAS as an active-cell change, and the
/// registration NAS that follows reaches the real gNB as a new Initial UE
/// Message. Before #276 the UE kept camping on the lost cell and nothing
/// followed the failure.
#[tokio::test]
async fn after_a_radio_link_failure_the_ue_reselects_and_reaches_the_gnb_again() {
    let mut h = harness();
    let (setup_complete, _nas) = run_setup_handshake(&mut h).await;
    h.gnb
        .handle_uplink_rrc(UE_ID, RrcChannel::UlDcch, setup_complete)
        .await;
    assert!(try_take_initial_nas(&mut h.gnb_ngap_rx).is_some());
    while h.ue_nas_rx.try_recv().is_ok() {}

    // The serving cell disappears (RLS reports it lost), then the RLF.
    h.ue.handle_signal_changed(CELL_ID, i32::MIN).await;
    h.ue.handle_radio_link_failure(nextgsim_ue::tasks::RlfCause::SignalLostToConnectedCell)
        .await;
    let mut saw_rlf = false;
    while let Ok(msg) = h.ue_nas_rx.try_recv() {
        if let UeTaskMessage::Message(NasMessage::RadioLinkFailure) = msg {
            saw_rlf = true;
        }
    }
    assert!(saw_rlf, "NAS learns the connection is gone (CM-IDLE)");

    // The gNB comes back as a new cell; the UE selects it and tells NAS.
    const NEW_CELL: i32 = 2;
    h.ue.handle_signal_changed(NEW_CELL, -60).await;
    h.ue.perform_cycle().await;
    let mut changed = false;
    while let Ok(msg) = h.ue_nas_rx.try_recv() {
        if let UeTaskMessage::Message(NasMessage::ActiveCellChanged { .. }) = msg {
            changed = true;
        }
    }
    assert!(changed, "the reselected cell reaches NAS");
    assert_eq!(h.ue.serving_cell_id(), Some(NEW_CELL));

    // NAS's registration update: a new RRC connection to the gNB.
    let update = vec![0x7E, 0x00, 0x41, 0x03];
    h.ue.handle_uplink_nas_delivery(2, OctetString::from_slice(&update))
        .await;
    let (ch, setup_req) = next_ue_uplink_rrc(&mut h.ue_rls_rx);
    assert_eq!(ch, RrcChannel::UlCcch);
    const NEW_UE_ID: i32 = 2;
    h.gnb
        .handle_uplink_rrc(NEW_UE_ID, RrcChannel::UlCcch, setup_req)
        .await;
    let (_, ch, rrc_setup) = next_gnb_downlink_rrc(&mut h.gnb_rls_rx);
    assert_eq!(ch, RrcChannel::DlCcch);
    h.ue.handle_downlink_rrc(NEW_CELL, RrcChannel::DlCcch, rrc_setup)
        .await;
    let (_, complete) = next_ue_uplink_rrc(&mut h.ue_rls_rx);
    h.gnb
        .handle_uplink_rrc(NEW_UE_ID, RrcChannel::UlDcch, complete)
        .await;
    let (ue_id, pdu, _) =
        try_take_initial_nas(&mut h.gnb_ngap_rx).expect("a new Initial UE Message");
    assert_eq!(ue_id, NEW_UE_ID);
    assert_eq!(
        pdu.data(),
        &update[..],
        "the registration update NAS, byte for byte"
    );
}
