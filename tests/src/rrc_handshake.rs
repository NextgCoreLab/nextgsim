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
//! Also pins the CURRENT tolerated SMC-hop behavior as the C5 red/green
//! baseline: the gNB's real ASN.1 SecurityModeCommand (DL-DCCH c1 index 4,
//! tid 0 → leading byte 0x20, low nibble 0x0) is misrouted by the UE nibble
//! dispatcher into its RRC-reconfiguration arm, which answers with the
//! bespoke ReconfigurationComplete `[0x08, smc[1]]` that the gNB then drops.
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
    encode_security_mode_command, CipheringAlgorithmType, IntegrityAlgorithmType,
    SecurityAlgorithms, SecurityModeCommandParams,
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

/// SMC-hop pin (C3 step 4 — the C5 red/green baseline, NOT desired behavior):
/// the gNB's real ASN.1 SecurityModeCommand is misrouted by the UE's DL-DCCH
/// nibble dispatcher into its RRC-reconfiguration arm; the UE answers with
/// the bespoke ReconfigurationComplete `[0x08, smc[1]]` (no
/// SecurityModeComplete, no AS security activation) and the gNB then drops
/// that response. C5 (typed DL/UL-DCCH dispatch) must flip this test.
#[tokio::test]
async fn smc_hop_current_tolerated_behavior_pinned_for_c5_baseline() {
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

    // UE: the nibble dispatcher has no SMC arm; nibble 0x0 lands in the
    // RRC-reconfiguration arm which answers with the bespoke
    // ReconfigurationComplete [0x08, bytes[1]].
    h.ue.handle_downlink_rrc(CELL_ID, RrcChannel::DlDcch, OctetString::from_slice(&smc))
        .await;
    let (ch, response) = next_ue_uplink_rrc(&mut h.ue_rls_rx);
    assert_eq!(ch, RrcChannel::UlDcch);
    assert_eq!(
        response.data(),
        &[0x08, smc[1]][..],
        "current tolerated behavior: bespoke ReconfigurationComplete, \
         NOT a SecurityModeComplete (C5 baseline)"
    );
    assert!(
        !ue_received_nas_delivery(&mut h.ue_nas_rx),
        "the SMC misroute must not fabricate NAS toward the UE NAS task"
    );

    // gNB: the bespoke response hits the UL-information-transfer arm and is
    // dropped as too short — no NGAP message results.
    h.gnb
        .handle_uplink_rrc(UE_ID, RrcChannel::UlDcch, response)
        .await;
    assert!(
        h.gnb_ngap_rx.try_recv().is_err(),
        "current tolerated behavior: the gNB ignores the UE's bespoke \
         reconfiguration-complete (C5 baseline)"
    );
}
