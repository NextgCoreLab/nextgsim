//! #165 — conditional handover end to end: the gNB arms candidates, and the UE
//! executes one **on its own** when a candidate cell becomes better.
//!
//! Runs the REAL gNB and UE RRC tasks in one process:
//!
//! ```text
//! gNB send_conditional_handover_configuration   (#160: candidates from the SIB3
//!   -> DL-DCCH container                          neighbour list, condEventA3)
//!   -> UE handle_conditional_reconfiguration    (#114: candidates stored)
//!   -> the candidate cell's level rises
//!   -> UE perform_cycle -> evaluate_conditional_handover
//!   -> UE executes the handover with NO further signalling from the gNB
//! ```
//!
//! # Why this is the test #160 was missing
//!
//! The A3 margin exists at both ends — the gNB's `cho_a3_offset_db` /
//! `cho_hysteresis_db` and whatever the UE's measurement configuration applies — and
//! nothing checked that they agree. Arming from one end and triggering at the other is
//! the only thing that would catch a disagreement, and it is the first ceiling #160's
//! spec recorded.
//!
//! The whole path only became reachable recently: #151 made the DL-DCCH dispatch typed,
//! #160 gave the container a production sender, and #114 built the UE runtime.
//!
//! # Since #170: the two margins AGREE, and this file asserts it
//!
//! The ceiling this file used to record — "the two A3 margins are still independent"
//! — is gone. `RRCReconfiguration.measConfig` was hardcoded `None`, so the gNB could
//! not configure the UE's A3 **reporting** trigger at all: it ran off a hard-coded
//! UE-local default while only CHO *execution* used the gNB's number. #170 sends a
//! real generated-codec `MeasConfig`, so one configured margin now governs both, and
//! `the_gnbs_configured_margin_governs_the_ues_a3_reporting` below asserts the
//! AGREEMENT rather than documenting the gap.

use nextgsim_common::config::{GnbConfig, IntraFreqNeighbourConfig, UeConfig};
use nextgsim_common::{OctetString, Plmn};
use nextgsim_gnb::tasks::{
    GnbTaskBase, RlsMessage as GnbRlsMessage, TaskMessage as GnbTaskMessage,
};
use nextgsim_gnb::RrcTask as GnbRrcTask;
use nextgsim_rls::RrcChannel;
use nextgsim_ue::tasks::{RlsMessage as UeRlsMessage, TaskMessage as UeTaskMessage, UeTaskBase};
use nextgsim_ue::RrcTask as UeRrcTask;
use tokio::sync::mpsc;

/// The UE's serving cell.
const SERVING_CELL: i32 = 1;
/// The neighbour the gNB nominates as a CHO candidate. The RLS cell id doubles as the
/// PCI throughout this tree, so one number is both.
const CANDIDATE_CELL: i32 = 2;
const UE_ID: i32 = 1;

/// A gNB with conditional handover armed and one intra-frequency neighbour.
///
/// The neighbour list is the SIB3 one, which is where #160 takes candidates from: a
/// separate CHO list would let the two disagree about which neighbours exist.
fn gnb_config() -> GnbConfig {
    let mut config = GnbConfig {
        nci: u64::try_from(SERVING_CELL).expect("positive"),
        plmn: Plmn::new(1, 1, false),
        ..GnbConfig::default()
    };
    config.conditional_handover = true;
    config.reselection.intra_freq_neighbours = vec![IntraFreqNeighbourConfig {
        phys_cell_id: u16::try_from(CANDIDATE_CELL).expect("in range"),
        q_offset_db: 0,
    }];
    config
}

/// A UE that acts on stored candidates. Off by default (#114), so the test says so.
fn ue_config() -> UeConfig {
    UeConfig {
        conditional_handover: true,
        ..UeConfig::default()
    }
}

struct Harness {
    ue: UeRrcTask,
    gnb: GnbRrcTask,
    ue_rls_rx: mpsc::Receiver<UeTaskMessage<UeRlsMessage>>,
    gnb_rls_rx: mpsc::Receiver<GnbTaskMessage<GnbRlsMessage>>,
}

fn harness() -> Harness {
    harness_with(gnb_config())
}

/// A harness on a specific gNB configuration, so a test can use a margin that is NOT
/// the default — which is what makes an agreement assertion mean anything (#170).
fn harness_with(gnb: GnbConfig) -> Harness {
    let (ue_base, _app, _nas, _rrc, ue_rls_rx) = UeTaskBase::new(ue_config(), 32);
    let (gnb_base, _gapp, _gngap, _grrc, _ggtp, gnb_rls_rx, _gsctp) = GnbTaskBase::new(gnb, 32);
    Harness {
        ue: UeRrcTask::new(ue_base),
        gnb: GnbRrcTask::new(gnb_base),
        ue_rls_rx,
        gnb_rls_rx,
    }
}

fn next_ue_uplink_rrc(
    rx: &mut mpsc::Receiver<UeTaskMessage<UeRlsMessage>>,
) -> Option<(RrcChannel, OctetString)> {
    while let Ok(msg) = rx.try_recv() {
        if let UeTaskMessage::Message(UeRlsMessage::RrcPduDelivery { channel, pdu, .. }) = msg {
            return Some((channel, pdu));
        }
    }
    None
}

fn next_gnb_downlink_rrc(
    rx: &mut mpsc::Receiver<GnbTaskMessage<GnbRlsMessage>>,
) -> Option<(RrcChannel, OctetString)> {
    while let Ok(msg) = rx.try_recv() {
        if let GnbTaskMessage::Message(GnbRlsMessage::DownlinkRrc {
            rrc_channel, data, ..
        }) = msg
        {
            return Some((rrc_channel, data));
        }
    }
    None
}

/// Camp, establish, and arm the UE with the gNB's CHO candidates. Returns with the UE
/// holding one candidate for [`CANDIDATE_CELL`].
async fn arm_conditional_handover(h: &mut Harness) {
    h.gnb.handle_radio_power_on();

    // The UE camps on the serving cell through the real detection + selection path.
    h.ue.handle_signal_changed(SERVING_CELL, -90).await;
    h.ue.perform_cycle().await;

    h.ue.handle_uplink_nas_delivery(1, OctetString::from_slice(&[0x7E, 0x00, 0x41]))
        .await;
    let (channel, setup_request) =
        next_ue_uplink_rrc(&mut h.ue_rls_rx).expect("the UE sends an RRCSetupRequest");
    assert_eq!(channel, RrcChannel::UlCcch);

    h.gnb
        .handle_uplink_rrc(UE_ID, RrcChannel::UlCcch, setup_request)
        .await;
    let (channel, rrc_setup) =
        next_gnb_downlink_rrc(&mut h.gnb_rls_rx).expect("the gNB answers with an RRCSetup");
    assert_eq!(channel, RrcChannel::DlCcch);
    h.ue.handle_downlink_rrc(SERVING_CELL, RrcChannel::DlCcch, rrc_setup)
        .await;

    let (_, setup_complete) =
        next_ue_uplink_rrc(&mut h.ue_rls_rx).expect("the UE answers with an RRCSetupComplete");
    h.gnb
        .handle_uplink_rrc(UE_ID, RrcChannel::UlDcch, setup_complete)
        .await;
    // Drain the handshake traffic so the container is the next PDU either way.
    while next_gnb_downlink_rrc(&mut h.gnb_rls_rx).is_some() {}

    // The gNB arms the candidates (#160).
    h.gnb.send_conditional_handover_configuration(UE_ID).await;
    let (channel, container) = next_gnb_downlink_rrc(&mut h.gnb_rls_rx)
        .expect("an armed gNB must send a conditional reconfiguration");
    assert_eq!(channel, RrcChannel::DlDcch);

    h.ue.handle_downlink_rrc(SERVING_CELL, RrcChannel::DlDcch, container)
        .await;
    assert_eq!(
        h.ue.cho_candidate_count(),
        1,
        "the UE must store the candidate the gNB armed; without this the trigger below \
         would be asserting about an empty candidate set"
    );
    // Drain the UE's acknowledgement so a later uplink message is the handover's.
    while next_ue_uplink_rrc(&mut h.ue_rls_rx).is_some() {}
}

/// The candidate becomes better than the serving cell, and the UE executes — with no
/// further signalling from the gNB, which is the whole point of a conditional handover.
#[tokio::test]
async fn a_candidate_that_becomes_better_is_executed_by_the_ue_alone() {
    let mut h = harness();
    arm_conditional_handover(&mut h).await;
    let serving_before = h.ue.serving_cell_id();
    assert_eq!(serving_before, Some(SERVING_CELL), "precondition");

    // 20 dB better than the serving cell's -90: past the gNB's configured A3 offset
    // (3 dB) plus hysteresis (1 dB) by a margin no rounding could close.
    h.ue.handle_signal_changed(CANDIDATE_CELL, -70).await;
    h.ue.perform_cycle().await;

    assert_eq!(
        h.ue.cho_candidate_count(),
        0,
        "executing a reconfigurationWithSync releases every stored candidate \
         (TS 38.331 §5.3.5.3), so an empty set is what execution looks like"
    );
    assert_eq!(
        h.ue.serving_cell_id(),
        Some(CANDIDATE_CELL),
        "and the UE must now be on the candidate cell"
    );
}

/// #170: the gNB's ONE configured A3 margin governs the UE's **reporting** trigger too,
/// not just CHO execution — asserted as AGREEMENT, which is what replaced #165's
/// recorded ceiling.
///
/// The margin is deliberately not the default (9 dB / 3.5 dB): a UE that ignored the
/// wire and kept its own 3 dB / 1 dB default would pass a default-valued version of
/// this test, which is exactly how the gap went unnoticed.
///
/// Both ends of the agreement are asserted from the SAME `GnbConfig`:
/// * the reporting margin the UE ends up evaluating, via `a3_margin`;
/// * the execution margin the CHO container carried, via the fact that the candidate
///   the gNB armed executes at a level past that margin and not before it.
#[tokio::test]
async fn the_gnbs_configured_margin_governs_the_ues_a3_reporting() {
    use nextgsim_gnb::rrc::meas::a3_meas_config_params;
    use nextgsim_rrc::procedures::rrc_reconfiguration::{
        build_drb_reconfiguration_params, encode_rrc_reconfiguration, DrbIntegrityProtection,
    };

    let mut config = gnb_config();
    config.cho_a3_offset_db = 9.0;
    config.cho_hysteresis_db = 3.5;

    let mut h = harness_with(config.clone());
    arm_conditional_handover(&mut h).await;

    assert_eq!(
        h.ue.a3_margin(1),
        Some((3, 2)),
        "precondition: before the measConfig the UE is on its OWN default (3 dB / 1 dB \
         = 2 half-dB), which is NOT the gNB's 9 dB / 3.5 dB"
    );

    // The gNB's live DRB reconfiguration, built from its own configuration by the same
    // two functions `NgapTask::establish_drb` calls.
    let params = build_drb_reconfiguration_params(
        0,
        1,
        1,
        4,
        &[1],
        true,
        DrbIntegrityProtection::Disabled,
        a3_meas_config_params(&config),
    )
    .expect("the gNB's DRB reconfiguration must build");
    let pdu = encode_rrc_reconfiguration(&params).expect("and encode");
    h.ue.handle_downlink_rrc(
        SERVING_CELL,
        RrcChannel::DlDcch,
        OctetString::from_slice(&pdu),
    )
    .await;

    // AGREEMENT: the reporting margin the UE now evaluates IS the gNB's configuration,
    // derived from the same two config fields the CHO container was built from.
    assert_eq!(
        h.ue.a3_margin(1),
        Some((
            config.cho_a3_offset_db as i32,
            (config.cho_hysteresis_db * 2.0) as i32
        )),
        "the UE's A3 REPORTING margin must equal the gNB's configured margin -- the \
         independence #165's spec recorded as a ceiling"
    );

    // And the CHO execution margin is the same one: the candidate still executes only
    // once it beats the serving cell by more than 9 dB + 3.5 dB hysteresis.
    h.ue.handle_signal_changed(CANDIDATE_CELL, -80).await; // 10 dB better: inside 12.5
    h.ue.perform_cycle().await;
    assert_eq!(
        h.ue.serving_cell_id(),
        Some(SERVING_CELL),
        "10 dB is short of the configured 9 dB offset + 3.5 dB hysteresis, so the \
         candidate must NOT execute -- this is what proves the execution margin is the \
         configured one and not the 3 dB default"
    );

    h.ue.handle_signal_changed(CANDIDATE_CELL, -70).await; // 20 dB better: past 12.5
    h.ue.perform_cycle().await;
    assert_eq!(
        h.ue.serving_cell_id(),
        Some(CANDIDATE_CELL),
        "and 20 dB is past it, so it must"
    );
}

/// The negative control: a candidate that does NOT beat the serving cell by the margin
/// is left armed and unexecuted. Without it, the test above would pass for a UE that
/// executed the first candidate it was given.
#[tokio::test]
async fn a_candidate_that_stays_worse_is_left_armed() {
    let mut h = harness();
    arm_conditional_handover(&mut h).await;

    // Worse than the serving cell's -90.
    h.ue.handle_signal_changed(CANDIDATE_CELL, -100).await;
    h.ue.perform_cycle().await;

    assert_eq!(
        h.ue.cho_candidate_count(),
        1,
        "a candidate that has not triggered must stay armed"
    );
    assert_eq!(
        h.ue.serving_cell_id(),
        Some(SERVING_CELL),
        "and the UE must stay where it is"
    );
}
