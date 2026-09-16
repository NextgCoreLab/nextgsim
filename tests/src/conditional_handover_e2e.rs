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
    let (ue_base, _app, _nas, _rrc, ue_rls_rx) = UeTaskBase::new(ue_config(), 32);
    let (gnb_base, _gapp, _gngap, _grrc, _ggtp, gnb_rls_rx, _gsctp) =
        GnbTaskBase::new(gnb_config(), 32);
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
