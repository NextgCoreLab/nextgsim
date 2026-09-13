//! #34 — two-PDU-session user plane over the real RLS transport.
//!
//! Runs the REAL gNB and UE RLS tasks in one process over loopback UDP, with
//! real cell discovery, and carries user data on two PDU sessions at once:
//!
//! ```text
//! gNB RlsMessage::DownlinkData{psi 1}  ──┐
//! gNB RlsMessage::DownlinkData{psi 5}  ──┤ per-(UE, bearer) RLC entities
//!                                        ▼
//!                              UDP (RLS PduTransmission)
//!                                        ▼
//!                        UE per-PSI RLC entities → reassembly
//!                                        ▼
//!                     NasMessage::UplinkDataDelivery{psi, data}
//! ```
//!
//! Before this landed the gNB kept ONE RLC entity per UE while the UE kept one
//! per PSI, so both sessions shared a single sequence-number space and a single
//! reassembly buffer on the gNB side — TS 38.322 §4.2.1 gives each radio bearer
//! its own entity precisely so that cannot happen. The per-bearer SN assertion
//! lives in `nextgsim-gnb`'s unit tests (`two_pdu_sessions_get_independent_
//! sequence_number_spaces`); this test is the end-to-end half: both sessions
//! carry their own payload, segmented and unsegmented, with no cross-delivery.
//!
//! The Docker/E2E job (`workflow_dispatch`) covers the same path against
//! nextgcore; this one runs on every push.

use std::net::SocketAddr;
use std::time::Duration;

use nextgsim_common::config::{GnbConfig, UeConfig};
use nextgsim_common::{OctetString, Plmn};
use nextgsim_gnb::tasks::{
    GnbTaskBase, RlsMessage as GnbRlsMessage, RrcMessage as GnbRrcMessage,
    TaskMessage as GnbTaskMessage,
};
use nextgsim_gnb::RlsTask as GnbRlsTask;
use nextgsim_gnb::Task as GnbTask;
use nextgsim_ue::rls::{RlsTask as UeRlsTask, RlsTaskConfig};
use nextgsim_ue::{
    NasMessage, RlsMessage as UeRlsMessage, Task as UeTask, TaskMessage as UeTaskMessage,
    UeTaskBase,
};
use tokio::net::UdpSocket;
use tokio::sync::mpsc;

const PSI_A: i32 = 1;
const PSI_B: i32 = 5;
const DISCOVERY_TIMEOUT: Duration = Duration::from_secs(10);

fn gnb_config() -> GnbConfig {
    GnbConfig {
        nci: 0x000000010,
        gnb_id_length: 32,
        plmn: Plmn::new(1, 1, false),
        tac: 1,
        ..Default::default()
    }
}

/// A loopback address with a port the OS has just confirmed free.
///
/// The gNB RLS task binds a fixed address and the UE has to be told that address
/// up front, so the port cannot be left to the OS at bind time.
async fn free_loopback_addr() -> SocketAddr {
    let probe = UdpSocket::bind("127.0.0.1:0")
        .await
        .expect("probe socket for a free port");
    probe.local_addr().expect("probe local address")
}

/// The running two-sided harness: both real RLS tasks, plus the channels their
/// handlers emit on.
struct Harness {
    gnb_rls_tx: nextgsim_gnb::tasks::TaskHandle<GnbRlsMessage>,
    ue_rls_tx: nextgsim_ue::TaskHandle<UeRlsMessage>,
    gnb_rrc_rx: mpsc::Receiver<GnbTaskMessage<GnbRrcMessage>>,
    ue_nas_rx: mpsc::Receiver<UeTaskMessage<NasMessage>>,
    ue_rrc_rx: mpsc::Receiver<UeTaskMessage<nextgsim_ue::RrcMessage>>,
}

async fn start_harness() -> Harness {
    let gnb_addr = free_loopback_addr().await;

    let (gnb_base, _gnb_app_rx, _gnb_ngap_rx, gnb_rrc_rx, _gnb_gtp_rx, gnb_rls_rx, _gnb_sctp_rx) =
        GnbTaskBase::new(gnb_config(), 64);
    let gnb_rls_tx = gnb_base.rls_tx.clone();
    let mut gnb_rls = GnbRlsTask::with_bind_address(gnb_base, gnb_addr);
    tokio::spawn(async move { gnb_rls.run(gnb_rls_rx).await });

    let (ue_base, _ue_app_rx, ue_nas_rx, ue_rrc_rx, ue_rls_rx) =
        UeTaskBase::new(UeConfig::default(), 64);
    let ue_rls_tx = ue_base.rls_tx.clone();
    let mut ue_rls = UeRlsTask::new(
        ue_base,
        RlsTaskConfig {
            gnb_search_list: vec![gnb_addr],
            bind_address: Some("127.0.0.1:0".parse().expect("UE bind address")),
            heartbeat_interval: Duration::from_millis(100),
            heartbeat_threshold: Duration::from_millis(2000),
        },
    );
    tokio::spawn(async move { ue_rls.run(ue_rls_rx).await });

    Harness {
        gnb_rls_tx,
        ue_rls_tx,
        gnb_rrc_rx,
        ue_nas_rx,
        ue_rrc_rx,
    }
}

/// Waits until the gNB has discovered the UE and the UE has discovered the cell,
/// then makes that cell the UE's serving cell — the step the UE RRC task
/// normally performs, and the one that lets the UE accept user-plane PDUs.
///
/// Returns the UE id the gNB assigned; it is the cell tracker's own numbering,
/// so the test takes it from the gNB rather than assuming a value.
async fn complete_discovery(h: &mut Harness) -> i32 {
    let gnb_saw_ue = tokio::time::timeout(DISCOVERY_TIMEOUT, async {
        while let Some(msg) = h.gnb_rrc_rx.recv().await {
            if let GnbTaskMessage::Message(GnbRrcMessage::SignalDetected { ue_id }) = msg {
                return ue_id;
            }
        }
        panic!("gNB RRC channel closed before the UE was detected");
    })
    .await
    .expect("the gNB must discover the UE");

    let cell_id = tokio::time::timeout(DISCOVERY_TIMEOUT, async {
        while let Some(msg) = h.ue_rrc_rx.recv().await {
            if let UeTaskMessage::Message(nextgsim_ue::RrcMessage::SignalChanged {
                cell_id, ..
            }) = msg
            {
                return cell_id;
            }
        }
        panic!("UE RRC channel closed before a cell was found");
    })
    .await
    .expect("the UE must discover the cell");

    h.ue_rls_tx
        .send(UeRlsMessage::AssignCurrentCell { cell_id })
        .await
        .expect("UE RLS task alive");

    gnb_saw_ue
}

/// Collects `count` downlink SDUs the UE handed up to NAS, as `(psi, bytes)`.
async fn collect_delivered(
    rx: &mut mpsc::Receiver<UeTaskMessage<NasMessage>>,
    count: usize,
) -> Vec<(i32, Vec<u8>)> {
    let mut out = Vec::with_capacity(count);
    tokio::time::timeout(Duration::from_secs(10), async {
        while out.len() < count {
            match rx.recv().await {
                Some(UeTaskMessage::Message(NasMessage::UplinkDataDelivery { psi, data })) => {
                    out.push((psi, data.data().to_vec()));
                }
                Some(_) => continue,
                None => panic!("UE NAS channel closed after {} SDU(s)", out.len()),
            }
        }
    })
    .await
    .unwrap_or_else(|_| panic!("expected {count} SDU(s), got {}", out.len()));
    out
}

#[tokio::test]
async fn two_pdu_sessions_carry_their_own_traffic_end_to_end() {
    let mut h = start_harness().await;
    let ue_id = complete_discovery(&mut h).await;

    // Payloads chosen so a swap between the two sessions cannot look correct.
    let small_a = vec![0xA1u8; 32];
    let small_b = vec![0xB2u8; 48];
    // Larger than the 1500-byte grant, so this one is segmented across PDUs and
    // has to be reassembled by the receiving bearer's own entity.
    let large_a: Vec<u8> = (0..3000u32).map(|i| (i % 251) as u8).collect();

    for (psi, payload) in [
        (PSI_A, small_a.clone()),
        (PSI_B, small_b.clone()),
        (PSI_A, large_a.clone()),
    ] {
        h.gnb_rls_tx
            .send(GnbRlsMessage::DownlinkData {
                ue_id,
                psi,
                pdu: OctetString::from_slice(&payload),
            })
            .await
            .expect("gNB RLS task alive");
    }

    let delivered = collect_delivered(&mut h.ue_nas_rx, 3).await;

    let on_a: Vec<&Vec<u8>> = delivered
        .iter()
        .filter(|(psi, _)| *psi == PSI_A)
        .map(|(_, data)| data)
        .collect();
    let on_b: Vec<&Vec<u8>> = delivered
        .iter()
        .filter(|(psi, _)| *psi == PSI_B)
        .map(|(_, data)| data)
        .collect();

    assert_eq!(
        on_a,
        vec![&small_a, &large_a],
        "PSI 1 must receive exactly its own two SDUs, the segmented one intact"
    );
    assert_eq!(
        on_b,
        vec![&small_b],
        "PSI 5 must receive exactly its own SDU and nothing from PSI 1"
    );
}
