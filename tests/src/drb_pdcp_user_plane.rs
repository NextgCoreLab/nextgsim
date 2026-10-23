//! #33 — DRB user-plane PDCP end to end, over the real RLS transport.
//!
//! Runs the REAL gNB and UE RLS tasks in one process over loopback UDP with real
//! cell discovery, and carries user data in both directions through the PDCP
//! sublayer this issue added:
//!
//! ```text
//! gNB RlsMessage::DownlinkData  →  PDCP (SN, COUNT, discardTimer)
//!                               →  RLC  →  UDP (RLS)
//!                               →  UE RLC reassembly
//!                               →  UE PDCP (reorder, in-order delivery)
//!                               →  NasMessage::UplinkDataDelivery
//! ```
//!
//! Before #33 the RLC entity was wired straight to NAS on the UE and to GTP-U on
//! the gNB, so a DRB had no PDCP header, no SN, no COUNT, no in-order delivery and
//! no discardTimer.
//!
//! **This file only compiles under `--features drb-pdcp`**: without it there is no
//! PDCP on the path and every assertion here would be about a sublayer that is not
//! there. The default `cargo test --workspace` therefore skips it, and the
//! `drb-pdcp` CI job runs it.

#![cfg(feature = "drb-pdcp")]

use std::net::SocketAddr;
use std::time::Duration;

use nextgsim_common::config::{GnbConfig, UeConfig};
use nextgsim_common::{OctetString, Plmn};
use nextgsim_gnb::tasks::{
    GnbTaskBase, GtpMessage, RlsMessage as GnbRlsMessage, RrcMessage as GnbRrcMessage,
    TaskMessage as GnbTaskMessage,
};
use nextgsim_gnb::RlsTask as GnbRlsTask;
use nextgsim_gnb::Task as GnbTask;
use nextgsim_pdcp::{Pdcp, PdcpConfig, PdcpSnSize};
use nextgsim_ue::rls::{RlsTask as UeRlsTask, RlsTaskConfig};
use nextgsim_ue::{
    NasMessage, RlsMessage as UeRlsMessage, Task as UeTask, TaskMessage as UeTaskMessage,
    UeTaskBase,
};
use tokio::net::UdpSocket;
use tokio::sync::mpsc;

const PSI: i32 = 1;
/// The DRB identity `PSI`'s session uses. Equal to the PSI because a session's
/// DEFAULT DRB keeps the identity the PSI produced
/// (`nextgsim_gtp::qfi_drb::allocate_drbs`), which is what lets this suite describe
/// the same bearer with `sdap-dataplane` on or off (issue #44).
const DRB_ID: i32 = PSI;
const DISCOVERY_TIMEOUT: Duration = Duration::from_secs(10);
const DELIVERY_TIMEOUT: Duration = Duration::from_secs(10);

fn gnb_config() -> GnbConfig {
    GnbConfig {
        nci: 0x000000010,
        gnb_id_length: 32,
        plmn: Plmn::new(1, 1, false),
        tac: 1,
        ..Default::default()
    }
}

async fn free_loopback_addr() -> SocketAddr {
    let probe = UdpSocket::bind("127.0.0.1:0")
        .await
        .expect("probe socket for a free port");
    probe.local_addr().expect("probe local address")
}

struct Harness {
    gnb_rls_tx: nextgsim_gnb::tasks::TaskHandle<GnbRlsMessage>,
    ue_rls_tx: nextgsim_ue::TaskHandle<UeRlsMessage>,
    gnb_rrc_rx: mpsc::Receiver<GnbTaskMessage<GnbRrcMessage>>,
    gnb_gtp_rx: mpsc::Receiver<GnbTaskMessage<GtpMessage>>,
    ue_nas_rx: mpsc::Receiver<UeTaskMessage<NasMessage>>,
    ue_rrc_rx: mpsc::Receiver<UeTaskMessage<nextgsim_ue::RrcMessage>>,
}

async fn start_harness() -> Harness {
    let gnb_addr = free_loopback_addr().await;

    let (gnb_base, _gnb_app_rx, _gnb_ngap_rx, gnb_rrc_rx, gnb_gtp_rx, gnb_rls_rx, _gnb_sctp_rx) =
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
            ..Default::default()
        },
    );
    tokio::spawn(async move { ue_rls.run(ue_rls_rx).await });

    Harness {
        gnb_rls_tx,
        ue_rls_tx,
        gnb_rrc_rx,
        gnb_gtp_rx,
        ue_nas_rx,
        ue_rrc_rx,
    }
}

/// Waits for mutual discovery and makes the found cell the UE's serving cell,
/// returning the UE id the gNB assigned.
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

/// Collects `count` downlink SDUs the UE handed up to NAS.
async fn collect_at_ue(
    rx: &mut mpsc::Receiver<UeTaskMessage<NasMessage>>,
    count: usize,
) -> Vec<Vec<u8>> {
    let mut out = Vec::with_capacity(count);
    tokio::time::timeout(DELIVERY_TIMEOUT, async {
        while out.len() < count {
            match rx.recv().await {
                Some(UeTaskMessage::Message(NasMessage::UplinkDataDelivery { data, .. })) => {
                    out.push(data.data().to_vec());
                }
                Some(_) => continue,
                None => panic!("UE NAS channel closed after {} SDU(s)", out.len()),
            }
        }
    })
    .await
    .unwrap_or_else(|_| panic!("expected {count} SDU(s) at the UE, got {}", out.len()));
    out
}

/// Collects `count` downlink SDUs at the UE that satisfy `keep`, ignoring others.
async fn collect_at_ue_where(
    rx: &mut mpsc::Receiver<UeTaskMessage<NasMessage>>,
    count: usize,
    keep: impl Fn(&[u8]) -> bool,
) -> Vec<Vec<u8>> {
    let mut out = Vec::with_capacity(count);
    tokio::time::timeout(DELIVERY_TIMEOUT, async {
        while out.len() < count {
            match rx.recv().await {
                Some(UeTaskMessage::Message(NasMessage::UplinkDataDelivery { data, .. })) => {
                    if keep(data.data()) {
                        out.push(data.data().to_vec());
                    }
                }
                Some(_) => continue,
                None => panic!("UE NAS channel closed after {} SDU(s)", out.len()),
            }
        }
    })
    .await
    .unwrap_or_else(|_| panic!("expected {count} SDU(s) at the UE, got {}", out.len()));
    out
}

/// Collects `count` uplink SDUs at the gNB that satisfy `keep`, ignoring others.
async fn collect_at_gnb_where(
    rx: &mut mpsc::Receiver<GnbTaskMessage<GtpMessage>>,
    count: usize,
    keep: impl Fn(&[u8]) -> bool,
) -> Vec<Vec<u8>> {
    let mut out = Vec::with_capacity(count);
    tokio::time::timeout(DELIVERY_TIMEOUT, async {
        while out.len() < count {
            match rx.recv().await {
                Some(GnbTaskMessage::Message(GtpMessage::DataPduDelivery { pdu, .. })) => {
                    if keep(pdu.data()) {
                        out.push(pdu.data().to_vec());
                    }
                }
                Some(_) => continue,
                None => panic!("gNB GTP channel closed after {} SDU(s)", out.len()),
            }
        }
    })
    .await
    .unwrap_or_else(|_| panic!("expected {count} SDU(s) at the gNB, got {}", out.len()));
    out
}

/// Collects `count` uplink SDUs the gNB handed up to GTP.
async fn collect_at_gnb(
    rx: &mut mpsc::Receiver<GnbTaskMessage<GtpMessage>>,
    count: usize,
) -> Vec<Vec<u8>> {
    let mut out = Vec::with_capacity(count);
    tokio::time::timeout(DELIVERY_TIMEOUT, async {
        while out.len() < count {
            match rx.recv().await {
                Some(GnbTaskMessage::Message(GtpMessage::DataPduDelivery { pdu, .. })) => {
                    out.push(pdu.data().to_vec());
                }
                Some(_) => continue,
                None => panic!("gNB GTP channel closed after {} SDU(s)", out.len()),
            }
        }
    })
    .await
    .unwrap_or_else(|_| panic!("expected {count} SDU(s) at the gNB, got {}", out.len()));
    out
}

/// CRITERION 6: DRB traffic in both directions arrives intact and in order with
/// the PDCP sublayer interposed.
#[tokio::test]
async fn drb_traffic_survives_the_pdcp_sublayer_in_both_directions() {
    let mut h = start_harness().await;
    let ue_id = complete_discovery(&mut h).await;

    // ---- downlink: gNB -> UE -------------------------------------------
    let downlink: Vec<Vec<u8>> = (0..5u8).map(|i| vec![0xD0 + i; 60]).collect();
    for payload in &downlink {
        h.gnb_rls_tx
            .send(GnbRlsMessage::DownlinkData {
                ue_id,
                psi: PSI,
                drb_id: DRB_ID,
                pdu: OctetString::from_slice(payload),
            })
            .await
            .expect("gNB RLS task alive");
    }
    let at_ue = collect_at_ue(&mut h.ue_nas_rx, downlink.len()).await;
    assert_eq!(
        at_ue, downlink,
        "downlink payload must reach NAS intact and in order through PDCP"
    );

    // ---- uplink: UE -> gNB ---------------------------------------------
    let uplink: Vec<Vec<u8>> = (0..5u8).map(|i| vec![0xA0 + i; 60]).collect();
    for payload in &uplink {
        h.ue_rls_tx
            .send(UeRlsMessage::DataPduDelivery {
                psi: PSI,
                pdu: OctetString::from_slice(payload),
            })
            .await
            .expect("UE RLS task alive");
    }
    let at_gnb = collect_at_gnb(&mut h.gnb_gtp_rx, uplink.len()).await;
    assert_eq!(
        at_gnb, uplink,
        "uplink payload must reach GTP intact and in order through PDCP"
    );
}

/// The PDCP header is really on the wire: what the gNB sends is a PDCP PDU whose
/// SN the UE's entity can read, and it is *longer* than the payload by the header.
///
/// Asserted by decoding with an independent PDCP entity rather than by trusting the
/// two tasks to agree — they share the crate, so they would agree on a wrong format.
#[tokio::test]
async fn the_downlink_carries_a_decodable_pdcp_header() {
    let mut h = start_harness().await;
    let ue_id = complete_discovery(&mut h).await;

    let payload = vec![0x5A; 40];
    h.gnb_rls_tx
        .send(GnbRlsMessage::DownlinkData {
            ue_id,
            psi: PSI,
            drb_id: DRB_ID,
            pdu: OctetString::from_slice(&payload),
        })
        .await
        .expect("gNB RLS task alive");

    let delivered = collect_at_ue(&mut h.ue_nas_rx, 1).await;
    assert_eq!(
        delivered[0], payload,
        "the header must be stripped, not kept"
    );

    // And an independent entity, fed the same payload, produces a PDU two octets
    // longer with SN 0 in it — which is what the UE had to parse to get here.
    let mut reference = Pdcp::new(PdcpConfig::default());
    reference.submit_sdu(&payload, 0);
    let pdu = reference
        .take_transmittable(0)
        .pop()
        .expect("one PDU per SDU");
    assert_eq!(
        pdu.len(),
        payload.len() + PdcpSnSize::Sn12.header_len(),
        "a 12-bit-SN DRB adds a 2-octet header"
    );
    assert_eq!(pdu[0] & 0x80, 0x80, "D/C must mark a data PDU");
    assert_eq!(&pdu[2..], &payload[..]);
}

/// Sustained traffic keeps its order across many SDUs, which is what catches an
/// SN or COUNT that drifts rather than one that is simply wrong from the start.
#[tokio::test]
async fn a_long_downlink_burst_stays_in_order() {
    let mut h = start_harness().await;
    let ue_id = complete_discovery(&mut h).await;

    // 40 SDUs, each tagged with its index so a reorder is visible rather than
    // merely a length mismatch.
    let burst: Vec<Vec<u8>> = (0..40u8).map(|i| vec![i; 32]).collect();
    for payload in &burst {
        h.gnb_rls_tx
            .send(GnbRlsMessage::DownlinkData {
                ue_id,
                psi: PSI,
                drb_id: DRB_ID,
                pdu: OctetString::from_slice(payload),
            })
            .await
            .expect("gNB RLS task alive");
    }

    let delivered = collect_at_ue(&mut h.ue_nas_rx, burst.len()).await;
    assert_eq!(delivered, burst, "40 SDUs must arrive in order and intact");
}

/// #315: a new RRC connection starts every DRB at COUNT 0 on BOTH ends.
///
/// More than half the 12-bit SN space (2048) is sent first, so the UE's entity
/// is far from COUNT 0. Then the gNB side starts fresh -- what a restarted gNB,
/// or the same gNB after an RRC release and re-setup, does -- and the UE
/// releases its DRB entities as its RRC does on the new RRCSetup. The traffic
/// that follows must arrive. Before #315 no entity was ever released or
/// re-established: the UE kept RX_DELIV at 2100, read the new COUNT 0..9 as
/// older than that, and discarded every PDU.
#[tokio::test]
async fn drb_traffic_survives_a_new_rrc_connection_after_2048_pdus() {
    let mut h = start_harness().await;
    let ue_id = complete_discovery(&mut h).await;

    const FIRST: usize = 2100;
    for i in 0..FIRST {
        h.gnb_rls_tx
            .send(GnbRlsMessage::DownlinkData {
                ue_id,
                psi: PSI,
                drb_id: DRB_ID,
                pdu: OctetString::from_slice(&(i as u32).to_be_bytes()),
            })
            .await
            .expect("gNB RLS task alive");
        // Paced so loopback UDP does not drop under a full-speed burst.
        if i % 50 == 49 {
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        h.ue_rls_tx
            .send(UeRlsMessage::DataPduDelivery {
                psi: PSI,
                pdu: OctetString::from_slice(&(i as u32).to_be_bytes()),
            })
            .await
            .expect("UE RLS task alive");
        // Drain as we go, so neither 64-deep channel fills.
        while let Ok(_msg) = h.ue_nas_rx.try_recv() {}
        while let Ok(_msg) = h.gnb_gtp_rx.try_recv() {}
    }
    // Long enough for t-Reordering to flush anything the old entities still held.
    tokio::time::sleep(Duration::from_millis(1500)).await;
    while let Ok(_msg) = h.ue_nas_rx.try_recv() {}
    while let Ok(_msg) = h.gnb_gtp_rx.try_recv() {}

    // The new connection: the gNB's entities go (a restart, or AN release then
    // RRCSetup), and the UE's go on its RRCSetup.
    h.gnb_rls_tx
        .send(GnbRlsMessage::ReleaseUeDrbs { ue_id })
        .await
        .expect("gNB RLS task alive");
    h.ue_rls_tx
        .send(UeRlsMessage::ReleaseDrbs)
        .await
        .expect("UE RLS task alive");

    let after: Vec<Vec<u8>> = (0..10u8).map(|i| vec![0xE0 + i; 24]).collect();
    for payload in &after {
        h.gnb_rls_tx
            .send(GnbRlsMessage::DownlinkData {
                ue_id,
                psi: PSI,
                drb_id: DRB_ID,
                pdu: OctetString::from_slice(payload),
            })
            .await
            .expect("gNB RLS task alive");
    }
    // The new connection's SDUs are 24 octets; a first-phase one (4 octets) still
    // on the loopback when the connection was replaced is not part of it.
    let delivered: Vec<Vec<u8>> =
        collect_at_ue_where(&mut h.ue_nas_rx, after.len(), |sdu| sdu.len() != 4).await;
    assert_eq!(
        delivered, after,
        "traffic on the new connection arrives in order"
    );

    // And the uplink: the UE's TX entity had reached COUNT 2100 too.
    for payload in &after {
        h.ue_rls_tx
            .send(UeRlsMessage::DataPduDelivery {
                psi: PSI,
                pdu: OctetString::from_slice(payload),
            })
            .await
            .expect("UE RLS task alive");
    }
    let at_gnb = collect_at_gnb_where(&mut h.gnb_gtp_rx, after.len(), |sdu| sdu.len() != 4).await;
    assert_eq!(at_gnb, after, "and so does the uplink");
}
