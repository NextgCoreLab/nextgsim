//! #452: the core tasks reach the 6G (and Rel-18) tasks through the wiring the
//! binaries run.
//!
//! Each task gets a clone of the task base, and the 6G / Rel-18 handles in it
//! are a by-value `Option`. `nr-gnb` and `nr-ue` used to clone the RRC and RLS
//! bases before installing those handles, so the core tasks held `sixg: None`
//! and nothing they sent reached a 6G task. The task manager's own base never
//! got the handles either, so `shutdown` never reached those tasks.
//!
//! These tests drive `GnbWiring` / `UeWiring`, the library wiring the binaries
//! call, end to end:
//!
//! ```text
//! UDP heartbeat ──► gNB RlsTask ──NwdafMessage::UeMeasurement──► gNB NwdafTask
//! RrcMessage::SixgInferenceRequest ──► UE RrcTask ──InferenceRequest──► UE SheClientTask
//! TaskManager::shutdown ──Shutdown──► 6G / Energy tasks
//! ```
//!
//! A probe sits in front of each target task's receiver: it records what
//! arrives and forwards it, so the real task still runs and still gets every
//! message. The assertion is that the message crossed from the real producer
//! task into the target task's inbox.
//!
//! Each gNB test binds its own loopback address, because the RLS listens on the
//! fixed port 4997 and the GTP task on 2152.

#![cfg(feature = "6g")]

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::Duration;

use bytes::Bytes;
use nextgsim_common::config::{GnbConfig, UeConfig};
use nextgsim_common::Plmn;
use nextgsim_gnb::tasks::{
    AgentMessage, FlAggregatorMessage, NwdafMessage, TaskMessage as GnbTaskMessage,
    DEFAULT_CHANNEL_CAPACITY,
};
use nextgsim_gnb::GnbWiring;
use nextgsim_rls::{codec, RlsHeartbeat, RlsMessage as RlsProtocolMessage, SimCoord};
use nextgsim_ue::tasks::{NwdafReporterMessage, SheClientMessage};
use nextgsim_ue::wiring::UeWiring;
use nextgsim_ue::{RrcMessage as UeRrcMessage, TaskMessage as UeTaskMessage};
use tokio::net::UdpSocket;
use tokio::sync::mpsc;
use tokio::time::{timeout, Instant};

/// The port the gNB RLS task listens on (`nextgsim_gnb`'s `DEFAULT_RLS_PORT`,
/// which is crate-private).
const DEFAULT_RLS_PORT: u16 = 4997;

/// How long to wait for a message to cross. Generous: a loaded shared host has
/// starved socket binds before, and a hang must fail rather than pass.
const CROSSING_TIMEOUT: Duration = Duration::from_secs(10);

/// Replaces `rx` with the receiving end of a forwarder, and returns what the
/// forwarder saw. `observe` maps each message to a record, or `None` to skip
/// it. Every message, observed or not, is passed on to the original consumer.
fn probe<M, R>(
    rx: &mut mpsc::Receiver<M>,
    observe: impl Fn(&M) -> Option<R> + Send + 'static,
) -> mpsc::UnboundedReceiver<R>
where
    M: Send + 'static,
    R: Send + 'static,
{
    let (forward_tx, forward_rx) = mpsc::channel(DEFAULT_CHANNEL_CAPACITY);
    let mut upstream = std::mem::replace(rx, forward_rx);
    let (seen_tx, seen_rx) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        while let Some(msg) = upstream.recv().await {
            if let Some(record) = observe(&msg) {
                let _ = seen_tx.send(record);
            }
            if forward_tx.send(msg).await.is_err() {
                break;
            }
        }
    });
    seen_rx
}

/// A gNB config on its own loopback address, with `nwdaf_enabled`.
fn gnb_config(octet: u8) -> GnbConfig {
    let ip = IpAddr::V4(Ipv4Addr::new(127, 0, 45, octet));
    GnbConfig {
        nci: 0x000000010,
        gnb_id_length: 32,
        plmn: Plmn::new(1, 1, false),
        tac: 1,
        link_ip: ip,
        ngap_ip: ip,
        gtp_ip: ip,
        nwdaf_enabled: true,
        ..Default::default()
    }
}

/// A UDP heartbeat from a UE, as the UE RLS task sends it.
fn heartbeat_bytes(sti: u64) -> Bytes {
    let msg =
        RlsProtocolMessage::Heartbeat(RlsHeartbeat::with_position(sti, SimCoord::new(10, 20, 0)));
    codec::encode(&msg)
}

/// Criterion 1 (gNB half): an RLS heartbeat reaches the gNB NWDAF task.
///
/// The RLS task's measurement forward used to return early on `sixg: None`, so
/// with the old wiring no `UeMeasurement` ever arrives and this times out.
#[tokio::test]
async fn an_rls_heartbeat_reaches_the_gnb_nwdaf_task() {
    let config = gnb_config(2);
    let rls_addr = SocketAddr::new(config.link_ip, DEFAULT_RLS_PORT);

    let mut wiring = GnbWiring::new(config, DEFAULT_CHANNEL_CAPACITY);
    let sixg = wiring
        .sixg
        .as_mut()
        .expect("NWDAF is enabled and compiled in");
    let mut measurements = probe(&mut sixg.nwdaf_rx, |msg| match msg {
        GnbTaskMessage::Message(NwdafMessage::UeMeasurement { ue_id, rsrp, .. }) => {
            Some((*ue_id, *rsrp))
        }
        _ => None,
    });
    let mut task_manager = wiring.spawn(true);

    let ue = UdpSocket::bind("127.0.0.1:0").await.expect("UE socket");
    let heartbeat = heartbeat_bytes(0x0452);
    let deadline = Instant::now() + CROSSING_TIMEOUT;
    // Resent until it lands: the RLS socket binds inside the spawned task, so
    // the first datagram can arrive before anything is listening.
    let (ue_id, rsrp) = loop {
        ue.send_to(&heartbeat, rls_addr)
            .await
            .expect("send heartbeat");
        match timeout(Duration::from_millis(200), measurements.recv()).await {
            Ok(Some(seen)) => break seen,
            Ok(None) => panic!("the NWDAF probe closed"),
            Err(_) if Instant::now() < deadline => continue,
            Err(_) => panic!("no RLS measurement reached the gNB NWDAF task"),
        }
    };

    assert_eq!(
        ue_id, 0,
        "filed under the id RLS assigned its first detected UE"
    );
    assert!(
        rsrp.is_some(),
        "the measurement carries the RSRP the RLS channel model computed"
    );

    let _ = task_manager.shutdown().await;
}

/// The task manager's own base carries the 6G and Rel-18 handles, so
/// `shutdown` reaches the NWDAF and Energy tasks. This is criterion 2's check
/// for `init_rel18_tasks`: the gNB has no core-task producer for the Energy
/// task, so its shutdown signal is the message that shows the handle is shared.
#[tokio::test]
async fn gnb_shutdown_reaches_the_6g_and_rel18_tasks() {
    let config = GnbConfig {
        energy_saving_enabled: true,
        ..gnb_config(3)
    };

    let mut wiring = GnbWiring::new(config, DEFAULT_CHANNEL_CAPACITY);
    let sixg = wiring
        .sixg
        .as_mut()
        .expect("NWDAF is enabled and compiled in");
    let mut nwdaf_shutdown = probe(&mut sixg.nwdaf_rx, |msg| {
        matches!(msg, GnbTaskMessage::Shutdown).then_some(())
    });
    let rel18 = wiring.rel18.as_mut().expect("energy saving is enabled");
    let mut energy_shutdown = probe(&mut rel18.energy_rx, |msg| {
        matches!(msg, GnbTaskMessage::Shutdown).then_some(())
    });
    let mut task_manager = wiring.spawn(true);

    let _ = task_manager.shutdown().await;

    timeout(CROSSING_TIMEOUT, nwdaf_shutdown.recv())
        .await
        .expect("shutdown did not reach the NWDAF task")
        .expect("NWDAF probe closed before shutdown arrived");
    timeout(CROSSING_TIMEOUT, energy_shutdown.recv())
        .await
        .expect("shutdown did not reach the Energy task")
        .expect("Energy probe closed before shutdown arrived");
}

/// A UE config with the SHE client enabled and no gNB to search for.
fn ue_config() -> UeConfig {
    UeConfig {
        she_enabled: true,
        gnb_search_list: vec![],
        ..Default::default()
    }
}

/// Criterion 1 (UE half): a UE RRC event reaches the UE SHE client.
///
/// The UE RRC task's `route_6g_inference` used to find `sixg: None` and drop
/// the request with "6G tasks not initialized".
#[tokio::test]
async fn a_ue_rrc_event_reaches_the_ue_she_client() {
    let mut wiring = UeWiring::new(ue_config(), DEFAULT_CHANNEL_CAPACITY);
    let sixg = wiring.sixg.as_mut().expect("the SHE client is enabled");
    let mut requests = probe(&mut sixg.she_client_rx, |msg| match msg {
        UeTaskMessage::Message(SheClientMessage::InferenceRequest {
            model_id, input, ..
        }) => Some((model_id.clone(), input.clone())),
        _ => None,
    });
    let task_base = wiring.task_base();
    // Held, not run: the NAS loop is binary code, and the channel must stay open.
    let (mut task_manager, _nas_rx) = wiring.spawn(None);

    task_base
        .rrc_tx
        .send(UeRrcMessage::SixgInferenceRequest {
            model_id: "model-452".to_string(),
            input_data: vec![1.0, 2.0],
        })
        .await
        .expect("send to the UE RRC task");

    let (model_id, input) = timeout(CROSSING_TIMEOUT, requests.recv())
        .await
        .expect("the RRC inference request did not reach the SHE client")
        .expect("SHE client probe closed");
    assert_eq!(model_id, "model-452");
    assert_eq!(input, vec![1.0, 2.0]);

    let _ = task_manager.shutdown().await;
}

/// The UE task manager's base carries the 6G handles, so `shutdown` reaches
/// the SHE client.
#[tokio::test]
async fn ue_shutdown_reaches_the_6g_tasks() {
    let mut wiring = UeWiring::new(ue_config(), DEFAULT_CHANNEL_CAPACITY);
    let sixg = wiring.sixg.as_mut().expect("the SHE client is enabled");
    let mut she_shutdown = probe(&mut sixg.she_client_rx, |msg| {
        matches!(msg, UeTaskMessage::Shutdown).then_some(())
    });
    let (mut task_manager, _nas_rx) = wiring.spawn(None);

    let _ = task_manager.shutdown().await;

    timeout(CROSSING_TIMEOUT, she_shutdown.recv())
        .await
        .expect("shutdown did not reach the SHE client")
        .expect("SHE client probe closed before shutdown arrived");
}

/// #459: the UE NWDAF reporter gets its first message from its production
/// producer. A real gNB (library wiring) and a real UE (library wiring) run on
/// loopback; once the UE RRC selects the cell from the RLS heartbeat exchange,
/// its cycle sends the serving cell's RSRP to the reporter. Nothing else sends
/// `ReportMeasurement`, so the probe seeing one is the producer working.
#[tokio::test]
async fn the_ue_rrc_cycle_feeds_the_ue_nwdaf_reporter() {
    let gnb_config = gnb_config(4);
    let gnb_ip = gnb_config.link_ip;
    let gnb_manager = GnbWiring::new(gnb_config, DEFAULT_CHANNEL_CAPACITY).spawn(true);

    let ue_config = UeConfig {
        ai_ml_enabled: true,
        gnb_search_list: vec![gnb_ip.to_string()],
        ..Default::default()
    };
    let mut wiring = UeWiring::new(ue_config, DEFAULT_CHANNEL_CAPACITY);
    let sixg = wiring.sixg.as_mut().expect("the NWDAF reporter is enabled");
    let mut reports = probe(&mut sixg.nwdaf_reporter_rx, |msg| match msg {
        UeTaskMessage::Message(NwdafReporterMessage::ReportMeasurement { rsrp, rsrq, .. }) => {
            Some((*rsrp, *rsrq))
        }
        _ => None,
    });
    let (mut ue_manager, _nas_rx) = wiring.spawn(None);

    // Cell detection, then selection on the RRC cycle (2.5 s), then a report.
    let (rsrp, rsrq) = timeout(Duration::from_secs(30), reports.recv())
        .await
        .expect("no serving-cell measurement reached the UE NWDAF reporter")
        .expect("NWDAF reporter probe closed");
    assert!(rsrp < 0.0, "a received power in dBm, got {rsrp}");
    assert_eq!(rsrq, None, "no RSRQ invented");

    let _ = ue_manager.shutdown().await;
    let mut gnb_manager = gnb_manager;
    let _ = gnb_manager.shutdown().await;
}

/// #459, FL: the hook the FL demo plugs its producer into reaches the running
/// FL aggregator through the library wiring. The FL tasks have no production
/// producer by design (the FL demo redesigns them); this pins the hook.
#[tokio::test]
async fn the_fl_hook_reaches_the_gnb_fl_aggregator() {
    let config = GnbConfig {
        federated_learning_enabled: true,
        nwdaf_enabled: false,
        ..gnb_config(5)
    };
    let mut wiring = GnbWiring::new(config, DEFAULT_CHANNEL_CAPACITY);
    let sixg = wiring.sixg.as_mut().expect("FL is enabled and compiled in");
    let mut seen = probe(&mut sixg.fl_rx, |msg| match msg {
        GnbTaskMessage::Message(FlAggregatorMessage::RegisterParticipant {
            participant_id,
            ..
        }) => Some(participant_id.clone()),
        _ => None,
    });
    let task_base = wiring.task_base();
    let mut task_manager = wiring.spawn(true);

    task_base
        .fl_aggregator_tx()
        .expect("the hook exists once the 6G handles are installed")
        .send(FlAggregatorMessage::RegisterParticipant {
            participant_id: "ue-459".to_string(),
            compute_capability: 1,
        })
        .await
        .expect("send through the FL hook");

    let id = timeout(CROSSING_TIMEOUT, seen.recv())
        .await
        .expect("nothing reached the FL aggregator")
        .expect("FL probe closed");
    assert_eq!(id, "ue-459");
    let _ = task_manager.shutdown().await;
}

/// #459, Agent: the gNB Agent task gets its first message from its production
/// producer, the RLS task's periodic cell observation, through the library
/// wiring. Nothing else sends `CellObservation`.
#[tokio::test]
async fn the_rls_tick_feeds_the_gnb_agent() {
    let config = GnbConfig {
        agent_enabled: true,
        nwdaf_enabled: false,
        ..gnb_config(6)
    };
    let mut wiring = GnbWiring::new(config, DEFAULT_CHANNEL_CAPACITY);
    let sixg = wiring
        .sixg
        .as_mut()
        .expect("the Agent is enabled and compiled in");
    let mut observations = probe(&mut sixg.agent_rx, |msg| match msg {
        GnbTaskMessage::Message(AgentMessage::CellObservation { connected_ues, .. }) => {
            Some(*connected_ues)
        }
        _ => None,
    });
    let mut task_manager = wiring.spawn(true);

    let connected_ues = timeout(CROSSING_TIMEOUT, observations.recv())
        .await
        .expect("no cell observation reached the gNB Agent task")
        .expect("Agent probe closed");
    assert_eq!(connected_ues, 0, "no UE has attached to this cell");
    let _ = task_manager.shutdown().await;
}
