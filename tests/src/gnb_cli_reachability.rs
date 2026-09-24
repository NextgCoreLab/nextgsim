//! `nr-cli` reachability for the gNB (issue #197).
//!
//! Proves the property that was silently false: that a running gNB is discoverable
//! **the way `nr-cli` discovers it**, and that a command sent to the discovered port
//! is executed and answered.
//!
//! `nr-cli` never learns a port directly. It resolves a node name to a port by
//! scanning `PROC_TABLE_DIR` for an entry that names it at a matching protocol
//! version (`nextgsim-cli/src/proc_table.rs`, `discover_node`), then speaks the
//! `CliMessage` framing at that port. The gNB used to bind an ephemeral port and
//! register nowhere, so `nr-cli gnb --exec ...` failed with
//! `No node found with name 'gnb'` and the gNB's whole command set — `ue-list`,
//! `ue-info`, `ue-release`, `ue-suspend`, `xn-path-switch`, `ran-config-update` —
//! was unreachable from outside the process.
//!
//! These tests therefore assert POSITIVELY at each link in that chain:
//!
//! 1. the proc-table entry exists and its decoded contents name the node at the
//!    port the server actually bound (not merely that registration returned `Ok`);
//! 2. a `ue-suspend` encoded exactly as `nr-cli` encodes it, sent to the
//!    *discovered* port, produces the observable effect — `RrcMessage::SuspendUe`
//!    arriving on the real RRC inbox — and a non-error result back to the client.
//!
//! Step 2 is what makes this end to end rather than a registration unit test: it is
//! satisfied only if discovery, framing, node-name filtering, command parsing,
//! dispatch and the response path all agree. Removing the `register_nodes` call in
//! `AppTask::init_cli_server` makes `discovers_gnb_and_executes_ue_suspend` fail at
//! the discovery assertion, because there is then no entry to find.
//!
//! `ue-suspend` is the command exercised on purpose: nextgcore #403 needs it to
//! drive a UE to CM-IDLE and back, which is the only way to reach that core's
//! SERVICE REQUEST fire point (TS 38.331 §5.3.8.3 for the RRC_INACTIVE transition).
//!
//! # Issue #199: the registry the reachability tests had to seed
//!
//! The three tests above reach the gNB, but they seeded `AppTask.ue_contexts` by
//! hand — and that hid the *next* defect completely. `AppTask.ue_contexts` had **no
//! production writer**: `update_ue_context` was called only from unit tests and from
//! this file, `AppMessage` had no UE-lifecycle variant, and the only `app_tx.send`
//! in the whole gNB carried `StatusType::NgapIsUp`. So a real gNB answered
//! `ues: []` for a fully registered UE with an ACTIVE PDU session, `ue-suspend`
//! answered `UE not found with ID: n`, and because the dispatch to RRC is gated on
//! `!response.is_error`, `RrcMessage::SuspendUe` was unreachable in production.
//!
//! `a_real_registration_makes_the_ue_visible_and_suspendable` is the regression
//! test for that, and it deliberately **does not touch the registry**. It drives a
//! genuine registration — real UE RRC, real gNB RRC, the real NGAP task running its
//! own message loop, a captured NG Setup Response and a real Downlink NAS Transport
//! encoded by the production codec — and then
//! asserts, through `nr-cli`'s own discovery and framing, that `ue-list` names the
//! UE by the `ran_ue_ngap_id` the NGAP task allocated and that `ue-suspend <that
//! id>` reaches the real RRC inbox.
//!
//! Revert-verified, and the result is worth writing down because it is not what it
//! looks like. Making `NgapTask::sync_ue_context_to_app` publish nothing makes both
//! #199 tests fail with the exact symptoms from the issue — `ues: []` and
//! `UE not found with ID: 3` — while the three #197 tests above keep passing.
//! Neutralising the `UeContextRemove` send in `delete_ue_context` fails
//! `a_released_ue_is_no_longer_offered_to_ue_suspend` alone, with the CLI still
//! answering `Suspending UE 3` for a released UE.
//!
//! But removing *only* the `sync_ue_context_to_app` call in `create_ue_context`
//! leaves both tests passing, because the republish on the AMF's Downlink NAS
//! Transport then covers it. That is a real property of the design rather than a gap
//! in the test: the publish is deliberately repeated at every point the NGAP ID pair
//! can change, so no single one of them is individually load-bearing. The creation
//! one still earns its place — it is the only one that fires for a UE the AMF has not
//! yet answered, which is what `ue-list` should show during an in-flight
//! registration — but a test asserting a *registered* UE cannot distinguish that, and
//! pretending otherwise would be the same self-satisfied assertion this file exists
//! to avoid.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicU16, Ordering};
use std::time::Duration;

use nextgsim_common::cli_server::{
    CliMessage, CliMessageType, ProcTableEntry, PROC_TABLE_DIR, VERSION_MAJOR, VERSION_MINOR,
    VERSION_PATCH,
};
use nextgsim_common::config::{GnbConfig, UeConfig};
use nextgsim_common::{lookup_node_port, OctetString, Plmn};
use nextgsim_gnb::tasks::{
    AppMessage, GnbTaskBase, NgapMessage, RlsMessage as GnbRlsMessage, RrcMessage, SctpMessage,
    Task, TaskMessage, UeContextUpdate, DEFAULT_CHANNEL_CAPACITY,
};
use nextgsim_gnb::{AppTask, NgapTask, RrcTask as GnbRrcTask};
use nextgsim_ngap::procedures::initial_ue_message::decode_initial_ue_message;
use nextgsim_ngap::procedures::nas_transport::{
    encode_downlink_nas_transport, DownlinkNasTransportParams,
};
use nextgsim_ngap::procedures::ng_setup::{NasCause, NgSetupFailureCause};
use nextgsim_ngap::procedures::ue_context_release::{
    encode_ue_context_release_command, UeContextReleaseCommandParams, UeNgapIds,
};
use nextgsim_rls::RrcChannel;
use nextgsim_ue::{RrcTask as UeRrcTask, UeTaskBase};
use tokio::net::UdpSocket;

/// The UE the suspend is aimed at in the three ORIGINAL #197 reachability tests,
/// which seed the registry deliberately: they pin discovery and framing, and a
/// registration would add a second reason for them to fail.
///
/// `a_real_registration_makes_the_ue_visible_and_suspendable` uses none of these —
/// it learns the ID from the wire, because a test that asserts an ID it also chose
/// cannot tell whether production published anything.
const UE_ID: i32 = 7;
const RAN_UE_NGAP_ID: i64 = 700;
const AMF_UE_NGAP_ID: i64 = 7000;

/// A `t380` that is a legal `PeriodicRNAU-TimerValue`, so the command is accepted
/// and its value can be asserted on the far side.
const T380_MINUTES: u16 = 20;

/// Hands out a distinct node-name suffix per test in this binary.
///
/// The proc table is a process-global directory keyed by PID, and every test here
/// shares this process's PID. Distinct node names keep one test's entry from
/// answering another's lookup, which is the isolation a per-test store would have
/// given for free.
static NODE_SEQ: AtomicU16 = AtomicU16::new(0);

fn unique_node_name() -> String {
    // Kept within `nr-cli`'s MIN_NODE_NAME..=MAX_NODE_NAME (3..=30) bounds so the
    // name under test is one `nr-cli` would actually accept on its command line.
    format!("gnb-cli-t{}", NODE_SEQ.fetch_add(1, Ordering::SeqCst))
}

fn test_config() -> GnbConfig {
    GnbConfig {
        nci: 0x0000_0000_0010,
        gnb_id_length: 32,
        plmn: Plmn::new(310, 410, false),
        tac: 1,
        ..Default::default()
    }
}

/// Builds an App task with the CLI enabled, holding one UE context, plus the RRC
/// inbox the suspend is expected to arrive on.
///
/// The UE is planted by SENDING the App task the same `AppMessage::UeContextUpdate`
/// the NGAP task sends, rather than by reaching into its map: the direct writer is
/// private now, and going through the message keeps these tests honest about the
/// only way a UE can enter the registry. They still do not prove that anything
/// production-side sends it — that is
/// `a_real_registration_makes_the_ue_visible_and_suspendable`'s job.
async fn spawn_app_task() -> (
    AppTask,
    tokio::sync::mpsc::Receiver<TaskMessage<AppMessage>>,
    tokio::sync::mpsc::Receiver<TaskMessage<RrcMessage>>,
) {
    let (base, app_rx, _ngap_rx, rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
        GnbTaskBase::new(test_config(), DEFAULT_CHANNEL_CAPACITY);

    base.app_tx
        .send(AppMessage::UeContextUpdate(UeContextUpdate {
            ue_id: UE_ID,
            ran_ue_ngap_id: RAN_UE_NGAP_ID,
            amf_ue_ngap_id: Some(AMF_UE_NGAP_ID),
        }))
        .await
        .expect("the App task's own inbox accepts the seed");

    let task = AppTask::new(base);

    (task, app_rx, rrc_rx)
}

/// Reads the proc-table entry that names `node_name`, as `nr-cli` parses it.
///
/// Returns the decoded entry rather than just a port so a caller can assert the
/// entry's full contents — the PID, the version triple and the node list — instead
/// of only the field it happens to need.
fn read_proc_table_entry(node_name: &str) -> Option<ProcTableEntry> {
    for entry in std::fs::read_dir(PROC_TABLE_DIR).ok()?.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Ok(content) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Some(decoded) = ProcTableEntry::decode(&content) else {
            continue;
        };
        if decoded.nodes.iter().any(|n| n == node_name) {
            return Some(decoded);
        }
    }
    None
}

/// The gNB writes a proc-table entry whose CONTENTS name it at the port it bound.
///
/// Asserts the entry's fields rather than the absence of an error: the previous
/// behaviour also produced no error, it simply produced no entry.
#[tokio::test]
async fn registers_a_discoverable_proc_table_entry() {
    let node_name = unique_node_name();

    let (mut task, _app_rx, _rrc_rx) = spawn_app_task().await;
    let port = task
        .init_cli_server(node_name.clone())
        .await
        .expect("the CLI server binds and registers");

    assert!(port > 0, "the CLI server must bind a real port");
    assert_eq!(
        task.cli_port(),
        port,
        "the task must report the port it registered"
    );

    let entry = read_proc_table_entry(&node_name)
        .expect("a proc-table entry naming the gNB must exist -- this is what nr-cli reads");

    assert_eq!(
        entry.port, port,
        "the registered port must be the port the server bound, or nr-cli connects nowhere"
    );
    assert_eq!(
        entry.nodes,
        vec![node_name.clone()],
        "the entry must name exactly the node the gNB was started as"
    );
    assert_eq!(
        entry.pid,
        std::process::id(),
        "the entry must carry this process's PID, or nr-cli prunes it as stale"
    );
    assert_eq!(
        (entry.major, entry.minor, entry.patch),
        (VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH),
        "a version-mismatched entry is skipped by nr-cli's discover_node"
    );

    // And the shared lookup -- the same scan nr-cli performs -- resolves the name.
    assert_eq!(
        lookup_node_port(&node_name),
        Some(port),
        "discovery by node name must yield the bound port"
    );
}

/// End to end: discover the gNB by name, send `ue-suspend` framed as `nr-cli` frames
/// it, and observe both the RRC effect and the CLI answer.
///
/// This is the criterion from issue #197. It fails at the discovery assertion if
/// `AppTask::init_cli_server` stops calling `register_nodes`.
#[tokio::test]
async fn discovers_gnb_and_executes_ue_suspend() {
    let node_name = unique_node_name();

    let (mut task, app_rx, mut rrc_rx) = spawn_app_task().await;
    let bound_port = task
        .init_cli_server(node_name.clone())
        .await
        .expect("the CLI server binds and registers");

    // Resolve the port the way nr-cli does. Deliberately NOT reusing `bound_port`:
    // the whole defect was that the bound port was undiscoverable, so the test must
    // reach the gNB through discovery or it proves nothing.
    let discovered_port = lookup_node_port(&node_name)
        .expect("nr-cli's node lookup must find the running gNB by name");
    assert_eq!(
        discovered_port, bound_port,
        "discovery must resolve to the port the gNB is actually listening on"
    );

    // Run the App task so it polls its CLI server.
    let handle = tokio::spawn(async move { task.run(app_rx).await });

    // A client speaking exactly what nr-cli speaks: same framing, same version.
    let client = UdpSocket::bind("127.0.0.1:0")
        .await
        .expect("the client socket binds");
    let client_addr = client.local_addr().expect("the client has a local address");
    let target: SocketAddr = format!("127.0.0.1:{discovered_port}")
        .parse()
        .expect("the discovered port forms a valid address");

    let command = CliMessage {
        msg_type: CliMessageType::Command,
        node_name: node_name.clone(),
        value: format!("ue-suspend {UE_ID} {T380_MINUTES}"),
        client_addr,
    };
    client
        .send_to(&command.encode(), target)
        .await
        .expect("the command reaches the discovered port");

    // The observable effect: the real RRC inbox receives the suspend, carrying the
    // arguments parsed out of the command string.
    let rrc_msg = tokio::time::timeout(Duration::from_secs(5), rrc_rx.recv())
        .await
        .expect("the gNB must act on a command sent to its discovered port")
        .expect("the RRC channel stays open");

    match rrc_msg {
        TaskMessage::Message(RrcMessage::SuspendUe {
            ue_id,
            t380_minutes,
        }) => {
            assert_eq!(
                ue_id, UE_ID,
                "the suspend must target the UE named on the CLI"
            );
            assert_eq!(
                t380_minutes,
                Some(T380_MINUTES),
                "the t380 argument must survive the CLI round trip"
            );
        }
        other => panic!("expected RrcMessage::SuspendUe from the CLI command, got {other:?}"),
    }

    // And the client gets a non-error result back, so `nr-cli` reports success
    // rather than timing out.
    let mut buf = [0u8; 8192];
    let (size, _from) = tokio::time::timeout(Duration::from_secs(5), client.recv_from(&mut buf))
        .await
        .expect("the gNB must answer the CLI client")
        .expect("the response is received");

    let response = CliMessage::decode(&buf[..size], target)
        .expect("the response must decode at nr-cli's protocol version");
    assert_eq!(
        response.msg_type,
        CliMessageType::Result,
        "a successful ue-suspend must answer with Result, not Error: {}",
        response.value
    );
    assert!(
        response.value.contains(&UE_ID.to_string()),
        "the result must describe the UE it suspended, got {:?}",
        response.value
    );

    handle.abort();
}

/// A command naming a DIFFERENT node is ignored, so two instances sharing the host
/// do not answer each other's traffic.
///
/// Guards the node-name filter in `CliServer::receive_command`, which only has teeth
/// once `register_nodes` has been called -- an unregistered server has an empty node
/// list and answers everything.
#[tokio::test]
async fn ignores_commands_addressed_to_another_node() {
    let node_name = unique_node_name();

    let (mut task, app_rx, mut rrc_rx) = spawn_app_task().await;
    let port = task
        .init_cli_server(node_name.clone())
        .await
        .expect("the CLI server binds and registers");

    let handle = tokio::spawn(async move { task.run(app_rx).await });

    let client = UdpSocket::bind("127.0.0.1:0")
        .await
        .expect("the client socket binds");
    let client_addr = client.local_addr().expect("the client has a local address");
    let target: SocketAddr = format!("127.0.0.1:{port}")
        .parse()
        .expect("the port forms a valid address");

    let misaddressed = CliMessage {
        msg_type: CliMessageType::Command,
        node_name: format!("{node_name}-other"),
        value: format!("ue-suspend {UE_ID}"),
        client_addr,
    };
    client
        .send_to(&misaddressed.encode(), target)
        .await
        .expect("the datagram is sent");

    // A positively-asserted absence: the RRC inbox must still be EMPTY after the
    // poll interval has elapsed several times over, and `try_recv` distinguishes
    // "nothing was sent" from "not yet delivered" -- the send above has completed.
    tokio::time::sleep(Duration::from_millis(600)).await;
    assert!(
        matches!(
            rrc_rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ),
        "a command for another node must not drive this gNB's RRC"
    );

    handle.abort();
}

// ============================================================================
// Issue #199: the CLI registry must be filled by a REAL registration
// ============================================================================

/// The UE the registration test drives. Only the internal `ue_id` is chosen here —
/// the `ran_ue_ngap_id` that `ue-list` is asserted against is read off the wire, so
/// the test cannot agree with an implementation that publishes the wrong one.
const REG_UE_ID: i32 = 3;

/// The AMF association the registration runs over.
const AMF_CLIENT_ID: i32 = 1;
const NGAP_STREAM: u16 = 0;
const CELL_ID: i32 = 1;

/// The AMF UE NGAP ID the mock AMF assigns in its first downlink message.
const ASSIGNED_AMF_UE_NGAP_ID: u64 = 0x00A1_B2C3;

/// NG Setup Response as produced by the core's own ogs-ngap codec (AMFName
/// "nextgcore-amf", GUAMI 001-01, capacity 255, PLMN 001-01 with S-NSSAI sst=1).
///
/// The same captured vector `paging_mt_service_request` uses, and reused here for the
/// same reason: the AMF association must reach `Ready` on a real decoded message, not
/// on a test-only state mutation, because `handle_ngap_pdu` refuses to route any
/// operational PDU until it does.
const CORE_NG_SETUP_RESPONSE: [u8; 55] = [
    0x20, 0x15, 0x00, 0x33, 0x00, 0x00, 0x04, 0x00, 0x01, 0x00, 0x0f, 0x06, 0x00, 0x6e, 0x65, 0x78,
    0x74, 0x67, 0x63, 0x6f, 0x72, 0x65, 0x2d, 0x61, 0x6d, 0x66, 0x00, 0x60, 0x00, 0x08, 0x00, 0x00,
    0x00, 0xf1, 0x10, 0x02, 0x00, 0x41, 0x00, 0x56, 0x40, 0x01, 0xff, 0x00, 0x50, 0x00, 0x08, 0x00,
    0x00, 0xf1, 0x10, 0x00, 0x00, 0x00, 0x08,
];

/// Registration Request NAS PDU (5GMM EPD 0x7E, message type 0x41), as
/// `rrc_handshake` uses. Only its opacity matters: the gNB relays it untouched.
fn registration_nas() -> Vec<u8> {
    vec![0x7E, 0x00, 0x41, 0x79, 0x00, 0x0D, 0x01, 0x00, 0xF1, 0x10]
}

/// A gNB serving PLMN 001-01, matching both the captured NG Setup Response's served
/// PLMN and the UE's default HPLMN, so AMF selection and cell selection both succeed.
fn registration_gnb_config() -> GnbConfig {
    GnbConfig {
        nci: 0x0000_0000_0010,
        gnb_id_length: 32,
        plmn: Plmn::new(1, 1, false),
        tac: 1,
        ..Default::default()
    }
}

/// Everything the registration harness hands back: the live channel ends, plus the
/// discovered CLI port and the RAN UE NGAP ID production allocated.
struct RegisteredUe {
    /// The node name the gNB registered under, needed to frame CLI datagrams.
    node_name: String,
    /// The port resolved through `nr-cli`'s own proc-table lookup.
    discovered_port: u16,
    /// The RAN UE NGAP ID read out of the Initial UE Message on the wire.
    wire_ran_ue_ngap_id: i64,
    /// The gNB's real RRC inbox — where `RrcMessage::SuspendUe` must arrive.
    rrc_rx: tokio::sync::mpsc::Receiver<TaskMessage<RrcMessage>>,
    /// The gNB's real SCTP outbox — the AMF's view of what this node sent.
    sctp_rx: tokio::sync::mpsc::Receiver<TaskMessage<SctpMessage>>,
    /// Handle to send the AMF's NGAP PDUs into the real NGAP task.
    ngap_tx: nextgsim_gnb::TaskHandle<NgapMessage>,
    /// Kept so the spawned App and NGAP tasks are not dropped mid-test.
    handles: Vec<tokio::task::JoinHandle<()>>,
}

/// Pops the next RRC PDU the UE handed to its RLS, skipping non-PDU RLS traffic.
fn next_ue_uplink_rrc(
    rx: &mut tokio::sync::mpsc::Receiver<nextgsim_ue::TaskMessage<nextgsim_ue::RlsMessage>>,
) -> (RrcChannel, OctetString) {
    loop {
        match rx.try_recv() {
            Ok(nextgsim_ue::TaskMessage::Message(nextgsim_ue::RlsMessage::RrcPduDelivery {
                channel,
                pdu,
                ..
            })) => return (channel, pdu),
            Ok(_) => continue,
            Err(e) => panic!("expected an uplink RRC PDU from the UE, got none: {e}"),
        }
    }
}

/// Pops the next downlink RRC PDU the gNB handed to its RLS.
fn next_gnb_downlink_rrc(
    rx: &mut tokio::sync::mpsc::Receiver<TaskMessage<GnbRlsMessage>>,
) -> (RrcChannel, OctetString) {
    loop {
        match rx.try_recv() {
            Ok(TaskMessage::Message(GnbRlsMessage::DownlinkRrc {
                rrc_channel, data, ..
            })) => return (rrc_channel, data),
            Ok(_) => continue,
            Err(e) => panic!("expected a downlink RRC PDU from the gNB, got none: {e}"),
        }
    }
}

/// Waits up to `budget` for an NGAP PDU the gNB sent to the AMF that `pick` accepts.
///
/// Needed because the NGAP task runs in its own tokio task here: `try_recv` would
/// race its progress, and this test must not depend on scheduler luck.
async fn await_sctp_pdu<T>(
    rx: &mut tokio::sync::mpsc::Receiver<TaskMessage<SctpMessage>>,
    budget: Duration,
    mut pick: impl FnMut(&[u8]) -> Option<T>,
) -> Option<T> {
    let deadline = tokio::time::Instant::now() + budget;
    while tokio::time::Instant::now() < deadline {
        match tokio::time::timeout(Duration::from_millis(200), rx.recv()).await {
            Ok(Some(TaskMessage::Message(SctpMessage::SendMessage { buffer, .. }))) => {
                if let Some(found) = pick(buffer.data()) {
                    return Some(found);
                }
            }
            Ok(Some(_)) => continue,
            Ok(None) => return None,
            Err(_) => continue,
        }
    }
    None
}

/// Waits up to `budget` for a `RrcMessage::SuspendUe` on the gNB's real RRC inbox.
///
/// Scans rather than taking the head: a registration also puts `RadioPowerOn` and
/// `NasDelivery` on this channel, and asserting on the head would make the test
/// depend on how many of those the gNB happened to emit.
async fn await_rrc_suspend(
    rx: &mut tokio::sync::mpsc::Receiver<TaskMessage<RrcMessage>>,
    budget: Duration,
) -> Option<(i32, Option<u16>)> {
    let deadline = tokio::time::Instant::now() + budget;
    while tokio::time::Instant::now() < deadline {
        match tokio::time::timeout(Duration::from_millis(200), rx.recv()).await {
            Ok(Some(TaskMessage::Message(RrcMessage::SuspendUe {
                ue_id,
                t380_minutes,
            }))) => return Some((ue_id, t380_minutes)),
            Ok(Some(_)) => continue,
            Ok(None) => return None,
            Err(_) => continue,
        }
    }
    None
}

/// A CLI client speaking exactly what `nr-cli` speaks, bound to a fresh local port.
async fn cli_client() -> (UdpSocket, SocketAddr) {
    let sock = UdpSocket::bind("127.0.0.1:0")
        .await
        .expect("the client socket binds");
    let addr = sock.local_addr().expect("the client has a local address");
    (sock, addr)
}

/// Sends `command` to the discovered gNB port as `nr-cli` frames it and returns the
/// decoded answer.
async fn exec_cli(reg: &RegisteredUe, command: &str) -> CliMessage {
    let (client, client_addr) = cli_client().await;
    let target: SocketAddr = format!("127.0.0.1:{}", reg.discovered_port)
        .parse()
        .expect("the discovered port forms a valid address");

    let msg = CliMessage {
        msg_type: CliMessageType::Command,
        node_name: reg.node_name.clone(),
        value: command.to_string(),
        client_addr,
    };
    client
        .send_to(&msg.encode(), target)
        .await
        .expect("the command reaches the discovered port");

    let mut buf = [0u8; 8192];
    let (size, _from) = tokio::time::timeout(Duration::from_secs(5), client.recv_from(&mut buf))
        .await
        .unwrap_or_else(|_| panic!("the gNB must answer `{command}`"))
        .expect("the response is received");

    CliMessage::decode(&buf[..size], target)
        .expect("the response must decode at nr-cli's protocol version")
}

/// Drives a genuine registration to the point where the gNB holds a UE-associated
/// logical NG-connection, then returns the live ends plus the discovered CLI port.
///
/// **Nothing here writes the CLI registry.** The chain is:
///
/// ```text
/// SctpAssociationUp + captured NG Setup Response  -> AMF context Ready
/// UE RRC RRCSetupRequest (real ASN.1, UL-CCCH)    -> gNB RRC RRCSetup (DL-CCCH)
/// UE RRC RRCSetupComplete (UL-DCCH)               -> NgapMessage::InitialNasDelivery
/// real NGAP task handle_initial_nas_delivery       -> create_ue_context
///                                                  -> Initial UE Message on the wire
///                                                  -> AppMessage::UeContextUpdate
/// AMF Downlink NAS Transport (real encoder)        -> AMF UE NGAP ID adopted
///                                                  -> AppMessage::UeContextUpdate
/// ```
///
/// Every hop is production code reached through a real message; the only test-side
/// inputs are the AMF's own PDUs and the UE's stimulus.
async fn register_a_ue() -> RegisteredUe {
    let node_name = unique_node_name();

    // One shared task base, so the App, RRC and NGAP tasks are wired to each other
    // the way `main.rs` wires them.
    let (base, app_rx, ngap_rx, rrc_rx, _gtp_rx, mut rls_rx, sctp_rx) =
        GnbTaskBase::new(registration_gnb_config(), DEFAULT_CHANNEL_CAPACITY);
    let mut sctp_rx = sctp_rx;
    let ngap_tx = base.ngap_tx.clone();

    // The App task, with its CLI server registered so `nr-cli` can find it.
    let mut app_task = AppTask::new(base.clone());
    let bound_port = app_task
        .init_cli_server(node_name.clone())
        .await
        .expect("the CLI server binds and registers");

    // Resolve the port the way `nr-cli` does, not from `bound_port`.
    let discovered_port = lookup_node_port(&node_name)
        .expect("nr-cli's node lookup must find the running gNB by name");
    assert_eq!(
        discovered_port, bound_port,
        "discovery must resolve to the port the gNB is actually listening on"
    );

    let mut handles = Vec::new();
    handles.push(tokio::spawn(async move { app_task.run(app_rx).await }));

    // The real NGAP task, run as a task so its own message dispatch -- not a
    // test-side call to a private handler -- routes the registration.
    let mut ngap_task = NgapTask::new(base.clone());
    handles.push(tokio::spawn(async move { ngap_task.run(ngap_rx).await }));

    // Bring the AMF association up and answer the NG Setup Request, so the NGAP task
    // will route operational PDUs and will select this AMF for a new UE.
    ngap_tx
        .send(NgapMessage::SctpAssociationUp {
            client_id: AMF_CLIENT_ID,
            association_id: 1,
            in_streams: 2,
            out_streams: 2,
        })
        .await
        .expect("the NGAP task accepts the association-up");
    assert!(
        await_sctp_pdu(&mut sctp_rx, Duration::from_secs(5), |bytes| {
            // NG Setup Request: initiatingMessage (0x00), procedure code 21 (0x15).
            (bytes.first() == Some(&0x00) && bytes.get(1) == Some(&0x15)).then_some(())
        })
        .await
        .is_some(),
        "the gNB must send an NG Setup Request before it can be Ready"
    );
    ngap_tx
        .send(NgapMessage::ReceiveNgapPdu {
            client_id: AMF_CLIENT_ID,
            stream: NGAP_STREAM,
            pdu: OctetString::from_slice(&CORE_NG_SETUP_RESPONSE),
        })
        .await
        .expect("the NGAP task accepts the NG Setup Response");

    // The RRC peers. The gNB RRC task shares the base above, so its
    // `InitialNasDelivery` goes to the real NGAP task rather than to a test channel.
    let mut gnb_rrc = GnbRrcTask::new(base.clone());
    let (ue_base, _ue_app_rx, _ue_nas_rx, _ue_rrc_rx, mut ue_rls_rx) =
        UeTaskBase::new(UeConfig::default(), 32);
    let mut ue_rrc = UeRrcTask::new(ue_base);

    // The real RRC connection establishment (TS 38.331 §5.3.3), exactly as
    // `rrc_handshake` drives it.
    gnb_rrc.handle_radio_power_on();
    ue_rrc.handle_signal_changed(CELL_ID, -60).await;
    ue_rrc.perform_cycle().await;

    let nas = registration_nas();
    ue_rrc
        .handle_uplink_nas_delivery(1, OctetString::from_slice(&nas))
        .await;
    let (ch, setup_req) = next_ue_uplink_rrc(&mut ue_rls_rx);
    assert_eq!(ch, RrcChannel::UlCcch, "RRCSetupRequest goes on UL-CCCH");

    gnb_rrc
        .handle_uplink_rrc(REG_UE_ID, RrcChannel::UlCcch, setup_req)
        .await;
    let (ch, rrc_setup) = next_gnb_downlink_rrc(&mut rls_rx);
    assert_eq!(ch, RrcChannel::DlCcch, "RRCSetup goes on DL-CCCH");

    ue_rrc
        .handle_downlink_rrc(CELL_ID, RrcChannel::DlCcch, rrc_setup)
        .await;
    let (ch, setup_complete) = next_ue_uplink_rrc(&mut ue_rls_rx);
    assert_eq!(ch, RrcChannel::UlDcch, "RRCSetupComplete goes on UL-DCCH");

    // This is the hop that reaches NGAP: the gNB RRC task emits
    // `NgapMessage::InitialNasDelivery` onto the shared channel, and the spawned
    // NGAP task picks it up, creates the UE context and sends the Initial UE Message.
    gnb_rrc
        .handle_uplink_rrc(REG_UE_ID, RrcChannel::UlDcch, setup_complete)
        .await;

    // Learn the RAN UE NGAP ID from the WIRE. This is the value production
    // allocated; the test has no other way to know it, which is what makes the
    // `ue-list` assertion below meaningful rather than self-fulfilling.
    let wire_ran_ue_ngap_id = await_sctp_pdu(&mut sctp_rx, Duration::from_secs(5), |bytes| {
        decode_initial_ue_message(bytes)
            .ok()
            .map(|data| i64::from(data.ran_ue_ngap_id))
    })
    .await
    .expect("the registration must produce an Initial UE Message on the wire");
    assert_eq!(
        wire_ran_ue_ngap_id, 1,
        "the first UE of a fresh gNB takes RAN UE NGAP ID 1"
    );

    // The AMF answers with a Downlink NAS Transport (an Authentication Request, in a
    // real registration), which is where the AMF UE NGAP ID first arrives.
    let dl_nas = encode_downlink_nas_transport(&DownlinkNasTransportParams {
        amf_ue_ngap_id: ASSIGNED_AMF_UE_NGAP_ID,
        ran_ue_ngap_id: wire_ran_ue_ngap_id as u32,
        nas_pdu: vec![0x7E, 0x00, 0x56, 0x00, 0x02],
        old_amf: None,
        ran_paging_priority: None,
        index_to_rfsp: None,
        ue_ambr: None,
        allowed_nssai: None,
    })
    .expect("encode Downlink NAS Transport");
    ngap_tx
        .send(NgapMessage::ReceiveNgapPdu {
            client_id: AMF_CLIENT_ID,
            stream: NGAP_STREAM,
            pdu: OctetString::from_slice(&dl_nas),
        })
        .await
        .expect("the NGAP task accepts the Downlink NAS Transport");

    RegisteredUe {
        node_name,
        discovered_port,
        wire_ran_ue_ngap_id,
        rrc_rx,
        sctp_rx,
        ngap_tx,
        handles,
    }
}

/// **The issue #199 criterion.** A UE that registered for real is named by `ue-list`
/// and can be suspended by `ue-suspend`, with the suspend observably reaching RRC.
///
/// Nothing in this test writes the CLI registry: the only writer exercised is
/// `NgapTask::sync_ue_context_to_app`, on the path a registration actually takes.
/// Remove that call and this test fails at the `ue-list` assertion — which is the
/// property #198's tests could not have, because they supplied the registry contents
/// themselves.
///
/// Asserts the UE's ACTUAL identifiers, not merely a non-empty list: the
/// `ran_ue_ngap_id` compared against is the one decoded from the Initial UE Message
/// the gNB put on the wire.
#[tokio::test]
async fn a_real_registration_makes_the_ue_visible_and_suspendable() {
    let mut reg = register_a_ue().await;

    // 1. `ue-list` names the UE, by both identifiers.
    let listed = exec_cli(&reg, "ue-list").await;
    assert_eq!(
        listed.msg_type,
        CliMessageType::Result,
        "ue-list must succeed: {}",
        listed.value
    );
    assert_ne!(
        listed.value.trim(),
        "ues: []",
        "a registered UE must not leave ue-list empty -- this is issue #199's symptom"
    );
    assert!(
        listed.value.contains(&format!("ue_id: {REG_UE_ID}")),
        "ue-list must name the registered UE's id, got {:?}",
        listed.value
    );
    assert!(
        listed
            .value
            .contains(&format!("ran_ngap_id: {}", reg.wire_ran_ue_ngap_id)),
        "ue-list must report the RAN UE NGAP ID the gNB put on the wire ({}), got {:?}",
        reg.wire_ran_ue_ngap_id,
        listed.value
    );

    // 2. `ue-info` carries the AMF UE NGAP ID the AMF assigned, so the two-phase
    //    publish (creation, then ID adoption) actually landed both halves.
    let info = exec_cli(&reg, &format!("ue-info {REG_UE_ID}")).await;
    assert_eq!(
        info.msg_type,
        CliMessageType::Result,
        "ue-info must succeed for a registered UE: {}",
        info.value
    );
    assert!(
        info.value
            .contains(&format!("amf_ue_ngap_id: {ASSIGNED_AMF_UE_NGAP_ID}")),
        "ue-info must report the AMF-assigned AMF UE NGAP ID, got {:?}",
        info.value
    );

    // 3. `ue-suspend` succeeds and REACHES RRC. This is the nextgcore #403
    //    dependency: `RrcMessage::SuspendUe` is the only production trigger for
    //    RRC_INACTIVE, and it is gated on this command not erroring.
    let suspended = exec_cli(&reg, &format!("ue-suspend {REG_UE_ID} {T380_MINUTES}")).await;
    assert_eq!(
        suspended.msg_type,
        CliMessageType::Result,
        "ue-suspend must succeed for a registered UE, not answer `UE not found`: {}",
        suspended.value
    );

    let (ue_id, t380) = await_rrc_suspend(&mut reg.rrc_rx, Duration::from_secs(5))
        .await
        .expect(
            "RrcMessage::SuspendUe must reach the real RRC inbox -- it is unreachable in \
             production while the CLI registry has no writer",
        );
    assert_eq!(
        ue_id, REG_UE_ID,
        "the suspend must target the UE that registered"
    );
    assert_eq!(
        t380,
        Some(T380_MINUTES),
        "the t380 argument must survive the CLI round trip"
    );

    for handle in reg.handles {
        handle.abort();
    }
}

/// Removal: once the AMF releases the UE, `ue-suspend` must stop accepting its id.
///
/// The half of issue #199 a write-only registry would still get wrong. A registry
/// that only ever grows answers `ue-suspend` for a released UE, and the resulting
/// `RrcMessage::SuspendUe` names a UE the RRC task no longer holds — and ue_ids are
/// reused, so the stale entry can eventually name somebody else's UE.
///
/// Driven through the AMF's real `UEContextReleaseCommand`, so the assertion covers
/// the production release path rather than a direct call to `delete_ue_context`.
/// Remove the `UeContextRemove` send from `NgapTask::delete_ue_context` and this
/// fails at the final assertion.
#[tokio::test]
async fn a_released_ue_is_no_longer_offered_to_ue_suspend() {
    let mut reg = register_a_ue().await;

    // Precondition, asserted rather than assumed: the UE is suspendable BEFORE the
    // release. Without this, a test that never saw the UE would pass for the wrong
    // reason.
    let before = exec_cli(&reg, &format!("ue-info {REG_UE_ID}")).await;
    assert_eq!(
        before.msg_type,
        CliMessageType::Result,
        "the UE must be known before the release, or this test proves nothing: {}",
        before.value
    );

    // The AMF releases the UE (TS 38.413 §8.3.3).
    let release_cmd = encode_ue_context_release_command(&UeContextReleaseCommandParams {
        ue_ngap_ids: UeNgapIds::Pair {
            amf_ue_ngap_id: ASSIGNED_AMF_UE_NGAP_ID,
            ran_ue_ngap_id: reg.wire_ran_ue_ngap_id as u32,
        },
        cause: NgSetupFailureCause::Nas(NasCause::NormalRelease),
    })
    .expect("encode UE Context Release Command");
    reg.ngap_tx
        .send(NgapMessage::ReceiveNgapPdu {
            client_id: AMF_CLIENT_ID,
            stream: NGAP_STREAM,
            pdu: OctetString::from_slice(&release_cmd),
        })
        .await
        .expect("the NGAP task accepts the release command");

    // Wait for the gNB's UE Context Release Complete, so the release has definitely
    // run rather than merely been queued. Sequencing on the gNB's own answer keeps
    // this off a sleep.
    assert!(
        await_sctp_pdu(&mut reg.sctp_rx, Duration::from_secs(5), |bytes| {
            // UEContextReleaseComplete: successfulOutcome (0x20), procedure code 41 (0x29).
            (bytes.first() == Some(&0x20) && bytes.get(1) == Some(&0x29)).then_some(())
        })
        .await
        .is_some(),
        "the gNB must answer the release command with a UE Context Release Complete"
    );

    // The registry must have dropped the UE. Retried briefly because the removal
    // travels to the App task over a channel: the assertion is about the steady
    // state, not about winning a race with the App task's next poll.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        let answer = exec_cli(&reg, &format!("ue-suspend {REG_UE_ID}")).await;
        if answer.msg_type == CliMessageType::Error {
            assert!(
                answer.value.contains("not found"),
                "the refusal must say the UE is unknown, got {:?}",
                answer.value
            );
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "a released UE must stop being offered to ue-suspend; the CLI still \
             answered {:?}",
            answer.value
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // And it is gone from `ue-list` too, not merely refused by `ue-suspend`.
    let listed = exec_cli(&reg, "ue-list").await;
    assert!(
        !listed.value.contains(&format!("ue_id: {REG_UE_ID}")),
        "a released UE must not still be listed, got {:?}",
        listed.value
    );

    for handle in reg.handles {
        handle.abort();
    }
}
