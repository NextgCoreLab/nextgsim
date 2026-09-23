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

use std::net::SocketAddr;
use std::sync::atomic::{AtomicU16, Ordering};
use std::time::Duration;

use nextgsim_common::cli_server::{
    CliMessage, CliMessageType, ProcTableEntry, PROC_TABLE_DIR, VERSION_MAJOR, VERSION_MINOR,
    VERSION_PATCH,
};
use nextgsim_common::config::GnbConfig;
use nextgsim_common::{lookup_node_port, Plmn};
use nextgsim_gnb::tasks::{
    AppMessage, GnbTaskBase, RrcMessage, Task, TaskMessage, DEFAULT_CHANNEL_CAPACITY,
};
use nextgsim_gnb::AppTask;
use tokio::net::UdpSocket;

/// The UE the suspend is aimed at. Registered in the App task's UE contexts before
/// the task is spawned, because `ue-suspend` for an unknown UE is refused before any
/// RRC message is produced (`GnbCmdHandler::handle_ue_suspend`).
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
fn spawn_app_task() -> (
    AppTask,
    tokio::sync::mpsc::Receiver<TaskMessage<AppMessage>>,
    tokio::sync::mpsc::Receiver<TaskMessage<RrcMessage>>,
) {
    let (base, app_rx, _ngap_rx, rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
        GnbTaskBase::new(test_config(), DEFAULT_CHANNEL_CAPACITY);

    let mut task = AppTask::new(base);
    task.update_ue_context(UE_ID, RAN_UE_NGAP_ID, Some(AMF_UE_NGAP_ID));

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

    let (mut task, _app_rx, _rrc_rx) = spawn_app_task();
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

    let (mut task, app_rx, mut rrc_rx) = spawn_app_task();
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

    let (mut task, app_rx, mut rrc_rx) = spawn_app_task();
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
