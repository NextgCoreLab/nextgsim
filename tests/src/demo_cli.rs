//! `nr-cli <node> demo ...` on a running gNB and UE (nextgsim#461, the toggle
//! convention of #458).
//!
//! Drives the real App tasks through the real CLI server, with commands framed
//! exactly as `nr-cli` frames them, and asserts the three paths the issue names:
//! `show`, `set mode=<m>` (applies to the next read), and a bogus mode (an
//! `Error` answer, an audit line, and the mode unchanged).
//!
//! A default build registers no demo, so the test installs a registry of its own,
//! standing in for a binary built with a demo's cargo feature. The demo state is
//! process-global and write-once, so this file is its own test binary with a
//! single test, and the gNB and UE share the one state, as they would share it
//! in one process.

use std::io::Write;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use nextgsim_common::cli_server::{CliMessage, CliMessageType};
use nextgsim_common::config::{GnbConfig, UeConfig};
use nextgsim_common::Plmn;
use nextgsim_demo::{DemoRegistry, DemoRuntime, DemoSpec};
use nextgsim_gnb::tasks::{GnbTaskBase, Task as _, DEFAULT_CHANNEL_CAPACITY};
use nextgsim_gnb::AppTask as GnbAppTask;
use nextgsim_ue::tasks::Task as _;
use nextgsim_ue::{AppTask as UeAppTask, UeTaskBase};
use tokio::net::UdpSocket;

const KEY: &str = "cli-selftest";
const SPEC: DemoSpec = DemoSpec {
    key: KEY,
    modes: &["baseline", "treated"],
    requires: &[],
};

#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<u8>>>);

impl Write for Capture {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Capture {
    fn text(&self) -> String {
        String::from_utf8(self.0.lock().unwrap().clone()).unwrap()
    }
}

/// Sends `command` to `node` at `port`, framed as `nr-cli` frames it, and
/// returns the answer's type and text.
async fn nr_cli(port: u16, node: &str, command: &str) -> (CliMessageType, String) {
    let client = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let target: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
    let msg = CliMessage {
        msg_type: CliMessageType::Command,
        node_name: node.to_string(),
        value: command.to_string(),
        client_addr: client.local_addr().unwrap(),
    };
    client.send_to(&msg.encode(), target).await.unwrap();
    let mut buf = [0u8; 8192];
    let (n, _) = tokio::time::timeout(Duration::from_secs(5), client.recv_from(&mut buf))
        .await
        .unwrap_or_else(|_| panic!("{node} must answer {command:?}"))
        .unwrap();
    let r = CliMessage::decode(&buf[..n], target).expect("the answer decodes");
    (r.msg_type, r.value)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn nr_cli_demo_show_set_and_a_bogus_mode_on_a_gnb_and_a_ue() {
    // The App tasks run on worker threads, so the audit lines are captured with
    // a process-wide subscriber (this binary has no other test).
    let cap = Capture::default();
    let writer = cap.clone();
    tracing_subscriber::fmt()
        .with_writer(move || writer.clone())
        .with_ansi(false)
        .without_time()
        .with_target(false)
        .init();

    let config: serde_yaml::Value = serde_yaml::from_str(&format!(
        "demo:\n  enabled: true\n  {KEY}:\n    enabled: true\n    mode: baseline\n"
    ))
    .unwrap();
    let runtime = DemoRuntime::from_root(&config, DemoRegistry::new(&[SPEC])).unwrap();
    nextgsim_demo::install(runtime).expect("first install");
    assert!(cap
        .text()
        .contains(&format!("WARN DEMO MODE ACTIVE: {KEY}=baseline")));

    // gNB.
    let gnb_cfg = GnbConfig {
        nci: 0x10,
        gnb_id_length: 32,
        plmn: Plmn::new(310, 410, false),
        tac: 1,
        ..Default::default()
    };
    let (base, app_rx, ..) = GnbTaskBase::new(gnb_cfg, DEFAULT_CHANNEL_CAPACITY);
    let mut gnb = GnbAppTask::new(base);
    let gnb_node = "gnb-demo-cli";
    let gnb_port = gnb.init_cli_server(gnb_node.into()).await.unwrap();
    let gnb_handle = tokio::spawn(async move { gnb.run(app_rx).await });

    let show = format!("demo {KEY} show");
    assert_eq!(
        nr_cli(gnb_port, gnb_node, &show).await,
        (
            CliMessageType::Result,
            format!("demo={KEY} active=true mode=baseline modes=baseline,treated requires=-")
        )
    );
    assert_eq!(
        nr_cli(gnb_port, gnb_node, &format!("demo {KEY} set mode=treated")).await,
        (
            CliMessageType::Result,
            format!("demo={KEY} mode=treated (was baseline; applies to the next decision)")
        )
    );
    // The next decision reads the new mode.
    assert_eq!(nextgsim_demo::active_mode(KEY).as_deref(), Some("treated"));

    let (kind, text) = nr_cli(gnb_port, gnb_node, &format!("demo {KEY} set mode=bogus")).await;
    assert_eq!(kind, CliMessageType::Error, "{text}");
    assert!(text.contains("unknown mode \"bogus\""), "{text}");
    assert_eq!(nextgsim_demo::active_mode(KEY).as_deref(), Some("treated"));
    assert!(nr_cli(gnb_port, gnb_node, &show)
        .await
        .1
        .contains("mode=treated"));

    let (kind, text) = nr_cli(gnb_port, gnb_node, "demo metrics").await;
    assert_eq!(kind, CliMessageType::Result);
    assert!(text.contains(&format!(
        "nextg_demo_enabled{{demo=\"{KEY}\",mode=\"treated\"}} 1"
    )));

    // A non-demo command still reaches the gNB command set.
    let (kind, text) = nr_cli(gnb_port, gnb_node, "status").await;
    assert_eq!(kind, CliMessageType::Result, "{text}");
    assert!(!text.contains("demo"), "{text}");
    gnb_handle.abort();

    // UE: the same process state, through the UE's own CLI path.
    let (ue_base, ue_app_rx, ..) = UeTaskBase::new(UeConfig::default(), 64);
    let mut ue = UeAppTask::new(ue_base);
    let ue_node = "imsi-001010000099461";
    let ue_port = ue.init_cli_server(vec![ue_node.into()]).await.unwrap();
    let ue_handle = tokio::spawn(async move { ue.run(ue_app_rx).await });
    assert!(nr_cli(ue_port, ue_node, &show)
        .await
        .1
        .contains("mode=treated"));
    assert_eq!(
        nr_cli(ue_port, ue_node, &format!("demo {KEY} set mode=baseline")).await,
        (
            CliMessageType::Result,
            format!("demo={KEY} mode=baseline (was treated; applies to the next decision)")
        )
    );
    let (kind, text) = nr_cli(ue_port, ue_node, &format!("demo {KEY} set mode=nope")).await;
    assert_eq!(kind, CliMessageType::Error, "{text}");
    assert_eq!(nextgsim_demo::active_mode(KEY).as_deref(), Some("baseline"));
    ue_handle.abort();

    // Every set was audited, the rejected ones included.
    let log = cap.text();
    for want in [
        format!("WARN [demo:{KEY}] audit: mode baseline -> treated from nr-cli"),
        format!("WARN [demo:{KEY}] audit: rejected set mode=\"bogus\" from nr-cli"),
        "mode unchanged (treated)".to_string(),
        format!("WARN [demo:{KEY}] audit: mode treated -> baseline from nr-cli"),
        format!("WARN [demo:{KEY}] audit: rejected set mode=\"nope\" from nr-cli"),
    ] {
        assert!(log.contains(&want), "missing {want:?} in:\n{log}");
    }
}
