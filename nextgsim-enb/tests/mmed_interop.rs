//! Cross-repo interop: the nextgsim eNB against nextgcore's real `mmed`.
//!
//! Needs Linux kernel SCTP, the `kernel-sctp` feature, and an mmed binary built
//! with nextgcore's own `kernel-sctp` feature (the only S1-MME transport mmed
//! has). Point `NEXTGCORE_MMED` at it:
//!
//! ```bash
//! (cd nextgcore/src && cargo build -p nextgcore-mmed --features kernel-sctp)
//! NEXTGCORE_MMED=nextgcore/src/target/debug/nextgcore-mmed \
//!   cargo test -p nextgsim-enb --features kernel-sctp --test mmed_interop -- --ignored
//! ```
//!
//! mmed starts standalone for S1 Setup: S6a and S11 peers are dialled lazily
//! and their absence is not fatal, so no HSS or SGW is needed.
//!
//! Before nextgcore `5879548` (nextgcore #455, PR #456) every test here
//! failed, correctly: mmed's hand-written S1AP omitted the extension bit of
//! every extensible message SEQUENCE (X.691 §19.1), so it read this eNB's
//! conformant S1 SETUP REQUEST as zero IEs. That is the defect an independent
//! codec exists to catch (#45, #205). Run against `5879548` or later.
#![cfg(all(target_os = "linux", feature = "kernel-sctp"))]

use std::io::{BufRead, BufReader};
use std::net::{IpAddr, Ipv4Addr, SocketAddr, UdpSocket};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

use nextgsim_common::Plmn;
use nextgsim_enb::config::{EnbConfig, EnbIdType, MmeConfig, PagingDrx};
use nextgsim_enb::s1_task::{
    establish_s1, kernel::KernelConnector, run_s1_setup_attempt, AttemptResult, S1Connector,
};

struct Mmed {
    child: Child,
    lines: mpsc::Receiver<String>,
    seen: Vec<String>,
}

impl Mmed {
    /// Wait until mmed logs a line containing `needle`.
    fn wait_for(&mut self, needle: &str, within: Duration) -> bool {
        if self.seen.iter().any(|l| l.contains(needle)) {
            return true;
        }
        let deadline = std::time::Instant::now() + within;
        while let Some(left) = deadline.checked_duration_since(std::time::Instant::now()) {
            match self.lines.recv_timeout(left) {
                Ok(l) => {
                    let hit = l.contains(needle);
                    self.seen.push(l);
                    if hit {
                        return true;
                    }
                }
                Err(_) => break,
            }
        }
        false
    }
}

impl Drop for Mmed {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// A port no one is using right now, for both S1-MME and S11.
fn free_port() -> u16 {
    UdpSocket::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn start_mmed(dir: &std::path::Path, s1ap_port: u16) -> Mmed {
    let bin = std::env::var("NEXTGCORE_MMED")
        .expect("set NEXTGCORE_MMED to an mmed built with --features kernel-sctp");
    let gtpc_port = free_port();
    let cfg = format!(
        r#"logger:
  file:
    path: {log}
  level: info
mme:
  s1ap:
    server:
      - address: 127.0.0.1
        port: {s1ap_port}
  gtpc:
    server:
      - address: 127.0.0.1
        port: {gtpc_port}
  gummei:
    - plmn_id: {{ mcc: 999, mnc: 70 }}
      mme_gid: 2
      mme_code: 1
  tai:
    - plmn_id: {{ mcc: 999, mnc: 70 }}
      tac: 1
  security:
    integrity_order: [EIA2, EIA1, EIA0]
    ciphering_order: [EEA0, EEA1, EEA2]
  mme_name: nextgcore-mme0
"#,
        log = dir.join("mme.log").display()
    );
    let path = dir.join("mme.yaml");
    std::fs::write(&path, cfg).unwrap();
    let mut child = Command::new(bin)
        .arg("-c")
        .arg(&path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn mmed");
    let (tx, rx) = mpsc::channel();
    for stream in [
        Box::new(child.stdout.take().unwrap()) as Box<dyn std::io::Read + Send>,
        Box::new(child.stderr.take().unwrap()),
    ] {
        let tx = tx.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(stream).lines().map_while(Result::ok) {
                eprintln!("[mmed] {line}");
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
    }
    let mut m = Mmed {
        child,
        lines: rx,
        seen: vec![],
    };
    assert!(
        m.wait_for("S1-MME interface ready", Duration::from_secs(20)),
        "mmed did not bind S1-MME (built without kernel-sctp?)"
    );
    m
}

fn enb_config(port: u16, plmn: Plmn) -> EnbConfig {
    EnbConfig {
        enb_id: 0x19B,
        enb_id_type: EnbIdType::Macro,
        plmn,
        tac: 1,
        supported_tas: vec![],
        enb_name: Some("nextgsim-enb".into()),
        paging_drx: PagingDrx::V128,
        mme_configs: vec![MmeConfig {
            address: IpAddr::V4(Ipv4Addr::LOCALHOST),
            port,
        }],
        s1ap_ip: IpAddr::V4(Ipv4Addr::LOCALHOST),
        s1_setup_retry_secs: 1,
        s1_setup_response_timeout_ms: 5000,
        radio: Default::default(),
    }
}

/// Route this process's `log` output (eNB and UE) to stderr, filtered by
/// `RUST_LOG` (default `info`), so a failing run shows both sides.
fn init_logging() {
    let _ = tracing_log::LogTracer::init();
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();
}

fn tmpdir(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("nextgsim-enb-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[tokio::test]
#[ignore = "needs NEXTGCORE_MMED and kernel SCTP"]
async fn s1_setup_completes_against_nextgcore_mmed() {
    let port = free_port();
    let mut mmed = start_mmed(&tmpdir("ok"), port);
    let cfg = enb_config(port, Plmn::new(999, 70, false));
    let remote = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port);

    let (mme, _assoc) = tokio::time::timeout(
        Duration::from_secs(30),
        establish_s1(&cfg, &mut KernelConnector::default(), remote),
    )
    .await
    .expect("S1 Setup did not complete against mmed");

    // What mmed's config says it serves, as received over the wire.
    assert_eq!(mme.mme_name.as_deref(), Some("nextgcore-mme0"));
    assert_eq!(mme.served_gummeis.len(), 1);
    assert_eq!(mme.served_gummeis[0].plmns, vec![Plmn::new(999, 70, false)]);
    assert_eq!(mme.served_gummeis[0].group_ids, vec![2]);
    assert_eq!(mme.served_gummeis[0].mme_codes, vec![1]);
    assert!(mme.relative_capacity.is_some());

    // And mmed's own record of it: the eNB is registered under its ID and name.
    assert!(
        mmed.wait_for(
            "S1 Setup: eNB 0x19b (nextgsim-enb) accepted",
            Duration::from_secs(5)
        ),
        "mmed did not log the eNB as accepted"
    );
}

#[tokio::test]
#[ignore = "needs NEXTGCORE_MMED and kernel SCTP"]
async fn an_unserved_plmn_gets_s1_setup_failure_with_time_to_wait() {
    // TS 36.413 §8.7.3.4: no PLMN the MME identifies -> reject, e.g. Unknown PLMN.
    let port = free_port();
    let mut mmed = start_mmed(&tmpdir("fail"), port);
    let cfg = enb_config(port, Plmn::new(1, 1, false));
    let remote = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port);
    let local = SocketAddr::new(cfg.s1ap_ip, 0);

    let mut transport = KernelConnector::default()
        .connect(local, remote)
        .await
        .expect("SCTP association with mmed");
    let r = run_s1_setup_attempt(&cfg, &mut transport).await;
    // mmed answers Unknown PLMN with Time To Wait v10s; §8.7.3.3 has the eNB
    // wait at least that long.
    assert_eq!(r, AttemptResult::RetryAfter(Duration::from_secs(10)));
    assert!(mmed.wait_for("no served TAI matches", Duration::from_secs(5)));
}

/// TS 24.301 §8.2.4 ATTACH REQUEST, plain NAS: IMSI 999700000000001, EEA0-2 /
/// EIA0-2, PDN CONNECTIVITY REQUEST (IPv4) in the ESM container. A TEST INPUT
/// only -- the product UE has no EMM until nextgsim #186 increment 3.
const ATTACH_REQUEST: &[u8] = &[
    0x07, 0x41, 0x71, 0x08, 0x99, 0x99, 0x07, 0x00, 0x00, 0x00, 0x00, 0x10, 0x02, 0xE0, 0xE0, 0x00,
    0x04, 0x02, 0x01, 0xD0, 0x11,
];

struct AttachOnce {
    asked: bool,
    downlink: std::sync::Arc<std::sync::Mutex<Vec<Vec<u8>>>>,
}

impl nextgsim_lte_ue::nas::NasLayer for AttachOnce {
    fn initial_nas(&mut self) -> Option<nextgsim_lte_ue::nas::InitialNas> {
        if std::mem::replace(&mut self.asked, true) {
            return None;
        }
        Some(nextgsim_lte_ue::nas::InitialNas {
            nas: ATTACH_REQUEST.to_vec(),
            cause: nextgsim_lte_rrc::msg::EstablishmentCauseValue::MoSignalling,
            s_tmsi: None,
        })
    }
    fn on_downlink_nas(&mut self, nas: &[u8]) -> Vec<Vec<u8>> {
        self.downlink.lock().unwrap().push(nas.to_vec());
        Vec::new()
    }
    fn on_rrc_indication(&mut self, _ind: nextgsim_lte_ue::nas::RrcIndication) {}
}

/// nextgsim #186 increment 2's exit criterion: the LTE UE's NAS, carried over
/// E-UTRA RRC and SRB1 PDCP on RLS, reaches mmed in an S1AP INITIAL UE MESSAGE
/// that mmed decodes.
// Multi-threaded: `Mmed::wait_for` blocks on a std channel, which on the
// default current-thread runtime would stop the eNB and UE tasks it waits on.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs NEXTGCORE_MMED and kernel SCTP"]
async fn lte_ue_attach_request_reaches_nextgcore_mmed() {
    use nextgsim_enb::radio::{run_operational, RlsRadio};
    use nextgsim_lte_ue::rrc::UeRrcState;

    init_logging();
    let port = free_port();
    let mut mmed = start_mmed(&tmpdir("attach"), port);
    let mut cfg = enb_config(port, Plmn::new(999, 70, false));
    cfg.radio.rls_port = 0;
    let remote = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port);
    let (_mme, mut transport) = tokio::time::timeout(
        Duration::from_secs(30),
        establish_s1(&cfg, &mut KernelConnector::default(), remote),
    )
    .await
    .expect("S1 Setup did not complete against mmed");
    let mut radio = RlsRadio::bind(&cfg).await.unwrap();
    let enb_addr = radio.local_addr().unwrap();
    let enb_cfg = cfg.clone();
    tokio::spawn(async move { run_operational(&enb_cfg, &mut radio, &mut transport).await });

    let downlink = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let ue_cfg = nextgsim_lte_ue::config::LteUeConfig {
        plmn: Plmn::new(999, 70, false),
        enb_search_list: vec![enb_addr.to_string()],
        sti: Some(0xABCD),
        position: [0, 0, 0],
    };
    let socket = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let (state_tx, mut state_rx) = tokio::sync::watch::channel(UeRrcState::NoCell);
    let (_stop_tx, stop_rx) = tokio::sync::watch::channel(false);
    let nas = Box::new(AttachOnce {
        asked: false,
        downlink: std::sync::Arc::clone(&downlink),
    });
    tokio::spawn(async move {
        nextgsim_lte_ue::task::run(&ue_cfg, nas, socket, state_tx, stop_rx).await
    });

    tokio::time::timeout(Duration::from_secs(15), async {
        while !matches!(*state_rx.borrow_and_update(), UeRrcState::Connected { .. }) {
            state_rx.changed().await.unwrap();
        }
    })
    .await
    .expect("UE did not reach RRC_CONNECTED");
    // mmed decoded INITIAL UE MESSAGE and the ATTACH REQUEST inside it: it
    // names the IMSI from the EPS mobile identity and the eNB UE S1AP ID.
    assert!(
        mmed.wait_for(
            "Attach Request: IMSI[999700000000001] attach_type[1] on enb_ue_s1ap_id=",
            Duration::from_secs(10)
        ),
        "mmed did not decode the ATTACH REQUEST from the INITIAL UE MESSAGE"
    );
    // And mmed's answer travels the whole downlink path back to the UE's NAS:
    // DOWNLINK NAS TRANSPORT, DLInformationTransfer, SRB1 PDCP. With no HSS
    // configured mmed's S6a AIR fails, so the answer it sends is ATTACH REJECT
    // (07 44, EMM cause #17 "network failure"); with an HSS it would be an
    // AUTHENTICATION REQUEST. Either way it is a plain EMM message (PD 7).
    let got = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            if let Some(first) = downlink.lock().unwrap().first().cloned() {
                return first;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .expect("no downlink NAS reached the UE");
    eprintln!("[test] downlink NAS delivered to the UE: {got:02x?}");
    assert_eq!(
        got.first(),
        Some(&0x07),
        "plain EPS mobility management message"
    );
}
