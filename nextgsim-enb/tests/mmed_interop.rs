//! Cross-repo interop: the nextgsim eNB and LTE UE against nextgcore's real
//! `mmed`, and -- for the attach -- its real HSS, SGW and PGW.
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
//! and their absence is not fatal, so no HSS or SGW is needed. The attach test
//! ([`lte_ue_attaches_through_the_nextgcore_epc`]) needs the rest of the EPC
//! running on loopback first -- `tests/epc/bring-up.sh` starts MongoDB (with the
//! test subscriber), hssd, sgwcd, sgwud, smfd as PGW-C and upfd as PGW-U at the
//! addresses this file uses, and `NEXTGCORE_EPC=1` enables the test.
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
use nextgsim_eps_nas::EmmState;
use nextgsim_lte_ue::config::{LteUeConfig, SubscriberConfig};
use nextgsim_lte_ue::nas::{EpsNas, EpsNasStatus};
use nextgsim_lte_ue::rrc::UeRrcState;

/// The loopback addresses `tests/epc/bring-up.sh` puts the EPC on.
const MME_ADDR: Ipv4Addr = Ipv4Addr::new(127, 0, 0, 5);
const SGWC_ADDR: &str = "127.0.0.3";
const HSS_ADDR: &str = "127.0.0.8";

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
    start_mmed_with(dir, s1ap_port, false)
}

/// Start mmed. With `epc` it binds the MME address of `tests/epc/bring-up.sh`,
/// reaches the HSS over S6a (Diameter over TCP, `No_SCTP`) and the SGW-C over
/// S11; without, it runs standalone on 127.0.0.1.
fn start_mmed_with(dir: &std::path::Path, s1ap_port: u16, epc: bool) -> Mmed {
    let bin = std::env::var("NEXTGCORE_MMED")
        .expect("set NEXTGCORE_MMED to an mmed built with --features kernel-sctp");
    let (addr, gtpc_port) = if epc {
        (MME_ADDR.to_string(), 2123)
    } else {
        ("127.0.0.1".to_string(), free_port())
    };
    let peers = if epc {
        let fd = dir.join("mme-fd.conf");
        std::fs::write(
            &fd,
            format!(
                "Identity = \"mme.localdomain\";\nRealm = \"localdomain\";\nNo_IPv6;\nNo_SCTP;\n\
                 ListenOn = \"{addr}\";\nNoRelay;\n\
                 ConnectPeer = \"hss.localdomain\" {{ ConnectTo = \"{HSS_ADDR}\"; No_TLS; }};\n"
            ),
        )
        .unwrap();
        format!(
            "  freeDiameter: {}\n  metrics:\n    server:\n      - address: {addr}\n        port: 19095\n",
            fd.display()
        )
    } else {
        String::new()
    };
    let client = if epc {
        format!("    client:\n      sgwc:\n        - address: {SGWC_ADDR}\n")
    } else {
        String::new()
    };
    let cfg = format!(
        r#"logger:
  file:
    path: {log}
  level: info
mme:
{peers}  s1ap:
    server:
      - address: {addr}
        port: {s1ap_port}
  gtpc:
{client}    server:
      - address: {addr}
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
        .env("MME_RESTART_COUNTER_FILE", dir.join("mme-restart-counter"))
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
        security: Default::default(),
        // S1-U on its own loopback address: port 2152 on 127.0.0.1 belongs to
        // whatever else runs GTP-U on this host (the EPC's SGW-U is 127.0.0.6).
        gtp_ip: Some(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 10))),
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

fn subscriber() -> SubscriberConfig {
    // The test subscriber of nextgcore's k8s/manifests/mongodb-init.yaml, which
    // tests/epc/bring-up.sh provisions.
    SubscriberConfig {
        imsi: "999700000000001".into(),
        key: [
            0x46, 0x5B, 0x5C, 0xE8, 0xB1, 0x99, 0xB4, 0x9F, 0xAA, 0x5F, 0x0A, 0x2E, 0xE2, 0x38,
            0xA6, 0xBC,
        ],
        op: [
            0xE8, 0xED, 0x28, 0x9D, 0xEB, 0xA9, 0x52, 0xE4, 0x28, 0x3B, 0x54, 0xE8, 0x8E, 0x6D,
            0x83, 0x4D,
        ],
        op_type: nextgsim_common::OpType::Opc,
        sqn: 0,
        imei_sv: Some("3534900698733019".into()),
        eea: vec![0, 1, 2, 3],
        eia: vec![1, 2, 3],
        apn: std::env::var("NEXTGSIM_TEST_APN").ok(),
        pdn_type: Default::default(),
    }
}

/// eNB with S1 set up against `mmed` on `remote`, and the LTE UE with its EPS
/// NAS camped on it. Returns the UE's RRC state and NAS status.
/// The UE's IP seam: packets into its uplink, and out of its downlink.
type IpEnds = (
    tokio::sync::mpsc::Sender<Vec<u8>>,
    tokio::sync::mpsc::Receiver<Vec<u8>>,
);

async fn enb_and_ue(
    cfg: EnbConfig,
    remote: SocketAddr,
) -> (
    tokio::sync::watch::Receiver<UeRrcState>,
    std::sync::Arc<std::sync::Mutex<EpsNasStatus>>,
    tokio::sync::watch::Sender<bool>,
    IpEnds,
) {
    use nextgsim_enb::radio::{run_operational, RlsRadio, S1uEndpoint};
    let (_mme, mut transport) = tokio::time::timeout(
        Duration::from_secs(30),
        establish_s1(&cfg, &mut KernelConnector::default(), remote),
    )
    .await
    .expect("S1 Setup did not complete against mmed");
    let mut radio = RlsRadio::bind(&cfg).await.unwrap();
    let enb_addr = radio.local_addr().unwrap();
    let mut s1u = S1uEndpoint::bind(&cfg, None).await.unwrap();
    let enb_cfg = cfg.clone();
    tokio::spawn(
        async move { run_operational(&enb_cfg, &mut radio, &mut s1u, &mut transport).await },
    );

    let ue_cfg = LteUeConfig {
        plmn: Plmn::new(999, 70, false),
        enb_search_list: vec![enb_addr.to_string()],
        sti: Some(0xABCD),
        position: [0, 0, 0],
        subscriber: Some(subscriber()),
        tun: None,
    };
    nextgsim_lte_ue::config::validate(&ue_cfg).unwrap();
    let nas = EpsNas::new(ue_cfg.eps_nas().unwrap());
    let status = nas.status();
    let socket = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let (state_tx, state_rx) = tokio::sync::watch::channel(UeRrcState::NoCell);
    let (stop_tx, stop_rx) = tokio::sync::watch::channel(false);
    let (ip, ip_ul, ip_dl) = nextgsim_lte_ue::task::IpIo::channel(64);
    tokio::spawn(async move {
        nextgsim_lte_ue::task::run(&ue_cfg, Box::new(nas), socket, state_tx, stop_rx, ip).await
    });
    (state_rx, status, stop_tx, (ip_ul, ip_dl))
}

async fn wait_nas(
    status: &std::sync::Arc<std::sync::Mutex<EpsNasStatus>>,
    within: Duration,
    f: impl Fn(&EpsNasStatus) -> bool,
) -> EpsNasStatus {
    let deadline = tokio::time::Instant::now() + within;
    loop {
        let s = status.lock().unwrap().clone();
        if f(&s) {
            return s;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "NAS did not reach the expected state; last {s:?}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// With no HSS, mmed's S6a request fails and the UE's own ATTACH REQUEST is
/// answered with ATTACH REJECT #17, which the UE's EMM treats as abnormal case
/// §5.5.1.2.6 d) -- attempt counter, T3411 -- while mmed releases the S1 UE
/// context (nextgcore #472/#473), returning the UE to RRC_IDLE.
// Multi-threaded: `Mmed::wait_for` blocks on a std channel, which on the
// default current-thread runtime would stop the eNB and UE tasks it waits on.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs NEXTGCORE_MMED and kernel SCTP"]
async fn lte_ue_attach_without_an_hss_is_rejected_and_retried() {
    init_logging();
    let port = free_port();
    let mut mmed = start_mmed(&tmpdir("attach-nohss"), port);
    let mut cfg = enb_config(port, Plmn::new(999, 70, false));
    cfg.radio.rls_port = 0;
    let remote = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port);
    let (mut state_rx, status, _stop, _ip) = enb_and_ue(cfg, remote).await;
    assert!(
        mmed.wait_for(
            "Attach Request: IMSI[999700000000001] attach_type[1] on enb_ue_s1ap_id=",
            Duration::from_secs(15)
        ),
        "mmed did not decode the UE's ATTACH REQUEST"
    );
    wait_nas(&status, Duration::from_secs(15), |s| {
        s.state == EmmState::AttemptingToAttach
    })
    .await;
    tokio::time::timeout(Duration::from_secs(15), async {
        while !matches!(*state_rx.borrow_and_update(), UeRrcState::Camped { .. }) {
            state_rx.changed().await.unwrap();
        }
    })
    .await
    .expect("mmed's release after ATTACH REJECT did not return the UE to RRC_IDLE");
}

/// nextgsim #186 increment 3's exit criterion: the UE attaches through the
/// real EPC -- EPS AKA against hssd, NAS SMC, ULR, Create Session through
/// SGW-C to the PGW-C, INITIAL CONTEXT SETUP with the AS Security Mode Command,
/// ATTACH ACCEPT in RRCConnectionReconfiguration -- and is EMM-REGISTERED with
/// the default EPS bearer activated.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs NEXTGCORE_MMED, kernel SCTP and the EPC of tests/epc/bring-up.sh"]
async fn lte_ue_attaches_through_the_nextgcore_epc() {
    assert!(
        std::env::var("NEXTGCORE_EPC").is_ok(),
        "start the EPC with tests/epc/bring-up.sh and set NEXTGCORE_EPC=1"
    );
    init_logging();
    let port = 36412;
    let mut mmed = start_mmed_with(&tmpdir("epc"), port, true);
    let mut cfg = enb_config(port, Plmn::new(999, 70, false));
    cfg.mme_configs[0].address = IpAddr::V4(MME_ADDR);
    cfg.radio.rls_port = 0;
    let remote = SocketAddr::new(IpAddr::V4(MME_ADDR), port);
    let (_state_rx, status, _stop, _ip) = enb_and_ue(cfg, remote).await;
    let s = wait_nas(&status, Duration::from_secs(40), |s| {
        s.state == EmmState::Registered || matches!(s.state, EmmState::NoService(_))
    })
    .await;
    assert_eq!(s.state, EmmState::Registered, "attach did not complete");
    let bearer = s.bearer.expect("default EPS bearer activated");
    eprintln!("[test] default EPS bearer {bearer:?}");
    assert!(
        mmed.wait_for("Attach Complete", Duration::from_secs(5))
            || mmed.wait_for("ATTACH COMPLETE", Duration::from_secs(1))
            || mmed.wait_for("attach complete", Duration::from_secs(1)),
        "mmed did not log the ATTACH COMPLETE"
    );
}

/// RFC 1071 Internet checksum.
fn inet_checksum(b: &[u8]) -> u16 {
    let mut sum: u32 = b
        .chunks(2)
        .map(|c| u32::from(u16::from_be_bytes([c[0], *c.get(1).unwrap_or(&0)])))
        .sum();
    while sum >> 16 != 0 {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }
    !(sum as u16)
}

/// An IPv4 ICMP Echo Request (RFC 792) from `src` to `dst`.
fn icmp_echo(src: Ipv4Addr, dst: Ipv4Addr, id: u16, seq: u16) -> Vec<u8> {
    let mut icmp = vec![8, 0, 0, 0];
    icmp.extend_from_slice(&id.to_be_bytes());
    icmp.extend_from_slice(&seq.to_be_bytes());
    icmp.extend_from_slice(b"nextgsim-lte-ping");
    let c = inet_checksum(&icmp);
    icmp[2..4].copy_from_slice(&c.to_be_bytes());
    let len = (20 + icmp.len()) as u16;
    let mut ip = vec![0x45, 0];
    ip.extend_from_slice(&len.to_be_bytes());
    ip.extend_from_slice(&[0, 0, 0x40, 0, 64, 1, 0, 0]);
    ip.extend_from_slice(&src.octets());
    ip.extend_from_slice(&dst.octets());
    let c = inet_checksum(&ip);
    ip[10..12].copy_from_slice(&c.to_be_bytes());
    ip.extend_from_slice(&icmp);
    ip
}

/// #186 increment 4's exit criterion: after the attach, ICMP echo through the
/// default bearer -- UE PDCP/RLC over RLS, the eNB's S1-U, nextgcore's SGW-U
/// and PGW-U -- with 0% loss. Needs everything the attach test needs, plus a
/// PGW-U with its SGi data plane up and a host that answers `NEXTGSIM_PING_TARGET`
/// (default 10.45.0.1, the PGW's gateway address): `bring-up.sh` starts upfd
/// without a data plane, because its N6 interface needs CAP_NET_ADMIN.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs the attach test's EPC plus a PGW-U SGi data plane (CAP_NET_ADMIN)"]
async fn lte_ue_pings_through_the_nextgcore_epc_with_no_loss() {
    assert!(
        std::env::var("NEXTGCORE_EPC_DATAPLANE").is_ok(),
        "start the EPC with a PGW-U data plane and set NEXTGCORE_EPC_DATAPLANE=1"
    );
    init_logging();
    let port = 36412;
    let _mmed = start_mmed_with(&tmpdir("epc-ping"), port, true);
    let mut cfg = enb_config(port, Plmn::new(999, 70, false));
    cfg.mme_configs[0].address = IpAddr::V4(MME_ADDR);
    cfg.radio.rls_port = 0;
    let remote = SocketAddr::new(IpAddr::V4(MME_ADDR), port);
    let (_state_rx, status, _stop, (ul, mut dl)) = enb_and_ue(cfg, remote).await;
    let s = wait_nas(&status, Duration::from_secs(40), |s| s.bearer.is_some()).await;
    let bearer = s.bearer.unwrap();
    let ue_ip = match bearer.pdn_address {
        nextgsim_eps_nas::ie::PdnAddress::Ipv4(a)
        | nextgsim_eps_nas::ie::PdnAddress::Ipv4v6(_, a) => a,
        other => panic!("no IPv4 PDN address: {other:?}"),
    };
    let target: Ipv4Addr = std::env::var("NEXTGSIM_PING_TARGET")
        .ok()
        .and_then(|t| t.parse().ok())
        .unwrap_or(Ipv4Addr::new(10, 45, 0, 1));
    let count = 20u16;
    for seq in 0..count {
        ul.send(icmp_echo(ue_ip, target, 0x186, seq)).await.unwrap();
        let reply = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let p = dl.recv().await.expect("UE downlink closed");
                // Echo Reply (type 0) for this id/seq, to the UE's address.
                if p.len() >= 28
                    && p[9] == 1
                    && p[16..20] == ue_ip.octets()
                    && p[20] == 0
                    && p[24..26] == 0x186u16.to_be_bytes()
                    && p[26..28] == seq.to_be_bytes()
                {
                    return p;
                }
            }
        })
        .await;
        assert!(
            reply.is_ok(),
            "echo {seq} of {count} lost: the loss must be 0%"
        );
    }
}
