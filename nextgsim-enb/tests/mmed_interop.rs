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
//! As of nextgcore main `7805754` both tests FAIL, and correctly so: mmed's
//! hand-written S1AP omits the extension bit of every extensible message
//! SEQUENCE (X.691 §19.1; `S1SetupRequest ::= SEQUENCE { protocolIEs ..., ... }`),
//! so it reads this eNB's conformant request as zero IEs and logs "Missing
//! mandatory IE: Global-ENB-ID". nextgcore's NGAP codec carries that bit
//! (`encode_message_value` in nextgcore-asn1c's `ngap/pdu.rs`); its S1AP
//! `pdu.rs` does not. With the bit added, both tests pass. This is the defect an
//! independent codec exists to catch (#45, #205).
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
    }
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
