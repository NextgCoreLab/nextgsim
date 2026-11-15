//! nextgsim#471 (demo d443a-edge-budget): the sidecar's reflector, job
//! listener and EDGE-3 registration, and the UE's EEC client, against a stub
//! eesd over h2c and real loopback sockets.

use std::net::{IpAddr, SocketAddr};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use nextgsim_common::config::EecClientConfig;
use nextgsim_sbi_client::{SbiClient, SbiClientConfig};
use serde_json::{json, Value};

use super::eas::{Check, Registrar};
use super::eel::{self, EasAd};
use super::stub_ees::{EecReply, StubEes};
use super::*;

// ---- STAMP ---------------------------------------------------------------

/// upfd's Session-Sender test packet (`n6_delay.rs` `stamp_test_packet`):
/// Sequence Number, Timestamp T1, Error Estimate 0x0001, zeros; 44 octets.
fn upfd_probe(seq: u32, t1: u64) -> [u8; stamp::STAMP_LEN] {
    let mut p = [0u8; stamp::STAMP_LEN];
    p[0..4].copy_from_slice(&seq.to_be_bytes());
    p[4..12].copy_from_slice(&t1.to_be_bytes());
    p[12..14].copy_from_slice(&0x0001u16.to_be_bytes());
    p
}

/// upfd's acceptance of a reflection (`n6_delay.rs` `on_reply`): at least 44
/// octets, sender Sequence Number at octet 24 and sender Timestamp at 28
/// matching the probe; the turnaround is T3 (octet 4) - T2 (octet 16).
fn upfd_accepts(probe: &[u8], reply: &[u8]) -> Option<u64> {
    let u32_at = |p: &[u8], at: usize| u32::from_be_bytes(p[at..at + 4].try_into().unwrap());
    let u64_at = |p: &[u8], at: usize| u64::from_be_bytes(p[at..at + 8].try_into().unwrap());
    (reply.len() >= stamp::STAMP_LEN
        && u32_at(reply, 24) == u32_at(probe, 0)
        && u64_at(reply, 28) == u64_at(probe, 4))
    .then(|| u64_at(reply, 4).wrapping_sub(u64_at(reply, 16)))
}

/// RFC 8762 §4.3.1, byte for byte, for a probe in upfd's layout: stateless
/// Sequence Number, T3, Error Estimate, MBZ, T2, the sender's Sequence
/// Number, Timestamp and Error Estimate, MBZ, TTL 255, MBZ.
#[test]
fn the_reflection_is_rfc8762_for_upfd_s_probe_471() {
    let probe = upfd_probe(0x0000_0102, 0xE9A1_B2C3_D4E5_F607);
    let t2 = 0x1111_2222_3333_4444;
    let t3 = 0x1111_2222_3333_5555;
    let r = stamp::reflect(&probe, t2, t3).unwrap();
    let want: [u8; 44] = [
        0x00, 0x00, 0x01, 0x02, // Sequence Number (stateless: the sender's)
        0x11, 0x11, 0x22, 0x22, 0x33, 0x33, 0x55, 0x55, // Timestamp T3
        0x00, 0x01, // Error Estimate
        0x00, 0x00, // MBZ
        0x11, 0x11, 0x22, 0x22, 0x33, 0x33, 0x44, 0x44, // Receive Timestamp T2
        0x00, 0x00, 0x01, 0x02, // Session-Sender Sequence Number
        0xE9, 0xA1, 0xB2, 0xC3, 0xD4, 0xE5, 0xF6, 0x07, // Session-Sender Timestamp
        0x00, 0x01, // Session-Sender Error Estimate
        0x00, 0x00, // MBZ
        0xFF, // Session-Sender TTL
        0x00, 0x00, 0x00, // MBZ
    ];
    assert_eq!(r, want);
    assert_eq!(upfd_accepts(&probe, &r), Some(0x1111));
    // Shorter than a test packet: not answered. Longer: answered with 44
    // octets, never more than was received.
    assert!(stamp::reflect(&probe[..43], t2, t3).is_none());
    let mut long = [0u8; 100];
    long[..44].copy_from_slice(&probe);
    assert_eq!(stamp::reflect(&long, t2, t3).unwrap().len(), 44);
}

/// Through a real socket: the reflector answers upfd's probe at once, upfd
/// would accept the reflection, and the turnaround is small and non-negative.
#[tokio::test]
async fn the_reflector_answers_a_probe_on_its_socket_471() {
    let sock = stamp::bind("127.0.0.1:0".parse().unwrap()).await.unwrap();
    let at = sock.local_addr().unwrap();
    let stats = Arc::new(stamp::ReflectorStats::default());
    tokio::spawn(stamp::run(sock, stats.clone()));
    let sender = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
    sender.connect(at).await.unwrap();
    let probe = upfd_probe(7, ntp64(std::time::SystemTime::now()));
    sender.send(&probe[..20]).await.unwrap(); // too short: dropped
    sender.send(&probe).await.unwrap();
    let mut buf = [0u8; 128];
    let n = tokio::time::timeout(Duration::from_secs(2), sender.recv(&mut buf))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(n, 44);
    let turn = upfd_accepts(&probe, &buf[..n]).expect("upfd accepts it");
    assert!(turn < (1u64 << 32), "turnaround under 1 s: {turn}");
    assert_eq!(stats.reflected.load(Ordering::Relaxed), 1);
    assert_eq!(stats.dropped.load(Ordering::Relaxed), 1);
}

// ---- jobs ----------------------------------------------------------------

#[test]
fn the_job_frames_round_trip_and_anything_else_is_dropped_471() {
    let f = job::job_frame(3, 100);
    assert_eq!(f.len(), job::FRAME_LEN);
    assert_eq!(
        job::parse_job(&f),
        Some(job::Job {
            job: 3,
            req_comp: 100
        })
    );
    assert_eq!(job::parse_done(&f), None, "a job is not an answer");
    let d = job::Done {
        job: 3,
        compute_ms: 2,
        avl_comp: 50,
        eas_id: "eas-local".into(),
    };
    let df = job::done_frame(&d);
    assert_eq!(df.len(), job::FRAME_LEN, "an answer is the request's size");
    assert_eq!(job::parse_done(&df), Some(d));
    assert_eq!(job::parse_job(&f[..63]), None);
    let mut bad = f;
    bad[0] = b'X';
    assert_eq!(job::parse_job(&bad), None);
    let mut v2 = f;
    v2[4] = 2;
    assert_eq!(job::parse_job(&v2), None);
    // The compute stand-in: ceil(reqComp / avlComp), capped; none at 0.
    assert_eq!(job::compute_ms(100, 50), Some(2));
    assert_eq!(job::compute_ms(101, 50), Some(3));
    assert_eq!(job::compute_ms(0, 50), Some(0));
    assert_eq!(job::compute_ms(100, 0), None);
    assert_eq!(job::compute_ms(u32::MAX, 1), Some(job::MAX_COMPUTE_MS));
}

/// An EAS job listener on loopback with `avl` available compute.
async fn eas_listener(eas_id: &str, avl: u32) -> (SocketAddr, Arc<AtomicU32>, Arc<job::JobStats>) {
    let sock = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let at = sock.local_addr().unwrap();
    let avl = Arc::new(AtomicU32::new(avl));
    let stats = Arc::new(job::JobStats::default());
    tokio::spawn(job::serve(
        sock,
        eas_id.to_string(),
        avl.clone(),
        stats.clone(),
    ));
    (at, avl, stats)
}

/// The EAS answers after `ceil(reqComp / avlComp)` ms with its easId; a
/// lowered avlComp (applied load) slows the next job.
#[tokio::test]
async fn the_eas_computes_for_ceil_reqcomp_over_avlcomp_471() {
    let (at, avl, stats) = eas_listener("eas-local", 50).await;
    let a = job::run(at, None, 1, 100, Duration::from_secs(2))
        .await
        .unwrap();
    assert_eq!(a.done.eas_id, "eas-local");
    assert_eq!((a.done.job, a.done.compute_ms, a.done.avl_comp), (1, 2, 50));
    assert!(a.rtt_ms >= 2, "{a:?}");
    avl.store(5, Ordering::Relaxed);
    let a = job::run(at, None, 2, 100, Duration::from_secs(2))
        .await
        .unwrap();
    assert_eq!((a.done.compute_ms, a.done.avl_comp), (20, 5));
    assert!(a.rtt_ms >= 20, "{a:?}");
    assert_eq!(stats.executed.load(Ordering::Relaxed), 2);
    // Garbage is dropped and answered with nothing.
    let s = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
    s.send_to(b"hello", at).await.unwrap();
    let mut buf = [0u8; 64];
    assert!(
        tokio::time::timeout(Duration::from_millis(200), s.recv(&mut buf))
            .await
            .is_err()
    );
    assert_eq!(stats.dropped.load(Ordering::Relaxed), 1);
}

// ---- EDGE-3 / EDGE-1 bodies ----------------------------------------------

fn local_ad() -> EasAd {
    EasAd {
        eas_id: "eas-local".into(),
        ip: "10.60.0.11".parse().unwrap(),
        max_resp_time_ms: 10,
        avl_comp: 50,
        ac_ids: vec![],
    }
}

/// TS 29.558 `EASRegistration`: `easProf` with `easId`, an `endPt` of one
/// address list (`oneOf`; no port member exists) and `svcKpi`
/// {`maxRespTime`, `avlComp`}; nothing else. The patch repeats `easId` and
/// `endPt` (an `EASProfile` requires them) and carries only `avlComp`.
#[test]
fn the_eas_bodies_are_the_ts29558_shapes_471() {
    assert_eq!(
        eel::eas_registration(&local_ad()),
        json!({"easProf":{"easId":"eas-local","endPt":{"ipv4Addrs":["10.60.0.11"]},
               "svcKpi":{"maxRespTime":10,"avlComp":50}}})
    );
    let mut v6 = local_ad();
    v6.ip = "fd00::11".parse().unwrap();
    v6.ac_ids = vec!["ac-d443a".into()];
    assert_eq!(
        eel::eas_registration(&v6)["easProf"],
        json!({"easId":"eas-local","endPt":{"ipv6Addrs":["fd00::11"]},"acIds":["ac-d443a"],
               "svcKpi":{"maxRespTime":10,"avlComp":50}})
    );
    assert_eq!(
        eel::avl_comp_patch(&local_ad(), 10),
        json!({"easProf":{"easId":"eas-local","endPt":{"ipv4Addrs":["10.60.0.11"]},
               "svcKpi":{"avlComp":10}}})
    );
}

fn eec_cfg(ees: &str) -> EecClientConfig {
    serde_yaml::from_str(&format!(
        "ees: \"{ees}\"\neec_id: eec-471\neass: [eas-local, eas-regional, eas-cloud]\n"
    ))
    .unwrap()
}

/// TS 24.558 `EECRegistration`: `eecId`, one `ACProfile` whose `eass` are
/// `EasDetail` {`easId`, `minimumReqSvcKPIs.reqComp` (a string)}, and
/// `easSelReqInd: true`. The `eec:` block's defaults are the demo's.
#[test]
fn the_eec_body_is_the_ts24558_shape_471() {
    let mut c = eec_cfg("http://172.23.0.31:7777");
    assert_eq!(
        (c.ac_id.as_str(), c.req_comp, c.deadline_ms, c.job_port),
        ("ac-d443a", 100, 15, 9471)
    );
    assert!(c.device.is_none() && c.oauth2.is_none() && c.ue_id.is_none());
    let detail = |id: &str| json!({"easId": id, "minimumReqSvcKPIs": {"reqComp": "100"}});
    assert_eq!(
        eel::eec_registration(&c),
        json!({"eecId":"eec-471","easSelReqInd":true,"acProfs":[{"acId":"ac-d443a",
               "eass":[detail("eas-local"),detail("eas-regional"),detail("eas-cloud")]}]})
    );
    c.ue_id = Some("msisdn-001010000000001".into());
    assert_eq!(eel::eec_registration(&c)["ueId"], "msisdn-001010000000001");
    assert!(serde_yaml::from_str::<EecClientConfig>(
        "ees: \"http://x:1\"\neec_id: e\neass: [a]\nbogus: 1\n"
    )
    .is_err());
    assert!(eec::validate(&c).is_ok());
    for bad in [
        EecClientConfig {
            eass: vec![],
            ..c.clone()
        },
        EecClientConfig {
            eec_id: " ".into(),
            ..c.clone()
        },
        EecClientConfig {
            ees: "https://x:1".into(),
            ..c.clone()
        },
        EecClientConfig {
            deadline_ms: 0,
            ..c.clone()
        },
    ] {
        assert!(eec::validate(&bad).is_err(), "{bad:?}");
    }
}

/// A selection in eesd's shape (PR #1523's `DiscoveredEas`: `eas` plus
/// `vendorSpecific-032473` {achievableLatency, budgetSrc}) reads back in
/// order; without the member (demo off at eesd) the prediction is absent.
#[test]
fn a_selection_reads_back_in_eesd_s_shape_471() {
    let body = json!({"eecId":"eec-471","registrationId":"r","discoveredEas":[
        {"eas":{"easId":"eas-regional","endPt":{"ipv4Addrs":["10.60.0.12"]},
                "svcKpi":{"maxRespTime":14,"avlComp":50}},
         "vendorSpecific-032473":{"achievableLatency":12,"budgetSrc":"upf-n3+upf-n6"}},
        {"eas":{"easId":"eas-local","endPt":{"ipv6Addrs":["fd00::11"]}}},
        {"eas":{"endPt":{}}}
    ]});
    let s = eel::selection(&body);
    assert_eq!(s.len(), 2);
    assert_eq!(s[0].eas_id, "eas-regional");
    assert_eq!(s[0].ip, Some("10.60.0.12".parse::<IpAddr>().unwrap()));
    assert_eq!(s[0].achievable_latency_ms, Some(12));
    assert_eq!(s[0].budget_src.as_deref(), Some("upf-n3+upf-n6"));
    assert_eq!(s[1].ip, Some("fd00::11".parse::<IpAddr>().unwrap()));
    assert_eq!(
        (s[1].achievable_latency_ms, s[1].budget_src.as_deref()),
        (None, None)
    );
    assert!(eel::selection(&json!({"eecId":"e"})).is_empty());
}

// ---- the sidecar's registration ------------------------------------------

fn plain() -> SbiClient {
    SbiClient::new(SbiClientConfig {
        timeout: Duration::from_secs(2),
        retries: 0,
        ..SbiClientConfig::default()
    })
}

/// Registered once and then left alone: checks do not touch the
/// registration, so it stays byte-identical. A lost registration (eesd
/// restart) is re-created. Load merge-patches `avlComp` only. Shutdown
/// deletes it.
#[tokio::test]
async fn the_sidecar_registers_once_patches_avlcomp_and_deregisters_471() {
    let ees = StubEes::start().await;
    let mut reg = Registrar::new(plain(), &ees.api_root, local_ad());
    let Check::Registered(loc) = reg.ensure().await else {
        panic!("not registered");
    };
    assert!(
        loc.starts_with(&format!("{}{EASREG_PATH}/", ees.api_root)),
        "{loc}"
    );
    let before = ees.eas();
    assert_eq!(before.len(), 1);
    let stored = before.values().next().unwrap().clone();
    assert_eq!(
        stored["easProf"],
        eel::eas_registration(&local_ad())["easProf"]
    );
    for _ in 0..3 {
        assert_eq!(reg.ensure().await, Check::Alive);
    }
    assert_eq!(ees.eas(), before, "byte-identical: the checks only read it");
    let posts = |e: &StubEes| e.seen().iter().filter(|s| s.method == "POST").count();
    assert_eq!(posts(&ees), 1);

    reg.patch_avl_comp(10).await.unwrap();
    let after = ees.eas().values().next().unwrap().clone();
    assert_eq!(
        after["easProf"]["svcKpi"],
        json!({"maxRespTime":10,"avlComp":10})
    );
    assert_eq!(after["easProf"]["endPt"], stored["easProf"]["endPt"]);
    let patch = ees
        .seen()
        .into_iter()
        .find(|s| s.method == "PATCH")
        .unwrap();
    assert_eq!(
        patch.content_type.as_deref(),
        Some("application/merge-patch+json")
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&patch.body).unwrap(),
        eel::avl_comp_patch(&local_ad(), 10)
    );

    ees.restart();
    assert!(matches!(reg.ensure().await, Check::Registered(_)));
    assert_eq!(posts(&ees), 2);
    let again = ees.eas().values().next().unwrap().clone();
    assert_eq!(
        again["easProf"]["svcKpi"]["avlComp"], 10,
        "re-registered with the patched KPI"
    );

    reg.deregister().await.unwrap();
    assert!(ees.eas().is_empty());
    assert!(reg.location().is_none());
}

/// An EES that refuses (401, no token) is reported and retried, never
/// treated as registered.
#[tokio::test]
async fn a_refused_registration_is_retried_not_assumed_471() {
    let ees = StubEes::start().await;
    ees.require_bearer("t");
    let mut reg = Registrar::new(plain(), &ees.api_root, local_ad());
    assert!(matches!(reg.ensure().await, Check::Failed(_)));
    assert!(reg.location().is_none());
    assert!(reg.patch_avl_comp(5).await.is_err());
}

// ---- the EEC client --------------------------------------------------------

async fn eec_setup(avl_local: u32) -> (StubEes, EecClientConfig, Arc<job::JobStats>) {
    let ees = StubEes::start().await;
    let (at, _, stats) = eas_listener("eas-local", avl_local).await;
    let mut ad = local_ad();
    ad.ip = at.ip();
    let mut reg = Registrar::new(plain(), &ees.api_root, ad);
    assert!(matches!(reg.ensure().await, Check::Registered(_)));
    let mut cfg = eec_cfg(&ees.api_root);
    cfg.job_port = at.port();
    (ees, cfg, stats)
}

/// Act 1 shape: the EES selects eas-local; the job runs there within the
/// deadline; the log carries `eas=`, `rtt=`, `predicted=`. The registration
/// that asked is deleted at once.
#[tokio::test]
async fn the_job_goes_to_the_first_discovered_eas_471() {
    let (ees, mut cfg, stats) = eec_setup(50).await;
    // A loose UE deadline, so a loaded CI host cannot turn this into a miss;
    // the deadline itself is the next test's.
    cfg.deadline_ms = 1000;
    ees.set_reply(EecReply::Select(vec![("eas-local".into(), 12)]));
    let c = ees_client(None, None, Duration::from_secs(2))
        .await
        .unwrap();
    let out = eec::offload(&c, &cfg, 1).await;
    let eec::Outcome::Offloaded {
        eas_id,
        rtt_ms,
        predicted_ms,
        in_time,
        ..
    } = &out
    else {
        panic!("{out:?}");
    };
    assert_eq!(
        (eas_id.as_str(), *predicted_ms, *in_time),
        ("eas-local", Some(12), true)
    );
    assert!(*rtt_ms >= 2);
    let line = out.line(1, cfg.deadline_ms, "measured-budget");
    assert!(
        line.starts_with("JOB job=1 eas=eas-local rtt=")
            && line.contains(
                " predicted=12 src=upf-n3+upf-n6 deadline=1000 mode=measured-budget result=ok"
            ),
        "{line}"
    );
    assert_eq!(stats.executed.load(Ordering::Relaxed), 1);
    assert_eq!(ees.eec_count(), 0, "the asking registration is deleted");
    let post = ees
        .seen()
        .into_iter()
        .find(|s| s.path == EECREG_PATH)
        .unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&post.body).unwrap(),
        eel::eec_registration(&cfg)
    );
}

/// Baseline shape: the EES keeps a slow EAS; the job overruns the UE's
/// deadline: `deadline-miss`. No answer at all is a miss too.
#[tokio::test]
async fn a_slow_or_silent_eas_is_a_deadline_miss_471() {
    let (ees, mut cfg, _) = eec_setup(5).await; // 100/5 = 20 ms > 15 ms
    ees.set_reply(EecReply::Select(vec![("eas-local".into(), 10)]));
    let c = ees_client(None, None, Duration::from_secs(2))
        .await
        .unwrap();
    let out = eec::offload(&c, &cfg, 2).await;
    assert!(out.missed(), "{out:?}");
    assert!(out
        .line(2, 15, "static")
        .ends_with("mode=static result=deadline-miss"));
    cfg.job_port = 9; // nothing listens
    cfg.job_timeout_ms = 100;
    let out = eec::offload(&c, &cfg, 3).await;
    assert!(matches!(out, eec::Outcome::NoAnswer { .. }), "{out:?}");
    assert!(out.missed());
    assert!(out
        .line(3, 15, "static")
        .contains("rtt=? deadline=15 mode=static result=deadline-miss"));
}

/// Act 3 shape: the EES answers 404 (nothing within its policy): on-device,
/// `reason=ees-404`, and no job is sent. 401 (no token) and an unreachable
/// EES are on-device too, each with its reason.
#[tokio::test]
async fn no_eas_means_on_device_with_the_reason_471() {
    let (ees, cfg, stats) = eec_setup(50).await;
    let c = ees_client(None, None, Duration::from_secs(2))
        .await
        .unwrap();
    ees.set_reply(EecReply::Status(404));
    let out = eec::offload(&c, &cfg, 4).await;
    assert_eq!(
        out,
        eec::Outcome::OnDevice {
            reason: "ees-404".into()
        }
    );
    assert_eq!(
        out.line(4, 15, "measured-budget"),
        "JOB job=4 on-device fallback reason=ees-404 mode=measured-budget"
    );
    ees.require_bearer("t");
    assert_eq!(
        eec::offload(&c, &cfg, 5).await,
        eec::Outcome::OnDevice {
            reason: "ees-401".into()
        }
    );
    ees.set_reply(EecReply::Select(vec![]));
    let mut unreach = cfg.clone();
    unreach.ees = "http://127.0.0.1:9".into();
    assert_eq!(
        eec::offload(&c, &unreach, 6).await,
        eec::Outcome::OnDevice {
            reason: "ees-unreachable".into()
        }
    );
    assert_eq!(
        stats.executed.load(Ordering::Relaxed),
        0,
        "no job left the device"
    );
}

/// Every socket of the EEC is bound to `device`: on `lo` the loopback EES
/// and EAS are reached; on an interface that does not exist neither the EES
/// nor the EAS sees anything (the `bind_device: eth0` revert arm's shape:
/// bound elsewhere, the job never takes the session's path).
#[tokio::test]
async fn the_eec_is_bound_to_its_device_471() {
    let (ees, mut cfg, stats) = eec_setup(50).await;
    ees.set_reply(EecReply::Select(vec![("eas-local".into(), 12)]));
    cfg.device = Some("lo".into());
    let c = ees_client(None, cfg.device.as_deref(), Duration::from_secs(2))
        .await
        .unwrap();
    assert!(matches!(
        eec::offload(&c, &cfg, 7).await,
        eec::Outcome::Offloaded { .. }
    ));
    assert_eq!(stats.executed.load(Ordering::Relaxed), 1);

    let seen = ees.seen().len();
    cfg.device = Some("nosuchif471".into());
    let c = ees_client(None, cfg.device.as_deref(), Duration::from_secs(2))
        .await
        .unwrap();
    assert_eq!(
        eec::offload(&c, &cfg, 8).await,
        eec::Outcome::OnDevice {
            reason: "ees-unreachable".into()
        }
    );
    assert_eq!(ees.seen().len(), seen, "nothing reached the EES");
    // The job socket alone, bound to a missing device, cannot send either.
    let at = SocketAddr::new("127.0.0.1".parse().unwrap(), cfg.job_port);
    assert!(
        job::run(at, Some("nosuchif471"), 9, 100, Duration::from_millis(200))
            .await
            .is_err()
    );
    assert_eq!(stats.executed.load(Ordering::Relaxed), 1);
}

/// The client's AF profile is re-registered at the NRF every half heartbeat,
/// so the NRF never drops it and keeps granting tokens on a long run.
#[tokio::test]
async fn the_af_profile_is_kept_registered_at_the_nrf_471() {
    let nrf = StubEes::start().await;
    let task = tokio::spawn(super::keep_registered(
        nrf.api_root.clone(),
        "3f6d7a8e-0000-4000-8000-000000000471".into(),
        SbiClientConfig {
            timeout: Duration::from_secs(1),
            ..SbiClientConfig::default()
        },
        Duration::from_millis(30),
    ));
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    let puts = |s: &StubEes| {
        s.seen()
            .iter()
            .filter(|r| {
                r.method == "PUT"
                    && r.path == "/nnrf-nfm/v1/nf-instances/3f6d7a8e-0000-4000-8000-000000000471"
            })
            .count()
    };
    while puts(&nrf) < 3 {
        assert!(tokio::time::Instant::now() < deadline, "{}", puts(&nrf));
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    task.abort();
    let put = nrf.seen().into_iter().find(|r| r.method == "PUT").unwrap();
    let body: Value = serde_json::from_slice(&put.body).unwrap();
    assert_eq!(body["nfType"], "AF");
    assert_eq!(body["heartBeatTimer"], super::HEARTBEAT_SECS);
}
