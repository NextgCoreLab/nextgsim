//! The AF end to end over loopback: real TCP clients bound to distinct source
//! addresses, a stub nefd over h2c, and T8 notifications delivered to the AF's
//! callback listener over h2c.

use super::*;
use crate::d445a::client::{run_session, ClientParams, Trainer};
use crate::d445a::label;
use crate::d445a::stub_nef::{gpsi_of, loc_body, StubNef};
use http::StatusCode;
use tokio::net::TcpSocket;
use tokio::task::JoinHandle;

/// The source address of member `m` in these tests: 127.0.0.(10 + m).
fn octet(m: u32) -> u8 {
    10 + m as u8
}

fn yaml(nef: &str, mode: &str, extra: &str) -> ServerConfig {
    let _ = tracing_subscriber::fmt().with_test_writer().try_init();
    let base = format!(
        "listen: 127.0.0.1:0\nt8_listen: 127.0.0.1:0\nnef: {{ api_root: \"{nef}\", timeout_ms: 2000 }}\n\
         seed: 480\njoin_timeout_s: 20\ndemo:\n  enabled: true\n  d445a-fl-t8:\n    \
         enabled: true\n    mode: {mode}\n"
    );
    let mut root: serde_yaml::Value = serde_yaml::from_str(&base).unwrap();
    if !extra.trim().is_empty() {
        let over: serde_yaml::Mapping = serde_yaml::from_str(extra).unwrap();
        let m = root.as_mapping_mut().unwrap();
        for (k, v) in over {
            m.insert(k, v);
        }
    }
    serde_yaml::from_value(root).unwrap()
}

struct Running {
    af: JoinHandle<RunReport>,
    progress: broadcast::Receiver<Progress>,
    fl: SocketAddr,
    t8: Option<SocketAddr>,
}

async fn start(mut cfg: ServerConfig, mode: &str, knobs: Knobs) -> Running {
    let fl = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let fl_addr = fl.local_addr().unwrap();
    let (t8, t8_addr) = if knobs.no_t8 {
        (None, None)
    } else {
        let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let a = l.local_addr().unwrap();
        cfg.t8_notify_base = Some(format!("http://{a}"));
        (Some(l), Some(a))
    };
    let af = Af::new(cfg, mode, knobs).await.unwrap();
    let progress = af.subscribe();
    Running {
        af: tokio::spawn(af.run(fl, t8)),
        progress,
        fl: fl_addr,
        t8: t8_addr,
    }
}

async fn connect(m: u32, af: SocketAddr) -> TcpStream {
    let s = TcpSocket::new_v4().unwrap();
    s.bind(SocketAddr::from(([127, 0, 0, octet(m)], 0)))
        .unwrap();
    s.connect(af).await.unwrap()
}

/// An honest member in `mode`, reconnect-free: one session.
fn member(
    m: u32,
    clients: u32,
    af: SocketAddr,
    mode: &'static str,
    leak_raw: bool,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut p = ClientParams::demo(m, 480, clients);
        p.leak_raw = leak_raw;
        let t = Trainer::new(p);
        let stream = connect(m, af).await;
        let _ = run_session(
            stream,
            &t,
            Duration::from_secs(60),
            || Some(mode.to_string()),
            |_| {},
        )
        .await;
    })
}

/// A member that joins and then never answers, its connection left open: a UE
/// whose user plane is gone with no FIN.
fn silent(m: u32, af: SocketAddr) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut s = connect(m, af).await;
        let hello = frame::encode(&Frame::Hello { label: label(m) }).unwrap();
        s.write_all(&hello).await.unwrap();
        std::future::pending::<()>().await;
        drop(s);
    })
}

async fn until(progress: &mut broadcast::Receiver<Progress>, want: impl Fn(&Progress) -> bool) {
    tokio::time::timeout(Duration::from_secs(60), async {
        loop {
            match progress.recv().await {
                Ok(p) if want(&p) => return,
                Ok(_) | Err(broadcast::error::RecvError::Lagged(_)) => {}
                Err(e) => panic!("progress: {e}"),
            }
        }
    })
    .await
    .expect("the AF never got there");
}

async fn finish(r: Running) -> RunReport {
    tokio::time::timeout(Duration::from_secs(120), r.af)
        .await
        .expect("the AF finished")
        .unwrap()
}

/// The success signal: ue3 departs while the AF holds before round 4, the
/// NEF's T8 LoC report excludes it, round 4 runs at participants=2 quickly, and
/// after ue3 restarts it is readmitted and rounds 7-10 run at 3. The network
/// run equals the in-process run with the same membership, bit for bit.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn t8_exclusion_excludes_ue3_and_readmits_it() {
    let nef = StubNef::start().await;
    let cfg = yaml(
        &nef.api_root,
        MODE_T8_EXCLUSION,
        "round_deadline_ms: 10000\nholds:\n  - { before_round: 4, until: member-excluded, timeout_s: 30 }\n  \
         - { before_round: 7, until: members-full, timeout_s: 30 }",
    );
    let mut run = start(cfg, MODE_T8_EXCLUSION, Knobs::default()).await;
    let mut clients: Vec<_> = (1..=3)
        .map(|m| member(m, 3, run.fl, MODE_T8_EXCLUSION, false))
        .collect();

    until(&mut run.progress, |p| *p == Progress::Holding(4)).await;
    assert_eq!(nef.live().len(), 3, "one T8 subscription per member");
    // An id the AF did not mint is 404 at its callback listener.
    let t8 = run.t8.unwrap();
    let forged = format!(
        "http://{t8}{}{}",
        nef::CALLBACK_PREFIX,
        nef::new_callback_id()
    );
    assert_eq!(
        nef.post(&forged, &loc_body("sub-3", "0010000000013")).await,
        StatusCode::NOT_FOUND
    );
    // ue3 deregisters: its transport goes, then the NEF reports the LoC.
    clients.remove(2).abort();
    assert_eq!(
        nef.notify_loc(&gpsi_of(octet(3))).await,
        StatusCode::NO_CONTENT
    );

    until(&mut run.progress, |p| *p == Progress::Holding(7)).await;
    clients.push(member(3, 3, run.fl, MODE_T8_EXCLUSION, false));
    let report = finish(run).await;

    assert_eq!(
        report.exclusions,
        vec![("ue3".to_string(), gpsi_of(octet(3)))]
    );
    assert_eq!(report.readmissions, vec!["ue3".to_string()]);
    assert_eq!(report.canary_hits, 0);
    let arm = &report.arms[0];
    let n = |r: usize| {
        (
            arm.rounds[r - 1].selected.len(),
            arm.rounds[r - 1].updates.len(),
        )
    };
    for r in 1..=3 {
        assert_eq!(n(r), (3, 3), "round {r}");
    }
    for r in 4..=6 {
        assert_eq!(n(r), (2, 2), "round {r}");
        assert!(arm.rounds[r - 1].stragglers.is_empty());
    }
    assert!(
        arm.rounds[3].dur_ms < 3000,
        "round 4 dur_ms {} < 0.3T",
        arm.rounds[3].dur_ms
    );
    for r in 7..=10 {
        assert_eq!(n(r), (3, 3), "round {r}");
    }
    assert!(
        arm.oracle_bit_identical,
        "the network run is the in-process run"
    );
    // Every seed: finite and better than local-only. Seed 480 only (one run;
    // see the sim module's seed table): ≤ 0.5× round 1 and ≤ 1.2× pooled.
    assert!(
        arm.last().is_finite() && arm.last() <= 0.8 * arm.local_only,
        "{arm:?}"
    );
    assert!(
        arm.last() <= 0.5 * arm.first(),
        "seed 480: {} vs {}",
        arm.last(),
        arm.first()
    );
    assert!(
        arm.last() <= 1.2 * arm.pooled,
        "seed 480: {} vs {}",
        arm.last(),
        arm.pooled
    );
    // The first ue3 subscription is deleted after the readmission; a second one
    // is live.
    tokio::time::sleep(Duration::from_millis(300)).await;
    let ue3: Vec<_> = nef
        .all()
        .into_iter()
        .filter(|s| s.msisdn == "0010000000013")
        .collect();
    assert_eq!(ue3.len(), 2);
    assert!(ue3[0].deleted && !ue3[1].deleted, "{ue3:?}");
    for c in clients {
        c.abort();
    }
}

/// A T8 report arriving while the AF waits for a silent member ends the wait at
/// once: the round finishes with the two others far inside the deadline.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_exclusion_mid_round_stops_the_wait() {
    let nef = StubNef::start().await;
    let cfg = yaml(
        &nef.api_root,
        MODE_T8_EXCLUSION,
        "rounds: 2\nround_deadline_ms: 20000",
    );
    let mut run = start(cfg, MODE_T8_EXCLUSION, Knobs::default()).await;
    let a = member(1, 3, run.fl, MODE_T8_EXCLUSION, false);
    let b = member(2, 3, run.fl, MODE_T8_EXCLUSION, false);
    let c = silent(3, run.fl);
    until(&mut run.progress, |p| {
        matches!(p, Progress::RoundStarted { round: 1, .. })
    })
    .await;
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(
        nef.notify_loc(&gpsi_of(octet(3))).await,
        StatusCode::NO_CONTENT
    );
    let report = finish(run).await;
    let r1 = &report.arms[0].rounds[0];
    assert_eq!((r1.selected.len(), r1.updates.len()), (3, 2));
    assert!(r1.stragglers.is_empty());
    assert!(r1.dur_ms < 6000, "dur_ms {} < 0.3T", r1.dur_ms);
    assert_eq!(report.arms[0].rounds[1].selected, vec!["ue1", "ue2"]);
    for h in [a, b, c] {
        h.abort();
    }
}

/// Revert arm `--no-t8`: no subscription is made, so the silent member is
/// waited for until the deadline (`straggler_timeout`, dur_ms ≥ T) and nobody
/// is excluded.
#[cfg(feature = "revert-hooks")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn without_t8_the_round_waits_for_the_deadline() {
    let nef = StubNef::start().await;
    let cfg = yaml(
        &nef.api_root,
        MODE_T8_EXCLUSION,
        "rounds: 1\nround_deadline_ms: 1500",
    );
    let knobs = Knobs {
        no_t8: true,
        freeze_global: false,
    };
    let run = start(cfg, MODE_T8_EXCLUSION, knobs).await;
    let hs = [
        member(1, 3, run.fl, MODE_T8_EXCLUSION, false),
        member(2, 3, run.fl, MODE_T8_EXCLUSION, false),
        silent(3, run.fl),
    ];
    let report = finish(run).await;
    let r1 = &report.arms[0].rounds[0];
    assert_eq!(r1.stragglers, vec!["ue3"]);
    assert!(r1.dur_ms >= 1500, "dur_ms {}", r1.dur_ms);
    assert!(report.exclusions.is_empty());
    assert!(nef.all().is_empty(), "no T8 subscription");
    for h in hs {
        h.abort();
    }
}

/// Revert arm `--freeze-global`: the model never moves, so round 10 is round 1
/// and the scorecard (≤ 0.5× round 1) fails.
#[cfg(feature = "revert-hooks")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_frozen_global_fails_the_scorecard() {
    let nef = StubNef::start().await;
    let cfg = yaml(&nef.api_root, MODE_T8_EXCLUSION, "rounds: 4");
    let knobs = Knobs {
        no_t8: false,
        freeze_global: true,
    };
    let run = start(cfg, MODE_T8_EXCLUSION, knobs).await;
    let hs: Vec<_> = (1..=3)
        .map(|m| member(m, 3, run.fl, MODE_T8_EXCLUSION, false))
        .collect();
    let report = finish(run).await;
    let arm = &report.arms[0];
    assert_eq!(arm.last().to_bits(), arm.first().to_bits());
    assert!(arm.last() > 0.5 * arm.first());
    assert!(!arm.oracle_bit_identical);
    for h in hs {
        h.abort();
    }
}

/// Revert arm `leak-raw`: a client that appends a training row to its update
/// has the frame refused and its connection closed; the canary is counted.
#[cfg(feature = "revert-hooks")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_leaky_update_is_refused_and_its_canary_counted() {
    let nef = StubNef::start().await;
    let cfg = yaml(
        &nef.api_root,
        MODE_T8_EXCLUSION,
        "rounds: 1\nround_deadline_ms: 1500",
    );
    let run = start(cfg, MODE_T8_EXCLUSION, Knobs::default()).await;
    let hs = [
        member(1, 3, run.fl, MODE_T8_EXCLUSION, true),
        member(2, 3, run.fl, MODE_T8_EXCLUSION, false),
        member(3, 3, run.fl, MODE_T8_EXCLUSION, false),
    ];
    let report = finish(run).await;
    assert!(report.canary_hits >= 1);
    assert_eq!(
        report.refused_frames.len(),
        1,
        "{:?}",
        report.refused_frames
    );
    assert!(
        report.refused_frames[0].starts_with("ue1: "),
        "{:?}",
        report.refused_frames
    );
    assert_eq!(report.arms[0].rounds[0].updates, vec!["ue2", "ue3"]);
    assert!(
        report.exclusions.is_empty(),
        "a refused frame is not an exclusion"
    );
    for h in hs {
        h.abort();
    }
}

/// Poisoned-UE arm (5 UEs, UE 005 ×−10): FedAvg diverges (≥ 10× its clean run),
/// median and trimmed mean stay within 1.1× of theirs (seed-480 results), and
/// Krum names ue5 worst, and never selects it, in every round. Every arm equals
/// its in-process run bit for bit.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_poisoned_arm_separates_the_rules() {
    let nef = StubNef::start().await;
    let cfg = yaml(&nef.api_root, MODE_POISONED_UE, "");
    let run = start(cfg, MODE_POISONED_UE, Knobs::default()).await;
    let hs: Vec<_> = (1..=5)
        .map(|m| member(m, 5, run.fl, MODE_POISONED_UE, false))
        .collect();
    let report = finish(run).await;
    let rules: Vec<_> = report.arms.iter().map(|a| a.rule).collect();
    assert_eq!(
        rules,
        [
            RuleName::Fedavg,
            RuleName::Median,
            RuleName::TrimmedMean,
            RuleName::Krum
        ]
    );
    for a in &report.arms {
        assert!(a.oracle_bit_identical, "{:?}", a.rule);
        let ratio = a.last() / a.clean_last.unwrap();
        eprintln!("{:?}: poisoned/clean {ratio}", a.rule);
        match a.rule {
            RuleName::Fedavg => assert!(ratio >= 10.0, "fedavg {ratio}"),
            RuleName::Median | RuleName::TrimmedMean => {
                assert!(ratio <= 1.1, "{:?} {ratio}", a.rule)
            }
            RuleName::Krum => {
                for r in &a.rounds {
                    let (sel, worst) = r.krum.clone().expect("5 > 2f + 2");
                    assert_eq!(worst, "ue5", "round {}", r.round);
                    assert_ne!(sel, "ue5");
                }
            }
        }
    }
    for h in hs {
        h.abort();
    }
}

/// `Update.samples` is client-supplied: a `u32::MAX` claim is capped at the
/// per-member count, so it weighs the same as the honest 240 and the run is
/// the honest in-process run, bit for bit.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_huge_samples_claim_is_capped() {
    let nef = StubNef::start().await;
    let cfg = yaml(&nef.api_root, MODE_T8_EXCLUSION, "rounds: 3");
    let run = start(cfg, MODE_T8_EXCLUSION, Knobs::default()).await;
    let fl = run.fl;
    let liar = tokio::spawn(async move {
        let t = Trainer::new(ClientParams::demo(1, 480, 3));
        let s = connect(1, fl).await;
        let (mut rd, mut wr) = tokio::io::split(s);
        wr.write_all(
            &frame::encode(&Frame::Hello {
                label: "ue1".into(),
            })
            .unwrap(),
        )
        .await
        .unwrap();
        let mut reader = FrameReader::default();
        while let Ok(Some((Frame::GlobalModel { round, weights }, _))) = reader.next(&mut rd).await
        {
            let u = t.update(round, &weights, false);
            let f = Frame::Update {
                round,
                samples: u32::MAX,
                loss: u.loss,
                delta: u.delta,
            };
            wr.write_all(&frame::encode(&f).unwrap()).await.unwrap();
        }
    });
    let hs = [
        member(2, 3, fl, MODE_T8_EXCLUSION, false),
        member(3, 3, fl, MODE_T8_EXCLUSION, false),
    ];
    let report = finish(run).await;
    assert_eq!(report.capped, vec!["ue1"]);
    assert!(
        report.arms[0].oracle_bit_identical,
        "capped to the honest weight"
    );
    liar.abort();
    for h in hs {
        h.abort();
    }
}

/// A peer the NEF cannot identify (not behind a PDU session) is refused and
/// never selected: the provenance check behind "AF on any address, client on
/// eth0".
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_peer_the_nef_cannot_identify_is_refused() {
    let nef = StubNef::start().await;
    let cfg = yaml(
        &nef.api_root,
        MODE_T8_EXCLUSION,
        "rounds: 1\njoin_timeout_s: 2\nround_deadline_ms: 500",
    );
    let run = start(cfg, MODE_T8_EXCLUSION, Knobs::default()).await;
    let fl = run.fl;
    let stranger = tokio::spawn(async move {
        let t = Trainer::new(ClientParams::demo(1, 480, 3));
        let s = TcpSocket::new_v4().unwrap();
        s.bind(SocketAddr::from(([127, 0, 0, 200], 0))).unwrap();
        let stream = s.connect(fl).await.unwrap();
        run_session(stream, &t, Duration::from_secs(30), || None, |_| {}).await
    });
    let report = finish(run).await;
    assert!(report.arms[0].rounds[0].selected.is_empty());
    assert!(nef.all().is_empty());
    let end = tokio::time::timeout(Duration::from_secs(5), stranger)
        .await
        .unwrap()
        .unwrap();
    assert!(
        matches!(end, crate::d445a::client::SessionEnd::Closed),
        "{end}"
    );
}

/// A notification on our callback id whose `subscription` link is not the
/// subscription we hold is ignored; the right link excludes.
#[tokio::test]
async fn a_notification_must_name_our_subscription() {
    let cfg = yaml(
        "http://127.0.0.1:9",
        MODE_T8_EXCLUSION,
        "t8_notify_base: \"http://127.0.0.1:7481\"",
    );
    let mut af = Af::new(cfg, MODE_T8_EXCLUSION, Knobs::default())
        .await
        .unwrap();
    let sub = Subscription {
        location: "/3gpp-monitoring-event/v1/fl-af/subscriptions/sub-7".into(),
        callback_id: "cb7".into(),
    };
    af.members.insert(
        "ue3".into(),
        Member {
            gpsi: Gpsi("msisdn-1".into()),
            conn: None,
            excluded: false,
            sub: Some(sub),
        },
    );
    let wrong = loc_body("sub-8", "1").to_string();
    af.on_t8("cb7", wrong.as_bytes());
    assert!(!af.members["ue3"].excluded);
    af.on_t8("cb-other", loc_body("sub-7", "1").to_string().as_bytes());
    assert!(!af.members["ue3"].excluded);
    af.on_t8("cb7", loc_body("sub-7", "1").to_string().as_bytes());
    assert!(af.members["ue3"].excluded);
}

/// Startup refusals: a wildcard callback address, too few members for the
/// attacker, a bad trim ratio.
#[tokio::test]
async fn configs_that_cannot_run_are_refused() {
    let nef = "http://127.0.0.1:9";
    let bad = [
        (MODE_T8_EXCLUSION, "t8_listen: 0.0.0.0:7481"),
        (MODE_T8_EXCLUSION, "t8_notify_base: \"http://0.0.0.0:7481\""),
        (MODE_POISONED_UE, "members: 4"),
        (MODE_T8_EXCLUSION, "trim_ratio: 1.0"),
        (MODE_T8_EXCLUSION, "rules: []"),
    ];
    for (mode, extra) in bad {
        let text = format!("nef: {{ api_root: \"{nef}\" }}\n{extra}\n");
        let cfg: ServerConfig = serde_yaml::from_str(&text).unwrap();
        assert!(
            Af::new(cfg, mode, Knobs::default()).await.is_err(),
            "{mode} {extra}"
        );
    }
    // The same wildcard bind is fine with an explicit, concrete base.
    let ok: ServerConfig = serde_yaml::from_str(&format!(
        "nef: {{ api_root: \"{nef}\" }}\nt8_listen: 0.0.0.0:7481\nt8_notify_base: \"http://10.46.0.10:7481\"\n"
    ))
    .unwrap();
    assert!(Af::new(ok, MODE_T8_EXCLUSION, Knobs::default())
        .await
        .is_ok());
}
