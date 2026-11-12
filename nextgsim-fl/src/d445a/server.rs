//! `nr-fl-server`: the FL application server (an AF on the N6 side) of demo
//! D445A. **Non-normative**: the AI/ML logic is out of 3GPP scope (TS 23.501
//! §5.46.1); only the NEF interactions follow 3GPP procedures.
//!
//! # What it does
//!
//! 1. **Join.** Each UE client connects over its PDU session and says
//!    `Hello { label }`. The AF asks the NEF who is behind the source address
//!    ([`Nef::ue_id`]) and subscribes to T8 `LOSS_OF_CONNECTIVITY` for that UE
//!    ([`Nef::subscribe_loc`]). A peer the NEF cannot identify is refused: that is
//!    what ties a member to a real PDU session (the "client on eth0" revert arm).
//! 2. **Rounds.** Each round selects the members that are joined and not
//!    excluded, sends each the global model, and waits for their updates until
//!    all are in or the straggler deadline `round_deadline_ms` (T) passes. The
//!    updates are aggregated with [`crate::sim::aggregate`], and the new global
//!    model is scored on the held-out set.
//! 3. **Exclusion.** A T8 report for a member excludes it at once (the AF stops
//!    waiting for it) and from every later round:
//!    `member_selection excluded=<gpsi> member=ue3 reason=nef_event`. A TCP
//!    EOF/RST is only logged (`transport_closed`): after a deregistration or a
//!    power cycle there is no user plane left to carry a FIN, as with a real
//!    handset that loses radio, so a closed transport is not evidence of anything.
//! 4. **Readmission.** A member that joins again with the same identity is
//!    readmitted (`member_selection readmitted=<gpsi>`) from the next round.
//! 5. **Scoring.** After each arm the AF logs the scorecard: round 1 and the last
//!    round against the local-only and pooled baselines, which it computes itself
//!    from the pinned synthetic task (the data is synthetic and declared so). In
//!    `poisoned-ue` it also scores the rule's own run without the attacker.
//!
//! # Trust
//!
//! - `Update.samples` is client-supplied and is the FedAvg weight, so it is capped
//!   at `samples` (the per-member count the AF's task gives every member): a
//!   `u32::MAX` claim weighs the same as an honest one.
//! - Frames are the strict `NFL1` codec ([`crate::frame`]); a refused frame
//!   closes the connection. Every frame received, refused or not, is scanned for
//!   the task's canary rows; a hit is logged and counted.
//! - Both listeners bind 127.0.0.1 by default. The callback address advertised to
//!   the NEF is never a wildcard.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::net::{IpAddr, SocketAddr};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use nextgsim_demo::demo_log;
use serde::Deserialize;
use tokio::io::AsyncWriteExt;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{broadcast, mpsc, Semaphore};

use super::nef::{self, CallbackIds, Gpsi, Nef, NefConfig, Subscription};
use super::{
    member_of, FrameReader, ReadError, ATTACKER_MEMBER, KEY, MODE_POISONED_UE, MODE_T8_EXCLUSION,
    POISON_SCALE,
};
use crate::frame::{self, Frame};
use crate::mlp;
use crate::sim::{self, Rule, RunConfig};
use crate::synthetic::{contains_row, Dataset, Task};

/// How long a new connection has to say Hello.
const HELLO_TIMEOUT: Duration = Duration::from_secs(10);
/// How long one frame write may take before the connection is dropped.
const WRITE_TIMEOUT: Duration = Duration::from_secs(10);

/// An aggregation rule as written in the config.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RuleName {
    Fedavg,
    Median,
    TrimmedMean,
    Krum,
}

impl RuleName {
    fn as_str(self) -> &'static str {
        match self {
            Self::Fedavg => "fedavg",
            Self::Median => "median",
            Self::TrimmedMean => "trimmed-mean",
            Self::Krum => "krum",
        }
    }
}

/// What a hold waits for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HoldUntil {
    /// A member has been excluded by a T8 report.
    MemberExcluded,
    /// Every expected member is joined and none is excluded.
    MembersFull,
}

/// Pause before a round until a membership condition holds (or a timeout), so a
/// check script can act at a known point: deregister `ue3` while the AF holds
/// before round 4, restart it while it holds before round 7.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Hold {
    pub before_round: u32,
    pub until: HoldUntil,
    pub timeout_s: u64,
}

/// The AF config (`nr-fl-server -c af.yaml`).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    /// The FL listener. Default 127.0.0.1:7480; the overlay binds the `fl-dn`
    /// address.
    #[serde(default = "default_listen")]
    pub listen: SocketAddr,
    /// The T8 callback listener. Default 127.0.0.1:7481. It belongs on the
    /// network nefd reaches the AF on (the SBI or pod network), never on `fl-dn`,
    /// which every UE reaches through the UPF.
    #[serde(default = "default_t8_listen")]
    pub t8_listen: SocketAddr,
    /// The base of the `notificationDestination` given to the NEF, e.g.
    /// `http://10.244.1.20:7481`. Default: `http://<t8_listen>`, which must then
    /// not be a wildcard address.
    #[serde(default)]
    pub t8_notify_base: Option<String>,
    /// The NEF.
    pub nef: NefConfig,
    /// The run seed.
    #[serde(default = "default_seed")]
    pub seed: u64,
    /// Expected members (default 3 in `t8-exclusion`, 5 in `poisoned-ue`); also
    /// the synthetic task's region count.
    #[serde(default)]
    pub members: Option<u32>,
    /// Rounds per arm.
    #[serde(default = "default_rounds")]
    pub rounds: u32,
    /// Training rows per member: the task's, and the cap on `Update.samples`.
    #[serde(default = "default_samples")]
    pub samples: usize,
    /// Held-out rows per member region.
    #[serde(default = "default_held_out")]
    pub held_out_per_client: usize,
    /// The arms, run one after another from a fresh model. Default `[fedavg]` in
    /// `t8-exclusion`, all four in `poisoned-ue`.
    #[serde(default)]
    pub rules: Option<Vec<RuleName>>,
    /// Krum's tolerated Byzantine members.
    #[serde(default = "default_krum_f")]
    pub krum_f: usize,
    /// The trimmed mean's TOTAL trimmed fraction (0.4 of 5 drops one each end).
    #[serde(default = "default_trim")]
    pub trim_ratio: f32,
    /// The straggler deadline T, per round.
    #[serde(default = "default_deadline_ms")]
    pub round_deadline_ms: u64,
    /// A pause between rounds.
    #[serde(default)]
    pub round_interval_ms: u64,
    /// How long to wait for the expected members before round 1.
    #[serde(default = "default_join_timeout")]
    pub join_timeout_s: u64,
    /// Holds before given rounds (each arm).
    #[serde(default)]
    pub holds: Vec<Hold>,
    /// Concurrent FL connections accepted.
    #[serde(default = "default_max_connections")]
    pub max_connections: usize,
    /// The largest L2 norm of one update's delta. A larger (or non-finite-norm)
    /// delta is refused (`update_refused … reason=norm`) and not aggregated.
    /// Default 1000: the seed-480 runs' largest update is about 90 (the poisoned
    /// FedAvg arm), the honest ones below 10.
    #[serde(default = "default_max_update_norm")]
    pub max_update_norm: f64,
    /// Read by `nextgsim_demo`, not here.
    #[serde(default)]
    #[allow(dead_code)]
    pub demo: serde_yaml::Value,
}

fn default_listen() -> SocketAddr {
    SocketAddr::from(([127, 0, 0, 1], 7480))
}
fn default_t8_listen() -> SocketAddr {
    SocketAddr::from(([127, 0, 0, 1], 7481))
}
fn default_seed() -> u64 {
    480
}
fn default_rounds() -> u32 {
    10
}
fn default_samples() -> usize {
    RunConfig::demo(480).samples_per_client
}
fn default_held_out() -> usize {
    RunConfig::demo(480).held_out_per_client
}
fn default_krum_f() -> usize {
    1
}
fn default_trim() -> f32 {
    0.4
}
fn default_deadline_ms() -> u64 {
    30_000
}
fn default_join_timeout() -> u64 {
    300
}
fn default_max_connections() -> usize {
    16
}
fn default_max_update_norm() -> f64 {
    1000.0
}

/// The revert-arm knobs (`revert-hooks` builds only; both off otherwise).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Knobs {
    /// `--no-t8`: no T8 subscription, so a departed member is waited for until
    /// the straggler deadline.
    pub no_t8: bool,
    /// `--freeze-global`: aggregate but never apply, so the scorecard fails.
    pub freeze_global: bool,
}

/// One round, as the AF saw it.
#[derive(Debug, Clone, PartialEq)]
pub struct RoundReport {
    pub round: u32,
    /// Selected at the start (joined, not excluded).
    pub selected: Vec<String>,
    /// Whose update was aggregated.
    pub updates: Vec<String>,
    /// Selected but no update by the deadline.
    pub stragglers: Vec<String>,
    pub dur_ms: u64,
    /// Held-out MSE after the round.
    pub mse: f32,
    /// Krum's (selected, worst) members.
    pub krum: Option<(String, String)>,
    /// Why the rule applied nothing.
    pub refused: Option<String>,
}

/// One arm (one rule over every round).
#[derive(Debug, Clone, PartialEq)]
pub struct ArmReport {
    pub rule: RuleName,
    pub rounds: Vec<RoundReport>,
    pub local_only: f32,
    pub pooled: f32,
    /// `poisoned-ue`: the same rule's last-round MSE with nobody poisoning.
    pub clean_last: Option<f32>,
    /// The final model equals, bit for bit, the in-process run with the same
    /// membership per round (same platform image only).
    pub oracle_bit_identical: bool,
    pub model: Vec<f32>,
}

impl ArmReport {
    /// Round 1's MSE.
    pub fn first(&self) -> f32 {
        self.rounds.first().map_or(f32::NAN, |r| r.mse)
    }
    /// The last round's MSE.
    pub fn last(&self) -> f32 {
        self.rounds.last().map_or(f32::NAN, |r| r.mse)
    }
}

/// The whole run.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RunReport {
    pub arms: Vec<ArmReport>,
    /// `(member, gpsi)` excluded by T8, in order.
    pub exclusions: Vec<(String, String)>,
    /// Members readmitted, in order.
    pub readmissions: Vec<String>,
    /// Members whose `samples` claim was capped.
    pub capped: Vec<String>,
    /// Refused frames, `member-or-peer: why`.
    pub refused_frames: Vec<String>,
    /// Updates refused for their norm, `member round=<r>`.
    pub refused_updates: Vec<String>,
    /// Joins refused for an identity clash, `label gpsi`.
    pub identity_mismatches: Vec<String>,
    /// Canary rows found in received bytes.
    pub canary_hits: u64,
}

/// What the AF announces, for tests and the check script's ordering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Progress {
    Joined(String),
    Holding(u32),
    RoundStarted { arm: usize, round: u32 },
    Excluded(String),
    Readmitted(String),
    RoundDone { arm: usize, round: u32 },
    Finished,
}

enum Event {
    Hello {
        conn: u64,
        label: String,
        peer: SocketAddr,
        out: mpsc::Sender<Out>,
    },
    Update {
        conn: u64,
        round: u32,
        samples: u32,
        delta: Vec<f32>,
    },
    Closed {
        conn: u64,
        why: String,
    },
    Refused {
        conn: u64,
        why: String,
    },
    /// The NEF's answer to UE ID retrieval for a Hello.
    Identified {
        conn: u64,
        label: String,
        result: Result<Gpsi, String>,
    },
    /// The T8 subscription for an identified member (or why it failed).
    Admitted {
        conn: u64,
        label: String,
        result: Result<(Gpsi, Option<Subscription>), String>,
    },
    T8 {
        callback_id: String,
        body: Bytes,
    },
}

enum Out {
    Frame(Vec<u8>),
    Close,
}

struct Conn {
    peer: SocketAddr,
    out: mpsc::Sender<Out>,
}

struct Member {
    gpsi: Gpsi,
    conn: Option<u64>,
    excluded: bool,
    sub: Option<Subscription>,
}

struct RoundState {
    round: u32,
    awaiting: BTreeSet<String>,
    updates: BTreeMap<String, (Vec<f32>, u64)>,
}

/// The AF.
pub struct Af {
    cfg: ServerConfig,
    mode: String,
    expected: u32,
    rules: Vec<RuleName>,
    knobs: Knobs,
    nef: Arc<Nef>,
    notify_base: Option<String>,
    ids: CallbackIds,
    tx: mpsc::Sender<Event>,
    rx: mpsc::Receiver<Event>,
    t8_tx: mpsc::Sender<(String, Bytes)>,
    t8_rx: mpsc::Receiver<(String, Bytes)>,
    progress: broadcast::Sender<Progress>,
    conns: HashMap<u64, Conn>,
    members: BTreeMap<String, Member>,
    round: Option<RoundState>,
    canaries: Arc<Vec<Vec<f32>>>,
    canary_hits: Arc<AtomicU64>,
    held_out: Dataset,
    report: RunReport,
}

/// Check `cfg` against `mode`: the member count, the rules and the T8 callback
/// base. Returns the expected members, the arms and the advertised base.
fn check(
    cfg: &ServerConfig,
    mode: &str,
    knobs: Knobs,
) -> Result<(u32, Vec<RuleName>, Option<String>), String> {
    let (members, rules) = match mode {
        MODE_T8_EXCLUSION => (
            cfg.members.unwrap_or(3),
            cfg.rules.clone().unwrap_or(vec![RuleName::Fedavg]),
        ),
        MODE_POISONED_UE => (
            cfg.members.unwrap_or(5),
            cfg.rules.clone().unwrap_or(vec![
                RuleName::Fedavg,
                RuleName::Median,
                RuleName::TrimmedMean,
                RuleName::Krum,
            ]),
        ),
        other => return Err(format!("unknown mode {other:?}")),
    };
    if !(1..=64).contains(&members) {
        return Err(format!("members {members} must be in 1..=64"));
    }
    if mode == MODE_POISONED_UE && members < ATTACKER_MEMBER {
        return Err(format!(
            "poisoned-ue needs at least {ATTACKER_MEMBER} members (UE 005 is the attacker), got {members}"
        ));
    }
    if rules.is_empty() {
        return Err("rules must name at least one rule".to_string());
    }
    if !(1..=1000).contains(&cfg.rounds) {
        return Err(format!("rounds {} must be in 1..=1000", cfg.rounds));
    }
    if cfg.samples == 0 || cfg.samples > 1_000_000 {
        return Err(format!("samples {} must be in 1..=1000000", cfg.samples));
    }
    if cfg.held_out_per_client == 0 {
        return Err("held_out_per_client must be at least 1".to_string());
    }
    if !(0.0..1.0).contains(&cfg.trim_ratio) {
        return Err(format!("trim_ratio {} must be in [0, 1)", cfg.trim_ratio));
    }
    if !(cfg.max_update_norm.is_finite() && cfg.max_update_norm > 0.0) {
        return Err(format!(
            "max_update_norm {} must be finite and positive",
            cfg.max_update_norm
        ));
    }
    if cfg.round_deadline_ms == 0 || cfg.max_connections == 0 {
        return Err("round_deadline_ms and max_connections must be at least 1".to_string());
    }
    if rules.contains(&RuleName::Krum) && !crate::krum_within_bound(members as usize, cfg.krum_f) {
        tracing::warn!(
            "[demo:{KEY}] krum with {members} members and f={} is below its n > 2f + 2 bound: \
             every krum round is refused and names nobody (exclusion lines come from T8 only)",
            cfg.krum_f
        );
    }
    cfg.nef.validate()?;
    let base = if knobs.no_t8 {
        None
    } else {
        let base = match &cfg.t8_notify_base {
            Some(b) => b.trim_end_matches('/').to_string(),
            None if nef::advertisable(&cfg.t8_listen) => format!("http://{}", cfg.t8_listen),
            None => {
                return Err(format!(
                    "t8_listen {} is a wildcard address: set t8_notify_base to the address the \
                     NEF reaches this AF on",
                    cfg.t8_listen
                ))
            }
        };
        let host = base
            .strip_prefix("http://")
            .ok_or_else(|| format!("t8_notify_base {base:?} must be http://"))?;
        let host_only = host.rsplit_once(':').map_or(host, |(h, _)| h);
        if host_only.is_empty() || ["0.0.0.0", "[::]", "::"].contains(&host_only) {
            return Err(format!(
                "t8_notify_base {base:?} must name a reachable host, not a wildcard"
            ));
        }
        Some(base)
    };
    Ok((members, rules, base))
}

impl Af {
    /// An AF for `cfg` in `mode`. Fails on a config that cannot run.
    pub async fn new(cfg: ServerConfig, mode: &str, knobs: Knobs) -> Result<Self, String> {
        let (members, rules, notify_base) = check(&cfg, mode, knobs)?;
        if knobs.no_t8 || knobs.freeze_global {
            demo_log!(
                warn,
                KEY,
                "REVERT ARM: no_t8={} freeze_global={} (revert-hooks build)",
                knobs.no_t8,
                knobs.freeze_global
            );
        }
        let nef = Arc::new(Nef::new(cfg.nef.clone()).await?);
        let task = Task::new(cfg.seed, members);
        let canaries = (0..members)
            .flat_map(|c| {
                task.client_data(c, cfg.samples)
                    .canaries()
                    .map(<[f32]>::to_vec)
                    .collect::<Vec<_>>()
            })
            .collect();
        let held_out = task.held_out(cfg.held_out_per_client);
        let (tx, rx) = mpsc::channel(1024);
        let (t8_tx, t8_rx) = mpsc::channel(64);
        let (progress, _) = broadcast::channel(256);
        Ok(Self {
            cfg,
            mode: mode.to_string(),
            expected: members,
            rules,
            knobs,
            nef,
            notify_base,
            ids: CallbackIds::default(),
            tx,
            rx,
            t8_tx,
            t8_rx,
            progress,
            conns: HashMap::new(),
            members: BTreeMap::new(),
            round: None,
            canaries: Arc::new(canaries),
            canary_hits: Arc::new(AtomicU64::new(0)),
            held_out,
            report: RunReport::default(),
        })
    }

    /// Progress announcements.
    pub fn subscribe(&self) -> broadcast::Receiver<Progress> {
        self.progress.subscribe()
    }

    fn announce(&self, p: Progress) {
        let _ = self.progress.send(p);
    }

    fn expected(&self) -> u32 {
        self.expected
    }

    fn rules(&self) -> Vec<RuleName> {
        self.rules.clone()
    }

    fn rule(&self, r: RuleName) -> Rule {
        match r {
            RuleName::Fedavg => Rule::FedAvg,
            RuleName::Median => Rule::Median,
            RuleName::TrimmedMean => Rule::TrimmedMean(self.cfg.trim_ratio),
            RuleName::Krum => Rule::Krum(self.cfg.krum_f),
        }
    }

    fn run_config(&self) -> RunConfig {
        RunConfig {
            seed: self.cfg.seed,
            clients: self.expected(),
            samples_per_client: self.cfg.samples,
            held_out_per_client: self.cfg.held_out_per_client,
            rounds: self.cfg.rounds,
            ..RunConfig::demo(self.cfg.seed)
        }
    }

    /// Serve both listeners and run every arm. Returns the run's report; the
    /// listeners keep running until the returned future's task is dropped.
    pub async fn run(mut self, fl: TcpListener, t8: Option<TcpListener>) -> RunReport {
        demo_log!(
            info,
            KEY,
            "nr-fl-server mode={} seed={} members={} rounds={} rules={:?} T={} ms; FL on {}, T8 \
             callbacks {}",
            self.mode,
            self.cfg.seed,
            self.expected(),
            self.cfg.rounds,
            self.rules().iter().map(|r| r.as_str()).collect::<Vec<_>>(),
            self.cfg.round_deadline_ms,
            fl.local_addr()
                .map_or_else(|e| e.to_string(), |a| a.to_string()),
            match (&t8, &self.notify_base) {
                (Some(l), Some(b)) => format!(
                    "on {} advertised as {b}",
                    l.local_addr()
                        .map_or_else(|e| e.to_string(), |a| a.to_string())
                ),
                _ => "off (--no-t8)".to_string(),
            }
        );
        tokio::spawn(accept_loop(
            fl,
            self.cfg.max_connections,
            self.tx.clone(),
            self.canaries.clone(),
            self.canary_hits.clone(),
        ));
        if let Some(t8) = t8 {
            tokio::spawn(nef::serve_callbacks(
                t8,
                self.ids.clone(),
                self.t8_tx.clone(),
            ));
        }

        let join_deadline = Instant::now() + Duration::from_secs(self.cfg.join_timeout_s);
        let expected = self.expected() as usize;
        self.pump_until(join_deadline, |af| af.admitted().len() >= expected)
            .await;
        demo_log!(
            info,
            KEY,
            "joined {}/{} members: {}",
            self.admitted().len(),
            expected,
            self.admitted().join(",")
        );

        for (arm, rule) in self.rules().into_iter().enumerate() {
            let report = self.run_arm(arm, rule).await;
            self.report.arms.push(report);
        }
        self.report.canary_hits = self.canary_hits.load(Ordering::Relaxed);
        demo_log!(
            info,
            KEY,
            "all arms done: canary_rows={} refused_frames={} exclusions={} readmissions={}",
            self.report.canary_hits,
            self.report.refused_frames.len(),
            self.report.exclusions.len(),
            self.report.readmissions.len()
        );
        self.announce(Progress::Finished);
        self.report
    }

    /// Members joined (identity known), sorted.
    fn admitted(&self) -> Vec<String> {
        self.members.keys().cloned().collect()
    }

    fn selected(&self) -> Vec<String> {
        self.members
            .iter()
            .filter(|(_, m)| !m.excluded)
            .map(|(l, _)| l.clone())
            .collect()
    }

    async fn run_arm(&mut self, arm: usize, name: RuleName) -> ArmReport {
        let rule = self.rule(name);
        let mut global = mlp::init(self.cfg.seed);
        let mut rounds = Vec::new();
        demo_log!(
            info,
            KEY,
            "arm {} rule={} starts from the seeded initial model",
            arm + 1,
            name.as_str()
        );
        for r in 1..=self.cfg.rounds {
            for hold in self
                .cfg
                .holds
                .clone()
                .iter()
                .filter(|h| h.before_round == r)
            {
                self.hold(*hold).await;
            }
            let rep = self.run_round(arm, name, rule, r, &mut global).await;
            rounds.push(rep);
            self.announce(Progress::RoundDone { arm, round: r });
            if self.cfg.round_interval_ms > 0 && r < self.cfg.rounds {
                let until = Instant::now() + Duration::from_millis(self.cfg.round_interval_ms);
                self.pump_until(until, |_| false).await;
            }
        }
        self.score(name, rule, rounds, global)
    }

    async fn hold(&mut self, hold: Hold) {
        let expected = self.expected() as usize;
        let cond = move |af: &Af| match hold.until {
            HoldUntil::MemberExcluded => af.members.values().any(|m| m.excluded),
            HoldUntil::MembersFull => af.selected().len() >= expected,
        };
        let what = match hold.until {
            HoldUntil::MemberExcluded => "member-excluded",
            HoldUntil::MembersFull => "members-full",
        };
        demo_log!(
            info,
            KEY,
            "hold before round={} until={what} (timeout {} s)",
            hold.before_round,
            hold.timeout_s
        );
        self.announce(Progress::Holding(hold.before_round));
        let until = Instant::now() + Duration::from_secs(hold.timeout_s);
        if self.pump_until(until, cond).await {
            demo_log!(
                info,
                KEY,
                "hold before round={} released: {what}",
                hold.before_round
            );
        } else {
            demo_log!(
                warn,
                KEY,
                "hold before round={} timed out after {} s waiting for {what}",
                hold.before_round,
                hold.timeout_s
            );
        }
    }

    async fn run_round(
        &mut self,
        arm: usize,
        name: RuleName,
        rule: Rule,
        r: u32,
        global: &mut [f32],
    ) -> RoundReport {
        let started = Instant::now();
        let selected = self.selected();
        let excluded: Vec<String> = self
            .members
            .iter()
            .filter(|(_, m)| m.excluded)
            .map(|(l, m)| format!("{}({l})", m.gpsi))
            .collect();
        demo_log!(
            info,
            KEY,
            "round={r} arm={} rule={} participants={} selected={} excluded={}{}",
            arm + 1,
            name.as_str(),
            selected.len(),
            selected.join(","),
            if excluded.is_empty() {
                "-".to_string()
            } else {
                excluded.join(",")
            },
            if excluded.is_empty() {
                ""
            } else {
                " reason=nef_event"
            }
        );
        self.announce(Progress::RoundStarted { arm, round: r });
        // The global model only ever takes finite steps (`sim::apply`), so this
        // cannot fail; if it ever did, the round goes out to nobody and is
        // refused with "no updates" rather than taking the AF down.
        let sent: Vec<String> = match frame::encode(&Frame::GlobalModel {
            round: r,
            weights: global.to_vec(),
        }) {
            Ok(bytes) => {
                for label in &selected {
                    if let Some(conn) = self.members.get(label).and_then(|m| m.conn) {
                        if let Some(c) = self.conns.get(&conn) {
                            if c.out.try_send(Out::Frame(bytes.clone())).is_err() {
                                tracing::debug!(
                                    "[demo:{KEY}] {label}: output queue full or closed"
                                );
                            }
                        }
                    }
                }
                selected.clone()
            }
            Err(e) => {
                demo_log!(
                    error,
                    KEY,
                    "round={r}: the global model cannot be sent: {e:?}"
                );
                Vec::new()
            }
        };
        self.round = Some(RoundState {
            round: r,
            awaiting: sent.into_iter().collect(),
            updates: BTreeMap::new(),
        });
        let deadline = started + Duration::from_millis(self.cfg.round_deadline_ms);
        self.pump_until(deadline, |af| {
            af.round.as_ref().is_some_and(|s| s.awaiting.is_empty())
        })
        .await;
        let state = self.round.take().expect("the round state");
        let stragglers: Vec<String> = state.awaiting.iter().cloned().collect();
        for s in &stragglers {
            demo_log!(
                warn,
                KEY,
                "straggler_timeout round={r} member={s} after {} ms",
                self.cfg.round_deadline_ms
            );
        }
        let (krum, refused) = if state.updates.is_empty() {
            (None, Some("no updates".to_string()))
        } else {
            match sim::aggregate(rule, &state.updates) {
                Ok((_, krum)) if self.knobs.freeze_global => (krum, None),
                Ok((agg, krum)) => match sim::apply(global, &agg) {
                    Ok(()) => (krum, None),
                    // The previous global carries over, as in `sim::run`.
                    Err(why) => (krum, Some(why)),
                },
                Err(why) => (None, Some(why)),
            }
        };
        let mse = mlp::mse(global, &self.held_out.xs, &self.held_out.ys);
        let dur_ms = started.elapsed().as_millis() as u64;
        let updates: Vec<String> = state.updates.keys().cloned().collect();
        demo_log!(
            info,
            KEY,
            "round={r} done arm={} rule={} updates={} from={} dur_ms={dur_ms} mse={mse:.6}{}",
            arm + 1,
            name.as_str(),
            updates.len(),
            updates.join(","),
            refused
                .as_ref()
                .map(|w| format!(" refused=\"{w}\""))
                .unwrap_or_default()
        );
        if let Some((sel, worst)) = &krum {
            let gpsi = |l: &str| {
                self.members
                    .get(l)
                    .map_or_else(|| l.to_string(), |m| m.gpsi.to_string())
            };
            demo_log!(
                info,
                KEY,
                "member_selection round={r} rule=krum excluded={} member={worst} reason=krum_worst \
                 selected={} ({sel})",
                gpsi(worst),
                gpsi(sel)
            );
        }
        RoundReport {
            round: r,
            selected,
            updates,
            stragglers,
            dur_ms,
            mse,
            krum,
            refused,
        }
    }

    fn score(
        &self,
        name: RuleName,
        rule: Rule,
        rounds: Vec<RoundReport>,
        model: Vec<f32>,
    ) -> ArmReport {
        let cfg = self.run_config();
        let local_only = sim::local_only_mse(&cfg);
        let pooled = sim::pooled_mse(&cfg);
        let poisoned = self.mode == MODE_POISONED_UE;
        let clean_last = poisoned.then(|| {
            let n = cfg.clients;
            sim::run(&cfg, rule, |_| (0..n).collect())
                .rounds
                .last()
                .map_or(f32::NAN, |r| r.mse)
        });
        // Replay the membership the AF saw, in process.
        let per_round: Vec<Vec<u32>> = rounds
            .iter()
            .map(|r| {
                r.updates
                    .iter()
                    .filter_map(|l| member_of(l))
                    .map(|m| m - 1)
                    .collect()
            })
            .collect();
        let oracle_cfg = RunConfig {
            poisoned: poisoned.then_some((ATTACKER_MEMBER - 1, POISON_SCALE)),
            ..cfg
        };
        let oracle = sim::run(&oracle_cfg, rule, |r| per_round[(r - 1) as usize].clone());
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let oracle_bit_identical = !self.knobs.freeze_global && bits(&oracle.model) == bits(&model);
        let report = ArmReport {
            rule: name,
            rounds,
            local_only,
            pooled,
            clean_last,
            oracle_bit_identical,
            model,
        };
        let (first, last) = (report.first(), report.last());
        demo_log!(
            info,
            KEY,
            "scorecard rule={} seed={} rounds={} r1_mse={first:.6} last_mse={last:.6} \
             local_only_mse={local_only:.6} pooled_mse={pooled:.6} last/r1={:.3} last/local={:.3} \
             last/pooled={:.3} oracle_bit_identical={oracle_bit_identical}",
            name.as_str(),
            self.cfg.seed,
            report.rounds.len(),
            last / first,
            last / local_only,
            last / pooled
        );
        if let Some(clean) = clean_last {
            demo_log!(
                info,
                KEY,
                "robustness rule={} poisoned_last_mse={last:.6} clean_last_mse={clean:.6} ratio={:.3} \
                 (attacker {} x-{POISON_SCALE})",
                name.as_str(),
                last / clean,
                super::label(ATTACKER_MEMBER)
            );
        }
        report
    }

    /// Process events until `cond` holds (checked first) or `until`. Returns
    /// whether `cond` held.
    async fn pump_until(&mut self, until: Instant, cond: impl Fn(&Af) -> bool) -> bool {
        loop {
            if cond(self) {
                return true;
            }
            let sleep = tokio::time::sleep_until(until.into());
            tokio::select! {
                ev = self.rx.recv() => match ev {
                    Some(ev) => self.handle(ev),
                    None => return cond(self),
                },
                cb = self.t8_rx.recv() => {
                    if let Some((id, body)) = cb {
                        self.handle(Event::T8 { callback_id: id, body });
                    }
                }
                () = sleep => return cond(self),
            }
        }
    }

    fn close(&mut self, conn: u64) {
        if let Some(c) = self.conns.remove(&conn) {
            let _ = c.out.try_send(Out::Close);
        }
    }

    fn label_of(&self, conn: u64) -> Option<String> {
        self.members
            .iter()
            .find(|(_, m)| m.conn == Some(conn))
            .map(|(l, _)| l.clone())
    }

    fn handle(&mut self, ev: Event) {
        match ev {
            Event::Hello {
                conn,
                label,
                peer,
                out,
            } => self.on_hello(conn, label, peer, out),
            Event::Identified {
                conn,
                label,
                result,
            } => self.on_identified(conn, label, result),
            Event::Admitted {
                conn,
                label,
                result,
            } => self.on_admitted(conn, label, result),
            Event::Update {
                conn,
                round,
                samples,
                delta,
            } => self.on_update(conn, round, samples, delta),
            Event::Closed { conn, why } => {
                let peer = self.conns.get(&conn).map(|c| c.peer);
                if let Some(label) = self.label_of(conn) {
                    demo_log!(
                        info,
                        KEY,
                        "transport_closed member={label} ({why}); not an exclusion: the member \
                         stays selected until a T8 report or the straggler deadline"
                    );
                    if let Some(m) = self.members.get_mut(&label) {
                        m.conn = None;
                    }
                } else if let Some(peer) = peer {
                    tracing::debug!(
                        "[demo:{KEY}] connection from {peer} closed before joining: {why}"
                    );
                }
                self.conns.remove(&conn);
            }
            Event::Refused { conn, why } => {
                let who = self
                    .label_of(conn)
                    .or_else(|| self.conns.get(&conn).map(|c| c.peer.to_string()))
                    .unwrap_or_else(|| format!("conn {conn}"));
                demo_log!(
                    warn,
                    KEY,
                    "frame_refused member={who}: {why}; connection closed"
                );
                self.report.refused_frames.push(format!("{who}: {why}"));
                if let Some(label) = self.label_of(conn) {
                    if let Some(m) = self.members.get_mut(&label) {
                        m.conn = None;
                    }
                }
                self.close(conn);
            }
            Event::T8 { callback_id, body } => self.on_t8(&callback_id, &body),
        }
    }

    fn on_hello(&mut self, conn: u64, label: String, peer: SocketAddr, out: mpsc::Sender<Out>) {
        if member_of(&label).is_none_or(|m| m > self.expected()) {
            demo_log!(
                warn,
                KEY,
                "refused {peer}: label {label:?} is not a member of this run"
            );
            let _ = out.try_send(Out::Close);
            return;
        }
        self.conns.insert(conn, Conn { peer, out });
        let (nef, tx) = (self.nef.clone(), self.tx.clone());
        tokio::spawn(async move {
            let result = identify(&nef, peer).await;
            let _ = tx
                .send(Event::Identified {
                    conn,
                    label,
                    result,
                })
                .await;
        });
    }

    /// Who else holds `label` or `gpsi`: a label must keep its GPSI, and a GPSI
    /// its label (one UE, one weight; nobody takes over an excluded member).
    fn clash(&self, label: &str, gpsi: &Gpsi) -> Option<String> {
        self.members
            .iter()
            .find(|(l, m)| (l.as_str() == label) != (m.gpsi == *gpsi))
            .map(|(l, m)| format!("{l} is {}", m.gpsi))
    }

    /// Refuse `conn` for an identity clash.
    fn refuse_mismatch(&mut self, conn: u64, label: &str, gpsi: &Gpsi, clash: &str) {
        let peer = self
            .conns
            .get(&conn)
            .map_or_else(|| "?".to_string(), |c| c.peer.to_string());
        demo_log!(
            warn,
            KEY,
            "identity_mismatch member={label} gpsi={gpsi} peer={peer}: {clash}; refused"
        );
        self.report
            .identity_mismatches
            .push(format!("{label} {gpsi}"));
        self.close(conn);
    }

    /// Step 2 of a join: the NEF has said who `conn` is. A clash is refused here,
    /// before any T8 subscription is made for it.
    fn on_identified(&mut self, conn: u64, label: String, result: Result<Gpsi, String>) {
        let gpsi = match result {
            Ok(g) => g,
            Err(why) => return self.on_admitted(conn, label, Err(why)),
        };
        if !self.conns.contains_key(&conn) {
            return;
        }
        if let Some(clash) = self.clash(&label, &gpsi) {
            return self.refuse_mismatch(conn, &label, &gpsi, &clash);
        }
        let Some(base) = self.notify_base.clone() else {
            return self.on_admitted(conn, label, Ok((gpsi, None)));
        };
        let (nef, ids, tx) = (self.nef.clone(), self.ids.clone(), self.tx.clone());
        tokio::spawn(async move {
            let result = subscribe(&nef, &ids, &label, &gpsi, &base).await;
            let _ = tx
                .send(Event::Admitted {
                    conn,
                    label,
                    result,
                })
                .await;
        });
    }

    fn drop_subscription(&self, sub: Subscription) {
        self.ids.remove(&sub.callback_id);
        let nef = self.nef.clone();
        tokio::spawn(async move {
            if let Err(e) = nef.unsubscribe(&sub).await {
                tracing::warn!("[demo:{KEY}] {e}");
            }
        });
    }

    fn on_admitted(
        &mut self,
        conn: u64,
        label: String,
        result: Result<(Gpsi, Option<Subscription>), String>,
    ) {
        let peer = self.conns.get(&conn).map(|c| c.peer);
        let (gpsi, sub) = match result {
            Err(why) => {
                demo_log!(
                    warn,
                    KEY,
                    "identity_unresolved member={label} peer={}: {why}; refused",
                    peer.map_or_else(|| "?".to_string(), |p| p.to_string())
                );
                self.close(conn);
                return;
            }
            Ok(v) => v,
        };
        let Some(peer) = peer else {
            // The connection went away while the NEF was asked.
            if let Some(sub) = sub {
                self.drop_subscription(sub);
            }
            return;
        };
        // Checked again: another join may have completed while subscribing.
        if let Some(clash) = self.clash(&label, &gpsi) {
            if let Some(sub) = sub {
                self.drop_subscription(sub);
            }
            return self.refuse_mismatch(conn, &label, &gpsi, &clash);
        }
        let t8 = sub.as_ref().map_or_else(
            || "off".to_string(),
            |s| format!("subscribed {}", s.location),
        );
        let old = self.members.remove(&label);
        let readmitted = old.as_ref().is_some_and(|m| m.excluded);
        if let Some(old) = old {
            if let Some(c) = old.conn.filter(|c| *c != conn) {
                tracing::info!("[demo:{KEY}] {label}: a new connection replaces the old one");
                self.close(c);
            }
            if let Some(s) = old.sub {
                self.drop_subscription(s);
            }
        }
        self.members.insert(
            label.clone(),
            Member {
                gpsi: gpsi.clone(),
                conn: Some(conn),
                excluded: false,
                sub,
            },
        );
        if readmitted {
            demo_log!(
                info,
                KEY,
                "member_selection readmitted={gpsi} member={label} peer={peer} t8={t8}"
            );
            self.report.readmissions.push(label.clone());
            self.announce(Progress::Readmitted(label));
        } else {
            demo_log!(
                info,
                KEY,
                "member_joined member={label} gpsi={gpsi} peer={peer} t8={t8}"
            );
            self.announce(Progress::Joined(label));
        }
    }

    fn on_update(&mut self, conn: u64, round: u32, samples: u32, delta: Vec<f32>) {
        let Some(label) = self.label_of(conn) else {
            tracing::debug!("[demo:{KEY}] update on a connection that has not joined; ignored");
            return;
        };
        let cap = self.cfg.samples as u64;
        let Some(state) = self.round.as_mut() else {
            tracing::info!("[demo:{KEY}] stale_update member={label} round={round}: no round open");
            return;
        };
        if state.round != round || !state.awaiting.remove(&label) {
            tracing::info!(
                "[demo:{KEY}] stale_update member={label} round={round} (open round {}, awaited={})",
                state.round,
                state.awaiting.contains(&label)
            );
            return;
        }
        let norm = delta
            .iter()
            .map(|x| f64::from(*x) * f64::from(*x))
            .sum::<f64>()
            .sqrt();
        if norm.is_nan() || norm > self.cfg.max_update_norm {
            demo_log!(
                warn,
                KEY,
                "update_refused member={label} round={round} reason=norm norm={norm:.3e} cap={}",
                self.cfg.max_update_norm
            );
            self.report
                .refused_updates
                .push(format!("{label} round={round}"));
            return;
        }
        let weight = u64::from(samples).min(cap);
        if u64::from(samples) > cap {
            demo_log!(
                warn,
                KEY,
                "samples_capped member={label} round={round} claimed={samples} cap={cap}"
            );
            if !self.report.capped.contains(&label) {
                self.report.capped.push(label.clone());
            }
        }
        state.updates.insert(label, (delta, weight));
    }

    fn on_t8(&mut self, callback_id: &str, body: &[u8]) {
        let report = match nef::parse_loc_notification(callback_id, body) {
            Ok(r) => r,
            Err(why) => {
                demo_log!(warn, KEY, "T8 notification ignored: {why}");
                return;
            }
        };
        let Some((label, m)) = self
            .members
            .iter_mut()
            .find(|(_, m)| m.sub.as_ref().is_some_and(|s| s.callback_id == callback_id))
        else {
            demo_log!(
                info,
                KEY,
                "T8 notification for a subscription no longer in use; ignored"
            );
            return;
        };
        let location = m
            .sub
            .as_ref()
            .map(|s| s.location.clone())
            .expect("matched on the subscription");
        if report.subscription != location && !report.subscription.ends_with(&location) {
            demo_log!(
                warn,
                KEY,
                "T8 notification for {label} names subscription {:?}, not ours ({location}); ignored",
                report.subscription
            );
            return;
        }
        let reason = report.reason.as_ref().map_or_else(
            || "-".to_string(),
            |v| v.as_str().map_or_else(|| v.to_string(), str::to_string),
        );
        if m.excluded {
            demo_log!(
                info,
                KEY,
                "T8 LOSS_OF_CONNECTIVITY for {label} again (lossOfConnectReason={reason})"
            );
            return;
        }
        m.excluded = true;
        let label = label.clone();
        let gpsi = m.gpsi.to_string();
        demo_log!(
            info,
            KEY,
            "member_selection excluded={gpsi} member={label} reason=nef_event \
             lossOfConnectReason={reason} subscription={location}"
        );
        if let Some(state) = self.round.as_mut() {
            if state.awaiting.remove(&label) {
                demo_log!(
                    info,
                    KEY,
                    "round={} no longer waits for {label}: excluded",
                    state.round
                );
            }
        }
        self.report.exclusions.push((label.clone(), gpsi));
        self.announce(Progress::Excluded(label));
    }
}

/// Ask the NEF who is behind `peer`.
async fn identify(nef: &Nef, peer: SocketAddr) -> Result<Gpsi, String> {
    let ip = match peer.ip() {
        IpAddr::V4(ip) => ip,
        IpAddr::V6(ip) => ip
            .to_ipv4_mapped()
            .ok_or_else(|| format!("{peer} is not an IPv4 UE address"))?,
    };
    nef.ue_id(ip).await
}

/// Subscribe to `gpsi`'s LoC, notifications to a fresh callback id on `base`.
async fn subscribe(
    nef: &Nef,
    ids: &CallbackIds,
    label: &str,
    gpsi: &Gpsi,
    base: &str,
) -> Result<(Gpsi, Option<Subscription>), String> {
    let id = nef::new_callback_id();
    // Registered first: the NEF may notify before its 201 arrives here.
    ids.insert(&id, label);
    match nef.subscribe_loc(gpsi, base, &id).await {
        Ok(sub) => Ok((gpsi.clone(), Some(sub))),
        Err(e) => {
            ids.remove(&id);
            Err(e)
        }
    }
}

async fn accept_loop(
    listener: TcpListener,
    max: usize,
    tx: mpsc::Sender<Event>,
    canaries: Arc<Vec<Vec<f32>>>,
    hits: Arc<AtomicU64>,
) {
    let slots = Arc::new(Semaphore::new(max));
    let mut next_id = 0u64;
    loop {
        let (tcp, peer) = match listener.accept().await {
            Ok(a) => a,
            Err(e) => {
                tracing::warn!("[demo:{KEY}] FL listener: accept: {e}");
                tokio::time::sleep(Duration::from_millis(200)).await;
                continue;
            }
        };
        let Ok(permit) = slots.clone().try_acquire_owned() else {
            tracing::warn!("[demo:{KEY}] FL listener: {peer} refused: {max} connections open");
            continue;
        };
        next_id += 1;
        let (tx, canaries, hits) = (tx.clone(), canaries.clone(), hits.clone());
        let conn = next_id;
        tokio::spawn(async move {
            let _permit = permit;
            connection(conn, tcp, peer, tx, canaries, hits).await;
        });
    }
}

fn scan(raw: &[u8], canaries: &[Vec<f32>], hits: &AtomicU64, peer: SocketAddr) {
    let n = canaries.iter().filter(|c| contains_row(raw, c)).count() as u64;
    if n > 0 {
        hits.fetch_add(n, Ordering::Relaxed);
        demo_log!(
            warn,
            KEY,
            "canary_hit peer={peer}: {n} training row(s) in {} received bytes",
            raw.len()
        );
    }
}

/// One FL connection: Hello, then frames in and out until either side ends it.
async fn connection(
    conn: u64,
    tcp: TcpStream,
    peer: SocketAddr,
    tx: mpsc::Sender<Event>,
    canaries: Arc<Vec<Vec<f32>>>,
    hits: Arc<AtomicU64>,
) {
    let _ = tcp.set_nodelay(true);
    let (mut rd, mut wr) = tcp.into_split();
    let mut reader = FrameReader::default();
    let (out_tx, mut out_rx) = mpsc::channel::<Out>(8);
    // The AF learns of the connection with its Hello; until then it has no handle.
    let label = match tokio::time::timeout(HELLO_TIMEOUT, reader.next(&mut rd)).await {
        Ok(Ok(Some((Frame::Hello { label }, raw)))) => {
            scan(&raw, &canaries, &hits, peer);
            label
        }
        Ok(Err(ReadError::Frame { error, raw })) => {
            scan(&raw, &canaries, &hits, peer);
            demo_log!(
                warn,
                KEY,
                "frame_refused peer={peer}: {error} before Hello; connection closed"
            );
            return;
        }
        other => {
            tracing::info!(
                "[demo:{KEY}] {peer}: no Hello ({})",
                match other {
                    Err(_) => "timeout".to_string(),
                    Ok(Ok(Some((f, _)))) => format!("first frame {f:?}"),
                    Ok(Ok(None)) => "closed".to_string(),
                    Ok(Err(e)) => e.to_string(),
                }
            );
            return;
        }
    };
    if tx
        .send(Event::Hello {
            conn,
            label,
            peer,
            out: out_tx,
        })
        .await
        .is_err()
    {
        return;
    }
    loop {
        tokio::select! {
            r = reader.next(&mut rd) => match r {
                Ok(Some((Frame::Update { round, samples, delta, .. }, raw))) => {
                    scan(&raw, &canaries, &hits, peer);
                    if tx.send(Event::Update { conn, round, samples, delta }).await.is_err() {
                        return;
                    }
                }
                Ok(Some((other, raw))) => {
                    scan(&raw, &canaries, &hits, peer);
                    let kind = match other {
                        Frame::Hello { .. } => "a second Hello",
                        Frame::GlobalModel { .. } => "a GlobalModel",
                        Frame::Update { .. } => "an Update",
                    };
                    let _ = tx.send(Event::Refused { conn, why: format!("unexpected {kind}") }).await;
                    return;
                }
                Ok(None) => {
                    let _ = tx.send(Event::Closed { conn, why: "EOF".to_string() }).await;
                    return;
                }
                Err(ReadError::Io(e)) => {
                    let _ = tx.send(Event::Closed { conn, why: e.to_string() }).await;
                    return;
                }
                Err(ReadError::Frame { error, raw }) => {
                    scan(&raw, &canaries, &hits, peer);
                    let _ = tx.send(Event::Refused { conn, why: error.to_string() }).await;
                    return;
                }
            },
            o = out_rx.recv() => match o {
                Some(Out::Frame(bytes)) => {
                    match tokio::time::timeout(WRITE_TIMEOUT, wr.write_all(&bytes)).await {
                        Ok(Ok(())) => {}
                        Ok(Err(e)) => {
                            let _ = tx.send(Event::Closed { conn, why: format!("write: {e}") }).await;
                            return;
                        }
                        Err(_) => {
                            let _ = tx.send(Event::Closed { conn, why: "write timed out".to_string() }).await;
                            return;
                        }
                    }
                }
                Some(Out::Close) | None => return,
            },
        }
    }
}

#[cfg(test)]
#[path = "server_tests.rs"]
mod tests;
