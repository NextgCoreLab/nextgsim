//! The D440A seeded replay (nextgsim#462): every mode on the same committed
//! route seeds, through the same [`recommend`] the live NWDAF task runs.
//!
//! For each seed of `config/demo-ai-paging/d440a-scenario.yaml` and each
//! route family ([`Family`]):
//! 1. the route is built from the seed (an arc sampled into `sim_route`
//!    waypoints) and the UE's position is read from it with
//!    [`SimRoute::position_at`], the function the live UE's heartbeat uses;
//! 2. the UE camps by the R-criterion the UE implements (q_hyst, Treselection,
//!    the distance model's -1 dB per metre), sampled once a second;
//! 3. while connected, each second's position is recorded into a fresh
//!    [`NwdafManager`] as the serving gNB's heartbeat history;
//! 4. at release the serving gNB's [`recommend`] gives the recommended gNBs;
//! 5. at page time the UE is camped on some gNB. The AMF is modelled as
//!    nextgcore amfd `paging.strategy: recommended-first` (#1448): attempt 1
//!    pages the recommended gNBs within the registration area (all six), and a
//!    retransmission pages all six. `page-all` sends no IE, so attempt 1 is
//!    all six.
//!
//! **The route generator is a motion model, and it favours the predictor that
//! shares it.** Constant-turn routes are exactly `ct`'s model, so `ct` is
//! scored in-model there; straight routes are `lsq`'s model (on them lsq
//! scores at least as well as ct); turn-change routes (the turn rate redrawn
//! at release) are neither's. The table reports all three families, and the
//! #535 review's finding that a 50-seed block is seed-specific is why each
//! family also has a 2000-seed row with a 95% Wilson interval and a paired
//! exact McNemar test of ct against lsq.
//!
//! **Compare modes within a family, not families with each other.** Seed N
//! takes the same first draw in every family, but a family whose route leaves
//! coverage redraws on its own stream, so many seeds are not paired across
//! families (`the_route_families_are_paired_by_seed` counts them), and each
//! family is conditioned on staying in coverage under its own motion. Within a
//! family every mode sees exactly the same routes.
//!
//! The table counts, per mode, the first-attempt hits and the mean PAGING
//! messages per mobile-terminated call. It is checked in
//! (`config/demo-ai-paging/d440a-replay-table.txt`), and every row is asserted
//! exactly (the replay is deterministic). Every committed line is also checked
//! against itself: its rate and interval must be what its own hits and calls
//! give, and its mean must be possible for its hit count. Nothing asserts that
//! one predictor beats another.
//!
//! What it does not model: amfd itself, the NGAP round trips, the UE's
//! connected-mode mobility (the serving gNB at release is where the
//! R-criterion has the UE camped, as if connected mobility followed the same
//! rule), heartbeat jitter, and a missed page's T3513 delay (a miss costs six
//! PAGINGs more and is counted as answered on attempt 2).

use super::{recommend, Grid, GridCell, Predictor};
use nextgsim_common::config::{SimPosition, SimRoute};
use nextgsim_nwdaf::{NwdafManager, UeMeasurement, Vector3};
use serde::Deserialize;
use std::time::Duration;

const SCENARIO: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../config/demo-ai-paging/d440a-scenario.yaml"
);
const TABLE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../config/demo-ai-paging/d440a-replay-table.txt"
);

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GridEntry {
    nci: u64,
    position: SimPosition,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScenarioRanges {
    start_x_m: [f64; 2],
    start_y_m: [f64; 2],
    speed_mps: [f64; 2],
    turn_rate_rad_s: [f64; 2],
    connected_s: u32,
    page_after_release_s: [u32; 2],
    waypoint_every_s: u32,
    route_s: u32,
    coverage_margin_m: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Reselection {
    q_hyst_db: i32,
    t_reselection_s: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Seeds {
    live: Vec<u64>,
    replay_first: u64,
    replay_count: u64,
    large_sample_first: u64,
    large_sample_count: u64,
}

/// How a route turns. Every family makes its draws in one order, so seed N
/// takes the same *first* draw in each; a family whose route leaves coverage
/// redraws on its own, and from then on that seed's routes differ between
/// families (see `the_route_families_are_paired_by_seed`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Family {
    /// One turn rate for the whole route: `ct`'s own motion model.
    ConstantTurn,
    /// Turn rate 0: `lsq`'s own motion model.
    Straight,
    /// The turn rate is redrawn at release: what the UE does idle is not what
    /// any predictor saw it do connected.
    TurnChange,
}

impl Family {
    /// Every family, in table order.
    pub(crate) const ALL: [Family; 3] = [Self::ConstantTurn, Self::Straight, Self::TurnChange];

    /// The table's name for it.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::ConstantTurn => "constant-turn",
            Self::Straight => "straight",
            Self::TurnChange => "turn-change",
        }
    }
}

/// The position after `t` seconds along an arc from `(x0, y0)` with `heading`,
/// `speed` and turn rate `omega`, and the heading there.
fn arc(x0: f64, y0: f64, heading: f64, speed: f64, omega: f64, t: f64) -> (f64, f64, f64) {
    if omega.abs() < 1e-9 {
        (
            x0 + speed * t * heading.cos(),
            y0 + speed * t * heading.sin(),
            heading,
        )
    } else {
        let r = speed / omega;
        (
            x0 + r * ((heading + omega * t).sin() - heading.sin()),
            y0 - r * ((heading + omega * t).cos() - heading.cos()),
            heading + omega * t,
        )
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ScenarioFile {
    grid: Vec<GridEntry>,
    gnb_id_length: u8,
    scenario: ScenarioRanges,
    reselection: Reselection,
    seeds: Seeds,
}

pub(crate) fn scenario_file() -> ScenarioFile {
    serde_yaml::from_str(&std::fs::read_to_string(SCENARIO).expect("read the scenario"))
        .expect("parse the scenario")
}

/// SplitMix64: a fixed, dependency-free stream, so a seed means the same
/// route on every build.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn range(&mut self, [lo, hi]: [f64; 2]) -> f64 {
        lo + (hi - lo) * ((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64)
    }

    fn range_u32(&mut self, [lo, hi]: [u32; 2]) -> u32 {
        lo + (self.next_u64() % u64::from(hi - lo + 1)) as u32
    }
}

/// One seed's run: where it starts, its route, and when it is released and
/// paged (seconds after the route starts).
#[derive(Debug, Clone)]
pub(crate) struct Run {
    pub(crate) seed: u64,
    pub(crate) start: SimPosition,
    pub(crate) route: SimRoute,
    pub(crate) release_at_s: u32,
    pub(crate) page_at_s: u32,
}

impl Run {
    fn position(&self, t_s: u32) -> SimPosition {
        self.route
            .position_at(self.start, Duration::from_secs(u64::from(t_s)))
    }
}

impl ScenarioFile {
    fn cells(&self) -> Vec<GridCell> {
        let len = self.gnb_id_length.clamp(22, 32);
        self.grid
            .iter()
            .map(|e| GridCell {
                nci: e.nci,
                gnb_id: (e.nci >> (36 - u32::from(len))) as u32,
                gnb_id_len: len,
                position: Vector3::new(
                    f64::from(e.position.x),
                    f64::from(e.position.y),
                    f64::from(e.position.z),
                ),
            })
            .collect()
    }

    pub(crate) fn replay_seeds(&self) -> std::ops::Range<u64> {
        self.seeds.replay_first..self.seeds.replay_first + self.seeds.replay_count
    }

    pub(crate) fn large_sample_seeds(&self) -> std::ops::Range<u64> {
        self.seeds.large_sample_first..self.seeds.large_sample_first + self.seeds.large_sample_count
    }

    /// The constant-turn run for `seed` (the committed 50-seed rows).
    pub(crate) fn run(&self, seed: u64) -> Run {
        self.run_family(seed, Family::ConstantTurn)
    }

    /// The run for `seed` in `family`. A draw whose route leaves the coverage
    /// box is redrawn from the same stream, so the seed still fixes the route.
    pub(crate) fn run_family(&self, seed: u64, family: Family) -> Run {
        let s = &self.scenario;
        let cells = self.cells();
        let (lo_x, hi_x, lo_y, hi_y) = cells.iter().fold(
            (f64::MAX, f64::MIN, f64::MAX, f64::MIN),
            |(a, b, c, d), cell| {
                (
                    a.min(cell.position.x),
                    b.max(cell.position.x),
                    c.min(cell.position.y),
                    d.max(cell.position.y),
                )
            },
        );
        let m = s.coverage_margin_m;
        let inside = |p: SimPosition| {
            let (x, y) = (f64::from(p.x), f64::from(p.y));
            x >= lo_x - m && x <= hi_x + m && y >= lo_y - m && y <= hi_y + m
        };
        let mut rng = SplitMix64(seed ^ 0xD440_A000_0000_0462);
        for _ in 0..10_000 {
            let (x0, y0) = (rng.range(s.start_x_m), rng.range(s.start_y_m));
            let heading = rng.range([0.0, std::f64::consts::TAU]);
            let speed = rng.range(s.speed_mps);
            let omega = rng.range(s.turn_rate_rad_s);
            let page_after = rng.range_u32(s.page_after_release_s);
            // Drawn last, so the constant-turn family's draws are the ones its
            // committed rows were made with.
            let omega_idle = if family == Family::TurnChange {
                rng.range(s.turn_rate_rad_s)
            } else {
                omega
            };
            let omega = if family == Family::Straight {
                0.0
            } else {
                omega
            };
            let omega_idle = if family == Family::Straight {
                0.0
            } else {
                omega_idle
            };
            let t_release = f64::from(s.connected_s);
            let at = |t: f64| {
                let (x, y) = if t <= t_release {
                    let (x, y, _) = arc(x0, y0, heading, speed, omega, t);
                    (x, y)
                } else {
                    let (xr, yr, hr) = arc(x0, y0, heading, speed, omega, t_release);
                    let (x, y, _) = arc(xr, yr, hr, speed, omega_idle, t - t_release);
                    (x, y)
                };
                SimPosition {
                    x: x.round() as i32,
                    y: y.round() as i32,
                    z: 0,
                }
            };
            let run = Run {
                seed,
                start: at(0.0),
                route: SimRoute {
                    waypoints: (1..=s.route_s / s.waypoint_every_s)
                        .map(|k| at(f64::from(k * s.waypoint_every_s)))
                        .collect(),
                    speed_mps: speed,
                    start_after_secs: 0.0,
                },
                release_at_s: s.connected_s,
                page_at_s: s.connected_s + page_after,
            };
            if run.route.validate().is_ok() && (0..=run.page_at_s).all(|t| inside(run.position(t)))
            {
                return run;
            }
        }
        panic!("seed {seed}: no route inside the coverage box in 10000 draws");
    }
}

/// The distance model's level at `p` from `cell` (`nextgsim_rls::DistanceModel`):
/// minus the whole metres, -1 at zero distance.
fn dbm(cell: &GridCell, p: SimPosition) -> i32 {
    let d = (cell.position.x - f64::from(p.x))
        .hypot(cell.position.y - f64::from(p.y))
        .hypot(cell.position.z - f64::from(p.z)) as i32;
    if d == 0 {
        -1
    } else {
        -d
    }
}

/// One seed under one mode.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Outcome {
    pub(crate) seed: u64,
    pub(crate) serving_gnb: u32,
    pub(crate) recommended: Vec<u32>,
    pub(crate) answering_gnb: u32,
    pub(crate) first_attempt_hit: bool,
    pub(crate) pagings: usize,
    pub(crate) served_by: &'static str,
}

/// Replays `run` under `mode` (`None`: page-all). `prepare` sets the fresh
/// NWDAF manager up (the onnx arm loads its verified model there).
pub(crate) fn replay(
    file: &ScenarioFile,
    run: &Run,
    mode: Option<Predictor>,
    prepare: &dyn Fn(&mut NwdafManager),
) -> Outcome {
    const UE: i32 = 1;
    let cells = file.cells();
    let strongest = |p: SimPosition| {
        *cells
            .iter()
            .max_by_key(|c| (dbm(c, p), std::cmp::Reverse(c.nci)))
            .expect("a grid")
    };
    let mut nwdaf = NwdafManager::new(100);
    prepare(&mut nwdaf);
    let mut camped = strongest(run.start);
    let mut candidate: Option<(u64, u32)> = None;
    let mut recommended: Option<(Vec<u32>, &'static str)> = None;
    let mut serving = camped;
    for t in 0..=run.page_at_s {
        let p = run.position(t);
        if t > 0 {
            let best = strongest(p);
            let r_s = dbm(&camped, p) + file.reselection.q_hyst_db;
            if best.nci != camped.nci && dbm(&best, p) > r_s {
                match candidate {
                    Some((nci, since))
                        if nci == best.nci && t - since >= file.reselection.t_reselection_s =>
                    {
                        camped = best;
                        candidate = None;
                    }
                    Some((nci, _)) if nci == best.nci => {}
                    _ => candidate = Some((best.nci, t)),
                }
            } else {
                candidate = None;
            }
        }
        if t <= run.release_at_s {
            nwdaf.record_measurement(UeMeasurement {
                ue_id: UE,
                rsrp: dbm(&camped, p) as f32,
                rsrq: None,
                sinr: None,
                position: Vector3::new(f64::from(p.x), f64::from(p.y), f64::from(p.z)),
                velocity: None,
                serving_cell_id: camped.nci as i32,
                timestamp_ms: 1_700_000_000_000 + 1000 * u64::from(t),
            });
        }
        if t == run.release_at_s {
            serving = camped;
            if let Some(predictor) = mode {
                let grid = Grid {
                    plmn: [0x99, 0xF9, 0x07],
                    own: camped,
                    neighbours: cells
                        .iter()
                        .filter(|c| c.nci != camped.nci)
                        .copied()
                        .collect(),
                };
                let rec = recommend(&nwdaf, &grid, UE, predictor);
                recommended = Some((rec.gnb_ids(), rec.predictor));
            }
        }
    }
    let all: Vec<u32> = cells.iter().map(|c| c.gnb_id).collect();
    let (first, served_by) = match recommended {
        // recommended-first: the listed gNBs within the area, de-duplicated.
        Some((ids, by)) => {
            let mut first: Vec<u32> = Vec::new();
            for id in ids {
                if all.contains(&id) && !first.contains(&id) {
                    first.push(id);
                }
            }
            if first.is_empty() {
                (all.clone(), by)
            } else {
                (first, by)
            }
        }
        None => (all.clone(), "page-all"),
    };
    let hit = first.contains(&camped.gnb_id);
    Outcome {
        seed: run.seed,
        serving_gnb: serving.gnb_id,
        answering_gnb: camped.gnb_id,
        first_attempt_hit: hit,
        pagings: first.len() + if hit { 0 } else { all.len() },
        recommended: first,
        served_by,
    }
}

/// The Wilson score 95% interval of `hits` in `n`, in percent.
pub(crate) fn wilson_95(hits: usize, n: usize) -> (f64, f64) {
    let (n, p, z) = (n as f64, hits as f64 / n as f64, 1.96_f64);
    let denom = 1.0 + z * z / n;
    let centre = (p + z * z / (2.0 * n)) / denom;
    let half = z * (p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt() / denom;
    (
        100.0 * (centre - half).max(0.0),
        100.0 * (centre + half).min(1.0),
    )
}

/// The key a table line starts with: family, seed range, mode.
fn key(family: Family, seeds: &std::ops::Range<u64>, mode: &str) -> String {
    format!(
        "| {:<13} | {:<9} | {mode:<11} |",
        family.name(),
        format!("{}..={}", seeds.start, seeds.end - 1)
    )
}

/// The table's line for one family, seed range and mode.
pub(crate) fn table_line(
    family: Family,
    seeds: &std::ops::Range<u64>,
    mode: &str,
    outcomes: &[Outcome],
) -> String {
    let n = outcomes.len();
    let hits = outcomes.iter().filter(|o| o.first_attempt_hit).count();
    let pagings: usize = outcomes.iter().map(|o| o.pagings).sum();
    let (lo, hi) = wilson_95(hits, n);
    format!(
        "{} {n:>5} | {hits:>5} | {:>6.1}% | {lo:>5.1}..{hi:>5.1}% | {:>6.2} |",
        key(family, seeds, mode),
        100.0 * hits as f64 / n as f64,
        pagings as f64 / n as f64
    )
}

/// The committed table's line for `family`, `seeds` and `mode`.
pub(crate) fn committed_line(family: Family, seeds: &std::ops::Range<u64>, mode: &str) -> String {
    let text = std::fs::read_to_string(TABLE).expect("read the committed table");
    let k = key(family, seeds, mode);
    text.lines().find(|l| l.starts_with(&k)).map_or_else(
        || format!("{k} (no such line in the committed table)"),
        str::to_string,
    )
}

/// The cells of a table line, trimmed.
fn cells(line: &str) -> Vec<&str> {
    line.split('|').map(str::trim).collect()
}

/// Checks a committed data line against itself: the rate and interval are
/// what its own hits and calls give (formatted as [`table_line`] formats
/// them), and its mean PAGINGs are possible for its hit count. A first attempt
/// pages at least one gNB and a recommendation names at most two (page-all:
/// all six), and a miss costs six more.
pub(crate) fn check_self_consistent(line: &str) -> Result<(), String> {
    let c = cells(line);
    // | family | seeds | mode | calls | hits | rate% | lo..hi% | mean |
    if c.len() != 10 {
        return Err(format!("not a data line: {line}"));
    }
    let calls: usize = c[4].parse().map_err(|_| format!("calls: {line}"))?;
    let hits: usize = c[5].parse().map_err(|_| format!("hits: {line}"))?;
    let mean: f64 = c[8].parse().map_err(|_| format!("mean: {line}"))?;
    if calls == 0 || hits > calls {
        return Err(format!("{hits} hits of {calls} calls: {line}"));
    }
    let rate = format!("{:.1}%", 100.0 * hits as f64 / calls as f64);
    let (lo, hi) = wilson_95(hits, calls);
    let interval = format!("{lo:>5.1}..{hi:>5.1}%");
    if c[6] != rate {
        return Err(format!(
            "rate {} but {hits}/{calls} is {rate}: {line}",
            c[6]
        ));
    }
    if c[7] != interval.trim() {
        return Err(format!(
            "interval {} but Wilson gives {}: {line}",
            c[7],
            interval.trim()
        ));
    }
    let miss = (calls - hits) as f64 / calls as f64;
    let (first_lo, first_hi) = if c[3] == "page-all" {
        (6.0, 6.0)
    } else {
        (1.0, 2.0)
    };
    let (min, max) = (first_lo + 6.0 * miss, first_hi + 6.0 * miss);
    if mean < min - 0.005 || mean > max + 0.005 {
        return Err(format!(
            "mean {mean} outside {min:.2}..{max:.2} for {hits}/{calls}: {line}"
        ));
    }
    Ok(())
}

/// The exact two-sided McNemar p-value for `b` and `c` discordant pairs: the
/// binomial(b + c, 1/2) probability of a split at least this uneven.
pub(crate) fn mcnemar_exact(b: usize, c: usize) -> f64 {
    let n = b + c;
    if n == 0 {
        return 1.0;
    }
    let k = b.min(c);
    // Sum C(n, i) / 2^n for i in 0..=k, in log space.
    let ln_half_n = n as f64 * 0.5_f64.ln();
    let mut ln_c = 0.0_f64;
    let mut tail = 0.0_f64;
    for i in 0..=k {
        if i > 0 {
            ln_c += ((n - i + 1) as f64).ln() - (i as f64).ln();
        }
        tail += (ln_c + ln_half_n).exp();
    }
    (2.0 * tail).min(1.0)
}

/// The key a McNemar line starts with.
fn mcnemar_key(family: Family, seeds: &std::ops::Range<u64>, a: &str, b: &str) -> String {
    format!(
        "| mcnemar | {:<13} | {:<9} | {a} vs {b} |",
        family.name(),
        format!("{}..={}", seeds.start, seeds.end - 1)
    )
}

/// The McNemar line for modes `a` and `b` on the same seeds of `family`: the
/// seeds only `a`'s first attempt hit, only `b`'s, and the exact p-value.
pub(crate) fn mcnemar_line(
    family: Family,
    seeds: &std::ops::Range<u64>,
    (a, oa): (&str, &[Outcome]),
    (b, ob): (&str, &[Outcome]),
) -> String {
    assert_eq!(oa.len(), ob.len());
    let only_a = oa
        .iter()
        .zip(ob)
        .filter(|(x, y)| x.first_attempt_hit && !y.first_attempt_hit)
        .count();
    let only_b = oa
        .iter()
        .zip(ob)
        .filter(|(x, y)| !x.first_attempt_hit && y.first_attempt_hit)
        .count();
    format!(
        "{} {a}-only {only_a} | {b}-only {only_b} | p={:.1e} |",
        mcnemar_key(family, seeds, a, b),
        mcnemar_exact(only_a, only_b)
    )
}

/// The committed line starting with `key`.
fn committed_with_key(key: &str) -> String {
    let text = std::fs::read_to_string(TABLE).expect("read the committed table");
    text.lines().find(|l| l.starts_with(key)).map_or_else(
        || format!("{key} (no such line in the committed table)"),
        str::to_string,
    )
}

pub(crate) fn run_mode(
    file: &ScenarioFile,
    mode: Option<Predictor>,
    prepare: &dyn Fn(&mut NwdafManager),
) -> Vec<Outcome> {
    run_seeds(
        file,
        Family::ConstantTurn,
        file.replay_seeds(),
        mode,
        prepare,
    )
}

pub(crate) fn run_seeds(
    file: &ScenarioFile,
    family: Family,
    seeds: std::ops::Range<u64>,
    mode: Option<Predictor>,
    prepare: &dyn Fn(&mut NwdafManager),
) -> Vec<Outcome> {
    seeds
        .map(|seed| replay(file, &file.run_family(seed, family), mode, prepare))
        .collect()
}

/// Every table line for `(name, mode)`, with the committed one: the 50
/// constant-turn seeds and each family's large sample.
pub(crate) fn table_lines(
    file: &ScenarioFile,
    name: &str,
    mode: Option<Predictor>,
    prepare: &dyn Fn(&mut NwdafManager),
) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let seeds = file.replay_seeds();
    let o = run_seeds(file, Family::ConstantTurn, seeds.clone(), mode, prepare);
    out.push((
        table_line(Family::ConstantTurn, &seeds, name, &o),
        committed_line(Family::ConstantTurn, &seeds, name),
    ));
    let large = file.large_sample_seeds();
    for family in Family::ALL {
        let o = run_seeds(file, family, large.clone(), mode, prepare);
        out.push((
            table_line(family, &large, name, &o),
            committed_line(family, &large, name),
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODES: [(&str, Option<Predictor>); 4] = [
        ("page-all", None),
        ("last-served", Some(Predictor::LastServed)),
        ("ct", Some(Predictor::ConstantTurn)),
        ("lsq", Some(Predictor::LeastSquares)),
    ];

    /// #462: the committed scenario has at least 5 live and 50 replay seeds,
    /// the live ones among the replayed, and every seed gives a route that
    /// stays in coverage, travels, and is the same route twice.
    #[test]
    fn the_committed_seeds_give_reproducible_routes() {
        let file = scenario_file();
        assert!(file.seeds.live.len() >= 5);
        assert!(file.seeds.replay_count >= 50);
        for s in &file.seeds.live {
            assert!(file.replay_seeds().contains(s), "live seed {s} is replayed");
        }
        assert_eq!(file.grid.len(), 6);
        for seed in file.replay_seeds() {
            let (a, b) = (file.run(seed), file.run(seed));
            assert_eq!(a.route, b.route, "seed {seed}");
            assert_ne!(
                a.position(0),
                a.position(a.release_at_s),
                "seed {seed} moves"
            );
            assert!(a.page_at_s > a.release_at_s);
        }
        assert_ne!(file.run(1).route, file.run(2).route);
    }

    /// #462: the five-arm table's four model-free modes, replayed from the
    /// committed seeds, are exactly the committed table: the 50 constant-turn
    /// seeds, each route family's 2000-seed row, and each family's McNemar
    /// line for ct against lsq. `page-all` is
    /// the check that the replay is honest: with no IE every first attempt
    /// pages the whole area, so its hit rate is 100% at 6 PAGINGs per call.
    #[test]
    fn the_replay_reproduces_the_committed_table() {
        let file = scenario_file();
        assert!(file.seeds.large_sample_count >= 2000);
        let mut lines = Vec::new();
        for (name, mode) in MODES {
            let outcomes = run_mode(&file, mode, &|_| {});
            if std::env::var_os("D440A_REPLAY_DETAIL").is_some() {
                for o in &outcomes {
                    println!("{name} {o:?}");
                }
            }
            if mode.is_none() {
                assert!(outcomes
                    .iter()
                    .all(|o| o.first_attempt_hit && o.pagings == 6));
            }
            if let Some(p) = mode {
                assert!(
                    outcomes.iter().all(|o| o.served_by == p.mode()),
                    "{name}: served by its own predictor"
                );
            }
            lines.extend(table_lines(&file, name, mode, &|_| {}));
        }
        let large = file.large_sample_seeds();
        for family in Family::ALL {
            let ct = run_seeds(
                &file,
                family,
                large.clone(),
                Some(Predictor::ConstantTurn),
                &|_| {},
            );
            let lsq = run_seeds(
                &file,
                family,
                large.clone(),
                Some(Predictor::LeastSquares),
                &|_| {},
            );
            let line = mcnemar_line(family, &large, ("ct", &ct), ("lsq", &lsq));
            let key = mcnemar_key(family, &large, "ct", "lsq");
            lines.push((line, committed_with_key(&key)));
        }
        for (line, _) in &lines {
            println!("{line}");
        }
        for (line, committed) in &lines {
            assert_eq!(line, committed);
        }
    }

    /// #462 (#535 review round 2): every data line of the committed table,
    /// onnx rows included, is consistent with itself, and every McNemar line's
    /// p-value is the one its own counts give. So a hand edit cannot make the
    /// table say what its hits do not, even where the replay cannot run.
    #[test]
    fn every_committed_line_is_self_consistent() {
        let text = std::fs::read_to_string(TABLE).expect("read the table");
        let (mut data, mut tests) = (0, 0);
        for line in text
            .lines()
            .filter(|l| l.starts_with("| ") && !l.contains("---"))
        {
            let c = cells(line);
            if c[1] == "family" {
                continue; // the header
            }
            if c[1] == "mcnemar" {
                // | mcnemar | family | seeds | a vs b | a-only N | b-only M | p=P |
                let n = |cell: &str| cell.rsplit(' ').next().unwrap().parse::<usize>().unwrap();
                let p = format!("p={:.1e}", mcnemar_exact(n(c[5]), n(c[6])));
                assert_eq!(c[7], p, "{line}");
                tests += 1;
            } else {
                check_self_consistent(line).unwrap_or_else(|e| panic!("{e}"));
                data += 1;
            }
        }
        assert_eq!((data, tests), (20, 3), "every committed row was checked");
    }

    /// The self-consistency check refuses the #535 round-2 probe's tampered
    /// line (rate, hits and interval that disagree) and accepts the real one.
    #[test]
    fn a_tampered_line_is_refused() {
        let real = committed_line(Family::Straight, &(1..2001), "ct");
        assert!(check_self_consistent(&real).is_ok(), "{real}");
        let tampered = real
            .replace("|  1911 |", "|  1500 |")
            .replace("95.5%", "96.5%")
            .replace(" 94.6.. 96.4%", " 10.0.. 20.0%");
        assert_ne!(tampered, real);
        assert!(check_self_consistent(&tampered).is_err(), "{tampered}");
        // Only the rate moved: still refused.
        assert!(check_self_consistent(&real.replace("95.5%", "96.5%")).is_err());
        // An impossible mean for the hit count: refused.
        assert!(check_self_consistent(&real.replace("  1.73 |", "  4.00 |")).is_err());
    }

    /// The exact McNemar p-value of known splits.
    #[test]
    fn the_mcnemar_p_value_is_the_exact_binomial_one() {
        assert_eq!(mcnemar_exact(0, 0), 1.0);
        assert_eq!(mcnemar_exact(5, 5), 1.0);
        // 2 * (1 + 10 + 45) / 1024 for a 2:8 split of 10.
        assert!((mcnemar_exact(2, 8) - 112.0 / 1024.0).abs() < 1e-12);
        assert!((mcnemar_exact(8, 2) - mcnemar_exact(2, 8)).abs() < 1e-15);
    }

    /// The seeds of 1..=2000 whose straight and turn-change runs are not
    /// paired with the constant-turn run (another start, speed or page time,
    /// because that family redrew to stay in coverage).
    const UNPAIRED_STRAIGHT: usize = 477;
    const UNPAIRED_TURN_CHANGE: usize = 803;

    /// #462 (#535 review rounds 1-2): seed N takes the same first draw in
    /// every family. Where the families are paired, the straight route has no
    /// turn and the turn-change route agrees with the constant-turn one up to
    /// release and then turns differently. Where a family redrew to stay in
    /// coverage the seed is not paired; those seeds are counted, not skipped
    /// silently, and the counts are the ones the scenario and table quote.
    #[test]
    fn the_route_families_are_paired_by_seed() {
        let file = scenario_file();
        let same = |a: &Run, b: &Run| {
            a.start == b.start
                && a.route.speed_mps == b.route.speed_mps
                && a.page_at_s == b.page_at_s
        };
        let (mut unpaired_st, mut unpaired_tc, mut differs_after_release) = (0, 0, 0);
        for seed in file.large_sample_seeds() {
            let ct = file.run_family(seed, Family::ConstantTurn);
            let st = file.run_family(seed, Family::Straight);
            let tc = file.run_family(seed, Family::TurnChange);
            if same(&ct, &st) {
                // Straight: every position within whole-metre rounding (of it
                // and of both ends, so 2.5 m) of the
                // line from the start to the page-time position.
                let (a, b) = (st.position(0), st.position(st.page_at_s));
                let (dx, dy) = (f64::from(b.x - a.x), f64::from(b.y - a.y));
                let len = dx.hypot(dy);
                for t in 0..=st.page_at_s {
                    let p = st.position(t);
                    let off = (dx * f64::from(p.y - a.y) - dy * f64::from(p.x - a.x)).abs() / len;
                    assert!(
                        off <= 2.5,
                        "seed {seed} t {t}: straight route is {off:.1} m off its line"
                    );
                }
            } else {
                unpaired_st += 1;
            }
            if same(&ct, &tc) {
                for t in 0..=ct.release_at_s {
                    let (a, b) = (ct.position(t), tc.position(t));
                    assert!(
                        (a.x - b.x).abs() <= 1 && (a.y - b.y).abs() <= 1,
                        "seed {seed} t {t}"
                    );
                }
                if ct.position(ct.page_at_s) != tc.position(tc.page_at_s) {
                    differs_after_release += 1;
                }
            } else {
                unpaired_tc += 1;
            }
        }
        println!("unpaired: straight {unpaired_st}, turn-change {unpaired_tc} of 2000");
        assert_eq!(
            (unpaired_st, unpaired_tc),
            (UNPAIRED_STRAIGHT, UNPAIRED_TURN_CHANGE),
            "the counts the scenario and table quote"
        );
        // Bounded: at least half of each family is paired.
        assert!(unpaired_st < 1000 && unpaired_tc < 1000);
        assert!(differs_after_release >= 500, "{differs_after_release}");
    }

    /// The Wilson interval of a few known cases.
    #[test]
    fn the_wilson_interval_is_the_textbook_one() {
        let (lo, hi) = wilson_95(50, 50);
        assert!((lo - 92.9).abs() < 0.1 && hi == 100.0, "{lo} {hi}");
        let (lo, hi) = wilson_95(1000, 2000);
        assert!(
            (lo - 47.8).abs() < 0.1 && (hi - 52.2).abs() < 0.1,
            "{lo} {hi}"
        );
    }

    /// #462: the replay's recommendations are the ones the live path would
    /// send: a seed whose UE moves to another gNB is, under `ct`, recommended
    /// [serving, predicted] and under `last-served` [serving] only.
    #[test]
    fn a_replayed_recommendation_lists_the_serving_gnb_first() {
        let file = scenario_file();
        let outcomes = run_mode(&file, Some(Predictor::ConstantTurn), &|_| {});
        let moved = outcomes
            .iter()
            .find(|o| o.answering_gnb != o.serving_gnb && o.recommended.len() == 2)
            .expect("some seed moves to another gNB and ct predicts a move");
        assert_eq!(moved.recommended[0], moved.serving_gnb);
        let ls = replay(
            &file,
            &file.run(moved.seed),
            Some(Predictor::LastServed),
            &|_| {},
        );
        assert_eq!(ls.recommended, vec![moved.serving_gnb]);
        assert!(!ls.first_attempt_hit);
        assert_eq!(ls.pagings, 7);
    }

    /// #462: the `onnx` row, with the d440b reference model verified and
    /// activated through the NWDAF manager (#463). Every prediction must be
    /// the model's (none `fallback`), and the row is the committed one.
    #[test]
    #[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH); run by the model-verify CI job"]
    fn the_onnx_row_reproduces_the_committed_table() {
        assert!(
            std::env::var_os("ORT_DYLIB_PATH").is_some(),
            "this test needs ONNX Runtime: set ORT_DYLIB_PATH to the pinned libonnxruntime.so"
        );
        let model = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tools/d440b-model/nwdaf-trajectory-cv.onnx"
        ));
        let file = scenario_file();
        let load = |nwdaf: &mut NwdafManager| {
            nwdaf
                .load_verified_trajectory_model(&model)
                .expect("the reference model verifies");
        };
        let outcomes = run_mode(&file, Some(Predictor::Onnx), &load);
        assert!(
            outcomes.iter().all(|o| o.served_by == "onnx"),
            "every prediction served by the verified model"
        );
        let lines = table_lines(&file, "onnx", Some(Predictor::Onnx), &load);
        for (line, _) in &lines {
            println!("{line}");
        }
        for (line, committed) in &lines {
            assert_eq!(line, committed, "exact, as every row");
        }
    }
}
