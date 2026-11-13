//! D440C (nextgsim#464): the seeded commuter-lap route of
//! `config/demo-nwdaf-paging/ue-lap.yaml`, and the geometry it must hold.
//!
//! Three gNBs serve one TAC (`config/demo-nwdaf-paging/gnb-{a,b,c}.yaml`):
//! A at (0, 0), B at (100, 0), C at (50, 87). The UE rides laps A -> B -> C -> A
//! while CM-CONNECTED, so each N2 handover gives amfd a HANDOVER NOTIFY and
//! nwdafd a LOCATION_REPORT; the transitions out of A are then all A -> B. It
//! then makes an excursion that stays deep in A's coverage (the harness
//! releases it to CM-IDLE there), and finally moves to B while idle and
//! reselects it. On the downlink page amfd's attempt 1 targets {B (predicted),
//! A (last known)} and B answers.
//!
//! The route is generated from a seed (per-waypoint jitter of up to 3 m), and
//! the committed config is seed [`COMMITTED_SEED`]'s. These tests check the
//! committed file is that seed's route, and that the geometry holds on every
//! one of [`SWEEP_SEEDS`], so the committed route is not one tuned to pass.
//! They model the simulator's radio as the UE sees it: the
//! `nextgsim_rls::channel::DistanceModel` level, reported no higher than the
//! -31 dBm RSRP-range limit, the A3 rule of the gNB's default measurement
//! config (offset 3 dB + hysteresis 1 dB, timeToTrigger 640 ms) while
//! connected, and the R-criterion (q-Hyst 4 dB, Treselection 1 s) while idle.
//! Each is also run with a slower 3 s trigger, for a loaded runner.
//!
//! Regenerate the committed route (print a seed's `sim_position` and
//! `sim_route`):
//!
//! ```text
//! D440C_SEED=464 cargo test -p nextgsim-integration-tests --test d440c_lap_route \
//!     -- --ignored --nocapture print_the_lap_route
//! ```

use std::time::Duration;

use nextgsim_common::config::{SimPosition, SimRoute, UeConfig};
use nextgsim_rls::cell_search::MIN_ALLOWED_DBM;
use nextgsim_rls::channel::{ChannelModel, DistanceModel};
use nextgsim_rls::protocol::SimCoord;

/// The seed of the committed `config/demo-nwdaf-paging/ue-lap.yaml` route.
const COMMITTED_SEED: u64 = 464;
/// The seeds the geometry is checked on: 32 besides the committed one.
const SWEEP_SEEDS: std::ops::Range<u64> = 0..32;

/// Laps ridden while CM-CONNECTED.
const LAPS: usize = 3;
/// UE speed along every segment, m/s.
const SPEED_MPS: f64 = 2.5;
/// Seconds before the route starts: registration and the PDU session first.
const START_AFTER_SECS: f64 = 60.0;
/// The shortest dwell in A the harness gets to release the UE to CM-IDLE.
const MIN_DWELL_SECS: f64 = 40.0;
/// The most a waypoint is moved off its nominal point, per axis, metres.
const JITTER_M: i32 = 3;

/// The gNBs, by name, NCI, position and address
/// (config/demo-nwdaf-paging/gnb-*.yaml).
const GNBS: [(&str, u64, SimPosition, &str); 3] = [
    ("A", 0x10, SimPosition { x: 0, y: 0, z: 0 }, "172.23.0.100"),
    (
        "B",
        0x20,
        SimPosition { x: 100, y: 0, z: 0 },
        "172.23.0.102",
    ),
    (
        "C",
        0x30,
        SimPosition { x: 50, y: 87, z: 0 },
        "172.23.0.103",
    ),
];
const A: usize = 0;
const B: usize = 1;
const C: usize = 2;

/// The lap's corners: about 35 m from each gNB toward the triangle's centre, so
/// every level is above the -31 dBm clamp and the A3 margin is crossed mid-edge.
const LAP_A: SimPosition = SimPosition { x: 30, y: 17, z: 0 };
const LAP_B: SimPosition = SimPosition { x: 70, y: 17, z: 0 };
const LAP_C: SimPosition = SimPosition { x: 50, y: 52, z: 0 };
/// The dwell excursion's far point: behind A, away from B and C.
const DWELL_A: SimPosition = SimPosition {
    x: -20,
    y: -12,
    z: 0,
};
/// Where the idle UE ends: 23 m from B.
const IDLE_END_B: SimPosition = SimPosition { x: 78, y: 8, z: 0 };

/// splitmix64: a fixed, dependency-free generator, so a seed names one route
/// on every machine and toolchain.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn jitter(&mut self) -> i32 {
        (self.next() % (2 * JITTER_M as u64 + 1)) as i32 - JITTER_M
    }

    fn around(&mut self, p: SimPosition) -> SimPosition {
        SimPosition {
            x: p.x + self.jitter(),
            y: p.y + self.jitter(),
            z: 0,
        }
    }
}

/// The route's shape: the start, then where each phase ends (as waypoint
/// counts), so the checks know which part of the route they are in.
struct LapRoute {
    start: SimPosition,
    route: SimRoute,
    /// Waypoints up to and including the last lap's return to A.
    laps_end: usize,
    /// Waypoints up to and including the dwell's return to A.
    dwell_end: usize,
}

/// Seed `seed`'s route: `LAPS` laps from A, the dwell excursion, then to B.
fn lap_route(seed: u64) -> LapRoute {
    let mut rng = SplitMix64(seed);
    let start = LAP_A;
    let mut waypoints = Vec::new();
    for _ in 0..LAPS {
        for corner in [LAP_B, LAP_C, LAP_A] {
            waypoints.push(rng.around(corner));
        }
    }
    let laps_end = waypoints.len();
    waypoints.push(rng.around(DWELL_A));
    waypoints.push(rng.around(LAP_A));
    let dwell_end = waypoints.len();
    waypoints.push(rng.around(IDLE_END_B));
    LapRoute {
        start,
        route: SimRoute {
            waypoints,
            speed_mps: SPEED_MPS,
            start_after_secs: START_AFTER_SECS,
        },
        laps_end,
        dwell_end,
    }
}

/// Seconds from start-up until the UE reaches waypoint `n` (1-based count).
fn arrival_secs(r: &LapRoute, n: usize) -> f64 {
    let mut at = r.start;
    let mut metres = 0.0;
    for p in &r.route.waypoints[..n] {
        metres += distance(at, *p);
        at = *p;
    }
    r.route.start_after_secs + metres / r.route.speed_mps
}

fn distance(a: SimPosition, b: SimPosition) -> f64 {
    let (dx, dy) = (f64::from(a.x - b.x), f64::from(a.y - b.y));
    (dx * dx + dy * dy).sqrt()
}

fn coord(p: SimPosition) -> SimCoord {
    SimCoord {
        x: p.x,
        y: p.y,
        z: p.z,
    }
}

/// Each gNB's level at `ue` as the UE reports it: the simulator's channel
/// model, no higher than the RSRP range's -31 dBm (TS 38.133 §10.1.6), and
/// `None` below the level the UE hears at all.
fn levels(ue: SimPosition) -> [Option<i32>; 3] {
    GNBS.map(|(_, _, at, _)| {
        let dbm = DistanceModel.estimate_dbm(&coord(at), &coord(ue));
        (dbm >= MIN_ALLOWED_DBM).then_some(dbm.min(-31))
    })
}

/// A serving-cell tracker: move to a neighbour once it beats the serving cell
/// by more than `margin_db` for `trigger` (A3 entering condition with its
/// timeToTrigger, or the R-criterion with Treselection).
struct Tracker {
    serving: usize,
    since: [Option<f64>; 3],
    margin_db: i32,
    trigger: f64,
    visited: Vec<usize>,
    /// When each cell after the first was entered, seconds from start-up.
    entered_at: Vec<f64>,
}

impl Tracker {
    fn new(serving: usize, margin_db: i32, trigger: f64) -> Self {
        Self {
            serving,
            since: [None; 3],
            margin_db,
            trigger,
            visited: vec![serving],
            entered_at: Vec::new(),
        }
    }

    fn step(&mut self, t: f64, lv: [Option<i32>; 3]) {
        let serving = lv[self.serving].expect("the serving cell is always heard");
        let mut fired = None;
        for (n, l) in lv.iter().enumerate() {
            let better = n != self.serving && l.is_some_and(|l| l > serving + self.margin_db);
            self.since[n] = if better {
                self.since[n].or(Some(t))
            } else {
                None
            };
            if self.since[n].is_some_and(|s| t - s >= self.trigger) && fired.is_none() {
                fired = Some(n);
            }
        }
        if let Some(n) = fired {
            self.serving = n;
            self.since = [None; 3];
            self.visited.push(n);
            self.entered_at.push(t);
        }
    }
}

fn names(cells: &[usize]) -> Vec<&'static str> {
    cells.iter().map(|&c| GNBS[c].0).collect()
}

/// Every claim the demo makes of the route, checked on `config`'s route (whose
/// phases `r` names) under a trigger time of `trigger` seconds.
fn check_route(config: &UeConfig, r: &LapRoute, trigger: f64, label: &str) {
    let step = 0.1;
    let laps_end = arrival_secs(r, r.laps_end);
    let dwell_end = arrival_secs(r, r.dwell_end);
    let end = arrival_secs(r, r.route.waypoints.len()) + 10.0;
    let at = |t: f64| config.sim_position_at(Duration::from_secs_f64(t));

    // Registration: the UE starts best heard by A.
    let lv0 = levels(at(0.0));
    let best = (0..3).max_by_key(|&c| lv0[c].unwrap_or(i32::MIN)).unwrap();
    assert_eq!(best, A, "{label}: the UE must register on A ({lv0:?})");

    // Connected laps: A3 hands over exactly A -> B -> C -> A, LAPS times.
    let mut connected = Tracker::new(A, 4, trigger);
    let mut t = 0.0;
    while t <= laps_end + trigger + 1.0 {
        connected.step(t, levels(at(t)));
        t += step;
    }
    let mut expected = vec![A];
    for _ in 0..LAPS {
        expected.extend([B, C, A]);
    }
    assert_eq!(
        names(&connected.visited),
        names(&expected),
        "{label}: the connected laps must hand over A->B->C->A {LAPS} times"
    );

    // The dwell: long enough to release in, and no neighbour ever comes within
    // the A3 margin of A, so no handover can start while the harness releases.
    assert!(
        dwell_end - laps_end >= MIN_DWELL_SECS,
        "{label}: the dwell in A lasts {:.1} s, under {MIN_DWELL_SECS} s",
        dwell_end - laps_end
    );
    let mut t = laps_end;
    while t <= dwell_end {
        let lv = levels(at(t));
        let a = lv[A].expect("A is heard throughout the dwell");
        for n in [B, C] {
            assert!(
                lv[n].is_none_or(|l| l <= a + 4),
                "{label}: at {t:.1} s in the dwell {} is within the A3 margin of A ({lv:?})",
                GNBS[n].0
            );
        }
        t += step;
    }

    // Idle, released at the earliest point of the dwell: the R-criterion
    // reselects A -> B and nothing else, and the UE ends camped on B.
    let mut idle = Tracker::new(A, 4, trigger.max(1.0));
    let mut t = laps_end;
    while t <= end {
        idle.step(t, levels(at(t)));
        t += step;
    }
    assert_eq!(
        names(&idle.visited),
        ["A", "B"],
        "{label}: the idle UE must reselect A -> B only"
    );
}

fn config_with(r: &LapRoute) -> UeConfig {
    UeConfig {
        sim_position: r.start,
        sim_route: Some(r.route.clone()),
        ..Default::default()
    }
}

/// The committed `ue-lap.yaml` is a UE config the binary accepts, and its route
/// is exactly seed 464's, so the run reproduces from the seed.
#[test]
fn the_committed_lap_route_is_seed_464s() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../config/demo-nwdaf-paging/ue-lap.yaml"
    );
    let config = nextgsim_ue::app::load_and_validate_ue_config(path)
        .expect("ue-lap.yaml loads and validates");
    config.validate_sim_motion().expect("a travellable route");
    let r = lap_route(COMMITTED_SEED);
    assert_eq!(config.sim_position, r.start, "sim_position");
    assert_eq!(config.sim_route.as_ref(), Some(&r.route), "sim_route");
    assert!(config.sim_movement.is_none());
    for (_, nci, _, ip) in GNBS {
        assert!(
            config.gnb_search_list.iter().any(|g| g == ip),
            "the UE must search gNB {nci:#x} at {ip}"
        );
    }
    for trigger in [0.64, 3.0] {
        check_route(
            &config,
            &r,
            trigger,
            &format!("committed seed {COMMITTED_SEED}, trigger {trigger} s"),
        );
    }
}

/// The geometry holds on every sweep seed, not only the committed one
/// (LEARNINGS 2026-11-12: a route tuned on one seed proves only that seed).
#[test]
fn every_seed_rides_three_laps_dwells_in_a_and_idles_to_b() {
    for seed in SWEEP_SEEDS {
        let r = lap_route(seed);
        r.route.validate().expect("a travellable route");
        let config = config_with(&r);
        for trigger in [0.64, 3.0] {
            check_route(
                &config,
                &r,
                trigger,
                &format!("seed {seed}, trigger {trigger} s"),
            );
        }
    }
}

/// The checks can fail: a route that skips C on its laps, or one whose dwell
/// is too short to release in, is refused.
#[test]
fn the_checks_refuse_a_route_without_the_laps_or_the_dwell() {
    let skip_c = {
        let mut r = lap_route(COMMITTED_SEED);
        r.route.waypoints.retain(|p| distance(*p, LAP_C) > 10.0);
        r.laps_end -= LAPS;
        r.dwell_end -= LAPS;
        r
    };
    let short_dwell = {
        let mut r = lap_route(COMMITTED_SEED);
        r.route.waypoints[r.laps_end] = SimPosition { x: 25, y: 14, z: 0 };
        r
    };
    for (what, r) in [("skip C", skip_c), ("short dwell", short_dwell)] {
        let config = config_with(&r);
        let refused = std::panic::catch_unwind(|| check_route(&config, &r, 0.64, what));
        assert!(
            refused.is_err(),
            "{what}: the checks must refuse this route"
        );
    }
}

/// The modelled run of `seed` with a `trigger` s A3 timeToTrigger: the
/// connected handover times, the dwell, the release at the dwell's far point
/// (where scenario.sh releases), the idle reselection after it, and what the
/// first-order transition count over the reported cells (nwdafd's predictor,
/// fed the registration cell and each handover's target) predicts out of the
/// last one. One row of the docs' seed table.
fn seed_row(seed: u64, trigger: f64) -> String {
    let r = lap_route(seed);
    let config = config_with(&r);
    let at = |t: f64| config.sim_position_at(Duration::from_secs_f64(t));
    let laps_end = arrival_secs(&r, r.laps_end);
    let far = arrival_secs(&r, r.laps_end + 1);
    let end = arrival_secs(&r, r.route.waypoints.len());
    let mut connected = Tracker::new(A, 4, trigger);
    let mut t = 0.0;
    while t <= far {
        connected.step(t, levels(at(t)));
        t += 0.1;
    }
    let mut idle = Tracker::new(A, 4, trigger.max(1.0));
    let mut t = far;
    while t <= end + 10.0 {
        idle.step(t, levels(at(t)));
        t += 0.1;
    }
    let history = &connected.visited;
    let last = *history.last().expect("registered");
    let mut next = [0u32; 3];
    for w in history.windows(2) {
        if w[0] == last {
            next[w[1]] += 1;
        }
    }
    let total: u32 = next.iter().sum();
    let (best, n) = (0..3)
        .map(|c| (c, next[c]))
        .max_by_key(|&(_, n)| n)
        .expect("three cells");
    format!(
        "| {seed} | {} | {:.0} | {:.0}-{:.0} | {:.0} | {} | {:.0} | {} {}/{} |",
        connected.entered_at.len(),
        connected.entered_at.last().copied().unwrap_or(0.0),
        laps_end,
        arrival_secs(&r, r.dwell_end),
        far,
        names(&idle.visited).join("->"),
        idle.entered_at.first().copied().unwrap_or(f64::NAN),
        GNBS[best].0,
        n,
        total
    )
}

/// The rows behave as the docs' table says on the committed seed: nine
/// handovers, the release inside the dwell, A -> B idle, B predicted 3/3.
#[test]
fn the_seed_table_row_of_the_committed_seed() {
    let row = seed_row(COMMITTED_SEED, 0.64);
    let cols: Vec<&str> = row.trim_matches('|').split('|').map(str::trim).collect();
    assert_eq!(cols[0], "464", "{row}");
    assert_eq!(cols[1], "9", "nine handovers: {row}");
    assert_eq!(cols[5], "A->B", "idle A -> B: {row}");
    assert_eq!(cols[7], "B 3/3", "B predicted out of A: {row}");
    let far: f64 = cols[4].parse().expect("far point");
    let dwell: Vec<f64> = cols[3]
        .split('-')
        .map(|v| v.parse().expect("dwell"))
        .collect();
    assert!(
        dwell[0] < far && far < dwell[1],
        "release inside the dwell: {row}"
    );
}

/// Prints the docs' seed table: the committed seed, then every sweep seed.
#[test]
#[ignore = "a generator, run by hand"]
fn print_the_seed_table() {
    println!("| seed | handovers | last handover (s) | dwell (s) | release (s) | idle | reselected (s) | predicted out of A |");
    println!("|---|---|---|---|---|---|---|---|");
    for seed in std::iter::once(COMMITTED_SEED).chain(SWEEP_SEEDS) {
        println!("{}", seed_row(seed, 0.64));
    }
}

/// Prints seed `D440C_SEED`'s (default 464) `sim_position` and `sim_route` as
/// YAML, for `ue-lap.yaml`.
#[test]
#[ignore = "a generator, run by hand"]
fn print_the_lap_route() {
    let seed = std::env::var("D440C_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(COMMITTED_SEED);
    let r = lap_route(seed);
    println!("# D440C lap route, seed {seed}");
    println!(
        "sim_position: {{ x: {}, y: {}, z: 0 }}",
        r.start.x, r.start.y
    );
    println!("sim_route:\n  waypoints:");
    for (i, p) in r.route.waypoints.iter().enumerate() {
        let note = if i < r.laps_end {
            format!("lap {}, {}", i / 3 + 1, ["B", "C", "A"][i % 3])
        } else if i < r.dwell_end {
            ["dwell: out behind A", "dwell: back to A"][i - r.laps_end].to_string()
        } else {
            "idle: to B".to_string()
        };
        println!("    - {{ x: {}, y: {}, z: 0 }}  # {note}", p.x, p.y);
    }
    println!("  speed_mps: {}", r.route.speed_mps);
    println!("  start_after_secs: {}", r.route.start_after_secs);
}
