//! The D440A seeded replay (nextgsim#462): every mode on the same committed
//! route seeds, through the same [`recommend`] the live NWDAF task runs.
//!
//! For each seed of `config/demo-ai-paging/d440a-scenario.yaml`:
//! 1. the route is built from the seed (a constant-turn arc sampled into
//!    `sim_route` waypoints) and the UE's position is read from it with
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
//! The table counts, per mode, the first-attempt hits and the mean PAGING
//! messages per mobile-terminated call. It is checked in
//! (`config/demo-ai-paging/d440a-replay-table.txt`) and asserted
//! reproducible. Nothing asserts that one predictor beats another.
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

    /// The run for `seed`. A draw whose route leaves the coverage box is
    /// redrawn from the same stream, so the seed still fixes the route.
    pub(crate) fn run(&self, seed: u64) -> Run {
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
            let at = |t: f64| {
                let (dx, dy) = if omega.abs() < 1e-9 {
                    (speed * t * heading.cos(), speed * t * heading.sin())
                } else {
                    let r = speed / omega;
                    (
                        r * ((heading + omega * t).sin() - heading.sin()),
                        -r * ((heading + omega * t).cos() - heading.cos()),
                    )
                };
                SimPosition {
                    x: (x0 + dx).round() as i32,
                    y: (y0 + dy).round() as i32,
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

/// The table's line for one mode.
pub(crate) fn table_line(mode: &str, outcomes: &[Outcome]) -> String {
    let n = outcomes.len();
    let hits = outcomes.iter().filter(|o| o.first_attempt_hit).count();
    let pagings: usize = outcomes.iter().map(|o| o.pagings).sum();
    format!(
        "| {mode:<11} | {n:>5} | {hits:>18} | {:>7.1}% | {:>17.2} |",
        100.0 * hits as f64 / n as f64,
        pagings as f64 / n as f64
    )
}

/// The committed table's line for `mode`.
pub(crate) fn committed_line(mode: &str) -> String {
    let text = std::fs::read_to_string(TABLE).expect("read the committed table");
    text.lines()
        .find(|l| l.starts_with(&format!("| {mode:<11} |")))
        .unwrap_or_else(|| panic!("no line for {mode} in {TABLE}"))
        .to_string()
}

pub(crate) fn run_mode(
    file: &ScenarioFile,
    mode: Option<Predictor>,
    prepare: &dyn Fn(&mut NwdafManager),
) -> Vec<Outcome> {
    file.replay_seeds()
        .map(|seed| replay(file, &file.run(seed), mode, prepare))
        .collect()
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

    /// #462: the five-arm table's four model-free rows, replayed from the
    /// committed seeds, are exactly the committed table. `page-all` is the
    /// check that the replay is honest: with no IE every first attempt pages
    /// the whole area, so its hit rate is 100% at 6 PAGINGs per call.
    #[test]
    fn the_replay_reproduces_the_committed_table() {
        let file = scenario_file();
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
            lines.push((name, table_line(name, &outcomes)));
        }
        for (_, line) in &lines {
            println!("{line}");
        }
        for (name, line) in &lines {
            assert_eq!(*line, committed_line(name), "{name}");
        }
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
        let outcomes = run_mode(&file, Some(Predictor::Onnx), &|nwdaf| {
            nwdaf
                .load_verified_trajectory_model(&model)
                .expect("the reference model verifies");
        });
        assert!(
            outcomes.iter().all(|o| o.served_by == "onnx"),
            "every prediction served by the verified model"
        );
        let line = table_line("onnx", &outcomes);
        println!("{line}");
        assert_eq!(line, committed_line("onnx"));
    }
}
