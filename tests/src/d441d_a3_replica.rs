//! Demo D441D (nextgsim#467): the gNB's shadow A3 replica against the UE's own
//! A3 evaluator, on the same seeded series.
//!
//! The UE evaluates its measurement events once per RRC cycle, on the levels of
//! the latest heartbeat acks, at a phase the gNB cannot know. The replica runs
//! that evaluation at every phase of a 100 ms grid and reports the A3 time as a
//! range. This test runs the UE's real `MeasurementManager`, on its own clock,
//! at UE phases off that grid and with heartbeats at two offsets, and checks
//! that the UE's first A3 report always falls inside the replica's range.

use std::time::{Duration, Instant};

use nextgsim_common::config::{GnbConfig, SimPosition, UeConfig};
use nextgsim_gnb::nwdaf::d441d::{Engine, Event, PHASE_STEP_MS, UE_EVAL_PERIOD_MS};
use nextgsim_rls::{ChannelModel, DistanceModel, ShadowedModel, Shadowing, SimCoord};
use nextgsim_ue::rrc::{MeasConfig, MeasurementManager};

const WEST_CELL: i32 = 0;
const EAST_CELL: i32 = 1;
const EAST_NCI: u64 = 0x20;

fn config(name: &str) -> String {
    std::fs::read_to_string(format!(
        "{}/../config/demo-predictive-ho/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}

fn level(g: &GnbConfig, nci: u64, gnb: SimPosition, p: SimPosition) -> i32 {
    let s = g.shadowing;
    ShadowedModel::new(
        Box::new(DistanceModel),
        Shadowing::new(s.sigma_db, s.seed, s.decorrelation_m, nci),
    )
    .estimate_dbm(
        &SimCoord::new(gnb.x, gnb.y, gnb.z),
        &SimCoord::new(p.x, p.y, p.z),
    )
}

/// One heartbeat: when, west's level, east's level, where.
type Beat = (u64, i32, i32, SimPosition);

/// The heartbeats of the committed route under `west`'s shadowing, every 1 s
/// from `t0 + offset`.
fn beats(west: &GnbConfig, t0: u64, offset: u64) -> Vec<Beat> {
    let ue: UeConfig = serde_yaml::from_str(&config("ue.yaml")).unwrap();
    let mut route = ue.sim_route.unwrap();
    route.start_after_secs = 10.0;
    let east_pos = west.handover_neighbours[0].position.unwrap();
    (0..60u64)
        .map(|s| {
            let at = Duration::from_millis(s * 1000 + offset);
            let p = route.position_at(ue.sim_position, at);
            let t = t0 + s * 1000 + offset;
            (
                t,
                level(west, west.nci, west.sim_position, p),
                level(west, EAST_NCI, east_pos, p),
                p,
            )
        })
        .collect()
}

/// The replica's A3 range for the series.
fn replica(west: &GnbConfig, beats: &[Beat], t0: u64) -> (u64, u64) {
    let mut e = Engine::from_config(west).unwrap();
    e.set_served(&[1], t0);
    for &(t, mp, _, p) in beats {
        for ev in e.on_measurement(1, mp, SimCoord::new(p.x, p.y, p.z), t) {
            if let Event::A3WouldTrigger {
                target_nci,
                t_lo_ms,
                t_hi_ms,
                ..
            } = ev
            {
                assert_eq!(target_nci, EAST_NCI);
                return (t_lo_ms, t_hi_ms);
            }
        }
    }
    panic!("the replica never completed A3");
}

/// The UE's first A3 report time, its RRC cycle at `phase` ms, with the A3
/// configuration the gNB signals.
fn ue_a3(west: &GnbConfig, beats: &[Beat], t0: u64, phase: u64) -> u64 {
    let mut m = MeasurementManager::new();
    let mut c = MeasConfig::default();
    c.trigger_config.a3_offset = Some((west.cho_a3_offset_db * 2.0).round() as i32);
    c.trigger_config.hysteresis = (west.cho_hysteresis_db * 2.0).round() as i32;
    c.trigger_config.time_to_trigger = u64::from(nextgsim_gnb::rrc::meas::A3_TIME_TO_TRIGGER_MS);
    m.add_config(c);
    m.set_serving_cell(Some(WEST_CELL));
    let clock = Instant::now();
    let mut tau = t0 + phase;
    while tau < t0 + 60_000 {
        // The latest ack the UE holds at tau.
        if let Some(&(_, mp, mn, _)) = beats.iter().rev().find(|b| b.0 <= tau) {
            m.update_measurement(WEST_CELL, mp);
            m.update_measurement(EAST_CELL, mn);
            m.evaluate_events_at(clock + Duration::from_millis(tau - t0));
            if !m.take_pending_reports().is_empty() {
                return tau;
            }
        }
        // The UE's own cycle, not the replica's constant: the two are compared
        // by `d441d_replica_runs_at_the_ue_evaluation_period`.
        tau += nextgsim_ue::rrc::task::RRC_CYCLE_INTERVAL_MS;
    }
    panic!("the UE never reported A3");
}

/// The replica evaluates on the UE's own cadence.
#[test]
fn d441d_replica_runs_at_the_ue_evaluation_period() {
    assert_eq!(
        UE_EVAL_PERIOD_MS,
        nextgsim_ue::rrc::task::RRC_CYCLE_INTERVAL_MS
    );
}

/// On 24 seeds, two heartbeat offsets and fourteen UE phases (none on the
/// replica's 100 ms grid but 0), the UE's first A3 report is inside the
/// replica's `[lo, hi)`, and the range is at most one UE period plus two grid
/// steps wide.
#[test]
fn d441d_replica_brackets_the_ue_a3() {
    let t0 = 1_000_000u64;
    let mut widths = Vec::new();
    for seed in 1..=24u64 {
        let mut west: GnbConfig = serde_yaml::from_str(&config("gnb-west.yaml")).unwrap();
        west.shadowing.seed = seed;
        for offset in [0u64, 370] {
            let series = beats(&west, t0, offset);
            let (lo, hi) = replica(&west, &series, t0);
            assert!(
                hi - lo <= UE_EVAL_PERIOD_MS + 2 * PHASE_STEP_MS,
                "{lo}..{hi}"
            );
            widths.push(hi - lo);
            for phase in [
                0u64, 37, 99, 101, 413, 999, 1250, 1369, 1371, 1871, 2001, 2399, 2401, 2499,
            ] {
                let t_ue = ue_a3(&west, &series, t0, phase);
                assert!(
                    lo <= t_ue && t_ue < hi,
                    "seed {seed} offset {offset} phase {phase}: UE at {t_ue}, replica {lo}..{hi}"
                );
            }
        }
    }
    eprintln!("D441D replica range widths (ms): {widths:?}");
}
