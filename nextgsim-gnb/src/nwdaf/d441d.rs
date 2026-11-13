//! D441D, predictive N2 handover vs A3 (nextgsim#467): the decision core.
//!
//! Demo-only code, compiled only with the `demo-d441d-predictive-ho` feature
//! and run only while `demo.d441d-predictive-ho` is active. Pure: every call
//! takes the time it runs at, so the seeded replay below is deterministic.
//!
//! Two things are evaluated per UE, on the levels the RLS link already computes:
//!
//! - **The predictor** (modes `lsq`, `onnx`). Once a tick, for each UE this gNB
//!   serves: the NWDAF trajectory prediction `HORIZON_MS` ahead must be
//!   mobility ([`nextgsim_nwdaf::is_predicted_mobility`]: confidence above 0.7
//!   and moving), the measured serving level must be falling, and at the
//!   predicted position a positioned neighbour must beat this cell by the same
//!   A3 margin the UE is configured with (`a3-Offset` plus hysteresis,
//!   TS 38.331 §5.5.4.4), under the distance model alone. The predictor knows the
//!   neighbour positions and the distance model, not the shadowing field.
//! - **The shadow A3 evaluator** (every mode). On every heartbeat measurement:
//!   inequality A3-1 between this cell's measured level and each neighbour's
//!   level at the UE's position, held for `timeToTrigger`. The neighbour level
//!   is the one the UE measures: the same [`ShadowedModel`] over the
//!   [`DistanceModel`], keyed by the neighbour's NCI, with this gNB's
//!   `shadowing` block (every gNB of a run uses the same one). It keeps running
//!   after the UE has been handed away, because the UE still heartbeats this
//!   cell, so the A3 time is known even when the predictive handover came first.
//!   This is a gNB-side replica of the UE's A3 entering condition, at heartbeat
//!   cadence, and not the UE's report itself.
//!
//! Lead time is `A3 time - predictive time` for one UE, one target, one serving
//! episode. It is not asserted positive: shadowing can bring A3 forward.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use nextgsim_common::config::GnbConfig;
use nextgsim_nwdaf::{is_predicted_mobility, NwdafManager, TrajectoryPrediction, Vector3};
use nextgsim_rls::{ChannelModel, DistanceModel, ShadowedModel, Shadowing, SimCoord};

/// How often the predictor runs.
pub const TICK_MS: u64 = 1000;

/// How far ahead the predictor looks, in ms.
pub const HORIZON_MS: u32 = 2000;

/// The ping-pong guard: a UE served here for less than this is not handed away
/// by the predictor.
pub const MIN_DWELL_MS: u64 = 5000;

/// The most predictive attempts per UE per serving episode (#513's lesson: a
/// retry carries a cap and an `exhausted` line).
pub const MAX_ATTEMPTS: u32 = 3;

/// A UE still served this long after an attempt may be attempted again.
pub const RETRY_AFTER_MS: u64 = 3000;

/// `timeToTrigger` of the A3 reporting configuration this gNB signals.
pub const A3_TTT_MS: u64 = crate::rrc::meas::A3_TIME_TO_TRIGGER_MS as u64;

/// How many serving-level samples the RSRP trend is fitted over (the trajectory
/// fit's own window).
const RSRP_WINDOW: usize = nextgsim_nwdaf::predictor::LSQ_WINDOW;

/// How many lead times `status` keeps.
const LEADS_KEPT: usize = 32;

/// What produced a prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Predictor {
    /// The least-squares fit (mode `lsq`).
    Lsq,
    /// The verified ONNX model (mode `onnx`).
    Onnx,
    /// Mode `onnx`, but no verified model served it: the least-squares fit did.
    Fallback,
}

impl Predictor {
    /// The `predictor=` value in the log.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Lsq => "lsq",
            Self::Onnx => "onnx",
            Self::Fallback => "fallback",
        }
    }
}

/// Something the engine decided, for the task to log and act on.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// The predictor recommends handing `ue_id` over to `target_nci`.
    Prediction {
        /// The UE.
        ue_id: i32,
        /// The target cell's NCI.
        target_nci: u64,
        /// What produced the prediction.
        predictor: Predictor,
        /// The prediction's confidence.
        confidence: f32,
        /// 1..=[`MAX_ATTEMPTS`].
        attempt: u32,
    },
    /// The UE is still served after [`MAX_ATTEMPTS`] attempts: no more.
    Exhausted {
        /// The UE.
        ue_id: i32,
        /// The last target.
        target_nci: u64,
    },
    /// The shadow A3 evaluator's entering condition held for `timeToTrigger`.
    A3WouldTrigger {
        /// The UE.
        ue_id: i32,
        /// The neighbour that entered.
        target_nci: u64,
        /// When, in ms.
        t_ms: u64,
        /// This cell's level at the UE, dBm.
        serving_dbm: i32,
        /// The neighbour's level at the UE, dBm.
        neighbour_dbm: i32,
    },
    /// Both times are known for one serving episode.
    LeadTime {
        /// The UE.
        ue_id: i32,
        /// The predictive target.
        target_nci: u64,
        /// The first predictive attempt, ms.
        predictive_ms: u64,
        /// The shadow A3 trigger, ms.
        a3_ms: u64,
        /// `a3_ms - predictive_ms`.
        lead_ms: i64,
    },
}

struct Neighbour {
    nci: u64,
    pos: SimCoord,
    /// The level the UE measures from this neighbour.
    model: ShadowedModel,
}

#[derive(Default)]
struct UeState {
    /// When this gNB's RRC last reported the UE as newly served.
    served_since_ms: Option<u64>,
    /// `(t, dBm)` of this cell at the UE, newest last.
    rsrp: VecDeque<(u64, i32)>,
    /// A3-1 entering since, per neighbour NCI.
    entering_since: HashMap<u64, u64>,
    /// The shadow A3 trigger of this episode.
    a3: Option<(u64, u64)>,
    /// The first predictive attempt of this episode: `(target, t)`.
    predicted: Option<(u64, u64)>,
    attempts: u32,
    last_attempt_ms: Option<u64>,
    last_target: Option<u64>,
    exhausted: bool,
}

/// Counters for `nwdaf-status`.
#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct Stats {
    /// Predictive recommendations sent to RRC.
    pub predictions: u64,
    /// UEs whose attempts ran out.
    pub exhausted: u64,
    /// Shadow A3 triggers.
    pub a3_would_trigger: u64,
    /// The latest lead times, ms, newest last.
    pub lead_ms: Vec<i64>,
}

/// The D441D decision core of one gNB.
pub struct Engine {
    own_pos: SimCoord,
    neighbours: Vec<Neighbour>,
    /// `a3-Offset` and hysteresis in 0.5 dB units, as signalled.
    off_half: i32,
    hys_half: i32,
    ues: BTreeMap<i32, UeState>,
    served: BTreeSet<i32>,
    stats: Stats,
}

fn coord(p: nextgsim_common::config::SimPosition) -> SimCoord {
    SimCoord::new(p.x, p.y, p.z)
}

fn half_db(db: f64) -> i32 {
    (db * 2.0).round() as i32
}

/// Least-squares slope of `(t ms, dBm)` in dB/s; `None` below three samples.
fn slope_db_per_s(samples: &VecDeque<(u64, i32)>) -> Option<f64> {
    if samples.len() < 3 {
        return None;
    }
    let t0 = samples.back()?.0 as f64;
    let n = samples.len() as f64;
    let ts: Vec<f64> = samples
        .iter()
        .map(|(t, _)| (*t as f64 - t0) / 1000.0)
        .collect();
    let tm = ts.iter().sum::<f64>() / n;
    let ym = samples.iter().map(|(_, y)| f64::from(*y)).sum::<f64>() / n;
    let sxx: f64 = ts.iter().map(|t| (t - tm).powi(2)).sum();
    if sxx <= 0.0 {
        return None;
    }
    let sxy: f64 = ts
        .iter()
        .zip(samples)
        .map(|(t, (_, y))| (t - tm) * (f64::from(*y) - ym))
        .sum();
    Some(sxy / sxx)
}

fn distance(a: &SimCoord, b: &Vector3) -> f64 {
    let (dx, dy, dz) = (
        f64::from(a.x) - b.x,
        f64::from(a.y) - b.y,
        f64::from(a.z) - b.z,
    );
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// The least-squares prediction for `ue_id` from `nwdaf`'s history, anchored at
/// `now_ms`: what mode `lsq` predicts with, whatever model is loaded.
pub fn lsq_prediction(
    nwdaf: &NwdafManager,
    ue_id: i32,
    horizon_ms: u32,
    now_ms: u64,
) -> Option<TrajectoryPrediction> {
    let history = nwdaf.data_collector().get_ue_history(ue_id)?;
    if history.len() < 2 {
        return None;
    }
    let positions = history.position_sequence();
    let timestamps: Vec<u64> = history.all().iter().map(|m| m.timestamp_ms).collect();
    let out = nextgsim_nwdaf::predictor::least_squares_trajectory(
        &positions,
        &timestamps,
        horizon_ms,
        now_ms,
    );
    Some(TrajectoryPrediction {
        ue_id,
        waypoints: out.waypoints,
        confidence: out.confidence,
        horizon_ms,
    })
}

impl Engine {
    /// The engine for this gNB, or why the config cannot run the demo: it needs
    /// at least one `handover_neighbours` entry with a `position`.
    pub fn from_config(config: &GnbConfig) -> Result<Self, String> {
        let s = config.shadowing;
        let neighbours: Vec<Neighbour> = config
            .handover_neighbours
            .iter()
            .filter_map(|n| {
                n.position.map(|p| Neighbour {
                    nci: n.nci & 0xF_FFFF_FFFF,
                    pos: coord(p),
                    // The neighbour's own tracker keys its field by its NCI
                    // (`rls/task.rs` `build_cell_tracker`, sti = nci).
                    model: ShadowedModel::new(
                        Box::new(DistanceModel),
                        Shadowing::new(s.sigma_db, s.seed, s.decorrelation_m, n.nci),
                    ),
                })
            })
            .collect();
        if neighbours.is_empty() {
            return Err(
                "d441d-predictive-ho needs at least one handover_neighbours entry with a \
                 position (where the target cell stands)"
                    .into(),
            );
        }
        Ok(Self {
            own_pos: coord(config.sim_position),
            neighbours,
            off_half: half_db(config.cho_a3_offset_db),
            hys_half: half_db(config.cho_hysteresis_db),
            ues: BTreeMap::new(),
            served: BTreeSet::new(),
            stats: Stats::default(),
        })
    }

    /// The counters.
    pub fn stats(&self) -> &Stats {
        &self.stats
    }

    /// How many UEs this gNB serves, as last reported.
    pub fn served_count(&self) -> usize {
        self.served.len()
    }

    /// RRC's served set. A UE that newly appears starts a serving episode: its
    /// dwell clock, attempts, and A3 and predictive times start afresh.
    pub fn set_served(&mut self, ue_ids: &[i32], now_ms: u64) {
        let now: BTreeSet<i32> = ue_ids.iter().copied().collect();
        for &ue in now.difference(&self.served) {
            let st = self.ues.entry(ue).or_default();
            st.served_since_ms = Some(now_ms);
            st.entering_since.clear();
            st.a3 = None;
            st.predicted = None;
            st.attempts = 0;
            st.last_attempt_ms = None;
            st.last_target = None;
            st.exhausted = false;
        }
        self.served = now;
    }

    /// One heartbeat measurement of `ue_id`: this cell's level at the UE and
    /// where it is. Feeds the RSRP trend and the shadow A3 evaluator.
    pub fn on_measurement(
        &mut self,
        ue_id: i32,
        serving_dbm: i32,
        pos: SimCoord,
        now_ms: u64,
    ) -> Vec<Event> {
        let served = self.served.contains(&ue_id);
        let st = self.ues.entry(ue_id).or_default();
        st.rsrp.push_back((now_ms, serving_dbm));
        while st.rsrp.len() > RSRP_WINDOW {
            st.rsrp.pop_front();
        }
        // Served here, or handed away by the predictor and not yet seen by A3.
        if st.a3.is_some() || !(served || st.predicted.is_some()) {
            return Vec::new();
        }
        let mut fired: Option<(u64, i32)> = None;
        for nb in &self.neighbours {
            let mn = nb.model.estimate_dbm(&nb.pos, &pos);
            // A3-1, in 0.5 dB units: Mn - Hys > Mp + Off.
            if 2 * mn - self.hys_half > 2 * serving_dbm + self.off_half {
                let since = *st.entering_since.entry(nb.nci).or_insert(now_ms);
                if now_ms.saturating_sub(since) >= A3_TTT_MS
                    && fired.is_none_or(|(_, best)| mn > best)
                {
                    fired = Some((nb.nci, mn));
                }
            } else {
                st.entering_since.remove(&nb.nci);
            }
        }
        let Some((target_nci, neighbour_dbm)) = fired else {
            return Vec::new();
        };
        st.a3 = Some((target_nci, now_ms));
        self.stats.a3_would_trigger += 1;
        let mut events = vec![Event::A3WouldTrigger {
            ue_id,
            target_nci,
            t_ms: now_ms,
            serving_dbm,
            neighbour_dbm,
        }];
        if let Some((target, t)) = st.predicted {
            events.push(self.lead(ue_id, target, t, now_ms));
        }
        events
    }

    fn lead(&mut self, ue_id: i32, target_nci: u64, predictive_ms: u64, a3_ms: u64) -> Event {
        let lead_ms = a3_ms as i64 - predictive_ms as i64;
        self.stats.lead_ms.push(lead_ms);
        if self.stats.lead_ms.len() > LEADS_KEPT {
            self.stats.lead_ms.remove(0);
        }
        Event::LeadTime {
            ue_id,
            target_nci,
            predictive_ms,
            a3_ms,
            lead_ms,
        }
    }

    /// One predictor tick. `predict` gives a UE's trajectory prediction
    /// [`HORIZON_MS`] ahead and what produced it. With `predictive` false (mode
    /// `a3`) nothing is predicted.
    pub fn tick<F>(&mut self, now_ms: u64, predictive: bool, mut predict: F) -> Vec<Event>
    where
        F: FnMut(i32) -> Option<(TrajectoryPrediction, Predictor)>,
    {
        let mut events = Vec::new();
        if !predictive {
            return events;
        }
        let served: Vec<i32> = self.served.iter().copied().collect();
        for ue_id in served {
            let st = self.ues.entry(ue_id).or_default();
            if st.exhausted {
                continue;
            }
            let since = st.served_since_ms.unwrap_or(now_ms);
            if now_ms.saturating_sub(since) < MIN_DWELL_MS {
                continue;
            }
            if let Some(last) = st.last_attempt_ms {
                if now_ms.saturating_sub(last) < RETRY_AFTER_MS {
                    continue;
                }
                if st.attempts >= MAX_ATTEMPTS {
                    st.exhausted = true;
                    self.stats.exhausted += 1;
                    events.push(Event::Exhausted {
                        ue_id,
                        target_nci: st.last_target.unwrap_or_default(),
                    });
                    continue;
                }
            }
            // The measured serving level must be falling.
            if !slope_db_per_s(&st.rsrp).is_some_and(|s| s < 0.0) {
                continue;
            }
            let Some((prediction, predictor)) = predict(ue_id) else {
                continue;
            };
            if !is_predicted_mobility(prediction.confidence, &prediction.waypoints) {
                continue;
            }
            let Some((end, _)) = prediction.waypoints.last() else {
                continue;
            };
            let serving = -distance(&self.own_pos, end);
            let target = self
                .neighbours
                .iter()
                .map(|nb| (nb.nci, -distance(&nb.pos, end)))
                .filter(|(_, mn)| {
                    2.0 * mn - f64::from(self.hys_half) > 2.0 * serving + f64::from(self.off_half)
                })
                .max_by(|a, b| a.1.total_cmp(&b.1));
            let Some((target_nci, _)) = target else {
                continue;
            };
            st.attempts += 1;
            st.last_attempt_ms = Some(now_ms);
            st.last_target = Some(target_nci);
            let first = st.predicted.is_none();
            if first {
                st.predicted = Some((target_nci, now_ms));
            }
            let a3 = st.a3;
            let attempt = st.attempts;
            self.stats.predictions += 1;
            events.push(Event::Prediction {
                ue_id,
                target_nci,
                predictor,
                confidence: prediction.confidence,
                attempt,
            });
            if let (true, Some((_, a3_ms))) = (first, a3) {
                events.push(self.lead(ue_id, target_nci, now_ms, a3_ms));
            }
        }
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nextgsim_common::config::{
        HandoverNeighbourConfig, ShadowingConfig, SimPosition, SimRoute,
    };
    use nextgsim_nwdaf::UeMeasurement;

    const WEST: u64 = 0x10;
    const EAST: u64 = 0x20;

    fn pos(x: i32, y: i32) -> SimPosition {
        SimPosition { x, y, z: 0 }
    }

    /// gnb-west at the origin with gnb-east at x 120 as its neighbour: the
    /// geometry of nextgcore's two-gNB stack (#456) and the committed configs.
    fn west(sigma_db: f64, seed: u64) -> GnbConfig {
        GnbConfig {
            nci: WEST,
            sim_position: pos(0, 0),
            shadowing: ShadowingConfig {
                sigma_db,
                seed,
                ..ShadowingConfig::default()
            },
            handover_neighbours: vec![HandoverNeighbourConfig {
                nci: EAST,
                tac: 1,
                gnb_id_length: None,
                position: Some(pos(120, 0)),
            }],
            ..GnbConfig::default()
        }
    }

    /// The level the RLS link gives a UE at `p` from a gNB (`ack.dbm`).
    fn level(config: &GnbConfig, nci: u64, gnb: SimPosition, p: SimPosition) -> i32 {
        let s = config.shadowing;
        ShadowedModel::new(
            Box::new(DistanceModel),
            Shadowing::new(s.sigma_db, s.seed, s.decorrelation_m, nci),
        )
        .estimate_dbm(&coord(gnb), &coord(p))
    }

    /// The committed route's start (config/demo-predictive-ho/ue.yaml).
    const START: SimPosition = SimPosition { x: 20, y: 17, z: 0 };

    /// The committed route: an arc, `y = 30 - (x - 60)^2 / 120` sampled every
    /// 10 m from x 30 to 110, at 5 m/s, moving 10 s after the start (the
    /// config starts later; only the motion matters here).
    fn route() -> SimRoute {
        let text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../config/demo-predictive-ho/ue.yaml"
        ))
        .expect("read the committed UE config");
        let ue: nextgsim_common::config::UeConfig =
            serde_yaml::from_str(&text).expect("the committed UE config parses");
        assert_eq!(ue.sim_position, START);
        SimRoute {
            start_after_secs: 10.0,
            ..ue.sim_route
                .expect("the committed UE config has a sim_route")
        }
    }

    /// What one seeded run of gnb-west's engine saw.
    #[derive(Debug, Default)]
    struct Run {
        predictive_ms: Option<u64>,
        a3_ms: Option<u64>,
        lead_ms: Option<i64>,
        predictions: u32,
    }

    /// Replay the route at heartbeat cadence (1 s) through gnb-west's engine, as
    /// the NWDAF task drives it: measurements recorded in the manager and fed to
    /// the shadow A3 evaluator, a tick after each. The UE is served from t 0;
    /// after a predictive trigger it is handed away (no longer served) and keeps
    /// heartbeating, as it does over RLS.
    fn replay(config: &GnbConfig, predictive: bool) -> Run {
        let mut engine = Engine::from_config(config).unwrap();
        let mut nwdaf = NwdafManager::new(100);
        let (ue, t0) = (1, 1_000_000u64);
        engine.set_served(&[ue], t0);
        let mut run = Run::default();
        let route = route();
        for s in 0..60u64 {
            let now = t0 + s * 1000;
            let p = route.position_at(START, std::time::Duration::from_secs(s));
            let dbm = level(config, WEST, pos(0, 0), p);
            nwdaf.record_measurement(UeMeasurement {
                ue_id: ue,
                rsrp: dbm as f32,
                rsrq: None,
                sinr: None,
                position: Vector3::new(f64::from(p.x), f64::from(p.y), 0.0),
                velocity: None,
                serving_cell_id: WEST as i32,
                timestamp_ms: now,
            });
            let mut events = engine.on_measurement(ue, dbm, coord(p), now);
            events.extend(engine.tick(now, predictive, |id| {
                lsq_prediction(&nwdaf, id, HORIZON_MS, now).map(|p| (p, Predictor::Lsq))
            }));
            for e in events {
                match e {
                    Event::Prediction { target_nci, .. } => {
                        assert_eq!(target_nci, EAST);
                        run.predictions += 1;
                        run.predictive_ms.get_or_insert(now);
                        engine.set_served(&[], now); // handed over
                    }
                    Event::A3WouldTrigger { target_nci, .. } => {
                        assert_eq!(target_nci, EAST);
                        run.a3_ms = Some(now);
                    }
                    Event::LeadTime { lead_ms, .. } => run.lead_ms = Some(lead_ms),
                    Event::Exhausted { .. } => panic!("one attempt hands the UE away"),
                }
            }
        }
        run
    }

    /// A3-1 has to hold for timeToTrigger: one sample past the margin is not
    /// enough, two a heartbeat apart are; and once triggered it is not
    /// re-reported in the episode. A sample back inside the margin restarts it.
    #[test]
    fn the_shadow_a3_needs_the_margin_for_time_to_trigger() {
        let config = west(0.0, 1);
        let mut e = Engine::from_config(&config).unwrap();
        e.set_served(&[1], 0);
        // At x 62 east is -58, west -62: 4 dB, not past 3 + 1 dB.
        assert!(e
            .on_measurement(1, -62, SimCoord::new(62, 0, 0), 0)
            .is_empty());
        // At x 63: east -57 vs west -63, 6 dB > 4 dB: entering, TTT starts.
        assert!(e
            .on_measurement(1, -63, SimCoord::new(63, 0, 0), 1000)
            .is_empty());
        // Back to 4 dB: the run is broken.
        assert!(e
            .on_measurement(1, -62, SimCoord::new(62, 0, 0), 2000)
            .is_empty());
        assert!(e
            .on_measurement(1, -63, SimCoord::new(63, 0, 0), 3000)
            .is_empty());
        let ev = e.on_measurement(1, -64, SimCoord::new(64, 0, 0), 4000);
        assert_eq!(
            ev,
            vec![Event::A3WouldTrigger {
                ue_id: 1,
                target_nci: EAST,
                t_ms: 4000,
                serving_dbm: -64,
                neighbour_dbm: -56,
            }]
        );
        assert!(e
            .on_measurement(1, -70, SimCoord::new(70, 0, 0), 5000)
            .is_empty());
        assert_eq!(e.stats().a3_would_trigger, 1);
    }

    /// The neighbour level the evaluator uses is the shadowed one the UE
    /// measures, keyed by the neighbour's NCI: at a point where shadowing moves
    /// east's level across the margin, the verdict differs from σ 0.
    #[test]
    fn the_shadow_a3_sees_the_neighbours_shadowed_level() {
        // Find a seed and a point near the crossing where east's offset alone
        // decides: σ 0 says no (east 4 dB better), the shadowed level says yes.
        let mut found = None;
        'seeds: for seed in 1..200u64 {
            let c = west(6.0, seed);
            for x in 55..62 {
                let p = pos(x, 0);
                let mn = level(&c, EAST, pos(120, 0), p);
                if 2 * mn - 2 > 2 * (-x) + 6 {
                    found = Some((c, x));
                    break 'seeds;
                }
            }
        }
        let (shadowed, x) = found.expect("some seed lifts east past the margin");
        for (config, expect) in [(west(0.0, 1), false), (shadowed, true)] {
            let mut e = Engine::from_config(&config).unwrap();
            e.set_served(&[1], 0);
            // This cell's own level is passed in unshadowed in both, so only
            // the neighbour's level differs.
            e.on_measurement(1, -x, SimCoord::new(x, 0, 0), 0);
            let fired = !e
                .on_measurement(1, -x, SimCoord::new(x, 0, 0), 1000)
                .is_empty();
            assert_eq!(fired, expect, "x {x}");
        }
    }

    /// A UE that is neither served here nor handed away by the predictor is not
    /// evaluated: the target gNB hears every UE too.
    #[test]
    fn the_shadow_a3_ignores_a_ue_this_gnb_never_served() {
        let mut e = Engine::from_config(&west(0.0, 1)).unwrap();
        for t in 0..5 {
            assert!(e
                .on_measurement(9, -80, SimCoord::new(80, 0, 0), t * 1000)
                .is_empty());
        }
    }

    fn straight(engine: &mut Engine, nwdaf: &mut NwdafManager, ue: i32, from_s: u64, to_s: u64) {
        for s in from_s..to_s {
            let x = 20.0 + 5.0 * s as f64;
            nwdaf.record_measurement(UeMeasurement {
                ue_id: ue,
                rsrp: -x as f32,
                rsrq: None,
                sinr: None,
                position: Vector3::new(x, 0.0, 0.0),
                velocity: None,
                serving_cell_id: 16,
                timestamp_ms: s * 1000,
            });
            engine.on_measurement(ue, -(x as i32), SimCoord::new(x as i32, 0, 0), s * 1000);
        }
    }

    fn lsq(
        nwdaf: &NwdafManager,
        now: u64,
    ) -> impl FnMut(i32) -> Option<(TrajectoryPrediction, Predictor)> + '_ {
        move |id| lsq_prediction(nwdaf, id, HORIZON_MS, now).map(|p| (p, Predictor::Lsq))
    }

    /// On a straight walk east at 5 m/s the predictor names east once the
    /// position HORIZON_MS ahead is past the A3 margin, which is before A3 sees
    /// it; mode `a3` predicts nothing.
    #[test]
    fn the_predictor_names_the_neighbour_ahead_of_a3() {
        let mut nwdaf = NwdafManager::new(100);
        let mut e = Engine::from_config(&west(0.0, 1)).unwrap();
        e.set_served(&[1], 0);
        // x 20 .. 50 over 7 s: at 2 s ahead x <= 60, east not yet 4 dB better.
        straight(&mut e, &mut nwdaf, 1, 0, 7);
        assert!(e.tick(6000, true, lsq(&nwdaf, 6000)).is_empty());
        // x 55 at 7 s: ahead x 65, east -55 vs west -65.
        straight(&mut e, &mut nwdaf, 1, 7, 8);
        assert!(e.tick(7000, false, lsq(&nwdaf, 7000)).is_empty(), "mode a3");
        let ev = e.tick(7000, true, lsq(&nwdaf, 7000));
        match ev.as_slice() {
            [Event::Prediction {
                ue_id: 1,
                target_nci: EAST,
                predictor: Predictor::Lsq,
                confidence,
                attempt: 1,
            }] => assert!(*confidence > 0.7, "{confidence}"),
            other => panic!("{other:?}"),
        }
        assert_eq!(e.stats().a3_would_trigger, 0, "A3 has not seen it yet");
    }

    /// A UE that stands still, or wanders with no trend, is not predicted.
    #[test]
    fn the_predictor_needs_mobility() {
        let mut nwdaf = NwdafManager::new(100);
        let mut e = Engine::from_config(&west(0.0, 1)).unwrap();
        e.set_served(&[1], 0);
        // Scattered positions near x 70 (past the margin), with a falling level so
        // only the mobility test can refuse.
        for s in 0..10u64 {
            let x = [70.0, 76.0, 64.0, 71.0, 66.0, 75.0, 63.0, 72.0, 69.0, 74.0][s as usize];
            let y = [0.0, 9.0, -8.0, 4.0, -9.0, 7.0, -3.0, 8.0, -6.0, 2.0][s as usize];
            nwdaf.record_measurement(UeMeasurement {
                ue_id: 1,
                rsrp: -60.0 - s as f32,
                rsrq: None,
                sinr: None,
                position: Vector3::new(x, y, 0.0),
                velocity: None,
                serving_cell_id: 16,
                timestamp_ms: s * 1000,
            });
            e.on_measurement(1, -60 - s as i32, SimCoord::new(0, 0, 0), s * 1000);
        }
        assert!(e.tick(9000, true, lsq(&nwdaf, 9000)).is_empty());
    }

    /// The serving level must be falling: the same walk with this cell's
    /// measured level rising (e.g. a shadowing upswing) is not predicted away.
    #[test]
    fn the_predictor_needs_a_falling_serving_level() {
        for (rising, expect) in [(true, false), (false, true)] {
            let mut nwdaf = NwdafManager::new(100);
            let mut e = Engine::from_config(&west(0.0, 1)).unwrap();
            e.set_served(&[1], 0);
            for s in 0..8u64 {
                let x = 20.0 + 5.0 * s as f64;
                nwdaf.record_measurement(UeMeasurement {
                    ue_id: 1,
                    rsrp: -x as f32,
                    rsrq: None,
                    sinr: None,
                    position: Vector3::new(x, 0.0, 0.0),
                    velocity: None,
                    serving_cell_id: 16,
                    timestamp_ms: s * 1000,
                });
                let dbm = if rising { -80 + s as i32 } else { -(x as i32) };
                e.on_measurement(1, dbm, SimCoord::new(x as i32, 0, 0), s * 1000);
            }
            let fired = !e.tick(7000, true, lsq(&nwdaf, 7000)).is_empty();
            assert_eq!(fired, expect, "rising {rising}");
        }
    }

    /// The predictor uses the UE's A3 margin (3 dB offset + 1 dB hysteresis): a
    /// predicted end point where east is 2 dB better (x 61) is not enough,
    /// 6 dB (x 63) is. A revert to "any neighbour better" fires at x 61.
    #[test]
    fn the_predictor_uses_the_a3_margin() {
        for (end_x, expect) in [(61.0, false), (63.0, true)] {
            let mut e = Engine::from_config(&west(0.0, 1)).unwrap();
            e.set_served(&[1], 0);
            for (s, x) in [(0u64, 40), (1, 45), (2, 50)] {
                e.on_measurement(1, -x, SimCoord::new(x, 0, 0), s * 1000);
            }
            let fired = !e
                .tick(5000, true, |_| {
                    Some((
                        TrajectoryPrediction {
                            ue_id: 1,
                            waypoints: (1..=10u32)
                                .map(|i| {
                                    let x = end_x - f64::from(10 - i);
                                    (Vector3::new(x, 0.0, 0.0), u64::from(i) * 200)
                                })
                                .collect(),
                            confidence: 0.95,
                            horizon_ms: HORIZON_MS,
                        },
                        Predictor::Lsq,
                    ))
                })
                .is_empty();
            assert_eq!(fired, expect, "end x {end_x}");
        }
    }

    /// The ping-pong guard: a UE served here for less than MIN_DWELL_MS is not
    /// predicted away, however confident the prediction.
    #[test]
    fn a_newly_served_ue_dwells_before_the_predictor_moves_it() {
        let mut nwdaf = NwdafManager::new(100);
        let mut e = Engine::from_config(&west(0.0, 1)).unwrap();
        straight(&mut e, &mut nwdaf, 1, 0, 8);
        // Served (handed in) at 4 s: at 7 s it has dwelt 3 s.
        e.set_served(&[1], 4000);
        assert!(e.tick(7000, true, lsq(&nwdaf, 7000)).is_empty());
        straight(&mut e, &mut nwdaf, 1, 8, 10);
        let ev = e.tick(9000, true, lsq(&nwdaf, 9000));
        assert!(
            matches!(ev.as_slice(), [Event::Prediction { .. }]),
            "{ev:?}"
        );
    }

    /// A UE still served after an attempt is retried after RETRY_AFTER_MS, at
    /// most MAX_ATTEMPTS times; then one `Exhausted`, and nothing more until it
    /// is served afresh.
    #[test]
    fn attempts_are_capped_and_exhaust_once() {
        let mut nwdaf = NwdafManager::new(100);
        let mut e = Engine::from_config(&west(0.0, 1)).unwrap();
        e.set_served(&[1], 0);
        straight(&mut e, &mut nwdaf, 1, 0, 8);
        let mut attempts = Vec::new();
        let mut exhausted = 0;
        for s in 7..30u64 {
            // The UE stays where it was last measured moving: the fit still
            // predicts east, as for a UE whose handover keeps failing.
            for ev in e.tick(s * 1000, true, lsq(&nwdaf, 7000)) {
                match ev {
                    Event::Prediction { attempt, .. } => attempts.push((s, attempt)),
                    Event::Exhausted { target_nci, .. } => {
                        assert_eq!(target_nci, EAST);
                        exhausted += 1;
                    }
                    other => panic!("{other:?}"),
                }
            }
        }
        assert_eq!(attempts, vec![(7, 1), (10, 2), (13, 3)]);
        assert_eq!(exhausted, 1);
        assert_eq!(e.stats().exhausted, 1);
        // Served afresh (e.g. handed back in): a new episode, attempts again.
        e.set_served(&[], 30_000);
        e.set_served(&[1], 31_000);
        straight(&mut e, &mut nwdaf, 1, 30, 37);
        assert!(e
            .tick(36_000, true, lsq(&nwdaf, 36_000))
            .iter()
            .any(|ev| matches!(ev, Event::Prediction { attempt: 1, .. })));
    }

    /// The predictive time is kept for the A3 that follows the handover, and the
    /// lead time is emitted with its sign whichever comes second.
    #[test]
    fn the_lead_time_is_emitted_with_its_sign() {
        let mut e = Engine::from_config(&west(0.0, 1)).unwrap();
        e.set_served(&[1], 0);
        let pred = |x: f64| {
            move |_id: i32| {
                Some((
                    TrajectoryPrediction {
                        ue_id: 1,
                        waypoints: (1..=10u32)
                            .map(|i| (Vector3::new(x + f64::from(i), 0.0, 0.0), u64::from(i) * 200))
                            .collect(),
                        confidence: 0.95,
                        horizon_ms: HORIZON_MS,
                    },
                    Predictor::Lsq,
                ))
            }
        };
        for (s, x) in [(0u64, 40), (1, 45), (2, 50)] {
            e.on_measurement(1, -x, SimCoord::new(x, 0, 0), s * 1000);
        }
        e.tick(5000, true, pred(70.0));
        e.set_served(&[], 5500); // handed away; heartbeats go on
        e.on_measurement(1, -64, SimCoord::new(64, 0, 0), 7000);
        let ev = e.on_measurement(1, -65, SimCoord::new(65, 0, 0), 8000);
        assert!(
            ev.contains(&Event::LeadTime {
                ue_id: 1,
                target_nci: EAST,
                predictive_ms: 5000,
                a3_ms: 8000,
                lead_ms: 3000,
            }),
            "{ev:?}"
        );
        // A3 first, prediction later: a negative lead, on the prediction.
        let mut e = Engine::from_config(&west(0.0, 1)).unwrap();
        e.set_served(&[1], 0);
        for (s, x) in [(0u64, 60), (1, 63), (2, 64)] {
            e.on_measurement(1, -x, SimCoord::new(x, 0, 0), s * 1000);
        }
        let ev = e.tick(6000, true, pred(70.0));
        assert!(
            ev.contains(&Event::LeadTime {
                ue_id: 1,
                target_nci: EAST,
                predictive_ms: 6000,
                a3_ms: 2000,
                lead_ms: -4000,
            }),
            "{ev:?}"
        );
    }

    /// A config with no positioned neighbour cannot run the demo.
    #[test]
    fn the_engine_needs_a_positioned_neighbour() {
        let mut c = west(0.0, 1);
        c.handover_neighbours[0].position = None;
        assert!(Engine::from_config(&c).is_err());
    }

    /// The seeded replay of the committed route on 24 shadowing seeds (σ 4 dB,
    /// the committed configs' value): the claims that hold across seeds.
    ///
    /// - With σ 0 the predictor leads A3 (the geometry alone).
    /// - On every seed the predictor triggers once and A3 follows; the A3 time
    ///   moves with the seed (shadowing is what the arms differ on).
    /// - The lead is positive on most seeds (at least 20 of 24), and the
    ///   median is positive. It is not asserted positive on every seed.
    #[test]
    fn the_seeded_replay_leads_a3_on_most_seeds() {
        let base = replay(&west(0.0, 1), true);
        assert!(base.lead_ms.unwrap() > 0, "{base:?}");
        let mut leads = Vec::new();
        let mut a3_times = BTreeSet::new();
        for seed in 1..=24u64 {
            let run = replay(&west(4.0, seed), true);
            assert_eq!(run.predictions, 1, "seed {seed}: {run:?}");
            let a3 = replay(&west(4.0, seed), false);
            assert!(a3.predictive_ms.is_none(), "mode a3 predicts nothing");
            a3_times.insert(a3.a3_ms.expect("A3 triggers on the route"));
            leads.push(
                run.lead_ms
                    .unwrap_or_else(|| panic!("seed {seed}: {run:?}")),
            );
        }
        leads.sort_unstable();
        let positive = leads.iter().filter(|l| **l > 0).count();
        eprintln!("D441D replay leads (ms, sorted, 24 seeds): {leads:?}");
        assert!(a3_times.len() > 1, "the A3 time must move with the seed");
        assert!(positive >= 20, "{positive}/24 positive: {leads:?}");
        assert!(leads[12] > 0, "median {leads:?}");
    }
}
