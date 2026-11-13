//! Network Data Analytics Function (NWDAF) Task for gNB
//!
//! This task implements NWDAF analytics at the gNB level, providing:
//! - Layer 1: Real-time anomaly detection (z-score based)
//! - Layer 2: Predictive analytics (trajectory, load prediction)
//! - Layer 3: Prescriptive optimization (handover recommendations)
//! - Layer 4: Autonomous closed-loop control
//!
//! # Architecture
//!
//! The NWDAF task receives measurements from:
//! - UEs (RSRP, RSRQ, position, velocity)
//! - gNB cells (PRB usage, connected UE count)
//! - RRC layer (handover events, cell selection)
//!
//! It provides analytics to:
//! - RRC (handover recommendations)
//! - NGAP (load balancing decisions)
//! - External consumers via service APIs

use std::collections::HashMap;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tracing::{debug, error, info, warn};

use nextgsim_nwdaf::{CellLoad, NwdafManager, NwdafResponse, UeMeasurement, Vector3};

use crate::tasks::{GnbTaskBase, NwdafMessage, RrcMessage, Task, TaskMessage};

/// Nominal reference ceiling used to express PRB usage in Mbps-shaped units.
///
/// This is a placeholder, not a capacity model. The gNB config carries no cell
/// bandwidth or numerology, so a real capacity cannot be derived here; a proper
/// figure needs either a per-cell throughput counter on the data path or a
/// bandwidth field to compute from.
const NOMINAL_CELL_CAPACITY_MBPS: f32 = 1000.0;

/// NWDAF Task for gNB
///
/// Provides four-layer network data analytics with closed-loop automation.
pub struct NwdafTask {
    /// Task base for inter-task communication
    task_base: GnbTaskBase,
    /// NWDAF analytics manager
    nwdaf: NwdafManager,
    /// Cell load tracking
    cell_loads: HashMap<i32, CellLoad>,
    /// Measurement history length
    _max_history_length: usize,
    /// The process's demo state (nextgsim#458), read for the d440b mode at
    /// startup and before every served prediction.
    #[cfg(feature = "model-registry")]
    demo: Option<&'static nextgsim_demo::DemoRuntime>,
    /// The d440b mode the last served prediction ran under, to log a switch once.
    #[cfg(feature = "model-registry")]
    served_mode: Option<String>,
    /// Whether the first ONNX / fallback prediction has been logged.
    #[cfg(feature = "model-registry")]
    first_served: (bool, bool),
    /// Demo D441D's decision core, built when the demo is active at startup
    /// (nextgsim#467); `None` otherwise, and the task behaves as without it.
    #[cfg(feature = "demo-d441d-predictive-ho")]
    d441d: Option<D441dState>,
}

/// D441D's per-task state: the engine and the mode its last tick ran under.
#[cfg(feature = "demo-d441d-predictive-ho")]
struct D441dState {
    engine: crate::nwdaf::d441d::Engine,
    mode: Option<String>,
}

/// Milliseconds since the Unix epoch, the clock measurements are stamped with.
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

impl NwdafTask {
    /// Creates a new NWDAF task with default history length (100)
    pub fn new(task_base: GnbTaskBase) -> Self {
        Self::with_history_length(task_base, 100)
    }

    /// Creates a new NWDAF task with specified history length
    pub fn with_history_length(task_base: GnbTaskBase, max_history_length: usize) -> Self {
        #[cfg(feature = "model-registry")]
        return Self::with_demo(task_base, max_history_length, nextgsim_demo::installed());
        #[cfg(not(feature = "model-registry"))]
        Self::build(task_base, max_history_length)
    }

    /// The task under an explicit demo state (nextgsim#463): what
    /// [`Self::with_history_length`] does with the installed one.
    ///
    /// With `model_registry` built and `nwdaf_model_path` set, the model is
    /// activated only after its manifest's checks pass (`NWDAF: model verified`
    /// or `NWDAF: model refused (<reason>)`). Under the d440b demo in mode
    /// `fallback` it is neither verified nor activated, and every prediction uses
    /// the fallback; the demo (either mode) also serves one trajectory
    /// prediction per recorded UE measurement, which is what moves the
    /// `nwdaf-status` counters in a live run. With the demo off nothing is served
    /// and the task behaves as without it.
    #[cfg(feature = "model-registry")]
    pub fn with_demo(
        task_base: GnbTaskBase,
        max_history_length: usize,
        demo: Option<&'static nextgsim_demo::DemoRuntime>,
    ) -> Self {
        use crate::demo::D440B_KEY;
        use nextgsim_demo::demo_log;

        let mut nwdaf = NwdafManager::new(max_history_length);
        let mode = demo.and_then(|d| d.mode(D440B_KEY));
        match (task_base.config.nwdaf_model_path.as_ref(), mode.as_deref()) {
            (Some(path), Some("fallback")) => {
                nwdaf.defer_trajectory_model(path);
                nwdaf.set_force_fallback(true);
                demo_log!(
                    warn,
                    D440B_KEY,
                    "mode=fallback: the model at {} is neither verified nor activated; every \
                     trajectory prediction uses linear extrapolation",
                    path.display()
                );
            }
            (Some(path), _) => {
                // The outcome is logged by the manager (`NWDAF: model verified`
                // / `NWDAF: model refused (<reason>)`) and kept in its state; a
                // refusal degrades to the fallback rather than failing startup,
                // as an unloadable model always has.
                let _ = nwdaf.load_verified_trajectory_model(path);
            }
            (None, Some(m)) => {
                nwdaf.set_force_fallback(m == "fallback");
                demo_log!(
                    warn,
                    D440B_KEY,
                    "mode={m}: no nwdaf_model_path is configured, so there is no model to \
                     verify; every trajectory prediction uses linear extrapolation"
                );
            }
            (None, None) => {}
        }
        #[cfg(feature = "demo-d441d-predictive-ho")]
        let d441d = Self::d441d_state(&task_base.config, demo);
        Self {
            task_base,
            nwdaf,
            cell_loads: HashMap::new(),
            _max_history_length: max_history_length,
            demo,
            served_mode: mode,
            first_served: (false, false),
            #[cfg(feature = "demo-d441d-predictive-ho")]
            d441d,
        }
    }

    #[cfg(not(feature = "model-registry"))]
    fn build(task_base: GnbTaskBase, max_history_length: usize) -> Self {
        let mut nwdaf = NwdafManager::new(max_history_length);

        // Load the configured ONNX trajectory model, if any (issue #18). Without
        // this the working ONNX inference path in nextgsim-nwdaf was unreachable
        // from the gNB: nothing ever called the (already public)
        // NwdafManager::load_trajectory_model, so the operational path was always
        // linear extrapolation no matter what was installed.
        if let Some(ref path) = task_base.config.nwdaf_model_path {
            match nwdaf.load_trajectory_model(path) {
                Ok(()) => info!(
                    "NWDAF: loaded ONNX trajectory model from {}; predictions will use the model",
                    path.display()
                ),
                // Degrade, do not fail. The predictor's own fallback is
                // documented behaviour, and a gNB that will not start because an
                // optional analytics model is missing is worse than one that
                // starts and says so.
                Err(e) => warn!(
                    "NWDAF: could not load ONNX trajectory model from {} ({e}); \
                     continuing with linear extrapolation",
                    path.display()
                ),
            }
        }

        Self {
            task_base,
            nwdaf,
            cell_loads: HashMap::new(),
            _max_history_length: max_history_length,
        }
    }

    /// Handles a UE measurement report.
    ///
    /// A report without RSRP is **not** recorded as a measurement: RSRP is what
    /// every analytics reader needs, and there is no safe filler for it. 0.0
    /// reads as an implausibly strong signal, and any "typical" constant makes
    /// the series artificially stable so the z-score anomaly detector can never
    /// fire. Such reports (the ISAC task's position-only ones) are logged and
    /// dropped.
    ///
    /// RSRQ is optional (#453). No live producer has one: TS 38.215 §5.1.3
    /// defines it as N×RSRP/RSSI and the RLS channel model has no RSSI, so the
    /// RLS and RRC reports carry `None`. The analytics layer defines its own
    /// behaviour without it, so the report is recorded with `rsrq: None` rather
    /// than dropped or filled in.
    fn handle_ue_measurement(
        &mut self,
        ue_id: i32,
        rsrp: Option<f32>,
        rsrq: Option<f32>,
        position: (f32, f32, f32),
    ) {
        self.handle_ue_measurement_at(ue_id, rsrp, rsrq, position, now_ms());
    }

    /// [`Self::handle_ue_measurement`], stamped `timestamp_ms`.
    fn handle_ue_measurement_at(
        &mut self,
        ue_id: i32,
        rsrp: Option<f32>,
        rsrq: Option<f32>,
        position: (f32, f32, f32),
        timestamp_ms: u64,
    ) {
        let Some(rsrp) = rsrp else {
            debug!(
                "NWDAF: UE {ue_id} report carries no RSRP (pos=({}, {}, {})); \
                 not recording a radio measurement",
                position.0, position.1, position.2
            );
            return;
        };

        debug!(
            "NWDAF: UE {} measurement - RSRP={} dBm, RSRQ={:?} dB, pos=({}, {}, {})",
            ue_id, rsrp, rsrq, position.0, position.1, position.2
        );

        let measurement = UeMeasurement {
            ue_id,
            rsrp,
            rsrq,
            sinr: None,
            position: Vector3::new(position.0 as f64, position.1 as f64, position.2 as f64),
            velocity: None,
            serving_cell_id: self.task_base.config.cell_id() as i32,
            timestamp_ms,
        };

        self.nwdaf.record_measurement(measurement);

        #[cfg(feature = "demo-d441d-predictive-ho")]
        self.d441d_measurement(ue_id, rsrp, position, timestamp_ms);

        // Check for anomalies
        let anomalies = self.nwdaf.recent_anomalies();
        if !anomalies.is_empty() {
            warn!(
                "NWDAF: Detected {} anomalies for UE {}",
                anomalies.len(),
                ue_id
            );
        }

        #[cfg(feature = "model-registry")]
        self.serve_d440b(ue_id);
    }

    /// Under the d440b demo, serve one trajectory prediction for `ue_id` with
    /// the mode read now, so a runtime switch applies from this prediction
    /// (nextgsim#463). Nothing happens with the demo off.
    #[cfg(feature = "model-registry")]
    fn serve_d440b(&mut self, ue_id: i32) {
        use crate::demo::D440B_KEY;
        use nextgsim_demo::demo_log;
        use nextgsim_nwdaf::serving::ModelState;

        let Some(mode) = self.demo.and_then(|d| d.mode(D440B_KEY)) else {
            return;
        };
        if self.served_mode.as_deref() != Some(mode.as_str()) {
            demo_log!(
                warn,
                D440B_KEY,
                "mode is now {mode} (was {}); applies from this prediction",
                self.served_mode.as_deref().unwrap_or("-")
            );
        }
        match mode.as_str() {
            "fallback" => self.nwdaf.set_force_fallback(true),
            _ => {
                self.nwdaf.set_force_fallback(false);
                // A model deferred by a `fallback` start is verified on the
                // first `verify` prediction, never activated unchecked.
                if let ModelState::Deferred { .. } = self.nwdaf.model_state() {
                    if let Some(path) = self.task_base.config.nwdaf_model_path.clone() {
                        let _ = self.nwdaf.load_verified_trajectory_model(&path);
                    }
                }
            }
        }
        self.served_mode = Some(mode.clone());

        let before = self.nwdaf.serving_stats().onnx_inferences();
        let Some(prediction) = self.nwdaf.predict_trajectory(ue_id, 1000) else {
            return; // too little history yet: no prediction, nothing counted
        };
        let by_model = self.nwdaf.serving_stats().onnx_inferences() > before;
        let method = if by_model { "onnx" } else { "fallback" };
        let first = if by_model {
            &mut self.first_served.0
        } else {
            &mut self.first_served.1
        };
        if !*first {
            *first = true;
            demo_log!(
                info,
                D440B_KEY,
                "first trajectory prediction served by {method}: ue={ue_id} mode={mode} \
                 waypoints={}",
                prediction.waypoints.len()
            );
        } else {
            demo_log!(
                debug,
                D440B_KEY,
                "trajectory prediction served by {method}: ue={ue_id} mode={mode}"
            );
        }
    }

    /// Report one observation received from the 6G topic bus (issue #16).
    ///
    /// Logging only, deliberately: the analytics state is fed by the
    /// `NwdafMessage` channel, and having the bus feed it as well would double
    /// every sample. What this demonstrates is that the fan-out works — a second
    /// consumer receives the ISAC task's observation without the ISAC task
    /// knowing about it.
    ///
    /// A `Lagged` is reported at `warn` rather than swallowed: it means this
    /// consumer fell behind and lost samples, which is the bus's documented
    /// trade-off and worth seeing rather than guessing at.
    #[cfg(feature = "event-bus")]
    fn log_bus_event(
        event: Result<nextgsim_common::bus::BusEvent, tokio::sync::broadcast::error::RecvError>,
    ) {
        use tokio::sync::broadcast::error::RecvError;
        match event {
            Ok(event) => debug!(
                "NWDAF: bus event on {} from {}: {} = {:?}",
                event.topic.name(),
                event.source,
                event.measurement_type,
                event.measurements
            ),
            Err(RecvError::Lagged(skipped)) => warn!(
                "NWDAF: lagged on the 6G topic bus, {} event(s) lost",
                skipped
            ),
            Err(RecvError::Closed) => debug!("NWDAF: the 6G topic bus closed"),
        }
    }

    /// Handles a cell load report.
    ///
    /// `prb_usage` is what a producer *measured*, and there is no such producer
    /// today because this simulator has no PRB scheduler. `throughput_mbps` is
    /// measured, by the RLS task counting the user-plane octets it moves
    /// (issue #18), so the occupancy figure the analytics layer sees is now
    /// derived from real traffic:
    ///
    /// ```text
    /// occupancy = measured throughput / NOMINAL_CELL_CAPACITY_MBPS
    /// ```
    ///
    /// Only the *ceiling* in that ratio is nominal; the numerator is observed, so
    /// the series varies with load. This inverts what the code used to do —
    /// scale a supplied `prb_usage` by the same nominal ceiling to invent a
    /// throughput — which meant any analytics reading throughput was reading PRB
    /// usage twice. A real `prb_usage` still wins when a producer can supply one.
    fn handle_cell_load(
        &mut self,
        cell_id: i32,
        prb_usage: Option<f32>,
        connected_ues: u32,
        throughput_mbps: Option<f32>,
    ) {
        // A measured throughput is the only honest input to occupancy, so a
        // report carrying neither it nor a real PRB figure says nothing about
        // load. Dropping it beats recording a fabricated constant: a flat series
        // is invisible to the z-score anomaly detector and is extrapolated as
        // fact by load prediction.
        let occupancy = match (prb_usage, throughput_mbps) {
            (Some(measured_prb), _) => measured_prb.clamp(0.0, 1.0),
            (None, Some(mbps)) => (mbps / NOMINAL_CELL_CAPACITY_MBPS).clamp(0.0, 1.0),
            (None, None) => {
                debug!(
                    "NWDAF: Cell {} load report carries neither PRB usage nor throughput \
                     (connected UEs={}); not recorded, because a fabricated load poisons the \
                     anomaly detector and the load predictor",
                    cell_id, connected_ues
                );
                return;
            }
        };

        debug!(
            "NWDAF: Cell {} load - occupancy={:.1}% ({}), throughput={:.3} Mbps, connected UEs={}",
            cell_id,
            occupancy * 100.0,
            if prb_usage.is_some() {
                "measured PRB"
            } else {
                "traffic-derived, nominal ceiling"
            },
            throughput_mbps.unwrap_or(0.0),
            connected_ues
        );

        let load = CellLoad {
            cell_id,
            prb_usage: occupancy,
            connected_ues,
            // Measured when the producer counted traffic. Falls back to the old
            // scaling only for a producer that supplies PRB usage and no
            // throughput, where a restatement is the best available.
            avg_throughput_mbps: throughput_mbps.unwrap_or(occupancy * NOMINAL_CELL_CAPACITY_MBPS),
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        };

        self.nwdaf.record_cell_load(load.clone());
        self.cell_loads.insert(cell_id, load);
    }

    /// Handles a trajectory prediction request
    fn handle_predict_trajectory(
        &mut self,
        ue_id: i32,
        horizon_ms: u32,
        response_tx: Option<oneshot::Sender<NwdafResponse>>,
    ) {
        debug!(
            "NWDAF: Trajectory prediction request for UE {} with horizon {} ms",
            ue_id, horizon_ms
        );

        let prediction = self.nwdaf.predict_trajectory(ue_id, horizon_ms);

        let response = if let Some(pred) = prediction {
            info!(
                "NWDAF: Predicted trajectory for UE {} with {} waypoints (confidence={:.2})",
                ue_id,
                pred.waypoints.len(),
                pred.confidence
            );
            NwdafResponse::TrajectoryPrediction(pred)
        } else {
            warn!("NWDAF: No trajectory prediction available for UE {}", ue_id);
            NwdafResponse::Error(format!("Insufficient data for UE {ue_id}"))
        };

        if let Some(tx) = response_tx {
            if tx.send(response).is_err() {
                error!("NWDAF: Failed to send trajectory prediction response");
            }
        }
    }

    /// Handles a handover recommendation request
    fn handle_handover_recommendation(&mut self, ue_id: i32, target_cell: i32, confidence: f32) {
        debug!(
            "NWDAF: Handover recommendation for UE {} to cell {} (confidence={:.2})",
            ue_id, target_cell, confidence
        );

        info!(
            "NWDAF: Recommended handover for UE {} to cell {} with confidence {:.2}",
            ue_id, target_cell, confidence
        );

        // Send recommendation to RRC task for execution
        let msg = RrcMessage::NwdafHandoverRecommendation {
            ue_id,
            target_cell,
            confidence,
        };
        if let Err(e) = self.task_base.rrc_tx.try_send(msg) {
            warn!(
                "NWDAF: Failed to send handover recommendation to RRC: {}",
                e
            );
        }
    }
}

impl NwdafTask {
    /// The next message, or `None` when `tick` (if any) fires first.
    async fn recv_or_tick(
        rx: &mut mpsc::Receiver<TaskMessage<NwdafMessage>>,
        tick: &mut Option<tokio::time::Interval>,
    ) -> Option<Option<TaskMessage<NwdafMessage>>> {
        match tick {
            Some(t) => tokio::select! {
                m = rx.recv() => Some(m),
                _ = t.tick() => None,
            },
            None => Some(rx.recv().await),
        }
    }

    /// The `nwdaf-status` JSON: the manager's status, plus a `handover`
    /// section while D441D is active (nextgsim#467).
    #[cfg(feature = "model-registry")]
    fn status_json(&self) -> String {
        #[cfg(feature = "demo-d441d-predictive-ho")]
        if let Some(st) = &self.d441d {
            let mut v = serde_json::to_value(self.nwdaf.status())
                .unwrap_or_else(|e| serde_json::json!({ "error": e.to_string() }));
            if let Some(obj) = v.as_object_mut() {
                let stats = st.engine.stats();
                obj.insert(
                    "handover".into(),
                    serde_json::json!({
                        "demo": crate::demo::D441D_KEY,
                        "mode": self.demo.and_then(|d| d.mode(crate::demo::D441D_KEY)),
                        "served_ues": st.engine.served_count(),
                        "predictions": stats.predictions,
                        "exhausted": stats.exhausted,
                        "a3_would_trigger": stats.a3_would_trigger,
                        "a3_fallbacks": stats.a3_fallbacks,
                        "stranded": stats.stranded,
                        "lead_ms": stats.lead_ms,
                    }),
                );
            }
            return v.to_string();
        }
        self.nwdaf.status().to_json()
    }

    /// D441D's state for this gNB, when the demo is active at startup. A config
    /// the engine cannot run was refused at startup (`demo::d441d_startup`); if
    /// one gets here anyway the demo stays inert and says so.
    #[cfg(feature = "demo-d441d-predictive-ho")]
    fn d441d_state(
        config: &nextgsim_common::config::GnbConfig,
        demo: Option<&'static nextgsim_demo::DemoRuntime>,
    ) -> Option<D441dState> {
        use crate::demo::D441D_KEY;
        let mode = demo.and_then(|d| d.mode(D441D_KEY))?;
        match crate::nwdaf::d441d::Engine::from_config(config) {
            Ok(engine) => Some(D441dState {
                engine,
                mode: Some(mode),
            }),
            Err(e) => {
                nextgsim_demo::demo_log!(error, D441D_KEY, "inert: {e}");
                None
            }
        }
    }

    /// Feed one measurement to D441D's shadow A3 evaluator.
    #[cfg(feature = "demo-d441d-predictive-ho")]
    fn d441d_measurement(
        &mut self,
        ue_id: i32,
        rsrp: f32,
        position: (f32, f32, f32),
        timestamp_ms: u64,
    ) {
        let Some(st) = self.d441d.as_mut() else {
            return;
        };
        // The RLS producer sends integer levels and positions as f32.
        let pos = nextgsim_rls::SimCoord::new(
            position.0.round() as i32,
            position.1.round() as i32,
            position.2.round() as i32,
        );
        let events = st
            .engine
            .on_measurement(ue_id, rsrp.round() as i32, pos, timestamp_ms);
        for e in events {
            self.d441d_act(e);
        }
    }

    /// RRC's served set, for D441D.
    #[cfg(feature = "demo-d441d-predictive-ho")]
    fn d441d_served(&mut self, ue_ids: &[i32], now: u64) {
        if let Some(st) = self.d441d.as_mut() {
            st.engine.set_served(ue_ids, now);
        }
    }

    /// One D441D predictor tick at `now`, under the mode read now, so a
    /// runtime switch applies from this tick. Returns what it acted on.
    #[cfg(feature = "demo-d441d-predictive-ho")]
    fn d441d_tick(&mut self, now: u64) -> Vec<crate::nwdaf::d441d::Event> {
        use crate::demo::{D440B_KEY, D441D_KEY};
        use crate::nwdaf::d441d::{lsq_prediction, Predictor, HORIZON_MS};
        use nextgsim_demo::demo_log;

        let Some(mut st) = self.d441d.take() else {
            return Vec::new();
        };
        let mode = self.demo.and_then(|d| d.mode(D441D_KEY));
        if mode != st.mode {
            demo_log!(
                warn,
                D441D_KEY,
                "mode is now {} (was {}); applies from this tick",
                mode.as_deref().unwrap_or("-"),
                st.mode.as_deref().unwrap_or("-")
            );
            st.mode = mode.clone();
        }
        let predictive = matches!(mode.as_deref(), Some("lsq" | "onnx"));
        let onnx = mode.as_deref() == Some("onnx");
        // Only a model the d440b checks verified may predict (#463).
        let verified = self.demo.and_then(|d| d.mode(D440B_KEY)).as_deref() == Some("verify");
        let nwdaf = &self.nwdaf;
        let events = st.engine.tick(now, predictive, |ue| {
            if onnx && verified {
                let before = nwdaf.serving_stats().onnx_inferences();
                let p = nwdaf.predict_trajectory(ue, HORIZON_MS)?;
                let by_model = nwdaf.serving_stats().onnx_inferences() > before;
                Some((
                    p,
                    if by_model {
                        Predictor::Onnx
                    } else {
                        Predictor::Fallback
                    },
                ))
            } else {
                let p = lsq_prediction(nwdaf, ue, HORIZON_MS, now)?;
                Some((
                    p,
                    if onnx {
                        Predictor::Fallback
                    } else {
                        Predictor::Lsq
                    },
                ))
            }
        });
        self.d441d = Some(st);
        for e in &events {
            self.d441d_act(e.clone());
        }
        events
    }

    /// Log one D441D event and act on it: a prediction goes to RRC as an NWDAF
    /// handover recommendation, the path `handle_nwdaf_handover` takes to
    /// `execute_handover`.
    #[cfg(feature = "demo-d441d-predictive-ho")]
    fn d441d_act(&mut self, event: crate::nwdaf::d441d::Event) {
        use crate::demo::D441D_KEY;
        use crate::nwdaf::d441d::{Event, MAX_ATTEMPTS};
        use nextgsim_demo::demo_log;

        let cell = |nci: u64| (nci & 0xF_FFFF_FFFF) as i32;
        match event {
            Event::Prediction {
                ue_id,
                target_nci,
                predictor,
                confidence,
                attempt,
            } => {
                demo_log!(
                    info,
                    D441D_KEY,
                    "NWDAF: handover prediction ue={ue_id} predictor={} target={} conf={confidence:.2} \
                     attempt={attempt}/{MAX_ATTEMPTS}",
                    predictor.as_str(),
                    cell(target_nci)
                );
                let msg = RrcMessage::NwdafHandoverRecommendation {
                    ue_id,
                    target_cell: cell(target_nci),
                    confidence,
                };
                if let Err(e) = self.task_base.rrc_tx.try_send(msg) {
                    warn!("NWDAF: Failed to send handover recommendation to RRC: {e}");
                }
            }
            Event::Exhausted { ue_id, target_nci } => demo_log!(
                warn,
                D441D_KEY,
                "NWDAF: handover prediction exhausted ue={ue_id} target={}: still served after \
                 {MAX_ATTEMPTS} attempt(s); none until it is served afresh",
                cell(target_nci)
            ),
            Event::A3WouldTrigger {
                ue_id,
                target_nci,
                t_lo_ms,
                t_hi_ms,
                serving_dbm,
                neighbour_dbm,
            } => demo_log!(
                info,
                D441D_KEY,
                "A3 would trigger t={t_lo_ms}..{t_hi_ms} ue={ue_id} target={} \
                 serving={serving_dbm} dBm neighbour={neighbour_dbm} dBm (the UE's A3 on its \
                 2.5 s cycle, at an unknown phase)",
                cell(target_nci)
            ),
            Event::LeadTime {
                ue_id,
                target_nci,
                predictive_ms,
                a3_ms,
                lead_ms,
            } => demo_log!(
                info,
                D441D_KEY,
                "lead time ue={ue_id} target={} predictive_t={predictive_ms} \
                 a3_t={}..{} lead_ms={}..{}",
                cell(target_nci),
                a3_ms.0,
                a3_ms.1,
                lead_ms.0,
                lead_ms.1
            ),
            Event::Stranded {
                ue_id,
                target_nci,
                a3_ms,
            } => demo_log!(
                warn,
                D441D_KEY,
                "stranded ue={ue_id} target={}: A3 fired at t={}..{} and neither the predictor \
                 nor the A3 fallback moved the UE",
                cell(target_nci),
                a3_ms.0,
                a3_ms.1
            ),
        }
    }
}

#[async_trait::async_trait]
impl Task for NwdafTask {
    type Message = NwdafMessage;

    async fn run(&mut self, mut rx: mpsc::Receiver<TaskMessage<Self::Message>>) {
        info!("NWDAF task started");

        // Subscribe to the topic bus (issue #16), which is the reference consumer
        // for the reference producer in the ISAC task. This is a SECOND source of
        // the same observations, not a replacement for the NwdafMessage channel:
        // the direct channel is what the ISAC task's delivery depends on, and this
        // is how a further consumer would be added without editing that task.
        #[cfg(feature = "event-bus")]
        let mut bus_rx = self
            .task_base
            .sixg
            .as_ref()
            .and_then(|sixg| sixg.bus.as_ref())
            .map(|bus| bus.subscribe());
        #[cfg(feature = "event-bus")]
        if bus_rx.is_some() {
            info!("NWDAF task subscribed to the 6G topic bus");
        }

        // D441D's predictor tick (nextgsim#467): only with the demo active, so
        // every other run waits on its channel exactly as before.
        #[cfg(feature = "demo-d441d-predictive-ho")]
        let mut tick = self.d441d.as_ref().map(|_| {
            let mut t = tokio::time::interval(std::time::Duration::from_millis(
                crate::nwdaf::d441d::TICK_MS,
            ));
            t.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            t
        });
        #[cfg(not(feature = "demo-d441d-predictive-ho"))]
        let mut tick: Option<tokio::time::Interval> = None;

        loop {
            // The bus arm is selected over only when a subscription exists, so a
            // build with the feature on but no 6G task init behaves as before.
            #[cfg(feature = "event-bus")]
            let msg = {
                if let Some(ref mut bus_rx) = bus_rx {
                    tokio::select! {
                        received = Self::recv_or_tick(&mut rx, &mut tick) => received,
                        bus_event = bus_rx.recv() => {
                            Self::log_bus_event(bus_event);
                            continue;
                        }
                    }
                } else {
                    Self::recv_or_tick(&mut rx, &mut tick).await
                }
            };
            #[cfg(not(feature = "event-bus"))]
            let msg = Self::recv_or_tick(&mut rx, &mut tick).await;
            let Some(msg) = msg else {
                #[cfg(feature = "demo-d441d-predictive-ho")]
                let _ = self.d441d_tick(now_ms());
                continue;
            };

            match msg {
                Some(TaskMessage::Message(msg)) => match msg {
                    NwdafMessage::UeMeasurement {
                        ue_id,
                        rsrp,
                        rsrq,
                        position,
                    } => {
                        self.handle_ue_measurement(ue_id, rsrp, rsrq, position);
                    }
                    NwdafMessage::CellLoad {
                        cell_id,
                        prb_usage,
                        connected_ues,
                        throughput_mbps,
                    } => {
                        self.handle_cell_load(cell_id, prb_usage, connected_ues, throughput_mbps);
                    }
                    NwdafMessage::PredictTrajectory { ue_id, horizon_ms } => {
                        self.handle_predict_trajectory(ue_id, horizon_ms, None);
                    }
                    NwdafMessage::HandoverRecommendation {
                        ue_id,
                        target_cell,
                        confidence,
                    } => {
                        self.handle_handover_recommendation(ue_id, target_cell, confidence);
                    }
                    #[cfg(feature = "model-registry")]
                    NwdafMessage::Status { reply } => {
                        // A dropped receiver is a CLI that gave up waiting.
                        let _ = reply.send(self.status_json());
                    }
                    #[cfg(feature = "demo-d441d-predictive-ho")]
                    NwdafMessage::ServedUes { ue_ids } => self.d441d_served(&ue_ids, now_ms()),
                    #[cfg(feature = "demo-d441d-predictive-ho")]
                    NwdafMessage::A3Fallback { ue_id } => {
                        if let Some(st) = self.d441d.as_mut() {
                            st.engine.on_a3_fallback(ue_id);
                        }
                    }
                },
                Some(TaskMessage::Shutdown) => {
                    info!("NWDAF task received shutdown signal");
                    break;
                }
                None => {
                    info!("NWDAF task channel closed");
                    break;
                }
            }
        }

        info!(
            "NWDAF task stopped, tracked {} cells",
            self.cell_loads.len()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::{GnbTaskBase, DEFAULT_CHANNEL_CAPACITY};
    use nextgsim_common::config::GnbConfig;
    use nextgsim_common::Plmn;

    fn test_config() -> GnbConfig {
        GnbConfig {
            nci: 0x000000010,
            gnb_id_length: 32,
            plmn: Plmn::new(001, 01, false),
            tac: 1,
            nssai: vec![],
            amf_configs: vec![],
            link_ip: "127.0.0.1".parse().unwrap(),
            ngap_ip: "127.0.0.1".parse().unwrap(),
            gtp_ip: "127.0.0.1".parse().unwrap(),
            gtp_advertise_ip: None,
            ignore_stream_ids: false,
            upf_addr: None,
            upf_port: 2152,
            pqc_config: nextgsim_common::config::PqcConfig::default(),
            ntn_config: None,
            mbs_enabled: false,
            prose_enabled: false,
            lcs_enabled: false,
            snpn_config: None,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_nwdaf_task_creation() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);

        let task = NwdafTask::new(task_base);
        assert_eq!(task._max_history_length, 100);
        assert_eq!(task.cell_loads.len(), 0);
    }

    #[tokio::test]
    async fn test_nwdaf_task_ue_measurement() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);

        let mut task = NwdafTask::new(task_base);

        // Record measurement
        task.handle_ue_measurement(1, Some(-80.0), Some(-10.0), (100.0, 200.0, 0.0));

        // Verify measurement was recorded
        let history = task.nwdaf.get_ue_history(1);
        assert!(history.is_some());
        assert_eq!(history.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn a_report_without_rsrp_is_not_recorded_as_a_measurement() {
        // Regression: the ISAC task sends position-only reports. It used to send
        // 0.0 for RSRP, so the analytics layer recorded a constant 0 dBm series
        // -- physically implausible AND perfectly stable, so the z-score anomaly
        // detector could never fire. A report without RSRP must not become a
        // measurement, whether or not it carries RSRQ.
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);

        let mut task = NwdafTask::new(task_base);

        // Position-only, exactly what ISAC forwards.
        task.handle_ue_measurement(7, None, None, (10.0, 20.0, 0.0));
        // RSRQ alone is no better: there is no RSRP series to put it beside.
        task.handle_ue_measurement(7, None, Some(-10.0), (10.0, 20.0, 0.0));
        assert!(
            task.nwdaf.get_ue_history(7).is_none(),
            "a report without RSRP must not be recorded as a measurement"
        );
    }

    /// A report that does carry RSRQ keeps it: making RSRQ optional must not
    /// drop a measured value (#453 review).
    #[tokio::test]
    async fn a_report_with_rsrq_keeps_it() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);
        let mut task = NwdafTask::new(task_base);

        let (tx, rx) = mpsc::channel(4);
        tx.send(TaskMessage::Message(NwdafMessage::UeMeasurement {
            ue_id: 4,
            rsrp: Some(-80.0),
            rsrq: Some(-10.0),
            position: (1.0, 2.0, 0.0),
        }))
        .await
        .unwrap();
        tx.send(TaskMessage::Shutdown).await.unwrap();
        task.run(rx).await;

        let sample = task
            .nwdaf
            .get_ue_history(4)
            .and_then(|h| h.latest())
            .expect("a complete report is recorded");
        assert_eq!(sample.rsrp, -80.0);
        assert_eq!(sample.rsrq, Some(-10.0), "the measured RSRQ is kept");
    }

    /// #453: an RSRP-only report -- what the RLS heartbeat forward and the RRC
    /// MeasurementReport path send, because the simulator has no RSSI to derive
    /// RSRQ from -- is recorded, through the task's own message dispatch, with
    /// the RSRP and position it carried and `rsrq: None`.
    #[tokio::test]
    async fn an_rsrp_only_report_is_recorded_with_no_rsrq() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);
        let mut task = NwdafTask::new(task_base);

        let (tx, rx) = mpsc::channel(4);
        tx.send(TaskMessage::Message(NwdafMessage::UeMeasurement {
            ue_id: 3,
            rsrp: Some(-87.0),
            rsrq: None,
            position: (10.0, 20.0, 0.0),
        }))
        .await
        .unwrap();
        tx.send(TaskMessage::Shutdown).await.unwrap();
        task.run(rx).await;

        let history = task
            .nwdaf
            .get_ue_history(3)
            .expect("the RSRP-only report must be recorded");
        assert_eq!(history.len(), 1);
        let sample = history.latest().expect("one sample");
        assert_eq!(sample.rsrp, -87.0);
        assert_eq!(sample.rsrq, None, "no RSRQ invented");
        assert_eq!((sample.position.x, sample.position.y), (10.0, 20.0));
    }

    #[tokio::test]
    async fn test_nwdaf_task_cell_load() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);

        let mut task = NwdafTask::new(task_base);

        // Record cell load. A measured PRB figure wins over any derivation.
        task.handle_cell_load(1, Some(0.5), 10, None);

        // Verify load was recorded
        assert_eq!(task.cell_loads.len(), 1);
        let load = task.cell_loads.get(&1).unwrap();
        assert_eq!(load.cell_id, 1);
        assert_eq!(load.prb_usage, 0.5);
        assert_eq!(load.connected_ues, 10);
    }

    /// Build a bare NWDAF task for the load tests.
    fn load_test_task() -> NwdafTask {
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(test_config(), DEFAULT_CHANNEL_CAPACITY);
        NwdafTask::new(task_base)
    }

    #[tokio::test]
    async fn a_measured_throughput_drives_the_load_series() {
        // The point of issue #18: with no PRB scheduler, the occupancy series has
        // to come from measured traffic or it is a fabricated constant.
        let mut task = load_test_task();
        task.handle_cell_load(1, None, 3, Some(250.0));

        let load = task.cell_loads.get(&1).expect("load recorded");
        // 250 Mbps against the nominal 1 Gbps ceiling.
        assert!(
            (load.prb_usage - 0.25).abs() < f32::EPSILON,
            "occupancy must follow measured throughput, got {}",
            load.prb_usage
        );
        assert!((load.avg_throughput_mbps - 250.0).abs() < f32::EPSILON);
        assert_eq!(load.connected_ues, 3);
        // NB: this assertion on avg_throughput_mbps cannot distinguish a measured
        // throughput from the old `occupancy * nominal ceiling` restatement --
        // they are equal by construction here. That distinction is made in
        // `a_measured_prb_figure_wins_over_the_traffic_derivation`, and is noted
        // here so a later reader does not mistake this for the guard.
    }

    #[tokio::test]
    async fn a_load_series_varies_with_traffic_rather_than_staying_flat() {
        // A flat series is invisible to the z-score detector, so "it varies" is
        // the property that matters, not any single value.
        let mut task = load_test_task();
        let mut occupancies = Vec::new();
        for mbps in [100.0, 400.0, 50.0] {
            task.handle_cell_load(1, None, 2, Some(mbps));
            // Sampled from what the analytics layer stored, not from the input,
            // so a handler that recorded a constant would be caught here.
            occupancies.push(
                task.nwdaf
                    .get_cell_load(1)
                    .expect("load recorded")
                    .prb_usage,
            );
        }

        assert_eq!(occupancies.len(), 3);
        assert!(
            occupancies.windows(2).all(|w| w[0] != w[1]),
            "the recorded occupancy series must track traffic: {occupancies:?}"
        );
    }

    #[tokio::test]
    async fn a_report_with_neither_prb_nor_throughput_is_not_recorded() {
        // Recording it would mean inventing a load figure, which the anomaly
        // detector and the load predictor both consume as fact.
        let mut task = load_test_task();
        task.handle_cell_load(1, None, 5, None);

        assert!(
            task.cell_loads.is_empty(),
            "a report with no measurement must not become a load sample"
        );
        assert!(
            task.nwdaf.get_cell_load(1).is_none(),
            "nothing must reach the analytics layer either"
        );
    }

    #[tokio::test]
    async fn a_measured_prb_figure_wins_over_the_traffic_derivation() {
        // If a producer ever can measure PRB occupancy, that beats a proxy.
        let mut task = load_test_task();
        task.handle_cell_load(1, Some(0.80), 4, Some(100.0));

        let load = task.cell_loads.get(&1).expect("load recorded");
        assert!((load.prb_usage - 0.80).abs() < f32::EPSILON);
        // This is also the one case that can tell a MEASURED throughput from the
        // old restatement of occupancy through the nominal ceiling: those two are
        // algebraically identical whenever occupancy was derived from throughput
        // (mbps/1000*1000 == mbps), so only a report where PRB and throughput
        // disagree distinguishes them. 0.80 * 1000 would be 800.
        assert!(
            (load.avg_throughput_mbps - 100.0).abs() < f32::EPSILON,
            "throughput must be the measured 100 Mbps, not 0.80 x the nominal \
             ceiling; got {}",
            load.avg_throughput_mbps
        );
    }

    #[tokio::test]
    async fn a_throughput_above_the_nominal_ceiling_clamps_to_full_occupancy() {
        let mut task = load_test_task();
        task.handle_cell_load(1, None, 9, Some(5_000.0));

        let load = task.cell_loads.get(&1).expect("load recorded");
        assert!((load.prb_usage - 1.0).abs() < f32::EPSILON);
        // The throughput itself is NOT clamped: it is a measurement.
        assert!((load.avg_throughput_mbps - 5_000.0).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn an_unloadable_model_path_is_reported_and_the_task_still_starts() {
        // The ONNX flip itself is not verifiable here (no .onnx ships and CI has
        // no ONNX Runtime), but the degradation path is, and it is the one that
        // decides whether a gNB starts at all.
        let mut config = test_config();
        config.nwdaf_model_path = Some(std::path::PathBuf::from(
            "/nonexistent/nwdaf-trajectory.onnx",
        ));
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);

        let task = NwdafTask::new(task_base);
        // Constructed, and still usable: the predictor keeps its documented
        // linear fallback.
        assert!(task.cell_loads.is_empty());
    }

    #[tokio::test]
    async fn test_nwdaf_task_trajectory_prediction() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);

        let mut task = NwdafTask::new(task_base);

        // Record measurements to build history
        for i in 0..10 {
            task.handle_ue_measurement(
                1,
                Some(-80.0),
                Some(-10.0),
                (i as f32 * 10.0, i as f32 * 5.0, 0.0),
            );
        }

        // Request prediction
        task.handle_predict_trajectory(1, 1000, None);

        // Verify prediction was computed
        let history = task.nwdaf.get_ue_history(1);
        assert!(history.is_some());
        assert_eq!(history.unwrap().len(), 10);
    }

    #[tokio::test]
    async fn test_nwdaf_task_handover_recommendation() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);

        let mut task = NwdafTask::new(task_base);

        // Submit handover recommendation (should not panic)
        task.handle_handover_recommendation(1, 2, 0.9);

        // No assertions - just verify no panic
    }

    /// nextgsim#463: the d440b demo at the gNB NWDAF task. These need no ONNX
    /// Runtime except the `ort` module (the model-verify CI job).
    #[cfg(feature = "model-registry")]
    mod d440b {
        use super::*;
        use crate::demo::{registry, D440B_KEY};
        use nextgsim_demo::DemoRuntime;
        use nextgsim_nwdaf::serving::ModelState;
        use std::io::Write as _;
        use std::path::PathBuf;
        use std::sync::{Arc, Mutex};

        fn model_dir() -> PathBuf {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tools/d440b-model")
        }

        /// The shipped model and manifest copied into a temp dir, the model with
        /// its float 2.0 weight's last byte flipped when `flip`.
        fn drop_model(flip: bool) -> (tempfile::TempDir, PathBuf) {
            let dir = tempfile::tempdir().unwrap();
            let mut bytes = std::fs::read(model_dir().join("nwdaf-trajectory-cv.onnx")).unwrap();
            if flip {
                let at = bytes
                    .windows(4)
                    .position(|w| w == [0x00, 0x00, 0x00, 0x40])
                    .unwrap()
                    + 3;
                bytes[at] ^= 0x01;
            }
            let path = dir.path().join("nwdaf-trajectory-cv.onnx");
            std::fs::write(&path, bytes).unwrap();
            std::fs::copy(
                model_dir().join("nwdaf-trajectory-cv.manifest.json"),
                dir.path().join("nwdaf-trajectory-cv.manifest.json"),
            )
            .unwrap();
            (dir, path)
        }

        /// A demo state for `mode` (None: no demo block), leaked to `'static`
        /// as the installed one is; never installed, so tests stay independent.
        fn demo(mode: Option<&str>) -> Option<&'static DemoRuntime> {
            let yaml = match mode {
                Some(m) => format!(
                    "demo:\n  enabled: true\n  {D440B_KEY}:\n    enabled: true\n    mode: {m}\n"
                ),
                None => "nci: 16\n".to_string(),
            };
            let root: serde_yaml::Value = serde_yaml::from_str(&yaml).unwrap();
            Some(Box::leak(Box::new(
                DemoRuntime::from_root(&root, registry()).unwrap(),
            )))
        }

        fn task(model: Option<PathBuf>, demo: Option<&'static DemoRuntime>) -> NwdafTask {
            let mut config = test_config();
            config.nwdaf_model_path = model;
            let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
                GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);
            NwdafTask::with_demo(task_base, 100, demo)
        }

        /// Four measurements of UE 1 along a line.
        fn walk(task: &mut NwdafTask) {
            for i in 0..4 {
                task.handle_ue_measurement(1, Some(-80.0), None, (i as f32, 2.0 * i as f32, 0.0));
            }
        }

        #[derive(Clone, Default)]
        struct Captured(Arc<Mutex<Vec<u8>>>);
        impl std::io::Write for Captured {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        fn capture_logs<T>(f: impl FnOnce() -> T) -> (T, String) {
            let sink = Captured::default();
            let writer = sink.clone();
            let subscriber = tracing_subscriber::fmt()
                .with_writer(move || writer.clone())
                .with_ansi(false)
                .with_max_level(tracing::Level::DEBUG)
                .finish();
            let out = tracing::subscriber::with_default(subscriber, f);
            let mut text = sink.0.lock().unwrap().clone();
            let _ = text.flush();
            (out, String::from_utf8(text).unwrap())
        }

        /// The `consumer` check #462's `onnx` arm makes: a verified model served
        /// every prediction.
        fn consumer_check(task: &NwdafTask) -> bool {
            let s = task.nwdaf.status();
            s.verified && s.onnx_inferences > 0 && s.fallback_inferences == 0
        }

        /// Demo off (each of the three shapes): measurements are recorded and
        /// nothing is served, so no prediction is counted and no demo line is
        /// logged: the task behaves as it does without the demo.
        #[test]
        fn demo_off_serves_nothing() {
            let off: [Option<&'static DemoRuntime>; 4] = [
                None,
                demo(None),
                {
                    let root: serde_yaml::Value =
                        serde_yaml::from_str("demo:\n  enabled: false\n").unwrap();
                    Some(&*Box::leak(Box::new(
                        DemoRuntime::from_root(&root, registry()).unwrap(),
                    )))
                },
                {
                    let root: serde_yaml::Value = serde_yaml::from_str(&format!(
                        "demo:\n  enabled: false\n  {D440B_KEY}:\n    enabled: true\n    mode: verify\n"
                    ))
                    .unwrap();
                    Some(&*Box::leak(Box::new(
                        DemoRuntime::from_root(&root, registry()).unwrap(),
                    )))
                },
            ];
            for d in off {
                let (mut t, logs) = capture_logs(|| {
                    let mut t = task(None, d);
                    walk(&mut t);
                    t
                });
                let s = t.nwdaf.status();
                assert_eq!(s.measurements_recorded, 4);
                assert_eq!((s.onnx_inferences, s.fallback_inferences), (0, 0));
                assert!(!logs.contains("[demo:"), "{logs}");
                assert!(!s.force_fallback);
                // An explicit request still predicts, as it always has.
                t.handle_predict_trajectory(1, 1000, None);
                assert_eq!(t.nwdaf.status().fallback_inferences, 1);
            }
        }

        /// Mode `fallback`: the configured model is neither verified nor
        /// activated (no `model verified`, no ONNX Runtime touched), every served
        /// prediction is a fallback, and the consumer check fails.
        #[test]
        fn mode_fallback_never_verifies_and_serves_the_fallback() {
            let (_dir, path) = drop_model(false);
            let (t, logs) = capture_logs(|| {
                let mut t = task(Some(path), demo(Some("fallback")));
                walk(&mut t);
                t
            });
            assert!(!logs.contains("model verified"), "{logs}");
            assert!(!logs.contains("model refused"), "{logs}");
            assert!(logs.contains("[demo:d440b-model-verify] mode=fallback: the model at "));
            assert!(logs.contains(
                "[demo:d440b-model-verify] first trajectory prediction served by fallback: ue=1 mode=fallback"
            ), "{logs}");
            assert!(matches!(t.nwdaf.model_state(), ModelState::Deferred { .. }));
            let s = t.nwdaf.status();
            assert!(!s.verified && s.force_fallback);
            // 4 measurements, the first has too little history: 3 served.
            assert_eq!((s.onnx_inferences, s.fallback_inferences), (0, 3));
            assert!(!consumer_check(&t));
        }

        /// Mode `verify` with the one-byte-flipped copy: refused for its sha256,
        /// and the same served run is all fallback; the consumer check fails.
        #[test]
        fn mode_verify_refuses_a_flipped_model_and_serves_the_fallback() {
            let (_dir, path) = drop_model(true);
            let (t, logs) = capture_logs(|| {
                let mut t = task(Some(path), demo(Some("verify")));
                walk(&mut t);
                t
            });
            assert!(
                logs.contains(
                    "NWDAF: model refused (sha256 mismatch) id=nwdaf-trajectory-cv version=1.0.0"
                ),
                "{logs}"
            );
            assert!(!logs.contains("model verified"), "{logs}");
            let s = t.nwdaf.status();
            assert!(!s.verified);
            assert!(s.fallback_inferences > 0);
            assert_eq!(s.onnx_inferences, 0);
            assert!(!consumer_check(&t));
        }

        /// A runtime switch applies from the next prediction, and a model a
        /// `fallback` start deferred is verified then, never activated unchecked:
        /// here the flipped copy, refused at the first `verify` prediction.
        #[test]
        fn switching_to_verify_verifies_the_deferred_model_at_the_next_prediction() {
            let (_dir, path) = drop_model(true);
            let rt = demo(Some("fallback")).unwrap();
            let (t, logs) = capture_logs(|| {
                let mut t = task(Some(path), Some(rt));
                walk(&mut t);
                assert!(matches!(t.nwdaf.model_state(), ModelState::Deferred { .. }));
                rt.handle_cli("d440b-model-verify set mode=verify", "test")
                    .unwrap();
                // Not yet: the switch applies at the next prediction.
                assert!(matches!(t.nwdaf.model_state(), ModelState::Deferred { .. }));
                t.handle_ue_measurement(1, Some(-80.0), None, (9.0, 9.0, 0.0));
                t
            });
            assert!(
                logs.contains("[demo:d440b-model-verify] mode is now verify (was fallback)"),
                "{logs}"
            );
            assert!(
                logs.contains("NWDAF: model refused (sha256 mismatch)"),
                "{logs}"
            );
            assert!(matches!(
                t.nwdaf.model_state(),
                ModelState::Refused { reason, .. } if reason == "sha256 mismatch"
            ));
            let s = t.nwdaf.status();
            assert!(!s.force_fallback);
            assert_eq!((s.onnx_inferences, s.fallback_inferences), (0, 4));
        }

        /// The `nwdaf-status` query through the task's own channel.
        #[tokio::test]
        async fn the_status_query_answers_json_through_the_channel() {
            let mut t = task(None, demo(Some("fallback")));
            let (tx, rx) = mpsc::channel(8);
            for i in 0..3 {
                tx.send(TaskMessage::Message(NwdafMessage::UeMeasurement {
                    ue_id: 2,
                    rsrp: Some(-90.0),
                    rsrq: None,
                    position: (i as f32, 0.0, 0.0),
                }))
                .await
                .unwrap();
            }
            let (reply, answer) = tokio::sync::oneshot::channel();
            tx.send(TaskMessage::Message(NwdafMessage::Status { reply }))
                .await
                .unwrap();
            tx.send(TaskMessage::Shutdown).await.unwrap();
            t.run(rx).await;
            let v: serde_json::Value = serde_json::from_str(&answer.await.unwrap()).unwrap();
            assert_eq!(v["schema"], "nextgsim.nwdaf-status/1");
            assert_eq!(v["verified"], false);
            assert_eq!(v["model"]["state"], "not-configured");
            assert_eq!(v["measurements_recorded"], 3);
            assert_eq!(v["onnx_inferences"], 0);
            assert_eq!(v["fallback_inferences"], 2);
        }

        /// Tests that need ONNX Runtime (the model-verify CI job).
        mod ort {
            use super::*;

            fn require_ort() {
                assert!(
                    std::env::var_os("ORT_DYLIB_PATH").is_some(),
                    "this test needs ONNX Runtime: set ORT_DYLIB_PATH to the pinned \
                     libonnxruntime.so (the model-verify CI job does)"
                );
            }

            /// Mode `verify` with the shipped model: `model verified ... golden=ok`,
            /// then every served prediction is the model's and the consumer check
            /// passes. Switching to `fallback` flips the next prediction.
            #[test]
            #[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH); run by the model-verify CI job"]
            fn mode_verify_serves_the_verified_model() {
                require_ort();
                let (_dir, path) = drop_model(false);
                let rt = demo(Some("verify")).unwrap();
                let (mut t, logs) = capture_logs(|| {
                    let mut t = task(Some(path), Some(rt));
                    walk(&mut t);
                    t
                });
                assert!(
                    logs.contains(
                        "NWDAF: model verified id=nwdaf-trajectory-cv version=1.0.0 \
                     sha256=90a74d28c8e919ad8fdaa0f487976c8a4e803af3e318c5a30e02afcc3ec1f826"
                    ),
                    "{logs}"
                );
                assert!(logs.contains("golden=ok"), "{logs}");
                assert!(logs.contains(
                    "[demo:d440b-model-verify] first trajectory prediction served by onnx: ue=1 mode=verify waypoints=1"
                ), "{logs}");
                let s = t.nwdaf.status();
                assert_eq!((s.onnx_inferences, s.fallback_inferences), (3, 0));
                assert!(consumer_check(&t));

                rt.handle_cli("d440b-model-verify set mode=fallback", "test")
                    .unwrap();
                t.handle_ue_measurement(1, Some(-80.0), None, (9.0, 9.0, 0.0));
                let s = t.nwdaf.status();
                assert_eq!((s.onnx_inferences, s.fallback_inferences), (3, 1));
                assert!(
                    !consumer_check(&t),
                    "the fallback arm must fail the consumer"
                );
            }
        }
    }

    #[cfg(feature = "demo-d441d-predictive-ho")]
    mod d441d {
        use super::*;
        use crate::demo::registry;
        use crate::nwdaf::d441d::{Event, Predictor};
        use nextgsim_demo::DemoRuntime;

        fn demo(yaml: &str) -> Option<&'static DemoRuntime> {
            let rt =
                DemoRuntime::from_root(&serde_yaml::from_str(yaml).unwrap(), registry()).unwrap();
            Some(Box::leak(Box::new(rt)))
        }

        fn block(mode: &str, d440b: Option<&str>) -> String {
            let mut y = format!(
                "demo:\n  enabled: true\n  d441d-predictive-ho:\n    enabled: true\n    mode: {mode}\n"
            );
            if let Some(m) = d440b {
                y += &format!("  d440b-model-verify:\n    enabled: true\n    mode: {m}\n");
            }
            y
        }

        /// gnb-west of the shipped configs, shadowing off so the levels are the
        /// distance model's.
        fn west() -> GnbConfig {
            let text = std::fs::read_to_string(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../config/demo-predictive-ho/gnb-west.yaml"
            ))
            .unwrap();
            let mut c: GnbConfig = serde_yaml::from_str(&text).unwrap();
            c.shadowing.sigma_db = 0.0;
            c
        }

        /// A task in `demo`, serving UE 1 walking east at 5 m/s from x 20 for
        /// `secs` heartbeats, through the measurement path the RLS feeds.
        fn walked(
            demo: Option<&'static DemoRuntime>,
            secs: u64,
        ) -> (NwdafTask, mpsc::Receiver<TaskMessage<RrcMessage>>) {
            let (task_base, _app_rx, _ngap_rx, rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
                GnbTaskBase::new(west(), DEFAULT_CHANNEL_CAPACITY);
            let mut task = NwdafTask::with_demo(task_base, 100, demo);
            task.d441d_served(&[1], 0);
            for s in 0..secs {
                let x = 20.0 + 5.0 * s as f32;
                task.handle_ue_measurement_at(1, Some(-x), None, (x, 0.0, 0.0), s * 1000);
            }
            (task, rrc_rx)
        }

        fn recommendations(
            rx: &mut mpsc::Receiver<TaskMessage<RrcMessage>>,
        ) -> Vec<(i32, i32, f32)> {
            let mut out = Vec::new();
            while let Ok(TaskMessage::Message(RrcMessage::NwdafHandoverRecommendation {
                ue_id,
                target_cell,
                confidence,
            })) = rx.try_recv()
            {
                out.push((ue_id, target_cell, confidence));
            }
            out
        }

        /// The wiring: heartbeat measurements reach the engine, a tick predicts,
        /// and the prediction reaches RRC as an NWDAF recommendation for
        /// gnb-east's cell (0x20) above RRC's 0.7 threshold. `nwdaf-status`
        /// carries the handover section.
        #[test]
        fn a_predictive_tick_recommends_the_neighbour_to_rrc() {
            let (mut task, mut rrc_rx) = walked(demo(&block("lsq", None)), 8);
            assert!(recommendations(&mut rrc_rx).is_empty());
            let events = task.d441d_tick(7000);
            assert!(
                matches!(
                    events.as_slice(),
                    [Event::Prediction {
                        predictor: Predictor::Lsq,
                        attempt: 1,
                        ..
                    }]
                ),
                "{events:?}"
            );
            let recs = recommendations(&mut rrc_rx);
            assert_eq!(recs.len(), 1, "{recs:?}");
            assert_eq!((recs[0].0, recs[0].1), (1, 0x20));
            assert!(recs[0].2 > 0.7, "{recs:?}");
            let status: serde_json::Value = serde_json::from_str(&task.status_json()).unwrap();
            assert_eq!(status["handover"]["predictions"], 1, "{status}");
            assert_eq!(status["handover"]["mode"], "lsq", "{status}");
        }

        /// Mode `a3` predicts nothing, and a runtime switch to `lsq` applies
        /// from the next tick.
        #[test]
        fn mode_a3_predicts_nothing_until_switched() {
            let rt = demo(&block("a3", None));
            let (mut task, mut rrc_rx) = walked(rt, 8);
            assert!(task.d441d_tick(7000).is_empty());
            assert!(recommendations(&mut rrc_rx).is_empty());
            rt.unwrap()
                .handle_cli("d441d-predictive-ho set mode=lsq", "test")
                .unwrap();
            assert_eq!(task.d441d_tick(7000).len(), 1);
            assert_eq!(recommendations(&mut rrc_rx).len(), 1);
        }

        /// `onnx` with no verified model to predict with is labelled fallback,
        /// whether d440b is in `verify` with no model configured or not active.
        #[test]
        fn onnx_without_a_verified_model_is_labelled_fallback() {
            for d440b in [Some("verify"), None] {
                let (mut task, mut rrc_rx) = walked(demo(&block("onnx", d440b)), 8);
                let events = task.d441d_tick(7000);
                assert!(
                    matches!(
                        events.as_slice(),
                        [Event::Prediction {
                            predictor: Predictor::Fallback,
                            ..
                        }]
                    ),
                    "{d440b:?}: {events:?}"
                );
                assert_eq!(recommendations(&mut rrc_rx).len(), 1);
            }
        }

        /// The run loop routes RRC's `ServedUes` and `A3Fallback` to the engine:
        /// `nwdaf-status` then counts the served UEs and the fallback.
        #[tokio::test]
        async fn the_run_loop_routes_served_ues_to_the_engine() {
            let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
                GnbTaskBase::new(west(), DEFAULT_CHANNEL_CAPACITY);
            let mut task = NwdafTask::with_demo(task_base, 100, demo(&block("lsq", None)));
            let (tx, rx) = mpsc::channel(8);
            let run = tokio::spawn(async move { task.run(rx).await });
            tx.send(TaskMessage::Message(NwdafMessage::ServedUes {
                ue_ids: vec![4, 9],
            }))
            .await
            .unwrap();
            tx.send(TaskMessage::Message(NwdafMessage::A3Fallback { ue_id: 4 }))
                .await
                .unwrap();
            let (reply, answer) = oneshot::channel();
            tx.send(TaskMessage::Message(NwdafMessage::Status { reply }))
                .await
                .unwrap();
            let status: serde_json::Value = serde_json::from_str(&answer.await.unwrap()).unwrap();
            assert_eq!(status["handover"]["served_ues"], 2, "{status}");
            assert_eq!(status["handover"]["a3_fallbacks"], 1, "{status}");
            assert_eq!(status["handover"]["stranded"], 0, "{status}");
            tx.send(TaskMessage::Shutdown).await.unwrap();
            run.await.unwrap();
        }

        /// With the demo off the task has no D441D state: no tick, no section.
        #[test]
        fn with_the_demo_off_nothing_runs() {
            let (mut task, mut rrc_rx) = walked(demo("nci: 16\n"), 8);
            assert!(task.d441d.is_none());
            assert!(task.d441d_tick(7000).is_empty());
            assert!(recommendations(&mut rrc_rx).is_empty());
            let status: serde_json::Value = serde_json::from_str(&task.status_json()).unwrap();
            assert!(status.get("handover").is_none(), "{status}");
        }
    }
}
