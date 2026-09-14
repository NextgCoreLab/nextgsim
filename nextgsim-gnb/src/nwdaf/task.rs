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
}

impl NwdafTask {
    /// Creates a new NWDAF task with default history length (100)
    pub fn new(task_base: GnbTaskBase) -> Self {
        Self::with_history_length(task_base, 100)
    }

    /// Creates a new NWDAF task with specified history length
    pub fn with_history_length(task_base: GnbTaskBase, max_history_length: usize) -> Self {
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
    /// A report without RSRP/RSRQ is **not** recorded as a measurement. The
    /// analytics layer's `UeMeasurement` requires both as `f32`, so recording a
    /// position-only report means inventing values, and there is no safe filler:
    /// 0.0 reads as an implausibly strong signal, and any "typical" constant
    /// makes the series artificially stable so the z-score anomaly detector can
    /// never fire. Such reports are logged and dropped instead.
    fn handle_ue_measurement(
        &mut self,
        ue_id: i32,
        rsrp: Option<f32>,
        rsrq: Option<f32>,
        position: (f32, f32, f32),
    ) {
        let (Some(rsrp), Some(rsrq)) = (rsrp, rsrq) else {
            debug!(
                "NWDAF: UE {ue_id} report carries no RSRP/RSRQ (pos=({}, {}, {})); \
                 not recording a radio measurement",
                position.0, position.1, position.2
            );
            return;
        };

        debug!(
            "NWDAF: UE {} measurement - RSRP={} dBm, RSRQ={} dB, pos=({}, {}, {})",
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
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        };

        self.nwdaf.record_measurement(measurement);

        // Check for anomalies
        let anomalies = self.nwdaf.recent_anomalies();
        if !anomalies.is_empty() {
            warn!(
                "NWDAF: Detected {} anomalies for UE {}",
                anomalies.len(),
                ue_id
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

        loop {
            // The bus arm is selected over only when a subscription exists, so a
            // build with the feature on but no 6G task init behaves as before.
            #[cfg(feature = "event-bus")]
            let msg = {
                if let Some(ref mut bus_rx) = bus_rx {
                    tokio::select! {
                        received = rx.recv() => received,
                        bus_event = bus_rx.recv() => {
                            Self::log_bus_event(bus_event);
                            continue;
                        }
                    }
                } else {
                    rx.recv().await
                }
            };
            #[cfg(not(feature = "event-bus"))]
            let msg = rx.recv().await;

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
    async fn test_position_only_report_is_not_recorded_as_a_measurement() {
        // Regression: the ISAC task is the only live producer of UeMeasurement
        // and it has no serving-cell RSRP/RSRQ. It used to send 0.0 for both,
        // so the analytics layer recorded a constant 0 dBm series -- physically
        // implausible AND perfectly stable, meaning the z-score anomaly detector
        // could never fire while appearing to have real data to work on.
        //
        // A report with no radio measurement must not become a measurement.
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);

        let mut task = NwdafTask::new(task_base);

        // Position-only, exactly what ISAC forwards.
        task.handle_ue_measurement(7, None, None, (10.0, 20.0, 0.0));
        assert!(
            task.nwdaf.get_ue_history(7).is_none(),
            "a report without RSRP/RSRQ must not be recorded as a measurement"
        );

        // A partial report is equally unusable.
        task.handle_ue_measurement(7, Some(-80.0), None, (10.0, 20.0, 0.0));
        assert!(
            task.nwdaf.get_ue_history(7).is_none(),
            "a report missing RSRQ must not be recorded either"
        );

        // A complete report still is.
        task.handle_ue_measurement(7, Some(-80.0), Some(-10.0), (10.0, 20.0, 0.0));
        assert_eq!(
            task.nwdaf.get_ue_history(7).map(|h| h.len()),
            Some(1),
            "a complete report must be recorded"
        );
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
}
