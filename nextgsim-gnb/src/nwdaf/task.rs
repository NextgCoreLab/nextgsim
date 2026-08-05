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
        let nwdaf = NwdafManager::new(max_history_length);

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

    /// Handles a cell load report
    fn handle_cell_load(&mut self, cell_id: i32, prb_usage: f32, connected_ues: u32) {
        debug!(
            "NWDAF: Cell {} load - PRB usage={:.1}%, connected UEs={}",
            cell_id,
            prb_usage * 100.0,
            connected_ues
        );

        let load = CellLoad {
            cell_id,
            prb_usage,
            connected_ues,
            // NOT a measurement. The gNB keeps no per-cell throughput counter,
            // and its config carries no cell bandwidth or numerology from which
            // a capacity could be derived, so there is nothing to compute this
            // from. It is PRB usage scaled by a nominal 1 Gbps reference
            // ceiling: a restatement of prb_usage in Mbps-shaped units, not an
            // independent signal.
            //
            // Kept rather than zeroed because the field is not Option and a 0
            // would read as an idle cell. Any analytics treating this as
            // observed throughput is reading prb_usage twice.
            avg_throughput_mbps: prb_usage * NOMINAL_CELL_CAPACITY_MBPS,
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

        loop {
            match rx.recv().await {
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
                    } => {
                        self.handle_cell_load(cell_id, prb_usage, connected_ues);
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

        // Record cell load
        task.handle_cell_load(1, 0.5, 10);

        // Verify load was recorded
        assert_eq!(task.cell_loads.len(), 1);
        let load = task.cell_loads.get(&1).unwrap();
        assert_eq!(load.cell_id, 1);
        assert_eq!(load.prb_usage, 0.5);
        assert_eq!(load.connected_ues, 10);
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
