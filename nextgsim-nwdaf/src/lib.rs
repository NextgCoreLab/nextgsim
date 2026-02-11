//! Network Data Analytics Function (NWDAF) for 6G Networks
//!
//! Implements NWDAF per 3GPP TS 23.288 with four-layer analytics:
//! - Layer 1: Real-time anomaly detection
//! - Layer 2: Predictive analytics (trajectory, load)
//! - Layer 3: Prescriptive optimization
//! - Layer 4: Autonomous closed-loop control
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                              NWDAF                                       │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │ Layer 4: Closed-Loop Automation                                  │   │
//! │  │  • Network self-optimization                                     │   │
//! │  │  • Autonomous decision execution                                 │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │ Layer 3: Prescriptive Optimization                               │   │
//! │  │  • Handover recommendations                                      │   │
//! │  │  • Resource optimization                                         │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │ Layer 2: Predictive Analytics                                    │   │
//! │  │  • Trajectory prediction (ONNX + linear fallback)                │   │
//! │  │  • Load prediction                                               │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │ Layer 1: Real-time Analytics                                     │   │
//! │  │  • Z-score anomaly detection                                     │   │
//! │  │  • Performance monitoring                                        │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │ Data Collection Layer                                            │   │
//! │  │  • Data source registration (gNB/UE/NF)                          │   │
//! │  │  • UE measurements / Cell load                                   │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! │                                                                         │
//! │  MTLF (Model Training LF)     AnLF (Analytics LF)                      │
//! │  ┌──────────────────────┐     ┌──────────────────────────────────┐   │
//! │  │ • Model management    │────▶│ • Analytics execution             │   │
//! │  │ • ONNX model loading  │     │ • Subscription management         │   │
//! │  │ • Model provision     │     │ • Anomaly detection               │   │
//! │  └──────────────────────┘     └──────────────────────────────────┘   │
//! │                                                                         │
//! │  TS 23.288 Service APIs:                                                │
//! │  • Nnwdaf_AnalyticsSubscription                                         │
//! │  • Nnwdaf_AnalyticsInfo                                                 │
//! │  • Nnwdaf_MLModelProvision                                              │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # 3GPP Compliance
//!
//! This implementation aligns with:
//! - 3GPP TS 23.288: Network Data Analytics Services
//! - MTLF/AnLF logical separation (TS 23.288 Section 6.2A/6.2B)
//! - Standardized Analytics IDs (TS 23.288 Section 6.1)
//! - Service operations: Nnwdaf_AnalyticsSubscription, Nnwdaf_AnalyticsInfo,
//!   Nnwdaf_MLModelProvision (TS 23.288 Sections 7.2, 7.3, 7.5)
//!
//! # Modules
//!
//! - [`analytics_id`] - TS 23.288 standardized analytics identifiers
//! - [`anomaly`] - Statistical anomaly detection (z-score based)
//! - [`anlf`] - Analytics Logical Function
//! - [`data_collection`] - Data source registration and measurement storage
//! - [`error`] - Error types for all NWDAF operations
//! - [`mtlf`] - Model Training Logical Function
//! - [`predictor`] - ONNX model prediction with linear fallback
//! - [`service`] - TS 23.288 service operations

pub mod analytics_id;
pub mod anlf;
pub mod anomaly;
pub mod coordination;
pub mod data_collection;
pub mod dccf;
pub mod error;
pub mod llm_analytics;
pub mod ml_training;
pub mod model_transfer;
pub mod mtlf;
pub mod nkef_bridge;
pub mod predictor;
pub mod service;

// Re-export key types from new modules for convenience
pub use analytics_id::AnalyticsId;
pub use anlf::{
    AnalyticsPayload, AnalyticsResult, Anlf, CellEnergyModel, QosMetrics, QosThresholds,
    ResourceType, ServiceLevelMetrics, ServiceType,
};
pub use anomaly::{Anomaly, AnomalyDetector, AnomalySeverity};
pub use coordination::{
    DelegationRequest, DelegationResponse, LoadBalancingStrategy, NwdafCapabilities,
    NwdafCoordinator, NwdafInstance, SharedAnalyticsResult,
};
pub use data_collection::{
    DataCollector, DataSourceRegistration, DataSourceType, MeasurementCapability,
};
pub use dccf::{
    AggregatedData, AggregationMethod, DataCollectionFilter, DataCollectionSession,
    DataRoutingPolicy, DataTransformation, Dccf, DccfDataSource, GeographicArea,
    RoutingCondition, SessionStatus,
};
pub use error::{
    AnalyticsError, DataCollectionError, NwdafError, PredictionError, SubscriptionError,
};
pub use llm_analytics::{LlmAnalyticsEngine, LlmAnalyticsQuery, LlmAnalyticsResponse};
pub use ml_training::{
    DistributedTrainingCoordinator, FederatedRound, FederatedStrategy, MlModelTrainingService,
    ModelArchitecture, ModelTrainingRequest, ModelTrainingResponse, ModelUpdate, TrainingConfig,
    TrainingDataset, TrainingJob, TrainingMetrics, TrainingSample, TrainingStatus,
};
pub use model_transfer::{
    ModelFilter, ModelMetadata, ModelPackage, ModelTransferMessage, ModelTransferProtocol,
    ModelType, TransferStatistics,
};
pub use mtlf::{AbTestResult, MlModelInfo, Mtlf};
pub use nkef_bridge::{NwdafNkefBridge, NwdafNkefBridgeConfig};
pub use predictor::{OnnxPredictor, PredictionMethod, PredictionOutput};
pub use service::{
    AnalyticsAccuracyFeedback, AnalyticsAccuracyTracker, AnalyticsCallback, AnalyticsInfoRequest,
    AnalyticsInfoResponse, AnalyticsInfoService, AnalyticsQueryParams, AccuracyStats,
    DataManagementOp, DataManagementRequest, DataManagementResponse, DataManagementService,
    MlModelProvisionService, SubscriptionManager, SubscriptionRequest, SubscriptionResponse,
};

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Core data types (backward-compatible with original API)
// ---------------------------------------------------------------------------

/// 3D position vector
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Vector3 {
    /// X coordinate
    pub x: f64,
    /// Y coordinate
    pub y: f64,
    /// Z coordinate
    pub z: f64,
}

impl Vector3 {
    /// Creates a new Vector3
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Calculates distance to another point
    pub fn distance_to(&self, other: &Vector3) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// UE measurement data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UeMeasurement {
    /// UE identifier
    pub ue_id: i32,
    /// Reference Signal Received Power (dBm)
    pub rsrp: f32,
    /// Reference Signal Received Quality (dB)
    pub rsrq: f32,
    /// Signal to Interference plus Noise Ratio (dB)
    pub sinr: Option<f32>,
    /// UE position
    pub position: Vector3,
    /// UE velocity (m/s)
    pub velocity: Option<Vector3>,
    /// Serving cell ID
    pub serving_cell_id: i32,
    /// Measurement timestamp
    pub timestamp_ms: u64,
}

/// Cell load data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellLoad {
    /// Cell identifier
    pub cell_id: i32,
    /// Physical Resource Block usage (0.0 to 1.0)
    pub prb_usage: f32,
    /// Number of connected UEs
    pub connected_ues: u32,
    /// Average throughput (Mbps)
    pub avg_throughput_mbps: f32,
    /// Timestamp
    pub timestamp_ms: u64,
}

/// Trajectory prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryPrediction {
    /// UE identifier
    pub ue_id: i32,
    /// Predicted waypoints with timestamps
    pub waypoints: Vec<(Vector3, u64)>,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Prediction horizon in milliseconds
    pub horizon_ms: u32,
}

impl TrajectoryPrediction {
    /// Checks if UE will enter a cell based on position
    pub fn will_enter_cell(&self, _cell_id: i32, cell_position: Vector3, cell_radius: f64) -> bool {
        self.waypoints
            .iter()
            .any(|(pos, _)| pos.distance_to(&cell_position) < cell_radius)
    }
}

/// Handover recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoverRecommendation {
    /// UE identifier
    pub ue_id: i32,
    /// Recommended target cell
    pub target_cell_id: i32,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Reason for recommendation
    pub reason: HandoverReason,
    /// Recommended timing (ms from now)
    pub recommended_timing_ms: Option<u32>,
}

/// Reason for handover recommendation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HandoverReason {
    /// Traditional signal-based (A3 event)
    SignalBased,
    /// Predicted mobility
    PredictedMobility,
    /// Load balancing
    LoadBalancing,
    /// Coverage optimization
    CoverageOptimization,
}

/// Automation action type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AutomationAction {
    /// Adjust handover parameters
    AdjustHandoverParams {
        /// Cell ID
        cell_id: i32,
        /// New hysteresis value (dB)
        new_hysteresis: f32,
        /// New time-to-trigger (ms)
        new_ttt: u32,
    },
    /// Adjust cell power
    AdjustCellPower {
        /// Cell ID
        cell_id: i32,
        /// New transmit power (dBm)
        new_tx_power: f32,
    },
    /// Trigger load balancing
    TriggerLoadBalancing {
        /// Source cell ID
        source_cell_id: i32,
        /// Target cell ID
        target_cell_id: i32,
        /// Number of UEs to offload
        ue_count: u32,
    },
}

/// NWDAF messages
#[derive(Debug)]
pub enum NwdafMessage {
    /// UE measurement report
    UeMeasurement(UeMeasurement),
    /// Cell load report
    CellLoad(CellLoad),
    /// Request trajectory prediction
    PredictTrajectory {
        /// UE ID
        ue_id: i32,
        /// Prediction horizon (ms)
        horizon_ms: u32,
        /// Response channel
        response_tx: Option<tokio::sync::oneshot::Sender<NwdafResponse>>,
    },
    /// Request handover recommendation
    RequestHandoverRecommendation {
        /// UE ID
        ue_id: i32,
        /// Response channel
        response_tx: Option<tokio::sync::oneshot::Sender<NwdafResponse>>,
    },
    /// Execute automation action
    ExecuteAutomation(AutomationAction),
}

/// NWDAF responses
#[derive(Debug)]
pub enum NwdafResponse {
    /// Trajectory prediction result
    TrajectoryPrediction(TrajectoryPrediction),
    /// Handover recommendation
    HandoverRecommendation(HandoverRecommendation),
    /// Error
    Error(String),
}

/// Measurement history for a UE
#[derive(Debug)]
pub struct UeMeasurementHistory {
    /// UE identifier
    pub ue_id: i32,
    /// Historical measurements
    measurements: VecDeque<UeMeasurement>,
    /// Maximum history length
    max_length: usize,
}

impl UeMeasurementHistory {
    /// Creates a new history for a UE
    pub fn new(ue_id: i32, max_length: usize) -> Self {
        Self {
            ue_id,
            measurements: VecDeque::with_capacity(max_length),
            max_length,
        }
    }

    /// Adds a measurement to the history
    pub fn add(&mut self, measurement: UeMeasurement) {
        if self.measurements.len() >= self.max_length {
            self.measurements.pop_front();
        }
        self.measurements.push_back(measurement);
    }

    /// Returns the most recent measurement
    pub fn latest(&self) -> Option<&UeMeasurement> {
        self.measurements.back()
    }

    /// Returns all measurements
    pub fn all(&self) -> &VecDeque<UeMeasurement> {
        &self.measurements
    }

    /// Returns the number of measurements
    pub fn len(&self) -> usize {
        self.measurements.len()
    }

    /// Returns true if history is empty
    pub fn is_empty(&self) -> bool {
        self.measurements.is_empty()
    }

    /// Extracts position sequence for trajectory prediction
    pub fn position_sequence(&self) -> Vec<Vector3> {
        self.measurements.iter().map(|m| m.position).collect()
    }
}

// ---------------------------------------------------------------------------
// NwdafManager - backward-compatible facade over the new architecture
// ---------------------------------------------------------------------------

/// NWDAF analytics manager
///
/// This is the backward-compatible entry point that delegates to the
/// new MTLF/AnLF architecture internally. Existing users of `NwdafManager`
/// continue to work unchanged.
///
/// For new code, prefer using [`Anlf`], [`Mtlf`], [`DataCollector`], and
/// the TS 23.288 service types directly.
#[derive(Debug)]
pub struct NwdafManager {
    /// Data collector (replaces inline histories)
    data_collector: DataCollector,
    /// Analytics Logical Function
    anlf: Anlf,
    /// Model Training Logical Function
    mtlf: Mtlf,
    /// Maximum history length (kept for backward compat)
    max_history_length: usize,
    /// Handover confidence threshold
    handover_confidence_threshold: f32,
}

impl NwdafManager {
    /// Creates a new NWDAF manager
    pub fn new(max_history_length: usize) -> Self {
        Self {
            data_collector: DataCollector::new(max_history_length),
            anlf: Anlf::new(),
            mtlf: Mtlf::new(),
            max_history_length,
            handover_confidence_threshold: 0.8,
        }
    }

    /// Records a UE measurement
    ///
    /// Stores the measurement in the data collector and runs it through
    /// the AnLF anomaly detector.
    pub fn record_measurement(&mut self, measurement: UeMeasurement) {
        // Feed into anomaly detector before storing
        let _anomalies = self.anlf.process_measurement(&measurement);
        self.data_collector.report_ue_measurement(measurement);
    }

    /// Records cell load
    pub fn record_cell_load(&mut self, load: CellLoad) {
        let _anomalies = self.anlf.process_cell_load(&load);
        self.data_collector.report_cell_load(load);
    }

    /// Gets UE measurement history
    pub fn get_ue_history(&self, ue_id: i32) -> Option<&UeMeasurementHistory> {
        self.data_collector.get_ue_history(ue_id)
    }

    /// Gets latest cell load
    pub fn get_cell_load(&self, cell_id: i32) -> Option<&CellLoad> {
        self.data_collector.get_latest_cell_load(cell_id)
    }

    /// Predicts trajectory using ONNX model or linear extrapolation fallback
    ///
    /// If an ONNX model has been loaded via [`NwdafManager::load_trajectory_model`],
    /// uses ML-based prediction. Otherwise falls back to linear extrapolation.
    pub fn predict_trajectory(
        &self,
        ue_id: i32,
        horizon_ms: u32,
    ) -> Option<TrajectoryPrediction> {
        let history = self.data_collector.get_ue_history(ue_id)?;

        if history.len() < 2 {
            return None;
        }

        let positions = history.position_sequence();
        let timestamps: Vec<u64> = history.all().iter().map(|m| m.timestamp_ms).collect();
        let current_time = timestamps.last().copied().unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0)
        });

        // Try MTLF predictor first, then create a temporary fallback
        let predictor = self.mtlf.trajectory_predictor();

        if let Some(pred) = predictor {
            match pred.predict_trajectory(&positions, &timestamps, horizon_ms, current_time) {
                Ok(output) => {
                    return Some(TrajectoryPrediction {
                        ue_id,
                        waypoints: output.waypoints,
                        confidence: output.confidence,
                        horizon_ms,
                    });
                }
                Err(_) => {} // fall through to linear extrapolation
            }
        }

        // Linear extrapolation fallback (same algorithm as original)
        let n = positions.len();
        let p1 = &positions[n - 2];
        let p2 = &positions[n - 1];

        let dt = if timestamps.len() >= 2 {
            let t1 = timestamps[timestamps.len() - 2];
            let t2 = timestamps[timestamps.len() - 1];
            if t2 > t1 {
                (t2 - t1) as f64 / 1000.0
            } else {
                0.1
            }
        } else {
            0.1
        };

        let vx = (p2.x - p1.x) / dt;
        let vy = (p2.y - p1.y) / dt;
        let vz = (p2.z - p1.z) / dt;

        let num_waypoints = 10u32;
        let step_ms = horizon_ms / num_waypoints;

        let waypoints: Vec<(Vector3, u64)> = (1..=num_waypoints)
            .map(|i| {
                let t = (f64::from(i) * f64::from(step_ms)) / 1000.0;
                let pos = Vector3::new(p2.x + vx * t, p2.y + vy * t, p2.z + vz * t);
                let timestamp = current_time + (u64::from(i) * u64::from(step_ms));
                (pos, timestamp)
            })
            .collect();

        let confidence =
            (history.len() as f32 / self.max_history_length as f32).min(0.9);

        Some(TrajectoryPrediction {
            ue_id,
            waypoints,
            confidence,
            horizon_ms,
        })
    }

    /// Generates handover recommendation
    pub fn recommend_handover(
        &self,
        ue_id: i32,
        neighbor_cells: &[(i32, f32)], // (cell_id, rsrp)
    ) -> Option<HandoverRecommendation> {
        let history = self.data_collector.get_ue_history(ue_id)?;
        let latest = history.latest()?;

        // Find best neighbor cell
        let best_neighbor = neighbor_cells
            .iter()
            .filter(|(_, rsrp)| *rsrp > latest.rsrp + 3.0) // 3dB hysteresis
            .max_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })?;

        // Check if trajectory prediction suggests handover
        let prediction = self.predict_trajectory(ue_id, 1000);
        let reason = if prediction.map(|p| p.confidence > 0.7).unwrap_or(false) {
            HandoverReason::PredictedMobility
        } else {
            HandoverReason::SignalBased
        };

        let confidence = 0.85; // Would be computed by ML model

        Some(HandoverRecommendation {
            ue_id,
            target_cell_id: best_neighbor.0,
            confidence,
            reason,
            recommended_timing_ms: Some(100),
        })
    }

    /// Sets the handover confidence threshold
    pub fn set_handover_confidence_threshold(&mut self, threshold: f32) {
        self.handover_confidence_threshold = threshold;
        self.anlf.set_handover_confidence_threshold(threshold);
    }

    // --- New API methods that expose the underlying architecture ---

    /// Loads a trajectory prediction ONNX model into the MTLF
    ///
    /// Once loaded, [`predict_trajectory`](NwdafManager::predict_trajectory) will
    /// use the ML model instead of linear extrapolation.
    ///
    /// # Errors
    ///
    /// Returns an error if the model file cannot be loaded.
    pub fn load_trajectory_model(
        &mut self,
        path: &std::path::Path,
    ) -> Result<(), NwdafError> {
        self.mtlf.load_trajectory_model(path)
    }

    /// Returns a reference to the underlying data collector
    pub fn data_collector(&self) -> &DataCollector {
        &self.data_collector
    }

    /// Returns a mutable reference to the data collector
    pub fn data_collector_mut(&mut self) -> &mut DataCollector {
        &mut self.data_collector
    }

    /// Returns a reference to the AnLF
    pub fn anlf(&self) -> &Anlf {
        &self.anlf
    }

    /// Returns a mutable reference to the AnLF
    pub fn anlf_mut(&mut self) -> &mut Anlf {
        &mut self.anlf
    }

    /// Returns a reference to the MTLF
    pub fn mtlf(&self) -> &Mtlf {
        &self.mtlf
    }

    /// Returns a mutable reference to the MTLF
    pub fn mtlf_mut(&mut self) -> &mut Mtlf {
        &mut self.mtlf
    }

    /// Performs UE mobility analytics via the AnLF
    ///
    /// # Errors
    ///
    /// Returns an error if the UE has insufficient data.
    pub fn analyze_ue_mobility(
        &mut self,
        ue_id: i32,
        horizon_ms: u32,
    ) -> Result<AnalyticsResult, NwdafError> {
        self.anlf
            .analyze_ue_mobility(ue_id, horizon_ms, &self.data_collector, &self.mtlf)
    }

    /// Performs NF load analytics via the AnLF
    ///
    /// # Errors
    ///
    /// Returns an error if the cell has insufficient data.
    pub fn analyze_nf_load(
        &mut self,
        cell_id: i32,
        horizon_steps: usize,
    ) -> Result<AnalyticsResult, NwdafError> {
        self.anlf
            .analyze_nf_load(cell_id, horizon_steps, &self.data_collector, &self.mtlf)
    }

    /// Performs Service Experience analytics via the AnLF (TS 23.288 6.4)
    ///
    /// Computes MOS scores and service-level metrics for the given target.
    ///
    /// # Errors
    ///
    /// Returns an error if the target has insufficient data.
    pub fn analyze_service_experience(
        &mut self,
        target: &analytics_id::AnalyticsTarget,
    ) -> Result<AnalyticsResult, NwdafError> {
        self.anlf
            .analyze_service_experience(target, &self.data_collector)
    }

    /// Performs User Data Congestion analytics via the AnLF (TS 23.288 6.8)
    ///
    /// Analyzes congestion levels and predicts future congestion for the given cell.
    ///
    /// # Errors
    ///
    /// Returns an error if the target has insufficient data.
    pub fn analyze_user_data_congestion(
        &mut self,
        target: &analytics_id::AnalyticsTarget,
    ) -> Result<AnalyticsResult, NwdafError> {
        self.anlf
            .analyze_user_data_congestion(target, &self.data_collector, &self.mtlf)
    }

    /// Performs QoS Sustainability analytics via the AnLF (TS 23.288 6.6)
    ///
    /// Predicts whether current QoS levels can be sustained.
    ///
    /// # Errors
    ///
    /// Returns an error if the target has insufficient data.
    pub fn analyze_qos_sustainability(
        &mut self,
        target: &analytics_id::AnalyticsTarget,
    ) -> Result<AnalyticsResult, NwdafError> {
        self.anlf
            .analyze_qos_sustainability(target, &self.data_collector, &self.mtlf)
    }

    /// Returns recent anomalies detected by the AnLF
    pub fn recent_anomalies(&self) -> &std::collections::VecDeque<Anomaly> {
        self.anlf.anomaly_detector().recent_anomalies()
    }
}

impl Default for NwdafManager {
    fn default() -> Self {
        Self::new(100)
    }
}

// ---------------------------------------------------------------------------
// Tests (preserving original tests + adding new integration tests)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Original tests (backward compatibility) ---

    #[test]
    fn test_vector3_distance() {
        let p1 = Vector3::new(0.0, 0.0, 0.0);
        let p2 = Vector3::new(3.0, 4.0, 0.0);
        assert!((p1.distance_to(&p2) - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_measurement_history() {
        let mut history = UeMeasurementHistory::new(1, 10);

        for i in 0..15 {
            history.add(UeMeasurement {
                ue_id: 1,
                rsrp: -80.0 + i as f32,
                rsrq: -10.0,
                sinr: Some(15.0),
                position: Vector3::new(i as f64, 0.0, 0.0),
                velocity: None,
                serving_cell_id: 1,
                timestamp_ms: i as u64 * 100,
            });
        }

        assert_eq!(history.len(), 10); // Max length enforced
        assert_eq!(history.latest().unwrap().rsrp, -80.0 + 14.0);
    }

    #[test]
    fn test_nwdaf_manager() {
        let mut manager = NwdafManager::new(100);

        // Record measurements
        for i in 0..10 {
            manager.record_measurement(UeMeasurement {
                ue_id: 1,
                rsrp: -80.0,
                rsrq: -10.0,
                sinr: Some(15.0),
                position: Vector3::new(i as f64 * 10.0, 0.0, 0.0),
                velocity: None,
                serving_cell_id: 1,
                timestamp_ms: i as u64 * 100,
            });
        }

        assert!(manager.get_ue_history(1).is_some());
        assert_eq!(manager.get_ue_history(1).unwrap().len(), 10);
    }

    #[test]
    fn test_trajectory_prediction() {
        let mut manager = NwdafManager::new(100);

        // Record measurements with linear movement
        for i in 0..10 {
            manager.record_measurement(UeMeasurement {
                ue_id: 1,
                rsrp: -80.0,
                rsrq: -10.0,
                sinr: Some(15.0),
                position: Vector3::new(i as f64 * 10.0, i as f64 * 5.0, 0.0),
                velocity: None,
                serving_cell_id: 1,
                timestamp_ms: i as u64 * 100,
            });
        }

        let prediction = manager.predict_trajectory(1, 1000);
        assert!(prediction.is_some());

        let pred = prediction.unwrap();
        assert_eq!(pred.ue_id, 1);
        assert!(!pred.waypoints.is_empty());
        assert!(pred.confidence > 0.0);
    }

    // --- New integration tests ---

    #[test]
    fn test_nwdaf_manager_new_apis() {
        let mut manager = NwdafManager::new(100);

        // Record measurements
        for i in 0..10 {
            manager.record_measurement(UeMeasurement {
                ue_id: 1,
                rsrp: -80.0,
                rsrq: -10.0,
                sinr: Some(15.0),
                position: Vector3::new(i as f64 * 10.0, 0.0, 0.0),
                velocity: None,
                serving_cell_id: 1,
                timestamp_ms: i as u64 * 100,
            });
        }

        // Test new analyze_ue_mobility
        let result = manager.analyze_ue_mobility(1, 1000);
        assert!(result.is_ok());

        let analytics = result.expect("should succeed");
        assert_eq!(analytics.analytics_id, AnalyticsId::UeMobility);
    }

    #[test]
    fn test_nwdaf_manager_load_analytics() {
        let mut manager = NwdafManager::new(100);

        // Record cell loads
        for i in 0..20 {
            manager.record_cell_load(CellLoad {
                cell_id: 1,
                prb_usage: 0.3 + i as f32 * 0.02,
                connected_ues: 10,
                avg_throughput_mbps: 150.0,
                timestamp_ms: i as u64 * 100,
            });
        }

        let result = manager.analyze_nf_load(1, 5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_nwdaf_manager_accessor_methods() {
        let manager = NwdafManager::new(100);

        // Test that accessor methods work
        assert!(manager.data_collector().source_count() == 0);
        assert!(manager.anlf().supports_analytics(AnalyticsId::UeMobility));
        assert_eq!(manager.mtlf().model_count(), 0);
    }

    #[test]
    fn test_nwdaf_manager_anomaly_detection() {
        let mut manager = NwdafManager::new(100);

        // Build baseline
        for i in 0..30 {
            manager.record_measurement(UeMeasurement {
                ue_id: 1,
                rsrp: -80.0,
                rsrq: -10.0,
                sinr: Some(15.0),
                position: Vector3::new(i as f64, 0.0, 0.0),
                velocity: None,
                serving_cell_id: 1,
                timestamp_ms: i as u64 * 100,
            });
        }

        // Inject anomalous measurement
        manager.record_measurement(UeMeasurement {
            ue_id: 1,
            rsrp: -30.0, // huge spike
            rsrq: -10.0,
            sinr: Some(15.0),
            position: Vector3::new(30.0, 0.0, 0.0),
            velocity: None,
            serving_cell_id: 1,
            timestamp_ms: 3000,
        });

        // Should have detected at least one anomaly
        assert!(
            !manager.recent_anomalies().is_empty(),
            "Should detect RSRP anomaly"
        );
    }

    #[test]
    fn test_full_nwdaf_workflow() {
        // This test exercises the full flow:
        // 1. Create manager
        // 2. Register data source
        // 3. Report measurements
        // 4. Run analytics
        // 5. Check results

        let mut manager = NwdafManager::new(100);

        // Register a data source
        let reg = DataSourceRegistration {
            source_id: "gnb-1".to_string(),
            source_type: DataSourceType::Gnb,
            description: "Test gNB".to_string(),
            capabilities: vec![
                MeasurementCapability::RadioMeasurement,
                MeasurementCapability::CellLoad,
            ],
            reporting_interval_ms: 100,
            active: true,
        };
        manager.data_collector_mut().register_source(reg).expect("should register");

        // Report measurements
        for i in 0..20 {
            manager.record_measurement(UeMeasurement {
                ue_id: 1,
                rsrp: -80.0,
                rsrq: -10.0,
                sinr: Some(15.0),
                position: Vector3::new(i as f64 * 10.0, i as f64 * 5.0, 0.0),
                velocity: None,
                serving_cell_id: 1,
                timestamp_ms: i as u64 * 100,
            });

            manager.record_cell_load(CellLoad {
                cell_id: 1,
                prb_usage: 0.3 + i as f32 * 0.01,
                connected_ues: 5 + i as u32,
                avg_throughput_mbps: 150.0 - i as f32,
                timestamp_ms: i as u64 * 100,
            });
        }

        // Run trajectory prediction (backward compat)
        let traj = manager.predict_trajectory(1, 1000);
        assert!(traj.is_some());

        // Run analytics (new API)
        let mobility = manager.analyze_ue_mobility(1, 1000);
        assert!(mobility.is_ok());

        let load = manager.analyze_nf_load(1, 5);
        assert!(load.is_ok());

        // Handover recommendation
        let ho = manager.recommend_handover(1, &[(2, -70.0)]);
        assert!(ho.is_some());

        // Check data collector stats
        assert_eq!(manager.data_collector().tracked_ue_count(), 1);
        assert_eq!(manager.data_collector().tracked_cell_count(), 1);
        assert_eq!(manager.data_collector().source_count(), 1);
    }

    #[test]
    fn test_analytics_id_enum() {
        // Verify all analytics IDs are accessible
        let all = AnalyticsId::all();
        assert_eq!(all.len(), 8);
        assert!(all.contains(&AnalyticsId::UeMobility));
        assert!(all.contains(&AnalyticsId::NfLoad));
        assert!(all.contains(&AnalyticsId::ServiceExperience));
        assert!(all.contains(&AnalyticsId::AbnormalBehavior));
        assert!(all.contains(&AnalyticsId::UserDataCongestion));
        assert!(all.contains(&AnalyticsId::QosSustainability));
        assert!(all.contains(&AnalyticsId::EnergyEfficiency));
        assert!(all.contains(&AnalyticsId::SliceOptimization));
    }

    #[test]
    fn test_nwdaf_manager_service_experience() {
        let mut manager = NwdafManager::new(100);

        for i in 0..20 {
            manager.record_cell_load(CellLoad {
                cell_id: 1,
                prb_usage: 0.4 + i as f32 * 0.01,
                connected_ues: 10 + i as u32,
                avg_throughput_mbps: 120.0,
                timestamp_ms: i as u64 * 100,
            });
        }

        let result = manager.analyze_service_experience(
            &analytics_id::AnalyticsTarget::Cell { cell_id: 1 },
        );
        assert!(result.is_ok());

        let analytics = result.expect("should succeed");
        assert_eq!(analytics.analytics_id, AnalyticsId::ServiceExperience);
    }

    #[test]
    fn test_nwdaf_manager_user_data_congestion() {
        let mut manager = NwdafManager::new(100);

        for i in 0..20 {
            manager.record_cell_load(CellLoad {
                cell_id: 1,
                prb_usage: 0.5 + i as f32 * 0.02,
                connected_ues: 20 + i as u32 * 2,
                avg_throughput_mbps: 100.0 - i as f32 * 2.0,
                timestamp_ms: i as u64 * 100,
            });
        }

        let result = manager.analyze_user_data_congestion(
            &analytics_id::AnalyticsTarget::Cell { cell_id: 1 },
        );
        assert!(result.is_ok());

        let analytics = result.expect("should succeed");
        assert_eq!(analytics.analytics_id, AnalyticsId::UserDataCongestion);
    }

    #[test]
    fn test_nwdaf_manager_qos_sustainability() {
        let mut manager = NwdafManager::new(100);

        for i in 0..20 {
            manager.record_cell_load(CellLoad {
                cell_id: 1,
                prb_usage: 0.3 + i as f32 * 0.01,
                connected_ues: 15,
                avg_throughput_mbps: 150.0,
                timestamp_ms: i as u64 * 100,
            });
        }

        let result = manager.analyze_qos_sustainability(
            &analytics_id::AnalyticsTarget::Cell { cell_id: 1 },
        );
        assert!(result.is_ok());

        let analytics = result.expect("should succeed");
        assert_eq!(analytics.analytics_id, AnalyticsId::QosSustainability);
    }
}
