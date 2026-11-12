//! Network Data Analytics Function (NWDAF) for 6G Networks
//!
//! This crate is a **research prototype**. It is architecturally aligned with the
//! NWDAF analytics concepts of 3GPP TS 23.288, but runs entirely in-process: it
//! implements no Nnwdaf SBI surface, no TS 29.500 ProblemDetails error model and
//! no notification correlation IDs, and it is not conformance-tested.
//!
//! It models a four-layer analytics stack:
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
//! # 3GPP Alignment
//!
//! This implementation aligns with:
//! - 3GPP TS 23.288: Network Data Analytics Services
//! - MTLF/AnLF logical separation (TS 23.288 Section 6.2A/6.2B)
//! - Standardized Analytics IDs (TS 23.288 Section 6.1)
//! - Service operations: `Nnwdaf_AnalyticsSubscription`, `Nnwdaf_AnalyticsInfo`,
//!   `Nnwdaf_MLModelProvision` (TS 23.288 Sections 7.2, 7.3, 7.5)
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
pub mod event_exposure;
pub mod federation;
pub mod llm_analytics;
pub mod ml_training;
pub mod model_transfer;
pub mod mtlf;
pub mod nkef_bridge;
pub mod predictor;
pub mod service;
/// Verified trajectory-model loading and serving counters (nextgsim#463).
#[cfg(feature = "model-registry")]
pub mod serving;

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
    DataRoutingPolicy, DataTransformation, Dccf, DccfDataSource, GeographicArea, RoutingCondition,
    SessionStatus,
};
pub use error::{
    AnalyticsError, DataCollectionError, NwdafError, PredictionError, SubscriptionError,
};
pub use event_exposure::{
    EventData, EventExposureManager, EventNotification, EventSubscription, EventType,
    MobilityState, NfType, SubscriptionStatus, TargetArea,
};
pub use federation::{
    AggregatedFederationResult, AreaFilter, ConsentType, DataFederationManager,
    FederatedAggregationMethod, FederatedAnalyticsResult, FederationPeer, FederationRequest,
    FederationResponse, PlmnId, PrivacyPolicy, TrustLevel,
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
    AccuracyStats, AnalyticsAccuracyFeedback, AnalyticsAccuracyTracker, AnalyticsCallback,
    AnalyticsInfoRequest, AnalyticsInfoResponse, AnalyticsInfoService, AnalyticsQueryParams,
    DataManagementOp, DataManagementRequest, DataManagementResponse, DataManagementService,
    MlModelProvisionService, SubscriptionManager, SubscriptionRequest, SubscriptionResponse,
};

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Core data types (backward-compatible with original API)
// ---------------------------------------------------------------------------

pub use nextgsim_common::Vector3;

/// A trajectory prediction must be more confident than this for a handover
/// recommendation to be classed [`HandoverReason::PredictedMobility`].
///
/// A recommendation's confidence means different things per reason: for
/// `PredictedMobility` it is the trajectory fit's confidence (so 0.71 is
/// possible), for `SignalBased` it is the fixed
/// [`SIGNAL_BASED_HANDOVER_CONFIDENCE`]. Do not compare or rank confidences
/// across the two reasons.
pub const PREDICTED_MOBILITY_CONFIDENCE: f32 = 0.7;

/// A trajectory must also move at least this fast (m/s, from its waypoints)
/// to count as mobility. A parked UE is predicted to stay with full
/// confidence; that is a good prediction, but not a reason to hand over.
pub const MIN_PREDICTED_MOBILITY_SPEED_MPS: f64 = 0.5;

/// The confidence of a [`HandoverReason::SignalBased`] recommendation: the
/// neighbour was measured 3 dB above the serving cell, which no prediction
/// qualifies, so it is a fixed value (the 0.85 every recommendation carried
/// before nextgsim#467). Not comparable with a `PredictedMobility` confidence;
/// see [`PREDICTED_MOBILITY_CONFIDENCE`].
pub const SIGNAL_BASED_HANDOVER_CONFIDENCE: f32 = 0.85;

/// Whether a trajectory prediction is evidence of mobility (nextgsim#467): its
/// confidence is finite and above [`PREDICTED_MOBILITY_CONFIDENCE`], and its
/// waypoints move at least [`MIN_PREDICTED_MOBILITY_SPEED_MPS`].
pub fn is_predicted_mobility(confidence: f32, waypoints: &[(Vector3, u64)]) -> bool {
    confidence.is_finite()
        && confidence > PREDICTED_MOBILITY_CONFIDENCE
        && predictor::waypoint_speed_mps(waypoints) >= MIN_PREDICTED_MOBILITY_SPEED_MPS
}

/// UE measurement data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UeMeasurement {
    /// UE identifier
    pub ue_id: i32,
    /// Reference Signal Received Power (dBm)
    pub rsrp: f32,
    /// Reference Signal Received Quality (dB), or `None` when the producer has
    /// none (#453).
    ///
    /// Optional because RSRQ cannot be derived honestly from what every
    /// producer has: TS 38.215 §5.1.3 defines it as N×RSRP/RSSI, and the
    /// simulator's RLS channel model yields one received power and no RSSI. A
    /// filler value would be invented data. Readers that would use it define
    /// their behaviour without it; see `anlf`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rsrq: Option<f32>,
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
    /// Handover confidence threshold
    handover_confidence_threshold: f32,
    /// Serving counters and latencies (nextgsim#463).
    #[cfg(feature = "model-registry")]
    serving: serving::ServingStats,
    /// Where the configured trajectory model stands (nextgsim#463).
    #[cfg(feature = "model-registry")]
    model_state: serving::ModelState,
    /// The manifest of the verified model, for the inventory.
    #[cfg(feature = "model-registry")]
    verified_manifest: Option<nextgsim_ai::ModelManifest>,
    /// Every prediction uses the linear fallback, verified model or not (the
    /// d440b `fallback` mode).
    #[cfg(feature = "model-registry")]
    force_fallback: bool,
}

impl NwdafManager {
    /// Creates a new NWDAF manager
    pub fn new(max_history_length: usize) -> Self {
        Self {
            data_collector: DataCollector::new(max_history_length),
            anlf: Anlf::new(),
            mtlf: Mtlf::new(),
            handover_confidence_threshold: 0.8,
            #[cfg(feature = "model-registry")]
            serving: serving::ServingStats::default(),
            #[cfg(feature = "model-registry")]
            model_state: serving::ModelState::NotConfigured,
            #[cfg(feature = "model-registry")]
            verified_manifest: None,
            #[cfg(feature = "model-registry")]
            force_fallback: false,
        }
    }

    /// Records a UE measurement
    ///
    /// Stores the measurement in the data collector and runs it through
    /// the `AnLF` anomaly detector.
    pub fn record_measurement(&mut self, measurement: UeMeasurement) {
        // Feed into anomaly detector before storing
        let _anomalies = self.anlf.process_measurement(&measurement);
        self.data_collector.report_ue_measurement(measurement);
        #[cfg(feature = "model-registry")]
        self.serving.record_measurement();
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
    ///
    /// With `model-registry`, every prediction is counted by the method that
    /// produced it (see [`serving::ServingStats`]), and
    /// [`NwdafManager::set_force_fallback`] skips the model.
    pub fn predict_trajectory(&self, ue_id: i32, horizon_ms: u32) -> Option<TrajectoryPrediction> {
        #[cfg(feature = "model-registry")]
        {
            let start = std::time::Instant::now();
            let (prediction, by_model) =
                self.predict_trajectory_by(ue_id, horizon_ms, !self.force_fallback)?;
            self.serving.record_inference(by_model, start.elapsed());
            Some(prediction)
        }
        #[cfg(not(feature = "model-registry"))]
        self.predict_trajectory_by(ue_id, horizon_ms, true)
            .map(|(prediction, _)| prediction)
    }

    /// The prediction, and whether the ONNX model produced it. `use_model`
    /// false goes straight to linear extrapolation.
    fn predict_trajectory_by(
        &self,
        ue_id: i32,
        horizon_ms: u32,
        use_model: bool,
    ) -> Option<(TrajectoryPrediction, bool)> {
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
        let predictor = self.mtlf.trajectory_predictor().filter(|_| use_model);

        if let Some(pred) = predictor {
            if let Ok(output) =
                pred.predict_trajectory(&positions, &timestamps, horizon_ms, current_time)
            {
                let by_model = output.method == PredictionMethod::OnnxModel;
                return Some((
                    TrajectoryPrediction {
                        ue_id,
                        waypoints: output.waypoints,
                        confidence: output.confidence,
                        horizon_ms,
                    },
                    by_model,
                ));
            }
            // fall through to linear extrapolation
        }

        // The least-squares fallback, the same one the predictor runs without a
        // model (nextgsim#467), so both paths report the same R² confidence.
        let output = crate::predictor::least_squares_trajectory(
            &positions,
            &timestamps,
            horizon_ms,
            current_time,
        );
        let (waypoints, confidence) = (output.waypoints, output.confidence);

        Some((
            TrajectoryPrediction {
                ue_id,
                waypoints,
                confidence,
                horizon_ms,
            },
            false,
        ))
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
        // A prediction confident enough, and moving fast enough, to call it
        // mobility carries its own confidence (nextgsim#467). A signal-based recommendation rests on the
        // measured 3 dB margin above, not on a prediction, and keeps the fixed
        // SIGNAL_BASED_HANDOVER_CONFIDENCE.
        let prediction = self.predict_trajectory(ue_id, 1000);
        let (reason, confidence) = match prediction {
            Some(p) if is_predicted_mobility(p.confidence, &p.waypoints) => {
                (HandoverReason::PredictedMobility, p.confidence)
            }
            _ => (
                HandoverReason::SignalBased,
                SIGNAL_BASED_HANDOVER_CONFIDENCE,
            ),
        };

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
    pub fn load_trajectory_model(&mut self, path: &std::path::Path) -> Result<(), NwdafError> {
        self.mtlf.load_trajectory_model(path)
    }

    /// Verify the trajectory model at `path` against its manifest
    /// (`<stem>.manifest.json` beside it) and activate it only if every check
    /// passes (nextgsim#463). Logs `NWDAF: model verified ...` or
    /// `NWDAF: model refused (<reason>) ...`; a refused model leaves predictions
    /// on the fallback.
    ///
    /// # Errors
    /// The refusal.
    #[cfg(feature = "model-registry")]
    pub fn load_verified_trajectory_model(
        &mut self,
        path: &std::path::Path,
    ) -> Result<nextgsim_ai::VerifiedModel, nextgsim_ai::ModelRefused> {
        let manifest_path = serving::manifest_path(path);
        let outcome = std::fs::read_to_string(&manifest_path)
            .map_err(|e| {
                nextgsim_ai::RefusalReason::ManifestInvalid(format!(
                    "{}: {e}",
                    manifest_path.display()
                ))
            })
            .and_then(|text| nextgsim_ai::ModelManifest::from_json(&text))
            .map_err(|reason| nextgsim_ai::ModelRefused {
                id: "-".into(),
                version: "-".into(),
                reason,
            })
            .and_then(|manifest| {
                self.mtlf
                    .load_verified_trajectory_model(path, &manifest)
                    .map(|v| (v, manifest))
            });
        match outcome {
            Ok((verified, manifest)) => {
                tracing::info!(
                    "NWDAF: model verified {} path={}; trajectory predictions use the model",
                    verified.summary(),
                    path.display()
                );
                self.model_state = serving::ModelState::Verified {
                    id: manifest.id.clone(),
                    version: manifest.version.clone(),
                    sha256: verified.sha256.clone(),
                    opset: manifest.opset,
                    path: path.display().to_string(),
                };
                self.verified_manifest = Some(manifest);
                Ok(verified)
            }
            Err(refused) => {
                tracing::warn!(
                    "NWDAF: {refused} path={}; trajectory predictions use linear extrapolation",
                    path.display()
                );
                self.model_state = serving::ModelState::Refused {
                    id: refused.id.clone(),
                    version: refused.version.clone(),
                    reason: refused.reason.label().to_string(),
                    detail: refused.reason.detail(),
                    path: path.display().to_string(),
                };
                Err(refused)
            }
        }
    }

    /// Record that the model at `path` is configured but neither verified nor
    /// activated (the d440b `fallback` mode at startup).
    #[cfg(feature = "model-registry")]
    pub fn defer_trajectory_model(&mut self, path: &std::path::Path) {
        self.model_state = serving::ModelState::Deferred {
            path: path.display().to_string(),
        };
    }

    /// Force every prediction onto the linear fallback (`true`), or let a
    /// verified model serve again (`false`). Applies from the next prediction.
    #[cfg(feature = "model-registry")]
    pub fn set_force_fallback(&mut self, force: bool) {
        self.force_fallback = force;
    }

    /// Where the configured trajectory model stands.
    #[cfg(feature = "model-registry")]
    pub fn model_state(&self) -> &serving::ModelState {
        &self.model_state
    }

    /// The serving counters.
    #[cfg(feature = "model-registry")]
    pub fn serving_stats(&self) -> &serving::ServingStats {
        &self.serving
    }

    /// The `nwdaf-status` answer.
    #[cfg(feature = "model-registry")]
    pub fn status(&self) -> serving::NwdafStatus {
        serving::NwdafStatus::new(
            &self.serving,
            &self.model_state,
            self.verified_manifest.as_ref(),
            self.force_fallback,
        )
    }

    /// Returns a reference to the underlying data collector
    pub fn data_collector(&self) -> &DataCollector {
        &self.data_collector
    }

    /// Returns a mutable reference to the data collector
    pub fn data_collector_mut(&mut self) -> &mut DataCollector {
        &mut self.data_collector
    }

    /// Returns a reference to the `AnLF`
    pub fn anlf(&self) -> &Anlf {
        &self.anlf
    }

    /// Returns a mutable reference to the `AnLF`
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

    /// Performs UE mobility analytics via the `AnLF`
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

    /// Performs NF load analytics via the `AnLF`
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

    /// Performs Service Experience analytics via the `AnLF` (TS 23.288 6.4)
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

    /// Performs User Data Congestion analytics via the `AnLF` (TS 23.288 6.8)
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

    /// Performs `QoS` Sustainability analytics via the `AnLF` (TS 23.288 6.6)
    ///
    /// Predicts whether current `QoS` levels can be sustained.
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

    /// Returns recent anomalies detected by the `AnLF`
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

    /// `rsrq` is optional on the wire too (#453): a record without it reads as
    /// `None`, one with it reads back the value, and `None` is not serialised.
    #[test]
    fn ue_measurement_rsrq_round_trips_through_serde() {
        let base = r#"{"ue_id":1,"rsrp":-80.0,"sinr":null,"position":{"x":0.0,"y":0.0,"z":0.0},"velocity":null,"serving_cell_id":1,"timestamp_ms":0"#;
        let without: UeMeasurement = serde_json::from_str(&format!("{base}}}")).unwrap();
        assert_eq!(without.rsrq, None);
        let with: UeMeasurement =
            serde_json::from_str(&format!("{base},\"rsrq\":-10.0}}")).unwrap();
        assert_eq!(with.rsrq, Some(-10.0));
        assert!(!serde_json::to_string(&without).unwrap().contains("rsrq"));
        assert!(serde_json::to_string(&with)
            .unwrap()
            .contains("\"rsrq\":-10.0"));
    }

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
                rsrq: Some(-10.0),
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
                rsrq: Some(-10.0),
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
                rsrq: Some(-10.0),
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
                rsrq: Some(-10.0),
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
                rsrq: Some(-10.0),
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
            rsrq: Some(-10.0),
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
        manager
            .data_collector_mut()
            .register_source(reg)
            .expect("should register");

        // Report measurements
        for i in 0..20 {
            manager.record_measurement(UeMeasurement {
                ue_id: 1,
                rsrp: -80.0,
                rsrq: Some(-10.0),
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

        let result =
            manager.analyze_service_experience(&analytics_id::AnalyticsTarget::Cell { cell_id: 1 });
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

        let result = manager
            .analyze_user_data_congestion(&analytics_id::AnalyticsTarget::Cell { cell_id: 1 });
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

        let result =
            manager.analyze_qos_sustainability(&analytics_id::AnalyticsTarget::Cell { cell_id: 1 });
        assert!(result.is_ok());

        let analytics = result.expect("should succeed");
        assert_eq!(analytics.analytics_id, AnalyticsId::QosSustainability);
    }

    /// nextgsim#467: the manager's own fallback is the least-squares fit, and
    /// recommend_handover classes a confident prediction as PredictedMobility
    /// with the prediction's confidence, instead of a fixed 0.85 behind a test
    /// the 0.7 cap made unreachable. A scattered track stays SignalBased with
    /// the fixed signal-based confidence.
    /// #527 review MAJOR: a parked UE is predicted to stay with full
    /// confidence, but staying is not mobility, so a better neighbour gives a
    /// SignalBased recommendation at the fixed confidence.
    #[test]
    fn a_stationary_ue_is_signal_based_not_predicted_mobility() {
        let mut manager = NwdafManager::new(20);
        for i in 0..10u64 {
            manager.record_measurement(UeMeasurement {
                ue_id: 1,
                rsrp: -80.0,
                rsrq: None,
                sinr: None,
                position: Vector3::new(40.0, 7.0, 0.0),
                velocity: None,
                serving_cell_id: 1,
                timestamp_ms: 1_000 + i * 1_000,
            });
        }
        let traj = manager.predict_trajectory(1, 1000).expect("predicts");
        assert!(
            traj.confidence > 0.99,
            "staying put is a confident prediction"
        );
        let ho = manager
            .recommend_handover(1, &[(2, -70.0)])
            .expect("a 10 dB better neighbour");
        assert_eq!(ho.reason, HandoverReason::SignalBased);
        assert_eq!(ho.confidence, SIGNAL_BASED_HANDOVER_CONFIDENCE);
    }

    /// The mobility test: finite, above the threshold, and moving.
    #[test]
    fn is_predicted_mobility_needs_confidence_and_motion() {
        let moving: Vec<(Vector3, u64)> = (0..10)
            .map(|i| {
                (
                    Vector3::new(f64::from(i), 0.0, 0.0),
                    1_000 + u64::from(i as u32) * 100,
                )
            })
            .collect(); // 10 m/s
        let slow: Vec<(Vector3, u64)> = (0..10)
            .map(|i| {
                (
                    Vector3::new(0.04 * f64::from(i), 0.0, 0.0),
                    u64::from(i as u32) * 100,
                )
            })
            .collect(); // 0.4 m/s
        assert!(is_predicted_mobility(0.9, &moving));
        assert!(!is_predicted_mobility(0.7, &moving), "0.7 is not above 0.7");
        assert!(!is_predicted_mobility(f32::NAN, &moving));
        assert!(!is_predicted_mobility(0.9, &slow), "below the speed floor");
        assert!(!is_predicted_mobility(0.9, &[]));
    }

    #[test]
    fn recommend_handover_carries_the_least_squares_confidence() {
        let record = |manager: &mut NwdafManager, zigzag: bool| {
            for i in 0..10 {
                let y = if zigzag {
                    if i % 2 == 0 {
                        15.0
                    } else {
                        -15.0
                    }
                } else {
                    0.0
                };
                manager.record_measurement(UeMeasurement {
                    ue_id: 1,
                    rsrp: -80.0,
                    rsrq: None,
                    sinr: None,
                    position: Vector3::new((35.0 + 2.5 * f64::from(i)).round(), y, 0.0),
                    velocity: None,
                    serving_cell_id: 1,
                    timestamp_ms: 1_000 + i as u64 * 1_000,
                });
            }
        };

        let mut clean = NwdafManager::new(20);
        record(&mut clean, false);
        let traj = clean.predict_trajectory(1, 1000).expect("predicts");
        assert!(traj.confidence > 0.7, "confidence {}", traj.confidence);
        let ho = clean
            .recommend_handover(1, &[(2, -70.0)])
            .expect("a 10 dB better neighbour");
        assert_eq!(ho.reason, HandoverReason::PredictedMobility);
        assert_eq!(ho.confidence, traj.confidence);

        let mut noisy = NwdafManager::new(20);
        record(&mut noisy, true);
        assert!(noisy.predict_trajectory(1, 1000).unwrap().confidence < 0.7);
        let ho = noisy
            .recommend_handover(1, &[(2, -70.0)])
            .expect("a 10 dB better neighbour");
        assert_eq!(ho.reason, HandoverReason::SignalBased);
        assert_eq!(ho.confidence, SIGNAL_BASED_HANDOVER_CONFIDENCE);
    }
}
