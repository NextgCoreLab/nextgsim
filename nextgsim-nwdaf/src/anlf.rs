//! Analytics Logical Function (AnLF)
//!
//! Implements the AnLF component of NWDAF as defined in 3GPP TS 23.288.
//! AnLF is responsible for:
//! - Performing analytics inference using models from MTLF
//! - Exposing analytics results via Nnwdaf services
//! - Running anomaly detection on incoming data
//! - Generating handover recommendations and load predictions
//!
//! AnLF consumes trained models from MTLF and collected data from
//! the DataCollector to produce analytics outputs.

use std::collections::VecDeque;

use tracing::{debug, warn};

use crate::analytics_id::{AnalyticsId, AnalyticsOutputType, AnalyticsTarget};
use crate::anomaly::{Anomaly, AnomalyDetector};
use crate::data_collection::DataCollector;
use crate::error::{AnalyticsError, NwdafError};
use crate::mtlf::Mtlf;
use crate::predictor::OnnxPredictor;
use crate::{
    CellLoad, HandoverReason, HandoverRecommendation, TrajectoryPrediction,
    UeMeasurement,
};
use serde::{Deserialize, Serialize};

/// Analytics result container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsResult {
    /// Analytics ID that produced this result
    pub analytics_id: AnalyticsId,
    /// Target of the analytics
    pub target: AnalyticsTarget,
    /// Output type (statistics or predictions)
    pub output_type: AnalyticsOutputType,
    /// Timestamp of the result
    pub timestamp_ms: u64,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// The actual analytics payload
    pub payload: AnalyticsPayload,
}

/// Payload variants for different analytics types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalyticsPayload {
    /// UE mobility analytics result
    UeMobility {
        /// Predicted trajectory (if predictions requested)
        trajectory: Option<TrajectoryPrediction>,
        /// Handover recommendation (if applicable)
        handover_recommendation: Option<HandoverRecommendation>,
    },
    /// NF load analytics result
    NfLoad {
        /// Current load value (0.0 to 1.0)
        current_load: f32,
        /// Predicted future load values
        predicted_load: Vec<f32>,
        /// Estimated time to overload (None if not expected)
        time_to_overload_ms: Option<u64>,
    },
    /// Abnormal behaviour detection result
    AbnormalBehavior {
        /// Detected anomalies
        anomalies: Vec<Anomaly>,
    },
    /// QoS sustainability result
    QosSustainability {
        /// Whether current QoS can be sustained
        sustainable: bool,
        /// Estimated remaining time at current QoS level
        remaining_time_ms: Option<u64>,
        /// Current QoS metrics
        current_qos: QosMetrics,
    },
    /// Service experience result (TS 23.288 6.4)
    ServiceExperience {
        /// Mean Opinion Score (1.0 to 5.0)
        mos: f32,
        /// Service-level metrics
        metrics: ServiceLevelMetrics,
        /// List of degraded services
        degraded_services: Vec<String>,
    },
    /// User data congestion result (TS 23.288 6.8)
    UserDataCongestion {
        /// Congestion level (0.0 to 1.0)
        congestion_level: f32,
        /// Predicted congestion in future time windows
        predicted_congestion: Vec<(u64, f32)>, // (timestamp_ms, level)
        /// Affected areas or cells
        affected_areas: Vec<i32>,
    },
    /// Energy efficiency analytics result (Rel-19, IMT-2030)
    EnergyEfficiency {
        /// Energy efficiency score (0.0 to 1.0, higher = more efficient)
        efficiency_score: f32,
        /// Estimated power consumption in watts
        power_consumption_watts: f32,
        /// Energy per bit in joules/bit
        energy_per_bit: f32,
        /// Recommendations for energy optimization
        recommendations: Vec<String>,
    },
    /// Network slice optimization result (Rel-19)
    SliceOptimization {
        /// Per-slice utilization (slice_id, utilization 0.0-1.0)
        slice_utilization: Vec<(i32, f32)>,
        /// SLA compliance per slice (slice_id, compliant)
        sla_compliance: Vec<(i32, bool)>,
        /// Reallocation recommendations
        reallocation_recommendations: Vec<String>,
    },
}

/// QoS metrics for sustainability analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QosMetrics {
    /// Average latency in milliseconds
    pub avg_latency_ms: f32,
    /// Packet loss rate (0.0 to 1.0)
    pub packet_loss_rate: f32,
    /// Average throughput in Mbps
    pub throughput_mbps: f32,
}

/// Service-level metrics for experience analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceLevelMetrics {
    /// Average response time in milliseconds
    pub avg_response_time_ms: f32,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f32,
    /// Service availability (0.0 to 1.0)
    pub availability: f32,
    /// Data rate in Mbps
    pub data_rate_mbps: f32,
}

/// Analytics Logical Function
///
/// Consumes ML models from MTLF and data from DataCollector to produce
/// analytics. Provides the main analytics execution engine for NWDAF.
///
/// # 3GPP Reference
///
/// - TS 23.288 Section 6.2B: NWDAF containing AnLF
/// - TS 23.288 Section 7.3: Nnwdaf_AnalyticsInfo service
/// - TS 23.288 Section 7.2: Nnwdaf_AnalyticsSubscription service
#[derive(Debug)]
pub struct Anlf {
    /// Anomaly detector instance
    anomaly_detector: AnomalyDetector,
    /// Fallback predictor (when MTLF has no model loaded)
    fallback_predictor: Option<OnnxPredictor>,
    /// Supported analytics IDs
    supported_analytics: Vec<AnalyticsId>,
    /// Recent analytics results (bounded buffer)
    recent_results: VecDeque<AnalyticsResult>,
    /// Maximum stored results
    max_results: usize,
    /// Handover confidence threshold
    handover_confidence_threshold: f32,
}

impl Anlf {
    /// Creates a new AnLF instance
    pub fn new() -> Self {
        let fallback = OnnxPredictor::new().ok();

        Self {
            anomaly_detector: AnomalyDetector::new(2.5, 200).with_min_samples(10),
            fallback_predictor: fallback,
            supported_analytics: vec![
                AnalyticsId::UeMobility,
                AnalyticsId::NfLoad,
                AnalyticsId::AbnormalBehavior,
                AnalyticsId::QosSustainability,
                AnalyticsId::ServiceExperience,
                AnalyticsId::UserDataCongestion,
                AnalyticsId::EnergyEfficiency,
                AnalyticsId::SliceOptimization,
            ],
            recent_results: VecDeque::with_capacity(1000),
            max_results: 1000,
            handover_confidence_threshold: 0.8,
        }
    }

    /// Returns the list of supported analytics IDs
    pub fn supported_analytics(&self) -> &[AnalyticsId] {
        &self.supported_analytics
    }

    /// Checks if a given analytics ID is supported
    pub fn supports_analytics(&self, id: AnalyticsId) -> bool {
        self.supported_analytics.contains(&id)
    }

    /// Returns a reference to the anomaly detector
    pub fn anomaly_detector(&self) -> &AnomalyDetector {
        &self.anomaly_detector
    }

    /// Returns a mutable reference to the anomaly detector
    pub fn anomaly_detector_mut(&mut self) -> &mut AnomalyDetector {
        &mut self.anomaly_detector
    }

    /// Sets the handover confidence threshold
    pub fn set_handover_confidence_threshold(&mut self, threshold: f32) {
        self.handover_confidence_threshold = threshold;
    }

    /// Processes a UE measurement through the anomaly detector
    ///
    /// Feeds the measurement values into the anomaly detector and
    /// returns any detected anomalies.
    pub fn process_measurement(&mut self, measurement: &UeMeasurement) -> Vec<Anomaly> {
        let entity_id = format!("ue-{}", measurement.ue_id);
        let ts = measurement.timestamp_ms;
        let mut anomalies = Vec::new();

        // Check RSRP
        anomalies.extend(self.anomaly_detector.check(
            "rsrp",
            &entity_id,
            f64::from(measurement.rsrp),
            ts,
        ));

        // Check RSRQ
        anomalies.extend(self.anomaly_detector.check(
            "rsrq",
            &entity_id,
            f64::from(measurement.rsrq),
            ts,
        ));

        // Check SINR if available
        if let Some(sinr) = measurement.sinr {
            anomalies.extend(
                self.anomaly_detector
                    .check("sinr", &entity_id, f64::from(sinr), ts),
            );
        }

        anomalies
    }

    /// Processes a cell load measurement through the anomaly detector
    pub fn process_cell_load(&mut self, load: &CellLoad) -> Vec<Anomaly> {
        let entity_id = format!("cell-{}", load.cell_id);
        let ts = load.timestamp_ms;
        let mut anomalies = Vec::new();

        anomalies.extend(self.anomaly_detector.check(
            "prb_usage",
            &entity_id,
            f64::from(load.prb_usage),
            ts,
        ));

        anomalies.extend(self.anomaly_detector.check(
            "throughput",
            &entity_id,
            f64::from(load.avg_throughput_mbps),
            ts,
        ));

        anomalies
    }

    /// Performs UE mobility analytics (trajectory prediction)
    ///
    /// Uses the MTLF's trajectory predictor if available, otherwise
    /// falls back to the AnLF's local fallback predictor.
    ///
    /// # Errors
    ///
    /// Returns `AnalyticsError::TargetNotFound` if the UE has no measurement history.
    /// Returns `AnalyticsError::InsufficientData` if too few measurements exist.
    pub fn analyze_ue_mobility(
        &mut self,
        ue_id: i32,
        horizon_ms: u32,
        data_collector: &DataCollector,
        mtlf: &Mtlf,
    ) -> Result<AnalyticsResult, NwdafError> {
        let history = data_collector
            .get_ue_history(ue_id)
            .ok_or(AnalyticsError::TargetNotFound {
                target: format!("ue-{ue_id}"),
            })?;

        if history.len() < 2 {
            return Err(AnalyticsError::InsufficientData {
                required: 2,
                available: history.len(),
            }
            .into());
        }

        let positions = history.position_sequence();
        let timestamps: Vec<u64> = history.all().iter().map(|m| m.timestamp_ms).collect();
        let current_time = timestamps.last().copied().unwrap_or(0);

        // Try MTLF predictor first, then fallback
        let predictor: &OnnxPredictor = mtlf
            .trajectory_predictor()
            .or(self.fallback_predictor.as_ref())
            .ok_or(AnalyticsError::ComputationFailed {
                reason: "No predictor available".to_string(),
            })?;

        let prediction_output =
            predictor.predict_trajectory(&positions, &timestamps, horizon_ms, current_time)?;

        let trajectory = TrajectoryPrediction {
            ue_id,
            waypoints: prediction_output.waypoints,
            confidence: prediction_output.confidence,
            horizon_ms,
        };

        let result = AnalyticsResult {
            analytics_id: AnalyticsId::UeMobility,
            target: AnalyticsTarget::Ue { ue_id },
            output_type: AnalyticsOutputType::Predictions,
            timestamp_ms: current_time,
            confidence: trajectory.confidence,
            payload: AnalyticsPayload::UeMobility {
                trajectory: Some(trajectory),
                handover_recommendation: None,
            },
        };

        self.store_result(result.clone());
        Ok(result)
    }

    /// Generates a handover recommendation for a UE
    ///
    /// Combines trajectory prediction with signal measurements from
    /// neighbor cells to determine the best handover target.
    ///
    /// # Errors
    ///
    /// Returns an error if the UE has no measurement history.
    pub fn recommend_handover(
        &mut self,
        ue_id: i32,
        neighbor_cells: &[(i32, f32)], // (cell_id, rsrp)
        data_collector: &DataCollector,
        mtlf: &Mtlf,
    ) -> Result<Option<HandoverRecommendation>, NwdafError> {
        let history = data_collector
            .get_ue_history(ue_id)
            .ok_or(AnalyticsError::TargetNotFound {
                target: format!("ue-{ue_id}"),
            })?;

        let latest = history.latest().ok_or(AnalyticsError::InsufficientData {
            required: 1,
            available: 0,
        })?;

        // Find best neighbor cell (must be at least 3dB better - hysteresis)
        let best_neighbor = neighbor_cells
            .iter()
            .filter(|(_, rsrp)| *rsrp > latest.rsrp + 3.0)
            .max_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        let best_neighbor = match best_neighbor {
            Some(n) => n,
            None => return Ok(None),
        };

        // Check trajectory prediction for mobility-based handover
        let positions = history.position_sequence();
        let timestamps: Vec<u64> = history.all().iter().map(|m| m.timestamp_ms).collect();
        let current_time = timestamps.last().copied().unwrap_or(0);

        let predictor = mtlf
            .trajectory_predictor()
            .or(self.fallback_predictor.as_ref());

        let reason = if let Some(pred) = predictor {
            match pred.predict_trajectory(&positions, &timestamps, 1000, current_time) {
                Ok(output) if output.confidence > 0.7 => HandoverReason::PredictedMobility,
                _ => HandoverReason::SignalBased,
            }
        } else {
            HandoverReason::SignalBased
        };

        let confidence = 0.85;
        let recommendation = HandoverRecommendation {
            ue_id,
            target_cell_id: best_neighbor.0,
            confidence,
            reason,
            recommended_timing_ms: Some(100),
        };

        debug!(
            "AnLF: Handover recommendation for ue-{}: target cell {} (reason={:?}, confidence={})",
            ue_id, best_neighbor.0, reason, confidence
        );

        Ok(Some(recommendation))
    }

    /// Performs NF load analytics for a cell
    ///
    /// Analyzes cell load history and predicts future load levels.
    ///
    /// # Errors
    ///
    /// Returns an error if the cell has no load history.
    pub fn analyze_nf_load(
        &mut self,
        cell_id: i32,
        horizon_steps: usize,
        data_collector: &DataCollector,
        mtlf: &Mtlf,
    ) -> Result<AnalyticsResult, NwdafError> {
        let load_history = data_collector
            .get_cell_load_history(cell_id)
            .ok_or(AnalyticsError::TargetNotFound {
                target: format!("cell-{cell_id}"),
            })?;

        if load_history.len() < 2 {
            return Err(AnalyticsError::InsufficientData {
                required: 2,
                available: load_history.len(),
            }
            .into());
        }

        let load_values: Vec<f32> = load_history.iter().map(|l| l.prb_usage).collect();
        let timestamps: Vec<u64> = load_history.iter().map(|l| l.timestamp_ms).collect();
        let current_load = *load_values.last().unwrap_or(&0.0);
        let current_time = *timestamps.last().unwrap_or(&0);

        // Predict future load using predictor
        let predictor = mtlf
            .trajectory_predictor()
            .or(self.fallback_predictor.as_ref());

        let predicted_load = if let Some(pred) = predictor {
            match pred.predict_load(&load_values, &timestamps, horizon_steps) {
                Ok((predictions, _method)) => predictions,
                Err(e) => {
                    warn!("Load prediction failed: {}, using simple trend", e);
                    simple_load_trend(&load_values, horizon_steps)
                }
            }
        } else {
            simple_load_trend(&load_values, horizon_steps)
        };

        // Estimate time to overload (PRB usage > 0.9)
        let time_to_overload_ms = predicted_load
            .iter()
            .position(|&l| l > 0.9)
            .map(|step| {
                let interval_ms = if timestamps.len() >= 2 {
                    timestamps[timestamps.len() - 1] - timestamps[timestamps.len() - 2]
                } else {
                    100
                };
                (step as u64 + 1) * interval_ms
            });

        let result = AnalyticsResult {
            analytics_id: AnalyticsId::NfLoad,
            target: AnalyticsTarget::Cell { cell_id },
            output_type: AnalyticsOutputType::Predictions,
            timestamp_ms: current_time,
            confidence: (load_history.len() as f32 / 50.0).min(0.9),
            payload: AnalyticsPayload::NfLoad {
                current_load,
                predicted_load,
                time_to_overload_ms,
            },
        };

        self.store_result(result.clone());
        Ok(result)
    }

    /// Retrieves abnormal behaviour analytics
    ///
    /// Returns all detected anomalies for the specified target.
    pub fn analyze_abnormal_behavior(
        &mut self,
        target: &AnalyticsTarget,
    ) -> Result<AnalyticsResult, NwdafError> {
        let anomalies = match target {
            AnalyticsTarget::Ue { ue_id } => {
                let entity_id = format!("ue-{ue_id}");
                self.anomaly_detector
                    .anomalies_for_entity(&entity_id)
                    .into_iter()
                    .cloned()
                    .collect()
            }
            AnalyticsTarget::Cell { cell_id } => {
                let entity_id = format!("cell-{cell_id}");
                self.anomaly_detector
                    .anomalies_for_entity(&entity_id)
                    .into_iter()
                    .cloned()
                    .collect()
            }
            AnalyticsTarget::Any => self
                .anomaly_detector
                .recent_anomalies()
                .iter()
                .cloned()
                .collect(),
            AnalyticsTarget::Slice { .. } => Vec::new(),
        };

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let result = AnalyticsResult {
            analytics_id: AnalyticsId::AbnormalBehavior,
            target: target.clone(),
            output_type: AnalyticsOutputType::Statistics,
            timestamp_ms: now_ms,
            confidence: 0.85,
            payload: AnalyticsPayload::AbnormalBehavior { anomalies },
        };

        self.store_result(result.clone());
        Ok(result)
    }

    /// Performs Service Experience analytics (TS 23.288 6.4)
    ///
    /// Analyzes service-level metrics and computes Mean Opinion Score (MOS)
    /// based on latency, throughput, and packet loss.
    ///
    /// # Errors
    ///
    /// Returns an error if the target has insufficient data.
    pub fn analyze_service_experience(
        &mut self,
        target: &AnalyticsTarget,
        data_collector: &DataCollector,
    ) -> Result<AnalyticsResult, NwdafError> {
        // Extract metrics based on target type
        let (avg_latency_ms, throughput_mbps, packet_loss_rate, success_rate): (f32, f32, f32, f32) = match target {
            AnalyticsTarget::Cell { cell_id } => {
                let load_history = data_collector
                    .get_cell_load_history(*cell_id)
                    .ok_or(AnalyticsError::TargetNotFound {
                        target: format!("cell-{cell_id}"),
                    })?;

                if load_history.len() < 5 {
                    return Err(AnalyticsError::InsufficientData {
                        required: 5,
                        available: load_history.len(),
                    }
                    .into());
                }

                // Derive metrics from cell load
                let avg_throughput = load_history.iter()
                    .map(|l| l.avg_throughput_mbps)
                    .sum::<f32>() / load_history.len() as f32;
                let avg_prb = load_history.iter()
                    .map(|l| l.prb_usage)
                    .sum::<f32>() / load_history.len() as f32;

                // Estimate latency from PRB usage (higher usage -> higher latency)
                let latency = 10.0 + (avg_prb * 40.0);
                // Estimate packet loss from PRB usage
                let loss = if avg_prb > 0.9 { 0.05 } else { 0.001 };
                let success = 1.0 - loss;

                (latency, avg_throughput, loss, success)
            }
            AnalyticsTarget::Ue { ue_id } => {
                let history = data_collector
                    .get_ue_history(*ue_id)
                    .ok_or(AnalyticsError::TargetNotFound {
                        target: format!("ue-{ue_id}"),
                    })?;

                if history.len() < 5 {
                    return Err(AnalyticsError::InsufficientData {
                        required: 5,
                        available: history.len(),
                    }
                    .into());
                }

                // Derive metrics from RSRP/RSRQ
                let avg_rsrp = history.all().iter()
                    .map(|m| m.rsrp)
                    .sum::<f32>() / history.len() as f32;
                let avg_rsrq = history.all().iter()
                    .map(|m| m.rsrq)
                    .sum::<f32>() / history.len() as f32;

                // Better signal -> lower latency, higher throughput
                let latency = 20.0 + ((-avg_rsrp - 60.0).max(0.0) / 2.0);
                let throughput = (100.0 + avg_rsrp).max(10.0);
                let loss = if avg_rsrq < -15.0 { 0.02 } else { 0.001 };
                let success = 1.0 - loss;

                (latency, throughput, loss, success)
            }
            _ => {
                return Err(AnalyticsError::ComputationFailed {
                    reason: "Service experience requires UE or Cell target".to_string(),
                }
                .into());
            }
        };

        // Compute MOS (Mean Opinion Score) based on ITU-T G.107 E-model concepts
        // MOS ranges from 1 (bad) to 5 (excellent)
        let latency_score = (5.0 - (avg_latency_ms / 50.0).min(4.0)).max(1.0);
        let loss_score = (5.0 - (packet_loss_rate * 100.0).min(4.0)).max(1.0);
        let throughput_score = (throughput_mbps / 20.0).min(5.0).max(1.0);
        let mos = (latency_score + loss_score + throughput_score) / 3.0;

        let metrics = ServiceLevelMetrics {
            avg_response_time_ms: avg_latency_ms,
            success_rate,
            availability: 0.99, // Simplified assumption
            data_rate_mbps: throughput_mbps,
        };

        // Identify degraded services (MOS < 3.5)
        let degraded_services = if mos < 3.5 {
            vec!["voice".to_string(), "video".to_string()]
        } else {
            vec![]
        };

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let result = AnalyticsResult {
            analytics_id: AnalyticsId::ServiceExperience,
            target: target.clone(),
            output_type: AnalyticsOutputType::Statistics,
            timestamp_ms: now_ms,
            confidence: 0.85,
            payload: AnalyticsPayload::ServiceExperience {
                mos,
                metrics,
                degraded_services,
            },
        };

        self.store_result(result.clone());
        Ok(result)
    }

    /// Performs User Data Congestion analytics (TS 23.288 6.8)
    ///
    /// Analyzes network congestion levels and predicts future congestion.
    ///
    /// # Errors
    ///
    /// Returns an error if the target has insufficient data.
    pub fn analyze_user_data_congestion(
        &mut self,
        target: &AnalyticsTarget,
        data_collector: &DataCollector,
        mtlf: &Mtlf,
    ) -> Result<AnalyticsResult, NwdafError> {
        match target {
            AnalyticsTarget::Cell { cell_id } => {
                let load_history = data_collector
                    .get_cell_load_history(*cell_id)
                    .ok_or(AnalyticsError::TargetNotFound {
                        target: format!("cell-{cell_id}"),
                    })?;

                if load_history.len() < 10 {
                    return Err(AnalyticsError::InsufficientData {
                        required: 10,
                        available: load_history.len(),
                    }
                    .into());
                }

                // Current congestion based on PRB usage and connected UEs
                let latest = load_history.back().unwrap();
                let congestion_level = (latest.prb_usage * 0.7
                    + (latest.connected_ues as f32 / 100.0).min(1.0) * 0.3)
                    .min(1.0);

                // Predict future congestion using load prediction
                let load_values: Vec<f32> = load_history.iter().map(|l| l.prb_usage).collect();
                let timestamps: Vec<u64> = load_history.iter().map(|l| l.timestamp_ms).collect();
                let current_time = *timestamps.last().unwrap_or(&0);

                let predictor = mtlf
                    .trajectory_predictor()
                    .or(self.fallback_predictor.as_ref());

                let predicted_congestion = if let Some(pred) = predictor {
                    match pred.predict_load(&load_values, &timestamps, 5) {
                        Ok((predictions, _)) => {
                            let interval_ms = if timestamps.len() >= 2 {
                                timestamps[timestamps.len() - 1] - timestamps[timestamps.len() - 2]
                            } else {
                                100
                            };
                            predictions
                                .iter()
                                .enumerate()
                                .map(|(i, &prb)| {
                                    let ts = current_time + ((i + 1) as u64 * interval_ms);
                                    let congestion = prb.min(1.0);
                                    (ts, congestion)
                                })
                                .collect()
                        }
                        Err(_) => vec![],
                    }
                } else {
                    vec![]
                };

                let now_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0);

                let result = AnalyticsResult {
                    analytics_id: AnalyticsId::UserDataCongestion,
                    target: target.clone(),
                    output_type: AnalyticsOutputType::Predictions,
                    timestamp_ms: now_ms,
                    confidence: 0.80,
                    payload: AnalyticsPayload::UserDataCongestion {
                        congestion_level,
                        predicted_congestion,
                        affected_areas: vec![*cell_id],
                    },
                };

                self.store_result(result.clone());
                Ok(result)
            }
            _ => Err(AnalyticsError::ComputationFailed {
                reason: "User data congestion requires Cell target".to_string(),
            }
            .into()),
        }
    }

    /// Performs QoS Sustainability analytics (TS 23.288 6.6)
    ///
    /// Predicts whether current QoS levels can be sustained given network conditions.
    ///
    /// # Errors
    ///
    /// Returns an error if the target has insufficient data.
    pub fn analyze_qos_sustainability(
        &mut self,
        target: &AnalyticsTarget,
        data_collector: &DataCollector,
        mtlf: &Mtlf,
    ) -> Result<AnalyticsResult, NwdafError> {
        match target {
            AnalyticsTarget::Cell { cell_id } => {
                let load_history = data_collector
                    .get_cell_load_history(*cell_id)
                    .ok_or(AnalyticsError::TargetNotFound {
                        target: format!("cell-{cell_id}"),
                    })?;

                if load_history.len() < 10 {
                    return Err(AnalyticsError::InsufficientData {
                        required: 10,
                        available: load_history.len(),
                    }
                    .into());
                }

                let latest = load_history.back().unwrap();

                // Current QoS metrics
                let current_qos = QosMetrics {
                    avg_latency_ms: 10.0 + (latest.prb_usage * 40.0),
                    packet_loss_rate: if latest.prb_usage > 0.9 { 0.05 } else { 0.001 },
                    throughput_mbps: latest.avg_throughput_mbps,
                };

                // Check sustainability based on load trend
                let load_values: Vec<f32> = load_history.iter().map(|l| l.prb_usage).collect();
                let timestamps: Vec<u64> = load_history.iter().map(|l| l.timestamp_ms).collect();

                let predictor = mtlf
                    .trajectory_predictor()
                    .or(self.fallback_predictor.as_ref());

                let sustainable = if let Some(pred) = predictor {
                    match pred.predict_load(&load_values, &timestamps, 10) {
                        Ok((predictions, _)) => {
                            // QoS is sustainable if predicted load stays below 0.85
                            predictions.iter().all(|&load| load < 0.85)
                        }
                        Err(_) => latest.prb_usage < 0.75,
                    }
                } else {
                    latest.prb_usage < 0.75
                };

                // Estimate remaining time at current QoS if not sustainable
                let remaining_time_ms = if !sustainable {
                    Some(300_000) // 5 minutes estimate
                } else {
                    None
                };

                let now_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0);

                let result = AnalyticsResult {
                    analytics_id: AnalyticsId::QosSustainability,
                    target: target.clone(),
                    output_type: AnalyticsOutputType::Predictions,
                    timestamp_ms: now_ms,
                    confidence: 0.82,
                    payload: AnalyticsPayload::QosSustainability {
                        sustainable,
                        remaining_time_ms,
                        current_qos,
                    },
                };

                self.store_result(result.clone());
                Ok(result)
            }
            AnalyticsTarget::Ue { ue_id } => {
                let history = data_collector
                    .get_ue_history(*ue_id)
                    .ok_or(AnalyticsError::TargetNotFound {
                        target: format!("ue-{ue_id}"),
                    })?;

                if history.len() < 10 {
                    return Err(AnalyticsError::InsufficientData {
                        required: 10,
                        available: history.len(),
                    }
                    .into());
                }

                let latest = history.latest().unwrap();
                let avg_rsrp = history.all().iter()
                    .map(|m| m.rsrp)
                    .sum::<f32>() / history.len() as f32;

                let current_qos = QosMetrics {
                    avg_latency_ms: 20.0 + ((-avg_rsrp - 60.0).max(0.0) / 2.0),
                    packet_loss_rate: if latest.rsrq < -15.0 { 0.02 } else { 0.001 },
                    throughput_mbps: (100.0 + avg_rsrp).max(10.0),
                };

                // QoS sustainable if signal quality is good
                let sustainable = latest.rsrp > -90.0 && latest.rsrq > -12.0;
                let remaining_time_ms = if !sustainable {
                    Some(180_000) // 3 minutes
                } else {
                    None
                };

                let now_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0);

                let result = AnalyticsResult {
                    analytics_id: AnalyticsId::QosSustainability,
                    target: target.clone(),
                    output_type: AnalyticsOutputType::Predictions,
                    timestamp_ms: now_ms,
                    confidence: 0.80,
                    payload: AnalyticsPayload::QosSustainability {
                        sustainable,
                        remaining_time_ms,
                        current_qos,
                    },
                };

                self.store_result(result.clone());
                Ok(result)
            }
            _ => Err(AnalyticsError::ComputationFailed {
                reason: "QoS sustainability requires UE or Cell target".to_string(),
            }
            .into()),
        }
    }

    /// Performs Energy Efficiency analytics (TS 23.288 6.14, Rel-19 / IMT-2030)
    ///
    /// Evaluates network energy efficiency based on cell load, throughput,
    /// and estimated power consumption using 3GPP ETSI ES 203 228 energy model.
    pub fn analyze_energy_efficiency(
        &mut self,
        target: &AnalyticsTarget,
        data_collector: &DataCollector,
    ) -> Result<AnalyticsResult, NwdafError> {
        match target {
            AnalyticsTarget::Cell { cell_id } => {
                let load_history = data_collector
                    .get_cell_load_history(*cell_id)
                    .ok_or(AnalyticsError::TargetNotFound {
                        target: format!("cell-{cell_id}"),
                    })?;

                if load_history.len() < 5 {
                    return Err(AnalyticsError::InsufficientData {
                        required: 5,
                        available: load_history.len(),
                    }
                    .into());
                }

                // ETSI ES 203 228 power model: P = P_idle + (P_max - P_idle) * load
                let p_idle: f32 = 50.0; // watts, idle power
                let p_max: f32 = 200.0; // watts, max power
                let avg_load = load_history.iter().map(|l| l.prb_usage).sum::<f32>()
                    / load_history.len() as f32;
                let avg_throughput = load_history.iter().map(|l| l.avg_throughput_mbps).sum::<f32>()
                    / load_history.len() as f32;

                let power_consumption = p_idle + (p_max - p_idle) * avg_load;
                // Energy per bit: power / throughput (J/bit = W / (Mbps * 1e6))
                let energy_per_bit = if avg_throughput > 0.01 {
                    power_consumption / (avg_throughput * 1_000_000.0)
                } else {
                    f32::MAX
                };
                // Efficiency score: inverse relationship with energy per bit (normalized)
                let efficiency_score = (1.0 - (energy_per_bit * 1e7).min(1.0)).max(0.0);

                let mut recommendations = Vec::new();
                if avg_load < 0.2 {
                    recommendations.push("Consider cell sleep mode for low-traffic periods".to_string());
                }
                if avg_load > 0.8 {
                    recommendations.push("High load detected; consider load balancing across cells".to_string());
                }
                if energy_per_bit > 1e-7 {
                    recommendations.push("Enable MIMO sleep and carrier shutdown features".to_string());
                }

                let now_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0);

                let result = AnalyticsResult {
                    analytics_id: AnalyticsId::EnergyEfficiency,
                    target: target.clone(),
                    output_type: AnalyticsOutputType::Statistics,
                    timestamp_ms: now_ms,
                    confidence: 0.78,
                    payload: AnalyticsPayload::EnergyEfficiency {
                        efficiency_score,
                        power_consumption_watts: power_consumption,
                        energy_per_bit,
                        recommendations,
                    },
                };

                self.store_result(result.clone());
                Ok(result)
            }
            AnalyticsTarget::Any => {
                // Aggregate across all known cells
                let cell_ids: Vec<i32> = data_collector.known_cell_ids();
                if cell_ids.is_empty() {
                    return Err(AnalyticsError::TargetNotFound {
                        target: "any cells".to_string(),
                    }
                    .into());
                }

                let mut total_power = 0.0f32;
                let mut total_throughput = 0.0f32;
                let mut cell_count = 0u32;

                for cid in &cell_ids {
                    if let Some(history) = data_collector.get_cell_load_history(*cid) {
                        if history.len() >= 5 {
                            let avg_load = history.iter().map(|l| l.prb_usage).sum::<f32>()
                                / history.len() as f32;
                            let avg_tp = history.iter().map(|l| l.avg_throughput_mbps).sum::<f32>()
                                / history.len() as f32;
                            total_power += 50.0 + 150.0 * avg_load;
                            total_throughput += avg_tp;
                            cell_count += 1;
                        }
                    }
                }

                if cell_count == 0 {
                    return Err(AnalyticsError::InsufficientData { required: 5, available: 0 }.into());
                }

                let energy_per_bit = if total_throughput > 0.01 {
                    total_power / (total_throughput * 1_000_000.0)
                } else {
                    f32::MAX
                };
                let efficiency_score = (1.0 - (energy_per_bit * 1e7).min(1.0)).max(0.0);

                let now_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0);

                let result = AnalyticsResult {
                    analytics_id: AnalyticsId::EnergyEfficiency,
                    target: target.clone(),
                    output_type: AnalyticsOutputType::Statistics,
                    timestamp_ms: now_ms,
                    confidence: 0.75,
                    payload: AnalyticsPayload::EnergyEfficiency {
                        efficiency_score,
                        power_consumption_watts: total_power,
                        energy_per_bit,
                        recommendations: vec![
                            format!("Network-wide energy analysis across {} cells", cell_count),
                        ],
                    },
                };

                self.store_result(result.clone());
                Ok(result)
            }
            _ => Err(AnalyticsError::ComputationFailed {
                reason: "Energy efficiency requires Cell or Any target".to_string(),
            }
            .into()),
        }
    }

    /// Performs Network Slice Optimization analytics (TS 23.288 6.12, Rel-19)
    ///
    /// Analyzes per-slice resource utilization and SLA compliance, providing
    /// reallocation recommendations when slices are over/under-provisioned.
    pub fn analyze_slice_optimization(
        &mut self,
        target: &AnalyticsTarget,
        data_collector: &DataCollector,
    ) -> Result<AnalyticsResult, NwdafError> {
        // Slice analytics works on Cell or Any target
        let cell_ids: Vec<i32> = match target {
            AnalyticsTarget::Cell { cell_id } => vec![*cell_id],
            AnalyticsTarget::Slice { ref snssai } => {
                let ids = data_collector.known_cell_ids();
                if ids.is_empty() {
                    return Err(AnalyticsError::TargetNotFound {
                        target: format!("slice-{}", snssai),
                    }
                    .into());
                }
                ids
            }
            AnalyticsTarget::Any => {
                let ids = data_collector.known_cell_ids();
                if ids.is_empty() {
                    return Err(AnalyticsError::TargetNotFound {
                        target: "any cells for slice analytics".to_string(),
                    }
                    .into());
                }
                ids
            }
            _ => {
                return Err(AnalyticsError::ComputationFailed {
                    reason: "Slice optimization requires Cell, Slice, or Any target".to_string(),
                }
                .into());
            }
        };

        // Standard 5G slice types (SST values per TS 23.501):
        // 1 = eMBB, 2 = URLLC, 3 = MIoT, 4 = V2X
        let slice_types = [1_i32, 2, 3, 4];
        let mut slice_utilization = Vec::new();
        let mut sla_compliance = Vec::new();
        let mut recommendations = Vec::new();

        // Gather load data across cells
        let mut total_load = 0.0f32;
        let mut cell_count = 0u32;

        for cid in &cell_ids {
            if let Some(history) = data_collector.get_cell_load_history(*cid) {
                if history.len() >= 3 {
                    let avg = history.iter().map(|l| l.prb_usage).sum::<f32>()
                        / history.len() as f32;
                    total_load += avg;
                    cell_count += 1;
                }
            }
        }

        let network_load = if cell_count > 0 {
            total_load / cell_count as f32
        } else {
            0.5 // default assumption
        };

        // Model per-slice utilization based on typical traffic distribution
        for &sst in &slice_types {
            let (utilization, sla_target) = match sst {
                1 => {
                    // eMBB: consumes ~60% of resources, SLA: throughput > 10 Mbps
                    let util = (network_load * 0.6).min(1.0);
                    let sla_ok = util < 0.9; // overloaded eMBB fails SLA
                    (util, sla_ok)
                }
                2 => {
                    // URLLC: consumes ~15% of resources, SLA: latency < 1ms
                    let util = (network_load * 0.15).min(1.0);
                    let sla_ok = util < 0.7; // URLLC very sensitive to load
                    (util, sla_ok)
                }
                3 => {
                    // MIoT: consumes ~20% of resources, SLA: 99% delivery
                    let util = (network_load * 0.20).min(1.0);
                    let sla_ok = util < 0.95;
                    (util, sla_ok)
                }
                4 => {
                    // V2X: consumes ~5% of resources, SLA: latency < 10ms
                    let util = (network_load * 0.05).min(1.0);
                    let sla_ok = util < 0.8;
                    (util, sla_ok)
                }
                _ => (0.0, true),
            };

            slice_utilization.push((sst, utilization));
            sla_compliance.push((sst, sla_target));

            if !sla_target {
                let name = match sst {
                    1 => "eMBB",
                    2 => "URLLC",
                    3 => "MIoT",
                    4 => "V2X",
                    _ => "Unknown",
                };
                recommendations.push(format!(
                    "Slice SST={sst} ({name}) SLA at risk: utilization={utilization:.1}%, consider increasing allocation"
                ));
            }
        }

        // Cross-slice recommendations
        let embb_util = slice_utilization.iter().find(|(s, _)| *s == 1).map(|(_, u)| *u).unwrap_or(0.0);
        let urllc_util = slice_utilization.iter().find(|(s, _)| *s == 2).map(|(_, u)| *u).unwrap_or(0.0);
        if embb_util > 0.85 && urllc_util < 0.3 {
            recommendations.push("Reallocate spare URLLC resources to eMBB to improve throughput".to_string());
        }

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let result = AnalyticsResult {
            analytics_id: AnalyticsId::SliceOptimization,
            target: target.clone(),
            output_type: AnalyticsOutputType::Statistics,
            timestamp_ms: now_ms,
            confidence: 0.80,
            payload: AnalyticsPayload::SliceOptimization {
                slice_utilization,
                sla_compliance,
                reallocation_recommendations: recommendations,
            },
        };

        self.store_result(result.clone());
        Ok(result)
    }

    /// Returns recent analytics results
    pub fn recent_results(&self) -> &VecDeque<AnalyticsResult> {
        &self.recent_results
    }

    /// Returns recent results filtered by analytics ID
    pub fn results_for_analytics(&self, analytics_id: AnalyticsId) -> Vec<&AnalyticsResult> {
        self.recent_results
            .iter()
            .filter(|r| r.analytics_id == analytics_id)
            .collect()
    }

    /// Stores a result in the bounded buffer
    fn store_result(&mut self, result: AnalyticsResult) {
        if self.recent_results.len() >= self.max_results {
            self.recent_results.pop_front();
        }
        self.recent_results.push_back(result);
    }
}

impl Default for Anlf {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple linear trend extrapolation for load values
fn simple_load_trend(values: &[f32], steps: usize) -> Vec<f32> {
    if values.len() < 2 {
        return vec![values.last().copied().unwrap_or(0.0); steps];
    }
    let last = values[values.len() - 1];
    let prev = values[values.len() - 2];
    let slope = last - prev;
    (1..=steps)
        .map(|i| (last + slope * i as f32).clamp(0.0, 1.0))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vector3;

    #[test]
    fn test_anlf_creation() {
        let anlf = Anlf::new();
        assert!(!anlf.supported_analytics().is_empty());
        assert!(anlf.supports_analytics(AnalyticsId::UeMobility));
        assert!(anlf.supports_analytics(AnalyticsId::NfLoad));
        assert!(anlf.supports_analytics(AnalyticsId::AbnormalBehavior));
    }

    #[test]
    fn test_process_measurement_no_anomaly() {
        let mut anlf = Anlf::new();

        // Build baseline
        for i in 0..20 {
            let meas = UeMeasurement {
                ue_id: 1,
                rsrp: -80.0,
                rsrq: -10.0,
                sinr: Some(15.0),
                position: Vector3::new(i as f64, 0.0, 0.0),
                velocity: None,
                serving_cell_id: 1,
                timestamp_ms: i as u64 * 100,
            };
            let _ = anlf.process_measurement(&meas);
        }

        // Normal measurement should not trigger anomaly
        let meas = UeMeasurement {
            ue_id: 1,
            rsrp: -80.0,
            rsrq: -10.0,
            sinr: Some(15.0),
            position: Vector3::new(20.0, 0.0, 0.0),
            velocity: None,
            serving_cell_id: 1,
            timestamp_ms: 2000,
        };
        let anomalies = anlf.process_measurement(&meas);
        assert!(anomalies.is_empty());
    }

    #[test]
    fn test_process_measurement_with_anomaly() {
        let mut anlf = Anlf::new();

        // Build baseline
        for i in 0..30 {
            let meas = UeMeasurement {
                ue_id: 1,
                rsrp: -80.0,
                rsrq: -10.0,
                sinr: Some(15.0),
                position: Vector3::new(i as f64, 0.0, 0.0),
                velocity: None,
                serving_cell_id: 1,
                timestamp_ms: i as u64 * 100,
            };
            let _ = anlf.process_measurement(&meas);
        }

        // Anomalous RSRP spike
        let meas = UeMeasurement {
            ue_id: 1,
            rsrp: -30.0, // huge spike from baseline of -80
            rsrq: -10.0,
            sinr: Some(15.0),
            position: Vector3::new(30.0, 0.0, 0.0),
            velocity: None,
            serving_cell_id: 1,
            timestamp_ms: 3000,
        };
        let anomalies = anlf.process_measurement(&meas);
        assert!(!anomalies.is_empty(), "Should detect RSRP anomaly");
    }

    #[test]
    fn test_analyze_ue_mobility() {
        let mut anlf = Anlf::new();
        let mut collector = DataCollector::new(100);
        let mtlf = Mtlf::new();

        // Add measurement history
        for i in 0..10 {
            collector.report_ue_measurement(UeMeasurement {
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

        let result = anlf.analyze_ue_mobility(1, 1000, &collector, &mtlf);
        assert!(result.is_ok());

        let analytics = result.expect("should succeed");
        assert_eq!(analytics.analytics_id, AnalyticsId::UeMobility);
        if let AnalyticsPayload::UeMobility {
            trajectory,
            handover_recommendation: _,
        } = &analytics.payload
        {
            assert!(trajectory.is_some());
            let traj = trajectory.as_ref().expect("should have trajectory");
            assert!(!traj.waypoints.is_empty());
        } else {
            panic!("Expected UeMobility payload");
        }
    }

    #[test]
    fn test_analyze_ue_mobility_not_found() {
        let mut anlf = Anlf::new();
        let collector = DataCollector::new(100);
        let mtlf = Mtlf::new();

        let result = anlf.analyze_ue_mobility(999, 1000, &collector, &mtlf);
        assert!(result.is_err());
    }

    #[test]
    fn test_analyze_nf_load() {
        let mut anlf = Anlf::new();
        let mut collector = DataCollector::new(100);
        let mtlf = Mtlf::new();

        // Add cell load history
        for i in 0..20 {
            collector.report_cell_load(CellLoad {
                cell_id: 1,
                prb_usage: 0.3 + i as f32 * 0.02,
                connected_ues: 10,
                avg_throughput_mbps: 150.0,
                timestamp_ms: i as u64 * 100,
            });
        }

        let result = anlf.analyze_nf_load(1, 5, &collector, &mtlf);
        assert!(result.is_ok());

        let analytics = result.expect("should succeed");
        assert_eq!(analytics.analytics_id, AnalyticsId::NfLoad);
        if let AnalyticsPayload::NfLoad {
            current_load,
            predicted_load,
            ..
        } = &analytics.payload
        {
            assert!(*current_load > 0.0);
            assert_eq!(predicted_load.len(), 5);
        } else {
            panic!("Expected NfLoad payload");
        }
    }

    #[test]
    fn test_analyze_abnormal_behavior() {
        let mut anlf = Anlf::new();

        // Build baseline and inject anomaly
        for i in 0..30 {
            let meas = UeMeasurement {
                ue_id: 1,
                rsrp: -80.0,
                rsrq: -10.0,
                sinr: Some(15.0),
                position: Vector3::new(0.0, 0.0, 0.0),
                velocity: None,
                serving_cell_id: 1,
                timestamp_ms: i as u64 * 100,
            };
            anlf.process_measurement(&meas);
        }

        // Inject anomaly
        let meas = UeMeasurement {
            ue_id: 1,
            rsrp: -30.0,
            rsrq: -10.0,
            sinr: Some(15.0),
            position: Vector3::new(0.0, 0.0, 0.0),
            velocity: None,
            serving_cell_id: 1,
            timestamp_ms: 3000,
        };
        anlf.process_measurement(&meas);

        let result = anlf.analyze_abnormal_behavior(&AnalyticsTarget::Ue { ue_id: 1 });
        assert!(result.is_ok());

        let analytics = result.expect("should succeed");
        if let AnalyticsPayload::AbnormalBehavior { anomalies } = &analytics.payload {
            assert!(!anomalies.is_empty());
        } else {
            panic!("Expected AbnormalBehavior payload");
        }
    }

    #[test]
    fn test_recommend_handover() {
        let mut anlf = Anlf::new();
        let mut collector = DataCollector::new(100);
        let mtlf = Mtlf::new();

        // Add measurements for UE moving toward cell 2
        for i in 0..10 {
            collector.report_ue_measurement(UeMeasurement {
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

        // Neighbor cell with better signal
        let neighbors = vec![(2, -70.0f32)]; // 10dB better than serving cell

        let result = anlf.recommend_handover(1, &neighbors, &collector, &mtlf);
        assert!(result.is_ok());

        let rec = result.expect("should succeed");
        assert!(rec.is_some());
        let ho = rec.expect("should have recommendation");
        assert_eq!(ho.target_cell_id, 2);
    }

    #[test]
    fn test_simple_load_trend() {
        let values = vec![0.3, 0.4, 0.5];
        let predicted = simple_load_trend(&values, 3);
        assert_eq!(predicted.len(), 3);
        assert!((predicted[0] - 0.6).abs() < 0.01);
        assert!((predicted[1] - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_analyze_service_experience_cell() {
        let mut anlf = Anlf::new();
        let mut collector = DataCollector::new(100);

        // Add cell load history (need >= 5 entries)
        for i in 0..20 {
            collector.report_cell_load(CellLoad {
                cell_id: 1,
                prb_usage: 0.4 + i as f32 * 0.01,
                connected_ues: 10 + i as u32,
                avg_throughput_mbps: 120.0,
                timestamp_ms: i as u64 * 100,
            });
        }

        let result = anlf.analyze_service_experience(
            &AnalyticsTarget::Cell { cell_id: 1 },
            &collector,
        );
        assert!(result.is_ok());

        let analytics = result.expect("should succeed");
        assert_eq!(analytics.analytics_id, AnalyticsId::ServiceExperience);
        if let AnalyticsPayload::ServiceExperience {
            mos,
            metrics,
            degraded_services: _,
        } = &analytics.payload
        {
            assert!(*mos >= 1.0 && *mos <= 5.0, "MOS={mos} out of range");
            assert!(metrics.avg_response_time_ms > 0.0);
            assert!(metrics.success_rate > 0.0 && metrics.success_rate <= 1.0);
            assert!(metrics.data_rate_mbps > 0.0);
        } else {
            panic!("Expected ServiceExperience payload");
        }
    }

    #[test]
    fn test_analyze_service_experience_ue() {
        let mut anlf = Anlf::new();
        let mut collector = DataCollector::new(100);

        // Add UE measurement history (need >= 5 entries)
        for i in 0..10 {
            collector.report_ue_measurement(UeMeasurement {
                ue_id: 1,
                rsrp: -75.0,
                rsrq: -9.0,
                sinr: Some(18.0),
                position: Vector3::new(i as f64, 0.0, 0.0),
                velocity: None,
                serving_cell_id: 1,
                timestamp_ms: i as u64 * 100,
            });
        }

        let result = anlf.analyze_service_experience(
            &AnalyticsTarget::Ue { ue_id: 1 },
            &collector,
        );
        assert!(result.is_ok());

        let analytics = result.expect("should succeed");
        if let AnalyticsPayload::ServiceExperience { mos, metrics, .. } = &analytics.payload {
            assert!(*mos >= 1.0 && *mos <= 5.0, "MOS={mos} out of range");
            // Good signal should give decent throughput
            assert!(metrics.data_rate_mbps > 0.0);
        } else {
            panic!("Expected ServiceExperience payload");
        }
    }

    #[test]
    fn test_analyze_service_experience_insufficient_data() {
        let mut anlf = Anlf::new();
        let mut collector = DataCollector::new(100);

        // Only add 2 entries (need >= 5)
        for i in 0..2 {
            collector.report_cell_load(CellLoad {
                cell_id: 1,
                prb_usage: 0.5,
                connected_ues: 10,
                avg_throughput_mbps: 100.0,
                timestamp_ms: i as u64 * 100,
            });
        }

        let result = anlf.analyze_service_experience(
            &AnalyticsTarget::Cell { cell_id: 1 },
            &collector,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_analyze_service_experience_degraded() {
        let mut anlf = Anlf::new();
        let mut collector = DataCollector::new(100);

        // High PRB usage -> high latency -> low MOS -> degraded services
        for i in 0..10 {
            collector.report_cell_load(CellLoad {
                cell_id: 1,
                prb_usage: 0.95,
                connected_ues: 80,
                avg_throughput_mbps: 5.0, // very low throughput
                timestamp_ms: i as u64 * 100,
            });
        }

        let result = anlf.analyze_service_experience(
            &AnalyticsTarget::Cell { cell_id: 1 },
            &collector,
        );
        assert!(result.is_ok());

        let analytics = result.expect("should succeed");
        if let AnalyticsPayload::ServiceExperience {
            mos,
            degraded_services,
            ..
        } = &analytics.payload
        {
            // High load -> low MOS
            assert!(*mos < 3.5, "MOS={mos} should be degraded under high load");
            assert!(!degraded_services.is_empty(), "Should report degraded services");
        } else {
            panic!("Expected ServiceExperience payload");
        }
    }

    #[test]
    fn test_analyze_user_data_congestion() {
        let mut anlf = Anlf::new();
        let mut collector = DataCollector::new(100);
        let mtlf = Mtlf::new();

        // Add cell load history (need >= 10 entries)
        for i in 0..20 {
            collector.report_cell_load(CellLoad {
                cell_id: 1,
                prb_usage: 0.5 + i as f32 * 0.02,
                connected_ues: 20 + i as u32 * 2,
                avg_throughput_mbps: 100.0 - i as f32 * 2.0,
                timestamp_ms: i as u64 * 100,
            });
        }

        let result = anlf.analyze_user_data_congestion(
            &AnalyticsTarget::Cell { cell_id: 1 },
            &collector,
            &mtlf,
        );
        assert!(result.is_ok());

        let analytics = result.expect("should succeed");
        assert_eq!(analytics.analytics_id, AnalyticsId::UserDataCongestion);
        if let AnalyticsPayload::UserDataCongestion {
            congestion_level,
            affected_areas,
            ..
        } = &analytics.payload
        {
            assert!(
                *congestion_level >= 0.0 && *congestion_level <= 1.0,
                "congestion_level={congestion_level} out of range"
            );
            assert!(affected_areas.contains(&1), "Should include cell_id 1");
        } else {
            panic!("Expected UserDataCongestion payload");
        }
    }

    #[test]
    fn test_analyze_user_data_congestion_insufficient_data() {
        let mut anlf = Anlf::new();
        let mut collector = DataCollector::new(100);
        let mtlf = Mtlf::new();

        // Only 5 entries (need >= 10)
        for i in 0..5 {
            collector.report_cell_load(CellLoad {
                cell_id: 1,
                prb_usage: 0.5,
                connected_ues: 10,
                avg_throughput_mbps: 100.0,
                timestamp_ms: i as u64 * 100,
            });
        }

        let result = anlf.analyze_user_data_congestion(
            &AnalyticsTarget::Cell { cell_id: 1 },
            &collector,
            &mtlf,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_analyze_user_data_congestion_wrong_target() {
        let mut anlf = Anlf::new();
        let collector = DataCollector::new(100);
        let mtlf = Mtlf::new();

        // UserDataCongestion requires Cell target
        let result = anlf.analyze_user_data_congestion(
            &AnalyticsTarget::Ue { ue_id: 1 },
            &collector,
            &mtlf,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_analyze_qos_sustainability_cell() {
        let mut anlf = Anlf::new();
        let mut collector = DataCollector::new(100);
        let mtlf = Mtlf::new();

        // Add cell load history with moderate load (need >= 10)
        for i in 0..20 {
            collector.report_cell_load(CellLoad {
                cell_id: 1,
                prb_usage: 0.3 + i as f32 * 0.01,
                connected_ues: 15,
                avg_throughput_mbps: 150.0,
                timestamp_ms: i as u64 * 100,
            });
        }

        let result = anlf.analyze_qos_sustainability(
            &AnalyticsTarget::Cell { cell_id: 1 },
            &collector,
            &mtlf,
        );
        assert!(result.is_ok());

        let analytics = result.expect("should succeed");
        assert_eq!(analytics.analytics_id, AnalyticsId::QosSustainability);
        if let AnalyticsPayload::QosSustainability {
            sustainable,
            current_qos,
            ..
        } = &analytics.payload
        {
            // Low load -> should be sustainable
            assert!(*sustainable, "Low load should be sustainable");
            assert!(current_qos.avg_latency_ms > 0.0);
            assert!(current_qos.throughput_mbps > 0.0);
        } else {
            panic!("Expected QosSustainability payload");
        }
    }

    #[test]
    fn test_analyze_qos_sustainability_cell_unsustainable() {
        let mut anlf = Anlf::new();
        let mut collector = DataCollector::new(100);
        let mtlf = Mtlf::new();

        // Very high load -> unsustainable
        for i in 0..20 {
            collector.report_cell_load(CellLoad {
                cell_id: 1,
                prb_usage: 0.85 + i as f32 * 0.005,
                connected_ues: 80,
                avg_throughput_mbps: 30.0,
                timestamp_ms: i as u64 * 100,
            });
        }

        let result = anlf.analyze_qos_sustainability(
            &AnalyticsTarget::Cell { cell_id: 1 },
            &collector,
            &mtlf,
        );
        assert!(result.is_ok());

        let analytics = result.expect("should succeed");
        if let AnalyticsPayload::QosSustainability {
            sustainable,
            remaining_time_ms,
            current_qos,
        } = &analytics.payload
        {
            assert!(!sustainable, "High load should be unsustainable");
            assert!(remaining_time_ms.is_some(), "Should estimate remaining time");
            assert!(current_qos.packet_loss_rate > 0.0);
        } else {
            panic!("Expected QosSustainability payload");
        }
    }

    #[test]
    fn test_analyze_qos_sustainability_ue() {
        let mut anlf = Anlf::new();
        let mut collector = DataCollector::new(100);
        let mtlf = Mtlf::new();

        // Good signal UE (need >= 10)
        for i in 0..15 {
            collector.report_ue_measurement(UeMeasurement {
                ue_id: 1,
                rsrp: -70.0,
                rsrq: -8.0,
                sinr: Some(20.0),
                position: Vector3::new(i as f64, 0.0, 0.0),
                velocity: None,
                serving_cell_id: 1,
                timestamp_ms: i as u64 * 100,
            });
        }

        let result = anlf.analyze_qos_sustainability(
            &AnalyticsTarget::Ue { ue_id: 1 },
            &collector,
            &mtlf,
        );
        assert!(result.is_ok());

        let analytics = result.expect("should succeed");
        if let AnalyticsPayload::QosSustainability {
            sustainable,
            current_qos,
            ..
        } = &analytics.payload
        {
            assert!(*sustainable, "Good signal should be sustainable");
            assert!(current_qos.throughput_mbps > 0.0);
        } else {
            panic!("Expected QosSustainability payload");
        }
    }

    #[test]
    fn test_analyze_qos_sustainability_ue_poor_signal() {
        let mut anlf = Anlf::new();
        let mut collector = DataCollector::new(100);
        let mtlf = Mtlf::new();

        // Poor signal UE
        for i in 0..15 {
            collector.report_ue_measurement(UeMeasurement {
                ue_id: 1,
                rsrp: -110.0,
                rsrq: -18.0,
                sinr: Some(2.0),
                position: Vector3::new(i as f64, 0.0, 0.0),
                velocity: None,
                serving_cell_id: 1,
                timestamp_ms: i as u64 * 100,
            });
        }

        let result = anlf.analyze_qos_sustainability(
            &AnalyticsTarget::Ue { ue_id: 1 },
            &collector,
            &mtlf,
        );
        assert!(result.is_ok());

        let analytics = result.expect("should succeed");
        if let AnalyticsPayload::QosSustainability {
            sustainable,
            remaining_time_ms,
            ..
        } = &analytics.payload
        {
            assert!(!sustainable, "Poor signal should be unsustainable");
            assert!(remaining_time_ms.is_some());
        } else {
            panic!("Expected QosSustainability payload");
        }
    }

    #[test]
    fn test_analyze_qos_sustainability_insufficient_data() {
        let mut anlf = Anlf::new();
        let mut collector = DataCollector::new(100);
        let mtlf = Mtlf::new();

        // Only 5 entries (need >= 10)
        for i in 0..5 {
            collector.report_cell_load(CellLoad {
                cell_id: 1,
                prb_usage: 0.5,
                connected_ues: 10,
                avg_throughput_mbps: 100.0,
                timestamp_ms: i as u64 * 100,
            });
        }

        let result = anlf.analyze_qos_sustainability(
            &AnalyticsTarget::Cell { cell_id: 1 },
            &collector,
            &mtlf,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_results_storage() {
        let mut anlf = Anlf::new();
        let mut collector = DataCollector::new(100);
        let mtlf = Mtlf::new();

        for i in 0..5 {
            collector.report_ue_measurement(UeMeasurement {
                ue_id: 1,
                rsrp: -80.0,
                rsrq: -10.0,
                sinr: None,
                position: Vector3::new(i as f64 * 10.0, 0.0, 0.0),
                velocity: None,
                serving_cell_id: 1,
                timestamp_ms: i as u64 * 100,
            });
        }

        let _ = anlf.analyze_ue_mobility(1, 1000, &collector, &mtlf);
        assert_eq!(anlf.recent_results().len(), 1);
        assert_eq!(
            anlf.results_for_analytics(AnalyticsId::UeMobility).len(),
            1
        );
    }
}
