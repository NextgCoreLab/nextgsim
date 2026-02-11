//! AI/ML for NR Air Interface (3GPP TR 38.843 / TS 38.843)
//!
//! Implements ML models for:
//! - CSI (Channel State Information) prediction
//! - Beam management (beam selection, beam tracking)
//! - ML-based positioning enhancement
//! - Model lifecycle management at gNB

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// AI/ML NR model domain (TS 38.843 use cases)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NrModelDomain {
    /// CSI feedback compression/prediction
    CsiPrediction,
    /// Beam management (selection, tracking, failure prediction)
    BeamManagement,
    /// ML-enhanced positioning
    Positioning,
    /// Auto-encoder for CSI compression
    CsiCompression,
}

impl std::fmt::Display for NrModelDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NrModelDomain::CsiPrediction => write!(f, "CSI-Prediction"),
            NrModelDomain::BeamManagement => write!(f, "Beam-Management"),
            NrModelDomain::Positioning => write!(f, "ML-Positioning"),
            NrModelDomain::CsiCompression => write!(f, "CSI-Compression"),
        }
    }
}

/// Model lifecycle state (TS 38.843 Section 4.1)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelLifecycleState {
    /// Model registered but not yet loaded
    Registered,
    /// Model loaded into memory, ready for warmup
    Loaded,
    /// Model warming up (initial inferences)
    WarmingUp,
    /// Model active and serving inferences
    Active,
    /// Model performance degraded, needs retraining
    Degraded,
    /// Model deactivated (still in memory)
    Deactivated,
    /// Model released from memory
    Released,
}

/// CSI prediction model (TS 38.843 Use Case 1)
///
/// Predicts future CSI based on historical channel measurements.
/// Reduces CSI-RS overhead by predicting CSI between measurement occasions.
#[derive(Debug)]
pub struct CsiPredictionModel {
    /// Model identifier
    pub model_id: String,
    /// Prediction horizon in slots
    pub prediction_horizon_slots: u32,
    /// Number of antenna ports
    pub num_antenna_ports: u16,
    /// Number of subbands for CQI/PMI
    pub num_subbands: u16,
    /// Compression ratio (for CSI feedback compression)
    pub compression_ratio: f32,
    /// Current prediction accuracy (NMSE in dB)
    pub accuracy_nmse_db: f64,
    /// Lifecycle state
    pub state: ModelLifecycleState,
    /// Total inferences performed
    pub inference_count: u64,
}

impl CsiPredictionModel {
    pub fn new(model_id: &str, num_antenna_ports: u16, num_subbands: u16) -> Self {
        Self {
            model_id: model_id.to_string(),
            prediction_horizon_slots: 4,
            num_antenna_ports,
            num_subbands,
            compression_ratio: 4.0,
            accuracy_nmse_db: -15.0, // Typical good accuracy
            state: ModelLifecycleState::Registered,
            inference_count: 0,
        }
    }

    /// Predict CSI for future slots given historical measurements.
    /// Returns predicted rank indicator, PMI, and CQI per subband.
    pub fn predict(&mut self, historical_csi: &[CsiMeasurement]) -> CsiPrediction {
        self.inference_count += 1;

        // Simulate CSI prediction using linear extrapolation of channel quality
        let latest = historical_csi.last();
        let prev = if historical_csi.len() >= 2 {
            historical_csi.get(historical_csi.len() - 2)
        } else {
            None
        };

        let (predicted_cqi, predicted_ri, predicted_pmi) = match (latest, prev) {
            (Some(l), Some(p)) => {
                // Linear extrapolation
                let cqi_trend = l.wideband_cqi as f64 - p.wideband_cqi as f64;
                let predicted_cqi = (l.wideband_cqi as f64 + cqi_trend * 0.5)
                    .clamp(1.0, 15.0) as u8;
                let predicted_ri = l.rank_indicator;
                let predicted_pmi = l.pmi_i1;
                (predicted_cqi, predicted_ri, predicted_pmi)
            }
            (Some(l), None) => (l.wideband_cqi, l.rank_indicator, l.pmi_i1),
            _ => (7, 1, 0), // Default mid-range values
        };

        CsiPrediction {
            predicted_cqi,
            predicted_ri,
            predicted_pmi,
            confidence: 0.85 - (self.prediction_horizon_slots as f64 * 0.02),
            horizon_slots: self.prediction_horizon_slots,
        }
    }
}

/// A single CSI measurement (input to CSI prediction)
#[derive(Debug, Clone)]
pub struct CsiMeasurement {
    /// Rank Indicator (1-8 for NR)
    pub rank_indicator: u8,
    /// PMI index i1 (wideband)
    pub pmi_i1: u16,
    /// PMI index i2 (subband, per subband)
    pub pmi_i2: Vec<u16>,
    /// Wideband CQI (0-15)
    pub wideband_cqi: u8,
    /// Subband CQI differential
    pub subband_cqi: Vec<i8>,
    /// SINR estimate in dB
    pub sinr_db: f64,
}

/// Predicted CSI output
#[derive(Debug, Clone)]
pub struct CsiPrediction {
    /// Predicted wideband CQI
    pub predicted_cqi: u8,
    /// Predicted rank indicator
    pub predicted_ri: u8,
    /// Predicted PMI
    pub predicted_pmi: u16,
    /// Prediction confidence (0.0 - 1.0)
    pub confidence: f64,
    /// Prediction horizon in slots
    pub horizon_slots: u32,
}

/// Beam management ML model (TS 38.843 Use Case 2)
///
/// Supports:
/// - Beam selection: predict best beam from partial measurements
/// - Beam tracking: predict beam trajectory for mobile UEs
/// - Beam failure prediction: predict impending beam failures
#[derive(Debug)]
pub struct BeamManagementModel {
    /// Model identifier
    pub model_id: String,
    /// Number of SSB beams
    pub num_ssb_beams: u16,
    /// Number of CSI-RS beams
    pub num_csi_rs_beams: u16,
    /// Beam prediction accuracy (top-K)
    pub top_k_accuracy: f64,
    /// Lifecycle state
    pub state: ModelLifecycleState,
    /// Inference count
    pub inference_count: u64,
}

impl BeamManagementModel {
    pub fn new(model_id: &str, num_ssb_beams: u16, num_csi_rs_beams: u16) -> Self {
        Self {
            model_id: model_id.to_string(),
            num_ssb_beams,
            num_csi_rs_beams,
            top_k_accuracy: 0.92,
            state: ModelLifecycleState::Registered,
            inference_count: 0,
        }
    }

    /// Predict best beam given partial beam measurements.
    /// Uses a subset of measured beams to predict the best beam without exhaustive sweep.
    pub fn predict_best_beam(&mut self, measurements: &[BeamMeasurement]) -> BeamPrediction {
        self.inference_count += 1;

        // Find best measured beam and predict based on spatial correlation
        let best = measurements
            .iter()
            .max_by(|a, b| a.rsrp_dbm.partial_cmp(&b.rsrp_dbm).unwrap_or(std::cmp::Ordering::Equal));

        let (best_beam_id, best_rsrp) = match best {
            Some(m) => (m.beam_id, m.rsrp_dbm),
            None => (0, -120.0),
        };

        // Predict adjacent beams that might be better (spatial correlation model)
        let alt_beam = if best_beam_id > 0 { best_beam_id - 1 } else { best_beam_id + 1 };

        BeamPrediction {
            predicted_beam_id: best_beam_id,
            predicted_rsrp_dbm: best_rsrp + 0.5, // ML refinement
            alternative_beam_id: alt_beam,
            confidence: self.top_k_accuracy,
            beam_failure_probability: if best_rsrp < -110.0 { 0.3 } else { 0.02 },
        }
    }

    /// Predict beam trajectory for a mobile UE over the next N slots
    pub fn predict_beam_trajectory(
        &mut self,
        history: &[BeamMeasurement],
        prediction_slots: u32,
    ) -> Vec<u16> {
        self.inference_count += 1;

        let current = history.last().map(|m| m.beam_id).unwrap_or(0);
        let prev = if history.len() >= 2 {
            history[history.len() - 2].beam_id
        } else {
            current
        };

        // Simple trajectory extrapolation
        let beam_delta = current as i32 - prev as i32;
        (0..prediction_slots)
            .map(|i| {
                let predicted = current as i32 + beam_delta * (i as i32 + 1);
                predicted.clamp(0, self.num_ssb_beams as i32 - 1) as u16
            })
            .collect()
    }
}

/// A beam measurement (L1-RSRP per beam)
#[derive(Debug, Clone)]
pub struct BeamMeasurement {
    /// Beam index
    pub beam_id: u16,
    /// L1-RSRP in dBm
    pub rsrp_dbm: f64,
    /// L1-SINR in dB (optional)
    pub sinr_db: Option<f64>,
    /// SSB or CSI-RS
    pub is_ssb: bool,
}

/// Beam prediction output
#[derive(Debug, Clone)]
pub struct BeamPrediction {
    /// Predicted best beam
    pub predicted_beam_id: u16,
    /// Predicted RSRP at best beam
    pub predicted_rsrp_dbm: f64,
    /// Alternative beam (second best)
    pub alternative_beam_id: u16,
    /// Prediction confidence
    pub confidence: f64,
    /// Estimated probability of beam failure within prediction horizon
    pub beam_failure_probability: f64,
}

/// ML-enhanced positioning model (TS 38.843 Use Case 3)
///
/// Improves positioning accuracy beyond conventional methods by learning
/// the mapping from radio measurements to position.
#[derive(Debug)]
pub struct MlPositioningModel {
    /// Model identifier
    pub model_id: String,
    /// Positioning method (direct/indirect)
    pub method: MlPositioningMethod,
    /// Achieved accuracy in meters (horizontal 95%)
    pub accuracy_m: f64,
    /// Lifecycle state
    pub state: ModelLifecycleState,
    /// Inference count
    pub inference_count: u64,
}

/// ML positioning method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlPositioningMethod {
    /// Direct: measurements → position (end-to-end)
    Direct,
    /// Indirect: measurements → improved measurements → conventional positioning
    Indirect,
    /// Hybrid: combines ML and conventional
    Hybrid,
}

impl MlPositioningModel {
    pub fn new(model_id: &str, method: MlPositioningMethod) -> Self {
        Self {
            model_id: model_id.to_string(),
            method,
            accuracy_m: match method {
                MlPositioningMethod::Direct => 1.5,
                MlPositioningMethod::Indirect => 2.0,
                MlPositioningMethod::Hybrid => 1.0,
            },
            state: ModelLifecycleState::Registered,
            inference_count: 0,
        }
    }

    /// Estimate position from radio measurements
    pub fn estimate_position(&mut self, measurements: &[PositioningMeasurement]) -> PositionEstimateResult {
        self.inference_count += 1;

        // Aggregate measurement types for positioning
        let mut lat_sum = 0.0f64;
        let mut lon_sum = 0.0f64;
        let mut weight_sum = 0.0f64;

        for m in measurements {
            // Weight by measurement quality (higher RSRP = better)
            let weight = (m.rsrp_dbm + 140.0).max(1.0); // Normalize to positive
            lat_sum += m.cell_lat * weight;
            lon_sum += m.cell_lon * weight;
            weight_sum += weight;
        }

        if weight_sum > 0.0 {
            PositionEstimateResult {
                latitude: lat_sum / weight_sum,
                longitude: lon_sum / weight_sum,
                altitude_m: measurements.first().map(|m| m.altitude_m).unwrap_or(0.0),
                horizontal_accuracy_m: self.accuracy_m,
                vertical_accuracy_m: self.accuracy_m * 2.0,
                confidence: 0.95,
                method: self.method,
            }
        } else {
            PositionEstimateResult {
                latitude: 0.0,
                longitude: 0.0,
                altitude_m: 0.0,
                horizontal_accuracy_m: 100.0,
                vertical_accuracy_m: 200.0,
                confidence: 0.1,
                method: self.method,
            }
        }
    }
}

/// Radio measurement for ML positioning
#[derive(Debug, Clone)]
pub struct PositioningMeasurement {
    /// Cell ID
    pub cell_id: u32,
    /// Cell latitude
    pub cell_lat: f64,
    /// Cell longitude
    pub cell_lon: f64,
    /// Cell altitude
    pub altitude_m: f64,
    /// RSRP measurement
    pub rsrp_dbm: f64,
    /// Timing advance (if available)
    pub timing_advance_us: Option<f64>,
    /// Angle of arrival (if available)
    pub aoa_deg: Option<f64>,
}

/// Position estimate from ML model
#[derive(Debug, Clone)]
pub struct PositionEstimateResult {
    pub latitude: f64,
    pub longitude: f64,
    pub altitude_m: f64,
    pub horizontal_accuracy_m: f64,
    pub vertical_accuracy_m: f64,
    pub confidence: f64,
    pub method: MlPositioningMethod,
}

/// Model lifecycle manager for gNB (TS 38.843 Section 4)
///
/// Manages ML model lifecycle at the gNB including:
/// - Model registration, loading, activation, deactivation
/// - Performance monitoring (accuracy, latency)
/// - Automatic retraining triggers
#[derive(Debug)]
pub struct GnbModelLifecycleManager {
    /// Registered models by domain
    models: HashMap<NrModelDomain, GnbManagedModel>,
    /// Performance thresholds for retraining triggers
    retraining_thresholds: HashMap<NrModelDomain, f64>,
}

/// A managed model at the gNB
#[derive(Debug)]
pub struct GnbManagedModel {
    /// Model domain
    pub domain: NrModelDomain,
    /// Model version
    pub version: String,
    /// Current lifecycle state
    pub state: ModelLifecycleState,
    /// When the model was loaded
    pub loaded_at: Option<Instant>,
    /// Total inferences
    pub total_inferences: u64,
    /// Average inference latency
    pub avg_latency_us: f64,
    /// Current performance metric (accuracy or loss)
    pub performance_metric: f64,
    /// Performance history (last N measurements)
    pub performance_history: Vec<f64>,
}

impl GnbModelLifecycleManager {
    pub fn new() -> Self {
        let mut thresholds = HashMap::new();
        // Default retraining thresholds (minimum acceptable performance)
        thresholds.insert(NrModelDomain::CsiPrediction, -10.0); // NMSE threshold
        thresholds.insert(NrModelDomain::BeamManagement, 0.80); // Top-K accuracy
        thresholds.insert(NrModelDomain::Positioning, 5.0); // Max error meters
        thresholds.insert(NrModelDomain::CsiCompression, -8.0); // NMSE threshold

        Self {
            models: HashMap::new(),
            retraining_thresholds: thresholds,
        }
    }

    /// Register a new model
    pub fn register_model(&mut self, domain: NrModelDomain, version: &str) {
        self.models.insert(domain, GnbManagedModel {
            domain,
            version: version.to_string(),
            state: ModelLifecycleState::Registered,
            loaded_at: None,
            total_inferences: 0,
            avg_latency_us: 0.0,
            performance_metric: 0.0,
            performance_history: Vec::new(),
        });
    }

    /// Activate a model (transition: Registered/Loaded → Active)
    pub fn activate_model(&mut self, domain: NrModelDomain) -> bool {
        if let Some(model) = self.models.get_mut(&domain) {
            model.state = ModelLifecycleState::Active;
            model.loaded_at = Some(Instant::now());
            true
        } else {
            false
        }
    }

    /// Deactivate a model
    pub fn deactivate_model(&mut self, domain: NrModelDomain) -> bool {
        if let Some(model) = self.models.get_mut(&domain) {
            model.state = ModelLifecycleState::Deactivated;
            true
        } else {
            false
        }
    }

    /// Record inference result and update metrics
    pub fn record_inference(
        &mut self,
        domain: NrModelDomain,
        latency: Duration,
        performance: f64,
    ) {
        if let Some(model) = self.models.get_mut(&domain) {
            model.total_inferences += 1;
            let lat_us = latency.as_micros() as f64;
            // Exponential moving average for latency
            model.avg_latency_us = model.avg_latency_us * 0.95 + lat_us * 0.05;
            model.performance_metric = performance;
            model.performance_history.push(performance);
            if model.performance_history.len() > 100 {
                model.performance_history.remove(0);
            }

            // Check if performance has degraded
            if let Some(threshold) = self.retraining_thresholds.get(&domain) {
                let needs_retrain = match domain {
                    NrModelDomain::CsiPrediction | NrModelDomain::CsiCompression => {
                        // NMSE: lower (more negative) is better
                        performance > *threshold
                    }
                    NrModelDomain::BeamManagement => {
                        // Accuracy: higher is better
                        performance < *threshold
                    }
                    NrModelDomain::Positioning => {
                        // Error: lower is better
                        performance > *threshold
                    }
                };
                if needs_retrain {
                    model.state = ModelLifecycleState::Degraded;
                }
            }
        }
    }

    /// Get all models that need retraining
    pub fn models_needing_retraining(&self) -> Vec<NrModelDomain> {
        self.models
            .iter()
            .filter(|(_, m)| m.state == ModelLifecycleState::Degraded)
            .map(|(d, _)| *d)
            .collect()
    }

    /// Get model status
    pub fn get_model_status(&self, domain: NrModelDomain) -> Option<&GnbManagedModel> {
        self.models.get(&domain)
    }

    /// List all registered domains
    pub fn list_domains(&self) -> Vec<NrModelDomain> {
        self.models.keys().copied().collect()
    }
}

impl Default for GnbModelLifecycleManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csi_prediction() {
        let mut model = CsiPredictionModel::new("csi_v1", 32, 16);
        assert_eq!(model.state, ModelLifecycleState::Registered);

        let measurements = vec![
            CsiMeasurement {
                rank_indicator: 2, pmi_i1: 5, pmi_i2: vec![1, 2],
                wideband_cqi: 10, subband_cqi: vec![0, 1, -1], sinr_db: 15.0,
            },
            CsiMeasurement {
                rank_indicator: 2, pmi_i1: 5, pmi_i2: vec![1, 3],
                wideband_cqi: 11, subband_cqi: vec![0, 1, 0], sinr_db: 16.0,
            },
        ];

        let prediction = model.predict(&measurements);
        assert!(prediction.predicted_cqi >= 1 && prediction.predicted_cqi <= 15);
        assert!(prediction.confidence > 0.0 && prediction.confidence <= 1.0);
        assert_eq!(model.inference_count, 1);
    }

    #[test]
    fn test_beam_management() {
        let mut model = BeamManagementModel::new("beam_v1", 64, 128);

        let measurements = vec![
            BeamMeasurement { beam_id: 10, rsrp_dbm: -85.0, sinr_db: Some(15.0), is_ssb: true },
            BeamMeasurement { beam_id: 11, rsrp_dbm: -90.0, sinr_db: Some(12.0), is_ssb: true },
            BeamMeasurement { beam_id: 12, rsrp_dbm: -82.0, sinr_db: Some(18.0), is_ssb: true },
        ];

        let prediction = model.predict_best_beam(&measurements);
        assert_eq!(prediction.predicted_beam_id, 12); // Highest RSRP
        assert!(prediction.beam_failure_probability < 0.1);

        let trajectory = model.predict_beam_trajectory(&measurements, 4);
        assert_eq!(trajectory.len(), 4);
    }

    #[test]
    fn test_ml_positioning() {
        let mut model = MlPositioningModel::new("pos_v1", MlPositioningMethod::Hybrid);
        assert_eq!(model.accuracy_m, 1.0);

        let measurements = vec![
            PositioningMeasurement {
                cell_id: 1, cell_lat: 40.0, cell_lon: 29.0, altitude_m: 50.0,
                rsrp_dbm: -80.0, timing_advance_us: Some(1.0), aoa_deg: Some(45.0),
            },
            PositioningMeasurement {
                cell_id: 2, cell_lat: 40.001, cell_lon: 29.001, altitude_m: 50.0,
                rsrp_dbm: -90.0, timing_advance_us: Some(2.0), aoa_deg: Some(135.0),
            },
        ];

        let result = model.estimate_position(&measurements);
        assert!(result.latitude > 39.0 && result.latitude < 41.0);
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_model_lifecycle_manager() {
        let mut mgr = GnbModelLifecycleManager::new();
        mgr.register_model(NrModelDomain::CsiPrediction, "1.0");
        mgr.register_model(NrModelDomain::BeamManagement, "1.0");

        assert!(mgr.activate_model(NrModelDomain::CsiPrediction));
        assert_eq!(
            mgr.get_model_status(NrModelDomain::CsiPrediction).unwrap().state,
            ModelLifecycleState::Active
        );

        // Record good performance
        mgr.record_inference(NrModelDomain::CsiPrediction, Duration::from_micros(100), -15.0);
        assert_eq!(
            mgr.get_model_status(NrModelDomain::CsiPrediction).unwrap().state,
            ModelLifecycleState::Active
        );

        // Record degraded performance (NMSE above threshold)
        mgr.record_inference(NrModelDomain::CsiPrediction, Duration::from_micros(100), -5.0);
        assert_eq!(
            mgr.get_model_status(NrModelDomain::CsiPrediction).unwrap().state,
            ModelLifecycleState::Degraded
        );

        let needs_retrain = mgr.models_needing_retraining();
        assert!(needs_retrain.contains(&NrModelDomain::CsiPrediction));
    }
}
