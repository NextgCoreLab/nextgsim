//! AI-native sensing (ML-based positioning)
//!
//! Neural model augmentation for ISAC positioning and tracking.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{SensingMeasurement, Vector3};

/// ML model type for sensing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MlModelType {
    /// Deep neural network for positioning
    DeepPositioning,
    /// LSTM for trajectory prediction
    TrajectoryLstm,
    /// Transformer for multi-target tracking
    TransformerTracking,
    /// CNN for environment fingerprinting
    FingerprintCnn,
}

impl std::fmt::Display for MlModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MlModelType::DeepPositioning => write!(f, "DeepPositioning"),
            MlModelType::TrajectoryLstm => write!(f, "TrajectoryLSTM"),
            MlModelType::TransformerTracking => write!(f, "TransformerTracking"),
            MlModelType::FingerprintCnn => write!(f, "FingerprintCNN"),
        }
    }
}

/// ML-based positioning result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlPositioningResult {
    /// Estimated position
    pub position: Vector3,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Model used
    pub model_type: MlModelType,
    /// Latency (milliseconds)
    pub inference_latency_ms: f64,
    /// Feature vector dimension used
    pub feature_dim: usize,
}

/// Feature extractor for ML sensing
#[derive(Debug, Clone)]
pub struct SensingFeatureExtractor {
    /// Maximum number of measurements to include
    pub max_measurements: usize,
    /// Include temporal features
    pub include_temporal: bool,
    /// Include spatial features
    pub include_spatial: bool,
}

impl Default for SensingFeatureExtractor {
    fn default() -> Self {
        Self {
            max_measurements: 16,
            include_temporal: true,
            include_spatial: true,
        }
    }
}

impl SensingFeatureExtractor {
    /// Extracts features from sensing measurements
    pub fn extract(&self, measurements: &[SensingMeasurement]) -> Vec<f64> {
        let mut features = Vec::new();

        // Limit measurements
        let meas = if measurements.len() > self.max_measurements {
            &measurements[..self.max_measurements]
        } else {
            measurements
        };

        // Raw measurement features
        for m in meas {
            features.push(m.value);
            features.push(m.uncertainty);
            features.push(m.anchor_id as f64);

            if self.include_temporal {
                features.push(m.timestamp_ms as f64);
            }
        }

        // Pad if needed
        while features.len() < self.max_measurements * 4 {
            features.push(0.0);
        }

        // Statistical features
        if self.include_spatial {
            let values: Vec<f64> = meas.iter().map(|m| m.value).collect();
            if !values.is_empty() {
                let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
                let variance: f64 = values
                    .iter()
                    .map(|v| (v - mean).powi(2))
                    .sum::<f64>()
                    / values.len() as f64;

                features.push(mean);
                features.push(variance.sqrt());
                features.push(*values.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0));
                features.push(*values.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0));
            }
        }

        features
    }

    /// Returns the expected feature dimension
    pub fn feature_dimension(&self) -> usize {
        let base = self.max_measurements * 4;
        let stats = if self.include_spatial { 4 } else { 0 };
        base + stats
    }
}

/// ML-based positioning engine
#[derive(Debug)]
pub struct MlPositioningEngine {
    /// Model type
    model_type: MlModelType,
    /// Feature extractor
    feature_extractor: SensingFeatureExtractor,
    /// Training data cache (for online learning)
    training_cache: Vec<(Vec<f64>, Vector3)>,
    /// Maximum cache size
    max_cache_size: usize,
    /// Is model trained
    is_trained: bool,
}

impl MlPositioningEngine {
    /// Creates a new ML positioning engine
    pub fn new(model_type: MlModelType) -> Self {
        Self {
            model_type,
            feature_extractor: SensingFeatureExtractor::default(),
            training_cache: Vec::new(),
            max_cache_size: 10000,
            is_trained: false,
        }
    }

    /// Adds a training sample
    pub fn add_training_sample(&mut self, measurements: &[SensingMeasurement], ground_truth: Vector3) {
        let features = self.feature_extractor.extract(measurements);
        self.training_cache.push((features, ground_truth));

        // Trim cache if needed
        if self.training_cache.len() > self.max_cache_size {
            self.training_cache.drain(0..self.training_cache.len() - self.max_cache_size);
        }
    }

    /// Trains the model (simplified simulation)
    pub fn train(&mut self, _epochs: usize) {
        // In a real implementation, this would:
        // 1. Build and train a neural network
        // 2. Use frameworks like PyTorch/TensorFlow
        // 3. Optimize for edge inference

        // For simulation, just mark as trained
        if self.training_cache.len() >= 100 {
            self.is_trained = true;
        }
    }

    /// Performs ML-based inference for positioning
    pub fn infer(&self, measurements: &[SensingMeasurement]) -> Option<MlPositioningResult> {
        if !self.is_trained {
            return None;
        }

        let features = self.feature_extractor.extract(measurements);

        // Simplified inference: k-NN over training cache
        // Real implementation would use trained neural network
        let k = 5;
        let mut distances: Vec<(f64, &Vector3)> = self
            .training_cache
            .iter()
            .map(|(cached_features, pos)| {
                let dist: f64 = features
                    .iter()
                    .zip(cached_features.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                (dist, pos)
            })
            .collect();

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        if distances.len() < k {
            return None;
        }

        // Average top-k positions
        let mut pos_sum = Vector3::default();
        for item in distances.iter().take(k) {
            let pos = item.1;
            pos_sum.x += pos.x;
            pos_sum.y += pos.y;
            pos_sum.z += pos.z;
        }

        let position = Vector3::new(
            pos_sum.x / k as f64,
            pos_sum.y / k as f64,
            pos_sum.z / k as f64,
        );

        // Compute confidence from distance variance
        let avg_dist: f64 = distances.iter().take(k).map(|(d, _)| d).sum::<f64>() / k as f64;
        let confidence = (1.0 / (1.0 + avg_dist)).clamp(0.0, 1.0);

        Some(MlPositioningResult {
            position,
            confidence,
            model_type: self.model_type,
            inference_latency_ms: 5.0, // Simulated edge inference latency
            feature_dim: features.len(),
        })
    }

    /// Returns if the model is trained
    pub fn is_trained(&self) -> bool {
        self.is_trained
    }

    /// Returns the number of training samples
    pub fn training_sample_count(&self) -> usize {
        self.training_cache.len()
    }

    /// Clears the training cache
    pub fn clear_training_cache(&mut self) {
        self.training_cache.clear();
        self.is_trained = false;
    }
}

/// Trajectory prediction using LSTM-like model
#[derive(Debug)]
pub struct TrajectoryPredictor {
    /// Historical positions (`object_id` -> position history)
    position_history: HashMap<u64, Vec<(f64, Vector3)>>, // (timestamp, position)
    /// Maximum history length
    max_history: usize,
}

impl TrajectoryPredictor {
    /// Creates a new trajectory predictor
    pub fn new(max_history: usize) -> Self {
        Self {
            position_history: HashMap::new(),
            max_history,
        }
    }

    /// Updates position history for an object
    pub fn update(&mut self, object_id: u64, timestamp: f64, position: Vector3) {
        let history = self.position_history.entry(object_id).or_default();
        history.push((timestamp, position));

        // Trim history
        if history.len() > self.max_history {
            history.drain(0..history.len() - self.max_history);
        }
    }

    /// Predicts future position (simplified linear extrapolation)
    pub fn predict(&self, object_id: u64, future_timestamp: f64) -> Option<Vector3> {
        let history = self.position_history.get(&object_id)?;

        if history.len() < 2 {
            return None;
        }

        // Simple linear extrapolation from last two points
        let (t1, p1) = history[history.len() - 2];
        let (t2, p2) = history[history.len() - 1];

        let dt = t2 - t1;
        if dt <= 0.0 {
            return Some(p2);
        }

        let dt_future = future_timestamp - t2;
        let velocity = Vector3::new(
            (p2.x - p1.x) / dt,
            (p2.y - p1.y) / dt,
            (p2.z - p1.z) / dt,
        );

        Some(Vector3::new(
            p2.x + velocity.x * dt_future,
            p2.y + velocity.y * dt_future,
            p2.z + velocity.z * dt_future,
        ))
    }

    /// Returns the number of tracked objects
    pub fn tracked_object_count(&self) -> usize {
        self.position_history.len()
    }
}

impl Default for TrajectoryPredictor {
    fn default() -> Self {
        Self::new(100)
    }
}

// ─── AI-Native Target Detection and Classification ────────────────────────────

/// Target classification category from radar returns
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TargetClass {
    /// Pedestrian
    Pedestrian,
    /// Vehicle (car, truck, etc.)
    Vehicle,
    /// Cyclist
    Cyclist,
    /// Drone / UAV
    Drone,
    /// Static object (building, wall, etc.)
    StaticObject,
    /// Unknown
    Unknown,
}

impl std::fmt::Display for TargetClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TargetClass::Pedestrian => write!(f, "Pedestrian"),
            TargetClass::Vehicle => write!(f, "Vehicle"),
            TargetClass::Cyclist => write!(f, "Cyclist"),
            TargetClass::Drone => write!(f, "Drone"),
            TargetClass::StaticObject => write!(f, "StaticObject"),
            TargetClass::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Detection result from AI-native sensing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiDetectionResult {
    /// Detected target class
    pub target_class: TargetClass,
    /// Classification confidence (0.0 to 1.0)
    pub confidence: f64,
    /// Estimated range (meters)
    pub range_m: f64,
    /// Estimated velocity (m/s)
    pub velocity_ms: f64,
    /// Estimated radar cross-section (m^2)
    pub rcs_m2: f64,
    /// Feature vector used for classification
    pub features: Vec<f64>,
}

/// AI-native sensing engine for target detection and classification
///
/// Uses ML-based feature extraction from radar range-Doppler maps
/// to detect and classify targets from radar returns.
#[derive(Debug)]
pub struct AiSensingEngine {
    /// Training samples: (features, class)
    training_data: Vec<(Vec<f64>, TargetClass)>,
    /// Class centroids computed from training data
    class_centroids: HashMap<TargetClass, Vec<f64>>,
    /// Whether the engine is trained
    is_trained: bool,
}

impl AiSensingEngine {
    /// Creates a new AI sensing engine
    pub fn new(_feature_dim: usize) -> Self {
        Self {
            training_data: Vec::new(),
            class_centroids: HashMap::new(),
            is_trained: false,
        }
    }

    /// Extracts features from a range-Doppler map cell
    ///
    /// Features: [power, range_bin_norm, doppler_bin_norm, spread_range, spread_doppler, peak_ratio]
    pub fn extract_features(
        range_doppler_map: &[Vec<f64>],
        range_bin: usize,
        doppler_bin: usize,
    ) -> Vec<f64> {
        let num_range = range_doppler_map.len();
        if num_range == 0 {
            return vec![0.0; 6];
        }
        let num_doppler = range_doppler_map[0].len();

        let power = range_doppler_map[range_bin.min(num_range - 1)]
            [doppler_bin.min(num_doppler - 1)];

        // Normalized bin positions
        let range_norm = range_bin as f64 / num_range as f64;
        let doppler_norm = doppler_bin as f64 / num_doppler as f64;

        // Compute local spread (variance in 3x3 neighborhood)
        let mut spread_range = 0.0;
        let mut spread_doppler = 0.0;
        let mut count = 0.0;

        for dr in 0..=2usize {
            for dd in 0..=2usize {
                let r = (range_bin + dr).saturating_sub(1);
                let d = (doppler_bin + dd).saturating_sub(1);
                if r < num_range && d < num_doppler {
                    let val = range_doppler_map[r][d];
                    spread_range += (r as f64 - range_bin as f64).powi(2) * val;
                    spread_doppler += (d as f64 - doppler_bin as f64).powi(2) * val;
                    count += val;
                }
            }
        }

        if count > 0.0 {
            spread_range /= count;
            spread_doppler /= count;
        }

        // Peak-to-average ratio in local neighborhood
        let max_val: f64 = range_doppler_map
            .iter()
            .flat_map(|row| row.iter())
            .copied()
            .fold(0.0, f64::max);
        let peak_ratio = if max_val > 0.0 { power / max_val } else { 0.0 };

        vec![power, range_norm, doppler_norm, spread_range, spread_doppler, peak_ratio]
    }

    /// Adds a training sample
    pub fn add_training_sample(&mut self, features: Vec<f64>, class: TargetClass) {
        self.training_data.push((features, class));
    }

    /// Trains the classifier using centroid-based approach
    pub fn train(&mut self) {
        if self.training_data.len() < 5 {
            return;
        }

        self.class_centroids.clear();

        // Group by class
        let mut class_groups: HashMap<TargetClass, Vec<&Vec<f64>>> = HashMap::new();
        for (features, class) in &self.training_data {
            class_groups.entry(*class).or_default().push(features);
        }

        // Compute centroid for each class
        for (class, samples) in &class_groups {
            let dim = samples[0].len();
            let mut centroid = vec![0.0; dim];
            let n = samples.len() as f64;

            for sample in samples {
                for (i, &val) in sample.iter().enumerate() {
                    if i < dim {
                        centroid[i] += val / n;
                    }
                }
            }

            self.class_centroids.insert(*class, centroid);
        }

        self.is_trained = !self.class_centroids.is_empty();
    }

    /// Detects and classifies a target from a range-Doppler map detection
    pub fn detect_and_classify(
        &self,
        range_doppler_map: &[Vec<f64>],
        range_bin: usize,
        doppler_bin: usize,
        range_per_bin_m: f64,
        velocity_per_bin_ms: f64,
    ) -> AiDetectionResult {
        let features = Self::extract_features(range_doppler_map, range_bin, doppler_bin);
        let power = features[0];
        let range_m = range_bin as f64 * range_per_bin_m;
        let velocity_ms = doppler_bin as f64 * velocity_per_bin_ms;

        // Estimate RCS from power (simplified radar equation)
        let rcs_m2 = power * range_m.powi(4) / 1e6;

        if !self.is_trained {
            // Default classification heuristic based on RCS and velocity
            let target_class = Self::heuristic_classify(rcs_m2, velocity_ms);
            return AiDetectionResult {
                target_class,
                confidence: 0.5,
                range_m,
                velocity_ms,
                rcs_m2,
                features,
            };
        }

        // Nearest-centroid classification
        let mut best_class = TargetClass::Unknown;
        let mut best_dist = f64::MAX;

        for (class, centroid) in &self.class_centroids {
            let dist: f64 = features
                .iter()
                .zip(centroid.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            if dist < best_dist {
                best_dist = dist;
                best_class = *class;
            }
        }

        let confidence = (1.0 / (1.0 + best_dist)).clamp(0.0, 1.0);

        AiDetectionResult {
            target_class: best_class,
            confidence,
            range_m,
            velocity_ms,
            rcs_m2,
            features,
        }
    }

    /// Heuristic classification based on RCS and velocity
    fn heuristic_classify(rcs_m2: f64, velocity_ms: f64) -> TargetClass {
        let abs_vel = velocity_ms.abs();
        if abs_vel < 0.1 {
            TargetClass::StaticObject
        } else if rcs_m2 > 10.0 && abs_vel > 5.0 {
            TargetClass::Vehicle
        } else if rcs_m2 < 0.5 && abs_vel < 2.0 {
            TargetClass::Pedestrian
        } else if rcs_m2 < 1.0 && abs_vel > 2.0 && abs_vel < 10.0 {
            TargetClass::Cyclist
        } else if rcs_m2 < 0.1 && abs_vel > 1.0 {
            TargetClass::Drone
        } else {
            TargetClass::Unknown
        }
    }

    /// Returns whether the engine is trained
    pub fn is_trained(&self) -> bool {
        self.is_trained
    }

    /// Returns the number of training samples
    pub fn training_sample_count(&self) -> usize {
        self.training_data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SensingType;

    #[test]
    fn test_feature_extractor() {
        let extractor = SensingFeatureExtractor::default();

        let measurements = vec![
            SensingMeasurement {
                measurement_type: SensingType::ToA,
                anchor_id: 1,
                value: 100.0,
                uncertainty: 1.0,
                timestamp_ms: 1000,
            },
            SensingMeasurement {
                measurement_type: SensingType::ToA,
                anchor_id: 2,
                value: 150.0,
                uncertainty: 2.0,
                timestamp_ms: 1000,
            },
        ];

        let features = extractor.extract(&measurements);
        assert!(!features.is_empty());
        assert_eq!(features.len(), extractor.feature_dimension());
    }

    #[test]
    fn test_ml_positioning_engine() {
        let mut engine = MlPositioningEngine::new(MlModelType::DeepPositioning);

        assert!(!engine.is_trained());

        // Add training samples
        for i in 0..150 {
            let measurements = vec![SensingMeasurement {
                measurement_type: SensingType::ToA,
                anchor_id: 1,
                value: 100.0 + i as f64,
                uncertainty: 1.0,
                timestamp_ms: 1000,
            }];
            let position = Vector3::new(i as f64, i as f64, 0.0);
            engine.add_training_sample(&measurements, position);
        }

        engine.train(10);
        assert!(engine.is_trained());

        // Test inference
        let test_measurements = vec![SensingMeasurement {
            measurement_type: SensingType::ToA,
            anchor_id: 1,
            value: 120.0,
            uncertainty: 1.0,
            timestamp_ms: 1000,
        }];

        let result = engine.infer(&test_measurements);
        assert!(result.is_some());

        let res = result.unwrap();
        assert!(res.confidence > 0.0 && res.confidence <= 1.0);
        assert_eq!(res.model_type, MlModelType::DeepPositioning);
    }

    #[test]
    fn test_trajectory_predictor() {
        let mut predictor = TrajectoryPredictor::new(10);

        // Simulate object moving in a straight line
        predictor.update(1, 0.0, Vector3::new(0.0, 0.0, 0.0));
        predictor.update(1, 1.0, Vector3::new(10.0, 5.0, 0.0));
        predictor.update(1, 2.0, Vector3::new(20.0, 10.0, 0.0));

        // Predict at t=3.0
        let predicted = predictor.predict(1, 3.0);
        assert!(predicted.is_some());

        let pos = predicted.unwrap();
        assert!((pos.x - 30.0).abs() < 0.1);
        assert!((pos.y - 15.0).abs() < 0.1);

        assert_eq!(predictor.tracked_object_count(), 1);
    }

    #[test]
    fn test_trajectory_predictor_multiple_objects() {
        let mut predictor = TrajectoryPredictor::new(10);

        predictor.update(1, 0.0, Vector3::new(0.0, 0.0, 0.0));
        predictor.update(2, 0.0, Vector3::new(100.0, 100.0, 0.0));

        assert_eq!(predictor.tracked_object_count(), 2);
    }

    #[test]
    fn test_ml_engine_clear_cache() {
        let mut engine = MlPositioningEngine::new(MlModelType::DeepPositioning);

        engine.add_training_sample(&[], Vector3::default());
        assert_eq!(engine.training_sample_count(), 1);

        engine.clear_training_cache();
        assert_eq!(engine.training_sample_count(), 0);
        assert!(!engine.is_trained());
    }

    // ── AI-Native Sensing tests ───────────────────────────────────────────

    #[test]
    fn test_ai_sensing_engine_creation() {
        let engine = AiSensingEngine::new(6);
        assert!(!engine.is_trained());
        assert_eq!(engine.training_sample_count(), 0);
    }

    #[test]
    fn test_feature_extraction() {
        let map = vec![vec![0.0; 32]; 32];
        let features = AiSensingEngine::extract_features(&map, 16, 16);
        assert_eq!(features.len(), 6);
    }

    #[test]
    fn test_heuristic_classify() {
        // Vehicle: high RCS, high speed
        let result = AiSensingEngine::heuristic_classify(20.0, 30.0);
        assert_eq!(result, TargetClass::Vehicle);

        // Static object: very low velocity
        let result = AiSensingEngine::heuristic_classify(5.0, 0.01);
        assert_eq!(result, TargetClass::StaticObject);
    }

    #[test]
    fn test_ai_sensing_detect_untrained() {
        let engine = AiSensingEngine::new(6);
        let map = vec![vec![1.0; 32]; 32];
        let result = engine.detect_and_classify(&map, 16, 16, 1.0, 1.0);
        assert_eq!(result.confidence, 0.5); // Default confidence for untrained
    }

    #[test]
    fn test_ai_sensing_train_and_classify() {
        let mut engine = AiSensingEngine::new(6);

        // Add training samples for vehicles
        for _ in 0..10 {
            engine.add_training_sample(
                vec![50.0, 0.5, 0.7, 1.0, 2.0, 0.8],
                TargetClass::Vehicle,
            );
        }
        // Add training samples for pedestrians
        for _ in 0..10 {
            engine.add_training_sample(
                vec![5.0, 0.3, 0.2, 0.1, 0.1, 0.3],
                TargetClass::Pedestrian,
            );
        }

        engine.train();
        assert!(engine.is_trained());
    }
}
