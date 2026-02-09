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
        for i in 0..k {
            let pos = distances[i].1;
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
    /// Historical positions (object_id -> position history)
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
}
