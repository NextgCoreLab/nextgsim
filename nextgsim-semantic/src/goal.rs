//! Goal-Oriented Semantic Coding (A18.3)
//!
//! Encodes data not for perfect reconstruction, but for task success.
//! Optimizes transmission to maximize downstream task effectiveness.

use serde::{Deserialize, Serialize};

use crate::{SemanticFeatures, SemanticTask};

/// Task effectiveness metric
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EffectivenessMetric {
    /// Classification accuracy
    Accuracy,
    /// F1 score (precision-recall balance)
    F1Score,
    /// mAP (mean Average Precision) for detection
    MeanAveragePrecision,
    /// `IoU` (Intersection over Union) for segmentation
    IoU,
    /// BLEU score for translation
    BleuScore,
    /// Custom metric
    Custom,
}

/// Goal-oriented encoding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalConfig {
    /// Target task
    pub task: SemanticTask,
    /// Effectiveness metric to optimize
    pub metric: EffectivenessMetric,
    /// Minimum acceptable effectiveness (0.0 to 1.0)
    pub min_effectiveness: f32,
    /// Feature importance threshold
    pub importance_threshold: f32,
}

impl Default for GoalConfig {
    fn default() -> Self {
        Self {
            task: SemanticTask::ImageClassification,
            metric: EffectivenessMetric::Accuracy,
            min_effectiveness: 0.8,
            importance_threshold: 0.1,
        }
    }
}

/// Goal-oriented encoder that prioritizes task-relevant features
pub struct GoalOrientedEncoder {
    /// Configuration
    config: GoalConfig,
    /// Learned feature importance weights (from task-specific training)
    feature_importance: Vec<f32>,
}

impl GoalOrientedEncoder {
    /// Creates a new goal-oriented encoder
    pub fn new(config: GoalConfig, feature_dim: usize) -> Self {
        Self {
            config,
            feature_importance: vec![1.0; feature_dim],
        }
    }

    /// Sets learned feature importance weights
    pub fn set_importance(&mut self, importance: Vec<f32>) {
        self.feature_importance = importance;
    }

    /// Encodes data focusing on task-critical features
    pub fn encode(&self, data: &[f32]) -> SemanticFeatures {
        let dim = data.len().min(self.feature_importance.len());

        // Weight features by importance
        let weighted: Vec<(usize, f32, f32)> = (0..dim)
            .map(|i| {
                let importance = self.feature_importance.get(i).copied().unwrap_or(1.0);
                (i, data[i], importance)
            })
            .collect();

        // Select features above importance threshold
        let selected: Vec<(usize, f32, f32)> = weighted
            .into_iter()
            .filter(|(_, _, importance)| *importance >= self.config.importance_threshold)
            .collect();

        // Extract features and importance
        let features: Vec<f32> = selected.iter().map(|(_, val, _)| *val).collect();
        let importance: Vec<f32> = selected.iter().map(|(_, _, imp)| *imp).collect();

        let task_id = task_to_id(self.config.task);
        SemanticFeatures::new(task_id, features, vec![data.len()]).with_importance(importance)
    }

    /// Evaluates task effectiveness of transmitted features
    /// Returns a score between 0.0 and 1.0
    pub fn evaluate_effectiveness(&self, transmitted: &SemanticFeatures, ground_truth: &[f32]) -> f32 {
        // Simulate task-specific evaluation
        // In practice, this would run the downstream task (e.g., classifier)

        match self.config.metric {
            EffectivenessMetric::Accuracy => {
                // Simulate classification: check if top-k features are preserved
                let k = (ground_truth.len() as f32 * 0.1) as usize;
                self.top_k_preservation(transmitted, ground_truth, k)
            }
            EffectivenessMetric::F1Score => {
                // Simulate F1: balance between precision and recall
                let preserved = self.top_k_preservation(transmitted, ground_truth, 10);
                let completeness = transmitted.num_features() as f32 / ground_truth.len() as f32;
                2.0 * (preserved * completeness) / (preserved + completeness + 1e-6)
            }
            EffectivenessMetric::MeanAveragePrecision => {
                // Simulate mAP: average precision across importance rankings
                self.average_precision(transmitted, ground_truth)
            }
            _ => {
                // Default: cosine similarity as proxy
                self.feature_similarity(transmitted, ground_truth)
            }
        }
    }

    /// Checks if top-k features are preserved
    fn top_k_preservation(&self, transmitted: &SemanticFeatures, ground_truth: &[f32], k: usize) -> f32 {
        let k = k.min(ground_truth.len());

        // Find top-k indices in ground truth
        let mut gt_scored: Vec<(usize, f32)> = ground_truth
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v.abs()))
            .collect();
        gt_scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_k_indices: Vec<usize> = gt_scored.iter().take(k).map(|(i, _)| *i).collect();

        // Check how many are in the transmitted features
        let mut preserved = 0;
        for &idx in &top_k_indices {
            if idx < transmitted.num_features() {
                preserved += 1;
            }
        }

        preserved as f32 / k as f32
    }

    /// Computes average precision
    fn average_precision(&self, transmitted: &SemanticFeatures, ground_truth: &[f32]) -> f32 {
        // Simplified AP: fraction of high-importance features preserved
        let total_important = ground_truth
            .iter()
            .filter(|&&v| v.abs() >= self.config.importance_threshold)
            .count();

        if total_important == 0 {
            return 1.0;
        }

        let preserved_important = transmitted
            .features
            .iter()
            .filter(|&&v| v.abs() >= self.config.importance_threshold)
            .count();

        preserved_important as f32 / total_important as f32
    }

    /// Computes feature similarity
    fn feature_similarity(&self, transmitted: &SemanticFeatures, ground_truth: &[f32]) -> f32 {
        let len = transmitted.num_features().min(ground_truth.len());
        if len == 0 {
            return 0.0;
        }

        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;

        for i in 0..len {
            let a = transmitted.features.get(i).copied().unwrap_or(0.0);
            let b = ground_truth.get(i).copied().unwrap_or(0.0);
            dot += a * b;
            norm_a += a * a;
            norm_b += b * b;
        }

        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom == 0.0 {
            0.0
        } else {
            (dot / denom).clamp(0.0, 1.0)
        }
    }
}

/// Converts `SemanticTask` to task ID
fn task_to_id(task: SemanticTask) -> u32 {
    match task {
        SemanticTask::ImageClassification => 0,
        SemanticTask::ObjectDetection => 1,
        SemanticTask::SpeechRecognition => 2,
        SemanticTask::TextUnderstanding => 3,
        SemanticTask::SensorFusion => 4,
        SemanticTask::VideoAnalytics => 5,
        SemanticTask::Custom(id) => id,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_goal_config() {
        let config = GoalConfig::default();
        assert_eq!(config.min_effectiveness, 0.8);
        assert_eq!(config.metric, EffectivenessMetric::Accuracy);
    }

    #[test]
    fn test_goal_oriented_encoder() {
        let config = GoalConfig {
            importance_threshold: 0.3,
            ..Default::default()
        };
        let encoder = GoalOrientedEncoder::new(config, 10);

        assert_eq!(encoder.feature_importance.len(), 10);
    }

    #[test]
    fn test_encode_with_importance() {
        let config = GoalConfig {
            importance_threshold: 0.5,
            ..Default::default()
        };
        let mut encoder = GoalOrientedEncoder::new(config, 5);

        // Set importance: only first 3 features are important
        encoder.set_importance(vec![0.9, 0.8, 0.7, 0.3, 0.2]);

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let features = encoder.encode(&data);

        // Should select only features with importance >= 0.5
        assert_eq!(features.num_features(), 3);
    }

    #[test]
    fn test_effectiveness_evaluation() {
        let config = GoalConfig {
            metric: EffectivenessMetric::Accuracy,
            ..Default::default()
        };
        let encoder = GoalOrientedEncoder::new(config, 10);

        let ground_truth = vec![1.0, 0.5, 0.8, 0.2, 0.9, 0.1, 0.7, 0.3, 0.6, 0.4];
        let features = SemanticFeatures::new(0, vec![1.0, 0.5, 0.8, 0.9, 0.7], vec![10]);

        let effectiveness = encoder.evaluate_effectiveness(&features, &ground_truth);
        assert!((0.0..=1.0).contains(&effectiveness));
    }

    #[test]
    fn test_top_k_preservation() {
        let config = GoalConfig::default();
        let encoder = GoalOrientedEncoder::new(config, 5);

        let ground_truth = vec![5.0, 1.0, 4.0, 2.0, 3.0]; // Top-3: indices 0, 2, 4
        let features = SemanticFeatures::new(0, vec![5.0, 1.0, 4.0, 2.0, 3.0], vec![5]);

        let score = encoder.top_k_preservation(&features, &ground_truth, 3);
        assert_eq!(score, 1.0); // All top-3 preserved
    }

    #[test]
    fn test_feature_similarity() {
        let config = GoalConfig::default();
        let encoder = GoalOrientedEncoder::new(config, 3);

        let ground_truth = vec![1.0, 2.0, 3.0];
        let features = SemanticFeatures::new(0, vec![1.0, 2.0, 3.0], vec![3]);

        let similarity = encoder.feature_similarity(&features, &ground_truth);
        assert!((similarity - 1.0).abs() < 0.01); // Perfect match
    }

    #[test]
    fn test_f1_score_metric() {
        let config = GoalConfig {
            metric: EffectivenessMetric::F1Score,
            ..Default::default()
        };
        let encoder = GoalOrientedEncoder::new(config, 10);

        let ground_truth = vec![1.0; 10];
        let features = SemanticFeatures::new(0, vec![1.0; 5], vec![10]);

        let f1 = encoder.evaluate_effectiveness(&features, &ground_truth);
        assert!(f1 > 0.0 && f1 <= 1.0);
    }
}
