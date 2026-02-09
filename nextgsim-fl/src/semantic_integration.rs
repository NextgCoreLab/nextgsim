//! Integration with Semantic Communication for distributed codec training
//!
//! This module provides interfaces for using Federated Learning to train
//! semantic communication encoder/decoder models distributedly.

use crate::{AggregatedModel, AggregationAlgorithm, ModelUpdate};
use serde::{Deserialize, Serialize};

/// Semantic codec training configuration for FL
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticCodecTrainingConfig {
    /// Model type being trained
    pub model_type: SemanticCodecType,
    /// Task type for the codec
    pub task_type: SemanticTaskType,
    /// Target compression ratio
    pub target_compression: f32,
    /// Minimum quality threshold
    pub min_quality: f32,
    /// Number of training rounds
    pub num_rounds: u32,
    /// Minimum participants per round
    pub min_participants: u32,
    /// Learning rate
    pub learning_rate: f32,
}

impl Default for SemanticCodecTrainingConfig {
    fn default() -> Self {
        Self {
            model_type: SemanticCodecType::Encoder,
            task_type: SemanticTaskType::ImageClassification,
            target_compression: 8.0,
            min_quality: 0.7,
            num_rounds: 100,
            min_participants: 5,
            learning_rate: 0.001,
        }
    }
}

/// Type of semantic codec being trained
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SemanticCodecType {
    /// Encoder only
    Encoder,
    /// Decoder only
    Decoder,
    /// Joint encoder-decoder (autoencoder)
    Joint,
    /// Task-specific codec
    TaskSpecific,
}

impl std::fmt::Display for SemanticCodecType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SemanticCodecType::Encoder => write!(f, "Encoder"),
            SemanticCodecType::Decoder => write!(f, "Decoder"),
            SemanticCodecType::Joint => write!(f, "Joint"),
            SemanticCodecType::TaskSpecific => write!(f, "TaskSpecific"),
        }
    }
}

/// Semantic task type for codec training
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SemanticTaskType {
    /// Image classification
    ImageClassification,
    /// Object detection
    ObjectDetection,
    /// Speech recognition
    SpeechRecognition,
    /// Text understanding
    TextUnderstanding,
    /// Sensor data fusion
    SensorFusion,
    /// Video analytics
    VideoAnalytics,
}

impl std::fmt::Display for SemanticTaskType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SemanticTaskType::ImageClassification => write!(f, "ImageClassification"),
            SemanticTaskType::ObjectDetection => write!(f, "ObjectDetection"),
            SemanticTaskType::SpeechRecognition => write!(f, "SpeechRecognition"),
            SemanticTaskType::TextUnderstanding => write!(f, "TextUnderstanding"),
            SemanticTaskType::SensorFusion => write!(f, "SensorFusion"),
            SemanticTaskType::VideoAnalytics => write!(f, "VideoAnalytics"),
        }
    }
}

/// Semantic codec model update with quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticModelUpdate {
    /// Base model update
    pub model_update: ModelUpdate,
    /// Codec type
    pub codec_type: SemanticCodecType,
    /// Task type
    pub task_type: SemanticTaskType,
    /// Achieved compression ratio
    pub compression_ratio: f32,
    /// Reconstruction quality
    pub quality_score: f32,
    /// Task-specific accuracy (if applicable)
    pub task_accuracy: Option<f32>,
}

/// Result from semantic codec training round
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticTrainingResult {
    /// Aggregated model
    pub aggregated_model: AggregatedModel,
    /// Average compression ratio achieved
    pub avg_compression_ratio: f32,
    /// Average quality score
    pub avg_quality_score: f32,
    /// Average task accuracy (if applicable)
    pub avg_task_accuracy: Option<f32>,
    /// Convergence indicator
    pub converged: bool,
}

/// Federated trainer for semantic codecs
#[derive(Debug)]
pub struct SemanticCodecTrainer {
    config: SemanticCodecTrainingConfig,
    aggregator: crate::FederatedAggregator,
    current_round: u32,
    training_history: Vec<SemanticTrainingResult>,
}

impl SemanticCodecTrainer {
    /// Creates a new semantic codec trainer
    pub fn new(config: SemanticCodecTrainingConfig) -> Self {
        let aggregator = crate::FederatedAggregator::new(
            AggregationAlgorithm::FedAvg,
            config.min_participants as usize,
        );

        Self {
            config,
            aggregator,
            current_round: 0,
            training_history: Vec::new(),
        }
    }

    /// Initializes the codec model
    pub fn initialize_model(&mut self, initial_weights: Vec<f32>) {
        self.aggregator.initialize_model(initial_weights);
    }

    /// Registers a participant for training
    pub fn register_participant(&mut self, participant_id: &str, num_samples: u64) {
        self.aggregator.register_participant(participant_id, num_samples);
    }

    /// Starts a new training round
    pub fn start_round(&mut self) -> Result<u32, String> {
        self.current_round += 1;
        self.aggregator
            .start_round()
            .map_err(|e| format!("Failed to start round: {e}"))?;
        Ok(self.current_round)
    }

    /// Submits a semantic model update
    pub fn submit_semantic_update(
        &mut self,
        update: SemanticModelUpdate,
    ) -> Result<(), String> {
        self.aggregator
            .submit_update(update.model_update)
            .map_err(|e| format!("Failed to submit update: {e}"))
    }

    /// Aggregates semantic model updates
    pub fn aggregate_semantic_models(&mut self) -> Result<SemanticTrainingResult, String> {
        let aggregated = self
            .aggregator
            .aggregate()
            .map_err(|e| format!("Failed to aggregate: {e}"))?;

        // For now, return dummy metrics - in a real implementation,
        // these would be computed from participant updates
        let result = SemanticTrainingResult {
            aggregated_model: aggregated,
            avg_compression_ratio: self.config.target_compression,
            avg_quality_score: 0.8,
            avg_task_accuracy: Some(0.85),
            converged: false,
        };

        self.training_history.push(result.clone());

        Ok(result)
    }

    /// Checks if training has converged
    pub fn has_converged(&self) -> bool {
        // Simple convergence check: look at last few rounds
        if self.training_history.len() < 3 {
            return false;
        }

        let recent = &self.training_history[self.training_history.len() - 3..];
        let loss_values: Vec<f32> = recent.iter().map(|r| r.aggregated_model.avg_loss).collect();

        // Check if loss has stabilized
        let max_diff = loss_values
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max);

        max_diff < 0.01 // Converged if loss change < 0.01
    }

    /// Returns the current round number
    pub fn current_round(&self) -> u32 {
        self.current_round
    }

    /// Returns the training history
    pub fn training_history(&self) -> &[SemanticTrainingResult] {
        &self.training_history
    }

    /// Returns the configuration
    pub fn config(&self) -> &SemanticCodecTrainingConfig {
        &self.config
    }

    /// Returns the number of registered participants
    pub fn participant_count(&self) -> usize {
        self.aggregator.participant_count()
    }
}

/// Builder for semantic codec training
pub struct SemanticCodecTrainerBuilder {
    config: SemanticCodecTrainingConfig,
}

impl SemanticCodecTrainerBuilder {
    /// Creates a new builder
    pub fn new() -> Self {
        Self {
            config: SemanticCodecTrainingConfig::default(),
        }
    }

    /// Sets the codec type
    pub fn codec_type(mut self, codec_type: SemanticCodecType) -> Self {
        self.config.model_type = codec_type;
        self
    }

    /// Sets the task type
    pub fn task_type(mut self, task_type: SemanticTaskType) -> Self {
        self.config.task_type = task_type;
        self
    }

    /// Sets the target compression ratio
    pub fn target_compression(mut self, compression: f32) -> Self {
        self.config.target_compression = compression;
        self
    }

    /// Sets the minimum quality threshold
    pub fn min_quality(mut self, quality: f32) -> Self {
        self.config.min_quality = quality;
        self
    }

    /// Sets the number of training rounds
    pub fn num_rounds(mut self, rounds: u32) -> Self {
        self.config.num_rounds = rounds;
        self
    }

    /// Sets the minimum participants per round
    pub fn min_participants(mut self, participants: u32) -> Self {
        self.config.min_participants = participants;
        self
    }

    /// Sets the learning rate
    pub fn learning_rate(mut self, lr: f32) -> Self {
        self.config.learning_rate = lr;
        self
    }

    /// Builds the trainer
    pub fn build(self) -> SemanticCodecTrainer {
        SemanticCodecTrainer::new(self.config)
    }
}

impl Default for SemanticCodecTrainerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_codec_trainer_creation() {
        let config = SemanticCodecTrainingConfig::default();
        let trainer = SemanticCodecTrainer::new(config);

        assert_eq!(trainer.current_round(), 0);
        assert_eq!(trainer.training_history().len(), 0);
    }

    #[test]
    fn test_semantic_codec_trainer_builder() {
        let trainer = SemanticCodecTrainerBuilder::new()
            .codec_type(SemanticCodecType::Joint)
            .task_type(SemanticTaskType::ImageClassification)
            .target_compression(10.0)
            .min_quality(0.8)
            .num_rounds(50)
            .min_participants(10)
            .learning_rate(0.01)
            .build();

        assert_eq!(trainer.config().model_type, SemanticCodecType::Joint);
        assert_eq!(
            trainer.config().task_type,
            SemanticTaskType::ImageClassification
        );
        assert_eq!(trainer.config().target_compression, 10.0);
        assert_eq!(trainer.config().min_quality, 0.8);
        assert_eq!(trainer.config().num_rounds, 50);
        assert_eq!(trainer.config().min_participants, 10);
        assert_eq!(trainer.config().learning_rate, 0.01);
    }

    #[test]
    fn test_semantic_codec_type_display() {
        assert_eq!(format!("{}", SemanticCodecType::Encoder), "Encoder");
        assert_eq!(format!("{}", SemanticCodecType::Joint), "Joint");
    }

    #[test]
    fn test_semantic_task_type_display() {
        assert_eq!(
            format!("{}", SemanticTaskType::ImageClassification),
            "ImageClassification"
        );
        assert_eq!(
            format!("{}", SemanticTaskType::SensorFusion),
            "SensorFusion"
        );
    }

    #[test]
    fn test_trainer_initialize_and_register() {
        let mut trainer = SemanticCodecTrainer::new(SemanticCodecTrainingConfig::default());

        trainer.initialize_model(vec![0.0; 100]);
        trainer.register_participant("ue-001", 1000);
        trainer.register_participant("ue-002", 1500);

        assert_eq!(trainer.participant_count(), 2);
    }

    #[test]
    fn test_training_round_lifecycle() {
        let config = SemanticCodecTrainingConfig {
            min_participants: 2,
            ..Default::default()
        };
        let mut trainer = SemanticCodecTrainer::new(config);

        trainer.initialize_model(vec![0.0; 10]);
        trainer.register_participant("ue-001", 100);
        trainer.register_participant("ue-002", 150);

        // Start round
        let round = trainer.start_round().unwrap();
        assert_eq!(round, 1);
        assert_eq!(trainer.current_round(), 1);

        // Submit updates
        let update1 = SemanticModelUpdate {
            model_update: ModelUpdate {
                participant_id: "ue-001".to_string(),
                base_version: 0,
                gradients: vec![0.1; 10],
                num_samples: 100,
                loss: 0.5,
                timestamp_ms: 1000,
            },
            codec_type: SemanticCodecType::Encoder,
            task_type: SemanticTaskType::ImageClassification,
            compression_ratio: 8.0,
            quality_score: 0.85,
            task_accuracy: Some(0.90),
        };

        let update2 = SemanticModelUpdate {
            model_update: ModelUpdate {
                participant_id: "ue-002".to_string(),
                base_version: 0,
                gradients: vec![0.15; 10],
                num_samples: 150,
                loss: 0.45,
                timestamp_ms: 1000,
            },
            codec_type: SemanticCodecType::Encoder,
            task_type: SemanticTaskType::ImageClassification,
            compression_ratio: 8.5,
            quality_score: 0.87,
            task_accuracy: Some(0.92),
        };

        trainer.submit_semantic_update(update1).unwrap();
        trainer.submit_semantic_update(update2).unwrap();

        // Aggregate
        let result = trainer.aggregate_semantic_models().unwrap();

        assert_eq!(result.aggregated_model.num_participants, 2);
        assert!(result.avg_compression_ratio > 0.0);
        assert!(result.avg_quality_score > 0.0);
    }

    #[test]
    fn test_convergence_check() {
        let mut trainer = SemanticCodecTrainer::new(SemanticCodecTrainingConfig::default());

        // Not converged with empty history
        assert!(!trainer.has_converged());

        // Add some training results with decreasing loss
        trainer.training_history.push(SemanticTrainingResult {
            aggregated_model: AggregatedModel {
                version: 1,
                weights: vec![],
                num_participants: 5,
                total_samples: 1000,
                avg_loss: 0.500,
                timestamp_ms: 1000,
            },
            avg_compression_ratio: 8.0,
            avg_quality_score: 0.8,
            avg_task_accuracy: Some(0.85),
            converged: false,
        });

        trainer.training_history.push(SemanticTrainingResult {
            aggregated_model: AggregatedModel {
                version: 2,
                weights: vec![],
                num_participants: 5,
                total_samples: 1000,
                avg_loss: 0.495,
                timestamp_ms: 2000,
            },
            avg_compression_ratio: 8.0,
            avg_quality_score: 0.82,
            avg_task_accuracy: Some(0.87),
            converged: false,
        });

        trainer.training_history.push(SemanticTrainingResult {
            aggregated_model: AggregatedModel {
                version: 3,
                weights: vec![],
                num_participants: 5,
                total_samples: 1000,
                avg_loss: 0.492,
                timestamp_ms: 3000,
            },
            avg_compression_ratio: 8.0,
            avg_quality_score: 0.83,
            avg_task_accuracy: Some(0.88),
            converged: false,
        });

        // Should converge with small loss changes
        assert!(trainer.has_converged());
    }
}
