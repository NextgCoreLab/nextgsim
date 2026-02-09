//! Semantic Communication Protocols for 6G Networks
//!
//! Implements task-relevant feature transmission:
//! - Learned compression codecs
//! - Joint source-channel coding
//! - Semantic importance weighting
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                     Semantic Communication                               │
//! │                                                                          │
//! │  Transmitter                                    Receiver                 │
//! │  ┌──────────────────┐                         ┌──────────────────┐      │
//! │  │ Semantic Encoder │    Channel              │ Semantic Decoder │      │
//! │  │  • Feature       │──────────────────────>  │  • Feature       │      │
//! │  │    extraction    │   Compressed            │    reconstruction│      │
//! │  │  • Importance    │   features              │  • Task-aware    │      │
//! │  │    weighting     │                         │    decoding      │      │
//! │  └──────────────────┘                         └──────────────────┘      │
//! │           │                                            │                 │
//! │           ▼                                            ▼                 │
//! │  ┌──────────────────┐                         ┌──────────────────┐      │
//! │  │ Source Data      │                         │ Reconstructed    │      │
//! │  │ (image, video,   │                         │ Data / Task      │      │
//! │  │  sensor, etc.)   │                         │ Output           │      │
//! │  └──────────────────┘                         └──────────────────┘      │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Semantic Communication Concepts
//!
//! 1. **Task-Oriented**: Only transmit information relevant to the task
//! 2. **Learned Codecs**: Use neural networks for encoding/decoding
//! 3. **Joint Source-Channel**: Combine source and channel coding
//! 4. **Graceful Degradation**: Quality scales with channel conditions
//!
//! # Modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`codec`] | ONNX-based neural encoder / decoder with mean-pooling fallback |
//! | [`jscc`] | Joint Source-Channel Coding with channel-adaptive symbols |
//! | [`metrics`] | Cosine similarity, MSE, PSNR, top-k accuracy |
//! | [`rate_distortion`] | Rate-distortion optimisation controller |
//! | [`multimodal`] | Generic `SemanticEncode<T>` / `SemanticDecode<T>` traits |

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── New modules ──────────────────────────────────────────────────────────────

/// ONNX-based neural codec (encoder + decoder) with mean-pooling fallback.
pub mod codec;

/// Joint Source-Channel Coding: channel-adaptive encoding in one step.
pub mod jscc;

/// Semantic similarity and quality metrics.
pub mod metrics;

/// Rate-distortion optimisation controller.
pub mod rate_distortion;

/// Multi-modal trait interfaces (`SemanticEncode<T>`, `SemanticDecode<T>`).
pub mod multimodal;

/// Learned codec training loop (A18.1)
pub mod training;

/// Knowledge graph and shared semantic context (A18.2)
pub mod knowledge;

/// Goal-oriented coding for task effectiveness (A18.3)
pub mod goal;

/// Multi-user semantic broadcast (A18.9)
pub mod broadcast;

/// Integration with Service Hosting Environment (SHE) for edge semantic coding
pub mod she_integration;

/// Integration with ISAC for sensing data compression
pub mod isac_integration;

// ── Re-exports for convenience ───────────────────────────────────────────────

pub use codec::{CodecError, NeuralCodec, NeuralDecoder, NeuralEncoder};
pub use jscc::{JsccCodec, JsccConfig, JsccDecoder, JsccEncoder, JsccError, JsccSymbols};
pub use metrics::{
    cosine_similarity, evaluate as evaluate_quality, mse, psnr, top_k_accuracy, QualityMetrics,
};
pub use multimodal::{
    AudioData, ImageData, SemanticDecode, SemanticEncode, VectorDecoder, VectorEncoder, VideoData,
};
pub use rate_distortion::{
    auto_compression_ratio, RdController, RdDecision, RdMode,
};
pub use she_integration::{
    CodingMetrics, QualityParameters, SemanticCodingInput, SemanticCodingOperation,
    SemanticCodingOutput, SemanticCodingRequest, SemanticCodingResult, SheSemanticClient,
};
pub use isac_integration::{
    CompressedSensingData, CompressionStats, IsacSemanticCompressor, MapFeatureData,
    MeasurementMetadata, Position3D, SensingCompressionParams, SensingCompressionRequest,
    SensingDataPayload, Velocity3D,
};

/// Semantic feature representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticFeatures {
    /// Task ID this feature set is for
    pub task_id: u32,
    /// Feature vector (compressed representation)
    pub features: Vec<f32>,
    /// Importance weights for each feature
    pub importance: Vec<f32>,
    /// Original data dimensions
    pub original_dims: Vec<usize>,
    /// Compression ratio achieved
    pub compression_ratio: f32,
}

impl SemanticFeatures {
    /// Creates new semantic features
    pub fn new(task_id: u32, features: Vec<f32>, original_dims: Vec<usize>) -> Self {
        let num_features = features.len();
        let original_size: usize = original_dims.iter().product();
        let compression_ratio = if num_features > 0 {
            original_size as f32 / num_features as f32
        } else {
            1.0
        };

        Self {
            task_id,
            features,
            importance: vec![1.0; num_features], // Default uniform importance
            original_dims,
            compression_ratio,
        }
    }

    /// Sets importance weights
    pub fn with_importance(mut self, importance: Vec<f32>) -> Self {
        if importance.len() == self.features.len() {
            self.importance = importance;
        }
        self
    }

    /// Returns the number of features
    pub fn num_features(&self) -> usize {
        self.features.len()
    }

    /// Applies importance-based pruning (keeps top k% features)
    pub fn prune(&self, keep_ratio: f32) -> SemanticFeatures {
        let keep_count = ((self.features.len() as f32 * keep_ratio) as usize).max(1);

        // Sort indices by importance
        let mut indices: Vec<usize> = (0..self.features.len()).collect();
        indices.sort_by(|&a, &b| {
            self.importance[b]
                .partial_cmp(&self.importance[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Keep top features
        let kept_indices: Vec<usize> = indices.into_iter().take(keep_count).collect();

        let pruned_features: Vec<f32> = kept_indices.iter().map(|&i| self.features[i]).collect();
        let pruned_importance: Vec<f32> = kept_indices.iter().map(|&i| self.importance[i]).collect();

        let new_compression = self.compression_ratio / keep_ratio;

        SemanticFeatures {
            task_id: self.task_id,
            features: pruned_features,
            importance: pruned_importance,
            original_dims: self.original_dims.clone(),
            compression_ratio: new_compression,
        }
    }

    /// Applies attention-based importance weighting (A18.5)
    /// Uses a simplified self-attention mechanism to compute feature importance
    pub fn apply_attention(&mut self, temperature: f32) {
        if self.features.is_empty() {
            return;
        }

        // Compute attention scores using scaled dot-product
        let dim = self.features.len();
        let scale = 1.0 / (dim as f32).sqrt();

        // Compute self-attention: each feature attends to all others
        let mut attention_scores = vec![0.0f32; dim];

        for i in 0..dim {
            let query = self.features[i];
            let mut score = 0.0f32;

            for j in 0..dim {
                let key = self.features[j];
                // Scaled dot-product attention
                let dot_product = query * key * scale;
                let attention_weight = (dot_product / temperature).exp();
                score += attention_weight;
            }

            attention_scores[i] = score;
        }

        // Normalize attention scores
        let max_score = attention_scores
            .iter()
            .copied()
            .fold(f32::MIN, f32::max);

        if max_score > 0.0 {
            for score in &mut attention_scores {
                *score /= max_score;
            }
        }

        // Update importance with attention scores
        self.importance = attention_scores;
    }
}

/// Task type for semantic communication
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SemanticTask {
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
    /// Custom task
    Custom(u32),
}

/// Channel quality indicator
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ChannelQuality {
    /// Signal-to-noise ratio (dB)
    pub snr_db: f32,
    /// Available bandwidth (kHz)
    pub bandwidth_khz: f32,
    /// Packet error rate
    pub per: f32,
}

impl ChannelQuality {
    /// Creates a new channel quality indicator
    pub fn new(snr_db: f32, bandwidth_khz: f32, per: f32) -> Self {
        Self {
            snr_db,
            bandwidth_khz,
            per,
        }
    }

    /// Returns recommended compression ratio based on channel quality
    pub fn recommended_compression(&self) -> f32 {
        // Higher SNR = can send more features = lower compression
        // Lower SNR = need more compression
        let snr_factor = (self.snr_db / 20.0).clamp(0.1, 1.0);
        let per_factor = (1.0 - self.per).max(0.1);

        // Compression ratio: 1.0 = no compression, higher = more compression
        1.0 / (snr_factor * per_factor)
    }

    /// Returns quality category
    pub fn category(&self) -> ChannelCategory {
        if self.snr_db >= 20.0 && self.per < 0.01 {
            ChannelCategory::Excellent
        } else if self.snr_db >= 10.0 && self.per < 0.05 {
            ChannelCategory::Good
        } else if self.snr_db >= 5.0 && self.per < 0.1 {
            ChannelCategory::Fair
        } else {
            ChannelCategory::Poor
        }
    }
}

/// Channel quality category
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelCategory {
    /// Excellent channel (high SNR, low PER)
    Excellent,
    /// Good channel
    Good,
    /// Fair channel
    Fair,
    /// Poor channel (low SNR, high PER)
    Poor,
}

/// Semantic encoder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderConfig {
    /// Target compression ratio
    pub target_compression: f32,
    /// Minimum quality threshold (0.0 to 1.0)
    pub min_quality: f32,
    /// Enable adaptive compression based on channel
    pub adaptive: bool,
    /// Task-specific optimization
    pub task: SemanticTask,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            target_compression: 10.0,
            min_quality: 0.8,
            adaptive: true,
            task: SemanticTask::ImageClassification,
        }
    }
}

/// Semantic encoder (would use ONNX model in production)
pub struct SemanticEncoder {
    /// Configuration
    config: EncoderConfig,
    /// Task-specific feature extractors registered
    task_configs: HashMap<SemanticTask, TaskConfig>,
}

/// Task-specific configuration
#[derive(Debug, Clone)]
pub struct TaskConfig {
    /// Feature dimension
    pub feature_dim: usize,
    /// Importance threshold for pruning
    pub importance_threshold: f32,
}

impl SemanticEncoder {
    /// Creates a new encoder with configuration
    pub fn new(config: EncoderConfig) -> Self {
        Self {
            config,
            task_configs: HashMap::new(),
        }
    }

    /// Registers a task configuration
    pub fn register_task(&mut self, task: SemanticTask, config: TaskConfig) {
        self.task_configs.insert(task, config);
    }

    /// Encodes data into semantic features
    /// In production, this would run the ONNX encoder model
    pub fn encode(&self, data: &[f32], task: SemanticTask) -> SemanticFeatures {
        let task_config = self.task_configs.get(&task);
        let feature_dim = task_config.map(|c| c.feature_dim).unwrap_or(64);

        // Simplified encoding: mean pooling with stride
        let stride = (data.len() / feature_dim).max(1);
        let mut features = Vec::with_capacity(feature_dim);
        let mut importance = Vec::with_capacity(feature_dim);

        for i in 0..feature_dim {
            let start = i * stride;
            let end = ((i + 1) * stride).min(data.len());

            if start < data.len() {
                let chunk = &data[start..end];
                let mean: f32 = chunk.iter().sum::<f32>() / chunk.len() as f32;
                let variance: f32 = chunk.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / chunk.len() as f32;

                features.push(mean);
                // Importance based on variance (more variable = more important)
                importance.push((variance + 0.1).sqrt());
            }
        }

        // Normalize importance
        let max_importance = importance.iter().copied().fold(f32::MIN, f32::max);
        if max_importance > 0.0 {
            for imp in &mut importance {
                *imp /= max_importance;
            }
        }

        let task_id = match task {
            SemanticTask::ImageClassification => 0,
            SemanticTask::ObjectDetection => 1,
            SemanticTask::SpeechRecognition => 2,
            SemanticTask::TextUnderstanding => 3,
            SemanticTask::SensorFusion => 4,
            SemanticTask::VideoAnalytics => 5,
            SemanticTask::Custom(id) => id,
        };

        SemanticFeatures::new(task_id, features, vec![data.len()])
            .with_importance(importance)
    }

    /// Adapts compression based on channel quality
    pub fn adaptive_encode(
        &self,
        data: &[f32],
        task: SemanticTask,
        channel: &ChannelQuality,
    ) -> SemanticFeatures {
        let features = self.encode(data, task);

        if self.config.adaptive {
            let keep_ratio = 1.0 / channel.recommended_compression();
            features.prune(keep_ratio.clamp(0.1, 1.0))
        } else {
            features
        }
    }
}

impl Default for SemanticEncoder {
    fn default() -> Self {
        Self::new(EncoderConfig::default())
    }
}

/// Semantic decoder (would use ONNX model in production)
pub struct SemanticDecoder {
    /// Expected feature dimension per task
    task_dims: HashMap<SemanticTask, usize>,
}

impl SemanticDecoder {
    /// Creates a new decoder
    pub fn new() -> Self {
        Self {
            task_dims: HashMap::new(),
        }
    }

    /// Registers expected dimension for a task
    pub fn register_task(&mut self, task: SemanticTask, dim: usize) {
        self.task_dims.insert(task, dim);
    }

    /// Decodes features back to reconstruction
    /// In production, this would run the ONNX decoder model
    pub fn decode(&self, features: &SemanticFeatures) -> Vec<f32> {
        let output_size: usize = features.original_dims.iter().product();

        // Simplified decoding: nearest-neighbor upsampling
        let mut output = Vec::with_capacity(output_size);
        let stride = output_size / features.num_features().max(1);

        for &feature in features.features.iter() {
            for _ in 0..stride {
                output.push(feature);
            }
        }

        // Fill remaining if needed
        while output.len() < output_size {
            output.push(features.features.last().copied().unwrap_or(0.0));
        }

        output.truncate(output_size);
        output
    }

    /// Decodes for a specific task (task-aware decoding)
    pub fn decode_for_task(&self, features: &SemanticFeatures, _task: SemanticTask) -> TaskOutput {
        // In production, this would output task-specific results
        // For now, we just return the decoded features
        let decoded = self.decode(features);

        TaskOutput {
            task_id: features.task_id,
            result: decoded,
            confidence: features.importance.iter().sum::<f32>() / features.importance.len() as f32,
        }
    }
}

impl Default for SemanticDecoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Task output from semantic decoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskOutput {
    /// Task ID
    pub task_id: u32,
    /// Task result (interpretation depends on task)
    pub result: Vec<f32>,
    /// Confidence score
    pub confidence: f32,
}

/// Semantic communication message
#[derive(Debug)]
pub enum SemanticMessage {
    /// Encode data using task-specific compression
    Encode {
        /// Raw data to encode
        data: Vec<f32>,
        /// Semantic task type for encoding
        task: SemanticTask,
        /// Response channel for async result
        response_tx: Option<tokio::sync::oneshot::Sender<SemanticResponse>>,
    },
    /// Adaptive encode with channel-aware compression
    AdaptiveEncode {
        /// Raw data to encode
        data: Vec<f32>,
        /// Semantic task type for encoding
        task: SemanticTask,
        /// Current channel quality for adaptation
        channel: ChannelQuality,
        /// Response channel for async result
        response_tx: Option<tokio::sync::oneshot::Sender<SemanticResponse>>,
    },
    /// Decode semantic features back to data
    Decode {
        /// Encoded semantic features
        features: SemanticFeatures,
        /// Response channel for async result
        response_tx: Option<tokio::sync::oneshot::Sender<SemanticResponse>>,
    },
}

/// Semantic communication response
#[derive(Debug)]
pub enum SemanticResponse {
    /// Encoded features
    Encoded(SemanticFeatures),
    /// Decoded output
    Decoded(TaskOutput),
    /// Error
    Error(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_features_creation() {
        let features = SemanticFeatures::new(1, vec![0.1, 0.2, 0.3], vec![100]);

        assert_eq!(features.num_features(), 3);
        assert!(features.compression_ratio > 1.0);
    }

    #[test]
    fn test_semantic_features_pruning() {
        let features = SemanticFeatures::new(1, vec![0.1, 0.2, 0.3, 0.4, 0.5], vec![100])
            .with_importance(vec![0.9, 0.1, 0.5, 0.8, 0.3]);

        let pruned = features.prune(0.4); // Keep 40% = 2 features

        assert_eq!(pruned.num_features(), 2);
        // Should keep highest importance features (indices 0 and 3)
    }

    #[test]
    fn test_channel_quality() {
        let excellent = ChannelQuality::new(25.0, 1000.0, 0.001);
        assert_eq!(excellent.category(), ChannelCategory::Excellent);

        let poor = ChannelQuality::new(3.0, 100.0, 0.2);
        assert_eq!(poor.category(), ChannelCategory::Poor);
    }

    #[test]
    fn test_encoder_decode_roundtrip() {
        let encoder = SemanticEncoder::default();
        let decoder = SemanticDecoder::default();

        // Generate some test data
        let data: Vec<f32> = (0..256).map(|i| (i as f32) / 255.0).collect();

        // Encode
        let features = encoder.encode(&data, SemanticTask::ImageClassification);
        assert!(features.num_features() > 0);
        assert!(features.compression_ratio > 1.0);

        // Decode
        let output = decoder.decode(&features);
        assert_eq!(output.len(), data.len());
    }

    #[test]
    fn test_adaptive_encoding() {
        let encoder = SemanticEncoder::new(EncoderConfig {
            adaptive: true,
            ..Default::default()
        });

        let data: Vec<f32> = (0..256).map(|i| (i as f32) / 255.0).collect();

        // Good channel - less compression
        let good_channel = ChannelQuality::new(20.0, 1000.0, 0.01);
        let good_features = encoder.adaptive_encode(&data, SemanticTask::ImageClassification, &good_channel);

        // Poor channel - more compression
        let poor_channel = ChannelQuality::new(5.0, 100.0, 0.1);
        let poor_features = encoder.adaptive_encode(&data, SemanticTask::ImageClassification, &poor_channel);

        // Poor channel should result in fewer features
        assert!(poor_features.num_features() <= good_features.num_features());
    }

    #[test]
    fn test_task_aware_decoding() {
        let encoder = SemanticEncoder::default();
        let decoder = SemanticDecoder::default();

        let data: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let features = encoder.encode(&data, SemanticTask::SensorFusion);

        let output = decoder.decode_for_task(&features, SemanticTask::SensorFusion);

        // SensorFusion maps to task_id 4
        assert_eq!(output.task_id, 4);
        assert!(output.confidence > 0.0);
    }

    // Test for A18.5: Attention-based importance

    #[test]
    fn test_attention_based_importance() {
        let mut features = SemanticFeatures::new(
            0,
            vec![1.0, 0.5, 2.0, 0.3, 1.5],
            vec![5],
        );

        // Apply attention
        features.apply_attention(1.0);

        // Check that importance was updated
        assert_eq!(features.importance.len(), 5);

        // All importance scores should be between 0 and 1 (normalized)
        for &imp in &features.importance {
            assert!((0.0..=1.0).contains(&imp));
        }
    }

    #[test]
    fn test_attention_temperature() {
        let mut features1 = SemanticFeatures::new(
            0,
            vec![1.0, 0.5, 2.0, 0.3, 1.5],
            vec![5],
        );

        let mut features2 = features1.clone();

        // Low temperature (sharper attention)
        features1.apply_attention(0.1);

        // High temperature (softer attention)
        features2.apply_attention(10.0);

        // Low temperature should have more variation in importance
        let var1: f32 = features1.importance.iter().map(|x| x * x).sum::<f32>();
        let var2: f32 = features2.importance.iter().map(|x| x * x).sum::<f32>();

        assert!(var1 != var2);
    }
}
