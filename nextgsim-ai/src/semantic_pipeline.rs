//! Semantic communication pipeline
//!
//! This module implements encoder/decoder execution for semantic communication,
//! enabling intelligent data compression and transmission based on semantic meaning
//! rather than bit-level accuracy.

#![allow(missing_docs)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use thiserror::Error;
use tracing::{debug, info};

use crate::config::SemanticConfig;
use crate::error::ModelError;
use crate::inference::{InferenceEngine, OnnxEngine};
use crate::tensor::TensorData;

/// Errors that can occur during semantic communication
#[derive(Error, Debug)]
pub enum SemanticError {
    /// Encoder error
    #[error("Encoder error: {reason}")]
    EncoderError { reason: String },

    /// Decoder error
    #[error("Decoder error: {reason}")]
    DecoderError { reason: String },

    /// Quality threshold not met
    #[error("Quality threshold not met: expected {expected}, got {actual}")]
    QualityBelowThreshold { expected: f32, actual: f32 },

    /// Model not loaded
    #[error("Model not loaded: {model_type}")]
    ModelNotLoaded { model_type: String },

    /// Invalid input
    #[error("Invalid input: {reason}")]
    InvalidInput { reason: String },

    /// Compression failed
    #[error("Compression failed: {reason}")]
    CompressionFailed { reason: String },
}

/// Encoded semantic representation of data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticEncoding {
    /// Compressed semantic features
    pub features: TensorData,
    /// Original data shape
    pub original_shape: Vec<i64>,
    /// Compression ratio achieved
    pub compression_ratio: f32,
    /// Quality score (0.0 to 1.0)
    pub quality_score: f32,
    /// Metadata about the encoding
    pub metadata: HashMap<String, String>,
}

/// Decoded semantic data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticDecoding {
    /// Reconstructed data
    pub data: TensorData,
    /// Quality score (0.0 to 1.0)
    pub quality_score: f32,
    /// Metadata from the encoding
    pub metadata: HashMap<String, String>,
}

/// Semantic communication pipeline
///
/// Provides end-to-end semantic encoding and decoding for intelligent
/// data compression in 6G networks.
pub struct SemanticPipeline {
    /// Configuration
    config: SemanticConfig,
    /// Encoder model
    encoder: Option<Box<dyn InferenceEngine>>,
    /// Decoder model
    decoder: Option<Box<dyn InferenceEngine>>,
    /// Encoding statistics
    total_encodings: usize,
    /// Total compression ratio sum (for averaging)
    total_compression: f32,
    /// Total quality score sum (for averaging)
    total_quality: f32,
}

impl SemanticPipeline {
    /// Creates a new semantic pipeline with the given configuration
    pub fn new(config: SemanticConfig) -> Self {
        Self {
            config,
            encoder: None,
            decoder: None,
            total_encodings: 0,
            total_compression: 0.0,
            total_quality: 0.0,
        }
    }

    /// Loads the encoder model
    pub fn load_encoder(&mut self, path: &Path) -> Result<(), ModelError> {
        info!("Loading semantic encoder model from {:?}", path);

        let mut engine = OnnxEngine::new(crate::config::ExecutionProvider::Cpu)?;
        engine.load_model(path)?;

        // Warmup the model
        if let Err(e) = engine.warmup() {
            debug!("Encoder warmup warning: {:?}", e);
        }

        self.encoder = Some(Box::new(engine));
        info!("Semantic encoder loaded successfully");
        Ok(())
    }

    /// Loads the decoder model
    pub fn load_decoder(&mut self, path: &Path) -> Result<(), ModelError> {
        info!("Loading semantic decoder model from {:?}", path);

        let mut engine = OnnxEngine::new(crate::config::ExecutionProvider::Cpu)?;
        engine.load_model(path)?;

        // Warmup the model
        if let Err(e) = engine.warmup() {
            debug!("Decoder warmup warning: {:?}", e);
        }

        self.decoder = Some(Box::new(engine));
        info!("Semantic decoder loaded successfully");
        Ok(())
    }

    /// Encodes data using the semantic encoder
    ///
    /// Returns compressed semantic features along with quality metrics
    pub fn encode(&mut self, input: &TensorData) -> Result<SemanticEncoding, SemanticError> {
        let encoder = self.encoder.as_ref().ok_or_else(|| SemanticError::ModelNotLoaded {
            model_type: "encoder".to_string(),
        })?;

        debug!("Encoding data with shape {:?}", input.shape().dims());

        // Run encoder inference
        let encoded = encoder.infer(input).map_err(|e| SemanticError::EncoderError {
            reason: format!("Inference failed: {e}"),
        })?;

        // Calculate compression ratio
        let original_size = input.len();
        let compressed_size = encoded.len();
        let compression_ratio = if original_size > 0 {
            compressed_size as f32 / original_size as f32
        } else {
            1.0
        };

        // Estimate quality (placeholder - would use actual quality metric in production)
        let quality_score = self.estimate_quality(input, &encoded);

        // Check if quality meets threshold
        if quality_score < self.config.quality_threshold {
            return Err(SemanticError::QualityBelowThreshold {
                expected: self.config.quality_threshold,
                actual: quality_score,
            });
        }

        // Update statistics
        self.total_encodings += 1;
        self.total_compression += compression_ratio;
        self.total_quality += quality_score;

        let mut metadata = HashMap::new();
        metadata.insert("encoder_version".to_string(), "1.0".to_string());
        metadata.insert("original_dtype".to_string(), input.dtype().to_string());

        info!(
            "Encoding complete: compression {:.2}%, quality {:.2}",
            compression_ratio * 100.0,
            quality_score
        );

        Ok(SemanticEncoding {
            features: encoded,
            original_shape: input.shape().dims().to_vec(),
            compression_ratio,
            quality_score,
            metadata,
        })
    }

    /// Decodes semantic features back to original data
    pub fn decode(&mut self, encoding: &SemanticEncoding) -> Result<SemanticDecoding, SemanticError> {
        let decoder = self.decoder.as_ref().ok_or_else(|| SemanticError::ModelNotLoaded {
            model_type: "decoder".to_string(),
        })?;

        debug!("Decoding features to shape {:?}", encoding.original_shape);

        // Run decoder inference
        let decoded = decoder.infer(&encoding.features).map_err(|e| SemanticError::DecoderError {
            reason: format!("Inference failed: {e}"),
        })?;

        // Validate output shape matches original
        if decoded.shape().dims() != encoding.original_shape.as_slice() {
            debug!(
                "Shape mismatch: expected {:?}, got {:?}",
                encoding.original_shape,
                decoded.shape().dims()
            );
        }

        info!(
            "Decoding complete: reconstructed {} elements",
            decoded.len()
        );

        Ok(SemanticDecoding {
            data: decoded,
            quality_score: encoding.quality_score,
            metadata: encoding.metadata.clone(),
        })
    }

    /// Performs end-to-end semantic communication (encode + decode)
    pub fn process(
        &mut self,
        input: &TensorData,
    ) -> Result<(SemanticEncoding, SemanticDecoding), SemanticError> {
        let encoding = self.encode(input)?;
        let decoding = self.decode(&encoding)?;
        Ok((encoding, decoding))
    }

    /// Estimates quality of encoding (placeholder implementation)
    ///
    /// In production, this would compute metrics like:
    /// - PSNR (Peak Signal-to-Noise Ratio)
    /// - SSIM (Structural Similarity Index)
    /// - Perceptual quality metrics
    fn estimate_quality(&self, _original: &TensorData, _encoded: &TensorData) -> f32 {
        // Placeholder: return a high quality score
        // In production, this would compare the decoded output with the original
        0.95
    }

    /// Returns average compression ratio across all encodings
    pub fn avg_compression_ratio(&self) -> f32 {
        if self.total_encodings > 0 {
            self.total_compression / self.total_encodings as f32
        } else {
            0.0
        }
    }

    /// Returns average quality score across all encodings
    pub fn avg_quality_score(&self) -> f32 {
        if self.total_encodings > 0 {
            self.total_quality / self.total_encodings as f32
        } else {
            0.0
        }
    }

    /// Returns the total number of encodings performed
    pub fn total_encodings(&self) -> usize {
        self.total_encodings
    }

    /// Returns true if both encoder and decoder are loaded
    pub fn is_ready(&self) -> bool {
        self.encoder.is_some() && self.decoder.is_some()
    }

    /// Returns the configuration
    pub fn config(&self) -> &SemanticConfig {
        &self.config
    }

    /// Resets statistics
    pub fn reset_statistics(&mut self) {
        self.total_encodings = 0;
        self.total_compression = 0.0;
        self.total_quality = 0.0;
    }
}

/// Builder for creating a semantic pipeline with custom configuration
pub struct SemanticPipelineBuilder {
    config: SemanticConfig,
    encoder_path: Option<std::path::PathBuf>,
    decoder_path: Option<std::path::PathBuf>,
}

impl SemanticPipelineBuilder {
    /// Creates a new builder
    pub fn new() -> Self {
        Self {
            config: SemanticConfig::default(),
            encoder_path: None,
            decoder_path: None,
        }
    }

    /// Sets the configuration
    pub fn with_config(mut self, config: SemanticConfig) -> Self {
        self.config = config;
        self
    }

    /// Sets the encoder model path
    pub fn with_encoder(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.encoder_path = Some(path.into());
        self
    }

    /// Sets the decoder model path
    pub fn with_decoder(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.decoder_path = Some(path.into());
        self
    }

    /// Builds the semantic pipeline
    pub fn build(self) -> Result<SemanticPipeline, ModelError> {
        let mut pipeline = SemanticPipeline::new(self.config);

        if let Some(encoder_path) = self.encoder_path {
            pipeline.load_encoder(&encoder_path)?;
        }

        if let Some(decoder_path) = self.decoder_path {
            pipeline.load_decoder(&decoder_path)?;
        }

        Ok(pipeline)
    }
}

impl Default for SemanticPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let config = SemanticConfig::default();
        let pipeline = SemanticPipeline::new(config);
        assert!(!pipeline.is_ready());
        assert_eq!(pipeline.total_encodings(), 0);
    }

    #[test]
    fn test_pipeline_statistics() {
        let mut pipeline = SemanticPipeline::new(SemanticConfig::default());

        // Simulate some encodings
        pipeline.total_encodings = 3;
        pipeline.total_compression = 0.3; // 0.1 each
        pipeline.total_quality = 2.85; // 0.95 each

        assert_eq!(pipeline.total_encodings(), 3);
        assert!((pipeline.avg_compression_ratio() - 0.1).abs() < 0.01);
        assert!((pipeline.avg_quality_score() - 0.95).abs() < 0.01);

        pipeline.reset_statistics();
        assert_eq!(pipeline.total_encodings(), 0);
        assert_eq!(pipeline.avg_compression_ratio(), 0.0);
    }

    #[test]
    fn test_semantic_encoding_creation() {
        let features = TensorData::float32(vec![1.0, 2.0, 3.0], vec![3]);
        let original_shape = vec![1, 10, 10];

        let encoding = SemanticEncoding {
            features,
            original_shape: original_shape.clone(),
            compression_ratio: 0.03, // 3 / 100
            quality_score: 0.95,
            metadata: HashMap::new(),
        };

        assert_eq!(encoding.original_shape, original_shape);
        assert!((encoding.compression_ratio - 0.03).abs() < 0.01);
        assert!((encoding.quality_score - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_builder_pattern() {
        let builder = SemanticPipelineBuilder::new()
            .with_config(SemanticConfig {
                enabled: true,
                compression_ratio: 0.1,
                quality_threshold: 0.9,
                ..Default::default()
            });

        // Can't actually build without valid model paths, but we can test the builder
        assert!(builder.config.enabled);
        assert!((builder.config.compression_ratio - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_encode_without_model() {
        let mut pipeline = SemanticPipeline::new(SemanticConfig::default());
        let input = TensorData::float32(vec![1.0, 2.0, 3.0], vec![3]);

        let result = pipeline.encode(&input);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            SemanticError::ModelNotLoaded { .. }
        ));
    }

    #[test]
    fn test_decode_without_model() {
        let mut pipeline = SemanticPipeline::new(SemanticConfig::default());
        let encoding = SemanticEncoding {
            features: TensorData::float32(vec![1.0, 2.0, 3.0], vec![3]),
            original_shape: vec![10],
            compression_ratio: 0.3,
            quality_score: 0.95,
            metadata: HashMap::new(),
        };

        let result = pipeline.decode(&encoding);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            SemanticError::ModelNotLoaded { .. }
        ));
    }
}
