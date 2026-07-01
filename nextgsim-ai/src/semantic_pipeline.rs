//! Semantic communication pipeline
//!
//! This module implements encoder/decoder execution for semantic communication,
//! enabling intelligent data compression and transmission based on semantic meaning
//! rather than bit-level accuracy.

#![allow(missing_docs)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Once;
use thiserror::Error;
use tracing::{debug, info, warn};

/// Emits the "quality not measured" warning at most once per process.
static QUALITY_UNMEASURED_WARN: Once = Once::new();

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
    /// Measured quality score in `[0.0, 1.0]`, or `None` when the pipeline could
    /// not measure quality (e.g. no model / no round-trip available). `None` is
    /// distinct from "measured high" and must not be treated as a passing KPI.
    pub quality_score: Option<f32>,
    /// Metadata about the encoding
    pub metadata: HashMap<String, String>,
}

/// Decoded semantic data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticDecoding {
    /// Reconstructed data
    pub data: TensorData,
    /// Measured quality score in `[0.0, 1.0]`, or `None` when not measured
    /// (carried over from the encoding).
    pub quality_score: Option<f32>,
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
    /// Total measured quality score sum (for averaging)
    total_quality: f32,
    /// Count of encodings whose quality was actually measured (the divisor for
    /// the average — unmeasured encodings are excluded so the average is not
    /// diluted toward zero).
    total_measured: usize,
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
            total_measured: 0,
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
        let encoder = self
            .encoder
            .as_ref()
            .ok_or_else(|| SemanticError::ModelNotLoaded {
                model_type: "encoder".to_string(),
            })?;

        debug!("Encoding data with shape {:?}", input.shape().dims());

        // Run encoder inference
        let encoded = encoder
            .infer(input)
            .map_err(|e| SemanticError::EncoderError {
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

        // Measure quality if possible (None = not measurable here).
        let quality_score = self.estimate_quality(input, &encoded);

        // Only enforce the quality gate on a MEASURED score. When quality is not
        // measured we skip the gate (with a one-time warning) rather than letting
        // a fabricated constant silently pass it.
        match quality_score {
            Some(q) if q < self.config.quality_threshold => {
                return Err(SemanticError::QualityBelowThreshold {
                    expected: self.config.quality_threshold,
                    actual: q,
                });
            }
            None => {
                QUALITY_UNMEASURED_WARN.call_once(|| {
                    warn!(
                        "semantic encode: quality not measured (no decoder round-trip / \
                         comparable tensor) — skipping quality-threshold gate"
                    );
                });
            }
            _ => {}
        }

        // Update statistics. Only measured scores feed the quality average.
        self.total_encodings += 1;
        self.total_compression += compression_ratio;
        if let Some(q) = quality_score {
            self.total_quality += q;
            self.total_measured += 1;
        }

        let mut metadata = HashMap::new();
        metadata.insert("encoder_version".to_string(), "1.0".to_string());
        metadata.insert("original_dtype".to_string(), input.dtype().to_string());

        match quality_score {
            Some(q) => info!(
                "Encoding complete: compression {:.2}%, quality {:.2}",
                compression_ratio * 100.0,
                q
            ),
            None => info!(
                "Encoding complete: compression {:.2}%, quality not measured",
                compression_ratio * 100.0
            ),
        }

        Ok(SemanticEncoding {
            features: encoded,
            original_shape: input.shape().dims().to_vec(),
            compression_ratio,
            quality_score,
            metadata,
        })
    }

    /// Decodes semantic features back to original data
    pub fn decode(
        &mut self,
        encoding: &SemanticEncoding,
    ) -> Result<SemanticDecoding, SemanticError> {
        let decoder = self
            .decoder
            .as_ref()
            .ok_or_else(|| SemanticError::ModelNotLoaded {
                model_type: "decoder".to_string(),
            })?;

        debug!("Decoding features to shape {:?}", encoding.original_shape);

        // Run decoder inference
        let decoded =
            decoder
                .infer(&encoding.features)
                .map_err(|e| SemanticError::DecoderError {
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

    /// Estimates the quality of an encoding, or `None` when it cannot be
    /// measured.
    ///
    /// Quality is only *measurable* when we can compare like-for-like, i.e. when
    /// the encoded tensor has the same element count as the input (e.g. an
    /// identity / round-trip path), in which case we return a cosine-similarity
    /// score mapped to `[0, 1]`. For a genuinely compressing model (different
    /// dimensionality) or no model, quality cannot be measured here without a
    /// decoder round-trip, so we return `None` ("not measured") rather than a
    /// fabricated constant. (A fuller implementation would compute PSNR/SSIM on
    /// the decoded reconstruction.)
    fn estimate_quality(&self, original: &TensorData, encoded: &TensorData) -> Option<f32> {
        let o: Vec<f32> = original.as_f32_array()?.iter().copied().collect();
        let e: Vec<f32> = encoded.as_f32_array()?.iter().copied().collect();
        if o.is_empty() || o.len() != e.len() {
            return None;
        }
        Some(cosine_quality(&o, &e))
    }

    /// Returns average compression ratio across all encodings
    pub fn avg_compression_ratio(&self) -> f32 {
        if self.total_encodings > 0 {
            self.total_compression / self.total_encodings as f32
        } else {
            0.0
        }
    }

    /// Returns the average of the *measured* quality scores, or `0.0` when no
    /// encoding has had its quality measured. Encodings whose quality was not
    /// measured are excluded from the divisor (they neither raise nor dilute the
    /// average).
    pub fn avg_quality_score(&self) -> f32 {
        if self.total_measured > 0 {
            self.total_quality / self.total_measured as f32
        } else {
            0.0
        }
    }

    /// Number of encodings whose quality was actually measured.
    pub fn measured_quality_count(&self) -> usize {
        self.total_measured
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
        self.total_measured = 0;
    }
}

/// Cosine similarity of two equal-length vectors mapped to `[0, 1]`
/// (`0.5 * (1 + cos)`). Returns `0.5` (orthogonal/degenerate) when either
/// vector has zero norm.
fn cosine_quality(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.5;
    }
    let cos = (dot / (na * nb)).clamp(-1.0, 1.0);
    0.5 * (1.0 + cos)
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

        // Simulate some encodings (all three had their quality measured).
        pipeline.total_encodings = 3;
        pipeline.total_compression = 0.3; // 0.1 each
        pipeline.total_quality = 2.7; // 0.9 each
        pipeline.total_measured = 3;

        assert_eq!(pipeline.total_encodings(), 3);
        assert!((pipeline.avg_compression_ratio() - 0.1).abs() < 0.01);
        // Average is over the MEASURED count, not the total encodings.
        assert!((pipeline.avg_quality_score() - 0.9).abs() < 0.01);
        assert_eq!(pipeline.measured_quality_count(), 3);

        pipeline.reset_statistics();
        assert_eq!(pipeline.total_encodings(), 0);
        assert_eq!(pipeline.avg_compression_ratio(), 0.0);
        assert_eq!(pipeline.measured_quality_count(), 0);
        // No measured scores -> average is 0.0, not a fabricated constant.
        assert_eq!(pipeline.avg_quality_score(), 0.0);
    }

    #[test]
    fn estimate_quality_measures_when_comparable_else_none() {
        let pipeline = SemanticPipeline::new(SemanticConfig::default());
        let input = TensorData::float32(vec![1.0, 2.0, 3.0, 4.0], vec![4]);

        // Same length but perturbed -> a measured score strictly inside (0, 1)
        // (no longer the hardcoded 0.95 constant).
        let perturbed = TensorData::float32(vec![1.0, 2.0, 3.0, -4.0], vec![4]);
        let q = pipeline
            .estimate_quality(&input, &perturbed)
            .expect("comparable tensors should be measurable");
        assert!(
            q > 0.0 && q < 1.0,
            "expected measured quality in (0,1), got {q}"
        );

        // A compressing model (different element count) is not measurable here.
        let compressed = TensorData::float32(vec![1.0], vec![1]);
        assert_eq!(
            pipeline.estimate_quality(&input, &compressed),
            None,
            "non-comparable shapes must yield not-measured"
        );

        // Identical tensors -> cosine 1 -> quality 1.0 (computed, not a constant).
        let same = TensorData::float32(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        assert_eq!(pipeline.estimate_quality(&input, &same), Some(1.0));
    }

    #[test]
    fn test_semantic_encoding_creation() {
        let features = TensorData::float32(vec![1.0, 2.0, 3.0], vec![3]);
        let original_shape = vec![1, 10, 10];

        let encoding = SemanticEncoding {
            features,
            original_shape: original_shape.clone(),
            compression_ratio: 0.03, // 3 / 100
            quality_score: Some(0.95),
            metadata: HashMap::new(),
        };

        assert_eq!(encoding.original_shape, original_shape);
        assert!((encoding.compression_ratio - 0.03).abs() < 0.01);
        assert!((encoding.quality_score.unwrap() - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_builder_pattern() {
        let builder = SemanticPipelineBuilder::new().with_config(SemanticConfig {
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
            quality_score: Some(0.95),
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
