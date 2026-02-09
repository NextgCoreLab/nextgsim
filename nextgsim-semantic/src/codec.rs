//! Neural codec interface wrapping nextgsim-ai's InferenceEngine
//!
//! Provides ONNX-based neural encoding and decoding for semantic features.
//! When no ONNX models are loaded, falls back to the existing mean-pooling encoder.

use std::path::Path;

use tracing::{debug, info};

use nextgsim_ai::inference::{InferenceEngine, OnnxEngine};
use nextgsim_ai::config::ExecutionProvider;
use nextgsim_ai::error::ModelError;
use nextgsim_ai::tensor::TensorData;

use crate::{SemanticFeatures, SemanticTask};

/// Error type for neural codec operations
#[derive(Debug, thiserror::Error)]
pub enum CodecError {
    /// Model loading failed
    #[error("Failed to load codec model: {0}")]
    ModelLoad(#[from] ModelError),
    /// Inference failed
    #[error("Codec inference failed: {0}")]
    Inference(#[from] nextgsim_ai::error::InferenceError),
    /// Invalid input dimensions
    #[error("Invalid input dimensions: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension
        expected: usize,
        /// Actual dimension
        actual: usize,
    },
    /// Codec not ready (no model loaded and no fallback available)
    #[error("Codec not ready: {reason}")]
    NotReady {
        /// Reason the codec is not ready
        reason: String,
    },
}

/// Neural encoder that compresses feature vectors using an ONNX model.
///
/// Falls back to mean-pooling when no model is loaded.
pub struct NeuralEncoder {
    /// ONNX inference engine for the encoder model
    engine: OnnxEngine,
    /// Target feature dimension for the compressed representation
    target_dim: usize,
    /// Whether the ONNX model is loaded and ready
    model_loaded: bool,
}

impl NeuralEncoder {
    /// Creates a new neural encoder with the given target dimension.
    ///
    /// The encoder starts without a model; call `load_model` to enable
    /// ONNX-based encoding. Until then, mean-pooling fallback is used.
    ///
    /// # Errors
    /// Returns `CodecError::ModelLoad` if the ONNX engine cannot be initialized.
    pub fn new(target_dim: usize) -> Result<Self, CodecError> {
        let engine = OnnxEngine::new(ExecutionProvider::Cpu)?;
        Ok(Self {
            engine,
            target_dim,
            model_loaded: false,
        })
    }

    /// Creates a neural encoder with a custom execution provider.
    ///
    /// # Errors
    /// Returns `CodecError::ModelLoad` if the ONNX engine cannot be initialized.
    pub fn with_provider(target_dim: usize, provider: ExecutionProvider) -> Result<Self, CodecError> {
        let engine = OnnxEngine::new(provider)?;
        Ok(Self {
            engine,
            target_dim,
            model_loaded: false,
        })
    }

    /// Loads an ONNX encoder model from the given file path.
    ///
    /// The model should accept a 2D tensor `[1, input_dim]` of f32 values
    /// and produce a 2D tensor `[1, compressed_dim]` of f32 values.
    ///
    /// # Errors
    /// Returns `CodecError::ModelLoad` if the model file cannot be loaded.
    pub fn load_model(&mut self, path: &Path) -> Result<(), CodecError> {
        info!("Loading neural encoder model from {:?}", path);
        self.engine.load_model(path)?;
        self.model_loaded = true;
        info!("Neural encoder model loaded successfully");
        Ok(())
    }

    /// Returns whether the ONNX model is loaded.
    pub fn is_model_loaded(&self) -> bool {
        self.model_loaded
    }

    /// Returns the target compressed dimension.
    pub fn target_dim(&self) -> usize {
        self.target_dim
    }

    /// Encodes raw feature data into a compressed representation.
    ///
    /// If an ONNX model is loaded, runs the model. Otherwise, falls back
    /// to mean-pooling compression (the original behavior).
    ///
    /// # Errors
    /// Returns `CodecError::Inference` if ONNX inference fails.
    pub fn encode(&self, data: &[f32], task: SemanticTask) -> Result<SemanticFeatures, CodecError> {
        if self.model_loaded {
            self.encode_neural(data, task)
        } else {
            debug!("No encoder model loaded, using mean-pooling fallback");
            Ok(self.encode_fallback(data, task))
        }
    }

    /// Runs the ONNX encoder model on the input data.
    fn encode_neural(&self, data: &[f32], task: SemanticTask) -> Result<SemanticFeatures, CodecError> {
        let input = TensorData::float32(data.to_vec(), vec![1i64, data.len() as i64]);
        let output = self.engine.infer(&input)?;

        let compressed = output
            .as_f32_slice()
            .ok_or_else(|| CodecError::NotReady {
                reason: "Encoder model did not produce f32 output".to_string(),
            })?
            .to_vec();

        let task_id = task_to_id(task);
        let features = SemanticFeatures::new(task_id, compressed, vec![data.len()]);
        Ok(features)
    }

    /// Mean-pooling fallback encoder (matches the original `SemanticEncoder::encode` logic).
    fn encode_fallback(&self, data: &[f32], task: SemanticTask) -> SemanticFeatures {
        let feature_dim = self.target_dim;
        let stride = (data.len() / feature_dim).max(1);
        let mut features = Vec::with_capacity(feature_dim);
        let mut importance = Vec::with_capacity(feature_dim);

        for i in 0..feature_dim {
            let start = i * stride;
            let end = ((i + 1) * stride).min(data.len());

            if start < data.len() {
                let chunk = &data[start..end];
                let mean: f32 = chunk.iter().sum::<f32>() / chunk.len() as f32;
                let variance: f32 =
                    chunk.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / chunk.len() as f32;

                features.push(mean);
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

        let task_id = task_to_id(task);
        SemanticFeatures::new(task_id, features, vec![data.len()]).with_importance(importance)
    }
}

/// Neural decoder that reconstructs feature vectors from compressed representations.
///
/// Falls back to nearest-neighbor upsampling when no model is loaded.
pub struct NeuralDecoder {
    /// ONNX inference engine for the decoder model
    engine: OnnxEngine,
    /// Whether the ONNX model is loaded and ready
    model_loaded: bool,
}

impl NeuralDecoder {
    /// Creates a new neural decoder.
    ///
    /// The decoder starts without a model; call `load_model` to enable
    /// ONNX-based decoding. Until then, nearest-neighbor fallback is used.
    ///
    /// # Errors
    /// Returns `CodecError::ModelLoad` if the ONNX engine cannot be initialized.
    pub fn new() -> Result<Self, CodecError> {
        let engine = OnnxEngine::new(ExecutionProvider::Cpu)?;
        Ok(Self {
            engine,
            model_loaded: false,
        })
    }

    /// Creates a neural decoder with a custom execution provider.
    ///
    /// # Errors
    /// Returns `CodecError::ModelLoad` if the ONNX engine cannot be initialized.
    pub fn with_provider(provider: ExecutionProvider) -> Result<Self, CodecError> {
        let engine = OnnxEngine::new(provider)?;
        Ok(Self {
            engine,
            model_loaded: false,
        })
    }

    /// Loads an ONNX decoder model from the given file path.
    ///
    /// The model should accept a 2D tensor `[1, compressed_dim]` of f32 values
    /// and produce a 2D tensor `[1, output_dim]` of f32 values.
    ///
    /// # Errors
    /// Returns `CodecError::ModelLoad` if the model file cannot be loaded.
    pub fn load_model(&mut self, path: &Path) -> Result<(), CodecError> {
        info!("Loading neural decoder model from {:?}", path);
        self.engine.load_model(path)?;
        self.model_loaded = true;
        info!("Neural decoder model loaded successfully");
        Ok(())
    }

    /// Returns whether the ONNX model is loaded.
    pub fn is_model_loaded(&self) -> bool {
        self.model_loaded
    }

    /// Decodes compressed features back to the original dimension.
    ///
    /// If an ONNX model is loaded, runs the model. Otherwise, falls back
    /// to nearest-neighbor upsampling (the original behavior).
    ///
    /// # Errors
    /// Returns `CodecError::Inference` if ONNX inference fails.
    pub fn decode(&self, features: &SemanticFeatures) -> Result<Vec<f32>, CodecError> {
        if self.model_loaded {
            self.decode_neural(features)
        } else {
            debug!("No decoder model loaded, using nearest-neighbor fallback");
            Ok(self.decode_fallback(features))
        }
    }

    /// Runs the ONNX decoder model on the compressed features.
    fn decode_neural(&self, features: &SemanticFeatures) -> Result<Vec<f32>, CodecError> {
        let input = TensorData::float32(
            features.features.clone(),
            vec![1i64, features.features.len() as i64],
        );
        let output = self.engine.infer(&input)?;

        let decoded = output
            .as_f32_slice()
            .ok_or_else(|| CodecError::NotReady {
                reason: "Decoder model did not produce f32 output".to_string(),
            })?
            .to_vec();

        Ok(decoded)
    }

    /// Nearest-neighbor upsampling fallback (matches the original `SemanticDecoder::decode` logic).
    fn decode_fallback(&self, features: &SemanticFeatures) -> Vec<f32> {
        let output_size: usize = features.original_dims.iter().product();
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
}

/// Combined neural codec holding both an encoder and a decoder.
///
/// Provides a convenient single-object interface for encode/decode round-trips.
pub struct NeuralCodec {
    /// The neural encoder
    pub encoder: NeuralEncoder,
    /// The neural decoder
    pub decoder: NeuralDecoder,
}

impl NeuralCodec {
    /// Creates a new neural codec with the given target compressed dimension.
    ///
    /// # Errors
    /// Returns `CodecError` if engine initialization fails.
    pub fn new(target_dim: usize) -> Result<Self, CodecError> {
        Ok(Self {
            encoder: NeuralEncoder::new(target_dim)?,
            decoder: NeuralDecoder::new()?,
        })
    }

    /// Loads encoder and decoder ONNX models from file paths.
    ///
    /// # Errors
    /// Returns `CodecError::ModelLoad` if either model fails to load.
    pub fn load_models(
        &mut self,
        encoder_path: &Path,
        decoder_path: &Path,
    ) -> Result<(), CodecError> {
        self.encoder.load_model(encoder_path)?;
        self.decoder.load_model(decoder_path)?;
        Ok(())
    }

    /// Returns whether both models are loaded and ready.
    pub fn is_ready(&self) -> bool {
        self.encoder.is_model_loaded() && self.decoder.is_model_loaded()
    }

    /// Encodes data, falling back to mean-pooling if no model is loaded.
    ///
    /// # Errors
    /// Returns `CodecError` on encoding failure.
    pub fn encode(&self, data: &[f32], task: SemanticTask) -> Result<SemanticFeatures, CodecError> {
        self.encoder.encode(data, task)
    }

    /// Decodes features, falling back to nearest-neighbor if no model is loaded.
    ///
    /// # Errors
    /// Returns `CodecError` on decoding failure.
    pub fn decode(&self, features: &SemanticFeatures) -> Result<Vec<f32>, CodecError> {
        self.decoder.decode(features)
    }
}

/// Converts a `SemanticTask` to its numeric task ID.
pub fn task_to_id(task: SemanticTask) -> u32 {
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
    fn test_neural_encoder_fallback() {
        let encoder = NeuralEncoder::new(32).expect("Failed to create encoder");
        assert!(!encoder.is_model_loaded());

        let data: Vec<f32> = (0..256).map(|i| i as f32 / 255.0).collect();
        let features = encoder
            .encode(&data, SemanticTask::ImageClassification)
            .expect("Encoding failed");

        assert!(features.num_features() > 0);
        assert!(features.compression_ratio > 1.0);
    }

    #[test]
    fn test_neural_decoder_fallback() {
        let decoder = NeuralDecoder::new().expect("Failed to create decoder");
        assert!(!decoder.is_model_loaded());

        let features = SemanticFeatures::new(0, vec![0.1, 0.2, 0.3, 0.4], vec![16]);
        let decoded = decoder.decode(&features).expect("Decoding failed");

        assert_eq!(decoded.len(), 16);
    }

    #[test]
    fn test_neural_codec_roundtrip_fallback() {
        let codec = NeuralCodec::new(16).expect("Failed to create codec");
        assert!(!codec.is_ready());

        let data: Vec<f32> = (0..128).map(|i| i as f32 / 127.0).collect();
        let features = codec
            .encode(&data, SemanticTask::SensorFusion)
            .expect("Encoding failed");
        let decoded = codec.decode(&features).expect("Decoding failed");

        assert_eq!(decoded.len(), data.len());
    }

    #[test]
    fn test_load_nonexistent_model() {
        let mut encoder = NeuralEncoder::new(32).expect("Failed to create encoder");
        let result = encoder.load_model(Path::new("/nonexistent/encoder.onnx"));
        assert!(result.is_err());
        assert!(!encoder.is_model_loaded());
    }

    #[test]
    fn test_task_to_id_mapping() {
        assert_eq!(task_to_id(SemanticTask::ImageClassification), 0);
        assert_eq!(task_to_id(SemanticTask::ObjectDetection), 1);
        assert_eq!(task_to_id(SemanticTask::SpeechRecognition), 2);
        assert_eq!(task_to_id(SemanticTask::TextUnderstanding), 3);
        assert_eq!(task_to_id(SemanticTask::SensorFusion), 4);
        assert_eq!(task_to_id(SemanticTask::VideoAnalytics), 5);
        assert_eq!(task_to_id(SemanticTask::Custom(42)), 42);
    }
}
