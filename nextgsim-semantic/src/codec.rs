//! Semantic codec for nextgsim: operational path is mean-pooling encoder + nearest-neighbor decoder.
//!
//! # Operational path
//!
//! **No `.onnx` model ships with the repository.** Unless a caller explicitly
//! loads a model via [`NeuralEncoder::load_model`] / [`NeuralDecoder::load_model`],
//! every encode/decode call uses the non-neural fallback algorithms:
//!
//! - **Encode**: stride-based **mean-pooling** of the input vector, producing
//!   per-chunk means with variance-derived importance weights.
//! - **Decode**: **nearest-neighbor upsampling** — each compressed feature value
//!   is repeated `stride` times to reconstruct the original length.
//!
//! ONNX-based neural inference is supported as an optional upgrade when a
//! compatible model file is provided at runtime; it is not the default or
//! shipped path.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use tracing::{debug, info, warn};

use nextgsim_ai::config::ExecutionProvider;
use nextgsim_ai::error::ModelError;
use nextgsim_ai::inference::{InferenceEngine, OnnxEngine};
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

/// Emits a one-time `warn!` (then `debug!` on subsequent calls) when a codec
/// falls back to its non-neural path because no ONNX model is loaded.
///
/// `encode`/`decode` run per message, so an unconditional `warn!` would flood
/// the logs; this surfaces the (otherwise easy-to-miss) quality degradation
/// loudly exactly once per codec instance.
fn warn_fallback_once(flag: &AtomicBool, role: &str, method: &str) {
    if flag.swap(true, Ordering::Relaxed) {
        debug!("No {role} model loaded, using {method} fallback");
    } else {
        warn!(
            "No {role} model loaded — using {method} fallback; semantic codec output is \
             degraded (not neural). Load a model via load_model() to silence this warning."
        );
    }
}

/// Which algorithm a codec is actually running (issue #28).
///
/// Reported rather than inferred: the module docs used to state in prose that the
/// mean-pooling path is "always the default", which a caller could not check and
/// a test could not assert. With no `.onnx` in the tree an ONNX-backed and a
/// fallback codec produce differently-shaped output but nothing named which ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecAlgorithm {
    /// Stride-based mean pooling with variance-derived importance weights.
    MeanPooling,
    /// Nearest-neighbour upsampling.
    NearestNeighbour,
    /// A learned model executed through ONNX Runtime.
    Onnx,
}

impl CodecAlgorithm {
    /// Whether this is a learned model rather than an analytic stand-in.
    pub fn is_learned(self) -> bool {
        matches!(self, Self::Onnx)
    }
}

/// Compresses a signal into task-relevant semantic features.
///
/// The trait exists so a consumer can be written against "a semantic encoder"
/// instead of against `NeuralEncoder` specifically -- which is what let
/// `isac_integration` grow its own copy of mean pooling rather than reusing the
/// codec's.
pub trait SemanticEncode {
    /// Encode `data` for `task`.
    ///
    /// # Errors
    /// Returns [`CodecError`] when a loaded model fails to run or produces
    /// output the caller cannot use. The analytic fallback cannot fail.
    fn encode(&self, data: &[f32], task: SemanticTask) -> Result<SemanticFeatures, CodecError>;

    /// The compressed feature dimension this encoder targets.
    fn target_dim(&self) -> usize;

    /// Which algorithm the next [`Self::encode`] will run.
    fn algorithm(&self) -> CodecAlgorithm;
}

/// Reconstructs a signal from semantic features.
pub trait SemanticDecode {
    /// Decode `features` back to a signal of the original length.
    ///
    /// # Errors
    /// Returns [`CodecError`] when a loaded model fails to run or produces
    /// output the caller cannot use.
    fn decode(&self, features: &SemanticFeatures) -> Result<Vec<f32>, CodecError>;

    /// Which algorithm the next [`Self::decode`] will run.
    fn algorithm(&self) -> CodecAlgorithm;
}

/// Stride-based mean pooling: the analytic encoder both the codec fallback and
/// the ISAC measurement compressor use (issue #28).
///
/// Returns the per-chunk means and a variance-derived importance weight per
/// chunk, normalised to a maximum of 1.0. ONE implementation, because two copies
/// of a compression algorithm are two things a decoder can disagree with.
///
/// `feature_dim` is clamped to at least 1, and a `data` shorter than
/// `feature_dim` yields one feature per available chunk rather than padding --
/// padding would invent signal.
pub fn mean_pool(data: &[f32], feature_dim: usize) -> (Vec<f32>, Vec<f32>) {
    let feature_dim = feature_dim.max(1);
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

    let max_importance = importance.iter().copied().fold(f32::MIN, f32::max);
    if max_importance > 0.0 {
        for imp in &mut importance {
            *imp /= max_importance;
        }
    }

    (features, importance)
}

/// Semantic encoder for feature vector compression.
///
/// **Operational algorithm**: stride-based **mean-pooling** — the input is
/// divided into `target_dim` equal-sized chunks; each chunk is summarised by
/// its mean value and variance-derived importance weight.
///
/// ONNX-based neural encoding is an optional upgrade: call [`Self::load_model`]
/// with a compatible `.onnx` encoder model to enable it. No model ships with
/// the repository, so the mean-pooling path is always the default.
pub struct NeuralEncoder {
    /// ONNX inference engine for the encoder model, owned by this encoder.
    engine: OnnxEngine,
    /// An engine resolved from the shared registry (issue #17), which takes
    /// precedence over `engine` when present.
    ///
    /// This is where the sharing pays: an encoder and a decoder pointing at the
    /// same `.onnx` used to load it twice and hold two `ort::Session`s. Resolving
    /// one model id gives them one session. Additive — `load_model` is unchanged.
    #[cfg(feature = "model-registry")]
    shared_engine: Option<std::sync::Arc<OnnxEngine>>,
    /// Target feature dimension for the compressed representation
    target_dim: usize,
    /// Whether the ONNX model is loaded and ready
    model_loaded: bool,
    /// Tracks whether the mean-pooling fallback warning has been emitted yet,
    /// so it is logged loudly once instead of on every `encode` call.
    fallback_warned: AtomicBool,
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
            #[cfg(feature = "model-registry")]
            shared_engine: None,
            target_dim,
            model_loaded: false,
            fallback_warned: AtomicBool::new(false),
        })
    }

    /// Creates a neural encoder with a custom execution provider.
    ///
    /// # Errors
    /// Returns `CodecError::ModelLoad` if the ONNX engine cannot be initialized.
    pub fn with_provider(
        target_dim: usize,
        provider: ExecutionProvider,
    ) -> Result<Self, CodecError> {
        let engine = OnnxEngine::new(provider)?;
        Ok(Self {
            engine,
            #[cfg(feature = "model-registry")]
            shared_engine: None,
            target_dim,
            model_loaded: false,
            fallback_warned: AtomicBool::new(false),
        })
    }

    /// Resolve the encoder's model from the shared registry (issue #17), so an
    /// encoder and decoder on the same model id share one warmed session instead
    /// of loading the file twice.
    ///
    /// # Errors
    ///
    /// `RegistryError::NotProvisioned` when nothing is provisioned under `id` —
    /// the expected case here, since no `.onnx` ships. The encoder then keeps its
    /// mean-pooling fallback, which is what it does with no model anyway.
    #[cfg(feature = "model-registry")]
    pub fn resolve_from_registry(
        &mut self,
        registry: &nextgsim_ai::SharedModelRegistry,
        id: &str,
    ) -> Result<(), nextgsim_ai::RegistryError> {
        let engine = registry.load_by_id(id)?;
        self.shared_engine = Some(engine);
        self.model_loaded = true;
        Ok(())
    }

    /// The identity of the engine [`Self::active_engine`] routes to. Exists
    /// because that routing is otherwise unobservable here: with no `.onnx` to
    /// load, an owned engine and a registry engine are both not ready, so nothing
    /// in an encode result distinguishes them.
    #[cfg(feature = "model-registry")]
    pub fn active_engine_ptr(&self) -> *const OnnxEngine {
        if let Some(ref shared) = self.shared_engine {
            return std::sync::Arc::as_ptr(shared);
        }
        &self.engine as *const OnnxEngine
    }

    /// The engine encoding runs against: the shared one when the registry
    /// supplied it, else the one this encoder owns.
    fn active_engine(&self) -> &dyn InferenceEngine {
        #[cfg(feature = "model-registry")]
        if let Some(ref shared) = self.shared_engine {
            return shared.as_ref();
        }
        &self.engine
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
    /// **Operational path**: stride-based **mean-pooling** (see struct doc).
    /// If an ONNX model has been loaded via [`Self::load_model`], that model
    /// is used instead; otherwise the mean-pooling fallback runs and a one-time
    /// warning is logged.
    ///
    /// # Errors
    /// Returns `CodecError::Inference` if ONNX inference fails.
    pub fn encode(&self, data: &[f32], task: SemanticTask) -> Result<SemanticFeatures, CodecError> {
        if self.model_loaded {
            self.encode_neural(data, task)
        } else {
            warn_fallback_once(&self.fallback_warned, "encoder", "mean-pooling");
            Ok(self.encode_fallback(data, task))
        }
    }

    /// Runs the ONNX encoder model on the input data.
    fn encode_neural(
        &self,
        data: &[f32],
        task: SemanticTask,
    ) -> Result<SemanticFeatures, CodecError> {
        let input = TensorData::float32(data.to_vec(), vec![1i64, data.len() as i64]);
        let output = self.active_engine().infer(&input)?;

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

    /// Mean-pooling fallback encoder, via the shared [`mean_pool`] so the ISAC
    /// measurement compressor and this path cannot drift apart.
    fn encode_fallback(&self, data: &[f32], task: SemanticTask) -> SemanticFeatures {
        let (features, importance) = mean_pool(data, self.target_dim);
        let task_id = task_to_id(task);
        SemanticFeatures::new(task_id, features, vec![data.len()]).with_importance(importance)
    }
}

/// Semantic decoder that reconstructs feature vectors from compressed representations.
///
/// **Operational algorithm**: **nearest-neighbor upsampling** — each compressed
/// feature value is repeated `stride` times to fill the original output length.
///
/// ONNX-based neural decoding is an optional upgrade: call [`Self::load_model`]
/// with a compatible `.onnx` decoder model to enable it. No model ships with
/// the repository, so the nearest-neighbor path is always the default.
pub struct NeuralDecoder {
    /// ONNX inference engine for the decoder model, owned by this decoder.
    engine: OnnxEngine,
    /// An engine resolved from the shared registry (issue #17). See
    /// [`NeuralEncoder::shared_engine`] for why it exists.
    #[cfg(feature = "model-registry")]
    shared_engine: Option<std::sync::Arc<OnnxEngine>>,
    /// Whether the ONNX model is loaded and ready
    model_loaded: bool,
    /// Tracks whether the nearest-neighbor fallback warning has been emitted
    /// yet, so it is logged loudly once instead of on every `decode` call.
    fallback_warned: AtomicBool,
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
            #[cfg(feature = "model-registry")]
            shared_engine: None,
            model_loaded: false,
            fallback_warned: AtomicBool::new(false),
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
            #[cfg(feature = "model-registry")]
            shared_engine: None,
            model_loaded: false,
            fallback_warned: AtomicBool::new(false),
        })
    }

    /// Resolve the decoder's model from the shared registry (issue #17). See
    /// [`NeuralEncoder::resolve_from_registry`].
    ///
    /// # Errors
    ///
    /// `RegistryError::NotProvisioned` when nothing is provisioned under `id`; the
    /// decoder then keeps its nearest-neighbour fallback.
    #[cfg(feature = "model-registry")]
    pub fn resolve_from_registry(
        &mut self,
        registry: &nextgsim_ai::SharedModelRegistry,
        id: &str,
    ) -> Result<(), nextgsim_ai::RegistryError> {
        let engine = registry.load_by_id(id)?;
        self.shared_engine = Some(engine);
        self.model_loaded = true;
        Ok(())
    }

    /// The identity of the engine [`Self::active_engine`] routes to. See
    /// [`NeuralEncoder::active_engine_ptr`].
    #[cfg(feature = "model-registry")]
    pub fn active_engine_ptr(&self) -> *const OnnxEngine {
        if let Some(ref shared) = self.shared_engine {
            return std::sync::Arc::as_ptr(shared);
        }
        &self.engine as *const OnnxEngine
    }

    /// The engine decoding runs against: the shared one when the registry
    /// supplied it, else the one this decoder owns.
    fn active_engine(&self) -> &dyn InferenceEngine {
        #[cfg(feature = "model-registry")]
        if let Some(ref shared) = self.shared_engine {
            return shared.as_ref();
        }
        &self.engine
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
    /// **Operational path**: **nearest-neighbor upsampling** (see struct doc).
    /// If an ONNX model has been loaded via [`Self::load_model`], that model
    /// is used instead; otherwise the nearest-neighbor fallback runs and a
    /// one-time warning is logged.
    ///
    /// # Errors
    /// Returns `CodecError::Inference` if ONNX inference fails.
    pub fn decode(&self, features: &SemanticFeatures) -> Result<Vec<f32>, CodecError> {
        if self.model_loaded {
            self.decode_neural(features)
        } else {
            warn_fallback_once(&self.fallback_warned, "decoder", "nearest-neighbor");
            Ok(self.decode_fallback(features))
        }
    }

    /// Runs the ONNX decoder model on the compressed features.
    fn decode_neural(&self, features: &SemanticFeatures) -> Result<Vec<f32>, CodecError> {
        let input = TensorData::float32(
            features.features.clone(),
            vec![1i64, features.features.len() as i64],
        );
        let output = self.active_engine().infer(&input)?;

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

impl SemanticEncode for NeuralEncoder {
    fn encode(&self, data: &[f32], task: SemanticTask) -> Result<SemanticFeatures, CodecError> {
        NeuralEncoder::encode(self, data, task)
    }

    fn target_dim(&self) -> usize {
        self.target_dim
    }

    fn algorithm(&self) -> CodecAlgorithm {
        // Reads the same flag `encode` dispatches on, so this cannot claim ONNX
        // while the fallback runs.
        if self.model_loaded {
            CodecAlgorithm::Onnx
        } else {
            CodecAlgorithm::MeanPooling
        }
    }
}

impl SemanticDecode for NeuralDecoder {
    fn decode(&self, features: &SemanticFeatures) -> Result<Vec<f32>, CodecError> {
        NeuralDecoder::decode(self, features)
    }

    fn algorithm(&self) -> CodecAlgorithm {
        if self.model_loaded {
            CodecAlgorithm::Onnx
        } else {
            CodecAlgorithm::NearestNeighbour
        }
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

    /// Honesty guard: the codec must stay labelled as falling back to
    /// mean-pooling when no model is loaded, so the caveat cannot be silently
    /// dropped. If this fails, restore the label rather than deleting the test.
    #[test]
    fn honesty_mean_pooling_fallback_label_present() {
        let src = include_str!("codec.rs");
        assert!(
            src.contains("mean-pooling fallback"),
            "the mean-pooling fallback label must remain present"
        );
    }

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

    // --- Trait-based codec, fidelity and one mean-pooling implementation (#28) ---

    #[test]
    fn a_fallback_round_trip_preserves_the_signal_to_a_documented_fidelity() {
        // The pre-existing round-trip test asserted only that the output LENGTH
        // matched, which every codec including a constant-zero one satisfies.
        // This asserts a task-relevant FIDELITY on a smooth signal, which is what
        // mean pooling plus nearest-neighbour upsampling can actually preserve.
        use crate::metrics::{cosine_similarity, mse};

        let codec = NeuralCodec::new(32).expect("codec");
        // A smooth ramp with a slow sinusoid: representative of a sensor trace,
        // and band-limited enough that 4:1 mean pooling is a fair summary.
        let data: Vec<f32> = (0..128)
            .map(|i| {
                let t = i as f32 / 127.0;
                t + 0.2 * (t * std::f32::consts::TAU).sin()
            })
            .collect();

        let features = codec
            .encode(&data, SemanticTask::SensorFusion)
            .expect("encode");
        let decoded = codec.decode(&features).expect("decode");

        assert_eq!(decoded.len(), data.len());
        // DOCUMENTED BOUNDS for the analytic fallback at 4:1 compression on a
        // smooth signal: cosine similarity above 0.99 and mean squared error
        // below 0.01. They are bounds on the FALLBACK, not on semantic
        // communication in general -- a learned codec would be judged on a task
        // metric instead.
        let similarity = cosine_similarity(&data, &decoded);
        let error = mse(&data, &decoded);
        assert!(
            similarity > 0.99,
            "cosine similarity {similarity:.4} below the 0.99 bound"
        );
        assert!(error < 0.01, "MSE {error:.5} above the 0.01 bound");
    }

    #[test]
    fn a_round_trip_of_noise_is_not_claimed_to_be_faithful() {
        // The counterpart that stops the bound above from being a tautology: mean
        // pooling DISCARDS high-frequency content, so on alternating noise the
        // same round trip must NOT clear the same bar. A codec that passed both
        // would be preserving something it cannot.
        use crate::metrics::mse;

        let codec = NeuralCodec::new(32).expect("codec");
        let data: Vec<f32> = (0..128)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();

        let features = codec
            .encode(&data, SemanticTask::SensorFusion)
            .expect("encode");
        let decoded = codec.decode(&features).expect("decode");

        assert!(
            mse(&data, &decoded) > 0.5,
            "alternating noise cannot survive mean pooling; the fidelity bound \
             above would then be measuring nothing"
        );
    }

    #[test]
    fn the_codec_reports_which_algorithm_it_will_run() {
        // With no model loaded the encoder must SAY mean-pooling rather than
        // leaving a caller to infer it from prose.
        let encoder = NeuralEncoder::new(16).expect("encoder");
        assert_eq!(
            SemanticEncode::algorithm(&encoder),
            CodecAlgorithm::MeanPooling
        );
        assert!(!SemanticEncode::algorithm(&encoder).is_learned());
        assert_eq!(SemanticEncode::target_dim(&encoder), 16);

        let decoder = NeuralDecoder::new().expect("decoder");
        assert_eq!(
            SemanticDecode::algorithm(&decoder),
            CodecAlgorithm::NearestNeighbour
        );
        assert!(!SemanticDecode::algorithm(&decoder).is_learned());
    }

    #[test]
    fn a_consumer_can_be_generic_over_any_semantic_encoder() {
        // The point of the trait: a consumer written against `dyn SemanticEncode`
        // rather than against NeuralEncoder. Without it, `isac_integration` grew
        // its own copy of mean pooling.
        fn compress(encoder: &dyn SemanticEncode, data: &[f32]) -> (usize, usize) {
            // Both the encoder's declared dimension and what it actually
            // produces, read through the trait: a consumer sizing a buffer off
            // `target_dim` and then receiving a different count is the failure
            // this pins.
            let produced = encoder
                .encode(data, SemanticTask::SensorFusion)
                .map(|f| f.features.len())
                .unwrap_or(0);
            (encoder.target_dim(), produced)
        }

        let encoder = NeuralEncoder::new(8).expect("encoder");
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        assert_eq!(compress(&encoder, &data), (8, 8));
    }

    #[test]
    fn the_isac_compressor_and_the_codec_fallback_pool_identically() {
        // One implementation, asserted rather than assumed: these are the two
        // call sites that used to hold separate copies of the algorithm.
        use crate::isac_integration::{
            IsacSemanticCompressor, MeasurementMetadata, SensingCompressionParams,
        };

        let measurements: Vec<f32> = (0..120).map(|i| (i as f32) * 0.25).collect();
        let mut compressor = IsacSemanticCompressor::new();
        let params = SensingCompressionParams {
            target_compression: 4.0,
            ..Default::default()
        };
        let metadata = MeasurementMetadata {
            measurement_types: vec!["ToA".to_string()],
            anchor_ids: vec![1],
            uncertainties: vec![0.5],
            timestamp_ms: 0,
        };
        let compressed = compressor.compress_measurements(measurements.clone(), metadata, params);

        let (expected, _) = mean_pool(&measurements, 30);
        assert_eq!(
            compressed.features.features, expected,
            "the ISAC compressor must pool exactly as the codec does"
        );
    }

    #[test]
    fn mean_pool_survives_asking_for_more_features_than_samples() {
        // The defect the deduplication removed: the ISAC copy computed its chunk
        // size without a `.max(1)`, so this divided by zero and produced NaN.
        let data = vec![1.0, 2.0, 3.0];
        let (features, importance) = mean_pool(&data, 10);
        assert!(
            features.iter().all(|f| f.is_finite()),
            "no NaN: {features:?}"
        );
        assert_eq!(
            features.len(),
            data.len(),
            "one feature per available sample"
        );
        assert_eq!(importance.len(), features.len());
    }

    #[test]
    fn mean_pool_normalises_importance_to_a_maximum_of_one() {
        // Importance weights drive feature pruning elsewhere in this crate, so
        // their SCALE is load-bearing: un-normalised weights make a pruning
        // threshold mean something different for every input.
        let data: Vec<f32> = (0..64)
            .map(|i| if i < 32 { i as f32 * 10.0 } else { 5.0 })
            .collect();
        let (_, importance) = mean_pool(&data, 8);

        let max = importance.iter().copied().fold(f32::MIN, f32::max);
        assert!(
            (max - 1.0).abs() < 1e-6,
            "the largest importance weight must be 1.0, got {max}"
        );
        // And the quiet half must weigh less than the varying half, or the
        // normalisation has flattened the signal it exists to express.
        assert!(
            importance[7] < importance[0],
            "a constant chunk must matter less: {importance:?}"
        );
    }

    #[test]
    fn mean_pool_clamps_a_zero_feature_dimension() {
        let (features, _) = mean_pool(&[1.0, 2.0, 3.0, 4.0], 0);
        assert_eq!(
            features.len(),
            1,
            "a zero dimension must not panic or divide by zero"
        );
    }
}

// ============================================================================
// Shared model registry (issue #17)
// ============================================================================

#[cfg(all(test, feature = "model-registry"))]
mod registry_tests {
    use super::*;
    use nextgsim_ai::{ExecutionProvider, OnnxEngine, SharedModelRegistry};
    use std::sync::Arc;

    /// **The observable form of #17's payoff.** An encoder and a decoder are two
    /// logical consumers of one model id, and before this each owned its own
    /// `OnnxEngine` — so the same `.onnx` was loaded twice and two `ort::Session`s
    /// were held. Resolving through the registry gives them one.
    ///
    /// The registry is seeded with `register_preloaded` because no `.onnx` ships
    /// with this repo, so nothing can be loaded from disk. It populates exactly the
    /// cache slot the loading path fills, so what is proven is the sharing.
    #[test]
    fn an_encoder_and_decoder_on_one_model_id_share_a_single_engine() {
        let registry = SharedModelRegistry::new(ExecutionProvider::Cpu);
        let engine = Arc::new(OnnxEngine::new(ExecutionProvider::Cpu).expect("engine"));
        registry
            .register_preloaded(
                "semantic-codec",
                "1.0.0",
                "/models/codec.onnx",
                Arc::clone(&engine),
            )
            .expect("register");

        let mut encoder = NeuralEncoder::new(64).expect("encoder");
        let mut decoder = NeuralDecoder::new().expect("decoder");

        encoder
            .resolve_from_registry(&registry, "semantic-codec")
            .expect("the encoder resolves");
        decoder
            .resolve_from_registry(&registry, "semantic-codec")
            .expect("the decoder resolves");

        // One engine, held by the registry and by both consumers.
        assert_eq!(
            registry.loaded_count(),
            1,
            "one engine for one model id, not one per consumer"
        );
        // Exactly four holders: this test, the registry, the encoder and the
        // decoder. An inequality would be satisfied by only one of them having
        // resolved, which is the whole thing being asserted.
        assert_eq!(
            Arc::strong_count(&engine),
            4,
            "this test, the registry, the encoder and the decoder"
        );
        // And both actually *route* to it, which the counts do not show.
        assert_eq!(encoder.active_engine_ptr(), Arc::as_ptr(&engine));
        assert_eq!(decoder.active_engine_ptr(), Arc::as_ptr(&engine));
    }

    /// An unprovisioned id leaves both consumers on their existing fallbacks
    /// rather than failing the deployment — which is the normal case here, since no
    /// `.onnx` ships.
    #[test]
    fn an_unprovisioned_id_leaves_the_codec_on_its_fallback() {
        let registry = SharedModelRegistry::new(ExecutionProvider::Cpu);
        let mut encoder = NeuralEncoder::new(64).expect("encoder");

        let err = encoder
            .resolve_from_registry(&registry, "absent")
            .expect_err("nothing is provisioned");
        assert!(err.is_not_provisioned(), "{err}");

        // And the encoder still works, on its mean-pooling fallback.
        let encoded = encoder
            .encode(&[1.0, 2.0, 3.0, 4.0], SemanticTask::SensorFusion)
            .expect("the fallback still encodes");
        assert!(!encoded.features.is_empty());
    }
}
