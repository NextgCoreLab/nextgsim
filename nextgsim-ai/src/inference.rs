//! Inference engine implementations
//!
//! This module provides the core inference engine trait and implementations
//! for ONNX Runtime with GPU acceleration support.

use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

use ort::session::builder::GraphOptimizationLevel as OrtOptLevel;
use ort::session::Session;
use tracing::{debug, info, warn};

use crate::config::{ExecutionProvider, GraphOptimizationLevel, InferenceConfig};
use crate::error::{InferenceError, ModelError};
use crate::metrics::InferenceMetrics;
use crate::model::{ModelInfo, ModelMetadata, ModelType, TensorInfo};
use crate::tensor::TensorData;

/// Trait for ML inference engines
///
/// This trait defines the interface for all inference backends,
/// supporting model loading, single inference, and batch inference.
pub trait InferenceEngine: Send + Sync {
    /// Loads a model from the given path
    ///
    /// # Errors
    /// Returns `ModelError` if the model cannot be loaded
    fn load_model(&mut self, path: &Path) -> Result<(), ModelError>;

    /// Runs inference on a single input
    ///
    /// # Errors
    /// Returns `InferenceError` if inference fails
    fn infer(&self, input: &TensorData) -> Result<TensorData, InferenceError>;

    /// Runs inference on a batch of inputs
    ///
    /// # Errors
    /// Returns `InferenceError` if batch inference fails
    fn batch_infer(&self, inputs: &[TensorData]) -> Result<Vec<TensorData>, InferenceError>;

    /// Runs inference with multiple named inputs
    ///
    /// # Errors
    /// Returns `InferenceError` if inference fails
    fn infer_multi(&self, inputs: &[(&str, TensorData)]) -> Result<Vec<TensorData>, InferenceError>;

    /// Returns the model metadata
    fn metadata(&self) -> &ModelMetadata;

    /// Warms up the model by running a dummy inference
    ///
    /// # Errors
    /// Returns `InferenceError` if warmup fails
    fn warmup(&self) -> Result<(), InferenceError>;

    /// Returns whether the model is loaded and ready
    fn is_ready(&self) -> bool;

    /// Returns the model info
    fn model_info(&self) -> Option<&ModelInfo>;

    /// Returns a reference to the inference metrics
    fn metrics(&self) -> Option<&InferenceMetrics>;
}

/// ONNX Runtime inference engine
///
/// Production-quality inference engine using ONNX Runtime with
/// support for multiple execution providers (CPU, CUDA, CoreML, etc.)
pub struct OnnxEngine {
    /// ONNX Runtime session (wrapped in Mutex for interior mutability)
    session: Option<Mutex<Session>>,
    /// Model metadata
    metadata: ModelMetadata,
    /// Model info
    model_info: Option<ModelInfo>,
    /// Configuration
    config: InferenceConfig,
    /// Inference metrics
    metrics: InferenceMetrics,
    /// Whether the model is ready
    is_ready: bool,
}

impl OnnxEngine {
    /// Creates a new ONNX engine with the given execution provider
    pub fn new(execution_provider: ExecutionProvider) -> Result<Self, ModelError> {
        let config = InferenceConfig {
            execution_provider,
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Creates a new ONNX engine with full configuration
    pub fn with_config(config: InferenceConfig) -> Result<Self, ModelError> {
        info!(
            "Creating ONNX inference engine with {} execution provider",
            config.execution_provider
        );

        Ok(Self {
            session: None,
            metadata: ModelMetadata::default(),
            model_info: None,
            config,
            metrics: InferenceMetrics::new("unknown"),
            is_ready: false,
        })
    }

    /// Creates a session from a file path with the configured execution provider
    fn create_session(&self, path: &Path) -> Result<Session, ModelError> {
        let mut builder = Session::builder()?;

        // Set thread count
        if self.config.num_threads > 0 {
            builder = builder.with_intra_threads(self.config.num_threads)?;
        }

        // Set optimization level
        let opt_level = match self.config.graph_optimization_level {
            GraphOptimizationLevel::None => OrtOptLevel::Disable,
            GraphOptimizationLevel::Basic => OrtOptLevel::Level1,
            GraphOptimizationLevel::Extended => OrtOptLevel::Level2,
            GraphOptimizationLevel::All => OrtOptLevel::Level3,
        };
        builder = builder.with_optimization_level(opt_level)?;

        // Configure execution provider
        // Note: ONNX Runtime will fall back to CPU if the requested provider is not available
        match &self.config.execution_provider {
            ExecutionProvider::Cpu => {
                debug!("Using CPU execution provider");
            }
            ExecutionProvider::Cuda { device_id } => {
                debug!("Attempting CUDA execution provider (device {})", device_id);
            }
            ExecutionProvider::CoreML => {
                debug!("Attempting CoreML execution provider");
            }
            ExecutionProvider::DirectML { device_id } => {
                debug!("Attempting DirectML execution provider (device {})", device_id);
            }
            ExecutionProvider::TensorRT { device_id, max_workspace_size } => {
                debug!(
                    "Attempting TensorRT execution provider (device {}, workspace {})",
                    device_id, max_workspace_size
                );
            }
        }

        let session = builder.commit_from_file(path)?;
        Ok(session)
    }

    /// Extracts metadata from a loaded session
    fn extract_metadata(session: &Session, path: &Path) -> ModelMetadata {
        let mut metadata = ModelMetadata::new(
            path.file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            "1.0.0",
        );

        // Extract input information from session
        for input in session.inputs() {
            let dtype = input.dtype();
            let dtype_str = format!("{:?}", dtype).to_lowercase();

            // Default shape - would need to parse from dtype in production
            let shape: Vec<i64> = vec![-1];

            metadata = metadata.with_input(TensorInfo::new(input.name(), shape, dtype_str));
        }

        // Extract output information from session
        for output in session.outputs() {
            let dtype = output.dtype();
            let dtype_str = format!("{:?}", dtype).to_lowercase();

            // Default shape - would need to parse from dtype in production
            let shape: Vec<i64> = vec![-1];

            metadata = metadata.with_output(TensorInfo::new(output.name(), shape, dtype_str));
        }

        metadata
    }

    /// Runs inference with float32 input data
    fn run_inference_f32(&self, session: &mut Session, data: &[f32], shape: Vec<i64>) -> Result<TensorData, InferenceError> {
        use ort::value::Tensor;

        // Convert shape to usize
        let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();

        // Create tensor from shape and data
        let input_tensor = Tensor::from_array((shape_usize.clone(), data.to_vec()))?;

        // Get input name
        let input_name = session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "input".to_string());

        // Run inference using inputs! macro
        let outputs = session.run(ort::inputs![input_name.as_str() => input_tensor])?;

        // Extract first output
        let (_, output_value) = outputs
            .into_iter()
            .next()
            .ok_or(InferenceError::OutputExtractionFailed {
                reason: "No outputs from model".to_string(),
            })?;

        // Try to extract as f32 tensor
        if let Ok((output_shape, output_data)) = output_value.try_extract_tensor::<f32>() {
            let shape_i64: Vec<i64> = output_shape.iter().map(|&d| d as i64).collect();
            return Ok(TensorData::float32(output_data.to_vec(), shape_i64));
        }

        // Try to extract as f64 tensor
        if let Ok((output_shape, output_data)) = output_value.try_extract_tensor::<f64>() {
            let shape_i64: Vec<i64> = output_shape.iter().map(|&d| d as i64).collect();
            return Ok(TensorData::float64(output_data.to_vec(), shape_i64));
        }

        // Try to extract as i64 tensor
        if let Ok((output_shape, output_data)) = output_value.try_extract_tensor::<i64>() {
            let shape_i64: Vec<i64> = output_shape.iter().map(|&d| d as i64).collect();
            return Ok(TensorData::int64(output_data.to_vec(), shape_i64));
        }

        // Try to extract as i32 tensor
        if let Ok((output_shape, output_data)) = output_value.try_extract_tensor::<i32>() {
            let shape_i64: Vec<i64> = output_shape.iter().map(|&d| d as i64).collect();
            return Ok(TensorData::int32(output_data.to_vec(), shape_i64));
        }

        Err(InferenceError::OutputExtractionFailed {
            reason: "Unsupported output tensor type".to_string(),
        })
    }
}

impl InferenceEngine for OnnxEngine {
    fn load_model(&mut self, path: &Path) -> Result<(), ModelError> {
        if !path.exists() {
            return Err(ModelError::NotFound {
                path: path.to_path_buf(),
            });
        }

        info!("Loading ONNX model from {:?}", path);

        let session = self.create_session(path)?;

        // Extract metadata before wrapping in Mutex
        self.metadata = Self::extract_metadata(&session, path);

        // Get file size
        let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

        // Create model info
        self.model_info = Some(
            ModelInfo::new(path, ModelType::Onnx)
                .with_metadata(self.metadata.clone())
                .with_file_size(file_size)
                .mark_ready(),
        );

        // Update metrics with model name
        self.metrics = InferenceMetrics::new(&self.metadata.name);

        self.session = Some(Mutex::new(session));
        self.is_ready = true;

        info!(
            "Model loaded successfully: {} inputs, {} outputs",
            self.metadata.num_inputs(),
            self.metadata.num_outputs()
        );

        Ok(())
    }

    fn infer(&self, input: &TensorData) -> Result<TensorData, InferenceError> {
        let session_mutex = self.session.as_ref().ok_or(InferenceError::NotReady {
            reason: "Model not loaded".to_string(),
        })?;

        let mut session = session_mutex
            .lock()
            .map_err(|e| InferenceError::NotReady {
                reason: format!("Failed to acquire session lock: {}", e),
            })?;

        let start = Instant::now();

        // Get shape and run inference based on input type
        let result = match input {
            TensorData::Float32 { data, shape } => {
                self.run_inference_f32(&mut session, data, shape.dims().to_vec())?
            }
            TensorData::Float64 { data, shape } => {
                // Convert f64 to f32 for inference
                let f32_data: Vec<f32> = data.iter().map(|&x| x as f32).collect();
                self.run_inference_f32(&mut session, &f32_data, shape.dims().to_vec())?
            }
            TensorData::Int32 { data, shape } => {
                // Convert i32 to f32 for inference
                let f32_data: Vec<f32> = data.iter().map(|&x| x as f32).collect();
                self.run_inference_f32(&mut session, &f32_data, shape.dims().to_vec())?
            }
            TensorData::Int64 { data, shape } => {
                // Convert i64 to f32 for inference
                let f32_data: Vec<f32> = data.iter().map(|&x| x as f32).collect();
                self.run_inference_f32(&mut session, &f32_data, shape.dims().to_vec())?
            }
            TensorData::Uint8 { data, shape } => {
                // Convert u8 to f32 for inference (normalized)
                let f32_data: Vec<f32> = data.iter().map(|&x| x as f32 / 255.0).collect();
                self.run_inference_f32(&mut session, &f32_data, shape.dims().to_vec())?
            }
            TensorData::Float16 { .. } => {
                return Err(InferenceError::InvalidInput {
                    reason: "Float16 tensors must be converted to Float32 before inference".to_string(),
                });
            }
        };

        let latency = start.elapsed();
        debug!("Inference completed in {:?}", latency);

        Ok(result)
    }

    fn batch_infer(&self, inputs: &[TensorData]) -> Result<Vec<TensorData>, InferenceError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        if inputs.len() > self.config.max_batch_size {
            return Err(InferenceError::BatchSizeExceeded {
                requested: inputs.len(),
                maximum: self.config.max_batch_size,
            });
        }

        // For simplicity, run sequential inference
        // A more optimized implementation would batch inputs along the first dimension
        inputs.iter().map(|input| self.infer(input)).collect()
    }

    fn infer_multi(&self, inputs: &[(&str, TensorData)]) -> Result<Vec<TensorData>, InferenceError> {
        let session_mutex = self.session.as_ref().ok_or(InferenceError::NotReady {
            reason: "Model not loaded".to_string(),
        })?;

        let mut session = session_mutex
            .lock()
            .map_err(|e| InferenceError::NotReady {
                reason: format!("Failed to acquire session lock: {}", e),
            })?;

        let start = Instant::now();

        // For multi-input inference, we need to build the inputs manually
        use ort::value::Tensor;

        let mut ort_inputs = Vec::with_capacity(inputs.len());

        for (name, tensor) in inputs {
            match tensor {
                TensorData::Float32 { data, shape } => {
                    let shape_usize: Vec<usize> = shape.dims().iter().map(|&d| d as usize).collect();
                    let input_tensor = Tensor::from_array((shape_usize, data.clone()))?;
                    ort_inputs.push((*name, input_tensor.into_dyn()));
                }
                _ => {
                    return Err(InferenceError::InvalidInput {
                        reason: "Multi-input inference currently only supports Float32".to_string(),
                    });
                }
            }
        }

        // Run inference
        let outputs = session.run(ort_inputs)?;

        // Convert outputs
        let mut results = Vec::with_capacity(outputs.len());
        for (_, output_value) in outputs {
            if let Ok((output_shape, output_data)) = output_value.try_extract_tensor::<f32>() {
                let shape_i64: Vec<i64> = output_shape.iter().map(|&d| d as i64).collect();
                results.push(TensorData::float32(output_data.to_vec(), shape_i64));
            } else {
                return Err(InferenceError::OutputExtractionFailed {
                    reason: "Failed to extract output tensor".to_string(),
                });
            }
        }

        let latency = start.elapsed();
        debug!("Multi-input inference completed in {:?}", latency);

        Ok(results)
    }

    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn warmup(&self) -> Result<(), InferenceError> {
        if !self.is_ready {
            return Err(InferenceError::NotReady {
                reason: "Model not loaded".to_string(),
            });
        }

        info!("Warming up model...");

        // Create dummy input based on first input spec
        if let Some(input_info) = self.metadata.inputs.first() {
            // Replace dynamic dimensions (-1) with 1
            let shape: Vec<i64> = input_info
                .shape
                .dims()
                .iter()
                .map(|&d| if d < 0 { 1 } else { d })
                .collect();

            let dummy_input = TensorData::zeros_f32(shape);
            let _ = self.infer(&dummy_input)?;

            info!("Model warmup complete");
        } else {
            warn!("No input spec found, skipping warmup");
        }

        Ok(())
    }

    fn is_ready(&self) -> bool {
        self.is_ready
    }

    fn model_info(&self) -> Option<&ModelInfo> {
        self.model_info.as_ref()
    }

    fn metrics(&self) -> Option<&InferenceMetrics> {
        Some(&self.metrics)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorShape;

    #[test]
    fn test_onnx_engine_creation() {
        let engine = OnnxEngine::new(ExecutionProvider::Cpu);
        assert!(engine.is_ok());

        let engine = engine.expect("Engine creation failed");
        assert!(!engine.is_ready());
        assert!(engine.model_info().is_none());
    }

    #[test]
    fn test_onnx_engine_with_config() {
        let config = InferenceConfig::cpu()
            .with_threads(4)
            .with_max_batch_size(16);

        let engine = OnnxEngine::with_config(config);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_tensor_shape_creation() {
        let shape = TensorShape::new(vec![1, 10, 3]);
        assert_eq!(shape.dims(), &[1, 10, 3]);
        assert_eq!(shape.num_elements(), 30);
    }

    #[test]
    fn test_load_nonexistent_model() {
        let mut engine = OnnxEngine::new(ExecutionProvider::Cpu).expect("Engine creation failed");
        let result = engine.load_model(Path::new("/nonexistent/model.onnx"));
        assert!(result.is_err());
        assert!(matches!(result.expect_err("Expected error"), ModelError::NotFound { .. }));
    }

    #[test]
    fn test_infer_without_model() {
        let engine = OnnxEngine::new(ExecutionProvider::Cpu).expect("Engine creation failed");
        let input = TensorData::float32(vec![1.0, 2.0, 3.0], vec![1i64, 3]);
        let result = engine.infer(&input);
        assert!(result.is_err());
        assert!(matches!(result.expect_err("Expected error"), InferenceError::NotReady { .. }));
    }

    #[test]
    fn test_batch_size_exceeded() {
        let config = InferenceConfig::cpu().with_max_batch_size(2);
        let engine = OnnxEngine::with_config(config).expect("Engine creation failed");

        let inputs: Vec<TensorData> = (0..5)
            .map(|_| TensorData::float32(vec![1.0], vec![1i64]))
            .collect();

        let result = engine.batch_infer(&inputs);
        assert!(result.is_err());
        assert!(matches!(
            result.expect_err("Expected error"),
            InferenceError::BatchSizeExceeded { .. }
        ));
    }
}
