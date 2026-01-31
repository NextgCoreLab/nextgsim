//! Error types for AI/ML operations
//!
//! This module defines the error hierarchy for all AI-related operations
//! including model loading, inference, and tensor manipulation.

use std::path::PathBuf;
use thiserror::Error;

/// Top-level error type for AI operations
#[derive(Error, Debug)]
pub enum AiError {
    /// Model-related errors
    #[error("Model error: {0}")]
    Model(#[from] ModelError),

    /// Inference-related errors
    #[error("Inference error: {0}")]
    Inference(#[from] InferenceError),

    /// Configuration errors
    #[error("Configuration error: {0}")]
    Config(String),

    /// IO errors
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Errors that occur during model loading and management
#[derive(Error, Debug)]
pub enum ModelError {
    /// Model file not found
    #[error("Model file not found: {path}")]
    NotFound {
        /// Path to the missing model file
        path: PathBuf,
    },

    /// Invalid model format
    #[error("Invalid model format: {reason}")]
    InvalidFormat {
        /// Description of why the format is invalid
        reason: String,
    },

    /// Model version incompatibility
    #[error("Model version incompatible: expected {expected}, got {actual}")]
    VersionMismatch {
        /// Expected version
        expected: String,
        /// Actual version found
        actual: String,
    },

    /// Failed to load model into runtime
    #[error("Failed to load model: {reason}")]
    LoadFailed {
        /// Reason for load failure
        reason: String,
    },

    /// Model metadata parsing error
    #[error("Failed to parse model metadata: {reason}")]
    MetadataError {
        /// Reason for metadata parsing failure
        reason: String,
    },

    /// ONNX runtime error
    #[error("ONNX runtime error: {0}")]
    OnnxError(String),

    /// Model not loaded
    #[error("Model not loaded - call load_model() first")]
    NotLoaded,
}

/// Errors that occur during inference
#[derive(Error, Debug)]
pub enum InferenceError {
    /// Input tensor shape mismatch
    #[error("Input shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// Expected shape
        expected: Vec<i64>,
        /// Actual shape provided
        actual: Vec<i64>,
    },

    /// Input tensor data type mismatch
    #[error("Input data type mismatch: expected {expected}, got {actual}")]
    DataTypeMismatch {
        /// Expected data type
        expected: String,
        /// Actual data type provided
        actual: String,
    },

    /// Inference execution failed
    #[error("Inference execution failed: {reason}")]
    ExecutionFailed {
        /// Reason for execution failure
        reason: String,
    },

    /// Inference timeout
    #[error("Inference timeout after {timeout_ms}ms")]
    Timeout {
        /// Timeout duration in milliseconds
        timeout_ms: u64,
    },

    /// Batch size exceeds maximum
    #[error("Batch size {requested} exceeds maximum {maximum}")]
    BatchSizeExceeded {
        /// Requested batch size
        requested: usize,
        /// Maximum allowed batch size
        maximum: usize,
    },

    /// Output tensor extraction failed
    #[error("Failed to extract output tensor: {reason}")]
    OutputExtractionFailed {
        /// Reason for extraction failure
        reason: String,
    },

    /// Model not ready for inference
    #[error("Model not ready: {reason}")]
    NotReady {
        /// Reason model is not ready
        reason: String,
    },

    /// GPU memory exhausted
    #[error("GPU memory exhausted during inference")]
    GpuMemoryExhausted,

    /// Invalid input provided
    #[error("Invalid input: {reason}")]
    InvalidInput {
        /// Reason input is invalid
        reason: String,
    },
}

impl From<ort::Error> for ModelError {
    fn from(err: ort::Error) -> Self {
        ModelError::OnnxError(err.to_string())
    }
}

impl From<ort::Error> for InferenceError {
    fn from(err: ort::Error) -> Self {
        InferenceError::ExecutionFailed {
            reason: err.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_error_display() {
        let err = ModelError::NotFound {
            path: PathBuf::from("/path/to/model.onnx"),
        };
        assert!(err.to_string().contains("Model file not found"));
        assert!(err.to_string().contains("/path/to/model.onnx"));
    }

    #[test]
    fn test_inference_error_display() {
        let err = InferenceError::ShapeMismatch {
            expected: vec![1, 10, 3],
            actual: vec![1, 5, 3],
        };
        assert!(err.to_string().contains("Input shape mismatch"));
    }

    #[test]
    fn test_ai_error_from_model_error() {
        let model_err = ModelError::NotLoaded;
        let ai_err: AiError = model_err.into();
        assert!(matches!(ai_err, AiError::Model(_)));
    }

    #[test]
    fn test_ai_error_from_inference_error() {
        let infer_err = InferenceError::Timeout { timeout_ms: 1000 };
        let ai_err: AiError = infer_err.into();
        assert!(matches!(ai_err, AiError::Inference(_)));
    }
}
