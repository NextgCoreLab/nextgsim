//! AI/ML Infrastructure for nextgsim
//!
//! This crate provides the core AI/ML infrastructure for 6G AI-native network functions,
//! implementing production-quality inference engines using ONNX Runtime.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                        nextgsim-ai                                   │
//! │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
//! │  │  InferenceEngine│  │  TensorData     │  │  ModelMetadata      │  │
//! │  │  - OnnxEngine   │  │  - Float32      │  │  - Input shapes     │  │
//! │  │  - BatchInfer   │  │  - Float16      │  │  - Output shapes    │  │
//! │  │  - GPU Support  │  │  - Int64        │  │  - Model version    │  │
//! │  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Supported Inference Backends
//!
//! - **ONNX Runtime**: Primary inference backend with GPU acceleration support
//!   - CPU execution provider (default)
//!   - CUDA execution provider (NVIDIA GPUs)
//!   - `CoreML` execution provider (Apple Silicon)
//!   - `DirectML` execution provider (Windows)
//!   - `TensorRT` execution provider (NVIDIA optimized)
//!
//! # Example Usage
//!
//! ```ignore
//! use nextgsim_ai::{OnnxEngine, InferenceEngine, TensorData};
//!
//! // Load an ONNX model
//! let mut engine = OnnxEngine::new(ExecutionProvider::Cpu)?;
//! engine.load_model(Path::new("trajectory_predictor.onnx"))?;
//!
//! // Prepare input tensor
//! let input = TensorData::Float32(vec![1.0, 2.0, 3.0]);
//!
//! // Run inference
//! let output = engine.infer(&input)?;
//! ```
//!
//! # 3GPP Compliance
//!
//! This implementation supports the AI/ML framework requirements from:
//! - 3GPP TS 23.288: Network Data Analytics Function (NWDAF)
//! - 3GPP TR 23.700-80: Study on AI/ML for 5G System
//! - 3GPP TS 23.558: Edge Computing

pub mod config;
pub mod error;
pub mod fl_training;
pub mod inference;
pub mod isac_pipeline;
pub mod metrics;
pub mod model;
pub mod nr_models;
pub mod semantic_pipeline;
pub mod xr_traffic;
pub mod tensor;
pub mod tflite;

// Re-export main types
pub use config::{AiConfig, ExecutionProvider, InferenceConfig};
pub use error::{AiError, InferenceError, ModelError};
pub use fl_training::{FlError, FlParticipant, FlTrainer, ParticipantStatus, RoundResult};
pub use inference::{InferenceEngine, OnnxEngine};
pub use isac_pipeline::{
    FusedSensingResult, IsacError, IsacPipeline, IsacPipelineBuilder, PositionEstimate,
    SensingData,
};
pub use metrics::{InferenceMetrics, ModelMetrics};
pub use model::{ModelInfo, ModelMetadata};
pub use semantic_pipeline::{
    SemanticDecoding, SemanticEncoding, SemanticError, SemanticPipeline, SemanticPipelineBuilder,
};
pub use tensor::{TensorData, TensorShape};
pub use tflite::TfLiteEngine;
pub use xr_traffic::{
    CdrxState, PduSet, PduSetManager, Xr5Qi, XrCdrxController, XrFrame, XrQosFlow,
    XrTrafficModel,
};
