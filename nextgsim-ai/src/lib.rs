//! AI/ML Infrastructure for nextgsim
//!
//! This crate provides the core AI/ML infrastructure for 6G AI-native network functions.
//! The **only operational inference backend** is `OnnxEngine` (ONNX Runtime). No `.onnx`
//! model files ship with the repository, so callers that do not load a model at runtime
//! receive non-neural fallback results from the using crate (e.g. mean-pooling in
//! `nextgsim-semantic`, linear extrapolation in `nextgsim-nwdaf`).
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
//! # Operational Inference Backend
//!
//! - **ONNX Runtime** (`OnnxEngine`): the sole operational backend.
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
//! # Design alignment (non-conformant)
//!
//! This crate provides inference infrastructure conceptually aligned with the
//! AI/ML frameworks described in (not conformant to, not tested against):
//! - 3GPP TS 23.288: Network Data Analytics Function (NWDAF)
//! - 3GPP TR 23.700-80: Study on AI/ML for 5G System
//! - 3GPP TS 23.558: Edge Computing
//!
//! No `.onnx` model ships; no analytics procedure of these specs is implemented.
//!
//! # TR 22.870 mapping
//!
//! Prototypes the AI/ML-service use cases of TR 22.870 (e.g. §6.12 6G System
//! supporting AI model training service, §6.25 AI/ML model training and
//! inference). TR 22.870 is a Stage-1 study; these are potential requirements
//! with no Stage-3 wire spec — this crate is a research prototype, not a
//! conformant implementation.

pub mod config;
pub mod error;
pub mod fl_training;
pub mod inference;
pub mod isac_pipeline;
pub mod metrics;
pub mod model;
pub mod nr_models;
pub mod semantic_pipeline;
pub mod tensor;
pub mod xr_traffic;

// Re-export main types
pub use config::{AiConfig, ExecutionProvider, InferenceConfig};
pub use error::{AiError, InferenceError, ModelError};
pub use fl_training::{FlError, FlParticipant, FlTrainer, ParticipantStatus, RoundResult};
pub use inference::{InferenceEngine, OnnxEngine};
pub use isac_pipeline::{
    FusedSensingResult, IsacError, IsacPipeline, IsacPipelineBuilder, PositionEstimate, SensingData,
};
pub use metrics::{InferenceMetrics, ModelMetrics};
pub use model::{ModelInfo, ModelMetadata};
pub use semantic_pipeline::{
    SemanticDecoding, SemanticEncoding, SemanticError, SemanticPipeline, SemanticPipelineBuilder,
};
pub use tensor::{TensorData, TensorShape};
pub use xr_traffic::{
    CdrxState, PduSet, PduSetManager, Xr5Qi, XrCdrxController, XrFrame, XrQosFlow, XrTrafficModel,
};
