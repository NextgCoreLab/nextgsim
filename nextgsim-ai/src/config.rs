//! AI configuration types
//!
//! This module provides configuration structures for AI/ML components,
//! including execution providers and inference settings.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Execution provider for ML inference
///
/// Supports multiple hardware acceleration backends through ONNX Runtime.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum ExecutionProvider {
    /// CPU execution (default, always available)
    #[default]
    Cpu,
    /// NVIDIA CUDA acceleration
    Cuda {
        /// CUDA device ID (0 for first GPU)
        device_id: i32,
    },
    /// Apple `CoreML` acceleration (macOS/iOS)
    CoreML,
    /// Windows `DirectML` acceleration
    DirectML {
        /// Device ID
        device_id: i32,
    },
    /// NVIDIA `TensorRT` optimization
    TensorRT {
        /// Device ID
        device_id: i32,
        /// Maximum workspace size in bytes
        max_workspace_size: usize,
    },
}


impl std::fmt::Display for ExecutionProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutionProvider::Cpu => write!(f, "CPU"),
            ExecutionProvider::Cuda { device_id } => write!(f, "CUDA(device={device_id})"),
            ExecutionProvider::CoreML => write!(f, "CoreML"),
            ExecutionProvider::DirectML { device_id } => write!(f, "DirectML(device={device_id})"),
            ExecutionProvider::TensorRT { device_id, .. } => write!(f, "TensorRT(device={device_id})"),
        }
    }
}

/// Configuration for inference engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Execution provider to use
    pub execution_provider: ExecutionProvider,
    /// Number of threads for CPU inference (0 = auto)
    pub num_threads: usize,
    /// Enable memory optimization
    pub optimize_memory: bool,
    /// Enable graph optimization
    pub graph_optimization_level: GraphOptimizationLevel,
    /// Maximum batch size for inference
    pub max_batch_size: usize,
    /// Inference timeout in milliseconds (0 = no timeout)
    pub timeout_ms: u64,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            execution_provider: ExecutionProvider::Cpu,
            num_threads: 0, // Auto-detect
            optimize_memory: true,
            graph_optimization_level: GraphOptimizationLevel::All,
            max_batch_size: 32,
            timeout_ms: 0, // No timeout
        }
    }
}

impl InferenceConfig {
    /// Creates a new config with CPU execution
    pub fn cpu() -> Self {
        Self::default()
    }

    /// Creates a new config with CUDA execution
    pub fn cuda(device_id: i32) -> Self {
        Self {
            execution_provider: ExecutionProvider::Cuda { device_id },
            ..Default::default()
        }
    }

    /// Creates a new config with `CoreML` execution
    pub fn coreml() -> Self {
        Self {
            execution_provider: ExecutionProvider::CoreML,
            ..Default::default()
        }
    }

    /// Sets the number of threads
    pub fn with_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = num_threads;
        self
    }

    /// Sets the maximum batch size
    pub fn with_max_batch_size(mut self, max_batch_size: usize) -> Self {
        self.max_batch_size = max_batch_size;
        self
    }

    /// Sets the timeout in milliseconds
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }
}

/// Graph optimization level for ONNX Runtime
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum GraphOptimizationLevel {
    /// No optimization
    None,
    /// Basic optimizations
    Basic,
    /// Extended optimizations
    Extended,
    /// All optimizations (default)
    #[default]
    All,
}


/// Top-level AI configuration for the entire system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiConfig {
    /// Whether AI features are enabled
    pub enabled: bool,
    /// Default inference configuration
    pub inference: InferenceConfig,
    /// Model directory path
    pub model_dir: Option<PathBuf>,
    /// SHE (Service Hosting Environment) configuration
    pub she: SheConfig,
    /// NWDAF configuration
    pub nwdaf: NwdafConfig,
    /// NKEF configuration
    pub nkef: NkefConfig,
    /// ISAC configuration
    pub isac: IsacConfig,
    /// FL (Federated Learning) configuration
    pub fl: FlConfig,
    /// Semantic communication configuration
    pub semantic: SemanticConfig,
}

impl Default for AiConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            inference: InferenceConfig::default(),
            model_dir: None,
            she: SheConfig::default(),
            nwdaf: NwdafConfig::default(),
            nkef: NkefConfig::default(),
            isac: IsacConfig::default(),
            fl: FlConfig::default(),
            semantic: SemanticConfig::default(),
        }
    }
}

/// Service Hosting Environment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SheConfig {
    /// Whether SHE is enabled
    pub enabled: bool,
    /// Local edge latency constraint in milliseconds
    pub local_edge_latency_ms: u32,
    /// Regional edge latency constraint in milliseconds
    pub regional_edge_latency_ms: u32,
    /// Local edge compute capacity (FLOPS)
    pub local_edge_capacity_flops: u64,
    /// Regional edge compute capacity (FLOPS)
    pub regional_edge_capacity_flops: u64,
    /// Core cloud compute capacity (FLOPS)
    pub core_cloud_capacity_flops: u64,
    /// Local edge memory capacity in MB
    pub local_edge_memory_mb: u64,
    /// Regional edge memory capacity in MB
    pub regional_edge_memory_mb: u64,
    /// Core cloud memory capacity in MB
    pub core_cloud_memory_mb: u64,
}

impl Default for SheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            local_edge_latency_ms: 10,
            regional_edge_latency_ms: 20,
            local_edge_capacity_flops: 1_000_000_000_000,      // 1 TFLOPS
            regional_edge_capacity_flops: 10_000_000_000_000,  // 10 TFLOPS
            core_cloud_capacity_flops: 100_000_000_000_000,    // 100 TFLOPS
            local_edge_memory_mb: 8 * 1024,    // 8 GB
            regional_edge_memory_mb: 64 * 1024, // 64 GB
            core_cloud_memory_mb: 512 * 1024,   // 512 GB
        }
    }
}

/// NWDAF (Network Data Analytics Function) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NwdafConfig {
    /// Whether NWDAF is enabled
    pub enabled: bool,
    /// Path to trajectory prediction model
    pub trajectory_model_path: Option<PathBuf>,
    /// Path to load prediction model
    pub load_prediction_model_path: Option<PathBuf>,
    /// Path to handover decision model
    pub handover_model_path: Option<PathBuf>,
    /// Prediction horizon in milliseconds
    pub prediction_horizon_ms: u32,
    /// Data collection interval in milliseconds
    pub collection_interval_ms: u32,
    /// Confidence threshold for handover recommendations
    pub handover_confidence_threshold: f32,
    /// Enable closed-loop automation
    pub enable_automation: bool,
    /// Maximum history length for trajectory prediction
    pub max_history_length: usize,
}

impl Default for NwdafConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            trajectory_model_path: None,
            load_prediction_model_path: None,
            handover_model_path: None,
            prediction_horizon_ms: 1000,
            collection_interval_ms: 100,
            handover_confidence_threshold: 0.8,
            enable_automation: false,
            max_history_length: 100,
        }
    }
}

/// NKEF (Network Knowledge Exposure Function) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NkefConfig {
    /// Whether NKEF is enabled
    pub enabled: bool,
    /// Path to embedding model for semantic search
    pub embedding_model_path: Option<PathBuf>,
    /// Vector dimension for embeddings
    pub embedding_dim: usize,
    /// Maximum number of results for queries
    pub max_results: usize,
    /// Similarity threshold for semantic search
    pub similarity_threshold: f32,
}

impl Default for NkefConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            embedding_model_path: None,
            embedding_dim: 384,
            max_results: 10,
            similarity_threshold: 0.7,
        }
    }
}

/// ISAC (Integrated Sensing and Communication) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsacConfig {
    /// Whether ISAC is enabled
    pub enabled: bool,
    /// Path to positioning model
    pub positioning_model_path: Option<PathBuf>,
    /// Sensing data fusion interval in milliseconds
    pub fusion_interval_ms: u32,
    /// Maximum number of data sources for fusion
    pub max_data_sources: usize,
    /// Position uncertainty threshold in meters
    pub position_uncertainty_threshold_m: f32,
}

impl Default for IsacConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            positioning_model_path: None,
            fusion_interval_ms: 50,
            max_data_sources: 8,
            position_uncertainty_threshold_m: 1.0,
        }
    }
}

/// Federated Learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlConfig {
    /// Whether FL is enabled
    pub enabled: bool,
    /// Aggregation algorithm
    pub aggregation_algorithm: AggregationAlgorithm,
    /// Minimum number of participants for aggregation
    pub min_participants: usize,
    /// Maximum rounds before timeout
    pub max_rounds: usize,
    /// Enable differential privacy
    pub enable_differential_privacy: bool,
    /// Noise multiplier for differential privacy
    pub dp_noise_multiplier: f32,
    /// Clipping threshold for differential privacy
    pub dp_clipping_threshold: f32,
    /// `FedProx` proximal term coefficient (mu). Controls how much local
    /// models are penalised for deviating from the global model. Only
    /// used when `aggregation_algorithm` is `FedProx`. Typical values
    /// range from 0.001 to 1.0.
    pub fedprox_mu: f32,
}

impl Default for FlConfig {
    fn default() -> Self {
        Self {
            enabled: false, // Disabled by default
            aggregation_algorithm: AggregationAlgorithm::FedAvg,
            min_participants: 3,
            max_rounds: 100,
            enable_differential_privacy: false,
            dp_noise_multiplier: 1.0,
            dp_clipping_threshold: 1.0,
            fedprox_mu: 0.01,
        }
    }
}

/// Federated learning aggregation algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum AggregationAlgorithm {
    /// Federated Averaging
    #[default]
    FedAvg,
    /// Federated Proximal
    FedProx,
    /// Secure Aggregation
    SecAgg,
}


/// Semantic communication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticConfig {
    /// Whether semantic communication is enabled
    pub enabled: bool,
    /// Path to semantic encoder model
    pub encoder_model_path: Option<PathBuf>,
    /// Path to semantic decoder model
    pub decoder_model_path: Option<PathBuf>,
    /// Compression ratio target
    pub compression_ratio: f32,
    /// Quality threshold (0.0 to 1.0)
    pub quality_threshold: f32,
}

impl Default for SemanticConfig {
    fn default() -> Self {
        Self {
            enabled: false, // Disabled by default
            encoder_model_path: None,
            decoder_model_path: None,
            compression_ratio: 0.1,
            quality_threshold: 0.9,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_provider_default() {
        let ep = ExecutionProvider::default();
        assert_eq!(ep, ExecutionProvider::Cpu);
    }

    #[test]
    fn test_execution_provider_display() {
        assert_eq!(format!("{}", ExecutionProvider::Cpu), "CPU");
        assert_eq!(
            format!("{}", ExecutionProvider::Cuda { device_id: 0 }),
            "CUDA(device=0)"
        );
        assert_eq!(format!("{}", ExecutionProvider::CoreML), "CoreML");
    }

    #[test]
    fn test_inference_config_builders() {
        let cpu_config = InferenceConfig::cpu();
        assert_eq!(cpu_config.execution_provider, ExecutionProvider::Cpu);

        let cuda_config = InferenceConfig::cuda(0);
        assert_eq!(
            cuda_config.execution_provider,
            ExecutionProvider::Cuda { device_id: 0 }
        );

        let coreml_config = InferenceConfig::coreml();
        assert_eq!(coreml_config.execution_provider, ExecutionProvider::CoreML);
    }

    #[test]
    fn test_inference_config_with_methods() {
        let config = InferenceConfig::cpu()
            .with_threads(4)
            .with_max_batch_size(64)
            .with_timeout(1000);

        assert_eq!(config.num_threads, 4);
        assert_eq!(config.max_batch_size, 64);
        assert_eq!(config.timeout_ms, 1000);
    }

    #[test]
    fn test_ai_config_default() {
        let config = AiConfig::default();
        assert!(config.enabled);
        assert!(config.she.enabled);
        assert!(config.nwdaf.enabled);
        assert!(!config.fl.enabled); // FL disabled by default
    }

    #[test]
    fn test_she_config_latencies() {
        let config = SheConfig::default();
        assert_eq!(config.local_edge_latency_ms, 10);
        assert_eq!(config.regional_edge_latency_ms, 20);
    }

    #[test]
    fn test_nwdaf_config_defaults() {
        let config = NwdafConfig::default();
        assert!(config.enabled);
        assert!(config.trajectory_model_path.is_none());
        assert_eq!(config.handover_confidence_threshold, 0.8);
    }

    #[test]
    fn test_fl_aggregation_algorithms() {
        assert_eq!(
            AggregationAlgorithm::default(),
            AggregationAlgorithm::FedAvg
        );
    }

    #[test]
    fn test_config_serialization() {
        let config = AiConfig::default();
        let json = serde_json::to_string(&config).expect("Failed to serialize");
        let deserialized: AiConfig =
            serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(config.enabled, deserialized.enabled);
    }
}
