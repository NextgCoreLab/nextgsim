//! AI/ML Configuration Extension
//!
//! 6G extension: RRC configuration for AI/ML-based radio resource management.
//! Supports model deployment, inference configuration, and reporting for
//! AI-native 6G networks.
//!
//! This module implements:
//! - `AiMlConfig` - AI/ML model configuration with inference mode and reporting
//! - Model lifecycle management (deployment, activation, deactivation)
//! - Federated learning and split inference configuration

use thiserror::Error;

/// Errors that can occur during AI/ML configuration procedures
#[derive(Debug, Error)]
pub enum AiMlConfigError {
    /// Invalid AI/ML configuration
    #[error("Invalid AI/ML configuration: {0}")]
    InvalidConfig(String),

    /// Missing mandatory field
    #[error("Missing mandatory field: {0}")]
    MissingMandatoryField(String),

    /// Encoding/decoding error
    #[error("Codec error: {0}")]
    CodecError(String),
}

/// AI/ML model use case (as defined by 3GPP study items)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AiMlUseCase {
    /// CSI feedback compression/enhancement
    CsiFeedback,
    /// Beam management prediction
    BeamManagement,
    /// Positioning enhancement
    Positioning,
    /// Channel estimation
    ChannelEstimation,
    /// Link adaptation
    LinkAdaptation,
    /// Mobility optimization
    MobilityOptimization,
    /// Load balancing
    LoadBalancing,
    /// Energy saving
    EnergySaving,
    /// Network slicing optimization
    NetworkSlicing,
    /// Spectrum sharing
    SpectrumSharing,
    /// Custom use case
    Custom(u8),
}

/// AI/ML model lifecycle state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AiMlModelState {
    /// Model is idle (not deployed)
    Idle,
    /// Model is being downloaded/deployed
    Deploying,
    /// Model is deployed and ready
    Ready,
    /// Model is actively running inference
    Active,
    /// Model is being updated
    Updating,
    /// Model is deactivated (still deployed but not running)
    Deactivated,
    /// Model has been released/removed
    Released,
}

/// Inference mode for AI/ML model
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AiMlInferenceMode {
    /// UE-side inference only
    UeSide,
    /// Network-side inference only
    NetworkSide,
    /// Joint inference (split between UE and network)
    Joint,
    /// Two-sided model (encoder at UE, decoder at network)
    TwoSided,
}

/// Model performance monitoring configuration
#[derive(Debug, Clone)]
pub struct AiMlPerformanceConfig {
    /// Performance monitoring periodicity in milliseconds
    pub monitoring_periodicity_ms: u32,
    /// Minimum acceptable accuracy (0.0 to 1.0)
    pub min_accuracy_threshold: f64,
    /// Maximum acceptable latency in milliseconds
    pub max_latency_ms: u32,
    /// Enable automatic model fallback on performance degradation
    pub auto_fallback_enabled: bool,
    /// Number of consecutive failures before fallback
    pub fallback_threshold_count: u8,
}

/// Federated learning configuration
#[derive(Debug, Clone)]
pub struct FederatedLearningConfig {
    /// Federated learning round ID
    pub round_id: u32,
    /// Global model version
    pub global_model_version: String,
    /// Local training epochs
    pub local_epochs: u16,
    /// Learning rate
    pub learning_rate: f64,
    /// Batch size for local training
    pub batch_size: u16,
    /// Minimum number of samples before participating
    pub min_samples: u32,
    /// Differential privacy epsilon (privacy budget)
    pub dp_epsilon: Option<f64>,
    /// Gradient clipping norm
    pub gradient_clip_norm: Option<f64>,
    /// Aggregation strategy
    pub aggregation_strategy: AggregationStrategy,
}

/// Aggregation strategy for federated learning
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationStrategy {
    /// Federated Averaging (FedAvg)
    FedAvg,
    /// Federated SGD
    FedSgd,
    /// Weighted averaging
    WeightedAvg,
    /// Secure aggregation
    SecureAgg,
}

/// Split inference configuration
#[derive(Debug, Clone)]
pub struct SplitInferenceConfig {
    /// Split point layer index (where to split the model)
    pub split_point: u16,
    /// Maximum intermediate data size in bytes
    pub max_intermediate_size_bytes: u32,
    /// Compression type for intermediate data
    pub intermediate_compression: IntermediateCompression,
    /// Quantization bits for intermediate data (0 = no quantization)
    pub quantization_bits: u8,
    /// Maximum allowed one-way latency for split inference (ms)
    pub max_one_way_latency_ms: u32,
}

/// Compression type for intermediate data in split inference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntermediateCompression {
    /// No compression
    None,
    /// Entropy coding
    EntropyCoding,
    /// Learned compression
    LearnedCompression,
    /// Quantization-based compression
    Quantization,
}

/// AI/ML reporting configuration
#[derive(Debug, Clone)]
pub struct AiMlReportingConfig {
    /// Reporting type
    pub reporting_type: AiMlReportingType,
    /// Reporting periodicity in milliseconds (for periodic reporting)
    pub periodicity_ms: Option<u32>,
    /// Maximum number of reports (0 = unlimited)
    pub max_reports: u32,
    /// Include model performance metrics in report
    pub include_performance_metrics: bool,
    /// Include data distribution statistics in report
    pub include_data_statistics: bool,
    /// Include resource utilization in report
    pub include_resource_utilization: bool,
}

/// AI/ML reporting type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AiMlReportingType {
    /// Periodic reporting
    Periodic,
    /// Event-triggered reporting (on performance degradation)
    EventTriggered,
    /// One-shot reporting
    OneShot,
    /// Federated learning update reporting
    FederatedUpdate,
}

/// AI/ML model configuration
///
/// Contains the full configuration for an AI/ML model deployment,
/// including model identity, inference mode, performance monitoring,
/// and reporting configuration.
#[derive(Debug, Clone)]
pub struct AiMlConfig {
    /// Configuration ID (unique identifier)
    pub config_id: u16,
    /// Model ID (reference to the AI/ML model)
    pub model_id: String,
    /// Model version
    pub model_version: String,
    /// Use case for this model
    pub use_case: AiMlUseCase,
    /// Current model lifecycle state
    pub model_state: AiMlModelState,
    /// Inference mode
    pub inference_mode: AiMlInferenceMode,
    /// Model input dimensionality
    pub input_dimensions: Vec<u32>,
    /// Model output dimensionality
    pub output_dimensions: Vec<u32>,
    /// Performance monitoring configuration
    pub performance_config: Option<AiMlPerformanceConfig>,
    /// Federated learning configuration (if applicable)
    pub federated_config: Option<FederatedLearningConfig>,
    /// Split inference configuration (if applicable)
    pub split_inference_config: Option<SplitInferenceConfig>,
    /// Reporting configuration
    pub reporting_config: Option<AiMlReportingConfig>,
    /// Model validity timer in seconds (0 = indefinite)
    pub validity_timer_s: u32,
    /// Fallback to traditional algorithm on model failure
    pub fallback_to_traditional: bool,
}

/// AI/ML model activation/deactivation command
#[derive(Debug, Clone)]
pub struct AiMlModelCommand {
    /// Configuration ID of the model
    pub config_id: u16,
    /// Command type
    pub command: AiMlModelCommandType,
    /// Optional new model data (for update command)
    pub model_data: Option<Vec<u8>>,
}

/// AI/ML model command type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AiMlModelCommandType {
    /// Activate the model
    Activate,
    /// Deactivate the model
    Deactivate,
    /// Update the model
    Update,
    /// Release the model
    Release,
    /// Reset model to initial state
    Reset,
}

/// AI/ML performance report
#[derive(Debug, Clone)]
pub struct AiMlPerformanceReport {
    /// Configuration ID
    pub config_id: u16,
    /// Model ID
    pub model_id: String,
    /// Current accuracy (0.0 to 1.0)
    pub accuracy: Option<f64>,
    /// Average inference latency in milliseconds
    pub avg_latency_ms: Option<f64>,
    /// Number of inferences performed
    pub inference_count: u64,
    /// Number of fallback activations
    pub fallback_count: u32,
    /// Resource utilization (0.0 to 1.0)
    pub resource_utilization: Option<f64>,
    /// Timestamp of report in ms since epoch
    pub timestamp_ms: u64,
}

impl AiMlConfig {
    /// Validate the AI/ML configuration
    pub fn validate(&self) -> Result<(), AiMlConfigError> {
        if self.model_id.is_empty() {
            return Err(AiMlConfigError::MissingMandatoryField(
                "model_id".to_string(),
            ));
        }
        if self.model_version.is_empty() {
            return Err(AiMlConfigError::MissingMandatoryField(
                "model_version".to_string(),
            ));
        }
        if self.input_dimensions.is_empty() {
            return Err(AiMlConfigError::MissingMandatoryField(
                "input_dimensions".to_string(),
            ));
        }
        if self.output_dimensions.is_empty() {
            return Err(AiMlConfigError::MissingMandatoryField(
                "output_dimensions".to_string(),
            ));
        }

        // Validate split inference config consistency
        if let Some(ref split) = self.split_inference_config {
            if self.inference_mode != AiMlInferenceMode::Joint
                && self.inference_mode != AiMlInferenceMode::TwoSided
            {
                return Err(AiMlConfigError::InvalidConfig(
                    "Split inference config requires Joint or TwoSided inference mode".to_string(),
                ));
            }
            if split.quantization_bits > 32 {
                return Err(AiMlConfigError::InvalidConfig(
                    "Quantization bits must be 0-32".to_string(),
                ));
            }
        }

        // Validate federated learning config
        if let Some(ref fl) = self.federated_config {
            if fl.learning_rate <= 0.0 || fl.learning_rate > 1.0 {
                return Err(AiMlConfigError::InvalidConfig(
                    "Learning rate must be in range (0.0, 1.0]".to_string(),
                ));
            }
            if let Some(epsilon) = fl.dp_epsilon {
                if epsilon <= 0.0 {
                    return Err(AiMlConfigError::InvalidConfig(
                        "Differential privacy epsilon must be > 0".to_string(),
                    ));
                }
            }
        }

        // Validate performance config
        if let Some(ref perf) = self.performance_config {
            if perf.min_accuracy_threshold < 0.0 || perf.min_accuracy_threshold > 1.0 {
                return Err(AiMlConfigError::InvalidConfig(
                    "Min accuracy threshold must be in range [0.0, 1.0]".to_string(),
                ));
            }
        }

        Ok(())
    }
}

impl AiMlPerformanceReport {
    /// Validate the performance report
    pub fn validate(&self) -> Result<(), AiMlConfigError> {
        if self.model_id.is_empty() {
            return Err(AiMlConfigError::MissingMandatoryField(
                "model_id".to_string(),
            ));
        }
        if let Some(acc) = self.accuracy {
            if !(0.0..=1.0).contains(&acc) {
                return Err(AiMlConfigError::InvalidConfig(
                    "Accuracy must be in range [0.0, 1.0]".to_string(),
                ));
            }
        }
        if let Some(util) = self.resource_utilization {
            if !(0.0..=1.0).contains(&util) {
                return Err(AiMlConfigError::InvalidConfig(
                    "Resource utilization must be in range [0.0, 1.0]".to_string(),
                ));
            }
        }
        Ok(())
    }
}

/// Encode AI/ML config to bytes (simplified serialization)
pub fn encode_ai_ml_config(config: &AiMlConfig) -> Result<Vec<u8>, AiMlConfigError> {
    config.validate()?;
    let mut bytes = Vec::with_capacity(64);

    // config_id (2 bytes)
    bytes.extend_from_slice(&config.config_id.to_be_bytes());
    // use_case (1 byte)
    bytes.push(match config.use_case {
        AiMlUseCase::CsiFeedback => 0,
        AiMlUseCase::BeamManagement => 1,
        AiMlUseCase::Positioning => 2,
        AiMlUseCase::ChannelEstimation => 3,
        AiMlUseCase::LinkAdaptation => 4,
        AiMlUseCase::MobilityOptimization => 5,
        AiMlUseCase::LoadBalancing => 6,
        AiMlUseCase::EnergySaving => 7,
        AiMlUseCase::NetworkSlicing => 8,
        AiMlUseCase::SpectrumSharing => 9,
        AiMlUseCase::Custom(c) => 128 + c,
    });
    // model_state (1 byte)
    bytes.push(match config.model_state {
        AiMlModelState::Idle => 0,
        AiMlModelState::Deploying => 1,
        AiMlModelState::Ready => 2,
        AiMlModelState::Active => 3,
        AiMlModelState::Updating => 4,
        AiMlModelState::Deactivated => 5,
        AiMlModelState::Released => 6,
    });
    // inference_mode (1 byte)
    bytes.push(match config.inference_mode {
        AiMlInferenceMode::UeSide => 0,
        AiMlInferenceMode::NetworkSide => 1,
        AiMlInferenceMode::Joint => 2,
        AiMlInferenceMode::TwoSided => 3,
    });
    // validity_timer_s (4 bytes)
    bytes.extend_from_slice(&config.validity_timer_s.to_be_bytes());
    // fallback_to_traditional (1 byte)
    bytes.push(if config.fallback_to_traditional { 1 } else { 0 });
    // model_id length + data
    let model_id_bytes = config.model_id.as_bytes();
    bytes.push(model_id_bytes.len() as u8);
    bytes.extend_from_slice(model_id_bytes);

    Ok(bytes)
}

/// Decode AI/ML config header from bytes (simplified deserialization)
pub fn decode_ai_ml_config_header(
    bytes: &[u8],
) -> Result<(u16, AiMlUseCase, AiMlModelState, AiMlInferenceMode), AiMlConfigError> {
    if bytes.len() < 5 {
        return Err(AiMlConfigError::CodecError(
            "Insufficient bytes for AI/ML config header".to_string(),
        ));
    }

    let config_id = u16::from_be_bytes(bytes[0..2].try_into().unwrap());
    let use_case = match bytes[2] {
        0 => AiMlUseCase::CsiFeedback,
        1 => AiMlUseCase::BeamManagement,
        2 => AiMlUseCase::Positioning,
        3 => AiMlUseCase::ChannelEstimation,
        4 => AiMlUseCase::LinkAdaptation,
        5 => AiMlUseCase::MobilityOptimization,
        6 => AiMlUseCase::LoadBalancing,
        7 => AiMlUseCase::EnergySaving,
        8 => AiMlUseCase::NetworkSlicing,
        9 => AiMlUseCase::SpectrumSharing,
        v if v >= 128 => AiMlUseCase::Custom(v - 128),
        _ => return Err(AiMlConfigError::CodecError("Unknown AI/ML use case".to_string())),
    };
    let model_state = match bytes[3] {
        0 => AiMlModelState::Idle,
        1 => AiMlModelState::Deploying,
        2 => AiMlModelState::Ready,
        3 => AiMlModelState::Active,
        4 => AiMlModelState::Updating,
        5 => AiMlModelState::Deactivated,
        6 => AiMlModelState::Released,
        _ => return Err(AiMlConfigError::CodecError("Unknown model state".to_string())),
    };
    let inference_mode = match bytes[4] {
        0 => AiMlInferenceMode::UeSide,
        1 => AiMlInferenceMode::NetworkSide,
        2 => AiMlInferenceMode::Joint,
        3 => AiMlInferenceMode::TwoSided,
        _ => return Err(AiMlConfigError::CodecError("Unknown inference mode".to_string())),
    };

    Ok((config_id, use_case, model_state, inference_mode))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> AiMlConfig {
        AiMlConfig {
            config_id: 1,
            model_id: "csi-feedback-v2".to_string(),
            model_version: "2.1.0".to_string(),
            use_case: AiMlUseCase::CsiFeedback,
            model_state: AiMlModelState::Active,
            inference_mode: AiMlInferenceMode::TwoSided,
            input_dimensions: vec![32, 64],
            output_dimensions: vec![16],
            performance_config: Some(AiMlPerformanceConfig {
                monitoring_periodicity_ms: 1000,
                min_accuracy_threshold: 0.9,
                max_latency_ms: 10,
                auto_fallback_enabled: true,
                fallback_threshold_count: 3,
            }),
            federated_config: None,
            split_inference_config: Some(SplitInferenceConfig {
                split_point: 4,
                max_intermediate_size_bytes: 2048,
                intermediate_compression: IntermediateCompression::LearnedCompression,
                quantization_bits: 8,
                max_one_way_latency_ms: 5,
            }),
            reporting_config: Some(AiMlReportingConfig {
                reporting_type: AiMlReportingType::Periodic,
                periodicity_ms: Some(5000),
                max_reports: 0,
                include_performance_metrics: true,
                include_data_statistics: false,
                include_resource_utilization: true,
            }),
            validity_timer_s: 3600,
            fallback_to_traditional: true,
        }
    }

    #[test]
    fn test_ai_ml_config_validate() {
        let config = create_test_config();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_ai_ml_config_missing_model_id() {
        let mut config = create_test_config();
        config.model_id = String::new();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_ai_ml_config_missing_version() {
        let mut config = create_test_config();
        config.model_version = String::new();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_ai_ml_config_empty_dimensions() {
        let mut config = create_test_config();
        config.input_dimensions = vec![];
        assert!(config.validate().is_err());

        let mut config = create_test_config();
        config.output_dimensions = vec![];
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_ai_ml_config_split_inference_wrong_mode() {
        let mut config = create_test_config();
        config.inference_mode = AiMlInferenceMode::UeSide;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_ai_ml_config_split_inference_valid_modes() {
        let mut config = create_test_config();
        config.inference_mode = AiMlInferenceMode::Joint;
        assert!(config.validate().is_ok());

        config.inference_mode = AiMlInferenceMode::TwoSided;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_ai_ml_config_invalid_quantization_bits() {
        let mut config = create_test_config();
        if let Some(ref mut split) = config.split_inference_config {
            split.quantization_bits = 33;
        }
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_ai_ml_federated_config_valid() {
        let mut config = create_test_config();
        config.split_inference_config = None;
        config.inference_mode = AiMlInferenceMode::UeSide;
        config.federated_config = Some(FederatedLearningConfig {
            round_id: 1,
            global_model_version: "1.0".to_string(),
            local_epochs: 5,
            learning_rate: 0.01,
            batch_size: 32,
            min_samples: 100,
            dp_epsilon: Some(1.0),
            gradient_clip_norm: Some(1.0),
            aggregation_strategy: AggregationStrategy::FedAvg,
        });
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_ai_ml_federated_config_invalid_lr() {
        let mut config = create_test_config();
        config.split_inference_config = None;
        config.inference_mode = AiMlInferenceMode::UeSide;
        config.federated_config = Some(FederatedLearningConfig {
            round_id: 1,
            global_model_version: "1.0".to_string(),
            local_epochs: 5,
            learning_rate: 0.0, // invalid
            batch_size: 32,
            min_samples: 100,
            dp_epsilon: None,
            gradient_clip_norm: None,
            aggregation_strategy: AggregationStrategy::FedAvg,
        });
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_ai_ml_federated_config_invalid_dp() {
        let mut config = create_test_config();
        config.split_inference_config = None;
        config.inference_mode = AiMlInferenceMode::UeSide;
        config.federated_config = Some(FederatedLearningConfig {
            round_id: 1,
            global_model_version: "1.0".to_string(),
            local_epochs: 5,
            learning_rate: 0.01,
            batch_size: 32,
            min_samples: 100,
            dp_epsilon: Some(-1.0), // invalid
            gradient_clip_norm: None,
            aggregation_strategy: AggregationStrategy::FedAvg,
        });
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_ai_ml_performance_config_invalid_accuracy() {
        let mut config = create_test_config();
        if let Some(ref mut perf) = config.performance_config {
            perf.min_accuracy_threshold = 1.5;
        }
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_ai_ml_performance_report_validate() {
        let report = AiMlPerformanceReport {
            config_id: 1,
            model_id: "test-model".to_string(),
            accuracy: Some(0.95),
            avg_latency_ms: Some(5.0),
            inference_count: 1000,
            fallback_count: 2,
            resource_utilization: Some(0.6),
            timestamp_ms: 1700000000000,
        };
        assert!(report.validate().is_ok());
    }

    #[test]
    fn test_ai_ml_performance_report_invalid_accuracy() {
        let report = AiMlPerformanceReport {
            config_id: 1,
            model_id: "test-model".to_string(),
            accuracy: Some(1.5), // invalid
            avg_latency_ms: None,
            inference_count: 0,
            fallback_count: 0,
            resource_utilization: None,
            timestamp_ms: 0,
        };
        assert!(report.validate().is_err());
    }

    #[test]
    fn test_ai_ml_performance_report_invalid_utilization() {
        let report = AiMlPerformanceReport {
            config_id: 1,
            model_id: "test-model".to_string(),
            accuracy: None,
            avg_latency_ms: None,
            inference_count: 0,
            fallback_count: 0,
            resource_utilization: Some(-0.1), // invalid
            timestamp_ms: 0,
        };
        assert!(report.validate().is_err());
    }

    #[test]
    fn test_ai_ml_model_command() {
        let cmd = AiMlModelCommand {
            config_id: 1,
            command: AiMlModelCommandType::Activate,
            model_data: None,
        };
        assert_eq!(cmd.command, AiMlModelCommandType::Activate);
    }

    #[test]
    fn test_ai_ml_model_command_update_with_data() {
        let cmd = AiMlModelCommand {
            config_id: 1,
            command: AiMlModelCommandType::Update,
            model_data: Some(vec![0x01, 0x02, 0x03]),
        };
        assert_eq!(cmd.command, AiMlModelCommandType::Update);
        assert!(cmd.model_data.is_some());
    }

    #[test]
    fn test_encode_decode_ai_ml_config() {
        let config = create_test_config();
        let encoded = encode_ai_ml_config(&config).expect("Failed to encode");
        assert!(!encoded.is_empty());

        let (config_id, use_case, model_state, inference_mode) =
            decode_ai_ml_config_header(&encoded).expect("Failed to decode");
        assert_eq!(config_id, 1);
        assert_eq!(use_case, AiMlUseCase::CsiFeedback);
        assert_eq!(model_state, AiMlModelState::Active);
        assert_eq!(inference_mode, AiMlInferenceMode::TwoSided);
    }

    #[test]
    fn test_all_use_cases() {
        let use_cases = [
            AiMlUseCase::CsiFeedback,
            AiMlUseCase::BeamManagement,
            AiMlUseCase::Positioning,
            AiMlUseCase::ChannelEstimation,
            AiMlUseCase::LinkAdaptation,
            AiMlUseCase::MobilityOptimization,
            AiMlUseCase::LoadBalancing,
            AiMlUseCase::EnergySaving,
            AiMlUseCase::NetworkSlicing,
            AiMlUseCase::SpectrumSharing,
            AiMlUseCase::Custom(42),
        ];

        for use_case in use_cases {
            let mut config = create_test_config();
            config.use_case = use_case;
            let encoded = encode_ai_ml_config(&config).expect("Failed to encode");
            let (_, decoded_use_case, _, _) =
                decode_ai_ml_config_header(&encoded).expect("Failed to decode");
            assert_eq!(decoded_use_case, use_case);
        }
    }

    #[test]
    fn test_all_model_states() {
        let states = [
            AiMlModelState::Idle,
            AiMlModelState::Deploying,
            AiMlModelState::Ready,
            AiMlModelState::Active,
            AiMlModelState::Updating,
            AiMlModelState::Deactivated,
            AiMlModelState::Released,
        ];

        for state in states {
            let mut config = create_test_config();
            config.model_state = state;
            let encoded = encode_ai_ml_config(&config).expect("Failed to encode");
            let (_, _, decoded_state, _) =
                decode_ai_ml_config_header(&encoded).expect("Failed to decode");
            assert_eq!(decoded_state, state);
        }
    }

    #[test]
    fn test_all_aggregation_strategies() {
        let strategies = [
            AggregationStrategy::FedAvg,
            AggregationStrategy::FedSgd,
            AggregationStrategy::WeightedAvg,
            AggregationStrategy::SecureAgg,
        ];

        for strategy in strategies {
            let fl_config = FederatedLearningConfig {
                round_id: 1,
                global_model_version: "1.0".to_string(),
                local_epochs: 5,
                learning_rate: 0.01,
                batch_size: 32,
                min_samples: 100,
                dp_epsilon: None,
                gradient_clip_norm: None,
                aggregation_strategy: strategy,
            };
            assert_eq!(fl_config.aggregation_strategy, strategy);
        }
    }

    #[test]
    fn test_network_side_inference() {
        let mut config = create_test_config();
        config.inference_mode = AiMlInferenceMode::NetworkSide;
        config.split_inference_config = None;
        assert!(config.validate().is_ok());
    }
}
