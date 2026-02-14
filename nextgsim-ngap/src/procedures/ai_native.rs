//! AI-Native NGAP Procedures
//!
//! 6G extension: NGAP procedures for AI/ML operations between gNB and AMF.
//!
//! This module implements:
//! - `AiModelTransfer` - Transfer AI model between nodes
//! - `AiInferenceRequest` / `AiInferenceResponse` - Distributed inference

use thiserror::Error;

/// Errors that can occur during AI-native procedures
#[derive(Debug, Error)]
pub enum AiNativeError {
    /// Invalid model configuration
    #[error("Invalid AI model configuration: {0}")]
    InvalidModelConfig(String),

    /// Model transfer error
    #[error("Model transfer error: {0}")]
    TransferError(String),

    /// Inference error
    #[error("Inference error: {0}")]
    InferenceError(String),

    /// Missing mandatory field
    #[error("Missing mandatory field: {0}")]
    MissingMandatoryField(String),
}

/// AI/ML model type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AiModelType {
    /// Neural network model
    NeuralNetwork,
    /// Reinforcement learning model
    ReinforcementLearning,
    /// Federated learning model
    FederatedLearning,
    /// Transfer learning model
    TransferLearning,
    /// Model for beam management
    BeamManagement,
    /// Model for positioning
    Positioning,
    /// Model for CSI feedback
    CsiFeedback,
    /// Custom model type
    Custom(u8),
}

/// AI model format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AiModelFormat {
    /// ONNX format
    Onnx,
    /// TensorFlow Lite format
    TfLite,
    /// `PyTorch` serialized format
    PyTorch,
    /// Custom binary format
    CustomBinary,
}

/// AI inference mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AiInferenceMode {
    /// Run inference at gNB
    Local,
    /// Run inference at AMF/core
    Remote,
    /// Split inference between gNB and AMF
    Split,
    /// Collaborative inference across multiple nodes
    Collaborative,
}

/// AI model transfer direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AiModelTransferDirection {
    /// AMF to gNB (model deployment)
    AmfToGnb,
    /// gNB to AMF (model upload/federated aggregation)
    GnbToAmf,
    /// gNB to gNB (peer transfer via AMF relay)
    GnbToGnb,
}

/// AI model metadata
#[derive(Debug, Clone)]
pub struct AiModelMetadata {
    /// Unique model identifier
    pub model_id: String,
    /// Model version
    pub model_version: u32,
    /// Model type
    pub model_type: AiModelType,
    /// Model format
    pub model_format: AiModelFormat,
    /// Model size in bytes
    pub model_size_bytes: u64,
    /// Input feature dimensions
    pub input_dimensions: Vec<u32>,
    /// Output dimensions
    pub output_dimensions: Vec<u32>,
    /// Model accuracy/performance metric
    pub performance_metric: Option<f64>,
    /// Training dataset identifier
    pub training_dataset_id: Option<String>,
    /// Timestamp of model creation (ms since epoch)
    pub created_at_ms: u64,
}

/// AI model transfer request
#[derive(Debug, Clone)]
pub struct AiModelTransfer {
    /// Transfer ID
    pub transfer_id: u64,
    /// Transfer direction
    pub direction: AiModelTransferDirection,
    /// Model metadata
    pub model_metadata: AiModelMetadata,
    /// Model data (serialized model bytes)
    pub model_data: Vec<u8>,
    /// Source node ID
    pub source_node_id: u32,
    /// Destination node ID
    pub destination_node_id: u32,
    /// Priority (0 = lowest, 15 = highest)
    pub priority: u8,
    /// Whether this is a delta update (incremental)
    pub is_delta_update: bool,
    /// Compression algorithm used (if any)
    pub compression: Option<String>,
}

/// AI model transfer acknowledgment
#[derive(Debug, Clone)]
pub struct AiModelTransferAck {
    /// Transfer ID being acknowledged
    pub transfer_id: u64,
    /// Whether transfer was successful
    pub success: bool,
    /// Error message (if failed)
    pub error_message: Option<String>,
    /// Time taken to receive and validate (ms)
    pub processing_time_ms: Option<u64>,
}

/// AI inference input data
#[derive(Debug, Clone)]
pub struct AiInferenceInput {
    /// Input feature data (serialized tensor)
    pub features: Vec<u8>,
    /// Input data format description
    pub data_format: String,
    /// Number of samples in batch
    pub batch_size: u32,
}

/// AI inference request
#[derive(Debug, Clone)]
pub struct AiInferenceRequest {
    /// Request ID
    pub request_id: u64,
    /// Model ID to use for inference
    pub model_id: String,
    /// Model version (optional, latest if not specified)
    pub model_version: Option<u32>,
    /// Inference mode
    pub inference_mode: AiInferenceMode,
    /// AMF UE NGAP ID (optional, for UE-specific inference)
    pub amf_ue_ngap_id: Option<u64>,
    /// RAN UE NGAP ID (optional, for UE-specific inference)
    pub ran_ue_ngap_id: Option<u32>,
    /// Input data
    pub input: AiInferenceInput,
    /// Maximum latency requirement in ms
    pub max_latency_ms: Option<u32>,
    /// Request timestamp (ms since epoch)
    pub timestamp_ms: u64,
}

/// AI inference result
#[derive(Debug, Clone)]
pub struct AiInferenceResult {
    /// Output data (serialized tensor)
    pub output: Vec<u8>,
    /// Output data format description
    pub data_format: String,
    /// Confidence/probability score (optional)
    pub confidence: Option<f64>,
}

/// AI inference response
#[derive(Debug, Clone)]
pub struct AiInferenceResponse {
    /// Request ID this response corresponds to
    pub request_id: u64,
    /// Whether inference was successful
    pub success: bool,
    /// Inference result (if successful)
    pub result: Option<AiInferenceResult>,
    /// Error message (if failed)
    pub error_message: Option<String>,
    /// Inference latency in ms
    pub latency_ms: u64,
    /// Node that performed the inference
    pub serving_node_id: u32,
}

impl AiModelTransfer {
    /// Validate the model transfer request
    pub fn validate(&self) -> Result<(), AiNativeError> {
        if self.model_metadata.model_id.is_empty() {
            return Err(AiNativeError::InvalidModelConfig(
                "Model ID cannot be empty".to_string(),
            ));
        }
        if self.model_data.is_empty() {
            return Err(AiNativeError::TransferError(
                "Model data cannot be empty".to_string(),
            ));
        }
        if self.priority > 15 {
            return Err(AiNativeError::InvalidModelConfig(
                "Priority must be 0-15".to_string(),
            ));
        }
        if self.model_metadata.model_size_bytes != self.model_data.len() as u64 {
            return Err(AiNativeError::InvalidModelConfig(
                "Model size metadata does not match actual data size".to_string(),
            ));
        }
        Ok(())
    }
}

impl AiInferenceRequest {
    /// Validate the inference request
    pub fn validate(&self) -> Result<(), AiNativeError> {
        if self.model_id.is_empty() {
            return Err(AiNativeError::InferenceError(
                "Model ID cannot be empty".to_string(),
            ));
        }
        if self.input.features.is_empty() {
            return Err(AiNativeError::InferenceError(
                "Input features cannot be empty".to_string(),
            ));
        }
        if self.input.batch_size == 0 {
            return Err(AiNativeError::InferenceError(
                "Batch size must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Encode AI model metadata to bytes (simplified serialization)
pub fn encode_ai_model_metadata(metadata: &AiModelMetadata) -> Vec<u8> {
    let mut bytes = Vec::new();
    // model_id length (2 bytes) + model_id
    let id_bytes = metadata.model_id.as_bytes();
    bytes.extend_from_slice(&(id_bytes.len() as u16).to_be_bytes());
    bytes.extend_from_slice(id_bytes);
    // model_version (4 bytes)
    bytes.extend_from_slice(&metadata.model_version.to_be_bytes());
    // model_type (1 byte)
    bytes.push(match metadata.model_type {
        AiModelType::NeuralNetwork => 0,
        AiModelType::ReinforcementLearning => 1,
        AiModelType::FederatedLearning => 2,
        AiModelType::TransferLearning => 3,
        AiModelType::BeamManagement => 4,
        AiModelType::Positioning => 5,
        AiModelType::CsiFeedback => 6,
        AiModelType::Custom(v) => v,
    });
    // model_format (1 byte)
    bytes.push(match metadata.model_format {
        AiModelFormat::Onnx => 0,
        AiModelFormat::TfLite => 1,
        AiModelFormat::PyTorch => 2,
        AiModelFormat::CustomBinary => 3,
    });
    // model_size_bytes (8 bytes)
    bytes.extend_from_slice(&metadata.model_size_bytes.to_be_bytes());
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_model_metadata() -> AiModelMetadata {
        AiModelMetadata {
            model_id: "beam-mgmt-v1".to_string(),
            model_version: 1,
            model_type: AiModelType::BeamManagement,
            model_format: AiModelFormat::Onnx,
            model_size_bytes: 4,
            input_dimensions: vec![1, 64],
            output_dimensions: vec![1, 8],
            performance_metric: Some(0.95),
            training_dataset_id: Some("dataset-001".to_string()),
            created_at_ms: 1700000000000,
        }
    }

    #[test]
    fn test_ai_model_transfer_validate() {
        let transfer = AiModelTransfer {
            transfer_id: 1,
            direction: AiModelTransferDirection::AmfToGnb,
            model_metadata: create_test_model_metadata(),
            model_data: vec![0x01, 0x02, 0x03, 0x04],
            source_node_id: 1,
            destination_node_id: 2,
            priority: 5,
            is_delta_update: false,
            compression: None,
        };

        assert!(transfer.validate().is_ok());
    }

    #[test]
    fn test_ai_model_transfer_empty_data() {
        let transfer = AiModelTransfer {
            transfer_id: 1,
            direction: AiModelTransferDirection::AmfToGnb,
            model_metadata: create_test_model_metadata(),
            model_data: vec![],
            source_node_id: 1,
            destination_node_id: 2,
            priority: 5,
            is_delta_update: false,
            compression: None,
        };

        assert!(transfer.validate().is_err());
    }

    #[test]
    fn test_ai_model_transfer_invalid_priority() {
        let mut metadata = create_test_model_metadata();
        metadata.model_size_bytes = 4;
        let transfer = AiModelTransfer {
            transfer_id: 1,
            direction: AiModelTransferDirection::GnbToAmf,
            model_metadata: metadata,
            model_data: vec![0x01, 0x02, 0x03, 0x04],
            source_node_id: 1,
            destination_node_id: 2,
            priority: 20, // Invalid: > 15
            is_delta_update: false,
            compression: None,
        };

        assert!(transfer.validate().is_err());
    }

    #[test]
    fn test_ai_inference_request_validate() {
        let request = AiInferenceRequest {
            request_id: 1,
            model_id: "beam-mgmt-v1".to_string(),
            model_version: Some(1),
            inference_mode: AiInferenceMode::Local,
            amf_ue_ngap_id: Some(12345),
            ran_ue_ngap_id: Some(67890),
            input: AiInferenceInput {
                features: vec![0x01, 0x02, 0x03],
                data_format: "float32".to_string(),
                batch_size: 1,
            },
            max_latency_ms: Some(10),
            timestamp_ms: 1700000000000,
        };

        assert!(request.validate().is_ok());
    }

    #[test]
    fn test_ai_inference_request_empty_model_id() {
        let request = AiInferenceRequest {
            request_id: 1,
            model_id: "".to_string(),
            model_version: None,
            inference_mode: AiInferenceMode::Remote,
            amf_ue_ngap_id: None,
            ran_ue_ngap_id: None,
            input: AiInferenceInput {
                features: vec![0x01],
                data_format: "float32".to_string(),
                batch_size: 1,
            },
            max_latency_ms: None,
            timestamp_ms: 1700000000000,
        };

        assert!(request.validate().is_err());
    }

    #[test]
    fn test_ai_inference_response() {
        let response = AiInferenceResponse {
            request_id: 1,
            success: true,
            result: Some(AiInferenceResult {
                output: vec![0x00, 0x01, 0x02],
                data_format: "float32".to_string(),
                confidence: Some(0.92),
            }),
            error_message: None,
            latency_ms: 5,
            serving_node_id: 100,
        };

        assert!(response.success);
        assert!(response.result.is_some());
        assert_eq!(response.latency_ms, 5);
    }

    #[test]
    fn test_ai_model_transfer_ack() {
        let ack = AiModelTransferAck {
            transfer_id: 1,
            success: true,
            error_message: None,
            processing_time_ms: Some(100),
        };

        assert!(ack.success);
        assert_eq!(ack.processing_time_ms, Some(100));
    }

    #[test]
    fn test_encode_ai_model_metadata() {
        let metadata = create_test_model_metadata();
        let encoded = encode_ai_model_metadata(&metadata);
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_all_inference_modes() {
        let modes = [
            AiInferenceMode::Local,
            AiInferenceMode::Remote,
            AiInferenceMode::Split,
            AiInferenceMode::Collaborative,
        ];

        for mode in modes {
            let request = AiInferenceRequest {
                request_id: 1,
                model_id: "test".to_string(),
                model_version: None,
                inference_mode: mode,
                amf_ue_ngap_id: None,
                ran_ue_ngap_id: None,
                input: AiInferenceInput {
                    features: vec![0x01],
                    data_format: "float32".to_string(),
                    batch_size: 1,
                },
                max_latency_ms: None,
                timestamp_ms: 1700000000000,
            };
            assert!(request.validate().is_ok());
        }
    }

    #[test]
    fn test_all_model_types() {
        let types = [
            AiModelType::NeuralNetwork,
            AiModelType::ReinforcementLearning,
            AiModelType::FederatedLearning,
            AiModelType::TransferLearning,
            AiModelType::BeamManagement,
            AiModelType::Positioning,
            AiModelType::CsiFeedback,
            AiModelType::Custom(100),
        ];

        for model_type in types {
            let metadata = AiModelMetadata {
                model_id: "test".to_string(),
                model_version: 1,
                model_type,
                model_format: AiModelFormat::Onnx,
                model_size_bytes: 1,
                input_dimensions: vec![1],
                output_dimensions: vec![1],
                performance_metric: None,
                training_dataset_id: None,
                created_at_ms: 0,
            };
            let encoded = encode_ai_model_metadata(&metadata);
            assert!(!encoded.is_empty());
        }
    }
}
