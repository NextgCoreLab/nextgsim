//! Integration with Service Hosting Environment (SHE) for edge semantic coding
//!
//! This module provides interfaces for semantic communication to offload
//! encoding/decoding workloads to the Service Hosting Environment.

#![allow(missing_docs)]

use crate::{SemanticFeatures, SemanticTask};
use serde::{Deserialize, Serialize};

/// Semantic coding workload request for SHE
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticCodingRequest {
    /// Workload ID
    pub workload_id: u64,
    /// Coding operation type
    pub operation: SemanticCodingOperation,
    /// Input data
    pub input: SemanticCodingInput,
    /// Task type
    pub task: SemanticTask,
    /// Quality parameters
    pub quality_params: QualityParameters,
    /// Latency requirement (milliseconds)
    pub latency_requirement_ms: u32,
    /// Priority (higher = more urgent)
    pub priority: u8,
}

/// Type of semantic coding operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SemanticCodingOperation {
    /// Encode raw data to semantic features
    Encode,
    /// Decode semantic features to reconstructed data
    Decode,
    /// Joint encode-decode (for testing/validation)
    Roundtrip,
    /// Adaptive encoding based on channel conditions
    AdaptiveEncode,
}

impl std::fmt::Display for SemanticCodingOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SemanticCodingOperation::Encode => write!(f, "Encode"),
            SemanticCodingOperation::Decode => write!(f, "Decode"),
            SemanticCodingOperation::Roundtrip => write!(f, "Roundtrip"),
            SemanticCodingOperation::AdaptiveEncode => write!(f, "AdaptiveEncode"),
        }
    }
}

/// Input data for semantic coding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SemanticCodingInput {
    /// Raw data to encode (e.g., image, audio, sensor data)
    RawData {
        data: Vec<f32>,
        dimensions: Vec<usize>,
    },
    /// Semantic features to decode
    Features(SemanticFeatures),
    /// Both for roundtrip testing
    Both {
        raw_data: Vec<f32>,
        dimensions: Vec<usize>,
    },
}

/// Quality parameters for semantic coding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityParameters {
    /// Target compression ratio
    pub target_compression: f32,
    /// Minimum acceptable quality (0.0 to 1.0)
    pub min_quality: f32,
    /// Enable adaptive compression
    pub adaptive: bool,
    /// Channel SNR (dB) for adaptive coding
    pub channel_snr_db: Option<f32>,
}

impl Default for QualityParameters {
    fn default() -> Self {
        Self {
            target_compression: 8.0,
            min_quality: 0.7,
            adaptive: false,
            channel_snr_db: None,
        }
    }
}

/// Result from semantic coding workload in SHE
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticCodingResult {
    /// Original workload ID
    pub workload_id: u64,
    /// Operation that was executed
    pub operation: SemanticCodingOperation,
    /// Result data
    pub result: SemanticCodingOutput,
    /// Processing latency (milliseconds)
    pub processing_latency_ms: u32,
    /// Tier where processing occurred
    pub processing_tier: String,
    /// Quality metrics
    pub metrics: CodingMetrics,
}

/// Output data from semantic coding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SemanticCodingOutput {
    /// Encoded semantic features
    EncodedFeatures(SemanticFeatures),
    /// Decoded reconstructed data
    DecodedData {
        data: Vec<f32>,
        dimensions: Vec<usize>,
    },
    /// Roundtrip result with both
    Roundtrip {
        features: SemanticFeatures,
        reconstructed: Vec<f32>,
        dimensions: Vec<usize>,
    },
}

/// Coding quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodingMetrics {
    /// Achieved compression ratio
    pub compression_ratio: f32,
    /// Reconstruction quality (cosine similarity or MSE)
    pub quality_score: f32,
    /// Number of features
    pub num_features: usize,
    /// Data size reduction (bytes)
    pub size_reduction_bytes: usize,
}

/// Client for submitting semantic coding workloads to SHE
#[derive(Debug)]
pub struct SheSemanticClient {
    next_workload_id: u64,
}

impl SheSemanticClient {
    /// Creates a new SHE-Semantic client
    pub fn new() -> Self {
        Self {
            next_workload_id: 1,
        }
    }

    /// Submits a semantic coding workload to SHE
    ///
    /// # Arguments
    ///
    /// * `operation` - Type of coding operation (encode/decode/roundtrip)
    /// * `input` - Input data for the operation
    /// * `task` - Semantic task type
    /// * `quality_params` - Quality parameters
    /// * `latency_requirement_ms` - Maximum acceptable latency
    /// * `priority` - Priority level (0-10, higher is more urgent)
    ///
    /// # Returns
    ///
    /// The workload ID assigned to this request
    pub fn submit_coding_workload(
        &mut self,
        operation: SemanticCodingOperation,
        input: SemanticCodingInput,
        task: SemanticTask,
        quality_params: QualityParameters,
        latency_requirement_ms: u32,
        priority: u8,
    ) -> SemanticCodingRequest {
        let workload_id = self.next_workload_id;
        self.next_workload_id += 1;

        SemanticCodingRequest {
            workload_id,
            operation,
            input,
            task,
            quality_params,
            latency_requirement_ms,
            priority,
        }
    }

    /// Submits encoding workload to SHE
    pub fn submit_encode(
        &mut self,
        raw_data: Vec<f32>,
        dimensions: Vec<usize>,
        task: SemanticTask,
        target_compression: f32,
    ) -> SemanticCodingRequest {
        let quality_params = QualityParameters {
            target_compression,
            min_quality: 0.7,
            adaptive: false,
            channel_snr_db: None,
        };

        self.submit_coding_workload(
            SemanticCodingOperation::Encode,
            SemanticCodingInput::RawData {
                data: raw_data,
                dimensions,
            },
            task,
            quality_params,
            15, // 15ms latency requirement for edge
            6,  // Standard priority
        )
    }

    /// Submits decoding workload to SHE
    pub fn submit_decode(
        &mut self,
        features: SemanticFeatures,
        task: SemanticTask,
    ) -> SemanticCodingRequest {
        self.submit_coding_workload(
            SemanticCodingOperation::Decode,
            SemanticCodingInput::Features(features),
            task,
            QualityParameters::default(),
            15, // 15ms latency requirement
            6,
        )
    }

    /// Submits adaptive encoding workload to SHE
    pub fn submit_adaptive_encode(
        &mut self,
        raw_data: Vec<f32>,
        dimensions: Vec<usize>,
        task: SemanticTask,
        channel_snr_db: f32,
        target_compression: f32,
    ) -> SemanticCodingRequest {
        let quality_params = QualityParameters {
            target_compression,
            min_quality: 0.6,
            adaptive: true,
            channel_snr_db: Some(channel_snr_db),
        };

        self.submit_coding_workload(
            SemanticCodingOperation::AdaptiveEncode,
            SemanticCodingInput::RawData {
                data: raw_data,
                dimensions,
            },
            task,
            quality_params,
            20, // Slightly higher latency for adaptive processing
            7,  // Higher priority for adaptive coding
        )
    }

    /// Submits roundtrip workload to SHE (for testing)
    pub fn submit_roundtrip(
        &mut self,
        raw_data: Vec<f32>,
        dimensions: Vec<usize>,
        task: SemanticTask,
    ) -> SemanticCodingRequest {
        self.submit_coding_workload(
            SemanticCodingOperation::Roundtrip,
            SemanticCodingInput::Both {
                raw_data,
                dimensions,
            },
            task,
            QualityParameters::default(),
            25, // Higher latency for roundtrip
            4,  // Lower priority for testing
        )
    }
}

impl Default for SheSemanticClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_she_semantic_client_creation() {
        let client = SheSemanticClient::new();
        assert_eq!(client.next_workload_id, 1);
    }

    #[test]
    fn test_submit_encode_workload() {
        let mut client = SheSemanticClient::new();

        let raw_data: Vec<f32> = (0..256).map(|i| (i as f32) / 255.0).collect();
        let dimensions = vec![16, 16];

        let request = client.submit_encode(
            raw_data.clone(),
            dimensions.clone(),
            SemanticTask::ImageClassification,
            10.0,
        );

        assert_eq!(request.operation, SemanticCodingOperation::Encode);
        assert_eq!(request.task, SemanticTask::ImageClassification);
        assert_eq!(request.quality_params.target_compression, 10.0);
        assert_eq!(request.latency_requirement_ms, 15);
        assert_eq!(request.priority, 6);

        match request.input {
            SemanticCodingInput::RawData { data, dimensions: dims } => {
                assert_eq!(data.len(), 256);
                assert_eq!(dims, vec![16, 16]);
            }
            _ => panic!("Expected RawData input"),
        }
    }

    #[test]
    fn test_submit_decode_workload() {
        let mut client = SheSemanticClient::new();

        let features = SemanticFeatures::new(1, vec![0.1, 0.2, 0.3], vec![256]);

        let request = client.submit_decode(features.clone(), SemanticTask::SensorFusion);

        assert_eq!(request.operation, SemanticCodingOperation::Decode);
        assert_eq!(request.task, SemanticTask::SensorFusion);

        match request.input {
            SemanticCodingInput::Features(f) => {
                assert_eq!(f.features.len(), 3);
            }
            _ => panic!("Expected Features input"),
        }
    }

    #[test]
    fn test_submit_adaptive_encode() {
        let mut client = SheSemanticClient::new();

        let raw_data: Vec<f32> = (0..128).map(|i| (i as f32) / 127.0).collect();
        let dimensions = vec![8, 16];

        let request = client.submit_adaptive_encode(
            raw_data,
            dimensions,
            SemanticTask::VideoAnalytics,
            15.0, // 15 dB SNR
            8.0,
        );

        assert_eq!(request.operation, SemanticCodingOperation::AdaptiveEncode);
        assert_eq!(request.task, SemanticTask::VideoAnalytics);
        assert!(request.quality_params.adaptive);
        assert_eq!(request.quality_params.channel_snr_db, Some(15.0));
        assert_eq!(request.priority, 7);
    }

    #[test]
    fn test_submit_roundtrip() {
        let mut client = SheSemanticClient::new();

        let raw_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let dimensions = vec![2, 2];

        let request = client.submit_roundtrip(raw_data, dimensions, SemanticTask::Custom(42));

        assert_eq!(request.operation, SemanticCodingOperation::Roundtrip);
        assert_eq!(request.task, SemanticTask::Custom(42));
        assert_eq!(request.latency_requirement_ms, 25);
        assert_eq!(request.priority, 4);
    }

    #[test]
    fn test_semantic_coding_operation_display() {
        assert_eq!(
            format!("{}", SemanticCodingOperation::Encode),
            "Encode"
        );
        assert_eq!(
            format!("{}", SemanticCodingOperation::AdaptiveEncode),
            "AdaptiveEncode"
        );
    }

    #[test]
    fn test_quality_parameters_default() {
        let params = QualityParameters::default();
        assert_eq!(params.target_compression, 8.0);
        assert_eq!(params.min_quality, 0.7);
        assert!(!params.adaptive);
        assert_eq!(params.channel_snr_db, None);
    }

    #[test]
    fn test_workload_id_increment() {
        let mut client = SheSemanticClient::new();

        let data = vec![1.0, 2.0];
        let dims = vec![2];

        let req1 = client.submit_encode(data.clone(), dims.clone(), SemanticTask::SensorFusion, 5.0);
        let req2 = client.submit_encode(data, dims, SemanticTask::SensorFusion, 5.0);

        assert_eq!(req1.workload_id, 1);
        assert_eq!(req2.workload_id, 2);
    }
}
