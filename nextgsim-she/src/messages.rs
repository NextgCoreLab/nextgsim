//! Message types for SHE task communication
//!
//! Defines messages for workload placement, resource management, and inference requests.

use nextgsim_ai::TensorData;
use serde::{Deserialize, Serialize};

use crate::tier::ComputeTier;
use crate::workload::{WorkloadId, WorkloadRequirements};

/// Messages for the SHE task
#[derive(Debug)]
pub enum SheMessage {
    // ========================================================================
    // Workload Management
    // ========================================================================

    /// Request to place a workload
    PlaceWorkload {
        /// Unique workload ID
        workload_id: WorkloadId,
        /// Workload requirements
        requirements: WorkloadRequirements,
        /// Response channel
        response_tx: Option<tokio::sync::oneshot::Sender<SheResponse>>,
    },

    /// Workload placement confirmed
    WorkloadPlaced {
        /// Workload ID
        workload_id: WorkloadId,
        /// Assigned tier
        tier: ComputeTier,
        /// Assigned node ID
        node_id: u32,
    },

    /// Request to release a workload
    ReleaseWorkload {
        /// Workload ID
        workload_id: WorkloadId,
    },

    /// Request to migrate a workload
    MigrateWorkload {
        /// Workload ID
        workload_id: WorkloadId,
        /// Target tier
        target_tier: ComputeTier,
    },

    // ========================================================================
    // Resource Management
    // ========================================================================

    /// Resource capacity update from a node
    ResourceUpdate {
        /// Node ID
        node_id: u32,
        /// Compute tier
        tier: ComputeTier,
        /// Available compute (FLOPS)
        available_compute: u64,
        /// Available memory (bytes)
        available_memory: u64,
    },

    /// Node health update
    NodeHealthUpdate {
        /// Node ID
        node_id: u32,
        /// Whether the node is available
        is_available: bool,
    },

    // ========================================================================
    // Inference Requests
    // ========================================================================

    /// Request inference on a model
    InferenceRequest {
        /// Request ID
        request_id: u64,
        /// Model name/ID
        model_id: String,
        /// Input tensor data
        input: TensorData,
        /// Deadline in milliseconds (0 = no deadline)
        deadline_ms: u32,
        /// Response channel
        response_tx: Option<tokio::sync::oneshot::Sender<SheResponse>>,
    },

    /// Batch inference request
    BatchInferenceRequest {
        /// Request ID
        request_id: u64,
        /// Model name/ID
        model_id: String,
        /// Batch of input tensors
        inputs: Vec<TensorData>,
        /// Deadline in milliseconds
        deadline_ms: u32,
        /// Response channel
        response_tx: Option<tokio::sync::oneshot::Sender<SheResponse>>,
    },

    // ========================================================================
    // Model Management
    // ========================================================================

    /// Load a model on a tier
    LoadModel {
        /// Model name/ID
        model_id: String,
        /// Path to model file
        model_path: std::path::PathBuf,
        /// Target tier
        tier: ComputeTier,
    },

    /// Unload a model
    UnloadModel {
        /// Model name/ID
        model_id: String,
        /// Tier to unload from
        tier: ComputeTier,
    },

    // ========================================================================
    // Status and Queries
    // ========================================================================

    /// Query tier status
    QueryTierStatus {
        /// Tier to query
        tier: ComputeTier,
        /// Response channel
        response_tx: tokio::sync::oneshot::Sender<SheResponse>,
    },

    /// Query workload status
    QueryWorkloadStatus {
        /// Workload ID
        workload_id: WorkloadId,
        /// Response channel
        response_tx: tokio::sync::oneshot::Sender<SheResponse>,
    },
}

/// Response types from SHE
#[derive(Debug)]
pub enum SheResponse {
    /// Workload placement result
    WorkloadPlaced {
        /// Workload ID
        workload_id: WorkloadId,
        /// Assigned tier
        tier: ComputeTier,
        /// Assigned node ID
        node_id: u32,
    },

    /// Workload placement failed
    PlacementFailed {
        /// Workload ID
        workload_id: WorkloadId,
        /// Error message
        error: String,
    },

    /// Inference result
    InferenceResult {
        /// Request ID
        request_id: u64,
        /// Output tensor
        output: TensorData,
        /// Latency in milliseconds
        latency_ms: u32,
        /// Tier that processed the request
        tier: ComputeTier,
    },

    /// Batch inference result
    BatchInferenceResult {
        /// Request ID
        request_id: u64,
        /// Output tensors
        outputs: Vec<TensorData>,
        /// Latency in milliseconds
        latency_ms: u32,
    },

    /// Inference failed
    InferenceFailed {
        /// Request ID
        request_id: u64,
        /// Error message
        error: String,
    },

    /// Tier status response
    TierStatus {
        /// Tier
        tier: ComputeTier,
        /// Total compute capacity (FLOPS)
        total_compute: u64,
        /// Available compute (FLOPS)
        available_compute: u64,
        /// Total memory (bytes)
        total_memory: u64,
        /// Available memory (bytes)
        available_memory: u64,
        /// Number of active workloads
        active_workloads: u32,
        /// Number of nodes
        node_count: usize,
    },

    /// Workload status response
    WorkloadStatus {
        /// Workload ID
        workload_id: WorkloadId,
        /// Current state
        state: crate::workload::WorkloadState,
        /// Assigned tier (if any)
        tier: Option<ComputeTier>,
        /// Assigned node (if any)
        node_id: Option<u32>,
    },

    /// Error response
    Error {
        /// Error message
        message: String,
    },

    /// Success acknowledgment
    Ok,
}

/// Simplified message for external API (serializable)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SheApiMessage {
    /// Place workload request
    PlaceWorkload {
        /// Requirements as JSON
        requirements: serde_json::Value,
    },
    /// Release workload
    ReleaseWorkload {
        /// Workload ID
        workload_id: u64,
    },
    /// Query status
    QueryStatus {
        /// Tier name
        tier: String,
    },
}

/// Simplified response for external API (serializable)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SheApiResponse {
    /// Placement result
    Placed {
        /// Workload ID
        workload_id: u64,
        /// Tier name
        tier: String,
        /// Node ID
        node_id: u32,
    },
    /// Status result
    Status {
        /// Tier name
        tier: String,
        /// Available compute in TFLOPS
        available_tflops: f64,
        /// Available memory in GB
        available_gb: f64,
        /// Active workloads
        active_workloads: u32,
    },
    /// Error
    Error {
        /// Error message
        message: String,
    },
    /// Success
    Ok,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workload::WorkloadRequirements;

    #[test]
    fn test_she_message_creation() {
        let msg = SheMessage::PlaceWorkload {
            workload_id: WorkloadId::new(1),
            requirements: WorkloadRequirements::inference(),
            response_tx: None,
        };

        match msg {
            SheMessage::PlaceWorkload { workload_id, .. } => {
                assert_eq!(workload_id.value(), 1);
            }
            _ => panic!("Unexpected message type"),
        }
    }

    #[test]
    fn test_she_response_creation() {
        let response = SheResponse::WorkloadPlaced {
            workload_id: WorkloadId::new(1),
            tier: ComputeTier::LocalEdge,
            node_id: 1,
        };

        match response {
            SheResponse::WorkloadPlaced { tier, .. } => {
                assert_eq!(tier, ComputeTier::LocalEdge);
            }
            _ => panic!("Unexpected response type"),
        }
    }

    #[test]
    fn test_api_message_serialization() {
        let msg = SheApiMessage::PlaceWorkload {
            requirements: serde_json::json!({
                "type": "inference",
                "latency_ms": 10
            }),
        };

        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("PlaceWorkload"));

        let parsed: SheApiMessage = serde_json::from_str(&json).unwrap();
        match parsed {
            SheApiMessage::PlaceWorkload { requirements } => {
                assert!(requirements.get("type").is_some());
            }
            _ => panic!("Unexpected message type"),
        }
    }
}
