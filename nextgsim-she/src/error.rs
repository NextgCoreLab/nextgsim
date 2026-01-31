//! Error types for Service Hosting Environment

use thiserror::Error;

use crate::tier::ComputeTier;
use crate::workload::WorkloadId;

/// SHE error types
#[derive(Error, Debug)]
pub enum SheError {
    /// Workload placement failed
    #[error("Failed to place workload {workload_id}: {reason}")]
    PlacementFailed {
        /// Workload ID
        workload_id: WorkloadId,
        /// Failure reason
        reason: String,
    },

    /// No suitable tier found for workload
    #[error("No suitable tier for workload {workload_id}: latency={latency_ms}ms, required compute={required_flops} FLOPS")]
    NoSuitableTier {
        /// Workload ID
        workload_id: WorkloadId,
        /// Required latency constraint
        latency_ms: u32,
        /// Required compute capacity
        required_flops: u64,
    },

    /// Insufficient resources on tier
    #[error("Insufficient resources on {tier}: available={available}, required={required}")]
    InsufficientResources {
        /// Compute tier
        tier: ComputeTier,
        /// Available resources description
        available: String,
        /// Required resources description
        required: String,
    },

    /// Workload not found
    #[error("Workload not found: {workload_id}")]
    WorkloadNotFound {
        /// Workload ID
        workload_id: WorkloadId,
    },

    /// Migration failed
    #[error("Migration of workload {workload_id} from {source_tier} to {target_tier} failed: {reason}")]
    MigrationFailed {
        /// Workload ID
        workload_id: WorkloadId,
        /// Source tier
        source_tier: ComputeTier,
        /// Target tier
        target_tier: ComputeTier,
        /// Failure reason
        reason: String,
    },

    /// Tier not available
    #[error("Tier {tier} is not available: {reason}")]
    TierUnavailable {
        /// Compute tier
        tier: ComputeTier,
        /// Reason
        reason: String,
    },

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// Internal error
    #[error("Internal error: {0}")]
    Internal(String),

    /// AI inference error
    #[error("AI inference error: {0}")]
    Inference(#[from] nextgsim_ai::AiError),
}

/// Result type for SHE operations
pub type SheResult<T> = Result<T, SheError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = SheError::PlacementFailed {
            workload_id: WorkloadId::new(1),
            reason: "No available nodes".to_string(),
        };
        assert!(err.to_string().contains("Failed to place workload"));
    }

    #[test]
    fn test_no_suitable_tier() {
        let err = SheError::NoSuitableTier {
            workload_id: WorkloadId::new(1),
            latency_ms: 5,
            required_flops: 1_000_000_000_000,
        };
        assert!(err.to_string().contains("No suitable tier"));
    }
}
