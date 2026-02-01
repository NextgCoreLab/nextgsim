//! Workload definitions for Service Hosting Environment
//!
//! Defines workload types, requirements, and lifecycle states.

use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::tier::{ComputeCapability, ComputeTier};

/// Unique workload identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorkloadId(u64);

impl WorkloadId {
    /// Creates a new workload ID
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns the inner ID value
    pub fn value(&self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for WorkloadId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WL-{}", self.0)
    }
}

/// Workload type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkloadType {
    /// ML inference workload
    Inference,
    /// Model fine-tuning workload
    FineTuning,
    /// Model training workload
    Training,
    /// Data processing workload
    DataProcessing,
    /// Analytics workload
    Analytics,
}

impl WorkloadType {
    /// Returns the required capability for this workload type
    pub fn required_capability(&self) -> ComputeCapability {
        match self {
            WorkloadType::Inference | WorkloadType::DataProcessing | WorkloadType::Analytics => {
                ComputeCapability::Inference
            }
            WorkloadType::FineTuning => ComputeCapability::FineTuning,
            WorkloadType::Training => ComputeCapability::Training,
        }
    }
}

impl std::fmt::Display for WorkloadType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WorkloadType::Inference => write!(f, "Inference"),
            WorkloadType::FineTuning => write!(f, "FineTuning"),
            WorkloadType::Training => write!(f, "Training"),
            WorkloadType::DataProcessing => write!(f, "DataProcessing"),
            WorkloadType::Analytics => write!(f, "Analytics"),
        }
    }
}

/// Workload state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkloadState {
    /// Workload is pending placement
    Pending,
    /// Workload is being scheduled
    Scheduling,
    /// Workload is placed and running
    Running,
    /// Workload is being migrated
    Migrating,
    /// Workload completed successfully
    Completed,
    /// Workload failed
    Failed,
    /// Workload was cancelled
    Cancelled,
}

impl std::fmt::Display for WorkloadState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WorkloadState::Pending => write!(f, "Pending"),
            WorkloadState::Scheduling => write!(f, "Scheduling"),
            WorkloadState::Running => write!(f, "Running"),
            WorkloadState::Migrating => write!(f, "Migrating"),
            WorkloadState::Completed => write!(f, "Completed"),
            WorkloadState::Failed => write!(f, "Failed"),
            WorkloadState::Cancelled => write!(f, "Cancelled"),
        }
    }
}

/// Workload requirements specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadRequirements {
    /// Workload type
    pub workload_type: WorkloadType,
    /// Maximum acceptable latency in milliseconds
    pub latency_constraint_ms: Option<u32>,
    /// Required compute capacity in FLOPS
    pub compute_flops: u64,
    /// Required memory in bytes
    pub memory_bytes: u64,
    /// Required capability
    pub capability: ComputeCapability,
    /// Preferred tier (if any)
    pub preferred_tier: Option<ComputeTier>,
    /// Associated UE ID (if any)
    pub ue_id: Option<i32>,
    /// Associated cell ID (if any)
    pub cell_id: Option<i32>,
    /// Model name (for inference workloads)
    pub model_name: Option<String>,
    /// Priority (higher = more important)
    pub priority: u8,
}

impl WorkloadRequirements {
    /// Creates new requirements for an inference workload
    pub fn inference() -> Self {
        Self {
            workload_type: WorkloadType::Inference,
            latency_constraint_ms: Some(10), // Local edge latency
            compute_flops: 1_000_000_000, // 1 GFLOPS
            memory_bytes: 512 * 1024 * 1024, // 512 MB
            capability: ComputeCapability::Inference,
            preferred_tier: Some(ComputeTier::LocalEdge),
            ue_id: None,
            cell_id: None,
            model_name: None,
            priority: 5,
        }
    }

    /// Creates new requirements for a fine-tuning workload
    pub fn fine_tuning() -> Self {
        Self {
            workload_type: WorkloadType::FineTuning,
            latency_constraint_ms: Some(20), // Regional edge latency
            compute_flops: 100_000_000_000, // 100 GFLOPS
            memory_bytes: 8 * 1024 * 1024 * 1024, // 8 GB
            capability: ComputeCapability::FineTuning,
            preferred_tier: Some(ComputeTier::RegionalEdge),
            ue_id: None,
            cell_id: None,
            model_name: None,
            priority: 3,
        }
    }

    /// Creates new requirements for a training workload
    pub fn training() -> Self {
        Self {
            workload_type: WorkloadType::Training,
            latency_constraint_ms: None, // No constraint
            compute_flops: 1_000_000_000_000, // 1 TFLOPS
            memory_bytes: 64 * 1024 * 1024 * 1024, // 64 GB
            capability: ComputeCapability::Training,
            preferred_tier: Some(ComputeTier::CoreCloud),
            ue_id: None,
            cell_id: None,
            model_name: None,
            priority: 1,
        }
    }

    /// Sets the latency constraint
    pub fn with_latency_constraint_ms(mut self, ms: u32) -> Self {
        self.latency_constraint_ms = Some(ms);
        self
    }

    /// Sets the compute requirement in FLOPS
    pub fn with_compute_flops(mut self, flops: u64) -> Self {
        self.compute_flops = flops;
        self
    }

    /// Sets the compute requirement in GFLOPS
    pub fn with_compute_gflops(mut self, gflops: u64) -> Self {
        self.compute_flops = gflops * 1_000_000_000;
        self
    }

    /// Sets the memory requirement in bytes
    pub fn with_memory_bytes(mut self, bytes: u64) -> Self {
        self.memory_bytes = bytes;
        self
    }

    /// Sets the memory requirement in MB
    pub fn with_memory_mb(mut self, mb: u64) -> Self {
        self.memory_bytes = mb * 1024 * 1024;
        self
    }

    /// Sets the memory requirement in GB
    pub fn with_memory_gb(mut self, gb: u64) -> Self {
        self.memory_bytes = gb * 1024 * 1024 * 1024;
        self
    }

    /// Sets the associated UE ID
    pub fn with_ue_id(mut self, ue_id: i32) -> Self {
        self.ue_id = Some(ue_id);
        self
    }

    /// Sets the associated cell ID
    pub fn with_cell_id(mut self, cell_id: i32) -> Self {
        self.cell_id = Some(cell_id);
        self
    }

    /// Sets the model name
    pub fn with_model_name(mut self, name: impl Into<String>) -> Self {
        self.model_name = Some(name.into());
        self
    }

    /// Sets the priority
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// Returns the minimum tier that can satisfy the latency constraint
    pub fn minimum_tier(&self) -> ComputeTier {
        match self.latency_constraint_ms {
            Some(ms) if ms <= 10 => ComputeTier::LocalEdge,
            Some(ms) if ms <= 20 => ComputeTier::RegionalEdge,
            _ => ComputeTier::CoreCloud,
        }
    }
}

/// A workload instance
#[derive(Debug, Clone)]
pub struct Workload {
    /// Unique identifier
    pub id: WorkloadId,
    /// Workload requirements
    pub requirements: WorkloadRequirements,
    /// Current state
    pub state: WorkloadState,
    /// Assigned tier (if placed)
    pub assigned_tier: Option<ComputeTier>,
    /// Assigned node ID (if placed)
    pub assigned_node_id: Option<u32>,
    /// Creation time
    pub created_at: Instant,
    /// Start time (if running)
    pub started_at: Option<Instant>,
    /// Completion time (if completed)
    pub completed_at: Option<Instant>,
    /// Error message (if failed)
    pub error: Option<String>,
}

impl Workload {
    /// Creates a new workload with the given ID and requirements
    pub fn new(id: WorkloadId, requirements: WorkloadRequirements) -> Self {
        Self {
            id,
            requirements,
            state: WorkloadState::Pending,
            assigned_tier: None,
            assigned_node_id: None,
            created_at: Instant::now(),
            started_at: None,
            completed_at: None,
            error: None,
        }
    }

    /// Marks the workload as scheduled
    pub fn mark_scheduling(&mut self) {
        self.state = WorkloadState::Scheduling;
    }

    /// Marks the workload as running
    pub fn mark_running(&mut self, tier: ComputeTier, node_id: u32) {
        self.state = WorkloadState::Running;
        self.assigned_tier = Some(tier);
        self.assigned_node_id = Some(node_id);
        self.started_at = Some(Instant::now());
    }

    /// Marks the workload as migrating
    pub fn mark_migrating(&mut self) {
        self.state = WorkloadState::Migrating;
    }

    /// Marks the workload as completed
    pub fn mark_completed(&mut self) {
        self.state = WorkloadState::Completed;
        self.completed_at = Some(Instant::now());
    }

    /// Marks the workload as failed
    pub fn mark_failed(&mut self, error: impl Into<String>) {
        self.state = WorkloadState::Failed;
        self.error = Some(error.into());
        self.completed_at = Some(Instant::now());
    }

    /// Marks the workload as cancelled
    pub fn mark_cancelled(&mut self) {
        self.state = WorkloadState::Cancelled;
        self.completed_at = Some(Instant::now());
    }

    /// Returns the duration the workload has been running
    pub fn running_duration(&self) -> Option<std::time::Duration> {
        self.started_at.map(|start| {
            self.completed_at
                .unwrap_or_else(Instant::now)
                .duration_since(start)
        })
    }

    /// Returns the queue time (time from creation to start)
    pub fn queue_time(&self) -> Option<std::time::Duration> {
        self.started_at.map(|start| start.duration_since(self.created_at))
    }

    /// Returns true if the workload is terminal (completed, failed, or cancelled)
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.state,
            WorkloadState::Completed | WorkloadState::Failed | WorkloadState::Cancelled
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workload_id() {
        let id = WorkloadId::new(42);
        assert_eq!(id.value(), 42);
        assert_eq!(format!("{}", id), "WL-42");
    }

    #[test]
    fn test_workload_requirements_inference() {
        let req = WorkloadRequirements::inference()
            .with_ue_id(1)
            .with_model_name("trajectory_model");

        assert_eq!(req.workload_type, WorkloadType::Inference);
        assert_eq!(req.latency_constraint_ms, Some(10));
        assert_eq!(req.ue_id, Some(1));
        assert_eq!(req.model_name, Some("trajectory_model".to_string()));
    }

    #[test]
    fn test_workload_requirements_minimum_tier() {
        let inference = WorkloadRequirements::inference();
        assert_eq!(inference.minimum_tier(), ComputeTier::LocalEdge);

        let fine_tuning = WorkloadRequirements::fine_tuning();
        assert_eq!(fine_tuning.minimum_tier(), ComputeTier::RegionalEdge);

        let training = WorkloadRequirements::training();
        assert_eq!(training.minimum_tier(), ComputeTier::CoreCloud);
    }

    #[test]
    fn test_workload_lifecycle() {
        let req = WorkloadRequirements::inference();
        let mut workload = Workload::new(WorkloadId::new(1), req);

        assert_eq!(workload.state, WorkloadState::Pending);
        assert!(!workload.is_terminal());

        workload.mark_scheduling();
        assert_eq!(workload.state, WorkloadState::Scheduling);

        workload.mark_running(ComputeTier::LocalEdge, 1);
        assert_eq!(workload.state, WorkloadState::Running);
        assert_eq!(workload.assigned_tier, Some(ComputeTier::LocalEdge));
        assert_eq!(workload.assigned_node_id, Some(1));

        workload.mark_completed();
        assert_eq!(workload.state, WorkloadState::Completed);
        assert!(workload.is_terminal());
        assert!(workload.running_duration().is_some());
    }

    #[test]
    fn test_workload_failure() {
        let req = WorkloadRequirements::inference();
        let mut workload = Workload::new(WorkloadId::new(1), req);

        workload.mark_failed("Resource exhausted");

        assert_eq!(workload.state, WorkloadState::Failed);
        assert_eq!(workload.error, Some("Resource exhausted".to_string()));
        assert!(workload.is_terminal());
    }
}
