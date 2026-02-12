//! Integration with Service Hosting Environment for FL Workloads
//!
//! Maps FL workload types (training, aggregation, inference) to SHE compute
//! tiers. Training goes to Core Cloud, aggregation to Regional Edge, and
//! inference to Local Edge.

use serde::{Deserialize, Serialize};

/// SHE compute tier for FL workload placement
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlComputeTier {
    /// Local edge: low-latency inference
    LocalEdge,
    /// Regional edge: aggregation + fine-tuning
    RegionalEdge,
    /// Core cloud: full training + global aggregation
    CoreCloud,
}

/// FL workload type for SHE placement
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlWorkloadType {
    /// Local model training on a participant
    LocalTraining,
    /// Edge/regional aggregation
    Aggregation,
    /// Global aggregation (cloud-level)
    GlobalAggregation,
    /// Model inference (serving)
    Inference,
    /// Privacy mechanism (DP noise, `SecAgg` masking)
    PrivacyProcessing,
}

/// Resource requirements for an FL workload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlResourceRequirements {
    /// Workload type
    pub workload_type: FlWorkloadType,
    /// Required compute (FLOPS)
    pub compute_flops: u64,
    /// Required memory (bytes)
    pub memory_bytes: u64,
    /// Maximum acceptable latency (ms)
    pub max_latency_ms: u32,
    /// Model size in parameters
    pub model_params: u64,
    /// Number of participants (for aggregation workloads)
    pub num_participants: u32,
}

impl FlResourceRequirements {
    /// Returns the recommended SHE compute tier for this workload
    pub fn recommended_tier(&self) -> FlComputeTier {
        match self.workload_type {
            FlWorkloadType::Inference => FlComputeTier::LocalEdge,
            FlWorkloadType::LocalTraining => {
                if self.max_latency_ms < 20 {
                    FlComputeTier::LocalEdge
                } else {
                    FlComputeTier::RegionalEdge
                }
            }
            FlWorkloadType::Aggregation | FlWorkloadType::PrivacyProcessing => {
                FlComputeTier::RegionalEdge
            }
            FlWorkloadType::GlobalAggregation => FlComputeTier::CoreCloud,
        }
    }

    /// Estimates compute FLOPS needed for aggregation of N model updates
    pub fn estimate_aggregation_flops(model_params: u64, num_participants: u32) -> u64 {
        // Weighted average: ~2 FLOPS per parameter per participant
        model_params * num_participants as u64 * 2
    }

    /// Estimates memory for aggregation (need to hold all updates + result)
    pub fn estimate_aggregation_memory(model_params: u64, num_participants: u32) -> u64 {
        // Each update + global model: (N+1) * params * 4 bytes (f32)
        (num_participants as u64 + 1) * model_params * 4
    }
}

/// SHE placement decision for an FL workload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlPlacementDecision {
    /// Selected compute tier
    pub tier: FlComputeTier,
    /// Workload type
    pub workload_type: FlWorkloadType,
    /// Estimated execution time (ms)
    pub estimated_time_ms: u64,
    /// Whether an accelerator (GPU/NPU) is recommended
    pub needs_accelerator: bool,
}

/// SHE-FL placement planner
pub struct FlPlacementPlanner {
    /// Latency budget per tier (ms): [`local_edge`, `regional_edge`, `core_cloud`]
    tier_latency_ms: [u32; 3],
    /// Compute capacity per tier (FLOPS): [`local_edge`, `regional_edge`, `core_cloud`]
    tier_compute_flops: [u64; 3],
}

impl FlPlacementPlanner {
    /// Creates a new placement planner
    pub fn new(
        tier_latency_ms: [u32; 3],
        tier_compute_flops: [u64; 3],
    ) -> Self {
        Self {
            tier_latency_ms,
            tier_compute_flops,
        }
    }

    /// Plans placement for an FL workload
    pub fn plan(&self, req: &FlResourceRequirements) -> FlPlacementDecision {
        let recommended = req.recommended_tier();

        // Check if recommended tier meets latency requirements
        let tier_idx = match recommended {
            FlComputeTier::LocalEdge => 0,
            FlComputeTier::RegionalEdge => 1,
            FlComputeTier::CoreCloud => 2,
        };

        let tier = if self.tier_latency_ms[tier_idx] <= req.max_latency_ms {
            recommended
        } else {
            // Fall back to a lower-latency tier
            if self.tier_latency_ms[0] <= req.max_latency_ms {
                FlComputeTier::LocalEdge
            } else {
                recommended // No choice, use recommended
            }
        };

        let final_idx = match tier {
            FlComputeTier::LocalEdge => 0,
            FlComputeTier::RegionalEdge => 1,
            FlComputeTier::CoreCloud => 2,
        };

        let estimated_time_ms = if self.tier_compute_flops[final_idx] > 0 {
            (req.compute_flops as f64 / self.tier_compute_flops[final_idx] as f64 * 1000.0) as u64
        } else {
            0
        };

        // Large models (>10M params) benefit from accelerators
        let needs_accelerator = req.model_params > 10_000_000;

        FlPlacementDecision {
            tier,
            workload_type: req.workload_type,
            estimated_time_ms,
            needs_accelerator,
        }
    }
}

impl Default for FlPlacementPlanner {
    fn default() -> Self {
        Self::new(
            [5, 15, 50],                               // latency ms
            [1_000_000_000, 10_000_000_000, 100_000_000_000], // FLOPS
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recommended_tier_inference() {
        let req = FlResourceRequirements {
            workload_type: FlWorkloadType::Inference,
            compute_flops: 1_000_000,
            memory_bytes: 100_000_000,
            max_latency_ms: 10,
            model_params: 1_000_000,
            num_participants: 0,
        };
        assert_eq!(req.recommended_tier(), FlComputeTier::LocalEdge);
    }

    #[test]
    fn test_recommended_tier_global_aggregation() {
        let req = FlResourceRequirements {
            workload_type: FlWorkloadType::GlobalAggregation,
            compute_flops: 100_000_000,
            memory_bytes: 4_000_000_000,
            max_latency_ms: 1000,
            model_params: 100_000_000,
            num_participants: 100,
        };
        assert_eq!(req.recommended_tier(), FlComputeTier::CoreCloud);
    }

    #[test]
    fn test_estimate_aggregation() {
        let flops = FlResourceRequirements::estimate_aggregation_flops(1_000_000, 10);
        assert_eq!(flops, 20_000_000);

        let mem = FlResourceRequirements::estimate_aggregation_memory(1_000_000, 10);
        assert_eq!(mem, 44_000_000); // 11 * 1M * 4
    }

    #[test]
    fn test_placement_planner() {
        let planner = FlPlacementPlanner::default();

        let req = FlResourceRequirements {
            workload_type: FlWorkloadType::Inference,
            compute_flops: 1_000_000,
            memory_bytes: 100_000_000,
            max_latency_ms: 10,
            model_params: 1_000_000,
            num_participants: 0,
        };

        let decision = planner.plan(&req);
        assert_eq!(decision.tier, FlComputeTier::LocalEdge);
        assert!(!decision.needs_accelerator);
    }

    #[test]
    fn test_placement_planner_large_model() {
        let planner = FlPlacementPlanner::default();

        let req = FlResourceRequirements {
            workload_type: FlWorkloadType::GlobalAggregation,
            compute_flops: 1_000_000_000,
            memory_bytes: 4_000_000_000,
            max_latency_ms: 100,
            model_params: 100_000_000,
            num_participants: 50,
        };

        let decision = planner.plan(&req);
        assert!(decision.needs_accelerator);
    }
}
