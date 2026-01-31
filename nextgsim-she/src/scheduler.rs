//! Workload scheduler for Service Hosting Environment
//!
//! Implements resource-aware workload placement across the three-tier compute hierarchy.

use std::collections::HashMap;

use tracing::{debug, info};

use crate::error::{SheError, SheResult};
use crate::tier::{ComputeTier, TierManager};
use crate::workload::{Workload, WorkloadId, WorkloadRequirements, WorkloadState};

/// Placement decision for a workload
#[derive(Debug, Clone)]
pub struct PlacementDecision {
    /// Workload ID
    pub workload_id: WorkloadId,
    /// Selected tier
    pub tier: ComputeTier,
    /// Selected node ID
    pub node_id: u32,
    /// Reason for this placement
    pub reason: PlacementReason,
}

/// Reason for placement decision
#[derive(Debug, Clone)]
pub enum PlacementReason {
    /// Latency constraint satisfied
    LatencyConstraint,
    /// Capability requirement satisfied
    CapabilityRequired,
    /// Preferred tier available
    PreferredTier,
    /// Best available resource
    BestAvailable,
    /// Fallback placement
    Fallback,
}

impl std::fmt::Display for PlacementReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlacementReason::LatencyConstraint => write!(f, "LatencyConstraint"),
            PlacementReason::CapabilityRequired => write!(f, "CapabilityRequired"),
            PlacementReason::PreferredTier => write!(f, "PreferredTier"),
            PlacementReason::BestAvailable => write!(f, "BestAvailable"),
            PlacementReason::Fallback => write!(f, "Fallback"),
        }
    }
}

/// Scheduling policy for workload placement
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingPolicy {
    /// Place on the tier closest to the edge that satisfies requirements
    ClosestToEdge,
    /// Place on the tier with the most available resources
    MostAvailable,
    /// Place on the tier with lowest utilization
    LowestUtilization,
    /// Respect preferred tier if possible
    PreferredFirst,
}

impl Default for SchedulingPolicy {
    fn default() -> Self {
        SchedulingPolicy::ClosestToEdge
    }
}

/// Workload scheduler
#[derive(Debug)]
pub struct WorkloadScheduler {
    /// Tier manager
    tier_manager: TierManager,
    /// Active workloads
    workloads: HashMap<WorkloadId, Workload>,
    /// Next workload ID
    next_workload_id: u64,
    /// Scheduling policy
    policy: SchedulingPolicy,
}

impl WorkloadScheduler {
    /// Creates a new scheduler with the given tier manager
    pub fn new(tier_manager: TierManager) -> Self {
        Self {
            tier_manager,
            workloads: HashMap::new(),
            next_workload_id: 1,
            policy: SchedulingPolicy::default(),
        }
    }

    /// Sets the scheduling policy
    pub fn with_policy(mut self, policy: SchedulingPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Returns a reference to the tier manager
    pub fn tier_manager(&self) -> &TierManager {
        &self.tier_manager
    }

    /// Returns a mutable reference to the tier manager
    pub fn tier_manager_mut(&mut self) -> &mut TierManager {
        &mut self.tier_manager
    }

    /// Allocates a new workload ID
    pub fn allocate_id(&mut self) -> WorkloadId {
        let id = WorkloadId::new(self.next_workload_id);
        self.next_workload_id += 1;
        id
    }

    /// Submits a workload for scheduling
    pub fn submit(&mut self, requirements: WorkloadRequirements) -> SheResult<WorkloadId> {
        let id = self.allocate_id();
        let workload = Workload::new(id, requirements);

        info!("Submitted workload {}: type={}", id, workload.requirements.workload_type);

        self.workloads.insert(id, workload);
        Ok(id)
    }

    /// Places a workload on a suitable tier
    pub fn place(&mut self, workload_id: WorkloadId) -> SheResult<PlacementDecision> {
        // First, extract requirements and mark as scheduling
        let requirements = {
            let workload = self.workloads.get_mut(&workload_id).ok_or(SheError::WorkloadNotFound {
                workload_id,
            })?;
            workload.mark_scheduling();
            workload.requirements.clone()
        };

        // Find placement (now requirements is owned, no borrow conflict)
        let decision = self.find_placement(workload_id, &requirements)?;

        // Allocate resources on the selected node
        if let Some(node) = self.tier_manager.get_node_mut(decision.node_id) {
            node.allocate(requirements.compute_flops, requirements.memory_bytes);
        }

        // Update workload state
        if let Some(workload) = self.workloads.get_mut(&workload_id) {
            workload.mark_running(decision.tier, decision.node_id);
        }

        info!(
            "Placed workload {} on {} (node {}): {}",
            workload_id, decision.tier, decision.node_id, decision.reason
        );

        Ok(decision)
    }

    /// Finds the best placement for a workload
    fn find_placement(
        &self,
        workload_id: WorkloadId,
        requirements: &WorkloadRequirements,
    ) -> SheResult<PlacementDecision> {
        let capability = requirements.capability;
        let compute = requirements.compute_flops;
        let memory = requirements.memory_bytes;

        // Determine candidate tiers based on constraints
        let candidate_tiers = self.get_candidate_tiers(requirements);

        if candidate_tiers.is_empty() {
            return Err(SheError::NoSuitableTier {
                workload_id,
                latency_ms: requirements.latency_constraint_ms.unwrap_or(u32::MAX),
                required_flops: compute,
            });
        }

        // Try each tier in order
        for tier in candidate_tiers {
            if let Some(node) = self.tier_manager.find_best_node(tier, capability, compute, memory) {
                let reason = self.determine_reason(tier, requirements);
                return Ok(PlacementDecision {
                    workload_id,
                    tier,
                    node_id: node.id,
                    reason,
                });
            }
        }

        Err(SheError::NoSuitableTier {
            workload_id,
            latency_ms: requirements.latency_constraint_ms.unwrap_or(u32::MAX),
            required_flops: compute,
        })
    }

    /// Gets candidate tiers for a workload based on requirements and policy
    fn get_candidate_tiers(&self, requirements: &WorkloadRequirements) -> Vec<ComputeTier> {
        let minimum_tier = requirements.minimum_tier();
        let capability_tier = requirements.capability.minimum_tier();

        // The actual minimum is the max of latency and capability requirements
        let effective_minimum = if minimum_tier.priority() >= capability_tier.priority() {
            minimum_tier
        } else {
            capability_tier
        };

        match self.policy {
            SchedulingPolicy::ClosestToEdge => {
                // Order by proximity (closest first) starting from effective minimum
                ComputeTier::all_ordered()
                    .iter()
                    .filter(|t| t.priority() >= effective_minimum.priority())
                    .copied()
                    .collect()
            }
            SchedulingPolicy::PreferredFirst => {
                // Try preferred tier first, then others
                let mut tiers = Vec::new();
                if let Some(preferred) = requirements.preferred_tier {
                    if preferred.priority() >= effective_minimum.priority() {
                        tiers.push(preferred);
                    }
                }
                for tier in ComputeTier::all_ordered() {
                    if tier.priority() >= effective_minimum.priority()
                        && !tiers.contains(tier)
                    {
                        tiers.push(*tier);
                    }
                }
                tiers
            }
            SchedulingPolicy::MostAvailable | SchedulingPolicy::LowestUtilization => {
                // Order by available resources (most first)
                let mut tiers: Vec<_> = ComputeTier::all_ordered()
                    .iter()
                    .filter(|t| t.priority() >= effective_minimum.priority())
                    .copied()
                    .collect();

                tiers.sort_by_key(|tier| {
                    let capacity = self.tier_manager.tier_capacity(*tier);
                    let usage = self.tier_manager.tier_usage(*tier);
                    let available = capacity.compute_flops.saturating_sub(usage.compute_flops);
                    std::cmp::Reverse(available) // Most available first
                });

                tiers
            }
        }
    }

    /// Determines the reason for a placement decision
    fn determine_reason(&self, tier: ComputeTier, requirements: &WorkloadRequirements) -> PlacementReason {
        if requirements.preferred_tier == Some(tier) {
            PlacementReason::PreferredTier
        } else if let Some(latency) = requirements.latency_constraint_ms {
            if tier.max_latency_ms() <= latency {
                PlacementReason::LatencyConstraint
            } else {
                PlacementReason::Fallback
            }
        } else if tier == requirements.capability.minimum_tier() {
            PlacementReason::CapabilityRequired
        } else {
            PlacementReason::BestAvailable
        }
    }

    /// Releases a workload and frees its resources
    pub fn release(&mut self, workload_id: WorkloadId) -> SheResult<()> {
        let workload = self.workloads.get_mut(&workload_id).ok_or(SheError::WorkloadNotFound {
            workload_id,
        })?;

        // Release resources if the workload was running
        if let (Some(node_id), WorkloadState::Running | WorkloadState::Migrating) =
            (workload.assigned_node_id, workload.state)
        {
            if let Some(node) = self.tier_manager.get_node_mut(node_id) {
                node.release(
                    workload.requirements.compute_flops,
                    workload.requirements.memory_bytes,
                );
            }
        }

        workload.mark_completed();

        debug!("Released workload {}", workload_id);
        Ok(())
    }

    /// Cancels a workload
    pub fn cancel(&mut self, workload_id: WorkloadId) -> SheResult<()> {
        let workload = self.workloads.get_mut(&workload_id).ok_or(SheError::WorkloadNotFound {
            workload_id,
        })?;

        // Release resources if allocated
        if let Some(node_id) = workload.assigned_node_id {
            if let Some(node) = self.tier_manager.get_node_mut(node_id) {
                node.release(
                    workload.requirements.compute_flops,
                    workload.requirements.memory_bytes,
                );
            }
        }

        workload.mark_cancelled();

        debug!("Cancelled workload {}", workload_id);
        Ok(())
    }

    /// Migrates a workload to a different tier
    pub fn migrate(&mut self, workload_id: WorkloadId, target_tier: ComputeTier) -> SheResult<PlacementDecision> {
        // Extract necessary info and mark as migrating
        let (source_tier, source_node_id, requirements) = {
            let workload = self.workloads.get_mut(&workload_id).ok_or(SheError::WorkloadNotFound {
                workload_id,
            })?;

            let source_tier = workload.assigned_tier.ok_or(SheError::MigrationFailed {
                workload_id,
                source_tier: ComputeTier::LocalEdge, // Placeholder
                target_tier,
                reason: "Workload not currently placed".to_string(),
            })?;

            let source_node_id = workload.assigned_node_id.expect("Node ID should be set");

            workload.mark_migrating();

            (source_tier, source_node_id, workload.requirements.clone())
        };

        // Check if target tier can accommodate
        let capability = requirements.capability;

        let target_node = self
            .tier_manager
            .find_best_node(target_tier, capability, requirements.compute_flops, requirements.memory_bytes)
            .ok_or(SheError::InsufficientResources {
                tier: target_tier,
                available: "Insufficient".to_string(),
                required: format!(
                    "{} FLOPS, {} bytes",
                    requirements.compute_flops, requirements.memory_bytes
                ),
            })?;

        let target_node_id = target_node.id;

        // Release from source
        if let Some(source_node) = self.tier_manager.get_node_mut(source_node_id) {
            source_node.release(requirements.compute_flops, requirements.memory_bytes);
        }

        // Allocate on target
        if let Some(target_node) = self.tier_manager.get_node_mut(target_node_id) {
            target_node.allocate(requirements.compute_flops, requirements.memory_bytes);
        }

        // Update workload
        if let Some(workload) = self.workloads.get_mut(&workload_id) {
            workload.mark_running(target_tier, target_node_id);
        }

        info!(
            "Migrated workload {} from {} to {} (node {})",
            workload_id, source_tier, target_tier, target_node_id
        );

        Ok(PlacementDecision {
            workload_id,
            tier: target_tier,
            node_id: target_node_id,
            reason: PlacementReason::BestAvailable,
        })
    }

    /// Gets a workload by ID
    pub fn get_workload(&self, workload_id: WorkloadId) -> Option<&Workload> {
        self.workloads.get(&workload_id)
    }

    /// Gets all workloads
    pub fn workloads(&self) -> impl Iterator<Item = &Workload> {
        self.workloads.values()
    }

    /// Gets workloads by state
    pub fn workloads_by_state(&self, state: WorkloadState) -> impl Iterator<Item = &Workload> {
        self.workloads.values().filter(move |w| w.state == state)
    }

    /// Gets the number of active workloads
    pub fn active_workload_count(&self) -> usize {
        self.workloads
            .values()
            .filter(|w| !w.is_terminal())
            .count()
    }

    /// Cleans up terminal workloads
    pub fn cleanup_terminal(&mut self) {
        self.workloads.retain(|_, w| !w.is_terminal());
    }
}

impl Default for WorkloadScheduler {
    fn default() -> Self {
        Self::new(TierManager::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::ResourceCapacity;

    fn setup_scheduler() -> WorkloadScheduler {
        let mut tier_manager = TierManager::new();

        // Add nodes to each tier
        tier_manager.add_node(ComputeNode::new(
            1,
            "edge-1",
            ComputeTier::LocalEdge,
            ResourceCapacity::with_tflops(1).with_memory_gb(8),
        ));
        tier_manager.add_node(ComputeNode::new(
            2,
            "regional-1",
            ComputeTier::RegionalEdge,
            ResourceCapacity::with_tflops(10).with_memory_gb(64),
        ));
        tier_manager.add_node(ComputeNode::new(
            3,
            "cloud-1",
            ComputeTier::CoreCloud,
            ResourceCapacity::with_tflops(100).with_memory_gb(512),
        ));

        WorkloadScheduler::new(tier_manager)
    }

    #[test]
    fn test_submit_and_place_inference() {
        let mut scheduler = setup_scheduler();

        let requirements = WorkloadRequirements::inference();
        let id = scheduler.submit(requirements).unwrap();

        let decision = scheduler.place(id).unwrap();

        assert_eq!(decision.tier, ComputeTier::LocalEdge);
        assert_eq!(decision.node_id, 1);
    }

    #[test]
    fn test_submit_and_place_training() {
        let mut scheduler = setup_scheduler();

        let requirements = WorkloadRequirements::training();
        let id = scheduler.submit(requirements).unwrap();

        let decision = scheduler.place(id).unwrap();

        assert_eq!(decision.tier, ComputeTier::CoreCloud);
        assert_eq!(decision.node_id, 3);
    }

    #[test]
    fn test_release_workload() {
        let mut scheduler = setup_scheduler();

        let requirements = WorkloadRequirements::inference();
        let id = scheduler.submit(requirements).unwrap();
        scheduler.place(id).unwrap();

        // Check node has allocated resources
        let node = scheduler.tier_manager().get_node(1).unwrap();
        assert!(node.usage.compute_flops > 0);

        // Release
        scheduler.release(id).unwrap();

        // Check resources freed
        let node = scheduler.tier_manager().get_node(1).unwrap();
        assert_eq!(node.usage.compute_flops, 0);
    }

    #[test]
    fn test_migrate_workload() {
        let mut scheduler = setup_scheduler();

        // Use 10ms latency to start at LocalEdge (<=10ms = LocalEdge minimum)
        let requirements = WorkloadRequirements::inference()
            .with_latency_constraint_ms(10);

        let id = scheduler.submit(requirements).unwrap();
        let initial = scheduler.place(id).unwrap();

        assert_eq!(initial.tier, ComputeTier::LocalEdge);

        // Migrate to regional edge
        let migrated = scheduler.migrate(id, ComputeTier::RegionalEdge).unwrap();

        assert_eq!(migrated.tier, ComputeTier::RegionalEdge);

        // Check old node freed
        let old_node = scheduler.tier_manager().get_node(1).unwrap();
        assert_eq!(old_node.usage.active_workloads, 0);

        // Check new node allocated
        let new_node = scheduler.tier_manager().get_node(2).unwrap();
        assert_eq!(new_node.usage.active_workloads, 1);
    }

    #[test]
    fn test_no_suitable_tier() {
        let mut scheduler = setup_scheduler();

        // Request more compute than any tier has
        let requirements = WorkloadRequirements::inference()
            .with_compute_flops(1_000_000_000_000_000); // 1000 TFLOPS

        let id = scheduler.submit(requirements).unwrap();
        let result = scheduler.place(id);

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), SheError::NoSuitableTier { .. }));
    }

    #[test]
    fn test_scheduling_policy_preferred_first() {
        let mut scheduler = setup_scheduler().with_policy(SchedulingPolicy::PreferredFirst);

        // Use 10ms latency to allow LocalEdge tier (<=10ms = LocalEdge minimum)
        let requirements = WorkloadRequirements::inference()
            .with_latency_constraint_ms(10);

        // Without preference, should go to local edge (closest to edge)
        let id1 = scheduler.submit(requirements.clone()).unwrap();
        let decision1 = scheduler.place(id1).unwrap();
        assert_eq!(decision1.tier, ComputeTier::LocalEdge);

        // With preference for regional edge (20ms latency allows RegionalEdge)
        let mut req = WorkloadRequirements::inference()
            .with_latency_constraint_ms(20)
            .with_memory_mb(100); // Small enough for edge
        req.preferred_tier = Some(ComputeTier::RegionalEdge);

        let id2 = scheduler.submit(req).unwrap();
        let decision2 = scheduler.place(id2).unwrap();
        assert_eq!(decision2.tier, ComputeTier::RegionalEdge);
    }
}
