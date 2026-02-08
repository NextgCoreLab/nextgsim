//! Compute tier definitions for Service Hosting Environment
//!
//! This module defines the three-tier compute model per 3GPP TS 23.558:
//! - Local Edge: Ultra-low latency (<10ms), inference-only
//! - Regional Edge: Low latency (<20ms), fine-tuning capable
//! - Core Cloud: No latency constraints, full training capabilities

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::resource::{ResourceCapacity, ResourceUsage};

/// Compute tier in the SHE hierarchy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComputeTier {
    /// Local edge - closest to UE, lowest latency (<10ms)
    LocalEdge,
    /// Regional edge - aggregation point, low latency (<20ms)
    RegionalEdge,
    /// Core cloud - centralized, no latency constraints
    CoreCloud,
}

impl ComputeTier {
    /// Returns the maximum latency for this tier in milliseconds
    pub fn max_latency_ms(&self) -> u32 {
        match self {
            ComputeTier::LocalEdge => 10,
            ComputeTier::RegionalEdge => 20,
            ComputeTier::CoreCloud => u32::MAX, // No constraint
        }
    }

    /// Returns the tier priority (lower is better for latency-sensitive workloads)
    pub fn priority(&self) -> u8 {
        match self {
            ComputeTier::LocalEdge => 0,
            ComputeTier::RegionalEdge => 1,
            ComputeTier::CoreCloud => 2,
        }
    }

    /// Returns all tiers in order of proximity (closest first)
    pub fn all_ordered() -> &'static [ComputeTier] {
        &[
            ComputeTier::LocalEdge,
            ComputeTier::RegionalEdge,
            ComputeTier::CoreCloud,
        ]
    }
}

impl std::fmt::Display for ComputeTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComputeTier::LocalEdge => write!(f, "LocalEdge"),
            ComputeTier::RegionalEdge => write!(f, "RegionalEdge"),
            ComputeTier::CoreCloud => write!(f, "CoreCloud"),
        }
    }
}

/// Compute capabilities supported by a node
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComputeCapability {
    /// Inference only (forward pass)
    Inference,
    /// Fine-tuning (limited backward pass)
    FineTuning,
    /// Full training capability
    Training,
}

impl ComputeCapability {
    /// Returns the minimum tier required for this capability
    pub fn minimum_tier(&self) -> ComputeTier {
        match self {
            ComputeCapability::Inference => ComputeTier::LocalEdge,
            ComputeCapability::FineTuning => ComputeTier::RegionalEdge,
            ComputeCapability::Training => ComputeTier::CoreCloud,
        }
    }

    /// Checks if this capability is supported on a given tier
    pub fn supported_on(&self, tier: ComputeTier) -> bool {
        tier.priority() >= self.minimum_tier().priority()
    }
}

impl std::fmt::Display for ComputeCapability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComputeCapability::Inference => write!(f, "Inference"),
            ComputeCapability::FineTuning => write!(f, "FineTuning"),
            ComputeCapability::Training => write!(f, "Training"),
        }
    }
}

/// A compute node in the SHE infrastructure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeNode {
    /// Unique node identifier
    pub id: u32,
    /// Node name
    pub name: String,
    /// Compute tier this node belongs to
    pub tier: ComputeTier,
    /// Total resource capacity
    pub capacity: ResourceCapacity,
    /// Current resource usage
    pub usage: ResourceUsage,
    /// Supported capabilities
    pub capabilities: Vec<ComputeCapability>,
    /// Whether the node is healthy and available
    pub is_available: bool,
    /// Associated cell IDs (for local edge nodes)
    pub cell_ids: Vec<i32>,
    /// Network slice ID (TS 23.501 slice-specific resources)
    pub slice_id: Option<u32>,
    /// Tenant ID for multi-tenancy isolation
    pub tenant_id: Option<u32>,
}

impl ComputeNode {
    /// Creates a new compute node
    pub fn new(id: u32, name: impl Into<String>, tier: ComputeTier, capacity: ResourceCapacity) -> Self {
        // Set default capabilities based on tier
        let capabilities = match tier {
            ComputeTier::LocalEdge => vec![ComputeCapability::Inference],
            ComputeTier::RegionalEdge => vec![ComputeCapability::Inference, ComputeCapability::FineTuning],
            ComputeTier::CoreCloud => vec![
                ComputeCapability::Inference,
                ComputeCapability::FineTuning,
                ComputeCapability::Training,
            ],
        };

        Self {
            id,
            name: name.into(),
            tier,
            capacity,
            usage: ResourceUsage::default(),
            capabilities,
            is_available: true,
            cell_ids: Vec::new(),
            slice_id: None,
            tenant_id: None,
        }
    }

    /// Returns the available compute capacity (FLOPS)
    pub fn available_compute(&self) -> u64 {
        self.capacity.compute_flops.saturating_sub(self.usage.compute_flops)
    }

    /// Returns the available memory (bytes)
    pub fn available_memory(&self) -> u64 {
        self.capacity.memory_bytes.saturating_sub(self.usage.memory_bytes)
    }

    /// Returns the compute utilization (0.0 to 1.0)
    pub fn compute_utilization(&self) -> f64 {
        if self.capacity.compute_flops == 0 {
            return 0.0;
        }
        self.usage.compute_flops as f64 / self.capacity.compute_flops as f64
    }

    /// Returns the memory utilization (0.0 to 1.0)
    pub fn memory_utilization(&self) -> f64 {
        if self.capacity.memory_bytes == 0 {
            return 0.0;
        }
        self.usage.memory_bytes as f64 / self.capacity.memory_bytes as f64
    }

    /// Checks if the node supports a capability
    pub fn supports(&self, capability: ComputeCapability) -> bool {
        self.capabilities.contains(&capability)
    }

    /// Checks if the node can accommodate a workload
    pub fn can_accommodate(&self, compute_flops: u64, memory_bytes: u64) -> bool {
        self.is_available
            && self.available_compute() >= compute_flops
            && self.available_memory() >= memory_bytes
    }

    /// Allocates resources for a workload
    pub fn allocate(&mut self, compute_flops: u64, memory_bytes: u64) {
        self.usage.compute_flops += compute_flops;
        self.usage.memory_bytes += memory_bytes;
        self.usage.active_workloads += 1;
    }

    /// Releases resources from a workload
    pub fn release(&mut self, compute_flops: u64, memory_bytes: u64) {
        self.usage.compute_flops = self.usage.compute_flops.saturating_sub(compute_flops);
        self.usage.memory_bytes = self.usage.memory_bytes.saturating_sub(memory_bytes);
        self.usage.active_workloads = self.usage.active_workloads.saturating_sub(1);
    }

    /// Associates the node with a cell ID
    pub fn with_cell(mut self, cell_id: i32) -> Self {
        self.cell_ids.push(cell_id);
        self
    }

    /// Sets the network slice ID (TS 23.501)
    pub fn with_slice(mut self, slice_id: u32) -> Self {
        self.slice_id = Some(slice_id);
        self
    }

    /// Sets the tenant ID for multi-tenancy
    pub fn with_tenant(mut self, tenant_id: u32) -> Self {
        self.tenant_id = Some(tenant_id);
        self
    }

    /// Simulates inter-tier latency in milliseconds
    pub fn inter_tier_latency_ms(&self, target_tier: ComputeTier) -> u32 {
        if self.tier == target_tier {
            return 0; // Same tier
        }

        // Simplified latency model based on tier distance
        match (self.tier, target_tier) {
            (ComputeTier::LocalEdge, ComputeTier::RegionalEdge) |
            (ComputeTier::RegionalEdge, ComputeTier::LocalEdge) => 5, // 5ms between adjacent tiers
            (ComputeTier::RegionalEdge, ComputeTier::CoreCloud) |
            (ComputeTier::CoreCloud, ComputeTier::RegionalEdge) => 15, // 15ms to core
            (ComputeTier::LocalEdge, ComputeTier::CoreCloud) |
            (ComputeTier::CoreCloud, ComputeTier::LocalEdge) => 25, // 25ms edge-to-core
            _ => 0,
        }
    }
}

/// Manager for compute tiers
#[derive(Debug)]
pub struct TierManager {
    /// Nodes organized by tier
    nodes_by_tier: HashMap<ComputeTier, Vec<ComputeNode>>,
    /// Node lookup by ID
    node_by_id: HashMap<u32, (ComputeTier, usize)>,
}

impl TierManager {
    /// Creates a new tier manager
    pub fn new() -> Self {
        let mut nodes_by_tier = HashMap::new();
        nodes_by_tier.insert(ComputeTier::LocalEdge, Vec::new());
        nodes_by_tier.insert(ComputeTier::RegionalEdge, Vec::new());
        nodes_by_tier.insert(ComputeTier::CoreCloud, Vec::new());

        Self {
            nodes_by_tier,
            node_by_id: HashMap::new(),
        }
    }

    /// Adds a node to the appropriate tier
    pub fn add_node(&mut self, node: ComputeNode) {
        let tier = node.tier;
        let id = node.id;

        if let Some(nodes) = self.nodes_by_tier.get_mut(&tier) {
            let index = nodes.len();
            nodes.push(node);
            self.node_by_id.insert(id, (tier, index));
        }
    }

    /// Gets a node by ID
    pub fn get_node(&self, node_id: u32) -> Option<&ComputeNode> {
        self.node_by_id.get(&node_id).and_then(|(tier, index)| {
            self.nodes_by_tier.get(tier).and_then(|nodes| nodes.get(*index))
        })
    }

    /// Gets a mutable node by ID
    pub fn get_node_mut(&mut self, node_id: u32) -> Option<&mut ComputeNode> {
        if let Some(&(tier, index)) = self.node_by_id.get(&node_id) {
            return self.nodes_by_tier.get_mut(&tier).and_then(|nodes| nodes.get_mut(index));
        }
        None
    }

    /// Gets all nodes in a tier
    pub fn nodes_in_tier(&self, tier: ComputeTier) -> &[ComputeNode] {
        self.nodes_by_tier.get(&tier).map(std::vec::Vec::as_slice).unwrap_or(&[])
    }

    /// Gets all available nodes in a tier that support a capability
    pub fn available_nodes(&self, tier: ComputeTier, capability: ComputeCapability) -> Vec<&ComputeNode> {
        self.nodes_by_tier
            .get(&tier)
            .map(|nodes| {
                nodes
                    .iter()
                    .filter(|n| n.is_available && n.supports(capability))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Finds the best node for a workload in a tier
    pub fn find_best_node(
        &self,
        tier: ComputeTier,
        capability: ComputeCapability,
        compute_flops: u64,
        memory_bytes: u64,
    ) -> Option<&ComputeNode> {
        self.available_nodes(tier, capability)
            .into_iter()
            .filter(|n| n.can_accommodate(compute_flops, memory_bytes))
            .min_by_key(|n| {
                // Prefer nodes with lower utilization
                ((n.compute_utilization() + n.memory_utilization()) * 1000.0) as u64
            })
    }

    /// Returns aggregate capacity for a tier
    pub fn tier_capacity(&self, tier: ComputeTier) -> ResourceCapacity {
        self.nodes_by_tier
            .get(&tier)
            .map(|nodes| {
                nodes.iter().fold(ResourceCapacity::default(), |acc, node| {
                    ResourceCapacity {
                        compute_flops: acc.compute_flops + node.capacity.compute_flops,
                        memory_bytes: acc.memory_bytes + node.capacity.memory_bytes,
                        gpu_count: acc.gpu_count + node.capacity.gpu_count,
                        npu_count: acc.npu_count + node.capacity.npu_count,
                        tpu_count: acc.tpu_count + node.capacity.tpu_count,
                        fpga_count: acc.fpga_count + node.capacity.fpga_count,
                    }
                })
            })
            .unwrap_or_default()
    }

    /// Returns aggregate usage for a tier
    pub fn tier_usage(&self, tier: ComputeTier) -> ResourceUsage {
        self.nodes_by_tier
            .get(&tier)
            .map(|nodes| {
                nodes.iter().fold(ResourceUsage::default(), |acc, node| {
                    ResourceUsage {
                        compute_flops: acc.compute_flops + node.usage.compute_flops,
                        memory_bytes: acc.memory_bytes + node.usage.memory_bytes,
                        active_workloads: acc.active_workloads + node.usage.active_workloads,
                    }
                })
            })
            .unwrap_or_default()
    }

    /// Returns the number of nodes in each tier
    pub fn node_counts(&self) -> HashMap<ComputeTier, usize> {
        self.nodes_by_tier
            .iter()
            .map(|(tier, nodes)| (*tier, nodes.len()))
            .collect()
    }
}

impl Default for TierManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_capacity() -> ResourceCapacity {
        ResourceCapacity {
            compute_flops: 1_000_000_000_000, // 1 TFLOPS
            memory_bytes: 8 * 1024 * 1024 * 1024, // 8 GB
            gpu_count: 1,
            npu_count: 0,
            tpu_count: 0,
            fpga_count: 0,
        }
    }

    #[test]
    fn test_compute_tier_latency() {
        assert_eq!(ComputeTier::LocalEdge.max_latency_ms(), 10);
        assert_eq!(ComputeTier::RegionalEdge.max_latency_ms(), 20);
        assert_eq!(ComputeTier::CoreCloud.max_latency_ms(), u32::MAX);
    }

    #[test]
    fn test_compute_tier_priority() {
        assert!(ComputeTier::LocalEdge.priority() < ComputeTier::RegionalEdge.priority());
        assert!(ComputeTier::RegionalEdge.priority() < ComputeTier::CoreCloud.priority());
    }

    #[test]
    fn test_capability_minimum_tier() {
        assert_eq!(ComputeCapability::Inference.minimum_tier(), ComputeTier::LocalEdge);
        assert_eq!(ComputeCapability::FineTuning.minimum_tier(), ComputeTier::RegionalEdge);
        assert_eq!(ComputeCapability::Training.minimum_tier(), ComputeTier::CoreCloud);
    }

    #[test]
    fn test_capability_supported_on() {
        // Inference supported everywhere
        assert!(ComputeCapability::Inference.supported_on(ComputeTier::LocalEdge));
        assert!(ComputeCapability::Inference.supported_on(ComputeTier::RegionalEdge));
        assert!(ComputeCapability::Inference.supported_on(ComputeTier::CoreCloud));

        // Fine-tuning not on local edge
        assert!(!ComputeCapability::FineTuning.supported_on(ComputeTier::LocalEdge));
        assert!(ComputeCapability::FineTuning.supported_on(ComputeTier::RegionalEdge));

        // Training only on core
        assert!(!ComputeCapability::Training.supported_on(ComputeTier::LocalEdge));
        assert!(!ComputeCapability::Training.supported_on(ComputeTier::RegionalEdge));
        assert!(ComputeCapability::Training.supported_on(ComputeTier::CoreCloud));
    }

    #[test]
    fn test_compute_node_creation() {
        let node = ComputeNode::new(1, "edge-1", ComputeTier::LocalEdge, test_capacity());

        assert_eq!(node.id, 1);
        assert_eq!(node.tier, ComputeTier::LocalEdge);
        assert!(node.supports(ComputeCapability::Inference));
        assert!(!node.supports(ComputeCapability::FineTuning));
    }

    #[test]
    fn test_compute_node_resources() {
        let mut node = ComputeNode::new(1, "edge-1", ComputeTier::LocalEdge, test_capacity());

        assert_eq!(node.available_compute(), 1_000_000_000_000);
        assert_eq!(node.compute_utilization(), 0.0);

        node.allocate(500_000_000_000, 4 * 1024 * 1024 * 1024);

        assert_eq!(node.available_compute(), 500_000_000_000);
        assert!((node.compute_utilization() - 0.5).abs() < 0.01);

        node.release(500_000_000_000, 4 * 1024 * 1024 * 1024);

        assert_eq!(node.available_compute(), 1_000_000_000_000);
    }

    #[test]
    fn test_tier_manager() {
        let mut manager = TierManager::new();

        manager.add_node(ComputeNode::new(1, "edge-1", ComputeTier::LocalEdge, test_capacity()));
        manager.add_node(ComputeNode::new(2, "edge-2", ComputeTier::LocalEdge, test_capacity()));
        manager.add_node(ComputeNode::new(3, "regional-1", ComputeTier::RegionalEdge, test_capacity()));

        let counts = manager.node_counts();
        assert_eq!(counts.get(&ComputeTier::LocalEdge), Some(&2));
        assert_eq!(counts.get(&ComputeTier::RegionalEdge), Some(&1));
        assert_eq!(counts.get(&ComputeTier::CoreCloud), Some(&0));
    }

    #[test]
    fn test_tier_manager_find_best_node() {
        let mut manager = TierManager::new();

        let mut node1 = ComputeNode::new(1, "edge-1", ComputeTier::LocalEdge, test_capacity());
        node1.allocate(500_000_000_000, 0); // 50% utilized

        let node2 = ComputeNode::new(2, "edge-2", ComputeTier::LocalEdge, test_capacity());
        // 0% utilized

        manager.add_node(node1);
        manager.add_node(node2);

        // Should find node2 (lower utilization)
        let best = manager.find_best_node(
            ComputeTier::LocalEdge,
            ComputeCapability::Inference,
            100_000_000_000,
            1024 * 1024 * 1024,
        );

        assert!(best.is_some());
        assert_eq!(best.unwrap().id, 2);
    }
}
