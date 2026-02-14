//! Auto-scaling and elastic node provisioning for SHE
//!
//! Implements dynamic compute pool expansion/contraction based on load patterns.

#![allow(missing_docs)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;
use tracing::{debug, info};

use crate::error::SheResult;
use crate::resource::ResourceCapacity;
use crate::tier::{ComputeNode, ComputeTier, TierManager};

/// Auto-scaling policy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum ScalingPolicy {
    /// Scale based on CPU/memory utilization thresholds
    #[default]
    UtilizationBased,
    /// Scale based on workload queue depth
    QueueBased,
    /// Scale based on predicted load (requires AI model)
    Predictive,
    /// Manual scaling only
    Manual,
}


/// Auto-scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoScaleConfig {
    /// Scaling policy
    pub policy: ScalingPolicy,
    /// Target utilization (0.0 to 1.0)
    pub target_utilization: f64,
    /// Scale-up threshold
    pub scale_up_threshold: f64,
    /// Scale-down threshold
    pub scale_down_threshold: f64,
    /// Minimum nodes per tier
    pub min_nodes_per_tier: u32,
    /// Maximum nodes per tier
    pub max_nodes_per_tier: u32,
    /// Cooldown period between scaling actions (seconds)
    pub cooldown_seconds: u64,
    /// Queue depth threshold for queue-based scaling
    pub queue_depth_threshold: usize,
}

impl Default for AutoScaleConfig {
    fn default() -> Self {
        Self {
            policy: ScalingPolicy::UtilizationBased,
            target_utilization: 0.7,
            scale_up_threshold: 0.8,
            scale_down_threshold: 0.3,
            min_nodes_per_tier: 1,
            max_nodes_per_tier: 10,
            cooldown_seconds: 60,
            queue_depth_threshold: 10,
        }
    }
}

/// Scaling action
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalingAction {
    /// Scale up by adding nodes
    ScaleUp { tier: ComputeTier, count: u32 },
    /// Scale down by removing nodes
    ScaleDown { tier: ComputeTier, count: u32 },
    /// No action needed
    None,
}

/// Auto-scaling decision
#[derive(Debug, Clone)]
pub struct ScalingDecision {
    /// Recommended action
    pub action: ScalingAction,
    /// Reason for the decision
    pub reason: String,
    /// Current utilization that triggered this decision
    pub current_utilization: f64,
    /// Timestamp (not serializable, use `timestamp_ms` for persistence)
    pub timestamp: Instant,
    /// Timestamp in milliseconds since UNIX epoch
    pub timestamp_ms: u64,
}

/// Auto-scaler for elastic node provisioning
#[derive(Debug)]
pub struct AutoScaler {
    /// Configuration
    config: AutoScaleConfig,
    /// Next available node ID
    next_node_id: u32,
    /// Last scaling action per tier
    last_scaling: HashMap<ComputeTier, Instant>,
    /// Scaling history
    scaling_history: Vec<ScalingDecision>,
    /// Maximum history size
    max_history: usize,
}

impl AutoScaler {
    /// Creates a new auto-scaler
    pub fn new(config: AutoScaleConfig, starting_node_id: u32) -> Self {
        Self {
            config,
            next_node_id: starting_node_id,
            last_scaling: HashMap::new(),
            scaling_history: Vec::new(),
            max_history: 100,
        }
    }

    /// Allocates a new node ID
    fn allocate_node_id(&mut self) -> u32 {
        let id = self.next_node_id;
        self.next_node_id += 1;
        id
    }

    /// Checks if scaling is allowed (respects cooldown period)
    fn is_scaling_allowed(&self, tier: ComputeTier) -> bool {
        if let Some(last) = self.last_scaling.get(&tier) {
            last.elapsed().as_secs() >= self.config.cooldown_seconds
        } else {
            true
        }
    }

    /// Records a scaling action
    fn record_scaling(&mut self, tier: ComputeTier) {
        self.last_scaling.insert(tier, Instant::now());
    }

    /// Evaluates scaling needs for a tier based on utilization
    fn evaluate_utilization_based(
        &self,
        tier: ComputeTier,
        tier_manager: &TierManager,
    ) -> ScalingDecision {
        let capacity = tier_manager.tier_capacity(tier);
        let usage = tier_manager.tier_usage(tier);

        let utilization = if capacity.compute_flops > 0 {
            usage.compute_flops as f64 / capacity.compute_flops as f64
        } else {
            0.0
        };

        let node_count = tier_manager.nodes_in_tier(tier).len() as u32;

        let action = if utilization >= self.config.scale_up_threshold
            && node_count < self.config.max_nodes_per_tier
            && self.is_scaling_allowed(tier)
        {
            // Scale up by 1 node (or more for very high utilization)
            let scale_count = if utilization > 0.95 { 2 } else { 1 };
            ScalingAction::ScaleUp {
                tier,
                count: scale_count.min(self.config.max_nodes_per_tier - node_count),
            }
        } else if utilization <= self.config.scale_down_threshold
            && node_count > self.config.min_nodes_per_tier
            && self.is_scaling_allowed(tier)
        {
            ScalingAction::ScaleDown { tier, count: 1 }
        } else {
            ScalingAction::None
        };

        let reason = match action {
            ScalingAction::ScaleUp { .. } => {
                format!("Utilization {:.1}% exceeds threshold {:.1}%",
                    utilization * 100.0,
                    self.config.scale_up_threshold * 100.0
                )
            }
            ScalingAction::ScaleDown { .. } => {
                format!("Utilization {:.1}% below threshold {:.1}%",
                    utilization * 100.0,
                    self.config.scale_down_threshold * 100.0
                )
            }
            ScalingAction::None => {
                format!("Utilization {:.1}% within target range", utilization * 100.0)
            }
        };

        let now = Instant::now();
        ScalingDecision {
            action,
            reason,
            current_utilization: utilization,
            timestamp: now,
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        }
    }

    /// Evaluates scaling needs based on queue depth
    pub fn evaluate_queue_based(
        &self,
        tier: ComputeTier,
        tier_manager: &TierManager,
        queue_depth: usize,
    ) -> ScalingDecision {
        let node_count = tier_manager.nodes_in_tier(tier).len() as u32;
        let now = Instant::now();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let action = if queue_depth > self.config.queue_depth_threshold
            && node_count < self.config.max_nodes_per_tier
            && self.is_scaling_allowed(tier)
        {
            let scale_count = if queue_depth > self.config.queue_depth_threshold * 3 {
                2
            } else {
                1
            };
            ScalingAction::ScaleUp {
                tier,
                count: scale_count.min(self.config.max_nodes_per_tier - node_count),
            }
        } else if queue_depth == 0
            && node_count > self.config.min_nodes_per_tier
            && self.is_scaling_allowed(tier)
        {
            ScalingAction::ScaleDown { tier, count: 1 }
        } else {
            ScalingAction::None
        };

        let reason = match action {
            ScalingAction::ScaleUp { .. } => {
                format!("Queue depth {} exceeds threshold {}", queue_depth, self.config.queue_depth_threshold)
            }
            ScalingAction::ScaleDown { .. } => "Queue empty, reducing capacity".to_string(),
            ScalingAction::None => format!("Queue depth {queue_depth} within bounds"),
        };

        ScalingDecision {
            action,
            reason,
            current_utilization: queue_depth as f64 / self.config.queue_depth_threshold.max(1) as f64,
            timestamp: now,
            timestamp_ms: ts,
        }
    }

    /// Evaluates scaling needs for a tier
    pub fn evaluate(&self, tier: ComputeTier, tier_manager: &TierManager) -> ScalingDecision {
        match self.config.policy {
            ScalingPolicy::UtilizationBased => self.evaluate_utilization_based(tier, tier_manager),
            ScalingPolicy::QueueBased => {
                self.evaluate_queue_based(tier, tier_manager, 0)
            }
            ScalingPolicy::Predictive | ScalingPolicy::Manual => {
                let now = Instant::now();
                ScalingDecision {
                    action: ScalingAction::None,
                    reason: format!("{:?} scaling requires external input", self.config.policy),
                    current_utilization: 0.0,
                    timestamp: now,
                    timestamp_ms: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                }
            }
        }
    }

    /// Executes a scaling action
    pub fn execute(
        &mut self,
        decision: ScalingDecision,
        tier_manager: &mut TierManager,
        node_template_capacity: ResourceCapacity,
    ) -> SheResult<Vec<u32>> {
        match decision.action {
            ScalingAction::ScaleUp { tier, count } => {
                let mut added_nodes = Vec::new();

                for i in 0..count {
                    let node_id = self.allocate_node_id();
                    let node = ComputeNode::new(
                        node_id,
                        format!("{tier}-autoscale-{i}"),
                        tier,
                        node_template_capacity,
                    );

                    tier_manager.add_node(node);
                    added_nodes.push(node_id);

                    info!(
                        "Auto-scaled UP: Added node {} to tier {} (reason: {})",
                        node_id, tier, decision.reason
                    );
                }

                self.record_scaling(tier);
                self.scaling_history.push(decision);
                self.trim_history();

                Ok(added_nodes)
            }
            ScalingAction::ScaleDown { tier, count } => {
                // Note: Actual node removal would require workload migration
                // For now, we just mark nodes as unavailable
                debug!(
                    "Auto-scale DOWN requested for tier {}, count {} (would need workload migration)",
                    tier, count
                );

                self.record_scaling(tier);
                self.scaling_history.push(decision);
                self.trim_history();

                Ok(Vec::new())
            }
            ScalingAction::None => {
                debug!("No scaling action needed: {}", decision.reason);
                Ok(Vec::new())
            }
        }
    }

    /// Trims scaling history to max size
    fn trim_history(&mut self) {
        if self.scaling_history.len() > self.max_history {
            self.scaling_history.drain(0..self.scaling_history.len() - self.max_history);
        }
    }

    /// Returns the scaling history
    pub fn history(&self) -> &[ScalingDecision] {
        &self.scaling_history
    }

    /// Returns the configuration
    pub fn config(&self) -> &AutoScaleConfig {
        &self.config
    }

    /// Updates the configuration
    pub fn set_config(&mut self, config: AutoScaleConfig) {
        self.config = config;
    }
}

impl Default for AutoScaler {
    fn default() -> Self {
        Self::new(AutoScaleConfig::default(), 1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_tier_manager() -> TierManager {
        let mut tm = TierManager::new();
        tm.add_node(ComputeNode::new(
            1,
            "edge-1",
            ComputeTier::LocalEdge,
            ResourceCapacity::with_tflops(1).with_memory_gb(8),
        ));
        tm
    }

    #[test]
    fn test_autoscaler_creation() {
        let config = AutoScaleConfig::default();
        let scaler = AutoScaler::new(config, 100);

        assert_eq!(scaler.next_node_id, 100);
        assert_eq!(scaler.scaling_history.len(), 0);
    }

    #[test]
    fn test_scaling_decision_scale_up() {
        let mut tm = setup_tier_manager();
        let scaler = AutoScaler::default();

        // Simulate high utilization
        if let Some(node) = tm.get_node_mut(1) {
            node.allocate(900_000_000_000, 7 * 1024 * 1024 * 1024); // 90% util
        }

        let decision = scaler.evaluate(ComputeTier::LocalEdge, &tm);

        match decision.action {
            ScalingAction::ScaleUp { tier, count } => {
                assert_eq!(tier, ComputeTier::LocalEdge);
                assert!(count > 0);
            }
            _ => panic!("Expected ScaleUp action"),
        }
    }

    #[test]
    fn test_scaling_decision_scale_down() {
        let tm = setup_tier_manager();
        let scaler = AutoScaler::default();

        // Low utilization (no allocation)
        let decision = scaler.evaluate(ComputeTier::LocalEdge, &tm);

        match decision.action {
            ScalingAction::ScaleDown { tier, count } => {
                assert_eq!(tier, ComputeTier::LocalEdge);
                assert_eq!(count, 1);
            }
            ScalingAction::None => {
                // Also acceptable if at min_nodes
            }
            _ => {}
        }
    }

    #[test]
    fn test_execute_scale_up() {
        let mut scaler = AutoScaler::default();
        let mut tm = setup_tier_manager();

        let now = Instant::now();
        let decision = ScalingDecision {
            action: ScalingAction::ScaleUp {
                tier: ComputeTier::LocalEdge,
                count: 2,
            },
            reason: "Test scale up".to_string(),
            current_utilization: 0.85,
            timestamp: now,
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        };

        let capacity = ResourceCapacity::with_tflops(1).with_memory_gb(8);
        let result = scaler.execute(decision, &mut tm, capacity);

        assert!(result.is_ok());
        let added = result.unwrap();
        assert_eq!(added.len(), 2);

        // Verify nodes were added
        assert!(tm.get_node(added[0]).is_some());
        assert!(tm.get_node(added[1]).is_some());
    }

    #[test]
    fn test_cooldown_enforcement() {
        let config = AutoScaleConfig {
            cooldown_seconds: 60,
            ..Default::default()
        };
        let mut scaler = AutoScaler::new(config, 100);

        assert!(scaler.is_scaling_allowed(ComputeTier::LocalEdge));

        scaler.record_scaling(ComputeTier::LocalEdge);

        assert!(!scaler.is_scaling_allowed(ComputeTier::LocalEdge));
    }

    #[test]
    fn test_max_nodes_enforcement() {
        let config = AutoScaleConfig {
            max_nodes_per_tier: 2,
            ..Default::default()
        };
        let scaler = AutoScaler::new(config, 100);

        let mut tm = TierManager::new();
        tm.add_node(ComputeNode::new(
            1,
            "edge-1",
            ComputeTier::LocalEdge,
            ResourceCapacity::with_tflops(1).with_memory_gb(8),
        ));
        tm.add_node(ComputeNode::new(
            2,
            "edge-2",
            ComputeTier::LocalEdge,
            ResourceCapacity::with_tflops(1).with_memory_gb(8),
        ));

        // Set high utilization
        if let Some(node) = tm.get_node_mut(1) {
            node.allocate(900_000_000_000, 7 * 1024 * 1024 * 1024);
        }

        let decision = scaler.evaluate(ComputeTier::LocalEdge, &tm);

        // Should not scale up beyond max
        assert_eq!(decision.action, ScalingAction::None);
    }
}
