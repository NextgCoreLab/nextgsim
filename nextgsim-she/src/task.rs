//! SHE Task implementation
//!
//! Implements the async task for the Service Hosting Environment,
//! handling workload placement, resource management, and inference requests.

use std::collections::HashMap;
use std::path::Path;

use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use nextgsim_ai::{InferenceEngine, OnnxEngine, ExecutionProvider};

use crate::error::SheResult;
use crate::messages::{SheMessage, SheResponse};
use crate::resource::ResourceCapacity;
use crate::scheduler::{WorkloadScheduler, SchedulingPolicy};
use crate::tier::{ComputeNode, ComputeTier, TierManager};
use crate::workload::WorkloadRequirements;

/// Configuration for the SHE task
#[derive(Debug, Clone)]
pub struct SheTaskConfig {
    /// Local edge latency constraint (ms)
    pub local_edge_latency_ms: u32,
    /// Regional edge latency constraint (ms)
    pub regional_edge_latency_ms: u32,
    /// Scheduling policy
    pub scheduling_policy: SchedulingPolicy,
    /// Default execution provider for inference
    pub execution_provider: ExecutionProvider,
}

impl Default for SheTaskConfig {
    fn default() -> Self {
        Self {
            local_edge_latency_ms: 10,
            regional_edge_latency_ms: 20,
            scheduling_policy: SchedulingPolicy::ClosestToEdge,
            execution_provider: ExecutionProvider::Cpu,
        }
    }
}

/// Service Hosting Environment Task
///
/// Manages the three-tier compute infrastructure and handles
/// workload placement, resource management, and inference requests.
pub struct SheTask {
    /// Task configuration
    config: SheTaskConfig,
    /// Workload scheduler
    scheduler: WorkloadScheduler,
    /// Loaded inference engines by model ID
    inference_engines: HashMap<String, Box<dyn InferenceEngine>>,
    /// Model to tier mapping
    model_tiers: HashMap<String, ComputeTier>,
    /// Next request ID
    next_request_id: u64,
}

impl SheTask {
    /// Creates a new SHE task with default configuration
    pub fn new() -> Self {
        Self::with_config(SheTaskConfig::default())
    }

    /// Creates a new SHE task with the given configuration
    pub fn with_config(config: SheTaskConfig) -> Self {
        let tier_manager = Self::create_default_tier_manager(&config);
        let scheduler = WorkloadScheduler::new(tier_manager)
            .with_policy(config.scheduling_policy);

        Self {
            config,
            scheduler,
            inference_engines: HashMap::new(),
            model_tiers: HashMap::new(),
            next_request_id: 1,
        }
    }

    /// Creates the default tier manager with standard nodes
    fn create_default_tier_manager(_config: &SheTaskConfig) -> TierManager {
        let mut tier_manager = TierManager::new();

        // Local edge node (smallest, fastest)
        tier_manager.add_node(ComputeNode::new(
            1,
            "local-edge-1",
            ComputeTier::LocalEdge,
            ResourceCapacity::with_tflops(1).with_memory_gb(8).with_gpus(1),
        ));

        // Regional edge node (medium)
        tier_manager.add_node(ComputeNode::new(
            2,
            "regional-edge-1",
            ComputeTier::RegionalEdge,
            ResourceCapacity::with_tflops(10).with_memory_gb(64).with_gpus(4),
        ));

        // Core cloud node (largest)
        tier_manager.add_node(ComputeNode::new(
            3,
            "core-cloud-1",
            ComputeTier::CoreCloud,
            ResourceCapacity::with_tflops(100).with_memory_gb(512).with_gpus(8),
        ));

        tier_manager
    }

    /// Adds a compute node to the tier manager
    pub fn add_node(&mut self, node: ComputeNode) {
        self.scheduler.tier_manager_mut().add_node(node);
    }

    /// Runs the SHE task main loop
    pub async fn run(&mut self, mut rx: mpsc::Receiver<SheMessage>) {
        info!("SHE task started");

        while let Some(msg) = rx.recv().await {
            if let Err(e) = self.handle_message(msg).await {
                error!("Error handling SHE message: {}", e);
            }
        }

        info!("SHE task stopped");
    }

    /// Handles a single message
    async fn handle_message(&mut self, msg: SheMessage) -> SheResult<()> {
        match msg {
            SheMessage::PlaceWorkload {
                workload_id,
                requirements,
                response_tx,
            } => {
                self.handle_place_workload(workload_id, requirements, response_tx)
                    .await
            }

            SheMessage::WorkloadPlaced {
                workload_id,
                tier,
                node_id,
            } => {
                debug!(
                    "Workload {} confirmed on {} (node {})",
                    workload_id, tier, node_id
                );
                Ok(())
            }

            SheMessage::ReleaseWorkload { workload_id } => {
                self.scheduler.release(workload_id)?;
                Ok(())
            }

            SheMessage::MigrateWorkload {
                workload_id,
                target_tier,
            } => {
                self.scheduler.migrate(workload_id, target_tier)?;
                Ok(())
            }

            SheMessage::ResourceUpdate {
                node_id,
                tier,
                available_compute,
                available_memory,
            } => {
                debug!(
                    "Resource update for node {} on {}: {} FLOPS, {} bytes",
                    node_id, tier, available_compute, available_memory
                );
                // In a real implementation, we'd update the tier manager
                Ok(())
            }

            SheMessage::NodeHealthUpdate { node_id, is_available } => {
                if let Some(node) = self.scheduler.tier_manager_mut().get_node_mut(node_id) {
                    node.is_available = is_available;
                    info!("Node {} availability updated to {}", node_id, is_available);
                }
                Ok(())
            }

            SheMessage::InferenceRequest {
                request_id,
                model_id,
                input,
                deadline_ms,
                response_tx,
            } => {
                self.handle_inference_request(request_id, model_id, input, deadline_ms, response_tx)
                    .await
            }

            SheMessage::BatchInferenceRequest {
                request_id,
                model_id,
                inputs,
                deadline_ms,
                response_tx,
            } => {
                self.handle_batch_inference(request_id, model_id, inputs, deadline_ms, response_tx)
                    .await
            }

            SheMessage::LoadModel {
                model_id,
                model_path,
                tier,
            } => {
                self.handle_load_model(&model_id, &model_path, tier).await
            }

            SheMessage::UnloadModel { model_id, tier } => {
                self.handle_unload_model(&model_id, tier).await
            }

            SheMessage::QueryTierStatus { tier, response_tx } => {
                self.handle_query_tier_status(tier, response_tx).await
            }

            SheMessage::QueryWorkloadStatus {
                workload_id,
                response_tx,
            } => {
                self.handle_query_workload_status(workload_id, response_tx)
                    .await
            }
        }
    }

    /// Handles workload placement request
    async fn handle_place_workload(
        &mut self,
        workload_id: crate::workload::WorkloadId,
        requirements: WorkloadRequirements,
        response_tx: Option<tokio::sync::oneshot::Sender<SheResponse>>,
    ) -> SheResult<()> {
        // Submit and place workload
        let id = self.scheduler.submit(requirements)?;

        match self.scheduler.place(id) {
            Ok(decision) => {
                if let Some(tx) = response_tx {
                    let _ = tx.send(SheResponse::WorkloadPlaced {
                        workload_id,
                        tier: decision.tier,
                        node_id: decision.node_id,
                    });
                }
            }
            Err(e) => {
                if let Some(tx) = response_tx {
                    let _ = tx.send(SheResponse::PlacementFailed {
                        workload_id,
                        error: e.to_string(),
                    });
                }
            }
        }

        Ok(())
    }

    /// Handles inference request
    async fn handle_inference_request(
        &mut self,
        request_id: u64,
        model_id: String,
        input: nextgsim_ai::TensorData,
        _deadline_ms: u32,
        response_tx: Option<tokio::sync::oneshot::Sender<SheResponse>>,
    ) -> SheResult<()> {
        let start = std::time::Instant::now();

        // Get the inference engine for this model
        let result = if let Some(engine) = self.inference_engines.get(&model_id) {
            engine.infer(&input)
        } else {
            Err(nextgsim_ai::InferenceError::NotReady {
                reason: format!("Model '{model_id}' not loaded"),
            })
        };

        let latency_ms = start.elapsed().as_millis() as u32;

        if let Some(tx) = response_tx {
            match result {
                Ok(output) => {
                    let tier = self.model_tiers.get(&model_id).copied().unwrap_or(ComputeTier::LocalEdge);
                    let _ = tx.send(SheResponse::InferenceResult {
                        request_id,
                        output,
                        latency_ms,
                        tier,
                    });
                }
                Err(e) => {
                    let _ = tx.send(SheResponse::InferenceFailed {
                        request_id,
                        error: e.to_string(),
                    });
                }
            }
        }

        Ok(())
    }

    /// Handles batch inference request
    async fn handle_batch_inference(
        &mut self,
        request_id: u64,
        model_id: String,
        inputs: Vec<nextgsim_ai::TensorData>,
        _deadline_ms: u32,
        response_tx: Option<tokio::sync::oneshot::Sender<SheResponse>>,
    ) -> SheResult<()> {
        let start = std::time::Instant::now();

        let result = if let Some(engine) = self.inference_engines.get(&model_id) {
            engine.batch_infer(&inputs)
        } else {
            Err(nextgsim_ai::InferenceError::NotReady {
                reason: format!("Model '{model_id}' not loaded"),
            })
        };

        let latency_ms = start.elapsed().as_millis() as u32;

        if let Some(tx) = response_tx {
            match result {
                Ok(outputs) => {
                    let _ = tx.send(SheResponse::BatchInferenceResult {
                        request_id,
                        outputs,
                        latency_ms,
                    });
                }
                Err(e) => {
                    let _ = tx.send(SheResponse::InferenceFailed {
                        request_id,
                        error: e.to_string(),
                    });
                }
            }
        }

        Ok(())
    }

    /// Handles model load request
    async fn handle_load_model(
        &mut self,
        model_id: &str,
        model_path: &Path,
        tier: ComputeTier,
    ) -> SheResult<()> {
        info!("Loading model '{}' from {:?} to {}", model_id, model_path, tier);

        let mut engine = OnnxEngine::new(self.config.execution_provider.clone())
            .map_err(|e| crate::error::SheError::Internal(e.to_string()))?;

        engine
            .load_model(model_path)
            .map_err(|e| crate::error::SheError::Internal(e.to_string()))?;

        // Warmup the model
        if let Err(e) = engine.warmup() {
            warn!("Model warmup failed: {}", e);
        }

        self.inference_engines.insert(model_id.to_string(), Box::new(engine));
        self.model_tiers.insert(model_id.to_string(), tier);

        info!("Model '{}' loaded successfully on {}", model_id, tier);
        Ok(())
    }

    /// Handles model unload request
    async fn handle_unload_model(&mut self, model_id: &str, tier: ComputeTier) -> SheResult<()> {
        if self.model_tiers.get(model_id) == Some(&tier) {
            self.inference_engines.remove(model_id);
            self.model_tiers.remove(model_id);
            info!("Model '{}' unloaded from {}", model_id, tier);
        }
        Ok(())
    }

    /// Handles tier status query
    async fn handle_query_tier_status(
        &self,
        tier: ComputeTier,
        response_tx: tokio::sync::oneshot::Sender<SheResponse>,
    ) -> SheResult<()> {
        let capacity = self.scheduler.tier_manager().tier_capacity(tier);
        let usage = self.scheduler.tier_manager().tier_usage(tier);
        let node_count = self.scheduler.tier_manager().nodes_in_tier(tier).len();

        let _ = response_tx.send(SheResponse::TierStatus {
            tier,
            total_compute: capacity.compute_flops,
            available_compute: capacity.compute_flops.saturating_sub(usage.compute_flops),
            total_memory: capacity.memory_bytes,
            available_memory: capacity.memory_bytes.saturating_sub(usage.memory_bytes),
            active_workloads: usage.active_workloads,
            node_count,
        });

        Ok(())
    }

    /// Handles workload status query
    async fn handle_query_workload_status(
        &self,
        workload_id: crate::workload::WorkloadId,
        response_tx: tokio::sync::oneshot::Sender<SheResponse>,
    ) -> SheResult<()> {
        let response = if let Some(workload) = self.scheduler.get_workload(workload_id) {
            SheResponse::WorkloadStatus {
                workload_id,
                state: workload.state,
                tier: workload.assigned_tier,
                node_id: workload.assigned_node_id,
            }
        } else {
            SheResponse::Error {
                message: format!("Workload {workload_id} not found"),
            }
        };

        let _ = response_tx.send(response);
        Ok(())
    }

    /// Returns a reference to the scheduler
    pub fn scheduler(&self) -> &WorkloadScheduler {
        &self.scheduler
    }

    /// Returns a mutable reference to the scheduler
    pub fn scheduler_mut(&mut self) -> &mut WorkloadScheduler {
        &mut self.scheduler
    }

    /// Allocates a new request ID
    pub fn allocate_request_id(&mut self) -> u64 {
        let id = self.next_request_id;
        self.next_request_id += 1;
        id
    }
}

impl Default for SheTask {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_she_task_creation() {
        let task = SheTask::new();
        assert_eq!(task.scheduler.tier_manager().nodes_in_tier(ComputeTier::LocalEdge).len(), 1);
        assert_eq!(task.scheduler.tier_manager().nodes_in_tier(ComputeTier::RegionalEdge).len(), 1);
        assert_eq!(task.scheduler.tier_manager().nodes_in_tier(ComputeTier::CoreCloud).len(), 1);
    }

    #[test]
    fn test_she_task_with_config() {
        let config = SheTaskConfig {
            local_edge_latency_ms: 5,
            regional_edge_latency_ms: 15,
            scheduling_policy: SchedulingPolicy::MostAvailable,
            execution_provider: ExecutionProvider::Cpu,
        };

        let task = SheTask::with_config(config);
        assert!(task.inference_engines.is_empty());
    }

    #[test]
    fn test_add_node() {
        let mut task = SheTask::new();

        task.add_node(ComputeNode::new(
            10,
            "extra-edge",
            ComputeTier::LocalEdge,
            ResourceCapacity::with_tflops(2).with_memory_gb(16),
        ));

        assert_eq!(task.scheduler.tier_manager().nodes_in_tier(ComputeTier::LocalEdge).len(), 2);
    }

    #[test]
    fn test_request_id_allocation() {
        let mut task = SheTask::new();

        let id1 = task.allocate_request_id();
        let id2 = task.allocate_request_id();
        let id3 = task.allocate_request_id();

        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(id3, 3);
    }

    #[tokio::test]
    async fn test_tier_status_query() {
        let task = SheTask::new();
        let (tx, rx) = tokio::sync::oneshot::channel();

        task.handle_query_tier_status(ComputeTier::LocalEdge, tx).await.unwrap();

        let response = rx.await.unwrap();
        match response {
            SheResponse::TierStatus { tier, node_count, .. } => {
                assert_eq!(tier, ComputeTier::LocalEdge);
                assert_eq!(node_count, 1);
            }
            _ => panic!("Unexpected response"),
        }
    }
}
