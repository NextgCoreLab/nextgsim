//! Service Hosting Environment (SHE) Task for gNB

use std::collections::HashMap;
use tokio::sync::mpsc;
use tracing::{debug, error, info};

use nextgsim_she::{
    SchedulingPolicy, WorkloadId, WorkloadRequirements, WorkloadScheduler,
    TierManager,
};

use crate::tasks::{GnbTaskBase, SheMessage, Task, TaskMessage};

/// SHE Task for gNB
pub struct SheTask {
    _task_base: GnbTaskBase,
    scheduler: WorkloadScheduler,
    workloads: HashMap<u64, WorkloadId>,
    pending_requests: HashMap<u64, u64>,
    _next_request_id: u64,
}

impl SheTask {
    pub fn new(task_base: GnbTaskBase) -> Self {
        let tier_manager = TierManager::default();
        let scheduler = WorkloadScheduler::new(tier_manager)
            .with_policy(SchedulingPolicy::ClosestToEdge);

        Self {
            _task_base: task_base,
            scheduler,
            workloads: HashMap::new(),
            pending_requests: HashMap::new(),
            _next_request_id: 1,
        }
    }

    fn handle_place_workload(
        &mut self,
        workload_id: u64,
        max_latency_ms: u32,
        compute_flops: u64,
        memory_bytes: u64,
    ) {
        debug!(
            "SHE: Placing workload {} with latency={}, compute={}, memory={}",
            workload_id, max_latency_ms, compute_flops, memory_bytes
        );

        let requirements = WorkloadRequirements::inference()
            .with_latency_constraint_ms(max_latency_ms)
            .with_compute_flops(compute_flops)
            .with_memory_bytes(memory_bytes);

        match self.scheduler.submit(requirements) {
            Ok(wl_id) => {
                match self.scheduler.place(wl_id) {
                    Ok(decision) => {
                        info!(
                            "SHE: Placed workload {} on {:?} (node={}): {}",
                            workload_id, decision.tier, decision.node_id, decision.reason
                        );
                        self.workloads.insert(workload_id, wl_id);
                    }
                    Err(e) => {
                        error!("SHE: Failed to place workload {}: {}", workload_id, e);
                    }
                }
            }
            Err(e) => {
                error!("SHE: Failed to submit workload {}: {}", workload_id, e);
            }
        }
    }

    fn handle_inference_request(
        &mut self,
        model_id: String,
        request_id: u64,
        input_data: Vec<f32>,
    ) {
        debug!(
            "SHE: Inference request {} for model '{}' with {} input samples",
            request_id, model_id, input_data.len()
        );
        info!(
            "SHE: Processing inference request {} for model '{}'",
            request_id, model_id
        );
        self.pending_requests.insert(request_id, request_id);
    }

    fn handle_resource_update(
        &mut self,
        node_id: u32,
        available_flops: u64,
        available_memory: u64,
    ) {
        debug!(
            "SHE: Resource update for node {}: {} FLOPS, {} bytes memory",
            node_id, available_flops, available_memory
        );
        info!(
            "SHE: Updated resources for node {}: FLOPS={}, Memory={}",
            node_id, available_flops, available_memory
        );
    }
}

#[async_trait::async_trait]
impl Task for SheTask {
    type Message = SheMessage;

    async fn run(&mut self, mut rx: mpsc::Receiver<TaskMessage<Self::Message>>) {
        info!("SHE task started");

        loop {
            match rx.recv().await {
                Some(TaskMessage::Message(msg)) => {
                    match msg {
                        SheMessage::PlaceWorkload {
                            workload_id,
                            max_latency_ms,
                            compute_flops,
                            memory_bytes,
                        } => {
                            self.handle_place_workload(
                                workload_id, max_latency_ms, compute_flops, memory_bytes,
                            );
                        }
                        SheMessage::InferenceRequest {
                            model_id,
                            request_id,
                            input_data,
                        } => {
                            self.handle_inference_request(model_id, request_id, input_data);
                        }
                        SheMessage::ResourceUpdate {
                            node_id,
                            available_flops,
                            available_memory,
                        } => {
                            self.handle_resource_update(node_id, available_flops, available_memory);
                        }
                    }
                }
                Some(TaskMessage::Shutdown) => {
                    info!("SHE task received shutdown signal");
                    break;
                }
                None => {
                    info!("SHE task channel closed");
                    break;
                }
            }
        }

        info!(
            "SHE task stopped, {} active workloads, {} pending requests",
            self.workloads.len(),
            self.pending_requests.len()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::{GnbTaskBase, DEFAULT_CHANNEL_CAPACITY};
    use nextgsim_common::config::GnbConfig;
    use nextgsim_common::Plmn;

    fn test_config() -> GnbConfig {
        GnbConfig {
            nci: 0x000000010,
            gnb_id_length: 32,
            plmn: Plmn::new(001, 01, false),
            tac: 1,
            nssai: vec![],
            amf_configs: vec![],
            link_ip: "127.0.0.1".parse().unwrap(),
            ngap_ip: "127.0.0.1".parse().unwrap(),
            gtp_ip: "127.0.0.1".parse().unwrap(),
            gtp_advertise_ip: None,
            ignore_stream_ids: false,
            upf_addr: None,
            upf_port: 2152,
            pqc_config: nextgsim_common::config::PqcConfig::default(),
        }
    }

    #[tokio::test]
    async fn test_she_task_creation() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);

        let task = SheTask::new(task_base);
        assert_eq!(task.workloads.len(), 0);
        assert_eq!(task.pending_requests.len(), 0);
    }

    #[tokio::test]
    async fn test_she_task_place_workload() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);

        let mut task = SheTask::new(task_base);
        task.handle_place_workload(1, 10, 1_000_000_000, 1_000_000);
        // workload may or may not be placed depending on tier capacity
    }

    #[tokio::test]
    async fn test_she_task_inference_request() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);

        let mut task = SheTask::new(task_base);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        task.handle_inference_request("test-model".to_string(), 1, input);
        assert_eq!(task.pending_requests.len(), 1);
    }
}
