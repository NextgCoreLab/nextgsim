//! SHE Client Task for UE - Edge inference and workload offloading

use tokio::sync::mpsc;
use tracing::{debug, info};
use crate::tasks::{UeTaskBase, SheClientMessage, Task, TaskMessage};

pub struct SheClientTask {
    _task_base: UeTaskBase,
    pending_requests: u64,
}

impl SheClientTask {
    pub fn new(task_base: UeTaskBase) -> Self {
        Self { _task_base: task_base, pending_requests: 0 }
    }
}

#[async_trait::async_trait]
impl Task for SheClientTask {
    type Message = SheClientMessage;

    async fn run(&mut self, mut rx: mpsc::Receiver<TaskMessage<Self::Message>>) {
        info!("SHE Client task started");
        loop {
            match rx.recv().await {
                Some(TaskMessage::Message(msg)) => {
                    match msg {
                        SheClientMessage::InferenceRequest { model_id, input: _, input_shape: _, deadline_ms: _, response_tx: _ } => {
                            debug!("SHE Client: Inference request for model '{}'", model_id);
                            self.pending_requests += 1;
                        }
                        SheClientMessage::CancelInference { request_id } => {
                            debug!("SHE Client: Cancel request {}", request_id);
                        }
                        SheClientMessage::EdgeNodeUpdate { nodes } => {
                            debug!("SHE Client: Edge node update ({} nodes)", nodes.len());
                        }
                        SheClientMessage::OffloadComputation { computation_type, data: _, response_tx: _ } => {
                            debug!("SHE Client: Offload computation '{}'", computation_type);
                        }
                    }
                }
                Some(TaskMessage::Shutdown) => break,
                None => break,
            }
        }
        info!("SHE Client task stopped, {} pending requests", self.pending_requests);
    }
}
