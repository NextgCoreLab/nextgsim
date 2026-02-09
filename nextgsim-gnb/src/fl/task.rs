//! Federated Learning Aggregator Task for gNB

use std::collections::HashMap;
use tokio::sync::mpsc;
use tracing::{debug, info};
use nextgsim_fl::{FederatedAggregator, AggregationAlgorithm};
use crate::tasks::{GnbTaskBase, FlAggregatorMessage, Task, TaskMessage};

pub struct FlAggregatorTask {
    _task_base: GnbTaskBase,
    _aggregator: FederatedAggregator,
    participants: HashMap<String, u64>,
    current_round: u64,
}

impl FlAggregatorTask {
    pub fn new(task_base: GnbTaskBase) -> Self {
        Self {
            _task_base: task_base,
            _aggregator: FederatedAggregator::new(AggregationAlgorithm::FedAvg, 2),
            participants: HashMap::new(),
            current_round: 0,
        }
    }
}

#[async_trait::async_trait]
impl Task for FlAggregatorTask {
    type Message = FlAggregatorMessage;

    async fn run(&mut self, mut rx: mpsc::Receiver<TaskMessage<Self::Message>>) {
        info!("FL Aggregator task started");
        loop {
            match rx.recv().await {
                Some(TaskMessage::Message(msg)) => {
                    match msg {
                        FlAggregatorMessage::RegisterParticipant { participant_id, compute_capability } => {
                            debug!("FL: Registering participant {} (compute={})", participant_id, compute_capability);
                            self.participants.insert(participant_id, compute_capability);
                        }
                        FlAggregatorMessage::SubmitUpdate { participant_id, round, gradients: _, num_samples } => {
                            debug!("FL: Update from {} for round {} ({} samples)", participant_id, round, num_samples);
                        }
                        FlAggregatorMessage::Aggregate { round } => {
                            debug!("FL: Aggregating round {}", round);
                            self.current_round = round;
                        }
                        FlAggregatorMessage::DistributeModel { version, weights } => {
                            debug!("FL: Distributing model v{} ({} params)", version, weights.len());
                        }
                    }
                }
                Some(TaskMessage::Shutdown) => break,
                None => break,
            }
        }
        info!("FL Aggregator task stopped, {} participants, round {}", self.participants.len(), self.current_round);
    }
}
