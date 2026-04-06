//! Federated Learning Aggregator Task for gNB

use std::collections::HashMap;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};
use nextgsim_fl::{FederatedAggregator, AggregationAlgorithm, ModelUpdate};
use crate::tasks::{GnbTaskBase, FlAggregatorMessage, Task, TaskMessage};

pub struct FlAggregatorTask {
    _task_base: GnbTaskBase,
    aggregator: FederatedAggregator,
    participants: HashMap<String, u64>,
    current_round: u64,
}

impl FlAggregatorTask {
    pub fn new(task_base: GnbTaskBase) -> Self {
        Self {
            _task_base: task_base,
            aggregator: FederatedAggregator::new(AggregationAlgorithm::FedAvg, 2),
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
                            self.aggregator.register_participant(&participant_id, compute_capability);
                            self.participants.insert(participant_id, compute_capability);
                        }
                        FlAggregatorMessage::SubmitUpdate { participant_id, round, gradients, num_samples } => {
                            debug!("FL: Update from {} for round {} ({} samples)", participant_id, round, num_samples);
                            let update = ModelUpdate {
                                participant_id: participant_id.clone(),
                                base_version: round,
                                gradients,
                                num_samples,
                                loss: 0.0,
                                timestamp_ms: std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .map(|d| d.as_millis() as u64)
                                    .unwrap_or(0),
                            };
                            if let Err(e) = self.aggregator.submit_update(update) {
                                warn!("FL: Failed to submit update from {}: {}", participant_id, e);
                            }
                        }
                        FlAggregatorMessage::Aggregate { round } => {
                            debug!("FL: Aggregating round {}", round);
                            self.current_round = round;
                            match self.aggregator.aggregate() {
                                Ok(model) => {
                                    info!("FL: Aggregation complete for round {}, {} participants, {} params",
                                        round, model.num_participants, model.weights.len());
                                }
                                Err(e) => {
                                    warn!("FL: Aggregation failed for round {}: {}", round, e);
                                }
                            }
                        }
                        FlAggregatorMessage::DistributeModel { version, weights } => {
                            debug!("FL: Distributing model v{} ({} params)", version, weights.len());
                            self.aggregator.initialize_model(weights);
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
