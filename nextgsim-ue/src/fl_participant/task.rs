//! FL Participant Task for UE

use tokio::sync::mpsc;
use tracing::{debug, info};
use crate::tasks::{UeTaskBase, FlParticipantMessage, Task, TaskMessage};

pub struct FlParticipantTask {
    _task_base: UeTaskBase,
    current_round: u32,
}

impl FlParticipantTask {
    pub fn new(task_base: UeTaskBase) -> Self {
        Self { _task_base: task_base, current_round: 0 }
    }
}

#[async_trait::async_trait]
impl Task for FlParticipantTask {
    type Message = FlParticipantMessage;

    async fn run(&mut self, mut rx: mpsc::Receiver<TaskMessage<Self::Message>>) {
        info!("FL Participant task started");
        loop {
            match rx.recv().await {
                Some(TaskMessage::Message(msg)) => {
                    match msg {
                        FlParticipantMessage::ReceiveGlobalModel { round, weights: _, version } => {
                            debug!("FL Participant: Received model v{} for round {}", version, round);
                            self.current_round = round;
                        }
                        FlParticipantMessage::StartTraining { config } => {
                            debug!("FL Participant: Start training epochs={}", config.local_epochs);
                        }
                        FlParticipantMessage::AddTrainingSample { features: _, label: _ } => {
                            debug!("FL Participant: Add sample");
                        }
                        FlParticipantMessage::SubmitUpdate { response_tx: _ } => {
                            debug!("FL Participant: Submit update for round {}", self.current_round);
                        }
                        FlParticipantMessage::AbortTraining { reason } => {
                            debug!("FL Participant: Abort training: {}", reason);
                        }
                        FlParticipantMessage::GetStatus { response_tx: _ } => {
                            debug!("FL Participant: Status request");
                        }
                    }
                }
                Some(TaskMessage::Shutdown) => break,
                None => break,
            }
        }
        info!("FL Participant task stopped, round {}", self.current_round);
    }
}
