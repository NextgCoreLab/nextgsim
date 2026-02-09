//! ISAC Task for gNB - Integrated Sensing and Communication

use tokio::sync::mpsc;
use tracing::{debug, info};
use nextgsim_isac::TrackingState;
use crate::tasks::{GnbTaskBase, IsacMessage, Task, TaskMessage};

pub struct IsacTask {
    _task_base: GnbTaskBase,
    trackers: std::collections::HashMap<u64, TrackingState>,
}

impl IsacTask {
    pub fn new(task_base: GnbTaskBase) -> Self {
        Self {
            _task_base: task_base,
            trackers: std::collections::HashMap::new(),
        }
    }
}

#[async_trait::async_trait]
impl Task for IsacTask {
    type Message = IsacMessage;

    async fn run(&mut self, mut rx: mpsc::Receiver<TaskMessage<Self::Message>>) {
        info!("ISAC task started");
        loop {
            match rx.recv().await {
                Some(TaskMessage::Message(msg)) => {
                    match msg {
                        IsacMessage::SensingData { cell_id, measurement_type, measurements: _ } => {
                            debug!("ISAC: Sensing data from cell {} ({})", cell_id, measurement_type);
                        }
                        IsacMessage::FusionRequest { ue_id, source_ids } => {
                            debug!("ISAC: Fusion request for UE {} from {} sources", ue_id, source_ids.len());
                        }
                        IsacMessage::TrackingUpdate { object_id, position, velocity: _ } => {
                            debug!("ISAC: Tracking update for object {}", object_id);
                            let pos = nextgsim_isac::Vector3::new(position.0 as f64, position.1 as f64, position.2 as f64);
                            self.trackers.entry(object_id)
                                .and_modify(|t| t.update(pos, 1.0))
                                .or_insert_with(|| TrackingState::new(object_id, pos));
                        }
                    }
                }
                Some(TaskMessage::Shutdown) => break,
                None => break,
            }
        }
        info!("ISAC task stopped, {} tracked objects", self.trackers.len());
    }
}
