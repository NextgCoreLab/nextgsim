//! ISAC Sensor Task for UE

use tokio::sync::mpsc;
use tracing::{debug, info};
use crate::tasks::{UeTaskBase, IsacSensorMessage, Task, TaskMessage};

pub struct IsacSensorTask {
    _task_base: UeTaskBase,
}

impl IsacSensorTask {
    pub fn new(task_base: UeTaskBase) -> Self {
        Self { _task_base: task_base }
    }
}

#[async_trait::async_trait]
impl Task for IsacSensorTask {
    type Message = IsacSensorMessage;

    async fn run(&mut self, mut rx: mpsc::Receiver<TaskMessage<Self::Message>>) {
        info!("ISAC Sensor task started");
        loop {
            match rx.recv().await {
                Some(TaskMessage::Message(msg)) => {
                    match msg {
                        IsacSensorMessage::StartSensing { config } => {
                            debug!("ISAC Sensor: Start sensing mode={:?}", config.mode);
                        }
                        IsacSensorMessage::StopSensing => {
                            debug!("ISAC Sensor: Stop sensing");
                        }
                        IsacSensorMessage::SensingMeasurement { measurement_type, data: _, timestamp_ms: _ } => {
                            debug!("ISAC Sensor: Measurement {:?}", measurement_type);
                        }
                        IsacSensorMessage::RequestFusedPosition { response_tx: _ } => {
                            debug!("ISAC Sensor: Fused position request");
                        }
                        IsacSensorMessage::PositioningUpdate { position: _, uncertainty, timestamp_ms: _ } => {
                            debug!("ISAC Sensor: Position update uncertainty={}m", uncertainty);
                        }
                    }
                }
                Some(TaskMessage::Shutdown) => break,
                None => break,
            }
        }
        info!("ISAC Sensor task stopped");
    }
}
