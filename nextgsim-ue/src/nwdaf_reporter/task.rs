//! NWDAF Reporter Task for UE

use tokio::sync::mpsc;
use tracing::{debug, info};
use crate::tasks::{UeTaskBase, NwdafReporterMessage, Task, TaskMessage};

pub struct NwdafReporterTask {
    _task_base: UeTaskBase,
}

impl NwdafReporterTask {
    pub fn new(task_base: UeTaskBase) -> Self {
        Self { _task_base: task_base }
    }
}

#[async_trait::async_trait]
impl Task for NwdafReporterTask {
    type Message = NwdafReporterMessage;

    async fn run(&mut self, mut rx: mpsc::Receiver<TaskMessage<Self::Message>>) {
        info!("NWDAF Reporter task started");
        loop {
            match rx.recv().await {
                Some(TaskMessage::Message(msg)) => {
                    match msg {
                        NwdafReporterMessage::ReportMeasurement { rsrp, rsrq, sinr: _, position: _, velocity: _, timestamp_ms: _ } => {
                            debug!("NWDAF Reporter: Measurement RSRP={} RSRQ={}", rsrp, rsrq);
                        }
                        NwdafReporterMessage::ReportNeighborMeasurements { neighbors } => {
                            debug!("NWDAF Reporter: {} neighbor measurements", neighbors.len());
                        }
                        NwdafReporterMessage::RequestTrajectoryPrediction { horizon_ms: _, response_tx: _ } => {
                            debug!("NWDAF Reporter: Trajectory prediction request");
                        }
                        NwdafReporterMessage::HandoverRecommendation { target_cell_id, confidence: _, reason: _ } => {
                            debug!("NWDAF Reporter: HO recommendation to cell {}", target_cell_id);
                        }
                        NwdafReporterMessage::UpdateReportingConfig { interval_ms, report_position: _, report_velocity: _ } => {
                            debug!("NWDAF Reporter: Update config interval={}ms", interval_ms);
                        }
                    }
                }
                Some(TaskMessage::Shutdown) => break,
                None => break,
            }
        }
        info!("NWDAF Reporter task stopped");
    }
}
