//! NWDAF Reporter Task for UE

use crate::tasks::{NwdafReporterMessage, Task, TaskMessage, UeTaskBase};
use tokio::sync::mpsc;
use tracing::{debug, info};

pub struct NwdafReporterTask {
    _task_base: UeTaskBase,
    /// Serving-cell measurements received from the RRC task (#459).
    measurements_received: u64,
    /// The latest of them: (RSRP dBm, RSRQ dB if measured, timestamp ms).
    latest_measurement: Option<(f32, Option<f32>, u64)>,
}

impl NwdafReporterTask {
    pub fn new(task_base: UeTaskBase) -> Self {
        Self {
            _task_base: task_base,
            measurements_received: 0,
            latest_measurement: None,
        }
    }

    /// Serving-cell measurements received so far.
    pub fn measurements_received(&self) -> u64 {
        self.measurements_received
    }

    /// The latest serving-cell measurement: (RSRP dBm, RSRQ dB, timestamp ms).
    pub fn latest_measurement(&self) -> Option<(f32, Option<f32>, u64)> {
        self.latest_measurement
    }
}

#[async_trait::async_trait]
impl Task for NwdafReporterTask {
    type Message = NwdafReporterMessage;

    async fn run(&mut self, mut rx: mpsc::Receiver<TaskMessage<Self::Message>>) {
        info!("NWDAF Reporter task started");
        loop {
            match rx.recv().await {
                Some(TaskMessage::Message(msg)) => match msg {
                    NwdafReporterMessage::ReportMeasurement {
                        rsrp,
                        rsrq,
                        sinr: _,
                        position: _,
                        velocity: _,
                        timestamp_ms,
                    } => {
                        if self.measurements_received == 0 {
                            info!(
                                "NWDAF Reporter: first serving-cell measurement, RSRP={} dBm",
                                rsrp
                            );
                        }
                        debug!("NWDAF Reporter: Measurement RSRP={} RSRQ={:?}", rsrp, rsrq);
                        self.measurements_received += 1;
                        self.latest_measurement = Some((rsrp, rsrq, timestamp_ms));
                    }
                    NwdafReporterMessage::ReportNeighborMeasurements { neighbors } => {
                        debug!("NWDAF Reporter: {} neighbor measurements", neighbors.len());
                    }
                    NwdafReporterMessage::RequestTrajectoryPrediction {
                        horizon_ms: _,
                        response_tx: _,
                    } => {
                        debug!("NWDAF Reporter: Trajectory prediction request");
                    }
                    NwdafReporterMessage::HandoverRecommendation {
                        target_cell_id,
                        confidence: _,
                        reason: _,
                    } => {
                        debug!(
                            "NWDAF Reporter: HO recommendation to cell {}",
                            target_cell_id
                        );
                    }
                    NwdafReporterMessage::UpdateReportingConfig {
                        interval_ms,
                        report_position: _,
                        report_velocity: _,
                    } => {
                        debug!("NWDAF Reporter: Update config interval={}ms", interval_ms);
                    }
                },
                Some(TaskMessage::Shutdown) => break,
                None => break,
            }
        }
        info!("NWDAF Reporter task stopped");
    }
}
