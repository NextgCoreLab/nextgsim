//! ISAC Task for gNB - Integrated Sensing and Communication

use crate::tasks::{GnbTaskBase, IsacMessage, NwdafMessage, Task, TaskMessage};
use nextgsim_isac::{
    IsacManager, SensingData, SensingMeasurement, SensingType, TrackingState, Vector3,
};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

pub struct IsacTask {
    task_base: GnbTaskBase,
    engine: IsacManager,
    trackers: std::collections::HashMap<u64, TrackingState>,
}

impl IsacTask {
    pub fn new(task_base: GnbTaskBase) -> Self {
        Self {
            task_base,
            engine: IsacManager::new(50),
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
                        IsacMessage::SensingData {
                            cell_id,
                            measurement_type,
                            measurements,
                        } => {
                            debug!(
                                "ISAC: Sensing data from cell {} ({})",
                                cell_id, measurement_type
                            );
                            let sensing_type = match measurement_type.as_str() {
                                "ToA" => SensingType::ToA,
                                "TDoA" => SensingType::TDoA,
                                "AoA" => SensingType::AoA,
                                "Rss" | "RSS" => SensingType::Rss,
                                "Doppler" => SensingType::Doppler,
                                "Rtt" | "RTT" => SensingType::Rtt,
                                _ => SensingType::Rss,
                            };
                            let timestamp_ms = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|d| d.as_millis() as u64)
                                .unwrap_or(0);
                            let sensing_measurements: Vec<SensingMeasurement> = measurements
                                .iter()
                                .map(|&v| SensingMeasurement {
                                    measurement_type: sensing_type,
                                    anchor_id: cell_id,
                                    value: v as f64,
                                    uncertainty: 1.0,
                                    timestamp_ms,
                                })
                                .collect();
                            let data = SensingData {
                                target_id: cell_id,
                                cell_id,
                                measurements: sensing_measurements,
                                timestamp_ms,
                            };
                            self.engine.record_sensing_data(data);
                        }
                        IsacMessage::FusionRequest { ue_id, source_ids } => {
                            debug!(
                                "ISAC: Fusion request for UE {} from {} sources",
                                ue_id,
                                source_ids.len()
                            );
                            // Register source anchors using configured positions.
                            // source_ids are matched to isac_anchors by index; if the index
                            // exceeds the configured anchor list the anchor is skipped.
                            let anchors = &self.task_base.config.isac_anchors;
                            for (idx, &src) in source_ids.iter().enumerate() {
                                if let Some(&[x, y, z]) = anchors.get(idx) {
                                    self.engine
                                        .register_anchor(src as i32, Vector3::new(x, y, z));
                                } else {
                                    debug!(
                                        "ISAC: No anchor position configured for source index {} (id {}), skipping",
                                        idx, src
                                    );
                                }
                            }
                            if let Some(fused) = self.engine.fuse_position(ue_id) {
                                info!(
                                    "ISAC: Fused position for UE {}: ({:.1}, {:.1}, {:.1}), confidence={:.2}",
                                    ue_id,
                                    fused.position.x,
                                    fused.position.y,
                                    fused.position.z,
                                    fused.confidence
                                );
                                // Forward the fused position to NWDAF for analytics.
                                //
                                // Position only: ISAC fuses ranging measurements
                                // and has no serving-cell RSRP/RSRQ. This used to
                                // send 0.0 for both, which the NWDAF anomaly
                                // detector then treated as a real reading -- a
                                // constant 0 dBm series, which is both physically
                                // implausible and perfectly stable, so the z-score
                                // detector could never flag anything.
                                if let Some(ref sixg) = self.task_base.sixg {
                                    let nwdaf_msg = NwdafMessage::UeMeasurement {
                                        ue_id,
                                        rsrp: None,
                                        rsrq: None,
                                        position: (
                                            fused.position.x as f32,
                                            fused.position.y as f32,
                                            fused.position.z as f32,
                                        ),
                                    };
                                    if let Err(e) = sixg.nwdaf_tx.send(nwdaf_msg).await {
                                        warn!("ISAC: Failed to forward position to NWDAF: {}", e);
                                    }
                                }
                            }
                        }
                        IsacMessage::TrackingUpdate {
                            object_id,
                            position,
                            velocity: _,
                        } => {
                            debug!("ISAC: Tracking update for object {}", object_id);
                            let pos = Vector3::new(
                                position.0 as f64,
                                position.1 as f64,
                                position.2 as f64,
                            );
                            self.trackers
                                .entry(object_id)
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
