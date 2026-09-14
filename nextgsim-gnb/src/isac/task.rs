//! ISAC Task for gNB - Integrated Sensing and Communication

use crate::tasks::{GnbTaskBase, IsacMessage, NwdafMessage, Task, TaskMessage};
#[cfg(feature = "event-bus")]
use nextgsim_common::bus::{BusEvent, Topic};
use nextgsim_isac::{
    IsacManager, SensingData, SensingMeasurement, SensingType, TrackingFilter, Vector3,
};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

pub struct IsacTask {
    task_base: GnbTaskBase,
    engine: IsacManager,
}

impl IsacTask {
    pub fn new(task_base: GnbTaskBase) -> Self {
        // Tracking lives in the engine and nowhere else (issue #27). This task
        // used to keep its own `HashMap<u64, TrackingState>` beside the engine
        // and update that, so the engine's tracking -- and therefore its EKF --
        // had no live caller at all: the better filter existed, was tested, and
        // could not be reached from a running gNB.
        let mut engine = IsacManager::new(50);
        let filter = if task_base.config.isac_ekf_tracking {
            TrackingFilter::Ekf
        } else {
            TrackingFilter::Linear
        };
        engine.set_tracking_filter(filter);
        if let Some(process_noise) = task_base.config.isac_ekf_process_noise_accel {
            engine.set_ekf_process_noise_accel(process_noise);
        }
        info!("ISAC: tracking filter is {filter:?}");

        Self { task_base, engine }
    }

    /// Applies a tracking update through the engine (issue #27).
    ///
    /// A method rather than inline in the message loop so a test can drive the
    /// same code the loop runs: the previous inline version could only be
    /// exercised by starting the task, which is why the duplicate tracker map it
    /// used to write went unnoticed.
    fn handle_tracking_update(&mut self, object_id: u64, position: (f32, f32, f32)) {
        debug!("ISAC: Tracking update for object {}", object_id);
        let pos = Vector3::new(position.0 as f64, position.1 as f64, position.2 as f64);
        // Measurement uncertainty is 1 m because the TrackingUpdate message
        // carries none. Passing the real figure needs the message to grow a
        // field, which no issue has asked for; recorded here rather than left as
        // an unexplained literal.
        self.engine.update_tracking(object_id, pos, 1.0);
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

                            // Publish the raw observation on the topic bus
                            // (issue #16). This arm has no point-to-point
                            // counterpart -- the sensing data goes into the
                            // engine and no other task is told -- so the bus is
                            // additive here in the strongest sense: it makes an
                            // observation available that previously reached
                            // nobody, without this task acquiring a handle to
                            // whoever wants it.
                            #[cfg(feature = "event-bus")]
                            if let Some(bus) =
                                self.task_base.sixg.as_ref().and_then(|s| s.bus.as_ref())
                            {
                                let delivered = bus.publish(BusEvent::new(
                                    Topic::SensingData,
                                    format!("gnb-isac-cell-{cell_id}"),
                                    measurement_type.clone(),
                                    measurements.clone(),
                                ));
                                debug!(
                                    "ISAC: published {} sensing sample(s) from cell {} to {} bus subscriber(s)",
                                    measurements.len(), cell_id, delivered
                                );
                            }
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
                                    // Additionally publish the same observation on
                                    // the topic bus (issue #16). The
                                    // point-to-point send above is unchanged and
                                    // remains the delivery guarantee; the bus is
                                    // how a *second* consumer -- the agent
                                    // framework, say -- gets the fused position
                                    // without this task growing another handle.
                                    #[cfg(feature = "event-bus")]
                                    if let Some(ref bus) = sixg.bus {
                                        let delivered = bus.publish(BusEvent::new(
                                            Topic::SensingData,
                                            "gnb-isac",
                                            "fused-position",
                                            vec![
                                                fused.position.x as f32,
                                                fused.position.y as f32,
                                                fused.position.z as f32,
                                                fused.confidence,
                                            ],
                                        ));
                                        debug!(
                                            "ISAC: published fused position for UE {} to {} bus subscriber(s)",
                                            ue_id, delivered
                                        );
                                    }
                                }
                            }
                        }
                        IsacMessage::TrackingUpdate {
                            object_id,
                            position,
                            velocity: _,
                        } => {
                            self.handle_tracking_update(object_id, position);
                        }
                    }
                }
                Some(TaskMessage::Shutdown) => break,
                None => break,
            }
        }
        info!(
            "ISAC task stopped, {} tracked objects",
            self.engine.active_track_count()
        );
    }
}

// ============================================================================
// Topic-bus reference path (issue #16)
// ============================================================================

#[cfg(all(test, feature = "event-bus"))]
mod bus_tests {
    use super::*;
    use nextgsim_common::config::GnbConfig;

    /// The reference end-to-end path #16's criterion 4 asks for: an observation
    /// the gNB ISAC task publishes reaches a bus subscriber, which is exactly how
    /// the NWDAF task subscribes at the top of its run loop.
    ///
    /// A clone of the bus is held for the duration, because the only other sender
    /// lives inside the `GnbTaskBase` that moves into the task -- when the task
    /// ends, that sender drops and an un-cloned receiver reports `Closed` before
    /// it can be read. The NWDAF task does not have this problem, since its own
    /// `task_base` keeps a sender alive for as long as it is subscribed.
    #[test]
    fn a_sensing_observation_reaches_a_bus_subscriber() {
        let (mut task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(GnbConfig::default(), 32);
        let _sixg_rx = task_base.init_6g_tasks(32);

        let bus = task_base
            .sixg
            .as_ref()
            .and_then(|sixg| sixg.bus.as_ref())
            .expect("the bus exists with the feature on")
            .clone();
        let mut bus_rx = bus.subscribe();

        let (isac_tx, isac_rx) = mpsc::channel(32);
        let mut task = IsacTask::new(task_base);

        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime")
            .block_on(async move {
                isac_tx
                    .send(TaskMessage::Message(IsacMessage::SensingData {
                        cell_id: 4,
                        measurement_type: "Rtt".to_string(),
                        measurements: vec![50.0, 55.5],
                    }))
                    .await
                    .expect("send sensing data");
                isac_tx.send(TaskMessage::Shutdown).await.expect("shutdown");
                task.run(isac_rx).await;
            });

        let event = bus_rx
            .try_recv()
            .expect("the bus subscriber receives the ISAC observation");
        assert_eq!(event.topic, Topic::SensingData);
        assert_eq!(event.source, "gnb-isac-cell-4");
        assert_eq!(event.measurement_type, "Rtt");
        assert_eq!(event.measurements, vec![50.0, 55.5]);
    }

    /// The bus is additive: with nobody subscribed the ISAC task runs exactly as
    /// it did, and publishing is a no-op rather than an error. This is the half
    /// that keeps a default build honest -- a producer must not depend on a
    /// consumer existing.
    #[test]
    fn publishing_with_no_subscriber_does_not_disturb_the_isac_task() {
        let (mut task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(GnbConfig::default(), 32);
        let _sixg_rx = task_base.init_6g_tasks(32);
        let bus = task_base
            .sixg
            .as_ref()
            .and_then(|sixg| sixg.bus.as_ref())
            .expect("bus")
            .clone();
        assert_eq!(bus.subscriber_count(), 0);

        let (isac_tx, isac_rx) = mpsc::channel(32);
        let mut task = IsacTask::new(task_base);

        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime")
            .block_on(async move {
                for sample in [1.0f32, 2.0, 3.0] {
                    isac_tx
                        .send(TaskMessage::Message(IsacMessage::SensingData {
                            cell_id: 1,
                            measurement_type: "Rss".to_string(),
                            measurements: vec![sample],
                        }))
                        .await
                        .expect("send");
                }
                isac_tx.send(TaskMessage::Shutdown).await.expect("shutdown");
                // The task completing at all is the assertion: a publish with no
                // subscriber must not error out of the loop.
                task.run(isac_rx).await;
            });

        assert_eq!(bus.subscriber_count(), 0, "still nobody listening");
    }

    /// Several consumers, one producer, no change to the producer -- which is the
    /// ergonomics problem #16 exists to fix. With point-to-point channels this
    /// would need a second `TaskHandle` and a second message variant.
    #[test]
    fn two_consumers_both_receive_one_isac_observation() {
        let (mut task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(GnbConfig::default(), 32);
        let _sixg_rx = task_base.init_6g_tasks(32);
        let bus = task_base
            .sixg
            .as_ref()
            .and_then(|sixg| sixg.bus.as_ref())
            .expect("bus")
            .clone();
        let mut analytics = bus.subscribe();
        let mut agent = bus.subscribe();

        let (isac_tx, isac_rx) = mpsc::channel(32);
        let mut task = IsacTask::new(task_base);

        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime")
            .block_on(async move {
                isac_tx
                    .send(TaskMessage::Message(IsacMessage::SensingData {
                        cell_id: 9,
                        measurement_type: "AoA".to_string(),
                        measurements: vec![42.0],
                    }))
                    .await
                    .expect("send");
                isac_tx.send(TaskMessage::Shutdown).await.expect("shutdown");
                task.run(isac_rx).await;
            });

        for (label, rx) in [("analytics", &mut analytics), ("agent", &mut agent)] {
            let event = rx
                .try_recv()
                .unwrap_or_else(|e| panic!("{label} must receive it, got {e:?}"));
            assert_eq!(event.measurements, vec![42.0]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::DEFAULT_CHANNEL_CAPACITY;
    use nextgsim_common::config::GnbConfig;

    fn task_for(config: GnbConfig) -> IsacTask {
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);
        IsacTask::new(task_base)
    }

    #[test]
    fn the_gnb_isac_task_selects_the_ekf_when_configured() {
        // Default: unchanged behaviour.
        let default_task = task_for(GnbConfig::default());
        assert_eq!(
            default_task.engine.tracking_filter(),
            TrackingFilter::Linear,
            "the default build must keep the smoother"
        );

        let mut config = GnbConfig::default();
        config.isac_ekf_tracking = true;
        let ekf_task = task_for(config);
        assert_eq!(ekf_task.engine.tracking_filter(), TrackingFilter::Ekf);
    }

    #[test]
    fn the_gnb_isac_task_tracks_through_the_engine() {
        // This task used to keep its own tracker map beside the engine, which is
        // why the engine's EKF had no live caller. The engine must be the only
        // place a track lives.
        let mut task = task_for(GnbConfig::default());
        assert_eq!(task.engine.active_track_count(), 0);

        // Driven through the same handler the message loop calls, so this cannot
        // pass while the loop writes somewhere else.
        task.handle_tracking_update(42, (1.0, 2.0, 3.0));

        assert_eq!(
            task.engine.active_track_count(),
            1,
            "the track must live in the engine"
        );
        let track = task.engine.get_tracking(42).expect("the track exists");
        assert!(track.position.distance_to(&Vector3::new(1.0, 2.0, 3.0)) < 1e-9);
    }

    #[test]
    fn a_configured_process_noise_reaches_the_engine() {
        // An EKF whose process noise never left the config would be tuned by
        // nothing, which is indistinguishable from having no knob at all.
        let mut config = GnbConfig::default();
        config.isac_ekf_tracking = true;
        config.isac_ekf_process_noise_accel = Some(0.05);
        let mut task = task_for(config);

        // Drive one update so the EKF is created, then read the value back off it.
        task.engine
            .update_tracking(1, Vector3::new(0.0, 0.0, 0.0), 1.0);
        let ekf = task.engine.get_ekf(1).expect("an EKF was created");
        assert!(
            (ekf.process_noise_accel - 0.05).abs() < f64::EPSILON,
            "configured process noise must reach the filter, got {}",
            ekf.process_noise_accel
        );
    }
}
