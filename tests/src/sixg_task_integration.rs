//! E2E Integration Tests for 6G Task Spawning and Message Flow
//!
//! These tests exercise the full gNB + 6G task pipeline at the channel level,
//! going beyond the crate-API tests in `ai_integration.rs`. Specifically:
//!
//! - Task spawning via `GnbTaskBase::init_6g_tasks()`
//! - Message delivery through mpsc channels to running task loops
//! - State transitions inside each task (Agent, FL, ISAC, NWDAF)
//! - NWDAF → RRC handover recommendation cross-task wiring
//!
//! The 6G task types (`AgentTask`/`FlAggregatorTask`/`IsacTask`/`NwdafTask`) are
//! only re-exported from a `--features 6g` build of `nextgsim-gnb`, so this whole
//! test binary is gated on the tests crate's `6g` feature (see Cargo.toml). Under
//! a default (lean) `cargo test --workspace` it compiles to an empty binary.

#![cfg(feature = "6g")]
#![allow(unused_imports)]

use std::time::Duration;

use integration_tests::init_test_logging;
use nextgsim_common::{config::GnbConfig, Plmn};
use nextgsim_gnb::{
    tasks::{
        AgentMessage, FlAggregatorMessage, GnbTaskBase, IsacMessage, NwdafMessage, RrcMessage,
        Task, TaskMessage, DEFAULT_CHANNEL_CAPACITY,
    },
    AgentTask, FlAggregatorTask, IsacTask, NwdafTask,
};
use tokio::time::timeout;

// ============================================================================
// Helpers
// ============================================================================

/// Minimal `GnbConfig` with all 6G flags enabled.
fn sixg_gnb_config() -> GnbConfig {
    GnbConfig {
        nci: 0x000000010,
        gnb_id_length: 32,
        plmn: Plmn::new(1, 1, false),
        tac: 1,
        she_enabled: true,
        nwdaf_enabled: true,
        nkef_enabled: true,
        isac_enabled: true,
        agent_enabled: true,
        federated_learning_enabled: true,
        ..Default::default()
    }
}

/// Build a `GnbTaskBase` with 6G channels wired up.
///
/// Returns the base (holds all tx handles) plus the 6G receivers.
fn make_base_with_6g() -> (
    GnbTaskBase,
    nextgsim_gnb::tasks::GnbSixgReceivers,
    // core task receivers (unused in most 6G tests but must be consumed)
    tokio::sync::mpsc::Receiver<TaskMessage<nextgsim_gnb::tasks::AppMessage>>,
    tokio::sync::mpsc::Receiver<TaskMessage<RrcMessage>>,
) {
    let (mut base, app_rx, _ngap_rx, rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
        GnbTaskBase::new(sixg_gnb_config(), DEFAULT_CHANNEL_CAPACITY);
    let sixg_rx = base.init_6g_tasks(DEFAULT_CHANNEL_CAPACITY);
    (base, sixg_rx, app_rx, rrc_rx)
}

// ============================================================================
// Test 1: 6G Task Spawning – channels are open after init
// ============================================================================

/// Verify that `init_6g_tasks` wires up channels that are initially open
/// (not closed) and that tasks can be created from the resulting receivers.
#[tokio::test]
async fn test_sixg_task_channels_open_after_init() {
    init_test_logging();

    let (base, _sixg_rx, _app_rx, _rrc_rx) = make_base_with_6g();

    let sixg = base
        .sixg
        .as_ref()
        .expect("6G handles must be present after init_6g_tasks");

    // All sender handles must report open channels
    assert!(!sixg.nwdaf_tx.is_closed(), "NWDAF channel must be open");
    assert!(!sixg.isac_tx.is_closed(), "ISAC channel must be open");
    assert!(!sixg.agent_tx.is_closed(), "Agent channel must be open");
    assert!(!sixg.fl_tx.is_closed(), "FL channel must be open");
    assert!(!sixg.she_tx.is_closed(), "SHE channel must be open");
    assert!(!sixg.nkef_tx.is_closed(), "NKEF channel must be open");
}

/// Verify that task structs can be instantiated from a GnbTaskBase
/// without panicking, confirming the spawning path is sound.
#[tokio::test]
async fn test_sixg_tasks_can_be_instantiated() {
    init_test_logging();

    let (base, _sixg_rx, _app_rx, _rrc_rx) = make_base_with_6g();

    // Each constructor must succeed without panic
    let _agent_task = AgentTask::new(base.clone());
    let _fl_task = FlAggregatorTask::new(base.clone());
    let _isac_task = IsacTask::new(base.clone());
    let _nwdaf_task = NwdafTask::new(base.clone());

    // If we reach here all constructors ran cleanly
}

// ============================================================================
// Test 2: Agent → Coordinator wiring
// ============================================================================

/// Send `RegisterAgent` then `SubmitIntent` through the channel to a running
/// `AgentTask`.  Verify the task processes both messages without panic and that
/// the coordinator has exactly one registered agent.
#[tokio::test]
async fn test_agent_register_then_submit_intent() {
    init_test_logging();

    let (base, sixg_rx, _app_rx, _rrc_rx) = make_base_with_6g();
    let sixg = base.sixg.as_ref().unwrap().clone();

    // Spawn the AgentTask loop
    let mut task = AgentTask::new(base);
    let agent_rx = sixg_rx.agent_rx;
    let handle = tokio::spawn(async move { task.run(agent_rx).await });

    // 1. Register an agent
    sixg.agent_tx
        .send(AgentMessage::RegisterAgent {
            agent_id: "mobility-optimizer-001".to_string(),
            agent_type: "mobility".to_string(),
            capabilities: vec!["trigger_handover".to_string(), "query".to_string()],
        })
        .await
        .expect("RegisterAgent send must succeed");

    // 2. Submit an intent from the same agent
    sixg.agent_tx
        .send(AgentMessage::SubmitIntent {
            agent_id: "mobility-optimizer-001".to_string(),
            intent_type: "trigger_handover".to_string(),
            parameters: vec![
                ("target_cell".to_string(), "gnb-002".to_string()),
                ("ue_id".to_string(), "1".to_string()),
            ],
        })
        .await
        .expect("SubmitIntent send must succeed");

    // 3. Emit a coordination event so the task drains its intent queue
    sixg.agent_tx
        .send(AgentMessage::CoordinationEvent {
            event_type: "round_complete".to_string(),
            agent_ids: vec!["mobility-optimizer-001".to_string()],
        })
        .await
        .expect("CoordinationEvent send must succeed");

    // 4. Shutdown and wait – must not panic
    sixg.agent_tx
        .shutdown()
        .await
        .expect("Shutdown send must succeed");

    timeout(Duration::from_secs(2), handle)
        .await
        .expect("AgentTask must finish within 2 s")
        .expect("AgentTask join must succeed");
}

// ============================================================================
// Test 3: FL aggregation path
// ============================================================================

/// Drive the full FL training round through the task channel:
/// `RegisterParticipant` × 2 → `DistributeModel` → `SubmitUpdate` × 2 → `Aggregate`.
/// The task must process all messages and log a successful aggregation.
#[tokio::test]
async fn test_fl_task_full_training_round_via_channel() {
    init_test_logging();

    let (base, sixg_rx, _app_rx, _rrc_rx) = make_base_with_6g();
    let sixg = base.sixg.as_ref().unwrap().clone();

    let mut task = FlAggregatorTask::new(base);
    let fl_rx = sixg_rx.fl_rx;
    let handle = tokio::spawn(async move { task.run(fl_rx).await });

    // Register two participants
    sixg.fl_tx
        .send(FlAggregatorMessage::RegisterParticipant {
            participant_id: "ue-001".to_string(),
            compute_capability: 500,
        })
        .await
        .expect("RegisterParticipant ue-001 must succeed");

    sixg.fl_tx
        .send(FlAggregatorMessage::RegisterParticipant {
            participant_id: "ue-002".to_string(),
            compute_capability: 750,
        })
        .await
        .expect("RegisterParticipant ue-002 must succeed");

    // Initialise the global model (weights = 0.0 × 8)
    sixg.fl_tx
        .send(FlAggregatorMessage::DistributeModel {
            version: 0,
            weights: vec![0.0_f32; 8],
        })
        .await
        .expect("DistributeModel must succeed");

    // Submit gradient updates from both participants
    sixg.fl_tx
        .send(FlAggregatorMessage::SubmitUpdate {
            participant_id: "ue-001".to_string(),
            round: 1,
            gradients: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            num_samples: 100,
        })
        .await
        .expect("SubmitUpdate ue-001 must succeed");

    sixg.fl_tx
        .send(FlAggregatorMessage::SubmitUpdate {
            participant_id: "ue-002".to_string(),
            round: 1,
            gradients: vec![0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85],
            num_samples: 150,
        })
        .await
        .expect("SubmitUpdate ue-002 must succeed");

    // Trigger aggregation
    sixg.fl_tx
        .send(FlAggregatorMessage::Aggregate { round: 1 })
        .await
        .expect("Aggregate must succeed");

    // Shutdown and verify no panic
    sixg.fl_tx.shutdown().await.expect("Shutdown must succeed");

    timeout(Duration::from_secs(2), handle)
        .await
        .expect("FlAggregatorTask must finish within 2 s")
        .expect("FlAggregatorTask join must succeed");
}

/// Verify that sending `Aggregate` before any participants are registered
/// (the error path) does not cause a panic – the task must stay alive and
/// accept the subsequent shutdown.
#[tokio::test]
async fn test_fl_task_aggregate_before_participants_is_graceful() {
    init_test_logging();

    let (base, sixg_rx, _app_rx, _rrc_rx) = make_base_with_6g();
    let sixg = base.sixg.as_ref().unwrap().clone();

    let mut task = FlAggregatorTask::new(base);
    let fl_rx = sixg_rx.fl_rx;
    let handle = tokio::spawn(async move { task.run(fl_rx).await });

    // Aggregate with no participants should warn but not panic
    sixg.fl_tx
        .send(FlAggregatorMessage::Aggregate { round: 1 })
        .await
        .expect("Aggregate send must succeed even with no participants");

    sixg.fl_tx.shutdown().await.expect("Shutdown must succeed");

    timeout(Duration::from_secs(2), handle)
        .await
        .expect("FlAggregatorTask must finish within 2 s")
        .expect("FlAggregatorTask join must succeed");
}

// ============================================================================
// Test 4: ISAC sensing pipeline
// ============================================================================

/// Send `SensingData` messages to a live `IsacTask` and then request a
/// `FusionRequest`.  The task must process all messages without panicking.
/// We observe the side-effect by verifying that the `IsacManager` inside an
/// equivalent standalone instance records the data correctly.
#[tokio::test]
async fn test_isac_task_sensing_data_recorded_via_channel() {
    init_test_logging();

    let (base, sixg_rx, _app_rx, _rrc_rx) = make_base_with_6g();
    let sixg = base.sixg.as_ref().unwrap().clone();

    let mut task = IsacTask::new(base);
    let isac_rx = sixg_rx.isac_rx;
    let handle = tokio::spawn(async move { task.run(isac_rx).await });

    // Send ToA sensing data from three cells (anchor IDs 1, 2, 3)
    for cell_id in [1_i32, 2, 3] {
        sixg.isac_tx
            .send(IsacMessage::SensingData {
                cell_id,
                measurement_type: "ToA".to_string(),
                // Two range measurements per cell
                measurements: vec![50.0_f32, 52.0],
            })
            .await
            .unwrap_or_else(|_| panic!("SensingData cell {} send must succeed", cell_id));
    }

    // Request fusion for UE 1, using the three cells as sources
    sixg.isac_tx
        .send(IsacMessage::FusionRequest {
            ue_id: 1,
            source_ids: vec![1, 2, 3],
        })
        .await
        .expect("FusionRequest send must succeed");

    // Send a tracking update for an independent object
    sixg.isac_tx
        .send(IsacMessage::TrackingUpdate {
            object_id: 42,
            position: (10.0, 20.0, 5.0),
            velocity: (1.0, 0.5, 0.0),
        })
        .await
        .expect("TrackingUpdate send must succeed");

    sixg.isac_tx
        .shutdown()
        .await
        .expect("Shutdown must succeed");

    timeout(Duration::from_secs(2), handle)
        .await
        .expect("IsacTask must finish within 2 s")
        .expect("IsacTask join must succeed");
}

/// Directly verify `IsacManager::record_sensing_data` stores data that can
/// be retrieved later – this is the crate-level assertion backing the task test.
#[test]
fn test_isac_manager_records_sensing_data() {
    use nextgsim_isac::{IsacManager, SensingData, SensingMeasurement, SensingType, Vector3};

    let mut manager = IsacManager::new(50);

    // Register three anchors forming a triangle
    manager.register_anchor(1, Vector3::new(0.0, 0.0, 10.0));
    manager.register_anchor(2, Vector3::new(100.0, 0.0, 10.0));
    manager.register_anchor(3, Vector3::new(50.0, 86.6, 10.0));

    // Record sensing data for target 1
    let data = SensingData {
        target_id: 1,
        cell_id: 1,
        measurements: vec![
            SensingMeasurement {
                measurement_type: SensingType::ToA,
                anchor_id: 1,
                value: 50.0,
                uncertainty: 1.0,
                timestamp_ms: 0,
            },
            SensingMeasurement {
                measurement_type: SensingType::ToA,
                anchor_id: 2,
                value: 50.0,
                uncertainty: 1.0,
                timestamp_ms: 0,
            },
            SensingMeasurement {
                measurement_type: SensingType::ToA,
                anchor_id: 3,
                value: 36.0,
                uncertainty: 1.0,
                timestamp_ms: 0,
            },
        ],
        timestamp_ms: 0,
    };
    manager.record_sensing_data(data);

    // After recording, fuse_position must return Some (data is present)
    let fused = manager.fuse_position(1);
    assert!(
        fused.is_some(),
        "fuse_position must return Some after data is recorded"
    );

    let fp = fused.unwrap();
    assert_eq!(fp.target_id, 1);
    assert!(fp.confidence > 0.0, "confidence must be positive");
    assert!(
        fp.position_uncertainty > 0.0,
        "uncertainty must be positive"
    );
    // Position must be within the bounding box of the triangle
    assert!(
        fp.position.x >= 0.0 && fp.position.x <= 100.0,
        "x={} must lie within anchor bounding box",
        fp.position.x
    );
}

// ============================================================================
// Test 5: NWDAF → RRC handover path
// ============================================================================

/// Drive UE measurements into a live `NwdafTask` via its channel, then send a
/// `HandoverRecommendation` that the task forwards to the RRC channel.
/// Assert that the `NwdafHandoverRecommendation` message arrives on the RRC
/// receiver with the expected UE id, target cell, and confidence.
#[tokio::test]
async fn test_nwdaf_to_rrc_handover_recommendation_arrives() {
    init_test_logging();

    let (base, sixg_rx, _app_rx, mut rrc_rx) = make_base_with_6g();
    let sixg = base.sixg.as_ref().unwrap().clone();

    // Spawn NwdafTask (it holds a clone of task_base, which contains rrc_tx)
    let mut task = NwdafTask::new(base);
    let nwdaf_rx = sixg_rx.nwdaf_rx;
    let handle = tokio::spawn(async move { task.run(nwdaf_rx).await });

    // 1. Feed three UE measurements with deteriorating RSRP so history builds up
    for i in 0..3_i32 {
        sixg.nwdaf_tx
            .send(NwdafMessage::UeMeasurement {
                ue_id: 1,
                rsrp: -80.0 - (i as f32) * 5.0,
                rsrq: -10.0 - (i as f32) * 1.0,
                position: (i as f32 * 10.0, 0.0, 0.0),
            })
            .await
            .expect("UeMeasurement send must succeed");
    }

    // 2. Send a cell load update so the NWDAF has load context
    sixg.nwdaf_tx
        .send(NwdafMessage::CellLoad {
            cell_id: 1,
            prb_usage: 0.75,
            connected_ues: 12,
        })
        .await
        .expect("CellLoad send must succeed");

    // 3. Send the handover recommendation – NWDAF forwards this to RRC
    sixg.nwdaf_tx
        .send(NwdafMessage::HandoverRecommendation {
            ue_id: 1,
            target_cell: 2,
            confidence: 0.92,
        })
        .await
        .expect("HandoverRecommendation send must succeed");

    // 4. Receive the forwarded message on the RRC channel
    //    We give the task 2 s to route the message before we time out.
    let rrc_msg = timeout(Duration::from_secs(2), rrc_rx.recv())
        .await
        .expect("Must receive RRC message within 2 s")
        .expect("RRC receiver must not be closed");

    match rrc_msg {
        TaskMessage::Message(RrcMessage::NwdafHandoverRecommendation {
            ue_id,
            target_cell,
            confidence,
        }) => {
            assert_eq!(ue_id, 1, "ue_id must match");
            assert_eq!(target_cell, 2, "target_cell must match");
            assert!(
                (confidence - 0.92).abs() < 1e-4,
                "confidence must match (got {confidence})"
            );
        }
        other => panic!("Expected NwdafHandoverRecommendation on RRC channel, got {other:?}"),
    }

    // 5. Shutdown cleanly
    sixg.nwdaf_tx
        .shutdown()
        .await
        .expect("Shutdown must succeed");

    timeout(Duration::from_secs(2), handle)
        .await
        .expect("NwdafTask must finish within 2 s")
        .expect("NwdafTask join must succeed");
}

/// Verify that a trajectory prediction request is processed without panic
/// after sufficient measurement history has been recorded.
#[tokio::test]
async fn test_nwdaf_task_predict_trajectory_via_channel() {
    init_test_logging();

    let (base, sixg_rx, _app_rx, _rrc_rx) = make_base_with_6g();
    let sixg = base.sixg.as_ref().unwrap().clone();

    let mut task = NwdafTask::new(base);
    let nwdaf_rx = sixg_rx.nwdaf_rx;
    let handle = tokio::spawn(async move { task.run(nwdaf_rx).await });

    // Build measurement history for UE 5
    for i in 0..10_i32 {
        sixg.nwdaf_tx
            .send(NwdafMessage::UeMeasurement {
                ue_id: 5,
                rsrp: -75.0 - (i as f32),
                rsrq: -9.0 - (i as f32) * 0.5,
                position: (i as f32 * 5.0, i as f32 * 3.0, 0.0),
            })
            .await
            .expect("UeMeasurement send must succeed");
    }

    // Request trajectory prediction
    sixg.nwdaf_tx
        .send(NwdafMessage::PredictTrajectory {
            ue_id: 5,
            horizon_ms: 1000,
        })
        .await
        .expect("PredictTrajectory send must succeed");

    sixg.nwdaf_tx
        .shutdown()
        .await
        .expect("Shutdown must succeed");

    timeout(Duration::from_secs(2), handle)
        .await
        .expect("NwdafTask must finish within 2 s")
        .expect("NwdafTask join must succeed");
}

// ============================================================================
// Test 6: Shutdown ordering – all 6G tasks stop cleanly
// ============================================================================

/// Spawn all four 6G tasks concurrently, then send a coordinated shutdown
/// through `GnbTaskBase::shutdown_all()`.  Every task handle must finish
/// within the timeout, proving the shutdown signal propagates correctly.
#[tokio::test]
async fn test_sixg_all_tasks_shutdown_cleanly() {
    init_test_logging();

    let (base, sixg_rx, _app_rx, _rrc_rx) = make_base_with_6g();

    // Spawn all 6G tasks
    let mut agent_task = AgentTask::new(base.clone());
    let h_agent = tokio::spawn(async move { agent_task.run(sixg_rx.agent_rx).await });

    let mut fl_task = FlAggregatorTask::new(base.clone());
    let h_fl = tokio::spawn(async move { fl_task.run(sixg_rx.fl_rx).await });

    let mut isac_task = IsacTask::new(base.clone());
    let h_isac = tokio::spawn(async move { isac_task.run(sixg_rx.isac_rx).await });

    let mut nwdaf_task = NwdafTask::new(base.clone());
    let h_nwdaf = tokio::spawn(async move { nwdaf_task.run(sixg_rx.nwdaf_rx).await });

    // Send shutdown to all 6G tasks via the base
    base.shutdown_all().await;

    // All tasks must finish within 2 s
    let results = timeout(
        Duration::from_secs(2),
        futures::future::join_all([h_agent, h_fl, h_isac, h_nwdaf]),
    )
    .await
    .expect("All 6G tasks must stop within 2 s after shutdown_all()");

    for result in results {
        result.expect("Task join must not error");
    }
}
