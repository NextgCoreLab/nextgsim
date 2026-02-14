//! Cross-crate integration tests for 6G features
//!
//! This module tests integration between different 6G crates:
//! - SHE <-> ISAC: Sensing inference on edge
//! - SHE <-> Semantic: Codec on edge
//! - FL <-> Semantic: Distributed codec training
//! - ISAC <-> Semantic: Sensing data compression

#![allow(unused_imports)]

use nextgsim_isac::{SheIsacClient, SensingData, SensingMeasurement, SensingType, SensingProcessingType};
use nextgsim_she::{TierManager, ComputeNode, ComputeTier, WorkloadRequirements};
use nextgsim_she::workload::WorkloadType;
use nextgsim_semantic::{SheSemanticClient, SemanticCodingOperation, SemanticTask};

// ============================================================================
// A19.2: SHE <-> ISAC Integration Tests
// ============================================================================

#[test]
fn test_she_isac_sensing_workload_submission() {
    // Create ISAC client for submitting to SHE
    let mut isac_client = SheIsacClient::new();

    // Create sensing data
    let sensing_data = SensingData {
        target_id: 1,
        cell_id: 100,
        measurements: vec![
            SensingMeasurement {
                measurement_type: SensingType::ToA,
                anchor_id: 1,
                value: 50.0,
                uncertainty: 5.0,
                timestamp_ms: 1000,
            },
            SensingMeasurement {
                measurement_type: SensingType::ToA,
                anchor_id: 2,
                value: 60.0,
                uncertainty: 5.0,
                timestamp_ms: 1000,
            },
            SensingMeasurement {
                measurement_type: SensingType::ToA,
                anchor_id: 3,
                value: 45.0,
                uncertainty: 5.0,
                timestamp_ms: 1000,
            },
        ],
        timestamp_ms: 1000,
    };

    // Submit position fusion workload to SHE
    let request = isac_client.submit_position_fusion(sensing_data.clone());

    assert_eq!(request.processing_type, SensingProcessingType::PositionFusion);
    assert_eq!(request.latency_requirement_ms, 10);
    assert_eq!(request.priority, 7);
    assert_eq!(request.sensing_data.target_id, 1);
    assert_eq!(request.sensing_data.measurements.len(), 3);
}

#[test]
fn test_she_sensing_workload_placement() {
    // Create SHE tier manager
    let mut tier_manager = TierManager::new();

    // Add edge nodes for sensing workloads
    tier_manager.add_node(ComputeNode::new(
        1,
        "edge-sensing-1",
        ComputeTier::LocalEdge,
        nextgsim_she::resource::ResourceCapacity::with_tflops(5).with_memory_gb(16),
    ));

    tier_manager.add_node(ComputeNode::new(
        2,
        "regional-sensing-1",
        ComputeTier::RegionalEdge,
        nextgsim_she::resource::ResourceCapacity::with_tflops(20).with_memory_gb(64),
    ));

    // Create scheduler
    let mut scheduler = nextgsim_she::scheduler::WorkloadScheduler::new(tier_manager);

    // Submit sensing processing workload
    let sensing_req = WorkloadRequirements::sensing_processing();
    let id = scheduler.submit(sensing_req).unwrap();
    let decision = scheduler.place(id).unwrap();

    // Should be placed at local edge for low latency
    assert_eq!(decision.tier, ComputeTier::LocalEdge);
}

#[test]
fn test_she_multiple_sensing_workloads() {
    let mut tier_manager = TierManager::new();
    tier_manager.add_node(ComputeNode::new(
        1,
        "edge-1",
        ComputeTier::LocalEdge,
        nextgsim_she::resource::ResourceCapacity::with_tflops(10).with_memory_gb(32),
    ));

    let mut scheduler = nextgsim_she::scheduler::WorkloadScheduler::new(tier_manager);

    // Submit multiple sensing workloads
    let req1 = WorkloadRequirements::sensing_processing().with_priority(8);
    let req2 = WorkloadRequirements::sensing_processing().with_priority(6);
    let req3 = WorkloadRequirements::sensing_processing().with_priority(7);

    let id1 = scheduler.submit(req1).unwrap();
    let id2 = scheduler.submit(req2).unwrap();
    let id3 = scheduler.submit(req3).unwrap();

    // All should be placeable
    let decision1 = scheduler.place(id1);
    let decision2 = scheduler.place(id2);
    let decision3 = scheduler.place(id3);

    assert!(decision1.is_ok());
    assert!(decision2.is_ok());
    assert!(decision3.is_ok());
}

#[test]
fn test_isac_object_tracking_workload() {
    let mut isac_client = SheIsacClient::new();

    let sensing_data = SensingData {
        target_id: 2,
        cell_id: 200,
        measurements: vec![
            SensingMeasurement {
                measurement_type: SensingType::Doppler,
                anchor_id: 1,
                value: 100.0, // 100 Hz Doppler shift
                uncertainty: 5.0,
                timestamp_ms: 2000,
            },
            SensingMeasurement {
                measurement_type: SensingType::ToA,
                anchor_id: 1,
                value: 30.0,
                uncertainty: 3.0,
                timestamp_ms: 2000,
            },
        ],
        timestamp_ms: 2000,
    };

    // Submit object tracking workload
    let request = isac_client.submit_object_tracking(sensing_data);

    assert_eq!(request.processing_type, SensingProcessingType::ObjectTracking);
    assert_eq!(request.latency_requirement_ms, 15);
    assert_eq!(request.priority, 6);
}

#[test]
fn test_isac_ml_positioning_workload() {
    let mut isac_client = SheIsacClient::new();

    let sensing_data = SensingData {
        target_id: 3,
        cell_id: 300,
        measurements: vec![
            SensingMeasurement {
                measurement_type: SensingType::Rss,
                anchor_id: 1,
                value: -80.0,
                uncertainty: 2.0,
                timestamp_ms: 3000,
            },
            SensingMeasurement {
                measurement_type: SensingType::Rss,
                anchor_id: 2,
                value: -85.0,
                uncertainty: 2.0,
                timestamp_ms: 3000,
            },
            SensingMeasurement {
                measurement_type: SensingType::Rss,
                anchor_id: 3,
                value: -75.0,
                uncertainty: 2.0,
                timestamp_ms: 3000,
            },
        ],
        timestamp_ms: 3000,
    };

    // Submit ML positioning workload
    let request = isac_client.submit_ml_positioning(sensing_data);

    assert_eq!(request.processing_type, SensingProcessingType::MlPositioning);
    assert_eq!(request.latency_requirement_ms, 20);
    assert_eq!(request.priority, 5);
}

#[test]
fn test_sensing_workload_type_display() {
    let workload_type = WorkloadType::SensingProcessing;
    assert_eq!(format!("{}", workload_type), "SensingProcessing");
}

#[test]
fn test_sensing_workload_requirements() {
    let req = WorkloadRequirements::sensing_processing()
        .with_ue_id(1)
        .with_cell_id(100);

    assert_eq!(req.workload_type, WorkloadType::SensingProcessing);
    assert_eq!(req.latency_constraint_ms, Some(10));
    assert_eq!(req.priority, 7);
    assert_eq!(req.ue_id, Some(1));
    assert_eq!(req.cell_id, Some(100));
    assert_eq!(req.preferred_tier, Some(ComputeTier::LocalEdge));
}

// ============================================================================
// A19.3: SHE <-> Semantic Integration Tests
// ============================================================================

#[test]
fn test_she_semantic_encoding_workload_submission() {
    // Create Semantic client for submitting to SHE
    let mut semantic_client = SheSemanticClient::new();

    // Create raw data to encode (simulated image)
    let raw_data: Vec<f32> = (0..256).map(|i| (i as f32) / 255.0).collect();
    let dimensions = vec![16, 16];

    // Submit encoding workload to SHE
    let request = semantic_client.submit_encode(
        raw_data.clone(),
        dimensions.clone(),
        SemanticTask::ImageClassification,
        10.0, // 10x compression
    );

    assert_eq!(request.operation, SemanticCodingOperation::Encode);
    assert_eq!(request.task, SemanticTask::ImageClassification);
    assert_eq!(request.quality_params.target_compression, 10.0);
    assert_eq!(request.latency_requirement_ms, 15);
    assert_eq!(request.priority, 6);
}

#[test]
fn test_she_semantic_coding_workload_placement() {
    // Create SHE tier manager
    let mut tier_manager = TierManager::new();

    // Add edge nodes for semantic coding workloads
    tier_manager.add_node(ComputeNode::new(
        1,
        "edge-semantic-1",
        ComputeTier::LocalEdge,
        nextgsim_she::resource::ResourceCapacity::with_tflops(10).with_memory_gb(32),
    ));

    tier_manager.add_node(ComputeNode::new(
        2,
        "regional-semantic-1",
        ComputeTier::RegionalEdge,
        nextgsim_she::resource::ResourceCapacity::with_tflops(50).with_memory_gb(128),
    ));

    // Create scheduler
    let mut scheduler = nextgsim_she::scheduler::WorkloadScheduler::new(tier_manager);

    // Submit semantic coding workload
    let semantic_req = WorkloadRequirements::semantic_coding();
    let id = scheduler.submit(semantic_req).unwrap();
    let decision = scheduler.place(id).unwrap();

    // Should be placed at an edge tier (local or regional depending on scheduler heuristic)
    assert!(decision.tier == ComputeTier::LocalEdge || decision.tier == ComputeTier::RegionalEdge);
}

#[test]
fn test_she_semantic_adaptive_encoding() {
    let mut semantic_client = SheSemanticClient::new();

    let raw_data: Vec<f32> = (0..512).map(|i| (i as f32).sin()).collect();
    let dimensions = vec![32, 16];

    // Submit adaptive encoding with channel conditions
    let request = semantic_client.submit_adaptive_encode(
        raw_data,
        dimensions,
        SemanticTask::SensorFusion,
        15.0, // 15 dB SNR
        8.0,  // 8x target compression
    );

    assert_eq!(request.operation, SemanticCodingOperation::AdaptiveEncode);
    assert_eq!(request.task, SemanticTask::SensorFusion);
    assert!(request.quality_params.adaptive);
    assert_eq!(request.quality_params.channel_snr_db, Some(15.0));
    assert_eq!(request.priority, 7);
}

#[test]
fn test_she_semantic_decoding_workload() {
    let mut semantic_client = SheSemanticClient::new();

    // Create semantic features to decode
    let features =
        nextgsim_semantic::SemanticFeatures::new(1, vec![0.1, 0.2, 0.3, 0.4, 0.5], vec![256]);

    // Submit decoding workload
    let request = semantic_client.submit_decode(features, SemanticTask::VideoAnalytics);

    assert_eq!(request.operation, SemanticCodingOperation::Decode);
    assert_eq!(request.task, SemanticTask::VideoAnalytics);
    assert_eq!(request.latency_requirement_ms, 15);
}

#[test]
fn test_she_semantic_roundtrip_workload() {
    let mut semantic_client = SheSemanticClient::new();

    let raw_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let dimensions = vec![2, 4];

    // Submit roundtrip workload for testing
    let request = semantic_client.submit_roundtrip(
        raw_data,
        dimensions,
        SemanticTask::SpeechRecognition,
    );

    assert_eq!(request.operation, SemanticCodingOperation::Roundtrip);
    assert_eq!(request.task, SemanticTask::SpeechRecognition);
    assert_eq!(request.latency_requirement_ms, 25);
    assert_eq!(request.priority, 4);
}

#[test]
fn test_semantic_workload_type_display() {
    let workload_type = WorkloadType::SemanticCoding;
    assert_eq!(format!("{}", workload_type), "SemanticCoding");
}

#[test]
fn test_semantic_workload_requirements() {
    let req = WorkloadRequirements::semantic_coding()
        .with_ue_id(2)
        .with_model_name("semantic_encoder_v1");

    assert_eq!(req.workload_type, WorkloadType::SemanticCoding);
    assert_eq!(req.latency_constraint_ms, Some(15));
    assert_eq!(req.priority, 6);
    assert_eq!(req.ue_id, Some(2));
    assert_eq!(req.model_name, Some("semantic_encoder_v1".to_string()));
    assert_eq!(req.preferred_tier, Some(ComputeTier::LocalEdge));
}

#[test]
fn test_she_multiple_semantic_workloads() {
    let mut tier_manager = TierManager::new();
    tier_manager.add_node(ComputeNode::new(
        1,
        "edge-1",
        ComputeTier::LocalEdge,
        nextgsim_she::resource::ResourceCapacity::with_tflops(100).with_memory_gb(256),
    ));
    tier_manager.add_node(ComputeNode::new(
        2,
        "regional-1",
        ComputeTier::RegionalEdge,
        nextgsim_she::resource::ResourceCapacity::with_tflops(200).with_memory_gb(512),
    ));

    let mut scheduler = nextgsim_she::scheduler::WorkloadScheduler::new(tier_manager);

    // Submit multiple semantic coding workloads
    let req1 = WorkloadRequirements::semantic_coding().with_priority(8);
    let req2 = WorkloadRequirements::semantic_coding().with_priority(5);
    let req3 = WorkloadRequirements::semantic_coding().with_priority(7);

    let id1 = scheduler.submit(req1).unwrap();
    let id2 = scheduler.submit(req2).unwrap();
    let id3 = scheduler.submit(req3).unwrap();

    // All should be placeable across the available nodes
    let decision1 = scheduler.place(id1);
    let decision2 = scheduler.place(id2);
    let decision3 = scheduler.place(id3);

    assert!(decision1.is_ok());
    assert!(decision2.is_ok());
    assert!(decision3.is_ok());
}
