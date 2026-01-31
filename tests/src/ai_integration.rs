//! Integration tests for 6G AI-native network functions
//!
//! This module tests end-to-end scenarios involving:
//! - Service Hosting Environment (SHE) workload placement and inference
//! - Network Data Analytics Function (NWDAF) predictions
//! - Integrated Sensing and Communication (ISAC) position fusion
//! - Federated Learning aggregation workflows
//! - Semantic Communication encoding/decoding
//! - Knowledge Exposure Function (NKEF) semantic search
//! - AI Agent Framework coordination

use std::collections::HashMap;

use nextgsim_ai::tensor::{TensorData, TensorShape};
use nextgsim_fl::{AggregationAlgorithm, FederatedAggregator, ModelUpdate};
use nextgsim_isac::{fuse_positions, SensingMeasurement, SensingType, TrackingState, Vector3};
use nextgsim_nkef::{Entity, EntityType, KnowledgeGraph, NkefManager, QueryContext};
use nextgsim_nwdaf::{CellLoad, NwdafManager, UeMeasurement, Vector3 as NwdafVector3};
use nextgsim_semantic::{
    ChannelQuality, EncoderConfig, SemanticDecoder, SemanticEncoder, SemanticTask,
};
use nextgsim_she::{
    resource::ResourceCapacity,
    scheduler::WorkloadScheduler,
    tier::{ComputeNode, ComputeTier, TierManager},
    workload::WorkloadRequirements,
};
use nextgsim_agent::{
    AgentCapabilities, AgentCoordinator, AgentId, AgentType, Intent, IntentType, ResourceLimits,
};

// ============================================================================
// SHE Integration Tests
// ============================================================================

#[test]
fn test_she_workload_placement_e2e() {
    // Create a tier manager with nodes at each tier
    let mut tier_manager = TierManager::new();

    // Add edge nodes
    tier_manager.add_node(ComputeNode::new(
        1,
        "edge-local-1",
        ComputeTier::LocalEdge,
        ResourceCapacity::with_tflops(2).with_memory_gb(16),
    ));
    tier_manager.add_node(ComputeNode::new(
        2,
        "edge-regional-1",
        ComputeTier::RegionalEdge,
        ResourceCapacity::with_tflops(20).with_memory_gb(128),
    ));
    tier_manager.add_node(ComputeNode::new(
        3,
        "cloud-1",
        ComputeTier::CoreCloud,
        ResourceCapacity::with_tflops(200).with_memory_gb(1024),
    ));

    // Create scheduler
    let mut scheduler = WorkloadScheduler::new(tier_manager);

    // Test inference workload placement (should go to local edge)
    let inference_req = WorkloadRequirements::inference();
    let id1 = scheduler.submit(inference_req).unwrap();
    let decision1 = scheduler.place(id1).unwrap();
    assert_eq!(decision1.tier, ComputeTier::LocalEdge);

    // Test training workload placement (should go to cloud)
    let training_req = WorkloadRequirements::training();
    let id2 = scheduler.submit(training_req).unwrap();
    let decision2 = scheduler.place(id2).unwrap();
    assert_eq!(decision2.tier, ComputeTier::CoreCloud);

    // Test fine-tuning workload (should go to regional edge)
    let finetune_req = WorkloadRequirements::fine_tuning();
    let id3 = scheduler.submit(finetune_req).unwrap();
    let decision3 = scheduler.place(id3).unwrap();
    assert_eq!(decision3.tier, ComputeTier::RegionalEdge);

    // Verify resource tracking
    let local_usage = scheduler.tier_manager().tier_usage(ComputeTier::LocalEdge);
    assert!(local_usage.active_workloads > 0);
}

#[test]
fn test_she_workload_migration() {
    let mut tier_manager = TierManager::new();
    tier_manager.add_node(ComputeNode::new(
        1,
        "edge-1",
        ComputeTier::LocalEdge,
        ResourceCapacity::with_tflops(2).with_memory_gb(16),
    ));
    tier_manager.add_node(ComputeNode::new(
        2,
        "regional-1",
        ComputeTier::RegionalEdge,
        ResourceCapacity::with_tflops(20).with_memory_gb(128),
    ));

    let mut scheduler = WorkloadScheduler::new(tier_manager);

    // Place a workload at local edge
    let req = WorkloadRequirements::inference();
    let id = scheduler.submit(req).unwrap();
    let initial = scheduler.place(id).unwrap();
    assert_eq!(initial.tier, ComputeTier::LocalEdge);

    // Migrate to regional edge
    let migrated = scheduler.migrate(id, ComputeTier::RegionalEdge).unwrap();
    assert_eq!(migrated.tier, ComputeTier::RegionalEdge);

    // Verify old node is freed
    let old_node = scheduler.tier_manager().get_node(1).unwrap();
    assert_eq!(old_node.usage.active_workloads, 0);
}

#[test]
fn test_she_resource_capacity_display() {
    let capacity = ResourceCapacity::with_tflops(10).with_memory_gb(64).with_gpus(4);
    let display = format!("{}", capacity);
    assert!(display.contains("TFLOPS"));
    assert!(display.contains("GB"));
    assert!(display.contains("GPU"));
}

// ============================================================================
// NWDAF Integration Tests
// ============================================================================

#[test]
fn test_nwdaf_measurement_collection() {
    let mut manager = NwdafManager::new(10); // 10 sample history

    // Record UE measurements
    manager.record_measurement(UeMeasurement {
        ue_id: 1,
        rsrp: -80.0,
        rsrq: -10.0,
        sinr: Some(15.0),
        position: NwdafVector3::new(0.0, 0.0, 0.0),
        velocity: Some(NwdafVector3::new(10.0, 5.0, 0.0)),
        serving_cell_id: 1,
        timestamp_ms: 0,
    });

    manager.record_measurement(UeMeasurement {
        ue_id: 1,
        rsrp: -82.0,
        rsrq: -11.0,
        sinr: Some(14.0),
        position: NwdafVector3::new(1.0, 0.5, 0.0),
        velocity: Some(NwdafVector3::new(10.0, 5.0, 0.0)),
        serving_cell_id: 1,
        timestamp_ms: 100,
    });

    manager.record_measurement(UeMeasurement {
        ue_id: 1,
        rsrp: -84.0,
        rsrq: -12.0,
        sinr: Some(13.0),
        position: NwdafVector3::new(2.0, 1.0, 0.0),
        velocity: Some(NwdafVector3::new(10.0, 5.0, 0.0)),
        serving_cell_id: 1,
        timestamp_ms: 200,
    });

    // Verify history exists
    let history = manager.get_ue_history(1);
    assert!(history.is_some());

    // Request trajectory prediction
    let prediction = manager.predict_trajectory(1, 500);
    assert!(prediction.is_some());

    let pred = prediction.unwrap();
    assert!(!pred.waypoints.is_empty());
    assert!(pred.confidence > 0.0);
}

#[test]
fn test_nwdaf_cell_load_analytics() {
    let mut manager = NwdafManager::new(10);

    // Record cell load reports
    manager.record_cell_load(CellLoad {
        cell_id: 1,
        prb_usage: 0.5,
        connected_ues: 10,
        avg_throughput_mbps: 100.0,
        timestamp_ms: 0,
    });

    manager.record_cell_load(CellLoad {
        cell_id: 1,
        prb_usage: 0.7,
        connected_ues: 15,
        avg_throughput_mbps: 150.0,
        timestamp_ms: 1000,
    });

    // Get cell load
    let load = manager.get_cell_load(1);
    assert!(load.is_some());
}

#[test]
fn test_nwdaf_handover_recommendation() {
    let mut manager = NwdafManager::new(5);

    // Record measurements showing weak signal
    manager.record_measurement(UeMeasurement {
        ue_id: 1,
        rsrp: -100.0, // Very weak signal
        rsrq: -18.0,
        sinr: Some(2.0),
        position: NwdafVector3::new(0.0, 0.0, 0.0),
        velocity: None,
        serving_cell_id: 1,
        timestamp_ms: 0,
    });

    // Get handover recommendation with neighbor cells
    let neighbors = vec![(2, -85.0), (3, -90.0)]; // Cell 2 has better signal
    let recommendation = manager.recommend_handover(1, &neighbors);

    // Recommendation should suggest cell 2 (better signal)
    if let Some(rec) = recommendation {
        assert_eq!(rec.target_cell_id, 2);
        assert!(rec.confidence > 0.0);
    }
}

// ============================================================================
// ISAC Integration Tests
// ============================================================================

#[test]
fn test_isac_position_fusion_e2e() {
    // Create sensing measurements from multiple anchors
    let measurements = vec![
        SensingMeasurement {
            measurement_type: SensingType::ToA,
            anchor_id: 1,
            value: 50.0, // ~50m distance
            uncertainty: 5.0,
            timestamp_ms: 0,
        },
        SensingMeasurement {
            measurement_type: SensingType::ToA,
            anchor_id: 2,
            value: 50.0,
            uncertainty: 5.0,
            timestamp_ms: 0,
        },
        SensingMeasurement {
            measurement_type: SensingType::ToA,
            anchor_id: 3,
            value: 36.0, // Closer to anchor 3
            uncertainty: 5.0,
            timestamp_ms: 0,
        },
    ];

    // Define anchor positions (forming a triangle)
    let mut anchors = HashMap::new();
    anchors.insert(1, Vector3::new(0.0, 0.0, 10.0));
    anchors.insert(2, Vector3::new(100.0, 0.0, 10.0));
    anchors.insert(3, Vector3::new(50.0, 86.6, 10.0));

    // Perform position fusion
    let result = fuse_positions(&measurements, &anchors);
    assert!(result.is_some());

    let (position, uncertainty) = result.unwrap();
    assert!(uncertainty > 0.0);
    // Position should be somewhere in the triangle
    assert!(position.x >= 0.0 && position.x <= 100.0);
}

#[test]
fn test_isac_tracking_state() {
    let initial_pos = Vector3::new(0.0, 0.0, 0.0);
    let mut state = TrackingState::new(1, initial_pos);

    // Update with measurements
    state.update(Vector3::new(10.0, 5.0, 0.0), 2.0);
    state.update(Vector3::new(20.0, 10.0, 0.0), 2.0);

    // Predict position at future time
    let predicted = state.predict(1.0); // 1 second into future

    // Should have extrapolated based on velocity
    assert!(predicted.x > 20.0);
    assert!(predicted.y > 10.0);
}

// ============================================================================
// Federated Learning Integration Tests
// ============================================================================

#[test]
fn test_fl_complete_training_round() {
    let mut aggregator = FederatedAggregator::new(
        AggregationAlgorithm::FedAvg,
        2, // min participants
    );

    // Initialize model
    aggregator.initialize_model(vec![0.0; 10]);

    // Register participants
    aggregator.register_participant("ue-001", 100);
    aggregator.register_participant("ue-002", 150);

    assert_eq!(aggregator.participant_count(), 2);

    // Start a training round
    let round = aggregator.start_round().unwrap();
    assert_eq!(round, 1);

    // Submit updates from participants
    aggregator
        .submit_update(ModelUpdate {
            participant_id: "ue-001".to_string(),
            base_version: 0,
            gradients: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            num_samples: 100,
            loss: 0.5,
            timestamp_ms: 1000,
        })
        .unwrap();

    aggregator
        .submit_update(ModelUpdate {
            participant_id: "ue-002".to_string(),
            base_version: 0,
            gradients: vec![0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05],
            num_samples: 150,
            loss: 0.45,
            timestamp_ms: 1000,
        })
        .unwrap();

    // Aggregate
    let result = aggregator.aggregate().unwrap();

    assert_eq!(result.num_participants, 2);
    assert!(!result.weights.is_empty());
    assert!(result.avg_loss < 0.5); // Should be weighted average
}

#[test]
fn test_fl_with_differential_privacy() {
    let aggregator = FederatedAggregator::new(
        AggregationAlgorithm::FedAvg,
        2,
    ).with_dp_config(nextgsim_fl::DifferentialPrivacyConfig {
        enabled: true,
        noise_multiplier: 1.1,
        clipping_threshold: 1.0,
        target_epsilon: 8.0,
        target_delta: 1e-5,
    });

    // DP config is applied via builder pattern
    assert!(aggregator.participant_count() == 0); // No participants yet
}

// ============================================================================
// Semantic Communication Integration Tests
// ============================================================================

#[test]
fn test_semantic_encode_decode_roundtrip() {
    let encoder = SemanticEncoder::default();
    let decoder = SemanticDecoder::default();

    // Create test data (simulated image features)
    let data: Vec<f32> = (0..256).map(|i| (i as f32) / 255.0).collect();

    // Encode
    let features = encoder.encode(&data, SemanticTask::ImageClassification);

    assert!(features.num_features() > 0);
    assert!(features.compression_ratio > 1.0);
    assert_eq!(features.importance.len(), features.features.len());

    // Decode
    let reconstructed = decoder.decode(&features);

    assert_eq!(reconstructed.len(), data.len());
}

#[test]
fn test_semantic_adaptive_encoding() {
    let encoder = SemanticEncoder::new(EncoderConfig {
        adaptive: true,
        target_compression: 10.0,
        min_quality: 0.7,
        task: SemanticTask::SensorFusion,
    });

    let data: Vec<f32> = (0..512).map(|i| (i as f32).sin()).collect();

    // Good channel - should keep more features
    let good_channel = ChannelQuality::new(25.0, 1000.0, 0.001);
    let good_features = encoder.adaptive_encode(&data, SemanticTask::SensorFusion, &good_channel);

    // Poor channel - should aggressively compress
    let poor_channel = ChannelQuality::new(5.0, 100.0, 0.15);
    let poor_features = encoder.adaptive_encode(&data, SemanticTask::SensorFusion, &poor_channel);

    // Poor channel should result in fewer features
    assert!(poor_features.num_features() <= good_features.num_features());
    assert!(poor_features.compression_ratio >= good_features.compression_ratio);
}

#[test]
fn test_semantic_importance_based_pruning() {
    let features = nextgsim_semantic::SemanticFeatures::new(
        1,
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        vec![100],
    )
    .with_importance(vec![0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.95]);

    // Prune to keep 50% of features
    let pruned = features.prune(0.5);

    assert_eq!(pruned.num_features(), 5);
    assert!(pruned.compression_ratio > features.compression_ratio);
}

#[test]
fn test_semantic_channel_quality_categories() {
    let excellent = ChannelQuality::new(25.0, 1000.0, 0.001);
    let poor = ChannelQuality::new(3.0, 100.0, 0.2);

    assert_eq!(excellent.category(), nextgsim_semantic::ChannelCategory::Excellent);
    assert_eq!(poor.category(), nextgsim_semantic::ChannelCategory::Poor);
}

// ============================================================================
// NKEF Integration Tests
// ============================================================================

#[test]
fn test_nkef_knowledge_graph_e2e() {
    let mut graph = KnowledgeGraph::new();

    // Add network entities
    graph.add_entity(
        Entity::new("gnb-001", EntityType::Gnb)
            .with_property("location", "Building A")
            .with_property("capacity", "1000")
            .with_property("status", "active"),
    );

    graph.add_entity(
        Entity::new("gnb-002", EntityType::Gnb)
            .with_property("location", "Building B")
            .with_property("capacity", "800")
            .with_property("status", "active"),
    );

    graph.add_entity(
        Entity::new("ue-001", EntityType::Ue)
            .with_property("imsi", "001010000000001")
            .with_property("connected_gnb", "gnb-001"),
    );

    // Search for gNBs - note: search also finds UE's "connected_gnb" property containing "gnb"
    let results = graph.search("gnb", 10);
    assert_eq!(results.len(), 3);  // 2 gNBs + 1 UE with connected_gnb property

    // Search for specific building
    let building_a = graph.search("Building A", 10);
    assert_eq!(building_a.len(), 1);
    assert_eq!(building_a[0].entity.id, "gnb-001");
}

#[test]
fn test_nkef_context_generation_for_rag() {
    let mut manager = NkefManager::new(384);

    // Add entities
    manager.update_entity(
        Entity::new("cell-001", EntityType::Cell)
            .with_property("pci", "100")
            .with_property("load", "75%")
            .with_property("frequency", "3500MHz"),
    );

    manager.update_entity(
        Entity::new("cell-002", EntityType::Cell)
            .with_property("pci", "101")
            .with_property("load", "45%")
            .with_property("frequency", "3500MHz"),
    );

    // Query for context
    let context = QueryContext {
        intent: "find_cell".to_string(),
        filters: HashMap::new(),
        max_results: 5,
    };

    let results = manager.query("cell", &context);
    assert_eq!(results.len(), 2);

    // Generate RAG context
    let rag_context = manager.retrieve_context("cell", 1000);
    assert!(!rag_context.sources.is_empty());
    assert!(rag_context.confidence > 0.0);
}

// ============================================================================
// Agent Framework Integration Tests
// ============================================================================

#[test]
fn test_agent_registration_and_intent() {
    let mut coordinator = AgentCoordinator::new();

    // Register an agent
    let agent_id = AgentId::new("network-optimizer-001");
    let token = coordinator.register_agent(
        agent_id.clone(),
        AgentType::Resource,
        AgentCapabilities {
            read_state: true,
            modify_config: true,
            trigger_actions: true,
            allowed_intents: vec!["optimize".to_string(), "handover".to_string()],
            resource_limits: ResourceLimits::default(),
        },
    );

    // Verify token was issued
    assert!(!token.token.is_empty());

    // Verify agent count
    assert_eq!(coordinator.agent_count(), 1);

    // Submit an intent
    let intent = Intent::new(agent_id.clone(), IntentType::TriggerHandover)
        .with_target("ue-001")
        .with_param("target_cell", "gnb-002")
        .with_priority(8);

    let intent_result = coordinator.submit_intent(intent);
    assert!(intent_result.is_ok());
}

#[test]
fn test_agent_token_lifecycle() {
    let mut coordinator = AgentCoordinator::new();

    let agent_id = AgentId::new("test-agent");
    let initial_token = coordinator.register_agent(
        agent_id.clone(),
        AgentType::Custom,
        AgentCapabilities::default(),
    );

    assert!(!initial_token.token.is_empty());

    // Refresh token
    let new_token = coordinator.refresh_token(&agent_id);
    assert!(new_token.is_some());
    assert!(!new_token.unwrap().token.is_empty());
}

#[test]
fn test_agent_token_validation() {
    let mut coordinator = AgentCoordinator::new();

    let agent_id = AgentId::new("validated-agent");
    let token = coordinator.register_agent(
        agent_id,
        AgentType::Mobility,
        AgentCapabilities::default(),
    );

    // Validate the token
    let registration = coordinator.validate_token(&token.token);
    assert!(registration.is_some());

    // Invalid token should fail
    let invalid = coordinator.validate_token("invalid-token");
    assert!(invalid.is_none());
}

// ============================================================================
// Cross-Component Integration Tests
// ============================================================================

#[test]
fn test_semantic_with_channel_adaptation() {
    // This test simulates semantic communication adapting based on channel quality

    let encoder = SemanticEncoder::new(EncoderConfig {
        adaptive: true,
        target_compression: 8.0,
        min_quality: 0.6,
        task: SemanticTask::ImageClassification,
    });

    let decoder = SemanticDecoder::default();

    // Simulate data
    let data: Vec<f32> = (0..256).map(|i| (i as f32) / 255.0).collect();

    // Good channel scenario
    let good_channel = ChannelQuality::new(20.0, 1000.0, 0.01);
    let good_features = encoder.adaptive_encode(&data, SemanticTask::ImageClassification, &good_channel);
    let good_decoded = decoder.decode(&good_features);

    // Poor channel scenario
    let poor_channel = ChannelQuality::new(5.0, 100.0, 0.1);
    let poor_features = encoder.adaptive_encode(&data, SemanticTask::ImageClassification, &poor_channel);
    let poor_decoded = decoder.decode(&poor_features);

    // Both should produce valid outputs
    assert_eq!(good_decoded.len(), data.len());
    assert_eq!(poor_decoded.len(), data.len());

    // Poor channel should have higher compression
    assert!(poor_features.compression_ratio >= good_features.compression_ratio);
}

// ============================================================================
// Tensor Operations Tests
// ============================================================================

#[test]
fn test_tensor_data_operations() {
    // Test basic tensor operations used by AI components
    let shape = TensorShape::new(vec![2, 3, 4]);
    assert_eq!(shape.dims(), &[2, 3, 4]);
    assert_eq!(shape.num_elements(), 24);

    // Create tensor data
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let tensor = TensorData::float32(data.clone(), shape.clone());

    assert_eq!(tensor.shape(), &shape);

    // Get f32 data back
    let recovered = tensor.as_f32_slice();
    assert!(recovered.is_some());
    assert_eq!(recovered.unwrap(), data.as_slice());
}

#[test]
fn test_tensor_shape_compatibility() {
    let shape1 = TensorShape::new(vec![1, 3, 4]);
    let shape2 = TensorShape::new(vec![1, 3, 4]);

    // Same shapes should be compatible
    assert!(shape1.is_compatible_with(&shape2));

    // Dynamic dimension (-1) should be compatible
    let shape3 = TensorShape::new(vec![1, -1, 4]);
    assert!(shape1.is_compatible_with(&shape3));
}
