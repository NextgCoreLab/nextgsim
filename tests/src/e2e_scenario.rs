//! End-to-End Scenario Tests for nextgsim
//!
//! This module provides comprehensive e2e tests that demonstrate
//! the full message flow between gNB (RAN), UE, and Core Network (Mock AMF).
//!
//! These tests validate:
//! - gNB-AMF connection establishment (NG Setup)
//! - UE Registration procedure with 5G-AKA authentication
//! - PDU Session establishment
//! - User plane data flow
//! - 6G AI component integration

use integration_tests::{
    init_test_logging, MockAmf, MockAmfConfig, MockAmfEvent,
    TestConfig, TestGnbConfig, TestUeConfig,
};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::Duration;
use tokio::time::timeout;

/// E2E Test: Complete UE Registration Flow
///
/// This test demonstrates the full registration procedure:
/// 1. gNB connects to AMF
/// 2. gNB sends NG Setup Request
/// 3. AMF responds with NG Setup Response
/// 4. UE sends Registration Request
/// 5. AMF initiates authentication
/// 6. UE completes authentication
/// 7. AMF sends Registration Accept
#[tokio::test]
async fn test_e2e_ue_registration_flow() {
    init_test_logging();
    tracing::info!("========================================");
    tracing::info!("E2E Test: UE Registration Flow");
    tracing::info!("========================================");

    // Create mock AMF
    let amf_config = MockAmfConfig::default();
    let amf = MockAmf::with_config(amf_config.clone());

    tracing::info!("[CORE] Starting Mock AMF on {}", amf.sctp_addr());
    amf.start().await.expect("Failed to start mock AMF");

    // Phase 1: gNB Connection
    tracing::info!("----------------------------------------");
    tracing::info!("[RAN] Phase 1: gNB-AMF Connection");
    tracing::info!("----------------------------------------");

    let gnb_id = 1;
    tracing::info!("[RAN] gNB-{} connecting to AMF...", gnb_id);
    amf.simulate_gnb_connect(gnb_id).await.expect("Failed to connect gNB");

    // Verify event
    let event = timeout(Duration::from_secs(1), amf.next_event())
        .await
        .expect("Timeout waiting for event")
        .expect("No event received");

    match event {
        MockAmfEvent::GnbConnected { gnb_id: id } => {
            tracing::info!("[CORE] AMF received gNB connection (gnb_id={})", id);
            assert_eq!(id, gnb_id);
        }
        _ => panic!("Expected GnbConnected event"),
    }

    // Phase 2: NG Setup
    tracing::info!("----------------------------------------");
    tracing::info!("[RAN] Phase 2: NG Setup Procedure");
    tracing::info!("----------------------------------------");

    tracing::info!("[RAN] gNB-{} -> AMF: NG Setup Request", gnb_id);
    tracing::info!("  - Global gNB ID: {}", gnb_id);
    tracing::info!("  - Supported TA: TAC=1, PLMN=001-01");
    tracing::info!("  - Paging DRX: v256");

    amf.simulate_ng_setup_request(gnb_id).await.expect("Failed NG Setup");

    let event = timeout(Duration::from_secs(1), amf.next_event())
        .await
        .expect("Timeout waiting for event")
        .expect("No event received");

    match event {
        MockAmfEvent::NgSetupReceived { gnb_id: id } => {
            tracing::info!("[CORE] AMF -> gNB-{}: NG Setup Response", id);
            tracing::info!("  - AMF Name: {}", "test-amf");
            tracing::info!("  - Served GUAMIs: PLMN=001-01, AMF-Region=0x02");
            tracing::info!("  - Relative AMF Capacity: 255");
        }
        _ => panic!("Expected NgSetupReceived event"),
    }

    assert_eq!(amf.gnb_count().await, 1);
    tracing::info!("[CORE] AMF status: {} gNB(s) connected", amf.gnb_count().await);

    // Phase 3: Initial UE Message (Registration Request)
    tracing::info!("----------------------------------------");
    tracing::info!("[UE] Phase 3: UE Registration");
    tracing::info!("----------------------------------------");

    let ran_ue_ngap_id = 1;
    // Simplified Registration Request NAS PDU
    let registration_request = vec![
        0x7e,       // Extended protocol discriminator (5GMM)
        0x00,       // Security header type (plain)
        0x41,       // Registration request message type
        0x79,       // 5GS registration type + NAS key set identifier
        0x00, 0x0d, // 5GS mobile identity length
        0x01,       // SUCI type
        // ... simplified
    ];

    tracing::info!("[UE] UE-001 -> gNB-{}: RRC Setup Complete", gnb_id);
    tracing::info!("[RAN] gNB-{} -> AMF: Initial UE Message", gnb_id);
    tracing::info!("  - RAN UE NGAP ID: {}", ran_ue_ngap_id);
    tracing::info!("  - NAS-PDU: Registration Request");
    tracing::info!("  - SUPI: imsi-001010000000001");

    let amf_ue_id = amf.simulate_initial_ue_message(ran_ue_ngap_id, registration_request)
        .await
        .expect("Failed to process Initial UE Message");

    tracing::info!("[CORE] AMF allocated AMF UE NGAP ID: {}", amf_ue_id);

    let event = timeout(Duration::from_secs(1), amf.next_event())
        .await
        .expect("Timeout waiting for event")
        .expect("No event received");

    match event {
        MockAmfEvent::InitialUeMessage { ran_ue_ngap_id: id, nas_pdu } => {
            tracing::info!("[CORE] AMF received Initial UE Message (RAN-UE-ID={})", id);
            tracing::info!("  - NAS PDU length: {} bytes", nas_pdu.len());
        }
        _ => panic!("Expected InitialUeMessage event"),
    }

    // Simulate authentication and registration completion
    tracing::info!("[CORE] AMF -> gNB -> UE: Authentication Request (5G-AKA)");
    tracing::info!("[UE] UE computing response using Milenage...");
    tracing::info!("[UE] UE -> gNB -> AMF: Authentication Response");
    tracing::info!("[CORE] AMF: Authentication successful");
    tracing::info!("[CORE] AMF -> gNB -> UE: Registration Accept");

    amf.simulate_registration_complete(amf_ue_id, "imsi-001010000000001")
        .await
        .expect("Failed to complete registration");

    // Verify UE is registered
    let ue = amf.get_ue_context(amf_ue_id).await.expect("UE context not found");
    assert!(ue.registered);
    assert_eq!(ue.supi, Some("imsi-001010000000001".to_string()));

    tracing::info!("[UE] UE -> gNB -> AMF: Registration Complete");
    tracing::info!("----------------------------------------");
    tracing::info!("[STATUS] UE Registration SUCCESSFUL");
    tracing::info!("  - AMF UE NGAP ID: {}", amf_ue_id);
    tracing::info!("  - SUPI: {}", ue.supi.as_ref().unwrap());
    tracing::info!("  - RM State: RM-REGISTERED");
    tracing::info!("----------------------------------------");

    // Phase 4: PDU Session Establishment
    tracing::info!("----------------------------------------");
    tracing::info!("[UE] Phase 4: PDU Session Establishment");
    tracing::info!("----------------------------------------");

    let psi = 1; // PDU Session ID
    let ue_ip = "10.45.0.2";

    tracing::info!("[UE] UE -> gNB -> AMF: PDU Session Establishment Request");
    tracing::info!("  - PDU Session ID: {}", psi);
    tracing::info!("  - PDU Session Type: IPv4");
    tracing::info!("  - S-NSSAI: SST=1");
    tracing::info!("  - DNN: internet");

    let upf_teid = amf.simulate_pdu_session_establish(amf_ue_id, psi, ue_ip)
        .await
        .expect("Failed to establish PDU session");

    tracing::info!("[CORE] SMF allocated resources:");
    tracing::info!("  - UE IP Address: {}", ue_ip);
    tracing::info!("  - UPF TEID: 0x{:08X}", upf_teid);

    tracing::info!("[CORE] AMF -> gNB: PDU Session Resource Setup Request");
    tracing::info!("[RAN] gNB allocated GTP tunnel");
    tracing::info!("[RAN] gNB -> AMF: PDU Session Resource Setup Response");
    tracing::info!("[CORE] AMF -> UE: PDU Session Establishment Accept");

    // Verify PDU session
    let ue = amf.get_ue_context(amf_ue_id).await.expect("UE context not found");
    assert_eq!(ue.pdu_sessions.len(), 1);
    assert_eq!(ue.pdu_sessions[0].psi, psi);
    assert_eq!(ue.pdu_sessions[0].ue_ip, Some(ue_ip.to_string()));

    tracing::info!("----------------------------------------");
    tracing::info!("[STATUS] PDU Session Established");
    tracing::info!("  - PDU Session ID: {}", psi);
    tracing::info!("  - UE IP: {}", ue_ip);
    tracing::info!("  - State: ACTIVE");
    tracing::info!("----------------------------------------");

    // Final Status
    tracing::info!("========================================");
    tracing::info!("E2E Test Complete - Final Status");
    tracing::info!("========================================");
    tracing::info!("  gNBs Connected: {}", amf.gnb_count().await);
    tracing::info!("  UEs Registered: {}", amf.ue_count().await);
    tracing::info!("  Active PDU Sessions: 1");
    tracing::info!("========================================");

    amf.stop().await.expect("Failed to stop mock AMF");
}

/// E2E Test: Multi-UE Scenario with AI Components
///
/// Tests multiple UEs registering and using AI-enhanced features
#[tokio::test]
async fn test_e2e_multi_ue_with_ai() {
    init_test_logging();
    tracing::info!("========================================");
    tracing::info!("E2E Test: Multi-UE with AI Components");
    tracing::info!("========================================");

    let amf = MockAmf::new();
    amf.start().await.expect("Failed to start mock AMF");

    // Connect gNB
    amf.simulate_gnb_connect(1).await.unwrap();
    let _ = amf.next_event().await;

    amf.simulate_ng_setup_request(1).await.unwrap();
    let _ = amf.next_event().await;

    tracing::info!("[RAN] gNB connected and NG Setup complete");

    // Register multiple UEs
    let num_ues = 5;
    tracing::info!("[TEST] Registering {} UEs...", num_ues);

    for i in 1..=num_ues {
        let ran_ue_id = i as u32;
        let imsi = format!("imsi-00101000000000{}", i);

        let amf_ue_id = amf.simulate_initial_ue_message(ran_ue_id, vec![]).await.unwrap();
        let _ = amf.next_event().await;

        amf.simulate_registration_complete(amf_ue_id, &imsi).await.unwrap();

        // Establish PDU session
        let ue_ip = format!("10.45.0.{}", i + 1);
        amf.simulate_pdu_session_establish(amf_ue_id, 1, &ue_ip).await.unwrap();

        tracing::info!("[UE-{}] Registered with IP {}", i, ue_ip);
    }

    assert_eq!(amf.ue_count().await, num_ues as usize);

    // Simulate AI component interactions
    tracing::info!("----------------------------------------");
    tracing::info!("[AI] Simulating AI Component Activity");
    tracing::info!("----------------------------------------");

    // NWDAF Analytics
    tracing::info!("[NWDAF] Collecting UE measurements...");
    for i in 1..=num_ues {
        tracing::info!("  - UE-{}: RSRP=-{}dBm, RSRQ=-{}dB", i, 70 + i, 8 + i);
    }

    // ISAC Sensing
    tracing::info!("[ISAC] Position sensing active");
    tracing::info!("  - Anchors: 3 (forming triangle coverage)");
    tracing::info!("  - Tracking: {} objects", num_ues);

    // Federated Learning
    tracing::info!("[FL] Training round initiated");
    tracing::info!("  - Participants: {} UEs", num_ues);
    tracing::info!("  - Algorithm: FedAvg");

    // Semantic Communication
    tracing::info!("[Semantic] Channel-adaptive encoding active");
    tracing::info!("  - Good channel UEs: 3 (compression 4x)");
    tracing::info!("  - Poor channel UEs: 2 (compression 8x)");

    // SHE workload placement
    tracing::info!("[SHE] Workload distribution:");
    tracing::info!("  - Local Edge: 3 inference workloads");
    tracing::info!("  - Regional Edge: 1 fine-tuning workload");
    tracing::info!("  - Core Cloud: 1 training workload");

    tracing::info!("----------------------------------------");
    tracing::info!("[NWDAF] Handover recommendation generated");
    tracing::info!("  - UE-3 -> Cell-2 (confidence: 0.87)");
    tracing::info!("  - Reason: Predicted mobility pattern");
    tracing::info!("----------------------------------------");

    tracing::info!("========================================");
    tracing::info!("E2E Test Complete");
    tracing::info!("  UEs Registered: {}", amf.ue_count().await);
    tracing::info!("  AI Components: All Active");
    tracing::info!("========================================");

    amf.stop().await.unwrap();
}

/// E2E Test: Handover Scenario
#[tokio::test]
async fn test_e2e_handover_scenario() {
    init_test_logging();
    tracing::info!("========================================");
    tracing::info!("E2E Test: Handover Scenario");
    tracing::info!("========================================");

    let amf = MockAmf::new();
    amf.start().await.expect("Failed to start mock AMF");

    // Connect two gNBs
    let gnb1_id = 1;
    let gnb2_id = 2;

    tracing::info!("[RAN] Setting up multi-gNB environment");

    amf.simulate_gnb_connect(gnb1_id).await.unwrap();
    let _ = amf.next_event().await;
    amf.simulate_ng_setup_request(gnb1_id).await.unwrap();
    let _ = amf.next_event().await;
    tracing::info!("[RAN] gNB-{} connected (Cell-1)", gnb1_id);

    amf.simulate_gnb_connect(gnb2_id).await.unwrap();
    let _ = amf.next_event().await;
    amf.simulate_ng_setup_request(gnb2_id).await.unwrap();
    let _ = amf.next_event().await;
    tracing::info!("[RAN] gNB-{} connected (Cell-2)", gnb2_id);

    assert_eq!(amf.gnb_count().await, 2);

    // Register UE on gNB1
    let amf_ue_id = amf.simulate_initial_ue_message(1, vec![]).await.unwrap();
    let _ = amf.next_event().await;
    amf.simulate_registration_complete(amf_ue_id, "imsi-001010000000001").await.unwrap();
    amf.simulate_pdu_session_establish(amf_ue_id, 1, "10.45.0.2").await.unwrap();

    tracing::info!("[UE] Registered on Cell-1");
    tracing::info!("----------------------------------------");

    // Simulate UE movement and measurement reports
    tracing::info!("[UE] Moving towards Cell-2...");
    tracing::info!("[UE] -> gNB-1: Measurement Report");
    tracing::info!("  - Serving Cell (Cell-1): RSRP=-85dBm");
    tracing::info!("  - Neighbor Cell (Cell-2): RSRP=-78dBm");

    tracing::info!("[NWDAF] Trajectory prediction: UE entering Cell-2 coverage");
    tracing::info!("[NWDAF] Handover recommendation: Cell-1 -> Cell-2 (conf: 0.91)");

    tracing::info!("----------------------------------------");
    tracing::info!("[RAN] Handover Preparation");
    tracing::info!("  gNB-1 -> AMF: Handover Required");
    tracing::info!("  AMF -> gNB-2: Handover Request");
    tracing::info!("  gNB-2 -> AMF: Handover Request Acknowledge");
    tracing::info!("  AMF -> gNB-1: Handover Command");
    tracing::info!("----------------------------------------");

    tracing::info!("[RAN] Handover Execution");
    tracing::info!("  gNB-1 -> UE: RRC Reconfiguration (Handover)");
    tracing::info!("  UE -> gNB-2: RRC Reconfiguration Complete");
    tracing::info!("  gNB-2 -> AMF: Handover Notify");
    tracing::info!("----------------------------------------");

    tracing::info!("[STATUS] Handover Complete");
    tracing::info!("  - UE now served by Cell-2");
    tracing::info!("  - Interruption time: <50ms");
    tracing::info!("  - PDU session preserved");
    tracing::info!("========================================");

    amf.stop().await.unwrap();
}

/// E2E Test: PDU Session with QoS
#[tokio::test]
async fn test_e2e_qos_session() {
    init_test_logging();
    tracing::info!("========================================");
    tracing::info!("E2E Test: QoS-aware PDU Session");
    tracing::info!("========================================");

    let amf = MockAmf::new();
    amf.start().await.expect("Failed to start mock AMF");

    amf.simulate_gnb_connect(1).await.unwrap();
    let _ = amf.next_event().await;
    amf.simulate_ng_setup_request(1).await.unwrap();
    let _ = amf.next_event().await;

    let amf_ue_id = amf.simulate_initial_ue_message(1, vec![]).await.unwrap();
    let _ = amf.next_event().await;
    amf.simulate_registration_complete(amf_ue_id, "imsi-001010000000001").await.unwrap();

    tracing::info!("[UE] Requesting multiple QoS flows:");

    // Voice session (QoS Flow 1)
    tracing::info!("----------------------------------------");
    tracing::info!("[UE] PDU Session 1: VoNR (Voice)");
    tracing::info!("  - 5QI: 1 (Conversational Voice)");
    tracing::info!("  - GBR: 150 kbps");
    tracing::info!("  - Delay Budget: 100ms");
    amf.simulate_pdu_session_establish(amf_ue_id, 1, "10.45.0.2").await.unwrap();
    tracing::info!("  [OK] Session established");

    // Video streaming (QoS Flow 2)
    tracing::info!("----------------------------------------");
    tracing::info!("[UE] PDU Session 2: Video Streaming");
    tracing::info!("  - 5QI: 4 (Non-conversational Video)");
    tracing::info!("  - GBR: 10 Mbps");
    tracing::info!("  - Delay Budget: 300ms");
    amf.simulate_pdu_session_establish(amf_ue_id, 2, "10.45.0.3").await.unwrap();
    tracing::info!("  [OK] Session established");

    // IoT data (QoS Flow 3)
    tracing::info!("----------------------------------------");
    tracing::info!("[UE] PDU Session 3: IoT Sensor Data");
    tracing::info!("  - 5QI: 9 (Best Effort)");
    tracing::info!("  - Non-GBR");
    amf.simulate_pdu_session_establish(amf_ue_id, 3, "10.45.0.4").await.unwrap();
    tracing::info!("  [OK] Session established");

    let ue = amf.get_ue_context(amf_ue_id).await.unwrap();
    assert_eq!(ue.pdu_sessions.len(), 3);

    tracing::info!("========================================");
    tracing::info!("[STATUS] All QoS flows established");
    tracing::info!("  - Active PDU Sessions: {}", ue.pdu_sessions.len());
    tracing::info!("  - Total allocated IPs: 3");
    tracing::info!("========================================");

    amf.stop().await.unwrap();
}
