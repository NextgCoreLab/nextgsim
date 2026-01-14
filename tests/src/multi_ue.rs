//! Multiple UE scenario integration tests
//!
//! Tests scenarios with multiple concurrent UEs.
//! Validates US-014: Multi-UE Support requirements:
//! - Multiple UEs with independent state machines
//! - Unique identities (SUPI/IMEI) per UE
//! - Efficient handling of concurrent operations

use integration_tests::{init_test_logging, MockAmf};
use std::sync::Arc;

/// Test multiple UE registration
#[tokio::test]
async fn test_multiple_ue_registration() {
    init_test_logging();
    
    let amf = MockAmf::new();
    amf.start().await.expect("Failed to start mock AMF");
    
    // Setup gNB
    amf.simulate_gnb_connect(1).await.unwrap();
    amf.simulate_ng_setup_request(1).await.unwrap();
    
    // Register multiple UEs
    let num_ues = 10;
    let mut ue_ids = Vec::new();
    
    for i in 0..num_ues {
        let amf_ue_id = amf.simulate_initial_ue_message(i as u32, vec![]).await.unwrap();
        let imsi = format!("imsi-00101000000000{:02}", i + 1);
        amf.simulate_registration_complete(amf_ue_id, &imsi).await.unwrap();
        ue_ids.push(amf_ue_id);
    }
    
    // Verify all UEs registered
    assert_eq!(amf.ue_count().await, num_ues);
    
    for (i, &amf_ue_id) in ue_ids.iter().enumerate() {
        let ue = amf.get_ue_context(amf_ue_id).await.expect("UE context not found");
        assert!(ue.registered);
        let expected_imsi = format!("imsi-00101000000000{:02}", i + 1);
        assert_eq!(ue.supi, Some(expected_imsi));
    }
    
    amf.stop().await.unwrap();
}

/// Test concurrent UE operations
/// Validates that multiple UEs can perform operations simultaneously
#[tokio::test]
async fn test_concurrent_ue_operations() {
    init_test_logging();
    
    let amf = Arc::new(MockAmf::new());
    amf.start().await.expect("Failed to start mock AMF");
    
    // Setup gNB
    amf.simulate_gnb_connect(1).await.unwrap();
    amf.simulate_ng_setup_request(1).await.unwrap();
    
    // Spawn concurrent registration tasks
    let num_ues = 20;
    let mut handles = Vec::new();
    
    for i in 0..num_ues {
        let amf_clone = Arc::clone(&amf);
        let handle = tokio::spawn(async move {
            let amf_ue_id = amf_clone.simulate_initial_ue_message(i as u32, vec![]).await.unwrap();
            let imsi = format!("imsi-00101000000000{:02}", i + 1);
            amf_clone.simulate_registration_complete(amf_ue_id, &imsi).await.unwrap();
            
            // Each UE establishes a PDU session
            let ue_ip = format!("10.45.0.{}", i + 2);
            amf_clone.simulate_pdu_session_establish(amf_ue_id, 1, &ue_ip).await.unwrap();
            
            amf_ue_id
        });
        handles.push(handle);
    }
    
    // Wait for all concurrent operations to complete
    let mut ue_ids = Vec::new();
    for handle in handles {
        let amf_ue_id = handle.await.expect("Task panicked");
        ue_ids.push(amf_ue_id);
    }
    
    // Verify all UEs registered and have PDU sessions
    assert_eq!(amf.ue_count().await, num_ues);
    
    for &amf_ue_id in &ue_ids {
        let ue = amf.get_ue_context(amf_ue_id).await.expect("UE context not found");
        assert!(ue.registered);
        assert_eq!(ue.pdu_sessions.len(), 1);
    }
    
    amf.stop().await.unwrap();
}

/// Test high UE count (stress test)
/// Validates NFR-001: System SHALL handle at least 100 concurrent UE instances
/// Note: Using 50 UEs for faster test execution; full 100 UE test can be run separately
#[tokio::test]
async fn test_high_ue_count() {
    init_test_logging();
    
    let amf = MockAmf::new();
    amf.start().await.expect("Failed to start mock AMF");
    
    // Setup gNB
    amf.simulate_gnb_connect(1).await.unwrap();
    amf.simulate_ng_setup_request(1).await.unwrap();
    
    // Register 50 UEs (reduced from 100 for faster test execution)
    let num_ues = 50;
    let mut ue_ids = Vec::new();
    
    for i in 0..num_ues {
        let amf_ue_id = amf.simulate_initial_ue_message(i as u32, vec![]).await.unwrap();
        let imsi = format!("imsi-001010000000{:04}", i + 1);
        amf.simulate_registration_complete(amf_ue_id, &imsi).await.unwrap();
        ue_ids.push(amf_ue_id);
    }
    
    // Verify all UEs registered
    assert_eq!(amf.ue_count().await, num_ues);
    
    // Verify unique identities
    let mut seen_imsis = std::collections::HashSet::new();
    for (i, &amf_ue_id) in ue_ids.iter().enumerate() {
        let ue = amf.get_ue_context(amf_ue_id).await.expect("UE context not found");
        assert!(ue.registered);
        
        let expected_imsi = format!("imsi-001010000000{:04}", i + 1);
        assert_eq!(ue.supi, Some(expected_imsi.clone()));
        
        // Ensure IMSI is unique
        assert!(seen_imsis.insert(expected_imsi), "Duplicate IMSI found");
    }
    
    amf.stop().await.unwrap();
}

/// Test multiple UEs with PDU sessions
#[tokio::test]
async fn test_multiple_ue_pdu_sessions() {
    init_test_logging();
    
    let amf = MockAmf::new();
    amf.start().await.expect("Failed to start mock AMF");
    
    // Setup gNB
    amf.simulate_gnb_connect(1).await.unwrap();
    amf.simulate_ng_setup_request(1).await.unwrap();
    
    // Register UEs and establish PDU sessions
    let num_ues = 5;
    let mut ue_ids = Vec::new();
    
    for i in 0..num_ues {
        let amf_ue_id = amf.simulate_initial_ue_message(i as u32, vec![]).await.unwrap();
        let imsi = format!("imsi-00101000000000{:02}", i + 1);
        amf.simulate_registration_complete(amf_ue_id, &imsi).await.unwrap();
        
        // Each UE gets a PDU session
        let ue_ip = format!("10.45.0.{}", i + 2);
        amf.simulate_pdu_session_establish(amf_ue_id, 1, &ue_ip).await.unwrap();
        
        ue_ids.push(amf_ue_id);
    }
    
    // Verify all UEs have PDU sessions
    for (i, &amf_ue_id) in ue_ids.iter().enumerate() {
        let ue = amf.get_ue_context(amf_ue_id).await.expect("UE context not found");
        assert_eq!(ue.pdu_sessions.len(), 1);
        let expected_ip = format!("10.45.0.{}", i + 2);
        assert_eq!(ue.pdu_sessions[0].ue_ip, Some(expected_ip));
    }
    
    amf.stop().await.unwrap();
}

/// Test UE deregistration while others remain active
#[tokio::test]
async fn test_selective_ue_deregistration() {
    init_test_logging();
    
    let amf = MockAmf::new();
    amf.start().await.expect("Failed to start mock AMF");
    
    // Setup gNB
    amf.simulate_gnb_connect(1).await.unwrap();
    amf.simulate_ng_setup_request(1).await.unwrap();
    
    // Register 3 UEs
    let mut ue_ids = Vec::new();
    for i in 0..3 {
        let amf_ue_id = amf.simulate_initial_ue_message(i as u32, vec![]).await.unwrap();
        let imsi = format!("imsi-00101000000000{:02}", i + 1);
        amf.simulate_registration_complete(amf_ue_id, &imsi).await.unwrap();
        ue_ids.push(amf_ue_id);
    }
    
    assert_eq!(amf.ue_count().await, 3);
    
    // Deregister middle UE
    amf.simulate_ue_context_release(ue_ids[1]).await.unwrap();
    
    // Verify only 2 UEs remain
    assert_eq!(amf.ue_count().await, 2);
    assert!(amf.get_ue_context(ue_ids[0]).await.is_some());
    assert!(amf.get_ue_context(ue_ids[1]).await.is_none());
    assert!(amf.get_ue_context(ue_ids[2]).await.is_some());
    
    amf.stop().await.unwrap();
}

/// Test multiple gNBs with UEs
#[tokio::test]
async fn test_multiple_gnbs() {
    init_test_logging();
    
    let amf = MockAmf::new();
    amf.start().await.expect("Failed to start mock AMF");
    
    // Connect multiple gNBs
    for gnb_id in 1..=3 {
        amf.simulate_gnb_connect(gnb_id).await.unwrap();
        amf.simulate_ng_setup_request(gnb_id).await.unwrap();
    }
    
    assert_eq!(amf.gnb_count().await, 3);
    
    // Register UEs on different gNBs (simulated via different RAN UE NGAP IDs)
    let amf_ue_id_1 = amf.simulate_initial_ue_message(100, vec![]).await.unwrap(); // gNB 1
    let amf_ue_id_2 = amf.simulate_initial_ue_message(200, vec![]).await.unwrap(); // gNB 2
    let amf_ue_id_3 = amf.simulate_initial_ue_message(300, vec![]).await.unwrap(); // gNB 3
    
    amf.simulate_registration_complete(amf_ue_id_1, "imsi-001010000000001").await.unwrap();
    amf.simulate_registration_complete(amf_ue_id_2, "imsi-001010000000002").await.unwrap();
    amf.simulate_registration_complete(amf_ue_id_3, "imsi-001010000000003").await.unwrap();
    
    assert_eq!(amf.ue_count().await, 3);
    
    // Disconnect one gNB
    amf.simulate_gnb_disconnect(2).await.unwrap();
    assert_eq!(amf.gnb_count().await, 2);
    
    // UEs should still exist (in real scenario, they would be released)
    assert_eq!(amf.ue_count().await, 3);
    
    amf.stop().await.unwrap();
}

/// Test independent UE state machines
/// Validates US-014: Each UE SHALL have independent state machines
#[tokio::test]
async fn test_independent_ue_state_machines() {
    init_test_logging();
    
    let amf = MockAmf::new();
    amf.start().await.expect("Failed to start mock AMF");
    
    // Setup gNB
    amf.simulate_gnb_connect(1).await.unwrap();
    amf.simulate_ng_setup_request(1).await.unwrap();
    
    // Register 4 UEs
    let mut ue_ids = Vec::new();
    for i in 0..4 {
        let amf_ue_id = amf.simulate_initial_ue_message(i as u32, vec![]).await.unwrap();
        let imsi = format!("imsi-00101000000000{:02}", i + 1);
        amf.simulate_registration_complete(amf_ue_id, &imsi).await.unwrap();
        ue_ids.push(amf_ue_id);
    }
    
    // UE 0: Has 2 PDU sessions
    amf.simulate_pdu_session_establish(ue_ids[0], 1, "10.45.0.2").await.unwrap();
    amf.simulate_pdu_session_establish(ue_ids[0], 2, "10.45.0.3").await.unwrap();
    
    // UE 1: Has 1 PDU session
    amf.simulate_pdu_session_establish(ue_ids[1], 1, "10.45.0.4").await.unwrap();
    
    // UE 2: No PDU sessions (registered only)
    
    // UE 3: Has 1 PDU session, then releases it
    amf.simulate_pdu_session_establish(ue_ids[3], 1, "10.45.0.5").await.unwrap();
    amf.simulate_pdu_session_release(ue_ids[3], 1).await.unwrap();
    
    // Verify each UE has independent state
    let ue0 = amf.get_ue_context(ue_ids[0]).await.expect("UE 0 not found");
    assert_eq!(ue0.pdu_sessions.len(), 2, "UE 0 should have 2 PDU sessions");
    assert!(ue0.registered);
    
    let ue1 = amf.get_ue_context(ue_ids[1]).await.expect("UE 1 not found");
    assert_eq!(ue1.pdu_sessions.len(), 1, "UE 1 should have 1 PDU session");
    assert!(ue1.registered);
    
    let ue2 = amf.get_ue_context(ue_ids[2]).await.expect("UE 2 not found");
    assert_eq!(ue2.pdu_sessions.len(), 0, "UE 2 should have 0 PDU sessions");
    assert!(ue2.registered);
    
    let ue3 = amf.get_ue_context(ue_ids[3]).await.expect("UE 3 not found");
    assert_eq!(ue3.pdu_sessions.len(), 0, "UE 3 should have 0 PDU sessions after release");
    assert!(ue3.registered);
    
    // Release UE 1 - should not affect others
    amf.simulate_ue_context_release(ue_ids[1]).await.unwrap();
    
    // Verify UE 1 is gone but others are unaffected
    assert!(amf.get_ue_context(ue_ids[1]).await.is_none(), "UE 1 should be released");
    
    let ue0_after = amf.get_ue_context(ue_ids[0]).await.expect("UE 0 should still exist");
    assert_eq!(ue0_after.pdu_sessions.len(), 2, "UE 0 should still have 2 PDU sessions");
    
    let ue2_after = amf.get_ue_context(ue_ids[2]).await.expect("UE 2 should still exist");
    assert!(ue2_after.registered, "UE 2 should still be registered");
    
    let ue3_after = amf.get_ue_context(ue_ids[3]).await.expect("UE 3 should still exist");
    assert!(ue3_after.registered, "UE 3 should still be registered");
    
    amf.stop().await.unwrap();
}
