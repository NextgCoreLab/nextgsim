//! PDU Session integration tests
//!
//! Tests PDU session establishment, modification, and release.

use integration_tests::{init_test_logging, MockAmf};

/// Test PDU session establishment
#[tokio::test]
async fn test_pdu_session_establishment() {
    init_test_logging();
    
    let amf = MockAmf::new();
    amf.start().await.expect("Failed to start mock AMF");
    
    // Setup: Register UE
    amf.simulate_gnb_connect(1).await.unwrap();
    amf.simulate_ng_setup_request(1).await.unwrap();
    let amf_ue_id = amf.simulate_initial_ue_message(1, vec![]).await.unwrap();
    amf.simulate_registration_complete(amf_ue_id, "imsi-001010000000001").await.unwrap();
    
    // Establish PDU session
    let upf_teid = amf.simulate_pdu_session_establish(amf_ue_id, 1, "10.45.0.2")
        .await
        .expect("Failed to establish PDU session");
    
    // Verify PDU session
    let ue = amf.get_ue_context(amf_ue_id).await.expect("UE context not found");
    assert_eq!(ue.pdu_sessions.len(), 1);
    assert_eq!(ue.pdu_sessions[0].psi, 1);
    assert_eq!(ue.pdu_sessions[0].ue_ip, Some("10.45.0.2".to_string()));
    assert_eq!(ue.pdu_sessions[0].upf_teid, upf_teid);
    
    amf.stop().await.unwrap();
}

/// Test multiple PDU sessions
#[tokio::test]
async fn test_multiple_pdu_sessions() {
    init_test_logging();
    
    let amf = MockAmf::new();
    amf.start().await.expect("Failed to start mock AMF");
    
    // Setup: Register UE
    amf.simulate_gnb_connect(1).await.unwrap();
    amf.simulate_ng_setup_request(1).await.unwrap();
    let amf_ue_id = amf.simulate_initial_ue_message(1, vec![]).await.unwrap();
    amf.simulate_registration_complete(amf_ue_id, "imsi-001010000000001").await.unwrap();
    
    // Establish multiple PDU sessions
    amf.simulate_pdu_session_establish(amf_ue_id, 1, "10.45.0.2").await.unwrap();
    amf.simulate_pdu_session_establish(amf_ue_id, 2, "10.45.0.3").await.unwrap();
    amf.simulate_pdu_session_establish(amf_ue_id, 3, "10.45.0.4").await.unwrap();
    
    // Verify all sessions
    let ue = amf.get_ue_context(amf_ue_id).await.expect("UE context not found");
    assert_eq!(ue.pdu_sessions.len(), 3);
    
    amf.stop().await.unwrap();
}

/// Test PDU session release
#[tokio::test]
async fn test_pdu_session_release() {
    init_test_logging();
    
    let amf = MockAmf::new();
    amf.start().await.expect("Failed to start mock AMF");
    
    // Setup: Register UE and establish session
    amf.simulate_gnb_connect(1).await.unwrap();
    amf.simulate_ng_setup_request(1).await.unwrap();
    let amf_ue_id = amf.simulate_initial_ue_message(1, vec![]).await.unwrap();
    amf.simulate_registration_complete(amf_ue_id, "imsi-001010000000001").await.unwrap();
    amf.simulate_pdu_session_establish(amf_ue_id, 1, "10.45.0.2").await.unwrap();
    
    // Release PDU session
    amf.simulate_pdu_session_release(amf_ue_id, 1)
        .await
        .expect("Failed to release PDU session");
    
    // Verify session removed
    let ue = amf.get_ue_context(amf_ue_id).await.expect("UE context not found");
    assert_eq!(ue.pdu_sessions.len(), 0);
    
    amf.stop().await.unwrap();
}

/// Test PDU session release with multiple sessions
#[tokio::test]
async fn test_pdu_session_selective_release() {
    init_test_logging();
    
    let amf = MockAmf::new();
    amf.start().await.expect("Failed to start mock AMF");
    
    // Setup: Register UE and establish multiple sessions
    amf.simulate_gnb_connect(1).await.unwrap();
    amf.simulate_ng_setup_request(1).await.unwrap();
    let amf_ue_id = amf.simulate_initial_ue_message(1, vec![]).await.unwrap();
    amf.simulate_registration_complete(amf_ue_id, "imsi-001010000000001").await.unwrap();
    amf.simulate_pdu_session_establish(amf_ue_id, 1, "10.45.0.2").await.unwrap();
    amf.simulate_pdu_session_establish(amf_ue_id, 2, "10.45.0.3").await.unwrap();
    
    // Release only session 1
    amf.simulate_pdu_session_release(amf_ue_id, 1).await.unwrap();
    
    // Verify only session 2 remains
    let ue = amf.get_ue_context(amf_ue_id).await.expect("UE context not found");
    assert_eq!(ue.pdu_sessions.len(), 1);
    assert_eq!(ue.pdu_sessions[0].psi, 2);
    
    amf.stop().await.unwrap();
}
