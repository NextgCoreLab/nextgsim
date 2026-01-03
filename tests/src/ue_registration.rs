//! UE Registration integration tests
//!
//! Tests UE registration procedures with mock AMF.

use integration_tests::{init_test_logging, MockAmf};

/// Test basic UE registration flow
#[tokio::test]
async fn test_ue_registration_basic() {
    init_test_logging();
    
    // Create mock AMF
    let amf = MockAmf::new();
    amf.start().await.expect("Failed to start mock AMF");
    
    // Simulate gNB connection and NG Setup
    amf.simulate_gnb_connect(1).await.expect("Failed to connect gNB");
    amf.simulate_ng_setup_request(1).await.expect("Failed NG Setup");
    
    // Simulate Initial UE Message (registration request)
    let nas_pdu = vec![0x7e, 0x00, 0x41, 0x79, 0x00, 0x0d]; // Simplified registration request
    let amf_ue_id = amf.simulate_initial_ue_message(1, nas_pdu)
        .await
        .expect("Failed to process Initial UE Message");
    
    // Verify UE context created
    assert_eq!(amf.ue_count().await, 1);
    
    // Simulate registration completion
    amf.simulate_registration_complete(amf_ue_id, "imsi-001010000000001")
        .await
        .expect("Failed to complete registration");
    
    // Verify UE is registered
    let ue = amf.get_ue_context(amf_ue_id).await.expect("UE context not found");
    assert!(ue.registered);
    assert_eq!(ue.supi, Some("imsi-001010000000001".to_string()));
    
    amf.stop().await.expect("Failed to stop mock AMF");
}

/// Test UE deregistration
#[tokio::test]
async fn test_ue_deregistration() {
    init_test_logging();
    
    let amf = MockAmf::new();
    amf.start().await.expect("Failed to start mock AMF");
    
    // Setup: Register UE
    amf.simulate_gnb_connect(1).await.unwrap();
    amf.simulate_ng_setup_request(1).await.unwrap();
    let amf_ue_id = amf.simulate_initial_ue_message(1, vec![]).await.unwrap();
    amf.simulate_registration_complete(amf_ue_id, "imsi-001010000000001").await.unwrap();
    
    // Deregister UE
    amf.simulate_ue_context_release(amf_ue_id).await.expect("Failed to release UE context");
    
    // Verify UE context removed
    assert_eq!(amf.ue_count().await, 0);
    assert!(amf.get_ue_context(amf_ue_id).await.is_none());
    
    amf.stop().await.unwrap();
}

/// Test gNB disconnection during registration
#[tokio::test]
async fn test_gnb_disconnect_during_registration() {
    init_test_logging();
    
    let amf = MockAmf::new();
    amf.start().await.expect("Failed to start mock AMF");
    
    // Connect gNB and start registration
    amf.simulate_gnb_connect(1).await.unwrap();
    amf.simulate_ng_setup_request(1).await.unwrap();
    let _amf_ue_id = amf.simulate_initial_ue_message(1, vec![]).await.unwrap();
    
    // Disconnect gNB
    amf.simulate_gnb_disconnect(1).await.expect("Failed to disconnect gNB");
    
    // Verify gNB removed
    assert_eq!(amf.gnb_count().await, 0);
    
    amf.stop().await.unwrap();
}
