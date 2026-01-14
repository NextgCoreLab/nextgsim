//! User Plane data flow integration tests
//!
//! Tests GTP-U data flow between UE and network.

use integration_tests::init_test_logging;
use nextgsim_gtp::{GtpHeader, GtpMessageType, GtpTunnel, PduSession, TunnelManager};
use bytes::Bytes;

/// Test GTP-U encapsulation
#[tokio::test]
async fn test_gtp_encapsulation() {
    init_test_logging();
    
    let mut tunnel_mgr = TunnelManager::new();
    
    // Create a PDU session
    let ue_id = 1u32;
    let psi = 1u8;
    let uplink_teid = 0x10000001;
    let downlink_teid = 0x20000001;
    let upf_addr = "10.0.0.1:2152".parse().unwrap();
    let gnb_addr = "10.0.0.2:2152".parse().unwrap();
    
    let session = PduSession::new(
        ue_id,
        psi,
        GtpTunnel::new(uplink_teid, upf_addr),
        GtpTunnel::new(downlink_teid, gnb_addr),
    ).with_qfi(1);
    
    tunnel_mgr.create_session(session).expect("Failed to create session");
    
    // Test uplink encapsulation
    let user_data = Bytes::from_static(&[0x45, 0x00, 0x00, 0x1c]); // IP header start
    let (gtp_header, dest_addr) = tunnel_mgr.encapsulate_uplink(ue_id, psi, user_data.clone())
        .expect("Failed to encapsulate uplink");
    
    // Verify GTP header
    assert_eq!(gtp_header.message_type, GtpMessageType::GPdu);
    assert_eq!(gtp_header.teid, uplink_teid);
    assert_eq!(gtp_header.payload, user_data);
    assert_eq!(dest_addr, upf_addr);
}

/// Test GTP-U decapsulation
#[tokio::test]
async fn test_gtp_decapsulation() {
    init_test_logging();
    
    let mut tunnel_mgr = TunnelManager::new();
    
    // Create a PDU session
    let ue_id = 1u32;
    let psi = 1u8;
    let uplink_teid = 0x10000001;
    let downlink_teid = 0x20000001;
    let upf_addr = "10.0.0.1:2152".parse().unwrap();
    let gnb_addr = "10.0.0.2:2152".parse().unwrap();
    
    let session = PduSession::new(
        ue_id,
        psi,
        GtpTunnel::new(uplink_teid, upf_addr),
        GtpTunnel::new(downlink_teid, gnb_addr),
    ).with_qfi(1);
    
    tunnel_mgr.create_session(session).expect("Failed to create session");
    
    // Create downlink GTP-U packet
    let user_data = Bytes::from_static(&[0x45, 0x00, 0x00, 0x1c, 0x00, 0x01]);
    let gtp_header = GtpHeader::g_pdu(downlink_teid, user_data.clone());
    
    // Test decapsulation
    let (session_ue_id, session_psi, payload) = tunnel_mgr.decapsulate_downlink(&gtp_header)
        .expect("Failed to decapsulate downlink");
    
    assert_eq!(session_ue_id, ue_id);
    assert_eq!(session_psi, psi);
    assert_eq!(*payload, user_data);
}

/// Test tunnel management
#[tokio::test]
async fn test_tunnel_management() {
    init_test_logging();
    
    let mut tunnel_mgr = TunnelManager::new();
    
    // Create multiple sessions for same UE
    let ue_id = 1u32;
    let upf_addr = "10.0.0.1:2152".parse().unwrap();
    let gnb_addr = "10.0.0.2:2152".parse().unwrap();
    
    let session1 = PduSession::new(
        ue_id, 1,
        GtpTunnel::new(0x10000001, upf_addr),
        GtpTunnel::new(0x20000001, gnb_addr),
    ).with_qfi(1);
    
    let session2 = PduSession::new(
        ue_id, 2,
        GtpTunnel::new(0x10000002, upf_addr),
        GtpTunnel::new(0x20000002, gnb_addr),
    ).with_qfi(2);
    
    tunnel_mgr.create_session(session1).unwrap();
    tunnel_mgr.create_session(session2).unwrap();
    
    // Verify sessions exist
    assert!(tunnel_mgr.get_session(ue_id, 1).is_some());
    assert!(tunnel_mgr.get_session(ue_id, 2).is_some());
    
    // Delete one session
    tunnel_mgr.delete_session(ue_id, 1).expect("Failed to delete session");
    
    // Verify only session 2 remains
    assert!(tunnel_mgr.get_session(ue_id, 1).is_none());
    assert!(tunnel_mgr.get_session(ue_id, 2).is_some());
    
    // Delete all sessions for UE
    tunnel_mgr.delete_sessions_for_ue(ue_id);
    assert!(tunnel_mgr.get_session(ue_id, 2).is_none());
}

/// Test GTP-U echo request/response
#[tokio::test]
async fn test_gtp_echo() {
    init_test_logging();
    
    // Create echo request
    let seq_num = 1234u16;
    let echo_req = GtpHeader::echo_request(0)
        .with_sequence_number(seq_num);
    
    // Verify
    assert_eq!(echo_req.message_type, GtpMessageType::EchoRequest);
    assert_eq!(echo_req.sequence_number, Some(seq_num));
    
    // Create echo response
    let echo_resp = GtpHeader::echo_response(0)
        .with_sequence_number(seq_num);
    
    assert_eq!(echo_resp.message_type, GtpMessageType::EchoResponse);
    assert_eq!(echo_resp.sequence_number, Some(seq_num));
}
