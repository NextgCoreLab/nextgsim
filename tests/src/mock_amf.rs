//! Mock AMF for integration testing
//!
//! Provides a simplified AMF implementation for testing UE registration,
//! authentication, and PDU session management.

use crate::test_fixtures::{TestAmfConfig, Guami, PlmnSupport};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex, RwLock};
use thiserror::Error;

/// Mock AMF errors
#[derive(Debug, Error)]
pub enum MockAmfError {
    #[error("AMF not started")]
    NotStarted,
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
    #[error("NGAP encoding error: {0}")]
    NgapEncodingError(String),
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Events emitted by the mock AMF
#[derive(Debug, Clone)]
pub enum MockAmfEvent {
    /// gNB connected via SCTP
    GnbConnected { gnb_id: u32 },
    /// gNB disconnected
    GnbDisconnected { gnb_id: u32 },
    /// NG Setup Request received
    NgSetupReceived { gnb_id: u32 },
    /// Initial UE Message received
    InitialUeMessage { ran_ue_ngap_id: u32, nas_pdu: Vec<u8> },
    /// Uplink NAS Transport received
    UplinkNasTransport { amf_ue_ngap_id: u64, nas_pdu: Vec<u8> },
    /// UE Context Release Request received
    UeContextReleaseRequest { amf_ue_ngap_id: u64 },
}

/// Mock AMF configuration
#[derive(Debug, Clone)]
pub struct MockAmfConfig {
    /// AMF name
    pub amf_name: String,
    /// Served GUAMIs
    pub served_guamis: Vec<Guami>,
    /// Supported PLMNs
    pub plmn_support: Vec<PlmnSupport>,
    /// SCTP listen address
    pub sctp_addr: SocketAddr,
    /// Relative AMF capacity
    pub relative_capacity: u8,
    /// Auto-respond to NG Setup
    pub auto_ng_setup_response: bool,
    /// Auto-respond to Initial UE Message with authentication
    pub auto_authentication: bool,
}

impl Default for MockAmfConfig {
    fn default() -> Self {
        let test_config = TestAmfConfig::default();
        Self {
            amf_name: test_config.amf_name,
            served_guamis: test_config.served_guamis,
            plmn_support: test_config.plmn_support,
            sctp_addr: test_config.sctp_addr,
            relative_capacity: test_config.relative_capacity,
            auto_ng_setup_response: true,
            auto_authentication: true,
        }
    }
}

/// UE context in the mock AMF
#[derive(Debug, Clone)]
pub struct MockUeContext {
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// SUPI (if known)
    pub supi: Option<String>,
    /// Registration state
    pub registered: bool,
    /// PDU sessions
    pub pdu_sessions: Vec<MockPduSession>,
}

/// PDU session in the mock AMF
#[derive(Debug, Clone)]
pub struct MockPduSession {
    /// PDU Session ID
    pub psi: u8,
    /// Session type (IPv4, IPv6, IPv4v6)
    pub session_type: u8,
    /// UE IP address (if assigned)
    pub ue_ip: Option<String>,
    /// UPF TEID
    pub upf_teid: u32,
    /// gNB TEID
    pub gnb_teid: Option<u32>,
}

/// Mock AMF state
struct MockAmfState {
    /// Connected gNBs
    gnbs: HashMap<u32, GnbContext>,
    /// UE contexts indexed by AMF UE NGAP ID
    ue_contexts: HashMap<u64, MockUeContext>,
    /// Next AMF UE NGAP ID
    next_amf_ue_ngap_id: u64,
    /// Next UPF TEID
    next_upf_teid: u32,
}

/// gNB context in the mock AMF
#[derive(Debug)]
#[allow(dead_code)]
struct GnbContext {
    gnb_id: u32,
    ng_setup_complete: bool,
}

/// Mock AMF for integration testing
pub struct MockAmf {
    config: MockAmfConfig,
    state: Arc<RwLock<MockAmfState>>,
    event_tx: mpsc::Sender<MockAmfEvent>,
    event_rx: Arc<Mutex<mpsc::Receiver<MockAmfEvent>>>,
    running: Arc<RwLock<bool>>,
}

impl MockAmf {
    /// Create a new mock AMF with default configuration
    pub fn new() -> Self {
        Self::with_config(MockAmfConfig::default())
    }

    /// Create a new mock AMF with custom configuration
    pub fn with_config(config: MockAmfConfig) -> Self {
        let (event_tx, event_rx) = mpsc::channel(100);
        
        Self {
            config,
            state: Arc::new(RwLock::new(MockAmfState {
                gnbs: HashMap::new(),
                ue_contexts: HashMap::new(),
                next_amf_ue_ngap_id: 1,
                next_upf_teid: 0x10000000,
            })),
            event_tx,
            event_rx: Arc::new(Mutex::new(event_rx)),
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// Get the SCTP listen address
    pub fn sctp_addr(&self) -> SocketAddr {
        self.config.sctp_addr
    }

    /// Check if the mock AMF is running
    pub async fn is_running(&self) -> bool {
        *self.running.read().await
    }

    /// Start the mock AMF
    ///
    /// Note: This is a placeholder. In a full implementation, this would
    /// start an SCTP server to accept gNB connections.
    pub async fn start(&self) -> Result<(), MockAmfError> {
        let mut running = self.running.write().await;
        *running = true;
        tracing::info!("Mock AMF started on {}", self.config.sctp_addr);
        Ok(())
    }

    /// Stop the mock AMF
    pub async fn stop(&self) -> Result<(), MockAmfError> {
        let mut running = self.running.write().await;
        *running = false;
        tracing::info!("Mock AMF stopped");
        Ok(())
    }

    /// Wait for the next event
    pub async fn next_event(&self) -> Option<MockAmfEvent> {
        let mut rx = self.event_rx.lock().await;
        rx.recv().await
    }

    /// Get the number of connected gNBs
    pub async fn gnb_count(&self) -> usize {
        self.state.read().await.gnbs.len()
    }

    /// Get the number of registered UEs
    pub async fn ue_count(&self) -> usize {
        self.state.read().await.ue_contexts.len()
    }

    /// Get UE context by AMF UE NGAP ID
    pub async fn get_ue_context(&self, amf_ue_ngap_id: u64) -> Option<MockUeContext> {
        self.state.read().await.ue_contexts.get(&amf_ue_ngap_id).cloned()
    }

    /// Simulate gNB connection
    pub async fn simulate_gnb_connect(&self, gnb_id: u32) -> Result<(), MockAmfError> {
        let mut state = self.state.write().await;
        state.gnbs.insert(gnb_id, GnbContext {
            gnb_id,
            ng_setup_complete: false,
        });
        
        let _ = self.event_tx.send(MockAmfEvent::GnbConnected { gnb_id }).await;
        Ok(())
    }

    /// Simulate gNB disconnection
    pub async fn simulate_gnb_disconnect(&self, gnb_id: u32) -> Result<(), MockAmfError> {
        let mut state = self.state.write().await;
        state.gnbs.remove(&gnb_id);
        
        let _ = self.event_tx.send(MockAmfEvent::GnbDisconnected { gnb_id }).await;
        Ok(())
    }

    /// Simulate receiving NG Setup Request
    pub async fn simulate_ng_setup_request(&self, gnb_id: u32) -> Result<(), MockAmfError> {
        let mut state = self.state.write().await;
        if let Some(gnb) = state.gnbs.get_mut(&gnb_id) {
            gnb.ng_setup_complete = true;
        }
        
        let _ = self.event_tx.send(MockAmfEvent::NgSetupReceived { gnb_id }).await;
        Ok(())
    }

    /// Simulate receiving Initial UE Message
    pub async fn simulate_initial_ue_message(
        &self,
        ran_ue_ngap_id: u32,
        nas_pdu: Vec<u8>,
    ) -> Result<u64, MockAmfError> {
        let mut state = self.state.write().await;
        let amf_ue_ngap_id = state.next_amf_ue_ngap_id;
        state.next_amf_ue_ngap_id += 1;
        
        state.ue_contexts.insert(amf_ue_ngap_id, MockUeContext {
            ran_ue_ngap_id,
            amf_ue_ngap_id,
            supi: None,
            registered: false,
            pdu_sessions: Vec::new(),
        });
        
        let _ = self.event_tx.send(MockAmfEvent::InitialUeMessage {
            ran_ue_ngap_id,
            nas_pdu,
        }).await;
        
        Ok(amf_ue_ngap_id)
    }

    /// Simulate UE registration completion
    pub async fn simulate_registration_complete(
        &self,
        amf_ue_ngap_id: u64,
        supi: &str,
    ) -> Result<(), MockAmfError> {
        let mut state = self.state.write().await;
        if let Some(ue) = state.ue_contexts.get_mut(&amf_ue_ngap_id) {
            ue.supi = Some(supi.to_string());
            ue.registered = true;
        }
        Ok(())
    }

    /// Simulate PDU session establishment
    pub async fn simulate_pdu_session_establish(
        &self,
        amf_ue_ngap_id: u64,
        psi: u8,
        ue_ip: &str,
    ) -> Result<u32, MockAmfError> {
        let mut state = self.state.write().await;
        let upf_teid = state.next_upf_teid;
        state.next_upf_teid += 1;
        
        if let Some(ue) = state.ue_contexts.get_mut(&amf_ue_ngap_id) {
            ue.pdu_sessions.push(MockPduSession {
                psi,
                session_type: 1, // IPv4
                ue_ip: Some(ue_ip.to_string()),
                upf_teid,
                gnb_teid: None,
            });
        }
        
        Ok(upf_teid)
    }

    /// Simulate PDU session release
    pub async fn simulate_pdu_session_release(
        &self,
        amf_ue_ngap_id: u64,
        psi: u8,
    ) -> Result<(), MockAmfError> {
        let mut state = self.state.write().await;
        if let Some(ue) = state.ue_contexts.get_mut(&amf_ue_ngap_id) {
            ue.pdu_sessions.retain(|s| s.psi != psi);
        }
        Ok(())
    }

    /// Simulate UE context release
    pub async fn simulate_ue_context_release(
        &self,
        amf_ue_ngap_id: u64,
    ) -> Result<(), MockAmfError> {
        let mut state = self.state.write().await;
        state.ue_contexts.remove(&amf_ue_ngap_id);
        Ok(())
    }
}

impl Default for MockAmf {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_amf_creation() {
        let amf = MockAmf::new();
        assert!(!amf.is_running().await);
    }

    #[tokio::test]
    async fn test_mock_amf_start_stop() {
        let amf = MockAmf::new();
        
        amf.start().await.unwrap();
        assert!(amf.is_running().await);
        
        amf.stop().await.unwrap();
        assert!(!amf.is_running().await);
    }

    #[tokio::test]
    async fn test_gnb_connect_disconnect() {
        let amf = MockAmf::new();
        amf.start().await.unwrap();
        
        amf.simulate_gnb_connect(1).await.unwrap();
        assert_eq!(amf.gnb_count().await, 1);
        
        amf.simulate_gnb_disconnect(1).await.unwrap();
        assert_eq!(amf.gnb_count().await, 0);
    }

    #[tokio::test]
    async fn test_ue_registration() {
        let amf = MockAmf::new();
        amf.start().await.unwrap();
        
        let amf_ue_id = amf.simulate_initial_ue_message(1, vec![0x7e, 0x00, 0x41]).await.unwrap();
        assert_eq!(amf.ue_count().await, 1);
        
        amf.simulate_registration_complete(amf_ue_id, "imsi-001010000000001").await.unwrap();
        
        let ue = amf.get_ue_context(amf_ue_id).await.unwrap();
        assert!(ue.registered);
        assert_eq!(ue.supi, Some("imsi-001010000000001".to_string()));
    }

    #[tokio::test]
    async fn test_pdu_session() {
        let amf = MockAmf::new();
        amf.start().await.unwrap();
        
        let amf_ue_id = amf.simulate_initial_ue_message(1, vec![]).await.unwrap();
        amf.simulate_registration_complete(amf_ue_id, "imsi-001010000000001").await.unwrap();
        
        let upf_teid = amf.simulate_pdu_session_establish(amf_ue_id, 1, "10.45.0.2").await.unwrap();
        
        let ue = amf.get_ue_context(amf_ue_id).await.unwrap();
        assert_eq!(ue.pdu_sessions.len(), 1);
        assert_eq!(ue.pdu_sessions[0].psi, 1);
        assert_eq!(ue.pdu_sessions[0].upf_teid, upf_teid);
        
        amf.simulate_pdu_session_release(amf_ue_id, 1).await.unwrap();
        
        let ue = amf.get_ue_context(amf_ue_id).await.unwrap();
        assert_eq!(ue.pdu_sessions.len(), 0);
    }
}
