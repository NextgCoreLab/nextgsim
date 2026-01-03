//! UE Context Management for NGAP
//!
//! This module manages UE contexts within the NGAP task. Each UE has an associated
//! context that tracks:
//! - NGAP IDs (RAN UE NGAP ID and AMF UE NGAP ID)
//! - Associated AMF
//! - PDU sessions
//! - UE state

use std::collections::HashMap;

/// UE state within NGAP
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UeState {
    /// Initial state, waiting for Initial Context Setup
    Initial,
    /// Initial Context Setup in progress
    InitialContextSetup,
    /// UE context is established and active
    Active,
    /// UE context release in progress
    Releasing,
}

impl Default for UeState {
    fn default() -> Self {
        UeState::Initial
    }
}

impl std::fmt::Display for UeState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UeState::Initial => write!(f, "Initial"),
            UeState::InitialContextSetup => write!(f, "InitialContextSetup"),
            UeState::Active => write!(f, "Active"),
            UeState::Releasing => write!(f, "Releasing"),
        }
    }
}

/// PDU session information within NGAP UE context
#[derive(Debug, Clone)]
pub struct NgapPduSession {
    /// PDU Session ID (1-15)
    pub psi: u8,
    /// QoS Flow Identifier
    pub qfi: Option<u8>,
    /// Uplink TEID (gNB -> UPF)
    pub uplink_teid: u32,
    /// Downlink TEID (UPF -> gNB)
    pub downlink_teid: u32,
    /// UPF transport layer address
    pub upf_address: std::net::IpAddr,
}

/// NGAP UE context
#[derive(Debug, Clone)]
pub struct NgapUeContext {
    /// UE ID (internal, same as RRC UE ID)
    pub ue_id: i32,
    /// RAN UE NGAP ID (assigned by gNB)
    pub ran_ue_ngap_id: i64,
    /// AMF UE NGAP ID (assigned by AMF)
    pub amf_ue_ngap_id: Option<i64>,
    /// Associated AMF context ID
    pub amf_ctx_id: i32,
    /// SCTP stream ID for this UE
    pub stream_id: u16,
    /// Current state
    pub state: UeState,
    /// PDU sessions indexed by PSI
    pub pdu_sessions: HashMap<u8, NgapPduSession>,
    /// Requested NSSAI (slice type)
    pub requested_nssai: Option<i32>,
}

impl NgapUeContext {
    /// Creates a new UE context
    pub fn new(ue_id: i32, ran_ue_ngap_id: i64, amf_ctx_id: i32, stream_id: u16) -> Self {
        Self {
            ue_id,
            ran_ue_ngap_id,
            amf_ue_ngap_id: None,
            amf_ctx_id,
            stream_id,
            state: UeState::Initial,
            pdu_sessions: HashMap::new(),
            requested_nssai: None,
        }
    }

    /// Sets the AMF UE NGAP ID
    pub fn set_amf_ue_ngap_id(&mut self, id: i64) {
        self.amf_ue_ngap_id = Some(id);
    }

    /// Returns true if the UE has both RAN and AMF NGAP IDs
    pub fn has_ngap_id_pair(&self) -> bool {
        self.amf_ue_ngap_id.is_some()
    }

    /// Transitions to Initial Context Setup state
    pub fn on_initial_context_setup(&mut self) {
        self.state = UeState::InitialContextSetup;
    }

    /// Transitions to Active state
    pub fn on_context_setup_complete(&mut self) {
        self.state = UeState::Active;
    }

    /// Transitions to Releasing state
    pub fn on_context_release(&mut self) {
        self.state = UeState::Releasing;
    }

    /// Adds a PDU session
    pub fn add_pdu_session(&mut self, session: NgapPduSession) {
        self.pdu_sessions.insert(session.psi, session);
    }

    /// Removes a PDU session
    pub fn remove_pdu_session(&mut self, psi: u8) -> Option<NgapPduSession> {
        self.pdu_sessions.remove(&psi)
    }

    /// Gets a PDU session by PSI
    pub fn get_pdu_session(&self, psi: u8) -> Option<&NgapPduSession> {
        self.pdu_sessions.get(&psi)
    }

    /// Returns the number of active PDU sessions
    pub fn pdu_session_count(&self) -> usize {
        self.pdu_sessions.len()
    }

    /// Returns true if the UE is in an active state
    pub fn is_active(&self) -> bool {
        self.state == UeState::Active
    }
}

/// NGAP ID pair for UE identification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NgapIdPair {
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: i64,
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: i64,
}

impl NgapIdPair {
    /// Creates a new NGAP ID pair
    pub fn new(ran_ue_ngap_id: i64, amf_ue_ngap_id: i64) -> Self {
        Self {
            ran_ue_ngap_id,
            amf_ue_ngap_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ue_context_new() {
        let ctx = NgapUeContext::new(1, 100, 1, 1);
        assert_eq!(ctx.ue_id, 1);
        assert_eq!(ctx.ran_ue_ngap_id, 100);
        assert_eq!(ctx.amf_ctx_id, 1);
        assert_eq!(ctx.stream_id, 1);
        assert_eq!(ctx.state, UeState::Initial);
        assert!(ctx.amf_ue_ngap_id.is_none());
        assert!(!ctx.has_ngap_id_pair());
    }

    #[test]
    fn test_ue_context_state_transitions() {
        let mut ctx = NgapUeContext::new(1, 100, 1, 1);
        ctx.set_amf_ue_ngap_id(200);
        assert!(ctx.has_ngap_id_pair());

        ctx.on_initial_context_setup();
        assert_eq!(ctx.state, UeState::InitialContextSetup);

        ctx.on_context_setup_complete();
        assert_eq!(ctx.state, UeState::Active);
        assert!(ctx.is_active());

        ctx.on_context_release();
        assert_eq!(ctx.state, UeState::Releasing);
        assert!(!ctx.is_active());
    }

    #[test]
    fn test_ue_context_pdu_sessions() {
        let mut ctx = NgapUeContext::new(1, 100, 1, 1);

        let session = NgapPduSession {
            psi: 1,
            qfi: Some(1),
            uplink_teid: 0x12345678,
            downlink_teid: 0x87654321,
            upf_address: "10.0.0.1".parse().unwrap(),
        };

        ctx.add_pdu_session(session.clone());
        assert_eq!(ctx.pdu_session_count(), 1);

        let retrieved = ctx.get_pdu_session(1);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().psi, 1);

        let removed = ctx.remove_pdu_session(1);
        assert!(removed.is_some());
        assert_eq!(ctx.pdu_session_count(), 0);
    }

    #[test]
    fn test_ngap_id_pair() {
        let pair = NgapIdPair::new(100, 200);
        assert_eq!(pair.ran_ue_ngap_id, 100);
        assert_eq!(pair.amf_ue_ngap_id, 200);
    }
}
