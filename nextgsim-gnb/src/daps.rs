//! DAPS (Dual Active Protocol Stack) Handover for gNB (Rel-17, TS 38.300 §10.1.2.4)
//!
//! DAPS maintains uplink data transmission on the source cell while receiving
//! downlink data on the target cell during handover, reducing interruption time.

use std::collections::HashMap;

/// DAPS handover state for a UE
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DapsState {
    /// DAPS not active
    Inactive,
    /// DAPS handover preparation (RRC Reconfiguration sent)
    Preparation,
    /// DAPS active: UE connected to both source and target cells
    Active,
    /// Target cell path established; releasing source
    ReleasingSource,
    /// DAPS complete: connected to target only
    Complete,
    /// DAPS failed; UE reconnected to source
    Failed,
}

/// DAPS RLC bearer configuration
#[derive(Debug, Clone)]
pub struct DapsRlcBearer {
    /// Radio Bearer ID
    pub rb_id: u8,
    /// Source cell C-RNTI
    pub source_c_rnti: u16,
    /// Target cell C-RNTI (assigned during preparation)
    pub target_c_rnti: Option<u16>,
    /// Whether this bearer is active on source (UL)
    pub source_active: bool,
    /// Whether this bearer is active on target (DL)
    pub target_active: bool,
}

impl DapsRlcBearer {
    pub fn new(rb_id: u8, source_c_rnti: u16) -> Self {
        Self {
            rb_id,
            source_c_rnti,
            target_c_rnti: None,
            source_active: true,
            target_active: false,
        }
    }
}

/// DAPS handover session for a UE
#[derive(Debug, Clone)]
pub struct DapsSession {
    /// UE C-RNTI at source cell
    pub source_c_rnti: u16,
    /// Source cell ID
    pub source_cell_id: i32,
    /// Target cell ID
    pub target_cell_id: i32,
    /// Target gNB ID (for inter-gNB DAPS)
    pub target_gnb_id: Option<u32>,
    /// Current DAPS state
    pub state: DapsState,
    /// RLC bearers active in DAPS mode
    pub bearers: Vec<DapsRlcBearer>,
    /// Whether downlink data forwarding to target is active
    pub dl_forwarding_active: bool,
    /// UE has confirmed DAPS activation
    pub ue_confirmed: bool,
}

impl DapsSession {
    pub fn new(source_c_rnti: u16, source_cell_id: i32, target_cell_id: i32) -> Self {
        Self {
            source_c_rnti,
            source_cell_id,
            target_cell_id,
            target_gnb_id: None,
            state: DapsState::Preparation,
            bearers: Vec::new(),
            dl_forwarding_active: false,
            ue_confirmed: false,
        }
    }

    /// Activates DAPS after UE RRC Reconfiguration Complete
    pub fn activate(&mut self) {
        self.state = DapsState::Active;
        self.dl_forwarding_active = true;
        for bearer in &mut self.bearers {
            bearer.target_active = true;
        }
    }

    /// Starts source release (target path fully established)
    pub fn start_source_release(&mut self) {
        self.state = DapsState::ReleasingSource;
        for bearer in &mut self.bearers {
            bearer.source_active = false;
        }
    }

    /// Completes DAPS handover
    pub fn complete(&mut self) {
        self.state = DapsState::Complete;
        self.dl_forwarding_active = false;
    }

    /// Marks DAPS as failed; UE stays on source
    pub fn fail(&mut self) {
        self.state = DapsState::Failed;
        self.dl_forwarding_active = false;
        for bearer in &mut self.bearers {
            bearer.source_active = true;
            bearer.target_active = false;
        }
    }

    /// Returns true if this session requires data forwarding
    pub fn needs_forwarding(&self) -> bool {
        self.dl_forwarding_active && self.state == DapsState::Active
    }
}

/// gNB DAPS manager: tracks all active DAPS sessions
#[derive(Debug, Default)]
pub struct DapsManager {
    /// Sessions keyed by source C-RNTI
    sessions: HashMap<u16, DapsSession>,
}

impl DapsManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new DAPS session for a UE
    pub fn create_session(&mut self, session: DapsSession) {
        self.sessions.insert(session.source_c_rnti, session);
    }

    /// Returns a session by source C-RNTI
    pub fn get(&self, source_c_rnti: u16) -> Option<&DapsSession> {
        self.sessions.get(&source_c_rnti)
    }

    /// Returns a session mutably
    pub fn get_mut(&mut self, source_c_rnti: u16) -> Option<&mut DapsSession> {
        self.sessions.get_mut(&source_c_rnti)
    }

    /// Removes a completed or failed session
    pub fn remove(&mut self, source_c_rnti: u16) -> Option<DapsSession> {
        self.sessions.remove(&source_c_rnti)
    }

    /// Returns count of active DAPS sessions
    pub fn active_count(&self) -> usize {
        self.sessions.values()
            .filter(|s| s.state == DapsState::Active)
            .count()
    }

    /// Returns sessions that need downlink forwarding
    pub fn forwarding_sessions(&self) -> Vec<&DapsSession> {
        self.sessions.values()
            .filter(|s| s.needs_forwarding())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_daps_session_lifecycle() {
        let mut s = DapsSession::new(0x0100, 1, 2);
        assert_eq!(s.state, DapsState::Preparation);
        s.activate();
        assert_eq!(s.state, DapsState::Active);
        assert!(s.needs_forwarding());
        s.start_source_release();
        assert_eq!(s.state, DapsState::ReleasingSource);
        s.complete();
        assert_eq!(s.state, DapsState::Complete);
        assert!(!s.needs_forwarding());
    }

    #[test]
    fn test_daps_session_fail() {
        let mut s = DapsSession::new(0x0100, 1, 2);
        s.bearers.push(DapsRlcBearer::new(1, 0x0100));
        s.activate();
        s.fail();
        assert_eq!(s.state, DapsState::Failed);
        assert!(s.bearers[0].source_active);
        assert!(!s.bearers[0].target_active);
    }

    #[test]
    fn test_daps_manager_active_count() {
        let mut mgr = DapsManager::new();
        let mut s1 = DapsSession::new(0x0100, 1, 2);
        let s2 = DapsSession::new(0x0200, 1, 3);
        s1.activate();
        mgr.create_session(s1);
        mgr.create_session(s2);
        assert_eq!(mgr.active_count(), 1);
    }

    #[test]
    fn test_daps_forwarding_sessions() {
        let mut mgr = DapsManager::new();
        let mut s = DapsSession::new(0x0100, 1, 2);
        s.activate();
        mgr.create_session(s);
        assert_eq!(mgr.forwarding_sessions().len(), 1);
    }
}
