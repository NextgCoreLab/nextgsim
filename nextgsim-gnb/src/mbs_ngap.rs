//! MBS (Multicast-Broadcast Services) NGAP Session Procedures for gNB (Rel-17, TS 38.413 §8.21)
//!
//! Implements gNB-side MBS NGAP signaling:
//! - MBSSessionStart / MBSSessionStartResponse
//! - MBSSessionStop / MBSSessionStopResponse
//! - MBSSessionProgressTransfer for MCCH broadcast
//! - Per-cell MBS area session tracking

use std::collections::{HashMap, HashSet};

/// MBS session state at gNB
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GnbMbsState {
    /// Session setup in progress
    Starting,
    /// Session active, broadcasting/multicasting
    Active,
    /// Session stop in progress
    Stopping,
    /// Session terminated
    Stopped,
    /// Session setup failed
    Failed,
}

/// MBS flow (individual content stream within an MBS session)
#[derive(Debug, Clone)]
pub struct MbsFlow {
    /// MBS flow ID
    pub flow_id: u8,
    /// Data radio bearer ID for this flow
    pub mrb_id: u8,
    /// QoS flow indicator
    pub qfi: u8,
    /// Whether PDCP is active for this flow
    pub pdcp_active: bool,
}

/// GTP-U tunnel for MBS downlink
#[derive(Debug, Clone)]
pub struct MbsTunnel {
    /// TEID at UPF (data source)
    pub upf_teid: u32,
    /// UPF IP address
    pub upf_ip: [u8; 4],
    /// TEID at gNB (local receive)
    pub gnb_teid: u32,
}

/// MBS session context at gNB
#[derive(Debug, Clone)]
pub struct GnbMbsSession {
    /// MBS session ID (from AMF/SMF)
    pub mbs_session_id: String,
    /// TMGI (6 octets)
    pub tmgi: [u8; 6],
    /// Area session ID
    pub area_session_id: Option<u32>,
    /// Session state
    pub state: GnbMbsState,
    /// MBS flows in this session
    pub flows: Vec<MbsFlow>,
    /// GTP-U tunnel for downlink
    pub tunnel: Option<MbsTunnel>,
    /// Cell IDs that are broadcasting this session
    pub active_cells: HashSet<i32>,
    /// UE C-RNTIs that have joined (multicast mode)
    pub joined_ues: HashSet<u16>,
    /// Whether this is a broadcast session (no per-UE membership)
    pub is_broadcast: bool,
}

impl GnbMbsSession {
    pub fn new_broadcast(mbs_session_id: String, tmgi: [u8; 6]) -> Self {
        Self {
            mbs_session_id,
            tmgi,
            area_session_id: None,
            state: GnbMbsState::Starting,
            flows: Vec::new(),
            tunnel: None,
            active_cells: HashSet::new(),
            joined_ues: HashSet::new(),
            is_broadcast: true,
        }
    }

    pub fn new_multicast(mbs_session_id: String, tmgi: [u8; 6]) -> Self {
        let mut s = Self::new_broadcast(mbs_session_id, tmgi);
        s.is_broadcast = false;
        s
    }

    /// Activates the session in a cell
    pub fn activate_cell(&mut self, cell_id: i32) {
        self.active_cells.insert(cell_id);
        self.state = GnbMbsState::Active;
    }

    /// Deactivates the session in a cell
    pub fn deactivate_cell(&mut self, cell_id: i32) {
        self.active_cells.remove(&cell_id);
        if self.active_cells.is_empty() {
            self.state = GnbMbsState::Stopping;
        }
    }

    /// UE joins multicast session
    pub fn ue_join(&mut self, c_rnti: u16) -> bool {
        if self.is_broadcast {
            return false;
        }
        self.joined_ues.insert(c_rnti)
    }

    /// UE leaves multicast session
    pub fn ue_leave(&mut self, c_rnti: u16) -> bool {
        self.joined_ues.remove(&c_rnti)
    }

    /// Returns TMGI as hex string
    pub fn tmgi_hex(&self) -> String {
        self.tmgi.iter().map(|b| format!("{b:02X}")).collect()
    }

    pub fn is_active(&self) -> bool {
        self.state == GnbMbsState::Active
    }
}

/// NGAP MBS session manager at gNB
#[derive(Debug, Default)]
pub struct NgapMbsManager {
    /// Active MBS sessions keyed by session ID
    sessions: HashMap<String, GnbMbsSession>,
}

impl NgapMbsManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Processes MBSSessionStart from AMF
    pub fn start_session(&mut self, session: GnbMbsSession) -> &GnbMbsSession {
        let id = session.mbs_session_id.clone();
        self.sessions.insert(id.clone(), session);
        &self.sessions[&id]
    }

    /// Processes MBSSessionStop from AMF
    pub fn stop_session(&mut self, session_id: &str) -> Option<GnbMbsSession> {
        if let Some(s) = self.sessions.get_mut(session_id) {
            s.state = GnbMbsState::Stopped;
        }
        self.sessions.remove(session_id)
    }

    /// Gets a session mutably
    pub fn get_mut(&mut self, session_id: &str) -> Option<&mut GnbMbsSession> {
        self.sessions.get_mut(session_id)
    }

    /// Returns all sessions active on a given cell
    pub fn sessions_for_cell(&self, cell_id: i32) -> Vec<&GnbMbsSession> {
        self.sessions
            .values()
            .filter(|s| s.active_cells.contains(&cell_id))
            .collect()
    }

    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_session() -> GnbMbsSession {
        GnbMbsSession::new_multicast("mbs-001".into(), [0xAB, 0xCD, 0xEF, 0x01, 0x02, 0x03])
    }

    #[test]
    fn test_mbs_session_activate() {
        let mut s = test_session();
        assert_eq!(s.state, GnbMbsState::Starting);
        s.activate_cell(1);
        assert_eq!(s.state, GnbMbsState::Active);
    }

    #[test]
    fn test_mbs_session_last_cell_deactivate() {
        let mut s = test_session();
        s.activate_cell(1);
        s.deactivate_cell(1);
        assert_eq!(s.state, GnbMbsState::Stopping);
    }

    #[test]
    fn test_multicast_ue_join_leave() {
        let mut s = test_session();
        assert!(s.ue_join(0x0100));
        assert_eq!(s.joined_ues.len(), 1);
        assert!(s.ue_leave(0x0100));
        assert!(s.joined_ues.is_empty());
    }

    #[test]
    fn test_broadcast_ue_join_rejected() {
        let mut s = GnbMbsSession::new_broadcast("bc-001".into(), [0; 6]);
        assert!(!s.ue_join(0x0100));
    }

    #[test]
    fn test_manager_start_stop() {
        let mut mgr = NgapMbsManager::new();
        mgr.start_session(test_session());
        assert_eq!(mgr.session_count(), 1);
        mgr.stop_session("mbs-001");
        assert_eq!(mgr.session_count(), 0);
    }

    #[test]
    fn test_sessions_for_cell() {
        let mut mgr = NgapMbsManager::new();
        let mut s = test_session();
        s.activate_cell(5);
        mgr.start_session(s);
        assert_eq!(mgr.sessions_for_cell(5).len(), 1);
        assert!(mgr.sessions_for_cell(99).is_empty());
    }
}
