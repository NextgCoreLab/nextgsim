//! MBS (Multicast-Broadcast Services) session state at the gNB (Rel-17)
//!
//! This is the state machine the NGAP task drives when an MBS procedure arrives
//! on the NG-C association. The wire codec lives in
//! `nextgsim_ngap::procedures::mbs`; this module holds only what the gNB
//! remembers between messages:
//!
//! - which MBS sessions exist, keyed by the session id the AMF sent;
//! - which cells are currently radiating each session;
//! - which UEs have joined a multicast (as opposed to broadcast) session.
//!
//! # Which procedures reach it
//!
//! `NgapMbsManager` is driven from `NgapTask::handle_ngap_pdu` for the three
//! AMF-initiated MBS procedures of TS 38.413 §9.2.9:
//!
//! - 71 `id-MulticastSessionActivation` -> [`NgapMbsManager::start_session`],
//!   answered with a `MulticastSessionActivationResponse`;
//! - 72 `id-MulticastSessionDeactivation` -> [`NgapMbsManager::stop_session`],
//!   answered with a `MulticastSessionDeactivationResponse`;
//! - 74 `id-MulticastGroupPaging` -> [`NgapMbsManager::sessions_for_cell`] to
//!   decide whether this cell carries the paged group.
//!
//! Procedure 68 (`id-BroadcastSessionSetup`) is deliberately NOT routed here.
//! It is a *broadcast* session setup whose `BroadcastSessionSetupRequest`
//! carries `MBS-ServiceArea` and `MBS-SessionTNLInfo5GC` — an MB-UPF tunnel this
//! state machine has no field for — so accepting it would record a session the
//! gNB cannot actually deliver. See `handle_unroutable_pdu`: it is answered with
//! an Error Indication, which is the conformant response to a procedure the node
//! does not support (TS 38.413 §8.7.5).
//!
//! # Session identity
//!
//! Sessions are keyed by the 6 TMGI octets as they arrived, not by a formatted
//! string, so a lookup cannot miss because two code paths disagreed on hex case
//! or on which half of the TMGI comes first (TS 23.003 §30.2).

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
///
/// Driven by `NgapTask::handle_ngap_pdu` for procedures 71, 72 and 74; see the
/// module docs.
#[derive(Debug, Default)]
pub struct NgapMbsManager {
    /// Active MBS sessions keyed by the 6 TMGI octets.
    ///
    /// Keyed by the raw octets rather than by `mbs_session_id`'s formatted text
    /// because the TMGI is what the AMF actually puts in `id-MBS-SessionID`
    /// (299) on every MBS message, so it is the only identifier a deactivation
    /// or a group paging is guaranteed to arrive with.
    sessions: HashMap<[u8; 6], GnbMbsSession>,
}

impl NgapMbsManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Records the session a `MulticastSessionActivationRequest` activated
    /// (TS 38.413 §9.2.9.1).
    ///
    /// Re-activating an existing TMGI replaces the session, which is what the
    /// AMF asking again means: the previous context is stale.
    pub fn start_session(&mut self, session: GnbMbsSession) -> &GnbMbsSession {
        let tmgi = session.tmgi;
        self.sessions.insert(tmgi, session);
        &self.sessions[&tmgi]
    }

    /// Removes the session a `MulticastSessionDeactivationRequest` named
    /// (TS 38.413 §9.2.9.3), returning it in the `Stopped` state.
    ///
    /// Returns `None` for a TMGI this gNB never activated, so the caller can
    /// tell "torn down" from "was never here".
    pub fn stop_session(&mut self, tmgi: &[u8; 6]) -> Option<GnbMbsSession> {
        let mut session = self.sessions.remove(tmgi)?;
        session.state = GnbMbsState::Stopped;
        Some(session)
    }

    /// Gets a session mutably by TMGI
    pub fn get_mut(&mut self, tmgi: &[u8; 6]) -> Option<&mut GnbMbsSession> {
        self.sessions.get_mut(tmgi)
    }

    /// Gets a session by TMGI
    pub fn get(&self, tmgi: &[u8; 6]) -> Option<&GnbMbsSession> {
        self.sessions.get(tmgi)
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

    const TEST_TMGI: [u8; 6] = [0xAB, 0xCD, 0xEF, 0x01, 0x02, 0x03];

    fn test_session() -> GnbMbsSession {
        GnbMbsSession::new_multicast("mbs-001".into(), TEST_TMGI)
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
        let stopped = mgr
            .stop_session(&TEST_TMGI)
            .expect("the session was started");
        assert_eq!(stopped.state, GnbMbsState::Stopped);
        assert_eq!(mgr.session_count(), 0);
    }

    /// Deactivating a TMGI that was never activated reports that, rather than
    /// reporting a successful teardown of nothing.
    #[test]
    fn stopping_an_unknown_tmgi_reports_it() {
        let mut mgr = NgapMbsManager::new();
        mgr.start_session(test_session());
        assert!(mgr.stop_session(&[0x99; 6]).is_none());
        assert_eq!(mgr.session_count(), 1, "the real session must survive");
    }

    /// A session is looked up by the TMGI octets the AMF sends, so the NGAP
    /// dispatch path can find it from `id-MBS-SessionID` (299) alone.
    #[test]
    fn a_session_is_found_by_its_tmgi() {
        let mut mgr = NgapMbsManager::new();
        mgr.start_session(test_session());
        assert_eq!(mgr.get(&TEST_TMGI).map(|s| s.tmgi), Some(TEST_TMGI));
        assert!(mgr.get(&[0x00; 6]).is_none());
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
