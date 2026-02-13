//! MBS (Multicast/Broadcast Service) Context for gNB
//!
//! Manages MBS sessions at the gNB including TMGI, joined UEs, and multicast tunnel info.
//! Implements Rel-17 3GPP TS 23.247 MBS functionality.
//!
//! # 3GPP Reference
//!
//! - TS 23.247: Multicast/Broadcast Service Architecture
//! - TS 38.413: NGAP (NG Application Protocol) specification

use std::collections::HashMap;
use std::net::IpAddr;

/// MBS session state at gNB
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MbsSessionState {
    /// Session is being activated
    Activating,
    /// Session is active and delivering MBS traffic
    Active,
    /// Session is being deactivated
    Deactivating,
    /// Session has been deactivated
    Deactivated,
    /// Session activation failed
    Failed,
}

/// Temporary Mobile Group Identity (TMGI)
///
/// Uniquely identifies an MBS session
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Tmgi {
    /// Service ID (3 bytes)
    pub service_id: [u8; 3],
    /// PLMN identity (3 bytes)
    pub plmn_identity: [u8; 3],
}

impl Tmgi {
    /// Creates a new TMGI
    pub fn new(service_id: [u8; 3], plmn_identity: [u8; 3]) -> Self {
        Self {
            service_id,
            plmn_identity,
        }
    }

    /// Encodes TMGI to bytes
    pub fn to_bytes(&self) -> [u8; 6] {
        let mut bytes = [0u8; 6];
        bytes[0..3].copy_from_slice(&self.service_id);
        bytes[3..6].copy_from_slice(&self.plmn_identity);
        bytes
    }

    /// Decodes TMGI from bytes
    pub fn from_bytes(bytes: &[u8; 6]) -> Self {
        let mut service_id = [0u8; 3];
        let mut plmn_identity = [0u8; 3];
        service_id.copy_from_slice(&bytes[0..3]);
        plmn_identity.copy_from_slice(&bytes[3..6]);
        Self::new(service_id, plmn_identity)
    }
}

/// MBS session area info
#[derive(Debug, Clone)]
pub struct MbsSessionAreaInfo {
    /// List of tracking area codes in the session area
    pub tac_list: Vec<u32>,
    /// List of cell IDs in the session area
    pub cell_id_list: Vec<u64>,
}

/// Multicast tunnel information
#[derive(Debug, Clone)]
pub struct MulticastTunnelInfo {
    /// Multicast group IP address
    pub multicast_ip: IpAddr,
    /// Multicast source IP address (for source-specific multicast)
    pub source_ip: Option<IpAddr>,
    /// GTP-U TEID for multicast traffic
    pub teid: u32,
    /// QoS Flow Identifier
    pub qfi: u8,
}

/// gNB MBS session context
///
/// Represents an MBS session at the gNB with all relevant state
/// for multicast/broadcast service delivery
#[derive(Debug, Clone)]
pub struct GnbMbsContext {
    /// MBS session ID assigned by AMF
    pub session_id: u32,
    /// Temporary Mobile Group Identity
    pub tmgi: Tmgi,
    /// Current session state
    pub state: MbsSessionState,
    /// List of UE IDs that have joined this MBS session
    pub joined_ues: Vec<i32>,
    /// Multicast downlink tunnel information
    pub dl_tnl_info: Option<MulticastTunnelInfo>,
    /// MBS session area information
    pub area_info: Option<MbsSessionAreaInfo>,
    /// MBS service area priority (0-15, 0 is highest)
    pub service_area_priority: Option<u8>,
    /// Whether this is a broadcast session (true) or multicast (false)
    pub is_broadcast: bool,
    /// Session start time (milliseconds since epoch)
    pub session_start_time_ms: Option<u64>,
    /// Session duration in seconds
    pub session_duration_s: Option<u32>,
}

impl GnbMbsContext {
    /// Creates a new MBS session context
    pub fn new(session_id: u32, tmgi: Tmgi, is_broadcast: bool) -> Self {
        Self {
            session_id,
            tmgi,
            state: MbsSessionState::Activating,
            joined_ues: Vec::new(),
            dl_tnl_info: None,
            area_info: None,
            service_area_priority: None,
            is_broadcast,
            session_start_time_ms: None,
            session_duration_s: None,
        }
    }

    /// Adds a UE to this MBS session
    pub fn add_ue(&mut self, ue_id: i32) -> bool {
        if !self.joined_ues.contains(&ue_id) {
            self.joined_ues.push(ue_id);
            true
        } else {
            false
        }
    }

    /// Removes a UE from this MBS session
    pub fn remove_ue(&mut self, ue_id: i32) -> bool {
        if let Some(pos) = self.joined_ues.iter().position(|&id| id == ue_id) {
            self.joined_ues.swap_remove(pos);
            true
        } else {
            false
        }
    }

    /// Returns true if the session is active
    pub fn is_active(&self) -> bool {
        self.state == MbsSessionState::Active
    }

    /// Returns the number of UEs in this session
    pub fn ue_count(&self) -> usize {
        self.joined_ues.len()
    }

    /// Checks if a UE is in this session
    pub fn has_ue(&self, ue_id: i32) -> bool {
        self.joined_ues.contains(&ue_id)
    }

    /// Activates the MBS session
    pub fn activate(&mut self, dl_tnl_info: Option<MulticastTunnelInfo>) {
        self.state = MbsSessionState::Active;
        if let Some(tnl) = dl_tnl_info {
            self.dl_tnl_info = Some(tnl);
        }
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        self.session_start_time_ms = Some(now_ms);
    }

    /// Deactivates the MBS session
    pub fn deactivate(&mut self) {
        self.state = MbsSessionState::Deactivating;
        self.joined_ues.clear();
    }

    /// Marks the session as failed
    pub fn mark_failed(&mut self) {
        self.state = MbsSessionState::Failed;
    }
}

/// MBS session manager for the gNB
///
/// Manages multiple MBS sessions at the gNB
#[derive(Debug)]
pub struct MbsSessionManager {
    /// Active MBS sessions indexed by session ID
    sessions_by_id: HashMap<u32, GnbMbsContext>,
    /// Active MBS sessions indexed by TMGI
    sessions_by_tmgi: HashMap<Tmgi, u32>,
}

impl MbsSessionManager {
    /// Creates a new MBS session manager
    pub fn new() -> Self {
        Self {
            sessions_by_id: HashMap::new(),
            sessions_by_tmgi: HashMap::new(),
        }
    }

    /// Adds a new MBS session
    pub fn add_session(&mut self, session: GnbMbsContext) -> bool {
        if self.sessions_by_id.contains_key(&session.session_id) {
            return false; // Session already exists
        }
        self.sessions_by_tmgi
            .insert(session.tmgi, session.session_id);
        self.sessions_by_id.insert(session.session_id, session);
        true
    }

    /// Removes an MBS session
    pub fn remove_session(&mut self, session_id: u32) -> Option<GnbMbsContext> {
        if let Some(session) = self.sessions_by_id.remove(&session_id) {
            self.sessions_by_tmgi.remove(&session.tmgi);
            Some(session)
        } else {
            None
        }
    }

    /// Gets a session by ID
    pub fn get_session(&self, session_id: u32) -> Option<&GnbMbsContext> {
        self.sessions_by_id.get(&session_id)
    }

    /// Gets a mutable session by ID
    pub fn get_session_mut(&mut self, session_id: u32) -> Option<&mut GnbMbsContext> {
        self.sessions_by_id.get_mut(&session_id)
    }

    /// Gets a session by TMGI
    pub fn get_session_by_tmgi(&self, tmgi: &Tmgi) -> Option<&GnbMbsContext> {
        self.sessions_by_tmgi
            .get(tmgi)
            .and_then(|&id| self.sessions_by_id.get(&id))
    }

    /// Gets a mutable session by TMGI
    pub fn get_session_by_tmgi_mut(&mut self, tmgi: &Tmgi) -> Option<&mut GnbMbsContext> {
        if let Some(&id) = self.sessions_by_tmgi.get(tmgi) {
            self.sessions_by_id.get_mut(&id)
        } else {
            None
        }
    }

    /// Returns the number of active sessions
    pub fn session_count(&self) -> usize {
        self.sessions_by_id.len()
    }

    /// Lists all sessions
    pub fn list_sessions(&self) -> Vec<&GnbMbsContext> {
        self.sessions_by_id.values().collect()
    }

    /// Finds all sessions that a UE has joined
    pub fn find_sessions_for_ue(&self, ue_id: i32) -> Vec<&GnbMbsContext> {
        self.sessions_by_id
            .values()
            .filter(|s| s.has_ue(ue_id))
            .collect()
    }

    /// Removes a UE from all sessions
    pub fn remove_ue_from_all_sessions(&mut self, ue_id: i32) {
        for session in self.sessions_by_id.values_mut() {
            session.remove_ue(ue_id);
        }
    }
}

impl Default for MbsSessionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tmgi_encode_decode() {
        let tmgi = Tmgi::new([0x01, 0x02, 0x03], [0xf1, 0x10, 0x01]);
        let bytes = tmgi.to_bytes();
        assert_eq!(bytes, [0x01, 0x02, 0x03, 0xf1, 0x10, 0x01]);

        let decoded = Tmgi::from_bytes(&bytes);
        assert_eq!(decoded, tmgi);
    }

    #[test]
    fn test_mbs_context_new() {
        let tmgi = Tmgi::new([0x01, 0x02, 0x03], [0xf1, 0x10, 0x01]);
        let ctx = GnbMbsContext::new(1, tmgi, false);

        assert_eq!(ctx.session_id, 1);
        assert_eq!(ctx.tmgi, tmgi);
        assert_eq!(ctx.state, MbsSessionState::Activating);
        assert!(!ctx.is_broadcast);
        assert_eq!(ctx.ue_count(), 0);
    }

    #[test]
    fn test_mbs_context_add_remove_ue() {
        let tmgi = Tmgi::new([0x01, 0x02, 0x03], [0xf1, 0x10, 0x01]);
        let mut ctx = GnbMbsContext::new(1, tmgi, false);

        assert!(ctx.add_ue(100));
        assert_eq!(ctx.ue_count(), 1);
        assert!(ctx.has_ue(100));

        // Adding the same UE again should return false
        assert!(!ctx.add_ue(100));
        assert_eq!(ctx.ue_count(), 1);

        assert!(ctx.add_ue(200));
        assert_eq!(ctx.ue_count(), 2);

        assert!(ctx.remove_ue(100));
        assert_eq!(ctx.ue_count(), 1);
        assert!(!ctx.has_ue(100));
        assert!(ctx.has_ue(200));

        // Removing non-existent UE should return false
        assert!(!ctx.remove_ue(100));
    }

    #[test]
    fn test_mbs_context_activate() {
        let tmgi = Tmgi::new([0x01, 0x02, 0x03], [0xf1, 0x10, 0x01]);
        let mut ctx = GnbMbsContext::new(1, tmgi, false);

        let tnl_info = MulticastTunnelInfo {
            multicast_ip: "239.1.1.1".parse().unwrap(),
            source_ip: Some("10.0.0.1".parse().unwrap()),
            teid: 0x12345678,
            qfi: 9,
        };

        ctx.activate(Some(tnl_info));
        assert_eq!(ctx.state, MbsSessionState::Active);
        assert!(ctx.is_active());
        assert!(ctx.dl_tnl_info.is_some());
        assert!(ctx.session_start_time_ms.is_some());
    }

    #[test]
    fn test_mbs_context_deactivate() {
        let tmgi = Tmgi::new([0x01, 0x02, 0x03], [0xf1, 0x10, 0x01]);
        let mut ctx = GnbMbsContext::new(1, tmgi, false);

        ctx.add_ue(100);
        ctx.add_ue(200);
        assert_eq!(ctx.ue_count(), 2);

        ctx.deactivate();
        assert_eq!(ctx.state, MbsSessionState::Deactivating);
        assert_eq!(ctx.ue_count(), 0);
    }

    #[test]
    fn test_mbs_session_manager() {
        let mut manager = MbsSessionManager::new();

        let tmgi1 = Tmgi::new([0x01, 0x02, 0x03], [0xf1, 0x10, 0x01]);
        let ctx1 = GnbMbsContext::new(1, tmgi1, false);

        let tmgi2 = Tmgi::new([0x04, 0x05, 0x06], [0xf1, 0x10, 0x01]);
        let ctx2 = GnbMbsContext::new(2, tmgi2, true);

        assert!(manager.add_session(ctx1));
        assert!(manager.add_session(ctx2));
        assert_eq!(manager.session_count(), 2);

        // Adding duplicate session should fail
        let ctx1_dup = GnbMbsContext::new(1, tmgi1, false);
        assert!(!manager.add_session(ctx1_dup));

        assert!(manager.get_session(1).is_some());
        assert!(manager.get_session_by_tmgi(&tmgi2).is_some());

        let removed = manager.remove_session(1);
        assert!(removed.is_some());
        assert_eq!(manager.session_count(), 1);
        assert!(manager.get_session(1).is_none());
    }

    #[test]
    fn test_mbs_session_manager_ue_operations() {
        let mut manager = MbsSessionManager::new();

        let tmgi1 = Tmgi::new([0x01, 0x02, 0x03], [0xf1, 0x10, 0x01]);
        let mut ctx1 = GnbMbsContext::new(1, tmgi1, false);
        ctx1.add_ue(100);
        ctx1.add_ue(200);

        let tmgi2 = Tmgi::new([0x04, 0x05, 0x06], [0xf1, 0x10, 0x01]);
        let mut ctx2 = GnbMbsContext::new(2, tmgi2, false);
        ctx2.add_ue(100);

        manager.add_session(ctx1);
        manager.add_session(ctx2);

        let sessions = manager.find_sessions_for_ue(100);
        assert_eq!(sessions.len(), 2);

        let sessions = manager.find_sessions_for_ue(200);
        assert_eq!(sessions.len(), 1);

        manager.remove_ue_from_all_sessions(100);
        let sessions = manager.find_sessions_for_ue(100);
        assert_eq!(sessions.len(), 0);

        let sessions = manager.find_sessions_for_ue(200);
        assert_eq!(sessions.len(), 1);
    }
}
