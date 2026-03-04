//! ProSe (Proximity Services) Protocol for UE (Rel-12/13/17, TS 23.303, TS 23.304)
//!
//! Implements UE-side ProSe:
//! - PC5 unicast link establishment and management
//! - ProSe direct discovery (Model A: announcing, Model B: requesting)
//! - UE-to-UE relay (Layer 2 relay over PC5)
//! - Relay selection and RSC (Relay Service Code) matching

use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::sidelink::Pc5RrcState;

/// ProSe UE ID (Layer 2 destination address for PC5)
pub type ProseL2Id = u32;

/// Relay Service Code: identifies a ProSe UE-to-Network relay service
pub type RelayServiceCode = u32;

/// ProSe discovery status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProseDiscoveryStatus {
    /// Not actively discovering
    Idle,
    /// Broadcasting announcements (Model A)
    Announcing,
    /// Monitoring for announcements (Model A monitor)
    Monitoring,
    /// Sending discovery requests (Model B requester)
    Requesting,
    /// Responding to discovery requests (Model B responder)
    Responding,
}

/// A discovered ProSe peer
#[derive(Debug, Clone)]
pub struct ProsePeer {
    /// Layer 2 destination address
    pub l2_id: ProseL2Id,
    /// Application layer ID (if known)
    pub app_id: Option<String>,
    /// RSRP measurement (dBm)
    pub rsrp_dbm: i32,
    /// Whether this peer can serve as a relay
    pub is_relay: bool,
    /// Relay Service Codes supported by this peer
    pub relay_service_codes: Vec<RelayServiceCode>,
    /// Discovery timestamp (UNIX seconds)
    pub discovered_at: u64,
}

/// PC5 bearer for a ProSe unicast link
#[derive(Debug, Clone)]
pub struct Pc5Bearer {
    /// Local Layer 2 ID
    pub local_l2_id: ProseL2Id,
    /// Remote Layer 2 ID
    pub remote_l2_id: ProseL2Id,
    /// PC5 RRC connection state
    pub rrc_state: Pc5RrcState,
    /// PC5 link established timestamp
    pub established_at: u64,
    /// Bytes sent over this bearer
    pub bytes_tx: u64,
    /// Bytes received over this bearer
    pub bytes_rx: u64,
}

impl Pc5Bearer {
    pub fn new(local_l2_id: ProseL2Id, remote_l2_id: ProseL2Id) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs();
        Self {
            local_l2_id,
            remote_l2_id,
            rrc_state: Pc5RrcState::Connected,
            established_at: now,
            bytes_tx: 0,
            bytes_rx: 0,
        }
    }

    pub fn is_connected(&self) -> bool {
        self.rrc_state == Pc5RrcState::Connected
    }
}

/// UE relay context: when this UE acts as a UE-to-Network relay
#[derive(Debug, Clone)]
pub struct UeRelayContext {
    /// Relay Service Code being served
    pub rsc: RelayServiceCode,
    /// Remote UEs relayed (L2 ID → bearer)
    pub relayed_ues: HashMap<ProseL2Id, Pc5Bearer>,
    /// Whether relay is currently active
    pub active: bool,
}

impl UeRelayContext {
    pub fn new(rsc: RelayServiceCode) -> Self {
        Self {
            rsc,
            relayed_ues: HashMap::new(),
            active: true,
        }
    }

    /// Adds a remote UE to relay
    pub fn add_ue(&mut self, bearer: Pc5Bearer) {
        self.relayed_ues.insert(bearer.remote_l2_id, bearer);
    }

    /// Removes a remote UE
    pub fn remove_ue(&mut self, l2_id: ProseL2Id) -> bool {
        self.relayed_ues.remove(&l2_id).is_some()
    }
}

/// ProSe context for a UE
#[derive(Debug)]
pub struct ProseContext {
    /// This UE's ProSe Layer 2 ID
    pub local_l2_id: ProseL2Id,
    /// Discovery status
    pub discovery_status: ProseDiscoveryStatus,
    /// Discovered peers (keyed by L2 ID)
    pub peers: HashMap<ProseL2Id, ProsePeer>,
    /// Active PC5 unicast bearers (keyed by remote L2 ID)
    pub bearers: HashMap<ProseL2Id, Pc5Bearer>,
    /// Relay context (if this UE is acting as relay)
    pub relay_context: Option<UeRelayContext>,
    /// RSC of the relay this UE is using (if acting as remote UE)
    pub using_relay_rsc: Option<(ProseL2Id, RelayServiceCode)>,
}

impl ProseContext {
    pub fn new(local_l2_id: ProseL2Id) -> Self {
        Self {
            local_l2_id,
            discovery_status: ProseDiscoveryStatus::Idle,
            peers: HashMap::new(),
            bearers: HashMap::new(),
            relay_context: None,
            using_relay_rsc: None,
        }
    }

    /// Adds a discovered peer
    pub fn add_peer(&mut self, peer: ProsePeer) {
        self.peers.insert(peer.l2_id, peer);
    }

    /// Selects the best relay peer for a given RSC (highest RSRP)
    pub fn select_relay(&self, rsc: RelayServiceCode) -> Option<&ProsePeer> {
        self.peers.values()
            .filter(|p| p.is_relay && p.relay_service_codes.contains(&rsc))
            .max_by_key(|p| p.rsrp_dbm)
    }

    /// Establishes a PC5 unicast bearer to a peer
    pub fn establish_bearer(&mut self, remote_l2_id: ProseL2Id) {
        let bearer = Pc5Bearer::new(self.local_l2_id, remote_l2_id);
        self.bearers.insert(remote_l2_id, bearer);
    }

    /// Releases a PC5 bearer
    pub fn release_bearer(&mut self, remote_l2_id: ProseL2Id) -> bool {
        self.bearers.remove(&remote_l2_id).is_some()
    }

    /// Activates relay mode for a given RSC
    pub fn start_relay(&mut self, rsc: RelayServiceCode) {
        self.relay_context = Some(UeRelayContext::new(rsc));
    }

    pub fn active_bearers(&self) -> usize {
        self.bearers.values().filter(|b| b.is_connected()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prose_context_creation() {
        let ctx = ProseContext::new(0xABCD);
        assert_eq!(ctx.local_l2_id, 0xABCD);
        assert_eq!(ctx.discovery_status, ProseDiscoveryStatus::Idle);
        assert!(ctx.peers.is_empty());
    }

    #[test]
    fn test_add_and_select_relay_peer() {
        let mut ctx = ProseContext::new(0x0001);
        ctx.add_peer(ProsePeer {
            l2_id: 0x0002,
            app_id: None,
            rsrp_dbm: -95,
            is_relay: true,
            relay_service_codes: vec![0x001234],
            discovered_at: 0,
        });
        ctx.add_peer(ProsePeer {
            l2_id: 0x0003,
            app_id: None,
            rsrp_dbm: -85, // better signal
            is_relay: true,
            relay_service_codes: vec![0x001234],
            discovered_at: 0,
        });
        let relay = ctx.select_relay(0x001234).unwrap();
        assert_eq!(relay.l2_id, 0x0003); // higher RSRP selected
    }

    #[test]
    fn test_select_relay_no_match() {
        let mut ctx = ProseContext::new(0x0001);
        ctx.add_peer(ProsePeer {
            l2_id: 0x0002,
            app_id: None,
            rsrp_dbm: -85,
            is_relay: true,
            relay_service_codes: vec![0x001234],
            discovered_at: 0,
        });
        assert!(ctx.select_relay(0x999999).is_none());
    }

    #[test]
    fn test_bearer_establish_and_release() {
        let mut ctx = ProseContext::new(0x0001);
        ctx.establish_bearer(0x0002);
        assert_eq!(ctx.active_bearers(), 1);
        assert!(ctx.release_bearer(0x0002));
        assert_eq!(ctx.active_bearers(), 0);
    }

    #[test]
    fn test_relay_context_activation() {
        let mut ctx = ProseContext::new(0x0001);
        ctx.start_relay(0x001234);
        assert!(ctx.relay_context.is_some());
        assert_eq!(ctx.relay_context.as_ref().unwrap().rsc, 0x001234);
    }

    #[test]
    fn test_relay_add_and_remove_ue() {
        let mut relay_ctx = UeRelayContext::new(0x001234);
        let bearer = Pc5Bearer::new(0x0001, 0x0010);
        relay_ctx.add_ue(bearer);
        assert_eq!(relay_ctx.relayed_ues.len(), 1);
        assert!(relay_ctx.remove_ue(0x0010));
        assert!(relay_ctx.relayed_ues.is_empty());
    }
}
