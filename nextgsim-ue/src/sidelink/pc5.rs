//! PC5 MAC/RRC Procedures (Rel-17 ProSe)
//!
//! Implements PC5 interface procedures for device-to-device communication
//! including direct communication and discovery.
//!
//! # 3GPP Reference
//!
//! - TS 23.304: Proximity-based Services (ProSe)
//! - TS 24.554: ProSe PC5 signaling protocol
//! - TS 38.331: RRC for PC5

use std::collections::HashMap;

/// PC5 RRC connection state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pc5RrcState {
    /// No PC5 RRC connection
    Idle,
    /// PC5 RRC connection request sent
    Requested,
    /// PC5 RRC connection established
    Connected,
    /// PC5 RRC connection being reconfigured
    Reconfiguring,
    /// PC5 RRC connection being released
    Releasing,
}

/// PC5 RRC connection context
#[derive(Debug, Clone)]
pub struct Pc5RrcConnection {
    /// Peer UE identifier (Layer-2 ID)
    pub peer_id: u32,
    /// Current PC5 RRC state
    pub state: Pc5RrcState,
    /// Radio bearer configuration for this connection
    pub radio_bearer_config: Option<Pc5RadioBearerConfig>,
    /// PC5 QoS Flow Identifier
    pub pfi: Option<u8>,
    /// Connection establishment timestamp
    pub established_time_ms: Option<u64>,
    /// Last activity timestamp
    pub last_activity_ms: u64,
}

impl Pc5RrcConnection {
    /// Creates a new PC5 RRC connection
    pub fn new(peer_id: u32) -> Self {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            peer_id,
            state: Pc5RrcState::Idle,
            radio_bearer_config: None,
            pfi: None,
            established_time_ms: None,
            last_activity_ms: now_ms,
        }
    }

    /// Requests PC5 RRC connection
    pub fn request(&mut self) {
        self.state = Pc5RrcState::Requested;
        self.update_activity();
    }

    /// Establishes PC5 RRC connection
    pub fn establish(&mut self, bearer_config: Pc5RadioBearerConfig, pfi: u8) {
        self.state = Pc5RrcState::Connected;
        self.radio_bearer_config = Some(bearer_config);
        self.pfi = Some(pfi);

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        self.established_time_ms = Some(now_ms);
        self.update_activity();
    }

    /// Releases PC5 RRC connection
    pub fn release(&mut self) {
        self.state = Pc5RrcState::Releasing;
        self.update_activity();
    }

    /// Marks connection as idle
    pub fn mark_idle(&mut self) {
        self.state = Pc5RrcState::Idle;
        self.radio_bearer_config = None;
        self.pfi = None;
        self.established_time_ms = None;
    }

    /// Updates last activity timestamp
    fn update_activity(&mut self) {
        self.last_activity_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
    }

    /// Returns true if connection is active
    pub fn is_active(&self) -> bool {
        self.state == Pc5RrcState::Connected
    }
}

/// PC5 radio bearer configuration
#[derive(Debug, Clone)]
pub struct Pc5RadioBearerConfig {
    /// PDCP configuration
    pub pdcp_config: Pc5PdcpConfig,
    /// RLC configuration
    pub rlc_config: Pc5RlcConfig,
    /// Logical channel configuration
    pub lc_config: Pc5LogicalChannelConfig,
}

/// PC5 PDCP configuration
#[derive(Debug, Clone)]
pub struct Pc5PdcpConfig {
    /// PDCP SN size (12 or 18 bits)
    pub sn_size: u8,
    /// Integrity protection enabled
    pub integrity_protection: bool,
    /// Ciphering enabled
    pub ciphering: bool,
}

/// PC5 RLC configuration
#[derive(Debug, Clone)]
pub struct Pc5RlcConfig {
    /// RLC mode (AM, UM, TM)
    pub mode: Pc5RlcMode,
    /// RLC SN size
    pub sn_size: u8,
}

/// PC5 RLC mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pc5RlcMode {
    /// Acknowledged Mode
    Am,
    /// Unacknowledged Mode
    Um,
    /// Transparent Mode
    Tm,
}

/// PC5 logical channel configuration
#[derive(Debug, Clone)]
pub struct Pc5LogicalChannelConfig {
    /// Logical channel ID
    pub lc_id: u8,
    /// Priority
    pub priority: u8,
    /// Bucket size duration (ms)
    pub bucket_size_duration_ms: u16,
}

/// PC5 resource allocation mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pc5ResourceMode {
    /// Mode 1: gNB-scheduled (network-controlled)
    Mode1,
    /// Mode 2: UE autonomous selection
    Mode2,
}

/// PC5 discovery mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pc5DiscoveryMode {
    /// Model A: Announcing (UE broadcasts discovery messages)
    ModelA,
    /// Model B: Soliciting (UE sends discovery requests)
    ModelB,
}

/// PC5 discovery parameters
#[derive(Debug, Clone)]
pub struct Pc5Discovery {
    /// Discovery mode
    pub mode: Pc5DiscoveryMode,
    /// Announce parameters (for Model A)
    pub announce_params: Option<Pc5AnnounceParams>,
    /// Discovery filter (for Model B)
    pub discovery_filter: Option<Pc5DiscoveryFilter>,
    /// Monitored peers
    pub monitored_peers: HashMap<u32, Pc5DiscoveredPeer>,
}

/// PC5 announce parameters (Model A)
#[derive(Debug, Clone)]
pub struct Pc5AnnounceParams {
    /// Announcement period (ms)
    pub period_ms: u32,
    /// ProSe application code
    pub app_code: u32,
    /// Discovery message content
    pub discovery_msg: Vec<u8>,
}

/// PC5 discovery filter (Model B)
#[derive(Debug, Clone)]
pub struct Pc5DiscoveryFilter {
    /// Target ProSe application codes
    pub target_app_codes: Vec<u32>,
    /// Maximum discovery range (meters)
    pub max_range_m: Option<f64>,
    /// Minimum signal quality (dBm)
    pub min_rsrp_dbm: Option<i32>,
}

/// Discovered peer information
#[derive(Debug, Clone)]
pub struct Pc5DiscoveredPeer {
    /// Peer Layer-2 ID
    pub peer_id: u32,
    /// ProSe application code
    pub app_code: u32,
    /// Last discovery time
    pub last_discovery_ms: u64,
    /// Signal quality (RSRP in dBm)
    pub rsrp_dbm: i32,
    /// Discovery message content
    pub discovery_msg: Vec<u8>,
}

/// Sidelink HARQ feedback
#[derive(Debug, Clone)]
pub struct Pc5HarqFeedback {
    /// HARQ process ID
    pub harq_id: u8,
    /// ACK/NACK bitmap
    pub ack_nack: Vec<bool>,
    /// Timestamp
    pub timestamp_ms: u64,
}

impl Pc5Discovery {
    /// Creates a new PC5 discovery with Model A (announcing)
    pub fn model_a(period_ms: u32, app_code: u32) -> Self {
        Self {
            mode: Pc5DiscoveryMode::ModelA,
            announce_params: Some(Pc5AnnounceParams {
                period_ms,
                app_code,
                discovery_msg: vec![],
            }),
            discovery_filter: None,
            monitored_peers: HashMap::new(),
        }
    }

    /// Creates a new PC5 discovery with Model B (soliciting)
    pub fn model_b(target_app_codes: Vec<u32>) -> Self {
        Self {
            mode: Pc5DiscoveryMode::ModelB,
            announce_params: None,
            discovery_filter: Some(Pc5DiscoveryFilter {
                target_app_codes,
                max_range_m: None,
                min_rsrp_dbm: Some(-120),
            }),
            monitored_peers: HashMap::new(),
        }
    }

    /// Adds a discovered peer
    pub fn add_discovered_peer(&mut self, peer: Pc5DiscoveredPeer) {
        self.monitored_peers.insert(peer.peer_id, peer);
    }

    /// Gets a discovered peer
    pub fn get_peer(&self, peer_id: u32) -> Option<&Pc5DiscoveredPeer> {
        self.monitored_peers.get(&peer_id)
    }

    /// Filters discovered peers by application code
    pub fn filter_by_app_code(&self, app_code: u32) -> Vec<&Pc5DiscoveredPeer> {
        self.monitored_peers
            .values()
            .filter(|p| p.app_code == app_code)
            .collect()
    }

    /// Prunes stale peers (not discovered within timeout)
    pub fn prune_stale_peers(&mut self, timeout_ms: u64) {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        self.monitored_peers
            .retain(|_, peer| now_ms - peer.last_discovery_ms < timeout_ms);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pc5_rrc_connection_lifecycle() {
        let mut conn = Pc5RrcConnection::new(12345);
        assert_eq!(conn.state, Pc5RrcState::Idle);
        assert!(!conn.is_active());

        conn.request();
        assert_eq!(conn.state, Pc5RrcState::Requested);

        let bearer_config = Pc5RadioBearerConfig {
            pdcp_config: Pc5PdcpConfig {
                sn_size: 12,
                integrity_protection: true,
                ciphering: true,
            },
            rlc_config: Pc5RlcConfig {
                mode: Pc5RlcMode::Am,
                sn_size: 12,
            },
            lc_config: Pc5LogicalChannelConfig {
                lc_id: 4,
                priority: 1,
                bucket_size_duration_ms: 100,
            },
        };

        conn.establish(bearer_config, 5);
        assert_eq!(conn.state, Pc5RrcState::Connected);
        assert!(conn.is_active());
        assert_eq!(conn.pfi, Some(5));

        conn.release();
        assert_eq!(conn.state, Pc5RrcState::Releasing);

        conn.mark_idle();
        assert_eq!(conn.state, Pc5RrcState::Idle);
        assert!(!conn.is_active());
    }

    #[test]
    fn test_pc5_discovery_model_a() {
        let mut discovery = Pc5Discovery::model_a(1000, 0x12345678);
        assert_eq!(discovery.mode, Pc5DiscoveryMode::ModelA);
        assert!(discovery.announce_params.is_some());
        assert!(discovery.discovery_filter.is_none());

        let peer = Pc5DiscoveredPeer {
            peer_id: 100,
            app_code: 0x12345678,
            last_discovery_ms: 0,
            rsrp_dbm: -80,
            discovery_msg: vec![1, 2, 3],
        };

        discovery.add_discovered_peer(peer);
        assert_eq!(discovery.monitored_peers.len(), 1);
        assert!(discovery.get_peer(100).is_some());
    }

    #[test]
    fn test_pc5_discovery_model_b() {
        let discovery = Pc5Discovery::model_b(vec![0x11111111, 0x22222222]);
        assert_eq!(discovery.mode, Pc5DiscoveryMode::ModelB);
        assert!(discovery.announce_params.is_none());
        assert!(discovery.discovery_filter.is_some());

        let filter = discovery.discovery_filter.unwrap();
        assert_eq!(filter.target_app_codes.len(), 2);
        assert_eq!(filter.min_rsrp_dbm, Some(-120));
    }

    #[test]
    fn test_pc5_discovery_filter_by_app_code() {
        let mut discovery = Pc5Discovery::model_a(1000, 0x12345678);

        let peer1 = Pc5DiscoveredPeer {
            peer_id: 100,
            app_code: 0x11111111,
            last_discovery_ms: 0,
            rsrp_dbm: -80,
            discovery_msg: vec![],
        };

        let peer2 = Pc5DiscoveredPeer {
            peer_id: 200,
            app_code: 0x22222222,
            last_discovery_ms: 0,
            rsrp_dbm: -85,
            discovery_msg: vec![],
        };

        let peer3 = Pc5DiscoveredPeer {
            peer_id: 300,
            app_code: 0x11111111,
            last_discovery_ms: 0,
            rsrp_dbm: -75,
            discovery_msg: vec![],
        };

        discovery.add_discovered_peer(peer1);
        discovery.add_discovered_peer(peer2);
        discovery.add_discovered_peer(peer3);

        let filtered = discovery.filter_by_app_code(0x11111111);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_rlc_modes() {
        assert_eq!(Pc5RlcMode::Am, Pc5RlcMode::Am);
        assert_ne!(Pc5RlcMode::Am, Pc5RlcMode::Um);
    }

    #[test]
    fn test_resource_modes() {
        assert_eq!(Pc5ResourceMode::Mode1, Pc5ResourceMode::Mode1);
        assert_ne!(Pc5ResourceMode::Mode1, Pc5ResourceMode::Mode2);
    }
}
