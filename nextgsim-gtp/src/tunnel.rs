//! GTP-U tunnel management
//!
//! Implements tunnel management for GTP-U user plane data transport.
//! Provides structures for managing GTP tunnels and PDU sessions.

use bytes::Bytes;
use std::collections::HashMap;
use std::net::SocketAddr;
use thiserror::Error;

use crate::codec::{GtpError, GtpHeader, GtpMessageType, PduSessionInfo};

/// Default QoS Flow Identifier used for uplink G-PDUs when a PDU session has no
/// QFI assigned yet (TS 38.415: the UL PDU SESSION INFORMATION QFI field is
/// mandatory, so a value must always be sent).
const DEFAULT_UPLINK_QFI: u8 = 1;

/// GTP-U default port
pub const GTP_U_PORT: u16 = 2152;

/// Tunnel management errors
#[derive(Debug, Error)]
pub enum TunnelError {
    /// Tunnel not found
    #[error("tunnel not found: TEID {0:#x}")]
    TunnelNotFound(u32),
    /// PDU session not found
    #[error("PDU session not found: UE {ue_id}, PSI {psi}")]
    SessionNotFound {
        /// UE identifier
        ue_id: u32,
        /// PDU session identifier
        psi: u8,
    },
    /// Duplicate tunnel
    #[error("duplicate tunnel: TEID {0:#x}")]
    DuplicateTunnel(u32),
    /// GTP codec error
    #[error("GTP codec error: {0}")]
    CodecError(#[from] GtpError),
}

/// GTP tunnel endpoint
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GtpTunnel {
    /// Tunnel Endpoint Identifier
    pub teid: u32,
    /// Remote endpoint address
    pub address: SocketAddr,
}

impl GtpTunnel {
    /// Create a new GTP tunnel
    pub fn new(teid: u32, address: SocketAddr) -> Self {
        Self { teid, address }
    }
}

/// PDU session resource
///
/// Represents a PDU session with uplink and downlink GTP tunnels.
#[derive(Debug, Clone)]
pub struct PduSession {
    /// UE identifier
    pub ue_id: u32,
    /// PDU session identifier (1-15)
    pub psi: u8,
    /// Uplink tunnel (gNB -> UPF)
    pub uplink_tunnel: GtpTunnel,
    /// Downlink tunnel (UPF -> gNB)
    pub downlink_tunnel: GtpTunnel,
    /// `QoS` Flow Identifier (0-63)
    pub qfi: Option<u8>,
}

impl PduSession {
    /// Create a new PDU session
    pub fn new(ue_id: u32, psi: u8, uplink_tunnel: GtpTunnel, downlink_tunnel: GtpTunnel) -> Self {
        Self {
            ue_id,
            psi,
            uplink_tunnel,
            downlink_tunnel,
            qfi: None,
        }
    }

    /// Set the `QoS` Flow Identifier
    pub fn with_qfi(mut self, qfi: u8) -> Self {
        self.qfi = Some(qfi);
        self
    }

    /// Create a unique session key from UE ID and PSI
    pub fn session_key(&self) -> u64 {
        make_session_key(self.ue_id, self.psi)
    }
}

/// Create a unique session key from UE ID and PSI
#[inline]
pub fn make_session_key(ue_id: u32, psi: u8) -> u64 {
    ((ue_id as u64) << 32) | (psi as u64)
}

/// Extract UE ID from session key
#[inline]
pub fn get_ue_id(session_key: u64) -> u32 {
    (session_key >> 32) as u32
}

/// Extract PSI from session key
#[inline]
pub fn get_psi(session_key: u64) -> u8 {
    (session_key & 0xFF) as u8
}

/// Tunnel manager for GTP-U
///
/// Manages GTP tunnels and PDU sessions, providing lookup by TEID
/// and session identifiers.
#[derive(Debug, Default)]
pub struct TunnelManager {
    /// PDU sessions indexed by session key (`ue_id` << 32 | psi)
    sessions: HashMap<u64, PduSession>,
    /// Downlink TEID to session key mapping
    downlink_teid_map: HashMap<u32, u64>,
}

impl TunnelManager {
    /// Create a new tunnel manager
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new PDU session
    pub fn create_session(&mut self, session: PduSession) -> Result<(), TunnelError> {
        let session_key = session.session_key();
        let downlink_teid = session.downlink_tunnel.teid;

        // Check for duplicate downlink TEID
        if self.downlink_teid_map.contains_key(&downlink_teid) {
            return Err(TunnelError::DuplicateTunnel(downlink_teid));
        }

        // Insert session
        self.sessions.insert(session_key, session);
        self.downlink_teid_map.insert(downlink_teid, session_key);

        Ok(())
    }

    /// Delete a PDU session
    pub fn delete_session(&mut self, ue_id: u32, psi: u8) -> Result<PduSession, TunnelError> {
        let session_key = make_session_key(ue_id, psi);

        let session = self
            .sessions
            .remove(&session_key)
            .ok_or(TunnelError::SessionNotFound { ue_id, psi })?;

        // Remove from TEID map
        self.downlink_teid_map.remove(&session.downlink_tunnel.teid);

        Ok(session)
    }

    /// Get a PDU session by UE ID and PSI
    pub fn get_session(&self, ue_id: u32, psi: u8) -> Option<&PduSession> {
        let session_key = make_session_key(ue_id, psi);
        self.sessions.get(&session_key)
    }

    /// Get a mutable PDU session by UE ID and PSI
    pub fn get_session_mut(&mut self, ue_id: u32, psi: u8) -> Option<&mut PduSession> {
        let session_key = make_session_key(ue_id, psi);
        self.sessions.get_mut(&session_key)
    }

    /// Find a PDU session by downlink TEID
    pub fn find_by_downlink_teid(&self, teid: u32) -> Option<&PduSession> {
        self.downlink_teid_map
            .get(&teid)
            .and_then(|key| self.sessions.get(key))
    }

    /// Get all sessions for a UE
    pub fn get_sessions_for_ue(&self, ue_id: u32) -> Vec<&PduSession> {
        self.sessions
            .values()
            .filter(|s| s.ue_id == ue_id)
            .collect()
    }

    /// Delete all sessions for a UE
    pub fn delete_sessions_for_ue(&mut self, ue_id: u32) -> Vec<PduSession> {
        let keys_to_remove: Vec<u64> = self
            .sessions
            .iter()
            .filter(|(_, s)| s.ue_id == ue_id)
            .map(|(k, _)| *k)
            .collect();

        let mut removed = Vec::with_capacity(keys_to_remove.len());
        for key in keys_to_remove {
            if let Some(session) = self.sessions.remove(&key) {
                self.downlink_teid_map.remove(&session.downlink_tunnel.teid);
                removed.push(session);
            }
        }
        removed
    }

    /// Get the number of active sessions
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Check if a session exists
    pub fn has_session(&self, ue_id: u32, psi: u8) -> bool {
        let session_key = make_session_key(ue_id, psi);
        self.sessions.contains_key(&session_key)
    }
}

impl TunnelManager {
    /// Encapsulate user data for uplink transmission
    ///
    /// Creates a GTP-U G-PDU message for sending data through the uplink tunnel.
    pub fn encapsulate_uplink(
        &self,
        ue_id: u32,
        psi: u8,
        data: Bytes,
    ) -> Result<(GtpHeader, SocketAddr), TunnelError> {
        let session = self
            .get_session(ue_id, psi)
            .ok_or(TunnelError::SessionNotFound { ue_id, psi })?;

        // amfg-02: every uplink G-PDU MUST carry a UL PDU SESSION INFORMATION
        // container (TS 38.415 §5.5.2.2) conveyed via the PDU Session Container
        // GTP-U extension header (type 0x85, TS 29.281 §5.2.2.7). The QFI field
        // is mandatory, so fall back to a default flow when none is assigned.
        let qfi = match session.qfi {
            Some(qfi) => qfi,
            None => {
                tracing::debug!(
                    "uplink G-PDU for UE {} PSI {} has no QFI; defaulting to QFI {}",
                    ue_id,
                    psi,
                    DEFAULT_UPLINK_QFI
                );
                DEFAULT_UPLINK_QFI
            }
        };

        let header = GtpHeader::g_pdu(session.uplink_tunnel.teid, data)
            .with_pdu_session_info(PduSessionInfo::uplink(qfi));
        Ok((header, session.uplink_tunnel.address))
    }

    /// Decapsulate received downlink data
    ///
    /// Extracts the payload from a GTP-U message and identifies the target
    /// session. The DL PDU SESSION INFORMATION container (TS 38.415 §5.5.2.1),
    /// when present, also yields the QFI (for DRB/QoS-flow selection) and the
    /// Reflective QoS Indicator (RQI).
    pub fn decapsulate_downlink<'a>(
        &self,
        header: &'a GtpHeader,
    ) -> Result<DecapsulatedDownlink<'a>, TunnelError> {
        // Only handle G-PDU messages
        if header.message_type != GtpMessageType::GPdu {
            return Err(TunnelError::TunnelNotFound(header.teid));
        }

        let session = self
            .find_by_downlink_teid(header.teid)
            .ok_or(TunnelError::TunnelNotFound(header.teid))?;

        // amfg-09: extract the DL QFI / RQI from the PDU Session Container, if
        // the UPF included one.
        let (qfi, rqi) = match header.pdu_session_info() {
            Some(info) => (Some(info.qfi), info.rqi),
            None => (None, false),
        };

        Ok(DecapsulatedDownlink {
            ue_id: session.ue_id,
            psi: session.psi,
            qfi,
            rqi,
            payload: &header.payload,
        })
    }
}

/// Result of decapsulating a downlink G-PDU.
///
/// Carries the identifying session keys plus the QoS metadata extracted from
/// the DL PDU SESSION INFORMATION container (TS 38.415 §5.5.2.1) so the gNB can
/// map the SDU onto the correct DRB / QoS flow toward the UE.
#[derive(Debug, Clone)]
pub struct DecapsulatedDownlink<'a> {
    /// Target UE identifier.
    pub ue_id: u32,
    /// PDU session identifier.
    pub psi: u8,
    /// QoS Flow Identifier from the DL PDU Session Container, if present.
    pub qfi: Option<u8>,
    /// Reflective QoS Indicator (DL only).
    pub rqi: bool,
    /// Decapsulated user-plane payload.
    pub payload: &'a Bytes,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    fn make_test_addr(port: u16) -> SocketAddr {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)), port)
    }

    #[test]
    fn test_session_key_functions() {
        let ue_id = 0x12345678u32;
        let psi = 5u8;

        let key = make_session_key(ue_id, psi);
        assert_eq!(get_ue_id(key), ue_id);
        assert_eq!(get_psi(key), psi);
    }

    #[test]
    fn test_create_and_get_session() {
        let mut manager = TunnelManager::new();

        let session = PduSession::new(
            1,
            5,
            GtpTunnel::new(0x1000, make_test_addr(GTP_U_PORT)),
            GtpTunnel::new(0x2000, make_test_addr(GTP_U_PORT)),
        );

        manager.create_session(session).unwrap();

        let retrieved = manager.get_session(1, 5).unwrap();
        assert_eq!(retrieved.ue_id, 1);
        assert_eq!(retrieved.psi, 5);
        assert_eq!(retrieved.uplink_tunnel.teid, 0x1000);
        assert_eq!(retrieved.downlink_tunnel.teid, 0x2000);
    }

    #[test]
    fn test_find_by_downlink_teid() {
        let mut manager = TunnelManager::new();

        let session = PduSession::new(
            1,
            5,
            GtpTunnel::new(0x1000, make_test_addr(GTP_U_PORT)),
            GtpTunnel::new(0x2000, make_test_addr(GTP_U_PORT)),
        );

        manager.create_session(session).unwrap();

        let found = manager.find_by_downlink_teid(0x2000).unwrap();
        assert_eq!(found.ue_id, 1);
        assert_eq!(found.psi, 5);

        assert!(manager.find_by_downlink_teid(0x9999).is_none());
    }

    #[test]
    fn test_delete_session() {
        let mut manager = TunnelManager::new();

        let session = PduSession::new(
            1,
            5,
            GtpTunnel::new(0x1000, make_test_addr(GTP_U_PORT)),
            GtpTunnel::new(0x2000, make_test_addr(GTP_U_PORT)),
        );

        manager.create_session(session).unwrap();
        assert!(manager.has_session(1, 5));

        let deleted = manager.delete_session(1, 5).unwrap();
        assert_eq!(deleted.ue_id, 1);
        assert!(!manager.has_session(1, 5));
        assert!(manager.find_by_downlink_teid(0x2000).is_none());
    }

    #[test]
    fn test_duplicate_teid_error() {
        let mut manager = TunnelManager::new();

        let session1 = PduSession::new(
            1,
            5,
            GtpTunnel::new(0x1000, make_test_addr(GTP_U_PORT)),
            GtpTunnel::new(0x2000, make_test_addr(GTP_U_PORT)),
        );

        let session2 = PduSession::new(
            2,
            6,
            GtpTunnel::new(0x3000, make_test_addr(GTP_U_PORT)),
            GtpTunnel::new(0x2000, make_test_addr(GTP_U_PORT)), // Same downlink TEID
        );

        manager.create_session(session1).unwrap();
        let result = manager.create_session(session2);
        assert!(matches!(result, Err(TunnelError::DuplicateTunnel(0x2000))));
    }

    #[test]
    fn test_get_sessions_for_ue() {
        let mut manager = TunnelManager::new();

        // Create multiple sessions for UE 1
        for psi in 1..=3 {
            let session = PduSession::new(
                1,
                psi,
                GtpTunnel::new(0x1000 + psi as u32, make_test_addr(GTP_U_PORT)),
                GtpTunnel::new(0x2000 + psi as u32, make_test_addr(GTP_U_PORT)),
            );
            manager.create_session(session).unwrap();
        }

        // Create session for UE 2
        let session = PduSession::new(
            2,
            1,
            GtpTunnel::new(0x5000, make_test_addr(GTP_U_PORT)),
            GtpTunnel::new(0x6000, make_test_addr(GTP_U_PORT)),
        );
        manager.create_session(session).unwrap();

        let ue1_sessions = manager.get_sessions_for_ue(1);
        assert_eq!(ue1_sessions.len(), 3);

        let ue2_sessions = manager.get_sessions_for_ue(2);
        assert_eq!(ue2_sessions.len(), 1);
    }

    #[test]
    fn test_delete_sessions_for_ue() {
        let mut manager = TunnelManager::new();

        // Create sessions for UE 1
        for psi in 1..=2 {
            let session = PduSession::new(
                1,
                psi,
                GtpTunnel::new(0x1000 + psi as u32, make_test_addr(GTP_U_PORT)),
                GtpTunnel::new(0x2000 + psi as u32, make_test_addr(GTP_U_PORT)),
            );
            manager.create_session(session).unwrap();
        }

        // Create session for UE 2
        let session = PduSession::new(
            2,
            1,
            GtpTunnel::new(0x5000, make_test_addr(GTP_U_PORT)),
            GtpTunnel::new(0x6000, make_test_addr(GTP_U_PORT)),
        );
        manager.create_session(session).unwrap();

        assert_eq!(manager.session_count(), 3);

        let deleted = manager.delete_sessions_for_ue(1);
        assert_eq!(deleted.len(), 2);
        assert_eq!(manager.session_count(), 1);
        assert!(manager.get_session(2, 1).is_some());
    }

    #[test]
    fn test_encapsulate_uplink() {
        let mut manager = TunnelManager::new();

        let upf_addr = make_test_addr(GTP_U_PORT);
        let session = PduSession::new(
            1,
            5,
            GtpTunnel::new(0x1000, upf_addr),
            GtpTunnel::new(0x2000, make_test_addr(GTP_U_PORT)),
        );

        manager.create_session(session).unwrap();

        let data = Bytes::from_static(b"test payload");
        let (header, addr) = manager.encapsulate_uplink(1, 5, data.clone()).unwrap();

        assert_eq!(header.message_type, GtpMessageType::GPdu);
        assert_eq!(header.teid, 0x1000);
        assert_eq!(header.payload, data);
        assert_eq!(addr, upf_addr);

        // amfg-02: with no QFI assigned, the UL PDU SESSION INFORMATION
        // container is present with the default flow QFI.
        let info = header
            .pdu_session_info()
            .expect("uplink G-PDU must carry a PDU Session Container");
        assert_eq!(info.qfi, DEFAULT_UPLINK_QFI);
    }

    #[test]
    fn test_encapsulate_uplink_qfi_container_roundtrip() {
        let mut manager = TunnelManager::new();
        let upf_addr = make_test_addr(GTP_U_PORT);
        let session = PduSession::new(
            1,
            5,
            GtpTunnel::new(0x1000, upf_addr),
            GtpTunnel::new(0x2000, make_test_addr(GTP_U_PORT)),
        )
        .with_qfi(9);
        manager.create_session(session).unwrap();

        let data = Bytes::from_static(b"qos payload");
        let (header, _addr) = manager.encapsulate_uplink(1, 5, data.clone()).unwrap();

        // The session QFI is carried in the UL PDU SESSION INFORMATION.
        assert_eq!(header.pdu_session_info().unwrap().qfi, 9);

        // Wire round-trip: encode the G-PDU and decode it back; the PDU Session
        // Container (type 0x85) survives with the same QFI (TS 38.415).
        let bytes = header.encode();
        let decoded = GtpHeader::decode(&bytes).expect("decode G-PDU");
        assert_eq!(decoded.message_type, GtpMessageType::GPdu);
        assert_eq!(decoded.teid, 0x1000);
        let info = decoded
            .pdu_session_info()
            .expect("decoded G-PDU must carry a PDU Session Container");
        assert_eq!(info.qfi, 9);
        assert_eq!(decoded.payload, data);
    }

    #[test]
    fn test_decapsulate_downlink() {
        let mut manager = TunnelManager::new();

        let session = PduSession::new(
            1,
            5,
            GtpTunnel::new(0x1000, make_test_addr(GTP_U_PORT)),
            GtpTunnel::new(0x2000, make_test_addr(GTP_U_PORT)),
        );

        manager.create_session(session).unwrap();

        let payload = Bytes::from_static(b"downlink data");
        let header = GtpHeader::g_pdu(0x2000, payload.clone());

        let dl = manager.decapsulate_downlink(&header).unwrap();
        assert_eq!(dl.ue_id, 1);
        assert_eq!(dl.psi, 5);
        assert_eq!(*dl.payload, payload);
        // No PDU Session Container on this G-PDU.
        assert_eq!(dl.qfi, None);
        assert!(!dl.rqi);
    }

    #[test]
    fn test_decapsulate_downlink_with_qfi_container() {
        use crate::codec::PduSessionInfo;

        let mut manager = TunnelManager::new();
        let session = PduSession::new(
            1,
            5,
            GtpTunnel::new(0x1000, make_test_addr(GTP_U_PORT)),
            GtpTunnel::new(0x2000, make_test_addr(GTP_U_PORT)),
        );
        manager.create_session(session).unwrap();

        // amfg-09: a DL G-PDU carrying a PDU Session Container with QFI 7 and
        // the Reflective QoS Indicator set.
        let payload = Bytes::from_static(b"qos downlink");
        let info = PduSessionInfo::downlink(7).with_rqi(true);
        let header = GtpHeader::g_pdu(0x2000, payload.clone()).with_pdu_session_info(info);

        // Round-trip through the wire to prove the container survives encode.
        let bytes = header.encode();
        let decoded = GtpHeader::decode(&bytes).expect("decode DL G-PDU");

        let dl = manager.decapsulate_downlink(&decoded).unwrap();
        assert_eq!(dl.ue_id, 1);
        assert_eq!(dl.psi, 5);
        assert_eq!(dl.qfi, Some(7));
        assert!(dl.rqi);
        assert_eq!(*dl.payload, payload);
    }

    #[test]
    fn test_decapsulate_unknown_teid() {
        let manager = TunnelManager::new();

        let header = GtpHeader::g_pdu(0x9999, Bytes::from_static(b"data"));
        let result = manager.decapsulate_downlink(&header);

        assert!(matches!(result, Err(TunnelError::TunnelNotFound(0x9999))));
    }
}
