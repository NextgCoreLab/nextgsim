//! RLS (Radio Link Simulation) message types
//!
//! This module defines the message types used in the RLS protocol,
//! which provides simulated radio link communication between UE and gNB
//! over UDP without requiring real radio hardware.

use bytes::Bytes;
use std::fmt;

/// RLS protocol version information
pub mod version {
    /// Major version number
    pub const MAJOR: u8 = 3;
    /// Minor version number
    pub const MINOR: u8 = 2;
    /// Patch version number
    pub const PATCH: u8 = 7;
}

/// RLS message type identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MessageType {
    /// Reserved message type
    Reserved = 0,
    /// Deprecated message type 1
    Deprecated1 = 1,
    /// Deprecated message type 2
    Deprecated2 = 2,
    /// Deprecated message type 3
    Deprecated3 = 3,
    /// Heartbeat message for cell search and connection maintenance
    Heartbeat = 4,
    /// Acknowledgment of heartbeat message
    HeartbeatAck = 5,
    /// PDU transmission message for RRC and user plane data
    PduTransmission = 6,
    /// Acknowledgment of PDU transmission
    PduTransmissionAck = 7,
}

impl MessageType {
    /// Creates a MessageType from a u8 value
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Reserved),
            1 => Some(Self::Deprecated1),
            2 => Some(Self::Deprecated2),
            3 => Some(Self::Deprecated3),
            4 => Some(Self::Heartbeat),
            5 => Some(Self::HeartbeatAck),
            6 => Some(Self::PduTransmission),
            7 => Some(Self::PduTransmissionAck),
            _ => None,
        }
    }
}

/// PDU type for RLS transmission
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PduType {
    /// Reserved PDU type
    Reserved = 0,
    /// RRC (Radio Resource Control) message
    Rrc = 1,
    /// User plane data
    Data = 2,
}

impl PduType {
    /// Creates a PduType from a u8 value
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Reserved),
            1 => Some(Self::Rrc),
            2 => Some(Self::Data),
            _ => None,
        }
    }
}

/// 3D position vector for simulated radio positioning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Vector3 {
    /// X coordinate
    pub x: i32,
    /// Y coordinate
    pub y: i32,
    /// Z coordinate
    pub z: i32,
}

impl Vector3 {
    /// Creates a new Vector3 with the given coordinates
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }
}

impl fmt::Display for Vector3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

/// Base trait for all RLS messages
pub trait RlsMessageTrait {
    /// Returns the message type
    fn message_type(&self) -> MessageType;
    /// Returns the STI (Simulated Transmission Identifier)
    fn sti(&self) -> u64;
}

/// RLS message enum containing all message variants
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RlsMessage {
    /// Heartbeat message for cell search
    Heartbeat(RlsHeartbeat),
    /// Heartbeat acknowledgment
    HeartbeatAck(RlsHeartbeatAck),
    /// PDU transmission message
    PduTransmission(RlsPduTransmission),
    /// PDU transmission acknowledgment
    PduTransmissionAck(RlsPduTransmissionAck),
}

impl RlsMessage {
    /// Returns the message type
    pub fn message_type(&self) -> MessageType {
        match self {
            Self::Heartbeat(_) => MessageType::Heartbeat,
            Self::HeartbeatAck(_) => MessageType::HeartbeatAck,
            Self::PduTransmission(_) => MessageType::PduTransmission,
            Self::PduTransmissionAck(_) => MessageType::PduTransmissionAck,
        }
    }

    /// Returns the STI (Simulated Transmission Identifier)
    pub fn sti(&self) -> u64 {
        match self {
            Self::Heartbeat(m) => m.sti,
            Self::HeartbeatAck(m) => m.sti,
            Self::PduTransmission(m) => m.sti,
            Self::PduTransmissionAck(m) => m.sti,
        }
    }
}

/// Heartbeat message for cell search and connection maintenance
///
/// Sent by UE to discover and maintain connection with gNB.
/// Contains the simulated position of the UE.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RlsHeartbeat {
    /// Simulated Transmission Identifier - unique identifier for the sender
    pub sti: u64,
    /// Simulated position of the UE
    pub sim_pos: Vector3,
}

impl RlsHeartbeat {
    /// Creates a new heartbeat message
    pub fn new(sti: u64) -> Self {
        Self {
            sti,
            sim_pos: Vector3::default(),
        }
    }

    /// Creates a new heartbeat message with position
    pub fn with_position(sti: u64, sim_pos: Vector3) -> Self {
        Self { sti, sim_pos }
    }
}

/// Heartbeat acknowledgment message
///
/// Sent by gNB in response to a heartbeat, indicating signal strength.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RlsHeartbeatAck {
    /// Simulated Transmission Identifier
    pub sti: u64,
    /// Signal strength in dBm
    pub dbm: i32,
}

impl RlsHeartbeatAck {
    /// Creates a new heartbeat acknowledgment
    pub fn new(sti: u64) -> Self {
        Self { sti, dbm: 0 }
    }

    /// Creates a new heartbeat acknowledgment with signal strength
    pub fn with_dbm(sti: u64, dbm: i32) -> Self {
        Self { sti, dbm }
    }
}

/// PDU transmission message for RRC and user plane data
///
/// Used to transport RRC messages and user plane data between UE and gNB.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RlsPduTransmission {
    /// Simulated Transmission Identifier
    pub sti: u64,
    /// Type of PDU being transmitted
    pub pdu_type: PduType,
    /// PDU identifier for acknowledgment tracking
    pub pdu_id: u32,
    /// Payload metadata (e.g., RRC channel for RRC PDUs)
    pub payload: u32,
    /// The actual PDU data
    pub pdu: Bytes,
}

impl RlsPduTransmission {
    /// Creates a new PDU transmission message
    pub fn new(sti: u64) -> Self {
        Self {
            sti,
            pdu_type: PduType::Reserved,
            pdu_id: 0,
            payload: 0,
            pdu: Bytes::new(),
        }
    }

    /// Creates a new RRC PDU transmission
    pub fn rrc(sti: u64, pdu_id: u32, rrc_channel: u32, pdu: Bytes) -> Self {
        Self {
            sti,
            pdu_type: PduType::Rrc,
            pdu_id,
            payload: rrc_channel,
            pdu,
        }
    }

    /// Creates a new data PDU transmission
    pub fn data(sti: u64, pdu_id: u32, pdu: Bytes) -> Self {
        Self {
            sti,
            pdu_type: PduType::Data,
            pdu_id,
            payload: 0,
            pdu,
        }
    }
}

/// PDU transmission acknowledgment
///
/// Sent to acknowledge receipt of PDU transmissions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RlsPduTransmissionAck {
    /// Simulated Transmission Identifier
    pub sti: u64,
    /// List of acknowledged PDU IDs
    pub pdu_ids: Vec<u32>,
}

impl RlsPduTransmissionAck {
    /// Creates a new PDU transmission acknowledgment
    pub fn new(sti: u64) -> Self {
        Self {
            sti,
            pdu_ids: Vec::new(),
        }
    }

    /// Creates a new PDU transmission acknowledgment with PDU IDs
    pub fn with_pdu_ids(sti: u64, pdu_ids: Vec<u32>) -> Self {
        Self { sti, pdu_ids }
    }
}

/// Radio Link Failure cause
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RlfCause {
    /// PDU ID already exists
    PduIdExists,
    /// PDU ID buffer is full
    PduIdFull,
    /// Signal lost to connected cell
    SignalLostToConnectedCell,
}

/// PDU information for tracking transmissions
#[derive(Debug, Clone)]
pub struct PduInfo {
    /// PDU identifier
    pub id: u32,
    /// The PDU data
    pub pdu: Bytes,
    /// RRC channel (if applicable)
    pub rrc_channel: u32,
    /// Time when the PDU was sent (Unix timestamp in milliseconds)
    pub sent_time: i64,
    /// Endpoint identifier
    pub endpoint_id: i32,
}

impl PduInfo {
    /// Creates a new PDU info
    pub fn new(id: u32, pdu: Bytes) -> Self {
        Self {
            id,
            pdu,
            rrc_channel: 0,
            sent_time: 0,
            endpoint_id: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_type_from_u8() {
        assert_eq!(MessageType::from_u8(0), Some(MessageType::Reserved));
        assert_eq!(MessageType::from_u8(4), Some(MessageType::Heartbeat));
        assert_eq!(MessageType::from_u8(5), Some(MessageType::HeartbeatAck));
        assert_eq!(MessageType::from_u8(6), Some(MessageType::PduTransmission));
        assert_eq!(
            MessageType::from_u8(7),
            Some(MessageType::PduTransmissionAck)
        );
        assert_eq!(MessageType::from_u8(8), None);
    }

    #[test]
    fn test_pdu_type_from_u8() {
        assert_eq!(PduType::from_u8(0), Some(PduType::Reserved));
        assert_eq!(PduType::from_u8(1), Some(PduType::Rrc));
        assert_eq!(PduType::from_u8(2), Some(PduType::Data));
        assert_eq!(PduType::from_u8(3), None);
    }

    #[test]
    fn test_vector3() {
        let v = Vector3::new(1, 2, 3);
        assert_eq!(v.x, 1);
        assert_eq!(v.y, 2);
        assert_eq!(v.z, 3);
        assert_eq!(format!("{}", v), "(1, 2, 3)");
    }

    #[test]
    fn test_heartbeat() {
        let hb = RlsHeartbeat::with_position(12345, Vector3::new(100, 200, 300));
        assert_eq!(hb.sti, 12345);
        assert_eq!(hb.sim_pos.x, 100);
    }

    #[test]
    fn test_heartbeat_ack() {
        let ack = RlsHeartbeatAck::with_dbm(12345, -80);
        assert_eq!(ack.sti, 12345);
        assert_eq!(ack.dbm, -80);
    }

    #[test]
    fn test_pdu_transmission() {
        let pdu = RlsPduTransmission::rrc(12345, 1, 2, Bytes::from_static(b"test"));
        assert_eq!(pdu.sti, 12345);
        assert_eq!(pdu.pdu_type, PduType::Rrc);
        assert_eq!(pdu.pdu_id, 1);
        assert_eq!(pdu.payload, 2);
    }

    #[test]
    fn test_pdu_transmission_ack() {
        let ack = RlsPduTransmissionAck::with_pdu_ids(12345, vec![1, 2, 3]);
        assert_eq!(ack.sti, 12345);
        assert_eq!(ack.pdu_ids, vec![1, 2, 3]);
    }

    #[test]
    fn test_rls_message_enum() {
        let msg = RlsMessage::Heartbeat(RlsHeartbeat::new(12345));
        assert_eq!(msg.message_type(), MessageType::Heartbeat);
        assert_eq!(msg.sti(), 12345);
    }
}
