//! RLS message encoding/decoding
//!
//! This module provides functions to encode and decode RLS messages
//! for transmission over UDP.

use bytes::{Buf, BufMut, Bytes, BytesMut};
use thiserror::Error;

use crate::protocol::{
    version, MessageType, PduType, RlsHeartbeat, RlsHeartbeatAck, RlsMessage, RlsPduTransmission,
    RlsPduTransmissionAck, Vector3,
};

/// Maximum PDU length allowed (16KB)
const MAX_PDU_LENGTH: usize = 16384;

/// RLS compatibility marker (for old RLS compatibility)
const RLS_COMPAT_MARKER: u8 = 0x03;

/// Errors that can occur during RLS message encoding/decoding
#[derive(Debug, Error)]
pub enum RlsCodecError {
    /// Invalid compatibility marker
    #[error("invalid RLS compatibility marker: expected 0x03, got 0x{0:02X}")]
    InvalidCompatMarker(u8),

    /// Version mismatch
    #[error("RLS version mismatch: expected {}.{}.{}, got {}.{}.{}", 
            version::MAJOR, version::MINOR, version::PATCH, .0, .1, .2)]
    VersionMismatch(u8, u8, u8),

    /// Unknown message type
    #[error("unknown RLS message type: {0}")]
    UnknownMessageType(u8),

    /// Unknown PDU type
    #[error("unknown PDU type: {0}")]
    UnknownPduType(u8),

    /// PDU too large
    #[error("PDU length {0} exceeds maximum allowed {}", MAX_PDU_LENGTH)]
    PduTooLarge(usize),

    /// Buffer too short
    #[error("buffer too short: need {needed} bytes, have {available}")]
    BufferTooShort {
        /// Number of bytes needed
        needed: usize,
        /// Number of bytes available
        available: usize,
    },

    /// Deprecated message type
    #[error("deprecated message type: {0}")]
    DeprecatedMessageType(u8),
}

/// Result type for RLS codec operations
pub type Result<T> = std::result::Result<T, RlsCodecError>;

/// Encodes an RLS message into a byte buffer
pub fn encode(msg: &RlsMessage) -> Bytes {
    let mut buf = BytesMut::with_capacity(256);
    encode_into(msg, &mut buf);
    buf.freeze()
}

/// Encodes an RLS message into an existing buffer
pub fn encode_into(msg: &RlsMessage, buf: &mut BytesMut) {
    // Compatibility marker
    buf.put_u8(RLS_COMPAT_MARKER);

    // Version
    buf.put_u8(version::MAJOR);
    buf.put_u8(version::MINOR);
    buf.put_u8(version::PATCH);

    // Message type
    buf.put_u8(msg.message_type() as u8);

    // STI
    buf.put_u64(msg.sti());

    // Message-specific encoding
    match msg {
        RlsMessage::Heartbeat(m) => {
            buf.put_i32(m.sim_pos.x);
            buf.put_i32(m.sim_pos.y);
            buf.put_i32(m.sim_pos.z);
        }
        RlsMessage::HeartbeatAck(m) => {
            buf.put_i32(m.dbm);
        }
        RlsMessage::PduTransmission(m) => {
            buf.put_u8(m.pdu_type as u8);
            buf.put_u32(m.pdu_id);
            buf.put_u32(m.payload);
            buf.put_u32(m.pdu.len() as u32);
            buf.extend_from_slice(&m.pdu);
        }
        RlsMessage::PduTransmissionAck(m) => {
            buf.put_u32(m.pdu_ids.len() as u32);
            for pdu_id in &m.pdu_ids {
                buf.put_u32(*pdu_id);
            }
        }
    }
}

/// Decodes an RLS message from a byte buffer
pub fn decode(data: &[u8]) -> Result<RlsMessage> {
    let mut buf = data;

    // Check minimum length (marker + version + type + sti = 1 + 3 + 1 + 8 = 13)
    if buf.len() < 13 {
        return Err(RlsCodecError::BufferTooShort {
            needed: 13,
            available: buf.len(),
        });
    }

    // Compatibility marker
    let marker = buf.get_u8();
    if marker != RLS_COMPAT_MARKER {
        return Err(RlsCodecError::InvalidCompatMarker(marker));
    }

    // Version check
    let major = buf.get_u8();
    let minor = buf.get_u8();
    let patch = buf.get_u8();
    if major != version::MAJOR || minor != version::MINOR || patch != version::PATCH {
        return Err(RlsCodecError::VersionMismatch(major, minor, patch));
    }

    // Message type
    let msg_type_byte = buf.get_u8();
    let msg_type = MessageType::from_u8(msg_type_byte)
        .ok_or(RlsCodecError::UnknownMessageType(msg_type_byte))?;

    // STI
    if buf.len() < 8 {
        return Err(RlsCodecError::BufferTooShort {
            needed: 8,
            available: buf.len(),
        });
    }
    let sti = buf.get_u64();

    // Decode based on message type
    match msg_type {
        MessageType::Reserved => Err(RlsCodecError::UnknownMessageType(0)),
        MessageType::Deprecated1 | MessageType::Deprecated2 | MessageType::Deprecated3 => {
            Err(RlsCodecError::DeprecatedMessageType(msg_type_byte))
        }
        MessageType::Heartbeat => decode_heartbeat(sti, buf),
        MessageType::HeartbeatAck => decode_heartbeat_ack(sti, buf),
        MessageType::PduTransmission => decode_pdu_transmission(sti, buf),
        MessageType::PduTransmissionAck => decode_pdu_transmission_ack(sti, buf),
    }
}

fn decode_heartbeat(sti: u64, mut buf: &[u8]) -> Result<RlsMessage> {
    if buf.len() < 12 {
        return Err(RlsCodecError::BufferTooShort {
            needed: 12,
            available: buf.len(),
        });
    }

    let x = buf.get_i32();
    let y = buf.get_i32();
    let z = buf.get_i32();

    Ok(RlsMessage::Heartbeat(RlsHeartbeat::with_position(
        sti,
        Vector3::new(x, y, z),
    )))
}

fn decode_heartbeat_ack(sti: u64, mut buf: &[u8]) -> Result<RlsMessage> {
    if buf.len() < 4 {
        return Err(RlsCodecError::BufferTooShort {
            needed: 4,
            available: buf.len(),
        });
    }

    let dbm = buf.get_i32();

    Ok(RlsMessage::HeartbeatAck(RlsHeartbeatAck::with_dbm(
        sti, dbm,
    )))
}

fn decode_pdu_transmission(sti: u64, mut buf: &[u8]) -> Result<RlsMessage> {
    // Need at least: pdu_type(1) + pdu_id(4) + payload(4) + pdu_length(4) = 13
    if buf.len() < 13 {
        return Err(RlsCodecError::BufferTooShort {
            needed: 13,
            available: buf.len(),
        });
    }

    let pdu_type_byte = buf.get_u8();
    let pdu_type =
        PduType::from_u8(pdu_type_byte).ok_or(RlsCodecError::UnknownPduType(pdu_type_byte))?;

    let pdu_id = buf.get_u32();
    let payload = buf.get_u32();
    let pdu_length = buf.get_u32() as usize;

    if pdu_length > MAX_PDU_LENGTH {
        return Err(RlsCodecError::PduTooLarge(pdu_length));
    }

    if buf.len() < pdu_length {
        return Err(RlsCodecError::BufferTooShort {
            needed: pdu_length,
            available: buf.len(),
        });
    }

    let pdu = Bytes::copy_from_slice(&buf[..pdu_length]);

    Ok(RlsMessage::PduTransmission(RlsPduTransmission {
        sti,
        pdu_type,
        pdu_id,
        payload,
        pdu,
    }))
}

fn decode_pdu_transmission_ack(sti: u64, mut buf: &[u8]) -> Result<RlsMessage> {
    if buf.len() < 4 {
        return Err(RlsCodecError::BufferTooShort {
            needed: 4,
            available: buf.len(),
        });
    }

    let count = buf.get_u32() as usize;

    if buf.len() < count * 4 {
        return Err(RlsCodecError::BufferTooShort {
            needed: count * 4,
            available: buf.len(),
        });
    }

    let mut pdu_ids = Vec::with_capacity(count);
    for _ in 0..count {
        pdu_ids.push(buf.get_u32());
    }

    Ok(RlsMessage::PduTransmissionAck(
        RlsPduTransmissionAck::with_pdu_ids(sti, pdu_ids),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heartbeat_roundtrip() {
        let msg = RlsMessage::Heartbeat(RlsHeartbeat::with_position(
            0x123456789ABCDEF0,
            Vector3::new(100, -200, 300),
        ));

        let encoded = encode(&msg);
        let decoded = decode(&encoded).unwrap();

        assert_eq!(msg, decoded);
    }

    #[test]
    fn test_heartbeat_ack_roundtrip() {
        let msg = RlsMessage::HeartbeatAck(RlsHeartbeatAck::with_dbm(0xFEDCBA9876543210, -85));

        let encoded = encode(&msg);
        let decoded = decode(&encoded).unwrap();

        assert_eq!(msg, decoded);
    }

    #[test]
    fn test_pdu_transmission_roundtrip() {
        let msg = RlsMessage::PduTransmission(RlsPduTransmission {
            sti: 12345,
            pdu_type: PduType::Rrc,
            pdu_id: 42,
            payload: 7,
            pdu: Bytes::from_static(b"test pdu data"),
        });

        let encoded = encode(&msg);
        let decoded = decode(&encoded).unwrap();

        assert_eq!(msg, decoded);
    }

    #[test]
    fn test_pdu_transmission_ack_roundtrip() {
        let msg = RlsMessage::PduTransmissionAck(RlsPduTransmissionAck::with_pdu_ids(
            12345,
            vec![1, 2, 3, 4, 5],
        ));

        let encoded = encode(&msg);
        let decoded = decode(&encoded).unwrap();

        assert_eq!(msg, decoded);
    }

    #[test]
    fn test_empty_pdu_transmission() {
        let msg = RlsMessage::PduTransmission(RlsPduTransmission {
            sti: 12345,
            pdu_type: PduType::Data,
            pdu_id: 1,
            payload: 0,
            pdu: Bytes::new(),
        });

        let encoded = encode(&msg);
        let decoded = decode(&encoded).unwrap();

        assert_eq!(msg, decoded);
    }

    #[test]
    fn test_empty_pdu_ack() {
        let msg =
            RlsMessage::PduTransmissionAck(RlsPduTransmissionAck::with_pdu_ids(12345, vec![]));

        let encoded = encode(&msg);
        let decoded = decode(&encoded).unwrap();

        assert_eq!(msg, decoded);
    }

    #[test]
    fn test_invalid_compat_marker() {
        let data = [0x00, 0x03, 0x02, 0x07, 0x04, 0, 0, 0, 0, 0, 0, 0, 0];
        let result = decode(&data);
        assert!(matches!(result, Err(RlsCodecError::InvalidCompatMarker(0))));
    }

    #[test]
    fn test_version_mismatch() {
        let data = [0x03, 0x02, 0x02, 0x07, 0x04, 0, 0, 0, 0, 0, 0, 0, 0];
        let result = decode(&data);
        assert!(matches!(
            result,
            Err(RlsCodecError::VersionMismatch(2, 2, 7))
        ));
    }

    #[test]
    fn test_buffer_too_short() {
        let data = [0x03, 0x03, 0x02];
        let result = decode(&data);
        assert!(matches!(result, Err(RlsCodecError::BufferTooShort { .. })));
    }

    #[test]
    fn test_pdu_too_large() {
        let mut buf = BytesMut::new();
        buf.put_u8(RLS_COMPAT_MARKER);
        buf.put_u8(version::MAJOR);
        buf.put_u8(version::MINOR);
        buf.put_u8(version::PATCH);
        buf.put_u8(MessageType::PduTransmission as u8);
        buf.put_u64(12345);
        buf.put_u8(PduType::Data as u8);
        buf.put_u32(1);
        buf.put_u32(0);
        buf.put_u32(20000); // Too large

        let result = decode(&buf);
        assert!(matches!(result, Err(RlsCodecError::PduTooLarge(20000))));
    }
}
