//! Notification Message (3GPP TS 24.501 Section 8.2.21)
//!
//! This module implements the Notification message sent by the network
//! to the UE to indicate pending downlink data or signalling.

use bytes::{Buf, BufMut};
use thiserror::Error;

use crate::enums::MmMessageType;
use crate::header::PlainMmHeader;
use crate::ies::ie1::AccessType;

/// Error type for Notification message encoding/decoding
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum NotificationError {
    /// Buffer too short for decoding
    #[error("Buffer too short: expected at least {expected} bytes, got {actual}")]
    BufferTooShort {
        /// Expected minimum bytes
        expected: usize,
        /// Actual bytes available
        actual: usize,
    },
    /// Invalid IE value
    #[error("Invalid IE value: {0}")]
    InvalidIeValue(String),
    /// Header decoding error
    #[error("Header error: {0}")]
    HeaderError(#[from] crate::header::HeaderError),
}

// ============================================================================
// Notification (3GPP TS 24.501 Section 8.2.21)
// ============================================================================

/// Notification message (network to UE)
///
/// This message is sent by the network to the UE to indicate access type
/// for the pending downlink data or signalling.
///
/// The access type is encoded in the spare half-octet of the security
/// header type octet (bits 5-8).
///
/// 3GPP TS 24.501 Section 8.2.21
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Notification {
    /// Access type (mandatory, half-octet)
    pub access_type: AccessType,
}

impl Default for Notification {
    fn default() -> Self {
        Self {
            access_type: AccessType::ThreeGppAccess,
        }
    }
}

impl Notification {
    /// Create a new Notification message
    pub fn new(access_type: AccessType) -> Self {
        Self { access_type }
    }

    /// Build from an already-extracted Access type half octet (bits 1-4 of the
    /// octet after the message type). Reserved values are refused.
    pub fn decode_with_access_type(access_type_val: u8) -> Result<Self, NotificationError> {
        let access_type = AccessType::try_from(access_type_val & 0x03).map_err(|_| {
            NotificationError::InvalidIeValue(format!(
                "Invalid access type: 0x{access_type_val:02X}"
            ))
        })?;
        Ok(Self { access_type })
    }

    /// Decode from bytes (after header has been parsed)
    ///
    /// The Access type is a mandatory V IE in bits 1-4 of the octet after the
    /// message type (TS 24.501 Table 8.2.23.1.1, §8.1 b)); its value is in bits
    /// 1-2 (§9.11.2.1A). A missing octet or a reserved value is an error: the
    /// old decoder silently defaulted both to 3GPP access.
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, NotificationError> {
        if buf.remaining() < 1 {
            return Err(NotificationError::BufferTooShort {
                expected: 1,
                actual: 0,
            });
        }
        Self::decode_with_access_type(buf.get_u8() & 0x0F)
    }

    /// Encode to bytes (including header)
    ///
    /// TS 24.501 Table 8.2.23.1.1: after the message type comes the Access type
    /// (V, 1/2) followed by a spare half octet, so per §8.1 b) the access type
    /// occupies bits 1-4 of octet 4 and bits 5-8 are spare. It used to be
    /// written into the spare half of the security header type octet (octet 2,
    /// which a receiver reads as part of the security header) and octet 4 was
    /// never written at all (nextgsim #257).
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        let header = PlainMmHeader::new(MmMessageType::Notification);
        header.encode(buf);
        let access_val: u8 = self.access_type.into();
        buf.put_u8(access_val & 0x0F);
    }

    /// Get the message type
    pub fn message_type() -> MmMessageType {
        MmMessageType::Notification
    }
}

// ============================================================================
// Notification Response (3GPP TS 24.501 Section 8.2.22)
// ============================================================================

/// Notification Response message (UE to network)
///
/// This message is sent by the UE to the network in response to a
/// Notification message. The UE acknowledges receipt of the notification.
///
/// 3GPP TS 24.501 Section 8.2.22
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct NotificationResponse {
    /// PDU session status (optional, Type 4, IEI 0x50)
    pub pdu_session_status: Option<Vec<u8>>,
}

/// IEI constants for `NotificationResponse` optional IEs
#[allow(dead_code)]
mod notification_response_iei {
    /// PDU session status
    pub const PDU_SESSION_STATUS: u8 = 0x50;
}

impl NotificationResponse {
    /// Create a new Notification Response message
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new Notification Response with PDU session status
    pub fn with_pdu_session_status(pdu_session_status: Vec<u8>) -> Self {
        Self {
            pdu_session_status: Some(pdu_session_status),
        }
    }

    /// Decode from bytes (after header has been parsed)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, NotificationError> {
        let mut msg = Self::new();

        // Parse optional IEs
        while buf.remaining() > 0 {
            let iei = buf.chunk()[0];

            match iei {
                notification_response_iei::PDU_SESSION_STATUS => {
                    buf.advance(1);
                    if buf.remaining() < 1 {
                        break;
                    }
                    let len = buf.get_u8() as usize;
                    if buf.remaining() < len {
                        break;
                    }
                    let mut data = vec![0u8; len];
                    buf.copy_to_slice(&mut data);
                    msg.pdu_session_status = Some(data);
                }
                _ => {
                    // Skip unknown IEs
                    buf.advance(1);
                    if buf.remaining() > 0 {
                        let len = buf.get_u8() as usize;
                        if buf.remaining() >= len {
                            buf.advance(len);
                        }
                    }
                }
            }
        }

        Ok(msg)
    }

    /// Encode to bytes (including header)
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        // Header
        buf.put_u8(0x7E); // EPD
        buf.put_u8(0x00); // Security header type
        buf.put_u8(MmMessageType::NotificationResponse as u8);

        // Optional IEs
        if let Some(ref status) = self.pdu_session_status {
            buf.put_u8(notification_response_iei::PDU_SESSION_STATUS);
            buf.put_u8(status.len() as u8);
            buf.put_slice(status);
        }
    }

    /// Get the message type
    pub fn message_type() -> MmMessageType {
        MmMessageType::NotificationResponse
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_notification_new() {
        let msg = Notification::new(AccessType::ThreeGppAccess);
        assert_eq!(msg.access_type, AccessType::ThreeGppAccess);
    }

    #[test]
    fn test_notification_new_non_3gpp() {
        let msg = Notification::new(AccessType::NonThreeGppAccess);
        assert_eq!(msg.access_type, AccessType::NonThreeGppAccess);
    }

    #[test]
    fn test_notification_encode() {
        let msg = Notification::new(AccessType::ThreeGppAccess);
        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Hand-derived from Table 8.2.23.1.1 + §8.1 b): EPD 7E; security
        // header type 0 / spare 0; message type 65; Access type 01 (3GPP) in
        // bits 1-4, spare half octet 0 in bits 5-8. The old encoder wrote
        // `7E 10 65`, folding the access type into the security header octet.
        assert_eq!(buf, [0x7E, 0x00, 0x65, 0x01]);
    }

    #[test]
    fn test_notification_encode_non_3gpp() {
        let msg = Notification::new(AccessType::NonThreeGppAccess);
        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Access type 10 (non-3GPP) in bits 1-4 of octet 4.
        assert_eq!(buf, [0x7E, 0x00, 0x65, 0x02]);
    }

    #[test]
    fn test_notification_decode_with_access_type() {
        let decoded = Notification::decode_with_access_type(0x01).unwrap();
        assert_eq!(decoded.access_type, AccessType::ThreeGppAccess);
    }

    #[test]
    fn test_notification_decode_with_access_type_non_3gpp() {
        let decoded = Notification::decode_with_access_type(0x02).unwrap();
        assert_eq!(decoded.access_type, AccessType::NonThreeGppAccess);
    }

    #[test]
    fn test_notification_message_type() {
        assert_eq!(Notification::message_type(), MmMessageType::Notification);
    }

    #[test]
    fn test_notification_default() {
        let msg = Notification::default();
        assert_eq!(msg.access_type, AccessType::ThreeGppAccess);
    }

    #[test]
    fn test_notification_encode_decode_roundtrip() {
        let msg = Notification::new(AccessType::ThreeGppAccess);
        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // The body after the 3-octet header is the Access type octet.
        let decoded = Notification::decode(&mut &buf[3..]).unwrap();
        assert_eq!(decoded.access_type, AccessType::ThreeGppAccess);
        let msg = Notification::new(AccessType::NonThreeGppAccess);
        let mut buf = Vec::new();
        msg.encode(&mut buf);
        let decoded = Notification::decode(&mut &buf[3..]).unwrap();
        assert_eq!(decoded.access_type, AccessType::NonThreeGppAccess);
    }

    /// The Access type is mandatory: a missing octet or a reserved value is an
    /// error, not a silent default to 3GPP access (nextgsim #257).
    #[test]
    fn test_notification_decode_refuses_missing_or_reserved_access_type() {
        assert!(Notification::decode(&mut &[][..]).is_err());
        assert!(Notification::decode(&mut &[0x00u8][..]).is_err());
        assert!(Notification::decode(&mut &[0x03u8][..]).is_err());
        // Bits 5-8 are the spare half octet and are ignored on receipt.
        assert_eq!(
            Notification::decode(&mut &[0xF2u8][..])
                .unwrap()
                .access_type,
            AccessType::NonThreeGppAccess
        );
    }

    // ========================================================================
    // Notification Response Tests
    // ========================================================================

    #[test]
    fn test_notification_response_new() {
        let msg = NotificationResponse::new();
        assert!(msg.pdu_session_status.is_none());
    }

    // The PDU session status value is carried as raw octets here (TS 24.501
    // §9.11.3.44): octet 3 is PSI(7)..PSI(0), octet 4 is PSI(15)..PSI(8). So
    // `00 20` below is PSI 13 (octet 4, bit 6), not PSI 5, which would be
    // `20 00`. These tests only round-trip the octets. A caller that fills the
    // field from a u16 bitmap (bit n = PSI n) must use `to_le_bytes()`.
    #[test]
    fn test_notification_response_with_pdu_session_status() {
        let msg = NotificationResponse::with_pdu_session_status(vec![0x00, 0x20]);
        assert_eq!(msg.pdu_session_status, Some(vec![0x00, 0x20]));
    }

    #[test]
    fn test_notification_response_encode_minimal() {
        let msg = NotificationResponse::new();
        let mut buf = Vec::new();
        msg.encode(&mut buf);

        assert_eq!(buf.len(), 3);
        assert_eq!(buf[0], 0x7E); // EPD
        assert_eq!(buf[1], 0x00); // Security header
        assert_eq!(buf[2], 0x66); // Message Type (NotificationResponse)
    }

    #[test]
    fn test_notification_response_encode_decode() {
        let msg = NotificationResponse::with_pdu_session_status(vec![0x00, 0x20]);
        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Skip header (3 bytes) and decode
        let decoded = NotificationResponse::decode(&mut &buf[3..]).unwrap();
        assert_eq!(decoded.pdu_session_status, Some(vec![0x00, 0x20]));
    }

    #[test]
    fn test_notification_response_encode_decode_empty() {
        let msg = NotificationResponse::new();
        let mut buf = Vec::new();
        msg.encode(&mut buf);

        let decoded = NotificationResponse::decode(&mut &buf[3..]).unwrap();
        assert!(decoded.pdu_session_status.is_none());
    }

    #[test]
    fn test_notification_response_message_type() {
        assert_eq!(
            NotificationResponse::message_type(),
            MmMessageType::NotificationResponse
        );
    }
}
