//! Notification Message (3GPP TS 24.501 Section 8.2.21)
//!
//! This module implements the Notification message sent by the network
//! to the UE to indicate pending downlink data or signalling.

use bytes::{Buf, BufMut};
use thiserror::Error;

use crate::enums::MmMessageType;
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

    /// Decode from bytes (after header has been parsed)
    ///
    /// The access type is encoded in the spare half-octet that was part
    /// of the header. It should be passed in from the header parsing.
    pub fn decode_with_access_type(access_type_val: u8) -> Result<Self, NotificationError> {
        let access_type = AccessType::try_from(access_type_val & 0x03)
            .map_err(|_| NotificationError::InvalidIeValue(
                format!("Invalid access type: 0x{access_type_val:02X}"),
            ))?;
        Ok(Self { access_type })
    }

    /// Decode from bytes (after header has been parsed)
    ///
    /// For the Notification message, the access type is in the spare
    /// half-octet of the security header type byte. Since the header
    /// decoder may have already consumed it, we read it from the
    /// remaining buffer if present. If the header parsing already
    /// extracted the access type, use `decode_with_access_type` instead.
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, NotificationError> {
        // The access type is encoded in the spare half of the
        // security header type octet. If the caller already parsed
        // the header, this byte may already be consumed.
        // In the simplest case with no additional IEs, the body is empty.
        // Default to 3GPP access if we cannot determine from buffer.
        if buf.remaining() >= 1 {
            let val = buf.get_u8();
            let access_type = AccessType::try_from(val & 0x03)
                .unwrap_or(AccessType::ThreeGppAccess);
            Ok(Self { access_type })
        } else {
            // No additional data; access type was in header spare bits
            Ok(Self::default())
        }
    }

    /// Encode to bytes (including header)
    ///
    /// The access type is encoded in the spare half-octet of the
    /// security header type octet (upper nibble).
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        // EPD
        buf.put_u8(0x7E); // MobilityManagement EPD

        // Security header type (lower nibble = 0x00) + access type (upper nibble)
        let access_val: u8 = self.access_type.into();
        buf.put_u8((access_val << 4) & 0xF0);

        // Message type
        buf.put_u8(MmMessageType::Notification as u8);
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
#[derive(Debug, Clone, PartialEq, Eq)]
#[derive(Default)]
pub struct NotificationResponse {
    /// PDU session status (optional, Type 4, IEI 0x50)
    pub pdu_session_status: Option<Vec<u8>>,
}

/// IEI constants for NotificationResponse optional IEs
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

        assert_eq!(buf.len(), 3);
        assert_eq!(buf[0], 0x7E); // EPD
        assert_eq!(buf[1], 0x10); // Access type (01) in upper nibble
        assert_eq!(buf[2], 0x65); // Message Type (Notification)
    }

    #[test]
    fn test_notification_encode_non_3gpp() {
        let msg = Notification::new(AccessType::NonThreeGppAccess);
        let mut buf = Vec::new();
        msg.encode(&mut buf);

        assert_eq!(buf.len(), 3);
        assert_eq!(buf[0], 0x7E); // EPD
        assert_eq!(buf[1], 0x20); // Access type (10) in upper nibble
        assert_eq!(buf[2], 0x65); // Message Type
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

        // Extract access type from the encoded header byte
        let access_type_val = (buf[1] >> 4) & 0x0F;
        let decoded = Notification::decode_with_access_type(access_type_val).unwrap();
        assert_eq!(decoded.access_type, AccessType::ThreeGppAccess);
    }

    // ========================================================================
    // Notification Response Tests
    // ========================================================================

    #[test]
    fn test_notification_response_new() {
        let msg = NotificationResponse::new();
        assert!(msg.pdu_session_status.is_none());
    }

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
