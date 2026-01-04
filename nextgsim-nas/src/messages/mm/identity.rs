//! Identity Messages (3GPP TS 24.501 Section 8.2.21, 8.2.22)
//!
//! This module implements the Identity procedure messages:
//! - Identity Request (network to UE)
//! - Identity Response (UE to network)

use bytes::{Buf, BufMut};
use thiserror::Error;

use crate::enums::MmMessageType;
use crate::header::PlainMmHeader;
use crate::ies::ie1::{Ie5gsIdentityType, IdentityType, InformationElement1};

use super::registration::Ie5gsMobileIdentity;

/// Error type for Identity message encoding/decoding
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum IdentityError {
    /// Buffer too short for decoding
    #[error("Buffer too short: expected at least {expected} bytes, got {actual}")]
    BufferTooShort {
        /// Expected minimum bytes
        expected: usize,
        /// Actual bytes available
        actual: usize,
    },
    /// Invalid message type
    #[error("Invalid message type: expected {expected:?}, got {actual:?}")]
    InvalidMessageType {
        /// Expected message type
        expected: MmMessageType,
        /// Actual message type
        actual: MmMessageType,
    },
    /// Invalid IE value
    #[error("Invalid IE value: {0}")]
    InvalidIeValue(String),
    /// Header decoding error
    #[error("Header error: {0}")]
    HeaderError(#[from] crate::header::HeaderError),
    /// IE1 decoding error
    #[error("IE1 error: {0}")]
    Ie1Error(#[from] crate::ies::ie1::Ie1Error),
}

// ============================================================================
// Identity Request (3GPP TS 24.501 Section 8.2.21)
// ============================================================================

/// Identity Request message (network to UE)
///
/// This message is sent by the network to request the UE to provide
/// a specific identity.
///
/// ## Message Structure
/// ```text
/// +------------------+------------------+------------------+------------------+
/// |       EPD        |  Security Header |   Message Type   |  Spare | ID Type |
/// |     (1 byte)     |  Type (1 byte)   |    (1 byte)      | (4 bits)|(4 bits) |
/// +------------------+------------------+------------------+------------------+
/// ```
///
/// ## Information Elements
/// - Identity type (M, Type 1, 1/2 octet) - 3GPP TS 24.501 Section 9.11.3.3
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IdentityRequest {
    /// 5GS Identity Type (mandatory)
    pub identity_type: Ie5gsIdentityType,
}

impl IdentityRequest {
    /// Minimum encoded size (header + identity type half-octet)
    pub const MIN_SIZE: usize = PlainMmHeader::SIZE + 1;

    /// Create a new Identity Request message
    pub fn new(identity_type: IdentityType) -> Self {
        Self {
            identity_type: Ie5gsIdentityType::new(identity_type),
        }
    }

    /// Decode an Identity Request message from bytes
    ///
    /// The buffer should start at the beginning of the NAS message (including header).
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, IdentityError> {
        if buf.remaining() < Self::MIN_SIZE {
            return Err(IdentityError::BufferTooShort {
                expected: Self::MIN_SIZE,
                actual: buf.remaining(),
            });
        }

        // Decode header
        let header = PlainMmHeader::decode(buf)?;
        if header.message_type != MmMessageType::IdentityRequest {
            return Err(IdentityError::InvalidMessageType {
                expected: MmMessageType::IdentityRequest,
                actual: header.message_type,
            });
        }

        // Decode identity type (Type 1 IE - lower 4 bits of next octet)
        if buf.remaining() < 1 {
            return Err(IdentityError::BufferTooShort {
                expected: 1,
                actual: buf.remaining(),
            });
        }
        let ie_octet = buf.get_u8();
        // Identity type is in the lower 4 bits (spare in upper 4 bits)
        let identity_type = Ie5gsIdentityType::decode(ie_octet & 0x0F)?;

        Ok(Self { identity_type })
    }

    /// Encode the Identity Request message to bytes
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        // Encode header
        let header = PlainMmHeader::new(MmMessageType::IdentityRequest);
        header.encode(buf);

        // Encode identity type (Type 1 IE - lower 4 bits, spare in upper 4 bits)
        let ie_octet = self.identity_type.encode() & 0x0F;
        buf.put_u8(ie_octet);
    }

    /// Get the encoded size of the message
    pub fn encoded_size(&self) -> usize {
        Self::MIN_SIZE
    }
}

impl Default for IdentityRequest {
    fn default() -> Self {
        Self::new(IdentityType::Suci)
    }
}

// ============================================================================
// Identity Response (3GPP TS 24.501 Section 8.2.22)
// ============================================================================

/// Identity Response message (UE to network)
///
/// This message is sent by the UE to the network in response to an
/// Identity Request message.
///
/// ## Message Structure
/// ```text
/// +------------------+------------------+------------------+
/// |       EPD        |  Security Header |   Message Type   |
/// |     (1 byte)     |  Type (1 byte)   |    (1 byte)      |
/// +------------------+------------------+------------------+
/// |              Mobile Identity (variable)                |
/// +--------------------------------------------------------+
/// ```
///
/// ## Information Elements
/// - Mobile identity (M, Type 6, 3-n octets) - 3GPP TS 24.501 Section 9.11.3.4
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IdentityResponse {
    /// 5GS Mobile Identity (mandatory)
    pub mobile_identity: Ie5gsMobileIdentity,
}

impl IdentityResponse {
    /// Minimum encoded size (header + minimum mobile identity)
    pub const MIN_SIZE: usize = PlainMmHeader::SIZE + 3; // header + 2-byte length + at least 1 byte

    /// Create a new Identity Response message
    pub fn new(mobile_identity: Ie5gsMobileIdentity) -> Self {
        Self { mobile_identity }
    }

    /// Create an Identity Response with no identity
    pub fn no_identity() -> Self {
        Self {
            mobile_identity: Ie5gsMobileIdentity::no_identity(),
        }
    }

    /// Decode an Identity Response message from bytes
    ///
    /// The buffer should start at the beginning of the NAS message (including header).
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, IdentityError> {
        if buf.remaining() < PlainMmHeader::SIZE {
            return Err(IdentityError::BufferTooShort {
                expected: PlainMmHeader::SIZE,
                actual: buf.remaining(),
            });
        }

        // Decode header
        let header = PlainMmHeader::decode(buf)?;
        if header.message_type != MmMessageType::IdentityResponse {
            return Err(IdentityError::InvalidMessageType {
                expected: MmMessageType::IdentityResponse,
                actual: header.message_type,
            });
        }

        // Decode mobile identity (Type 6 IE - 2-byte length prefix)
        let mobile_identity = Ie5gsMobileIdentity::decode(buf).map_err(|e| {
            IdentityError::InvalidIeValue(format!("Failed to decode mobile identity: {}", e))
        })?;

        Ok(Self { mobile_identity })
    }

    /// Encode the Identity Response message to bytes
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        // Encode header
        let header = PlainMmHeader::new(MmMessageType::IdentityResponse);
        header.encode(buf);

        // Encode mobile identity (Type 6 IE)
        self.mobile_identity.encode(buf);
    }

    /// Get the encoded size of the message
    pub fn encoded_size(&self) -> usize {
        PlainMmHeader::SIZE + self.mobile_identity.encoded_len()
    }
}

impl Default for IdentityResponse {
    fn default() -> Self {
        Self::no_identity()
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_request_encode_decode() {
        let msg = IdentityRequest::new(IdentityType::Suci);

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Verify encoded bytes
        assert_eq!(buf.len(), IdentityRequest::MIN_SIZE);
        assert_eq!(buf[0], 0x7E); // EPD: Mobility Management
        assert_eq!(buf[1], 0x00); // Security header type: Not protected
        assert_eq!(buf[2], 0x5B); // Message type: Identity Request
        assert_eq!(buf[3] & 0x0F, 0x01); // Identity type: SUCI

        // Decode and verify
        let decoded = IdentityRequest::decode(&mut buf.as_slice()).unwrap();
        assert_eq!(decoded.identity_type.value, IdentityType::Suci);
    }

    #[test]
    fn test_identity_request_all_identity_types() {
        let identity_types = [
            IdentityType::NoIdentity,
            IdentityType::Suci,
            IdentityType::Guti,
            IdentityType::Imei,
            IdentityType::Tmsi,
            IdentityType::ImeiSv,
        ];

        for identity_type in identity_types {
            let msg = IdentityRequest::new(identity_type);
            let mut buf = Vec::new();
            msg.encode(&mut buf);

            let decoded = IdentityRequest::decode(&mut buf.as_slice()).unwrap();
            assert_eq!(decoded.identity_type.value, identity_type);
        }
    }

    #[test]
    fn test_identity_response_encode_decode() {
        let mobile_identity = Ie5gsMobileIdentity::new(
            MobileIdentityType::Suci,
            vec![0x01, 0x02, 0x03, 0x04, 0x05],
        );
        let msg = IdentityResponse::new(mobile_identity.clone());

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Verify encoded bytes
        assert_eq!(buf[0], 0x7E); // EPD: Mobility Management
        assert_eq!(buf[1], 0x00); // Security header type: Not protected
        assert_eq!(buf[2], 0x5C); // Message type: Identity Response
        // Length is 2 bytes (big-endian)
        assert_eq!(buf[3], 0x00);
        assert_eq!(buf[4], 0x05); // Length = 5

        // Decode and verify
        let decoded = IdentityResponse::decode(&mut buf.as_slice()).unwrap();
        assert_eq!(decoded.mobile_identity.identity_type, MobileIdentityType::Suci);
        assert_eq!(decoded.mobile_identity.data, vec![0x01, 0x02, 0x03, 0x04, 0x05]);
    }

    #[test]
    fn test_identity_response_no_identity() {
        let msg = IdentityResponse::no_identity();

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        let decoded = IdentityResponse::decode(&mut buf.as_slice()).unwrap();
        assert_eq!(decoded.mobile_identity.identity_type, MobileIdentityType::NoIdentity);
    }

    #[test]
    fn test_identity_request_buffer_too_short() {
        let buf = vec![0x7E, 0x00]; // Only 2 bytes, need at least 4
        let result = IdentityRequest::decode(&mut buf.as_slice());
        assert!(matches!(result, Err(IdentityError::BufferTooShort { .. })));
    }

    #[test]
    fn test_identity_request_wrong_message_type() {
        // Create a valid header but with wrong message type (Registration Request)
        let buf = vec![0x7E, 0x00, 0x41, 0x01];
        let result = IdentityRequest::decode(&mut buf.as_slice());
        assert!(matches!(result, Err(IdentityError::InvalidMessageType { .. })));
    }

    #[test]
    fn test_identity_response_buffer_too_short() {
        let buf = vec![0x7E, 0x00]; // Only 2 bytes
        let result = IdentityResponse::decode(&mut buf.as_slice());
        assert!(matches!(result, Err(IdentityError::BufferTooShort { .. })));
    }

    #[test]
    fn test_identity_response_wrong_message_type() {
        // Create a valid header but with wrong message type (Identity Request)
        let buf = vec![0x7E, 0x00, 0x5B, 0x00, 0x01, 0x00];
        let result = IdentityResponse::decode(&mut buf.as_slice());
        assert!(matches!(result, Err(IdentityError::InvalidMessageType { .. })));
    }

    #[test]
    fn test_identity_request_encoded_size() {
        let msg = IdentityRequest::new(IdentityType::Guti);
        assert_eq!(msg.encoded_size(), IdentityRequest::MIN_SIZE);
    }

    #[test]
    fn test_identity_response_encoded_size() {
        let mobile_identity = Ie5gsMobileIdentity::new(
            MobileIdentityType::Guti,
            vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A],
        );
        let msg = IdentityResponse::new(mobile_identity);
        // Header (3) + Length (2) + Data (10) = 15
        assert_eq!(msg.encoded_size(), 15);
    }
}
