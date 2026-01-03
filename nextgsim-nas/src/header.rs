//! NAS message header structures
//!
//! Implements 5G NAS message headers according to 3GPP TS 24.501
//!
//! # Header Types
//!
//! There are two main header formats:
//! - Plain NAS header (3 bytes for MM, 4 bytes for SM)
//! - Security protected NAS header (7 bytes)
//!
//! ## Plain 5GMM Header (3 bytes)
//! ```text
//! +------------------+------------------+------------------+
//! |       EPD        |  Security Header |   Message Type   |
//! |     (1 byte)     |  Type (4 bits)   |    (1 byte)      |
//! |                  |  Spare (4 bits)  |                  |
//! +------------------+------------------+------------------+
//! ```
//!
//! ## Plain 5GSM Header (4 bytes)
//! ```text
//! +------------------+------------------+------------------+------------------+
//! |       EPD        | PDU Session ID   |       PTI        |   Message Type   |
//! |     (1 byte)     |    (1 byte)      |    (1 byte)      |    (1 byte)      |
//! +------------------+------------------+------------------+------------------+
//! ```
//!
//! ## Security Protected Header (7 bytes)
//! ```text
//! +------------------+------------------+------------------+------------------+
//! |       EPD        |  Security Header |        Message Authentication      |
//! |     (1 byte)     |  Type (1 byte)   |           Code (4 bytes)           |
//! +------------------+------------------+------------------+------------------+
//! |  Sequence Number |                  Plain NAS Message                    |
//! |     (1 byte)     |                      (variable)                       |
//! +------------------+------------------------------------------------------ +
//! ```

use crate::enums::{
    ExtendedProtocolDiscriminator, MessageType, MmMessageType, SecurityHeaderType, SmMessageType,
};
use bytes::{Buf, BufMut};
use thiserror::Error;

/// NAS header decoding/encoding errors
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum HeaderError {
    #[error("Buffer too short: expected at least {expected} bytes, got {actual}")]
    BufferTooShort { expected: usize, actual: usize },

    #[error("Invalid extended protocol discriminator: 0x{0:02X}")]
    InvalidEpd(u8),

    #[error("Invalid security header type: 0x{0:02X}")]
    InvalidSecurityHeaderType(u8),

    #[error("Invalid message type: 0x{0:02X}")]
    InvalidMessageType(u8),

    #[error("EPD mismatch: expected {expected:?}, got {actual:?}")]
    EpdMismatch {
        expected: ExtendedProtocolDiscriminator,
        actual: ExtendedProtocolDiscriminator,
    },
}

/// Plain 5GMM NAS message header
///
/// Used for unprotected 5G Mobility Management messages.
/// Total size: 3 bytes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlainMmHeader {
    /// Extended Protocol Discriminator (always MobilityManagement)
    pub epd: ExtendedProtocolDiscriminator,
    /// Security header type (should be NotProtected for plain messages)
    pub security_header_type: SecurityHeaderType,
    /// Message type
    pub message_type: MmMessageType,
}

impl PlainMmHeader {
    /// Size of the plain MM header in bytes
    pub const SIZE: usize = 3;

    /// Create a new plain MM header
    pub fn new(message_type: MmMessageType) -> Self {
        Self {
            epd: ExtendedProtocolDiscriminator::MobilityManagement,
            security_header_type: SecurityHeaderType::NotProtected,
            message_type,
        }
    }

    /// Decode a plain MM header from bytes
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, HeaderError> {
        if buf.remaining() < Self::SIZE {
            return Err(HeaderError::BufferTooShort {
                expected: Self::SIZE,
                actual: buf.remaining(),
            });
        }

        let epd_byte = buf.get_u8();
        let epd = ExtendedProtocolDiscriminator::try_from(epd_byte)
            .map_err(|_| HeaderError::InvalidEpd(epd_byte))?;

        if epd != ExtendedProtocolDiscriminator::MobilityManagement {
            return Err(HeaderError::EpdMismatch {
                expected: ExtendedProtocolDiscriminator::MobilityManagement,
                actual: epd,
            });
        }

        let sht_byte = buf.get_u8();
        // Security header type is in the lower 4 bits
        let sht = SecurityHeaderType::try_from(sht_byte & 0x0F)
            .map_err(|_| HeaderError::InvalidSecurityHeaderType(sht_byte))?;

        let mt_byte = buf.get_u8();
        let message_type = MmMessageType::try_from(mt_byte)
            .map_err(|_| HeaderError::InvalidMessageType(mt_byte))?;

        Ok(Self {
            epd,
            security_header_type: sht,
            message_type,
        })
    }

    /// Encode the header to bytes
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        buf.put_u8(self.epd.into());
        // Security header type in lower 4 bits, spare in upper 4 bits
        buf.put_u8(u8::from(self.security_header_type) & 0x0F);
        buf.put_u8(self.message_type.into());
    }

    /// Get the encoded size
    pub fn encoded_size(&self) -> usize {
        Self::SIZE
    }
}

/// Plain 5GSM NAS message header
///
/// Used for 5G Session Management messages.
/// Total size: 4 bytes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlainSmHeader {
    /// Extended Protocol Discriminator (always SessionManagement)
    pub epd: ExtendedProtocolDiscriminator,
    /// PDU Session Identity
    pub pdu_session_id: u8,
    /// Procedure Transaction Identity
    pub pti: u8,
    /// Message type
    pub message_type: SmMessageType,
}

impl PlainSmHeader {
    /// Size of the plain SM header in bytes
    pub const SIZE: usize = 4;

    /// Create a new plain SM header
    pub fn new(pdu_session_id: u8, pti: u8, message_type: SmMessageType) -> Self {
        Self {
            epd: ExtendedProtocolDiscriminator::SessionManagement,
            pdu_session_id,
            pti,
            message_type,
        }
    }

    /// Decode a plain SM header from bytes
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, HeaderError> {
        if buf.remaining() < Self::SIZE {
            return Err(HeaderError::BufferTooShort {
                expected: Self::SIZE,
                actual: buf.remaining(),
            });
        }

        let epd_byte = buf.get_u8();
        let epd = ExtendedProtocolDiscriminator::try_from(epd_byte)
            .map_err(|_| HeaderError::InvalidEpd(epd_byte))?;

        if epd != ExtendedProtocolDiscriminator::SessionManagement {
            return Err(HeaderError::EpdMismatch {
                expected: ExtendedProtocolDiscriminator::SessionManagement,
                actual: epd,
            });
        }

        let pdu_session_id = buf.get_u8();
        let pti = buf.get_u8();

        let mt_byte = buf.get_u8();
        let message_type = SmMessageType::try_from(mt_byte)
            .map_err(|_| HeaderError::InvalidMessageType(mt_byte))?;

        Ok(Self {
            epd,
            pdu_session_id,
            pti,
            message_type,
        })
    }

    /// Encode the header to bytes
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        buf.put_u8(self.epd.into());
        buf.put_u8(self.pdu_session_id);
        buf.put_u8(self.pti);
        buf.put_u8(self.message_type.into());
    }

    /// Get the encoded size
    pub fn encoded_size(&self) -> usize {
        Self::SIZE
    }
}

/// Security protected NAS message header
///
/// Used for integrity protected and/or ciphered NAS messages.
/// Total size: 7 bytes (header only, excluding the plain NAS message)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SecuredHeader {
    /// Extended Protocol Discriminator (always MobilityManagement for secured messages)
    pub epd: ExtendedProtocolDiscriminator,
    /// Security header type
    pub security_header_type: SecurityHeaderType,
    /// Message Authentication Code (MAC)
    pub mac: [u8; 4],
    /// Sequence number
    pub sequence_number: u8,
}

impl SecuredHeader {
    /// Size of the secured header in bytes (excluding plain NAS message)
    pub const SIZE: usize = 7;

    /// Create a new secured header
    pub fn new(security_header_type: SecurityHeaderType, mac: [u8; 4], sequence_number: u8) -> Self {
        Self {
            epd: ExtendedProtocolDiscriminator::MobilityManagement,
            security_header_type,
            mac,
            sequence_number,
        }
    }

    /// Decode a secured header from bytes
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, HeaderError> {
        if buf.remaining() < Self::SIZE {
            return Err(HeaderError::BufferTooShort {
                expected: Self::SIZE,
                actual: buf.remaining(),
            });
        }

        let epd_byte = buf.get_u8();
        let epd = ExtendedProtocolDiscriminator::try_from(epd_byte)
            .map_err(|_| HeaderError::InvalidEpd(epd_byte))?;

        let sht_byte = buf.get_u8();
        let security_header_type = SecurityHeaderType::try_from(sht_byte & 0x0F)
            .map_err(|_| HeaderError::InvalidSecurityHeaderType(sht_byte))?;

        let mut mac = [0u8; 4];
        buf.copy_to_slice(&mut mac);

        let sequence_number = buf.get_u8();

        Ok(Self {
            epd,
            security_header_type,
            mac,
            sequence_number,
        })
    }

    /// Encode the header to bytes
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        buf.put_u8(self.epd.into());
        buf.put_u8(u8::from(self.security_header_type) & 0x0F);
        buf.put_slice(&self.mac);
        buf.put_u8(self.sequence_number);
    }

    /// Get the encoded size
    pub fn encoded_size(&self) -> usize {
        Self::SIZE
    }
}

/// Unified NAS header that can represent any header type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NasHeader {
    /// Plain 5GMM header
    PlainMm(PlainMmHeader),
    /// Plain 5GSM header
    PlainSm(PlainSmHeader),
    /// Security protected header
    Secured(SecuredHeader),
}

impl NasHeader {
    /// Peek at the first two bytes to determine header type without consuming
    pub fn peek_header_type(data: &[u8]) -> Result<NasHeaderType, HeaderError> {
        if data.len() < 2 {
            return Err(HeaderError::BufferTooShort {
                expected: 2,
                actual: data.len(),
            });
        }

        let epd = ExtendedProtocolDiscriminator::try_from(data[0])
            .map_err(|_| HeaderError::InvalidEpd(data[0]))?;

        match epd {
            ExtendedProtocolDiscriminator::MobilityManagement => {
                let sht = SecurityHeaderType::try_from(data[1] & 0x0F)
                    .map_err(|_| HeaderError::InvalidSecurityHeaderType(data[1]))?;

                if sht.is_protected() {
                    Ok(NasHeaderType::Secured)
                } else {
                    Ok(NasHeaderType::PlainMm)
                }
            }
            ExtendedProtocolDiscriminator::SessionManagement => Ok(NasHeaderType::PlainSm),
        }
    }

    /// Decode a NAS header from bytes
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, HeaderError> {
        if buf.remaining() < 2 {
            return Err(HeaderError::BufferTooShort {
                expected: 2,
                actual: buf.remaining(),
            });
        }

        // Peek at EPD and security header type
        let chunk = buf.chunk();
        let epd = ExtendedProtocolDiscriminator::try_from(chunk[0])
            .map_err(|_| HeaderError::InvalidEpd(chunk[0]))?;

        match epd {
            ExtendedProtocolDiscriminator::MobilityManagement => {
                let sht = SecurityHeaderType::try_from(chunk[1] & 0x0F)
                    .map_err(|_| HeaderError::InvalidSecurityHeaderType(chunk[1]))?;

                if sht.is_protected() {
                    Ok(NasHeader::Secured(SecuredHeader::decode(buf)?))
                } else {
                    Ok(NasHeader::PlainMm(PlainMmHeader::decode(buf)?))
                }
            }
            ExtendedProtocolDiscriminator::SessionManagement => {
                Ok(NasHeader::PlainSm(PlainSmHeader::decode(buf)?))
            }
        }
    }

    /// Encode the header to bytes
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        match self {
            NasHeader::PlainMm(h) => h.encode(buf),
            NasHeader::PlainSm(h) => h.encode(buf),
            NasHeader::Secured(h) => h.encode(buf),
        }
    }

    /// Get the encoded size
    pub fn encoded_size(&self) -> usize {
        match self {
            NasHeader::PlainMm(h) => h.encoded_size(),
            NasHeader::PlainSm(h) => h.encoded_size(),
            NasHeader::Secured(h) => h.encoded_size(),
        }
    }

    /// Get the EPD
    pub fn epd(&self) -> ExtendedProtocolDiscriminator {
        match self {
            NasHeader::PlainMm(h) => h.epd,
            NasHeader::PlainSm(h) => h.epd,
            NasHeader::Secured(h) => h.epd,
        }
    }

    /// Get the message type (if available without decryption)
    pub fn message_type(&self) -> Option<MessageType> {
        match self {
            NasHeader::PlainMm(h) => Some(MessageType::Mm(h.message_type)),
            NasHeader::PlainSm(h) => Some(MessageType::Sm(h.message_type)),
            NasHeader::Secured(_) => None, // Message type is in the encrypted payload
        }
    }

    /// Check if this is a security protected message
    pub fn is_protected(&self) -> bool {
        matches!(self, NasHeader::Secured(_))
    }
}

/// Type of NAS header
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NasHeaderType {
    /// Plain 5GMM header
    PlainMm,
    /// Plain 5GSM header
    PlainSm,
    /// Security protected header
    Secured,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plain_mm_header_encode_decode() {
        let header = PlainMmHeader::new(MmMessageType::RegistrationRequest);

        let mut buf = Vec::new();
        header.encode(&mut buf);

        assert_eq!(buf.len(), PlainMmHeader::SIZE);
        assert_eq!(buf[0], 0x7E); // EPD
        assert_eq!(buf[1], 0x00); // Security header type
        assert_eq!(buf[2], 0x41); // Message type

        let decoded = PlainMmHeader::decode(&mut buf.as_slice()).unwrap();
        assert_eq!(decoded, header);
    }

    #[test]
    fn test_plain_sm_header_encode_decode() {
        let header = PlainSmHeader::new(5, 1, SmMessageType::PduSessionEstablishmentRequest);

        let mut buf = Vec::new();
        header.encode(&mut buf);

        assert_eq!(buf.len(), PlainSmHeader::SIZE);
        assert_eq!(buf[0], 0x2E); // EPD
        assert_eq!(buf[1], 5); // PDU Session ID
        assert_eq!(buf[2], 1); // PTI
        assert_eq!(buf[3], 0xC1); // Message type

        let decoded = PlainSmHeader::decode(&mut buf.as_slice()).unwrap();
        assert_eq!(decoded, header);
    }

    #[test]
    fn test_secured_header_encode_decode() {
        let header = SecuredHeader::new(
            SecurityHeaderType::IntegrityProtectedAndCiphered,
            [0x12, 0x34, 0x56, 0x78],
            42,
        );

        let mut buf = Vec::new();
        header.encode(&mut buf);

        assert_eq!(buf.len(), SecuredHeader::SIZE);
        assert_eq!(buf[0], 0x7E); // EPD
        assert_eq!(buf[1], 0x02); // Security header type
        assert_eq!(&buf[2..6], &[0x12, 0x34, 0x56, 0x78]); // MAC
        assert_eq!(buf[6], 42); // Sequence number

        let decoded = SecuredHeader::decode(&mut buf.as_slice()).unwrap();
        assert_eq!(decoded, header);
    }

    #[test]
    fn test_nas_header_peek() {
        // Plain MM
        let plain_mm = [0x7E, 0x00, 0x41];
        assert_eq!(
            NasHeader::peek_header_type(&plain_mm).unwrap(),
            NasHeaderType::PlainMm
        );

        // Secured
        let secured = [0x7E, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00];
        assert_eq!(
            NasHeader::peek_header_type(&secured).unwrap(),
            NasHeaderType::Secured
        );

        // Plain SM
        let plain_sm = [0x2E, 0x05, 0x01, 0xC1];
        assert_eq!(
            NasHeader::peek_header_type(&plain_sm).unwrap(),
            NasHeaderType::PlainSm
        );
    }

    #[test]
    fn test_nas_header_decode_plain_mm() {
        let data = [0x7E, 0x00, 0x56]; // Authentication Request
        let header = NasHeader::decode(&mut data.as_slice()).unwrap();

        match header {
            NasHeader::PlainMm(h) => {
                assert_eq!(h.message_type, MmMessageType::AuthenticationRequest);
                assert_eq!(h.security_header_type, SecurityHeaderType::NotProtected);
            }
            _ => panic!("Expected PlainMm header"),
        }
    }

    #[test]
    fn test_nas_header_decode_secured() {
        let data = [0x7E, 0x01, 0xAA, 0xBB, 0xCC, 0xDD, 0x10];
        let header = NasHeader::decode(&mut data.as_slice()).unwrap();

        match header {
            NasHeader::Secured(h) => {
                assert_eq!(h.security_header_type, SecurityHeaderType::IntegrityProtected);
                assert_eq!(h.mac, [0xAA, 0xBB, 0xCC, 0xDD]);
                assert_eq!(h.sequence_number, 0x10);
            }
            _ => panic!("Expected Secured header"),
        }
    }

    #[test]
    fn test_buffer_too_short() {
        let data = [0x7E];
        let result = NasHeader::decode(&mut data.as_slice());
        assert!(matches!(result, Err(HeaderError::BufferTooShort { .. })));
    }

    #[test]
    fn test_invalid_epd() {
        let data = [0xFF, 0x00, 0x41];
        let result = NasHeader::decode(&mut data.as_slice());
        assert!(matches!(result, Err(HeaderError::InvalidEpd(0xFF))));
    }
}
