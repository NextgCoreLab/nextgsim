//! NAS message encoding/decoding traits and utilities
//!
//! This module provides traits for encoding and decoding NAS protocol messages
//! and Information Elements (IEs) according to 3GPP TS 24.501.
//!
//! # Traits
//!
//! - [`NasEncode`]: Trait for encoding types to bytes
//! - [`NasDecode`]: Trait for decoding types from bytes
//!
//! # Information Element Types
//!
//! NAS protocol defines several IE formats:
//! - Type 1: Half-octet (4 bits) value
//! - Type 2: Type-only IE (no value)
//! - Type 3: Fixed-length value
//! - Type 4: Variable-length with 1-byte length field
//! - Type 6: Variable-length with 2-byte length field
//!
//! # Example
//!
//! ```rust
//! use nextgsim_nas::codec::{NasEncode, NasDecode, CodecError};
//! use nextgsim_nas::header::PlainMmHeader;
//! use nextgsim_nas::enums::MmMessageType;
//!
//! // Encode a header
//! let header = PlainMmHeader::new(MmMessageType::RegistrationRequest);
//! let mut buf = Vec::new();
//! header.nas_encode(&mut buf).unwrap();
//!
//! // Decode a header
//! let decoded = PlainMmHeader::nas_decode(&mut buf.as_slice()).unwrap();
//! assert_eq!(decoded.message_type, MmMessageType::RegistrationRequest);
//! ```

use bytes::{Buf, BufMut};
use thiserror::Error;

/// Errors that can occur during NAS encoding/decoding
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CodecError {
    /// Buffer does not have enough bytes for decoding
    #[error("Buffer too short: expected at least {expected} bytes, got {actual}")]
    BufferTooShort {
        /// Expected minimum bytes
        expected: usize,
        /// Actual bytes available
        actual: usize,
    },

    /// Invalid value encountered during decoding
    #[error("Invalid value: {0}")]
    InvalidValue(String),

    /// Invalid Information Element Identifier
    #[error("Invalid IEI: 0x{0:02X}")]
    InvalidIei(u8),

    /// Length field exceeds remaining buffer
    #[error("Length exceeds buffer: length field is {length}, but only {remaining} bytes remain")]
    LengthExceedsBuffer {
        /// Length specified in the length field
        length: usize,
        /// Remaining bytes in buffer
        remaining: usize,
    },

    /// Unexpected end of buffer
    #[error("Unexpected end of buffer")]
    UnexpectedEnd,

    /// Invalid protocol discriminator
    #[error("Invalid protocol discriminator: 0x{0:02X}")]
    InvalidProtocolDiscriminator(u8),

    /// Invalid message type
    #[error("Invalid message type: 0x{0:02X}")]
    InvalidMessageType(u8),

    /// Invalid security header type
    #[error("Invalid security header type: 0x{0:02X}")]
    InvalidSecurityHeaderType(u8),

    /// Encoding error
    #[error("Encoding error: {0}")]
    EncodingError(String),
}

/// Result type for NAS codec operations
pub type CodecResult<T> = Result<T, CodecError>;

/// Trait for encoding NAS messages and Information Elements to bytes
///
/// Types implementing this trait can be serialized to a byte buffer
/// following the NAS protocol encoding rules.
pub trait NasEncode {
    /// Encode this value to the provided buffer
    ///
    /// # Arguments
    /// * `buf` - Mutable buffer to write encoded bytes to
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err(CodecError)` if encoding fails
    fn nas_encode<B: BufMut>(&self, buf: &mut B) -> CodecResult<()>;

    /// Returns the encoded size in bytes
    ///
    /// This is useful for pre-allocating buffers or calculating
    /// length fields before encoding.
    fn encoded_len(&self) -> usize;
}

/// Trait for decoding NAS messages and Information Elements from bytes
///
/// Types implementing this trait can be deserialized from a byte buffer
/// following the NAS protocol decoding rules.
pub trait NasDecode: Sized {
    /// Decode a value from the provided buffer
    ///
    /// # Arguments
    /// * `buf` - Buffer to read encoded bytes from
    ///
    /// # Returns
    /// * `Ok(Self)` on success
    /// * `Err(CodecError)` if decoding fails
    fn nas_decode<B: Buf>(buf: &mut B) -> CodecResult<Self>;
}

/// Marker trait for Type 1 Information Elements (half-octet, 4 bits)
///
/// Type 1 IEs contain a 4-bit value in the lower nibble of an octet.
/// The upper nibble may contain an IEI or another Type 1 IE value.
pub trait InformationElement1: NasEncode + NasDecode {
    /// Encode the IE value to a 4-bit nibble (0-15)
    fn encode_value(&self) -> u8;

    /// Decode the IE from a 4-bit nibble value
    fn decode_value(value: u8) -> CodecResult<Self>;
}

/// Marker trait for Type 2 Information Elements (type-only, no value)
///
/// Type 2 IEs consist only of an IEI octet with no value field.
/// Their presence indicates a boolean condition.
pub trait InformationElement2: Default {}

/// Marker trait for Type 3 Information Elements (fixed-length value)
///
/// Type 3 IEs have a fixed-length value field. The length is
/// determined by the IE type and is not encoded in the message.
pub trait InformationElement3: NasEncode + NasDecode {
    /// The fixed length of this IE's value in bytes
    const LENGTH: usize;
}

/// Marker trait for Type 4 Information Elements (variable-length, 1-byte length)
///
/// Type 4 IEs have a variable-length value field preceded by a
/// 1-byte length indicator (max 255 bytes).
pub trait InformationElement4: NasEncode + NasDecode {}

/// Marker trait for Type 6 Information Elements (variable-length, 2-byte length)
///
/// Type 6 IEs have a variable-length value field preceded by a
/// 2-byte length indicator (max 65535 bytes).
pub trait InformationElement6: NasEncode + NasDecode {}

// ============================================================================
// Encoding/Decoding helper functions for IE types
// ============================================================================

/// Encode two Type 1 IEs into a single octet
///
/// # Arguments
/// * `high` - IE value for the high nibble (bits 7-4)
/// * `low` - IE value for the low nibble (bits 3-0)
/// * `buf` - Buffer to write to
pub fn encode_ie1_pair<B: BufMut, H: InformationElement1, L: InformationElement1>(
    high: &H,
    low: &L,
    buf: &mut B,
) {
    let high_nibble = high.encode_value() & 0x0F;
    let low_nibble = low.encode_value() & 0x0F;
    buf.put_u8((high_nibble << 4) | low_nibble);
}

/// Encode a Type 1 IE with an IEI in the high nibble
///
/// # Arguments
/// * `iei` - Information Element Identifier (4 bits)
/// * `ie` - The Type 1 IE to encode
/// * `buf` - Buffer to write to
pub fn encode_ie1_with_iei<B: BufMut, T: InformationElement1>(iei: u8, ie: &T, buf: &mut B) {
    let value = ie.encode_value() & 0x0F;
    buf.put_u8(((iei & 0x0F) << 4) | value);
}

/// Decode a Type 1 IE from the low nibble of an octet
///
/// # Arguments
/// * `buf` - Buffer to read from
///
/// # Returns
/// The decoded IE and the high nibble value
pub fn decode_ie1<B: Buf, T: InformationElement1>(buf: &mut B) -> CodecResult<(u8, T)> {
    if buf.remaining() < 1 {
        return Err(CodecError::BufferTooShort {
            expected: 1,
            actual: buf.remaining(),
        });
    }
    let octet = buf.get_u8();
    let high_nibble = (octet >> 4) & 0x0F;
    let low_nibble = octet & 0x0F;
    let ie = T::decode_value(low_nibble)?;
    Ok((high_nibble, ie))
}

/// Encode a Type 3 IE with an IEI prefix
///
/// # Arguments
/// * `iei` - Information Element Identifier
/// * `ie` - The Type 3 IE to encode
/// * `buf` - Buffer to write to
pub fn encode_ie3_with_iei<B: BufMut, T: InformationElement3>(
    iei: u8,
    ie: &T,
    buf: &mut B,
) -> CodecResult<()> {
    buf.put_u8(iei);
    ie.nas_encode(buf)
}

/// Decode a Type 3 IE (assumes IEI already consumed)
///
/// # Arguments
/// * `buf` - Buffer to read from
pub fn decode_ie3<B: Buf, T: InformationElement3>(buf: &mut B) -> CodecResult<T> {
    T::nas_decode(buf)
}

/// Encode a Type 4 IE with an IEI prefix and length field
///
/// # Arguments
/// * `iei` - Information Element Identifier
/// * `ie` - The Type 4 IE to encode
/// * `buf` - Buffer to write to
pub fn encode_ie4_with_iei<B: BufMut, T: InformationElement4>(
    iei: u8,
    ie: &T,
    buf: &mut B,
) -> CodecResult<()> {
    buf.put_u8(iei);
    let len = ie.encoded_len();
    if len > 255 {
        return Err(CodecError::EncodingError(format!(
            "Type 4 IE length {} exceeds maximum of 255",
            len
        )));
    }
    buf.put_u8(len as u8);
    ie.nas_encode(buf)
}

/// Decode a Type 4 IE (assumes IEI already consumed)
///
/// # Arguments
/// * `buf` - Buffer to read from
pub fn decode_ie4<B: Buf, T: InformationElement4>(buf: &mut B) -> CodecResult<T> {
    if buf.remaining() < 1 {
        return Err(CodecError::BufferTooShort {
            expected: 1,
            actual: buf.remaining(),
        });
    }
    let length = buf.get_u8() as usize;
    if buf.remaining() < length {
        return Err(CodecError::LengthExceedsBuffer {
            length,
            remaining: buf.remaining(),
        });
    }
    T::nas_decode(buf)
}

/// Encode a Type 6 IE with an IEI prefix and 2-byte length field
///
/// # Arguments
/// * `iei` - Information Element Identifier
/// * `ie` - The Type 6 IE to encode
/// * `buf` - Buffer to write to
pub fn encode_ie6_with_iei<B: BufMut, T: InformationElement6>(
    iei: u8,
    ie: &T,
    buf: &mut B,
) -> CodecResult<()> {
    buf.put_u8(iei);
    let len = ie.encoded_len();
    if len > 65535 {
        return Err(CodecError::EncodingError(format!(
            "Type 6 IE length {} exceeds maximum of 65535",
            len
        )));
    }
    buf.put_u16(len as u16);
    ie.nas_encode(buf)
}

/// Decode a Type 6 IE (assumes IEI already consumed)
///
/// # Arguments
/// * `buf` - Buffer to read from
pub fn decode_ie6<B: Buf, T: InformationElement6>(buf: &mut B) -> CodecResult<T> {
    if buf.remaining() < 2 {
        return Err(CodecError::BufferTooShort {
            expected: 2,
            actual: buf.remaining(),
        });
    }
    let length = buf.get_u16() as usize;
    if buf.remaining() < length {
        return Err(CodecError::LengthExceedsBuffer {
            length,
            remaining: buf.remaining(),
        });
    }
    T::nas_decode(buf)
}

// ============================================================================
// NasEncode/NasDecode implementations for primitive types
// ============================================================================

impl NasEncode for u8 {
    fn nas_encode<B: BufMut>(&self, buf: &mut B) -> CodecResult<()> {
        buf.put_u8(*self);
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        1
    }
}

impl NasDecode for u8 {
    fn nas_decode<B: Buf>(buf: &mut B) -> CodecResult<Self> {
        if buf.remaining() < 1 {
            return Err(CodecError::BufferTooShort {
                expected: 1,
                actual: buf.remaining(),
            });
        }
        Ok(buf.get_u8())
    }
}

impl NasEncode for u16 {
    fn nas_encode<B: BufMut>(&self, buf: &mut B) -> CodecResult<()> {
        buf.put_u16(*self);
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        2
    }
}

impl NasDecode for u16 {
    fn nas_decode<B: Buf>(buf: &mut B) -> CodecResult<Self> {
        if buf.remaining() < 2 {
            return Err(CodecError::BufferTooShort {
                expected: 2,
                actual: buf.remaining(),
            });
        }
        Ok(buf.get_u16())
    }
}

impl NasEncode for u32 {
    fn nas_encode<B: BufMut>(&self, buf: &mut B) -> CodecResult<()> {
        buf.put_u32(*self);
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        4
    }
}

impl NasDecode for u32 {
    fn nas_decode<B: Buf>(buf: &mut B) -> CodecResult<Self> {
        if buf.remaining() < 4 {
            return Err(CodecError::BufferTooShort {
                expected: 4,
                actual: buf.remaining(),
            });
        }
        Ok(buf.get_u32())
    }
}

impl NasEncode for [u8; 4] {
    fn nas_encode<B: BufMut>(&self, buf: &mut B) -> CodecResult<()> {
        buf.put_slice(self);
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        4
    }
}

impl NasDecode for [u8; 4] {
    fn nas_decode<B: Buf>(buf: &mut B) -> CodecResult<Self> {
        if buf.remaining() < 4 {
            return Err(CodecError::BufferTooShort {
                expected: 4,
                actual: buf.remaining(),
            });
        }
        let mut arr = [0u8; 4];
        buf.copy_to_slice(&mut arr);
        Ok(arr)
    }
}

impl NasEncode for Vec<u8> {
    fn nas_encode<B: BufMut>(&self, buf: &mut B) -> CodecResult<()> {
        buf.put_slice(self);
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        self.len()
    }
}

// ============================================================================
// NasEncode/NasDecode implementations for header types
// ============================================================================

use crate::enums::{
    ExtendedProtocolDiscriminator, MmMessageType, SecurityHeaderType, SmMessageType,
};
use crate::header::{HeaderError, NasHeader, PlainMmHeader, PlainSmHeader, SecuredHeader};

impl NasEncode for PlainMmHeader {
    fn nas_encode<B: BufMut>(&self, buf: &mut B) -> CodecResult<()> {
        self.encode(buf);
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        Self::SIZE
    }
}

impl NasDecode for PlainMmHeader {
    fn nas_decode<B: Buf>(buf: &mut B) -> CodecResult<Self> {
        PlainMmHeader::decode(buf).map_err(|e| match e {
            HeaderError::BufferTooShort { expected, actual } => {
                CodecError::BufferTooShort { expected, actual }
            }
            HeaderError::InvalidEpd(v) => CodecError::InvalidProtocolDiscriminator(v),
            HeaderError::InvalidSecurityHeaderType(v) => CodecError::InvalidSecurityHeaderType(v),
            HeaderError::InvalidMessageType(v) => CodecError::InvalidMessageType(v),
            HeaderError::EpdMismatch { expected, actual } => CodecError::InvalidValue(format!(
                "EPD mismatch: expected {:?}, got {:?}",
                expected, actual
            )),
        })
    }
}

impl NasEncode for PlainSmHeader {
    fn nas_encode<B: BufMut>(&self, buf: &mut B) -> CodecResult<()> {
        self.encode(buf);
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        Self::SIZE
    }
}

impl NasDecode for PlainSmHeader {
    fn nas_decode<B: Buf>(buf: &mut B) -> CodecResult<Self> {
        PlainSmHeader::decode(buf).map_err(|e| match e {
            HeaderError::BufferTooShort { expected, actual } => {
                CodecError::BufferTooShort { expected, actual }
            }
            HeaderError::InvalidEpd(v) => CodecError::InvalidProtocolDiscriminator(v),
            HeaderError::InvalidSecurityHeaderType(v) => CodecError::InvalidSecurityHeaderType(v),
            HeaderError::InvalidMessageType(v) => CodecError::InvalidMessageType(v),
            HeaderError::EpdMismatch { expected, actual } => CodecError::InvalidValue(format!(
                "EPD mismatch: expected {:?}, got {:?}",
                expected, actual
            )),
        })
    }
}

impl NasEncode for SecuredHeader {
    fn nas_encode<B: BufMut>(&self, buf: &mut B) -> CodecResult<()> {
        self.encode(buf);
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        Self::SIZE
    }
}

impl NasDecode for SecuredHeader {
    fn nas_decode<B: Buf>(buf: &mut B) -> CodecResult<Self> {
        SecuredHeader::decode(buf).map_err(|e| match e {
            HeaderError::BufferTooShort { expected, actual } => {
                CodecError::BufferTooShort { expected, actual }
            }
            HeaderError::InvalidEpd(v) => CodecError::InvalidProtocolDiscriminator(v),
            HeaderError::InvalidSecurityHeaderType(v) => CodecError::InvalidSecurityHeaderType(v),
            HeaderError::InvalidMessageType(v) => CodecError::InvalidMessageType(v),
            HeaderError::EpdMismatch { expected, actual } => CodecError::InvalidValue(format!(
                "EPD mismatch: expected {:?}, got {:?}",
                expected, actual
            )),
        })
    }
}

impl NasEncode for NasHeader {
    fn nas_encode<B: BufMut>(&self, buf: &mut B) -> CodecResult<()> {
        self.encode(buf);
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        self.encoded_size()
    }
}

impl NasDecode for NasHeader {
    fn nas_decode<B: Buf>(buf: &mut B) -> CodecResult<Self> {
        NasHeader::decode(buf).map_err(|e| match e {
            HeaderError::BufferTooShort { expected, actual } => {
                CodecError::BufferTooShort { expected, actual }
            }
            HeaderError::InvalidEpd(v) => CodecError::InvalidProtocolDiscriminator(v),
            HeaderError::InvalidSecurityHeaderType(v) => CodecError::InvalidSecurityHeaderType(v),
            HeaderError::InvalidMessageType(v) => CodecError::InvalidMessageType(v),
            HeaderError::EpdMismatch { expected, actual } => CodecError::InvalidValue(format!(
                "EPD mismatch: expected {:?}, got {:?}",
                expected, actual
            )),
        })
    }
}

// ============================================================================
// NasEncode/NasDecode implementations for enum types
// ============================================================================

impl NasEncode for ExtendedProtocolDiscriminator {
    fn nas_encode<B: BufMut>(&self, buf: &mut B) -> CodecResult<()> {
        buf.put_u8((*self).into());
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        1
    }
}

impl NasDecode for ExtendedProtocolDiscriminator {
    fn nas_decode<B: Buf>(buf: &mut B) -> CodecResult<Self> {
        if buf.remaining() < 1 {
            return Err(CodecError::BufferTooShort {
                expected: 1,
                actual: buf.remaining(),
            });
        }
        let value = buf.get_u8();
        ExtendedProtocolDiscriminator::try_from(value)
            .map_err(|_| CodecError::InvalidProtocolDiscriminator(value))
    }
}

impl NasEncode for SecurityHeaderType {
    fn nas_encode<B: BufMut>(&self, buf: &mut B) -> CodecResult<()> {
        buf.put_u8((*self).into());
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        1
    }
}

impl NasDecode for SecurityHeaderType {
    fn nas_decode<B: Buf>(buf: &mut B) -> CodecResult<Self> {
        if buf.remaining() < 1 {
            return Err(CodecError::BufferTooShort {
                expected: 1,
                actual: buf.remaining(),
            });
        }
        let value = buf.get_u8();
        SecurityHeaderType::try_from(value & 0x0F)
            .map_err(|_| CodecError::InvalidSecurityHeaderType(value))
    }
}

impl NasEncode for MmMessageType {
    fn nas_encode<B: BufMut>(&self, buf: &mut B) -> CodecResult<()> {
        buf.put_u8((*self).into());
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        1
    }
}

impl NasDecode for MmMessageType {
    fn nas_decode<B: Buf>(buf: &mut B) -> CodecResult<Self> {
        if buf.remaining() < 1 {
            return Err(CodecError::BufferTooShort {
                expected: 1,
                actual: buf.remaining(),
            });
        }
        let value = buf.get_u8();
        MmMessageType::try_from(value).map_err(|_| CodecError::InvalidMessageType(value))
    }
}

impl NasEncode for SmMessageType {
    fn nas_encode<B: BufMut>(&self, buf: &mut B) -> CodecResult<()> {
        buf.put_u8((*self).into());
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        1
    }
}

impl NasDecode for SmMessageType {
    fn nas_decode<B: Buf>(buf: &mut B) -> CodecResult<Self> {
        if buf.remaining() < 1 {
            return Err(CodecError::BufferTooShort {
                expected: 1,
                actual: buf.remaining(),
            });
        }
        let value = buf.get_u8();
        SmMessageType::try_from(value).map_err(|_| CodecError::InvalidMessageType(value))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enums::MmMessageType;

    #[test]
    fn test_u8_encode_decode() {
        let value: u8 = 0x42;
        let mut buf = Vec::new();
        value.nas_encode(&mut buf).unwrap();
        assert_eq!(buf, vec![0x42]);

        let decoded = u8::nas_decode(&mut buf.as_slice()).unwrap();
        assert_eq!(decoded, value);
    }

    #[test]
    fn test_u16_encode_decode() {
        let value: u16 = 0x1234;
        let mut buf = Vec::new();
        value.nas_encode(&mut buf).unwrap();
        assert_eq!(buf, vec![0x12, 0x34]);

        let decoded = u16::nas_decode(&mut buf.as_slice()).unwrap();
        assert_eq!(decoded, value);
    }

    #[test]
    fn test_u32_encode_decode() {
        let value: u32 = 0x12345678;
        let mut buf = Vec::new();
        value.nas_encode(&mut buf).unwrap();
        assert_eq!(buf, vec![0x12, 0x34, 0x56, 0x78]);

        let decoded = u32::nas_decode(&mut buf.as_slice()).unwrap();
        assert_eq!(decoded, value);
    }

    #[test]
    fn test_array_encode_decode() {
        let value: [u8; 4] = [0xDE, 0xAD, 0xBE, 0xEF];
        let mut buf = Vec::new();
        value.nas_encode(&mut buf).unwrap();
        assert_eq!(buf, vec![0xDE, 0xAD, 0xBE, 0xEF]);

        let decoded = <[u8; 4]>::nas_decode(&mut buf.as_slice()).unwrap();
        assert_eq!(decoded, value);
    }

    #[test]
    fn test_plain_mm_header_encode_decode() {
        let header = PlainMmHeader::new(MmMessageType::RegistrationRequest);
        let mut buf = Vec::new();
        header.nas_encode(&mut buf).unwrap();

        assert_eq!(header.encoded_len(), PlainMmHeader::SIZE);
        assert_eq!(buf.len(), PlainMmHeader::SIZE);

        let decoded = PlainMmHeader::nas_decode(&mut buf.as_slice()).unwrap();
        assert_eq!(decoded, header);
    }

    #[test]
    fn test_plain_sm_header_encode_decode() {
        use crate::enums::SmMessageType;

        let header = PlainSmHeader::new(5, 1, SmMessageType::PduSessionEstablishmentRequest);
        let mut buf = Vec::new();
        header.nas_encode(&mut buf).unwrap();

        assert_eq!(header.encoded_len(), PlainSmHeader::SIZE);
        assert_eq!(buf.len(), PlainSmHeader::SIZE);

        let decoded = PlainSmHeader::nas_decode(&mut buf.as_slice()).unwrap();
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
        header.nas_encode(&mut buf).unwrap();

        assert_eq!(header.encoded_len(), SecuredHeader::SIZE);
        assert_eq!(buf.len(), SecuredHeader::SIZE);

        let decoded = SecuredHeader::nas_decode(&mut buf.as_slice()).unwrap();
        assert_eq!(decoded, header);
    }

    #[test]
    fn test_nas_header_encode_decode_plain_mm() {
        let header = NasHeader::PlainMm(PlainMmHeader::new(MmMessageType::AuthenticationRequest));
        let mut buf = Vec::new();
        header.nas_encode(&mut buf).unwrap();

        let decoded = NasHeader::nas_decode(&mut buf.as_slice()).unwrap();
        assert_eq!(decoded, header);
    }

    #[test]
    fn test_nas_header_encode_decode_secured() {
        let header = NasHeader::Secured(SecuredHeader::new(
            SecurityHeaderType::IntegrityProtected,
            [0xAA, 0xBB, 0xCC, 0xDD],
            0x10,
        ));
        let mut buf = Vec::new();
        header.nas_encode(&mut buf).unwrap();

        let decoded = NasHeader::nas_decode(&mut buf.as_slice()).unwrap();
        assert_eq!(decoded, header);
    }

    #[test]
    fn test_epd_encode_decode() {
        let epd = ExtendedProtocolDiscriminator::MobilityManagement;
        let mut buf = Vec::new();
        epd.nas_encode(&mut buf).unwrap();
        assert_eq!(buf, vec![0x7E]);

        let decoded = ExtendedProtocolDiscriminator::nas_decode(&mut buf.as_slice()).unwrap();
        assert_eq!(decoded, epd);
    }

    #[test]
    fn test_security_header_type_encode_decode() {
        let sht = SecurityHeaderType::IntegrityProtectedAndCiphered;
        let mut buf = Vec::new();
        sht.nas_encode(&mut buf).unwrap();
        assert_eq!(buf, vec![0x02]);

        let decoded = SecurityHeaderType::nas_decode(&mut buf.as_slice()).unwrap();
        assert_eq!(decoded, sht);
    }

    #[test]
    fn test_mm_message_type_encode_decode() {
        let mt = MmMessageType::SecurityModeCommand;
        let mut buf = Vec::new();
        mt.nas_encode(&mut buf).unwrap();
        assert_eq!(buf, vec![0x5D]);

        let decoded = MmMessageType::nas_decode(&mut buf.as_slice()).unwrap();
        assert_eq!(decoded, mt);
    }

    #[test]
    fn test_buffer_too_short_error() {
        let buf: &[u8] = &[];
        let result = u8::nas_decode(&mut &buf[..]);
        assert!(matches!(
            result,
            Err(CodecError::BufferTooShort {
                expected: 1,
                actual: 0
            })
        ));
    }

    #[test]
    fn test_invalid_epd_error() {
        let buf = [0xFF];
        let result = ExtendedProtocolDiscriminator::nas_decode(&mut buf.as_slice());
        assert!(matches!(
            result,
            Err(CodecError::InvalidProtocolDiscriminator(0xFF))
        ));
    }

    #[test]
    fn test_invalid_message_type_error() {
        let buf = [0x00]; // Invalid MM message type
        let result = MmMessageType::nas_decode(&mut buf.as_slice());
        assert!(matches!(result, Err(CodecError::InvalidMessageType(0x00))));
    }

    #[test]
    fn test_vec_encode() {
        let data = vec![0x01, 0x02, 0x03, 0x04];
        let mut buf = Vec::new();
        data.nas_encode(&mut buf).unwrap();
        assert_eq!(buf, data);
        assert_eq!(data.encoded_len(), 4);
    }
}
