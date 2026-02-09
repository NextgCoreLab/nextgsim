//! Deregistration Messages (3GPP TS 24.501 Section 8.2.11-8.2.14)
//!
//! This module implements the Deregistration procedure messages:
//! - Deregistration Request (UE originating) - UE to network
//! - Deregistration Accept (UE originating) - network to UE
//! - Deregistration Request (UE terminated) - network to UE
//! - Deregistration Accept (UE terminated) - UE to network

use bytes::{Buf, BufMut};
use thiserror::Error;

use crate::enums::MmMessageType;
use crate::header::PlainMmHeader;
use crate::ies::ie1::{IeDeRegistrationType, InformationElement1};
use crate::messages::mm::registration::{Ie5gMmCause, Ie5gsMobileIdentity, MmCause};
use crate::security::NasKeySetIdentifier;

/// Error type for Deregistration message encoding/decoding
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum DeregistrationError {
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
    /// Unknown IEI
    #[error("Unknown IEI: 0x{0:02X}")]
    UnknownIei(u8),
}

// ============================================================================
// IEI values for Deregistration messages
// ============================================================================

/// IEI values for Deregistration Request (UE terminated) optional IEs
#[allow(dead_code)]
mod deregistration_ue_terminated_iei {
    /// 5GMM cause
    pub const MM_CAUSE: u8 = 0x58;
    /// T3346 value (GPRS Timer 2)
    pub const T3346_VALUE: u8 = 0x5F;
}

// ============================================================================
// Deregistration Request (UE Originating) - 3GPP TS 24.501 Section 8.2.11
// ============================================================================

/// Deregistration Request message (UE originating - UE to network)
///
/// 3GPP TS 24.501 Section 8.2.11
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeregistrationRequestUeOriginating {
    /// De-registration type (mandatory, Type 1)
    pub deregistration_type: IeDeRegistrationType,
    /// ngKSI - NAS key set identifier (mandatory, Type 1)
    pub ng_ksi: NasKeySetIdentifier,
    /// 5GS mobile identity (mandatory, Type 6)
    pub mobile_identity: Ie5gsMobileIdentity,
}

impl Default for DeregistrationRequestUeOriginating {
    fn default() -> Self {
        Self {
            deregistration_type: IeDeRegistrationType::default(),
            ng_ksi: NasKeySetIdentifier::no_key(),
            mobile_identity: Ie5gsMobileIdentity::no_identity(),
        }
    }
}

impl DeregistrationRequestUeOriginating {
    /// Create a new Deregistration Request (UE originating) with mandatory fields
    pub fn new(
        deregistration_type: IeDeRegistrationType,
        ng_ksi: NasKeySetIdentifier,
        mobile_identity: Ie5gsMobileIdentity,
    ) -> Self {
        Self {
            deregistration_type,
            ng_ksi,
            mobile_identity,
        }
    }

    /// Decode from bytes (after header has been parsed)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, DeregistrationError> {
        if buf.remaining() < 1 {
            return Err(DeregistrationError::BufferTooShort {
                expected: 1,
                actual: buf.remaining(),
            });
        }

        // First octet: ngKSI (high nibble) + De-registration type (low nibble)
        let first_octet = buf.get_u8();
        let ng_ksi = NasKeySetIdentifier::decode((first_octet >> 4) & 0x0F)
            .map_err(|e| DeregistrationError::InvalidIeValue(e.to_string()))?;
        let deregistration_type = IeDeRegistrationType::decode(first_octet & 0x0F)
            .map_err(|e| DeregistrationError::InvalidIeValue(e.to_string()))?;

        // 5GS mobile identity (mandatory, Type 6)
        let mobile_identity = Ie5gsMobileIdentity::decode(buf)
            .map_err(|e| DeregistrationError::InvalidIeValue(e.to_string()))?;

        Ok(Self {
            deregistration_type,
            ng_ksi,
            mobile_identity,
        })
    }

    /// Encode to bytes (including header)
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        // Header
        let header = PlainMmHeader::new(MmMessageType::DeregistrationRequestUeOriginating);
        header.encode(buf);

        // First octet: ngKSI (high nibble) + De-registration type (low nibble)
        let first_octet =
            (self.ng_ksi.encode() << 4) | (self.deregistration_type.encode() & 0x0F);
        buf.put_u8(first_octet);

        // 5GS mobile identity (mandatory)
        self.mobile_identity.encode(buf);
    }

    /// Get the message type
    pub fn message_type() -> MmMessageType {
        MmMessageType::DeregistrationRequestUeOriginating
    }
}

// ============================================================================
// Deregistration Accept (UE Originating) - 3GPP TS 24.501 Section 8.2.12
// ============================================================================

/// Deregistration Accept message (UE originating - network to UE)
///
/// 3GPP TS 24.501 Section 8.2.12
///
/// This message has no information elements beyond the header.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DeregistrationAcceptUeOriginating;

impl DeregistrationAcceptUeOriginating {
    /// Create a new Deregistration Accept (UE originating)
    pub fn new() -> Self {
        Self
    }

    /// Decode from bytes (after header has been parsed)
    pub fn decode<B: Buf>(_buf: &mut B) -> Result<Self, DeregistrationError> {
        // No IEs to decode
        Ok(Self)
    }

    /// Encode to bytes (including header)
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        // Header only
        let header = PlainMmHeader::new(MmMessageType::DeregistrationAcceptUeOriginating);
        header.encode(buf);
    }

    /// Get the message type
    pub fn message_type() -> MmMessageType {
        MmMessageType::DeregistrationAcceptUeOriginating
    }
}

// ============================================================================
// Deregistration Request (UE Terminated) - 3GPP TS 24.501 Section 8.2.13
// ============================================================================

/// Deregistration Request message (UE terminated - network to UE)
///
/// 3GPP TS 24.501 Section 8.2.13
#[derive(Debug, Clone, PartialEq, Eq)]
#[derive(Default)]
pub struct DeregistrationRequestUeTerminated {
    /// De-registration type (mandatory, Type 1)
    pub deregistration_type: IeDeRegistrationType,
    /// 5GMM cause (optional, Type 3, IEI 0x58)
    pub mm_cause: Option<Ie5gMmCause>,
    /// T3346 value (optional, Type 4, IEI 0x5F)
    pub t3346_value: Option<u8>,
}


impl DeregistrationRequestUeTerminated {
    /// Create a new Deregistration Request (UE terminated) with mandatory fields
    pub fn new(deregistration_type: IeDeRegistrationType) -> Self {
        Self {
            deregistration_type,
            mm_cause: None,
            t3346_value: None,
        }
    }

    /// Decode from bytes (after header has been parsed)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, DeregistrationError> {
        if buf.remaining() < 1 {
            return Err(DeregistrationError::BufferTooShort {
                expected: 1,
                actual: buf.remaining(),
            });
        }

        // First octet: spare (high nibble) + De-registration type (low nibble)
        let first_octet = buf.get_u8();
        let deregistration_type = IeDeRegistrationType::decode(first_octet & 0x0F)
            .map_err(|e| DeregistrationError::InvalidIeValue(e.to_string()))?;

        let mut msg = Self::new(deregistration_type);

        // Parse optional IEs
        while buf.remaining() > 0 {
            let iei = buf.chunk()[0];

            match iei {
                deregistration_ue_terminated_iei::MM_CAUSE => {
                    buf.advance(1);
                    if buf.remaining() < 1 {
                        break;
                    }
                    let cause = MmCause::try_from(buf.get_u8())
                        .map_err(|e| DeregistrationError::InvalidIeValue(e.to_string()))?;
                    msg.mm_cause = Some(Ie5gMmCause::new(cause));
                }
                deregistration_ue_terminated_iei::T3346_VALUE => {
                    buf.advance(1);
                    if buf.remaining() < 1 {
                        break;
                    }
                    let len = buf.get_u8() as usize;
                    if buf.remaining() < len || len < 1 {
                        break;
                    }
                    msg.t3346_value = Some(buf.get_u8());
                    if len > 1 {
                        buf.advance(len - 1);
                    }
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
        let header = PlainMmHeader::new(MmMessageType::DeregistrationRequestUeTerminated);
        header.encode(buf);

        // First octet: spare (high nibble) + De-registration type (low nibble)
        let first_octet = self.deregistration_type.encode() & 0x0F;
        buf.put_u8(first_octet);

        // Optional IEs
        if let Some(ref cause) = self.mm_cause {
            buf.put_u8(deregistration_ue_terminated_iei::MM_CAUSE);
            cause.encode(buf);
        }

        if let Some(t3346) = self.t3346_value {
            buf.put_u8(deregistration_ue_terminated_iei::T3346_VALUE);
            buf.put_u8(1); // Length
            buf.put_u8(t3346);
        }
    }

    /// Get the message type
    pub fn message_type() -> MmMessageType {
        MmMessageType::DeregistrationRequestUeTerminated
    }
}

// ============================================================================
// Deregistration Accept (UE Terminated) - 3GPP TS 24.501 Section 8.2.14
// ============================================================================

/// Deregistration Accept message (UE terminated - UE to network)
///
/// 3GPP TS 24.501 Section 8.2.14
///
/// This message has no information elements beyond the header.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DeregistrationAcceptUeTerminated;

impl DeregistrationAcceptUeTerminated {
    /// Create a new Deregistration Accept (UE terminated)
    pub fn new() -> Self {
        Self
    }

    /// Decode from bytes (after header has been parsed)
    pub fn decode<B: Buf>(_buf: &mut B) -> Result<Self, DeregistrationError> {
        // No IEs to decode
        Ok(Self)
    }

    /// Encode to bytes (including header)
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        // Header only
        let header = PlainMmHeader::new(MmMessageType::DeregistrationAcceptUeTerminated);
        header.encode(buf);
    }

    /// Get the message type
    pub fn message_type() -> MmMessageType {
        MmMessageType::DeregistrationAcceptUeTerminated
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ies::ie1::{DeRegistrationAccessType, ReRegistrationRequired, SwitchOff};

    #[test]
    fn test_deregistration_request_ue_originating_encode_decode() {
        let dereg_type = IeDeRegistrationType::new(
            DeRegistrationAccessType::ThreeGppAccess,
            ReRegistrationRequired::NotRequired,
            SwitchOff::NormalDeRegistration,
        );
        let ng_ksi = NasKeySetIdentifier::no_key();
        let mobile_identity = Ie5gsMobileIdentity::no_identity();

        let msg =
            DeregistrationRequestUeOriginating::new(dereg_type, ng_ksi, mobile_identity.clone());

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Skip header (3 bytes) for decoding
        let decoded =
            DeregistrationRequestUeOriginating::decode(&mut &buf[3..]).unwrap();

        assert_eq!(decoded.deregistration_type, dereg_type);
        assert_eq!(decoded.ng_ksi, ng_ksi);
        assert_eq!(decoded.mobile_identity, mobile_identity);
    }

    #[test]
    fn test_deregistration_accept_ue_originating_encode_decode() {
        let msg = DeregistrationAcceptUeOriginating::new();

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Should be header only (3 bytes)
        assert_eq!(buf.len(), 3);
        assert_eq!(buf[0], 0x7E); // EPD
        assert_eq!(buf[1], 0x00); // Security header type
        assert_eq!(buf[2], 0x46); // Message type

        // Skip header for decoding
        let decoded = DeregistrationAcceptUeOriginating::decode(&mut &buf[3..]).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn test_deregistration_request_ue_terminated_encode_decode() {
        let dereg_type = IeDeRegistrationType::new(
            DeRegistrationAccessType::ThreeGppAccess,
            ReRegistrationRequired::Required,
            SwitchOff::NormalDeRegistration,
        );

        let mut msg = DeregistrationRequestUeTerminated::new(dereg_type);
        msg.mm_cause = Some(Ie5gMmCause::new(MmCause::ImplicitlyDeregistered));
        msg.t3346_value = Some(0x21); // Example timer value

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Skip header (3 bytes) for decoding
        let decoded =
            DeregistrationRequestUeTerminated::decode(&mut &buf[3..]).unwrap();

        assert_eq!(decoded.deregistration_type, dereg_type);
        assert_eq!(
            decoded.mm_cause,
            Some(Ie5gMmCause::new(MmCause::ImplicitlyDeregistered))
        );
        assert_eq!(decoded.t3346_value, Some(0x21));
    }

    #[test]
    fn test_deregistration_request_ue_terminated_minimal() {
        let dereg_type = IeDeRegistrationType::new(
            DeRegistrationAccessType::NonThreeGppAccess,
            ReRegistrationRequired::NotRequired,
            SwitchOff::SwitchOff,
        );

        let msg = DeregistrationRequestUeTerminated::new(dereg_type);

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Skip header (3 bytes) for decoding
        let decoded =
            DeregistrationRequestUeTerminated::decode(&mut &buf[3..]).unwrap();

        assert_eq!(decoded.deregistration_type, dereg_type);
        assert_eq!(decoded.mm_cause, None);
        assert_eq!(decoded.t3346_value, None);
    }

    #[test]
    fn test_deregistration_accept_ue_terminated_encode_decode() {
        let msg = DeregistrationAcceptUeTerminated::new();

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Should be header only (3 bytes)
        assert_eq!(buf.len(), 3);
        assert_eq!(buf[0], 0x7E); // EPD
        assert_eq!(buf[1], 0x00); // Security header type
        assert_eq!(buf[2], 0x48); // Message type

        // Skip header for decoding
        let decoded = DeregistrationAcceptUeTerminated::decode(&mut &buf[3..]).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn test_message_types() {
        assert_eq!(
            DeregistrationRequestUeOriginating::message_type(),
            MmMessageType::DeregistrationRequestUeOriginating
        );
        assert_eq!(
            DeregistrationAcceptUeOriginating::message_type(),
            MmMessageType::DeregistrationAcceptUeOriginating
        );
        assert_eq!(
            DeregistrationRequestUeTerminated::message_type(),
            MmMessageType::DeregistrationRequestUeTerminated
        );
        assert_eq!(
            DeregistrationAcceptUeTerminated::message_type(),
            MmMessageType::DeregistrationAcceptUeTerminated
        );
    }
}
