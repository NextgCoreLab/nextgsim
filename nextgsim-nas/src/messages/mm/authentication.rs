//! Authentication Messages (3GPP TS 24.501 Section 8.2.1-8.2.5)
//!
//! This module implements the Authentication procedure messages:
//! - Authentication Request (network to UE)
//! - Authentication Response (UE to network)
//! - Authentication Reject (network to UE)
//! - Authentication Failure (UE to network)
//! - Authentication Result (network to UE)

use bytes::{Buf, BufMut};
use thiserror::Error;

use crate::enums::MmMessageType;
use crate::header::PlainMmHeader;
use crate::security::NasKeySetIdentifier;

use super::registration::{Ie5gMmCause, MmCause};

/// Error type for Authentication message encoding/decoding
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum AuthenticationError {
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
    /// Header decoding error
    #[error("Header error: {0}")]
    HeaderError(#[from] crate::header::HeaderError),
}


// ============================================================================
// Authentication Parameter RAND (3GPP TS 24.501 Section 9.11.3.16)
// ============================================================================

/// Authentication Parameter RAND IE (Type 3 - 16 bytes)
///
/// Contains the random challenge used in authentication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub struct AuthenticationParameterRand {
    /// 128-bit random value
    pub value: [u8; 16],
}

impl AuthenticationParameterRand {
    /// Create a new RAND parameter
    pub fn new(value: [u8; 16]) -> Self {
        Self { value }
    }

    /// Decode from bytes
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, AuthenticationError> {
        if buf.remaining() < 16 {
            return Err(AuthenticationError::BufferTooShort {
                expected: 16,
                actual: buf.remaining(),
            });
        }
        let mut value = [0u8; 16];
        buf.copy_to_slice(&mut value);
        Ok(Self { value })
    }

    /// Encode to bytes
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        buf.put_slice(&self.value);
    }
}


// ============================================================================
// Authentication Parameter AUTN (3GPP TS 24.501 Section 9.11.3.15)
// ============================================================================

/// Authentication Parameter AUTN IE (Type 4 - variable length, typically 16 bytes)
///
/// Contains the authentication token.
#[derive(Debug, Clone, PartialEq, Eq)]
#[derive(Default)]
pub struct AuthenticationParameterAutn {
    /// AUTN value (typically 16 bytes)
    pub value: Vec<u8>,
}

impl AuthenticationParameterAutn {
    /// Create a new AUTN parameter
    pub fn new(value: Vec<u8>) -> Self {
        Self { value }
    }

    /// Decode from bytes (with length prefix)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, AuthenticationError> {
        if buf.remaining() < 1 {
            return Err(AuthenticationError::BufferTooShort {
                expected: 1,
                actual: buf.remaining(),
            });
        }
        let length = buf.get_u8() as usize;
        if buf.remaining() < length {
            return Err(AuthenticationError::BufferTooShort {
                expected: length,
                actual: buf.remaining(),
            });
        }
        let mut value = vec![0u8; length];
        buf.copy_to_slice(&mut value);
        Ok(Self { value })
    }

    /// Encode to bytes (with length prefix)
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        buf.put_u8(self.value.len() as u8);
        buf.put_slice(&self.value);
    }

    /// Get encoded length (including length field)
    pub fn encoded_len(&self) -> usize {
        1 + self.value.len()
    }
}


// ============================================================================
// Authentication Response Parameter (3GPP TS 24.501 Section 9.11.3.17)
// ============================================================================

/// Authentication Response Parameter IE (Type 4 - variable length)
///
/// Contains the RES* value computed by the UE.
#[derive(Debug, Clone, PartialEq, Eq)]
#[derive(Default)]
pub struct AuthenticationResponseParameter {
    /// RES* value
    pub value: Vec<u8>,
}

impl AuthenticationResponseParameter {
    /// Create a new Authentication Response Parameter
    pub fn new(value: Vec<u8>) -> Self {
        Self { value }
    }

    /// Decode from bytes (with length prefix)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, AuthenticationError> {
        if buf.remaining() < 1 {
            return Err(AuthenticationError::BufferTooShort {
                expected: 1,
                actual: buf.remaining(),
            });
        }
        let length = buf.get_u8() as usize;
        if buf.remaining() < length {
            return Err(AuthenticationError::BufferTooShort {
                expected: length,
                actual: buf.remaining(),
            });
        }
        let mut value = vec![0u8; length];
        buf.copy_to_slice(&mut value);
        Ok(Self { value })
    }

    /// Encode to bytes (with length prefix)
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        buf.put_u8(self.value.len() as u8);
        buf.put_slice(&self.value);
    }

    /// Get encoded length (including length field)
    pub fn encoded_len(&self) -> usize {
        1 + self.value.len()
    }
}


// ============================================================================
// Authentication Failure Parameter (3GPP TS 24.501 Section 9.11.3.14)
// ============================================================================

/// Authentication Failure Parameter IE (Type 4 - 14 bytes)
///
/// Contains the AUTS value for synchronization failure.
#[derive(Debug, Clone, PartialEq, Eq)]
#[derive(Default)]
pub struct AuthenticationFailureParameter {
    /// AUTS value (14 bytes)
    pub value: Vec<u8>,
}

impl AuthenticationFailureParameter {
    /// Create a new Authentication Failure Parameter
    pub fn new(value: Vec<u8>) -> Self {
        Self { value }
    }

    /// Decode from bytes (with length prefix)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, AuthenticationError> {
        if buf.remaining() < 1 {
            return Err(AuthenticationError::BufferTooShort {
                expected: 1,
                actual: buf.remaining(),
            });
        }
        let length = buf.get_u8() as usize;
        if buf.remaining() < length {
            return Err(AuthenticationError::BufferTooShort {
                expected: length,
                actual: buf.remaining(),
            });
        }
        let mut value = vec![0u8; length];
        buf.copy_to_slice(&mut value);
        Ok(Self { value })
    }

    /// Encode to bytes (with length prefix)
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        buf.put_u8(self.value.len() as u8);
        buf.put_slice(&self.value);
    }

    /// Get encoded length (including length field)
    pub fn encoded_len(&self) -> usize {
        1 + self.value.len()
    }
}


// ============================================================================
// EAP Message IE (3GPP TS 24.501 Section 9.11.2.2)
// ============================================================================

/// EAP Message IE (Type 6 - variable length with 2-byte LI)
#[derive(Debug, Clone, PartialEq, Eq)]
#[derive(Default)]
pub struct EapMessage {
    /// EAP message data
    pub data: Vec<u8>,
}

impl EapMessage {
    /// Create a new EAP Message
    pub fn new(data: Vec<u8>) -> Self {
        Self { data }
    }

    /// Decode from bytes (with 2-byte length prefix)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, AuthenticationError> {
        if buf.remaining() < 2 {
            return Err(AuthenticationError::BufferTooShort {
                expected: 2,
                actual: buf.remaining(),
            });
        }
        let length = buf.get_u16() as usize;
        if buf.remaining() < length {
            return Err(AuthenticationError::BufferTooShort {
                expected: length,
                actual: buf.remaining(),
            });
        }
        let mut data = vec![0u8; length];
        buf.copy_to_slice(&mut data);
        Ok(Self { data })
    }

    /// Encode to bytes (with 2-byte length prefix)
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        buf.put_u16(self.data.len() as u16);
        buf.put_slice(&self.data);
    }

    /// Get encoded length (including 2-byte length field)
    pub fn encoded_len(&self) -> usize {
        2 + self.data.len()
    }
}


// ============================================================================
// ABBA IE (3GPP TS 24.501 Section 9.11.3.10)
// ============================================================================

/// ABBA (Anti-Bidding down Between Architectures) IE (Type 4)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Abba {
    /// ABBA value
    pub value: Vec<u8>,
}

impl Abba {
    /// Create a new ABBA IE
    pub fn new(value: Vec<u8>) -> Self {
        Self { value }
    }

    /// Decode from bytes (with length prefix)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, AuthenticationError> {
        if buf.remaining() < 1 {
            return Err(AuthenticationError::BufferTooShort {
                expected: 1,
                actual: buf.remaining(),
            });
        }
        let length = buf.get_u8() as usize;
        if buf.remaining() < length {
            return Err(AuthenticationError::BufferTooShort {
                expected: length,
                actual: buf.remaining(),
            });
        }
        let mut value = vec![0u8; length];
        buf.copy_to_slice(&mut value);
        Ok(Self { value })
    }

    /// Encode to bytes (with length prefix)
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        buf.put_u8(self.value.len() as u8);
        buf.put_slice(&self.value);
    }

    /// Get encoded length (including length field)
    pub fn encoded_len(&self) -> usize {
        1 + self.value.len()
    }
}

impl Default for Abba {
    fn default() -> Self {
        Self { value: vec![0x00, 0x00] } // Default ABBA value
    }
}

// ============================================================================
// IEI Constants for Authentication Messages
// ============================================================================

/// IEI values for Authentication Request optional IEs
mod authentication_request_iei {
    /// EAP message
    pub const EAP_MESSAGE: u8 = 0x78;
}

/// IEI values for Authentication Response optional IEs
mod authentication_response_iei {
    /// Authentication response parameter
    pub const AUTH_RESPONSE_PARAMETER: u8 = 0x2D;
    /// EAP message
    pub const EAP_MESSAGE: u8 = 0x78;
}

/// IEI values for Authentication Reject optional IEs
mod authentication_reject_iei {
    /// EAP message
    pub const EAP_MESSAGE: u8 = 0x78;
}

/// IEI values for Authentication Failure optional IEs
mod authentication_failure_iei {
    /// Authentication failure parameter
    pub const AUTH_FAILURE_PARAMETER: u8 = 0x30;
}

/// IEI values for Authentication Result optional IEs
mod authentication_result_iei {
    /// ABBA
    pub const ABBA: u8 = 0x38;
}

// ============================================================================
// Authentication Request Message (3GPP TS 24.501 Section 8.2.1)
// ============================================================================

/// Authentication Request message (network to UE)
///
/// 3GPP TS 24.501 Section 8.2.1
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthenticationRequest {
    /// ngKSI - NAS key set identifier (mandatory, Type 1)
    pub ng_ksi: NasKeySetIdentifier,
    /// ABBA (mandatory, Type 4)
    pub abba: Abba,
    /// Authentication parameter RAND (optional, Type 3, IEI 0x21)
    pub rand: Option<AuthenticationParameterRand>,
    /// Authentication parameter AUTN (optional, Type 4, IEI 0x20)
    pub autn: Option<AuthenticationParameterAutn>,
    /// EAP message (optional, Type 6, IEI 0x78)
    pub eap_message: Option<EapMessage>,
}

impl Default for AuthenticationRequest {
    fn default() -> Self {
        Self {
            ng_ksi: NasKeySetIdentifier::no_key(),
            abba: Abba::default(),
            rand: None,
            autn: None,
            eap_message: None,
        }
    }
}

impl AuthenticationRequest {
    /// Create a new Authentication Request with mandatory fields
    pub fn new(ng_ksi: NasKeySetIdentifier, abba: Abba) -> Self {
        Self {
            ng_ksi,
            abba,
            ..Default::default()
        }
    }

    /// Create an Authentication Request for 5G-AKA
    pub fn for_5g_aka(
        ng_ksi: NasKeySetIdentifier,
        abba: Abba,
        rand: [u8; 16],
        autn: Vec<u8>,
    ) -> Self {
        Self {
            ng_ksi,
            abba,
            rand: Some(AuthenticationParameterRand::new(rand)),
            autn: Some(AuthenticationParameterAutn::new(autn)),
            eap_message: None,
        }
    }

    /// Create an Authentication Request for EAP-AKA'
    pub fn for_eap_aka(ng_ksi: NasKeySetIdentifier, abba: Abba, eap_message: Vec<u8>) -> Self {
        Self {
            ng_ksi,
            abba,
            rand: None,
            autn: None,
            eap_message: Some(EapMessage::new(eap_message)),
        }
    }

    /// Decode from bytes (after header has been parsed)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, AuthenticationError> {
        if buf.remaining() < 1 {
            return Err(AuthenticationError::BufferTooShort {
                expected: 1,
                actual: buf.remaining(),
            });
        }

        // First octet: spare (high nibble) + ngKSI (low nibble)
        let first_octet = buf.get_u8();
        let ng_ksi = NasKeySetIdentifier::decode(first_octet & 0x0F)
            .map_err(|e| AuthenticationError::InvalidIeValue(e.to_string()))?;

        // ABBA (mandatory, Type 4)
        let abba = Abba::decode(buf)?;

        let mut msg = Self::new(ng_ksi, abba);

        // Parse optional IEs
        while buf.remaining() > 0 {
            let iei = buf.chunk()[0];

            match iei {
                0x21 => {
                    // Authentication parameter RAND (Type 3, 16 bytes)
                    buf.advance(1);
                    msg.rand = Some(AuthenticationParameterRand::decode(buf)?);
                }
                0x20 => {
                    // Authentication parameter AUTN (Type 4)
                    buf.advance(1);
                    msg.autn = Some(AuthenticationParameterAutn::decode(buf)?);
                }
                authentication_request_iei::EAP_MESSAGE => {
                    // EAP message (Type 6)
                    buf.advance(1);
                    msg.eap_message = Some(EapMessage::decode(buf)?);
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
        let header = PlainMmHeader::new(MmMessageType::AuthenticationRequest);
        header.encode(buf);

        // First octet: spare (high nibble) + ngKSI (low nibble)
        buf.put_u8(self.ng_ksi.encode() & 0x0F);

        // ABBA (mandatory)
        self.abba.encode(buf);

        // Optional IEs
        if let Some(ref rand) = self.rand {
            buf.put_u8(0x21);
            rand.encode(buf);
        }

        if let Some(ref autn) = self.autn {
            buf.put_u8(0x20);
            autn.encode(buf);
        }

        if let Some(ref eap) = self.eap_message {
            buf.put_u8(authentication_request_iei::EAP_MESSAGE);
            eap.encode(buf);
        }
    }

    /// Get the message type
    pub fn message_type() -> MmMessageType {
        MmMessageType::AuthenticationRequest
    }
}

// ============================================================================
// Authentication Response Message (3GPP TS 24.501 Section 8.2.2)
// ============================================================================

/// Authentication Response message (UE to network)
///
/// 3GPP TS 24.501 Section 8.2.2
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AuthenticationResponse {
    /// Authentication response parameter (optional, Type 4, IEI 0x2D)
    pub auth_response_parameter: Option<AuthenticationResponseParameter>,
    /// EAP message (optional, Type 6, IEI 0x78)
    pub eap_message: Option<EapMessage>,
}

impl AuthenticationResponse {
    /// Create a new empty Authentication Response
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an Authentication Response for 5G-AKA with RES*
    pub fn with_res_star(res_star: Vec<u8>) -> Self {
        Self {
            auth_response_parameter: Some(AuthenticationResponseParameter::new(res_star)),
            eap_message: None,
        }
    }

    /// Create an Authentication Response for EAP-AKA'
    pub fn with_eap_message(eap_message: Vec<u8>) -> Self {
        Self {
            auth_response_parameter: None,
            eap_message: Some(EapMessage::new(eap_message)),
        }
    }

    /// Decode from bytes (after header has been parsed)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, AuthenticationError> {
        let mut msg = Self::new();

        // Parse optional IEs
        while buf.remaining() > 0 {
            let iei = buf.chunk()[0];

            match iei {
                authentication_response_iei::AUTH_RESPONSE_PARAMETER => {
                    buf.advance(1);
                    msg.auth_response_parameter = Some(AuthenticationResponseParameter::decode(buf)?);
                }
                authentication_response_iei::EAP_MESSAGE => {
                    buf.advance(1);
                    msg.eap_message = Some(EapMessage::decode(buf)?);
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
        let header = PlainMmHeader::new(MmMessageType::AuthenticationResponse);
        header.encode(buf);

        // Optional IEs
        if let Some(ref param) = self.auth_response_parameter {
            buf.put_u8(authentication_response_iei::AUTH_RESPONSE_PARAMETER);
            param.encode(buf);
        }

        if let Some(ref eap) = self.eap_message {
            buf.put_u8(authentication_response_iei::EAP_MESSAGE);
            eap.encode(buf);
        }
    }

    /// Get the message type
    pub fn message_type() -> MmMessageType {
        MmMessageType::AuthenticationResponse
    }
}

// ============================================================================
// Authentication Reject Message (3GPP TS 24.501 Section 8.2.3)
// ============================================================================

/// Authentication Reject message (network to UE)
///
/// 3GPP TS 24.501 Section 8.2.3
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AuthenticationReject {
    /// EAP message (optional, Type 6, IEI 0x78)
    pub eap_message: Option<EapMessage>,
}

impl AuthenticationReject {
    /// Create a new empty Authentication Reject
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an Authentication Reject with EAP message
    pub fn with_eap_message(eap_message: Vec<u8>) -> Self {
        Self {
            eap_message: Some(EapMessage::new(eap_message)),
        }
    }

    /// Decode from bytes (after header has been parsed)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, AuthenticationError> {
        let mut msg = Self::new();

        // Parse optional IEs
        while buf.remaining() > 0 {
            let iei = buf.chunk()[0];

            match iei {
                authentication_reject_iei::EAP_MESSAGE => {
                    buf.advance(1);
                    msg.eap_message = Some(EapMessage::decode(buf)?);
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
        let header = PlainMmHeader::new(MmMessageType::AuthenticationReject);
        header.encode(buf);

        // Optional IEs
        if let Some(ref eap) = self.eap_message {
            buf.put_u8(authentication_reject_iei::EAP_MESSAGE);
            eap.encode(buf);
        }
    }

    /// Get the message type
    pub fn message_type() -> MmMessageType {
        MmMessageType::AuthenticationReject
    }
}

// ============================================================================
// Authentication Failure Message (3GPP TS 24.501 Section 8.2.4)
// ============================================================================

/// Authentication Failure message (UE to network)
///
/// 3GPP TS 24.501 Section 8.2.4
#[derive(Debug, Clone, PartialEq, Eq)]
#[derive(Default)]
pub struct AuthenticationFailure {
    /// 5GMM cause (mandatory, Type 3)
    pub mm_cause: Ie5gMmCause,
    /// Authentication failure parameter (optional, Type 4, IEI 0x30)
    pub auth_failure_parameter: Option<AuthenticationFailureParameter>,
}


impl AuthenticationFailure {
    /// Create a new Authentication Failure with mandatory fields
    pub fn new(mm_cause: Ie5gMmCause) -> Self {
        Self {
            mm_cause,
            auth_failure_parameter: None,
        }
    }

    /// Create an Authentication Failure with a cause value
    pub fn with_cause(cause: MmCause) -> Self {
        Self::new(Ie5gMmCause::new(cause))
    }

    /// Create an Authentication Failure for MAC failure
    pub fn mac_failure() -> Self {
        Self::with_cause(MmCause::MacFailure)
    }

    /// Create an Authentication Failure for synch failure with AUTS
    pub fn synch_failure(auts: Vec<u8>) -> Self {
        Self {
            mm_cause: Ie5gMmCause::new(MmCause::SynchFailure),
            auth_failure_parameter: Some(AuthenticationFailureParameter::new(auts)),
        }
    }

    /// Decode from bytes (after header has been parsed)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, AuthenticationError> {
        // 5GMM cause (mandatory, Type 3)
        let mm_cause = Ie5gMmCause::decode(buf).map_err(|e| {
            AuthenticationError::InvalidIeValue(format!("Failed to decode 5GMM cause: {e}"))
        })?;

        let mut msg = Self::new(mm_cause);

        // Parse optional IEs
        while buf.remaining() > 0 {
            let iei = buf.chunk()[0];

            match iei {
                authentication_failure_iei::AUTH_FAILURE_PARAMETER => {
                    buf.advance(1);
                    msg.auth_failure_parameter = Some(AuthenticationFailureParameter::decode(buf)?);
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
        let header = PlainMmHeader::new(MmMessageType::AuthenticationFailure);
        header.encode(buf);

        // 5GMM cause (mandatory)
        self.mm_cause.encode(buf);

        // Optional IEs
        if let Some(ref param) = self.auth_failure_parameter {
            buf.put_u8(authentication_failure_iei::AUTH_FAILURE_PARAMETER);
            param.encode(buf);
        }
    }

    /// Get the message type
    pub fn message_type() -> MmMessageType {
        MmMessageType::AuthenticationFailure
    }
}

// ============================================================================
// Authentication Result Message (3GPP TS 24.501 Section 8.2.5)
// ============================================================================

/// Authentication Result message (network to UE)
///
/// 3GPP TS 24.501 Section 8.2.5
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthenticationResult {
    /// ngKSI - NAS key set identifier (mandatory, Type 1)
    pub ng_ksi: NasKeySetIdentifier,
    /// EAP message (mandatory, Type 6)
    pub eap_message: EapMessage,
    /// ABBA (optional, Type 4, IEI 0x38)
    pub abba: Option<Abba>,
}

impl Default for AuthenticationResult {
    fn default() -> Self {
        Self {
            ng_ksi: NasKeySetIdentifier::no_key(),
            eap_message: EapMessage::default(),
            abba: None,
        }
    }
}

impl AuthenticationResult {
    /// Create a new Authentication Result with mandatory fields
    pub fn new(ng_ksi: NasKeySetIdentifier, eap_message: EapMessage) -> Self {
        Self {
            ng_ksi,
            eap_message,
            abba: None,
        }
    }

    /// Create an Authentication Result with EAP message data
    pub fn with_eap_data(ng_ksi: NasKeySetIdentifier, eap_data: Vec<u8>) -> Self {
        Self::new(ng_ksi, EapMessage::new(eap_data))
    }

    /// Decode from bytes (after header has been parsed)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, AuthenticationError> {
        if buf.remaining() < 1 {
            return Err(AuthenticationError::BufferTooShort {
                expected: 1,
                actual: buf.remaining(),
            });
        }

        // First octet: spare (high nibble) + ngKSI (low nibble)
        let first_octet = buf.get_u8();
        let ng_ksi = NasKeySetIdentifier::decode(first_octet & 0x0F)
            .map_err(|e| AuthenticationError::InvalidIeValue(e.to_string()))?;

        // EAP message (mandatory, Type 6)
        let eap_message = EapMessage::decode(buf)?;

        let mut msg = Self::new(ng_ksi, eap_message);

        // Parse optional IEs
        while buf.remaining() > 0 {
            let iei = buf.chunk()[0];

            match iei {
                authentication_result_iei::ABBA => {
                    buf.advance(1);
                    msg.abba = Some(Abba::decode(buf)?);
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
        let header = PlainMmHeader::new(MmMessageType::AuthenticationResult);
        header.encode(buf);

        // First octet: spare (high nibble) + ngKSI (low nibble)
        buf.put_u8(self.ng_ksi.encode() & 0x0F);

        // EAP message (mandatory)
        self.eap_message.encode(buf);

        // Optional IEs
        if let Some(ref abba) = self.abba {
            buf.put_u8(authentication_result_iei::ABBA);
            abba.encode(buf);
        }
    }

    /// Get the message type
    pub fn message_type() -> MmMessageType {
        MmMessageType::AuthenticationResult
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::security::SecurityContextType;

    #[test]
    fn test_authentication_parameter_rand() {
        let rand = AuthenticationParameterRand::new([0x01; 16]);
        let mut buf = Vec::new();
        rand.encode(&mut buf);
        assert_eq!(buf.len(), 16);

        let decoded = AuthenticationParameterRand::decode(&mut buf.as_slice()).unwrap();
        assert_eq!(decoded.value, [0x01; 16]);
    }

    #[test]
    fn test_authentication_parameter_autn() {
        let autn = AuthenticationParameterAutn::new(vec![0x02; 16]);
        let mut buf = Vec::new();
        autn.encode(&mut buf);
        assert_eq!(buf.len(), 17); // 1 byte length + 16 bytes data

        let decoded = AuthenticationParameterAutn::decode(&mut buf.as_slice()).unwrap();
        assert_eq!(decoded.value, vec![0x02; 16]);
    }

    #[test]
    fn test_authentication_response_parameter() {
        let param = AuthenticationResponseParameter::new(vec![0x03; 16]);
        let mut buf = Vec::new();
        param.encode(&mut buf);

        let decoded = AuthenticationResponseParameter::decode(&mut buf.as_slice()).unwrap();
        assert_eq!(decoded.value, vec![0x03; 16]);
    }

    #[test]
    fn test_authentication_failure_parameter() {
        let param = AuthenticationFailureParameter::new(vec![0x04; 14]);
        let mut buf = Vec::new();
        param.encode(&mut buf);

        let decoded = AuthenticationFailureParameter::decode(&mut buf.as_slice()).unwrap();
        assert_eq!(decoded.value, vec![0x04; 14]);
    }

    #[test]
    fn test_eap_message() {
        let eap = EapMessage::new(vec![0x01, 0x02, 0x03, 0x04]);
        let mut buf = Vec::new();
        eap.encode(&mut buf);
        assert_eq!(buf.len(), 6); // 2 bytes length + 4 bytes data

        let decoded = EapMessage::decode(&mut buf.as_slice()).unwrap();
        assert_eq!(decoded.data, vec![0x01, 0x02, 0x03, 0x04]);
    }

    #[test]
    fn test_abba() {
        let abba = Abba::new(vec![0x00, 0x00]);
        let mut buf = Vec::new();
        abba.encode(&mut buf);

        let decoded = Abba::decode(&mut buf.as_slice()).unwrap();
        assert_eq!(decoded.value, vec![0x00, 0x00]);
    }

    #[test]
    fn test_authentication_request_basic() {
        let ng_ksi = NasKeySetIdentifier::new(SecurityContextType::Native, 1);
        let abba = Abba::default();
        let msg = AuthenticationRequest::new(ng_ksi, abba);

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Verify header
        assert_eq!(buf[0], 0x7E); // EPD
        assert_eq!(buf[1], 0x00); // Security header type
        assert_eq!(buf[2], 0x56); // Message type (Authentication Request)
        assert_eq!(buf[3] & 0x0F, 0x01); // ngKSI = 1
    }

    #[test]
    fn test_authentication_request_5g_aka() {
        let ng_ksi = NasKeySetIdentifier::new(SecurityContextType::Native, 2);
        let abba = Abba::new(vec![0x00, 0x00]);
        let rand = [0xAA; 16];
        let autn = vec![0xBB; 16];
        let msg = AuthenticationRequest::for_5g_aka(ng_ksi, abba, rand, autn.clone());

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Skip header (3 bytes) and decode
        let decoded = AuthenticationRequest::decode(&mut buf[3..].as_ref()).unwrap();

        assert_eq!(decoded.ng_ksi.ksi, 2);
        assert!(decoded.rand.is_some());
        assert_eq!(decoded.rand.unwrap().value, [0xAA; 16]);
        assert!(decoded.autn.is_some());
        assert_eq!(decoded.autn.unwrap().value, autn);
    }

    #[test]
    fn test_authentication_request_eap_aka() {
        let ng_ksi = NasKeySetIdentifier::new(SecurityContextType::Native, 3);
        let abba = Abba::default();
        let eap_data = vec![0x01, 0x02, 0x03, 0x04, 0x05];
        let msg = AuthenticationRequest::for_eap_aka(ng_ksi, abba, eap_data.clone());

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Skip header (3 bytes) and decode
        let decoded = AuthenticationRequest::decode(&mut buf[3..].as_ref()).unwrap();

        assert_eq!(decoded.ng_ksi.ksi, 3);
        assert!(decoded.rand.is_none());
        assert!(decoded.autn.is_none());
        assert!(decoded.eap_message.is_some());
        assert_eq!(decoded.eap_message.unwrap().data, eap_data);
    }

    #[test]
    fn test_authentication_response_with_res_star() {
        let res_star = vec![0x11; 16];
        let msg = AuthenticationResponse::with_res_star(res_star.clone());

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Verify header
        assert_eq!(buf[0], 0x7E); // EPD
        assert_eq!(buf[1], 0x00); // Security header type
        assert_eq!(buf[2], 0x57); // Message type (Authentication Response)

        // Skip header (3 bytes) and decode
        let decoded = AuthenticationResponse::decode(&mut buf[3..].as_ref()).unwrap();

        assert!(decoded.auth_response_parameter.is_some());
        assert_eq!(decoded.auth_response_parameter.unwrap().value, res_star);
    }

    #[test]
    fn test_authentication_response_with_eap() {
        let eap_data = vec![0x22; 8];
        let msg = AuthenticationResponse::with_eap_message(eap_data.clone());

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Skip header (3 bytes) and decode
        let decoded = AuthenticationResponse::decode(&mut buf[3..].as_ref()).unwrap();

        assert!(decoded.eap_message.is_some());
        assert_eq!(decoded.eap_message.unwrap().data, eap_data);
    }

    #[test]
    fn test_authentication_reject_basic() {
        let msg = AuthenticationReject::new();

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Verify header
        assert_eq!(buf[0], 0x7E); // EPD
        assert_eq!(buf[1], 0x00); // Security header type
        assert_eq!(buf[2], 0x58); // Message type (Authentication Reject)
        assert_eq!(buf.len(), 3); // No optional IEs
    }

    #[test]
    fn test_authentication_reject_with_eap() {
        let eap_data = vec![0x33; 4];
        let msg = AuthenticationReject::with_eap_message(eap_data.clone());

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Skip header (3 bytes) and decode
        let decoded = AuthenticationReject::decode(&mut buf[3..].as_ref()).unwrap();

        assert!(decoded.eap_message.is_some());
        assert_eq!(decoded.eap_message.unwrap().data, eap_data);
    }

    #[test]
    fn test_authentication_failure_mac_failure() {
        let msg = AuthenticationFailure::mac_failure();

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Verify header
        assert_eq!(buf[0], 0x7E); // EPD
        assert_eq!(buf[1], 0x00); // Security header type
        assert_eq!(buf[2], 0x59); // Message type (Authentication Failure)
        assert_eq!(buf[3], 20); // 5GMM cause (MAC failure)

        // Skip header (3 bytes) and decode
        let decoded = AuthenticationFailure::decode(&mut buf[3..].as_ref()).unwrap();

        assert_eq!(decoded.mm_cause.value, MmCause::MacFailure);
        assert!(decoded.auth_failure_parameter.is_none());
    }

    #[test]
    fn test_authentication_failure_synch_failure() {
        let auts = vec![0x44; 14];
        let msg = AuthenticationFailure::synch_failure(auts.clone());

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Skip header (3 bytes) and decode
        let decoded = AuthenticationFailure::decode(&mut buf[3..].as_ref()).unwrap();

        assert_eq!(decoded.mm_cause.value, MmCause::SynchFailure);
        assert!(decoded.auth_failure_parameter.is_some());
        assert_eq!(decoded.auth_failure_parameter.unwrap().value, auts);
    }

    #[test]
    fn test_authentication_result_basic() {
        let ng_ksi = NasKeySetIdentifier::new(SecurityContextType::Native, 4);
        let eap_data = vec![0x55; 8];
        let msg = AuthenticationResult::with_eap_data(ng_ksi, eap_data.clone());

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Verify header
        assert_eq!(buf[0], 0x7E); // EPD
        assert_eq!(buf[1], 0x00); // Security header type
        assert_eq!(buf[2], 0x5A); // Message type (Authentication Result)
        assert_eq!(buf[3] & 0x0F, 0x04); // ngKSI = 4

        // Skip header (3 bytes) and decode
        let decoded = AuthenticationResult::decode(&mut buf[3..].as_ref()).unwrap();

        assert_eq!(decoded.ng_ksi.ksi, 4);
        assert_eq!(decoded.eap_message.data, eap_data);
    }

    #[test]
    fn test_authentication_result_with_abba() {
        let ng_ksi = NasKeySetIdentifier::new(SecurityContextType::Native, 5);
        let eap_data = vec![0x66; 4];
        let mut msg = AuthenticationResult::with_eap_data(ng_ksi, eap_data.clone());
        msg.abba = Some(Abba::new(vec![0x00, 0x01]));

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Skip header (3 bytes) and decode
        let decoded = AuthenticationResult::decode(&mut buf[3..].as_ref()).unwrap();

        assert_eq!(decoded.ng_ksi.ksi, 5);
        assert_eq!(decoded.eap_message.data, eap_data);
        assert!(decoded.abba.is_some());
        assert_eq!(decoded.abba.unwrap().value, vec![0x00, 0x01]);
    }
}
