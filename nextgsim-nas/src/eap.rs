//! EAP (Extensible Authentication Protocol) message parsing
//!
//! This module implements EAP message encoding/decoding as defined in RFC 3748
//! and EAP-AKA' support as defined in RFC 5448 and 3GPP TS 33.402.
//!
//! # Overview
//!
//! EAP is used in 5G for authentication procedures, particularly EAP-AKA' which
//! is the primary authentication method for 5G networks.
//!
//! # Message Structure
//!
//! EAP messages have the following structure:
//! - Code (1 byte): Request, Response, Success, Failure, etc.
//! - Identifier (1 byte): Matches requests with responses
//! - Length (2 bytes): Total length including header
//! - Type (1 byte, optional): EAP method type
//! - Type-Data (variable): Method-specific data

use bytes::{Buf, BufMut};
use std::collections::BTreeMap;
use thiserror::Error;

/// Error type for EAP encoding/decoding
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum EapError {
    /// Buffer too short for decoding
    #[error("Buffer too short: expected at least {expected} bytes, got {actual}")]
    BufferTooShort {
        /// Expected minimum bytes
        expected: usize,
        /// Actual bytes available
        actual: usize,
    },
    /// Invalid EAP code
    #[error("Invalid EAP code: {0}")]
    InvalidCode(u8),
    /// Invalid EAP type
    #[error("Invalid EAP type: {0}")]
    InvalidType(u8),
    /// Invalid EAP-AKA' subtype
    #[error("Invalid EAP-AKA' subtype: {0}")]
    InvalidSubType(u8),
    /// Invalid attribute type
    #[error("Invalid attribute type: {0}")]
    InvalidAttributeType(u8),
    /// Invalid attribute length
    #[error("Invalid attribute length: {0}")]
    InvalidAttributeLength(u8),
    /// Read bytes exceeds message length
    #[error("Read bytes ({read}) exceeds message length ({length})")]
    LengthMismatch {
        /// Bytes read
        read: usize,
        /// Expected length
        length: usize,
    },
}

// ============================================================================
// EAP Code (RFC 3748 Section 4)
// ============================================================================

/// EAP Code values
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum EapCode {
    /// Request (1)
    Request = 1,
    /// Response (2)
    Response = 2,
    /// Success (3)
    Success = 3,
    /// Failure (4)
    Failure = 4,
    /// Initiate (5) - EAP-Initiate/Re-auth-Start
    Initiate = 5,
    /// Finish (6) - EAP-Finish/Re-auth
    Finish = 6,
}

impl TryFrom<u8> for EapCode {
    type Error = EapError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(EapCode::Request),
            2 => Ok(EapCode::Response),
            3 => Ok(EapCode::Success),
            4 => Ok(EapCode::Failure),
            5 => Ok(EapCode::Initiate),
            6 => Ok(EapCode::Finish),
            _ => Err(EapError::InvalidCode(value)),
        }
    }
}

impl From<EapCode> for u8 {
    fn from(code: EapCode) -> u8 {
        code as u8
    }
}

// ============================================================================
// EAP Type (RFC 3748 Section 5)
// ============================================================================

/// EAP Type values
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum EapType {
    /// No type (for Success/Failure messages)
    NoType = 0,
    /// Identity (1)
    Identity = 1,
    /// Notification (2)
    Notification = 2,
    /// Legacy Nak (Response only) (3)
    LegacyNak = 3,
    /// EAP-AKA (23)
    EapAka = 23,
    /// EAP-AKA' (50)
    EapAkaPrime = 50,
    /// Expanded EAP Type (254)
    Expanded = 254,
    /// Experimental (255)
    Experimental = 255,
}

impl TryFrom<u8> for EapType {
    type Error = EapError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(EapType::NoType),
            1 => Ok(EapType::Identity),
            2 => Ok(EapType::Notification),
            3 => Ok(EapType::LegacyNak),
            23 => Ok(EapType::EapAka),
            50 => Ok(EapType::EapAkaPrime),
            254 => Ok(EapType::Expanded),
            255 => Ok(EapType::Experimental),
            _ => Err(EapError::InvalidType(value)),
        }
    }
}

impl From<EapType> for u8 {
    fn from(t: EapType) -> u8 {
        t as u8
    }
}

// ============================================================================
// EAP-AKA' Subtype (RFC 4187 Section 11, RFC 5448)
// ============================================================================

/// EAP-AKA' Subtype values
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum EapAkaSubType {
    /// AKA-Challenge (1)
    AkaChallenge = 1,
    /// AKA-Authentication-Reject (2)
    AkaAuthenticationReject = 2,
    /// AKA-Synchronization-Failure (4)
    AkaSynchronizationFailure = 4,
    /// AKA-Identity (5)
    AkaIdentity = 5,
    /// AKA-Notification (12)
    AkaNotification = 12,
    /// AKA-Reauthentication (13)
    AkaReauthentication = 13,
    /// AKA-Client-Error (14)
    AkaClientError = 14,
}

impl TryFrom<u8> for EapAkaSubType {
    type Error = EapError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(EapAkaSubType::AkaChallenge),
            2 => Ok(EapAkaSubType::AkaAuthenticationReject),
            4 => Ok(EapAkaSubType::AkaSynchronizationFailure),
            5 => Ok(EapAkaSubType::AkaIdentity),
            12 => Ok(EapAkaSubType::AkaNotification),
            13 => Ok(EapAkaSubType::AkaReauthentication),
            14 => Ok(EapAkaSubType::AkaClientError),
            _ => Err(EapError::InvalidSubType(value)),
        }
    }
}

impl From<EapAkaSubType> for u8 {
    fn from(st: EapAkaSubType) -> u8 {
        st as u8
    }
}

// ============================================================================
// EAP-AKA' Attribute Type (RFC 4187 Section 11, RFC 5448)
// ============================================================================

/// EAP-AKA' Attribute Type values
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum EapAttributeType {
    /// AT_RAND (1) - Random challenge
    AtRand = 1,
    /// AT_AUTN (2) - Authentication token
    AtAutn = 2,
    /// AT_RES (3) - Authentication response
    AtRes = 3,
    /// AT_AUTS (4) - Resynchronization parameter
    AtAuts = 4,
    /// AT_PADDING (6) - Padding
    AtPadding = 6,
    /// AT_NONCE_MT (7) - Nonce from MT
    AtNonceMt = 7,
    /// AT_PERMANENT_ID_REQ (10) - Permanent identity request
    AtPermanentIdReq = 10,
    /// AT_MAC (11) - Message authentication code
    AtMac = 11,
    /// AT_NOTIFICATION (12) - Notification code
    AtNotification = 12,
    /// AT_ANY_ID_REQ (13) - Any identity request
    AtAnyIdReq = 13,
    /// AT_IDENTITY (14) - Identity
    AtIdentity = 14,
    /// AT_VERSION_LIST (15) - Version list
    AtVersionList = 15,
    /// AT_SELECTED_VERSION (16) - Selected version
    AtSelectedVersion = 16,
    /// AT_FULLAUTH_ID_REQ (17) - Full authentication identity request
    AtFullauthIdReq = 17,
    /// AT_COUNTER (19) - Counter
    AtCounter = 19,
    /// AT_COUNTER_TOO_SMALL (20) - Counter too small
    AtCounterTooSmall = 20,
    /// AT_NONCE_S (21) - Nonce from server
    AtNonceS = 21,
    /// AT_CLIENT_ERROR_CODE (22) - Client error code
    AtClientErrorCode = 22,
    /// AT_KDF_INPUT (23) - KDF input (network name)
    AtKdfInput = 23,
    /// AT_KDF (24) - Key derivation function
    AtKdf = 24,
    /// AT_IV (129) - Initialization vector
    AtIv = 129,
    /// AT_ENCR_DATA (130) - Encrypted data
    AtEncrData = 130,
    /// AT_NEXT_PSEUDONYM (132) - Next pseudonym
    AtNextPseudonym = 132,
    /// AT_NEXT_REAUTH_ID (133) - Next reauthentication identity
    AtNextReauthId = 133,
    /// AT_CHECKCODE (134) - Checkcode
    AtCheckcode = 134,
    /// AT_RESULT_IND (135) - Result indication
    AtResultInd = 135,
    /// AT_BIDDING (136) - Bidding
    AtBidding = 136,
    /// AT_IPMS_IND (137) - IPMS indication
    AtIpmsInd = 137,
    /// AT_IPMS_RES (138) - IPMS response
    AtIpmsRes = 138,
    /// AT_TRUST_IND (139) - Trust indication
    AtTrustInd = 139,
}

impl TryFrom<u8> for EapAttributeType {
    type Error = EapError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(EapAttributeType::AtRand),
            2 => Ok(EapAttributeType::AtAutn),
            3 => Ok(EapAttributeType::AtRes),
            4 => Ok(EapAttributeType::AtAuts),
            6 => Ok(EapAttributeType::AtPadding),
            7 => Ok(EapAttributeType::AtNonceMt),
            10 => Ok(EapAttributeType::AtPermanentIdReq),
            11 => Ok(EapAttributeType::AtMac),
            12 => Ok(EapAttributeType::AtNotification),
            13 => Ok(EapAttributeType::AtAnyIdReq),
            14 => Ok(EapAttributeType::AtIdentity),
            15 => Ok(EapAttributeType::AtVersionList),
            16 => Ok(EapAttributeType::AtSelectedVersion),
            17 => Ok(EapAttributeType::AtFullauthIdReq),
            19 => Ok(EapAttributeType::AtCounter),
            20 => Ok(EapAttributeType::AtCounterTooSmall),
            21 => Ok(EapAttributeType::AtNonceS),
            22 => Ok(EapAttributeType::AtClientErrorCode),
            23 => Ok(EapAttributeType::AtKdfInput),
            24 => Ok(EapAttributeType::AtKdf),
            129 => Ok(EapAttributeType::AtIv),
            130 => Ok(EapAttributeType::AtEncrData),
            132 => Ok(EapAttributeType::AtNextPseudonym),
            133 => Ok(EapAttributeType::AtNextReauthId),
            134 => Ok(EapAttributeType::AtCheckcode),
            135 => Ok(EapAttributeType::AtResultInd),
            136 => Ok(EapAttributeType::AtBidding),
            137 => Ok(EapAttributeType::AtIpmsInd),
            138 => Ok(EapAttributeType::AtIpmsRes),
            139 => Ok(EapAttributeType::AtTrustInd),
            _ => Err(EapError::InvalidAttributeType(value)),
        }
    }
}

impl From<EapAttributeType> for u8 {
    fn from(t: EapAttributeType) -> u8 {
        t as u8
    }
}


// ============================================================================
// EAP Attributes Container
// ============================================================================

/// Container for EAP-AKA' attributes
///
/// Stores attributes in order of receipt for proper encoding.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EapAttributes {
    /// Attributes stored by type
    attributes: BTreeMap<EapAttributeType, Vec<u8>>,
    /// Order in which attributes were received/added
    order: Vec<EapAttributeType>,
}

impl EapAttributes {
    /// Create a new empty attributes container
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the RAND attribute value (skipping 2-byte reserved field)
    pub fn get_rand(&self) -> Option<Vec<u8>> {
        self.attributes.get(&EapAttributeType::AtRand).and_then(|v| {
            if v.len() >= 2 {
                Some(v[2..].to_vec())
            } else {
                None
            }
        })
    }

    /// Get the MAC attribute value (skipping 2-byte reserved field)
    pub fn get_mac(&self) -> Option<Vec<u8>> {
        self.attributes.get(&EapAttributeType::AtMac).and_then(|v| {
            if v.len() >= 2 {
                Some(v[2..].to_vec())
            } else {
                None
            }
        })
    }

    /// Get the AUTN attribute value (skipping 2-byte reserved field)
    pub fn get_autn(&self) -> Option<Vec<u8>> {
        self.attributes.get(&EapAttributeType::AtAutn).and_then(|v| {
            if v.len() >= 2 {
                Some(v[2..].to_vec())
            } else {
                None
            }
        })
    }

    /// Get the client error code
    pub fn get_client_error_code(&self) -> Option<u16> {
        self.attributes
            .get(&EapAttributeType::AtClientErrorCode)
            .and_then(|v| {
                if v.len() == 2 {
                    Some(u16::from_be_bytes([v[0], v[1]]))
                } else {
                    None
                }
            })
    }

    /// Get the KDF value
    pub fn get_kdf(&self) -> Option<u16> {
        self.attributes.get(&EapAttributeType::AtKdf).and_then(|v| {
            if v.len() == 2 {
                Some(u16::from_be_bytes([v[0], v[1]]))
            } else {
                None
            }
        })
    }

    /// Get the KDF input (network name)
    pub fn get_kdf_input(&self) -> Option<Vec<u8>> {
        self.attributes
            .get(&EapAttributeType::AtKdfInput)
            .and_then(|v| {
                if v.len() >= 2 {
                    let len = u16::from_be_bytes([v[0], v[1]]) as usize;
                    if len + 2 <= v.len() {
                        Some(v[2..2 + len].to_vec())
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
    }

    /// Put RES attribute (with bit length prefix)
    pub fn put_res(&mut self, value: &[u8]) {
        let bit_length = (value.len() * 8) as u16;
        let mut data = Vec::with_capacity(2 + value.len());
        data.extend_from_slice(&bit_length.to_be_bytes());
        data.extend_from_slice(value);
        self.put_raw_attribute(EapAttributeType::AtRes, data);
    }

    /// Put MAC attribute (with 2-byte reserved field)
    pub fn put_mac(&mut self, value: &[u8]) {
        let mut data = Vec::with_capacity(2 + value.len());
        data.extend_from_slice(&[0, 0]); // Reserved
        data.extend_from_slice(value);
        self.put_raw_attribute(EapAttributeType::AtMac, data);
    }

    /// Replace MAC attribute value (for MAC calculation)
    pub fn replace_mac(&mut self, value: &[u8]) {
        let mut data = Vec::with_capacity(2 + value.len());
        data.extend_from_slice(&[0, 0]); // Reserved
        data.extend_from_slice(value);
        self.attributes.insert(EapAttributeType::AtMac, data);
    }

    /// Put KDF attribute
    pub fn put_kdf(&mut self, value: u16) {
        self.put_raw_attribute(EapAttributeType::AtKdf, value.to_be_bytes().to_vec());
    }

    /// Put client error code attribute
    pub fn put_client_error_code(&mut self, code: u16) {
        self.put_raw_attribute(
            EapAttributeType::AtClientErrorCode,
            code.to_be_bytes().to_vec(),
        );
    }

    /// Put AUTS attribute
    pub fn put_auts(&mut self, auts: Vec<u8>) {
        self.put_raw_attribute(EapAttributeType::AtAuts, auts);
    }

    /// Put a raw attribute value
    pub fn put_raw_attribute(&mut self, key: EapAttributeType, value: Vec<u8>) {
        if !self.attributes.contains_key(&key) {
            self.order.push(key);
        }
        self.attributes.insert(key, value);
    }

    /// Iterate over attributes in insertion order
    pub fn iter_ordered(&self) -> impl Iterator<Item = (EapAttributeType, &Vec<u8>)> {
        self.order
            .iter()
            .filter_map(|k| self.attributes.get(k).map(|v| (*k, v)))
    }

    /// Check if the container is empty
    pub fn is_empty(&self) -> bool {
        self.attributes.is_empty()
    }

    /// Get the number of attributes
    pub fn len(&self) -> usize {
        self.attributes.len()
    }
}

// ============================================================================
// EAP Message Types
// ============================================================================

/// Base EAP message
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Eap {
    /// Simple EAP message (Success/Failure with no type)
    Simple {
        /// EAP code
        code: EapCode,
        /// Identifier
        id: u8,
    },
    /// EAP Identity message
    Identity(EapIdentity),
    /// EAP Notification message
    Notification(EapNotification),
    /// EAP-AKA' message
    AkaPrime(EapAkaPrime),
}

impl Eap {
    /// Get the EAP code
    pub fn code(&self) -> EapCode {
        match self {
            Eap::Simple { code, .. } => *code,
            Eap::Identity(e) => e.code,
            Eap::Notification(e) => e.code,
            Eap::AkaPrime(e) => e.code,
        }
    }

    /// Get the identifier
    pub fn id(&self) -> u8 {
        match self {
            Eap::Simple { id, .. } => *id,
            Eap::Identity(e) => e.id,
            Eap::Notification(e) => e.id,
            Eap::AkaPrime(e) => e.id,
        }
    }

    /// Get the EAP type
    pub fn eap_type(&self) -> EapType {
        match self {
            Eap::Simple { .. } => EapType::NoType,
            Eap::Identity(_) => EapType::Identity,
            Eap::Notification(_) => EapType::Notification,
            Eap::AkaPrime(_) => EapType::EapAkaPrime,
        }
    }
}

/// EAP Identity message
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EapIdentity {
    /// EAP code
    pub code: EapCode,
    /// Identifier
    pub id: u8,
    /// Raw identity data
    pub raw_data: Vec<u8>,
}

impl EapIdentity {
    /// Create a new EAP Identity message
    pub fn new(code: EapCode, id: u8) -> Self {
        Self {
            code,
            id,
            raw_data: Vec::new(),
        }
    }

    /// Create a new EAP Identity message with data
    pub fn with_data(code: EapCode, id: u8, raw_data: Vec<u8>) -> Self {
        Self { code, id, raw_data }
    }
}

/// EAP Notification message
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EapNotification {
    /// EAP code
    pub code: EapCode,
    /// Identifier
    pub id: u8,
    /// Raw notification data
    pub raw_data: Vec<u8>,
}

impl EapNotification {
    /// Create a new EAP Notification message
    pub fn new(code: EapCode, id: u8) -> Self {
        Self {
            code,
            id,
            raw_data: Vec::new(),
        }
    }

    /// Create a new EAP Notification message with data
    pub fn with_data(code: EapCode, id: u8, raw_data: Vec<u8>) -> Self {
        Self { code, id, raw_data }
    }
}

/// EAP-AKA' message
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EapAkaPrime {
    /// EAP code
    pub code: EapCode,
    /// Identifier
    pub id: u8,
    /// EAP-AKA' subtype
    pub sub_type: EapAkaSubType,
    /// Attributes
    pub attributes: EapAttributes,
}

impl EapAkaPrime {
    /// Create a new EAP-AKA' message
    pub fn new(code: EapCode, id: u8, sub_type: EapAkaSubType) -> Self {
        Self {
            code,
            id,
            sub_type,
            attributes: EapAttributes::new(),
        }
    }

    /// Create an AKA-Challenge message
    pub fn challenge(code: EapCode, id: u8) -> Self {
        Self::new(code, id, EapAkaSubType::AkaChallenge)
    }

    /// Create an AKA-Authentication-Reject message
    pub fn authentication_reject(code: EapCode, id: u8) -> Self {
        Self::new(code, id, EapAkaSubType::AkaAuthenticationReject)
    }

    /// Create an AKA-Synchronization-Failure message
    pub fn synchronization_failure(code: EapCode, id: u8) -> Self {
        Self::new(code, id, EapAkaSubType::AkaSynchronizationFailure)
    }

    /// Create an AKA-Client-Error message
    pub fn client_error(code: EapCode, id: u8, error_code: u16) -> Self {
        let mut msg = Self::new(code, id, EapAkaSubType::AkaClientError);
        msg.attributes.put_client_error_code(error_code);
        msg
    }
}


// ============================================================================
// EAP Encoding
// ============================================================================

/// Encode an EAP message to bytes
pub fn encode_eap<B: BufMut>(buf: &mut B, eap: &Eap) {
    // Write code and id
    buf.put_u8(eap.code().into());
    buf.put_u8(eap.id());

    match eap {
        Eap::Simple { .. } => {
            // Success/Failure messages have length 4 and no type
            buf.put_u16(4);
        }
        Eap::Identity(identity) => {
            // Length placeholder (will be updated)
            let len = 5 + identity.raw_data.len();
            buf.put_u16(len as u16);
            buf.put_u8(EapType::Identity.into());
            buf.put_slice(&identity.raw_data);
        }
        Eap::Notification(notification) => {
            let len = 5 + notification.raw_data.len();
            buf.put_u16(len as u16);
            buf.put_u8(EapType::Notification.into());
            buf.put_slice(&notification.raw_data);
        }
        Eap::AkaPrime(aka_prime) => {
            // Calculate total length
            let mut attr_len = 0;
            for (_, value) in aka_prime.attributes.iter_ordered() {
                // Each attribute: 1 byte type + 1 byte length + value (padded to 4 bytes)
                attr_len += 2 + value.len();
            }
            // Header (4) + type (1) + subtype (1) + reserved (2) + attributes
            let total_len = 4 + 1 + 1 + 2 + attr_len;
            
            buf.put_u16(total_len as u16);
            buf.put_u8(EapType::EapAkaPrime.into());
            buf.put_u8(aka_prime.sub_type.into());
            buf.put_u16(0); // Reserved

            // Encode attributes in order
            for (attr_type, value) in aka_prime.attributes.iter_ordered() {
                buf.put_u8(attr_type.into());
                // Length is in 4-byte units, including type and length bytes
                let attr_length = (value.len() + 2) / 4;
                buf.put_u8(attr_length as u8);
                buf.put_slice(value);
            }
        }
    }
}

/// Encode an EAP message to a new Vec<u8>
pub fn encode_eap_to_vec(eap: &Eap) -> Vec<u8> {
    let mut buf = Vec::new();
    encode_eap(&mut buf, eap);
    buf
}

// ============================================================================
// EAP Decoding
// ============================================================================

/// Decode an EAP message from bytes
pub fn decode_eap<B: Buf>(buf: &mut B) -> Result<Eap, EapError> {
    if buf.remaining() < 4 {
        return Err(EapError::BufferTooShort {
            expected: 4,
            actual: buf.remaining(),
        });
    }

    let code = EapCode::try_from(buf.get_u8())?;
    let id = buf.get_u8();
    let length = buf.get_u16() as usize;

    // Validate length
    if length < 4 {
        return Err(EapError::BufferTooShort {
            expected: 4,
            actual: length,
        });
    }

    // Success/Failure messages have length 4 and no type
    if length == 4 {
        return Ok(Eap::Simple { code, id });
    }

    // Read type
    if buf.remaining() < 1 {
        return Err(EapError::BufferTooShort {
            expected: 1,
            actual: buf.remaining(),
        });
    }
    let eap_type = EapType::try_from(buf.get_u8())?;

    // Inner length (excluding code, id, length, type)
    let inner_length = length - 5;

    if buf.remaining() < inner_length {
        return Err(EapError::BufferTooShort {
            expected: inner_length,
            actual: buf.remaining(),
        });
    }

    match eap_type {
        EapType::EapAkaPrime => decode_eap_aka_prime(buf, code, id, inner_length),
        EapType::Identity => {
            let mut raw_data = vec![0u8; inner_length];
            buf.copy_to_slice(&mut raw_data);
            Ok(Eap::Identity(EapIdentity::with_data(code, id, raw_data)))
        }
        EapType::Notification => {
            let mut raw_data = vec![0u8; inner_length];
            buf.copy_to_slice(&mut raw_data);
            Ok(Eap::Notification(EapNotification::with_data(
                code, id, raw_data,
            )))
        }
        _ => {
            // Skip unknown types
            buf.advance(inner_length);
            Err(EapError::InvalidType(eap_type.into()))
        }
    }
}

/// Decode an EAP-AKA' message
fn decode_eap_aka_prime<B: Buf>(
    buf: &mut B,
    code: EapCode,
    id: u8,
    inner_length: usize,
) -> Result<Eap, EapError> {
    if inner_length < 3 {
        return Err(EapError::BufferTooShort {
            expected: 3,
            actual: inner_length,
        });
    }

    let mut read_bytes = 0;

    // Decode subtype
    let sub_type = EapAkaSubType::try_from(buf.get_u8())?;
    read_bytes += 1;

    // Skip reserved 2 bytes
    let _ = buf.get_u16();
    read_bytes += 2;

    let mut aka_prime = EapAkaPrime::new(code, id, sub_type);

    // Decode attributes
    while read_bytes < inner_length {
        if buf.remaining() < 2 {
            return Err(EapError::BufferTooShort {
                expected: 2,
                actual: buf.remaining(),
            });
        }

        // Decode attribute type
        let attr_type = EapAttributeType::try_from(buf.get_u8())?;
        read_bytes += 1;

        // Decode attribute length (in 4-byte units)
        let attr_length_units = buf.get_u8();
        read_bytes += 1;

        if attr_length_units < 1 {
            return Err(EapError::InvalidAttributeLength(attr_length_units));
        }

        // Calculate actual value length (length * 4 - 2 for type and length bytes)
        let value_length = (attr_length_units as usize) * 4 - 2;

        if buf.remaining() < value_length {
            return Err(EapError::BufferTooShort {
                expected: value_length,
                actual: buf.remaining(),
            });
        }

        let mut value = vec![0u8; value_length];
        buf.copy_to_slice(&mut value);
        read_bytes += value_length;

        aka_prime.attributes.put_raw_attribute(attr_type, value);
    }

    if read_bytes != inner_length {
        return Err(EapError::LengthMismatch {
            read: read_bytes,
            length: inner_length,
        });
    }

    Ok(Eap::AkaPrime(aka_prime))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eap_code_conversion() {
        assert_eq!(EapCode::try_from(1).unwrap(), EapCode::Request);
        assert_eq!(EapCode::try_from(2).unwrap(), EapCode::Response);
        assert_eq!(EapCode::try_from(3).unwrap(), EapCode::Success);
        assert_eq!(EapCode::try_from(4).unwrap(), EapCode::Failure);
        assert!(EapCode::try_from(0).is_err());
        assert!(EapCode::try_from(7).is_err());
    }

    #[test]
    fn test_eap_type_conversion() {
        assert_eq!(EapType::try_from(1).unwrap(), EapType::Identity);
        assert_eq!(EapType::try_from(50).unwrap(), EapType::EapAkaPrime);
        assert!(EapType::try_from(100).is_err());
    }

    #[test]
    fn test_eap_aka_subtype_conversion() {
        assert_eq!(
            EapAkaSubType::try_from(1).unwrap(),
            EapAkaSubType::AkaChallenge
        );
        assert_eq!(
            EapAkaSubType::try_from(14).unwrap(),
            EapAkaSubType::AkaClientError
        );
        assert!(EapAkaSubType::try_from(0).is_err());
    }

    #[test]
    fn test_eap_attribute_type_conversion() {
        assert_eq!(
            EapAttributeType::try_from(1).unwrap(),
            EapAttributeType::AtRand
        );
        assert_eq!(
            EapAttributeType::try_from(24).unwrap(),
            EapAttributeType::AtKdf
        );
        assert!(EapAttributeType::try_from(200).is_err());
    }

    #[test]
    fn test_eap_attributes_put_and_get() {
        let mut attrs = EapAttributes::new();
        
        // Test KDF
        attrs.put_kdf(1);
        assert_eq!(attrs.get_kdf(), Some(1));

        // Test client error code
        attrs.put_client_error_code(0x1234);
        assert_eq!(attrs.get_client_error_code(), Some(0x1234));

        // Test MAC
        let mac = vec![0x01, 0x02, 0x03, 0x04];
        attrs.put_mac(&mac);
        assert_eq!(attrs.get_mac(), Some(mac.clone()));

        // Test replace MAC
        let new_mac = vec![0x05, 0x06, 0x07, 0x08];
        attrs.replace_mac(&new_mac);
        assert_eq!(attrs.get_mac(), Some(new_mac));
    }

    #[test]
    fn test_encode_decode_simple_eap() {
        let eap = Eap::Simple {
            code: EapCode::Success,
            id: 42,
        };

        let encoded = encode_eap_to_vec(&eap);
        assert_eq!(encoded, vec![0x03, 0x2A, 0x00, 0x04]);

        let decoded = decode_eap(&mut encoded.as_slice()).unwrap();
        assert_eq!(decoded, eap);
    }

    #[test]
    fn test_encode_decode_identity() {
        let eap = Eap::Identity(EapIdentity::with_data(
            EapCode::Request,
            1,
            b"test@example.com".to_vec(),
        ));

        let encoded = encode_eap_to_vec(&eap);
        let decoded = decode_eap(&mut encoded.as_slice()).unwrap();
        assert_eq!(decoded, eap);
    }

    #[test]
    fn test_encode_decode_aka_prime() {
        let mut aka_prime = EapAkaPrime::new(EapCode::Request, 1, EapAkaSubType::AkaChallenge);
        aka_prime.attributes.put_kdf(1);

        let eap = Eap::AkaPrime(aka_prime);
        let encoded = encode_eap_to_vec(&eap);
        let decoded = decode_eap(&mut encoded.as_slice()).unwrap();
        
        if let (Eap::AkaPrime(orig), Eap::AkaPrime(dec)) = (&eap, &decoded) {
            assert_eq!(orig.code, dec.code);
            assert_eq!(orig.id, dec.id);
            assert_eq!(orig.sub_type, dec.sub_type);
            assert_eq!(orig.attributes.get_kdf(), dec.attributes.get_kdf());
        } else {
            panic!("Expected AkaPrime messages");
        }
    }

    #[test]
    fn test_decode_buffer_too_short() {
        let buf: &[u8] = &[0x01, 0x02];
        let result = decode_eap(&mut &buf[..]);
        assert!(matches!(result, Err(EapError::BufferTooShort { .. })));
    }

    #[test]
    fn test_decode_invalid_code() {
        let buf: &[u8] = &[0x00, 0x01, 0x00, 0x04];
        let result = decode_eap(&mut &buf[..]);
        assert!(matches!(result, Err(EapError::InvalidCode(0))));
    }

    #[test]
    fn test_eap_aka_prime_client_error() {
        let eap = EapAkaPrime::client_error(EapCode::Response, 5, 0x0001);
        assert_eq!(eap.sub_type, EapAkaSubType::AkaClientError);
        assert_eq!(eap.attributes.get_client_error_code(), Some(0x0001));
    }
}
