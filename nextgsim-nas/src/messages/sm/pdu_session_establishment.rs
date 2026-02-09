//! PDU Session Establishment Messages (3GPP TS 24.501 Section 8.3.1)
//!
//! This module implements the PDU Session Establishment procedure messages:
//! - PDU Session Establishment Request (UE to network)
//! - PDU Session Establishment Accept (network to UE)
//! - PDU Session Establishment Reject (network to UE)

use bytes::{Buf, BufMut};
use thiserror::Error;

use crate::enums::SmMessageType;


/// Error type for PDU Session Establishment message encoding/decoding
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PduSessionEstablishmentError {
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
        expected: SmMessageType,
        /// Actual message type
        actual: SmMessageType,
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
// 5GSM Cause (Type 3 - fixed length)
// ============================================================================

/// 5GSM Cause values (3GPP TS 24.501 Section 9.11.4.2)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum SmCause {
    /// Operator determined barring
    OperatorDeterminedBarring = 8,
    /// Insufficient resources
    InsufficientResources = 26,
    /// Missing or unknown DNN
    MissingOrUnknownDnn = 27,
    /// Unknown PDU session type
    UnknownPduSessionType = 28,
    /// User authentication or authorization failed
    UserAuthenticationFailed = 29,
    /// Request rejected, unspecified
    RequestRejectedUnspecified = 31,
    /// Service option not supported
    ServiceOptionNotSupported = 32,
    /// Requested service option not subscribed
    RequestedServiceOptionNotSubscribed = 33,
    /// PTI already in use
    PtiAlreadyInUse = 35,
    /// Regular deactivation
    RegularDeactivation = 36,
    /// Network failure
    NetworkFailure = 38,
    /// Reactivation requested
    ReactivationRequested = 39,
    /// Semantic error in the TFT operation
    SemanticErrorInTftOperation = 41,
    /// Syntactical error in the TFT operation
    SyntacticalErrorInTftOperation = 42,
    /// Invalid PDU session identity
    InvalidPduSessionIdentity = 43,
    /// Semantic errors in packet filter(s)
    SemanticErrorsInPacketFilters = 44,
    /// Syntactical errors in packet filter(s)
    SyntacticalErrorsInPacketFilters = 45,
    /// Out of LADN service area
    OutOfLadnServiceArea = 46,
    /// PTI mismatch
    PtiMismatch = 47,
    /// PDU session type IPv4 only allowed
    PduSessionTypeIpv4OnlyAllowed = 50,
    /// PDU session type IPv6 only allowed
    PduSessionTypeIpv6OnlyAllowed = 51,
    /// PDU session does not exist
    PduSessionDoesNotExist = 54,
    /// Insufficient resources for specific slice and DNN
    InsufficientResourcesForSliceAndDnn = 67,
    /// Not supported SSC mode
    NotSupportedSscMode = 68,
    /// Insufficient resources for specific slice
    InsufficientResourcesForSlice = 69,
    /// Missing or unknown DNN in a slice
    MissingOrUnknownDnnInSlice = 70,
    /// Invalid PTI value
    InvalidPtiValue = 81,
    /// Maximum data rate per UE for user-plane integrity protection is too low
    MaxDataRateTooLow = 82,
    /// Semantic error in the QoS operation
    SemanticErrorInQosOperation = 83,
    /// Syntactical error in the QoS operation
    SyntacticalErrorInQosOperation = 84,
    /// Invalid mapped EPS bearer identity
    InvalidMappedEpsBearerIdentity = 85,
    /// Semantically incorrect message
    SemanticallyIncorrectMessage = 95,
    /// Invalid mandatory information
    InvalidMandatoryInformation = 96,
    /// Message type non-existent or not implemented
    MessageTypeNonExistent = 97,
    /// Message type not compatible with the protocol state
    MessageTypeNotCompatible = 98,
    /// Information element non-existent or not implemented
    IeNonExistent = 99,
    /// Conditional IE error
    ConditionalIeError = 100,
    /// Message not compatible with the protocol state
    MessageNotCompatible = 101,
    /// Protocol error, unspecified
    #[default]
    ProtocolErrorUnspecified = 111,
}

impl TryFrom<u8> for SmCause {
    type Error = PduSessionEstablishmentError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            8 => Ok(SmCause::OperatorDeterminedBarring),
            26 => Ok(SmCause::InsufficientResources),
            27 => Ok(SmCause::MissingOrUnknownDnn),
            28 => Ok(SmCause::UnknownPduSessionType),
            29 => Ok(SmCause::UserAuthenticationFailed),
            31 => Ok(SmCause::RequestRejectedUnspecified),
            32 => Ok(SmCause::ServiceOptionNotSupported),
            33 => Ok(SmCause::RequestedServiceOptionNotSubscribed),
            35 => Ok(SmCause::PtiAlreadyInUse),
            36 => Ok(SmCause::RegularDeactivation),
            38 => Ok(SmCause::NetworkFailure),
            39 => Ok(SmCause::ReactivationRequested),
            41 => Ok(SmCause::SemanticErrorInTftOperation),
            42 => Ok(SmCause::SyntacticalErrorInTftOperation),
            43 => Ok(SmCause::InvalidPduSessionIdentity),
            44 => Ok(SmCause::SemanticErrorsInPacketFilters),
            45 => Ok(SmCause::SyntacticalErrorsInPacketFilters),
            46 => Ok(SmCause::OutOfLadnServiceArea),
            47 => Ok(SmCause::PtiMismatch),
            50 => Ok(SmCause::PduSessionTypeIpv4OnlyAllowed),
            51 => Ok(SmCause::PduSessionTypeIpv6OnlyAllowed),
            54 => Ok(SmCause::PduSessionDoesNotExist),
            67 => Ok(SmCause::InsufficientResourcesForSliceAndDnn),
            68 => Ok(SmCause::NotSupportedSscMode),
            69 => Ok(SmCause::InsufficientResourcesForSlice),
            70 => Ok(SmCause::MissingOrUnknownDnnInSlice),
            81 => Ok(SmCause::InvalidPtiValue),
            82 => Ok(SmCause::MaxDataRateTooLow),
            83 => Ok(SmCause::SemanticErrorInQosOperation),
            84 => Ok(SmCause::SyntacticalErrorInQosOperation),
            85 => Ok(SmCause::InvalidMappedEpsBearerIdentity),
            95 => Ok(SmCause::SemanticallyIncorrectMessage),
            96 => Ok(SmCause::InvalidMandatoryInformation),
            97 => Ok(SmCause::MessageTypeNonExistent),
            98 => Ok(SmCause::MessageTypeNotCompatible),
            99 => Ok(SmCause::IeNonExistent),
            100 => Ok(SmCause::ConditionalIeError),
            101 => Ok(SmCause::MessageNotCompatible),
            111 => Ok(SmCause::ProtocolErrorUnspecified),
            _ => Ok(SmCause::ProtocolErrorUnspecified), // Unknown causes map to protocol error
        }
    }
}

/// 5GSM Cause IE (Type 3 - 1 byte)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Ie5gSmCause {
    /// Cause value
    pub value: SmCause,
}

impl Ie5gSmCause {
    /// Create a new 5GSM Cause IE
    pub fn new(value: SmCause) -> Self {
        Self { value }
    }

    /// Decode from bytes
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, PduSessionEstablishmentError> {
        if buf.remaining() < 1 {
            return Err(PduSessionEstablishmentError::BufferTooShort {
                expected: 1,
                actual: buf.remaining(),
            });
        }
        let value = SmCause::try_from(buf.get_u8())?;
        Ok(Self { value })
    }

    /// Encode to bytes
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        buf.put_u8(self.value as u8);
    }
}


// ============================================================================
// PDU Session Type (Type 1 - half octet)
// ============================================================================

/// PDU Session Type values (3GPP TS 24.501 Section 9.11.4.11)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum PduSessionTypeValue {
    /// IPv4
    #[default]
    Ipv4 = 0b001,
    /// IPv6
    Ipv6 = 0b010,
    /// IPv4v6
    Ipv4v6 = 0b011,
    /// Unstructured
    Unstructured = 0b100,
    /// Ethernet
    Ethernet = 0b101,
}

impl TryFrom<u8> for PduSessionTypeValue {
    type Error = PduSessionEstablishmentError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value & 0x07 {
            0b001 => Ok(PduSessionTypeValue::Ipv4),
            0b010 => Ok(PduSessionTypeValue::Ipv6),
            0b011 => Ok(PduSessionTypeValue::Ipv4v6),
            0b100 => Ok(PduSessionTypeValue::Unstructured),
            0b101 => Ok(PduSessionTypeValue::Ethernet),
            _ => Err(PduSessionEstablishmentError::InvalidIeValue(format!(
                "Invalid PDU session type: 0x{value:02X}"
            ))),
        }
    }
}

/// Selected PDU Session Type IE (Type 1 - half octet)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IeSelectedPduSessionType {
    /// PDU session type value
    pub value: PduSessionTypeValue,
}

impl IeSelectedPduSessionType {
    /// Create a new Selected PDU Session Type IE
    pub fn new(value: PduSessionTypeValue) -> Self {
        Self { value }
    }

    /// Decode from a 4-bit value
    pub fn decode(value: u8) -> Result<Self, PduSessionEstablishmentError> {
        let pdu_type = PduSessionTypeValue::try_from(value & 0x0F)?;
        Ok(Self { value: pdu_type })
    }

    /// Encode to a 4-bit value
    pub fn encode(&self) -> u8 {
        self.value as u8
    }
}

// ============================================================================
// SSC Mode (Type 1 - half octet)
// ============================================================================

/// SSC Mode values (3GPP TS 24.501 Section 9.11.4.16)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum SscModeValue {
    /// SSC mode 1
    #[default]
    SscMode1 = 0b001,
    /// SSC mode 2
    SscMode2 = 0b010,
    /// SSC mode 3
    SscMode3 = 0b011,
}

impl TryFrom<u8> for SscModeValue {
    type Error = PduSessionEstablishmentError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value & 0x07 {
            0b001 => Ok(SscModeValue::SscMode1),
            0b010 => Ok(SscModeValue::SscMode2),
            0b011 => Ok(SscModeValue::SscMode3),
            _ => Err(PduSessionEstablishmentError::InvalidIeValue(format!(
                "Invalid SSC mode: 0x{value:02X}"
            ))),
        }
    }
}

/// Selected SSC Mode IE (Type 1 - half octet)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IeSelectedSscMode {
    /// SSC mode value
    pub value: SscModeValue,
}

impl IeSelectedSscMode {
    /// Create a new Selected SSC Mode IE
    pub fn new(value: SscModeValue) -> Self {
        Self { value }
    }

    /// Decode from a 4-bit value
    pub fn decode(value: u8) -> Result<Self, PduSessionEstablishmentError> {
        let ssc_mode = SscModeValue::try_from(value & 0x0F)?;
        Ok(Self { value: ssc_mode })
    }

    /// Encode to a 4-bit value
    pub fn encode(&self) -> u8 {
        self.value as u8
    }
}

// ============================================================================
// Integrity Protection Maximum Data Rate (Type 4)
// ============================================================================

/// Maximum data rate values for integrity protection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum MaxDataRate {
    /// 64 kbps
    Rate64Kbps = 0x00,
    /// Full data rate
    #[default]
    FullRate = 0xFF,
}

impl TryFrom<u8> for MaxDataRate {
    type Error = PduSessionEstablishmentError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0x00 => Ok(MaxDataRate::Rate64Kbps),
            0xFF => Ok(MaxDataRate::FullRate),
            _ => Ok(MaxDataRate::FullRate), // Default to full rate for unknown values
        }
    }
}

/// Integrity Protection Maximum Data Rate IE (Type 4)
///
/// 3GPP TS 24.501 Section 9.11.4.7
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IeIntegrityProtectionMaxDataRate {
    /// Maximum data rate for uplink
    pub uplink: MaxDataRate,
    /// Maximum data rate for downlink
    pub downlink: MaxDataRate,
}

impl IeIntegrityProtectionMaxDataRate {
    /// Create a new Integrity Protection Maximum Data Rate IE
    pub fn new(uplink: MaxDataRate, downlink: MaxDataRate) -> Self {
        Self { uplink, downlink }
    }

    /// Create with full rate for both directions
    pub fn full_rate() -> Self {
        Self {
            uplink: MaxDataRate::FullRate,
            downlink: MaxDataRate::FullRate,
        }
    }

    /// Decode from bytes (without IEI, with length)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, PduSessionEstablishmentError> {
        if buf.remaining() < 2 {
            return Err(PduSessionEstablishmentError::BufferTooShort {
                expected: 2,
                actual: buf.remaining(),
            });
        }

        let uplink = MaxDataRate::try_from(buf.get_u8())?;
        let downlink = MaxDataRate::try_from(buf.get_u8())?;

        Ok(Self { uplink, downlink })
    }

    /// Encode to bytes (without IEI)
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        buf.put_u8(self.uplink as u8);
        buf.put_u8(self.downlink as u8);
    }

    /// Get encoded length
    pub fn encoded_len(&self) -> usize {
        2
    }
}


// ============================================================================
// QoS Rules (Type 6 - variable length)
// ============================================================================

/// Authorized QoS Rules IE (Type 6)
///
/// 3GPP TS 24.501 Section 9.11.4.13
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct IeQosRules {
    /// Raw QoS rules data
    pub data: Vec<u8>,
}

impl IeQosRules {
    /// Create a new QoS Rules IE with raw data
    pub fn new(data: Vec<u8>) -> Self {
        Self { data }
    }

    /// Decode from bytes (without IEI, with 2-byte length)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, PduSessionEstablishmentError> {
        if buf.remaining() < 2 {
            return Err(PduSessionEstablishmentError::BufferTooShort {
                expected: 2,
                actual: buf.remaining(),
            });
        }

        let length = buf.get_u16() as usize;
        if buf.remaining() < length {
            return Err(PduSessionEstablishmentError::BufferTooShort {
                expected: length,
                actual: buf.remaining(),
            });
        }

        let mut data = vec![0u8; length];
        buf.copy_to_slice(&mut data);

        Ok(Self { data })
    }

    /// Encode to bytes (without IEI, with 2-byte length)
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
// Session-AMBR (Type 4 - variable length)
// ============================================================================

/// Session-AMBR IE (Type 4)
///
/// 3GPP TS 24.501 Section 9.11.4.14
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct IeSessionAmbr {
    /// Unit for downlink session AMBR
    pub downlink_unit: u8,
    /// Downlink session AMBR value
    pub downlink: u16,
    /// Unit for uplink session AMBR
    pub uplink_unit: u8,
    /// Uplink session AMBR value
    pub uplink: u16,
}

impl IeSessionAmbr {
    /// Create a new Session-AMBR IE
    pub fn new(downlink_unit: u8, downlink: u16, uplink_unit: u8, uplink: u16) -> Self {
        Self {
            downlink_unit,
            downlink,
            uplink_unit,
            uplink,
        }
    }

    /// Decode from bytes (without IEI, with 1-byte length)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, PduSessionEstablishmentError> {
        if buf.remaining() < 1 {
            return Err(PduSessionEstablishmentError::BufferTooShort {
                expected: 1,
                actual: buf.remaining(),
            });
        }

        let length = buf.get_u8() as usize;
        if buf.remaining() < length || length < 6 {
            return Err(PduSessionEstablishmentError::BufferTooShort {
                expected: length.max(6),
                actual: buf.remaining(),
            });
        }

        let downlink_unit = buf.get_u8();
        let downlink = buf.get_u16();
        let uplink_unit = buf.get_u8();
        let uplink = buf.get_u16();

        // Skip any remaining bytes
        if length > 6 {
            buf.advance(length - 6);
        }

        Ok(Self {
            downlink_unit,
            downlink,
            uplink_unit,
            uplink,
        })
    }

    /// Encode to bytes (without IEI, with 1-byte length)
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        buf.put_u8(6); // Length
        buf.put_u8(self.downlink_unit);
        buf.put_u16(self.downlink);
        buf.put_u8(self.uplink_unit);
        buf.put_u16(self.uplink);
    }

    /// Get encoded length (including 1-byte length field)
    pub fn encoded_len(&self) -> usize {
        7 // 1 byte length + 6 bytes value
    }
}

// ============================================================================
// PDU Address (Type 4 - variable length)
// ============================================================================

/// PDU Address type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum PduAddressType {
    /// IPv4
    #[default]
    Ipv4 = 0b001,
    /// IPv6
    Ipv6 = 0b010,
    /// IPv4v6
    Ipv4v6 = 0b011,
}

impl TryFrom<u8> for PduAddressType {
    type Error = PduSessionEstablishmentError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value & 0x07 {
            0b001 => Ok(PduAddressType::Ipv4),
            0b010 => Ok(PduAddressType::Ipv6),
            0b011 => Ok(PduAddressType::Ipv4v6),
            _ => Err(PduSessionEstablishmentError::InvalidIeValue(format!(
                "Invalid PDU address type: 0x{value:02X}"
            ))),
        }
    }
}

/// PDU Address IE (Type 4)
///
/// 3GPP TS 24.501 Section 9.11.4.10
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct IePduAddress {
    /// PDU address type
    pub address_type: PduAddressType,
    /// Address data (4 bytes for IPv4, 8 bytes for IPv6 interface ID, 12 bytes for IPv4v6)
    pub address: Vec<u8>,
}

impl IePduAddress {
    /// Create a new PDU Address IE
    pub fn new(address_type: PduAddressType, address: Vec<u8>) -> Self {
        Self {
            address_type,
            address,
        }
    }

    /// Create an IPv4 PDU address
    pub fn ipv4(addr: [u8; 4]) -> Self {
        Self {
            address_type: PduAddressType::Ipv4,
            address: addr.to_vec(),
        }
    }

    /// Create an IPv6 PDU address (interface identifier only)
    pub fn ipv6(interface_id: [u8; 8]) -> Self {
        Self {
            address_type: PduAddressType::Ipv6,
            address: interface_id.to_vec(),
        }
    }

    /// Decode from bytes (without IEI, with 1-byte length)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, PduSessionEstablishmentError> {
        if buf.remaining() < 1 {
            return Err(PduSessionEstablishmentError::BufferTooShort {
                expected: 1,
                actual: buf.remaining(),
            });
        }

        let length = buf.get_u8() as usize;
        if buf.remaining() < length || length < 1 {
            return Err(PduSessionEstablishmentError::BufferTooShort {
                expected: length.max(1),
                actual: buf.remaining(),
            });
        }

        let type_octet = buf.get_u8();
        let address_type = PduAddressType::try_from(type_octet & 0x07)?;

        let addr_len = length - 1;
        let mut address = vec![0u8; addr_len];
        buf.copy_to_slice(&mut address);

        Ok(Self {
            address_type,
            address,
        })
    }

    /// Encode to bytes (without IEI, with 1-byte length)
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        let length = 1 + self.address.len();
        buf.put_u8(length as u8);
        buf.put_u8(self.address_type as u8);
        buf.put_slice(&self.address);
    }

    /// Get encoded length (including 1-byte length field)
    pub fn encoded_len(&self) -> usize {
        2 + self.address.len() // 1 byte length + 1 byte type + address
    }
}

// ============================================================================
// DNN (Data Network Name) (Type 4 - variable length)
// ============================================================================

/// DNN (Data Network Name) IE (Type 4)
///
/// 3GPP TS 24.501 Section 9.11.2.1A
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct IeDnn {
    /// DNN value (encoded as length-prefixed labels)
    pub value: Vec<u8>,
}

impl IeDnn {
    /// Create a new DNN IE from raw encoded data
    pub fn new(value: Vec<u8>) -> Self {
        Self { value }
    }

    /// Create a DNN IE from a string (e.g., "internet")
    pub fn from_string(dnn: &str) -> Self {
        let mut value = Vec::new();
        for label in dnn.split('.') {
            value.push(label.len() as u8);
            value.extend_from_slice(label.as_bytes());
        }
        Self { value }
    }

    /// Decode from bytes (without IEI, with 1-byte length)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, PduSessionEstablishmentError> {
        if buf.remaining() < 1 {
            return Err(PduSessionEstablishmentError::BufferTooShort {
                expected: 1,
                actual: buf.remaining(),
            });
        }

        let length = buf.get_u8() as usize;
        if buf.remaining() < length {
            return Err(PduSessionEstablishmentError::BufferTooShort {
                expected: length,
                actual: buf.remaining(),
            });
        }

        let mut value = vec![0u8; length];
        buf.copy_to_slice(&mut value);

        Ok(Self { value })
    }

    /// Encode to bytes (without IEI, with 1-byte length)
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        buf.put_u8(self.value.len() as u8);
        buf.put_slice(&self.value);
    }

    /// Get encoded length (including 1-byte length field)
    pub fn encoded_len(&self) -> usize {
        1 + self.value.len()
    }
}
