//! PDU Session Establishment Messages (3GPP TS 24.501 Section 8.3.1)
//!
//! This module implements the PDU Session Establishment procedure messages:
//! - PDU Session Establishment Request (UE to network, Section 8.3.1)
//! - PDU Session Establishment Accept (network to UE, Section 8.3.2)
//! - PDU Session Establishment Reject (network to UE, Section 8.3.3)

use bytes::{Buf, BufMut};
use thiserror::Error;

use crate::enums::SmMessageType;
use crate::header::PlainSmHeader;

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
    /// Missing mandatory IE (TS 24.501 Section 7.7.2)
    #[error("Missing mandatory IE: {0}")]
    MissingMandatoryIe(&'static str),
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
    /// PDU session type `IPv4v6` only allowed
    PduSessionTypeIpv4v6OnlyAllowed = 57,
    /// PDU session type Unstructured only allowed
    PduSessionTypeUnstructuredOnlyAllowed = 58,
    /// Unsupported 5QI value
    Unsupported5qiValue = 59,
    /// PDU session type Ethernet only allowed
    PduSessionTypeEthernetOnlyAllowed = 61,
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
    /// Semantic error in the `QoS` operation
    SemanticErrorInQosOperation = 83,
    /// Syntactical error in the `QoS` operation
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
            57 => Ok(SmCause::PduSessionTypeIpv4v6OnlyAllowed),
            58 => Ok(SmCause::PduSessionTypeUnstructuredOnlyAllowed),
            59 => Ok(SmCause::Unsupported5qiValue),
            61 => Ok(SmCause::PduSessionTypeEthernetOnlyAllowed),
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
    /// `IPv4v6`
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

/// Authorized `QoS` Rules IE (Type 6)
///
/// 3GPP TS 24.501 Section 9.11.4.13
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct IeQosRules {
    /// Raw `QoS` rules data
    pub data: Vec<u8>,
}

impl IeQosRules {
    /// Create a new `QoS` Rules IE with raw data
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
    /// `IPv4v6`
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
    /// Address data (4 bytes for IPv4, 8 bytes for IPv6 interface ID, 12 bytes for `IPv4v6`)
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

// ============================================================================
// IEI Constants for PDU Session Establishment Messages
// ============================================================================

/// IEI values for PDU Session Establishment Request optional IEs
/// (3GPP TS 24.501 Table 8.3.1.1.1)
mod establishment_request_iei {
    /// PDU session type (Type 1 TV, IEI in high nibble)
    pub const PDU_SESSION_TYPE_HIGH_NIBBLE: u8 = 0x9;
    /// SSC mode (Type 1 TV, IEI in high nibble)
    pub const SSC_MODE_HIGH_NIBBLE: u8 = 0xA;
    /// Always-on PDU session requested (Type 1 TV, IEI in high nibble)
    pub const ALWAYS_ON_REQUESTED_HIGH_NIBBLE: u8 = 0xB;
    /// 5GSM capability (TLV)
    pub const SM_CAPABILITY: u8 = 0x28;
    /// Maximum number of supported packet filters (TV, 3 octets)
    pub const MAX_PACKET_FILTERS: u8 = 0x55;
    /// SM PDU DN request container (TLV)
    pub const SM_PDU_DN_REQUEST_CONTAINER: u8 = 0x39;
    /// Extended protocol configuration options (TLV-E)
    pub const EXTENDED_PROTOCOL_CONFIG_OPTIONS: u8 = 0x7B;
}

/// IEI values for PDU Session Establishment Accept optional IEs
/// (3GPP TS 24.501 Table 8.3.2.1.1)
mod establishment_accept_iei {
    /// 5GSM cause (TV, 2 octets)
    pub const SM_CAUSE: u8 = 0x59;
    /// PDU address (TLV)
    pub const PDU_ADDRESS: u8 = 0x29;
    /// RQ timer value (TV, 2 octets, GPRS timer)
    pub const RQ_TIMER_VALUE: u8 = 0x56;
    /// S-NSSAI (TLV)
    pub const S_NSSAI: u8 = 0x22;
    /// Always-on PDU session indication (Type 1 TV, IEI in high nibble)
    pub const ALWAYS_ON_INDICATION_HIGH_NIBBLE: u8 = 0x8;
    /// Mapped EPS bearer contexts (TLV-E)
    pub const MAPPED_EPS_BEARER_CONTEXTS: u8 = 0x75;
    /// EAP message (TLV-E)
    pub const EAP_MESSAGE: u8 = 0x78;
    /// Authorized `QoS` flow descriptions (TLV-E)
    pub const AUTHORIZED_QOS_FLOW_DESCRIPTIONS: u8 = 0x79;
    /// Extended protocol configuration options (TLV-E)
    pub const EXTENDED_PROTOCOL_CONFIG_OPTIONS: u8 = 0x7B;
    /// DNN (TLV)
    pub const DNN: u8 = 0x25;
}

/// IEI values for PDU Session Establishment Reject optional IEs
/// (3GPP TS 24.501 Table 8.3.3.1.1)
mod establishment_reject_iei {
    /// Back-off timer value (TLV, GPRS timer 3)
    pub const BACK_OFF_TIMER_VALUE: u8 = 0x37;
    /// Allowed SSC mode (Type 1 TV, IEI in high nibble)
    pub const ALLOWED_SSC_MODE_HIGH_NIBBLE: u8 = 0xF;
    /// EAP message (TLV-E)
    pub const EAP_MESSAGE: u8 = 0x78;
    /// Re-attempt indicator (TLV)
    pub const RE_ATTEMPT_INDICATOR: u8 = 0x1D;
    /// 5GSM congestion re-attempt indicator (TLV)
    pub const CONGESTION_RE_ATTEMPT_INDICATOR: u8 = 0x61;
    /// Extended protocol configuration options (TLV-E)
    pub const EXTENDED_PROTOCOL_CONFIG_OPTIONS: u8 = 0x7B;
}

/// Skip an unknown optional IE: Type 1 TV IEIs known in 5GSM messages are
/// consumed by the callers; everything else is assumed TLV (1-octet length)
/// unless the IEI is in the TLV-E set.
fn skip_unknown_ie<B: Buf>(buf: &mut B, iei: u8) {
    buf.advance(1);
    // TLV-E IEs in 5GSM messages (TS 24.501 Section 9.11.4)
    let is_tlv_e = matches!(iei, 0x75 | 0x78 | 0x79 | 0x7A | 0x7B);
    if is_tlv_e {
        if buf.remaining() < 2 {
            return;
        }
        let len = buf.get_u16() as usize;
        if buf.remaining() >= len {
            buf.advance(len);
        }
    } else {
        if buf.remaining() < 1 {
            return;
        }
        let len = buf.get_u8() as usize;
        if buf.remaining() >= len {
            buf.advance(len);
        }
    }
}

// ============================================================================
// PDU Session Establishment Request (3GPP TS 24.501 Section 8.3.1)
// ============================================================================

/// PDU Session Establishment Request message (UE to network)
///
/// 3GPP TS 24.501 Section 8.3.1
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PduSessionEstablishmentRequest {
    /// PDU Session ID (from header)
    pub pdu_session_id: u8,
    /// Procedure Transaction Identity (from header)
    pub pti: u8,
    /// Integrity protection maximum data rate (mandatory, V, 2 octets)
    pub integrity_protection_max_data_rate: IeIntegrityProtectionMaxDataRate,
    /// Requested PDU session type (optional, Type 1 TV, IEI 0x9)
    pub pdu_session_type: Option<IeSelectedPduSessionType>,
    /// Requested SSC mode (optional, Type 1 TV, IEI 0xA)
    pub ssc_mode: Option<IeSelectedSscMode>,
    /// 5GSM capability (optional, TLV, IEI 0x28)
    pub sm_capability: Option<Vec<u8>>,
    /// Maximum number of supported packet filters (optional, TV, IEI 0x55)
    pub max_packet_filters: Option<u16>,
    /// Always-on PDU session requested (optional, Type 1 TV, IEI 0xB)
    pub always_on_requested: Option<bool>,
    /// SM PDU DN request container (optional, TLV, IEI 0x39)
    pub sm_pdu_dn_request_container: Option<Vec<u8>>,
    /// Extended protocol configuration options (optional, TLV-E, IEI 0x7B)
    pub extended_protocol_config_options: Option<Vec<u8>>,
}

impl PduSessionEstablishmentRequest {
    /// Create a new PDU Session Establishment Request
    pub fn new(
        pdu_session_id: u8,
        pti: u8,
        integrity_protection_max_data_rate: IeIntegrityProtectionMaxDataRate,
    ) -> Self {
        Self {
            pdu_session_id,
            pti,
            integrity_protection_max_data_rate,
            ..Default::default()
        }
    }

    /// Decode from bytes (after header has been parsed)
    pub fn decode<B: Buf>(
        buf: &mut B,
        pdu_session_id: u8,
        pti: u8,
    ) -> Result<Self, PduSessionEstablishmentError> {
        // Integrity protection maximum data rate (mandatory, V, 2 octets:
        // octet 1 = uplink, octet 2 = downlink per TS 24.501 9.11.4.7)
        if buf.remaining() < 2 {
            return Err(PduSessionEstablishmentError::MissingMandatoryIe(
                "Integrity protection maximum data rate",
            ));
        }
        let uplink = MaxDataRate::try_from(buf.get_u8())?;
        let downlink = MaxDataRate::try_from(buf.get_u8())?;

        let mut msg = Self::new(
            pdu_session_id,
            pti,
            IeIntegrityProtectionMaxDataRate::new(uplink, downlink),
        );

        // Optional IEs
        while buf.remaining() > 0 {
            let iei = buf.chunk()[0];
            match (iei >> 4) & 0x0F {
                establishment_request_iei::PDU_SESSION_TYPE_HIGH_NIBBLE => {
                    buf.advance(1);
                    msg.pdu_session_type = Some(IeSelectedPduSessionType::decode(iei & 0x0F)?);
                }
                establishment_request_iei::SSC_MODE_HIGH_NIBBLE => {
                    buf.advance(1);
                    msg.ssc_mode = Some(IeSelectedSscMode::decode(iei & 0x0F)?);
                }
                establishment_request_iei::ALWAYS_ON_REQUESTED_HIGH_NIBBLE => {
                    buf.advance(1);
                    msg.always_on_requested = Some(iei & 0x01 == 0x01);
                }
                _ => match iei {
                    establishment_request_iei::SM_CAPABILITY => {
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
                        msg.sm_capability = Some(data);
                    }
                    establishment_request_iei::MAX_PACKET_FILTERS => {
                        buf.advance(1);
                        if buf.remaining() < 2 {
                            break;
                        }
                        let high = buf.get_u8() as u16;
                        let low = buf.get_u8() as u16;
                        msg.max_packet_filters = Some((high << 3) | ((low >> 5) & 0x07));
                    }
                    establishment_request_iei::SM_PDU_DN_REQUEST_CONTAINER => {
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
                        msg.sm_pdu_dn_request_container = Some(data);
                    }
                    establishment_request_iei::EXTENDED_PROTOCOL_CONFIG_OPTIONS => {
                        buf.advance(1);
                        if buf.remaining() < 2 {
                            break;
                        }
                        let len = buf.get_u16() as usize;
                        if buf.remaining() < len {
                            break;
                        }
                        let mut data = vec![0u8; len];
                        buf.copy_to_slice(&mut data);
                        msg.extended_protocol_config_options = Some(data);
                    }
                    _ => skip_unknown_ie(buf, iei),
                },
            }
        }

        Ok(msg)
    }

    /// Encode to bytes (including header)
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        let header = PlainSmHeader::new(
            self.pdu_session_id,
            self.pti,
            SmMessageType::PduSessionEstablishmentRequest,
        );
        header.encode(buf);

        // Integrity protection maximum data rate (mandatory, V):
        // octet 1 = uplink, octet 2 = downlink (TS 24.501 9.11.4.7)
        buf.put_u8(self.integrity_protection_max_data_rate.uplink as u8);
        buf.put_u8(self.integrity_protection_max_data_rate.downlink as u8);

        // Optional IEs
        if let Some(ref t) = self.pdu_session_type {
            buf.put_u8((establishment_request_iei::PDU_SESSION_TYPE_HIGH_NIBBLE << 4) | t.encode());
        }
        if let Some(ref m) = self.ssc_mode {
            buf.put_u8((establishment_request_iei::SSC_MODE_HIGH_NIBBLE << 4) | m.encode());
        }
        if let Some(ref cap) = self.sm_capability {
            buf.put_u8(establishment_request_iei::SM_CAPABILITY);
            buf.put_u8(cap.len() as u8);
            buf.put_slice(cap);
        }
        if let Some(max_filters) = self.max_packet_filters {
            buf.put_u8(establishment_request_iei::MAX_PACKET_FILTERS);
            buf.put_u8(((max_filters >> 3) & 0xFF) as u8);
            buf.put_u8(((max_filters & 0x07) << 5) as u8);
        }
        if let Some(requested) = self.always_on_requested {
            buf.put_u8(
                (establishment_request_iei::ALWAYS_ON_REQUESTED_HIGH_NIBBLE << 4)
                    | u8::from(requested),
            );
        }
        if let Some(ref dn_req) = self.sm_pdu_dn_request_container {
            buf.put_u8(establishment_request_iei::SM_PDU_DN_REQUEST_CONTAINER);
            buf.put_u8(dn_req.len() as u8);
            buf.put_slice(dn_req);
        }
        if let Some(ref epco) = self.extended_protocol_config_options {
            buf.put_u8(establishment_request_iei::EXTENDED_PROTOCOL_CONFIG_OPTIONS);
            buf.put_u16(epco.len() as u16);
            buf.put_slice(epco);
        }
    }

    /// Get the message type
    pub fn message_type() -> SmMessageType {
        SmMessageType::PduSessionEstablishmentRequest
    }
}

// ============================================================================
// PDU Session Establishment Accept (3GPP TS 24.501 Section 8.3.2)
// ============================================================================

/// PDU Session Establishment Accept message (network to UE)
///
/// The decoder is strict per TS 24.501 Table 8.3.2.1.1: the mandatory
/// selected PDU session type, selected SSC mode, authorized QoS rules and
/// session AMBR must all be present and valid, otherwise decoding fails
/// with [`PduSessionEstablishmentError::MissingMandatoryIe`].
///
/// 3GPP TS 24.501 Section 8.3.2
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PduSessionEstablishmentAccept {
    /// PDU Session ID (from header)
    pub pdu_session_id: u8,
    /// Procedure Transaction Identity (from header)
    pub pti: u8,
    /// Selected PDU session type (mandatory, bits 1-3 of octet 5)
    pub selected_pdu_session_type: IeSelectedPduSessionType,
    /// Selected SSC mode (mandatory, bits 5-7 of octet 5)
    pub selected_ssc_mode: IeSelectedSscMode,
    /// Authorized `QoS` rules (mandatory, LV-E)
    pub authorized_qos_rules: IeQosRules,
    /// Session AMBR (mandatory, LV)
    pub session_ambr: IeSessionAmbr,
    /// 5GSM cause (optional, TV, IEI 0x59)
    pub sm_cause: Option<Ie5gSmCause>,
    /// PDU address (optional, TLV, IEI 0x29)
    pub pdu_address: Option<IePduAddress>,
    /// RQ timer value (optional, TV, IEI 0x56)
    pub rq_timer_value: Option<u8>,
    /// S-NSSAI (optional, TLV, IEI 0x22)
    pub s_nssai: Option<Vec<u8>>,
    /// Always-on PDU session indication (optional, Type 1 TV, IEI 0x8)
    pub always_on_indication: Option<bool>,
    /// Mapped EPS bearer contexts (optional, TLV-E, IEI 0x75)
    pub mapped_eps_bearer_contexts: Option<Vec<u8>>,
    /// EAP message (optional, TLV-E, IEI 0x78)
    pub eap_message: Option<Vec<u8>>,
    /// Authorized `QoS` flow descriptions (optional, TLV-E, IEI 0x79)
    pub authorized_qos_flow_descriptions: Option<Vec<u8>>,
    /// Extended protocol configuration options (optional, TLV-E, IEI 0x7B)
    pub extended_protocol_config_options: Option<Vec<u8>>,
    /// DNN (optional, TLV, IEI 0x25)
    pub dnn: Option<IeDnn>,
}

impl PduSessionEstablishmentAccept {
    /// Create a new PDU Session Establishment Accept with the mandatory IEs
    pub fn new(
        pdu_session_id: u8,
        pti: u8,
        selected_pdu_session_type: IeSelectedPduSessionType,
        selected_ssc_mode: IeSelectedSscMode,
        authorized_qos_rules: IeQosRules,
        session_ambr: IeSessionAmbr,
    ) -> Self {
        Self {
            pdu_session_id,
            pti,
            selected_pdu_session_type,
            selected_ssc_mode,
            authorized_qos_rules,
            session_ambr,
            ..Default::default()
        }
    }

    /// Strict decode per TS 24.501 Table 8.3.2.1.1 (after the SM header).
    ///
    /// Fails with [`PduSessionEstablishmentError::MissingMandatoryIe`] when
    /// any mandatory IE (selected PDU session type, selected SSC mode,
    /// authorized QoS rules, session AMBR) is absent or malformed.
    pub fn decode<B: Buf>(
        buf: &mut B,
        pdu_session_id: u8,
        pti: u8,
    ) -> Result<Self, PduSessionEstablishmentError> {
        // Octet 5: selected PDU session type (bits 1-3) and selected SSC
        // mode (bits 5-7) share one octet
        if buf.remaining() < 1 {
            return Err(PduSessionEstablishmentError::MissingMandatoryIe(
                "Selected PDU session type / SSC mode",
            ));
        }
        let octet5 = buf.get_u8();
        let selected_pdu_session_type =
            IeSelectedPduSessionType::decode(octet5 & 0x0F).map_err(|_| {
                PduSessionEstablishmentError::MissingMandatoryIe("Selected PDU session type")
            })?;
        let selected_ssc_mode = IeSelectedSscMode::decode((octet5 >> 4) & 0x0F)
            .map_err(|_| PduSessionEstablishmentError::MissingMandatoryIe("Selected SSC mode"))?;

        // Authorized QoS rules (mandatory, LV-E)
        let authorized_qos_rules = IeQosRules::decode(buf).map_err(|_| {
            PduSessionEstablishmentError::MissingMandatoryIe("Authorized QoS rules")
        })?;
        if authorized_qos_rules.data.is_empty() {
            return Err(PduSessionEstablishmentError::MissingMandatoryIe(
                "Authorized QoS rules",
            ));
        }

        // Session AMBR (mandatory, LV, length must be 6 per TS 24.501
        // 9.11.4.14)
        if buf.remaining() < 7 {
            return Err(PduSessionEstablishmentError::MissingMandatoryIe(
                "Session AMBR",
            ));
        }
        let ambr_len = buf.get_u8();
        if ambr_len != 6 {
            return Err(PduSessionEstablishmentError::MissingMandatoryIe(
                "Session AMBR",
            ));
        }
        let session_ambr = IeSessionAmbr {
            downlink_unit: buf.get_u8(),
            downlink: buf.get_u16(),
            uplink_unit: buf.get_u8(),
            uplink: buf.get_u16(),
        };

        let mut msg = Self::new(
            pdu_session_id,
            pti,
            selected_pdu_session_type,
            selected_ssc_mode,
            authorized_qos_rules,
            session_ambr,
        );

        // Optional IEs
        while buf.remaining() > 0 {
            let iei = buf.chunk()[0];
            if (iei >> 4) & 0x0F == establishment_accept_iei::ALWAYS_ON_INDICATION_HIGH_NIBBLE {
                buf.advance(1);
                msg.always_on_indication = Some(iei & 0x01 == 0x01);
                continue;
            }
            match iei {
                establishment_accept_iei::SM_CAUSE => {
                    buf.advance(1);
                    if buf.remaining() < 1 {
                        break;
                    }
                    msg.sm_cause = Some(Ie5gSmCause::new(SmCause::try_from(buf.get_u8())?));
                }
                establishment_accept_iei::PDU_ADDRESS => {
                    buf.advance(1);
                    msg.pdu_address = Some(IePduAddress::decode(buf)?);
                }
                establishment_accept_iei::RQ_TIMER_VALUE => {
                    buf.advance(1);
                    if buf.remaining() < 1 {
                        break;
                    }
                    msg.rq_timer_value = Some(buf.get_u8());
                }
                establishment_accept_iei::S_NSSAI => {
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
                    msg.s_nssai = Some(data);
                }
                establishment_accept_iei::MAPPED_EPS_BEARER_CONTEXTS
                | establishment_accept_iei::EAP_MESSAGE
                | establishment_accept_iei::AUTHORIZED_QOS_FLOW_DESCRIPTIONS
                | establishment_accept_iei::EXTENDED_PROTOCOL_CONFIG_OPTIONS => {
                    buf.advance(1);
                    if buf.remaining() < 2 {
                        break;
                    }
                    let len = buf.get_u16() as usize;
                    if buf.remaining() < len {
                        break;
                    }
                    let mut data = vec![0u8; len];
                    buf.copy_to_slice(&mut data);
                    match iei {
                        establishment_accept_iei::MAPPED_EPS_BEARER_CONTEXTS => {
                            msg.mapped_eps_bearer_contexts = Some(data);
                        }
                        establishment_accept_iei::EAP_MESSAGE => msg.eap_message = Some(data),
                        establishment_accept_iei::AUTHORIZED_QOS_FLOW_DESCRIPTIONS => {
                            msg.authorized_qos_flow_descriptions = Some(data);
                        }
                        _ => msg.extended_protocol_config_options = Some(data),
                    }
                }
                establishment_accept_iei::DNN => {
                    buf.advance(1);
                    msg.dnn = Some(IeDnn::decode(buf)?);
                }
                _ => skip_unknown_ie(buf, iei),
            }
        }

        Ok(msg)
    }

    /// Encode to bytes (including header)
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        let header = PlainSmHeader::new(
            self.pdu_session_id,
            self.pti,
            SmMessageType::PduSessionEstablishmentAccept,
        );
        header.encode(buf);

        // Octet 5: SSC mode (bits 5-7) | PDU session type (bits 1-3)
        buf.put_u8(
            (self.selected_ssc_mode.encode() << 4) | self.selected_pdu_session_type.encode(),
        );

        // Authorized QoS rules (mandatory, LV-E)
        self.authorized_qos_rules.encode(buf);

        // Session AMBR (mandatory, LV)
        self.session_ambr.encode(buf);

        // Optional IEs
        if let Some(ref cause) = self.sm_cause {
            buf.put_u8(establishment_accept_iei::SM_CAUSE);
            buf.put_u8(cause.value as u8);
        }
        if let Some(ref addr) = self.pdu_address {
            buf.put_u8(establishment_accept_iei::PDU_ADDRESS);
            addr.encode(buf);
        }
        if let Some(timer) = self.rq_timer_value {
            buf.put_u8(establishment_accept_iei::RQ_TIMER_VALUE);
            buf.put_u8(timer);
        }
        if let Some(ref nssai) = self.s_nssai {
            buf.put_u8(establishment_accept_iei::S_NSSAI);
            buf.put_u8(nssai.len() as u8);
            buf.put_slice(nssai);
        }
        if let Some(indication) = self.always_on_indication {
            buf.put_u8(
                (establishment_accept_iei::ALWAYS_ON_INDICATION_HIGH_NIBBLE << 4)
                    | u8::from(indication),
            );
        }
        if let Some(ref ctxs) = self.mapped_eps_bearer_contexts {
            buf.put_u8(establishment_accept_iei::MAPPED_EPS_BEARER_CONTEXTS);
            buf.put_u16(ctxs.len() as u16);
            buf.put_slice(ctxs);
        }
        if let Some(ref eap) = self.eap_message {
            buf.put_u8(establishment_accept_iei::EAP_MESSAGE);
            buf.put_u16(eap.len() as u16);
            buf.put_slice(eap);
        }
        if let Some(ref desc) = self.authorized_qos_flow_descriptions {
            buf.put_u8(establishment_accept_iei::AUTHORIZED_QOS_FLOW_DESCRIPTIONS);
            buf.put_u16(desc.len() as u16);
            buf.put_slice(desc);
        }
        if let Some(ref epco) = self.extended_protocol_config_options {
            buf.put_u8(establishment_accept_iei::EXTENDED_PROTOCOL_CONFIG_OPTIONS);
            buf.put_u16(epco.len() as u16);
            buf.put_slice(epco);
        }
        if let Some(ref dnn) = self.dnn {
            buf.put_u8(establishment_accept_iei::DNN);
            dnn.encode(buf);
        }
    }

    /// Get the message type
    pub fn message_type() -> SmMessageType {
        SmMessageType::PduSessionEstablishmentAccept
    }
}

// ============================================================================
// PDU Session Establishment Reject (3GPP TS 24.501 Section 8.3.3)
// ============================================================================

/// PDU Session Establishment Reject message (network to UE)
///
/// 3GPP TS 24.501 Section 8.3.3
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PduSessionEstablishmentReject {
    /// PDU Session ID (from header)
    pub pdu_session_id: u8,
    /// Procedure Transaction Identity (from header)
    pub pti: u8,
    /// 5GSM cause (mandatory, V)
    pub sm_cause: Ie5gSmCause,
    /// Back-off timer value (optional, TLV, IEI 0x37, GPRS timer 3)
    pub back_off_timer_value: Option<u8>,
    /// Allowed SSC mode (optional, Type 1 TV, IEI 0xF)
    pub allowed_ssc_mode: Option<u8>,
    /// EAP message (optional, TLV-E, IEI 0x78)
    pub eap_message: Option<Vec<u8>>,
    /// Re-attempt indicator (optional, TLV, IEI 0x1D)
    pub re_attempt_indicator: Option<u8>,
    /// 5GSM congestion re-attempt indicator (optional, TLV, IEI 0x61)
    pub congestion_re_attempt_indicator: Option<u8>,
    /// Extended protocol configuration options (optional, TLV-E, IEI 0x7B)
    pub extended_protocol_config_options: Option<Vec<u8>>,
}

impl Default for PduSessionEstablishmentReject {
    fn default() -> Self {
        Self {
            pdu_session_id: 0,
            pti: 0,
            sm_cause: Ie5gSmCause::new(SmCause::ProtocolErrorUnspecified),
            back_off_timer_value: None,
            allowed_ssc_mode: None,
            eap_message: None,
            re_attempt_indicator: None,
            congestion_re_attempt_indicator: None,
            extended_protocol_config_options: None,
        }
    }
}

impl PduSessionEstablishmentReject {
    /// Create a new PDU Session Establishment Reject
    pub fn new(pdu_session_id: u8, pti: u8, cause: SmCause) -> Self {
        Self {
            pdu_session_id,
            pti,
            sm_cause: Ie5gSmCause::new(cause),
            ..Default::default()
        }
    }

    /// Decode from bytes (after header has been parsed)
    pub fn decode<B: Buf>(
        buf: &mut B,
        pdu_session_id: u8,
        pti: u8,
    ) -> Result<Self, PduSessionEstablishmentError> {
        // 5GSM cause (mandatory, V)
        if buf.remaining() < 1 {
            return Err(PduSessionEstablishmentError::MissingMandatoryIe(
                "5GSM cause",
            ));
        }
        let cause = SmCause::try_from(buf.get_u8())?;
        let mut msg = Self::new(pdu_session_id, pti, cause);

        // Optional IEs
        while buf.remaining() > 0 {
            let iei = buf.chunk()[0];
            if (iei >> 4) & 0x0F == establishment_reject_iei::ALLOWED_SSC_MODE_HIGH_NIBBLE {
                buf.advance(1);
                msg.allowed_ssc_mode = Some(iei & 0x07);
                continue;
            }
            match iei {
                establishment_reject_iei::BACK_OFF_TIMER_VALUE => {
                    buf.advance(1);
                    if buf.remaining() < 1 {
                        break;
                    }
                    let len = buf.get_u8() as usize;
                    if buf.remaining() < len || len < 1 {
                        break;
                    }
                    msg.back_off_timer_value = Some(buf.get_u8());
                    if len > 1 {
                        buf.advance(len - 1);
                    }
                }
                establishment_reject_iei::RE_ATTEMPT_INDICATOR => {
                    buf.advance(1);
                    if buf.remaining() < 1 {
                        break;
                    }
                    let len = buf.get_u8() as usize;
                    if buf.remaining() < len || len < 1 {
                        break;
                    }
                    msg.re_attempt_indicator = Some(buf.get_u8());
                    if len > 1 {
                        buf.advance(len - 1);
                    }
                }
                establishment_reject_iei::CONGESTION_RE_ATTEMPT_INDICATOR => {
                    buf.advance(1);
                    if buf.remaining() < 1 {
                        break;
                    }
                    let len = buf.get_u8() as usize;
                    if buf.remaining() < len || len < 1 {
                        break;
                    }
                    msg.congestion_re_attempt_indicator = Some(buf.get_u8());
                    if len > 1 {
                        buf.advance(len - 1);
                    }
                }
                establishment_reject_iei::EAP_MESSAGE
                | establishment_reject_iei::EXTENDED_PROTOCOL_CONFIG_OPTIONS => {
                    buf.advance(1);
                    if buf.remaining() < 2 {
                        break;
                    }
                    let len = buf.get_u16() as usize;
                    if buf.remaining() < len {
                        break;
                    }
                    let mut data = vec![0u8; len];
                    buf.copy_to_slice(&mut data);
                    if iei == establishment_reject_iei::EAP_MESSAGE {
                        msg.eap_message = Some(data);
                    } else {
                        msg.extended_protocol_config_options = Some(data);
                    }
                }
                _ => skip_unknown_ie(buf, iei),
            }
        }

        Ok(msg)
    }

    /// Encode to bytes (including header)
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        let header = PlainSmHeader::new(
            self.pdu_session_id,
            self.pti,
            SmMessageType::PduSessionEstablishmentReject,
        );
        header.encode(buf);

        // 5GSM cause (mandatory)
        buf.put_u8(self.sm_cause.value as u8);

        // Optional IEs
        if let Some(timer) = self.back_off_timer_value {
            buf.put_u8(establishment_reject_iei::BACK_OFF_TIMER_VALUE);
            buf.put_u8(1);
            buf.put_u8(timer);
        }
        if let Some(ssc) = self.allowed_ssc_mode {
            buf.put_u8(
                (establishment_reject_iei::ALLOWED_SSC_MODE_HIGH_NIBBLE << 4) | (ssc & 0x07),
            );
        }
        if let Some(ref eap) = self.eap_message {
            buf.put_u8(establishment_reject_iei::EAP_MESSAGE);
            buf.put_u16(eap.len() as u16);
            buf.put_slice(eap);
        }
        if let Some(rai) = self.re_attempt_indicator {
            buf.put_u8(establishment_reject_iei::RE_ATTEMPT_INDICATOR);
            buf.put_u8(1);
            buf.put_u8(rai);
        }
        if let Some(cri) = self.congestion_re_attempt_indicator {
            buf.put_u8(establishment_reject_iei::CONGESTION_RE_ATTEMPT_INDICATOR);
            buf.put_u8(1);
            buf.put_u8(cri);
        }
        if let Some(ref epco) = self.extended_protocol_config_options {
            buf.put_u8(establishment_reject_iei::EXTENDED_PROTOCOL_CONFIG_OPTIONS);
            buf.put_u16(epco.len() as u16);
            buf.put_slice(epco);
        }
    }

    /// Get the message type
    pub fn message_type() -> SmMessageType {
        SmMessageType::PduSessionEstablishmentReject
    }

    /// Get the cause value
    pub fn cause(&self) -> SmCause {
        self.sm_cause.value
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // PDU Session Establishment Request Tests
    // ========================================================================

    #[test]
    fn test_establishment_request_encode_decode_roundtrip() {
        let mut msg = PduSessionEstablishmentRequest::new(
            1,
            1,
            IeIntegrityProtectionMaxDataRate::full_rate(),
        );
        msg.pdu_session_type = Some(IeSelectedPduSessionType::new(PduSessionTypeValue::Ipv4v6));
        msg.ssc_mode = Some(IeSelectedSscMode::new(SscModeValue::SscMode1));
        msg.sm_capability = Some(vec![0x00]);
        msg.max_packet_filters = Some(16);

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Header: EPD 0x2E, PSI 1, PTI 1, type 0xC1
        assert_eq!(&buf[..4], &[0x2E, 1, 1, 0xC1]);
        // Mandatory integrity protection maximum data rate (UL, DL)
        assert_eq!(&buf[4..6], &[0xFF, 0xFF]);

        let decoded = PduSessionEstablishmentRequest::decode(&mut &buf[4..], 1, 1).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn test_establishment_request_missing_mandatory_ie() {
        // No integrity protection maximum data rate at all
        let buf: &[u8] = &[0xFF];
        let result = PduSessionEstablishmentRequest::decode(&mut &buf[..], 1, 1);
        assert_eq!(
            result,
            Err(PduSessionEstablishmentError::MissingMandatoryIe(
                "Integrity protection maximum data rate"
            ))
        );
    }

    #[test]
    fn test_establishment_request_parsable_by_strict_core_shape() {
        // The nextgcore smfd parser reads buf[4..6] as the data rate and
        // half-octet IEIs 0x9 / 0xA for PDU type and SSC mode: verify our
        // wire layout puts them exactly there.
        let mut msg = PduSessionEstablishmentRequest::new(
            5,
            7,
            IeIntegrityProtectionMaxDataRate::full_rate(),
        );
        msg.pdu_session_type = Some(IeSelectedPduSessionType::new(PduSessionTypeValue::Ipv4));
        msg.ssc_mode = Some(IeSelectedSscMode::new(SscModeValue::SscMode1));

        let mut buf = Vec::new();
        msg.encode(&mut buf);
        assert_eq!(buf[4], 0xFF); // UL rate
        assert_eq!(buf[5], 0xFF); // DL rate
        assert_eq!(buf[6], 0x91); // PDU session type TV: IEI 0x9, IPv4
        assert_eq!(buf[7], 0xA1); // SSC mode TV: IEI 0xA, SSC mode 1
    }

    // ========================================================================
    // PDU Session Establishment Accept Tests
    // ========================================================================

    fn conformant_accept() -> PduSessionEstablishmentAccept {
        let mut acc = PduSessionEstablishmentAccept::new(
            1,
            1,
            IeSelectedPduSessionType::new(PduSessionTypeValue::Ipv4),
            IeSelectedSscMode::new(SscModeValue::SscMode1),
            IeQosRules::new(vec![0x01, 0x00, 0x06, 0x31, 0x31, 0x01, 0x01, 0x01, 0x01]),
            IeSessionAmbr::new(0x06, 200, 0x06, 50),
        );
        acc.pdu_address = Some(IePduAddress::ipv4([10, 45, 0, 2]));
        acc.s_nssai = Some(vec![0x01]);
        acc.dnn = Some(IeDnn::from_string("internet"));
        acc
    }

    #[test]
    fn test_establishment_accept_encode_decode_roundtrip() {
        let acc = conformant_accept();
        let mut buf = Vec::new();
        acc.encode(&mut buf);

        assert_eq!(&buf[..4], &[0x2E, 1, 1, 0xC2]);
        // Octet 5: SSC mode 1 in bits 5-7, PDU type IPv4 in bits 1-3
        assert_eq!(buf[4], 0x11);

        let decoded = PduSessionEstablishmentAccept::decode(&mut &buf[4..], 1, 1).unwrap();
        assert_eq!(decoded, acc);
    }

    #[test]
    fn test_establishment_accept_missing_session_ambr_rejected() {
        // Encode a conformant accept, then truncate just before the AMBR
        let acc = conformant_accept();
        let mut buf = Vec::new();
        acc.encode(&mut buf);
        // header(4) + octet5(1) + qos LV-E (2 + 9) = 16; AMBR starts at 16
        let truncated = &buf[4..16];
        let result = PduSessionEstablishmentAccept::decode(&mut &truncated[..], 1, 1);
        assert_eq!(
            result,
            Err(PduSessionEstablishmentError::MissingMandatoryIe(
                "Session AMBR"
            ))
        );
    }

    #[test]
    fn test_establishment_accept_missing_qos_rules_rejected() {
        // Only octet 5 present
        let buf: &[u8] = &[0x11];
        let result = PduSessionEstablishmentAccept::decode(&mut &buf[..], 1, 1);
        assert_eq!(
            result,
            Err(PduSessionEstablishmentError::MissingMandatoryIe(
                "Authorized QoS rules"
            ))
        );
    }

    #[test]
    fn test_establishment_accept_empty_qos_rules_rejected() {
        // Octet 5 + zero-length QoS rules LV-E + valid AMBR
        let buf: &[u8] = &[0x11, 0x00, 0x00, 0x06, 0x06, 0x00, 0xC8, 0x06, 0x00, 0x32];
        let result = PduSessionEstablishmentAccept::decode(&mut &buf[..], 1, 1);
        assert_eq!(
            result,
            Err(PduSessionEstablishmentError::MissingMandatoryIe(
                "Authorized QoS rules"
            ))
        );
    }

    #[test]
    fn test_establishment_accept_invalid_ssc_mode_rejected() {
        // SSC mode 0 (reserved) in the high nibble of octet 5
        let buf: &[u8] = &[
            0x01, 0x00, 0x01, 0xAA, 0x06, 0x06, 0x00, 0xC8, 0x06, 0x00, 0x32,
        ];
        let result = PduSessionEstablishmentAccept::decode(&mut &buf[..], 1, 1);
        assert_eq!(
            result,
            Err(PduSessionEstablishmentError::MissingMandatoryIe(
                "Selected SSC mode"
            ))
        );
    }

    #[test]
    fn test_establishment_accept_wrong_ambr_length_rejected() {
        // AMBR length 7 instead of 6 (the legacy nextgcore double-length bug)
        let buf: &[u8] = &[
            0x11, 0x00, 0x01, 0xAA, 0x07, 0x06, 0x06, 0x00, 0xC8, 0x06, 0x00, 0x32,
        ];
        let result = PduSessionEstablishmentAccept::decode(&mut &buf[..], 1, 1);
        assert_eq!(
            result,
            Err(PduSessionEstablishmentError::MissingMandatoryIe(
                "Session AMBR"
            ))
        );
    }

    /// External vector: the exact byte construction of nextgcore smfd's
    /// `policy::build_establishment_accept(5, 3, IPV4, 1, 9, 200 Mbps,
    /// 50 Mbps, 10.45.0.2, "internet")` replicated byte-for-byte.
    fn nextgcore_legacy_accept_bytes() -> Vec<u8> {
        let mut msg = vec![0x2E, 5, 3, 0xC2];
        msg.push(0x01); // selected PDU session type (IPv4), SSC bits zero
        msg.push(0x01); // selected SSC mode as a SEPARATE octet (non-spec)
        msg.extend_from_slice(&[0x06, 0x01, 0x03, 0x01, 0x01, 0x09]); // "QoS rules"
        msg.push(0x06); // session AMBR length
        msg.extend_from_slice(&[0x06, 0x00, 0xC8]); // DL: unit 1 Mbps, 200
        msg.extend_from_slice(&[0x06, 0x00, 0x32]); // UL: unit 1 Mbps, 50
        msg.extend_from_slice(&[0x29, 0x05, 0x01, 10, 45, 0, 2]); // PDU address
        msg.extend_from_slice(&[0x25, 0x09, 0x08]); // DNN TLV
        msg.extend_from_slice(b"internet");
        msg
    }

    #[test]
    fn test_strict_decode_rejects_current_nextgcore_emission() {
        let bytes = nextgcore_legacy_accept_bytes();
        // The strict parser misreads the legacy layout (separate SSC octet
        // shifts the QoS rules LV-E length) and must fail.
        let result = PduSessionEstablishmentAccept::decode(&mut &bytes[4..], 5, 3);
        assert!(
            result.is_err(),
            "strict parser must reject the legacy core shape"
        );
    }

    /// Cross-stack guard: the EXACT octet layout the conformant nextgcore
    /// smfd `policy::build_establishment_accept(5, 3, IPV4, ssc2, 9,
    /// 200 Mbps, 50 Mbps, 10.45.0.2, "internet")` now emits per TS 24.501
    /// Table 8.3.2.1.1, replicated byte-for-byte. Keeping this in lock-step
    /// with the core builder proves both stacks agree on the wire format.
    fn nextgcore_conformant_accept_bytes() -> Vec<u8> {
        let mut msg = vec![0x2E, 5, 3, 0xC2];
        // Octet 5: SSC mode (high nibble) | PDU type (low nibble).
        // SSC mode 2 (0b010) << 4 | IPv4 (0b001) = 0x21
        msg.push(0x21);
        // Authorized QoS rules as LV-E (2-octet length). One default CREATE
        // rule with a match-all packet filter, precedence 255, QFI 9:
        //   [qfi=9][rule_len_hi=0][rule_len_lo=6][op+flags=0x31]
        //   [pf_hdr=0x31][pf_len=1][match-all=0x01][prec=0xFF][qfi=9]
        let qos_rule = [0x09u8, 0x00, 0x06, 0x31, 0x31, 0x01, 0x01, 0xFF, 0x09];
        msg.extend_from_slice(&(qos_rule.len() as u16).to_be_bytes());
        msg.extend_from_slice(&qos_rule);
        // Session-AMBR LV (1-octet length = 6).
        msg.push(0x06);
        msg.extend_from_slice(&[0x06, 0x00, 0xC8]); // DL: unit 1 Mbps, 200
        msg.extend_from_slice(&[0x06, 0x00, 0x32]); // UL: unit 1 Mbps, 50
        msg.extend_from_slice(&[0x29, 0x05, 0x01, 10, 45, 0, 2]); // PDU address
        msg.extend_from_slice(&[0x25, 0x09, 0x08]); // DNN TLV
        msg.extend_from_slice(b"internet");
        msg
    }

    #[test]
    fn test_strict_decode_accepts_conformant_nextgcore_emission() {
        let bytes = nextgcore_conformant_accept_bytes();
        // Decode via the strict (non-legacy) parser, after the SM header.
        let acc = PduSessionEstablishmentAccept::decode(&mut &bytes[4..], 5, 3)
            .expect("strict parser must accept the conformant core shape");

        assert_eq!(acc.pdu_session_id, 5);
        assert_eq!(acc.pti, 3);
        // SSC mode and PDU session type both decode from the packed octet 5.
        assert_eq!(
            acc.selected_pdu_session_type.value,
            PduSessionTypeValue::Ipv4
        );
        assert_eq!(acc.selected_ssc_mode.value, SscModeValue::SscMode2);

        // Authorized QoS rule decoded from the LV-E body.
        assert_eq!(
            acc.authorized_qos_rules.data,
            vec![0x09, 0x00, 0x06, 0x31, 0x31, 0x01, 0x01, 0xFF, 0x09]
        );
        // QoS rule identifier and the trailing QFI are both 9.
        assert_eq!(acc.authorized_qos_rules.data.first(), Some(&0x09));
        assert_eq!(acc.authorized_qos_rules.data.last(), Some(&0x09));

        // Session-AMBR decoded from the single-octet-length LV (no double
        // length): DL 200 / UL 50, both at unit 0x06 (1 Mbps).
        assert_eq!(acc.session_ambr.downlink_unit, 0x06);
        assert_eq!(acc.session_ambr.downlink, 200);
        assert_eq!(acc.session_ambr.uplink_unit, 0x06);
        assert_eq!(acc.session_ambr.uplink, 50);

        // Optional IEs still parse.
        let addr = acc.pdu_address.unwrap();
        assert_eq!(addr.address_type, PduAddressType::Ipv4);
        assert_eq!(addr.address, vec![10, 45, 0, 2]);
        assert_eq!(acc.dnn.unwrap().value, {
            let mut v = vec![8u8];
            v.extend_from_slice(b"internet");
            v
        });
    }

    // ========================================================================
    // PDU Session Establishment Reject Tests
    // ========================================================================

    #[test]
    fn test_establishment_reject_encode_decode_roundtrip() {
        let mut rej =
            PduSessionEstablishmentReject::new(1, 2, SmCause::InsufficientResourcesForSlice);
        rej.back_off_timer_value = Some(0x21); // GPRS timer 3
        rej.allowed_ssc_mode = Some(0x02);
        rej.re_attempt_indicator = Some(0x01);

        let mut buf = Vec::new();
        rej.encode(&mut buf);
        assert_eq!(&buf[..4], &[0x2E, 1, 2, 0xC3]);
        assert_eq!(buf[4], 69);

        let decoded = PduSessionEstablishmentReject::decode(&mut &buf[4..], 1, 2).unwrap();
        assert_eq!(decoded, rej);
    }

    /// External vector: nextgcore smfd `policy::build_establishment_reject`
    /// emits exactly `[0x2E, PSI, PTI, 0xC3, cause]`.
    #[test]
    fn test_establishment_reject_decodes_nextgcore_emission() {
        let bytes = [0x2Eu8, 1, 2, 0xC3, 29];
        let rej = PduSessionEstablishmentReject::decode(&mut &bytes[4..], 1, 2).unwrap();
        assert_eq!(rej.cause(), SmCause::UserAuthenticationFailed);
        assert!(rej.back_off_timer_value.is_none());
    }

    #[test]
    fn test_establishment_reject_missing_cause() {
        let buf: &[u8] = &[];
        let result = PduSessionEstablishmentReject::decode(&mut &buf[..], 1, 1);
        assert_eq!(
            result,
            Err(PduSessionEstablishmentError::MissingMandatoryIe(
                "5GSM cause"
            ))
        );
    }

    #[test]
    fn test_sm_cause_slice_values_roundtrip() {
        for (raw, cause) in [
            (26u8, SmCause::InsufficientResources),
            (28, SmCause::UnknownPduSessionType),
            (50, SmCause::PduSessionTypeIpv4OnlyAllowed),
            (51, SmCause::PduSessionTypeIpv6OnlyAllowed),
            (57, SmCause::PduSessionTypeIpv4v6OnlyAllowed),
            (67, SmCause::InsufficientResourcesForSliceAndDnn),
            (69, SmCause::InsufficientResourcesForSlice),
            (70, SmCause::MissingOrUnknownDnnInSlice),
        ] {
            assert_eq!(SmCause::try_from(raw).unwrap(), cause);
            assert_eq!(cause as u8, raw);
        }
    }
}
