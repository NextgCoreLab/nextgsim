//! Registration Messages (3GPP TS 24.501 Section 8.2.6)
//!
//! This module implements the Registration procedure messages:
//! - Registration Request (UE to network)
//! - Registration Accept (network to UE)
//! - Registration Reject (network to UE)
//! - Registration Complete (UE to network)

use bytes::{Buf, BufMut};
use thiserror::Error;

use crate::enums::MmMessageType;
use crate::header::PlainMmHeader;
use crate::ies::ie1::{Ie5gsRegistrationType, IeMicoIndication, InformationElement1, NssaiInclusionMode};
use crate::ies::ie4::{
    IeAiMlCapability, IeIsacParameter, IeSemanticCommParameter,
    IeSubThzBandParameter, IeNtnTimingAdvance, IeNtnAccessBarring,
};
use crate::security::NasKeySetIdentifier;

/// Error type for Registration message encoding/decoding
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RegistrationError {
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
    /// Invalid IEI
    #[error("Unknown IEI: 0x{0:02X}")]
    UnknownIei(u8),
}

// ============================================================================
// 5GS Mobile Identity (simplified for Registration messages)
// ============================================================================

/// 5GS Mobile Identity Type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum MobileIdentityType {
    /// No identity
    #[default]
    NoIdentity = 0b000,
    /// SUCI
    Suci = 0b001,
    /// 5G-GUTI
    Guti = 0b010,
    /// IMEI
    Imei = 0b011,
    /// 5G-S-TMSI
    Tmsi = 0b100,
    /// IMEISV
    ImeiSv = 0b101,
    /// MAC address
    MacAddress = 0b110,
    /// EUI-64
    Eui64 = 0b111,
}

impl TryFrom<u8> for MobileIdentityType {
    type Error = RegistrationError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value & 0x07 {
            0b000 => Ok(MobileIdentityType::NoIdentity),
            0b001 => Ok(MobileIdentityType::Suci),
            0b010 => Ok(MobileIdentityType::Guti),
            0b011 => Ok(MobileIdentityType::Imei),
            0b100 => Ok(MobileIdentityType::Tmsi),
            0b101 => Ok(MobileIdentityType::ImeiSv),
            0b110 => Ok(MobileIdentityType::MacAddress),
            0b111 => Ok(MobileIdentityType::Eui64),
            _ => Err(RegistrationError::InvalidIeValue(format!(
                "Invalid mobile identity type: 0x{value:02X}"
            ))),
        }
    }
}

/// 5GS Mobile Identity IE (Type 6 - variable length with 2-byte LI)
///
/// 3GPP TS 24.501 Section 9.11.3.4
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Ie5gsMobileIdentity {
    /// Identity type
    pub identity_type: MobileIdentityType,
    /// Raw identity data (type-specific encoding)
    pub data: Vec<u8>,
}

impl Ie5gsMobileIdentity {
    /// Create a new 5GS Mobile Identity with raw data
    pub fn new(identity_type: MobileIdentityType, data: Vec<u8>) -> Self {
        Self {
            identity_type,
            data,
        }
    }

    /// Create a "no identity" mobile identity
    pub fn no_identity() -> Self {
        Self {
            identity_type: MobileIdentityType::NoIdentity,
            data: vec![0x00], // Type octet only
        }
    }

    /// Decode from bytes (without IEI, with length)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, RegistrationError> {
        if buf.remaining() < 2 {
            return Err(RegistrationError::BufferTooShort {
                expected: 2,
                actual: buf.remaining(),
            });
        }

        // Read 2-byte length
        let length = buf.get_u16() as usize;
        if buf.remaining() < length {
            return Err(RegistrationError::BufferTooShort {
                expected: length,
                actual: buf.remaining(),
            });
        }

        if length == 0 {
            return Ok(Self::no_identity());
        }

        // First octet contains identity type in bits 2-0
        let first_octet = buf.chunk()[0];
        let identity_type = MobileIdentityType::try_from(first_octet & 0x07)?;

        // Read the full identity data
        let mut data = vec![0u8; length];
        buf.copy_to_slice(&mut data);

        Ok(Self {
            identity_type,
            data,
        })
    }

    /// Encode to bytes (without IEI, with length)
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        let length = self.data.len();
        buf.put_u16(length as u16);
        buf.put_slice(&self.data);
    }

    /// Get encoded length (including 2-byte length field)
    pub fn encoded_len(&self) -> usize {
        2 + self.data.len()
    }
}

// ============================================================================
// 5GMM Cause (Type 3 - fixed length)
// ============================================================================

/// 5GMM Cause values (3GPP TS 24.501 Section 9.11.3.2)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum MmCause {
    /// Illegal UE
    IllegalUe = 3,
    /// PEI not accepted
    PeiNotAccepted = 5,
    /// Illegal ME
    IllegalMe = 6,
    /// 5GS services not allowed
    FiveGsServicesNotAllowed = 7,
    /// UE identity cannot be derived by the network
    UeIdentityCannotBeDerived = 9,
    /// Implicitly de-registered
    ImplicitlyDeregistered = 10,
    /// PLMN not allowed
    PlmnNotAllowed = 11,
    /// Tracking area not allowed
    TrackingAreaNotAllowed = 12,
    /// Roaming not allowed in this tracking area
    RoamingNotAllowedInTa = 13,
    /// No suitable cells in tracking area
    NoSuitableCellsInTa = 15,
    /// MAC failure
    MacFailure = 20,
    /// Synch failure
    SynchFailure = 21,
    /// Congestion
    Congestion = 22,
    /// UE security capabilities mismatch
    UeSecurityCapabilitiesMismatch = 23,
    /// Security mode rejected, unspecified
    SecurityModeRejectedUnspecified = 24,
    /// Non-5G authentication unacceptable
    Non5gAuthenticationUnacceptable = 26,
    /// N1 mode not allowed
    N1ModeNotAllowed = 27,
    /// Restricted service area
    RestrictedServiceArea = 28,
    /// Redirection to EPC required
    RedirectionToEpcRequired = 31,
    /// LADN not available
    LadnNotAvailable = 43,
    /// No network slices available
    NoNetworkSlicesAvailable = 62,
    /// Maximum number of PDU sessions reached
    MaxPduSessionsReached = 65,
    /// Insufficient resources for specific slice and DNN
    InsufficientResourcesForSliceAndDnn = 67,
    /// Insufficient resources for specific slice
    InsufficientResourcesForSlice = 69,
    /// ngKSI already in use
    NgKsiAlreadyInUse = 71,
    /// Non-3GPP access to 5GCN not allowed
    Non3gppAccessTo5gcnNotAllowed = 72,
    /// Serving network not authorized
    ServingNetworkNotAuthorized = 73,
    /// Temporarily not authorized for this SNPN
    TemporarilyNotAuthorizedForSnpn = 74,
    /// Permanently not authorized for this SNPN
    PermanentlyNotAuthorizedForSnpn = 75,
    /// Not authorized for this CAG or authorized for CAG cells only
    NotAuthorizedForCag = 76,
    /// Wireline access area not allowed
    WirelineAccessAreaNotAllowed = 77,
    /// Payload was not forwarded
    PayloadNotForwarded = 90,
    /// DNN not supported or not subscribed in the slice
    DnnNotSupportedOrNotSubscribed = 91,
    /// Insufficient user-plane resources for the PDU session
    InsufficientUserPlaneResources = 92,
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

impl TryFrom<u8> for MmCause {
    type Error = RegistrationError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            3 => Ok(MmCause::IllegalUe),
            5 => Ok(MmCause::PeiNotAccepted),
            6 => Ok(MmCause::IllegalMe),
            7 => Ok(MmCause::FiveGsServicesNotAllowed),
            9 => Ok(MmCause::UeIdentityCannotBeDerived),
            10 => Ok(MmCause::ImplicitlyDeregistered),
            11 => Ok(MmCause::PlmnNotAllowed),
            12 => Ok(MmCause::TrackingAreaNotAllowed),
            13 => Ok(MmCause::RoamingNotAllowedInTa),
            15 => Ok(MmCause::NoSuitableCellsInTa),
            20 => Ok(MmCause::MacFailure),
            21 => Ok(MmCause::SynchFailure),
            22 => Ok(MmCause::Congestion),
            23 => Ok(MmCause::UeSecurityCapabilitiesMismatch),
            24 => Ok(MmCause::SecurityModeRejectedUnspecified),
            26 => Ok(MmCause::Non5gAuthenticationUnacceptable),
            27 => Ok(MmCause::N1ModeNotAllowed),
            28 => Ok(MmCause::RestrictedServiceArea),
            31 => Ok(MmCause::RedirectionToEpcRequired),
            43 => Ok(MmCause::LadnNotAvailable),
            62 => Ok(MmCause::NoNetworkSlicesAvailable),
            65 => Ok(MmCause::MaxPduSessionsReached),
            67 => Ok(MmCause::InsufficientResourcesForSliceAndDnn),
            69 => Ok(MmCause::InsufficientResourcesForSlice),
            71 => Ok(MmCause::NgKsiAlreadyInUse),
            72 => Ok(MmCause::Non3gppAccessTo5gcnNotAllowed),
            73 => Ok(MmCause::ServingNetworkNotAuthorized),
            74 => Ok(MmCause::TemporarilyNotAuthorizedForSnpn),
            75 => Ok(MmCause::PermanentlyNotAuthorizedForSnpn),
            76 => Ok(MmCause::NotAuthorizedForCag),
            77 => Ok(MmCause::WirelineAccessAreaNotAllowed),
            90 => Ok(MmCause::PayloadNotForwarded),
            91 => Ok(MmCause::DnnNotSupportedOrNotSubscribed),
            92 => Ok(MmCause::InsufficientUserPlaneResources),
            95 => Ok(MmCause::SemanticallyIncorrectMessage),
            96 => Ok(MmCause::InvalidMandatoryInformation),
            97 => Ok(MmCause::MessageTypeNonExistent),
            98 => Ok(MmCause::MessageTypeNotCompatible),
            99 => Ok(MmCause::IeNonExistent),
            100 => Ok(MmCause::ConditionalIeError),
            101 => Ok(MmCause::MessageNotCompatible),
            111 => Ok(MmCause::ProtocolErrorUnspecified),
            _ => Ok(MmCause::ProtocolErrorUnspecified), // Unknown causes map to protocol error
        }
    }
}

/// 5GMM Cause IE (Type 3 - 1 byte)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Ie5gMmCause {
    /// Cause value
    pub value: MmCause,
}

impl Ie5gMmCause {
    /// Create a new 5GMM Cause IE
    pub fn new(value: MmCause) -> Self {
        Self { value }
    }

    /// Decode from bytes
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, RegistrationError> {
        if buf.remaining() < 1 {
            return Err(RegistrationError::BufferTooShort {
                expected: 1,
                actual: buf.remaining(),
            });
        }
        let value = MmCause::try_from(buf.get_u8())?;
        Ok(Self { value })
    }

    /// Encode to bytes
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        buf.put_u8(self.value as u8);
    }
}

// ============================================================================
// 5GS Registration Result (Type 4 - variable length)
// ============================================================================

/// 5GS Registration Result value
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum RegistrationResultValue {
    /// 3GPP access
    #[default]
    ThreeGppAccess = 0b001,
    /// Non-3GPP access
    NonThreeGppAccess = 0b010,
    /// 3GPP access and non-3GPP access
    ThreeGppAndNonThreeGppAccess = 0b011,
}

impl TryFrom<u8> for RegistrationResultValue {
    type Error = RegistrationError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value & 0x07 {
            0b001 => Ok(RegistrationResultValue::ThreeGppAccess),
            0b010 => Ok(RegistrationResultValue::NonThreeGppAccess),
            0b011 => Ok(RegistrationResultValue::ThreeGppAndNonThreeGppAccess),
            _ => Err(RegistrationError::InvalidIeValue(format!(
                "Invalid registration result: 0x{value:02X}"
            ))),
        }
    }
}

/// SMS over NAS allowed indicator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum SmsOverNasAllowed {
    /// SMS over NAS not allowed
    #[default]
    NotAllowed = 0,
    /// SMS over NAS allowed
    Allowed = 1,
}

/// 5GS Registration Result IE (Type 4)
///
/// 3GPP TS 24.501 Section 9.11.3.6
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Ie5gsRegistrationResult {
    /// SMS over NAS allowed
    pub sms_allowed: SmsOverNasAllowed,
    /// Registration result value
    pub result: RegistrationResultValue,
}

impl Ie5gsRegistrationResult {
    /// Create a new 5GS Registration Result IE
    pub fn new(sms_allowed: SmsOverNasAllowed, result: RegistrationResultValue) -> Self {
        Self { sms_allowed, result }
    }

    /// Decode from bytes (without IEI, with length)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, RegistrationError> {
        if buf.remaining() < 1 {
            return Err(RegistrationError::BufferTooShort {
                expected: 1,
                actual: buf.remaining(),
            });
        }

        let length = buf.get_u8() as usize;
        if buf.remaining() < length || length < 1 {
            return Err(RegistrationError::BufferTooShort {
                expected: length.max(1),
                actual: buf.remaining(),
            });
        }

        let octet = buf.get_u8();
        let sms_allowed = if (octet >> 3) & 0x01 == 1 {
            SmsOverNasAllowed::Allowed
        } else {
            SmsOverNasAllowed::NotAllowed
        };
        let result = RegistrationResultValue::try_from(octet & 0x07)?;

        // Skip any remaining bytes
        if length > 1 {
            buf.advance(length - 1);
        }

        Ok(Self { sms_allowed, result })
    }

    /// Encode to bytes (without IEI, with length)
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        buf.put_u8(1); // Length
        let octet = ((self.sms_allowed as u8) << 3) | (self.result as u8);
        buf.put_u8(octet);
    }

    /// Get encoded length (including length field)
    pub fn encoded_len(&self) -> usize {
        2 // 1 byte length + 1 byte value
    }
}


// ============================================================================
// Registration Request Message (3GPP TS 24.501 Section 8.2.6)
// ============================================================================

/// IEI values for Registration Request optional IEs
#[allow(dead_code)]
mod registration_request_iei {
    /// Non-current native NAS key set identifier
    pub const NON_CURRENT_NGKSI: u8 = 0xC;
    /// 5GMM capability
    pub const MM_CAPABILITY: u8 = 0x10;
    /// UE security capability
    pub const UE_SECURITY_CAPABILITY: u8 = 0x2E;
    /// Requested NSSAI
    pub const REQUESTED_NSSAI: u8 = 0x2F;
    /// Last visited registered TAI
    pub const LAST_VISITED_TAI: u8 = 0x52;
    /// S1 UE network capability
    pub const S1_UE_NETWORK_CAPABILITY: u8 = 0x17;
    /// Uplink data status
    pub const UPLINK_DATA_STATUS: u8 = 0x40;
    /// PDU session status
    pub const PDU_SESSION_STATUS: u8 = 0x50;
    /// MICO indication
    pub const MICO_INDICATION: u8 = 0xB;
    /// UE status
    pub const UE_STATUS: u8 = 0x2B;
    /// Additional GUTI
    pub const ADDITIONAL_GUTI: u8 = 0x77;
    /// Allowed PDU session status
    pub const ALLOWED_PDU_SESSION_STATUS: u8 = 0x25;
    /// UE's usage setting
    pub const UES_USAGE_SETTING: u8 = 0x18;
    /// Requested DRX parameters
    pub const REQUESTED_DRX_PARAMETERS: u8 = 0x51;
    /// EPS NAS message container
    pub const EPS_NAS_MESSAGE_CONTAINER: u8 = 0x70;
    /// LADN indication
    pub const LADN_INDICATION: u8 = 0x74;
    /// Payload container
    pub const PAYLOAD_CONTAINER: u8 = 0x7B;
    /// Network slicing indication
    pub const NETWORK_SLICING_INDICATION: u8 = 0x9;
    /// 5GS update type
    pub const UPDATE_TYPE: u8 = 0x53;
    /// NAS message container
    pub const NAS_MESSAGE_CONTAINER: u8 = 0x71;
    /// AI/ML capability (6G extension)
    pub const AI_ML_CAPABILITY: u8 = 0xA0;
    /// ISAC parameter (6G extension)
    pub const ISAC_PARAMETER: u8 = 0xA1;
    /// Semantic communication parameter (6G extension)
    pub const SEMANTIC_COMM_PARAMETER: u8 = 0xA2;
    /// Sub-THz band parameter (6G extension)
    pub const SUB_THZ_BAND_PARAMETER: u8 = 0xA3;
    /// NTN timing advance (6G extension)
    pub const NTN_TIMING_ADVANCE: u8 = 0xA4;
    /// NTN access barring (6G extension)
    pub const NTN_ACCESS_BARRING: u8 = 0xA5;
}

/// Registration Request message (UE to network)
///
/// 3GPP TS 24.501 Section 8.2.6
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegistrationRequest {
    /// 5GS registration type (mandatory, Type 1)
    pub registration_type: Ie5gsRegistrationType,
    /// ngKSI - NAS key set identifier (mandatory, Type 1)
    pub ng_ksi: NasKeySetIdentifier,
    /// 5GS mobile identity (mandatory, Type 6)
    pub mobile_identity: Ie5gsMobileIdentity,
    /// Non-current native NAS key set identifier (optional, Type 1, IEI 0xC)
    pub non_current_ng_ksi: Option<NasKeySetIdentifier>,
    /// UE security capability (optional, Type 4, IEI 0x2E)
    pub ue_security_capability: Option<Vec<u8>>,
    /// Requested NSSAI (optional, Type 4, IEI 0x2F)
    pub requested_nssai: Option<Vec<u8>>,
    /// Last visited registered TAI (optional, Type 3, IEI 0x52)
    pub last_visited_tai: Option<[u8; 6]>,
    /// UE status (optional, Type 4, IEI 0x2B)
    pub ue_status: Option<u8>,
    /// Additional GUTI (optional, Type 6, IEI 0x77)
    pub additional_guti: Option<Ie5gsMobileIdentity>,
    /// Uplink data status (optional, Type 4, IEI 0x40)
    pub uplink_data_status: Option<u16>,
    /// PDU session status (optional, Type 4, IEI 0x50)
    pub pdu_session_status: Option<u16>,
    /// NAS message container (optional, Type 6, IEI 0x71)
    pub nas_message_container: Option<Vec<u8>>,
    /// MICO indication (optional, Type 1, IEI 0xB)
    pub mico_indication: Option<IeMicoIndication>,
    /// LADN indication (optional, Type 6, IEI 0x74)
    pub ladn_indication: Option<Vec<u8>>,
    /// AI/ML capability (optional, Type 4, IEI 0xA0) - 6G extension
    pub ai_ml_capability: Option<IeAiMlCapability>,
    /// ISAC parameter (optional, Type 4, IEI 0xA1) - 6G extension
    pub isac_parameter: Option<IeIsacParameter>,
    /// Semantic communication parameter (optional, Type 4, IEI 0xA2) - 6G extension
    pub semantic_comm_parameter: Option<IeSemanticCommParameter>,
    /// Sub-THz band parameter (optional, Type 4, IEI 0xA3) - 6G extension
    pub sub_thz_band_parameter: Option<IeSubThzBandParameter>,
    /// NTN timing advance (optional, Type 4, IEI 0xA4) - 6G extension
    pub ntn_timing_advance: Option<IeNtnTimingAdvance>,
    /// NTN access barring (optional, Type 4, IEI 0xA5) - 6G extension
    pub ntn_access_barring: Option<IeNtnAccessBarring>,
}

impl Default for RegistrationRequest {
    fn default() -> Self {
        Self {
            registration_type: Ie5gsRegistrationType::default(),
            ng_ksi: NasKeySetIdentifier::no_key(),
            mobile_identity: Ie5gsMobileIdentity::no_identity(),
            non_current_ng_ksi: None,
            ue_security_capability: None,
            requested_nssai: None,
            last_visited_tai: None,
            ue_status: None,
            additional_guti: None,
            uplink_data_status: None,
            pdu_session_status: None,
            nas_message_container: None,
            mico_indication: None,
            ladn_indication: None,
            ai_ml_capability: None,
            isac_parameter: None,
            semantic_comm_parameter: None,
            sub_thz_band_parameter: None,
            ntn_timing_advance: None,
            ntn_access_barring: None,
        }
    }
}

impl RegistrationRequest {
    /// Create a new Registration Request with mandatory fields
    pub fn new(
        registration_type: Ie5gsRegistrationType,
        ng_ksi: NasKeySetIdentifier,
        mobile_identity: Ie5gsMobileIdentity,
    ) -> Self {
        Self {
            registration_type,
            ng_ksi,
            mobile_identity,
            ..Default::default()
        }
    }

    /// Decode from bytes (after header has been parsed)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, RegistrationError> {
        if buf.remaining() < 1 {
            return Err(RegistrationError::BufferTooShort {
                expected: 1,
                actual: buf.remaining(),
            });
        }

        // First octet: ngKSI (high nibble) + 5GS registration type (low nibble)
        let first_octet = buf.get_u8();
        let ng_ksi = NasKeySetIdentifier::decode((first_octet >> 4) & 0x0F)
            .map_err(|e| RegistrationError::InvalidIeValue(e.to_string()))?;
        let registration_type = Ie5gsRegistrationType::decode(first_octet & 0x0F)
            .map_err(|e| RegistrationError::InvalidIeValue(e.to_string()))?;

        // 5GS mobile identity (mandatory, Type 6)
        let mobile_identity = Ie5gsMobileIdentity::decode(buf)?;

        let mut msg = Self::new(registration_type, ng_ksi, mobile_identity);

        // Parse optional IEs
        while buf.remaining() > 0 {
            let iei = buf.chunk()[0];

            // Check for Type 1 IEs (4-bit IEI in high nibble)
            let iei_high = (iei >> 4) & 0x0F;
            match iei_high {
                0xC => {
                    // Non-current native NAS key set identifier
                    buf.advance(1);
                    msg.non_current_ng_ksi = Some(
                        NasKeySetIdentifier::decode(iei & 0x0F)
                            .map_err(|e| RegistrationError::InvalidIeValue(e.to_string()))?,
                    );
                    continue;
                }
                0xB => {
                    // MICO indication (Type 1, IEI 0xB)
                    let val = buf.get_u8() & 0x0F;
                    msg.mico_indication = Some(
                        IeMicoIndication::decode(val)
                            .unwrap_or_default()
                    );
                    continue;
                }
                0x9 => {
                    // Network slicing indication
                    buf.advance(1);
                    continue;
                }
                _ => {}
            }

            // Full octet IEI
            match iei {
                registration_request_iei::UE_SECURITY_CAPABILITY => {
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
                    msg.ue_security_capability = Some(data);
                }
                registration_request_iei::REQUESTED_NSSAI => {
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
                    msg.requested_nssai = Some(data);
                }
                registration_request_iei::LAST_VISITED_TAI => {
                    buf.advance(1);
                    if buf.remaining() < 6 {
                        break;
                    }
                    let mut tai = [0u8; 6];
                    buf.copy_to_slice(&mut tai);
                    msg.last_visited_tai = Some(tai);
                }
                registration_request_iei::UE_STATUS => {
                    buf.advance(1);
                    if buf.remaining() < 1 {
                        break;
                    }
                    let len = buf.get_u8() as usize;
                    if buf.remaining() < len || len < 1 {
                        break;
                    }
                    msg.ue_status = Some(buf.get_u8());
                    if len > 1 {
                        buf.advance(len - 1);
                    }
                }
                registration_request_iei::ADDITIONAL_GUTI => {
                    buf.advance(1);
                    msg.additional_guti = Some(Ie5gsMobileIdentity::decode(buf)?);
                }
                registration_request_iei::UPLINK_DATA_STATUS => {
                    buf.advance(1);
                    if buf.remaining() < 1 {
                        break;
                    }
                    let len = buf.get_u8() as usize;
                    if buf.remaining() < len || len < 2 {
                        break;
                    }
                    msg.uplink_data_status = Some(buf.get_u16());
                    if len > 2 {
                        buf.advance(len - 2);
                    }
                }
                registration_request_iei::PDU_SESSION_STATUS => {
                    buf.advance(1);
                    if buf.remaining() < 1 {
                        break;
                    }
                    let len = buf.get_u8() as usize;
                    if buf.remaining() < len || len < 2 {
                        break;
                    }
                    msg.pdu_session_status = Some(buf.get_u16());
                    if len > 2 {
                        buf.advance(len - 2);
                    }
                }
                registration_request_iei::NAS_MESSAGE_CONTAINER => {
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
                    msg.nas_message_container = Some(data);
                }
                registration_request_iei::LADN_INDICATION => {
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
                    msg.ladn_indication = Some(data);
                }
                // 6G extension IEs
                registration_request_iei::AI_ML_CAPABILITY => {
                    buf.advance(1);
                    if let Ok(ie) = IeAiMlCapability::decode(buf) {
                        msg.ai_ml_capability = Some(ie);
                    }
                }
                registration_request_iei::ISAC_PARAMETER => {
                    buf.advance(1);
                    if let Ok(ie) = IeIsacParameter::decode(buf) {
                        msg.isac_parameter = Some(ie);
                    }
                }
                registration_request_iei::SEMANTIC_COMM_PARAMETER => {
                    buf.advance(1);
                    if let Ok(ie) = IeSemanticCommParameter::decode(buf) {
                        msg.semantic_comm_parameter = Some(ie);
                    }
                }
                registration_request_iei::SUB_THZ_BAND_PARAMETER => {
                    buf.advance(1);
                    if let Ok(ie) = IeSubThzBandParameter::decode(buf) {
                        msg.sub_thz_band_parameter = Some(ie);
                    }
                }
                registration_request_iei::NTN_TIMING_ADVANCE => {
                    buf.advance(1);
                    if let Ok(ie) = IeNtnTimingAdvance::decode(buf) {
                        msg.ntn_timing_advance = Some(ie);
                    }
                }
                registration_request_iei::NTN_ACCESS_BARRING => {
                    buf.advance(1);
                    if let Ok(ie) = IeNtnAccessBarring::decode(buf) {
                        msg.ntn_access_barring = Some(ie);
                    }
                }
                _ => {
                    // Skip unknown IEs
                    buf.advance(1);
                    // Try to determine length and skip
                    if buf.remaining() > 0 {
                        // Assume Type 4 format (1-byte length)
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
        let header = PlainMmHeader::new(MmMessageType::RegistrationRequest);
        header.encode(buf);

        // First octet: ngKSI (high nibble) + 5GS registration type (low nibble)
        let first_octet = (self.ng_ksi.encode() << 4) | (self.registration_type.encode() & 0x0F);
        buf.put_u8(first_octet);

        // 5GS mobile identity (mandatory)
        self.mobile_identity.encode(buf);

        // Optional IEs
        if let Some(ref ng_ksi) = self.non_current_ng_ksi {
            buf.put_u8((registration_request_iei::NON_CURRENT_NGKSI << 4) | (ng_ksi.encode() & 0x0F));
        }

        if let Some(ref cap) = self.ue_security_capability {
            buf.put_u8(registration_request_iei::UE_SECURITY_CAPABILITY);
            buf.put_u8(cap.len() as u8);
            buf.put_slice(cap);
        }

        if let Some(ref nssai) = self.requested_nssai {
            buf.put_u8(registration_request_iei::REQUESTED_NSSAI);
            buf.put_u8(nssai.len() as u8);
            buf.put_slice(nssai);
        }

        if let Some(ref tai) = self.last_visited_tai {
            buf.put_u8(registration_request_iei::LAST_VISITED_TAI);
            buf.put_slice(tai);
        }

        if let Some(status) = self.ue_status {
            buf.put_u8(registration_request_iei::UE_STATUS);
            buf.put_u8(1);
            buf.put_u8(status);
        }

        if let Some(ref guti) = self.additional_guti {
            buf.put_u8(registration_request_iei::ADDITIONAL_GUTI);
            guti.encode(buf);
        }

        if let Some(status) = self.uplink_data_status {
            buf.put_u8(registration_request_iei::UPLINK_DATA_STATUS);
            buf.put_u8(2);
            buf.put_u16(status);
        }

        if let Some(status) = self.pdu_session_status {
            buf.put_u8(registration_request_iei::PDU_SESSION_STATUS);
            buf.put_u8(2);
            buf.put_u16(status);
        }

        if let Some(ref container) = self.nas_message_container {
            buf.put_u8(registration_request_iei::NAS_MESSAGE_CONTAINER);
            buf.put_u16(container.len() as u16);
            buf.put_slice(container);
        }

        if let Some(ref mico) = self.mico_indication {
            buf.put_u8((registration_request_iei::MICO_INDICATION << 4) | (mico.encode() & 0x0F));
        }

        if let Some(ref ladn) = self.ladn_indication {
            buf.put_u8(registration_request_iei::LADN_INDICATION);
            buf.put_u16(ladn.len() as u16);
            buf.put_slice(ladn);
        }

        // 6G extension IEs
        if let Some(ref ie) = self.ai_ml_capability {
            buf.put_u8(registration_request_iei::AI_ML_CAPABILITY);
            ie.encode(buf);
        }
        if let Some(ref ie) = self.isac_parameter {
            buf.put_u8(registration_request_iei::ISAC_PARAMETER);
            ie.encode(buf);
        }
        if let Some(ref ie) = self.semantic_comm_parameter {
            buf.put_u8(registration_request_iei::SEMANTIC_COMM_PARAMETER);
            ie.encode(buf);
        }
        if let Some(ref ie) = self.sub_thz_band_parameter {
            buf.put_u8(registration_request_iei::SUB_THZ_BAND_PARAMETER);
            ie.encode(buf);
        }
        if let Some(ref ie) = self.ntn_timing_advance {
            buf.put_u8(registration_request_iei::NTN_TIMING_ADVANCE);
            ie.encode(buf);
        }
        if let Some(ref ie) = self.ntn_access_barring {
            buf.put_u8(registration_request_iei::NTN_ACCESS_BARRING);
            ie.encode(buf);
        }
    }

    /// Get the message type
    pub fn message_type() -> MmMessageType {
        MmMessageType::RegistrationRequest
    }
}


// ============================================================================
// Registration Accept Message (3GPP TS 24.501 Section 8.2.7)
// ============================================================================

/// IEI values for Registration Accept optional IEs
#[allow(dead_code)]
mod registration_accept_iei {
    /// 5G-GUTI
    pub const GUTI: u8 = 0x77;
    /// Equivalent PLMNs
    pub const EQUIVALENT_PLMNS: u8 = 0x4A;
    /// TAI list
    pub const TAI_LIST: u8 = 0x54;
    /// Allowed NSSAI
    pub const ALLOWED_NSSAI: u8 = 0x15;
    /// Rejected NSSAI
    pub const REJECTED_NSSAI: u8 = 0x11;
    /// Configured NSSAI
    pub const CONFIGURED_NSSAI: u8 = 0x31;
    /// 5GS network feature support
    pub const NETWORK_FEATURE_SUPPORT: u8 = 0x21;
    /// PDU session status
    pub const PDU_SESSION_STATUS: u8 = 0x50;
    /// PDU session reactivation result
    pub const PDU_SESSION_REACTIVATION_RESULT: u8 = 0x26;
    /// PDU session reactivation result error cause
    pub const PDU_SESSION_REACTIVATION_RESULT_ERROR_CAUSE: u8 = 0x72;
    /// LADN information
    pub const LADN_INFORMATION: u8 = 0x79;
    /// MICO indication
    pub const MICO_INDICATION: u8 = 0xB;
    /// Network slicing indication
    pub const NETWORK_SLICING_INDICATION: u8 = 0x9;
    /// Service area list
    pub const SERVICE_AREA_LIST: u8 = 0x27;
    /// T3512 value
    pub const T3512_VALUE: u8 = 0x5E;
    /// Non-3GPP de-registration timer value
    pub const NON_3GPP_DEREGISTRATION_TIMER: u8 = 0x5D;
    /// T3502 value
    pub const T3502_VALUE: u8 = 0x16;
    /// Emergency number list
    pub const EMERGENCY_NUMBER_LIST: u8 = 0x34;
    /// Extended emergency number list
    pub const EXTENDED_EMERGENCY_NUMBER_LIST: u8 = 0x7A;
    /// SOR transparent container
    pub const SOR_TRANSPARENT_CONTAINER: u8 = 0x73;
    /// EAP message
    pub const EAP_MESSAGE: u8 = 0x78;
    /// NSSAI inclusion mode
    pub const NSSAI_INCLUSION_MODE: u8 = 0xA;
    /// Operator-defined access category definitions
    pub const OPERATOR_DEFINED_ACCESS_CATEGORY_DEFINITIONS: u8 = 0x76;
    /// Negotiated DRX parameters
    pub const NEGOTIATED_DRX_PARAMETERS: u8 = 0x51;
    /// AI/ML capability (6G extension)
    pub const AI_ML_CAPABILITY: u8 = 0xA0;
    /// ISAC parameter (6G extension)
    pub const ISAC_PARAMETER: u8 = 0xA1;
    /// Semantic communication parameter (6G extension)
    pub const SEMANTIC_COMM_PARAMETER: u8 = 0xA2;
    /// Sub-THz band parameter (6G extension)
    pub const SUB_THZ_BAND_PARAMETER: u8 = 0xA3;
    /// NTN timing advance (6G extension)
    pub const NTN_TIMING_ADVANCE: u8 = 0xA4;
    /// NTN access barring (6G extension)
    pub const NTN_ACCESS_BARRING: u8 = 0xA5;
}

/// Registration Accept message (network to UE)
///
/// 3GPP TS 24.501 Section 8.2.7
#[derive(Debug, Clone, PartialEq, Eq)]
#[derive(Default)]
pub struct RegistrationAccept {
    /// 5GS registration result (mandatory, Type 4)
    pub registration_result: Ie5gsRegistrationResult,
    /// 5G-GUTI (optional, Type 6, IEI 0x77)
    pub guti: Option<Ie5gsMobileIdentity>,
    /// Equivalent PLMNs (optional, Type 4, IEI 0x4A)
    pub equivalent_plmns: Option<Vec<u8>>,
    /// TAI list (optional, Type 4, IEI 0x54)
    pub tai_list: Option<Vec<u8>>,
    /// Allowed NSSAI (optional, Type 4, IEI 0x15)
    pub allowed_nssai: Option<Vec<u8>>,
    /// Rejected NSSAI (optional, Type 4, IEI 0x11)
    pub rejected_nssai: Option<Vec<u8>>,
    /// Configured NSSAI (optional, Type 4, IEI 0x31)
    pub configured_nssai: Option<Vec<u8>>,
    /// 5GS network feature support (optional, Type 4, IEI 0x21)
    pub network_feature_support: Option<Vec<u8>>,
    /// PDU session status (optional, Type 4, IEI 0x50)
    pub pdu_session_status: Option<u16>,
    /// PDU session reactivation result (optional, Type 4, IEI 0x26)
    pub pdu_session_reactivation_result: Option<u16>,
    /// T3512 value (optional, Type 4, IEI 0x5E)
    pub t3512_value: Option<u8>,
    /// T3502 value (optional, Type 4, IEI 0x16)
    pub t3502_value: Option<u8>,
    /// EAP message (optional, Type 6, IEI 0x78)
    pub eap_message: Option<Vec<u8>>,
    /// NSSAI inclusion mode (optional, Type 1, IEI 0xA)
    pub nssai_inclusion_mode: Option<NssaiInclusionMode>,
    /// Negotiated DRX parameters (optional, Type 4, IEI 0x51)
    pub negotiated_drx_parameters: Option<u8>,
    /// MICO indication (optional, Type 1, IEI 0xB)
    pub mico_indication: Option<IeMicoIndication>,
    /// LADN information (optional, Type 6, IEI 0x79)
    pub ladn_information: Option<Vec<u8>>,
    /// AI/ML capability (optional, Type 4, IEI 0xA0) - 6G extension
    pub ai_ml_capability: Option<IeAiMlCapability>,
    /// ISAC parameter (optional, Type 4, IEI 0xA1) - 6G extension
    pub isac_parameter: Option<IeIsacParameter>,
    /// Semantic communication parameter (optional, Type 4, IEI 0xA2) - 6G extension
    pub semantic_comm_parameter: Option<IeSemanticCommParameter>,
    /// Sub-THz band parameter (optional, Type 4, IEI 0xA3) - 6G extension
    pub sub_thz_band_parameter: Option<IeSubThzBandParameter>,
    /// NTN timing advance (optional, Type 4, IEI 0xA4) - 6G extension
    pub ntn_timing_advance: Option<IeNtnTimingAdvance>,
    /// NTN access barring (optional, Type 4, IEI 0xA5) - 6G extension
    pub ntn_access_barring: Option<IeNtnAccessBarring>,
}


impl RegistrationAccept {
    /// Create a new Registration Accept with mandatory fields
    pub fn new(registration_result: Ie5gsRegistrationResult) -> Self {
        Self {
            registration_result,
            ..Default::default()
        }
    }

    /// Decode from bytes (after header has been parsed)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, RegistrationError> {
        // 5GS registration result (mandatory, Type 4)
        let registration_result = Ie5gsRegistrationResult::decode(buf)?;

        let mut msg = Self::new(registration_result);

        // Parse optional IEs
        while buf.remaining() > 0 {
            let iei = buf.chunk()[0];

            // Check for Type 1 IEs (4-bit IEI in high nibble)
            let iei_high = (iei >> 4) & 0x0F;
            match iei_high {
                0xB => {
                    // MICO indication (Type 1, IEI 0xB)
                    let val = buf.get_u8() & 0x0F;
                    msg.mico_indication = Some(
                        IeMicoIndication::decode(val)
                            .unwrap_or_default()
                    );
                    continue;
                }
                0x9 => {
                    // Network slicing indication
                    buf.advance(1);
                    continue;
                }
                0xA => {
                    // NSSAI inclusion mode
                    buf.advance(1);
                    msg.nssai_inclusion_mode = Some(NssaiInclusionMode::try_from(iei & 0x03)
                        .unwrap_or(NssaiInclusionMode::A));
                    continue;
                }
                _ => {}
            }

            // Full octet IEI
            match iei {
                registration_accept_iei::GUTI => {
                    buf.advance(1);
                    msg.guti = Some(Ie5gsMobileIdentity::decode(buf)?);
                }
                registration_accept_iei::EQUIVALENT_PLMNS => {
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
                    msg.equivalent_plmns = Some(data);
                }
                registration_accept_iei::TAI_LIST => {
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
                    msg.tai_list = Some(data);
                }
                registration_accept_iei::ALLOWED_NSSAI => {
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
                    msg.allowed_nssai = Some(data);
                }
                registration_accept_iei::REJECTED_NSSAI => {
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
                    msg.rejected_nssai = Some(data);
                }
                registration_accept_iei::CONFIGURED_NSSAI => {
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
                    msg.configured_nssai = Some(data);
                }
                registration_accept_iei::NETWORK_FEATURE_SUPPORT => {
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
                    msg.network_feature_support = Some(data);
                }
                registration_accept_iei::PDU_SESSION_STATUS => {
                    buf.advance(1);
                    if buf.remaining() < 1 {
                        break;
                    }
                    let len = buf.get_u8() as usize;
                    if buf.remaining() < len || len < 2 {
                        break;
                    }
                    msg.pdu_session_status = Some(buf.get_u16());
                    if len > 2 {
                        buf.advance(len - 2);
                    }
                }
                registration_accept_iei::PDU_SESSION_REACTIVATION_RESULT => {
                    buf.advance(1);
                    if buf.remaining() < 1 {
                        break;
                    }
                    let len = buf.get_u8() as usize;
                    if buf.remaining() < len || len < 2 {
                        break;
                    }
                    msg.pdu_session_reactivation_result = Some(buf.get_u16());
                    if len > 2 {
                        buf.advance(len - 2);
                    }
                }
                registration_accept_iei::T3512_VALUE => {
                    buf.advance(1);
                    if buf.remaining() < 1 {
                        break;
                    }
                    let len = buf.get_u8() as usize;
                    if buf.remaining() < len || len < 1 {
                        break;
                    }
                    msg.t3512_value = Some(buf.get_u8());
                    if len > 1 {
                        buf.advance(len - 1);
                    }
                }
                registration_accept_iei::T3502_VALUE => {
                    buf.advance(1);
                    if buf.remaining() < 1 {
                        break;
                    }
                    let len = buf.get_u8() as usize;
                    if buf.remaining() < len || len < 1 {
                        break;
                    }
                    msg.t3502_value = Some(buf.get_u8());
                    if len > 1 {
                        buf.advance(len - 1);
                    }
                }
                registration_accept_iei::EAP_MESSAGE => {
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
                    msg.eap_message = Some(data);
                }
                registration_accept_iei::NEGOTIATED_DRX_PARAMETERS => {
                    buf.advance(1);
                    if buf.remaining() < 1 {
                        break;
                    }
                    let len = buf.get_u8() as usize;
                    if buf.remaining() < len || len < 1 {
                        break;
                    }
                    msg.negotiated_drx_parameters = Some(buf.get_u8());
                    if len > 1 {
                        buf.advance(len - 1);
                    }
                }
                registration_accept_iei::LADN_INFORMATION => {
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
                    msg.ladn_information = Some(data);
                }
                // 6G extension IEs
                registration_accept_iei::AI_ML_CAPABILITY => {
                    buf.advance(1);
                    if let Ok(ie) = IeAiMlCapability::decode(buf) {
                        msg.ai_ml_capability = Some(ie);
                    }
                }
                registration_accept_iei::ISAC_PARAMETER => {
                    buf.advance(1);
                    if let Ok(ie) = IeIsacParameter::decode(buf) {
                        msg.isac_parameter = Some(ie);
                    }
                }
                registration_accept_iei::SEMANTIC_COMM_PARAMETER => {
                    buf.advance(1);
                    if let Ok(ie) = IeSemanticCommParameter::decode(buf) {
                        msg.semantic_comm_parameter = Some(ie);
                    }
                }
                registration_accept_iei::SUB_THZ_BAND_PARAMETER => {
                    buf.advance(1);
                    if let Ok(ie) = IeSubThzBandParameter::decode(buf) {
                        msg.sub_thz_band_parameter = Some(ie);
                    }
                }
                registration_accept_iei::NTN_TIMING_ADVANCE => {
                    buf.advance(1);
                    if let Ok(ie) = IeNtnTimingAdvance::decode(buf) {
                        msg.ntn_timing_advance = Some(ie);
                    }
                }
                registration_accept_iei::NTN_ACCESS_BARRING => {
                    buf.advance(1);
                    if let Ok(ie) = IeNtnAccessBarring::decode(buf) {
                        msg.ntn_access_barring = Some(ie);
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
        let header = PlainMmHeader::new(MmMessageType::RegistrationAccept);
        header.encode(buf);

        // 5GS registration result (mandatory)
        self.registration_result.encode(buf);

        // Optional IEs
        if let Some(ref guti) = self.guti {
            buf.put_u8(registration_accept_iei::GUTI);
            guti.encode(buf);
        }

        if let Some(ref plmns) = self.equivalent_plmns {
            buf.put_u8(registration_accept_iei::EQUIVALENT_PLMNS);
            buf.put_u8(plmns.len() as u8);
            buf.put_slice(plmns);
        }

        if let Some(ref tai_list) = self.tai_list {
            buf.put_u8(registration_accept_iei::TAI_LIST);
            buf.put_u8(tai_list.len() as u8);
            buf.put_slice(tai_list);
        }

        if let Some(ref nssai) = self.allowed_nssai {
            buf.put_u8(registration_accept_iei::ALLOWED_NSSAI);
            buf.put_u8(nssai.len() as u8);
            buf.put_slice(nssai);
        }

        if let Some(ref nssai) = self.rejected_nssai {
            buf.put_u8(registration_accept_iei::REJECTED_NSSAI);
            buf.put_u8(nssai.len() as u8);
            buf.put_slice(nssai);
        }

        if let Some(ref nssai) = self.configured_nssai {
            buf.put_u8(registration_accept_iei::CONFIGURED_NSSAI);
            buf.put_u8(nssai.len() as u8);
            buf.put_slice(nssai);
        }

        if let Some(ref support) = self.network_feature_support {
            buf.put_u8(registration_accept_iei::NETWORK_FEATURE_SUPPORT);
            buf.put_u8(support.len() as u8);
            buf.put_slice(support);
        }

        if let Some(status) = self.pdu_session_status {
            buf.put_u8(registration_accept_iei::PDU_SESSION_STATUS);
            buf.put_u8(2);
            buf.put_u16(status);
        }

        if let Some(result) = self.pdu_session_reactivation_result {
            buf.put_u8(registration_accept_iei::PDU_SESSION_REACTIVATION_RESULT);
            buf.put_u8(2);
            buf.put_u16(result);
        }

        if let Some(value) = self.t3512_value {
            buf.put_u8(registration_accept_iei::T3512_VALUE);
            buf.put_u8(1);
            buf.put_u8(value);
        }

        if let Some(value) = self.t3502_value {
            buf.put_u8(registration_accept_iei::T3502_VALUE);
            buf.put_u8(1);
            buf.put_u8(value);
        }

        if let Some(ref eap) = self.eap_message {
            buf.put_u8(registration_accept_iei::EAP_MESSAGE);
            buf.put_u16(eap.len() as u16);
            buf.put_slice(eap);
        }

        if let Some(mode) = self.nssai_inclusion_mode {
            buf.put_u8((registration_accept_iei::NSSAI_INCLUSION_MODE << 4) | (mode as u8 & 0x03));
        }

        if let Some(drx) = self.negotiated_drx_parameters {
            buf.put_u8(registration_accept_iei::NEGOTIATED_DRX_PARAMETERS);
            buf.put_u8(1);
            buf.put_u8(drx);
        }

        if let Some(ref mico) = self.mico_indication {
            buf.put_u8((registration_accept_iei::MICO_INDICATION << 4) | (mico.encode() & 0x0F));
        }

        if let Some(ref ladn) = self.ladn_information {
            buf.put_u8(registration_accept_iei::LADN_INFORMATION);
            buf.put_u16(ladn.len() as u16);
            buf.put_slice(ladn);
        }

        // 6G extension IEs
        if let Some(ref ie) = self.ai_ml_capability {
            buf.put_u8(registration_accept_iei::AI_ML_CAPABILITY);
            ie.encode(buf);
        }
        if let Some(ref ie) = self.isac_parameter {
            buf.put_u8(registration_accept_iei::ISAC_PARAMETER);
            ie.encode(buf);
        }
        if let Some(ref ie) = self.semantic_comm_parameter {
            buf.put_u8(registration_accept_iei::SEMANTIC_COMM_PARAMETER);
            ie.encode(buf);
        }
        if let Some(ref ie) = self.sub_thz_band_parameter {
            buf.put_u8(registration_accept_iei::SUB_THZ_BAND_PARAMETER);
            ie.encode(buf);
        }
        if let Some(ref ie) = self.ntn_timing_advance {
            buf.put_u8(registration_accept_iei::NTN_TIMING_ADVANCE);
            ie.encode(buf);
        }
        if let Some(ref ie) = self.ntn_access_barring {
            buf.put_u8(registration_accept_iei::NTN_ACCESS_BARRING);
            ie.encode(buf);
        }
    }

    /// Get the message type
    pub fn message_type() -> MmMessageType {
        MmMessageType::RegistrationAccept
    }
}


// ============================================================================
// Registration Reject Message (3GPP TS 24.501 Section 8.2.8)
// ============================================================================

/// IEI values for Registration Reject optional IEs
mod registration_reject_iei {
    /// T3346 value
    pub const T3346_VALUE: u8 = 0x5F;
    /// T3502 value
    pub const T3502_VALUE: u8 = 0x16;
    /// EAP message
    pub const EAP_MESSAGE: u8 = 0x78;
    /// Rejected NSSAI
    pub const REJECTED_NSSAI: u8 = 0x69;
}

/// Registration Reject message (network to UE)
///
/// 3GPP TS 24.501 Section 8.2.8
#[derive(Debug, Clone, PartialEq, Eq)]
#[derive(Default)]
pub struct RegistrationReject {
    /// 5GMM cause (mandatory, Type 3)
    pub mm_cause: Ie5gMmCause,
    /// T3346 value (optional, Type 4, IEI 0x5F)
    pub t3346_value: Option<u8>,
    /// T3502 value (optional, Type 4, IEI 0x16)
    pub t3502_value: Option<u8>,
    /// EAP message (optional, Type 6, IEI 0x78)
    pub eap_message: Option<Vec<u8>>,
    /// Rejected NSSAI (optional, Type 4, IEI 0x69)
    pub rejected_nssai: Option<Vec<u8>>,
}


impl RegistrationReject {
    /// Create a new Registration Reject with mandatory fields
    pub fn new(mm_cause: Ie5gMmCause) -> Self {
        Self {
            mm_cause,
            ..Default::default()
        }
    }

    /// Create a Registration Reject with a cause value
    pub fn with_cause(cause: MmCause) -> Self {
        Self::new(Ie5gMmCause::new(cause))
    }

    /// Decode from bytes (after header has been parsed)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, RegistrationError> {
        // 5GMM cause (mandatory, Type 3)
        let mm_cause = Ie5gMmCause::decode(buf)?;

        let mut msg = Self::new(mm_cause);

        // Parse optional IEs
        while buf.remaining() > 0 {
            let iei = buf.chunk()[0];

            match iei {
                registration_reject_iei::T3346_VALUE => {
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
                registration_reject_iei::T3502_VALUE => {
                    buf.advance(1);
                    if buf.remaining() < 1 {
                        break;
                    }
                    let len = buf.get_u8() as usize;
                    if buf.remaining() < len || len < 1 {
                        break;
                    }
                    msg.t3502_value = Some(buf.get_u8());
                    if len > 1 {
                        buf.advance(len - 1);
                    }
                }
                registration_reject_iei::EAP_MESSAGE => {
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
                    msg.eap_message = Some(data);
                }
                registration_reject_iei::REJECTED_NSSAI => {
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
                    msg.rejected_nssai = Some(data);
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
        let header = PlainMmHeader::new(MmMessageType::RegistrationReject);
        header.encode(buf);

        // 5GMM cause (mandatory)
        self.mm_cause.encode(buf);

        // Optional IEs
        if let Some(value) = self.t3346_value {
            buf.put_u8(registration_reject_iei::T3346_VALUE);
            buf.put_u8(1);
            buf.put_u8(value);
        }

        if let Some(value) = self.t3502_value {
            buf.put_u8(registration_reject_iei::T3502_VALUE);
            buf.put_u8(1);
            buf.put_u8(value);
        }

        if let Some(ref eap) = self.eap_message {
            buf.put_u8(registration_reject_iei::EAP_MESSAGE);
            buf.put_u16(eap.len() as u16);
            buf.put_slice(eap);
        }

        if let Some(ref nssai) = self.rejected_nssai {
            buf.put_u8(registration_reject_iei::REJECTED_NSSAI);
            buf.put_u8(nssai.len() as u8);
            buf.put_slice(nssai);
        }
    }

    /// Get the message type
    pub fn message_type() -> MmMessageType {
        MmMessageType::RegistrationReject
    }
}

// ============================================================================
// Registration Complete Message (3GPP TS 24.501 Section 8.2.9)
// ============================================================================

/// IEI values for Registration Complete optional IEs
mod registration_complete_iei {
    /// SOR transparent container
    pub const SOR_TRANSPARENT_CONTAINER: u8 = 0x73;
}

/// Registration Complete message (UE to network)
///
/// 3GPP TS 24.501 Section 8.2.9
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RegistrationComplete {
    /// SOR transparent container (optional, Type 6, IEI 0x73)
    pub sor_transparent_container: Option<Vec<u8>>,
}

impl RegistrationComplete {
    /// Create a new Registration Complete
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a Registration Complete with SOR transparent container
    pub fn with_sor_container(container: Vec<u8>) -> Self {
        Self {
            sor_transparent_container: Some(container),
        }
    }

    /// Decode from bytes (after header has been parsed)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, RegistrationError> {
        let mut msg = Self::new();

        // Parse optional IEs
        while buf.remaining() > 0 {
            let iei = buf.chunk()[0];

            match iei {
                registration_complete_iei::SOR_TRANSPARENT_CONTAINER => {
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
                    msg.sor_transparent_container = Some(data);
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
        let header = PlainMmHeader::new(MmMessageType::RegistrationComplete);
        header.encode(buf);

        // Optional IEs
        if let Some(ref container) = self.sor_transparent_container {
            buf.put_u8(registration_complete_iei::SOR_TRANSPARENT_CONTAINER);
            buf.put_u16(container.len() as u16);
            buf.put_slice(container);
        }
    }

    /// Get the message type
    pub fn message_type() -> MmMessageType {
        MmMessageType::RegistrationComplete
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ies::ie1::{FollowOnRequest, RegistrationType};

    #[test]
    fn test_mobile_identity_no_identity() {
        let mi = Ie5gsMobileIdentity::no_identity();
        assert_eq!(mi.identity_type, MobileIdentityType::NoIdentity);

        let mut buf = Vec::new();
        mi.encode(&mut buf);

        let decoded = Ie5gsMobileIdentity::decode(&mut buf.as_slice()).unwrap();
        assert_eq!(decoded.identity_type, MobileIdentityType::NoIdentity);
    }

    #[test]
    fn test_5gmm_cause_encode_decode() {
        let cause = Ie5gMmCause::new(MmCause::IllegalUe);

        let mut buf = Vec::new();
        cause.encode(&mut buf);
        assert_eq!(buf, vec![3]);

        let decoded = Ie5gMmCause::decode(&mut buf.as_slice()).unwrap();
        assert_eq!(decoded.value, MmCause::IllegalUe);
    }

    #[test]
    fn test_registration_result_encode_decode() {
        let result = Ie5gsRegistrationResult::new(
            SmsOverNasAllowed::Allowed,
            RegistrationResultValue::ThreeGppAccess,
        );

        let mut buf = Vec::new();
        result.encode(&mut buf);
        assert_eq!(buf, vec![1, 0b00001001]); // Length=1, SMS allowed (bit 3) + 3GPP access (bits 2-0)

        let decoded = Ie5gsRegistrationResult::decode(&mut buf.as_slice()).unwrap();
        assert_eq!(decoded.sms_allowed, SmsOverNasAllowed::Allowed);
        assert_eq!(decoded.result, RegistrationResultValue::ThreeGppAccess);
    }

    #[test]
    fn test_registration_request_basic() {
        let reg_type = Ie5gsRegistrationType::new(
            FollowOnRequest::NoPending,
            RegistrationType::InitialRegistration,
        );
        let ng_ksi = NasKeySetIdentifier::no_key();
        let mobile_id = Ie5gsMobileIdentity::no_identity();

        let msg = RegistrationRequest::new(reg_type, ng_ksi, mobile_id);

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Verify header
        assert_eq!(buf[0], 0x7E); // EPD
        assert_eq!(buf[1], 0x00); // Security header type
        assert_eq!(buf[2], 0x41); // Message type (Registration Request)

        // Verify first octet after header: ngKSI (high) + reg type (low)
        assert_eq!(buf[3], 0x71); // ngKSI=7 (no key), reg_type=1 (initial)
    }

    #[test]
    fn test_registration_request_encode_decode() {
        let reg_type = Ie5gsRegistrationType::new(
            FollowOnRequest::Pending,
            RegistrationType::MobilityRegistrationUpdating,
        );
        let ng_ksi = NasKeySetIdentifier::new(crate::security::SecurityContextType::Native, 3);
        let mobile_id = Ie5gsMobileIdentity::new(MobileIdentityType::Suci, vec![0x01, 0x02, 0x03]);

        let mut msg = RegistrationRequest::new(reg_type, ng_ksi, mobile_id);
        msg.ue_security_capability = Some(vec![0xE0, 0xE0]); // EA0, EA1, EA2 + IA0, IA1, IA2

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Skip header (3 bytes) and decode
        let decoded = RegistrationRequest::decode(&mut buf[3..].as_ref()).unwrap();

        assert_eq!(decoded.registration_type.registration_type, RegistrationType::MobilityRegistrationUpdating);
        assert_eq!(decoded.registration_type.follow_on_request_pending, FollowOnRequest::Pending);
        assert_eq!(decoded.ng_ksi.ksi, 3);
        assert_eq!(decoded.mobile_identity.identity_type, MobileIdentityType::Suci);
        assert_eq!(decoded.ue_security_capability, Some(vec![0xE0, 0xE0]));
    }

    #[test]
    fn test_registration_accept_basic() {
        let result = Ie5gsRegistrationResult::new(
            SmsOverNasAllowed::NotAllowed,
            RegistrationResultValue::ThreeGppAccess,
        );
        let msg = RegistrationAccept::new(result);

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Verify header
        assert_eq!(buf[0], 0x7E); // EPD
        assert_eq!(buf[1], 0x00); // Security header type
        assert_eq!(buf[2], 0x42); // Message type (Registration Accept)
    }

    #[test]
    fn test_registration_accept_encode_decode() {
        let result = Ie5gsRegistrationResult::new(
            SmsOverNasAllowed::Allowed,
            RegistrationResultValue::ThreeGppAndNonThreeGppAccess,
        );
        let mut msg = RegistrationAccept::new(result);
        msg.t3512_value = Some(0x21); // Timer value
        msg.allowed_nssai = Some(vec![0x01, 0x02]); // Sample NSSAI

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Skip header (3 bytes) and decode
        let decoded = RegistrationAccept::decode(&mut buf[3..].as_ref()).unwrap();

        assert_eq!(decoded.registration_result.sms_allowed, SmsOverNasAllowed::Allowed);
        assert_eq!(decoded.registration_result.result, RegistrationResultValue::ThreeGppAndNonThreeGppAccess);
        assert_eq!(decoded.t3512_value, Some(0x21));
        assert_eq!(decoded.allowed_nssai, Some(vec![0x01, 0x02]));
    }

    #[test]
    fn test_registration_reject_basic() {
        let msg = RegistrationReject::with_cause(MmCause::PlmnNotAllowed);

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Verify header
        assert_eq!(buf[0], 0x7E); // EPD
        assert_eq!(buf[1], 0x00); // Security header type
        assert_eq!(buf[2], 0x44); // Message type (Registration Reject)
        assert_eq!(buf[3], 11); // 5GMM cause (PLMN not allowed)
    }

    #[test]
    fn test_registration_reject_encode_decode() {
        let mut msg = RegistrationReject::with_cause(MmCause::Congestion);
        msg.t3346_value = Some(0x10);

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Skip header (3 bytes) and decode
        let decoded = RegistrationReject::decode(&mut buf[3..].as_ref()).unwrap();

        assert_eq!(decoded.mm_cause.value, MmCause::Congestion);
        assert_eq!(decoded.t3346_value, Some(0x10));
    }

    #[test]
    fn test_registration_complete_basic() {
        let msg = RegistrationComplete::new();

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Verify header
        assert_eq!(buf[0], 0x7E); // EPD
        assert_eq!(buf[1], 0x00); // Security header type
        assert_eq!(buf[2], 0x43); // Message type (Registration Complete)
        assert_eq!(buf.len(), 3); // No optional IEs
    }

    #[test]
    fn test_registration_complete_with_sor() {
        let msg = RegistrationComplete::with_sor_container(vec![0x01, 0x02, 0x03, 0x04]);

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Skip header (3 bytes) and decode
        let decoded = RegistrationComplete::decode(&mut buf[3..].as_ref()).unwrap();

        assert_eq!(decoded.sor_transparent_container, Some(vec![0x01, 0x02, 0x03, 0x04]));
    }
}
