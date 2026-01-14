//! Type 1 Information Elements (half-octet)
//!
//! Type 1 IEs occupy only 4 bits (half an octet) and are used for
//! compact encoding of small values in NAS messages.
//!
//! Based on 3GPP TS 24.501 specification.

use num_enum::{IntoPrimitive, TryFromPrimitive};

/// Error type for Type 1 IE decoding
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum Ie1Error {
    /// Invalid value for the IE type
    #[error("Invalid value 0x{0:X} for {1}")]
    InvalidValue(u8, &'static str),
}

// ============================================================================
// Enumerations for Type 1 IEs
// ============================================================================

/// 5GS Identity Type (3GPP TS 24.501 Section 9.11.3.3)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum IdentityType {
    /// No identity
    #[default]
    NoIdentity = 0b000,
    /// SUCI (Subscription Concealed Identifier)
    Suci = 0b001,
    /// 5G-GUTI (5G Globally Unique Temporary Identifier)
    Guti = 0b010,
    /// IMEI (International Mobile Equipment Identity)
    Imei = 0b011,
    /// 5G-S-TMSI (5G S-Temporary Mobile Subscriber Identity)
    Tmsi = 0b100,
    /// IMEISV (IMEI Software Version)
    ImeiSv = 0b101,
}

/// Follow-on Request indicator (3GPP TS 24.501 Section 9.11.3.8)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum FollowOnRequest {
    /// No follow-on request pending
    #[default]
    NoPending = 0b0,
    /// Follow-on request pending
    Pending = 0b1,
}

/// 5GS Registration Type (3GPP TS 24.501 Section 9.11.3.7)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum RegistrationType {
    /// Initial registration
    #[default]
    InitialRegistration = 0b001,
    /// Mobility registration updating
    MobilityRegistrationUpdating = 0b010,
    /// Periodic registration updating
    PeriodicRegistrationUpdating = 0b011,
    /// Emergency registration
    EmergencyRegistration = 0b100,
}

/// Access Type (3GPP TS 24.501 Section 9.11.2.1A)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum AccessType {
    /// 3GPP access
    #[default]
    ThreeGppAccess = 0b01,
    /// Non-3GPP access
    NonThreeGppAccess = 0b10,
}

/// SSC mode 1 allowed indicator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum Ssc1 {
    /// SSC mode 1 not allowed
    #[default]
    NotAllowed = 0b0,
    /// SSC mode 1 allowed
    Allowed = 0b1,
}

/// SSC mode 2 allowed indicator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum Ssc2 {
    /// SSC mode 2 not allowed
    #[default]
    NotAllowed = 0b0,
    /// SSC mode 2 allowed
    Allowed = 0b1,
}

/// SSC mode 3 allowed indicator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum Ssc3 {
    /// SSC mode 3 not allowed
    #[default]
    NotAllowed = 0b0,
    /// SSC mode 3 allowed
    Allowed = 0b1,
}

/// Always-on PDU session indication (3GPP TS 24.501 Section 9.11.4.3)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum AlwaysOnPduSessionIndication {
    /// Always-on PDU session not allowed
    #[default]
    NotAllowed = 0b0,
    /// Always-on PDU session required
    Required = 0b1,
}

/// Always-on PDU session requested (3GPP TS 24.501 Section 9.11.4.4)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum AlwaysOnPduSessionRequested {
    /// Always-on PDU session not requested
    #[default]
    NotRequested = 0b0,
    /// Always-on PDU session requested
    Requested = 0b1,
}

/// Acknowledgement indicator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum Acknowledgement {
    /// Acknowledgement not requested
    #[default]
    NotRequested = 0b0,
    /// Acknowledgement requested
    Requested = 0b1,
}

/// Registration requested indicator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum RegistrationRequested {
    /// Registration not requested
    #[default]
    NotRequested = 0b0,
    /// Registration requested
    Requested = 0b1,
}

/// De-registration access type (3GPP TS 24.501 Section 9.11.3.20)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum DeRegistrationAccessType {
    /// 3GPP access
    #[default]
    ThreeGppAccess = 0b01,
    /// Non-3GPP access
    NonThreeGppAccess = 0b10,
    /// 3GPP access and non-3GPP access
    ThreeGppAndNonThreeGppAccess = 0b11,
}

/// Re-registration required indicator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum ReRegistrationRequired {
    /// Re-registration not required
    #[default]
    NotRequired = 0b0,
    /// Re-registration required
    Required = 0b1,
}

/// Switch off indicator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum SwitchOff {
    /// Normal de-registration
    #[default]
    NormalDeRegistration = 0b0,
    /// Switch off
    SwitchOff = 0b1,
}

/// IMEISV request (3GPP TS 24.501 Section 9.11.3.28)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum ImeiSvRequest {
    /// IMEISV not requested
    #[default]
    NotRequested = 0b000,
    /// IMEISV requested
    Requested = 0b001,
}

/// Type of security context (3GPP TS 24.501 Section 9.11.3.32)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum TypeOfSecurityContext {
    /// Native security context
    #[default]
    NativeSecurityContext = 0b0,
    /// Mapped security context
    MappedSecurityContext = 0b1,
}

/// Registration area allocation indication
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum RegistrationAreaAllocationIndication {
    /// Registration area not allocated
    #[default]
    NotAllocated = 0b0,
    /// Registration area allocated
    Allocated = 0b1,
}

/// Network slicing subscription change indication
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum NetworkSlicingSubscriptionChangeIndication {
    /// Network slicing subscription not changed
    #[default]
    NotChanged = 0b0,
    /// Network slicing subscription changed
    Changed = 0b1,
}

/// Default configured NSSAI indication
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum DefaultConfiguredNssaiIndication {
    /// Not created from default configured NSSAI
    #[default]
    NotCreatedFromDefault = 0b0,
    /// Created from default configured NSSAI
    CreatedFromDefault = 0b1,
}

/// NSSAI inclusion mode (3GPP TS 24.501 Section 9.11.3.37A)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum NssaiInclusionMode {
    /// Mode A
    #[default]
    A = 0b00,
    /// Mode B
    B = 0b01,
    /// Mode C
    C = 0b10,
    /// Mode D
    D = 0b11,
}

/// Payload container type (3GPP TS 24.501 Section 9.11.3.40)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum PayloadContainerType {
    /// N1 SM information
    #[default]
    N1SmInformation = 0b0001,
    /// SMS
    Sms = 0b0010,
    /// LPP message
    LppMessage = 0b0011,
    /// SOR transparent container
    SorTransparentContainer = 0b0100,
    /// UE policy container
    UePolicyContainer = 0b0101,
    /// UE parameters update transparent container
    UeParametersUpdateTransparentContainer = 0b0110,
    /// Multiple payloads
    MultiplePayloads = 0b1111,
}

/// PDU session type (3GPP TS 24.501 Section 9.11.4.11)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum PduSessionType {
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

/// Request type (3GPP TS 24.501 Section 9.11.3.47)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum RequestType {
    /// Initial request
    #[default]
    InitialRequest = 0b001,
    /// Existing PDU session
    ExistingPduSession = 0b010,
    /// Initial emergency request
    InitialEmergencyRequest = 0b011,
    /// Existing emergency PDU session
    ExistingEmergencyPduSession = 0b100,
    /// Modification request
    ModificationRequest = 0b101,
}

/// Service type (3GPP TS 24.501 Section 9.11.3.50)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum ServiceType {
    /// Signalling
    #[default]
    Signalling = 0b0000,
    /// Data
    Data = 0b0001,
    /// Mobile terminated services
    MobileTerminatedServices = 0b0010,
    /// Emergency services
    EmergencyServices = 0b0011,
    /// Emergency services fallback
    EmergencyServicesFallback = 0b0100,
    /// High priority access
    HighPriorityAccess = 0b0101,
    /// Elevated signalling
    ElevatedSignalling = 0b0110,
}

/// SMS availability indication
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum SmsAvailabilityIndication {
    /// SMS not available
    #[default]
    NotAvailable = 0b0,
    /// SMS available
    Available = 0b1,
}

/// SSC mode (3GPP TS 24.501 Section 9.11.4.16)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum SscMode {
    /// SSC mode 1
    #[default]
    SscMode1 = 0b001,
    /// SSC mode 2
    SscMode2 = 0b010,
    /// SSC mode 3
    SscMode3 = 0b011,
}


// ============================================================================
// Type 1 IE Structures
// ============================================================================

/// Trait for Type 1 Information Elements (half-octet)
pub trait InformationElement1: Sized {
    /// Decode from a 4-bit value (lower nibble)
    fn decode(val: u8) -> Result<Self, Ie1Error>;

    /// Encode to a 4-bit value (lower nibble)
    fn encode(&self) -> u8;
}

/// 5GS Identity Type IE (3GPP TS 24.501 Section 9.11.3.3)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Ie5gsIdentityType {
    /// Identity type value
    pub value: IdentityType,
}

impl Ie5gsIdentityType {
    /// Create a new 5GS Identity Type IE
    pub fn new(value: IdentityType) -> Self {
        Self { value }
    }
}

impl InformationElement1 for Ie5gsIdentityType {
    fn decode(val: u8) -> Result<Self, Ie1Error> {
        let value = IdentityType::try_from(val & 0x07)
            .map_err(|_| Ie1Error::InvalidValue(val, "IdentityType"))?;
        Ok(Self { value })
    }

    fn encode(&self) -> u8 {
        self.value.into()
    }
}

/// 5GS Registration Type IE (3GPP TS 24.501 Section 9.11.3.7)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Ie5gsRegistrationType {
    /// Follow-on request pending indicator
    pub follow_on_request_pending: FollowOnRequest,
    /// Registration type
    pub registration_type: RegistrationType,
}

impl Ie5gsRegistrationType {
    /// Create a new 5GS Registration Type IE
    pub fn new(follow_on_request_pending: FollowOnRequest, registration_type: RegistrationType) -> Self {
        Self {
            follow_on_request_pending,
            registration_type,
        }
    }
}

impl InformationElement1 for Ie5gsRegistrationType {
    fn decode(val: u8) -> Result<Self, Ie1Error> {
        // Bit 3: follow-on request, Bits 2-0: registration type
        let follow_on_request_pending = FollowOnRequest::try_from((val >> 3) & 0x01)
            .map_err(|_| Ie1Error::InvalidValue(val, "FollowOnRequest"))?;
        let registration_type = RegistrationType::try_from(val & 0x07)
            .map_err(|_| Ie1Error::InvalidValue(val, "RegistrationType"))?;
        Ok(Self {
            follow_on_request_pending,
            registration_type,
        })
    }

    fn encode(&self) -> u8 {
        let for_val: u8 = self.follow_on_request_pending.into();
        let reg_val: u8 = self.registration_type.into();
        (for_val << 3) | (reg_val & 0x07)
    }
}

/// Access Type IE (3GPP TS 24.501 Section 9.11.2.1A)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IeAccessType {
    /// Access type value
    pub value: AccessType,
}

impl IeAccessType {
    /// Create a new Access Type IE
    pub fn new(value: AccessType) -> Self {
        Self { value }
    }
}

impl InformationElement1 for IeAccessType {
    fn decode(val: u8) -> Result<Self, Ie1Error> {
        let value = AccessType::try_from(val & 0x03)
            .map_err(|_| Ie1Error::InvalidValue(val, "AccessType"))?;
        Ok(Self { value })
    }

    fn encode(&self) -> u8 {
        self.value.into()
    }
}

/// Allowed SSC Mode IE (3GPP TS 24.501 Section 9.11.4.5)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IeAllowedSscMode {
    /// SSC mode 1 allowed
    pub ssc1: Ssc1,
    /// SSC mode 2 allowed
    pub ssc2: Ssc2,
    /// SSC mode 3 allowed
    pub ssc3: Ssc3,
}

impl IeAllowedSscMode {
    /// Create a new Allowed SSC Mode IE
    pub fn new(ssc1: Ssc1, ssc2: Ssc2, ssc3: Ssc3) -> Self {
        Self { ssc1, ssc2, ssc3 }
    }
}

impl InformationElement1 for IeAllowedSscMode {
    fn decode(val: u8) -> Result<Self, Ie1Error> {
        // Bit 0: SSC1, Bit 1: SSC2, Bit 2: SSC3, Bit 3: spare
        let ssc1 = Ssc1::try_from(val & 0x01)
            .map_err(|_| Ie1Error::InvalidValue(val, "Ssc1"))?;
        let ssc2 = Ssc2::try_from((val >> 1) & 0x01)
            .map_err(|_| Ie1Error::InvalidValue(val, "Ssc2"))?;
        let ssc3 = Ssc3::try_from((val >> 2) & 0x01)
            .map_err(|_| Ie1Error::InvalidValue(val, "Ssc3"))?;
        Ok(Self { ssc1, ssc2, ssc3 })
    }

    fn encode(&self) -> u8 {
        let ssc1_val: u8 = self.ssc1.into();
        let ssc2_val: u8 = self.ssc2.into();
        let ssc3_val: u8 = self.ssc3.into();
        ssc1_val | (ssc2_val << 1) | (ssc3_val << 2)
    }
}

/// Always-on PDU Session Indication IE (3GPP TS 24.501 Section 9.11.4.3)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IeAlwaysOnPduSessionIndication {
    /// Always-on PDU session indication value
    pub value: AlwaysOnPduSessionIndication,
}

impl IeAlwaysOnPduSessionIndication {
    /// Create a new Always-on PDU Session Indication IE
    pub fn new(value: AlwaysOnPduSessionIndication) -> Self {
        Self { value }
    }
}

impl InformationElement1 for IeAlwaysOnPduSessionIndication {
    fn decode(val: u8) -> Result<Self, Ie1Error> {
        let value = AlwaysOnPduSessionIndication::try_from(val & 0x01)
            .map_err(|_| Ie1Error::InvalidValue(val, "AlwaysOnPduSessionIndication"))?;
        Ok(Self { value })
    }

    fn encode(&self) -> u8 {
        self.value.into()
    }
}

/// Always-on PDU Session Requested IE (3GPP TS 24.501 Section 9.11.4.4)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IeAlwaysOnPduSessionRequested {
    /// Always-on PDU session requested value
    pub value: AlwaysOnPduSessionRequested,
}

impl IeAlwaysOnPduSessionRequested {
    /// Create a new Always-on PDU Session Requested IE
    pub fn new(value: AlwaysOnPduSessionRequested) -> Self {
        Self { value }
    }
}

impl InformationElement1 for IeAlwaysOnPduSessionRequested {
    fn decode(val: u8) -> Result<Self, Ie1Error> {
        let value = AlwaysOnPduSessionRequested::try_from(val & 0x01)
            .map_err(|_| Ie1Error::InvalidValue(val, "AlwaysOnPduSessionRequested"))?;
        Ok(Self { value })
    }

    fn encode(&self) -> u8 {
        self.value.into()
    }
}

/// Configuration Update Indication IE (3GPP TS 24.501 Section 9.11.3.18)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IeConfigurationUpdateIndication {
    /// Acknowledgement indicator
    pub ack: Acknowledgement,
    /// Registration requested indicator
    pub red: RegistrationRequested,
}

impl IeConfigurationUpdateIndication {
    /// Create a new Configuration Update Indication IE
    pub fn new(ack: Acknowledgement, red: RegistrationRequested) -> Self {
        Self { ack, red }
    }
}

impl InformationElement1 for IeConfigurationUpdateIndication {
    fn decode(val: u8) -> Result<Self, Ie1Error> {
        // Bit 0: ACK, Bit 1: RED, Bits 2-3: spare
        let ack = Acknowledgement::try_from(val & 0x01)
            .map_err(|_| Ie1Error::InvalidValue(val, "Acknowledgement"))?;
        let red = RegistrationRequested::try_from((val >> 1) & 0x01)
            .map_err(|_| Ie1Error::InvalidValue(val, "RegistrationRequested"))?;
        Ok(Self { ack, red })
    }

    fn encode(&self) -> u8 {
        let ack_val: u8 = self.ack.into();
        let red_val: u8 = self.red.into();
        ack_val | (red_val << 1)
    }
}

/// De-registration Type IE (3GPP TS 24.501 Section 9.11.3.20)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IeDeRegistrationType {
    /// Access type
    pub access_type: DeRegistrationAccessType,
    /// Re-registration required (spare in UE to Network direction)
    pub re_registration_required: ReRegistrationRequired,
    /// Switch off indicator
    pub switch_off: SwitchOff,
}

impl IeDeRegistrationType {
    /// Create a new De-registration Type IE
    pub fn new(
        access_type: DeRegistrationAccessType,
        re_registration_required: ReRegistrationRequired,
        switch_off: SwitchOff,
    ) -> Self {
        Self {
            access_type,
            re_registration_required,
            switch_off,
        }
    }
}

impl InformationElement1 for IeDeRegistrationType {
    fn decode(val: u8) -> Result<Self, Ie1Error> {
        // Bits 1-0: access type, Bit 2: re-registration required, Bit 3: switch off
        let access_type = DeRegistrationAccessType::try_from(val & 0x03)
            .map_err(|_| Ie1Error::InvalidValue(val, "DeRegistrationAccessType"))?;
        let re_registration_required = ReRegistrationRequired::try_from((val >> 2) & 0x01)
            .map_err(|_| Ie1Error::InvalidValue(val, "ReRegistrationRequired"))?;
        let switch_off = SwitchOff::try_from((val >> 3) & 0x01)
            .map_err(|_| Ie1Error::InvalidValue(val, "SwitchOff"))?;
        Ok(Self {
            access_type,
            re_registration_required,
            switch_off,
        })
    }

    fn encode(&self) -> u8 {
        let access_val: u8 = self.access_type.into();
        let rereg_val: u8 = self.re_registration_required.into();
        let switch_val: u8 = self.switch_off.into();
        access_val | (rereg_val << 2) | (switch_val << 3)
    }
}

/// IMEISV Request IE (3GPP TS 24.501 Section 9.11.3.28)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IeImeiSvRequest {
    /// IMEISV request value
    pub imeisv_request: ImeiSvRequest,
}

impl IeImeiSvRequest {
    /// Create a new IMEISV Request IE
    pub fn new(imeisv_request: ImeiSvRequest) -> Self {
        Self { imeisv_request }
    }
}

impl InformationElement1 for IeImeiSvRequest {
    fn decode(val: u8) -> Result<Self, Ie1Error> {
        let imeisv_request = ImeiSvRequest::try_from(val & 0x07)
            .map_err(|_| Ie1Error::InvalidValue(val, "ImeiSvRequest"))?;
        Ok(Self { imeisv_request })
    }

    fn encode(&self) -> u8 {
        self.imeisv_request.into()
    }
}

/// MICO Indication IE (3GPP TS 24.501 Section 9.11.3.31)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IeMicoIndication {
    /// Registration area allocation indication
    pub raai: RegistrationAreaAllocationIndication,
}

impl IeMicoIndication {
    /// Create a new MICO Indication IE
    pub fn new(raai: RegistrationAreaAllocationIndication) -> Self {
        Self { raai }
    }
}

impl InformationElement1 for IeMicoIndication {
    fn decode(val: u8) -> Result<Self, Ie1Error> {
        let raai = RegistrationAreaAllocationIndication::try_from(val & 0x01)
            .map_err(|_| Ie1Error::InvalidValue(val, "RegistrationAreaAllocationIndication"))?;
        Ok(Self { raai })
    }

    fn encode(&self) -> u8 {
        self.raai.into()
    }
}


/// NAS Key Set Identifier IE (3GPP TS 24.501 Section 9.11.3.32)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IeNasKeySetIdentifier {
    /// Type of security context
    pub tsc: TypeOfSecurityContext,
    /// Key set identifier (0-6, 7 = not available or reserved)
    pub ksi: u8,
}

impl IeNasKeySetIdentifier {
    /// Value indicating NAS key set identifier is not available or reserved
    pub const NOT_AVAILABLE_OR_RESERVED: u8 = 0b111;

    /// Create a new NAS Key Set Identifier IE
    pub fn new(tsc: TypeOfSecurityContext, ksi: u8) -> Self {
        Self { tsc, ksi: ksi & 0x07 }
    }

    /// Create a NAS Key Set Identifier indicating not available
    pub fn not_available() -> Self {
        Self {
            tsc: TypeOfSecurityContext::NativeSecurityContext,
            ksi: Self::NOT_AVAILABLE_OR_RESERVED,
        }
    }

    /// Check if the key set identifier is available
    pub fn is_available(&self) -> bool {
        self.ksi != Self::NOT_AVAILABLE_OR_RESERVED
    }
}

impl Default for IeNasKeySetIdentifier {
    fn default() -> Self {
        Self::not_available()
    }
}

impl InformationElement1 for IeNasKeySetIdentifier {
    fn decode(val: u8) -> Result<Self, Ie1Error> {
        // Bit 3: TSC, Bits 2-0: KSI
        let tsc = TypeOfSecurityContext::try_from((val >> 3) & 0x01)
            .map_err(|_| Ie1Error::InvalidValue(val, "TypeOfSecurityContext"))?;
        let ksi = val & 0x07;
        Ok(Self { tsc, ksi })
    }

    fn encode(&self) -> u8 {
        let tsc_val: u8 = self.tsc.into();
        (tsc_val << 3) | (self.ksi & 0x07)
    }
}

/// Network Slicing Indication IE (3GPP TS 24.501 Section 9.11.3.36)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IeNetworkSlicingIndication {
    /// Network slicing subscription change indication (spare if UE->NW)
    pub nssci: NetworkSlicingSubscriptionChangeIndication,
    /// Default configured NSSAI indication (spare if NW->UE)
    pub dcni: DefaultConfiguredNssaiIndication,
}

impl IeNetworkSlicingIndication {
    /// Create a new Network Slicing Indication IE
    pub fn new(
        nssci: NetworkSlicingSubscriptionChangeIndication,
        dcni: DefaultConfiguredNssaiIndication,
    ) -> Self {
        Self { nssci, dcni }
    }
}

impl InformationElement1 for IeNetworkSlicingIndication {
    fn decode(val: u8) -> Result<Self, Ie1Error> {
        // Bit 0: NSSCI, Bit 1: DCNI, Bits 2-3: spare
        let nssci = NetworkSlicingSubscriptionChangeIndication::try_from(val & 0x01)
            .map_err(|_| Ie1Error::InvalidValue(val, "NetworkSlicingSubscriptionChangeIndication"))?;
        let dcni = DefaultConfiguredNssaiIndication::try_from((val >> 1) & 0x01)
            .map_err(|_| Ie1Error::InvalidValue(val, "DefaultConfiguredNssaiIndication"))?;
        Ok(Self { nssci, dcni })
    }

    fn encode(&self) -> u8 {
        let nssci_val: u8 = self.nssci.into();
        let dcni_val: u8 = self.dcni.into();
        nssci_val | (dcni_val << 1)
    }
}

/// NSSAI Inclusion Mode IE (3GPP TS 24.501 Section 9.11.3.37A)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IeNssaiInclusionMode {
    /// NSSAI inclusion mode value
    pub nssai_inclusion_mode: NssaiInclusionMode,
}

impl IeNssaiInclusionMode {
    /// Create a new NSSAI Inclusion Mode IE
    pub fn new(nssai_inclusion_mode: NssaiInclusionMode) -> Self {
        Self { nssai_inclusion_mode }
    }
}

impl InformationElement1 for IeNssaiInclusionMode {
    fn decode(val: u8) -> Result<Self, Ie1Error> {
        let nssai_inclusion_mode = NssaiInclusionMode::try_from(val & 0x03)
            .map_err(|_| Ie1Error::InvalidValue(val, "NssaiInclusionMode"))?;
        Ok(Self { nssai_inclusion_mode })
    }

    fn encode(&self) -> u8 {
        self.nssai_inclusion_mode.into()
    }
}

/// Payload Container Type IE (3GPP TS 24.501 Section 9.11.3.40)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IePayloadContainerType {
    /// Payload container type value
    pub payload_container_type: PayloadContainerType,
}

impl IePayloadContainerType {
    /// Create a new Payload Container Type IE
    pub fn new(payload_container_type: PayloadContainerType) -> Self {
        Self { payload_container_type }
    }
}

impl InformationElement1 for IePayloadContainerType {
    fn decode(val: u8) -> Result<Self, Ie1Error> {
        let payload_container_type = PayloadContainerType::try_from(val & 0x0F)
            .map_err(|_| Ie1Error::InvalidValue(val, "PayloadContainerType"))?;
        Ok(Self { payload_container_type })
    }

    fn encode(&self) -> u8 {
        self.payload_container_type.into()
    }
}

/// PDU Session Type IE (3GPP TS 24.501 Section 9.11.4.11)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IePduSessionType {
    /// PDU session type value
    pub pdu_session_type: PduSessionType,
}

impl IePduSessionType {
    /// Create a new PDU Session Type IE
    pub fn new(pdu_session_type: PduSessionType) -> Self {
        Self { pdu_session_type }
    }
}

impl InformationElement1 for IePduSessionType {
    fn decode(val: u8) -> Result<Self, Ie1Error> {
        let pdu_session_type = PduSessionType::try_from(val & 0x07)
            .map_err(|_| Ie1Error::InvalidValue(val, "PduSessionType"))?;
        Ok(Self { pdu_session_type })
    }

    fn encode(&self) -> u8 {
        self.pdu_session_type.into()
    }
}

/// Request Type IE (3GPP TS 24.501 Section 9.11.3.47)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IeRequestType {
    /// Request type value
    pub request_type: RequestType,
}

impl IeRequestType {
    /// Create a new Request Type IE
    pub fn new(request_type: RequestType) -> Self {
        Self { request_type }
    }
}

impl InformationElement1 for IeRequestType {
    fn decode(val: u8) -> Result<Self, Ie1Error> {
        let request_type = RequestType::try_from(val & 0x07)
            .map_err(|_| Ie1Error::InvalidValue(val, "RequestType"))?;
        Ok(Self { request_type })
    }

    fn encode(&self) -> u8 {
        self.request_type.into()
    }
}

/// Service Type IE (3GPP TS 24.501 Section 9.11.3.50)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IeServiceType {
    /// Service type value
    pub service_type: ServiceType,
}

impl IeServiceType {
    /// Create a new Service Type IE
    pub fn new(service_type: ServiceType) -> Self {
        Self { service_type }
    }
}

impl InformationElement1 for IeServiceType {
    fn decode(val: u8) -> Result<Self, Ie1Error> {
        let service_type = ServiceType::try_from(val & 0x0F)
            .map_err(|_| Ie1Error::InvalidValue(val, "ServiceType"))?;
        Ok(Self { service_type })
    }

    fn encode(&self) -> u8 {
        self.service_type.into()
    }
}

/// SMS Indication IE (3GPP TS 24.501 Section 9.11.3.50A)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IeSmsIndication {
    /// SMS availability indication
    pub sai: SmsAvailabilityIndication,
}

impl IeSmsIndication {
    /// Create a new SMS Indication IE
    pub fn new(sai: SmsAvailabilityIndication) -> Self {
        Self { sai }
    }
}

impl InformationElement1 for IeSmsIndication {
    fn decode(val: u8) -> Result<Self, Ie1Error> {
        let sai = SmsAvailabilityIndication::try_from(val & 0x01)
            .map_err(|_| Ie1Error::InvalidValue(val, "SmsAvailabilityIndication"))?;
        Ok(Self { sai })
    }

    fn encode(&self) -> u8 {
        self.sai.into()
    }
}

/// SSC Mode IE (3GPP TS 24.501 Section 9.11.4.16)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IeSscMode {
    /// SSC mode value
    pub ssc_mode: SscMode,
}

impl IeSscMode {
    /// Create a new SSC Mode IE
    pub fn new(ssc_mode: SscMode) -> Self {
        Self { ssc_mode }
    }
}

impl InformationElement1 for IeSscMode {
    fn decode(val: u8) -> Result<Self, Ie1Error> {
        let ssc_mode = SscMode::try_from(val & 0x07)
            .map_err(|_| Ie1Error::InvalidValue(val, "SscMode"))?;
        Ok(Self { ssc_mode })
    }

    fn encode(&self) -> u8 {
        self.ssc_mode.into()
    }
}


// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_type_values() {
        assert_eq!(u8::from(IdentityType::NoIdentity), 0b000);
        assert_eq!(u8::from(IdentityType::Suci), 0b001);
        assert_eq!(u8::from(IdentityType::Guti), 0b010);
        assert_eq!(u8::from(IdentityType::Imei), 0b011);
        assert_eq!(u8::from(IdentityType::Tmsi), 0b100);
        assert_eq!(u8::from(IdentityType::ImeiSv), 0b101);
    }

    #[test]
    fn test_ie5gs_identity_type_encode_decode() {
        let ie = Ie5gsIdentityType::new(IdentityType::Suci);
        let encoded = ie.encode();
        assert_eq!(encoded, 0b001);

        let decoded = Ie5gsIdentityType::decode(encoded).unwrap();
        assert_eq!(decoded.value, IdentityType::Suci);
    }

    #[test]
    fn test_ie5gs_registration_type_encode_decode() {
        let ie = Ie5gsRegistrationType::new(
            FollowOnRequest::Pending,
            RegistrationType::InitialRegistration,
        );
        let encoded = ie.encode();
        assert_eq!(encoded, 0b1001); // FOR=1, RegType=001

        let decoded = Ie5gsRegistrationType::decode(encoded).unwrap();
        assert_eq!(decoded.follow_on_request_pending, FollowOnRequest::Pending);
        assert_eq!(decoded.registration_type, RegistrationType::InitialRegistration);
    }

    #[test]
    fn test_ie5gs_registration_type_all_types() {
        for reg_type in [
            RegistrationType::InitialRegistration,
            RegistrationType::MobilityRegistrationUpdating,
            RegistrationType::PeriodicRegistrationUpdating,
            RegistrationType::EmergencyRegistration,
        ] {
            for for_req in [FollowOnRequest::NoPending, FollowOnRequest::Pending] {
                let ie = Ie5gsRegistrationType::new(for_req, reg_type);
                let encoded = ie.encode();
                let decoded = Ie5gsRegistrationType::decode(encoded).unwrap();
                assert_eq!(decoded.follow_on_request_pending, for_req);
                assert_eq!(decoded.registration_type, reg_type);
            }
        }
    }

    #[test]
    fn test_ie_access_type_encode_decode() {
        let ie = IeAccessType::new(AccessType::ThreeGppAccess);
        let encoded = ie.encode();
        assert_eq!(encoded, 0b01);

        let decoded = IeAccessType::decode(encoded).unwrap();
        assert_eq!(decoded.value, AccessType::ThreeGppAccess);

        let ie2 = IeAccessType::new(AccessType::NonThreeGppAccess);
        let encoded2 = ie2.encode();
        assert_eq!(encoded2, 0b10);

        let decoded2 = IeAccessType::decode(encoded2).unwrap();
        assert_eq!(decoded2.value, AccessType::NonThreeGppAccess);
    }

    #[test]
    fn test_ie_allowed_ssc_mode_encode_decode() {
        let ie = IeAllowedSscMode::new(Ssc1::Allowed, Ssc2::NotAllowed, Ssc3::Allowed);
        let encoded = ie.encode();
        assert_eq!(encoded, 0b101); // SSC3=1, SSC2=0, SSC1=1

        let decoded = IeAllowedSscMode::decode(encoded).unwrap();
        assert_eq!(decoded.ssc1, Ssc1::Allowed);
        assert_eq!(decoded.ssc2, Ssc2::NotAllowed);
        assert_eq!(decoded.ssc3, Ssc3::Allowed);
    }

    #[test]
    fn test_ie_always_on_pdu_session_indication() {
        let ie = IeAlwaysOnPduSessionIndication::new(AlwaysOnPduSessionIndication::Required);
        let encoded = ie.encode();
        assert_eq!(encoded, 0b1);

        let decoded = IeAlwaysOnPduSessionIndication::decode(encoded).unwrap();
        assert_eq!(decoded.value, AlwaysOnPduSessionIndication::Required);
    }

    #[test]
    fn test_ie_configuration_update_indication() {
        let ie = IeConfigurationUpdateIndication::new(
            Acknowledgement::Requested,
            RegistrationRequested::Requested,
        );
        let encoded = ie.encode();
        assert_eq!(encoded, 0b11); // RED=1, ACK=1

        let decoded = IeConfigurationUpdateIndication::decode(encoded).unwrap();
        assert_eq!(decoded.ack, Acknowledgement::Requested);
        assert_eq!(decoded.red, RegistrationRequested::Requested);
    }

    #[test]
    fn test_ie_de_registration_type() {
        let ie = IeDeRegistrationType::new(
            DeRegistrationAccessType::ThreeGppAndNonThreeGppAccess,
            ReRegistrationRequired::Required,
            SwitchOff::SwitchOff,
        );
        let encoded = ie.encode();
        // SwitchOff=1 (bit 3), ReReg=1 (bit 2), AccessType=11 (bits 1-0)
        assert_eq!(encoded, 0b1111);

        let decoded = IeDeRegistrationType::decode(encoded).unwrap();
        assert_eq!(decoded.access_type, DeRegistrationAccessType::ThreeGppAndNonThreeGppAccess);
        assert_eq!(decoded.re_registration_required, ReRegistrationRequired::Required);
        assert_eq!(decoded.switch_off, SwitchOff::SwitchOff);
    }

    #[test]
    fn test_ie_imeisv_request() {
        let ie = IeImeiSvRequest::new(ImeiSvRequest::Requested);
        let encoded = ie.encode();
        assert_eq!(encoded, 0b001);

        let decoded = IeImeiSvRequest::decode(encoded).unwrap();
        assert_eq!(decoded.imeisv_request, ImeiSvRequest::Requested);
    }

    #[test]
    fn test_ie_mico_indication() {
        let ie = IeMicoIndication::new(RegistrationAreaAllocationIndication::Allocated);
        let encoded = ie.encode();
        assert_eq!(encoded, 0b1);

        let decoded = IeMicoIndication::decode(encoded).unwrap();
        assert_eq!(decoded.raai, RegistrationAreaAllocationIndication::Allocated);
    }

    #[test]
    fn test_ie_nas_key_set_identifier() {
        let ie = IeNasKeySetIdentifier::new(TypeOfSecurityContext::MappedSecurityContext, 5);
        let encoded = ie.encode();
        assert_eq!(encoded, 0b1101); // TSC=1, KSI=101

        let decoded = IeNasKeySetIdentifier::decode(encoded).unwrap();
        assert_eq!(decoded.tsc, TypeOfSecurityContext::MappedSecurityContext);
        assert_eq!(decoded.ksi, 5);
    }

    #[test]
    fn test_ie_nas_key_set_identifier_not_available() {
        let ie = IeNasKeySetIdentifier::not_available();
        assert!(!ie.is_available());
        assert_eq!(ie.ksi, IeNasKeySetIdentifier::NOT_AVAILABLE_OR_RESERVED);

        let ie2 = IeNasKeySetIdentifier::new(TypeOfSecurityContext::NativeSecurityContext, 3);
        assert!(ie2.is_available());
    }

    #[test]
    fn test_ie_network_slicing_indication() {
        let ie = IeNetworkSlicingIndication::new(
            NetworkSlicingSubscriptionChangeIndication::Changed,
            DefaultConfiguredNssaiIndication::CreatedFromDefault,
        );
        let encoded = ie.encode();
        assert_eq!(encoded, 0b11); // DCNI=1, NSSCI=1

        let decoded = IeNetworkSlicingIndication::decode(encoded).unwrap();
        assert_eq!(decoded.nssci, NetworkSlicingSubscriptionChangeIndication::Changed);
        assert_eq!(decoded.dcni, DefaultConfiguredNssaiIndication::CreatedFromDefault);
    }

    #[test]
    fn test_ie_nssai_inclusion_mode() {
        for mode in [
            NssaiInclusionMode::A,
            NssaiInclusionMode::B,
            NssaiInclusionMode::C,
            NssaiInclusionMode::D,
        ] {
            let ie = IeNssaiInclusionMode::new(mode);
            let encoded = ie.encode();
            let decoded = IeNssaiInclusionMode::decode(encoded).unwrap();
            assert_eq!(decoded.nssai_inclusion_mode, mode);
        }
    }

    #[test]
    fn test_ie_payload_container_type() {
        let ie = IePayloadContainerType::new(PayloadContainerType::Sms);
        let encoded = ie.encode();
        assert_eq!(encoded, 0b0010);

        let decoded = IePayloadContainerType::decode(encoded).unwrap();
        assert_eq!(decoded.payload_container_type, PayloadContainerType::Sms);
    }

    #[test]
    fn test_ie_pdu_session_type() {
        for pdu_type in [
            PduSessionType::Ipv4,
            PduSessionType::Ipv6,
            PduSessionType::Ipv4v6,
            PduSessionType::Unstructured,
            PduSessionType::Ethernet,
        ] {
            let ie = IePduSessionType::new(pdu_type);
            let encoded = ie.encode();
            let decoded = IePduSessionType::decode(encoded).unwrap();
            assert_eq!(decoded.pdu_session_type, pdu_type);
        }
    }

    #[test]
    fn test_ie_request_type() {
        let ie = IeRequestType::new(RequestType::ModificationRequest);
        let encoded = ie.encode();
        assert_eq!(encoded, 0b101);

        let decoded = IeRequestType::decode(encoded).unwrap();
        assert_eq!(decoded.request_type, RequestType::ModificationRequest);
    }

    #[test]
    fn test_ie_service_type() {
        let ie = IeServiceType::new(ServiceType::EmergencyServices);
        let encoded = ie.encode();
        assert_eq!(encoded, 0b0011);

        let decoded = IeServiceType::decode(encoded).unwrap();
        assert_eq!(decoded.service_type, ServiceType::EmergencyServices);
    }

    #[test]
    fn test_ie_sms_indication() {
        let ie = IeSmsIndication::new(SmsAvailabilityIndication::Available);
        let encoded = ie.encode();
        assert_eq!(encoded, 0b1);

        let decoded = IeSmsIndication::decode(encoded).unwrap();
        assert_eq!(decoded.sai, SmsAvailabilityIndication::Available);
    }

    #[test]
    fn test_ie_ssc_mode() {
        for mode in [SscMode::SscMode1, SscMode::SscMode2, SscMode::SscMode3] {
            let ie = IeSscMode::new(mode);
            let encoded = ie.encode();
            let decoded = IeSscMode::decode(encoded).unwrap();
            assert_eq!(decoded.ssc_mode, mode);
        }
    }

    #[test]
    fn test_invalid_decode() {
        // Test invalid identity type
        let result = Ie5gsIdentityType::decode(0b111);
        assert!(result.is_err());

        // Test invalid registration type
        let result = Ie5gsRegistrationType::decode(0b0000);
        assert!(result.is_err());
    }
}
