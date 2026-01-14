//! Type 3 Information Elements (fixed length)
//!
//! Type 3 IEs have a fixed length and contain specific protocol values.
//! They are used for encoding/decoding NAS message fields with known sizes.
//!
//! Based on 3GPP TS 24.501 specification.

use nextgsim_common::{OctetString, OctetView, Plmn, Tai};
use num_enum::{IntoPrimitive, TryFromPrimitive};

/// Error type for Type 3 IE decoding
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum Ie3Error {
    /// Invalid value for the IE type
    #[error("Invalid value 0x{0:X} for {1}")]
    InvalidValue(u8, &'static str),
    /// Insufficient data for decoding
    #[error("Insufficient data: expected {expected} bytes, got {actual}")]
    InsufficientData { expected: usize, actual: usize },
}

// ============================================================================
// Enumerations for Type 3 IEs
// ============================================================================

/// 5GMM Cause values (3GPP TS 24.501 Section 9.11.3.2)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum MmCause {
    /// Illegal UE
    IllegalUe = 0x03,
    /// PEI not accepted
    PeiNotAccepted = 0x05,
    /// Illegal ME
    IllegalMe = 0x06,
    /// 5GS services not allowed
    FiveGsServicesNotAllowed = 0x07,
    /// UE identity cannot be derived by the network
    UeIdentityCannotBeDerived = 0x09,
    /// Implicitly de-registered
    ImplicitlyDeregistered = 0x0A,
    /// PLMN not allowed
    PlmnNotAllowed = 0x0B,
    /// Tracking area not allowed
    TaNotAllowed = 0x0C,
    /// Roaming not allowed in this tracking area
    RoamingNotAllowedInTa = 0x0D,
    /// No suitable cells in tracking area
    NoSuitableCellsInTa = 0x0F,
    /// MAC failure
    MacFailure = 0x14,
    /// Synch failure
    SynchFailure = 0x15,
    /// Congestion
    Congestion = 0x16,
    /// UE security capabilities mismatch
    UeSecurityCapMismatch = 0x17,
    /// Security mode rejected, unspecified
    SecModeRejectedUnspecified = 0x18,
    /// Non-5G authentication unacceptable
    Non5gAuthenticationUnacceptable = 0x1A,
    /// N1 mode not allowed
    N1ModeNotAllowed = 0x1B,
    /// Restricted service area
    RestrictedServiceArea = 0x1C,
    /// LADN not available
    LadnNotAvailable = 0x2B,
    /// Maximum number of PDU sessions reached
    MaxPduSessionsReached = 0x41,
    /// Insufficient resources for specific slice and DNN
    InsufficientResourcesForSliceAndDnn = 0x43,
    /// Insufficient resources for specific slice
    InsufficientResourcesForSlice = 0x45,
    /// ngKSI already in use
    NgksiAlreadyInUse = 0x47,
    /// Non-3GPP access to 5GCN not allowed
    Non3gppAccessTo5gcnNotAllowed = 0x48,
    /// Serving network not authorized
    ServingNetworkNotAuthorized = 0x49,
    /// Payload was not forwarded
    PayloadNotForwarded = 0x5A,
    /// DNN not supported or not subscribed in the slice
    DnnNotSupportedOrNotSubscribed = 0x5B,
    /// Insufficient user-plane resources for the PDU session
    InsufficientUserPlaneResources = 0x5C,
    /// Semantically incorrect message
    #[default]
    SemanticallyIncorrectMessage = 0x5F,
    /// Invalid mandatory information
    InvalidMandatoryInformation = 0x60,
    /// Message type non-existent or not implemented
    MessageTypeNonExistent = 0x61,
    /// Message type not compatible with protocol state
    MessageTypeNotCompatible = 0x62,
    /// Information element non-existent or not implemented
    IeNonExistent = 0x63,
    /// Conditional IE error
    ConditionalIeError = 0x64,
    /// Message not compatible with protocol state
    MessageNotCompatible = 0x65,
    /// Protocol error, unspecified
    ProtocolErrorUnspecified = 0x6F,
}

/// 5GSM Cause values (3GPP TS 24.501 Section 9.11.4.2)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum SmCause {
    /// Insufficient resources
    InsufficientResources = 0x1A,
    /// Missing or unknown DNN
    MissingOrUnknownDnn = 0x1B,
    /// Unknown PDU session type
    UnknownPduSessionType = 0x1C,
    /// User authentication or authorization failed
    UserAuthFailed = 0x1D,
    /// Request rejected, unspecified
    RequestRejectedUnspecified = 0x1F,
    /// Service option temporarily out of order
    ServiceOptionTemporarilyOutOfOrder = 0x22,
    /// PTI already in use
    PtiAlreadyInUse = 0x23,
    /// Regular deactivation
    RegularDeactivation = 0x24,
    /// Network failure
    NetworkFailure = 0x26,
    /// Reactivation requested
    ReactivationRequested = 0x27,
    /// Semantic error in the TFT operation
    SemanticErrorInTftOperation = 0x29,
    /// Syntactical error in the TFT operation
    SyntacticalErrorInTftOperation = 0x2A,
    /// Invalid PDU session identity
    InvalidPduSessionIdentity = 0x2B,
    /// Semantic errors in packet filter(s)
    SemanticErrorsInPacketFilters = 0x2C,
    /// Syntactical error in packet filter(s)
    SyntacticalErrorInPacketFilters = 0x2D,
    /// Out of LADN service area
    OutOfLadnServiceArea = 0x2E,
    /// PTI mismatch
    PtiMismatch = 0x2F,
    /// PDU session type IPv4 only allowed
    PduSessionTypeIpv4OnlyAllowed = 0x32,
    /// PDU session type IPv6 only allowed
    PduSessionTypeIpv6OnlyAllowed = 0x33,
    /// PDU session does not exist
    PduSessionDoesNotExist = 0x36,
    /// PDU session type IPv4v6 only allowed
    PduSessionTypeIpv4v6OnlyAllowed = 0x39,
    /// PDU session type Unstructured only allowed
    PduSessionTypeUnstructuredOnlyAllowed = 0x3A,
    /// Unsupported 5QI value
    Unsupported5qiValue = 0x3B,
    /// PDU session type Ethernet only allowed
    PduSessionTypeEthernetOnlyAllowed = 0x3D,
    /// Insufficient resources for specific slice and DNN
    InsufficientResourcesForSliceAndDnn = 0x43,
    /// Not supported SSC mode
    NotSupportedSscMode = 0x44,
    /// Insufficient resources for specific slice
    InsufficientResourcesForSlice = 0x45,
    /// Missing or unknown DNN in a slice
    MissingOrUnknownDnnInSlice = 0x46,
    /// Invalid PTI value
    InvalidPtiValue = 0x51,
    /// Maximum data rate per UE for user-plane integrity protection is too low
    MaxDataRateForIntegrityProtectionTooLow = 0x52,
    /// Semantic error in the QoS operation
    SemanticErrorInQosOperation = 0x53,
    /// Syntactical error in the QoS operation
    SyntacticalErrorInQosOperation = 0x54,
    /// Semantically incorrect message
    #[default]
    SemanticallyIncorrectMessage = 0x5F,
    /// Invalid mandatory information
    InvalidMandatoryInformation = 0x60,
    /// Message type non-existent or not implemented
    MessageTypeNonExistent = 0x61,
    /// Message type not compatible with the protocol state
    MessageTypeNotCompatible = 0x62,
    /// Information element non-existent or not implemented
    IeNonExistent = 0x63,
    /// Conditional IE error
    ConditionalIeError = 0x64,
    /// Message not compatible with the protocol state
    MessageNotCompatible = 0x65,
    /// Protocol error, unspecified
    ProtocolErrorUnspecified = 0x6F,
}

/// Type of ciphering algorithm (3GPP TS 24.501 Section 9.11.3.34)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum TypeOfCipheringAlgorithm {
    /// 5G-EA0 (null ciphering)
    #[default]
    Ea0 = 0x00,
    /// 128-5G-EA1
    Ea1_128 = 0x01,
    /// 128-5G-EA2
    Ea2_128 = 0x02,
    /// 128-5G-EA3
    Ea3_128 = 0x03,
    /// 5G-EA4
    Ea4 = 0x04,
    /// 5G-EA5
    Ea5 = 0x05,
    /// 5G-EA6
    Ea6 = 0x06,
    /// 5G-EA7
    Ea7 = 0x07,
}

/// Type of integrity protection algorithm (3GPP TS 24.501 Section 9.11.3.34)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum TypeOfIntegrityProtectionAlgorithm {
    /// 5G-IA0 (null integrity)
    #[default]
    Ia0 = 0x00,
    /// 128-5G-IA1
    Ia1_128 = 0x01,
    /// 128-5G-IA2
    Ia2_128 = 0x02,
    /// 128-5G-IA3
    Ia3_128 = 0x03,
    /// 5G-IA4
    Ia4 = 0x04,
    /// 5G-IA5
    Ia5 = 0x05,
    /// 5G-IA6
    Ia6 = 0x06,
    /// 5G-IA7
    Ia7 = 0x07,
}

/// EPS type of ciphering algorithm (3GPP TS 24.301)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum EpsTypeOfCipheringAlgorithm {
    /// EEA0 (null ciphering)
    #[default]
    Eea0 = 0x00,
    /// 128-EEA1
    Eea1_128 = 0x01,
    /// 128-EEA2
    Eea2_128 = 0x02,
    /// 128-EEA3
    Eea3_128 = 0x03,
    /// EEA4
    Eea4 = 0x04,
    /// EEA5
    Eea5 = 0x05,
    /// EEA6
    Eea6 = 0x06,
    /// EEA7
    Eea7 = 0x07,
}

/// EPS type of integrity protection algorithm (3GPP TS 24.301)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum EpsTypeOfIntegrityProtectionAlgorithm {
    /// EIA0 (null integrity)
    #[default]
    Eia0 = 0x00,
    /// 128-EIA1
    Eia1_128 = 0x01,
    /// 128-EIA2
    Eia2_128 = 0x02,
    /// 128-EIA3
    Eia3_128 = 0x03,
    /// EIA4
    Eia4 = 0x04,
    /// EIA5
    Eia5 = 0x05,
    /// EIA6
    Eia6 = 0x06,
    /// EIA7
    Eia7 = 0x07,
}

/// GPRS timer value unit (3GPP TS 24.008 Section 10.5.7.3)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum GprsTimerValueUnit {
    /// Value is incremented in multiples of 2 seconds
    #[default]
    MultiplesOf2Seconds = 0b000,
    /// Value is incremented in multiples of 1 minute
    MultiplesOf1Minute = 0b001,
    /// Value is incremented in multiples of decihours (6 minutes)
    MultiplesOfDecihours = 0b010,
    /// Timer is deactivated
    TimerDeactivated = 0b111,
}

/// Maximum data rate per UE for user-plane integrity protection (uplink)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum MaxDataRateUplink {
    /// 64 kbps
    #[default]
    SixtyFourKbps = 0x00,
    /// Full data rate
    FullDataRate = 0xFF,
}

/// Maximum data rate per UE for user-plane integrity protection (downlink)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum MaxDataRateDownlink {
    /// 64 kbps
    #[default]
    SixtyFourKbps = 0x00,
    /// Full data rate
    FullDataRate = 0xFF,
}

// ============================================================================
// Trait for Type 3 Information Elements
// ============================================================================

/// Trait for Type 3 Information Elements (fixed length)
pub trait InformationElement3: Sized {
    /// The fixed length of this IE in bytes
    const LENGTH: usize;

    /// Decode from an OctetView
    fn decode(stream: &OctetView) -> Result<Self, Ie3Error>;

    /// Encode to an OctetString
    fn encode(&self, stream: &mut OctetString);
}

// ============================================================================
// Type 3 IE Structures
// ============================================================================

/// 5GMM Cause IE (3GPP TS 24.501 Section 9.11.3.2)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Ie5gMmCause {
    /// The cause value
    pub value: MmCause,
}

impl Ie5gMmCause {
    /// Create a new 5GMM Cause IE
    pub fn new(value: MmCause) -> Self {
        Self { value }
    }
}

impl InformationElement3 for Ie5gMmCause {
    const LENGTH: usize = 1;

    fn decode(stream: &OctetView) -> Result<Self, Ie3Error> {
        let val = stream.read();
        let value = MmCause::try_from(val)
            .map_err(|_| Ie3Error::InvalidValue(val, "MmCause"))?;
        Ok(Self { value })
    }

    fn encode(&self, stream: &mut OctetString) {
        stream.append_octet(self.value.into());
    }
}

/// 5GSM Cause IE (3GPP TS 24.501 Section 9.11.4.2)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Ie5gSmCause {
    /// The cause value
    pub value: SmCause,
}

impl Ie5gSmCause {
    /// Create a new 5GSM Cause IE
    pub fn new(value: SmCause) -> Self {
        Self { value }
    }
}

impl InformationElement3 for Ie5gSmCause {
    const LENGTH: usize = 1;

    fn decode(stream: &OctetView) -> Result<Self, Ie3Error> {
        let val = stream.read();
        let value = SmCause::try_from(val)
            .map_err(|_| Ie3Error::InvalidValue(val, "SmCause"))?;
        Ok(Self { value })
    }

    fn encode(&self, stream: &mut OctetString) {
        stream.append_octet(self.value.into());
    }
}

/// 5GS Tracking Area Identity IE (3GPP TS 24.501 Section 9.11.3.8)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Ie5gsTrackingAreaIdentity {
    /// The TAI value (PLMN + TAC)
    pub tai: Tai,
}

impl Ie5gsTrackingAreaIdentity {
    /// Create a new 5GS Tracking Area Identity IE
    pub fn new(tai: Tai) -> Self {
        Self { tai }
    }

    /// Create from individual components
    pub fn from_parts(mcc: u16, mnc: u16, long_mnc: bool, tac: u32) -> Self {
        Self {
            tai: Tai::from_parts(mcc, mnc, long_mnc, tac),
        }
    }
}

impl InformationElement3 for Ie5gsTrackingAreaIdentity {
    const LENGTH: usize = 6;

    fn decode(stream: &OctetView) -> Result<Self, Ie3Error> {
        let plmn_bytes = [stream.read(), stream.read(), stream.read()];
        let plmn = Plmn::decode(plmn_bytes);
        let tac = stream.read_u24();
        Ok(Self {
            tai: Tai::new(plmn, tac),
        })
    }

    fn encode(&self, stream: &mut OctetString) {
        let encoded = self.tai.encode();
        for byte in encoded {
            stream.append_octet(byte);
        }
    }
}

/// Authentication Parameter RAND IE (3GPP TS 24.501 Section 9.11.3.16)
/// Fixed 16-byte random value used in authentication
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct IeAuthenticationParameterRand {
    /// The 16-byte RAND value
    pub value: OctetString,
}

impl IeAuthenticationParameterRand {
    /// Create a new Authentication Parameter RAND IE
    pub fn new(value: OctetString) -> Self {
        Self { value }
    }

    /// Create from a 16-byte array
    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        Self {
            value: OctetString::from_slice(&bytes),
        }
    }
}

impl InformationElement3 for IeAuthenticationParameterRand {
    const LENGTH: usize = 16;

    fn decode(stream: &OctetView) -> Result<Self, Ie3Error> {
        let value = stream.read_octet_string(16);
        Ok(Self { value })
    }

    fn encode(&self, stream: &mut OctetString) {
        stream.append(&self.value);
    }
}

/// NAS Security Algorithms IE (3GPP TS 24.501 Section 9.11.3.34)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IeNasSecurityAlgorithms {
    /// Type of integrity protection algorithm
    pub integrity: TypeOfIntegrityProtectionAlgorithm,
    /// Type of ciphering algorithm
    pub ciphering: TypeOfCipheringAlgorithm,
}

impl IeNasSecurityAlgorithms {
    /// Create a new NAS Security Algorithms IE
    pub fn new(
        integrity: TypeOfIntegrityProtectionAlgorithm,
        ciphering: TypeOfCipheringAlgorithm,
    ) -> Self {
        Self { integrity, ciphering }
    }
}

impl InformationElement3 for IeNasSecurityAlgorithms {
    const LENGTH: usize = 1;

    fn decode(stream: &OctetView) -> Result<Self, Ie3Error> {
        let val = stream.read();
        let ciphering = TypeOfCipheringAlgorithm::try_from((val >> 4) & 0x0F)
            .map_err(|_| Ie3Error::InvalidValue(val, "TypeOfCipheringAlgorithm"))?;
        let integrity = TypeOfIntegrityProtectionAlgorithm::try_from(val & 0x0F)
            .map_err(|_| Ie3Error::InvalidValue(val, "TypeOfIntegrityProtectionAlgorithm"))?;
        Ok(Self { integrity, ciphering })
    }

    fn encode(&self, stream: &mut OctetString) {
        let ciphering: u8 = self.ciphering.into();
        let integrity: u8 = self.integrity.into();
        stream.append_octet((ciphering << 4) | (integrity & 0x0F));
    }
}

/// EPS NAS Security Algorithms IE (3GPP TS 24.301)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IeEpsNasSecurityAlgorithms {
    /// Type of integrity protection algorithm
    pub integrity: EpsTypeOfIntegrityProtectionAlgorithm,
    /// Type of ciphering algorithm
    pub ciphering: EpsTypeOfCipheringAlgorithm,
}

impl IeEpsNasSecurityAlgorithms {
    /// Create a new EPS NAS Security Algorithms IE
    pub fn new(
        integrity: EpsTypeOfIntegrityProtectionAlgorithm,
        ciphering: EpsTypeOfCipheringAlgorithm,
    ) -> Self {
        Self { integrity, ciphering }
    }
}

impl InformationElement3 for IeEpsNasSecurityAlgorithms {
    const LENGTH: usize = 1;

    fn decode(stream: &OctetView) -> Result<Self, Ie3Error> {
        let val = stream.read();
        let ciphering = EpsTypeOfCipheringAlgorithm::try_from((val >> 4) & 0x07)
            .map_err(|_| Ie3Error::InvalidValue(val, "EpsTypeOfCipheringAlgorithm"))?;
        let integrity = EpsTypeOfIntegrityProtectionAlgorithm::try_from(val & 0x07)
            .map_err(|_| Ie3Error::InvalidValue(val, "EpsTypeOfIntegrityProtectionAlgorithm"))?;
        Ok(Self { integrity, ciphering })
    }

    fn encode(&self, stream: &mut OctetString) {
        let ciphering: u8 = self.ciphering.into();
        let integrity: u8 = self.integrity.into();
        stream.append_octet((ciphering << 4) | (integrity & 0x07));
    }
}

/// GPRS Timer IE (3GPP TS 24.008 Section 10.5.7.3)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IeGprsTimer {
    /// Timer value (5-bit, 0-31)
    pub timer_value: u8,
    /// Timer value unit
    pub timer_value_unit: GprsTimerValueUnit,
}

impl IeGprsTimer {
    /// Create a new GPRS Timer IE
    pub fn new(timer_value: u8, timer_value_unit: GprsTimerValueUnit) -> Self {
        Self {
            timer_value: timer_value & 0x1F,
            timer_value_unit,
        }
    }
}

impl InformationElement3 for IeGprsTimer {
    const LENGTH: usize = 1;

    fn decode(stream: &OctetView) -> Result<Self, Ie3Error> {
        let val = stream.read();
        let timer_value = val & 0x1F;
        let timer_value_unit = GprsTimerValueUnit::try_from((val >> 5) & 0x07)
            .map_err(|_| Ie3Error::InvalidValue(val, "GprsTimerValueUnit"))?;
        Ok(Self {
            timer_value,
            timer_value_unit,
        })
    }

    fn encode(&self, stream: &mut OctetString) {
        let unit: u8 = self.timer_value_unit.into();
        stream.append_octet((unit << 5) | (self.timer_value & 0x1F));
    }
}

/// Integrity Protection Maximum Data Rate IE (3GPP TS 24.501 Section 9.11.4.7)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IeIntegrityProtectionMaximumDataRate {
    /// Maximum data rate for uplink
    pub max_rate_uplink: MaxDataRateUplink,
    /// Maximum data rate for downlink
    pub max_rate_downlink: MaxDataRateDownlink,
}

impl IeIntegrityProtectionMaximumDataRate {
    /// Create a new Integrity Protection Maximum Data Rate IE
    pub fn new(max_rate_uplink: MaxDataRateUplink, max_rate_downlink: MaxDataRateDownlink) -> Self {
        Self {
            max_rate_uplink,
            max_rate_downlink,
        }
    }
}

impl InformationElement3 for IeIntegrityProtectionMaximumDataRate {
    const LENGTH: usize = 2;

    fn decode(stream: &OctetView) -> Result<Self, Ie3Error> {
        let uplink = stream.read();
        let downlink = stream.read();
        let max_rate_uplink = MaxDataRateUplink::try_from(uplink)
            .map_err(|_| Ie3Error::InvalidValue(uplink, "MaxDataRateUplink"))?;
        let max_rate_downlink = MaxDataRateDownlink::try_from(downlink)
            .map_err(|_| Ie3Error::InvalidValue(downlink, "MaxDataRateDownlink"))?;
        Ok(Self {
            max_rate_uplink,
            max_rate_downlink,
        })
    }

    fn encode(&self, stream: &mut OctetString) {
        stream.append_octet(self.max_rate_uplink.into());
        stream.append_octet(self.max_rate_downlink.into());
    }
}

/// Maximum Number of Supported Packet Filters IE (3GPP TS 24.501 Section 9.11.4.9)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IeMaximumNumberOfSupportedPacketFilters {
    /// Maximum number of packet filters (11-bit value, 0-2047)
    pub value: u16,
}

impl IeMaximumNumberOfSupportedPacketFilters {
    /// Create a new Maximum Number of Supported Packet Filters IE
    pub fn new(value: u16) -> Self {
        Self {
            value: value & 0x07FF,
        }
    }
}

impl InformationElement3 for IeMaximumNumberOfSupportedPacketFilters {
    const LENGTH: usize = 2;

    fn decode(stream: &OctetView) -> Result<Self, Ie3Error> {
        let high = stream.read() as u16;
        let low = stream.read() as u16;
        // 11-bit value: bits 10-3 in first octet, bits 2-0 in second octet (high 3 bits)
        let value = (high << 3) | ((low >> 5) & 0x07);
        Ok(Self { value })
    }

    fn encode(&self, stream: &mut OctetString) {
        // 11-bit value: bits 10-3 in first octet, bits 2-0 in second octet (high 3 bits)
        let high = ((self.value >> 3) & 0xFF) as u8;
        let low = ((self.value & 0x07) << 5) as u8;
        stream.append_octet(high);
        stream.append_octet(low);
    }
}

/// N1 Mode to S1 Mode NAS Transparent Container IE (3GPP TS 24.501 Section 9.11.2.7)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IeN1ModeToS1ModeNasTransparentContainer {
    /// Sequence number
    pub sequence_number: u8,
}

impl IeN1ModeToS1ModeNasTransparentContainer {
    /// Create a new N1 Mode to S1 Mode NAS Transparent Container IE
    pub fn new(sequence_number: u8) -> Self {
        Self { sequence_number }
    }
}

impl InformationElement3 for IeN1ModeToS1ModeNasTransparentContainer {
    const LENGTH: usize = 1;

    fn decode(stream: &OctetView) -> Result<Self, Ie3Error> {
        let sequence_number = stream.read();
        Ok(Self { sequence_number })
    }

    fn encode(&self, stream: &mut OctetString) {
        stream.append_octet(self.sequence_number);
    }
}

/// PDU Session Identity 2 IE (3GPP TS 24.501 Section 9.11.3.41)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IePduSessionIdentity2 {
    /// PDU session identity value
    pub value: u8,
}

impl IePduSessionIdentity2 {
    /// Create a new PDU Session Identity 2 IE
    pub fn new(value: u8) -> Self {
        Self { value }
    }
}

impl InformationElement3 for IePduSessionIdentity2 {
    const LENGTH: usize = 1;

    fn decode(stream: &OctetView) -> Result<Self, Ie3Error> {
        let value = stream.read();
        Ok(Self { value })
    }

    fn encode(&self, stream: &mut OctetString) {
        stream.append_octet(self.value);
    }
}

/// Time Zone IE (3GPP TS 24.501 Section 9.11.3.52)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IeTimeZone {
    /// Time zone value (encoded as per 3GPP TS 23.040)
    pub value: u8,
}

impl IeTimeZone {
    /// Create a new Time Zone IE
    pub fn new(value: u8) -> Self {
        Self { value }
    }
}

impl InformationElement3 for IeTimeZone {
    const LENGTH: usize = 1;

    fn decode(stream: &OctetView) -> Result<Self, Ie3Error> {
        let value = stream.read();
        Ok(Self { value })
    }

    fn encode(&self, stream: &mut OctetString) {
        stream.append_octet(self.value);
    }
}

/// Universal time structure for TimeZoneAndTime IE
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VTime {
    /// Year (BCD encoded)
    pub year: u8,
    /// Month (BCD encoded)
    pub month: u8,
    /// Day (BCD encoded)
    pub day: u8,
    /// Hour (BCD encoded)
    pub hour: u8,
    /// Minute (BCD encoded)
    pub minute: u8,
    /// Second (BCD encoded)
    pub second: u8,
}

impl VTime {
    /// Create a new VTime
    pub fn new(year: u8, month: u8, day: u8, hour: u8, minute: u8, second: u8) -> Self {
        Self {
            year,
            month,
            day,
            hour,
            minute,
            second,
        }
    }

    /// Decode from OctetView (6 bytes)
    pub fn decode(stream: &OctetView) -> Self {
        Self {
            year: stream.read(),
            month: stream.read(),
            day: stream.read(),
            hour: stream.read(),
            minute: stream.read(),
            second: stream.read(),
        }
    }

    /// Encode to OctetString
    pub fn encode(&self, stream: &mut OctetString) {
        stream.append_octet(self.year);
        stream.append_octet(self.month);
        stream.append_octet(self.day);
        stream.append_octet(self.hour);
        stream.append_octet(self.minute);
        stream.append_octet(self.second);
    }
}

/// Time Zone and Time IE (3GPP TS 24.501 Section 9.11.3.53)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IeTimeZoneAndTime {
    /// Universal time
    pub time: VTime,
    /// Time zone value
    pub timezone: u8,
}

impl IeTimeZoneAndTime {
    /// Create a new Time Zone and Time IE
    pub fn new(time: VTime, timezone: u8) -> Self {
        Self { time, timezone }
    }
}

impl InformationElement3 for IeTimeZoneAndTime {
    const LENGTH: usize = 7;

    fn decode(stream: &OctetView) -> Result<Self, Ie3Error> {
        let time = VTime::decode(stream);
        let timezone = stream.read();
        Ok(Self { time, timezone })
    }

    fn encode(&self, stream: &mut OctetString) {
        self.time.encode(stream);
        stream.append_octet(self.timezone);
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ie5g_mm_cause_encode_decode() {
        let ie = Ie5gMmCause::new(MmCause::IllegalUe);
        let mut stream = OctetString::new();
        ie.encode(&mut stream);
        assert_eq!(stream.data(), &[0x03]);

        let view = OctetView::new(stream.data());
        let decoded = Ie5gMmCause::decode(&view).unwrap();
        assert_eq!(decoded.value, MmCause::IllegalUe);
    }

    #[test]
    fn test_ie5g_sm_cause_encode_decode() {
        let ie = Ie5gSmCause::new(SmCause::RegularDeactivation);
        let mut stream = OctetString::new();
        ie.encode(&mut stream);
        assert_eq!(stream.data(), &[0x24]);

        let view = OctetView::new(stream.data());
        let decoded = Ie5gSmCause::decode(&view).unwrap();
        assert_eq!(decoded.value, SmCause::RegularDeactivation);
    }

    #[test]
    fn test_ie5gs_tracking_area_identity_encode_decode() {
        let ie = Ie5gsTrackingAreaIdentity::from_parts(310, 410, true, 0x123456);
        let mut stream = OctetString::new();
        ie.encode(&mut stream);
        assert_eq!(stream.len(), 6);

        let view = OctetView::new(stream.data());
        let decoded = Ie5gsTrackingAreaIdentity::decode(&view).unwrap();
        assert_eq!(decoded.tai.plmn.mcc, 310);
        assert_eq!(decoded.tai.plmn.mnc, 410);
        assert!(decoded.tai.plmn.long_mnc);
        assert_eq!(decoded.tai.tac, 0x123456);
    }

    #[test]
    fn test_ie_authentication_parameter_rand_encode_decode() {
        let bytes = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
                     0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10];
        let ie = IeAuthenticationParameterRand::from_bytes(bytes);
        let mut stream = OctetString::new();
        ie.encode(&mut stream);
        assert_eq!(stream.len(), 16);
        assert_eq!(stream.data(), &bytes);

        let view = OctetView::new(stream.data());
        let decoded = IeAuthenticationParameterRand::decode(&view).unwrap();
        assert_eq!(decoded.value.data(), &bytes);
    }

    #[test]
    fn test_ie_nas_security_algorithms_encode_decode() {
        let ie = IeNasSecurityAlgorithms::new(
            TypeOfIntegrityProtectionAlgorithm::Ia2_128,
            TypeOfCipheringAlgorithm::Ea1_128,
        );
        let mut stream = OctetString::new();
        ie.encode(&mut stream);
        // Ciphering (EA1=0x01) in high nibble, Integrity (IA2=0x02) in low nibble
        assert_eq!(stream.data(), &[0x12]);

        let view = OctetView::new(stream.data());
        let decoded = IeNasSecurityAlgorithms::decode(&view).unwrap();
        assert_eq!(decoded.integrity, TypeOfIntegrityProtectionAlgorithm::Ia2_128);
        assert_eq!(decoded.ciphering, TypeOfCipheringAlgorithm::Ea1_128);
    }

    #[test]
    fn test_ie_eps_nas_security_algorithms_encode_decode() {
        let ie = IeEpsNasSecurityAlgorithms::new(
            EpsTypeOfIntegrityProtectionAlgorithm::Eia1_128,
            EpsTypeOfCipheringAlgorithm::Eea2_128,
        );
        let mut stream = OctetString::new();
        ie.encode(&mut stream);
        // Ciphering (EEA2=0x02) in high nibble, Integrity (EIA1=0x01) in low nibble
        assert_eq!(stream.data(), &[0x21]);

        let view = OctetView::new(stream.data());
        let decoded = IeEpsNasSecurityAlgorithms::decode(&view).unwrap();
        assert_eq!(decoded.integrity, EpsTypeOfIntegrityProtectionAlgorithm::Eia1_128);
        assert_eq!(decoded.ciphering, EpsTypeOfCipheringAlgorithm::Eea2_128);
    }

    #[test]
    fn test_ie_gprs_timer_encode_decode() {
        let ie = IeGprsTimer::new(10, GprsTimerValueUnit::MultiplesOf1Minute);
        let mut stream = OctetString::new();
        ie.encode(&mut stream);
        // Unit (001) in bits 7-5, value (01010) in bits 4-0
        assert_eq!(stream.data(), &[0x2A]);

        let view = OctetView::new(stream.data());
        let decoded = IeGprsTimer::decode(&view).unwrap();
        assert_eq!(decoded.timer_value, 10);
        assert_eq!(decoded.timer_value_unit, GprsTimerValueUnit::MultiplesOf1Minute);
    }

    #[test]
    fn test_ie_integrity_protection_max_data_rate_encode_decode() {
        let ie = IeIntegrityProtectionMaximumDataRate::new(
            MaxDataRateUplink::FullDataRate,
            MaxDataRateDownlink::SixtyFourKbps,
        );
        let mut stream = OctetString::new();
        ie.encode(&mut stream);
        assert_eq!(stream.data(), &[0xFF, 0x00]);

        let view = OctetView::new(stream.data());
        let decoded = IeIntegrityProtectionMaximumDataRate::decode(&view).unwrap();
        assert_eq!(decoded.max_rate_uplink, MaxDataRateUplink::FullDataRate);
        assert_eq!(decoded.max_rate_downlink, MaxDataRateDownlink::SixtyFourKbps);
    }

    #[test]
    fn test_ie_max_packet_filters_encode_decode() {
        let ie = IeMaximumNumberOfSupportedPacketFilters::new(256);
        let mut stream = OctetString::new();
        ie.encode(&mut stream);

        let view = OctetView::new(stream.data());
        let decoded = IeMaximumNumberOfSupportedPacketFilters::decode(&view).unwrap();
        assert_eq!(decoded.value, 256);
    }

    #[test]
    fn test_ie_n1_mode_container_encode_decode() {
        let ie = IeN1ModeToS1ModeNasTransparentContainer::new(42);
        let mut stream = OctetString::new();
        ie.encode(&mut stream);
        assert_eq!(stream.data(), &[42]);

        let view = OctetView::new(stream.data());
        let decoded = IeN1ModeToS1ModeNasTransparentContainer::decode(&view).unwrap();
        assert_eq!(decoded.sequence_number, 42);
    }

    #[test]
    fn test_ie_pdu_session_identity2_encode_decode() {
        let ie = IePduSessionIdentity2::new(5);
        let mut stream = OctetString::new();
        ie.encode(&mut stream);
        assert_eq!(stream.data(), &[5]);

        let view = OctetView::new(stream.data());
        let decoded = IePduSessionIdentity2::decode(&view).unwrap();
        assert_eq!(decoded.value, 5);
    }

    #[test]
    fn test_ie_time_zone_encode_decode() {
        let ie = IeTimeZone::new(0x40);
        let mut stream = OctetString::new();
        ie.encode(&mut stream);
        assert_eq!(stream.data(), &[0x40]);

        let view = OctetView::new(stream.data());
        let decoded = IeTimeZone::decode(&view).unwrap();
        assert_eq!(decoded.value, 0x40);
    }

    #[test]
    fn test_ie_time_zone_and_time_encode_decode() {
        let time = VTime::new(0x21, 0x12, 0x25, 0x10, 0x30, 0x00);
        let ie = IeTimeZoneAndTime::new(time, 0x40);
        let mut stream = OctetString::new();
        ie.encode(&mut stream);
        assert_eq!(stream.len(), 7);
        assert_eq!(stream.data(), &[0x21, 0x12, 0x25, 0x10, 0x30, 0x00, 0x40]);

        let view = OctetView::new(stream.data());
        let decoded = IeTimeZoneAndTime::decode(&view).unwrap();
        assert_eq!(decoded.time.year, 0x21);
        assert_eq!(decoded.time.month, 0x12);
        assert_eq!(decoded.time.day, 0x25);
        assert_eq!(decoded.time.hour, 0x10);
        assert_eq!(decoded.time.minute, 0x30);
        assert_eq!(decoded.time.second, 0x00);
        assert_eq!(decoded.timezone, 0x40);
    }
}
