//! NAS (Non-Access Stratum) protocol library
#![allow(missing_docs)]
//!
//! Implements 5G NAS message encoding/decoding for:
//! - 5GMM (Mobility Management)
//! - 5GSM (Session Management)
//!
//! # Overview
//!
//! This crate provides types and functions for working with 5G NAS messages
//! as defined in 3GPP TS 24.501.
//!
//! # Message Structure
//!
//! NAS messages consist of:
//! - A header (plain or security protected)
//! - Message-specific information elements (IEs)
//!
//! ## Header Types
//!
//! - [`PlainMmHeader`]: Plain 5GMM message header (3 bytes)
//! - [`PlainSmHeader`]: Plain 5GSM message header (4 bytes)
//! - [`SecuredHeader`]: Security protected header (7 bytes)
//!
//! # Example
//!
//! ```rust
//! use nextgsim_nas::header::{PlainMmHeader, NasHeader};
//! use nextgsim_nas::enums::MmMessageType;
//!
//! // Create a plain MM header
//! let header = PlainMmHeader::new(MmMessageType::RegistrationRequest);
//!
//! // Encode to bytes
//! let mut buf = Vec::new();
//! header.encode(&mut buf);
//!
//! // Decode from bytes
//! let decoded = PlainMmHeader::decode(&mut buf.as_slice()).expect("value expected");
//! assert_eq!(decoded.message_type, MmMessageType::RegistrationRequest);
//! ```

pub mod codec;
pub mod eap;
pub mod enums;
pub mod header;
pub mod ies;
pub mod messages;
pub mod security;

#[cfg(test)]
mod capture_tests;

// Re-export commonly used types
pub use enums::{
    ExtendedProtocolDiscriminator, MessageType, MmMessageType, SecurityHeaderType, SmMessageType,
};
pub use header::{
    HeaderError, NasHeader, NasHeaderType, PlainMmHeader, PlainSmHeader, SecuredHeader,
};
pub use ies::{
    // Type 1 IE enums
    AccessType,
    Acknowledgement,
    AlwaysOnPduSessionIndication,
    AlwaysOnPduSessionRequested,
    DeRegistrationAccessType,
    DefaultConfiguredNssaiIndication,
    FollowOnRequest,
    IdentityType,
    Ie1Error,
    Ie4Error,
    // Type 1 IE structs
    Ie5gsIdentityType,
    Ie5gsRegistrationType,
    Ie6Error,
    IeAccessType,
    // Type 4 6G IEs (A2.12-A2.17)
    IeAiMlCapability,
    IeAllowedSscMode,
    IeAlwaysOnPduSessionIndication,
    IeAlwaysOnPduSessionRequested,
    IeConfigurationUpdateIndication,
    IeDeRegistrationType,
    IeImeiSvRequest,
    IeIsacParameter,
    // Type 6 IEs (LADN Information - A2.10)
    IeLadnInformation,
    IeMicoIndication,
    IeNasKeySetIdentifier,
    IeNetworkSlicingIndication,
    IeNssaiInclusionMode,
    IeNtnAccessBarring,
    IeNtnTimingAdvance,
    IePayloadContainerType,
    IePduSessionType,
    IeRequestType,
    IeSemanticCommParameter,
    IeServiceType,
    IeSmsIndication,
    IeSscMode,
    IeSubThzBandParameter,
    // Type 4 IEs (UE Security Capability - A2.9)
    IeUeSecurityCapability,
    ImeiSvRequest,
    InformationElement1,
    LadnEntry,
    NetworkSlicingSubscriptionChangeIndication,
    NssaiInclusionMode,
    PayloadContainerType,
    PduSessionType,
    ReRegistrationRequired,
    RegistrationAreaAllocationIndication,
    RegistrationRequested,
    RegistrationType,
    RequestType,
    ServiceType,
    SmsAvailabilityIndication,
    Ssc1,
    Ssc2,
    Ssc3,
    SscMode,
    SwitchOff,
    TypeOfSecurityContext,
};
pub use security::{
    compute_nas_mac, nas_cipher, verify_nas_mac, CipheringAlgorithm, IntegrityAlgorithm, NasCount,
    NasDirection, NasKeySetIdentifier, NasSecurityAlgorithms, SecuredNasMessage,
    SecurityContextType, SecurityError, NAS_BEARER, NAS_BEARER_NON_3GPP,
};

// Re-export EAP types
pub use eap::{
    decode_eap, encode_eap, encode_eap_to_vec, Eap, EapAkaPrime, EapAkaSubType, EapAttributeType,
    EapAttributes, EapCode, EapError, EapIdentity, EapNotification, EapType,
};

// Re-export message types
pub use messages::mm::{
    // Authentication messages
    Abba,
    AuthenticationError,
    AuthenticationFailure,
    AuthenticationFailureParameter,
    AuthenticationParameterAutn,
    AuthenticationParameterRand,
    AuthenticationReject,
    AuthenticationRequest,
    AuthenticationResponse,
    AuthenticationResponseParameter,
    AuthenticationResult,
    // NAS Transport messages
    DlNasTransport,
    EapMessage,
    // Status messages
    FiveGMmStatus,
    // Registration messages
    Ie5gMmCause,
    Ie5gsMobileIdentity,
    Ie5gsRegistrationResult,
    MmCause,
    MobileIdentityType,
    NasTransportError,
    // Notification messages
    Notification,
    NotificationError,
    NotificationResponse,
    RegistrationAccept,
    RegistrationComplete,
    RegistrationError,
    RegistrationReject,
    RegistrationRequest,
    RegistrationResultValue,
    SmsOverNasAllowed,
    StatusError,
    UlNasTransport,
};
