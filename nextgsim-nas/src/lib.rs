//! NAS (Non-Access Stratum) protocol library
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
//! let decoded = PlainMmHeader::decode(&mut buf.as_slice()).unwrap();
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
pub use header::{HeaderError, NasHeader, NasHeaderType, PlainMmHeader, PlainSmHeader, SecuredHeader};
pub use ies::{
    Ie1Error, InformationElement1,
    // Type 1 IE enums
    AccessType, Acknowledgement, AlwaysOnPduSessionIndication, AlwaysOnPduSessionRequested,
    DefaultConfiguredNssaiIndication, DeRegistrationAccessType, FollowOnRequest, IdentityType,
    ImeiSvRequest, NetworkSlicingSubscriptionChangeIndication, NssaiInclusionMode,
    PayloadContainerType, PduSessionType, RegistrationAreaAllocationIndication,
    RegistrationRequested, RegistrationType, ReRegistrationRequired, RequestType, ServiceType,
    SmsAvailabilityIndication, Ssc1, Ssc2, Ssc3, SscMode, SwitchOff, TypeOfSecurityContext,
    // Type 1 IE structs
    Ie5gsIdentityType, Ie5gsRegistrationType, IeAccessType, IeAllowedSscMode,
    IeAlwaysOnPduSessionIndication, IeAlwaysOnPduSessionRequested, IeConfigurationUpdateIndication,
    IeDeRegistrationType, IeImeiSvRequest, IeMicoIndication, IeNasKeySetIdentifier,
    IeNetworkSlicingIndication, IeNssaiInclusionMode, IePayloadContainerType, IePduSessionType,
    IeRequestType, IeServiceType, IeSmsIndication, IeSscMode,
};
pub use security::{
    CipheringAlgorithm, IntegrityAlgorithm, NasKeySetIdentifier, NasSecurityAlgorithms,
    SecuredNasMessage, SecurityContextType, SecurityError, NasCount, NasDirection,
    compute_nas_mac, verify_nas_mac, NAS_BEARER,
};

// Re-export EAP types
pub use eap::{
    Eap, EapAkaPrime, EapAkaSubType, EapAttributeType, EapAttributes, EapCode, EapError,
    EapIdentity, EapNotification, EapType, decode_eap, encode_eap, encode_eap_to_vec,
};

// Re-export message types
pub use messages::mm::{
    // Registration messages
    Ie5gMmCause, Ie5gsMobileIdentity, Ie5gsRegistrationResult, MmCause, MobileIdentityType,
    RegistrationAccept, RegistrationComplete, RegistrationError, RegistrationReject,
    RegistrationRequest, RegistrationResultValue, SmsOverNasAllowed,
    // Authentication messages
    Abba, AuthenticationError, AuthenticationFailure, AuthenticationFailureParameter,
    AuthenticationParameterAutn, AuthenticationParameterRand, AuthenticationReject,
    AuthenticationRequest, AuthenticationResponse, AuthenticationResponseParameter,
    AuthenticationResult, EapMessage,
    // Status messages
    FiveGMmStatus, StatusError,
};
