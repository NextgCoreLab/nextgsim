//! NAS protocol enumerations
//!
//! Based on 3GPP TS 24.501 specification

use num_enum::{IntoPrimitive, TryFromPrimitive};

/// Extended Protocol Discriminator (EPD)
/// 3GPP TS 24.501 Section 9.2
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum ExtendedProtocolDiscriminator {
    /// 5GS Mobility Management messages
    MobilityManagement = 0x7E,
    /// 5GS Session Management messages
    SessionManagement = 0x2E,
}

/// Security Header Type
/// 3GPP TS 24.501 Section 9.3
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IntoPrimitive, TryFromPrimitive, Default)]
#[repr(u8)]
pub enum SecurityHeaderType {
    /// Plain NAS message, not security protected
    #[default]
    NotProtected = 0x00,
    /// Integrity protected
    IntegrityProtected = 0x01,
    /// Integrity protected and ciphered
    IntegrityProtectedAndCiphered = 0x02,
    /// Integrity protected with new 5G NAS security context
    IntegrityProtectedWithNewSecurityContext = 0x03,
    /// Integrity protected and ciphered with new 5G NAS security context
    IntegrityProtectedAndCipheredWithNewSecurityContext = 0x04,
}

impl SecurityHeaderType {
    /// Returns true if the message is security protected
    pub fn is_protected(&self) -> bool {
        !matches!(self, SecurityHeaderType::NotProtected)
    }

    /// Returns true if the message is ciphered
    pub fn is_ciphered(&self) -> bool {
        matches!(
            self,
            SecurityHeaderType::IntegrityProtectedAndCiphered
                | SecurityHeaderType::IntegrityProtectedAndCipheredWithNewSecurityContext
        )
    }

    /// Returns true if this indicates a new security context
    pub fn is_new_security_context(&self) -> bool {
        matches!(
            self,
            SecurityHeaderType::IntegrityProtectedWithNewSecurityContext
                | SecurityHeaderType::IntegrityProtectedAndCipheredWithNewSecurityContext
        )
    }
}

/// 5GMM Message Type
/// 3GPP TS 24.501 Section 9.7
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum MmMessageType {
    // Registration messages
    RegistrationRequest = 0x41,
    RegistrationAccept = 0x42,
    RegistrationComplete = 0x43,
    RegistrationReject = 0x44,

    // Deregistration messages
    DeregistrationRequestUeOriginating = 0x45,
    DeregistrationAcceptUeOriginating = 0x46,
    DeregistrationRequestUeTerminated = 0x47,
    DeregistrationAcceptUeTerminated = 0x48,

    // Service request messages
    ServiceRequest = 0x4C,
    ServiceReject = 0x4D,
    ServiceAccept = 0x4E,

    // Configuration update messages
    ConfigurationUpdateCommand = 0x54,
    ConfigurationUpdateComplete = 0x55,

    // Authentication messages
    AuthenticationRequest = 0x56,
    AuthenticationResponse = 0x57,
    AuthenticationReject = 0x58,
    AuthenticationFailure = 0x59,
    AuthenticationResult = 0x5A,

    // Identity messages
    IdentityRequest = 0x5B,
    IdentityResponse = 0x5C,

    // Security mode messages
    SecurityModeCommand = 0x5D,
    SecurityModeComplete = 0x5E,
    SecurityModeReject = 0x5F,

    // Status and notification messages
    FiveGMmStatus = 0x64,
    Notification = 0x65,
    NotificationResponse = 0x66,

    // NAS transport messages
    UlNasTransport = 0x67,
    DlNasTransport = 0x68,
}

/// 5GSM Message Type
/// 3GPP TS 24.501 Section 9.7
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum SmMessageType {
    // PDU session establishment messages
    PduSessionEstablishmentRequest = 0xC1,
    PduSessionEstablishmentAccept = 0xC2,
    PduSessionEstablishmentReject = 0xC3,

    // PDU session authentication messages
    PduSessionAuthenticationCommand = 0xC5,
    PduSessionAuthenticationComplete = 0xC6,
    PduSessionAuthenticationResult = 0xC7,

    // PDU session modification messages
    PduSessionModificationRequest = 0xC9,
    PduSessionModificationReject = 0xCA,
    PduSessionModificationCommand = 0xCB,
    PduSessionModificationComplete = 0xCC,
    PduSessionModificationCommandReject = 0xCD,

    // PDU session release messages
    PduSessionReleaseRequest = 0xD1,
    PduSessionReleaseReject = 0xD2,
    PduSessionReleaseCommand = 0xD3,
    PduSessionReleaseComplete = 0xD4,

    // Status message
    FiveGSmStatus = 0xD6,
}

/// Combined NAS Message Type (either MM or SM)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MessageType {
    /// 5GMM message type
    Mm(MmMessageType),
    /// 5GSM message type
    Sm(SmMessageType),
}

impl MessageType {
    /// Get the raw u8 value of the message type
    pub fn as_u8(&self) -> u8 {
        match self {
            MessageType::Mm(mt) => (*mt).into(),
            MessageType::Sm(mt) => (*mt).into(),
        }
    }

    /// Try to create a MessageType from EPD and raw value
    pub fn from_epd_and_value(
        epd: ExtendedProtocolDiscriminator,
        value: u8,
    ) -> Result<Self, MessageTypeError> {
        match epd {
            ExtendedProtocolDiscriminator::MobilityManagement => {
                MmMessageType::try_from(value)
                    .map(MessageType::Mm)
                    .map_err(|_| MessageTypeError::UnknownMmType(value))
            }
            ExtendedProtocolDiscriminator::SessionManagement => {
                SmMessageType::try_from(value)
                    .map(MessageType::Sm)
                    .map_err(|_| MessageTypeError::UnknownSmType(value))
            }
        }
    }
}

/// Error type for message type parsing
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum MessageTypeError {
    #[error("Unknown 5GMM message type: 0x{0:02X}")]
    UnknownMmType(u8),
    #[error("Unknown 5GSM message type: 0x{0:02X}")]
    UnknownSmType(u8),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epd_values() {
        assert_eq!(
            u8::from(ExtendedProtocolDiscriminator::MobilityManagement),
            0x7E
        );
        assert_eq!(
            u8::from(ExtendedProtocolDiscriminator::SessionManagement),
            0x2E
        );
    }

    #[test]
    fn test_security_header_type_values() {
        assert_eq!(u8::from(SecurityHeaderType::NotProtected), 0x00);
        assert_eq!(u8::from(SecurityHeaderType::IntegrityProtected), 0x01);
        assert_eq!(
            u8::from(SecurityHeaderType::IntegrityProtectedAndCiphered),
            0x02
        );
    }

    #[test]
    fn test_security_header_type_methods() {
        assert!(!SecurityHeaderType::NotProtected.is_protected());
        assert!(SecurityHeaderType::IntegrityProtected.is_protected());
        assert!(SecurityHeaderType::IntegrityProtectedAndCiphered.is_ciphered());
        assert!(!SecurityHeaderType::IntegrityProtected.is_ciphered());
        assert!(SecurityHeaderType::IntegrityProtectedWithNewSecurityContext.is_new_security_context());
    }

    #[test]
    fn test_mm_message_type_values() {
        assert_eq!(u8::from(MmMessageType::RegistrationRequest), 0x41);
        assert_eq!(u8::from(MmMessageType::AuthenticationRequest), 0x56);
        assert_eq!(u8::from(MmMessageType::SecurityModeCommand), 0x5D);
    }

    #[test]
    fn test_sm_message_type_values() {
        assert_eq!(u8::from(SmMessageType::PduSessionEstablishmentRequest), 0xC1);
        assert_eq!(u8::from(SmMessageType::PduSessionReleaseCommand), 0xD3);
    }

    #[test]
    fn test_message_type_from_epd() {
        let mt = MessageType::from_epd_and_value(
            ExtendedProtocolDiscriminator::MobilityManagement,
            0x41,
        );
        assert_eq!(mt, Ok(MessageType::Mm(MmMessageType::RegistrationRequest)));

        let mt = MessageType::from_epd_and_value(
            ExtendedProtocolDiscriminator::SessionManagement,
            0xC1,
        );
        assert_eq!(
            mt,
            Ok(MessageType::Sm(SmMessageType::PduSessionEstablishmentRequest))
        );
    }
}
