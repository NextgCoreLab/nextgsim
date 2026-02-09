//! PDU Session Release Messages (3GPP TS 24.501 Section 8.3.12-8.3.15)
//!
//! This module implements the PDU Session Release procedure messages:
//! - PDU Session Release Request (UE to network, Section 8.3.12)
//! - PDU Session Release Reject (network to UE, Section 8.3.13)
//! - PDU Session Release Command (network to UE, Section 8.3.14)
//! - PDU Session Release Complete (UE to network, Section 8.3.15)

use bytes::{Buf, BufMut};
use thiserror::Error;

use crate::enums::SmMessageType;
use crate::header::PlainSmHeader;
use crate::ies::ie3::{Ie5gSmCause, SmCause};

/// Error type for PDU Session Release message encoding/decoding
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PduSessionReleaseError {
    /// Buffer too short for decoding
    #[error("Buffer too short: expected at least {expected} bytes, got {actual}")]
    BufferTooShort {
        /// Expected minimum bytes
        expected: usize,
        /// Actual bytes available
        actual: usize,
    },
    /// Invalid IE value
    #[error("Invalid IE value: {0}")]
    InvalidIeValue(String),
    /// Header decoding error
    #[error("Header error: {0}")]
    HeaderError(#[from] crate::header::HeaderError),
}

// ============================================================================
// IEI Constants for PDU Session Release Messages
// ============================================================================

/// IEI values for PDU Session Release Request optional IEs
#[allow(dead_code)]
mod release_request_iei {
    /// 5GSM cause
    pub const SM_CAUSE: u8 = 0x59;
    /// Extended protocol configuration options
    pub const EXTENDED_PROTOCOL_CONFIG_OPTIONS: u8 = 0x7B;
}

/// IEI values for PDU Session Release Reject optional IEs
#[allow(dead_code)]
mod release_reject_iei {
    /// Extended protocol configuration options
    pub const EXTENDED_PROTOCOL_CONFIG_OPTIONS: u8 = 0x7B;
}

/// IEI values for PDU Session Release Command optional IEs
#[allow(dead_code)]
mod release_command_iei {
    /// Back-off timer value
    pub const BACK_OFF_TIMER_VALUE: u8 = 0x37;
    /// EAP message
    pub const EAP_MESSAGE: u8 = 0x78;
    /// Extended protocol configuration options
    pub const EXTENDED_PROTOCOL_CONFIG_OPTIONS: u8 = 0x7B;
}

/// IEI values for PDU Session Release Complete optional IEs
#[allow(dead_code)]
mod release_complete_iei {
    /// Extended protocol configuration options
    pub const EXTENDED_PROTOCOL_CONFIG_OPTIONS: u8 = 0x7B;
    /// 5GSM cause
    pub const SM_CAUSE: u8 = 0x59;
}


// ============================================================================
// PDU Session Release Request (3GPP TS 24.501 Section 8.3.12)
// ============================================================================

/// PDU Session Release Request message (UE to network)
///
/// This message is sent by the UE to the network to request release
/// of an existing PDU session.
///
/// 3GPP TS 24.501 Section 8.3.12
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PduSessionReleaseRequest {
    /// PDU Session ID (from header)
    pub pdu_session_id: u8,
    /// Procedure Transaction Identity (from header)
    pub pti: u8,
    /// 5GSM cause (optional, Type 3, IEI 0x59)
    pub sm_cause: Option<Ie5gSmCause>,
    /// Extended protocol configuration options (optional, Type 6, IEI 0x7B)
    pub extended_protocol_config_options: Option<Vec<u8>>,
}

impl PduSessionReleaseRequest {
    /// Create a new PDU Session Release Request
    pub fn new(pdu_session_id: u8, pti: u8) -> Self {
        Self {
            pdu_session_id,
            pti,
            ..Default::default()
        }
    }

    /// Decode from bytes (after header has been parsed)
    pub fn decode<B: Buf>(
        buf: &mut B,
        pdu_session_id: u8,
        pti: u8,
    ) -> Result<Self, PduSessionReleaseError> {
        let mut msg = Self::new(pdu_session_id, pti);

        // All IEs are optional, parse until buffer is empty
        while buf.remaining() > 0 {
            let iei = buf.chunk()[0];

            match iei {
                release_request_iei::SM_CAUSE => {
                    buf.advance(1);
                    if buf.remaining() < 1 {
                        break;
                    }
                    let cause_val = buf.get_u8();
                    let cause = SmCause::try_from(cause_val)
                        .unwrap_or(SmCause::ProtocolErrorUnspecified);
                    msg.sm_cause = Some(Ie5gSmCause::new(cause));
                }
                release_request_iei::EXTENDED_PROTOCOL_CONFIG_OPTIONS => {
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
        let header = PlainSmHeader::new(
            self.pdu_session_id,
            self.pti,
            SmMessageType::PduSessionReleaseRequest,
        );
        header.encode(buf);

        // Optional IEs
        if let Some(ref cause) = self.sm_cause {
            buf.put_u8(release_request_iei::SM_CAUSE);
            buf.put_u8(cause.value as u8);
        }

        if let Some(ref epco) = self.extended_protocol_config_options {
            buf.put_u8(release_request_iei::EXTENDED_PROTOCOL_CONFIG_OPTIONS);
            buf.put_u16(epco.len() as u16);
            buf.put_slice(epco);
        }
    }

    /// Get the message type
    pub fn message_type() -> SmMessageType {
        SmMessageType::PduSessionReleaseRequest
    }
}


// ============================================================================
// PDU Session Release Reject (3GPP TS 24.501 Section 8.3.13)
// ============================================================================

/// PDU Session Release Reject message (network to UE)
///
/// This message is sent by the network to the UE to reject a PDU session
/// release request.
///
/// 3GPP TS 24.501 Section 8.3.13
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PduSessionReleaseReject {
    /// PDU Session ID (from header)
    pub pdu_session_id: u8,
    /// Procedure Transaction Identity (from header)
    pub pti: u8,
    /// 5GSM cause (mandatory, Type 3)
    pub sm_cause: Ie5gSmCause,
    /// Extended protocol configuration options (optional, Type 6, IEI 0x7B)
    pub extended_protocol_config_options: Option<Vec<u8>>,
}

impl Default for PduSessionReleaseReject {
    fn default() -> Self {
        Self {
            pdu_session_id: 0,
            pti: 0,
            sm_cause: Ie5gSmCause::new(SmCause::ProtocolErrorUnspecified),
            extended_protocol_config_options: None,
        }
    }
}

impl PduSessionReleaseReject {
    /// Create a new PDU Session Release Reject
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
    ) -> Result<Self, PduSessionReleaseError> {
        // 5GSM cause (mandatory, Type 3 - 1 byte)
        if buf.remaining() < 1 {
            return Err(PduSessionReleaseError::BufferTooShort {
                expected: 1,
                actual: buf.remaining(),
            });
        }
        let cause_val = buf.get_u8();
        let cause = SmCause::try_from(cause_val).unwrap_or(SmCause::ProtocolErrorUnspecified);

        let mut msg = Self::new(pdu_session_id, pti, cause);

        // Parse optional IEs
        while buf.remaining() > 0 {
            let iei = buf.chunk()[0];

            match iei {
                release_reject_iei::EXTENDED_PROTOCOL_CONFIG_OPTIONS => {
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
        let header = PlainSmHeader::new(
            self.pdu_session_id,
            self.pti,
            SmMessageType::PduSessionReleaseReject,
        );
        header.encode(buf);

        // 5GSM cause (mandatory)
        buf.put_u8(self.sm_cause.value as u8);

        // Optional IEs
        if let Some(ref epco) = self.extended_protocol_config_options {
            buf.put_u8(release_reject_iei::EXTENDED_PROTOCOL_CONFIG_OPTIONS);
            buf.put_u16(epco.len() as u16);
            buf.put_slice(epco);
        }
    }

    /// Get the message type
    pub fn message_type() -> SmMessageType {
        SmMessageType::PduSessionReleaseReject
    }

    /// Get the cause value
    pub fn cause(&self) -> SmCause {
        self.sm_cause.value
    }
}


// ============================================================================
// PDU Session Release Command (3GPP TS 24.501 Section 8.3.14)
// ============================================================================

/// PDU Session Release Command message (network to UE)
///
/// This message is sent by the network to the UE to release
/// an existing PDU session.
///
/// 3GPP TS 24.501 Section 8.3.14
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PduSessionReleaseCommand {
    /// PDU Session ID (from header)
    pub pdu_session_id: u8,
    /// Procedure Transaction Identity (from header)
    pub pti: u8,
    /// 5GSM cause (mandatory, Type 3)
    pub sm_cause: Ie5gSmCause,
    /// Back-off timer value (optional, Type 4, IEI 0x37)
    pub back_off_timer_value: Option<u8>,
    /// EAP message (optional, Type 6, IEI 0x78)
    pub eap_message: Option<Vec<u8>>,
    /// Extended protocol configuration options (optional, Type 6, IEI 0x7B)
    pub extended_protocol_config_options: Option<Vec<u8>>,
}

impl Default for PduSessionReleaseCommand {
    fn default() -> Self {
        Self {
            pdu_session_id: 0,
            pti: 0,
            sm_cause: Ie5gSmCause::new(SmCause::ProtocolErrorUnspecified),
            back_off_timer_value: None,
            eap_message: None,
            extended_protocol_config_options: None,
        }
    }
}

impl PduSessionReleaseCommand {
    /// Create a new PDU Session Release Command
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
    ) -> Result<Self, PduSessionReleaseError> {
        // 5GSM cause (mandatory, Type 3 - 1 byte)
        if buf.remaining() < 1 {
            return Err(PduSessionReleaseError::BufferTooShort {
                expected: 1,
                actual: buf.remaining(),
            });
        }
        let cause_val = buf.get_u8();
        let cause = SmCause::try_from(cause_val).unwrap_or(SmCause::ProtocolErrorUnspecified);

        let mut msg = Self::new(pdu_session_id, pti, cause);

        // Parse optional IEs
        while buf.remaining() > 0 {
            let iei = buf.chunk()[0];

            match iei {
                release_command_iei::BACK_OFF_TIMER_VALUE => {
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
                release_command_iei::EAP_MESSAGE => {
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
                release_command_iei::EXTENDED_PROTOCOL_CONFIG_OPTIONS => {
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
        let header = PlainSmHeader::new(
            self.pdu_session_id,
            self.pti,
            SmMessageType::PduSessionReleaseCommand,
        );
        header.encode(buf);

        // 5GSM cause (mandatory)
        buf.put_u8(self.sm_cause.value as u8);

        // Optional IEs
        if let Some(timer) = self.back_off_timer_value {
            buf.put_u8(release_command_iei::BACK_OFF_TIMER_VALUE);
            buf.put_u8(1);
            buf.put_u8(timer);
        }

        if let Some(ref eap) = self.eap_message {
            buf.put_u8(release_command_iei::EAP_MESSAGE);
            buf.put_u16(eap.len() as u16);
            buf.put_slice(eap);
        }

        if let Some(ref epco) = self.extended_protocol_config_options {
            buf.put_u8(release_command_iei::EXTENDED_PROTOCOL_CONFIG_OPTIONS);
            buf.put_u16(epco.len() as u16);
            buf.put_slice(epco);
        }
    }

    /// Get the message type
    pub fn message_type() -> SmMessageType {
        SmMessageType::PduSessionReleaseCommand
    }

    /// Get the cause value
    pub fn cause(&self) -> SmCause {
        self.sm_cause.value
    }
}


// ============================================================================
// PDU Session Release Complete (3GPP TS 24.501 Section 8.3.15)
// ============================================================================

/// PDU Session Release Complete message (UE to network)
///
/// This message is sent by the UE to the network to acknowledge the release
/// of a PDU session.
///
/// 3GPP TS 24.501 Section 8.3.15
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PduSessionReleaseComplete {
    /// PDU Session ID (from header)
    pub pdu_session_id: u8,
    /// Procedure Transaction Identity (from header)
    pub pti: u8,
    /// Extended protocol configuration options (optional, Type 6, IEI 0x7B)
    pub extended_protocol_config_options: Option<Vec<u8>>,
    /// 5GSM cause (optional, Type 3, IEI 0x59)
    pub sm_cause: Option<Ie5gSmCause>,
}

impl PduSessionReleaseComplete {
    /// Create a new PDU Session Release Complete
    pub fn new(pdu_session_id: u8, pti: u8) -> Self {
        Self {
            pdu_session_id,
            pti,
            extended_protocol_config_options: None,
            sm_cause: None,
        }
    }

    /// Decode from bytes (after header has been parsed)
    pub fn decode<B: Buf>(
        buf: &mut B,
        pdu_session_id: u8,
        pti: u8,
    ) -> Result<Self, PduSessionReleaseError> {
        let mut msg = Self::new(pdu_session_id, pti);

        // Parse optional IEs
        while buf.remaining() > 0 {
            let iei = buf.chunk()[0];

            match iei {
                release_complete_iei::EXTENDED_PROTOCOL_CONFIG_OPTIONS => {
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
                release_complete_iei::SM_CAUSE => {
                    buf.advance(1);
                    if buf.remaining() < 1 {
                        break;
                    }
                    let cause_val = buf.get_u8();
                    let cause = SmCause::try_from(cause_val)
                        .unwrap_or(SmCause::ProtocolErrorUnspecified);
                    msg.sm_cause = Some(Ie5gSmCause::new(cause));
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
        let header = PlainSmHeader::new(
            self.pdu_session_id,
            self.pti,
            SmMessageType::PduSessionReleaseComplete,
        );
        header.encode(buf);

        // Optional IEs
        if let Some(ref epco) = self.extended_protocol_config_options {
            buf.put_u8(release_complete_iei::EXTENDED_PROTOCOL_CONFIG_OPTIONS);
            buf.put_u16(epco.len() as u16);
            buf.put_slice(epco);
        }

        if let Some(ref cause) = self.sm_cause {
            buf.put_u8(release_complete_iei::SM_CAUSE);
            buf.put_u8(cause.value as u8);
        }
    }

    /// Get the message type
    pub fn message_type() -> SmMessageType {
        SmMessageType::PduSessionReleaseComplete
    }
}


// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // PDU Session Release Request Tests
    // ========================================================================

    #[test]
    fn test_release_request_new() {
        let msg = PduSessionReleaseRequest::new(5, 1);
        assert_eq!(msg.pdu_session_id, 5);
        assert_eq!(msg.pti, 1);
        assert!(msg.sm_cause.is_none());
        assert!(msg.extended_protocol_config_options.is_none());
    }

    #[test]
    fn test_release_request_encode_minimal() {
        let msg = PduSessionReleaseRequest::new(5, 1);
        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Header only: EPD (1) + PDU Session ID (1) + PTI (1) + Message Type (1) = 4 bytes
        assert_eq!(buf.len(), 4);
        assert_eq!(buf[0], 0x2E); // EPD
        assert_eq!(buf[1], 5);    // PDU Session ID
        assert_eq!(buf[2], 1);    // PTI
        assert_eq!(buf[3], 0xD1); // Message Type
    }

    #[test]
    fn test_release_request_encode_decode_with_cause() {
        let mut msg = PduSessionReleaseRequest::new(5, 1);
        msg.sm_cause = Some(Ie5gSmCause::new(SmCause::RegularDeactivation));

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Skip header (4 bytes) and decode
        let decoded = PduSessionReleaseRequest::decode(&mut &buf[4..], 5, 1).unwrap();
        assert_eq!(decoded.sm_cause.unwrap().value, SmCause::RegularDeactivation);
    }

    #[test]
    fn test_release_request_encode_decode_with_epco() {
        let mut msg = PduSessionReleaseRequest::new(5, 1);
        msg.extended_protocol_config_options = Some(vec![0x01, 0x02, 0x03]);

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        let decoded = PduSessionReleaseRequest::decode(&mut &buf[4..], 5, 1).unwrap();
        assert_eq!(
            decoded.extended_protocol_config_options,
            Some(vec![0x01, 0x02, 0x03])
        );
    }

    #[test]
    fn test_release_request_message_type() {
        assert_eq!(
            PduSessionReleaseRequest::message_type(),
            SmMessageType::PduSessionReleaseRequest
        );
    }

    #[test]
    fn test_release_request_decode_empty_buffer() {
        let buf: &[u8] = &[];
        let result = PduSessionReleaseRequest::decode(&mut &buf[..], 5, 1);
        assert!(result.is_ok());
    }

    // ========================================================================
    // PDU Session Release Reject Tests
    // ========================================================================

    #[test]
    fn test_release_reject_new() {
        let msg = PduSessionReleaseReject::new(5, 1, SmCause::InsufficientResources);
        assert_eq!(msg.pdu_session_id, 5);
        assert_eq!(msg.pti, 1);
        assert_eq!(msg.sm_cause.value, SmCause::InsufficientResources);
    }

    #[test]
    fn test_release_reject_encode_minimal() {
        let msg = PduSessionReleaseReject::new(5, 1, SmCause::InsufficientResources);
        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Header (4) + Cause (1) = 5 bytes
        assert_eq!(buf.len(), 5);
        assert_eq!(buf[0], 0x2E); // EPD
        assert_eq!(buf[1], 5);    // PDU Session ID
        assert_eq!(buf[2], 1);    // PTI
        assert_eq!(buf[3], 0xD2); // Message Type
        assert_eq!(buf[4], SmCause::InsufficientResources as u8); // Cause
    }

    #[test]
    fn test_release_reject_encode_decode() {
        let msg = PduSessionReleaseReject::new(5, 1, SmCause::NetworkFailure);
        let mut buf = Vec::new();
        msg.encode(&mut buf);

        let decoded = PduSessionReleaseReject::decode(&mut &buf[4..], 5, 1).unwrap();
        assert_eq!(decoded.cause(), SmCause::NetworkFailure);
    }

    #[test]
    fn test_release_reject_encode_decode_with_epco() {
        let mut msg = PduSessionReleaseReject::new(5, 1, SmCause::NetworkFailure);
        msg.extended_protocol_config_options = Some(vec![0xAA, 0xBB]);

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        let decoded = PduSessionReleaseReject::decode(&mut &buf[4..], 5, 1).unwrap();
        assert_eq!(decoded.cause(), SmCause::NetworkFailure);
        assert_eq!(decoded.extended_protocol_config_options, Some(vec![0xAA, 0xBB]));
    }

    #[test]
    fn test_release_reject_message_type() {
        assert_eq!(
            PduSessionReleaseReject::message_type(),
            SmMessageType::PduSessionReleaseReject
        );
    }

    // ========================================================================
    // PDU Session Release Command Tests
    // ========================================================================

    #[test]
    fn test_release_command_new() {
        let msg = PduSessionReleaseCommand::new(5, 1, SmCause::RegularDeactivation);
        assert_eq!(msg.pdu_session_id, 5);
        assert_eq!(msg.pti, 1);
        assert_eq!(msg.sm_cause.value, SmCause::RegularDeactivation);
        assert!(msg.back_off_timer_value.is_none());
        assert!(msg.eap_message.is_none());
    }

    #[test]
    fn test_release_command_encode_minimal() {
        let msg = PduSessionReleaseCommand::new(5, 1, SmCause::RegularDeactivation);
        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Header (4) + Cause (1) = 5 bytes
        assert_eq!(buf.len(), 5);
        assert_eq!(buf[0], 0x2E); // EPD
        assert_eq!(buf[1], 5);    // PDU Session ID
        assert_eq!(buf[2], 1);    // PTI
        assert_eq!(buf[3], 0xD3); // Message Type
        assert_eq!(buf[4], SmCause::RegularDeactivation as u8); // Cause
    }

    #[test]
    fn test_release_command_encode_decode() {
        let msg = PduSessionReleaseCommand::new(5, 1, SmCause::NetworkFailure);
        let mut buf = Vec::new();
        msg.encode(&mut buf);

        let decoded = PduSessionReleaseCommand::decode(&mut &buf[4..], 5, 1).unwrap();
        assert_eq!(decoded.cause(), SmCause::NetworkFailure);
    }

    #[test]
    fn test_release_command_encode_decode_with_timer() {
        let mut msg = PduSessionReleaseCommand::new(5, 1, SmCause::RegularDeactivation);
        msg.back_off_timer_value = Some(0x21); // 1 minute

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        let decoded = PduSessionReleaseCommand::decode(&mut &buf[4..], 5, 1).unwrap();
        assert_eq!(decoded.cause(), SmCause::RegularDeactivation);
        assert_eq!(decoded.back_off_timer_value, Some(0x21));
    }

    #[test]
    fn test_release_command_encode_decode_with_eap() {
        let mut msg = PduSessionReleaseCommand::new(5, 1, SmCause::RegularDeactivation);
        msg.eap_message = Some(vec![0x01, 0x02, 0x03, 0x04]);

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        let decoded = PduSessionReleaseCommand::decode(&mut &buf[4..], 5, 1).unwrap();
        assert_eq!(decoded.eap_message, Some(vec![0x01, 0x02, 0x03, 0x04]));
    }

    #[test]
    fn test_release_command_message_type() {
        assert_eq!(
            PduSessionReleaseCommand::message_type(),
            SmMessageType::PduSessionReleaseCommand
        );
    }

    // ========================================================================
    // PDU Session Release Complete Tests
    // ========================================================================

    #[test]
    fn test_release_complete_new() {
        let msg = PduSessionReleaseComplete::new(5, 1);
        assert_eq!(msg.pdu_session_id, 5);
        assert_eq!(msg.pti, 1);
        assert!(msg.extended_protocol_config_options.is_none());
        assert!(msg.sm_cause.is_none());
    }

    #[test]
    fn test_release_complete_encode_minimal() {
        let msg = PduSessionReleaseComplete::new(5, 1);
        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Header only: 4 bytes
        assert_eq!(buf.len(), 4);
        assert_eq!(buf[0], 0x2E); // EPD
        assert_eq!(buf[1], 5);    // PDU Session ID
        assert_eq!(buf[2], 1);    // PTI
        assert_eq!(buf[3], 0xD4); // Message Type
    }

    #[test]
    fn test_release_complete_encode_decode_with_epco() {
        let mut msg = PduSessionReleaseComplete::new(5, 1);
        msg.extended_protocol_config_options = Some(vec![0x01, 0x02, 0x03]);

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        let decoded = PduSessionReleaseComplete::decode(&mut &buf[4..], 5, 1).unwrap();
        assert_eq!(
            decoded.extended_protocol_config_options,
            Some(vec![0x01, 0x02, 0x03])
        );
    }

    #[test]
    fn test_release_complete_encode_decode_with_cause() {
        let mut msg = PduSessionReleaseComplete::new(5, 1);
        msg.sm_cause = Some(Ie5gSmCause::new(SmCause::RegularDeactivation));

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        let decoded = PduSessionReleaseComplete::decode(&mut &buf[4..], 5, 1).unwrap();
        assert_eq!(decoded.sm_cause.unwrap().value, SmCause::RegularDeactivation);
    }

    #[test]
    fn test_release_complete_message_type() {
        assert_eq!(
            PduSessionReleaseComplete::message_type(),
            SmMessageType::PduSessionReleaseComplete
        );
    }

    #[test]
    fn test_release_complete_decode_empty_buffer() {
        let buf: &[u8] = &[];
        let result = PduSessionReleaseComplete::decode(&mut &buf[..], 5, 1);
        assert!(result.is_ok());
    }
}
