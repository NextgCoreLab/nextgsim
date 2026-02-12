//! PDU Session Modification Messages (3GPP TS 24.501 Section 8.3.7-8.3.11)
//!
//! This module implements the PDU Session Modification procedure messages:
//! - PDU Session Modification Request (UE to network)
//! - PDU Session Modification Reject (network to UE)
//! - PDU Session Modification Command (network to UE)
//! - PDU Session Modification Complete (UE to network)

use bytes::{Buf, BufMut};
use thiserror::Error;

use crate::enums::SmMessageType;
use crate::header::PlainSmHeader;
use crate::ies::ie3::{Ie5gSmCause, SmCause};

/// Error type for PDU Session Modification message encoding/decoding
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PduSessionModificationError {
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
// IEI Constants for PDU Session Modification Messages
// ============================================================================

/// IEI values for PDU Session Modification Request optional IEs
#[allow(dead_code)]
mod modification_request_iei {
    /// 5GSM capability
    pub const SM_CAPABILITY: u8 = 0x28;
    /// 5GSM cause
    pub const SM_CAUSE: u8 = 0x59;
    /// Maximum number of supported packet filters
    pub const MAX_PACKET_FILTERS: u8 = 0x55;
    /// Always-on PDU session requested
    pub const ALWAYS_ON_PDU_SESSION_REQUESTED: u8 = 0xB;
    /// Integrity protection maximum data rate
    pub const INTEGRITY_PROTECTION_MAX_DATA_RATE: u8 = 0x13;
    /// Requested `QoS` rules
    pub const REQUESTED_QOS_RULES: u8 = 0x7A;
    /// Requested `QoS` flow descriptions
    pub const REQUESTED_QOS_FLOW_DESCRIPTIONS: u8 = 0x79;
    /// Mapped EPS bearer contexts
    pub const MAPPED_EPS_BEARER_CONTEXTS: u8 = 0x75;
    /// Extended protocol configuration options
    pub const EXTENDED_PROTOCOL_CONFIG_OPTIONS: u8 = 0x7B;
}

/// IEI values for PDU Session Modification Reject optional IEs
#[allow(dead_code)]
mod modification_reject_iei {
    /// Back-off timer value
    pub const BACK_OFF_TIMER_VALUE: u8 = 0x37;
    /// Extended protocol configuration options
    pub const EXTENDED_PROTOCOL_CONFIG_OPTIONS: u8 = 0x7B;
}

/// IEI values for PDU Session Modification Command optional IEs
#[allow(dead_code)]
mod modification_command_iei {
    /// 5GSM cause
    pub const SM_CAUSE: u8 = 0x59;
    /// Session AMBR
    pub const SESSION_AMBR: u8 = 0x2A;
    /// RQ timer value
    pub const RQ_TIMER_VALUE: u8 = 0x56;
    /// Always-on PDU session indication
    pub const ALWAYS_ON_PDU_SESSION_INDICATION: u8 = 0x8;
    /// Authorized `QoS` rules
    pub const AUTHORIZED_QOS_RULES: u8 = 0x7A;
    /// Mapped EPS bearer contexts
    pub const MAPPED_EPS_BEARER_CONTEXTS: u8 = 0x75;
    /// Authorized `QoS` flow descriptions
    pub const AUTHORIZED_QOS_FLOW_DESCRIPTIONS: u8 = 0x79;
    /// Extended protocol configuration options
    pub const EXTENDED_PROTOCOL_CONFIG_OPTIONS: u8 = 0x7B;
}

/// IEI values for PDU Session Modification Complete optional IEs
#[allow(dead_code)]
mod modification_complete_iei {
    /// Extended protocol configuration options
    pub const EXTENDED_PROTOCOL_CONFIG_OPTIONS: u8 = 0x7B;
}


// ============================================================================
// PDU Session Modification Request (3GPP TS 24.501 Section 8.3.7)
// ============================================================================

/// PDU Session Modification Request message (UE to network)
///
/// This message is sent by the UE to the network to request modification
/// of an existing PDU session.
///
/// 3GPP TS 24.501 Section 8.3.7
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PduSessionModificationRequest {
    /// PDU Session ID (from header)
    pub pdu_session_id: u8,
    /// Procedure Transaction Identity (from header)
    pub pti: u8,
    /// 5GSM capability (optional, Type 4, IEI 0x28)
    pub sm_capability: Option<Vec<u8>>,
    /// 5GSM cause (optional, Type 3, IEI 0x59)
    pub sm_cause: Option<Ie5gSmCause>,
    /// Maximum number of supported packet filters (optional, Type 3, IEI 0x55)
    pub max_packet_filters: Option<u16>,
    /// Always-on PDU session requested (optional, Type 1, IEI 0xB)
    pub always_on_pdu_session_requested: Option<bool>,
    /// Integrity protection maximum data rate (optional, Type 4, IEI 0x13)
    pub integrity_protection_max_data_rate: Option<[u8; 2]>,
    /// Requested `QoS` rules (optional, Type 6, IEI 0x7A)
    pub requested_qos_rules: Option<Vec<u8>>,
    /// Requested `QoS` flow descriptions (optional, Type 6, IEI 0x79)
    pub requested_qos_flow_descriptions: Option<Vec<u8>>,
    /// Mapped EPS bearer contexts (optional, Type 6, IEI 0x75)
    pub mapped_eps_bearer_contexts: Option<Vec<u8>>,
    /// Extended protocol configuration options (optional, Type 6, IEI 0x7B)
    pub extended_protocol_config_options: Option<Vec<u8>>,
}

impl PduSessionModificationRequest {
    /// Create a new PDU Session Modification Request
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
    ) -> Result<Self, PduSessionModificationError> {
        let mut msg = Self::new(pdu_session_id, pti);

        // All IEs are optional, parse until buffer is empty
        while buf.remaining() > 0 {
            let iei = buf.chunk()[0];

            // Check for Type 1 IEs (4-bit IEI in high nibble)
            let iei_high = (iei >> 4) & 0x0F;
            if iei_high == 0xB {
                // Always-on PDU session requested
                buf.advance(1);
                msg.always_on_pdu_session_requested = Some((iei & 0x01) == 0x01);
                continue;
            }

            // Full octet IEI
            match iei {
                modification_request_iei::SM_CAPABILITY => {
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
                modification_request_iei::SM_CAUSE => {
                    buf.advance(1);
                    if buf.remaining() < 1 {
                        break;
                    }
                    let cause_val = buf.get_u8();
                    let cause = SmCause::try_from(cause_val).unwrap_or(SmCause::ProtocolErrorUnspecified);
                    msg.sm_cause = Some(Ie5gSmCause::new(cause));
                }
                modification_request_iei::MAX_PACKET_FILTERS => {
                    buf.advance(1);
                    if buf.remaining() < 2 {
                        break;
                    }
                    let high = buf.get_u8() as u16;
                    let low = buf.get_u8() as u16;
                    msg.max_packet_filters = Some((high << 3) | ((low >> 5) & 0x07));
                }
                modification_request_iei::INTEGRITY_PROTECTION_MAX_DATA_RATE => {
                    buf.advance(1);
                    if buf.remaining() < 1 {
                        break;
                    }
                    let len = buf.get_u8() as usize;
                    if buf.remaining() < len || len < 2 {
                        break;
                    }
                    let mut data = [0u8; 2];
                    data[0] = buf.get_u8();
                    data[1] = buf.get_u8();
                    if len > 2 {
                        buf.advance(len - 2);
                    }
                    msg.integrity_protection_max_data_rate = Some(data);
                }
                modification_request_iei::REQUESTED_QOS_RULES => {
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
                    msg.requested_qos_rules = Some(data);
                }
                modification_request_iei::REQUESTED_QOS_FLOW_DESCRIPTIONS => {
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
                    msg.requested_qos_flow_descriptions = Some(data);
                }
                modification_request_iei::MAPPED_EPS_BEARER_CONTEXTS => {
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
                    msg.mapped_eps_bearer_contexts = Some(data);
                }
                modification_request_iei::EXTENDED_PROTOCOL_CONFIG_OPTIONS => {
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
            SmMessageType::PduSessionModificationRequest,
        );
        header.encode(buf);

        // Optional IEs
        if let Some(ref cap) = self.sm_capability {
            buf.put_u8(modification_request_iei::SM_CAPABILITY);
            buf.put_u8(cap.len() as u8);
            buf.put_slice(cap);
        }

        if let Some(ref cause) = self.sm_cause {
            buf.put_u8(modification_request_iei::SM_CAUSE);
            buf.put_u8(cause.value as u8);
        }

        if let Some(max_filters) = self.max_packet_filters {
            buf.put_u8(modification_request_iei::MAX_PACKET_FILTERS);
            let high = ((max_filters >> 3) & 0xFF) as u8;
            let low = ((max_filters & 0x07) << 5) as u8;
            buf.put_u8(high);
            buf.put_u8(low);
        }

        if let Some(requested) = self.always_on_pdu_session_requested {
            let val = if requested { 0x01 } else { 0x00 };
            buf.put_u8((0xB << 4) | val);
        }

        if let Some(ref rate) = self.integrity_protection_max_data_rate {
            buf.put_u8(modification_request_iei::INTEGRITY_PROTECTION_MAX_DATA_RATE);
            buf.put_u8(2);
            buf.put_slice(rate);
        }

        if let Some(ref rules) = self.requested_qos_rules {
            buf.put_u8(modification_request_iei::REQUESTED_QOS_RULES);
            buf.put_u16(rules.len() as u16);
            buf.put_slice(rules);
        }

        if let Some(ref desc) = self.requested_qos_flow_descriptions {
            buf.put_u8(modification_request_iei::REQUESTED_QOS_FLOW_DESCRIPTIONS);
            buf.put_u16(desc.len() as u16);
            buf.put_slice(desc);
        }

        if let Some(ref contexts) = self.mapped_eps_bearer_contexts {
            buf.put_u8(modification_request_iei::MAPPED_EPS_BEARER_CONTEXTS);
            buf.put_u16(contexts.len() as u16);
            buf.put_slice(contexts);
        }

        if let Some(ref epco) = self.extended_protocol_config_options {
            buf.put_u8(modification_request_iei::EXTENDED_PROTOCOL_CONFIG_OPTIONS);
            buf.put_u16(epco.len() as u16);
            buf.put_slice(epco);
        }
    }

    /// Get the message type
    pub fn message_type() -> SmMessageType {
        SmMessageType::PduSessionModificationRequest
    }
}


// ============================================================================
// PDU Session Modification Reject (3GPP TS 24.501 Section 8.3.8)
// ============================================================================

/// PDU Session Modification Reject message (network to UE)
///
/// This message is sent by the network to the UE to reject a PDU session
/// modification request.
///
/// 3GPP TS 24.501 Section 8.3.8
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PduSessionModificationReject {
    /// PDU Session ID (from header)
    pub pdu_session_id: u8,
    /// Procedure Transaction Identity (from header)
    pub pti: u8,
    /// 5GSM cause (mandatory, Type 3)
    pub sm_cause: Ie5gSmCause,
    /// Back-off timer value (optional, Type 4, IEI 0x37)
    pub back_off_timer_value: Option<u8>,
    /// Extended protocol configuration options (optional, Type 6, IEI 0x7B)
    pub extended_protocol_config_options: Option<Vec<u8>>,
}

impl Default for PduSessionModificationReject {
    fn default() -> Self {
        Self {
            pdu_session_id: 0,
            pti: 0,
            sm_cause: Ie5gSmCause::new(SmCause::ProtocolErrorUnspecified),
            back_off_timer_value: None,
            extended_protocol_config_options: None,
        }
    }
}

impl PduSessionModificationReject {
    /// Create a new PDU Session Modification Reject
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
    ) -> Result<Self, PduSessionModificationError> {
        // 5GSM cause (mandatory, Type 3 - 1 byte)
        if buf.remaining() < 1 {
            return Err(PduSessionModificationError::BufferTooShort {
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
                modification_reject_iei::BACK_OFF_TIMER_VALUE => {
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
                modification_reject_iei::EXTENDED_PROTOCOL_CONFIG_OPTIONS => {
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
            SmMessageType::PduSessionModificationReject,
        );
        header.encode(buf);

        // 5GSM cause (mandatory)
        buf.put_u8(self.sm_cause.value as u8);

        // Optional IEs
        if let Some(timer) = self.back_off_timer_value {
            buf.put_u8(modification_reject_iei::BACK_OFF_TIMER_VALUE);
            buf.put_u8(1);
            buf.put_u8(timer);
        }

        if let Some(ref epco) = self.extended_protocol_config_options {
            buf.put_u8(modification_reject_iei::EXTENDED_PROTOCOL_CONFIG_OPTIONS);
            buf.put_u16(epco.len() as u16);
            buf.put_slice(epco);
        }
    }

    /// Get the message type
    pub fn message_type() -> SmMessageType {
        SmMessageType::PduSessionModificationReject
    }

    /// Get the cause value
    pub fn cause(&self) -> SmCause {
        self.sm_cause.value
    }
}


// ============================================================================
// PDU Session Modification Command (3GPP TS 24.501 Section 8.3.9)
// ============================================================================

/// PDU Session Modification Command message (network to UE)
///
/// This message is sent by the network to the UE to modify an existing
/// PDU session.
///
/// 3GPP TS 24.501 Section 8.3.9
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PduSessionModificationCommand {
    /// PDU Session ID (from header)
    pub pdu_session_id: u8,
    /// Procedure Transaction Identity (from header)
    pub pti: u8,
    /// 5GSM cause (optional, Type 3, IEI 0x59)
    pub sm_cause: Option<Ie5gSmCause>,
    /// Session AMBR (optional, Type 4, IEI 0x2A)
    pub session_ambr: Option<Vec<u8>>,
    /// RQ timer value (optional, Type 3, IEI 0x56)
    pub rq_timer_value: Option<u8>,
    /// Always-on PDU session indication (optional, Type 1, IEI 0x8)
    pub always_on_pdu_session_indication: Option<bool>,
    /// Authorized `QoS` rules (optional, Type 6, IEI 0x7A)
    pub authorized_qos_rules: Option<Vec<u8>>,
    /// Mapped EPS bearer contexts (optional, Type 6, IEI 0x75)
    pub mapped_eps_bearer_contexts: Option<Vec<u8>>,
    /// Authorized `QoS` flow descriptions (optional, Type 6, IEI 0x79)
    pub authorized_qos_flow_descriptions: Option<Vec<u8>>,
    /// Extended protocol configuration options (optional, Type 6, IEI 0x7B)
    pub extended_protocol_config_options: Option<Vec<u8>>,
}

impl PduSessionModificationCommand {
    /// Create a new PDU Session Modification Command
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
    ) -> Result<Self, PduSessionModificationError> {
        let mut msg = Self::new(pdu_session_id, pti);

        // All IEs are optional, parse until buffer is empty
        while buf.remaining() > 0 {
            let iei = buf.chunk()[0];

            // Check for Type 1 IEs (4-bit IEI in high nibble)
            let iei_high = (iei >> 4) & 0x0F;
            if iei_high == 0x8 {
                // Always-on PDU session indication
                buf.advance(1);
                msg.always_on_pdu_session_indication = Some((iei & 0x01) == 0x01);
                continue;
            }

            // Full octet IEI
            match iei {
                modification_command_iei::SM_CAUSE => {
                    buf.advance(1);
                    if buf.remaining() < 1 {
                        break;
                    }
                    let cause_val = buf.get_u8();
                    let cause = SmCause::try_from(cause_val).unwrap_or(SmCause::ProtocolErrorUnspecified);
                    msg.sm_cause = Some(Ie5gSmCause::new(cause));
                }
                modification_command_iei::SESSION_AMBR => {
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
                    msg.session_ambr = Some(data);
                }
                modification_command_iei::RQ_TIMER_VALUE => {
                    buf.advance(1);
                    if buf.remaining() < 1 {
                        break;
                    }
                    msg.rq_timer_value = Some(buf.get_u8());
                }
                modification_command_iei::AUTHORIZED_QOS_RULES => {
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
                    msg.authorized_qos_rules = Some(data);
                }
                modification_command_iei::MAPPED_EPS_BEARER_CONTEXTS => {
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
                    msg.mapped_eps_bearer_contexts = Some(data);
                }
                modification_command_iei::AUTHORIZED_QOS_FLOW_DESCRIPTIONS => {
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
                    msg.authorized_qos_flow_descriptions = Some(data);
                }
                modification_command_iei::EXTENDED_PROTOCOL_CONFIG_OPTIONS => {
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
            SmMessageType::PduSessionModificationCommand,
        );
        header.encode(buf);

        // Optional IEs
        if let Some(ref cause) = self.sm_cause {
            buf.put_u8(modification_command_iei::SM_CAUSE);
            buf.put_u8(cause.value as u8);
        }

        if let Some(ref ambr) = self.session_ambr {
            buf.put_u8(modification_command_iei::SESSION_AMBR);
            buf.put_u8(ambr.len() as u8);
            buf.put_slice(ambr);
        }

        if let Some(timer) = self.rq_timer_value {
            buf.put_u8(modification_command_iei::RQ_TIMER_VALUE);
            buf.put_u8(timer);
        }

        if let Some(indication) = self.always_on_pdu_session_indication {
            let val = if indication { 0x01 } else { 0x00 };
            buf.put_u8((0x8 << 4) | val);
        }

        if let Some(ref rules) = self.authorized_qos_rules {
            buf.put_u8(modification_command_iei::AUTHORIZED_QOS_RULES);
            buf.put_u16(rules.len() as u16);
            buf.put_slice(rules);
        }

        if let Some(ref contexts) = self.mapped_eps_bearer_contexts {
            buf.put_u8(modification_command_iei::MAPPED_EPS_BEARER_CONTEXTS);
            buf.put_u16(contexts.len() as u16);
            buf.put_slice(contexts);
        }

        if let Some(ref desc) = self.authorized_qos_flow_descriptions {
            buf.put_u8(modification_command_iei::AUTHORIZED_QOS_FLOW_DESCRIPTIONS);
            buf.put_u16(desc.len() as u16);
            buf.put_slice(desc);
        }

        if let Some(ref epco) = self.extended_protocol_config_options {
            buf.put_u8(modification_command_iei::EXTENDED_PROTOCOL_CONFIG_OPTIONS);
            buf.put_u16(epco.len() as u16);
            buf.put_slice(epco);
        }
    }

    /// Get the message type
    pub fn message_type() -> SmMessageType {
        SmMessageType::PduSessionModificationCommand
    }
}


// ============================================================================
// PDU Session Modification Complete (3GPP TS 24.501 Section 8.3.10)
// ============================================================================

/// PDU Session Modification Complete message (UE to network)
///
/// This message is sent by the UE to the network to acknowledge a PDU session
/// modification command.
///
/// 3GPP TS 24.501 Section 8.3.10
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PduSessionModificationComplete {
    /// PDU Session ID (from header)
    pub pdu_session_id: u8,
    /// Procedure Transaction Identity (from header)
    pub pti: u8,
    /// Extended protocol configuration options (optional, Type 6, IEI 0x7B)
    pub extended_protocol_config_options: Option<Vec<u8>>,
}

impl PduSessionModificationComplete {
    /// Create a new PDU Session Modification Complete
    pub fn new(pdu_session_id: u8, pti: u8) -> Self {
        Self {
            pdu_session_id,
            pti,
            extended_protocol_config_options: None,
        }
    }

    /// Decode from bytes (after header has been parsed)
    pub fn decode<B: Buf>(
        buf: &mut B,
        pdu_session_id: u8,
        pti: u8,
    ) -> Result<Self, PduSessionModificationError> {
        let mut msg = Self::new(pdu_session_id, pti);

        // Parse optional IEs
        while buf.remaining() > 0 {
            let iei = buf.chunk()[0];

            match iei {
                modification_complete_iei::EXTENDED_PROTOCOL_CONFIG_OPTIONS => {
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
            SmMessageType::PduSessionModificationComplete,
        );
        header.encode(buf);

        // Optional IEs
        if let Some(ref epco) = self.extended_protocol_config_options {
            buf.put_u8(modification_complete_iei::EXTENDED_PROTOCOL_CONFIG_OPTIONS);
            buf.put_u16(epco.len() as u16);
            buf.put_slice(epco);
        }
    }

    /// Get the message type
    pub fn message_type() -> SmMessageType {
        SmMessageType::PduSessionModificationComplete
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // PDU Session Modification Request Tests
    // ========================================================================

    #[test]
    fn test_modification_request_new() {
        let msg = PduSessionModificationRequest::new(5, 1);
        assert_eq!(msg.pdu_session_id, 5);
        assert_eq!(msg.pti, 1);
        assert!(msg.sm_capability.is_none());
        assert!(msg.sm_cause.is_none());
    }

    #[test]
    fn test_modification_request_encode_minimal() {
        let msg = PduSessionModificationRequest::new(5, 1);
        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Header only: EPD (1) + PDU Session ID (1) + PTI (1) + Message Type (1) = 4 bytes
        assert_eq!(buf.len(), 4);
        assert_eq!(buf[0], 0x2E); // EPD
        assert_eq!(buf[1], 5);    // PDU Session ID
        assert_eq!(buf[2], 1);    // PTI
        assert_eq!(buf[3], 0xC9); // Message Type
    }

    #[test]
    fn test_modification_request_encode_decode_with_cause() {
        let mut msg = PduSessionModificationRequest::new(5, 1);
        msg.sm_cause = Some(Ie5gSmCause::new(SmCause::RegularDeactivation));

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Skip header (4 bytes) and decode
        let decoded = PduSessionModificationRequest::decode(&mut &buf[4..], 5, 1).unwrap();
        assert_eq!(decoded.sm_cause.unwrap().value, SmCause::RegularDeactivation);
    }

    #[test]
    fn test_modification_request_message_type() {
        assert_eq!(
            PduSessionModificationRequest::message_type(),
            SmMessageType::PduSessionModificationRequest
        );
    }

    // ========================================================================
    // PDU Session Modification Reject Tests
    // ========================================================================

    #[test]
    fn test_modification_reject_new() {
        let msg = PduSessionModificationReject::new(5, 1, SmCause::InsufficientResources);
        assert_eq!(msg.pdu_session_id, 5);
        assert_eq!(msg.pti, 1);
        assert_eq!(msg.sm_cause.value, SmCause::InsufficientResources);
    }

    #[test]
    fn test_modification_reject_encode_minimal() {
        let msg = PduSessionModificationReject::new(5, 1, SmCause::InsufficientResources);
        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Header (4) + Cause (1) = 5 bytes
        assert_eq!(buf.len(), 5);
        assert_eq!(buf[0], 0x2E); // EPD
        assert_eq!(buf[1], 5);    // PDU Session ID
        assert_eq!(buf[2], 1);    // PTI
        assert_eq!(buf[3], 0xCA); // Message Type
        assert_eq!(buf[4], 0x1A); // Cause: InsufficientResources
    }

    #[test]
    fn test_modification_reject_encode_decode() {
        let msg = PduSessionModificationReject::new(5, 1, SmCause::NetworkFailure);
        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Skip header (4 bytes) and decode
        let decoded = PduSessionModificationReject::decode(&mut &buf[4..], 5, 1).unwrap();
        assert_eq!(decoded.cause(), SmCause::NetworkFailure);
    }

    #[test]
    fn test_modification_reject_message_type() {
        assert_eq!(
            PduSessionModificationReject::message_type(),
            SmMessageType::PduSessionModificationReject
        );
    }

    // ========================================================================
    // PDU Session Modification Command Tests
    // ========================================================================

    #[test]
    fn test_modification_command_new() {
        let msg = PduSessionModificationCommand::new(5, 1);
        assert_eq!(msg.pdu_session_id, 5);
        assert_eq!(msg.pti, 1);
        assert!(msg.sm_cause.is_none());
        assert!(msg.session_ambr.is_none());
    }

    #[test]
    fn test_modification_command_encode_minimal() {
        let msg = PduSessionModificationCommand::new(5, 1);
        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Header only: 4 bytes
        assert_eq!(buf.len(), 4);
        assert_eq!(buf[0], 0x2E); // EPD
        assert_eq!(buf[1], 5);    // PDU Session ID
        assert_eq!(buf[2], 1);    // PTI
        assert_eq!(buf[3], 0xCB); // Message Type
    }

    #[test]
    fn test_modification_command_encode_decode_with_cause() {
        let mut msg = PduSessionModificationCommand::new(5, 1);
        msg.sm_cause = Some(Ie5gSmCause::new(SmCause::ReactivationRequested));

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Skip header (4 bytes) and decode
        let decoded = PduSessionModificationCommand::decode(&mut &buf[4..], 5, 1).unwrap();
        assert_eq!(decoded.sm_cause.unwrap().value, SmCause::ReactivationRequested);
    }

    #[test]
    fn test_modification_command_message_type() {
        assert_eq!(
            PduSessionModificationCommand::message_type(),
            SmMessageType::PduSessionModificationCommand
        );
    }

    // ========================================================================
    // PDU Session Modification Complete Tests
    // ========================================================================

    #[test]
    fn test_modification_complete_new() {
        let msg = PduSessionModificationComplete::new(5, 1);
        assert_eq!(msg.pdu_session_id, 5);
        assert_eq!(msg.pti, 1);
        assert!(msg.extended_protocol_config_options.is_none());
    }

    #[test]
    fn test_modification_complete_encode_minimal() {
        let msg = PduSessionModificationComplete::new(5, 1);
        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Header only: 4 bytes
        assert_eq!(buf.len(), 4);
        assert_eq!(buf[0], 0x2E); // EPD
        assert_eq!(buf[1], 5);    // PDU Session ID
        assert_eq!(buf[2], 1);    // PTI
        assert_eq!(buf[3], 0xCC); // Message Type
    }

    #[test]
    fn test_modification_complete_encode_decode_with_epco() {
        let mut msg = PduSessionModificationComplete::new(5, 1);
        msg.extended_protocol_config_options = Some(vec![0x01, 0x02, 0x03]);

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Skip header (4 bytes) and decode
        let decoded = PduSessionModificationComplete::decode(&mut &buf[4..], 5, 1).unwrap();
        assert_eq!(
            decoded.extended_protocol_config_options,
            Some(vec![0x01, 0x02, 0x03])
        );
    }

    #[test]
    fn test_modification_complete_message_type() {
        assert_eq!(
            PduSessionModificationComplete::message_type(),
            SmMessageType::PduSessionModificationComplete
        );
    }

    #[test]
    fn test_modification_complete_decode_empty_buffer() {
        let buf: &[u8] = &[];
        let result = PduSessionModificationComplete::decode(&mut &buf[..], 5, 1);
        // Should succeed with no optional IEs
        assert!(result.is_ok());
    }
}
