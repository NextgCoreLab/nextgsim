//! Service Messages (3GPP TS 24.501 Section 8.2.15, 8.2.16, 8.2.17)
//!
//! This module implements the Service procedure messages:
//! - Service Request (UE to network)
//! - Service Accept (network to UE)
//! - Service Reject (network to UE)

use bytes::{Buf, BufMut};
use thiserror::Error;

use crate::enums::MmMessageType;
use crate::header::PlainMmHeader;
use crate::ies::ie1::{IeServiceType, InformationElement1, ServiceType};
use crate::security::NasKeySetIdentifier;

use super::registration::{Ie5gMmCause, Ie5gsMobileIdentity, MmCause};

/// Error type for Service message encoding/decoding
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ServiceError {
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
    /// Unknown IEI
    #[error("Unknown IEI: 0x{0:02X}")]
    UnknownIei(u8),
    /// Header decoding error
    #[error("Header error: {0}")]
    HeaderError(#[from] crate::header::HeaderError),
}

// ============================================================================
// IEI values for Service messages
// ============================================================================

/// IEI values for Service Request optional IEs
#[allow(dead_code)]
mod service_request_iei {
    /// Uplink data status
    pub const UPLINK_DATA_STATUS: u8 = 0x40;
    /// PDU session status
    pub const PDU_SESSION_STATUS: u8 = 0x50;
    /// Allowed PDU session status
    pub const ALLOWED_PDU_SESSION_STATUS: u8 = 0x25;
    /// NAS message container
    pub const NAS_MESSAGE_CONTAINER: u8 = 0x71;
}

/// IEI values for Service Accept optional IEs
#[allow(dead_code)]
mod service_accept_iei {
    /// PDU session status
    pub const PDU_SESSION_STATUS: u8 = 0x50;
    /// PDU session reactivation result
    pub const PDU_SESSION_REACTIVATION_RESULT: u8 = 0x26;
    /// PDU session reactivation result error cause
    pub const PDU_SESSION_REACTIVATION_RESULT_ERROR_CAUSE: u8 = 0x72;
    /// EAP message
    pub const EAP_MESSAGE: u8 = 0x78;
}

/// IEI values for Service Reject optional IEs
#[allow(dead_code)]
mod service_reject_iei {
    /// PDU session status
    pub const PDU_SESSION_STATUS: u8 = 0x50;
    /// T3346 value (GPRS timer 2)
    pub const T3346_VALUE: u8 = 0x5F;
    /// EAP message
    pub const EAP_MESSAGE: u8 = 0x78;
}

// ============================================================================
// Service Request Message (3GPP TS 24.501 Section 8.2.15)
// ============================================================================

/// Service Request message (UE to network)
///
/// This message is sent by the UE to the network to request a service.
///
/// ## Message Structure
/// ```text
/// +------------------+------------------+------------------+
/// |       EPD        |  Security Header |   Message Type   |
/// |     (1 byte)     |  Type (1 byte)   |    (1 byte)      |
/// +------------------+------------------+------------------+
/// | ngKSI | Service  |        5G-S-TMSI (variable)         |
/// |(4 bits)| Type    |                                     |
/// +------------------+-------------------------------------+
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceRequest {
    /// ngKSI - NAS key set identifier (mandatory, Type 1)
    pub ng_ksi: NasKeySetIdentifier,
    /// Service type (mandatory, Type 1)
    pub service_type: IeServiceType,
    /// 5G-S-TMSI (mandatory, Type 6)
    pub tmsi: Ie5gsMobileIdentity,
    /// Uplink data status (optional, Type 4, IEI 0x40)
    pub uplink_data_status: Option<u16>,
    /// PDU session status (optional, Type 4, IEI 0x50)
    pub pdu_session_status: Option<u16>,
    /// Allowed PDU session status (optional, Type 4, IEI 0x25)
    pub allowed_pdu_session_status: Option<u16>,
    /// NAS message container (optional, Type 6, IEI 0x71)
    pub nas_message_container: Option<Vec<u8>>,
}

impl Default for ServiceRequest {
    fn default() -> Self {
        Self {
            ng_ksi: NasKeySetIdentifier::no_key(),
            service_type: IeServiceType::new(ServiceType::Signalling),
            tmsi: Ie5gsMobileIdentity::no_identity(),
            uplink_data_status: None,
            pdu_session_status: None,
            allowed_pdu_session_status: None,
            nas_message_container: None,
        }
    }
}

impl ServiceRequest {
    /// Create a new Service Request with mandatory fields
    pub fn new(
        ng_ksi: NasKeySetIdentifier,
        service_type: IeServiceType,
        tmsi: Ie5gsMobileIdentity,
    ) -> Self {
        Self {
            ng_ksi,
            service_type,
            tmsi,
            ..Default::default()
        }
    }

    /// Decode from bytes (after header has been parsed)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, ServiceError> {
        if buf.remaining() < 1 {
            return Err(ServiceError::BufferTooShort {
                expected: 1,
                actual: buf.remaining(),
            });
        }

        // First octet: ngKSI (high nibble) + Service type (low nibble)
        let first_octet = buf.get_u8();
        let ng_ksi = NasKeySetIdentifier::decode((first_octet >> 4) & 0x0F)
            .map_err(|e| ServiceError::InvalidIeValue(e.to_string()))?;
        let service_type = IeServiceType::decode(first_octet & 0x0F)
            .map_err(|e| ServiceError::InvalidIeValue(e.to_string()))?;

        // 5G-S-TMSI (mandatory, Type 6)
        let tmsi = Ie5gsMobileIdentity::decode(buf)
            .map_err(|e| ServiceError::InvalidIeValue(e.to_string()))?;

        let mut msg = Self::new(ng_ksi, service_type, tmsi);

        // Parse optional IEs
        while buf.remaining() > 0 {
            let iei = buf.chunk()[0];

            match iei {
                service_request_iei::UPLINK_DATA_STATUS => {
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
                service_request_iei::PDU_SESSION_STATUS => {
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
                service_request_iei::ALLOWED_PDU_SESSION_STATUS => {
                    buf.advance(1);
                    if buf.remaining() < 1 {
                        break;
                    }
                    let len = buf.get_u8() as usize;
                    if buf.remaining() < len || len < 2 {
                        break;
                    }
                    msg.allowed_pdu_session_status = Some(buf.get_u16());
                    if len > 2 {
                        buf.advance(len - 2);
                    }
                }
                service_request_iei::NAS_MESSAGE_CONTAINER => {
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
        let header = PlainMmHeader::new(MmMessageType::ServiceRequest);
        header.encode(buf);

        // First octet: ngKSI (high nibble) + Service type (low nibble)
        let first_octet = (self.ng_ksi.encode() << 4) | (self.service_type.encode() & 0x0F);
        buf.put_u8(first_octet);

        // 5G-S-TMSI (mandatory)
        self.tmsi.encode(buf);

        // Optional IEs
        if let Some(status) = self.uplink_data_status {
            buf.put_u8(service_request_iei::UPLINK_DATA_STATUS);
            buf.put_u8(2);
            buf.put_u16(status);
        }

        if let Some(status) = self.pdu_session_status {
            buf.put_u8(service_request_iei::PDU_SESSION_STATUS);
            buf.put_u8(2);
            buf.put_u16(status);
        }

        if let Some(status) = self.allowed_pdu_session_status {
            buf.put_u8(service_request_iei::ALLOWED_PDU_SESSION_STATUS);
            buf.put_u8(2);
            buf.put_u16(status);
        }

        if let Some(ref container) = self.nas_message_container {
            buf.put_u8(service_request_iei::NAS_MESSAGE_CONTAINER);
            buf.put_u16(container.len() as u16);
            buf.put_slice(container);
        }
    }

    /// Get the message type
    pub fn message_type() -> MmMessageType {
        MmMessageType::ServiceRequest
    }
}

// ============================================================================
// Service Accept Message (3GPP TS 24.501 Section 8.2.16)
// ============================================================================

/// Service Accept message (network to UE)
///
/// This message is sent by the network to the UE to indicate that
/// the requested service has been accepted.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ServiceAccept {
    /// PDU session status (optional, Type 4, IEI 0x50)
    pub pdu_session_status: Option<u16>,
    /// PDU session reactivation result (optional, Type 4, IEI 0x26)
    pub pdu_session_reactivation_result: Option<u16>,
    /// PDU session reactivation result error cause (optional, Type 6, IEI 0x72)
    pub pdu_session_reactivation_result_error_cause: Option<Vec<u8>>,
    /// EAP message (optional, Type 6, IEI 0x78)
    pub eap_message: Option<Vec<u8>>,
}

impl ServiceAccept {
    /// Create a new Service Accept message
    pub fn new() -> Self {
        Self::default()
    }

    /// Decode from bytes (after header has been parsed)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, ServiceError> {
        let mut msg = Self::new();

        // Parse optional IEs
        while buf.remaining() > 0 {
            let iei = buf.chunk()[0];

            match iei {
                service_accept_iei::PDU_SESSION_STATUS => {
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
                service_accept_iei::PDU_SESSION_REACTIVATION_RESULT => {
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
                service_accept_iei::PDU_SESSION_REACTIVATION_RESULT_ERROR_CAUSE => {
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
                    msg.pdu_session_reactivation_result_error_cause = Some(data);
                }
                service_accept_iei::EAP_MESSAGE => {
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
        let header = PlainMmHeader::new(MmMessageType::ServiceAccept);
        header.encode(buf);

        // Optional IEs
        if let Some(status) = self.pdu_session_status {
            buf.put_u8(service_accept_iei::PDU_SESSION_STATUS);
            buf.put_u8(2);
            buf.put_u16(status);
        }

        if let Some(result) = self.pdu_session_reactivation_result {
            buf.put_u8(service_accept_iei::PDU_SESSION_REACTIVATION_RESULT);
            buf.put_u8(2);
            buf.put_u16(result);
        }

        if let Some(ref cause) = self.pdu_session_reactivation_result_error_cause {
            buf.put_u8(service_accept_iei::PDU_SESSION_REACTIVATION_RESULT_ERROR_CAUSE);
            buf.put_u16(cause.len() as u16);
            buf.put_slice(cause);
        }

        if let Some(ref eap) = self.eap_message {
            buf.put_u8(service_accept_iei::EAP_MESSAGE);
            buf.put_u16(eap.len() as u16);
            buf.put_slice(eap);
        }
    }

    /// Get the message type
    pub fn message_type() -> MmMessageType {
        MmMessageType::ServiceAccept
    }
}

// ============================================================================
// Service Reject Message (3GPP TS 24.501 Section 8.2.17)
// ============================================================================

/// Service Reject message (network to UE)
///
/// This message is sent by the network to the UE to indicate that
/// the requested service has been rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceReject {
    /// 5GMM cause (mandatory, Type 3)
    pub mm_cause: Ie5gMmCause,
    /// PDU session status (optional, Type 4, IEI 0x50)
    pub pdu_session_status: Option<u16>,
    /// T3346 value (optional, Type 4, IEI 0x5F)
    pub t3346_value: Option<u8>,
    /// EAP message (optional, Type 6, IEI 0x78)
    pub eap_message: Option<Vec<u8>>,
}

impl Default for ServiceReject {
    fn default() -> Self {
        Self {
            mm_cause: Ie5gMmCause::new(MmCause::ProtocolErrorUnspecified),
            pdu_session_status: None,
            t3346_value: None,
            eap_message: None,
        }
    }
}

impl ServiceReject {
    /// Create a new Service Reject with mandatory cause
    pub fn new(mm_cause: MmCause) -> Self {
        Self {
            mm_cause: Ie5gMmCause::new(mm_cause),
            ..Default::default()
        }
    }

    /// Decode from bytes (after header has been parsed)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, ServiceError> {
        // 5GMM cause (mandatory, Type 3 - 1 byte)
        let mm_cause = Ie5gMmCause::decode(buf)
            .map_err(|e| ServiceError::InvalidIeValue(e.to_string()))?;

        let mut msg = Self {
            mm_cause,
            ..Default::default()
        };

        // Parse optional IEs
        while buf.remaining() > 0 {
            let iei = buf.chunk()[0];

            match iei {
                service_reject_iei::PDU_SESSION_STATUS => {
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
                service_reject_iei::T3346_VALUE => {
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
                service_reject_iei::EAP_MESSAGE => {
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
        let header = PlainMmHeader::new(MmMessageType::ServiceReject);
        header.encode(buf);

        // 5GMM cause (mandatory)
        self.mm_cause.encode(buf);

        // Optional IEs
        if let Some(status) = self.pdu_session_status {
            buf.put_u8(service_reject_iei::PDU_SESSION_STATUS);
            buf.put_u8(2);
            buf.put_u16(status);
        }

        if let Some(value) = self.t3346_value {
            buf.put_u8(service_reject_iei::T3346_VALUE);
            buf.put_u8(1);
            buf.put_u8(value);
        }

        if let Some(ref eap) = self.eap_message {
            buf.put_u8(service_reject_iei::EAP_MESSAGE);
            buf.put_u16(eap.len() as u16);
            buf.put_slice(eap);
        }
    }

    /// Get the message type
    pub fn message_type() -> MmMessageType {
        MmMessageType::ServiceReject
    }
}


// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ies::ie1::ServiceType;
    use crate::messages::mm::registration::MobileIdentityType;
    use crate::security::SecurityContextType;

    #[test]
    fn test_service_request_encode_decode() {
        let tmsi = Ie5gsMobileIdentity::new(
            MobileIdentityType::Tmsi,
            vec![0xF4, 0x01, 0x02, 0x03, 0x04], // Type octet + 4 bytes TMSI
        );
        let msg = ServiceRequest::new(
            NasKeySetIdentifier::new(SecurityContextType::Native, 1),
            IeServiceType::new(ServiceType::Data),
            tmsi,
        );

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Verify header
        assert_eq!(buf[0], 0x7E); // EPD: Mobility Management
        assert_eq!(buf[1], 0x00); // Security header type: Not protected
        assert_eq!(buf[2], 0x4C); // Message type: Service Request

        // Decode and verify
        let decoded = ServiceRequest::decode(&mut &buf[3..]).unwrap();
        assert_eq!(decoded.ng_ksi.tsc, msg.ng_ksi.tsc);
        assert_eq!(decoded.ng_ksi.ksi, msg.ng_ksi.ksi);
        assert_eq!(decoded.service_type.service_type, ServiceType::Data);
        assert_eq!(decoded.tmsi.identity_type, MobileIdentityType::Tmsi);
    }

    #[test]
    fn test_service_request_with_optional_ies() {
        let tmsi = Ie5gsMobileIdentity::new(
            MobileIdentityType::Tmsi,
            vec![0xF4, 0x01, 0x02, 0x03, 0x04],
        );
        let mut msg = ServiceRequest::new(
            NasKeySetIdentifier::new(SecurityContextType::Native, 2),
            IeServiceType::new(ServiceType::Signalling),
            tmsi,
        );
        msg.uplink_data_status = Some(0x00FF);
        msg.pdu_session_status = Some(0xFF00);

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        let decoded = ServiceRequest::decode(&mut &buf[3..]).unwrap();
        assert_eq!(decoded.uplink_data_status, Some(0x00FF));
        assert_eq!(decoded.pdu_session_status, Some(0xFF00));
    }

    #[test]
    fn test_service_accept_encode_decode() {
        let msg = ServiceAccept::new();

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Verify header
        assert_eq!(buf[0], 0x7E); // EPD: Mobility Management
        assert_eq!(buf[1], 0x00); // Security header type: Not protected
        assert_eq!(buf[2], 0x4E); // Message type: Service Accept

        // Decode and verify (no optional IEs)
        let decoded = ServiceAccept::decode(&mut &buf[3..]).unwrap();
        assert_eq!(decoded.pdu_session_status, None);
    }

    #[test]
    fn test_service_accept_with_optional_ies() {
        let mut msg = ServiceAccept::new();
        msg.pdu_session_status = Some(0x0001);
        msg.pdu_session_reactivation_result = Some(0x0002);

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        let decoded = ServiceAccept::decode(&mut &buf[3..]).unwrap();
        assert_eq!(decoded.pdu_session_status, Some(0x0001));
        assert_eq!(decoded.pdu_session_reactivation_result, Some(0x0002));
    }

    #[test]
    fn test_service_reject_encode_decode() {
        let msg = ServiceReject::new(MmCause::Congestion);

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Verify header
        assert_eq!(buf[0], 0x7E); // EPD: Mobility Management
        assert_eq!(buf[1], 0x00); // Security header type: Not protected
        assert_eq!(buf[2], 0x4D); // Message type: Service Reject
        assert_eq!(buf[3], 22);   // 5GMM Cause: Congestion

        // Decode and verify
        let decoded = ServiceReject::decode(&mut &buf[3..]).unwrap();
        assert_eq!(decoded.mm_cause.value, MmCause::Congestion);
    }

    #[test]
    fn test_service_reject_with_optional_ies() {
        let mut msg = ServiceReject::new(MmCause::FiveGsServicesNotAllowed);
        msg.pdu_session_status = Some(0x0003);
        msg.t3346_value = Some(0x1E); // 30 seconds

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        let decoded = ServiceReject::decode(&mut &buf[3..]).unwrap();
        assert_eq!(decoded.mm_cause.value, MmCause::FiveGsServicesNotAllowed);
        assert_eq!(decoded.pdu_session_status, Some(0x0003));
        assert_eq!(decoded.t3346_value, Some(0x1E));
    }

    #[test]
    fn test_service_request_buffer_too_short() {
        let buf: &[u8] = &[];
        let result = ServiceRequest::decode(&mut &buf[..]);
        assert!(matches!(result, Err(ServiceError::BufferTooShort { .. })));
    }

    #[test]
    fn test_service_reject_buffer_too_short() {
        let buf: &[u8] = &[];
        let result = ServiceReject::decode(&mut &buf[..]);
        // When buffer is empty, Ie5gMmCause::decode returns RegistrationError::BufferTooShort
        // which gets converted to ServiceError::InvalidIeValue
        assert!(matches!(result, Err(ServiceError::InvalidIeValue(_))));
    }

    #[test]
    fn test_service_request_all_service_types() {
        let service_types = [
            ServiceType::Signalling,
            ServiceType::Data,
            ServiceType::MobileTerminatedServices,
            ServiceType::EmergencyServices,
            ServiceType::EmergencyServicesFallback,
            ServiceType::HighPriorityAccess,
            ServiceType::ElevatedSignalling,
        ];

        for st in service_types {
            let tmsi = Ie5gsMobileIdentity::new(
                MobileIdentityType::Tmsi,
                vec![0xF4, 0x01, 0x02, 0x03, 0x04],
            );
            let msg = ServiceRequest::new(
                NasKeySetIdentifier::new(SecurityContextType::Native, 0),
                IeServiceType::new(st),
                tmsi,
            );

            let mut buf = Vec::new();
            msg.encode(&mut buf);

            let decoded = ServiceRequest::decode(&mut &buf[3..]).unwrap();
            assert_eq!(decoded.service_type.service_type, st);
        }
    }

    #[test]
    fn test_message_types() {
        assert_eq!(ServiceRequest::message_type(), MmMessageType::ServiceRequest);
        assert_eq!(ServiceAccept::message_type(), MmMessageType::ServiceAccept);
        assert_eq!(ServiceReject::message_type(), MmMessageType::ServiceReject);
    }
}
