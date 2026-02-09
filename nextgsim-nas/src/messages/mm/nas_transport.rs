//! NAS Transport Messages (3GPP TS 24.501 Section 8.2.10-8.2.11)
//!
//! This module implements the NAS Transport messages:
//! - UL NAS Transport (UE to network, Section 8.2.10)
//! - DL NAS Transport (network to UE, Section 8.2.11)

use bytes::{Buf, BufMut};
use thiserror::Error;

use crate::enums::MmMessageType;
use crate::header::PlainMmHeader;
use crate::ies::ie1::{PayloadContainerType, RequestType};

/// Error type for NAS Transport message encoding/decoding
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum NasTransportError {
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
// IEI Constants for NAS Transport Messages
// ============================================================================

/// IEI values for DL NAS Transport optional IEs
#[allow(dead_code)]
mod dl_nas_transport_iei {
    /// PDU session ID
    pub const PDU_SESSION_ID: u8 = 0x12;
    /// Additional information
    pub const ADDITIONAL_INFO: u8 = 0x24;
    /// 5GMM cause
    pub const MM_CAUSE: u8 = 0x58;
    /// Back-off timer value
    pub const BACK_OFF_TIMER_VALUE: u8 = 0x37;
}

/// IEI values for UL NAS Transport optional IEs
#[allow(dead_code)]
mod ul_nas_transport_iei {
    /// PDU session ID
    pub const PDU_SESSION_ID: u8 = 0x12;
    /// Old PDU session ID
    pub const OLD_PDU_SESSION_ID: u8 = 0x59;
    /// Request type (Type 1, IEI high nibble 0x8)
    pub const REQUEST_TYPE_HIGH_NIBBLE: u8 = 0x8;
    /// S-NSSAI
    pub const S_NSSAI: u8 = 0x22;
    /// DNN
    pub const DNN: u8 = 0x25;
}


// ============================================================================
// DL NAS Transport (3GPP TS 24.501 Section 8.2.11)
// ============================================================================

/// DL NAS Transport message (network to UE)
///
/// This message is used by the network to transport NAS messages
/// in the downlink direction.
///
/// 3GPP TS 24.501 Section 8.2.11
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DlNasTransport {
    /// Payload container type (mandatory, 4 bits)
    pub payload_container_type: PayloadContainerType,
    /// Payload container (mandatory, TLV-E)
    pub payload_container: Vec<u8>,
    /// PDU session ID (optional, Type 3, IEI 0x12)
    pub pdu_session_id: Option<u8>,
    /// Additional information (optional, Type 4, IEI 0x24)
    pub additional_info: Option<Vec<u8>>,
    /// 5GMM cause (optional, Type 3, IEI 0x58)
    pub mm_cause: Option<u8>,
    /// Back-off timer value (optional, Type 4, IEI 0x37)
    pub back_off_timer_value: Option<u8>,
}

impl Default for DlNasTransport {
    fn default() -> Self {
        Self {
            payload_container_type: PayloadContainerType::N1SmInformation,
            payload_container: Vec::new(),
            pdu_session_id: None,
            additional_info: None,
            mm_cause: None,
            back_off_timer_value: None,
        }
    }
}

impl DlNasTransport {
    /// Create a new DL NAS Transport message
    pub fn new(payload_container_type: PayloadContainerType, payload_container: Vec<u8>) -> Self {
        Self {
            payload_container_type,
            payload_container,
            ..Default::default()
        }
    }

    /// Decode from bytes (after header has been parsed)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, NasTransportError> {
        // Payload container type (mandatory, 4 bits - lower nibble of next byte)
        if buf.remaining() < 1 {
            return Err(NasTransportError::BufferTooShort {
                expected: 1,
                actual: buf.remaining(),
            });
        }
        let pct_byte = buf.get_u8();
        let payload_container_type = PayloadContainerType::try_from(pct_byte & 0x0F)
            .map_err(|_| NasTransportError::InvalidIeValue(
                format!("Invalid payload container type: 0x{pct_byte:02X}"),
            ))?;

        // Payload container (mandatory, TLV-E: IEI(1) + Length(2) + Value)
        // In the mandatory position there is no IEI, just the length-value
        if buf.remaining() < 2 {
            return Err(NasTransportError::BufferTooShort {
                expected: 2,
                actual: buf.remaining(),
            });
        }
        let pc_len = buf.get_u16() as usize;
        if buf.remaining() < pc_len {
            return Err(NasTransportError::BufferTooShort {
                expected: pc_len,
                actual: buf.remaining(),
            });
        }
        let mut payload_container = vec![0u8; pc_len];
        buf.copy_to_slice(&mut payload_container);

        let mut msg = Self::new(payload_container_type, payload_container);

        // Parse optional IEs
        while buf.remaining() > 0 {
            let iei = buf.chunk()[0];

            match iei {
                dl_nas_transport_iei::PDU_SESSION_ID => {
                    buf.advance(1);
                    if buf.remaining() < 1 {
                        break;
                    }
                    msg.pdu_session_id = Some(buf.get_u8());
                }
                dl_nas_transport_iei::ADDITIONAL_INFO => {
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
                    msg.additional_info = Some(data);
                }
                dl_nas_transport_iei::MM_CAUSE => {
                    buf.advance(1);
                    if buf.remaining() < 1 {
                        break;
                    }
                    msg.mm_cause = Some(buf.get_u8());
                }
                dl_nas_transport_iei::BACK_OFF_TIMER_VALUE => {
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
        let header = PlainMmHeader::new(MmMessageType::DlNasTransport);
        header.encode(buf);

        // Payload container type (mandatory, 4 bits in lower nibble)
        let pct_val: u8 = self.payload_container_type.into();
        buf.put_u8(pct_val & 0x0F);

        // Payload container (mandatory, length-value with 2-byte length)
        buf.put_u16(self.payload_container.len() as u16);
        buf.put_slice(&self.payload_container);

        // Optional IEs
        if let Some(psi) = self.pdu_session_id {
            buf.put_u8(dl_nas_transport_iei::PDU_SESSION_ID);
            buf.put_u8(psi);
        }

        if let Some(ref info) = self.additional_info {
            buf.put_u8(dl_nas_transport_iei::ADDITIONAL_INFO);
            buf.put_u8(info.len() as u8);
            buf.put_slice(info);
        }

        if let Some(cause) = self.mm_cause {
            buf.put_u8(dl_nas_transport_iei::MM_CAUSE);
            buf.put_u8(cause);
        }

        if let Some(timer) = self.back_off_timer_value {
            buf.put_u8(dl_nas_transport_iei::BACK_OFF_TIMER_VALUE);
            buf.put_u8(1);
            buf.put_u8(timer);
        }
    }

    /// Get the message type
    pub fn message_type() -> MmMessageType {
        MmMessageType::DlNasTransport
    }
}


// ============================================================================
// UL NAS Transport (3GPP TS 24.501 Section 8.2.10)
// ============================================================================

/// UL NAS Transport message (UE to network)
///
/// This message is used by the UE to transport NAS messages
/// in the uplink direction.
///
/// 3GPP TS 24.501 Section 8.2.10
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UlNasTransport {
    /// Payload container type (mandatory, 4 bits)
    pub payload_container_type: PayloadContainerType,
    /// Payload container (mandatory, TLV-E)
    pub payload_container: Vec<u8>,
    /// PDU session ID (optional, Type 3, IEI 0x12)
    pub pdu_session_id: Option<u8>,
    /// Old PDU session ID (optional, Type 3, IEI 0x59)
    pub old_pdu_session_id: Option<u8>,
    /// Request type (optional, Type 1, IEI high nibble 0x8)
    pub request_type: Option<RequestType>,
    /// S-NSSAI (optional, Type 4, IEI 0x22)
    pub s_nssai: Option<Vec<u8>>,
    /// DNN (optional, Type 4, IEI 0x25)
    pub dnn: Option<Vec<u8>>,
}

impl Default for UlNasTransport {
    fn default() -> Self {
        Self {
            payload_container_type: PayloadContainerType::N1SmInformation,
            payload_container: Vec::new(),
            pdu_session_id: None,
            old_pdu_session_id: None,
            request_type: None,
            s_nssai: None,
            dnn: None,
        }
    }
}

impl UlNasTransport {
    /// Create a new UL NAS Transport message
    pub fn new(payload_container_type: PayloadContainerType, payload_container: Vec<u8>) -> Self {
        Self {
            payload_container_type,
            payload_container,
            ..Default::default()
        }
    }

    /// Decode from bytes (after header has been parsed)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, NasTransportError> {
        // Payload container type (mandatory, 4 bits)
        if buf.remaining() < 1 {
            return Err(NasTransportError::BufferTooShort {
                expected: 1,
                actual: buf.remaining(),
            });
        }
        let pct_byte = buf.get_u8();
        let payload_container_type = PayloadContainerType::try_from(pct_byte & 0x0F)
            .map_err(|_| NasTransportError::InvalidIeValue(
                format!("Invalid payload container type: 0x{pct_byte:02X}"),
            ))?;

        // Payload container (mandatory, length-value with 2-byte length)
        if buf.remaining() < 2 {
            return Err(NasTransportError::BufferTooShort {
                expected: 2,
                actual: buf.remaining(),
            });
        }
        let pc_len = buf.get_u16() as usize;
        if buf.remaining() < pc_len {
            return Err(NasTransportError::BufferTooShort {
                expected: pc_len,
                actual: buf.remaining(),
            });
        }
        let mut payload_container = vec![0u8; pc_len];
        buf.copy_to_slice(&mut payload_container);

        let mut msg = Self::new(payload_container_type, payload_container);

        // Parse optional IEs
        while buf.remaining() > 0 {
            let iei = buf.chunk()[0];

            // Check for Type 1 IEs (4-bit IEI in high nibble)
            let iei_high = (iei >> 4) & 0x0F;
            if iei_high == ul_nas_transport_iei::REQUEST_TYPE_HIGH_NIBBLE {
                buf.advance(1);
                let rt = RequestType::try_from(iei & 0x07)
                    .unwrap_or(RequestType::InitialRequest);
                msg.request_type = Some(rt);
                continue;
            }

            // Full octet IEI
            match iei {
                ul_nas_transport_iei::PDU_SESSION_ID => {
                    buf.advance(1);
                    if buf.remaining() < 1 {
                        break;
                    }
                    msg.pdu_session_id = Some(buf.get_u8());
                }
                ul_nas_transport_iei::OLD_PDU_SESSION_ID => {
                    buf.advance(1);
                    if buf.remaining() < 1 {
                        break;
                    }
                    msg.old_pdu_session_id = Some(buf.get_u8());
                }
                ul_nas_transport_iei::S_NSSAI => {
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
                ul_nas_transport_iei::DNN => {
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
                    msg.dnn = Some(data);
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
        let header = PlainMmHeader::new(MmMessageType::UlNasTransport);
        header.encode(buf);

        // Payload container type (mandatory, 4 bits in lower nibble)
        let pct_val: u8 = self.payload_container_type.into();
        buf.put_u8(pct_val & 0x0F);

        // Payload container (mandatory, length-value with 2-byte length)
        buf.put_u16(self.payload_container.len() as u16);
        buf.put_slice(&self.payload_container);

        // Optional IEs
        if let Some(psi) = self.pdu_session_id {
            buf.put_u8(ul_nas_transport_iei::PDU_SESSION_ID);
            buf.put_u8(psi);
        }

        if let Some(old_psi) = self.old_pdu_session_id {
            buf.put_u8(ul_nas_transport_iei::OLD_PDU_SESSION_ID);
            buf.put_u8(old_psi);
        }

        if let Some(rt) = self.request_type {
            let rt_val: u8 = rt.into();
            buf.put_u8((ul_nas_transport_iei::REQUEST_TYPE_HIGH_NIBBLE << 4) | (rt_val & 0x07));
        }

        if let Some(ref nssai) = self.s_nssai {
            buf.put_u8(ul_nas_transport_iei::S_NSSAI);
            buf.put_u8(nssai.len() as u8);
            buf.put_slice(nssai);
        }

        if let Some(ref dnn) = self.dnn {
            buf.put_u8(ul_nas_transport_iei::DNN);
            buf.put_u8(dnn.len() as u8);
            buf.put_slice(dnn);
        }
    }

    /// Get the message type
    pub fn message_type() -> MmMessageType {
        MmMessageType::UlNasTransport
    }
}


// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // DL NAS Transport Tests
    // ========================================================================

    #[test]
    fn test_dl_nas_transport_new() {
        let msg = DlNasTransport::new(
            PayloadContainerType::N1SmInformation,
            vec![0x01, 0x02],
        );
        assert_eq!(msg.payload_container_type, PayloadContainerType::N1SmInformation);
        assert_eq!(msg.payload_container, vec![0x01, 0x02]);
        assert!(msg.pdu_session_id.is_none());
    }

    #[test]
    fn test_dl_nas_transport_encode_minimal() {
        let msg = DlNasTransport::new(
            PayloadContainerType::N1SmInformation,
            vec![0xAA, 0xBB],
        );
        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Header (3) + PCT (1) + PC length (2) + PC (2) = 8 bytes
        assert_eq!(buf.len(), 8);
        assert_eq!(buf[0], 0x7E); // EPD
        assert_eq!(buf[1], 0x00); // Security header
        assert_eq!(buf[2], 0x68); // Message Type (DlNasTransport)
        assert_eq!(buf[3], 0x01); // Payload container type (N1 SM information)
        assert_eq!(buf[4], 0x00); // PC length high
        assert_eq!(buf[5], 0x02); // PC length low
        assert_eq!(buf[6], 0xAA); // Payload
        assert_eq!(buf[7], 0xBB); // Payload
    }

    #[test]
    fn test_dl_nas_transport_encode_decode() {
        let mut msg = DlNasTransport::new(
            PayloadContainerType::N1SmInformation,
            vec![0x01, 0x02, 0x03],
        );
        msg.pdu_session_id = Some(5);
        msg.mm_cause = Some(0x16); // Congestion

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Skip header (3 bytes) and decode
        let decoded = DlNasTransport::decode(&mut &buf[3..]).unwrap();
        assert_eq!(decoded.payload_container_type, PayloadContainerType::N1SmInformation);
        assert_eq!(decoded.payload_container, vec![0x01, 0x02, 0x03]);
        assert_eq!(decoded.pdu_session_id, Some(5));
        assert_eq!(decoded.mm_cause, Some(0x16));
    }

    #[test]
    fn test_dl_nas_transport_encode_decode_with_timer() {
        let mut msg = DlNasTransport::new(
            PayloadContainerType::Sms,
            vec![0x10],
        );
        msg.back_off_timer_value = Some(0x21);

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        let decoded = DlNasTransport::decode(&mut &buf[3..]).unwrap();
        assert_eq!(decoded.payload_container_type, PayloadContainerType::Sms);
        assert_eq!(decoded.back_off_timer_value, Some(0x21));
    }

    #[test]
    fn test_dl_nas_transport_message_type() {
        assert_eq!(DlNasTransport::message_type(), MmMessageType::DlNasTransport);
    }

    // ========================================================================
    // UL NAS Transport Tests
    // ========================================================================

    #[test]
    fn test_ul_nas_transport_new() {
        let msg = UlNasTransport::new(
            PayloadContainerType::N1SmInformation,
            vec![0x01, 0x02],
        );
        assert_eq!(msg.payload_container_type, PayloadContainerType::N1SmInformation);
        assert_eq!(msg.payload_container, vec![0x01, 0x02]);
        assert!(msg.pdu_session_id.is_none());
        assert!(msg.request_type.is_none());
    }

    #[test]
    fn test_ul_nas_transport_encode_minimal() {
        let msg = UlNasTransport::new(
            PayloadContainerType::N1SmInformation,
            vec![0xCC],
        );
        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Header (3) + PCT (1) + PC length (2) + PC (1) = 7 bytes
        assert_eq!(buf.len(), 7);
        assert_eq!(buf[0], 0x7E); // EPD
        assert_eq!(buf[1], 0x00); // Security header
        assert_eq!(buf[2], 0x67); // Message Type (UlNasTransport)
        assert_eq!(buf[3], 0x01); // Payload container type
    }

    #[test]
    fn test_ul_nas_transport_encode_decode() {
        let mut msg = UlNasTransport::new(
            PayloadContainerType::N1SmInformation,
            vec![0x01, 0x02],
        );
        msg.pdu_session_id = Some(5);
        msg.request_type = Some(RequestType::InitialRequest);
        msg.s_nssai = Some(vec![0x01]); // SST only
        msg.dnn = Some(vec![0x08, 0x69, 0x6E, 0x74, 0x65, 0x72, 0x6E, 0x65, 0x74]); // "internet"

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        let decoded = UlNasTransport::decode(&mut &buf[3..]).unwrap();
        assert_eq!(decoded.payload_container_type, PayloadContainerType::N1SmInformation);
        assert_eq!(decoded.payload_container, vec![0x01, 0x02]);
        assert_eq!(decoded.pdu_session_id, Some(5));
        assert_eq!(decoded.request_type, Some(RequestType::InitialRequest));
        assert_eq!(decoded.s_nssai, Some(vec![0x01]));
        assert_eq!(
            decoded.dnn,
            Some(vec![0x08, 0x69, 0x6E, 0x74, 0x65, 0x72, 0x6E, 0x65, 0x74])
        );
    }

    #[test]
    fn test_ul_nas_transport_encode_decode_with_old_pdu_session() {
        let mut msg = UlNasTransport::new(
            PayloadContainerType::N1SmInformation,
            vec![0xFF],
        );
        msg.pdu_session_id = Some(3);
        msg.old_pdu_session_id = Some(1);

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        let decoded = UlNasTransport::decode(&mut &buf[3..]).unwrap();
        assert_eq!(decoded.pdu_session_id, Some(3));
        assert_eq!(decoded.old_pdu_session_id, Some(1));
    }

    #[test]
    fn test_ul_nas_transport_message_type() {
        assert_eq!(UlNasTransport::message_type(), MmMessageType::UlNasTransport);
    }

    #[test]
    fn test_ul_nas_transport_decode_empty_payload() {
        let msg = UlNasTransport::new(
            PayloadContainerType::N1SmInformation,
            vec![],
        );
        let mut buf = Vec::new();
        msg.encode(&mut buf);

        let decoded = UlNasTransport::decode(&mut &buf[3..]).unwrap();
        assert_eq!(decoded.payload_container.len(), 0);
    }
}
