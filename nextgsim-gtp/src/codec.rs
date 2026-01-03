//! GTP-U header encoding/decoding
//!
//! Implements GTP-U (GPRS Tunneling Protocol - User Plane) header encoding and decoding
//! according to 3GPP TS 29.281.

use bytes::{Buf, BufMut, Bytes, BytesMut};
use thiserror::Error;

/// GTP-U protocol version (always 1)
pub const GTP_VERSION: u8 = 1;

/// GTP-U protocol type (1 for GTP-U, 0 for GTP')
pub const GTP_PROTOCOL_TYPE: u8 = 1;

/// GTP-U Message Types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum GtpMessageType {
    /// Echo Request
    EchoRequest = 1,
    /// Echo Response
    EchoResponse = 2,
    /// Error Indication
    ErrorIndication = 26,
    /// Supported Extension Headers Notification
    SupportedExtHeadersNotification = 31,
    /// End Marker
    EndMarker = 254,
    /// G-PDU (user data)
    GPdu = 255,
}

impl GtpMessageType {
    /// Convert from u8 to GtpMessageType
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::EchoRequest),
            2 => Some(Self::EchoResponse),
            26 => Some(Self::ErrorIndication),
            31 => Some(Self::SupportedExtHeadersNotification),
            254 => Some(Self::EndMarker),
            255 => Some(Self::GPdu),
            _ => None,
        }
    }
}


/// Extension Header Types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ExtHeaderType {
    /// No more extension headers
    NoMore = 0x00,
    /// UDP Port extension header
    UdpPort = 0x40,
    /// Long PDCP PDU Number extension header
    LongPdcpPduNumber = 0x82,
    /// NR RAN Container extension header
    NrRanContainer = 0x84,
    /// PDU Session Container extension header
    PduSessionContainer = 0x85,
    /// PDCP PDU Number extension header
    PdcpPduNumber = 0xC0,
}

impl ExtHeaderType {
    /// Convert from u8 to ExtHeaderType
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0x00 => Some(Self::NoMore),
            0x40 => Some(Self::UdpPort),
            0x82 => Some(Self::LongPdcpPduNumber),
            0x84 => Some(Self::NrRanContainer),
            0x85 => Some(Self::PduSessionContainer),
            0xC0 => Some(Self::PdcpPduNumber),
            _ => None,
        }
    }
}

/// GTP-U Extension Header
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GtpExtHeader {
    /// UDP Port extension header
    UdpPort {
        /// UDP port number
        port: u16,
    },
    /// PDCP PDU Number extension header (16-bit)
    PdcpPduNumber {
        /// PDCP PDU number (16-bit)
        pdu_number: u16,
    },
    /// Long PDCP PDU Number extension header (18-bit)
    LongPdcpPduNumber {
        /// PDCP PDU number (18-bit, stored in lower 18 bits)
        pdu_number: u32,
    },
    /// PDU Session Container extension header
    PduSessionContainer {
        /// PDU session information data
        data: Bytes,
    },
    /// NR RAN Container extension header
    NrRanContainer {
        /// NR RAN container data
        data: Bytes,
    },
}


/// GTP-U codec errors
#[derive(Debug, Error)]
pub enum GtpError {
    /// Buffer too short for header
    #[error("buffer too short: need {needed} bytes, have {available}")]
    BufferTooShort {
        /// Number of bytes needed
        needed: usize,
        /// Number of bytes available
        available: usize,
    },
    /// Invalid GTP version
    #[error("invalid GTP version: {0}, expected 1")]
    InvalidVersion(u8),
    /// Invalid protocol type
    #[error("invalid protocol type: {0}, expected 1 for GTP-U")]
    InvalidProtocolType(u8),
    /// Invalid message type
    #[error("invalid message type: {0}")]
    InvalidMessageType(u8),
    /// Invalid extension header type
    #[error("invalid extension header type: {0:#x}")]
    InvalidExtHeaderType(u8),
    /// Invalid extension header length
    #[error("invalid extension header length: {0}")]
    InvalidExtHeaderLength(u8),
    /// Payload length mismatch
    #[error("payload length mismatch: header says {header_len}, actual {actual_len}")]
    PayloadLengthMismatch {
        /// Length specified in header
        header_len: usize,
        /// Actual payload length
        actual_len: usize,
    },
}

/// GTP-U Header
///
/// The GTP-U header is at least 8 bytes:
/// - Flags (1 byte): version, PT, E, S, PN
/// - Message Type (1 byte)
/// - Length (2 bytes)
/// - TEID (4 bytes)
///
/// Optional fields (4 bytes if any flag E/S/PN is set):
/// - Sequence Number (2 bytes)
/// - N-PDU Number (1 byte)
/// - Next Extension Header Type (1 byte)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GtpHeader {
    /// Message type
    pub message_type: GtpMessageType,
    /// Tunnel Endpoint Identifier
    pub teid: u32,
    /// Sequence number (optional)
    pub sequence_number: Option<u16>,
    /// N-PDU number (optional)
    pub n_pdu_number: Option<u8>,
    /// Extension headers (optional)
    pub extension_headers: Vec<GtpExtHeader>,
    /// Payload data
    pub payload: Bytes,
}


impl GtpHeader {
    /// Minimum GTP-U header size (without optional fields)
    pub const MIN_HEADER_SIZE: usize = 8;

    /// Create a new GTP-U header with minimal fields
    pub fn new(message_type: GtpMessageType, teid: u32, payload: Bytes) -> Self {
        Self {
            message_type,
            teid,
            sequence_number: None,
            n_pdu_number: None,
            extension_headers: Vec::new(),
            payload,
        }
    }

    /// Create a G-PDU message (user data)
    pub fn g_pdu(teid: u32, payload: Bytes) -> Self {
        Self::new(GtpMessageType::GPdu, teid, payload)
    }

    /// Create an Echo Request message
    pub fn echo_request(teid: u32) -> Self {
        Self::new(GtpMessageType::EchoRequest, teid, Bytes::new())
    }

    /// Create an Echo Response message
    pub fn echo_response(teid: u32) -> Self {
        Self::new(GtpMessageType::EchoResponse, teid, Bytes::new())
    }

    /// Set sequence number
    pub fn with_sequence_number(mut self, seq: u16) -> Self {
        self.sequence_number = Some(seq);
        self
    }

    /// Set N-PDU number
    pub fn with_n_pdu_number(mut self, n_pdu: u8) -> Self {
        self.n_pdu_number = Some(n_pdu);
        self
    }

    /// Add an extension header
    pub fn with_extension_header(mut self, ext: GtpExtHeader) -> Self {
        self.extension_headers.push(ext);
        self
    }


    /// Check if optional fields are present
    fn has_optional_fields(&self) -> bool {
        self.sequence_number.is_some()
            || self.n_pdu_number.is_some()
            || !self.extension_headers.is_empty()
    }

    /// Calculate the encoded size of this header
    pub fn encoded_size(&self) -> usize {
        let mut size = Self::MIN_HEADER_SIZE;

        if self.has_optional_fields() {
            size += 4; // seq (2) + n_pdu (1) + next_ext_type (1)

            for ext in &self.extension_headers {
                size += ext.encoded_size();
            }
        }

        size + self.payload.len()
    }

    /// Encode the GTP-U header to bytes
    pub fn encode(&self) -> BytesMut {
        let mut buf = BytesMut::with_capacity(self.encoded_size());
        self.encode_to(&mut buf);
        buf
    }

    /// Encode the GTP-U header to an existing buffer
    pub fn encode_to(&self, buf: &mut BytesMut) {
        let has_ext = !self.extension_headers.is_empty();
        let has_seq = self.sequence_number.is_some();
        let has_n_pdu = self.n_pdu_number.is_some();
        let has_optional = has_ext || has_seq || has_n_pdu;

        // Build flags byte: version (3 bits) | PT (1 bit) | reserved (1 bit) | E | S | PN
        let flags: u8 = (GTP_VERSION << 5)
            | (GTP_PROTOCOL_TYPE << 4)
            | (if has_ext { 0x04 } else { 0 })
            | (if has_seq { 0x02 } else { 0 })
            | (if has_n_pdu { 0x01 } else { 0 });

        buf.put_u8(flags);
        buf.put_u8(self.message_type as u8);

        // Calculate length (everything after TEID)
        let length = self.calculate_length();
        buf.put_u16(length as u16);
        buf.put_u32(self.teid);

        if has_optional {
            buf.put_u16(self.sequence_number.unwrap_or(0));
            buf.put_u8(self.n_pdu_number.unwrap_or(0));

            // Encode extension headers
            for ext in &self.extension_headers {
                ext.encode_to(buf);
            }

            // No more extension headers
            buf.put_u8(ExtHeaderType::NoMore as u8);
        }

        buf.put_slice(&self.payload);
    }


    /// Calculate the length field value (bytes after TEID)
    fn calculate_length(&self) -> usize {
        let mut length = 0;

        if self.has_optional_fields() {
            length += 4; // seq (2) + n_pdu (1) + next_ext_type (1)

            for ext in &self.extension_headers {
                length += ext.encoded_size();
            }
        }

        length + self.payload.len()
    }

    /// Decode a GTP-U header from bytes
    pub fn decode(data: &[u8]) -> Result<Self, GtpError> {
        if data.len() < Self::MIN_HEADER_SIZE {
            return Err(GtpError::BufferTooShort {
                needed: Self::MIN_HEADER_SIZE,
                available: data.len(),
            });
        }

        let mut buf = data;

        // Parse flags
        let flags = buf.get_u8();
        let version = (flags >> 5) & 0x07;
        let protocol_type = (flags >> 4) & 0x01;
        let has_ext = (flags & 0x04) != 0;
        let has_seq = (flags & 0x02) != 0;
        let has_n_pdu = (flags & 0x01) != 0;

        if version != GTP_VERSION {
            return Err(GtpError::InvalidVersion(version));
        }

        if protocol_type != GTP_PROTOCOL_TYPE {
            return Err(GtpError::InvalidProtocolType(protocol_type));
        }

        // Parse message type
        let msg_type_raw = buf.get_u8();
        let message_type = GtpMessageType::from_u8(msg_type_raw)
            .ok_or(GtpError::InvalidMessageType(msg_type_raw))?;

        // Parse length and TEID
        let length = buf.get_u16() as usize;
        let teid = buf.get_u32();

        // Validate we have enough data
        let total_needed = Self::MIN_HEADER_SIZE + length;
        if data.len() < total_needed {
            return Err(GtpError::BufferTooShort {
                needed: total_needed,
                available: data.len(),
            });
        }

        let mut sequence_number = None;
        let mut n_pdu_number = None;
        let mut extension_headers = Vec::new();
        let mut header_bytes_after_teid = 0;

        // Parse optional fields if any flag is set
        if has_ext || has_seq || has_n_pdu {
            if buf.remaining() < 4 {
                return Err(GtpError::BufferTooShort {
                    needed: Self::MIN_HEADER_SIZE + 4,
                    available: data.len(),
                });
            }

            let seq = buf.get_u16();
            let n_pdu = buf.get_u8();
            let mut next_ext_type = buf.get_u8();
            header_bytes_after_teid += 4;

            if has_seq {
                sequence_number = Some(seq);
            }
            if has_n_pdu {
                n_pdu_number = Some(n_pdu);
            }

            // Parse extension headers
            while next_ext_type != ExtHeaderType::NoMore as u8 {
                let (ext, bytes_consumed, next_type) = GtpExtHeader::decode(next_ext_type, buf)?;
                extension_headers.push(ext);
                buf.advance(bytes_consumed);
                header_bytes_after_teid += bytes_consumed;
                next_ext_type = next_type;
            }
        }

        // Remaining bytes are payload
        let payload_len = length - header_bytes_after_teid;
        let payload = Bytes::copy_from_slice(&buf[..payload_len]);

        Ok(Self {
            message_type,
            teid,
            sequence_number,
            n_pdu_number,
            extension_headers,
            payload,
        })
    }
}


impl GtpExtHeader {
    /// Calculate the encoded size of this extension header
    pub fn encoded_size(&self) -> usize {
        match self {
            Self::UdpPort { .. } => 4,           // type(1) + len(1) + port(2)
            Self::PdcpPduNumber { .. } => 4,     // type(1) + len(1) + pdu_num(2)
            Self::LongPdcpPduNumber { .. } => 8, // type(1) + len(1) + pdu_num(3) + padding(3)
            Self::PduSessionContainer { data } => {
                // type(1) + len(1) + data + padding to 4-byte boundary
                let content_len = data.len() + 2;
                ((content_len + 3) / 4) * 4
            }
            Self::NrRanContainer { data } => {
                let content_len = data.len() + 2;
                ((content_len + 3) / 4) * 4
            }
        }
    }

    /// Encode the extension header to a buffer
    pub fn encode_to(&self, buf: &mut BytesMut) {
        match self {
            Self::UdpPort { port } => {
                buf.put_u8(ExtHeaderType::UdpPort as u8);
                buf.put_u8(1); // length in 4-byte units
                buf.put_u16(*port);
            }
            Self::PdcpPduNumber { pdu_number } => {
                buf.put_u8(ExtHeaderType::PdcpPduNumber as u8);
                buf.put_u8(1);
                buf.put_u16(*pdu_number);
            }
            Self::LongPdcpPduNumber { pdu_number } => {
                buf.put_u8(ExtHeaderType::LongPdcpPduNumber as u8);
                buf.put_u8(2); // length in 4-byte units
                // 18-bit PDU number in 3 bytes
                buf.put_u8(((pdu_number >> 16) & 0x03) as u8);
                buf.put_u8(((pdu_number >> 8) & 0xFF) as u8);
                buf.put_u8((pdu_number & 0xFF) as u8);
                buf.put_u8(0); // padding
                buf.put_u8(0); // padding
                buf.put_u8(0); // padding
            }
            Self::PduSessionContainer { data } => {
                buf.put_u8(ExtHeaderType::PduSessionContainer as u8);
                let len_units = (data.len() + 2 + 3) / 4;
                buf.put_u8(len_units as u8);
                buf.put_slice(data);
                // Add padding to 4-byte boundary
                let padding = (4 - ((data.len() + 2) % 4)) % 4;
                for _ in 0..padding {
                    buf.put_u8(0);
                }
            }
            Self::NrRanContainer { data } => {
                buf.put_u8(ExtHeaderType::NrRanContainer as u8);
                let len_units = (data.len() + 2 + 3) / 4;
                buf.put_u8(len_units as u8);
                buf.put_slice(data);
                let padding = (4 - ((data.len() + 2) % 4)) % 4;
                for _ in 0..padding {
                    buf.put_u8(0);
                }
            }
        }
    }


    /// Decode an extension header from bytes
    /// Returns (header, bytes_consumed, next_ext_type)
    fn decode(ext_type: u8, data: &[u8]) -> Result<(Self, usize, u8), GtpError> {
        if data.is_empty() {
            return Err(GtpError::BufferTooShort {
                needed: 1,
                available: 0,
            });
        }

        let len_units = data[0] as usize;
        if len_units == 0 {
            return Err(GtpError::InvalidExtHeaderLength(0));
        }

        let total_len = len_units * 4;
        if data.len() < total_len {
            return Err(GtpError::BufferTooShort {
                needed: total_len,
                available: data.len(),
            });
        }

        let next_ext_type = data[total_len - 1];

        let ext = match ext_type {
            0x40 => {
                // UDP Port
                if len_units != 1 {
                    return Err(GtpError::InvalidExtHeaderLength(len_units as u8));
                }
                let port = u16::from_be_bytes([data[1], data[2]]);
                Self::UdpPort { port }
            }
            0xC0 => {
                // PDCP PDU Number
                if len_units != 1 {
                    return Err(GtpError::InvalidExtHeaderLength(len_units as u8));
                }
                let pdu_number = u16::from_be_bytes([data[1], data[2]]);
                Self::PdcpPduNumber { pdu_number }
            }
            0x82 => {
                // Long PDCP PDU Number
                if len_units != 2 {
                    return Err(GtpError::InvalidExtHeaderLength(len_units as u8));
                }
                let pdu_number = ((data[1] as u32 & 0x03) << 16)
                    | ((data[2] as u32) << 8)
                    | (data[3] as u32);
                Self::LongPdcpPduNumber { pdu_number }
            }
            0x85 => {
                // PDU Session Container
                let content_len = total_len - 2; // minus len byte and next_ext_type byte
                let content = Bytes::copy_from_slice(&data[1..1 + content_len]);
                Self::PduSessionContainer { data: content }
            }
            0x84 => {
                // NR RAN Container
                let content_len = total_len - 2;
                let content = Bytes::copy_from_slice(&data[1..1 + content_len]);
                Self::NrRanContainer { data: content }
            }
            _ => return Err(GtpError::InvalidExtHeaderType(ext_type)),
        };

        Ok((ext, total_len, next_ext_type))
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_minimal_header() {
        let header = GtpHeader::g_pdu(0x12345678, Bytes::from_static(b"hello"));

        let encoded = header.encode();
        let decoded = GtpHeader::decode(&encoded).unwrap();

        assert_eq!(decoded.message_type, GtpMessageType::GPdu);
        assert_eq!(decoded.teid, 0x12345678);
        assert_eq!(decoded.sequence_number, None);
        assert_eq!(decoded.n_pdu_number, None);
        assert!(decoded.extension_headers.is_empty());
        assert_eq!(decoded.payload, Bytes::from_static(b"hello"));
    }

    #[test]
    fn test_encode_decode_with_sequence_number() {
        let header = GtpHeader::g_pdu(0xABCDEF01, Bytes::from_static(b"test"))
            .with_sequence_number(0x1234);

        let encoded = header.encode();
        let decoded = GtpHeader::decode(&encoded).unwrap();

        assert_eq!(decoded.sequence_number, Some(0x1234));
        assert_eq!(decoded.n_pdu_number, None);
        assert_eq!(decoded.payload, Bytes::from_static(b"test"));
    }

    #[test]
    fn test_encode_decode_with_n_pdu_number() {
        let header = GtpHeader::g_pdu(0x11223344, Bytes::new())
            .with_n_pdu_number(0x42);

        let encoded = header.encode();
        let decoded = GtpHeader::decode(&encoded).unwrap();

        assert_eq!(decoded.n_pdu_number, Some(0x42));
    }

    #[test]
    fn test_encode_decode_with_all_optional_fields() {
        let header = GtpHeader::g_pdu(0xDEADBEEF, Bytes::from_static(b"payload"))
            .with_sequence_number(0x5678)
            .with_n_pdu_number(0xAB);

        let encoded = header.encode();
        let decoded = GtpHeader::decode(&encoded).unwrap();

        assert_eq!(decoded.teid, 0xDEADBEEF);
        assert_eq!(decoded.sequence_number, Some(0x5678));
        assert_eq!(decoded.n_pdu_number, Some(0xAB));
        assert_eq!(decoded.payload, Bytes::from_static(b"payload"));
    }


    #[test]
    fn test_encode_decode_echo_request() {
        let header = GtpHeader::echo_request(0x00000001);

        let encoded = header.encode();
        let decoded = GtpHeader::decode(&encoded).unwrap();

        assert_eq!(decoded.message_type, GtpMessageType::EchoRequest);
        assert_eq!(decoded.teid, 0x00000001);
        assert!(decoded.payload.is_empty());
    }

    #[test]
    fn test_encode_decode_echo_response() {
        let header = GtpHeader::echo_response(0x00000002);

        let encoded = header.encode();
        let decoded = GtpHeader::decode(&encoded).unwrap();

        assert_eq!(decoded.message_type, GtpMessageType::EchoResponse);
        assert_eq!(decoded.teid, 0x00000002);
    }

    #[test]
    fn test_encode_decode_with_udp_port_ext_header() {
        let header = GtpHeader::g_pdu(0x12345678, Bytes::from_static(b"data"))
            .with_extension_header(GtpExtHeader::UdpPort { port: 2152 });

        let encoded = header.encode();
        let decoded = GtpHeader::decode(&encoded).unwrap();

        assert_eq!(decoded.extension_headers.len(), 1);
        match &decoded.extension_headers[0] {
            GtpExtHeader::UdpPort { port } => assert_eq!(*port, 2152),
            _ => panic!("Expected UdpPort extension header"),
        }
    }

    #[test]
    fn test_encode_decode_with_pdcp_pdu_number_ext_header() {
        let header = GtpHeader::g_pdu(0x12345678, Bytes::new())
            .with_extension_header(GtpExtHeader::PdcpPduNumber { pdu_number: 0x1234 });

        let encoded = header.encode();
        let decoded = GtpHeader::decode(&encoded).unwrap();

        assert_eq!(decoded.extension_headers.len(), 1);
        match &decoded.extension_headers[0] {
            GtpExtHeader::PdcpPduNumber { pdu_number } => assert_eq!(*pdu_number, 0x1234),
            _ => panic!("Expected PdcpPduNumber extension header"),
        }
    }

    #[test]
    fn test_encode_decode_with_long_pdcp_pdu_number_ext_header() {
        let header = GtpHeader::g_pdu(0x12345678, Bytes::new())
            .with_extension_header(GtpExtHeader::LongPdcpPduNumber { pdu_number: 0x3FFFF });

        let encoded = header.encode();
        let decoded = GtpHeader::decode(&encoded).unwrap();

        assert_eq!(decoded.extension_headers.len(), 1);
        match &decoded.extension_headers[0] {
            GtpExtHeader::LongPdcpPduNumber { pdu_number } => assert_eq!(*pdu_number, 0x3FFFF),
            _ => panic!("Expected LongPdcpPduNumber extension header"),
        }
    }


    #[test]
    fn test_encode_decode_with_pdu_session_container() {
        let container_data = Bytes::from_static(&[0x00, 0x01, 0x02, 0x03]);
        let header = GtpHeader::g_pdu(0x12345678, Bytes::new())
            .with_extension_header(GtpExtHeader::PduSessionContainer { data: container_data.clone() });

        let encoded = header.encode();
        let decoded = GtpHeader::decode(&encoded).unwrap();

        assert_eq!(decoded.extension_headers.len(), 1);
        match &decoded.extension_headers[0] {
            GtpExtHeader::PduSessionContainer { data } => {
                // Data includes padding, so check prefix
                assert!(data.starts_with(&container_data));
            }
            _ => panic!("Expected PduSessionContainer extension header"),
        }
    }

    #[test]
    fn test_encode_decode_with_multiple_ext_headers() {
        let header = GtpHeader::g_pdu(0x12345678, Bytes::from_static(b"test"))
            .with_sequence_number(0x0001)
            .with_extension_header(GtpExtHeader::UdpPort { port: 2152 })
            .with_extension_header(GtpExtHeader::PdcpPduNumber { pdu_number: 0x5678 });

        let encoded = header.encode();
        let decoded = GtpHeader::decode(&encoded).unwrap();

        assert_eq!(decoded.sequence_number, Some(0x0001));
        assert_eq!(decoded.extension_headers.len(), 2);
    }

    #[test]
    fn test_decode_invalid_version() {
        // Create a buffer with invalid version (0)
        let data = [0x00, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let result = GtpHeader::decode(&data);
        assert!(matches!(result, Err(GtpError::InvalidVersion(0))));
    }

    #[test]
    fn test_decode_invalid_protocol_type() {
        // Create a buffer with invalid protocol type (0 = GTP')
        let data = [0x20, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let result = GtpHeader::decode(&data);
        assert!(matches!(result, Err(GtpError::InvalidProtocolType(0))));
    }

    #[test]
    fn test_decode_buffer_too_short() {
        let data = [0x30, 0xFF, 0x00];
        let result = GtpHeader::decode(&data);
        assert!(matches!(result, Err(GtpError::BufferTooShort { .. })));
    }

    #[test]
    fn test_header_size_calculation() {
        // Minimal header
        let header = GtpHeader::g_pdu(0, Bytes::new());
        assert_eq!(header.encoded_size(), 8);

        // With sequence number
        let header = GtpHeader::g_pdu(0, Bytes::new()).with_sequence_number(0);
        assert_eq!(header.encoded_size(), 12);

        // With payload
        let header = GtpHeader::g_pdu(0, Bytes::from_static(b"hello"));
        assert_eq!(header.encoded_size(), 13); // 8 + 5
    }

    #[test]
    fn test_message_type_conversion() {
        assert_eq!(GtpMessageType::from_u8(1), Some(GtpMessageType::EchoRequest));
        assert_eq!(GtpMessageType::from_u8(2), Some(GtpMessageType::EchoResponse));
        assert_eq!(GtpMessageType::from_u8(26), Some(GtpMessageType::ErrorIndication));
        assert_eq!(GtpMessageType::from_u8(31), Some(GtpMessageType::SupportedExtHeadersNotification));
        assert_eq!(GtpMessageType::from_u8(254), Some(GtpMessageType::EndMarker));
        assert_eq!(GtpMessageType::from_u8(255), Some(GtpMessageType::GPdu));
        assert_eq!(GtpMessageType::from_u8(100), None);
    }

    #[test]
    fn test_large_payload() {
        let payload = Bytes::from(vec![0xAB; 1500]); // MTU-sized payload
        let header = GtpHeader::g_pdu(0x12345678, payload.clone());

        let encoded = header.encode();
        let decoded = GtpHeader::decode(&encoded).unwrap();

        assert_eq!(decoded.payload.len(), 1500);
        assert_eq!(decoded.payload, payload);
    }
}
