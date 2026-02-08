//! GTP-U header encoding/decoding
//!
//! Implements GTP-U (GPRS Tunneling Protocol - User Plane) header encoding and decoding
//! according to 3GPP TS 29.281.
//!
//! # Extension Header Chaining
//!
//! GTP-U supports chaining multiple extension headers. Each extension header contains
//! a `next_extension_header_type` field that points to the next header in the chain.
//! The chain is terminated when `next_extension_header_type` is 0x00 (`NoMore`).
//!
//! Supported extension header types:
//! - `UdpPort` (0x40) - UDP port number
//! - `LongPdcpPduNumber` (0x82) - 18-bit PDCP PDU number
//! - `NrRanContainer` (0x84) - NR RAN container data
//! - `PduSessionContainer` (0x85) - PDU Session Container (5GC)
//! - `PdcpPduNumber` (0xC0) - 16-bit PDCP PDU number
//! - `TsnMarker` (0xE1) - 6G Time-Sensitive Networking marker
//! - `InNetworkCompute` (0xE2) - 6G In-Network Computing marker
//!
//! # PDU Session Container
//!
//! The PDU Session Container (type 0x85) is a 5GC-specific extension header that
//! carries PDU session information including QoS Flow Identifier (QFI), PDU type
//! (DL/UL), and additional indicators. See [`PduSessionInfo`] for details.
//!
//! # 6G Extensions
//!
//! This module includes forward-looking 6G extensions:
//! - [`TsnMarker`] - Time-Sensitive Networking markers for deterministic networking
//! - [`InNetworkComputeMarker`] - Markers for in-network computing tasks

use bytes::{Buf, BufMut, Bytes, BytesMut};
use thiserror::Error;

/// GTP-U protocol version (always 1)
pub const GTP_VERSION: u8 = 1;

/// GTP-U protocol type (1 for GTP-U, 0 for GTP')
pub const GTP_PROTOCOL_TYPE: u8 = 1;

/// Custom extension header type for TSN markers (6G experimental range)
pub const EXT_HEADER_TYPE_TSN_MARKER: u8 = 0xE1;

/// Custom extension header type for in-network compute markers (6G experimental range)
pub const EXT_HEADER_TYPE_IN_NETWORK_COMPUTE: u8 = 0xE2;

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
    /// Convert from u8 to `GtpMessageType`
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
    /// 6G TSN (Time-Sensitive Networking) marker extension header
    TsnMarker = 0xE1,
    /// 6G In-Network Compute marker extension header
    InNetworkCompute = 0xE2,
}

impl ExtHeaderType {
    /// Convert from u8 to `ExtHeaderType`
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0x00 => Some(Self::NoMore),
            0x40 => Some(Self::UdpPort),
            0x82 => Some(Self::LongPdcpPduNumber),
            0x84 => Some(Self::NrRanContainer),
            0x85 => Some(Self::PduSessionContainer),
            0xC0 => Some(Self::PdcpPduNumber),
            0xE1 => Some(Self::TsnMarker),
            0xE2 => Some(Self::InNetworkCompute),
            _ => None,
        }
    }
}


// ---------------------------------------------------------------------------
// PDU Session Container IE (A5.2 - 5GC-specific GTP-U extension header)
// ---------------------------------------------------------------------------

/// PDU Session type (DL or UL) as defined in 3GPP TS 38.415
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PduSessionType {
    /// Downlink PDU Session Information
    Dl = 0,
    /// Uplink PDU Session Information
    Ul = 1,
}

impl PduSessionType {
    /// Convert from u8 to `PduSessionType`
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Dl),
            1 => Some(Self::Ul),
            _ => None,
        }
    }
}

/// PDU Session Container Information Element
///
/// Carries 5GC-specific PDU session information inside the GTP-U
/// PDU Session Container extension header (type 0x85) as defined in
/// 3GPP TS 38.415.
///
/// # Downlink PDU Session Information
///
/// Contains the QoS Flow Identifier (QFI), Reflective QoS Indicator (RQI),
/// Paging Policy Presence (PPP), and Paging Policy Indicator (PPI).
///
/// # Uplink PDU Session Information
///
/// Contains the QoS Flow Identifier (QFI) and an optional DL Sending
/// Timestamp for N9 interface usage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PduSessionInfo {
    /// PDU type: DL (0) or UL (1)
    pub pdu_type: PduSessionType,
    /// QoS Flow Identifier (6 bits, 0-63)
    pub qfi: u8,
    /// Reflective QoS Indicator (DL only, 1 bit)
    pub rqi: bool,
    /// Paging Policy Presence (DL only, 1 bit)
    pub ppp: bool,
    /// Paging Policy Indicator (DL only, 3 bits, valid when ppp is true)
    pub ppi: u8,
    /// DL Sending Timestamp or additional data (UL, for N9 interface)
    pub dl_sending_timestamp: Option<u32>,
}

impl PduSessionInfo {
    /// Create a downlink PDU session info
    pub fn downlink(qfi: u8) -> Self {
        Self {
            pdu_type: PduSessionType::Dl,
            qfi: qfi & 0x3F,
            rqi: false,
            ppp: false,
            ppi: 0,
            dl_sending_timestamp: None,
        }
    }

    /// Create an uplink PDU session info
    pub fn uplink(qfi: u8) -> Self {
        Self {
            pdu_type: PduSessionType::Ul,
            qfi: qfi & 0x3F,
            rqi: false,
            ppp: false,
            ppi: 0,
            dl_sending_timestamp: None,
        }
    }

    /// Set the Reflective QoS Indicator (DL only)
    pub fn with_rqi(mut self, rqi: bool) -> Self {
        self.rqi = rqi;
        self
    }

    /// Set the Paging Policy Presence and Indicator (DL only)
    pub fn with_paging_policy(mut self, ppi: u8) -> Self {
        self.ppp = true;
        self.ppi = ppi & 0x07;
        self
    }

    /// Set the DL Sending Timestamp (UL, for N9 interface)
    pub fn with_dl_sending_timestamp(mut self, ts: u32) -> Self {
        self.dl_sending_timestamp = Some(ts);
        self
    }

    /// Encode the PDU Session Info to bytes (content only, without ext header framing)
    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(8);

        // First octet: PDU Type (4 bits) | spare (4 bits)
        // Bit layout per TS 38.415:
        //   Octet 1: PDU Type (4 bits) | QFI (6 bits) spread across octets 1-2
        // Simplified encoding:
        //   Octet 1: [PDU Type (4b)] [RQI (1b)] [QFI high 3 bits]
        //   Octet 2: [QFI low 3 bits] [PPP (1b)] [PPI (3b)] [spare (1b)]

        let pdu_type_val = self.pdu_type as u8;
        let rqi_val: u8 = if self.rqi { 1 } else { 0 };
        let ppp_val: u8 = if self.ppp { 1 } else { 0 };

        let octet1 = (pdu_type_val << 4) | (rqi_val << 3) | ((self.qfi >> 3) & 0x07);
        let octet2 = ((self.qfi & 0x07) << 5) | (ppp_val << 4) | ((self.ppi & 0x07) << 1);

        buf.put_u8(octet1);
        buf.put_u8(octet2);

        if let Some(ts) = self.dl_sending_timestamp {
            buf.put_u32(ts);
        }

        buf.freeze()
    }

    /// Decode PDU Session Info from bytes (content only, without ext header framing)
    ///
    /// # Errors
    ///
    /// Returns `GtpError::BufferTooShort` if the data is shorter than 2 bytes.
    /// Returns `GtpError::InvalidExtHeaderType` if the PDU type field is invalid.
    pub fn decode(data: &[u8]) -> Result<Self, GtpError> {
        if data.len() < 2 {
            return Err(GtpError::BufferTooShort {
                needed: 2,
                available: data.len(),
            });
        }

        let octet1 = data[0];
        let octet2 = data[1];

        let pdu_type_val = (octet1 >> 4) & 0x0F;
        let pdu_type = PduSessionType::from_u8(pdu_type_val)
            .ok_or(GtpError::InvalidExtHeaderType(pdu_type_val))?;

        let rqi = ((octet1 >> 3) & 0x01) != 0;
        let qfi_high = octet1 & 0x07;
        let qfi_low = (octet2 >> 5) & 0x07;
        let qfi = (qfi_high << 3) | qfi_low;
        let ppp = ((octet2 >> 4) & 0x01) != 0;
        let ppi = (octet2 >> 1) & 0x07;

        let dl_sending_timestamp = if data.len() >= 6 {
            Some(u32::from_be_bytes([data[2], data[3], data[4], data[5]]))
        } else {
            None
        };

        Ok(Self {
            pdu_type,
            qfi,
            rqi,
            ppp,
            ppi,
            dl_sending_timestamp,
        })
    }
}


// ---------------------------------------------------------------------------
// 6G Extension: TSN Marker (A5.3 - Deterministic Networking)
// ---------------------------------------------------------------------------

/// Time-Sensitive Networking (TSN) marker for 6G deterministic networking
///
/// This is a 6G-forward extension that embeds TSN scheduling information
/// into GTP-U packets, enabling deterministic latency guarantees for
/// time-critical traffic (e.g. industrial automation, XR).
///
/// Encoded as a custom GTP-U extension header (type 0xE1).
///
/// # Wire format (12 bytes content, 16 bytes total with framing)
///
/// ```text
/// +-------+-------+-------+-------+
/// |  stream_id    | sequence_num  |
/// +-------+-------+-------+-------+
/// |          timestamp            |
/// +-------+-------+-------+-------+
/// |priority| reserved (3 bytes)   |
/// +-------+-------+-------+-------+
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TsnMarker {
    /// TSN stream identifier
    pub stream_id: u16,
    /// Sequence number within the stream
    pub sequence_number: u16,
    /// Timestamp in microseconds (relative to TSN epoch)
    pub timestamp: u64,
    /// Priority level (0 = highest, 7 = lowest)
    pub priority: u8,
}

impl TsnMarker {
    /// Create a new TSN marker
    pub fn new(stream_id: u16, sequence_number: u16, timestamp: u64, priority: u8) -> Self {
        Self {
            stream_id,
            sequence_number,
            timestamp,
            priority: priority & 0x07,
        }
    }

    /// Encoded content size (without extension header length/type framing)
    pub const CONTENT_SIZE: usize = 13; // 2+2+8+1

    /// Encode TSN marker content to bytes
    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(Self::CONTENT_SIZE);
        buf.put_u16(self.stream_id);
        buf.put_u16(self.sequence_number);
        buf.put_u64(self.timestamp);
        buf.put_u8(self.priority);
        buf.freeze()
    }

    /// Decode TSN marker content from bytes
    ///
    /// # Errors
    ///
    /// Returns `GtpError::BufferTooShort` if the data is shorter than 13 bytes.
    pub fn decode(data: &[u8]) -> Result<Self, GtpError> {
        if data.len() < Self::CONTENT_SIZE {
            return Err(GtpError::BufferTooShort {
                needed: Self::CONTENT_SIZE,
                available: data.len(),
            });
        }
        let stream_id = u16::from_be_bytes([data[0], data[1]]);
        let sequence_number = u16::from_be_bytes([data[2], data[3]]);
        let timestamp = u64::from_be_bytes([
            data[4], data[5], data[6], data[7],
            data[8], data[9], data[10], data[11],
        ]);
        let priority = data[12] & 0x07;

        Ok(Self {
            stream_id,
            sequence_number,
            timestamp,
            priority,
        })
    }
}


// ---------------------------------------------------------------------------
// 6G Extension: In-Network Compute Marker (A5.4)
// ---------------------------------------------------------------------------

/// Processing hint for in-network computing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ProcessingHint {
    /// No specific processing required
    None = 0,
    /// Aggregate data with other packets
    Aggregate = 1,
    /// Apply compression
    Compress = 2,
    /// Apply filtering/sampling
    Filter = 3,
    /// Apply inference/ML model
    Inference = 4,
    /// Apply transcoding (e.g. video)
    Transcode = 5,
    /// Cache the result at this node
    Cache = 6,
}

impl ProcessingHint {
    /// Convert from u8 to `ProcessingHint`
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::None),
            1 => Some(Self::Aggregate),
            2 => Some(Self::Compress),
            3 => Some(Self::Filter),
            4 => Some(Self::Inference),
            5 => Some(Self::Transcode),
            6 => Some(Self::Cache),
            _ => None,
        }
    }
}

/// Data locality hint for in-network computing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DataLocality {
    /// No locality preference
    Any = 0,
    /// Process at the edge (closest UPF)
    Edge = 1,
    /// Process at regional node
    Regional = 2,
    /// Process at core network
    Core = 3,
}

impl DataLocality {
    /// Convert from u8 to `DataLocality`
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Any),
            1 => Some(Self::Edge),
            2 => Some(Self::Regional),
            3 => Some(Self::Core),
            _ => None,
        }
    }
}

/// In-Network Compute marker for 6G in-network computing
///
/// This is a 6G-forward extension that embeds compute task metadata
/// into GTP-U packets, enabling UPFs and intermediate nodes to perform
/// in-network processing (aggregation, compression, inference, etc.).
///
/// Encoded as a custom GTP-U extension header (type 0xE2).
///
/// # Wire format (6 bytes content, 8 bytes total with framing)
///
/// ```text
/// +-------+-------+-------+-------+
/// |     compute_task_id           |
/// +-------+-------+-------+-------+
/// | locality| hint  | reserved    |
/// +-------+-------+-------+-------+
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InNetworkComputeMarker {
    /// Compute task identifier (globally unique within the slice)
    pub compute_task_id: u32,
    /// Data locality preference
    pub data_locality: DataLocality,
    /// Processing hint
    pub processing_hint: ProcessingHint,
}

impl InNetworkComputeMarker {
    /// Create a new in-network compute marker
    pub fn new(
        compute_task_id: u32,
        data_locality: DataLocality,
        processing_hint: ProcessingHint,
    ) -> Self {
        Self {
            compute_task_id,
            data_locality,
            processing_hint,
        }
    }

    /// Encoded content size (without extension header framing)
    pub const CONTENT_SIZE: usize = 6; // 4 + 1 + 1

    /// Encode in-network compute marker content to bytes
    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(Self::CONTENT_SIZE);
        buf.put_u32(self.compute_task_id);
        buf.put_u8(self.data_locality as u8);
        buf.put_u8(self.processing_hint as u8);
        buf.freeze()
    }

    /// Decode in-network compute marker content from bytes
    ///
    /// # Errors
    ///
    /// Returns `GtpError::BufferTooShort` if the data is shorter than 6 bytes.
    /// Returns `GtpError::InvalidExtHeaderType` if locality or hint values are invalid.
    pub fn decode(data: &[u8]) -> Result<Self, GtpError> {
        if data.len() < Self::CONTENT_SIZE {
            return Err(GtpError::BufferTooShort {
                needed: Self::CONTENT_SIZE,
                available: data.len(),
            });
        }
        let compute_task_id = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
        let data_locality = DataLocality::from_u8(data[4])
            .ok_or(GtpError::InvalidExtHeaderType(data[4]))?;
        let processing_hint = ProcessingHint::from_u8(data[5])
            .ok_or(GtpError::InvalidExtHeaderType(data[5]))?;

        Ok(Self {
            compute_task_id,
            data_locality,
            processing_hint,
        })
    }
}


/// GTP-U Extension Header
///
/// Represents a single extension header in a GTP-U extension header chain.
/// Multiple extension headers can be chained together; each carries a
/// `next_extension_header_type` field during encoding/decoding.
///
/// Extension header format (per 3GPP TS 29.281):
/// - Length (1 byte): content length in 4-byte units (including length and next-type bytes)
/// - Content (variable): extension-specific data
/// - Next Extension Header Type (1 byte): type of the next header, or 0x00 for end of chain
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
    /// PDU Session Container extension header (raw data)
    PduSessionContainer {
        /// PDU session information data
        data: Bytes,
    },
    /// PDU Session Container extension header (structured, 5GC)
    PduSessionContainerInfo {
        /// Structured PDU session information
        info: PduSessionInfo,
    },
    /// NR RAN Container extension header
    NrRanContainer {
        /// NR RAN container data
        data: Bytes,
    },
    /// 6G TSN (Time-Sensitive Networking) marker extension header
    TsnMarker {
        /// TSN marker data
        marker: TsnMarker,
    },
    /// 6G In-Network Compute marker extension header
    InNetworkCompute {
        /// In-network compute marker data
        marker: InNetworkComputeMarker,
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
    /// Invalid PDU session info
    #[error("invalid PDU session info: {0}")]
    InvalidPduSessionInfo(String),
    /// Extension header chain too long
    #[error("extension header chain too long: {0} headers")]
    ExtHeaderChainTooLong(usize),
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

    /// Maximum number of extension headers allowed in a chain
    pub const MAX_EXT_HEADER_CHAIN: usize = 16;

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

    /// Add an extension header to the chain
    ///
    /// Extension headers are chained in order: each header's
    /// `next_extension_header_type` field will point to the following header,
    /// and the last header in the chain points to `NoMore` (0x00).
    pub fn with_extension_header(mut self, ext: GtpExtHeader) -> Self {
        self.extension_headers.push(ext);
        self
    }

    /// Add multiple extension headers to the chain at once
    ///
    /// This is a convenience method for building extension header chains.
    /// Headers are appended in the order provided.
    pub fn with_extension_headers(mut self, exts: Vec<GtpExtHeader>) -> Self {
        self.extension_headers.extend(exts);
        self
    }

    /// Add a structured PDU Session Container extension header
    pub fn with_pdu_session_info(self, info: PduSessionInfo) -> Self {
        self.with_extension_header(GtpExtHeader::PduSessionContainerInfo { info })
    }

    /// Add a TSN marker extension header
    pub fn with_tsn_marker(self, marker: TsnMarker) -> Self {
        self.with_extension_header(GtpExtHeader::TsnMarker { marker })
    }

    /// Add an in-network compute marker extension header
    pub fn with_in_network_compute(self, marker: InNetworkComputeMarker) -> Self {
        self.with_extension_header(GtpExtHeader::InNetworkCompute { marker })
    }

    /// Find the first extension header of a given type in the chain
    pub fn find_extension_header(&self, ext_type: ExtHeaderType) -> Option<&GtpExtHeader> {
        self.extension_headers.iter().find(|ext| ext.ext_type() == ext_type)
    }

    /// Get the number of extension headers in the chain
    pub fn extension_header_count(&self) -> usize {
        self.extension_headers.len()
    }

    /// Extract the PDU Session Info from the extension header chain, if present
    pub fn pdu_session_info(&self) -> Option<&PduSessionInfo> {
        for ext in &self.extension_headers {
            if let GtpExtHeader::PduSessionContainerInfo { info } = ext {
                return Some(info);
            }
        }
        None
    }

    /// Extract the TSN marker from the extension header chain, if present
    pub fn tsn_marker(&self) -> Option<&TsnMarker> {
        for ext in &self.extension_headers {
            if let GtpExtHeader::TsnMarker { marker } = ext {
                return Some(marker);
            }
        }
        None
    }

    /// Extract the in-network compute marker from the extension header chain, if present
    pub fn in_network_compute_marker(&self) -> Option<&InNetworkComputeMarker> {
        for ext in &self.extension_headers {
            if let GtpExtHeader::InNetworkCompute { marker } = ext {
                return Some(marker);
            }
        }
        None
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

            // Parse extension header chain
            while next_ext_type != ExtHeaderType::NoMore as u8 {
                if extension_headers.len() >= Self::MAX_EXT_HEADER_CHAIN {
                    return Err(GtpError::ExtHeaderChainTooLong(extension_headers.len()));
                }
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
    /// Get the extension header type identifier for this header
    pub fn ext_type(&self) -> ExtHeaderType {
        match self {
            Self::UdpPort { .. } => ExtHeaderType::UdpPort,
            Self::PdcpPduNumber { .. } => ExtHeaderType::PdcpPduNumber,
            Self::LongPdcpPduNumber { .. } => ExtHeaderType::LongPdcpPduNumber,
            Self::PduSessionContainer { .. } | Self::PduSessionContainerInfo { .. } => {
                ExtHeaderType::PduSessionContainer
            }
            Self::NrRanContainer { .. } => ExtHeaderType::NrRanContainer,
            Self::TsnMarker { .. } => ExtHeaderType::TsnMarker,
            Self::InNetworkCompute { .. } => ExtHeaderType::InNetworkCompute,
        }
    }

    /// Calculate the encoded size of this extension header
    ///
    /// The size includes the type byte (which serves as the previous header's
    /// `next_extension_header_type`), the length byte, content, and any padding
    /// needed to align to a 4-byte boundary.
    pub fn encoded_size(&self) -> usize {
        match self {
            Self::UdpPort { .. } => 4,           // type(1) + len(1) + port(2)
            Self::PdcpPduNumber { .. } => 4,     // type(1) + len(1) + pdu_num(2)
            Self::LongPdcpPduNumber { .. } => 8, // type(1) + len(1) + pdu_num(3) + padding(3)
            Self::PduSessionContainer { data } => {
                // type(1) + len(1) + data + padding to 4-byte boundary
                let content_len = data.len() + 2;
                content_len.div_ceil(4) * 4
            }
            Self::PduSessionContainerInfo { info } => {
                let encoded = info.encode();
                let content_len = encoded.len() + 2;
                content_len.div_ceil(4) * 4
            }
            Self::NrRanContainer { data } => {
                let content_len = data.len() + 2;
                content_len.div_ceil(4) * 4
            }
            Self::TsnMarker { .. } => {
                // type(1) + len(1) + content(13) + padding to 4-byte boundary
                // 13 + 2 = 15, padded to 16
                let content_len = TsnMarker::CONTENT_SIZE + 2;
                content_len.div_ceil(4) * 4
            }
            Self::InNetworkCompute { .. } => {
                // type(1) + len(1) + content(6) = 8, already 4-byte aligned
                let content_len = InNetworkComputeMarker::CONTENT_SIZE + 2;
                content_len.div_ceil(4) * 4
            }
        }
    }

    /// Encode the extension header to a buffer
    ///
    /// Writes the type byte (which the GTP-U framing uses as the previous
    /// header's `next_extension_header_type`), followed by the length in
    /// 4-byte units, content data, and padding.
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
                let len_units = (data.len() + 2).div_ceil(4);
                buf.put_u8(len_units as u8);
                buf.put_slice(data);
                // Add padding to 4-byte boundary
                let padding = (4 - ((data.len() + 2) % 4)) % 4;
                for _ in 0..padding {
                    buf.put_u8(0);
                }
            }
            Self::PduSessionContainerInfo { info } => {
                let encoded = info.encode();
                buf.put_u8(ExtHeaderType::PduSessionContainer as u8);
                let len_units = (encoded.len() + 2).div_ceil(4);
                buf.put_u8(len_units as u8);
                buf.put_slice(&encoded);
                let padding = (4 - ((encoded.len() + 2) % 4)) % 4;
                for _ in 0..padding {
                    buf.put_u8(0);
                }
            }
            Self::NrRanContainer { data } => {
                buf.put_u8(ExtHeaderType::NrRanContainer as u8);
                let len_units = (data.len() + 2).div_ceil(4);
                buf.put_u8(len_units as u8);
                buf.put_slice(data);
                let padding = (4 - ((data.len() + 2) % 4)) % 4;
                for _ in 0..padding {
                    buf.put_u8(0);
                }
            }
            Self::TsnMarker { marker } => {
                let encoded = marker.encode();
                buf.put_u8(ExtHeaderType::TsnMarker as u8);
                let len_units = (encoded.len() + 2).div_ceil(4);
                buf.put_u8(len_units as u8);
                buf.put_slice(&encoded);
                let padding = (4 - ((encoded.len() + 2) % 4)) % 4;
                for _ in 0..padding {
                    buf.put_u8(0);
                }
            }
            Self::InNetworkCompute { marker } => {
                let encoded = marker.encode();
                buf.put_u8(ExtHeaderType::InNetworkCompute as u8);
                let len_units = (encoded.len() + 2).div_ceil(4);
                buf.put_u8(len_units as u8);
                buf.put_slice(&encoded);
                let padding = (4 - ((encoded.len() + 2) % 4)) % 4;
                for _ in 0..padding {
                    buf.put_u8(0);
                }
            }
        }
    }


    /// Decode an extension header from bytes
    ///
    /// Decodes a single extension header from the chain. The `ext_type` parameter
    /// is the `next_extension_header_type` value from the previous header or
    /// the optional fields area.
    ///
    /// Returns (header, bytes_consumed, next_ext_type) where `next_ext_type`
    /// is the type of the next header in the chain (0x00 = end of chain).
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
                // PDU Session Container - try structured decode first
                let content_len = total_len - 2; // minus len byte and next_ext_type byte
                let content = &data[1..1 + content_len];
                match PduSessionInfo::decode(content) {
                    Ok(info) => Self::PduSessionContainerInfo { info },
                    Err(_) => {
                        // Fall back to raw data
                        Self::PduSessionContainer {
                            data: Bytes::copy_from_slice(content),
                        }
                    }
                }
            }
            0x84 => {
                // NR RAN Container
                let content_len = total_len - 2;
                let content = Bytes::copy_from_slice(&data[1..1 + content_len]);
                Self::NrRanContainer { data: content }
            }
            0xE1 => {
                // TSN Marker (6G extension)
                let content_len = total_len - 2;
                let content = &data[1..1 + content_len];
                let marker = TsnMarker::decode(content)?;
                Self::TsnMarker { marker }
            }
            0xE2 => {
                // In-Network Compute Marker (6G extension)
                let content_len = total_len - 2;
                let content = &data[1..1 + content_len];
                let marker = InNetworkComputeMarker::decode(content)?;
                Self::InNetworkCompute { marker }
            }
            _ => return Err(GtpError::InvalidExtHeaderType(ext_type)),
        };

        Ok((ext, total_len, next_ext_type))
    }
}


// ---------------------------------------------------------------------------
// Extension Header Chain builder (A5.1)
// ---------------------------------------------------------------------------

/// Builder for constructing GTP-U extension header chains
///
/// Provides a fluent API for building ordered chains of extension headers.
/// The chain is validated before being applied to a GTP-U header.
///
/// # Example
///
/// ```
/// use nextgsim_gtp::codec::*;
///
/// let chain = ExtHeaderChain::new()
///     .push(GtpExtHeader::UdpPort { port: 2152 })
///     .push(GtpExtHeader::PdcpPduNumber { pdu_number: 0x1234 });
///
/// assert_eq!(chain.len(), 2);
/// assert!(chain.contains(ExtHeaderType::UdpPort));
/// ```
#[derive(Debug, Clone, Default)]
pub struct ExtHeaderChain {
    headers: Vec<GtpExtHeader>,
}

impl ExtHeaderChain {
    /// Create a new empty extension header chain
    pub fn new() -> Self {
        Self {
            headers: Vec::new(),
        }
    }

    /// Add an extension header to the end of the chain
    pub fn push(mut self, ext: GtpExtHeader) -> Self {
        self.headers.push(ext);
        self
    }

    /// Add a PDU Session Container with structured info
    pub fn push_pdu_session_info(self, info: PduSessionInfo) -> Self {
        self.push(GtpExtHeader::PduSessionContainerInfo { info })
    }

    /// Add a TSN marker
    pub fn push_tsn_marker(self, marker: TsnMarker) -> Self {
        self.push(GtpExtHeader::TsnMarker { marker })
    }

    /// Add an in-network compute marker
    pub fn push_in_network_compute(self, marker: InNetworkComputeMarker) -> Self {
        self.push(GtpExtHeader::InNetworkCompute { marker })
    }

    /// Get the number of headers in the chain
    pub fn len(&self) -> usize {
        self.headers.len()
    }

    /// Check if the chain is empty
    pub fn is_empty(&self) -> bool {
        self.headers.is_empty()
    }

    /// Check if the chain contains a header of the given type
    pub fn contains(&self, ext_type: ExtHeaderType) -> bool {
        self.headers.iter().any(|h| h.ext_type() == ext_type)
    }

    /// Find the first header of the given type
    pub fn find(&self, ext_type: ExtHeaderType) -> Option<&GtpExtHeader> {
        self.headers.iter().find(|h| h.ext_type() == ext_type)
    }

    /// Get an iterator over the headers in the chain
    pub fn iter(&self) -> std::slice::Iter<'_, GtpExtHeader> {
        self.headers.iter()
    }

    /// Convert the chain into a vector of extension headers
    pub fn into_headers(self) -> Vec<GtpExtHeader> {
        self.headers
    }

    /// Validate the chain length
    ///
    /// # Errors
    ///
    /// Returns `GtpError::ExtHeaderChainTooLong` if the chain exceeds
    /// `GtpHeader::MAX_EXT_HEADER_CHAIN` headers.
    pub fn validate(&self) -> Result<(), GtpError> {
        if self.headers.len() > GtpHeader::MAX_EXT_HEADER_CHAIN {
            return Err(GtpError::ExtHeaderChainTooLong(self.headers.len()));
        }
        Ok(())
    }

    /// Apply this chain to a GTP-U header, replacing any existing extension headers
    ///
    /// # Errors
    ///
    /// Returns `GtpError::ExtHeaderChainTooLong` if the chain exceeds the maximum length.
    pub fn apply_to(self, mut header: GtpHeader) -> Result<GtpHeader, GtpError> {
        self.validate()?;
        header.extension_headers = self.headers;
        Ok(header)
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
        // The decoder will attempt structured decode first; if the raw bytes
        // happen to parse as valid PduSessionInfo, we get PduSessionContainerInfo.
        // Otherwise we get raw PduSessionContainer.
        match &decoded.extension_headers[0] {
            GtpExtHeader::PduSessionContainer { data } => {
                // Data includes padding, so check prefix
                assert!(data.starts_with(&container_data));
            }
            GtpExtHeader::PduSessionContainerInfo { info } => {
                // Structured decode succeeded - this is valid behavior
                assert_eq!(info.pdu_type, PduSessionType::Dl);
            }
            other => panic!("Expected PduSessionContainer or PduSessionContainerInfo, got {other:?}"),
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

    // =======================================================================
    // A5.1: Extension Header Chaining tests
    // =======================================================================

    #[test]
    fn test_extension_header_chain_three_headers() {
        let header = GtpHeader::g_pdu(0x12345678, Bytes::from_static(b"data"))
            .with_sequence_number(1)
            .with_extension_header(GtpExtHeader::UdpPort { port: 2152 })
            .with_extension_header(GtpExtHeader::PdcpPduNumber { pdu_number: 0x5678 })
            .with_extension_header(GtpExtHeader::LongPdcpPduNumber { pdu_number: 0x1FFFF });

        let encoded = header.encode();
        let decoded = GtpHeader::decode(&encoded).unwrap();

        assert_eq!(decoded.extension_header_count(), 3);
        assert_eq!(decoded.sequence_number, Some(1));

        match &decoded.extension_headers[0] {
            GtpExtHeader::UdpPort { port } => assert_eq!(*port, 2152),
            other => panic!("Expected UdpPort, got {other:?}"),
        }
        match &decoded.extension_headers[1] {
            GtpExtHeader::PdcpPduNumber { pdu_number } => assert_eq!(*pdu_number, 0x5678),
            other => panic!("Expected PdcpPduNumber, got {other:?}"),
        }
        match &decoded.extension_headers[2] {
            GtpExtHeader::LongPdcpPduNumber { pdu_number } => assert_eq!(*pdu_number, 0x1FFFF),
            other => panic!("Expected LongPdcpPduNumber, got {other:?}"),
        }

        assert_eq!(decoded.payload, Bytes::from_static(b"data"));
    }

    #[test]
    fn test_ext_header_chain_builder() {
        let chain = ExtHeaderChain::new()
            .push(GtpExtHeader::UdpPort { port: 2152 })
            .push(GtpExtHeader::PdcpPduNumber { pdu_number: 0x1234 });

        assert_eq!(chain.len(), 2);
        assert!(!chain.is_empty());
        assert!(chain.contains(ExtHeaderType::UdpPort));
        assert!(chain.contains(ExtHeaderType::PdcpPduNumber));
        assert!(!chain.contains(ExtHeaderType::TsnMarker));

        let header = chain
            .apply_to(GtpHeader::g_pdu(0x1000, Bytes::new()))
            .unwrap();
        assert_eq!(header.extension_headers.len(), 2);

        let encoded = header.encode();
        let decoded = GtpHeader::decode(&encoded).unwrap();
        assert_eq!(decoded.extension_headers.len(), 2);
    }

    #[test]
    fn test_ext_header_chain_with_multiple_extension_headers() {
        let header = GtpHeader::g_pdu(0xAABBCCDD, Bytes::from_static(b"chain_test"))
            .with_extension_headers(vec![
                GtpExtHeader::UdpPort { port: 8080 },
                GtpExtHeader::PdcpPduNumber { pdu_number: 42 },
            ]);

        let encoded = header.encode();
        let decoded = GtpHeader::decode(&encoded).unwrap();

        assert_eq!(decoded.extension_headers.len(), 2);
        assert_eq!(decoded.payload, Bytes::from_static(b"chain_test"));
    }

    #[test]
    fn test_find_extension_header() {
        let header = GtpHeader::g_pdu(0x1000, Bytes::new())
            .with_extension_header(GtpExtHeader::UdpPort { port: 2152 })
            .with_extension_header(GtpExtHeader::PdcpPduNumber { pdu_number: 0x5678 });

        let found = header.find_extension_header(ExtHeaderType::UdpPort);
        assert!(found.is_some());
        match found.unwrap() {
            GtpExtHeader::UdpPort { port } => assert_eq!(*port, 2152),
            other => panic!("Expected UdpPort, got {other:?}"),
        }

        assert!(header.find_extension_header(ExtHeaderType::TsnMarker).is_none());
    }

    #[test]
    fn test_ext_header_type_method() {
        assert_eq!(GtpExtHeader::UdpPort { port: 0 }.ext_type(), ExtHeaderType::UdpPort);
        assert_eq!(GtpExtHeader::PdcpPduNumber { pdu_number: 0 }.ext_type(), ExtHeaderType::PdcpPduNumber);
        assert_eq!(
            GtpExtHeader::TsnMarker { marker: TsnMarker::new(0, 0, 0, 0) }.ext_type(),
            ExtHeaderType::TsnMarker
        );
        assert_eq!(
            GtpExtHeader::InNetworkCompute {
                marker: InNetworkComputeMarker::new(0, DataLocality::Any, ProcessingHint::None)
            }.ext_type(),
            ExtHeaderType::InNetworkCompute
        );
    }

    // =======================================================================
    // A5.2: PDU Session Container IE tests
    // =======================================================================

    #[test]
    fn test_pdu_session_info_dl_encode_decode() {
        let info = PduSessionInfo::downlink(42)
            .with_rqi(true)
            .with_paging_policy(3);

        let encoded = info.encode();
        let decoded = PduSessionInfo::decode(&encoded).unwrap();

        assert_eq!(decoded.pdu_type, PduSessionType::Dl);
        assert_eq!(decoded.qfi, 42);
        assert!(decoded.rqi);
        assert!(decoded.ppp);
        assert_eq!(decoded.ppi, 3);
        assert!(decoded.dl_sending_timestamp.is_none());
    }

    #[test]
    fn test_pdu_session_info_ul_encode_decode() {
        let info = PduSessionInfo::uplink(15)
            .with_dl_sending_timestamp(0xDEADBEEF);

        let encoded = info.encode();
        let decoded = PduSessionInfo::decode(&encoded).unwrap();

        assert_eq!(decoded.pdu_type, PduSessionType::Ul);
        assert_eq!(decoded.qfi, 15);
        assert!(!decoded.rqi);
        assert!(!decoded.ppp);
        assert_eq!(decoded.dl_sending_timestamp, Some(0xDEADBEEF));
    }

    #[test]
    fn test_pdu_session_info_qfi_range() {
        // QFI should be masked to 6 bits (0-63)
        let info = PduSessionInfo::downlink(63);
        assert_eq!(info.qfi, 63);

        let info = PduSessionInfo::downlink(0xFF); // should mask to 0x3F = 63
        assert_eq!(info.qfi, 63);
    }

    #[test]
    fn test_pdu_session_info_as_ext_header() {
        let info = PduSessionInfo::downlink(10).with_rqi(true);
        let header = GtpHeader::g_pdu(0x12345678, Bytes::from_static(b"5g_data"))
            .with_pdu_session_info(info);

        let encoded = header.encode();
        let decoded = GtpHeader::decode(&encoded).unwrap();

        assert_eq!(decoded.extension_headers.len(), 1);
        let pdu_info = decoded.pdu_session_info().unwrap();
        assert_eq!(pdu_info.pdu_type, PduSessionType::Dl);
        assert_eq!(pdu_info.qfi, 10);
        assert!(pdu_info.rqi);
    }

    #[test]
    fn test_pdu_session_container_info_in_chain() {
        let info = PduSessionInfo::uplink(7);
        let header = GtpHeader::g_pdu(0xABCD1234, Bytes::new())
            .with_pdu_session_info(info)
            .with_extension_header(GtpExtHeader::UdpPort { port: 2152 });

        let encoded = header.encode();
        let decoded = GtpHeader::decode(&encoded).unwrap();

        assert_eq!(decoded.extension_headers.len(), 2);
        assert!(decoded.pdu_session_info().is_some());
        assert_eq!(decoded.pdu_session_info().unwrap().qfi, 7);
    }

    #[test]
    fn test_pdu_session_type_conversion() {
        assert_eq!(PduSessionType::from_u8(0), Some(PduSessionType::Dl));
        assert_eq!(PduSessionType::from_u8(1), Some(PduSessionType::Ul));
        assert_eq!(PduSessionType::from_u8(2), None);
    }

    // =======================================================================
    // A5.3: TSN Marker tests
    // =======================================================================

    #[test]
    fn test_tsn_marker_encode_decode() {
        let marker = TsnMarker::new(42, 100, 1_000_000, 3);

        let encoded = marker.encode();
        let decoded = TsnMarker::decode(&encoded).unwrap();

        assert_eq!(decoded.stream_id, 42);
        assert_eq!(decoded.sequence_number, 100);
        assert_eq!(decoded.timestamp, 1_000_000);
        assert_eq!(decoded.priority, 3);
    }

    #[test]
    fn test_tsn_marker_priority_mask() {
        // Priority should be masked to 3 bits (0-7)
        let marker = TsnMarker::new(1, 1, 1, 0xFF);
        assert_eq!(marker.priority, 7);
    }

    #[test]
    fn test_tsn_marker_as_ext_header() {
        let marker = TsnMarker::new(1, 42, 999_999, 0);
        let header = GtpHeader::g_pdu(0x12345678, Bytes::from_static(b"tsn_data"))
            .with_tsn_marker(marker);

        let encoded = header.encode();
        let decoded = GtpHeader::decode(&encoded).unwrap();

        assert_eq!(decoded.extension_headers.len(), 1);
        let tsn = decoded.tsn_marker().unwrap();
        assert_eq!(tsn.stream_id, 1);
        assert_eq!(tsn.sequence_number, 42);
        assert_eq!(tsn.timestamp, 999_999);
        assert_eq!(tsn.priority, 0);
    }

    #[test]
    fn test_tsn_marker_in_chain_with_other_headers() {
        let tsn = TsnMarker::new(5, 10, 500_000, 2);
        let header = GtpHeader::g_pdu(0x1000, Bytes::from_static(b"chain"))
            .with_sequence_number(7)
            .with_extension_header(GtpExtHeader::UdpPort { port: 2152 })
            .with_tsn_marker(tsn)
            .with_extension_header(GtpExtHeader::PdcpPduNumber { pdu_number: 0xABCD });

        let encoded = header.encode();
        let decoded = GtpHeader::decode(&encoded).unwrap();

        assert_eq!(decoded.extension_header_count(), 3);
        assert_eq!(decoded.sequence_number, Some(7));

        // Verify chain order
        assert_eq!(decoded.extension_headers[0].ext_type(), ExtHeaderType::UdpPort);
        assert_eq!(decoded.extension_headers[1].ext_type(), ExtHeaderType::TsnMarker);
        assert_eq!(decoded.extension_headers[2].ext_type(), ExtHeaderType::PdcpPduNumber);

        let tsn = decoded.tsn_marker().unwrap();
        assert_eq!(tsn.stream_id, 5);
        assert_eq!(tsn.sequence_number, 10);
        assert_eq!(tsn.timestamp, 500_000);
        assert_eq!(tsn.priority, 2);
    }

    #[test]
    fn test_tsn_marker_decode_buffer_too_short() {
        let result = TsnMarker::decode(&[0x00, 0x01]);
        assert!(matches!(result, Err(GtpError::BufferTooShort { .. })));
    }

    // =======================================================================
    // A5.4: In-Network Compute Marker tests
    // =======================================================================

    #[test]
    fn test_in_network_compute_marker_encode_decode() {
        let marker = InNetworkComputeMarker::new(
            0xCAFEBABE,
            DataLocality::Edge,
            ProcessingHint::Inference,
        );

        let encoded = marker.encode();
        let decoded = InNetworkComputeMarker::decode(&encoded).unwrap();

        assert_eq!(decoded.compute_task_id, 0xCAFEBABE);
        assert_eq!(decoded.data_locality, DataLocality::Edge);
        assert_eq!(decoded.processing_hint, ProcessingHint::Inference);
    }

    #[test]
    fn test_in_network_compute_as_ext_header() {
        let marker = InNetworkComputeMarker::new(
            42,
            DataLocality::Regional,
            ProcessingHint::Aggregate,
        );
        let header = GtpHeader::g_pdu(0x12345678, Bytes::from_static(b"compute"))
            .with_in_network_compute(marker);

        let encoded = header.encode();
        let decoded = GtpHeader::decode(&encoded).unwrap();

        assert_eq!(decoded.extension_headers.len(), 1);
        let inc = decoded.in_network_compute_marker().unwrap();
        assert_eq!(inc.compute_task_id, 42);
        assert_eq!(inc.data_locality, DataLocality::Regional);
        assert_eq!(inc.processing_hint, ProcessingHint::Aggregate);
    }

    #[test]
    fn test_in_network_compute_all_hints() {
        let hints = [
            ProcessingHint::None,
            ProcessingHint::Aggregate,
            ProcessingHint::Compress,
            ProcessingHint::Filter,
            ProcessingHint::Inference,
            ProcessingHint::Transcode,
            ProcessingHint::Cache,
        ];
        for hint in hints {
            let marker = InNetworkComputeMarker::new(1, DataLocality::Any, hint);
            let encoded = marker.encode();
            let decoded = InNetworkComputeMarker::decode(&encoded).unwrap();
            assert_eq!(decoded.processing_hint, hint);
        }
    }

    #[test]
    fn test_in_network_compute_all_localities() {
        let localities = [
            DataLocality::Any,
            DataLocality::Edge,
            DataLocality::Regional,
            DataLocality::Core,
        ];
        for locality in localities {
            let marker = InNetworkComputeMarker::new(1, locality, ProcessingHint::None);
            let encoded = marker.encode();
            let decoded = InNetworkComputeMarker::decode(&encoded).unwrap();
            assert_eq!(decoded.data_locality, locality);
        }
    }

    #[test]
    fn test_in_network_compute_decode_buffer_too_short() {
        let result = InNetworkComputeMarker::decode(&[0x00, 0x01]);
        assert!(matches!(result, Err(GtpError::BufferTooShort { .. })));
    }

    #[test]
    fn test_processing_hint_from_u8() {
        assert_eq!(ProcessingHint::from_u8(0), Some(ProcessingHint::None));
        assert_eq!(ProcessingHint::from_u8(4), Some(ProcessingHint::Inference));
        assert_eq!(ProcessingHint::from_u8(7), None);
    }

    #[test]
    fn test_data_locality_from_u8() {
        assert_eq!(DataLocality::from_u8(0), Some(DataLocality::Any));
        assert_eq!(DataLocality::from_u8(3), Some(DataLocality::Core));
        assert_eq!(DataLocality::from_u8(4), None);
    }

    // =======================================================================
    // Combined chain tests (TSN + InNetworkCompute + PDU Session)
    // =======================================================================

    #[test]
    fn test_full_6g_extension_chain() {
        let pdu_info = PduSessionInfo::downlink(30).with_rqi(true);
        let tsn = TsnMarker::new(100, 1, 2_000_000, 1);
        let inc = InNetworkComputeMarker::new(777, DataLocality::Edge, ProcessingHint::Cache);

        let chain = ExtHeaderChain::new()
            .push_pdu_session_info(pdu_info)
            .push_tsn_marker(tsn)
            .push_in_network_compute(inc);

        assert_eq!(chain.len(), 3);
        assert!(chain.contains(ExtHeaderType::PduSessionContainer));
        assert!(chain.contains(ExtHeaderType::TsnMarker));
        assert!(chain.contains(ExtHeaderType::InNetworkCompute));

        let header = chain
            .apply_to(GtpHeader::g_pdu(0xDEAD, Bytes::from_static(b"6g")))
            .unwrap();

        let encoded = header.encode();
        let decoded = GtpHeader::decode(&encoded).unwrap();

        assert_eq!(decoded.extension_header_count(), 3);
        assert_eq!(decoded.pdu_session_info().unwrap().qfi, 30);
        assert_eq!(decoded.tsn_marker().unwrap().stream_id, 100);
        assert_eq!(decoded.in_network_compute_marker().unwrap().compute_task_id, 777);
        assert_eq!(decoded.payload, Bytes::from_static(b"6g"));
    }

    #[test]
    fn test_ext_header_type_conversion_new_types() {
        assert_eq!(ExtHeaderType::from_u8(0xE1), Some(ExtHeaderType::TsnMarker));
        assert_eq!(ExtHeaderType::from_u8(0xE2), Some(ExtHeaderType::InNetworkCompute));
        assert_eq!(ExtHeaderType::from_u8(0xE3), None);
    }

    #[test]
    fn test_ext_header_chain_validate_too_long() {
        let mut chain = ExtHeaderChain::new();
        for i in 0..=GtpHeader::MAX_EXT_HEADER_CHAIN {
            chain = chain.push(GtpExtHeader::UdpPort { port: i as u16 });
        }
        assert!(chain.validate().is_err());
    }
}
