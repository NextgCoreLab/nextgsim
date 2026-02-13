//! GTP-U (GPRS Tunneling Protocol - User Plane) library
//!
//! Implements GTP-U header encoding/decoding and tunnel management for user plane
//! tunneling according to 3GPP TS 29.281.
//!
//! # Example
//!
//! ```
//! use nextgsim_gtp::codec::{GtpHeader, GtpMessageType, GtpExtHeader};
//! use nextgsim_gtp::tunnel::{TunnelManager, PduSession, GtpTunnel, GTP_U_PORT};
//! use bytes::Bytes;
//! use std::net::{SocketAddr, IpAddr, Ipv4Addr};
//!
//! // Create a G-PDU message
//! let header = GtpHeader::g_pdu(0x12345678, Bytes::from_static(b"user data"))
//!     .with_sequence_number(1);
//!
//! // Encode to bytes
//! let encoded = header.encode();
//!
//! // Decode from bytes
//! let decoded = GtpHeader::decode(&encoded).unwrap();
//! assert_eq!(decoded.teid, 0x12345678);
//!
//! // Tunnel management
//! let mut manager = TunnelManager::new();
//! let upf_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)), GTP_U_PORT);
//! let gnb_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 2)), GTP_U_PORT);
//!
//! let session = PduSession::new(
//!     1,  // UE ID
//!     5,  // PSI
//!     GtpTunnel::new(0x1000, upf_addr),   // Uplink tunnel
//!     GtpTunnel::new(0x2000, gnb_addr),   // Downlink tunnel
//! );
//! manager.create_session(session).unwrap();
//! ```

pub mod codec;
pub mod qos;
pub mod tunnel;

// Re-export main types for convenience
pub use codec::{
    ExtHeaderType, GtpError, GtpExtHeader, GtpHeader, GtpMessageType,
    GTP_PROTOCOL_TYPE, GTP_VERSION,
    // Extension header chaining (A5.1)
    ExtHeaderChain,
    // PDU Session Container IE (A5.2)
    PduSessionInfo, PduSessionType,
    // 6G TSN markers (A5.3)
    TsnMarker, EXT_HEADER_TYPE_TSN_MARKER,
    // 6G In-Network Compute markers (A5.4)
    InNetworkComputeMarker, ProcessingHint, DataLocality, EXT_HEADER_TYPE_IN_NETWORK_COMPUTE,
};

pub use tunnel::{
    GtpTunnel, PduSession, TunnelError, TunnelManager,
    GTP_U_PORT, get_psi, get_ue_id, make_session_key,
};

pub use qos::{
    FiveQiCharacteristics, QosResourceType, QfiDscpMapper, QosFlowEnforcer,
    FlowStats, TokenBucket, lookup_5qi, default_qfi_to_dscp, standard_5qi_table,
};
