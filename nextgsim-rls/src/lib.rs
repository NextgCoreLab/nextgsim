//! Radio Link Simulation (RLS) protocol
#![allow(missing_docs)]
//!
//! This crate provides the RLS protocol implementation for simulated radio link
//! communication between UE and gNB over UDP. RLS is UERANSIM's protocol for
//! testing 5G NR without real radio hardware.
//!
//! # Protocol Overview
//!
//! RLS uses UDP for transport and supports the following message types:
//!
//! - **Heartbeat**: Sent by UE for cell search and connection maintenance
//! - **`HeartbeatAck`**: Sent by gNB in response, indicating signal strength
//! - **`PduTransmission`**: Carries RRC messages or user plane data
//! - **`PduTransmissionAck`**: Acknowledges received PDUs
//!
//! # Modules
//!
//! - [`protocol`]: RLS message types and structures
//! - [`codec`]: Message encoding and decoding
//! - [`cell_search`]: Cell discovery and tracking for UE and gNB
//! - [`transport`]: RRC and user plane data transport
//!
//! # Example
//!
//! ```rust
//! use nextgsim_rls::protocol::{RlsMessage, RlsHeartbeat, Vector3};
//! use nextgsim_rls::codec;
//!
//! // Create a heartbeat message
//! let heartbeat = RlsMessage::Heartbeat(RlsHeartbeat::with_position(
//!     12345,
//!     Vector3::new(100, 200, 0),
//! ));
//!
//! // Encode for transmission
//! let encoded = codec::encode(&heartbeat);
//!
//! // Decode received message
//! let decoded = codec::decode(&encoded).unwrap();
//! assert_eq!(heartbeat, decoded);
//! ```
//!
//! # Cell Search Example
//!
//! ```rust
//! use nextgsim_rls::cell_search::{UeCellSearch, GnbCellTracker};
//! use nextgsim_rls::protocol::Vector3;
//! use std::net::SocketAddr;
//!
//! // UE side: create cell search manager
//! let search_space: Vec<SocketAddr> = vec!["127.0.0.1:4997".parse().unwrap()];
//! let mut ue_search = UeCellSearch::new(12345, search_space);
//!
//! // Create heartbeats to send
//! let heartbeats = ue_search.create_heartbeats();
//!
//! // gNB side: create cell tracker
//! let mut gnb_tracker = GnbCellTracker::new(67890, Vector3::new(0, 0, 0));
//! ```

pub mod cell_search;
pub mod codec;
pub mod protocol;
pub mod transport;

// Re-export commonly used types from protocol
pub use codec::{decode, encode, RlsCodecError};
pub use protocol::{
    MessageType, PduInfo, PduType, RlfCause, RlsHeartbeat, RlsHeartbeatAck, RlsMessage,
    RlsPduTransmission, RlsPduTransmissionAck, Vector3,
};

// Re-export cell search types
pub use cell_search::{
    CellInfo, CellSearchEvent, GnbCellTracker, GnbTrackerEvent, UeCellSearch, UeInfo,
    DEFAULT_HEARTBEAT_INTERVAL_MS, DEFAULT_HEARTBEAT_THRESHOLD_MS, MIN_ALLOWED_DBM,
};

// Re-export transport types
pub use transport::{RlsTransport, RrcChannel, TransportEvent, MAX_PDU_COUNT, MAX_PDU_TTL_MS};
