//! SCTP transport layer for nextgsim
#![allow(missing_docs)]
//!
//! This crate provides SCTP association management for NGAP transport,
//! wrapping the `sctp-proto` crate with an async tokio-based interface.
//!
//! # Overview
//!
//! SCTP (Stream Control Transmission Protocol) is used in 5G networks for
//! NGAP (NG Application Protocol) communication between gNB and AMF.
//!
//! This implementation uses `sctp-proto`'s Sans-IO design pattern, which
//! separates protocol logic from I/O operations. The actual network I/O
//! is performed using tokio UDP sockets.
//!
//! # Compatibility
//!
//! This library is compatible with nextgcore's SCTP implementation, which
//! also uses `sctp-proto`. Both use SCTP-over-UDP, ensuring wire compatibility.
//!
//! # Features
//!
//! - Async/await interface using tokio
//! - Multi-stream support for NGAP
//! - NGAP PPID (60) handling
//! - Graceful shutdown support
//! - Event-driven architecture
//! - Server mode support (for accepting connections)
//!
//! # Example (Client - gNB connecting to AMF)
//!
//! ```rust,no_run
//! use nextgsim_sctp::{SctpAssociation, SctpConfig};
//! use std::net::SocketAddr;
//!
//! async fn connect_to_amf() -> Result<(), Box<dyn std::error::Error>> {
//!     let amf_addr: SocketAddr = "192.168.1.1:38412".parse()?;
//!     let config = SctpConfig::default();
//!
//!     let mut assoc = SctpAssociation::connect(amf_addr, config).await?;
//!
//!     // Send NGAP message on stream 0
//!     assoc.send(0, b"NGAP message").await?;
//!
//!     // Receive response
//!     if let Some(msg) = assoc.recv().await? {
//!         println!("Received on stream {}: {:?}", msg.stream_id, msg.data);
//!     }
//!
//!     // Graceful shutdown
//!     assoc.shutdown().await?;
//!
//!     Ok(())
//! }
//! ```

pub mod association;
pub mod server;

// Re-export main types
pub use association::{
    AssociationState, ReceivedMessage, SctpAssociation, SctpConfig, SctpError, SctpEvent,
    DEFAULT_MAX_MESSAGE_SIZE, DEFAULT_NUM_STREAMS, DEFAULT_RECEIVE_BUFFER_SIZE, NGAP_PPID,
};

// Re-export server types
pub use server::{SctpServer, SctpServerConfig, ServerEvent};
