//! Common types and utilities for nextgsim
//!
//! This crate provides shared types, configuration structures, and utilities
//! used across all nextgsim crates.

pub mod bit_buffer;
pub mod bit_string;
pub mod cli_server;
pub mod config;
pub mod error;
pub mod logging;
pub mod octet;
pub mod octet_string;
pub mod octet_view;
pub mod transport;
pub mod types;

pub use bit_buffer::{BitBuffer, BitBufferReader};
pub use bit_string::BitString;
pub use cli_server::{
    CliCommand, CliMessage, CliMessageType, CliResponse, CliServer, ProcTableEntry,
    CMD_SERVER_IP, PROC_TABLE_DIR, VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH,
};
pub use config::{
    AmfConfig, GnbConfig, OpType, PduSessionType, SessionConfig, SupportedAlgs, UeConfig,
};
pub use error::Error;
pub use logging::{
    init_logging, init_logging_with_filter, format_hex_compact, format_hex_dump,
    log_gtp_message, log_nas_message, log_ngap_message, log_protocol_message, log_rls_message,
    log_rrc_message, Direction, HexDump, LogLevel,
};
pub use octet_string::OctetString;
pub use octet_view::OctetView;
pub use transport::UdpTransport;
pub use types::*;
