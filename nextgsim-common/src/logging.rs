//! Logging infrastructure for nextgsim
//!
//! This module provides configurable logging using the `tracing` crate,
//! protocol message logging utilities, and hex dump formatting for debugging.

use std::fmt;
use tracing::Level;
use tracing_subscriber::{fmt::format::FmtSpan, EnvFilter};

/// Log level configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LogLevel {
    /// Trace level - most verbose
    Trace,
    /// Debug level
    Debug,
    /// Info level (default)
    #[default]
    Info,
    /// Warn level
    Warn,
    /// Error level - least verbose
    Error,
}

impl From<LogLevel> for Level {
    fn from(level: LogLevel) -> Self {
        match level {
            LogLevel::Trace => Level::TRACE,
            LogLevel::Debug => Level::DEBUG,
            LogLevel::Info => Level::INFO,
            LogLevel::Warn => Level::WARN,
            LogLevel::Error => Level::ERROR,
        }
    }
}

impl fmt::Display for LogLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LogLevel::Trace => write!(f, "trace"),
            LogLevel::Debug => write!(f, "debug"),
            LogLevel::Info => write!(f, "info"),
            LogLevel::Warn => write!(f, "warn"),
            LogLevel::Error => write!(f, "error"),
        }
    }
}

impl std::str::FromStr for LogLevel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "trace" => Ok(LogLevel::Trace),
            "debug" => Ok(LogLevel::Debug),
            "info" => Ok(LogLevel::Info),
            "warn" | "warning" => Ok(LogLevel::Warn),
            "error" => Ok(LogLevel::Error),
            _ => Err(format!("unknown log level: {s}")),
        }
    }
}

/// Initialize the tracing subscriber with the specified log level.
///
/// This should be called once at application startup. The log level can be
/// overridden by the `RUST_LOG` environment variable.
///
/// # Example
///
/// ```
/// use nextgsim_common::logging::{init_logging, LogLevel};
///
/// init_logging(LogLevel::Debug);
/// ```
pub fn init_logging(level: LogLevel) {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(level.to_string()));

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_thread_ids(false)
        .with_span_events(FmtSpan::NONE)
        .init();
}

/// Initialize logging with a custom filter string.
///
/// Allows fine-grained control over which modules log at which levels.
///
/// # Example
///
/// ```
/// use nextgsim_common::logging::init_logging_with_filter;
///
/// // Set default to info, but enable debug for NAS module
/// init_logging_with_filter("info,nextgsim_nas=debug");
/// ```
pub fn init_logging_with_filter(filter: &str) {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(filter));

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_thread_ids(false)
        .with_span_events(FmtSpan::NONE)
        .init();
}

/// Protocol direction for logging
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Incoming/received message
    Rx,
    /// Outgoing/transmitted message
    Tx,
}

impl fmt::Display for Direction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Direction::Rx => write!(f, "RX"),
            Direction::Tx => write!(f, "TX"),
        }
    }
}

/// Log a protocol message at debug level with optional hex dump at trace level.
///
/// # Arguments
///
/// * `protocol` - Protocol name (e.g., "NGAP", "NAS", "RRC")
/// * `direction` - Message direction (RX or TX)
/// * `msg_type` - Message type description
/// * `data` - Raw message bytes
///
/// # Example
///
/// ```
/// use nextgsim_common::logging::{log_protocol_message, Direction};
///
/// let data = vec![0x7e, 0x00, 0x41];
/// log_protocol_message("NAS", Direction::Rx, "Registration Request", &data);
/// ```
pub fn log_protocol_message(protocol: &str, direction: Direction, msg_type: &str, data: &[u8]) {
    tracing::debug!(
        protocol = protocol,
        direction = %direction,
        msg_type = msg_type,
        len = data.len(),
        "{} {} message",
        direction,
        protocol
    );
    tracing::trace!(
        protocol = protocol,
        hex = %HexDump(data),
        "{} payload",
        protocol
    );
}

/// Log an NGAP message
pub fn log_ngap_message(direction: Direction, msg_type: &str, data: &[u8]) {
    log_protocol_message("NGAP", direction, msg_type, data);
}

/// Log a NAS message
pub fn log_nas_message(direction: Direction, msg_type: &str, data: &[u8]) {
    log_protocol_message("NAS", direction, msg_type, data);
}

/// Log an RRC message
pub fn log_rrc_message(direction: Direction, msg_type: &str, data: &[u8]) {
    log_protocol_message("RRC", direction, msg_type, data);
}

/// Log a GTP message
pub fn log_gtp_message(direction: Direction, msg_type: &str, data: &[u8]) {
    log_protocol_message("GTP", direction, msg_type, data);
}

/// Log an RLS message
pub fn log_rls_message(direction: Direction, msg_type: &str, data: &[u8]) {
    log_protocol_message("RLS", direction, msg_type, data);
}

/// Wrapper for hex dump formatting
pub struct HexDump<'a>(pub &'a [u8]);

impl fmt::Display for HexDump<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex::encode(self.0))
    }
}

/// Format bytes as a hex dump with offset, hex, and ASCII columns.
///
/// # Example
///
/// ```
/// use nextgsim_common::logging::format_hex_dump;
///
/// let data = b"Hello, World!";
/// let dump = format_hex_dump(data);
/// println!("{}", dump);
/// // Output:
/// // 00000000  48 65 6c 6c 6f 2c 20 57  6f 72 6c 64 21           |Hello, World!|
/// ```
pub fn format_hex_dump(data: &[u8]) -> String {
    if data.is_empty() {
        return String::from("(empty)");
    }

    let mut result = String::new();
    let mut offset = 0;

    for chunk in data.chunks(16) {
        // Offset column
        result.push_str(&format!("{offset:08x}  "));

        // Hex column (first 8 bytes)
        for (i, byte) in chunk.iter().enumerate() {
            if i == 8 {
                result.push(' ');
            }
            result.push_str(&format!("{byte:02x} "));
        }

        // Padding for incomplete lines
        let padding = 16 - chunk.len();
        for i in 0..padding {
            if chunk.len() + i == 8 {
                result.push(' ');
            }
            result.push_str("   ");
        }

        // ASCII column
        result.push_str(" |");
        for byte in chunk {
            if byte.is_ascii_graphic() || *byte == b' ' {
                result.push(*byte as char);
            } else {
                result.push('.');
            }
        }
        result.push('|');
        result.push('\n');

        offset += 16;
    }

    // Remove trailing newline
    result.pop();
    result
}

/// Format bytes as a compact hex string with optional grouping.
///
/// # Arguments
///
/// * `data` - Bytes to format
/// * `group_size` - Number of bytes per group (0 for no grouping)
///
/// # Example
///
/// ```
/// use nextgsim_common::logging::format_hex_compact;
///
/// let data = [0x12, 0x34, 0x56, 0x78];
/// assert_eq!(format_hex_compact(&data, 0), "12345678");
/// assert_eq!(format_hex_compact(&data, 2), "1234 5678");
/// ```
pub fn format_hex_compact(data: &[u8], group_size: usize) -> String {
    if group_size == 0 {
        return hex::encode(data);
    }

    data.chunks(group_size)
        .map(hex::encode)
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_level_from_str() {
        assert_eq!("trace".parse::<LogLevel>().unwrap(), LogLevel::Trace);
        assert_eq!("DEBUG".parse::<LogLevel>().unwrap(), LogLevel::Debug);
        assert_eq!("Info".parse::<LogLevel>().unwrap(), LogLevel::Info);
        assert_eq!("warn".parse::<LogLevel>().unwrap(), LogLevel::Warn);
        assert_eq!("warning".parse::<LogLevel>().unwrap(), LogLevel::Warn);
        assert_eq!("error".parse::<LogLevel>().unwrap(), LogLevel::Error);
        assert!("invalid".parse::<LogLevel>().is_err());
    }

    #[test]
    fn test_log_level_display() {
        assert_eq!(LogLevel::Trace.to_string(), "trace");
        assert_eq!(LogLevel::Debug.to_string(), "debug");
        assert_eq!(LogLevel::Info.to_string(), "info");
        assert_eq!(LogLevel::Warn.to_string(), "warn");
        assert_eq!(LogLevel::Error.to_string(), "error");
    }

    #[test]
    fn test_direction_display() {
        assert_eq!(Direction::Rx.to_string(), "RX");
        assert_eq!(Direction::Tx.to_string(), "TX");
    }

    #[test]
    fn test_hex_dump_empty() {
        assert_eq!(format_hex_dump(&[]), "(empty)");
    }

    #[test]
    fn test_hex_dump_short() {
        let data = b"Hi";
        let dump = format_hex_dump(data);
        assert!(dump.contains("48 69"));
        assert!(dump.contains("|Hi|"));
    }

    #[test]
    fn test_hex_dump_full_line() {
        let data: Vec<u8> = (0..16).collect();
        let dump = format_hex_dump(&data);
        assert!(dump.starts_with("00000000"));
        assert!(dump.contains("00 01 02 03 04 05 06 07  08 09 0a 0b 0c 0d 0e 0f"));
    }

    #[test]
    fn test_hex_dump_multiline() {
        let data: Vec<u8> = (0..20).collect();
        let dump = format_hex_dump(&data);
        assert!(dump.contains("00000000"));
        assert!(dump.contains("00000010"));
    }

    #[test]
    fn test_hex_compact_no_grouping() {
        let data = [0x12, 0x34, 0x56, 0x78];
        assert_eq!(format_hex_compact(&data, 0), "12345678");
    }

    #[test]
    fn test_hex_compact_with_grouping() {
        let data = [0x12, 0x34, 0x56, 0x78];
        assert_eq!(format_hex_compact(&data, 2), "1234 5678");
        assert_eq!(format_hex_compact(&data, 1), "12 34 56 78");
    }

    #[test]
    fn test_hex_dump_wrapper() {
        let data = [0xde, 0xad, 0xbe, 0xef];
        assert_eq!(format!("{}", HexDump(&data)), "deadbeef");
    }
}
