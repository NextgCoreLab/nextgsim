//! CLI message protocol for communication with running instances
//!
//! The CLI communicates with running gNB/UE instances via UDP messages.
//! Messages are encoded with version information for compatibility checking.
//!
//! # Message Format
//!
//! ```text
//! +-------+-------+-------+------+------------+-----------+------------+---------+
//! | Major | Minor | Patch | Type | NodeLen(4) | NodeName  | ValueLen(4)| Value   |
//! +-------+-------+-------+------+------------+-----------+------------+---------+
//! | 1     | 1     | 1     | 1    | 4          | variable  | 4          | variable|
//! +-------+-------+-------+------+------------+-----------+------------+---------+
//! ```
//!
//! # Reference
//!
//! Based on UERANSIM's `src/lib/app/cli_base.cpp` implementation.

use std::net::SocketAddr;

use anyhow::{Context, Result};

use crate::proc_table::{VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH};

/// CLI message types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MessageType {
    /// Empty/invalid message
    Empty = 0,
    /// Echo message (informational output)
    Echo = 1,
    /// Error message
    Error = 2,
    /// Result message (command output)
    Result = 3,
    /// Command message (from CLI to instance)
    Command = 4,
}

impl TryFrom<u8> for MessageType {
    type Error = anyhow::Error;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            0 => Ok(MessageType::Empty),
            1 => Ok(MessageType::Echo),
            2 => Ok(MessageType::Error),
            3 => Ok(MessageType::Result),
            4 => Ok(MessageType::Command),
            _ => anyhow::bail!("Invalid message type: {}", value),
        }
    }
}

/// A CLI message for communication between CLI and running instances
#[derive(Debug, Clone)]
pub struct CliMessage {
    /// Message type
    pub msg_type: MessageType,
    /// Node name (target or source)
    pub node_name: String,
    /// Message value (command or response)
    pub value: String,
    /// Client address (for responses)
    #[allow(dead_code)]
    pub client_addr: Option<SocketAddr>,
}

impl CliMessage {
    /// Creates a new command message
    pub fn command(node_name: impl Into<String>, command: impl Into<String>) -> Self {
        Self {
            msg_type: MessageType::Command,
            node_name: node_name.into(),
            value: command.into(),
            client_addr: None,
        }
    }

    /// Creates a new error message
    #[allow(dead_code)]
    pub fn error(node_name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            msg_type: MessageType::Error,
            node_name: node_name.into(),
            value: message.into(),
            client_addr: None,
        }
    }

    /// Creates a new result message
    #[allow(dead_code)]
    pub fn result(node_name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            msg_type: MessageType::Result,
            node_name: node_name.into(),
            value: message.into(),
            client_addr: None,
        }
    }

    /// Creates a new echo message
    #[allow(dead_code)]
    pub fn echo(message: impl Into<String>) -> Self {
        Self {
            msg_type: MessageType::Echo,
            node_name: String::new(),
            value: message.into(),
            client_addr: None,
        }
    }

    /// Encodes the message to bytes
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(12 + self.node_name.len() + self.value.len());

        // Version header
        buf.push(VERSION_MAJOR);
        buf.push(VERSION_MINOR);
        buf.push(VERSION_PATCH);

        // Message type
        buf.push(self.msg_type as u8);

        // Node name (4-byte length + data)
        let node_bytes = self.node_name.as_bytes();
        buf.extend_from_slice(&(node_bytes.len() as u32).to_be_bytes());
        buf.extend_from_slice(node_bytes);

        // Value (4-byte length + data)
        let value_bytes = self.value.as_bytes();
        buf.extend_from_slice(&(value_bytes.len() as u32).to_be_bytes());
        buf.extend_from_slice(value_bytes);

        buf
    }

    /// Decodes a message from bytes
    pub fn decode(data: &[u8]) -> Result<Self> {
        if data.len() < 12 {
            anyhow::bail!("Message too short: {} bytes", data.len());
        }

        // Check version
        let major = data[0];
        let minor = data[1];
        let patch = data[2];

        if major != VERSION_MAJOR || minor != VERSION_MINOR || patch != VERSION_PATCH {
            anyhow::bail!(
                "Version mismatch: got {}.{}.{}, expected {}.{}.{}",
                major,
                minor,
                patch,
                VERSION_MAJOR,
                VERSION_MINOR,
                VERSION_PATCH
            );
        }

        // Message type
        let msg_type = MessageType::try_from(data[3])?;

        // Node name
        let node_len = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
        if data.len() < 12 + node_len {
            anyhow::bail!("Message truncated: missing node name");
        }
        let node_name = String::from_utf8(data[8..8 + node_len].to_vec())
            .context("Invalid UTF-8 in node name")?;

        // Value
        let value_offset = 8 + node_len;
        if data.len() < value_offset + 4 {
            anyhow::bail!("Message truncated: missing value length");
        }
        let value_len = u32::from_be_bytes([
            data[value_offset],
            data[value_offset + 1],
            data[value_offset + 2],
            data[value_offset + 3],
        ]) as usize;

        let value_data_offset = value_offset + 4;
        if data.len() < value_data_offset + value_len {
            anyhow::bail!("Message truncated: missing value data");
        }
        let value = String::from_utf8(data[value_data_offset..value_data_offset + value_len].to_vec())
            .context("Invalid UTF-8 in value")?;

        Ok(Self {
            msg_type,
            node_name,
            value,
            client_addr: None,
        })
    }

    /// Returns true if this is an error message
    #[allow(dead_code)]
    pub fn is_error(&self) -> bool {
        self.msg_type == MessageType::Error
    }

    /// Returns true if this is a result message
    #[allow(dead_code)]
    pub fn is_result(&self) -> bool {
        self.msg_type == MessageType::Result
    }

    /// Returns true if this is an echo message
    #[allow(dead_code)]
    pub fn is_echo(&self) -> bool {
        self.msg_type == MessageType::Echo
    }

    /// Returns true if this is a command message
    #[allow(dead_code)]
    pub fn is_command(&self) -> bool {
        self.msg_type == MessageType::Command
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_encode_decode() {
        let msg = CliMessage::command("gnb1", "status");
        let encoded = msg.encode();
        let decoded = CliMessage::decode(&encoded).unwrap();

        assert_eq!(decoded.msg_type, MessageType::Command);
        assert_eq!(decoded.node_name, "gnb1");
        assert_eq!(decoded.value, "status");
    }

    #[test]
    fn test_message_types() {
        let error = CliMessage::error("node", "error message");
        assert!(error.is_error());

        let result = CliMessage::result("node", "result message");
        assert!(result.is_result());

        let echo = CliMessage::echo("echo message");
        assert!(echo.is_echo());

        let command = CliMessage::command("node", "command");
        assert!(command.is_command());
    }

    #[test]
    fn test_decode_invalid() {
        // Too short
        assert!(CliMessage::decode(&[]).is_err());
        assert!(CliMessage::decode(&[1, 0, 0]).is_err());

        // Invalid version
        let mut msg = CliMessage::command("n", "c").encode();
        msg[0] = 99; // Invalid major version
        assert!(CliMessage::decode(&msg).is_err());
    }
}
