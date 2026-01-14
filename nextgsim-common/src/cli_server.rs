//! CLI server for accepting connections from the CLI tool
//!
//! Running gNB and UE instances use this server to accept commands
//! from the CLI tool. The server listens on a UDP port and processes
//! incoming command messages.
//!
//! # Reference
//!
//! Based on UERANSIM's `src/lib/app/cli_base.cpp` implementation.

use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokio::net::UdpSocket as TokioUdpSocket;

/// Directory where process table entries are stored
pub const PROC_TABLE_DIR: &str = "/tmp/nextgsim.proc-table/";

/// Default command server IP (localhost)
pub const CMD_SERVER_IP: &str = "127.0.0.1";

/// Version information for compatibility checking - major version
pub const VERSION_MAJOR: u8 = 1;
/// Version information for compatibility checking - minor version
pub const VERSION_MINOR: u8 = 0;
/// Version information for compatibility checking - patch version
pub const VERSION_PATCH: u8 = 0;

/// Maximum buffer size for receiving messages
const CMD_BUFFER_SIZE: usize = 8192;

/// CLI message types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CliMessageType {
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

impl TryFrom<u8> for CliMessageType {
    type Error = ();

    fn try_from(value: u8) -> std::result::Result<Self, ()> {
        match value {
            0 => Ok(CliMessageType::Empty),
            1 => Ok(CliMessageType::Echo),
            2 => Ok(CliMessageType::Error),
            3 => Ok(CliMessageType::Result),
            4 => Ok(CliMessageType::Command),
            _ => Err(()),
        }
    }
}

/// A CLI message for communication between CLI and running instances
#[derive(Debug, Clone)]
pub struct CliMessage {
    /// Message type
    pub msg_type: CliMessageType,
    /// Node name (target or source)
    pub node_name: String,
    /// Message value (command or response)
    pub value: String,
    /// Client address (for responses)
    pub client_addr: SocketAddr,
}

impl CliMessage {
    /// Creates a new error response
    pub fn error(addr: SocketAddr, node_name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            msg_type: CliMessageType::Error,
            node_name: node_name.into(),
            value: message.into(),
            client_addr: addr,
        }
    }

    /// Creates a new result response
    pub fn result(addr: SocketAddr, node_name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            msg_type: CliMessageType::Result,
            node_name: node_name.into(),
            value: message.into(),
            client_addr: addr,
        }
    }

    /// Creates a new echo response
    pub fn echo(addr: SocketAddr, message: impl Into<String>) -> Self {
        Self {
            msg_type: CliMessageType::Echo,
            node_name: String::new(),
            value: message.into(),
            client_addr: addr,
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
    pub fn decode(data: &[u8], client_addr: SocketAddr) -> Option<Self> {
        if data.len() < 12 {
            return None;
        }

        // Check version
        let major = data[0];
        let minor = data[1];
        let patch = data[2];

        if major != VERSION_MAJOR || minor != VERSION_MINOR || patch != VERSION_PATCH {
            return None;
        }

        // Message type
        let msg_type = CliMessageType::try_from(data[3]).ok()?;

        // Node name
        let node_len = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
        if data.len() < 12 + node_len {
            return None;
        }
        let node_name = String::from_utf8(data[8..8 + node_len].to_vec()).ok()?;

        // Value
        let value_offset = 8 + node_len;
        if data.len() < value_offset + 4 {
            return None;
        }
        let value_len = u32::from_be_bytes([
            data[value_offset],
            data[value_offset + 1],
            data[value_offset + 2],
            data[value_offset + 3],
        ]) as usize;

        let value_data_offset = value_offset + 4;
        if data.len() < value_data_offset + value_len {
            return None;
        }
        let value = String::from_utf8(data[value_data_offset..value_data_offset + value_len].to_vec()).ok()?;

        Some(Self {
            msg_type,
            node_name,
            value,
            client_addr,
        })
    }
}

/// A CLI command received from the CLI tool
#[derive(Debug, Clone)]
pub struct CliCommand {
    /// The command string
    pub command: String,
    /// The node name the command is for
    pub node_name: String,
    /// The client address to send responses to
    pub client_addr: SocketAddr,
}

/// A CLI response to send back to the CLI tool
#[derive(Debug, Clone)]
pub struct CliResponse {
    /// The response message
    pub message: String,
    /// Whether this is an error response
    pub is_error: bool,
    /// The client address to send to
    pub client_addr: SocketAddr,
}

/// Process table entry for registering running instances
#[derive(Debug, Clone)]
pub struct ProcTableEntry {
    /// Major version number
    pub major: u8,
    /// Minor version number
    pub minor: u8,
    /// Patch version number
    pub patch: u8,
    /// Process ID
    pub pid: u32,
    /// Command port for CLI communication
    pub port: u16,
    /// Node names registered by this process
    pub nodes: Vec<String>,
}

impl ProcTableEntry {
    /// Encodes a process table entry to a string
    pub fn encode(&self) -> String {
        let mut s = format!(
            "{} {} {} {} {} {}",
            self.major,
            self.minor,
            self.patch,
            self.pid,
            self.port,
            self.nodes.len()
        );
        for node in &self.nodes {
            s.push(' ');
            s.push_str(node);
        }
        s
    }
}

/// CLI server for accepting commands from the CLI tool
pub struct CliServer {
    /// UDP socket for communication
    socket: Arc<TokioUdpSocket>,
    /// Local address the server is bound to
    local_addr: SocketAddr,
    /// Process table file path (for cleanup)
    proc_table_path: Option<PathBuf>,
    /// Node names registered with this server
    node_names: Vec<String>,
}

impl CliServer {
    /// Creates a new CLI server bound to localhost on a random port
    pub async fn new() -> std::io::Result<Self> {
        let socket = TokioUdpSocket::bind(format!("{}:0", CMD_SERVER_IP)).await?;
        let local_addr = socket.local_addr()?;

        Ok(Self {
            socket: Arc::new(socket),
            local_addr,
            proc_table_path: None,
            node_names: Vec::new(),
        })
    }

    /// Creates a new CLI server bound to the specified port
    pub async fn with_port(port: u16) -> std::io::Result<Self> {
        let socket = TokioUdpSocket::bind(format!("{}:{}", CMD_SERVER_IP, port)).await?;
        let local_addr = socket.local_addr()?;

        Ok(Self {
            socket: Arc::new(socket),
            local_addr,
            proc_table_path: None,
            node_names: Vec::new(),
        })
    }

    /// Returns the port the server is listening on
    pub fn port(&self) -> u16 {
        self.local_addr.port()
    }

    /// Returns the local address the server is bound to
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Registers node names in the process table
    ///
    /// This creates a process table entry so the CLI can discover this instance.
    pub fn register_nodes(&mut self, nodes: Vec<String>) -> std::io::Result<()> {
        self.node_names = nodes.clone();

        // Create process table directory
        fs::create_dir_all(PROC_TABLE_DIR)?;

        // Generate unique filename
        let pid = std::process::id();
        let filename = format!("{:016x}", {
            let mut hasher = DefaultHasher::new();
            self.node_names.hash(&mut hasher);
            self.port().hash(&mut hasher);
            pid.hash(&mut hasher);
            hasher.finish()
        });

        let file_path = Path::new(PROC_TABLE_DIR).join(filename);

        // Create entry
        let entry = ProcTableEntry {
            major: VERSION_MAJOR,
            minor: VERSION_MINOR,
            patch: VERSION_PATCH,
            pid,
            port: self.port(),
            nodes,
        };

        // Write entry
        fs::write(&file_path, entry.encode())?;

        self.proc_table_path = Some(file_path);

        Ok(())
    }

    /// Receives a command from the CLI tool
    ///
    /// Returns `None` if the message is invalid or not a command.
    pub async fn receive_command(&self) -> std::io::Result<Option<CliCommand>> {
        let mut buffer = [0u8; CMD_BUFFER_SIZE];

        let (size, addr) = self.socket.recv_from(&mut buffer).await?;

        if size == 0 {
            return Ok(None);
        }

        let msg = match CliMessage::decode(&buffer[..size], addr) {
            Some(m) => m,
            None => return Ok(None),
        };

        if msg.msg_type != CliMessageType::Command {
            return Ok(None);
        }

        // Check if this command is for one of our nodes
        if !self.node_names.is_empty() && !self.node_names.contains(&msg.node_name) {
            // Not for us, ignore
            return Ok(None);
        }

        Ok(Some(CliCommand {
            command: msg.value,
            node_name: msg.node_name,
            client_addr: msg.client_addr,
        }))
    }

    /// Sends a response to the CLI tool
    pub async fn send_response(&self, response: CliResponse) -> std::io::Result<()> {
        let msg = if response.is_error {
            CliMessage::error(response.client_addr, "", &response.message)
        } else {
            CliMessage::result(response.client_addr, "", &response.message)
        };

        let data = msg.encode();
        self.socket.send_to(&data, response.client_addr).await?;

        Ok(())
    }

    /// Sends an error response to the CLI tool
    pub async fn send_error(&self, addr: SocketAddr, message: impl Into<String>) -> std::io::Result<()> {
        self.send_response(CliResponse {
            message: message.into(),
            is_error: true,
            client_addr: addr,
        }).await
    }

    /// Sends a result response to the CLI tool
    pub async fn send_result(&self, addr: SocketAddr, message: impl Into<String>) -> std::io::Result<()> {
        self.send_response(CliResponse {
            message: message.into(),
            is_error: false,
            client_addr: addr,
        }).await
    }

    /// Sends an echo message to the CLI tool
    pub async fn send_echo(&self, addr: SocketAddr, message: impl Into<String>) -> std::io::Result<()> {
        let msg = CliMessage::echo(addr, message);
        let data = msg.encode();
        self.socket.send_to(&data, addr).await?;
        Ok(())
    }

    /// Returns a clone of the socket for use in async tasks
    pub fn socket(&self) -> Arc<TokioUdpSocket> {
        Arc::clone(&self.socket)
    }
}

impl Drop for CliServer {
    fn drop(&mut self) {
        // Clean up process table entry
        if let Some(path) = &self.proc_table_path {
            let _ = fs::remove_file(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_message_encode_decode() {
        let addr: SocketAddr = "127.0.0.1:5000".parse().unwrap();
        let msg = CliMessage::result(addr, "node1", "test result");
        let encoded = msg.encode();
        let decoded = CliMessage::decode(&encoded, addr).unwrap();

        assert_eq!(decoded.msg_type, CliMessageType::Result);
        assert_eq!(decoded.node_name, "node1");
        assert_eq!(decoded.value, "test result");
    }

    #[test]
    fn test_proc_table_entry_encode() {
        let entry = ProcTableEntry {
            major: 1,
            minor: 0,
            patch: 0,
            pid: 12345,
            port: 5000,
            nodes: vec!["gnb1".to_string(), "gnb2".to_string()],
        };

        let encoded = entry.encode();
        assert!(encoded.contains("1 0 0 12345 5000 2 gnb1 gnb2"));
    }

    #[tokio::test]
    async fn test_cli_server_creation() {
        let server = CliServer::new().await.unwrap();
        assert!(server.port() > 0);
    }
}
