//! CLI Server for gNB Application
//!
//! This module implements the UDP-based IPC server for CLI communication.
//! The CLI tool connects to this server to send commands and receive responses.
//!
//! # Protocol
//!
//! The CLI protocol uses UDP messages with the following format:
//! - Version: 3 bytes (major, minor, patch)
//! - Type: 1 byte (Command, Result, Error, Echo)
//! - Node name length: 4 bytes (big-endian)
//! - Node name: variable length UTF-8 string
//! - Value length: 4 bytes (big-endian)
//! - Value: variable length UTF-8 string
//!
//! # Reference
//!
//! Based on UERANSIM's `src/lib/app/cli_base.cpp` implementation.

use std::net::SocketAddr;
use tokio::net::UdpSocket;

/// CLI protocol version.
pub const CLI_VERSION_MAJOR: u8 = 3;
pub const CLI_VERSION_MINOR: u8 = 2;
pub const CLI_VERSION_PATCH: u8 = 7;

/// Maximum CLI message buffer size.
pub const CLI_BUFFER_SIZE: usize = 8192;

/// Minimum CLI message length (version + type + node name length + value length).
pub const CLI_MIN_LENGTH: usize = 3 + 1 + 4 + 4;

/// CLI receive timeout in milliseconds.
pub const CLI_RECV_TIMEOUT_MS: u64 = 2500;

/// CLI message types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CliMessageType {
    /// Empty/invalid message
    Empty = 0,
    /// Echo message (for testing)
    Echo = 1,
    /// Error response
    Error = 2,
    /// Success result
    Result = 3,
    /// Command request
    Command = 4,
}

impl From<u8> for CliMessageType {
    fn from(value: u8) -> Self {
        match value {
            1 => CliMessageType::Echo,
            2 => CliMessageType::Error,
            3 => CliMessageType::Result,
            4 => CliMessageType::Command,
            _ => CliMessageType::Empty,
        }
    }
}

/// CLI message structure.
#[derive(Debug, Clone)]
pub struct CliMessage {
    /// Message type
    pub msg_type: CliMessageType,
    /// Node name (gNB or UE identifier)
    pub node_name: String,
    /// Message value (command or response)
    pub value: String,
    /// Client address (for responses)
    pub client_addr: Option<SocketAddr>,
}

impl CliMessage {
    /// Creates a new command message.
    pub fn command(node_name: String, value: String, client_addr: SocketAddr) -> Self {
        Self {
            msg_type: CliMessageType::Command,
            node_name,
            value,
            client_addr: Some(client_addr),
        }
    }

    /// Creates a result response.
    pub fn result(node_name: String, value: String, client_addr: SocketAddr) -> Self {
        Self {
            msg_type: CliMessageType::Result,
            node_name,
            value,
            client_addr: Some(client_addr),
        }
    }

    /// Creates an error response.
    pub fn error(node_name: String, value: String, client_addr: SocketAddr) -> Self {
        Self {
            msg_type: CliMessageType::Error,
            node_name,
            value,
            client_addr: Some(client_addr),
        }
    }

    /// Creates an echo response.
    pub fn echo(value: String, client_addr: SocketAddr) -> Self {
        Self {
            msg_type: CliMessageType::Echo,
            node_name: String::new(),
            value,
            client_addr: Some(client_addr),
        }
    }

    /// Encodes the message to bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut buffer = Vec::with_capacity(CLI_MIN_LENGTH + self.node_name.len() + self.value.len());

        // Version
        buffer.push(CLI_VERSION_MAJOR);
        buffer.push(CLI_VERSION_MINOR);
        buffer.push(CLI_VERSION_PATCH);

        // Type
        buffer.push(self.msg_type as u8);

        // Node name length and value
        let node_name_bytes = self.node_name.as_bytes();
        buffer.extend_from_slice(&(node_name_bytes.len() as u32).to_be_bytes());
        buffer.extend_from_slice(node_name_bytes);

        // Value length and value
        let value_bytes = self.value.as_bytes();
        buffer.extend_from_slice(&(value_bytes.len() as u32).to_be_bytes());
        buffer.extend_from_slice(value_bytes);

        buffer
    }

    /// Decodes a message from bytes.
    pub fn decode(data: &[u8], client_addr: SocketAddr) -> Option<Self> {
        if data.len() < CLI_MIN_LENGTH {
            return None;
        }

        // Check version
        if data[0] != CLI_VERSION_MAJOR
            || data[1] != CLI_VERSION_MINOR
            || data[2] != CLI_VERSION_PATCH
        {
            return None;
        }

        let msg_type = CliMessageType::from(data[3]);
        if msg_type == CliMessageType::Empty {
            return None;
        }

        // Parse node name
        let node_name_len = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
        if data.len() < CLI_MIN_LENGTH + node_name_len {
            return None;
        }
        let node_name = String::from_utf8_lossy(&data[8..8 + node_name_len]).to_string();

        // Parse value
        let value_offset = 8 + node_name_len;
        if data.len() < value_offset + 4 {
            return None;
        }
        let value_len = u32::from_be_bytes([
            data[value_offset],
            data[value_offset + 1],
            data[value_offset + 2],
            data[value_offset + 3],
        ]) as usize;

        let value_start = value_offset + 4;
        if data.len() < value_start + value_len {
            return None;
        }
        let value = String::from_utf8_lossy(&data[value_start..value_start + value_len]).to_string();

        Some(Self {
            msg_type,
            node_name,
            value,
            client_addr: Some(client_addr),
        })
    }
}

/// CLI server error types.
#[derive(Debug, thiserror::Error)]
pub enum CliServerError {
    /// I/O error
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Receive timeout
    #[error("Receive timeout")]
    Timeout,

    /// Invalid message
    #[error("Invalid message")]
    InvalidMessage,
}

/// CLI server for receiving commands and sending responses.
pub struct CliServer {
    /// UDP socket for communication
    socket: UdpSocket,
    /// Node name for this server
    node_name: String,
}

impl CliServer {
    /// Creates a new CLI server bound to the specified address.
    ///
    /// # Arguments
    ///
    /// * `bind_addr` - Address to bind the server to (use port 0 for auto-assign)
    /// * `node_name` - Name of this node (e.g., "UERANSIM-gnb-310-410-1")
    pub async fn new(bind_addr: SocketAddr, node_name: String) -> Result<Self, CliServerError> {
        let socket = UdpSocket::bind(bind_addr).await?;
        Ok(Self { socket, node_name })
    }

    /// Returns the local address the server is bound to.
    pub fn local_addr(&self) -> Result<SocketAddr, CliServerError> {
        Ok(self.socket.local_addr()?)
    }

    /// Returns the node name.
    pub fn node_name(&self) -> &str {
        &self.node_name
    }

    /// Receives a CLI message with timeout.
    ///
    /// # Returns
    ///
    /// * `Ok(CliMessage)` - Successfully received and decoded message
    /// * `Err(CliServerError)` - Receive failed or message invalid
    pub async fn receive(&self) -> Result<CliMessage, CliServerError> {
        let mut buffer = [0u8; CLI_BUFFER_SIZE];

        let timeout = tokio::time::Duration::from_millis(CLI_RECV_TIMEOUT_MS);
        let result = tokio::time::timeout(timeout, self.socket.recv_from(&mut buffer)).await;

        match result {
            Ok(Ok((size, addr))) => {
                if size < CLI_MIN_LENGTH || size >= CLI_BUFFER_SIZE {
                    return Err(CliServerError::InvalidMessage);
                }
                CliMessage::decode(&buffer[..size], addr).ok_or(CliServerError::InvalidMessage)
            }
            Ok(Err(e)) => Err(CliServerError::IoError(e)),
            Err(_) => Err(CliServerError::Timeout),
        }
    }

    /// Tries to receive a CLI message without blocking.
    ///
    /// # Returns
    ///
    /// * `Ok(Some(CliMessage))` - Successfully received message
    /// * `Ok(None)` - No message available
    /// * `Err(CliServerError)` - Receive failed
    pub fn try_receive(&self) -> Result<Option<CliMessage>, CliServerError> {
        let mut buffer = [0u8; CLI_BUFFER_SIZE];

        match self.socket.try_recv_from(&mut buffer) {
            Ok((size, addr)) => {
                if size < CLI_MIN_LENGTH || size >= CLI_BUFFER_SIZE {
                    return Err(CliServerError::InvalidMessage);
                }
                Ok(CliMessage::decode(&buffer[..size], addr))
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => Ok(None),
            Err(e) => Err(CliServerError::IoError(e)),
        }
    }

    /// Sends a CLI message to the specified address.
    pub async fn send(&self, msg: &CliMessage) -> Result<(), CliServerError> {
        let addr = msg.client_addr.ok_or_else(|| {
            CliServerError::IoError(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "No destination address",
            ))
        })?;

        let data = msg.encode();
        self.socket.send_to(&data, addr).await?;
        Ok(())
    }

    /// Sends a result response.
    pub async fn send_result(
        &self,
        value: String,
        client_addr: SocketAddr,
    ) -> Result<(), CliServerError> {
        let msg = CliMessage::result(self.node_name.clone(), value, client_addr);
        self.send(&msg).await
    }

    /// Sends an error response.
    pub async fn send_error(
        &self,
        value: String,
        client_addr: SocketAddr,
    ) -> Result<(), CliServerError> {
        let msg = CliMessage::error(self.node_name.clone(), value, client_addr);
        self.send(&msg).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    fn test_addr() -> SocketAddr {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 12345)
    }

    #[test]
    fn test_cli_message_type_from_u8() {
        assert_eq!(CliMessageType::from(0), CliMessageType::Empty);
        assert_eq!(CliMessageType::from(1), CliMessageType::Echo);
        assert_eq!(CliMessageType::from(2), CliMessageType::Error);
        assert_eq!(CliMessageType::from(3), CliMessageType::Result);
        assert_eq!(CliMessageType::from(4), CliMessageType::Command);
        assert_eq!(CliMessageType::from(255), CliMessageType::Empty);
    }

    #[test]
    fn test_cli_message_encode_decode_roundtrip() {
        let original = CliMessage::command(
            "test-gnb".to_string(),
            "status".to_string(),
            test_addr(),
        );

        let encoded = original.encode();
        let decoded = CliMessage::decode(&encoded, test_addr()).unwrap();

        assert_eq!(decoded.msg_type, CliMessageType::Command);
        assert_eq!(decoded.node_name, "test-gnb");
        assert_eq!(decoded.value, "status");
    }

    #[test]
    fn test_cli_message_encode_result() {
        let msg = CliMessage::result(
            "gnb-1".to_string(),
            "ngap_up: true".to_string(),
            test_addr(),
        );

        let encoded = msg.encode();
        let decoded = CliMessage::decode(&encoded, test_addr()).unwrap();

        assert_eq!(decoded.msg_type, CliMessageType::Result);
        assert_eq!(decoded.node_name, "gnb-1");
        assert_eq!(decoded.value, "ngap_up: true");
    }

    #[test]
    fn test_cli_message_encode_error() {
        let msg = CliMessage::error(
            "gnb-1".to_string(),
            "UE not found".to_string(),
            test_addr(),
        );

        let encoded = msg.encode();
        let decoded = CliMessage::decode(&encoded, test_addr()).unwrap();

        assert_eq!(decoded.msg_type, CliMessageType::Error);
        assert_eq!(decoded.value, "UE not found");
    }

    #[test]
    fn test_cli_message_decode_too_short() {
        let data = [0u8; 5]; // Too short
        assert!(CliMessage::decode(&data, test_addr()).is_none());
    }

    #[test]
    fn test_cli_message_decode_wrong_version() {
        let msg = CliMessage::command("test".to_string(), "cmd".to_string(), test_addr());
        let mut encoded = msg.encode();
        encoded[0] = 99; // Wrong major version
        assert!(CliMessage::decode(&encoded, test_addr()).is_none());
    }

    #[test]
    fn test_cli_message_decode_invalid_type() {
        let msg = CliMessage::command("test".to_string(), "cmd".to_string(), test_addr());
        let mut encoded = msg.encode();
        encoded[3] = 0; // Empty type
        assert!(CliMessage::decode(&encoded, test_addr()).is_none());
    }

    #[test]
    fn test_cli_message_echo() {
        let msg = CliMessage::echo("hello".to_string(), test_addr());
        assert_eq!(msg.msg_type, CliMessageType::Echo);
        assert!(msg.node_name.is_empty());
        assert_eq!(msg.value, "hello");
    }

    #[test]
    fn test_cli_message_empty_strings() {
        let msg = CliMessage::command(String::new(), String::new(), test_addr());
        let encoded = msg.encode();
        let decoded = CliMessage::decode(&encoded, test_addr()).unwrap();

        assert!(decoded.node_name.is_empty());
        assert!(decoded.value.is_empty());
    }

    #[test]
    fn test_cli_message_unicode() {
        let msg = CliMessage::command(
            "gnb-日本語".to_string(),
            "状態".to_string(),
            test_addr(),
        );
        let encoded = msg.encode();
        let decoded = CliMessage::decode(&encoded, test_addr()).unwrap();

        assert_eq!(decoded.node_name, "gnb-日本語");
        assert_eq!(decoded.value, "状態");
    }

    #[tokio::test]
    async fn test_cli_server_creation() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 0);
        let server = CliServer::new(addr, "test-gnb".to_string()).await.unwrap();

        assert_eq!(server.node_name(), "test-gnb");
        assert!(server.local_addr().is_ok());
    }

    #[tokio::test]
    async fn test_cli_server_send_receive() {
        // Create server
        let server_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 0);
        let server = CliServer::new(server_addr, "test-gnb".to_string())
            .await
            .unwrap();
        let server_port = server.local_addr().unwrap().port();

        // Create client socket
        let client_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 0);
        let client = UdpSocket::bind(client_addr).await.unwrap();
        let client_local = client.local_addr().unwrap();

        // Send command from client
        let cmd = CliMessage::command(
            "test-gnb".to_string(),
            "status".to_string(),
            client_local,
        );
        let server_target = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), server_port);
        client.send_to(&cmd.encode(), server_target).await.unwrap();

        // Receive on server
        let received = server.receive().await.unwrap();
        assert_eq!(received.msg_type, CliMessageType::Command);
        assert_eq!(received.value, "status");

        // Send response from server
        server
            .send_result("ngap_up: true".to_string(), client_local)
            .await
            .unwrap();

        // Receive response on client
        let mut buffer = [0u8; CLI_BUFFER_SIZE];
        let (size, _) = client.recv_from(&mut buffer).await.unwrap();
        let response = CliMessage::decode(&buffer[..size], server_target).unwrap();
        assert_eq!(response.msg_type, CliMessageType::Result);
        assert_eq!(response.value, "ngap_up: true");
    }
}
