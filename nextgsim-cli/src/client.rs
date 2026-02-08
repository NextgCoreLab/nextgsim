//! CLI client for connecting to running nextgsim instances
//!
//! The client uses UDP to communicate with running gNB/UE instances.
//! It discovers instances via the process table and sends commands
//! to the instance's command port.
//!
//! # Reference
//!
//! Based on UERANSIM's `src/cli.cpp` implementation.

use std::net::{SocketAddr, UdpSocket};
use std::time::Duration;

use anyhow::{Context, Result};

use crate::protocol::{CliMessage, MessageType};

/// Default command server IP (localhost)
pub const CMD_SERVER_IP: &str = "127.0.0.1";

/// Maximum buffer size for receiving messages
const CMD_BUFFER_SIZE: usize = 8192;

/// Receive timeout in milliseconds
const CMD_RCV_TIMEOUT_MS: u64 = 2500;

/// CLI client for communicating with running instances
pub struct CliClient {
    /// UDP socket for communication
    socket: UdpSocket,
    /// Target address (instance's command port)
    target_addr: SocketAddr,
    /// Node name we're connected to
    node_name: String,
}

impl CliClient {
    /// Creates a new CLI client connected to the specified port
    ///
    /// # Arguments
    ///
    /// * `port` - The command port of the target instance
    /// * `node_name` - The name of the node we're connecting to
    pub fn connect(port: u16, node_name: impl Into<String>) -> Result<Self> {
        let node_name = node_name.into();

        // Bind to any available port on localhost
        let socket = UdpSocket::bind(format!("{CMD_SERVER_IP}:0"))
            .context("Failed to bind UDP socket")?;

        // Set receive timeout
        socket
            .set_read_timeout(Some(Duration::from_millis(CMD_RCV_TIMEOUT_MS)))
            .context("Failed to set socket timeout")?;

        let target_addr: SocketAddr = format!("{CMD_SERVER_IP}:{port}")
            .parse()
            .context("Failed to parse target address")?;

        Ok(Self {
            socket,
            target_addr,
            node_name,
        })
    }

    /// Sends a command to the connected instance
    pub fn send_command(&self, command: &str) -> Result<()> {
        let msg = CliMessage::command(&self.node_name, command);
        let data = msg.encode();

        self.socket
            .send_to(&data, self.target_addr)
            .context("Failed to send command")?;

        Ok(())
    }

    /// Receives a message from the instance
    ///
    /// Returns `None` if the receive times out or an empty message is received.
    pub fn receive_message(&self) -> Result<Option<CliMessage>> {
        let mut buffer = [0u8; CMD_BUFFER_SIZE];

        match self.socket.recv_from(&mut buffer) {
            Ok((size, _addr)) => {
                if size == 0 {
                    return Ok(None);
                }

                let msg = CliMessage::decode(&buffer[..size])?;
                if msg.msg_type == MessageType::Empty {
                    return Ok(None);
                }

                Ok(Some(msg))
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                // Timeout
                Ok(None)
            }
            Err(e) if e.kind() == std::io::ErrorKind::TimedOut => {
                // Timeout (Windows)
                Ok(None)
            }
            Err(e) => Err(e).context("Failed to receive message"),
        }
    }

    /// Sends a command and waits for the response
    ///
    /// This method handles the response loop, collecting all echo messages
    /// and returning when a result or error is received.
    ///
    /// # Returns
    ///
    /// A tuple of (output, is_error) where output is the collected response
    /// and is_error indicates if the final message was an error.
    pub fn execute_command(&self, command: &str) -> Result<(String, bool)> {
        self.send_command(command)?;

        let mut output = String::new();

        loop {
            match self.receive_message()? {
                Some(msg) => {
                    match msg.msg_type {
                        MessageType::Echo => {
                            if !output.is_empty() {
                                output.push('\n');
                            }
                            output.push_str(&msg.value);
                        }
                        MessageType::Result => {
                            if !output.is_empty() {
                                output.push('\n');
                            }
                            output.push_str(&msg.value);
                            return Ok((output, false));
                        }
                        MessageType::Error => {
                            return Ok((msg.value, true));
                        }
                        _ => {
                            // Ignore other message types
                        }
                    }
                }
                None => {
                    // Timeout - return what we have
                    if output.is_empty() {
                        anyhow::bail!("No response from instance (timeout)");
                    }
                    return Ok((output, false));
                }
            }
        }
    }

    /// Returns the node name this client is connected to
    pub fn node_name(&self) -> &str {
        &self.node_name
    }

    /// Returns the target address
    #[allow(dead_code)]
    pub fn target_addr(&self) -> SocketAddr {
        self.target_addr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        // This test just verifies the client can be created
        // Actual communication tests would require a running server
        let result = CliClient::connect(0, "test-node");
        // Port 0 should fail since there's nothing listening
        // But the socket binding should succeed
        assert!(result.is_ok());
    }
}
