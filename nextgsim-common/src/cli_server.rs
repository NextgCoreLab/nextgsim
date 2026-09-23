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
    pub fn error(
        addr: SocketAddr,
        node_name: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            msg_type: CliMessageType::Error,
            node_name: node_name.into(),
            value: message.into(),
            client_addr: addr,
        }
    }

    /// Creates a new result response
    pub fn result(
        addr: SocketAddr,
        node_name: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
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
        let value =
            String::from_utf8(data[value_data_offset..value_data_offset + value_len].to_vec())
                .ok()?;

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

    /// Decodes a process table entry from its on-disk string form.
    ///
    /// The inverse of [`Self::encode`], mirroring `nr-cli`'s reader in
    /// `nextgsim-cli/src/proc_table.rs`. Returns `None` for any malformed entry
    /// rather than erroring, because a reader scanning the directory must skip a
    /// bad file and keep going.
    pub fn decode(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split_whitespace().collect();
        if parts.len() < 6 {
            return None;
        }

        let major: u8 = parts[0].parse().ok()?;
        let minor: u8 = parts[1].parse().ok()?;
        let patch: u8 = parts[2].parse().ok()?;
        let pid: u32 = parts[3].parse().ok()?;
        let port: u16 = parts[4].parse().ok()?;
        let node_count: usize = parts[5].parse().ok()?;

        if parts.len() < 6 + node_count {
            return None;
        }

        let nodes = parts[6..6 + node_count]
            .iter()
            .map(|s| (*s).to_string())
            .collect();

        Some(Self {
            major,
            minor,
            patch,
            pid,
            port,
            nodes,
        })
    }
}

/// Looks up the command port a node registered under, the way `nr-cli` does.
///
/// Scans [`PROC_TABLE_DIR`] for an entry naming `node_name` at this crate's
/// protocol version and returns its port, or `None` when the node is not
/// registered. This is the same lookup `nr-cli`'s `discover_node` performs
/// (`nextgsim-cli/src/proc_table.rs`), exposed here so that a test can assert the
/// property `nr-cli` actually depends on instead of approximating it — the gNB
/// used to bind a port that no reader could ever find (issue #197).
///
/// Version-mismatched entries are skipped, matching `nr-cli`: a port reached with
/// the wrong framing would have every datagram rejected by the peer's decoder.
pub fn lookup_node_port(node_name: &str) -> Option<u16> {
    let entries = fs::read_dir(PROC_TABLE_DIR).ok()?;

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let Ok(content) = fs::read_to_string(&path) else {
            continue;
        };

        let Some(table_entry) = ProcTableEntry::decode(&content) else {
            continue;
        };

        if table_entry.major != VERSION_MAJOR
            || table_entry.minor != VERSION_MINOR
            || table_entry.patch != VERSION_PATCH
        {
            continue;
        }

        if table_entry.nodes.iter().any(|n| n == node_name) {
            return Some(table_entry.port);
        }
    }

    None
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
        let socket = TokioUdpSocket::bind(format!("{CMD_SERVER_IP}:0")).await?;
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
        let socket = TokioUdpSocket::bind(format!("{CMD_SERVER_IP}:{port}")).await?;
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

    /// Decodes a received datagram into a command addressed to one of our nodes.
    ///
    /// Returns `None` if the datagram is empty, is not a well-formed `CliMessage`
    /// of this protocol version, is not a `Command`, or names a node this server
    /// did not register. Shared by the blocking and non-blocking receive paths so
    /// the two cannot drift in what they accept.
    fn decode_command(&self, data: &[u8], addr: SocketAddr) -> Option<CliCommand> {
        if data.is_empty() {
            return None;
        }

        let msg = CliMessage::decode(data, addr)?;

        if msg.msg_type != CliMessageType::Command {
            return None;
        }

        // Check if this command is for one of our nodes
        if !self.node_names.is_empty() && !self.node_names.contains(&msg.node_name) {
            // Not for us, ignore
            return None;
        }

        Some(CliCommand {
            command: msg.value,
            node_name: msg.node_name,
            client_addr: msg.client_addr,
        })
    }

    /// Receives a command from the CLI tool, waiting until a datagram arrives.
    ///
    /// Returns `None` if the message is invalid or not a command.
    pub async fn receive_command(&self) -> std::io::Result<Option<CliCommand>> {
        let mut buffer = [0u8; CMD_BUFFER_SIZE];

        let (size, addr) = self.socket.recv_from(&mut buffer).await?;

        Ok(self.decode_command(&buffer[..size], addr))
    }

    /// Receives a command from the CLI tool without waiting.
    ///
    /// Returns `Ok(None)` when no datagram is queued, so a caller that polls this
    /// from inside a `tokio::select!` arm keeps servicing its other arms. The
    /// awaiting [`Self::receive_command`] would instead park the whole loop until a
    /// CLI client happened to connect, which is why the gNB's App task needs this
    /// variant (issue #197).
    pub fn try_receive_command(&self) -> std::io::Result<Option<CliCommand>> {
        let mut buffer = [0u8; CMD_BUFFER_SIZE];

        match self.socket.try_recv_from(&mut buffer) {
            Ok((size, addr)) => Ok(self.decode_command(&buffer[..size], addr)),
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => Ok(None),
            Err(e) => Err(e),
        }
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
    pub async fn send_error(
        &self,
        addr: SocketAddr,
        message: impl Into<String>,
    ) -> std::io::Result<()> {
        self.send_response(CliResponse {
            message: message.into(),
            is_error: true,
            client_addr: addr,
        })
        .await
    }

    /// Sends a result response to the CLI tool
    pub async fn send_result(
        &self,
        addr: SocketAddr,
        message: impl Into<String>,
    ) -> std::io::Result<()> {
        self.send_response(CliResponse {
            message: message.into(),
            is_error: false,
            client_addr: addr,
        })
        .await
    }

    /// Sends an echo message to the CLI tool
    pub async fn send_echo(
        &self,
        addr: SocketAddr,
        message: impl Into<String>,
    ) -> std::io::Result<()> {
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

    // The codec cases below were carried over from the gNB-local `CliServer` that
    // issue #197 deleted. They exercise the shared `CliMessage` framing, not anything
    // gNB-specific, so they belong with the implementation that survived -- dropping
    // them would have lost real coverage of the decoder's reject paths.

    #[test]
    fn test_cli_message_type_try_from() {
        assert_eq!(CliMessageType::try_from(0), Ok(CliMessageType::Empty));
        assert_eq!(CliMessageType::try_from(1), Ok(CliMessageType::Echo));
        assert_eq!(CliMessageType::try_from(2), Ok(CliMessageType::Error));
        assert_eq!(CliMessageType::try_from(3), Ok(CliMessageType::Result));
        assert_eq!(CliMessageType::try_from(4), Ok(CliMessageType::Command));
        assert!(CliMessageType::try_from(255).is_err());
    }

    #[test]
    fn test_cli_message_decode_too_short() {
        let addr: SocketAddr = "127.0.0.1:5000".parse().unwrap();
        assert!(CliMessage::decode(&[], addr).is_none());
        assert!(CliMessage::decode(&[0u8; 5], addr).is_none());
    }

    #[test]
    fn test_cli_message_decode_wrong_version() {
        let addr: SocketAddr = "127.0.0.1:5000".parse().unwrap();
        let mut encoded = CliMessage::result(addr, "node1", "value").encode();
        encoded[0] = 99;
        assert!(
            CliMessage::decode(&encoded, addr).is_none(),
            "a mismatched protocol version must be rejected, not misread"
        );
    }

    #[test]
    fn test_cli_message_decode_invalid_type() {
        let addr: SocketAddr = "127.0.0.1:5000".parse().unwrap();
        let mut encoded = CliMessage::result(addr, "node1", "value").encode();
        encoded[3] = 99;
        assert!(CliMessage::decode(&encoded, addr).is_none());
    }

    #[test]
    fn test_cli_message_echo_roundtrip() {
        let addr: SocketAddr = "127.0.0.1:5000".parse().unwrap();
        let msg = CliMessage::echo(addr, "hello");
        let decoded = CliMessage::decode(&msg.encode(), addr).unwrap();

        assert_eq!(decoded.msg_type, CliMessageType::Echo);
        assert!(decoded.node_name.is_empty());
        assert_eq!(decoded.value, "hello");
    }

    #[test]
    fn test_cli_message_empty_strings() {
        let addr: SocketAddr = "127.0.0.1:5000".parse().unwrap();
        let msg = CliMessage::result(addr, "", "");
        let decoded = CliMessage::decode(&msg.encode(), addr).unwrap();

        assert!(decoded.node_name.is_empty());
        assert!(decoded.value.is_empty());
    }

    #[test]
    fn test_cli_message_unicode() {
        let addr: SocketAddr = "127.0.0.1:5000".parse().unwrap();
        // The length prefixes count BYTES, so a multi-byte node name is the case a
        // char-counting encoder would get wrong.
        let msg = CliMessage::result(addr, "gnb-日本語", "状態");
        let decoded = CliMessage::decode(&msg.encode(), addr).unwrap();

        assert_eq!(decoded.node_name, "gnb-日本語");
        assert_eq!(decoded.value, "状態");
    }

    #[test]
    fn test_proc_table_entry_decode_roundtrip() {
        let entry = ProcTableEntry {
            major: 1,
            minor: 0,
            patch: 0,
            pid: 4242,
            port: 6000,
            nodes: vec!["gnb".to_string(), "ue1".to_string()],
        };

        let decoded = ProcTableEntry::decode(&entry.encode()).unwrap();

        assert_eq!(decoded.major, 1);
        assert_eq!(decoded.minor, 0);
        assert_eq!(decoded.patch, 0);
        assert_eq!(decoded.pid, 4242);
        assert_eq!(decoded.port, 6000);
        assert_eq!(decoded.nodes, vec!["gnb", "ue1"]);
    }

    #[test]
    fn test_proc_table_entry_decode_rejects_malformed() {
        // Too few fields, a non-numeric port, and a node count that overruns the
        // node list: a directory scan must skip each of these rather than stop.
        assert!(ProcTableEntry::decode("").is_none());
        assert!(ProcTableEntry::decode("1 0 0").is_none());
        assert!(ProcTableEntry::decode("1 0 0 42 notaport 1 gnb").is_none());
        assert!(ProcTableEntry::decode("1 0 0 42 6000 2 gnb").is_none());
    }

    /// `try_receive_command` returns a queued command and does not block when the
    /// socket is empty — the property the gNB's App task needs to poll from inside
    /// its `select!` without stalling its other arms (issue #197).
    #[tokio::test]
    async fn test_try_receive_command_is_non_blocking_and_delivers() {
        let mut server = CliServer::new().await.unwrap();
        server.register_nodes(vec!["gnb-unit".to_string()]).unwrap();
        let target: SocketAddr = format!("{CMD_SERVER_IP}:{}", server.port())
            .parse()
            .unwrap();

        // Empty socket: returns immediately with nothing, rather than waiting.
        assert!(server.try_receive_command().unwrap().is_none());

        let client = TokioUdpSocket::bind(format!("{CMD_SERVER_IP}:0"))
            .await
            .unwrap();
        let client_addr = client.local_addr().unwrap();

        let msg = CliMessage {
            msg_type: CliMessageType::Command,
            node_name: "gnb-unit".to_string(),
            value: "ue-list".to_string(),
            client_addr,
        };
        client.send_to(&msg.encode(), target).await.unwrap();

        // Poll until the datagram lands; UDP delivery to loopback is not instant.
        let mut received = None;
        for _ in 0..100 {
            if let Some(cmd) = server.try_receive_command().unwrap() {
                received = Some(cmd);
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }

        let cmd = received.expect("the queued command must be delivered");
        assert_eq!(cmd.command, "ue-list");
        assert_eq!(cmd.node_name, "gnb-unit");
        assert_eq!(cmd.client_addr, client_addr);
    }

    /// A command naming a node this server did not register is dropped, so two
    /// instances on one host do not answer each other.
    #[tokio::test]
    async fn test_try_receive_command_filters_other_nodes() {
        let mut server = CliServer::new().await.unwrap();
        server.register_nodes(vec!["gnb-mine".to_string()]).unwrap();
        let target: SocketAddr = format!("{CMD_SERVER_IP}:{}", server.port())
            .parse()
            .unwrap();

        let client = TokioUdpSocket::bind(format!("{CMD_SERVER_IP}:0"))
            .await
            .unwrap();
        let client_addr = client.local_addr().unwrap();

        let msg = CliMessage {
            msg_type: CliMessageType::Command,
            node_name: "gnb-theirs".to_string(),
            value: "ue-list".to_string(),
            client_addr,
        };
        client.send_to(&msg.encode(), target).await.unwrap();

        // Drain for long enough that the datagram has certainly arrived, and assert
        // it was consumed without producing a command.
        for _ in 0..30 {
            assert!(
                server.try_receive_command().unwrap().is_none(),
                "a command for another node must not be surfaced"
            );
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    }
}
