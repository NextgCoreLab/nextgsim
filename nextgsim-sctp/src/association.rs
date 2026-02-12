//! SCTP association management for NGAP transport.

use bytes::Bytes;
use sctp_proto::{
    Association, AssociationHandle, ClientConfig, DatagramEvent, Endpoint, EndpointConfig, Event,
    Payload, PayloadProtocolIdentifier, TransportConfig, Transmit,
};
use std::{collections::VecDeque, io, net::SocketAddr, sync::Arc, time::{Duration, Instant}};
use thiserror::Error;
use tokio::{net::UdpSocket, sync::mpsc, time::timeout};
use tracing::{debug, error, info, trace, warn};

/// NGAP Payload Protocol Identifier (PPID) as defined in 3GPP TS 38.412
pub const NGAP_PPID: u32 = 60;
/// Default number of SCTP streams for NGAP
pub const DEFAULT_NUM_STREAMS: u16 = 2;
/// Default maximum message size (64KB)
pub const DEFAULT_MAX_MESSAGE_SIZE: u32 = 65536;
/// Default receive buffer size (256KB)
pub const DEFAULT_RECEIVE_BUFFER_SIZE: u32 = 262144;

/// SCTP association errors
#[derive(Debug, Error)]
pub enum SctpError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
    #[error("Association closed")]
    AssociationClosed,
    #[error("Stream error: {0}")]
    StreamError(String),
    #[error("Timeout: {0}")]
    Timeout(String),
    #[error("Protocol error: {0}")]
    Protocol(String),
    #[error("Invalid state: {0}")]
    InvalidState(String),
}

/// Result type for SCTP operations
pub type Result<T> = std::result::Result<T, SctpError>;

/// SCTP association state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssociationState {
    Closed,
    Connecting,
    Established,
    ShuttingDown,
}

/// Configuration for SCTP association
#[derive(Debug, Clone)]
pub struct SctpConfig {
    pub max_outbound_streams: u16,
    pub max_inbound_streams: u16,
    pub max_message_size: u32,
    pub max_receive_buffer_size: u32,
    pub connect_timeout: Duration,
    pub rto_initial_ms: u64,
    pub rto_min_ms: u64,
    pub rto_max_ms: u64,
}

impl Default for SctpConfig {
    fn default() -> Self {
        Self {
            max_outbound_streams: DEFAULT_NUM_STREAMS,
            max_inbound_streams: DEFAULT_NUM_STREAMS,
            max_message_size: DEFAULT_MAX_MESSAGE_SIZE,
            max_receive_buffer_size: DEFAULT_RECEIVE_BUFFER_SIZE,
            connect_timeout: Duration::from_secs(30),
            rto_initial_ms: 3000,
            rto_min_ms: 1000,
            rto_max_ms: 60000,
        }
    }
}

/// Received SCTP message
#[derive(Debug, Clone)]
pub struct ReceivedMessage {
    pub stream_id: u16,
    pub data: Bytes,
    pub ppid: u32,
}

/// SCTP association events
#[derive(Debug, Clone)]
pub enum SctpEvent {
    Connected,
    Disconnected,
    DataReceived(ReceivedMessage),
    StreamOpened(u16),
    StreamClosed(u16),
}

/// SCTP association wrapper for NGAP transport
pub struct SctpAssociation {
    socket: Arc<UdpSocket>,
    remote_addr: SocketAddr,
    local_addr: SocketAddr,
    endpoint: Endpoint,
    handle: AssociationHandle,
    association: Association,
    state: AssociationState,
    pending_transmits: VecDeque<Transmit>,
    event_tx: Option<mpsc::UnboundedSender<SctpEvent>>,
    config: SctpConfig,
    /// Track streams that we've opened for sending (so we can also read responses from them)
    opened_streams: Vec<u16>,
}

impl SctpAssociation {
    /// Connect to a remote SCTP endpoint (AMF)
    pub async fn connect(remote_addr: SocketAddr, config: SctpConfig) -> Result<Self> {
        let local_addr: SocketAddr = if remote_addr.is_ipv6() {
            "[::]:0".parse().unwrap()
        } else {
            "0.0.0.0:0".parse().unwrap()
        };
        Self::connect_with_local(local_addr, remote_addr, config).await
    }

    /// Connect to a remote SCTP endpoint with a specific local address
    pub async fn connect_with_local(
        local_addr: SocketAddr,
        remote_addr: SocketAddr,
        config: SctpConfig,
    ) -> Result<Self> {
        info!("Connecting to SCTP endpoint at {}", remote_addr);

        // Bind UDP socket
        let socket = UdpSocket::bind(local_addr).await?;
        let actual_local = socket.local_addr()?;
        debug!("Bound to local address: {}", actual_local);

        // Create endpoint config
        let endpoint_config = EndpointConfig::new();
        let mut endpoint = Endpoint::new(Arc::new(endpoint_config), None);

        // Create transport config with builder pattern
        let transport_config = TransportConfig::default()
            .with_max_num_outbound_streams(config.max_outbound_streams)
            .with_max_num_inbound_streams(config.max_inbound_streams)
            .with_max_message_size(config.max_message_size)
            .with_max_receive_buffer_size(config.max_receive_buffer_size)
            .with_rto_initial_ms(config.rto_initial_ms)
            .with_rto_min_ms(config.rto_min_ms)
            .with_rto_max_ms(config.rto_max_ms);

        // Create client config
        let mut client_config = ClientConfig::new();
        client_config.transport = Arc::new(transport_config);

        // Initiate connection
        let (handle, association) = endpoint
            .connect(client_config, remote_addr)
            .map_err(|e| SctpError::ConnectionFailed(e.to_string()))?;

        let socket = Arc::new(socket);
        let mut assoc = Self {
            socket,
            remote_addr,
            local_addr: actual_local,
            endpoint,
            handle,
            association,
            state: AssociationState::Connecting,
            pending_transmits: VecDeque::new(),
            event_tx: None,
            config,
            opened_streams: Vec::new(),
        };

        // Perform handshake
        assoc.perform_handshake().await?;

        Ok(assoc)
    }

    /// Perform SCTP 4-way handshake
    async fn perform_handshake(&mut self) -> Result<()> {
        let deadline = Instant::now() + self.config.connect_timeout;

        while self.state == AssociationState::Connecting {
            if Instant::now() > deadline {
                return Err(SctpError::Timeout("Connection handshake timed out".into()));
            }

            // Flush any pending transmits
            self.flush_transmits().await?;

            // Poll for events
            self.poll_events();

            // Check if handshake completed
            if !self.association.is_handshaking() {
                self.state = AssociationState::Established;
                info!("SCTP association established with {}", self.remote_addr);
                if let Some(tx) = &self.event_tx {
                    let _ = tx.send(SctpEvent::Connected);
                }
                return Ok(());
            }

            // Receive incoming packets with timeout
            let recv_timeout = Duration::from_millis(100);
            match timeout(recv_timeout, self.handle_incoming()).await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    warn!("Error handling incoming packet: {}", e);
                }
                Err(_) => {
                    // Timeout, continue loop
                    trace!("Receive timeout, continuing handshake");
                }
            }
        }

        Ok(())
    }

    /// Handle incoming UDP packets
    async fn handle_incoming(&mut self) -> Result<()> {
        let mut buf = vec![0u8; self.config.max_receive_buffer_size as usize];
        let (len, from) = self.socket.recv_from(&mut buf).await?;
        buf.truncate(len);

        trace!("Received {} bytes from {}", len, from);

        let now = Instant::now();
        if let Some((handle, event)) = self.endpoint.handle(now, from, None, None, Bytes::from(buf)) {
            if handle == self.handle {
                match event {
                    DatagramEvent::AssociationEvent(assoc_event) => {
                        self.association.handle_event(assoc_event);
                    }
                    DatagramEvent::NewAssociation(_) => {
                        // We're a client, ignore new associations
                        debug!("Ignoring new association event (client mode)");
                    }
                }
            }
        }

        Ok(())
    }

    /// Poll for association events and process them
    fn poll_events(&mut self) {
        while let Some(event) = self.association.poll() {
            match event {
                Event::Connected => {
                    debug!("Association connected event");
                    self.state = AssociationState::Established;
                }
                Event::AssociationLost { reason } => {
                    warn!("Association lost: {}", reason);
                    self.state = AssociationState::Closed;
                    if let Some(tx) = &self.event_tx {
                        let _ = tx.send(SctpEvent::Disconnected);
                    }
                }
                Event::Stream(stream_event) => {
                    trace!("Stream event: {:?}", stream_event);
                }
                Event::DatagramReceived => {
                    trace!("Datagram received event");
                }
            }
        }

        // Handle timeouts
        if let Some(timeout_instant) = self.association.poll_timeout() {
            if Instant::now() >= timeout_instant {
                self.association.handle_timeout(Instant::now());
            }
        }

        // Collect transmits from association
        while let Some(transmit) = self.association.poll_transmit(Instant::now()) {
            self.pending_transmits.push_back(transmit);
        }

        // Collect transmits from endpoint
        while let Some(transmit) = self.endpoint.poll_transmit() {
            self.pending_transmits.push_back(transmit);
        }
    }

    /// Flush pending transmits to the network
    async fn flush_transmits(&mut self) -> Result<()> {
        while let Some(transmit) = self.pending_transmits.pop_front() {
            match &transmit.payload {
                Payload::RawEncode(chunks) => {
                    for chunk in chunks {
                        self.socket.send_to(chunk, transmit.remote).await?;
                        trace!("Sent {} bytes to {}", chunk.len(), transmit.remote);
                    }
                }
                Payload::PartialDecode(_) => {
                    // PartialDecode is for incoming packets, skip for outgoing
                    trace!("Skipping PartialDecode payload for transmit");
                }
            }
        }
        Ok(())
    }

    /// Send data on a specific stream with NGAP PPID
    pub async fn send(&mut self, stream_id: u16, data: &[u8]) -> Result<()> {
        self.send_with_ppid(stream_id, data, NGAP_PPID).await
    }

    /// Send data on a specific stream with custom PPID
    pub async fn send_with_ppid(&mut self, stream_id: u16, data: &[u8], ppid: u32) -> Result<()> {
        if self.state != AssociationState::Established {
            return Err(SctpError::InvalidState(
                "Cannot send: association not established".into(),
            ));
        }

        let ppi = PayloadProtocolIdentifier::from(ppid);

        // Try to get existing stream first, otherwise open a new one
        let mut stream = match self.association.stream(stream_id) {
            Ok(s) => s,
            Err(_) => {
                // Open new stream and track it
                let s = self.association.open_stream(stream_id, ppi)
                    .map_err(|e| SctpError::StreamError(e.to_string()))?;
                if !self.opened_streams.contains(&stream_id) {
                    self.opened_streams.push(stream_id);
                    debug!("Opened new stream {} (now tracking {} streams)", stream_id, self.opened_streams.len());
                }
                s
            }
        };

        stream.write_with_ppi(data, ppi)
            .map_err(|e| SctpError::StreamError(e.to_string()))?;

        debug!("Queued {} bytes on stream {} with PPID {}", data.len(), stream_id, ppid);

        // Poll and flush
        self.poll_events();
        self.flush_transmits().await?;

        Ok(())
    }

    /// Receive a message (blocking)
    pub async fn recv(&mut self) -> Result<Option<ReceivedMessage>> {
        if self.state == AssociationState::Closed {
            return Err(SctpError::AssociationClosed);
        }

        loop {
            // Check for available data first
            if let Some(msg) = self.try_recv()? {
                return Ok(Some(msg));
            }

            // Handle incoming packets
            self.handle_incoming().await?;
            self.poll_events();
            self.flush_transmits().await?;

            // Check state
            if self.state == AssociationState::Closed {
                return Ok(None);
            }
        }
    }

    /// Poll for incoming data - receives UDP packets and processes them
    /// This MUST be called periodically to receive incoming SCTP data
    pub async fn poll(&mut self) -> Result<()> {
        if self.state == AssociationState::Closed {
            return Ok(());
        }

        // Try to receive incoming UDP packets with a short timeout
        let recv_timeout = Duration::from_millis(1);
        match timeout(recv_timeout, self.handle_incoming()).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                // Only log real errors, not timeouts
                if !matches!(e, SctpError::Io(ref io_err) if io_err.kind() == std::io::ErrorKind::WouldBlock) {
                    trace!("Handle incoming error: {}", e);
                }
            }
            Err(_) => {
                // Timeout, no data available
            }
        }

        // Process events
        self.poll_events();

        // Flush any pending transmits
        self.flush_transmits().await?;

        Ok(())
    }

    /// Try to receive a message (non-blocking)
    /// Note: You must call `poll()` first to receive incoming UDP packets
    pub fn try_recv(&mut self) -> Result<Option<ReceivedMessage>> {
        // Accept any incoming streams
        while let Some(stream) = self.association.accept_stream() {
            let stream_id = stream.stream_identifier();
            debug!("Accepted stream {}", stream_id);

            // Track newly accepted streams for future reads
            if !self.opened_streams.contains(&stream_id) {
                self.opened_streams.push(stream_id);
            }

            if let Some(tx) = &self.event_tx {
                let _ = tx.send(SctpEvent::StreamOpened(stream_id));
            }
        }

        // Check all tracked streams for incoming data
        for &stream_id in &self.opened_streams.clone() {
            if let Ok(mut stream) = self.association.stream(stream_id) {
                // Try to read from the stream
                if let Ok(Some(chunks)) = stream.read() {
                    let ppid = match chunks.ppi {
                        PayloadProtocolIdentifier::Dcep => 50,
                        PayloadProtocolIdentifier::String => 51,
                        PayloadProtocolIdentifier::Binary => 53,
                        PayloadProtocolIdentifier::StringEmpty => 56,
                        PayloadProtocolIdentifier::BinaryEmpty => 57,
                        PayloadProtocolIdentifier::Unknown => NGAP_PPID,
                    };
                    // Read all data from chunks into a buffer
                    let total_len = chunks.len();
                    if total_len > 0 {
                        let mut buf = vec![0u8; total_len];
                        if chunks.read(&mut buf).is_ok() {
                            let msg = ReceivedMessage {
                                stream_id,
                                data: Bytes::from(buf),
                                ppid,
                            };
                            debug!("Received {} bytes on stream {} with PPID {}", msg.data.len(), stream_id, msg.ppid);

                            if let Some(tx) = &self.event_tx {
                                let _ = tx.send(SctpEvent::DataReceived(msg.clone()));
                            }

                            return Ok(Some(msg));
                        }
                    }
                }
            }
        }

        Ok(None)
    }

    /// Initiate graceful shutdown
    pub async fn shutdown(&mut self) -> Result<()> {
        if self.state == AssociationState::Closed {
            return Ok(());
        }

        info!("Initiating SCTP shutdown");
        self.state = AssociationState::ShuttingDown;

        let _ = self.association.shutdown();
        self.poll_events();
        self.flush_transmits().await?;

        // Wait for shutdown to complete with timeout
        let deadline = Instant::now() + Duration::from_secs(5);
        while self.state == AssociationState::ShuttingDown && Instant::now() < deadline {
            if let Ok(Ok(())) = timeout(Duration::from_millis(100), self.handle_incoming()).await {}
            self.poll_events();
            self.flush_transmits().await?;

            if self.association.is_closed() {
                break;
            }
        }

        self.state = AssociationState::Closed;
        if let Some(tx) = &self.event_tx {
            let _ = tx.send(SctpEvent::Disconnected);
        }

        info!("SCTP shutdown complete");
        Ok(())
    }

    /// Close the association immediately
    pub fn close(&mut self) {
        if self.state != AssociationState::Closed {
            let _ = self.association.close();
            self.state = AssociationState::Closed;
            if let Some(tx) = &self.event_tx {
                let _ = tx.send(SctpEvent::Disconnected);
            }
        }
    }

    /// Run the association event loop (for use with `tokio::spawn`)
    pub async fn run(&mut self) -> Result<()> {
        info!("Starting SCTP association event loop");

        while self.state == AssociationState::Established {
            // Handle incoming with timeout
            match timeout(Duration::from_millis(50), self.handle_incoming()).await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    error!("Error in event loop: {}", e);
                    break;
                }
                Err(_) => {
                    // Timeout, continue
                }
            }

            self.poll_events();
            self.flush_transmits().await?;

            // Check for received data
            if let Some(msg) = self.try_recv()? {
                if let Some(tx) = &self.event_tx {
                    let _ = tx.send(SctpEvent::DataReceived(msg));
                }
            }
        }

        Ok(())
    }

    // Accessor methods

    /// Check if the association is established
    pub fn is_established(&self) -> bool {
        self.state == AssociationState::Established
    }

    /// Check if the association is closed
    pub fn is_closed(&self) -> bool {
        self.state == AssociationState::Closed
    }

    /// Get the current state
    pub fn state(&self) -> AssociationState {
        self.state
    }

    /// Get the remote address
    pub fn remote_addr(&self) -> SocketAddr {
        self.remote_addr
    }

    /// Get the local address
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Get the current RTT estimate
    pub fn rtt(&self) -> Duration {
        self.association.rtt()
    }

    /// Set the event sender for receiving association events
    pub fn set_event_sender(&mut self, tx: mpsc::UnboundedSender<SctpEvent>) {
        self.event_tx = Some(tx);
    }
}

impl Drop for SctpAssociation {
    fn drop(&mut self) {
        if self.state != AssociationState::Closed {
            self.close();
        }
    }
}


// ===========================================================================
// A6.1: Multi-homing support
// ===========================================================================

/// State of a single network path in a multi-homed association
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathState {
    /// Path is active and reachable
    Active,
    /// Path is inactive (heartbeat lost)
    Inactive,
    /// Path is being probed for reachability
    Probing,
    /// Path has permanently failed
    Failed,
}

/// A single network path in a multi-homed SCTP association
///
/// Each path represents a (`local_addr`, `remote_addr`) pair that can be used
/// for data transmission. SCTP multi-homing allows an association to use
/// multiple network paths for redundancy.
#[derive(Debug, Clone)]
pub struct SctpPath {
    /// Local address for this path
    pub local_addr: SocketAddr,
    /// Remote address for this path
    pub remote_addr: SocketAddr,
    /// Current state of the path
    pub state: PathState,
    /// Estimated RTT for this path
    pub rtt: Duration,
    /// Number of consecutive heartbeat failures
    pub error_count: u32,
    /// Maximum allowed consecutive errors before marking path as failed
    pub max_retransmissions: u32,
    /// Interval between heartbeat probes
    pub heartbeat_interval: Duration,
    /// Timestamp of last successful heartbeat acknowledgment
    pub last_heartbeat_ack: Option<Instant>,
    /// Timestamp of last sent heartbeat
    pub last_heartbeat_sent: Option<Instant>,
}

impl SctpPath {
    /// Create a new path
    pub fn new(local_addr: SocketAddr, remote_addr: SocketAddr) -> Self {
        Self {
            local_addr,
            remote_addr,
            state: PathState::Active,
            rtt: Duration::from_millis(200), // Initial RTT estimate
            error_count: 0,
            max_retransmissions: 5,
            heartbeat_interval: Duration::from_secs(30),
            last_heartbeat_ack: None,
            last_heartbeat_sent: None,
        }
    }

    /// Check if this path is usable for data transmission
    pub fn is_usable(&self) -> bool {
        self.state == PathState::Active
    }

    /// Record a heartbeat acknowledgment
    pub fn heartbeat_ack(&mut self) {
        self.error_count = 0;
        self.state = PathState::Active;
        self.last_heartbeat_ack = Some(Instant::now());
    }

    /// Record a heartbeat failure
    pub fn heartbeat_failure(&mut self) {
        self.error_count += 1;
        if self.error_count >= self.max_retransmissions {
            self.state = PathState::Failed;
        } else {
            self.state = PathState::Inactive;
        }
    }

    /// Check if a heartbeat probe is due
    pub fn needs_heartbeat(&self) -> bool {
        match self.last_heartbeat_sent {
            Some(last) => Instant::now().duration_since(last) >= self.heartbeat_interval,
            None => true,
        }
    }

    /// Record that a heartbeat was sent
    pub fn mark_heartbeat_sent(&mut self) {
        self.last_heartbeat_sent = Some(Instant::now());
        if self.state == PathState::Inactive {
            self.state = PathState::Probing;
        }
    }
}


/// Configuration for SCTP multi-homing
///
/// Multi-homing allows an SCTP association to use multiple network
/// interfaces/addresses for redundancy. If the primary path fails,
/// traffic is automatically switched to an alternate path.
#[derive(Debug, Clone)]
pub struct MultihomingConfig {
    /// Primary local addresses (first is the primary path)
    pub local_addresses: Vec<SocketAddr>,
    /// Primary remote addresses (first is the primary path)
    pub remote_addresses: Vec<SocketAddr>,
    /// Heartbeat interval for path probing
    pub heartbeat_interval: Duration,
    /// Maximum path retransmissions before declaring path failed
    pub max_path_retransmissions: u32,
    /// Automatic failover when primary path fails
    pub auto_failover: bool,
}

impl Default for MultihomingConfig {
    fn default() -> Self {
        Self {
            local_addresses: Vec::new(),
            remote_addresses: Vec::new(),
            heartbeat_interval: Duration::from_secs(30),
            max_path_retransmissions: 5,
            auto_failover: true,
        }
    }
}

impl MultihomingConfig {
    /// Create a new multi-homing configuration with a primary address pair
    pub fn new(primary_local: SocketAddr, primary_remote: SocketAddr) -> Self {
        Self {
            local_addresses: vec![primary_local],
            remote_addresses: vec![primary_remote],
            ..Self::default()
        }
    }

    /// Add an alternate local address
    pub fn with_local_address(mut self, addr: SocketAddr) -> Self {
        if !self.local_addresses.contains(&addr) {
            self.local_addresses.push(addr);
        }
        self
    }

    /// Add an alternate remote address
    pub fn with_remote_address(mut self, addr: SocketAddr) -> Self {
        if !self.remote_addresses.contains(&addr) {
            self.remote_addresses.push(addr);
        }
        self
    }

    /// Set the heartbeat interval
    pub fn with_heartbeat_interval(mut self, interval: Duration) -> Self {
        self.heartbeat_interval = interval;
        self
    }

    /// Set the maximum path retransmissions
    pub fn with_max_path_retransmissions(mut self, max: u32) -> Self {
        self.max_path_retransmissions = max;
        self
    }

    /// Get the primary local address
    pub fn primary_local(&self) -> Option<SocketAddr> {
        self.local_addresses.first().copied()
    }

    /// Get the primary remote address
    pub fn primary_remote(&self) -> Option<SocketAddr> {
        self.remote_addresses.first().copied()
    }

    /// Get all address pairs (local, remote) as paths
    pub fn all_paths(&self) -> Vec<(SocketAddr, SocketAddr)> {
        let mut paths = Vec::new();
        for local in &self.local_addresses {
            for remote in &self.remote_addresses {
                paths.push((*local, *remote));
            }
        }
        paths
    }
}


/// Multi-homed path manager for SCTP associations
///
/// Manages multiple network paths for a single SCTP association,
/// providing path selection, heartbeat monitoring, and failover.
#[derive(Debug)]
pub struct PathManager {
    /// All available paths
    paths: Vec<SctpPath>,
    /// Index of the currently active primary path
    primary_index: usize,
    /// Whether automatic failover is enabled
    auto_failover: bool,
}

impl PathManager {
    /// Create a new path manager from a multi-homing configuration
    pub fn from_config(config: &MultihomingConfig) -> Self {
        let mut paths = Vec::new();
        for (local, remote) in config.all_paths() {
            let mut path = SctpPath::new(local, remote);
            path.heartbeat_interval = config.heartbeat_interval;
            path.max_retransmissions = config.max_path_retransmissions;
            paths.push(path);
        }

        Self {
            paths,
            primary_index: 0,
            auto_failover: config.auto_failover,
        }
    }

    /// Create a path manager with a single path
    pub fn single_path(local: SocketAddr, remote: SocketAddr) -> Self {
        Self {
            paths: vec![SctpPath::new(local, remote)],
            primary_index: 0,
            auto_failover: true,
        }
    }

    /// Get the current primary path
    pub fn primary_path(&self) -> Option<&SctpPath> {
        self.paths.get(self.primary_index)
    }

    /// Get a mutable reference to the primary path
    pub fn primary_path_mut(&mut self) -> Option<&mut SctpPath> {
        self.paths.get_mut(self.primary_index)
    }

    /// Get all paths
    pub fn paths(&self) -> &[SctpPath] {
        &self.paths
    }

    /// Get the number of paths
    pub fn path_count(&self) -> usize {
        self.paths.len()
    }

    /// Get the number of active (usable) paths
    pub fn active_path_count(&self) -> usize {
        self.paths.iter().filter(|p| p.is_usable()).count()
    }

    /// Select the best available path for data transmission
    ///
    /// Returns the primary path if it is usable, otherwise selects
    /// the first available alternate path. Returns `None` if all paths
    /// have failed.
    pub fn select_path(&self) -> Option<&SctpPath> {
        // Try primary first
        if let Some(path) = self.paths.get(self.primary_index) {
            if path.is_usable() {
                return Some(path);
            }
        }

        // Try alternates
        self.paths.iter().find(|p| p.is_usable())
    }

    /// Handle a heartbeat acknowledgment for a specific path
    pub fn handle_heartbeat_ack(&mut self, remote_addr: SocketAddr) {
        for path in &mut self.paths {
            if path.remote_addr == remote_addr {
                path.heartbeat_ack();
                return;
            }
        }
    }

    /// Handle a heartbeat failure for a specific path
    ///
    /// If the primary path fails and `auto_failover` is enabled,
    /// automatically switches to the next available path.
    pub fn handle_heartbeat_failure(&mut self, remote_addr: SocketAddr) {
        let mut primary_failed = false;

        for (i, path) in self.paths.iter_mut().enumerate() {
            if path.remote_addr == remote_addr {
                path.heartbeat_failure();
                if i == self.primary_index && !path.is_usable() {
                    primary_failed = true;
                }
                break;
            }
        }

        if primary_failed && self.auto_failover {
            self.failover();
        }
    }

    /// Perform failover: switch primary to the next available path
    ///
    /// Returns `true` if failover succeeded, `false` if no alternate
    /// path is available.
    pub fn failover(&mut self) -> bool {
        for (i, path) in self.paths.iter().enumerate() {
            if i != self.primary_index && path.is_usable() {
                info!(
                    "SCTP path failover: {} -> {}",
                    self.paths[self.primary_index].remote_addr,
                    path.remote_addr
                );
                self.primary_index = i;
                return true;
            }
        }
        warn!("SCTP path failover failed: no usable alternate paths");
        false
    }

    /// Manually set the primary path index
    pub fn set_primary(&mut self, index: usize) -> bool {
        if index < self.paths.len() {
            self.primary_index = index;
            true
        } else {
            false
        }
    }

    /// Get paths that need heartbeat probes
    pub fn paths_needing_heartbeat(&self) -> Vec<usize> {
        self.paths
            .iter()
            .enumerate()
            .filter(|(_, p)| p.needs_heartbeat() && p.state != PathState::Failed)
            .map(|(i, _)| i)
            .collect()
    }

    /// Mark a heartbeat as sent for a specific path
    pub fn mark_heartbeat_sent(&mut self, index: usize) {
        if let Some(path) = self.paths.get_mut(index) {
            path.mark_heartbeat_sent();
        }
    }
}


// ===========================================================================
// A5.3: Multi-SCTP Stream Management (TS 38.412)
// ===========================================================================

/// NGAP message category for stream routing (TS 38.412 Section 7)
///
/// Different NGAP procedure types are mapped to different SCTP streams
/// to prevent head-of-line blocking between critical signaling and
/// non-UE-associated procedures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NgapStreamCategory {
    /// Non-UE-associated signaling (NG Setup, Reset, Error Indication)
    /// Must always use stream 0 per TS 38.412
    NonUeAssociated,
    /// UE-associated signaling (Initial UE Message, UL/DL NAS Transport, etc.)
    /// Can use any stream > 0
    UeAssociated,
    /// High-priority UE signaling (Handover, PDU Session Resource Setup)
    UeHighPriority,
}

/// Stream allocation policy for NGAP
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamAllocationPolicy {
    /// Round-robin across available streams
    RoundRobin,
    /// Use stream 0 for non-UE, stream 1+ for UE-associated (hash-based)
    CategoryBased,
}

impl Default for StreamAllocationPolicy {
    fn default() -> Self {
        Self::CategoryBased
    }
}

/// SCTP stream manager for NGAP multi-stream transport
///
/// Manages stream allocation and routing per 3GPP TS 38.412, which requires:
/// - Stream 0: Non-UE-associated signaling (NG Setup, Reset, etc.)
/// - Stream 1+: UE-associated signaling (distributed by UE context)
///
/// This prevents head-of-line blocking between different UE contexts
/// and between non-UE and UE-associated procedures.
#[derive(Debug)]
pub struct StreamManager {
    /// Total number of outbound streams negotiated
    num_streams: u16,
    /// Stream allocation policy
    policy: StreamAllocationPolicy,
    /// Round-robin counter for UE-associated stream allocation
    rr_counter: u16,
    /// Per-stream message count (for load monitoring)
    stream_msg_count: Vec<u64>,
}

impl StreamManager {
    /// Create a new stream manager with the given number of streams
    pub fn new(num_streams: u16) -> Self {
        let effective = num_streams.max(1);
        Self {
            num_streams: effective,
            policy: StreamAllocationPolicy::default(),
            rr_counter: 0,
            stream_msg_count: vec![0; effective as usize],
        }
    }

    /// Create a stream manager with a specific allocation policy
    pub fn with_policy(mut self, policy: StreamAllocationPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Get the number of available streams
    pub fn num_streams(&self) -> u16 {
        self.num_streams
    }

    /// Get the allocation policy
    pub fn policy(&self) -> StreamAllocationPolicy {
        self.policy
    }

    /// Select a stream for the given NGAP message category
    ///
    /// Per TS 38.412:
    /// - Non-UE-associated: always stream 0
    /// - UE-associated: hash-based or round-robin on streams 1..N
    pub fn select_stream(&mut self, category: NgapStreamCategory) -> u16 {
        match self.policy {
            StreamAllocationPolicy::CategoryBased => {
                self.select_stream_category_based(category)
            }
            StreamAllocationPolicy::RoundRobin => {
                self.select_stream_round_robin(category)
            }
        }
    }

    /// Select stream for a UE-associated message with a specific UE context ID
    ///
    /// Ensures all messages for the same UE use the same stream,
    /// preserving ordering guarantees per UE context.
    pub fn select_stream_for_ue(&mut self, ue_id: u32) -> u16 {
        if self.num_streams <= 1 {
            return 0;
        }
        // Hash UE ID to stream 1..N (stream 0 reserved for non-UE)
        let stream = 1 + (ue_id as u16 % (self.num_streams - 1));
        self.record_usage(stream);
        stream
    }

    /// Get per-stream message counts for load monitoring
    pub fn stream_msg_counts(&self) -> &[u64] {
        &self.stream_msg_count
    }

    /// Get the total number of messages sent across all streams
    pub fn total_messages(&self) -> u64 {
        self.stream_msg_count.iter().sum()
    }

    /// Get the least loaded stream (excluding stream 0)
    pub fn least_loaded_stream(&self) -> u16 {
        if self.num_streams <= 1 {
            return 0;
        }
        let mut min_stream = 1u16;
        let mut min_count = u64::MAX;
        for i in 1..self.num_streams {
            let count = self.stream_msg_count[i as usize];
            if count < min_count {
                min_count = count;
                min_stream = i;
            }
        }
        min_stream
    }

    fn select_stream_category_based(&mut self, category: NgapStreamCategory) -> u16 {
        let stream = match category {
            NgapStreamCategory::NonUeAssociated => 0,
            NgapStreamCategory::UeAssociated => {
                if self.num_streams <= 1 {
                    0
                } else {
                    self.least_loaded_stream()
                }
            }
            NgapStreamCategory::UeHighPriority => {
                if self.num_streams <= 1 {
                    0
                } else {
                    // High-priority uses stream 1 (dedicated)
                    1
                }
            }
        };
        self.record_usage(stream);
        stream
    }

    fn select_stream_round_robin(&mut self, category: NgapStreamCategory) -> u16 {
        let stream = match category {
            NgapStreamCategory::NonUeAssociated => 0,
            _ => {
                if self.num_streams <= 1 {
                    0
                } else {
                    let stream = 1 + (self.rr_counter % (self.num_streams - 1));
                    self.rr_counter = self.rr_counter.wrapping_add(1);
                    stream
                }
            }
        };
        self.record_usage(stream);
        stream
    }

    fn record_usage(&mut self, stream: u16) {
        if (stream as usize) < self.stream_msg_count.len() {
            self.stream_msg_count[stream as usize] += 1;
        }
    }
}

// ===========================================================================
// A6.2: PR-SCTP (Partial Reliability) support
// ===========================================================================

/// Partial Reliability policy for SCTP messages (RFC 3758)
///
/// PR-SCTP allows per-message reliability configuration. Some messages
/// (e.g. real-time data) may be discarded if they cannot be delivered
/// within timing or retransmission constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartialReliabilityPolicy {
    /// Fully reliable transfer (standard SCTP behavior)
    ReliableTransfer,
    /// Timed reliability: message may be discarded after the given duration
    /// since first transmission attempt
    TimedReliability(Duration),
    /// Limited retransmissions: message may be discarded after the given
    /// number of retransmission attempts
    LimitedRetransmissions(u32),
}

impl PartialReliabilityPolicy {
    /// Check if this policy is fully reliable
    pub fn is_reliable(&self) -> bool {
        matches!(self, Self::ReliableTransfer)
    }

    /// Check if a message should be abandoned based on this policy
    ///
    /// `first_send_time` is when the message was first transmitted,
    /// `retransmission_count` is the number of retransmissions so far.
    pub fn should_abandon(&self, first_send_time: Instant, retransmission_count: u32) -> bool {
        match self {
            Self::ReliableTransfer => false,
            Self::TimedReliability(max_duration) => {
                Instant::now().duration_since(first_send_time) > *max_duration
            }
            Self::LimitedRetransmissions(max_retx) => retransmission_count > *max_retx,
        }
    }
}

impl Default for PartialReliabilityPolicy {
    fn default() -> Self {
        Self::ReliableTransfer
    }
}


/// Forward TSN chunk information for PR-SCTP
///
/// When a sender decides to abandon a message under PR-SCTP, it sends
/// a Forward TSN chunk to inform the receiver that certain TSNs will
/// never be retransmitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForwardTsnChunk {
    /// New cumulative TSN: all TSNs up to and including this value
    /// are considered received (or abandoned) by the sender
    pub new_cumulative_tsn: u32,
    /// Per-stream information about skipped sequence numbers
    pub stream_info: Vec<ForwardTsnStreamInfo>,
}

/// Per-stream information in a Forward TSN chunk
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForwardTsnStreamInfo {
    /// Stream identifier
    pub stream_id: u16,
    /// Stream sequence number being skipped
    pub stream_sequence_number: u16,
}

impl ForwardTsnChunk {
    /// Create a new Forward TSN chunk
    pub fn new(new_cumulative_tsn: u32) -> Self {
        Self {
            new_cumulative_tsn,
            stream_info: Vec::new(),
        }
    }

    /// Add stream info for a skipped message
    pub fn with_stream_info(mut self, stream_id: u16, sequence_number: u16) -> Self {
        self.stream_info.push(ForwardTsnStreamInfo {
            stream_id,
            stream_sequence_number: sequence_number,
        });
        self
    }

    /// Encode the Forward TSN chunk to bytes
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(4 + self.stream_info.len() * 4);

        // New cumulative TSN
        buf.extend_from_slice(&self.new_cumulative_tsn.to_be_bytes());

        // Stream info entries
        for info in &self.stream_info {
            buf.extend_from_slice(&info.stream_id.to_be_bytes());
            buf.extend_from_slice(&info.stream_sequence_number.to_be_bytes());
        }

        buf
    }

    /// Decode a Forward TSN chunk from bytes
    ///
    /// # Errors
    ///
    /// Returns an error if the data is too short.
    pub fn decode(data: &[u8]) -> std::result::Result<Self, String> {
        if data.len() < 4 {
            return Err(format!(
                "Forward TSN too short: need 4 bytes, have {}",
                data.len()
            ));
        }

        let new_cumulative_tsn =
            u32::from_be_bytes([data[0], data[1], data[2], data[3]]);

        let mut stream_info = Vec::new();
        let mut offset = 4;
        while offset + 4 <= data.len() {
            let stream_id =
                u16::from_be_bytes([data[offset], data[offset + 1]]);
            let stream_sequence_number =
                u16::from_be_bytes([data[offset + 2], data[offset + 3]]);
            stream_info.push(ForwardTsnStreamInfo {
                stream_id,
                stream_sequence_number,
            });
            offset += 4;
        }

        Ok(Self {
            new_cumulative_tsn,
            stream_info,
        })
    }
}


/// Tracks per-message partial reliability state
///
/// Each outbound message that uses PR-SCTP gets a `PrSctpMessage` entry
/// to track when it was first sent and how many times it has been retransmitted.
#[derive(Debug, Clone)]
pub struct PrSctpMessage {
    /// TSN assigned to this message
    pub tsn: u32,
    /// Stream identifier
    pub stream_id: u16,
    /// Stream sequence number
    pub stream_sequence_number: u16,
    /// Reliability policy for this message
    pub policy: PartialReliabilityPolicy,
    /// Timestamp of first transmission attempt
    pub first_send_time: Instant,
    /// Number of retransmission attempts
    pub retransmission_count: u32,
    /// Whether this message has been abandoned
    pub abandoned: bool,
}

impl PrSctpMessage {
    /// Create a new PR-SCTP tracked message
    pub fn new(
        tsn: u32,
        stream_id: u16,
        stream_sequence_number: u16,
        policy: PartialReliabilityPolicy,
    ) -> Self {
        Self {
            tsn,
            stream_id,
            stream_sequence_number,
            policy,
            first_send_time: Instant::now(),
            retransmission_count: 0,
            abandoned: false,
        }
    }

    /// Check if this message should be abandoned per its policy
    pub fn should_abandon(&self) -> bool {
        if self.abandoned {
            return true;
        }
        self.policy.should_abandon(self.first_send_time, self.retransmission_count)
    }

    /// Increment the retransmission counter
    pub fn retransmit(&mut self) {
        self.retransmission_count += 1;
    }

    /// Mark this message as abandoned
    pub fn abandon(&mut self) {
        self.abandoned = true;
    }
}


/// PR-SCTP tracker for managing partially reliable messages
///
/// Maintains a list of outbound messages with their reliability policies
/// and determines when messages should be abandoned.
#[derive(Debug, Default)]
pub struct PrSctpTracker {
    /// Outstanding messages being tracked
    messages: Vec<PrSctpMessage>,
    /// Current cumulative TSN acknowledged by peer
    cumulative_tsn_ack: u32,
}

impl PrSctpTracker {
    /// Create a new PR-SCTP tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Track a new outbound message
    pub fn track_message(&mut self, msg: PrSctpMessage) {
        self.messages.push(msg);
    }

    /// Update the cumulative TSN acknowledgment
    pub fn update_cumulative_tsn(&mut self, tsn: u32) {
        self.cumulative_tsn_ack = tsn;
        // Remove acknowledged messages
        self.messages.retain(|m| m.tsn > tsn);
    }

    /// Check for messages that should be abandoned and generate Forward TSN
    ///
    /// Returns a `ForwardTsnChunk` if any messages need to be abandoned,
    /// or `None` if all outstanding messages are still within their policy.
    pub fn check_abandonments(&mut self) -> Option<ForwardTsnChunk> {
        let mut highest_abandoned_tsn = self.cumulative_tsn_ack;
        let mut stream_info = Vec::new();
        let mut any_abandoned = false;

        for msg in &mut self.messages {
            if msg.should_abandon() && !msg.abandoned {
                msg.abandon();
                any_abandoned = true;
                if msg.tsn > highest_abandoned_tsn {
                    highest_abandoned_tsn = msg.tsn;
                    stream_info.push(ForwardTsnStreamInfo {
                        stream_id: msg.stream_id,
                        stream_sequence_number: msg.stream_sequence_number,
                    });
                }
            }
        }

        if any_abandoned {
            // Clean up abandoned messages
            self.messages.retain(|m| !m.abandoned);

            let mut fwd = ForwardTsnChunk::new(highest_abandoned_tsn);
            fwd.stream_info = stream_info;
            Some(fwd)
        } else {
            None
        }
    }

    /// Get the number of outstanding tracked messages
    pub fn outstanding_count(&self) -> usize {
        self.messages.len()
    }

    /// Check if there are any partially reliable messages being tracked
    pub fn has_pr_messages(&self) -> bool {
        self.messages.iter().any(|m| !m.policy.is_reliable())
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SctpConfig::default();
        assert_eq!(config.max_outbound_streams, DEFAULT_NUM_STREAMS);
        assert_eq!(config.max_inbound_streams, DEFAULT_NUM_STREAMS);
        assert_eq!(config.max_message_size, DEFAULT_MAX_MESSAGE_SIZE);
        assert_eq!(config.max_receive_buffer_size, DEFAULT_RECEIVE_BUFFER_SIZE);
        assert_eq!(config.connect_timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_ngap_ppid() {
        assert_eq!(NGAP_PPID, 60);
    }

    #[test]
    fn test_association_state() {
        assert_ne!(AssociationState::Closed, AssociationState::Connecting);
        assert_ne!(AssociationState::Connecting, AssociationState::Established);
        assert_ne!(AssociationState::Established, AssociationState::ShuttingDown);
    }

    #[test]
    fn test_received_message() {
        let msg = ReceivedMessage {
            stream_id: 0,
            data: Bytes::from_static(b"test"),
            ppid: NGAP_PPID,
        };
        assert_eq!(msg.stream_id, 0);
        assert_eq!(msg.ppid, NGAP_PPID);
        assert_eq!(&msg.data[..], b"test");
    }

    #[test]
    fn test_sctp_event_variants() {
        let _connected = SctpEvent::Connected;
        let _disconnected = SctpEvent::Disconnected;
        let _stream_opened = SctpEvent::StreamOpened(0);
        let _stream_closed = SctpEvent::StreamClosed(0);
        let _data = SctpEvent::DataReceived(ReceivedMessage {
            stream_id: 0,
            data: Bytes::new(),
            ppid: 0,
        });
    }

    #[test]
    fn test_sctp_error_display() {
        let io_err = SctpError::Io(io::Error::other("test"));
        assert!(io_err.to_string().contains("I/O error"));

        let conn_err = SctpError::ConnectionFailed("test".into());
        assert!(conn_err.to_string().contains("Connection failed"));

        let closed_err = SctpError::AssociationClosed;
        assert!(closed_err.to_string().contains("closed"));
    }

    // =======================================================================
    // A6.1: Multi-homing tests
    // =======================================================================

    fn addr(ip: &str, port: u16) -> SocketAddr {
        format!("{ip}:{port}").parse().unwrap()
    }

    #[test]
    fn test_sctp_path_new() {
        let path = SctpPath::new(addr("10.0.0.1", 38412), addr("10.0.0.2", 38412));
        assert_eq!(path.state, PathState::Active);
        assert!(path.is_usable());
        assert_eq!(path.error_count, 0);
        assert!(path.last_heartbeat_ack.is_none());
    }

    #[test]
    fn test_sctp_path_heartbeat_ack() {
        let mut path = SctpPath::new(addr("10.0.0.1", 38412), addr("10.0.0.2", 38412));
        path.state = PathState::Inactive;
        path.error_count = 3;

        path.heartbeat_ack();
        assert_eq!(path.state, PathState::Active);
        assert_eq!(path.error_count, 0);
        assert!(path.last_heartbeat_ack.is_some());
    }

    #[test]
    fn test_sctp_path_heartbeat_failure() {
        let mut path = SctpPath::new(addr("10.0.0.1", 38412), addr("10.0.0.2", 38412));
        path.max_retransmissions = 3;

        path.heartbeat_failure();
        assert_eq!(path.state, PathState::Inactive);
        assert_eq!(path.error_count, 1);

        path.heartbeat_failure();
        path.heartbeat_failure();
        assert_eq!(path.state, PathState::Failed);
        assert!(!path.is_usable());
    }

    #[test]
    fn test_sctp_path_needs_heartbeat() {
        let path = SctpPath::new(addr("10.0.0.1", 38412), addr("10.0.0.2", 38412));
        // Never sent a heartbeat -> needs one
        assert!(path.needs_heartbeat());
    }

    #[test]
    fn test_sctp_path_mark_heartbeat_sent() {
        let mut path = SctpPath::new(addr("10.0.0.1", 38412), addr("10.0.0.2", 38412));
        path.state = PathState::Inactive;

        path.mark_heartbeat_sent();
        assert_eq!(path.state, PathState::Probing);
        assert!(path.last_heartbeat_sent.is_some());
    }

    #[test]
    fn test_multihoming_config() {
        let config = MultihomingConfig::new(addr("10.0.0.1", 0), addr("10.0.0.2", 38412))
            .with_local_address(addr("10.0.1.1", 0))
            .with_remote_address(addr("10.0.1.2", 38412))
            .with_heartbeat_interval(Duration::from_secs(15))
            .with_max_path_retransmissions(3);

        assert_eq!(config.local_addresses.len(), 2);
        assert_eq!(config.remote_addresses.len(), 2);
        assert_eq!(config.heartbeat_interval, Duration::from_secs(15));
        assert_eq!(config.max_path_retransmissions, 3);
        assert_eq!(config.primary_local(), Some(addr("10.0.0.1", 0)));
        assert_eq!(config.primary_remote(), Some(addr("10.0.0.2", 38412)));

        let paths = config.all_paths();
        assert_eq!(paths.len(), 4); // 2 local x 2 remote
    }

    #[test]
    fn test_multihoming_config_no_duplicates() {
        let config = MultihomingConfig::new(addr("10.0.0.1", 0), addr("10.0.0.2", 38412))
            .with_local_address(addr("10.0.0.1", 0)) // duplicate
            .with_remote_address(addr("10.0.0.2", 38412)); // duplicate

        assert_eq!(config.local_addresses.len(), 1);
        assert_eq!(config.remote_addresses.len(), 1);
    }

    #[test]
    fn test_path_manager_single_path() {
        let mgr = PathManager::single_path(addr("10.0.0.1", 0), addr("10.0.0.2", 38412));
        assert_eq!(mgr.path_count(), 1);
        assert_eq!(mgr.active_path_count(), 1);
        assert!(mgr.primary_path().is_some());
        assert!(mgr.select_path().is_some());
    }

    #[test]
    fn test_path_manager_from_config() {
        let config = MultihomingConfig::new(addr("10.0.0.1", 0), addr("10.0.0.2", 38412))
            .with_local_address(addr("10.0.1.1", 0))
            .with_remote_address(addr("10.0.1.2", 38412));

        let mgr = PathManager::from_config(&config);
        assert_eq!(mgr.path_count(), 4);
        assert_eq!(mgr.active_path_count(), 4);
    }

    #[test]
    fn test_path_manager_failover() {
        let config = MultihomingConfig::new(addr("10.0.0.1", 0), addr("10.0.0.2", 38412))
            .with_remote_address(addr("10.0.1.2", 38412))
            .with_max_path_retransmissions(1);

        let mut mgr = PathManager::from_config(&config);

        // Primary should be first path
        assert_eq!(mgr.primary_path().unwrap().remote_addr, addr("10.0.0.2", 38412));

        // Fail the primary path
        mgr.handle_heartbeat_failure(addr("10.0.0.2", 38412));
        // After 1 failure with max_retransmissions=1, path should be failed and failover triggers
        assert_eq!(mgr.primary_path().unwrap().remote_addr, addr("10.0.1.2", 38412));
    }

    #[test]
    fn test_path_manager_heartbeat_ack() {
        let mut mgr = PathManager::single_path(addr("10.0.0.1", 0), addr("10.0.0.2", 38412));

        mgr.handle_heartbeat_ack(addr("10.0.0.2", 38412));
        let path = mgr.primary_path().unwrap();
        assert_eq!(path.state, PathState::Active);
        assert!(path.last_heartbeat_ack.is_some());
    }

    #[test]
    fn test_path_manager_select_alternate() {
        let config = MultihomingConfig::new(addr("10.0.0.1", 0), addr("10.0.0.2", 38412))
            .with_remote_address(addr("10.0.1.2", 38412));

        let mut mgr = PathManager::from_config(&config);

        // Mark primary as failed (directly)
        mgr.primary_path_mut().unwrap().state = PathState::Failed;

        // select_path should return an alternate
        let selected = mgr.select_path();
        assert!(selected.is_some());
        assert!(selected.unwrap().is_usable());
    }

    #[test]
    fn test_path_manager_set_primary() {
        let config = MultihomingConfig::new(addr("10.0.0.1", 0), addr("10.0.0.2", 38412))
            .with_remote_address(addr("10.0.1.2", 38412));

        let mut mgr = PathManager::from_config(&config);
        assert!(mgr.set_primary(1));
        assert_eq!(mgr.primary_path().unwrap().remote_addr, addr("10.0.1.2", 38412));

        // Invalid index
        assert!(!mgr.set_primary(100));
    }

    #[test]
    fn test_path_manager_paths_needing_heartbeat() {
        let config = MultihomingConfig::new(addr("10.0.0.1", 0), addr("10.0.0.2", 38412))
            .with_remote_address(addr("10.0.1.2", 38412));

        let mgr = PathManager::from_config(&config);

        // All paths need heartbeat initially (never sent)
        let needing = mgr.paths_needing_heartbeat();
        assert_eq!(needing.len(), mgr.path_count());
    }

    #[test]
    fn test_path_manager_mark_heartbeat_sent() {
        let mut mgr = PathManager::single_path(addr("10.0.0.1", 0), addr("10.0.0.2", 38412));
        mgr.mark_heartbeat_sent(0);
        assert!(mgr.primary_path().unwrap().last_heartbeat_sent.is_some());
    }

    // =======================================================================
    // A6.2: PR-SCTP tests
    // =======================================================================

    #[test]
    fn test_partial_reliability_reliable() {
        let policy = PartialReliabilityPolicy::ReliableTransfer;
        assert!(policy.is_reliable());
        // Should never abandon
        assert!(!policy.should_abandon(Instant::now() - Duration::from_secs(3600), 1000));
    }

    #[test]
    fn test_partial_reliability_timed() {
        let policy = PartialReliabilityPolicy::TimedReliability(Duration::from_millis(100));
        assert!(!policy.is_reliable());

        // Not expired yet
        assert!(!policy.should_abandon(Instant::now(), 0));

        // Expired
        assert!(policy.should_abandon(Instant::now() - Duration::from_millis(200), 0));
    }

    #[test]
    fn test_partial_reliability_limited_retransmissions() {
        let policy = PartialReliabilityPolicy::LimitedRetransmissions(3);
        assert!(!policy.is_reliable());

        assert!(!policy.should_abandon(Instant::now(), 2));
        assert!(!policy.should_abandon(Instant::now(), 3));
        assert!(policy.should_abandon(Instant::now(), 4));
    }

    #[test]
    fn test_partial_reliability_default() {
        let policy = PartialReliabilityPolicy::default();
        assert_eq!(policy, PartialReliabilityPolicy::ReliableTransfer);
    }

    #[test]
    fn test_forward_tsn_chunk_encode_decode() {
        let chunk = ForwardTsnChunk::new(100)
            .with_stream_info(0, 5)
            .with_stream_info(1, 3);

        let encoded = chunk.encode();
        let decoded = ForwardTsnChunk::decode(&encoded).unwrap();

        assert_eq!(decoded.new_cumulative_tsn, 100);
        assert_eq!(decoded.stream_info.len(), 2);
        assert_eq!(decoded.stream_info[0].stream_id, 0);
        assert_eq!(decoded.stream_info[0].stream_sequence_number, 5);
        assert_eq!(decoded.stream_info[1].stream_id, 1);
        assert_eq!(decoded.stream_info[1].stream_sequence_number, 3);
    }

    #[test]
    fn test_forward_tsn_chunk_empty() {
        let chunk = ForwardTsnChunk::new(42);
        let encoded = chunk.encode();
        assert_eq!(encoded.len(), 4);

        let decoded = ForwardTsnChunk::decode(&encoded).unwrap();
        assert_eq!(decoded.new_cumulative_tsn, 42);
        assert!(decoded.stream_info.is_empty());
    }

    #[test]
    fn test_forward_tsn_decode_too_short() {
        let result = ForwardTsnChunk::decode(&[0x00, 0x01]);
        assert!(result.is_err());
    }

    #[test]
    fn test_pr_sctp_message() {
        let msg = PrSctpMessage::new(
            1,
            0,
            0,
            PartialReliabilityPolicy::LimitedRetransmissions(2),
        );
        assert_eq!(msg.tsn, 1);
        assert!(!msg.abandoned);
        assert!(!msg.should_abandon());

        let mut msg2 = msg.clone();
        msg2.retransmit();
        msg2.retransmit();
        msg2.retransmit();
        assert!(msg2.should_abandon());
    }

    #[test]
    fn test_pr_sctp_message_abandon() {
        let mut msg = PrSctpMessage::new(1, 0, 0, PartialReliabilityPolicy::ReliableTransfer);
        assert!(!msg.should_abandon());

        msg.abandon();
        assert!(msg.should_abandon());
        assert!(msg.abandoned);
    }

    #[test]
    fn test_pr_sctp_tracker_track_and_ack() {
        let mut tracker = PrSctpTracker::new();

        tracker.track_message(PrSctpMessage::new(
            1, 0, 0, PartialReliabilityPolicy::ReliableTransfer,
        ));
        tracker.track_message(PrSctpMessage::new(
            2, 0, 1, PartialReliabilityPolicy::ReliableTransfer,
        ));

        assert_eq!(tracker.outstanding_count(), 2);

        tracker.update_cumulative_tsn(1);
        assert_eq!(tracker.outstanding_count(), 1);

        tracker.update_cumulative_tsn(2);
        assert_eq!(tracker.outstanding_count(), 0);
    }

    #[test]
    fn test_pr_sctp_tracker_check_abandonments() {
        let mut tracker = PrSctpTracker::new();

        // Add a message with very short timed reliability
        let mut msg = PrSctpMessage::new(
            1,
            0,
            0,
            PartialReliabilityPolicy::LimitedRetransmissions(0),
        );
        // Force one retransmission to trigger abandonment
        msg.retransmit();
        tracker.track_message(msg);

        // Add a reliable message
        tracker.track_message(PrSctpMessage::new(
            2, 0, 1, PartialReliabilityPolicy::ReliableTransfer,
        ));

        assert_eq!(tracker.outstanding_count(), 2);

        let fwd = tracker.check_abandonments();
        assert!(fwd.is_some());
        let fwd = fwd.unwrap();
        assert_eq!(fwd.new_cumulative_tsn, 1);

        // Abandoned message should be removed
        assert_eq!(tracker.outstanding_count(), 1);
    }

    #[test]
    fn test_pr_sctp_tracker_has_pr_messages() {
        let mut tracker = PrSctpTracker::new();
        assert!(!tracker.has_pr_messages());

        tracker.track_message(PrSctpMessage::new(
            1, 0, 0, PartialReliabilityPolicy::ReliableTransfer,
        ));
        assert!(!tracker.has_pr_messages());

        tracker.track_message(PrSctpMessage::new(
            2, 0, 1, PartialReliabilityPolicy::TimedReliability(Duration::from_secs(1)),
        ));
        assert!(tracker.has_pr_messages());
    }

    #[test]
    fn test_pr_sctp_tracker_no_abandonments() {
        let mut tracker = PrSctpTracker::new();

        tracker.track_message(PrSctpMessage::new(
            1, 0, 0, PartialReliabilityPolicy::ReliableTransfer,
        ));

        let fwd = tracker.check_abandonments();
        assert!(fwd.is_none());
    }
}
