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
        
        let mut stream = self.association.open_stream(stream_id, ppi)
            .map_err(|e| SctpError::StreamError(e.to_string()))?;

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

    /// Try to receive a message (non-blocking)
    pub fn try_recv(&mut self) -> Result<Option<ReceivedMessage>> {
        // Accept any incoming streams
        while let Some(mut stream) = self.association.accept_stream() {
            let stream_id = stream.stream_identifier();
            debug!("Accepted stream {}", stream_id);

            if let Some(tx) = &self.event_tx {
                let _ = tx.send(SctpEvent::StreamOpened(stream_id));
            }

            // Try to read from the stream
            if let Ok(Some(chunks)) = stream.read() {
                let ppid = match chunks.ppi { PayloadProtocolIdentifier::Dcep => 50, PayloadProtocolIdentifier::String => 51, PayloadProtocolIdentifier::Binary => 53, PayloadProtocolIdentifier::StringEmpty => 56, PayloadProtocolIdentifier::BinaryEmpty => 57, PayloadProtocolIdentifier::Unknown => NGAP_PPID, };
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
            match timeout(Duration::from_millis(100), self.handle_incoming()).await {
                Ok(Ok(())) => {}
                Ok(Err(_)) | Err(_) => {}
            }
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

    /// Run the association event loop (for use with tokio::spawn)
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
        let io_err = SctpError::Io(io::Error::new(io::ErrorKind::Other, "test"));
        assert!(io_err.to_string().contains("I/O error"));

        let conn_err = SctpError::ConnectionFailed("test".into());
        assert!(conn_err.to_string().contains("Connection failed"));

        let closed_err = SctpError::AssociationClosed;
        assert!(closed_err.to_string().contains("closed"));
    }
}
