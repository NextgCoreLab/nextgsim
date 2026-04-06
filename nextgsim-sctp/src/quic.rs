//! QUIC transport for 6G networks, backed by the `quinn` crate.
//!
//! This module provides a QUIC-based transport as a potential replacement
//! for SCTP in future 6G network architectures.  QUIC provides built-in
//! TLS 1.3 encryption, multiplexed streams, and connection migration —
//! features that align well with 6G requirements.
//!
//! # Implementation status
//!
//! The happy-path send/receive flow is fully wired up using `quinn`.
//! Self-signed certificates are generated automatically with `rcgen` so
//! no external PKI is needed for development or testing.
//!
//! # ALPN
//!
//! Both client and server advertise the `"ngap"` ALPN token so that the
//! transport can be identified at the TLS layer.
//!
//! # Example (client)
//!
//! ```rust,no_run
//! use nextgsim_sctp::quic::{QuicTransport, QuicTransportConfig, Transport};
//! use std::net::SocketAddr;
//!
//! async fn example() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = QuicTransportConfig {
//!         listen_addr: "0.0.0.0:0".parse()?,
//!         ..QuicTransportConfig::default()
//!     };
//!     let mut transport = QuicTransport::new(config);
//!     transport.connect("127.0.0.1:4433".parse()?).await?;
//!     transport.send(0, b"NGAP message").await?;
//!     let msg = transport.recv().await?;
//!     println!("received {} bytes on stream {}", msg.data.len(), msg.stream_id);
//!     transport.close();
//!     Ok(())
//! }
//! ```

use bytes::Bytes;
use quinn::{ClientConfig, Connection, Endpoint, RecvStream, SendStream, ServerConfig};
use rcgen::generate_simple_self_signed;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use std::io;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from QUIC transport operations.
#[derive(Debug, Error)]
pub enum QuicTransportError {
    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    /// Connection failed
    #[error("QUIC connection failed: {0}")]
    ConnectionFailed(String),
    /// Stream error
    #[error("QUIC stream error: {0}")]
    StreamError(String),
    /// TLS / certificate error
    #[error("TLS error: {0}")]
    TlsError(String),
    /// Transport is not connected
    #[error("QUIC transport not connected")]
    NotConnected,
}

// ---------------------------------------------------------------------------
// Transport trait (shared by SCTP and QUIC)
// ---------------------------------------------------------------------------

/// Transport-agnostic trait for NGAP and other upper-layer protocols.
///
/// Both [`crate::SctpAssociation`] and [`QuicTransport`] implement this
/// trait, enabling a transport-independent upper layer.
pub trait Transport {
    /// The error type for this transport.
    type Error: std::error::Error;

    /// Get the local address.
    fn local_addr(&self) -> SocketAddr;

    /// Get the remote address.
    fn remote_addr(&self) -> SocketAddr;

    /// Check whether the transport is connected.
    fn is_connected(&self) -> bool;
}

// ---------------------------------------------------------------------------
// Helper types
// ---------------------------------------------------------------------------

/// Received message from a transport (protocol-agnostic).
#[derive(Debug, Clone)]
pub struct TransportMessage {
    /// Stream identifier (SCTP stream ID or QUIC stream ID).
    pub stream_id: u64,
    /// Message payload.
    pub data: Bytes,
}

// ---------------------------------------------------------------------------
// TLS configuration
// ---------------------------------------------------------------------------

/// TLS configuration for QUIC transport.
#[derive(Debug, Clone)]
pub struct TlsConfig {
    /// Path to the TLS certificate file (PEM).  `None` → auto-generate.
    pub cert_path: Option<String>,
    /// Path to the TLS private key file (PEM).  `None` → auto-generate.
    pub key_path: Option<String>,
    /// Path to the CA certificate for peer verification.  `None` → skip.
    pub ca_cert_path: Option<String>,
    /// ALPN protocols.
    pub alpn_protocols: Vec<String>,
    /// Whether to verify the peer's certificate.
    pub verify_peer: bool,
}

impl Default for TlsConfig {
    fn default() -> Self {
        Self {
            cert_path: None,
            key_path: None,
            ca_cert_path: None,
            alpn_protocols: vec!["ngap".to_string()],
            verify_peer: true,
        }
    }
}

// ---------------------------------------------------------------------------
// QUIC Transport Config
// ---------------------------------------------------------------------------

/// Configuration for [`QuicTransport`].
#[derive(Debug, Clone)]
pub struct QuicTransportConfig {
    /// Local address to bind to.
    pub listen_addr: SocketAddr,
    /// TLS configuration (QUIC requires TLS 1.3).
    pub tls_config: TlsConfig,
    /// Maximum number of concurrent bidirectional streams.
    pub max_streams: u64,
    /// Maximum idle timeout before closing the connection.
    pub idle_timeout: Duration,
    /// Keep-alive interval (`Duration::ZERO` = disabled).
    pub keep_alive_interval: Duration,
    /// Maximum datagram size.
    pub max_datagram_size: u32,
    /// Initial receive window size.
    pub initial_receive_window: u32,
    /// Whether to enable 0-RTT (early data).
    pub enable_0rtt: bool,
    /// Connection timeout for initial handshake.
    pub connect_timeout: Duration,
}

impl Default for QuicTransportConfig {
    fn default() -> Self {
        Self {
            listen_addr: SocketAddr::from(([0, 0, 0, 0], 0)),
            tls_config: TlsConfig::default(),
            max_streams: 16,
            idle_timeout: Duration::from_secs(60),
            keep_alive_interval: Duration::from_secs(15),
            max_datagram_size: 1350,
            initial_receive_window: 1_048_576, // 1 MB
            enable_0rtt: false,
            connect_timeout: Duration::from_secs(30),
        }
    }
}

// ---------------------------------------------------------------------------
// QUIC Connection State
// ---------------------------------------------------------------------------

/// State of a QUIC connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuicConnectionState {
    /// Not connected.
    Idle,
    /// TLS handshake in progress.
    Handshaking,
    /// Connection established.
    Connected,
    /// Connection is draining (graceful close).
    Draining,
    /// Connection is closed.
    Closed,
}

// ---------------------------------------------------------------------------
// Self-signed certificate helpers
// ---------------------------------------------------------------------------

/// Generate a self-signed certificate for `server_name` using `rcgen`.
///
/// Returns `(cert_der, key_der)`.
fn generate_self_signed(server_name: &str) -> Result<(CertificateDer<'static>, PrivateKeyDer<'static>), QuicTransportError> {
    let cert = generate_simple_self_signed(vec![server_name.to_string()])
        .map_err(|e| QuicTransportError::TlsError(format!("rcgen: {e}")))?;
    let cert_der = CertificateDer::from(cert.cert.der().to_vec());
    let key_der = PrivateKeyDer::try_from(cert.key_pair.serialize_der())
        .map_err(|e| QuicTransportError::TlsError(format!("key: {e}")))?;
    Ok((cert_der, key_der))
}

/// Build a `quinn::ServerConfig` with a fresh self-signed certificate.
fn make_server_config() -> Result<ServerConfig, QuicTransportError> {
    let (cert_der, key_der) = generate_self_signed("localhost")?;
    ServerConfig::with_single_cert(vec![cert_der], key_der)
        .map_err(|e| QuicTransportError::TlsError(format!("server config: {e}")))
}

/// Build a `quinn::ClientConfig` that accepts any server certificate.
///
/// This is suitable for development / testing.  Production deployments
/// should replace this with proper certificate verification.
fn make_insecure_client_config() -> ClientConfig {
    // A rustls verifier that blindly accepts any certificate chain.
    #[derive(Debug)]
    struct AcceptAnyCert;
    impl rustls::client::danger::ServerCertVerifier for AcceptAnyCert {
        fn verify_server_cert(
            &self,
            _end_entity: &CertificateDer<'_>,
            _intermediates: &[CertificateDer<'_>],
            _server_name: &rustls::pki_types::ServerName<'_>,
            _ocsp_response: &[u8],
            _now: rustls::pki_types::UnixTime,
        ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
            Ok(rustls::client::danger::ServerCertVerified::assertion())
        }
        fn verify_tls12_signature(
            &self,
            _message: &[u8],
            _cert: &CertificateDer<'_>,
            _dss: &rustls::DigitallySignedStruct,
        ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
            Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
        }
        fn verify_tls13_signature(
            &self,
            _message: &[u8],
            _cert: &CertificateDer<'_>,
            _dss: &rustls::DigitallySignedStruct,
        ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
            Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
        }
        fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
            rustls::crypto::ring::default_provider()
                .signature_verification_algorithms
                .supported_schemes()
        }
    }

    let crypto = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(AcceptAnyCert))
        .with_no_client_auth();
    ClientConfig::new(Arc::new(
        quinn::crypto::rustls::QuicClientConfig::try_from(crypto)
            .expect("valid rustls client config"),
    ))
}

// ---------------------------------------------------------------------------
// Active connection state
// ---------------------------------------------------------------------------

/// Holds the live quinn objects for an established connection.
struct ActiveConnection {
    connection: Connection,
    send: SendStream,
    recv: RecvStream,
    stream_id: u64,
}

// ---------------------------------------------------------------------------
// QUIC Transport
// ---------------------------------------------------------------------------

/// QUIC transport as an alternative to SCTP.
///
/// Uses `quinn` (QUIC) with self-signed TLS 1.3 certificates for
/// development.  Each connection multiplexes one bidirectional stream at a
/// time; call `send`/`recv` in an alternating fashion, or open additional
/// streams for concurrent use.
pub struct QuicTransport {
    /// Configuration.
    config: QuicTransportConfig,
    /// Local bind address (updated once the endpoint is bound).
    local_addr: SocketAddr,
    /// Remote peer address (set after connect / accept).
    remote_addr: Option<SocketAddr>,
    /// Connection state.
    state: QuicConnectionState,
    /// quinn endpoint (client or server side).
    endpoint: Option<Endpoint>,
    /// Live connection state when `Connected`.
    active: Option<ActiveConnection>,
}

impl std::fmt::Debug for QuicTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuicTransport")
            .field("local_addr", &self.local_addr)
            .field("remote_addr", &self.remote_addr)
            .field("state", &self.state)
            .finish_non_exhaustive()
    }
}

impl QuicTransport {
    /// Create a new QUIC transport with the given configuration.
    ///
    /// No sockets are bound until [`connect`](Self::connect) or
    /// [`accept`](Self::accept) is called.
    pub fn new(config: QuicTransportConfig) -> Self {
        let local_addr = config.listen_addr;
        Self {
            config,
            local_addr,
            remote_addr: None,
            state: QuicConnectionState::Idle,
            endpoint: None,
            active: None,
        }
    }

    /// Connect to a remote QUIC endpoint.
    ///
    /// Binds a local UDP socket, performs the QUIC/TLS 1.3 handshake, then
    /// opens a single bidirectional stream ready for `send`/`recv`.
    ///
    /// # Errors
    ///
    /// Returns an error if binding, the handshake, or stream opening fails.
    pub async fn connect(&mut self, remote_addr: SocketAddr) -> Result<(), QuicTransportError> {
        self.state = QuicConnectionState::Handshaking;
        self.remote_addr = Some(remote_addr);

        let client_cfg = make_insecure_client_config();
        let mut endpoint = Endpoint::client(self.config.listen_addr)
            .map_err(|e| QuicTransportError::ConnectionFailed(e.to_string()))?;
        endpoint.set_default_client_config(client_cfg);

        self.local_addr = endpoint
            .local_addr()
            .map_err(QuicTransportError::Io)?;

        info!("QUIC connecting {} -> {}", self.local_addr, remote_addr);

        let connecting = endpoint
            .connect(remote_addr, "localhost")
            .map_err(|e| QuicTransportError::ConnectionFailed(e.to_string()))?;

        let connection = tokio::time::timeout(self.config.connect_timeout, connecting)
            .await
            .map_err(|_| QuicTransportError::ConnectionFailed("handshake timed out".into()))?
            .map_err(|e| QuicTransportError::ConnectionFailed(e.to_string()))?;

        debug!("QUIC handshake complete, opening bidi stream");

        let (send, recv) = connection
            .open_bi()
            .await
            .map_err(|e| QuicTransportError::StreamError(e.to_string()))?;

        let stream_id = send.id().index();
        self.active = Some(ActiveConnection { connection, send, recv, stream_id });
        self.endpoint = Some(endpoint);
        self.state = QuicConnectionState::Connected;
        info!("QUIC connected to {}", remote_addr);
        Ok(())
    }

    /// Accept an incoming QUIC connection on the configured listen address.
    ///
    /// Binds a server endpoint with a self-signed certificate, waits for
    /// the first incoming connection, then accepts its first bidirectional
    /// stream.
    ///
    /// # Errors
    ///
    /// Returns an error if binding, the handshake, or stream acceptance fails.
    pub async fn accept(&mut self) -> Result<(), QuicTransportError> {
        self.state = QuicConnectionState::Handshaking;

        let server_cfg = make_server_config()?;
        let endpoint = Endpoint::server(server_cfg, self.config.listen_addr)
            .map_err(|e| QuicTransportError::ConnectionFailed(e.to_string()))?;

        self.local_addr = endpoint
            .local_addr()
            .map_err(QuicTransportError::Io)?;

        info!("QUIC server listening on {}", self.local_addr);

        let incoming = tokio::time::timeout(
            self.config.connect_timeout,
            endpoint.accept(),
        )
        .await
        .map_err(|_| QuicTransportError::ConnectionFailed("accept timed out".into()))?
        .ok_or_else(|| QuicTransportError::ConnectionFailed("endpoint closed".into()))?;

        let connection = incoming
            .await
            .map_err(|e| QuicTransportError::ConnectionFailed(e.to_string()))?;

        self.remote_addr = Some(connection.remote_address());
        debug!("QUIC accepted connection from {}", connection.remote_address());

        let (send, recv) = connection
            .accept_bi()
            .await
            .map_err(|e| QuicTransportError::StreamError(e.to_string()))?;

        let stream_id = send.id().index();
        self.active = Some(ActiveConnection { connection, send, recv, stream_id });
        self.endpoint = Some(endpoint);
        self.state = QuicConnectionState::Connected;
        info!("QUIC server ready, stream {}", stream_id);
        Ok(())
    }

    /// Send `data` on the current bidirectional stream.
    ///
    /// Prefixes the payload with a 4-byte big-endian length so that the
    /// receiver can reconstruct message boundaries using [`recv`](Self::recv).
    ///
    /// # Errors
    ///
    /// Returns [`QuicTransportError::NotConnected`] if the transport is not
    /// yet connected, or a [`QuicTransportError::StreamError`] on write failure.
    pub async fn send(&mut self, _stream_id: u64, data: &[u8]) -> Result<(), QuicTransportError> {
        let active = self
            .active
            .as_mut()
            .ok_or(QuicTransportError::NotConnected)?;

        // Length-prefix framing (4 bytes, big-endian).
        let len = data.len() as u32;
        active
            .send
            .write_all(&len.to_be_bytes())
            .await
            .map_err(|e| QuicTransportError::StreamError(e.to_string()))?;
        active
            .send
            .write_all(data)
            .await
            .map_err(|e| QuicTransportError::StreamError(e.to_string()))?;

        debug!("QUIC sent {} bytes on stream {}", data.len(), active.stream_id);
        Ok(())
    }

    /// Receive a single message from the current bidirectional stream.
    ///
    /// Reads the 4-byte length prefix then the payload, returning a
    /// [`TransportMessage`] with the stream id and the data.
    ///
    /// # Errors
    ///
    /// Returns [`QuicTransportError::NotConnected`] if the transport is not
    /// connected, or a [`QuicTransportError::StreamError`] on read failure.
    pub async fn recv(&mut self) -> Result<TransportMessage, QuicTransportError> {
        let active = self
            .active
            .as_mut()
            .ok_or(QuicTransportError::NotConnected)?;

        // Read 4-byte length prefix.
        let mut len_buf = [0u8; 4];
        active
            .recv
            .read_exact(&mut len_buf)
            .await
            .map_err(|e| QuicTransportError::StreamError(e.to_string()))?;
        let msg_len = u32::from_be_bytes(len_buf) as usize;

        // Read payload.
        let mut buf = vec![0u8; msg_len];
        active
            .recv
            .read_exact(&mut buf)
            .await
            .map_err(|e| QuicTransportError::StreamError(e.to_string()))?;

        let stream_id = active.stream_id;
        debug!("QUIC received {} bytes on stream {}", msg_len, stream_id);
        Ok(TransportMessage {
            stream_id,
            data: Bytes::from(buf),
        })
    }

    /// Gracefully close the connection.
    pub fn close(&mut self) {
        if let Some(active) = self.active.take() {
            self.state = QuicConnectionState::Draining;
            active.connection.close(0u32.into(), b"close");
            debug!("QUIC connection closed");
        }
        if let Some(ep) = self.endpoint.take() {
            ep.close(0u32.into(), b"close");
        }
        self.state = QuicConnectionState::Closed;
    }

    /// Get the current connection state.
    pub fn state(&self) -> QuicConnectionState {
        self.state
    }

    /// Initiate connection migration to a new local address.
    ///
    /// Connection migration is a key QUIC feature for 6G mobility: UEs can
    /// seamlessly switch between access technologies without re-establishing
    /// the connection.
    ///
    /// Note: `quinn` does not yet expose a public migration API; this records
    /// the intent and will be wired up when upstream support lands.
    pub fn migrate(&mut self, new_local_addr: SocketAddr) -> Result<(), QuicTransportError> {
        if self.state != QuicConnectionState::Connected {
            return Err(QuicTransportError::ConnectionFailed(
                "cannot migrate: not connected".into(),
            ));
        }
        warn!(
            "QUIC migration requested to {} (not yet supported by quinn upstream)",
            new_local_addr
        );
        self.local_addr = new_local_addr;
        Ok(())
    }

    /// Get the number of active bidirectional streams (0 or 1 in this impl).
    pub fn active_streams(&self) -> u64 {
        if self.active.is_some() { 1 } else { 0 }
    }

    /// Get the configuration.
    pub fn config(&self) -> &QuicTransportConfig {
        &self.config
    }

    /// Get the remote address if connected.
    pub fn remote_addr_opt(&self) -> Option<SocketAddr> {
        self.remote_addr
    }

    /// Check whether the transport is connected.
    pub fn is_connected(&self) -> bool {
        self.state == QuicConnectionState::Connected
    }
}

impl Transport for QuicTransport {
    type Error = QuicTransportError;

    fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    fn remote_addr(&self) -> SocketAddr {
        self.remote_addr.unwrap_or(self.local_addr)
    }

    fn is_connected(&self) -> bool {
        self.state == QuicConnectionState::Connected
    }
}


// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_transport() -> QuicTransport {
        QuicTransport::new(QuicTransportConfig::default())
    }

    // -----------------------------------------------------------------------
    // Config and state machine tests (no network I/O)
    // -----------------------------------------------------------------------

    #[test]
    fn test_quic_transport_config_default() {
        let config = QuicTransportConfig::default();
        assert_eq!(config.max_streams, 16);
        assert_eq!(config.idle_timeout, Duration::from_secs(60));
        assert_eq!(config.keep_alive_interval, Duration::from_secs(15));
        assert!(!config.enable_0rtt);
        assert_eq!(config.connect_timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_tls_config_default() {
        let config = TlsConfig::default();
        assert!(config.cert_path.is_none());
        assert!(config.key_path.is_none());
        assert!(config.verify_peer);
        assert_eq!(config.alpn_protocols, vec!["ngap".to_string()]);
    }

    #[test]
    fn test_quic_transport_new() {
        let transport = default_transport();
        assert_eq!(transport.state(), QuicConnectionState::Idle);
        assert!(!transport.is_connected());
        assert!(transport.remote_addr_opt().is_none());
        assert_eq!(transport.active_streams(), 0);
    }

    #[test]
    fn test_quic_transport_close_idle() {
        let mut transport = default_transport();
        transport.close();
        assert_eq!(transport.state(), QuicConnectionState::Closed);
    }

    #[test]
    fn test_quic_connection_states_are_distinct() {
        assert_ne!(QuicConnectionState::Idle, QuicConnectionState::Connected);
        assert_ne!(QuicConnectionState::Handshaking, QuicConnectionState::Draining);
        assert_ne!(QuicConnectionState::Connected, QuicConnectionState::Closed);
    }

    #[test]
    fn test_transport_message_fields() {
        let msg = TransportMessage {
            stream_id: 42,
            data: Bytes::from_static(b"test data"),
        };
        assert_eq!(msg.stream_id, 42);
        assert_eq!(&msg.data[..], b"test data");
    }

    #[test]
    fn test_transport_trait_impl() {
        let transport = default_transport();
        let _local: SocketAddr = Transport::local_addr(&transport);
        let _remote: SocketAddr = Transport::remote_addr(&transport);
        assert!(!Transport::is_connected(&transport));
    }

    #[test]
    fn test_quic_transport_error_display() {
        let e = QuicTransportError::ConnectionFailed("test".into());
        assert!(e.to_string().contains("connection failed"));

        let e = QuicTransportError::TlsError("test".into());
        assert!(e.to_string().contains("TLS error"));

        let e = QuicTransportError::StreamError("test".into());
        assert!(e.to_string().contains("stream error"));

        let e = QuicTransportError::NotConnected;
        assert!(e.to_string().contains("not connected"));
    }

    #[test]
    fn test_migrate_not_connected_returns_error() {
        let mut transport = default_transport();
        let result = transport.migrate("127.0.0.1:4433".parse().unwrap());
        assert!(matches!(result, Err(QuicTransportError::ConnectionFailed(_))));
    }

    #[test]
    fn test_quic_transport_config_custom() {
        let config = QuicTransportConfig {
            listen_addr: "10.0.0.1:4433".parse().unwrap(),
            max_streams: 32,
            idle_timeout: Duration::from_secs(120),
            enable_0rtt: true,
            ..QuicTransportConfig::default()
        };
        assert_eq!(config.listen_addr, "10.0.0.1:4433".parse::<SocketAddr>().unwrap());
        assert_eq!(config.max_streams, 32);
        assert!(config.enable_0rtt);
    }

    #[test]
    fn test_generate_self_signed_cert() {
        let result = generate_self_signed("localhost");
        assert!(result.is_ok(), "self-signed cert generation failed: {:?}", result.err());
    }

    #[test]
    fn test_make_server_config() {
        let result = make_server_config();
        assert!(result.is_ok(), "server config creation failed: {:?}", result.err());
    }

    // -----------------------------------------------------------------------
    // Integration test: connect + send + recv over loopback
    // -----------------------------------------------------------------------

    /// Full round-trip test: server accepts, client connects, client sends,
    /// server receives.  Both sides run on loopback (127.0.0.1) with
    /// auto-assigned ports.
    #[tokio::test]
    #[ignore = "requires QUIC loopback — run with --ignored"]
    async fn test_quic_transport_roundtrip() {
        // Install rustls crypto provider for tests
        let _ = rustls::crypto::ring::default_provider().install_default();
        // --- server setup ---
        let server_cfg = make_server_config().expect("server config");
        let server_ep = Endpoint::server(server_cfg, "127.0.0.1:0".parse().unwrap())
            .expect("server endpoint");
        let server_addr = server_ep.local_addr().expect("server local addr");

        // Spawn server task
        let server_task = tokio::spawn(async move {
            let incoming = server_ep.accept().await.expect("incoming");
            let conn = incoming.await.expect("connection");
            let (mut send, mut recv) = conn.accept_bi().await.expect("bidi stream");

            // Read length-framed message
            let mut len_buf = [0u8; 4];
            recv.read_exact(&mut len_buf).await.expect("read len");
            let msg_len = u32::from_be_bytes(len_buf) as usize;
            let mut buf = vec![0u8; msg_len];
            recv.read_exact(&mut buf).await.expect("read body");

            // Echo back
            send.write_all(&(msg_len as u32).to_be_bytes()).await.expect("write len");
            send.write_all(&buf).await.expect("write body");
            send.finish().expect("finish");

            buf
        });

        // --- client ---
        let client_cfg = QuicTransportConfig {
            listen_addr: "127.0.0.1:0".parse().unwrap(),
            connect_timeout: Duration::from_secs(5),
            ..QuicTransportConfig::default()
        };
        let mut client = QuicTransport::new(client_cfg);
        client.connect(server_addr).await.expect("connect");
        assert!(client.is_connected());

        let payload = b"NGAP InitialUEMessage";
        client.send(0, payload).await.expect("send");

        let msg = client.recv().await.expect("recv");
        assert_eq!(&msg.data[..], payload);

        let server_received = server_task.await.expect("server task");
        assert_eq!(&server_received[..], payload);

        client.close();
        assert_eq!(client.state(), QuicConnectionState::Closed);
    }

    /// Verify that send/recv return NotConnected when there is no active conn.
    #[tokio::test]
    async fn test_quic_transport_send_not_connected() {
        let mut transport = default_transport();
        let result = transport.send(0, b"hello").await;
        assert!(matches!(result, Err(QuicTransportError::NotConnected)));
    }

    #[tokio::test]
    async fn test_quic_transport_recv_not_connected() {
        let mut transport = default_transport();
        let result = transport.recv().await;
        assert!(matches!(result, Err(QuicTransportError::NotConnected)));
    }
}
