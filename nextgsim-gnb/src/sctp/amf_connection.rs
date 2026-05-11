//! AMF Connection Management
//!
//! This module manages SCTP connections to AMF (Access and Mobility Management Function).
//! Each AMF connection is represented by an `AmfConnection` which wraps an SCTP association.
//!
//! When `AmfConnectionConfig::secondary_addresses` is non-empty a
//! `MultihomeSctpAssociation` is used so the transport can fail over to an
//! alternate path without tearing down the NGAP session.

use std::net::SocketAddr;
use tracing::{debug, info, warn};

use nextgsim_common::OctetString;
use nextgsim_sctp::{
    MultihomeSctpAssociation, MultihomingConfig, ReceivedMessage, SctpAssociation, SctpConfig,
    SctpError,
};

// ---------------------------------------------------------------------------
// Internal association abstraction
// ---------------------------------------------------------------------------

/// Thin enum that lets `AmfConnection` hold either a plain SCTP association
/// or a multi-homed one without boxing.  Both variants expose the same
/// async API (`send`, `recv`, `shutdown`, `poll`, `try_recv`).
enum AssociationKind {
    Single(SctpAssociation),
    Multihome(MultihomeSctpAssociation),
}

impl AssociationKind {
    async fn send(&mut self, stream_id: u16, data: &[u8]) -> Result<(), SctpError> {
        match self {
            Self::Single(a) => a.send(stream_id, data).await,
            Self::Multihome(a) => a.send(stream_id, data).await,
        }
    }

    async fn recv(&mut self) -> Result<Option<ReceivedMessage>, SctpError> {
        match self {
            Self::Single(a) => a.recv().await,
            Self::Multihome(a) => a.recv().await,
        }
    }

    async fn shutdown(&mut self) -> Result<(), SctpError> {
        match self {
            Self::Single(a) => a.shutdown().await,
            Self::Multihome(a) => a.shutdown().await,
        }
    }

    async fn poll(&mut self) -> Result<(), SctpError> {
        match self {
            Self::Single(a) => a.poll().await,
            Self::Multihome(a) => a.poll().await,
        }
    }

    fn try_recv(&mut self) -> Result<Option<ReceivedMessage>, SctpError> {
        match self {
            Self::Single(a) => a.try_recv(),
            Self::Multihome(a) => a.try_recv(),
        }
    }

    fn local_addr(&self) -> SocketAddr {
        match self {
            Self::Single(a) => a.local_addr(),
            Self::Multihome(a) => a.local_addr(),
        }
    }
}

// ---------------------------------------------------------------------------

/// AMF connection state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmfConnectionState {
    /// Connection is being established
    Connecting,
    /// Connection is established and ready
    Connected,
    /// Connection is being closed
    Closing,
    /// Connection is closed
    Closed,
}

/// Events from an AMF connection
#[derive(Debug)]
pub enum AmfConnectionEvent {
    /// Association has been established
    AssociationUp {
        /// Client ID
        client_id: i32,
        /// Association ID (from SCTP)
        association_id: i32,
        /// Number of inbound streams
        in_streams: u16,
        /// Number of outbound streams
        out_streams: u16,
    },
    /// Association has been closed
    AssociationDown {
        /// Client ID
        client_id: i32,
    },
    /// Received NGAP PDU from AMF
    MessageReceived {
        /// Client ID
        client_id: i32,
        /// Stream ID
        stream: u16,
        /// Message data
        data: OctetString,
    },
    /// Unhandled SCTP notification
    UnhandledNotification {
        /// Client ID
        client_id: i32,
    },
}

/// Configuration for AMF connection
#[derive(Debug, Clone)]
pub struct AmfConnectionConfig {
    /// Local address to bind
    pub local_address: SocketAddr,
    /// Remote AMF address (primary)
    pub remote_address: SocketAddr,
    /// Secondary remote addresses for SCTP multi-homing.
    ///
    /// When non-empty a `MultihomeSctpAssociation` is used.  Each entry must
    /// be a valid `SocketAddr` string (e.g. `"192.168.1.2:38412"`).
    pub secondary_addresses: Vec<SocketAddr>,
    /// SCTP configuration
    pub sctp_config: SctpConfig,
}

impl Default for AmfConnectionConfig {
    fn default() -> Self {
        Self {
            local_address: "0.0.0.0:0".parse().expect("value expected"),
            remote_address: "127.0.0.1:38412".parse().expect("value expected"),
            secondary_addresses: Vec::new(),
            sctp_config: SctpConfig::default(),
        }
    }
}

/// Represents a connection to an AMF
pub struct AmfConnection {
    /// Client ID for this connection
    client_id: i32,
    /// SCTP association (single-homed or multi-homed)
    association: Option<AssociationKind>,
    /// Connection state
    state: AmfConnectionState,
    /// Remote AMF address
    remote_address: SocketAddr,
    /// Local address
    local_address: SocketAddr,
    /// Association ID (assigned after connection)
    association_id: Option<i32>,
    /// Number of inbound streams
    in_streams: u16,
    /// Number of outbound streams
    out_streams: u16,
}

impl AmfConnection {
    /// Creates a new AMF connection (not yet connected)
    pub fn new(client_id: i32, config: AmfConnectionConfig) -> Self {
        Self {
            client_id,
            association: None,
            state: AmfConnectionState::Closed,
            remote_address: config.remote_address,
            local_address: config.local_address,
            association_id: None,
            in_streams: 0,
            out_streams: 0,
        }
    }

    /// Establishes the SCTP connection to the AMF.
    ///
    /// When `AmfConnectionConfig::secondary_addresses` is non-empty a
    /// `MultihomeSctpAssociation` is used, enabling automatic path failover.
    pub async fn connect(
        &mut self,
        config: &AmfConnectionConfig,
    ) -> Result<AmfConnectionEvent, SctpError> {
        info!(
            "Connecting to AMF at {} (client_id: {}, multi-homed: {})",
            self.remote_address,
            self.client_id,
            !config.secondary_addresses.is_empty(),
        );

        self.state = AmfConnectionState::Connecting;

        let kind = if config.secondary_addresses.is_empty() {
            // Plain single-homed SCTP association.
            let association = SctpAssociation::connect_with_local(
                self.local_address,
                self.remote_address,
                config.sctp_config.clone(),
            )
            .await?;
            self.local_address = association.local_addr();
            AssociationKind::Single(association)
        } else {
            // Multi-homed SCTP association.
            let mut mh_config = MultihomingConfig::new(self.local_address, self.remote_address);
            for &secondary in &config.secondary_addresses {
                warn!(
                    "SCTP multi-homing: registering secondary AMF address {} (client_id: {})",
                    secondary, self.client_id
                );
                mh_config = mh_config.with_remote_address(secondary);
            }
            let association =
                MultihomeSctpAssociation::connect(mh_config, config.sctp_config.clone()).await?;
            self.local_address = association.local_addr();
            AssociationKind::Multihome(association)
        };

        // Use config values; sctp-proto doesn't expose negotiated counts directly.
        self.in_streams = config.sctp_config.max_inbound_streams;
        self.out_streams = config.sctp_config.max_outbound_streams;
        self.association_id = Some(self.client_id);

        self.association = Some(kind);
        self.state = AmfConnectionState::Connected;

        info!(
            "Connected to AMF at {} (client_id: {}, in_streams: {}, out_streams: {})",
            self.remote_address, self.client_id, self.in_streams, self.out_streams
        );

        Ok(AmfConnectionEvent::AssociationUp {
            client_id: self.client_id,
            association_id: self.association_id.unwrap_or(0),
            in_streams: self.in_streams,
            out_streams: self.out_streams,
        })
    }

    /// Sends data to the AMF on the specified stream.
    pub async fn send(&mut self, stream: u16, data: &[u8]) -> Result<(), SctpError> {
        if self.state != AmfConnectionState::Connected {
            return Err(SctpError::InvalidState(
                "Cannot send: not connected".to_string(),
            ));
        }

        if let Some(ref mut kind) = self.association {
            debug!(
                "Sending {} bytes to AMF on stream {} (client_id: {})",
                data.len(),
                stream,
                self.client_id
            );
            kind.send(stream, data).await
        } else {
            Err(SctpError::AssociationClosed)
        }
    }

    /// Receives a message from the AMF (blocking).
    pub async fn recv(&mut self) -> Result<Option<AmfConnectionEvent>, SctpError> {
        if self.state == AmfConnectionState::Closed {
            return Ok(None);
        }

        if let Some(ref mut kind) = self.association {
            match kind.recv().await {
                Ok(Some(msg)) => {
                    debug!(
                        "Received {} bytes from AMF on stream {} (client_id: {})",
                        msg.data.len(),
                        msg.stream_id,
                        self.client_id
                    );
                    Ok(Some(AmfConnectionEvent::MessageReceived {
                        client_id: self.client_id,
                        stream: msg.stream_id,
                        data: OctetString::from_slice(&msg.data),
                    }))
                }
                Ok(None) => {
                    self.state = AmfConnectionState::Closed;
                    Ok(Some(AmfConnectionEvent::AssociationDown {
                        client_id: self.client_id,
                    }))
                }
                Err(SctpError::AssociationClosed) => {
                    self.state = AmfConnectionState::Closed;
                    Ok(Some(AmfConnectionEvent::AssociationDown {
                        client_id: self.client_id,
                    }))
                }
                Err(e) => Err(e),
            }
        } else {
            Ok(None)
        }
    }

    /// Tries to receive a message without blocking.
    ///
    /// Polls the underlying SCTP association for incoming UDP packets, then
    /// returns any buffered message.
    pub async fn try_recv(&mut self) -> Result<Option<AmfConnectionEvent>, SctpError> {
        if self.state == AmfConnectionState::Closed {
            return Ok(None);
        }

        if let Some(ref mut kind) = self.association {
            // Drive the state machine with any pending UDP datagrams.
            kind.poll().await?;

            match kind.try_recv() {
                Ok(Some(msg)) => {
                    debug!(
                        "Received {} bytes from AMF on stream {} (client_id: {})",
                        msg.data.len(),
                        msg.stream_id,
                        self.client_id
                    );
                    Ok(Some(AmfConnectionEvent::MessageReceived {
                        client_id: self.client_id,
                        stream: msg.stream_id,
                        data: OctetString::from_slice(&msg.data),
                    }))
                }
                Ok(None) => Ok(None),
                Err(e) => Err(e),
            }
        } else {
            Ok(None)
        }
    }

    /// Gracefully closes the connection.
    pub async fn close(&mut self) -> Result<(), SctpError> {
        if self.state == AmfConnectionState::Closed {
            return Ok(());
        }

        info!("Closing AMF connection (client_id: {})", self.client_id);
        self.state = AmfConnectionState::Closing;

        if let Some(ref mut kind) = self.association {
            kind.shutdown().await?;
        }

        self.association = None;
        self.state = AmfConnectionState::Closed;

        Ok(())
    }

    /// Returns the client ID
    pub fn client_id(&self) -> i32 {
        self.client_id
    }

    /// Returns the current state
    pub fn state(&self) -> AmfConnectionState {
        self.state
    }

    /// Returns true if connected
    pub fn is_connected(&self) -> bool {
        self.state == AmfConnectionState::Connected
    }

    /// Returns the remote address
    pub fn remote_address(&self) -> SocketAddr {
        self.remote_address
    }

    /// Returns the local address (reflects the OS-assigned port after connect).
    pub fn local_address(&self) -> SocketAddr {
        // If connected, read the actual local address from the association so
        // that the OS-assigned port is always returned correctly.
        if let Some(ref kind) = self.association {
            kind.local_addr()
        } else {
            self.local_address
        }
    }

    /// Returns the association ID if connected
    pub fn association_id(&self) -> Option<i32> {
        self.association_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amf_connection_config_default() {
        let config = AmfConnectionConfig::default();
        assert_eq!(config.remote_address.port(), 38412);
    }

    #[test]
    fn test_amf_connection_new() {
        let config = AmfConnectionConfig::default();
        let conn = AmfConnection::new(1, config);
        assert_eq!(conn.client_id(), 1);
        assert_eq!(conn.state(), AmfConnectionState::Closed);
        assert!(!conn.is_connected());
    }

    #[test]
    fn test_amf_connection_state() {
        assert_ne!(
            AmfConnectionState::Connecting,
            AmfConnectionState::Connected
        );
        assert_ne!(AmfConnectionState::Connected, AmfConnectionState::Closing);
        assert_ne!(AmfConnectionState::Closing, AmfConnectionState::Closed);
    }
}
