//! AMF Connection Management
//!
//! This module manages SCTP connections to AMF (Access and Mobility Management Function).
//! Each AMF connection is represented by an `AmfConnection` which wraps an SCTP association.

use std::net::SocketAddr;
use tracing::{debug, info};

use nextgsim_common::OctetString;
use nextgsim_sctp::{SctpAssociation, SctpConfig, SctpError};

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
    /// Remote AMF address
    pub remote_address: SocketAddr,
    /// SCTP configuration
    pub sctp_config: SctpConfig,
}

impl Default for AmfConnectionConfig {
    fn default() -> Self {
        Self {
            local_address: "0.0.0.0:0".parse().unwrap(),
            remote_address: "127.0.0.1:38412".parse().unwrap(),
            sctp_config: SctpConfig::default(),
        }
    }
}

/// Represents a connection to an AMF
pub struct AmfConnection {
    /// Client ID for this connection
    client_id: i32,
    /// SCTP association
    association: Option<SctpAssociation>,
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

    /// Establishes the SCTP connection to the AMF
    pub async fn connect(&mut self, config: &SctpConfig) -> Result<AmfConnectionEvent, SctpError> {
        info!(
            "Connecting to AMF at {} (client_id: {})",
            self.remote_address, self.client_id
        );

        self.state = AmfConnectionState::Connecting;

        let association = SctpAssociation::connect_with_local(
            self.local_address,
            self.remote_address,
            config.clone(),
        )
        .await?;

        // Get stream counts from the association
        // For now, use default values since sctp-proto doesn't expose negotiated stream counts directly
        self.in_streams = config.max_inbound_streams;
        self.out_streams = config.max_outbound_streams;
        self.association_id = Some(self.client_id); // Use client_id as association_id for simplicity
        self.local_address = association.local_addr();

        self.association = Some(association);
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

    /// Sends data to the AMF on the specified stream
    pub async fn send(&mut self, stream: u16, data: &[u8]) -> Result<(), SctpError> {
        if self.state != AmfConnectionState::Connected {
            return Err(SctpError::InvalidState(
                "Cannot send: not connected".to_string(),
            ));
        }

        if let Some(ref mut association) = self.association {
            debug!(
                "Sending {} bytes to AMF on stream {} (client_id: {})",
                data.len(),
                stream,
                self.client_id
            );
            association.send(stream, data).await
        } else {
            Err(SctpError::AssociationClosed)
        }
    }

    /// Receives a message from the AMF (blocking)
    pub async fn recv(&mut self) -> Result<Option<AmfConnectionEvent>, SctpError> {
        if self.state == AmfConnectionState::Closed {
            return Ok(None);
        }

        if let Some(ref mut association) = self.association {
            match association.recv().await {
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
                    // Association closed
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

    /// Tries to receive a message without blocking
    /// This polls the underlying SCTP association for incoming UDP packets
    pub async fn try_recv(&mut self) -> Result<Option<AmfConnectionEvent>, SctpError> {
        if self.state == AmfConnectionState::Closed {
            return Ok(None);
        }

        if let Some(ref mut association) = self.association {
            // First poll for incoming UDP packets
            association.poll().await?;

            // Then try to receive any available messages
            match association.try_recv() {
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

    /// Gracefully closes the connection
    pub async fn close(&mut self) -> Result<(), SctpError> {
        if self.state == AmfConnectionState::Closed {
            return Ok(());
        }

        info!("Closing AMF connection (client_id: {})", self.client_id);
        self.state = AmfConnectionState::Closing;

        if let Some(ref mut association) = self.association {
            association.shutdown().await?;
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

    /// Returns the local address
    pub fn local_address(&self) -> SocketAddr {
        self.local_address
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
        assert_ne!(AmfConnectionState::Connecting, AmfConnectionState::Connected);
        assert_ne!(AmfConnectionState::Connected, AmfConnectionState::Closing);
        assert_ne!(AmfConnectionState::Closing, AmfConnectionState::Closed);
    }
}
