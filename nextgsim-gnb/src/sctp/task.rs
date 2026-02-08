//! SCTP Task Implementation
//!
//! This module implements the SCTP task for the gNB. The SCTP task is responsible for:
//! - Managing SCTP connections to AMF(s)
//! - Handling SCTP association events (up/down)
//! - Routing received NGAP messages to the NGAP task
//! - Sending NGAP messages from the NGAP task to AMF(s)
//!
//! # Message Flow
//!
//! ```text
//! AMF <--SCTP--> SCTP Task <--Channel--> NGAP Task
//! ```
//!
//! The SCTP task receives `SctpMessage` from other tasks and routes NGAP PDUs
//! to the NGAP task via `NgapMessage`.

use std::collections::HashMap;
use std::net::SocketAddr;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use crate::tasks::{
    GnbTaskBase, NgapMessage, SctpMessage, Task, TaskMessage,
};
use nextgsim_common::OctetString;
use nextgsim_sctp::SctpConfig;

use super::amf_connection::{AmfConnection, AmfConnectionConfig, AmfConnectionEvent};

/// SCTP Task for managing AMF connections
pub struct SctpTask {
    /// Task base with handles to other tasks
    task_base: GnbTaskBase,
    /// Active AMF connections indexed by client ID
    connections: HashMap<i32, AmfConnection>,
}

impl SctpTask {
    /// Creates a new SCTP task
    pub fn new(task_base: GnbTaskBase) -> Self {
        Self {
            task_base,
            connections: HashMap::new(),
        }
    }

    /// Handles a connection request
    async fn handle_connection_request(
        &mut self,
        client_id: i32,
        local_address: String,
        local_port: u16,
        remote_address: String,
        remote_port: u16,
        ppid: u32,
    ) {
        info!(
            "SCTP connection request: {}:{} -> {}:{} (client_id: {}, ppid: {})",
            local_address, local_port, remote_address, remote_port, client_id, ppid
        );

        // Parse addresses
        let local_addr: SocketAddr = match format!("{}:{}", local_address, local_port).parse() {
            Ok(addr) => addr,
            Err(e) => {
                error!("Invalid local address {}:{}: {}", local_address, local_port, e);
                return;
            }
        };

        let remote_addr: SocketAddr = match format!("{}:{}", remote_address, remote_port).parse() {
            Ok(addr) => addr,
            Err(e) => {
                error!("Invalid remote address {}:{}: {}", remote_address, remote_port, e);
                return;
            }
        };

        // Create connection config
        let config = AmfConnectionConfig {
            local_address: local_addr,
            remote_address: remote_addr,
            sctp_config: SctpConfig::default(),
        };

        // Create and connect
        let mut connection = AmfConnection::new(client_id, config.clone());
        match connection.connect(&config.sctp_config).await {
            Ok(event) => {
                // Store the connection
                self.connections.insert(client_id, connection);

                // Route association up event to NGAP task
                if let AmfConnectionEvent::AssociationUp {
                    client_id,
                    association_id,
                    in_streams,
                    out_streams,
                } = event
                {
                    self.route_association_up(client_id, association_id, in_streams, out_streams)
                        .await;
                }
            }
            Err(e) => {
                error!("Failed to connect to AMF at {}: {}", remote_addr, e);
                // Notify NGAP task of connection failure
                self.route_association_down(client_id).await;
            }
        }
    }

    /// Handles a connection close request
    async fn handle_connection_close(&mut self, client_id: i32) {
        info!("SCTP connection close request (client_id: {})", client_id);

        if let Some(mut connection) = self.connections.remove(&client_id) {
            if let Err(e) = connection.close().await {
                warn!("Error closing connection {}: {}", client_id, e);
            }
            // Notify NGAP task
            self.route_association_down(client_id).await;
        } else {
            warn!("Connection not found for client_id: {}", client_id);
        }
    }

    /// Handles an association setup event (internal)
    async fn handle_association_setup(
        &mut self,
        client_id: i32,
        association_id: i32,
        in_streams: u16,
        out_streams: u16,
    ) {
        debug!(
            "SCTP association setup (client_id: {}, association_id: {}, in: {}, out: {})",
            client_id, association_id, in_streams, out_streams
        );

        // Route to NGAP task
        self.route_association_up(client_id, association_id, in_streams, out_streams)
            .await;
    }

    /// Handles an association shutdown event (internal)
    async fn handle_association_shutdown(&mut self, client_id: i32) {
        debug!("SCTP association shutdown (client_id: {})", client_id);

        // Remove the connection
        self.connections.remove(&client_id);

        // Route to NGAP task
        self.route_association_down(client_id).await;
    }

    /// Handles a received message from AMF
    async fn handle_receive_message(&mut self, client_id: i32, stream: u16, buffer: OctetString) {
        debug!(
            "SCTP received {} bytes on stream {} (client_id: {})",
            buffer.len(),
            stream,
            client_id
        );

        // Route the NGAP PDU to the NGAP task
        self.route_ngap_pdu(client_id, stream, buffer).await;
    }

    /// Handles a send message request
    async fn handle_send_message(&mut self, client_id: i32, stream: u16, buffer: OctetString) {
        debug!(
            "SCTP sending {} bytes on stream {} (client_id: {})",
            buffer.len(),
            stream,
            client_id
        );

        if let Some(connection) = self.connections.get_mut(&client_id) {
            if let Err(e) = connection.send(stream, buffer.data()).await {
                error!("Failed to send message to AMF (client_id: {}): {}", client_id, e);
                // Connection may be broken, notify NGAP task
                self.route_association_down(client_id).await;
                self.connections.remove(&client_id);
            }
        } else {
            warn!("Connection not found for client_id: {}", client_id);
        }
    }

    /// Handles an unhandled SCTP notification
    fn handle_unhandled_notification(&self, client_id: i32) {
        warn!("Unhandled SCTP notification (client_id: {})", client_id);
    }

    /// Routes an association up event to the NGAP task
    async fn route_association_up(
        &self,
        client_id: i32,
        association_id: i32,
        in_streams: u16,
        out_streams: u16,
    ) {
        let msg = NgapMessage::SctpAssociationUp {
            client_id,
            association_id,
            in_streams,
            out_streams,
        };

        if let Err(e) = self.task_base.ngap_tx.send(msg).await {
            error!("Failed to send association up to NGAP task: {}", e);
        }
    }

    /// Routes an association down event to the NGAP task
    async fn route_association_down(&self, client_id: i32) {
        let msg = NgapMessage::SctpAssociationDown { client_id };

        if let Err(e) = self.task_base.ngap_tx.send(msg).await {
            error!("Failed to send association down to NGAP task: {}", e);
        }
    }

    /// Routes a received NGAP PDU to the NGAP task
    async fn route_ngap_pdu(&self, client_id: i32, stream: u16, pdu: OctetString) {
        let msg = NgapMessage::ReceiveNgapPdu {
            client_id,
            stream,
            pdu,
        };

        if let Err(e) = self.task_base.ngap_tx.send(msg).await {
            error!("Failed to route NGAP PDU to NGAP task: {}", e);
        }
    }

    /// Polls all connections for incoming messages
    async fn poll_connections(&mut self) {
        let client_ids: Vec<i32> = self.connections.keys().copied().collect();

        for client_id in client_ids {
            if let Some(connection) = self.connections.get_mut(&client_id) {
                match connection.try_recv().await {
                    Ok(Some(event)) => {
                        match event {
                            AmfConnectionEvent::MessageReceived { client_id, stream, data } => {
                                self.route_ngap_pdu(client_id, stream, data).await;
                            }
                            AmfConnectionEvent::AssociationDown { client_id } => {
                                self.route_association_down(client_id).await;
                                self.connections.remove(&client_id);
                            }
                            AmfConnectionEvent::AssociationUp { .. } => {
                                // Already handled during connect
                            }
                            AmfConnectionEvent::UnhandledNotification { client_id } => {
                                self.handle_unhandled_notification(client_id);
                            }
                        }
                    }
                    Ok(None) => {
                        // No message available
                    }
                    Err(e) => {
                        error!("Error polling connection {}: {}", client_id, e);
                        self.route_association_down(client_id).await;
                        self.connections.remove(&client_id);
                    }
                }
            }
        }
    }
}

#[async_trait::async_trait]
impl Task for SctpTask {
    type Message = SctpMessage;

    async fn run(&mut self, mut rx: mpsc::Receiver<TaskMessage<Self::Message>>) {
        info!("SCTP task started");

        loop {
            // Use select to handle both incoming messages and connection polling
            tokio::select! {
                // Handle incoming task messages
                msg = rx.recv() => {
                    match msg {
                        Some(TaskMessage::Message(sctp_msg)) => {
                            match sctp_msg {
                                SctpMessage::ConnectionRequest {
                                    client_id,
                                    local_address,
                                    local_port,
                                    remote_address,
                                    remote_port,
                                    ppid,
                                } => {
                                    self.handle_connection_request(
                                        client_id,
                                        local_address,
                                        local_port,
                                        remote_address,
                                        remote_port,
                                        ppid,
                                    )
                                    .await;
                                }
                                SctpMessage::ConnectionClose { client_id } => {
                                    self.handle_connection_close(client_id).await;
                                }
                                SctpMessage::AssociationSetup {
                                    client_id,
                                    association_id,
                                    in_streams,
                                    out_streams,
                                } => {
                                    self.handle_association_setup(
                                        client_id,
                                        association_id,
                                        in_streams,
                                        out_streams,
                                    )
                                    .await;
                                }
                                SctpMessage::AssociationShutdown { client_id } => {
                                    self.handle_association_shutdown(client_id).await;
                                }
                                SctpMessage::ReceiveMessage {
                                    client_id,
                                    stream,
                                    buffer,
                                } => {
                                    self.handle_receive_message(client_id, stream, buffer).await;
                                }
                                SctpMessage::SendMessage {
                                    client_id,
                                    stream,
                                    buffer,
                                } => {
                                    self.handle_send_message(client_id, stream, buffer).await;
                                }
                                SctpMessage::UnhandledNotification { client_id } => {
                                    self.handle_unhandled_notification(client_id);
                                }
                            }
                        }
                        Some(TaskMessage::Shutdown) => {
                            info!("SCTP task received shutdown signal");
                            break;
                        }
                        None => {
                            info!("SCTP task channel closed");
                            break;
                        }
                    }
                }

                // Poll connections periodically
                _ = tokio::time::sleep(tokio::time::Duration::from_millis(10)) => {
                    self.poll_connections().await;
                }
            }
        }

        // Cleanup: close all connections
        info!("SCTP task shutting down, closing {} connections", self.connections.len());
        for (client_id, mut connection) in self.connections.drain() {
            if let Err(e) = connection.close().await {
                warn!("Error closing connection {} during shutdown: {}", client_id, e);
            }
        }

        info!("SCTP task stopped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::{GnbTaskBase, DEFAULT_CHANNEL_CAPACITY};
    use nextgsim_common::config::GnbConfig;
    use nextgsim_common::Plmn;

    fn test_config() -> GnbConfig {
        GnbConfig {
            nci: 0x000000010,
            gnb_id_length: 32,
            plmn: Plmn::new(001, 01, false),
            tac: 1,
            nssai: vec![],
            amf_configs: vec![],
            link_ip: "127.0.0.1".parse().unwrap(),
            ngap_ip: "127.0.0.1".parse().unwrap(),
            gtp_ip: "127.0.0.1".parse().unwrap(),
            gtp_advertise_ip: None,
            ignore_stream_ids: false, upf_addr: None, upf_port: 2152,
            pqc_config: nextgsim_common::config::PqcConfig::default(),
        }
    }

    #[test]
    fn test_sctp_task_creation() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);

        let task = SctpTask::new(task_base);
        assert!(task.connections.is_empty());
    }

    #[tokio::test]
    async fn test_sctp_task_routes_to_ngap() {
        let config = test_config();
        let (task_base, _app_rx, mut ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);

        let task = SctpTask::new(task_base.clone());

        // Simulate routing an NGAP PDU
        let pdu = OctetString::from_slice(&[0x00, 0x15, 0x00, 0x2f]);
        task.route_ngap_pdu(1, 0, pdu.clone()).await;

        // Verify NGAP task received the message
        match ngap_rx.recv().await {
            Some(TaskMessage::Message(NgapMessage::ReceiveNgapPdu {
                client_id,
                stream,
                pdu: received_pdu,
            })) => {
                assert_eq!(client_id, 1);
                assert_eq!(stream, 0);
                assert_eq!(received_pdu.data(), pdu.data());
            }
            _ => panic!("Expected ReceiveNgapPdu message"),
        }
    }

    #[tokio::test]
    async fn test_sctp_task_routes_association_up() {
        let config = test_config();
        let (task_base, _app_rx, mut ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);

        let task = SctpTask::new(task_base.clone());

        // Simulate association up
        task.route_association_up(1, 100, 2, 2).await;

        // Verify NGAP task received the message
        match ngap_rx.recv().await {
            Some(TaskMessage::Message(NgapMessage::SctpAssociationUp {
                client_id,
                association_id,
                in_streams,
                out_streams,
            })) => {
                assert_eq!(client_id, 1);
                assert_eq!(association_id, 100);
                assert_eq!(in_streams, 2);
                assert_eq!(out_streams, 2);
            }
            _ => panic!("Expected SctpAssociationUp message"),
        }
    }

    #[tokio::test]
    async fn test_sctp_task_routes_association_down() {
        let config = test_config();
        let (task_base, _app_rx, mut ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);

        let task = SctpTask::new(task_base.clone());

        // Simulate association down
        task.route_association_down(1).await;

        // Verify NGAP task received the message
        match ngap_rx.recv().await {
            Some(TaskMessage::Message(NgapMessage::SctpAssociationDown { client_id })) => {
                assert_eq!(client_id, 1);
            }
            _ => panic!("Expected SctpAssociationDown message"),
        }
    }
}
