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
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use crate::tasks::{GnbTaskBase, NgapMessage, SctpMessage, Task, TaskMessage};
use nextgsim_common::config::SctpBackendKind;
use nextgsim_common::peer_addr::PeerAddr;
use nextgsim_common::OctetString;
use nextgsim_sctp::{SctpBackend, SctpConfig};

use super::amf_connection::{AmfConnection, AmfConnectionConfig, AmfConnectionEvent};

/// How long the gNB waits before reconnecting to an AMF whose association
/// failed or was lost (issue #286).
pub const AMF_RECONNECT_DELAY: Duration = Duration::from_secs(2);

/// How often an AMF named by hostname is re-resolved while its association is
/// up (issue #286). A different answer means the AMF moved (a restarted pod),
/// and the association to the old address is treated as lost.
pub const AMF_RERESOLVE_INTERVAL: Duration = Duration::from_secs(5);

/// What the gNB was asked to connect to, kept so it can reconnect (issue #286).
#[derive(Debug, Clone)]
struct AmfTarget {
    local_address: String,
    local_port: u16,
    remote: PeerAddr,
}

/// SCTP Task for managing AMF connections
pub struct SctpTask {
    /// Task base with handles to other tasks
    task_base: GnbTaskBase,
    /// Active AMF connections indexed by client ID
    connections: HashMap<i32, AmfConnection>,
    /// Every AMF this task was asked to connect to and not asked to close,
    /// by client ID. Survives the connection, so a lost one is reconnected.
    targets: HashMap<i32, AmfTarget>,
    /// When each disconnected target is next retried.
    reconnect_due: HashMap<i32, Instant>,
    /// When hostname targets were last re-resolved.
    last_reresolve: Instant,
    reconnect_delay: Duration,
    reresolve_interval: Duration,
    /// Test seam: answers resolution instead of DNS when set.
    #[cfg(test)]
    resolve_override: Option<std::sync::Arc<std::sync::Mutex<Vec<SocketAddr>>>>,
}

impl SctpTask {
    /// Creates a new SCTP task
    pub fn new(task_base: GnbTaskBase) -> Self {
        Self {
            task_base,
            connections: HashMap::new(),
            targets: HashMap::new(),
            reconnect_due: HashMap::new(),
            last_reresolve: Instant::now(),
            reconnect_delay: AMF_RECONNECT_DELAY,
            reresolve_interval: AMF_RERESOLVE_INTERVAL,
            #[cfg(test)]
            resolve_override: None,
        }
    }

    /// Resolve a target's AMF address; a literal resolves to itself.
    async fn resolve(&self, remote: &PeerAddr) -> std::io::Result<Vec<SocketAddr>> {
        #[cfg(test)]
        if let (Some(over), PeerAddr::Host(..)) = (&self.resolve_override, remote) {
            let addrs = over.lock().unwrap().clone();
            return if addrs.is_empty() {
                Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "override: none",
                ))
            } else {
                Ok(addrs)
            };
        }
        remote.resolve().await
    }

    /// A connection failed or was lost: tell NGAP (which releases the AMF's UE
    /// contexts and re-runs NG Setup once the association is back), drop the
    /// connection and schedule a reconnect if the target is still wanted.
    ///
    /// Before issue #286 there was no reconnect at all: a gNB whose AMF
    /// restarted stayed disconnected until it was itself restarted.
    async fn on_link_lost(&mut self, client_id: i32) {
        self.connections.remove(&client_id);
        self.route_association_down(client_id).await;
        if self.targets.contains_key(&client_id) {
            self.reconnect_due
                .insert(client_id, Instant::now() + self.reconnect_delay);
        }
    }

    /// Reconnect due targets, and re-resolve connected hostname targets to
    /// detect an AMF that moved (issue #286).
    async fn maintain_targets(&mut self) {
        let now = Instant::now();
        let due: Vec<i32> = self
            .reconnect_due
            .iter()
            .filter(|(_, at)| **at <= now)
            .map(|(id, _)| *id)
            .collect();
        for client_id in due {
            self.reconnect_due.remove(&client_id);
            if self.targets.contains_key(&client_id) && !self.connections.contains_key(&client_id) {
                self.connect_target(client_id).await;
            }
        }

        if now.duration_since(self.last_reresolve) < self.reresolve_interval {
            return;
        }
        self.last_reresolve = now;
        let named: Vec<(i32, PeerAddr)> = self
            .targets
            .iter()
            .filter(|(id, t)| t.remote.is_hostname() && self.connections.contains_key(id))
            .map(|(id, t)| (*id, t.remote.clone()))
            .collect();
        for (client_id, remote) in named {
            // A lookup failure is not evidence the AMF moved (the record can be
            // briefly absent while a pod turns over); only a different answer is.
            let Ok(addrs) = self.resolve(&remote).await else {
                continue;
            };
            let Some(current) = self
                .connections
                .get(&client_id)
                .map(AmfConnection::remote_address)
            else {
                continue;
            };
            if !addrs.contains(&current) {
                warn!(
                    "AMF {} ({}) now resolves to {:?}, not {}: the association to the old \
                     address is treated as lost and the gNB reconnects",
                    client_id, remote, addrs, current
                );
                // Dropped, not shut down: the old address no longer answers,
                // and a graceful SHUTDOWN would block this task until it times
                // out.
                self.on_link_lost(client_id).await;
                // Reconnect now: the new address is already known.
                self.reconnect_due.insert(client_id, Instant::now());
            }
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

        // Warn early if QUIC transport is configured — full wiring is a
        // separate work item; we fall through to SCTP for now.
        if self.task_base.config.quic_enabled {
            warn!(
                "quic_enabled=true in GnbConfig but QUIC transport selection is not yet \
                 fully wired (client_id: {}). Falling back to SCTP.",
                client_id
            );
            // TODO: instantiate QuicTransport here when the QUIC path is fully integrated.
        }

        let remote = match PeerAddr::parse(&remote_address, remote_port) {
            Ok(PeerAddr::Literal(sa)) => PeerAddr::Literal(SocketAddr::new(sa.ip(), remote_port)),
            Ok(PeerAddr::Host(h, _)) => PeerAddr::Host(h, remote_port),
            Err(e) => {
                error!("Invalid remote AMF address {remote_address}: {e}");
                return;
            }
        };
        self.targets.insert(
            client_id,
            AmfTarget {
                local_address,
                local_port,
                remote,
            },
        );
        self.reconnect_due.remove(&client_id);
        self.connect_target(client_id).await;
    }

    /// One connection attempt to a stored target, resolving its address afresh
    /// (issue #286). A failure is reported to NGAP and retried after
    /// [`AMF_RECONNECT_DELAY`].
    async fn connect_target(&mut self, client_id: i32) {
        let Some(target) = self.targets.get(&client_id).cloned() else {
            return;
        };
        let local_address = target.local_address;
        let local_port = target.local_port;
        let remote_port = target.remote.port();

        let local_addr: SocketAddr = match format!("{local_address}:{local_port}").parse() {
            Ok(addr) => addr,
            Err(e) => {
                error!(
                    "Invalid local address {}:{}: {}",
                    local_address, local_port, e
                );
                return;
            }
        };
        let remote_addr: SocketAddr = match self.resolve(&target.remote).await {
            Ok(addrs) => addrs[0],
            Err(e) => {
                warn!(
                    "AMF {} ({}) does not resolve yet: {}; retrying in {:?}",
                    client_id, target.remote, e, self.reconnect_delay
                );
                self.on_link_lost(client_id).await;
                return;
            }
        };
        if target.remote.is_hostname() {
            info!(
                "AMF {} ({}) resolved to {}",
                client_id, target.remote, remote_addr
            );
        }

        // Resolve secondary addresses from the gNB config using client_id as
        // an index into amf_configs.  An out-of-range client_id means no
        // secondary addresses are registered (safe default).
        let secondary_addresses: Vec<SocketAddr> = self
            .task_base
            .config
            .amf_configs
            .get(client_id as usize)
            .map(|amf_cfg| {
                amf_cfg
                    .secondary_addresses
                    .iter()
                    .filter_map(|s| {
                        // Secondary address strings may or may not include a
                        // port; if there is no colon we append the same port
                        // as the primary address.
                        let with_port = if s.contains(':') {
                            s.clone()
                        } else {
                            format!("{s}:{remote_port}")
                        };
                        match with_port.parse::<SocketAddr>() {
                            Ok(a) => Some(a),
                            Err(e) => {
                                warn!("Ignoring unparseable secondary AMF address {:?}: {}", s, e);
                                None
                            }
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();

        if !secondary_addresses.is_empty() {
            info!(
                "SCTP multi-homing enabled for AMF {} with {} secondary address(es)",
                client_id,
                secondary_addresses.len()
            );
        }

        // Map the gNB-level SCTP backend knob (`sctp_backend` in the YAML) onto
        // the transport-level selector. Defaults to userspace so all existing
        // deployments are unchanged.
        let backend = match self.task_base.config.sctp_backend {
            SctpBackendKind::Userspace => SctpBackend::Userspace,
            SctpBackendKind::Kernel => SctpBackend::Kernel,
        };

        // Build the full connection config.
        let conn_config = AmfConnectionConfig {
            local_address: local_addr,
            remote_address: remote_addr,
            secondary_addresses,
            sctp_config: SctpConfig {
                backend,
                ..SctpConfig::default()
            },
        };

        // Create and connect.
        let mut connection = AmfConnection::new(client_id, conn_config.clone());
        match connection.connect(&conn_config).await {
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
                error!(
                    "Failed to connect to AMF at {}: {}; retrying in {:?}",
                    remote_addr, e, self.reconnect_delay
                );
                // Notify NGAP task of connection failure, and retry.
                self.on_link_lost(client_id).await;
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
                error!(
                    "Failed to send message to AMF (client_id: {}): {}",
                    client_id, e
                );
                // Connection may be broken, notify NGAP task and reconnect
                self.on_link_lost(client_id).await;
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
                            AmfConnectionEvent::MessageReceived {
                                client_id,
                                stream,
                                data,
                            } => {
                                self.route_ngap_pdu(client_id, stream, data).await;
                            }
                            AmfConnectionEvent::AssociationDown { client_id } => {
                                self.on_link_lost(client_id).await;
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
                        self.on_link_lost(client_id).await;
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
                    self.maintain_targets().await;
                }
            }
        }

        // Cleanup: close all connections
        info!(
            "SCTP task shutting down, closing {} connections",
            self.connections.len()
        );
        for (client_id, mut connection) in self.connections.drain() {
            if let Err(e) = connection.close().await {
                warn!(
                    "Error closing connection {} during shutdown: {}",
                    client_id, e
                );
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
            ignore_stream_ids: false,
            upf_addr: None,
            upf_port: 2152,
            pqc_config: nextgsim_common::config::PqcConfig::default(),
            ntn_config: None,
            mbs_enabled: false,
            prose_enabled: false,
            lcs_enabled: false,
            snpn_config: None,
            ..Default::default()
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

    /// Wait for the next association up/down event on `ngap_rx`.
    async fn next_association_event(
        ngap_rx: &mut mpsc::Receiver<TaskMessage<NgapMessage>>,
    ) -> NgapMessage {
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                match ngap_rx.recv().await {
                    Some(TaskMessage::Message(
                        m @ (NgapMessage::SctpAssociationUp { .. }
                        | NgapMessage::SctpAssociationDown { .. }),
                    )) => break m,
                    Some(_) => continue,
                    None => panic!("NGAP channel closed"),
                }
            }
        })
        .await
        .expect("an association event")
    }

    /// Run a userspace SCTP server (the AMF stand-in) at `addr` until dropped.
    async fn amf_stand_in(addr: SocketAddr) -> (SocketAddr, tokio::task::JoinHandle<()>) {
        let mut server = nextgsim_sctp::SctpServer::bind(addr, Default::default())
            .await
            .expect("bind AMF stand-in");
        let bound = server.local_addr();
        let handle = tokio::spawn(async move {
            loop {
                let _ = server.recv(Duration::from_millis(50)).await;
            }
        });
        (bound, handle)
    }

    /// #286: an AMF named by hostname that comes back at a NEW address (a
    /// restarted pod) is found again with no gNB restart.
    ///
    /// The gNB associates with the AMF at A. The AMF moves to B, and the name
    /// now resolves to B. The SCTP task notices on its periodic re-resolution,
    /// reports the association to A as down (NGAP releases that AMF's UEs), and
    /// reconnects to B (NGAP then re-runs NG Setup on the new association).
    /// Before #286 the address was fixed at startup and nothing reconnected.
    #[tokio::test]
    async fn an_amf_named_by_hostname_is_reconnected_at_its_new_address() {
        let (a, a_task) = amf_stand_in("127.0.0.1:0".parse().unwrap()).await;
        let (task_base, _app_rx, mut ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(test_config(), DEFAULT_CHANNEL_CAPACITY);
        let mut task = SctpTask::new(task_base);
        task.reconnect_delay = Duration::from_millis(100);
        task.reresolve_interval = Duration::from_millis(200);
        let dns = std::sync::Arc::new(std::sync::Mutex::new(vec![a]));
        task.resolve_override = Some(dns.clone());

        let (tx, rx) = mpsc::channel(16);
        let runner = tokio::spawn(async move { task.run(rx).await });
        tx.send(TaskMessage::Message(SctpMessage::ConnectionRequest {
            client_id: 0,
            local_address: "127.0.0.1".into(),
            local_port: 0,
            remote_address: "amf.test".into(),
            remote_port: a.port(),
            ppid: nextgsim_sctp::NGAP_PPID,
        }))
        .await
        .unwrap();
        assert!(matches!(
            next_association_event(&mut ngap_rx).await,
            NgapMessage::SctpAssociationUp { client_id: 0, .. }
        ));

        // The AMF restarts elsewhere: A is gone, the name now points at B.
        a_task.abort();
        let (b, _b_task) = amf_stand_in("127.0.0.2:0".parse().unwrap()).await;
        *dns.lock().unwrap() = vec![b];

        assert!(matches!(
            next_association_event(&mut ngap_rx).await,
            NgapMessage::SctpAssociationDown { client_id: 0 }
        ));
        assert!(
            matches!(
                next_association_event(&mut ngap_rx).await,
                NgapMessage::SctpAssociationUp { client_id: 0, .. }
            ),
            "the gNB must reconnect to the AMF's new address"
        );
        tx.send(TaskMessage::Shutdown).await.unwrap();
        let _ = runner.await;
    }

    /// #286: a failed connect is retried, re-resolving each time, so a gNB
    /// that starts before its AMF's DNS record exists still connects.
    #[tokio::test]
    async fn a_failed_amf_connect_is_retried_and_re_resolved() {
        let (task_base, _app_rx, mut ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(test_config(), DEFAULT_CHANNEL_CAPACITY);
        let mut task = SctpTask::new(task_base);
        task.reconnect_delay = Duration::from_millis(100);
        let dns = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        task.resolve_override = Some(dns.clone());

        let (tx, rx) = mpsc::channel(16);
        let runner = tokio::spawn(async move { task.run(rx).await });
        tx.send(TaskMessage::Message(SctpMessage::ConnectionRequest {
            client_id: 3,
            local_address: "127.0.0.1".into(),
            local_port: 0,
            remote_address: "amf.test".into(),
            remote_port: 38412,
            ppid: nextgsim_sctp::NGAP_PPID,
        }))
        .await
        .unwrap();
        // No DNS record yet: reported down, not fatal.
        assert!(matches!(
            next_association_event(&mut ngap_rx).await,
            NgapMessage::SctpAssociationDown { client_id: 3 }
        ));
        // The AMF appears.
        let (amf, _amf_task) = amf_stand_in("127.0.0.1:0".parse().unwrap()).await;
        *dns.lock().unwrap() = vec![amf];
        loop {
            match next_association_event(&mut ngap_rx).await {
                NgapMessage::SctpAssociationUp { client_id: 3, .. } => break,
                NgapMessage::SctpAssociationDown { client_id: 3 } => continue,
                other => panic!("unexpected {other:?}"),
            }
        }
        tx.send(TaskMessage::Shutdown).await.unwrap();
        let _ = runner.await;
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
