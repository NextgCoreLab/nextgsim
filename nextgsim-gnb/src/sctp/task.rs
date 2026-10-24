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

use nextgsim_common::outcome_guard::OutcomeGuard;
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

/// Upper bound on one connection attempt (resolve + SCTP association setup).
///
/// The attempt runs off the task loop (#311), so this does not stall other
/// associations; it only bounds how long one black-holed AMF holds its own
/// retry slot. The userspace backend's own INIT timeout is 30 s, so this is a
/// little above it rather than cutting it short.
pub const AMF_CONNECT_TIMEOUT: Duration = Duration::from_secs(35);

/// The result of one connection attempt, sent back to the task loop (#311).
struct ConnectOutcome {
    client_id: i32,
    /// Which attempt this was: an outcome whose attempt is no longer the
    /// current one (the target was closed or replaced meanwhile) is discarded.
    attempt: u64,
    result: Result<(AmfConnection, AmfConnectionEvent), String>,
}

/// The result of one round of hostname re-resolution (#311).
type ReresolveOutcome = Vec<(i32, PeerAddr, std::io::Result<Vec<SocketAddr>>)>;

/// How a target's address is resolved: DNS, or (tests) a fixed answer.
#[derive(Clone, Default)]
struct Resolver {
    #[cfg(test)]
    override_addrs: Option<std::sync::Arc<std::sync::Mutex<Vec<SocketAddr>>>>,
    /// Test seam: a hostname lookup that never answers, as against an
    /// unreachable DNS server.
    #[cfg(test)]
    hang: bool,
    /// Test seam: the next `panics` hostname lookups panic, as a connect task
    /// that dies mid-attempt (#367 review).
    #[cfg(test)]
    panics: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl Resolver {
    async fn resolve(&self, remote: &PeerAddr) -> std::io::Result<Vec<SocketAddr>> {
        #[cfg(test)]
        if let PeerAddr::Host(..) = remote {
            if self.hang {
                std::future::pending::<()>().await;
            }
            use std::sync::atomic::Ordering;
            if self
                .panics
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
                .is_ok()
            {
                panic!("test seam: the resolver panics");
            }
            if let Some(over) = &self.override_addrs {
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
        }
        remote.resolve().await
    }
}

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
    connect_timeout: Duration,
    resolver: Resolver,
    /// Targets with a connection attempt in flight, and that attempt's number.
    connecting: HashMap<i32, u64>,
    next_attempt: u64,
    connect_tx: mpsc::UnboundedSender<ConnectOutcome>,
    connect_rx: mpsc::UnboundedReceiver<ConnectOutcome>,
    /// Whether a re-resolution round is in flight; one at a time.
    reresolving: bool,
    reresolve_tx: mpsc::UnboundedSender<ReresolveOutcome>,
    reresolve_rx: mpsc::UnboundedReceiver<ReresolveOutcome>,
}

impl SctpTask {
    /// Creates a new SCTP task
    pub fn new(task_base: GnbTaskBase) -> Self {
        let (connect_tx, connect_rx) = mpsc::unbounded_channel();
        let (reresolve_tx, reresolve_rx) = mpsc::unbounded_channel();
        Self {
            task_base,
            connections: HashMap::new(),
            targets: HashMap::new(),
            reconnect_due: HashMap::new(),
            last_reresolve: Instant::now(),
            reconnect_delay: AMF_RECONNECT_DELAY,
            reresolve_interval: AMF_RERESOLVE_INTERVAL,
            connect_timeout: AMF_CONNECT_TIMEOUT,
            resolver: Resolver::default(),
            connecting: HashMap::new(),
            next_attempt: 0,
            connect_tx,
            connect_rx,
            reresolving: false,
            reresolve_tx,
            reresolve_rx,
        }
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
    ///
    /// Neither waits on the network here (#311): a reconnect spawns its attempt
    /// ([`Self::connect_target`]) and a re-resolution round runs as its own task,
    /// whose answer comes back through [`Self::apply_reresolution`]. This loop
    /// also carries every other association's NGAP traffic, and a DNS lookup or
    /// SCTP INIT toward a black-holed AMF used to stall it for seconds.
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

        if self.reresolving || now.duration_since(self.last_reresolve) < self.reresolve_interval {
            return;
        }
        self.last_reresolve = now;
        let named: Vec<(i32, PeerAddr)> = self
            .targets
            .iter()
            .filter(|(id, t)| t.remote.is_hostname() && self.connections.contains_key(id))
            .map(|(id, t)| (*id, t.remote.clone()))
            .collect();
        if named.is_empty() {
            return;
        }
        self.reresolving = true;
        let resolver = self.resolver.clone();
        // An empty round if the task ends without reporting, so `reresolving`
        // is cleared whatever happens to it (#342 review).
        let report = OutcomeGuard::new(self.reresolve_tx.clone(), Vec::new());
        tokio::spawn(async move {
            let mut out = Vec::with_capacity(named.len());
            for (client_id, remote) in named {
                let result = resolver.resolve(&remote).await;
                out.push((client_id, remote, result));
            }
            report.send(out);
        });
    }

    /// Act on one round of re-resolution: an association whose AMF now
    /// resolves elsewhere is treated as lost and reconnected (issue #286).
    async fn apply_reresolution(&mut self, results: ReresolveOutcome) {
        self.reresolving = false;
        for (client_id, remote, result) in results {
            // A lookup failure is not evidence the AMF moved (the record can be
            // briefly absent while a pod turns over); only a different answer is.
            let Ok(addrs) = result else {
                continue;
            };
            // The answer is for the target as it was when the round started.
            if self.targets.get(&client_id).map(|t| &t.remote) != Some(&remote) {
                continue;
            }
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
                // and a graceful SHUTDOWN would wait for it to time out.
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
        // A request for a target replaces it: an attempt still in flight for the
        // previous one is now stale and its outcome is discarded.
        self.connecting.remove(&client_id);
        // So is a live association to the previous target. It was replaced by the
        // new one without a close, leaving the old association up at the AMF
        // (#342 review). Its loss is reported to NGAP, and the SHUTDOWN runs off
        // the loop.
        if let Some(mut old) = self.connections.remove(&client_id) {
            info!("AMF {client_id}: closing the association the new request replaces");
            self.route_association_down(client_id).await;
            tokio::spawn(async move {
                let _ = old.close().await;
            });
        }
        self.connect_target(client_id).await;
    }

    /// Start one connection attempt to a stored target, resolving its address
    /// afresh (issue #286). The attempt runs as its own task (#311) and reports
    /// back through [`Self::on_connect_outcome`]; a failure is reported to NGAP
    /// and retried after [`AMF_RECONNECT_DELAY`].
    ///
    /// At most one attempt per target is in flight.
    async fn connect_target(&mut self, client_id: i32) {
        if self.connecting.contains_key(&client_id) {
            return;
        }
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

        self.next_attempt += 1;
        let attempt = self.next_attempt;
        self.connecting.insert(client_id, attempt);
        let resolver = self.resolver.clone();
        // A failed attempt if the task ends without reporting, so the target
        // leaves `connecting` and is retried (#342 review).
        let report = OutcomeGuard::new(
            self.connect_tx.clone(),
            ConnectOutcome {
                client_id,
                attempt,
                result: Err("the connection attempt ended without an outcome".to_string()),
            },
        );
        let connect_timeout = self.connect_timeout;
        let remote = target.remote;
        tokio::spawn(async move {
            let work = async {
                let remote_addr = match resolver.resolve(&remote).await {
                    Ok(addrs) => addrs[0],
                    Err(e) => return Err(format!("{remote} does not resolve yet: {e}")),
                };
                if remote.is_hostname() {
                    info!("AMF {} ({}) resolved to {}", client_id, remote, remote_addr);
                }
                let conn_config = AmfConnectionConfig {
                    local_address: local_addr,
                    remote_address: remote_addr,
                    secondary_addresses,
                    sctp_config: SctpConfig {
                        backend,
                        ..SctpConfig::default()
                    },
                };
                let mut connection = AmfConnection::new(client_id, conn_config.clone());
                match connection.connect(&conn_config).await {
                    Ok(event) => Ok((connection, event)),
                    Err(e) => Err(format!("connect to {remote_addr}: {e}")),
                }
            };
            let result = match tokio::time::timeout(connect_timeout, work).await {
                Ok(r) => r,
                Err(_) => Err(format!(
                    "{remote}: no association within {connect_timeout:?}"
                )),
            };
            report.send(ConnectOutcome {
                client_id,
                attempt,
                result,
            });
        });
    }

    /// A connection attempt finished (#311).
    async fn on_connect_outcome(&mut self, outcome: ConnectOutcome) {
        let ConnectOutcome {
            client_id,
            attempt,
            result,
        } = outcome;
        let current = self.connecting.get(&client_id) == Some(&attempt)
            && self.targets.contains_key(&client_id);
        if !current {
            // The target was closed or replaced while this attempt ran.
            if let Ok((mut connection, _)) = result {
                debug!("AMF {client_id}: discarding the association of a superseded attempt");
                // Closed off the loop: the SHUTDOWN waits on the peer, and nothing
                // peer-dependent is awaited in the task loop (#311, #342 review).
                tokio::spawn(async move {
                    let _ = connection.close().await;
                });
            }
            return;
        }
        self.connecting.remove(&client_id);
        match result {
            Ok((connection, event)) => {
                self.connections.insert(client_id, connection);
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
                warn!(
                    "AMF {}: connection attempt failed ({}); retrying in {:?}",
                    client_id, e, self.reconnect_delay
                );
                // Notify NGAP task of connection failure, and retry.
                self.on_link_lost(client_id).await;
            }
        }
    }

    /// Handles a connection close request
    async fn handle_connection_close(&mut self, client_id: i32) {
        info!("SCTP connection close request (client_id: {})", client_id);
        // An attempt still in flight is now stale.
        self.connecting.remove(&client_id);

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

                // A connection attempt finished (#311).
                Some(outcome) = self.connect_rx.recv() => {
                    self.on_connect_outcome(outcome).await;
                }

                // A re-resolution round finished (#311).
                Some(results) = self.reresolve_rx.recv() => {
                    self.apply_reresolution(results).await;
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
        task.resolver.override_addrs = Some(dns.clone());

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

    /// #342 review: a connection request for a client that is already associated
    /// replaces the association. The old one is reported down to NGAP and closed,
    /// and only then does the new one come up. It used to be overwritten in the map
    /// with no close, leaving the old association up at the AMF.
    #[tokio::test]
    async fn a_new_connection_request_closes_the_association_it_replaces() {
        let (a, _a_task) = amf_stand_in("127.0.0.1:0".parse().unwrap()).await;
        let (b, _b_task) = amf_stand_in("127.0.0.2:0".parse().unwrap()).await;
        let (task_base, _app_rx, mut ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(test_config(), DEFAULT_CHANNEL_CAPACITY);
        let task = SctpTask::new(task_base);
        let (tx, rx) = mpsc::channel(16);
        let runner = tokio::spawn(async move {
            let mut task = task;
            task.run(rx).await
        });
        for amf in [a, b] {
            tx.send(TaskMessage::Message(SctpMessage::ConnectionRequest {
                client_id: 0,
                local_address: "127.0.0.1".into(),
                local_port: 0,
                remote_address: amf.ip().to_string(),
                remote_port: amf.port(),
                ppid: nextgsim_sctp::NGAP_PPID,
            }))
            .await
            .unwrap();
            if amf == b {
                assert!(
                    matches!(
                        next_association_event(&mut ngap_rx).await,
                        NgapMessage::SctpAssociationDown { client_id: 0 }
                    ),
                    "the replaced association is reported down first"
                );
            }
            assert!(matches!(
                next_association_event(&mut ngap_rx).await,
                NgapMessage::SctpAssociationUp { client_id: 0, .. }
            ));
        }
        tx.send(TaskMessage::Shutdown).await.unwrap();
        let _ = runner.await;
    }

    /// #367 review: the connect task's OutcomeGuard, end to end. A connect task
    /// that panics still reports a failed attempt, so the target leaves
    /// `connecting`, NGAP hears the association is down, and the target is retried
    /// and comes up. With a plain sender the panic left `connecting` set and the
    /// target was never retried.
    #[tokio::test]
    async fn a_connect_task_that_panics_is_still_retried() {
        let (amf, _amf_task) = amf_stand_in("127.0.0.1:0".parse().unwrap()).await;
        let (task_base, _app_rx, mut ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(test_config(), DEFAULT_CHANNEL_CAPACITY);
        let mut task = SctpTask::new(task_base);
        task.reconnect_delay = Duration::from_millis(100);
        task.resolver.override_addrs = Some(std::sync::Arc::new(std::sync::Mutex::new(vec![amf])));
        task.resolver
            .panics
            .store(1, std::sync::atomic::Ordering::SeqCst);
        let (tx, rx) = mpsc::channel(16);
        let runner = tokio::spawn(async move { task.run(rx).await });
        tx.send(TaskMessage::Message(SctpMessage::ConnectionRequest {
            client_id: 4,
            local_address: "127.0.0.1".into(),
            local_port: 0,
            remote_address: "amf.test".into(),
            remote_port: amf.port(),
            ppid: nextgsim_sctp::NGAP_PPID,
        }))
        .await
        .unwrap();
        assert!(
            matches!(
                next_association_event(&mut ngap_rx).await,
                NgapMessage::SctpAssociationDown { client_id: 4 }
            ),
            "the panicked attempt is reported as a failure"
        );
        assert!(
            matches!(
                next_association_event(&mut ngap_rx).await,
                NgapMessage::SctpAssociationUp { client_id: 4, .. }
            ),
            "and the target is retried"
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
        task.resolver.override_addrs = Some(dns.clone());

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

    /// #311: a hostname target whose DNS lookup never answers (an unreachable
    /// resolver) does not stall the task loop: an association to another AMF
    /// still comes up, and an NGAP message on it still reaches that AMF at once.
    /// Before, resolve and connect ran inline in the loop.
    #[tokio::test]
    async fn a_hanging_resolver_does_not_delay_another_association() {
        // AMF 2: a stand-in that reports what it receives.
        let mut server =
            nextgsim_sctp::SctpServer::bind("127.0.0.1:0".parse().unwrap(), Default::default())
                .await
                .expect("bind AMF stand-in");
        let amf2 = server.local_addr();
        let (ev_tx, mut ev_rx) = mpsc::unbounded_channel();
        server.set_event_sender(ev_tx);
        let _amf2_task = tokio::spawn(async move {
            loop {
                let _ = server.recv(Duration::from_millis(20)).await;
            }
        });

        let (task_base, _app_rx, mut ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(test_config(), DEFAULT_CHANNEL_CAPACITY);
        let mut task = SctpTask::new(task_base);
        task.resolver.hang = true; // every hostname lookup hangs
        let (tx, rx) = mpsc::channel(16);
        let runner = tokio::spawn(async move { task.run(rx).await });

        // AMF 1 by hostname (hangs), then AMF 2 by literal address.
        for (client_id, remote) in [(1, "amf1.test".to_string()), (2, amf2.ip().to_string())] {
            tx.send(TaskMessage::Message(SctpMessage::ConnectionRequest {
                client_id,
                local_address: "127.0.0.1".into(),
                local_port: 0,
                remote_address: remote,
                remote_port: amf2.port(),
                ppid: nextgsim_sctp::NGAP_PPID,
            }))
            .await
            .unwrap();
        }
        let started = Instant::now();
        assert!(matches!(
            next_association_event(&mut ngap_rx).await,
            NgapMessage::SctpAssociationUp { client_id: 2, .. }
        ));
        tx.send(TaskMessage::Message(SctpMessage::SendMessage {
            client_id: 2,
            stream: 0,
            buffer: OctetString::from_slice(&[0xAB, 0xCD]),
        }))
        .await
        .unwrap();
        let got = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if let Some(nextgsim_sctp::ServerEvent::DataReceived { message, .. }) =
                    ev_rx.recv().await
                {
                    break message;
                }
            }
        })
        .await
        .expect("AMF 2 receives the message while AMF 1's lookup hangs");
        assert_eq!(&got.data[..], &[0xAB, 0xCD]);
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "took {:?}",
            started.elapsed()
        );
        tx.send(TaskMessage::Shutdown).await.unwrap();
        let _ = tokio::time::timeout(Duration::from_secs(5), runner).await;
    }

    /// #311: an attempt's outcome for a target that was closed meanwhile is
    /// discarded, not installed as a live association.
    #[tokio::test]
    async fn a_superseded_connect_outcome_is_discarded() {
        let (amf, _amf_task) = amf_stand_in("127.0.0.1:0".parse().unwrap()).await;
        let (task_base, _app_rx, mut ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(test_config(), DEFAULT_CHANNEL_CAPACITY);
        let mut task = SctpTask::new(task_base);
        task.handle_connection_request(
            4,
            "127.0.0.1".into(),
            0,
            amf.ip().to_string(),
            amf.port(),
            nextgsim_sctp::NGAP_PPID,
        )
        .await;
        assert!(task.connecting.contains_key(&4), "the attempt is in flight");
        // Closed before the attempt reports.
        task.targets.remove(&4);
        task.handle_connection_close(4).await;
        let outcome = tokio::time::timeout(Duration::from_secs(10), task.connect_rx.recv())
            .await
            .expect("the attempt reports")
            .expect("channel open");
        task.on_connect_outcome(outcome).await;
        assert!(
            task.connections.is_empty(),
            "no association for a closed target"
        );
        assert!(ngap_rx.try_recv().is_err(), "nothing reported to NGAP");
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
