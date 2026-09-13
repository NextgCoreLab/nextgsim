//! GTP Task Implementation
//!
//! Implements the GTP-U task for user plane data forwarding.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use bytes::Bytes;
use tokio::net::UdpSocket;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use nextgsim_gtp::codec::{GtpHeader, GtpMessageType};
use nextgsim_gtp::path::{EchoOutcome, PathSupervisor};
use nextgsim_gtp::restart::RestartCounter;
use nextgsim_gtp::tunnel::{GtpTunnel, PduSession, TunnelError, TunnelManager, GTP_U_PORT};

use crate::tasks::{
    GnbTaskBase, GtpMessage, GtpUeContextUpdate, PduSessionResource, RlsMessage, Task, TaskMessage,
};

/// GTP-U UE context
#[derive(Debug)]
#[allow(dead_code)]
struct GtpUeContext {
    /// UE ID
    ue_id: i32,
    /// AMF UE NGAP ID (if assigned)
    amf_ue_ngap_id: Option<i64>,
}

impl GtpUeContext {
    fn new(ue_id: i32) -> Self {
        Self {
            ue_id,
            amf_ue_ngap_id: None,
        }
    }
}

/// GTP Task
///
/// Handles GTP-U tunnel management and user plane data forwarding.
pub struct GtpTask {
    /// Task base with handles to other tasks
    task_base: GnbTaskBase,
    /// UDP socket for GTP-U
    udp_socket: Option<Arc<UdpSocket>>,
    /// UE contexts indexed by UE ID
    ue_contexts: HashMap<i32, GtpUeContext>,
    /// Tunnel manager for PDU sessions
    tunnel_manager: TunnelManager,
    /// Receive buffer size
    recv_buffer_size: usize,
    /// Enable loopback mode (for testing without UPF)
    loopback_mode: bool,
    /// This node's GTP-U restart counter, advertised in every Echo Response
    /// Recovery IE (TS 29.281 §8.2, TS 23.007).
    restart_counter: RestartCounter,
    /// N3 path supervision state (TS 29.281 §7.2.1). Present regardless of whether
    /// the prober runs: it also records peer restart counters seen in Echo Responses,
    /// which arrive whether or not this node probes.
    path_supervisor: PathSupervisor,
}

impl GtpTask {
    /// Create a new GTP task
    pub fn new(task_base: GnbTaskBase) -> Self {
        // Determine loopback mode from config - if upf_addr is set, disable loopback
        let loopback_mode = task_base.config.upf_addr.is_none();
        if loopback_mode {
            info!("GTP-U loopback mode enabled (no UPF configured)");
        } else {
            info!(
                "GTP-U forwarding to UPF at {}:{}",
                task_base.config.upf_addr.expect("value expected"),
                task_base.config.upf_port
            );
        }
        Self::build(task_base, loopback_mode)
    }

    /// Create a new GTP task with loopback mode setting
    pub fn with_loopback(task_base: GnbTaskBase, loopback_mode: bool) -> Self {
        Self::build(task_base, loopback_mode)
    }

    fn build(task_base: GnbTaskBase, loopback_mode: bool) -> Self {
        // TS 23.007: the restart counter comes from non-volatile storage and advances
        // once per process start, so a peer can tell this node restarted. Loaded here,
        // at construction, so the very first Echo Response already carries it.
        let restart_counter = match task_base.config.gtpu_restart_counter_path.as_ref() {
            Some(path) => match RestartCounter::load_and_advance(path) {
                Ok(counter) => {
                    info!(
                        "GTP-U restart counter {} (from {})",
                        counter.value(),
                        path.display()
                    );
                    counter
                }
                Err(e) => {
                    // Named rather than swallowed: with no storage the counter is a
                    // fixed 0, so no peer will ever detect a restart of this node.
                    warn!(
                        "GTP-U restart counter at {} unusable ({e}); advertising a fixed 0, so \
                         peers cannot detect a restart of this gNB",
                        path.display()
                    );
                    RestartCounter::in_memory()
                }
            },
            None => RestartCounter::in_memory(),
        };
        let path_supervisor = PathSupervisor::new(task_base.config.gtpu_echo_max_misses);
        Self {
            task_base,
            udp_socket: None,
            ue_contexts: HashMap::new(),
            tunnel_manager: TunnelManager::new(),
            recv_buffer_size: 65535,
            loopback_mode,
            restart_counter,
            path_supervisor,
        }
    }

    /// Initialize the UDP socket for GTP-U
    async fn init_udp_socket(&mut self) -> Result<(), std::io::Error> {
        let gtp_ip = self.task_base.config.gtp_ip;
        let bind_addr = SocketAddr::new(gtp_ip, GTP_U_PORT);

        let socket = UdpSocket::bind(bind_addr).await?;
        info!("GTP-U socket bound to {}", bind_addr);

        self.udp_socket = Some(Arc::new(socket));
        Ok(())
    }

    /// Handle UE context update from NGAP
    fn handle_ue_context_update(&mut self, ue_id: i32, update: GtpUeContextUpdate) {
        let context = self
            .ue_contexts
            .entry(ue_id)
            .or_insert_with(|| GtpUeContext::new(ue_id));

        if let Some(amf_id) = update.amf_ue_ngap_id {
            context.amf_ue_ngap_id = Some(amf_id);
        }

        debug!("UE context updated: ue_id={}", ue_id);
    }

    /// Handle UE context release from NGAP
    fn handle_ue_context_release(&mut self, ue_id: i32) {
        // Delete all PDU sessions for this UE
        let deleted = self.tunnel_manager.delete_sessions_for_ue(ue_id as u32);
        debug!("Deleted {} PDU sessions for UE {}", deleted.len(), ue_id);

        // Remove UE context
        self.ue_contexts.remove(&ue_id);
        debug!("UE context released: ue_id={}", ue_id);
    }

    /// Handle PDU session create from NGAP
    fn handle_session_create(&mut self, ue_id: i32, resource: PduSessionResource) {
        if !self.ue_contexts.contains_key(&ue_id) {
            error!(
                "PDU session create failed: UE context not found for ue_id={}",
                ue_id
            );
            return;
        }

        let gtp_ip = self.task_base.config.gtp_ip;
        let upf_addr = SocketAddr::new(resource.upf_address, GTP_U_PORT);
        let gnb_addr = SocketAddr::new(gtp_ip, GTP_U_PORT);

        let session = PduSession::new(
            ue_id as u32,
            resource.psi as u8,
            GtpTunnel::new(resource.uplink_teid, upf_addr),
            GtpTunnel::new(resource.downlink_teid, gnb_addr),
        );

        let session = if let Some(qfi) = resource.qfi {
            session.with_qfi(qfi)
        } else {
            session
        };

        match self.tunnel_manager.create_session(session) {
            Ok(()) => {
                info!(
                    "PDU session created: ue_id={}, psi={}, ul_teid={:#x}, dl_teid={:#x}",
                    ue_id, resource.psi, resource.uplink_teid, resource.downlink_teid
                );
            }
            Err(e) => {
                error!("PDU session create failed: {}", e);
            }
        }
    }

    /// Handle PDU session modify from NGAP (updated tunnel endpoints)
    fn handle_session_modify(&mut self, ue_id: i32, resource: PduSessionResource) {
        // Delete existing session and recreate with updated tunnel info
        let _ = self
            .tunnel_manager
            .delete_session(ue_id as u32, resource.psi as u8);
        self.handle_session_create(ue_id, resource);
    }

    /// Handle PDU session release from NGAP
    fn handle_session_release(&mut self, ue_id: i32, psi: i32) {
        match self.tunnel_manager.delete_session(ue_id as u32, psi as u8) {
            Ok(_) => {
                info!("PDU session released: ue_id={}, psi={}", ue_id, psi);
            }
            Err(e) => {
                error!("PDU session release failed: {}", e);
            }
        }
    }

    /// Handle uplink data PDU from RLS (UE -> UPF)
    async fn handle_uplink_data(&mut self, ue_id: i32, psi: i32, pdu: Vec<u8>) {
        // Check if it's an IPv4 packet (version field in first nibble)
        if pdu.is_empty() || (pdu[0] >> 4) != 4 {
            debug!("Ignoring non-IPv4 packet");
            return;
        }

        // Auto-create session if needed
        if !self.tunnel_manager.has_session(ue_id as u32, psi as u8) {
            if self.loopback_mode {
                self.auto_create_loopback_session(ue_id, psi);
            } else {
                // Auto-create session to UPF
                self.auto_create_upf_session(ue_id, psi);
            }
        }

        // In loopback mode, echo the packet back to the UE
        if self.loopback_mode {
            self.handle_loopback_data(ue_id, psi, pdu).await;
            return;
        }

        // Normal mode: send to UPF
        let socket = match &self.udp_socket {
            Some(s) => s,
            None => {
                error!("Uplink data failed: UDP socket not initialized");
                return;
            }
        };

        // Encapsulate in GTP-U
        let payload = Bytes::from(pdu);
        match self
            .tunnel_manager
            .encapsulate_uplink(ue_id as u32, psi as u8, payload)
        {
            Ok((header, dest_addr)) => {
                let encoded = header.encode();
                if let Err(e) = socket.send_to(&encoded, dest_addr).await {
                    error!("Failed to send GTP-U uplink: {}", e);
                } else {
                    debug!(
                        "Sent uplink GTP-U: ue_id={}, psi={}, teid={:#x}, {} bytes",
                        ue_id,
                        psi,
                        header.teid,
                        encoded.len()
                    );
                }
            }
            Err(e) => {
                error!("Uplink encapsulation failed: {}", e);
            }
        }
    }

    /// Auto-create a loopback PDU session for testing
    fn auto_create_loopback_session(&mut self, ue_id: i32, psi: i32) {
        // Ensure UE context exists
        self.ue_contexts.entry(ue_id).or_insert_with(|| {
            info!("Auto-creating UE context for loopback: ue_id={}", ue_id);
            GtpUeContext::new(ue_id)
        });

        // Create a loopback session with dummy TEIDs
        // TEID format: 0xFFUUPP0X where FF=loopback marker, UU=ue_id (lower 8 bits), PP=psi, X=direction
        let teid_base = 0xFF000000u32;
        let uplink_teid =
            teid_base | ((ue_id as u32 & 0xFF) << 16) | ((psi as u32 & 0xFF) << 8) | 0x01;
        let downlink_teid =
            teid_base | ((ue_id as u32 & 0xFF) << 16) | ((psi as u32 & 0xFF) << 8) | 0x02;

        let gtp_ip = self.task_base.config.gtp_ip;
        let local_addr = SocketAddr::new(gtp_ip, GTP_U_PORT);

        let session = PduSession::new(
            ue_id as u32,
            psi as u8,
            GtpTunnel::new(uplink_teid, local_addr), // Loopback to self
            GtpTunnel::new(downlink_teid, local_addr),
        )
        .with_qfi(1);

        match self.tunnel_manager.create_session(session) {
            Ok(()) => {
                info!(
                    "Loopback PDU session created: ue_id={}, psi={}, ul_teid={:#x}, dl_teid={:#x}",
                    ue_id, psi, uplink_teid, downlink_teid
                );
            }
            Err(e) => {
                error!("Failed to create loopback session: {}", e);
            }
        }
    }

    /// Auto-create a PDU session to UPF for user plane forwarding
    fn auto_create_upf_session(&mut self, ue_id: i32, psi: i32) {
        // Get UPF address from config
        let upf_addr = match self.task_base.config.upf_addr {
            Some(addr) => addr,
            None => {
                error!("Cannot create UPF session: upf_addr not configured");
                return;
            }
        };
        let upf_port = self.task_base.config.upf_port;

        // Ensure UE context exists
        self.ue_contexts.entry(ue_id).or_insert_with(|| {
            info!("Auto-creating UE context for UPF session: ue_id={}", ue_id);
            GtpUeContext::new(ue_id)
        });

        // Create session with TEIDs
        // Use a simple TEID allocation: 0x0001UUPP where UU=ue_id (lower 8 bits), PP=psi
        let teid_base = 0x00010000u32;
        let teid = teid_base | ((ue_id as u32 & 0xFF) << 8) | (psi as u32 & 0xFF);
        // Use the same TEID for both uplink and downlink
        // This allows the UPF to echo back with the same TEID it receives
        let uplink_teid = teid;
        let downlink_teid = teid;

        let upf_socket_addr = SocketAddr::new(upf_addr, upf_port);
        let gtp_ip = self.task_base.config.gtp_ip;
        let local_addr = SocketAddr::new(gtp_ip, GTP_U_PORT);

        let session = PduSession::new(
            ue_id as u32,
            psi as u8,
            GtpTunnel::new(uplink_teid, upf_socket_addr), // Send uplink to UPF
            GtpTunnel::new(downlink_teid, local_addr),    // Receive downlink from UPF
        )
        .with_qfi(1);

        match self.tunnel_manager.create_session(session) {
            Ok(()) => {
                info!(
                    "UPF PDU session created: ue_id={}, psi={}, teid={:#x}, upf={}",
                    ue_id, psi, teid, upf_socket_addr
                );
            }
            Err(e) => {
                error!("Failed to create UPF session: {}", e);
            }
        }
    }

    /// Handle loopback data - echo packet back to UE
    async fn handle_loopback_data(&self, ue_id: i32, psi: i32, pdu: Vec<u8>) {
        // For ICMP echo request, swap source and destination and change type to echo reply
        let mut response = pdu.clone();

        if response.len() >= 20 {
            // Extract IP header fields
            let ip_header_len = ((response[0] & 0x0F) * 4) as usize;

            if response.len() >= ip_header_len + 8 {
                // Check if it's ICMP (protocol = 1)
                let protocol = response[9];
                if protocol == 1 {
                    // Swap source and destination IP addresses
                    let src_ip: [u8; 4] = response[12..16].try_into().expect("value expected");
                    let dst_ip: [u8; 4] = response[16..20].try_into().expect("value expected");
                    response[12..16].copy_from_slice(&dst_ip);
                    response[16..20].copy_from_slice(&src_ip);

                    // Check if it's ICMP Echo Request (type = 8)
                    let icmp_offset = ip_header_len;
                    if response[icmp_offset] == 8 {
                        // Change to Echo Reply (type = 0)
                        response[icmp_offset] = 0;

                        // Recalculate ICMP checksum
                        // First, zero out the old checksum
                        response[icmp_offset + 2] = 0;
                        response[icmp_offset + 3] = 0;

                        // Calculate new checksum over ICMP message
                        let icmp_data = &response[icmp_offset..];
                        let checksum = self.calculate_icmp_checksum(icmp_data);
                        response[icmp_offset + 2] = (checksum >> 8) as u8;
                        response[icmp_offset + 3] = (checksum & 0xFF) as u8;

                        // Recalculate IP header checksum
                        response[10] = 0;
                        response[11] = 0;
                        let ip_checksum = self.calculate_ip_checksum(&response[..ip_header_len]);
                        response[10] = (ip_checksum >> 8) as u8;
                        response[11] = (ip_checksum & 0xFF) as u8;

                        info!(
                            "Loopback ICMP: {} -> {} (echo reply), {} bytes",
                            format!("{}.{}.{}.{}", dst_ip[0], dst_ip[1], dst_ip[2], dst_ip[3]),
                            format!("{}.{}.{}.{}", src_ip[0], src_ip[1], src_ip[2], src_ip[3]),
                            response.len()
                        );
                    }
                }
            }
        }

        // Send the response back to the UE via RLS
        let msg = RlsMessage::DownlinkData {
            ue_id,
            psi,
            pdu: response.into(),
        };

        if let Err(e) = self.task_base.rls_tx.send(msg).await {
            error!("Failed to send loopback data to RLS: {}", e);
        } else {
            debug!("Sent loopback data: ue_id={}, psi={}", ue_id, psi);
        }
    }

    /// Calculate ICMP checksum
    fn calculate_icmp_checksum(&self, data: &[u8]) -> u16 {
        self.calculate_checksum(data)
    }

    /// Calculate IP header checksum
    fn calculate_ip_checksum(&self, header: &[u8]) -> u16 {
        self.calculate_checksum(header)
    }

    /// Calculate internet checksum (RFC 1071)
    fn calculate_checksum(&self, data: &[u8]) -> u16 {
        let mut sum: u32 = 0;
        let mut i = 0;

        // Sum 16-bit words
        while i + 1 < data.len() {
            sum += ((data[i] as u32) << 8) | (data[i + 1] as u32);
            i += 2;
        }

        // Add odd byte if present
        if i < data.len() {
            sum += (data[i] as u32) << 8;
        }

        // Fold 32-bit sum to 16 bits
        while (sum >> 16) != 0 {
            sum = (sum & 0xFFFF) + (sum >> 16);
        }

        // Return one's complement
        !sum as u16
    }

    /// Handle received GTP-U packet from network (UPF -> UE)
    async fn handle_udp_receive(&mut self, data: &[u8], _source: SocketAddr) {
        // Decode GTP-U header
        let header = match GtpHeader::decode(data) {
            Ok(h) => h,
            Err(e) => {
                error!("Failed to decode GTP-U: {}", e);
                return;
            }
        };

        match header.message_type {
            GtpMessageType::GPdu => {
                self.handle_downlink_gpdu(&header, _source).await;
            }
            GtpMessageType::EchoRequest => {
                self.handle_echo_request(&header, _source).await;
            }
            GtpMessageType::EchoResponse => {
                self.handle_echo_response(&header, _source);
            }
            GtpMessageType::ErrorIndication => {
                self.handle_error_indication(&header, _source);
            }
            other => {
                warn!("Unhandled GTP-U message type: {:?}", other);
            }
        }
    }

    /// Handle downlink G-PDU (user data from UPF)
    async fn handle_downlink_gpdu(&self, header: &GtpHeader, source: SocketAddr) {
        match self.tunnel_manager.decapsulate_downlink(header) {
            Ok(dl) => {
                // amfg-09: the DL QFI/RQI from the PDU Session Container drive
                // DRB/QoS-flow selection toward the UE. Until the SDAP/DRB layer
                // is wired (amfg-04/AS), the session is selected by PSI and the
                // QoS metadata is surfaced for diagnostics / reflective QoS.
                let msg = RlsMessage::DownlinkData {
                    ue_id: dl.ue_id as i32,
                    psi: dl.psi as i32,
                    pdu: dl.payload.to_vec().into(),
                };

                if let Err(e) = self.task_base.rls_tx.send(msg).await {
                    error!("Failed to send downlink data to RLS: {}", e);
                } else {
                    debug!(
                        "Forwarded downlink data: ue_id={}, psi={}, qfi={:?}, rqi={}, {} bytes",
                        dl.ue_id,
                        dl.psi,
                        dl.qfi,
                        dl.rqi,
                        dl.payload.len()
                    );
                }
            }
            Err(TunnelError::TunnelNotFound(teid)) => {
                // TS 29.281 §7.3.1: discard the G-PDU and, for a non-zero TEID, tell
                // the sender the tunnel is invalid so it can release it. An all-zeros
                // TEID names no tunnel, so there is nothing to invalidate and the
                // spec explicitly excludes it -- answering one would ask a peer to
                // release a tunnel that does not exist.
                if teid == 0 {
                    debug!(
                        "Discarded G-PDU on the all-zeros TEID from {source}: no Error \
                            Indication is owed (TS 29.281 §7.3.1)"
                    );
                    return;
                }
                self.send_error_indication(teid, source).await;
            }
            Err(e) => {
                error!("Downlink decapsulation failed: {}", e);
            }
        }
    }

    /// Send a GTP-U Error Indication naming `teid` to `dest` (TS 29.281 §7.3.1).
    async fn send_error_indication(&self, teid: u32, dest: SocketAddr) {
        let Some(socket) = &self.udp_socket else {
            return;
        };
        let indication = GtpHeader::error_indication(teid, dest.ip());
        let encoded = indication.encode();
        if let Err(e) = socket.send_to(&encoded, dest).await {
            error!("Failed to send Error Indication to {dest}: {e}");
        } else {
            debug!(
                "Sent Error Indication for unknown TEID {teid:#x} to {dest} \
                 (TS 29.281 §7.3.1)"
            );
        }
    }

    /// Handle a received GTP-U Error Indication (TS 29.281 §4.4.2.4, TS 23.007).
    ///
    /// The peer could not match a G-PDU **this node sent**, so the tunnel is gone at
    /// the far end and keeping it here only black-holes traffic. Release it.
    fn handle_error_indication(&mut self, header: &GtpHeader, source: SocketAddr) {
        let Some(teid) = header.error_indication_teid() else {
            warn!(
                "Error Indication from {source} carries no Tunnel Endpoint Identifier \
                   Data I; nothing to release"
            );
            return;
        };
        // The GTP-U Peer Address IE names the tunnel together with the TEID. Prefer it
        // over the datagram source: they agree on a direct path, and where they do not
        // (a relay), the IE is the one that identifies the tunnel.
        let peer = header
            .error_indication_peer()
            .unwrap_or_else(|| source.ip());

        let Some(session) = self.tunnel_manager.find_by_uplink_teid(teid, peer) else {
            debug!(
                "Error Indication from {source} names uplink TEID {teid:#x} at {peer}, \
                 which this gNB holds no session for; nothing to release"
            );
            return;
        };
        let (ue_id, psi) = (session.ue_id, session.psi);
        match self.tunnel_manager.delete_session(ue_id, psi) {
            Ok(_) => info!(
                "Released PDU session ue_id={ue_id} psi={psi} on an Error Indication for \
                 uplink TEID {teid:#x} from {peer} (TS 29.281 §4.4.2.4)"
            ),
            Err(e) => error!("Failed to release ue_id={ue_id} psi={psi}: {e}"),
        }
    }

    /// Handle a received GTP-U Echo Response: the path is alive.
    ///
    /// Required by path supervision, not optional bookkeeping -- without it every
    /// answered Echo Request stays outstanding and a live path is counted as missing.
    fn handle_echo_response(&mut self, response: &GtpHeader, source: SocketAddr) {
        match self
            .path_supervisor
            .note_echo_response(source, response.recovery_restart_counter())
        {
            EchoOutcome::Alive => {
                debug!("GTP-U path to {source} confirmed alive");
            }
            EchoOutcome::Recovered => {
                info!("GTP-U path to {source} recovered");
            }
            EchoOutcome::PeerRestarted { previous, current } => {
                // Detected and reported, deliberately NOT acted on here. Purging this
                // peer's tunnels is the peer-restart half of TS 23.007 §20, and it
                // belongs with the UPF-side handling tracked in nextgcore#61 rather
                // than being invented on one side. Recording the new counter means the
                // next response does not report the same restart again.
                warn!(
                    "GTP-U peer {source} restarted: Recovery counter {previous:?} -> \
                     {current}. Tunnels toward it may be stale (TS 23.007)"
                );
            }
        }
    }

    /// Handle GTP-U Echo Request
    async fn handle_echo_request(&self, request: &GtpHeader, source: SocketAddr) {
        let socket = match &self.udp_socket {
            Some(s) => s,
            None => return,
        };

        // Build Echo Response
        let mut response = GtpHeader::echo_response(request.teid);
        if let Some(seq) = request.sequence_number {
            response = response.with_sequence_number(seq);
        }
        // Recovery IE, mandatory in an Echo Response (TS 29.281 Table 7.2.2-1),
        // carrying this node's real restart counter. It used to be a hardcoded
        // `[14, 0]`, which encodes correctly and defeats the mechanism: a constant
        // makes every restart of this gNB invisible to its peers (TS 23.007).
        let response = response.with_recovery(self.restart_counter.value());

        let encoded = response.encode();
        if let Err(e) = socket.send_to(&encoded, source).await {
            error!("Failed to send Echo Response: {}", e);
        } else {
            debug!("Sent Echo Response to {}", source);
        }
    }

    /// The peers whose N3 path is worth supervising: every distinct UPF address an
    /// active uplink tunnel points at.
    ///
    /// Derived from live sessions rather than from config, so a path is probed exactly
    /// while there is traffic that depends on it.
    fn supervised_peers(&self) -> Vec<SocketAddr> {
        let mut peers: Vec<SocketAddr> = Vec::new();
        for session in self.tunnel_manager.all_sessions() {
            let peer = session.uplink_tunnel.address;
            if !peers.contains(&peer) {
                peers.push(peer);
            }
        }
        peers
    }

    /// One supervision period: close the previous one, then probe every peer.
    ///
    /// Closing first is what makes a miss a miss: anything still outstanding from the
    /// last period went unanswered, and TS 23.007 §20.3.1 counts consecutive periods,
    /// not elapsed wall-clock. Guarding on elapsed time alone would declare a path down
    /// on liveness rather than on missed responses -- the defect nextgcore#61 records
    /// on the UPF side.
    async fn run_echo_supervision_period(&mut self) {
        let peers = self.supervised_peers();
        for peer in &peers {
            if self.path_supervisor.note_period_elapsed(*peer) {
                warn!(
                    "GTP-U path to {peer} declared DOWN after {} consecutive unanswered \
                     Echo Requests (TS 23.007 §20.3.1)",
                    self.path_supervisor.consecutive_misses(peer)
                );
            }
        }

        let Some(socket) = self.udp_socket.clone() else {
            return;
        };
        for peer in peers {
            let seq = self.path_supervisor.next_sequence();
            // TS 29.281 §5.1: TEID all zeros and the S flag set for an Echo Request.
            let request = GtpHeader::echo_request(0).with_sequence_number(seq);
            match socket.send_to(&request.encode(), peer).await {
                Ok(_) => {
                    self.path_supervisor.note_echo_sent(peer, seq);
                    debug!("Sent GTP-U Echo Request seq={seq} to {peer}");
                }
                // Not recorded as outstanding: nothing was put on the wire, so counting
                // it as a missed RESPONSE would blame the peer for a local failure.
                Err(e) => error!("Failed to send Echo Request to {peer}: {e}"),
            }
        }
    }
}

#[async_trait::async_trait]
impl Task for GtpTask {
    type Message = GtpMessage;

    async fn run(&mut self, mut rx: mpsc::Receiver<TaskMessage<GtpMessage>>) {
        // Initialize UDP socket
        if let Err(e) = self.init_udp_socket().await {
            error!("Failed to initialize GTP-U socket: {}", e);
            return;
        }

        let socket = self.udp_socket.clone().expect("value expected");
        let mut recv_buf = vec![0u8; self.recv_buffer_size];

        // Log mode
        if self.loopback_mode {
            info!("GTP-U loopback mode enabled (no UPF configured)");
        } else if let Some(upf_addr) = self.task_base.config.upf_addr {
            info!(
                "GTP-U forwarding to UPF at {}:{}",
                upf_addr, self.task_base.config.upf_port
            );
        }

        // TS 29.281 §7.2.1 path supervision. A period of 0 disables it, and that is
        // the default: the prober is opt-in so the shipped datapath is unchanged.
        // `far_future` rather than an Option<Interval> because a `select!` branch must
        // still be a valid expression when supervision is off; the branch then simply
        // never fires.
        let echo_period = self.task_base.config.gtpu_echo_period_secs;
        let mut echo_interval = match echo_period {
            0 => {
                info!("GTP-U Echo path supervision disabled (gtpu_echo_period_secs = 0)");
                tokio::time::interval_at(
                    tokio::time::Instant::now() + std::time::Duration::from_secs(86_400 * 365),
                    std::time::Duration::from_secs(86_400 * 365),
                )
            }
            secs => {
                info!(
                    "GTP-U Echo path supervision every {secs}s, path down after {} misses",
                    self.task_base.config.gtpu_echo_max_misses
                );
                let mut i = tokio::time::interval(std::time::Duration::from_secs(secs));
                // The first tick fires immediately; skip probing before any session
                // exists rather than sending to nobody.
                i.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                i
            }
        };

        info!("GTP task started");

        loop {
            tokio::select! {
                // GTP-U Echo path supervision (never fires when disabled)
                _ = echo_interval.tick(), if echo_period > 0 => {
                    self.run_echo_supervision_period().await;
                }

                // Handle incoming messages from other tasks
                msg = rx.recv() => {
                    match msg {
                        Some(TaskMessage::Message(gtp_msg)) => {
                            match gtp_msg {
                                GtpMessage::UeContextUpdate { ue_id, update } => {
                                    self.handle_ue_context_update(ue_id, update);
                                }
                                GtpMessage::UeContextRelease { ue_id } => {
                                    self.handle_ue_context_release(ue_id);
                                }
                                GtpMessage::SessionCreate { ue_id, resource } => {
                                    self.handle_session_create(ue_id, resource);
                                }
                                GtpMessage::SessionModify { ue_id, resource } => {
                                    self.handle_session_modify(ue_id, resource);
                                }
                                GtpMessage::SessionRelease { ue_id, psi } => {
                                    self.handle_session_release(ue_id, psi);
                                }
                                GtpMessage::DataPduDelivery { ue_id, psi, pdu } => {
                                    self.handle_uplink_data(ue_id, psi, pdu.into_vec()).await;
                                }
                            }
                        }
                        Some(TaskMessage::Shutdown) | None => {
                            info!("GTP task shutting down");
                            break;
                        }
                    }
                }

                // Handle incoming UDP packets
                result = socket.recv_from(&mut recv_buf) => {
                    match result {
                        Ok((len, source)) => {
                            self.handle_udp_receive(&recv_buf[..len], source).await;
                        }
                        Err(e) => {
                            error!("UDP receive error: {}", e);
                        }
                    }
                }
            }
        }

        info!("GTP task stopped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nextgsim_common::config::GnbConfig;
    use nextgsim_common::Plmn;
    use std::net::IpAddr;
    use tokio::net::UdpSocket;

    fn test_config() -> GnbConfig {
        GnbConfig {
            nci: 0x000000010,
            gnb_id_length: 32,
            plmn: Plmn::new(1, 1, false),
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
    fn test_ue_context_create() {
        let context = GtpUeContext::new(1);
        assert_eq!(context.ue_id, 1);
        assert!(context.amf_ue_ngap_id.is_none());
    }

    #[test]
    fn test_gtp_task_new() {
        let (task_base, _, _, _, _, _, _) = GnbTaskBase::new(test_config(), 16);
        let task = GtpTask::new(task_base);

        assert!(task.udp_socket.is_none());
        assert!(task.ue_contexts.is_empty());
        assert_eq!(task.tunnel_manager.session_count(), 0);
    }

    #[test]
    fn test_handle_ue_context_update() {
        let (task_base, _, _, _, _, _, _) = GnbTaskBase::new(test_config(), 16);
        let mut task = GtpTask::new(task_base);

        let update = GtpUeContextUpdate {
            ue_id: 1,
            amf_ue_ngap_id: Some(12345),
        };

        task.handle_ue_context_update(1, update);

        assert!(task.ue_contexts.contains_key(&1));
        assert_eq!(task.ue_contexts[&1].amf_ue_ngap_id, Some(12345));
    }

    #[test]
    fn test_handle_session_create_without_context() {
        let (task_base, _, _, _, _, _, _) = GnbTaskBase::new(test_config(), 16);
        let mut task = GtpTask::new(task_base);

        let resource = PduSessionResource {
            psi: 1,
            qfi: Some(1),
            uplink_teid: 0x1000,
            downlink_teid: 0x2000,
            upf_address: IpAddr::from([10, 0, 0, 1]),
        };

        // Should fail because UE context doesn't exist
        task.handle_session_create(1, resource);
        assert_eq!(task.tunnel_manager.session_count(), 0);
    }

    #[test]
    fn test_handle_session_create_with_context() {
        let (task_base, _, _, _, _, _, _) = GnbTaskBase::new(test_config(), 16);
        let mut task = GtpTask::new(task_base);

        // First create UE context
        let update = GtpUeContextUpdate {
            ue_id: 1,
            amf_ue_ngap_id: None,
        };
        task.handle_ue_context_update(1, update);

        // Now create session
        let resource = PduSessionResource {
            psi: 1,
            qfi: Some(1),
            uplink_teid: 0x1000,
            downlink_teid: 0x2000,
            upf_address: IpAddr::from([10, 0, 0, 1]),
        };

        task.handle_session_create(1, resource);
        assert_eq!(task.tunnel_manager.session_count(), 1);
        assert!(task.tunnel_manager.has_session(1, 1));
    }

    #[test]
    fn test_handle_session_release() {
        let (task_base, _, _, _, _, _, _) = GnbTaskBase::new(test_config(), 16);
        let mut task = GtpTask::new(task_base);

        // Create UE context and session
        let update = GtpUeContextUpdate {
            ue_id: 1,
            amf_ue_ngap_id: None,
        };
        task.handle_ue_context_update(1, update);

        let resource = PduSessionResource {
            psi: 1,
            qfi: Some(1),
            uplink_teid: 0x1000,
            downlink_teid: 0x2000,
            upf_address: IpAddr::from([10, 0, 0, 1]),
        };
        task.handle_session_create(1, resource);

        // Release session
        task.handle_session_release(1, 1);
        assert_eq!(task.tunnel_manager.session_count(), 0);
    }

    #[test]
    fn test_handle_ue_context_release() {
        let (task_base, _, _, _, _, _, _) = GnbTaskBase::new(test_config(), 16);
        let mut task = GtpTask::new(task_base);

        // Create UE context and multiple sessions
        let update = GtpUeContextUpdate {
            ue_id: 1,
            amf_ue_ngap_id: None,
        };
        task.handle_ue_context_update(1, update);

        for psi in 1..=3 {
            let resource = PduSessionResource {
                psi,
                qfi: Some(1),
                uplink_teid: 0x1000 + psi as u32,
                downlink_teid: 0x2000 + psi as u32,
                upf_address: IpAddr::from([10, 0, 0, 1]),
            };
            task.handle_session_create(1, resource);
        }

        assert_eq!(task.tunnel_manager.session_count(), 3);

        // Release UE context (should delete all sessions)
        task.handle_ue_context_release(1);

        assert!(!task.ue_contexts.contains_key(&1));
        assert_eq!(task.tunnel_manager.session_count(), 0);
    }

    // -----------------------------------------------------------------------
    // #43: Error Indication, Recovery counter, path supervision
    // -----------------------------------------------------------------------

    /// Bind a gNB socket into the task and hand back a second socket standing in for
    /// the UPF, so the assertions are about what actually leaves the process.
    async fn task_with_sockets() -> (GtpTask, UdpSocket, SocketAddr) {
        let (task_base, _ngap_rx, _rrc_rx, _rls_rx, _gtp_rx, _sctp_rx, _app_rx) =
            GnbTaskBase::new(test_config(), 16);
        let mut task = GtpTask::new(task_base);
        let gnb = UdpSocket::bind("127.0.0.1:0").await.expect("bind gnb");
        task.udp_socket = Some(Arc::new(gnb));
        let upf = UdpSocket::bind("127.0.0.1:0").await.expect("bind upf");
        let upf_addr = upf.local_addr().expect("upf addr");
        (task, upf, upf_addr)
    }

    /// A session whose uplink tunnel points at EXACTLY `upf_addr`, port included.
    ///
    /// The production path (`handle_session_create`) pins the peer port to the
    /// well-known 2152, which no test socket can bind, so a socket-level test of the
    /// prober has to build the session directly. `find_by_uplink_teid` matches on the
    /// IP, which is why the Error Indication tests can use the production path.
    fn live_session_to(task: &mut GtpTask, upf_addr: SocketAddr) {
        let gnb_addr = SocketAddr::new(task.task_base.config.gtp_ip, GTP_U_PORT);
        task.tunnel_manager
            .create_session(PduSession::new(
                1,
                1,
                GtpTunnel::new(0x1000, upf_addr),
                GtpTunnel::new(0x2000, gnb_addr),
            ))
            .expect("create session");
        assert_eq!(task.tunnel_manager.session_count(), 1, "precondition");
    }

    fn live_session(task: &mut GtpTask, upf_addr: SocketAddr) {
        task.handle_ue_context_update(
            1,
            GtpUeContextUpdate {
                ue_id: 1,
                amf_ue_ngap_id: None,
            },
        );
        task.handle_session_create(
            1,
            PduSessionResource {
                psi: 1,
                qfi: Some(1),
                uplink_teid: 0x1000,
                downlink_teid: 0x2000,
                upf_address: upf_addr.ip(),
            },
        );
        assert_eq!(task.tunnel_manager.session_count(), 1, "precondition");
    }

    /// Criterion 1: an unknown, non-zero TEID must produce an Error Indication naming
    /// that TEID and this node's peer address, sent to the source of the G-PDU.
    #[tokio::test]
    async fn an_unknown_teid_gpdu_returns_an_error_indication() {
        let (mut task, upf, upf_addr) = task_with_sockets().await;
        let gnb_addr = task
            .udp_socket
            .as_ref()
            .expect("socket")
            .local_addr()
            .expect("addr");

        let gpdu = GtpHeader::g_pdu(0x0BAD_F00D, Bytes::from_static(b"payload"));
        task.handle_udp_receive(&gpdu.encode(), upf_addr).await;

        let mut buf = [0u8; 256];
        let (len, from) =
            tokio::time::timeout(std::time::Duration::from_secs(5), upf.recv_from(&mut buf))
                .await
                .expect("an Error Indication must be sent for an unknown non-zero TEID")
                .expect("recv");
        assert_eq!(from, gnb_addr, "it must come from the gNB GTP-U socket");

        let decoded = GtpHeader::decode(&buf[..len]).expect("decode");
        assert_eq!(decoded.message_type, GtpMessageType::ErrorIndication);
        assert_eq!(
            decoded.error_indication_teid(),
            Some(0x0BAD_F00D),
            "the Tunnel Endpoint Identifier Data I must name the TEID that was not found"
        );
        assert_eq!(
            decoded.error_indication_peer(),
            Some(upf_addr.ip()),
            "the GTP-U Peer Address must name the node that sent the G-PDU"
        );
        assert_eq!(decoded.teid, 0, "TS 29.281 §5.1: header TEID all zeros");
    }

    /// Criterion 2: an all-zeros TEID names no tunnel, so nothing is owed and nothing
    /// is sent.
    ///
    /// The non-zero case is driven FIRST in the same test, against the same socket and
    /// the same timeout, so the silence that follows is calibrated: a bug that stopped
    /// the sender working entirely would fail the first half rather than pass here by
    /// never arriving.
    #[tokio::test]
    async fn an_all_zeros_teid_gpdu_is_discarded_silently() {
        let (mut task, upf, upf_addr) = task_with_sockets().await;
        let mut buf = [0u8; 256];

        let nonzero = GtpHeader::g_pdu(0x4321, Bytes::from_static(b"x"));
        task.handle_udp_receive(&nonzero.encode(), upf_addr).await;
        tokio::time::timeout(std::time::Duration::from_secs(5), upf.recv_from(&mut buf))
            .await
            .expect("calibration: a non-zero unknown TEID DOES answer")
            .expect("recv");

        let zero = GtpHeader::g_pdu(0, Bytes::from_static(b"x"));
        task.handle_udp_receive(&zero.encode(), upf_addr).await;
        let quiet = tokio::time::timeout(
            std::time::Duration::from_millis(300),
            upf.recv_from(&mut buf),
        )
        .await;
        assert!(
            quiet.is_err(),
            "TS 29.281 §7.3.1 excludes the all-zeros TEID: it names no tunnel to invalidate"
        );
    }

    /// Criterion 3: a received Error Indication releases the referenced session.
    #[tokio::test]
    async fn a_received_error_indication_releases_the_referenced_session() {
        let (mut task, _upf, upf_addr) = task_with_sockets().await;
        live_session(&mut task, upf_addr);

        // The UPF could not match the UPLINK TEID we send to, so that is what it names.
        let indication = GtpHeader::error_indication(0x1000, upf_addr.ip());
        task.handle_udp_receive(&indication.encode(), upf_addr)
            .await;

        assert_eq!(
            task.tunnel_manager.session_count(),
            0,
            "the session must be released: keeping it only black-holes traffic \
             (TS 29.281 §4.4.2.4, TS 23.007)"
        );
    }

    /// The same message naming a TEID this gNB does not hold must not release anything.
    /// A handler that released "the first session it found" would pass the test above.
    #[tokio::test]
    async fn an_error_indication_for_an_unknown_teid_releases_nothing() {
        let (mut task, _upf, upf_addr) = task_with_sockets().await;
        live_session(&mut task, upf_addr);

        let indication = GtpHeader::error_indication(0x7777, upf_addr.ip());
        task.handle_udp_receive(&indication.encode(), upf_addr)
            .await;
        assert_eq!(task.tunnel_manager.session_count(), 1);

        // And the right TEID from the WRONG peer must not match either: TEIDs are only
        // unique per node.
        let other_peer = GtpHeader::error_indication(0x1000, IpAddr::from([203, 0, 113, 9]));
        task.handle_udp_receive(&other_peer.encode(), upf_addr)
            .await;
        assert_eq!(
            task.tunnel_manager.session_count(),
            1,
            "the (TEID, peer) pair identifies the tunnel, not the TEID alone"
        );
    }

    /// Criterion 4, the wire half: the Echo Response carries the task's restart counter.
    #[tokio::test]
    async fn the_echo_response_carries_the_restart_counter() {
        let (mut task, upf, upf_addr) = task_with_sockets().await;
        // A NON-ZERO counter, and that is the whole point: the value this replaces was a
        // hardcoded `[14, 0]`, so a counter of 0 would satisfy the assertion below
        // whether or not the fix is present. The revert-verify pass caught exactly that
        // -- the first version of this test used `in_memory()` (value 0) and passed
        // against the hardcoded bytes.
        let seed = std::env::temp_dir().join(format!(
            "nextgsim-echo-recovery-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::write(&seed, "41").expect("seed the counter storage");
        task.restart_counter = RestartCounter::load_and_advance(&seed).expect("load");
        let _ = std::fs::remove_file(&seed);
        assert_eq!(
            task.restart_counter.value(),
            42,
            "precondition: the counter must differ from the 0 the old code hardcoded"
        );

        let request = GtpHeader::echo_request(0).with_sequence_number(99);
        task.handle_udp_receive(&request.encode(), upf_addr).await;

        let mut buf = [0u8; 256];
        let (len, _) =
            tokio::time::timeout(std::time::Duration::from_secs(5), upf.recv_from(&mut buf))
                .await
                .expect("an Echo Response must be sent")
                .expect("recv");
        let decoded = GtpHeader::decode(&buf[..len]).expect("decode");
        assert_eq!(decoded.message_type, GtpMessageType::EchoResponse);
        assert_eq!(
            decoded.sequence_number,
            Some(99),
            "the request's sequence is echoed"
        );
        assert_eq!(
            decoded.recovery_restart_counter(),
            Some(task.restart_counter.value()),
            "the Recovery IE is mandatory and must carry the real counter"
        );
    }

    /// Criterion 6: with `gtpu_echo_period_secs` at its default of 0 the prober is off,
    /// so a supervision period sends nothing even when sessions exist.
    #[tokio::test]
    async fn path_supervision_is_off_by_default() {
        assert_eq!(
            test_config().gtpu_echo_period_secs,
            0,
            "the shipped default must leave the datapath unchanged"
        );

        let (mut task, upf, upf_addr) = task_with_sockets().await;
        live_session_to(&mut task, upf_addr);

        // The period function itself is what the disabled interval never calls; calling
        // it directly proves the prober WOULD probe, so the default's silence is a
        // property of the switch rather than of an unimplemented sender.
        task.run_echo_supervision_period().await;
        let mut buf = [0u8; 256];
        let (len, _) =
            tokio::time::timeout(std::time::Duration::from_secs(5), upf.recv_from(&mut buf))
                .await
                .expect("the prober must send when it runs")
                .expect("recv");
        let decoded = GtpHeader::decode(&buf[..len]).expect("decode");
        assert_eq!(decoded.message_type, GtpMessageType::EchoRequest);
        assert_eq!(
            decoded.teid, 0,
            "TS 29.281 §5.1: Echo Request TEID all zeros"
        );
        assert!(decoded.sequence_number.is_some(), "the S flag must be set");
    }

    /// Criterion 5, the end-to-end half: a non-responding peer is declared down after
    /// the configured number of periods, while a responding one stays alive.
    #[tokio::test]
    async fn a_silent_peer_is_declared_down_and_a_responding_one_is_not() {
        let (mut task, upf, upf_addr) = task_with_sockets().await;
        live_session_to(&mut task, upf_addr);
        let misses = test_config().gtpu_echo_max_misses;
        assert!(misses >= 2, "the default threshold must be worth testing");

        let mut buf = [0u8; 256];
        for _ in 0..misses {
            task.run_echo_supervision_period().await;
            // Drain the request without answering it: this is the silent peer.
            tokio::time::timeout(std::time::Duration::from_secs(5), upf.recv_from(&mut buf))
                .await
                .expect("a request per period")
                .expect("recv");
        }
        // The transition happens when the NEXT period closes on the last unanswered
        // request.
        task.run_echo_supervision_period().await;
        assert!(
            !task.path_supervisor.is_alive(&upf_addr),
            "a peer that answered none of {misses} requests must be declared down"
        );

        // Now answer, and the path recovers.
        let response = GtpHeader::echo_response(0).with_recovery(3);
        task.handle_udp_receive(&response.encode(), upf_addr).await;
        assert!(
            task.path_supervisor.is_alive(&upf_addr),
            "an answer must bring the path back"
        );
    }
}
