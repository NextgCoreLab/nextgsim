//! GTP Task Implementation
//!
//! Implements the GTP-U task for user plane data forwarding.

use std::collections::{BTreeMap, HashMap};
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;

use bytes::Bytes;
use tokio::net::UdpSocket;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use nextgsim_gtp::codec::{GtpHeader, GtpMessageType};
use nextgsim_gtp::path::{EchoOutcome, PathSupervisor};
#[cfg(feature = "sdap-dataplane")]
use nextgsim_gtp::qfi_drb::{allocate_drbs, QfiDrbMap};
#[cfg(feature = "sdap-dataplane")]
use nextgsim_gtp::qos::{Dscp, QosFlowEnforcer};
use nextgsim_gtp::restart::RestartCounter;
use nextgsim_gtp::tunnel::{GtpTunnel, PduSession, TunnelError, TunnelManager, GTP_U_PORT};

use nextgsim_ngap::procedures::pdu_session_resource_notify::NotifyCause;

use crate::tasks::{
    GnbTaskBase, GtpMessage, GtpUeContextUpdate, NgapMessage, PduSessionResource, RlsMessage, Task,
    TaskMessage,
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

/// The SDAP entity's per-PDU-session state (TS 37.324 §5.1, issue #44).
///
/// One struct rather than two maps so the mapping and the enforcement cannot fall
/// out of step: both are populated from the same `QosFlowSetupInfo` and both are
/// meaningless for a session the other does not know about. A downlink packet whose
/// QFI the mapping knew but the enforcer did not would be forwarded with no DSCP
/// resolved, which is the silent half-configured state this shape rules out.
#[cfg(feature = "sdap-dataplane")]
#[derive(Debug, Default)]
struct SdapSessionState {
    /// Which DRB each admitted QFI maps to — the SDAP entity §5.1 requires.
    qfi_drb: QfiDrbMap,
    /// Per-flow DSCP resolution and MBR policing (TS 23.501 §5.7.2.6).
    enforcer: QosFlowEnforcer,
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
    /// SDAP state keyed by `(ue_id, psi)` — one entity per PDU session, which is
    /// what TS 37.324 §5.1 asks for (issue #44).
    ///
    /// Held here and not in `TunnelManager`'s `PduSession` because it is a RAN
    /// decision about the *radio* side: the tunnel is the N3 leg and knows nothing
    /// about DRBs. Keyed on the pair and not on the session key so the reader does
    /// not have to round-trip through `make_session_key` to look a flow up on the
    /// hot downlink path.
    #[cfg(feature = "sdap-dataplane")]
    sdap_sessions: HashMap<(i32, u8), SdapSessionState>,
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
            #[cfg(feature = "sdap-dataplane")]
            sdap_sessions: HashMap::new(),
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

        // The SDAP entities go with the sessions they belong to (issue #44). Dropped
        // rather than left to be overwritten because `ue_id`s are reallocated: the
        // next UE handed this id would inherit this one's QFI→DRB mapping, and a
        // downlink packet could be steered onto a GBR bearer nobody established.
        #[cfg(feature = "sdap-dataplane")]
        self.sdap_sessions.retain(|(id, _), _| *id != ue_id);

        // Remove UE context
        self.ue_contexts.remove(&ue_id);
        debug!("UE context released: ue_id={}", ue_id);
    }

    /// Handle PDU session create from NGAP
    fn handle_session_create(&mut self, ue_id: i32, resource: PduSessionResource) {
        let created = self.create_tunnel_session(ue_id, &resource);

        // Stand up the session's SDAP entity from the flows the core admitted
        // (issue #44), but only for a session that actually exists -- an entity for a
        // session the tunnel layer refused would map flows onto bearers no downlink
        // packet can reach, and would outlive the failed setup.
        //
        // Done here and not lazily on the first downlink packet because the mapping is
        // a function of the 5QIs, and the 5QIs only ever arrive on this message -- a
        // G-PDU carries a QFI and nothing else, so an entity built from downlink
        // traffic would map every flow to the default DRB, and the GBR bearer the UE
        // was just told to build would never carry anything.
        #[cfg(feature = "sdap-dataplane")]
        if created {
            self.install_sdap_session(ue_id, resource.psi as u8, &resource.qos_flows);
        }
        #[cfg(not(feature = "sdap-dataplane"))]
        let _ = created;
    }

    /// Create the N3 tunnel half of a PDU session; `false` if it could not be made.
    ///
    /// Split out from [`Self::handle_session_create`] because create and modify share
    /// the tunnel work but must NOT share what they do to the SDAP entity: a create
    /// states the session's whole flow set, while a Modify Request's
    /// `qosFlowAddOrModifyRequestList` names only the flows it touches, so treating
    /// the two alike would have a modify silently release every flow it did not
    /// mention.
    fn create_tunnel_session(&mut self, ue_id: i32, resource: &PduSessionResource) -> bool {
        if !self.ue_contexts.contains_key(&ue_id) {
            error!(
                "PDU session create failed: UE context not found for ue_id={}",
                ue_id
            );
            return false;
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
        true
    }

    /// Build the SDAP entity for one PDU session from its admitted QoS flows
    /// (TS 37.324 §5.1, issue #44).
    ///
    /// Replaces any previous entity for the same `(ue_id, psi)` outright: this is
    /// called for a *create*, which states the session's whole flow set, and a flow
    /// left over from a previous session on the same id would otherwise keep a DRB
    /// assignment nothing established. See [`Self::amend_sdap_session`] for the
    /// modify case, which must not replace.
    #[cfg(feature = "sdap-dataplane")]
    fn install_sdap_session(&mut self, ue_id: i32, psi: u8, qos_flows: &[(u8, Option<u16>)]) {
        let mut state = SdapSessionState::default();
        Self::admit_flows(&mut state, qos_flows);
        debug!(
            "SDAP entity for ue_id={ue_id} psi={psi}: {} QoS flow(s) admitted {qos_flows:?}",
            qos_flows.len()
        );
        self.sdap_sessions.insert((ue_id, psi), state);
    }

    /// Add or update the named flows on an existing session's SDAP entity, leaving
    /// the flows it does not name alone (issue #44).
    ///
    /// What a Modify Request means: `qosFlowAddOrModifyRequestList` (TS 38.413
    /// §9.3.4.3) is a delta, not a restatement, so flows absent from it are still
    /// admitted. Rebuilding the entity from the delta would release them — and since
    /// `QfiDrbMap` answers `Default` for an unknown QFI, the release would be silent:
    /// a GBR flow would quietly start taking the default bearer instead of the GBR one
    /// the UE is still configured for.
    #[cfg(feature = "sdap-dataplane")]
    fn amend_sdap_session(&mut self, ue_id: i32, psi: u8, qos_flows: &[(u8, Option<u16>)]) {
        let state = self.sdap_sessions.entry((ue_id, psi)).or_default();
        Self::admit_flows(state, qos_flows);
        debug!(
            "SDAP entity for ue_id={ue_id} psi={psi} amended with {} QoS flow(s) {qos_flows:?}",
            qos_flows.len()
        );
    }

    /// Admit `qos_flows` into both halves of one session's SDAP state.
    ///
    /// Shared so the mapping and the enforcer can never be given different flow sets:
    /// a QFI in one and not the other is the half-configured state
    /// [`SdapSessionState`] exists to prevent.
    #[cfg(feature = "sdap-dataplane")]
    fn admit_flows(state: &mut SdapSessionState, qos_flows: &[(u8, Option<u16>)]) {
        for &(qfi, five_qi) in qos_flows {
            state.qfi_drb.admit_flow(qfi, five_qi);
            // `mbr_kbps` 0 is unlimited, which is deliberate: NGAP's
            // `QosFlowLevelQosParameters` carries the GBR/MBR only for a GBR flow
            // (TS 38.413 §9.3.1.12) and this gNB does not plumb it through yet, so a
            // made-up ceiling would police traffic the core never capped. With no
            // limiter the enforcer resolves the DSCP and admits every packet, which
            // is the honest behaviour -- and the drop path is still live for the day
            // a real MBR is configured.
            //
            // 9 stands in for a dynamic 5QI (non-GBR, default-bearer best effort,
            // TS 23.501 Table 5.7.4-1): the enforcer keys its DSCP lookup on a 5QI
            // and has no "unknown" value, and refusing to configure the flow at all
            // would make `enforce` return `None` for a flow the core DID admit.
            state
                .enforcer
                .configure_flow(qfi, five_qi.unwrap_or(9), 0, 0);
        }
    }

    /// Handle PDU session modify from NGAP (updated tunnel endpoints)
    fn handle_session_modify(&mut self, ue_id: i32, resource: PduSessionResource) {
        // Delete existing session and recreate with updated tunnel info
        let _ = self
            .tunnel_manager
            .delete_session(ue_id as u32, resource.psi as u8);
        let created = self.create_tunnel_session(ue_id, &resource);

        // The SDAP entity survives the tunnel being rebuilt, and only the named flows
        // change (issue #44). Deliberately NOT `install_sdap_session`: the modify's
        // flow list is a delta and a replace would release every flow the SMF did not
        // restate -- and because the tunnel endpoints are all that usually change, the
        // common modify names no flows at all and must leave the mapping untouched.
        #[cfg(feature = "sdap-dataplane")]
        if created {
            self.amend_sdap_session(ue_id, resource.psi as u8, &resource.qos_flows);
        }
        #[cfg(not(feature = "sdap-dataplane"))]
        let _ = created;
    }

    /// Handle PDU session release from NGAP
    fn handle_session_release(&mut self, ue_id: i32, psi: i32) {
        // See `handle_ue_context_release` for why this is dropped and not left.
        #[cfg(feature = "sdap-dataplane")]
        self.sdap_sessions.remove(&(ue_id, psi as u8));
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

        // See `auto_create_upf_session`: one invented flow matching the `with_qfi(1)`
        // this session stamps on its uplink container.
        #[cfg(feature = "sdap-dataplane")]
        self.install_sdap_session(ue_id, psi as u8, &[(1, None)]);
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

        // Matches the `with_qfi(1)` above: this path invents the session because
        // traffic arrived without one, so the only flow it can claim is the one it
        // stamps on the uplink container. A dynamic 5QI, because nothing told it one.
        #[cfg(feature = "sdap-dataplane")]
        self.install_sdap_session(ue_id, psi as u8, &[(1, None)]);
    }

    /// Handle loopback data - echo packet back to UE
    ///
    /// `&mut self` since issue #44: with `sdap-dataplane` the echo is a downlink SDU
    /// like any other and goes through the SDAP transmit operation, which advances the
    /// flow's token bucket.
    async fn handle_loopback_data(&mut self, ue_id: i32, psi: i32, pdu: Vec<u8>) {
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

        // The loopback echo is a downlink SDU, so it takes the same SDAP path a real
        // one does -- otherwise the loopback mode would be the one configuration in
        // which the UE receives a bearer's PDUs without the header it was told to
        // strip, and every loopback ping would lose its first payload octet.
        //
        // The session's own default QFI, because a locally generated echo has no PDU
        // Session Container to read one from, and no RQI: reflective QoS is the core's
        // instruction to the UE (TS 23.501 §5.7.5) and this packet never saw the core.
        #[cfg(feature = "sdap-dataplane")]
        let (drb_id, response) = {
            let qfi = self
                .tunnel_manager
                .get_session(ue_id as u32, psi as u8)
                .and_then(|s| s.qfi)
                .unwrap_or(0);
            let Some(resolved) = self.sdap_downlink(ue_id, psi as u8, qfi, false, &response) else {
                return;
            };
            resolved
        };
        // Without the feature the bearer is the session, so the DRB id is the PSI and
        // the payload is the bare IP packet -- unchanged from before #44.
        #[cfg(not(feature = "sdap-dataplane"))]
        let drb_id = psi;

        // Send the response back to the UE via RLS
        let msg = RlsMessage::DownlinkData {
            ue_id,
            psi,
            drb_id,
            pdu: response.into(),
        };

        if let Err(e) = self.task_base.rls_tx.send(msg).await {
            error!("Failed to send loopback data to RLS: {}", e);
        } else {
            debug!(
                "Sent loopback data: ue_id={}, psi={}, drb_id={}",
                ue_id, psi, drb_id
            );
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
                self.handle_echo_response(&header, _source).await;
            }
            GtpMessageType::ErrorIndication => {
                self.handle_error_indication(&header, _source).await;
            }
            other => {
                warn!("Unhandled GTP-U message type: {:?}", other);
            }
        }
    }

    /// Handle downlink G-PDU (user data from UPF)
    async fn handle_downlink_gpdu(&mut self, header: &GtpHeader, source: SocketAddr) {
        match self.tunnel_manager.decapsulate_downlink(header) {
            Ok(dl) => {
                let ue_id = dl.ue_id as i32;
                let psi = dl.psi as i32;

                // amfg-09: the DL QFI/RQI arrive in the PDU Session Container
                // (TS 38.415 §5.5.2.1) and are what selects the QoS flow toward the
                // UE. Without `sdap-dataplane` there is no SDAP sublayer to act on
                // them: the radio bearer is the one-per-session bearer keyed by PSI,
                // so `drb_id` IS the PSI and the metadata only reaches the log. This
                // is the pre-#44 path, byte for byte.
                #[cfg(not(feature = "sdap-dataplane"))]
                let (drb_id, pdu) = (psi, dl.payload.to_vec());

                // With the feature the QFI does the two jobs TS 37.324 gives it: §5.1
                // picks the DRB, and §6.2.2.2 puts the QFI and RQI on the wire so the
                // UE can attribute the SDU to a flow (and act on reflective QoS)
                // rather than inferring it from the bearer. Returns `None` for a
                // packet the enforcer policed away, which must not be forwarded.
                #[cfg(feature = "sdap-dataplane")]
                let (drb_id, pdu) = {
                    // A G-PDU may arrive with no PDU Session Container at all, and a
                    // header still has to carry SOME QFI. The session's default flow
                    // is the honest answer -- it is the flow the core set the session
                    // up with -- and 0 only if even that is unknown, which keeps the
                    // SDU on the default DRB per §5.3.1 rather than dropping it.
                    let qfi = dl.qfi.or_else(|| {
                        self.tunnel_manager
                            .get_session(dl.ue_id, dl.psi)
                            .and_then(|s| s.qfi)
                    });
                    let Some(resolved) =
                        self.sdap_downlink(ue_id, dl.psi, qfi.unwrap_or(0), dl.rqi, dl.payload)
                    else {
                        return;
                    };
                    resolved
                };

                let msg = RlsMessage::DownlinkData {
                    ue_id,
                    psi,
                    drb_id,
                    pdu: pdu.into(),
                };

                if let Err(e) = self.task_base.rls_tx.send(msg).await {
                    error!("Failed to send downlink data to RLS: {}", e);
                } else {
                    debug!(
                        "Forwarded downlink data: ue_id={}, psi={}, drb_id={}, qfi={:?}, rqi={}, \
                         {} bytes",
                        dl.ue_id,
                        dl.psi,
                        drb_id,
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

    /// The SDAP downlink transmit operation: police the flow, pick its DRB, and
    /// prepend the header (TS 37.324 §5.2.1/§5.1/§6.2.2.2, issue #44).
    ///
    /// Returns the `(drb_id, sdap_pdu)` to forward, or `None` when the packet must be
    /// discarded. The two decisions are made together because they read the same
    /// per-session state, and splitting them would let a caller forward a packet the
    /// enforcer refused.
    #[cfg(feature = "sdap-dataplane")]
    fn sdap_downlink(
        &mut self,
        ue_id: i32,
        psi: u8,
        qfi: u8,
        rqi: bool,
        payload: &[u8],
    ) -> Option<(i32, Vec<u8>)> {
        // The DRB identities are a pure function of the PSI (see
        // `nextgsim_gtp::qfi_drb::allocate_drbs`), so both ends compute the same
        // numbers from the same input and nothing has to be signalled to keep the
        // gNB's and the UE's entity maps in agreement.
        let alloc = allocate_drbs(psi);

        // Enforcement before the header, so a policed packet costs no allocation.
        //
        // `None` from `enforce` means this QFI was never configured -- an unadmitted
        // flow, or a session created on a path that carried no `QosFlowSetupInfo`.
        // Forwarded, deliberately: the core is the authority on what it admitted, and
        // dropping here would lose traffic the UPF accepted and already billed. The
        // same reasoning is why `QfiDrbMap::drb_for` answers `Default` rather than
        // erroring for an unknown QFI.
        let (drb_choice, dscp) = match self.sdap_sessions.get_mut(&(ue_id, psi)) {
            Some(state) => {
                let choice = state.qfi_drb.drb_for(qfi);
                match state.enforcer.enforce(qfi, payload.len()) {
                    Some((false, _)) => {
                        // Over the flow's MBR (TS 23.501 §5.7.2.6). Dropped at the
                        // gNB rather than passed to RLC, because the point of a
                        // per-flow ceiling is that the excess never reaches the air
                        // interface -- forwarding it and letting the radio shed it
                        // would police nothing.
                        warn!(
                            "Dropped a downlink SDU over its MBR: ue_id={ue_id}, psi={psi}, \
                             qfi={qfi}, {} bytes",
                            payload.len()
                        );
                        return None;
                    }
                    Some((true, dscp)) => (choice, Some(dscp)),
                    None => (choice, None),
                }
            }
            // No SDAP entity for this session at all: a session the auto-create
            // paths stood up, or a downlink packet that beat its SessionCreate.
            // Everything goes to the default DRB, which is §5.3.1's rule and keeps
            // the SDU moving.
            None => {
                debug!(
                    "No SDAP entity for ue_id={ue_id} psi={psi}; QFI {qfi} takes the default \
                     DRB (TS 37.324 §5.3.1)"
                );
                (nextgsim_gtp::qfi_drb::DrbChoice::Default, None)
            }
        };

        let drb_id = alloc.id_of(drb_choice);
        // A QFI wider than the six-bit field cannot be signalled at all
        // (TS 38.413 caps `QosFlowIdentifier` at 63, so this needs a non-conformant
        // peer). Dropped rather than truncated: a truncated QFI would have the UE
        // attribute the SDU to a DIFFERENT flow, which is worse than losing it.
        let pdu = match nextgsim_pdcp::sdap::build_dl_pdu(
            nextgsim_pdcp::SdapHeader { qfi, rqi },
            payload,
        ) {
            Ok(pdu) => pdu,
            Err(e) => {
                warn!("Dropped a downlink SDU with an unencodable SDAP header: {e}");
                return None;
            }
        };

        debug!(
            "SDAP DL: ue_id={ue_id}, psi={psi}, qfi={qfi}, rqi={rqi} -> DRB {drb_id} \
             ({drb_choice:?}), dscp={}",
            dscp.map_or_else(|| "unconfigured".to_string(), |d: Dscp| d.to_string())
        );
        Some((drb_id as i32, pdu))
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
    async fn handle_error_indication(&mut self, header: &GtpHeader, source: SocketAddr) {
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
            Ok(_) => {
                info!(
                    "Released PDU session ue_id={ue_id} psi={psi} on an Error Indication for \
                     uplink TEID {teid:#x} from {peer} (TS 29.281 §4.4.2.4)"
                );
                // Report it northbound as well (#91). Releasing locally and saying
                // nothing leaves the AMF and SMF holding a session the RAN has dropped,
                // which is the same one-sided state this release exists to end -- just in
                // the other direction. Added here rather than left for the restart path
                // alone, because a module where one purge reports and its neighbour does
                // not is worse than either behaviour on its own.
                self.report_released_sessions(vec![(ue_id, psi)], "an Error Indication")
                    .await;
            }
            Err(e) => error!("Failed to release ue_id={ue_id} psi={psi}: {e}"),
        }
    }

    /// Handle a received GTP-U Echo Response: the path is alive.
    ///
    /// Required by path supervision, not optional bookkeeping -- without it every
    /// answered Echo Request stays outstanding and a live path is counted as missing.
    async fn handle_echo_response(&mut self, response: &GtpHeader, source: SocketAddr) {
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
                warn!(
                    "GTP-U peer {source} restarted: Recovery counter {previous:?} -> \
                     {current}; purging this gNB's tunnels toward it (TS 23.007 §20)"
                );
                // TS 23.007 §20: the peer of a restarted node purges the contexts it held
                // for it. A restarted UPF has lost its side of every tunnel, so uplink
                // G-PDUs into them are answered with an Error Indication at best and
                // dropped silently at worst -- #43 made the inbound Error Indication
                // converge the state one tunnel at a time, and only if the UPF bothers to
                // answer. This is the same convergence in one step.
                //
                // Scoped to the restarted peer by uplink tunnel address: a gNB with
                // sessions on two UPFs must not lose the healthy one's.
                self.purge_peer_tunnels(source.ip(), "a peer restart").await;
            }
        }
    }

    /// Release every PDU session whose uplink tunnel points at `peer`, and report them
    /// (TS 23.007 §20).
    ///
    /// The session list is collected before anything is deleted: `delete_session` needs
    /// `&mut self` and the iterator from `all_sessions` borrows it immutably.
    async fn purge_peer_tunnels(&mut self, peer: IpAddr, reason: &str) {
        let doomed: Vec<(u32, u8)> = self
            .tunnel_manager
            .all_sessions()
            .filter(|session| session.uplink_tunnel.address.ip() == peer)
            .map(|session| (session.ue_id, session.psi))
            .collect();

        if doomed.is_empty() {
            info!(
                "GTP-U peer {peer}: {reason}, and this gNB holds no tunnels toward it; \
                 nothing to purge"
            );
            return;
        }

        let mut released = Vec::with_capacity(doomed.len());
        for (ue_id, psi) in doomed {
            match self.tunnel_manager.delete_session(ue_id, psi) {
                Ok(_) => released.push((ue_id, psi)),
                // Kept going rather than returned: one session that cannot be deleted
                // must not leave the rest of a restarted peer's tunnels in place.
                Err(e) => error!(
                    "Failed to purge ue_id={ue_id} psi={psi} toward {peer} after {reason}: {e}"
                ),
            }
        }
        info!(
            "Purged {} PDU session(s) toward {peer} after {reason} (TS 23.007 §20)",
            released.len()
        );
        self.report_released_sessions(released, reason).await;
    }

    /// Tell the AMF about PDU sessions this gNB has already released, so the 5GC stops
    /// believing otherwise (TS 38.413 §8.3.5, PDU SESSION RESOURCE NOTIFY).
    ///
    /// This is the answer to #91's first criterion: **notify, do not release locally and
    /// stay quiet**. The sessions exist in the 5GC too, so a purely local delete leaves
    /// the AMF and SMF holding them -- the mirror image of the stale state being cleaned
    /// up. `PDU SESSION RESOURCE NOTIFY` is the NG-RAN-initiated procedure for exactly
    /// this, and unlike `UE CONTEXT RELEASE REQUEST` it has per-session granularity, so a
    /// UE with a second session on a healthy UPF keeps it.
    ///
    /// `TransportResourceUnavailable` (TS 38.413 §9.3.1.2, `CauseTransport`) is the cause,
    /// because that is what actually happened: the GTP-U path's state is gone. Reporting a
    /// radio-network cause would tell the SMF the radio failed.
    ///
    /// Grouped per UE because the procedure is UE-associated -- one Notify per UE, however
    /// many of its sessions were on the restarted peer. `BTreeMap` rather than `HashMap`
    /// so the order is deterministic: a test reading two Notifies off a channel would
    /// otherwise be asserting on hash iteration order.
    ///
    /// Delivery is best-effort and there is no retry: the procedure has no response
    /// message, so an AMF that is down when this is sent never learns. That ceiling is
    /// `send_pdu_session_resource_notify`'s and is recorded there; it is not made worse
    /// here.
    async fn report_released_sessions(&self, released: Vec<(u32, u8)>, reason: &str) {
        let mut per_ue: BTreeMap<u32, Vec<(u8, NotifyCause)>> = BTreeMap::new();
        for (ue_id, psi) in released {
            per_ue
                .entry(ue_id)
                .or_default()
                .push((psi, NotifyCause::TransportResourceUnavailable));
        }

        for (ue_id, released_sessions) in per_ue {
            let msg = NgapMessage::PduSessionResourceNotify {
                ue_id: ue_id as i32,
                released_sessions,
                // No surviving-session QoS changes to report: every session named here is
                // gone, not degraded.
                notified_sessions: Vec::new(),
            };
            if let Err(e) = self.task_base.ngap_tx.send(msg).await {
                error!(
                    "Could not report the sessions released for ue_id={ue_id} after \
                     {reason} to the NGAP task: {e}"
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
            qos_flows: vec![(1, None)],
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
            qos_flows: vec![(1, None)],
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
            qos_flows: vec![(1, None)],
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
                qos_flows: vec![(1, None)],
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
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
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

    /// A task whose NGAP receiver is KEPT, so a test can read the PDU Session Resource
    /// Notify the purge sends (#91).
    ///
    /// `task_with_sockets` drops it, which is why the pre-#91 release paths could not be
    /// observed northbound at all: the message went into a channel whose receiver had been
    /// dropped, and `send` on a closed channel is an `Err` this code logs and moves past.
    async fn task_with_ngap_channel() -> (
        GtpTask,
        mpsc::Receiver<TaskMessage<NgapMessage>>,
        SocketAddr,
    ) {
        // `GnbTaskBase::new` returns its receivers in declaration order:
        // app, ngap, rrc, gtp, rls, sctp.
        let (task_base, _app_rx, ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(test_config(), 16);
        let mut task = GtpTask::new(task_base);
        let gnb = UdpSocket::bind("127.0.0.1:0").await.expect("bind gnb");
        task.udp_socket = Some(Arc::new(gnb));
        let upf = UdpSocket::bind("127.0.0.1:0").await.expect("bind upf");
        let upf_addr = upf.local_addr().expect("upf addr");
        // The socket itself is not needed -- these tests read the NGAP channel, not the
        // wire -- but binding it is what makes `upf_addr` a real, unique address.
        drop(upf);
        (task, ngap_rx, upf_addr)
    }

    /// A session for `ue_id`/`psi` whose uplink tunnel points at `peer`.
    ///
    /// TEIDs are derived from the ids so two sessions never collide, which matters because
    /// `create_session` keys on (ue_id, psi) but `find_by_uplink_teid` keys on (TEID, peer).
    fn session_on_peer(task: &mut GtpTask, ue_id: u32, psi: u8, peer: SocketAddr) {
        let gnb_addr = SocketAddr::new(task.task_base.config.gtp_ip, GTP_U_PORT);
        let base = (u32::from(psi) << 8) | ue_id;
        task.tunnel_manager
            .create_session(PduSession::new(
                ue_id,
                psi,
                GtpTunnel::new(0x1000 + base, peer),
                GtpTunnel::new(0x2000 + base, gnb_addr),
            ))
            .expect("create session");
    }

    /// Drives two Echo Responses from `peer`: the first records `counter`, the second
    /// reports the restart. One counter alone is `Alive`, never `PeerRestarted`.
    async fn observe_restart(task: &mut GtpTask, peer: SocketAddr, before: u8, after: u8) {
        let first = GtpHeader::echo_response(0).with_recovery(before);
        task.handle_udp_receive(&first.encode(), peer).await;
        let second = GtpHeader::echo_response(0).with_recovery(after);
        task.handle_udp_receive(&second.encode(), peer).await;
    }

    /// Collects every PDU Session Resource Notify waiting on the NGAP channel.
    fn drain_notifies(
        rx: &mut mpsc::Receiver<TaskMessage<NgapMessage>>,
    ) -> Vec<(i32, Vec<(u8, NotifyCause)>)> {
        let mut out = Vec::new();
        while let Ok(msg) = rx.try_recv() {
            if let TaskMessage::Message(NgapMessage::PduSessionResourceNotify {
                ue_id,
                released_sessions,
                ..
            }) = msg
            {
                out.push((ue_id, released_sessions));
            }
        }
        out
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
                qos_flows: vec![(1, None)],
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

    // ====================================================================
    // #91: acting on a detected peer restart (TS 23.007 §20)
    // ====================================================================

    /// **#91**, the whole point: a detected peer restart purges that peer's tunnels AND
    /// reports them to the AMF.
    ///
    /// #43 detected the restart and logged it. The reaction was missing, so the gNB kept
    /// every PDU session pointing at a UPF that had demonstrably lost its side of them --
    /// uplink G-PDUs into tunnels the UPF answers with an Error Indication at best and
    /// drops silently at worst.
    ///
    /// Both halves are asserted because either alone is a defect. Purging without reporting
    /// leaves the AMF and SMF holding sessions the RAN has dropped, which is the mirror
    /// image of the stale state being cleaned up (#91's first criterion). Reporting without
    /// purging leaves the black hole in place.
    #[tokio::test]
    async fn a_peer_restart_purges_that_peers_tunnels_and_reports_them() {
        let (mut task, mut ngap_rx, upf_addr) = task_with_ngap_channel().await;
        session_on_peer(&mut task, 1, 1, upf_addr);
        assert_eq!(task.tunnel_manager.session_count(), 1, "precondition");

        observe_restart(&mut task, upf_addr, 7, 8).await;

        assert_eq!(
            task.tunnel_manager.session_count(),
            0,
            "TS 23.007 §20: the peer of a restarted node purges the contexts it held for \
             it; before #91 the restart was logged and nothing else"
        );
        let notifies = drain_notifies(&mut ngap_rx);
        assert_eq!(
            notifies,
            vec![(1, vec![(1, NotifyCause::TransportResourceUnavailable)])],
            "and the AMF is told, per session, with the cause that actually happened: the \
             GTP-U transport resource is gone (TS 38.413 §9.3.1.2). A radio-network cause \
             would tell the SMF the radio failed"
        );
    }

    /// **#91** criterion 3: a restart of peer A leaves peer B's sessions alone.
    ///
    /// The purge is scoped by uplink tunnel address. A gNB with sessions on two UPFs that
    /// dropped both on one restart would be worse than the defect: it would turn one peer's
    /// restart into an outage for the other's traffic.
    #[tokio::test]
    async fn a_peer_restart_leaves_another_peers_sessions_alone() {
        let (mut task, mut ngap_rx, restarted) = task_with_ngap_channel().await;
        let healthy = SocketAddr::new(IpAddr::from([203, 0, 113, 9]), GTP_U_PORT);
        session_on_peer(&mut task, 1, 1, restarted);
        session_on_peer(&mut task, 2, 1, healthy);
        assert_eq!(task.tunnel_manager.session_count(), 2, "precondition");

        observe_restart(&mut task, restarted, 3, 4).await;

        assert_eq!(
            task.tunnel_manager.session_count(),
            1,
            "only the restarted peer's session may go"
        );
        assert!(
            task.tunnel_manager.has_session(2, 1),
            "and it must be the OTHER one that survives -- a purge that dropped the wrong \
             session would leave the same count"
        );
        assert_eq!(
            drain_notifies(&mut ngap_rx),
            vec![(1, vec![(1, NotifyCause::TransportResourceUnavailable)])],
            "and only the released session is reported: telling the AMF that a live session \
             is gone would make the SMF tear it down"
        );
    }

    /// **#91** criterion 4: the FIRST counter seen from a peer is not a restart.
    ///
    /// `EchoOutcome` already distinguishes this -- a first sighting is `Alive` -- and a
    /// regression here would purge every session the first time supervision runs, which is
    /// the worst possible failure mode for this feature: it would break a healthy network
    /// the moment `gtpu_echo_period_secs` was set.
    #[tokio::test]
    async fn a_first_sighting_of_a_peers_counter_purges_nothing() {
        let (mut task, mut ngap_rx, upf_addr) = task_with_ngap_channel().await;
        session_on_peer(&mut task, 1, 1, upf_addr);

        // One Echo Response, carrying a counter this gNB has never seen.
        let first = GtpHeader::echo_response(0).with_recovery(9);
        task.handle_udp_receive(&first.encode(), upf_addr).await;
        assert_eq!(
            task.tunnel_manager.session_count(),
            1,
            "a first counter is a baseline, not a restart"
        );

        // And the SAME counter again is not a restart either.
        let repeat = GtpHeader::echo_response(0).with_recovery(9);
        task.handle_udp_receive(&repeat.encode(), upf_addr).await;
        assert_eq!(
            task.tunnel_manager.session_count(),
            1,
            "an unchanged counter means the peer did not restart"
        );
        assert!(
            drain_notifies(&mut ngap_rx).is_empty(),
            "and nothing is reported to the AMF either"
        );

        // The control: a CHANGED counter does purge, so this test cannot pass by the
        // purge never working at all.
        let changed = GtpHeader::echo_response(0).with_recovery(10);
        task.handle_udp_receive(&changed.encode(), upf_addr).await;
        assert_eq!(
            task.tunnel_manager.session_count(),
            0,
            "calibration: a changed counter IS a restart"
        );
    }

    /// **#91**: every session of one UE on the restarted peer is reported in ONE Notify.
    ///
    /// The procedure is UE-associated, so one message carries the whole released list.
    /// Sending one Notify per session would be conformant but wasteful, and — more to the
    /// point — a per-session loop is where a `HashMap` iteration order becomes visible;
    /// the grouping is a `BTreeMap` so the order is deterministic.
    #[tokio::test]
    async fn every_session_of_one_ue_on_the_restarted_peer_is_reported_in_one_notify() {
        let (mut task, mut ngap_rx, upf_addr) = task_with_ngap_channel().await;
        session_on_peer(&mut task, 1, 1, upf_addr);
        session_on_peer(&mut task, 1, 5, upf_addr);
        assert_eq!(task.tunnel_manager.session_count(), 2, "precondition");

        observe_restart(&mut task, upf_addr, 1, 2).await;

        assert_eq!(task.tunnel_manager.session_count(), 0);
        let notifies = drain_notifies(&mut ngap_rx);
        assert_eq!(notifies.len(), 1, "one UE, one Notify");
        let (ue_id, mut released) = notifies.into_iter().next().expect("the Notify");
        assert_eq!(ue_id, 1);
        released.sort_by_key(|(psi, _)| *psi);
        assert_eq!(
            released,
            vec![
                (1, NotifyCause::TransportResourceUnavailable),
                (5, NotifyCause::TransportResourceUnavailable),
            ],
            "and it names BOTH sessions: a Notify listing one of the two would leave the \
             SMF holding the other"
        );
    }

    /// **#91**: the Error Indication release is reported northbound too.
    ///
    /// #43 released the session locally and said nothing, so the AMF and SMF kept holding
    /// it — the same one-sided state the release exists to end, in the other direction.
    /// Folded in here rather than filed, because a module where the restart purge reports
    /// and its neighbour does not is worse than either behaviour on its own.
    #[tokio::test]
    async fn an_error_indication_release_is_reported_northbound() {
        let (mut task, mut ngap_rx, upf_addr) = task_with_ngap_channel().await;
        session_on_peer(&mut task, 1, 1, upf_addr);
        // `session_on_peer` derives the uplink TEID from (ue_id, psi): 0x1000 + (1<<8 | 1).
        let uplink_teid = 0x1000 + ((1u32 << 8) | 1);

        let indication = GtpHeader::error_indication(uplink_teid, upf_addr.ip());
        task.handle_udp_receive(&indication.encode(), upf_addr)
            .await;

        assert_eq!(
            task.tunnel_manager.session_count(),
            0,
            "precondition: #43's local release still happens"
        );
        assert_eq!(
            drain_notifies(&mut ngap_rx),
            vec![(1, vec![(1, NotifyCause::TransportResourceUnavailable)])],
            "and it is now reported, so the SMF stops believing the session is up"
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
