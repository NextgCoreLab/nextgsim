//! RLS Task Implementation
//!
//! This module implements the RLS (Radio Link Simulation) task for the gNB,
//! handling UE discovery, RRC message relay, and user plane data relay.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use tokio::net::UdpSocket;
use tokio::sync::mpsc;
use tokio::time::interval;
use tracing::{debug, error, info, warn};

use crate::tasks::{
    GnbTaskBase, GtpMessage, NwdafMessage, RlsMessage, RrcMessage, Task, TaskMessage,
};
use nextgsim_common::OctetString;
use nextgsim_rlc::{RlcEntity, RlcMode, SnSize};
use nextgsim_rls::{
    codec, GnbCellTracker, GnbTrackerEvent, PduType, RlsHeartbeatAck,
    RlsMessage as RlsProtocolMessage, RlsPduTransmission, RlsPduTransmissionAck, RrcChannel,
    SimCoord,
};

/// Default RLS port for gNB
pub const DEFAULT_RLS_PORT: u16 = 4997;

/// Heartbeat check interval in milliseconds
const HEARTBEAT_CHECK_INTERVAL_MS: u64 = 500;

/// Maximum UDP receive buffer size
const UDP_BUFFER_SIZE: usize = 65535;

/// MAC grant size handed to RLC when building PDUs.
///
/// This simulator has no MAC scheduler; a 1500-byte grant stands in for one so a
/// typical IP packet fits in a single PDU.
const MAC_GRANT_BYTES: usize = 1500;

/// RLS Task for managing radio link simulation
pub struct RlsTask {
    /// Task base for inter-task communication
    task_base: GnbTaskBase,
    /// Cell tracker for UE discovery
    cell_tracker: GnbCellTracker,
    /// Mapping from UE ID to socket address
    ue_addresses: HashMap<i32, SocketAddr>,
    /// Mapping from STI to UE ID
    sti_to_ue_id: HashMap<u64, i32>,
    /// Pending acknowledgments per UE (UE ID -> list of PDU IDs)
    pending_acks: HashMap<i32, Vec<u32>>,
    /// gNB STI (Simulated Transmission Identifier)
    sti: u64,
    /// UDP socket for RLS communication
    socket: Option<Arc<UdpSocket>>,
    /// Local bind address
    bind_address: SocketAddr,
    /// RLC entities keyed by `(UE ID, PSI)` — one per radio bearer, as
    /// TS 38.322 §4.2.1 requires, so each bearer owns its sequence-number space
    /// and reassembly buffer.
    rlc_entities: HashMap<(i32, i32), RlcEntity>,
    /// User-plane octets moved since the last cell-load report (uplink plus
    /// downlink). Counted here because the RLS task is the only place in the gNB
    /// that sees every user-plane PDU on the radio side, so it is the only
    /// producer that can report a *measured* load rather than a guessed one.
    load_window_octets: u64,
    /// When the current cell-load reporting window opened. The report is a rate,
    /// so the elapsed time has to be measured rather than assumed from the timer
    /// period -- a busy task ticks late and would otherwise overstate throughput.
    load_window_start: Instant,
}

impl RlsTask {
    /// Creates a new RLS task
    pub fn new(task_base: GnbTaskBase) -> Self {
        // Generate STI from gNB NCI
        let sti = task_base.config.nci;
        let phy_location = SimCoord::new(0, 0, 0);

        // Get bind address from config
        let bind_address = SocketAddr::new(task_base.config.link_ip, DEFAULT_RLS_PORT);

        Self {
            task_base,
            cell_tracker: GnbCellTracker::new(sti, phy_location),
            ue_addresses: HashMap::new(),
            sti_to_ue_id: HashMap::new(),
            pending_acks: HashMap::new(),
            sti,
            socket: None,
            bind_address,
            rlc_entities: HashMap::new(),
            load_window_octets: 0,
            load_window_start: Instant::now(),
        }
    }

    /// Creates a new RLS task with custom bind address
    pub fn with_bind_address(task_base: GnbTaskBase, bind_address: SocketAddr) -> Self {
        let sti = task_base.config.nci;
        let phy_location = SimCoord::new(0, 0, 0);

        Self {
            task_base,
            cell_tracker: GnbCellTracker::new(sti, phy_location),
            ue_addresses: HashMap::new(),
            sti_to_ue_id: HashMap::new(),
            pending_acks: HashMap::new(),
            sti,
            socket: None,
            bind_address,
            rlc_entities: HashMap::new(),
            load_window_octets: 0,
            load_window_start: Instant::now(),
        }
    }

    /// Returns the RLC entity for one UE's radio bearer, creating a UM entity on
    /// first use.
    ///
    /// Keyed on `(ue_id, psi)` because TS 38.322 §4.2.1 gives every radio bearer
    /// its own RLC entity, and therefore its own sequence-number space and
    /// reassembly buffer. Keying on the UE alone interleaved every PDU session
    /// into one SN counter while the UE demultiplexed them into per-PSI entities
    /// each expecting a contiguous sequence — so a second PDU session corrupted
    /// both sessions' numbering rather than failing cleanly.
    ///
    /// The PSI stands in for the DRB identity: this simulator maps one DRB per
    /// PDU session (see `nextgsim-gnb/src/gtp`), so the two are one to one.
    fn rlc_entity_for(&mut self, ue_id: i32, psi: i32) -> &mut RlcEntity {
        let mode = if self.task_base.config.rlc_am_psis.contains(&(psi as u8)) {
            RlcMode::AcknowledgedMode
        } else {
            RlcMode::UnacknowledgedMode
        };
        self.rlc_entities
            .entry((ue_id, psi))
            .or_insert_with(|| RlcEntity::new(mode, SnSize::Sn12))
    }

    /// Drives the RLC timers on every entity and transmits whatever they
    /// produce: a UM `t-Reassembly` expiry discards a stranded SDU
    /// (TS 38.322 §5.2.2.2.4), an AM `t-Reassembly` expiry triggers a STATUS
    /// report (§5.2.3.2.4), and an AM `t-PollRetransmit` expiry re-offers an
    /// unacknowledged PDU (§5.3.3.4).
    async fn poll_rlc_timers(&mut self) {
        let now = Instant::now();
        let mut outbound: Vec<(i32, i32, Vec<u8>)> = Vec::new();
        for ((ue_id, psi), rlc) in &mut self.rlc_entities {
            if rlc.poll_timers(now) {
                debug!(
                    "RLC timer expired: ue_id={}, psi={}, rx_next_reassembly={}",
                    ue_id,
                    psi,
                    rlc.rx_next_reassembly()
                );
            }
            if let Some(status) = rlc.build_status_pdu() {
                outbound.push((*ue_id, *psi, status));
            }
            while let Some(retx) = rlc.build_pdu(MAC_GRANT_BYTES) {
                outbound.push((*ue_id, *psi, retx));
            }
        }
        for (ue_id, psi, pdu) in outbound {
            self.send_rlc_pdu(ue_id, psi, pdu).await;
        }
    }

    /// Sends one RLC PDU (data or STATUS) to a UE over RLS.
    async fn send_rlc_pdu(&mut self, ue_id: i32, psi: i32, pdu: Vec<u8>) {
        let Some(&dest) = self.ue_addresses.get(&ue_id) else {
            warn!("RLC PDU for unknown UE[{}] dropped", ue_id);
            return;
        };
        // Downlink half of the cell-load measurement (issue #18). Counted here
        // rather than at the GTP boundary so a PDU the RLC layer splits is
        // counted as the two PDUs the radio actually carries.
        self.load_window_octets += pdu.len() as u64;
        let transmission = RlsPduTransmission {
            sti: self.sti,
            pdu_type: PduType::Data,
            pdu_id: 0,
            payload: psi as u32,
            pdu: Bytes::from(pdu),
        };
        self.send_rls_message(dest, &RlsProtocolMessage::PduTransmission(transmission))
            .await;
    }

    /// Sends the STATUS report an AM bearer owes its peer, if one is due
    /// (TS 38.322 §5.3.4). A no-op for a UM bearer, which has no STATUS PDU.
    async fn send_pending_status(&mut self, ue_id: i32, psi: i32) {
        let status = self
            .rlc_entities
            .get_mut(&(ue_id, psi))
            .and_then(RlcEntity::build_status_pdu);
        if let Some(status) = status {
            self.send_rlc_pdu(ue_id, psi, status).await;
        }
    }

    /// Initializes the UDP socket
    async fn init_socket(&mut self) -> Result<(), std::io::Error> {
        let socket = UdpSocket::bind(self.bind_address).await?;
        info!("RLS task bound to {}", self.bind_address);
        self.socket = Some(Arc::new(socket));
        Ok(())
    }

    /// Handles a received RLS message from the network
    async fn handle_receive_rls_message(&mut self, data: OctetString, source: SocketAddr) {
        let bytes = Bytes::copy_from_slice(data.data());

        match codec::decode(&bytes) {
            Ok(msg) => {
                self.process_rls_message(msg, source).await;
            }
            Err(e) => {
                warn!("Failed to decode RLS message from {}: {}", source, e);
            }
        }
    }

    /// Processes a decoded RLS protocol message
    async fn process_rls_message(&mut self, msg: RlsProtocolMessage, source: SocketAddr) {
        match msg {
            RlsProtocolMessage::Heartbeat(heartbeat) => {
                self.handle_heartbeat(heartbeat.sti, source, &heartbeat)
                    .await;
            }
            RlsProtocolMessage::PduTransmission(pdu) => {
                self.handle_pdu_transmission(pdu.sti, source, &pdu).await;
            }
            RlsProtocolMessage::PduTransmissionAck(ack) => {
                self.handle_pdu_ack(&ack);
            }
            RlsProtocolMessage::HeartbeatAck(_) => {
                // gNB doesn't process heartbeat acks (it sends them)
                debug!("Ignoring heartbeat ack from {}", source);
            }
        }
    }

    /// Handles a heartbeat message from a UE
    async fn handle_heartbeat(
        &mut self,
        sti: u64,
        source: SocketAddr,
        heartbeat: &nextgsim_rls::RlsHeartbeat,
    ) {
        let (ack, events) = self.cell_tracker.process_heartbeat(sti, source, heartbeat);

        // Process tracker events
        for event in events {
            match event {
                GnbTrackerEvent::UeDetected { ue_id, sti } => {
                    info!("UE detected: ue_id={}, sti={}", ue_id, sti);
                    let ue_id_i32 = ue_id as i32;
                    self.ue_addresses.insert(ue_id_i32, source);
                    self.sti_to_ue_id.insert(sti, ue_id_i32);

                    // Notify RRC task about signal detection
                    if let Err(e) = self
                        .task_base
                        .rrc_tx
                        .send(RrcMessage::SignalDetected { ue_id: ue_id_i32 })
                        .await
                    {
                        error!("Failed to send SignalDetected to RRC: {}", e);
                    }
                }
                GnbTrackerEvent::UeLost { ue_id } => {
                    info!("UE lost: ue_id={}", ue_id);
                    let ue_id_i32 = ue_id as i32;
                    self.ue_addresses.remove(&ue_id_i32);
                    // Note: sti_to_ue_id cleanup happens in check_lost_ues
                }
            }
        }

        // Send heartbeat acknowledgment
        if let Some(ack) = ack {
            self.forward_measurement_to_nwdaf(sti, &ack, heartbeat)
                .await;
            self.send_heartbeat_ack(source, &ack).await;
        }
    }

    /// Forward a UE's radio measurement to NWDAF (issue #18).
    ///
    /// This is the gNB's only source of *real* per-UE radio data. The heartbeat
    /// ack's `dbm` is what the cell tracker's channel model computed for this
    /// UE's position, and `sim_pos` is the position the UE reported -- so both
    /// values are observed rather than assumed. Before this, the only producer
    /// was the ISAC task, which has a fused position but no RSRP at all, so the
    /// analytics engine never saw a radio measurement.
    ///
    /// RSRQ stays `None`: the RLS channel model produces a single received-power
    /// figure and models no interference, so there is nothing to derive a
    /// quality ratio from. Sending a number would be inventing one.
    async fn forward_measurement_to_nwdaf(
        &self,
        sti: u64,
        ack: &RlsHeartbeatAck,
        heartbeat: &nextgsim_rls::RlsHeartbeat,
    ) {
        let Some(ref sixg) = self.task_base.sixg else {
            return;
        };
        // The UE id is assigned when the tracker first sees the STI, so a
        // heartbeat from a UE the tracker has not admitted has no id to report
        // under. Skipped rather than reported under a placeholder id, which
        // would merge two UEs' measurement histories.
        let Some(&ue_id) = self.sti_to_ue_id.get(&sti) else {
            return;
        };

        let msg = NwdafMessage::UeMeasurement {
            ue_id,
            rsrp: Some(ack.dbm as f32),
            rsrq: None,
            position: (
                heartbeat.sim_pos.x as f32,
                heartbeat.sim_pos.y as f32,
                heartbeat.sim_pos.z as f32,
            ),
        };
        if let Err(e) = sixg.nwdaf_tx.send(msg).await {
            warn!("RLS: failed to forward UE measurement to NWDAF: {}", e);
        }
    }

    /// Report the cell's measured load to NWDAF and open a new window (issue #18).
    ///
    /// `connected_ues` is the number of UEs with a live radio association, which
    /// the RLS task knows first-hand. `throughput_mbps` is derived from the
    /// octets actually moved over the elapsed window, so the series varies with
    /// real traffic -- which is the point: a constant load series is invisible to
    /// the z-score anomaly detector and is extrapolated as fact by load
    /// prediction.
    ///
    /// `prb_usage` is `None` because this simulator has no PRB scheduler; see the
    /// field's own documentation on [`NwdafMessage::CellLoad`].
    async fn report_cell_load_to_nwdaf(&mut self) {
        let elapsed = self.load_window_start.elapsed();
        let octets = self.load_window_octets;
        // Open the next window before any early return, so a window whose report
        // is dropped does not have its traffic counted twice in the next one.
        self.load_window_octets = 0;
        self.load_window_start = Instant::now();

        let Some(ref sixg) = self.task_base.sixg else {
            return;
        };

        // A zero-length window would divide by zero; a window with no traffic is
        // a real measurement of zero and is reported as such.
        let seconds = elapsed.as_secs_f32();
        let throughput_mbps = if seconds > 0.0 {
            Some((octets as f32 * 8.0) / seconds / 1_000_000.0)
        } else {
            None
        };

        let msg = NwdafMessage::CellLoad {
            cell_id: self.task_base.config.cell_id() as i32,
            prb_usage: None,
            connected_ues: self.ue_addresses.len() as u32,
            throughput_mbps,
        };
        if let Err(e) = sixg.nwdaf_tx.send(msg).await {
            warn!("RLS: failed to forward cell load to NWDAF: {}", e);
        }
    }

    /// Sends a heartbeat acknowledgment to a UE
    async fn send_heartbeat_ack(&self, dest: SocketAddr, ack: &RlsHeartbeatAck) {
        let msg = RlsProtocolMessage::HeartbeatAck(ack.clone());
        self.send_rls_message(dest, &msg).await;
    }

    /// Handles a PDU transmission from a UE
    async fn handle_pdu_transmission(
        &mut self,
        sti: u64,
        source: SocketAddr,
        pdu: &RlsPduTransmission,
    ) {
        // Get UE ID from STI
        let ue_id = match self.sti_to_ue_id.get(&sti) {
            Some(&id) => id,
            None => {
                warn!("PDU from unknown STI {}", sti);
                return;
            }
        };

        // Update UE address if changed
        self.ue_addresses.insert(ue_id, source);

        // Queue acknowledgment if PDU ID is non-zero
        if pdu.pdu_id != 0 {
            self.pending_acks.entry(ue_id).or_default().push(pdu.pdu_id);
        }

        match pdu.pdu_type {
            PduType::Rrc => {
                self.handle_uplink_rrc(ue_id, pdu).await;
            }
            PduType::Data => {
                self.handle_uplink_data(ue_id, pdu).await;
            }
            PduType::Reserved => {
                warn!("Received reserved PDU type from UE[{}]", ue_id);
            }
        }
    }

    /// Handles uplink RRC message from UE
    async fn handle_uplink_rrc(&self, ue_id: i32, pdu: &RlsPduTransmission) {
        let channel = match RrcChannel::from_u32(pdu.payload) {
            Some(ch) => ch,
            None => {
                warn!("Invalid RRC channel {} from UE[{}]", pdu.payload, ue_id);
                return;
            }
        };

        debug!(
            "Uplink RRC: ue_id={}, channel={:?}, len={}",
            ue_id,
            channel,
            pdu.pdu.len()
        );

        // Forward to RRC task
        let data = OctetString::from_slice(&pdu.pdu);
        let msg = RrcMessage::UplinkRrc {
            ue_id,
            rrc_channel: channel,
            data,
        };

        if let Err(e) = self.task_base.rrc_tx.send(msg).await {
            error!("Failed to send uplink RRC to RRC task: {}", e);
        }
    }

    /// Handles uplink user plane data from UE.
    ///
    /// The raw bytes arriving over RLS are an RLC PDU (UM, SN12).  They are
    /// fed into the per-UE RLC entity for reassembly; only complete SDUs are
    /// forwarded to the GTP task.
    async fn handle_uplink_data(&mut self, ue_id: i32, pdu: &RlsPduTransmission) {
        let psi = pdu.payload as i32;

        debug!(
            "Uplink data (RLC): ue_id={}, psi={}, len={}",
            ue_id,
            psi,
            pdu.pdu.len()
        );

        // Count the octets on the wire, not the reassembled SDU: the load being
        // measured is radio occupancy, which the RLC header and a discarded
        // duplicate both consume (issue #18).
        self.load_window_octets += pdu.pdu.len() as u64;

        // Feed the RLC PDU into the entity and collect any reassembled SDUs,
        // releasing the mutable borrow before the async send below.
        let reassembled_sdus = {
            let rlc = self.rlc_entity_for(ue_id, psi);
            rlc.receive_pdu(&pdu.pdu);
            let mut sdus = Vec::new();
            while let Some(sdu) = rlc.poll_reassembled() {
                sdus.push(sdu);
            }
            sdus
        };

        // An AM bearer answers a poll (or a detected gap) with a STATUS report
        // (TS 38.322 §5.3.4); without this the peer's ARQ never learns anything
        // and its t-PollRetransmit fires forever.
        self.send_pending_status(ue_id, psi).await;

        for sdu in reassembled_sdus {
            debug!(
                "RLC reassembled SDU: ue_id={}, psi={}, len={}",
                ue_id,
                psi,
                sdu.len()
            );
            let data = OctetString::from_slice(&sdu);
            let msg = GtpMessage::DataPduDelivery {
                ue_id,
                psi,
                pdu: data,
            };
            if let Err(e) = self.task_base.gtp_tx.send(msg).await {
                error!("Failed to send uplink data to GTP task: {}", e);
            }
        }
    }

    /// Handles PDU transmission acknowledgment
    fn handle_pdu_ack(&mut self, ack: &RlsPduTransmissionAck) {
        debug!("PDU ack received: {} PDUs acknowledged", ack.pdu_ids.len());
        // PDU acknowledgment tracking can be extended here if needed
    }

    /// Handles downlink RRC message from RRC task
    async fn handle_downlink_rrc(
        &mut self,
        ue_id: i32,
        rrc_channel: RrcChannel,
        pdu_id: u32,
        data: OctetString,
    ) {
        let dest = match self.ue_addresses.get(&ue_id) {
            Some(&addr) => addr,
            None => {
                warn!("Downlink RRC for unknown UE[{}]", ue_id);
                return;
            }
        };

        debug!(
            "Downlink RRC: ue_id={}, channel={:?}, pdu_id={}, len={}",
            ue_id,
            rrc_channel,
            pdu_id,
            data.len()
        );

        let pdu = RlsPduTransmission {
            sti: self.sti,
            pdu_type: PduType::Rrc,
            pdu_id,
            payload: rrc_channel as u32,
            pdu: Bytes::copy_from_slice(data.data()),
        };

        let msg = RlsProtocolMessage::PduTransmission(pdu);
        self.send_rls_message(dest, &msg).await;
    }

    /// Broadcasts an RRC PDU on a downlink common channel to every UE the cell
    /// has discovered (TS 38.331 §5.3.2.2 for PCCH paging).
    ///
    /// The real air interface has one transmission that every camped UE can
    /// receive; RLS is a unicast UDP transport, so the cell-wide broadcast is
    /// simulated by sending the same PDU to each discovered UE address. Distinct
    /// addresses are sent to once even when several UE IDs share one (a single
    /// `nr-ue` process holding several UEs), because the receiving RLS layer
    /// dispatches on the cell, not on the UE ID.
    ///
    /// Delivery is best-effort and unacknowledged, matching PCCH: there is no
    /// per-UE ack, no retransmission and no queueing for a UE that is not
    /// currently discovered.
    async fn handle_broadcast_rrc(
        &mut self,
        rrc_channel: RrcChannel,
        pdu_id: u32,
        data: OctetString,
    ) {
        if !rrc_channel.is_downlink() {
            warn!("Refusing to broadcast on uplink channel {rrc_channel:?}");
            return;
        }

        let mut destinations: Vec<SocketAddr> = self.ue_addresses.values().copied().collect();
        destinations.sort_unstable();
        destinations.dedup();

        if destinations.is_empty() {
            debug!(
                "Broadcast RRC on {:?} dropped: no UE discovered on this cell",
                rrc_channel
            );
            return;
        }

        debug!(
            "Broadcast RRC: channel={:?}, pdu_id={}, len={}, destinations={}",
            rrc_channel,
            pdu_id,
            data.len(),
            destinations.len()
        );

        let pdu = RlsPduTransmission {
            sti: self.sti,
            pdu_type: PduType::Rrc,
            pdu_id,
            payload: rrc_channel as u32,
            pdu: Bytes::copy_from_slice(data.data()),
        };
        let msg = RlsProtocolMessage::PduTransmission(pdu);

        for dest in destinations {
            self.send_rls_message(dest, &msg).await;
        }
    }

    /// Handles downlink user plane data from GTP task.
    ///
    /// The SDU from GTP is submitted to the per-UE RLC entity (UM, SN12).
    /// One or more RLC PDUs are then built and sent over RLS to the UE.
    /// A 1500-byte MTU is used as the MAC grant size so that typical IP
    /// packets fit in a single PDU.
    async fn handle_downlink_data(&mut self, ue_id: i32, psi: i32, data: OctetString) {
        let dest = match self.ue_addresses.get(&ue_id) {
            Some(&addr) => addr,
            None => {
                warn!("Downlink data for unknown UE[{}]", ue_id);
                return;
            }
        };

        debug!(
            "Downlink data (RLC): ue_id={}, psi={}, len={}",
            ue_id,
            psi,
            data.len()
        );

        // Submit SDU to RLC and collect all resulting PDUs before releasing
        // the mutable borrow so that self.sti and self.socket are accessible
        // again for transmission.
        let rlc_pdus = {
            let rlc = self.rlc_entity_for(ue_id, psi);
            rlc.submit_sdu(data.data().to_vec());
            let mut pdus = Vec::new();
            while let Some(rlc_pdu) = rlc.build_pdu(MAC_GRANT_BYTES) {
                pdus.push(rlc_pdu);
            }
            pdus
        };

        let sti = self.sti;
        for rlc_pdu in rlc_pdus {
            let transmission = RlsPduTransmission {
                sti,
                pdu_type: PduType::Data,
                pdu_id: 0, // Data PDUs don't require RLS acknowledgment
                payload: psi as u32,
                pdu: Bytes::from(rlc_pdu),
            };
            let msg = RlsProtocolMessage::PduTransmission(transmission);
            self.send_rls_message(dest, &msg).await;
        }
    }

    /// Sends an RLS message to a destination
    async fn send_rls_message(&self, dest: SocketAddr, msg: &RlsProtocolMessage) {
        let socket = match &self.socket {
            Some(s) => s,
            None => {
                error!("Cannot send RLS message: socket not initialized");
                return;
            }
        };

        let encoded = codec::encode(msg);
        if let Err(e) = socket.send_to(&encoded, dest).await {
            error!("Failed to send RLS message to {}: {}", dest, e);
        }
    }

    /// Sends pending acknowledgments to UEs
    async fn send_pending_acks(&mut self) {
        let pending = std::mem::take(&mut self.pending_acks);

        for (ue_id, pdu_ids) in pending {
            if pdu_ids.is_empty() {
                continue;
            }

            let dest = match self.ue_addresses.get(&ue_id) {
                Some(&addr) => addr,
                None => continue,
            };

            let ack = RlsPduTransmissionAck::with_pdu_ids(self.sti, pdu_ids);
            let msg = RlsProtocolMessage::PduTransmissionAck(ack);
            self.send_rls_message(dest, &msg).await;
        }
    }

    /// Checks for lost UEs and handles cleanup
    async fn check_lost_ues(&mut self) {
        let events = self.cell_tracker.check_lost_ues();

        for event in events {
            if let GnbTrackerEvent::UeLost { ue_id } = event {
                let ue_id_i32 = ue_id as i32;
                info!("UE[{}] lost due to heartbeat timeout", ue_id_i32);

                self.ue_addresses.remove(&ue_id_i32);
                self.pending_acks.remove(&ue_id_i32);

                // Find and remove STI mapping
                let sti_to_remove: Vec<u64> = self
                    .sti_to_ue_id
                    .iter()
                    .filter(|(_, &id)| id == ue_id_i32)
                    .map(|(&sti, _)| sti)
                    .collect();

                for sti in sti_to_remove {
                    self.sti_to_ue_id.remove(&sti);
                }
            }
        }
    }

    /// Receives data from the UDP socket
    #[allow(dead_code)]
    async fn receive_udp(&self) -> Option<(OctetString, SocketAddr)> {
        let socket = self.socket.as_ref()?;
        let mut buf = vec![0u8; UDP_BUFFER_SIZE];

        match socket.recv_from(&mut buf).await {
            Ok((len, addr)) => {
                buf.truncate(len);
                Some((OctetString::from_slice(&buf), addr))
            }
            Err(e) => {
                error!("UDP receive error: {}", e);
                None
            }
        }
    }
}

#[async_trait::async_trait]
impl Task for RlsTask {
    type Message = RlsMessage;

    async fn run(&mut self, mut rx: mpsc::Receiver<TaskMessage<Self::Message>>) {
        info!("RLS task starting");

        // Initialize UDP socket
        if let Err(e) = self.init_socket().await {
            error!("Failed to initialize RLS socket: {}", e);
            return;
        }

        info!("RLS task started on {}", self.bind_address);

        let mut heartbeat_timer = interval(Duration::from_millis(HEARTBEAT_CHECK_INTERVAL_MS));
        // `interval` fires its first tick immediately, and at start-up there is
        // nothing for it to do: no UE is known, no ack is pending and no RLC
        // entity exists. Consuming it here also makes the periodic work land at
        // predictable multiples of the period — otherwise the immediate tick
        // races the first received packet, so whether periodic work runs before
        // or after it is chance.
        heartbeat_timer.tick().await;

        loop {
            tokio::select! {
                // Handle messages from other tasks
                Some(msg) = rx.recv() => {
                    match msg {
                        TaskMessage::Message(rls_msg) => {
                            match rls_msg {
                                RlsMessage::ReceiveRlsMessage { data, source } => {
                                    self.handle_receive_rls_message(data, source).await;
                                }
                                RlsMessage::DownlinkRrc { ue_id, rrc_channel, pdu_id, data } => {
                                    self.handle_downlink_rrc(ue_id, rrc_channel, pdu_id, data).await;
                                }
                                RlsMessage::BroadcastRrc { rrc_channel, pdu_id, data } => {
                                    self.handle_broadcast_rrc(rrc_channel, pdu_id, data).await;
                                }
                                RlsMessage::DownlinkData { ue_id, psi, pdu } => {
                                    self.handle_downlink_data(ue_id, psi, pdu).await;
                                }
                                RlsMessage::SignalDetected { ue_id } => {
                                    debug!("Signal detected notification for UE[{}]", ue_id);
                                }
                                RlsMessage::SignalLost { ue_id } => {
                                    debug!("Signal lost notification for UE[{}]", ue_id);
                                    self.ue_addresses.remove(&ue_id);
                                }
                                RlsMessage::UplinkRrc { ue_id, rrc_channel, data } => {
                                    // Internal uplink RRC (forwarded from receive handler)
                                    let msg = RrcMessage::UplinkRrc { ue_id, rrc_channel, data };
                                    if let Err(e) = self.task_base.rrc_tx.send(msg).await {
                                        error!("Failed to forward uplink RRC: {}", e);
                                    }
                                }
                                RlsMessage::UplinkData { ue_id, psi, pdu } => {
                                    // Internal uplink data (forwarded from receive handler)
                                    let msg = GtpMessage::DataPduDelivery { ue_id, psi, pdu };
                                    if let Err(e) = self.task_base.gtp_tx.send(msg).await {
                                        error!("Failed to forward uplink data: {}", e);
                                    }
                                }
                                RlsMessage::RadioLinkFailure { ue_id, cause } => {
                                    warn!("Radio link failure for UE[{}]: {:?}", ue_id, cause);
                                }
                                RlsMessage::TransmissionFailure { pdu_list } => {
                                    warn!("Transmission failure: {} PDUs failed", pdu_list.len());
                                }
                            }
                        }
                        TaskMessage::Shutdown => {
                            info!("RLS task received shutdown signal");
                            break;
                        }
                    }
                }

                // Handle incoming UDP packets
                result = async {
                    if let Some(socket) = &self.socket {
                        let mut buf = vec![0u8; UDP_BUFFER_SIZE];
                        socket.recv_from(&mut buf).await.ok().map(|(len, addr)| {
                            buf.truncate(len);
                            (OctetString::from_slice(&buf), addr)
                        })
                    } else {
                        None
                    }
                } => {
                    if let Some((data, source)) = result {
                        self.handle_receive_rls_message(data, source).await;
                    }
                }

                // Periodic heartbeat check and ack sending
                _ = heartbeat_timer.tick() => {
                    self.check_lost_ues().await;
                    self.send_pending_acks().await;
                    self.poll_rlc_timers().await;
                    // Cell load rides the same tick rather than owning a timer:
                    // it is a rate over the elapsed window, which is measured,
                    // so the period only sets the reporting granularity.
                    self.report_cell_load_to_nwdaf().await;
                }
            }
        }

        info!(
            "RLS task stopped with {} tracked UEs",
            self.cell_tracker.ue_count()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn test_rls_task_creation() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, 16);
        let task = RlsTask::new(task_base);

        assert_eq!(task.sti, 0x000000010);
        assert!(task.ue_addresses.is_empty());
        assert!(task.sti_to_ue_id.is_empty());
    }

    #[test]
    fn test_rls_task_with_custom_address() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, 16);

        let bind_addr: SocketAddr = "127.0.0.1:5000".parse().unwrap();
        let task = RlsTask::with_bind_address(task_base, bind_addr);

        assert_eq!(task.bind_address, bind_addr);
    }

    // ========================================================================
    // Broadcast fan-out (#35): PCCH reaches every discovered UE, once each
    // ========================================================================

    /// A cell-side RLS task bound to an ephemeral port, with a live socket.
    async fn broadcasting_task() -> RlsTask {
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(test_config(), 16);
        let mut task =
            RlsTask::with_bind_address(task_base, "127.0.0.1:0".parse().expect("bind address"));
        task.init_socket().await.expect("RLS socket");
        task
    }

    /// Receives one datagram (with a bounded wait so a missing broadcast fails
    /// the test instead of hanging) and returns the RRC channel and payload it
    /// carries.
    async fn recv_rrc_pdu(socket: &UdpSocket) -> Option<(RrcChannel, Vec<u8>)> {
        let mut buf = vec![0u8; UDP_BUFFER_SIZE];
        let len = tokio::time::timeout(Duration::from_millis(500), socket.recv(&mut buf))
            .await
            .ok()?
            .ok()?;
        buf.truncate(len);
        match codec::decode(&Bytes::from(buf)).ok()? {
            RlsProtocolMessage::PduTransmission(pdu) => {
                Some((RrcChannel::from_u32(pdu.payload)?, pdu.pdu.to_vec()))
            }
            _ => None,
        }
    }

    #[tokio::test]
    async fn a_pcch_broadcast_reaches_every_discovered_ue() {
        let ue_a = UdpSocket::bind("127.0.0.1:0").await.expect("UE A socket");
        let ue_b = UdpSocket::bind("127.0.0.1:0").await.expect("UE B socket");

        let mut task = broadcasting_task().await;
        task.ue_addresses.insert(1, ue_a.local_addr().unwrap());
        task.ue_addresses.insert(2, ue_b.local_addr().unwrap());

        let payload = OctetString::from_slice(&[0x20, 0x00, 0x48]);
        task.handle_broadcast_rrc(RrcChannel::Pcch, 0, payload)
            .await;

        for (name, socket) in [("A", &ue_a), ("B", &ue_b)] {
            let (channel, data) = recv_rrc_pdu(socket)
                .await
                .unwrap_or_else(|| panic!("UE {name} received no broadcast"));
            assert_eq!(channel, RrcChannel::Pcch);
            assert_eq!(data, vec![0x20, 0x00, 0x48]);
        }
    }

    /// Two UE IDs behind one address (one `nr-ue` process holding several UEs)
    /// must get ONE transmission, not one per UE: the receiving RLS layer
    /// dispatches the PDU on the cell, so a duplicate would be processed twice
    /// and a UE paged twice would start two service requests.
    #[tokio::test]
    async fn ue_ids_sharing_an_address_receive_one_transmission() {
        let ue = UdpSocket::bind("127.0.0.1:0").await.expect("UE socket");
        let addr = ue.local_addr().unwrap();

        let mut task = broadcasting_task().await;
        task.ue_addresses.insert(1, addr);
        task.ue_addresses.insert(2, addr);

        task.handle_broadcast_rrc(RrcChannel::Pcch, 0, OctetString::from_slice(&[0x20]))
            .await;

        assert!(
            recv_rrc_pdu(&ue).await.is_some(),
            "the shared address must receive the broadcast"
        );
        assert!(
            recv_rrc_pdu(&ue).await.is_none(),
            "and must not receive it a second time"
        );
    }

    /// An uplink channel is not broadcastable; the guard exists so a wiring
    /// mistake is refused rather than transmitted.
    #[tokio::test]
    async fn a_broadcast_on_an_uplink_channel_is_refused() {
        let ue = UdpSocket::bind("127.0.0.1:0").await.expect("UE socket");

        let mut task = broadcasting_task().await;
        task.ue_addresses.insert(1, ue.local_addr().unwrap());

        task.handle_broadcast_rrc(RrcChannel::UlDcch, 0, OctetString::from_slice(&[0x01]))
            .await;

        assert!(recv_rrc_pdu(&ue).await.is_none());
    }

    /// System information must survive the transport, not just the codec: the
    /// MIB and SIB1 are broadcast through the real RLS path and decoded on the
    /// far side, on the BCCH channels TS 38.331 §5.2.1 assigns them.
    #[tokio::test]
    async fn system_information_reaches_a_ue_over_the_rls_transport() {
        use crate::rrc::system_info::{encode_cell_mib, encode_cell_sib1};
        use nextgsim_rrc::procedures::system_information::{decode_mib, decode_sib1};

        let ue = UdpSocket::bind("127.0.0.1:0").await.expect("UE socket");
        let config = test_config();
        let (nci, tac) = (config.nci, config.tac);
        let mib = encode_cell_mib().expect("MIB");
        let sib1 = encode_cell_sib1(&config).expect("SIB1");

        let mut task = broadcasting_task().await;
        task.ue_addresses.insert(1, ue.local_addr().unwrap());

        task.handle_broadcast_rrc(RrcChannel::BcchBch, 0, OctetString::from_slice(&mib))
            .await;
        task.handle_broadcast_rrc(RrcChannel::BcchDlSch, 0, OctetString::from_slice(&sib1))
            .await;

        let (mib_channel, mib_bytes) = recv_rrc_pdu(&ue).await.expect("the MIB must arrive");
        assert_eq!(mib_channel, RrcChannel::BcchBch);
        decode_mib(&mib_bytes).expect("the MIB must decode after the transport");

        let (sib1_channel, sib1_bytes) = recv_rrc_pdu(&ue).await.expect("SIB1 must arrive");
        assert_eq!(sib1_channel, RrcChannel::BcchDlSch);
        let decoded = decode_sib1(&sib1_bytes).expect("SIB1 must decode after the transport");
        let info = &decoded.plmn_identity_info_list[0];
        assert_eq!(
            (info.cell_identity, info.tracking_area_code),
            (nci, Some(tac)),
            "the cell's own identity survives the round trip"
        );
    }

    // ========================================================================
    // Per-bearer RLC entities (#34, TS 38.322 §4.2.1)
    // ========================================================================

    /// Receives one datagram and returns the PSI it was sent on plus the decoded
    /// UM PDU.
    async fn recv_um_pdu(socket: &UdpSocket) -> Option<(u32, nextgsim_rlc::RlcUmPdu)> {
        let mut buf = vec![0u8; UDP_BUFFER_SIZE];
        let len = tokio::time::timeout(Duration::from_millis(500), socket.recv(&mut buf))
            .await
            .ok()?
            .ok()?;
        buf.truncate(len);
        match codec::decode(&Bytes::from(buf)).ok()? {
            RlsProtocolMessage::PduTransmission(pdu) => Some((
                pdu.payload,
                nextgsim_rlc::RlcUmPdu::decode_sn12(&pdu.pdu).ok()?,
            )),
            _ => None,
        }
    }

    /// Each PDU session is its own radio bearer, so each owns its sequence-number
    /// space: the first SDU of BOTH sessions must be SN 0. Keyed on the UE alone,
    /// the second session's first SDU went out as SN 1 into a UE entity that was
    /// waiting for SN 0 — silent corruption rather than a clean failure.
    #[tokio::test]
    async fn two_pdu_sessions_get_independent_sequence_number_spaces() {
        let ue = UdpSocket::bind("127.0.0.1:0").await.expect("UE socket");
        let mut task = broadcasting_task().await;
        task.ue_addresses.insert(1, ue.local_addr().unwrap());

        task.handle_downlink_data(1, 1, OctetString::from_slice(&[0x11; 16]))
            .await;
        task.handle_downlink_data(1, 5, OctetString::from_slice(&[0x55; 16]))
            .await;

        let (first_psi, first) = recv_um_pdu(&ue).await.expect("PSI 1 PDU");
        let (second_psi, second) = recv_um_pdu(&ue).await.expect("PSI 5 PDU");

        assert_eq!((first_psi, second_psi), (1, 5));
        assert_eq!(first.sn, 0, "PSI 1 starts its own SN space at 0");
        assert_eq!(
            second.sn, 0,
            "PSI 5 must start at 0 too, not continue PSI 1's counter"
        );
        assert_eq!(
            task.rlc_entities.len(),
            2,
            "one RLC entity per (UE, bearer)"
        );
    }

    /// The `t-Reassembly` tick is wired into the task's run loop, not just
    /// implemented on the entity: a partial SDU abandoned by the timer must NOT
    /// be resurrected by a segment that arrives late.
    ///
    /// Drives the real run loop over loopback UDP with a test socket standing in
    /// for the UE's radio, because the tick lives in the loop's periodic arm and
    /// nothing else would exercise it.
    #[tokio::test]
    async fn a_late_segment_cannot_complete_an_sdu_t_reassembly_abandoned() {
        let ue = UdpSocket::bind("127.0.0.1:0").await.expect("UE socket");
        let gnb_addr = {
            let probe = UdpSocket::bind("127.0.0.1:0").await.expect("probe");
            probe.local_addr().unwrap()
        };

        let (task_base, _app_rx, _ngap_rx, _rrc_rx, mut gtp_rx, rls_rx, _sctp_rx) =
            GnbTaskBase::new(test_config(), 32);
        let mut task = RlsTask::with_bind_address(task_base, gnb_addr);
        tokio::spawn(async move { task.run(rls_rx).await });

        // A heartbeat makes the cell tracker discover this "UE".
        let heartbeat = codec::encode(&RlsProtocolMessage::Heartbeat(nextgsim_rls::RlsHeartbeat {
            sti: 0xDEAD_BEEF,
            sim_pos: SimCoord::new(0, 0, 0),
        }));
        ue.send_to(&heartbeat, gnb_addr).await.expect("heartbeat");

        let data_pdu = |pdu: Vec<u8>| {
            codec::encode(&RlsProtocolMessage::PduTransmission(RlsPduTransmission {
                sti: 0xDEAD_BEEF,
                pdu_type: PduType::Data,
                pdu_id: 0,
                payload: 1, // PSI 1
                pdu: Bytes::from(pdu),
            }))
        };
        let um = |si: nextgsim_rlc::SegmentationInfo, sn: u16, so: Option<u16>, data: Vec<u8>| {
            nextgsim_rlc::RlcUmPdu { si, sn, so, data }.encode_sn12()
        };

        // POSITIVE CONTROL first: a complete SDU must arrive at GTP. Without it
        // the negative assertion below would also pass if the heartbeat were
        // rejected or the data path were broken for an unrelated reason.
        ue.send_to(
            &data_pdu(um(
                nextgsim_rlc::SegmentationInfo::FullSdu,
                0,
                None,
                vec![0xEE; 4],
            )),
            gnb_addr,
        )
        .await
        .expect("complete SDU");
        let delivered = tokio::time::timeout(Duration::from_secs(2), gtp_rx.recv())
            .await
            .expect("the data path must deliver a complete SDU");
        assert!(
            matches!(
                delivered,
                Some(TaskMessage::Message(GtpMessage::DataPduDelivery { .. }))
            ),
            "expected the complete SDU at GTP, got {delivered:?}"
        );

        // Now only the FIRST segment of SN 1: the rest never arrives.
        ue.send_to(
            &data_pdu(um(
                nextgsim_rlc::SegmentationInfo::FirstSegment,
                1,
                None,
                vec![1, 2, 3, 4],
            )),
            gnb_addr,
        )
        .await
        .expect("first segment");

        // The run loop's periodic arm ticks every 500 ms and t-Reassembly is
        // 50 ms, so one tick is enough; two are allowed for scheduling slack.
        tokio::time::sleep(Duration::from_millis(1200)).await;

        // The last segment arrives after the SDU was abandoned.
        ue.send_to(
            &data_pdu(um(
                nextgsim_rlc::SegmentationInfo::LastSegment,
                1,
                Some(4),
                vec![5, 6],
            )),
            gnb_addr,
        )
        .await
        .expect("last segment");
        tokio::time::sleep(Duration::from_millis(200)).await;

        assert!(
            gtp_rx.try_recv().is_err(),
            "an SDU t-Reassembly abandoned must not be delivered by a late segment"
        );
    }

    // ========================================================================
    // AM ARQ through the real task (#15)
    // ========================================================================

    /// A gNB RLS task whose PSI 5 bearer is RLC AM, bound and running its loop.
    fn am_config() -> GnbConfig {
        GnbConfig {
            rlc_am_psis: vec![5],
            ..test_config()
        }
    }

    #[test]
    fn only_a_configured_psi_gets_an_acknowledged_mode_entity() {
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(am_config(), 16);
        let mut task = RlsTask::new(task_base);

        assert_eq!(
            task.rlc_entity_for(1, 5).mode,
            RlcMode::AcknowledgedMode,
            "PSI 5 is configured for AM"
        );
        assert_eq!(
            task.rlc_entity_for(1, 1).mode,
            RlcMode::UnacknowledgedMode,
            "an unlisted PSI keeps the UM default"
        );
    }

    /// A poll must be answered from the receive path, not by waiting for the run
    /// loop's periodic tick: the tick is 500 ms and the sender's
    /// `t-PollRetransmit` is 45 ms, so a tick-only answer would have the sender
    /// re-poll ten times before hearing anything.
    #[tokio::test]
    async fn a_poll_is_answered_without_waiting_for_the_timer_tick() {
        let ue = UdpSocket::bind("127.0.0.1:0").await.expect("UE socket");
        let gnb_addr = {
            let probe = UdpSocket::bind("127.0.0.1:0").await.expect("probe");
            probe.local_addr().unwrap()
        };

        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, rls_rx, _sctp_rx) =
            GnbTaskBase::new(am_config(), 32);
        let mut task = RlsTask::with_bind_address(task_base, gnb_addr);
        tokio::spawn(async move { task.run(rls_rx).await });

        let heartbeat = codec::encode(&RlsProtocolMessage::Heartbeat(nextgsim_rls::RlsHeartbeat {
            sti: 0x0BAD_CAFE,
            sim_pos: SimCoord::new(0, 0, 0),
        }));
        ue.send_to(&heartbeat, gnb_addr).await.expect("heartbeat");

        let mut ue_rlc = RlcEntity::new(RlcMode::AcknowledgedMode, SnSize::Sn12);
        ue_rlc.submit_sdu(vec![0xD1u8; 16]);
        let polled = ue_rlc.build_pdu(64).expect("a polled AM PDU");
        let framed = codec::encode(&RlsProtocolMessage::PduTransmission(RlsPduTransmission {
            sti: 0x0BAD_CAFE,
            pdu_type: PduType::Data,
            pdu_id: 0,
            payload: 5,
            pdu: Bytes::from(polled),
        }));
        ue.send_to(&framed, gnb_addr).await.unwrap();

        // 150 ms: far above loopback latency, far below the 500 ms tick.
        let status = tokio::time::timeout(Duration::from_millis(150), async {
            loop {
                let mut buf = vec![0u8; UDP_BUFFER_SIZE];
                let len = ue.recv(&mut buf).await.expect("recv");
                buf.truncate(len);
                if let Ok(RlsProtocolMessage::PduTransmission(pdu)) =
                    codec::decode(&Bytes::from(buf))
                {
                    if pdu.pdu_type == PduType::Data && (pdu.pdu[0] & 0x80) == 0 {
                        return pdu.pdu.to_vec();
                    }
                }
            }
        })
        .await
        .expect("the poll must be answered from the receive path");

        let decoded = nextgsim_rlc::RlcStatusPdu::decode_sn12(&status).unwrap();
        assert_eq!(decoded.ack_sn, 1, "the polled SDU is acknowledged");
    }

    /// The AM loop end to end through the running task, with a PDU dropped on
    /// purpose: the gNB must NACK it, the peer must resend it on the strength of
    /// that NACK alone, and the SDU must then reach GTP.
    ///
    /// The "UE" here is a plain socket plus its own `RlcEntity`, so both sides of
    /// the ARQ exchange are real RLC code and only the loss is simulated.
    #[tokio::test]
    async fn an_am_bearer_recovers_a_dropped_pdu_through_the_task() {
        let ue = UdpSocket::bind("127.0.0.1:0").await.expect("UE socket");
        let gnb_addr = {
            let probe = UdpSocket::bind("127.0.0.1:0").await.expect("probe");
            probe.local_addr().unwrap()
        };

        let (task_base, _app_rx, _ngap_rx, _rrc_rx, mut gtp_rx, rls_rx, _sctp_rx) =
            GnbTaskBase::new(am_config(), 32);
        let mut task = RlsTask::with_bind_address(task_base, gnb_addr);
        tokio::spawn(async move { task.run(rls_rx).await });

        // Discovery, so the gNB knows where to send the STATUS report back.
        let heartbeat = codec::encode(&RlsProtocolMessage::Heartbeat(nextgsim_rls::RlsHeartbeat {
            sti: 0x0BAD_F00D,
            sim_pos: SimCoord::new(0, 0, 0),
        }));
        ue.send_to(&heartbeat, gnb_addr).await.expect("heartbeat");

        // The UE side of the AM bearer, with its own entity.
        let mut ue_rlc = RlcEntity::new(RlcMode::AcknowledgedMode, SnSize::Sn12);
        let sdus: Vec<Vec<u8>> = (0..3u8).map(|i| vec![0xC0 + i; 20]).collect();
        for sdu in &sdus {
            ue_rlc.submit_sdu(sdu.clone());
        }
        let pdus: Vec<Vec<u8>> = std::iter::from_fn(|| ue_rlc.build_pdu(64)).collect();
        assert_eq!(pdus.len(), 3);

        let wrap = |pdu: Vec<u8>| {
            codec::encode(&RlsProtocolMessage::PduTransmission(RlsPduTransmission {
                sti: 0x0BAD_F00D,
                pdu_type: PduType::Data,
                pdu_id: 0,
                payload: 5, // PSI 5 — the AM bearer
                pdu: Bytes::from(pdu),
            }))
        };

        // Drop SN 1 on the way up.
        ue.send_to(&wrap(pdus[0].clone()), gnb_addr).await.unwrap();
        ue.send_to(&wrap(pdus[2].clone()), gnb_addr).await.unwrap();

        // Two SDUs arrive; the third is the one that was lost.
        for _ in 0..2 {
            let delivered = tokio::time::timeout(Duration::from_secs(2), gtp_rx.recv())
                .await
                .expect("the received SDUs must reach GTP");
            assert!(matches!(
                delivered,
                Some(TaskMessage::Message(GtpMessage::DataPduDelivery { .. }))
            ));
        }

        // The poll on SN 2 is answered with an ACK-only report first: TS 38.322
        // §5.2.3.2.3 advances RX_Highest_Status only over SDUs that ARE received,
        // so the gap is not reported until the gNB's t-Reassembly declares it
        // lost (§5.2.3.2.4) and the run loop's tick sends the second report.
        let (reports, status) = tokio::time::timeout(Duration::from_secs(3), async {
            let mut reports = 0usize;
            loop {
                let mut buf = vec![0u8; UDP_BUFFER_SIZE];
                let len = ue.recv(&mut buf).await.expect("recv");
                buf.truncate(len);
                let Ok(RlsProtocolMessage::PduTransmission(pdu)) = codec::decode(&Bytes::from(buf))
                else {
                    continue;
                };
                if pdu.pdu_type != PduType::Data || (pdu.pdu[0] & 0x80) != 0 {
                    continue;
                }
                reports += 1;
                let decoded = nextgsim_rlc::RlcStatusPdu::decode_sn12(&pdu.pdu)
                    .expect("a decodable STATUS report");
                if !decoded.nacks.is_empty() {
                    return (reports, pdu.pdu.to_vec());
                }
            }
        })
        .await
        .expect("the gNB must send a STATUS report naming the gap");
        assert!(
            reports >= 2,
            "the poll should be answered before the timer reports the gap, got {reports} report(s)"
        );

        let decoded = nextgsim_rlc::RlcStatusPdu::decode_sn12(&status).unwrap();
        assert_eq!(
            decoded.nacks.iter().map(|n| n.nack_sn).collect::<Vec<_>>(),
            vec![1],
            "the STATUS report must NACK exactly the dropped SN"
        );

        // The NACK alone drives the retransmission — no manual request.
        ue_rlc.receive_pdu(&status);
        let retx = ue_rlc
            .build_pdu(64)
            .expect("the NACK must schedule the retransmission");
        ue.send_to(&wrap(retx), gnb_addr).await.unwrap();

        let recovered = tokio::time::timeout(Duration::from_secs(2), gtp_rx.recv())
            .await
            .expect("the retransmitted SDU must reach GTP");
        match recovered {
            Some(TaskMessage::Message(GtpMessage::DataPduDelivery { psi, pdu, .. })) => {
                assert_eq!(psi, 5);
                assert_eq!(pdu.data(), &sdus[1][..], "the recovered SDU is intact");
            }
            other => panic!("expected the recovered SDU at GTP, got {other:?}"),
        }
    }

    /// The uplink direction keys the same way, so two sessions reassembling at
    /// once do not share one buffer: both hold their own SN 0 partial SDU.
    #[tokio::test]
    async fn two_pdu_sessions_reassemble_in_separate_buffers() {
        let mut task = broadcasting_task().await;

        let first_segment = nextgsim_rlc::RlcUmPdu {
            si: nextgsim_rlc::SegmentationInfo::FirstSegment,
            sn: 0,
            so: None,
            data: vec![1, 2, 3, 4],
        }
        .encode_sn12();

        for psi in [1u32, 5u32] {
            let pdu = RlsPduTransmission {
                sti: 0,
                pdu_type: PduType::Data,
                pdu_id: 0,
                payload: psi,
                pdu: Bytes::from(first_segment.clone()),
            };
            task.handle_uplink_data(1, &pdu).await;
        }

        assert_eq!(task.rlc_entities.len(), 2, "one entity per (UE, bearer)");
        for (key, entity) in &task.rlc_entities {
            assert_eq!(
                entity.reassembly_buffer_len(),
                1,
                "{key:?} must hold its own partial SDU"
            );
        }
    }

    // ------------------------------------------------------------------------
    // NWDAF forwarding (issue #18)
    // ------------------------------------------------------------------------

    /// An RLS task with 6G handles wired, plus the NWDAF receiver to observe.
    fn rls_task_with_nwdaf() -> (RlsTask, mpsc::Receiver<TaskMessage<NwdafMessage>>) {
        let (mut task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(test_config(), 16);
        let sixg = task_base.init_6g_tasks(16);
        (RlsTask::new(task_base), sixg.nwdaf_rx)
    }

    #[tokio::test]
    async fn a_heartbeat_forwards_a_real_rsrp_and_position_to_nwdaf() {
        let (mut task, mut nwdaf_rx) = rls_task_with_nwdaf();
        // The UE has to be known to the tracker before its measurement has an id
        // to be reported under.
        task.sti_to_ue_id.insert(0xABCD, 7);

        let heartbeat =
            nextgsim_rls::RlsHeartbeat::with_position(0xABCD, SimCoord::new(10, 20, 30));
        let ack = RlsHeartbeatAck::with_dbm(0xABCD, -87);
        task.forward_measurement_to_nwdaf(0xABCD, &ack, &heartbeat)
            .await;

        let Some(TaskMessage::Message(NwdafMessage::UeMeasurement {
            ue_id,
            rsrp,
            rsrq,
            position,
        })) = nwdaf_rx.recv().await
        else {
            panic!("a heartbeat must produce a UE measurement");
        };
        assert_eq!(ue_id, 7);
        // The channel model's dBm, not a placeholder: -87 is what the ack carried.
        assert_eq!(rsrp, Some(-87.0));
        // RSRQ stays absent rather than invented -- RLS models no interference.
        assert_eq!(rsrq, None);
        assert_eq!(position, (10.0, 20.0, 30.0));
    }

    #[tokio::test]
    async fn a_heartbeat_from_an_unadmitted_ue_reports_nothing() {
        // Reporting under a placeholder id would merge two UEs' histories.
        let (task, mut nwdaf_rx) = rls_task_with_nwdaf();
        let heartbeat = nextgsim_rls::RlsHeartbeat::with_position(0x1111, SimCoord::new(1, 2, 3));
        let ack = RlsHeartbeatAck::with_dbm(0x1111, -70);
        task.forward_measurement_to_nwdaf(0x1111, &ack, &heartbeat)
            .await;

        assert!(nwdaf_rx.try_recv().is_err(), "no id, no measurement");
    }

    #[tokio::test]
    async fn a_cell_load_report_carries_the_measured_traffic_and_the_live_ue_count() {
        let (mut task, mut nwdaf_rx) = rls_task_with_nwdaf();
        task.ue_addresses
            .insert(1, "127.0.0.1:1000".parse().unwrap());
        task.ue_addresses
            .insert(2, "127.0.0.1:1001".parse().unwrap());
        task.load_window_octets = 125_000; // 1 Mbit
        task.load_window_start = Instant::now() - Duration::from_secs(1);

        task.report_cell_load_to_nwdaf().await;

        let Some(TaskMessage::Message(NwdafMessage::CellLoad {
            cell_id,
            prb_usage,
            connected_ues,
            throughput_mbps,
        })) = nwdaf_rx.recv().await
        else {
            panic!("the load timer must produce a cell load report");
        };
        assert_eq!(cell_id, test_config().cell_id() as i32);
        // No PRB scheduler exists, so this must NOT carry a fabricated figure.
        assert_eq!(prb_usage, None);
        assert_eq!(connected_ues, 2);
        // 125 000 octets over ~1 s is ~1 Mbps. Bounded rather than exact because
        // the window is real elapsed time.
        let mbps = throughput_mbps.expect("measured throughput");
        assert!(
            (0.9..1.2).contains(&mbps),
            "expected ~1 Mbps from 125 000 octets in ~1 s, got {mbps}"
        );
    }

    #[tokio::test]
    async fn a_reported_window_is_not_counted_again_in_the_next_one() {
        let (mut task, mut nwdaf_rx) = rls_task_with_nwdaf();
        task.load_window_octets = 100_000;
        task.load_window_start = Instant::now() - Duration::from_secs(1);
        task.report_cell_load_to_nwdaf().await;
        let _ = nwdaf_rx.recv().await;

        // Second window, no traffic: the counter must have been reset, so this
        // reports zero rather than re-reporting the first window's octets.
        task.load_window_start = Instant::now() - Duration::from_secs(1);
        task.report_cell_load_to_nwdaf().await;

        let Some(TaskMessage::Message(NwdafMessage::CellLoad {
            throughput_mbps, ..
        })) = nwdaf_rx.recv().await
        else {
            panic!("a second report is due");
        };
        assert_eq!(throughput_mbps, Some(0.0), "the window must have reset");
    }

    #[tokio::test]
    async fn user_plane_octets_are_counted_in_both_directions() {
        let (mut task, _nwdaf_rx) = rls_task_with_nwdaf();
        assert_eq!(task.load_window_octets, 0);

        // Uplink: an RLC PDU arriving over RLS.
        let psi = 1;
        let mut rlc = RlcEntity::new(RlcMode::UnacknowledgedMode, SnSize::Sn12);
        rlc.submit_sdu(vec![0x42; 40]);
        let pdu_bytes = rlc.build_pdu(MAC_GRANT_BYTES).expect("a PDU to send");
        let uplink_len = pdu_bytes.len() as u64;
        let pdu = RlsPduTransmission {
            sti: 0,
            pdu_type: PduType::Data,
            pdu_id: 0,
            payload: psi as u32,
            pdu: Bytes::from(pdu_bytes.clone()),
        };
        task.handle_uplink_data(1, &pdu).await;
        assert_eq!(task.load_window_octets, uplink_len, "uplink counted");

        // Downlink: a PDU sent to a known UE.
        task.ue_addresses
            .insert(1, "127.0.0.1:1000".parse().unwrap());
        task.send_rlc_pdu(1, psi, pdu_bytes.clone()).await;
        assert_eq!(
            task.load_window_octets,
            uplink_len * 2,
            "downlink counted too"
        );
    }

    #[tokio::test]
    async fn without_6g_handles_nothing_is_forwarded_and_the_window_still_resets() {
        // The default build has no 6G tasks; the counters must not grow without
        // bound just because nobody is listening.
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(test_config(), 16);
        let mut task = RlsTask::new(task_base);
        task.load_window_octets = 500_000;

        task.report_cell_load_to_nwdaf().await;

        assert_eq!(task.load_window_octets, 0);
    }
}
