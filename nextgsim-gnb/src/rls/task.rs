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
#[cfg(feature = "drb-pdcp")]
use nextgsim_pdcp::{Pdcp, PdcpConfig};
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

/// MAC grant size handed to RLC when building PDUs, for a UE under no
/// capability-derived restriction.
///
/// This simulator has no MAC scheduler; a 1500-byte grant stands in for one so a
/// typical IP packet fits in a single PDU. It is the *cell* budget: the whole of
/// what a normal FR1 UE (100 MHz / 273 PRB at 30 kHz SCS, TS 38.101-1
/// Table 5.3.2-1) may be handed in one transmission here.
///
/// A RedCap UE gets less, and gets it through [`RlsTask::mac_grant_for`] rather
/// than by reading this constant directly — see that function for why the ceiling
/// is applied on this value and not on a resource grid.
///
/// `pub(crate)` so the RRC task can scale a per-UE ceiling *off this value* when it
/// derives one (issue #57). Writing 1500 there instead would make the cell budget
/// two constants that must be changed together, and the ceiling would silently stop
/// matching the grant it is supposed to bound.
pub(crate) const MAC_GRANT_BYTES: usize = 1500;

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
    /// RLC entities keyed by `(UE ID, DRB identity)` — one per radio bearer, as
    /// TS 38.322 §4.2.1 requires, so each bearer owns its sequence-number space
    /// and reassembly buffer.
    ///
    /// Keyed on the DRB identity and not the PSI since issue #44: with
    /// `sdap-dataplane` one PDU session has two DRBs, and a PSI key would make them
    /// share an entity — interleaving two flows into one sequence-number space while
    /// the UE demultiplexed them into two entities each expecting a contiguous
    /// sequence, which is silent corruption rather than a clean failure. Without the
    /// feature the DRB identity IS the PSI, so the keys are exactly as they were.
    rlc_entities: HashMap<(i32, i32), RlcEntity>,
    /// PDCP entities keyed by `(UE ID, DRB identity)` -- one per DRB (TS 38.323 §5.2,
    /// issue #33). Keyed the same way as the RLC entities, because a PDCP entity
    /// and its RLC entity serve the same bearer.
    #[cfg(feature = "drb-pdcp")]
    pdcp_entities: HashMap<(i32, i32), Pdcp>,
    /// The PDU session each `(UE ID, DRB identity)` belongs to (issue #44).
    ///
    /// Needed because the radio side is per-DRB while GTP-U is per-session: an uplink
    /// SDU arrives on a DRB and has to leave on that session's N3 tunnel, so the
    /// forwarding needs the PSI back. Recorded when the bearer is first seen rather
    /// than recomputed, because DRB→PSI is not a function this side can evaluate:
    /// `allocate_drbs` wraps two identities into 1..=32, so inverting it would take a
    /// search that can match the wrong session.
    drb_to_psi: HashMap<(i32, i32), i32>,
    /// When this task started, the origin for the PDCP timers.
    #[cfg(feature = "drb-pdcp")]
    started_at: Instant,
    /// User-plane octets moved since the last cell-load report (uplink plus
    /// downlink). Counted here because the RLS task is the only place in the gNB
    /// that sees every user-plane PDU on the radio side, so it is the only
    /// producer that can report a *measured* load rather than a guessed one.
    load_window_octets: u64,
    /// When the current cell-load reporting window opened. The report is a rate,
    /// so the elapsed time has to be measured rather than assumed from the timer
    /// period -- a busy task ticks late and would otherwise overstate throughput.
    load_window_start: Instant,
    /// The MAC grant ceiling in force for one UE, in octets, for those UEs whose
    /// declared capabilities bound them below the cell budget (issue #57).
    ///
    /// Absent means unrestricted, which is every UE until RRC says otherwise: a
    /// normal UE has no capability-derived ceiling to record, so the map stays
    /// empty in the common case rather than holding `MAC_GRANT_BYTES` per UE.
    ///
    /// Populated from [`RlsMessage::SetUeGrantCeiling`], which the RRC task sends
    /// when it learns a UE is RedCap from `supportOfRedCap-r17`
    /// (TS 38.331 §6.3.3). Read by [`RlsTask::mac_grant_for`] at every point a
    /// grant is handed to RLC, which is what makes the ceiling enforced rather
    /// than merely stored -- the defect issue #57 was filed about.
    ue_grant_ceiling: HashMap<i32, usize>,
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
            #[cfg(feature = "drb-pdcp")]
            pdcp_entities: HashMap::new(),
            drb_to_psi: HashMap::new(),
            #[cfg(feature = "drb-pdcp")]
            started_at: Instant::now(),
            load_window_octets: 0,
            load_window_start: Instant::now(),
            ue_grant_ceiling: HashMap::new(),
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
            #[cfg(feature = "drb-pdcp")]
            pdcp_entities: HashMap::new(),
            drb_to_psi: HashMap::new(),
            #[cfg(feature = "drb-pdcp")]
            started_at: Instant::now(),
            load_window_octets: 0,
            load_window_start: Instant::now(),
            ue_grant_ceiling: HashMap::new(),
        }
    }

    /// The PDCP entity for one UE's DRB, created on first use (issue #33).
    ///
    /// Keyed on `drb_id`, not the PSI: TS 38.323 §5.2 gives every DRB its own entity,
    /// and with `sdap-dataplane` a session has two of them. Sharing one would have the
    /// two bearers share a COUNT, which under `up-security` means two flows ciphered
    /// with the same keystream.
    #[cfg(feature = "drb-pdcp")]
    fn pdcp_entity_for(&mut self, ue_id: i32, drb_id: i32) -> &mut Pdcp {
        self.pdcp_entities
            .entry((ue_id, drb_id))
            .or_insert_with(|| Pdcp::new(PdcpConfig::default()))
    }

    /// Install (or remove) user-plane security on one UE's DRB (issue #32).
    ///
    /// Applied to the entity for `(ue_id, drb_id)`, creating it if the DRB has not
    /// carried a packet yet — the keys arrive from NGAP before the first G-PDU does,
    /// and an entity created later with no security would leave the first packets
    /// unprotected.
    ///
    /// `drb_id` and not the PSI since issue #44, and the distinction is load-bearing
    /// under `up-security`: the `PdcpSecurity` NGAP builds binds BEARER to the DRB
    /// identity minus one (TS 33.501 Annex D.3.1.2), so installing it on a
    /// PSI-keyed entity would put the right BEARER on the wrong bearer's entity and
    /// every MAC-I would fail with no other symptom.
    #[cfg(feature = "up-security")]
    fn install_drb_security(
        &mut self,
        ue_id: i32,
        drb_id: i32,
        security: Option<nextgsim_pdcp::PdcpSecurity>,
    ) {
        let protected = security.is_some();
        self.pdcp_entity_for(ue_id, drb_id).set_security(security);
        info!(
            "DRB user-plane security {} for UE[{}] DRB {}",
            if protected { "installed" } else { "removed" },
            ue_id,
            drb_id
        );
    }

    /// Milliseconds since the task started, for the PDCP timers.
    #[cfg(feature = "drb-pdcp")]
    fn pdcp_now_ms(&self) -> u64 {
        self.started_at.elapsed().as_millis() as u64
    }

    /// Returns the RLC entity for one UE's radio bearer, creating a UM entity on
    /// first use.
    ///
    /// Keyed on `(ue_id, drb_id)` because TS 38.322 §4.2.1 gives every radio bearer
    /// its own RLC entity, and therefore its own sequence-number space and
    /// reassembly buffer. Keying on the UE alone interleaved every PDU session
    /// into one SN counter while the UE demultiplexed them into per-bearer entities
    /// each expecting a contiguous sequence — so a second PDU session corrupted
    /// both sessions' numbering rather than failing cleanly. Keying on the PSI has
    /// the same failure one level down once `sdap-dataplane` gives a session two DRBs.
    ///
    /// `psi` comes along even though it is not the key, for two reasons. The RLC
    /// **mode** is configured per PDU session (`rlc_am_psis` is a list of PSIs), so
    /// the AM decision can only be made from the PSI; and the uplink has to find its
    /// way back to a GTP-U tunnel, which is per session, so [`Self::drb_to_psi`] is
    /// recorded here — the one place every bearer passes through.
    fn rlc_entity_for(&mut self, ue_id: i32, drb_id: i32, psi: i32) -> &mut RlcEntity {
        let mode = if self.task_base.config.rlc_am_psis.contains(&(psi as u8)) {
            RlcMode::AcknowledgedMode
        } else {
            RlcMode::UnacknowledgedMode
        };
        self.drb_to_psi.insert((ue_id, drb_id), psi);
        self.rlc_entities
            .entry((ue_id, drb_id))
            .or_insert_with(|| RlcEntity::new(mode, SnSize::Sn12))
    }

    /// The PDU session a `(ue_id, drb_id)` bearer belongs to (issue #44).
    ///
    /// Falls back to the DRB identity when the bearer has not been recorded, which is
    /// the identity mapping the pre-SDAP path had and the right answer for the DEFAULT
    /// DRB in every case (`allocate_drbs` keeps `default_drb_id == psi.clamp(1, 32)`).
    /// A fallback rather than a drop because losing an uplink SDU to a bookkeeping gap
    /// would be worse than sending it to the session it almost certainly belongs to.
    fn psi_for_drb(&self, ue_id: i32, drb_id: i32) -> i32 {
        self.drb_to_psi
            .get(&(ue_id, drb_id))
            .copied()
            .unwrap_or(drb_id)
    }

    /// The MAC grant, in octets, that this UE may be handed in one transmission.
    ///
    /// [`MAC_GRANT_BYTES`] for an unrestricted UE, and the recorded ceiling for a UE
    /// whose declared capabilities bound it below that — today only a RedCap UE, whose
    /// maximum bandwidth is 20 MHz in FR1 (TS 38.306 §4.2.21.1, vendored at
    /// `6g_docs/specs/38306-j30.txt:29939-29942`: *"The maximum bandwidth is 20 MHz for
    /// FR1 [...] UE features and corresponding capabilities related to UE bandwidths
    /// wider than 20 MHz in FR1 [...] are not supported by RedCap UEs"*).
    ///
    /// # What is enforced here, and what is not
    ///
    /// This clamps the **transport-block-sized grant** handed to
    /// [`RlcEntity::build_pdu`], which is the only per-UE resource quantity this
    /// simulator actually allocates. It is NOT a PRB-level allocation on a resource
    /// grid: there is no resource grid, and issue #164 recorded the decision not to
    /// invent one. The grant is proportional to bandwidth (a narrower carrier carries
    /// fewer bits per transmission), so scaling it by the same ratio as the PRB ceiling
    /// is the faithful reduction available at this layer — and unlike the stored-and-
    /// unread ceiling #57 was filed about, a UE's traffic demonstrably changes shape
    /// when it is applied: an SDU over the ceiling is segmented into more, smaller RLC
    /// PDUs. UM and AM both segment (TS 38.322 §5.2.2), so a smaller grant reduces
    /// per-transmission throughput without stalling the bearer.
    ///
    /// Called at every `build_pdu` site rather than baked into the entity at
    /// construction, because the RedCap declaration arrives in UE capability transfer —
    /// after `RRCSetup`, and so potentially after a bearer already exists.
    fn mac_grant_for(&self, ue_id: i32) -> usize {
        Self::grant_from(&self.ue_grant_ceiling, ue_id)
    }

    /// [`Self::mac_grant_for`] over a borrowed ceiling map rather than `&self`.
    ///
    /// Separate so [`Self::poll_rlc_timers`] can consult the ceilings while it holds
    /// `rlc_entities` mutably: a `&self` method would borrow the whole task and
    /// conflict, whereas two disjoint field borrows do not. The alternative — copying
    /// the map every tick — would allocate on a timer path to avoid a borrow that is
    /// provably fine.
    fn grant_from(ceilings: &HashMap<i32, usize>, ue_id: i32) -> usize {
        ceilings
            .get(&ue_id)
            .copied()
            .map_or(MAC_GRANT_BYTES, |ceiling| ceiling.min(MAC_GRANT_BYTES))
    }

    /// Records (or lifts) the MAC grant ceiling for one UE, from
    /// [`RlsMessage::SetUeGrantCeiling`].
    ///
    /// A ceiling of zero is refused rather than stored: `build_pdu(0)` yields no PDU at
    /// all, so it would silence the bearer entirely instead of narrowing it — a UE with
    /// a very small bandwidth still has *some* capacity (the same reason
    /// `redcap_prb_ceiling` floors to 1 rather than 0).
    fn set_ue_grant_ceiling(&mut self, ue_id: i32, grant_octets: Option<usize>) {
        match grant_octets {
            Some(0) => {
                warn!(
                    "Refusing a zero MAC grant ceiling for UE[{ue_id}]: it would stop the \
                     bearer rather than narrow it; leaving the previous ceiling in force"
                );
            }
            Some(octets) => {
                info!(
                    "UE[{ue_id}]: MAC grant ceiling set to {octets} octets (was {}) -- \
                     enforced on every RLC grant (TS 38.306 §4.2.21.1)",
                    self.mac_grant_for(ue_id)
                );
                self.ue_grant_ceiling.insert(ue_id, octets);
            }
            None => {
                if self.ue_grant_ceiling.remove(&ue_id).is_some() {
                    info!(
                        "UE[{ue_id}]: MAC grant ceiling lifted, back to the cell budget of \
                         {MAC_GRANT_BYTES} octets"
                    );
                }
            }
        }
    }

    /// Drives the RLC timers on every entity and transmits whatever they
    /// produce: a UM `t-Reassembly` expiry discards a stranded SDU
    /// (TS 38.322 §5.2.2.2.4), an AM `t-Reassembly` expiry triggers a STATUS
    /// report (§5.2.3.2.4), and an AM `t-PollRetransmit` expiry re-offers an
    /// unacknowledged PDU (§5.3.3.4).
    async fn poll_rlc_timers(&mut self) {
        let now = Instant::now();
        // `(ue_id, drb_id, pdu)`: the entity map is keyed by DRB identity since
        // issue #44, and a re-offered PDU has to go back out on the bearer it came
        // from -- a PSI here would put a retransmission on the wrong DRB when a
        // session has two.
        let mut outbound: Vec<(i32, i32, Vec<u8>)> = Vec::new();
        // Disjoint borrow: the loop holds `rlc_entities` mutably, so the ceilings are
        // read through a separate field reference rather than `self.mac_grant_for`.
        let ceilings = &self.ue_grant_ceiling;
        for ((ue_id, drb_id), rlc) in &mut self.rlc_entities {
            if rlc.poll_timers(now) {
                debug!(
                    "RLC timer expired: ue_id={}, drb_id={}, rx_next_reassembly={}",
                    ue_id,
                    drb_id,
                    rlc.rx_next_reassembly()
                );
            }
            if let Some(status) = rlc.build_status_pdu() {
                outbound.push((*ue_id, *drb_id, status));
            }
            // A retransmission is re-segmented against the grant in force NOW, so a UE
            // that declared RedCap after the original transmission gets the narrower
            // grant on the re-offer too (TS 38.306 §4.2.21.1).
            while let Some(retx) = rlc.build_pdu(Self::grant_from(ceilings, *ue_id)) {
                outbound.push((*ue_id, *drb_id, retx));
            }
        }
        for (ue_id, drb_id, pdu) in outbound {
            self.send_rlc_pdu(ue_id, drb_id, pdu).await;
        }
    }

    /// Drive every PDCP entity's `t-Reordering` and deliver whatever expires
    /// (TS 38.323 §5.2.2.2, issue #33).
    ///
    /// Needed because the timer is otherwise only evaluated when a PDU arrives, so
    /// a gap at the END of a flow would hold the SDUs behind it indefinitely -- the
    /// very case reordering exists to bound. Ticked from the same timer that drives
    /// the RLC timers.
    #[cfg(feature = "drb-pdcp")]
    async fn poll_pdcp_timers(&mut self) {
        let now_ms = self.pdcp_now_ms();
        let mut delivered: Vec<(i32, i32, Vec<u8>)> = Vec::new();
        for ((ue_id, drb_id), pdcp) in &mut self.pdcp_entities {
            for sdu in pdcp.poll_t_reordering(now_ms) {
                delivered.push((*ue_id, *drb_id, sdu));
            }
        }
        for (ue_id, drb_id, sdu) in delivered {
            debug!(
                "PDCP t-Reordering released an SDU: ue_id={ue_id}, drb_id={drb_id}, len={}",
                sdu.len()
            );
            // This is the SAME uplink SDU `handle_uplink_data` would have forwarded had
            // t-Reordering not held it, so it owes the same SDAP strip (issue #44).
            // Missing it here is the subtle half: only SDUs that arrived out of order
            // take this path, so the header octet would leak into the UPF for exactly
            // the packets a reordering bug is hardest to attribute to.
            #[cfg(feature = "sdap-dataplane")]
            let sdu = match nextgsim_pdcp::sdap::decode_ul_pdu(&sdu) {
                Ok((_, payload)) => payload.to_vec(),
                Err(e) => {
                    warn!("Dropped a reordered uplink SDU with an undecodable SDAP header: {e}");
                    continue;
                }
            };
            // GTP-U is per PDU session, so the DRB the SDU was reordered on has to be
            // resolved back to its PSI: the tunnel is named by the session, and a DRB
            // identity here would miss the tunnel entirely once the two differ.
            let msg = GtpMessage::DataPduDelivery {
                ue_id,
                psi: self.psi_for_drb(ue_id, drb_id),
                pdu: OctetString::from_slice(&sdu),
            };
            if let Err(e) = self.task_base.gtp_tx.send(msg).await {
                error!("Failed to send a reordered uplink SDU to GTP: {e}");
            }
        }
    }

    /// Sends one RLC PDU (data or STATUS) to a UE over RLS.
    ///
    /// `drb_id` goes on the wire in `payload`, which is what the UE demultiplexes its
    /// own per-bearer entities on — so this is the field that has to name the DRB and
    /// not the session, or the two ends key their entity maps differently.
    async fn send_rlc_pdu(&mut self, ue_id: i32, drb_id: i32, pdu: Vec<u8>) {
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
            payload: drb_id as u32,
            pdu: Bytes::from(pdu),
        };
        self.send_rls_message(dest, &RlsProtocolMessage::PduTransmission(transmission))
            .await;
    }

    /// Sends the STATUS report an AM bearer owes its peer, if one is due
    /// (TS 38.322 §5.3.4). A no-op for a UM bearer, which has no STATUS PDU.
    async fn send_pending_status(&mut self, ue_id: i32, drb_id: i32) {
        let status = self
            .rlc_entities
            .get_mut(&(ue_id, drb_id))
            .and_then(RlcEntity::build_status_pdu);
        if let Some(status) = status {
            self.send_rlc_pdu(ue_id, drb_id, status).await;
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
    /// fed into the per-bearer RLC entity for reassembly; only complete SDUs are
    /// forwarded to the GTP task.
    async fn handle_uplink_data(&mut self, ue_id: i32, pdu: &RlsPduTransmission) {
        // `payload` is the DRB identity the UE transmitted on (see `send_rlc_pdu` for
        // the downlink half), so it keys the entity maps. Without `sdap-dataplane`
        // that number IS the PSI, which is what this line meant before issue #44.
        let drb_id = pdu.payload as i32;
        // The PDU session the bearer belongs to, which is what GTP-U needs: the N3
        // tunnel is per session, and the radio bearer is not.
        let psi = self.psi_for_drb(ue_id, drb_id);

        debug!(
            "Uplink data (RLC): ue_id={}, drb_id={}, psi={}, len={}",
            ue_id,
            drb_id,
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
            let rlc = self.rlc_entity_for(ue_id, drb_id, psi);
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
        self.send_pending_status(ue_id, drb_id).await;

        // PDCP receive (issue #33): what RLC reassembled is a PDCP PDU, so it goes
        // through the reordering entity and only in-order SDUs reach GTP-U. Without
        // the feature the reassembled bytes go straight up, as before.
        #[cfg(feature = "drb-pdcp")]
        let to_gtp: Vec<Vec<u8>> = {
            let now_ms = self.pdcp_now_ms();
            let pdcp = self.pdcp_entity_for(ue_id, drb_id);
            let mut delivered = Vec::new();
            for sdu in reassembled_sdus {
                match pdcp.receive_pdu(&sdu, now_ms) {
                    Ok(sdus) => delivered.extend(sdus),
                    Err(e) => {
                        debug!("PDCP discarded an uplink PDU on ue {ue_id} DRB {drb_id}: {e:?}")
                    }
                }
            }
            delivered.extend(pdcp.poll_t_reordering(now_ms));
            delivered
        };
        #[cfg(not(feature = "drb-pdcp"))]
        let to_gtp: Vec<Vec<u8>> = reassembled_sdus;

        for sdu in to_gtp {
            // SDAP receive (TS 37.324 §5.2.2, issue #44): with the feature the UE
            // prepends a one-octet UL header, and it MUST be stripped here. GTP-U
            // encapsulates what it is handed verbatim, so leaving the octet on would
            // send the UPF an IP packet whose first byte is an SDAP header -- a
            // corrupt version/IHL nibble, which is a silent black hole rather than a
            // failure. The UL QFI is decoded and logged but not used to select the
            // tunnel: the session already determines that, and TS 38.415 has the gNB
            // put the QFI in the uplink PDU Session Container, which `encapsulate_uplink`
            // builds from the session's own QFI.
            #[cfg(feature = "sdap-dataplane")]
            let sdu = match nextgsim_pdcp::sdap::decode_ul_pdu(&sdu) {
                Ok((qfi, payload)) => {
                    debug!("SDAP UL: ue_id={ue_id}, drb_id={drb_id}, qfi={qfi}");
                    payload.to_vec()
                }
                Err(e) => {
                    // A malformed header means the peer is not speaking this sublayer
                    // (a UE built without the feature) or sent a Control PDU, which
                    // carries no user data at all. Dropped rather than forwarded,
                    // because the alternative is handing the UPF the header octet.
                    warn!("Dropped an uplink SDU with an undecodable SDAP header: {e}");
                    continue;
                }
            };

            debug!(
                "DRB SDU for GTP: ue_id={}, psi={}, len={}",
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
    /// The SDU from GTP is submitted to the PDCP and RLC entities of the bearer
    /// `drb_id` names (UM, SN12). One or more RLC PDUs are then built and sent over
    /// RLS to the UE. The MAC grant size comes from [`Self::mac_grant_for`]: a
    /// 1500-byte MTU for an unrestricted UE, so typical IP packets fit in a single
    /// PDU, and the narrower capability-derived ceiling for a RedCap UE.
    ///
    /// Takes BOTH `psi` and `drb_id` because they answer different questions and are
    /// only the same number without `sdap-dataplane`: `drb_id` selects the entities and
    /// goes on the wire, while `psi` is what the RLC mode is configured per
    /// (`rlc_am_psis`) and what an uplink SDU on this bearer will need to reach its
    /// GTP-U tunnel. Dropping `psi` would silently move every AM bearer to UM.
    async fn handle_downlink_data(&mut self, ue_id: i32, psi: i32, drb_id: i32, data: OctetString) {
        let dest = match self.ue_addresses.get(&ue_id) {
            Some(&addr) => addr,
            None => {
                warn!("Downlink data for unknown UE[{}]", ue_id);
                return;
            }
        };

        debug!(
            "Downlink data (RLC): ue_id={}, psi={}, drb_id={}, len={}",
            ue_id,
            psi,
            drb_id,
            data.len()
        );

        // Submit SDU to RLC and collect all resulting PDUs before releasing
        // the mutable borrow so that self.sti and self.socket are accessible
        // again for transmission.
        let rlc_pdus = {
            // PDCP first (issue #33): the GTP-delivered SDU gets a PDCP header, an
            // SN and a discardTimer, and it is the PDCP PDU that RLC segments. With
            // `sdap-dataplane` that SDU already carries its SDAP header, which GTP
            // prepended -- SDAP is above PDCP (TS 37.324 §4.2), so the header is
            // inside what PDCP protects and what RLC segments.
            #[cfg(feature = "drb-pdcp")]
            let to_rlc: Vec<Vec<u8>> = {
                let now_ms = self.pdcp_now_ms();
                let pdcp = self.pdcp_entity_for(ue_id, drb_id);
                pdcp.submit_sdu(data.data(), now_ms);
                pdcp.take_transmittable(now_ms)
            };
            #[cfg(not(feature = "drb-pdcp"))]
            let to_rlc: Vec<Vec<u8>> = vec![data.data().to_vec()];

            // Resolved before the entity is borrowed mutably, and per UE rather than
            // from the constant: this is the point at which issue #57's ceiling becomes
            // enforcement instead of a stored number.
            let grant = self.mac_grant_for(ue_id);
            let rlc = self.rlc_entity_for(ue_id, drb_id, psi);
            for sdu in to_rlc {
                rlc.submit_sdu(sdu);
            }
            let mut pdus = Vec::new();
            while let Some(rlc_pdu) = rlc.build_pdu(grant) {
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
                // The DRB identity, matching `send_rlc_pdu`: this is what the UE keys
                // its own entity maps on, so the two ends have to name the same thing.
                payload: drb_id as u32,
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
                // A capability-derived ceiling belongs to the UE, not to the id. Left
                // behind, it would narrow the grants of whichever UE is allocated this
                // id next -- which may not be RedCap at all.
                self.ue_grant_ceiling.remove(&ue_id_i32);

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
                                RlsMessage::DownlinkData { ue_id, psi, drb_id, pdu } => {
                                    self.handle_downlink_data(ue_id, psi, drb_id, pdu).await;
                                }
                                #[cfg(feature = "up-security")]
                                RlsMessage::InstallDrbSecurity { ue_id, psi, drb_id, security } => {
                                    // The PSI is recorded alongside the entity so an
                                    // uplink SDU on this bearer can find its GTP-U
                                    // tunnel even when the keys arrive before any
                                    // traffic does -- which is the normal order.
                                    self.drb_to_psi.insert((ue_id, drb_id), psi);
                                    self.install_drb_security(ue_id, drb_id, security.map(|b| *b));
                                }
                                RlsMessage::SetUeGrantCeiling { ue_id, grant_octets } => {
                                    self.set_ue_grant_ceiling(ue_id, grant_octets);
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
                    #[cfg(feature = "drb-pdcp")]
                    self.poll_pdcp_timers().await;
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

    /// Collects every datagram the task has already sent, returning each PDU's length
    /// on the wire.
    ///
    /// Drains rather than waiting for a fixed count, because the number of PDUs is
    /// exactly what the grant ceiling changes — a helper that asked for *n* PDUs would
    /// have to know the answer the test is measuring. The short timeout ends the drain
    /// once the socket is quiet; `handle_downlink_data` has already sent everything it
    /// is going to before it returns, so nothing is still in flight.
    async fn drain_pdu_payload_lengths(socket: &UdpSocket) -> Vec<usize> {
        let mut lengths = Vec::new();
        let mut buf = vec![0u8; UDP_BUFFER_SIZE];
        while let Ok(Ok(len)) =
            tokio::time::timeout(Duration::from_millis(150), socket.recv(&mut buf)).await
        {
            let datagram = Bytes::copy_from_slice(&buf[..len]);
            if let Ok(RlsProtocolMessage::PduTransmission(pdu)) = codec::decode(&datagram) {
                lengths.push(pdu.pdu.len());
            }
        }
        lengths
    }

    /// A RedCap UE's grants are clamped to its 20 MHz FR1 ceiling; a normal UE in the
    /// same cell keeps the full budget (issue #57).
    ///
    /// # Why this is the test the issue asks for
    ///
    /// #57's live defect was a ceiling that was computed, stored and never read: the
    /// getter exposing it had zero callers, so a RedCap UE was granted the whole cell
    /// budget anyway. A test that only asserted the ceiling was *stored* would have
    /// passed against that defect. So this drives the real downlink path
    /// ([`RlsTask::handle_downlink_data`], the same function the `DownlinkData` arm of
    /// the task loop calls) and measures the PDUs that actually leave the socket.
    ///
    /// # Why it is a contrast pair, and asserts equality
    ///
    /// Both UEs are given the SAME SDU on the SAME kind of bearer, differing only in
    /// whether a ceiling was installed. The RedCap UE's largest PDU must equal its
    /// ceiling and the normal UE's must equal the cell grant — positive assertions on
    /// both sides. "Not more than 1500" would be satisfied by a path that sent nothing
    /// at all, and by the pre-#57 code for the normal UE; "fewer PDUs" alone would not
    /// show *which* UE was restricted. The pair is what distinguishes "the clamp is
    /// applied to RedCap UEs" from "the grant got smaller for everyone".
    ///
    /// # Revert-verification
    ///
    /// Verified, not asserted: restoring `handle_downlink_data`'s `build_pdu(grant)` to
    /// `build_pdu(MAC_GRANT_BYTES)` fails this test on the RedCap assertion with
    /// `left: 1500, right: 300` — the UE takes the whole cell grant in 3 PDUs instead of
    /// its ceiling — while the normal-UE assertion still passes. That asymmetry is what
    /// shows the clamp, and not merely the plumbing, is what is being measured.
    #[tokio::test]
    async fn a_redcap_ues_grants_are_clamped_while_a_normal_ues_are_not() {
        let redcap_ue = UdpSocket::bind("127.0.0.1:0").await.expect("RedCap socket");
        let normal_ue = UdpSocket::bind("127.0.0.1:0").await.expect("normal socket");
        let mut task = broadcasting_task().await;

        const REDCAP_ID: i32 = 1;
        const NORMAL_ID: i32 = 2;
        task.ue_addresses
            .insert(REDCAP_ID, redcap_ue.local_addr().unwrap());
        task.ue_addresses
            .insert(NORMAL_ID, normal_ue.local_addr().unwrap());

        // A ceiling well under the cell grant, so the clamp is unambiguous in the PDU
        // lengths. Installed through the same entry point the RRC task uses, so this
        // test cannot pass against a ceiling the message handler ignores.
        const CEILING: usize = 300;
        task.set_ue_grant_ceiling(REDCAP_ID, Some(CEILING));
        assert_eq!(
            task.mac_grant_for(REDCAP_ID),
            CEILING,
            "precondition: the ceiling must be in force for the RedCap UE"
        );
        assert_eq!(
            task.mac_grant_for(NORMAL_ID),
            MAC_GRANT_BYTES,
            "precondition: the normal UE must be unrestricted, or the contrast is void"
        );

        // Big enough that BOTH UEs segment, so neither side's result is "it fitted in
        // one PDU" -- which would make the two indistinguishable.
        let payload = OctetString::from_slice(&[0x7Eu8; 4000]);

        task.handle_downlink_data(REDCAP_ID, 1, 1, payload.clone())
            .await;
        let redcap_pdus = drain_pdu_payload_lengths(&redcap_ue).await;

        task.handle_downlink_data(NORMAL_ID, 1, 1, payload).await;
        let normal_pdus = drain_pdu_payload_lengths(&normal_ue).await;

        let redcap_largest = *redcap_pdus.iter().max().expect("RedCap UE got PDUs");
        let normal_largest = *normal_pdus.iter().max().expect("normal UE got PDUs");

        // Equality, not an upper bound: the RLC fills each grant before it segments, so
        // the largest PDU of a segmented SDU IS the grant. An inequality here would also
        // hold if the path sent one tiny PDU and dropped the rest.
        assert_eq!(
            redcap_largest,
            CEILING,
            "the RedCap UE's largest PDU must be exactly its {CEILING}-octet ceiling \
             (TS 38.306 §4.2.21.1); got {redcap_largest} from {} PDU(s)",
            redcap_pdus.len()
        );
        assert_eq!(
            normal_largest, MAC_GRANT_BYTES,
            "the normal UE must still get the full {MAC_GRANT_BYTES}-octet cell grant -- \
             a clamp that narrowed every UE would be a regression, not a RedCap \
             restriction; got {normal_largest}"
        );
        assert!(
            redcap_pdus.len() > normal_pdus.len(),
            "the same SDU under a narrower grant must take MORE PDUs: RedCap {} vs \
             normal {}",
            redcap_pdus.len(),
            normal_pdus.len()
        );
    }

    /// The ceiling belongs to the UE, not to the id it was allocated: a lost UE's
    /// ceiling must not narrow the grants of whoever gets that id next.
    ///
    /// Asserted positively on both sides of the lift -- clamped first, then back to the
    /// full cell grant -- because `mac_grant_for` returning the cell grant is also what
    /// it does when nothing was ever installed, so only the transition distinguishes a
    /// lift from a ceiling that never took effect.
    #[test]
    fn lifting_a_grant_ceiling_restores_the_full_cell_grant() {
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(test_config(), 16);
        let mut task = RlsTask::new(task_base);

        task.set_ue_grant_ceiling(7, Some(300));
        assert_eq!(task.mac_grant_for(7), 300, "the ceiling is in force");

        task.set_ue_grant_ceiling(7, None);
        assert_eq!(
            task.mac_grant_for(7),
            MAC_GRANT_BYTES,
            "lifting the ceiling returns the UE to the cell grant"
        );
    }

    /// A zero ceiling is refused, because `build_pdu(0)` yields nothing at all: it would
    /// silence the bearer rather than narrow it. The previously installed ceiling stays
    /// in force, which is the safe answer -- a UE with a tiny carrier still has capacity.
    #[test]
    fn a_zero_grant_ceiling_is_refused_rather_than_silencing_the_bearer() {
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(test_config(), 16);
        let mut task = RlsTask::new(task_base);

        task.set_ue_grant_ceiling(7, Some(300));
        task.set_ue_grant_ceiling(7, Some(0));
        assert_eq!(
            task.mac_grant_for(7),
            300,
            "a zero ceiling must be refused and the previous one kept, not stored"
        );
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

        // `drb_id == psi`: each session's default DRB keeps the identity the PSI
        // produced, so this is the same pair of bearers with the feature on or off.
        task.handle_downlink_data(1, 1, 1, OctetString::from_slice(&[0x11; 16]))
            .await;
        task.handle_downlink_data(1, 5, 5, OctetString::from_slice(&[0x55; 16]))
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

    /// The same property one level down, for the split issue #44 introduced: the TWO
    /// DRBs of ONE PDU session must get independent sequence-number spaces, because
    /// TS 38.322 §4.2.1 gives every radio bearer its own RLC entity and the session is
    /// no longer the bearer.
    ///
    /// # Why this is a unit test and not an end-to-end one
    ///
    /// The corruption a shared entity causes needs both streams to be IN FLIGHT at
    /// once: two segment streams interleaved into one SN counter, which the peer
    /// demultiplexes into two entities each expecting a contiguous sequence. The
    /// end-to-end harness cannot produce that, because
    /// [`RlsTask::handle_downlink_data`] submits one SDU and then drains
    /// `build_pdu` to exhaustion before it returns — so two sequential
    /// `DownlinkData` messages segment, transmit and reassemble one at a time and
    /// the two streams never coexist. `tests/src/sdap_user_plane.rs`'s
    /// `two_drbs_of_one_session_carry_their_own_traffic` therefore covers addressing
    /// and routing, and this test covers the numbering.
    ///
    /// Here the SDUs go to both entities BEFORE either is drained, and the drain then
    /// takes one PDU per bearer per round the way a MAC scheduler would interleave
    /// them. Keyed on the PSI both submissions land in one entity, so the second SDU
    /// queues behind the first and goes out as SN 1 while the PDUs attributed to the
    /// two bearers are whichever ones happened to be pulled on their turn — which is
    /// what "reassembles into neither payload" means at the peer.
    ///
    /// Both payloads are segmented (over the 1500-byte grant) and of DIFFERENT
    /// lengths, so a wholesale swap between the bearers cannot look correct either.
    #[cfg(feature = "sdap-dataplane")]
    #[test]
    fn two_drbs_of_one_session_get_independent_sequence_number_spaces() {
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(test_config(), 16);
        let mut task = RlsTask::new(task_base);

        // The identities the live path uses, taken from the allocator rather than
        // written out, so a change to the numbering moves this test with it.
        let psi = 1;
        let alloc = nextgsim_gtp::qfi_drb::allocate_drbs(psi as u8);
        let (default_drb, gbr_drb) = (i32::from(alloc.default_drb_id), i32::from(alloc.gbr_drb_id));
        assert_ne!(
            default_drb, gbr_drb,
            "precondition: the session's two DRBs must be distinct, or this test is \
             asserting about one bearer twice"
        );

        // Distinct fills and distinct lengths, both over `MAC_GRANT_BYTES` so each is
        // segmented -- segmentation is what makes an interleave possible at all.
        let on_default = vec![0xA5u8; 2000];
        let on_gbr = vec![0x5Au8; 2400];

        // BOTH bearers loaded before EITHER is drained. This is the state the
        // end-to-end harness cannot reach, and the only state in which sharing an
        // entity corrupts rather than merely renumbers.
        //
        // `psi` is the same for both calls on purpose: it is one PDU session, so the
        // PSI cannot be what separates these two bearers.
        task.rlc_entity_for(1, default_drb, psi)
            .submit_sdu(on_default.clone());
        task.rlc_entity_for(1, gbr_drb, psi)
            .submit_sdu(on_gbr.clone());

        // One PDU per bearer per round, until both bearers are empty -- the
        // interleave a MAC scheduler would produce.
        let mut produced: HashMap<i32, Vec<nextgsim_rlc::RlcUmPdu>> = HashMap::new();
        loop {
            let mut any = false;
            for drb_id in [default_drb, gbr_drb] {
                if let Some(bytes) = task
                    .rlc_entity_for(1, drb_id, psi)
                    .build_pdu(MAC_GRANT_BYTES)
                {
                    any = true;
                    let pdu = nextgsim_rlc::RlcUmPdu::decode_sn12(&bytes)
                        .expect("the entity's own PDU must decode");
                    produced.entry(drb_id).or_default().push(pdu);
                }
            }
            if !any {
                break;
            }
        }

        for (drb_id, expected) in [(default_drb, &on_default), (gbr_drb, &on_gbr)] {
            let pdus = produced
                .get(&drb_id)
                .unwrap_or_else(|| panic!("DRB {drb_id} produced no PDU at all"));
            assert!(
                pdus.len() >= 2,
                "DRB {drb_id}'s payload must be segmented, or the interleave this test \
                 is about never happens -- got {} PDU(s)",
                pdus.len()
            );
            // The first SDU of a bearer is SN 0, and all segments of one SDU share
            // that SN (TS 38.322 §5.2.2.1), so EVERY PDU here is SN 0. A bearer that
            // shared the other's entity would carry the other bearer's SN instead.
            let sns: Vec<u16> = pdus.iter().map(|p| p.sn).collect();
            assert!(
                sns.iter().all(|&sn| sn == 0),
                "DRB {drb_id} must number in its OWN sequence-number space, so its \
                 first SDU's segments are all SN 0 -- got {sns:?}, which means the two \
                 DRBs of PSI {psi} are sharing one counter"
            );
            // And the segments attributed to this bearer really are this bearer's SDU:
            // a shared entity hands out the other flow's segments on this bearer's
            // turn, and the peer then reassembles bytes belonging to neither payload.
            let reassembled: Vec<u8> = pdus.iter().flat_map(|p| p.data.clone()).collect();
            assert_eq!(
                &reassembled, expected,
                "DRB {drb_id}'s segments must reassemble into ITS payload"
            );
        }

        assert_eq!(
            task.rlc_entities.len(),
            2,
            "one RLC entity per (UE, DRB), so one PDU session's two DRBs are two \
             entities -- a PSI key would make this 1"
        );
    }

    /// The `t-Reassembly` tick is wired into the task's run loop, not just
    /// implemented on the entity: a partial SDU abandoned by the timer must NOT
    /// be resurrected by a segment that arrives late.
    ///
    /// Drives the real run loop over loopback UDP with a test socket standing in
    /// for the UE's radio, because the tick lives in the loop's periodic arm and
    /// nothing else would exercise it.
    /// The QFI the simulated UE puts in its uplink SDAP headers.
    ///
    /// Any value in 0..=63 does: the gNB logs the uplink QFI but does not route on it
    /// (the PDU session already names the tunnel), so no assertion in this module
    /// depends on which one. 1 matches the QFI the GTP task's auto-created sessions
    /// use, so a reader is not led to think the number is arbitrary in the gNB too.
    #[cfg(feature = "sdap-dataplane")]
    const TEST_QFI: u8 = 1;

    /// What a conformant peer puts in an RLC SDU on a DRB.
    ///
    /// With the `drb-pdcp` feature the DRB carries PDCP PDUs, so a test that
    /// simulates a peer has to send one -- a bare payload would have its first two
    /// octets read as a PDCP header. Without the feature it is the payload itself,
    /// which is what the peer sent before issue #33.
    ///
    /// With `sdap-dataplane` it additionally carries a one-octet UL SDAP header
    /// (TS 37.324 §6.2.2.3), because that is what a conformant UE transmits and the
    /// gNB's uplink path strips one. A helper that skipped it would have the gNB read
    /// the payload's first octet as the header — which is not even a clean failure:
    /// a first byte with the D/C bit set decodes as a plausible header and the SDU
    /// arrives one octet short, while one with it clear is discarded as a Control PDU.
    ///
    /// `sn` is the PDCP SN, and it matters: PDCP delivers in order, so two SDUs
    /// sent with the same SN would be a duplicate and the second discarded.
    // `sn` is read only by the PDCP branch, so it is unused whenever that feature is
    // off -- regardless of `sdap-dataplane`, which does not carry a sequence number.
    #[cfg_attr(not(feature = "drb-pdcp"), allow(unused_variables))]
    fn drb_sdu(payload: Vec<u8>, sn: u32) -> Vec<u8> {
        // SDAP first, and PDCP second, because SDAP is ABOVE PDCP (TS 37.324 §4.2):
        // the header has to end up INSIDE what PDCP protects, which is the order the
        // gNB's own downlink path uses in `handle_downlink_data`.
        #[cfg(feature = "sdap-dataplane")]
        let payload = nextgsim_pdcp::sdap::build_ul_pdu(TEST_QFI, &payload)
            .expect("a test QFI must fit the 6-bit SDAP field");
        #[cfg(feature = "drb-pdcp")]
        {
            let mut pdcp = nextgsim_pdcp::Pdcp::new(nextgsim_pdcp::PdcpConfig::default());
            // Advance TX_NEXT to the requested SN so the PDU carries it.
            for _ in 0..sn {
                pdcp.submit_sdu(&[], 0);
            }
            pdcp.submit_sdu(&payload, 0);
            pdcp.take_transmittable(0)
                .pop()
                .expect("one PDCP PDU per SDU")
        }
        #[cfg(not(feature = "drb-pdcp"))]
        {
            payload
        }
    }

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
                drb_sdu(vec![0xEE; 4], 0),
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

        // `(ue_id, drb_id, psi)`: the mode is decided by the PSI even though the entity
        // is keyed by the DRB identity, because `rlc_am_psis` is a list of sessions.
        assert_eq!(
            task.rlc_entity_for(1, 5, 5).mode,
            RlcMode::AcknowledgedMode,
            "PSI 5 is configured for AM"
        );
        assert_eq!(
            task.rlc_entity_for(1, 1, 1).mode,
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
        // Each RLC SDU is what a DRB carries: a PDCP PDU with the feature on, the
        // bare payload without it. Their PDCP SNs are 0, 1, 2, so PDCP delivers
        // them in order rather than treating the second as a duplicate.
        let sdus: Vec<Vec<u8>> = (0..3u8)
            .map(|i| drb_sdu(vec![0xC0 + i; 20], u32::from(i)))
            .collect();
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

        // WITHOUT `drb-pdcp` two SDUs arrive now and the lost one follows after the
        // retransmission -- so the upper layer sees 0, 2, 1, which is exactly the
        // out-of-order delivery issue #33 exists to fix.
        //
        // WITH the feature PDCP delivers in COUNT order, so only SDU 0 arrives now
        // and SDUs 1 and 2 both follow once the retransmission fills the gap.
        //
        // Rather than asserting a different count per configuration, the payloads
        // are collected and their ORDER asserted at the end: that is the property
        // that actually differs, and asserting it is the point.
        let mut delivered_payloads: Vec<Vec<u8>> = Vec::new();
        let expected_now = if cfg!(feature = "drb-pdcp") { 1 } else { 2 };
        for _ in 0..expected_now {
            let delivered = tokio::time::timeout(Duration::from_secs(3), gtp_rx.recv())
                .await
                .expect("the received SDUs must reach GTP");
            match delivered {
                Some(TaskMessage::Message(GtpMessage::DataPduDelivery { pdu, .. })) => {
                    delivered_payloads.push(pdu.data().to_vec());
                }
                other => panic!("expected an SDU at GTP, got {other:?}"),
            }
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

        // The retransmission releases the rest: one SDU without the feature, two
        // with it (the recovered one plus the one that was waiting behind it).
        let expected_after = if cfg!(feature = "drb-pdcp") { 2 } else { 1 };
        for _ in 0..expected_after {
            let recovered = tokio::time::timeout(Duration::from_secs(3), gtp_rx.recv())
                .await
                .expect("the retransmitted SDU must reach GTP");
            match recovered {
                Some(TaskMessage::Message(GtpMessage::DataPduDelivery { psi, pdu, .. })) => {
                    assert_eq!(psi, 5);
                    delivered_payloads.push(pdu.data().to_vec());
                }
                other => panic!("expected the recovered SDU at GTP, got {other:?}"),
            }
        }

        // Every SDU arrived intact, in both configurations.
        assert_eq!(delivered_payloads.len(), 3);
        let payload_of = |i: u8| vec![0xC0 + i; 20];
        let mut sorted = delivered_payloads.clone();
        sorted.sort();
        assert_eq!(
            sorted,
            vec![payload_of(0), payload_of(1), payload_of(2)],
            "all three payloads must arrive intact"
        );

        // AND THE ORDER IS THE POINT OF ISSUE #33.
        #[cfg(feature = "drb-pdcp")]
        assert_eq!(
            delivered_payloads,
            vec![payload_of(0), payload_of(1), payload_of(2)],
            "PDCP must deliver in COUNT order even though SN 1 arrived last"
        );
        #[cfg(not(feature = "drb-pdcp"))]
        assert_eq!(
            delivered_payloads,
            vec![payload_of(0), payload_of(2), payload_of(1)],
            "without PDCP the upper layer sees the recovered SDU last -- the \
             out-of-order delivery this feature fixes"
        );
    }

    /// A gap at the END of a flow is released by the periodic PDCP tick, not by
    /// the arrival of another PDU (issue #33).
    ///
    /// The case that needs a tick at all: PDCP's `t-Reordering` is otherwise only
    /// evaluated inside `receive_pdu`, so an SDU stuck behind a lost one with no
    /// traffic following it would be held forever -- the very thing the reordering
    /// timer exists to bound.
    #[cfg(feature = "drb-pdcp")]
    #[tokio::test]
    async fn a_trailing_reordering_gap_is_released_by_the_periodic_tick() {
        let ue = UdpSocket::bind("127.0.0.1:0").await.expect("UE socket");
        let gnb_addr = {
            let probe = UdpSocket::bind("127.0.0.1:0").await.expect("probe");
            probe.local_addr().unwrap()
        };

        let (task_base, _app_rx, _ngap_rx, _rrc_rx, mut gtp_rx, rls_rx, _sctp_rx) =
            GnbTaskBase::new(test_config(), 32);
        let mut task = RlsTask::with_bind_address(task_base, gnb_addr);
        tokio::spawn(async move { task.run(rls_rx).await });

        let heartbeat = codec::encode(&RlsProtocolMessage::Heartbeat(nextgsim_rls::RlsHeartbeat {
            sti: 0x0C0F_FEE0,
            sim_pos: SimCoord::new(0, 0, 0),
        }));
        ue.send_to(&heartbeat, gnb_addr).await.expect("heartbeat");

        // Send PDCP SN 1 only, and then nothing at all. SN 0 never arrives, so
        // in-order delivery cannot release SN 1 until the timer gives up.
        let rlc = nextgsim_rlc::RlcUmPdu {
            si: nextgsim_rlc::SegmentationInfo::FullSdu,
            sn: 0,
            so: None,
            data: drb_sdu(vec![0x7F; 16], 1),
        }
        .encode_sn12();
        let framed = codec::encode(&RlsProtocolMessage::PduTransmission(RlsPduTransmission {
            sti: 0x0C0F_FEE0,
            pdu_type: PduType::Data,
            pdu_id: 0,
            payload: 1, // PSI 1
            pdu: Bytes::from(rlc),
        }));
        ue.send_to(&framed, gnb_addr).await.expect("the lone PDU");

        // Nothing may arrive before t-Reordering expires: the SDU is behind a gap.
        assert!(
            tokio::time::timeout(Duration::from_millis(300), gtp_rx.recv())
                .await
                .is_err(),
            "an SDU behind a gap must NOT be delivered before t-Reordering expires"
        );

        // The default t-Reordering is 1 s and the task ticks every 500 ms, so the
        // release lands within about 1.5 s.
        let released = tokio::time::timeout(Duration::from_secs(4), gtp_rx.recv())
            .await
            .expect("the periodic tick must release the SDU t-Reordering gave up on");
        match released {
            Some(TaskMessage::Message(GtpMessage::DataPduDelivery { pdu, .. })) => {
                assert_eq!(pdu.data(), &vec![0x7F; 16][..], "released intact");
            }
            other => panic!("expected the released SDU at GTP, got {other:?}"),
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
