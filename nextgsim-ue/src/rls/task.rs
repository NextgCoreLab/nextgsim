//! RLS Task Implementation for UE
//!
//! This module implements the RLS (Radio Link Simulation) task for the UE,
//! handling cell search, gNB connection, RRC message transport, and user plane data.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use tokio::net::UdpSocket;
use tokio::sync::mpsc;
use tokio::time::interval;
use tracing::{debug, error, info, warn};

use crate::tasks::{NasMessage, RlfCause, RlsMessage, RrcMessage, Task, TaskMessage, UeTaskBase};

/// The level reported to RRC for a cell that stopped answering: below
/// `rrc::cell_selection::CELL_LOST_THRESHOLD_DBM`, which is what makes the cell
/// selector drop the cell.
const CELL_SIGNAL_LOST_DBM: i32 = i32::MIN;
use nextgsim_common::OctetString;
#[cfg(feature = "drb-pdcp")]
use nextgsim_pdcp::{Pdcp, PdcpConfig};
use nextgsim_rlc::{RlcEntity, RlcMode, SnSize};
use nextgsim_rls::{
    codec, CellSearchEvent, RlsMessage as RlsProtocolMessage, RlsTransport, RrcChannel, SimCoord,
    TransportEvent, UeCellSearch,
};

use nextgsim_common::peer_addr::PeerAddr;

/// Default RLS port for gNB
pub const DEFAULT_RLS_PORT: u16 = 4997;

/// Parse one `gnb_search_list` entry.
///
/// Accepts a bare IP or hostname (port defaults to [`DEFAULT_RLS_PORT`]), or
/// `IP:port` / `host:port`. A hostname (issue #286; e.g. a Kubernetes Service
/// name) is not resolved here: the RLS task resolves it, and keeps re-resolving
/// it, so a gNB that restarts at a new address is found again. It used to be
/// rejected, which forced deployments to pin a literal IP once at startup. The
/// error names the offending entry so a misconfiguration is actionable from the
/// message alone.
pub fn parse_gnb_search_entry(entry: &str) -> Result<PeerAddr, String> {
    PeerAddr::parse(entry, DEFAULT_RLS_PORT)
}

/// How often hostnames in `gnb_search_list` are re-resolved (issue #286). A
/// serving-cell loss also re-resolves at once.
pub const GNB_SEARCH_RERESOLVE_INTERVAL: Duration = Duration::from_secs(5);

const HEARTBEAT_INTERVAL_MS: u64 = 1000;
const LOST_CELL_CHECK_INTERVAL_MS: u64 = 500;
const UDP_BUFFER_SIZE: usize = 65535;

/// MAC grant size handed to RLC when building PDUs.
///
/// This simulator has no MAC scheduler; a 1500-byte grant stands in for one so a
/// typical IP packet fits in a single PDU.
const MAC_GRANT_BYTES: usize = 1500;

/// RLS task configuration
#[derive(Debug, Clone)]
pub struct RlsTaskConfig {
    /// Literal gNB addresses.
    pub gnb_search_list: Vec<SocketAddr>,
    /// gNBs named by hostname, resolved and re-resolved by the task (#286).
    pub gnb_search_hosts: Vec<PeerAddr>,
    /// How often `gnb_search_hosts` are re-resolved.
    pub reresolve_interval: Duration,
    pub bind_address: Option<SocketAddr>,
    pub heartbeat_interval: Duration,
    pub heartbeat_threshold: Duration,
}

impl Default for RlsTaskConfig {
    fn default() -> Self {
        Self {
            gnb_search_list: Vec::new(),
            gnb_search_hosts: Vec::new(),
            reresolve_interval: GNB_SEARCH_RERESOLVE_INTERVAL,
            bind_address: None,
            heartbeat_interval: Duration::from_millis(HEARTBEAT_INTERVAL_MS),
            heartbeat_threshold: Duration::from_millis(2000),
        }
    }
}

/// RLS Task for managing radio link simulation on UE side
pub struct RlsTask {
    task_base: UeTaskBase,
    cell_search: UeCellSearch,
    transport: RlsTransport,
    cell_addresses: HashMap<i32, SocketAddr>,
    serving_cell: Option<i32>,
    socket: Option<Arc<UdpSocket>>,
    config: RlsTaskConfig,
    /// The last successful resolution of each `gnb_search_hosts` entry (#286).
    /// A failed lookup keeps the previous answer rather than emptying it.
    resolved_hosts: HashMap<PeerAddr, Vec<SocketAddr>>,
    /// Sources already warned about by `handle_pdu_transmission`.
    unknown_sources_warned: std::collections::HashSet<SocketAddr>,
    sti: u64,
    /// RLC entities keyed by DRB identity.
    /// Each entry is a UM SN12 entity used for user-plane data on that bearer.
    ///
    /// Keyed on the DRB identity and not the PSI since issue #44: with
    /// `sdap-dataplane` one PDU session has two DRBs, and a PSI key would make them
    /// share an entity — interleaving two flows into one sequence-number space while
    /// the gNB demultiplexed them into two entities each expecting a contiguous
    /// sequence, which is silent corruption rather than a clean failure. Without the
    /// feature the DRB identity IS the PSI, so the keys are exactly as they were.
    rlc_entities: HashMap<i32, RlcEntity>,
    /// PDCP entities keyed by DRB identity -- one per DRB (TS 38.323 §5.2,
    /// issue #33). Keyed the same way as the RLC entities, because a PDCP entity and
    /// its RLC entity serve the same bearer.
    #[cfg(feature = "drb-pdcp")]
    pdcp_entities: HashMap<i32, Pdcp>,
    /// The PDU session each DRB belongs to (issue #44).
    ///
    /// Needed because the radio side is per-DRB while NAS is per-session: a received
    /// SDU arrives on a DRB and has to be handed up as
    /// `NasMessage::UplinkDataDelivery { psi, .. }`, so the delivery needs the PSI
    /// back. Recorded when the bearer is first seen rather than recomputed, because
    /// DRB→PSI is not a function this side can evaluate: `allocate_drbs` wraps two
    /// identities into 1..=32, so inverting it would take a search that can match the
    /// wrong session.
    drb_to_psi: HashMap<i32, i32>,
    /// Which DRB each PDU session's uplink leaves on, and the QFI it is stamped with
    /// (issue #44, TS 37.324 §5.1).
    ///
    /// Populated from [`RlsMessage::InstallSdapMapping`], i.e. from the
    /// `mappedQoS-FlowsToAdd` the network signalled — never from a local recomputation
    /// of the gNB's policy, which would be a second copy of it.
    #[cfg(feature = "sdap-dataplane")]
    sdap_uplink: HashMap<i32, SdapUplinkBinding>,
    /// When this task started: the origin for the PDCP timers and for the
    /// UE's `sim_movement` (#324).
    started_at: std::time::Instant,
    /// The NTN uplink pre-compensation in force, derived from SIB19
    /// (TS 38.300 §16.14.2.2, issue #56).
    ///
    /// `None` on a terrestrial cell, and on an NTN cell before the UE has both a
    /// valid ephemeris and a GNSS position — in which case no shift is applied and
    /// the uplink leaves at the nominal instant, exactly as it did before this
    /// issue. Installed by [`RlsMessage::ApplyNtnPrecompensation`] and READ by
    /// [`Self::apply_ntn_uplink_timing`], which every uplink transmission passes
    /// through. That read is what criterion 2 of issue #56 asks for.
    ntn_precompensation: Option<NtnUplinkPrecompensation>,
    /// The uplink transmit timing offset, in microseconds, that the last uplink
    /// transmission actually went out with (issue #56).
    ///
    /// Recorded by [`Self::apply_ntn_uplink_timing`] on the transmit path, so it is
    /// the APPLIED value rather than a restatement of the configuration: if the read
    /// site were removed this would stay `None` and the end-to-end test would fail.
    /// `None` until an uplink has been sent.
    last_uplink_timing_offset_us: Option<f64>,
}

/// The NTN uplink pre-compensation a UE applies to its transmitter
/// (TS 38.300 §16.14.2.2, issue #56).
///
/// Public so an end-to-end test can assert the value that was **applied**, rather
/// than a log line saying it was — see `RlsTask::ntn_precompensation_applied`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NtnUplinkPrecompensation {
    /// `T_TA` in microseconds: how much EARLIER than the nominal instant every
    /// uplink transmission leaves.
    pub ta_us: f64,
    /// The Doppler pre-compensation in Hz applied to the transmitter.
    pub doppler_hz: f64,
    /// `cellSpecificKoffset` in slots (TS 38.300 §16.14.2.1).
    pub k_offset: u16,
}

/// Where one PDU session's uplink goes, and under which QoS flow (issue #44).
///
/// # Why a single DRB and QFI per session, and not a classifier
///
/// An uplink SDU reaches the RLS task carrying only a PSI
/// ([`RlsMessage::UplinkData`] / [`RlsMessage::DataPduDelivery`]): the TUN read and
/// the NAS path above it never inspect the packet. Choosing a QoS flow per packet is
/// what a URSP / TFT matcher does (TS 23.503 §6.6.2) — it matches the 5-tuple, the
/// application id and the DNN against traffic descriptors the network provisioned —
/// and this UE has none of that, so there is nothing to classify *with*.
///
/// So the uplink uses the session's **default QoS flow** on its **default DRB**,
/// which is TS 37.324 §5.3.1's own rule for a flow with no explicit mapping. That is
/// a deliberate limit and not a stand-in for a missing lookup: inventing a
/// classifier here would mean inventing the traffic descriptors too, and the QFI it
/// produced would not be one the network admitted.
#[cfg(feature = "sdap-dataplane")]
#[derive(Debug, Clone, Copy)]
struct SdapUplinkBinding {
    /// The DRB the uplink rides, which keys the entities and goes on the wire.
    drb_id: i32,
    /// The QFI stamped into every uplink SDAP header on this session.
    ///
    /// The lowest QFI the network mapped to the default DRB. Lowest rather than
    /// arbitrary so the choice is deterministic: `mappedQoS-FlowsToAdd` order is not
    /// something the UE should depend on, and two reconfigurations listing the same
    /// flows differently must not move the session's traffic to another flow.
    qfi: u8,
}

impl RlsTask {
    pub fn new(task_base: UeTaskBase, config: RlsTaskConfig) -> Self {
        let sti = rand::random::<u64>();
        let search_space: Vec<SocketAddr> = config
            .gnb_search_list
            .iter()
            .map(|addr| {
                if addr.port() == 0 {
                    SocketAddr::new(addr.ip(), DEFAULT_RLS_PORT)
                } else {
                    *addr
                }
            })
            .collect();

        let mut cell_search = UeCellSearch::new(sti, search_space);
        cell_search.set_heartbeat_interval(config.heartbeat_interval);
        cell_search.set_heartbeat_threshold(config.heartbeat_threshold);

        Self {
            task_base,
            cell_search,
            transport: RlsTransport::new(sti),
            cell_addresses: HashMap::new(),
            serving_cell: None,
            socket: None,
            config,
            resolved_hosts: HashMap::new(),
            unknown_sources_warned: std::collections::HashSet::new(),
            sti,
            rlc_entities: HashMap::new(),
            #[cfg(feature = "drb-pdcp")]
            pdcp_entities: HashMap::new(),
            drb_to_psi: HashMap::new(),
            #[cfg(feature = "sdap-dataplane")]
            sdap_uplink: HashMap::new(),
            started_at: std::time::Instant::now(),
            // No pre-compensation until a SIB19 says otherwise (issue #56). A
            // terrestrial UE never leaves this state, which is what keeps every
            // non-NTN scenario timed exactly as it was.
            ntn_precompensation: None,
            last_uplink_timing_offset_us: None,
        }
    }

    pub fn from_ue_config(task_base: UeTaskBase) -> Self {
        // Entries that do not parse are logged rather than dropped in silence.
        // A bad entry used to vanish here, leaving an empty search space and a
        // UE that never finds a cell with nothing in the log to explain why.
        // `parse_gnb_search_entry` is also called from the UE's startup
        // validation, so a malformed list normally fails before reaching this
        // point; the warning covers callers that build a config directly.
        let mut gnb_search_list = Vec::new();
        let mut gnb_search_hosts = Vec::new();
        for entry in &task_base.config.gnb_search_list {
            match parse_gnb_search_entry(entry) {
                Ok(PeerAddr::Literal(addr)) => gnb_search_list.push(addr),
                Ok(host) => gnb_search_hosts.push(host),
                Err(e) => warn!("ignoring gnb_search_list entry {entry:?}: {e}"),
            }
        }
        if gnb_search_list.is_empty()
            && gnb_search_hosts.is_empty()
            && !task_base.config.gnb_search_list.is_empty()
        {
            error!(
                "no usable gNB addresses: all {} gnb_search_list entries were rejected. \
                 The UE cannot find a cell.",
                task_base.config.gnb_search_list.len()
            );
        }
        let config = RlsTaskConfig {
            gnb_search_list,
            gnb_search_hosts,
            ..Default::default()
        };
        Self::new(task_base, config)
    }

    /// The literal addresses plus the latest resolution of every hostname.
    fn effective_search_space(&self) -> Vec<SocketAddr> {
        let mut space: Vec<SocketAddr> = self
            .config
            .gnb_search_list
            .iter()
            .map(|addr| {
                if addr.port() == 0 {
                    SocketAddr::new(addr.ip(), DEFAULT_RLS_PORT)
                } else {
                    *addr
                }
            })
            .collect();
        for host in &self.config.gnb_search_hosts {
            for addr in self.resolved_hosts.get(host).into_iter().flatten() {
                if !space.contains(addr) {
                    space.push(*addr);
                }
            }
        }
        space
    }

    /// Re-resolve every gNB named by hostname and update the heartbeat search
    /// space if any answer changed (issue #286). A gNB pod that restarts comes
    /// back at a new address; the cell at the old one stops answering and is
    /// lost as usual, and the new one is discovered by the next heartbeat.
    async fn reresolve_search_hosts(&mut self) {
        let mut changed = false;
        for host in self.config.gnb_search_hosts.clone() {
            match host.resolve().await {
                Ok(addrs) => {
                    if self.resolved_hosts.get(&host) != Some(&addrs) {
                        info!("gNB {host} resolves to {addrs:?}");
                        self.resolved_hosts.insert(host, addrs);
                        changed = true;
                    }
                }
                Err(e) => debug!("gNB {host} does not resolve now: {e}"),
            }
        }
        if changed {
            let space = self.effective_search_space();
            self.cell_search.set_search_space(space);
        }
    }

    pub fn cell_count(&self) -> usize {
        self.cell_search.cell_count()
    }
    pub fn serving_cell(&self) -> Option<i32> {
        self.serving_cell
    }

    /// The PDCP entity for one DRB, created on first use (issue #33).
    ///
    /// Only compiled with the `drb-pdcp` feature: interposing a sublayer changes
    /// the live data path, so the default build keeps RLC wired straight to NAS
    /// exactly as before.
    ///
    /// Keyed on `drb_id`, not the PSI: TS 38.323 §5.2 gives every DRB its own entity,
    /// and with `sdap-dataplane` a session has two of them. Sharing one would have the
    /// two bearers share a COUNT, which under `up-security` means two flows ciphered
    /// with the same keystream.
    #[cfg(feature = "drb-pdcp")]
    fn pdcp_entity_for(&mut self, drb_id: i32) -> &mut Pdcp {
        self.pdcp_entities
            .entry(drb_id)
            .or_insert_with(|| Pdcp::new(PdcpConfig::default()))
    }

    /// Install (or remove) user-plane security on one DRB (issue #32).
    ///
    /// Applied to the entity for `drb_id`, creating it if the DRB has not carried a
    /// packet yet — the keys arrive from RRC before the first uplink packet does, and
    /// an entity created later with no security would send the first packets in the
    /// clear.
    ///
    /// `drb_id` and not the PSI since issue #44, and the distinction is load-bearing:
    /// `PdcpSecurity::bearer` is the DRB identity minus one (TS 33.501 Annex D.3.1.2),
    /// so installing a binding on a PSI-keyed entity would put the right BEARER on the
    /// wrong bearer's entity and every MAC-I would fail with no other symptom.
    #[cfg(feature = "up-security")]
    fn install_drb_security(&mut self, drb_id: i32, security: Option<nextgsim_pdcp::PdcpSecurity>) {
        let protected = security.is_some();
        self.pdcp_entity_for(drb_id).set_security(security);
        info!(
            "DRB user-plane security {} for DRB {}",
            if protected { "installed" } else { "removed" },
            drb_id
        );
    }

    /// Record the QoS-flow-to-DRB mapping the network signalled for one DRB
    /// (issue #44, TS 37.324 §5.1).
    ///
    /// Only the **default** DRB is recorded as an uplink binding, because that is the
    /// only one this UE can send on: it has no per-packet classifier to select the
    /// other with (see [`SdapUplinkBinding`]). A non-default DRB is still noted in
    /// [`Self::drb_to_psi`] so its *downlink* SDUs reach NAS under the right PSI —
    /// dropping it there would strand every SDU the gNB sent on the second bearer.
    #[cfg(feature = "sdap-dataplane")]
    fn install_sdap_mapping(&mut self, psi: i32, drb_id: i32, qfis: &[u8], default_drb: bool) {
        // Both directions need this regardless of which bearer it is: a downlink SDU
        // on the GBR DRB has to be delivered under its session's PSI too.
        self.drb_to_psi.insert(drb_id, psi);

        if !default_drb {
            debug!(
                "SDAP: DRB {drb_id} (PSI {psi}) carries QFIs {qfis:?} downlink only -- \
                 uplink uses the session's default DRB"
            );
            return;
        }
        // The lowest mapped QFI is the session's default flow. `None` for a default
        // DRB with no mapped flow at all, which the gNB sends when the core admitted
        // nothing on the session: a QFI would have to be invented, and an invented one
        // would name a flow the network never admitted. Uplink then falls back to the
        // pre-SDAP behaviour for that session rather than guessing (see
        // `handle_data_pdu_delivery`).
        let Some(&qfi) = qfis.iter().min() else {
            warn!(
                "SDAP: PSI {psi}'s default DRB {drb_id} has no mapped QoS flow; uplink on \
                 this session cannot be stamped with a QFI the network admitted"
            );
            return;
        };
        self.sdap_uplink
            .insert(psi, SdapUplinkBinding { drb_id, qfi });
        info!("SDAP: PSI {psi} uplink rides DRB {drb_id} as QFI {qfi} (mapped QFIs {qfis:?})");
    }

    /// The PDU session a DRB belongs to (issue #44).
    ///
    /// Falls back to the DRB identity when the bearer has not been recorded, which is
    /// the identity mapping the pre-SDAP path had and the right answer for the DEFAULT
    /// DRB in every case (`allocate_drbs` keeps `default_drb_id == psi.clamp(1, 32)`).
    /// A fallback rather than a drop because losing a received SDU to a bookkeeping gap
    /// would be worse than delivering it to the session it almost certainly belongs to.
    fn psi_for_drb(&self, drb_id: i32) -> i32 {
        self.drb_to_psi.get(&drb_id).copied().unwrap_or(drb_id)
    }

    /// Milliseconds since the task started, for the PDCP timers.
    ///
    /// A monotonic elapsed time rather than a wall clock: the PDCP timers measure
    /// durations, and a wall-clock step would move them.
    #[cfg(feature = "drb-pdcp")]
    fn pdcp_now_ms(&self) -> u64 {
        self.started_at.elapsed().as_millis() as u64
    }

    /// Returns the RLC entity for one radio bearer, creating a UM SN12 entity on
    /// first use.
    ///
    /// One entity per bearer (TS 38.322 §4.2.1), so each DRB has its own
    /// sequence-number space and reassembly buffer. Keyed on `drb_id` since issue #44:
    /// keying on the PSI interleaved a session's two DRBs into one SN counter while
    /// the gNB demultiplexed them into per-bearer entities each expecting a contiguous
    /// sequence, so a second bearer corrupted both rather than failing cleanly.
    ///
    /// `psi` comes along even though it is not the key, for two reasons. The RLC
    /// **mode** is configured per PDU session (`rlc_am_psis` is a list of PSIs), so the
    /// AM decision can only be made from the PSI; and a received SDU has to be handed
    /// to NAS under its session, so [`Self::drb_to_psi`] is recorded here — the one
    /// place every bearer passes through.
    fn rlc_entity_for(&mut self, drb_id: i32, psi: i32) -> &mut RlcEntity {
        let mode = if self.task_base.config.rlc_am_psis.contains(&(psi as u8)) {
            RlcMode::AcknowledgedMode
        } else {
            RlcMode::UnacknowledgedMode
        };
        self.drb_to_psi.insert(drb_id, psi);
        self.rlc_entities
            .entry(drb_id)
            .or_insert_with(|| RlcEntity::new(mode, SnSize::Sn12))
    }

    /// Drive every PDCP entity's `t-Reordering` and deliver whatever expires
    /// (TS 38.323 §5.2.2.2, issue #33).
    ///
    /// Needed because the timer is otherwise only evaluated when a PDU arrives, so
    /// a gap at the END of a flow would hold the SDUs behind it indefinitely -- the
    /// very case reordering exists to bound.
    #[cfg(feature = "drb-pdcp")]
    async fn poll_pdcp_timers(&mut self) {
        let now_ms = self.pdcp_now_ms();
        // `(drb_id, sdu)`: the entity map is keyed by DRB identity since issue #44,
        // and the PSI the SDU is delivered under is resolved from it below.
        let mut delivered: Vec<(i32, Vec<u8>)> = Vec::new();
        for (drb_id, pdcp) in &mut self.pdcp_entities {
            for sdu in pdcp.poll_t_reordering(now_ms) {
                delivered.push((*drb_id, sdu));
            }
        }
        for (drb_id, sdu) in delivered {
            debug!(
                "PDCP t-Reordering released an SDU: drb_id={drb_id}, len={}",
                sdu.len()
            );
            // This is the SAME downlink SDU `handle_pdu_transmission` would have
            // forwarded had t-Reordering not held it, so it owes the same SDAP strip
            // (issue #44). Missing it here is the subtle half: only SDUs that arrived
            // out of order take this path, so the header octet would leak into NAS for
            // exactly the packets a reordering bug is hardest to attribute to.
            #[cfg(feature = "sdap-dataplane")]
            let sdu = match nextgsim_pdcp::sdap::decode_dl_pdu(&sdu) {
                Ok((header, payload)) => {
                    debug!(
                        "SDAP DL (reordered): drb_id={drb_id}, qfi={}, rqi={}",
                        header.qfi, header.rqi
                    );
                    payload.to_vec()
                }
                Err(e) => {
                    warn!("Dropped a reordered downlink SDU with an undecodable SDAP header: {e}");
                    continue;
                }
            };
            // NAS is per session, so the DRB the SDU was reordered on has to be
            // resolved back to its PSI: `UplinkDataDelivery` is keyed by PSI, and a DRB
            // identity here would attribute the packet to the wrong session once the
            // two differ.
            let _ = self
                .task_base
                .nas_tx
                .send(NasMessage::UplinkDataDelivery {
                    psi: self.psi_for_drb(drb_id),
                    data: OctetString::from_slice(&sdu),
                })
                .await;
        }
    }

    /// Sends one RLC PDU (data or STATUS) to the serving cell.
    ///
    /// `drb_id` goes on the wire in `payload`, which is what the gNB demultiplexes its
    /// own per-bearer entities on — so this is the field that has to name the DRB and
    /// not the session, or the two ends key their entity maps differently.
    async fn send_rlc_pdu(&mut self, drb_id: i32, pdu: Vec<u8>) {
        let Some(dest) = self
            .serving_cell
            .and_then(|id| self.cell_addresses.get(&id).copied())
        else {
            warn!("Cannot send RLC PDU for drb_id={}: no serving cell", drb_id);
            return;
        };
        let transmission = self
            .transport
            .create_data_transmission(drb_id as u32, Bytes::from(pdu));
        self.send_rls_message(dest, &RlsProtocolMessage::PduTransmission(transmission))
            .await;
    }

    /// Sends the STATUS report an AM bearer owes its peer, if one is due
    /// (TS 38.322 §5.3.4). A no-op for a UM bearer, which has no STATUS PDU.
    async fn send_pending_status(&mut self, drb_id: i32) {
        let status = self
            .rlc_entities
            .get_mut(&drb_id)
            .and_then(RlcEntity::build_status_pdu);
        if let Some(status) = status {
            self.send_rlc_pdu(drb_id, status).await;
        }
    }

    /// Drives `t-Reassembly` on every RLC entity (TS 38.322 §5.2.2.2.4), so a
    /// partially received SDU whose missing segment never arrives is discarded
    /// instead of occupying the reassembly buffer forever.
    async fn poll_rlc_timers(&mut self) {
        let now = Instant::now();
        // `(drb_id, pdu)`: the entity map is keyed by DRB identity since issue #44, and
        // a re-offered PDU has to go back out on the bearer it came from -- a PSI here
        // would put a retransmission on the wrong DRB when a session has two.
        let mut outbound: Vec<(i32, Vec<u8>)> = Vec::new();
        for (drb_id, rlc) in &mut self.rlc_entities {
            if rlc.poll_timers(now) {
                debug!(
                    "RLC timer expired: drb_id={}, rx_next_reassembly={}",
                    drb_id,
                    rlc.rx_next_reassembly()
                );
            }
            if let Some(status) = rlc.build_status_pdu() {
                outbound.push((*drb_id, status));
            }
            while let Some(retx) = rlc.build_pdu(MAC_GRANT_BYTES) {
                outbound.push((*drb_id, retx));
            }
        }
        for (drb_id, pdu) in outbound {
            self.send_rlc_pdu(drb_id, pdu).await;
        }
    }

    async fn init_socket(&mut self) -> Result<(), std::io::Error> {
        let bind_addr = self
            .config
            .bind_address
            .unwrap_or_else(|| "0.0.0.0:0".parse().expect("value expected"));
        let socket = UdpSocket::bind(bind_addr).await?;
        info!("RLS task bound to {}", socket.local_addr()?);
        self.socket = Some(Arc::new(socket));
        Ok(())
    }

    async fn send_heartbeats(&mut self) {
        // Paced solely by `heartbeat_timer` (1 s tokio interval). The previous
        // `should_send_heartbeats()` wall-clock gate double-rate-limited against
        // that timer: under scheduling jitter (e.g. the registration RRC burst) a
        // tick could land <1 s after the last send, get gated out, and push the
        // next heartbeat ~2 s out — exceeding the 2 s cell-lost threshold and
        // spuriously dropping the serving cell (SignalLostToConnectedCell) right
        // after registration, before the PDU session could complete.
        // The UE's simulated position now (#324): its configured start, moved
        // along `sim_movement` if one is set. Each gNB derives this UE's received
        // level from it, so a moving UE's measurements change and a handover can
        // be driven. Carried in every heartbeat.
        let cfg = &self.task_base.config;
        let pos = match cfg.sim_movement {
            Some(movement) => movement.position_at(cfg.sim_position, self.started_at.elapsed()),
            None => cfg.sim_position,
        };
        self.cell_search
            .set_position(SimCoord::new(pos.x, pos.y, pos.z));
        for (addr, heartbeat) in self.cell_search.create_heartbeats() {
            self.send_rls_message(addr, &RlsProtocolMessage::Heartbeat(heartbeat))
                .await;
        }
    }

    /// A cell stopped answering heartbeats. Losing the serving cell is a radio
    /// link failure (TS 38.331 §5.3.10.3). Every lost cell, serving or not, is
    /// then reported to RRC as gone, so its cell selector stops offering it
    /// (issue #264). It used to hear only the RLF, so the dead cell stayed
    /// selectable forever.
    ///
    /// The RLF goes first: RRC handles it with the failed cell still as its
    /// serving cell, which the re-establishment request names.
    async fn on_cell_lost(&mut self, cell_id: u32) {
        info!("Cell lost: cell_id={}", cell_id);
        let cell_id = cell_id as i32;
        self.cell_addresses.remove(&cell_id);
        if self.serving_cell == Some(cell_id) {
            self.serving_cell = None;
            self.transport.clear_serving_endpoint();
            // The serving gNB may have restarted at a new address (#286).
            self.reresolve_search_hosts().await;
            let _ = self
                .task_base
                .rrc_tx
                .send(RrcMessage::RadioLinkFailure {
                    cause: RlfCause::SignalLostToConnectedCell,
                })
                .await;
        }
        let _ = self
            .task_base
            .rrc_tx
            .send(RrcMessage::SignalChanged {
                cell_id,
                dbm: CELL_SIGNAL_LOST_DBM,
            })
            .await;
    }

    async fn check_lost_cells(&mut self) {
        for event in self.cell_search.check_lost_cells() {
            if let CellSearchEvent::CellLost { cell_id } = event {
                self.on_cell_lost(cell_id).await;
            }
        }
    }

    async fn send_pending_acks(&mut self) {
        for (endpoint_id, ack) in self.transport.create_pending_acks() {
            if let Some(&addr) = self.cell_addresses.get(&(endpoint_id as i32)) {
                self.send_rls_message(addr, &RlsProtocolMessage::PduTransmissionAck(ack))
                    .await;
            }
        }
    }

    async fn check_expired_pdus(&mut self) {
        for event in self.transport.check_expired_pdus() {
            if let TransportEvent::TransmissionFailure { pdus } = event {
                warn!("Transmission failure: {} PDUs expired", pdus.len());
            }
        }
    }

    async fn send_rls_message(&self, dest: SocketAddr, msg: &RlsProtocolMessage) {
        if let Some(socket) = &self.socket {
            if let Err(e) = socket.send_to(&codec::encode(msg), dest).await {
                error!("Failed to send RLS message to {}: {}", dest, e);
            }
        }
    }

    /// The NTN pre-compensation in force, or `None` on a terrestrial cell.
    ///
    /// Exposed so a test can assert the value that was **applied** to the uplink
    /// rather than a log line claiming it was (issue #56, criterion 5).
    pub fn ntn_precompensation_applied(&self) -> Option<NtnUplinkPrecompensation> {
        self.ntn_precompensation
    }

    /// The uplink transmit timing of the last uplink this task sent, in
    /// microseconds relative to the downlink frame boundary it is referenced to
    /// (issue #56).
    ///
    /// Negative under an NTN timing advance, because the UE transmits `T_TA`
    /// microseconds EARLIER than the downlink frame boundary so its signal arrives
    /// at the uplink time synchronisation reference point frame aligned
    /// (TS 38.300 §16.14.2.1). Exactly `0.0` on a terrestrial cell.
    ///
    /// `None` until the task has sent an uplink, so a test cannot mistake "not yet
    /// transmitted" for "transmitted with no advance".
    pub fn last_uplink_timing_offset_us(&self) -> Option<f64> {
        self.last_uplink_timing_offset_us
    }

    /// Applies the NTN uplink pre-compensation to the transmission about to go out,
    /// and returns the transmit timing offset it produced
    /// (TS 38.300 §16.14.2.2, issue #56).
    ///
    /// **This is the read site criterion 2 asks for.** Every uplink RLS transmission
    /// calls it, so `self.ntn_precompensation` is consulted on the transmit path
    /// rather than merely stored.
    ///
    /// # What "applying" means on this simulated air interface
    ///
    /// A timing advance is a shift of the UE's uplink frame timing: the UE transmits
    /// `T_TA` earlier than the downlink frame boundary it synchronised to, so that
    /// after the propagation delay its signal is frame aligned at the reference point
    /// (§16.14.2.1). The quantity that IS the timing advance is therefore the uplink
    /// transmit instant relative to that boundary, and that is what this computes and
    /// records — it is observable, it is derived from the ephemeris, and a receiver
    /// with a slot grid would see it directly.
    ///
    /// What it deliberately does NOT do is `sleep`. The RLS air interface is loopback
    /// UDP with no propagation model: there is no delay for an advance to cancel, so
    /// delaying the `send_to` would make the uplink arrive *late* by the very amount
    /// the advance exists to remove — applying the compensation backwards. And an
    /// advance cannot be slept at all, because it moves the transmission earlier than
    /// now. Adding a propagation delay to the RLS transport so that an advance had
    /// something to cancel is a change to the air interface every non-NTN scenario
    /// shares, and is out of this issue's scope; the timing offset recorded here is
    /// the input such a model would consume.
    ///
    /// The Doppler pre-compensation is applied to the transmitter's carrier
    /// (`doppler_hz`), which likewise has no byte on a wire that carries no carrier.
    /// Both applied values are readable via [`Self::ntn_precompensation_applied`] and
    /// [`Self::last_uplink_timing_offset_us`], which is what the end-to-end test
    /// asserts rather than a log line.
    fn apply_ntn_uplink_timing(&mut self) -> f64 {
        let offset_us = match self.ntn_precompensation {
            // Negated: an ADVANCE moves the transmission earlier than the reference
            // boundary, so the offset is negative. A positive number here would be a
            // delay, which is the opposite pre-compensation.
            Some(pre) => -pre.ta_us,
            // A terrestrial UE transmits on the boundary: no NTN advance to apply.
            None => 0.0,
        };
        self.last_uplink_timing_offset_us = Some(offset_us);
        offset_us
    }

    async fn handle_receive_rls_message(&mut self, data: &[u8], source: SocketAddr) {
        match codec::decode(&Bytes::copy_from_slice(data)) {
            Ok(msg) => self.process_rls_message(msg, source).await,
            Err(e) => warn!("Failed to decode RLS message from {}: {}", source, e),
        }
    }

    async fn process_rls_message(&mut self, msg: RlsProtocolMessage, source: SocketAddr) {
        match msg {
            RlsProtocolMessage::HeartbeatAck(ack) => self.handle_heartbeat_ack(source, &ack).await,
            RlsProtocolMessage::PduTransmission(pdu) => {
                self.handle_pdu_transmission(source, &pdu).await
            }
            RlsProtocolMessage::PduTransmissionAck(ack) => self.transport.process_pdu_ack(&ack),
            RlsProtocolMessage::Heartbeat(_) => debug!("Ignoring heartbeat from {}", source),
        }
    }

    async fn handle_heartbeat_ack(
        &mut self,
        source: SocketAddr,
        ack: &nextgsim_rls::RlsHeartbeatAck,
    ) {
        for event in self.cell_search.process_heartbeat_ack(ack.sti, source, ack) {
            match event {
                CellSearchEvent::CellDiscovered {
                    cell_id,
                    sti: _,
                    dbm,
                } => {
                    info!("Cell discovered: cell_id={}, dbm={}", cell_id, dbm);
                    self.cell_addresses.insert(cell_id as i32, source);
                    let _ = self
                        .task_base
                        .rrc_tx
                        .send(RrcMessage::SignalChanged {
                            cell_id: cell_id as i32,
                            dbm,
                        })
                        .await;
                }
                CellSearchEvent::SignalChanged {
                    cell_id,
                    old_dbm: _,
                    new_dbm,
                } => {
                    let _ = self
                        .task_base
                        .rrc_tx
                        .send(RrcMessage::SignalChanged {
                            cell_id: cell_id as i32,
                            dbm: new_dbm,
                        })
                        .await;
                }
                CellSearchEvent::CellLost { cell_id } => {
                    self.on_cell_lost(cell_id).await;
                }
            }
        }
    }

    async fn handle_pdu_transmission(
        &mut self,
        source: SocketAddr,
        pdu: &nextgsim_rls::RlsPduTransmission,
    ) {
        let cell_id = match self
            .cell_addresses
            .iter()
            .find(|(_, addr)| **addr == source)
            .map(|(id, _)| *id)
        {
            Some(id) => id,
            None => {
                // Expected while two gNB incarnations share an STI (a pod
                // restart's overlap, #292): the one not adopted as a cell still
                // broadcasts to this UE. Warned once per source, not per PDU.
                if self.unknown_sources_warned.insert(source) {
                    warn!(
                        "PDU from unknown source {} (not a discovered cell); further PDUs \
                         from it are dropped silently",
                        source
                    );
                }
                return;
            }
        };

        // Set when an AM bearer received data and may owe a STATUS report; the
        // send happens after the loop so the entity borrow is released first. Names the
        // DRB since issue #44, because that is what `send_pending_status` keys and what
        // a STATUS report has to go back out on.
        let mut pending_status_drb: Option<i32> = None;

        // Collect events first so the transport borrow is released before we
        // mutate self.rlc_entities below.
        let events: Vec<TransportEvent> =
            self.transport.process_pdu_transmission(cell_id as u32, pdu);

        for event in events {
            match event {
                TransportEvent::RrcReceived { channel, data, .. } => {
                    let pdu = OctetString::from_slice(&data);
                    let _ = self
                        .task_base
                        .rrc_tx
                        .send(RrcMessage::DownlinkRrcDelivery {
                            cell_id,
                            channel,
                            pdu,
                        })
                        .await;
                }
                // `psi` here is `msg.payload` off the wire, which the gNB sets to the
                // DRB identity (see its `send_rlc_pdu`) — so it is bound as `drb_id`.
                // Without `sdap-dataplane` that number IS the PSI, which is what this
                // field meant before issue #44.
                TransportEvent::DataReceived { psi: drb_id, data } => {
                    // Feed the received RLC PDU into the per-bearer entity and
                    // forward any fully-reassembled SDUs up to NAS.
                    // Collect reassembled SDUs first so the mutable borrow on
                    // self.rlc_entities is released before the async send.
                    let drb_id = drb_id as i32;
                    // The session the bearer belongs to, which is what NAS needs: the
                    // delivery is per session and the radio bearer is not. Resolved
                    // rather than derived -- see `psi_for_drb`.
                    let psi = self.psi_for_drb(drb_id);
                    debug!(
                        "Downlink data (RLC): drb_id={}, psi={}, len={}",
                        drb_id,
                        psi,
                        data.len()
                    );
                    let reassembled = {
                        let rlc = self.rlc_entity_for(drb_id, psi);
                        rlc.receive_pdu(&data);
                        let mut sdus = Vec::new();
                        while let Some(sdu) = rlc.poll_reassembled() {
                            sdus.push(sdu);
                        }
                        sdus
                    };
                    // An AM bearer answers a poll (or a detected gap) with a
                    // STATUS report (TS 38.322 §5.3.4); without it the gNB's ARQ
                    // never learns anything and re-polls forever.
                    pending_status_drb = Some(drb_id);

                    // PDCP receive (issue #33): what RLC reassembled is a PDCP
                    // PDU, so it goes through the reordering entity and only
                    // in-order SDUs reach NAS. Without the feature the reassembled
                    // bytes go straight up, as before.
                    #[cfg(feature = "drb-pdcp")]
                    let to_nas: Vec<Vec<u8>> = {
                        let now_ms = self.pdcp_now_ms();
                        let pdcp = self.pdcp_entity_for(drb_id);
                        let mut delivered = Vec::new();
                        for sdu in reassembled {
                            match pdcp.receive_pdu(&sdu, now_ms) {
                                Ok(sdus) => delivered.extend(sdus),
                                Err(e) => {
                                    debug!("PDCP discarded a downlink PDU on DRB {drb_id}: {e:?}")
                                }
                            }
                        }
                        delivered.extend(pdcp.poll_t_reordering(now_ms));
                        delivered
                    };
                    #[cfg(not(feature = "drb-pdcp"))]
                    let to_nas: Vec<Vec<u8>> = reassembled;

                    for sdu in to_nas {
                        // SDAP receive (TS 37.324 §5.2.2, issue #44): the gNB prepends a
                        // one-octet DL header, and it MUST come off here. NAS writes what
                        // it is handed to the TUN verbatim, so leaving the octet on would
                        // deliver an IP packet whose first byte is an SDAP header -- a
                        // corrupt version/IHL nibble, which the kernel drops silently
                        // rather than reporting. Stripped AFTER PDCP because SDAP is
                        // above it (TS 37.324 §4.2): the header is inside what PDCP
                        // protected, so it is only in the clear once PDCP is done.
                        #[cfg(feature = "sdap-dataplane")]
                        let sdu = match nextgsim_pdcp::sdap::decode_dl_pdu(&sdu) {
                            Ok((header, payload)) => {
                                // The QFI and RQI are logged and not acted on. The QFI
                                // does not re-route anything -- the DRB already reached
                                // the right entities and the PSI names the session -- and
                                // reflective QoS (RQI, TS 23.501 §5.7.5.3) would have the
                                // UE derive an uplink packet filter from the downlink
                                // flow, which needs the URSP machinery this UE does not
                                // have. Logging it is honest; silently ignoring it would
                                // not be.
                                debug!(
                                    "SDAP DL: drb_id={drb_id}, psi={psi}, qfi={}, rqi={}",
                                    header.qfi, header.rqi
                                );
                                payload.to_vec()
                            }
                            Err(e) => {
                                // A malformed header means the peer is not speaking this
                                // sublayer (a gNB built without the feature) or sent a
                                // Control PDU, which carries no user data at all.
                                // DISCARDED rather than delivered, because the
                                // alternative is handing NAS the header octet.
                                warn!(
                                    "Dropped a downlink SDU with an undecodable SDAP header on \
                                     DRB {drb_id}: {e}"
                                );
                                continue;
                            }
                        };

                        debug!(
                            "DRB SDU for NAS: drb_id={}, psi={}, len={}",
                            drb_id,
                            psi,
                            sdu.len()
                        );
                        let octet = OctetString::from_slice(&sdu);
                        let _ = self
                            .task_base
                            .nas_tx
                            .send(NasMessage::UplinkDataDelivery { psi, data: octet })
                            .await;
                    }
                }
                TransportEvent::TransmissionFailure { pdus } => {
                    warn!("Transmission failure: {} PDUs", pdus.len())
                }
                TransportEvent::RadioLinkFailure { cause } => {
                    let rlf_cause = match cause {
                        nextgsim_rls::RlfCause::PduIdExists => RlfCause::PduIdExists,
                        nextgsim_rls::RlfCause::PduIdFull => RlfCause::PduIdFull,
                        nextgsim_rls::RlfCause::SignalLostToConnectedCell => {
                            RlfCause::SignalLostToConnectedCell
                        }
                    };
                    let _ = self
                        .task_base
                        .rrc_tx
                        .send(RrcMessage::RadioLinkFailure { cause: rlf_cause })
                        .await;
                }
            }
        }

        if let Some(drb_id) = pending_status_drb {
            self.send_pending_status(drb_id).await;
        }
    }

    fn handle_assign_current_cell(&mut self, cell_id: i32) {
        if self.cell_search.get_cell(cell_id as u32).is_some() {
            info!("Assigning serving cell: cell_id={}", cell_id);
            self.serving_cell = Some(cell_id);
            self.transport.set_serving_endpoint(cell_id as u32);
        } else {
            warn!("Cannot assign unknown cell as serving: cell_id={}", cell_id);
        }
    }

    fn handle_reset_sti(&mut self) {
        info!("Resetting STI");
        self.sti = rand::random::<u64>();
        self.serving_cell = None;
        self.cell_addresses.clear();
        self.transport = RlsTransport::new(self.sti);
        let search_space: Vec<SocketAddr> = self.effective_search_space();
        self.cell_search = UeCellSearch::new(self.sti, search_space);
        self.cell_search
            .set_heartbeat_interval(self.config.heartbeat_interval);
        self.cell_search
            .set_heartbeat_threshold(self.config.heartbeat_threshold);
    }

    async fn handle_rrc_pdu_delivery(
        &mut self,
        channel: RrcChannel,
        pdu_id: u32,
        pdu: OctetString,
    ) {
        let (cell_id, dest) = match (
            self.serving_cell,
            self.serving_cell
                .and_then(|id| self.cell_addresses.get(&id).copied()),
        ) {
            (Some(id), Some(addr)) => (id, addr),
            _ => {
                warn!("Cannot send uplink RRC: no serving cell");
                return;
            }
        };

        // Apply the NTN uplink timing advance and Doppler pre-compensation before the
        // transmission leaves (TS 38.300 §16.14.2.2, issue #56). Zero on a
        // terrestrial cell, so a non-NTN uplink is unchanged.
        let ntn_offset_us = self.apply_ntn_uplink_timing();
        debug!(
            "Uplink RRC: cell_id={}, channel={:?}, pdu_id={}, len={}, \
             ntn_transmit_offset={}us",
            cell_id,
            channel,
            pdu_id,
            pdu.len(),
            ntn_offset_us
        );
        let require_ack = pdu_id != 0;
        match self.transport.create_rrc_transmission(
            cell_id as u32,
            channel,
            Bytes::copy_from_slice(pdu.data()),
            require_ack,
        ) {
            Ok(transmission) => {
                self.send_rls_message(dest, &RlsProtocolMessage::PduTransmission(transmission))
                    .await
            }
            Err(cause) => {
                let rlf_cause = match cause {
                    nextgsim_rls::RlfCause::PduIdExists => RlfCause::PduIdExists,
                    nextgsim_rls::RlfCause::PduIdFull => RlfCause::PduIdFull,
                    nextgsim_rls::RlfCause::SignalLostToConnectedCell => {
                        RlfCause::SignalLostToConnectedCell
                    }
                };
                let _ = self
                    .task_base
                    .rrc_tx
                    .send(RrcMessage::RadioLinkFailure { cause: rlf_cause })
                    .await;
            }
        }
    }

    /// Send uplink user-plane data from NAS/TUN to the gNB.
    ///
    /// The SDU is first submitted to the bearer's RLC entity (UM, SN12) which
    /// segments it if necessary.  Each resulting RLC PDU is then wrapped in an
    /// RLS frame and sent to the serving gNB.
    ///
    /// # With `sdap-dataplane`: the SDU gains an SDAP header and picks a DRB
    ///
    /// The SDU arrives here carrying only a PSI — nothing above has inspected the
    /// packet — so the QoS flow it belongs to is the session's **default** flow on its
    /// **default DRB**, which is the mapping the network signalled and this task
    /// recorded in [`Self::sdap_uplink`]. Per-packet uplink classification would need a
    /// URSP / TFT matcher (TS 23.503 §6.6.2) to test the 5-tuple against traffic
    /// descriptors the network provisioned, and this UE has neither the descriptors nor
    /// the matcher — so inventing a classifier here would produce QFIs the network
    /// never admitted. See [`SdapUplinkBinding`].
    async fn handle_data_pdu_delivery(&mut self, psi: i32, pdu: OctetString) {
        let dest = match self
            .serving_cell
            .and_then(|id| self.cell_addresses.get(&id).copied())
        {
            Some(addr) => addr,
            None => {
                warn!("Cannot send uplink data: no serving cell");
                return;
            }
        };

        // Which bearer this session's uplink rides, and what goes on the wire.
        //
        // Without the feature there is one DRB per session and its identity IS the
        // PSI, so this is the number the pre-SDAP path always sent -- byte for byte.
        #[cfg(not(feature = "sdap-dataplane"))]
        let drb_id = psi;
        // With it, the network told the UE which DRB carries the session's default flow
        // and which QFI names that flow. `None` when no mapping has arrived yet -- a
        // packet that beat its RRCReconfiguration -- and then the SDU goes out
        // unstamped on the PSI-numbered bearer: that is the session's default DRB in
        // every case (`allocate_drbs` keeps `default_drb_id == psi.clamp(1, 32)`), so
        // the gNB's entity lookup still lands, and a gNB with the feature on will
        // discard the headerless SDU rather than mis-deliver it. Dropping it here
        // instead would lose the first packet of every session to a race.
        #[cfg(feature = "sdap-dataplane")]
        let (drb_id, sdap_qfi) = match self.sdap_uplink.get(&psi) {
            Some(binding) => (binding.drb_id, Some(binding.qfi)),
            None => {
                debug!(
                    "No SDAP uplink mapping for PSI {psi} yet; sending on the PSI-numbered \
                     default DRB without a header"
                );
                (psi, None)
            }
        };

        // Apply the NTN uplink timing advance and Doppler pre-compensation
        // (TS 38.300 §16.14.2.2, issue #56). The user plane gets it as well as the
        // control plane, because §16.14.2.2 pre-compensates "the uplink
        // transmissions" -- a UE that advanced its SRB but not its DRB would have the
        // two arrive at the reference point in different frames. Zero on a
        // terrestrial cell.
        let ntn_offset_us = self.apply_ntn_uplink_timing();
        debug!(
            "Uplink data (RLC): psi={}, drb_id={}, len={}, ntn_transmit_offset={}us",
            psi,
            drb_id,
            pdu.len(),
            ntn_offset_us
        );

        // SDAP transmit (TS 37.324 §5.2.1, issue #44): the one-octet UL header goes on
        // FIRST, before PDCP and therefore before RLC, because SDAP is above PDCP
        // (§4.2) -- the header has to end up inside what PDCP protects and what RLC
        // segments, which is also the order the gNB's receive path unwinds.
        #[cfg(feature = "sdap-dataplane")]
        let pdu = match sdap_qfi {
            // A QFI wider than the six-bit field cannot be signalled at all
            // (TS 38.413 caps `QosFlowIdentifier` at 63), so this needs a
            // non-conformant network. The SDU is dropped rather than sent unstamped: a
            // gNB with the feature on would discard it anyway, and truncating the QFI
            // would attribute the traffic to a DIFFERENT flow.
            Some(qfi) => match nextgsim_pdcp::sdap::build_ul_pdu(qfi, pdu.data()) {
                Ok(sdap_pdu) => OctetString::from_slice(&sdap_pdu),
                Err(e) => {
                    warn!("Dropped an uplink SDU with an unencodable SDAP header: {e}");
                    return;
                }
            },
            None => pdu,
        };

        // Submit SDU to RLC and collect all resulting PDUs before releasing
        // the mutable borrow so that self.transport and self.socket are
        // accessible again for transmission.
        let rlc_pdus = {
            // PDCP next (issue #33): the SDU gets a PDCP header, an SN and a
            // discardTimer, and it is the PDCP PDU -- not the raw IP packet -- that
            // RLC segments. Without the feature the IP packet goes to RLC directly,
            // as before. Keyed by DRB since issue #44, matching the entity the gNB
            // will verify against.
            #[cfg(feature = "drb-pdcp")]
            let to_rlc: Vec<Vec<u8>> = {
                let now_ms = self.pdcp_now_ms();
                let pdcp = self.pdcp_entity_for(drb_id);
                pdcp.submit_sdu(pdu.data(), now_ms);
                pdcp.take_transmittable(now_ms)
            };
            #[cfg(not(feature = "drb-pdcp"))]
            let to_rlc: Vec<Vec<u8>> = vec![pdu.data().to_vec()];

            let rlc = self.rlc_entity_for(drb_id, psi);
            for sdu in to_rlc {
                rlc.submit_sdu(sdu);
            }
            let mut pdus = Vec::new();
            while let Some(rlc_pdu) = rlc.build_pdu(MAC_GRANT_BYTES) {
                pdus.push(rlc_pdu);
            }
            pdus
        };

        for rlc_pdu in rlc_pdus {
            // The DRB identity, not the PSI: this is what the gNB keys its own RLC and
            // PDCP entities on, so the two ends have to name the same thing.
            let transmission = self
                .transport
                .create_data_transmission(drb_id as u32, Bytes::from(rlc_pdu));
            self.send_rls_message(dest, &RlsProtocolMessage::PduTransmission(transmission))
                .await;
        }
    }

    async fn handle_rls_message(&mut self, msg: RlsMessage) {
        match msg {
            RlsMessage::AssignCurrentCell { cell_id } => self.handle_assign_current_cell(cell_id),
            #[cfg(feature = "up-security")]
            RlsMessage::InstallDrbSecurity {
                psi,
                drb_id,
                security,
            } => {
                // The PSI is recorded alongside the entity so a downlink SDU on this
                // bearer can be delivered under the right session even when the keys
                // arrive before any traffic does -- which is the normal order.
                self.drb_to_psi.insert(drb_id, psi);
                self.install_drb_security(drb_id, security.map(|b| *b));
            }
            #[cfg(feature = "sdap-dataplane")]
            RlsMessage::InstallSdapMapping {
                psi,
                drb_id,
                qfis,
                default_drb,
            } => self.install_sdap_mapping(psi, drb_id, &qfis, default_drb),
            // Install the pre-compensation the RRC task derived from SIB19
            // (TS 38.300 §16.14.2.2, issue #56). Applied by
            // `apply_ntn_uplink_timing` on every subsequent uplink.
            RlsMessage::ApplyNtnPrecompensation {
                ta_us,
                doppler_hz,
                k_offset,
            } => {
                info!(
                    "NTN uplink pre-compensation installed: T_TA={ta_us:.3} us, \
                     Doppler={doppler_hz:.1} Hz, K_offset={k_offset} slots \
                     (TS 38.300 §16.14.2.2)"
                );
                self.ntn_precompensation = Some(NtnUplinkPrecompensation {
                    ta_us,
                    doppler_hz,
                    k_offset,
                });
            }
            RlsMessage::RrcPduDelivery {
                channel,
                pdu_id,
                pdu,
            } => self.handle_rrc_pdu_delivery(channel, pdu_id, pdu).await,
            RlsMessage::ResetSti => self.handle_reset_sti(),
            RlsMessage::DataPduDelivery { psi, pdu } => {
                self.handle_data_pdu_delivery(psi, pdu).await
            }
            RlsMessage::ReceiveRlsMessage { data, .. } => {
                debug!("Received internal RLS message, len={}", data.len())
            }
            RlsMessage::SignalChanged { cell_id, dbm } => {
                debug!("Signal changed: cell_id={}, dbm={}", cell_id, dbm)
            }
            RlsMessage::UplinkData { psi, data } => self.handle_data_pdu_delivery(psi, data).await,
            RlsMessage::UplinkRrc {
                channel,
                pdu_id,
                data,
                ..
            } => self.handle_rrc_pdu_delivery(channel, pdu_id, data).await,
            RlsMessage::DownlinkData { psi, data } => {
                let _ = self
                    .task_base
                    .nas_tx
                    .send(NasMessage::UplinkDataDelivery { psi, data })
                    .await;
            }
            RlsMessage::DownlinkRrc {
                cell_id,
                channel,
                data,
            } => {
                let _ = self
                    .task_base
                    .rrc_tx
                    .send(RrcMessage::DownlinkRrcDelivery {
                        cell_id,
                        channel,
                        pdu: data,
                    })
                    .await;
            }
            RlsMessage::RadioLinkFailure { cause } => warn!("Radio link failure: {:?}", cause),
            RlsMessage::TransmissionFailure { pdu_list } => {
                warn!("Transmission failure: {} PDUs", pdu_list.len())
            }
        }
    }
}

#[async_trait::async_trait]
impl Task for RlsTask {
    type Message = RlsMessage;

    async fn run(&mut self, mut rx: mpsc::Receiver<TaskMessage<Self::Message>>) {
        info!("RLS task starting");
        if let Err(e) = self.init_socket().await {
            error!("Failed to initialize RLS socket: {}", e);
            return;
        }
        // Hostnames are resolved before the first heartbeat; one that does not
        // resolve yet is retried on `resolve_timer` (#286).
        self.reresolve_search_hosts().await;
        info!(
            "RLS task started with {} gNB address(es) and {} gNB hostname(s) in search list",
            self.config.gnb_search_list.len(),
            self.config.gnb_search_hosts.len()
        );
        let mut resolve_timer = interval(self.config.reresolve_interval);
        resolve_timer.tick().await;
        let has_hosts = !self.config.gnb_search_hosts.is_empty();

        let mut heartbeat_timer = interval(Duration::from_millis(HEARTBEAT_INTERVAL_MS));
        let mut lost_cell_timer = interval(Duration::from_millis(LOST_CELL_CHECK_INTERVAL_MS));
        // `interval` fires its first tick immediately; for this timer there is
        // nothing to do at start-up (no cell known, no PDU pending, no RLC
        // entity), and consuming it keeps the periodic work at predictable
        // multiples of the period instead of racing the first packet. The
        // heartbeat timer above deliberately keeps its immediate tick: that one
        // starts cell discovery.
        lost_cell_timer.tick().await;

        loop {
            tokio::select! {
                Some(msg) = rx.recv() => {
                    match msg {
                        TaskMessage::Message(rls_msg) => self.handle_rls_message(rls_msg).await,
                        TaskMessage::Shutdown => { info!("RLS task received shutdown signal"); break; }
                    }
                }
                result = async {
                    if let Some(socket) = &self.socket {
                        let mut buf = vec![0u8; UDP_BUFFER_SIZE];
                        socket.recv_from(&mut buf).await.ok().map(|(len, addr)| { buf.truncate(len); (buf, addr) })
                    } else { None }
                } => {
                    if let Some((data, source)) = result {
                        self.handle_receive_rls_message(&data, source).await;
                    }
                }
                _ = heartbeat_timer.tick() => self.send_heartbeats().await,
                _ = resolve_timer.tick(), if has_hosts => self.reresolve_search_hosts().await,
                _ = lost_cell_timer.tick() => {
                    self.check_lost_cells().await;
                    self.send_pending_acks().await;
                    self.check_expired_pdus().await;
                    self.poll_rlc_timers().await;
                    #[cfg(feature = "drb-pdcp")]
                    self.poll_pdcp_timers().await;
                }
            }
        }
        info!(
            "RLS task stopped with {} discovered cells",
            self.cell_search.cell_count()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nextgsim_common::config::UeConfig;

    fn test_config() -> UeConfig {
        UeConfig {
            gnb_search_list: vec!["127.0.0.1".to_string()],
            ..Default::default()
        }
    }

    #[test]
    fn test_rls_task_config_default() {
        let config = RlsTaskConfig::default();
        assert!(config.gnb_search_list.is_empty());
        assert!(config.bind_address.is_none());
    }

    #[test]
    fn test_rls_task_creation() {
        let config = test_config();
        let (task_base, _app_rx, _nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(config, 16);
        let rls_config = RlsTaskConfig {
            gnb_search_list: vec!["127.0.0.1:4997".parse().unwrap()],
            ..Default::default()
        };
        let task = RlsTask::new(task_base, rls_config);
        assert!(task.serving_cell.is_none());
        assert_eq!(task.cell_count(), 0);
    }

    #[test]
    fn test_rls_task_from_ue_config() {
        let config = test_config();
        let (task_base, _app_rx, _nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(config, 16);
        let task = RlsTask::from_ue_config(task_base);
        assert!(task.serving_cell.is_none());
        assert_eq!(task.config.gnb_search_list.len(), 1);
    }

    fn lit(s: &str) -> PeerAddr {
        PeerAddr::Literal(s.parse().unwrap())
    }

    #[test]
    fn test_parse_gnb_search_entry_bare_ip_gets_default_port() {
        let addr = parse_gnb_search_entry("10.0.0.1").expect("bare IP must parse");
        assert_eq!(addr, lit(&format!("10.0.0.1:{DEFAULT_RLS_PORT}")));
    }

    #[test]
    fn test_parse_gnb_search_entry_explicit_port_is_preserved() {
        // Previously dropped: the old filter_map only tried IpAddr, so an
        // explicit IP:port never parsed and vanished.
        let addr = parse_gnb_search_entry("10.0.0.1:5000").expect("IP:port must parse");
        assert_eq!(addr.port(), 5000);
    }

    /// #286: a hostname is accepted (it used to be rejected, and before that
    /// silently dropped) and kept as a name for the RLS task to resolve.
    #[test]
    fn test_parse_gnb_search_entry_accepts_a_hostname() {
        assert_eq!(
            parse_gnb_search_entry("gnb.nextg-system.svc.cluster.local"),
            Ok(PeerAddr::Host(
                "gnb.nextg-system.svc.cluster.local".into(),
                DEFAULT_RLS_PORT
            ))
        );
        assert_eq!(
            parse_gnb_search_entry("gnb:5000"),
            Ok(PeerAddr::Host("gnb".into(), 5000))
        );
    }

    #[test]
    fn test_parse_gnb_search_entry_rejects_empty_and_garbage() {
        assert!(parse_gnb_search_entry("").is_err());
        assert!(parse_gnb_search_entry("   ").is_err());
        let err = parse_gnb_search_entry("not an ip").unwrap_err();
        assert!(
            err.contains("not an ip"),
            "error must name the entry: {err}"
        );
    }

    #[test]
    fn test_parse_gnb_search_entry_tolerates_surrounding_whitespace() {
        let addr = parse_gnb_search_entry("  10.0.0.1  ").expect("whitespace must be trimmed");
        assert_eq!(addr, lit(&format!("10.0.0.1:{DEFAULT_RLS_PORT}")));
    }

    #[test]
    fn test_from_ue_config_splits_literals_and_hostnames() {
        let mut config = test_config();
        config.gnb_search_list = vec![
            "gnb.example.svc.cluster.local".to_string(),
            "10.0.0.7".to_string(),
            "not an ip".to_string(),
        ];
        let (task_base, _app_rx, _nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(config, 16);
        let task = RlsTask::from_ue_config(task_base);
        assert_eq!(task.config.gnb_search_list.len(), 1);
        assert_eq!(task.config.gnb_search_list[0].ip().to_string(), "10.0.0.7");
        assert_eq!(
            task.config.gnb_search_hosts,
            vec![PeerAddr::Host(
                "gnb.example.svc.cluster.local".into(),
                DEFAULT_RLS_PORT
            )]
        );
    }

    /// #286: re-resolving a gNB hostname replaces the address heartbeats go
    /// to. The UE last resolved the name to a dead address (the gNB pod that
    /// restarted); the name now resolves elsewhere (`localhost` stands in for
    /// the Service), and the search space follows it, while literals stay.
    #[tokio::test]
    async fn a_gnb_hostname_is_re_resolved_into_the_search_space() {
        let mut config = test_config();
        config.gnb_search_list = vec!["10.0.0.7".to_string(), "localhost:4321".to_string()];
        let (task_base, _app_rx, _nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(config, 16);
        let mut task = RlsTask::from_ue_config(task_base);
        let host = PeerAddr::Host("localhost".into(), 4321);
        let stale: SocketAddr = "192.0.2.1:4321".parse().unwrap();
        task.resolved_hosts.insert(host.clone(), vec![stale]);
        let space = task.effective_search_space();
        task.cell_search.set_search_space(space);
        assert!(task.cell_search.search_space().contains(&stale));

        task.reresolve_search_hosts().await;

        let space = task.cell_search.search_space();
        assert!(
            !space.contains(&stale),
            "the dead address must go: {space:?}"
        );
        assert!(
            space
                .iter()
                .any(|a| a.ip().is_loopback() && a.port() == 4321),
            "the new address must be searched: {space:?}"
        );
        assert!(
            space.contains(&"10.0.0.7:4997".parse().unwrap()),
            "literals stay"
        );
    }

    #[test]
    fn test_assign_current_cell() {
        let config = test_config();
        let (task_base, _app_rx, _nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(config, 16);
        let rls_config = RlsTaskConfig {
            gnb_search_list: vec!["127.0.0.1:4997".parse().unwrap()],
            ..Default::default()
        };
        let mut task = RlsTask::new(task_base, rls_config);
        task.handle_assign_current_cell(1);
        assert!(task.serving_cell.is_none()); // Unknown cell
    }

    /// Issue #264: losing a cell must reach RRC as the cell being gone, after
    /// the RLF for a serving cell. RRC heard only the RLF, so its cell selector
    /// kept offering the dead cell.
    #[tokio::test]
    async fn a_lost_serving_cell_reaches_rrc_as_an_rlf_then_as_gone() {
        let (task_base, _app_rx, _nas_rx, mut rrc_rx, _rls_rx) = UeTaskBase::new(test_config(), 16);
        let mut task = RlsTask::new(task_base, RlsTaskConfig::default());
        task.serving_cell = Some(3);

        task.on_cell_lost(3).await;

        assert!(task.serving_cell.is_none());
        assert!(matches!(
            rrc_rx.try_recv(),
            Ok(TaskMessage::Message(RrcMessage::RadioLinkFailure {
                cause: RlfCause::SignalLostToConnectedCell
            }))
        ));
        match rrc_rx.try_recv() {
            Ok(TaskMessage::Message(RrcMessage::SignalChanged { cell_id, dbm })) => {
                assert_eq!(cell_id, 3);
                assert!(dbm < crate::rrc::cell_selection::CELL_LOST_THRESHOLD_DBM);
            }
            other => panic!("expected the cell reported gone, got {other:?}"),
        }
    }

    /// A non-serving cell's loss is no RLF, but RRC still learns it is gone.
    #[tokio::test]
    async fn a_lost_neighbour_reaches_rrc_as_gone_only() {
        let (task_base, _app_rx, _nas_rx, mut rrc_rx, _rls_rx) = UeTaskBase::new(test_config(), 16);
        let mut task = RlsTask::new(task_base, RlsTaskConfig::default());
        task.serving_cell = Some(1);
        task.on_cell_lost(4).await;
        assert_eq!(task.serving_cell, Some(1));
        assert!(matches!(
            rrc_rx.try_recv(),
            Ok(TaskMessage::Message(RrcMessage::SignalChanged {
                cell_id: 4,
                ..
            }))
        ));
        assert!(rrc_rx.try_recv().is_err());
    }

    // ── RLC mode selector (#15) ──────────────────────────────────────────────

    /// A bearer listed in `rlc_am_psis` gets an AM entity; everything else keeps
    /// the UM SN12 default, so a deployment that configures nothing is unchanged.
    #[test]
    fn only_a_configured_psi_gets_an_acknowledged_mode_entity() {
        let config = UeConfig {
            rlc_am_psis: vec![5],
            ..test_config()
        };
        let (task_base, _app_rx, _nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(config, 16);
        let mut task = RlsTask::new(task_base, RlsTaskConfig::default());

        // `(drb_id, psi)`: the mode is decided by the PSI even though the entity is
        // keyed by the DRB identity, because `rlc_am_psis` is a list of sessions.
        assert_eq!(
            task.rlc_entity_for(5, 5).mode,
            RlcMode::AcknowledgedMode,
            "PSI 5 is configured for AM"
        );
        assert_eq!(
            task.rlc_entity_for(1, 1).mode,
            RlcMode::UnacknowledgedMode,
            "an unlisted PSI keeps the UM default"
        );
    }

    #[test]
    fn with_no_configuration_every_bearer_is_unacknowledged_mode() {
        let (task_base, _app_rx, _nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(test_config(), 16);
        let mut task = RlsTask::new(task_base, RlsTaskConfig::default());
        for psi in [1, 5, 15] {
            // The default DRB keeps the identity the PSI produced, so the two arguments
            // are the same number for every session this simulator sets up.
            assert_eq!(
                task.rlc_entity_for(psi, psi).mode,
                RlcMode::UnacknowledgedMode
            );
        }
    }

    /// #292 (seen on devtest1): the UE is attached to gNB A; A's pod is
    /// replaced by gNB B with the SAME STI (the NCI) at a new address. B's
    /// heartbeat ack must end the A cell (radio link failure to RRC) and map
    /// B as a new cell, so B's PDUs are accepted. Before, the cell was
    /// re-pointed inside the cell search only, the link layer stayed bound to
    /// A, and every PDU from B was dropped as "from unknown source".
    #[tokio::test]
    async fn a_serving_gnb_restarted_at_a_new_address_is_a_new_cell() {
        let (task_base, _app_rx, _nas_rx, mut rrc_rx, _rls_rx) = UeTaskBase::new(test_config(), 16);
        let a: SocketAddr = "10.0.0.1:4997".parse().unwrap();
        let b: SocketAddr = "10.0.0.2:4997".parse().unwrap();
        let mut task = RlsTask::new(
            task_base,
            RlsTaskConfig {
                gnb_search_list: vec![a],
                ..Default::default()
            },
        );
        let ack = nextgsim_rls::RlsHeartbeatAck::with_dbm(0x10, -60);
        task.handle_heartbeat_ack(a, &ack).await;
        task.handle_assign_current_cell(0);
        assert_eq!(task.serving_cell, Some(0));
        while rrc_rx.try_recv().is_ok() {}

        // The name now resolves to B only (what re-resolution installs).
        task.cell_search.set_search_space(vec![b]);
        task.handle_heartbeat_ack(b, &ack).await;

        assert_eq!(task.serving_cell, None, "the A cell is gone");
        assert_eq!(task.cell_addresses.get(&1), Some(&b), "B is a new cell");
        assert!(!task.cell_addresses.values().any(|addr| *addr == a));
        let mut rlf = false;
        while let Ok(msg) = rrc_rx.try_recv() {
            if matches!(
                msg,
                TaskMessage::Message(RrcMessage::RadioLinkFailure { .. })
            ) {
                rlf = true;
            }
        }
        assert!(rlf, "losing the serving cell is a radio link failure");
    }

    #[test]
    fn test_reset_sti() {
        let config = test_config();
        let (task_base, _app_rx, _nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(config, 16);
        let rls_config = RlsTaskConfig {
            gnb_search_list: vec!["127.0.0.1:4997".parse().unwrap()],
            ..Default::default()
        };
        let mut task = RlsTask::new(task_base, rls_config);
        let old_sti = task.sti;
        task.handle_reset_sti();
        assert_ne!(task.sti, old_sti);
        assert!(task.serving_cell.is_none());
        assert_eq!(task.cell_count(), 0);
    }
}
