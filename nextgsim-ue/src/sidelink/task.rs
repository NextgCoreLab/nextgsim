//! Sidelink Task for UE - NR relay, discovery, PC5, and sidelink positioning
//!
//! Runs the **real PC5 procedures** of TS 23.304 (issue #141): a
//! `Direct Communication Request`/`Accept` handshake before a unicast link is usable,
//! Model A/B direct discovery as an actual exchange, UE-to-UE relay forwarding against
//! the live link table, and an RRC `SidelinkUEInformation` resource request answered by
//! an `sl-ConfigDedicatedNR` grant.
//!
//! Increment 1 (issue #54) gated this module behind the off-by-default `sidelink` Cargo
//! feature because the surface was then a facade. The feature stays off by default — PC5
//! is not a capability a default UE should advertise, and the gate is also what keeps
//! this module out of `cargo test --workspace`'s lean build — but what it now gates is
//! working procedures rather than shapes. `UeConfig::prose_enabled` is the runtime half
//! of the gate: with the feature on but the flag off, this task registers and idles.
//!
//! # What drives what, and where the senders are
//!
//! Every [`SidelinkMessage`] variant now has a production sender. That was the substance
//! of issue #141's complaint — `EstablishPc5Link`, `PeerDiscovered`, `SetRelayMode`,
//! `RelayData`, `PositioningMeasurement` and `CooperativePositioning` were unreachable
//! handlers:
//!
//! | Message / handler | Driven by |
//! |---|---|
//! | `StartDiscovery` / `StopDiscovery` | `main.rs`, on registration change |
//! | `SlPrsOccasion` | `main.rs`, on the ranging interval (issue #136) |
//! | `Pc5SReceived` | **the peer UE**, over the PC5 channel — the whole receive path, which did not exist |
//! | `EstablishPc5Link` | **this task**, on discovering a relay serving a Relay Service Code it wants (TS 23.304 §6.3.2 into §6.4.3.1) |
//! | `PeerDiscovered` handling | **this task**, having decoded a peer's discovery message |
//! | `RelayPayload` | **this task**, forwarding onward to the destination |
//! | relay role | `ProseConfig::relay_service_code` at construction, so a configured relay IS one from startup |
//!
//! Two handlers are driven only by an operator or a test, and that is stated rather than
//! dressed up: `ReleasePc5Link` (nothing in this simulator decides to tear a working link
//! down — there is no application above PC5 to finish with one) and
//! `CooperativePositioning` (a request for a position estimate, which only a caller
//! wanting one can originate). `SetRelayMode` likewise, since the configured role now
//! covers the production case. These are *requests from outside*, not internal state
//! transitions, so having no internal sender is correct for them — unlike the six
//! handlers #141 named, which described procedures that should have driven themselves.
//!
//! # The PC5 "radio"
//!
//! There is none. PC5-S messages leave through [`Pc5Transmitter`], a channel the caller
//! supplies — in the two-UE tests, the peer's own `SidelinkMessage` inbox. That is the
//! same arrangement the SL-PRS measurement uses (it models propagation from configured
//! geometry rather than observing it), and it is stated rather than implied because the
//! bytes are real UPER/PC5-S while the medium is not.

use std::collections::HashMap;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::sidelink::discovery::{DiscoveryFilter, DiscoveryModel, Pc5DiscoveryEngine};
use crate::sidelink::link::{Pc5LinkContext, Pc5LinkTable, Pc5UnicastState};
use crate::sidelink::pc5s::{DirectCommunicationRequest, Pc5SMessage, ProseL2Id, RelayServiceCode};
use crate::sidelink::positioning::{
    AnchorUe, Position3D, RttMeasurement as PositioningRttMeasurement, SidelinkPositioningEngine,
    SlPrsResourceConfig,
};
use crate::sidelink::relay::{RelayForwardDecision, RelayForwarder, RelayRole};
use crate::tasks::{RangingMessage, RrcMessage, SidelinkMessage, Task, TaskMessage, UeTaskBase};

/// Speed of light in metres per nanosecond, the constant the RTT-to-distance
/// conversion in `positioning::RttMeasurement` and `ranging::RangingSession` both
/// use. Named once here so the stimulus cannot drift from what consumes it.
const SPEED_OF_LIGHT_M_PER_NS: f64 = 0.3;

/// Speed of light in metres per microsecond, i.e. `c / 1 MHz`. Dividing it by a
/// frequency in MHz gives a wavelength in metres.
const SPEED_OF_LIGHT_M_PER_US: f64 = 300.0;

/// Quality reported for an SL-PRS measurement.
///
/// High, and deliberately so: the range is computed from configured geometry with
/// no noise model, so claiming a middling quality would misdescribe it. It has to
/// exceed the 0.3 floor `resolve_carrier_phase_ambiguity` applies, or the
/// carrier-phase estimate is discarded and the pipeline this stimulus exists to
/// reach stops at the RTT.
const SL_PRS_MEASUREMENT_QUALITY: f64 = 0.95;

/// The PC5 QoS Flow Identifier a responder settles a unicast link on
/// (TS 23.304 §5.6.1, §6.4.3.1 step 5's `QoS Info`).
///
/// One value, because this simulator has one PC5 QoS flow: nothing here maps a ProSe
/// service to a PQI, and allocating PFIs from a pool would imply a per-flow treatment no
/// code applies. 1 rather than 0 so that "the PFI the link negotiated" and "the field
/// nobody set" are distinguishable.
const PC5_DEFAULT_PFI: u8 = 1;

/// The remote UE's end-to-end Uu bearer this relay adapts (issue #190).
///
/// SRB1, which TS 38.331 §9.2.5 names as the default SRAP bearer. It matches the bearer the
/// gNB puts in its `sl-SRAP-ConfigRelay` (`nextgsim_gnb::rrc::sidelink::DEFAULT_SRAP_BEARER`),
/// and the two must agree: a relay adapting DRB1 while the gNB mapped SRB1 would produce a
/// header naming a bearer the gNB has no PDCP entity for.
const SRAP_RELAYED_BEARER: nextgsim_rlc::srap::RemoteBearerId =
    nextgsim_rlc::srap::RemoteBearerId::Srb(nextgsim_rlc::srap::DEFAULT_SRAP_SRB);

/// The relay RLC channel index this UE's SRAP mappings use.
///
/// 1, matching the single channel the gNB grants in `sl-RLC-BearerToAddModList`
/// (`nextgsim_gnb::rrc::sidelink::FIRST_UU_RELAY_RLC_CHANNEL`). One channel, because
/// TS 38.300 §16.12.2.1 permits many end-to-end bearers to multiplex onto one and nothing
/// here differentiates their treatment.
const SRAP_EGRESS_CHANNEL: u8 = 1;

/// Milliseconds since the UNIX epoch, the clock the discovery peer table and the task's
/// peer table both timestamp against.
///
/// Saturates to 0 if the system clock is before the epoch, which the pruning path treats
/// as "never expires" rather than "everything is stale" -- see
/// `Pc5DiscoveryEngine::prune_stale_peers`.
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// The line the UE binary emits when it spawns the sidelink task.
///
/// A constant rather than a literal at the `info!` site because the site is in
/// `main.rs`, inside a task closure no test can reach. Naming it here is what
/// lets `the_startup_lines_advertise_no_active_sidelink_capability` pin the
/// binary's wording: the only way to change what the binary logs is to change
/// this string, and the test reads this string.
///
/// Since issue #141 it names what is real and what is not. Issue #190 made "no
/// UE-to-Network relay" false — the TS 38.351 SRAP adaptation layer, the gNB-signalled
/// bearer mapping and the remote-UE identity all run now — so that clause is **retired**,
/// as #190's criterion 5 requires. Issue #54's contract is unchanged: the lines must not
/// advertise a capability the code does not deliver. What changed is which capabilities
/// the code delivers.
///
/// What remains absent, and is still named: there is no PC5 radio, and no L3 relay.
pub const SPAWN_LOG: &str = "Sidelink task spawned (feature-gated: real PC5-S \
     Direct Communication and Model A/B discovery over an in-process PC5 channel, no radio)";

/// The line [`SidelinkTask::run`] emits on entry. Same no-overclaim contract as
/// [`SPAWN_LOG`], and unlike it this one is reached by a test.
pub const START_LOG: &str = "Sidelink task started (PC5-S unicast link establishment, \
     Model A/B discovery, UE-to-UE relay forwarding, SidelinkUEInformation and L2 \
     UE-to-Network relay with the TS 38.351 SRAP adaptation layer are live; \
     no PC5 radio, no L3 relay)";

/// PC5 link state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pc5LinkState {
    /// No link
    Idle,
    /// Discovery in progress
    Discovering,
    /// Link being established
    Establishing,
    /// Link active
    Active,
    /// Link releasing
    Releasing,
}

/// Sidelink peer information.
#[derive(Debug, Clone)]
struct SidelinkPeer {
    /// Peer UE identifier
    _peer_ue_id: u64,
    /// PC5 link state
    link_state: Pc5LinkState,
    /// Whether this peer is acting as relay
    _is_relay: bool,
    /// Signal quality (dBm)
    signal_dbm: i32,
    /// Last discovery timestamp (ms)
    last_discovery_ms: u64,
    /// Sidelink positioning: distance estimate (meters)
    sl_distance_m: Option<f64>,
    /// Sidelink positioning: position estimate (x, y, z)
    sl_position: Option<(f64, f64, f64)>,
}

impl SidelinkPeer {
    fn new(peer_ue_id: u64) -> Self {
        Self {
            _peer_ue_id: peer_ue_id,
            link_state: Pc5LinkState::Idle,
            _is_relay: false,
            signal_dbm: -120,
            last_discovery_ms: 0,
            sl_distance_m: None,
            sl_position: None,
        }
    }
}

/// Relay mode configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelayMode {
    /// Not a relay
    None,
    /// Layer-2 UE-to-UE relay
    L2Relay,
    /// Layer-3 UE-to-UE relay
    L3Relay,
    /// UE-to-Network relay
    UeToNetworkRelay,
}

/// Where a UE's PC5-S messages go (issue #141).
///
/// A `SidelinkMessage` sender rather than a byte sink, so the receiving end is the peer's
/// own real message loop: the bytes go in as `Pc5SReceived` and are decoded by
/// [`SidelinkTask::handle_pc5s_received`], the same path a radio would feed. A raw byte
/// channel would have let a test bypass the receive handler, which is where the
/// procedure lives.
pub type Pc5Transmitter = crate::tasks::TaskHandle<SidelinkMessage>;

pub struct SidelinkTask {
    task_base: UeTaskBase,
    /// Known sidelink peers
    peers: HashMap<u64, SidelinkPeer>,
    /// This UE's relay mode.
    ///
    /// Kept beside [`Self::relay`] rather than replaced by it: `RelayMode` distinguishes
    /// L2/L3/UE-to-Network, which `SetRelayMode` still carries, while `RelayForwarder`
    /// owns the forwarding decision and its counters. The former is what was configured;
    /// the latter is what happens.
    relay_mode: RelayMode,
    /// PC5 direct discovery, Model A and B (TS 23.304 §6.3.1.2, §6.3.1.3).
    ///
    /// Replaces the `discovery_active: bool` this task used to carry, which was set and
    /// read by nothing else: no message was announced and no peer was ever discovered.
    discovery: Pc5DiscoveryEngine,
    /// The PC5 unicast links this UE holds (TS 23.304 §6.4.3.1).
    links: Pc5LinkTable,
    /// UE-to-UE relay forwarding (TS 23.304 §6.4.3.10).
    relay: RelayForwarder,
    /// The SRAP adaptation sublayer, for L2 UE-to-Network relay
    /// (TS 38.351, TS 38.300 §16.12.2.1; issue #190).
    ///
    /// Configured from the wire, by `SidelinkMessage::SrapMappingConfigured` — which the
    /// RRC task sends on decoding the gNB's `sl-L2RelayUE-Config`. So a mapping exists here
    /// only because the network signalled one, which is what makes
    /// [`SidelinkTask::srap_adapted_payload_octets`] a meaningful observable rather than a
    /// locally-invented value.
    srap: nextgsim_rlc::srap::SrapEntity,
    /// Which local Remote UE ID this UE holds for each remote UE's PC5 Layer-2 ID.
    ///
    /// The translation between the two identity spaces TS 38.300 §16.12.2.1 keeps separate:
    /// the SRAP entity is keyed by local ID (what a header carries), while the forwarding
    /// path knows a peer by its PC5 Layer-2 ID.
    srap_local_ids: HashMap<u32, u8>,
    /// Where PC5-S messages this UE sends go.
    ///
    /// `None` means this UE has no PC5 peer wired up, which is the production case today:
    /// `main.rs` spawns one UE per process, so there is no second UE in it to reach. The
    /// two-UE tests supply each other's inboxes. A message that cannot be sent is logged
    /// rather than dropped silently -- see [`SidelinkTask::transmit`].
    pc5_tx: Option<Pc5Transmitter>,
    /// Whether ProSe is enabled at run time (`UeConfig::prose_enabled`, issue #141).
    ///
    /// Cached at construction so every handler reads one bool rather than walking the
    /// config. With it false the task registers and idles: no announcement, no link, no
    /// resource request.
    prose_enabled: bool,
    /// The ProSe parameters in force, defaulted when the config named none.
    prose: nextgsim_common::config::ProseConfig,
    /// SL-PRS resources and the measurements taken on them (TS 23.586, issue
    /// #136). Built from `RangingConfig` at construction: one resource per
    /// configured anchor, offset by a slot each so two anchors do not nominally
    /// transmit in the same one.
    positioning: SidelinkPositioningEngine,
    /// Anchors this UE has already opened a ranging session for, so the session
    /// is started once rather than on every occasion.
    ranging_sessions_started: std::collections::HashSet<u64>,
}

impl SidelinkTask {
    pub fn new(task_base: UeTaskBase) -> Self {
        let mut positioning = SidelinkPositioningEngine::new();
        if let Some(ranging) = task_base.config.ranging_config.as_ref() {
            for (index, anchor) in ranging.anchors.iter().enumerate() {
                // `resource_id` is the anchor's index, not its UE id: the id is a
                // u16 and a sidelink UE id is a u64, so using the latter would
                // truncate two anchors onto one resource.
                let resource_id = u16::try_from(index).unwrap_or(u16::MAX);
                let slot_offset = u16::try_from(index).unwrap_or(u16::MAX);
                positioning.add_sl_prs_resource(
                    SlPrsResourceConfig::new(resource_id).with_offset(slot_offset),
                );
                positioning.add_anchor(AnchorUe::new(
                    anchor.ue_id,
                    Position3D::new(anchor.position[0], anchor.position[1], anchor.position[2]),
                    // No distance is known before the first occasion measures one.
                    0.0,
                    0.0,
                ));
            }
        }
        // The runtime half of the gate (issue #141): the Cargo feature decides whether
        // this code exists, `prose_enabled` decides whether it runs. Read here rather
        // than at each handler so there is one place the flag is consulted.
        let prose_enabled = task_base.config.prose_enabled;
        let prose = task_base.config.prose_config.clone().unwrap_or_default();

        let local_l2_id = ProseL2Id::new(prose.local_l2_id);
        let mut discovery = Pc5DiscoveryEngine::new(local_l2_id);
        // The UE monitors for exactly the application code it announces: two UEs on
        // different codes should NOT discover each other, and this is what makes that
        // true rather than a comment (TS 23.304 §5.8.1).
        discovery.add_filter(DiscoveryFilter::exact(prose.prose_app_code));
        if let Some(rsc) = prose.relay_service_code {
            discovery.serve_relay_service_code(RelayServiceCode::new(rsc));
        }

        let mut relay = RelayForwarder::new(local_l2_id);
        // A UE configured with a relay service code IS a relay, from startup. This is
        // one of the two production senders `SetRelayMode` never had: the other is the
        // message itself, which an operator or a test may still send to change it.
        //
        // WHICH kind of relay is `ProseConfig::ue_to_network_relay` (issue #190). That is
        // the production path to `RelayRole::L2UeToNetwork`: without it the variant would be
        // reachable only by an operator sending `SetRelayMode`, which is the
        // "correct but unreachable" defect this tree keeps producing.
        if prose.relay_service_code.is_some() {
            relay.set_role(if prose.ue_to_network_relay {
                RelayRole::L2UeToNetwork
            } else {
                RelayRole::L2UeToUe
            });
        }

        Self {
            task_base,
            peers: HashMap::new(),
            relay_mode: match (prose.relay_service_code, prose.ue_to_network_relay) {
                // Issue #190: a configured relay is UE-to-Network when the config says so.
                (Some(_), true) => RelayMode::UeToNetworkRelay,
                (Some(_), false) => RelayMode::L2Relay,
                (None, _) => RelayMode::None,
            },
            discovery,
            links: Pc5LinkTable::new(),
            relay,
            srap: nextgsim_rlc::srap::SrapEntity::new(),
            srap_local_ids: HashMap::new(),
            pc5_tx: None,
            prose_enabled,
            prose,
            positioning,
            ranging_sessions_started: std::collections::HashSet::new(),
        }
    }

    /// Wires this UE's PC5 transmit path to a peer (issue #141).
    ///
    /// Supplied by the caller because there is no PC5 radio: the "medium" is whatever
    /// channel the caller has. In the two-UE tests each UE is given the other's
    /// `SidelinkMessage` inbox, so a `Direct Communication Request` this UE encodes is
    /// decoded by the peer's real receive path.
    pub fn set_pc5_transmitter(&mut self, tx: Pc5Transmitter) {
        self.pc5_tx = Some(tx);
    }

    /// This UE's ProSe Layer-2 ID (TS 23.304 §5.8.2.1).
    pub fn local_l2_id(&self) -> ProseL2Id {
        self.discovery.local_l2_id()
    }

    /// The state of this UE's PC5 unicast link to `peer`, if it has one.
    ///
    /// The observable the handshake tests assert on: `Active` here is reachable only by
    /// having decoded a peer's `DIRECT COMMUNICATION ACCEPT`.
    pub fn link_state(&self, peer: ProseL2Id) -> Option<Pc5UnicastState> {
        self.links.get(peer).map(Pc5LinkContext::state)
    }

    /// The peer Layer-2 ID this UE learned from the link to `peer`
    /// (TS 23.304 §6.4.3.1 step 4).
    pub fn link_peer_l2_id(&self, peer: ProseL2Id) -> Option<ProseL2Id> {
        self.links.get(peer).and_then(Pc5LinkContext::peer_l2_id)
    }

    /// How many PC5 unicast links are usable.
    pub fn active_link_count(&self) -> usize {
        self.links.active_count()
    }

    /// How many peers discovery has found.
    pub fn discovered_peer_count(&self) -> usize {
        self.discovery.peer_count()
    }

    /// Octets this UE has forwarded as a relay on behalf of other UEs.
    pub fn relay_forwarded_octets(&self) -> u64 {
        self.relay.forwarded_octets()
    }

    /// Whether discovery is running.
    pub fn discovery_active(&self) -> bool {
        self.discovery.is_active()
    }

    /// Sends a PC5-S message to this UE's PC5 peer.
    ///
    /// A message with no transmitter wired is logged at `debug` rather than dropped
    /// silently: that is the production case (one UE per process, no peer in it), and a
    /// warning per announcement would be noise. It is not silent, because "PC5 looked
    /// like it was working and nothing left the UE" is the defect class this issue is
    /// about.
    async fn transmit(&self, pdu: Vec<u8>) {
        let Some(ref tx) = self.pc5_tx else {
            debug!(
                "Sidelink: no PC5 peer wired; {} octet(s) of PC5-S not transmitted",
                pdu.len()
            );
            return;
        };
        let _ = tx.send(SidelinkMessage::Pc5SReceived { pdu }).await;
    }

    /// **Starts direct discovery** (TS 23.304 §6.3.1.2 step 3a, §6.3.1.3).
    ///
    /// A real announcement or solicitation leaves the UE, where the old handler set a
    /// bool. With `prose_enabled` false nothing is sent — that is the runtime gate.
    async fn handle_start_discovery(&mut self) {
        if !self.prose_enabled {
            debug!(
                "Sidelink: StartDiscovery ignored, prose_enabled is false; PC5 is \
                 compiled in but not enabled"
            );
            return;
        }
        let model = match self.prose.discovery_model.to_ascii_lowercase().as_str() {
            "b" => DiscoveryModel::ModelB,
            // Model A for anything else, including a typo: it is the unsolicited case, so
            // a mis-set model still makes the UE discoverable rather than silent.
            _ => DiscoveryModel::ModelA,
        };
        let msg = self.discovery.start(self.prose.prose_app_code, model);
        debug!(
            "Sidelink: Discovery started ({:?}, app code 0x{:08X})",
            model, self.prose.prose_app_code
        );
        self.transmit(msg.encode()).await;
    }

    /// **Applies a received PC5-S message** (TS 24.554), the receive half of every PC5
    /// procedure.
    ///
    /// Public so the two-UE integration test drives the real handler: a test that called
    /// the link state machine directly would pass whether or not this dispatch existed,
    /// and this dispatch is what makes the procedures reachable.
    pub async fn handle_pc5s_received(&mut self, pdu: &[u8]) {
        if !self.prose_enabled {
            debug!("Sidelink: PC5-S message ignored, prose_enabled is false");
            return;
        }
        let msg = match Pc5SMessage::decode(pdu) {
            Ok(msg) => msg,
            Err(e) => {
                // A peer using a procedure this UE has not implemented, or a corrupt
                // PDU. Warned and dropped: there is no PC5-S error message to answer
                // with that this UE implements, and guessing the intent of bytes it
                // cannot parse is worse than ignoring them.
                warn!(
                    "Sidelink: undecodable PC5-S message ({} octets): {e}",
                    pdu.len()
                );
                return;
            }
        };

        match msg {
            Pc5SMessage::Discovery(discovery) => {
                let now_ms = now_ms();
                let outcome = self.discovery.on_message_received(&discovery, now_ms);
                if let Some(peer) = outcome.peer {
                    // `PeerDiscovered`'s production sender, which it had none of: this
                    // UE, having decoded a peer's announcement.
                    self.record_discovered_peer(peer.l2_id, now_ms);
                    debug!(
                        "Sidelink: discovered peer {} (app code 0x{:08X}, relay={:?})",
                        peer.l2_id, peer.prose_app_code, peer.relay_service_code
                    );
                    // A change of interest triggers the RRC resource request
                    // (TS 38.331 §5.8.3.2).
                    self.request_sidelink_resources().await;

                    // **`EstablishPc5Link`'s production sender** (TS 23.304 §6.3.2 into
                    // §6.4.3.1): a remote UE that discovers a relay serving the Relay
                    // Service Code it wants opens a unicast link to it, unprompted. That
                    // is what relay discovery is FOR -- discovering a relay and then
                    // waiting to be told to use it would leave the remote UE with no
                    // connectivity and no reason for the discovery to have happened.
                    //
                    // Only for a relay, and only when this UE is not itself one: two
                    // relays that each opened a link to the other on sight would
                    // establish a link neither has traffic for. And only when no link
                    // exists, which `handle_establish_pc5_link` re-checks.
                    let wants_relay = self.prose.relay_service_code.is_none();
                    if wants_relay && peer.relay_service_code.is_some() {
                        debug!(
                            "Sidelink: {} serves relay service code {:?}; opening a PC5 \
                             unicast link to it",
                            peer.l2_id, peer.relay_service_code
                        );
                        self.handle_establish_pc5_link(u64::from(peer.l2_id.value()))
                            .await;
                    }
                }
                if let Some(response) = outcome.response {
                    // Model B: answer the solicitation (§6.3.1.3).
                    self.transmit(response.encode()).await;
                }
            }
            Pc5SMessage::Request(request) => {
                self.handle_direct_communication_request(&request).await;
            }
            Pc5SMessage::Accept(accept) => {
                // Routed by the accept's own source: that is the peer whose link this is.
                let peer = accept.source_l2_id;
                let became_active = match self.links.get_mut(peer) {
                    Some(link) => match link.on_accept_received(&accept) {
                        Ok(()) => {
                            self.mark_peer_active(peer);
                            debug!("Sidelink: PC5 link to {peer} is active");
                            true
                        }
                        Err(e) => {
                            warn!("Sidelink: {e}");
                            false
                        }
                    },
                    None => {
                        warn!(
                            "Sidelink: DIRECT COMMUNICATION ACCEPT from {peer}, which this \
                             UE has no pending link with"
                        );
                        false
                    }
                };
                if became_active {
                    // A link going ACTIVE is a change of interest (TS 38.331 §5.8.3.2), and
                    // since issue #190 it can also change this UE's declared `ue-Type-r17`:
                    // a UE that has just completed a link to a relay is now a **remote UE**,
                    // and must say so or the gNB never assigns it the local Remote UE ID it
                    // needs for its own SRAP headers (TS 38.300 §16.12.2.1).
                    //
                    // The initiator's request at `handle_establish_pc5_link` cannot carry
                    // that: the link is only `AwaitingAccept` there, so the UE is not yet
                    // reachable through the relay and declaring it would be premature.
                    self.request_sidelink_resources().await;
                }
            }
            Pc5SMessage::Reject(reject) => {
                let peer = reject.source_l2_id;
                match self.links.get_mut(peer) {
                    Some(link) => {
                        if let Err(e) = link.on_reject_received(&reject) {
                            warn!("Sidelink: {e}");
                        }
                    }
                    None => warn!(
                        "Sidelink: DIRECT COMMUNICATION REJECT from {peer}, which this \
                         UE has no pending link with"
                    ),
                }
            }
            Pc5SMessage::Release(release) => {
                let peer = release.source_l2_id;
                match self.links.get_mut(peer) {
                    Some(link) => match link.on_release_received(&release) {
                        Ok(()) => {
                            self.mark_peer_idle(peer);
                            debug!("Sidelink: PC5 link to {peer} released by the peer");
                        }
                        Err(e) => warn!("Sidelink: {e}"),
                    },
                    None => debug!(
                        "Sidelink: DIRECT COMMUNICATION RELEASE from {peer}, which this \
                         UE holds no link with; nothing to release"
                    ),
                }
            }
        }
    }

    /// **Responder, TS 23.304 §6.4.3.1 steps 4-5**: decide on a peer's request and answer
    /// it.
    async fn handle_direct_communication_request(&mut self, request: &DirectCommunicationRequest) {
        let peer = request.source_l2_id;
        let mut link = Pc5LinkContext::new_responder(self.local_l2_id());
        let announced_code = self.prose.prose_app_code;
        // Step 5b's interest test: this UE is interested in a service-oriented request
        // whose ProSe Service Info names the code it announces. An empty Service Info is
        // treated as "any", because a request that names no service cannot be filtered
        // on one.
        let interested = |info: &[u8]| info.is_empty() || info == announced_code.to_be_bytes();
        let served = self.discovery.served_relay_service_code();

        match link.on_request_received(request, PC5_DEFAULT_PFI, served, interested) {
            Ok(accept) => {
                self.links.insert(peer, link);
                self.record_discovered_peer(peer, now_ms());
                self.mark_peer_active(peer);
                debug!(
                    "Sidelink: accepted a PC5 unicast link from {peer} (relay={})",
                    accept.acting_as_relay
                );
                self.transmit(accept.encode()).await;
                // A new destination is a change of interest (TS 38.331 §5.8.3.2).
                self.request_sidelink_resources().await;
            }
            Err(reject) => {
                debug!(
                    "Sidelink: refusing a PC5 link from {peer}: {:?}",
                    reject.cause
                );
                self.transmit(reject.encode()).await;
            }
        }
    }

    /// **Initiator, TS 23.304 §6.4.3.1 step 3**: open a PC5 unicast link to a peer.
    ///
    /// Sends a `DIRECT COMMUNICATION REQUEST` and waits. The link does **not** become
    /// active here — that is the correction this issue makes, and
    /// `an_unsolicited_accept_does_not_activate_a_link` is the guard on it.
    ///
    /// A peer that discovery has not found is refused: TS 23.304 §6.4.3.1 step 2 has the
    /// application supply a target it knows of, and requesting a link to a Layer-2 ID
    /// nothing has been heard from would address a UE that may not exist.
    async fn handle_establish_pc5_link(&mut self, peer_ue_id: u64) {
        if !self.prose_enabled {
            debug!("Sidelink: EstablishPc5Link ignored, prose_enabled is false");
            return;
        }
        let peer = ProseL2Id::new(peer_ue_id as u32);
        if self.discovery.peer(peer).is_none() {
            warn!("Sidelink: Cannot establish PC5 link, peer {peer} not discovered");
            return;
        }
        if self
            .links
            .get(peer)
            .is_some_and(|l| l.state() == Pc5UnicastState::Active)
        {
            debug!("Sidelink: already hold an active PC5 link to {peer}");
            return;
        }

        // If the peer announced a relay service code this UE wants, ask for it: that is
        // what turns a plain unicast link into a relay link.
        let rsc = self.discovery.peer(peer).and_then(|p| p.relay_service_code);
        let mut link = Pc5LinkContext::new_initiator(self.local_l2_id());
        let request = link.build_request(
            Some(peer),
            rsc,
            // The ProSe Service Info is this UE's announced application code, which is
            // what the responder's step-5b interest test matches against.
            self.prose.prose_app_code.to_be_bytes().to_vec(),
            // Security Information: empty, because TS 33.503's security establishment is
            // not implemented. Deliberately empty rather than filled with plausible
            // bytes -- see `pc5s`'s module docs.
            Vec::new(),
        );
        self.links.insert(peer, link);
        self.mark_peer_establishing(peer);
        self.transmit(request.encode()).await;
        // Requesting resources for a destination this UE is about to transmit to
        // (TS 38.331 §5.8.3.2).
        self.request_sidelink_resources().await;
    }

    /// Releases a PC5 unicast link, telling the peer (TS 24.554 §6.1.2.4).
    async fn handle_release_pc5_link(&mut self, peer_ue_id: u64) {
        let peer = ProseL2Id::new(peer_ue_id as u32);
        let Some(link) = self.links.get_mut(peer) else {
            debug!("Sidelink: no PC5 link to {peer} to release");
            return;
        };
        let release = link.build_release();
        self.mark_peer_idle(peer);
        debug!("Sidelink: PC5 link released with peer {peer}");
        self.transmit(release.encode()).await;
        // One fewer destination is also a change of interest (TS 38.331 §5.8.3.2).
        self.request_sidelink_resources().await;
    }

    /// Forwards a relayed payload, and passes it on to its destination when it can
    /// (TS 23.304 §6.4.3.10).
    async fn handle_relay_payload(
        &mut self,
        source: ProseL2Id,
        destination: ProseL2Id,
        payload: Vec<u8>,
    ) {
        match self.forward_relay_payload(source, destination, &payload) {
            RelayForwardDecision::Forward {
                next_hop,
                payload_len,
            } => {
                debug!("Sidelink: relaying {payload_len} octet(s) from {source} to {next_hop}");
                // Onward over PC5, preserving the ORIGINAL source: a relay is not the
                // originator, and rewriting the source would make the destination answer
                // the relay instead of the peer it is talking to.
                if let Some(ref tx) = self.pc5_tx {
                    let _ = tx
                        .send(SidelinkMessage::RelayPayload {
                            source_l2_id: source.value(),
                            destination_l2_id: destination.value(),
                            payload,
                        })
                        .await;
                }
            }
            RelayForwardDecision::AdaptToNetwork {
                remote_l2_id,
                payload_len,
            } => {
                // **The L2 UE-to-Network uplink** (TS 38.300 §16.12.2.1; issue #190). The
                // payload is a remote UE's end-to-end PDCP PDU, and it leaves on one of this
                // relay's own Uu relay RLC channels rather than on a PC5 link — so it is
                // adapted through SRAP, not forwarded to a peer.
                self.adapt_to_network(remote_l2_id, payload_len, payload)
                    .await;
            }
            RelayForwardDecision::DeliverLocally => {
                debug!(
                    "Sidelink: {} octet(s) from {source} are for this UE",
                    payload.len()
                );
            }
            other => {
                debug!("Sidelink: not relaying {source} -> {destination}: {other:?}");
            }
        }
    }

    /// Adapts a remote UE's payload through SRAP and submits it towards the network
    /// (TS 38.351; TS 38.300 §16.12.2.1; issue #190).
    ///
    /// The SRAP header carries the local Remote UE ID the **gNB** assigned, looked up from
    /// the mapping the network signalled — not invented here. A remote UE with no signalled
    /// mapping is refused: §16.12.2.1 makes the local Remote UE ID the key the gNB
    /// correlates the traffic by, so a header with a made-up one would name a remote UE the
    /// gNB has assigned no identity to.
    async fn adapt_to_network(
        &mut self,
        remote_l2_id: ProseL2Id,
        payload_len: usize,
        payload: Vec<u8>,
    ) {
        use nextgsim_rlc::srap::SrapDecision;

        let Some(local_id) = self.srap_local_ids.get(&remote_l2_id.value()).copied() else {
            warn!(
                "Sidelink: no SRAP mapping signalled for remote UE {remote_l2_id}; \
                 dropping {payload_len} octet(s) rather than inventing a local Remote UE ID"
            );
            return;
        };

        match self.srap.adapt(local_id, SRAP_RELAYED_BEARER, &payload) {
            SrapDecision::Submit { egress, pdu } => {
                debug!(
                    "Sidelink: SRAP adapted {payload_len} octet(s) from remote UE \
                     {remote_l2_id} (local ID {local_id}) onto {egress:?}; {} octet(s) on \
                     the relay RLC channel",
                    pdu.len()
                );
                // The SRAP PDU goes up this relay's OWN Uu connection, which is the RRC
                // task's channel -- the relay's Uu leg, not a PC5 link. `RelayedUplink`
                // carries it there.
                let _ = self
                    .task_base
                    .rrc_tx
                    .send(RrcMessage::RelayedUplink {
                        remote_l2_id: remote_l2_id.value(),
                        pdu,
                    })
                    .await;
            }
            other => {
                warn!(
                    "Sidelink: SRAP refused {payload_len} octet(s) from remote UE \
                     {remote_l2_id}: {other:?}"
                );
            }
        }
    }

    /// Applies a SRAP bearer mapping the network signalled (TS 38.331 §5.3.5.17;
    /// issue #190).
    ///
    /// The relay's egress is a **Uu** relay RLC channel (towards the gNB) and the remote
    /// UE's is a **PC5** one (towards the relay), which is why `is_relay` decides the hop:
    /// applying one role's mapping to the other would send the traffic the wrong way.
    fn handle_srap_mapping_configured(
        &mut self,
        remote_l2_id: u32,
        local_remote_ue_id: u8,
        is_relay: bool,
    ) {
        use nextgsim_rlc::srap::{EgressChannel, RemoteUeMapping};

        let mut mapping = RemoteUeMapping::new(local_remote_ue_id);
        mapping.map_bearer(
            SRAP_RELAYED_BEARER,
            if is_relay {
                EgressChannel::Uu(SRAP_EGRESS_CHANNEL)
            } else {
                EgressChannel::Pc5(u16::from(SRAP_EGRESS_CHANNEL))
            },
        );
        self.srap.configure_remote_ue(mapping);
        self.srap_local_ids.insert(remote_l2_id, local_remote_ue_id);
        debug!(
            "Sidelink: SRAP mapping configured for remote UE {remote_l2_id:#08x}: local \
             Remote UE ID {local_remote_ue_id}, {} egress",
            if is_relay { "Uu" } else { "PC5" }
        );
    }

    /// End-to-end payload octets this UE's SRAP entity has carried, headers excluded.
    ///
    /// The positive observable for the UE-to-Network path: it is non-zero only because
    /// `SrapEntity::adapt` built a header, which requires a mapping the network signalled.
    pub fn srap_adapted_payload_octets(&self) -> u64 {
        self.srap.submitted_payload_octets()
    }

    /// How many SRAP PDUs this UE has submitted to a relay RLC channel.
    pub fn srap_submitted_pdus(&self) -> u64 {
        self.srap.submitted_pdus()
    }

    /// The local Remote UE ID this UE holds for `remote_l2_id`, if the network signalled
    /// one.
    pub fn srap_local_id_for(&self, remote_l2_id: u32) -> Option<u8> {
        self.srap_local_ids.get(&remote_l2_id).copied()
    }

    /// The relay forwarding decision, against the live link table.
    ///
    /// Split out so the payload and size-only relay messages take the same decision
    /// rather than two that could drift.
    fn forward_relay_payload(
        &mut self,
        source: ProseL2Id,
        destination: ProseL2Id,
        payload: &[u8],
    ) -> RelayForwardDecision {
        self.relay
            .forward(&self.links, source, destination, payload)
    }

    /// Asks the network for sidelink resources for this UE's current destinations
    /// (TS 38.331 §5.8.3.2; issue #141).
    ///
    /// The **production sender of the RRC request**. The destination set is derived from
    /// the live link table, so the request describes what PC5 is actually doing rather
    /// than a configured guess — and a UE with no links asks for receive resources only,
    /// which is what §5.8.3.3 means by an interest in receiving.
    ///
    /// The RRC task suppresses an unchanged request, so calling this on every change is
    /// safe: §5.8.3.2 triggers on change, and "changed" is decided once, there.
    async fn request_sidelink_resources(&self) {
        if !self.prose_enabled {
            return;
        }
        // Every peer this UE holds or is opening a link to. Pending links count: the
        // point of asking is to have resources by the time the link is up.
        let mut destinations: Vec<(u32, bool)> = self
            .discovery
            .peers()
            .into_iter()
            .filter(|p| self.links.get(p.l2_id).is_some())
            .map(|p| (p.l2_id.value(), true))
            .collect();
        destinations.sort_unstable();
        destinations.dedup();

        // Straight to the RRC task's own handle on `UeTaskBase`, not through
        // `rel18`: the request is an ordinary UL-DCCH message on SRB1, so it belongs to
        // the RRC task every task already has a handle to, not to the Rel-18 set.
        let _ = self
            .task_base
            .rrc_tx
            .send(RrcMessage::SidelinkInterestChanged {
                rx_interested_freqs: self.prose.rx_interested_freqs.clone(),
                tx_destinations: destinations,
                // **The production trigger for the whole L2 UE-to-Network relay
                // configuration** (issue #190). Derived from the configured relay mode
                // rather than from a separate flag: a UE the operator put into
                // `UeToNetworkRelay` mode IS asking the network for relay resources, and
                // deciding it here means there is exactly one place the role comes from.
                //
                // `RelayMode::L2Relay` (UE-to-UE) declares NOTHING, because UE-to-UE relay
                // is a PC5-only procedure the network is not in the path of — declaring it
                // to the gNB would ask for a Uu bearer mapping no traffic will use.
                relay_role: self.declared_relay_role(),
            })
            .await;
    }

    /// The L2 UE-to-Network relay role this UE declares to the network
    /// (`ue-Type-r17`, TS 38.331 §6.2.2; issue #190).
    ///
    /// Three cases, and the distinction is the substance:
    ///
    /// * a UE in [`RelayMode::UeToNetworkRelay`] declares **`Relay`** — it is offering to
    ///   carry other UEs' traffic to the 5GC;
    /// * a UE that has *discovered* a relay and holds a link to it declares **`Remote`** —
    ///   it is asking to be reached through one, which is what makes the gNB assign it a
    ///   local Remote UE ID (TS 38.300 §16.12.2.1);
    /// * everything else declares nothing, including UE-to-UE relay: see the call site.
    fn declared_relay_role(&self) -> Option<crate::tasks::SidelinkRelayRole> {
        use crate::tasks::SidelinkRelayRole;
        match self.relay_mode {
            RelayMode::UeToNetworkRelay => Some(SidelinkRelayRole::Relay),
            // A UE that is not a relay but holds a link to one is a remote UE. Decided from
            // the live link table rather than from configuration, so that a UE only claims
            // to be reachable through a relay once it actually has a link to one.
            RelayMode::None if self.holds_link_to_a_relay() => Some(SidelinkRelayRole::Remote),
            _ => None,
        }
    }

    /// Whether this UE holds an active PC5 link to a peer that serves a Relay Service Code.
    ///
    /// The link table and the discovery table together: a peer is a relay because it
    /// announced an RSC (§6.3.2), and it is usable because the PC5-S handshake completed.
    /// Either alone would be wrong — an announced relay with no link cannot carry anything,
    /// and a link to a non-relay peer is an ordinary unicast link.
    fn holds_link_to_a_relay(&self) -> bool {
        self.discovery
            .peers()
            .into_iter()
            .filter(|peer| peer.relay_service_code.is_some())
            .any(|peer| {
                self.links
                    .get(peer.l2_id)
                    .is_some_and(|link| link.state() == Pc5UnicastState::Active)
            })
    }

    /// Records a peer in the task's peer table, which the positioning path reads.
    fn record_discovered_peer(&mut self, peer: ProseL2Id, now_ms: u64) {
        let entry = self
            .peers
            .entry(u64::from(peer.value()))
            .or_insert_with(|| SidelinkPeer::new(u64::from(peer.value())));
        entry.last_discovery_ms = now_ms;
        if entry.link_state == Pc5LinkState::Idle {
            entry.link_state = Pc5LinkState::Discovering;
        }
    }

    fn mark_peer_establishing(&mut self, peer: ProseL2Id) {
        if let Some(entry) = self.peers.get_mut(&u64::from(peer.value())) {
            entry.link_state = Pc5LinkState::Establishing;
        }
    }

    fn mark_peer_active(&mut self, peer: ProseL2Id) {
        let entry = self
            .peers
            .entry(u64::from(peer.value()))
            .or_insert_with(|| SidelinkPeer::new(u64::from(peer.value())));
        entry.link_state = Pc5LinkState::Active;
    }

    fn mark_peer_idle(&mut self, peer: ProseL2Id) {
        if let Some(entry) = self.peers.get_mut(&u64::from(peer.value())) {
            entry.link_state = Pc5LinkState::Idle;
        }
    }

    /// Number of SL-PRS resources configured, for tests and for the startup log.
    pub fn sl_prs_resource_count(&self) -> usize {
        self.positioning.sl_prs_resources.len()
    }

    /// The RTT measurement this UE holds for `peer_ue_id`, if any.
    pub fn rtt_distance_m(&self, peer_ue_id: u64) -> Option<f64> {
        self.positioning
            .rtt_measurements
            .get(&peer_ue_id)
            .map(|m| m.distance_m)
    }

    /// Measure every configured anchor on an SL-PRS occasion and report the
    /// results to the ranging task (TS 23.586 §5.3.3, issue #136).
    ///
    /// The propagation delay is MODELLED from the configured geometry, not
    /// observed: there is no PC5 radio here. What that buys is that the RTT and
    /// carrier-phase maths in `RangingSession` run on live input, and that a test
    /// can assert the distance they recover rather than a log line.
    ///
    /// The carrier phase reported per frequency is the fractional part of
    /// `range / wavelength`, which is what a phase measurement actually gives:
    /// ambiguous modulo one wavelength. Resolving that ambiguity against the RTT
    /// estimate is the widelane combination the ranging task already implements,
    /// so reporting an unambiguous phase would bypass the very code this stimulus
    /// exists to reach.
    async fn handle_sl_prs_occasion(&mut self, timestamp_ms: u64) {
        let Some(ranging) = self.task_base.config.ranging_config.clone() else {
            return;
        };
        if !ranging.enabled {
            return;
        }
        let Some(rel18) = self.task_base.rel18.clone() else {
            warn!("SL-PRS occasion with no Rel-18 task handles; no ranging task to report to");
            return;
        };

        let own = Position3D::new(
            ranging.own_position[0],
            ranging.own_position[1],
            ranging.own_position[2],
        );

        for anchor in &ranging.anchors {
            let peer = Position3D::new(anchor.position[0], anchor.position[1], anchor.position[2]);
            let range_m = own.distance_to(&peer);
            if range_m > ranging.max_distance_meters {
                debug!(
                    "SL-PRS: anchor {} is {:.1}m away, beyond max_distance_meters \
                     {:.1}; no measurement",
                    anchor.ue_id, range_m, ranging.max_distance_meters
                );
                continue;
            }

            // A ranging session has to exist before a measurement can land in it:
            // the ranging task warns and drops a measurement for an unknown peer.
            if self.ranging_sessions_started.insert(anchor.ue_id) {
                let _ = rel18
                    .ranging_tx
                    .send(RangingMessage::StartRanging {
                        peer_ue_id: anchor.ue_id,
                        method: format!("{:?}", ranging.method).to_lowercase(),
                    })
                    .await;
            }

            // RTT of a round trip over `range_m`, at 0.3 m/ns.
            let rtt_ns = ((2.0 * range_m) / SPEED_OF_LIGHT_M_PER_NS).round() as u64;
            self.positioning
                .add_rtt_measurement(PositioningRttMeasurement::new(
                    anchor.ue_id,
                    rtt_ns,
                    SL_PRS_MEASUREMENT_QUALITY,
                    timestamp_ms,
                ));
            let _ = rel18
                .ranging_tx
                .send(RangingMessage::RttMeasurement {
                    peer_ue_id: anchor.ue_id,
                    rtt_ns,
                    timestamp_ms,
                })
                .await;

            for frequency_mhz in &ranging.carrier_frequencies_mhz {
                if *frequency_mhz <= 0.0 {
                    warn!("SL-PRS: ignoring a non-positive carrier frequency {frequency_mhz}");
                    continue;
                }
                let wavelength_m = SPEED_OF_LIGHT_M_PER_US / *frequency_mhz;
                let cycles = range_m / wavelength_m;
                let phase_rad = cycles.fract() * std::f64::consts::TAU;
                let _ = rel18
                    .ranging_tx
                    .send(RangingMessage::CarrierPhaseMeasurement {
                        peer_ue_id: anchor.ue_id,
                        frequency_mhz: *frequency_mhz,
                        phase_rad,
                        quality: SL_PRS_MEASUREMENT_QUALITY,
                    })
                    .await;
            }

            debug!(
                "SL-PRS: measured anchor {} at {:.2}m (rtt={}ns, {} carrier phases) \
                 on occasion {}",
                anchor.ue_id,
                range_m,
                rtt_ns,
                ranging.carrier_frequencies_mhz.len(),
                timestamp_ms
            );
        }
    }
}

#[async_trait::async_trait]
impl Task for SidelinkTask {
    type Message = SidelinkMessage;

    async fn run(&mut self, mut rx: mpsc::Receiver<TaskMessage<Self::Message>>) {
        info!("{}", START_LOG);
        loop {
            match rx.recv().await {
                Some(TaskMessage::Message(msg)) => self.handle_message(msg).await,
                Some(TaskMessage::Shutdown) => break,
                None => break,
            }
        }
        info!(
            "Sidelink task stopped, {} peers, {} active PC5 link(s), relay={:?}, {} \
             octet(s) relayed",
            self.peers.len(),
            self.links.active_count(),
            self.relay_mode,
            self.relay.forwarded_octets()
        );
    }
}

impl SidelinkTask {
    /// Dispatches one [`SidelinkMessage`].
    ///
    /// Split out of [`SidelinkTask::run`] and public so the two-UE integration test
    /// (`tests/src/sidelink_pc5_e2e.rs`) drives the REAL dispatch rather than a
    /// test-only copy of it. `run` is a `loop` over a channel, so a test that wanted to
    /// assert between steps could not use it -- and a second dispatch written for the
    /// test could pass while this one was wrong, which is the failure mode the whole
    /// issue is about.
    pub async fn handle_message(&mut self, msg: SidelinkMessage) {
        match msg {
            SidelinkMessage::StartDiscovery => {
                self.handle_start_discovery().await;
            }
            SidelinkMessage::StopDiscovery => {
                self.discovery.stop();
                debug!("Sidelink: Discovery stopped");
            }
            SidelinkMessage::PeerDiscovered {
                peer_ue_id,
                signal_dbm,
                timestamp_ms,
            } => {
                let peer = self
                    .peers
                    .entry(peer_ue_id)
                    .or_insert_with(|| SidelinkPeer::new(peer_ue_id));
                peer.signal_dbm = signal_dbm;
                peer.last_discovery_ms = timestamp_ms;
                if peer.link_state == Pc5LinkState::Idle {
                    peer.link_state = Pc5LinkState::Discovering;
                }
                debug!(
                    "Sidelink: Peer {} discovered signal={}dBm",
                    peer_ue_id, signal_dbm
                );
            }
            SidelinkMessage::EstablishPc5Link { peer_ue_id } => {
                self.handle_establish_pc5_link(peer_ue_id).await;
            }
            SidelinkMessage::ReleasePc5Link { peer_ue_id } => {
                self.handle_release_pc5_link(peer_ue_id).await;
            }
            SidelinkMessage::SetRelayMode { mode } => {
                let relay_mode = match mode.as_str() {
                    "l2" => RelayMode::L2Relay,
                    "l3" => RelayMode::L3Relay,
                    "ue-to-network" => RelayMode::UeToNetworkRelay,
                    _ => RelayMode::None,
                };
                self.relay_mode = relay_mode;
                // The forwarder follows the configured mode, so setting the mode
                // actually changes what this UE forwards -- it used to change
                // only which log line a subsequent `RelayData` produced.
                //
                // `UeToNetworkRelay` maps to a REAL role since issue #190: the TS 38.351
                // SRAP adaptation layer it needs now exists (`nextgsim_rlc::srap`), the
                // gNB signals the bearer mapping, and the UE applies it. It was
                // `RelayRole::None` with a `warn!` until that was true.
                //
                // `L3Relay` still maps to `RelayRole::None`: L3 relaying forwards IP
                // packets rather than adapting Layer-2 bearers, so it needs an IP
                // forwarding plane that does not exist here. Accepting the mode and
                // forwarding as though it were L2 would be a different procedure wearing
                // the requested name.
                self.relay.set_role(match relay_mode {
                    RelayMode::L2Relay => RelayRole::L2UeToUe,
                    RelayMode::UeToNetworkRelay => RelayRole::L2UeToNetwork,
                    RelayMode::None => RelayRole::None,
                    RelayMode::L3Relay => {
                        warn!(
                            "Sidelink: relay mode L3Relay is not implemented (it needs an \
                             IP forwarding plane, not the Layer-2 adaptation of TS 38.351); \
                             not relaying"
                        );
                        RelayRole::None
                    }
                });
                debug!("Sidelink: Relay mode set to {:?}", relay_mode);
            }
            SidelinkMessage::RelayData {
                source_ue_id,
                destination_ue_id,
                data_size,
            } => {
                // The size-only form, kept because it is what the CLI and the
                // older senders carry. It is answered by the same decision
                // `RelayPayload` gets, against a zero-filled payload of the
                // stated size: a relay that accepted one and refused the other
                // would be two procedures.
                let decision = self.forward_relay_payload(
                    ProseL2Id::new(source_ue_id as u32),
                    ProseL2Id::new(destination_ue_id as u32),
                    &vec![0u8; data_size as usize],
                );
                debug!(
                    "Sidelink: RelayData {} bytes from {} to {}: {:?}",
                    data_size, source_ue_id, destination_ue_id, decision
                );
            }
            SidelinkMessage::RelayPayload {
                source_l2_id,
                destination_l2_id,
                payload,
            } => {
                self.handle_relay_payload(
                    ProseL2Id::new(source_l2_id),
                    ProseL2Id::new(destination_l2_id),
                    payload,
                )
                .await;
            }
            SidelinkMessage::Pc5SReceived { pdu } => {
                self.handle_pc5s_received(&pdu).await;
            }
            SidelinkMessage::SrapMappingConfigured {
                remote_l2_id,
                local_remote_ue_id,
                is_relay,
            } => {
                // TS 38.331 §5.3.5.17 (issue #190): the network signalled a SRAP bearer
                // mapping, and the RRC task decoded it. This is where it takes effect.
                self.handle_srap_mapping_configured(remote_l2_id, local_remote_ue_id, is_relay);
            }
            SidelinkMessage::PositioningMeasurement {
                peer_ue_id,
                distance_m,
                peer_position,
            } => {
                if let Some(peer) = self.peers.get_mut(&peer_ue_id) {
                    peer.sl_distance_m = Some(distance_m);
                    peer.sl_position = peer_position;
                    debug!(
                        "Sidelink: Positioning measurement peer={} distance={:.2}m",
                        peer_ue_id, distance_m
                    );
                }
            }
            SidelinkMessage::SlPrsOccasion { timestamp_ms } => {
                self.handle_sl_prs_occasion(timestamp_ms).await;
            }
            SidelinkMessage::CooperativePositioning { response_tx } => {
                // Compute position estimate from sidelink measurements
                // using trilateration from known peer positions
                let position = self.compute_cooperative_position();
                debug!("Sidelink: Cooperative positioning result={:?}", position);
                if let Some(tx) = response_tx {
                    let _ = tx.send(position);
                }
            }
        }
    }

    /// Compute cooperative position from sidelink distance measurements
    /// using weighted centroid of peer positions adjusted by distances.
    fn compute_cooperative_position(&self) -> Option<(f64, f64, f64)> {
        let mut known_peers: Vec<(&SidelinkPeer, f64, (f64, f64, f64))> = Vec::new();

        for peer in self.peers.values() {
            if let (Some(dist), Some(pos)) = (peer.sl_distance_m, peer.sl_position) {
                if peer.link_state == Pc5LinkState::Active {
                    known_peers.push((peer, dist, pos));
                }
            }
        }

        if known_peers.len() < 3 {
            return None; // Need at least 3 peers for trilateration
        }

        // Weighted centroid approach (simplified trilateration)
        let mut wx = 0.0;
        let mut wy = 0.0;
        let mut wz = 0.0;
        let mut total_weight = 0.0;

        for (_peer, _dist, pos) in &known_peers {
            let weight = 1.0; // Equal weight for simplicity
            wx += pos.0 * weight;
            wy += pos.1 * weight;
            wz += pos.2 * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            Some((wx / total_weight, wy / total_weight, wz / total_weight))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{capture_task_logs, task_base};

    /// #54, criterion 2: with the feature on, the task must still not present
    /// sidelink as an active Rel-18 capability — enabling the feature compiles
    /// the facade back in, it does not make PC5 work.
    ///
    /// This test only exists in the `sidelink` build, so the default build's
    /// guarantee is the stronger one: the task does not exist to log anything.
    /// That half is asserted in `tasks.rs` by
    /// `a_default_build_registers_no_sidelink_task`.
    #[test]
    fn the_task_logs_no_active_rel_18_sidelink_capability() {
        let logged = capture_task_logs(SidelinkTask::new(task_base()));

        // Positive control FIRST. The assertion that matters is an absence, and
        // an absence is satisfied by every path that never arrives — a
        // subscriber never installed, a task that never started. Pinning the
        // startup line proves the buffer holds this task's own startup before
        // anything is concluded from what is missing.
        assert!(
            logged.contains(START_LOG),
            "the sidelink task's startup line is missing from the capture, so \
             nothing can be concluded from what else is absent; captured: {logged:?}"
        );

        assert!(
            !logged.contains("Rel-18 NR Sidelink"),
            "the task advertised an active Rel-18 NR Sidelink capability; \
             captured: {logged:?}"
        );
    }

    /// The binary's spawn line is unreachable from a test (it is inside
    /// `main.rs`'s task closure), so what is pinned instead is the constant it
    /// logs. Changing the binary's wording means changing this string, and this
    /// test reads this string.
    ///
    /// **Updated by issues #141 and #190, not deleted** — as each issue's honesty
    /// criterion requires. The guard keeps its shape through both revisions: a presence
    /// plus an absence, so it can still fail.
    ///
    /// #54 pinned the word "scaffold" because the surface then was one. #141 made that the
    /// *opposite* dishonesty — understating working code sends a reader looking for a
    /// feature that is there — and replaced it with "no PC5 radio, no UE-to-Network relay".
    ///
    /// #190 then made "no UE-to-Network relay" false in turn: the TS 38.351 SRAP adaptation
    /// layer, the gNB-signalled bearer mapping and the remote-UE identity all run. So that
    /// clause is retired and the assertion **inverted**: the start line must no longer claim
    /// UE-to-Network relay is absent. What it must still name is what genuinely is absent —
    /// no PC5 radio, and no L3 relay.
    #[test]
    fn the_startup_lines_advertise_no_active_sidelink_capability() {
        for line in [SPAWN_LOG, START_LOG] {
            assert!(
                line.contains("no radio") || line.contains("no PC5 radio"),
                "{line:?} must say there is no PC5 radio, or a reader takes the \
                 in-process PC5 channel for one"
            );
            assert!(
                !line.contains("Rel-18 NR Sidelink"),
                "{line:?} advertises a complete Rel-18 NR Sidelink capability the \
                 code does not deliver (a PC5 radio and L3 relay are absent)"
            );
            // And it must not go back to calling working procedures a scaffold.
            assert!(
                !line.contains("scaffold"),
                "{line:?} still calls the PC5 surface a scaffold, which issue #141 made \
                 false: the Direct Communication handshake and Model A/B discovery run"
            );
        }
        // The spawn line must name what IS live, or "no radio" reads as "nothing works".
        assert!(
            SPAWN_LOG.contains("PC5-S"),
            "the spawn line does not name the PC5-S procedures it now runs: {SPAWN_LOG:?}"
        );

        // Issue #190 inverted this clause. The start line must NOT say UE-to-Network relay
        // is unimplemented, because it is implemented -- understating it would send a
        // reader looking for a feature that is there, which is the #141 lesson applied
        // again.
        assert!(
            !START_LOG.contains("no UE-to-Network"),
            "the start line still says UE-to-Network relay is absent, which issue #190 \
             made false: the SRAP adaptation layer, the gNB bearer mapping and the \
             remote-UE identity all run: {START_LOG:?}"
        );
        // And it must name what IS absent instead, or the line claims completeness.
        assert!(
            START_LOG.contains("no L3 relay"),
            "the start line must name L3 relay as the relay case that is NOT implemented, \
             or a reader assumes every relay mode works: {START_LOG:?}"
        );
        // The positive half: the line has to name SRAP, or "L2 UE-to-Network relay" reads
        // as the same unqualified claim #54 forbade.
        assert!(
            START_LOG.contains("SRAP") && START_LOG.contains("UE-to-Network"),
            "the start line must name the L2 UE-to-Network relay and its SRAP adaptation \
             layer as live: {START_LOG:?}"
        );
    }
}
