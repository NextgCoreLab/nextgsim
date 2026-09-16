//! Sidelink Task for UE - NR relay, discovery, PC5, and sidelink positioning
//!
//! **Scaffold, not wired end-to-end.** Models the Rel-18 sidelink concepts of
//! TS 23.304 (NR sidelink relay, discovery, PC5 link establishment, cooperative
//! positioning). There is no PC5 OTA exchange and no RRC `SidelinkUEInformation`
//! / `sl-Config`, so the PC5-link and relay handlers below are unreachable at
//! runtime (issue #141 is the real procedures).
//!
//! Three messages are sent in production, all only when ranging is enabled:
//! `StartDiscovery`/`StopDiscovery`, and since issue #136 `SlPrsOccasion` — which
//! measures the configured anchors and reports to the ranging task. That
//! measurement is computed from configured geometry, not from a radio.

use std::collections::HashMap;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::sidelink::positioning::{
    AnchorUe, Position3D, RttMeasurement as PositioningRttMeasurement, SidelinkPositioningEngine,
    SlPrsResourceConfig,
};
use crate::tasks::{RangingMessage, SidelinkMessage, Task, TaskMessage, UeTaskBase};

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

/// The line the UE binary emits when it spawns the sidelink task.
///
/// A constant rather than a literal at the `info!` site because the site is in
/// `main.rs`, inside a task closure no test can reach. Naming it here is what
/// lets `the_startup_lines_advertise_no_active_sidelink_capability` pin the
/// binary's wording: the only way to change what the binary logs is to change
/// this string, and the test reads this string.
///
/// It must not present sidelink as an active Rel-18 capability — the surface is
/// a facade (see the module docs). Issue #54.
pub const SPAWN_LOG: &str =
    "Sidelink task spawned (scaffold, feature-gated: no PC5 OTA exchange, no sl-Config)";

/// The line [`SidelinkTask::run`] emits on entry. Same no-capability-claim
/// contract as [`SPAWN_LOG`], and unlike it this one is reached by a test.
pub const START_LOG: &str = "Sidelink task started (scaffold: discovery start/stop and SL-PRS \
     occasions are reachable; PC5 link and relay handlers are unwired)";

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

pub struct SidelinkTask {
    task_base: UeTaskBase,
    /// Known sidelink peers
    peers: HashMap<u64, SidelinkPeer>,
    /// This UE's relay mode
    relay_mode: RelayMode,
    /// Discovery enabled
    discovery_active: bool,
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
        Self {
            task_base,
            peers: HashMap::new(),
            relay_mode: RelayMode::None,
            discovery_active: false,
            positioning,
            ranging_sessions_started: std::collections::HashSet::new(),
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
                Some(TaskMessage::Message(msg)) => match msg {
                    SidelinkMessage::StartDiscovery => {
                        self.discovery_active = true;
                        debug!("Sidelink: Discovery started");
                    }
                    SidelinkMessage::StopDiscovery => {
                        self.discovery_active = false;
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
                        if let Some(peer) = self.peers.get_mut(&peer_ue_id) {
                            peer.link_state = Pc5LinkState::Establishing;
                            // Simulate link establishment completing
                            peer.link_state = Pc5LinkState::Active;
                            debug!("Sidelink: PC5 link established with peer {}", peer_ue_id);
                        } else {
                            warn!(
                                "Sidelink: Cannot establish PC5 link, peer {} not discovered",
                                peer_ue_id
                            );
                        }
                    }
                    SidelinkMessage::ReleasePc5Link { peer_ue_id } => {
                        if let Some(peer) = self.peers.get_mut(&peer_ue_id) {
                            peer.link_state = Pc5LinkState::Idle;
                            debug!("Sidelink: PC5 link released with peer {}", peer_ue_id);
                        }
                    }
                    SidelinkMessage::SetRelayMode { mode } => {
                        let relay_mode = match mode.as_str() {
                            "l2" => RelayMode::L2Relay,
                            "l3" => RelayMode::L3Relay,
                            "ue-to-network" => RelayMode::UeToNetworkRelay,
                            _ => RelayMode::None,
                        };
                        self.relay_mode = relay_mode;
                        debug!("Sidelink: Relay mode set to {:?}", relay_mode);
                    }
                    SidelinkMessage::RelayData {
                        source_ue_id,
                        destination_ue_id,
                        data_size,
                    } => {
                        if self.relay_mode == RelayMode::None {
                            warn!("Sidelink: Relay data received but relay mode is None");
                        } else {
                            debug!(
                                "Sidelink: Relaying {} bytes from UE {} to UE {}",
                                data_size, source_ue_id, destination_ue_id
                            );
                        }
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
                },
                Some(TaskMessage::Shutdown) => break,
                None => break,
            }
        }
        info!(
            "Sidelink task stopped, {} peers, relay={:?}",
            self.peers.len(),
            self.relay_mode
        );
    }
}

impl SidelinkTask {
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
        // scaffold line proves the buffer holds this task's own startup before
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
    #[test]
    fn the_startup_lines_advertise_no_active_sidelink_capability() {
        for line in [SPAWN_LOG, START_LOG] {
            assert!(
                line.contains("scaffold"),
                "{line:?} must say it is a scaffold, or a reader takes the spawn \
                 for a working feature"
            );
            assert!(
                !line.contains("Rel-18 NR Sidelink"),
                "{line:?} advertises an active Rel-18 NR Sidelink capability the \
                 code does not deliver"
            );
        }
    }
}
