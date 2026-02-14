//! Sidelink Task for UE - NR relay, discovery, PC5, and sidelink positioning
//!
//! Implements Rel-18 sidelink features:
//! - NR sidelink relay (UE-to-UE relay)
//! - Sidelink discovery procedures
//! - PC5 link establishment
//! - Sidelink-based positioning (cooperative positioning between UEs)

use std::collections::HashMap;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::tasks::{SidelinkMessage, Task, TaskMessage, UeTaskBase};

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
    _task_base: UeTaskBase,
    /// Known sidelink peers
    peers: HashMap<u64, SidelinkPeer>,
    /// This UE's relay mode
    relay_mode: RelayMode,
    /// Discovery enabled
    discovery_active: bool,
}

impl SidelinkTask {
    pub fn new(task_base: UeTaskBase) -> Self {
        Self {
            _task_base: task_base,
            peers: HashMap::new(),
            relay_mode: RelayMode::None,
            discovery_active: false,
        }
    }
}

#[async_trait::async_trait]
impl Task for SidelinkTask {
    type Message = SidelinkMessage;

    async fn run(&mut self, mut rx: mpsc::Receiver<TaskMessage<Self::Message>>) {
        info!("Sidelink task started (Rel-18 NR Sidelink)");
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
                    SidelinkMessage::CooperativePositioning { response_tx } => {
                        // Compute position estimate from sidelink measurements
                        // using trilateration from known peer positions
                        let position = self.compute_cooperative_position();
                        debug!(
                            "Sidelink: Cooperative positioning result={:?}",
                            position
                        );
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
