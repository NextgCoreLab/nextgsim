//! PC5 direct discovery, Model A and Model B (TS 23.304 §6.3.1.2, §6.3.1.3).
//!
//! # Why this module exists (issue #141)
//!
//! Discovery was a single `bool`:
//!
//! ```text
//! SidelinkMessage::StartDiscovery => { self.discovery_active = true; }
//! SidelinkMessage::StopDiscovery  => { self.discovery_active = false; }
//! ```
//!
//! Nothing was announced, nothing was monitored, no filter was applied and no peer was
//! ever discovered by discovery — `SidelinkMessage::PeerDiscovered` had no sender at
//! all. A UE with `discovery_active = true` behaved identically to one with it `false`.
//!
//! This module is a real exchange: a UE **announces** (Model A) or **solicits**
//! (Model B) a ProSe Application Code, and a peer that matches its own discovery filter
//! learns the announcer's Layer-2 ID — and, in Model B, answers.
//!
//! # Both models, not one
//!
//! The issue's criterion says "Model A and/or Model B", licensing either. Both are here
//! because the difference between them is one message type and one `if`, and
//! implementing only Model A would leave [`crate::sidelink::pc5s::Pc5SMessageType`]
//! carrying two solicit/response variants with no sender — the exact unreachable-handler
//! defect this issue is about.
//!
//! * **Model A** (§6.3.1.2) — the announcer broadcasts unsolicited; monitors listen.
//!   One message, no answer.
//! * **Model B** (§6.3.1.3) — the discoverer broadcasts a solicitation; every matching
//!   discoveree unicasts a response back. Two messages.
//!
//! # The filter is what makes it discovery rather than a broadcast
//!
//! TS 23.304 §6.3.1.2 step 3b: a monitor is "provided with a Discovery Filter consisting
//! of ProSe Application Code(s) ... and/or ProSe Application Mask(s)" and reports only
//! "one or more ProSe Application Code(s) ... that match the filter (see clause 5.8.1)".
//! So a monitor with a filter that does not match learns nothing — and
//! [`Pc5DiscoveryEngine::on_message_received`] returns `None` for it, which is what
//! distinguishes this from the old unconditional bool.

use std::collections::HashMap;

use tracing::debug;

use crate::sidelink::pc5s::{DirectDiscoveryMessage, Pc5SMessageType, ProseL2Id, RelayServiceCode};

/// Which discovery model this UE is running (TS 23.304 §6.3.1.2, §6.3.1.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscoveryModel {
    /// Model A: announce unsolicited, or monitor for announcements.
    ModelA,
    /// Model B: solicit, or answer solicitations.
    ModelB,
}

/// A peer this UE learned about from a discovery message.
///
/// Every field here came off the wire. That is the point: before this, a "discovered
/// peer" could only be injected by a test, because nothing produced one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiscoveredPeer {
    /// The peer's Layer-2 ID, from the discovery message's `Source User Info`.
    pub l2_id: ProseL2Id,
    /// The `ProSe Application Code` the peer announced (TS 23.304 §5.8.1).
    pub prose_app_code: u32,
    /// The Relay Service Code the peer serves, when it announced one
    /// (TS 23.304 §6.3.2).
    ///
    /// `Some` is what makes a peer eligible to be selected as a relay: a remote UE
    /// picks its relay from peers that *said* they serve the code it needs.
    pub relay_service_code: Option<RelayServiceCode>,
    /// Which model the peer was discovered under.
    pub model: DiscoveryModel,
    /// When it was last heard from, in milliseconds on the caller's clock.
    pub last_seen_ms: u64,
}

/// A UE's discovery filter: which ProSe Application Codes it cares about
/// (TS 23.304 §5.8.1, §6.3.1.2 step 3b).
///
/// The mask is applied before the comparison, which is what `§5.8.1`'s "ProSe
/// Application Mask" is for: an application that allocates a block of codes to one
/// service matches the block with one filter entry rather than enumerating it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiscoveryFilter {
    /// The code to match, after masking.
    pub prose_app_code: u32,
    /// The mask applied to both sides before comparing. `u32::MAX` is an exact match.
    pub mask: u32,
}

impl DiscoveryFilter {
    /// A filter matching exactly one ProSe Application Code.
    pub fn exact(prose_app_code: u32) -> Self {
        Self {
            prose_app_code,
            mask: u32::MAX,
        }
    }

    /// A filter matching every code whose masked bits equal `prose_app_code`'s.
    pub fn masked(prose_app_code: u32, mask: u32) -> Self {
        Self {
            prose_app_code,
            mask,
        }
    }

    /// Whether `candidate` matches this filter (TS 23.304 §5.8.1).
    pub fn matches(&self, candidate: u32) -> bool {
        (candidate & self.mask) == (self.prose_app_code & self.mask)
    }
}

/// What a UE should do having received a discovery message.
///
/// A returned message is the caller's to transmit; this engine does no I/O for the same
/// reason [`crate::sidelink::link::Pc5LinkContext`] does not — the PC5 transport belongs
/// to the caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveryOutcome {
    /// The peer that was learned, if the message matched this UE's filter.
    pub peer: Option<DiscoveredPeer>,
    /// The Model B response to unicast back, if this UE is answering a solicitation.
    pub response: Option<DirectDiscoveryMessage>,
}

/// A UE's discovery state: what it announces, what it listens for, and what it has
/// found.
#[derive(Debug)]
pub struct Pc5DiscoveryEngine {
    /// This UE's Layer-2 ID, the `Source User Info` of everything it sends.
    local_l2_id: ProseL2Id,
    /// The code this UE announces or solicits, once started. `None` when stopped.
    ///
    /// This replaces `discovery_active: bool`: "running" and "which code" are one fact,
    /// and a UE that is announcing without a code to announce is not a state worth
    /// representing.
    announcing: Option<u32>,
    /// Which model [`Self::announcing`] is running under.
    model: DiscoveryModel,
    /// The Relay Service Code this UE serves and announces, if it is a relay.
    served_relay_service_code: Option<RelayServiceCode>,
    /// What this UE will report a match on (TS 23.304 §6.3.1.2 step 3b).
    filters: Vec<DiscoveryFilter>,
    /// Peers learned from received discovery messages, keyed by Layer-2 ID.
    peers: HashMap<ProseL2Id, DiscoveredPeer>,
}

impl Pc5DiscoveryEngine {
    /// A stopped engine with no filters.
    pub fn new(local_l2_id: ProseL2Id) -> Self {
        Self {
            local_l2_id,
            announcing: None,
            model: DiscoveryModel::ModelA,
            served_relay_service_code: None,
            filters: Vec::new(),
            peers: HashMap::new(),
        }
    }

    /// This UE's Layer-2 ID.
    pub fn local_l2_id(&self) -> ProseL2Id {
        self.local_l2_id
    }

    /// Declares the Relay Service Code this UE serves, which it then announces
    /// (TS 23.304 §6.3.2).
    pub fn serve_relay_service_code(&mut self, rsc: RelayServiceCode) {
        self.served_relay_service_code = Some(rsc);
    }

    /// The Relay Service Code this UE serves, if any.
    pub fn served_relay_service_code(&self) -> Option<RelayServiceCode> {
        self.served_relay_service_code
    }

    /// Adds a discovery filter (TS 23.304 §6.3.1.2 step 3b).
    pub fn add_filter(&mut self, filter: DiscoveryFilter) {
        self.filters.push(filter);
    }

    /// Whether discovery is running.
    pub fn is_active(&self) -> bool {
        self.announcing.is_some()
    }

    /// Which model discovery is running under.
    pub fn model(&self) -> DiscoveryModel {
        self.model
    }

    /// **Starts discovery**, returning the message to broadcast.
    ///
    /// Model A returns an announcement (§6.3.1.2 step 3a, "it starts announcing on PC5
    /// interface"); Model B returns a solicitation (§6.3.1.3). Either way a message
    /// *leaves* — which is the difference from setting a bool.
    ///
    /// The announcement carries this UE's served Relay Service Code when it has one, so
    /// a remote UE monitoring for a relay can find it. That is the only way relay
    /// discovery works at all: without the RSC on the wire, relay selection could only
    /// read local configuration.
    pub fn start(&mut self, prose_app_code: u32, model: DiscoveryModel) -> DirectDiscoveryMessage {
        self.announcing = Some(prose_app_code);
        self.model = model;
        let mut msg = match model {
            DiscoveryModel::ModelA => {
                DirectDiscoveryMessage::announcement(self.local_l2_id, prose_app_code)
            }
            DiscoveryModel::ModelB => {
                DirectDiscoveryMessage::solicitation(self.local_l2_id, prose_app_code)
            }
        };
        if let Some(rsc) = self.served_relay_service_code {
            msg = msg.with_relay_service_code(rsc);
        }
        debug!(
            "PC5 discovery: {} started {:?} for app code 0x{:08X}",
            self.local_l2_id, model, prose_app_code
        );
        msg
    }

    /// **Stops discovery.** Nothing is transmitted; a UE simply ceases to announce.
    ///
    /// Peers already discovered are kept: TS 23.304 has no "forget everything" on stop,
    /// and a link established to a peer found earlier does not become invalid because
    /// announcing stopped. [`Self::prune_stale_peers`] is what expires them.
    pub fn stop(&mut self) {
        debug!("PC5 discovery: {} stopped announcing", self.local_l2_id);
        self.announcing = None;
    }

    /// The periodic re-announcement, when discovery is running.
    ///
    /// `None` when stopped. Model A announcing is periodic by nature (TS 23.304
    /// §6.3.1.2: the UE "starts announcing", not "announces once"), so the caller drives
    /// this on its own interval.
    pub fn periodic_announcement(&self) -> Option<DirectDiscoveryMessage> {
        let code = self.announcing?;
        let mut msg = match self.model {
            DiscoveryModel::ModelA => DirectDiscoveryMessage::announcement(self.local_l2_id, code),
            DiscoveryModel::ModelB => DirectDiscoveryMessage::solicitation(self.local_l2_id, code),
        };
        if let Some(rsc) = self.served_relay_service_code {
            msg = msg.with_relay_service_code(rsc);
        }
        Some(msg)
    }

    /// **Applies a received discovery message.**
    ///
    /// Returns what the UE learned and what it must answer:
    ///
    /// * A message whose ProSe Application Code matches no filter yields
    ///   `DiscoveryOutcome { peer: None, response: None }` — the monitor learns nothing,
    ///   per §6.3.1.2 step 4b's "that match the filter". This is the case the old
    ///   `discovery_active` bool could not express.
    /// * A matching **announcement** or **response** yields a peer and no answer.
    /// * A matching **solicitation** yields a peer *and* a Model B response to unicast
    ///   back (§6.3.1.3), carrying this UE's own code and served RSC.
    ///
    /// This UE's own messages are ignored: a broadcast destination Layer-2 ID
    /// (§5.8.2.4) means a UE can hear its own announcement, and discovering itself
    /// would put a bogus peer in the table.
    pub fn on_message_received(
        &mut self,
        msg: &DirectDiscoveryMessage,
        now_ms: u64,
    ) -> DiscoveryOutcome {
        let none = DiscoveryOutcome {
            peer: None,
            response: None,
        };

        if msg.source_l2_id == self.local_l2_id {
            return none;
        }

        if !self.filters.iter().any(|f| f.matches(msg.prose_app_code)) {
            debug!(
                "PC5 discovery: {} ignoring app code 0x{:08X} from {}: matches no filter",
                self.local_l2_id, msg.prose_app_code, msg.source_l2_id
            );
            return none;
        }

        let model = match msg.message_type {
            Pc5SMessageType::DirectDiscoveryAnnouncement => DiscoveryModel::ModelA,
            _ => DiscoveryModel::ModelB,
        };
        let peer = DiscoveredPeer {
            l2_id: msg.source_l2_id,
            prose_app_code: msg.prose_app_code,
            relay_service_code: msg.relay_service_code,
            model,
            last_seen_ms: now_ms,
        };
        self.peers.insert(peer.l2_id, peer);
        debug!(
            "PC5 discovery: {} discovered {} (app code 0x{:08X}, rsc={:?}, {:?})",
            self.local_l2_id, peer.l2_id, peer.prose_app_code, peer.relay_service_code, model
        );

        // Model B: a solicitation is answered, an announcement and a response are not.
        let response = if msg.expects_response() {
            let code = self.announcing.unwrap_or(msg.prose_app_code);
            let mut reply = DirectDiscoveryMessage::response(self.local_l2_id, code);
            if let Some(rsc) = self.served_relay_service_code {
                reply = reply.with_relay_service_code(rsc);
            }
            Some(reply)
        } else {
            None
        };

        DiscoveryOutcome {
            peer: Some(peer),
            response,
        }
    }

    /// Every peer discovered so far, ordered by Layer-2 ID.
    ///
    /// Sorted because a `HashMap`'s order is not stable and a caller choosing a relay
    /// from this list would otherwise pick nondeterministically.
    pub fn peers(&self) -> Vec<DiscoveredPeer> {
        let mut peers: Vec<DiscoveredPeer> = self.peers.values().copied().collect();
        peers.sort_unstable_by_key(|p| p.l2_id);
        peers
    }

    /// A discovered peer by Layer-2 ID.
    pub fn peer(&self, l2_id: ProseL2Id) -> Option<&DiscoveredPeer> {
        self.peers.get(&l2_id)
    }

    /// How many peers have been discovered.
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// **Selects a relay** for a Relay Service Code, from the peers that announced it
    /// (TS 23.304 §6.3.2, §5.4.2).
    ///
    /// The lowest matching Layer-2 ID, for determinism. TS 23.304 leaves relay selection
    /// to implementation — §5.4.2 gives no ordering — and the honest choice here is a
    /// stable one rather than an RSRP comparison this simulator has no PC5 measurement
    /// to make. (`prose.rs`'s old `select_relay` ranked by RSRP, but nothing ever
    /// populated an RSRP for a sidelink peer: the PC5 receive path had no signal
    /// measurement at all, so that ranking read a value no code wrote.)
    pub fn select_relay(&self, rsc: RelayServiceCode) -> Option<DiscoveredPeer> {
        self.peers
            .values()
            .filter(|p| p.relay_service_code == Some(rsc))
            .min_by_key(|p| p.l2_id)
            .copied()
    }

    /// Drops peers not heard from within `timeout_ms` of `now_ms`.
    ///
    /// Returns how many were dropped. Saturating arithmetic, so a `now_ms` behind a
    /// peer's `last_seen_ms` (a clock that went backwards) expires nothing rather than
    /// expiring everything.
    pub fn prune_stale_peers(&mut self, now_ms: u64, timeout_ms: u64) -> usize {
        let before = self.peers.len();
        self.peers
            .retain(|_, p| now_ms.saturating_sub(p.last_seen_ms) < timeout_ms);
        before - self.peers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const UE1: u32 = 0x00_00_01;
    const UE2: u32 = 0x00_00_02;
    const APP_CODE: u32 = 0xCAFE_0001;
    const OTHER_APP_CODE: u32 = 0xBEEF_0002;
    const RSC: u32 = 0x00_AB_CD;

    fn ue1() -> ProseL2Id {
        ProseL2Id::new(UE1)
    }
    fn ue2() -> ProseL2Id {
        ProseL2Id::new(UE2)
    }

    /// Model A, end to end over the wire: the announcer announces, the monitor decodes
    /// the bytes and learns the announcer's Layer-2 ID.
    ///
    /// The assertion is the discovered peer's identity — a value that can only have come
    /// from the announcer's own octets. The old `discovery_active` bool could not
    /// produce it, because nothing was ever sent.
    #[test]
    fn model_a_announcing_is_discovered_by_a_monitor_with_a_matching_filter() {
        let mut announcer = Pc5DiscoveryEngine::new(ue1());
        let mut monitor = Pc5DiscoveryEngine::new(ue2());
        monitor.add_filter(DiscoveryFilter::exact(APP_CODE));

        let announcement = announcer.start(APP_CODE, DiscoveryModel::ModelA);
        assert!(announcer.is_active());

        // Across the wire as bytes.
        let decoded = DirectDiscoveryMessage::decode(&announcement.encode()).expect("decode");
        let outcome = monitor.on_message_received(&decoded, 1_000);

        let peer = outcome
            .peer
            .expect("a matching announcement discovers the announcer");
        assert_eq!(peer.l2_id, ue1());
        assert_eq!(peer.prose_app_code, APP_CODE);
        assert_eq!(peer.model, DiscoveryModel::ModelA);
        // Model A is unsolicited: nothing is sent back.
        assert_eq!(outcome.response, None);
        assert_eq!(monitor.peer_count(), 1);
    }

    /// A monitor whose filter does not match learns nothing. This is the behaviour the
    /// old bool could not have: it was on or off, never selective.
    #[test]
    fn a_monitor_whose_filter_does_not_match_discovers_nothing() {
        let mut announcer = Pc5DiscoveryEngine::new(ue1());
        let mut monitor = Pc5DiscoveryEngine::new(ue2());
        monitor.add_filter(DiscoveryFilter::exact(OTHER_APP_CODE));

        let announcement = announcer.start(APP_CODE, DiscoveryModel::ModelA);
        let outcome = monitor.on_message_received(
            &DirectDiscoveryMessage::decode(&announcement.encode()).unwrap(),
            1_000,
        );
        assert_eq!(outcome.peer, None);
        assert_eq!(monitor.peer_count(), 0);
    }

    /// A ProSe Application Mask matches a block of codes with one filter entry
    /// (TS 23.304 §5.8.1).
    #[test]
    fn a_masked_filter_matches_a_block_of_application_codes() {
        let filter = DiscoveryFilter::masked(0xCAFE_0000, 0xFFFF_0000);
        assert!(filter.matches(0xCAFE_0001));
        assert!(filter.matches(0xCAFE_FFFF));
        assert!(!filter.matches(0xCAFF_0001));
        // And an exact filter is not fooled by a neighbour.
        assert!(!DiscoveryFilter::exact(0xCAFE_0001).matches(0xCAFE_0002));
    }

    /// Model B, end to end: the solicitation is answered, and BOTH ends end up knowing
    /// each other — which is the two-way exchange Model A does not have.
    #[test]
    fn model_b_solicitation_is_answered_and_both_ends_learn_each_other() {
        let mut discoverer = Pc5DiscoveryEngine::new(ue1());
        discoverer.add_filter(DiscoveryFilter::exact(APP_CODE));
        let mut discoveree = Pc5DiscoveryEngine::new(ue2());
        discoveree.add_filter(DiscoveryFilter::exact(APP_CODE));

        let solicitation = discoverer.start(APP_CODE, DiscoveryModel::ModelB);
        assert!(solicitation.expects_response());

        // The discoveree hears it, learns the discoverer, and must answer.
        let outcome = discoveree.on_message_received(
            &DirectDiscoveryMessage::decode(&solicitation.encode()).unwrap(),
            10,
        );
        assert_eq!(outcome.peer.expect("learned").l2_id, ue1());
        let response = outcome
            .response
            .expect("Model B solicitation must be answered");
        assert_eq!(
            response.message_type,
            Pc5SMessageType::DirectDiscoveryResponse
        );

        // The discoverer hears the response and learns the discoveree.
        let back = discoverer.on_message_received(
            &DirectDiscoveryMessage::decode(&response.encode()).unwrap(),
            20,
        );
        let peer = back.peer.expect("the response discovers the discoveree");
        assert_eq!(peer.l2_id, ue2());
        assert_eq!(peer.model, DiscoveryModel::ModelB);
        // A response is not itself answered, or the two would ping-pong forever.
        assert_eq!(back.response, None);
    }

    /// A UE does not discover itself off its own broadcast.
    #[test]
    fn a_ue_ignores_its_own_announcement() {
        let mut ue = Pc5DiscoveryEngine::new(ue1());
        ue.add_filter(DiscoveryFilter::exact(APP_CODE));
        let own = ue.start(APP_CODE, DiscoveryModel::ModelA);
        let outcome =
            ue.on_message_received(&DirectDiscoveryMessage::decode(&own.encode()).unwrap(), 1);
        assert_eq!(outcome.peer, None);
        assert_eq!(ue.peer_count(), 0);
    }

    /// Relay discovery: the relay announces its RSC, and the remote UE selects it from
    /// what it heard — not from local configuration.
    #[test]
    fn a_relay_is_selected_from_the_relay_service_code_it_announced() {
        let rsc = RelayServiceCode::new(RSC);
        let mut relay = Pc5DiscoveryEngine::new(ue2());
        relay.serve_relay_service_code(rsc);
        let mut remote = Pc5DiscoveryEngine::new(ue1());
        remote.add_filter(DiscoveryFilter::exact(APP_CODE));

        // Before hearing anything, there is no relay to pick.
        assert_eq!(remote.select_relay(rsc), None);

        let announcement = relay.start(APP_CODE, DiscoveryModel::ModelA);
        remote.on_message_received(
            &DirectDiscoveryMessage::decode(&announcement.encode()).unwrap(),
            5,
        );

        let selected = remote.select_relay(rsc).expect("the announced relay");
        assert_eq!(selected.l2_id, ue2());
        assert_eq!(selected.relay_service_code, Some(rsc));
        // A code nobody announced still selects nothing.
        assert_eq!(remote.select_relay(RelayServiceCode::new(0x00_00_99)), None);
    }

    /// Stopping ceases announcing but does not forget peers; pruning is what expires
    /// them.
    #[test]
    fn stopping_ceases_announcing_and_pruning_expires_peers() {
        let mut ue = Pc5DiscoveryEngine::new(ue1());
        ue.add_filter(DiscoveryFilter::exact(APP_CODE));
        ue.start(APP_CODE, DiscoveryModel::ModelA);

        let peer_msg = DirectDiscoveryMessage::announcement(ue2(), APP_CODE);
        ue.on_message_received(&peer_msg, 1_000);
        assert_eq!(ue.peer_count(), 1);
        assert!(ue.periodic_announcement().is_some());

        ue.stop();
        assert!(!ue.is_active());
        // Stopped: nothing more to announce.
        assert_eq!(ue.periodic_announcement(), None);
        // But the peer is still known.
        assert_eq!(ue.peer_count(), 1);

        // Not yet stale at 1500ms with a 1000ms timeout... it is 500ms old.
        assert_eq!(ue.prune_stale_peers(1_500, 1_000), 0);
        assert_eq!(ue.peer_count(), 1);
        // Stale at 2500ms.
        assert_eq!(ue.prune_stale_peers(2_500, 1_000), 1);
        assert_eq!(ue.peer_count(), 0);
    }

    /// A clock that went backwards expires nothing, rather than expiring everything.
    #[test]
    fn a_backwards_clock_does_not_expire_every_peer() {
        let mut ue = Pc5DiscoveryEngine::new(ue1());
        ue.add_filter(DiscoveryFilter::exact(APP_CODE));
        ue.on_message_received(
            &DirectDiscoveryMessage::announcement(ue2(), APP_CODE),
            10_000,
        );
        assert_eq!(ue.prune_stale_peers(5_000, 1_000), 0);
        assert_eq!(ue.peer_count(), 1);
    }

    /// The periodic announcement carries the served RSC, so a relay keeps advertising
    /// itself rather than only doing so on its first announcement.
    #[test]
    fn a_periodic_announcement_still_carries_the_relay_service_code() {
        let rsc = RelayServiceCode::new(RSC);
        let mut relay = Pc5DiscoveryEngine::new(ue2());
        relay.serve_relay_service_code(rsc);
        relay.start(APP_CODE, DiscoveryModel::ModelA);
        let periodic = relay.periodic_announcement().expect("running");
        assert_eq!(periodic.relay_service_code, Some(rsc));
        assert_eq!(
            periodic.message_type,
            Pc5SMessageType::DirectDiscoveryAnnouncement
        );
    }
}
