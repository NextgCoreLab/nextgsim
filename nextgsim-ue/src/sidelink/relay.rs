//! UE-to-UE relay forwarding over PC5 (TS 23.304 §6.4.3, TS 38.300 §16.9).
//!
//! # Why this module exists (issue #141)
//!
//! `SidelinkMessage::RelayData` used to be:
//!
//! ```text
//! if self.relay_mode == RelayMode::None {
//!     warn!("Sidelink: Relay data received but relay mode is None");
//! } else {
//!     debug!("Sidelink: Relaying {} bytes from UE {} to UE {}", ...);
//! }
//! ```
//!
//! A log line. Nothing was forwarded, no link was consulted, and `RelayData` had no
//! sender anywhere in the tree — so even the log was unreachable. `SetRelayMode` was
//! likewise senderless, which meant the `relay_mode == None` branch was the only one any
//! execution could have taken.
//!
//! This module actually forwards: given a source, a destination and a payload, it
//! decides — against the **real PC5 link table** — whether the traffic can go, and to
//! which peer. A forward that succeeds proves a unicast link to the destination was
//! established by a PC5-S handshake, because that is the only thing that puts a peer in
//! [`Pc5LinkTable::active_peers`].
//!
//! # Scope: layer-2 UE-to-UE relay, not UE-to-Network relay
//!
//! What is implemented is **UE-to-UE** relaying: UE-1 → relay → UE-2, all three over
//! PC5. TS 23.304 §5.4.2's **UE-to-Network** relay — a remote UE reaching the 5GC
//! *through* a relay UE's Uu connection — is deliberately not, and issue #141's
//! criterion is scoped to "a two-UE relay forwards data", which is the UE-to-UE case.
//!
//! UE-to-Network relaying needs an adaptation layer (SRAP, TS 38.351) that multiplexes
//! remote-UE bearers onto the relay's own Uu RLC channels, a relay-side bearer mapping
//! signalled by the gNB, and remote-UE identity handling in the relay's RRC — none of
//! which exists in this tree, and none of which this module pretends to. It is filed as
//! **issue #190** rather than stubbed.
//!
//! # Why a decision type rather than a `bool`
//!
//! A relay refuses for several distinguishable reasons, and the whole defect this
//! replaces was a path where "did not forward" and "forwarded" produced the same
//! observable (a log line). [`RelayForwardDecision`] makes the refusal reason a value a
//! caller — and a test — can assert on positively.

use tracing::{debug, warn};

use crate::sidelink::link::Pc5LinkTable;
use crate::sidelink::pc5s::ProseL2Id;

/// Which relay role this UE is playing (TS 23.304 §5.4.2, TS 38.300 §16.9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelayRole {
    /// Not relaying. Traffic that is not for this UE is dropped.
    None,
    /// Layer-2 UE-to-UE relay: forwards PC5 traffic between two remote UEs
    /// (TS 23.304 §6.4.3.10, TS 38.300 §16.9).
    L2UeToUe,
}

/// What a relay decided to do with a payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RelayForwardDecision {
    /// Forward `payload_len` octets to `next_hop`.
    ///
    /// `next_hop` is a peer from the PC5 link table, so it is reachable by definition:
    /// it got there by completing a `Direct Communication Request`/`Accept` handshake.
    Forward {
        /// The peer the payload goes to.
        next_hop: ProseL2Id,
        /// How many octets are forwarded.
        payload_len: usize,
    },
    /// This UE is not a relay, so traffic between two other UEs is not its business.
    NotARelay,
    /// This UE is a relay, but has no usable PC5 link to the destination.
    ///
    /// The distinction from [`Self::NotARelay`] matters operationally: one is a
    /// configuration state, the other is a link that was never established or has been
    /// released.
    NoLinkToDestination {
        /// The destination that could not be reached.
        destination: ProseL2Id,
    },
    /// The payload was addressed to this UE itself, so there is nothing to relay.
    ///
    /// Not an error: a relay is also an ordinary PC5 peer, and traffic for it is
    /// delivered rather than forwarded. Returned so the caller can tell "deliver
    /// locally" from "forward" without re-checking the address.
    DeliverLocally,
    /// An empty payload. Forwarding nothing would consume a PC5 grant to no purpose.
    EmptyPayload,
}

impl RelayForwardDecision {
    /// Whether the payload is being forwarded.
    pub fn is_forwarded(&self) -> bool {
        matches!(self, Self::Forward { .. })
    }

    /// The next hop, when the payload is being forwarded.
    pub fn next_hop(&self) -> Option<ProseL2Id> {
        match self {
            Self::Forward { next_hop, .. } => Some(*next_hop),
            _ => None,
        }
    }
}

/// A UE's relay forwarding state, and the counters a test or an operator reads.
#[derive(Debug)]
pub struct RelayForwarder {
    /// This UE's own Layer-2 ID, so it can recognise traffic addressed to itself.
    local_l2_id: ProseL2Id,
    /// Which relay role this UE is playing.
    role: RelayRole,
    /// Octets forwarded on behalf of other UEs.
    ///
    /// The observable that proves relaying happened. A log line could not be asserted on
    /// without a subscriber; this can, and it is only ever incremented on the forwarding
    /// path.
    forwarded_octets: u64,
    /// How many payloads were forwarded.
    forwarded_count: u64,
    /// How many payloads were dropped because there was no link to the destination.
    dropped_no_link: u64,
}

impl RelayForwarder {
    /// A UE that is not relaying.
    pub fn new(local_l2_id: ProseL2Id) -> Self {
        Self {
            local_l2_id,
            role: RelayRole::None,
            forwarded_octets: 0,
            forwarded_count: 0,
            dropped_no_link: 0,
        }
    }

    /// This UE's Layer-2 ID.
    pub fn local_l2_id(&self) -> ProseL2Id {
        self.local_l2_id
    }

    /// Sets the relay role (TS 23.304 §5.4.2).
    pub fn set_role(&mut self, role: RelayRole) {
        if self.role != role {
            debug!(
                "PC5 relay: {} role {:?} -> {:?}",
                self.local_l2_id, self.role, role
            );
        }
        self.role = role;
    }

    /// The current relay role.
    pub fn role(&self) -> RelayRole {
        self.role
    }

    /// Whether this UE is relaying.
    pub fn is_relaying(&self) -> bool {
        self.role != RelayRole::None
    }

    /// Octets forwarded on behalf of other UEs.
    pub fn forwarded_octets(&self) -> u64 {
        self.forwarded_octets
    }

    /// Payloads forwarded on behalf of other UEs.
    pub fn forwarded_count(&self) -> u64 {
        self.forwarded_count
    }

    /// Payloads dropped for want of a link to the destination.
    pub fn dropped_no_link(&self) -> u64 {
        self.dropped_no_link
    }

    /// **Decides what to do with a payload**, against the real PC5 link table.
    ///
    /// The order of the checks is deliberate and is the substance of the procedure:
    ///
    /// 1. **Addressed to us** → deliver locally. Checked first because a relay is also an
    ///    ordinary peer, and a relay that forwarded its own traffic onward would loop it.
    /// 2. **Not a relay** → drop. TS 23.304 gives a non-relay UE no business carrying
    ///    another UE's traffic.
    /// 3. **Empty** → drop.
    /// 4. **No usable link to the destination** → drop, counted separately. `links` is
    ///    consulted rather than a local peer list, so a destination is only reachable if
    ///    a PC5-S handshake established a link to it and that link has not been
    ///    released.
    ///
    /// The counters are advanced here rather than by the caller, so "the relay forwarded
    /// this" cannot be claimed by a caller that did not go through this decision.
    pub fn forward(
        &mut self,
        links: &Pc5LinkTable,
        source: ProseL2Id,
        destination: ProseL2Id,
        payload: &[u8],
    ) -> RelayForwardDecision {
        if destination == self.local_l2_id {
            return RelayForwardDecision::DeliverLocally;
        }

        if !self.is_relaying() {
            warn!(
                "PC5 relay: {} is not a relay; dropping {} octet(s) from {source} for \
                 {destination}",
                self.local_l2_id,
                payload.len()
            );
            return RelayForwardDecision::NotARelay;
        }

        if payload.is_empty() {
            return RelayForwardDecision::EmptyPayload;
        }

        // The destination has to be a peer this UE holds a USABLE link to. That is the
        // check the old code had no way to make: it had no link table to consult.
        if !links.active_peers().contains(&destination) {
            self.dropped_no_link += 1;
            warn!(
                "PC5 relay: {} has no usable PC5 link to {destination}; dropping {} \
                 octet(s) from {source}",
                self.local_l2_id,
                payload.len()
            );
            return RelayForwardDecision::NoLinkToDestination { destination };
        }

        self.forwarded_octets += payload.len() as u64;
        self.forwarded_count += 1;
        debug!(
            "PC5 relay: {} forwarded {} octet(s) from {source} to {destination}",
            self.local_l2_id,
            payload.len()
        );
        RelayForwardDecision::Forward {
            next_hop: destination,
            payload_len: payload.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sidelink::link::Pc5LinkContext;
    use crate::sidelink::pc5s::DirectCommunicationAccept;

    const RELAY: u32 = 0x00_00_10;
    const UE1: u32 = 0x00_00_01;
    const UE2: u32 = 0x00_00_02;
    const UE3: u32 = 0x00_00_03;

    fn relay_id() -> ProseL2Id {
        ProseL2Id::new(RELAY)
    }
    fn ue1() -> ProseL2Id {
        ProseL2Id::new(UE1)
    }
    fn ue2() -> ProseL2Id {
        ProseL2Id::new(UE2)
    }
    fn ue3() -> ProseL2Id {
        ProseL2Id::new(UE3)
    }

    /// Builds a link table holding a genuinely established link to `peer` — through the
    /// real handshake, not by setting a state.
    ///
    /// This is what makes the forwarding assertions meaningful: `active_peers` can only
    /// contain a peer whose accept was decoded.
    fn table_with_active_link_to(local: ProseL2Id, peer: ProseL2Id) -> Pc5LinkTable {
        let mut table = Pc5LinkTable::new();
        let mut link = Pc5LinkContext::new_initiator(local);
        let _ = link.build_request(Some(peer), None, Vec::new(), Vec::new());
        link.on_accept_received(&DirectCommunicationAccept {
            source_l2_id: peer,
            pfi: 1,
            acting_as_relay: false,
        })
        .expect("the accept applies");
        table.insert(peer, link);
        table
    }

    /// The criterion-4 case: a relay with an established link to the destination
    /// forwards, and the forwarded octets are observable.
    #[test]
    fn a_relay_forwards_to_a_peer_it_holds_an_established_link_to() {
        let links = table_with_active_link_to(relay_id(), ue2());
        let mut relay = RelayForwarder::new(relay_id());
        relay.set_role(RelayRole::L2UeToUe);

        let decision = relay.forward(&links, ue1(), ue2(), &[0xAA, 0xBB, 0xCC]);
        assert_eq!(
            decision,
            RelayForwardDecision::Forward {
                next_hop: ue2(),
                payload_len: 3
            }
        );
        assert!(decision.is_forwarded());
        assert_eq!(decision.next_hop(), Some(ue2()));
        // Positive, and only reachable through `forward`.
        assert_eq!(relay.forwarded_octets(), 3);
        assert_eq!(relay.forwarded_count(), 1);
        assert_eq!(relay.dropped_no_link(), 0);
    }

    /// A UE that is not a relay does not carry another UE's traffic, and nothing is
    /// counted as forwarded.
    #[test]
    fn a_non_relay_ue_does_not_forward() {
        let links = table_with_active_link_to(relay_id(), ue2());
        let mut not_a_relay = RelayForwarder::new(relay_id());
        assert!(!not_a_relay.is_relaying());

        let decision = not_a_relay.forward(&links, ue1(), ue2(), &[0x01]);
        assert_eq!(decision, RelayForwardDecision::NotARelay);
        assert_eq!(not_a_relay.forwarded_octets(), 0);
    }

    /// A relay with no link to the destination drops, and says so distinguishably from
    /// "not a relay". The old code could express neither.
    #[test]
    fn a_relay_without_a_link_to_the_destination_drops_and_counts_it() {
        // A link to UE-2, but the traffic is for UE-3.
        let links = table_with_active_link_to(relay_id(), ue2());
        let mut relay = RelayForwarder::new(relay_id());
        relay.set_role(RelayRole::L2UeToUe);

        let decision = relay.forward(&links, ue1(), ue3(), &[0x01, 0x02]);
        assert_eq!(
            decision,
            RelayForwardDecision::NoLinkToDestination { destination: ue3() }
        );
        assert_eq!(relay.dropped_no_link(), 1);
        assert_eq!(relay.forwarded_octets(), 0);
    }

    /// Releasing the link stops the forwarding. This is the guard that proves the
    /// forwarding path really consults the link state rather than a peer list that
    /// never changes.
    #[test]
    fn releasing_the_link_stops_the_forwarding() {
        let mut links = table_with_active_link_to(relay_id(), ue2());
        let mut relay = RelayForwarder::new(relay_id());
        relay.set_role(RelayRole::L2UeToUe);

        assert!(relay.forward(&links, ue1(), ue2(), &[0x01]).is_forwarded());

        // Tear the link down through the real release path.
        let _ = links.get_mut(ue2()).expect("present").build_release();

        let decision = relay.forward(&links, ue1(), ue2(), &[0x01]);
        assert_eq!(
            decision,
            RelayForwardDecision::NoLinkToDestination { destination: ue2() }
        );
        // The first forward still counted; the second did not.
        assert_eq!(relay.forwarded_count(), 1);
        assert_eq!(relay.dropped_no_link(), 1);
    }

    /// Traffic for the relay itself is delivered, not forwarded onward — otherwise a
    /// relay would loop its own traffic back out.
    #[test]
    fn traffic_addressed_to_the_relay_itself_is_delivered_not_forwarded() {
        let links = table_with_active_link_to(relay_id(), ue2());
        let mut relay = RelayForwarder::new(relay_id());
        relay.set_role(RelayRole::L2UeToUe);

        let decision = relay.forward(&links, ue1(), relay_id(), &[0x01]);
        assert_eq!(decision, RelayForwardDecision::DeliverLocally);
        assert_eq!(relay.forwarded_octets(), 0);
    }

    #[test]
    fn an_empty_payload_is_not_forwarded() {
        let links = table_with_active_link_to(relay_id(), ue2());
        let mut relay = RelayForwarder::new(relay_id());
        relay.set_role(RelayRole::L2UeToUe);
        assert_eq!(
            relay.forward(&links, ue1(), ue2(), &[]),
            RelayForwardDecision::EmptyPayload
        );
        assert_eq!(relay.forwarded_count(), 0);
    }

    /// Octet counts accumulate across forwards, so an operator reading the counter sees
    /// the traffic rather than the last payload.
    #[test]
    fn forwarded_octets_accumulate_across_payloads() {
        let links = table_with_active_link_to(relay_id(), ue2());
        let mut relay = RelayForwarder::new(relay_id());
        relay.set_role(RelayRole::L2UeToUe);
        relay.forward(&links, ue1(), ue2(), &[0u8; 10]);
        relay.forward(&links, ue1(), ue2(), &[0u8; 25]);
        assert_eq!(relay.forwarded_octets(), 35);
        assert_eq!(relay.forwarded_count(), 2);
    }
}
