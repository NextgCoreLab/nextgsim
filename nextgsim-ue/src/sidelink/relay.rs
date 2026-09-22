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
//! # Two relay architectures, and how the forwarding decision differs
//!
//! **UE-to-UE relay** (TS 23.304 §6.4.3.10, issue #141): UE-1 → relay → UE-2, all three
//! over PC5. The relay forwards onto *another PC5 link*, so the decision turns on whether
//! it holds a usable link to the **destination**.
//!
//! **UE-to-Network relay** (TS 23.304 §5.4.2, TS 38.300 §16.12.2.1, issue #190): a remote
//! UE reaching the 5GC *through* the relay's own Uu connection. The relay does not forward
//! onto a PC5 link at all — it adapts the traffic through the SRAP sublayer
//! (`nextgsim_rlc::srap`) and submits it to one of its own Uu relay RLC channels. So the
//! link-table check that is right for UE-to-UE relaying is *wrong* here, and would drop
//! every packet: the destination is the network, which is not a PC5 peer. What is checked
//! instead is that the **source** is a remote UE this relay serves. See
//! [`RelayRole::is_ue_to_network`], which is where that fork is named.
//!
//! Issue #141 scoped its criterion to "a two-UE relay forwards data" and mapped
//! `RelayMode::UeToNetworkRelay` to [`RelayRole::None`] with a `warn!`, because the SRAP
//! layer did not exist. Issue #190 built it, so the variant is real and the `warn!` is
//! gone. **L3 relay** remains unimplemented and still warns: it forwards IP packets rather
//! than adapting Layer-2 bearers, and there is no IP forwarding plane in this tree.
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

/// Which relay role this UE is playing (TS 23.304 §5.4.2, TS 38.300 §16.9, §16.12).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelayRole {
    /// Not relaying. Traffic that is not for this UE is dropped.
    None,
    /// Layer-2 UE-to-UE relay: forwards PC5 traffic between two remote UEs
    /// (TS 23.304 §6.4.3.10, TS 38.300 §16.9).
    L2UeToUe,
    /// Layer-2 UE-to-Network relay: carries a remote UE's end-to-end Uu bearers to the
    /// gNB across its own Uu connection (TS 23.304 §5.4.2, TS 38.300 §16.12.2.1;
    /// issue #190).
    ///
    /// Distinct from [`Self::L2UeToUe`] in **where the traffic goes**, which is why it is a
    /// separate variant rather than a flag: a UE-to-UE relay forwards onto another PC5 link,
    /// while a UE-to-Network relay adapts the traffic through the SRAP sublayer
    /// (`nextgsim_rlc::srap`) and submits it to one of its own Uu relay RLC channels. The
    /// destination is the network, not a peer.
    ///
    /// Until issue #190 this variant did not exist and `RelayMode::UeToNetworkRelay` mapped
    /// to [`Self::None`] with a `warn!`, because the SRAP layer it needs was absent.
    L2UeToNetwork,
}

impl RelayRole {
    /// Whether this role carries traffic towards the **network** rather than towards
    /// another PC5 peer (TS 38.300 §16.12.2.1).
    ///
    /// Named because the forwarding decision differs on exactly this: a UE-to-Network relay
    /// has no PC5 link to the "destination" — the destination is the 5GC — so the
    /// link-table check that is correct for UE-to-UE relaying would drop every packet.
    pub fn is_ue_to_network(self) -> bool {
        matches!(self, Self::L2UeToNetwork)
    }
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
    /// Adapt `payload_len` octets from the remote UE towards the **network**, through the
    /// SRAP sublayer (TS 38.300 §16.12.2.1; issue #190).
    ///
    /// The uplink direction of L2 UE-to-Network relaying. Distinct from [`Self::Forward`]
    /// because there is no `next_hop`: the traffic leaves on one of this relay's own Uu
    /// relay RLC channels rather than on a PC5 link to a peer, so a decision carrying a
    /// peer Layer-2 ID would name a hop that does not exist.
    AdaptToNetwork {
        /// The remote UE the payload came from, by PC5 Layer-2 ID.
        ///
        /// Carried so the caller can look up the local Remote UE ID the gNB assigned it —
        /// the value that goes in the SRAP header.
        remote_l2_id: ProseL2Id,
        /// How many octets are adapted.
        payload_len: usize,
    },
    /// This UE is a UE-to-Network relay, but the source is not a remote UE it serves.
    ///
    /// Distinguished from [`Self::NoLinkToDestination`] because the missing thing is
    /// different: there a *destination* was unreachable, here a *source* is not one this
    /// relay carries traffic for. A relay that adapted traffic from an unserved UE would
    /// give the gNB a SRAP header naming a remote UE it has assigned no identity to.
    NotAServedRemoteUe {
        /// The source that is not served here.
        source: ProseL2Id,
    },
}

impl RelayForwardDecision {
    /// Whether the payload is being forwarded onto another PC5 link.
    pub fn is_forwarded(&self) -> bool {
        matches!(self, Self::Forward { .. })
    }

    /// Whether the payload is being adapted towards the network (issue #190).
    pub fn is_adapted_to_network(&self) -> bool {
        matches!(self, Self::AdaptToNetwork { .. })
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

        // **UE-to-Network relaying diverges here** (issue #190, TS 38.300 §16.12.2.1). The
        // destination is the 5GC, which this relay reaches over its own Uu connection — so
        // there is no PC5 link to the destination to check, and the link-table test below
        // would drop every packet. What matters instead is that the SOURCE is a remote UE
        // this relay actually serves: the relay holds a live PC5 link to it, and the gNB has
        // assigned it a local Remote UE ID.
        if self.role.is_ue_to_network() {
            if !links.active_peers().contains(&source) {
                self.dropped_no_link += 1;
                warn!(
                    "PC5 relay: {} holds no usable PC5 link to {source}, so it is not a \
                     served remote UE; dropping {} octet(s) bound for the network",
                    self.local_l2_id,
                    payload.len()
                );
                return RelayForwardDecision::NotAServedRemoteUe { source };
            }
            self.forwarded_octets += payload.len() as u64;
            self.forwarded_count += 1;
            debug!(
                "PC5 relay: {} adapting {} octet(s) from remote UE {source} towards the \
                 network",
                self.local_l2_id,
                payload.len()
            );
            return RelayForwardDecision::AdaptToNetwork {
                remote_l2_id: source,
                payload_len: payload.len(),
            };
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

    // ── Issue #190: UE-to-Network relaying ───────────────────────────────────────

    /// A UE-to-Network relay adapts a served remote UE's traffic towards the network, and
    /// the decision names the remote UE rather than a next hop.
    ///
    /// The `AdaptToNetwork` variant is the positive observable: a UE-to-UE relay could
    /// never produce it, so this cannot pass against the #141 forwarding path.
    #[test]
    fn a_ue_to_network_relay_adapts_a_served_remote_ues_traffic_towards_the_network() {
        // A link to UE-1, which is the remote UE this relay serves.
        let links = table_with_active_link_to(relay_id(), ue1());
        let mut relay = RelayForwarder::new(relay_id());
        relay.set_role(RelayRole::L2UeToNetwork);
        assert!(relay.role().is_ue_to_network());

        // The destination is a network address, NOT a PC5 peer -- and deliberately one the
        // relay holds no link to, because that is the whole point: a UE-to-UE relay would
        // drop this.
        let decision = relay.forward(&links, ue1(), ue3(), &[0xAA, 0xBB, 0xCC, 0xDD]);
        assert_eq!(
            decision,
            RelayForwardDecision::AdaptToNetwork {
                remote_l2_id: ue1(),
                payload_len: 4,
            }
        );
        assert!(decision.is_adapted_to_network());
        assert!(
            !decision.is_forwarded(),
            "adapting to the network is not forwarding onto a PC5 link"
        );
        assert_eq!(relay.forwarded_octets(), 4);
        assert_eq!(relay.dropped_no_link(), 0);
    }

    /// The same payload, on a UE-to-**UE** relay, is DROPPED — because there is no link to
    /// the destination.
    ///
    /// This is the contrast that makes the test above load-bearing: it proves the two roles
    /// take genuinely different decisions rather than the new variant being cosmetic.
    #[test]
    fn a_ue_to_ue_relay_drops_what_a_ue_to_network_relay_adapts() {
        let links = table_with_active_link_to(relay_id(), ue1());
        let mut relay = RelayForwarder::new(relay_id());
        relay.set_role(RelayRole::L2UeToUe);
        assert!(!relay.role().is_ue_to_network());

        assert_eq!(
            relay.forward(&links, ue1(), ue3(), &[0xAA; 4]),
            RelayForwardDecision::NoLinkToDestination { destination: ue3() }
        );
        assert_eq!(relay.forwarded_octets(), 0);
    }

    /// A UE-to-Network relay refuses traffic from a UE it does NOT serve: the gNB has
    /// assigned no local Remote UE ID for it, so there is no SRAP header to write.
    #[test]
    fn a_ue_to_network_relay_refuses_an_unserved_source() {
        // A link to UE-1 only; the traffic claims to come from UE-2.
        let links = table_with_active_link_to(relay_id(), ue1());
        let mut relay = RelayForwarder::new(relay_id());
        relay.set_role(RelayRole::L2UeToNetwork);

        let decision = relay.forward(&links, ue2(), ue3(), &[0xAA; 4]);
        assert_eq!(
            decision,
            RelayForwardDecision::NotAServedRemoteUe { source: ue2() }
        );
        assert_eq!(relay.forwarded_octets(), 0);
        assert_eq!(relay.dropped_no_link(), 1);
    }

    /// Releasing the remote UE's PC5 link stops the adaptation. The guard that proves the
    /// UE-to-Network path consults the LIVE link state, not a peer list recorded once.
    #[test]
    fn releasing_the_remote_ues_link_stops_the_adaptation() {
        let mut links = table_with_active_link_to(relay_id(), ue1());
        let mut relay = RelayForwarder::new(relay_id());
        relay.set_role(RelayRole::L2UeToNetwork);

        assert!(relay
            .forward(&links, ue1(), ue3(), &[0xAA; 4])
            .is_adapted_to_network());

        let _ = links.get_mut(ue1()).expect("present").build_release();

        assert_eq!(
            relay.forward(&links, ue1(), ue3(), &[0xAA; 4]),
            RelayForwardDecision::NotAServedRemoteUe { source: ue1() }
        );
        // The first adaptation still counted; the second did not.
        assert_eq!(relay.forwarded_count(), 1);
    }

    /// A UE that is not a relay at all adapts nothing, even towards the network.
    #[test]
    fn a_non_relay_ue_does_not_adapt_towards_the_network() {
        let links = table_with_active_link_to(relay_id(), ue1());
        let mut not_a_relay = RelayForwarder::new(relay_id());
        assert_eq!(
            not_a_relay.forward(&links, ue1(), ue3(), &[0xAA; 4]),
            RelayForwardDecision::NotARelay
        );
        assert_eq!(not_a_relay.forwarded_octets(), 0);
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
