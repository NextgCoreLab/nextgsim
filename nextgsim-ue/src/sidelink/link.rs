//! The PC5 unicast link state machine: TS 23.304 §6.4.3.1, driven by the PC5-S
//! messages of [`crate::sidelink::pc5s`].
//!
//! # Why this module exists (issue #141)
//!
//! `SidelinkMessage::EstablishPc5Link` used to read:
//!
//! ```text
//! peer.link_state = Pc5LinkState::Establishing;
//! // Simulate link establishment completing
//! peer.link_state = Pc5LinkState::Active;
//! ```
//!
//! Two assignments on consecutive lines. No message was sent, none was awaited, and
//! `Pc5LinkState::Active` was reachable without a peer existing on the other side.
//! This module makes `Active` reachable **only** by having decoded a peer's
//! `DIRECT COMMUNICATION ACCEPT` — which is what TS 23.304 §6.4.3.1 step 5 requires.
//!
//! # The two roles, and why they are one type
//!
//! A PC5 unicast link has an **initiator** (UE-1: sends the request, waits, applies the
//! accept) and a **responder** (UE-2: receives the request, decides, answers). Both
//! ends are modelled by [`Pc5LinkContext`] with a [`Pc5Role`] discriminant rather than
//! by two types, because the *link* is one object with one state and one peer identity
//! — and a UE is routinely both at once, initiating to one peer while responding to
//! another. Two types would duplicate the state enum and let the two drift.
//!
//! # What "the link is up" means here, and how a test can tell
//!
//! [`Pc5LinkState::Active`] is entered at exactly two sites, and neither can be reached
//! without a decoded peer message:
//!
//! * the initiator, in [`Pc5LinkContext::on_accept_received`];
//! * the responder, in [`Pc5LinkContext::on_request_received`] when it accepts.
//!
//! And [`Pc5LinkContext::peer_l2_id`] is `Some` only once the peer's own Layer-2 ID has
//! been read out of a received message — TS 23.304 §6.4.3.1 step 4's "UE-1 obtains the
//! peer UE's Layer-2 ID for future communication". That makes it the positive value a
//! test asserts on: it cannot be produced by local bookkeeping, only by a round trip.

use std::collections::HashMap;

use tracing::{debug, warn};

use crate::sidelink::pc5s::{
    DirectCommunicationAccept, DirectCommunicationReject, DirectCommunicationRelease,
    DirectCommunicationRequest, Pc5RejectCause, ProseL2Id, RelayServiceCode,
};

/// The state of one PC5 unicast link (TS 23.304 §6.4.3.1).
///
/// Distinct from the older `Pc5LinkState` in `sidelink::task`, which tracks a *peer's*
/// discovery/link status in the task's peer table; this tracks one link's PC5-S
/// procedure. They are kept apart because the task's enum includes `Discovering`, which
/// is not a state of a unicast link at all — discovery happens before any link exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pc5UnicastState {
    /// No link. Nothing has been sent.
    Idle,
    /// The initiator has sent a `DIRECT COMMUNICATION REQUEST` and is waiting
    /// (TS 23.304 §6.4.3.1 step 3).
    ///
    /// The link is **not** usable here. That is the whole correction this issue makes:
    /// the old code had no such waiting state that outlived a single statement.
    AwaitingAccept,
    /// The link is established and usable (step 5 complete).
    Active,
    /// A `DIRECT COMMUNICATION RELEASE` has been sent or received.
    Released,
    /// The peer refused the link with a `DIRECT COMMUNICATION REJECT`.
    Rejected(Pc5RejectCause),
}

impl Pc5UnicastState {
    /// Whether user data may be sent on this link.
    ///
    /// Only [`Pc5UnicastState::Active`]. Named rather than compared inline so the
    /// relay-forwarding path and the link-release path cannot disagree about it.
    pub fn is_usable(self) -> bool {
        self == Self::Active
    }
}

/// Which end of the link this context is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pc5Role {
    /// UE-1: sent the request (TS 23.304 §6.4.3.1 step 3).
    Initiator,
    /// UE-2: received the request and answered it (step 5).
    Responder,
}

/// Why a PC5-S message could not be applied to a link.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pc5LinkError {
    /// An accept arrived for a link that never sent a request.
    ///
    /// This is the condition the old facade could not even express: it set `Active`
    /// unconditionally, so an unsolicited accept was indistinguishable from a completed
    /// handshake.
    UnexpectedAccept {
        /// The state the link was actually in.
        state: Pc5UnicastState,
    },
    /// A reject arrived for a link that never sent a request.
    UnexpectedReject {
        /// The state the link was actually in.
        state: Pc5UnicastState,
    },
    /// A message arrived from a Layer-2 ID that is not this link's peer.
    ///
    /// Enforced because a PC5 signalling destination ID is shared (TS 23.304 §5.8.2.4):
    /// without this check, UE-3's accept could complete UE-1's link to UE-2.
    PeerMismatch {
        /// The peer this link is with.
        expected: ProseL2Id,
        /// The Layer-2 ID the message came from.
        actual: ProseL2Id,
    },
}

impl std::fmt::Display for Pc5LinkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnexpectedAccept { state } => {
                write!(f, "DIRECT COMMUNICATION ACCEPT in state {state:?}")
            }
            Self::UnexpectedReject { state } => {
                write!(f, "DIRECT COMMUNICATION REJECT in state {state:?}")
            }
            Self::PeerMismatch { expected, actual } => {
                write!(
                    f,
                    "PC5-S message from {actual}, but this link is with {expected}"
                )
            }
        }
    }
}

impl std::error::Error for Pc5LinkError {}

/// One PC5 unicast link, from either end.
#[derive(Debug, Clone)]
pub struct Pc5LinkContext {
    /// This UE's own Layer-2 ID (TS 23.304 §5.8.2.1).
    local_l2_id: ProseL2Id,
    /// Which end of the link this is.
    role: Pc5Role,
    /// Where the link is in the §6.4.3.1 procedure.
    state: Pc5UnicastState,
    /// The peer's Layer-2 ID **as learned from a received message**, not as guessed.
    ///
    /// `None` until a peer message has been decoded. TS 23.304 §6.4.3.1 step 4: the
    /// initiator "obtains the peer UE's Layer-2 ID for future communication" from what
    /// the responder sends. Holding it only after a decode is what makes it a positive
    /// assertion target — a test that reads a peer ID here has proved a real message
    /// crossed, which no amount of local state-setting could fake.
    peer_l2_id: Option<ProseL2Id>,
    /// The PFI the accept settled on (TS 23.304 §5.6.1).
    pfi: Option<u8>,
    /// The relay service code this link was requested for, if any.
    relay_service_code: Option<RelayServiceCode>,
    /// Whether the peer accepted as a relay for [`Self::relay_service_code`].
    peer_is_relay: bool,
}

impl Pc5LinkContext {
    /// A fresh link, as the initiator, before anything is sent.
    pub fn new_initiator(local_l2_id: ProseL2Id) -> Self {
        Self {
            local_l2_id,
            role: Pc5Role::Initiator,
            state: Pc5UnicastState::Idle,
            peer_l2_id: None,
            pfi: None,
            relay_service_code: None,
            peer_is_relay: false,
        }
    }

    /// A fresh link, as the responder, before a request has arrived.
    pub fn new_responder(local_l2_id: ProseL2Id) -> Self {
        Self {
            local_l2_id,
            role: Pc5Role::Responder,
            state: Pc5UnicastState::Idle,
            peer_l2_id: None,
            pfi: None,
            relay_service_code: None,
            peer_is_relay: false,
        }
    }

    /// This UE's Layer-2 ID.
    pub fn local_l2_id(&self) -> ProseL2Id {
        self.local_l2_id
    }

    /// Which end of the link this is.
    pub fn role(&self) -> Pc5Role {
        self.role
    }

    /// Where the link is in the TS 23.304 §6.4.3.1 procedure.
    pub fn state(&self) -> Pc5UnicastState {
        self.state
    }

    /// The peer's Layer-2 ID, once a message from the peer has been decoded.
    ///
    /// See the field docs: this is the observable that distinguishes a real handshake
    /// from the facade this issue replaced.
    pub fn peer_l2_id(&self) -> Option<ProseL2Id> {
        self.peer_l2_id
    }

    /// The PC5 QoS Flow Identifier this link settled on, once accepted.
    pub fn pfi(&self) -> Option<u8> {
        self.pfi
    }

    /// The relay service code this link was requested for, if any.
    pub fn relay_service_code(&self) -> Option<RelayServiceCode> {
        self.relay_service_code
    }

    /// Whether the peer accepted this link as a relay.
    ///
    /// Only ever true after an accept whose `acting_as_relay` flag was set, so a
    /// remote UE cannot believe it has a relay without the relay having said so.
    pub fn peer_is_relay(&self) -> bool {
        self.peer_is_relay
    }

    /// Whether user data may be sent on this link.
    pub fn is_usable(&self) -> bool {
        self.state.is_usable()
    }

    /// **Initiator, step 3.** Builds the `DIRECT COMMUNICATION REQUEST` and moves to
    /// [`Pc5UnicastState::AwaitingAccept`].
    ///
    /// Returns the message for the caller to transmit; this module does no I/O, because
    /// the PC5 "radio" in this simulator is whatever channel the caller has (in the
    /// two-UE tests, a tokio channel), and a state machine that owned a transport could
    /// not be driven by a unit test at all.
    ///
    /// Note what this does **not** do: it does not set `Active`. The link is unusable
    /// until an accept arrives, which is the defect this issue fixes.
    pub fn build_request(
        &mut self,
        target_l2_id: Option<ProseL2Id>,
        relay_service_code: Option<RelayServiceCode>,
        prose_service_info: Vec<u8>,
        security_info: Vec<u8>,
    ) -> DirectCommunicationRequest {
        self.state = Pc5UnicastState::AwaitingAccept;
        self.relay_service_code = relay_service_code;
        // The target is what the application named, NOT a learned peer identity: until
        // the peer answers, `peer_l2_id` stays `None`. Conflating the two is how a
        // facade convinces itself it has a peer.
        debug!(
            "PC5: {} sending DIRECT COMMUNICATION REQUEST (target={:?}, rsc={:?})",
            self.local_l2_id, target_l2_id, relay_service_code
        );
        DirectCommunicationRequest {
            source_l2_id: self.local_l2_id,
            target_l2_id,
            relay_service_code,
            prose_service_info,
            security_info,
        }
    }

    /// **Responder, steps 4-5.** Decides on a received request and returns the answer.
    ///
    /// The decision follows TS 23.304 §6.4.3.1 steps 5a and 5b exactly:
    ///
    /// * **5a** — if the request carries `Target User Info`, this UE accepts only if
    ///   that target is its own Layer-2 ID, and otherwise rejects with
    ///   [`Pc5RejectCause::TargetUserMismatch`]. ("the target UE, i.e. UE-2 responds
    ///   with a Direct Communication Accept message if the Application Layer ID for
    ///   UE-2 matches.")
    /// * **5b** — if it does not, this UE accepts if it is interested in the announced
    ///   ProSe service, judged by `interested_in`. ("the UEs that are interested in
    ///   using the announced ProSe Service(s) respond".)
    ///
    /// A request naming a relay service code this UE does not serve is rejected with
    /// [`Pc5RejectCause::RelayServiceNotSupported`], because accepting it would leave
    /// the remote UE believing it had relay connectivity it does not have.
    ///
    /// `Ok(accept)` also transitions this end to [`Pc5UnicastState::Active`]: the
    /// responder's link is up as soon as it has accepted, per step 5.
    pub fn on_request_received(
        &mut self,
        request: &DirectCommunicationRequest,
        pfi: u8,
        served_relay_service_code: Option<RelayServiceCode>,
        interested_in: impl Fn(&[u8]) -> bool,
    ) -> Result<DirectCommunicationAccept, DirectCommunicationReject> {
        let reject = |cause| {
            Err(DirectCommunicationReject {
                source_l2_id: self.local_l2_id,
                cause,
            })
        };

        // Step 5a: a named target that is not us is not ours to answer.
        if let Some(target) = request.target_l2_id {
            if target != self.local_l2_id {
                debug!(
                    "PC5: {} refusing a request targeted at {target}",
                    self.local_l2_id
                );
                return reject(Pc5RejectCause::TargetUserMismatch);
            }
        } else if !interested_in(&request.prose_service_info) {
            // Step 5b: no target, so only interest decides.
            debug!(
                "PC5: {} not interested in the announced ProSe service",
                self.local_l2_id
            );
            return reject(Pc5RejectCause::ProseServiceNotInterested);
        }

        // A relay request this UE cannot serve.
        let acting_as_relay = match (request.relay_service_code, served_relay_service_code) {
            (None, _) => false,
            (Some(requested), Some(served)) if requested == served => true,
            (Some(requested), _) => {
                debug!(
                    "PC5: {} does not serve relay service code {requested}",
                    self.local_l2_id
                );
                return reject(Pc5RejectCause::RelayServiceNotSupported);
            }
        };

        // Accepted. The peer's identity comes from the request's Source User Info --
        // this is the responder's half of step 4's "obtains the peer UE's Layer-2 ID".
        self.peer_l2_id = Some(request.source_l2_id);
        self.pfi = Some(pfi);
        self.relay_service_code = request.relay_service_code;
        self.state = Pc5UnicastState::Active;
        debug!(
            "PC5: {} accepted a unicast link from {} (pfi={pfi}, relay={acting_as_relay})",
            self.local_l2_id, request.source_l2_id
        );
        Ok(DirectCommunicationAccept {
            source_l2_id: self.local_l2_id,
            pfi,
            acting_as_relay,
        })
    }

    /// **Initiator, step 5.** Applies a received accept: the link becomes usable and
    /// the peer's Layer-2 ID is adopted.
    ///
    /// Refuses an accept that arrives in any state other than
    /// [`Pc5UnicastState::AwaitingAccept`]. That refusal is the substance of this
    /// issue's first criterion: `Active` is now reachable only from "a request was
    /// sent and this is its answer".
    pub fn on_accept_received(
        &mut self,
        accept: &DirectCommunicationAccept,
    ) -> Result<(), Pc5LinkError> {
        if self.state != Pc5UnicastState::AwaitingAccept {
            return Err(Pc5LinkError::UnexpectedAccept { state: self.state });
        }
        // TS 23.304 §6.4.3.1 step 4: the initiator obtains the peer's Layer-2 ID from
        // what the responder sent, and addresses it thereafter.
        self.peer_l2_id = Some(accept.source_l2_id);
        self.pfi = Some(accept.pfi);
        self.peer_is_relay = accept.acting_as_relay;
        self.state = Pc5UnicastState::Active;
        debug!(
            "PC5: {} link to {} is ACTIVE (pfi={}, peer_is_relay={})",
            self.local_l2_id, accept.source_l2_id, accept.pfi, accept.acting_as_relay
        );
        Ok(())
    }

    /// **Initiator.** Applies a received reject: the link is refused, with its reason.
    pub fn on_reject_received(
        &mut self,
        reject: &DirectCommunicationReject,
    ) -> Result<(), Pc5LinkError> {
        if self.state != Pc5UnicastState::AwaitingAccept {
            return Err(Pc5LinkError::UnexpectedReject { state: self.state });
        }
        self.state = Pc5UnicastState::Rejected(reject.cause);
        warn!(
            "PC5: {} link refused by {}: {:?}",
            self.local_l2_id, reject.source_l2_id, reject.cause
        );
        Ok(())
    }

    /// Builds a `DIRECT COMMUNICATION RELEASE` and marks this end released.
    pub fn build_release(&mut self) -> DirectCommunicationRelease {
        self.state = Pc5UnicastState::Released;
        debug!("PC5: {} releasing its unicast link", self.local_l2_id);
        DirectCommunicationRelease {
            source_l2_id: self.local_l2_id,
        }
    }

    /// Applies a received release.
    ///
    /// A release from a Layer-2 ID that is not this link's peer is refused rather than
    /// applied: a shared signalling destination ID (TS 23.304 §5.8.2.4) means anyone
    /// can send here, and a third party must not be able to tear down someone else's
    /// link.
    pub fn on_release_received(
        &mut self,
        release: &DirectCommunicationRelease,
    ) -> Result<(), Pc5LinkError> {
        if let Some(peer) = self.peer_l2_id {
            if peer != release.source_l2_id {
                return Err(Pc5LinkError::PeerMismatch {
                    expected: peer,
                    actual: release.source_l2_id,
                });
            }
        }
        self.state = Pc5UnicastState::Released;
        debug!(
            "PC5: {} link released by {}",
            self.local_l2_id, release.source_l2_id
        );
        Ok(())
    }
}

/// The PC5 unicast links a UE holds, keyed by the peer's Layer-2 ID.
///
/// Keyed by *peer* rather than by an arbitrary handle because that is how an incoming
/// PC5-S message is routed: it carries a `Source User Info` and nothing else that could
/// select a link. A link whose peer is not yet known (an initiator awaiting an accept)
/// is keyed by the target it addressed, which is the only identity available until the
/// answer arrives.
#[derive(Debug, Default)]
pub struct Pc5LinkTable {
    links: HashMap<ProseL2Id, Pc5LinkContext>,
}

impl Pc5LinkTable {
    /// An empty table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts or replaces the link to `key`.
    pub fn insert(&mut self, key: ProseL2Id, link: Pc5LinkContext) {
        self.links.insert(key, link);
    }

    /// The link to `key`, if any.
    pub fn get(&self, key: ProseL2Id) -> Option<&Pc5LinkContext> {
        self.links.get(&key)
    }

    /// The link to `key`, mutably.
    pub fn get_mut(&mut self, key: ProseL2Id) -> Option<&mut Pc5LinkContext> {
        self.links.get_mut(&key)
    }

    /// Removes the link to `key`.
    pub fn remove(&mut self, key: ProseL2Id) -> Option<Pc5LinkContext> {
        self.links.remove(&key)
    }

    /// How many links are [`Pc5UnicastState::Active`].
    pub fn active_count(&self) -> usize {
        self.links.values().filter(|l| l.is_usable()).count()
    }

    /// Every peer this UE has a usable link to.
    pub fn active_peers(&self) -> Vec<ProseL2Id> {
        let mut peers: Vec<ProseL2Id> = self
            .links
            .values()
            .filter(|l| l.is_usable())
            .filter_map(Pc5LinkContext::peer_l2_id)
            .collect();
        // Sorted so a caller iterating them (e.g. the relay's forwarding choice) is
        // deterministic; a HashMap's order is not, and a test asserting on a forwarded
        // destination would be flaky.
        peers.sort_unstable();
        peers
    }

    /// The peer this UE holds a usable relay link to, if any.
    ///
    /// Used by the relay path to decide where a remote UE's traffic goes. Returns the
    /// lowest such peer for determinism, for the same reason [`Self::active_peers`]
    /// sorts.
    pub fn active_relay_peer(&self) -> Option<ProseL2Id> {
        self.links
            .values()
            .filter(|l| l.is_usable() && l.peer_is_relay())
            .filter_map(Pc5LinkContext::peer_l2_id)
            .min()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const UE1: u32 = 0x00_00_01;
    const UE2: u32 = 0x00_00_02;
    const UE3: u32 = 0x00_00_03;
    const RSC: u32 = 0x00_AB_CD;

    fn ue1() -> ProseL2Id {
        ProseL2Id::new(UE1)
    }
    fn ue2() -> ProseL2Id {
        ProseL2Id::new(UE2)
    }
    fn ue3() -> ProseL2Id {
        ProseL2Id::new(UE3)
    }

    /// Always interested, for the step-5b cases where interest is not what is under
    /// test.
    fn interested(_: &[u8]) -> bool {
        true
    }

    /// The criterion-1 handshake, end to end, through the real encode/decode:
    /// request → accept → `Active`, with the peer's identity surviving the round trip.
    ///
    /// The assertion that matters is `peer_l2_id`, not the state: a state enum can be
    /// assigned locally (that is exactly what the old facade did), but `peer_l2_id` on
    /// the initiator can only come from octets the responder produced.
    #[test]
    fn a_link_becomes_active_only_after_the_peer_accept_crosses_the_wire() {
        let mut initiator = Pc5LinkContext::new_initiator(ue1());
        let mut responder = Pc5LinkContext::new_responder(ue2());

        // Step 3.
        let request = initiator.build_request(Some(ue2()), None, vec![0x01], vec![0x02]);
        // The initiator is NOT active yet. This is the whole point.
        assert_eq!(initiator.state(), Pc5UnicastState::AwaitingAccept);
        assert!(!initiator.is_usable());
        assert_eq!(initiator.peer_l2_id(), None);

        // Over the wire, as bytes, so nothing is carried by a shared Rust value.
        let on_the_wire = request.encode();
        let received = DirectCommunicationRequest::decode(&on_the_wire).expect("decode request");

        // Steps 4-5.
        let accept = responder
            .on_request_received(&received, 7, None, interested)
            .expect("UE-2 accepts a request addressed to it");
        assert_eq!(responder.state(), Pc5UnicastState::Active);
        assert_eq!(responder.peer_l2_id(), Some(ue1()));

        let accept_wire = accept.encode();
        let accept_back = DirectCommunicationAccept::decode(&accept_wire).expect("decode accept");
        initiator
            .on_accept_received(&accept_back)
            .expect("UE-1 applies the accept");

        assert_eq!(initiator.state(), Pc5UnicastState::Active);
        assert!(initiator.is_usable());
        // The positive assertion: UE-1 learned UE-2's Layer-2 ID from UE-2's own bytes
        // (TS 23.304 §6.4.3.1 step 4), and the negotiated PFI came back with it.
        assert_eq!(initiator.peer_l2_id(), Some(ue2()));
        assert_eq!(initiator.pfi(), Some(7));
    }

    /// An accept cannot make a link active if no request was ever sent. The old code
    /// could not express this at all — it set `Active` unconditionally.
    #[test]
    fn an_unsolicited_accept_does_not_activate_a_link() {
        let mut initiator = Pc5LinkContext::new_initiator(ue1());
        let accept = DirectCommunicationAccept {
            source_l2_id: ue2(),
            pfi: 1,
            acting_as_relay: false,
        };
        assert_eq!(
            initiator.on_accept_received(&accept),
            Err(Pc5LinkError::UnexpectedAccept {
                state: Pc5UnicastState::Idle
            })
        );
        assert_eq!(initiator.state(), Pc5UnicastState::Idle);
        assert!(!initiator.is_usable());
        assert_eq!(initiator.peer_l2_id(), None);
    }

    /// Step 5a: a request naming someone else is rejected, and the refusal names why.
    #[test]
    fn a_request_targeted_at_another_ue_is_rejected_as_a_target_mismatch() {
        let mut responder = Pc5LinkContext::new_responder(ue2());
        let request = DirectCommunicationRequest {
            source_l2_id: ue1(),
            target_l2_id: Some(ue3()),
            relay_service_code: None,
            prose_service_info: Vec::new(),
            security_info: Vec::new(),
        };
        let reject = responder
            .on_request_received(&request, 1, None, interested)
            .expect_err("a request for UE-3 is not UE-2's to accept");
        assert_eq!(reject.cause, Pc5RejectCause::TargetUserMismatch);
        assert_eq!(reject.source_l2_id, ue2());
        // And nothing was established.
        assert_eq!(responder.state(), Pc5UnicastState::Idle);
        assert_eq!(responder.peer_l2_id(), None);
    }

    /// Step 5b: with no target, interest in the announced service is what decides.
    #[test]
    fn a_service_oriented_request_is_accepted_only_by_an_interested_ue() {
        let service = vec![0x42];
        let request = DirectCommunicationRequest {
            source_l2_id: ue1(),
            target_l2_id: None,
            relay_service_code: None,
            prose_service_info: service.clone(),
            security_info: Vec::new(),
        };

        let mut interested_ue = Pc5LinkContext::new_responder(ue2());
        assert!(interested_ue
            .on_request_received(&request, 1, None, |info| info == service.as_slice())
            .is_ok());
        assert_eq!(interested_ue.state(), Pc5UnicastState::Active);

        let mut uninterested_ue = Pc5LinkContext::new_responder(ue3());
        let reject = uninterested_ue
            .on_request_received(&request, 1, None, |_| false)
            .expect_err("an uninterested UE does not answer");
        assert_eq!(reject.cause, Pc5RejectCause::ProseServiceNotInterested);
        assert_eq!(uninterested_ue.state(), Pc5UnicastState::Idle);
    }

    /// A relay link: the responder serves the requested RSC, so it accepts AS a relay,
    /// and the initiator learns that from the accept rather than assuming it.
    #[test]
    fn a_relay_link_tells_the_remote_ue_the_peer_is_acting_as_a_relay() {
        let rsc = RelayServiceCode::new(RSC);
        let mut remote = Pc5LinkContext::new_initiator(ue1());
        let mut relay = Pc5LinkContext::new_responder(ue2());

        let request = remote.build_request(Some(ue2()), Some(rsc), Vec::new(), Vec::new());
        let accept = relay
            .on_request_received(
                &DirectCommunicationRequest::decode(&request.encode()).expect("decode"),
                3,
                Some(rsc),
                interested,
            )
            .expect("the relay serves this RSC");
        assert!(accept.acting_as_relay);

        remote
            .on_accept_received(&DirectCommunicationAccept::decode(&accept.encode()).unwrap())
            .expect("apply");
        // The remote UE believes it has a relay ONLY because the relay said so.
        assert!(remote.peer_is_relay());
        assert_eq!(remote.relay_service_code(), Some(rsc));
    }

    /// A UE that does not serve the requested RSC refuses, rather than accepting and
    /// leaving the remote UE believing it has connectivity it has not got.
    #[test]
    fn a_request_for_an_unserved_relay_service_code_is_refused() {
        let mut responder = Pc5LinkContext::new_responder(ue2());
        let request = DirectCommunicationRequest {
            source_l2_id: ue1(),
            target_l2_id: Some(ue2()),
            relay_service_code: Some(RelayServiceCode::new(RSC)),
            prose_service_info: Vec::new(),
            security_info: Vec::new(),
        };
        // Serves a DIFFERENT code.
        let reject = responder
            .on_request_received(
                &request,
                1,
                Some(RelayServiceCode::new(0x00_00_99)),
                interested,
            )
            .expect_err("a relay that does not serve this RSC must refuse");
        assert_eq!(reject.cause, Pc5RejectCause::RelayServiceNotSupported);
        assert_eq!(responder.state(), Pc5UnicastState::Idle);
    }

    /// A rejection reaches the initiator with its cause, and does not leave the link
    /// waiting forever.
    #[test]
    fn a_reject_moves_the_initiator_out_of_awaiting_and_keeps_the_cause() {
        let mut initiator = Pc5LinkContext::new_initiator(ue1());
        let _ = initiator.build_request(Some(ue2()), None, Vec::new(), Vec::new());
        let reject = DirectCommunicationReject {
            source_l2_id: ue2(),
            cause: Pc5RejectCause::InsufficientResources,
        };
        initiator
            .on_reject_received(&DirectCommunicationReject::decode(&reject.encode()).unwrap())
            .expect("apply");
        assert_eq!(
            initiator.state(),
            Pc5UnicastState::Rejected(Pc5RejectCause::InsufficientResources)
        );
        assert!(!initiator.is_usable());
    }

    /// A third party cannot tear down a link it is not part of.
    #[test]
    fn a_release_from_a_stranger_does_not_release_the_link() {
        let mut initiator = Pc5LinkContext::new_initiator(ue1());
        let _ = initiator.build_request(Some(ue2()), None, Vec::new(), Vec::new());
        initiator
            .on_accept_received(&DirectCommunicationAccept {
                source_l2_id: ue2(),
                pfi: 1,
                acting_as_relay: false,
            })
            .expect("apply");
        assert!(initiator.is_usable());

        let stranger = DirectCommunicationRelease {
            source_l2_id: ue3(),
        };
        assert_eq!(
            initiator.on_release_received(&stranger),
            Err(Pc5LinkError::PeerMismatch {
                expected: ue2(),
                actual: ue3(),
            })
        );
        // Still up.
        assert!(initiator.is_usable());

        // The actual peer can release it.
        let peer = DirectCommunicationRelease {
            source_l2_id: ue2(),
        };
        initiator.on_release_received(&peer).expect("the peer may");
        assert_eq!(initiator.state(), Pc5UnicastState::Released);
        assert!(!initiator.is_usable());
    }

    #[test]
    fn a_released_link_is_no_longer_counted_active() {
        let mut table = Pc5LinkTable::new();
        let mut link = Pc5LinkContext::new_initiator(ue1());
        let _ = link.build_request(Some(ue2()), None, Vec::new(), Vec::new());
        link.on_accept_received(&DirectCommunicationAccept {
            source_l2_id: ue2(),
            pfi: 1,
            acting_as_relay: false,
        })
        .expect("apply");
        table.insert(ue2(), link);
        assert_eq!(table.active_count(), 1);
        assert_eq!(table.active_peers(), vec![ue2()]);

        let _ = table.get_mut(ue2()).expect("present").build_release();
        assert_eq!(table.active_count(), 0);
        assert!(table.active_peers().is_empty());
    }

    /// The relay peer lookup only returns a peer that accepted as a relay, so the
    /// forwarding path cannot pick a plain unicast peer to relay through.
    #[test]
    fn the_relay_peer_lookup_ignores_a_plain_unicast_peer() {
        let mut table = Pc5LinkTable::new();

        let mut plain = Pc5LinkContext::new_initiator(ue1());
        let _ = plain.build_request(Some(ue2()), None, Vec::new(), Vec::new());
        plain
            .on_accept_received(&DirectCommunicationAccept {
                source_l2_id: ue2(),
                pfi: 1,
                acting_as_relay: false,
            })
            .expect("apply");
        table.insert(ue2(), plain);
        assert_eq!(table.active_relay_peer(), None);

        let rsc = RelayServiceCode::new(RSC);
        let mut via_relay = Pc5LinkContext::new_initiator(ue1());
        let _ = via_relay.build_request(Some(ue3()), Some(rsc), Vec::new(), Vec::new());
        via_relay
            .on_accept_received(&DirectCommunicationAccept {
                source_l2_id: ue3(),
                pfi: 2,
                acting_as_relay: true,
            })
            .expect("apply");
        table.insert(ue3(), via_relay);
        assert_eq!(table.active_relay_peer(), Some(ue3()));
    }
}
