//! UUAA-MM: UAS authentication and authorization over NAS transport
//! (TS 23.256 §5.2.2, TS 24.501 §9.11.2.10)
//!
//! During UUAA-MM the AMF relays authentication messages transparently between
//! the UE and the UAS NF / USS. On the NAS leg they travel as a Service-level-AA
//! container inside the Payload container of UL/DL NAS TRANSPORT, selected by
//! payload container type `0b1001` (TS 24.501 §9.11.3.40).
//!
//! This module owns the UE half of that exchange: when to send, what to send,
//! and what a network answer means for the UAV's authorization state. The bytes
//! themselves belong to `nextgsim_nas::ies::service_level_aa`, which both this
//! path and the registration path share.
//!
//! ## What this models, and what it does not
//!
//! The Service-level-AA *payload* is opaque to NAS: TS 23.256 puts the actual
//! credential exchange in the application layer, between the UE's UAS
//! application and the USS. This simulator has no USS application, so the
//! payload it sends is a configured opaque blob and the exchange proves the NAS
//! transport, not an authentication. That distinction is why
//! [`UuaaProcedure::request_payload`] takes bytes from configuration rather than
//! deriving anything: a fabricated credential would look like a working UUAA
//! while authenticating nothing.

use nextgsim_nas::ies::ie1::PayloadContainerType;
use nextgsim_nas::ies::service_level_aa::{ServiceLevelAaContainer, ServiceLevelAaPayloadType};
use nextgsim_nas::messages::mm::UlNasTransport;
use tracing::{info, warn};

use crate::uav::UavAuthorizationState;

/// Where the UE is in the UUAA-MM exchange (TS 23.256 §5.2.2.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UuaaState {
    /// No exchange has been asked for or started.
    #[default]
    Idle,
    /// The network signalled a Service-level-AA pending indication, so the
    /// procedure is to be performed but the UE has not sent its payload yet.
    Pending,
    /// The UE has sent a UUAA payload and is waiting for the network's result.
    Requested,
    /// The network reported the service-level authentication and authorization
    /// succeeded.
    Authorized,
    /// The network reported failure, or revoked a previous authorization.
    Rejected,
}

/// What the caller should do after a Service-level-AA container arrives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UuaaReaction {
    /// Nothing to send. The container carried no parameter this UE acts on --
    /// which includes a container that only re-states a result already applied.
    Nothing,
    /// Send this UL NAS TRANSPORT PDU (plain, not yet security-protected): the
    /// network asked for the procedure with a pending indication and the UE has
    /// a payload to answer with.
    SendUplink(Vec<u8>),
}

/// The UE side of UUAA-MM.
///
/// Deliberately holds no timer. TS 24.501 gives the UUAA payload exchange no
/// UE-side NAS timer of its own -- the AMF drives the procedure and a UE that
/// invented a timeout would abandon a live exchange the network is still
/// working on. A UE stuck in [`UuaaState::Requested`] is therefore visible in
/// the state rather than papered over by an expiry.
#[derive(Debug, Clone, Default)]
pub struct UuaaProcedure {
    state: UuaaState,
    /// The opaque UUAA payload this UE sends when asked, from configuration.
    /// `None` means the UE cannot answer a pending indication, which is
    /// reported rather than answered with an empty payload.
    request_payload: Option<Vec<u8>>,
    /// Whether the network has said UAS services are enabled
    /// (TS 24.501 §9.11.2.18). Distinct from being authorized: a network can
    /// authorize a UAV and still not have UAS services enabled.
    uas_services_enabled: Option<bool>,
    /// How many UUAA payloads the UE has sent, so a caller can tell a
    /// never-started exchange from one the network has not answered.
    uplinks_sent: u32,
    /// Whether this UE may put a UUAA payload on the wire (the `uuaa-mm`
    /// feature). Decoding is never gated -- refusing to understand a conformant
    /// container the network sent is worse than understanding it -- but
    /// *originating* the exchange is, so the default build's registration-marker
    /// behaviour is unchanged.
    initiation_enabled: bool,
}

impl UuaaProcedure {
    /// Create the procedure with the opaque UUAA payload the UE will send when
    /// the network asks for the exchange, and with uplink initiation enabled.
    pub fn new(request_payload: Option<Vec<u8>>) -> Self {
        Self::with_initiation(request_payload, true)
    }

    /// Create the procedure, choosing whether it may originate uplinks.
    ///
    /// The caller passes `cfg!(feature = "uuaa-mm")` so both configurations
    /// compile and both are reachable from tests; a `#[cfg]` on the behaviour
    /// itself would leave the disabled path unbuilt and untested.
    pub fn with_initiation(request_payload: Option<Vec<u8>>, initiation_enabled: bool) -> Self {
        Self {
            state: UuaaState::Idle,
            request_payload,
            uas_services_enabled: None,
            uplinks_sent: 0,
            initiation_enabled,
        }
    }

    /// Whether this UE may originate a UUAA uplink.
    pub fn initiation_enabled(&self) -> bool {
        self.initiation_enabled
    }

    /// Current state of the exchange.
    pub fn state(&self) -> UuaaState {
        self.state
    }

    /// Whether the network has enabled UAS services, when it has said either way.
    pub fn uas_services_enabled(&self) -> Option<bool> {
        self.uas_services_enabled
    }

    /// How many UUAA payloads have been sent.
    pub fn uplinks_sent(&self) -> u32 {
        self.uplinks_sent
    }

    /// The UAV authorization state this exchange implies, for
    /// [`crate::uav::UavContext::auth_state`].
    ///
    /// `Requested` and `Pending` both map to `AuthorizationRequested`: from the
    /// aircraft's point of view an unanswered request and an unstarted one are
    /// the same non-authorization, and only an explicit success authorizes.
    pub fn authorization_state(&self) -> UavAuthorizationState {
        match self.state {
            UuaaState::Idle => UavAuthorizationState::NotAuthorized,
            UuaaState::Pending | UuaaState::Requested => {
                UavAuthorizationState::AuthorizationRequested
            }
            UuaaState::Authorized => UavAuthorizationState::Authorized,
            // A rejection *after* an authorization is a revocation, and a
            // rejection before one is simply a refusal. Both are modelled as
            // Revoked because the RRC-level enum has no separate "refused"
            // state and treating a refusal as merely NotAuthorized would let a
            // retry look like a fresh start.
            UuaaState::Rejected => UavAuthorizationState::Revoked,
        }
    }

    /// Build the UL NAS TRANSPORT carrying this UE's UUAA payload
    /// (TS 24.501 §8.2.10, payload container type `0b1001`).
    ///
    /// Returns the plain NAS PDU; the caller applies NAS security. `None` when
    /// no payload is configured or uplink initiation is disabled -- there is
    /// nothing honest to put in the container, and an empty UUAA payload is not
    /// a UUAA attempt.
    pub fn build_uplink(&mut self) -> Option<Vec<u8>> {
        if !self.initiation_enabled {
            return None;
        }
        let payload = self.request_payload.clone()?;
        let container = ServiceLevelAaContainer::with_uuaa_payload(payload);
        let msg = UlNasTransport::new(PayloadContainerType::ServiceLevelAa, container.encode());

        let mut pdu = Vec::new();
        msg.encode(&mut pdu);

        self.state = UuaaState::Requested;
        self.uplinks_sent += 1;
        info!(
            "UUAA-MM: sending UUAA payload over UL NAS TRANSPORT (container type 9), len={}",
            pdu.len()
        );
        Some(pdu)
    }

    /// Apply a Service-level-AA container received in DL NAS TRANSPORT.
    ///
    /// Handles the two things a network can say: "perform the procedure"
    /// (pending indication, §9.11.2.17) and "here is the outcome"
    /// (response, §9.11.2.14). A container carrying a payload *for* the UE is
    /// acknowledged in the log and not answered, because answering would need
    /// the USS application this simulator does not model.
    pub fn handle_downlink(&mut self, bytes: &[u8]) -> UuaaReaction {
        let container = ServiceLevelAaContainer::decode(bytes);
        if container.truncated {
            warn!(
                "UUAA-MM: Service-level-AA container was truncated; using what parsed: {container}"
            );
        }

        if let Some(enabled) = container.uas_services_enabled {
            self.uas_services_enabled = Some(enabled);
        }

        // The outcome takes precedence over a pending indication: a container
        // carrying both is telling the UE the result of the procedure it just
        // asked for, and re-sending a payload in answer to a result would loop.
        if let Some(response) = container.response {
            self.state = if response.slar.is_authorized() {
                UuaaState::Authorized
            } else {
                UuaaState::Rejected
            };
            info!(
                "UUAA-MM: network result slar={:?} c2ar={:?} -> {:?}",
                response.slar, response.c2ar, self.state
            );
            return UuaaReaction::Nothing;
        }

        if container.payload_type == Some(ServiceLevelAaPayloadType::Uuaa) {
            if let Some(ref payload) = container.payload {
                // Opaque by design (TS 23.256): NAS relays it to the UAS
                // application, which this simulator does not have. Length only
                // -- the payload is authentication material.
                info!(
                    "UUAA-MM: received a {}-octet UUAA payload for the UAS application (not modelled)",
                    payload.len()
                );
            }
        }

        if container.pending_indication == Some(true) {
            if self.state == UuaaState::Authorized {
                // Re-authorization: the network is asking for the procedure
                // again, so the previous authorization no longer holds.
                info!(
                    "UUAA-MM: network requested the procedure again; prior authorization dropped"
                );
            }
            self.state = UuaaState::Pending;
            return match self.build_uplink() {
                Some(pdu) => UuaaReaction::SendUplink(pdu),
                None if !self.initiation_enabled => {
                    info!(
                        "UUAA-MM: network signalled the procedure is to be performed; this build \
                         does not originate UUAA uplinks (uuaa-mm feature off), so the pending \
                         indication is recorded and not answered"
                    );
                    UuaaReaction::Nothing
                }
                None => {
                    warn!(
                        "UUAA-MM: network signalled the procedure is to be performed, but no UUAA \
                         payload is configured -- staying pending rather than sending an empty one"
                    );
                    UuaaReaction::Nothing
                }
            };
        }

        UuaaReaction::Nothing
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nextgsim_nas::ies::service_level_aa::{ServiceLevelAaResponse, ServiceLevelAaResult};

    fn container_bytes(container: &ServiceLevelAaContainer) -> Vec<u8> {
        container.encode()
    }

    #[test]
    fn a_pending_indication_produces_an_uplink_carrying_the_configured_payload() {
        let mut uuaa = UuaaProcedure::new(Some(vec![0x11, 0x22]));
        let dl = container_bytes(&ServiceLevelAaContainer {
            pending_indication: Some(true),
            ..Default::default()
        });

        let UuaaReaction::SendUplink(pdu) = uuaa.handle_downlink(&dl) else {
            panic!("a pending indication must start the exchange");
        };
        // Plain 5GMM UL NAS TRANSPORT with payload container type 9.
        assert_eq!(&pdu[..3], &[0x7E, 0x00, 0x67]);
        assert_eq!(pdu[3], 0x09);
        // ... and the container inside carries the configured payload.
        let container = ServiceLevelAaContainer::decode(&pdu[6..]);
        assert_eq!(
            container.payload_type,
            Some(ServiceLevelAaPayloadType::Uuaa)
        );
        assert_eq!(container.payload.as_deref(), Some(&[0x11, 0x22][..]));
        assert_eq!(uuaa.state(), UuaaState::Requested);
        assert_eq!(uuaa.uplinks_sent(), 1);
    }

    #[test]
    fn a_pending_indication_with_no_configured_payload_sends_nothing() {
        let mut uuaa = UuaaProcedure::new(None);
        let dl = container_bytes(&ServiceLevelAaContainer {
            pending_indication: Some(true),
            ..Default::default()
        });

        assert_eq!(uuaa.handle_downlink(&dl), UuaaReaction::Nothing);
        // Pending, not Requested: nothing was sent, and the state must not
        // claim otherwise.
        assert_eq!(uuaa.state(), UuaaState::Pending);
        assert_eq!(uuaa.uplinks_sent(), 0);
        assert_eq!(
            uuaa.authorization_state(),
            UavAuthorizationState::AuthorizationRequested
        );
    }

    #[test]
    fn a_successful_response_authorizes_the_uav() {
        let mut uuaa = UuaaProcedure::new(Some(vec![0x01]));
        let dl = container_bytes(&ServiceLevelAaContainer {
            response: Some(ServiceLevelAaResponse {
                slar: ServiceLevelAaResult::Successful,
                c2ar: ServiceLevelAaResult::Successful,
            }),
            ..Default::default()
        });

        assert_eq!(uuaa.handle_downlink(&dl), UuaaReaction::Nothing);
        assert_eq!(uuaa.state(), UuaaState::Authorized);
        assert_eq!(
            uuaa.authorization_state(),
            UavAuthorizationState::Authorized
        );
    }

    #[test]
    fn an_unsuccessful_response_revokes_rather_than_leaving_it_unauthorized() {
        let mut uuaa = UuaaProcedure::new(Some(vec![0x01]));
        let dl = container_bytes(&ServiceLevelAaContainer {
            response: Some(ServiceLevelAaResponse {
                slar: ServiceLevelAaResult::NotSuccessfulOrRevoked,
                c2ar: ServiceLevelAaResult::NoInformation,
            }),
            ..Default::default()
        });

        uuaa.handle_downlink(&dl);
        assert_eq!(uuaa.state(), UuaaState::Rejected);
        assert_eq!(uuaa.authorization_state(), UavAuthorizationState::Revoked);
    }

    #[test]
    fn a_no_information_result_does_not_authorize() {
        // The trap: SLAR 0b00 is "no information", and reading anything that is
        // not an explicit failure as success is how a UAV flies unauthorized.
        let mut uuaa = UuaaProcedure::new(Some(vec![0x01]));
        let dl = container_bytes(&ServiceLevelAaContainer {
            response: Some(ServiceLevelAaResponse::default()),
            ..Default::default()
        });

        uuaa.handle_downlink(&dl);
        assert_eq!(uuaa.state(), UuaaState::Rejected);
        assert!(!matches!(
            uuaa.authorization_state(),
            UavAuthorizationState::Authorized
        ));
    }

    #[test]
    fn a_result_arriving_with_a_pending_indication_does_not_loop() {
        // Both parameters in one container: the result wins, so the UE does not
        // answer its own outcome with another payload.
        let mut uuaa = UuaaProcedure::new(Some(vec![0x01]));
        let dl = container_bytes(&ServiceLevelAaContainer {
            response: Some(ServiceLevelAaResponse {
                slar: ServiceLevelAaResult::Successful,
                c2ar: ServiceLevelAaResult::NoInformation,
            }),
            pending_indication: Some(true),
            ..Default::default()
        });

        assert_eq!(uuaa.handle_downlink(&dl), UuaaReaction::Nothing);
        assert_eq!(uuaa.state(), UuaaState::Authorized);
        assert_eq!(uuaa.uplinks_sent(), 0);
    }

    #[test]
    fn a_re_authorization_request_drops_the_previous_authorization() {
        let mut uuaa = UuaaProcedure::new(Some(vec![0x01]));
        uuaa.handle_downlink(&container_bytes(&ServiceLevelAaContainer {
            response: Some(ServiceLevelAaResponse {
                slar: ServiceLevelAaResult::Successful,
                c2ar: ServiceLevelAaResult::NoInformation,
            }),
            ..Default::default()
        }));
        assert_eq!(uuaa.state(), UuaaState::Authorized);

        // The network asks again: the UE must not keep claiming authorization
        // while a new exchange is outstanding.
        let reaction = uuaa.handle_downlink(&container_bytes(&ServiceLevelAaContainer {
            pending_indication: Some(true),
            ..Default::default()
        }));
        assert!(matches!(reaction, UuaaReaction::SendUplink(_)));
        assert_eq!(uuaa.state(), UuaaState::Requested);
        assert_eq!(
            uuaa.authorization_state(),
            UavAuthorizationState::AuthorizationRequested
        );
    }

    #[test]
    fn a_service_status_indication_is_recorded_separately_from_authorization() {
        let mut uuaa = UuaaProcedure::new(Some(vec![0x01]));
        assert_eq!(uuaa.uas_services_enabled(), None);

        uuaa.handle_downlink(&container_bytes(&ServiceLevelAaContainer {
            uas_services_enabled: Some(false),
            response: Some(ServiceLevelAaResponse {
                slar: ServiceLevelAaResult::Successful,
                c2ar: ServiceLevelAaResult::NoInformation,
            }),
            ..Default::default()
        }));

        // Authorized, and yet UAS services are off: two different facts.
        assert_eq!(uuaa.state(), UuaaState::Authorized);
        assert_eq!(uuaa.uas_services_enabled(), Some(false));
    }

    #[test]
    fn a_downlink_uuaa_payload_is_accepted_without_being_answered() {
        // No USS application exists to answer it, so the honest reaction is to
        // take it and not fabricate a reply.
        let mut uuaa = UuaaProcedure::new(Some(vec![0x01]));
        let dl = container_bytes(&ServiceLevelAaContainer::with_uuaa_payload(vec![0xAA; 4]));

        assert_eq!(uuaa.handle_downlink(&dl), UuaaReaction::Nothing);
        assert_eq!(uuaa.state(), UuaaState::Idle);
    }

    #[test]
    fn a_truncated_container_still_applies_what_parsed() {
        // Device ID complete, then a response parameter claiming more octets
        // than are present.
        let bytes = [0x50, 0x01, 0x01, 0x30, 0x09, 0x00];
        let mut uuaa = UuaaProcedure::new(Some(vec![0x01]));
        uuaa.handle_downlink(&bytes);
        assert_eq!(uuaa.uas_services_enabled(), Some(true));
        assert_eq!(uuaa.state(), UuaaState::Idle, "no result was readable");
    }

    #[test]
    fn with_initiation_disabled_a_pending_indication_is_recorded_and_not_answered() {
        // What the default build does: the container is understood, the state
        // records that the network asked, and nothing goes on the wire.
        let mut uuaa = UuaaProcedure::with_initiation(Some(vec![0x11, 0x22]), false);
        assert!(!uuaa.initiation_enabled());

        let dl = container_bytes(&ServiceLevelAaContainer {
            pending_indication: Some(true),
            ..Default::default()
        });
        assert_eq!(uuaa.handle_downlink(&dl), UuaaReaction::Nothing);
        assert_eq!(uuaa.state(), UuaaState::Pending);
        assert_eq!(uuaa.uplinks_sent(), 0);
        assert!(uuaa.build_uplink().is_none());
    }

    #[test]
    fn with_initiation_disabled_a_network_result_is_still_applied() {
        // The gate is on originating, not on understanding: a UE that ignored
        // a revocation because a cargo feature was off would keep flying.
        let mut uuaa = UuaaProcedure::with_initiation(Some(vec![0x01]), false);
        uuaa.handle_downlink(&container_bytes(&ServiceLevelAaContainer {
            response: Some(ServiceLevelAaResponse {
                slar: ServiceLevelAaResult::NotSuccessfulOrRevoked,
                c2ar: ServiceLevelAaResult::NoInformation,
            }),
            ..Default::default()
        }));
        assert_eq!(uuaa.state(), UuaaState::Rejected);
        assert_eq!(uuaa.authorization_state(), UavAuthorizationState::Revoked);
    }

    #[test]
    fn an_empty_container_changes_nothing() {
        let mut uuaa = UuaaProcedure::new(Some(vec![0x01]));
        assert_eq!(uuaa.handle_downlink(&[]), UuaaReaction::Nothing);
        assert_eq!(uuaa.state(), UuaaState::Idle);
        assert_eq!(
            uuaa.authorization_state(),
            UavAuthorizationState::NotAuthorized
        );
    }
}
