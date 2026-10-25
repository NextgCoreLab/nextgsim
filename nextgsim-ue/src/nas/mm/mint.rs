//! MINT secondary-subscription driver (Rel-18, TS 23.761)
//!
//! Minimization of service interruption / multi-SUPI support. A MINT UE holds
//! more than one subscription (SUPI). This module models the *secondary*
//! subscriptions in the MM layer: each gets its own [`MmOrchestrator`] (with
//! its own SUCI, GUTI and NAS security context) and its own
//! [`SmOrchestrator`], so a PDU session routed to a secondary subscription
//! (per the DNN→subscription map, see [`SmSessionParams::subscription_index`])
//! is established under that SUPI's identity and the network resolves SUPI-N's
//! subscription in UDM/UDR.
//!
//! It is deliberately self-contained and additive: the primary registration is
//! still owned by the NAS task's main orchestrator, and the SNPN
//! Registration-Request additions live independently in
//! [`super::orchestrator`]. The driver is created from the UE configuration
//! and, after the primary subscription registers, the NAS task asks it to
//! initiate the secondary registrations (simultaneous registration,
//! TS 23.761).
//!
//! Single radio connection note: the simulator runs one RRC/NGAP UE connection
//! per UE process. The secondary Registration Requests are sent over that same
//! connection; the AMF allocates a distinct AMF UE NGAP ID (and thus a distinct
//! AMF UE context + 5G-GUTI) per InitialUEMessage, so the secondary SUPI gets
//! its own context. Downlink NAS demultiplexing between the primary and a
//! secondary context on the shared connection is resolved by routing a
//! downlink PDU to whichever MM context currently owns an in-flight procedure
//! (see [`MintSecondary::owns_pending_downlink`]).

use nextgsim_common::config::UeConfig;
use tracing::{info, warn};

use super::orchestrator::{MmOrchestrator, MmOutput, MmUeIdentity};
use super::state::MmSubState;
use super::suci::build_suci;
use crate::nas::sm::{SmOrchestrator, SmSessionParams};

/// One secondary subscription's MM + SM state.
pub struct SecondarySubscription {
    /// Subscription index (>= 1; 0 is the primary owned by the NAS task)
    pub index: u8,
    /// SUPI value signalled for this subscription
    pub supi: String,
    /// MM orchestrator owning this subscription's registration/security
    pub mm: MmOrchestrator,
    /// SM orchestrator owning this subscription's PDU sessions
    pub sm: SmOrchestrator,
    /// Whether the secondary registration has been initiated
    started: bool,
}

/// MINT secondary-subscription driver. Empty (and inert) when the UE is not
/// configured for MINT, so non-MINT UEs are unaffected.
pub struct MintSecondary {
    subs: Vec<SecondarySubscription>,
    disaster_roaming: bool,
}

impl MintSecondary {
    /// Build the driver from the UE configuration. Returns an empty driver
    /// (no secondary subscriptions) when MINT is disabled or no secondary
    /// SUPIs are configured.
    pub fn from_config(config: &UeConfig) -> Self {
        let Some(mint) = config.mint_config.as_ref() else {
            return Self {
                subs: Vec::new(),
                disaster_roaming: false,
            };
        };
        if !mint.enabled || mint.secondary_supis.is_empty() {
            return Self {
                subs: Vec::new(),
                disaster_roaming: false,
            };
        }

        let mut subs = Vec::new();
        for (i, secondary_supi) in mint.secondary_supis.iter().enumerate() {
            let index = (i + 1) as u8;
            // Build a SUCI for this SUPI by temporarily swapping the SUPI in a
            // clone of the configuration (home PLMN, keys and SUCI scheme are
            // shared with the primary subscription, TS 23.761).
            let mut sub_config = config.clone();
            sub_config.supi = Some(nextgsim_common::Supi::imsi(secondary_supi.clone()));
            let suci = match build_suci(&sub_config) {
                Ok(suci) => suci,
                Err(e) => {
                    warn!(
                        "MINT: cannot build SUCI for secondary SUPI {secondary_supi}: {e}; \
                         skipping this subscription"
                    );
                    continue;
                }
            };
            let identity = MmUeIdentity::for_secondary_supi(
                config,
                suci,
                secondary_supi,
                mint.disaster_roaming,
            );
            subs.push(SecondarySubscription {
                index,
                supi: secondary_supi.clone(),
                mm: MmOrchestrator::new(identity),
                sm: SmOrchestrator::new(),
                started: false,
            });
        }

        Self {
            subs,
            disaster_roaming: mint.disaster_roaming,
        }
    }

    /// Whether any secondary subscription is configured.
    pub fn is_active(&self) -> bool {
        !self.subs.is_empty()
    }

    /// Whether disaster-roaming is requested for the secondary registrations.
    pub fn disaster_roaming(&self) -> bool {
        self.disaster_roaming
    }

    /// Initiate registration for every secondary subscription that has not yet
    /// been started. Returns the per-subscription Registration-Request outputs
    /// the caller must transmit (each carries its own ngKSI/SUCI and, when
    /// configured, the disaster-roaming indication). TS 23.761: simultaneous
    /// registration of multiple subscriptions.
    pub fn start_secondary_registrations(&mut self) -> Vec<(u8, Vec<MmOutput>)> {
        use nextgsim_nas::ies::RegistrationType;
        let mut out = Vec::new();
        for sub in &mut self.subs {
            if sub.started {
                continue;
            }
            sub.started = true;
            info!(
                "MINT: initiating secondary registration for subscription {} (SUPI={})",
                sub.index, sub.supi
            );
            let outs = sub
                .mm
                .start_registration(RegistrationType::InitialRegistration);
            out.push((sub.index, outs));
        }
        out
    }

    /// Find the secondary subscription serving a given subscription index.
    pub fn get_mut(&mut self, index: u8) -> Option<&mut SecondarySubscription> {
        self.subs.iter_mut().find(|s| s.index == index)
    }

    /// Whether a secondary MM context currently owns an in-flight downlink
    /// procedure (registration/authentication/security-mode in progress, or a
    /// secondary session is establishing). Used to route a downlink NAS PDU on
    /// the shared radio connection to the right MM context.
    pub fn owns_pending_downlink(&self) -> Option<u8> {
        self.subs
            .iter()
            .find(|s| {
                // In-flight registration (Registration Accept / Auth Request /
                // Security Mode Command expected) or an in-flight SM procedure
                // (PDU session establish/modify/release awaiting a response).
                // Steady-state (registered, all sessions Active) does NOT claim
                // the downlink, so the primary keeps its own DL NAS.
                s.started
                    && (s.mm.state().mm_substate() == MmSubState::RegisteredInitiated
                        || s.sm.has_pending_procedure())
            })
            .map(|s| s.index)
    }

    /// Tick all secondary MM and SM timers; returns per-subscription outputs.
    pub fn tick(&mut self) -> Vec<(u8, Vec<MmOutput>)> {
        let mut out = Vec::new();
        for sub in &mut self.subs {
            if !sub.started {
                continue;
            }
            let mm_outs = sub.mm.tick();
            if !mm_outs.is_empty() {
                out.push((sub.index, mm_outs));
            }
        }
        out
    }

    /// Drive every started secondary subscription's SM orchestrator once a
    /// second (#409). The tick runs its procedure timers (T3580/T3581/T3582)
    /// and back-off timers (T3396/T3584/T3585). Configured sessions whose
    /// back-off is over are then re-attempted (#379, TS 24.501 §6.4.1.4.2),
    /// but only while the subscription is in 5GMM-REGISTERED.NORMAL-SERVICE and
    /// the shared radio connection is up (`radio_connected`). A secondary MM has
    /// no CM state of its own, so the primary's CM state stands in for it.
    /// Returns the per-subscription SM outputs.
    pub fn tick_sessions(
        &mut self,
        radio_connected: bool,
    ) -> Vec<(u8, Vec<crate::nas::sm::SmOutput>)> {
        let mut out = Vec::new();
        for sub in &mut self.subs {
            if !sub.started {
                continue;
            }
            // A 5GSM retransmission waits for the shared connection: a
            // secondary MM has no NAS signalling connection of its own and so
            // asks for none (#409 follow-up)
            let mut sm_outs = sub.sm.tick_with_connection(radio_connected);
            if radio_connected
                && sub.mm.state().mm_substate() == MmSubState::RegisteredNormalService
            {
                sm_outs.extend(sub.sm.retry_wanted_sessions());
            }
            if !sm_outs.is_empty() {
                out.push((sub.index, sm_outs));
            }
        }
        out
    }

    /// Derive the per-subscription session parameter list for a secondary
    /// subscription: the configured sessions whose DNN routes to this index.
    pub fn session_params_for(config: &UeConfig, index: u8) -> Vec<SmSessionParams> {
        SmSessionParams::from_config(config)
            .into_iter()
            .filter(|p| p.subscription_index == index)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nas::sm::SmOutput;
    use nextgsim_common::config::MintConfig;
    use nextgsim_nas::ies::ie1::PayloadContainerType;
    use nextgsim_nas::messages::mm::{DlNasTransport, UlNasTransport};
    use nextgsim_nas::messages::sm::{
        PduSessionEstablishmentReject, PduSessionTypeValue, SmCause, SscModeValue,
    };

    /// A MINT driver with one started secondary subscription in
    /// 5GMM-REGISTERED.NORMAL-SERVICE whose configured session was rejected
    /// #26 with a 60 s back-off (T3396).
    fn secondary_rejected_26() -> MintSecondary {
        let config = UeConfig {
            mint_config: Some(MintConfig {
                enabled: true,
                secondary_supis: vec!["999700000000002".to_string()],
                ..MintConfig::default()
            }),
            ..UeConfig::default()
        };
        let mut mint = MintSecondary::from_config(&config);
        assert!(mint.is_active());
        let sub = &mut mint.subs[0];
        sub.started = true;
        sub.mm
            .state_mut()
            .switch_mm_state(MmSubState::RegisteredNormalService);
        let outs = sub.sm.establish_default_sessions(&[SmSessionParams {
            session_type: PduSessionTypeValue::Ipv4,
            ssc_mode: SscModeValue::SscMode1,
            dnn: Some("enterprise".to_string()),
            s_nssai: Some(vec![0x01]),
            emergency: false,
            requested_5qi: None,
            subscription_index: 1,
        }]);
        let Some(SmOutput::SendNasPdu(ul)) = outs.first() else {
            panic!("an establishment request: {outs:?}");
        };
        let t = UlNasTransport::decode(&mut &ul[3..]).unwrap();
        let (psi, pti) = (t.payload_container[1], t.payload_container[2]);
        let mut rej = PduSessionEstablishmentReject::new(psi, pti, SmCause::InsufficientResources);
        rej.back_off_timer_value = Some(0xA1); // 1 minute
        let mut bytes = Vec::new();
        rej.encode(&mut bytes);
        let mut dl = DlNasTransport::new(PayloadContainerType::N1SmInformation, bytes);
        dl.pdu_session_id = Some(psi);
        sub.sm.handle_dl_nas_transport(&dl);
        assert!(sub.sm.has_pending_retry());
        mint
    }

    fn establishment_requests(outs: &[(u8, Vec<SmOutput>)]) -> usize {
        outs.iter()
            .flat_map(|(_, o)| o)
            .filter(|o| match o {
                SmOutput::SendNasPdu(ul) => UlNasTransport::decode(&mut &ul[3..])
                    .is_ok_and(|t| t.payload_container.get(3) == Some(&0xC1)),
                _ => false,
            })
            .count()
    }

    /// #409: the secondary SM orchestrator is ticked. Its back-off runs out
    /// and the configured session is asked for again on the shared radio
    /// connection, not before the back-off ends, and not while the radio
    /// connection is down.
    #[test]
    fn a_secondary_subscriptions_back_off_runs_out_and_its_session_is_re_requested() {
        let mut mint = secondary_rejected_26();
        assert_eq!(
            establishment_requests(&mint.tick_sessions(true)),
            0,
            "T3396 running"
        );
        mint.subs[0].sm.expire_backoffs_for_test();
        assert_eq!(
            establishment_requests(&mint.tick_sessions(false)),
            0,
            "the tick ran T3396 out, but the radio connection is down"
        );
        let outs = mint.tick_sessions(true);
        assert_eq!(establishment_requests(&outs), 1, "{outs:?}");
        assert_eq!(outs[0].0, 1, "on the secondary subscription");
    }

    /// #409: a secondary subscription's T3580 runs. An establishment request
    /// the network never answers is retransmitted, the identical octets, on
    /// T3580's expiry (TS 24.501 §6.4.1.6).
    #[test]
    fn a_secondary_subscriptions_t3580_retransmits_the_request() {
        let mut mint = secondary_rejected_26();
        mint.subs[0].sm.expire_backoffs_for_test();
        let outs = mint.tick_sessions(true);
        let Some((1, sent)) = outs.first() else {
            panic!("the re-attempt: {outs:?}");
        };
        let request = sent.clone();
        assert_eq!(establishment_requests(&outs), 1);
        assert!(
            establishment_requests(&mint.tick_sessions(true)) == 0,
            "T3580 running"
        );
        mint.subs[0].sm.expire_procedure_timers_for_test();
        let outs = mint.tick_sessions(true);
        assert_eq!(outs, vec![(1, request)], "the retransmission");
    }

    /// #409 follow-up, TS 24.501 §6.4.1.6 a): the whole T3580 sequence on a
    /// secondary subscription: four retransmissions of the identical octets,
    /// then the fifth expiry aborts the establishment. One of them falls while
    /// the shared radio connection is down: it is held, and goes when the
    /// connection is back (a secondary asks for no connection of its own).
    #[test]
    fn a_secondary_subscriptions_t3580_runs_its_whole_sequence() {
        let mut mint = secondary_rejected_26();
        mint.subs[0].sm.expire_backoffs_for_test();
        let outs = mint.tick_sessions(true);
        let Some((1, sent)) = outs.first() else {
            panic!("the re-attempt: {outs:?}");
        };
        let request = sent.clone();
        for n in 1..4u8 {
            mint.subs[0].sm.expire_procedure_timers_for_test();
            if n == 2 {
                assert!(
                    mint.tick_sessions(false).is_empty(),
                    "held while the radio is down"
                );
            }
            assert_eq!(mint.tick_sessions(true), vec![(1, request.clone())], "#{n}");
        }
        mint.subs[0].sm.expire_procedure_timers_for_test();
        assert_eq!(mint.tick_sessions(true), vec![(1, request)], "#4");
        mint.subs[0].sm.expire_procedure_timers_for_test();
        let outs = mint.tick_sessions(true);
        assert!(
            matches!(
                outs.as_slice(),
                [(1, o)] if matches!(o.as_slice(), [SmOutput::SessionEstablishmentFailed { .. }])
            ),
            "the fifth expiry aborts: {outs:?}"
        );
    }

    /// #409: a subscription not in NORMAL-SERVICE is not re-attempted, and an
    /// unstarted subscription is not driven at all.
    #[test]
    fn no_secondary_re_attempt_outside_normal_service_or_before_start() {
        let mut mint = secondary_rejected_26();
        mint.subs[0].sm.expire_backoffs_for_test();
        mint.subs[0]
            .mm
            .state_mut()
            .switch_mm_state(MmSubState::RegisteredInitiated);
        assert_eq!(establishment_requests(&mint.tick_sessions(true)), 0);
        mint.subs[0]
            .mm
            .state_mut()
            .switch_mm_state(MmSubState::RegisteredNormalService);
        mint.subs[0].started = false;
        assert!(mint.tick_sessions(true).is_empty());
    }
}
