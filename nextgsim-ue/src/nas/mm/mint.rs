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
    pub fn from_config(config: &UeConfig, legacy_core_accept_compat: bool) -> Self {
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
                sm: SmOrchestrator::new(legacy_core_accept_compat),
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
            let outs = sub.mm.start_registration(RegistrationType::InitialRegistration);
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

    /// Derive the per-subscription session parameter list for a secondary
    /// subscription: the configured sessions whose DNN routes to this index.
    pub fn session_params_for(
        config: &UeConfig,
        index: u8,
    ) -> Vec<SmSessionParams> {
        SmSessionParams::from_config(config)
            .into_iter()
            .filter(|p| p.subscription_index == index)
            .collect()
    }
}
