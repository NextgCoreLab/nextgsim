//! 5GSM Procedure Orchestrator (3GPP TS 24.501 Section 6)
//!
//! UE-side session-management procedure state:
//!
//! - PDU session establishment (TS 24.501 Section 6.4.1) with T3580
//!   arm/stop, retransmission to the spec maximum and the Section 6.4.1.6
//!   abnormal action on exhaustion.
//! - UE-initiated PDU session modification (Section 6.4.2, T3581) and
//!   release (Section 6.4.3, T3582), each with Accept/Reject/Command
//!   handling.
//! - Network-initiated modification command / release command handling
//!   (Complete responses, reactivation on cause #39).
//! - Full 5GSM cause handling (Section 9.11.4.2) including the per-DNN /
//!   per-S-NSSAI back-off timers (T3396 class) signalled in the back-off
//!   timer value IE for causes #26, #67, #69 and #70, and the PDU session
//!   type fallbacks for causes #50, #51 and #57.
//! - PSI allocation (1-15) and PTI allocation (1-254) per procedure.
//!
//! All 5GSM messages are exchanged wrapped in UL/DL NAS Transport
//! (TS 24.501 Section 5.4.5): the orchestrator builds the plain UL NAS
//! Transport PDU and the caller protects it with the MM orchestrator's
//! NAS security context before transmission.
//!
//! The orchestrator is sans-IO: callers feed it DL NAS Transport payloads,
//! trigger procedures and drive `tick()` once a second; it returns
//! [`SmOutput`] values describing NAS PDUs to transmit and procedure
//! outcomes.

use std::collections::HashMap;

use tracing::{debug, info, warn};

use nextgsim_common::config::{
    PduSessionType as ConfigPduSessionType, SessionConfig as ConfigSessionConfig, UeConfig,
};
use nextgsim_nas::enums::SmMessageType as NasSmMessageType;
use nextgsim_nas::header::PlainSmHeader;
use nextgsim_nas::ies::ie1::{PayloadContainerType, RequestType};
use nextgsim_nas::messages::mm::{DlNasTransport, UlNasTransport};
use nextgsim_nas::messages::sm::{
    IeDnn, IeIntegrityProtectionMaxDataRate, IeSelectedPduSessionType, IeSelectedSscMode,
    IeSessionAmbr, PduAddressType, PduSessionEstablishmentAccept, PduSessionEstablishmentReject,
    PduSessionEstablishmentRequest, PduSessionModificationCommand, PduSessionModificationComplete,
    PduSessionModificationReject, PduSessionModificationRequest, PduSessionReleaseCommand,
    PduSessionReleaseComplete, PduSessionReleaseReject, PduSessionReleaseRequest,
    PduSessionTypeValue, SmCause, SscModeValue,
};

use crate::timer::{
    GprsTimer3, GprsTimer3Unit, UeTimer, MAX_T3580_RETRIES, MAX_T3581_RETRIES, MAX_T3582_RETRIES,
};

use super::procedure::{
    ProcedureTransactionManager, SmMessageType as PtMessageType, SM_TIMER_T3580, SM_TIMER_T3581,
    SM_TIMER_T3582,
};

/// Maximum PDU session identity value (TS 24.501 Section 9.4)
pub const PSI_MAX: u8 = 15;

/// Timer code used for the T3396-class per-DNN / per-S-NSSAI back-off
/// timers (TS 24.008 Section 11.2.2.1A applied to 5GS, TS 24.501 6.4.1.4)
pub const SM_TIMER_BACKOFF: u16 = 3396;

/// Timer code used for the §6.4.1.4.3 back-off timer (#70), which has no
/// T-number of its own; distinct from T3396 so the two never share an entry
pub const SM_TIMER_GENERIC_BACKOFF: u16 = 0;

/// Timer code of T3584, the [S-NSSAI, DNN] back-off of cause #67
/// (TS 24.501 §6.2.8, §6.4.1.4.2)
pub const SM_TIMER_T3584: u16 = 3584;

/// Timer code of T3585, the S-NSSAI back-off of cause #69
/// (TS 24.501 §6.2.8, §6.4.1.4.2)
pub const SM_TIMER_T3585: u16 = 3585;

/// First wait of the UE's own re-attempt policy for a configured PDU session
/// whose establishment failed with no network-given wait: a reject with no
/// Back-off timer value IE or a zero one, a #90 "payload was not forwarded"
/// echo, T3580 exhaustion (TS 24.501 §6.4.1.4.2, §6.4.1.6, §5.4.5.3.3).
/// Doubled per consecutive failure up to [`SM_RETRY_MAX_SECS`].
pub const SM_RETRY_BASE_SECS: u32 = 2;

/// Ceiling of the UE's own re-attempt wait.
pub const SM_RETRY_MAX_SECS: u32 = 64;

/// The SM Retry Timer default (TS 24.501 §6.4.1.4.3, TS 24.368): the back-off
/// a #70 reject without a Back-off timer value IE starts.
pub const SM_RETRY_TIMER_DEFAULT_SECS: u32 = 12 * 60;

/// PDU session states (TS 24.501 Section 6.1.3.2.1)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PsState {
    /// PDU SESSION INACTIVE
    #[default]
    Inactive,
    /// PDU SESSION ACTIVE PENDING (establishment requested)
    ActivePending,
    /// PDU SESSION ACTIVE
    Active,
    /// PDU SESSION MODIFICATION PENDING (UE-initiated modification)
    ModificationPending,
    /// PDU SESSION INACTIVE PENDING (UE-initiated release)
    InactivePending,
}

/// Session parameters derived from the UE configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmSessionParams {
    /// Requested PDU session type
    pub session_type: PduSessionTypeValue,
    /// Requested SSC mode
    pub ssc_mode: SscModeValue,
    /// DNN ("APN") to request, if configured
    pub dnn: Option<String>,
    /// Encoded S-NSSAI IE value (SST [+ SD]), if configured
    pub s_nssai: Option<Vec<u8>>,
    /// Whether this is an emergency PDU session
    pub emergency: bool,
    /// Requested 5QI (TS 23.501 §5.7). When this is an XR delay-critical GBR
    /// 5QI (82-85, Rel-18), the UE requests an XR PDU session; the DNN selects
    /// the XR service so the SMF maps and installs the XR QoS flow.
    pub requested_5qi: Option<u8>,
    /// MINT subscription index that serves this session (Rel-18, TS 23.761).
    /// 0 = primary SUPI; >0 selects a secondary SUPI's subscription. The DNN
    /// drives the mapping (see [`SmSessionParams::from_config`]); the session
    /// is established under the matching subscription's MM/security context so
    /// the SMF resolves SUPI-N's subscription in UDM/UDR.
    pub subscription_index: u8,
}

impl SmSessionParams {
    /// Derive the default-session parameter list from the UE configuration.
    /// When no session is configured a single IPv4 "internet" session with
    /// the first configured slice is used.
    pub fn from_config(config: &UeConfig) -> Vec<Self> {
        let default_snssai = config.configured_nssai.slices.first().map(|s| {
            let mut v = vec![s.sst];
            if let Some(sd) = s.sd {
                v.extend_from_slice(&sd);
            }
            v
        });

        if config.sessions.is_empty() {
            return vec![Self {
                session_type: PduSessionTypeValue::Ipv4,
                ssc_mode: SscModeValue::SscMode1,
                dnn: Some("internet".to_string()),
                s_nssai: default_snssai,
                emergency: false,
                requested_5qi: None,
                subscription_index: 0,
            }];
        }

        config
            .sessions
            .iter()
            .map(|s| {
                let session_type = match s.session_type {
                    ConfigPduSessionType::Ipv4 => PduSessionTypeValue::Ipv4,
                    ConfigPduSessionType::Ipv6 => PduSessionTypeValue::Ipv6,
                    ConfigPduSessionType::Ipv4v6 => PduSessionTypeValue::Ipv4v6,
                    ConfigPduSessionType::Unstructured => PduSessionTypeValue::Unstructured,
                    ConfigPduSessionType::Ethernet => PduSessionTypeValue::Ethernet,
                };
                let s_nssai = s
                    .s_nssai
                    .as_ref()
                    .map(|n| {
                        let mut v = vec![n.sst];
                        if let Some(sd) = n.sd {
                            v.extend_from_slice(&sd);
                        }
                        v
                    })
                    .or_else(|| default_snssai.clone());
                // For an XR session with no explicit APN, select the XR DNN so
                // the SMF derives the XR 5QI and installs the XR QoS flow
                // (TS 23.501 §5.7.4, Rel-18).
                let dnn = match (s.apn.clone(), s.requested_5qi) {
                    (Some(apn), _) => Some(apn),
                    (None, Some(five_qi)) if s.is_xr() => {
                        let xr_dnn = ConfigSessionConfig::xr_dnn_for_5qi(five_qi);
                        info!(
                            "XR PDU session requested (5QI={five_qi}) — selecting XR DNN {xr_dnn:?}"
                        );
                        Some(xr_dnn.to_string())
                    }
                    (None, _) => None,
                };
                // MINT (Rel-18, TS 23.761): pick which subscription's SUPI
                // serves this session from the DNN→subscription map; absent a
                // MINT config or matching DNN the primary subscription (0) is
                // used, so non-MINT deployments are unaffected.
                let subscription_index = config
                    .mint_config
                    .as_ref()
                    .map_or(0, |m| m.subscription_for_dnn(dnn.as_deref()));
                Self {
                    session_type,
                    ssc_mode: SscModeValue::SscMode1,
                    dnn,
                    s_nssai,
                    emergency: s.is_emergency,
                    requested_5qi: s.requested_5qi,
                    subscription_index,
                }
            })
            .collect()
    }
}

/// An established / establishing PDU session context.
#[derive(Debug, Clone)]
pub struct PduSession {
    /// PDU session identity (1-15)
    pub psi: u8,
    /// Session state (TS 24.501 Section 6.1.3.2.1)
    pub state: PsState,
    /// Requested parameters
    pub params: SmSessionParams,
    /// Network-selected PDU session type (from the Accept)
    pub selected_type: Option<PduSessionTypeValue>,
    /// Network-selected SSC mode (from the Accept)
    pub selected_ssc_mode: Option<SscModeValue>,
    /// Assigned IPv4 address, if any
    pub ipv4: Option<[u8; 4]>,
    /// Authorized QoS rules from the network (opaque)
    pub authorized_qos_rules: Vec<u8>,
    /// Authorized session AMBR from the network
    pub session_ambr: Option<IeSessionAmbr>,
    /// The configured session this one serves, when the UE holds it as one of
    /// its configured sessions (#379): a failed establishment is then
    /// re-attempted. `None` for a one-off (CLI or application) session.
    wanted: Option<SmSessionParams>,
    /// Consecutive failed establishments of `wanted` before this attempt;
    /// scales the UE's own re-attempt wait.
    failures: u32,
}

impl PduSession {
    fn new(psi: u8, params: SmSessionParams) -> Self {
        Self {
            psi,
            state: PsState::Inactive,
            params,
            selected_type: None,
            selected_ssc_mode: None,
            ipv4: None,
            authorized_qos_rules: Vec::new(),
            session_ambr: None,
            wanted: None,
            failures: 0,
        }
    }
}

/// Outputs produced by the orchestrator for the caller (NAS task) to act on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SmOutput {
    /// A plain NAS PDU (UL NAS Transport carrying a 5GSM message) ready to
    /// be security protected by the MM orchestrator and transmitted
    SendNasPdu(Vec<u8>),
    /// A PDU session is now active
    SessionEstablished {
        /// PDU session identity
        psi: u8,
        /// Assigned IPv4 address, if any
        ipv4: Option<[u8; 4]>,
    },
    /// PDU session establishment failed terminally for this attempt
    SessionEstablishmentFailed {
        /// PDU session identity
        psi: u8,
        /// 5GSM cause (ProtocolErrorUnspecified for local aborts)
        cause: SmCause,
    },
    /// A PDU session was released (UE- or network-initiated)
    SessionReleased {
        /// PDU session identity
        psi: u8,
    },
    /// A UE-initiated modification completed
    SessionModified {
        /// PDU session identity
        psi: u8,
    },
    /// A UE-initiated modification was rejected
    SessionModificationFailed {
        /// PDU session identity
        psi: u8,
        /// 5GSM cause
        cause: SmCause,
    },
}

/// One component of a back-off scope (TS 24.501 §6.2.7, §6.2.8)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Scope<T> {
    /// The timer is not associated with this component (T3396 has no
    /// S-NSSAI, T3585 no DNN): any value matches
    Any,
    /// The timer is associated with this value, where `None` is "no DNN" /
    /// "no S-NSSAI" -- a request that provided none, which is not a wildcard
    /// (§6.2.7: "T3396 is associated with no DNN")
    Is(Option<T>),
}

/// Back-off scope key (TS 24.501 Section 6.4.1.4: the back-off applies per
/// [S-NSSAI, DNN] combination depending on the reject cause)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BackoffKey {
    /// Which timer (T3396, T3584, T3585, or the §6.4.1.4.3 back-off timer):
    /// each runs on its own, so one never replaces or stops another in the
    /// same scope (#380 review)
    timer: u16,
    s_nssai: Scope<Vec<u8>>,
    dnn: Scope<String>,
}

impl BackoffKey {
    /// Whether this back-off applies to a request for `s_nssai` / `dnn`.
    fn covers(&self, s_nssai: Option<&[u8]>, dnn: Option<&str>) -> bool {
        let nssai_match = match &self.s_nssai {
            Scope::Any => true,
            Scope::Is(v) => v.as_deref() == s_nssai,
        };
        let dnn_match = match &self.dnn {
            Scope::Any => true,
            Scope::Is(v) => v.as_deref() == dnn,
        };
        nssai_match && dnn_match
    }
}

/// A running (or permanently blocking) back-off entry
#[derive(Debug)]
struct BackoffEntry {
    /// Which timer this is, for the logs: T3396 (#26), T3584 (#67), T3585
    /// (#69) or the §6.4.1.4.3 back-off timer (#70)
    name: &'static str,
    /// Whether a PDU SESSION MODIFICATION / RELEASE COMMAND in its scope
    /// stops it: T3396, T3584 and T3585 (§6.3.2.3, §6.3.3.3); the
    /// §6.4.1.4.3 back-off timer runs to expiry
    stoppable: bool,
    /// Running timer; `None` for a deactivated back-off timer value, which
    /// blocks retries until the UE is restarted or the network sends a PDU
    /// SESSION MODIFICATION / RELEASE COMMAND in the same scope
    /// (TS 24.501 6.4.1.4.2)
    timer: Option<UeTimer>,
}

/// A configured PDU session waiting to be re-attempted after a failed
/// establishment (#379, TS 24.501 §6.4.1.4.2)
#[derive(Debug, Clone)]
struct RetryEntry {
    /// The configured session parameters (before URSP and type overrides)
    base: SmSessionParams,
    /// End of the UE's own re-attempt wait. A running back-off timer bars the
    /// re-attempt on top of this.
    due_at: std::time::Instant,
    /// Consecutive failed establishments so far
    failures: u32,
}

/// The UE's own re-attempt wait after `failures` consecutive failures.
fn retry_wait(failures: u32) -> u32 {
    let doublings = failures.saturating_sub(1).min(16);
    SM_RETRY_BASE_SECS
        .saturating_mul(1 << doublings)
        .min(SM_RETRY_MAX_SECS)
}

/// A pending UE-initiated SM procedure (for retransmission on timer expiry)
#[derive(Debug, Clone)]
struct PendingProcedure {
    /// The complete plain UL NAS Transport PDU (re-protected on each send)
    ul_pdu: Vec<u8>,
}

/// 5GSM procedure orchestrator (UE side).
pub struct SmOrchestrator {
    /// Sessions indexed by PSI (index 0 unused)
    sessions: [Option<PduSession>; 16],
    /// Procedure transactions (PTI allocation + per-procedure timers)
    pt: ProcedureTransactionManager,
    /// URSP rules and the evaluation switch (TS 24.526 §5.2, #47). Inert until
    /// `ursp_evaluation` is set, in which case a matching rule's route selection
    /// descriptor overrides the static session parameters.
    ursp: super::ursp::UrspPolicy,
    /// Pending UL PDUs by PTI for retransmission
    pending: HashMap<u8, PendingProcedure>,
    /// T3396-class back-off timers per [S-NSSAI, DNN]
    backoff: HashMap<BackoffKey, BackoffEntry>,
    /// PDU session type override per DNN after cause #50/#51/#57
    /// (TS 24.501 Section 6.4.1.4.2)
    type_override: HashMap<Option<String>, PduSessionTypeValue>,
    /// The configured sessions the UE holds (#379): the last set given to
    /// [`SmOrchestrator::establish_default_sessions`], less any the UE released
    /// itself.
    wanted: Vec<SmSessionParams>,
    /// Configured sessions whose establishment failed, waiting to be
    /// re-attempted (#379)
    retries: Vec<RetryEntry>,
    /// PTIs whose procedure timer expired while the UE had no NAS signalling
    /// connection: the retransmission waits for one (#409 follow-up). A 5GSM
    /// message travels in UL NAS TRANSPORT, which 5GMM sends only in
    /// 5GMM-CONNECTED; from 5GMM-IDLE it first sends a SERVICE REQUEST for
    /// the pending uplink signalling (TS 24.501 §5.6.1.1).
    /// With each, the deadline after which the held procedure is aborted as
    /// on its last expiry: the time its remaining retransmissions would have
    /// taken, so a hold is bounded even when no connection comes (#424
    /// review).
    held_retransmissions: std::collections::BTreeMap<u8, std::time::Instant>,
}

impl Default for SmOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}

impl SmOrchestrator {
    /// Create a new orchestrator.
    ///
    /// The UE only accepts the spec-conformant PDU Session Establishment
    /// Accept wire format (TS 24.501 Table 8.3.2.1.1); there is no legacy
    /// compatibility path.
    pub fn new() -> Self {
        Self {
            sessions: Default::default(),
            pt: ProcedureTransactionManager::new(),
            ursp: super::ursp::UrspPolicy::default(),
            pending: HashMap::new(),
            backoff: HashMap::new(),
            type_override: HashMap::new(),
            wanted: Vec::new(),
            retries: Vec::new(),
            held_retransmissions: std::collections::BTreeMap::new(),
        }
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    /// Whether `psi` is a valid PDU session identity (1..=15, TS 24.501
    /// Section 9.4). Wire-driven PSIs MUST be checked with this before they
    /// index [`SmOrchestrator::sessions`]: the array has 16 slots and an
    /// attacker-controlled PSI > 15 would otherwise panic the UE (remote DoS).
    #[inline]
    fn psi_in_range(psi: u8) -> bool {
        psi != 0 && psi <= PSI_MAX
    }

    /// Get a session by PSI
    pub fn session(&self, psi: u8) -> Option<&PduSession> {
        if !Self::psi_in_range(psi) {
            return None;
        }
        self.sessions[psi as usize].as_ref()
    }

    /// Get the session slot for a PSI, mutably, with a PSI bounds check.
    /// Returns `None` for an out-of-range PSI (never panics) or an empty slot.
    fn session_mut(&mut self, psi: u8) -> Option<&mut PduSession> {
        if !Self::psi_in_range(psi) {
            return None;
        }
        self.sessions[psi as usize].as_mut()
    }

    /// Clear (release) the session slot for a PSI, with a PSI bounds check.
    /// A no-op for an out-of-range PSI (never panics).
    fn clear_session(&mut self, psi: u8) {
        if Self::psi_in_range(psi) {
            self.sessions[psi as usize] = None;
        }
    }

    /// Take (remove and return) the session for a PSI, with a PSI bounds
    /// check. Returns `None` for an out-of-range PSI (never panics).
    fn take_session(&mut self, psi: u8) -> Option<PduSession> {
        if !Self::psi_in_range(psi) {
            return None;
        }
        self.sessions[psi as usize].take()
    }

    /// All sessions currently not inactive
    pub fn active_sessions(&self) -> Vec<&PduSession> {
        self.sessions
            .iter()
            .flatten()
            .filter(|s| s.state != PsState::Inactive)
            .collect()
    }

    /// Whether any session has a procedure in flight (establishment,
    /// modification or release pending). Used by MINT to route a downlink NAS
    /// PDU to the subscription whose SM procedure is awaiting a response on the
    /// shared radio connection (TS 23.761).
    pub fn has_pending_procedure(&self) -> bool {
        self.sessions.iter().flatten().any(|s| {
            matches!(
                s.state,
                PsState::ActivePending | PsState::ModificationPending | PsState::InactivePending
            )
        })
    }

    /// Whether a back-off timer currently blocks the given [S-NSSAI, DNN]
    pub fn is_backed_off(&self, s_nssai: Option<&[u8]>, dnn: Option<&str>) -> bool {
        self.backoff.iter().any(|(key, entry)| {
            let blocking = match &entry.timer {
                Some(t) => t.is_running(),
                None => true, // deactivated timer value: permanent block
            };
            key.covers(s_nssai, dnn) && blocking
        })
    }

    /// Whether a configured session waits to be re-attempted (#379).
    pub fn has_pending_retry(&self) -> bool {
        !self.retries.is_empty()
    }

    /// Whether a configured session's re-attempt is due now: its own wait is
    /// over and no back-off timer bars it (#379). The caller uses this to bring
    /// up a NAS signalling connection (a Service Request) before calling
    /// [`SmOrchestrator::retry_wanted_sessions`] from 5GMM-IDLE.
    pub fn has_due_retry(&self) -> bool {
        self.retries.iter().any(|r| self.retry_is_due(r))
    }

    /// The held retransmissions whose procedures are still pending, to send
    /// now that a NAS signalling connection is up; each one's timer starts
    /// again, keeping its expiry count (TS 24.501 §6.4.1.6 a): "reset and
    /// start timer T3580"). Called on the transition to 5GMM-CONNECTED and on
    /// the tick (#420 review).
    pub fn take_held_retransmissions(&mut self) -> Vec<SmOutput> {
        self.take_held(|_| true)
    }

    /// The held retransmissions of emergency PDU sessions only, which go on
    /// any NAS signalling connection (#424 review).
    pub fn take_held_emergency(&mut self) -> Vec<SmOutput> {
        let emergency: Vec<u8> = self
            .held_retransmissions
            .keys()
            .copied()
            .filter(|pti| self.is_emergency_procedure(*pti))
            .collect();
        self.take_held(|pti| emergency.contains(&pti))
    }

    /// Whether PTI `pti`'s procedure is for an emergency PDU session.
    fn is_emergency_procedure(&self, pti: u8) -> bool {
        self.pt
            .get(pti)
            .and_then(|t| self.session(t.psi()))
            .is_some_and(|s| s.params.emergency)
    }

    /// Whether a held retransmission is an emergency PDU session's.
    pub fn has_held_emergency_retransmission(&self) -> bool {
        self.held_retransmissions
            .keys()
            .any(|pti| self.pending.contains_key(pti) && self.is_emergency_procedure(*pti))
    }

    fn take_held(&mut self, which: impl Fn(u8) -> bool) -> Vec<SmOutput> {
        let mut outs = Vec::new();
        let taken: Vec<u8> = self
            .held_retransmissions
            .keys()
            .copied()
            .filter(|pti| which(*pti))
            .collect();
        for pti in taken {
            self.held_retransmissions.remove(&pti);
            let Some(pending) = self.pending.get(&pti) else {
                continue;
            };
            info!("Held 5GSM retransmission sent on the connection (PTI={pti})");
            outs.push(SmOutput::SendNasPdu(pending.ul_pdu.clone()));
            if let Some(timer) = self.pt.get_mut(pti).and_then(|t| t.timer_mut()) {
                timer.start(false);
            }
        }
        outs
    }

    /// Whether a procedure timer's retransmission waits for a NAS signalling
    /// connection (#409 follow-up).
    pub fn has_held_retransmission(&self) -> bool {
        self.held_retransmissions
            .keys()
            .any(|pti| self.pending.contains_key(pti))
    }

    /// Test hook: make every running procedure timer (T3580/T3581/T3582)
    /// expire on the next tick, as its first expiry.
    #[cfg(test)]
    pub(crate) fn expire_procedure_timers_for_test(&mut self) {
        for pti in self.pt.pending_ptis() {
            if let Some(timer) = self.pt.get_mut(pti).and_then(|t| t.timer_mut()) {
                // The next tick is the next expiry in sequence; a restart
                // then runs the real interval
                timer.run_out_for_test();
            }
        }
    }

    /// Test hook: end every queued re-attempt's own wait now.
    #[cfg(test)]
    pub(crate) fn end_retry_waits_for_test(&mut self) {
        for r in &mut self.retries {
            r.due_at = std::time::Instant::now();
        }
    }

    fn retry_is_due(&self, r: &RetryEntry) -> bool {
        if std::time::Instant::now() < r.due_at {
            return false;
        }
        let resolved = self.resolve_session_params(&self.application_for(&r.base), &r.base);
        resolved.emergency
            || !self.is_backed_off(resolved.s_nssai.as_deref(), resolved.dnn.as_deref())
    }

    /// Remaining seconds of the longest running back-off timer that applies
    /// to a request for the given [S-NSSAI, DNN], if any
    pub fn backoff_remaining(&self, s_nssai: Option<&[u8]>, dnn: Option<&str>) -> Option<u32> {
        self.backoff
            .iter()
            .filter(|(key, _)| key.covers(s_nssai, dnn))
            .filter_map(|(_, e)| e.timer.as_ref().filter(|t| t.is_running()))
            .map(UeTimer::remaining)
            .max()
    }

    // ========================================================================
    // PSI / PTI allocation
    // ========================================================================

    /// Allocate the lowest free PDU session identity (1-15).
    fn allocate_psi(&self) -> Option<u8> {
        (1..=PSI_MAX).find(|&psi| self.sessions[psi as usize].is_none())
    }

    // ========================================================================
    // Procedure initiation
    // ========================================================================

    /// Start establishment for every configured session that has not been
    /// set up yet.
    ///
    /// These are the sessions the UE holds (#379): one whose establishment
    /// fails with a cause that allows asking again, or that a back-off timer
    /// keeps from going out now, is re-attempted by
    /// [`SmOrchestrator::retry_wanted_sessions`] once the back-off timer
    /// expires or is stopped, or after the UE's own wait when the network gave
    /// none (TS 24.501 §6.4.1.4.2).
    pub fn establish_default_sessions(&mut self, params: &[SmSessionParams]) -> Vec<SmOutput> {
        self.wanted = params.to_vec();
        // This pass asks for every configured session again; a queued
        // re-attempt it does not get out is queued afresh.
        self.retries.clear();
        let mut outs = Vec::new();
        for p in params {
            outs.extend(self.establish_wanted(p.clone(), 0));
        }
        outs
    }

    /// Establish the configured session `base` unless a session already
    /// serves it. `failures` counts its consecutive failed establishments.
    fn establish_wanted(&mut self, base: SmSessionParams, failures: u32) -> Vec<SmOutput> {
        // TS 24.526 §5.2 / TS 23.503 §6.6.2.3: resolve the parameters through
        // URSP before deciding whether a session already serves them, so the
        // reuse check compares against what would actually be established.
        // Checking first and resolving second would reuse a session on the
        // wrong slice. A no-op while `ursp_evaluation` is off.
        let resolved = self.resolve_session_params(&self.application_for(&base), &base);
        let already = self.sessions.iter().flatten().any(|s| {
            s.state != PsState::Inactive
                && (s.params == resolved || s.wanted.as_ref() == Some(&base))
        });
        if already {
            return Vec::new();
        }
        self.start_establishment_inner(resolved, Some(base), failures)
    }

    /// Re-attempt every configured session whose re-attempt is due (#379,
    /// TS 24.501 §6.4.1.4.2): its back-off timer expired or was stopped, or the
    /// UE's own wait is over. Call when the UE may send 5GSM signalling: it is
    /// registered with a NAS signalling connection up.
    pub fn retry_wanted_sessions(&mut self) -> Vec<SmOutput> {
        let (due, waiting): (Vec<_>, Vec<_>) = std::mem::take(&mut self.retries)
            .into_iter()
            .partition(|r| self.retry_is_due(r));
        self.retries = waiting;
        let mut outs = Vec::new();
        for r in due {
            info!(
                "Re-attempting the configured PDU session for DNN {:?} / S-NSSAI {:?} \
                 ({} failed attempt(s) so far)",
                r.base.dnn, r.base.s_nssai, r.failures
            );
            outs.extend(self.establish_wanted(r.base, r.failures));
        }
        outs
    }

    /// Queue the configured session `base` for a re-attempt (#379). `failures`
    /// counts its consecutive failed establishments including this one.
    fn queue_retry(&mut self, base: SmSessionParams, failures: u32) {
        if !self.wanted.contains(&base) || self.retries.iter().any(|r| r.base == base) {
            return;
        }
        let resolved = self.resolve_session_params(&self.application_for(&base), &base);
        let backed_off = !resolved.emergency
            && self.is_backed_off(resolved.s_nssai.as_deref(), resolved.dnn.as_deref());
        // A running back-off timer is the wait: the request goes out once it
        // expires or is stopped. Otherwise the network named no wait, and
        // the UE's own one keeps a refusing network from being asked every
        // second.
        let wait = if backed_off { 0 } else { retry_wait(failures) };
        if backed_off {
            info!(
                "Configured PDU session for DNN {:?} / S-NSSAI {:?} re-attempted when its \
                 back-off timer expires or is stopped",
                base.dnn, base.s_nssai
            );
        } else {
            info!(
                "Configured PDU session for DNN {:?} / S-NSSAI {:?} re-attempted in {wait}s",
                base.dnn, base.s_nssai
            );
        }
        self.retries.push(RetryEntry {
            base,
            due_at: std::time::Instant::now() + std::time::Duration::from_secs(u64::from(wait)),
            failures,
        });
    }

    /// Queue a re-attempt for a session whose establishment just failed, if
    /// it serves a configured session (#379).
    fn retry_failed_establishment(&mut self, session: Option<PduSession>) {
        if let Some(PduSession {
            wanted: Some(base),
            failures,
            ..
        }) = session
        {
            self.queue_retry(base, failures.saturating_add(1));
        }
    }

    // ========================================================================
    // URSP evaluation (TS 24.526 §5.2) — #47
    // ========================================================================

    /// Install the URSP policy (configured rules + the evaluation switch).
    pub fn set_ursp_policy(&mut self, policy: super::ursp::UrspPolicy) {
        self.ursp = policy;
    }

    /// Install the network-delivered URSP rules, replacing any previous set.
    pub fn set_delivered_ursp_rules(
        &mut self,
        rules: Vec<nextgsim_nas::messages::mm::ue_policy::UrspRule>,
    ) {
        self.ursp.set_delivered_rules(rules);
    }

    /// Whether URSP evaluation is switched on (observability / test hook).
    pub fn ursp_enabled(&self) -> bool {
        self.ursp.is_enabled()
    }

    /// Number of URSP rules across both sources (observability / test hook).
    pub fn ursp_rule_count(&self) -> usize {
        self.ursp.rule_count()
    }

    /// Resolve session parameters for `app`, applying a matching URSP rule's
    /// route selection descriptor over `base`.
    ///
    /// Returns `base` unchanged when evaluation is off or nothing matches, so
    /// every existing caller is byte-for-byte unaffected until the switch is set.
    pub fn resolve_session_params(
        &self,
        app: &super::ursp::ApplicationDescriptor,
        base: &SmSessionParams,
    ) -> SmSessionParams {
        self.ursp.resolve(app, base)
    }

    /// The application information a configured session offers for matching.
    ///
    /// A configured session names a DNN and nothing else -- there is no detected
    /// application behind it -- so only DNN-keyed and match-all rules can steer
    /// it. Rules keyed on an OS App Id or a flow 5-tuple need a real detection;
    /// [`Self::start_establishment_for_app`] is the entry point for those.
    fn application_for(&self, params: &SmSessionParams) -> super::ursp::ApplicationDescriptor {
        super::ursp::ApplicationDescriptor {
            dnn: params.dnn.clone(),
            ..Default::default()
        }
    }

    /// Establish (or reuse) a PDU session for a detected application, with the
    /// parameters URSP selects (TS 23.503 §6.6.2.3).
    ///
    /// Returns `(psi, outputs)`. When an active session already carries the
    /// resolved parameters its PSI is returned with no outputs — §6.6.2.3's
    /// "can be associated with an existing PDU session" — so an application does
    /// not establish a duplicate session for a route it already has.
    pub fn start_establishment_for_app(
        &mut self,
        app: &super::ursp::ApplicationDescriptor,
        base: &SmSessionParams,
    ) -> (Option<u8>, Vec<SmOutput>) {
        let resolved = self.resolve_session_params(app, base);
        if let Some(psi) = self
            .sessions
            .iter()
            .flatten()
            .find(|s| s.params == resolved && s.state != PsState::Inactive)
            .map(|s| s.psi)
        {
            debug!("URSP: application bound to existing PDU session {psi}");
            return (Some(psi), Vec::new());
        }
        // `start_establishment` allocates the PSI internally and returns only
        // outputs, so the new session has to be identified by DIFFERENCE. Finding
        // "the session that is ActivePending" is wrong: an earlier establishment
        // that has not been accepted yet is also ActivePending, so a second
        // application would be told it got the first one's PSI.
        let occupied: Vec<u8> = self.sessions.iter().flatten().map(|s| s.psi).collect();
        let outs = self.start_establishment(resolved);
        let psi = self
            .sessions
            .iter()
            .flatten()
            .map(|s| s.psi)
            .find(|psi| !occupied.contains(psi));
        (psi, outs)
    }

    /// Start a PDU session establishment procedure (TS 24.501 6.4.1.2).
    ///
    /// A one-off request: it is not re-attempted when it fails. The configured
    /// sessions go through [`SmOrchestrator::establish_default_sessions`].
    pub fn start_establishment(&mut self, params: SmSessionParams) -> Vec<SmOutput> {
        self.start_establishment_inner(params, None, 0)
    }

    /// Start a PDU session establishment procedure (TS 24.501 6.4.1.2) for
    /// `params`, serving the configured session `wanted` if any. A configured
    /// session's request that cannot go out now is queued for a re-attempt.
    fn start_establishment_inner(
        &mut self,
        mut params: SmSessionParams,
        wanted: Option<SmSessionParams>,
        failures: u32,
    ) -> Vec<SmOutput> {
        // Back-off consultation (TS 24.501 6.4.1.4 / 6.2.8)
        if self.is_backed_off(params.s_nssai.as_deref(), params.dnn.as_deref()) && !params.emergency
        {
            warn!(
                "PDU session establishment for DNN {:?} blocked by back-off timer ({:?}s left)",
                params.dnn,
                self.backoff_remaining(params.s_nssai.as_deref(), params.dnn.as_deref())
            );
            if let Some(base) = wanted {
                self.queue_retry(base, failures);
            }
            return Vec::new();
        }

        // Apply the PDU session type restriction learned from a previous
        // cause #50/#51/#57 reject for this DNN (TS 24.501 6.4.1.4.2)
        if let Some(forced) = self.type_override.get(&params.dnn) {
            if params.session_type != *forced {
                info!(
                    "PDU session type {:?} overridden to {:?} (network restriction for DNN {:?})",
                    params.session_type, forced, params.dnn
                );
                params.session_type = *forced;
            }
        }

        let Some(psi) = self.allocate_psi() else {
            warn!("No free PDU session identity");
            if let Some(base) = wanted {
                self.queue_retry(base, failures.saturating_add(1));
            }
            return Vec::new();
        };
        let Some(pti) = self.pt.allocate() else {
            warn!("No free procedure transaction identity");
            if let Some(base) = wanted {
                self.queue_retry(base, failures.saturating_add(1));
            }
            return Vec::new();
        };

        // Build the PDU Session Establishment Request (TS 24.501 8.3.1)
        let mut req = PduSessionEstablishmentRequest::new(
            psi,
            pti,
            IeIntegrityProtectionMaxDataRate::full_rate(),
        );
        req.pdu_session_type = Some(IeSelectedPduSessionType::new(params.session_type));
        req.ssc_mode = Some(IeSelectedSscMode::new(params.ssc_mode));
        // 5GSM capability: no reflective QoS / multi-homed IPv6 support
        req.sm_capability = Some(vec![0x00]);
        let mut sm_pdu = Vec::new();
        req.encode(&mut sm_pdu);

        // Wrap in UL NAS Transport (TS 24.501 5.4.5.2.2): payload container
        // type 1, PDU session ID, request type, S-NSSAI and DNN
        let request_type = if params.emergency {
            RequestType::InitialEmergencyRequest
        } else {
            RequestType::InitialRequest
        };
        let ul_pdu = build_ul_transport(
            sm_pdu,
            psi,
            Some(request_type),
            params.s_nssai.clone(),
            params.dnn.as_deref().map(encode_dnn_value),
        );

        let mut session = PduSession::new(psi, params);
        session.state = PsState::ActivePending;
        session.wanted = wanted;
        session.failures = failures;
        self.sessions[psi as usize] = Some(session);

        if let Some(pt) = self.pt.get_mut(pti) {
            pt.start(
                psi,
                PtMessageType::PduSessionEstablishmentRequest,
                SM_TIMER_T3580,
            );
        }
        self.pending.insert(
            pti,
            PendingProcedure {
                ul_pdu: ul_pdu.clone(),
            },
        );

        info!("Sending PDU Session Establishment Request: PSI={psi}, PTI={pti}, T3580 started");
        vec![SmOutput::SendNasPdu(ul_pdu)]
    }

    /// Start a UE-initiated PDU session modification (TS 24.501 6.4.2.2).
    pub fn start_modification(&mut self, psi: u8) -> Vec<SmOutput> {
        let Some(session) = self.session(psi) else {
            warn!("Modification requested for unknown PSI {psi}");
            return Vec::new();
        };
        if session.state != PsState::Active {
            warn!(
                "Modification requested for PSI {psi} in state {:?}",
                session.state
            );
            return Vec::new();
        }
        let (s_nssai, dnn) = (session.params.s_nssai.clone(), session.params.dnn.clone());
        if self.is_backed_off(s_nssai.as_deref(), dnn.as_deref()) {
            warn!("Modification for PSI {psi} blocked by back-off timer");
            return Vec::new();
        }
        let Some(pti) = self.pt.allocate() else {
            warn!("No free procedure transaction identity");
            return Vec::new();
        };

        let mut req = PduSessionModificationRequest::new(psi, pti);
        req.integrity_protection_max_data_rate = Some([0xFF, 0xFF]);
        let mut sm_pdu = Vec::new();
        req.encode(&mut sm_pdu);
        let ul_pdu = build_ul_transport(sm_pdu, psi, None, None, None);

        if let Some(s) = self.sessions[psi as usize].as_mut() {
            s.state = PsState::ModificationPending;
        }
        if let Some(pt) = self.pt.get_mut(pti) {
            pt.start(
                psi,
                PtMessageType::PduSessionModificationRequest,
                SM_TIMER_T3581,
            );
        }
        self.pending.insert(
            pti,
            PendingProcedure {
                ul_pdu: ul_pdu.clone(),
            },
        );

        info!("Sending PDU Session Modification Request: PSI={psi}, PTI={pti}, T3581 started");
        vec![SmOutput::SendNasPdu(ul_pdu)]
    }

    /// Start a UE-initiated PDU session release (TS 24.501 6.4.3.2).
    pub fn start_release(&mut self, psi: u8) -> Vec<SmOutput> {
        let Some(session) = self.session(psi) else {
            warn!("Release requested for unknown PSI {psi}");
            return Vec::new();
        };
        if !matches!(
            session.state,
            PsState::Active | PsState::ModificationPending
        ) {
            warn!(
                "Release requested for PSI {psi} in state {:?}",
                session.state
            );
            return Vec::new();
        }
        self.send_release_request(psi, SmCause::RegularDeactivation)
    }

    /// Send a PDU SESSION RELEASE REQUEST for `psi` with `cause` on a fresh PTI, and
    /// start T3582 (TS 24.501 §6.4.3.2). The session goes to INACTIVE PENDING.
    fn send_release_request(&mut self, psi: u8, cause: SmCause) -> Vec<SmOutput> {
        let Some(pti) = self.pt.allocate() else {
            warn!("No free procedure transaction identity");
            return Vec::new();
        };

        // The UE no longer holds a configured session it releases itself, so
        // nothing re-establishes it (#379)
        if let Some(base) = self.session(psi).and_then(|s| s.wanted.clone()) {
            self.wanted.retain(|w| *w != base);
            self.retries.retain(|r| r.base != base);
        }

        let mut req = PduSessionReleaseRequest::new(psi, pti);
        req.sm_cause = Some(ie3_cause(cause));
        let mut sm_pdu = Vec::new();
        req.encode(&mut sm_pdu);
        let ul_pdu = build_ul_transport(sm_pdu, psi, None, None, None);

        if let Some(s) = self.sessions[psi as usize].as_mut() {
            s.state = PsState::InactivePending;
        }
        if let Some(pt) = self.pt.get_mut(pti) {
            pt.start(psi, PtMessageType::PduSessionReleaseRequest, SM_TIMER_T3582);
        }
        self.pending.insert(
            pti,
            PendingProcedure {
                ul_pdu: ul_pdu.clone(),
            },
        );

        info!(
            "Sending PDU Session Release Request: PSI={psi}, PTI={pti}, cause #{}, T3582 started",
            cause as u8
        );
        vec![SmOutput::SendNasPdu(ul_pdu)]
    }

    /// Reconcile UE session state against the network's PDU session status IE
    /// (TS 24.501 §5.5.1.3.4 on REGISTRATION ACCEPT, §5.6.1.4.2 a) on SERVICE
    /// ACCEPT): locally release every session that is active on the UE but
    /// indicated as **inactive** by the network.
    ///
    /// `status` is the little-endian IE value, where bit *n* set means PSI *n*
    /// is active in the network (TS 24.501 §9.11.2.2).
    ///
    /// Releases are LOCAL: no PDU SESSION RELEASE REQUEST is sent, because the
    /// network has already released these sessions — that is what the IE is
    /// telling us — so signalling a release for them would ask it to release a
    /// session it does not have.
    ///
    /// Sessions in a *pending* state are reconciled too. A session the UE is
    /// still establishing (`ActivePending`) that the network reports inactive is
    /// one whose establishment the network has abandoned, and leaving it pending
    /// would pin the PSI and its PTI until a timer the peer will never answer.
    pub fn reconcile_pdu_session_status(&mut self, status: u16) -> Vec<SmOutput> {
        let mut outs = Vec::new();
        for psi in 1..=PSI_MAX {
            let active_on_ue = self
                .session(psi)
                .is_some_and(|s| s.state != PsState::Inactive);
            let active_in_network = status & (1u16 << psi) != 0;
            if active_on_ue && !active_in_network {
                warn!(
                    "PDU session status IE marks PSI {psi} inactive while it is active on \
                     the UE: releasing locally (TS 24.501 §5.6.1.4.2)"
                );
                self.clear_session(psi);
                // Abort any in-flight procedure for this PSI so its PTI is
                // returned and its timer stopped, rather than left retransmitting
                // against a session that no longer exists.
                for (pti, _) in self.pt.abort_by_psi(psi) {
                    self.pending.remove(&pti);
                }
                outs.push(SmOutput::SessionReleased { psi });
            }
        }
        outs
    }

    /// Release all sessions locally (e.g. on deregistration).
    pub fn release_all_locally(&mut self) -> Vec<SmOutput> {
        let mut outs = Vec::new();
        for psi in 1..=PSI_MAX {
            if let Some(session) = self.sessions[psi as usize].take() {
                if session.state != PsState::Inactive {
                    outs.push(SmOutput::SessionReleased { psi });
                }
            }
        }
        for pti in self.pt.pending_ptis() {
            self.pt.abort(pti);
            self.pending.remove(&pti);
        }
        // The next registration asks for the configured sessions again
        // (establish_default_sessions); a re-attempt queued before must not race
        // it from 5GMM-DEREGISTERED.
        self.retries.clear();
        outs
    }

    // ========================================================================
    // Timer driving
    // ========================================================================

    /// Drive the SM procedure and back-off timers with a NAS signalling
    /// connection to send on; call roughly once a second.
    pub fn tick(&mut self) -> Vec<SmOutput> {
        self.tick_with_connection(true)
    }

    /// Drive the SM procedure and back-off timers; call roughly once a
    /// second. `connected`: the UE is in 5GMM-CONNECTED, so an uplink 5GSM
    /// message can go.
    ///
    /// TS 24.501 §6.4.1.6 a) / §6.4.2.6 a) / §6.4.3.6 a): a procedure timer's
    /// expiry retransmits the request. Without a connection the
    /// retransmission is held (the expiry still counts, and the timer
    /// restarts), and it goes on the first tick with one, if the procedure is
    /// still pending; the fifth expiry aborts as always (#409 follow-up).
    pub fn tick_with_connection(&mut self, connected: bool) -> Vec<SmOutput> {
        self.tick_with_transport(connected, connected)
    }

    /// [`Self::tick_with_connection`] with the emergency case apart (#424
    /// review): `connected`, a NAS signalling connection exists; `ready`, a
    /// 5GSM message other than an emergency PDU session's can go on it (not
    /// while the SERVICE REQUEST that asked for it is unanswered). An
    /// emergency PDU session's retransmission goes on any connection,
    /// including in 5GMM-REGISTERED.LIMITED-SERVICE (TS 24.501 §5.3.20,
    /// §6.4.1.2), and is held only when there is none.
    pub fn tick_with_transport(&mut self, connected: bool, ready: bool) -> Vec<SmOutput> {
        let mut outs = Vec::new();

        if ready {
            outs.extend(self.take_held_retransmissions());
        } else if connected {
            outs.extend(self.take_held_emergency());
        }

        // A hold is bounded: past its deadline the procedure is aborted as on
        // its last expiry (§6.4.1.6 a), §6.4.2.6 a), §6.4.3.6 a)).
        let now = std::time::Instant::now();
        let overdue: Vec<u8> = self
            .held_retransmissions
            .iter()
            .filter(|(_, deadline)| now >= **deadline)
            .map(|(pti, _)| *pti)
            .collect();
        for pti in overdue {
            warn!("5GSM retransmission (PTI={pti}) held past its timers' span; procedure aborted");
            outs.extend(self.abort_procedure(pti));
        }

        // Procedure timers (T3580 / T3581 / T3582)
        for pti in self.pt.check_timers() {
            for out in self.handle_procedure_timer_expiry(pti) {
                match out {
                    SmOutput::SendNasPdu(_)
                        if !(ready || (connected && self.is_emergency_procedure(pti))) =>
                    {
                        // §6.4.1.6 a): the timer supervises a retransmission
                        // that has not been sent; it stops until the flush,
                        // so idle time does not use up the four retransmissions
                        // (#420 review)
                        info!(
                            "5GSM retransmission (PTI={pti}) held: no NAS signalling connection \
                             (TS 24.501 §5.6.1.1); its timer waits for it"
                        );
                        let deadline = self.hold_deadline(pti);
                        if let Some(timer) = self.pt.get_mut(pti).and_then(|t| t.timer_mut()) {
                            timer.stop(false);
                        }
                        self.held_retransmissions.insert(pti, deadline);
                    }
                    out => outs.push(out),
                }
            }
        }

        // Back-off timers: drop entries whose timer has expired. A configured
        // session it barred is re-attempted by `retry_wanted_sessions` (#379,
        // TS 24.501 §6.4.1.4.2: no new request "until timer T3396 expires").
        self.backoff
            .retain(|key, entry| match entry.timer.as_mut() {
                Some(timer) => {
                    if timer.perform_tick() {
                        info!(
                            "Back-off timer expired for S-NSSAI {:?} / DNN {:?} ({})",
                            key.s_nssai, key.dnn, entry.name
                        );
                        false
                    } else {
                        timer.is_running()
                    }
                }
                None => true, // deactivated: keep blocking
            });

        outs
    }

    /// Retransmit or abort on procedure timer expiry (TS 24.501 6.4.1.6 /
    /// 6.4.2.6 / 6.4.3.6: retransmit on expiries 1-4, abort on the fifth).
    fn handle_procedure_timer_expiry(&mut self, pti: u8) -> Vec<SmOutput> {
        let Some(pt) = self.pt.get(pti) else {
            return Vec::new();
        };
        let Some(msg_type) = pt.message_type() else {
            return Vec::new();
        };
        let psi = pt.psi();
        let expiry_count = pt.timer().map_or(0, UeTimer::expiry_count);
        let (timer_code, max_retries) = match msg_type {
            PtMessageType::PduSessionEstablishmentRequest => (SM_TIMER_T3580, MAX_T3580_RETRIES),
            PtMessageType::PduSessionModificationRequest => (SM_TIMER_T3581, MAX_T3581_RETRIES),
            PtMessageType::PduSessionReleaseRequest => (SM_TIMER_T3582, MAX_T3582_RETRIES),
        };

        if expiry_count < max_retries {
            // Retransmit and restart the timer without clearing the count
            let Some(pending) = self.pending.get(&pti) else {
                return Vec::new();
            };
            let ul_pdu = pending.ul_pdu.clone();
            warn!(
                "T{timer_code} expired ({expiry_count} of {max_retries}): retransmitting {msg_type} (PSI={psi}, PTI={pti})"
            );
            if let Some(pt) = self.pt.get_mut(pti) {
                if let Some(timer) = pt.timer_mut() {
                    timer.start(false);
                }
            }
            return vec![SmOutput::SendNasPdu(ul_pdu)];
        }

        // Exhaustion: abort with the per-procedure abnormal action
        warn!("T{timer_code} exhausted after {expiry_count} expiries: aborting {msg_type} (PSI={psi}, PTI={pti})");
        self.abort_procedure(pti)
    }

    /// The (timer code, retransmission limit) of a procedure type.
    fn procedure_timer_limits(msg_type: PtMessageType) -> (u16, u32) {
        match msg_type {
            PtMessageType::PduSessionEstablishmentRequest => (SM_TIMER_T3580, MAX_T3580_RETRIES),
            PtMessageType::PduSessionModificationRequest => (SM_TIMER_T3581, MAX_T3581_RETRIES),
            PtMessageType::PduSessionReleaseRequest => (SM_TIMER_T3582, MAX_T3582_RETRIES),
        }
    }

    /// When a retransmission held now must have gone: the span its remaining
    /// expiries would have taken (the one held plus those left), at the
    /// timer's interval.
    fn hold_deadline(&self, pti: u8) -> std::time::Instant {
        let (interval, count, max) = self
            .pt
            .get(pti)
            .and_then(|t| {
                let timer = t.timer()?;
                let (_, max) = Self::procedure_timer_limits(t.message_type()?);
                Some((timer.interval(), timer.expiry_count(), max))
            })
            .unwrap_or((0, 0, 0));
        let left = max.saturating_sub(count) + 1;
        std::time::Instant::now()
            + std::time::Duration::from_secs(u64::from(interval) * u64::from(left))
    }

    /// Abort PTI `pti`'s procedure with its abnormal-case action (§6.4.1.6 a),
    /// §6.4.2.6 a), §6.4.3.6 a)): release the PTI and act per procedure.
    fn abort_procedure(&mut self, pti: u8) -> Vec<SmOutput> {
        let Some(pt) = self.pt.get(pti) else {
            return Vec::new();
        };
        let Some(msg_type) = pt.message_type() else {
            return Vec::new();
        };
        let psi = pt.psi();
        self.pt.abort(pti);
        self.pending.remove(&pti);
        self.held_retransmissions.remove(&pti);

        match msg_type {
            PtMessageType::PduSessionEstablishmentRequest => {
                // 6.4.1.6 d): abort the procedure, release the allocated PSI.
                // A configured session is asked for again after the UE's own
                // wait (#379).
                let session = self.take_session(psi);
                self.retry_failed_establishment(session);
                vec![SmOutput::SessionEstablishmentFailed {
                    psi,
                    cause: SmCause::ProtocolErrorUnspecified,
                }]
            }
            PtMessageType::PduSessionModificationRequest => {
                // 6.4.2.6 d): abort the procedure; the session stays active
                if let Some(s) = self.sessions[psi as usize].as_mut() {
                    s.state = PsState::Active;
                }
                vec![SmOutput::SessionModificationFailed {
                    psi,
                    cause: SmCause::ProtocolErrorUnspecified,
                }]
            }
            PtMessageType::PduSessionReleaseRequest => {
                // 6.4.3.6 c): abort the procedure and release the PDU
                // session locally
                self.sessions[psi as usize] = None;
                vec![SmOutput::SessionReleased { psi }]
            }
        }
    }

    // ========================================================================
    // Downlink handling
    // ========================================================================

    /// Process a (deciphered, plain) DL NAS Transport message carrying an
    /// N1 SM container (TS 24.501 5.4.5.3).
    pub fn handle_dl_nas_transport(&mut self, dl: &DlNasTransport) -> Vec<SmOutput> {
        if dl.payload_container_type != PayloadContainerType::N1SmInformation {
            debug!(
                "DL NAS Transport container type {:?} not owned by the SM orchestrator",
                dl.payload_container_type
            );
            return Vec::new();
        }

        // 5GMM cause present: the network did not forward our payload
        // (e.g. cause #90 "payload was not forwarded", TS 24.501 5.4.5.3.1)
        if let Some(cause) = dl.mm_cause {
            return self.handle_payload_not_forwarded(dl, cause);
        }

        self.handle_n1_sm_container(&dl.payload_container, dl.pdu_session_id)
    }

    /// Abort the pending procedure whose UL message the network echoed back
    /// with a 5GMM cause (TS 24.501 5.4.5.3.1).
    fn handle_payload_not_forwarded(&mut self, dl: &DlNasTransport, cause: u8) -> Vec<SmOutput> {
        warn!("DL NAS Transport with 5GMM cause #{cause}: payload not forwarded");
        let container = &dl.payload_container;
        if container.len() < 4 || container[0] != 0x2E {
            return Vec::new();
        }
        let psi = container[1];
        let pti = container[2];
        // `psi` is taken straight off the wire (TS 24.501 Section 9.4: valid
        // range 1..=15). Reject an out-of-range PSI before any
        // `self.sessions[psi]` access: the array has only 16 slots and a
        // remote PSI > 15 would panic the UE (remote DoS). The procedure
        // transaction is identified by the PTI alone, so an out-of-range PSI
        // here is a malformed/forged echo and is dropped.
        if !Self::psi_in_range(psi) {
            warn!("Payload-not-forwarded echo carries out-of-range PSI {psi}: dropping");
            return Vec::new();
        }
        let Some((msg_type, _)) = self.pt.abort(pti) else {
            return Vec::new();
        };
        self.pending.remove(&pti);
        match msg_type {
            PtMessageType::PduSessionEstablishmentRequest => {
                // TS 24.501 §5.4.5.3.3 g): the request was not forwarded. A
                // configured session is asked for again (#379):
                // - #22 / #67 / #69: congestion control, with the Back-off
                //   timer value as T3396 / T3584 / T3585 (§6.4.1.4.2);
                // - #90 (routing failure, e.g. no SMF reachable): per the UE's
                //   own policy; the Back-off timer value IE is ignored.
                // Any other cause is not re-attempted.
                let session = self.take_session(psi);
                let congestion = match cause {
                    22 => Some(SmCause::InsufficientResources),
                    67 => Some(SmCause::InsufficientResourcesForSliceAndDnn),
                    69 => Some(SmCause::InsufficientResourcesForSlice),
                    _ => None,
                };
                if let Some(sm_cause) = congestion {
                    if let Some(p) = session.as_ref().map(|s| s.params.clone()) {
                        self.arm_backoff(sm_cause, &p, dl.back_off_timer_value);
                    }
                    self.retry_failed_establishment(session);
                } else if cause == 90 {
                    self.retry_failed_establishment(session);
                }
                vec![SmOutput::SessionEstablishmentFailed {
                    psi,
                    cause: SmCause::NetworkFailure,
                }]
            }
            PtMessageType::PduSessionModificationRequest => {
                if let Some(s) = self.session_mut(psi) {
                    s.state = PsState::Active;
                }
                vec![SmOutput::SessionModificationFailed {
                    psi,
                    cause: SmCause::NetworkFailure,
                }]
            }
            PtMessageType::PduSessionReleaseRequest => {
                self.clear_session(psi);
                vec![SmOutput::SessionReleased { psi }]
            }
        }
    }

    /// Dispatch a plain 5GSM message from the N1 SM container.
    fn handle_n1_sm_container(&mut self, container: &[u8], dl_psi: Option<u8>) -> Vec<SmOutput> {
        if container.len() < 4 {
            warn!("N1 SM container too short: {} bytes", container.len());
            return Vec::new();
        }
        if container[0] != 0x2E {
            warn!("N1 SM container with wrong EPD 0x{:02x}", container[0]);
            return Vec::new();
        }
        let psi = container[1];
        let pti = container[2];
        let Ok(msg_type) = NasSmMessageType::try_from(container[3]) else {
            warn!("Unknown 5GSM message type 0x{:02x}", container[3]);
            return self.send_sm_status(psi, pti, SmCause::MessageTypeNonExistent);
        };
        if let Some(dl_psi) = dl_psi {
            if dl_psi != psi {
                warn!("DL NAS Transport PSI {dl_psi} != container PSI {psi}");
            }
        }
        debug!("5GSM message {msg_type:?} (PSI={psi}, PTI={pti})");
        let body = &container[4..];

        match msg_type {
            NasSmMessageType::PduSessionEstablishmentAccept => {
                self.handle_establishment_accept(body, psi, pti)
            }
            NasSmMessageType::PduSessionEstablishmentReject => {
                self.handle_establishment_reject(body, psi, pti)
            }
            NasSmMessageType::PduSessionModificationCommand => {
                self.handle_modification_command(body, psi, pti)
            }
            NasSmMessageType::PduSessionModificationReject => {
                self.handle_modification_reject(body, psi, pti)
            }
            NasSmMessageType::PduSessionReleaseCommand => {
                self.handle_release_command(body, psi, pti)
            }
            NasSmMessageType::PduSessionReleaseReject => self.handle_release_reject(body, psi, pti),
            NasSmMessageType::FiveGSmStatus => {
                let cause = body.first().copied().unwrap_or(111);
                warn!("5GSM Status received: cause #{cause} (PSI={psi}, PTI={pti})");
                // TS 24.501 6.7: abort the related procedure transaction
                self.abort_pt_local(pti);
                Vec::new()
            }
            _ => {
                warn!("5GSM message {msg_type:?} not compatible with the UE protocol state");
                self.send_sm_status(psi, pti, SmCause::MessageTypeNotCompatible)
            }
        }
    }

    /// Validate that the PTI in a Class-1 response matches a pending
    /// transaction of the expected type for the given PSI.
    /// Returns the per-cause 5GSM Status the UE must send on failure
    /// (TS 24.501 Section 7.3.1: causes #81 / #47).
    fn validate_response_pt(
        &self,
        pti: u8,
        psi: u8,
        expected: PtMessageType,
    ) -> Result<(), SmCause> {
        let Some(pt) = self.pt.get(pti) else {
            return Err(SmCause::InvalidPtiValue);
        };
        if !pt.is_pending() || pt.message_type() != Some(expected) {
            return Err(SmCause::InvalidPtiValue);
        }
        if pt.psi() != psi {
            return Err(SmCause::PtiMismatch);
        }
        Ok(())
    }

    /// PDU Session Establishment Accept (TS 24.501 6.4.1.3)
    fn handle_establishment_accept(&mut self, body: &[u8], psi: u8, pti: u8) -> Vec<SmOutput> {
        if let Err(cause) =
            self.validate_response_pt(pti, psi, PtMessageType::PduSessionEstablishmentRequest)
        {
            warn!("Establishment Accept with invalid PTI {pti} / PSI {psi}");
            return self.send_sm_status(psi, pti, cause);
        }

        let acc = match PduSessionEstablishmentAccept::decode(&mut &body[..], psi, pti) {
            Ok(acc) => acc,
            Err(e) => {
                // TS 24.501 §7.5.3 a): a mandatory-IE error in a PDU SESSION
                // ESTABLISHMENT ACCEPT is answered by releasing the session, with a
                // PDU SESSION RELEASE REQUEST carrying #96 (nextgsim #383). It is the
                // listed exception to §7.5.1's 5GSM STATUS. The network has
                // established the session, so the establishment procedure ends here
                // (T3580 stops) and the release procedure takes over (T3582). Only
                // the spec-conformant wire format (Table 8.3.2.1.1) is accepted.
                warn!(
                    "Establishment Accept failed strict decoding: {e:?}; releasing PSI {psi} \
                     with #96 (TS 24.501 §7.5.3 a)"
                );
                self.pt.abort(pti);
                self.pending.remove(&pti);
                // The attempt has failed for the layers above (#386 review); the
                // release request then ends the session the network holds.
                let mut outs = vec![SmOutput::SessionEstablishmentFailed {
                    psi,
                    cause: SmCause::InvalidMandatoryInformation,
                }];
                outs.extend(self.send_release_request(psi, SmCause::InvalidMandatoryInformation));
                return outs;
            }
        };

        // Procedure completed: stop T3580, release the PTI
        self.pt.abort(pti);
        self.pending.remove(&pti);

        let ipv4 = acc
            .pdu_address
            .as_ref()
            .and_then(|addr| match addr.address_type {
                PduAddressType::Ipv4 if addr.address.len() >= 4 => Some([
                    addr.address[0],
                    addr.address[1],
                    addr.address[2],
                    addr.address[3],
                ]),
                PduAddressType::Ipv4v6 if addr.address.len() >= 12 => Some([
                    addr.address[8],
                    addr.address[9],
                    addr.address[10],
                    addr.address[11],
                ]),
                _ => None,
            });

        if let Some(s) = self.session_mut(psi) {
            s.state = PsState::Active;
            s.selected_type = Some(acc.selected_pdu_session_type.value);
            s.selected_ssc_mode = Some(acc.selected_ssc_mode.value);
            s.ipv4 = ipv4;
            s.authorized_qos_rules = acc.authorized_qos_rules.data.clone();
            s.session_ambr = Some(acc.session_ambr.clone());
        }

        info!(
            "PDU Session Establishment Accept: PDU session {psi} established: type {:?}, SSC mode {:?}, IPv4 {:?}, \
             AMBR DL unit {} value {}",
            acc.selected_pdu_session_type.value,
            acc.selected_ssc_mode.value,
            ipv4,
            acc.session_ambr.downlink_unit,
            acc.session_ambr.downlink
        );
        vec![SmOutput::SessionEstablished { psi, ipv4 }]
    }

    /// PDU Session Establishment Reject (TS 24.501 6.4.1.4)
    fn handle_establishment_reject(&mut self, body: &[u8], psi: u8, pti: u8) -> Vec<SmOutput> {
        if let Err(cause) =
            self.validate_response_pt(pti, psi, PtMessageType::PduSessionEstablishmentRequest)
        {
            warn!("Establishment Reject with invalid PTI {pti} / PSI {psi}");
            return self.send_sm_status(psi, pti, cause);
        }

        let rej = match PduSessionEstablishmentReject::decode(&mut &body[..], psi, pti) {
            Ok(rej) => rej,
            Err(e) => {
                warn!("Failed to decode Establishment Reject: {e:?}");
                return self.send_sm_status(psi, pti, SmCause::InvalidMandatoryInformation);
            }
        };
        let cause = rej.cause();
        warn!(
            "PDU Session Establishment Reject: cause {:?} (#{})",
            cause, cause as u8
        );

        // Stop T3580, release the PTI and the allocated PSI
        self.pt.abort(pti);
        self.pending.remove(&pti);
        let session = self.take_session(psi);
        let params = session.as_ref().map(|s| s.params.clone());
        let wanted = session.as_ref().and_then(|s| s.wanted.clone());
        let failures = session.as_ref().map_or(0, |s| s.failures);

        let mut outs = Vec::new();
        match cause {
            // #26 / #67 / #69 / #70: back-off timer handling
            // (TS 24.501 6.4.1.4: scope depends on the cause)
            SmCause::InsufficientResources
            | SmCause::InsufficientResourcesForSliceAndDnn
            | SmCause::InsufficientResourcesForSlice
            | SmCause::MissingOrUnknownDnnInSlice => {
                if let Some(ref p) = params {
                    self.arm_backoff(cause, p, rej.back_off_timer_value);
                }
                // A configured session is asked for again once the back-off
                // timer expires or is stopped, or after the UE's own wait when
                // the timer value was zero or absent (#379, §6.4.1.4.2). A
                // deactivated value bars it until a command in its scope.
                self.retry_failed_establishment(session);
            }
            // #50 / #51 / #57: PDU session type restriction; retry once
            // with the network-allowed type (TS 24.501 6.4.1.4.2)
            SmCause::PduSessionTypeIpv4OnlyAllowed
            | SmCause::PduSessionTypeIpv6OnlyAllowed
            | SmCause::PduSessionTypeIpv4v6OnlyAllowed => {
                let allowed = match cause {
                    SmCause::PduSessionTypeIpv4OnlyAllowed => PduSessionTypeValue::Ipv4,
                    SmCause::PduSessionTypeIpv6OnlyAllowed => PduSessionTypeValue::Ipv6,
                    _ => PduSessionTypeValue::Ipv4v6,
                };
                if let Some(mut p) = params.clone() {
                    self.type_override.insert(p.dnn.clone(), allowed);
                    if p.session_type != allowed {
                        info!("Retrying establishment with allowed PDU session type {allowed:?}");
                        p.session_type = allowed;
                        outs.extend(self.start_establishment_inner(p, wanted, failures));
                    }
                }
            }
            // #68: not supported SSC mode; retry with the allowed SSC mode
            // when signalled (TS 24.501 6.4.1.4.2)
            SmCause::NotSupportedSscMode => {
                if let (Some(mut p), Some(allowed)) = (params.clone(), rej.allowed_ssc_mode) {
                    if let Ok(ssc) = SscModeValue::try_from(allowed) {
                        if p.ssc_mode != ssc {
                            info!("Retrying establishment with allowed SSC mode {ssc:?}");
                            p.ssc_mode = ssc;
                            outs.extend(self.start_establishment_inner(p, wanted, failures));
                        }
                    }
                }
            }
            // #28, #27 and everything else: no automatic retry
            _ => {}
        }

        outs.push(SmOutput::SessionEstablishmentFailed { psi, cause });
        outs
    }

    /// Arm the T3396-class back-off timer for the scope mandated by the
    /// reject cause (TS 24.501 6.4.1.4).
    fn arm_backoff(&mut self, cause: SmCause, params: &SmSessionParams, timer_byte: Option<u8>) {
        let dnn = Scope::Is(params.dnn.clone());
        let s_nssai = Scope::Is(params.s_nssai.clone());
        let (key, name, code) = match cause {
            // #26: T3396, associated with the DNN, or with no DNN (§6.2.7)
            SmCause::InsufficientResources => (
                BackoffKey {
                    timer: SM_TIMER_BACKOFF,
                    s_nssai: Scope::Any,
                    dnn,
                },
                "T3396",
                SM_TIMER_BACKOFF,
            ),
            // #69: T3585, associated with the S-NSSAI only (§6.2.8)
            SmCause::InsufficientResourcesForSlice => (
                BackoffKey {
                    timer: SM_TIMER_T3585,
                    s_nssai,
                    dnn: Scope::Any,
                },
                "T3585",
                SM_TIMER_T3585,
            ),
            // #67: T3584, associated with the [S-NSSAI, DNN] combination
            SmCause::InsufficientResourcesForSliceAndDnn => (
                BackoffKey {
                    timer: SM_TIMER_T3584,
                    s_nssai,
                    dnn,
                },
                "T3584",
                SM_TIMER_T3584,
            ),
            // #70: the §6.4.1.4.3 back-off timer, for the [S-NSSAI, DNN]
            // combination
            _ => (
                BackoffKey {
                    timer: SM_TIMER_GENERIC_BACKOFF,
                    s_nssai,
                    dnn,
                },
                "back-off timer",
                SM_TIMER_GENERIC_BACKOFF,
            ),
        };
        let stoppable = cause != SmCause::MissingOrUnknownDnnInSlice;

        let Some(byte) = timer_byte else {
            if !stoppable {
                // §6.4.1.4.3 a): #70 without the IE uses the SM Retry Timer,
                // 12 minutes when none is configured
                info!("#70 without a back-off timer value: {name} started for {key:?}: {SM_RETRY_TIMER_DEFAULT_SECS}s");
                let mut timer = UeTimer::new(code, false, SM_RETRY_TIMER_DEFAULT_SECS);
                timer.start(true);
                self.backoff.insert(
                    key,
                    BackoffEntry {
                        name,
                        stoppable,
                        timer: Some(timer),
                    },
                );
                return;
            }
            // No back-off timer IE: the UE may ask again, after a local
            // back-off of its own (§6.2.7: "a local back-off timer that has
            // exponential value"); `queue_retry` paces it
            info!(
                "Reject cause #{} without back-off timer IE: no back-off armed",
                cause as u8
            );
            return;
        };
        let timer3 = GprsTimer3::from_byte(byte);
        if timer3.unit == GprsTimer3Unit::Deactivated {
            // Deactivated: no new request in this scope until the UE is
            // switched off or the network sends a PDU SESSION MODIFICATION /
            // RELEASE COMMAND in it (§6.4.1.4.2)
            warn!("{name} deactivated for {key:?}: establishment blocked");
            self.backoff.insert(
                key,
                BackoffEntry {
                    name,
                    stoppable,
                    timer: None,
                },
            );
            return;
        }
        let secs = timer3.to_seconds();
        if secs == 0 {
            // Zero: stop the timer if it is running; the UE may ask again
            // (§6.4.1.4.2). Before #379 a zero value was taken for a
            // deactivated one and barred the DNN for good.
            if self.backoff.remove(&key).is_some() {
                info!("{name} for {key:?} stopped: back-off timer value zero");
            } else {
                info!("{name} value zero for {key:?}: no back-off");
            }
            return;
        }
        // A new value replaces a running timer (§6.4.1.4.2: "stop timer
        // T3396 ... if it is running" and start it with the new value)
        let mut timer = UeTimer::new(code, false, secs);
        timer.start(true);
        info!("Back-off timer started for {key:?}: {secs}s ({name})");
        self.backoff.insert(
            key,
            BackoffEntry {
                name,
                stoppable,
                timer: Some(timer),
            },
        );
    }

    /// Stop every back-off timer that applies to the session `params`, running
    /// or deactivated: the network sent a PDU SESSION MODIFICATION COMMAND or
    /// RELEASE COMMAND for it (TS 24.501 §6.4.1.4.2, §6.3.2.3, §6.3.3.3). A
    /// command for an emergency PDU session stops none. A configured session
    /// the timer barred is then re-attempted (#379).
    fn stop_backoff_for(&mut self, params: &SmSessionParams, command: &str) {
        if params.emergency {
            return;
        }
        let (s_nssai, dnn) = (params.s_nssai.as_deref(), params.dnn.as_deref());
        self.backoff.retain(|key, entry| {
            let stop = entry.stoppable && key.covers(s_nssai, dnn);
            if stop {
                info!("{} for {key:?} stopped: {command} received", entry.name);
            }
            !stop
        });
    }

    /// PDU Session Modification Command (TS 24.501 6.3.2 network-initiated
    /// or 6.4.2.3 answering a UE-initiated modification)
    fn handle_modification_command(&mut self, body: &[u8], psi: u8, pti: u8) -> Vec<SmOutput> {
        if self.session(psi).is_none() {
            warn!("Modification Command for unknown PSI {psi}");
            // TS 24.501 6.3.2.5: respond with 5GSM Status cause #43
            return self.send_sm_status(psi, pti, SmCause::InvalidPduSessionIdentity);
        }
        let cmd = match PduSessionModificationCommand::decode(&mut &body[..], psi, pti) {
            Ok(cmd) => cmd,
            Err(e) => {
                warn!("Failed to decode Modification Command: {e:?}");
                return self.send_sm_status(psi, pti, SmCause::InvalidMandatoryInformation);
            }
        };

        // §6.4.1.4.2 / §6.3.2.3: the command stops a back-off timer in the
        // session's scope (#379)
        if let Some(p) = self.session(psi).map(|s| s.params.clone()) {
            self.stop_backoff_for(&p, "PDU SESSION MODIFICATION COMMAND");
        }

        // If this answers our UE-initiated modification, complete it
        let mut outs = Vec::new();
        let answered_ue_procedure = pti != 0
            && self
                .validate_response_pt(pti, psi, PtMessageType::PduSessionModificationRequest)
                .is_ok();
        if answered_ue_procedure {
            self.pt.abort(pti);
            self.pending.remove(&pti);
            outs.push(SmOutput::SessionModified { psi });
        }

        // Apply the authorized parameters
        if let Some(s) = self.sessions[psi as usize].as_mut() {
            if let Some(ref ambr) = cmd.session_ambr {
                if ambr.len() >= 6 {
                    s.session_ambr = Some(IeSessionAmbr {
                        downlink_unit: ambr[0],
                        downlink: u16::from_be_bytes([ambr[1], ambr[2]]),
                        uplink_unit: ambr[3],
                        uplink: u16::from_be_bytes([ambr[4], ambr[5]]),
                    });
                }
            }
            if let Some(ref rules) = cmd.authorized_qos_rules {
                s.authorized_qos_rules = rules.clone();
            }
            s.state = PsState::Active;
        }

        // Reply with PDU Session Modification Complete (TS 24.501 6.3.2.3)
        let complete = PduSessionModificationComplete::new(psi, pti);
        let mut sm_pdu = Vec::new();
        complete.encode(&mut sm_pdu);
        info!("Sending PDU Session Modification Complete: PSI={psi}, PTI={pti}");
        outs.push(SmOutput::SendNasPdu(build_ul_transport(
            sm_pdu, psi, None, None, None,
        )));
        outs
    }

    /// PDU Session Modification Reject (TS 24.501 6.4.2.4)
    fn handle_modification_reject(&mut self, body: &[u8], psi: u8, pti: u8) -> Vec<SmOutput> {
        if let Err(cause) =
            self.validate_response_pt(pti, psi, PtMessageType::PduSessionModificationRequest)
        {
            warn!("Modification Reject with invalid PTI {pti} / PSI {psi}");
            return self.send_sm_status(psi, pti, cause);
        }
        let rej = match PduSessionModificationReject::decode(&mut &body[..], psi, pti) {
            Ok(rej) => rej,
            Err(e) => {
                warn!("Failed to decode Modification Reject: {e:?}");
                return self.send_sm_status(psi, pti, SmCause::InvalidMandatoryInformation);
            }
        };
        let cause = SmCause::try_from(rej.cause() as u8).unwrap_or_default();
        warn!(
            "PDU Session Modification Reject: cause {:?} (#{})",
            cause, cause as u8
        );

        self.pt.abort(pti);
        self.pending.remove(&pti);

        let mut outs = Vec::new();
        match cause {
            // 6.4.2.4: #43 "invalid PDU session identity" -> release locally
            SmCause::InvalidPduSessionIdentity => {
                self.clear_session(psi);
                outs.push(SmOutput::SessionReleased { psi });
            }
            // #26 / #67 / #69 / #70 with a back-off timer IE
            SmCause::InsufficientResources
            | SmCause::InsufficientResourcesForSliceAndDnn
            | SmCause::InsufficientResourcesForSlice
            | SmCause::MissingOrUnknownDnnInSlice => {
                let params = self.session(psi).map(|s| s.params.clone());
                if let Some(p) = params {
                    self.arm_backoff(cause, &p, rej.back_off_timer_value);
                }
                if let Some(s) = self.session_mut(psi) {
                    s.state = PsState::Active;
                }
            }
            _ => {
                if let Some(s) = self.session_mut(psi) {
                    s.state = PsState::Active;
                }
            }
        }
        outs.push(SmOutput::SessionModificationFailed { psi, cause });
        outs
    }

    /// PDU Session Release Command (TS 24.501 6.3.3 / 6.4.3.3)
    fn handle_release_command(&mut self, body: &[u8], psi: u8, pti: u8) -> Vec<SmOutput> {
        if self.session(psi).is_none() {
            warn!("Release Command for unknown PSI {psi}");
            // TS 24.501 6.3.3.5: 5GSM Status with cause #43
            return self.send_sm_status(psi, pti, SmCause::InvalidPduSessionIdentity);
        }
        let cmd = match PduSessionReleaseCommand::decode(&mut &body[..], psi, pti) {
            Ok(cmd) => cmd,
            Err(e) => {
                // TS 24.501 §7.5.3 c): a mandatory-IE error in a PDU SESSION RELEASE
                // COMMAND is answered with PDU SESSION RELEASE COMPLETE carrying #96,
                // not 5GSM STATUS (nextgsim #383). The complete ends the release, so
                // the session is released here too.
                warn!(
                    "Failed to decode Release Command: {e:?}; Release Complete #96 \
                     (TS 24.501 §7.5.3 c)"
                );
                if pti != 0
                    && self
                        .validate_response_pt(pti, psi, PtMessageType::PduSessionReleaseRequest)
                        .is_ok()
                {
                    self.pt.abort(pti);
                    self.pending.remove(&pti);
                }
                let _ = self.take_session(psi);
                let mut complete = PduSessionReleaseComplete::new(psi, pti);
                complete.sm_cause = Some(ie3_cause(SmCause::InvalidMandatoryInformation));
                let mut sm_pdu = Vec::new();
                complete.encode(&mut sm_pdu);
                return vec![
                    SmOutput::SendNasPdu(build_ul_transport(sm_pdu, psi, None, None, None)),
                    SmOutput::SessionReleased { psi },
                ];
            }
        };
        let cause = SmCause::try_from(cmd.cause() as u8).unwrap_or_default();
        info!(
            "PDU Session Release Command: PSI={psi}, cause {:?} (#{})",
            cause, cause as u8
        );

        // If this answers our UE-initiated release, stop T3582
        if pti != 0
            && self
                .validate_response_pt(pti, psi, PtMessageType::PduSessionReleaseRequest)
                .is_ok()
        {
            self.pt.abort(pti);
            self.pending.remove(&pti);
        }

        let session = self.take_session(psi);
        if let Some(ref s) = session {
            // §6.4.1.4.2 / §6.3.3.3: a command without a Back-off timer value
            // IE, or with #39, stops a back-off timer in the session's scope
            // (#379) ...
            if cmd.back_off_timer_value.is_none() || cause == SmCause::ReactivationRequested {
                self.stop_backoff_for(&s.params, "PDU SESSION RELEASE COMMAND");
            }
            // ... and a #26 / #67 / #69 / #70 with a Back-off timer value
            // starts one (§6.3.3.3)
            if matches!(
                cause,
                SmCause::InsufficientResources
                    | SmCause::InsufficientResourcesForSliceAndDnn
                    | SmCause::InsufficientResourcesForSlice
                    | SmCause::MissingOrUnknownDnnInSlice
            ) {
                self.arm_backoff(cause, &s.params, cmd.back_off_timer_value);
            }
        }

        // Reply with PDU Session Release Complete (TS 24.501 6.3.3.3)
        let complete = PduSessionReleaseComplete::new(psi, pti);
        let mut sm_pdu = Vec::new();
        complete.encode(&mut sm_pdu);
        info!("Sending PDU Session Release Complete: PSI={psi}, PTI={pti}");

        let mut outs = vec![
            SmOutput::SendNasPdu(build_ul_transport(sm_pdu, psi, None, None, None)),
            SmOutput::SessionReleased { psi },
        ];

        match (cause, session) {
            // Cause #39 "reactivation requested": re-establish the session
            // (TS 24.501 6.3.3.3). A configured session whose request cannot
            // go out now is queued for a re-attempt (#379).
            (SmCause::ReactivationRequested, Some(s)) => {
                info!("Release Command requested reactivation: re-establishing the session");
                outs.extend(self.start_establishment_inner(s.params, s.wanted, 0));
            }
            // Released for lack of resources: a configured session is asked
            // for again once the back-off timer it started runs out, or after
            // the UE's own wait (#379)
            (
                SmCause::InsufficientResources
                | SmCause::InsufficientResourcesForSliceAndDnn
                | SmCause::InsufficientResourcesForSlice
                | SmCause::MissingOrUnknownDnnInSlice,
                Some(PduSession {
                    wanted: Some(base), ..
                }),
            ) => self.queue_retry(base, 1),
            _ => {}
        }
        outs
    }

    /// PDU Session Release Reject (TS 24.501 6.4.3.4)
    fn handle_release_reject(&mut self, body: &[u8], psi: u8, pti: u8) -> Vec<SmOutput> {
        if let Err(cause) =
            self.validate_response_pt(pti, psi, PtMessageType::PduSessionReleaseRequest)
        {
            warn!("Release Reject with invalid PTI {pti} / PSI {psi}");
            return self.send_sm_status(psi, pti, cause);
        }
        let rej = match PduSessionReleaseReject::decode(&mut &body[..], psi, pti) {
            Ok(rej) => rej,
            Err(e) => {
                warn!("Failed to decode Release Reject: {e:?}");
                return self.send_sm_status(psi, pti, SmCause::InvalidMandatoryInformation);
            }
        };
        let cause = SmCause::try_from(rej.cause() as u8).unwrap_or_default();
        warn!(
            "PDU Session Release Reject: cause {:?} (#{})",
            cause, cause as u8
        );

        self.pt.abort(pti);
        self.pending.remove(&pti);

        match cause {
            // 6.4.3.4: #43 / #54 -> release the PDU session locally
            SmCause::InvalidPduSessionIdentity | SmCause::PduSessionDoesNotExist => {
                self.clear_session(psi);
                vec![SmOutput::SessionReleased { psi }]
            }
            _ => {
                // Any other cause: the session stays active
                if let Some(s) = self.session_mut(psi) {
                    s.state = PsState::Active;
                }
                Vec::new()
            }
        }
    }

    /// Abort a procedure transaction locally (no message sent).
    fn abort_pt_local(&mut self, pti: u8) {
        if let Some((msg_type, psi)) = self.pt.abort(pti) {
            self.pending.remove(&pti);
            match msg_type {
                PtMessageType::PduSessionEstablishmentRequest => {
                    self.sessions[psi as usize] = None;
                }
                PtMessageType::PduSessionModificationRequest
                | PtMessageType::PduSessionReleaseRequest => {
                    if let Some(s) = self.sessions[psi as usize].as_mut() {
                        s.state = PsState::Active;
                    }
                }
            }
        }
    }

    /// Build and wrap a 5GSM Status message (TS 24.501 Section 8.3.18)
    fn send_sm_status(&self, psi: u8, pti: u8, cause: SmCause) -> Vec<SmOutput> {
        let mut sm_pdu = Vec::new();
        PlainSmHeader::new(psi, pti, NasSmMessageType::FiveGSmStatus).encode(&mut sm_pdu);
        sm_pdu.push(cause as u8);
        info!(
            "Sending 5GSM Status: PSI={psi}, PTI={pti}, cause #{}",
            cause as u8
        );
        vec![SmOutput::SendNasPdu(build_ul_transport(
            sm_pdu, psi, None, None, None,
        ))]
    }
}

/// Build a plain UL NAS Transport PDU carrying an N1 SM container
/// (TS 24.501 Section 8.2.10).
fn build_ul_transport(
    sm_pdu: Vec<u8>,
    psi: u8,
    request_type: Option<RequestType>,
    s_nssai: Option<Vec<u8>>,
    dnn: Option<Vec<u8>>,
) -> Vec<u8> {
    let mut transport = UlNasTransport::new(PayloadContainerType::N1SmInformation, sm_pdu);
    transport.pdu_session_id = Some(psi);
    transport.request_type = request_type;
    transport.s_nssai = s_nssai;
    transport.dnn = dnn;
    let mut out = Vec::new();
    transport.encode(&mut out);
    out
}

/// Encode a DNN string into the DNN IE value (length-prefixed labels,
/// TS 24.501 Section 9.11.2.1B / TS 23.003)
fn encode_dnn_value(dnn: &str) -> Vec<u8> {
    IeDnn::from_string(dnn).value
}

/// Convert the shared [`SmCause`] into the ie3 representation used by the
/// modification / release message codecs.
fn ie3_cause(cause: SmCause) -> nextgsim_nas::ies::ie3::Ie5gSmCause {
    use nextgsim_nas::ies::ie3;
    let value = ie3::SmCause::try_from(cause as u8).unwrap_or_default();
    ie3::Ie5gSmCause::new(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nextgsim_nas::enums::MmMessageType;
    use nextgsim_nas::messages::sm::{IePduAddress, IeQosRules};

    fn test_params() -> SmSessionParams {
        SmSessionParams {
            session_type: PduSessionTypeValue::Ipv4,
            ssc_mode: SscModeValue::SscMode1,
            dnn: Some("internet".to_string()),
            s_nssai: Some(vec![0x01]),
            emergency: false,
            requested_5qi: None,
            subscription_index: 0,
        }
    }

    fn new_orch() -> SmOrchestrator {
        SmOrchestrator::new()
    }

    /// Extract the single sent NAS PDU from the outputs
    fn first_sent_pdu(outs: &[SmOutput]) -> &[u8] {
        for out in outs {
            if let SmOutput::SendNasPdu(pdu) = out {
                return pdu;
            }
        }
        panic!("no SendNasPdu in outputs: {outs:?}");
    }

    /// Decode the UL NAS Transport and return (transport, inner SM PDU)
    fn unwrap_ul(pdu: &[u8]) -> (UlNasTransport, Vec<u8>) {
        assert_eq!(pdu[0], 0x7E, "must be a 5GMM PDU");
        assert_eq!(pdu[2], u8::from(MmMessageType::UlNasTransport));
        let t = UlNasTransport::decode(&mut &pdu[3..]).unwrap();
        let inner = t.payload_container.clone();
        (t, inner)
    }

    /// Start an establishment and return (orchestrator, psi, pti)
    fn started_establishment() -> (SmOrchestrator, u8, u8) {
        let mut orch = new_orch();
        let outs = orch.start_establishment(test_params());
        let (t, inner) = unwrap_ul(first_sent_pdu(&outs));
        let psi = t.pdu_session_id.unwrap();
        let pti = inner[2];
        (orch, psi, pti)
    }

    fn conformant_accept_bytes(psi: u8, pti: u8) -> Vec<u8> {
        let mut acc = PduSessionEstablishmentAccept::new(
            psi,
            pti,
            IeSelectedPduSessionType::new(PduSessionTypeValue::Ipv4),
            IeSelectedSscMode::new(SscModeValue::SscMode1),
            IeQosRules::new(vec![0x01, 0x00, 0x06, 0x31, 0x31, 0x01, 0x01, 0x01, 0x01]),
            IeSessionAmbr::new(0x06, 200, 0x06, 50),
        );
        acc.pdu_address = Some(IePduAddress::ipv4([10, 45, 0, 2]));
        let mut buf = Vec::new();
        acc.encode(&mut buf);
        buf
    }

    fn dl_with_container(container: Vec<u8>, psi: u8) -> DlNasTransport {
        let mut dl = DlNasTransport::new(PayloadContainerType::N1SmInformation, container);
        dl.pdu_session_id = Some(psi);
        dl
    }

    // ========================================================================
    // Establishment: initiation + UL NAS Transport conformance
    // ========================================================================

    #[test]
    fn test_establishment_request_wrapped_in_ul_nas_transport() {
        let mut orch = new_orch();
        let outs = orch.start_establishment(test_params());
        let pdu = first_sent_pdu(&outs);
        let (t, inner) = unwrap_ul(pdu);

        // TS 24.501 5.4.5.2.2: container type 1, PSI, request type,
        // S-NSSAI and DNN IEs
        assert_eq!(
            t.payload_container_type,
            PayloadContainerType::N1SmInformation
        );
        assert_eq!(t.pdu_session_id, Some(1), "first PSI allocated must be 1");
        assert_eq!(t.request_type, Some(RequestType::InitialRequest));
        assert_eq!(t.s_nssai, Some(vec![0x01]));
        assert_eq!(t.dnn, Some(encode_dnn_value("internet")));

        // Inner 5GSM message: establishment request with allocated PTI
        assert_eq!(inner[0], 0x2E);
        assert_eq!(inner[1], 1); // PSI
        assert_eq!(inner[2], 1); // first PTI allocated
        assert_eq!(inner[3], 0xC1);
        let req = PduSessionEstablishmentRequest::decode(&mut &inner[4..], 1, 1).unwrap();
        assert_eq!(
            req.pdu_session_type.unwrap().value,
            PduSessionTypeValue::Ipv4
        );
        assert_eq!(req.ssc_mode.unwrap().value, SscModeValue::SscMode1);

        // Session and timer state
        assert_eq!(orch.session(1).unwrap().state, PsState::ActivePending);
        assert!(orch.pt.get(1).unwrap().is_pending());
    }

    #[test]
    fn test_psi_and_pti_are_allocated_not_hardcoded() {
        let mut orch = new_orch();
        orch.start_establishment(test_params());
        let mut p2 = test_params();
        p2.dnn = Some("ims".to_string());
        let outs = orch.start_establishment(p2);
        let (t, inner) = unwrap_ul(first_sent_pdu(&outs));
        assert_eq!(t.pdu_session_id, Some(2), "second session gets PSI 2");
        assert_eq!(inner[2], 2, "second procedure gets PTI 2");
    }

    // ========================================================================
    // Establishment: Accept handling
    // ========================================================================

    #[test]
    fn test_establishment_accept_activates_session() {
        let (mut orch, psi, pti) = started_establishment();
        let dl = dl_with_container(conformant_accept_bytes(psi, pti), psi);
        let outs = orch.handle_dl_nas_transport(&dl);
        assert_eq!(
            outs,
            vec![SmOutput::SessionEstablished {
                psi,
                ipv4: Some([10, 45, 0, 2])
            }]
        );
        let s = orch.session(psi).unwrap();
        assert_eq!(s.state, PsState::Active);
        assert_eq!(s.selected_type, Some(PduSessionTypeValue::Ipv4));
        assert_eq!(s.selected_ssc_mode, Some(SscModeValue::SscMode1));
        assert!(s.session_ambr.is_some());
        assert!(!s.authorized_qos_rules.is_empty());
        // PTI released
        assert!(orch.pt.get(pti).unwrap().is_inactive());
    }

    /// TS 24.501 §7.5.3 a) (nextgsim #383): what the UE sends for an ESTABLISHMENT
    /// ACCEPT with a bad mandatory IE: a RELEASE REQUEST #96 on a new PTI, not 5GSM
    /// STATUS. The establishment PTI is released, and the session is INACTIVE PENDING
    /// under T3582.
    fn assert_release_request_96(orch: &SmOrchestrator, outs: &[SmOutput], psi: u8, pti: u8) {
        assert_eq!(outs.len(), 2, "{outs:?}");
        assert_eq!(
            outs[0],
            SmOutput::SessionEstablishmentFailed {
                psi,
                cause: SmCause::InvalidMandatoryInformation
            },
            "the attempt is reported failed"
        );
        let (t, inner) = unwrap_ul(first_sent_pdu(outs));
        assert_eq!(t.pdu_session_id, Some(psi));
        assert_eq!(
            inner[3], 0xD1,
            "PDU SESSION RELEASE REQUEST, not 5GSM STATUS"
        );
        let rel_pti = inner[2];
        let req = PduSessionReleaseRequest::decode(&mut &inner[4..], psi, rel_pti).unwrap();
        assert_eq!(
            req.sm_cause.unwrap().value as u8,
            SmCause::InvalidMandatoryInformation as u8
        );
        assert_eq!(orch.session(psi).unwrap().state, PsState::InactivePending);
        // The establishment's PTI was released (it may be reused for the release):
        // the one pending procedure is the release, under T3582.
        let pt = orch.pt.get(rel_pti).unwrap();
        assert!(pt.is_pending());
        assert_eq!(
            pt.message_type(),
            Some(PtMessageType::PduSessionReleaseRequest)
        );
        if rel_pti != pti {
            assert!(
                orch.pt.get(pti).unwrap().is_inactive(),
                "T3580's PTI released"
            );
        }
    }

    #[test]
    fn test_establishment_accept_missing_ambr_releases_the_session_with_96() {
        let (mut orch, psi, pti) = started_establishment();
        // Truncate the conformant accept right before the session AMBR:
        // header(4) + octet5(1) + qos LV-E(2+9) = 16
        let bytes = conformant_accept_bytes(psi, pti)[..16].to_vec();
        let outs = orch.handle_dl_nas_transport(&dl_with_container(bytes, psi));
        assert_release_request_96(&orch, &outs, psi, pti);

        // The network's Release Command completes it, as for any UE release.
        let rel_pti = unwrap_ul(first_sent_pdu(&outs)).1[2];
        let cmd = vec![0x2E, psi, rel_pti, 0xD3, 96];
        let outs = orch.handle_dl_nas_transport(&dl_with_container(cmd, psi));
        assert!(outs.contains(&SmOutput::SessionReleased { psi }));
        assert!(orch.session(psi).is_none());
    }

    #[test]
    fn test_establishment_accept_with_unknown_pti_triggers_status_81() {
        let (mut orch, psi, _pti) = started_establishment();
        let dl = dl_with_container(conformant_accept_bytes(psi, 99), psi);
        let outs = orch.handle_dl_nas_transport(&dl);
        let (_, inner) = unwrap_ul(first_sent_pdu(&outs));
        assert_eq!(inner[3], 0xD6);
        assert_eq!(inner[4], SmCause::InvalidPtiValue as u8);
        assert_eq!(orch.session(psi).unwrap().state, PsState::ActivePending);
    }

    #[test]
    fn test_establishment_accept_with_psi_mismatch_triggers_status_47() {
        let (mut orch, _psi, pti) = started_establishment();
        // Same PTI but a different PSI in the container header
        let dl = dl_with_container(conformant_accept_bytes(7, pti), 7);
        let outs = orch.handle_dl_nas_transport(&dl);
        let (_, inner) = unwrap_ul(first_sent_pdu(&outs));
        assert_eq!(inner[3], 0xD6);
        assert_eq!(inner[4], SmCause::PtiMismatch as u8);
    }

    #[test]
    fn test_legacy_core_accept_now_rejected() {
        // Regression guard: the old non-conformant nextgcore emission (SSC
        // mode as a SEPARATE octet, QoS rules as a 6-octet blob behind a
        // 1-octet marker) must NOT parse under the strict-only parser. The UE
        // responds with 5GSM Status #96 and the session stays pending.
        fn legacy_accept(psi: u8, pti: u8) -> Vec<u8> {
            let mut msg = vec![0x2E, psi, pti, 0xC2, 0x01, 0x01];
            msg.extend_from_slice(&[0x06, 0x01, 0x03, 0x01, 0x01, 0x09]);
            msg.extend_from_slice(&[0x06, 0x06, 0x00, 0xC8, 0x06, 0x00, 0x32]);
            msg.extend_from_slice(&[0x29, 0x05, 0x01, 10, 45, 0, 2]);
            msg
        }

        let (mut orch, psi, pti) = started_establishment();
        let outs = orch.handle_dl_nas_transport(&dl_with_container(legacy_accept(psi, pti), psi));
        assert_release_request_96(&orch, &outs, psi, pti);
    }

    // ========================================================================
    // Establishment: Reject handling per cause
    // ========================================================================

    fn reject_bytes(psi: u8, pti: u8, cause: SmCause, backoff: Option<u8>) -> Vec<u8> {
        let mut rej = PduSessionEstablishmentReject::new(psi, pti, cause);
        rej.back_off_timer_value = backoff;
        let mut buf = Vec::new();
        rej.encode(&mut buf);
        buf
    }

    #[test]
    fn test_reject_69_with_backoff_arms_timer_and_blocks_retry() {
        let (mut orch, psi, pti) = started_establishment();
        // Back-off timer: GPRS timer 3 unit 101 (1 min), value 2 -> 120s
        let bytes = reject_bytes(psi, pti, SmCause::InsufficientResourcesForSlice, Some(0xA2));
        let outs = orch.handle_dl_nas_transport(&dl_with_container(bytes, psi));
        assert!(outs.contains(&SmOutput::SessionEstablishmentFailed {
            psi,
            cause: SmCause::InsufficientResourcesForSlice
        }));
        // #69: back-off scope is the S-NSSAI (any DNN)
        assert!(orch.is_backed_off(Some(&[0x01]), Some("internet")));
        assert!(orch.is_backed_off(Some(&[0x01]), Some("other-dnn")));
        assert_eq!(orch.backoff_remaining(Some(&[0x01]), None), Some(120));
        // Session and PSI released
        assert!(orch.session(psi).is_none());
        // New attempt for the backed-off slice is refused
        assert!(orch.start_establishment(test_params()).is_empty());
    }

    #[test]
    fn test_reject_26_backoff_is_per_dnn() {
        let (mut orch, psi, pti) = started_establishment();
        let bytes = reject_bytes(psi, pti, SmCause::InsufficientResources, Some(0xA1));
        orch.handle_dl_nas_transport(&dl_with_container(bytes, psi));
        // #26: scope is the DNN
        assert!(orch.is_backed_off(None, Some("internet")));
        assert!(orch.is_backed_off(Some(&[0x02]), Some("internet")));
        // A different DNN is not blocked
        let mut other = test_params();
        other.dnn = Some("ims".to_string());
        assert!(!orch.start_establishment(other).is_empty());
    }

    #[test]
    fn test_reject_26_without_backoff_timer_does_not_arm() {
        let (mut orch, psi, pti) = started_establishment();
        let bytes = reject_bytes(psi, pti, SmCause::InsufficientResources, None);
        orch.handle_dl_nas_transport(&dl_with_container(bytes, psi));
        assert!(!orch.is_backed_off(Some(&[0x01]), Some("internet")));
        // Retry is allowed immediately
        assert!(!orch.start_establishment(test_params()).is_empty());
    }

    #[test]
    fn test_reject_with_deactivated_backoff_blocks_permanently() {
        let (mut orch, psi, pti) = started_establishment();
        // GPRS timer 3 unit 111 = deactivated
        let bytes = reject_bytes(psi, pti, SmCause::MissingOrUnknownDnnInSlice, Some(0xE0));
        orch.handle_dl_nas_transport(&dl_with_container(bytes, psi));
        assert!(orch.is_backed_off(Some(&[0x01]), Some("internet")));
        // Ticking does not clear a deactivated back-off
        orch.tick();
        assert!(orch.is_backed_off(Some(&[0x01]), Some("internet")));
    }

    #[test]
    fn test_reject_50_retries_with_ipv4() {
        let mut orch = new_orch();
        let mut params = test_params();
        params.session_type = PduSessionTypeValue::Ipv4v6;
        let outs = orch.start_establishment(params);
        let (t, inner) = unwrap_ul(first_sent_pdu(&outs));
        let (psi, pti) = (t.pdu_session_id.unwrap(), inner[2]);

        let bytes = reject_bytes(psi, pti, SmCause::PduSessionTypeIpv4OnlyAllowed, None);
        let outs = orch.handle_dl_nas_transport(&dl_with_container(bytes, psi));
        // A new establishment request with PDU type IPv4 must be sent
        let (_, inner) = unwrap_ul(first_sent_pdu(&outs));
        let req =
            PduSessionEstablishmentRequest::decode(&mut &inner[4..], inner[1], inner[2]).unwrap();
        assert_eq!(
            req.pdu_session_type.unwrap().value,
            PduSessionTypeValue::Ipv4
        );
        // And the restriction is remembered for the DNN
        assert_eq!(
            orch.type_override.get(&Some("internet".to_string())),
            Some(&PduSessionTypeValue::Ipv4)
        );
    }

    #[test]
    fn test_reject_68_retries_with_allowed_ssc_mode() {
        let (mut orch, psi, pti) = started_establishment();
        let mut rej = PduSessionEstablishmentReject::new(psi, pti, SmCause::NotSupportedSscMode);
        rej.allowed_ssc_mode = Some(0x02);
        let mut bytes = Vec::new();
        rej.encode(&mut bytes);
        let outs = orch.handle_dl_nas_transport(&dl_with_container(bytes, psi));
        let (_, inner) = unwrap_ul(first_sent_pdu(&outs));
        let req =
            PduSessionEstablishmentRequest::decode(&mut &inner[4..], inner[1], inner[2]).unwrap();
        assert_eq!(req.ssc_mode.unwrap().value, SscModeValue::SscMode2);
    }

    #[test]
    fn test_reject_27_fails_without_retry() {
        let (mut orch, psi, pti) = started_establishment();
        let bytes = reject_bytes(psi, pti, SmCause::MissingOrUnknownDnn, None);
        let outs = orch.handle_dl_nas_transport(&dl_with_container(bytes, psi));
        assert_eq!(
            outs,
            vec![SmOutput::SessionEstablishmentFailed {
                psi,
                cause: SmCause::MissingOrUnknownDnn
            }]
        );
        assert!(orch.session(psi).is_none());
        assert!(orch.pt.get(pti).unwrap().is_inactive());
    }

    #[test]
    fn test_reject_28_unknown_pdu_type_fails() {
        let (mut orch, psi, pti) = started_establishment();
        let bytes = reject_bytes(psi, pti, SmCause::UnknownPduSessionType, None);
        let outs = orch.handle_dl_nas_transport(&dl_with_container(bytes, psi));
        assert_eq!(
            outs,
            vec![SmOutput::SessionEstablishmentFailed {
                psi,
                cause: SmCause::UnknownPduSessionType
            }]
        );
    }

    /// External vector: the exact Reject bytes nextgcore smfd emits
    /// (`policy::build_establishment_reject` -> `[2E PSI PTI C3 cause]`).
    #[test]
    fn test_reject_decodes_nextgcore_emission() {
        let (mut orch, psi, pti) = started_establishment();
        let bytes = vec![0x2E, psi, pti, 0xC3, 29];
        let outs = orch.handle_dl_nas_transport(&dl_with_container(bytes, psi));
        assert_eq!(
            outs,
            vec![SmOutput::SessionEstablishmentFailed {
                psi,
                cause: SmCause::UserAuthenticationFailed
            }]
        );
    }

    // ========================================================================
    // T3580 retransmission and exhaustion
    // ========================================================================

    /// #409 follow-up, TS 24.501 §6.4.1.6 a) / §5.6.1.1: a T3580 expiry with
    /// no NAS signalling connection holds the retransmission; it goes, the
    /// identical octets, on the first tick with one. The fifth expiry aborts
    /// whatever the connection, and an aborted procedure's held retransmission
    /// goes with it.
    #[test]
    fn a_retransmission_without_a_connection_waits_for_one() {
        let (mut orch, psi, pti) = started_establishment();
        let original = orch.pending.get(&pti).unwrap().ul_pdu.clone();
        let timer = |o: &SmOrchestrator| {
            let t = o.pt.get(pti).unwrap().timer().unwrap();
            (t.is_running(), t.expiry_count())
        };
        orch.expire_procedure_timers_for_test();
        assert!(orch.tick_with_connection(false).is_empty(), "held");
        assert!(orch.has_held_retransmission());
        assert_eq!(
            timer(&orch),
            (false, 1),
            "T3580 waits with the retransmission"
        );
        assert!(
            orch.tick_with_connection(false).is_empty(),
            "no expiry while held"
        );
        assert_eq!(
            orch.tick_with_connection(true),
            vec![SmOutput::SendNasPdu(original.clone())],
            "sent once the connection is up"
        );
        assert_eq!(timer(&orch), (true, 1), "restarted with the retransmission");
        assert!(!orch.has_held_retransmission());
        assert!(orch.tick_with_connection(true).is_empty(), "once");

        // Four real retransmissions in all, then the fifth expiry aborts
        for n in 2..MAX_T3580_RETRIES {
            orch.expire_procedure_timers_for_test();
            assert_eq!(
                orch.tick_with_connection(true),
                vec![SmOutput::SendNasPdu(original.clone())],
                "retransmission {n}"
            );
        }
        orch.expire_procedure_timers_for_test();
        assert_eq!(
            orch.tick_with_connection(false),
            vec![SmOutput::SessionEstablishmentFailed {
                psi,
                cause: SmCause::ProtocolErrorUnspecified
            }]
        );
        assert!(!orch.has_held_retransmission());
        assert!(orch.tick_with_connection(true).is_empty(), "nothing held");
    }

    /// #424 review, TS 24.501 §6.4.1.2: an emergency PDU session's
    /// retransmission goes on any NAS signalling connection, even one not
    /// "ready" for other 5GSM (the SERVICE REQUEST unanswered); held only with
    /// no connection at all. Another session's waits for readiness.
    #[test]
    fn an_emergency_retransmission_is_held_only_without_a_connection() {
        let mut orch = new_orch();
        let outs = orch.start_establishment(SmSessionParams {
            emergency: true,
            dnn: None,
            ..test_params()
        });
        let (_, inner) = unwrap_ul(first_sent_pdu(&outs));
        let pti = inner[2];
        let original = orch.pending.get(&pti).unwrap().ul_pdu.clone();
        orch.expire_procedure_timers_for_test();
        assert_eq!(
            orch.tick_with_transport(true, false),
            vec![SmOutput::SendNasPdu(original.clone())],
            "connected, not ready: an emergency one goes"
        );
        orch.expire_procedure_timers_for_test();
        assert!(
            orch.tick_with_transport(false, false).is_empty(),
            "no connection: held"
        );
        assert!(orch.has_held_emergency_retransmission());
        assert_eq!(
            orch.take_held_emergency(),
            vec![SmOutput::SendNasPdu(original)]
        );

        let (mut orch, _, _) = started_establishment();
        orch.expire_procedure_timers_for_test();
        assert!(
            orch.tick_with_transport(true, false).is_empty(),
            "not emergency: held"
        );
        assert!(!orch.has_held_emergency_retransmission());
    }

    /// #424 review: a hold is bounded. Past the span its remaining expiries
    /// would have taken, the held procedure is aborted as on its last expiry,
    /// though no connection came.
    #[test]
    fn a_held_retransmission_is_aborted_past_its_timers_span() {
        let (mut orch, psi, pti) = started_establishment();
        orch.expire_procedure_timers_for_test();
        assert!(orch.tick_with_connection(false).is_empty(), "held");
        let deadline = *orch.held_retransmissions.get(&pti).unwrap();
        let span = deadline - std::time::Instant::now();
        assert!(
            span > std::time::Duration::from_secs(4 * 16 - 1),
            "four more T3580 intervals and the one held: {span:?}"
        );
        assert!(orch.tick_with_connection(false).is_empty(), "within it");
        *orch.held_retransmissions.get_mut(&pti).unwrap() = std::time::Instant::now();
        assert_eq!(
            orch.tick_with_connection(false),
            vec![SmOutput::SessionEstablishmentFailed {
                psi,
                cause: SmCause::ProtocolErrorUnspecified
            }]
        );
        assert!(!orch.has_held_retransmission());
        assert!(
            orch.pt.get(pti).is_none_or(|t| !t.is_pending()),
            "the PTI is released"
        );
    }

    #[test]
    fn test_t3580_expiry_retransmits_then_aborts() {
        let (mut orch, psi, pti) = started_establishment();
        let original = orch.pending.get(&pti).unwrap().ul_pdu.clone();

        // Expiries 1-4: retransmission of the identical UL PDU
        for n in 1..MAX_T3580_RETRIES {
            let outs = {
                force_expire(&mut orch, pti);
                orch.handle_procedure_timer_expiry(pti)
            };
            assert_eq!(
                outs,
                vec![SmOutput::SendNasPdu(original.clone())],
                "expiry {n} must retransmit"
            );
            assert!(orch.pt.get(pti).unwrap().timer().unwrap().is_running());
        }

        // Fifth expiry: abort, release PSI and PTI
        let outs = {
            force_expire(&mut orch, pti);
            orch.handle_procedure_timer_expiry(pti)
        };
        assert_eq!(
            outs,
            vec![SmOutput::SessionEstablishmentFailed {
                psi,
                cause: SmCause::ProtocolErrorUnspecified
            }]
        );
        assert!(orch.session(psi).is_none());
        assert!(orch.pt.get(pti).unwrap().is_inactive());
        assert!(!orch.pending.contains_key(&pti));
    }

    /// Force the procedure timer for `pti` into the just-expired state
    /// (stopped, expiry count incremented) exactly like a real
    /// `UeTimer::perform_tick` expiry, and return the PTI.
    fn force_expire(orch: &mut SmOrchestrator, pti: u8) -> u8 {
        let timer = orch.pt.get_mut(pti).unwrap().timer_mut().unwrap();
        let prior_expiries = timer.expiry_count();
        // Replace with a zero-interval timer and replay the prior expiries
        // plus the new one through the real perform_tick path
        let mut replacement = UeTimer::new(timer.code(), false, 0);
        replacement.start(true);
        for _ in 0..=prior_expiries {
            assert!(replacement.perform_tick());
            replacement.start(false);
        }
        replacement.stop(false);
        *timer = replacement;
        pti
    }

    #[test]
    fn test_t3582_exhaustion_releases_locally() {
        let (mut orch, psi, pti) = started_establishment();
        let dl = dl_with_container(conformant_accept_bytes(psi, pti), psi);
        orch.handle_dl_nas_transport(&dl);

        let outs = orch.start_release(psi);
        let (_, inner) = unwrap_ul(first_sent_pdu(&outs));
        let rel_pti = inner[2];

        for _ in 1..MAX_T3582_RETRIES {
            let outs = {
                force_expire(&mut orch, rel_pti);
                orch.handle_procedure_timer_expiry(rel_pti)
            };
            assert!(matches!(outs[0], SmOutput::SendNasPdu(_)));
        }
        let outs = {
            force_expire(&mut orch, rel_pti);
            orch.handle_procedure_timer_expiry(rel_pti)
        };
        // TS 24.501 6.4.3.6 c): release locally
        assert_eq!(outs, vec![SmOutput::SessionReleased { psi }]);
        assert!(orch.session(psi).is_none());
    }

    #[test]
    fn test_t3581_exhaustion_keeps_session_active() {
        let (mut orch, psi, pti) = started_establishment();
        orch.handle_dl_nas_transport(&dl_with_container(conformant_accept_bytes(psi, pti), psi));

        let outs = orch.start_modification(psi);
        let (_, inner) = unwrap_ul(first_sent_pdu(&outs));
        let mod_pti = inner[2];
        assert_eq!(
            orch.session(psi).unwrap().state,
            PsState::ModificationPending
        );

        for _ in 1..MAX_T3581_RETRIES {
            {
                force_expire(&mut orch, mod_pti);
                orch.handle_procedure_timer_expiry(mod_pti)
            };
        }
        let outs = {
            force_expire(&mut orch, mod_pti);
            orch.handle_procedure_timer_expiry(mod_pti)
        };
        assert_eq!(
            outs,
            vec![SmOutput::SessionModificationFailed {
                psi,
                cause: SmCause::ProtocolErrorUnspecified
            }]
        );
        assert_eq!(orch.session(psi).unwrap().state, PsState::Active);
    }

    // ========================================================================
    // UE-initiated modification
    // ========================================================================

    /// Establish an active session and return (orch, psi)
    fn established() -> (SmOrchestrator, u8) {
        let (mut orch, psi, pti) = started_establishment();
        orch.handle_dl_nas_transport(&dl_with_container(conformant_accept_bytes(psi, pti), psi));
        (orch, psi)
    }

    #[test]
    fn test_ue_modification_completed_by_command() {
        let (mut orch, psi) = established();
        let outs = orch.start_modification(psi);
        let (t, inner) = unwrap_ul(first_sent_pdu(&outs));
        assert_eq!(t.pdu_session_id, Some(psi));
        assert!(t.request_type.is_none(), "no request type for modification");
        assert_eq!(inner[3], 0xC9);
        let mod_pti = inner[2];

        // Network answers with Modification Command (PTI echoed) carrying
        // a new session AMBR — exactly what nextgcore emits:
        // [2E PSI PTI CB 2A 06 <ambr>]
        let cmd = vec![
            0x2E, psi, mod_pti, 0xCB, 0x2A, 0x06, 0x06, 0x01, 0x2C, 0x06, 0x00, 0x64,
        ];
        let outs = orch.handle_dl_nas_transport(&dl_with_container(cmd, psi));

        assert!(outs.contains(&SmOutput::SessionModified { psi }));
        // The UE must answer with Modification Complete
        let (_, inner) = unwrap_ul(first_sent_pdu(&outs));
        assert_eq!(inner[3], 0xCC);
        assert_eq!(inner[2], mod_pti, "Complete echoes the command PTI");

        let s = orch.session(psi).unwrap();
        assert_eq!(s.state, PsState::Active);
        let ambr = s.session_ambr.as_ref().unwrap();
        assert_eq!(ambr.downlink, 300);
        assert_eq!(ambr.uplink, 100);
        assert!(orch.pt.get(mod_pti).unwrap().is_inactive());
    }

    #[test]
    fn test_ue_modification_rejected() {
        let (mut orch, psi) = established();
        let outs = orch.start_modification(psi);
        let (_, inner) = unwrap_ul(first_sent_pdu(&outs));
        let mod_pti = inner[2];

        // Modification Reject, cause #26 with back-off (unit 1min, value 1)
        let rej = vec![0x2E, psi, mod_pti, 0xCA, 26, 0x37, 0x01, 0xA1];
        let outs = orch.handle_dl_nas_transport(&dl_with_container(rej, psi));
        assert!(outs.contains(&SmOutput::SessionModificationFailed {
            psi,
            cause: SmCause::InsufficientResources
        }));
        assert_eq!(orch.session(psi).unwrap().state, PsState::Active);
        assert!(orch.is_backed_off(None, Some("internet")));
    }

    #[test]
    fn test_network_initiated_modification_command() {
        let (mut orch, psi) = established();
        // Network-initiated: PTI 0
        let cmd = vec![0x2E, psi, 0x00, 0xCB];
        let outs = orch.handle_dl_nas_transport(&dl_with_container(cmd, psi));
        let (_, inner) = unwrap_ul(first_sent_pdu(&outs));
        assert_eq!(inner[3], 0xCC, "must reply with Modification Complete");
        assert_eq!(inner[2], 0x00, "PTI 0 echoed");
        assert!(!outs.contains(&SmOutput::SessionModified { psi }));
    }

    // ========================================================================
    // UE-initiated release + network release command
    // ========================================================================

    #[test]
    fn test_ue_release_completed_by_command() {
        let (mut orch, psi) = established();
        let outs = orch.start_release(psi);
        let (t, inner) = unwrap_ul(first_sent_pdu(&outs));
        assert_eq!(t.pdu_session_id, Some(psi));
        assert_eq!(inner[3], 0xD1);
        let rel_pti = inner[2];
        // Release request must carry cause #36 regular deactivation
        let req = PduSessionReleaseRequest::decode(&mut &inner[4..], psi, rel_pti).unwrap();
        assert_eq!(req.sm_cause.unwrap().value as u8, 36);
        assert_eq!(orch.session(psi).unwrap().state, PsState::InactivePending);

        // Network answers with Release Command (PTI echoed, cause #36) —
        // exactly what nextgcore emits: [2E PSI PTI D3 cause]
        let cmd = vec![0x2E, psi, rel_pti, 0xD3, 36];
        let outs = orch.handle_dl_nas_transport(&dl_with_container(cmd, psi));
        assert!(outs.contains(&SmOutput::SessionReleased { psi }));
        let (_, inner) = unwrap_ul(first_sent_pdu(&outs));
        assert_eq!(inner[3], 0xD4, "must reply with Release Complete");
        assert_eq!(inner[2], rel_pti);
        assert!(orch.session(psi).is_none());
        assert!(orch.pt.get(rel_pti).unwrap().is_inactive());
    }

    #[test]
    fn test_ue_release_rejected_cause_54_releases_locally() {
        let (mut orch, psi) = established();
        let outs = orch.start_release(psi);
        let (_, inner) = unwrap_ul(first_sent_pdu(&outs));
        let rel_pti = inner[2];

        let rej = vec![0x2E, psi, rel_pti, 0xD2, 54];
        let outs = orch.handle_dl_nas_transport(&dl_with_container(rej, psi));
        assert_eq!(outs, vec![SmOutput::SessionReleased { psi }]);
        assert!(orch.session(psi).is_none());
    }

    #[test]
    fn test_ue_release_rejected_other_cause_keeps_session() {
        let (mut orch, psi) = established();
        let outs = orch.start_release(psi);
        let (_, inner) = unwrap_ul(first_sent_pdu(&outs));
        let rel_pti = inner[2];

        let rej = vec![0x2E, psi, rel_pti, 0xD2, 111];
        let outs = orch.handle_dl_nas_transport(&dl_with_container(rej, psi));
        assert!(outs.is_empty());
        assert_eq!(orch.session(psi).unwrap().state, PsState::Active);
    }

    #[test]
    fn test_network_release_command_with_reactivation() {
        let (mut orch, psi) = established();
        // Network-initiated release with cause #39 reactivation requested
        let cmd = vec![0x2E, psi, 0x00, 0xD3, 39];
        let outs = orch.handle_dl_nas_transport(&dl_with_container(cmd, psi));
        assert!(outs.contains(&SmOutput::SessionReleased { psi }));
        // Release Complete + a fresh Establishment Request must be sent
        let pdus: Vec<_> = outs
            .iter()
            .filter_map(|o| match o {
                SmOutput::SendNasPdu(p) => Some(p),
                _ => None,
            })
            .collect();
        assert_eq!(pdus.len(), 2);
        let (_, complete) = unwrap_ul(pdus[0]);
        assert_eq!(complete[3], 0xD4);
        let (t, request) = unwrap_ul(pdus[1]);
        assert_eq!(request[3], 0xC1);
        assert_eq!(t.request_type, Some(RequestType::InitialRequest));
    }

    /// TS 24.501 §7.5.3 c) (nextgsim #383): a RELEASE COMMAND whose mandatory 5GSM
    /// cause is missing gets a RELEASE COMPLETE carrying #96, not 5GSM STATUS, and
    /// the session is released, both network-initiated and as the answer to the UE's
    /// own release.
    #[test]
    fn a_release_command_with_a_bad_mandatory_ie_is_completed_with_96() {
        for ue_initiated in [false, true] {
            let (mut orch, psi) = established();
            let pti = if ue_initiated {
                unwrap_ul(first_sent_pdu(&orch.start_release(psi))).1[2]
            } else {
                0
            };
            let cmd = vec![0x2E, psi, pti, 0xD3]; // 5GSM cause (V) missing
            let outs = orch.handle_dl_nas_transport(&dl_with_container(cmd, psi));
            assert_eq!(outs.len(), 2, "{outs:?}");
            assert_eq!(outs[1], SmOutput::SessionReleased { psi });
            let (_, inner) = unwrap_ul(first_sent_pdu(&outs));
            assert_eq!(
                &inner[..4],
                &[0x2E, psi, pti, 0xD4],
                "PDU SESSION RELEASE COMPLETE on the command's PTI, not 5GSM STATUS"
            );
            // 5GSM cause IE (TV, IEI 0x59) = #96.
            assert_eq!(&inner[4..], &[0x59, 96]);
            assert!(orch.session(psi).is_none());
            if ue_initiated {
                assert!(orch.pt.get(pti).unwrap().is_inactive(), "T3582 stopped");
            }
        }
    }

    #[test]
    fn test_release_command_for_unknown_psi_sends_status_43() {
        let mut orch = new_orch();
        let cmd = vec![0x2E, 9, 0x00, 0xD3, 36];
        let outs = orch.handle_dl_nas_transport(&dl_with_container(cmd, 9));
        let (_, inner) = unwrap_ul(first_sent_pdu(&outs));
        assert_eq!(inner[3], 0xD6);
        assert_eq!(inner[4], SmCause::InvalidPduSessionIdentity as u8);
    }

    // ========================================================================
    // DL NAS Transport edge cases
    // ========================================================================

    #[test]
    fn test_payload_not_forwarded_aborts_establishment() {
        let (mut orch, psi, pti) = started_establishment();
        // The network echoes our own SM PDU back with 5GMM cause #90
        let (_, our_sm_pdu) = unwrap_ul(&orch.pending.get(&pti).unwrap().ul_pdu.clone());
        let mut dl = DlNasTransport::new(PayloadContainerType::N1SmInformation, our_sm_pdu);
        dl.pdu_session_id = Some(psi);
        dl.mm_cause = Some(90);
        let outs = orch.handle_dl_nas_transport(&dl);
        assert_eq!(
            outs,
            vec![SmOutput::SessionEstablishmentFailed {
                psi,
                cause: SmCause::NetworkFailure
            }]
        );
        assert!(orch.session(psi).is_none());
    }

    /// Build a DL NAS Transport that echoes an N1 SM container with the given
    /// wire PSI/PTI back to the UE carrying a 5GMM cause ("payload not
    /// forwarded", TS 24.501 5.4.5.3.1).
    fn payload_not_forwarded_dl(psi: u8, pti: u8) -> DlNasTransport {
        // [EPD=0x2E, PSI, PTI, msg_type] — only the first four octets are
        // inspected by handle_payload_not_forwarded; the message type byte is
        // a placeholder (establishment request 0xC1).
        let container = vec![0x2E, psi, pti, 0xC1];
        let mut dl = DlNasTransport::new(PayloadContainerType::N1SmInformation, container);
        dl.pdu_session_id = Some(psi);
        dl.mm_cause = Some(90);
        dl
    }

    #[test]
    fn test_payload_not_forwarded_out_of_range_psi_does_not_panic() {
        // A pending establishment exists (PTI matches), but the echoed
        // container carries a forged PSI = 200 (> PSI_MAX). The wire PSI must
        // not index self.sessions[200] (16-slot array) — that would panic the
        // UE (remote DoS). The malformed echo is dropped gracefully.
        let (mut orch, _psi, pti) = started_establishment();
        let outs = orch.handle_dl_nas_transport(&payload_not_forwarded_dl(200, pti));
        assert!(
            outs.is_empty(),
            "out-of-range PSI echo must be dropped, got {outs:?}"
        );
        // The real session 1 is untouched, the transaction still pending.
        assert_eq!(orch.session(1).unwrap().state, PsState::ActivePending);
        assert!(orch.pt.get(pti).unwrap().is_pending());
    }

    #[test]
    fn test_payload_not_forwarded_psi_255_does_not_panic() {
        // Boundary: PSI = 255 (max u8) must not panic.
        let (mut orch, _psi, pti) = started_establishment();
        let outs = orch.handle_dl_nas_transport(&payload_not_forwarded_dl(255, pti));
        assert!(outs.is_empty());
        assert_eq!(orch.session(1).unwrap().state, PsState::ActivePending);
    }

    #[test]
    fn test_payload_not_forwarded_zero_psi_does_not_panic() {
        // PSI = 0 is reserved/invalid (TS 24.501 Section 9.4); it must be
        // rejected before any session access, no panic.
        let (mut orch, _psi, pti) = started_establishment();
        let outs = orch.handle_dl_nas_transport(&payload_not_forwarded_dl(0, pti));
        assert!(outs.is_empty());
        assert_eq!(orch.session(1).unwrap().state, PsState::ActivePending);
    }

    #[test]
    fn test_payload_not_forwarded_psi_16_boundary_does_not_panic() {
        // PSI = 16 is exactly one past the 1..=15 valid range and exactly the
        // array length: the off-by-one that would panic sessions[16].
        let (mut orch, _psi, pti) = started_establishment();
        let outs = orch.handle_dl_nas_transport(&payload_not_forwarded_dl(16, pti));
        assert!(outs.is_empty());
        assert_eq!(orch.session(1).unwrap().state, PsState::ActivePending);
    }

    #[test]
    fn test_payload_not_forwarded_valid_psi_still_aborts() {
        // Regression: the in-range, matching-PSI path is unchanged — the
        // pending establishment is still aborted with NetworkFailure.
        let (mut orch, psi, pti) = started_establishment();
        let outs = orch.handle_dl_nas_transport(&payload_not_forwarded_dl(psi, pti));
        assert_eq!(
            outs,
            vec![SmOutput::SessionEstablishmentFailed {
                psi,
                cause: SmCause::NetworkFailure
            }]
        );
        assert!(orch.session(psi).is_none());
    }

    #[test]
    fn test_unknown_sm_message_type_sends_status_97() {
        let mut orch = new_orch();
        let container = vec![0x2E, 1, 1, 0xC4]; // 0xC4 is not assigned
        let outs = orch.handle_dl_nas_transport(&dl_with_container(container, 1));
        let (_, inner) = unwrap_ul(first_sent_pdu(&outs));
        assert_eq!(inner[3], 0xD6);
        assert_eq!(inner[4], SmCause::MessageTypeNonExistent as u8);
    }

    // ========================================================================
    // URSP-steered establishment (TS 24.526 §5.2, TS 23.503 §6.6.2.3) — #47
    // ========================================================================

    /// A UE config with one URSP rule steering everything to `dnn`/`sst`.
    fn ursp_config(enabled: bool, descriptor: &str, dnn: &str, sst: u8) -> UeConfig {
        use nextgsim_common::config::{RouteDescriptor, UrspRule as ConfigUrspRule};
        use nextgsim_common::types::SNssai;
        let mut config = UeConfig::default();
        config.ursp_evaluation = enabled;
        config.ursp_rules = vec![ConfigUrspRule {
            precedence: 10,
            traffic_descriptor: descriptor.to_string(),
            route_descriptors: vec![RouteDescriptor {
                s_nssai: Some(SNssai::new(sst)),
                dnn: Some(dnn.to_string()),
                session_type: None,
                ssc_mode: None,
            }],
        }];
        config
    }

    /// CRITERION 4 (issue #97): a TUN uplink packet whose 5-tuple matches an
    /// app-keyed URSP rule establishes a session carrying that rule's route.
    ///
    /// This is the test the issue said "cannot be written today", and the reason it
    /// could not was that nothing produced a populated `ApplicationDescriptor`.
    /// Asserted on the UL NAS TRANSPORT that goes out, not on the internal params:
    /// the DNN and S-NSSAI ride in the wrapper (TS 24.501 §5.4.5.2.2), so reading
    /// them off the encoded PDU is what proves the network is asked for the steered
    /// route.
    #[test]
    fn a_classified_uplink_flow_establishes_a_session_carrying_the_rules_route() {
        use super::super::ursp::ApplicationDescriptor;

        // An operator rule for HTTPS traffic, steered to the "video" DNN on SST 4.
        let mut orch = new_orch();
        orch.set_ursp_policy(super::super::ursp::UrspPolicy::from_config(&ursp_config(
            true, "port:443", "video", 4,
        )));

        // A real uplink IPv4 + TCP packet to port 443, classified exactly as the
        // live TUN path classifies it.
        let mut packet = vec![0u8; 40];
        packet[0] = 0x45;
        packet[9] = 6;
        packet[12..16].copy_from_slice(&[10, 45, 0, 2]);
        packet[16..20].copy_from_slice(&[93, 184, 216, 34]);
        packet[20..22].copy_from_slice(&51_000u16.to_be_bytes());
        packet[22..24].copy_from_slice(&443u16.to_be_bytes());
        let app =
            ApplicationDescriptor::from_uplink_packet(&packet).expect("a classifiable packet");

        let (psi, outs) = orch.start_establishment_for_app(&app, &test_params());
        assert!(psi.is_some(), "a session must be established for the flow");

        let (transport, _) = unwrap_ul(first_sent_pdu(&outs));
        assert_eq!(
            transport.dnn.as_deref(),
            Some(encode_dnn_value("video").as_slice()),
            "the session must request the DNN the port-keyed rule selected"
        );
        assert_eq!(
            transport.s_nssai.as_deref(),
            Some([4u8].as_slice()),
            "and its S-NSSAI"
        );
    }

    /// The counterpart: a flow the rule does not describe is not steered by it, so
    /// the session carries the configured defaults instead. Without this, the test
    /// above would pass against an implementation that steered everything.
    #[test]
    fn a_flow_that_matches_no_rule_keeps_the_configured_session_parameters() {
        use super::super::ursp::ApplicationDescriptor;

        let mut orch = new_orch();
        orch.set_ursp_policy(super::super::ursp::UrspPolicy::from_config(&ursp_config(
            true, "port:443", "video", 4,
        )));

        let mut packet = vec![0u8; 40];
        packet[0] = 0x45;
        packet[9] = 6;
        packet[16..20].copy_from_slice(&[93, 184, 216, 34]);
        packet[22..24].copy_from_slice(&25u16.to_be_bytes()); // SMTP, not 443
        let app = ApplicationDescriptor::from_uplink_packet(&packet).expect("classifiable");

        let (_, outs) = orch.start_establishment_for_app(&app, &test_params());
        let (transport, _) = unwrap_ul(first_sent_pdu(&outs));
        assert_ne!(
            transport.dnn.as_deref(),
            Some(encode_dnn_value("video").as_slice()),
            "a port-25 flow must not be steered by a port-443 rule"
        );
    }

    /// #47 criterion 2+5: with evaluation ON, a configured URSP rule steers the
    /// established session's DNN and S-NSSAI — asserted on the UL NAS TRANSPORT
    /// that goes out, not on the internal params.
    ///
    /// The DNN and S-NSSAI ride in the UL NAS Transport wrapper (TS 24.501
    /// §5.4.5.2.2), so reading them off the encoded PDU is what proves the network
    /// is actually asked for the steered route.
    #[test]
    fn an_enabled_ursp_rule_steers_the_established_session() {
        let mut orch = new_orch();
        orch.set_ursp_policy(super::super::ursp::UrspPolicy::from_config(&ursp_config(
            true, "*", "ims", 3,
        )));
        assert!(orch.ursp_enabled());

        let outs = orch.establish_default_sessions(&[test_params()]);
        let (transport, _) = unwrap_ul(first_sent_pdu(&outs));
        assert_eq!(
            transport.dnn.as_deref(),
            Some(encode_dnn_value("ims").as_slice()),
            "the URSP route's DNN must be the one requested"
        );
        assert_eq!(
            transport.s_nssai.as_deref(),
            Some([3u8].as_slice()),
            "the URSP route's S-NSSAI must be the one requested"
        );

        // And the stored session records the steered parameters, so a later reuse
        // check compares against what was actually established.
        let session = orch.session(1).expect("session on PSI 1");
        assert_eq!(session.params.dnn.as_deref(), Some("ims"));
        assert_eq!(session.params.s_nssai, Some(vec![3]));
    }

    /// #47: with evaluation OFF the same config establishes the STATIC session,
    /// byte-for-byte as before.
    ///
    /// This is the default and the regression that matters: the switch being off
    /// must be indistinguishable from the engine not existing.
    #[test]
    fn a_disabled_ursp_rule_leaves_the_static_session_alone() {
        let mut orch = new_orch();
        orch.set_ursp_policy(super::super::ursp::UrspPolicy::from_config(&ursp_config(
            false, "*", "ims", 3,
        )));
        assert!(!orch.ursp_enabled());

        let params = test_params();
        let outs = orch.establish_default_sessions(&[params.clone()]);
        let (transport, _) = unwrap_ul(first_sent_pdu(&outs));
        assert_eq!(
            transport.dnn.as_deref(),
            params.dnn.as_deref().map(encode_dnn_value).as_deref(),
            "the configured DNN, not the URSP route's"
        );
        assert_eq!(orch.session(1).unwrap().params, params);
    }

    /// #47: an application whose route already has a session REUSES it rather
    /// than establishing a duplicate (TS 23.503 §6.6.2.3).
    #[test]
    fn an_application_reuses_a_session_that_already_carries_its_route() {
        let mut orch = new_orch();
        orch.set_ursp_policy(super::super::ursp::UrspPolicy::from_config(&ursp_config(
            true, "*", "ims", 3,
        )));
        let base = test_params();

        let (first_psi, outs) = orch.start_establishment_for_app(
            &super::super::ursp::ApplicationDescriptor::for_dnn("internet"),
            &base,
        );
        let first_psi = first_psi.expect("a session was established");
        assert!(!outs.is_empty(), "the first application establishes one");

        // Bring it fully ACTIVE, so the reuse path sees a real session.
        let pti = unwrap_ul(first_sent_pdu(&outs)).1[2];
        orch.handle_dl_nas_transport(&dl_with_container(
            conformant_accept_bytes(first_psi, pti),
            first_psi,
        ));
        assert_eq!(
            orch.session(first_psi).map(|s| s.state),
            Some(PsState::Active)
        );

        let (reused_psi, outs) = orch.start_establishment_for_app(
            &super::super::ursp::ApplicationDescriptor::for_dnn("internet"),
            &base,
        );
        assert_eq!(
            reused_psi,
            Some(first_psi),
            "the same session must serve it"
        );
        assert!(
            outs.is_empty(),
            "reuse must send nothing, got {} output(s)",
            outs.len()
        );
        assert_eq!(orch.active_sessions().len(), 1, "no duplicate session");
    }

    /// #47: a second application steered to a DIFFERENT route gets its OWN
    /// session rather than being folded onto the first.
    ///
    /// Without this the reuse test above passes against an implementation that
    /// reuses whatever session exists, which is exactly the defect URSP steering
    /// is meant to prevent.
    #[test]
    fn a_differently_steered_application_gets_its_own_session() {
        use nextgsim_common::config::{RouteDescriptor, UrspRule as ConfigUrspRule};
        use nextgsim_common::types::SNssai;
        let mut config = UeConfig::default();
        config.ursp_evaluation = true;
        config.ursp_rules = vec![
            ConfigUrspRule {
                precedence: 10,
                traffic_descriptor: "dnn:internet".to_string(),
                route_descriptors: vec![RouteDescriptor {
                    s_nssai: Some(SNssai::new(1)),
                    dnn: Some("internet".to_string()),
                    session_type: None,
                    ssc_mode: None,
                }],
            },
            ConfigUrspRule {
                precedence: 20,
                traffic_descriptor: "dnn:ims".to_string(),
                route_descriptors: vec![RouteDescriptor {
                    s_nssai: Some(SNssai::new(2)),
                    dnn: Some("ims".to_string()),
                    session_type: None,
                    ssc_mode: None,
                }],
            },
        ];
        let mut orch = new_orch();
        orch.set_ursp_policy(super::super::ursp::UrspPolicy::from_config(&config));
        let base = test_params();

        let (first, _) = orch.start_establishment_for_app(
            &super::super::ursp::ApplicationDescriptor::for_dnn("internet"),
            &base,
        );
        let (second, outs) = orch.start_establishment_for_app(
            &super::super::ursp::ApplicationDescriptor::for_dnn("ims"),
            &base,
        );
        assert_ne!(first, second, "different routes need different sessions");
        assert!(!outs.is_empty(), "the second route establishes its own");
        assert_eq!(
            orch.session(second.unwrap()).unwrap().params.s_nssai,
            Some(vec![2]),
            "the second session carries the second rule's slice"
        );
    }

    /// #47 criterion 3: network-DELIVERED rules reach the evaluation source and
    /// influence establishment.
    #[test]
    fn delivered_ursp_rules_influence_establishment() {
        use nextgsim_nas::messages::mm::ue_policy::{
            RouteSelectionDescriptor, RouteSelectionDescriptorComponent,
            TrafficDescriptorComponent, UrspRule,
        };
        let mut orch = new_orch();
        // The switch still governs delivered rules: a PCF cannot turn evaluation
        // on for a UE whose operator has not.
        let mut config = UeConfig::default();
        config.ursp_evaluation = true;
        orch.set_ursp_policy(super::super::ursp::UrspPolicy::from_config(&config));
        orch.set_delivered_ursp_rules(vec![UrspRule {
            precedence: 5,
            traffic_descriptor: vec![TrafficDescriptorComponent::MatchAll],
            route_selection_descriptors: vec![RouteSelectionDescriptor {
                precedence: 1,
                components: vec![
                    RouteSelectionDescriptorComponent::Dnn("delivered".to_string()),
                    RouteSelectionDescriptorComponent::SNssai { sst: 9, sd: None },
                ],
            }],
            ureri: None,
        }]);
        assert_eq!(orch.ursp_rule_count(), 1);

        let outs = orch.establish_default_sessions(&[test_params()]);
        let (transport, _) = unwrap_ul(first_sent_pdu(&outs));
        assert_eq!(
            transport.dnn.as_deref(),
            Some(encode_dnn_value("delivered").as_slice())
        );
        assert_eq!(transport.s_nssai.as_deref(), Some([9u8].as_slice()));
    }

    #[test]
    fn test_release_all_locally() {
        let (mut orch, psi) = established();
        let outs = orch.release_all_locally();
        assert_eq!(outs, vec![SmOutput::SessionReleased { psi }]);
        assert!(orch.session(psi).is_none());
        assert!(!orch.pt.has_pending());
    }

    /// #51: a session the network reports INACTIVE in the PDU session status IE
    /// is released locally and its slot cleared (TS 24.501 §5.6.1.4.2 a).
    ///
    /// The bitmap deliberately has OTHER bits set (PSI 2 and 13): the release
    /// must follow from this PSI's own bit being clear, not from the bitmap being
    /// empty, which is what an `== 0` check would accept.
    #[test]
    fn a_session_the_network_reports_inactive_is_released_locally() {
        let (mut orch, psi) = established();
        assert!(orch.session(psi).is_some(), "precondition: session is up");
        let elsewhere = (1u16 << 2) | (1u16 << 13);
        assert_eq!(
            elsewhere & (1u16 << psi),
            0,
            "the fixture must leave THIS psi's bit clear"
        );

        let outs = orch.reconcile_pdu_session_status(elsewhere);

        assert_eq!(outs, vec![SmOutput::SessionReleased { psi }]);
        assert!(
            orch.session(psi).is_none(),
            "the session slot must be cleared, not merely reported"
        );
        assert!(
            !orch.pt.has_pending(),
            "the released session's procedure transaction must be returned"
        );
    }

    /// #51: a session the network reports ACTIVE survives reconciliation.
    ///
    /// Without this the previous test passes against a function that releases
    /// unconditionally — which would black-hole every session on any Accept
    /// carrying the IE.
    #[test]
    fn a_session_the_network_reports_active_survives() {
        let (mut orch, psi) = established();
        let outs = orch.reconcile_pdu_session_status(1u16 << psi);
        assert!(outs.is_empty(), "nothing to release, got {outs:?}");
        assert!(orch.session(psi).is_some());
    }

    /// #51: an establishment still in flight is reconciled too.
    ///
    /// A session in `ActivePending` that the network reports inactive is one whose
    /// establishment the network abandoned. Leaving it pending would pin the PSI
    /// and keep its T3580 retransmitting at a peer that has no such session.
    #[test]
    fn a_pending_establishment_the_network_does_not_know_is_released() {
        let (mut orch, psi, _pti) = started_establishment();
        assert_eq!(
            orch.session(psi).map(|s| s.state),
            Some(PsState::ActivePending)
        );
        assert!(orch.pt.has_pending(), "precondition: T3580 is running");

        let outs = orch.reconcile_pdu_session_status(0);

        assert_eq!(outs, vec![SmOutput::SessionReleased { psi }]);
        assert!(orch.session(psi).is_none());
        assert!(
            !orch.pt.has_pending(),
            "the aborted establishment must not leave a timer running"
        );
    }

    // ========================================================================
    // Config-derived session parameters
    // ========================================================================

    #[test]
    fn test_session_params_from_default_config() {
        let config = UeConfig::default();
        let params = SmSessionParams::from_config(&config);
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].session_type, PduSessionTypeValue::Ipv4);
        assert_eq!(params[0].dnn.as_deref(), Some("internet"));
        assert_eq!(params[0].requested_5qi, None);
    }

    #[test]
    fn test_xr_session_selects_xr_dnn() {
        use nextgsim_common::config::SessionConfig as ConfigSessionConfig;
        let mut config = UeConfig::default();
        // XR session with requested_5qi but no explicit APN: the UE must
        // select the XR DNN so the SMF derives the XR 5QI.
        config.sessions = vec![ConfigSessionConfig {
            requested_5qi: Some(82),
            apn: None,
            ..ConfigSessionConfig::default()
        }];
        let params = SmSessionParams::from_config(&config);
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].requested_5qi, Some(82));
        assert_eq!(params[0].dnn.as_deref(), Some("xr"));

        // 5QI 84 maps to the split-rendering XR DNN.
        config.sessions = vec![ConfigSessionConfig {
            requested_5qi: Some(84),
            apn: None,
            ..ConfigSessionConfig::default()
        }];
        let params = SmSessionParams::from_config(&config);
        assert_eq!(params[0].dnn.as_deref(), Some("xr-split"));

        // An explicit APN always wins over the XR-derived DNN.
        config.sessions = vec![ConfigSessionConfig {
            requested_5qi: Some(82),
            apn: Some("custom-xr".to_string()),
            ..ConfigSessionConfig::default()
        }];
        let params = SmSessionParams::from_config(&config);
        assert_eq!(params[0].dnn.as_deref(), Some("custom-xr"));
    }

    // ========================================================================
    // #379: re-attempting a configured session after a failed establishment
    // (TS 24.501 §6.4.1.4.2 / §6.4.1.4.3, §6.3.2.3, §6.3.3.3, §5.4.5.3.3)
    // ========================================================================

    /// GPRS timer 3 (TS 24.008 §10.5.7.4a), unit 2 s (0b011): `n` × 2 s.
    fn two_s(n: u8) -> u8 {
        0x60 | (n & 0x1F)
    }

    /// GPRS timer 3 value zero (unit 2 s, value 0)
    const BACKOFF_ZERO: u8 = 0x60;

    /// GPRS timer 3 "deactivated" (unit 0b111)
    const BACKOFF_DEACTIVATED: u8 = 0xE0;

    /// Every PDU SESSION ESTABLISHMENT REQUEST in `outs`, decoded off the
    /// wire: (UL NAS TRANSPORT, PSI, PTI).
    fn sent_establishment_requests(outs: &[SmOutput]) -> Vec<(UlNasTransport, u8, u8)> {
        outs.iter()
            .filter_map(|o| match o {
                SmOutput::SendNasPdu(pdu) => Some(unwrap_ul(pdu)),
                _ => None,
            })
            .filter(|(_, inner)| inner[3] == NasSmMessageType::PduSessionEstablishmentRequest as u8)
            .map(|(t, inner)| {
                let psi = t.pdu_session_id.unwrap();
                (t, psi, inner[2])
            })
            .collect()
    }

    /// The single establishment request in `outs`: (PSI, PTI).
    fn one_request(outs: &[SmOutput]) -> (u8, u8) {
        let reqs = sent_establishment_requests(outs);
        assert_eq!(
            reqs.len(),
            1,
            "one PDU SESSION ESTABLISHMENT REQUEST: {outs:?}"
        );
        let (t, psi, pti) = &reqs[0];
        assert_eq!(t.request_type, Some(RequestType::InitialRequest));
        (*psi, *pti)
    }

    /// An orchestrator holding `params` as its configured sessions, with each
    /// one's request sent: returns (orchestrator, [(psi, pti)]).
    fn configured(params: &[SmSessionParams]) -> (SmOrchestrator, Vec<(u8, u8)>) {
        let mut orch = new_orch();
        let outs = orch.establish_default_sessions(params);
        let ids = sent_establishment_requests(&outs)
            .into_iter()
            .map(|(_, psi, pti)| (psi, pti))
            .collect();
        (orch, ids)
    }

    fn deliver(orch: &mut SmOrchestrator, container: Vec<u8>, psi: u8) -> Vec<SmOutput> {
        orch.handle_dl_nas_transport(&dl_with_container(container, psi))
    }

    /// Run every back-off timer out through the real `perform_tick` path (the
    /// timers are wall-clock driven) and tick once.
    fn expire_backoffs(orch: &mut SmOrchestrator) -> Vec<SmOutput> {
        for entry in orch.backoff.values_mut() {
            if let Some(timer) = entry.timer.as_mut() {
                let mut zero = UeTimer::new(timer.code(), false, 0);
                zero.start(true);
                *timer = zero;
            }
        }
        orch.tick()
    }

    /// Let every re-attempt's own wait run out.
    fn elapse_retry_waits(orch: &mut SmOrchestrator) {
        orch.end_retry_waits_for_test();
    }

    /// The UE's own wait queued for the configured session `base`, in whole
    /// seconds (rounded up).
    fn queued_wait(orch: &SmOrchestrator, base: &SmSessionParams) -> u64 {
        let r = orch
            .retries
            .iter()
            .find(|r| r.base == *base)
            .expect("a queued re-attempt");
        let left = r
            .due_at
            .saturating_duration_since(std::time::Instant::now());
        left.as_secs() + u64::from(left.subsec_nanos() > 0)
    }

    fn release_command_bytes(psi: u8, cause: SmCause, backoff: Option<u8>) -> Vec<u8> {
        use nextgsim_nas::ies::ie3;
        let mut cmd =
            PduSessionReleaseCommand::new(psi, 0, ie3::SmCause::try_from(cause as u8).unwrap());
        cmd.back_off_timer_value = backoff;
        let mut buf = Vec::new();
        cmd.encode(&mut buf);
        buf
    }

    /// The devtest1 trace of #379: the configured session is released with #39,
    /// its re-establishment is rejected #26 with a back-off timer, and the UE
    /// asks again the moment T3396 expires -- not at T3512.
    #[test]
    fn a_configured_session_rejected_26_is_re_requested_when_t3396_expires() {
        let (mut orch, ids) = configured(&[test_params()]);
        let (psi, pti) = ids[0];
        deliver(&mut orch, conformant_accept_bytes(psi, pti), psi);

        // #39: Release Complete and a fresh request
        let outs = deliver(
            &mut orch,
            release_command_bytes(psi, SmCause::ReactivationRequested, None),
            psi,
        );
        let (psi, pti) = one_request(&outs);

        // #26 with a 2 s back-off: T3396 runs for the DNN
        let outs = deliver(
            &mut orch,
            reject_bytes(psi, pti, SmCause::InsufficientResources, Some(two_s(1))),
            psi,
        );
        assert!(outs.contains(&SmOutput::SessionEstablishmentFailed {
            psi,
            cause: SmCause::InsufficientResources
        }));
        assert!(orch.is_backed_off(Some(&[0x01]), Some("internet")));
        assert!(orch.has_pending_retry());

        // While T3396 runs nothing goes out for the DNN
        assert!(orch.tick().is_empty());
        assert!(!orch.has_due_retry());
        assert!(orch.retry_wanted_sessions().is_empty());

        // T3396 expires: the configured session is asked for again
        expire_backoffs(&mut orch);
        assert!(!orch.is_backed_off(Some(&[0x01]), Some("internet")));
        assert!(orch.has_due_retry());
        let outs = orch.retry_wanted_sessions();
        let reqs = sent_establishment_requests(&outs);
        assert_eq!(reqs.len(), 1, "{outs:?}");
        let (t, psi, pti) = &reqs[0];
        assert_eq!(t.dnn, Some(encode_dnn_value("internet")));
        assert_eq!(t.s_nssai, Some(vec![0x01]));
        assert!(!orch.has_pending_retry());

        // ... and the session comes back
        let outs = deliver(&mut orch, conformant_accept_bytes(*psi, *pti), *psi);
        assert!(outs.contains(&SmOutput::SessionEstablished {
            psi: *psi,
            ipv4: Some([10, 45, 0, 2])
        }));
        assert!(orch.retry_wanted_sessions().is_empty());
    }

    /// The same for the initial establishment, and only for the configured
    /// sessions: a one-off request is not re-attempted.
    #[test]
    fn only_configured_sessions_are_re_attempted() {
        let (mut orch, ids) = configured(&[test_params()]);
        let (psi, pti) = ids[0];
        deliver(
            &mut orch,
            reject_bytes(psi, pti, SmCause::InsufficientResources, Some(two_s(1))),
            psi,
        );
        expire_backoffs(&mut orch);
        one_request(&orch.retry_wanted_sessions());

        let mut one_off = test_params();
        one_off.dnn = Some("ims".to_string());
        let (psi, pti) = one_request(&orch.start_establishment(one_off));
        deliver(
            &mut orch,
            reject_bytes(psi, pti, SmCause::InsufficientResources, Some(two_s(1))),
            psi,
        );
        expire_backoffs(&mut orch);
        elapse_retry_waits(&mut orch);
        assert!(!orch.has_pending_retry());
        assert!(orch.retry_wanted_sessions().is_empty());
    }

    /// §6.4.1.4.2: a zero back-off timer value stops T3396 and lets the UE ask
    /// again; the UE's own wait paces it. Before #379 zero was taken for
    /// "deactivated" and barred the DNN until restart.
    #[test]
    fn a_zero_back_off_value_is_no_back_off() {
        let (mut orch, ids) = configured(&[test_params()]);
        let (psi, pti) = ids[0];
        deliver(
            &mut orch,
            reject_bytes(psi, pti, SmCause::InsufficientResources, Some(BACKOFF_ZERO)),
            psi,
        );
        assert!(!orch.is_backed_off(Some(&[0x01]), Some("internet")));
        assert_eq!(
            queued_wait(&orch, &test_params()),
            u64::from(SM_RETRY_BASE_SECS)
        );
        assert!(!orch.has_due_retry(), "the UE's own wait first");
        elapse_retry_waits(&mut orch);
        one_request(&orch.retry_wanted_sessions());
    }

    /// §6.4.1.4.2: a zero value also stops a T3396 that is running, and a new
    /// value replaces the running one.
    #[test]
    fn a_zero_back_off_value_stops_a_running_t3396() {
        let mut orch = new_orch();
        let p = test_params();
        orch.arm_backoff(SmCause::InsufficientResources, &p, Some(two_s(30)));
        assert_eq!(orch.backoff_remaining(None, Some("internet")), Some(60));
        orch.arm_backoff(SmCause::InsufficientResources, &p, Some(two_s(5)));
        assert_eq!(orch.backoff_remaining(None, Some("internet")), Some(10));
        orch.arm_backoff(SmCause::InsufficientResources, &p, Some(BACKOFF_ZERO));
        assert!(!orch.is_backed_off(Some(&[0x01]), Some("internet")));
    }

    /// No Back-off timer value IE: the UE asks again after its own wait, which
    /// doubles with each consecutive failure up to the ceiling.
    #[test]
    fn without_a_back_off_timer_the_ue_waits_its_own_growing_interval() {
        let (mut orch, ids) = configured(&[test_params()]);
        let (mut psi, mut pti) = ids[0];
        let mut expected = u64::from(SM_RETRY_BASE_SECS);
        for _ in 0..8 {
            deliver(
                &mut orch,
                reject_bytes(psi, pti, SmCause::InsufficientResources, None),
                psi,
            );
            assert!(!orch.is_backed_off(Some(&[0x01]), Some("internet")));
            assert_eq!(queued_wait(&orch, &test_params()), expected);
            assert!(
                orch.retry_wanted_sessions().is_empty(),
                "not before the wait"
            );
            elapse_retry_waits(&mut orch);
            (psi, pti) = one_request(&orch.retry_wanted_sessions());
            expected = (expected * 2).min(u64::from(SM_RETRY_MAX_SECS));
        }
        assert_eq!(expected, u64::from(SM_RETRY_MAX_SECS));
        // An accept ends the failures: the next failure waits the base again
        deliver(&mut orch, conformant_accept_bytes(psi, pti), psi);
        let outs = deliver(
            &mut orch,
            release_command_bytes(psi, SmCause::ReactivationRequested, None),
            psi,
        );
        let (psi, pti) = one_request(&outs);
        deliver(
            &mut orch,
            reject_bytes(psi, pti, SmCause::InsufficientResources, None),
            psi,
        );
        assert_eq!(
            queued_wait(&orch, &test_params()),
            u64::from(SM_RETRY_BASE_SECS)
        );
    }

    /// §6.4.1.4.2: a deactivated timer bars the DNN until the network sends a
    /// PDU SESSION MODIFICATION COMMAND or RELEASE COMMAND for it; either one
    /// stops the bar and the configured session is asked for again.
    #[test]
    fn a_deactivated_back_off_lifts_on_a_command_for_the_same_dnn() {
        for release in [false, true] {
            let first = test_params();
            let mut second = test_params();
            second.session_type = PduSessionTypeValue::Ipv6;
            let (mut orch, ids) = configured(&[first.clone(), second.clone()]);
            let [(psi1, pti1), (psi2, pti2)] = ids[..] else {
                panic!("two requests: {ids:?}");
            };
            deliver(&mut orch, conformant_accept_bytes(psi1, pti1), psi1);
            deliver(
                &mut orch,
                reject_bytes(
                    psi2,
                    pti2,
                    SmCause::InsufficientResources,
                    Some(BACKOFF_DEACTIVATED),
                ),
                psi2,
            );

            // Neither time nor ticking lifts it
            for _ in 0..3 {
                orch.tick();
            }
            elapse_retry_waits(&mut orch);
            assert!(orch.is_backed_off(Some(&[0x01]), Some("internet")));
            assert!(orch.retry_wanted_sessions().is_empty(), "release={release}");

            // A command for the session on the same DNN stops it
            if release {
                let outs = deliver(
                    &mut orch,
                    release_command_bytes(psi1, SmCause::RegularDeactivation, None),
                    psi1,
                );
                assert!(outs.contains(&SmOutput::SessionReleased { psi: psi1 }));
            } else {
                deliver(&mut orch, vec![0x2E, psi1, 0x00, 0xCB], psi1);
            }
            assert!(
                !orch.is_backed_off(Some(&[0x01]), Some("internet")),
                "release={release}"
            );
            let reqs = sent_establishment_requests(&orch.retry_wanted_sessions());
            assert_eq!(reqs.len(), 1, "release={release}: only the rejected one");
            let (t, _, _) = &reqs[0];
            assert_eq!(t.dnn, Some(encode_dnn_value("internet")));
        }
    }

    /// §6.4.1.4.2: a running T3396 is stopped by a command for the same DNN
    /// too, and a command for another DNN leaves it running.
    #[test]
    fn a_command_stops_a_running_back_off_only_in_its_scope() {
        let internet = test_params();
        let mut ims = test_params();
        ims.dnn = Some("ims".to_string());
        let mut internet6 = test_params();
        internet6.session_type = PduSessionTypeValue::Ipv6;
        let (mut orch, ids) = configured(&[ims.clone(), internet.clone(), internet6.clone()]);
        let [(p_ims, t_ims), (p_in, t_in), (p_in6, t_in6)] = ids[..] else {
            panic!("three requests: {ids:?}");
        };
        deliver(&mut orch, conformant_accept_bytes(p_ims, t_ims), p_ims);
        deliver(&mut orch, conformant_accept_bytes(p_in, t_in), p_in);
        deliver(
            &mut orch,
            reject_bytes(
                p_in6,
                t_in6,
                SmCause::InsufficientResources,
                Some(two_s(30)),
            ),
            p_in6,
        );
        assert!(orch.is_backed_off(None, Some("internet")));

        // A command on the "ims" session: T3396 for "internet" keeps running
        deliver(&mut orch, vec![0x2E, p_ims, 0x00, 0xCB], p_ims);
        assert!(orch.is_backed_off(None, Some("internet")));
        assert!(orch.retry_wanted_sessions().is_empty());

        // A command on the "internet" session stops it
        deliver(&mut orch, vec![0x2E, p_in, 0x00, 0xCB], p_in);
        assert!(!orch.is_backed_off(None, Some("internet")));
        one_request(&orch.retry_wanted_sessions());
    }

    /// The per-S-NSSAI variants (§6.2.8, §6.4.1.4.2): #67 starts T3584 for the
    /// [S-NSSAI, DNN], #69 starts T3585 for the S-NSSAI, and each re-attempts
    /// the configured session on expiry. #70 is keyed like #67.
    #[test]
    fn t3584_and_t3585_re_attempt_on_expiry() {
        for (cause, code, other_dnn_blocked) in [
            (
                SmCause::InsufficientResourcesForSliceAndDnn,
                SM_TIMER_T3584,
                false,
            ),
            (SmCause::InsufficientResourcesForSlice, SM_TIMER_T3585, true),
            (
                SmCause::MissingOrUnknownDnnInSlice,
                SM_TIMER_GENERIC_BACKOFF,
                false,
            ),
        ] {
            let (mut orch, ids) = configured(&[test_params()]);
            let (psi, pti) = ids[0];
            deliver(
                &mut orch,
                reject_bytes(psi, pti, cause, Some(two_s(1))),
                psi,
            );
            let codes: Vec<_> = orch
                .backoff
                .values()
                .filter_map(|e| e.timer.as_ref().map(UeTimer::code))
                .collect();
            assert_eq!(codes, vec![code], "{cause:?}");
            assert!(
                orch.is_backed_off(Some(&[0x01]), Some("internet")),
                "{cause:?}"
            );
            assert_eq!(
                orch.is_backed_off(Some(&[0x01]), Some("ims")),
                other_dnn_blocked,
                "{cause:?}"
            );
            assert!(
                !orch.is_backed_off(Some(&[0x02]), Some("internet")),
                "{cause:?}: other slice"
            );
            assert!(orch.retry_wanted_sessions().is_empty(), "{cause:?}");

            expire_backoffs(&mut orch);
            let reqs = sent_establishment_requests(&orch.retry_wanted_sessions());
            assert_eq!(reqs.len(), 1, "{cause:?}");
            assert_eq!(reqs[0].0.s_nssai, Some(vec![0x01]), "{cause:?}");
        }
    }

    /// T3585 is stopped by a command for a session on the same S-NSSAI,
    /// whatever its DNN.
    #[test]
    fn t3585_is_stopped_by_a_command_on_the_same_slice() {
        let mut ims = test_params();
        ims.dnn = Some("ims".to_string());
        let (mut orch, ids) = configured(&[ims, test_params()]);
        let [(p_ims, t_ims), (p_in, t_in)] = ids[..] else {
            panic!("two requests: {ids:?}");
        };
        deliver(&mut orch, conformant_accept_bytes(p_ims, t_ims), p_ims);
        deliver(
            &mut orch,
            reject_bytes(
                p_in,
                t_in,
                SmCause::InsufficientResourcesForSlice,
                Some(two_s(30)),
            ),
            p_in,
        );
        assert!(orch.is_backed_off(Some(&[0x01]), Some("internet")));
        deliver(&mut orch, vec![0x2E, p_ims, 0x00, 0xCB], p_ims);
        assert!(!orch.is_backed_off(Some(&[0x01]), Some("internet")));
        one_request(&orch.retry_wanted_sessions());
    }

    /// §6.3.3.3: a RELEASE COMMAND with #26 and a Back-off timer value starts
    /// T3396, and a configured session it released is asked for again on
    /// expiry -- not before.
    #[test]
    fn a_release_command_with_26_and_a_back_off_re_requests_on_expiry() {
        let (mut orch, ids) = configured(&[test_params()]);
        let (psi, pti) = ids[0];
        deliver(&mut orch, conformant_accept_bytes(psi, pti), psi);
        let outs = deliver(
            &mut orch,
            release_command_bytes(psi, SmCause::InsufficientResources, Some(two_s(1))),
            psi,
        );
        assert!(outs.contains(&SmOutput::SessionReleased { psi }));
        assert!(sent_establishment_requests(&outs).is_empty());
        assert!(orch.is_backed_off(None, Some("internet")));
        elapse_retry_waits(&mut orch);
        assert!(orch.retry_wanted_sessions().is_empty());
        expire_backoffs(&mut orch);
        one_request(&orch.retry_wanted_sessions());
    }

    /// A regular network release (#36) is not undone, and a session the UE
    /// released itself is no longer one it holds.
    #[test]
    fn a_released_session_is_not_re_requested() {
        let (mut orch, ids) = configured(&[test_params()]);
        let (psi, pti) = ids[0];
        deliver(&mut orch, conformant_accept_bytes(psi, pti), psi);
        deliver(
            &mut orch,
            release_command_bytes(psi, SmCause::RegularDeactivation, None),
            psi,
        );
        assert!(!orch.has_pending_retry());

        let (mut orch, ids) = configured(&[test_params()]);
        let (psi, pti) = ids[0];
        deliver(&mut orch, conformant_accept_bytes(psi, pti), psi);
        let (_, rel) = unwrap_ul(first_sent_pdu(&orch.start_release(psi)));
        // The network answers the release with #26 + back-off: still released
        let mut cmd = release_command_bytes(psi, SmCause::InsufficientResources, Some(two_s(1)));
        cmd[2] = rel[2];
        deliver(&mut orch, cmd, psi);
        expire_backoffs(&mut orch);
        elapse_retry_waits(&mut orch);
        assert!(!orch.has_pending_retry());
        assert!(orch.retry_wanted_sessions().is_empty());
    }

    /// TS 24.501 §5.4.5.3.3: a request the AMF could not forward (#90, no SMF
    /// reachable) is asked for again per the UE's own policy.
    #[test]
    fn a_request_not_forwarded_is_re_attempted_after_the_ue_wait() {
        let (mut orch, ids) = configured(&[test_params()]);
        let (psi, pti) = ids[0];
        let (_, sm) = unwrap_ul(&orch.pending.get(&pti).unwrap().ul_pdu.clone());
        let mut dl = DlNasTransport::new(PayloadContainerType::N1SmInformation, sm);
        dl.pdu_session_id = Some(psi);
        dl.mm_cause = Some(90);
        let outs = orch.handle_dl_nas_transport(&dl);
        assert!(outs.contains(&SmOutput::SessionEstablishmentFailed {
            psi,
            cause: SmCause::NetworkFailure
        }));
        assert_eq!(
            queued_wait(&orch, &test_params()),
            u64::from(SM_RETRY_BASE_SECS)
        );
        assert!(orch.retry_wanted_sessions().is_empty());
        elapse_retry_waits(&mut orch);
        one_request(&orch.retry_wanted_sessions());
    }

    /// Echo our request back in a DL NAS TRANSPORT with 5GMM `cause` and an
    /// optional Back-off timer value (TS 24.501 §5.4.5.3.3 g)).
    fn not_forwarded(
        orch: &mut SmOrchestrator,
        psi: u8,
        pti: u8,
        cause: u8,
        backoff: Option<u8>,
    ) -> Vec<SmOutput> {
        let (_, sm) = unwrap_ul(&orch.pending.get(&pti).unwrap().ul_pdu.clone());
        let mut dl = DlNasTransport::new(PayloadContainerType::N1SmInformation, sm);
        dl.pdu_session_id = Some(psi);
        dl.mm_cause = Some(cause);
        dl.back_off_timer_value = backoff;
        orch.handle_dl_nas_transport(&dl)
    }

    /// §5.4.5.3.3 g) 6): with #90 the UE ignores the Back-off timer value IE;
    /// its own wait applies.
    #[test]
    fn a_not_forwarded_back_off_value_is_ignored_for_90() {
        let (mut orch, ids) = configured(&[test_params()]);
        let (psi, pti) = ids[0];
        not_forwarded(&mut orch, psi, pti, 90, Some(two_s(30)));
        assert!(!orch.is_backed_off(Some(&[0x01]), Some("internet")));
        assert_eq!(
            queued_wait(&orch, &test_params()),
            u64::from(SM_RETRY_BASE_SECS)
        );
    }

    /// §5.4.5.3.3 g) 1), 4), 5) and §6.4.1.4.2: a request not forwarded for
    /// DNN or S-NSSAI based congestion (#22 / #67 / #69) starts T3396 / T3584 /
    /// T3585 with the Back-off timer value, and is re-attempted on expiry.
    #[test]
    fn a_request_not_forwarded_for_congestion_waits_the_back_off() {
        for (cause, code) in [
            (22, SM_TIMER_BACKOFF),
            (67, SM_TIMER_T3584),
            (69, SM_TIMER_T3585),
        ] {
            let (mut orch, ids) = configured(&[test_params()]);
            let (psi, pti) = ids[0];
            not_forwarded(&mut orch, psi, pti, cause, Some(two_s(1)));
            let codes: Vec<_> = orch
                .backoff
                .values()
                .filter_map(|e| e.timer.as_ref().map(UeTimer::code))
                .collect();
            assert_eq!(codes, vec![code], "#{cause}");
            elapse_retry_waits(&mut orch);
            assert!(
                orch.retry_wanted_sessions().is_empty(),
                "#{cause}: not before expiry"
            );
            expire_backoffs(&mut orch);
            one_request(&orch.retry_wanted_sessions());
        }
    }

    /// Any other not-forwarded cause (#28 restricted service area here) is
    /// not re-attempted.
    #[test]
    fn other_not_forwarded_causes_are_not_re_attempted() {
        let (mut orch, ids) = configured(&[test_params()]);
        let (psi, pti) = ids[0];
        not_forwarded(&mut orch, psi, pti, 28, None);
        assert!(!orch.has_pending_retry());
    }

    /// §6.4.1.4.3 a): #70 without a Back-off timer value starts the back-off
    /// with the SM Retry Timer default (12 minutes); a command does not stop
    /// it (it is not T3396 / T3584 / T3585).
    #[test]
    fn a_70_without_back_off_waits_the_sm_retry_timer() {
        let mut internet6 = test_params();
        internet6.session_type = PduSessionTypeValue::Ipv6;
        let (mut orch, ids) = configured(&[test_params(), internet6]);
        let [(p_in, t_in), (p_in6, t_in6)] = ids[..] else {
            panic!("two requests: {ids:?}");
        };
        deliver(&mut orch, conformant_accept_bytes(p_in6, t_in6), p_in6);
        deliver(
            &mut orch,
            reject_bytes(p_in, t_in, SmCause::MissingOrUnknownDnnInSlice, None),
            p_in,
        );
        assert_eq!(
            orch.backoff_remaining(Some(&[0x01]), Some("internet")),
            Some(SM_RETRY_TIMER_DEFAULT_SECS)
        );
        elapse_retry_waits(&mut orch);
        assert!(orch.retry_wanted_sessions().is_empty());
        // A command for a session in the same [S-NSSAI, DNN] does not stop the
        // §6.4.1.4.3 timer
        deliver(&mut orch, vec![0x2E, p_in6, 0x00, 0xCB], p_in6);
        assert!(orch.is_backed_off(Some(&[0x01]), Some("internet")));
        expire_backoffs(&mut orch);
        one_request(&orch.retry_wanted_sessions());
    }

    /// #380 review: T3584 (#67) and the §6.4.1.4.3 back-off (#70) for the same
    /// [S-NSSAI, DNN] are separate timers. A RELEASE COMMAND #67 with a new
    /// value, or with zero, leaves the 12-minute #70 back-off alone.
    #[test]
    fn a_67_release_leaves_the_70_back_off_alone() {
        for value in [two_s(1), BACKOFF_ZERO] {
            let mut internet6 = test_params();
            internet6.session_type = PduSessionTypeValue::Ipv6;
            let (mut orch, ids) = configured(&[test_params(), internet6]);
            let [(p_in, t_in), (p_in6, t_in6)] = ids[..] else {
                panic!("two requests: {ids:?}");
            };
            deliver(&mut orch, conformant_accept_bytes(p_in6, t_in6), p_in6);
            deliver(
                &mut orch,
                reject_bytes(p_in, t_in, SmCause::MissingOrUnknownDnnInSlice, None),
                p_in,
            );
            deliver(
                &mut orch,
                release_command_bytes(
                    p_in6,
                    SmCause::InsufficientResourcesForSliceAndDnn,
                    Some(value),
                ),
                p_in6,
            );
            let generic = orch
                .backoff
                .iter()
                .find(|(k, _)| k.timer == SM_TIMER_GENERIC_BACKOFF)
                .and_then(|(_, e)| e.timer.as_ref())
                .map(UeTimer::remaining);
            assert_eq!(
                generic,
                Some(SM_RETRY_TIMER_DEFAULT_SECS),
                "value {value:#04x}"
            );
            elapse_retry_waits(&mut orch);
            assert!(
                orch.retry_wanted_sessions().is_empty(),
                "value {value:#04x}: §6.4.1.4.3 still bars the [S-NSSAI, DNN]"
            );
        }
    }

    /// §6.3.3.3: a RELEASE COMMAND with #39 stops T3396 even when it carries a
    /// Back-off timer value IE, so the reactivation and the barred configured
    /// session both go out.
    #[test]
    fn a_39_release_with_a_back_off_value_still_stops_t3396() {
        let mut internet6 = test_params();
        internet6.session_type = PduSessionTypeValue::Ipv6;
        let (mut orch, ids) = configured(&[test_params(), internet6]);
        let [(p_in, t_in), (p_in6, t_in6)] = ids[..] else {
            panic!("two requests: {ids:?}");
        };
        deliver(&mut orch, conformant_accept_bytes(p_in, t_in), p_in);
        deliver(
            &mut orch,
            reject_bytes(
                p_in6,
                t_in6,
                SmCause::InsufficientResources,
                Some(two_s(30)),
            ),
            p_in6,
        );
        assert!(orch.is_backed_off(None, Some("internet")));
        let outs = deliver(
            &mut orch,
            release_command_bytes(p_in, SmCause::ReactivationRequested, Some(two_s(30))),
            p_in,
        );
        assert!(!orch.is_backed_off(None, Some("internet")));
        one_request(&outs);
        one_request(&orch.retry_wanted_sessions());
    }

    /// `backoff_remaining` reports the longest running back-off that applies.
    #[test]
    fn backoff_remaining_is_the_longest_that_applies() {
        let mut ims = test_params();
        ims.dnn = Some("ims".to_string());
        let (mut orch, ids) = configured(&[test_params(), ims]);
        let [(p_in, t_in), (p_ims, t_ims)] = ids[..] else {
            panic!("two requests: {ids:?}");
        };
        // T3396 60 s for "internet", T3585 10 s for the slice
        deliver(
            &mut orch,
            reject_bytes(p_in, t_in, SmCause::InsufficientResources, Some(two_s(30))),
            p_in,
        );
        deliver(
            &mut orch,
            reject_bytes(
                p_ims,
                t_ims,
                SmCause::InsufficientResourcesForSlice,
                Some(two_s(5)),
            ),
            p_ims,
        );
        assert_eq!(
            orch.backoff_remaining(Some(&[0x01]), Some("internet")),
            Some(60)
        );
        assert_eq!(orch.backoff_remaining(Some(&[0x01]), Some("ims")), Some(10));
    }

    /// §6.2.7: T3396 started for a request without a DNN is associated with
    /// "no DNN", not with every DNN.
    #[test]
    fn t3396_for_no_dnn_does_not_bar_other_dnns() {
        let mut no_dnn = test_params();
        no_dnn.dnn = None;
        let (mut orch, ids) = configured(&[no_dnn]);
        let (psi, pti) = ids[0];
        deliver(
            &mut orch,
            reject_bytes(psi, pti, SmCause::InsufficientResources, Some(two_s(30))),
            psi,
        );
        assert!(orch.is_backed_off(Some(&[0x01]), None));
        assert!(!orch.is_backed_off(Some(&[0x01]), Some("internet")));
        one_request(&orch.start_establishment(test_params()));
    }

    /// §6.3.3.3: a RELEASE COMMAND that carries a Back-off timer value IE
    /// (with a cause other than #39) does not lift a deactivated T3396.
    #[test]
    fn a_release_command_with_a_back_off_does_not_lift_a_deactivated_t3396() {
        let mut second = test_params();
        second.session_type = PduSessionTypeValue::Ipv6;
        let (mut orch, ids) = configured(&[test_params(), second]);
        let [(psi1, pti1), (psi2, pti2)] = ids[..] else {
            panic!("two requests: {ids:?}");
        };
        deliver(&mut orch, conformant_accept_bytes(psi1, pti1), psi1);
        deliver(
            &mut orch,
            reject_bytes(
                psi2,
                pti2,
                SmCause::InsufficientResources,
                Some(BACKOFF_DEACTIVATED),
            ),
            psi2,
        );
        deliver(
            &mut orch,
            release_command_bytes(psi1, SmCause::RegularDeactivation, Some(two_s(1))),
            psi1,
        );
        assert!(orch.is_backed_off(Some(&[0x01]), Some("internet")));
        elapse_retry_waits(&mut orch);
        assert!(orch.retry_wanted_sessions().is_empty());
    }

    /// TS 24.501 §6.4.1.6: T3580 exhausted -- a configured session is asked for
    /// again after the UE's own wait.
    #[test]
    fn t3580_exhaustion_re_attempts_a_configured_session() {
        let (mut orch, ids) = configured(&[test_params()]);
        let (psi, pti) = ids[0];
        for _ in 0..MAX_T3580_RETRIES {
            force_expire(&mut orch, pti);
            orch.handle_procedure_timer_expiry(pti);
        }
        assert!(orch.session(psi).is_none());
        assert_eq!(
            queued_wait(&orch, &test_params()),
            u64::from(SM_RETRY_BASE_SECS)
        );
        elapse_retry_waits(&mut orch);
        one_request(&orch.retry_wanted_sessions());
    }

    /// A configured session the back-off bars when the defaults are asked for
    /// (a registration while T3396 runs) is queued, not dropped.
    #[test]
    fn a_configured_session_barred_at_registration_is_asked_for_on_expiry() {
        let (mut orch, ids) = configured(&[test_params()]);
        let (psi, pti) = ids[0];
        deliver(
            &mut orch,
            reject_bytes(psi, pti, SmCause::InsufficientResources, Some(two_s(1))),
            psi,
        );
        // The next Registration Accept asks for the defaults again
        assert!(orch.establish_default_sessions(&[test_params()]).is_empty());
        assert!(orch.has_pending_retry());
        expire_backoffs(&mut orch);
        one_request(&orch.retry_wanted_sessions());
    }

    /// A local release of everything (deregistration) drops the queue: the
    /// next registration asks for the defaults itself.
    #[test]
    fn deregistration_drops_queued_re_attempts() {
        let (mut orch, ids) = configured(&[test_params()]);
        let (psi, pti) = ids[0];
        deliver(
            &mut orch,
            reject_bytes(psi, pti, SmCause::InsufficientResources, None),
            psi,
        );
        assert!(orch.has_pending_retry());
        orch.release_all_locally();
        assert!(!orch.has_pending_retry());
    }
}
