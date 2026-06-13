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

use crate::timer::{GprsTimer3, UeTimer, MAX_T3580_RETRIES, MAX_T3581_RETRIES, MAX_T3582_RETRIES};

use super::procedure::{
    ProcedureTransactionManager, SmMessageType as PtMessageType, SM_TIMER_T3580, SM_TIMER_T3581,
    SM_TIMER_T3582,
};

/// Maximum PDU session identity value (TS 24.501 Section 9.4)
pub const PSI_MAX: u8 = 15;

/// Timer code used for the T3396-class per-DNN / per-S-NSSAI back-off
/// timers (TS 24.008 Section 11.2.2.1A applied to 5GS, TS 24.501 6.4.1.4)
pub const SM_TIMER_BACKOFF: u16 = 3396;

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

/// Back-off scope key (TS 24.501 Section 6.4.1.4: the back-off applies per
/// [S-NSSAI, DNN] combination depending on the reject cause)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BackoffKey {
    s_nssai: Option<Vec<u8>>,
    dnn: Option<String>,
}

/// A running (or permanently blocking) back-off entry
#[derive(Debug)]
struct BackoffEntry {
    /// Running timer; `None` for a deactivated back-off timer value, which
    /// blocks retries until the UE is restarted (TS 24.501 6.4.1.4)
    timer: Option<UeTimer>,
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
    /// Pending UL PDUs by PTI for retransmission
    pending: HashMap<u8, PendingProcedure>,
    /// T3396-class back-off timers per [S-NSSAI, DNN]
    backoff: HashMap<BackoffKey, BackoffEntry>,
    /// PDU session type override per DNN after cause #50/#51/#57
    /// (TS 24.501 Section 6.4.1.4.2)
    type_override: HashMap<Option<String>, PduSessionTypeValue>,
    /// Accept transition leniency for the current nextgcore smfd emission
    /// (see [`PduSessionEstablishmentAccept::decode_legacy_nextgcore`]);
    /// strict parsing is always attempted first
    legacy_core_accept_compat: bool,
}

impl SmOrchestrator {
    /// Create a new orchestrator.
    ///
    /// `legacy_core_accept_compat` enables the documented transition
    /// leniency for the non-conformant PDU Session Establishment Accept
    /// currently emitted by nextgcore smfd. Strict TS 24.501 parsing is
    /// always attempted first.
    pub fn new(legacy_core_accept_compat: bool) -> Self {
        Self {
            sessions: Default::default(),
            pt: ProcedureTransactionManager::new(),
            pending: HashMap::new(),
            backoff: HashMap::new(),
            type_override: HashMap::new(),
            legacy_core_accept_compat,
        }
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    /// Get a session by PSI
    pub fn session(&self, psi: u8) -> Option<&PduSession> {
        if psi == 0 || psi > PSI_MAX {
            return None;
        }
        self.sessions[psi as usize].as_ref()
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
            let nssai_match = key.s_nssai.is_none() || key.s_nssai.as_deref() == s_nssai;
            let dnn_match = key.dnn.is_none() || key.dnn.as_deref() == dnn;
            let blocking = match &entry.timer {
                Some(t) => t.is_running(),
                None => true, // deactivated timer value: permanent block
            };
            nssai_match && dnn_match && blocking
        })
    }

    /// Remaining back-off seconds for the given [S-NSSAI, DNN], if running
    pub fn backoff_remaining(&self, s_nssai: Option<&[u8]>, dnn: Option<&str>) -> Option<u32> {
        let key = BackoffKey {
            s_nssai: s_nssai.map(<[u8]>::to_vec),
            dnn: dnn.map(str::to_string),
        };
        self.backoff
            .get(&key)
            .and_then(|e| e.timer.as_ref())
            .map(UeTimer::remaining)
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
    pub fn establish_default_sessions(&mut self, params: &[SmSessionParams]) -> Vec<SmOutput> {
        let mut outs = Vec::new();
        for p in params {
            let already = self
                .sessions
                .iter()
                .flatten()
                .any(|s| s.params == *p && s.state != PsState::Inactive);
            if !already {
                outs.extend(self.start_establishment(p.clone()));
            }
        }
        outs
    }

    /// Start a PDU session establishment procedure (TS 24.501 6.4.1.2).
    pub fn start_establishment(&mut self, mut params: SmSessionParams) -> Vec<SmOutput> {
        // Back-off consultation (TS 24.501 6.4.1.4 / 6.2.8)
        if self.is_backed_off(params.s_nssai.as_deref(), params.dnn.as_deref()) && !params.emergency
        {
            warn!(
                "PDU session establishment for DNN {:?} blocked by back-off timer ({:?}s left)",
                params.dnn,
                self.backoff_remaining(params.s_nssai.as_deref(), params.dnn.as_deref())
            );
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
            return Vec::new();
        };
        let Some(pti) = self.pt.allocate() else {
            warn!("No free procedure transaction identity");
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
        let Some(pti) = self.pt.allocate() else {
            warn!("No free procedure transaction identity");
            return Vec::new();
        };

        let mut req = PduSessionReleaseRequest::new(psi, pti);
        req.sm_cause = Some(ie3_cause(SmCause::RegularDeactivation));
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

        info!("Sending PDU Session Release Request: PSI={psi}, PTI={pti}, T3582 started");
        vec![SmOutput::SendNasPdu(ul_pdu)]
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
        outs
    }

    // ========================================================================
    // Timer driving
    // ========================================================================

    /// Drive the SM procedure and back-off timers; call roughly once a
    /// second.
    pub fn tick(&mut self) -> Vec<SmOutput> {
        let mut outs = Vec::new();

        // Procedure timers (T3580 / T3581 / T3582)
        for pti in self.pt.check_timers() {
            outs.extend(self.handle_procedure_timer_expiry(pti));
        }

        // Back-off timers: drop entries whose timer has expired
        self.backoff
            .retain(|key, entry| match entry.timer.as_mut() {
                Some(timer) => {
                    if timer.perform_tick() {
                        info!(
                            "Back-off timer expired for S-NSSAI {:?} / DNN {:?}",
                            key.s_nssai, key.dnn
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
        self.pt.abort(pti);
        self.pending.remove(&pti);

        match msg_type {
            PtMessageType::PduSessionEstablishmentRequest => {
                // 6.4.1.6 d): abort the procedure, release the allocated PSI
                self.sessions[psi as usize] = None;
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
        let Some((msg_type, _)) = self.pt.abort(pti) else {
            return Vec::new();
        };
        self.pending.remove(&pti);
        match msg_type {
            PtMessageType::PduSessionEstablishmentRequest => {
                self.sessions[psi as usize] = None;
                vec![SmOutput::SessionEstablishmentFailed {
                    psi,
                    cause: SmCause::NetworkFailure,
                }]
            }
            PtMessageType::PduSessionModificationRequest => {
                if let Some(s) = self.sessions[psi as usize].as_mut() {
                    s.state = PsState::Active;
                }
                vec![SmOutput::SessionModificationFailed {
                    psi,
                    cause: SmCause::NetworkFailure,
                }]
            }
            PtMessageType::PduSessionReleaseRequest => {
                self.sessions[psi as usize] = None;
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
                // Documented transition leniency: when enabled, retry with
                // the legacy nextgcore decoder before giving up
                let legacy = if self.legacy_core_accept_compat {
                    PduSessionEstablishmentAccept::decode_legacy_nextgcore(body, psi, pti).ok()
                } else {
                    None
                };
                match legacy {
                    Some(acc) => {
                        warn!(
                            "Establishment Accept parsed via the legacy nextgcore \
                             compatibility decoder (strict parse failed: {e:?})"
                        );
                        acc
                    }
                    None => {
                        // TS 24.501 7.7.2: missing/invalid mandatory IE ->
                        // 5GSM Status with cause #96; the procedure keeps
                        // running (T3580 will retransmit the request)
                        warn!("Establishment Accept failed strict decoding: {e:?}");
                        return self.send_sm_status(psi, pti, SmCause::InvalidMandatoryInformation);
                    }
                }
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

        if let Some(s) = self.sessions[psi as usize].as_mut() {
            s.state = PsState::Active;
            s.selected_type = Some(acc.selected_pdu_session_type.value);
            s.selected_ssc_mode = Some(acc.selected_ssc_mode.value);
            s.ipv4 = ipv4;
            s.authorized_qos_rules = acc.authorized_qos_rules.data.clone();
            s.session_ambr = Some(acc.session_ambr.clone());
        }

        info!(
            "PDU session {psi} established: type {:?}, SSC mode {:?}, IPv4 {:?}, \
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
        let session = self.sessions[psi as usize].take();
        let params = session.map(|s| s.params);

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
                        outs.extend(self.start_establishment(p));
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
                            outs.extend(self.start_establishment(p));
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
        let key = match cause {
            // #26: associated with the DNN (no S-NSSAI scope)
            SmCause::InsufficientResources => BackoffKey {
                s_nssai: None,
                dnn: params.dnn.clone(),
            },
            // #69: associated with the S-NSSAI only
            SmCause::InsufficientResourcesForSlice => BackoffKey {
                s_nssai: params.s_nssai.clone(),
                dnn: None,
            },
            // #67 / #70: associated with the [S-NSSAI, DNN] combination
            _ => BackoffKey {
                s_nssai: params.s_nssai.clone(),
                dnn: params.dnn.clone(),
            },
        };

        let Some(byte) = timer_byte else {
            // No back-off timer IE: the UE may retry after an
            // implementation specific delay (no timer armed)
            info!(
                "Reject cause #{} without back-off timer IE: no back-off armed",
                cause as u8
            );
            return;
        };
        let timer3 = GprsTimer3::from_byte(byte);
        let secs = timer3.to_seconds();
        if secs == 0 {
            // Deactivated timer value: block until restart
            warn!("Back-off timer deactivated for {key:?}: establishment blocked until restart");
            self.backoff.insert(key, BackoffEntry { timer: None });
            return;
        }
        let mut timer = UeTimer::new(SM_TIMER_BACKOFF, false, secs);
        timer.start(true);
        info!("Back-off timer started for {key:?}: {secs}s");
        self.backoff
            .insert(key, BackoffEntry { timer: Some(timer) });
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
                self.sessions[psi as usize] = None;
                outs.push(SmOutput::SessionReleased { psi });
            }
            // #26 / #67 / #69 / #70 with a back-off timer IE
            SmCause::InsufficientResources
            | SmCause::InsufficientResourcesForSliceAndDnn
            | SmCause::InsufficientResourcesForSlice
            | SmCause::MissingOrUnknownDnnInSlice => {
                let params = self.sessions[psi as usize]
                    .as_ref()
                    .map(|s| s.params.clone());
                if let Some(p) = params {
                    self.arm_backoff(cause, &p, rej.back_off_timer_value);
                }
                if let Some(s) = self.sessions[psi as usize].as_mut() {
                    s.state = PsState::Active;
                }
            }
            _ => {
                if let Some(s) = self.sessions[psi as usize].as_mut() {
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
                warn!("Failed to decode Release Command: {e:?}");
                return self.send_sm_status(psi, pti, SmCause::InvalidMandatoryInformation);
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

        let params = self.sessions[psi as usize].take().map(|s| s.params);

        // Reply with PDU Session Release Complete (TS 24.501 6.3.3.3)
        let complete = PduSessionReleaseComplete::new(psi, pti);
        let mut sm_pdu = Vec::new();
        complete.encode(&mut sm_pdu);
        info!("Sending PDU Session Release Complete: PSI={psi}, PTI={pti}");

        let mut outs = vec![
            SmOutput::SendNasPdu(build_ul_transport(sm_pdu, psi, None, None, None)),
            SmOutput::SessionReleased { psi },
        ];

        // Cause #39 "reactivation requested": re-establish the session
        // (TS 24.501 6.3.3.3)
        if cause == SmCause::ReactivationRequested {
            if let Some(p) = params {
                info!("Release Command requested reactivation: re-establishing the session");
                outs.extend(self.start_establishment(p));
            }
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
                self.sessions[psi as usize] = None;
                vec![SmOutput::SessionReleased { psi }]
            }
            _ => {
                // Any other cause: the session stays active
                if let Some(s) = self.sessions[psi as usize].as_mut() {
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
        SmOrchestrator::new(false)
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

    #[test]
    fn test_establishment_accept_missing_ambr_triggers_status_96() {
        let (mut orch, psi, pti) = started_establishment();
        // Truncate the conformant accept right before the session AMBR:
        // header(4) + octet5(1) + qos LV-E(2+9) = 16
        let bytes = conformant_accept_bytes(psi, pti)[..16].to_vec();
        let outs = orch.handle_dl_nas_transport(&dl_with_container(bytes, psi));
        let (_, inner) = unwrap_ul(first_sent_pdu(&outs));
        assert_eq!(inner[3], 0xD6, "must reply with 5GSM Status");
        assert_eq!(
            inner[4],
            SmCause::InvalidMandatoryInformation as u8,
            "cause must be #96"
        );
        // Session not activated; the procedure keeps running
        assert_eq!(orch.session(psi).unwrap().state, PsState::ActivePending);
        assert!(orch.pt.get(pti).unwrap().is_pending());
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
    fn test_legacy_core_accept_requires_compat_flag() {
        // The current nextgcore emission only parses with the compat flag
        fn legacy_accept(psi: u8, pti: u8) -> Vec<u8> {
            let mut msg = vec![0x2E, psi, pti, 0xC2, 0x01, 0x01];
            msg.extend_from_slice(&[0x06, 0x01, 0x03, 0x01, 0x01, 0x09]);
            msg.extend_from_slice(&[0x06, 0x06, 0x00, 0xC8, 0x06, 0x00, 0x32]);
            msg.extend_from_slice(&[0x29, 0x05, 0x01, 10, 45, 0, 2]);
            msg
        }

        // Without compat: 5GSM Status #96, session stays pending
        let (mut orch, psi, pti) = started_establishment();
        let outs = orch.handle_dl_nas_transport(&dl_with_container(legacy_accept(psi, pti), psi));
        let (_, inner) = unwrap_ul(first_sent_pdu(&outs));
        assert_eq!(inner[3], 0xD6);
        assert_eq!(orch.session(psi).unwrap().state, PsState::ActivePending);

        // With compat: the session activates
        let mut orch = SmOrchestrator::new(true);
        let outs = orch.start_establishment(test_params());
        let (t, inner) = unwrap_ul(first_sent_pdu(&outs));
        let (psi, pti) = (t.pdu_session_id.unwrap(), inner[2]);
        let outs = orch.handle_dl_nas_transport(&dl_with_container(legacy_accept(psi, pti), psi));
        assert_eq!(
            outs,
            vec![SmOutput::SessionEstablished {
                psi,
                ipv4: Some([10, 45, 0, 2])
            }]
        );
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

    #[test]
    fn test_unknown_sm_message_type_sends_status_97() {
        let mut orch = new_orch();
        let container = vec![0x2E, 1, 1, 0xC4]; // 0xC4 is not assigned
        let outs = orch.handle_dl_nas_transport(&dl_with_container(container, 1));
        let (_, inner) = unwrap_ul(first_sent_pdu(&outs));
        assert_eq!(inner[3], 0xD6);
        assert_eq!(inner[4], SmCause::MessageTypeNonExistent as u8);
    }

    #[test]
    fn test_release_all_locally() {
        let (mut orch, psi) = established();
        let outs = orch.release_all_locally();
        assert_eq!(outs, vec![SmOutput::SessionReleased { psi }]);
        assert!(orch.session(psi).is_none());
        assert!(!orch.pt.has_pending());
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
}
