//! RRC Resume Procedure (UE Side)
//!
//! This module implements the UE-side RRC Resume procedure as defined in
//! 3GPP TS 38.331 Section 5.3.13.
//!
//! # Overview
//!
//! The RRC Resume procedure allows a UE in `RRC_INACTIVE` state to quickly
//! resume its RRC connection without full re-establishment. The UE maintains
//! its AS (Access Stratum) context while in INACTIVE state.
//!
//! # Trigger Conditions
//!
//! The UE initiates RRC Resume when:
//! 1. MO (mobile-originated) data or signalling needs to be sent
//! 2. A paging message is received targeting the UE
//! 3. The UE needs to perform an RNA (RAN Notification Area) update
//!
//! # Procedure Flow
//!
//! ```text
//! UE (RRC_INACTIVE)                         gNB
//!  |-- RRCResumeRequest ------------------->|
//!  |<-- RRCResume (or RRCSetup fallback) ---|
//!  |-- RRCResumeComplete ------------------>|
//! ```
//!
//! # Timer T319
//!
//! - Started when `RRCResumeRequest` is sent
//! - Stopped when `RRCResume` or `RRCSetup` is received
//! - On expiry: UE goes back to RRC_IDLE (no fallback — full re-establishment needed)
//!
//! # Reference
//!
//! - 3GPP TS 38.331 Section 5.3.13
//! - UERANSIM `src/ue/rrc/` inactive state handling

use std::time::{Duration, Instant};

use super::state::{RrcState, RrcStateMachine, RrcStateTransition};

// ============================================================================
// Constants
// ============================================================================

/// T319: RRC Resume timer duration in ms (default 1000 ms per TS 38.331).
pub const T319_DEFAULT_MS: u64 = 1000;

// ============================================================================
// Resume Cause
// ============================================================================

/// Reason for initiating the RRC Resume procedure.
///
/// Maps directly to the `ResumeCause` IE in the `RRCResumeRequest` message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResumeCause {
    /// Emergency call or service
    Emergency,
    /// High-priority access
    HighPriorityAccess,
    /// Mobile-terminated access (paging response)
    MtAccess,
    /// Mobile-originated signalling
    MoSignalling,
    /// Mobile-originated data
    MoData,
    /// Mobile-originated voice call
    MoVoiceCall,
    /// Mobile-originated video call
    MoVideoCall,
    /// Mobile-originated SMS
    MoSms,
    /// RAN Notification Area (RNA) update
    RnaUpdate,
    /// MPS (Mission-Critical Push-to-talk Service) priority access
    MpsPriorityAccess,
    /// MCS (Mission-Critical Services) priority access
    McsPriorityAccess,
}

impl std::fmt::Display for ResumeCause {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Emergency => write!(f, "emergency"),
            Self::HighPriorityAccess => write!(f, "high-priority-access"),
            Self::MtAccess => write!(f, "mt-access"),
            Self::MoSignalling => write!(f, "mo-signalling"),
            Self::MoData => write!(f, "mo-data"),
            Self::MoVoiceCall => write!(f, "mo-voice-call"),
            Self::MoVideoCall => write!(f, "mo-video-call"),
            Self::MoSms => write!(f, "mo-sms"),
            Self::RnaUpdate => write!(f, "rna-update"),
            Self::MpsPriorityAccess => write!(f, "mps-priority-access"),
            Self::McsPriorityAccess => write!(f, "mcs-priority-access"),
        }
    }
}

// ============================================================================
// Resume Procedure State
// ============================================================================

/// State of the RRC Resume procedure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResumeProcedureState {
    /// No resume in progress
    #[default]
    Idle,
    /// Request sent, waiting for network response (T319 running)
    WaitingForResponse,
    /// Resume accepted — sending RRCResumeComplete
    Completing,
    /// Resume complete — in RRC_CONNECTED
    Complete,
    /// Resume failed (T319 expired or network rejected)
    Failed,
}

// ============================================================================
// Error Type
// ============================================================================

/// Errors for the RRC Resume procedure.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ResumeError {
    /// UE is not in RRC_INACTIVE state
    #[error("Cannot resume: UE is in state {0:?}, expected RRC_INACTIVE")]
    NotInactive(RrcState),
    /// Resume already in progress
    #[error("RRC Resume already in progress")]
    AlreadyInProgress,
    /// Procedure not in the expected state
    #[error("Procedure in unexpected state: {0:?}")]
    InvalidProcedureState(ResumeProcedureState),
    /// No resume identity available (short I-RNTI)
    #[error("No resume identity (I-RNTI) available")]
    NoResumeIdentity,
}

// ============================================================================
// Parameter Structs
// ============================================================================

/// Parameters for building an `RRCResumeRequest`.
#[derive(Debug, Clone)]
pub struct ResumeRequestParams {
    /// Short I-RNTI (24 bits), uniquely identifies the UE in INACTIVE
    pub resume_identity: u32,
    /// Resume MAC-I (16 bits), computed from AS security context
    pub resume_mac_i: u16,
    /// Cause for resuming
    pub cause: ResumeCause,
}

/// Parameters for building an `RRCResumeComplete`.
#[derive(Debug, Clone)]
pub struct ResumeCompleteParams {
    /// RRC Transaction Identifier from the received `RRCResume`
    pub rrc_transaction_id: u8,
    /// Optional dedicated NAS message to include (piggyback)
    pub dedicated_nas_message: Option<Vec<u8>>,
    /// Optional selected PLMN identity
    pub selected_plmn_identity: Option<u8>,
}

// ============================================================================
// Resume Procedure Manager
// ============================================================================

/// UE-side RRC Resume Procedure Manager.
///
/// Manages the full resume flow from INACTIVE → CONNECTED:
/// 1. Building and sending `RRCResumeRequest`
/// 2. Waiting for network response (T319)
/// 3. Processing `RRCResume` or `RRCSetup` fallback
/// 4. Sending `RRCResumeComplete`
#[derive(Debug)]
pub struct ResumeProcedure {
    /// Current state of the procedure
    state: ResumeProcedureState,
    /// Stored resume request params (for reference)
    last_request: Option<ResumeRequestParams>,
    /// T319 deadline
    t319_deadline: Option<Instant>,
}

impl Default for ResumeProcedure {
    fn default() -> Self {
        Self::new()
    }
}

impl ResumeProcedure {
    /// Creates a new Resume procedure manager.
    pub fn new() -> Self {
        Self {
            state: ResumeProcedureState::Idle,
            last_request: None,
            t319_deadline: None,
        }
    }

    /// Returns the current procedure state.
    pub fn state(&self) -> ResumeProcedureState {
        self.state
    }

    /// Returns true if a resume is in progress.
    pub fn is_in_progress(&self) -> bool {
        !matches!(
            self.state,
            ResumeProcedureState::Idle
                | ResumeProcedureState::Complete
                | ResumeProcedureState::Failed
        )
    }

    // ========================================================================
    // Step 1: Initiate (UE transitions from INACTIVE, sends RRCResumeRequest)
    // ========================================================================

    /// Initiates the RRC Resume procedure.
    ///
    /// Per 3GPP TS 38.331 Section 5.3.13.2:
    /// - UE must be in `RRC_INACTIVE` state
    /// - UE includes its short I-RNTI and computed resume MAC-I
    ///
    /// # Arguments
    /// * `cause` - Reason for resuming the connection
    /// * `resume_identity` - Short I-RNTI (24-bit value)
    /// * `resume_mac_i` - 16-bit MAC computed with AS security key
    /// * `rrc_sm` - UE RRC state machine (validated, not yet transitioned)
    ///
    /// # Returns
    /// * `Ok(ResumeRequestParams)` - Parameters for building `RRCResumeRequest`
    /// * `Err(error)` - If cannot initiate
    pub fn initiate(
        &mut self,
        cause: ResumeCause,
        resume_identity: u32,
        resume_mac_i: u16,
        rrc_sm: &RrcStateMachine,
    ) -> Result<ResumeRequestParams, ResumeError> {
        // Must be in INACTIVE
        if !rrc_sm.state().is_inactive() {
            return Err(ResumeError::NotInactive(rrc_sm.state()));
        }

        if self.is_in_progress() {
            return Err(ResumeError::AlreadyInProgress);
        }

        tracing::info!("RRC Resume initiated: cause={}", cause);

        let params = ResumeRequestParams {
            resume_identity,
            resume_mac_i,
            cause,
        };

        self.last_request = Some(params.clone());
        self.state = ResumeProcedureState::WaitingForResponse;
        self.t319_deadline = Some(Instant::now() + Duration::from_millis(T319_DEFAULT_MS));

        Ok(params)
    }

    // ========================================================================
    // Step 2a: Receive RRCResume (happy path)
    // ========================================================================

    /// Processes an `RRCResume` message from the network.
    ///
    /// Per 3GPP TS 38.331 Section 5.3.13.3:
    /// - Stop T319
    /// - Apply any new radio bearer / security configuration
    /// - Transition to `RRC_CONNECTED`
    /// - Send `RRCResumeComplete`
    ///
    /// # Arguments
    /// * `rrc_transaction_id` - From the received `RRCResume`
    /// * `dedicated_nas_message` - Optional NAS message to piggyback on Complete
    /// * `rrc_sm` - UE RRC state machine (will transition to CONNECTED)
    ///
    /// # Returns
    /// * `Ok(ResumeCompleteParams)` - Parameters for building `RRCResumeComplete`
    pub fn on_resume_received(
        &mut self,
        rrc_transaction_id: u8,
        dedicated_nas_message: Option<Vec<u8>>,
        rrc_sm: &mut RrcStateMachine,
    ) -> Result<ResumeCompleteParams, ResumeError> {
        if self.state != ResumeProcedureState::WaitingForResponse {
            return Err(ResumeError::InvalidProcedureState(self.state));
        }

        // Stop T319
        self.t319_deadline = None;

        // Transition RRC state to CONNECTED
        let _ = rrc_sm.transition(RrcStateTransition::Resume);

        self.state = ResumeProcedureState::Complete;

        tracing::info!("RRC Resume accepted — transitioning to RRC_CONNECTED");

        Ok(ResumeCompleteParams {
            rrc_transaction_id,
            dedicated_nas_message,
            selected_plmn_identity: None,
        })
    }

    // ========================================================================
    // Step 2b: Receive RRCRelease (network rejects resume — go to IDLE)
    // ========================================================================

    /// Handles an `RRCRelease` received in response to `RRCResumeRequest`.
    ///
    /// The network is releasing the connection entirely; UE goes to IDLE.
    /// Stops T319 and marks procedure as failed.
    pub fn on_release_received(&mut self, rrc_sm: &mut RrcStateMachine) {
        tracing::info!("RRC Release received during resume — moving to RRC_IDLE");
        self.t319_deadline = None;
        self.state = ResumeProcedureState::Failed;
        let _ = rrc_sm.on_rrc_release();
    }

    // ========================================================================
    // Step 2c: Receive RRCSetup (network falls back to fresh connection)
    // ========================================================================

    /// Handles an `RRCSetup` received instead of `RRCResume`.
    ///
    /// The network does not have the UE context and starts fresh.
    /// The AS context is lost; the UE should follow the `RRCSetup` procedure.
    pub fn on_setup_fallback(&mut self) {
        tracing::info!(
            "RRCSetup received during resume — falling back to fresh RRC connection (AS context lost)"
        );
        self.t319_deadline = None;
        self.state = ResumeProcedureState::Failed;
        // Caller handles RRCSetup normally
    }

    // ========================================================================
    // Timer handling
    // ========================================================================

    /// Checks whether T319 has expired.
    pub fn t319_expired(&self) -> bool {
        self.t319_deadline
            .is_some_and(|d| Instant::now() >= d)
    }

    /// Called when T319 expires.
    ///
    /// Per 3GPP TS 38.331 Section 5.3.13.4: UE moves to RRC_IDLE.
    pub fn on_t319_expired(&mut self, rrc_sm: &mut RrcStateMachine) {
        tracing::warn!("T319 expired — moving to RRC_IDLE");
        self.t319_deadline = None;
        self.state = ResumeProcedureState::Failed;
        // INACTIVE → IDLE via release
        let _ = rrc_sm.transition(RrcStateTransition::ReleaseFromInactive);
    }

    // ========================================================================
    // Reset
    // ========================================================================

    /// Resets the procedure to idle state.
    pub fn reset(&mut self) {
        self.state = ResumeProcedureState::Idle;
        self.last_request = None;
        self.t319_deadline = None;
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_inactive_sm() -> RrcStateMachine {
        let mut sm = RrcStateMachine::new();
        sm.transition(RrcStateTransition::SetupComplete).unwrap();
        sm.transition(RrcStateTransition::Suspend).unwrap();
        sm
    }

    #[test]
    fn test_initiate_from_inactive() {
        let sm = setup_inactive_sm();
        let mut proc = ResumeProcedure::new();

        let result = proc.initiate(ResumeCause::MoData, 0x123456, 0xABCD, &sm);

        assert!(result.is_ok());
        assert_eq!(proc.state(), ResumeProcedureState::WaitingForResponse);
        assert!(proc.is_in_progress());

        let params = result.unwrap();
        assert_eq!(params.resume_identity, 0x123456);
        assert_eq!(params.resume_mac_i, 0xABCD);
        assert!(matches!(params.cause, ResumeCause::MoData));
    }

    #[test]
    fn test_initiate_from_connected_fails() {
        let mut sm = RrcStateMachine::new();
        sm.transition(RrcStateTransition::SetupComplete).unwrap(); // CONNECTED
        let mut proc = ResumeProcedure::new();

        let result = proc.initiate(ResumeCause::MoData, 0x000001, 0x0001, &sm);

        assert!(matches!(result, Err(ResumeError::NotInactive(_))));
    }

    #[test]
    fn test_initiate_from_idle_fails() {
        let sm = RrcStateMachine::new(); // IDLE
        let mut proc = ResumeProcedure::new();

        let result = proc.initiate(ResumeCause::MoSignalling, 0x000002, 0x0002, &sm);

        assert!(matches!(result, Err(ResumeError::NotInactive(_))));
    }

    #[test]
    fn test_initiate_already_in_progress() {
        let sm = setup_inactive_sm();
        let mut proc = ResumeProcedure::new();

        let _ = proc.initiate(ResumeCause::MoData, 0x111111, 0x0001, &sm);

        let sm2 = setup_inactive_sm();
        let result = proc.initiate(ResumeCause::MoData, 0x222222, 0x0002, &sm2);

        assert!(matches!(result, Err(ResumeError::AlreadyInProgress)));
    }

    #[test]
    fn test_on_resume_received_happy_path() {
        let sm_ref = setup_inactive_sm();
        let mut proc = ResumeProcedure::new();
        let _ = proc.initiate(ResumeCause::MoData, 0x123456, 0xABCD, &sm_ref);

        let mut sm = setup_inactive_sm();
        let result = proc.on_resume_received(1, None, &mut sm);

        assert!(result.is_ok());
        let complete = result.unwrap();
        assert_eq!(complete.rrc_transaction_id, 1);
        assert!(complete.dedicated_nas_message.is_none());
        assert_eq!(proc.state(), ResumeProcedureState::Complete);
        assert_eq!(sm.state(), RrcState::Connected);
    }

    #[test]
    fn test_on_resume_received_with_nas_message() {
        let sm_ref = setup_inactive_sm();
        let mut proc = ResumeProcedure::new();
        let _ = proc.initiate(ResumeCause::MoSignalling, 0x000001, 0x0001, &sm_ref);

        let mut sm = setup_inactive_sm();
        let nas_msg = vec![0x7E, 0x00, 0x46]; // dummy NAS service request
        let result = proc.on_resume_received(0, Some(nas_msg.clone()), &mut sm);

        assert!(result.is_ok());
        let complete = result.unwrap();
        assert_eq!(complete.dedicated_nas_message, Some(nas_msg));
    }

    #[test]
    fn test_on_resume_received_wrong_state() {
        let mut proc = ResumeProcedure::new();
        let mut sm = setup_inactive_sm();

        let result = proc.on_resume_received(0, None, &mut sm);
        assert!(matches!(
            result,
            Err(ResumeError::InvalidProcedureState(_))
        ));
    }

    #[test]
    fn test_on_release_received_moves_to_idle() {
        let sm_ref = setup_inactive_sm();
        let mut proc = ResumeProcedure::new();
        let _ = proc.initiate(ResumeCause::MtAccess, 0x000001, 0x0001, &sm_ref);

        let mut sm = setup_inactive_sm();
        proc.on_release_received(&mut sm);

        assert_eq!(proc.state(), ResumeProcedureState::Failed);
        assert_eq!(sm.state(), RrcState::Idle);
    }

    #[test]
    fn test_on_setup_fallback() {
        let sm_ref = setup_inactive_sm();
        let mut proc = ResumeProcedure::new();
        let _ = proc.initiate(ResumeCause::MoData, 0x000001, 0x0001, &sm_ref);

        proc.on_setup_fallback();
        assert_eq!(proc.state(), ResumeProcedureState::Failed);
    }

    #[test]
    fn test_on_t319_expired() {
        let sm_ref = setup_inactive_sm();
        let mut proc = ResumeProcedure::new();
        let _ = proc.initiate(ResumeCause::MoData, 0x000001, 0x0001, &sm_ref);

        let mut sm = setup_inactive_sm();
        proc.on_t319_expired(&mut sm);

        assert_eq!(proc.state(), ResumeProcedureState::Failed);
        assert_eq!(sm.state(), RrcState::Idle);
    }

    #[test]
    fn test_reset_clears_state() {
        let sm_ref = setup_inactive_sm();
        let mut proc = ResumeProcedure::new();
        let _ = proc.initiate(ResumeCause::MoData, 0x123456, 0xABCD, &sm_ref);

        proc.reset();

        assert_eq!(proc.state(), ResumeProcedureState::Idle);
        assert!(!proc.is_in_progress());
    }

    #[test]
    fn test_resume_cause_display() {
        assert_eq!(format!("{}", ResumeCause::MoData), "mo-data");
        assert_eq!(format!("{}", ResumeCause::Emergency), "emergency");
        assert_eq!(format!("{}", ResumeCause::RnaUpdate), "rna-update");
        assert_eq!(
            format!("{}", ResumeCause::MpsPriorityAccess),
            "mps-priority-access"
        );
    }

    #[test]
    fn test_all_resume_causes_display() {
        let causes = [
            ResumeCause::Emergency,
            ResumeCause::HighPriorityAccess,
            ResumeCause::MtAccess,
            ResumeCause::MoSignalling,
            ResumeCause::MoData,
            ResumeCause::MoVoiceCall,
            ResumeCause::MoVideoCall,
            ResumeCause::MoSms,
            ResumeCause::RnaUpdate,
            ResumeCause::MpsPriorityAccess,
            ResumeCause::McsPriorityAccess,
        ];
        for cause in causes {
            // Just ensure display doesn't panic
            let _ = format!("{cause}");
        }
    }
}
