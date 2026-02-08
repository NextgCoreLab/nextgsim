//! Deregistration Procedure
//!
//! This module implements the UE-side deregistration procedure as defined in
//! 3GPP TS 24.501 Section 5.5.2.
//!
//! # Procedure Types
//!
//! - **UE-initiated deregistration** (Section 5.5.2.2): UE sends Deregistration Request
//!   to the network and waits for Deregistration Accept.
//! - **Network-initiated deregistration** (Section 5.5.2.3): Network sends Deregistration
//!   Request to the UE, and UE responds with Deregistration Accept.
//!
//! # Timer T3521
//!
//! Timer T3521 is used for UE-initiated deregistration:
//! - Started when Deregistration Request is sent (normal de-registration only)
//! - Stopped when Deregistration Accept is received
//! - On expiry: retransmit Deregistration Request (up to 5 times)
//!
//! # Reference
//!
//! Based on UERANSIM's `src/ue/nas/mm/dereg.cpp` implementation.

use thiserror::Error;

use nextgsim_nas::messages::mm::{
    DeregistrationAcceptUeOriginating, DeregistrationAcceptUeTerminated,
    DeregistrationRequestUeOriginating, DeregistrationRequestUeTerminated,
    Ie5gsMobileIdentity,
};
use nextgsim_nas::ies::ie1::{
    DeRegistrationAccessType, IeDeRegistrationType, ReRegistrationRequired, SwitchOff,
};
// Use the MmCause from messages::mm (re-exported at crate root)
// to match the types used in DeregistrationRequestUeTerminated
use nextgsim_nas::MmCause;
use nextgsim_nas::security::NasKeySetIdentifier;

use super::state::{MmState, MmSubState, RmState, UpdateStatus};
use crate::timer::UeTimer;

// ============================================================================
// Constants
// ============================================================================

/// Timer T3521 code
pub const T3521_CODE: u16 = 3521;

/// Default T3521 interval in seconds (15 seconds per 3GPP TS 24.501)
pub const T3521_DEFAULT_INTERVAL_SECS: u32 = 15;

/// Maximum T3521 expiry count before giving up
pub const T3521_MAX_RETRANSMISSION: u32 = 5;

// ============================================================================
// Deregistration Cause
// ============================================================================

/// Cause for UE-initiated deregistration.
///
/// Based on UERANSIM's `EDeregCause` enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeregistrationCause {
    /// Normal de-registration (explicit user request)
    ExplicitDeregistration,
    /// UE is switching off
    SwitchOff,
    /// USIM removal
    UsimRemoval,
    /// Disable 5G
    Disable5g,
}

impl std::fmt::Display for DeregistrationCause {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeregistrationCause::ExplicitDeregistration => write!(f, "explicit-deregistration"),
            DeregistrationCause::SwitchOff => write!(f, "switch-off"),
            DeregistrationCause::UsimRemoval => write!(f, "usim-removal"),
            DeregistrationCause::Disable5g => write!(f, "disable-5g"),
        }
    }
}

// ============================================================================
// Procedure Result
// ============================================================================

/// Result of a procedure operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcedureResult {
    /// Procedure completed successfully
    Ok,
    /// Procedure should stay in current state (retry later)
    Stay,
    /// Procedure was cancelled
    Cancel,
}

// ============================================================================
// Deregistration Error
// ============================================================================

/// Error type for deregistration procedure.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum DeregistrationProcedureError {
    /// UE is not in RM-REGISTERED state
    #[error("UE is not registered (current state: {0})")]
    NotRegistered(RmState),
    /// Deregistration already in progress
    #[error("Deregistration already in progress")]
    AlreadyInProgress,
    /// Invalid state for operation
    #[error("Invalid state for operation: {0}")]
    InvalidState(MmState),
    /// No current TAI available
    #[error("No current TAI available")]
    NoCurrentTai,
    /// UAC check failed
    #[error("UAC check failed")]
    UacFailed,
}

// ============================================================================
// Deregistration Procedure Handler
// ============================================================================

/// Handles the deregistration procedure for the UE.
///
/// This struct manages both UE-initiated and network-initiated deregistration
/// procedures, including state transitions, timer management, and message handling.
#[derive(Debug)]
pub struct DeregistrationProcedure {
    /// Timer T3521 for UE-initiated deregistration
    t3521: UeTimer,
    /// Last deregistration request sent (for retransmission)
    last_request: Option<DeregistrationRequestUeOriginating>,
    /// Cause of the last deregistration
    last_cause: Option<DeregistrationCause>,
}

impl Default for DeregistrationProcedure {
    fn default() -> Self {
        Self::new()
    }
}

impl DeregistrationProcedure {
    /// Creates a new deregistration procedure handler.
    pub fn new() -> Self {
        Self {
            t3521: UeTimer::new(T3521_CODE, true, T3521_DEFAULT_INTERVAL_SECS),
            last_request: None,
            last_cause: None,
        }
    }

    /// Returns a reference to the T3521 timer.
    pub fn t3521(&self) -> &UeTimer {
        &self.t3521
    }

    /// Returns a mutable reference to the T3521 timer.
    pub fn t3521_mut(&mut self) -> &mut UeTimer {
        &mut self.t3521
    }

    /// Returns the last deregistration cause.
    pub fn last_cause(&self) -> Option<DeregistrationCause> {
        self.last_cause
    }

    // ========================================================================
    // UE-Initiated Deregistration
    // ========================================================================

    /// Initiates UE-originated deregistration procedure.
    ///
    /// # Arguments
    /// * `cause` - The cause for deregistration
    /// * `rm_state` - Current RM state
    /// * `mm_state` - Current MM state
    /// * `mm_sub_state` - Current MM sub-state
    /// * `ng_ksi` - Current NAS key set identifier (if available)
    /// * `mobile_identity` - UE's mobile identity (5G-GUTI or SUCI)
    ///
    /// # Returns
    /// * `Ok((request, new_sub_state))` - The deregistration request to send and new sub-state
    /// * `Err(error)` - If deregistration cannot be initiated
    pub fn initiate_deregistration(
        &mut self,
        cause: DeregistrationCause,
        rm_state: RmState,
        mm_state: MmState,
        mm_sub_state: MmSubState,
        ng_ksi: Option<NasKeySetIdentifier>,
        mobile_identity: Ie5gsMobileIdentity,
    ) -> Result<(DeregistrationRequestUeOriginating, MmSubState), DeregistrationProcedureError>
    {
        // Check if UE is registered
        if rm_state != RmState::Registered {
            return Err(DeregistrationProcedureError::NotRegistered(rm_state));
        }

        // Check if deregistration is already in progress
        if mm_state == MmState::DeregisteredInitiated {
            return Err(DeregistrationProcedureError::AlreadyInProgress);
        }

        // Check if in UPDATE_NEEDED state (should stay and retry later)
        if mm_sub_state == MmSubState::RegisteredUpdateNeeded {
            return Err(DeregistrationProcedureError::InvalidState(mm_state));
        }

        // Build deregistration type IE
        let dereg_type = Self::make_deregistration_type(cause);

        // Build the request
        let request = DeregistrationRequestUeOriginating::new(
            dereg_type,
            ng_ksi.unwrap_or_else(NasKeySetIdentifier::no_key),
            mobile_identity,
        );

        // Store for potential retransmission
        self.last_request = Some(request.clone());
        self.last_cause = Some(cause);

        // Start T3521 for normal de-registration
        if dereg_type.switch_off == SwitchOff::NormalDeRegistration
            && (mm_state == MmState::Registered || mm_state == MmState::RegisteredInitiated) {
                self.t3521.start(true);
            }

        Ok((request, MmSubState::DeregisteredInitiated))
    }

    /// Handles reception of Deregistration Accept (UE originating).
    ///
    /// # Arguments
    /// * `_msg` - The received Deregistration Accept message
    /// * `mm_state` - Current MM state
    ///
    /// # Returns
    /// * `Ok(new_sub_state)` - The new MM sub-state to transition to
    /// * `Err(error)` - If the message cannot be processed
    pub fn receive_deregistration_accept(
        &mut self,
        _msg: &DeregistrationAcceptUeOriginating,
        mm_state: MmState,
    ) -> Result<MmSubState, DeregistrationProcedureError> {
        // Check state
        if mm_state != MmState::DeregisteredInitiated {
            return Err(DeregistrationProcedureError::InvalidState(mm_state));
        }

        // Stop timers
        self.t3521.stop(true);

        // Determine new state based on cause
        let new_state = match self.last_cause {
            Some(DeregistrationCause::Disable5g) => MmSubState::Null,
            _ => MmSubState::DeregisteredNormalService,
        };

        // Clear stored request
        self.last_request = None;

        Ok(new_state)
    }

    /// Handles T3521 timer expiry.
    ///
    /// # Returns
    /// * `Some(request)` - The request to retransmit
    /// * `None` - If max retransmissions reached (perform local deregistration)
    pub fn handle_t3521_expiry(&mut self) -> Option<DeregistrationRequestUeOriginating> {
        if self.t3521.expiry_count() >= T3521_MAX_RETRANSMISSION {
            // Max retransmissions reached, perform local deregistration
            self.last_request = None;
            return None;
        }

        // Retransmit the request
        if let Some(ref request) = self.last_request {
            self.t3521.start(false); // Don't clear expiry count
            return Some(request.clone());
        }

        None
    }

    // ========================================================================
    // Network-Initiated Deregistration
    // ========================================================================

    /// Handles reception of Deregistration Request (UE terminated / network-initiated).
    ///
    /// # Arguments
    /// * `msg` - The received Deregistration Request message
    /// * `rm_state` - Current RM state
    /// * `mm_state` - Current MM state
    ///
    /// # Returns
    /// * `Ok(result)` - The result containing response and state changes
    /// * `Err(error)` - If the message cannot be processed
    pub fn receive_deregistration_request(
        &mut self,
        msg: &DeregistrationRequestUeTerminated,
        rm_state: RmState,
        mm_state: MmState,
    ) -> Result<NetworkDeregistrationResult, DeregistrationProcedureError> {
        // Check if UE is registered
        if rm_state != RmState::Registered {
            return Err(DeregistrationProcedureError::NotRegistered(rm_state));
        }

        // Check access type
        if msg.deregistration_type.access_type == DeRegistrationAccessType::NonThreeGppAccess {
            return Ok(NetworkDeregistrationResult {
                accept: None, // Don't send accept, send MM status instead
                new_sub_state: None,
                clear_guti: false,
                clear_tai_list: false,
                clear_equivalent_plmn: false,
                invalidate_usim: false,
                new_update_status: None,
                re_registration_required: false,
            });
        }

        // Handle collision cases (5.5.2.2.6)
        if mm_state == MmState::DeregisteredInitiated {
            if let Some(ref last_req) = self.last_request {
                // If UE-initiated deregistration is switch-off, ignore network request
                if last_req.deregistration_type.switch_off == SwitchOff::SwitchOff {
                    return Ok(NetworkDeregistrationResult {
                        accept: None,
                        new_sub_state: None,
                        clear_guti: false,
                        clear_tai_list: false,
                        clear_equivalent_plmn: false,
                        invalidate_usim: false,
                        new_update_status: None,
                        re_registration_required: false,
                    });
                }
            }
        }

        let re_registration_required =
            msg.deregistration_type.re_registration_required == ReRegistrationRequired::Required;

        // Build the accept message
        let accept = DeregistrationAcceptUeTerminated::new();

        // Process based on cause and re-registration requirement
        let result = self.process_network_deregistration_cause(msg, re_registration_required);

        Ok(NetworkDeregistrationResult {
            accept: Some(accept),
            new_sub_state: Some(result.new_sub_state),
            clear_guti: result.clear_guti,
            clear_tai_list: result.clear_tai_list,
            clear_equivalent_plmn: result.clear_equivalent_plmn,
            invalidate_usim: result.invalidate_usim,
            new_update_status: result.new_update_status,
            re_registration_required,
        })
    }

    // ========================================================================
    // Local Deregistration
    // ========================================================================

    /// Performs local deregistration without network interaction.
    ///
    /// This is used when:
    /// - UAC check fails
    /// - T3521 expires maximum times
    /// - Other local conditions require deregistration
    pub fn perform_local_deregistration(&mut self) -> MmSubState {
        // Stop timers
        self.t3521.stop(true);

        // Clear stored request
        self.last_request = None;
        self.last_cause = None;

        MmSubState::DeregisteredNormalService
    }

    // ========================================================================
    // Helper Methods
    // ========================================================================

    /// Creates the deregistration type IE based on the cause.
    fn make_deregistration_type(cause: DeregistrationCause) -> IeDeRegistrationType {
        let switch_off = match cause {
            DeregistrationCause::SwitchOff | DeregistrationCause::UsimRemoval => SwitchOff::SwitchOff,
            _ => SwitchOff::NormalDeRegistration,
        };

        IeDeRegistrationType::new(
            DeRegistrationAccessType::ThreeGppAccess,
            ReRegistrationRequired::NotRequired,
            switch_off,
        )
    }

    /// Processes the network deregistration cause and determines state changes.
    fn process_network_deregistration_cause(
        &self,
        msg: &DeregistrationRequestUeTerminated,
        re_registration_required: bool,
    ) -> CauseProcessingResult {
        // If re-registration is required, ignore the cause
        if re_registration_required {
            return CauseProcessingResult {
                new_sub_state: MmSubState::DeregisteredNormalService,
                clear_guti: false,
                clear_tai_list: false,
                clear_equivalent_plmn: false,
                invalidate_usim: false,
                new_update_status: None,
            };
        }

        // Process based on 5GMM cause if present
        if let Some(ref cause_ie) = msg.mm_cause {
            return self.process_mm_cause(cause_ie.value);
        }

        // No cause and no re-registration required - abnormal case
        // Handle as per 5.5.2.3.4 item b)
        CauseProcessingResult {
            new_sub_state: MmSubState::DeregisteredPlmnSearch,
            clear_guti: true,
            clear_tai_list: true,
            clear_equivalent_plmn: true,
            invalidate_usim: false,
            new_update_status: Some(UpdateStatus::NotUpdated),
        }
    }

    /// Processes a specific 5GMM cause value.
    fn process_mm_cause(&self, cause: MmCause) -> CauseProcessingResult {
        match cause {
            MmCause::IllegalUe | MmCause::IllegalMe | MmCause::FiveGsServicesNotAllowed => {
                CauseProcessingResult {
                    new_sub_state: MmSubState::DeregisteredNormalService,
                    clear_guti: true,
                    clear_tai_list: true,
                    clear_equivalent_plmn: true,
                    invalidate_usim: true,
                    new_update_status: Some(UpdateStatus::RoamingNotAllowed),
                }
            }
            MmCause::PlmnNotAllowed | MmCause::RoamingNotAllowedInTa => CauseProcessingResult {
                new_sub_state: MmSubState::DeregisteredPlmnSearch,
                clear_guti: true,
                clear_tai_list: true,
                clear_equivalent_plmn: true,
                invalidate_usim: false,
                new_update_status: Some(UpdateStatus::RoamingNotAllowed),
            },
            MmCause::TrackingAreaNotAllowed | MmCause::NoSuitableCellsInTa => CauseProcessingResult {
                new_sub_state: MmSubState::DeregisteredLimitedService,
                clear_guti: true,
                clear_tai_list: true,
                clear_equivalent_plmn: false,
                invalidate_usim: false,
                new_update_status: Some(UpdateStatus::RoamingNotAllowed),
            },
            MmCause::N1ModeNotAllowed => CauseProcessingResult {
                new_sub_state: MmSubState::Null,
                clear_guti: true,
                clear_tai_list: true,
                clear_equivalent_plmn: false,
                invalidate_usim: false,
                new_update_status: Some(UpdateStatus::RoamingNotAllowed),
            },
            MmCause::Congestion => CauseProcessingResult {
                new_sub_state: MmSubState::DeregisteredAttemptingRegistration,
                clear_guti: false,
                clear_tai_list: false,
                clear_equivalent_plmn: false,
                invalidate_usim: false,
                new_update_status: Some(UpdateStatus::NotUpdated),
            },
            MmCause::ImplicitlyDeregistered => CauseProcessingResult {
                new_sub_state: MmSubState::DeregisteredNormalService,
                clear_guti: false,
                clear_tai_list: false,
                clear_equivalent_plmn: false,
                invalidate_usim: false,
                new_update_status: None,
            },
            // Other causes - handle as abnormal case
            _ => CauseProcessingResult {
                new_sub_state: MmSubState::DeregisteredPlmnSearch,
                clear_guti: true,
                clear_tai_list: true,
                clear_equivalent_plmn: true,
                invalidate_usim: false,
                new_update_status: Some(UpdateStatus::NotUpdated),
            },
        }
    }
}

// ============================================================================
// Result Types
// ============================================================================

/// Result of processing a network-initiated deregistration request.
#[derive(Debug, Clone)]
pub struct NetworkDeregistrationResult {
    /// Deregistration Accept message to send (None if should send MM Status instead)
    pub accept: Option<DeregistrationAcceptUeTerminated>,
    /// New MM sub-state to transition to
    pub new_sub_state: Option<MmSubState>,
    /// Whether to clear the stored 5G-GUTI
    pub clear_guti: bool,
    /// Whether to clear the TAI list
    pub clear_tai_list: bool,
    /// Whether to clear the equivalent PLMN list
    pub clear_equivalent_plmn: bool,
    /// Whether to invalidate the USIM
    pub invalidate_usim: bool,
    /// New update status (if changed)
    pub new_update_status: Option<UpdateStatus>,
    /// Whether re-registration is required
    pub re_registration_required: bool,
}

/// Internal result of processing a 5GMM cause.
struct CauseProcessingResult {
    new_sub_state: MmSubState,
    clear_guti: bool,
    clear_tai_list: bool,
    clear_equivalent_plmn: bool,
    invalidate_usim: bool,
    new_update_status: Option<UpdateStatus>,
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use nextgsim_nas::Ie5gMmCause;
    // Use the same MmCause as the main module (from messages::mm)

    #[test]
    fn test_deregistration_cause_display() {
        assert_eq!(
            format!("{}", DeregistrationCause::ExplicitDeregistration),
            "explicit-deregistration"
        );
        assert_eq!(format!("{}", DeregistrationCause::SwitchOff), "switch-off");
        assert_eq!(
            format!("{}", DeregistrationCause::UsimRemoval),
            "usim-removal"
        );
        assert_eq!(format!("{}", DeregistrationCause::Disable5g), "disable-5g");
    }

    #[test]
    fn test_initiate_deregistration_not_registered() {
        let mut proc = DeregistrationProcedure::new();
        let result = proc.initiate_deregistration(
            DeregistrationCause::ExplicitDeregistration,
            RmState::Deregistered,
            MmState::Deregistered,
            MmSubState::DeregisteredNormalService,
            None,
            Ie5gsMobileIdentity::no_identity(),
        );

        assert!(matches!(
            result,
            Err(DeregistrationProcedureError::NotRegistered(_))
        ));
    }

    #[test]
    fn test_initiate_deregistration_already_in_progress() {
        let mut proc = DeregistrationProcedure::new();
        let result = proc.initiate_deregistration(
            DeregistrationCause::ExplicitDeregistration,
            RmState::Registered,
            MmState::DeregisteredInitiated,
            MmSubState::DeregisteredInitiated,
            None,
            Ie5gsMobileIdentity::no_identity(),
        );

        assert!(matches!(
            result,
            Err(DeregistrationProcedureError::AlreadyInProgress)
        ));
    }

    #[test]
    fn test_initiate_deregistration_success() {
        let mut proc = DeregistrationProcedure::new();
        let result = proc.initiate_deregistration(
            DeregistrationCause::ExplicitDeregistration,
            RmState::Registered,
            MmState::Registered,
            MmSubState::RegisteredNormalService,
            None,
            Ie5gsMobileIdentity::no_identity(),
        );

        assert!(result.is_ok());
        let (request, new_state) = result.unwrap();
        assert_eq!(new_state, MmSubState::DeregisteredInitiated);
        assert_eq!(
            request.deregistration_type.switch_off,
            SwitchOff::NormalDeRegistration
        );
        assert!(proc.t3521.is_running());
    }

    #[test]
    fn test_initiate_deregistration_switch_off() {
        let mut proc = DeregistrationProcedure::new();
        let result = proc.initiate_deregistration(
            DeregistrationCause::SwitchOff,
            RmState::Registered,
            MmState::Registered,
            MmSubState::RegisteredNormalService,
            None,
            Ie5gsMobileIdentity::no_identity(),
        );

        assert!(result.is_ok());
        let (request, _) = result.unwrap();
        assert_eq!(request.deregistration_type.switch_off, SwitchOff::SwitchOff);
        // T3521 should NOT be started for switch-off
        assert!(!proc.t3521.is_running());
    }

    #[test]
    fn test_receive_deregistration_accept() {
        let mut proc = DeregistrationProcedure::new();

        // First initiate deregistration
        let _ = proc.initiate_deregistration(
            DeregistrationCause::ExplicitDeregistration,
            RmState::Registered,
            MmState::Registered,
            MmSubState::RegisteredNormalService,
            None,
            Ie5gsMobileIdentity::no_identity(),
        );

        // Now receive accept
        let accept = DeregistrationAcceptUeOriginating::new();
        let result = proc.receive_deregistration_accept(&accept, MmState::DeregisteredInitiated);

        assert!(result.is_ok());
        let new_state = result.unwrap();
        assert_eq!(new_state, MmSubState::DeregisteredNormalService);
        assert!(!proc.t3521.is_running());
    }

    #[test]
    fn test_receive_deregistration_accept_wrong_state() {
        let mut proc = DeregistrationProcedure::new();
        let accept = DeregistrationAcceptUeOriginating::new();
        let result = proc.receive_deregistration_accept(&accept, MmState::Registered);

        assert!(matches!(
            result,
            Err(DeregistrationProcedureError::InvalidState(_))
        ));
    }

    #[test]
    fn test_handle_t3521_expiry_retransmit() {
        let mut proc = DeregistrationProcedure::new();

        // Initiate deregistration
        let _ = proc.initiate_deregistration(
            DeregistrationCause::ExplicitDeregistration,
            RmState::Registered,
            MmState::Registered,
            MmSubState::RegisteredNormalService,
            None,
            Ie5gsMobileIdentity::no_identity(),
        );

        // Simulate timer expiry
        let result = proc.handle_t3521_expiry();
        assert!(result.is_some());
    }

    #[test]
    fn test_receive_network_deregistration() {
        let mut proc = DeregistrationProcedure::new();

        let dereg_type = IeDeRegistrationType::new(
            DeRegistrationAccessType::ThreeGppAccess,
            ReRegistrationRequired::NotRequired,
            SwitchOff::NormalDeRegistration,
        );
        let mut msg = DeregistrationRequestUeTerminated::new(dereg_type);
        msg.mm_cause = Some(Ie5gMmCause::new(MmCause::ImplicitlyDeregistered));

        let result =
            proc.receive_deregistration_request(&msg, RmState::Registered, MmState::Registered);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.accept.is_some());
        assert_eq!(
            result.new_sub_state,
            Some(MmSubState::DeregisteredNormalService)
        );
        assert!(!result.re_registration_required);
    }

    #[test]
    fn test_receive_network_deregistration_re_registration_required() {
        let mut proc = DeregistrationProcedure::new();

        let dereg_type = IeDeRegistrationType::new(
            DeRegistrationAccessType::ThreeGppAccess,
            ReRegistrationRequired::Required,
            SwitchOff::NormalDeRegistration,
        );
        let msg = DeregistrationRequestUeTerminated::new(dereg_type);

        let result =
            proc.receive_deregistration_request(&msg, RmState::Registered, MmState::Registered);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.accept.is_some());
        assert!(result.re_registration_required);
    }

    #[test]
    fn test_local_deregistration() {
        let mut proc = DeregistrationProcedure::new();

        // Initiate deregistration first
        let _ = proc.initiate_deregistration(
            DeregistrationCause::ExplicitDeregistration,
            RmState::Registered,
            MmState::Registered,
            MmSubState::RegisteredNormalService,
            None,
            Ie5gsMobileIdentity::no_identity(),
        );

        assert!(proc.t3521.is_running());

        // Perform local deregistration
        let new_state = proc.perform_local_deregistration();

        assert_eq!(new_state, MmSubState::DeregisteredNormalService);
        assert!(!proc.t3521.is_running());
        assert!(proc.last_request.is_none());
    }

    #[test]
    fn test_make_deregistration_type() {
        let normal = DeregistrationProcedure::make_deregistration_type(
            DeregistrationCause::ExplicitDeregistration,
        );
        assert_eq!(normal.switch_off, SwitchOff::NormalDeRegistration);

        let switch_off =
            DeregistrationProcedure::make_deregistration_type(DeregistrationCause::SwitchOff);
        assert_eq!(switch_off.switch_off, SwitchOff::SwitchOff);

        let usim_removal =
            DeregistrationProcedure::make_deregistration_type(DeregistrationCause::UsimRemoval);
        assert_eq!(usim_removal.switch_off, SwitchOff::SwitchOff);
    }
}
