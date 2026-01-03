//! MM State Machine
//!
//! This module defines the 5GMM state machine states as per 3GPP TS 24.501 Section 5.1.3.
//!
//! # RM States (Registration Management)
//!
//! - RM-DEREGISTERED: UE is not registered with the network
//! - RM-REGISTERED: UE is registered with the network
//!
//! # CM States (Connection Management)
//!
//! - CM-IDLE: No NAS signalling connection
//! - CM-CONNECTED: NAS signalling connection established
//!
//! # MM States and Sub-states
//!
//! The MM state machine has main states and sub-states that track the UE's
//! mobility management status.
//!
//! # State Machine Manager
//!
//! The `MmStateMachine` struct manages state transitions and provides callbacks
//! for state change events.

use std::fmt;

/// Registration Management (RM) state.
///
/// 3GPP TS 24.501 Section 5.1.3.1
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RmState {
    /// RM-DEREGISTERED: UE is not registered with the network
    #[default]
    Deregistered,
    /// RM-REGISTERED: UE is registered with the network
    Registered,
}

impl fmt::Display for RmState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RmState::Deregistered => write!(f, "RM-DEREGISTERED"),
            RmState::Registered => write!(f, "RM-REGISTERED"),
        }
    }
}

/// Connection Management (CM) state.
///
/// 3GPP TS 24.501 Section 5.1.3.4
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CmState {
    /// CM-IDLE: No NAS signalling connection
    #[default]
    Idle,
    /// CM-CONNECTED: NAS signalling connection established
    Connected,
}

impl fmt::Display for CmState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CmState::Idle => write!(f, "CM-IDLE"),
            CmState::Connected => write!(f, "CM-CONNECTED"),
        }
    }
}

/// Main MM state.
///
/// 3GPP TS 24.501 Section 5.1.3.2
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MmState {
    /// 5GMM-NULL
    #[default]
    Null,
    /// 5GMM-DEREGISTERED
    Deregistered,
    /// 5GMM-REGISTERED-INITIATED
    RegisteredInitiated,
    /// 5GMM-REGISTERED
    Registered,
    /// 5GMM-DEREGISTERED-INITIATED
    DeregisteredInitiated,
    /// 5GMM-SERVICE-REQUEST-INITIATED
    ServiceRequestInitiated,
}

impl fmt::Display for MmState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MmState::Null => write!(f, "5GMM-NULL"),
            MmState::Deregistered => write!(f, "5GMM-DEREGISTERED"),
            MmState::RegisteredInitiated => write!(f, "5GMM-REGISTERED-INITIATED"),
            MmState::Registered => write!(f, "5GMM-REGISTERED"),
            MmState::DeregisteredInitiated => write!(f, "5GMM-DEREGISTERED-INITIATED"),
            MmState::ServiceRequestInitiated => write!(f, "5GMM-SERVICE-REQUEST-INITIATED"),
        }
    }
}

/// MM sub-state for detailed state tracking.
///
/// 3GPP TS 24.501 Section 5.1.3.2
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MmSubState {
    /// 5GMM-NULL
    Null,
    /// 5GMM-DEREGISTERED (primary substate, used for substate selection)
    #[default]
    Deregistered,
    /// 5GMM-DEREGISTERED.NORMAL-SERVICE
    DeregisteredNormalService,
    /// 5GMM-DEREGISTERED.LIMITED-SERVICE
    DeregisteredLimitedService,
    /// 5GMM-DEREGISTERED.ATTEMPTING-REGISTRATION
    DeregisteredAttemptingRegistration,
    /// 5GMM-DEREGISTERED.PLMN-SEARCH
    DeregisteredPlmnSearch,
    /// 5GMM-DEREGISTERED.NO-SUPI
    DeregisteredNoSupi,
    /// 5GMM-DEREGISTERED.NO-CELL-AVAILABLE
    DeregisteredNoCellAvailable,
    /// 5GMM-DEREGISTERED.ECALL-INACTIVE
    DeregisteredEcallInactive,
    /// 5GMM-DEREGISTERED.INITIAL-REGISTRATION-NEEDED
    DeregisteredInitialRegistrationNeeded,
    /// 5GMM-REGISTERED-INITIATED
    RegisteredInitiated,
    /// 5GMM-REGISTERED (primary substate, used for substate selection)
    Registered,
    /// 5GMM-REGISTERED.NORMAL-SERVICE
    RegisteredNormalService,
    /// 5GMM-REGISTERED.NON-ALLOWED-SERVICE
    RegisteredNonAllowedService,
    /// 5GMM-REGISTERED.ATTEMPTING-REGISTRATION-UPDATE
    RegisteredAttemptingRegistrationUpdate,
    /// 5GMM-REGISTERED.LIMITED-SERVICE
    RegisteredLimitedService,
    /// 5GMM-REGISTERED.PLMN-SEARCH
    RegisteredPlmnSearch,
    /// 5GMM-REGISTERED.NO-CELL-AVAILABLE
    RegisteredNoCellAvailable,
    /// 5GMM-REGISTERED.UPDATE-NEEDED
    RegisteredUpdateNeeded,
    /// 5GMM-DEREGISTERED-INITIATED
    DeregisteredInitiated,
    /// 5GMM-SERVICE-REQUEST-INITIATED
    ServiceRequestInitiated,
}

impl MmSubState {
    /// Returns the main MM state corresponding to this sub-state.
    pub fn main_state(&self) -> MmState {
        match self {
            MmSubState::Null => MmState::Null,
            MmSubState::Deregistered
            | MmSubState::DeregisteredNormalService
            | MmSubState::DeregisteredLimitedService
            | MmSubState::DeregisteredAttemptingRegistration
            | MmSubState::DeregisteredPlmnSearch
            | MmSubState::DeregisteredNoSupi
            | MmSubState::DeregisteredNoCellAvailable
            | MmSubState::DeregisteredEcallInactive
            | MmSubState::DeregisteredInitialRegistrationNeeded => MmState::Deregistered,
            MmSubState::RegisteredInitiated => MmState::RegisteredInitiated,
            MmSubState::Registered
            | MmSubState::RegisteredNormalService
            | MmSubState::RegisteredNonAllowedService
            | MmSubState::RegisteredAttemptingRegistrationUpdate
            | MmSubState::RegisteredLimitedService
            | MmSubState::RegisteredPlmnSearch
            | MmSubState::RegisteredNoCellAvailable
            | MmSubState::RegisteredUpdateNeeded => MmState::Registered,
            MmSubState::DeregisteredInitiated => MmState::DeregisteredInitiated,
            MmSubState::ServiceRequestInitiated => MmState::ServiceRequestInitiated,
        }
    }

    /// Returns true if this is a primary substate (used for substate selection).
    pub fn is_primary(&self) -> bool {
        matches!(self, MmSubState::Deregistered | MmSubState::Registered)
    }
}

impl fmt::Display for MmSubState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MmSubState::Null => write!(f, "5GMM-NULL"),
            MmSubState::Deregistered => write!(f, "5GMM-DEREGISTERED"),
            MmSubState::DeregisteredNormalService => {
                write!(f, "5GMM-DEREGISTERED.NORMAL-SERVICE")
            }
            MmSubState::DeregisteredLimitedService => {
                write!(f, "5GMM-DEREGISTERED.LIMITED-SERVICE")
            }
            MmSubState::DeregisteredAttemptingRegistration => {
                write!(f, "5GMM-DEREGISTERED.ATTEMPTING-REGISTRATION")
            }
            MmSubState::DeregisteredPlmnSearch => write!(f, "5GMM-DEREGISTERED.PLMN-SEARCH"),
            MmSubState::DeregisteredNoSupi => write!(f, "5GMM-DEREGISTERED.NO-SUPI"),
            MmSubState::DeregisteredNoCellAvailable => {
                write!(f, "5GMM-DEREGISTERED.NO-CELL-AVAILABLE")
            }
            MmSubState::DeregisteredEcallInactive => {
                write!(f, "5GMM-DEREGISTERED.ECALL-INACTIVE")
            }
            MmSubState::DeregisteredInitialRegistrationNeeded => {
                write!(f, "5GMM-DEREGISTERED.INITIAL-REGISTRATION-NEEDED")
            }
            MmSubState::RegisteredInitiated => write!(f, "5GMM-REGISTERED-INITIATED"),
            MmSubState::Registered => write!(f, "5GMM-REGISTERED"),
            MmSubState::RegisteredNormalService => write!(f, "5GMM-REGISTERED.NORMAL-SERVICE"),
            MmSubState::RegisteredNonAllowedService => {
                write!(f, "5GMM-REGISTERED.NON-ALLOWED-SERVICE")
            }
            MmSubState::RegisteredAttemptingRegistrationUpdate => {
                write!(f, "5GMM-REGISTERED.ATTEMPTING-REGISTRATION-UPDATE")
            }
            MmSubState::RegisteredLimitedService => {
                write!(f, "5GMM-REGISTERED.LIMITED-SERVICE")
            }
            MmSubState::RegisteredPlmnSearch => write!(f, "5GMM-REGISTERED.PLMN-SEARCH"),
            MmSubState::RegisteredNoCellAvailable => {
                write!(f, "5GMM-REGISTERED.NO-CELL-AVAILABLE")
            }
            MmSubState::RegisteredUpdateNeeded => write!(f, "5GMM-REGISTERED.UPDATE-NEEDED"),
            MmSubState::DeregisteredInitiated => write!(f, "5GMM-DEREGISTERED-INITIATED"),
            MmSubState::ServiceRequestInitiated => write!(f, "5GMM-SERVICE-REQUEST-INITIATED"),
        }
    }
}

/// 5GS Update Status (U-state).
///
/// 3GPP TS 24.501 Section 5.1.3.3
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UpdateStatus {
    /// U1 UPDATED
    #[default]
    Updated,
    /// U2 NOT UPDATED
    NotUpdated,
    /// U3 ROAMING NOT ALLOWED
    RoamingNotAllowed,
}

impl fmt::Display for UpdateStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UpdateStatus::Updated => write!(f, "U1-UPDATED"),
            UpdateStatus::NotUpdated => write!(f, "U2-NOT-UPDATED"),
            UpdateStatus::RoamingNotAllowed => write!(f, "U3-ROAMING-NOT-ALLOWED"),
        }
    }
}

/// State transition event for MM state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MmStateTransition {
    /// Previous MM state
    pub old_state: MmState,
    /// New MM state
    pub new_state: MmState,
    /// Previous MM sub-state
    pub old_substate: MmSubState,
    /// New MM sub-state
    pub new_substate: MmSubState,
}

/// State transition event for RM state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RmStateTransition {
    /// Previous RM state
    pub old_state: RmState,
    /// New RM state
    pub new_state: RmState,
}

/// State transition event for CM state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CmStateTransition {
    /// Previous CM state
    pub old_state: CmState,
    /// New CM state
    pub new_state: CmState,
}

/// State transition event for Update Status (U-state).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UpdateStatusTransition {
    /// Previous U-state
    pub old_state: UpdateStatus,
    /// New U-state
    pub new_state: UpdateStatus,
}

/// MM State Machine Manager.
///
/// Manages the 5GMM state machine including RM, CM, MM, and U-states.
/// Provides state transition methods and tracks state change timestamps.
///
/// # Example
///
/// ```
/// use nextgsim_ue::nas::mm::state::{MmStateMachine, MmSubState, CmState, UpdateStatus};
///
/// let mut sm = MmStateMachine::new();
///
/// // Initial state
/// assert!(sm.is_deregistered());
/// assert!(sm.is_idle());
///
/// // Transition to normal service
/// let transition = sm.switch_mm_state(MmSubState::DeregisteredNormalService);
/// assert!(transition.is_some());
///
/// // Transition to CM-CONNECTED
/// let cm_transition = sm.switch_cm_state(CmState::Connected);
/// assert!(cm_transition.is_some());
/// ```
#[derive(Debug, Clone)]
pub struct MmStateMachine {
    /// Registration Management state
    rm_state: RmState,
    /// Connection Management state
    cm_state: CmState,
    /// Main MM state
    mm_state: MmState,
    /// MM sub-state
    mm_substate: MmSubState,
    /// 5GS Update Status
    update_status: UpdateStatus,
}

impl Default for MmStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

impl MmStateMachine {
    /// Creates a new MM state machine with default initial states.
    ///
    /// Initial states:
    /// - RM: RM-DEREGISTERED
    /// - CM: CM-IDLE
    /// - MM: 5GMM-DEREGISTERED (primary substate)
    /// - U-state: U1-UPDATED
    pub fn new() -> Self {
        Self {
            rm_state: RmState::Deregistered,
            cm_state: CmState::Idle,
            mm_state: MmState::Deregistered,
            mm_substate: MmSubState::Deregistered,
            update_status: UpdateStatus::Updated,
        }
    }

    // ========== State Getters ==========

    /// Returns the current RM state.
    pub fn rm_state(&self) -> RmState {
        self.rm_state
    }

    /// Returns the current CM state.
    pub fn cm_state(&self) -> CmState {
        self.cm_state
    }

    /// Returns the current MM state.
    pub fn mm_state(&self) -> MmState {
        self.mm_state
    }

    /// Returns the current MM sub-state.
    pub fn mm_substate(&self) -> MmSubState {
        self.mm_substate
    }

    /// Returns the current Update Status (U-state).
    pub fn update_status(&self) -> UpdateStatus {
        self.update_status
    }

    // ========== State Predicates ==========

    /// Returns true if UE is in RM-DEREGISTERED state.
    pub fn is_deregistered(&self) -> bool {
        self.rm_state == RmState::Deregistered
    }

    /// Returns true if UE is in RM-REGISTERED state.
    pub fn is_registered(&self) -> bool {
        self.rm_state == RmState::Registered
    }

    /// Returns true if UE is in CM-IDLE state.
    pub fn is_idle(&self) -> bool {
        self.cm_state == CmState::Idle
    }

    /// Returns true if UE is in CM-CONNECTED state.
    pub fn is_connected(&self) -> bool {
        self.cm_state == CmState::Connected
    }

    /// Returns true if UE is in 5GMM-NULL state.
    pub fn is_null(&self) -> bool {
        self.mm_state == MmState::Null
    }

    /// Returns true if UE is in a PLMN search sub-state.
    pub fn is_plmn_search(&self) -> bool {
        matches!(
            self.mm_substate,
            MmSubState::DeregisteredPlmnSearch | MmSubState::RegisteredPlmnSearch
        )
    }

    /// Returns true if UE is in a no-cell-available sub-state.
    pub fn is_no_cell_available(&self) -> bool {
        matches!(
            self.mm_substate,
            MmSubState::DeregisteredNoCellAvailable | MmSubState::RegisteredNoCellAvailable
        )
    }

    /// Returns true if UE is in normal service sub-state.
    pub fn is_normal_service(&self) -> bool {
        matches!(
            self.mm_substate,
            MmSubState::DeregisteredNormalService | MmSubState::RegisteredNormalService
        )
    }

    /// Returns true if UE is in limited service sub-state.
    pub fn is_limited_service(&self) -> bool {
        matches!(
            self.mm_substate,
            MmSubState::DeregisteredLimitedService | MmSubState::RegisteredLimitedService
        )
    }

    // ========== State Transitions ==========

    /// Switches the MM state to the specified sub-state.
    ///
    /// This method:
    /// 1. Derives the main MM state from the sub-state
    /// 2. Updates the RM state based on the new MM state
    /// 3. Returns the state transition if any state changed
    ///
    /// # Arguments
    ///
    /// * `substate` - The new MM sub-state
    ///
    /// # Returns
    ///
    /// Returns `Some(MmStateTransition)` if the state changed, `None` otherwise.
    pub fn switch_mm_state(&mut self, substate: MmSubState) -> Option<MmStateTransition> {
        let new_state = substate.main_state();

        // Update RM state based on MM state
        let new_rm_state = Self::derive_rm_state(new_state);
        if new_rm_state != self.rm_state {
            self.rm_state = new_rm_state;
        }

        // Check if state actually changed
        if self.mm_state == new_state && self.mm_substate == substate {
            return None;
        }

        let transition = MmStateTransition {
            old_state: self.mm_state,
            new_state,
            old_substate: self.mm_substate,
            new_substate: substate,
        };

        self.mm_state = new_state;
        self.mm_substate = substate;

        Some(transition)
    }

    /// Switches the CM state.
    ///
    /// # Arguments
    ///
    /// * `state` - The new CM state
    ///
    /// # Returns
    ///
    /// Returns `Some(CmStateTransition)` if the state changed, `None` otherwise.
    pub fn switch_cm_state(&mut self, state: CmState) -> Option<CmStateTransition> {
        if self.cm_state == state {
            return None;
        }

        let transition = CmStateTransition {
            old_state: self.cm_state,
            new_state: state,
        };

        self.cm_state = state;

        Some(transition)
    }

    /// Switches the Update Status (U-state).
    ///
    /// # Arguments
    ///
    /// * `status` - The new Update Status
    ///
    /// # Returns
    ///
    /// Returns `Some(UpdateStatusTransition)` if the state changed, `None` otherwise.
    pub fn switch_update_status(&mut self, status: UpdateStatus) -> Option<UpdateStatusTransition> {
        if self.update_status == status {
            return None;
        }

        let transition = UpdateStatusTransition {
            old_state: self.update_status,
            new_state: status,
        };

        self.update_status = status;

        Some(transition)
    }

    // ========== Helper Methods ==========

    /// Derives the RM state from the MM state.
    ///
    /// Per 3GPP TS 24.501:
    /// - RM-DEREGISTERED: MM-DEREGISTERED, MM-REGISTERED-INITIATED
    /// - RM-REGISTERED: MM-REGISTERED, MM-SERVICE-REQUEST-INITIATED, MM-DEREGISTERED-INITIATED
    fn derive_rm_state(mm_state: MmState) -> RmState {
        match mm_state {
            MmState::Null | MmState::Deregistered | MmState::RegisteredInitiated => {
                RmState::Deregistered
            }
            MmState::Registered
            | MmState::ServiceRequestInitiated
            | MmState::DeregisteredInitiated => RmState::Registered,
        }
    }

    /// Resets the state machine to initial state (after SIM removal or switch off).
    pub fn reset(&mut self) {
        self.rm_state = RmState::Deregistered;
        self.cm_state = CmState::Idle;
        self.mm_state = MmState::Deregistered;
        self.mm_substate = MmSubState::Deregistered;
        self.update_status = UpdateStatus::Updated;
    }

    /// Transitions to NULL state (for disabling 5GS services).
    pub fn switch_to_null(&mut self) -> Option<MmStateTransition> {
        self.switch_mm_state(MmSubState::Null)
    }

    /// Transitions to primary DEREGISTERED state.
    pub fn switch_to_deregistered(&mut self) -> Option<MmStateTransition> {
        self.switch_mm_state(MmSubState::Deregistered)
    }

    /// Transitions to primary REGISTERED state.
    pub fn switch_to_registered(&mut self) -> Option<MmStateTransition> {
        self.switch_mm_state(MmSubState::Registered)
    }

    /// Validates if a state transition is allowed.
    ///
    /// This method checks if the transition from the current state to the
    /// target state is valid according to 3GPP TS 24.501.
    ///
    /// # Arguments
    ///
    /// * `target` - The target MM sub-state
    ///
    /// # Returns
    ///
    /// Returns `true` if the transition is valid, `false` otherwise.
    pub fn is_valid_transition(&self, target: MmSubState) -> bool {
        // NULL state can only transition to DEREGISTERED
        if self.mm_state == MmState::Null && target.main_state() != MmState::Deregistered {
            return false;
        }

        // Cannot transition to NULL from REGISTERED states (except via local deregistration)
        if target == MmSubState::Null
            && matches!(
                self.mm_state,
                MmState::Registered | MmState::ServiceRequestInitiated
            )
        {
            return false;
        }

        true
    }
}

impl fmt::Display for MmStateMachine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MM[{}, {}, {}, {}]",
            self.rm_state, self.cm_state, self.mm_substate, self.update_status
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rm_state_display() {
        assert_eq!(format!("{}", RmState::Deregistered), "RM-DEREGISTERED");
        assert_eq!(format!("{}", RmState::Registered), "RM-REGISTERED");
    }

    #[test]
    fn test_cm_state_display() {
        assert_eq!(format!("{}", CmState::Idle), "CM-IDLE");
        assert_eq!(format!("{}", CmState::Connected), "CM-CONNECTED");
    }

    #[test]
    fn test_mm_state_display() {
        assert_eq!(format!("{}", MmState::Null), "5GMM-NULL");
        assert_eq!(format!("{}", MmState::Deregistered), "5GMM-DEREGISTERED");
        assert_eq!(
            format!("{}", MmState::DeregisteredInitiated),
            "5GMM-DEREGISTERED-INITIATED"
        );
    }

    #[test]
    fn test_mm_substate_main_state() {
        assert_eq!(MmSubState::Null.main_state(), MmState::Null);
        assert_eq!(
            MmSubState::DeregisteredNormalService.main_state(),
            MmState::Deregistered
        );
        assert_eq!(
            MmSubState::RegisteredNormalService.main_state(),
            MmState::Registered
        );
        assert_eq!(
            MmSubState::DeregisteredInitiated.main_state(),
            MmState::DeregisteredInitiated
        );
    }

    #[test]
    fn test_update_status_display() {
        assert_eq!(format!("{}", UpdateStatus::Updated), "U1-UPDATED");
        assert_eq!(format!("{}", UpdateStatus::NotUpdated), "U2-NOT-UPDATED");
        assert_eq!(
            format!("{}", UpdateStatus::RoamingNotAllowed),
            "U3-ROAMING-NOT-ALLOWED"
        );
    }

    #[test]
    fn test_state_machine_initial_state() {
        let sm = MmStateMachine::new();
        assert_eq!(sm.rm_state(), RmState::Deregistered);
        assert_eq!(sm.cm_state(), CmState::Idle);
        assert_eq!(sm.mm_state(), MmState::Deregistered);
        assert_eq!(sm.mm_substate(), MmSubState::Deregistered);
        assert_eq!(sm.update_status(), UpdateStatus::Updated);
    }

    #[test]
    fn test_state_machine_predicates() {
        let sm = MmStateMachine::new();
        assert!(sm.is_deregistered());
        assert!(!sm.is_registered());
        assert!(sm.is_idle());
        assert!(!sm.is_connected());
        assert!(!sm.is_null());
    }

    #[test]
    fn test_mm_state_transition() {
        let mut sm = MmStateMachine::new();

        // Transition to normal service
        let transition = sm.switch_mm_state(MmSubState::DeregisteredNormalService);
        assert!(transition.is_some());
        let t = transition.unwrap();
        assert_eq!(t.old_substate, MmSubState::Deregistered);
        assert_eq!(t.new_substate, MmSubState::DeregisteredNormalService);
        assert_eq!(sm.mm_substate(), MmSubState::DeregisteredNormalService);

        // Same state transition returns None
        let transition = sm.switch_mm_state(MmSubState::DeregisteredNormalService);
        assert!(transition.is_none());
    }

    #[test]
    fn test_rm_state_derivation() {
        let mut sm = MmStateMachine::new();

        // DEREGISTERED states -> RM-DEREGISTERED
        sm.switch_mm_state(MmSubState::DeregisteredNormalService);
        assert_eq!(sm.rm_state(), RmState::Deregistered);

        // REGISTERED-INITIATED -> RM-DEREGISTERED
        sm.switch_mm_state(MmSubState::RegisteredInitiated);
        assert_eq!(sm.rm_state(), RmState::Deregistered);

        // REGISTERED states -> RM-REGISTERED
        sm.switch_mm_state(MmSubState::RegisteredNormalService);
        assert_eq!(sm.rm_state(), RmState::Registered);

        // SERVICE-REQUEST-INITIATED -> RM-REGISTERED
        sm.switch_mm_state(MmSubState::ServiceRequestInitiated);
        assert_eq!(sm.rm_state(), RmState::Registered);

        // DEREGISTERED-INITIATED -> RM-REGISTERED
        sm.switch_mm_state(MmSubState::DeregisteredInitiated);
        assert_eq!(sm.rm_state(), RmState::Registered);
    }

    #[test]
    fn test_cm_state_transition() {
        let mut sm = MmStateMachine::new();

        // Transition to connected
        let transition = sm.switch_cm_state(CmState::Connected);
        assert!(transition.is_some());
        let t = transition.unwrap();
        assert_eq!(t.old_state, CmState::Idle);
        assert_eq!(t.new_state, CmState::Connected);
        assert!(sm.is_connected());

        // Same state transition returns None
        let transition = sm.switch_cm_state(CmState::Connected);
        assert!(transition.is_none());

        // Transition back to idle
        let transition = sm.switch_cm_state(CmState::Idle);
        assert!(transition.is_some());
        assert!(sm.is_idle());
    }

    #[test]
    fn test_update_status_transition() {
        let mut sm = MmStateMachine::new();

        // Transition to NOT UPDATED
        let transition = sm.switch_update_status(UpdateStatus::NotUpdated);
        assert!(transition.is_some());
        let t = transition.unwrap();
        assert_eq!(t.old_state, UpdateStatus::Updated);
        assert_eq!(t.new_state, UpdateStatus::NotUpdated);
        assert_eq!(sm.update_status(), UpdateStatus::NotUpdated);

        // Same state transition returns None
        let transition = sm.switch_update_status(UpdateStatus::NotUpdated);
        assert!(transition.is_none());
    }

    #[test]
    fn test_state_machine_reset() {
        let mut sm = MmStateMachine::new();

        // Change states
        sm.switch_mm_state(MmSubState::RegisteredNormalService);
        sm.switch_cm_state(CmState::Connected);
        sm.switch_update_status(UpdateStatus::NotUpdated);

        // Reset
        sm.reset();

        // Verify initial state
        assert_eq!(sm.rm_state(), RmState::Deregistered);
        assert_eq!(sm.cm_state(), CmState::Idle);
        assert_eq!(sm.mm_state(), MmState::Deregistered);
        assert_eq!(sm.mm_substate(), MmSubState::Deregistered);
        assert_eq!(sm.update_status(), UpdateStatus::Updated);
    }

    #[test]
    fn test_state_machine_display() {
        let sm = MmStateMachine::new();
        let display = format!("{}", sm);
        assert!(display.contains("RM-DEREGISTERED"));
        assert!(display.contains("CM-IDLE"));
        assert!(display.contains("5GMM-DEREGISTERED"));
        assert!(display.contains("U1-UPDATED"));
    }

    #[test]
    fn test_plmn_search_predicate() {
        let mut sm = MmStateMachine::new();

        assert!(!sm.is_plmn_search());

        sm.switch_mm_state(MmSubState::DeregisteredPlmnSearch);
        assert!(sm.is_plmn_search());

        sm.switch_mm_state(MmSubState::RegisteredPlmnSearch);
        assert!(sm.is_plmn_search());

        sm.switch_mm_state(MmSubState::RegisteredNormalService);
        assert!(!sm.is_plmn_search());
    }

    #[test]
    fn test_valid_transition() {
        let mut sm = MmStateMachine::new();

        // From DEREGISTERED, can go to REGISTERED-INITIATED
        assert!(sm.is_valid_transition(MmSubState::RegisteredInitiated));

        // From NULL, can only go to DEREGISTERED
        sm.switch_mm_state(MmSubState::Null);
        assert!(sm.is_valid_transition(MmSubState::Deregistered));
        assert!(!sm.is_valid_transition(MmSubState::RegisteredNormalService));
    }

    #[test]
    fn test_convenience_transitions() {
        let mut sm = MmStateMachine::new();

        // Switch to NULL
        let t = sm.switch_to_null();
        assert!(t.is_some());
        assert!(sm.is_null());

        // Switch to DEREGISTERED
        let t = sm.switch_to_deregistered();
        assert!(t.is_some());
        assert_eq!(sm.mm_state(), MmState::Deregistered);

        // Switch to REGISTERED (primary)
        let t = sm.switch_to_registered();
        assert!(t.is_some());
        assert_eq!(sm.mm_state(), MmState::Registered);
    }
}
