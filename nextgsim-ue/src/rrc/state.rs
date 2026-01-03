//! RRC State Machine
//!
//! This module implements the UE RRC state machine as defined in 3GPP TS 38.331.
//!
//! # RRC States
//!
//! - **RRC_IDLE**: UE is not connected to the network. The UE performs cell selection/reselection
//!   and monitors paging. No dedicated radio resources are allocated.
//!
//! - **RRC_CONNECTED**: UE has an active RRC connection with the network. The UE can send and
//!   receive data, and has dedicated radio resources allocated.
//!
//! - **RRC_INACTIVE**: UE has suspended its RRC connection but maintains the UE context in both
//!   UE and network. This allows for faster connection resumption compared to RRC_IDLE.
//!
//! # State Transitions
//!
//! | From State | To State | Trigger |
//! |------------|----------|---------|
//! | Idle | Connected | RRC Setup Complete |
//! | Connected | Idle | RRC Release |
//! | Connected | Inactive | RRC Suspend |
//! | Inactive | Connected | RRC Resume |
//! | Inactive | Idle | RRC Release |

use std::fmt;

/// RRC state as defined in 3GPP TS 38.331.
///
/// The UE can be in one of three RRC states:
/// - Idle: No RRC connection, UE performs cell selection
/// - Connected: Active RRC connection with dedicated resources
/// - Inactive: Suspended RRC connection with maintained context
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum RrcState {
    /// RRC_IDLE: No RRC connection established.
    ///
    /// In this state, the UE:
    /// - Performs cell selection and reselection
    /// - Monitors paging channel for incoming calls/data
    /// - Has no dedicated radio resources
    /// - Must establish RRC connection to send/receive data
    #[default]
    Idle,

    /// RRC_CONNECTED: Active RRC connection.
    ///
    /// In this state, the UE:
    /// - Has an established RRC connection with the gNB
    /// - Can send and receive user data
    /// - Has dedicated radio resources allocated
    /// - Performs measurements as configured by the network
    Connected,

    /// RRC_INACTIVE: Suspended RRC connection.
    ///
    /// In this state, the UE:
    /// - Has suspended its RRC connection
    /// - Maintains UE context (AS security context, UE capabilities)
    /// - Performs cell selection/reselection like in Idle
    /// - Can resume connection faster than from Idle state
    Inactive,
}

impl RrcState {
    /// Returns true if the UE is in RRC_IDLE state.
    pub fn is_idle(&self) -> bool {
        matches!(self, RrcState::Idle)
    }

    /// Returns true if the UE is in RRC_CONNECTED state.
    pub fn is_connected(&self) -> bool {
        matches!(self, RrcState::Connected)
    }

    /// Returns true if the UE is in RRC_INACTIVE state.
    pub fn is_inactive(&self) -> bool {
        matches!(self, RrcState::Inactive)
    }

    /// Returns true if the UE should perform cell selection/reselection.
    ///
    /// Cell selection is performed in both Idle and Inactive states.
    pub fn should_perform_cell_selection(&self) -> bool {
        matches!(self, RrcState::Idle | RrcState::Inactive)
    }

    /// Returns true if the UE has an active or suspended connection.
    ///
    /// This is useful for determining if the UE has context that can be reused.
    pub fn has_connection_context(&self) -> bool {
        matches!(self, RrcState::Connected | RrcState::Inactive)
    }
}

impl fmt::Display for RrcState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RrcState::Idle => write!(f, "RRC_IDLE"),
            RrcState::Connected => write!(f, "RRC_CONNECTED"),
            RrcState::Inactive => write!(f, "RRC_INACTIVE"),
        }
    }
}

/// RRC state transition types.
///
/// These represent the valid transitions between RRC states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RrcStateTransition {
    /// Transition from Idle to Connected via RRC Setup
    SetupComplete,
    /// Transition from Connected to Idle via RRC Release
    Release,
    /// Transition from Connected to Inactive via RRC Suspend
    Suspend,
    /// Transition from Inactive to Connected via RRC Resume
    Resume,
    /// Transition from Inactive to Idle via RRC Release
    ReleaseFromInactive,
    /// Radio Link Failure - transitions to Idle
    RadioLinkFailure,
}

impl fmt::Display for RrcStateTransition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RrcStateTransition::SetupComplete => write!(f, "RRC Setup Complete"),
            RrcStateTransition::Release => write!(f, "RRC Release"),
            RrcStateTransition::Suspend => write!(f, "RRC Suspend"),
            RrcStateTransition::Resume => write!(f, "RRC Resume"),
            RrcStateTransition::ReleaseFromInactive => write!(f, "RRC Release (from Inactive)"),
            RrcStateTransition::RadioLinkFailure => write!(f, "Radio Link Failure"),
        }
    }
}

/// Error type for invalid RRC state transitions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RrcStateError {
    /// The current state when the invalid transition was attempted
    pub current_state: RrcState,
    /// The transition that was attempted
    pub attempted_transition: RrcStateTransition,
    /// Human-readable error message
    pub message: String,
}

impl fmt::Display for RrcStateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Invalid RRC state transition: {} from state {}. {}",
            self.attempted_transition, self.current_state, self.message
        )
    }
}

impl std::error::Error for RrcStateError {}

/// RRC state machine for the UE.
///
/// Manages RRC state transitions and provides callbacks for state changes.
///
/// # Example
///
/// ```
/// use nextgsim_ue::rrc::{RrcStateMachine, RrcState, RrcStateTransition};
///
/// let mut sm = RrcStateMachine::new();
/// assert_eq!(sm.state(), RrcState::Idle);
///
/// // Establish RRC connection
/// sm.transition(RrcStateTransition::SetupComplete).unwrap();
/// assert_eq!(sm.state(), RrcState::Connected);
///
/// // Release connection
/// sm.transition(RrcStateTransition::Release).unwrap();
/// assert_eq!(sm.state(), RrcState::Idle);
/// ```
#[derive(Debug)]
pub struct RrcStateMachine {
    /// Current RRC state
    state: RrcState,
    /// Previous RRC state (for tracking transitions)
    previous_state: Option<RrcState>,
    /// Number of state transitions performed
    transition_count: u64,
}

impl RrcStateMachine {
    /// Creates a new RRC state machine in the Idle state.
    pub fn new() -> Self {
        Self {
            state: RrcState::Idle,
            previous_state: None,
            transition_count: 0,
        }
    }

    /// Returns the current RRC state.
    pub fn state(&self) -> RrcState {
        self.state
    }

    /// Returns the previous RRC state, if any.
    pub fn previous_state(&self) -> Option<RrcState> {
        self.previous_state
    }

    /// Returns the number of state transitions performed.
    pub fn transition_count(&self) -> u64 {
        self.transition_count
    }

    /// Attempts to perform a state transition.
    ///
    /// Returns `Ok(new_state)` if the transition is valid, or `Err(RrcStateError)`
    /// if the transition is not allowed from the current state.
    ///
    /// # Valid Transitions
    ///
    /// | Current State | Transition | New State |
    /// |---------------|------------|-----------|
    /// | Idle | SetupComplete | Connected |
    /// | Connected | Release | Idle |
    /// | Connected | Suspend | Inactive |
    /// | Connected | RadioLinkFailure | Idle |
    /// | Inactive | Resume | Connected |
    /// | Inactive | ReleaseFromInactive | Idle |
    /// | Inactive | RadioLinkFailure | Idle |
    pub fn transition(&mut self, transition: RrcStateTransition) -> Result<RrcState, RrcStateError> {
        let new_state = self.validate_transition(transition)?;
        
        self.previous_state = Some(self.state);
        self.state = new_state;
        self.transition_count += 1;
        
        Ok(new_state)
    }

    /// Validates a transition without performing it.
    ///
    /// Returns the new state if the transition would be valid, or an error otherwise.
    pub fn validate_transition(&self, transition: RrcStateTransition) -> Result<RrcState, RrcStateError> {
        match (self.state, transition) {
            // From Idle
            (RrcState::Idle, RrcStateTransition::SetupComplete) => Ok(RrcState::Connected),
            
            // From Connected
            (RrcState::Connected, RrcStateTransition::Release) => Ok(RrcState::Idle),
            (RrcState::Connected, RrcStateTransition::Suspend) => Ok(RrcState::Inactive),
            (RrcState::Connected, RrcStateTransition::RadioLinkFailure) => Ok(RrcState::Idle),
            
            // From Inactive
            (RrcState::Inactive, RrcStateTransition::Resume) => Ok(RrcState::Connected),
            (RrcState::Inactive, RrcStateTransition::ReleaseFromInactive) => Ok(RrcState::Idle),
            (RrcState::Inactive, RrcStateTransition::RadioLinkFailure) => Ok(RrcState::Idle),
            
            // Invalid transitions
            (state, transition) => Err(RrcStateError {
                current_state: state,
                attempted_transition: transition,
                message: format!(
                    "Transition '{}' is not valid from state '{}'",
                    transition, state
                ),
            }),
        }
    }

    /// Checks if a transition is valid from the current state.
    pub fn can_transition(&self, transition: RrcStateTransition) -> bool {
        self.validate_transition(transition).is_ok()
    }

    /// Forces the state machine to a specific state.
    ///
    /// This bypasses normal transition validation and should only be used
    /// for initialization or error recovery scenarios.
    ///
    /// # Warning
    ///
    /// Using this method may put the state machine in an inconsistent state
    /// if not used carefully.
    pub fn force_state(&mut self, state: RrcState) {
        self.previous_state = Some(self.state);
        self.state = state;
        self.transition_count += 1;
    }

    /// Resets the state machine to the initial Idle state.
    ///
    /// This clears the previous state and resets the transition count.
    pub fn reset(&mut self) {
        self.state = RrcState::Idle;
        self.previous_state = None;
        self.transition_count = 0;
    }
}

impl Default for RrcStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for RrcStateMachine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RrcStateMachine(state={})", self.state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrc_state_default() {
        let state = RrcState::default();
        assert_eq!(state, RrcState::Idle);
    }

    #[test]
    fn test_rrc_state_display() {
        assert_eq!(format!("{}", RrcState::Idle), "RRC_IDLE");
        assert_eq!(format!("{}", RrcState::Connected), "RRC_CONNECTED");
        assert_eq!(format!("{}", RrcState::Inactive), "RRC_INACTIVE");
    }

    #[test]
    fn test_rrc_state_predicates() {
        assert!(RrcState::Idle.is_idle());
        assert!(!RrcState::Idle.is_connected());
        assert!(!RrcState::Idle.is_inactive());

        assert!(!RrcState::Connected.is_idle());
        assert!(RrcState::Connected.is_connected());
        assert!(!RrcState::Connected.is_inactive());

        assert!(!RrcState::Inactive.is_idle());
        assert!(!RrcState::Inactive.is_connected());
        assert!(RrcState::Inactive.is_inactive());
    }

    #[test]
    fn test_should_perform_cell_selection() {
        assert!(RrcState::Idle.should_perform_cell_selection());
        assert!(!RrcState::Connected.should_perform_cell_selection());
        assert!(RrcState::Inactive.should_perform_cell_selection());
    }

    #[test]
    fn test_has_connection_context() {
        assert!(!RrcState::Idle.has_connection_context());
        assert!(RrcState::Connected.has_connection_context());
        assert!(RrcState::Inactive.has_connection_context());
    }

    #[test]
    fn test_state_machine_new() {
        let sm = RrcStateMachine::new();
        assert_eq!(sm.state(), RrcState::Idle);
        assert!(sm.previous_state().is_none());
        assert_eq!(sm.transition_count(), 0);
    }

    #[test]
    fn test_state_machine_idle_to_connected() {
        let mut sm = RrcStateMachine::new();
        
        let result = sm.transition(RrcStateTransition::SetupComplete);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), RrcState::Connected);
        assert_eq!(sm.state(), RrcState::Connected);
        assert_eq!(sm.previous_state(), Some(RrcState::Idle));
        assert_eq!(sm.transition_count(), 1);
    }

    #[test]
    fn test_state_machine_connected_to_idle() {
        let mut sm = RrcStateMachine::new();
        sm.transition(RrcStateTransition::SetupComplete).unwrap();
        
        let result = sm.transition(RrcStateTransition::Release);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), RrcState::Idle);
        assert_eq!(sm.state(), RrcState::Idle);
        assert_eq!(sm.previous_state(), Some(RrcState::Connected));
    }

    #[test]
    fn test_state_machine_connected_to_inactive() {
        let mut sm = RrcStateMachine::new();
        sm.transition(RrcStateTransition::SetupComplete).unwrap();
        
        let result = sm.transition(RrcStateTransition::Suspend);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), RrcState::Inactive);
        assert_eq!(sm.state(), RrcState::Inactive);
    }

    #[test]
    fn test_state_machine_inactive_to_connected() {
        let mut sm = RrcStateMachine::new();
        sm.transition(RrcStateTransition::SetupComplete).unwrap();
        sm.transition(RrcStateTransition::Suspend).unwrap();
        
        let result = sm.transition(RrcStateTransition::Resume);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), RrcState::Connected);
        assert_eq!(sm.state(), RrcState::Connected);
    }

    #[test]
    fn test_state_machine_inactive_to_idle() {
        let mut sm = RrcStateMachine::new();
        sm.transition(RrcStateTransition::SetupComplete).unwrap();
        sm.transition(RrcStateTransition::Suspend).unwrap();
        
        let result = sm.transition(RrcStateTransition::ReleaseFromInactive);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), RrcState::Idle);
        assert_eq!(sm.state(), RrcState::Idle);
    }

    #[test]
    fn test_state_machine_radio_link_failure_from_connected() {
        let mut sm = RrcStateMachine::new();
        sm.transition(RrcStateTransition::SetupComplete).unwrap();
        
        let result = sm.transition(RrcStateTransition::RadioLinkFailure);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), RrcState::Idle);
    }

    #[test]
    fn test_state_machine_radio_link_failure_from_inactive() {
        let mut sm = RrcStateMachine::new();
        sm.transition(RrcStateTransition::SetupComplete).unwrap();
        sm.transition(RrcStateTransition::Suspend).unwrap();
        
        let result = sm.transition(RrcStateTransition::RadioLinkFailure);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), RrcState::Idle);
    }

    #[test]
    fn test_state_machine_invalid_transition_from_idle() {
        let mut sm = RrcStateMachine::new();
        
        // Cannot release from Idle
        let result = sm.transition(RrcStateTransition::Release);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.current_state, RrcState::Idle);
        assert_eq!(err.attempted_transition, RrcStateTransition::Release);
        
        // Cannot suspend from Idle
        let result = sm.transition(RrcStateTransition::Suspend);
        assert!(result.is_err());
        
        // Cannot resume from Idle
        let result = sm.transition(RrcStateTransition::Resume);
        assert!(result.is_err());
    }

    #[test]
    fn test_state_machine_invalid_transition_from_connected() {
        let mut sm = RrcStateMachine::new();
        sm.transition(RrcStateTransition::SetupComplete).unwrap();
        
        // Cannot setup again from Connected
        let result = sm.transition(RrcStateTransition::SetupComplete);
        assert!(result.is_err());
        
        // Cannot resume from Connected
        let result = sm.transition(RrcStateTransition::Resume);
        assert!(result.is_err());
    }

    #[test]
    fn test_state_machine_invalid_transition_from_inactive() {
        let mut sm = RrcStateMachine::new();
        sm.transition(RrcStateTransition::SetupComplete).unwrap();
        sm.transition(RrcStateTransition::Suspend).unwrap();
        
        // Cannot setup from Inactive
        let result = sm.transition(RrcStateTransition::SetupComplete);
        assert!(result.is_err());
        
        // Cannot suspend from Inactive
        let result = sm.transition(RrcStateTransition::Suspend);
        assert!(result.is_err());
    }

    #[test]
    fn test_state_machine_can_transition() {
        let mut sm = RrcStateMachine::new();
        
        // From Idle
        assert!(sm.can_transition(RrcStateTransition::SetupComplete));
        assert!(!sm.can_transition(RrcStateTransition::Release));
        assert!(!sm.can_transition(RrcStateTransition::Suspend));
        assert!(!sm.can_transition(RrcStateTransition::Resume));
        
        sm.transition(RrcStateTransition::SetupComplete).unwrap();
        
        // From Connected
        assert!(!sm.can_transition(RrcStateTransition::SetupComplete));
        assert!(sm.can_transition(RrcStateTransition::Release));
        assert!(sm.can_transition(RrcStateTransition::Suspend));
        assert!(!sm.can_transition(RrcStateTransition::Resume));
        assert!(sm.can_transition(RrcStateTransition::RadioLinkFailure));
    }

    #[test]
    fn test_state_machine_validate_transition() {
        let sm = RrcStateMachine::new();
        
        let result = sm.validate_transition(RrcStateTransition::SetupComplete);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), RrcState::Connected);
        
        // State should not change from validate
        assert_eq!(sm.state(), RrcState::Idle);
    }

    #[test]
    fn test_state_machine_force_state() {
        let mut sm = RrcStateMachine::new();
        
        sm.force_state(RrcState::Connected);
        assert_eq!(sm.state(), RrcState::Connected);
        assert_eq!(sm.previous_state(), Some(RrcState::Idle));
        assert_eq!(sm.transition_count(), 1);
        
        sm.force_state(RrcState::Inactive);
        assert_eq!(sm.state(), RrcState::Inactive);
        assert_eq!(sm.previous_state(), Some(RrcState::Connected));
    }

    #[test]
    fn test_state_machine_reset() {
        let mut sm = RrcStateMachine::new();
        sm.transition(RrcStateTransition::SetupComplete).unwrap();
        sm.transition(RrcStateTransition::Suspend).unwrap();
        
        sm.reset();
        assert_eq!(sm.state(), RrcState::Idle);
        assert!(sm.previous_state().is_none());
        assert_eq!(sm.transition_count(), 0);
    }

    #[test]
    fn test_state_machine_display() {
        let sm = RrcStateMachine::new();
        assert_eq!(format!("{}", sm), "RrcStateMachine(state=RRC_IDLE)");
    }

    #[test]
    fn test_state_transition_display() {
        assert_eq!(format!("{}", RrcStateTransition::SetupComplete), "RRC Setup Complete");
        assert_eq!(format!("{}", RrcStateTransition::Release), "RRC Release");
        assert_eq!(format!("{}", RrcStateTransition::Suspend), "RRC Suspend");
        assert_eq!(format!("{}", RrcStateTransition::Resume), "RRC Resume");
        assert_eq!(format!("{}", RrcStateTransition::RadioLinkFailure), "Radio Link Failure");
    }

    #[test]
    fn test_state_error_display() {
        let err = RrcStateError {
            current_state: RrcState::Idle,
            attempted_transition: RrcStateTransition::Release,
            message: "Cannot release from Idle".to_string(),
        };
        let display = format!("{}", err);
        assert!(display.contains("Invalid RRC state transition"));
        assert!(display.contains("RRC Release"));
        assert!(display.contains("RRC_IDLE"));
    }

    #[test]
    fn test_full_connection_lifecycle() {
        let mut sm = RrcStateMachine::new();
        
        // Initial state
        assert_eq!(sm.state(), RrcState::Idle);
        
        // Establish connection
        sm.transition(RrcStateTransition::SetupComplete).unwrap();
        assert_eq!(sm.state(), RrcState::Connected);
        
        // Suspend connection
        sm.transition(RrcStateTransition::Suspend).unwrap();
        assert_eq!(sm.state(), RrcState::Inactive);
        
        // Resume connection
        sm.transition(RrcStateTransition::Resume).unwrap();
        assert_eq!(sm.state(), RrcState::Connected);
        
        // Release connection
        sm.transition(RrcStateTransition::Release).unwrap();
        assert_eq!(sm.state(), RrcState::Idle);
        
        assert_eq!(sm.transition_count(), 4);
    }
}
