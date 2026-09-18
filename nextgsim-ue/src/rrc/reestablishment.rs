//! RRC Re-establishment Procedure (UE Side)
//!
//! This module implements the UE-side RRC re-establishment procedure as defined
//! in 3GPP TS 38.331 Section 5.3.7.
//!
//! # Trigger Conditions
//!
//! The UE initiates RRC re-establishment when it detects:
//! 1. **Radio Link Failure (RLF)**: In-sync/out-of-sync detection via N310/N311 counters
//! 2. **Handover failure**: T304 expires or target cell unreachable
//! 3. **Integrity check failure**: Received RRC message fails integrity check
//! 4. **RRC reconfiguration failure**: Cannot apply received reconfiguration
//!
//! # Procedure Flow
//!
//! ```text
//! UE                                    gNB
//!  |-- RRCReestablishmentRequest ------->|
//!  |<-- RRCReestablishment --------------|
//!  |-- RRCReestablishmentComplete ------>|
//! ```
//!
//! If the network responds with `RRCSetup` instead, the UE falls back to
//! a fresh RRC connection (losing the AS context).
//!
//! # Timers
//!
//! - **T311**: Started when RLF detected, used for cell selection before sending request
//! - **T301**: Started when RRCReestablishmentRequest is sent; expires → RRC_IDLE
//!
//! # Reference
//!
//! - 3GPP TS 38.331 Section 5.3.7
//! - UERANSIM `src/ue/rrc/` reestablishment handling

use std::time::{Duration, Instant};

use super::state::{RrcState, RrcStateMachine, RrcStateTransition};

// ============================================================================
// Constants
// ============================================================================

/// T301: RRC re-establishment timer (ms), the PRE-SIB1 default.
///
/// The serving cell decides the operative value and broadcasts it in
/// `ue-TimersAndConstants` (TS 38.331 §7.1.1); see
/// [`ReestablishmentProcedure::apply_broadcast_timers`]. This constant is what the
/// procedure guards with until a SIB1 for the serving cell has been read -- which is a
/// real window, because re-establishment can be triggered on a cell whose SIB1 the UE
/// has not acquired.
pub const T301_DEFAULT_MS: u64 = 1000;

/// T311: RLF recovery timer (ms), the PRE-SIB1 default.
///
/// Same provenance as [`T301_DEFAULT_MS`]: the cell broadcasts the operative value and
/// this is the fallback until it has been read.
pub const T311_DEFAULT_MS: u64 = 1000;

/// N310: Out-of-sync indication count before RLF declared.
pub const N310_DEFAULT: u32 = 1;

/// N311: In-sync indication count to cancel RLF.
pub const N311_DEFAULT: u32 = 1;

// ============================================================================
// RLF Cause
// ============================================================================

/// Cause of Radio Link Failure or reestablishment trigger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReestablishmentTrigger {
    /// Radio link failure detected (T310 expiry or N310 counter)
    RadioLinkFailure,
    /// Handover failure (T304 expired or sync failure)
    HandoverFailure,
    /// RRC reconfiguration failure
    ReconfigurationFailure,
    /// Integrity check failure on received message
    IntegrityCheckFailure,
}

impl std::fmt::Display for ReestablishmentTrigger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RadioLinkFailure => write!(f, "radio-link-failure"),
            Self::HandoverFailure => write!(f, "handover-failure"),
            Self::ReconfigurationFailure => write!(f, "reconfiguration-failure"),
            Self::IntegrityCheckFailure => write!(f, "integrity-check-failure"),
        }
    }
}

impl ReestablishmentTrigger {
    /// Maps to the `ReestablishmentCauseValue` used in the RRC codec layer.
    /// Returns a string matching the codec enum variant names.
    pub fn to_rrc_cause_str(&self) -> &'static str {
        match self {
            Self::ReconfigurationFailure => "ReconfigurationFailure",
            Self::HandoverFailure => "HandoverFailure",
            Self::RadioLinkFailure | Self::IntegrityCheckFailure => "OtherFailure",
        }
    }
}

// ============================================================================
// RLF Detection State
// ============================================================================

/// Tracks radio link quality for RLF detection.
///
/// Per 3GPP TS 38.331 Section 5.3.10.3:
/// - N310 consecutive out-of-sync → start T310
/// - T310 expires → RLF declared
/// - N311 consecutive in-sync → cancel T310
#[derive(Debug, Clone, Default)]
pub struct RlfDetector {
    /// Count of consecutive out-of-sync indications
    out_of_sync_count: u32,
    /// Count of consecutive in-sync indications
    in_sync_count: u32,
    /// Whether T310 is running (RLF pending)
    t310_running: bool,
    /// When T310 was started
    t310_started: Option<Instant>,
    /// T310 duration in ms
    t310_duration_ms: u64,
    /// N310 threshold
    n310: u32,
    /// N311 threshold
    n311: u32,
}

impl RlfDetector {
    /// Creates a new RLF detector with default parameters.
    pub fn new() -> Self {
        Self {
            t310_duration_ms: 1000, // 1 second default
            n310: N310_DEFAULT,
            n311: N311_DEFAULT,
            ..Default::default()
        }
    }

    /// Creates a new RLF detector with custom parameters.
    pub fn with_params(n310: u32, n311: u32, t310_ms: u64) -> Self {
        Self {
            t310_duration_ms: t310_ms,
            n310,
            n311,
            ..Default::default()
        }
    }

    /// Processes an out-of-sync indication.
    ///
    /// Returns `true` if RLF has been detected (T310 has expired).
    pub fn on_out_of_sync(&mut self) -> bool {
        self.in_sync_count = 0;
        self.out_of_sync_count += 1;

        if !self.t310_running && self.out_of_sync_count >= self.n310 {
            self.t310_running = true;
            self.t310_started = Some(Instant::now());
            tracing::debug!("T310 started after {} out-of-sync indications", self.n310);
        }

        self.check_t310_expired()
    }

    /// Processes an in-sync indication.
    ///
    /// Returns `true` if T310 was cancelled (radio link recovered).
    pub fn on_in_sync(&mut self) -> bool {
        self.out_of_sync_count = 0;
        self.in_sync_count += 1;

        if self.t310_running && self.in_sync_count >= self.n311 {
            self.t310_running = false;
            self.t310_started = None;
            self.in_sync_count = 0;
            tracing::debug!("T310 cancelled after {} in-sync indications", self.n311);
            return true;
        }

        false
    }

    /// Checks if T310 has expired, indicating RLF.
    pub fn check_t310_expired(&self) -> bool {
        if let Some(started) = self.t310_started {
            if started.elapsed() >= Duration::from_millis(self.t310_duration_ms) {
                return true;
            }
        }
        false
    }

    /// Returns true if T310 is currently running.
    pub fn t310_is_running(&self) -> bool {
        self.t310_running
    }

    /// Applies the `ue-TimersAndConstants` the serving cell broadcast (TS 38.331 §7.1.1).
    ///
    /// The cell decides these, and SIB1 carries them. Before #168 the UE read none of them:
    /// the gNB broadcast `t310`, `n310` and `n311` and the UE ran on its own constants,
    /// which happened to agree only because both sides picked the same numbers.
    pub fn apply_broadcast_constants(&mut self, n310: u32, n311: u32, t310_ms: u64) {
        self.n310 = n310;
        self.n311 = n311;
        self.t310_duration_ms = t310_ms;
    }

    /// The T310 duration currently in force, in milliseconds.
    ///
    /// Exists so a test can assert the BROADCAST value reached the timer. Waiting the timer
    /// out would pass identically for the hardcoded default and say nothing about where the
    /// value came from.
    pub fn t310_duration_ms(&self) -> u64 {
        self.t310_duration_ms
    }

    /// Backdate T310 so the next [`Self::check_t310_expired`] finds it expired.
    ///
    /// Test-only, and the same shape as `ResumeProcedure::expire_t319_for_test`.
    #[cfg(test)]
    pub(crate) fn expire_t310_for_test(&mut self) {
        if let Some(started) = self.t310_started {
            self.t310_started = Some(
                started - Duration::from_millis(self.t310_duration_ms) - Duration::from_millis(1),
            );
        }
    }

    /// Resets the detector (called after re-establishment completes).
    pub fn reset(&mut self) {
        self.out_of_sync_count = 0;
        self.in_sync_count = 0;
        self.t310_running = false;
        self.t310_started = None;
    }
}

// ============================================================================
// Re-establishment State Machine
// ============================================================================

/// State of the re-establishment procedure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReestablishmentState {
    /// No re-establishment in progress
    #[default]
    Idle,
    /// Searching for a suitable cell (T311 running)
    CellSearch,
    /// Request sent, waiting for network response (T301 running)
    WaitingForResponse,
    /// Re-establishment complete
    Complete,
    /// Re-establishment failed — fell back to RRC_IDLE
    Failed,
}

/// UE-side RRC Re-establishment Procedure Manager.
///
/// Orchestrates the full re-establishment flow:
/// 1. RLF detection (via `RlfDetector`)
/// 2. Cell selection during T311
/// 3. Sending `RRCReestablishmentRequest`
/// 4. Processing the network response
/// 5. Sending `RRCReestablishmentComplete`
#[derive(Debug)]
pub struct ReestablishmentProcedure {
    /// Current state of the procedure
    state: ReestablishmentState,
    /// What triggered this re-establishment
    trigger: Option<ReestablishmentTrigger>,
    /// UE context for the request (C-RNTI, PCI, ShortMAC-I)
    ue_c_rnti: u16,
    ue_pci: u16,
    short_mac_i: u16,
    /// T311 expiry instant (cell selection deadline)
    t311_deadline: Option<Instant>,
    /// T301 expiry instant (response deadline)
    t301_deadline: Option<Instant>,
    /// T301 duration in force (ms): the serving cell's broadcast value once one has
    /// been read, [`T301_DEFAULT_MS`] before that.
    t301_duration_ms: u64,
    /// T311 duration in force (ms), same provenance as [`Self::t301_duration_ms`].
    t311_duration_ms: u64,
    /// Whether we received an `RRCReestablishment` (vs `RRCSetup` fallback)
    received_reestablishment: bool,
}

impl Default for ReestablishmentProcedure {
    fn default() -> Self {
        Self::new()
    }
}

impl ReestablishmentProcedure {
    /// Creates a new re-establishment procedure manager.
    pub fn new() -> Self {
        Self {
            state: ReestablishmentState::Idle,
            trigger: None,
            ue_c_rnti: 0,
            ue_pci: 0,
            short_mac_i: 0,
            t311_deadline: None,
            t301_deadline: None,
            t301_duration_ms: T301_DEFAULT_MS,
            t311_duration_ms: T311_DEFAULT_MS,
            received_reestablishment: false,
        }
    }

    /// Applies the T301 and T311 values the serving cell broadcast in
    /// `ue-TimersAndConstants` (TS 38.331 §7.1.1).
    ///
    /// The cell decides these and SIB1 has always carried them; before #176 the UE read
    /// its own [`T301_DEFAULT_MS`]/[`T311_DEFAULT_MS`] instead. The two agreed only
    /// because both sides happened to pick 1000 ms, so raising `t301` at the gNB left the
    /// UE giving up while the network still considered the procedure live -- the same
    /// shape as #167's dropped T304.
    ///
    /// Deliberately NOT reset by [`Self::reset`]: these describe the CELL, not the
    /// procedure, so they outlive any one re-establishment attempt on that cell.
    pub fn apply_broadcast_timers(&mut self, t301_ms: u64, t311_ms: u64) {
        self.t301_duration_ms = t301_ms;
        self.t311_duration_ms = t311_ms;
    }

    /// The T301 duration currently in force, in milliseconds.
    ///
    /// Exists so a test can assert the BROADCAST value reached the timer; waiting the
    /// timer out would pass identically for the hardcoded default and say nothing about
    /// where the value came from. Not a product knob -- nothing in production reads it.
    pub fn t301_duration_ms(&self) -> u64 {
        self.t301_duration_ms
    }

    /// The T311 duration currently in force, in milliseconds. See
    /// [`Self::t301_duration_ms`].
    pub fn t311_duration_ms(&self) -> u64 {
        self.t311_duration_ms
    }

    /// Time left on the armed T301, or `None` when it is not running.
    ///
    /// [`Self::t301_duration_ms`] alone cannot tell whether [`Self::on_cell_found`] reads
    /// the field or still reads [`T301_DEFAULT_MS`]: a stored value nobody arms from is
    /// exactly the defect #176 exists to fix. This observes the DEADLINE, so a broadcast
    /// value that never reached the arming site is visible.
    #[cfg(test)]
    pub(crate) fn t301_remaining(&self) -> Option<Duration> {
        self.t301_deadline
            .map(|d| d.saturating_duration_since(Instant::now()))
    }

    /// Time left on the armed T311, or `None` when it is not running. See
    /// [`Self::t301_remaining`].
    #[cfg(test)]
    pub(crate) fn t311_remaining(&self) -> Option<Duration> {
        self.t311_deadline
            .map(|d| d.saturating_duration_since(Instant::now()))
    }

    /// Returns the current procedure state.
    pub fn state(&self) -> ReestablishmentState {
        self.state
    }

    /// Returns the trigger that initiated this procedure.
    pub fn trigger(&self) -> Option<ReestablishmentTrigger> {
        self.trigger
    }

    /// Returns true if a re-establishment is in progress.
    pub fn is_in_progress(&self) -> bool {
        !matches!(
            self.state,
            ReestablishmentState::Idle
                | ReestablishmentState::Complete
                | ReestablishmentState::Failed
        )
    }

    // ========================================================================
    // Step 1: Initiate (on RLF / failure detection)
    // ========================================================================

    /// Initiates the re-establishment procedure after a failure is detected.
    ///
    /// Per 3GPP TS 38.331 Section 5.3.7.2:
    /// - Suspends the RRC connection
    /// - Starts T311 for cell selection
    /// - Returns the parameters needed to build `RRCReestablishmentRequest`
    ///
    /// # Arguments
    /// * `trigger` - What caused the re-establishment
    /// * `c_rnti` - UE's C-RNTI in the old cell
    /// * `pci` - Old serving cell PCI
    /// * `short_mac_i` - Computed ShortMAC-I for the old cell
    /// * `rrc_sm` - UE RRC state machine (will be transitioned)
    ///
    /// # Returns
    /// * `Ok(ReestablishmentRequestParams)` - Parameters for the RRC message
    /// * `Err` - If not in a valid state to re-establish
    pub fn initiate(
        &mut self,
        trigger: ReestablishmentTrigger,
        c_rnti: u16,
        pci: u16,
        short_mac_i: u16,
        rrc_sm: &mut RrcStateMachine,
    ) -> Result<ReestablishmentRequestParams, ReestablishmentError> {
        // Can only re-establish from CONNECTED or INACTIVE
        if !rrc_sm.state().has_connection_context() {
            return Err(ReestablishmentError::InvalidState(rrc_sm.state()));
        }

        if self.is_in_progress() {
            return Err(ReestablishmentError::AlreadyInProgress);
        }

        tracing::info!("RRC re-establishment initiated: trigger={}", trigger);

        // Transition RRC state machine to IDLE (connection is lost)
        let _ = rrc_sm.on_radio_link_failure();

        self.trigger = Some(trigger);
        self.ue_c_rnti = c_rnti;
        self.ue_pci = pci;
        self.short_mac_i = short_mac_i;
        self.state = ReestablishmentState::CellSearch;
        self.t311_deadline = Some(Instant::now() + Duration::from_millis(self.t311_duration_ms));
        self.received_reestablishment = false;

        Ok(ReestablishmentRequestParams {
            c_rnti,
            pci,
            short_mac_i,
            trigger,
        })
    }

    // ========================================================================
    // Step 2: Cell found — send request (T311 expires or suitable cell found)
    // ========================================================================

    /// Called when a suitable cell is found during cell search.
    ///
    /// Transitions to `WaitingForResponse` and starts T301.
    /// The caller should now encode and send `RRCReestablishmentRequest`.
    pub fn on_cell_found(&mut self) -> Result<(), ReestablishmentError> {
        if self.state != ReestablishmentState::CellSearch {
            return Err(ReestablishmentError::InvalidProcedureState(self.state));
        }

        self.state = ReestablishmentState::WaitingForResponse;
        self.t301_deadline = Some(Instant::now() + Duration::from_millis(self.t301_duration_ms));
        self.t311_deadline = None;

        tracing::debug!("Suitable cell found — sending RRCReestablishmentRequest (T301 started)");
        Ok(())
    }

    // ========================================================================
    // Step 3: Receive RRCReestablishment from network
    // ========================================================================

    /// Processes the network's `RRCReestablishment` response.
    ///
    /// On success:
    /// - Stops T301
    /// - Transitions to Complete
    /// - The caller should send `RRCReestablishmentComplete`
    ///
    /// # Arguments
    /// * `rrc_transaction_id` - Transaction ID from the received message (for validation)
    /// * `rrc_sm` - UE RRC state machine (will transition to CONNECTED)
    pub fn on_reestablishment_received(
        &mut self,
        _rrc_transaction_id: u8,
        rrc_sm: &mut RrcStateMachine,
    ) -> Result<ReestablishmentCompleteParams, ReestablishmentError> {
        if self.state != ReestablishmentState::WaitingForResponse {
            return Err(ReestablishmentError::InvalidProcedureState(self.state));
        }

        // Stop T301
        self.t301_deadline = None;
        self.received_reestablishment = true;

        // Transition to CONNECTED
        let _ = rrc_sm.transition(RrcStateTransition::SetupComplete);

        self.state = ReestablishmentState::Complete;

        tracing::info!("RRC re-establishment successful — sending Complete");

        Ok(ReestablishmentCompleteParams {
            rrc_transaction_id: _rrc_transaction_id,
        })
    }

    // ========================================================================
    // Step 3 (fallback): Receive RRCSetup (network rejected re-establishment)
    // ========================================================================

    /// Processes an `RRCSetup` received instead of `RRCReestablishment`.
    ///
    /// The network rejected re-establishment; the UE falls back to a fresh
    /// RRC connection following the normal RRC setup procedure.
    ///
    /// Stops T301 and marks as failed (caller handles `RRCSetup` normally).
    pub fn on_rrc_setup_fallback(&mut self) {
        tracing::info!(
            "RRC Setup received during re-establishment — falling back to fresh RRC connection"
        );
        self.t301_deadline = None;
        self.state = ReestablishmentState::Failed;
        self.trigger = None;
    }

    // ========================================================================
    // Timer checks
    // ========================================================================

    /// Checks if T311 (cell selection timer) has expired.
    ///
    /// If T311 expires before finding a suitable cell, the UE moves to RRC_IDLE.
    pub fn t311_expired(&self) -> bool {
        self.t311_deadline.is_some_and(|d| Instant::now() >= d)
    }

    /// Checks if T301 (response timer) has expired.
    ///
    /// If T301 expires, the UE moves to RRC_IDLE.
    pub fn t301_expired(&self) -> bool {
        self.t301_deadline.is_some_and(|d| Instant::now() >= d)
    }

    /// Called when T311 or T301 expires.
    ///
    /// Moves the procedure to Failed state and returns to RRC_IDLE.
    pub fn on_timer_expired(&mut self, rrc_sm: &mut RrcStateMachine) {
        tracing::warn!(
            "Re-establishment timer expired in state {:?} — moving to RRC_IDLE",
            self.state
        );
        self.state = ReestablishmentState::Failed;
        self.t301_deadline = None;
        self.t311_deadline = None;
        // Ensure we're in IDLE
        if !rrc_sm.state().is_idle() {
            let _ = rrc_sm.on_radio_link_failure();
        }
    }

    /// Back-dates whichever of T301/T311 is armed, so a test reaches the expiry
    /// without waiting the guard out. Same reasoning as
    /// `ResumeProcedure::expire_t319_for_test`.
    ///
    /// The backdate comes from the duration IN FORCE, not from the `*_DEFAULT_MS`
    /// constant. Reading the constant while the timer runs on a broadcast value is a
    /// helper that expires the wrong timer: a cell broadcasting T311 = 5000 ms leaves a
    /// deadline 5 s out, and subtracting 2 x 1000 ms lands 3 s in the FUTURE, so the
    /// expiry never fires and the test reports nothing.
    #[cfg(test)]
    pub(crate) fn expire_guard_timers_for_test(&mut self) {
        if let Some(deadline) = self.t301_deadline {
            self.t301_deadline = Some(deadline - Duration::from_millis(self.t301_duration_ms * 2));
        }
        if let Some(deadline) = self.t311_deadline {
            self.t311_deadline = Some(deadline - Duration::from_millis(self.t311_duration_ms * 2));
        }
    }

    /// Resets the procedure to idle state.
    ///
    /// The T301/T311 DURATIONS survive: they are the serving cell's configuration
    /// (TS 38.331 §7.1.1), not state belonging to one attempt, and clearing them here
    /// would silently put the UE back on its own defaults after every completed or
    /// abandoned re-establishment.
    pub fn reset(&mut self) {
        self.state = ReestablishmentState::Idle;
        self.trigger = None;
        self.ue_c_rnti = 0;
        self.ue_pci = 0;
        self.short_mac_i = 0;
        self.t311_deadline = None;
        self.t301_deadline = None;
        self.received_reestablishment = false;
    }
}

// ============================================================================
// Parameter Structs
// ============================================================================

/// Parameters for building an `RRCReestablishmentRequest`.
#[derive(Debug, Clone)]
pub struct ReestablishmentRequestParams {
    /// C-RNTI in the old cell
    pub c_rnti: u16,
    /// PCI of the old serving cell
    pub pci: u16,
    /// ShortMAC-I computed for the old cell
    pub short_mac_i: u16,
    /// Trigger cause
    pub trigger: ReestablishmentTrigger,
}

/// Parameters for building an `RRCReestablishmentComplete`.
#[derive(Debug, Clone)]
pub struct ReestablishmentCompleteParams {
    /// RRC Transaction Identifier from the received `RRCReestablishment`
    pub rrc_transaction_id: u8,
}

// ============================================================================
// Error Type
// ============================================================================

/// Errors for the re-establishment procedure.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ReestablishmentError {
    /// Not in a valid RRC state to re-establish
    #[error("Invalid RRC state for re-establishment: {0:?}")]
    InvalidState(RrcState),
    /// Re-establishment already in progress
    #[error("Re-establishment already in progress")]
    AlreadyInProgress,
    /// Procedure not in the expected state
    #[error("Procedure in unexpected state: {0:?}")]
    InvalidProcedureState(ReestablishmentState),
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_connected_sm() -> RrcStateMachine {
        let mut sm = RrcStateMachine::new();
        sm.transition(RrcStateTransition::SetupComplete).unwrap();
        sm
    }

    #[test]
    fn test_rlf_detector_out_of_sync_triggers_t310() {
        let mut detector = RlfDetector::with_params(2, 1, 100_000); // very long T310
        detector.on_out_of_sync();
        assert!(!detector.t310_is_running()); // N310=2, only 1 out-of-sync
        detector.on_out_of_sync();
        assert!(detector.t310_is_running()); // Now T310 starts
    }

    #[test]
    fn test_rlf_detector_in_sync_cancels_t310() {
        let mut detector = RlfDetector::with_params(1, 1, 100_000);
        detector.on_out_of_sync();
        assert!(detector.t310_is_running());

        let cancelled = detector.on_in_sync();
        assert!(cancelled);
        assert!(!detector.t310_is_running());
    }

    #[test]
    fn test_rlf_detector_reset() {
        let mut detector = RlfDetector::with_params(1, 1, 100_000);
        detector.on_out_of_sync();
        assert!(detector.t310_is_running());
        detector.reset();
        assert!(!detector.t310_is_running());
    }

    #[test]
    fn test_initiate_from_connected() {
        let mut sm = setup_connected_sm();
        let mut proc = ReestablishmentProcedure::new();

        let result = proc.initiate(
            ReestablishmentTrigger::RadioLinkFailure,
            0x1234,
            100,
            0xABCD,
            &mut sm,
        );

        assert!(result.is_ok());
        assert_eq!(proc.state(), ReestablishmentState::CellSearch);
        assert!(proc.is_in_progress());
        // RRC state machine goes to IDLE
        assert_eq!(sm.state(), RrcState::Idle);

        let params = result.unwrap();
        assert_eq!(params.c_rnti, 0x1234);
        assert_eq!(params.pci, 100);
        assert_eq!(params.short_mac_i, 0xABCD);
        assert!(matches!(
            params.trigger,
            ReestablishmentTrigger::RadioLinkFailure
        ));
    }

    #[test]
    fn test_initiate_from_idle_fails() {
        let mut sm = RrcStateMachine::new(); // starts in IDLE
        let mut proc = ReestablishmentProcedure::new();

        let result = proc.initiate(
            ReestablishmentTrigger::RadioLinkFailure,
            0x1000,
            50,
            0x0001,
            &mut sm,
        );

        assert!(matches!(result, Err(ReestablishmentError::InvalidState(_))));
    }

    #[test]
    fn test_initiate_already_in_progress() {
        let mut sm = setup_connected_sm();
        let mut proc = ReestablishmentProcedure::new();

        let _ = proc.initiate(ReestablishmentTrigger::RadioLinkFailure, 1, 1, 1, &mut sm);

        // Second initiation attempt
        let mut sm2 = setup_connected_sm();
        let result = proc.initiate(ReestablishmentTrigger::HandoverFailure, 2, 2, 2, &mut sm2);
        assert!(matches!(
            result,
            Err(ReestablishmentError::AlreadyInProgress)
        ));
    }

    #[test]
    fn test_on_cell_found_transitions_state() {
        let mut sm = setup_connected_sm();
        let mut proc = ReestablishmentProcedure::new();

        let _ = proc.initiate(ReestablishmentTrigger::RadioLinkFailure, 1, 1, 1, &mut sm);

        let result = proc.on_cell_found();
        assert!(result.is_ok());
        assert_eq!(proc.state(), ReestablishmentState::WaitingForResponse);
    }

    #[test]
    fn test_full_reestablishment_flow() {
        let mut sm = setup_connected_sm();
        let mut proc = ReestablishmentProcedure::new();

        // Step 1: Initiate
        let _ = proc.initiate(
            ReestablishmentTrigger::HandoverFailure,
            0x0100,
            42,
            0x1234,
            &mut sm,
        );
        assert_eq!(sm.state(), RrcState::Idle);

        // Step 2: Cell found
        proc.on_cell_found().unwrap();
        assert_eq!(proc.state(), ReestablishmentState::WaitingForResponse);

        // Step 3: Receive RRCReestablishment (T301 was running)
        let complete_params = proc.on_reestablishment_received(0, &mut sm).unwrap();
        assert_eq!(proc.state(), ReestablishmentState::Complete);
        assert_eq!(sm.state(), RrcState::Connected);
        assert_eq!(complete_params.rrc_transaction_id, 0);
    }

    #[test]
    fn test_rrc_setup_fallback() {
        let mut sm = setup_connected_sm();
        let mut proc = ReestablishmentProcedure::new();

        let _ = proc.initiate(ReestablishmentTrigger::RadioLinkFailure, 1, 1, 1, &mut sm);
        proc.on_cell_found().unwrap();

        proc.on_rrc_setup_fallback();
        assert_eq!(proc.state(), ReestablishmentState::Failed);
    }

    #[test]
    fn test_on_timer_expired_moves_to_failed() {
        let mut sm = setup_connected_sm();
        let mut proc = ReestablishmentProcedure::new();

        let _ = proc.initiate(ReestablishmentTrigger::RadioLinkFailure, 1, 1, 1, &mut sm);

        proc.on_timer_expired(&mut sm);
        assert_eq!(proc.state(), ReestablishmentState::Failed);
        assert_eq!(sm.state(), RrcState::Idle);
    }

    #[test]
    fn test_reset_clears_state() {
        let mut sm = setup_connected_sm();
        let mut proc = ReestablishmentProcedure::new();

        let _ = proc.initiate(ReestablishmentTrigger::RadioLinkFailure, 1, 1, 1, &mut sm);
        proc.reset();

        assert_eq!(proc.state(), ReestablishmentState::Idle);
        assert!(!proc.is_in_progress());
        assert!(proc.trigger().is_none());
    }

    #[test]
    fn test_trigger_to_cause_str() {
        assert_eq!(
            ReestablishmentTrigger::ReconfigurationFailure.to_rrc_cause_str(),
            "ReconfigurationFailure"
        );
        assert_eq!(
            ReestablishmentTrigger::HandoverFailure.to_rrc_cause_str(),
            "HandoverFailure"
        );
        assert_eq!(
            ReestablishmentTrigger::RadioLinkFailure.to_rrc_cause_str(),
            "OtherFailure"
        );
    }
}
