//! Procedure Transaction Handling
//!
//! This module implements procedure transaction management for 5G Session Management.
//! Procedure Transaction Identity (PTI) is used to track SM procedures between the UE
//! and the network.
//!
//! # PTI Range
//!
//! - PTI 0: Reserved for network-initiated procedures
//! - PTI 1-254: Valid range for UE-initiated procedures
//!
//! # Procedure Transaction States
//!
//! - INACTIVE: No procedure in progress
//! - PENDING: Procedure initiated, waiting for response
//!
//! # Reference
//!
//! Based on 3GPP TS 24.501 and UERANSIM's `src/ue/nas/sm/procedure.cpp`.

use crate::timer::{UeTimer, DEFAULT_T3580_INTERVAL, DEFAULT_T3581_INTERVAL, DEFAULT_T3582_INTERVAL};
use std::fmt;

/// Minimum valid PTI value (1)
pub const PTI_MIN: u8 = 1;
/// Maximum valid PTI value (254)
pub const PTI_MAX: u8 = 254;
/// Reserved PTI for network-initiated procedures
pub const PTI_UNASSIGNED: u8 = 0;

/// SM timer codes for procedure transactions
pub const SM_TIMER_T3580: u16 = 3580;
pub const SM_TIMER_T3581: u16 = 3581;
pub const SM_TIMER_T3582: u16 = 3582;

/// Procedure Transaction state.
///
/// 3GPP TS 24.501 Section 6.1.3.2
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PtState {
    /// No procedure in progress for this PTI
    #[default]
    Inactive,
    /// Procedure initiated, waiting for network response
    Pending,
}

impl fmt::Display for PtState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PtState::Inactive => write!(f, "INACTIVE"),
            PtState::Pending => write!(f, "PENDING"),
        }
    }
}

/// SM Message type stored in procedure transaction.
///
/// Used to identify the type of procedure for retry and abort handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmMessageType {
    /// PDU Session Establishment Request
    PduSessionEstablishmentRequest,
    /// PDU Session Modification Request
    PduSessionModificationRequest,
    /// PDU Session Release Request
    PduSessionReleaseRequest,
}

impl fmt::Display for SmMessageType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SmMessageType::PduSessionEstablishmentRequest => {
                write!(f, "PDU_SESSION_ESTABLISHMENT_REQUEST")
            }
            SmMessageType::PduSessionModificationRequest => {
                write!(f, "PDU_SESSION_MODIFICATION_REQUEST")
            }
            SmMessageType::PduSessionReleaseRequest => write!(f, "PDU_SESSION_RELEASE_REQUEST"),
        }
    }
}

/// Procedure Transaction context.
///
/// Tracks the state of an SM procedure including the associated timer,
/// message type, and PDU session identity.
#[derive(Debug)]
pub struct ProcedureTransaction {
    /// Procedure Transaction Identity (1-254)
    pti: u8,
    /// Current state of the procedure transaction
    state: PtState,
    /// Timer for the procedure (T3580, T3581, or T3582)
    timer: Option<UeTimer>,
    /// Type of SM message that initiated this procedure
    message_type: Option<SmMessageType>,
    /// Associated PDU Session Identity (1-15)
    psi: u8,
}

impl ProcedureTransaction {
    /// Create a new inactive procedure transaction.
    pub fn new(pti: u8) -> Self {
        Self {
            pti,
            state: PtState::Inactive,
            timer: None,
            message_type: None,
            psi: 0,
        }
    }

    /// Get the PTI value.
    pub fn pti(&self) -> u8 {
        self.pti
    }

    /// Get the current state.
    pub fn state(&self) -> PtState {
        self.state
    }

    /// Check if the procedure transaction is inactive.
    pub fn is_inactive(&self) -> bool {
        self.state == PtState::Inactive
    }

    /// Check if the procedure transaction is pending.
    pub fn is_pending(&self) -> bool {
        self.state == PtState::Pending
    }

    /// Get the associated PSI.
    pub fn psi(&self) -> u8 {
        self.psi
    }

    /// Get the message type.
    pub fn message_type(&self) -> Option<SmMessageType> {
        self.message_type
    }

    /// Get a reference to the timer.
    pub fn timer(&self) -> Option<&UeTimer> {
        self.timer.as_ref()
    }

    /// Get a mutable reference to the timer.
    pub fn timer_mut(&mut self) -> Option<&mut UeTimer> {
        self.timer.as_mut()
    }

    /// Start the procedure transaction with the given parameters.
    ///
    /// # Arguments
    /// * `psi` - PDU Session Identity
    /// * `message_type` - Type of SM message
    /// * `timer_code` - Timer code (3580, 3581, or 3582)
    pub fn start(&mut self, psi: u8, message_type: SmMessageType, timer_code: u16) {
        self.state = PtState::Pending;
        self.psi = psi;
        self.message_type = Some(message_type);

        // Create and start the appropriate timer
        let interval = match timer_code {
            SM_TIMER_T3580 => DEFAULT_T3580_INTERVAL,
            SM_TIMER_T3581 => DEFAULT_T3581_INTERVAL,
            SM_TIMER_T3582 => DEFAULT_T3582_INTERVAL,
            _ => DEFAULT_T3580_INTERVAL, // Default fallback
        };

        let mut timer = UeTimer::new(timer_code, false, interval);
        timer.start(true);
        self.timer = Some(timer);
    }

    /// Reset the procedure transaction to inactive state.
    pub fn reset(&mut self) {
        self.state = PtState::Inactive;
        self.timer = None;
        self.message_type = None;
        self.psi = 0;
    }

    /// Clear the PSI association (used when PS is released but PT needs to remain).
    pub fn clear_psi(&mut self) {
        self.psi = 0;
    }
}

impl Default for ProcedureTransaction {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Result of PTI/PSI validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PtiValidationResult {
    /// PTI and PSI are valid
    Valid,
    /// PTI value is out of valid range
    InvalidPti,
    /// PSI doesn't match the expected value for this PTI
    PsiMismatch { expected: u8, received: u8 },
}

/// Procedure Transaction Manager.
///
/// Manages all procedure transactions for the UE, providing allocation,
/// deallocation, and lookup functionality.
pub struct ProcedureTransactionManager {
    /// Array of procedure transactions indexed by PTI (0-254)
    /// Index 0 is unused (PTI 0 is reserved)
    transactions: [ProcedureTransaction; 255],
}

impl ProcedureTransactionManager {
    /// Create a new procedure transaction manager.
    pub fn new() -> Self {
        // Initialize all transactions with their PTI values
        let transactions = std::array::from_fn(|i| ProcedureTransaction::new(i as u8));
        Self { transactions }
    }

    /// Allocate a new PTI for a procedure.
    ///
    /// Returns `Some(pti)` if allocation succeeds, `None` if no PTI is available.
    pub fn allocate(&mut self) -> Option<u8> {
        for pti in PTI_MIN..=PTI_MAX {
            if self.transactions[pti as usize].is_inactive() {
                // Reset the transaction to ensure clean state
                self.transactions[pti as usize] = ProcedureTransaction::new(pti);
                return Some(pti);
            }
        }
        None
    }

    /// Free a PTI, returning it to the available pool.
    pub fn free(&mut self, pti: u8) {
        if (PTI_MIN..=PTI_MAX).contains(&pti) {
            self.transactions[pti as usize].reset();
        }
    }

    /// Get a reference to a procedure transaction by PTI.
    pub fn get(&self, pti: u8) -> Option<&ProcedureTransaction> {
        if pti <= PTI_MAX {
            Some(&self.transactions[pti as usize])
        } else {
            None
        }
    }

    /// Get a mutable reference to a procedure transaction by PTI.
    pub fn get_mut(&mut self, pti: u8) -> Option<&mut ProcedureTransaction> {
        if pti <= PTI_MAX {
            Some(&mut self.transactions[pti as usize])
        } else {
            None
        }
    }

    /// Validate PTI and PSI from a received SM message.
    ///
    /// Checks that:
    /// 1. PTI is in valid range (1-254)
    /// 2. PSI matches the expected PSI for this PTI
    pub fn validate_pti_psi(&self, pti: u8, psi: u8) -> PtiValidationResult {
        // Check PTI range
        if !(PTI_MIN..=PTI_MAX).contains(&pti) {
            return PtiValidationResult::InvalidPti;
        }

        // Check PSI matches
        let pt = &self.transactions[pti as usize];
        if pt.psi != psi {
            return PtiValidationResult::PsiMismatch {
                expected: pt.psi,
                received: psi,
            };
        }

        PtiValidationResult::Valid
    }

    /// Find all pending procedure transactions for a given PSI.
    ///
    /// Returns a vector of PTI values.
    pub fn find_by_psi(&self, psi: u8) -> Vec<u8> {
        let mut result = Vec::new();
        for pti in PTI_MIN..=PTI_MAX {
            let pt = &self.transactions[pti as usize];
            if pt.is_pending() && pt.psi == psi {
                result.push(pti);
            }
        }
        result
    }

    /// Abort a procedure by PTI.
    ///
    /// Returns the message type and PSI if the procedure was pending.
    pub fn abort(&mut self, pti: u8) -> Option<(SmMessageType, u8)> {
        if !(PTI_MIN..=PTI_MAX).contains(&pti) {
            return None;
        }

        let pt = &mut self.transactions[pti as usize];
        if !pt.is_pending() {
            return None;
        }

        let result = pt.message_type.map(|msg_type| (msg_type, pt.psi));
        pt.reset();
        result
    }

    /// Abort all procedures for a given PSI.
    ///
    /// Returns a vector of (PTI, message_type) pairs for aborted procedures.
    pub fn abort_by_psi(&mut self, psi: u8) -> Vec<(u8, SmMessageType)> {
        let mut aborted = Vec::new();

        for pti in PTI_MIN..=PTI_MAX {
            let pt = &self.transactions[pti as usize];
            if pt.is_pending() && pt.psi == psi {
                if let Some(msg_type) = pt.message_type {
                    aborted.push((pti, msg_type));
                }
            }
        }

        // Now abort them
        for (pti, _) in &aborted {
            self.transactions[*pti as usize].reset();
        }

        aborted
    }

    /// Abort procedures by both PTI and PSI.
    ///
    /// Aborts the specified PTI and any other pending procedures for the same PSI.
    pub fn abort_by_pti_or_psi(&mut self, pti: u8, psi: u8) -> Vec<(u8, SmMessageType)> {
        let mut to_abort = std::collections::HashSet::new();

        // Find all PTIs with matching PSI
        for i in PTI_MIN..=PTI_MAX {
            let pt = &self.transactions[i as usize];
            if pt.is_pending() && pt.psi == psi {
                to_abort.insert(i);
            }
        }

        // Also include the specified PTI
        if (PTI_MIN..=PTI_MAX).contains(&pti) {
            to_abort.insert(pti);
        }

        // Abort all and collect results
        let mut aborted = Vec::new();
        for pti in to_abort {
            if let Some((msg_type, _)) = self.abort(pti) {
                aborted.push((pti, msg_type));
            }
        }

        aborted
    }

    /// Check timers and return expired PTIs.
    ///
    /// This should be called periodically to check for timer expirations.
    pub fn check_timers(&mut self) -> Vec<u8> {
        let mut expired = Vec::new();

        for pti in PTI_MIN..=PTI_MAX {
            let pt = &mut self.transactions[pti as usize];
            if pt.is_pending() {
                if let Some(timer) = pt.timer_mut() {
                    if timer.perform_tick() {
                        expired.push(pti);
                    }
                }
            }
        }

        expired
    }

    /// Get the number of pending procedure transactions.
    pub fn pending_count(&self) -> usize {
        (PTI_MIN..=PTI_MAX)
            .filter(|&pti| self.transactions[pti as usize].is_pending())
            .count()
    }

    /// Check if any procedure transaction is pending.
    pub fn has_pending(&self) -> bool {
        (PTI_MIN..=PTI_MAX).any(|pti| self.transactions[pti as usize].is_pending())
    }

    /// Get all pending PTIs.
    pub fn pending_ptis(&self) -> Vec<u8> {
        (PTI_MIN..=PTI_MAX)
            .filter(|&pti| self.transactions[pti as usize].is_pending())
            .collect()
    }
}

impl Default for ProcedureTransactionManager {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ProcedureTransactionManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pending: Vec<_> = (PTI_MIN..=PTI_MAX)
            .filter(|&pti| self.transactions[pti as usize].is_pending())
            .map(|pti| {
                let pt = &self.transactions[pti as usize];
                format!("PTI[{}]->PSI[{}]", pti, pt.psi)
            })
            .collect();

        if pending.is_empty() {
            write!(f, "ProcedureTransactionManager {{ no pending }}")
        } else {
            write!(f, "ProcedureTransactionManager {{ {pending:?} }}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pt_state_display() {
        assert_eq!(format!("{}", PtState::Inactive), "INACTIVE");
        assert_eq!(format!("{}", PtState::Pending), "PENDING");
    }

    #[test]
    fn test_procedure_transaction_new() {
        let pt = ProcedureTransaction::new(5);
        assert_eq!(pt.pti(), 5);
        assert!(pt.is_inactive());
        assert_eq!(pt.psi(), 0);
        assert!(pt.message_type().is_none());
        assert!(pt.timer().is_none());
    }

    #[test]
    fn test_procedure_transaction_start() {
        let mut pt = ProcedureTransaction::new(10);
        pt.start(3, SmMessageType::PduSessionEstablishmentRequest, SM_TIMER_T3580);

        assert!(pt.is_pending());
        assert_eq!(pt.psi(), 3);
        assert_eq!(
            pt.message_type(),
            Some(SmMessageType::PduSessionEstablishmentRequest)
        );
        assert!(pt.timer().is_some());
        assert!(pt.timer().unwrap().is_running());
    }

    #[test]
    fn test_procedure_transaction_reset() {
        let mut pt = ProcedureTransaction::new(10);
        pt.start(3, SmMessageType::PduSessionEstablishmentRequest, SM_TIMER_T3580);
        pt.reset();

        assert!(pt.is_inactive());
        assert_eq!(pt.psi(), 0);
        assert!(pt.message_type().is_none());
        assert!(pt.timer().is_none());
    }

    #[test]
    fn test_manager_allocate() {
        let mut manager = ProcedureTransactionManager::new();

        // First allocation should return PTI 1
        let pti1 = manager.allocate();
        assert_eq!(pti1, Some(1));

        // Mark it as pending
        if let Some(pt) = manager.get_mut(1) {
            pt.start(1, SmMessageType::PduSessionEstablishmentRequest, SM_TIMER_T3580);
        }

        // Second allocation should return PTI 2
        let pti2 = manager.allocate();
        assert_eq!(pti2, Some(2));
    }

    #[test]
    fn test_manager_free() {
        let mut manager = ProcedureTransactionManager::new();

        let pti = manager.allocate().unwrap();
        if let Some(pt) = manager.get_mut(pti) {
            pt.start(1, SmMessageType::PduSessionEstablishmentRequest, SM_TIMER_T3580);
        }

        assert!(manager.get(pti).unwrap().is_pending());

        manager.free(pti);
        assert!(manager.get(pti).unwrap().is_inactive());
    }

    #[test]
    fn test_manager_validate_pti_psi() {
        let mut manager = ProcedureTransactionManager::new();

        let pti = manager.allocate().unwrap();
        if let Some(pt) = manager.get_mut(pti) {
            pt.start(5, SmMessageType::PduSessionEstablishmentRequest, SM_TIMER_T3580);
        }

        // Valid case
        assert_eq!(
            manager.validate_pti_psi(pti, 5),
            PtiValidationResult::Valid
        );

        // Invalid PTI
        assert_eq!(
            manager.validate_pti_psi(0, 5),
            PtiValidationResult::InvalidPti
        );

        // PSI mismatch
        assert_eq!(
            manager.validate_pti_psi(pti, 3),
            PtiValidationResult::PsiMismatch {
                expected: 5,
                received: 3
            }
        );
    }

    #[test]
    fn test_manager_find_by_psi() {
        let mut manager = ProcedureTransactionManager::new();

        // Create two procedures for PSI 5
        let pti1 = manager.allocate().unwrap();
        if let Some(pt) = manager.get_mut(pti1) {
            pt.start(5, SmMessageType::PduSessionEstablishmentRequest, SM_TIMER_T3580);
        }

        let pti2 = manager.allocate().unwrap();
        if let Some(pt) = manager.get_mut(pti2) {
            pt.start(5, SmMessageType::PduSessionReleaseRequest, SM_TIMER_T3582);
        }

        // Create one for PSI 3
        let pti3 = manager.allocate().unwrap();
        if let Some(pt) = manager.get_mut(pti3) {
            pt.start(3, SmMessageType::PduSessionEstablishmentRequest, SM_TIMER_T3580);
        }

        let found = manager.find_by_psi(5);
        assert_eq!(found.len(), 2);
        assert!(found.contains(&pti1));
        assert!(found.contains(&pti2));
    }

    #[test]
    fn test_manager_abort() {
        let mut manager = ProcedureTransactionManager::new();

        let pti = manager.allocate().unwrap();
        if let Some(pt) = manager.get_mut(pti) {
            pt.start(5, SmMessageType::PduSessionEstablishmentRequest, SM_TIMER_T3580);
        }

        let result = manager.abort(pti);
        assert_eq!(
            result,
            Some((SmMessageType::PduSessionEstablishmentRequest, 5))
        );
        assert!(manager.get(pti).unwrap().is_inactive());
    }

    #[test]
    fn test_manager_abort_by_psi() {
        let mut manager = ProcedureTransactionManager::new();

        // Create two procedures for PSI 5
        let pti1 = manager.allocate().unwrap();
        if let Some(pt) = manager.get_mut(pti1) {
            pt.start(5, SmMessageType::PduSessionEstablishmentRequest, SM_TIMER_T3580);
        }

        let pti2 = manager.allocate().unwrap();
        if let Some(pt) = manager.get_mut(pti2) {
            pt.start(5, SmMessageType::PduSessionReleaseRequest, SM_TIMER_T3582);
        }

        let aborted = manager.abort_by_psi(5);
        assert_eq!(aborted.len(), 2);

        assert!(manager.get(pti1).unwrap().is_inactive());
        assert!(manager.get(pti2).unwrap().is_inactive());
    }

    #[test]
    fn test_manager_pending_count() {
        let mut manager = ProcedureTransactionManager::new();

        assert_eq!(manager.pending_count(), 0);
        assert!(!manager.has_pending());

        let pti1 = manager.allocate().unwrap();
        if let Some(pt) = manager.get_mut(pti1) {
            pt.start(1, SmMessageType::PduSessionEstablishmentRequest, SM_TIMER_T3580);
        }

        assert_eq!(manager.pending_count(), 1);
        assert!(manager.has_pending());

        let pti2 = manager.allocate().unwrap();
        if let Some(pt) = manager.get_mut(pti2) {
            pt.start(2, SmMessageType::PduSessionReleaseRequest, SM_TIMER_T3582);
        }

        assert_eq!(manager.pending_count(), 2);
    }

    #[test]
    fn test_manager_pending_ptis() {
        let mut manager = ProcedureTransactionManager::new();

        let pti1 = manager.allocate().unwrap();
        if let Some(pt) = manager.get_mut(pti1) {
            pt.start(1, SmMessageType::PduSessionEstablishmentRequest, SM_TIMER_T3580);
        }

        let pti2 = manager.allocate().unwrap();
        if let Some(pt) = manager.get_mut(pti2) {
            pt.start(2, SmMessageType::PduSessionReleaseRequest, SM_TIMER_T3582);
        }

        let pending = manager.pending_ptis();
        assert_eq!(pending.len(), 2);
        assert!(pending.contains(&pti1));
        assert!(pending.contains(&pti2));
    }

    #[test]
    fn test_sm_message_type_display() {
        assert_eq!(
            format!("{}", SmMessageType::PduSessionEstablishmentRequest),
            "PDU_SESSION_ESTABLISHMENT_REQUEST"
        );
        assert_eq!(
            format!("{}", SmMessageType::PduSessionModificationRequest),
            "PDU_SESSION_MODIFICATION_REQUEST"
        );
        assert_eq!(
            format!("{}", SmMessageType::PduSessionReleaseRequest),
            "PDU_SESSION_RELEASE_REQUEST"
        );
    }

    #[test]
    fn test_procedure_transaction_clear_psi() {
        let mut pt = ProcedureTransaction::new(10);
        pt.start(5, SmMessageType::PduSessionReleaseRequest, SM_TIMER_T3582);

        assert_eq!(pt.psi(), 5);
        pt.clear_psi();
        assert_eq!(pt.psi(), 0);
        // Should still be pending
        assert!(pt.is_pending());
    }
}
