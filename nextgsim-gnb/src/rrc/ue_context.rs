//! RRC UE Context Management
//!
//! This module manages UE contexts within the RRC task. Each UE has an associated
//! context that tracks:
//! - UE identity (initial ID, S-TMSI)
//! - RRC establishment cause
//! - RRC connection state
//!
//! # Reference
//!
//! Based on UERANSIM's `RrcUeContext` from `src/gnb/types.hpp` and
//! UE management functions from `src/gnb/rrc/ues.cpp`.

use std::collections::HashMap;

use crate::tasks::GutiMobileIdentity;
use super::redcap::RedCapProcessor;

/// RRC connection state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RrcState {
    /// Initial state - no RRC connection
    #[default]
    Idle,
    /// RRC Setup Request received, waiting for setup
    SetupRequest,
    /// RRC Setup sent, waiting for completion
    SetupSent,
    /// RRC connection established
    Connected,
    /// RRC connection being released
    Releasing,
}

impl std::fmt::Display for RrcState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RrcState::Idle => write!(f, "Idle"),
            RrcState::SetupRequest => write!(f, "SetupRequest"),
            RrcState::SetupSent => write!(f, "SetupSent"),
            RrcState::Connected => write!(f, "Connected"),
            RrcState::Releasing => write!(f, "Releasing"),
        }
    }
}

/// RRC UE context
///
/// Tracks RRC-specific information for a UE, including identity and connection state.
/// Based on UERANSIM's `RrcUeContext` from `src/gnb/types.hpp`.
#[derive(Debug, Clone)]
pub struct RrcUeContext {
    /// UE ID (internal identifier)
    pub ue_id: i32,
    /// Initial UE identity (39-bit value, or None if not set)
    /// This is either a random value or TMSI-part-1 from 5G-S-TMSI
    pub initial_id: Option<i64>,
    /// Whether the initial ID is from S-TMSI (true) or random (false)
    pub is_initial_id_s_tmsi: bool,
    /// RRC establishment cause (from `RRCSetupRequest`)
    pub establishment_cause: i64,
    /// S-TMSI if available (from `RRCSetupComplete`)
    pub s_tmsi: Option<GutiMobileIdentity>,
    /// Current RRC connection state
    pub state: RrcState,
    /// RedCap processor for this UE (Rel-17)
    pub redcap: RedCapProcessor,
}

impl RrcUeContext {
    /// Creates a new RRC UE context with the given UE ID
    pub fn new(ue_id: i32) -> Self {
        Self {
            ue_id,
            initial_id: None,
            is_initial_id_s_tmsi: false,
            establishment_cause: 0,
            s_tmsi: None,
            state: RrcState::Idle,
            redcap: RedCapProcessor::new(),
        }
    }

    /// Sets the initial UE identity
    ///
    /// # Arguments
    /// * `initial_id` - 39-bit initial UE identity
    /// * `is_s_tmsi` - true if the ID is from S-TMSI, false if random
    pub fn set_initial_id(&mut self, initial_id: i64, is_s_tmsi: bool) {
        self.initial_id = Some(initial_id);
        self.is_initial_id_s_tmsi = is_s_tmsi;
    }

    /// Sets the RRC establishment cause
    pub fn set_establishment_cause(&mut self, cause: i64) {
        self.establishment_cause = cause;
    }

    /// Sets the S-TMSI from `RRCSetupComplete`
    pub fn set_s_tmsi(&mut self, s_tmsi: GutiMobileIdentity) {
        self.s_tmsi = Some(s_tmsi);
    }

    /// Transitions to `SetupRequest` state (`RRCSetupRequest` received)
    pub fn on_setup_request(&mut self) {
        self.state = RrcState::SetupRequest;
    }

    /// Transitions to `SetupSent` state (`RRCSetup` sent)
    pub fn on_setup_sent(&mut self) {
        self.state = RrcState::SetupSent;
    }

    /// Transitions to Connected state (`RRCSetupComplete` received)
    pub fn on_setup_complete(&mut self) {
        self.state = RrcState::Connected;
    }

    /// Transitions to Releasing state (`RRCRelease` being sent)
    pub fn on_release(&mut self) {
        self.state = RrcState::Releasing;
    }

    /// Returns true if the UE has an established RRC connection
    pub fn is_connected(&self) -> bool {
        self.state == RrcState::Connected
    }

    /// Returns true if the UE is in idle state
    pub fn is_idle(&self) -> bool {
        self.state == RrcState::Idle
    }
}

/// RRC UE context manager
///
/// Manages all UE contexts within the RRC task. Provides methods to create,
/// find, and delete UE contexts.
///
/// Based on UERANSIM's UE management from `src/gnb/rrc/ues.cpp`.
#[derive(Debug, Default)]
pub struct RrcUeContextManager {
    /// UE contexts indexed by UE ID
    contexts: HashMap<i32, RrcUeContext>,
}

impl RrcUeContextManager {
    /// Creates a new empty UE context manager
    pub fn new() -> Self {
        Self {
            contexts: HashMap::new(),
        }
    }

    /// Creates a new UE context with the given ID
    ///
    /// Returns a mutable reference to the created context.
    /// If a context with the same ID already exists, it will be replaced.
    pub fn create_ue(&mut self, ue_id: i32) -> &mut RrcUeContext {
        let ctx = RrcUeContext::new(ue_id);
        self.contexts.insert(ue_id, ctx);
        self.contexts.get_mut(&ue_id).unwrap()
    }

    /// Tries to find a UE context by ID
    ///
    /// Returns `Some(&RrcUeContext)` if found, `None` otherwise.
    pub fn try_find_ue(&self, ue_id: i32) -> Option<&RrcUeContext> {
        self.contexts.get(&ue_id)
    }

    /// Tries to find a mutable UE context by ID
    ///
    /// Returns `Some(&mut RrcUeContext)` if found, `None` otherwise.
    pub fn try_find_ue_mut(&mut self, ue_id: i32) -> Option<&mut RrcUeContext> {
        self.contexts.get_mut(&ue_id)
    }

    /// Finds a UE context by ID, creating it if it doesn't exist
    ///
    /// This is useful when receiving messages from a UE that may or may not
    /// have an existing context.
    pub fn find_or_create_ue(&mut self, ue_id: i32) -> &mut RrcUeContext {
        if !self.contexts.contains_key(&ue_id) {
            self.create_ue(ue_id);
        }
        self.contexts.get_mut(&ue_id).unwrap()
    }

    /// Deletes a UE context by ID
    ///
    /// Returns the removed context if it existed.
    pub fn delete_ue(&mut self, ue_id: i32) -> Option<RrcUeContext> {
        self.contexts.remove(&ue_id)
    }

    /// Returns the number of UE contexts
    pub fn count(&self) -> usize {
        self.contexts.len()
    }

    /// Returns true if there are no UE contexts
    pub fn is_empty(&self) -> bool {
        self.contexts.is_empty()
    }

    /// Returns an iterator over all UE contexts
    pub fn iter(&self) -> impl Iterator<Item = (&i32, &RrcUeContext)> {
        self.contexts.iter()
    }

    /// Returns a mutable iterator over all UE contexts
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&i32, &mut RrcUeContext)> {
        self.contexts.iter_mut()
    }

    /// Returns all UE IDs
    pub fn ue_ids(&self) -> Vec<i32> {
        self.contexts.keys().copied().collect()
    }

    /// Returns all connected UE IDs
    pub fn connected_ue_ids(&self) -> Vec<i32> {
        self.contexts
            .iter()
            .filter(|(_, ctx)| ctx.is_connected())
            .map(|(id, _)| *id)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nextgsim_common::Plmn;

    fn create_test_s_tmsi() -> GutiMobileIdentity {
        GutiMobileIdentity {
            plmn: Plmn::new(1, 1, false),
            amf_region_id: 1,
            amf_set_id: 1,
            amf_pointer: 1,
            tmsi: 0x12345678,
        }
    }

    #[test]
    fn test_rrc_state_default() {
        let state = RrcState::default();
        assert_eq!(state, RrcState::Idle);
    }

    #[test]
    fn test_rrc_state_display() {
        assert_eq!(format!("{}", RrcState::Idle), "Idle");
        assert_eq!(format!("{}", RrcState::SetupRequest), "SetupRequest");
        assert_eq!(format!("{}", RrcState::SetupSent), "SetupSent");
        assert_eq!(format!("{}", RrcState::Connected), "Connected");
        assert_eq!(format!("{}", RrcState::Releasing), "Releasing");
    }

    #[test]
    fn test_rrc_ue_context_new() {
        let ctx = RrcUeContext::new(1);
        assert_eq!(ctx.ue_id, 1);
        assert!(ctx.initial_id.is_none());
        assert!(!ctx.is_initial_id_s_tmsi);
        assert_eq!(ctx.establishment_cause, 0);
        assert!(ctx.s_tmsi.is_none());
        assert_eq!(ctx.state, RrcState::Idle);
        assert!(ctx.is_idle());
        assert!(!ctx.is_connected());
    }

    #[test]
    fn test_rrc_ue_context_set_initial_id() {
        let mut ctx = RrcUeContext::new(1);
        
        // Set random initial ID
        ctx.set_initial_id(0x123456789, false);
        assert_eq!(ctx.initial_id, Some(0x123456789));
        assert!(!ctx.is_initial_id_s_tmsi);
        
        // Set S-TMSI initial ID
        ctx.set_initial_id(0x987654321, true);
        assert_eq!(ctx.initial_id, Some(0x987654321));
        assert!(ctx.is_initial_id_s_tmsi);
    }

    #[test]
    fn test_rrc_ue_context_set_establishment_cause() {
        let mut ctx = RrcUeContext::new(1);
        ctx.set_establishment_cause(3); // e.g., mo-Data
        assert_eq!(ctx.establishment_cause, 3);
    }

    #[test]
    fn test_rrc_ue_context_set_s_tmsi() {
        let mut ctx = RrcUeContext::new(1);
        let s_tmsi = create_test_s_tmsi();
        ctx.set_s_tmsi(s_tmsi.clone());
        assert!(ctx.s_tmsi.is_some());
        assert_eq!(ctx.s_tmsi.as_ref().unwrap().tmsi, 0x12345678);
    }

    #[test]
    fn test_rrc_ue_context_state_transitions() {
        let mut ctx = RrcUeContext::new(1);
        assert!(ctx.is_idle());
        
        ctx.on_setup_request();
        assert_eq!(ctx.state, RrcState::SetupRequest);
        assert!(!ctx.is_idle());
        assert!(!ctx.is_connected());
        
        ctx.on_setup_sent();
        assert_eq!(ctx.state, RrcState::SetupSent);
        
        ctx.on_setup_complete();
        assert_eq!(ctx.state, RrcState::Connected);
        assert!(ctx.is_connected());
        
        ctx.on_release();
        assert_eq!(ctx.state, RrcState::Releasing);
        assert!(!ctx.is_connected());
    }

    #[test]
    fn test_rrc_ue_context_manager_new() {
        let manager = RrcUeContextManager::new();
        assert_eq!(manager.count(), 0);
        assert!(manager.is_empty());
    }

    #[test]
    fn test_rrc_ue_context_manager_create_ue() {
        let mut manager = RrcUeContextManager::new();
        
        let ctx = manager.create_ue(1);
        assert_eq!(ctx.ue_id, 1);
        assert_eq!(manager.count(), 1);
        assert!(!manager.is_empty());
        
        // Creating with same ID replaces
        let ctx = manager.create_ue(1);
        ctx.set_establishment_cause(5);
        assert_eq!(manager.count(), 1);
        assert_eq!(manager.try_find_ue(1).unwrap().establishment_cause, 5);
    }

    #[test]
    fn test_rrc_ue_context_manager_try_find_ue() {
        let mut manager = RrcUeContextManager::new();
        
        assert!(manager.try_find_ue(1).is_none());
        
        manager.create_ue(1);
        assert!(manager.try_find_ue(1).is_some());
        assert_eq!(manager.try_find_ue(1).unwrap().ue_id, 1);
    }

    #[test]
    fn test_rrc_ue_context_manager_try_find_ue_mut() {
        let mut manager = RrcUeContextManager::new();
        manager.create_ue(1);
        
        let ctx = manager.try_find_ue_mut(1).unwrap();
        ctx.set_establishment_cause(7);
        
        assert_eq!(manager.try_find_ue(1).unwrap().establishment_cause, 7);
    }

    #[test]
    fn test_rrc_ue_context_manager_find_or_create_ue() {
        let mut manager = RrcUeContextManager::new();
        
        // Creates new context
        let ctx = manager.find_or_create_ue(1);
        ctx.set_establishment_cause(3);
        assert_eq!(manager.count(), 1);
        
        // Returns existing context
        let ctx = manager.find_or_create_ue(1);
        assert_eq!(ctx.establishment_cause, 3);
        assert_eq!(manager.count(), 1);
    }

    #[test]
    fn test_rrc_ue_context_manager_delete_ue() {
        let mut manager = RrcUeContextManager::new();
        manager.create_ue(1);
        manager.create_ue(2);
        assert_eq!(manager.count(), 2);
        
        let removed = manager.delete_ue(1);
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().ue_id, 1);
        assert_eq!(manager.count(), 1);
        
        // Deleting non-existent returns None
        let removed = manager.delete_ue(1);
        assert!(removed.is_none());
    }

    #[test]
    fn test_rrc_ue_context_manager_ue_ids() {
        let mut manager = RrcUeContextManager::new();
        manager.create_ue(1);
        manager.create_ue(2);
        manager.create_ue(3);
        
        let mut ids = manager.ue_ids();
        ids.sort();
        assert_eq!(ids, vec![1, 2, 3]);
    }

    #[test]
    fn test_rrc_ue_context_manager_connected_ue_ids() {
        let mut manager = RrcUeContextManager::new();
        
        manager.create_ue(1);
        manager.try_find_ue_mut(1).unwrap().on_setup_complete();
        
        manager.create_ue(2);
        // UE 2 stays in Idle
        
        manager.create_ue(3);
        manager.try_find_ue_mut(3).unwrap().on_setup_complete();
        
        let mut connected = manager.connected_ue_ids();
        connected.sort();
        assert_eq!(connected, vec![1, 3]);
    }

    #[test]
    fn test_rrc_ue_context_manager_iter() {
        let mut manager = RrcUeContextManager::new();
        manager.create_ue(1);
        manager.create_ue(2);
        
        let count = manager.iter().count();
        assert_eq!(count, 2);
    }
}
