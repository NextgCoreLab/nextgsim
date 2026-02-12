//! Status Reporting for UE Application
//!
//! This module provides status tracking and reporting for the UE.
//! It maintains the operational status of various UE components and
//! provides serialization for CLI responses.
//!
//! # Reference
//!
//! Based on UERANSIM's `src/ue/types.hpp` (`UeStatusInfo`) and
//! `src/ue/app/task.cpp` (status update handling).

use serde::{Deserialize, Serialize};

use crate::nas::mm::{MmState, MmSubState, RmState};
use crate::tasks::{CmState, UeStatusUpdate};

/// UE operational status information.
///
/// This structure tracks the operational status of various UE components.
/// It is updated by status messages from other tasks and can be serialized
/// for CLI responses.
///
/// # Reference
///
/// Based on UERANSIM's `UeStatusInfo` from `src/ue/types.hpp`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UeStatusInfo {
    /// Registration Management state
    #[serde(rename = "rm-state")]
    pub rm_state: String,
    /// Mobility Management state
    #[serde(rename = "mm-state")]
    pub mm_state: String,
    /// Mobility Management sub-state
    #[serde(rename = "mm-sub-state")]
    pub mm_sub_state: String,
    /// Connection Management state
    #[serde(rename = "cm-state")]
    pub cm_state: String,
    /// Active PDU sessions (list of PSIs)
    #[serde(rename = "pdu-sessions")]
    pub pdu_sessions: Vec<u8>,
}

impl Default for UeStatusInfo {
    fn default() -> Self {
        Self {
            rm_state: RmState::default().to_string(),
            mm_state: MmState::default().to_string(),
            mm_sub_state: MmSubState::default().to_string(),
            cm_state: CmState::Idle.to_string(),
            pdu_sessions: Vec::new(),
        }
    }
}

impl UeStatusInfo {
    /// Creates a new status info with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Updates the RM state.
    pub fn set_rm_state(&mut self, state: RmState) {
        self.rm_state = state.to_string();
    }

    /// Updates the MM state.
    pub fn set_mm_state(&mut self, state: MmState) {
        self.mm_state = state.to_string();
    }

    /// Updates the MM sub-state.
    pub fn set_mm_sub_state(&mut self, state: MmSubState) {
        self.mm_sub_state = state.to_string();
    }

    /// Updates the CM state.
    pub fn set_cm_state(&mut self, state: CmState) {
        self.cm_state = state.to_string();
    }

    /// Applies a status update to the status info.
    ///
    /// # Arguments
    ///
    /// * `update` - The status update to apply
    pub fn apply_update(&mut self, update: &UeStatusUpdate) {
        match update {
            UeStatusUpdate::SessionEstablishment { psi } => {
                let psi_u8 = *psi as u8;
                if !self.pdu_sessions.contains(&psi_u8) {
                    self.pdu_sessions.push(psi_u8);
                    self.pdu_sessions.sort();
                }
            }
            UeStatusUpdate::SessionRelease { psi } => {
                let psi_u8 = *psi as u8;
                self.pdu_sessions.retain(|&p| p != psi_u8);
            }
            UeStatusUpdate::CmStateChanged { cm_state } => {
                self.cm_state = cm_state.to_string();
            }
        }
    }

    /// Returns the status as a YAML string for CLI output.
    ///
    /// # Returns
    ///
    /// A YAML-formatted string representation of the status.
    pub fn to_yaml(&self) -> Result<String, serde_yaml::Error> {
        serde_yaml::to_string(self)
    }

    /// Returns the status as a JSON string.
    ///
    /// # Returns
    ///
    /// A JSON-formatted string representation of the status.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Returns the number of active PDU sessions.
    pub fn session_count(&self) -> usize {
        self.pdu_sessions.len()
    }

    /// Returns true if the UE is registered.
    pub fn is_registered(&self) -> bool {
        self.rm_state == RmState::Registered.to_string()
    }

    /// Returns true if the UE is connected (CM-CONNECTED).
    pub fn is_connected(&self) -> bool {
        self.cm_state == CmState::Connected.to_string()
    }
}

/// Status reporter for collecting and formatting UE status.
///
/// This struct provides methods for collecting status from various
/// UE components and formatting it for display.
#[derive(Debug)]
pub struct StatusReporter {
    /// Current status info
    status: UeStatusInfo,
}

impl StatusReporter {
    /// Creates a new status reporter.
    pub fn new() -> Self {
        Self {
            status: UeStatusInfo::new(),
        }
    }

    /// Returns a reference to the current status.
    pub fn status(&self) -> &UeStatusInfo {
        &self.status
    }

    /// Returns a mutable reference to the current status.
    pub fn status_mut(&mut self) -> &mut UeStatusInfo {
        &mut self.status
    }

    /// Applies a status update.
    ///
    /// # Arguments
    ///
    /// * `update` - The status update to apply
    pub fn apply_update(&mut self, update: &UeStatusUpdate) {
        self.status.apply_update(update);
    }

    /// Updates the RM state.
    pub fn set_rm_state(&mut self, state: RmState) {
        self.status.set_rm_state(state);
    }

    /// Updates the MM state.
    pub fn set_mm_state(&mut self, state: MmState) {
        self.status.set_mm_state(state);
    }

    /// Updates the MM sub-state.
    pub fn set_mm_sub_state(&mut self, state: MmSubState) {
        self.status.set_mm_sub_state(state);
    }

    /// Updates the CM state.
    pub fn set_cm_state(&mut self, state: CmState) {
        self.status.set_cm_state(state);
    }

    /// Generates a status report as YAML.
    ///
    /// # Returns
    ///
    /// A YAML-formatted status report string.
    pub fn report_yaml(&self) -> Result<String, serde_yaml::Error> {
        self.status.to_yaml()
    }

    /// Generates a status report as JSON.
    ///
    /// # Returns
    ///
    /// A JSON-formatted status report string.
    pub fn report_json(&self) -> Result<String, serde_json::Error> {
        self.status.to_json()
    }
}

impl Default for StatusReporter {
    fn default() -> Self {
        Self::new()
    }
}

/// UE basic information for CLI display.
///
/// This structure contains static UE information from configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct UeInfo {
    /// SUPI (Subscriber Permanent Identifier) if configured
    #[serde(rename = "supi", skip_serializing_if = "Option::is_none")]
    pub supi: Option<String>,
    /// IMEI (International Mobile Equipment Identity) if configured
    #[serde(rename = "imei", skip_serializing_if = "Option::is_none")]
    pub imei: Option<String>,
    /// Active PDU sessions (list of PSIs)
    #[serde(rename = "pdu-sessions")]
    pub pdu_sessions: Vec<u8>,
    /// Number of pending SM procedures
    #[serde(rename = "pending-procedures")]
    pub pending_procedures: usize,
}


impl UeInfo {
    /// Creates a new `UeInfo` with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the SUPI from configuration.
    pub fn set_supi(&mut self, supi: impl Into<String>) {
        self.supi = Some(supi.into());
    }

    /// Sets the IMEI from configuration.
    pub fn set_imei(&mut self, imei: impl Into<String>) {
        self.imei = Some(imei.into());
    }

    /// Returns the info as a YAML string for CLI output.
    pub fn to_yaml(&self) -> Result<String, serde_yaml::Error> {
        serde_yaml::to_string(self)
    }

    /// Returns the info as a JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

/// Timer information for CLI display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimerInfo {
    /// Timer code (e.g., 3511, 3521, 3580)
    #[serde(rename = "code")]
    pub code: u16,
    /// Timer name (e.g., "T3511", "T3521")
    #[serde(rename = "name")]
    pub name: String,
    /// Whether the timer is currently running
    #[serde(rename = "running")]
    pub running: bool,
    /// Remaining time in seconds (if running)
    #[serde(rename = "remaining-secs", skip_serializing_if = "Option::is_none")]
    pub remaining_secs: Option<u32>,
    /// Expiry count (number of times the timer has expired)
    #[serde(rename = "expiry-count")]
    pub expiry_count: u32,
}

impl TimerInfo {
    /// Creates a new `TimerInfo`.
    pub fn new(code: u16, name: impl Into<String>, running: bool, expiry_count: u32) -> Self {
        Self {
            code,
            name: name.into(),
            running,
            remaining_secs: None,
            expiry_count,
        }
    }
}

/// Collection of timer information for CLI display.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TimersInfo {
    /// List of active timers
    #[serde(rename = "timers")]
    pub timers: Vec<TimerInfo>,
}

impl TimersInfo {
    /// Creates a new empty `TimersInfo`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a timer to the list.
    pub fn add_timer(&mut self, timer: TimerInfo) {
        self.timers.push(timer);
    }

    /// Returns the info as a YAML string for CLI output.
    pub fn to_yaml(&self) -> Result<String, serde_yaml::Error> {
        if self.timers.is_empty() {
            Ok("timers: []\n".to_string())
        } else {
            serde_yaml::to_string(self)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ue_status_info_default() {
        let status = UeStatusInfo::new();
        assert_eq!(status.rm_state, "RM-DEREGISTERED");
        assert_eq!(status.mm_state, MmState::default().to_string());
        assert_eq!(status.mm_sub_state, MmSubState::default().to_string());
        assert_eq!(status.cm_state, "CM-IDLE");
        assert!(status.pdu_sessions.is_empty());
    }

    #[test]
    fn test_ue_status_info_set_states() {
        let mut status = UeStatusInfo::new();
        
        status.set_rm_state(RmState::Registered);
        assert_eq!(status.rm_state, "RM-REGISTERED");
        
        status.set_mm_state(MmState::Registered);
        assert_eq!(status.mm_state, "5GMM-REGISTERED");
        
        status.set_mm_sub_state(MmSubState::RegisteredNormalService);
        assert_eq!(status.mm_sub_state, "5GMM-REGISTERED.NORMAL-SERVICE");
        
        status.set_cm_state(CmState::Connected);
        assert_eq!(status.cm_state, "CM-CONNECTED");
    }

    #[test]
    fn test_ue_status_info_apply_session_establishment() {
        let mut status = UeStatusInfo::new();
        
        let update = UeStatusUpdate::SessionEstablishment { psi: 1 };
        status.apply_update(&update);
        assert_eq!(status.pdu_sessions, vec![1]);
        
        let update = UeStatusUpdate::SessionEstablishment { psi: 3 };
        status.apply_update(&update);
        assert_eq!(status.pdu_sessions, vec![1, 3]);
        
        // Duplicate should not be added
        let update = UeStatusUpdate::SessionEstablishment { psi: 1 };
        status.apply_update(&update);
        assert_eq!(status.pdu_sessions, vec![1, 3]);
    }

    #[test]
    fn test_ue_status_info_apply_session_release() {
        let mut status = UeStatusInfo::new();
        status.pdu_sessions = vec![1, 2, 3];
        
        let update = UeStatusUpdate::SessionRelease { psi: 2 };
        status.apply_update(&update);
        assert_eq!(status.pdu_sessions, vec![1, 3]);
    }

    #[test]
    fn test_ue_status_info_apply_cm_state_changed() {
        let mut status = UeStatusInfo::new();
        
        let update = UeStatusUpdate::CmStateChanged { cm_state: CmState::Connected };
        status.apply_update(&update);
        assert_eq!(status.cm_state, "CM-CONNECTED");
    }

    #[test]
    fn test_ue_status_info_to_yaml() {
        let mut status = UeStatusInfo::new();
        status.set_rm_state(RmState::Registered);
        status.pdu_sessions = vec![1, 5];

        let yaml = status.to_yaml().unwrap();
        assert!(yaml.contains("rm-state: RM-REGISTERED"));
        assert!(yaml.contains("pdu-sessions:"));
    }

    #[test]
    fn test_ue_status_info_to_json() {
        let mut status = UeStatusInfo::new();
        status.set_rm_state(RmState::Registered);

        let json = status.to_json().unwrap();
        assert!(json.contains("\"rm-state\": \"RM-REGISTERED\""));
    }

    #[test]
    fn test_ue_status_info_helpers() {
        let mut status = UeStatusInfo::new();
        
        assert!(!status.is_registered());
        assert!(!status.is_connected());
        assert_eq!(status.session_count(), 0);
        
        status.set_rm_state(RmState::Registered);
        status.set_cm_state(CmState::Connected);
        status.pdu_sessions = vec![1, 2];
        
        assert!(status.is_registered());
        assert!(status.is_connected());
        assert_eq!(status.session_count(), 2);
    }

    #[test]
    fn test_status_reporter_new() {
        let reporter = StatusReporter::new();
        assert!(!reporter.status().is_registered());
    }

    #[test]
    fn test_status_reporter_apply_update() {
        let mut reporter = StatusReporter::new();

        let update = UeStatusUpdate::SessionEstablishment { psi: 1 };
        reporter.apply_update(&update);

        assert_eq!(reporter.status().pdu_sessions, vec![1]);
    }

    #[test]
    fn test_status_reporter_set_states() {
        let mut reporter = StatusReporter::new();
        
        reporter.set_rm_state(RmState::Registered);
        reporter.set_mm_state(MmState::Registered);
        reporter.set_mm_sub_state(MmSubState::RegisteredNormalService);
        reporter.set_cm_state(CmState::Connected);
        
        assert!(reporter.status().is_registered());
        assert!(reporter.status().is_connected());
    }

    #[test]
    fn test_status_reporter_report_yaml() {
        let mut reporter = StatusReporter::new();
        reporter.status_mut().set_rm_state(RmState::Registered);

        let yaml = reporter.report_yaml().unwrap();
        assert!(yaml.contains("rm-state: RM-REGISTERED"));
    }

    #[test]
    fn test_status_reporter_report_json() {
        let mut reporter = StatusReporter::new();
        reporter.status_mut().set_rm_state(RmState::Registered);

        let json = reporter.report_json().unwrap();
        assert!(json.contains("\"rm-state\": \"RM-REGISTERED\""));
    }
}
