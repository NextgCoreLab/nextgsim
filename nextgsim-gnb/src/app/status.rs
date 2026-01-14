//! Status Reporting for gNB Application
//!
//! This module provides status tracking and reporting for the gNB.
//! It maintains the operational status of various gNB components and
//! provides serialization for CLI responses.
//!
//! # Reference
//!
//! Based on UERANSIM's `src/gnb/types.hpp` (GnbStatusInfo) and
//! `src/gnb/app/task.cpp` (status update handling).

use serde::{Deserialize, Serialize};

use crate::tasks::{StatusType, StatusUpdate};

/// gNB operational status information.
///
/// This structure tracks the operational status of various gNB components.
/// It is updated by status messages from other tasks and can be serialized
/// for CLI responses.
///
/// # Reference
///
/// Based on UERANSIM's `GnbStatusInfo` from `src/gnb/types.hpp`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GnbStatusInfo {
    /// Whether the NGAP connection to AMF is up
    #[serde(rename = "is-ngap-up")]
    pub is_ngap_up: bool,
}

impl GnbStatusInfo {
    /// Creates a new status info with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Applies a status update to the status info.
    ///
    /// # Arguments
    ///
    /// * `update` - The status update to apply
    pub fn apply_update(&mut self, update: &StatusUpdate) {
        match update.status_type {
            StatusType::NgapIsUp => {
                self.is_ngap_up = update.value;
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
}

/// Status reporter for collecting and formatting gNB status.
///
/// This struct provides methods for collecting status from various
/// gNB components and formatting it for display.
#[derive(Debug)]
pub struct StatusReporter {
    /// Current status info
    status: GnbStatusInfo,
}

impl StatusReporter {
    /// Creates a new status reporter.
    pub fn new() -> Self {
        Self {
            status: GnbStatusInfo::new(),
        }
    }

    /// Returns a reference to the current status.
    pub fn status(&self) -> &GnbStatusInfo {
        &self.status
    }

    /// Returns a mutable reference to the current status.
    pub fn status_mut(&mut self) -> &mut GnbStatusInfo {
        &mut self.status
    }

    /// Applies a status update.
    ///
    /// # Arguments
    ///
    /// * `update` - The status update to apply
    pub fn apply_update(&mut self, update: &StatusUpdate) {
        self.status.apply_update(update);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gnb_status_info_default() {
        let status = GnbStatusInfo::new();
        assert!(!status.is_ngap_up);
    }

    #[test]
    fn test_gnb_status_info_apply_update() {
        let mut status = GnbStatusInfo::new();
        assert!(!status.is_ngap_up);

        // Apply NGAP up update
        let update = StatusUpdate {
            status_type: StatusType::NgapIsUp,
            value: true,
        };
        status.apply_update(&update);
        assert!(status.is_ngap_up);

        // Apply NGAP down update
        let update = StatusUpdate {
            status_type: StatusType::NgapIsUp,
            value: false,
        };
        status.apply_update(&update);
        assert!(!status.is_ngap_up);
    }

    #[test]
    fn test_gnb_status_info_to_yaml() {
        let mut status = GnbStatusInfo::new();
        status.is_ngap_up = true;

        let yaml = status.to_yaml().unwrap();
        assert!(yaml.contains("is-ngap-up: true"));
    }

    #[test]
    fn test_gnb_status_info_to_json() {
        let mut status = GnbStatusInfo::new();
        status.is_ngap_up = true;

        let json = status.to_json().unwrap();
        assert!(json.contains("\"is-ngap-up\": true"));
    }

    #[test]
    fn test_status_reporter_new() {
        let reporter = StatusReporter::new();
        assert!(!reporter.status().is_ngap_up);
    }

    #[test]
    fn test_status_reporter_apply_update() {
        let mut reporter = StatusReporter::new();

        let update = StatusUpdate {
            status_type: StatusType::NgapIsUp,
            value: true,
        };
        reporter.apply_update(&update);

        assert!(reporter.status().is_ngap_up);
    }

    #[test]
    fn test_status_reporter_report_yaml() {
        let mut reporter = StatusReporter::new();
        reporter.status_mut().is_ngap_up = true;

        let yaml = reporter.report_yaml().unwrap();
        assert!(yaml.contains("is-ngap-up: true"));
    }

    #[test]
    fn test_status_reporter_report_json() {
        let mut reporter = StatusReporter::new();
        reporter.status_mut().is_ngap_up = true;

        let json = reporter.report_json().unwrap();
        assert!(json.contains("\"is-ngap-up\": true"));
    }
}
