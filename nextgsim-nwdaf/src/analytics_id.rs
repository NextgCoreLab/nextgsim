//! TS 23.288 standardized Analytics IDs
//!
//! Defines the analytics identifiers specified in 3GPP TS 23.288 Section 6.1,
//! used to identify the type of analytics requested from NWDAF.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Analytics identifiers as defined in 3GPP TS 23.288
///
/// Each variant represents a standardized analytics type that NWDAF
/// can produce, covering mobility, load, `QoS`, and behavioral analytics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnalyticsId {
    /// UE Mobility analytics (clause 6.7)
    ///
    /// Provides predictions and statistics about UE movement patterns,
    /// including trajectory prediction and location distribution.
    UeMobility,

    /// NF Load analytics (clause 6.5)
    ///
    /// Provides current and predicted load information for network functions,
    /// including CPU usage, memory usage, and request throughput.
    NfLoad,

    /// Service Experience analytics (clause 6.4)
    ///
    /// Provides quality of experience analytics for network services,
    /// including MOS scores and service-level KPIs.
    ServiceExperience,

    /// Abnormal Behaviour analytics (clause 6.9)
    ///
    /// Detects anomalous behavior patterns in UEs or network elements,
    /// using statistical analysis and ML-based detection.
    AbnormalBehavior,

    /// User Data Congestion analytics (clause 6.8)
    ///
    /// Provides congestion information for user plane traffic,
    /// including per-cell and per-slice congestion levels.
    UserDataCongestion,

    /// `QoS` Sustainability analytics (clause 6.6)
    ///
    /// Predicts whether current `QoS` levels can be maintained,
    /// considering network conditions and resource availability.
    QosSustainability,

    /// Energy Efficiency analytics (Rel-19)
    ///
    /// Provides analytics on network energy consumption and efficiency metrics,
    /// supporting IMT-2030 sustainability requirements.
    EnergyEfficiency,

    /// Network Slicing Optimization analytics (Rel-19)
    ///
    /// Provides analytics for optimizing network slice resource allocation
    /// and performance across multiple slices.
    SliceOptimization,
}

impl AnalyticsId {
    /// Returns all defined analytics IDs
    pub fn all() -> &'static [AnalyticsId] {
        &[
            AnalyticsId::UeMobility,
            AnalyticsId::NfLoad,
            AnalyticsId::ServiceExperience,
            AnalyticsId::AbnormalBehavior,
            AnalyticsId::UserDataCongestion,
            AnalyticsId::QosSustainability,
            AnalyticsId::EnergyEfficiency,
            AnalyticsId::SliceOptimization,
        ]
    }

    /// Returns the TS 23.288 clause reference for this analytics ID
    pub fn clause_reference(&self) -> &'static str {
        match self {
            AnalyticsId::UeMobility => "6.7",
            AnalyticsId::NfLoad => "6.5",
            AnalyticsId::ServiceExperience => "6.4",
            AnalyticsId::AbnormalBehavior => "6.9",
            AnalyticsId::UserDataCongestion => "6.8",
            AnalyticsId::QosSustainability => "6.6",
            AnalyticsId::EnergyEfficiency => "Rel-19",
            AnalyticsId::SliceOptimization => "Rel-19",
        }
    }

    /// Returns a human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            AnalyticsId::UeMobility => "UE mobility analytics including trajectory prediction",
            AnalyticsId::NfLoad => "Network function load analytics and predictions",
            AnalyticsId::ServiceExperience => "Service experience quality analytics",
            AnalyticsId::AbnormalBehavior => "Abnormal behaviour detection and reporting",
            AnalyticsId::UserDataCongestion => "User data congestion analytics",
            AnalyticsId::QosSustainability => "QoS sustainability prediction analytics",
            AnalyticsId::EnergyEfficiency => "Energy efficiency and power consumption analytics",
            AnalyticsId::SliceOptimization => "Network slicing optimization analytics",
        }
    }
}

impl fmt::Display for AnalyticsId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AnalyticsId::UeMobility => write!(f, "UE_MOBILITY"),
            AnalyticsId::NfLoad => write!(f, "NF_LOAD"),
            AnalyticsId::ServiceExperience => write!(f, "SERVICE_EXPERIENCE"),
            AnalyticsId::AbnormalBehavior => write!(f, "ABNORMAL_BEHAVIOUR"),
            AnalyticsId::UserDataCongestion => write!(f, "USER_DATA_CONGESTION"),
            AnalyticsId::QosSustainability => write!(f, "QOS_SUSTAINABILITY"),
            AnalyticsId::EnergyEfficiency => write!(f, "ENERGY_EFFICIENCY"),
            AnalyticsId::SliceOptimization => write!(f, "SLICE_OPTIMIZATION"),
        }
    }
}

/// Time window specification for analytics requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeWindow {
    /// Start time in milliseconds since epoch
    pub start_ms: u64,
    /// End time in milliseconds since epoch
    pub end_ms: u64,
}

impl TimeWindow {
    /// Creates a new time window
    pub fn new(start_ms: u64, end_ms: u64) -> Self {
        Self { start_ms, end_ms }
    }

    /// Duration of the window in milliseconds
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

/// Target of analytics (which entity analytics are about)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalyticsTarget {
    /// Analytics for a specific UE
    Ue {
        /// UE identifier
        ue_id: i32,
    },
    /// Analytics for a specific cell
    Cell {
        /// Cell identifier
        cell_id: i32,
    },
    /// Analytics for a network slice
    Slice {
        /// S-NSSAI identifier
        snssai: String,
    },
    /// Analytics for all entities (aggregate)
    Any,
}

impl fmt::Display for AnalyticsTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AnalyticsTarget::Ue { ue_id } => write!(f, "UE({ue_id})"),
            AnalyticsTarget::Cell { cell_id } => write!(f, "Cell({cell_id})"),
            AnalyticsTarget::Slice { snssai } => write!(f, "Slice({snssai})"),
            AnalyticsTarget::Any => write!(f, "Any"),
        }
    }
}

/// Analytics output type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnalyticsOutputType {
    /// Statistical analytics (based on historical data)
    Statistics,
    /// Predictive analytics (future projections)
    Predictions,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analytics_id_all() {
        let all = AnalyticsId::all();
        assert_eq!(all.len(), 8);
    }

    #[test]
    fn test_analytics_id_display() {
        assert_eq!(format!("{}", AnalyticsId::UeMobility), "UE_MOBILITY");
        assert_eq!(format!("{}", AnalyticsId::NfLoad), "NF_LOAD");
        assert_eq!(
            format!("{}", AnalyticsId::AbnormalBehavior),
            "ABNORMAL_BEHAVIOUR"
        );
    }

    #[test]
    fn test_analytics_id_clause_reference() {
        assert_eq!(AnalyticsId::UeMobility.clause_reference(), "6.7");
        assert_eq!(AnalyticsId::NfLoad.clause_reference(), "6.5");
    }

    #[test]
    fn test_time_window() {
        let window = TimeWindow::new(1000, 2000);
        assert_eq!(window.duration_ms(), 1000);
    }

    #[test]
    fn test_analytics_target_display() {
        assert_eq!(
            format!("{}", AnalyticsTarget::Ue { ue_id: 42 }),
            "UE(42)"
        );
        assert_eq!(
            format!("{}", AnalyticsTarget::Cell { cell_id: 7 }),
            "Cell(7)"
        );
    }

    #[test]
    fn test_analytics_id_serialization() {
        let id = AnalyticsId::QosSustainability;
        let json = serde_json::to_string(&id).expect("serialize");
        let deserialized: AnalyticsId = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(id, deserialized);
    }
}
