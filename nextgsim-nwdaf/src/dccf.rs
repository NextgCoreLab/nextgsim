//! Data Collection Coordination Function (DCCF)
//!
//! DCCF coordinates data collection across multiple data sources (gNBs, UEs, NFs)
//! and provides unified data access for NWDAF analytics. Introduced in Rel-17.
//!
//! # 3GPP Reference
//!
//! - TS 23.288 Section 6.2.3: NWDAF containing DCCF

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};

use crate::data_collection::{DataSourceType, MeasurementCapability};
use crate::error::{DataCollectionError, NwdafError};

/// Data collection filter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataCollectionFilter {
    /// Source types to include
    pub source_types: Vec<DataSourceType>,
    /// Measurement capabilities required
    pub capabilities: Vec<MeasurementCapability>,
    /// Geographic area filter (optional)
    pub area_of_interest: Option<GeographicArea>,
    /// Time window for data collection
    pub time_window: Option<(u64, u64)>, // (start_ms, end_ms)
    /// Sampling rate (0.0 to 1.0, 1.0 = all data)
    pub sampling_rate: f32,
}

impl Default for DataCollectionFilter {
    fn default() -> Self {
        Self {
            source_types: vec![],
            capabilities: vec![],
            area_of_interest: None,
            time_window: None,
            sampling_rate: 1.0,
        }
    }
}

/// Geographic area specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GeographicArea {
    /// Circular area defined by center and radius
    Circle {
        /// Center latitude
        lat: f64,
        /// Center longitude
        lon: f64,
        /// Radius in meters
        radius_m: f64,
    },
    /// Rectangular area
    Rectangle {
        /// Southwest corner latitude
        lat_sw: f64,
        /// Southwest corner longitude
        lon_sw: f64,
        /// Northeast corner latitude
        lat_ne: f64,
        /// Northeast corner longitude
        lon_ne: f64,
    },
    /// Specific cell IDs
    Cells {
        /// List of cell IDs
        cell_ids: Vec<i32>,
    },
}

/// Data collection session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataCollectionSession {
    /// Session identifier
    pub session_id: String,
    /// Filter for this session
    pub filter: DataCollectionFilter,
    /// Participating data sources
    pub sources: Vec<String>,
    /// Session status
    pub status: SessionStatus,
    /// Creation time
    pub created_ms: u64,
    /// Expiration time
    pub expires_ms: Option<u64>,
    /// Collected data count
    pub data_count: usize,
}

/// Session status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionStatus {
    /// Session is active and collecting data
    Active,
    /// Session is paused
    Paused,
    /// Session has completed
    Completed,
    /// Session was cancelled
    Cancelled,
}

/// Data aggregation method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationMethod {
    /// No aggregation (raw data)
    None,
    /// Average values over time window
    Average,
    /// Sum values over time window
    Sum,
    /// Minimum value
    Min,
    /// Maximum value
    Max,
    /// Count of samples
    Count,
    /// Percentiles (e.g., p50, p95, p99)
    Percentile(u8),
}

/// Aggregated data result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedData {
    /// Data source ID
    pub source_id: String,
    /// Metric name
    pub metric: String,
    /// Aggregation method used
    pub method: AggregationMethod,
    /// Aggregated value
    pub value: f64,
    /// Number of samples aggregated
    pub sample_count: usize,
    /// Time window
    pub time_window: (u64, u64),
}

/// Data Collection Coordination Function
///
/// Manages data collection sessions, coordinates multiple data sources,
/// and provides aggregated views of collected data for NWDAF analytics.
#[derive(Debug)]
pub struct Dccf {
    /// Active data collection sessions
    sessions: HashMap<String, DataCollectionSession>,
    /// Registered data sources
    sources: HashMap<String, DccfDataSource>,
    /// Next session ID counter
    next_session_id: u64,
}

/// DCCF data source descriptor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DccfDataSource {
    /// Source identifier
    pub source_id: String,
    /// Source type
    pub source_type: DataSourceType,
    /// Capabilities
    pub capabilities: Vec<MeasurementCapability>,
    /// Whether source is active
    pub is_active: bool,
    /// Last data timestamp
    pub last_data_ms: u64,
}

impl Dccf {
    /// Creates a new DCCF instance
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
            sources: HashMap::new(),
            next_session_id: 1,
        }
    }

    /// Registers a data source with DCCF
    pub fn register_source(
        &mut self,
        source_id: String,
        source_type: DataSourceType,
        capabilities: Vec<MeasurementCapability>,
    ) -> Result<(), NwdafError> {
        if self.sources.contains_key(&source_id) {
            return Err(DataCollectionError::AlreadyRegistered { source_id }.into());
        }

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let source = DccfDataSource {
            source_id: source_id.clone(),
            source_type,
            capabilities,
            is_active: true,
            last_data_ms: now_ms,
        };

        info!(
            "DCCF: Registered data source {} (type: {:?})",
            source_id, source_type
        );

        self.sources.insert(source_id, source);
        Ok(())
    }

    /// Unregisters a data source
    pub fn unregister_source(&mut self, source_id: &str) -> Result<(), NwdafError> {
        self.sources
            .remove(source_id)
            .ok_or_else(|| DataCollectionError::SourceNotFound {
                source_id: source_id.to_string(),
            })?;

        info!("DCCF: Unregistered data source {}", source_id);
        Ok(())
    }

    /// Creates a new data collection session
    pub fn create_session(&mut self, filter: DataCollectionFilter) -> DataCollectionSession {
        let session_id = format!("dccf-session-{}", self.next_session_id);
        self.next_session_id += 1;

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        // Find matching sources based on filter
        let matching_sources: Vec<String> = self
            .sources
            .values()
            .filter(|s| {
                // Check source type
                if !filter.source_types.is_empty()
                    && !filter.source_types.contains(&s.source_type)
                {
                    return false;
                }

                // Check capabilities
                if !filter.capabilities.is_empty() {
                    let has_all_caps = filter
                        .capabilities
                        .iter()
                        .all(|cap| s.capabilities.contains(cap));
                    if !has_all_caps {
                        return false;
                    }
                }

                s.is_active
            })
            .map(|s| s.source_id.clone())
            .collect();

        let session = DataCollectionSession {
            session_id: session_id.clone(),
            filter,
            sources: matching_sources.clone(),
            status: SessionStatus::Active,
            created_ms: now_ms,
            expires_ms: None,
            data_count: 0,
        };

        info!(
            "DCCF: Created session {} with {} sources",
            session_id,
            matching_sources.len()
        );

        self.sessions.insert(session_id.clone(), session.clone());
        session
    }

    /// Gets a session by ID
    pub fn get_session(&self, session_id: &str) -> Option<&DataCollectionSession> {
        self.sessions.get(session_id)
    }

    /// Updates session status
    pub fn update_session_status(
        &mut self,
        session_id: &str,
        status: SessionStatus,
    ) -> Result<(), NwdafError> {
        let session = self.sessions.get_mut(session_id).ok_or_else(|| {
            DataCollectionError::InvalidData {
                reason: format!("Session {session_id} not found"),
            }
        })?;

        debug!("DCCF: Session {} status: {:?} -> {:?}",
            session_id, session.status, status);

        session.status = status;
        Ok(())
    }

    /// Records data arrival for a session
    pub fn record_data(&mut self, session_id: &str, count: usize) -> Result<(), NwdafError> {
        let session = self.sessions.get_mut(session_id).ok_or_else(|| {
            DataCollectionError::InvalidData {
                reason: format!("Session {session_id} not found"),
            }
        })?;

        session.data_count += count;
        Ok(())
    }

    /// Aggregates data according to the specified method
    ///
    /// In a real implementation, this would query the underlying data store.
    /// This is a simplified version that returns mock aggregated data.
    pub fn aggregate_data(
        &self,
        session_id: &str,
        metric: &str,
        method: AggregationMethod,
    ) -> Result<Vec<AggregatedData>, NwdafError> {
        let session = self.get_session(session_id).ok_or_else(|| {
            DataCollectionError::InvalidData {
                reason: format!("Session {session_id} not found"),
            }
        })?;

        if session.status != SessionStatus::Active {
            return Err(DataCollectionError::InvalidData {
                reason: format!("Session {session_id} is not active"),
            }
            .into());
        }

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let time_window = session
            .filter
            .time_window
            .unwrap_or((session.created_ms, now_ms));

        // Compute per-source aggregated values from data count and source metadata
        let per_source_count = session.data_count / session.sources.len().max(1);
        let results: Vec<AggregatedData> = session
            .sources
            .iter()
            .enumerate()
            .map(|(idx, source_id)| {
                // Derive value from source index, data count, and aggregation method
                // In production this would query the underlying time-series store
                let base_value = if per_source_count > 0 {
                    // Use source index to create variance across sources
                    let seed = (idx as f64 + 1.0) / (session.sources.len() as f64 + 1.0);
                    match method {
                        AggregationMethod::Average => seed * 0.8 + 0.1,
                        AggregationMethod::Sum => seed * per_source_count as f64,
                        AggregationMethod::Min => seed * 0.3,
                        AggregationMethod::Max => 0.7 + seed * 0.3,
                        AggregationMethod::Count => per_source_count as f64,
                        AggregationMethod::Percentile(p) => seed * (p as f64 / 100.0),
                        AggregationMethod::None => seed,
                    }
                } else {
                    0.0
                };

                AggregatedData {
                    source_id: source_id.clone(),
                    metric: metric.to_string(),
                    method,
                    value: base_value,
                    sample_count: per_source_count,
                    time_window,
                }
            })
            .collect();

        debug!(
            "DCCF: Aggregated {} metric for session {} ({} sources)",
            metric,
            session_id,
            results.len()
        );

        Ok(results)
    }

    /// Lists all active sessions
    pub fn list_sessions(&self) -> Vec<&DataCollectionSession> {
        self.sessions.values().collect()
    }

    /// Lists all registered data sources
    pub fn list_sources(&self) -> Vec<&DccfDataSource> {
        self.sources.values().collect()
    }

    /// Returns the number of active sessions
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Returns the number of registered sources
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Closes a data collection session
    pub fn close_session(&mut self, session_id: &str) -> Result<(), NwdafError> {
        self.update_session_status(session_id, SessionStatus::Completed)?;
        info!("DCCF: Closed session {}", session_id);
        Ok(())
    }

    /// Applies a data transformation pipeline to collected data.
    ///
    /// Transforms raw measurements according to a chain of transformations
    /// (normalization, feature extraction, downsampling, anonymization).
    pub fn apply_transformations(
        &self,
        values: &[f64],
        transforms: &[DataTransformation],
    ) -> Vec<f64> {
        let mut result = values.to_vec();
        for transform in transforms {
            result = match transform {
                DataTransformation::Normalize => {
                    let min = result.iter().copied().fold(f64::INFINITY, f64::min);
                    let max = result.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let range = (max - min).max(f64::EPSILON);
                    result.iter().map(|v| (v - min) / range).collect()
                }
                DataTransformation::Standardize => {
                    let n = result.len() as f64;
                    let mean = result.iter().sum::<f64>() / n;
                    let variance = result.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
                    let std = variance.sqrt().max(f64::EPSILON);
                    result.iter().map(|v| (v - mean) / std).collect()
                }
                DataTransformation::Downsample { factor } => {
                    result.iter().step_by(*factor).copied().collect()
                }
                DataTransformation::MovingAverage { window } => {
                    let w = (*window).min(result.len()).max(1);
                    (0..result.len())
                        .map(|i| {
                            let start = i.saturating_sub(w - 1);
                            let slice = &result[start..=i];
                            slice.iter().sum::<f64>() / slice.len() as f64
                        })
                        .collect()
                }
                DataTransformation::Clip { min, max } => {
                    result.iter().map(|v| v.clamp(*min, *max)).collect()
                }
            };
        }
        result
    }

    /// Policy-based data routing: determines which NWDAF instances should
    /// receive data from a given source based on routing policies.
    pub fn route_data(
        &self,
        source_id: &str,
        policies: &[DataRoutingPolicy],
    ) -> Vec<String> {
        let source = match self.sources.get(source_id) {
            Some(s) => s,
            None => return vec![],
        };

        let mut destinations = Vec::new();
        for policy in policies {
            let matches = match &policy.condition {
                RoutingCondition::SourceType(st) => *st == source.source_type,
                RoutingCondition::Capability(cap) => source.capabilities.contains(cap),
                RoutingCondition::Always => true,
            };
            if matches {
                destinations.extend(policy.target_nwdaf_ids.iter().cloned());
            }
        }
        // Deduplicate
        destinations.sort();
        destinations.dedup();
        destinations
    }
}

/// Data transformation operation for DCCF pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataTransformation {
    /// Min-max normalization to [0, 1]
    Normalize,
    /// Z-score standardization (zero mean, unit variance)
    Standardize,
    /// Downsample by keeping every N-th sample
    Downsample { factor: usize },
    /// Sliding window moving average
    MovingAverage { window: usize },
    /// Clip values to [min, max] range
    Clip { min: f64, max: f64 },
}

/// Data routing policy for DCCF
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataRoutingPolicy {
    /// Condition for this policy to apply
    pub condition: RoutingCondition,
    /// Target NWDAF instance IDs to receive data
    pub target_nwdaf_ids: Vec<String>,
}

/// Condition for data routing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoutingCondition {
    /// Route data from specific source type
    SourceType(DataSourceType),
    /// Route data with specific capability
    Capability(MeasurementCapability),
    /// Always route
    Always,
}

impl Default for Dccf {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dccf_creation() {
        let dccf = Dccf::new();
        assert_eq!(dccf.source_count(), 0);
        assert_eq!(dccf.session_count(), 0);
    }

    #[test]
    fn test_register_source() {
        let mut dccf = Dccf::new();

        dccf.register_source(
            "gnb-1".to_string(),
            DataSourceType::Gnb,
            vec![MeasurementCapability::RadioMeasurement],
        )
        .unwrap();

        assert_eq!(dccf.source_count(), 1);
        assert_eq!(dccf.list_sources().len(), 1);
    }

    #[test]
    fn test_register_duplicate_source() {
        let mut dccf = Dccf::new();

        dccf.register_source(
            "gnb-1".to_string(),
            DataSourceType::Gnb,
            vec![MeasurementCapability::RadioMeasurement],
        )
        .unwrap();

        let result = dccf.register_source(
            "gnb-1".to_string(),
            DataSourceType::Gnb,
            vec![MeasurementCapability::RadioMeasurement],
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_unregister_source() {
        let mut dccf = Dccf::new();

        dccf.register_source(
            "gnb-1".to_string(),
            DataSourceType::Gnb,
            vec![MeasurementCapability::RadioMeasurement],
        )
        .unwrap();

        dccf.unregister_source("gnb-1").unwrap();
        assert_eq!(dccf.source_count(), 0);
    }

    #[test]
    fn test_create_session() {
        let mut dccf = Dccf::new();

        dccf.register_source(
            "gnb-1".to_string(),
            DataSourceType::Gnb,
            vec![MeasurementCapability::RadioMeasurement],
        )
        .unwrap();

        let filter = DataCollectionFilter {
            source_types: vec![DataSourceType::Gnb],
            capabilities: vec![MeasurementCapability::RadioMeasurement],
            ..Default::default()
        };

        let session = dccf.create_session(filter);
        assert_eq!(session.status, SessionStatus::Active);
        assert_eq!(session.sources.len(), 1);
        assert!(session.sources.contains(&"gnb-1".to_string()));

        assert_eq!(dccf.session_count(), 1);
    }

    #[test]
    fn test_update_session_status() {
        let mut dccf = Dccf::new();

        dccf.register_source(
            "gnb-1".to_string(),
            DataSourceType::Gnb,
            vec![MeasurementCapability::RadioMeasurement],
        )
        .unwrap();

        let session = dccf.create_session(DataCollectionFilter::default());
        let session_id = session.session_id.clone();

        dccf.update_session_status(&session_id, SessionStatus::Paused)
            .unwrap();

        let updated_session = dccf.get_session(&session_id).unwrap();
        assert_eq!(updated_session.status, SessionStatus::Paused);
    }

    #[test]
    fn test_record_data() {
        let mut dccf = Dccf::new();

        dccf.register_source(
            "gnb-1".to_string(),
            DataSourceType::Gnb,
            vec![MeasurementCapability::RadioMeasurement],
        )
        .unwrap();

        let session = dccf.create_session(DataCollectionFilter::default());
        let session_id = session.session_id.clone();

        dccf.record_data(&session_id, 100).unwrap();
        dccf.record_data(&session_id, 50).unwrap();

        let session = dccf.get_session(&session_id).unwrap();
        assert_eq!(session.data_count, 150);
    }

    #[test]
    fn test_aggregate_data() {
        let mut dccf = Dccf::new();

        dccf.register_source(
            "gnb-1".to_string(),
            DataSourceType::Gnb,
            vec![MeasurementCapability::RadioMeasurement],
        )
        .unwrap();

        let session = dccf.create_session(DataCollectionFilter::default());
        let session_id = session.session_id.clone();

        dccf.record_data(&session_id, 100).unwrap();

        let aggregated = dccf
            .aggregate_data(&session_id, "rsrp", AggregationMethod::Average)
            .unwrap();

        assert_eq!(aggregated.len(), 1);
        assert_eq!(aggregated[0].metric, "rsrp");
        assert_eq!(aggregated[0].method, AggregationMethod::Average);
    }

    #[test]
    fn test_close_session() {
        let mut dccf = Dccf::new();

        dccf.register_source(
            "gnb-1".to_string(),
            DataSourceType::Gnb,
            vec![MeasurementCapability::RadioMeasurement],
        )
        .unwrap();

        let session = dccf.create_session(DataCollectionFilter::default());
        let session_id = session.session_id.clone();

        dccf.close_session(&session_id).unwrap();

        let session = dccf.get_session(&session_id).unwrap();
        assert_eq!(session.status, SessionStatus::Completed);
    }

    #[test]
    fn test_filter_by_source_type() {
        let mut dccf = Dccf::new();

        dccf.register_source(
            "gnb-1".to_string(),
            DataSourceType::Gnb,
            vec![MeasurementCapability::RadioMeasurement],
        )
        .unwrap();

        dccf.register_source(
            "amf-1".to_string(),
            DataSourceType::CoreNf,
            vec![MeasurementCapability::CellLoad],
        )
        .unwrap();

        let filter = DataCollectionFilter {
            source_types: vec![DataSourceType::Gnb],
            ..Default::default()
        };

        let session = dccf.create_session(filter);
        assert_eq!(session.sources.len(), 1);
        assert!(session.sources.contains(&"gnb-1".to_string()));
    }

    #[test]
    fn test_normalize_transform() {
        let dccf = Dccf::new();
        let values = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let result = dccf.apply_transformations(&values, &[DataTransformation::Normalize]);
        assert!((result[0] - 0.0).abs() < 1e-6);
        assert!((result[4] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_standardize_transform() {
        let dccf = Dccf::new();
        let values = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let result = dccf.apply_transformations(&values, &[DataTransformation::Standardize]);
        let mean: f64 = result.iter().sum::<f64>() / result.len() as f64;
        assert!(mean.abs() < 1e-6, "Standardized mean should be ~0, got {mean}");
    }

    #[test]
    fn test_downsample_transform() {
        let dccf = Dccf::new();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = dccf.apply_transformations(&values, &[DataTransformation::Downsample { factor: 2 }]);
        assert_eq!(result, vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn test_moving_average_transform() {
        let dccf = Dccf::new();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = dccf.apply_transformations(&values, &[DataTransformation::MovingAverage { window: 3 }]);
        assert_eq!(result.len(), 5);
        assert!((result[2] - 2.0).abs() < 1e-6); // avg(1,2,3) = 2
    }

    #[test]
    fn test_chained_transforms() {
        let dccf = Dccf::new();
        let values = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let result = dccf.apply_transformations(
            &values,
            &[
                DataTransformation::Downsample { factor: 2 },
                DataTransformation::Normalize,
            ],
        );
        assert_eq!(result.len(), 3); // 6/2 = 3
        assert!((result[0] - 0.0).abs() < 1e-6);
        assert!((result[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_data_routing() {
        let mut dccf = Dccf::new();
        dccf.register_source(
            "gnb-1".to_string(),
            DataSourceType::Gnb,
            vec![MeasurementCapability::RadioMeasurement],
        ).unwrap();
        dccf.register_source(
            "amf-1".to_string(),
            DataSourceType::CoreNf,
            vec![MeasurementCapability::CellLoad],
        ).unwrap();

        let policies = vec![
            DataRoutingPolicy {
                condition: RoutingCondition::SourceType(DataSourceType::Gnb),
                target_nwdaf_ids: vec!["nwdaf-1".to_string()],
            },
            DataRoutingPolicy {
                condition: RoutingCondition::Always,
                target_nwdaf_ids: vec!["nwdaf-central".to_string()],
            },
        ];

        let destinations = dccf.route_data("gnb-1", &policies);
        assert!(destinations.contains(&"nwdaf-1".to_string()));
        assert!(destinations.contains(&"nwdaf-central".to_string()));

        let destinations = dccf.route_data("amf-1", &policies);
        assert!(!destinations.contains(&"nwdaf-1".to_string()));
        assert!(destinations.contains(&"nwdaf-central".to_string()));
    }

    #[test]
    fn test_filter_by_capability() {
        let mut dccf = Dccf::new();

        dccf.register_source(
            "gnb-1".to_string(),
            DataSourceType::Gnb,
            vec![MeasurementCapability::RadioMeasurement],
        )
        .unwrap();

        dccf.register_source(
            "gnb-2".to_string(),
            DataSourceType::Gnb,
            vec![MeasurementCapability::CellLoad],
        )
        .unwrap();

        let filter = DataCollectionFilter {
            capabilities: vec![MeasurementCapability::RadioMeasurement],
            ..Default::default()
        };

        let session = dccf.create_session(filter);
        assert_eq!(session.sources.len(), 1);
        assert!(session.sources.contains(&"gnb-1".to_string()));
    }
}
