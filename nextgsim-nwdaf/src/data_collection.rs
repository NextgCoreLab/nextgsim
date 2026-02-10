//! Data collection and source registration for NWDAF
//!
//! Provides the `DataCollector` that manages data source registrations
//! from gNB, UE, and other network functions. Sources register themselves
//! and then push measurements which are stored in per-entity histories.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::error::DataCollectionError;
use crate::{CellLoad, UeMeasurement, UeMeasurementHistory};

/// Type of data source reporting measurements to NWDAF
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataSourceType {
    /// gNB (gNodeB) reporting cell-level measurements
    Gnb,
    /// UE reporting device-level measurements
    Ue,
    /// Core network function (AMF, SMF, etc.)
    CoreNf,
    /// OAM (Operations, Administration, and Maintenance)
    Oam,
}

impl std::fmt::Display for DataSourceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataSourceType::Gnb => write!(f, "gNB"),
            DataSourceType::Ue => write!(f, "UE"),
            DataSourceType::CoreNf => write!(f, "CoreNF"),
            DataSourceType::Oam => write!(f, "OAM"),
        }
    }
}

/// Registration information for a data source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSourceRegistration {
    /// Unique identifier for the data source
    pub source_id: String,
    /// Type of data source
    pub source_type: DataSourceType,
    /// Human-readable description
    pub description: String,
    /// Capabilities: which measurement types this source provides
    pub capabilities: Vec<MeasurementCapability>,
    /// Reporting interval hint in milliseconds (0 = event-driven)
    pub reporting_interval_ms: u64,
    /// Whether the source is currently active
    pub active: bool,
}

/// What kind of measurements a data source can provide
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MeasurementCapability {
    /// RSRP/RSRQ/SINR radio measurements
    RadioMeasurement,
    /// UE position/location reports
    LocationReport,
    /// Cell load (PRB usage, throughput)
    CellLoad,
    /// QoS flow statistics
    QosFlowStats,
    /// PDU session statistics
    PduSessionStats,
    /// Network slice load
    SliceLoad,
}

impl std::fmt::Display for MeasurementCapability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MeasurementCapability::RadioMeasurement => write!(f, "RadioMeasurement"),
            MeasurementCapability::LocationReport => write!(f, "LocationReport"),
            MeasurementCapability::CellLoad => write!(f, "CellLoad"),
            MeasurementCapability::QosFlowStats => write!(f, "QosFlowStats"),
            MeasurementCapability::PduSessionStats => write!(f, "PduSessionStats"),
            MeasurementCapability::SliceLoad => write!(f, "SliceLoad"),
        }
    }
}

/// Central data collector that manages data sources and measurement storage
///
/// Serves as the ingestion point for all measurement data flowing into NWDAF.
/// Data sources (gNB, UE, NFs) register themselves and then push measurements
/// which are stored in bounded per-entity histories.
///
/// # Example
///
/// ```
/// use nextgsim_nwdaf::data_collection::*;
/// use nextgsim_nwdaf::{UeMeasurement, Vector3};
///
/// let mut collector = DataCollector::new(100);
///
/// // Register a gNB data source
/// let reg = DataSourceRegistration {
///     source_id: "gnb-1".to_string(),
///     source_type: DataSourceType::Gnb,
///     description: "gNB sector 1".to_string(),
///     capabilities: vec![MeasurementCapability::RadioMeasurement, MeasurementCapability::CellLoad],
///     reporting_interval_ms: 100,
///     active: true,
/// };
/// collector.register_source(reg).unwrap();
///
/// // Push a UE measurement
/// let meas = UeMeasurement {
///     ue_id: 1,
///     rsrp: -80.0,
///     rsrq: -10.0,
///     sinr: Some(15.0),
///     position: Vector3::new(100.0, 200.0, 0.0),
///     velocity: None,
///     serving_cell_id: 1,
///     timestamp_ms: 1000,
/// };
/// collector.report_ue_measurement(meas);
/// ```
#[derive(Debug)]
pub struct DataCollector {
    /// Registered data sources
    sources: HashMap<String, DataSourceRegistration>,
    /// Per-UE measurement histories
    ue_histories: HashMap<i32, UeMeasurementHistory>,
    /// Per-cell load histories
    cell_loads: HashMap<i32, std::collections::VecDeque<CellLoad>>,
    /// Maximum history length per entity
    max_history_length: usize,
    /// Total measurements received
    total_measurements: u64,
}

impl DataCollector {
    /// Creates a new data collector with the given maximum history length
    pub fn new(max_history_length: usize) -> Self {
        Self {
            sources: HashMap::new(),
            ue_histories: HashMap::new(),
            cell_loads: HashMap::new(),
            max_history_length,
            total_measurements: 0,
        }
    }

    /// Registers a new data source
    ///
    /// # Errors
    ///
    /// Returns `DataCollectionError::AlreadyRegistered` if a source with the
    /// same ID is already registered.
    pub fn register_source(
        &mut self,
        registration: DataSourceRegistration,
    ) -> Result<(), DataCollectionError> {
        if self.sources.contains_key(&registration.source_id) {
            return Err(DataCollectionError::AlreadyRegistered {
                source_id: registration.source_id,
            });
        }

        info!(
            "Registered data source: {} (type={}, capabilities={:?})",
            registration.source_id, registration.source_type, registration.capabilities
        );
        self.sources
            .insert(registration.source_id.clone(), registration);
        Ok(())
    }

    /// Unregisters a data source
    ///
    /// # Errors
    ///
    /// Returns `DataCollectionError::SourceNotFound` if the source does not exist.
    pub fn unregister_source(
        &mut self,
        source_id: &str,
    ) -> Result<DataSourceRegistration, DataCollectionError> {
        self.sources
            .remove(source_id)
            .ok_or_else(|| DataCollectionError::SourceNotFound {
                source_id: source_id.to_string(),
            })
    }

    /// Returns all registered data sources
    pub fn registered_sources(&self) -> &HashMap<String, DataSourceRegistration> {
        &self.sources
    }

    /// Returns a specific data source registration
    pub fn get_source(&self, source_id: &str) -> Option<&DataSourceRegistration> {
        self.sources.get(source_id)
    }

    /// Returns the number of registered sources
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Reports a UE measurement
    ///
    /// Stores the measurement in the per-UE history, creating a new
    /// history entry if this is the first measurement for the UE.
    pub fn report_ue_measurement(&mut self, measurement: UeMeasurement) {
        let ue_id = measurement.ue_id;
        let history = self
            .ue_histories
            .entry(ue_id)
            .or_insert_with(|| UeMeasurementHistory::new(ue_id, self.max_history_length));
        history.add(measurement);
        self.total_measurements += 1;
        debug!("Recorded UE measurement for ue_id={}", ue_id);
    }

    /// Reports a cell load measurement
    ///
    /// Stores the load data in the per-cell history.
    pub fn report_cell_load(&mut self, load: CellLoad) {
        let cell_id = load.cell_id;
        let history = self
            .cell_loads
            .entry(cell_id)
            .or_insert_with(|| std::collections::VecDeque::with_capacity(self.max_history_length));

        if history.len() >= self.max_history_length {
            history.pop_front();
        }
        history.push_back(load);
        self.total_measurements += 1;
        debug!("Recorded cell load for cell_id={}", cell_id);
    }

    /// Returns the UE measurement history for a given UE
    pub fn get_ue_history(&self, ue_id: i32) -> Option<&UeMeasurementHistory> {
        self.ue_histories.get(&ue_id)
    }

    /// Returns a mutable reference to the UE history map
    pub fn ue_histories(&self) -> &HashMap<i32, UeMeasurementHistory> {
        &self.ue_histories
    }

    /// Returns the latest cell load for a given cell
    pub fn get_latest_cell_load(&self, cell_id: i32) -> Option<&CellLoad> {
        self.cell_loads.get(&cell_id).and_then(|h| h.back())
    }

    /// Returns the cell load history for a given cell
    pub fn get_cell_load_history(
        &self,
        cell_id: i32,
    ) -> Option<&std::collections::VecDeque<CellLoad>> {
        self.cell_loads.get(&cell_id)
    }

    /// Returns cell load histories map
    pub fn cell_loads(&self) -> &HashMap<i32, std::collections::VecDeque<CellLoad>> {
        &self.cell_loads
    }

    /// Returns the total number of measurements received
    pub fn total_measurements(&self) -> u64 {
        self.total_measurements
    }

    /// Returns the number of tracked UEs
    pub fn tracked_ue_count(&self) -> usize {
        self.ue_histories.len()
    }

    /// Returns the number of tracked cells
    pub fn tracked_cell_count(&self) -> usize {
        self.cell_loads.len()
    }

    /// Returns all known cell IDs
    pub fn known_cell_ids(&self) -> Vec<i32> {
        self.cell_loads.keys().copied().collect()
    }

    /// Deactivates a data source (marks it as inactive)
    pub fn deactivate_source(&mut self, source_id: &str) -> bool {
        if let Some(source) = self.sources.get_mut(source_id) {
            source.active = false;
            info!("Deactivated data source: {}", source_id);
            true
        } else {
            warn!("Attempted to deactivate unknown source: {}", source_id);
            false
        }
    }

    /// Activates a data source
    pub fn activate_source(&mut self, source_id: &str) -> bool {
        if let Some(source) = self.sources.get_mut(source_id) {
            source.active = true;
            info!("Activated data source: {}", source_id);
            true
        } else {
            warn!("Attempted to activate unknown source: {}", source_id);
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vector3;

    fn make_registration(id: &str, source_type: DataSourceType) -> DataSourceRegistration {
        DataSourceRegistration {
            source_id: id.to_string(),
            source_type,
            description: format!("Test source {id}"),
            capabilities: vec![MeasurementCapability::RadioMeasurement],
            reporting_interval_ms: 100,
            active: true,
        }
    }

    #[test]
    fn test_register_source() {
        let mut collector = DataCollector::new(100);
        let reg = make_registration("gnb-1", DataSourceType::Gnb);
        assert!(collector.register_source(reg).is_ok());
        assert_eq!(collector.source_count(), 1);
    }

    #[test]
    fn test_duplicate_registration() {
        let mut collector = DataCollector::new(100);
        let reg1 = make_registration("gnb-1", DataSourceType::Gnb);
        let reg2 = make_registration("gnb-1", DataSourceType::Gnb);
        assert!(collector.register_source(reg1).is_ok());
        let result = collector.register_source(reg2);
        assert!(result.is_err());
        assert!(matches!(
            result.expect_err("should fail"),
            DataCollectionError::AlreadyRegistered { .. }
        ));
    }

    #[test]
    fn test_unregister_source() {
        let mut collector = DataCollector::new(100);
        let reg = make_registration("gnb-1", DataSourceType::Gnb);
        collector.register_source(reg).expect("should register");
        let removed = collector.unregister_source("gnb-1");
        assert!(removed.is_ok());
        assert_eq!(collector.source_count(), 0);
    }

    #[test]
    fn test_unregister_nonexistent() {
        let mut collector = DataCollector::new(100);
        let result = collector.unregister_source("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_report_ue_measurement() {
        let mut collector = DataCollector::new(100);

        let meas = UeMeasurement {
            ue_id: 42,
            rsrp: -80.0,
            rsrq: -10.0,
            sinr: Some(15.0),
            position: Vector3::new(100.0, 200.0, 0.0),
            velocity: None,
            serving_cell_id: 1,
            timestamp_ms: 1000,
        };
        collector.report_ue_measurement(meas);

        assert_eq!(collector.tracked_ue_count(), 1);
        assert_eq!(collector.total_measurements(), 1);
        let history = collector.get_ue_history(42);
        assert!(history.is_some());
        assert_eq!(history.expect("should exist").len(), 1);
    }

    #[test]
    fn test_report_cell_load() {
        let mut collector = DataCollector::new(100);

        let load = CellLoad {
            cell_id: 7,
            prb_usage: 0.5,
            connected_ues: 10,
            avg_throughput_mbps: 150.0,
            timestamp_ms: 1000,
        };
        collector.report_cell_load(load);

        assert_eq!(collector.tracked_cell_count(), 1);
        let latest = collector.get_latest_cell_load(7);
        assert!(latest.is_some());
        assert!((latest.expect("should exist").prb_usage - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_history_bounded() {
        let mut collector = DataCollector::new(5);

        for i in 0..10 {
            collector.report_ue_measurement(UeMeasurement {
                ue_id: 1,
                rsrp: -80.0 + i as f32,
                rsrq: -10.0,
                sinr: None,
                position: Vector3::new(i as f64, 0.0, 0.0),
                velocity: None,
                serving_cell_id: 1,
                timestamp_ms: i as u64 * 100,
            });
        }

        let history = collector.get_ue_history(1).expect("should exist");
        assert_eq!(history.len(), 5); // bounded to max_history_length
    }

    #[test]
    fn test_activate_deactivate() {
        let mut collector = DataCollector::new(100);
        let reg = make_registration("gnb-1", DataSourceType::Gnb);
        collector.register_source(reg).expect("should register");

        assert!(collector.deactivate_source("gnb-1"));
        let source = collector.get_source("gnb-1").expect("should exist");
        assert!(!source.active);

        assert!(collector.activate_source("gnb-1"));
        let source = collector.get_source("gnb-1").expect("should exist");
        assert!(source.active);

        // Nonexistent source
        assert!(!collector.deactivate_source("nonexistent"));
        assert!(!collector.activate_source("nonexistent"));
    }
}
