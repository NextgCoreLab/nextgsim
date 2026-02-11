//! Core 5G types: PLMN, TAI, S-NSSAI, SUPI, GUTI, etc.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Public Land Mobile Network identifier.
///
/// A PLMN uniquely identifies a mobile network and consists of:
/// - MCC (Mobile Country Code): 3 decimal digits (001-999)
/// - MNC (Mobile Network Code): 2 or 3 decimal digits
///
/// The `long_mnc` field indicates whether the MNC uses 3 digits (true) or 2 digits (false).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[derive(Default)]
pub struct Plmn {
    /// Mobile Country Code (3 digits, range 0-999)
    pub mcc: u16,
    /// Mobile Network Code (2-3 digits, range 0-999)
    pub mnc: u16,
    /// True if MNC is 3 digits, false if 2 digits
    pub long_mnc: bool,
}

impl Plmn {
    /// Creates a new PLMN with the given MCC and MNC.
    ///
    /// # Arguments
    /// * `mcc` - Mobile Country Code (3 digits)
    /// * `mnc` - Mobile Network Code (2-3 digits)
    /// * `long_mnc` - Whether MNC is 3 digits
    pub const fn new(mcc: u16, mnc: u16, long_mnc: bool) -> Self {
        Self { mcc, mnc, long_mnc }
    }

    /// Returns true if this PLMN has valid values set.
    pub fn has_value(&self) -> bool {
        self.mcc > 0 || self.mnc > 0
    }

    /// Encodes the PLMN to 3GPP format (3 bytes).
    ///
    /// The encoding follows 3GPP TS 24.008 format:
    /// - Byte 0: MCC digit 2 (high nibble) | MCC digit 1 (low nibble)
    /// - Byte 1: MNC digit 3 or 0xF (high nibble) | MCC digit 3 (low nibble)
    /// - Byte 2: MNC digit 2 (high nibble) | MNC digit 1 (low nibble)
    pub fn encode(&self) -> [u8; 3] {
        let mcc = self.mcc;
        let mcc3 = (mcc % 10) as u8;
        let mcc2 = ((mcc % 100) / 10) as u8;
        let mcc1 = ((mcc % 1000) / 100) as u8;

        let mnc = self.mnc;
        let (mnc1, mnc2, mnc3) = if self.long_mnc {
            (
                ((mnc % 1000) / 100) as u8,
                ((mnc % 100) / 10) as u8,
                (mnc % 10) as u8,
            )
        } else {
            (((mnc % 100) / 10) as u8, (mnc % 10) as u8, 0x0F)
        };

        let octet1 = (mcc2 << 4) | mcc1;
        let octet2 = (mnc3 << 4) | mcc3;
        let octet3 = (mnc2 << 4) | mnc1;

        [octet1, octet2, octet3]
    }

    /// Decodes a PLMN from 3GPP format (3 bytes).
    ///
    /// # Arguments
    /// * `bytes` - 3-byte array in 3GPP PLMN encoding format
    ///
    /// # Returns
    /// The decoded PLMN
    pub fn decode(bytes: [u8; 3]) -> Self {
        let octet1 = bytes[0];
        let octet2 = bytes[1];
        let octet3 = bytes[2];

        // Decode MCC
        let mcc1 = (octet1 & 0x0F) as u16;
        let mcc2 = ((octet1 >> 4) & 0x0F) as u16;
        let mcc3 = (octet2 & 0x0F) as u16;
        let mcc = 100 * mcc1 + 10 * mcc2 + mcc3;

        // Decode MNC
        let mnc3 = (octet2 >> 4) & 0x0F;
        let mnc1 = (octet3 & 0x0F) as u16;
        let mnc2 = ((octet3 >> 4) & 0x0F) as u16;

        let (mnc, long_mnc) =
            if mnc3 != 0x0F || (octet1 == 0xFF && octet2 == 0xFF && octet3 == 0xFF) {
                // 3-digit MNC
                (10 * (10 * mnc1 + mnc2) + mnc3 as u16, true)
            } else {
                // 2-digit MNC
                (10 * mnc1 + mnc2, false)
            };

        Self { mcc, mnc, long_mnc }
    }
}

impl fmt::Debug for Plmn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.long_mnc {
            write!(f, "Plmn({:03}-{:03})", self.mcc, self.mnc)
        } else {
            write!(f, "Plmn({:03}-{:02})", self.mcc, self.mnc)
        }
    }
}

impl fmt::Display for Plmn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.long_mnc {
            write!(f, "{:03}{:03}", self.mcc, self.mnc)
        } else {
            write!(f, "{:03}{:02}", self.mcc, self.mnc)
        }
    }
}


// ============================================================================
// 6G ISAC (Integrated Sensing and Communication) Types
// ============================================================================

/// ISAC sensing measurement data captured from the environment.
///
/// Contains raw sensing data from ISAC operations including radar measurements,
/// object detection, and environmental sensing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensingMeasurement {
    /// Timestamp of the measurement (milliseconds since epoch)
    pub timestamp_ms: u64,
    /// Range to detected object (meters)
    pub range_meters: f64,
    /// Doppler velocity (m/s, positive = approaching)
    pub velocity_mps: f64,
    /// Azimuth angle (degrees, 0-360)
    pub azimuth_deg: f64,
    /// Elevation angle (degrees, -90 to +90)
    pub elevation_deg: f64,
    /// Signal strength (dBm)
    pub signal_strength_dbm: f32,
    /// Target ID (0 = unidentified)
    pub target_id: u32,
}

impl SensingMeasurement {
    /// Creates a new sensing measurement.
    ///
    /// # Arguments
    /// * `timestamp_ms` - Timestamp in milliseconds since epoch
    /// * `range_meters` - Range to target in meters
    /// * `velocity_mps` - Velocity in meters per second
    /// * `azimuth_deg` - Azimuth angle in degrees
    /// * `elevation_deg` - Elevation angle in degrees
    /// * `signal_strength_dbm` - Signal strength in dBm
    pub fn new(
        timestamp_ms: u64,
        range_meters: f64,
        velocity_mps: f64,
        azimuth_deg: f64,
        elevation_deg: f64,
        signal_strength_dbm: f32,
    ) -> Self {
        Self {
            timestamp_ms,
            range_meters,
            velocity_mps,
            azimuth_deg,
            elevation_deg,
            signal_strength_dbm,
            target_id: 0,
        }
    }

    /// Returns true if this measurement represents a valid detection.
    pub fn is_valid(&self) -> bool {
        self.signal_strength_dbm > -120.0 && self.range_meters > 0.0
    }
}

/// ISAC sensing configuration parameters.
///
/// Defines operational parameters for integrated sensing and communication,
/// including sensing mode, bandwidth, and detection thresholds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensingConfig {
    /// Whether ISAC sensing is enabled
    pub enabled: bool,
    /// Sensing mode (0=passive, 1=active radar, 2=hybrid)
    pub mode: u8,
    /// Sensing bandwidth in MHz
    pub bandwidth_mhz: u32,
    /// Maximum sensing range in meters
    pub max_range_meters: f64,
    /// Minimum detection threshold in dBm
    pub detection_threshold_dbm: f32,
    /// Sensing interval in milliseconds
    pub sensing_interval_ms: u32,
}

impl SensingConfig {
    /// Creates a new sensing configuration.
    pub fn new(mode: u8, bandwidth_mhz: u32, max_range_meters: f64) -> Self {
        Self {
            enabled: true,
            mode,
            bandwidth_mhz,
            max_range_meters,
            detection_threshold_dbm: -100.0,
            sensing_interval_ms: 100,
        }
    }
}

impl Default for SensingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: 0,
            bandwidth_mhz: 100,
            max_range_meters: 1000.0,
            detection_threshold_dbm: -100.0,
            sensing_interval_ms: 100,
        }
    }
}

/// ISAC sensing result containing processed detection information.
///
/// Aggregates multiple measurements into a coherent result with
/// statistical confidence and classification information.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensingResult {
    /// Timestamp of result generation
    pub timestamp_ms: u64,
    /// Number of detections in this result
    pub detection_count: u32,
    /// Individual measurements
    pub measurements: Vec<SensingMeasurement>,
    /// Confidence level (0.0-1.0)
    pub confidence: f32,
    /// Object classification (0=unknown, 1=person, 2=vehicle, 3=drone, etc.)
    pub classification: u8,
}

impl SensingResult {
    /// Creates a new sensing result.
    pub fn new(timestamp_ms: u64, measurements: Vec<SensingMeasurement>) -> Self {
        let detection_count = measurements.len() as u32;
        Self {
            timestamp_ms,
            detection_count,
            measurements,
            confidence: 0.0,
            classification: 0,
        }
    }
}

// ============================================================================
// 6G Semantic Communication Types
// ============================================================================

/// Modality type for semantic communication.
///
/// Specifies the type of data being transmitted in semantic communication
/// to optimize encoding and compression strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[derive(Default)]
pub enum ModalityType {
    /// Text/linguistic data
    #[default]
    Text,
    /// Image/visual data
    Image,
    /// Audio/speech data
    Audio,
    /// Video data
    Video,
    /// Sensor telemetry
    Sensor,
    /// Mixed/multimodal data
    Mixed,
}


impl fmt::Display for ModalityType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModalityType::Text => write!(f, "text"),
            ModalityType::Image => write!(f, "image"),
            ModalityType::Audio => write!(f, "audio"),
            ModalityType::Video => write!(f, "video"),
            ModalityType::Sensor => write!(f, "sensor"),
            ModalityType::Mixed => write!(f, "mixed"),
        }
    }
}

/// Compression level for semantic encoding.
///
/// Defines the trade-off between bandwidth efficiency and information fidelity
/// in semantic communication systems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[derive(Default)]
pub enum CompressionLevel {
    /// No compression, preserve all information
    None,
    /// Low compression, minimal information loss
    Low,
    /// Medium compression, balanced trade-off
    #[default]
    Medium,
    /// High compression, aggressive semantic extraction
    High,
    /// Maximum compression, only essential semantics
    Maximum,
}


impl fmt::Display for CompressionLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompressionLevel::None => write!(f, "none"),
            CompressionLevel::Low => write!(f, "low"),
            CompressionLevel::Medium => write!(f, "medium"),
            CompressionLevel::High => write!(f, "high"),
            CompressionLevel::Maximum => write!(f, "maximum"),
        }
    }
}

/// Semantic communication profile and metadata.
///
/// Contains metadata describing the semantic content, encoding parameters,
/// and quality requirements for semantic communication.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticProfile {
    /// Modality of the content
    pub modality: ModalityType,
    /// Compression level
    pub compression: CompressionLevel,
    /// Semantic importance score (0.0-1.0, higher = more important)
    pub importance: f32,
    /// Context identifier for semantic understanding
    pub context_id: u32,
    /// Expected latency budget in milliseconds
    pub latency_budget_ms: u32,
    /// Minimum acceptable quality (0.0-1.0)
    pub min_quality: f32,
}

impl SemanticProfile {
    /// Creates a new semantic profile.
    pub fn new(modality: ModalityType, compression: CompressionLevel, importance: f32) -> Self {
        Self {
            modality,
            compression,
            importance: importance.clamp(0.0, 1.0),
            context_id: 0,
            latency_budget_ms: 100,
            min_quality: 0.7,
        }
    }
}

impl Default for SemanticProfile {
    fn default() -> Self {
        Self {
            modality: ModalityType::default(),
            compression: CompressionLevel::default(),
            importance: 0.5,
            context_id: 0,
            latency_budget_ms: 100,
            min_quality: 0.7,
        }
    }
}

// ============================================================================
// 6G Federated Learning Types
// ============================================================================

/// Federated learning model identifier.
///
/// Uniquely identifies a machine learning model in federated learning operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelId(pub u64);

impl ModelId {
    /// Creates a new model ID.
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns the raw ID value.
    pub const fn value(&self) -> u64 {
        self.0
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "model-{:016x}", self.0)
    }
}

/// Federated learning model version.
///
/// Tracks version evolution of ML models through training rounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ModelVersion {
    /// Major version number
    pub major: u32,
    /// Minor version number
    pub minor: u32,
    /// Patch version number
    pub patch: u32,
}

impl ModelVersion {
    /// Creates a new model version.
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }
}

impl Default for ModelVersion {
    fn default() -> Self {
        Self::new(0, 0, 0)
    }
}

impl fmt::Display for ModelVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Federated learning training round identifier.
///
/// Identifies a specific training round in the federated learning process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TrainingRound(pub u64);

impl TrainingRound {
    /// Creates a new training round identifier.
    pub const fn new(round: u64) -> Self {
        Self(round)
    }

    /// Returns the round number.
    pub const fn value(&self) -> u64 {
        self.0
    }

    /// Increments to the next round.
    pub fn next(&self) -> Self {
        Self(self.0 + 1)
    }
}

impl fmt::Display for TrainingRound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "round-{}", self.0)
    }
}

// ============================================================================
// 6G Split-Hybrid Edge (SHE) Computing Types
// ============================================================================

/// Accelerator type for edge computing.
///
/// Specifies the type of hardware accelerator available or required
/// for computational offloading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[derive(Default)]
pub enum AcceleratorType {
    /// No specific accelerator required
    None,
    /// GPU acceleration
    Gpu,
    /// TPU (Tensor Processing Unit)
    Tpu,
    /// FPGA (Field-Programmable Gate Array)
    Fpga,
    /// NPU (Neural Processing Unit)
    Npu,
    /// General purpose CPU
    #[default]
    Cpu,
}


impl fmt::Display for AcceleratorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AcceleratorType::None => write!(f, "none"),
            AcceleratorType::Gpu => write!(f, "gpu"),
            AcceleratorType::Tpu => write!(f, "tpu"),
            AcceleratorType::Fpga => write!(f, "fpga"),
            AcceleratorType::Npu => write!(f, "npu"),
            AcceleratorType::Cpu => write!(f, "cpu"),
        }
    }
}

/// Placement constraint for compute offloading.
///
/// Defines constraints on where computation can be executed in the network.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlacementConstraint {
    /// Maximum latency budget in milliseconds
    pub max_latency_ms: u32,
    /// Required accelerator type
    pub required_accelerator: AcceleratorType,
    /// Minimum required memory in MB
    pub min_memory_mb: u32,
    /// Minimum required compute capacity (arbitrary units)
    pub min_compute_units: u32,
    /// Whether data locality is required
    pub data_locality_required: bool,
    /// Allowed deployment zones (empty = no restriction)
    pub allowed_zones: Vec<String>,
}

impl PlacementConstraint {
    /// Creates a new placement constraint with basic parameters.
    pub fn new(max_latency_ms: u32, required_accelerator: AcceleratorType) -> Self {
        Self {
            max_latency_ms,
            required_accelerator,
            min_memory_mb: 0,
            min_compute_units: 0,
            data_locality_required: false,
            allowed_zones: Vec::new(),
        }
    }
}

impl Default for PlacementConstraint {
    fn default() -> Self {
        Self {
            max_latency_ms: 1000,
            required_accelerator: AcceleratorType::default(),
            min_memory_mb: 0,
            min_compute_units: 0,
            data_locality_required: false,
            allowed_zones: Vec::new(),
        }
    }
}

/// Compute request for edge offloading.
///
/// Describes a computational task to be offloaded to edge computing resources.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComputeRequest {
    /// Unique request identifier
    pub request_id: u64,
    /// Task type identifier
    pub task_type: String,
    /// Estimated workload (in compute units)
    pub workload_units: u64,
    /// Placement constraints
    pub constraints: PlacementConstraint,
    /// Input data size in bytes
    pub input_size_bytes: u64,
    /// Expected output data size in bytes
    pub output_size_bytes: u64,
    /// Priority level (0=lowest, 255=highest)
    pub priority: u8,
}

impl ComputeRequest {
    /// Creates a new compute request.
    pub fn new(request_id: u64, task_type: String, workload_units: u64) -> Self {
        Self {
            request_id,
            task_type,
            workload_units,
            constraints: PlacementConstraint::default(),
            input_size_bytes: 0,
            output_size_bytes: 0,
            priority: 128,
        }
    }
}

/// Compute descriptor for SHE operations.
///
/// Describes the computational characteristics and requirements of an offloaded task.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComputeDescriptor {
    /// Compute request details
    pub request: ComputeRequest,
    /// Timestamp of request creation (milliseconds since epoch)
    pub timestamp_ms: u64,
    /// Session ID for tracking
    pub session_id: u64,
    /// Whether this is a real-time request
    pub real_time: bool,
}

impl ComputeDescriptor {
    /// Creates a new compute descriptor.
    pub fn new(request: ComputeRequest, session_id: u64) -> Self {
        Self {
            request,
            timestamp_ms: 0,
            session_id,
            real_time: false,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Rel-20 (6G) Additional Common Types
// ────────────────────────────────────────────────────────────────────────────

/// RIS (Reconfigurable Intelligent Surface) element configuration (TR 38.901 6G ext).
#[derive(Debug, Clone, PartialEq)]
pub struct RisElement {
    /// Element index within the RIS panel.
    pub index: u32,
    /// Phase shift in radians [0, 2π).
    pub phase_shift: f64,
    /// Amplitude coefficient [0.0, 1.0].
    pub amplitude: f64,
    /// Polarization state.
    pub polarization: RisPolarization,
}

/// Polarization mode for an RIS element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RisPolarization {
    /// Horizontal linear polarization.
    Horizontal,
    /// Vertical linear polarization.
    Vertical,
    /// Right-hand circular polarization.
    RhCircular,
    /// Left-hand circular polarization.
    LhCircular,
}

/// RIS panel configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct RisConfig {
    /// Panel identifier.
    pub panel_id: u32,
    /// Number of rows in the element grid.
    pub rows: u16,
    /// Number of columns in the element grid.
    pub cols: u16,
    /// Operating frequency in MHz.
    pub frequency_mhz: u32,
    /// Element spacing in wavelengths (λ).
    pub element_spacing: f64,
    /// Individual element configurations.
    pub elements: Vec<RisElement>,
    /// Whether the panel supports active amplification.
    pub active_ris: bool,
    /// Control link latency budget in microseconds.
    pub control_latency_us: u32,
}

impl RisConfig {
    /// Creates a new RIS panel with the given grid dimensions.
    pub fn new(panel_id: u32, rows: u16, cols: u16, frequency_mhz: u32) -> Self {
        Self {
            panel_id,
            rows,
            cols,
            frequency_mhz,
            element_spacing: 0.5,
            elements: Vec::new(),
            active_ris: false,
            control_latency_us: 100,
        }
    }

    /// Total number of RIS elements.
    pub fn element_count(&self) -> u32 {
        self.rows as u32 * self.cols as u32
    }
}

/// Digital Twin synchronization state (TS 23.700-Digital ext).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DigitalTwinSyncState {
    /// Twin is being initialized, no data synced yet.
    Initializing,
    /// Twin is actively synchronized with the physical entity.
    Synchronized,
    /// Twin has stale data beyond the freshness threshold.
    Stale,
    /// Synchronization is paused (e.g., UE in idle).
    Paused,
    /// Twin is detached from the physical entity.
    Detached,
}

/// Digital Twin descriptor for a network entity.
#[derive(Debug, Clone, PartialEq)]
pub struct DigitalTwinDescriptor {
    /// Unique twin identifier.
    pub twin_id: u64,
    /// Type of physical entity this twin represents.
    pub entity_type: DigitalTwinEntityType,
    /// Current synchronization state.
    pub sync_state: DigitalTwinSyncState,
    /// Maximum acceptable staleness in milliseconds.
    pub freshness_threshold_ms: u32,
    /// Last synchronization timestamp (epoch ms).
    pub last_sync_ms: u64,
    /// Fidelity level of the twin model.
    pub fidelity: DigitalTwinFidelity,
}

/// Type of physical entity represented by a Digital Twin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DigitalTwinEntityType {
    /// A user equipment device.
    Ue,
    /// A gNB base station.
    Gnb,
    /// A RIS panel.
    Ris,
    /// A network slice instance.
    Slice,
    /// An edge compute node.
    EdgeNode,
    /// A complete cell site.
    CellSite,
}

/// Fidelity level of a Digital Twin model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DigitalTwinFidelity {
    /// Low fidelity — statistical/aggregate model.
    Low,
    /// Medium fidelity — behavioral model.
    Medium,
    /// High fidelity — physics-based / ray-tracing model.
    High,
}

impl DigitalTwinDescriptor {
    /// Creates a new twin descriptor.
    pub fn new(twin_id: u64, entity_type: DigitalTwinEntityType) -> Self {
        Self {
            twin_id,
            entity_type,
            sync_state: DigitalTwinSyncState::Initializing,
            freshness_threshold_ms: 100,
            last_sync_ms: 0,
            fidelity: DigitalTwinFidelity::Medium,
        }
    }

    /// Returns true if the twin data is considered fresh.
    pub fn is_fresh(&self, now_ms: u64) -> bool {
        self.sync_state == DigitalTwinSyncState::Synchronized
            && now_ms.saturating_sub(self.last_sync_ms) <= self.freshness_threshold_ms as u64
    }
}

/// Intent-based networking action descriptor (TS 28.312 6G ext).
#[derive(Debug, Clone, PartialEq)]
pub struct IntentDescriptor {
    /// Unique intent identifier.
    pub intent_id: u64,
    /// High-level objective.
    pub objective: IntentObjective,
    /// Priority (lower = higher priority).
    pub priority: u8,
    /// Target scope for this intent.
    pub scope: IntentScope,
    /// Expected KPI targets.
    pub kpi_targets: Vec<IntentKpiTarget>,
    /// Whether the intent can be decomposed into sub-intents.
    pub decomposable: bool,
}

/// High-level objective of an intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntentObjective {
    /// Maximize throughput for a slice or bearer.
    MaximizeThroughput,
    /// Minimize end-to-end latency.
    MinimizeLatency,
    /// Minimize energy consumption.
    MinimizeEnergy,
    /// Maximize coverage area.
    MaximizeCoverage,
    /// Maintain reliability above a target.
    EnsureReliability,
    /// Balance load across cells.
    LoadBalancing,
    /// Optimize spectral efficiency.
    SpectralEfficiency,
}

/// Scope to which an intent applies.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IntentScope {
    /// Applies to a single UE.
    Ue(u64),
    /// Applies to a specific cell.
    Cell(u32),
    /// Applies to a network slice.
    Slice(SNssai),
    /// Applies to the entire network.
    Network,
}

/// KPI target for an intent.
#[derive(Debug, Clone, PartialEq)]
pub struct IntentKpiTarget {
    /// KPI name (e.g., "latency_ms", "throughput_mbps").
    pub kpi_name: String,
    /// Target value.
    pub target: f64,
    /// Comparison operator.
    pub operator: KpiOperator,
}

/// Comparison operator for a KPI target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KpiOperator {
    /// KPI must be less than target.
    LessThan,
    /// KPI must be less than or equal to target.
    LessOrEqual,
    /// KPI must be greater than target.
    GreaterThan,
    /// KPI must be greater than or equal to target.
    GreaterOrEqual,
    /// KPI must equal target (within tolerance).
    Equal,
}

impl IntentDescriptor {
    /// Creates a new intent descriptor.
    pub fn new(intent_id: u64, objective: IntentObjective, scope: IntentScope) -> Self {
        Self {
            intent_id,
            objective,
            priority: 128,
            scope,
            kpi_targets: Vec::new(),
            decomposable: true,
        }
    }
}

/// Sub-THz band configuration (FR3, 92–300 GHz, TR 38.901 6G ext).
#[derive(Debug, Clone, PartialEq)]
pub struct SubThzConfig {
    /// Center frequency in GHz.
    pub center_freq_ghz: f64,
    /// Channel bandwidth in GHz.
    pub bandwidth_ghz: f64,
    /// Molecular absorption loss model.
    pub absorption_model: AbsorptionModel,
    /// Maximum transmit power in dBm.
    pub max_tx_power_dbm: f64,
    /// Number of antenna elements (massive MIMO).
    pub antenna_elements: u32,
    /// Beam tracking update interval in microseconds.
    pub beam_tracking_interval_us: u32,
}

/// Molecular absorption model for sub-THz propagation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AbsorptionModel {
    /// ITU-R P.676 standard gaseous absorption.
    ItuR676,
    /// Simplified line-by-line model.
    LineByLine,
    /// Window-based model (low-absorption frequency windows).
    WindowBased,
}

impl SubThzConfig {
    /// Creates a new sub-THz configuration.
    pub fn new(center_freq_ghz: f64, bandwidth_ghz: f64) -> Self {
        Self {
            center_freq_ghz,
            bandwidth_ghz,
            absorption_model: AbsorptionModel::ItuR676,
            max_tx_power_dbm: 20.0,
            antenna_elements: 1024,
            beam_tracking_interval_us: 50,
        }
    }

    /// Returns true if the center frequency is in the sub-THz range (92–300 GHz).
    pub fn is_valid_sub_thz(&self) -> bool {
        self.center_freq_ghz >= 92.0 && self.center_freq_ghz <= 300.0
    }
}

/// Energy efficiency profile for network entities (TR 21.916 6G ext).
#[derive(Debug, Clone, PartialEq)]
pub struct EnergyProfile {
    /// Entity identifier (gNB ID, UE ID, etc.).
    pub entity_id: u64,
    /// Current power consumption in watts.
    pub power_watts: f64,
    /// Current energy efficiency in bits/joule.
    pub efficiency_bits_per_joule: f64,
    /// Active energy-saving features.
    pub saving_features: Vec<EnergySavingFeature>,
    /// Sleep mode state.
    pub sleep_state: SleepState,
    /// Renewable energy ratio [0.0, 1.0].
    pub renewable_ratio: f64,
}

/// Network energy-saving feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EnergySavingFeature {
    /// Carrier shutdown when low traffic.
    CarrierShutdown,
    /// MIMO layer reduction.
    MimoLayerReduction,
    /// Symbol shutdown within a slot.
    SymbolShutdown,
    /// Micro sleep between DRX cycles.
    MicroSleep,
    /// Cell DTX (Discontinuous Transmission).
    CellDtx,
    /// Bandwidth part adaptation.
    BwpAdaptation,
    /// Dynamic power back-off.
    PowerBackoff,
}

/// Sleep mode state for a network entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SleepState {
    /// Fully active.
    Active,
    /// Light sleep — fast wake-up (<1 ms).
    LightSleep,
    /// Deep sleep — moderate wake-up (<10 ms).
    DeepSleep,
    /// Hibernation — slow wake-up (<100 ms).
    Hibernation,
    /// Powered off.
    Off,
}

impl EnergyProfile {
    /// Creates a new energy profile for an entity.
    pub fn new(entity_id: u64) -> Self {
        Self {
            entity_id,
            power_watts: 0.0,
            efficiency_bits_per_joule: 0.0,
            saving_features: Vec::new(),
            sleep_state: SleepState::Active,
            renewable_ratio: 0.0,
        }
    }

    /// Returns true if any energy-saving feature is active.
    pub fn is_saving_active(&self) -> bool {
        !self.saving_features.is_empty() || self.sleep_state != SleepState::Active
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plmn_new() {
        let plmn = Plmn::new(310, 410, false);
        assert_eq!(plmn.mcc, 310);
        assert_eq!(plmn.mnc, 410);
        assert!(!plmn.long_mnc);
    }

    #[test]
    fn test_plmn_encode_2digit_mnc() {
        // MCC=310, MNC=41 (2-digit)
        let plmn = Plmn::new(310, 41, false);
        let encoded = plmn.encode();
        // MCC: 3-1-0 -> mcc1=3, mcc2=1, mcc3=0
        // MNC: 4-1 -> mnc1=4, mnc2=1, mnc3=0xF
        // octet1 = mcc2<<4 | mcc1 = 0x13
        // octet2 = mnc3<<4 | mcc3 = 0xF0
        // octet3 = mnc2<<4 | mnc1 = 0x14
        assert_eq!(encoded, [0x13, 0xF0, 0x14]);
    }

    #[test]
    fn test_plmn_encode_3digit_mnc() {
        // MCC=310, MNC=410 (3-digit)
        let plmn = Plmn::new(310, 410, true);
        let encoded = plmn.encode();
        // MCC: 3-1-0 -> mcc1=3, mcc2=1, mcc3=0
        // MNC: 4-1-0 -> mnc1=4, mnc2=1, mnc3=0
        // octet1 = mcc2<<4 | mcc1 = 0x13
        // octet2 = mnc3<<4 | mcc3 = 0x00
        // octet3 = mnc2<<4 | mnc1 = 0x14
        assert_eq!(encoded, [0x13, 0x00, 0x14]);
    }

    #[test]
    fn test_plmn_decode_2digit_mnc() {
        let bytes = [0x13, 0xF0, 0x14];
        let plmn = Plmn::decode(bytes);
        assert_eq!(plmn.mcc, 310);
        assert_eq!(plmn.mnc, 41);
        assert!(!plmn.long_mnc);
    }

    #[test]
    fn test_plmn_decode_3digit_mnc() {
        let bytes = [0x13, 0x00, 0x14];
        let plmn = Plmn::decode(bytes);
        assert_eq!(plmn.mcc, 310);
        assert_eq!(plmn.mnc, 410);
        assert!(plmn.long_mnc);
    }

    #[test]
    fn test_plmn_roundtrip_2digit() {
        let original = Plmn::new(234, 15, false);
        let encoded = original.encode();
        let decoded = Plmn::decode(encoded);
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_plmn_roundtrip_3digit() {
        let original = Plmn::new(234, 150, true);
        let encoded = original.encode();
        let decoded = Plmn::decode(encoded);
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_plmn_display_2digit() {
        let plmn = Plmn::new(310, 41, false);
        assert_eq!(format!("{plmn}"), "31041");
    }

    #[test]
    fn test_plmn_display_3digit() {
        let plmn = Plmn::new(310, 410, true);
        assert_eq!(format!("{plmn}"), "310410");
    }

    #[test]
    fn test_plmn_debug_2digit() {
        let plmn = Plmn::new(310, 41, false);
        assert_eq!(format!("{plmn:?}"), "Plmn(310-41)");
    }

    #[test]
    fn test_plmn_debug_3digit() {
        let plmn = Plmn::new(310, 410, true);
        assert_eq!(format!("{plmn:?}"), "Plmn(310-410)");
    }

    #[test]
    fn test_plmn_has_value() {
        let empty = Plmn::default();
        assert!(!empty.has_value());

        let with_mcc = Plmn::new(310, 0, false);
        assert!(with_mcc.has_value());

        let with_mnc = Plmn::new(0, 41, false);
        assert!(with_mnc.has_value());
    }

    #[test]
    fn test_plmn_equality() {
        let plmn1 = Plmn::new(310, 410, true);
        let plmn2 = Plmn::new(310, 410, true);
        let plmn3 = Plmn::new(310, 410, false);
        assert_eq!(plmn1, plmn2);
        assert_ne!(plmn1, plmn3);
    }

    // TAI tests

    #[test]
    fn test_tai_new() {
        let plmn = Plmn::new(310, 410, true);
        let tai = Tai::new(plmn, 0x123456);
        assert_eq!(tai.plmn, plmn);
        assert_eq!(tai.tac, 0x123456);
    }

    #[test]
    fn test_tai_from_parts() {
        let tai = Tai::from_parts(310, 410, true, 0x123456);
        assert_eq!(tai.plmn.mcc, 310);
        assert_eq!(tai.plmn.mnc, 410);
        assert!(tai.plmn.long_mnc);
        assert_eq!(tai.tac, 0x123456);
    }

    #[test]
    fn test_tai_has_value() {
        let empty = Tai::default();
        assert!(!empty.has_value());

        let with_plmn = Tai::new(Plmn::new(310, 0, false), 0);
        assert!(with_plmn.has_value());

        let with_tac = Tai::new(Plmn::default(), 1);
        assert!(with_tac.has_value());

        let full = Tai::from_parts(310, 410, true, 0x123456);
        assert!(full.has_value());
    }

    #[test]
    fn test_tai_encode() {
        // MCC=310, MNC=410 (3-digit), TAC=0x123456
        let tai = Tai::from_parts(310, 410, true, 0x123456);
        let encoded = tai.encode();
        // PLMN encoding: [0x13, 0x00, 0x14]
        // TAC encoding: [0x12, 0x34, 0x56]
        assert_eq!(encoded, [0x13, 0x00, 0x14, 0x12, 0x34, 0x56]);
    }

    #[test]
    fn test_tai_encode_2digit_mnc() {
        // MCC=310, MNC=41 (2-digit), TAC=0x000001
        let tai = Tai::from_parts(310, 41, false, 1);
        let encoded = tai.encode();
        // PLMN encoding: [0x13, 0xF0, 0x14]
        // TAC encoding: [0x00, 0x00, 0x01]
        assert_eq!(encoded, [0x13, 0xF0, 0x14, 0x00, 0x00, 0x01]);
    }

    #[test]
    fn test_tai_decode() {
        let bytes = [0x13, 0x00, 0x14, 0x12, 0x34, 0x56];
        let tai = Tai::decode(bytes);
        assert_eq!(tai.plmn.mcc, 310);
        assert_eq!(tai.plmn.mnc, 410);
        assert!(tai.plmn.long_mnc);
        assert_eq!(tai.tac, 0x123456);
    }

    #[test]
    fn test_tai_decode_2digit_mnc() {
        let bytes = [0x13, 0xF0, 0x14, 0x00, 0x00, 0x01];
        let tai = Tai::decode(bytes);
        assert_eq!(tai.plmn.mcc, 310);
        assert_eq!(tai.plmn.mnc, 41);
        assert!(!tai.plmn.long_mnc);
        assert_eq!(tai.tac, 1);
    }

    #[test]
    fn test_tai_roundtrip() {
        let original = Tai::from_parts(234, 150, true, 0xABCDEF);
        let encoded = original.encode();
        let decoded = Tai::decode(encoded);
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_tai_roundtrip_2digit() {
        let original = Tai::from_parts(234, 15, false, 0x000FFF);
        let encoded = original.encode();
        let decoded = Tai::decode(encoded);
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_tai_display() {
        let tai = Tai::from_parts(310, 410, true, 0x123456);
        assert_eq!(format!("{tai}"), "310410-1193046");
    }

    #[test]
    fn test_tai_debug() {
        let tai = Tai::from_parts(310, 410, true, 0x123456);
        assert_eq!(format!("{tai:?}"), "Tai(Plmn(310-410), tac=1193046)");
    }

    #[test]
    fn test_tai_equality() {
        let tai1 = Tai::from_parts(310, 410, true, 0x123456);
        let tai2 = Tai::from_parts(310, 410, true, 0x123456);
        let tai3 = Tai::from_parts(310, 410, true, 0x123457);
        let tai4 = Tai::from_parts(310, 410, false, 0x123456);
        assert_eq!(tai1, tai2);
        assert_ne!(tai1, tai3);
        assert_ne!(tai1, tai4);
    }

    #[test]
    fn test_tai_max_tac() {
        // TAC is 24-bit, max value is 0xFFFFFF (16777215)
        let tai = Tai::from_parts(999, 999, true, 0xFFFFFF);
        let encoded = tai.encode();
        let decoded = Tai::decode(encoded);
        assert_eq!(decoded.tac, 0xFFFFFF);
    }

    #[test]
    fn test_tai_default() {
        let tai = Tai::default();
        assert_eq!(tai.plmn.mcc, 0);
        assert_eq!(tai.plmn.mnc, 0);
        assert!(!tai.plmn.long_mnc);
        assert_eq!(tai.tac, 0);
    }

    // SNssai tests

    #[test]
    fn test_snssai_new() {
        let snssai = SNssai::new(1);
        assert_eq!(snssai.sst, 1);
        assert!(snssai.sd.is_none());
    }

    #[test]
    fn test_snssai_with_sd() {
        let snssai = SNssai::with_sd(1, [0x01, 0x02, 0x03]);
        assert_eq!(snssai.sst, 1);
        assert_eq!(snssai.sd, Some([0x01, 0x02, 0x03]));
    }

    #[test]
    fn test_snssai_with_sd_u32() {
        let snssai = SNssai::with_sd_u32(1, 0x010203);
        assert_eq!(snssai.sst, 1);
        assert_eq!(snssai.sd, Some([0x01, 0x02, 0x03]));
    }

    #[test]
    fn test_snssai_sd_as_u32() {
        let snssai = SNssai::with_sd(1, [0x01, 0x02, 0x03]);
        assert_eq!(snssai.sd_as_u32(), Some(0x010203));

        let snssai_no_sd = SNssai::new(1);
        assert_eq!(snssai_no_sd.sd_as_u32(), None);
    }

    #[test]
    fn test_snssai_has_value() {
        let empty = SNssai::default();
        assert!(!empty.has_value());

        let with_sst = SNssai::new(1);
        assert!(with_sst.has_value());

        let with_sd = SNssai::with_sd(0, [0x00, 0x00, 0x01]);
        assert!(with_sd.has_value());
    }

    #[test]
    fn test_snssai_encode_without_sd() {
        let snssai = SNssai::new(1);
        let encoded = snssai.encode();
        assert_eq!(encoded, vec![1]);
    }

    #[test]
    fn test_snssai_encode_with_sd() {
        let snssai = SNssai::with_sd(1, [0x01, 0x02, 0x03]);
        let encoded = snssai.encode();
        assert_eq!(encoded, vec![1, 0x01, 0x02, 0x03]);
    }

    #[test]
    fn test_snssai_decode_without_sd() {
        let decoded = SNssai::decode(&[1]).unwrap();
        assert_eq!(decoded.sst, 1);
        assert!(decoded.sd.is_none());
    }

    #[test]
    fn test_snssai_decode_with_sd() {
        let decoded = SNssai::decode(&[1, 0x01, 0x02, 0x03]).unwrap();
        assert_eq!(decoded.sst, 1);
        assert_eq!(decoded.sd, Some([0x01, 0x02, 0x03]));
    }

    #[test]
    fn test_snssai_decode_invalid() {
        assert!(SNssai::decode(&[]).is_none());
        assert!(SNssai::decode(&[1, 2]).is_none());
        assert!(SNssai::decode(&[1, 2, 3]).is_none());
        assert!(SNssai::decode(&[1, 2, 3, 4, 5]).is_none());
    }

    #[test]
    fn test_snssai_roundtrip_without_sd() {
        let original = SNssai::new(1);
        let encoded = original.encode();
        let decoded = SNssai::decode(&encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_snssai_roundtrip_with_sd() {
        let original = SNssai::with_sd(1, [0xAB, 0xCD, 0xEF]);
        let encoded = original.encode();
        let decoded = SNssai::decode(&encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_snssai_display_without_sd() {
        let snssai = SNssai::new(1);
        assert_eq!(format!("{snssai}"), "1");
    }

    #[test]
    fn test_snssai_display_with_sd() {
        let snssai = SNssai::with_sd(1, [0x01, 0x02, 0x03]);
        assert_eq!(format!("{snssai}"), "1-010203");
    }

    #[test]
    fn test_snssai_debug_without_sd() {
        let snssai = SNssai::new(1);
        assert_eq!(format!("{snssai:?}"), "SNssai(sst=1)");
    }

    #[test]
    fn test_snssai_debug_with_sd() {
        let snssai = SNssai::with_sd(1, [0x01, 0x02, 0x03]);
        assert_eq!(format!("{snssai:?}"), "SNssai(sst=1, sd=010203)");
    }

    #[test]
    fn test_snssai_equality() {
        let s1 = SNssai::new(1);
        let s2 = SNssai::new(1);
        let s3 = SNssai::new(2);
        let s4 = SNssai::with_sd(1, [0x01, 0x02, 0x03]);
        assert_eq!(s1, s2);
        assert_ne!(s1, s3);
        assert_ne!(s1, s4);
    }

    #[test]
    fn test_snssai_default() {
        let snssai = SNssai::default();
        assert_eq!(snssai.sst, 0);
        assert!(snssai.sd.is_none());
    }

    // NetworkSlice tests

    #[test]
    fn test_network_slice_new() {
        let ns = NetworkSlice::new();
        assert!(ns.is_empty());
        assert_eq!(ns.len(), 0);
    }

    #[test]
    fn test_network_slice_from_slices() {
        let slices = vec![SNssai::new(1), SNssai::new(2)];
        let ns = NetworkSlice::from_slices(slices);
        assert_eq!(ns.len(), 2);
    }

    #[test]
    fn test_network_slice_add_if_not_exists() {
        let mut ns = NetworkSlice::new();
        assert!(ns.add_if_not_exists(SNssai::new(1)));
        assert_eq!(ns.len(), 1);

        // Adding same slice should return false
        assert!(!ns.add_if_not_exists(SNssai::new(1)));
        assert_eq!(ns.len(), 1);

        // Adding different slice should return true
        assert!(ns.add_if_not_exists(SNssai::new(2)));
        assert_eq!(ns.len(), 2);
    }

    #[test]
    fn test_network_slice_contains() {
        let mut ns = NetworkSlice::new();
        ns.add_if_not_exists(SNssai::new(1));

        assert!(ns.contains(&SNssai::new(1)));
        assert!(!ns.contains(&SNssai::new(2)));
    }

    #[test]
    fn test_network_slice_iter() {
        let slices = vec![SNssai::new(1), SNssai::new(2)];
        let ns = NetworkSlice::from_slices(slices.clone());

        let collected: Vec<_> = ns.iter().copied().collect();
        assert_eq!(collected, slices);
    }

    #[test]
    fn test_network_slice_default() {
        let ns = NetworkSlice::default();
        assert!(ns.is_empty());
    }

    #[test]
    fn test_network_slice_equality() {
        let ns1 = NetworkSlice::from_slices(vec![SNssai::new(1), SNssai::new(2)]);
        let ns2 = NetworkSlice::from_slices(vec![SNssai::new(1), SNssai::new(2)]);
        let ns3 = NetworkSlice::from_slices(vec![SNssai::new(1)]);
        assert_eq!(ns1, ns2);
        assert_ne!(ns1, ns3);
    }

    // Supi tests

    #[test]
    fn test_supi_new() {
        let supi = Supi::new(SupiType::Imsi, "310410123456789");
        assert_eq!(supi.supi_type, SupiType::Imsi);
        assert_eq!(supi.value, "310410123456789");
    }

    #[test]
    fn test_supi_imsi() {
        let supi = Supi::imsi("310410123456789");
        assert_eq!(supi.supi_type, SupiType::Imsi);
        assert_eq!(supi.value, "310410123456789");
    }

    #[test]
    fn test_supi_nai() {
        let supi = Supi::nai("user@example.com");
        assert_eq!(supi.supi_type, SupiType::Nai);
        assert_eq!(supi.value, "user@example.com");
    }

    #[test]
    fn test_supi_parse_imsi() {
        let supi = Supi::parse("imsi-310410123456789").unwrap();
        assert_eq!(supi.supi_type, SupiType::Imsi);
        assert_eq!(supi.value, "310410123456789");
    }

    #[test]
    fn test_supi_parse_nai() {
        let supi = Supi::parse("nai-user@example.com").unwrap();
        assert_eq!(supi.supi_type, SupiType::Nai);
        assert_eq!(supi.value, "user@example.com");
    }

    #[test]
    fn test_supi_parse_case_insensitive() {
        let supi = Supi::parse("IMSI-310410123456789").unwrap();
        assert_eq!(supi.supi_type, SupiType::Imsi);
    }

    #[test]
    fn test_supi_parse_invalid() {
        assert!(Supi::parse("invalid").is_none());
        assert!(Supi::parse("unknown-123").is_none());
        assert!(Supi::parse("").is_none());
    }

    #[test]
    fn test_supi_has_value() {
        let supi = Supi::imsi("310410123456789");
        assert!(supi.has_value());

        let empty = Supi::imsi("");
        assert!(!empty.has_value());
    }

    #[test]
    fn test_supi_display() {
        let supi = Supi::imsi("310410123456789");
        assert_eq!(format!("{supi}"), "imsi-310410123456789");

        let nai = Supi::nai("user@example.com");
        assert_eq!(format!("{nai}"), "nai-user@example.com");
    }

    #[test]
    fn test_supi_debug() {
        let supi = Supi::imsi("310410123456789");
        assert_eq!(format!("{supi:?}"), "Supi(imsi-310410123456789)");
    }

    #[test]
    fn test_supi_equality() {
        let s1 = Supi::imsi("310410123456789");
        let s2 = Supi::imsi("310410123456789");
        let s3 = Supi::imsi("310410123456780");
        let s4 = Supi::nai("310410123456789");
        assert_eq!(s1, s2);
        assert_ne!(s1, s3);
        assert_ne!(s1, s4);
    }

    #[test]
    fn test_supi_type_prefix() {
        assert_eq!(SupiType::Imsi.prefix(), "imsi");
        assert_eq!(SupiType::Nai.prefix(), "nai");
    }

    // Guti tests

    #[test]
    fn test_guti_new() {
        let plmn = Plmn::new(310, 410, true);
        let guti = Guti::new(plmn, 0x12, 0x123, 0x15, 0xABCDEF01);
        assert_eq!(guti.plmn, plmn);
        assert_eq!(guti.amf_region_id, 0x12);
        assert_eq!(guti.amf_set_id, 0x123);
        assert_eq!(guti.amf_pointer, 0x15);
        assert_eq!(guti.tmsi, 0xABCDEF01);
    }

    #[test]
    fn test_guti_from_parts() {
        let guti = Guti::from_parts(310, 410, true, 0x12, 0x123, 0x15, 0xABCDEF01);
        assert_eq!(guti.plmn.mcc, 310);
        assert_eq!(guti.plmn.mnc, 410);
        assert!(guti.plmn.long_mnc);
        assert_eq!(guti.amf_region_id, 0x12);
        assert_eq!(guti.amf_set_id, 0x123);
        assert_eq!(guti.amf_pointer, 0x15);
        assert_eq!(guti.tmsi, 0xABCDEF01);
    }

    #[test]
    fn test_guti_amf_set_id_masking() {
        // AMF Set ID is 10-bit, should be masked
        let guti = Guti::from_parts(310, 410, true, 0x12, 0xFFF, 0x15, 0xABCDEF01);
        assert_eq!(guti.amf_set_id, 0x3FF); // Masked to 10 bits
    }

    #[test]
    fn test_guti_amf_pointer_masking() {
        // AMF Pointer is 6-bit, should be masked
        let guti = Guti::from_parts(310, 410, true, 0x12, 0x123, 0xFF, 0xABCDEF01);
        assert_eq!(guti.amf_pointer, 0x3F); // Masked to 6 bits
    }

    #[test]
    fn test_guti_has_value() {
        let empty = Guti::default();
        assert!(!empty.has_value());

        let with_plmn = Guti::from_parts(310, 0, false, 0, 0, 0, 0);
        assert!(with_plmn.has_value());

        let with_region = Guti::from_parts(0, 0, false, 1, 0, 0, 0);
        assert!(with_region.has_value());

        let with_tmsi = Guti::from_parts(0, 0, false, 0, 0, 0, 1);
        assert!(with_tmsi.has_value());
    }

    #[test]
    fn test_guti_amf_id() {
        let guti = Guti::from_parts(310, 410, true, 0x12, 0x123, 0x15, 0xABCDEF01);
        // AMF ID = region(8) | set(10) | pointer(6) = 0x12 << 16 | 0x123 << 6 | 0x15
        let expected = (0x12 << 16) | (0x123 << 6) | 0x15;
        assert_eq!(guti.amf_id(), expected);
    }

    #[test]
    fn test_guti_from_amf_id() {
        let plmn = Plmn::new(310, 410, true);
        let amf_id = (0x12 << 16) | (0x123 << 6) | 0x15;
        let guti = Guti::from_amf_id(plmn, amf_id, 0xABCDEF01);
        assert_eq!(guti.amf_region_id, 0x12);
        assert_eq!(guti.amf_set_id, 0x123);
        assert_eq!(guti.amf_pointer, 0x15);
        assert_eq!(guti.tmsi, 0xABCDEF01);
    }

    #[test]
    fn test_guti_s_tmsi() {
        let guti = Guti::from_parts(310, 410, true, 0x12, 0x123, 0x15, 0xABCDEF01);
        // S-TMSI = set(10) | pointer(6) | tmsi(32)
        let amf_set_pointer = (0x123_u64 << 6) | 0x15;
        let expected = (amf_set_pointer << 32) | 0xABCDEF01;
        assert_eq!(guti.s_tmsi(), expected);
    }

    #[test]
    fn test_guti_from_s_tmsi() {
        let amf_set_pointer = (0x123_u64 << 6) | 0x15;
        let s_tmsi = (amf_set_pointer << 32) | 0xABCDEF01;
        let guti = Guti::from_s_tmsi(s_tmsi);
        assert_eq!(guti.amf_set_id, 0x123);
        assert_eq!(guti.amf_pointer, 0x15);
        assert_eq!(guti.tmsi, 0xABCDEF01);
        // PLMN and region should be default
        assert_eq!(guti.plmn, Plmn::default());
        assert_eq!(guti.amf_region_id, 0);
    }

    #[test]
    fn test_guti_encode() {
        let guti = Guti::from_parts(310, 410, true, 0x12, 0x123, 0x15, 0xABCDEF01);
        let encoded = guti.encode();
        // PLMN: [0x13, 0x00, 0x14]
        // AMF Region ID: 0x12
        // AMF Set ID (10) | AMF Pointer (6): (0x123 << 6) | 0x15 = 0x48D5 -> [0x48, 0xD5]
        // TMSI: [0xAB, 0xCD, 0xEF, 0x01]
        assert_eq!(encoded[0..3], [0x13, 0x00, 0x14]); // PLMN
        assert_eq!(encoded[3], 0x12); // AMF Region ID
        let amf_set_pointer = (0x123_u16 << 6) | 0x15;
        assert_eq!(encoded[4], (amf_set_pointer >> 8) as u8);
        assert_eq!(encoded[5], (amf_set_pointer & 0xFF) as u8);
        assert_eq!(encoded[6..10], [0xAB, 0xCD, 0xEF, 0x01]); // TMSI
    }

    #[test]
    fn test_guti_decode() {
        let plmn_bytes = Plmn::new(310, 410, true).encode();
        let amf_set_pointer: u16 = (0x123 << 6) | 0x15;
        let bytes = [
            plmn_bytes[0],
            plmn_bytes[1],
            plmn_bytes[2],
            0x12,
            (amf_set_pointer >> 8) as u8,
            (amf_set_pointer & 0xFF) as u8,
            0xAB,
            0xCD,
            0xEF,
            0x01,
        ];
        let guti = Guti::decode(bytes);
        assert_eq!(guti.plmn.mcc, 310);
        assert_eq!(guti.plmn.mnc, 410);
        assert!(guti.plmn.long_mnc);
        assert_eq!(guti.amf_region_id, 0x12);
        assert_eq!(guti.amf_set_id, 0x123);
        assert_eq!(guti.amf_pointer, 0x15);
        assert_eq!(guti.tmsi, 0xABCDEF01);
    }

    #[test]
    fn test_guti_roundtrip() {
        let original = Guti::from_parts(234, 150, true, 0xAB, 0x2AA, 0x2A, 0x12345678);
        let encoded = original.encode();
        let decoded = Guti::decode(encoded);
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_guti_roundtrip_2digit_mnc() {
        let original = Guti::from_parts(234, 15, false, 0x01, 0x001, 0x01, 0x00000001);
        let encoded = original.encode();
        let decoded = Guti::decode(encoded);
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_guti_display() {
        let guti = Guti::from_parts(310, 410, true, 0x12, 0x123, 0x15, 0xABCDEF01);
        // Format: PLMN-RegionID-SetID-Pointer-TMSI
        assert_eq!(format!("{guti}"), "310410-12-123-15-ABCDEF01");
    }

    #[test]
    fn test_guti_debug() {
        let guti = Guti::from_parts(310, 410, true, 0x12, 0x123, 0x15, 0xABCDEF01);
        let debug_str = format!("{guti:?}");
        assert!(debug_str.contains("Guti"));
        assert!(debug_str.contains("310"));
        assert!(debug_str.contains("410"));
    }

    #[test]
    fn test_guti_equality() {
        let g1 = Guti::from_parts(310, 410, true, 0x12, 0x123, 0x15, 0xABCDEF01);
        let g2 = Guti::from_parts(310, 410, true, 0x12, 0x123, 0x15, 0xABCDEF01);
        let g3 = Guti::from_parts(310, 410, true, 0x12, 0x123, 0x15, 0xABCDEF02);
        let g4 = Guti::from_parts(310, 410, false, 0x12, 0x123, 0x15, 0xABCDEF01);
        assert_eq!(g1, g2);
        assert_ne!(g1, g3);
        assert_ne!(g1, g4);
    }

    #[test]
    fn test_guti_default() {
        let guti = Guti::default();
        assert_eq!(guti.plmn, Plmn::default());
        assert_eq!(guti.amf_region_id, 0);
        assert_eq!(guti.amf_set_id, 0);
        assert_eq!(guti.amf_pointer, 0);
        assert_eq!(guti.tmsi, 0);
    }

    #[test]
    fn test_guti_max_values() {
        let guti = Guti::from_parts(999, 999, true, 0xFF, 0x3FF, 0x3F, 0xFFFFFFFF);
        let encoded = guti.encode();
        let decoded = Guti::decode(encoded);
        assert_eq!(decoded.amf_region_id, 0xFF);
        assert_eq!(decoded.amf_set_id, 0x3FF);
        assert_eq!(decoded.amf_pointer, 0x3F);
        assert_eq!(decoded.tmsi, 0xFFFFFFFF);
    }

    // 6G ISAC tests

    #[test]
    fn test_sensing_measurement_new() {
        let measurement = SensingMeasurement::new(1000, 100.0, 5.0, 45.0, 10.0, -80.0);
        assert_eq!(measurement.timestamp_ms, 1000);
        assert_eq!(measurement.range_meters, 100.0);
        assert_eq!(measurement.velocity_mps, 5.0);
        assert_eq!(measurement.azimuth_deg, 45.0);
        assert_eq!(measurement.elevation_deg, 10.0);
        assert_eq!(measurement.signal_strength_dbm, -80.0);
        assert_eq!(measurement.target_id, 0);
        assert!(measurement.is_valid());
    }

    #[test]
    fn test_sensing_measurement_invalid() {
        let measurement = SensingMeasurement::new(1000, 0.0, 0.0, 0.0, 0.0, -130.0);
        assert!(!measurement.is_valid());
    }

    #[test]
    fn test_sensing_config_new() {
        let config = SensingConfig::new(1, 200, 500.0);
        assert!(config.enabled);
        assert_eq!(config.mode, 1);
        assert_eq!(config.bandwidth_mhz, 200);
        assert_eq!(config.max_range_meters, 500.0);
    }

    #[test]
    fn test_sensing_config_default() {
        let config = SensingConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.mode, 0);
        assert_eq!(config.bandwidth_mhz, 100);
    }

    #[test]
    fn test_sensing_result_new() {
        let measurements = vec![
            SensingMeasurement::new(1000, 100.0, 5.0, 45.0, 10.0, -80.0),
            SensingMeasurement::new(1000, 150.0, -3.0, 90.0, 5.0, -85.0),
        ];
        let result = SensingResult::new(1000, measurements);
        assert_eq!(result.detection_count, 2);
        assert_eq!(result.measurements.len(), 2);
    }

    // Semantic communication tests

    #[test]
    fn test_modality_type_display() {
        assert_eq!(ModalityType::Text.to_string(), "text");
        assert_eq!(ModalityType::Image.to_string(), "image");
        assert_eq!(ModalityType::Audio.to_string(), "audio");
    }

    #[test]
    fn test_compression_level_display() {
        assert_eq!(CompressionLevel::None.to_string(), "none");
        assert_eq!(CompressionLevel::Medium.to_string(), "medium");
        assert_eq!(CompressionLevel::Maximum.to_string(), "maximum");
    }

    #[test]
    fn test_semantic_profile_new() {
        let profile = SemanticProfile::new(ModalityType::Video, CompressionLevel::High, 0.9);
        assert_eq!(profile.modality, ModalityType::Video);
        assert_eq!(profile.compression, CompressionLevel::High);
        assert_eq!(profile.importance, 0.9);
    }

    #[test]
    fn test_semantic_profile_importance_clamping() {
        let profile = SemanticProfile::new(ModalityType::Text, CompressionLevel::Low, 1.5);
        assert_eq!(profile.importance, 1.0);

        let profile2 = SemanticProfile::new(ModalityType::Text, CompressionLevel::Low, -0.5);
        assert_eq!(profile2.importance, 0.0);
    }

    #[test]
    fn test_semantic_profile_default() {
        let profile = SemanticProfile::default();
        assert_eq!(profile.modality, ModalityType::Text);
        assert_eq!(profile.compression, CompressionLevel::Medium);
        assert_eq!(profile.importance, 0.5);
    }

    // Federated learning tests

    #[test]
    fn test_model_id_new() {
        let id = ModelId::new(12345);
        assert_eq!(id.value(), 12345);
    }

    #[test]
    fn test_model_id_display() {
        let id = ModelId::new(0xABCD);
        assert_eq!(id.to_string(), "model-000000000000abcd");
    }

    #[test]
    fn test_model_version_new() {
        let version = ModelVersion::new(1, 2, 3);
        assert_eq!(version.major, 1);
        assert_eq!(version.minor, 2);
        assert_eq!(version.patch, 3);
    }

    #[test]
    fn test_model_version_display() {
        let version = ModelVersion::new(2, 5, 10);
        assert_eq!(version.to_string(), "2.5.10");
    }

    #[test]
    fn test_model_version_ordering() {
        let v1 = ModelVersion::new(1, 0, 0);
        let v2 = ModelVersion::new(1, 1, 0);
        let v3 = ModelVersion::new(2, 0, 0);
        assert!(v1 < v2);
        assert!(v2 < v3);
    }

    #[test]
    fn test_training_round_new() {
        let round = TrainingRound::new(10);
        assert_eq!(round.value(), 10);
    }

    #[test]
    fn test_training_round_next() {
        let round = TrainingRound::new(5);
        let next = round.next();
        assert_eq!(next.value(), 6);
    }

    #[test]
    fn test_training_round_display() {
        let round = TrainingRound::new(42);
        assert_eq!(round.to_string(), "round-42");
    }

    // SHE computing tests

    #[test]
    fn test_accelerator_type_display() {
        assert_eq!(AcceleratorType::Gpu.to_string(), "gpu");
        assert_eq!(AcceleratorType::Tpu.to_string(), "tpu");
        assert_eq!(AcceleratorType::Npu.to_string(), "npu");
    }

    #[test]
    fn test_placement_constraint_new() {
        let constraint = PlacementConstraint::new(50, AcceleratorType::Gpu);
        assert_eq!(constraint.max_latency_ms, 50);
        assert_eq!(constraint.required_accelerator, AcceleratorType::Gpu);
    }

    #[test]
    fn test_placement_constraint_default() {
        let constraint = PlacementConstraint::default();
        assert_eq!(constraint.max_latency_ms, 1000);
        assert_eq!(constraint.required_accelerator, AcceleratorType::Cpu);
    }

    #[test]
    fn test_compute_request_new() {
        let request = ComputeRequest::new(12345, "inference".to_string(), 1000);
        assert_eq!(request.request_id, 12345);
        assert_eq!(request.task_type, "inference");
        assert_eq!(request.workload_units, 1000);
        assert_eq!(request.priority, 128);
    }

    #[test]
    fn test_compute_descriptor_new() {
        let request = ComputeRequest::new(1, "training".to_string(), 5000);
        let descriptor = ComputeDescriptor::new(request, 999);
        assert_eq!(descriptor.session_id, 999);
        assert!(!descriptor.real_time);
    }

    // Construction tests verify that all types can be created and compared

    #[test]
    fn test_sensing_measurement_construction_and_equality() {
        let measurement1 = SensingMeasurement::new(1000, 100.0, 5.0, 45.0, 10.0, -80.0);
        let measurement2 = SensingMeasurement::new(1000, 100.0, 5.0, 45.0, 10.0, -80.0);
        assert_eq!(measurement1, measurement2);
    }

    #[test]
    fn test_semantic_profile_construction_and_equality() {
        let profile1 = SemanticProfile::new(ModalityType::Audio, CompressionLevel::High, 0.8);
        let profile2 = SemanticProfile::new(ModalityType::Audio, CompressionLevel::High, 0.8);
        assert_eq!(profile1, profile2);
    }

    #[test]
    fn test_compute_request_construction_and_equality() {
        let request1 = ComputeRequest::new(999, "inference".to_string(), 2000);
        let request2 = ComputeRequest::new(999, "inference".to_string(), 2000);
        assert_eq!(request1, request2);
    }
}

/// Tracking Area Identity (TAI)
///
/// A TAI uniquely identifies a tracking area within a PLMN and consists of:
/// - PLMN: The Public Land Mobile Network identifier
/// - TAC: Tracking Area Code (24-bit value, range 0-16777215)
///
/// TAI is used in 5G networks for mobility management and paging.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[derive(Default)]
pub struct Tai {
    /// Public Land Mobile Network identifier
    pub plmn: Plmn,
    /// Tracking Area Code (24-bit, range 0-16777215)
    pub tac: u32,
}

impl Tai {
    /// Creates a new TAI with the given PLMN and TAC.
    ///
    /// # Arguments
    /// * `plmn` - The PLMN identifier
    /// * `tac` - Tracking Area Code (24-bit value)
    pub const fn new(plmn: Plmn, tac: u32) -> Self {
        Self { plmn, tac }
    }

    /// Creates a new TAI from individual MCC, MNC, and TAC values.
    ///
    /// # Arguments
    /// * `mcc` - Mobile Country Code (3 digits)
    /// * `mnc` - Mobile Network Code (2-3 digits)
    /// * `long_mnc` - Whether MNC is 3 digits
    /// * `tac` - Tracking Area Code (24-bit value)
    pub const fn from_parts(mcc: u16, mnc: u16, long_mnc: bool, tac: u32) -> Self {
        Self {
            plmn: Plmn::new(mcc, mnc, long_mnc),
            tac,
        }
    }

    /// Returns true if this TAI has valid values set.
    ///
    /// A TAI is considered to have a value if either the PLMN has a value
    /// or the TAC is non-zero.
    pub fn has_value(&self) -> bool {
        self.plmn.has_value() || self.tac > 0
    }

    /// Encodes the TAI to 3GPP format (6 bytes).
    ///
    /// The encoding follows 3GPP TS 24.501 format:
    /// - Bytes 0-2: PLMN in 3GPP encoding
    /// - Bytes 3-5: TAC in big-endian format (24-bit)
    pub fn encode(&self) -> [u8; 6] {
        let plmn_bytes = self.plmn.encode();
        let tac_bytes = [
            ((self.tac >> 16) & 0xFF) as u8,
            ((self.tac >> 8) & 0xFF) as u8,
            (self.tac & 0xFF) as u8,
        ];

        [
            plmn_bytes[0],
            plmn_bytes[1],
            plmn_bytes[2],
            tac_bytes[0],
            tac_bytes[1],
            tac_bytes[2],
        ]
    }

    /// Decodes a TAI from 3GPP format (6 bytes).
    ///
    /// # Arguments
    /// * `bytes` - 6-byte array in 3GPP TAI encoding format
    ///
    /// # Returns
    /// The decoded TAI
    pub fn decode(bytes: [u8; 6]) -> Self {
        let plmn = Plmn::decode([bytes[0], bytes[1], bytes[2]]);
        let tac = ((bytes[3] as u32) << 16) | ((bytes[4] as u32) << 8) | (bytes[5] as u32);

        Self { plmn, tac }
    }
}

impl fmt::Debug for Tai {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tai({:?}, tac={})", self.plmn, self.tac)
    }
}

impl fmt::Display for Tai {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-{}", self.plmn, self.tac)
    }
}


/// Single Network Slice Selection Assistance Information (S-NSSAI)
///
/// S-NSSAI identifies a network slice and consists of:
/// - SST (Slice/Service Type): 8-bit value identifying the slice type
/// - SD (Slice Differentiator): Optional 24-bit value for further differentiation
///
/// Standard SST values (3GPP TS 23.501):
/// - 1: eMBB (enhanced Mobile Broadband)
/// - 2: URLLC (Ultra-Reliable Low-Latency Communications)
/// - 3: MIoT (Massive IoT)
/// - 4: V2X (Vehicle-to-Everything)
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[derive(Default)]
pub struct SNssai {
    /// Slice/Service Type (8-bit)
    pub sst: u8,
    /// Slice Differentiator (optional 24-bit value)
    pub sd: Option<[u8; 3]>,
}

impl SNssai {
    /// Creates a new S-NSSAI with only SST (no SD).
    ///
    /// # Arguments
    /// * `sst` - Slice/Service Type
    pub const fn new(sst: u8) -> Self {
        Self { sst, sd: None }
    }

    /// Creates a new S-NSSAI with SST and SD.
    ///
    /// # Arguments
    /// * `sst` - Slice/Service Type
    /// * `sd` - Slice Differentiator (24-bit value as 3 bytes)
    pub const fn with_sd(sst: u8, sd: [u8; 3]) -> Self {
        Self { sst, sd: Some(sd) }
    }

    /// Creates a new S-NSSAI with SST and SD from a u32 value.
    ///
    /// # Arguments
    /// * `sst` - Slice/Service Type
    /// * `sd` - Slice Differentiator as u32 (only lower 24 bits used)
    pub const fn with_sd_u32(sst: u8, sd: u32) -> Self {
        Self {
            sst,
            sd: Some([
                ((sd >> 16) & 0xFF) as u8,
                ((sd >> 8) & 0xFF) as u8,
                (sd & 0xFF) as u8,
            ]),
        }
    }

    /// Returns the SD as a u32 value, or None if SD is not set.
    pub fn sd_as_u32(&self) -> Option<u32> {
        self.sd.map(|sd| ((sd[0] as u32) << 16) | ((sd[1] as u32) << 8) | (sd[2] as u32))
    }

    /// Returns true if this S-NSSAI has a valid SST value set.
    pub fn has_value(&self) -> bool {
        self.sst > 0 || self.sd.is_some()
    }

    /// Encodes the S-NSSAI to 3GPP format.
    ///
    /// The encoding follows 3GPP TS 24.501:
    /// - 1 byte: SST
    /// - 3 bytes (optional): SD in big-endian format
    ///
    /// Returns 1 byte if SD is None, 4 bytes if SD is present.
    pub fn encode(&self) -> Vec<u8> {
        match self.sd {
            Some(sd) => vec![self.sst, sd[0], sd[1], sd[2]],
            None => vec![self.sst],
        }
    }

    /// Decodes an S-NSSAI from 3GPP format.
    ///
    /// # Arguments
    /// * `bytes` - Byte slice containing the encoded S-NSSAI (1 or 4 bytes)
    ///
    /// # Returns
    /// The decoded S-NSSAI, or None if the input is invalid
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        match bytes.len() {
            1 => Some(Self {
                sst: bytes[0],
                sd: None,
            }),
            4 => Some(Self {
                sst: bytes[0],
                sd: Some([bytes[1], bytes[2], bytes[3]]),
            }),
            _ => None,
        }
    }
}

impl fmt::Debug for SNssai {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.sd {
            Some(sd) => {
                let sd_val = ((sd[0] as u32) << 16) | ((sd[1] as u32) << 8) | (sd[2] as u32);
                write!(f, "SNssai(sst={}, sd={:06X})", self.sst, sd_val)
            }
            None => write!(f, "SNssai(sst={})", self.sst),
        }
    }
}

impl fmt::Display for SNssai {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.sd {
            Some(sd) => {
                let sd_val = ((sd[0] as u32) << 16) | ((sd[1] as u32) << 8) | (sd[2] as u32);
                write!(f, "{}-{:06X}", self.sst, sd_val)
            }
            None => write!(f, "{}", self.sst),
        }
    }
}


/// Network Slice configuration containing multiple S-NSSAIs.
///
/// This represents a collection of network slices that a UE or gNB supports.
/// Duplicate slices are automatically prevented when using `add_if_not_exists`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NetworkSlice {
    /// List of S-NSSAIs in this network slice configuration
    pub slices: Vec<SNssai>,
}

impl NetworkSlice {
    /// Creates a new empty NetworkSlice.
    pub const fn new() -> Self {
        Self { slices: Vec::new() }
    }

    /// Creates a NetworkSlice from a vector of S-NSSAIs.
    pub fn from_slices(slices: Vec<SNssai>) -> Self {
        Self { slices }
    }

    /// Adds an S-NSSAI if it doesn't already exist in the collection.
    ///
    /// # Arguments
    /// * `slice` - The S-NSSAI to add
    ///
    /// # Returns
    /// `true` if the slice was added, `false` if it already existed
    pub fn add_if_not_exists(&mut self, slice: SNssai) -> bool {
        if !self.slices.contains(&slice) {
            self.slices.push(slice);
            true
        } else {
            false
        }
    }

    /// Returns true if the collection contains the given S-NSSAI.
    pub fn contains(&self, slice: &SNssai) -> bool {
        self.slices.contains(slice)
    }

    /// Returns the number of S-NSSAIs in the collection.
    pub fn len(&self) -> usize {
        self.slices.len()
    }

    /// Returns true if the collection is empty.
    pub fn is_empty(&self) -> bool {
        self.slices.is_empty()
    }

    /// Returns an iterator over the S-NSSAIs.
    pub fn iter(&self) -> impl Iterator<Item = &SNssai> {
        self.slices.iter()
    }
}

/// SUPI type enumeration.
///
/// Defines the type of Subscription Permanent Identifier per 3GPP TS 23.003.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SupiType {
    /// International Mobile Subscriber Identity (IMSI-based SUPI)
    Imsi,
    /// Network Access Identifier (NAI-based SUPI)
    Nai,
}

impl SupiType {
    /// Returns the string prefix for this SUPI type.
    pub fn prefix(&self) -> &'static str {
        match self {
            SupiType::Imsi => "imsi",
            SupiType::Nai => "nai",
        }
    }
}

impl fmt::Display for SupiType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.prefix())
    }
}

/// Subscription Permanent Identifier (SUPI).
///
/// SUPI is the permanent identity of a subscriber in 5G networks.
/// It can be either IMSI-based or NAI-based per 3GPP TS 23.003.
///
/// Format: `<type>-<value>` (e.g., "imsi-310410123456789")
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Supi {
    /// The type of SUPI (IMSI or NAI)
    pub supi_type: SupiType,
    /// The SUPI value (e.g., "310410123456789" for IMSI)
    pub value: String,
}

impl Supi {
    /// Creates a new SUPI with the given type and value.
    ///
    /// # Arguments
    /// * `supi_type` - The type of SUPI (IMSI or NAI)
    /// * `value` - The SUPI value string
    pub fn new(supi_type: SupiType, value: impl Into<String>) -> Self {
        Self {
            supi_type,
            value: value.into(),
        }
    }

    /// Creates a new IMSI-based SUPI.
    ///
    /// # Arguments
    /// * `value` - The IMSI value (e.g., "310410123456789")
    pub fn imsi(value: impl Into<String>) -> Self {
        Self::new(SupiType::Imsi, value)
    }

    /// Creates a new NAI-based SUPI.
    ///
    /// # Arguments
    /// * `value` - The NAI value
    pub fn nai(value: impl Into<String>) -> Self {
        Self::new(SupiType::Nai, value)
    }

    /// Parses a SUPI from a string in the format "type-value".
    ///
    /// # Arguments
    /// * `s` - The SUPI string (e.g., "imsi-310410123456789")
    ///
    /// # Returns
    /// The parsed SUPI, or None if the format is invalid
    pub fn parse(s: &str) -> Option<Self> {
        let (type_str, value) = s.split_once('-')?;
        let supi_type = match type_str.to_lowercase().as_str() {
            "imsi" => SupiType::Imsi,
            "nai" => SupiType::Nai,
            _ => return None,
        };
        Some(Self::new(supi_type, value))
    }

    /// Returns true if this SUPI has a non-empty value.
    pub fn has_value(&self) -> bool {
        !self.value.is_empty()
    }
}

impl fmt::Debug for Supi {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Supi({}-{})", self.supi_type, self.value)
    }
}

impl fmt::Display for Supi {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-{}", self.supi_type, self.value)
    }
}

/// 5G Globally Unique Temporary Identifier (5G-GUTI).
///
/// GUTI is a temporary identity assigned to a UE by the AMF.
/// It consists of:
/// - PLMN: Public Land Mobile Network identifier
/// - AMF Region ID: 8-bit identifier for the AMF region
/// - AMF Set ID: 10-bit identifier for the AMF set within the region
/// - AMF Pointer: 6-bit identifier for the AMF within the set
/// - 5G-TMSI: 32-bit Temporary Mobile Subscriber Identity
///
/// Per 3GPP TS 23.003 Section 2.10.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[derive(Default)]
pub struct Guti {
    /// Public Land Mobile Network identifier
    pub plmn: Plmn,
    /// AMF Region ID (8-bit)
    pub amf_region_id: u8,
    /// AMF Set ID (10-bit, range 0-1023)
    pub amf_set_id: u16,
    /// AMF Pointer (6-bit, range 0-63)
    pub amf_pointer: u8,
    /// 5G Temporary Mobile Subscriber Identity (32-bit)
    pub tmsi: u32,
}

impl Guti {
    /// Maximum value for AMF Set ID (10-bit)
    pub const MAX_AMF_SET_ID: u16 = 0x3FF;
    /// Maximum value for AMF Pointer (6-bit)
    pub const MAX_AMF_POINTER: u8 = 0x3F;

    /// Creates a new GUTI with the given values.
    ///
    /// # Arguments
    /// * `plmn` - Public Land Mobile Network identifier
    /// * `amf_region_id` - AMF Region ID (8-bit)
    /// * `amf_set_id` - AMF Set ID (10-bit, will be masked to 10 bits)
    /// * `amf_pointer` - AMF Pointer (6-bit, will be masked to 6 bits)
    /// * `tmsi` - 5G-TMSI (32-bit)
    pub fn new(plmn: Plmn, amf_region_id: u8, amf_set_id: u16, amf_pointer: u8, tmsi: u32) -> Self {
        Self {
            plmn,
            amf_region_id,
            amf_set_id: amf_set_id & Self::MAX_AMF_SET_ID,
            amf_pointer: amf_pointer & Self::MAX_AMF_POINTER,
            tmsi,
        }
    }

    /// Creates a new GUTI from individual PLMN components.
    ///
    /// # Arguments
    /// * `mcc` - Mobile Country Code
    /// * `mnc` - Mobile Network Code
    /// * `long_mnc` - Whether MNC is 3 digits
    /// * `amf_region_id` - AMF Region ID (8-bit)
    /// * `amf_set_id` - AMF Set ID (10-bit)
    /// * `amf_pointer` - AMF Pointer (6-bit)
    /// * `tmsi` - 5G-TMSI (32-bit)
    pub fn from_parts(
        mcc: u16,
        mnc: u16,
        long_mnc: bool,
        amf_region_id: u8,
        amf_set_id: u16,
        amf_pointer: u8,
        tmsi: u32,
    ) -> Self {
        Self::new(
            Plmn::new(mcc, mnc, long_mnc),
            amf_region_id,
            amf_set_id,
            amf_pointer,
            tmsi,
        )
    }

    /// Returns true if this GUTI has valid values set.
    pub fn has_value(&self) -> bool {
        self.plmn.has_value() || self.amf_region_id > 0 || self.amf_set_id > 0 || self.tmsi > 0
    }

    /// Returns the AMF Identifier (AMFI) as a 24-bit value.
    ///
    /// AMFI = AMF Region ID (8 bits) | AMF Set ID (10 bits) | AMF Pointer (6 bits)
    pub fn amf_id(&self) -> u32 {
        ((self.amf_region_id as u32) << 16)
            | ((self.amf_set_id as u32 & 0x3FF) << 6)
            | (self.amf_pointer as u32 & 0x3F)
    }

    /// Creates a GUTI from PLMN and AMF ID.
    ///
    /// # Arguments
    /// * `plmn` - Public Land Mobile Network identifier
    /// * `amf_id` - 24-bit AMF Identifier
    /// * `tmsi` - 5G-TMSI (32-bit)
    pub fn from_amf_id(plmn: Plmn, amf_id: u32, tmsi: u32) -> Self {
        let amf_region_id = ((amf_id >> 16) & 0xFF) as u8;
        let amf_set_id = ((amf_id >> 6) & 0x3FF) as u16;
        let amf_pointer = (amf_id & 0x3F) as u8;
        Self::new(plmn, amf_region_id, amf_set_id, amf_pointer, tmsi)
    }

    /// Returns the 5G-S-TMSI as a 48-bit value.
    ///
    /// 5G-S-TMSI = AMF Set ID (10 bits) | AMF Pointer (6 bits) | 5G-TMSI (32 bits)
    pub fn s_tmsi(&self) -> u64 {
        let amf_set_pointer =
            ((self.amf_set_id as u64 & 0x3FF) << 6) | (self.amf_pointer as u64 & 0x3F);
        (amf_set_pointer << 32) | (self.tmsi as u64)
    }

    /// Creates a partial GUTI from 5G-S-TMSI value.
    ///
    /// Note: This creates a GUTI with default PLMN and AMF Region ID,
    /// as those are not included in the S-TMSI.
    ///
    /// # Arguments
    /// * `s_tmsi` - 48-bit 5G-S-TMSI value
    pub fn from_s_tmsi(s_tmsi: u64) -> Self {
        let amf_set_id = ((s_tmsi >> 38) & 0x3FF) as u16;
        let amf_pointer = ((s_tmsi >> 32) & 0x3F) as u8;
        let tmsi = (s_tmsi & 0xFFFFFFFF) as u32;
        Self::new(Plmn::default(), 0, amf_set_id, amf_pointer, tmsi)
    }

    /// Encodes the GUTI to 3GPP format (10 bytes).
    ///
    /// The encoding follows 3GPP TS 24.501:
    /// - Bytes 0-2: PLMN in 3GPP encoding
    /// - Byte 3: AMF Region ID
    /// - Bytes 4-5: AMF Set ID (10 bits) | AMF Pointer (6 bits)
    /// - Bytes 6-9: 5G-TMSI in big-endian format
    pub fn encode(&self) -> [u8; 10] {
        let plmn_bytes = self.plmn.encode();
        let amf_set_pointer =
            ((self.amf_set_id & 0x3FF) << 6) | (self.amf_pointer as u16 & 0x3F);

        [
            plmn_bytes[0],
            plmn_bytes[1],
            plmn_bytes[2],
            self.amf_region_id,
            (amf_set_pointer >> 8) as u8,
            (amf_set_pointer & 0xFF) as u8,
            ((self.tmsi >> 24) & 0xFF) as u8,
            ((self.tmsi >> 16) & 0xFF) as u8,
            ((self.tmsi >> 8) & 0xFF) as u8,
            (self.tmsi & 0xFF) as u8,
        ]
    }

    /// Decodes a GUTI from 3GPP format (10 bytes).
    ///
    /// # Arguments
    /// * `bytes` - 10-byte array in 3GPP GUTI encoding format
    ///
    /// # Returns
    /// The decoded GUTI
    pub fn decode(bytes: [u8; 10]) -> Self {
        let plmn = Plmn::decode([bytes[0], bytes[1], bytes[2]]);
        let amf_region_id = bytes[3];
        let amf_set_pointer = ((bytes[4] as u16) << 8) | (bytes[5] as u16);
        let amf_set_id = (amf_set_pointer >> 6) & 0x3FF;
        let amf_pointer = (amf_set_pointer & 0x3F) as u8;
        let tmsi = ((bytes[6] as u32) << 24)
            | ((bytes[7] as u32) << 16)
            | ((bytes[8] as u32) << 8)
            | (bytes[9] as u32);

        Self {
            plmn,
            amf_region_id,
            amf_set_id,
            amf_pointer,
            tmsi,
        }
    }
}

impl fmt::Debug for Guti {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Guti({:?}, region={}, set={}, ptr={}, tmsi={:08X})",
            self.plmn, self.amf_region_id, self.amf_set_id, self.amf_pointer, self.tmsi
        )
    }
}

impl fmt::Display for Guti {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}-{:02X}-{:03X}-{:02X}-{:08X}",
            self.plmn, self.amf_region_id, self.amf_set_id, self.amf_pointer, self.tmsi
        )
    }
}

