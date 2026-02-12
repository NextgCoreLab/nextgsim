//! UE Task Framework
//!
//! This module implements the actor-based task model with message passing for the UE.
//! Each task runs as an independent async task and communicates via typed message channels.
//!
//! # Architecture
//!
//! The UE uses the following tasks:
//! - **App Task**: Application management, CLI handling, TUN interface, status reporting
//! - **NAS Task**: NAS protocol handling, MM/SM state machines
//! - **RRC Task**: RRC protocol handling, cell selection, connection management
//! - **RLS Task**: Radio Link Simulation, gNB discovery, RRC/data transport
//!
//! # Task Lifecycle
//!
//! Tasks follow a lifecycle managed by `TaskManager`:
//! 1. **Created**: Task is instantiated but not yet running
//! 2. **Running**: Task is actively processing messages
//! 3. **Stopping**: Task received shutdown signal, cleaning up
//! 4. **Stopped**: Task has terminated
//! 5. **Failed**: Task terminated due to an error
//!
//! # Reference
//!
//! Based on UERANSIM's NTS (Network Task System) from `src/ue/nts.hpp`

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use tokio::sync::watch;
use tokio::task::JoinHandle;

use nextgsim_common::config::UeConfig;
use nextgsim_common::types::Tai;
use nextgsim_common::OctetString;
use nextgsim_common::Plmn;
use nextgsim_rls::RrcChannel;

// ============================================================================
// Common Types
// ============================================================================

/// GUTI mobile identity for S-TMSI.
///
/// Based on 3GPP TS 24.501 Section 9.11.3.4.
#[derive(Debug, Clone)]
pub struct GutiMobileIdentity {
    /// PLMN
    pub plmn: Plmn,
    /// AMF region ID
    pub amf_region_id: u8,
    /// AMF set ID (10 bits)
    pub amf_set_id: u16,
    /// AMF pointer (6 bits)
    pub amf_pointer: u8,
    /// 5G-TMSI
    pub tmsi: u32,
}

// ============================================================================
// Task Message Envelope
// ============================================================================

/// Task message envelope wrapping typed messages with control signals.
///
/// This enum provides a uniform way to send messages to tasks while also
/// supporting graceful shutdown signaling.
#[derive(Debug)]
pub enum TaskMessage<T> {
    /// Regular message payload
    Message(T),
    /// Shutdown signal - task should terminate gracefully
    Shutdown,
}

impl<T> TaskMessage<T> {
    /// Creates a new message envelope containing the given payload.
    pub fn message(msg: T) -> Self {
        TaskMessage::Message(msg)
    }

    /// Creates a shutdown signal.
    pub fn shutdown() -> Self {
        TaskMessage::Shutdown
    }

    /// Returns true if this is a shutdown signal.
    pub fn is_shutdown(&self) -> bool {
        matches!(self, TaskMessage::Shutdown)
    }

    /// Unwraps the message payload, panicking if this is a shutdown signal.
    pub fn unwrap(self) -> T {
        match self {
            TaskMessage::Message(msg) => msg,
            TaskMessage::Shutdown => panic!("called unwrap on Shutdown"),
        }
    }

    /// Returns the message payload if present, or None for shutdown.
    pub fn into_message(self) -> Option<T> {
        match self {
            TaskMessage::Message(msg) => Some(msg),
            TaskMessage::Shutdown => None,
        }
    }
}

// ============================================================================
// Task Lifecycle State
// ============================================================================

/// Task lifecycle state.
///
/// Based on UERANSIM's `NtsTask` lifecycle from `src/utils/nts.hpp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum TaskState {
    /// Task is created but not yet started
    #[default]
    Created,
    /// Task is running and processing messages
    Running,
    /// Task is in the process of stopping
    Stopping,
    /// Task has stopped gracefully
    Stopped,
    /// Task terminated due to an error
    Failed,
}


impl std::fmt::Display for TaskState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskState::Created => write!(f, "Created"),
            TaskState::Running => write!(f, "Running"),
            TaskState::Stopping => write!(f, "Stopping"),
            TaskState::Stopped => write!(f, "Stopped"),
            TaskState::Failed => write!(f, "Failed"),
        }
    }
}

/// Task identifier for the UE tasks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskId {
    /// Application task
    App,
    /// NAS task (MM + SM)
    Nas,
    /// RRC task
    Rrc,
    /// RLS task
    Rls,

    // ========================================================================
    // 6G AI-native network function tasks
    // ========================================================================
    /// SHE Client - Service Hosting Environment client for edge inference
    SheClient,
    /// NWDAF Reporter - Reports measurements and receives analytics
    NwdafReporter,
    /// ISAC Sensor - Integrated Sensing and Communication data collection
    IsacSensor,
    /// FL Participant - Federated Learning local training participant
    FlParticipant,
    /// Semantic Codec - Semantic communication encoder/decoder
    SemanticCodec,
}

impl std::fmt::Display for TaskId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskId::App => write!(f, "App"),
            TaskId::Nas => write!(f, "NAS"),
            TaskId::Rrc => write!(f, "RRC"),
            TaskId::Rls => write!(f, "RLS"),
            // 6G AI-native network function tasks
            TaskId::SheClient => write!(f, "SHE-Client"),
            TaskId::NwdafReporter => write!(f, "NWDAF-Reporter"),
            TaskId::IsacSensor => write!(f, "ISAC-Sensor"),
            TaskId::FlParticipant => write!(f, "FL-Participant"),
            TaskId::SemanticCodec => write!(f, "Semantic-Codec"),
        }
    }
}

/// Information about a running task.
#[derive(Debug)]
pub struct TaskInfo {
    /// Task identifier
    pub id: TaskId,
    /// Current state
    pub state: TaskState,
    /// Time when the task was started
    pub started_at: Option<Instant>,
    /// Time when the task was stopped
    pub stopped_at: Option<Instant>,
    /// Error message if task failed
    pub error: Option<String>,
}

// ============================================================================
// Task Trait
// ============================================================================

/// Base trait for all UE tasks.
///
/// Tasks are async actors that process messages from their receive channel.
/// Each task implementation defines its own message type and processing logic.
#[async_trait::async_trait]
pub trait Task: Send + 'static {
    /// The message type this task processes.
    type Message: Send;

    /// Runs the task's main loop, processing messages until shutdown.
    ///
    /// The task should:
    /// 1. Poll the receiver for messages
    /// 2. Process each message according to its type
    /// 3. Exit gracefully when receiving `TaskMessage::Shutdown`
    async fn run(&mut self, rx: mpsc::Receiver<TaskMessage<Self::Message>>);
}


// ============================================================================
// App Task Messages
// ============================================================================

/// Messages for the Application task.
///
/// Based on `NmUeTunToApp`, `NmUeNasToApp`, `NmUeCliCommand` from `src/ue/nts.hpp`.
#[derive(Debug)]
pub enum AppMessage {
    /// Data PDU delivery from TUN interface (uplink)
    TunDataDelivery {
        /// PDU session ID
        psi: i32,
        /// User data
        data: OctetString,
    },
    /// TUN interface error
    TunError {
        /// Error message
        error: String,
    },
    /// Downlink data delivery from NAS (to TUN)
    DownlinkDataDelivery {
        /// PDU session ID
        psi: i32,
        /// User data
        data: OctetString,
    },
    /// Perform switch off (from NAS)
    PerformSwitchOff,
    /// Status update
    StatusUpdate(UeStatusUpdate),
    /// CLI command received
    CliCommand(UeCliCommand),
}

/// UE status update information.
#[derive(Debug, Clone)]
pub enum UeStatusUpdate {
    /// PDU session established
    SessionEstablishment {
        /// PDU session ID
        psi: i32,
    },
    /// PDU session released
    SessionRelease {
        /// PDU session ID
        psi: i32,
    },
    /// CM state changed
    CmStateChanged {
        /// New CM state
        cm_state: CmState,
    },
}

/// Connection Management state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmState {
    /// CM-IDLE state
    Idle,
    /// CM-CONNECTED state
    Connected,
}

impl std::fmt::Display for CmState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CmState::Idle => write!(f, "CM-IDLE"),
            CmState::Connected => write!(f, "CM-CONNECTED"),
        }
    }
}

/// CLI command for the UE.
#[derive(Debug)]
pub struct UeCliCommand {
    /// Command type
    pub command: UeCliCommandType,
    /// Response address (for sending reply)
    pub response_addr: Option<std::net::SocketAddr>,
}

/// Types of CLI commands for UE.
#[derive(Debug, Clone)]
pub enum UeCliCommandType {
    /// Show UE info
    Info,
    /// Show UE status
    Status,
    /// Show timers
    Timers,
    /// Deregister from network
    Deregister {
        /// Switch off flag
        switch_off: bool,
    },
    /// Establish PDU session
    PsEstablish {
        /// Session type (IPv4, IPv6, `IPv4v6`)
        session_type: Option<String>,
        /// APN/DNN
        apn: Option<String>,
        /// S-NSSAI
        s_nssai: Option<String>,
    },
    /// Release PDU session
    PsRelease {
        /// PDU session ID
        psi: i32,
    },
    /// Release all PDU sessions
    PsReleaseAll,
}


// ============================================================================
// NAS Task Messages
// ============================================================================

/// Messages for the NAS task.
///
/// Based on `NmUeRrcToNas`, `NmUeAppToNas`, `NmUeNasToNas` from `src/ue/nts.hpp`.
#[derive(Debug)]
pub enum NasMessage {
    /// NAS notify (from RRC)
    NasNotify,
    /// Downlink NAS delivery (from RRC)
    NasDelivery {
        /// NAS PDU
        pdu: OctetString,
    },
    /// RRC connection setup complete (from RRC)
    RrcConnectionSetup,
    /// RRC connection release (from RRC)
    RrcConnectionRelease,
    /// RRC establishment failure (from RRC)
    RrcEstablishmentFailure,
    /// Radio link failure (from RRC)
    RadioLinkFailure,
    /// Paging indication (from RRC)
    Paging {
        /// Paging TMSI list
        paging_tmsi: Vec<GutiMobileIdentity>,
    },
    /// Active cell changed (from RRC)
    ActiveCellChanged {
        /// Previous TAI
        previous_tai: Tai,
    },
    /// RRC fallback indication (from RRC)
    RrcFallbackIndication,
    /// Uplink data delivery (from App)
    UplinkDataDelivery {
        /// PDU session ID
        psi: i32,
        /// User data
        data: OctetString,
    },
    /// Perform MM cycle (internal)
    PerformMmCycle,
    /// NAS timer expired (internal)
    NasTimerExpire {
        /// Timer ID
        timer_id: i32,
    },
    /// Initiate PDU session establishment (from App)
    InitiatePduSessionEstablishment {
        /// PDU session ID
        psi: u8,
        /// Procedure transaction identity
        pti: u8,
        /// Session type (e.g., "IPv4", "IPv6", "`IPv4v6`")
        session_type: String,
        /// APN/DNN (optional)
        apn: Option<String>,
    },
    /// Initiate PDU session release (from App)
    InitiatePduSessionRelease {
        /// PDU session ID
        psi: u8,
        /// Procedure transaction identity
        pti: u8,
    },
    /// Initiate deregistration (from App)
    InitiateDeregistration {
        /// Deregistration cause
        switch_off: bool,
    },
    /// Downlink data delivery (from NAS for data plane)
    DownlinkDataDelivery {
        /// PDU session ID
        psi: i32,
        /// User data
        data: OctetString,
    },
}

// ============================================================================
// RRC Task Messages
// ============================================================================

/// Messages for the RRC task.
///
/// Based on `NmUeNasToRrc`, `NmUeRlsToRrc`, `NmUeRrcToRrc` from `src/ue/nts.hpp`.
#[derive(Debug)]
pub enum RrcMessage {
    /// Local release connection (from NAS)
    LocalReleaseConnection {
        /// Treat as barred
        treat_barred: bool,
    },
    /// Uplink NAS delivery (from NAS)
    UplinkNasDelivery {
        /// PDU ID for acknowledgment tracking
        pdu_id: u32,
        /// NAS PDU
        pdu: OctetString,
    },
    /// RRC notify (from NAS)
    RrcNotify,
    /// Perform UAC (from NAS)
    PerformUac {
        /// UAC access category
        access_category: i32,
        /// UAC access identities
        access_identities: u32,
    },
    /// Downlink RRC delivery (from RLS)
    DownlinkRrcDelivery {
        /// Cell ID
        cell_id: i32,
        /// RRC channel
        channel: RrcChannel,
        /// RRC PDU
        pdu: OctetString,
    },
    /// Signal changed (from RLS)
    SignalChanged {
        /// Cell ID
        cell_id: i32,
        /// Signal strength in dBm
        dbm: i32,
    },
    /// Radio link failure (from RLS)
    RadioLinkFailure {
        /// Failure cause
        cause: RlfCause,
    },
    /// Trigger cycle (internal)
    TriggerCycle,
    /// NTN timing advance received from gNB (in RRC Setup/Reconfiguration)
    NtnTimingAdvanceReceived {
        /// Common timing advance in microseconds
        common_ta_us: u64,
        /// K-offset for HARQ timing
        k_offset: u16,
        /// Whether UE should compute autonomous TA from ephemeris
        autonomous_ta: bool,
        /// Max Doppler shift in Hz
        max_doppler_hz: f64,
    },
}

/// Radio link failure cause.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RlfCause {
    /// PDU ID already exists
    PduIdExists,
    /// PDU ID buffer full
    PduIdFull,
    /// Signal lost to connected cell
    SignalLostToConnectedCell,
}


// ============================================================================
// RLS Task Messages
// ============================================================================

/// Messages for the RLS task.
///
/// Based on `NmUeRrcToRls`, `NmUeNasToRls`, `NmUeRlsToRls` from `src/ue/nts.hpp`.
#[derive(Debug)]
pub enum RlsMessage {
    /// Assign current cell (from RRC)
    AssignCurrentCell {
        /// Cell ID
        cell_id: i32,
    },
    /// RRC PDU delivery (from RRC)
    RrcPduDelivery {
        /// RRC channel
        channel: RrcChannel,
        /// PDU ID for acknowledgment tracking
        pdu_id: u32,
        /// RRC PDU
        pdu: OctetString,
    },
    /// Reset STI (from RRC)
    ResetSti,
    /// Data PDU delivery (from NAS)
    DataPduDelivery {
        /// PDU session ID
        psi: i32,
        /// User data PDU
        pdu: OctetString,
    },
    /// Received RLS message from network (internal)
    ReceiveRlsMessage {
        /// Cell ID
        cell_id: i32,
        /// Raw RLS message data
        data: OctetString,
    },
    /// Signal changed (internal)
    SignalChanged {
        /// Cell ID
        cell_id: i32,
        /// Signal strength in dBm
        dbm: i32,
    },
    /// Uplink data (internal)
    UplinkData {
        /// PDU session ID
        psi: i32,
        /// User data
        data: OctetString,
    },
    /// Uplink RRC (internal)
    UplinkRrc {
        /// Cell ID
        cell_id: i32,
        /// RRC channel
        channel: RrcChannel,
        /// PDU ID
        pdu_id: u32,
        /// RRC PDU
        data: OctetString,
    },
    /// Downlink data (internal)
    DownlinkData {
        /// PDU session ID
        psi: i32,
        /// User data
        data: OctetString,
    },
    /// Downlink RRC (internal)
    DownlinkRrc {
        /// Cell ID
        cell_id: i32,
        /// RRC channel
        channel: RrcChannel,
        /// RRC PDU
        data: OctetString,
    },
    /// Radio link failure (internal)
    RadioLinkFailure {
        /// Failure cause
        cause: RlfCause,
    },
    /// Transmission failure (internal)
    TransmissionFailure {
        /// List of failed PDU IDs
        pdu_list: Vec<u32>,
    },
}


// ============================================================================
// 6G AI-native Network Function Messages (UE-side)
// ============================================================================

/// Messages for the SHE Client task.
///
/// Handles edge inference requests and workload offloading from the UE.
#[derive(Debug)]
pub enum SheClientMessage {
    /// Request inference at the edge
    InferenceRequest {
        /// Model identifier
        model_id: String,
        /// Input tensor data (flattened)
        input: Vec<f32>,
        /// Input shape
        input_shape: Vec<usize>,
        /// Deadline in milliseconds (0 = no deadline)
        deadline_ms: u32,
        /// Response channel
        response_tx: Option<tokio::sync::oneshot::Sender<SheClientResponse>>,
    },
    /// Cancel pending inference request
    CancelInference {
        /// Request ID to cancel
        request_id: u64,
    },
    /// Update available edge nodes
    EdgeNodeUpdate {
        /// List of available edge node addresses
        nodes: Vec<std::net::SocketAddr>,
    },
    /// Offload computation to edge
    OffloadComputation {
        /// Computation type identifier
        computation_type: String,
        /// Input data
        data: Vec<u8>,
        /// Response channel
        response_tx: Option<tokio::sync::oneshot::Sender<SheClientResponse>>,
    },
}

/// Response from SHE Client operations.
#[derive(Debug)]
pub enum SheClientResponse {
    /// Inference completed successfully
    InferenceResult {
        /// Request ID
        request_id: u64,
        /// Output tensor data
        output: Vec<f32>,
        /// Output shape
        output_shape: Vec<usize>,
        /// Inference latency in milliseconds
        latency_ms: u32,
    },
    /// Computation offload result
    ComputationResult {
        /// Result data
        result: Vec<u8>,
        /// Latency in milliseconds
        latency_ms: u32,
    },
    /// Request failed
    Error {
        /// Error message
        message: String,
    },
}

/// Messages for the NWDAF Reporter task.
///
/// Reports UE measurements to NWDAF and receives analytics/predictions.
#[derive(Debug)]
pub enum NwdafReporterMessage {
    /// Report radio measurement to NWDAF
    ReportMeasurement {
        /// Serving cell RSRP (dBm)
        rsrp: f32,
        /// Serving cell RSRQ (dB)
        rsrq: f32,
        /// Serving cell SINR (dB)
        sinr: Option<f32>,
        /// UE position (x, y, z) if available
        position: Option<(f32, f32, f32)>,
        /// UE velocity (vx, vy, vz) if available
        velocity: Option<(f32, f32, f32)>,
        /// Timestamp in milliseconds
        timestamp_ms: u64,
    },
    /// Report neighbor cell measurements
    ReportNeighborMeasurements {
        /// List of neighbor measurements
        neighbors: Vec<NeighborMeasurementReport>,
    },
    /// Request trajectory prediction
    RequestTrajectoryPrediction {
        /// Prediction horizon in milliseconds
        horizon_ms: u32,
        /// Response channel
        response_tx: Option<tokio::sync::oneshot::Sender<NwdafReporterResponse>>,
    },
    /// Receive handover recommendation from NWDAF
    HandoverRecommendation {
        /// Recommended target cell ID
        target_cell_id: i32,
        /// Confidence score (0.0 to 1.0)
        confidence: f32,
        /// Reason for recommendation
        reason: String,
    },
    /// Update reporting configuration
    UpdateReportingConfig {
        /// Reporting interval in milliseconds
        interval_ms: u32,
        /// Enable position reporting
        report_position: bool,
        /// Enable velocity reporting
        report_velocity: bool,
    },
}

/// Neighbor cell measurement report.
#[derive(Debug, Clone)]
pub struct NeighborMeasurementReport {
    /// Cell ID
    pub cell_id: i32,
    /// RSRP (dBm)
    pub rsrp: f32,
    /// RSRQ (dB)
    pub rsrq: f32,
}

/// Response from NWDAF Reporter operations.
#[derive(Debug)]
pub enum NwdafReporterResponse {
    /// Trajectory prediction result
    TrajectoryPrediction {
        /// Predicted waypoints (position, `timestamp_ms`)
        waypoints: Vec<((f32, f32, f32), u64)>,
        /// Confidence scores for each waypoint
        confidence: Vec<f32>,
    },
    /// Analytics update
    AnalyticsUpdate {
        /// Analytics type
        analytics_type: String,
        /// Analytics data (JSON-encoded)
        data: String,
    },
    /// Error response
    Error {
        /// Error message
        message: String,
    },
}

/// Messages for the ISAC Sensor task.
///
/// Collects sensing data for Integrated Sensing and Communication.
#[derive(Debug)]
pub enum IsacSensorMessage {
    /// Start sensing with given configuration
    StartSensing {
        /// Sensing configuration
        config: IsacSensingConfig,
    },
    /// Stop sensing
    StopSensing,
    /// Report sensing measurement
    SensingMeasurement {
        /// Measurement type
        measurement_type: IsacMeasurementType,
        /// Raw measurement data
        data: Vec<f32>,
        /// Timestamp in milliseconds
        timestamp_ms: u64,
    },
    /// Request fused position estimate
    RequestFusedPosition {
        /// Response channel
        response_tx: Option<tokio::sync::oneshot::Sender<IsacSensorResponse>>,
    },
    /// Receive positioning update from network
    PositioningUpdate {
        /// Estimated position (x, y, z)
        position: (f32, f32, f32),
        /// Position uncertainty (meters)
        uncertainty: f32,
        /// Timestamp in milliseconds
        timestamp_ms: u64,
    },
}

/// ISAC sensing configuration.
#[derive(Debug, Clone)]
pub struct IsacSensingConfig {
    /// Sensing mode
    pub mode: IsacSensingMode,
    /// Measurement interval in milliseconds
    pub interval_ms: u32,
    /// Enable `ToA` (Time of Arrival) measurements
    pub enable_toa: bool,
    /// Enable `AoA` (Angle of Arrival) measurements
    pub enable_aoa: bool,
    /// Enable Doppler measurements
    pub enable_doppler: bool,
}

/// ISAC sensing mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsacSensingMode {
    /// Passive sensing (listen only)
    Passive,
    /// Active sensing (transmit and receive)
    Active,
    /// Cooperative sensing with network
    Cooperative,
}

/// ISAC measurement type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsacMeasurementType {
    /// Time of Arrival
    ToA,
    /// Angle of Arrival
    AoA,
    /// Doppler shift
    Doppler,
    /// Channel State Information
    Csi,
    /// Combined multi-path measurement
    MultiPath,
}

/// Response from ISAC Sensor operations.
#[derive(Debug)]
pub enum IsacSensorResponse {
    /// Fused position estimate
    FusedPosition {
        /// Position (x, y, z)
        position: (f32, f32, f32),
        /// Velocity (vx, vy, vz)
        velocity: (f32, f32, f32),
        /// Position uncertainty (meters)
        position_uncertainty: f32,
        /// Velocity uncertainty (m/s)
        velocity_uncertainty: f32,
        /// Confidence score (0.0 to 1.0)
        confidence: f32,
    },
    /// Error response
    Error {
        /// Error message
        message: String,
    },
}

/// Messages for the FL Participant task.
///
/// Handles local model training and updates for Federated Learning.
#[derive(Debug)]
pub enum FlParticipantMessage {
    /// Receive global model from aggregator
    ReceiveGlobalModel {
        /// Round number
        round: u32,
        /// Model weights (serialized)
        weights: Vec<u8>,
        /// Model version
        version: String,
    },
    /// Start local training
    StartTraining {
        /// Training configuration
        config: FlTrainingConfig,
    },
    /// Add training sample
    AddTrainingSample {
        /// Feature vector
        features: Vec<f32>,
        /// Label
        label: Vec<f32>,
    },
    /// Submit local update to aggregator
    SubmitUpdate {
        /// Response channel for acknowledgment
        response_tx: Option<tokio::sync::oneshot::Sender<FlParticipantResponse>>,
    },
    /// Abort current training round
    AbortTraining {
        /// Reason for abort
        reason: String,
    },
    /// Get current training status
    GetStatus {
        /// Response channel
        response_tx: Option<tokio::sync::oneshot::Sender<FlParticipantResponse>>,
    },
}

/// FL training configuration for local participant.
#[derive(Debug, Clone)]
pub struct FlTrainingConfig {
    /// Number of local epochs
    pub local_epochs: u32,
    /// Batch size
    pub batch_size: u32,
    /// Learning rate
    pub learning_rate: f32,
    /// Enable differential privacy
    pub enable_dp: bool,
    /// Noise multiplier for DP
    pub noise_multiplier: f32,
    /// Gradient clipping threshold
    pub max_grad_norm: f32,
}

impl Default for FlTrainingConfig {
    fn default() -> Self {
        Self {
            local_epochs: 1,
            batch_size: 32,
            learning_rate: 0.01,
            enable_dp: false,
            noise_multiplier: 1.0,
            max_grad_norm: 1.0,
        }
    }
}

/// Response from FL Participant operations.
#[derive(Debug)]
pub enum FlParticipantResponse {
    /// Training completed, update ready
    UpdateReady {
        /// Round number
        round: u32,
        /// Model update weights (serialized)
        weights: Vec<u8>,
        /// Number of samples used
        num_samples: u32,
        /// Local loss after training
        loss: f32,
    },
    /// Current training status
    Status {
        /// Current round
        round: u32,
        /// Current epoch within round
        epoch: u32,
        /// Number of samples processed
        samples_processed: u32,
        /// Current loss
        loss: f32,
        /// Is training active
        is_training: bool,
    },
    /// Update submission acknowledged
    Acknowledged {
        /// Round number
        round: u32,
    },
    /// Error response
    Error {
        /// Error message
        message: String,
    },
}

/// Messages for the Semantic Codec task.
///
/// Handles semantic encoding/decoding for task-oriented communication.
#[derive(Debug)]
pub enum SemanticCodecMessage {
    /// Encode data for transmission
    Encode {
        /// Task type for encoding
        task_type: SemanticTaskType,
        /// Raw input data
        data: Vec<f32>,
        /// Data dimensions
        dimensions: Vec<usize>,
        /// Channel quality information
        channel_quality: Option<ChannelQualityInfo>,
        /// Response channel
        response_tx: Option<tokio::sync::oneshot::Sender<SemanticCodecResponse>>,
    },
    /// Decode received features
    Decode {
        /// Task type for decoding
        task_type: SemanticTaskType,
        /// Received semantic features
        features: Vec<f32>,
        /// Feature importance weights
        importance: Vec<f32>,
        /// Original dimensions
        original_dims: Vec<usize>,
        /// Response channel
        response_tx: Option<tokio::sync::oneshot::Sender<SemanticCodecResponse>>,
    },
    /// Update encoder model
    UpdateEncoder {
        /// Model path or identifier
        model_id: String,
    },
    /// Update decoder model
    UpdateDecoder {
        /// Model path or identifier
        model_id: String,
    },
    /// Set adaptive compression parameters
    SetAdaptiveCompression {
        /// Enable adaptive compression
        enabled: bool,
        /// Minimum quality threshold (0.0 to 1.0)
        min_quality: f32,
        /// Target compression ratio
        target_compression: f32,
    },
}

/// Semantic task type for encoding/decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticTaskType {
    /// Image classification
    ImageClassification,
    /// Object detection
    ObjectDetection,
    /// Speech recognition
    SpeechRecognition,
    /// Sensor data fusion
    SensorFusion,
    /// Video analytics
    VideoAnalytics,
    /// Text understanding
    TextUnderstanding,
    /// Custom task with ID
    Custom(u32),
}

/// Channel quality information for adaptive encoding.
#[derive(Debug, Clone)]
pub struct ChannelQualityInfo {
    /// Signal-to-noise ratio (dB)
    pub snr_db: f32,
    /// Available bandwidth (kHz)
    pub bandwidth_khz: f32,
    /// Packet error rate
    pub per: f32,
}

/// Response from Semantic Codec operations.
#[derive(Debug)]
pub enum SemanticCodecResponse {
    /// Encoding completed
    Encoded {
        /// Semantic features
        features: Vec<f32>,
        /// Importance weights
        importance: Vec<f32>,
        /// Compression ratio achieved
        compression_ratio: f32,
    },
    /// Decoding completed
    Decoded {
        /// Reconstructed data
        data: Vec<f32>,
        /// Reconstruction quality score (0.0 to 1.0)
        quality: f32,
    },
    /// Task-specific output (for classification, detection, etc.)
    TaskOutput {
        /// Task result (interpretation depends on task type)
        result: Vec<f32>,
        /// Confidence score
        confidence: f32,
    },
    /// Error response
    Error {
        /// Error message
        message: String,
    },
}


// ============================================================================
// Task Handle
// ============================================================================

/// Handle for sending messages to a task.
///
/// This is a wrapper around `mpsc::Sender` that provides convenient methods
/// for sending messages and shutdown signals.
#[derive(Debug)]
pub struct TaskHandle<T> {
    tx: mpsc::Sender<TaskMessage<T>>,
}

impl<T> Clone for TaskHandle<T> {
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
        }
    }
}

impl<T> TaskHandle<T> {
    /// Creates a new task handle from a sender.
    pub fn new(tx: mpsc::Sender<TaskMessage<T>>) -> Self {
        Self { tx }
    }

    /// Sends a message to the task.
    ///
    /// Returns an error if the task has been dropped.
    pub async fn send(&self, msg: T) -> Result<(), mpsc::error::SendError<TaskMessage<T>>> {
        self.tx.send(TaskMessage::Message(msg)).await
    }

    /// Sends a message to the task without waiting.
    ///
    /// Returns an error if the channel is full or the task has been dropped.
    pub fn try_send(&self, msg: T) -> Result<(), mpsc::error::TrySendError<TaskMessage<T>>> {
        self.tx.try_send(TaskMessage::Message(msg))
    }

    /// Sends a shutdown signal to the task.
    pub async fn shutdown(&self) -> Result<(), mpsc::error::SendError<TaskMessage<T>>> {
        self.tx.send(TaskMessage::Shutdown).await
    }

    /// Returns true if the task channel is closed.
    pub fn is_closed(&self) -> bool {
        self.tx.is_closed()
    }
}

// ============================================================================
// UE Task Base
// ============================================================================

/// Base structure containing all task handles for the UE.
///
/// This structure is shared among all tasks to enable inter-task communication.
/// Each task receives a clone of this structure and can send messages to any
/// other task through the appropriate handle.
#[derive(Clone)]
pub struct UeTaskBase {
    /// UE configuration
    pub config: Arc<UeConfig>,
    /// Handle to the Application task
    pub app_tx: TaskHandle<AppMessage>,
    /// Handle to the NAS task
    pub nas_tx: TaskHandle<NasMessage>,
    /// Handle to the RRC task
    pub rrc_tx: TaskHandle<RrcMessage>,
    /// Handle to the RLS task
    pub rls_tx: TaskHandle<RlsMessage>,
    /// 6G task handles (initialized via `init_6g_tasks()`)
    pub sixg: Option<UeSixgHandles>,
}

/// 6G task handles for UE (Rel-20 extensions)
#[derive(Clone)]
pub struct UeSixgHandles {
    /// Handle to the SHE Client (edge inference/offloading) task
    pub she_client_tx: TaskHandle<SheClientMessage>,
    /// Handle to the NWDAF Reporter (analytics reporting) task
    pub nwdaf_reporter_tx: TaskHandle<NwdafReporterMessage>,
    /// Handle to the ISAC Sensor (sensing and communication) task
    pub isac_sensor_tx: TaskHandle<IsacSensorMessage>,
    /// Handle to the FL Participant (federated learning) task
    pub fl_participant_tx: TaskHandle<FlParticipantMessage>,
    /// Handle to the Semantic Codec (task-oriented communication) task
    pub semantic_codec_tx: TaskHandle<SemanticCodecMessage>,
}

/// 6G task receivers for UE
pub struct UeSixgReceivers {
    pub she_client_rx: mpsc::Receiver<TaskMessage<SheClientMessage>>,
    pub nwdaf_reporter_rx: mpsc::Receiver<TaskMessage<NwdafReporterMessage>>,
    pub isac_sensor_rx: mpsc::Receiver<TaskMessage<IsacSensorMessage>>,
    pub fl_participant_rx: mpsc::Receiver<TaskMessage<FlParticipantMessage>>,
    pub semantic_codec_rx: mpsc::Receiver<TaskMessage<SemanticCodecMessage>>,
}

impl UeTaskBase {
    /// Creates a new `UeTaskBase` with the given configuration and channel capacity.
    ///
    /// Returns the task base along with receivers for each task.
    #[allow(clippy::type_complexity)]
    pub fn new(
        config: UeConfig,
        channel_capacity: usize,
    ) -> (
        Self,
        mpsc::Receiver<TaskMessage<AppMessage>>,
        mpsc::Receiver<TaskMessage<NasMessage>>,
        mpsc::Receiver<TaskMessage<RrcMessage>>,
        mpsc::Receiver<TaskMessage<RlsMessage>>,
    ) {
        let (app_tx, app_rx) = mpsc::channel(channel_capacity);
        let (nas_tx, nas_rx) = mpsc::channel(channel_capacity);
        let (rrc_tx, rrc_rx) = mpsc::channel(channel_capacity);
        let (rls_tx, rls_rx) = mpsc::channel(channel_capacity);

        let base = Self {
            config: Arc::new(config),
            app_tx: TaskHandle::new(app_tx),
            nas_tx: TaskHandle::new(nas_tx),
            rrc_tx: TaskHandle::new(rrc_tx),
            rls_tx: TaskHandle::new(rls_tx),
            sixg: None,
        };

        (base, app_rx, nas_rx, rrc_rx, rls_rx)
    }

    /// Initialize 6G task handles and return their receivers.
    ///
    /// Call this after `new()` to enable 6G tasks (SHE Client, NWDAF Reporter,
    /// ISAC Sensor, FL Participant, Semantic Codec).
    /// The returned receivers should be used to spawn the 6G task loops.
    pub fn init_6g_tasks(&mut self, channel_capacity: usize) -> UeSixgReceivers {
        let (she_client_tx, she_client_rx) = mpsc::channel(channel_capacity);
        let (nwdaf_reporter_tx, nwdaf_reporter_rx) = mpsc::channel(channel_capacity);
        let (isac_sensor_tx, isac_sensor_rx) = mpsc::channel(channel_capacity);
        let (fl_participant_tx, fl_participant_rx) = mpsc::channel(channel_capacity);
        let (semantic_codec_tx, semantic_codec_rx) = mpsc::channel(channel_capacity);

        self.sixg = Some(UeSixgHandles {
            she_client_tx: TaskHandle::new(she_client_tx),
            nwdaf_reporter_tx: TaskHandle::new(nwdaf_reporter_tx),
            isac_sensor_tx: TaskHandle::new(isac_sensor_tx),
            fl_participant_tx: TaskHandle::new(fl_participant_tx),
            semantic_codec_tx: TaskHandle::new(semantic_codec_tx),
        });

        UeSixgReceivers {
            she_client_rx,
            nwdaf_reporter_rx,
            isac_sensor_rx,
            fl_participant_rx,
            semantic_codec_rx,
        }
    }

    /// Sends shutdown signals to all tasks.
    pub async fn shutdown_all(&self) {
        // Ignore errors - tasks may already be shut down
        let _ = self.app_tx.shutdown().await;
        let _ = self.nas_tx.shutdown().await;
        let _ = self.rrc_tx.shutdown().await;
        let _ = self.rls_tx.shutdown().await;
        // 6G tasks (if initialized)
        if let Some(ref sixg) = self.sixg {
            let _ = sixg.she_client_tx.shutdown().await;
            let _ = sixg.nwdaf_reporter_tx.shutdown().await;
            let _ = sixg.isac_sensor_tx.shutdown().await;
            let _ = sixg.fl_participant_tx.shutdown().await;
            let _ = sixg.semantic_codec_tx.shutdown().await;
        }
    }
}


// ============================================================================
// Constants
// ============================================================================

/// Default channel capacity for task message queues.
pub const DEFAULT_CHANNEL_CAPACITY: usize = 256;

/// Default shutdown timeout in milliseconds.
pub const DEFAULT_SHUTDOWN_TIMEOUT_MS: u64 = 5000;

// ============================================================================
// Task Manager
// ============================================================================

/// Manages the lifecycle of all UE tasks.
///
/// The `TaskManager` is responsible for:
/// - Spawning tasks and tracking their handles
/// - Monitoring task health and state
/// - Coordinating graceful shutdown across all tasks
/// - Handling task failures and restarts
///
/// Based on UERANSIM's `UserEquipment` class from `src/ue/ue.cpp`.
pub struct TaskManager {
    /// Task base with all message channels
    task_base: UeTaskBase,
    /// Task state information
    task_states: HashMap<TaskId, TaskInfo>,
    /// Shutdown signal sender
    shutdown_tx: watch::Sender<bool>,
    /// Shutdown signal receiver (cloneable)
    shutdown_rx: watch::Receiver<bool>,
    /// Join handles for spawned tasks
    join_handles: HashMap<TaskId, JoinHandle<Result<(), TaskError>>>,
}

/// Error type for task operations.
#[derive(Debug, Clone)]
pub struct TaskError {
    /// Task that failed
    pub task_id: TaskId,
    /// Error message
    pub message: String,
}

impl std::fmt::Display for TaskError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Task {} error: {}", self.task_id, self.message)
    }
}

impl std::error::Error for TaskError {}

impl TaskManager {
    /// Creates a new `TaskManager` with the given configuration.
    ///
    /// Returns the manager along with receivers for each task.
    #[allow(clippy::type_complexity)]
    pub fn new(
        config: UeConfig,
        channel_capacity: usize,
    ) -> (
        Self,
        mpsc::Receiver<TaskMessage<AppMessage>>,
        mpsc::Receiver<TaskMessage<NasMessage>>,
        mpsc::Receiver<TaskMessage<RrcMessage>>,
        mpsc::Receiver<TaskMessage<RlsMessage>>,
    ) {
        let (task_base, app_rx, nas_rx, rrc_rx, rls_rx) =
            UeTaskBase::new(config, channel_capacity);

        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        // Initialize task states
        let mut task_states = HashMap::new();
        for task_id in [
            TaskId::App,
            TaskId::Nas,
            TaskId::Rrc,
            TaskId::Rls,
            // 6G AI-native network function tasks
            TaskId::SheClient,
            TaskId::NwdafReporter,
            TaskId::IsacSensor,
            TaskId::FlParticipant,
            TaskId::SemanticCodec,
        ] {
            task_states.insert(
                task_id,
                TaskInfo {
                    id: task_id,
                    state: TaskState::Created,
                    started_at: None,
                    stopped_at: None,
                    error: None,
                },
            );
        }

        let manager = Self {
            task_base,
            task_states,
            shutdown_tx,
            shutdown_rx,
            join_handles: HashMap::new(),
        };

        (manager, app_rx, nas_rx, rrc_rx, rls_rx)
    }

    /// Returns a clone of the task base for inter-task communication.
    pub fn task_base(&self) -> UeTaskBase {
        self.task_base.clone()
    }

    /// Returns a receiver for the shutdown signal.
    ///
    /// Tasks can use this to detect when shutdown has been requested.
    pub fn shutdown_receiver(&self) -> watch::Receiver<bool> {
        self.shutdown_rx.clone()
    }

    /// Gets the current state of a task.
    pub fn get_task_state(&self, task_id: TaskId) -> Option<TaskState> {
        self.task_states.get(&task_id).map(|info| info.state)
    }

    /// Gets information about a task.
    pub fn get_task_info(&self, task_id: TaskId) -> Option<&TaskInfo> {
        self.task_states.get(&task_id)
    }

    /// Returns true if all tasks are in the Running state.
    pub fn all_tasks_running(&self) -> bool {
        self.task_states
            .values()
            .all(|info| info.state == TaskState::Running)
    }

    /// Returns true if any task has failed.
    pub fn any_task_failed(&self) -> bool {
        self.task_states
            .values()
            .any(|info| info.state == TaskState::Failed)
    }

    /// Returns true if all tasks have stopped (either Stopped or Failed).
    pub fn all_tasks_stopped(&self) -> bool {
        self.task_states
            .values()
            .all(|info| info.state == TaskState::Stopped || info.state == TaskState::Failed)
    }

    /// Marks a task as started.
    pub fn mark_task_started(&mut self, task_id: TaskId) {
        if let Some(info) = self.task_states.get_mut(&task_id) {
            info.state = TaskState::Running;
            info.started_at = Some(Instant::now());
        }
    }

    /// Marks a task as stopping.
    pub fn mark_task_stopping(&mut self, task_id: TaskId) {
        if let Some(info) = self.task_states.get_mut(&task_id) {
            info.state = TaskState::Stopping;
        }
    }

    /// Marks a task as stopped.
    pub fn mark_task_stopped(&mut self, task_id: TaskId) {
        if let Some(info) = self.task_states.get_mut(&task_id) {
            info.state = TaskState::Stopped;
            info.stopped_at = Some(Instant::now());
        }
    }

    /// Marks a task as failed with an error message.
    pub fn mark_task_failed(&mut self, task_id: TaskId, error: String) {
        if let Some(info) = self.task_states.get_mut(&task_id) {
            info.state = TaskState::Failed;
            info.stopped_at = Some(Instant::now());
            info.error = Some(error);
        }
    }

    /// Registers a join handle for a spawned task.
    pub fn register_task_handle(&mut self, task_id: TaskId, handle: JoinHandle<Result<(), TaskError>>) {
        self.join_handles.insert(task_id, handle);
    }


    /// Initiates graceful shutdown of all tasks.
    ///
    /// This sends shutdown signals to all tasks and waits for them to complete.
    pub async fn shutdown(&mut self) -> Result<(), TaskError> {
        // Signal shutdown to all watchers
        let _ = self.shutdown_tx.send(true);

        // Mark all running tasks as stopping
        for info in self.task_states.values_mut() {
            if info.state == TaskState::Running {
                info.state = TaskState::Stopping;
            }
        }

        // Send shutdown messages to all tasks
        self.task_base.shutdown_all().await;

        // Wait for all tasks to complete with timeout
        let timeout = tokio::time::Duration::from_millis(DEFAULT_SHUTDOWN_TIMEOUT_MS);
        let deadline = tokio::time::Instant::now() + timeout;

        // Collect results first, then update states
        let handles: Vec<_> = self.join_handles.drain().collect();
        let mut results: Vec<(TaskId, Result<(), String>)> = Vec::new();

        for (task_id, handle) in handles {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            let result = match tokio::time::timeout(remaining, handle).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(e.message),
                Ok(Err(_join_error)) => Err("Task panicked".to_string()),
                Err(_timeout) => Err("Shutdown timeout".to_string()),
            };
            results.push((task_id, result));
        }

        // Now update states
        for (task_id, result) in results {
            match result {
                Ok(()) => self.mark_task_stopped(task_id),
                Err(msg) => self.mark_task_failed(task_id, msg),
            }
        }

        // Check if any tasks failed
        if self.any_task_failed() {
            let failed: Vec<_> = self
                .task_states
                .values()
                .filter(|info| info.state == TaskState::Failed)
                .map(|info| {
                    format!(
                        "{}: {}",
                        info.id,
                        info.error.as_deref().unwrap_or("unknown error")
                    )
                })
                .collect();
            return Err(TaskError {
                task_id: TaskId::App, // Use App as placeholder
                message: format!("Tasks failed during shutdown: {}", failed.join(", ")),
            });
        }

        Ok(())
    }

    /// Returns a summary of all task states.
    pub fn status_summary(&self) -> Vec<(TaskId, TaskState)> {
        self.task_states
            .iter()
            .map(|(id, info)| (*id, info.state))
            .collect()
    }
}


// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    

    /// Creates a test `UeConfig` for unit tests.
    fn test_config() -> UeConfig {
        UeConfig::default()
    }

    #[test]
    fn test_task_message_variants() {
        let msg: TaskMessage<i32> = TaskMessage::message(42);
        assert!(!msg.is_shutdown());
        assert_eq!(msg.unwrap(), 42);

        let shutdown: TaskMessage<i32> = TaskMessage::shutdown();
        assert!(shutdown.is_shutdown());
        assert!(shutdown.into_message().is_none());
    }

    #[test]
    fn test_task_message_into_message() {
        let msg: TaskMessage<String> = TaskMessage::message("hello".to_string());
        assert_eq!(msg.into_message(), Some("hello".to_string()));

        let shutdown: TaskMessage<String> = TaskMessage::shutdown();
        assert_eq!(shutdown.into_message(), None);
    }

    #[tokio::test]
    async fn test_task_handle_send() {
        let (tx, mut rx) = mpsc::channel::<TaskMessage<i32>>(10);
        let handle = TaskHandle::new(tx);

        handle.send(42).await.unwrap();

        match rx.recv().await {
            Some(TaskMessage::Message(val)) => assert_eq!(val, 42),
            _ => panic!("expected message"),
        }
    }

    #[tokio::test]
    async fn test_task_handle_shutdown() {
        let (tx, mut rx) = mpsc::channel::<TaskMessage<i32>>(10);
        let handle = TaskHandle::new(tx);

        handle.shutdown().await.unwrap();

        match rx.recv().await {
            Some(TaskMessage::Shutdown) => {}
            _ => panic!("expected shutdown"),
        }
    }

    #[tokio::test]
    async fn test_ue_task_base_creation() {
        let config = test_config();
        let (base, app_rx, nas_rx, rrc_rx, rls_rx) =
            UeTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);

        // Verify all handles are functional
        assert!(!base.app_tx.is_closed());
        assert!(!base.nas_tx.is_closed());
        assert!(!base.rrc_tx.is_closed());
        assert!(!base.rls_tx.is_closed());

        // Drop receivers to close channels
        drop(app_rx);
        drop(nas_rx);
        drop(rrc_rx);
        drop(rls_rx);

        // Verify handles detect closed channels
        assert!(base.app_tx.is_closed());
        assert!(base.nas_tx.is_closed());
        assert!(base.rrc_tx.is_closed());
        assert!(base.rls_tx.is_closed());
    }

    #[tokio::test]
    async fn test_inter_task_communication() {
        let config = test_config();
        let (base, _app_rx, mut nas_rx, mut rrc_rx, _rls_rx) =
            UeTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);

        // Simulate RRC -> NAS communication
        base.nas_tx
            .send(NasMessage::NasDelivery {
                pdu: OctetString::from_slice(&[0x7e, 0x00, 0x41]),
            })
            .await
            .unwrap();

        // Simulate NAS -> RRC communication
        base.rrc_tx
            .send(RrcMessage::UplinkNasDelivery {
                pdu_id: 1,
                pdu: OctetString::from_slice(&[0x7e, 0x00, 0x42]),
            })
            .await
            .unwrap();

        // Verify messages received
        match nas_rx.recv().await {
            Some(TaskMessage::Message(NasMessage::NasDelivery { pdu })) => {
                assert_eq!(pdu.len(), 3);
            }
            _ => panic!("expected NasDelivery"),
        }

        match rrc_rx.recv().await {
            Some(TaskMessage::Message(RrcMessage::UplinkNasDelivery { pdu_id, .. })) => {
                assert_eq!(pdu_id, 1);
            }
            _ => panic!("expected UplinkNasDelivery"),
        }
    }

    #[test]
    fn test_cm_state_display() {
        assert_eq!(format!("{}", CmState::Idle), "CM-IDLE");
        assert_eq!(format!("{}", CmState::Connected), "CM-CONNECTED");
    }

    #[test]
    fn test_rlf_cause() {
        assert_eq!(RlfCause::PduIdExists, RlfCause::PduIdExists);
        assert_ne!(RlfCause::PduIdExists, RlfCause::PduIdFull);
    }

    // ========================================================================
    // Task Lifecycle Management Tests
    // ========================================================================

    #[test]
    fn test_task_state_default() {
        let state = TaskState::default();
        assert_eq!(state, TaskState::Created);
    }

    #[test]
    fn test_task_state_display() {
        assert_eq!(format!("{}", TaskState::Created), "Created");
        assert_eq!(format!("{}", TaskState::Running), "Running");
        assert_eq!(format!("{}", TaskState::Stopping), "Stopping");
        assert_eq!(format!("{}", TaskState::Stopped), "Stopped");
        assert_eq!(format!("{}", TaskState::Failed), "Failed");
    }

    #[test]
    fn test_task_id_display() {
        assert_eq!(format!("{}", TaskId::App), "App");
        assert_eq!(format!("{}", TaskId::Nas), "NAS");
        assert_eq!(format!("{}", TaskId::Rrc), "RRC");
        assert_eq!(format!("{}", TaskId::Rls), "RLS");
        // 6G AI-native network function tasks
        assert_eq!(format!("{}", TaskId::SheClient), "SHE-Client");
        assert_eq!(format!("{}", TaskId::NwdafReporter), "NWDAF-Reporter");
        assert_eq!(format!("{}", TaskId::IsacSensor), "ISAC-Sensor");
        assert_eq!(format!("{}", TaskId::FlParticipant), "FL-Participant");
        assert_eq!(format!("{}", TaskId::SemanticCodec), "Semantic-Codec");
    }

    #[tokio::test]
    async fn test_task_manager_creation() {
        let config = test_config();
        let (manager, _app_rx, _nas_rx, _rrc_rx, _rls_rx) =
            TaskManager::new(config, DEFAULT_CHANNEL_CAPACITY);

        // All tasks should start in Created state
        assert_eq!(manager.get_task_state(TaskId::App), Some(TaskState::Created));
        assert_eq!(manager.get_task_state(TaskId::Nas), Some(TaskState::Created));
        assert_eq!(manager.get_task_state(TaskId::Rrc), Some(TaskState::Created));
        assert_eq!(manager.get_task_state(TaskId::Rls), Some(TaskState::Created));
        // 6G AI-native network function tasks
        assert_eq!(manager.get_task_state(TaskId::SheClient), Some(TaskState::Created));
        assert_eq!(manager.get_task_state(TaskId::NwdafReporter), Some(TaskState::Created));
        assert_eq!(manager.get_task_state(TaskId::IsacSensor), Some(TaskState::Created));
        assert_eq!(manager.get_task_state(TaskId::FlParticipant), Some(TaskState::Created));
        assert_eq!(manager.get_task_state(TaskId::SemanticCodec), Some(TaskState::Created));
    }

    #[tokio::test]
    async fn test_task_manager_state_transitions() {
        let config = test_config();
        let (mut manager, _app_rx, _nas_rx, _rrc_rx, _rls_rx) =
            TaskManager::new(config, DEFAULT_CHANNEL_CAPACITY);

        // Test state transitions
        manager.mark_task_started(TaskId::App);
        assert_eq!(manager.get_task_state(TaskId::App), Some(TaskState::Running));
        assert!(manager.get_task_info(TaskId::App).unwrap().started_at.is_some());

        manager.mark_task_stopping(TaskId::App);
        assert_eq!(manager.get_task_state(TaskId::App), Some(TaskState::Stopping));

        manager.mark_task_stopped(TaskId::App);
        assert_eq!(manager.get_task_state(TaskId::App), Some(TaskState::Stopped));
        assert!(manager.get_task_info(TaskId::App).unwrap().stopped_at.is_some());
    }

    #[tokio::test]
    async fn test_task_manager_failure_tracking() {
        let config = test_config();
        let (mut manager, _app_rx, _nas_rx, _rrc_rx, _rls_rx) =
            TaskManager::new(config, DEFAULT_CHANNEL_CAPACITY);

        manager.mark_task_started(TaskId::Nas);
        manager.mark_task_failed(TaskId::Nas, "Connection refused".to_string());

        assert_eq!(manager.get_task_state(TaskId::Nas), Some(TaskState::Failed));
        assert!(manager.any_task_failed());

        let info = manager.get_task_info(TaskId::Nas).unwrap();
        assert_eq!(info.error.as_deref(), Some("Connection refused"));
    }

    #[tokio::test]
    async fn test_task_manager_all_running_check() {
        let config = test_config();
        let (mut manager, _app_rx, _nas_rx, _rrc_rx, _rls_rx) =
            TaskManager::new(config, DEFAULT_CHANNEL_CAPACITY);

        assert!(!manager.all_tasks_running());

        // Start all tasks
        for task_id in [
            TaskId::App,
            TaskId::Nas,
            TaskId::Rrc,
            TaskId::Rls,
            // 6G AI-native network function tasks
            TaskId::SheClient,
            TaskId::NwdafReporter,
            TaskId::IsacSensor,
            TaskId::FlParticipant,
            TaskId::SemanticCodec,
        ] {
            manager.mark_task_started(task_id);
        }

        assert!(manager.all_tasks_running());
    }

    #[tokio::test]
    async fn test_task_manager_shutdown_receiver() {
        let config = test_config();
        let (manager, _app_rx, _nas_rx, _rrc_rx, _rls_rx) =
            TaskManager::new(config, DEFAULT_CHANNEL_CAPACITY);

        let shutdown_rx = manager.shutdown_receiver();
        assert!(!*shutdown_rx.borrow());
    }

    #[tokio::test]
    async fn test_task_manager_status_summary() {
        let config = test_config();
        let (mut manager, _app_rx, _nas_rx, _rrc_rx, _rls_rx) =
            TaskManager::new(config, DEFAULT_CHANNEL_CAPACITY);

        manager.mark_task_started(TaskId::App);
        manager.mark_task_started(TaskId::Nas);

        let summary = manager.status_summary();
        // 4 core tasks + 5 AI tasks = 9 total
        assert_eq!(summary.len(), 9);

        // Find App and Nas in summary
        let app_state = summary.iter().find(|(id, _)| *id == TaskId::App).map(|(_, s)| *s);
        let nas_state = summary.iter().find(|(id, _)| *id == TaskId::Nas).map(|(_, s)| *s);

        assert_eq!(app_state, Some(TaskState::Running));
        assert_eq!(nas_state, Some(TaskState::Running));
    }

    #[test]
    fn test_task_error_display() {
        let error = TaskError {
            task_id: TaskId::Rls,
            message: "Connection timeout".to_string(),
        };
        assert_eq!(format!("{error}"), "Task RLS error: Connection timeout");
    }
}
