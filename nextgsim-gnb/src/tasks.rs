//! gNB Task Framework
//!
//! This module implements the actor-based task model with message passing for the gNB.
//! Each task runs as an independent async task and communicates via typed message channels.
//!
//! # Architecture
//!
//! The gNB uses the following tasks:
//! - **App Task**: Application management, CLI handling, status reporting
//! - **NGAP Task**: NGAP protocol handling, AMF communication
//! - **RRC Task**: RRC protocol handling, UE context management
//! - **GTP Task**: GTP-U tunnel management, user plane forwarding
//! - **RLS Task**: Radio Link Simulation, UE discovery
//! - **SCTP Task**: SCTP association management for NGAP transport
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
//! Based on UERANSIM's NTS (Network Task System) from `src/gnb/nts.hpp`

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use tokio::sync::watch;
use tokio::task::JoinHandle;

use nextgsim_common::config::GnbConfig;
use nextgsim_common::OctetString;
use nextgsim_rls::RrcChannel;

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

/// Task identifier for the gNB tasks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskId {
    /// Application task
    App,
    /// NGAP task
    Ngap,
    /// RRC task
    Rrc,
    /// GTP task
    Gtp,
    /// RLS task
    Rls,
    /// SCTP task
    Sctp,
    // 6G AI-native network function tasks
    /// Service Hosting Environment (SHE) task
    She,
    /// Network Data Analytics Function (NWDAF) task
    Nwdaf,
    /// Network Knowledge Exposure Function (NKEF) task
    Nkef,
    /// Integrated Sensing and Communication (ISAC) task
    Isac,
    /// AI Agent Framework (AAF) task
    Agent,
    /// Federated Learning Aggregator task
    FlAggregator,

    // ========================================================================
    // Rel-18 5G-Advanced tasks
    // ========================================================================
    /// Energy Savings - Cell sleep modes and energy efficiency metrics
    Energy,
}

impl std::fmt::Display for TaskId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskId::App => write!(f, "App"),
            TaskId::Ngap => write!(f, "NGAP"),
            TaskId::Rrc => write!(f, "RRC"),
            TaskId::Gtp => write!(f, "GTP"),
            TaskId::Rls => write!(f, "RLS"),
            TaskId::Sctp => write!(f, "SCTP"),
            TaskId::She => write!(f, "SHE"),
            TaskId::Nwdaf => write!(f, "NWDAF"),
            TaskId::Nkef => write!(f, "NKEF"),
            TaskId::Isac => write!(f, "ISAC"),
            TaskId::Agent => write!(f, "Agent"),
            TaskId::FlAggregator => write!(f, "FL"),
            TaskId::Energy => write!(f, "Energy"),
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

/// Base trait for all gNB tasks.
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
// Message Types
// ============================================================================

/// Messages for the Application task.
#[derive(Debug)]
pub enum AppMessage {
    /// Status update from another task
    StatusUpdate(StatusUpdate),
    /// CLI command received
    CliCommand(CliCommand),
}

/// Status update information.
#[derive(Debug, Clone)]
pub struct StatusUpdate {
    /// Status type identifier
    pub status_type: StatusType,
    /// Status value
    pub value: bool,
}

/// Types of status updates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusType {
    /// NGAP connection to AMF is up/down
    NgapIsUp,
}

/// CLI command for the gNB.
#[derive(Debug)]
pub struct CliCommand {
    /// Command type
    pub command: GnbCliCommandType,
    /// Response address (for sending reply)
    pub response_addr: Option<std::net::SocketAddr>,
}

/// Types of CLI commands.
#[derive(Debug, Clone)]
pub enum GnbCliCommandType {
    /// Show gNB info
    Info,
    /// Show gNB status
    Status,
    /// Show connected AMFs
    AmfList,
    /// Show connected UEs
    UeList,
    /// Show UE details
    UeInfo { ue_id: i32 },
    /// Release UE context
    UeRelease { ue_id: i32 },
}

// ============================================================================
// NGAP Messages
// ============================================================================

/// Messages for the NGAP task.
#[derive(Debug)]
pub enum NgapMessage {
    /// SCTP association established
    SctpAssociationUp {
        /// SCTP client ID
        client_id: i32,
        /// Association ID
        association_id: i32,
        /// Number of inbound streams
        in_streams: u16,
        /// Number of outbound streams
        out_streams: u16,
    },
    /// SCTP association closed
    SctpAssociationDown {
        /// SCTP client ID
        client_id: i32,
    },
    /// Received NGAP PDU from AMF
    ReceiveNgapPdu {
        /// SCTP client ID
        client_id: i32,
        /// SCTP stream ID
        stream: u16,
        /// NGAP PDU data
        pdu: OctetString,
    },
    /// Initial NAS delivery from RRC
    InitialNasDelivery {
        /// UE ID
        ue_id: i32,
        /// NAS PDU
        pdu: OctetString,
        /// RRC establishment cause
        rrc_establishment_cause: i64,
        /// S-TMSI if available
        s_tmsi: Option<GutiMobileIdentity>,
    },
    /// Uplink NAS delivery from RRC
    UplinkNasDelivery {
        /// UE ID
        ue_id: i32,
        /// NAS PDU
        pdu: OctetString,
    },
    /// Radio link failure notification from RRC
    RadioLinkFailure {
        /// UE ID
        ue_id: i32,
    },
    /// UE Context Release Request (from App for CLI-initiated release)
    UeContextReleaseRequest {
        /// UE ID
        ue_id: i32,
        /// Release cause
        cause: UeReleaseRequestCause,
    },
    /// NTN timing info received from AMF (NTN extension IE in NG Setup Response)
    NtnTimingInfoReceived {
        /// Satellite type
        satellite_type: String,
        /// Satellite ID
        satellite_id: u32,
        /// Propagation delay in microseconds
        propagation_delay_us: u64,
        /// Common timing advance in microseconds
        common_ta_us: u64,
        /// K-offset for HARQ timing
        k_offset: u16,
    },
    /// MBS Session Activation Request from AMF (Rel-17 MBS)
    MbsSessionActivationRequest {
        /// MBS session ID
        session_id: u32,
        /// TMGI (Temporary Mobile Group Identity)
        tmgi: [u8; 6],
        /// Is broadcast session (vs multicast)
        is_broadcast: bool,
        /// Multicast group IP address
        multicast_ip: Option<std::net::IpAddr>,
        /// MBS QFI
        qfi: u8,
    },
    /// MBS Session Deactivation Request from AMF (Rel-17 MBS)
    MbsSessionDeactivationRequest {
        /// MBS session ID
        session_id: u32,
    },
    /// Multicast Group Paging from AMF (Rel-17 MBS)
    MulticastGroupPaging {
        /// TMGI
        tmgi: [u8; 6],
        /// Area scope for paging
        area_scope: Vec<u32>,
    },
    /// MBS UE Join Request from RRC (Rel-17 MBS)
    MbsUeJoinRequest {
        /// UE ID
        ue_id: i32,
        /// TMGI to join
        tmgi: [u8; 6],
    },
    /// MBS UE Leave Request from RRC (Rel-17 MBS)
    MbsUeLeaveRequest {
        /// UE ID
        ue_id: i32,
        /// TMGI to leave
        tmgi: [u8; 6],
    },
}

/// Cause for UE context release request.
#[derive(Debug, Clone, Copy)]
pub enum UeReleaseRequestCause {
    /// User triggered (CLI command)
    UserTriggered,
    /// Radio link failure
    RadioLinkFailure,
    /// RAN originated release
    RanOriginated,
}

/// GUTI mobile identity for S-TMSI.
#[derive(Debug, Clone)]
pub struct GutiMobileIdentity {
    /// PLMN
    pub plmn: nextgsim_common::Plmn,
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
// RRC Messages
// ============================================================================

/// Messages for the RRC task.
#[derive(Debug)]
pub enum RrcMessage {
    /// Radio power on (from NGAP after NG Setup)
    RadioPowerOn,
    /// Signal detected from UE (from RLS)
    SignalDetected {
        /// UE ID
        ue_id: i32,
    },
    /// Uplink RRC message from UE (from RLS)
    UplinkRrc {
        /// UE ID
        ue_id: i32,
        /// RRC channel
        rrc_channel: RrcChannel,
        /// RRC PDU data
        data: OctetString,
    },
    /// Downlink NAS delivery (from NGAP)
    NasDelivery {
        /// UE ID
        ue_id: i32,
        /// NAS PDU
        pdu: OctetString,
    },
    /// AN release request (from NGAP)
    AnRelease {
        /// UE ID
        ue_id: i32,
    },
    /// Paging request (from NGAP)
    Paging {
        /// UE paging TMSI
        ue_paging_tmsi: Vec<u8>,
        /// TAI list for paging
        tai_list_for_paging: Vec<u8>,
    },
    /// NTN timing advance configuration (from NGAP, to include in RRC Setup/Reconfiguration)
    NtnTimingAdvanceConfig {
        /// Satellite type
        satellite_type: String,
        /// Common timing advance in microseconds
        common_ta_us: u64,
        /// K-offset for HARQ timing
        k_offset: u16,
        /// Max Doppler shift in Hz
        max_doppler_hz: f64,
        /// Whether UE should use autonomous TA calculation
        autonomous_ta: bool,
    },

    // ========================================================================
    // 6G Message Routing (Rel-20 extensions)
    // ========================================================================

    /// AI/ML model inference request from UE (routed to SHE)
    SixgAiMlInference {
        /// UE ID
        ue_id: i32,
        /// Model identifier
        model_id: String,
        /// Input data
        input_data: Vec<f32>,
    },
    /// ISAC sensing data from UE (routed to ISAC task)
    SixgIsacSensingData {
        /// UE ID
        ue_id: i32,
        /// Measurement type
        measurement_type: String,
        /// Measurements
        measurements: Vec<f32>,
    },
    /// Semantic communication message from UE (routed to NKEF)
    SixgSemanticMessage {
        /// UE ID
        ue_id: i32,
        /// Semantic content type
        content_type: String,
        /// Encoded semantic data
        data: Vec<u8>,
    },
}

// ============================================================================
// GTP Messages
// ============================================================================

/// Messages for the GTP task.
#[derive(Debug)]
pub enum GtpMessage {
    /// UE context update (from NGAP)
    UeContextUpdate {
        /// UE ID
        ue_id: i32,
        /// Update information
        update: GtpUeContextUpdate,
    },
    /// UE context release (from NGAP)
    UeContextRelease {
        /// UE ID
        ue_id: i32,
    },
    /// PDU session create (from NGAP)
    SessionCreate {
        /// UE ID
        ue_id: i32,
        /// PDU session resource
        resource: PduSessionResource,
    },
    /// PDU session modify (from NGAP)
    SessionModify {
        /// UE ID
        ue_id: i32,
        /// PDU session resource (updated tunnel info)
        resource: PduSessionResource,
    },
    /// PDU session release (from NGAP)
    SessionRelease {
        /// UE ID
        ue_id: i32,
        /// PDU session ID
        psi: i32,
    },
    /// Data PDU delivery from UE (from RLS)
    DataPduDelivery {
        /// UE ID
        ue_id: i32,
        /// PDU session ID
        psi: i32,
        /// User plane PDU
        pdu: OctetString,
    },
}

/// GTP UE context update information.
#[derive(Debug, Clone)]
pub struct GtpUeContextUpdate {
    /// UE ID
    pub ue_id: i32,
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: Option<i64>,
}

/// PDU session resource information.
#[derive(Debug, Clone)]
pub struct PduSessionResource {
    /// PDU session ID
    pub psi: i32,
    /// `QoS` flow identifier
    pub qfi: Option<u8>,
    /// Uplink TEID (gNB -> UPF)
    pub uplink_teid: u32,
    /// Downlink TEID (UPF -> gNB)
    pub downlink_teid: u32,
    /// UPF address
    pub upf_address: std::net::IpAddr,
}

// ============================================================================
// RLS Messages
// ============================================================================

/// Messages for the RLS task.
#[derive(Debug)]
pub enum RlsMessage {
    /// Signal detected from UE
    SignalDetected {
        /// UE ID
        ue_id: i32,
    },
    /// Signal lost from UE
    SignalLost {
        /// UE ID
        ue_id: i32,
    },
    /// Received RLS message from network
    ReceiveRlsMessage {
        /// Raw RLS message data
        data: OctetString,
        /// Source address
        source: std::net::SocketAddr,
    },
    /// Downlink RRC PDU (from RRC)
    DownlinkRrc {
        /// UE ID
        ue_id: i32,
        /// RRC channel
        rrc_channel: RrcChannel,
        /// PDU ID for acknowledgment tracking
        pdu_id: u32,
        /// RRC PDU data
        data: OctetString,
    },
    /// Downlink data PDU (from GTP)
    DownlinkData {
        /// UE ID
        ue_id: i32,
        /// PDU session ID
        psi: i32,
        /// User plane PDU
        pdu: OctetString,
    },
    /// Uplink RRC PDU (internal)
    UplinkRrc {
        /// UE ID
        ue_id: i32,
        /// RRC channel
        rrc_channel: RrcChannel,
        /// RRC PDU data
        data: OctetString,
    },
    /// Uplink data PDU (internal)
    UplinkData {
        /// UE ID
        ue_id: i32,
        /// PDU session ID
        psi: i32,
        /// User plane PDU
        pdu: OctetString,
    },
    /// Radio link failure
    RadioLinkFailure {
        /// UE ID
        ue_id: i32,
        /// Failure cause
        cause: RlfCause,
    },
    /// Transmission failure (PDUs not acknowledged)
    TransmissionFailure {
        /// List of failed PDU IDs
        pdu_list: Vec<u32>,
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
// SCTP Messages
// ============================================================================

/// Messages for the SCTP task.
#[derive(Debug)]
pub enum SctpMessage {
    /// Request to establish connection
    ConnectionRequest {
        /// Client ID for this connection
        client_id: i32,
        /// Local address to bind
        local_address: String,
        /// Local port
        local_port: u16,
        /// Remote address to connect
        remote_address: String,
        /// Remote port
        remote_port: u16,
        /// Payload protocol ID
        ppid: u32,
    },
    /// Request to close connection
    ConnectionClose {
        /// Client ID
        client_id: i32,
    },
    /// Association established (internal)
    AssociationSetup {
        /// Client ID
        client_id: i32,
        /// Association ID
        association_id: i32,
        /// Number of inbound streams
        in_streams: u16,
        /// Number of outbound streams
        out_streams: u16,
    },
    /// Association shutdown (internal)
    AssociationShutdown {
        /// Client ID
        client_id: i32,
    },
    /// Received message from peer
    ReceiveMessage {
        /// Client ID
        client_id: i32,
        /// Stream ID
        stream: u16,
        /// Message data
        buffer: OctetString,
    },
    /// Send message to peer
    SendMessage {
        /// Client ID
        client_id: i32,
        /// Stream ID
        stream: u16,
        /// Message data
        buffer: OctetString,
    },
    /// Unhandled SCTP notification
    UnhandledNotification {
        /// Client ID
        client_id: i32,
    },
}

// ============================================================================
// AI/ML Task Messages (6G AI-native network functions)
// ============================================================================

/// Messages for the Service Hosting Environment (SHE) task.
///
/// SHE provides three-tier distributed compute: Local Edge, Regional Edge, Core Cloud.
#[derive(Debug)]
pub enum SheMessage {
    /// Workload placement request
    PlaceWorkload {
        /// Workload identifier
        workload_id: u64,
        /// Required latency constraint (ms)
        max_latency_ms: u32,
        /// Required compute (FLOPS)
        compute_flops: u64,
        /// Required memory (bytes)
        memory_bytes: u64,
    },
    /// Inference request
    InferenceRequest {
        /// Model identifier
        model_id: String,
        /// Request identifier for response correlation
        request_id: u64,
        /// Input data
        input_data: Vec<f32>,
    },
    /// Resource status update
    ResourceUpdate {
        /// Node identifier
        node_id: u32,
        /// Available compute (FLOPS)
        available_flops: u64,
        /// Available memory (bytes)
        available_memory: u64,
    },
}

/// Messages for the Network Data Analytics Function (NWDAF) task.
///
/// NWDAF provides four-layer analytics with closed-loop automation per 3GPP TS 23.288.
#[derive(Debug)]
pub enum NwdafMessage {
    /// UE measurement report for analytics
    UeMeasurement {
        /// UE identifier
        ue_id: i32,
        /// Serving cell RSRP (dBm)
        rsrp: f32,
        /// Serving cell RSRQ (dB)
        rsrq: f32,
        /// Position (x, y, z in meters)
        position: (f32, f32, f32),
    },
    /// Cell load update
    CellLoad {
        /// Cell identifier
        cell_id: i32,
        /// PRB usage ratio (0.0 - 1.0)
        prb_usage: f32,
        /// Number of connected UEs
        connected_ues: u32,
    },
    /// Trajectory prediction request
    PredictTrajectory {
        /// UE identifier
        ue_id: i32,
        /// Prediction horizon (ms)
        horizon_ms: u32,
    },
    /// Handover optimization recommendation
    HandoverRecommendation {
        /// UE identifier
        ue_id: i32,
        /// Recommended target cell
        target_cell: i32,
        /// Confidence score (0.0 - 1.0)
        confidence: f32,
    },
}

/// Messages for the Network Knowledge Exposure Function (NKEF) task.
///
/// NKEF provides knowledge graphs and semantic search for LLM integration.
#[derive(Debug)]
pub enum NkefMessage {
    /// Knowledge update
    UpdateKnowledge {
        /// Entity type
        entity_type: String,
        /// Entity identifier
        entity_id: String,
        /// Properties as key-value pairs
        properties: Vec<(String, String)>,
    },
    /// Semantic query
    SemanticQuery {
        /// Query string
        query: String,
        /// Maximum results
        max_results: u32,
    },
    /// RAG context retrieval
    RetrieveContext {
        /// Prompt for context retrieval
        prompt: String,
        /// Maximum context tokens
        max_tokens: u32,
    },
}

/// Messages for the Integrated Sensing and Communication (ISAC) task.
///
/// ISAC provides sensing-communication convergence per 3GPP TR 22.837.
#[derive(Debug)]
pub enum IsacMessage {
    /// Sensing data from cell
    SensingData {
        /// Cell identifier
        cell_id: i32,
        /// Measurement type (`ToA`, `TDoA`, `AoA`, RSS, Doppler)
        measurement_type: String,
        /// Measurement values
        measurements: Vec<f32>,
    },
    /// Position fusion request
    FusionRequest {
        /// UE identifier
        ue_id: i32,
        /// Data source identifiers
        source_ids: Vec<u32>,
    },
    /// Tracking update
    TrackingUpdate {
        /// Object identifier
        object_id: u64,
        /// Position estimate (x, y, z)
        position: (f32, f32, f32),
        /// Velocity estimate (vx, vy, vz)
        velocity: (f32, f32, f32),
    },
}

/// Messages for the AI Agent Framework (AAF) task.
///
/// AAF provides multi-agent coordination with OAuth 2.0 authentication.
#[derive(Debug)]
pub enum AgentMessage {
    /// Agent registration
    RegisterAgent {
        /// Agent identifier
        agent_id: String,
        /// Agent type
        agent_type: String,
        /// Capabilities
        capabilities: Vec<String>,
    },
    /// Intent submission
    SubmitIntent {
        /// Agent identifier
        agent_id: String,
        /// Intent type
        intent_type: String,
        /// Intent parameters
        parameters: Vec<(String, String)>,
    },
    /// Agent coordination event
    CoordinationEvent {
        /// Event type
        event_type: String,
        /// Participating agents
        agent_ids: Vec<String>,
    },
}

/// Messages for the Federated Learning Aggregator task.
///
/// FL provides privacy-preserving distributed training per 3GPP TR 23.700-80.
#[derive(Debug)]
pub enum FlAggregatorMessage {
    /// Participant registration
    RegisterParticipant {
        /// Participant identifier
        participant_id: String,
        /// Device capabilities (FLOPS)
        compute_capability: u64,
    },
    /// Model update from participant
    SubmitUpdate {
        /// Participant identifier
        participant_id: String,
        /// Round number
        round: u64,
        /// Gradient updates
        gradients: Vec<f32>,
        /// Number of local samples
        num_samples: u64,
    },
    /// Trigger aggregation
    Aggregate {
        /// Round number
        round: u64,
    },
    /// Distribute global model
    DistributeModel {
        /// Model version
        version: u64,
        /// Model weights
        weights: Vec<f32>,
    },
}

// ============================================================================
// Rel-18 Energy Savings Messages
// ============================================================================

/// Messages for the Energy Savings task.
///
/// Handles cell sleep/dormant modes and energy efficiency metrics.
#[derive(Debug)]
pub enum EnergyMessage {
    /// Put cell into sleep mode
    CellSleep {
        /// Cell identifier
        cell_id: i32,
        /// Sleep mode: "light", "deep", or "dormant"
        sleep_mode: String,
        /// Timestamp in milliseconds
        timestamp_ms: u64,
    },
    /// Wake up cell from sleep
    CellWakeUp {
        /// Cell identifier
        cell_id: i32,
        /// Wake-up reason: "paging", "traffic", "timer", "manual"
        reason: String,
        /// Timestamp in milliseconds
        timestamp_ms: u64,
    },
    /// Report traffic volume for energy efficiency calculation
    TrafficReport {
        /// Cell identifier
        cell_id: i32,
        /// Data volume in bits
        data_bits: u64,
        /// Number of connected UEs
        connected_ues: u32,
    },
    /// Get energy efficiency metrics for all cells
    GetMetrics {
        /// Response channel
        response_tx: Option<tokio::sync::oneshot::Sender<Vec<crate::energy::task::CellEnergyReport>>>,
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
// gNB Task Base
// ============================================================================

/// Base structure containing all task handles for the gNB.
///
/// This structure is shared among all tasks to enable inter-task communication.
/// Each task receives a clone of this structure and can send messages to any
/// other task through the appropriate handle.
#[derive(Clone)]
pub struct GnbTaskBase {
    /// gNB configuration
    pub config: Arc<GnbConfig>,
    /// Handle to the Application task
    pub app_tx: TaskHandle<AppMessage>,
    /// Handle to the NGAP task
    pub ngap_tx: TaskHandle<NgapMessage>,
    /// Handle to the RRC task
    pub rrc_tx: TaskHandle<RrcMessage>,
    /// Handle to the GTP task
    pub gtp_tx: TaskHandle<GtpMessage>,
    /// Handle to the RLS task
    pub rls_tx: TaskHandle<RlsMessage>,
    /// Handle to the SCTP task
    pub sctp_tx: TaskHandle<SctpMessage>,
    /// 6G task handles (initialized via `init_6g_tasks()`)
    pub sixg: Option<GnbSixgHandles>,
    /// Rel-18 task handles (initialized via `init_rel18_tasks()`)
    pub rel18: Option<GnbRel18Handles>,
}

/// Rel-18 5G-Advanced task handles for gNB
#[derive(Clone)]
pub struct GnbRel18Handles {
    /// Handle to the Energy Savings task
    pub energy_tx: TaskHandle<EnergyMessage>,
}

/// Rel-18 task receivers for gNB
pub struct GnbRel18Receivers {
    pub energy_rx: mpsc::Receiver<TaskMessage<EnergyMessage>>,
}

/// 6G task handles for gNB (Rel-20 extensions)
#[derive(Clone)]
pub struct GnbSixgHandles {
    /// Handle to the SHE (Sub-network Hosted Entity) task
    pub she_tx: TaskHandle<SheMessage>,
    /// Handle to the NWDAF (Network Data Analytics) task
    pub nwdaf_tx: TaskHandle<NwdafMessage>,
    /// Handle to the NKEF (Network Knowledge Exchange) task
    pub nkef_tx: TaskHandle<NkefMessage>,
    /// Handle to the ISAC (Integrated Sensing and Communication) task
    pub isac_tx: TaskHandle<IsacMessage>,
    /// Handle to the Agent task
    pub agent_tx: TaskHandle<AgentMessage>,
    /// Handle to the FL (Federated Learning) Aggregator task
    pub fl_tx: TaskHandle<FlAggregatorMessage>,
}

/// 6G task receivers for gNB
pub struct GnbSixgReceivers {
    pub she_rx: mpsc::Receiver<TaskMessage<SheMessage>>,
    pub nwdaf_rx: mpsc::Receiver<TaskMessage<NwdafMessage>>,
    pub nkef_rx: mpsc::Receiver<TaskMessage<NkefMessage>>,
    pub isac_rx: mpsc::Receiver<TaskMessage<IsacMessage>>,
    pub agent_rx: mpsc::Receiver<TaskMessage<AgentMessage>>,
    pub fl_rx: mpsc::Receiver<TaskMessage<FlAggregatorMessage>>,
}

impl GnbTaskBase {
    /// Creates a new `GnbTaskBase` with the given configuration and channel capacity.
    ///
    /// Returns the task base along with receivers for each task.
    #[allow(clippy::type_complexity)]
    pub fn new(
        config: GnbConfig,
        channel_capacity: usize,
    ) -> (
        Self,
        mpsc::Receiver<TaskMessage<AppMessage>>,
        mpsc::Receiver<TaskMessage<NgapMessage>>,
        mpsc::Receiver<TaskMessage<RrcMessage>>,
        mpsc::Receiver<TaskMessage<GtpMessage>>,
        mpsc::Receiver<TaskMessage<RlsMessage>>,
        mpsc::Receiver<TaskMessage<SctpMessage>>,
    ) {
        let (app_tx, app_rx) = mpsc::channel(channel_capacity);
        let (ngap_tx, ngap_rx) = mpsc::channel(channel_capacity);
        let (rrc_tx, rrc_rx) = mpsc::channel(channel_capacity);
        let (gtp_tx, gtp_rx) = mpsc::channel(channel_capacity);
        let (rls_tx, rls_rx) = mpsc::channel(channel_capacity);
        let (sctp_tx, sctp_rx) = mpsc::channel(channel_capacity);

        let base = Self {
            config: Arc::new(config),
            app_tx: TaskHandle::new(app_tx),
            ngap_tx: TaskHandle::new(ngap_tx),
            rrc_tx: TaskHandle::new(rrc_tx),
            gtp_tx: TaskHandle::new(gtp_tx),
            rls_tx: TaskHandle::new(rls_tx),
            sctp_tx: TaskHandle::new(sctp_tx),
            sixg: None,
            rel18: None,
        };

        (base, app_rx, ngap_rx, rrc_rx, gtp_rx, rls_rx, sctp_rx)
    }

    /// Initialize 6G task handles and return their receivers.
    ///
    /// Call this after `new()` to enable 6G tasks (SHE, NWDAF, NKEF, ISAC, Agent, FL).
    /// The returned receivers should be used to spawn the 6G task loops.
    pub fn init_6g_tasks(&mut self, channel_capacity: usize) -> GnbSixgReceivers {
        let (she_tx, she_rx) = mpsc::channel(channel_capacity);
        let (nwdaf_tx, nwdaf_rx) = mpsc::channel(channel_capacity);
        let (nkef_tx, nkef_rx) = mpsc::channel(channel_capacity);
        let (isac_tx, isac_rx) = mpsc::channel(channel_capacity);
        let (agent_tx, agent_rx) = mpsc::channel(channel_capacity);
        let (fl_tx, fl_rx) = mpsc::channel(channel_capacity);

        self.sixg = Some(GnbSixgHandles {
            she_tx: TaskHandle::new(she_tx),
            nwdaf_tx: TaskHandle::new(nwdaf_tx),
            nkef_tx: TaskHandle::new(nkef_tx),
            isac_tx: TaskHandle::new(isac_tx),
            agent_tx: TaskHandle::new(agent_tx),
            fl_tx: TaskHandle::new(fl_tx),
        });

        GnbSixgReceivers {
            she_rx,
            nwdaf_rx,
            nkef_rx,
            isac_rx,
            agent_rx,
            fl_rx,
        }
    }

    /// Initialize Rel-18 5G-Advanced task handles and return their receivers.
    pub fn init_rel18_tasks(&mut self, channel_capacity: usize) -> GnbRel18Receivers {
        let (energy_tx, energy_rx) = mpsc::channel(channel_capacity);

        self.rel18 = Some(GnbRel18Handles {
            energy_tx: TaskHandle::new(energy_tx),
        });

        GnbRel18Receivers { energy_rx }
    }

    /// Sends shutdown signals to all tasks.
    pub async fn shutdown_all(&self) {
        // Ignore errors - tasks may already be shut down
        let _ = self.app_tx.shutdown().await;
        let _ = self.ngap_tx.shutdown().await;
        let _ = self.rrc_tx.shutdown().await;
        let _ = self.gtp_tx.shutdown().await;
        let _ = self.rls_tx.shutdown().await;
        let _ = self.sctp_tx.shutdown().await;
        // 6G tasks (if initialized)
        if let Some(ref sixg) = self.sixg {
            let _ = sixg.she_tx.shutdown().await;
            let _ = sixg.nwdaf_tx.shutdown().await;
            let _ = sixg.nkef_tx.shutdown().await;
            let _ = sixg.isac_tx.shutdown().await;
            let _ = sixg.agent_tx.shutdown().await;
            let _ = sixg.fl_tx.shutdown().await;
        }
        // Rel-18 tasks (if initialized)
        if let Some(ref rel18) = self.rel18 {
            let _ = rel18.energy_tx.shutdown().await;
        }
    }
}

// ============================================================================
// Constants
// ============================================================================

/// Default channel capacity for task message queues.
pub const DEFAULT_CHANNEL_CAPACITY: usize = 256;

/// NGAP Payload Protocol ID for SCTP.
pub const NGAP_PPID: u32 = 60;

/// Default shutdown timeout in milliseconds.
pub const DEFAULT_SHUTDOWN_TIMEOUT_MS: u64 = 5000;

// ============================================================================
// Task Manager
// ============================================================================

/// Manages the lifecycle of all gNB tasks.
///
/// The `TaskManager` is responsible for:
/// - Spawning tasks and tracking their handles
/// - Monitoring task health and state
/// - Coordinating graceful shutdown across all tasks
/// - Handling task failures and restarts
///
/// Based on UERANSIM's `GNodeB` class from `src/gnb/gnb.cpp`.
pub struct TaskManager {
    /// Task base with all message channels
    task_base: GnbTaskBase,
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
        config: GnbConfig,
        channel_capacity: usize,
    ) -> (
        Self,
        mpsc::Receiver<TaskMessage<AppMessage>>,
        mpsc::Receiver<TaskMessage<NgapMessage>>,
        mpsc::Receiver<TaskMessage<RrcMessage>>,
        mpsc::Receiver<TaskMessage<GtpMessage>>,
        mpsc::Receiver<TaskMessage<RlsMessage>>,
        mpsc::Receiver<TaskMessage<SctpMessage>>,
    ) {
        let (task_base, app_rx, ngap_rx, rrc_rx, gtp_rx, rls_rx, sctp_rx) =
            GnbTaskBase::new(config, channel_capacity);

        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        // Initialize task states
        let mut task_states = HashMap::new();
        for task_id in [
            TaskId::App,
            TaskId::Ngap,
            TaskId::Rrc,
            TaskId::Gtp,
            TaskId::Rls,
            TaskId::Sctp,
            // 6G AI-native network function tasks
            TaskId::She,
            TaskId::Nwdaf,
            TaskId::Nkef,
            TaskId::Isac,
            TaskId::Agent,
            TaskId::FlAggregator,
            // Rel-18 5G-Advanced tasks
            TaskId::Energy,
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

        (manager, app_rx, ngap_rx, rrc_rx, gtp_rx, rls_rx, sctp_rx)
    }

    /// Returns a clone of the task base for inter-task communication.
    pub fn task_base(&self) -> GnbTaskBase {
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
    use nextgsim_common::Plmn;

    /// Creates a test `GnbConfig` for unit tests.
    fn test_config() -> GnbConfig {
        GnbConfig {
            nci: 0x000000010,
            gnb_id_length: 32,
            plmn: Plmn::new(001, 01, false),
            tac: 1,
            nssai: vec![],
            amf_configs: vec![],
            link_ip: "127.0.0.1".parse().unwrap(),
            ngap_ip: "127.0.0.1".parse().unwrap(),
            gtp_ip: "127.0.0.1".parse().unwrap(),
            gtp_advertise_ip: None,
            ignore_stream_ids: false, upf_addr: None, upf_port: 2152,
            pqc_config: nextgsim_common::config::PqcConfig::default(),
            ntn_config: None,
            mbs_enabled: false,
            prose_enabled: false,
            lcs_enabled: false,
            snpn_config: None,
            ..Default::default()
        }
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
    async fn test_gnb_task_base_creation() {
        let config = test_config();
        let (base, app_rx, ngap_rx, rrc_rx, gtp_rx, rls_rx, sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);

        // Verify all handles are functional
        assert!(!base.app_tx.is_closed());
        assert!(!base.ngap_tx.is_closed());
        assert!(!base.rrc_tx.is_closed());
        assert!(!base.gtp_tx.is_closed());
        assert!(!base.rls_tx.is_closed());
        assert!(!base.sctp_tx.is_closed());

        // Drop receivers to close channels
        drop(app_rx);
        drop(ngap_rx);
        drop(rrc_rx);
        drop(gtp_rx);
        drop(rls_rx);
        drop(sctp_rx);

        // Verify handles detect closed channels
        assert!(base.app_tx.is_closed());
        assert!(base.ngap_tx.is_closed());
        assert!(base.rrc_tx.is_closed());
        assert!(base.gtp_tx.is_closed());
        assert!(base.rls_tx.is_closed());
        assert!(base.sctp_tx.is_closed());
    }

    #[tokio::test]
    async fn test_inter_task_communication() {
        let config = test_config();
        let (base, _app_rx, mut ngap_rx, mut rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);

        // Simulate RRC -> NGAP communication
        base.ngap_tx
            .send(NgapMessage::UplinkNasDelivery {
                ue_id: 1,
                pdu: OctetString::from_slice(&[0x7e, 0x00, 0x41]),
            })
            .await
            .unwrap();

        // Simulate NGAP -> RRC communication
        base.rrc_tx
            .send(RrcMessage::NasDelivery {
                ue_id: 1,
                pdu: OctetString::from_slice(&[0x7e, 0x00, 0x42]),
            })
            .await
            .unwrap();

        // Verify messages received
        match ngap_rx.recv().await {
            Some(TaskMessage::Message(NgapMessage::UplinkNasDelivery { ue_id, .. })) => {
                assert_eq!(ue_id, 1);
            }
            _ => panic!("expected UplinkNasDelivery"),
        }

        match rrc_rx.recv().await {
            Some(TaskMessage::Message(RrcMessage::NasDelivery { ue_id, .. })) => {
                assert_eq!(ue_id, 1);
            }
            _ => panic!("expected NasDelivery"),
        }
    }

    #[test]
    fn test_status_update() {
        let update = StatusUpdate {
            status_type: StatusType::NgapIsUp,
            value: true,
        };
        assert_eq!(update.status_type, StatusType::NgapIsUp);
        assert!(update.value);
    }

    #[test]
    fn test_cli_command_types() {
        let cmd = GnbCliCommandType::Info;
        assert!(matches!(cmd, GnbCliCommandType::Info));

        let cmd = GnbCliCommandType::UeInfo { ue_id: 42 };
        if let GnbCliCommandType::UeInfo { ue_id } = cmd {
            assert_eq!(ue_id, 42);
        } else {
            panic!("expected UeInfo");
        }
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
        assert_eq!(format!("{}", TaskId::Ngap), "NGAP");
        assert_eq!(format!("{}", TaskId::Rrc), "RRC");
        assert_eq!(format!("{}", TaskId::Gtp), "GTP");
        assert_eq!(format!("{}", TaskId::Rls), "RLS");
        assert_eq!(format!("{}", TaskId::Sctp), "SCTP");
    }

    #[tokio::test]
    async fn test_task_manager_creation() {
        let config = test_config();
        let (manager, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            TaskManager::new(config, DEFAULT_CHANNEL_CAPACITY);

        // All tasks should start in Created state
        assert_eq!(manager.get_task_state(TaskId::App), Some(TaskState::Created));
        assert_eq!(manager.get_task_state(TaskId::Ngap), Some(TaskState::Created));
        assert_eq!(manager.get_task_state(TaskId::Rrc), Some(TaskState::Created));
        assert_eq!(manager.get_task_state(TaskId::Gtp), Some(TaskState::Created));
        assert_eq!(manager.get_task_state(TaskId::Rls), Some(TaskState::Created));
        assert_eq!(manager.get_task_state(TaskId::Sctp), Some(TaskState::Created));
    }

    #[tokio::test]
    async fn test_task_manager_state_transitions() {
        let config = test_config();
        let (mut manager, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
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
        let (mut manager, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            TaskManager::new(config, DEFAULT_CHANNEL_CAPACITY);

        manager.mark_task_started(TaskId::Ngap);
        manager.mark_task_failed(TaskId::Ngap, "Connection refused".to_string());

        assert_eq!(manager.get_task_state(TaskId::Ngap), Some(TaskState::Failed));
        assert!(manager.any_task_failed());

        let info = manager.get_task_info(TaskId::Ngap).unwrap();
        assert_eq!(info.error.as_deref(), Some("Connection refused"));
    }

    #[tokio::test]
    async fn test_task_manager_all_running_check() {
        let config = test_config();
        let (mut manager, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            TaskManager::new(config, DEFAULT_CHANNEL_CAPACITY);

        assert!(!manager.all_tasks_running());

        // Start all tasks (including 6G + Rel-18 tasks)
        for task_id in [
            TaskId::App, TaskId::Ngap, TaskId::Rrc, TaskId::Gtp, TaskId::Rls, TaskId::Sctp,
            TaskId::She, TaskId::Nwdaf, TaskId::Nkef, TaskId::Isac, TaskId::Agent, TaskId::FlAggregator,
            TaskId::Energy,
        ] {
            manager.mark_task_started(task_id);
        }

        assert!(manager.all_tasks_running());
    }

    #[tokio::test]
    async fn test_task_manager_shutdown_receiver() {
        let config = test_config();
        let (manager, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            TaskManager::new(config, DEFAULT_CHANNEL_CAPACITY);

        let shutdown_rx = manager.shutdown_receiver();
        assert!(!*shutdown_rx.borrow());
    }

    #[tokio::test]
    async fn test_task_manager_status_summary() {
        let config = test_config();
        let (mut manager, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            TaskManager::new(config, DEFAULT_CHANNEL_CAPACITY);

        manager.mark_task_started(TaskId::App);
        manager.mark_task_started(TaskId::Ngap);

        let summary = manager.status_summary();
        assert_eq!(summary.len(), 13);

        // Find App and Ngap in summary
        let app_state = summary.iter().find(|(id, _)| *id == TaskId::App).map(|(_, s)| *s);
        let ngap_state = summary.iter().find(|(id, _)| *id == TaskId::Ngap).map(|(_, s)| *s);

        assert_eq!(app_state, Some(TaskState::Running));
        assert_eq!(ngap_state, Some(TaskState::Running));
    }

    #[test]
    fn test_task_error_display() {
        let error = TaskError {
            task_id: TaskId::Sctp,
            message: "Connection timeout".to_string(),
        };
        assert_eq!(format!("{error}"), "Task SCTP error: Connection timeout");
    }
}
