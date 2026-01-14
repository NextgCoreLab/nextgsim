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
/// Based on UERANSIM's NtsTask lifecycle from `src/utils/nts.hpp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskState {
    /// Task is created but not yet started
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

impl Default for TaskState {
    fn default() -> Self {
        TaskState::Created
    }
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
}

impl std::fmt::Display for TaskId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskId::App => write!(f, "App"),
            TaskId::Nas => write!(f, "NAS"),
            TaskId::Rrc => write!(f, "RRC"),
            TaskId::Rls => write!(f, "RLS"),
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
        /// Session type (IPv4, IPv6, IPv4v6)
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
        /// Session type (e.g., "IPv4", "IPv6", "IPv4v6")
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
}

impl UeTaskBase {
    /// Creates a new UeTaskBase with the given configuration and channel capacity.
    ///
    /// Returns the task base along with receivers for each task.
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
        };

        (base, app_rx, nas_rx, rrc_rx, rls_rx)
    }

    /// Sends shutdown signals to all tasks.
    pub async fn shutdown_all(&self) {
        // Ignore errors - tasks may already be shut down
        let _ = self.app_tx.shutdown().await;
        let _ = self.nas_tx.shutdown().await;
        let _ = self.rrc_tx.shutdown().await;
        let _ = self.rls_tx.shutdown().await;
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
/// The TaskManager is responsible for:
/// - Spawning tasks and tracking their handles
/// - Monitoring task health and state
/// - Coordinating graceful shutdown across all tasks
/// - Handling task failures and restarts
///
/// Based on UERANSIM's UserEquipment class from `src/ue/ue.cpp`.
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
    /// Creates a new TaskManager with the given configuration.
    ///
    /// Returns the manager along with receivers for each task.
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
        for task_id in [TaskId::App, TaskId::Nas, TaskId::Rrc, TaskId::Rls] {
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
    use nextgsim_common::Plmn;

    /// Creates a test UeConfig for unit tests.
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
        for task_id in [TaskId::App, TaskId::Nas, TaskId::Rrc, TaskId::Rls] {
            manager.mark_task_started(task_id);
        }

        assert!(manager.all_tasks_running());
    }

    #[tokio::test]
    async fn test_task_manager_shutdown_receiver() {
        let config = test_config();
        let (manager, _app_rx, _nas_rx, _rrc_rx, _rls_rx) =
            TaskManager::new(config, DEFAULT_CHANNEL_CAPACITY);

        let mut shutdown_rx = manager.shutdown_receiver();
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
        assert_eq!(summary.len(), 4);

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
        assert_eq!(format!("{}", error), "Task RLS error: Connection timeout");
    }
}
