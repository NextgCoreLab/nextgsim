//! Multi-agent coordination protocol
//!
//! Provides agent-to-agent messaging, role hierarchy, cooperative intents, and
//! message routing through the coordinator.

use crate::{AgentId, AgentType, Intent};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Agent role hierarchy
// ---------------------------------------------------------------------------

/// Hierarchical role that determines an agent's authority scope.
///
/// Higher-level agents can override or coordinate lower-level ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AgentRole {
    /// Manages a single cell.
    CellLevel = 0,
    /// Manages a region of cells.
    RegionLevel = 1,
    /// Manages the entire network.
    NetworkLevel = 2,
}

impl AgentRole {
    /// Returns true if `self` outranks `other`.
    pub fn outranks(self, other: AgentRole) -> bool {
        (self as u8) > (other as u8)
    }

    /// Returns true if `self` can coordinate agents at the given level.
    pub fn can_coordinate(self, subordinate: AgentRole) -> bool {
        self.outranks(subordinate) || self == subordinate
    }
}

impl Default for AgentRole {
    fn default() -> Self {
        Self::CellLevel
    }
}

// ---------------------------------------------------------------------------
// Agent-to-agent messages
// ---------------------------------------------------------------------------

/// A message exchanged between agents through the coordinator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationMessage {
    /// Unique message ID.
    pub id: String,
    /// Sending agent.
    pub from: AgentId,
    /// Receiving agent.
    pub to: AgentId,
    /// Message payload.
    pub payload: MessagePayload,
    /// Timestamp in milliseconds since epoch.
    pub timestamp_ms: u64,
    /// Optional correlation ID to link request/response pairs.
    pub correlation_id: Option<String>,
}

/// Payload variants for agent-to-agent communication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessagePayload {
    /// Request: one agent asks another to perform an action or provide data.
    Request {
        /// Description of the requested action.
        action: String,
        /// Key-value parameters.
        parameters: HashMap<String, String>,
    },
    /// Response: the result of a previous request.
    Response {
        /// Whether the request succeeded.
        success: bool,
        /// Key-value result data.
        data: HashMap<String, String>,
        /// Error detail if not successful.
        error: Option<String>,
    },
    /// Notify: a one-way informational message (no response expected).
    Notify {
        /// Event type / topic.
        event: String,
        /// Key-value event data.
        data: HashMap<String, String>,
    },
    /// Sub-intent contribution for a composite goal.
    SubIntentOffer {
        /// The composite intent ID this contributes to.
        composite_id: String,
        /// The sub-intent proposed by the sender.
        sub_intent: Intent,
    },
    /// Acceptance or rejection of a sub-intent offer.
    SubIntentDecision {
        /// The composite intent ID.
        composite_id: String,
        /// Whether the sub-intent was accepted.
        accepted: bool,
        /// Reason if rejected.
        reason: Option<String>,
    },
}

// ---------------------------------------------------------------------------
// Composite intent
// ---------------------------------------------------------------------------

/// A composite goal that multiple agents contribute sub-intents to.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositeIntent {
    /// Unique composite intent ID.
    pub id: String,
    /// The coordinating agent that owns this composite goal.
    pub coordinator_agent: AgentId,
    /// Human-readable description of the composite goal.
    pub description: String,
    /// Sub-intents contributed by various agents.
    pub sub_intents: Vec<SubIntentEntry>,
    /// Current state.
    pub state: CompositeState,
    /// Creation timestamp.
    pub created_at_ms: u64,
}

/// A sub-intent entry within a composite.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubIntentEntry {
    /// The contributing agent.
    pub agent_id: AgentId,
    /// The sub-intent.
    pub intent: Intent,
    /// Whether the coordinator has accepted it.
    pub accepted: bool,
}

/// State of a composite intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompositeState {
    /// Gathering sub-intents from agents.
    Gathering,
    /// All sub-intents collected; ready for execution.
    Ready,
    /// Currently executing.
    Executing,
    /// Completed.
    Completed,
    /// Failed or cancelled.
    Failed,
}

// ---------------------------------------------------------------------------
// Agent profile (extended registration info for coordination)
// ---------------------------------------------------------------------------

/// Extended agent profile used by the coordination system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentProfile {
    /// Agent identifier.
    pub agent_id: AgentId,
    /// Agent type.
    pub agent_type: AgentType,
    /// Hierarchical role.
    pub role: AgentRole,
    /// Region/scope this agent is responsible for (e.g., "region-east", "cell-5").
    pub scope: Option<String>,
}

// ---------------------------------------------------------------------------
// Message router
// ---------------------------------------------------------------------------

/// Routes messages between agents and manages composite intents.
#[derive(Debug)]
pub struct MessageRouter {
    /// Per-agent incoming message mailboxes.
    mailboxes: HashMap<AgentId, VecDeque<CoordinationMessage>>,
    /// Agent profiles (for role checks).
    profiles: HashMap<AgentId, AgentProfile>,
    /// Active composite intents.
    composites: HashMap<String, CompositeIntent>,
    /// Next message ID counter.
    next_msg_id: u64,
}

impl MessageRouter {
    /// Creates a new message router.
    pub fn new() -> Self {
        Self {
            mailboxes: HashMap::new(),
            profiles: HashMap::new(),
            composites: HashMap::new(),
            next_msg_id: 1,
        }
    }

    /// Register an agent profile so it can send/receive messages.
    pub fn register_agent(&mut self, profile: AgentProfile) {
        self.mailboxes
            .entry(profile.agent_id.clone())
            .or_default();
        self.profiles.insert(profile.agent_id.clone(), profile);
    }

    /// Unregister an agent.
    pub fn unregister_agent(&mut self, agent_id: &AgentId) {
        self.mailboxes.remove(agent_id);
        self.profiles.remove(agent_id);
    }

    /// Get an agent's profile.
    pub fn get_profile(&self, agent_id: &AgentId) -> Option<&AgentProfile> {
        self.profiles.get(agent_id)
    }

    /// Send a message from one agent to another.
    ///
    /// The message is placed in the recipient's mailbox.
    /// Returns `Err` if the recipient is not registered.
    pub fn send(
        &mut self,
        from: &AgentId,
        to: &AgentId,
        payload: MessagePayload,
        correlation_id: Option<String>,
    ) -> Result<String, String> {
        if !self.profiles.contains_key(from) {
            return Err(format!("Sender {from} is not registered"));
        }
        if !self.mailboxes.contains_key(to) {
            return Err(format!("Recipient {to} is not registered"));
        }

        let msg_id = format!("msg-{}", self.next_msg_id);
        self.next_msg_id += 1;

        let message = CoordinationMessage {
            id: msg_id.clone(),
            from: from.clone(),
            to: to.clone(),
            payload,
            timestamp_ms: crate::timestamp_now(),
            correlation_id,
        };

        if let Some(mailbox) = self.mailboxes.get_mut(to) {
            mailbox.push_back(message);
        }

        Ok(msg_id)
    }

    /// Deliver a message from a remote (cross-process) sender to a local recipient.
    /// Unlike `send`, this does NOT require the sender to be registered locally.
    pub fn deliver_remote(
        &mut self,
        from: &AgentId,
        to: &AgentId,
        payload: MessagePayload,
        correlation_id: Option<String>,
    ) -> Result<String, String> {
        if !self.mailboxes.contains_key(to) {
            return Err(format!("Recipient {to} is not registered"));
        }

        let msg_id = format!("msg-{}", self.next_msg_id);
        self.next_msg_id += 1;

        let message = CoordinationMessage {
            id: msg_id.clone(),
            from: from.clone(),
            to: to.clone(),
            payload,
            timestamp_ms: crate::timestamp_now(),
            correlation_id,
        };

        if let Some(mailbox) = self.mailboxes.get_mut(to) {
            mailbox.push_back(message);
        }

        Ok(msg_id)
    }

    /// Broadcast a notification to all registered agents except the sender.
    pub fn broadcast_notify(
        &mut self,
        from: &AgentId,
        event: String,
        data: HashMap<String, String>,
    ) -> Result<Vec<String>, String> {
        if !self.profiles.contains_key(from) {
            return Err(format!("Sender {from} is not registered"));
        }

        let recipients: Vec<AgentId> = self
            .profiles
            .keys()
            .filter(|id| *id != from)
            .cloned()
            .collect();

        let mut msg_ids = Vec::new();
        for recipient in recipients {
            let msg_id = self.send(
                from,
                &recipient,
                MessagePayload::Notify {
                    event: event.clone(),
                    data: data.clone(),
                },
                None,
            )?;
            msg_ids.push(msg_id);
        }
        Ok(msg_ids)
    }

    /// Receive (drain) all pending messages for an agent.
    pub fn receive(&mut self, agent_id: &AgentId) -> Vec<CoordinationMessage> {
        self.mailboxes
            .get_mut(agent_id)
            .map(|mb| mb.drain(..).collect())
            .unwrap_or_default()
    }

    /// Peek at the number of pending messages for an agent.
    pub fn pending_count(&self, agent_id: &AgentId) -> usize {
        self.mailboxes
            .get(agent_id)
            .map(VecDeque::len)
            .unwrap_or(0)
    }

    // -----------------------------------------------------------------------
    // Composite intent management
    // -----------------------------------------------------------------------

    /// Create a new composite intent.
    ///
    /// Only agents at `RegionLevel` or `NetworkLevel` may create composites.
    pub fn create_composite(
        &mut self,
        coordinator: &AgentId,
        description: String,
    ) -> Result<String, String> {
        let profile = self
            .profiles
            .get(coordinator)
            .ok_or_else(|| format!("Agent {coordinator} is not registered"))?;

        if profile.role == AgentRole::CellLevel {
            return Err("CellLevel agents cannot create composite intents".to_string());
        }

        let id = format!("composite-{}", self.next_msg_id);
        self.next_msg_id += 1;

        let composite = CompositeIntent {
            id: id.clone(),
            coordinator_agent: coordinator.clone(),
            description,
            sub_intents: Vec::new(),
            state: CompositeState::Gathering,
            created_at_ms: crate::timestamp_now(),
        };

        self.composites.insert(id.clone(), composite);
        Ok(id)
    }

    /// Submit a sub-intent to a composite goal.
    pub fn submit_sub_intent(
        &mut self,
        composite_id: &str,
        agent_id: &AgentId,
        intent: Intent,
    ) -> Result<(), String> {
        let composite = self
            .composites
            .get_mut(composite_id)
            .ok_or_else(|| format!("Composite {composite_id} not found"))?;

        if composite.state != CompositeState::Gathering {
            return Err(format!(
                "Composite {} is in {:?} state, cannot accept sub-intents",
                composite_id, composite.state
            ));
        }

        composite.sub_intents.push(SubIntentEntry {
            agent_id: agent_id.clone(),
            intent,
            accepted: false,
        });

        Ok(())
    }

    /// Accept or reject a sub-intent in a composite.
    pub fn decide_sub_intent(
        &mut self,
        composite_id: &str,
        sub_intent_idx: usize,
        accept: bool,
    ) -> Result<(), String> {
        let composite = self
            .composites
            .get_mut(composite_id)
            .ok_or_else(|| format!("Composite {composite_id} not found"))?;

        if sub_intent_idx >= composite.sub_intents.len() {
            return Err("Sub-intent index out of range".to_string());
        }

        composite.sub_intents[sub_intent_idx].accepted = accept;
        Ok(())
    }

    /// Mark a composite as ready for execution (no more sub-intents expected).
    pub fn finalize_composite(&mut self, composite_id: &str) -> Result<Vec<Intent>, String> {
        let composite = self
            .composites
            .get_mut(composite_id)
            .ok_or_else(|| format!("Composite {composite_id} not found"))?;

        if composite.state != CompositeState::Gathering {
            return Err(format!("Composite {composite_id} is not in Gathering state"));
        }

        composite.state = CompositeState::Ready;

        // Return the accepted sub-intents for execution.
        let accepted: Vec<Intent> = composite
            .sub_intents
            .iter()
            .filter(|e| e.accepted)
            .map(|e| e.intent.clone())
            .collect();

        Ok(accepted)
    }

    /// Update composite state.
    pub fn set_composite_state(
        &mut self,
        composite_id: &str,
        state: CompositeState,
    ) -> Result<(), String> {
        let composite = self
            .composites
            .get_mut(composite_id)
            .ok_or_else(|| format!("Composite {composite_id} not found"))?;
        composite.state = state;
        Ok(())
    }

    /// Get a composite intent by ID.
    pub fn get_composite(&self, composite_id: &str) -> Option<&CompositeIntent> {
        self.composites.get(composite_id)
    }

    /// List all composite intents.
    pub fn list_composites(&self) -> impl Iterator<Item = &CompositeIntent> {
        self.composites.values()
    }

    /// List all registered agent profiles.
    pub fn list_agents(&self) -> impl Iterator<Item = &AgentProfile> {
        self.profiles.values()
    }

    /// Find agents by role.
    pub fn agents_by_role(&self, role: AgentRole) -> Vec<&AgentProfile> {
        self.profiles.values().filter(|p| p.role == role).collect()
    }

    /// Find agents by type.
    pub fn agents_by_type(&self, agent_type: AgentType) -> Vec<&AgentProfile> {
        self.profiles
            .values()
            .filter(|p| p.agent_type == agent_type)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Cross-process coordination
// ---------------------------------------------------------------------------

use std::sync::Arc;
use tokio::sync::Mutex as TokioMutex;

/// Message for cross-process agent coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossProcessMessage {
    /// Unique message ID
    pub id: String,
    /// Source agent ID (including process identifier)
    pub from: String,
    /// Target agent ID (including process identifier)
    pub to: String,
    /// Message payload
    pub payload: MessagePayload,
    /// Timestamp
    pub timestamp_ms: u64,
}

/// Distributed coordinator for cross-process agent coordination
///
/// Extends MessageRouter with cross-process communication capabilities
/// using message passing and distributed consensus.
pub struct DistributedCoordinator {
    /// Local message router
    local_router: Arc<TokioMutex<MessageRouter>>,
    /// Process ID for this coordinator instance
    process_id: String,
    /// Remote message queue for sending to other processes
    outbound_queue: Arc<TokioMutex<Vec<CrossProcessMessage>>>,
    /// Remote message queue for receiving from other processes
    inbound_queue: Arc<TokioMutex<Vec<CrossProcessMessage>>>,
    /// Message ID counter
    next_msg_id: Arc<std::sync::atomic::AtomicU64>,
}

impl DistributedCoordinator {
    /// Creates a new distributed coordinator
    pub fn new(process_id: impl Into<String>) -> Self {
        Self {
            local_router: Arc::new(TokioMutex::new(MessageRouter::new())),
            process_id: process_id.into(),
            outbound_queue: Arc::new(TokioMutex::new(Vec::new())),
            inbound_queue: Arc::new(TokioMutex::new(Vec::new())),
            next_msg_id: Arc::new(std::sync::atomic::AtomicU64::new(1)),
        }
    }

    /// Registers a local agent
    pub async fn register_local_agent(&self, profile: AgentProfile) {
        let mut router = self.local_router.lock().await;
        router.register_agent(profile);
    }

    /// Sends a message to an agent (local or remote)
    pub async fn send_message(
        &self,
        from: &AgentId,
        to: &AgentId,
        payload: MessagePayload,
    ) -> Result<String, String> {
        let router = self.local_router.lock().await;

        // Check if target is local
        if router.get_profile(to).is_some() {
            // Local delivery
            drop(router); // Release lock
            let mut router = self.local_router.lock().await;
            router.send(from, to, payload, None)
        } else {
            // Remote delivery - queue for cross-process transport
            drop(router);

            let msg_id = format!(
                "{}:msg-{}",
                self.process_id,
                self.next_msg_id
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            );

            let cross_msg = CrossProcessMessage {
                id: msg_id.clone(),
                from: format!("{}:{}", self.process_id, from.0),
                to: to.0.clone(),
                payload,
                timestamp_ms: crate::timestamp_now(),
            };

            let mut queue = self.outbound_queue.lock().await;
            queue.push(cross_msg);

            Ok(msg_id)
        }
    }

    /// Receives pending messages for a local agent
    pub async fn receive_messages(&self, agent_id: &AgentId) -> Vec<CoordinationMessage> {
        let mut router = self.local_router.lock().await;
        router.receive(agent_id)
    }

    /// Processes inbound cross-process messages
    ///
    /// Should be called periodically to deliver remote messages to local agents
    pub async fn process_inbound(&self) -> usize {
        let mut inbound = self.inbound_queue.lock().await;
        let messages: Vec<_> = inbound.drain(..).collect();
        let count = messages.len();

        if count > 0 {
            let mut router = self.local_router.lock().await;

            for msg in messages {
                // Extract local agent ID from cross-process format
                let to_agent = AgentId::new(msg.to.split(':').next_back().unwrap_or(&msg.to));
                let from_agent = AgentId::new(&msg.from);

                // Deliver to local mailbox (use deliver_remote since sender is cross-process)
                if router.get_profile(&to_agent).is_some() {
                    let _ = router.deliver_remote(&from_agent, &to_agent, msg.payload, Some(msg.id));
                }
            }
        }

        count
    }

    /// Drains outbound messages for cross-process transport
    ///
    /// Returns messages that need to be sent to other processes
    pub async fn drain_outbound(&self) -> Vec<CrossProcessMessage> {
        let mut queue = self.outbound_queue.lock().await;
        queue.drain(..).collect()
    }

    /// Delivers a cross-process message from another process
    pub async fn deliver_inbound(&self, message: CrossProcessMessage) {
        let mut queue = self.inbound_queue.lock().await;
        queue.push(message);
    }

    /// Returns the process ID
    pub fn process_id(&self) -> &str {
        &self.process_id
    }

    /// Returns a clone of the local router for direct access
    pub fn local_router(&self) -> Arc<TokioMutex<MessageRouter>> {
        Arc::clone(&self.local_router)
    }

    /// Returns the number of pending outbound messages
    pub async fn outbound_count(&self) -> usize {
        let queue = self.outbound_queue.lock().await;
        queue.len()
    }

    /// Returns the number of pending inbound messages
    pub async fn inbound_count(&self) -> usize {
        let queue = self.inbound_queue.lock().await;
        queue.len()
    }
}

impl Default for MessageRouter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IntentType;

    fn profile(name: &str, role: AgentRole) -> AgentProfile {
        AgentProfile {
            agent_id: AgentId::new(name),
            agent_type: AgentType::Custom,
            role,
            scope: None,
        }
    }

    #[test]
    fn test_role_hierarchy() {
        assert!(AgentRole::NetworkLevel.outranks(AgentRole::RegionLevel));
        assert!(AgentRole::RegionLevel.outranks(AgentRole::CellLevel));
        assert!(!AgentRole::CellLevel.outranks(AgentRole::RegionLevel));
        assert!(AgentRole::NetworkLevel.can_coordinate(AgentRole::CellLevel));
    }

    #[test]
    fn test_send_and_receive() {
        let mut router = MessageRouter::new();
        router.register_agent(profile("a1", AgentRole::CellLevel));
        router.register_agent(profile("a2", AgentRole::CellLevel));

        let msg_id = router
            .send(
                &AgentId::new("a1"),
                &AgentId::new("a2"),
                MessagePayload::Request {
                    action: "get_load".to_string(),
                    parameters: HashMap::new(),
                },
                None,
            )
            .unwrap();

        assert!(msg_id.starts_with("msg-"));
        assert_eq!(router.pending_count(&AgentId::new("a2")), 1);

        let msgs = router.receive(&AgentId::new("a2"));
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].from, AgentId::new("a1"));
        assert_eq!(router.pending_count(&AgentId::new("a2")), 0);
    }

    #[test]
    fn test_send_to_unregistered_fails() {
        let mut router = MessageRouter::new();
        router.register_agent(profile("a1", AgentRole::CellLevel));
        let result = router.send(
            &AgentId::new("a1"),
            &AgentId::new("ghost"),
            MessagePayload::Notify {
                event: "test".to_string(),
                data: HashMap::new(),
            },
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_broadcast_notify() {
        let mut router = MessageRouter::new();
        router.register_agent(profile("a1", AgentRole::NetworkLevel));
        router.register_agent(profile("a2", AgentRole::CellLevel));
        router.register_agent(profile("a3", AgentRole::CellLevel));

        let ids = router
            .broadcast_notify(
                &AgentId::new("a1"),
                "topology_change".to_string(),
                HashMap::new(),
            )
            .unwrap();

        assert_eq!(ids.len(), 2); // a2 and a3
        assert_eq!(router.pending_count(&AgentId::new("a2")), 1);
        assert_eq!(router.pending_count(&AgentId::new("a3")), 1);
        assert_eq!(router.pending_count(&AgentId::new("a1")), 0); // sender excluded
    }

    #[test]
    fn test_composite_intent_lifecycle() {
        let mut router = MessageRouter::new();
        router.register_agent(profile("coordinator", AgentRole::NetworkLevel));
        router.register_agent(profile("cell-agent-1", AgentRole::CellLevel));
        router.register_agent(profile("cell-agent-2", AgentRole::CellLevel));

        // Create composite.
        let cid = router
            .create_composite(&AgentId::new("coordinator"), "Optimize region".to_string())
            .unwrap();

        // Cell-level agents submit sub-intents.
        let sub1 = Intent::new(AgentId::new("cell-agent-1"), IntentType::OptimizeResources)
            .with_target("cell-1");
        let sub2 = Intent::new(AgentId::new("cell-agent-2"), IntentType::OptimizeResources)
            .with_target("cell-2");

        router
            .submit_sub_intent(&cid, &AgentId::new("cell-agent-1"), sub1)
            .unwrap();
        router
            .submit_sub_intent(&cid, &AgentId::new("cell-agent-2"), sub2)
            .unwrap();

        // Coordinator accepts both.
        router.decide_sub_intent(&cid, 0, true).unwrap();
        router.decide_sub_intent(&cid, 1, true).unwrap();

        // Finalize.
        let accepted = router.finalize_composite(&cid).unwrap();
        assert_eq!(accepted.len(), 2);

        let composite = router.get_composite(&cid).unwrap();
        assert_eq!(composite.state, CompositeState::Ready);
    }

    #[test]
    fn test_cell_level_cannot_create_composite() {
        let mut router = MessageRouter::new();
        router.register_agent(profile("cell-only", AgentRole::CellLevel));
        let result =
            router.create_composite(&AgentId::new("cell-only"), "attempt".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_agents_by_role() {
        let mut router = MessageRouter::new();
        router.register_agent(profile("n1", AgentRole::NetworkLevel));
        router.register_agent(profile("r1", AgentRole::RegionLevel));
        router.register_agent(profile("c1", AgentRole::CellLevel));
        router.register_agent(profile("c2", AgentRole::CellLevel));

        assert_eq!(router.agents_by_role(AgentRole::CellLevel).len(), 2);
        assert_eq!(router.agents_by_role(AgentRole::NetworkLevel).len(), 1);
    }

    #[test]
    fn test_correlation_id() {
        let mut router = MessageRouter::new();
        router.register_agent(profile("a1", AgentRole::CellLevel));
        router.register_agent(profile("a2", AgentRole::CellLevel));

        // Send request with correlation ID.
        let _ = router
            .send(
                &AgentId::new("a1"),
                &AgentId::new("a2"),
                MessagePayload::Request {
                    action: "query".to_string(),
                    parameters: HashMap::new(),
                },
                Some("corr-123".to_string()),
            )
            .unwrap();

        let msgs = router.receive(&AgentId::new("a2"));
        assert_eq!(msgs[0].correlation_id.as_deref(), Some("corr-123"));
    }

    #[tokio::test]
    async fn test_distributed_coordinator_creation() {
        let coordinator = DistributedCoordinator::new("process-1");
        assert_eq!(coordinator.process_id(), "process-1");
        assert_eq!(coordinator.outbound_count().await, 0);
        assert_eq!(coordinator.inbound_count().await, 0);
    }

    #[tokio::test]
    async fn test_distributed_local_messaging() {
        let coordinator = DistributedCoordinator::new("process-1");

        coordinator
            .register_local_agent(profile("agent-1", AgentRole::CellLevel))
            .await;
        coordinator
            .register_local_agent(profile("agent-2", AgentRole::CellLevel))
            .await;

        // Send local message
        let result = coordinator
            .send_message(
                &AgentId::new("agent-1"),
                &AgentId::new("agent-2"),
                MessagePayload::Request {
                    action: "test".to_string(),
                    parameters: HashMap::new(),
                },
            )
            .await;

        assert!(result.is_ok());

        // Receive message
        let messages = coordinator.receive_messages(&AgentId::new("agent-2")).await;
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].from, AgentId::new("agent-1"));
    }

    #[tokio::test]
    async fn test_distributed_remote_messaging() {
        let coordinator = DistributedCoordinator::new("process-1");

        coordinator
            .register_local_agent(profile("agent-1", AgentRole::CellLevel))
            .await;

        // Send to remote agent (not registered locally)
        let result = coordinator
            .send_message(
                &AgentId::new("agent-1"),
                &AgentId::new("agent-remote"),
                MessagePayload::Notify {
                    event: "test".to_string(),
                    data: HashMap::new(),
                },
            )
            .await;

        assert!(result.is_ok());

        // Message should be queued for cross-process delivery
        assert_eq!(coordinator.outbound_count().await, 1);

        let outbound = coordinator.drain_outbound().await;
        assert_eq!(outbound.len(), 1);
        assert_eq!(outbound[0].to, "agent-remote");
    }

    #[tokio::test]
    async fn test_distributed_inbound_delivery() {
        let coordinator = DistributedCoordinator::new("process-2");

        coordinator
            .register_local_agent(profile("agent-local", AgentRole::CellLevel))
            .await;

        // Simulate receiving a cross-process message
        let cross_msg = CrossProcessMessage {
            id: "process-1:msg-1".to_string(),
            from: "process-1:agent-remote".to_string(),
            to: "agent-local".to_string(),
            payload: MessagePayload::Notify {
                event: "remote_event".to_string(),
                data: HashMap::new(),
            },
            timestamp_ms: crate::timestamp_now(),
        };

        coordinator.deliver_inbound(cross_msg).await;
        assert_eq!(coordinator.inbound_count().await, 1);

        // Process inbound messages
        let processed = coordinator.process_inbound().await;
        assert_eq!(processed, 1);
        assert_eq!(coordinator.inbound_count().await, 0);

        // Check if message was delivered to local agent
        let messages = coordinator.receive_messages(&AgentId::new("agent-local")).await;
        assert_eq!(messages.len(), 1);
    }
}
