//! AI Agent Framework (AAF) for 6G Networks
//!
//! Implements multi-agent coordination for 6G networks:
//! - OAuth 2.0 agent authentication
//! - Intent-based networking
//! - Coordinated decision making
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                           Agent Framework                                │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │ Agent Coordinator                                                │   │
//! │  │  • Agent registration                                            │   │
//! │  │  • Intent routing                                                │   │
//! │  │  • Conflict resolution                                           │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │ Agent Types                                                      │   │
//! │  │  • Mobility Agent                                                │   │
//! │  │  • Resource Agent                                                │   │
//! │  │  • QoS Agent                                                     │   │
//! │  │  • Security Agent                                                │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │ Authentication                                                   │   │
//! │  │  • OAuth 2.0 tokens                                              │   │
//! │  │  • Capability-based access                                       │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Returns current timestamp in milliseconds since UNIX epoch
fn timestamp_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Agent identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId(pub String);

impl AgentId {
    /// Creates a new agent ID
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl std::fmt::Display for AgentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Agent type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentType {
    /// Mobility management agent
    Mobility,
    /// Resource allocation agent
    Resource,
    /// QoS optimization agent
    Qos,
    /// Security monitoring agent
    Security,
    /// Network slicing agent
    Slicing,
    /// Custom agent
    Custom,
}

/// Agent capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCapabilities {
    /// Can read network state
    pub read_state: bool,
    /// Can modify network configuration
    pub modify_config: bool,
    /// Can trigger actions (handover, etc.)
    pub trigger_actions: bool,
    /// Allowed intents
    pub allowed_intents: Vec<String>,
    /// Resource limits
    pub resource_limits: ResourceLimits,
}

impl Default for AgentCapabilities {
    fn default() -> Self {
        Self {
            read_state: true,
            modify_config: false,
            trigger_actions: false,
            allowed_intents: vec!["query".to_string()],
            resource_limits: ResourceLimits::default(),
        }
    }
}

/// Resource limits for an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum requests per second
    pub max_requests_per_second: u32,
    /// Maximum concurrent operations
    pub max_concurrent_ops: u32,
    /// Maximum data access scope
    pub max_data_scope: DataScope,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_requests_per_second: 100,
            max_concurrent_ops: 10,
            max_data_scope: DataScope::Cell,
        }
    }
}

/// Data access scope
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataScope {
    /// Single UE
    Ue,
    /// Single cell
    Cell,
    /// Regional (multiple cells)
    Regional,
    /// Global (entire network)
    Global,
}

/// OAuth 2.0 token for agent authentication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentToken {
    /// Token string
    pub token: String,
    /// Agent ID
    pub agent_id: AgentId,
    /// Expiration time
    pub expires_at: u64,
    /// Granted scopes
    pub scopes: Vec<String>,
}

impl AgentToken {
    /// Checks if the token is valid
    pub fn is_valid(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        self.expires_at > now
    }

    /// Checks if token has a specific scope
    pub fn has_scope(&self, scope: &str) -> bool {
        self.scopes.iter().any(|s| s == scope || s == "*")
    }
}

/// Intent representing an agent's desired action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intent {
    /// Unique intent ID
    pub id: String,
    /// Source agent
    pub agent_id: AgentId,
    /// Intent type
    pub intent_type: IntentType,
    /// Target (UE ID, cell ID, etc.)
    pub target: Option<String>,
    /// Parameters
    pub parameters: HashMap<String, String>,
    /// Priority (1-10, higher = more important)
    pub priority: u8,
    /// Timestamp (milliseconds since epoch)
    pub timestamp_ms: u64,
}

impl Intent {
    /// Creates a new intent
    pub fn new(agent_id: AgentId, intent_type: IntentType) -> Self {
        Self {
            id: uuid_simple(),
            agent_id,
            intent_type,
            target: None,
            parameters: HashMap::new(),
            priority: 5,
            timestamp_ms: timestamp_now(),
        }
    }

    /// Sets the target
    pub fn with_target(mut self, target: impl Into<String>) -> Self {
        self.target = Some(target.into());
        self
    }

    /// Adds a parameter
    pub fn with_param(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.parameters.insert(key.into(), value.into());
        self
    }

    /// Sets the priority
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority.min(10);
        self
    }
}

/// Intent types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntentType {
    /// Query network state
    Query,
    /// Optimize resource allocation
    OptimizeResources,
    /// Trigger handover
    TriggerHandover,
    /// Adjust QoS
    AdjustQos,
    /// Create network slice
    CreateSlice,
    /// Modify network slice
    ModifySlice,
    /// Custom intent
    Custom(String),
}

/// Intent resolution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentResult {
    /// Intent ID
    pub intent_id: String,
    /// Whether the intent was fulfilled
    pub success: bool,
    /// Result data
    pub data: HashMap<String, String>,
    /// Error message if failed
    pub error: Option<String>,
}

/// Agent registration
#[derive(Debug, Clone)]
pub struct AgentRegistration {
    /// Agent ID
    pub agent_id: AgentId,
    /// Agent type
    pub agent_type: AgentType,
    /// Capabilities
    pub capabilities: AgentCapabilities,
    /// Registration time
    pub registered_at: Instant,
    /// Last heartbeat
    pub last_heartbeat: Instant,
    /// Active token
    pub token: Option<AgentToken>,
}

/// Agent coordinator for multi-agent management
#[derive(Debug)]
pub struct AgentCoordinator {
    /// Registered agents
    agents: HashMap<AgentId, AgentRegistration>,
    /// Pending intents
    pending_intents: Vec<Intent>,
    /// Token lifetime
    token_lifetime: Duration,
    /// Next token ID
    next_token_id: u64,
}

impl AgentCoordinator {
    /// Creates a new coordinator
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
            pending_intents: Vec::new(),
            token_lifetime: Duration::from_secs(3600), // 1 hour
            next_token_id: 1,
        }
    }

    /// Registers an agent
    pub fn register_agent(
        &mut self,
        agent_id: AgentId,
        agent_type: AgentType,
        capabilities: AgentCapabilities,
    ) -> AgentToken {
        let registration = AgentRegistration {
            agent_id: agent_id.clone(),
            agent_type,
            capabilities,
            registered_at: Instant::now(),
            last_heartbeat: Instant::now(),
            token: None,
        };

        // Generate token
        let token = self.generate_token(&agent_id);

        let mut reg = registration;
        reg.token = Some(token.clone());
        self.agents.insert(agent_id, reg);

        token
    }

    /// Generates an authentication token
    fn generate_token(&mut self, agent_id: &AgentId) -> AgentToken {
        let token_id = self.next_token_id;
        self.next_token_id += 1;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        AgentToken {
            token: format!("agent_token_{}_{}", agent_id.0, token_id),
            agent_id: agent_id.clone(),
            expires_at: now + self.token_lifetime.as_secs(),
            scopes: vec!["read".to_string(), "write".to_string()],
        }
    }

    /// Refreshes an agent's token
    pub fn refresh_token(&mut self, agent_id: &AgentId) -> Option<AgentToken> {
        // Check if agent exists first
        if !self.agents.contains_key(agent_id) {
            return None;
        }

        // Generate token (this borrows self immutably)
        let token = self.generate_token(agent_id);

        // Now mutate the registration
        if let Some(registration) = self.agents.get_mut(agent_id) {
            registration.token = Some(token.clone());
            registration.last_heartbeat = Instant::now();
            Some(token)
        } else {
            None
        }
    }

    /// Validates a token
    pub fn validate_token(&self, token: &str) -> Option<&AgentRegistration> {
        self.agents.values().find(|reg| {
            reg.token
                .as_ref()
                .map(|t| t.token == token && t.is_valid())
                .unwrap_or(false)
        })
    }

    /// Submits an intent
    pub fn submit_intent(&mut self, intent: Intent) -> Result<(), String> {
        // Validate agent is registered
        if !self.agents.contains_key(&intent.agent_id) {
            return Err("Agent not registered".to_string());
        }

        // Check capabilities
        let registration = self.agents.get(&intent.agent_id).unwrap();
        let caps = &registration.capabilities;

        match intent.intent_type {
            IntentType::Query => {
                if !caps.read_state {
                    return Err("Agent lacks read_state capability".to_string());
                }
            }
            IntentType::TriggerHandover | IntentType::AdjustQos => {
                if !caps.trigger_actions {
                    return Err("Agent lacks trigger_actions capability".to_string());
                }
            }
            IntentType::OptimizeResources | IntentType::CreateSlice | IntentType::ModifySlice => {
                if !caps.modify_config {
                    return Err("Agent lacks modify_config capability".to_string());
                }
            }
            _ => {}
        }

        self.pending_intents.push(intent);
        Ok(())
    }

    /// Processes pending intents
    pub fn process_intents(&mut self) -> Vec<IntentResult> {
        // Sort by priority (highest first)
        self.pending_intents
            .sort_by(|a, b| b.priority.cmp(&a.priority));

        // Process each intent
        let results: Vec<IntentResult> = self
            .pending_intents
            .drain(..)
            .map(|intent| {
                // Simplified processing - in production this would route to appropriate handlers
                IntentResult {
                    intent_id: intent.id,
                    success: true,
                    data: HashMap::new(),
                    error: None,
                }
            })
            .collect();

        results
    }

    /// Gets registered agents
    pub fn get_agents(&self) -> impl Iterator<Item = &AgentRegistration> {
        self.agents.values()
    }

    /// Removes stale agents (no heartbeat for specified duration)
    pub fn cleanup_stale_agents(&mut self, max_age: Duration) {
        self.agents.retain(|_, reg| reg.last_heartbeat.elapsed() < max_age);
    }

    /// Returns the number of registered agents
    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }
}

impl Default for AgentCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

/// Generates a simple UUID-like string
fn uuid_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("intent-{}-{}", now.as_secs(), now.subsec_nanos())
}

/// Agent message types
#[derive(Debug)]
pub enum AgentMessage {
    /// Register agent with ID, type, and capabilities
    Register {
        /// Unique agent identifier
        agent_id: AgentId,
        /// Type of agent (Mobility, Resource, QoS, etc.)
        agent_type: AgentType,
        /// Agent capabilities and permissions
        capabilities: AgentCapabilities,
        /// Response channel for async result
        response_tx: Option<tokio::sync::oneshot::Sender<AgentResponse>>,
    },
    /// Submit intent for processing
    SubmitIntent {
        /// Intent to be processed
        intent: Intent,
        /// Response channel for async result
        response_tx: Option<tokio::sync::oneshot::Sender<AgentResponse>>,
    },
    /// Refresh authentication token
    RefreshToken {
        /// Agent requesting token refresh
        agent_id: AgentId,
        /// Response channel for async result
        response_tx: Option<tokio::sync::oneshot::Sender<AgentResponse>>,
    },
    /// Heartbeat to maintain registration
    Heartbeat {
        /// Agent sending heartbeat
        agent_id: AgentId,
    },
}

/// Agent response types
#[derive(Debug)]
pub enum AgentResponse {
    /// Registration result
    Registered(AgentToken),
    /// Intent submitted
    IntentSubmitted(String),
    /// Token refreshed
    TokenRefreshed(AgentToken),
    /// Intent results
    IntentResults(Vec<IntentResult>),
    /// Error
    Error(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_registration() {
        let mut coordinator = AgentCoordinator::new();

        let token = coordinator.register_agent(
            AgentId::new("mobility-1"),
            AgentType::Mobility,
            AgentCapabilities::default(),
        );

        assert!(token.is_valid());
        assert_eq!(coordinator.agent_count(), 1);
    }

    #[test]
    fn test_token_validation() {
        let mut coordinator = AgentCoordinator::new();

        let token = coordinator.register_agent(
            AgentId::new("test-agent"),
            AgentType::Custom,
            AgentCapabilities::default(),
        );

        let registration = coordinator.validate_token(&token.token);
        assert!(registration.is_some());

        let invalid = coordinator.validate_token("invalid_token");
        assert!(invalid.is_none());
    }

    #[test]
    fn test_intent_submission() {
        let mut coordinator = AgentCoordinator::new();

        coordinator.register_agent(
            AgentId::new("query-agent"),
            AgentType::Custom,
            AgentCapabilities {
                read_state: true,
                ..Default::default()
            },
        );

        let intent = Intent::new(AgentId::new("query-agent"), IntentType::Query)
            .with_target("cell-1")
            .with_priority(8);

        let result = coordinator.submit_intent(intent);
        assert!(result.is_ok());
    }

    #[test]
    fn test_intent_capability_check() {
        let mut coordinator = AgentCoordinator::new();

        // Register agent without trigger_actions capability
        coordinator.register_agent(
            AgentId::new("limited-agent"),
            AgentType::Custom,
            AgentCapabilities {
                trigger_actions: false,
                ..Default::default()
            },
        );

        // Try to submit handover intent
        let intent = Intent::new(AgentId::new("limited-agent"), IntentType::TriggerHandover);

        let result = coordinator.submit_intent(intent);
        assert!(result.is_err());
    }

    #[test]
    fn test_intent_processing() {
        let mut coordinator = AgentCoordinator::new();

        coordinator.register_agent(
            AgentId::new("test-agent"),
            AgentType::Custom,
            AgentCapabilities {
                read_state: true,
                ..Default::default()
            },
        );

        // Submit multiple intents with different priorities
        let intent1 = Intent::new(AgentId::new("test-agent"), IntentType::Query).with_priority(3);
        let intent2 = Intent::new(AgentId::new("test-agent"), IntentType::Query).with_priority(8);

        coordinator.submit_intent(intent1).unwrap();
        coordinator.submit_intent(intent2).unwrap();

        let results = coordinator.process_intents();
        assert_eq!(results.len(), 2);
    }
}
