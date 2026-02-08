//! AI Agent Framework (AAF) for 6G Networks
//!
//! Implements multi-agent coordination for 6G networks:
//! - OAuth 2.0 agent authentication
//! - Intent-based networking with real execution
//! - Conflict detection and resolution between competing intents
//! - Safety constraints and guardrails
//! - Multi-agent coordination protocol with role hierarchy
//! - Decision audit trail
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                           Agent Framework                                │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │ Agent Coordinator                                                │   │
//! │  │  • Agent registration & authentication                          │   │
//! │  │  • Intent routing & execution                                   │   │
//! │  │  • Conflict resolution                                          │   │
//! │  │  • Safety guardrails                                            │   │
//! │  │  • Decision audit trail                                         │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │ Agent Types                                                      │   │
//! │  │  • Mobility Agent                                                │   │
//! │  │  • Resource Agent                                                │   │
//! │  │  • QoS Agent                                                     │   │
//! │  │  • Security Agent                                                │   │
//! │  │  • Slicing Agent                                                 │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │ Multi-Agent Coordination                                        │   │
//! │  │  • Role hierarchy (Network > Region > Cell)                     │   │
//! │  │  • Agent-to-agent messaging (Request/Response/Notify)           │   │
//! │  │  • Composite intents (cooperative multi-agent goals)            │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │ Authentication                                                   │   │
//! │  │  • OAuth 2.0 tokens                                              │   │
//! │  │  • Capability-based access                                       │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```

// ---------------------------------------------------------------------------
// Sub-modules
// ---------------------------------------------------------------------------

pub mod audit;
pub mod conflict;
pub mod coordination;
pub mod execution;
pub mod learning;
pub mod nkef_bridge;
pub mod nwdaf_bridge;
pub mod safety;

// ---------------------------------------------------------------------------
// Re-exports for convenience
// ---------------------------------------------------------------------------

pub use audit::{AuditEntry, AuditEvent, AuditTrail};
pub use conflict::{
    BlockedIntentNotification, Conflict, ConflictDetector, ConflictResolutionOutcome,
    ConflictResolver, ContestedResource, IntentQueue, PersistentIntentStore,
    PersistentIntentStoreBuilder, ResolutionStrategy,
};
pub use coordination::{
    AgentProfile, AgentRole, CompositeIntent, CompositeState, CoordinationMessage,
    CrossProcessMessage, DistributedCoordinator, MessagePayload, MessageRouter, SubIntentEntry,
};
pub use execution::{
    AdjustQosExecutor, AffectedResource, CreateSliceExecutor, ExecutorRegistry, InMemoryStateProvider,
    IntentExecutionResult, IntentExecutor, IntentStatus, ModifySliceExecutor,
    OptimizeResourcesExecutor, QueryExecutor, ResourceAccess, ResourceKind, StateProvider,
    TriggerHandoverExecutor,
};
pub use learning::{
    Action, Algorithm, Experience, LearningConfig, LearningError, RLAgent, State,
};
pub use safety::{
    ForbiddenAction, ForbiddenRule, SafetyChecker, SafetyPolicy, SafetyPolicyOverride,
    SafetyViolation, ViolationSeverity,
};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Monotonic intent counter (avoids duplicate IDs within the same second)
// ---------------------------------------------------------------------------

static INTENT_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Returns current timestamp in milliseconds since UNIX epoch.
fn timestamp_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Core types (unchanged public API)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// OAuth 2.0 Identity Provider (IdP) Integration
// ---------------------------------------------------------------------------

/// OAuth 2.0 grant types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GrantType {
    /// Authorization Code flow (most secure)
    AuthorizationCode,
    /// Client Credentials flow (service-to-service)
    ClientCredentials,
    /// Refresh Token flow
    RefreshToken,
}

/// OAuth 2.0 token request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenRequest {
    /// Grant type
    pub grant_type: GrantType,
    /// Client ID
    pub client_id: String,
    /// Client secret
    pub client_secret: String,
    /// Authorization code (for AuthorizationCode grant)
    pub code: Option<String>,
    /// Refresh token (for RefreshToken grant)
    pub refresh_token: Option<String>,
    /// Requested scopes
    pub scopes: Vec<String>,
}

/// OAuth 2.0 token response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenResponse {
    /// Access token
    pub access_token: String,
    /// Token type (usually "Bearer")
    pub token_type: String,
    /// Expiration time in seconds
    pub expires_in: u64,
    /// Refresh token (optional)
    pub refresh_token: Option<String>,
    /// Granted scopes
    pub scope: String,
}

/// OAuth 2.0 Identity Provider client
///
/// Provides integration with external OAuth 2.0 Identity Providers
/// for proper authentication and authorization.
pub struct OAuth2Client {
    /// IdP authorization endpoint
    _auth_endpoint: String,
    /// IdP token endpoint
    _token_endpoint: String,
    /// Client ID
    client_id: String,
    /// Client secret
    _client_secret: String,
    /// Cached tokens by agent ID
    token_cache: std::sync::Mutex<HashMap<AgentId, TokenResponse>>,
}

impl OAuth2Client {
    /// Creates a new OAuth 2.0 client
    pub fn new(
        auth_endpoint: impl Into<String>,
        token_endpoint: impl Into<String>,
        client_id: impl Into<String>,
        client_secret: impl Into<String>,
    ) -> Self {
        Self {
            _auth_endpoint: auth_endpoint.into(),
            _token_endpoint: token_endpoint.into(),
            client_id: client_id.into(),
            _client_secret: client_secret.into(),
            token_cache: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Requests an access token using client credentials flow
    ///
    /// This is suitable for service-to-service authentication where
    /// no user interaction is required.
    pub fn request_token_client_credentials(
        &self,
        agent_id: &AgentId,
        scopes: Vec<String>,
    ) -> Result<AgentToken, String> {
        // Placeholder implementation - in production this would:
        // 1. Make HTTP POST to token_endpoint with client credentials
        // 2. Parse the JSON response
        // 3. Validate the response
        // 4. Cache the token

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let token_response = TokenResponse {
            access_token: format!("oauth2_{}_{}", self.client_id, uuid_simple()),
            token_type: "Bearer".to_string(),
            expires_in: 3600,
            refresh_token: Some(format!("refresh_{}_{}", self.client_id, uuid_simple())),
            scope: scopes.join(" "),
        };

        // Cache the response
        {
            let mut cache = self.token_cache.lock().unwrap();
            cache.insert(agent_id.clone(), token_response.clone());
        }

        Ok(AgentToken {
            token: token_response.access_token,
            agent_id: agent_id.clone(),
            expires_at: now + token_response.expires_in,
            scopes,
        })
    }

    /// Validates a token with the IdP
    ///
    /// In production, this would make an introspection request to the IdP
    pub fn validate_token(&self, token: &str) -> Result<bool, String> {
        // Placeholder implementation - in production this would:
        // 1. Make HTTP POST to introspection endpoint
        // 2. Parse the response
        // 3. Return whether the token is active

        // For now, just check if it starts with our prefix
        Ok(token.starts_with("oauth2_"))
    }

    /// Refreshes a token using a refresh token
    pub fn refresh_token(
        &self,
        agent_id: &AgentId,
        refresh_token: &str,
    ) -> Result<AgentToken, String> {
        // Placeholder implementation - in production this would:
        // 1. Make HTTP POST to token_endpoint with refresh token
        // 2. Parse the JSON response
        // 3. Update the cached token

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let token_response = TokenResponse {
            access_token: format!("oauth2_refreshed_{}_{}", self.client_id, uuid_simple()),
            token_type: "Bearer".to_string(),
            expires_in: 3600,
            refresh_token: Some(refresh_token.to_string()),
            scope: "read write".to_string(),
        };

        // Update cache
        {
            let mut cache = self.token_cache.lock().unwrap();
            cache.insert(agent_id.clone(), token_response.clone());
        }

        Ok(AgentToken {
            token: token_response.access_token,
            agent_id: agent_id.clone(),
            expires_at: now + token_response.expires_in,
            scopes: vec!["read".to_string(), "write".to_string()],
        })
    }

    /// Revokes a token
    pub fn revoke_token(&self, token: &str) -> Result<(), String> {
        // Placeholder implementation - in production this would:
        // 1. Make HTTP POST to revocation endpoint
        // 2. Handle the response

        if !token.starts_with("oauth2_") {
            return Err("Invalid token format".to_string());
        }

        Ok(())
    }

    /// Returns the cached token for an agent, if available
    pub fn get_cached_token(&self, agent_id: &AgentId) -> Option<TokenResponse> {
        let cache = self.token_cache.lock().unwrap();
        cache.get(agent_id).cloned()
    }

    /// Clears the token cache
    pub fn clear_cache(&self) {
        let mut cache = self.token_cache.lock().unwrap();
        cache.clear();
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

/// Intent resolution result (backward-compatible simple form).
///
/// For richer results see [`IntentExecutionResult`].
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

impl From<IntentExecutionResult> for IntentResult {
    fn from(r: IntentExecutionResult) -> Self {
        Self {
            intent_id: r.intent_id,
            success: r.status == IntentStatus::Success,
            data: r.output,
            error: r.message,
        }
    }
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

// ---------------------------------------------------------------------------
// AgentCoordinator -- integrated with all new subsystems
// ---------------------------------------------------------------------------

/// Agent coordinator for multi-agent management.
///
/// The coordinator now integrates:
/// - **Execution**: real intent execution through [`ExecutorRegistry`]
/// - **Conflict resolution**: via [`IntentQueue`] with configurable strategy
/// - **Safety**: pre-execution validation through [`SafetyChecker`]
/// - **Coordination**: agent messaging and composite intents via [`MessageRouter`]
/// - **Audit**: decision trail via [`AuditTrail`]
///
/// The original public API (`register_agent`, `submit_intent`, `process_intents`,
/// etc.) is fully preserved.
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

    // -- New subsystems --
    /// Intent executor registry.
    executor_registry: ExecutorRegistry,
    /// Network state provider.
    state_provider: std::sync::Arc<dyn StateProvider>,
    /// Intent queue with conflict checking.
    intent_queue: IntentQueue,
    /// Safety checker.
    safety_checker: SafetyChecker,
    /// Multi-agent message router.
    message_router: MessageRouter,
    /// Decision audit trail.
    audit_trail: AuditTrail,
}

impl AgentCoordinator {
    /// Creates a new coordinator with default subsystems.
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
            pending_intents: Vec::new(),
            token_lifetime: Duration::from_secs(3600), // 1 hour
            next_token_id: 1,
            executor_registry: ExecutorRegistry::new(),
            state_provider: std::sync::Arc::new(InMemoryStateProvider::new()),
            intent_queue: IntentQueue::default(),
            safety_checker: SafetyChecker::default(),
            message_router: MessageRouter::new(),
            audit_trail: AuditTrail::default(),
        }
    }

    /// Creates a coordinator with custom configuration.
    pub fn with_config(
        state_provider: std::sync::Arc<dyn StateProvider>,
        resolution_strategy: ResolutionStrategy,
        safety_policy: SafetyPolicy,
        audit_capacity: usize,
    ) -> Self {
        Self {
            agents: HashMap::new(),
            pending_intents: Vec::new(),
            token_lifetime: Duration::from_secs(3600),
            next_token_id: 1,
            executor_registry: ExecutorRegistry::new(),
            state_provider,
            intent_queue: IntentQueue::new(resolution_strategy),
            safety_checker: SafetyChecker::new(safety_policy),
            message_router: MessageRouter::new(),
            audit_trail: AuditTrail::new(audit_capacity),
        }
    }

    // -----------------------------------------------------------------------
    // Original public API (backward compatible)
    // -----------------------------------------------------------------------

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
        self.agents.insert(agent_id.clone(), reg);

        // Register in the message router with a default CellLevel role.
        self.message_router.register_agent(AgentProfile {
            agent_id: agent_id.clone(),
            agent_type,
            role: AgentRole::CellLevel,
            scope: None,
        });

        // Audit.
        self.audit_trail.record_agent_registered(
            &agent_id,
            &format!("{agent_type:?}"),
        );

        token
    }

    /// Registers an agent with an explicit role for multi-agent coordination.
    pub fn register_agent_with_role(
        &mut self,
        agent_id: AgentId,
        agent_type: AgentType,
        capabilities: AgentCapabilities,
        role: AgentRole,
        scope: Option<String>,
    ) -> AgentToken {
        let token = self.register_agent(agent_id.clone(), agent_type, capabilities);

        // Override the default CellLevel profile with the requested role.
        self.message_router.register_agent(AgentProfile {
            agent_id,
            agent_type,
            role,
            scope,
        });

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

    /// Submits an intent.
    ///
    /// The intent goes through:
    /// 1. Agent capability check
    /// 2. Safety policy validation
    /// 3. Conflict detection against the intent queue
    ///
    /// If any check fails the intent is rejected and the error is returned.
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

        // Safety check -- project the resources that would be affected.
        let projected: Vec<AffectedResource> =
            ConflictDetector::projected_write_resources(&intent)
                .into_iter()
                .map(|cr| AffectedResource {
                    kind: cr.kind,
                    id: cr.id,
                    access: ResourceAccess::Write,
                })
                .collect();

        if let Err(violations) = self.safety_checker.validate(&intent, &projected) {
            for v in &violations {
                self.audit_trail.record_safety_violation(
                    &intent.agent_id,
                    &intent.id,
                    &v.rule_description,
                    &format!("{:?}", v.severity),
                );
            }
            let msgs: Vec<String> = violations.into_iter().map(|v| v.rule_description).collect();
            return Err(format!("Safety violation: {}", msgs.join("; ")));
        }

        // Conflict check via the intent queue.
        match self.intent_queue.enqueue(intent.clone()) {
            Ok(()) => {
                self.pending_intents.push(intent);
                Ok(())
            }
            Err(notification) => {
                self.audit_trail.record_blocked(
                    &notification.agent_id,
                    &notification.intent_id,
                    &notification.winner_intent_id,
                    &notification.reason,
                );
                Err(format!(
                    "Intent blocked by conflict: {}",
                    notification.reason
                ))
            }
        }
    }

    /// Processes pending intents with real execution.
    ///
    /// Returns the backward-compatible [`IntentResult`] vec.  For richer
    /// results use [`process_intents_full`].
    pub fn process_intents(&mut self) -> Vec<IntentResult> {
        self.process_intents_full()
            .into_iter()
            .map(IntentResult::from)
            .collect()
    }

    /// Processes pending intents and returns rich [`IntentExecutionResult`]s.
    ///
    /// Intents are:
    /// 1. Sorted by priority (highest first)
    /// 2. Executed through the [`ExecutorRegistry`]
    /// 3. Recorded in the [`AuditTrail`]
    pub fn process_intents_full(&mut self) -> Vec<IntentExecutionResult> {
        // Sort by priority (highest first)
        self.pending_intents
            .sort_by(|a, b| b.priority.cmp(&a.priority));

        let intents: Vec<Intent> = self.pending_intents.drain(..).collect();
        // Also drain the intent queue so it stays in sync.
        let _ = self.intent_queue.drain_ready();

        let mut results = Vec::with_capacity(intents.len());

        for intent in &intents {
            let result = self.executor_registry.execute(intent, self.state_provider.as_ref());

            // Record in audit trail.
            self.audit_trail
                .record_execution(&intent.agent_id, intent, &result);

            // If it was a handover and it succeeded, track completion for safety.
            if intent.intent_type == IntentType::TriggerHandover
                && result.status == IntentStatus::Success
            {
                self.safety_checker.handover_completed();
            }

            results.push(result);
        }

        results
    }

    /// Gets registered agents
    pub fn get_agents(&self) -> impl Iterator<Item = &AgentRegistration> {
        self.agents.values()
    }

    /// Removes stale agents (no heartbeat for specified duration)
    pub fn cleanup_stale_agents(&mut self, max_age: Duration) {
        let removed: Vec<AgentId> = self
            .agents
            .iter()
            .filter(|(_, reg)| reg.last_heartbeat.elapsed() >= max_age)
            .map(|(id, _)| id.clone())
            .collect();

        for id in &removed {
            self.agents.remove(id);
            self.message_router.unregister_agent(id);
        }
    }

    /// Returns the number of registered agents
    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }

    // -----------------------------------------------------------------------
    // New public API -- subsystem access
    // -----------------------------------------------------------------------

    /// Returns a reference to the executor registry for custom executor
    /// registration.
    pub fn executor_registry_mut(&mut self) -> &mut ExecutorRegistry {
        &mut self.executor_registry
    }

    /// Returns the current state provider.
    pub fn state_provider(&self) -> &dyn StateProvider {
        self.state_provider.as_ref()
    }

    /// Replaces the state provider.
    pub fn set_state_provider(&mut self, provider: std::sync::Arc<dyn StateProvider>) {
        self.state_provider = provider;
    }

    /// Returns a reference to the safety checker.
    pub fn safety_checker(&self) -> &SafetyChecker {
        &self.safety_checker
    }

    /// Returns a mutable reference to the safety checker for policy updates.
    pub fn safety_checker_mut(&mut self) -> &mut SafetyChecker {
        &mut self.safety_checker
    }

    /// Returns a reference to the message router for multi-agent coordination.
    pub fn message_router(&self) -> &MessageRouter {
        &self.message_router
    }

    /// Returns a mutable reference to the message router.
    pub fn message_router_mut(&mut self) -> &mut MessageRouter {
        &mut self.message_router
    }

    /// Returns a reference to the audit trail.
    pub fn audit_trail(&self) -> &AuditTrail {
        &self.audit_trail
    }

    /// Returns a mutable reference to the audit trail.
    pub fn audit_trail_mut(&mut self) -> &mut AuditTrail {
        &mut self.audit_trail
    }

    /// Returns a reference to the intent queue.
    pub fn intent_queue(&self) -> &IntentQueue {
        &self.intent_queue
    }

    /// Take all pending blocked-intent notifications.
    pub fn take_blocked_notifications(&mut self) -> Vec<BlockedIntentNotification> {
        self.intent_queue.take_notifications()
    }

    /// Returns the conflict log from the intent queue.
    pub fn conflict_log(&self) -> &[Conflict] {
        self.intent_queue.conflict_log()
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
    let counter = INTENT_COUNTER.fetch_add(1, Ordering::Relaxed);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("intent-{}-{}-{}", now.as_secs(), now.subsec_nanos(), counter)
}

// ---------------------------------------------------------------------------
// Async message types (unchanged public API)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Tests -- original tests preserved + new integration tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // === Original tests (unchanged) ===

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
        let intent1 =
            Intent::new(AgentId::new("test-agent"), IntentType::Query).with_priority(3);
        let intent2 =
            Intent::new(AgentId::new("test-agent"), IntentType::Query).with_priority(8);

        coordinator.submit_intent(intent1).unwrap();
        coordinator.submit_intent(intent2).unwrap();

        let results = coordinator.process_intents();
        assert_eq!(results.len(), 2);
    }

    // === New integration tests ===

    #[test]
    fn test_real_execution_query() {
        let state = std::sync::Arc::new(InMemoryStateProvider::new());
        state.insert("cell/cell-1/load", "0.75");

        let mut coordinator = AgentCoordinator::with_config(
            state,
            ResolutionStrategy::PriorityBased,
            SafetyPolicy::default(),
            1000,
        );

        coordinator.register_agent(
            AgentId::new("q-agent"),
            AgentType::Custom,
            AgentCapabilities {
                read_state: true,
                ..Default::default()
            },
        );

        let intent = Intent::new(AgentId::new("q-agent"), IntentType::Query)
            .with_target("cell/cell-1")
            .with_param("keys", "load");

        coordinator.submit_intent(intent).unwrap();
        let results = coordinator.process_intents_full();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].status, IntentStatus::Success);
        assert_eq!(
            results[0].output.get("cell/cell-1/load").map(String::as_str),
            Some("0.75")
        );
    }

    #[test]
    fn test_real_execution_handover() {
        let state = std::sync::Arc::new(InMemoryStateProvider::new());
        state.insert("ue/ue-1/serving_cell", "cell-A");
        state.insert("cell/cell-B", "active");
        state.insert("ue/ue-1/rsrp/cell/cell-B", "-80.0");

        let mut coordinator = AgentCoordinator::with_config(
            state,
            ResolutionStrategy::PriorityBased,
            SafetyPolicy::default(),
            1000,
        );

        coordinator.register_agent(
            AgentId::new("mob-agent"),
            AgentType::Mobility,
            AgentCapabilities {
                read_state: true,
                trigger_actions: true,
                ..Default::default()
            },
        );

        let intent = Intent::new(AgentId::new("mob-agent"), IntentType::TriggerHandover)
            .with_target("ue-1");

        coordinator.submit_intent(intent).unwrap();
        let results = coordinator.process_intents_full();
        assert_eq!(results[0].status, IntentStatus::Success);
        assert_eq!(
            results[0].output.get("target_cell").map(String::as_str),
            Some("cell/cell-B")
        );
    }

    #[test]
    fn test_conflict_resolution_blocks_lower_priority() {
        let mut coordinator = AgentCoordinator::new();

        // Register two agents that both want to handover the same UE.
        for name in &["agent-high", "agent-low"] {
            coordinator.register_agent(
                AgentId::new(*name),
                AgentType::Mobility,
                AgentCapabilities {
                    read_state: true,
                    trigger_actions: true,
                    ..Default::default()
                },
            );
        }

        let i1 = Intent::new(AgentId::new("agent-high"), IntentType::TriggerHandover)
            .with_target("ue-99")
            .with_priority(9);
        let i2 = Intent::new(AgentId::new("agent-low"), IntentType::TriggerHandover)
            .with_target("ue-99")
            .with_priority(2);

        coordinator.submit_intent(i1).unwrap();
        let result = coordinator.submit_intent(i2);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("blocked"));
    }

    #[test]
    fn test_safety_blocks_excessive_qos_change() {
        let mut coordinator = AgentCoordinator::new();

        coordinator.register_agent(
            AgentId::new("qos-agent"),
            AgentType::Qos,
            AgentCapabilities {
                read_state: true,
                trigger_actions: true,
                ..Default::default()
            },
        );

        let intent = Intent::new(AgentId::new("qos-agent"), IntentType::AdjustQos)
            .with_target("flow-1")
            .with_param("mbr_change_pct", "200"); // 200% exceeds default 50% limit

        let result = coordinator.submit_intent(intent);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Safety violation"));
    }

    #[test]
    fn test_audit_trail_records_executions() {
        let mut coordinator = AgentCoordinator::new();

        coordinator.register_agent(
            AgentId::new("a1"),
            AgentType::Custom,
            AgentCapabilities {
                read_state: true,
                ..Default::default()
            },
        );

        let intent =
            Intent::new(AgentId::new("a1"), IntentType::Query).with_target("anything");
        coordinator.submit_intent(intent).unwrap();
        coordinator.process_intents();

        // 1 registration + 1 execution = 2 entries.
        assert_eq!(coordinator.audit_trail().len(), 2);
    }

    #[test]
    fn test_multi_agent_messaging() {
        let mut coordinator = AgentCoordinator::new();

        coordinator.register_agent_with_role(
            AgentId::new("net-ctrl"),
            AgentType::Custom,
            AgentCapabilities::default(),
            AgentRole::NetworkLevel,
            Some("global".to_string()),
        );
        coordinator.register_agent_with_role(
            AgentId::new("cell-1"),
            AgentType::Custom,
            AgentCapabilities::default(),
            AgentRole::CellLevel,
            Some("cell-1".to_string()),
        );

        // Send a message from network controller to cell agent.
        let msg_id = coordinator
            .message_router_mut()
            .send(
                &AgentId::new("net-ctrl"),
                &AgentId::new("cell-1"),
                MessagePayload::Request {
                    action: "report_load".to_string(),
                    parameters: HashMap::new(),
                },
                None,
            )
            .unwrap();
        assert!(msg_id.starts_with("msg-"));

        let msgs = coordinator
            .message_router_mut()
            .receive(&AgentId::new("cell-1"));
        assert_eq!(msgs.len(), 1);
    }

    #[test]
    fn test_backward_compatible_intent_result() {
        // Ensure IntentResult still has the same shape as before.
        let result = IntentResult {
            intent_id: "test".to_string(),
            success: true,
            data: HashMap::new(),
            error: None,
        };
        assert!(result.success);
        assert!(result.error.is_none());
    }

    #[test]
    fn test_oauth2_client_creation() {
        let client = OAuth2Client::new(
            "https://idp.example.com/oauth2/authorize",
            "https://idp.example.com/oauth2/token",
            "client-id-123",
            "client-secret-456",
        );

        assert_eq!(client._auth_endpoint, "https://idp.example.com/oauth2/authorize");
        assert_eq!(client.client_id, "client-id-123");
    }

    #[test]
    fn test_oauth2_client_credentials_flow() {
        let client = OAuth2Client::new(
            "https://idp.example.com/oauth2/authorize",
            "https://idp.example.com/oauth2/token",
            "client-id-123",
            "client-secret-456",
        );

        let agent_id = AgentId::new("test-agent");
        let scopes = vec!["read".to_string(), "write".to_string()];

        let result = client.request_token_client_credentials(&agent_id, scopes);
        assert!(result.is_ok());

        let token = result.unwrap();
        assert!(token.is_valid());
        assert!(token.has_scope("read"));
        assert!(token.has_scope("write"));
    }

    #[test]
    fn test_oauth2_validate_token() {
        let client = OAuth2Client::new(
            "https://idp.example.com/oauth2/authorize",
            "https://idp.example.com/oauth2/token",
            "client-id-123",
            "client-secret-456",
        );

        let valid_token = "oauth2_client-id-123_some-token-id";
        assert!(client.validate_token(valid_token).unwrap());

        let invalid_token = "invalid-token";
        assert!(!client.validate_token(invalid_token).unwrap());
    }

    #[test]
    fn test_oauth2_refresh_token() {
        let client = OAuth2Client::new(
            "https://idp.example.com/oauth2/authorize",
            "https://idp.example.com/oauth2/token",
            "client-id-123",
            "client-secret-456",
        );

        let agent_id = AgentId::new("test-agent");
        let refresh_token = "refresh_token_xyz";

        let result = client.refresh_token(&agent_id, refresh_token);
        assert!(result.is_ok());

        let token = result.unwrap();
        assert!(token.is_valid());
        assert!(token.token.contains("refreshed"));
    }

    #[test]
    fn test_oauth2_token_cache() {
        let client = OAuth2Client::new(
            "https://idp.example.com/oauth2/authorize",
            "https://idp.example.com/oauth2/token",
            "client-id-123",
            "client-secret-456",
        );

        let agent_id = AgentId::new("test-agent");
        let scopes = vec!["read".to_string()];

        // Request token - should be cached
        client.request_token_client_credentials(&agent_id, scopes).unwrap();

        // Retrieve from cache
        let cached = client.get_cached_token(&agent_id);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().token_type, "Bearer");

        // Clear cache
        client.clear_cache();
        let cached_after_clear = client.get_cached_token(&agent_id);
        assert!(cached_after_clear.is_none());
    }

    #[test]
    fn test_oauth2_revoke_token() {
        let client = OAuth2Client::new(
            "https://idp.example.com/oauth2/authorize",
            "https://idp.example.com/oauth2/token",
            "client-id-123",
            "client-secret-456",
        );

        let valid_token = "oauth2_client-id-123_token";
        assert!(client.revoke_token(valid_token).is_ok());

        let invalid_token = "invalid-token";
        assert!(client.revoke_token(invalid_token).is_err());
    }

    #[test]
    fn test_create_slice_via_coordinator() {
        let state = std::sync::Arc::new(InMemoryStateProvider::new());
        let mut coordinator = AgentCoordinator::with_config(
            state.clone(),
            ResolutionStrategy::PriorityBased,
            SafetyPolicy::default(),
            1000,
        );

        coordinator.register_agent(
            AgentId::new("slice-agent"),
            AgentType::Slicing,
            AgentCapabilities {
                read_state: true,
                modify_config: true,
                ..Default::default()
            },
        );

        let intent = Intent::new(AgentId::new("slice-agent"), IntentType::CreateSlice)
            .with_param("slice_id", "s1")
            .with_param("sst", "1")
            .with_param("max_ues", "500");

        coordinator.submit_intent(intent).unwrap();
        let results = coordinator.process_intents_full();
        assert_eq!(results[0].status, IntentStatus::Success);
        assert_eq!(state.get("slice/s1/sst"), Some("1".to_string()));
        assert_eq!(state.get("slice/s1/max_ues"), Some("500".to_string()));
    }

    #[test]
    fn test_composite_intent_through_coordinator() {
        let mut coordinator = AgentCoordinator::new();

        coordinator.register_agent_with_role(
            AgentId::new("region-ctrl"),
            AgentType::Resource,
            AgentCapabilities {
                read_state: true,
                modify_config: true,
                ..Default::default()
            },
            AgentRole::RegionLevel,
            Some("region-east".to_string()),
        );
        coordinator.register_agent_with_role(
            AgentId::new("cell-a"),
            AgentType::Resource,
            AgentCapabilities {
                read_state: true,
                modify_config: true,
                ..Default::default()
            },
            AgentRole::CellLevel,
            Some("cell-a".to_string()),
        );

        // Create composite through the message router.
        let cid = coordinator
            .message_router_mut()
            .create_composite(&AgentId::new("region-ctrl"), "Optimize region-east".to_string())
            .unwrap();

        let sub = Intent::new(AgentId::new("cell-a"), IntentType::OptimizeResources)
            .with_target("cell-a");

        coordinator
            .message_router_mut()
            .submit_sub_intent(&cid, &AgentId::new("cell-a"), sub)
            .unwrap();

        coordinator
            .message_router_mut()
            .decide_sub_intent(&cid, 0, true)
            .unwrap();

        let accepted = coordinator
            .message_router_mut()
            .finalize_composite(&cid)
            .unwrap();
        assert_eq!(accepted.len(), 1);
    }
}
