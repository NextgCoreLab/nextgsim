//! Decision audit trail
//!
//! Records every intent execution, conflict resolution, and safety violation
//! in a bounded circular buffer with a query interface for post-hoc analysis.

use crate::execution::{AffectedResource, IntentExecutionResult, IntentStatus};
use crate::{AgentId, Intent, IntentType};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Audit entry
// ---------------------------------------------------------------------------

/// A single entry in the decision audit trail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Monotonically increasing sequence number.
    pub seq: u64,
    /// Timestamp in milliseconds since epoch.
    pub timestamp_ms: u64,
    /// The agent that initiated the action.
    pub agent_id: AgentId,
    /// Kind of audited event.
    pub event: AuditEvent,
}

/// The different kinds of events that can be audited.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditEvent {
    /// An intent was executed.
    IntentExecuted {
        /// The intent that was executed.
        intent_id: String,
        /// Intent type.
        intent_type: IntentType,
        /// Target resource.
        target: Option<String>,
        /// Execution result status.
        status: IntentStatus,
        /// Resources that were affected.
        affected_resources: Vec<AffectedResource>,
        /// Execution time in microseconds.
        execution_time_us: u64,
    },
    /// An intent was blocked by conflict resolution.
    IntentBlocked {
        /// The blocked intent ID.
        intent_id: String,
        /// The winning intent ID.
        winner_intent_id: String,
        /// Reason.
        reason: String,
    },
    /// A safety violation was detected.
    SafetyViolation {
        /// The offending intent ID.
        intent_id: String,
        /// Rule that was violated.
        rule: String,
        /// Severity.
        severity: String,
    },
    /// An agent was registered.
    AgentRegistered {
        /// Agent type.
        agent_type: String,
    },
    /// An agent was deregistered.
    AgentDeregistered,
    /// A composite intent was created.
    CompositeCreated {
        /// Composite intent ID.
        composite_id: String,
        /// Description.
        description: String,
    },
    /// A composite intent completed.
    CompositeCompleted {
        /// Composite intent ID.
        composite_id: String,
        /// Number of sub-intents executed.
        sub_intent_count: u32,
    },
}

// ---------------------------------------------------------------------------
// Audit trail (circular buffer)
// ---------------------------------------------------------------------------

/// Bounded circular buffer that stores the most recent audit entries.
#[derive(Debug)]
pub struct AuditTrail {
    /// Maximum number of entries to retain.
    capacity: usize,
    /// The circular buffer of entries.
    entries: VecDeque<AuditEntry>,
    /// Next sequence number.
    next_seq: u64,
}

impl AuditTrail {
    /// Creates a new audit trail with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            entries: VecDeque::with_capacity(capacity),
            next_seq: 1,
        }
    }

    /// Returns the capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of entries currently stored.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if no entries are stored.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Record an intent execution result.
    pub fn record_execution(&mut self, agent_id: &AgentId, intent: &Intent, result: &IntentExecutionResult) {
        let seq = self.next_seq();
        self.push(AuditEntry {
            seq,
            timestamp_ms: crate::timestamp_now(),
            agent_id: agent_id.clone(),
            event: AuditEvent::IntentExecuted {
                intent_id: intent.id.clone(),
                intent_type: intent.intent_type.clone(),
                target: intent.target.clone(),
                status: result.status.clone(),
                affected_resources: result.affected_resources.clone(),
                execution_time_us: result.execution_time_us,
            },
        });
    }

    /// Record a blocked intent.
    pub fn record_blocked(
        &mut self,
        agent_id: &AgentId,
        intent_id: &str,
        winner_id: &str,
        reason: &str,
    ) {
        let seq = self.next_seq();
        self.push(AuditEntry {
            seq,
            timestamp_ms: crate::timestamp_now(),
            agent_id: agent_id.clone(),
            event: AuditEvent::IntentBlocked {
                intent_id: intent_id.to_string(),
                winner_intent_id: winner_id.to_string(),
                reason: reason.to_string(),
            },
        });
    }

    /// Record a safety violation.
    pub fn record_safety_violation(
        &mut self,
        agent_id: &AgentId,
        intent_id: &str,
        rule: &str,
        severity: &str,
    ) {
        let seq = self.next_seq();
        self.push(AuditEntry {
            seq,
            timestamp_ms: crate::timestamp_now(),
            agent_id: agent_id.clone(),
            event: AuditEvent::SafetyViolation {
                intent_id: intent_id.to_string(),
                rule: rule.to_string(),
                severity: severity.to_string(),
            },
        });
    }

    /// Record an agent registration.
    pub fn record_agent_registered(&mut self, agent_id: &AgentId, agent_type: &str) {
        let seq = self.next_seq();
        self.push(AuditEntry {
            seq,
            timestamp_ms: crate::timestamp_now(),
            agent_id: agent_id.clone(),
            event: AuditEvent::AgentRegistered {
                agent_type: agent_type.to_string(),
            },
        });
    }

    /// Record a generic audit entry.
    pub fn record(&mut self, entry: AuditEntry) {
        self.push(entry);
    }

    // -----------------------------------------------------------------------
    // Query interface
    // -----------------------------------------------------------------------

    /// Get all entries (most recent last).
    pub fn all_entries(&self) -> impl Iterator<Item = &AuditEntry> {
        self.entries.iter()
    }

    /// Query entries by agent ID.
    pub fn by_agent(&self, agent_id: &AgentId) -> Vec<&AuditEntry> {
        self.entries.iter().filter(|e| e.agent_id == *agent_id).collect()
    }

    /// Query entries within a time range (inclusive).
    pub fn by_time_range(&self, from_ms: u64, to_ms: u64) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| e.timestamp_ms >= from_ms && e.timestamp_ms <= to_ms)
            .collect()
    }

    /// Query entries that affected a specific resource ID.
    pub fn by_resource(&self, resource_id: &str) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| match &e.event {
                AuditEvent::IntentExecuted {
                    affected_resources, ..
                } => affected_resources.iter().any(|r| r.id == resource_id),
                _ => false,
            })
            .collect()
    }

    /// Query entries by event type predicate.
    pub fn by_event_filter<F>(&self, predicate: F) -> Vec<&AuditEntry>
    where
        F: Fn(&AuditEvent) -> bool,
    {
        self.entries
            .iter()
            .filter(|e| predicate(&e.event))
            .collect()
    }

    /// Get the N most recent entries.
    pub fn recent(&self, n: usize) -> Vec<&AuditEntry> {
        let skip = self.entries.len().saturating_sub(n);
        self.entries.iter().skip(skip).collect()
    }

    /// Count entries matching a predicate.
    pub fn count_where<F>(&self, predicate: F) -> usize
    where
        F: Fn(&AuditEntry) -> bool,
    {
        self.entries.iter().filter(|e| predicate(e)).count()
    }

    /// Count safety violations.
    pub fn safety_violation_count(&self) -> usize {
        self.count_where(|e| matches!(e.event, AuditEvent::SafetyViolation { .. }))
    }

    /// Count blocked intents.
    pub fn blocked_intent_count(&self) -> usize {
        self.count_where(|e| matches!(e.event, AuditEvent::IntentBlocked { .. }))
    }

    // -----------------------------------------------------------------------
    // Internal
    // -----------------------------------------------------------------------

    fn push(&mut self, entry: AuditEntry) {
        if self.entries.len() >= self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
    }

    fn next_seq(&mut self) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;
        seq
    }
}

impl Default for AuditTrail {
    fn default() -> Self {
        Self::new(10_000)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{ResourceAccess, ResourceKind};

    fn sample_execution_result(intent_id: &str) -> IntentExecutionResult {
        IntentExecutionResult {
            intent_id: intent_id.to_string(),
            status: IntentStatus::Success,
            output: Default::default(),
            affected_resources: vec![AffectedResource {
                kind: ResourceKind::Cell,
                id: "cell-1".to_string(),
                access: ResourceAccess::Write,
            }],
            message: None,
            execution_time_us: 500,
        }
    }

    #[test]
    fn test_circular_buffer_eviction() {
        let mut trail = AuditTrail::new(3);
        for i in 0..5 {
            trail.record_agent_registered(&AgentId::new(format!("a{i}")), "Custom");
        }
        assert_eq!(trail.len(), 3);
        // Oldest entries (a0, a1) should be evicted; a2, a3, a4 remain.
        let entries: Vec<_> = trail.all_entries().collect();
        assert_eq!(entries[0].agent_id, AgentId::new("a2"));
        assert_eq!(entries[2].agent_id, AgentId::new("a4"));
    }

    #[test]
    fn test_record_execution() {
        let mut trail = AuditTrail::new(100);
        let agent = AgentId::new("test");
        let intent = Intent::new(agent.clone(), IntentType::Query).with_target("cell-1");
        let result = sample_execution_result(&intent.id);

        trail.record_execution(&agent, &intent, &result);
        assert_eq!(trail.len(), 1);
    }

    #[test]
    fn test_query_by_agent() {
        let mut trail = AuditTrail::new(100);
        trail.record_agent_registered(&AgentId::new("a1"), "Mobility");
        trail.record_agent_registered(&AgentId::new("a2"), "Resource");
        trail.record_agent_registered(&AgentId::new("a1"), "Mobility");

        let a1_entries = trail.by_agent(&AgentId::new("a1"));
        assert_eq!(a1_entries.len(), 2);
    }

    #[test]
    fn test_query_by_resource() {
        let mut trail = AuditTrail::new(100);
        let agent = AgentId::new("a1");
        let intent = Intent::new(agent.clone(), IntentType::OptimizeResources);
        let result = sample_execution_result(&intent.id);

        trail.record_execution(&agent, &intent, &result);

        let cell1_entries = trail.by_resource("cell-1");
        assert_eq!(cell1_entries.len(), 1);

        let cell99_entries = trail.by_resource("cell-99");
        assert_eq!(cell99_entries.len(), 0);
    }

    #[test]
    fn test_query_recent() {
        let mut trail = AuditTrail::new(100);
        for i in 0..10 {
            trail.record_agent_registered(&AgentId::new(format!("a{i}")), "Custom");
        }

        let recent = trail.recent(3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].agent_id, AgentId::new("a7"));
        assert_eq!(recent[2].agent_id, AgentId::new("a9"));
    }

    #[test]
    fn test_safety_violation_count() {
        let mut trail = AuditTrail::new(100);
        trail.record_safety_violation(&AgentId::new("a1"), "i1", "too many cells", "Block");
        trail.record_safety_violation(&AgentId::new("a1"), "i2", "qos too high", "Block");
        trail.record_agent_registered(&AgentId::new("a1"), "Custom");

        assert_eq!(trail.safety_violation_count(), 2);
        assert_eq!(trail.blocked_intent_count(), 0);
    }

    #[test]
    fn test_blocked_intent_recording() {
        let mut trail = AuditTrail::new(100);
        trail.record_blocked(&AgentId::new("a1"), "intent-1", "intent-2", "priority");
        assert_eq!(trail.blocked_intent_count(), 1);
    }

    #[test]
    fn test_sequence_numbers_monotonic() {
        let mut trail = AuditTrail::new(100);
        for _ in 0..5 {
            trail.record_agent_registered(&AgentId::new("a1"), "Custom");
        }
        let entries: Vec<_> = trail.all_entries().collect();
        for w in entries.windows(2) {
            assert!(w[1].seq > w[0].seq);
        }
    }

    #[test]
    fn test_event_filter() {
        let mut trail = AuditTrail::new(100);
        trail.record_agent_registered(&AgentId::new("a1"), "Mobility");
        trail.record_safety_violation(&AgentId::new("a1"), "i1", "rule", "Block");
        trail.record_agent_registered(&AgentId::new("a2"), "Resource");

        let registrations = trail.by_event_filter(|e| matches!(e, AuditEvent::AgentRegistered { .. }));
        assert_eq!(registrations.len(), 2);
    }
}
