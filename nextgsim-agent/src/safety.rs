//! Safety constraints and guardrails for agent intent execution
//!
//! Provides a `SafetyPolicy` that validates intents before execution to ensure
//! agents cannot perform dangerous operations (e.g., disabling all cells at
//! once, exceeding resource change limits, etc.).

use crate::execution::{AffectedResource, ResourceKind};
use crate::{AgentId, Intent, IntentType};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Safety policy
// ---------------------------------------------------------------------------

/// Bounds on what an agent can do in a single intent execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyPolicy {
    /// Maximum number of cells an intent can affect at once.
    pub max_cells_affected: u32,
    /// Maximum number of UEs an intent can impact at once.
    pub max_ues_impacted: u32,
    /// Maximum allowed `QoS` change percentage (absolute value).
    /// E.g., 50 means a single intent cannot change any `QoS` metric by more
    /// than 50% up or down.
    pub max_qos_change_pct: f64,
    /// Maximum number of slices that can be created per agent per minute.
    pub max_slice_creations_per_minute: u32,
    /// Forbidden actions.  Each entry is a rule checked before execution.
    pub forbidden_actions: Vec<ForbiddenAction>,
    /// Agent-specific overrides (`agent_id` -> policy override).
    /// If an agent has an override, those fields replace the defaults.
    pub agent_overrides: std::collections::HashMap<String, SafetyPolicyOverride>,
}

/// A rule that blocks specific actions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForbiddenAction {
    /// Human-readable description.
    pub description: String,
    /// The rule definition.
    pub rule: ForbiddenRule,
}

/// Concrete forbidden rules.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ForbiddenRule {
    /// Cannot disable (set status to inactive) all cells simultaneously.
    DisableAllCells,
    /// Cannot modify a specific resource.
    ResourceLocked {
        /// Resource kind.
        kind: ResourceKind,
        /// Resource id.
        id: String,
    },
    /// A specific intent type is forbidden for the given agent type.
    IntentTypeForbidden {
        /// The intent type name.
        intent_type: String,
    },
    /// Maximum number of concurrent handovers.
    MaxConcurrentHandovers {
        /// The ceiling.
        limit: u32,
    },
}

/// Per-agent overrides for safety limits.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SafetyPolicyOverride {
    /// Override `max_cells_affected` (if Some).
    pub max_cells_affected: Option<u32>,
    /// Override `max_ues_impacted` (if Some).
    pub max_ues_impacted: Option<u32>,
    /// Override `max_qos_change_pct` (if Some).
    pub max_qos_change_pct: Option<f64>,
}

impl Default for SafetyPolicy {
    fn default() -> Self {
        Self {
            max_cells_affected: 10,
            max_ues_impacted: 100,
            max_qos_change_pct: 50.0,
            max_slice_creations_per_minute: 5,
            forbidden_actions: vec![ForbiddenAction {
                description: "Cannot disable all cells simultaneously".to_string(),
                rule: ForbiddenRule::DisableAllCells,
            }],
            agent_overrides: std::collections::HashMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Safety violation
// ---------------------------------------------------------------------------

/// A safety check failure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyViolation {
    /// Which policy was violated.
    pub rule_description: String,
    /// The offending intent ID.
    pub intent_id: String,
    /// The agent that submitted the intent.
    pub agent_id: AgentId,
    /// Timestamp in milliseconds since epoch.
    pub timestamp_ms: u64,
    /// Severity level.
    pub severity: ViolationSeverity,
}

/// How severe the violation is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationSeverity {
    /// Informational -- intent can still proceed with caution.
    Warning,
    /// Intent must be blocked.
    Block,
    /// Critical -- agent should be flagged for review.
    Critical,
}

// ---------------------------------------------------------------------------
// Safety checker
// ---------------------------------------------------------------------------

/// Pre-execution safety checker.
#[derive(Debug)]
pub struct SafetyChecker {
    policy: SafetyPolicy,
    /// Log of all violations detected.
    violation_log: Vec<SafetyViolation>,
    /// Tracks recent slice creations per agent: (`agent_id`, `creation_timestamp_ms`).
    recent_slice_creations: Vec<(String, u64)>,
    /// Active handover count for `MaxConcurrentHandovers` rule.
    active_handovers: u32,
}

impl SafetyChecker {
    /// Creates a new checker with the given policy.
    pub fn new(policy: SafetyPolicy) -> Self {
        Self {
            policy,
            violation_log: Vec::new(),
            recent_slice_creations: Vec::new(),
            active_handovers: 0,
        }
    }

    /// Returns a reference to the current policy.
    pub fn policy(&self) -> &SafetyPolicy {
        &self.policy
    }

    /// Updates the policy at runtime.
    pub fn set_policy(&mut self, policy: SafetyPolicy) {
        self.policy = policy;
    }

    /// Notify the checker that a handover completed (decrement active count).
    pub fn handover_completed(&mut self) {
        self.active_handovers = self.active_handovers.saturating_sub(1);
    }

    /// Get all recorded violations.
    pub fn violations(&self) -> &[SafetyViolation] {
        &self.violation_log
    }

    /// Validate an intent against the safety policy.
    ///
    /// Returns `Ok(())` if the intent is safe, or `Err(violations)` listing
    /// all triggered rules.
    pub fn validate(
        &mut self,
        intent: &Intent,
        projected_affected: &[AffectedResource],
    ) -> Result<(), Vec<SafetyViolation>> {
        let mut violations = Vec::new();
        let now_ms = crate::timestamp_now();

        // Effective limits (apply per-agent overrides).
        let agent_key = intent.agent_id.0.clone();
        let ovr = self.policy.agent_overrides.get(&agent_key);
        let max_cells = ovr
            .and_then(|o| o.max_cells_affected)
            .unwrap_or(self.policy.max_cells_affected);
        let max_ues = ovr
            .and_then(|o| o.max_ues_impacted)
            .unwrap_or(self.policy.max_ues_impacted);
        let max_qos_pct = ovr
            .and_then(|o| o.max_qos_change_pct)
            .unwrap_or(self.policy.max_qos_change_pct);

        // 1. Count affected cells and UEs.
        let cells_affected = projected_affected
            .iter()
            .filter(|r| r.kind == ResourceKind::Cell)
            .count() as u32;
        let ues_impacted = projected_affected
            .iter()
            .filter(|r| r.kind == ResourceKind::Ue)
            .count() as u32;

        if cells_affected > max_cells {
            violations.push(SafetyViolation {
                rule_description: format!(
                    "Intent affects {cells_affected} cells, limit is {max_cells}"
                ),
                intent_id: intent.id.clone(),
                agent_id: intent.agent_id.clone(),
                timestamp_ms: now_ms,
                severity: ViolationSeverity::Block,
            });
        }

        if ues_impacted > max_ues {
            violations.push(SafetyViolation {
                rule_description: format!(
                    "Intent impacts {ues_impacted} UEs, limit is {max_ues}"
                ),
                intent_id: intent.id.clone(),
                agent_id: intent.agent_id.clone(),
                timestamp_ms: now_ms,
                severity: ViolationSeverity::Block,
            });
        }

        // 2. QoS change bounds.
        if intent.intent_type == IntentType::AdjustQos {
            for param_key in &["mbr_change_pct", "latency_change_pct"] {
                if let Some(val_str) = intent.parameters.get(*param_key) {
                    if let Ok(val) = val_str.parse::<f64>() {
                        if val.abs() > max_qos_pct {
                            violations.push(SafetyViolation {
                                rule_description: format!(
                                    "QoS change {param_key}={val:.1}% exceeds limit of {max_qos_pct:.1}%"
                                ),
                                intent_id: intent.id.clone(),
                                agent_id: intent.agent_id.clone(),
                                timestamp_ms: now_ms,
                                severity: ViolationSeverity::Block,
                            });
                        }
                    }
                }
            }
        }

        // 3. Slice creation rate limit.
        if intent.intent_type == IntentType::CreateSlice {
            // Clean old entries (older than 60 seconds).
            let cutoff = now_ms.saturating_sub(60_000);
            self.recent_slice_creations.retain(|(_, ts)| *ts >= cutoff);

            let agent_count = self
                .recent_slice_creations
                .iter()
                .filter(|(aid, _)| *aid == intent.agent_id.0)
                .count() as u32;

            if agent_count >= self.policy.max_slice_creations_per_minute {
                violations.push(SafetyViolation {
                    rule_description: format!(
                        "Agent {} exceeded slice creation rate limit ({}/min)",
                        intent.agent_id, self.policy.max_slice_creations_per_minute
                    ),
                    intent_id: intent.id.clone(),
                    agent_id: intent.agent_id.clone(),
                    timestamp_ms: now_ms,
                    severity: ViolationSeverity::Block,
                });
            } else {
                // Track this creation.
                self.recent_slice_creations
                    .push((intent.agent_id.0.clone(), now_ms));
            }
        }

        // 4. Forbidden actions.
        for fa in &self.policy.forbidden_actions {
            if let Some(v) = self.check_forbidden_rule(&fa.rule, intent, projected_affected) {
                violations.push(SafetyViolation {
                    rule_description: fa.description.clone(),
                    intent_id: intent.id.clone(),
                    agent_id: intent.agent_id.clone(),
                    timestamp_ms: now_ms,
                    severity: v,
                });
            }
        }

        // 5. Track handovers -- only if no violations so the intent will proceed.
        if violations.is_empty() && intent.intent_type == IntentType::TriggerHandover {
            self.active_handovers += 1;
        }

        // Record all violations.
        self.violation_log.extend(violations.clone());

        if violations.is_empty() {
            Ok(())
        } else {
            Err(violations)
        }
    }

    fn check_forbidden_rule(
        &self,
        rule: &ForbiddenRule,
        intent: &Intent,
        projected_affected: &[AffectedResource],
    ) -> Option<ViolationSeverity> {
        match rule {
            ForbiddenRule::DisableAllCells => {
                // Check if the intent would set status to inactive on all cells
                // (wildcard target with deactivate).
                if intent.target.is_none()
                    && intent.parameters.get("status").map(std::string::String::as_str) == Some("inactive")
                {
                    return Some(ViolationSeverity::Critical);
                }
                // Also block if projected affected cells include wildcard.
                let has_wildcard_cell = projected_affected.iter().any(|r| {
                    r.kind == ResourceKind::Cell && r.id == "*"
                });
                if has_wildcard_cell
                    && intent.parameters.get("status").map(std::string::String::as_str) == Some("inactive")
                {
                    return Some(ViolationSeverity::Critical);
                }
                None
            }
            ForbiddenRule::ResourceLocked { kind, id } => {
                let touches_locked = projected_affected
                    .iter()
                    .any(|r| r.kind == *kind && r.id == *id);
                if touches_locked {
                    Some(ViolationSeverity::Block)
                } else {
                    None
                }
            }
            ForbiddenRule::IntentTypeForbidden { intent_type } => {
                let type_str = match &intent.intent_type {
                    IntentType::Query => "Query",
                    IntentType::OptimizeResources => "OptimizeResources",
                    IntentType::TriggerHandover => "TriggerHandover",
                    IntentType::AdjustQos => "AdjustQos",
                    IntentType::CreateSlice => "CreateSlice",
                    IntentType::ModifySlice => "ModifySlice",
                    IntentType::Custom(n) => n.as_str(),
                };
                if type_str == intent_type {
                    Some(ViolationSeverity::Block)
                } else {
                    None
                }
            }
            ForbiddenRule::MaxConcurrentHandovers { limit } => {
                if intent.intent_type == IntentType::TriggerHandover
                    && self.active_handovers >= *limit
                {
                    Some(ViolationSeverity::Block)
                } else {
                    None
                }
            }
        }
    }
}

impl Default for SafetyChecker {
    fn default() -> Self {
        Self::new(SafetyPolicy::default())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::ResourceAccess;

    fn make_intent(itype: IntentType) -> Intent {
        Intent::new(AgentId::new("test-agent"), itype)
    }

    #[test]
    fn test_default_policy_allows_small_intent() {
        let mut checker = SafetyChecker::default();
        let intent = make_intent(IntentType::Query);
        let affected = vec![AffectedResource {
            kind: ResourceKind::Cell,
            id: "cell-1".to_string(),
            access: ResourceAccess::Read,
        }];
        assert!(checker.validate(&intent, &affected).is_ok());
    }

    #[test]
    fn test_too_many_cells() {
        let mut policy = SafetyPolicy::default();
        policy.max_cells_affected = 2;
        let mut checker = SafetyChecker::new(policy);

        let intent = make_intent(IntentType::OptimizeResources);
        let affected: Vec<AffectedResource> = (0..5)
            .map(|i| AffectedResource {
                kind: ResourceKind::Cell,
                id: format!("cell-{i}"),
                access: ResourceAccess::Write,
            })
            .collect();

        let result = checker.validate(&intent, &affected);
        assert!(result.is_err());
        let violations = result.unwrap_err();
        assert_eq!(violations.len(), 1);
        assert!(matches!(violations[0].severity, ViolationSeverity::Block));
    }

    #[test]
    fn test_qos_change_too_large() {
        let mut checker = SafetyChecker::default(); // max_qos_change_pct = 50
        let intent = make_intent(IntentType::AdjustQos)
            .with_target("flow-1")
            .with_param("mbr_change_pct", "80");

        let result = checker.validate(&intent, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_qos_change_within_limit() {
        let mut checker = SafetyChecker::default();
        let intent = make_intent(IntentType::AdjustQos)
            .with_target("flow-1")
            .with_param("mbr_change_pct", "30");

        assert!(checker.validate(&intent, &[]).is_ok());
    }

    #[test]
    fn test_forbidden_disable_all_cells() {
        let mut checker = SafetyChecker::default();
        let mut intent = make_intent(IntentType::OptimizeResources);
        intent.parameters.insert("status".to_string(), "inactive".to_string());
        // No target = global scope.

        let affected = vec![AffectedResource {
            kind: ResourceKind::Cell,
            id: "*".to_string(),
            access: ResourceAccess::Write,
        }];

        let result = checker.validate(&intent, &affected);
        assert!(result.is_err());
        let violations = result.unwrap_err();
        assert!(violations.iter().any(|v| v.severity == ViolationSeverity::Critical));
    }

    #[test]
    fn test_resource_locked_rule() {
        let mut policy = SafetyPolicy::default();
        policy.forbidden_actions.push(ForbiddenAction {
            description: "cell-critical is locked".to_string(),
            rule: ForbiddenRule::ResourceLocked {
                kind: ResourceKind::Cell,
                id: "cell-critical".to_string(),
            },
        });
        let mut checker = SafetyChecker::new(policy);

        let intent = make_intent(IntentType::OptimizeResources).with_target("cell-critical");
        let affected = vec![AffectedResource {
            kind: ResourceKind::Cell,
            id: "cell-critical".to_string(),
            access: ResourceAccess::Write,
        }];

        assert!(checker.validate(&intent, &affected).is_err());
    }

    #[test]
    fn test_max_concurrent_handovers() {
        let mut policy = SafetyPolicy::default();
        policy.forbidden_actions.push(ForbiddenAction {
            description: "Max 2 concurrent handovers".to_string(),
            rule: ForbiddenRule::MaxConcurrentHandovers { limit: 2 },
        });
        let mut checker = SafetyChecker::new(policy);

        let i1 = make_intent(IntentType::TriggerHandover).with_target("ue-1");
        let i2 = make_intent(IntentType::TriggerHandover).with_target("ue-2");
        let i3 = make_intent(IntentType::TriggerHandover).with_target("ue-3");

        assert!(checker.validate(&i1, &[]).is_ok());
        assert!(checker.validate(&i2, &[]).is_ok());
        // Third should be blocked.
        assert!(checker.validate(&i3, &[]).is_err());

        // Complete one handover, then third should succeed.
        checker.handover_completed();
        let i3b = make_intent(IntentType::TriggerHandover).with_target("ue-3");
        assert!(checker.validate(&i3b, &[]).is_ok());
    }

    #[test]
    fn test_agent_override() {
        let mut policy = SafetyPolicy::default();
        policy.max_cells_affected = 2;
        policy.agent_overrides.insert(
            "privileged-agent".to_string(),
            SafetyPolicyOverride {
                max_cells_affected: Some(100),
                ..Default::default()
            },
        );
        let mut checker = SafetyChecker::new(policy);

        let affected: Vec<AffectedResource> = (0..5)
            .map(|i| AffectedResource {
                kind: ResourceKind::Cell,
                id: format!("cell-{i}"),
                access: ResourceAccess::Write,
            })
            .collect();

        // Normal agent should fail.
        let normal_intent = Intent::new(AgentId::new("normal-agent"), IntentType::OptimizeResources);
        assert!(checker.validate(&normal_intent, &affected).is_err());

        // Privileged agent should pass.
        let priv_intent = Intent::new(AgentId::new("privileged-agent"), IntentType::OptimizeResources);
        assert!(checker.validate(&priv_intent, &affected).is_ok());
    }

    #[test]
    fn test_violation_log_persists() {
        let mut checker = SafetyChecker::default();
        let intent = make_intent(IntentType::AdjustQos)
            .with_target("f1")
            .with_param("mbr_change_pct", "999");
        let _ = checker.validate(&intent, &[]);
        assert!(!checker.violations().is_empty());
    }
}
