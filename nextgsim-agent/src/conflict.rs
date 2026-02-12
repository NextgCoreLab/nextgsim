//! Conflict detection and resolution for competing intents
//!
//! When multiple agents submit intents that affect the same resources, this
//! module detects the overlap and applies a configurable resolution strategy.

use crate::execution::ResourceKind;
use crate::{AgentId, Intent, IntentType};
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};

// ---------------------------------------------------------------------------
// Conflict types
// ---------------------------------------------------------------------------

/// A detected conflict between two intents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conflict {
    /// ID of the first intent.
    pub intent_a: String,
    /// ID of the second intent.
    pub intent_b: String,
    /// Resources in contention.
    pub contested_resources: Vec<ContestedResource>,
    /// How the conflict was resolved (filled after resolution).
    pub resolution: Option<ConflictResolutionOutcome>,
}

/// A resource that two intents both want to write.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContestedResource {
    /// Resource kind.
    pub kind: ResourceKind,
    /// Resource identifier.
    pub id: String,
}

/// Outcome of conflict resolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictResolutionOutcome {
    /// Intent A wins, intent B is blocked.
    WinnerA,
    /// Intent B wins, intent A is blocked.
    WinnerB,
    /// Both intents were merged into a combined execution.
    Merged,
    /// Both intents are blocked (irreconcilable).
    BothBlocked,
}

// ---------------------------------------------------------------------------
// Conflict detector
// ---------------------------------------------------------------------------

/// Detects resource conflicts between intents.
#[derive(Debug, Default)]
pub struct ConflictDetector;

impl ConflictDetector {
    /// Creates a new detector.
    pub fn new() -> Self {
        Self
    }

    /// Compute the set of resources an intent *would* write.
    ///
    /// This is a static analysis based on intent type and target -- it does not
    /// actually execute the intent.
    pub fn projected_write_resources(intent: &Intent) -> Vec<ContestedResource> {
        let mut resources = Vec::new();

        match &intent.intent_type {
            IntentType::Query => {
                // Queries are read-only, no write conflict.
            }
            IntentType::OptimizeResources => {
                // Affects cells in scope.
                if let Some(ref target) = intent.target {
                    resources.push(ContestedResource {
                        kind: ResourceKind::Cell,
                        id: target.clone(),
                    });
                } else {
                    // Global scope -- mark with a sentinel.
                    resources.push(ContestedResource {
                        kind: ResourceKind::Cell,
                        id: "*".to_string(),
                    });
                }
            }
            IntentType::TriggerHandover => {
                if let Some(ref target) = intent.target {
                    resources.push(ContestedResource {
                        kind: ResourceKind::Ue,
                        id: target.clone(),
                    });
                }
                if let Some(tc) = intent.parameters.get("target_cell") {
                    resources.push(ContestedResource {
                        kind: ResourceKind::Cell,
                        id: tc.clone(),
                    });
                }
            }
            IntentType::AdjustQos => {
                if let Some(ref target) = intent.target {
                    resources.push(ContestedResource {
                        kind: ResourceKind::QosFlow,
                        id: target.clone(),
                    });
                }
            }
            IntentType::CreateSlice => {
                let slice_id = intent
                    .parameters
                    .get("slice_id")
                    .cloned()
                    .unwrap_or_else(|| format!("slice-{}", &intent.id[..8.min(intent.id.len())]));
                resources.push(ContestedResource {
                    kind: ResourceKind::Slice,
                    id: slice_id,
                });
            }
            IntentType::ModifySlice => {
                if let Some(ref target) = intent.target {
                    resources.push(ContestedResource {
                        kind: ResourceKind::Slice,
                        id: target.clone(),
                    });
                }
            }
            IntentType::Custom(_) => {
                // Conservative: if a custom intent has a target, consider it a write.
                if let Some(ref target) = intent.target {
                    resources.push(ContestedResource {
                        kind: ResourceKind::Other("custom".to_string()),
                        id: target.clone(),
                    });
                }
            }
        }
        resources
    }

    /// Check whether two intents conflict (both write to at least one common
    /// resource).
    pub fn check_conflict(a: &Intent, b: &Intent) -> Option<Conflict> {
        let writes_a = Self::projected_write_resources(a);
        let writes_b = Self::projected_write_resources(b);

        // Build a set for quick lookup.
        let set_a: HashSet<(String, String)> = writes_a
            .iter()
            .map(|r| (format!("{:?}", r.kind), r.id.clone()))
            .collect();

        let mut contested = Vec::new();
        for rb in &writes_b {
            let key = (format!("{:?}", rb.kind), rb.id.clone());
            // Wildcard match: if either side uses "*", it conflicts with any
            // resource of the same kind.
            let matches_wildcard = writes_a.iter().any(|ra| {
                format!("{:?}", ra.kind) == key.0 && (ra.id == "*" || rb.id == "*")
            });
            if set_a.contains(&key) || matches_wildcard {
                contested.push(ContestedResource {
                    kind: rb.kind.clone(),
                    id: rb.id.clone(),
                });
            }
        }

        if contested.is_empty() {
            None
        } else {
            Some(Conflict {
                intent_a: a.id.clone(),
                intent_b: b.id.clone(),
                contested_resources: contested,
                resolution: None,
            })
        }
    }

    /// Check a new intent against all intents already in the queue.
    pub fn check_against_queue(
        new_intent: &Intent,
        queue: &[Intent],
    ) -> Vec<(usize, Conflict)> {
        queue
            .iter()
            .enumerate()
            .filter_map(|(idx, existing)| {
                Self::check_conflict(new_intent, existing).map(|c| (idx, c))
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Conflict resolution strategies
// ---------------------------------------------------------------------------

/// Strategy used to resolve conflicts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResolutionStrategy {
    /// Higher priority intent wins; ties broken by timestamp (first wins).
    PriorityBased,
    /// First-submitted intent wins regardless of priority.
    TimeBased,
    /// Attempt to merge non-conflicting parts of both intents.
    Merge,
}

/// Resolves conflicts between intents.
#[derive(Debug)]
pub struct ConflictResolver {
    strategy: ResolutionStrategy,
}

impl ConflictResolver {
    /// Creates a resolver with the given strategy.
    pub fn new(strategy: ResolutionStrategy) -> Self {
        Self { strategy }
    }

    /// Returns the active strategy.
    pub fn strategy(&self) -> ResolutionStrategy {
        self.strategy
    }

    /// Resolve a conflict between two intents.
    ///
    /// Returns the outcome indicating which intent(s) should proceed.
    pub fn resolve(&self, a: &Intent, b: &Intent, conflict: &mut Conflict) -> ConflictResolutionOutcome {
        let outcome = match self.strategy {
            ResolutionStrategy::PriorityBased => self.resolve_priority(a, b),
            ResolutionStrategy::TimeBased => self.resolve_time(a, b),
            ResolutionStrategy::Merge => self.resolve_merge(a, b, conflict),
        };
        conflict.resolution = Some(outcome.clone());
        outcome
    }

    fn resolve_priority(&self, a: &Intent, b: &Intent) -> ConflictResolutionOutcome {
        if a.priority > b.priority {
            ConflictResolutionOutcome::WinnerA
        } else if b.priority > a.priority {
            ConflictResolutionOutcome::WinnerB
        } else {
            // Tie-break: earlier timestamp wins.
            if a.timestamp_ms <= b.timestamp_ms {
                ConflictResolutionOutcome::WinnerA
            } else {
                ConflictResolutionOutcome::WinnerB
            }
        }
    }

    fn resolve_time(&self, a: &Intent, b: &Intent) -> ConflictResolutionOutcome {
        if a.timestamp_ms <= b.timestamp_ms {
            ConflictResolutionOutcome::WinnerA
        } else {
            ConflictResolutionOutcome::WinnerB
        }
    }

    fn resolve_merge(
        &self,
        a: &Intent,
        b: &Intent,
        _conflict: &Conflict,
    ) -> ConflictResolutionOutcome {
        // Merging is only possible when the contested resources don't have
        // contradictory parameter changes.  For now we support merging when:
        // - Both intents are of the same type
        // - Their parameter keys don't overlap on the contested resources
        if a.intent_type != b.intent_type {
            // Fall back to priority-based for different types.
            return self.resolve_priority(a, b);
        }

        // Check for parameter key overlap.
        let keys_a: HashSet<&String> = a.parameters.keys().collect();
        let keys_b: HashSet<&String> = b.parameters.keys().collect();
        let overlap: HashSet<&&String> = keys_a.intersection(&keys_b).collect();

        if overlap.is_empty() {
            // No overlapping parameters -- can merge.
            ConflictResolutionOutcome::Merged
        } else {
            // Check if overlapping keys have the same values.
            let all_same = overlap.iter().all(|k| a.parameters.get(**k) == b.parameters.get(**k));
            if all_same {
                ConflictResolutionOutcome::Merged
            } else {
                // Cannot merge, fall back to priority.
                self.resolve_priority(a, b)
            }
        }
    }
}

impl Default for ConflictResolver {
    fn default() -> Self {
        Self::new(ResolutionStrategy::PriorityBased)
    }
}

// ---------------------------------------------------------------------------
// Intent queue with conflict checking
// ---------------------------------------------------------------------------

/// Notification sent back to an agent whose intent was blocked.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockedIntentNotification {
    /// The blocked intent ID.
    pub intent_id: String,
    /// The agent that submitted the blocked intent.
    pub agent_id: AgentId,
    /// The winning intent ID.
    pub winner_intent_id: String,
    /// Contested resources.
    pub contested_resources: Vec<ContestedResource>,
    /// Reason string.
    pub reason: String,
}

/// A queue that validates intents against conflicts before allowing execution.
#[derive(Debug)]
pub struct IntentQueue {
    /// Intents ready for execution (passed conflict checks).
    ready: VecDeque<Intent>,
    /// Blocked intents awaiting notification delivery.
    blocked_notifications: Vec<BlockedIntentNotification>,
    /// Conflict resolver.
    resolver: ConflictResolver,
    /// All detected conflicts (for audit purposes).
    conflict_log: Vec<Conflict>,
}

impl IntentQueue {
    /// Creates a new intent queue with the given resolution strategy.
    pub fn new(strategy: ResolutionStrategy) -> Self {
        Self {
            ready: VecDeque::new(),
            blocked_notifications: Vec::new(),
            resolver: ConflictResolver::new(strategy),
            conflict_log: Vec::new(),
        }
    }

    /// Enqueue an intent.  Returns `Ok(())` if accepted or details of the
    /// blocking conflict.
    pub fn enqueue(&mut self, intent: Intent) -> Result<(), BlockedIntentNotification> {
        let ready_slice: Vec<Intent> = self.ready.iter().cloned().collect();
        let conflicts = ConflictDetector::check_against_queue(&intent, &ready_slice);

        if conflicts.is_empty() {
            self.ready.push_back(intent);
            return Ok(());
        }

        // Resolve each conflict. If the new intent loses any, it is blocked.
        for (idx, mut conflict) in conflicts {
            let existing = &ready_slice[idx];
            let outcome = self.resolver.resolve(&intent, existing, &mut conflict);
            self.conflict_log.push(conflict.clone());

            match outcome {
                ConflictResolutionOutcome::WinnerB | ConflictResolutionOutcome::BothBlocked => {
                    // New intent loses.
                    let notification = BlockedIntentNotification {
                        intent_id: intent.id.clone(),
                        agent_id: intent.agent_id.clone(),
                        winner_intent_id: existing.id.clone(),
                        contested_resources: conflict.contested_resources,
                        reason: format!(
                            "Blocked by existing intent {} (strategy: {:?})",
                            existing.id,
                            self.resolver.strategy()
                        ),
                    };
                    self.blocked_notifications.push(notification.clone());
                    return Err(notification);
                }
                ConflictResolutionOutcome::WinnerA => {
                    // New intent wins -- remove the existing intent from the queue.
                    // We need to find and remove it from the actual deque.
                    let existing_id = existing.id.clone();
                    self.ready.retain(|i| i.id != existing_id);

                    let notification = BlockedIntentNotification {
                        intent_id: existing_id.clone(),
                        agent_id: existing.agent_id.clone(),
                        winner_intent_id: intent.id.clone(),
                        contested_resources: conflict.contested_resources,
                        reason: format!(
                            "Superseded by intent {} (strategy: {:?})",
                            intent.id,
                            self.resolver.strategy()
                        ),
                    };
                    self.blocked_notifications.push(notification);
                }
                ConflictResolutionOutcome::Merged => {
                    // Both can proceed (merge strategy).  The existing stays,
                    // and we add the new one too.
                }
            }
        }

        self.ready.push_back(intent);
        Ok(())
    }

    /// Drain all ready intents for execution.
    pub fn drain_ready(&mut self) -> Vec<Intent> {
        self.ready.drain(..).collect()
    }

    /// Take all pending blocked notifications.
    pub fn take_notifications(&mut self) -> Vec<BlockedIntentNotification> {
        std::mem::take(&mut self.blocked_notifications)
    }

    /// Returns a reference to the conflict log.
    pub fn conflict_log(&self) -> &[Conflict] {
        &self.conflict_log
    }

    /// Number of intents ready for execution.
    pub fn ready_count(&self) -> usize {
        self.ready.len()
    }

    /// Returns true if there are no ready intents.
    pub fn is_empty(&self) -> bool {
        self.ready.is_empty()
    }
}

impl Default for IntentQueue {
    fn default() -> Self {
        Self::new(ResolutionStrategy::PriorityBased)
    }
}

// ---------------------------------------------------------------------------
// Persistent intent store
// ---------------------------------------------------------------------------

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

/// Persistent storage for intents
///
/// Provides disk-backed storage for intent history and recovery from crashes.
pub struct PersistentIntentStore {
    /// Storage directory
    storage_dir: PathBuf,
    /// In-memory cache of stored intents
    cache: HashMap<String, Intent>,
    /// Whether to auto-persist on changes
    auto_persist: bool,
}

impl PersistentIntentStore {
    /// Creates a new persistent intent store
    pub fn new(storage_dir: impl Into<PathBuf>) -> Result<Self, std::io::Error> {
        let storage_dir = storage_dir.into();

        // Create storage directory if it doesn't exist
        if !storage_dir.exists() {
            fs::create_dir_all(&storage_dir)?;
        }

        let mut store = Self {
            storage_dir,
            cache: HashMap::new(),
            auto_persist: true,
        };

        // Load existing intents from disk
        store.load_from_disk()?;

        Ok(store)
    }

    /// Sets auto-persist behavior
    pub fn set_auto_persist(&mut self, enabled: bool) {
        self.auto_persist = enabled;
    }

    /// Stores an intent
    pub fn store(&mut self, intent: Intent) -> Result<(), std::io::Error> {
        let intent_id = intent.id.clone();
        self.cache.insert(intent_id.clone(), intent);

        if self.auto_persist {
            self.persist_intent(&intent_id)?;
        }

        Ok(())
    }

    /// Retrieves an intent by ID
    pub fn get(&self, intent_id: &str) -> Option<&Intent> {
        self.cache.get(intent_id)
    }

    /// Removes an intent
    pub fn remove(&mut self, intent_id: &str) -> Result<Option<Intent>, std::io::Error> {
        let intent = self.cache.remove(intent_id);

        if intent.is_some() {
            self.delete_intent_file(intent_id)?;
        }

        Ok(intent)
    }

    /// Lists all stored intent IDs
    pub fn list_ids(&self) -> Vec<String> {
        self.cache.keys().cloned().collect()
    }

    /// Returns all stored intents
    pub fn all(&self) -> Vec<&Intent> {
        self.cache.values().collect()
    }

    /// Returns the number of stored intents
    pub fn count(&self) -> usize {
        self.cache.len()
    }

    /// Clears all intents from memory and disk
    pub fn clear(&mut self) -> Result<(), std::io::Error> {
        // Remove all files
        for intent_id in self.cache.keys() {
            self.delete_intent_file(intent_id)?;
        }

        self.cache.clear();
        Ok(())
    }

    /// Manually persists all cached intents to disk
    pub fn persist_all(&self) -> Result<(), std::io::Error> {
        for intent_id in self.cache.keys() {
            self.persist_intent(intent_id)?;
        }
        Ok(())
    }

    /// Loads intents from disk into cache
    fn load_from_disk(&mut self) -> Result<(), std::io::Error> {
        if !self.storage_dir.exists() {
            return Ok(());
        }

        for entry in fs::read_dir(&self.storage_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                match self.load_intent_file(&path) {
                    Ok(intent) => {
                        self.cache.insert(intent.id.clone(), intent);
                    }
                    Err(e) => {
                        eprintln!("Failed to load intent from {path:?}: {e}");
                    }
                }
            }
        }

        Ok(())
    }

    /// Persists a single intent to disk
    fn persist_intent(&self, intent_id: &str) -> Result<(), std::io::Error> {
        if let Some(intent) = self.cache.get(intent_id) {
            let file_path = self.intent_file_path(intent_id);
            let file = File::create(file_path)?;
            let writer = BufWriter::new(file);
            serde_json::to_writer_pretty(writer, intent)
                .map_err(std::io::Error::other)?;
        }
        Ok(())
    }

    /// Loads a single intent from a file
    fn load_intent_file(&self, path: &Path) -> Result<Intent, std::io::Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader)
            .map_err(std::io::Error::other)
    }

    /// Deletes an intent file from disk
    fn delete_intent_file(&self, intent_id: &str) -> Result<(), std::io::Error> {
        let file_path = self.intent_file_path(intent_id);
        if file_path.exists() {
            fs::remove_file(file_path)?;
        }
        Ok(())
    }

    /// Returns the file path for an intent
    fn intent_file_path(&self, intent_id: &str) -> PathBuf {
        // Sanitize intent ID for use in filename
        let safe_id = intent_id.replace(['/', '\\'], "_");
        self.storage_dir.join(format!("{safe_id}.json"))
    }

    /// Returns the storage directory path
    pub fn storage_dir(&self) -> &Path {
        &self.storage_dir
    }
}

/// Builder for creating a persistent intent store
pub struct PersistentIntentStoreBuilder {
    storage_dir: PathBuf,
    auto_persist: bool,
}

impl PersistentIntentStoreBuilder {
    /// Creates a new builder
    pub fn new(storage_dir: impl Into<PathBuf>) -> Self {
        Self {
            storage_dir: storage_dir.into(),
            auto_persist: true,
        }
    }

    /// Sets whether to automatically persist intents on changes
    pub fn auto_persist(mut self, enabled: bool) -> Self {
        self.auto_persist = enabled;
        self
    }

    /// Builds the persistent intent store
    pub fn build(self) -> Result<PersistentIntentStore, std::io::Error> {
        let mut store = PersistentIntentStore::new(self.storage_dir)?;
        store.set_auto_persist(self.auto_persist);
        Ok(store)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_intent(agent: &str, itype: IntentType, target: Option<&str>, priority: u8) -> Intent {
        let mut i = Intent::new(AgentId::new(agent), itype);
        if let Some(t) = target {
            i = i.with_target(t);
        }
        i = i.with_priority(priority);
        i
    }

    #[test]
    fn test_no_conflict_between_queries() {
        let a = make_intent("a1", IntentType::Query, Some("cell-1"), 5);
        let b = make_intent("a2", IntentType::Query, Some("cell-1"), 5);
        assert!(ConflictDetector::check_conflict(&a, &b).is_none());
    }

    #[test]
    fn test_conflict_same_ue_handover() {
        let a = make_intent("a1", IntentType::TriggerHandover, Some("ue-42"), 5);
        let b = make_intent("a2", IntentType::TriggerHandover, Some("ue-42"), 7);
        let conflict = ConflictDetector::check_conflict(&a, &b);
        assert!(conflict.is_some());
        let c = conflict.unwrap();
        assert_eq!(c.contested_resources.len(), 1);
        assert_eq!(c.contested_resources[0].id, "ue-42");
    }

    #[test]
    fn test_no_conflict_different_ues() {
        let a = make_intent("a1", IntentType::TriggerHandover, Some("ue-1"), 5);
        let b = make_intent("a2", IntentType::TriggerHandover, Some("ue-2"), 5);
        assert!(ConflictDetector::check_conflict(&a, &b).is_none());
    }

    #[test]
    fn test_conflict_wildcard_resource_optimization() {
        let a = make_intent("a1", IntentType::OptimizeResources, None, 5);
        let b = make_intent("a2", IntentType::OptimizeResources, Some("region-east"), 5);
        let conflict = ConflictDetector::check_conflict(&a, &b);
        assert!(conflict.is_some());
    }

    #[test]
    fn test_priority_resolver_higher_wins() {
        let a = make_intent("a1", IntentType::TriggerHandover, Some("ue-1"), 3);
        let b = make_intent("a2", IntentType::TriggerHandover, Some("ue-1"), 8);
        let mut conflict = ConflictDetector::check_conflict(&a, &b).unwrap();
        let resolver = ConflictResolver::new(ResolutionStrategy::PriorityBased);
        let outcome = resolver.resolve(&a, &b, &mut conflict);
        assert!(matches!(outcome, ConflictResolutionOutcome::WinnerB));
    }

    #[test]
    fn test_time_resolver_first_wins() {
        let mut a = make_intent("a1", IntentType::AdjustQos, Some("flow-1"), 3);
        a.timestamp_ms = 1000;
        let mut b = make_intent("a2", IntentType::AdjustQos, Some("flow-1"), 8);
        b.timestamp_ms = 2000;
        let mut conflict = ConflictDetector::check_conflict(&a, &b).unwrap();
        let resolver = ConflictResolver::new(ResolutionStrategy::TimeBased);
        let outcome = resolver.resolve(&a, &b, &mut conflict);
        assert!(matches!(outcome, ConflictResolutionOutcome::WinnerA));
    }

    #[test]
    fn test_merge_resolver_non_overlapping_params() {
        let mut a = make_intent("a1", IntentType::AdjustQos, Some("flow-1"), 5);
        a.parameters.insert("mbr_change_pct".to_string(), "10".to_string());
        let mut b = make_intent("a2", IntentType::AdjustQos, Some("flow-1"), 5);
        b.parameters.insert("latency_change_pct".to_string(), "-5".to_string());
        let mut conflict = ConflictDetector::check_conflict(&a, &b).unwrap();
        let resolver = ConflictResolver::new(ResolutionStrategy::Merge);
        let outcome = resolver.resolve(&a, &b, &mut conflict);
        assert!(matches!(outcome, ConflictResolutionOutcome::Merged));
    }

    #[test]
    fn test_intent_queue_no_conflict() {
        let mut queue = IntentQueue::new(ResolutionStrategy::PriorityBased);
        let a = make_intent("a1", IntentType::TriggerHandover, Some("ue-1"), 5);
        let b = make_intent("a2", IntentType::TriggerHandover, Some("ue-2"), 5);
        assert!(queue.enqueue(a).is_ok());
        assert!(queue.enqueue(b).is_ok());
        assert_eq!(queue.ready_count(), 2);
    }

    #[test]
    fn test_intent_queue_conflict_blocks_loser() {
        let mut queue = IntentQueue::new(ResolutionStrategy::PriorityBased);
        let a = make_intent("a1", IntentType::TriggerHandover, Some("ue-1"), 8);
        let b = make_intent("a2", IntentType::TriggerHandover, Some("ue-1"), 3);
        assert!(queue.enqueue(a).is_ok());
        let result = queue.enqueue(b);
        assert!(result.is_err());
        let notification = result.unwrap_err();
        assert_eq!(notification.agent_id, AgentId::new("a2"));
        assert_eq!(queue.ready_count(), 1);
    }

    #[test]
    fn test_intent_queue_higher_priority_evicts() {
        let mut queue = IntentQueue::new(ResolutionStrategy::PriorityBased);
        let a = make_intent("a1", IntentType::TriggerHandover, Some("ue-1"), 3);
        let b = make_intent("a2", IntentType::TriggerHandover, Some("ue-1"), 8);
        assert!(queue.enqueue(a).is_ok());
        assert!(queue.enqueue(b).is_ok());
        // a should have been evicted; only b remains.
        assert_eq!(queue.ready_count(), 1);
        let notifications = queue.take_notifications();
        assert_eq!(notifications.len(), 1);
        assert_eq!(notifications[0].agent_id, AgentId::new("a1"));
    }

    #[test]
    fn test_persistent_store_creation() {
        let temp_dir = std::env::temp_dir().join("nextgsim_test_store_1");
        let _ = std::fs::remove_dir_all(&temp_dir); // Clean up any previous test

        let store = PersistentIntentStore::new(&temp_dir);
        assert!(store.is_ok());

        let store = store.unwrap();
        assert_eq!(store.count(), 0);
        assert!(temp_dir.exists());

        // Cleanup
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_persistent_store_operations() {
        let temp_dir = std::env::temp_dir().join("nextgsim_test_store_2");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let mut store = PersistentIntentStore::new(&temp_dir).unwrap();

        // Store an intent
        let intent = make_intent("agent-1", IntentType::Query, Some("cell-1"), 5);
        let intent_id = intent.id.clone();
        assert!(store.store(intent).is_ok());
        assert_eq!(store.count(), 1);

        // Retrieve the intent
        let retrieved = store.get(&intent_id);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().agent_id, AgentId::new("agent-1"));

        // List IDs
        let ids = store.list_ids();
        assert_eq!(ids.len(), 1);
        assert!(ids.contains(&intent_id));

        // Remove the intent
        let removed = store.remove(&intent_id).unwrap();
        assert!(removed.is_some());
        assert_eq!(store.count(), 0);

        // Cleanup
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_persistent_store_persistence() {
        let temp_dir = std::env::temp_dir().join("nextgsim_test_store_3");
        let _ = std::fs::remove_dir_all(&temp_dir);

        // Create store and add intents
        {
            let mut store = PersistentIntentStore::new(&temp_dir).unwrap();
            let intent1 = make_intent("agent-1", IntentType::Query, Some("cell-1"), 5);
            let intent2 = make_intent("agent-2", IntentType::Query, Some("cell-2"), 5);

            store.store(intent1).unwrap();
            store.store(intent2).unwrap();
            assert_eq!(store.count(), 2);
        }

        // Recreate store - should load from disk
        {
            let store = PersistentIntentStore::new(&temp_dir).unwrap();
            assert_eq!(store.count(), 2);
        }

        // Cleanup
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_persistent_store_clear() {
        let temp_dir = std::env::temp_dir().join("nextgsim_test_store_4");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let mut store = PersistentIntentStore::new(&temp_dir).unwrap();

        // Add some intents
        for i in 0..5 {
            let intent = make_intent(&format!("agent-{i}"), IntentType::Query, None, 5);
            store.store(intent).unwrap();
        }
        assert_eq!(store.count(), 5);

        // Clear all
        assert!(store.clear().is_ok());
        assert_eq!(store.count(), 0);

        // Verify files are gone
        let entries: Vec<_> = std::fs::read_dir(&temp_dir)
            .unwrap()
            .filter_map(std::result::Result::ok)
            .collect();
        assert_eq!(entries.len(), 0);

        // Cleanup
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_persistent_store_builder() {
        let temp_dir = std::env::temp_dir().join("nextgsim_test_store_5");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let store = PersistentIntentStoreBuilder::new(&temp_dir)
            .auto_persist(false)
            .build();

        assert!(store.is_ok());

        // Cleanup
        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
