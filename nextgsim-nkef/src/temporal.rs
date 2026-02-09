//! Temporal knowledge graph extensions
//!
//! Provides temporal properties for relationships and entity attributes,
//! enabling time-windowed queries and history tracking.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Timestamp in milliseconds since Unix epoch.
pub type Timestamp = u64;

/// Returns the current timestamp in milliseconds since Unix epoch.
pub fn now_millis() -> Timestamp {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// A temporal relationship with validity window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalRelationship {
    /// Source entity ID
    pub source_id: String,
    /// Target entity ID
    pub target_id: String,
    /// Relationship type
    pub relation_type: String,
    /// Properties
    pub properties: HashMap<String, String>,
    /// Start of validity window (inclusive)
    pub valid_from: Timestamp,
    /// End of validity window (exclusive). `None` means still valid.
    pub valid_to: Option<Timestamp>,
}

impl TemporalRelationship {
    /// Creates a new temporal relationship that starts now and has no end.
    pub fn new(
        source_id: impl Into<String>,
        target_id: impl Into<String>,
        relation_type: impl Into<String>,
    ) -> Self {
        Self {
            source_id: source_id.into(),
            target_id: target_id.into(),
            relation_type: relation_type.into(),
            properties: HashMap::new(),
            valid_from: now_millis(),
            valid_to: None,
        }
    }

    /// Creates a new temporal relationship with explicit timestamps.
    pub fn with_validity(
        source_id: impl Into<String>,
        target_id: impl Into<String>,
        relation_type: impl Into<String>,
        valid_from: Timestamp,
        valid_to: Option<Timestamp>,
    ) -> Self {
        Self {
            source_id: source_id.into(),
            target_id: target_id.into(),
            relation_type: relation_type.into(),
            properties: HashMap::new(),
            valid_from,
            valid_to,
        }
    }

    /// Adds a property to this relationship.
    pub fn with_property(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }

    /// Returns true if this relationship is valid at the given timestamp.
    pub fn is_valid_at(&self, timestamp: Timestamp) -> bool {
        timestamp >= self.valid_from
            && self.valid_to.is_none_or(|end| timestamp < end)
    }

    /// Returns true if this relationship is currently valid (no end time or end is in the future).
    pub fn is_active(&self) -> bool {
        self.valid_to.is_none_or(|end| now_millis() < end)
    }

    /// Ends this relationship at the given timestamp.
    pub fn expire_at(&mut self, timestamp: Timestamp) {
        self.valid_to = Some(timestamp);
    }

    /// Ends this relationship now.
    pub fn expire_now(&mut self) {
        self.valid_to = Some(now_millis());
    }
}

/// A single attribute change record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributeChange {
    /// The attribute key that changed
    pub key: String,
    /// Previous value (None if newly added)
    pub old_value: Option<String>,
    /// New value (None if removed)
    pub new_value: Option<String>,
    /// When the change occurred
    pub timestamp: Timestamp,
}

/// Tracks the history of attribute changes for an entity.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EntityHistory {
    /// Entity ID
    pub entity_id: String,
    /// Ordered list of attribute changes (oldest first)
    pub changes: Vec<AttributeChange>,
}

impl EntityHistory {
    /// Creates a new empty history for the given entity.
    pub fn new(entity_id: impl Into<String>) -> Self {
        Self {
            entity_id: entity_id.into(),
            changes: Vec::new(),
        }
    }

    /// Records an attribute change.
    pub fn record_change(
        &mut self,
        key: impl Into<String>,
        old_value: Option<String>,
        new_value: Option<String>,
    ) {
        self.changes.push(AttributeChange {
            key: key.into(),
            old_value,
            new_value,
            timestamp: now_millis(),
        });
    }

    /// Records an attribute change at a specific timestamp.
    pub fn record_change_at(
        &mut self,
        key: impl Into<String>,
        old_value: Option<String>,
        new_value: Option<String>,
        timestamp: Timestamp,
    ) {
        self.changes.push(AttributeChange {
            key: key.into(),
            old_value,
            new_value,
            timestamp,
        });
    }

    /// Returns changes within a time window.
    pub fn changes_in_range(&self, from: Timestamp, to: Timestamp) -> Vec<&AttributeChange> {
        self.changes
            .iter()
            .filter(|c| c.timestamp >= from && c.timestamp < to)
            .collect()
    }

    /// Returns all changes for a specific attribute key.
    pub fn changes_for_key(&self, key: &str) -> Vec<&AttributeChange> {
        self.changes.iter().filter(|c| c.key == key).collect()
    }

    /// Reconstructs the entity's properties at a given point in time.
    ///
    /// Starts from an empty state and replays all changes up to and including
    /// the given timestamp.
    pub fn state_at(&self, timestamp: Timestamp) -> HashMap<String, String> {
        let mut state: HashMap<String, String> = HashMap::new();
        for change in &self.changes {
            if change.timestamp > timestamp {
                break;
            }
            match &change.new_value {
                Some(val) => {
                    state.insert(change.key.clone(), val.clone());
                }
                None => {
                    state.remove(&change.key);
                }
            }
        }
        state
    }

    /// Returns the total number of recorded changes.
    pub fn change_count(&self) -> usize {
        self.changes.len()
    }
}

/// Temporal store for relationships with validity windows.
///
/// Wraps a collection of `TemporalRelationship` instances and provides
/// time-windowed query methods.
#[derive(Debug, Clone, Default)]
pub struct TemporalRelationshipStore {
    /// All relationships (including expired ones)
    relationships: Vec<TemporalRelationship>,
}

impl TemporalRelationshipStore {
    /// Creates a new empty temporal relationship store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a temporal relationship to the store.
    pub fn add(&mut self, relationship: TemporalRelationship) {
        self.relationships.push(relationship);
    }

    /// Returns the total number of relationships (including expired).
    pub fn total_count(&self) -> usize {
        self.relationships.len()
    }

    /// Returns the number of currently active relationships.
    pub fn active_count(&self) -> usize {
        self.relationships.iter().filter(|r| r.is_active()).count()
    }

    /// Returns all relationships that are valid at the given timestamp.
    pub fn at_time(&self, timestamp: Timestamp) -> Vec<&TemporalRelationship> {
        self.relationships
            .iter()
            .filter(|r| r.is_valid_at(timestamp))
            .collect()
    }

    /// Returns all currently active relationships.
    pub fn active(&self) -> Vec<&TemporalRelationship> {
        self.relationships
            .iter()
            .filter(|r| r.is_active())
            .collect()
    }

    /// Returns all relationships (including expired) for a given entity.
    pub fn for_entity(&self, entity_id: &str) -> Vec<&TemporalRelationship> {
        self.relationships
            .iter()
            .filter(|r| r.source_id == entity_id || r.target_id == entity_id)
            .collect()
    }

    /// Returns relationships for a given entity that are valid at the given timestamp.
    pub fn for_entity_at_time(
        &self,
        entity_id: &str,
        timestamp: Timestamp,
    ) -> Vec<&TemporalRelationship> {
        self.relationships
            .iter()
            .filter(|r| {
                (r.source_id == entity_id || r.target_id == entity_id)
                    && r.is_valid_at(timestamp)
            })
            .collect()
    }

    /// Returns currently active relationships for a given entity.
    pub fn active_for_entity(&self, entity_id: &str) -> Vec<&TemporalRelationship> {
        self.relationships
            .iter()
            .filter(|r| {
                (r.source_id == entity_id || r.target_id == entity_id) && r.is_active()
            })
            .collect()
    }

    /// Expires all active relationships involving the given entity.
    pub fn expire_entity_relationships(&mut self, entity_id: &str) {
        let now = now_millis();
        for rel in &mut self.relationships {
            if (rel.source_id == entity_id || rel.target_id == entity_id) && rel.is_active() {
                rel.expire_at(now);
            }
        }
    }

    /// Removes all expired relationships older than the given timestamp.
    ///
    /// This is useful for garbage collection of old history.
    pub fn prune_before(&mut self, timestamp: Timestamp) {
        self.relationships.retain(|r| {
            // Keep if still active or if it ended after the cutoff
            r.valid_to.is_none_or(|end| end >= timestamp)
        });
    }

    /// Returns relationships within a time window for a specific entity.
    pub fn for_entity_in_range(
        &self,
        entity_id: &str,
        from: Timestamp,
        to: Timestamp,
    ) -> Vec<&TemporalRelationship> {
        self.relationships
            .iter()
            .filter(|r| {
                (r.source_id == entity_id || r.target_id == entity_id)
                    && r.valid_from < to
                    && r.valid_to.is_none_or(|end| end > from)
            })
            .collect()
    }
}

/// Combined temporal store for entity histories.
#[derive(Debug, Clone, Default)]
pub struct EntityHistoryStore {
    /// Entity histories keyed by entity ID
    histories: HashMap<String, EntityHistory>,
}

impl EntityHistoryStore {
    /// Creates a new empty entity history store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Gets or creates the history for an entity.
    pub fn get_or_create(&mut self, entity_id: &str) -> &mut EntityHistory {
        self.histories
            .entry(entity_id.to_string())
            .or_insert_with(|| EntityHistory::new(entity_id))
    }

    /// Gets the history for an entity, if it exists.
    pub fn get(&self, entity_id: &str) -> Option<&EntityHistory> {
        self.histories.get(entity_id)
    }

    /// Removes the history for an entity.
    pub fn remove(&mut self, entity_id: &str) -> Option<EntityHistory> {
        self.histories.remove(entity_id)
    }

    /// Records an attribute change for an entity, computing the diff automatically.
    ///
    /// `old_properties` is the entity's properties before the change,
    /// `new_properties` is the entity's properties after the change.
    pub fn record_entity_update(
        &mut self,
        entity_id: &str,
        old_properties: &HashMap<String, String>,
        new_properties: &HashMap<String, String>,
    ) {
        let history = self.get_or_create(entity_id);

        // Find changed and added properties
        for (key, new_val) in new_properties {
            let old_val = old_properties.get(key);
            if old_val.map(|v| v != new_val).unwrap_or(true) {
                history.record_change(
                    key.clone(),
                    old_val.cloned(),
                    Some(new_val.clone()),
                );
            }
        }

        // Find removed properties
        for (key, old_val) in old_properties {
            if !new_properties.contains_key(key) {
                history.record_change(key.clone(), Some(old_val.clone()), None);
            }
        }
    }

    /// Returns the number of entities with history.
    pub fn len(&self) -> usize {
        self.histories.len()
    }

    /// Returns true if there are no recorded histories.
    pub fn is_empty(&self) -> bool {
        self.histories.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_relationship_creation() {
        let rel = TemporalRelationship::new("ue-001", "gnb-001", "connected_to");
        assert_eq!(rel.source_id, "ue-001");
        assert_eq!(rel.target_id, "gnb-001");
        assert_eq!(rel.relation_type, "connected_to");
        assert!(rel.is_active());
        assert!(rel.valid_to.is_none());
    }

    #[test]
    fn test_temporal_relationship_validity() {
        let rel = TemporalRelationship::with_validity(
            "ue-001",
            "gnb-001",
            "connected_to",
            100,
            Some(200),
        );

        assert!(rel.is_valid_at(100));
        assert!(rel.is_valid_at(150));
        assert!(!rel.is_valid_at(200)); // exclusive end
        assert!(!rel.is_valid_at(50));
    }

    #[test]
    fn test_temporal_relationship_expire() {
        let mut rel = TemporalRelationship::with_validity(
            "ue-001",
            "gnb-001",
            "connected_to",
            100,
            None,
        );

        assert!(rel.is_valid_at(1000));
        rel.expire_at(500);
        assert!(rel.is_valid_at(400));
        assert!(!rel.is_valid_at(500));
    }

    #[test]
    fn test_temporal_store_at_time() {
        let mut store = TemporalRelationshipStore::new();

        store.add(TemporalRelationship::with_validity(
            "ue-001", "gnb-001", "connected_to", 100, Some(200),
        ));
        store.add(TemporalRelationship::with_validity(
            "ue-001", "gnb-002", "connected_to", 200, Some(300),
        ));
        store.add(TemporalRelationship::with_validity(
            "ue-002", "gnb-001", "connected_to", 100, None,
        ));

        let at_150 = store.at_time(150);
        assert_eq!(at_150.len(), 2); // ue-001->gnb-001 and ue-002->gnb-001

        let at_250 = store.at_time(250);
        assert_eq!(at_250.len(), 2); // ue-001->gnb-002 and ue-002->gnb-001
    }

    #[test]
    fn test_temporal_store_for_entity() {
        let mut store = TemporalRelationshipStore::new();

        store.add(TemporalRelationship::with_validity(
            "ue-001", "gnb-001", "connected_to", 100, Some(200),
        ));
        store.add(TemporalRelationship::with_validity(
            "ue-001", "gnb-002", "connected_to", 200, Some(300),
        ));
        store.add(TemporalRelationship::with_validity(
            "ue-002", "gnb-001", "connected_to", 100, None,
        ));

        let ue1_rels = store.for_entity("ue-001");
        assert_eq!(ue1_rels.len(), 2);

        let ue1_at_150 = store.for_entity_at_time("ue-001", 150);
        assert_eq!(ue1_at_150.len(), 1);
        assert_eq!(ue1_at_150[0].target_id, "gnb-001");
    }

    #[test]
    fn test_temporal_store_expire_entity() {
        let mut store = TemporalRelationshipStore::new();

        store.add(TemporalRelationship::with_validity(
            "ue-001", "gnb-001", "connected_to", 100, None,
        ));
        store.add(TemporalRelationship::with_validity(
            "ue-001", "gnb-002", "served_by", 100, None,
        ));

        assert_eq!(store.active_count(), 2);
        store.expire_entity_relationships("ue-001");
        assert_eq!(store.active_count(), 0);
    }

    #[test]
    fn test_temporal_store_prune() {
        let mut store = TemporalRelationshipStore::new();

        store.add(TemporalRelationship::with_validity(
            "ue-001", "gnb-001", "connected_to", 100, Some(200),
        ));
        store.add(TemporalRelationship::with_validity(
            "ue-001", "gnb-002", "connected_to", 200, Some(300),
        ));
        store.add(TemporalRelationship::with_validity(
            "ue-002", "gnb-001", "connected_to", 100, None,
        ));

        assert_eq!(store.total_count(), 3);
        store.prune_before(250);
        // Should remove first relationship (ended at 200 < 250)
        assert_eq!(store.total_count(), 2);
    }

    #[test]
    fn test_entity_history_record() {
        let mut history = EntityHistory::new("gnb-001");

        history.record_change_at("status", None, Some("active".to_string()), 100);
        history.record_change_at("load", None, Some("50%".to_string()), 100);
        history.record_change_at("load", Some("50%".to_string()), Some("75%".to_string()), 200);

        assert_eq!(history.change_count(), 3);
    }

    #[test]
    fn test_entity_history_state_at() {
        let mut history = EntityHistory::new("gnb-001");

        history.record_change_at("status", None, Some("active".to_string()), 100);
        history.record_change_at("load", None, Some("50%".to_string()), 100);
        history.record_change_at("load", Some("50%".to_string()), Some("75%".to_string()), 200);
        history.record_change_at("status", Some("active".to_string()), Some("degraded".to_string()), 300);

        let state_at_150 = history.state_at(150);
        assert_eq!(state_at_150.get("status"), Some(&"active".to_string()));
        assert_eq!(state_at_150.get("load"), Some(&"50%".to_string()));

        let state_at_250 = history.state_at(250);
        assert_eq!(state_at_250.get("status"), Some(&"active".to_string()));
        assert_eq!(state_at_250.get("load"), Some(&"75%".to_string()));

        let state_at_350 = history.state_at(350);
        assert_eq!(state_at_350.get("status"), Some(&"degraded".to_string()));
        assert_eq!(state_at_350.get("load"), Some(&"75%".to_string()));
    }

    #[test]
    fn test_entity_history_changes_in_range() {
        let mut history = EntityHistory::new("gnb-001");

        history.record_change_at("status", None, Some("active".to_string()), 100);
        history.record_change_at("load", None, Some("50%".to_string()), 200);
        history.record_change_at("load", Some("50%".to_string()), Some("75%".to_string()), 300);

        let changes = history.changes_in_range(150, 350);
        assert_eq!(changes.len(), 2);
    }

    #[test]
    fn test_entity_history_changes_for_key() {
        let mut history = EntityHistory::new("gnb-001");

        history.record_change_at("status", None, Some("active".to_string()), 100);
        history.record_change_at("load", None, Some("50%".to_string()), 100);
        history.record_change_at("load", Some("50%".to_string()), Some("75%".to_string()), 200);

        let load_changes = history.changes_for_key("load");
        assert_eq!(load_changes.len(), 2);
    }

    #[test]
    fn test_entity_history_store() {
        let mut store = EntityHistoryStore::new();
        assert!(store.is_empty());

        let mut old_props = HashMap::new();
        old_props.insert("status".to_string(), "active".to_string());

        let mut new_props = HashMap::new();
        new_props.insert("status".to_string(), "degraded".to_string());
        new_props.insert("reason".to_string(), "high load".to_string());

        store.record_entity_update("gnb-001", &old_props, &new_props);

        assert_eq!(store.len(), 1);
        let history = store.get("gnb-001").expect("history should exist");
        assert_eq!(history.change_count(), 2); // status changed + reason added
    }

    #[test]
    fn test_entity_history_store_remove() {
        let mut store = EntityHistoryStore::new();
        store.get_or_create("gnb-001");
        assert_eq!(store.len(), 1);

        store.remove("gnb-001");
        assert!(store.is_empty());
    }

    #[test]
    fn test_temporal_store_for_entity_in_range() {
        let mut store = TemporalRelationshipStore::new();

        store.add(TemporalRelationship::with_validity(
            "ue-001", "gnb-001", "connected_to", 100, Some(200),
        ));
        store.add(TemporalRelationship::with_validity(
            "ue-001", "gnb-002", "connected_to", 300, Some(400),
        ));
        store.add(TemporalRelationship::with_validity(
            "ue-001", "gnb-003", "connected_to", 500, None,
        ));

        // Range that overlaps first two relationships
        let rels = store.for_entity_in_range("ue-001", 150, 350);
        assert_eq!(rels.len(), 2); // first (100-200) overlaps, second (300-400) overlaps
    }
}
