//! Distributed Knowledge Graph (A14.3)
//!
//! Implements distributed knowledge graph with replication and consistency mechanisms
//! for multi-site deployments.

#![allow(missing_docs)]

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::{Entity, Relationship};

/// Node identifier in the distributed system
pub type NodeId = String;

/// Distributed knowledge graph node
#[derive(Debug, Clone)]
pub struct DistributedNode {
    /// This node's unique identifier
    pub node_id: NodeId,
    /// Local entities (owned by this node)
    pub local_entities: HashMap<String, Entity>,
    /// Replicated entities from other nodes
    pub replicated_entities: HashMap<String, ReplicatedEntity>,
    /// Local relationships
    pub local_relationships: Vec<Relationship>,
    /// Peer nodes in the cluster
    pub peers: HashSet<NodeId>,
    /// Replication configuration
    pub config: ReplicationConfig,
}

/// Replicated entity with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicatedEntity {
    /// The entity itself
    pub entity: Entity,
    /// Source node that owns this entity
    pub source_node: NodeId,
    /// Version number for conflict resolution
    pub version: u64,
    /// Last replication timestamp (ms since epoch)
    pub replicated_at_ms: u64,
}

/// Replication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationConfig {
    /// Replication factor (number of replicas)
    pub replication_factor: usize,
    /// Consistency level
    pub consistency: ConsistencyLevel,
    /// Replication timeout (ms)
    pub timeout_ms: u64,
    /// Enable automatic conflict resolution
    pub auto_resolve_conflicts: bool,
}

impl Default for ReplicationConfig {
    fn default() -> Self {
        Self {
            replication_factor: 3,
            consistency: ConsistencyLevel::EventualConsistency,
            timeout_ms: 5000,
            auto_resolve_conflicts: true,
        }
    }
}

/// Consistency level for replication
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsistencyLevel {
    /// Strong consistency (all replicas must acknowledge)
    StrongConsistency,
    /// Quorum consistency (majority must acknowledge)
    QuorumConsistency,
    /// Eventual consistency (async replication)
    EventualConsistency,
}

/// Replication message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReplicationMessage {
    /// Add or update entity
    EntityUpdate {
        entity: Entity,
        version: u64,
        source_node: NodeId,
    },
    /// Delete entity
    EntityDelete {
        entity_id: String,
        version: u64,
        source_node: NodeId,
    },
    /// Add relationship
    RelationshipAdd {
        relationship: Relationship,
        version: u64,
        source_node: NodeId,
    },
    /// Request full sync
    SyncRequest {
        requesting_node: NodeId,
        last_sync_version: u64,
    },
    /// Response to sync request
    SyncResponse {
        entities: Vec<ReplicatedEntity>,
        relationships: Vec<Relationship>,
        current_version: u64,
    },
}

/// Conflict resolution strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictResolution {
    /// Last-write-wins based on timestamp
    LastWriteWins,
    /// Higher version number wins
    VersionWins,
    /// Keep both versions (multi-value)
    KeepBoth,
}

impl DistributedNode {
    /// Creates a new distributed node
    pub fn new(node_id: NodeId, config: ReplicationConfig) -> Self {
        Self {
            node_id,
            local_entities: HashMap::new(),
            replicated_entities: HashMap::new(),
            local_relationships: Vec::new(),
            peers: HashSet::new(),
            config,
        }
    }

    /// Adds a peer node to the cluster
    pub fn add_peer(&mut self, peer_id: NodeId) {
        self.peers.insert(peer_id);
    }

    /// Removes a peer node
    pub fn remove_peer(&mut self, peer_id: &NodeId) {
        self.peers.remove(peer_id);
    }

    /// Adds a local entity and prepares for replication
    pub fn add_local_entity(&mut self, entity: Entity) -> ReplicationMessage {
        let entity_id = entity.id.clone();
        let version = self.generate_version();

        self.local_entities.insert(entity_id, entity.clone());

        ReplicationMessage::EntityUpdate {
            entity,
            version,
            source_node: self.node_id.clone(),
        }
    }

    /// Handles incoming replication message
    pub fn handle_replication_message(&mut self, message: ReplicationMessage) -> ReplicationResult {
        match message {
            ReplicationMessage::EntityUpdate {
                entity,
                version,
                source_node,
            } => self.replicate_entity(entity, source_node, version),

            ReplicationMessage::EntityDelete {
                entity_id,
                version,
                source_node,
            } => self.replicate_delete(&entity_id, &source_node, version),

            ReplicationMessage::RelationshipAdd {
                relationship,
                version: _,
                source_node: _,
            } => {
                self.local_relationships.push(relationship);
                ReplicationResult::Success
            }

            ReplicationMessage::SyncRequest {
                requesting_node: _,
                last_sync_version: _,
            } => self.handle_sync_request(),

            ReplicationMessage::SyncResponse {
                entities,
                relationships,
                current_version: _,
            } => self.handle_sync_response(entities, relationships),
        }
    }

    /// Replicates an entity from another node
    fn replicate_entity(&mut self, entity: Entity, source_node: NodeId, version: u64) -> ReplicationResult {
        let entity_id = entity.id.clone();

        // Check if we already have this entity
        if let Some(existing) = self.replicated_entities.get(&entity_id) {
            // Conflict detection
            if existing.source_node != source_node {
                let existing_clone = existing.clone();
                return self.resolve_conflict(&existing_clone, &entity, version);
            }
            // Same source: update if version is newer
            if version <= existing.version {
                return ReplicationResult::AlreadyUpToDate;
            }
        }

        // Add or update replicated entity
        let replicated = ReplicatedEntity {
            entity,
            source_node,
            version,
            replicated_at_ms: current_timestamp_ms(),
        };

        self.replicated_entities.insert(entity_id, replicated);
        ReplicationResult::Success
    }

    /// Handles entity deletion
    fn replicate_delete(&mut self, entity_id: &str, source_node: &NodeId, version: u64) -> ReplicationResult {
        if let Some(existing) = self.replicated_entities.get(entity_id) {
            if &existing.source_node == source_node && version > existing.version {
                self.replicated_entities.remove(entity_id);
                return ReplicationResult::Success;
            }
        }
        ReplicationResult::AlreadyUpToDate
    }

    /// Resolves conflicts between replicas
    fn resolve_conflict(
        &mut self,
        existing: &ReplicatedEntity,
        new_entity: &Entity,
        new_version: u64,
    ) -> ReplicationResult {
        if !self.config.auto_resolve_conflicts {
            return ReplicationResult::Conflict {
                entity_id: existing.entity.id.clone(),
                existing_version: existing.version,
                new_version,
            };
        }

        // Last-write-wins based on version
        match ConflictResolution::VersionWins {
            ConflictResolution::VersionWins => {
                if new_version > existing.version {
                    let entity_id = new_entity.id.clone();
                    let replicated = ReplicatedEntity {
                        entity: new_entity.clone(),
                        source_node: existing.source_node.clone(),
                        version: new_version,
                        replicated_at_ms: current_timestamp_ms(),
                    };
                    self.replicated_entities.insert(entity_id, replicated);
                    ReplicationResult::ConflictResolved
                } else {
                    ReplicationResult::AlreadyUpToDate
                }
            }
            _ => ReplicationResult::Conflict {
                entity_id: existing.entity.id.clone(),
                existing_version: existing.version,
                new_version,
            },
        }
    }

    /// Handles sync request from another node
    fn handle_sync_request(&self) -> ReplicationResult {
        // Return all local entities for sync
        let entities: Vec<ReplicatedEntity> = self
            .local_entities
            .values()
            .map(|entity| ReplicatedEntity {
                entity: entity.clone(),
                source_node: self.node_id.clone(),
                version: 1, // Simplified versioning
                replicated_at_ms: current_timestamp_ms(),
            })
            .collect();

        ReplicationResult::SyncData {
            entities,
            relationships: self.local_relationships.clone(),
        }
    }

    /// Handles sync response
    fn handle_sync_response(&mut self, entities: Vec<ReplicatedEntity>, relationships: Vec<Relationship>) -> ReplicationResult {
        for replicated in entities {
            let entity_id = replicated.entity.id.clone();
            self.replicated_entities.insert(entity_id, replicated);
        }

        for relationship in relationships {
            if !self.local_relationships.iter().any(|r| r.source_id == relationship.source_id && r.target_id == relationship.target_id) {
                self.local_relationships.push(relationship);
            }
        }

        ReplicationResult::Success
    }

    /// Generates a version number (monotonically increasing)
    fn generate_version(&self) -> u64 {
        current_timestamp_ms()
    }

    /// Gets all entities (local + replicated)
    pub fn all_entities(&self) -> Vec<&Entity> {
        let mut entities: Vec<&Entity> = self.local_entities.values().collect();
        entities.extend(self.replicated_entities.values().map(|r| &r.entity));
        entities
    }

    /// Gets replication status
    pub fn replication_status(&self) -> ReplicationStatus {
        ReplicationStatus {
            node_id: self.node_id.clone(),
            local_entity_count: self.local_entities.len(),
            replicated_entity_count: self.replicated_entities.len(),
            peer_count: self.peers.len(),
            consistency_level: self.config.consistency,
        }
    }
}

/// Result of replication operation
#[derive(Debug, Clone)]
pub enum ReplicationResult {
    /// Operation succeeded
    Success,
    /// Entity/relationship already up to date
    AlreadyUpToDate,
    /// Conflict detected
    Conflict {
        entity_id: String,
        existing_version: u64,
        new_version: u64,
    },
    /// Conflict resolved automatically
    ConflictResolved,
    /// Sync data response
    SyncData {
        entities: Vec<ReplicatedEntity>,
        relationships: Vec<Relationship>,
    },
}

/// Replication status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationStatus {
    /// Node identifier
    pub node_id: NodeId,
    /// Number of local entities
    pub local_entity_count: usize,
    /// Number of replicated entities
    pub replicated_entity_count: usize,
    /// Number of peer nodes
    pub peer_count: usize,
    /// Consistency level
    pub consistency_level: ConsistencyLevel,
}

/// Gets current timestamp in milliseconds
fn current_timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EntityType;

    #[test]
    fn test_distributed_node_creation() {
        let config = ReplicationConfig::default();
        let node = DistributedNode::new("node1".to_string(), config);

        assert_eq!(node.node_id, "node1");
        assert_eq!(node.local_entities.len(), 0);
        assert_eq!(node.replicated_entities.len(), 0);
    }

    #[test]
    fn test_add_peer() {
        let config = ReplicationConfig::default();
        let mut node = DistributedNode::new("node1".to_string(), config);

        node.add_peer("node2".to_string());
        node.add_peer("node3".to_string());

        assert_eq!(node.peers.len(), 2);
        assert!(node.peers.contains("node2"));
    }

    #[test]
    fn test_add_local_entity() {
        let config = ReplicationConfig::default();
        let mut node = DistributedNode::new("node1".to_string(), config);

        let entity = Entity {
            id: "ue1".to_string(),
            entity_type: EntityType::Ue,
            properties: HashMap::new(),
            embedding: None,
        };

        let msg = node.add_local_entity(entity);

        assert_eq!(node.local_entities.len(), 1);
        match msg {
            ReplicationMessage::EntityUpdate { entity, .. } => {
                assert_eq!(entity.id, "ue1");
            }
            _ => panic!("Expected EntityUpdate message"),
        }
    }

    #[test]
    fn test_replicate_entity() {
        let config = ReplicationConfig::default();
        let mut node = DistributedNode::new("node1".to_string(), config);

        let entity = Entity {
            id: "ue1".to_string(),
            entity_type: EntityType::Ue,
            properties: HashMap::new(),
            embedding: None,
        };

        let result = node.replicate_entity(entity, "node2".to_string(), 1);

        assert!(matches!(result, ReplicationResult::Success));
        assert_eq!(node.replicated_entities.len(), 1);
    }

    #[test]
    fn test_conflict_resolution() {
        let config = ReplicationConfig::default();
        let mut node = DistributedNode::new("node1".to_string(), config);

        // Add initial entity
        let mut entity1 = Entity {
            id: "ue1".to_string(),
            entity_type: EntityType::Ue,
            properties: HashMap::new(),
            embedding: None,
        };
        entity1.properties.insert("key".to_string(), "value1".to_string());

        node.replicate_entity(entity1, "node2".to_string(), 1);

        // Replicate updated entity with higher version
        let mut entity2 = Entity {
            id: "ue1".to_string(),
            entity_type: EntityType::Ue,
            properties: HashMap::new(),
            embedding: None,
        };
        entity2.properties.insert("key".to_string(), "value2".to_string());

        let result = node.replicate_entity(entity2, "node2".to_string(), 2);

        assert!(matches!(result, ReplicationResult::Success));
        let replicated = node.replicated_entities.get("ue1").unwrap();
        assert_eq!(replicated.version, 2);
        assert_eq!(replicated.entity.properties.get("key").unwrap(), "value2");
    }

    #[test]
    fn test_all_entities() {
        let config = ReplicationConfig::default();
        let mut node = DistributedNode::new("node1".to_string(), config);

        // Add local entity
        let local_entity = Entity {
            id: "local1".to_string(),
            entity_type: EntityType::Gnb,
            properties: HashMap::new(),
            embedding: None,
        };
        node.add_local_entity(local_entity);

        // Add replicated entity
        let remote_entity = Entity {
            id: "remote1".to_string(),
            entity_type: EntityType::Ue,
            properties: HashMap::new(),
            embedding: None,
        };
        node.replicate_entity(remote_entity, "node2".to_string(), 1);

        let all = node.all_entities();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_replication_status() {
        let config = ReplicationConfig::default();
        let mut node = DistributedNode::new("node1".to_string(), config);

        node.add_peer("node2".to_string());
        node.add_peer("node3".to_string());

        let status = node.replication_status();
        assert_eq!(status.node_id, "node1");
        assert_eq!(status.peer_count, 2);
        assert_eq!(status.consistency_level, ConsistencyLevel::EventualConsistency);
    }
}
