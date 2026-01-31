//! Network Knowledge Exposure Function (NKEF) for 6G Networks
//!
//! Implements knowledge management for LLM integration:
//! - Knowledge graphs with semantic search
//! - RAG (Retrieval Augmented Generation) support
//! - Network state exposure API
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                              NKEF                                        │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │ Knowledge Graph                                                  │   │
//! │  │  • Network topology                                              │   │
//! │  │  • UE context                                                    │   │
//! │  │  • Service instances                                             │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │ Semantic Search                                                  │   │
//! │  │  • Vector embeddings                                             │   │
//! │  │  • Similarity search                                             │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │ RAG Interface                                                    │   │
//! │  │  • Context retrieval                                             │   │
//! │  │  • Knowledge augmentation                                        │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Entity type in the knowledge graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityType {
    /// gNB (base station)
    Gnb,
    /// User Equipment
    Ue,
    /// Cell
    Cell,
    /// AMF (Access and Mobility Function)
    Amf,
    /// UPF (User Plane Function)
    Upf,
    /// Network slice
    Slice,
    /// PDU session
    PduSession,
    /// Service
    Service,
}

/// Entity in the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Entity ID
    pub id: String,
    /// Entity type
    pub entity_type: EntityType,
    /// Properties as key-value pairs
    pub properties: HashMap<String, String>,
    /// Vector embedding for semantic search
    pub embedding: Option<Vec<f32>>,
}

impl Entity {
    /// Creates a new entity
    pub fn new(id: impl Into<String>, entity_type: EntityType) -> Self {
        Self {
            id: id.into(),
            entity_type,
            properties: HashMap::new(),
            embedding: None,
        }
    }

    /// Adds a property
    pub fn with_property(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }

    /// Sets the embedding
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }
}

/// Relationship between entities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    /// Source entity ID
    pub source_id: String,
    /// Target entity ID
    pub target_id: String,
    /// Relationship type
    pub relation_type: String,
    /// Properties
    pub properties: HashMap<String, String>,
}

impl Relationship {
    /// Creates a new relationship
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
        }
    }
}

/// Query result from semantic search
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// Matching entity
    pub entity: Entity,
    /// Relevance score (0.0 to 1.0)
    pub relevance: f32,
}

/// Context for RAG queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryContext {
    /// Query intent (e.g., "find_ue", "network_status")
    pub intent: String,
    /// Additional filters
    pub filters: HashMap<String, String>,
    /// Maximum results
    pub max_results: usize,
}

/// Retrieved context for RAG
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievedContext {
    /// Context text
    pub context: String,
    /// Source entities
    pub sources: Vec<String>,
    /// Confidence score
    pub confidence: f32,
}

/// NKEF messages
#[derive(Debug)]
pub enum NkefMessage {
    /// Update knowledge with entity
    UpdateEntity(Entity),
    /// Add relationship
    AddRelationship(Relationship),
    /// Semantic query
    SemanticQuery {
        /// Query string
        query: String,
        /// Query context
        context: QueryContext,
        /// Response channel
        response_tx: Option<tokio::sync::oneshot::Sender<NkefResponse>>,
    },
    /// RAG context retrieval
    RetrieveContext {
        /// Prompt for context
        prompt: String,
        /// Maximum tokens
        max_tokens: u32,
        /// Response channel
        response_tx: Option<tokio::sync::oneshot::Sender<NkefResponse>>,
    },
    /// Remove entity
    RemoveEntity {
        /// Entity ID
        entity_id: String,
    },
}

/// NKEF responses
#[derive(Debug)]
pub enum NkefResponse {
    /// Query results
    QueryResults(Vec<QueryResult>),
    /// Retrieved context
    Context(RetrievedContext),
    /// Error
    Error(String),
    /// Success
    Ok,
}

/// Knowledge graph store
#[derive(Debug, Default)]
pub struct KnowledgeGraph {
    /// Entities by ID
    entities: HashMap<String, Entity>,
    /// Relationships
    relationships: Vec<Relationship>,
    /// Entity indices by type
    type_index: HashMap<EntityType, Vec<String>>,
}

impl KnowledgeGraph {
    /// Creates a new knowledge graph
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds an entity
    pub fn add_entity(&mut self, entity: Entity) {
        let id = entity.id.clone();
        let entity_type = entity.entity_type;

        self.type_index
            .entry(entity_type)
            .or_default()
            .push(id.clone());

        self.entities.insert(id, entity);
    }

    /// Gets an entity by ID
    pub fn get_entity(&self, id: &str) -> Option<&Entity> {
        self.entities.get(id)
    }

    /// Removes an entity
    pub fn remove_entity(&mut self, id: &str) -> Option<Entity> {
        if let Some(entity) = self.entities.remove(id) {
            // Remove from type index
            if let Some(ids) = self.type_index.get_mut(&entity.entity_type) {
                ids.retain(|i| i != id);
            }
            // Remove related relationships
            self.relationships.retain(|r| r.source_id != id && r.target_id != id);
            Some(entity)
        } else {
            None
        }
    }

    /// Adds a relationship
    pub fn add_relationship(&mut self, relationship: Relationship) {
        self.relationships.push(relationship);
    }

    /// Gets entities by type
    pub fn get_entities_by_type(&self, entity_type: EntityType) -> Vec<&Entity> {
        self.type_index
            .get(&entity_type)
            .map(|ids| ids.iter().filter_map(|id| self.entities.get(id)).collect())
            .unwrap_or_default()
    }

    /// Gets relationships for an entity
    pub fn get_relationships(&self, entity_id: &str) -> Vec<&Relationship> {
        self.relationships
            .iter()
            .filter(|r| r.source_id == entity_id || r.target_id == entity_id)
            .collect()
    }

    /// Simple keyword search (production would use vector similarity)
    pub fn search(&self, query: &str, max_results: usize) -> Vec<QueryResult> {
        let query_lower = query.to_lowercase();

        let mut results: Vec<QueryResult> = self
            .entities
            .values()
            .filter_map(|entity| {
                // Check if query matches ID or any property
                let id_match = entity.id.to_lowercase().contains(&query_lower);
                let prop_match = entity
                    .properties
                    .values()
                    .any(|v| v.to_lowercase().contains(&query_lower));

                if id_match || prop_match {
                    let relevance = if id_match { 0.9 } else { 0.7 };
                    Some(QueryResult {
                        entity: entity.clone(),
                        relevance,
                    })
                } else {
                    None
                }
            })
            .collect();

        results.sort_by(|a, b| b.relevance.partial_cmp(&a.relevance).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(max_results);
        results
    }

    /// Generates context for RAG
    pub fn generate_context(&self, prompt: &str, _max_tokens: u32) -> RetrievedContext {
        // Search for relevant entities
        let results = self.search(prompt, 5);

        // Calculate confidence before consuming results
        let confidence = if results.is_empty() { 0.0 } else { 0.8 };

        // Build context string
        let mut context_parts = Vec::new();
        let mut sources = Vec::new();

        for result in results {
            let entity = &result.entity;
            sources.push(entity.id.clone());

            let props: String = entity
                .properties
                .iter()
                .map(|(k, v)| format!("{}: {}", k, v))
                .collect::<Vec<_>>()
                .join(", ");

            context_parts.push(format!(
                "{} ({}): {}",
                entity.id,
                format!("{:?}", entity.entity_type),
                props
            ));
        }

        let context = context_parts.join("\n");

        RetrievedContext {
            context,
            sources,
            confidence,
        }
    }

    /// Returns the number of entities
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Returns the number of relationships
    pub fn relationship_count(&self) -> usize {
        self.relationships.len()
    }
}

/// NKEF manager
#[derive(Debug, Default)]
pub struct NkefManager {
    /// Knowledge graph
    graph: KnowledgeGraph,
    /// Embedding dimension
    embedding_dim: usize,
}

impl NkefManager {
    /// Creates a new NKEF manager
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            graph: KnowledgeGraph::new(),
            embedding_dim,
        }
    }

    /// Returns a reference to the knowledge graph
    pub fn graph(&self) -> &KnowledgeGraph {
        &self.graph
    }

    /// Returns a mutable reference to the knowledge graph
    pub fn graph_mut(&mut self) -> &mut KnowledgeGraph {
        &mut self.graph
    }

    /// Updates an entity
    pub fn update_entity(&mut self, entity: Entity) {
        self.graph.add_entity(entity);
    }

    /// Performs semantic query
    pub fn query(&self, query: &str, context: &QueryContext) -> Vec<QueryResult> {
        self.graph.search(query, context.max_results)
    }

    /// Retrieves context for RAG
    pub fn retrieve_context(&self, prompt: &str, max_tokens: u32) -> RetrievedContext {
        self.graph.generate_context(prompt, max_tokens)
    }

    /// Returns the configured embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_creation() {
        let entity = Entity::new("gnb-001", EntityType::Gnb)
            .with_property("location", "Building A")
            .with_property("status", "active");

        assert_eq!(entity.id, "gnb-001");
        assert_eq!(entity.entity_type, EntityType::Gnb);
        assert_eq!(entity.properties.get("location"), Some(&"Building A".to_string()));
    }

    #[test]
    fn test_knowledge_graph() {
        let mut graph = KnowledgeGraph::new();

        graph.add_entity(Entity::new("gnb-001", EntityType::Gnb).with_property("status", "active"));
        graph.add_entity(Entity::new("ue-001", EntityType::Ue).with_property("imsi", "12345"));

        graph.add_relationship(Relationship::new("ue-001", "gnb-001", "connected_to"));

        assert_eq!(graph.entity_count(), 2);
        assert_eq!(graph.relationship_count(), 1);

        let gnbs = graph.get_entities_by_type(EntityType::Gnb);
        assert_eq!(gnbs.len(), 1);
    }

    #[test]
    fn test_search() {
        let mut graph = KnowledgeGraph::new();

        graph.add_entity(Entity::new("gnb-001", EntityType::Gnb).with_property("name", "Main Tower"));
        graph.add_entity(Entity::new("gnb-002", EntityType::Gnb).with_property("name", "Secondary"));
        graph.add_entity(Entity::new("ue-001", EntityType::Ue).with_property("name", "Test UE"));

        let results = graph.search("tower", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].entity.id, "gnb-001");
    }

    #[test]
    fn test_context_generation() {
        let mut graph = KnowledgeGraph::new();

        graph.add_entity(
            Entity::new("gnb-001", EntityType::Gnb)
                .with_property("status", "active")
                .with_property("load", "75%"),
        );

        // Search for "gnb" which matches the entity ID "gnb-001"
        let context = graph.generate_context("gnb", 1000);
        assert!(!context.context.is_empty());
        assert!(!context.sources.is_empty());
    }

    #[test]
    fn test_nkef_manager() {
        let mut manager = NkefManager::new(384);

        manager.update_entity(Entity::new("cell-001", EntityType::Cell).with_property("pci", "100"));

        let context = QueryContext {
            intent: "find_cell".to_string(),
            filters: HashMap::new(),
            max_results: 10,
        };

        let results = manager.query("cell", &context);
        assert_eq!(results.len(), 1);
    }
}
