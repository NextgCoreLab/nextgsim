//! Network Knowledge Exposure Function (NKEF) for 6G Networks
//!
//! Implements knowledge management for LLM integration:
//! - Knowledge graphs with semantic search (vector similarity)
//! - RAG (Retrieval Augmented Generation) support
//! - Network state exposure API
//! - Temporal knowledge tracking
//! - Event-driven real-time updates
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                              NKEF                                        │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │ Knowledge Graph                                                  │   │
//! │  │  • Network topology       • Temporal relationships               │   │
//! │  │  • UE context             • Entity history tracking              │   │
//! │  │  • Service instances      • Time-windowed queries                │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │ Semantic Search                                                  │   │
//! │  │  • TF-IDF text embeddings • Vector similarity index              │   │
//! │  │  • Cosine similarity      • Top-k nearest neighbor               │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │ RAG Interface                                                    │   │
//! │  │  • Structured markdown context  • Relevance scoring              │   │
//! │  │  • Token-limited output         • Relationship-aware context     │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │ Event System                                                     │   │
//! │  │  • Knowledge events       • Handler registration                 │   │
//! │  │  • Event queue            • Batch processing                     │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```

pub mod access;
pub mod distributed;
pub mod embedder;
pub mod events;
pub mod ontology;
pub mod query;
pub mod rag;
pub mod storage;
pub mod temporal;
pub mod vector;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Re-export key types from submodules
pub use access::{AccessController, AccessLevel, AccessPolicy};
pub use distributed::{
    ConsistencyLevel, DistributedNode, NodeId, ReplicatedEntity, ReplicationConfig,
    ReplicationMessage, ReplicationResult, ReplicationStatus,
};
pub use embedder::{OnnxEmbedder, TextEmbedder};
pub use events::{EventBus, EventHandler, HandlerId, KnowledgeEvent, KnowledgeEventKind};
pub use ontology::{
    AttributeDefinition, AttributeType, Constraint, EntitySchema, Ontology, RelationshipSchema,
    SchemaError,
};
pub use query::{
    GraphPath, GraphPattern, GraphQuery, PathStep, QueryBuilder, QueryExecutor, QueryFilter,
    QueryOperator, QueryResult as GraphQueryResult, RelationshipPattern,
};
pub use rag::{BuiltContext, ContextBuilder, ContextFormat, RagConfig};
pub use storage::{PersistentStorage, StorageConfig, StorageError};
pub use temporal::{
    AttributeChange, EntityHistory, EntityHistoryStore, TemporalRelationship,
    TemporalRelationshipStore, Timestamp,
};
pub use vector::{SimilarityResult, VectorIndex, cosine_similarity, dot_product, l2_norm, normalize};

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
    /// Analytics result (from NWDAF)
    Analytics,
    /// Agent (AI agent)
    Agent,
    /// Intent (agent intent or network intent)
    Intent,
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
    /// Query intent (e.g., "`find_ue`", "`network_status`")
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

/// Knowledge graph store with vector similarity search, temporal tracking,
/// and event-driven updates.
pub struct KnowledgeGraph {
    /// Entities by ID
    entities: HashMap<String, Entity>,
    /// Relationships (non-temporal, for backward compatibility)
    relationships: Vec<Relationship>,
    /// Entity indices by type
    type_index: HashMap<EntityType, Vec<String>>,
    /// Vector index for embedding-based similarity search
    vector_index: VectorIndex,
    /// Text embedder for generating embeddings from entity descriptions
    embedder: TextEmbedder,
    /// Temporal relationship store
    temporal_relationships: TemporalRelationshipStore,
    /// Entity history store
    entity_histories: EntityHistoryStore,
    /// Event bus for knowledge graph change notifications
    event_bus: EventBus,
}

impl std::fmt::Debug for KnowledgeGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KnowledgeGraph")
            .field("entity_count", &self.entities.len())
            .field("relationship_count", &self.relationships.len())
            .field("vector_index_size", &self.vector_index.len())
            .field("temporal_relationship_count", &self.temporal_relationships.total_count())
            .field("entity_history_count", &self.entity_histories.len())
            .finish()
    }
}

impl Default for KnowledgeGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl KnowledgeGraph {
    /// Creates a new knowledge graph with default embedding dimension (128).
    pub fn new() -> Self {
        Self::with_embedding_dim(128)
    }

    /// Creates a new knowledge graph with the specified embedding dimension.
    pub fn with_embedding_dim(dim: usize) -> Self {
        Self {
            entities: HashMap::new(),
            relationships: Vec::new(),
            type_index: HashMap::new(),
            vector_index: VectorIndex::new(dim),
            embedder: TextEmbedder::new(dim),
            temporal_relationships: TemporalRelationshipStore::new(),
            entity_histories: EntityHistoryStore::new(),
            event_bus: EventBus::default(),
        }
    }

    /// Adds an entity to the knowledge graph.
    ///
    /// If an entity with the same ID already exists, it is replaced and the
    /// change is recorded in the entity history. An `EntityCreated` or
    /// `EntityUpdated` event is dispatched.
    pub fn add_entity(&mut self, entity: Entity) {
        let id = entity.id.clone();
        let entity_type = entity.entity_type;
        let now = temporal::now_millis();

        // Check if this is an update or creation
        let is_update = self.entities.contains_key(&id);

        if is_update {
            // Record history of property changes
            if let Some(old_entity) = self.entities.get(&id) {
                self.entity_histories.record_entity_update(
                    &id,
                    &old_entity.properties,
                    &entity.properties,
                );

                // Build changed properties map for the event
                let mut changed = HashMap::new();
                for (key, new_val) in &entity.properties {
                    let old_val = old_entity.properties.get(key);
                    if old_val.map(|v| v != new_val).unwrap_or(true) {
                        changed.insert(
                            key.clone(),
                            (old_val.cloned(), Some(new_val.clone())),
                        );
                    }
                }
                for (key, old_val) in &old_entity.properties {
                    if !entity.properties.contains_key(key) {
                        changed.insert(
                            key.clone(),
                            (Some(old_val.clone()), None),
                        );
                    }
                }

                self.event_bus.dispatch(KnowledgeEvent::entity_updated(&id, now, changed));
            }
        } else {
            // New entity - add to type index
            self.type_index
                .entry(entity_type)
                .or_default()
                .push(id.clone());

            // Record initial state in history
            let history = self.entity_histories.get_or_create(&id);
            for (key, value) in &entity.properties {
                history.record_change(key.clone(), None, Some(value.clone()));
            }

            self.event_bus.dispatch(KnowledgeEvent::entity_created(&id, now));
        }

        // Generate embedding if not already present
        let entity = if entity.embedding.is_some() {
            entity
        } else {
            let text = TextEmbedder::entity_text(
                &entity.id,
                &format!("{:?}", entity.entity_type),
                &entity.properties,
            );
            let emb = self.embedder.embed(&text);
            Entity {
                embedding: Some(emb),
                ..entity
            }
        };

        // Update vector index
        if let Some(ref emb) = entity.embedding {
            self.vector_index.upsert(&id, emb.clone());
        }

        self.entities.insert(id, entity);
    }

    /// Gets an entity by ID
    pub fn get_entity(&self, id: &str) -> Option<&Entity> {
        self.entities.get(id)
    }

    /// Removes an entity from the knowledge graph.
    ///
    /// Also removes the entity's embedding from the vector index, expires its
    /// temporal relationships, and dispatches an `EntityRemoved` event.
    pub fn remove_entity(&mut self, id: &str) -> Option<Entity> {
        if let Some(entity) = self.entities.remove(id) {
            // Remove from type index
            if let Some(ids) = self.type_index.get_mut(&entity.entity_type) {
                ids.retain(|i| i != id);
            }

            // Remove related non-temporal relationships
            let removed_rels: Vec<(String, String, String)> = self
                .relationships
                .iter()
                .filter(|r| r.source_id == id || r.target_id == id)
                .map(|r| (r.source_id.clone(), r.target_id.clone(), r.relation_type.clone()))
                .collect();
            self.relationships.retain(|r| r.source_id != id && r.target_id != id);

            // Dispatch relationship removal events
            let now = temporal::now_millis();
            for (source, target, rel_type) in &removed_rels {
                self.event_bus.dispatch(KnowledgeEvent::relationship_removed(
                    source, target, rel_type, now,
                ));
            }

            // Expire temporal relationships
            self.temporal_relationships.expire_entity_relationships(id);

            // Remove from vector index
            self.vector_index.remove(id);

            // Dispatch entity removed event
            self.event_bus.dispatch(KnowledgeEvent::entity_removed(id, now));

            Some(entity)
        } else {
            None
        }
    }

    /// Adds a (non-temporal) relationship.
    ///
    /// Dispatches a `RelationshipAdded` event.
    pub fn add_relationship(&mut self, relationship: Relationship) {
        let now = temporal::now_millis();
        self.event_bus.dispatch(KnowledgeEvent::relationship_added(
            &relationship.source_id,
            &relationship.target_id,
            &relationship.relation_type,
            now,
        ));
        self.relationships.push(relationship);
    }

    /// Adds a temporal relationship with validity window.
    ///
    /// Dispatches a `RelationshipAdded` event.
    pub fn add_temporal_relationship(&mut self, relationship: TemporalRelationship) {
        let now = temporal::now_millis();
        self.event_bus.dispatch(KnowledgeEvent::relationship_added(
            &relationship.source_id,
            &relationship.target_id,
            &relationship.relation_type,
            now,
        ));
        self.temporal_relationships.add(relationship);
    }

    /// Gets entities by type
    pub fn get_entities_by_type(&self, entity_type: EntityType) -> Vec<&Entity> {
        self.type_index
            .get(&entity_type)
            .map(|ids| ids.iter().filter_map(|id| self.entities.get(id)).collect())
            .unwrap_or_default()
    }

    /// Gets (non-temporal) relationships for an entity
    pub fn get_relationships(&self, entity_id: &str) -> Vec<&Relationship> {
        self.relationships
            .iter()
            .filter(|r| r.source_id == entity_id || r.target_id == entity_id)
            .collect()
    }

    /// Gets active temporal relationships for an entity.
    pub fn get_temporal_relationships(&self, entity_id: &str) -> Vec<&TemporalRelationship> {
        self.temporal_relationships.active_for_entity(entity_id)
    }

    /// Gets temporal relationships for an entity at a specific point in time.
    pub fn get_temporal_relationships_at(
        &self,
        entity_id: &str,
        timestamp: Timestamp,
    ) -> Vec<&TemporalRelationship> {
        self.temporal_relationships.for_entity_at_time(entity_id, timestamp)
    }

    /// Gets the history of attribute changes for an entity.
    pub fn get_entity_history(&self, entity_id: &str) -> Option<&EntityHistory> {
        self.entity_histories.get(entity_id)
    }

    /// Reconstructs an entity's properties at a historical point in time.
    pub fn get_entity_state_at(
        &self,
        entity_id: &str,
        timestamp: Timestamp,
    ) -> Option<HashMap<String, String>> {
        self.entity_histories.get(entity_id).map(|h| h.state_at(timestamp))
    }

    /// Keyword-based search (backward compatible).
    ///
    /// If embeddings are available, this method uses vector similarity search
    /// with keyword matching as a fallback for unembedded queries.
    pub fn search(&self, query: &str, max_results: usize) -> Vec<QueryResult> {
        // Try vector similarity search first
        if !self.vector_index.is_empty() {
            let query_embedding = self.embedder.embed_query(query);
            let has_nonzero = query_embedding.iter().any(|&v| v != 0.0);

            if has_nonzero {
                let sim_results = self.vector_index.search_topk(&query_embedding, max_results);

                if !sim_results.is_empty() {
                    return sim_results
                        .into_iter()
                        .filter_map(|sr| {
                            self.entities.get(&sr.id).map(|entity| {
                                // Map cosine similarity [-1, 1] to relevance [0, 1]
                                let relevance = (sr.score + 1.0) / 2.0;
                                QueryResult {
                                    entity: entity.clone(),
                                    relevance,
                                }
                            })
                        })
                        .collect();
                }
            }
        }

        // Fallback to keyword search
        self.keyword_search(query, max_results)
    }

    /// Performs vector similarity search using the query embedding.
    ///
    /// Returns entities ranked by cosine similarity to the query embedding.
    pub fn vector_search(&self, query_embedding: &[f32], max_results: usize) -> Vec<QueryResult> {
        let sim_results = self.vector_index.search_topk(query_embedding, max_results);

        sim_results
            .into_iter()
            .filter_map(|sr| {
                self.entities.get(&sr.id).map(|entity| {
                    let relevance = (sr.score + 1.0) / 2.0;
                    QueryResult {
                        entity: entity.clone(),
                        relevance,
                    }
                })
            })
            .collect()
    }

    /// Performs keyword-only search (the original search implementation).
    pub fn keyword_search(&self, query: &str, max_results: usize) -> Vec<QueryResult> {
        let query_lower = query.to_lowercase();

        let mut results: Vec<QueryResult> = self
            .entities
            .values()
            .filter_map(|entity| {
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

        results.sort_by(|a, b| {
            b.relevance
                .partial_cmp(&a.relevance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(max_results);
        results
    }

    /// Generates context for RAG (backward compatible).
    ///
    /// Uses the improved context builder with structured markdown output,
    /// relationship-aware context, and token limiting.
    pub fn generate_context(&self, prompt: &str, max_tokens: u32) -> RetrievedContext {
        let results = self.search(prompt, 5);

        let config = RagConfig::default()
            .with_max_tokens(max_tokens)
            .with_max_entries(10);
        let mut builder = ContextBuilder::with_config(config);

        for result in &results {
            let rels = self.get_relationships(&result.entity.id);
            builder.add_entity(&result.entity, result.relevance, &rels);
        }

        let built = builder.build();

        RetrievedContext {
            context: built.text,
            sources: built.sources,
            confidence: built.confidence,
        }
    }

    /// Generates RAG context with full configuration control.
    ///
    /// Returns a [`BuiltContext`] with detailed metadata about the context.
    pub fn generate_context_with_config(
        &self,
        prompt: &str,
        config: RagConfig,
    ) -> BuiltContext {
        let results = self.search(prompt, config.max_entries);

        let mut builder = ContextBuilder::with_config(config);

        for result in &results {
            let rels = self.get_relationships(&result.entity.id);
            builder.add_entity(&result.entity, result.relevance, &rels);
        }

        builder.build()
    }

    /// Rebuilds the TF-IDF vocabulary and all entity embeddings.
    ///
    /// Call this after bulk-loading entities to ensure the vocabulary and
    /// embeddings reflect the full corpus. Entities added via `add_entity`
    /// get embeddings automatically, but the TF-IDF vocabulary is more
    /// accurate when built from the full set of documents.
    pub fn rebuild_embeddings(&mut self) {
        // Build corpus from all entities
        let documents: Vec<String> = self
            .entities
            .values()
            .map(|entity| {
                TextEmbedder::entity_text(
                    &entity.id,
                    &format!("{:?}", entity.entity_type),
                    &entity.properties,
                )
            })
            .collect();

        self.embedder.build_vocabulary(&documents);

        // Regenerate all embeddings
        let entity_ids: Vec<String> = self.entities.keys().cloned().collect();
        for id in entity_ids {
            if let Some(entity) = self.entities.get(&id) {
                let text = TextEmbedder::entity_text(
                    &entity.id,
                    &format!("{:?}", entity.entity_type),
                    &entity.properties,
                );
                let emb = self.embedder.embed(&text);
                self.vector_index.upsert(&id, emb.clone());

                // Update entity's stored embedding
                if let Some(entity_mut) = self.entities.get_mut(&id) {
                    entity_mut.embedding = Some(emb);
                }
            }
        }
    }

    /// Returns the number of entities
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Returns the number of (non-temporal) relationships
    pub fn relationship_count(&self) -> usize {
        self.relationships.len()
    }

    /// Returns the number of temporal relationships (all, including expired).
    pub fn temporal_relationship_count(&self) -> usize {
        self.temporal_relationships.total_count()
    }

    /// Returns the number of currently active temporal relationships.
    pub fn active_temporal_relationship_count(&self) -> usize {
        self.temporal_relationships.active_count()
    }

    /// Returns a reference to the vector index.
    pub fn vector_index(&self) -> &VectorIndex {
        &self.vector_index
    }

    /// Returns a reference to the text embedder.
    pub fn embedder(&self) -> &TextEmbedder {
        &self.embedder
    }

    /// Returns a mutable reference to the event bus for handler registration.
    pub fn event_bus_mut(&mut self) -> &mut EventBus {
        &mut self.event_bus
    }

    /// Returns a reference to the event bus.
    pub fn event_bus(&self) -> &EventBus {
        &self.event_bus
    }

    /// Returns a reference to the temporal relationship store.
    pub fn temporal_relationships(&self) -> &TemporalRelationshipStore {
        &self.temporal_relationships
    }

    /// Returns a mutable reference to the temporal relationship store.
    pub fn temporal_relationships_mut(&mut self) -> &mut TemporalRelationshipStore {
        &mut self.temporal_relationships
    }

    /// Returns a reference to the entity history store.
    pub fn entity_histories(&self) -> &EntityHistoryStore {
        &self.entity_histories
    }

    /// Returns all entity IDs in the graph.
    pub fn entity_ids(&self) -> Vec<&str> {
        self.entities.keys().map(std::string::String::as_str).collect()
    }

    /// Returns an iterator over all entities.
    pub fn entities(&self) -> impl Iterator<Item = &Entity> {
        self.entities.values()
    }

    /// Returns an iterator over all (non-temporal) relationships.
    pub fn relationships(&self) -> impl Iterator<Item = &Relationship> {
        self.relationships.iter()
    }
}

/// NKEF manager
#[derive(Default)]
pub struct NkefManager {
    /// Knowledge graph
    graph: KnowledgeGraph,
    /// Embedding dimension
    embedding_dim: usize,
}

impl std::fmt::Debug for NkefManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NkefManager")
            .field("graph", &self.graph)
            .field("embedding_dim", &self.embedding_dim)
            .finish()
    }
}


impl NkefManager {
    /// Creates a new NKEF manager
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            graph: KnowledgeGraph::with_embedding_dim(embedding_dim),
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

    /// Updates an entity.
    ///
    /// Generates embeddings, records history, and dispatches events automatically.
    pub fn update_entity(&mut self, entity: Entity) {
        self.graph.add_entity(entity);
    }

    /// Performs semantic query using vector similarity search with keyword fallback.
    pub fn query(&self, query: &str, context: &QueryContext) -> Vec<QueryResult> {
        self.graph.search(query, context.max_results)
    }

    /// Performs vector similarity query with an explicit query embedding.
    pub fn vector_query(&self, query_embedding: &[f32], max_results: usize) -> Vec<QueryResult> {
        self.graph.vector_search(query_embedding, max_results)
    }

    /// Retrieves context for RAG (backward compatible)
    pub fn retrieve_context(&self, prompt: &str, max_tokens: u32) -> RetrievedContext {
        self.graph.generate_context(prompt, max_tokens)
    }

    /// Retrieves context for RAG with full configuration control.
    pub fn retrieve_context_with_config(
        &self,
        prompt: &str,
        config: RagConfig,
    ) -> BuiltContext {
        self.graph.generate_context_with_config(prompt, config)
    }

    /// Returns the configured embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Rebuilds all entity embeddings from the current knowledge graph state.
    pub fn rebuild_embeddings(&mut self) {
        self.graph.rebuild_embeddings();
    }

    /// Registers an event handler for a specific event kind.
    ///
    /// Returns a handler ID that can be used to remove the handler later.
    pub fn on_event(&mut self, kind: KnowledgeEventKind, handler: EventHandler) -> HandlerId {
        self.graph.event_bus_mut().on(kind, handler)
    }

    /// Adds a temporal relationship to the knowledge graph.
    pub fn add_temporal_relationship(&mut self, relationship: TemporalRelationship) {
        self.graph.add_temporal_relationship(relationship);
    }

    /// Gets the entity history for a given entity.
    pub fn get_entity_history(&self, entity_id: &str) -> Option<&EntityHistory> {
        self.graph.get_entity_history(entity_id)
    }

    /// Gets the entity state at a historical point in time.
    pub fn get_entity_state_at(
        &self,
        entity_id: &str,
        timestamp: Timestamp,
    ) -> Option<HashMap<String, String>> {
        self.graph.get_entity_state_at(entity_id, timestamp)
    }

    /// Drains the event queue for batch processing.
    pub fn drain_events(&mut self) -> Vec<KnowledgeEvent> {
        self.graph.event_bus_mut().drain_queue()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ──────────────────────────────────────────────────────────────────
    // Backward-compatible tests (exact same as original)
    // ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_entity_creation() {
        let entity = Entity::new("gnb-001", EntityType::Gnb)
            .with_property("location", "Building A")
            .with_property("status", "active");

        assert_eq!(entity.id, "gnb-001");
        assert_eq!(entity.entity_type, EntityType::Gnb);
        assert_eq!(
            entity.properties.get("location"),
            Some(&"Building A".to_string())
        );
    }

    #[test]
    fn test_knowledge_graph() {
        let mut graph = KnowledgeGraph::new();

        graph.add_entity(
            Entity::new("gnb-001", EntityType::Gnb).with_property("status", "active"),
        );
        graph.add_entity(
            Entity::new("ue-001", EntityType::Ue).with_property("imsi", "12345"),
        );

        graph.add_relationship(Relationship::new("ue-001", "gnb-001", "connected_to"));

        assert_eq!(graph.entity_count(), 2);
        assert_eq!(graph.relationship_count(), 1);

        let gnbs = graph.get_entities_by_type(EntityType::Gnb);
        assert_eq!(gnbs.len(), 1);
    }

    #[test]
    fn test_search() {
        let mut graph = KnowledgeGraph::new();

        graph.add_entity(
            Entity::new("gnb-001", EntityType::Gnb).with_property("name", "Main Tower"),
        );
        graph.add_entity(
            Entity::new("gnb-002", EntityType::Gnb).with_property("name", "Secondary"),
        );
        graph.add_entity(
            Entity::new("ue-001", EntityType::Ue).with_property("name", "Test UE"),
        );

        let results = graph.search("tower", 10);
        assert!(!results.is_empty());
        // The vector search or keyword search should find "Main Tower"
        assert!(results.iter().any(|r| r.entity.id == "gnb-001"));
    }

    #[test]
    fn test_context_generation() {
        let mut graph = KnowledgeGraph::new();

        graph.add_entity(
            Entity::new("gnb-001", EntityType::Gnb)
                .with_property("status", "active")
                .with_property("load", "75%"),
        );

        let context = graph.generate_context("gnb", 1000);
        assert!(!context.context.is_empty());
        assert!(!context.sources.is_empty());
    }

    #[test]
    fn test_nkef_manager() {
        let mut manager = NkefManager::new(384);

        manager.update_entity(
            Entity::new("cell-001", EntityType::Cell).with_property("pci", "100"),
        );

        let context = QueryContext {
            intent: "find_cell".to_string(),
            filters: HashMap::new(),
            max_results: 10,
        };

        let results = manager.query("cell", &context);
        assert_eq!(results.len(), 1);
    }

    // ──────────────────────────────────────────────────────────────────
    // New feature tests
    // ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_vector_similarity_search() {
        let mut graph = KnowledgeGraph::with_embedding_dim(128);

        // Add several entities
        graph.add_entity(
            Entity::new("gnb-001", EntityType::Gnb)
                .with_property("name", "Main Tower Base Station")
                .with_property("status", "active"),
        );
        graph.add_entity(
            Entity::new("gnb-002", EntityType::Gnb)
                .with_property("name", "Secondary Base Station")
                .with_property("status", "active"),
        );
        graph.add_entity(
            Entity::new("ue-001", EntityType::Ue)
                .with_property("name", "Mobile User Equipment")
                .with_property("imsi", "12345"),
        );

        // Rebuild embeddings for proper TF-IDF vocabulary
        graph.rebuild_embeddings();

        // Search should return results based on vector similarity
        let results = graph.search("base station tower", 10);
        assert!(!results.is_empty());

        // Direct vector search
        let query_emb = graph.embedder().embed_query("base station");
        let vec_results = graph.vector_search(&query_emb, 2);
        assert!(!vec_results.is_empty());
    }

    #[test]
    fn test_embedding_generation() {
        let mut graph = KnowledgeGraph::with_embedding_dim(64);

        graph.add_entity(
            Entity::new("gnb-001", EntityType::Gnb)
                .with_property("name", "Tower Alpha"),
        );

        // Entity should have an embedding automatically generated
        let entity = graph.get_entity("gnb-001").expect("entity should exist");
        assert!(entity.embedding.is_some());
        assert_eq!(entity.embedding.as_ref().map(std::vec::Vec::len), Some(64));

        // Vector index should contain the entity
        assert!(graph.vector_index().contains("gnb-001"));
    }

    #[test]
    fn test_rebuild_embeddings() {
        let mut graph = KnowledgeGraph::with_embedding_dim(64);

        graph.add_entity(
            Entity::new("gnb-001", EntityType::Gnb).with_property("name", "Alpha"),
        );
        graph.add_entity(
            Entity::new("gnb-002", EntityType::Gnb).with_property("name", "Beta"),
        );

        // Rebuild to get proper TF-IDF vocabulary
        graph.rebuild_embeddings();

        assert_eq!(graph.vector_index().len(), 2);
        assert!(graph.embedder().vocab_size() > 0);
    }

    #[test]
    fn test_temporal_relationship_in_graph() {
        let mut graph = KnowledgeGraph::new();

        graph.add_entity(
            Entity::new("ue-001", EntityType::Ue).with_property("status", "connected"),
        );
        graph.add_entity(
            Entity::new("gnb-001", EntityType::Gnb).with_property("status", "active"),
        );

        graph.add_temporal_relationship(TemporalRelationship::with_validity(
            "ue-001",
            "gnb-001",
            "connected_to",
            100,
            Some(200),
        ));
        graph.add_temporal_relationship(TemporalRelationship::with_validity(
            "ue-001",
            "gnb-001",
            "connected_to",
            200,
            None,
        ));

        assert_eq!(graph.temporal_relationship_count(), 2);

        // At time 150, first relationship is active
        let rels = graph.get_temporal_relationships_at("ue-001", 150);
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].valid_from, 100);

        // At time 250, second relationship is active
        let rels = graph.get_temporal_relationships_at("ue-001", 250);
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].valid_from, 200);
    }

    #[test]
    fn test_entity_history_tracking() {
        let mut graph = KnowledgeGraph::new();

        // Create entity
        graph.add_entity(
            Entity::new("gnb-001", EntityType::Gnb).with_property("status", "active"),
        );

        // Update entity
        graph.add_entity(
            Entity::new("gnb-001", EntityType::Gnb).with_property("status", "degraded"),
        );

        // Check history
        let history = graph.get_entity_history("gnb-001").expect("should have history");
        assert!(history.change_count() >= 2); // at least initial + update

        let status_changes = history.changes_for_key("status");
        assert!(status_changes.len() >= 2);
    }

    #[test]
    fn test_event_dispatching() {
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::sync::Arc;

        let mut graph = KnowledgeGraph::new();

        let create_count = Arc::new(AtomicU64::new(0));
        let update_count = Arc::new(AtomicU64::new(0));
        let remove_count = Arc::new(AtomicU64::new(0));
        let rel_count = Arc::new(AtomicU64::new(0));

        {
            let cc = create_count.clone();
            graph.event_bus_mut().on(
                KnowledgeEventKind::EntityCreated,
                Box::new(move |_| {
                    cc.fetch_add(1, Ordering::SeqCst);
                }),
            );
        }
        {
            let uc = update_count.clone();
            graph.event_bus_mut().on(
                KnowledgeEventKind::EntityUpdated,
                Box::new(move |_| {
                    uc.fetch_add(1, Ordering::SeqCst);
                }),
            );
        }
        {
            let rc = remove_count.clone();
            graph.event_bus_mut().on(
                KnowledgeEventKind::EntityRemoved,
                Box::new(move |_| {
                    rc.fetch_add(1, Ordering::SeqCst);
                }),
            );
        }
        {
            let rlc = rel_count.clone();
            graph.event_bus_mut().on(
                KnowledgeEventKind::RelationshipAdded,
                Box::new(move |_| {
                    rlc.fetch_add(1, Ordering::SeqCst);
                }),
            );
        }

        // Create
        graph.add_entity(Entity::new("gnb-001", EntityType::Gnb));
        assert_eq!(create_count.load(Ordering::SeqCst), 1);

        // Update
        graph.add_entity(
            Entity::new("gnb-001", EntityType::Gnb).with_property("status", "active"),
        );
        assert_eq!(update_count.load(Ordering::SeqCst), 1);

        // Add relationship
        graph.add_relationship(Relationship::new("gnb-001", "ue-001", "serves"));
        assert_eq!(rel_count.load(Ordering::SeqCst), 1);

        // Remove (also triggers relationship removal events)
        graph.remove_entity("gnb-001");
        assert_eq!(remove_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_event_queue_batch_processing() {
        let mut manager = NkefManager::new(64);

        manager.update_entity(Entity::new("a", EntityType::Gnb));
        manager.update_entity(Entity::new("b", EntityType::Ue));
        manager.update_entity(Entity::new("c", EntityType::Cell));

        let events = manager.drain_events();
        assert_eq!(events.len(), 3);
        assert!(events.iter().all(|e| e.kind == KnowledgeEventKind::EntityCreated));
    }

    #[test]
    fn test_improved_rag_context() {
        let mut graph = KnowledgeGraph::with_embedding_dim(64);

        graph.add_entity(
            Entity::new("gnb-001", EntityType::Gnb)
                .with_property("status", "active")
                .with_property("load", "75%"),
        );
        graph.add_entity(
            Entity::new("ue-001", EntityType::Ue)
                .with_property("imsi", "12345"),
        );
        graph.add_relationship(Relationship::new("ue-001", "gnb-001", "connected_to"));

        let config = RagConfig::default()
            .with_max_tokens(1000)
            .with_format(ContextFormat::Markdown);

        let context = graph.generate_context_with_config("gnb", config);
        assert!(!context.text.is_empty());
        assert!(context.entry_count > 0);
    }

    #[test]
    fn test_rag_context_json_format() {
        let mut graph = KnowledgeGraph::with_embedding_dim(64);

        graph.add_entity(
            Entity::new("gnb-001", EntityType::Gnb)
                .with_property("status", "active"),
        );

        let config = RagConfig::default()
            .with_max_tokens(2000)
            .with_format(ContextFormat::Json);

        let context = graph.generate_context_with_config("gnb", config);
        let parsed: Result<serde_json::Value, _> = serde_json::from_str(&context.text);
        assert!(parsed.is_ok());
    }

    #[test]
    fn test_manager_temporal_and_history() {
        let mut manager = NkefManager::new(64);

        manager.update_entity(
            Entity::new("gnb-001", EntityType::Gnb).with_property("status", "active"),
        );

        manager.add_temporal_relationship(TemporalRelationship::with_validity(
            "ue-001",
            "gnb-001",
            "connected_to",
            100,
            None,
        ));

        // Update entity
        manager.update_entity(
            Entity::new("gnb-001", EntityType::Gnb).with_property("status", "degraded"),
        );

        let history = manager.get_entity_history("gnb-001");
        assert!(history.is_some());
    }

    #[test]
    fn test_entity_remove_cleans_up() {
        let mut graph = KnowledgeGraph::with_embedding_dim(64);

        graph.add_entity(Entity::new("gnb-001", EntityType::Gnb));
        graph.add_entity(Entity::new("ue-001", EntityType::Ue));
        graph.add_relationship(Relationship::new("ue-001", "gnb-001", "connected_to"));
        graph.add_temporal_relationship(TemporalRelationship::with_validity(
            "ue-001",
            "gnb-001",
            "served_by",
            100,
            None,
        ));

        assert_eq!(graph.entity_count(), 2);
        assert_eq!(graph.relationship_count(), 1);
        assert!(graph.vector_index().contains("gnb-001"));

        graph.remove_entity("gnb-001");

        assert_eq!(graph.entity_count(), 1);
        assert_eq!(graph.relationship_count(), 0);
        assert!(!graph.vector_index().contains("gnb-001"));
        // Temporal relationships for gnb-001 should be expired
        assert_eq!(graph.active_temporal_relationship_count(), 0);
    }

    #[test]
    fn test_keyword_search_still_works() {
        let mut graph = KnowledgeGraph::new();

        graph.add_entity(
            Entity::new("gnb-001", EntityType::Gnb).with_property("name", "Main Tower"),
        );

        let results = graph.keyword_search("tower", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].entity.id, "gnb-001");
    }

    #[test]
    fn test_with_embedding_preserves_provided_embedding() {
        let mut graph = KnowledgeGraph::with_embedding_dim(3);
        let custom_emb = vec![0.5, 0.3, 0.1];

        graph.add_entity(
            Entity::new("gnb-001", EntityType::Gnb)
                .with_embedding(custom_emb.clone()),
        );

        let entity = graph.get_entity("gnb-001").expect("entity exists");
        assert_eq!(entity.embedding.as_ref(), Some(&custom_emb));
    }

    #[test]
    fn test_batch_vector_search() {
        let mut graph = KnowledgeGraph::with_embedding_dim(64);

        graph.add_entity(
            Entity::new("gnb-001", EntityType::Gnb).with_property("name", "Alpha Tower"),
        );
        graph.add_entity(
            Entity::new("gnb-002", EntityType::Gnb).with_property("name", "Beta Tower"),
        );
        graph.add_entity(
            Entity::new("ue-001", EntityType::Ue).with_property("name", "User Phone"),
        );

        graph.rebuild_embeddings();

        let queries = vec![
            graph.embedder().embed_query("tower"),
            graph.embedder().embed_query("user phone"),
        ];

        let batch_results = graph.vector_index().batch_search(&queries, 2);
        assert_eq!(batch_results.len(), 2);
    }

    #[test]
    fn test_graph_entity_iterators() {
        let mut graph = KnowledgeGraph::new();

        graph.add_entity(Entity::new("a", EntityType::Gnb));
        graph.add_entity(Entity::new("b", EntityType::Ue));

        let ids: Vec<&str> = graph.entity_ids();
        assert_eq!(ids.len(), 2);

        let entities: Vec<&Entity> = graph.entities().collect();
        assert_eq!(entities.len(), 2);
    }
}
