//! Agent <-> NKEF Cross-Crate Integration (A19.8)
//!
//! Enables AI agents to query and update the NKEF knowledge graph.
//! Agents can search for network knowledge, record their intent
//! execution results, and use knowledge-driven reasoning to make
//! better network management decisions.

use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use nextgsim_nkef::{Entity, EntityType, KnowledgeGraph, Relationship};

/// Configuration for the Agent-NKEF bridge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentNkefBridgeConfig {
    /// Whether to auto-record intent results in the knowledge graph
    pub record_intents: bool,
    /// Entity ID prefix for agent-sourced entities
    pub entity_prefix: String,
    /// Maximum search results to return
    pub max_search_results: usize,
}

impl Default for AgentNkefBridgeConfig {
    fn default() -> Self {
        Self {
            record_intents: true,
            entity_prefix: "agent".to_string(),
            max_search_results: 50,
        }
    }
}

/// Bridge between Agent framework and NKEF knowledge graph
///
/// Provides methods for agents to:
/// - Query the knowledge graph for network topology and state information
/// - Record intent execution results as knowledge graph entities
/// - Search for entities matching specific criteria
pub struct AgentNkefBridge {
    /// Configuration
    config: AgentNkefBridgeConfig,
    /// Number of intents recorded
    recorded_intents: u64,
    /// Number of queries executed
    query_count: u64,
}

impl AgentNkefBridge {
    /// Creates a new bridge with the given configuration
    pub fn new(config: AgentNkefBridgeConfig) -> Self {
        Self {
            config,
            recorded_intents: 0,
            query_count: 0,
        }
    }

    /// Records an agent's identity in the knowledge graph
    pub fn register_agent(
        &self,
        agent_id: &str,
        role: &str,
        knowledge_graph: &mut KnowledgeGraph,
    ) {
        let entity_id = format!("{}-{}", self.config.entity_prefix, agent_id);
        let entity = Entity::new(&entity_id, EntityType::Agent)
            .with_property("agent_id", agent_id)
            .with_property("role", role)
            .with_property("status", "active");

        knowledge_graph.add_entity(entity);
        info!("AgentNkefBridge: registered agent {} in KG", agent_id);
    }

    /// Records an intent and its result in the knowledge graph
    pub fn record_intent_result(
        &mut self,
        agent_id: &str,
        intent_type: &str,
        target: Option<&str>,
        success: bool,
        knowledge_graph: &mut KnowledgeGraph,
    ) {
        if !self.config.record_intents {
            return;
        }

        let intent_id = format!(
            "{}-intent-{}-{}",
            self.config.entity_prefix,
            intent_type,
            timestamp_now()
        );

        let entity = Entity::new(&intent_id, EntityType::Intent)
            .with_property("intent_type", intent_type)
            .with_property("success", success.to_string())
            .with_property("timestamp_ms", timestamp_now().to_string());

        knowledge_graph.add_entity(entity);

        // Link intent to the agent that issued it
        let agent_entity_id = format!("{}-{}", self.config.entity_prefix, agent_id);
        let rel = Relationship::new(&agent_entity_id, &intent_id, "issued_intent");
        knowledge_graph.add_relationship(rel);

        // Link intent to its target if specified
        if let Some(target_id) = target {
            let target_rel = Relationship::new(&intent_id, target_id, "targets");
            knowledge_graph.add_relationship(target_rel);
        }

        self.recorded_intents += 1;
        debug!(
            "AgentNkefBridge: recorded intent {} (type={}, success={})",
            intent_id, intent_type, success
        );
    }

    /// Queries the knowledge graph for entities by type
    pub fn query_by_type(
        &mut self,
        entity_type: EntityType,
        knowledge_graph: &KnowledgeGraph,
    ) -> Vec<Entity> {
        self.query_count += 1;
        knowledge_graph
            .get_entities_by_type(entity_type)
            .into_iter()
            .take(self.config.max_search_results)
            .cloned()
            .collect()
    }

    /// Queries the knowledge graph for a specific entity by ID
    pub fn query_entity(
        &mut self,
        entity_id: &str,
        knowledge_graph: &KnowledgeGraph,
    ) -> Option<Entity> {
        self.query_count += 1;
        knowledge_graph.get_entity(entity_id).cloned()
    }

    /// Queries the knowledge graph for relationships of a specific entity
    pub fn query_relationships(
        &mut self,
        entity_id: &str,
        knowledge_graph: &KnowledgeGraph,
    ) -> Vec<Relationship> {
        self.query_count += 1;
        knowledge_graph
            .get_relationships(entity_id)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Searches the knowledge graph for entities matching a text query
    /// using the NKEF semantic search capability
    pub fn semantic_search(
        &mut self,
        query: &str,
        knowledge_graph: &KnowledgeGraph,
    ) -> Vec<Entity> {
        self.query_count += 1;
        knowledge_graph
            .search(query, self.config.max_search_results)
            .into_iter()
            .map(|qr| qr.entity)
            .collect()
    }

    /// Returns the number of intents recorded
    pub fn recorded_intents(&self) -> u64 {
        self.recorded_intents
    }

    /// Returns the number of queries executed
    pub fn query_count(&self) -> u64 {
        self.query_count
    }

    /// Returns the bridge configuration
    pub fn config(&self) -> &AgentNkefBridgeConfig {
        &self.config
    }
}

/// Gets current timestamp in milliseconds
fn timestamp_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_creation() {
        let bridge = AgentNkefBridge::new(AgentNkefBridgeConfig::default());
        assert_eq!(bridge.recorded_intents(), 0);
        assert_eq!(bridge.query_count(), 0);
    }

    #[test]
    fn test_register_agent() {
        let bridge = AgentNkefBridge::new(AgentNkefBridgeConfig::default());
        let mut kg = KnowledgeGraph::new();

        bridge.register_agent("agent-1", "cell_level", &mut kg);
        assert!(kg.entity_count() > 0);

        let entity = kg.get_entity("agent-agent-1");
        assert!(entity.is_some());
        assert_eq!(entity.unwrap().entity_type, EntityType::Agent);
    }

    #[test]
    fn test_record_intent_result() {
        let mut bridge = AgentNkefBridge::new(AgentNkefBridgeConfig::default());
        let mut kg = KnowledgeGraph::new();

        bridge.register_agent("agent-1", "cell_level", &mut kg);
        bridge.record_intent_result("agent-1", "optimize_resources", None, true, &mut kg);

        assert_eq!(bridge.recorded_intents(), 1);
        // Should have 2 entities: agent + intent
        assert!(kg.entity_count() >= 2);
    }

    #[test]
    fn test_record_intents_disabled() {
        let mut bridge = AgentNkefBridge::new(AgentNkefBridgeConfig {
            record_intents: false,
            ..Default::default()
        });
        let mut kg = KnowledgeGraph::new();

        bridge.record_intent_result("agent-1", "query", None, true, &mut kg);
        assert_eq!(bridge.recorded_intents(), 0);
    }

    #[test]
    fn test_query_by_type() {
        let mut bridge = AgentNkefBridge::new(AgentNkefBridgeConfig::default());
        let mut kg = KnowledgeGraph::new();

        kg.add_entity(Entity::new("gnb-1", EntityType::Gnb));
        kg.add_entity(Entity::new("gnb-2", EntityType::Gnb));
        kg.add_entity(Entity::new("ue-1", EntityType::Ue));

        let gnbs = bridge.query_by_type(EntityType::Gnb, &kg);
        assert_eq!(gnbs.len(), 2);
        assert_eq!(bridge.query_count(), 1);
    }

    #[test]
    fn test_query_entity() {
        let mut bridge = AgentNkefBridge::new(AgentNkefBridgeConfig::default());
        let mut kg = KnowledgeGraph::new();

        kg.add_entity(
            Entity::new("gnb-1", EntityType::Gnb).with_property("status", "active"),
        );

        let entity = bridge.query_entity("gnb-1", &kg);
        assert!(entity.is_some());
        assert_eq!(
            entity.unwrap().properties.get("status"),
            Some(&"active".to_string())
        );
    }

    #[test]
    fn test_query_relationships() {
        let mut bridge = AgentNkefBridge::new(AgentNkefBridgeConfig::default());
        let mut kg = KnowledgeGraph::new();

        kg.add_entity(Entity::new("gnb-1", EntityType::Gnb));
        kg.add_entity(Entity::new("ue-1", EntityType::Ue));
        kg.add_relationship(Relationship::new("gnb-1", "ue-1", "serves"));

        let rels = bridge.query_relationships("gnb-1", &kg);
        assert_eq!(rels.len(), 1);
    }
}
