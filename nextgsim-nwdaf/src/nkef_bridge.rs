//! NWDAF <-> NKEF Cross-Crate Integration (A19.6)
//!
//! Bridges the Network Data Analytics Function with the Network Knowledge
//! Exposure Function. Analytics results produced by NWDAF are pushed into
//! the NKEF knowledge graph as entities and relationships, enabling
//! semantic search and knowledge-driven network operations.

use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::analytics_id::AnalyticsId;
use crate::anlf::AnalyticsResult;
use nextgsim_nkef::{Entity, EntityType, KnowledgeGraph, Relationship};

/// Configuration for the NWDAF-NKEF bridge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NwdafNkefBridgeConfig {
    /// Whether to auto-publish analytics results to the knowledge graph
    pub auto_publish: bool,
    /// Minimum confidence threshold for publishing (0.0 to 1.0)
    pub min_confidence: f32,
    /// Entity ID prefix for NWDAF-sourced entities
    pub entity_prefix: String,
}

impl Default for NwdafNkefBridgeConfig {
    fn default() -> Self {
        Self {
            auto_publish: true,
            min_confidence: 0.5,
            entity_prefix: "nwdaf".to_string(),
        }
    }
}

/// Bridge between NWDAF analytics output and NKEF knowledge graph
///
/// Converts analytics results into knowledge graph entities and
/// relationships, enabling downstream consumers to query analytics
/// knowledge through the NKEF semantic search interface.
pub struct NwdafNkefBridge {
    /// Bridge configuration
    config: NwdafNkefBridgeConfig,
    /// Number of analytics results published
    published_count: u64,
}

impl NwdafNkefBridge {
    /// Creates a new bridge with the given configuration
    pub fn new(config: NwdafNkefBridgeConfig) -> Self {
        Self {
            config,
            published_count: 0,
        }
    }

    /// Publishes an analytics result to the knowledge graph
    ///
    /// Creates an entity representing the analytics result and links it
    /// to the target entity (UE, cell, etc.) via a "has_analytics"
    /// relationship.
    pub fn publish_analytics(
        &mut self,
        result: &AnalyticsResult,
        knowledge_graph: &mut KnowledgeGraph,
    ) {
        if !self.config.auto_publish {
            return;
        }

        if result.confidence < self.config.min_confidence {
            debug!(
                "Skipping analytics publication: confidence {:.2} below threshold {:.2}",
                result.confidence, self.config.min_confidence
            );
            return;
        }

        let entity_id = format!(
            "{}-{:?}-{}",
            self.config.entity_prefix, result.analytics_id, result.timestamp_ms
        );

        // Create entity representing the analytics result
        let entity = Entity::new(&entity_id, EntityType::Analytics)
            .with_property("analytics_id", format!("{:?}", result.analytics_id))
            .with_property("output_type", format!("{:?}", result.output_type))
            .with_property("confidence", format!("{:.4}", result.confidence))
            .with_property("timestamp_ms", result.timestamp_ms.to_string());

        knowledge_graph.add_entity(entity);

        // Link analytics to its target entity if applicable
        let target_entity_id = self.target_entity_id(&result.target);
        if let Some(target_id) = target_entity_id {
            let relationship = Relationship::new(&entity_id, &target_id, "analyzes");
            knowledge_graph.add_relationship(relationship);
        }

        self.published_count += 1;
        info!(
            "Published {:?} analytics to knowledge graph (entity={})",
            result.analytics_id, entity_id
        );
    }

    /// Queries the knowledge graph for analytics entities matching an analytics ID
    pub fn query_analytics_knowledge(
        &self,
        analytics_id: AnalyticsId,
        knowledge_graph: &KnowledgeGraph,
    ) -> Vec<Entity> {
        let analytics_id_str = format!("{analytics_id:?}");
        knowledge_graph
            .entities()
            .filter(|e| {
                e.entity_type == EntityType::Analytics
                    && e.properties
                        .get("analytics_id")
                        .map(|v| v == &analytics_id_str)
                        .unwrap_or(false)
            })
            .cloned()
            .collect()
    }

    /// Returns the number of analytics results published
    pub fn published_count(&self) -> u64 {
        self.published_count
    }

    /// Returns the bridge configuration
    pub fn config(&self) -> &NwdafNkefBridgeConfig {
        &self.config
    }

    /// Derives a NKEF entity ID from an analytics target
    fn target_entity_id(
        &self,
        target: &crate::analytics_id::AnalyticsTarget,
    ) -> Option<String> {
        match target {
            crate::analytics_id::AnalyticsTarget::Ue { ue_id } => {
                Some(format!("ue-{ue_id}"))
            }
            crate::analytics_id::AnalyticsTarget::Cell { cell_id } => {
                Some(format!("cell-{cell_id}"))
            }
            crate::analytics_id::AnalyticsTarget::Slice { snssai } => {
                Some(format!("slice-{snssai}"))
            }
            crate::analytics_id::AnalyticsTarget::Any => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analytics_id::{AnalyticsOutputType, AnalyticsTarget};
    use crate::anlf::AnalyticsPayload;

    fn make_test_result() -> AnalyticsResult {
        AnalyticsResult {
            analytics_id: AnalyticsId::UeMobility,
            target: AnalyticsTarget::Ue { ue_id: 42 },
            output_type: AnalyticsOutputType::Predictions,
            timestamp_ms: 1000,
            confidence: 0.9,
            payload: AnalyticsPayload::UeMobility {
                trajectory: None,
                handover_recommendation: None,
            },
        }
    }

    #[test]
    fn test_bridge_creation() {
        let bridge = NwdafNkefBridge::new(NwdafNkefBridgeConfig::default());
        assert_eq!(bridge.published_count(), 0);
        assert!(bridge.config().auto_publish);
    }

    #[test]
    fn test_publish_analytics() {
        let mut bridge = NwdafNkefBridge::new(NwdafNkefBridgeConfig::default());
        let mut kg = KnowledgeGraph::new();

        let result = make_test_result();
        bridge.publish_analytics(&result, &mut kg);

        assert_eq!(bridge.published_count(), 1);
        assert!(kg.entity_count() > 0);
    }

    #[test]
    fn test_low_confidence_skipped() {
        let mut bridge = NwdafNkefBridge::new(NwdafNkefBridgeConfig {
            min_confidence: 0.95,
            ..Default::default()
        });
        let mut kg = KnowledgeGraph::new();

        let result = make_test_result(); // confidence=0.9, below 0.95
        bridge.publish_analytics(&result, &mut kg);

        assert_eq!(bridge.published_count(), 0);
    }

    #[test]
    fn test_auto_publish_disabled() {
        let mut bridge = NwdafNkefBridge::new(NwdafNkefBridgeConfig {
            auto_publish: false,
            ..Default::default()
        });
        let mut kg = KnowledgeGraph::new();

        bridge.publish_analytics(&make_test_result(), &mut kg);
        assert_eq!(bridge.published_count(), 0);
    }

    #[test]
    fn test_query_analytics_knowledge() {
        let mut bridge = NwdafNkefBridge::new(NwdafNkefBridgeConfig::default());
        let mut kg = KnowledgeGraph::new();

        bridge.publish_analytics(&make_test_result(), &mut kg);

        let results = bridge.query_analytics_knowledge(AnalyticsId::UeMobility, &kg);
        assert_eq!(results.len(), 1);

        let results = bridge.query_analytics_knowledge(AnalyticsId::NfLoad, &kg);
        assert!(results.is_empty());
    }
}
