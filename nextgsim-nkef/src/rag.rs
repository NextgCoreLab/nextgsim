//! Improved RAG (Retrieval Augmented Generation) context building
//!
//! Provides structured context generation for LLM integration, including
//! relevance-scored context entries with relationships, size limits,
//! and formatted markdown output.

use std::collections::HashMap;

use crate::{Entity, EntityType, Relationship};

/// A single context entry with relevance scoring.
#[derive(Debug, Clone)]
pub struct ContextEntry {
    /// Entity ID this entry is about
    pub entity_id: String,
    /// Entity type
    pub entity_type: EntityType,
    /// Relevance score (0.0 to 1.0)
    pub relevance: f32,
    /// Entity properties
    pub properties: HashMap<String, String>,
    /// Relationships involving this entity (as formatted strings)
    pub relationships: Vec<RelationshipContext>,
}

/// A relationship context entry for inclusion in RAG context.
#[derive(Debug, Clone)]
pub struct RelationshipContext {
    /// Source entity ID
    pub source_id: String,
    /// Target entity ID
    pub target_id: String,
    /// Relationship type
    pub relation_type: String,
    /// Relationship properties
    pub properties: HashMap<String, String>,
}

impl RelationshipContext {
    /// Creates a new relationship context from a `Relationship`.
    pub fn from_relationship(rel: &Relationship) -> Self {
        Self {
            source_id: rel.source_id.clone(),
            target_id: rel.target_id.clone(),
            relation_type: rel.relation_type.clone(),
            properties: rel.properties.clone(),
        }
    }
}

/// Configuration for RAG context generation.
#[derive(Debug, Clone)]
pub struct RagConfig {
    /// Maximum number of context entries
    pub max_entries: usize,
    /// Maximum approximate token count for the generated context.
    /// Tokens are estimated as `chars / 4` (a common heuristic).
    pub max_tokens: u32,
    /// Minimum relevance score to include an entry
    pub min_relevance: f32,
    /// Whether to include relationships in the context
    pub include_relationships: bool,
    /// Whether to include property details
    pub include_properties: bool,
    /// Output format
    pub format: ContextFormat,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            max_entries: 10,
            max_tokens: 2000,
            min_relevance: 0.0,
            include_relationships: true,
            include_properties: true,
            format: ContextFormat::Markdown,
        }
    }
}

impl RagConfig {
    /// Creates a new config with the given max tokens.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Sets the maximum number of context entries.
    pub fn with_max_entries(mut self, max_entries: usize) -> Self {
        self.max_entries = max_entries;
        self
    }

    /// Sets the minimum relevance score.
    pub fn with_min_relevance(mut self, min_relevance: f32) -> Self {
        self.min_relevance = min_relevance;
        self
    }

    /// Sets whether to include relationships.
    pub fn with_relationships(mut self, include: bool) -> Self {
        self.include_relationships = include;
        self
    }

    /// Sets the output format.
    pub fn with_format(mut self, format: ContextFormat) -> Self {
        self.format = format;
        self
    }
}

/// Output format for RAG context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextFormat {
    /// Structured Markdown
    Markdown,
    /// Plain text
    PlainText,
    /// JSON format
    Json,
}

/// Builder for RAG context from search results.
pub struct ContextBuilder {
    entries: Vec<ContextEntry>,
    config: RagConfig,
}

impl ContextBuilder {
    /// Creates a new context builder with default config.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            config: RagConfig::default(),
        }
    }

    /// Creates a new context builder with the given config.
    pub fn with_config(config: RagConfig) -> Self {
        Self {
            entries: Vec::new(),
            config,
        }
    }

    /// Adds a context entry for an entity with its relationships.
    pub fn add_entity(
        &mut self,
        entity: &Entity,
        relevance: f32,
        relationships: &[&Relationship],
    ) {
        if relevance < self.config.min_relevance {
            return;
        }

        let rel_contexts: Vec<RelationshipContext> = if self.config.include_relationships {
            relationships
                .iter()
                .map(|r| RelationshipContext::from_relationship(r))
                .collect()
        } else {
            Vec::new()
        };

        self.entries.push(ContextEntry {
            entity_id: entity.id.clone(),
            entity_type: entity.entity_type,
            relevance,
            properties: entity.properties.clone(),
            relationships: rel_contexts,
        });
    }

    /// Sorts entries by relevance (highest first) and truncates to `max_entries`.
    fn finalize_entries(&mut self) {
        self.entries.sort_by(|a, b| {
            b.relevance
                .partial_cmp(&a.relevance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.entries.truncate(self.config.max_entries);
    }

    /// Builds the final context string.
    ///
    /// Applies relevance filtering, sorting, and token limiting.
    pub fn build(&mut self) -> BuiltContext {
        self.finalize_entries();

        let max_chars = (self.config.max_tokens * 4) as usize;

        match self.config.format {
            ContextFormat::Markdown => self.build_markdown(max_chars),
            ContextFormat::PlainText => self.build_plain_text(max_chars),
            ContextFormat::Json => self.build_json(max_chars),
        }
    }

    /// Builds markdown-formatted context.
    fn build_markdown(&self, max_chars: usize) -> BuiltContext {
        let mut output = String::new();
        let mut sources = Vec::new();
        let mut included_count = 0;
        let mut total_relevance = 0.0_f32;

        output.push_str("## Network Knowledge Context\n\n");

        for entry in &self.entries {
            let mut entry_text = String::new();

            entry_text.push_str(&format!(
                "### {} ({:?})\n",
                entry.entity_id, entry.entity_type
            ));
            entry_text.push_str(&format!("**Relevance:** {:.2}\n\n", entry.relevance));

            if self.config.include_properties && !entry.properties.is_empty() {
                entry_text.push_str("**Properties:**\n");
                let mut props: Vec<(&String, &String)> = entry.properties.iter().collect();
                props.sort_by_key(|(k, _)| *k);
                for (key, value) in props {
                    entry_text.push_str(&format!("- **{key}:** {value}\n"));
                }
                entry_text.push('\n');
            }

            if !entry.relationships.is_empty() {
                entry_text.push_str("**Relationships:**\n");
                for rel in &entry.relationships {
                    let direction = if rel.source_id == entry.entity_id {
                        format!("{} --[{}]--> {}", rel.source_id, rel.relation_type, rel.target_id)
                    } else {
                        format!("{} <--[{}]-- {}", entry.entity_id, rel.relation_type, rel.source_id)
                    };
                    entry_text.push_str(&format!("- {direction}\n"));

                    if !rel.properties.is_empty() {
                        let mut rel_props: Vec<(&String, &String)> = rel.properties.iter().collect();
                        rel_props.sort_by_key(|(k, _)| *k);
                        for (k, v) in rel_props {
                            entry_text.push_str(&format!("  - {k}: {v}\n"));
                        }
                    }
                }
                entry_text.push('\n');
            }

            // Check token limit
            if output.len() + entry_text.len() > max_chars && included_count > 0 {
                break;
            }

            output.push_str(&entry_text);
            sources.push(entry.entity_id.clone());
            total_relevance += entry.relevance;
            included_count += 1;
        }

        let confidence = if included_count > 0 {
            total_relevance / included_count as f32
        } else {
            0.0
        };

        BuiltContext {
            text: output,
            sources,
            confidence,
            entry_count: included_count,
            truncated: included_count < self.entries.len(),
        }
    }

    /// Builds plain text context.
    fn build_plain_text(&self, max_chars: usize) -> BuiltContext {
        let mut output = String::new();
        let mut sources = Vec::new();
        let mut included_count = 0;
        let mut total_relevance = 0.0_f32;

        output.push_str("Network Knowledge Context\n");
        output.push_str(&"=".repeat(40));
        output.push('\n');

        for entry in &self.entries {
            let mut entry_text = String::new();

            entry_text.push_str(&format!(
                "\n{} ({:?}) [relevance: {:.2}]\n",
                entry.entity_id, entry.entity_type, entry.relevance
            ));

            if self.config.include_properties && !entry.properties.is_empty() {
                let mut props: Vec<(&String, &String)> = entry.properties.iter().collect();
                props.sort_by_key(|(k, _)| *k);
                for (key, value) in props {
                    entry_text.push_str(&format!("  {key}: {value}\n"));
                }
            }

            if !entry.relationships.is_empty() {
                entry_text.push_str("  Relationships:\n");
                for rel in &entry.relationships {
                    entry_text.push_str(&format!(
                        "    {} --[{}]--> {}\n",
                        rel.source_id, rel.relation_type, rel.target_id
                    ));
                }
            }

            if output.len() + entry_text.len() > max_chars && included_count > 0 {
                break;
            }

            output.push_str(&entry_text);
            sources.push(entry.entity_id.clone());
            total_relevance += entry.relevance;
            included_count += 1;
        }

        let confidence = if included_count > 0 {
            total_relevance / included_count as f32
        } else {
            0.0
        };

        BuiltContext {
            text: output,
            sources,
            confidence,
            entry_count: included_count,
            truncated: included_count < self.entries.len(),
        }
    }

    /// Builds JSON-formatted context.
    fn build_json(&self, max_chars: usize) -> BuiltContext {
        let mut entries_json = Vec::new();
        let mut sources = Vec::new();
        let mut included_count = 0;
        let mut total_relevance = 0.0_f32;
        let mut current_len = 0;

        for entry in &self.entries {
            let mut entry_map = serde_json::Map::new();
            entry_map.insert(
                "entity_id".to_string(),
                serde_json::Value::String(entry.entity_id.clone()),
            );
            entry_map.insert(
                "entity_type".to_string(),
                serde_json::Value::String(format!("{:?}", entry.entity_type)),
            );
            entry_map.insert(
                "relevance".to_string(),
                serde_json::json!(entry.relevance),
            );

            if self.config.include_properties {
                let props: serde_json::Map<String, serde_json::Value> = entry
                    .properties
                    .iter()
                    .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
                    .collect();
                entry_map.insert(
                    "properties".to_string(),
                    serde_json::Value::Object(props),
                );
            }

            if !entry.relationships.is_empty() {
                let rels: Vec<serde_json::Value> = entry
                    .relationships
                    .iter()
                    .map(|r| {
                        serde_json::json!({
                            "source": r.source_id,
                            "target": r.target_id,
                            "type": r.relation_type,
                        })
                    })
                    .collect();
                entry_map.insert(
                    "relationships".to_string(),
                    serde_json::Value::Array(rels),
                );
            }

            let entry_json = serde_json::to_string(&entry_map).unwrap_or_default();
            if current_len + entry_json.len() > max_chars && included_count > 0 {
                break;
            }

            current_len += entry_json.len();
            entries_json.push(serde_json::Value::Object(entry_map));
            sources.push(entry.entity_id.clone());
            total_relevance += entry.relevance;
            included_count += 1;
        }

        let wrapper = serde_json::json!({
            "context": entries_json,
            "entry_count": included_count,
        });

        let text = serde_json::to_string_pretty(&wrapper).unwrap_or_default();

        let confidence = if included_count > 0 {
            total_relevance / included_count as f32
        } else {
            0.0
        };

        BuiltContext {
            text,
            sources,
            confidence,
            entry_count: included_count,
            truncated: included_count < self.entries.len(),
        }
    }
}

impl Default for ContextBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// The result of building RAG context.
#[derive(Debug, Clone)]
pub struct BuiltContext {
    /// The formatted context text
    pub text: String,
    /// Source entity IDs that contributed to the context
    pub sources: Vec<String>,
    /// Average relevance confidence (0.0 to 1.0)
    pub confidence: f32,
    /// Number of entries included
    pub entry_count: usize,
    /// Whether the context was truncated due to token limits
    pub truncated: bool,
}

impl BuiltContext {
    /// Estimated token count (chars / 4).
    pub fn estimated_tokens(&self) -> u32 {
        (self.text.len() / 4) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entity(id: &str, entity_type: EntityType) -> Entity {
        Entity::new(id, entity_type)
            .with_property("status", "active")
            .with_property("load", "75%")
    }

    fn make_relationship(source: &str, target: &str, rel_type: &str) -> Relationship {
        Relationship::new(source, target, rel_type)
    }

    #[test]
    fn test_context_builder_markdown() {
        let entity = make_entity("gnb-001", EntityType::Gnb);
        let rel = make_relationship("ue-001", "gnb-001", "connected_to");

        let mut builder = ContextBuilder::with_config(RagConfig::default());
        builder.add_entity(&entity, 0.95, &[&rel]);

        let context = builder.build();
        assert!(!context.text.is_empty());
        assert!(context.text.contains("gnb-001"));
        assert!(context.text.contains("Gnb"));
        assert!(context.text.contains("0.95"));
        assert!(context.text.contains("connected_to"));
        assert!(context.text.contains("status"));
        assert_eq!(context.sources.len(), 1);
        assert_eq!(context.entry_count, 1);
        assert!(!context.truncated);
    }

    #[test]
    fn test_context_builder_plain_text() {
        let entity = make_entity("gnb-001", EntityType::Gnb);

        let config = RagConfig::default().with_format(ContextFormat::PlainText);
        let mut builder = ContextBuilder::with_config(config);
        builder.add_entity(&entity, 0.9, &[]);

        let context = builder.build();
        assert!(context.text.contains("gnb-001"));
        assert!(context.text.contains("Gnb"));
    }

    #[test]
    fn test_context_builder_json() {
        let entity = make_entity("gnb-001", EntityType::Gnb);

        let config = RagConfig::default().with_format(ContextFormat::Json);
        let mut builder = ContextBuilder::with_config(config);
        builder.add_entity(&entity, 0.9, &[]);

        let context = builder.build();
        // Should be valid JSON
        let parsed: Result<serde_json::Value, _> = serde_json::from_str(&context.text);
        assert!(parsed.is_ok());
    }

    #[test]
    fn test_context_builder_relevance_filtering() {
        let entity1 = make_entity("gnb-001", EntityType::Gnb);
        let entity2 = make_entity("gnb-002", EntityType::Gnb);

        let config = RagConfig::default().with_min_relevance(0.5);
        let mut builder = ContextBuilder::with_config(config);
        builder.add_entity(&entity1, 0.9, &[]);
        builder.add_entity(&entity2, 0.3, &[]); // below threshold

        let context = builder.build();
        assert_eq!(context.entry_count, 1);
        assert_eq!(context.sources[0], "gnb-001");
    }

    #[test]
    fn test_context_builder_token_limit() {
        let config = RagConfig::default().with_max_tokens(50); // very small limit
        let mut builder = ContextBuilder::with_config(config);

        // Add many entities
        for i in 0..20 {
            let entity = Entity::new(format!("gnb-{i:03}"), EntityType::Gnb)
                .with_property("description", "A very long description that takes up many tokens in the output context string");
            builder.add_entity(&entity, 0.9 - (i as f32 * 0.01), &[]);
        }

        let context = builder.build();
        // Should be truncated due to token limit
        assert!(context.entry_count < 20);
        assert!(context.truncated);
    }

    #[test]
    fn test_context_builder_max_entries() {
        let config = RagConfig::default().with_max_entries(2);
        let mut builder = ContextBuilder::with_config(config);

        for i in 0..5 {
            let entity = make_entity(&format!("gnb-{i:03}"), EntityType::Gnb);
            builder.add_entity(&entity, 0.9 - (i as f32 * 0.1), &[]);
        }

        let context = builder.build();
        assert_eq!(context.entry_count, 2);
        // Should have highest relevance entries
        assert_eq!(context.sources[0], "gnb-000");
        assert_eq!(context.sources[1], "gnb-001");
    }

    #[test]
    fn test_context_builder_sorting() {
        let mut builder = ContextBuilder::new();

        let e1 = make_entity("low", EntityType::Gnb);
        let e2 = make_entity("high", EntityType::Gnb);
        let e3 = make_entity("mid", EntityType::Gnb);

        builder.add_entity(&e1, 0.3, &[]);
        builder.add_entity(&e2, 0.9, &[]);
        builder.add_entity(&e3, 0.6, &[]);

        let context = builder.build();
        assert_eq!(context.sources[0], "high");
        assert_eq!(context.sources[1], "mid");
        assert_eq!(context.sources[2], "low");
    }

    #[test]
    fn test_context_builder_no_relationships() {
        let entity = make_entity("gnb-001", EntityType::Gnb);
        let rel = make_relationship("ue-001", "gnb-001", "connected_to");

        let config = RagConfig::default().with_relationships(false);
        let mut builder = ContextBuilder::with_config(config);
        builder.add_entity(&entity, 0.9, &[&rel]);

        let context = builder.build();
        assert!(!context.text.contains("connected_to"));
    }

    #[test]
    fn test_built_context_estimated_tokens() {
        let context = BuiltContext {
            text: "a".repeat(400),
            sources: vec![],
            confidence: 1.0,
            entry_count: 1,
            truncated: false,
        };
        assert_eq!(context.estimated_tokens(), 100);
    }

    #[test]
    fn test_empty_context() {
        let mut builder = ContextBuilder::new();
        let context = builder.build();
        assert_eq!(context.entry_count, 0);
        assert_eq!(context.confidence, 0.0);
        assert!(context.sources.is_empty());
    }

    #[test]
    fn test_relationship_context_from_relationship() {
        let mut rel = Relationship::new("a", "b", "connected_to");
        rel.properties.insert("quality".to_string(), "high".to_string());

        let rc = RelationshipContext::from_relationship(&rel);
        assert_eq!(rc.source_id, "a");
        assert_eq!(rc.target_id, "b");
        assert_eq!(rc.relation_type, "connected_to");
        assert_eq!(rc.properties.get("quality"), Some(&"high".to_string()));
    }
}
