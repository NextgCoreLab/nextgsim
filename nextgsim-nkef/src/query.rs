//! Graph Query Language for NKEF
//!
//! Provides a structured query language for querying the knowledge graph,
//! inspired by GQL, SPARQL, and Cypher. Supports pattern matching, filtering,
//! and graph traversal operations.

use serde::{Deserialize, Serialize};

use crate::{Entity, EntityType, Relationship};

/// Query operator
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryOperator {
    /// Equality (==)
    Eq,
    /// Inequality (!=)
    Ne,
    /// Greater than (>)
    Gt,
    /// Greater than or equal (>=)
    Gte,
    /// Less than (<)
    Lt,
    /// Less than or equal (<=)
    Lte,
    /// String contains
    Contains,
    /// String starts with
    StartsWith,
    /// String ends with
    EndsWith,
    /// In list
    In,
}

/// Query filter condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryFilter {
    /// Property name to filter on
    pub property: String,
    /// Operator to apply
    pub operator: QueryOperator,
    /// Value to compare against
    pub value: String,
}

impl QueryFilter {
    /// Creates a new query filter
    pub fn new(property: String, operator: QueryOperator, value: String) -> Self {
        Self {
            property,
            operator,
            value,
        }
    }

    /// Evaluates the filter against an entity
    pub fn matches(&self, entity: &Entity) -> bool {
        let prop_value = match entity.properties.get(&self.property) {
            Some(v) => v,
            None => return false,
        };

        match self.operator {
            QueryOperator::Eq => prop_value == &self.value,
            QueryOperator::Ne => prop_value != &self.value,
            QueryOperator::Contains => prop_value.contains(&self.value),
            QueryOperator::StartsWith => prop_value.starts_with(&self.value),
            QueryOperator::EndsWith => prop_value.ends_with(&self.value),
            QueryOperator::Gt | QueryOperator::Gte | QueryOperator::Lt | QueryOperator::Lte => {
                // Try numeric comparison
                if let (Ok(prop_num), Ok(val_num)) =
                    (prop_value.parse::<f64>(), self.value.parse::<f64>())
                {
                    match self.operator {
                        QueryOperator::Gt => prop_num > val_num,
                        QueryOperator::Gte => prop_num >= val_num,
                        QueryOperator::Lt => prop_num < val_num,
                        QueryOperator::Lte => prop_num <= val_num,
                        _ => false,
                    }
                } else {
                    // Fallback to string comparison
                    match self.operator {
                        QueryOperator::Gt => prop_value > &self.value,
                        QueryOperator::Gte => prop_value >= &self.value,
                        QueryOperator::Lt => prop_value < &self.value,
                        QueryOperator::Lte => prop_value <= &self.value,
                        _ => false,
                    }
                }
            }
            QueryOperator::In => {
                // value should be comma-separated list
                self.value.split(',').any(|v| v.trim() == prop_value)
            }
        }
    }
}

/// Graph pattern to match
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphPattern {
    /// Entity type to match (None = any type)
    pub entity_type: Option<EntityType>,
    /// Filters to apply to entity
    pub filters: Vec<QueryFilter>,
    /// Relationship pattern (if matching connected entities)
    pub relationship: Option<RelationshipPattern>,
}

/// Relationship pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipPattern {
    /// Relationship type to match (None = any type)
    pub relation_type: Option<String>,
    /// Direction: "outgoing", "incoming", or "any"
    pub direction: String,
    /// Target entity pattern
    pub target: Box<GraphPattern>,
}

/// Graph query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQuery {
    /// Patterns to match
    pub patterns: Vec<GraphPattern>,
    /// Optional limit on results
    pub limit: Option<usize>,
    /// Optional offset for pagination
    pub offset: Option<usize>,
    /// Order by property (optional)
    pub order_by: Option<String>,
    /// Order direction: "asc" or "desc"
    pub order_direction: String,
}

impl GraphQuery {
    /// Creates a new graph query
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
            limit: None,
            offset: None,
            order_by: None,
            order_direction: "asc".to_string(),
        }
    }

    /// Adds a pattern to the query
    pub fn pattern(mut self, pattern: GraphPattern) -> Self {
        self.patterns.push(pattern);
        self
    }

    /// Sets the result limit
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Sets the offset for pagination
    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Sets ordering
    pub fn order_by(mut self, property: String, descending: bool) -> Self {
        self.order_by = Some(property);
        self.order_direction = if descending { "desc" } else { "asc" }.to_string();
        self
    }
}

impl Default for GraphQuery {
    fn default() -> Self {
        Self::new()
    }
}

/// Query result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    /// Matched entities
    pub entities: Vec<Entity>,
    /// Matched relationships (if query included relationship patterns)
    pub relationships: Vec<Relationship>,
    /// Total count (before limit/offset applied)
    pub total_count: usize,
}

/// Query builder for fluent API
pub struct QueryBuilder {
    query: GraphQuery,
}

impl QueryBuilder {
    /// Creates a new query builder
    pub fn new() -> Self {
        Self {
            query: GraphQuery::new(),
        }
    }

    /// Matches entities of a specific type
    pub fn match_type(mut self, entity_type: EntityType) -> Self {
        self.query.patterns.push(GraphPattern {
            entity_type: Some(entity_type),
            filters: Vec::new(),
            relationship: None,
        });
        self
    }

    /// Adds a filter to the last pattern
    pub fn filter(mut self, property: String, operator: QueryOperator, value: String) -> Self {
        if let Some(last_pattern) = self.query.patterns.last_mut() {
            last_pattern
                .filters
                .push(QueryFilter::new(property, operator, value));
        }
        self
    }

    /// Sets the result limit
    pub fn limit(mut self, limit: usize) -> Self {
        self.query.limit = Some(limit);
        self
    }

    /// Sets the offset
    pub fn offset(mut self, offset: usize) -> Self {
        self.query.offset = Some(offset);
        self
    }

    /// Builds the query
    pub fn build(self) -> GraphQuery {
        self.query
    }
}

impl Default for QueryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Query executor
pub struct QueryExecutor;

impl QueryExecutor {
    /// Executes a query against a list of entities
    ///
    /// In a real implementation, this would query the underlying graph database.
    /// This simplified version filters entities in memory.
    pub fn execute(
        query: &GraphQuery,
        entities: &[Entity],
        relationships: &[Relationship],
    ) -> QueryResult {
        let mut matched_entities = Vec::new();
        let mut matched_relationships = Vec::new();

        // For each pattern, find matching entities
        for pattern in &query.patterns {
            for entity in entities {
                // Check entity type
                if let Some(ref et) = pattern.entity_type {
                    if entity.entity_type != *et {
                        continue;
                    }
                }

                // Check filters
                let all_filters_match = pattern.filters.iter().all(|f| f.matches(entity));
                if !all_filters_match {
                    continue;
                }

                // Check relationship pattern if present
                if let Some(ref rel_pattern) = pattern.relationship {
                    let has_matching_rel = Self::check_relationship(
                        entity,
                        rel_pattern,
                        relationships,
                        entities,
                    );
                    if !has_matching_rel {
                        continue;
                    }

                    // Add matching relationships
                    for rel in relationships {
                        if Self::relationship_matches(entity, rel, rel_pattern) {
                            matched_relationships.push(rel.clone());
                        }
                    }
                }

                matched_entities.push(entity.clone());
            }
        }

        // Deduplicate
        matched_entities.sort_by(|a, b| a.id.cmp(&b.id));
        matched_entities.dedup_by(|a, b| a.id == b.id);

        let total_count = matched_entities.len();

        // Apply ordering
        if let Some(ref order_prop) = query.order_by {
            let descending = query.order_direction == "desc";
            matched_entities.sort_by(|a, b| {
                let val_a = a.properties.get(order_prop).map(std::string::String::as_str).unwrap_or("");
                let val_b = b.properties.get(order_prop).map(std::string::String::as_str).unwrap_or("");
                if descending {
                    val_b.cmp(val_a)
                } else {
                    val_a.cmp(val_b)
                }
            });
        }

        // Apply offset and limit
        let offset = query.offset.unwrap_or(0);
        if offset < matched_entities.len() {
            matched_entities = matched_entities[offset..].to_vec();
        } else {
            matched_entities.clear();
        }

        if let Some(limit) = query.limit {
            matched_entities.truncate(limit);
        }

        QueryResult {
            entities: matched_entities,
            relationships: matched_relationships,
            total_count,
        }
    }

    fn check_relationship(
        entity: &Entity,
        rel_pattern: &RelationshipPattern,
        relationships: &[Relationship],
        entities: &[Entity],
    ) -> bool {
        for rel in relationships {
            if !Self::relationship_matches(entity, rel, rel_pattern) {
                continue;
            }

            // Check target entity matches target pattern
            let target_id = if rel.source_id == entity.id {
                &rel.target_id
            } else {
                &rel.source_id
            };

            if let Some(target_entity) = entities.iter().find(|e| e.id == *target_id) {
                // Check target entity type
                if let Some(ref target_type) = rel_pattern.target.entity_type {
                    if target_entity.entity_type != *target_type {
                        continue;
                    }
                }

                // Check target filters
                let target_filters_match = rel_pattern
                    .target
                    .filters
                    .iter()
                    .all(|f| f.matches(target_entity));

                if target_filters_match {
                    return true;
                }
            }
        }

        false
    }

    fn relationship_matches(
        entity: &Entity,
        rel: &Relationship,
        rel_pattern: &RelationshipPattern,
    ) -> bool {
        // Check direction
        let is_source = rel.source_id == entity.id;
        let is_target = rel.target_id == entity.id;

        let direction_matches = match rel_pattern.direction.as_str() {
            "outgoing" => is_source,
            "incoming" => is_target,
            "any" => is_source || is_target,
            _ => false,
        };

        if !direction_matches {
            return false;
        }

        // Check relation type
        if let Some(ref rel_type) = rel_pattern.relation_type {
            if &rel.relation_type != rel_type {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entity(id: &str, entity_type: EntityType, props: Vec<(&str, &str)>) -> Entity {
        let mut entity = Entity::new(id, entity_type);
        for (k, v) in props {
            entity = entity.with_property(k, v);
        }
        entity
    }

    #[test]
    fn test_query_filter_eq() {
        let entity = make_entity("e1", EntityType::Gnb, vec![("status", "active")]);
        let filter = QueryFilter::new("status".to_string(), QueryOperator::Eq, "active".to_string());

        assert!(filter.matches(&entity));

        let filter2 = QueryFilter::new("status".to_string(), QueryOperator::Eq, "inactive".to_string());
        assert!(!filter2.matches(&entity));
    }

    #[test]
    fn test_query_filter_contains() {
        let entity = make_entity("e1", EntityType::Gnb, vec![("name", "Main Tower")]);
        let filter = QueryFilter::new("name".to_string(), QueryOperator::Contains, "Tower".to_string());

        assert!(filter.matches(&entity));

        let filter2 = QueryFilter::new("name".to_string(), QueryOperator::Contains, "Cell".to_string());
        assert!(!filter2.matches(&entity));
    }

    #[test]
    fn test_query_filter_numeric() {
        let entity = make_entity("e1", EntityType::Cell, vec![("load", "75")]);

        let filter = QueryFilter::new("load".to_string(), QueryOperator::Gt, "50".to_string());
        assert!(filter.matches(&entity));

        let filter2 = QueryFilter::new("load".to_string(), QueryOperator::Lt, "50".to_string());
        assert!(!filter2.matches(&entity));
    }

    #[test]
    fn test_query_builder() {
        let query = QueryBuilder::new()
            .match_type(EntityType::Gnb)
            .filter("status".to_string(), QueryOperator::Eq, "active".to_string())
            .limit(10)
            .build();

        assert_eq!(query.patterns.len(), 1);
        assert_eq!(query.patterns[0].filters.len(), 1);
        assert_eq!(query.limit, Some(10));
    }

    #[test]
    fn test_query_executor_simple() {
        let entities = vec![
            make_entity("gnb-1", EntityType::Gnb, vec![("status", "active")]),
            make_entity("gnb-2", EntityType::Gnb, vec![("status", "inactive")]),
            make_entity("ue-1", EntityType::Ue, vec![("status", "active")]),
        ];

        let query = QueryBuilder::new()
            .match_type(EntityType::Gnb)
            .filter("status".to_string(), QueryOperator::Eq, "active".to_string())
            .build();

        let result = QueryExecutor::execute(&query, &entities, &[]);

        assert_eq!(result.entities.len(), 1);
        assert_eq!(result.entities[0].id, "gnb-1");
        assert_eq!(result.total_count, 1);
    }

    #[test]
    fn test_query_executor_limit_offset() {
        let entities = vec![
            make_entity("gnb-1", EntityType::Gnb, vec![("status", "active")]),
            make_entity("gnb-2", EntityType::Gnb, vec![("status", "active")]),
            make_entity("gnb-3", EntityType::Gnb, vec![("status", "active")]),
        ];

        let query = QueryBuilder::new()
            .match_type(EntityType::Gnb)
            .limit(2)
            .offset(1)
            .build();

        let result = QueryExecutor::execute(&query, &entities, &[]);

        assert_eq!(result.entities.len(), 2);
        assert_eq!(result.total_count, 3);
    }

    #[test]
    fn test_query_executor_ordering() {
        let entities = vec![
            make_entity("gnb-1", EntityType::Gnb, vec![("name", "Charlie")]),
            make_entity("gnb-2", EntityType::Gnb, vec![("name", "Alice")]),
            make_entity("gnb-3", EntityType::Gnb, vec![("name", "Bob")]),
        ];

        let query = QueryBuilder::new()
            .match_type(EntityType::Gnb)
            .build();

        let mut query_asc = query.clone();
        query_asc.order_by = Some("name".to_string());
        query_asc.order_direction = "asc".to_string();

        let result = QueryExecutor::execute(&query_asc, &entities, &[]);

        assert_eq!(result.entities.len(), 3);
        assert_eq!(result.entities[0].id, "gnb-2"); // Alice
        assert_eq!(result.entities[1].id, "gnb-3"); // Bob
        assert_eq!(result.entities[2].id, "gnb-1"); // Charlie
    }
}
