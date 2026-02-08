//! Formal Ontology and Schema Validation (A14.4)
//!
//! Provides formal ontology definition and schema validation for knowledge graph entities.

#![allow(missing_docs)]

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::{Entity, EntityType, Relationship};

/// Schema definition error
#[derive(Debug, thiserror::Error)]
pub enum SchemaError {
    #[error("Invalid entity: {reason}")]
    InvalidEntity { reason: String },

    #[error("Missing required attribute: {attribute}")]
    MissingAttribute { attribute: String },

    #[error("Invalid attribute type: expected {expected}, got {actual}")]
    InvalidAttributeType { expected: String, actual: String },

    #[error("Invalid relationship: {reason}")]
    InvalidRelationship { reason: String },

    #[error("Constraint violation: {constraint}")]
    ConstraintViolation { constraint: String },
}

/// Attribute data type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttributeType {
    /// String type
    String,
    /// Integer type
    Integer,
    /// Float type
    Float,
    /// Boolean type
    Boolean,
    /// Timestamp (milliseconds since epoch)
    Timestamp,
    /// List of values
    List(Box<AttributeType>),
    /// Reference to another entity
    EntityReference(EntityType),
}

/// Attribute definition in schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributeDefinition {
    /// Attribute name
    pub name: String,
    /// Attribute type
    pub attr_type: AttributeType,
    /// Whether this attribute is required
    pub required: bool,
    /// Default value (if not required)
    pub default: Option<String>,
    /// Description
    pub description: String,
}

impl AttributeDefinition {
    /// Creates a required attribute definition
    pub fn required(name: impl Into<String>, attr_type: AttributeType, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            attr_type,
            required: true,
            default: None,
            description: description.into(),
        }
    }

    /// Creates an optional attribute definition
    pub fn optional(
        name: impl Into<String>,
        attr_type: AttributeType,
        default: Option<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            attr_type,
            required: false,
            default,
            description: description.into(),
        }
    }
}

/// Entity schema definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntitySchema {
    /// Entity type
    pub entity_type: EntityType,
    /// Allowed attributes
    pub attributes: Vec<AttributeDefinition>,
    /// Allowed relationship types (from this entity)
    pub allowed_relationships: HashSet<String>,
    /// Entity-level constraints
    pub constraints: Vec<Constraint>,
}

/// Relationship schema definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipSchema {
    /// Relationship type
    pub relationship_type: String,
    /// Allowed source entity types
    pub allowed_sources: HashSet<EntityType>,
    /// Allowed target entity types
    pub allowed_targets: HashSet<EntityType>,
    /// Relationship attributes
    pub attributes: Vec<AttributeDefinition>,
}

/// Schema constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Constraint {
    /// Unique attribute value across all entities of this type
    Unique { attribute: String },
    /// Attribute value must match regex
    Pattern { attribute: String, pattern: String },
    /// Numeric range constraint
    Range {
        attribute: String,
        min: Option<f64>,
        max: Option<f64>,
    },
    /// Dependency: if attribute A is set, attribute B must also be set
    Dependency {
        source_attribute: String,
        required_attribute: String,
    },
}

/// Formal ontology for the knowledge graph
#[derive(Debug, Clone)]
pub struct Ontology {
    /// Entity schemas by type
    entity_schemas: HashMap<EntityType, EntitySchema>,
    /// Relationship schemas by type
    relationship_schemas: HashMap<String, RelationshipSchema>,
    /// Ontology version
    version: String,
}

impl Ontology {
    /// Creates a new empty ontology
    pub fn new(version: impl Into<String>) -> Self {
        Self {
            entity_schemas: HashMap::new(),
            relationship_schemas: HashMap::new(),
            version: version.into(),
        }
    }

    /// Creates a 3GPP-compliant ontology with standard entity and relationship schemas
    pub fn new_3gpp_compliant() -> Self {
        let mut ontology = Self::new("3GPP-R18");

        // UE entity schema
        ontology.add_entity_schema(EntitySchema {
            entity_type: EntityType::Ue,
            attributes: vec![
                AttributeDefinition::required("supi", AttributeType::String, "Subscription Permanent Identifier"),
                AttributeDefinition::required("imei", AttributeType::String, "International Mobile Equipment Identity"),
                AttributeDefinition::optional("cell_id", AttributeType::Integer, None, "Current serving cell ID"),
                AttributeDefinition::optional("connected", AttributeType::Boolean, Some("false".to_string()), "Connection status"),
            ],
            allowed_relationships: ["attached_to", "served_by", "has_session"].iter().map(|s| (*s).to_string()).collect(),
            constraints: vec![Constraint::Unique {
                attribute: "supi".to_string(),
            }],
        });

        // gNB entity schema
        ontology.add_entity_schema(EntitySchema {
            entity_type: EntityType::Gnb,
            attributes: vec![
                AttributeDefinition::required("gnb_id", AttributeType::Integer, "gNodeB identifier"),
                AttributeDefinition::optional("plmn_id", AttributeType::String, None, "PLMN identifier"),
                AttributeDefinition::optional("location", AttributeType::String, None, "Geographic location"),
            ],
            allowed_relationships: ["serves", "connects_to"].iter().map(|s| (*s).to_string()).collect(),
            constraints: vec![Constraint::Unique {
                attribute: "gnb_id".to_string(),
            }],
        });

        // AMF entity schema
        ontology.add_entity_schema(EntitySchema {
            entity_type: EntityType::Amf,
            attributes: vec![
                AttributeDefinition::required("amf_id", AttributeType::String, "AMF identifier"),
                AttributeDefinition::optional("region", AttributeType::String, None, "AMF region"),
                AttributeDefinition::optional("capacity", AttributeType::Integer, None, "AMF capacity"),
            ],
            allowed_relationships: ["manages", "routes_to"].iter().map(|s| (*s).to_string()).collect(),
            constraints: vec![],
        });

        // Network slice schema
        ontology.add_entity_schema(EntitySchema {
            entity_type: EntityType::Slice,
            attributes: vec![
                AttributeDefinition::required("s_nssai", AttributeType::String, "Single NSSAI identifier"),
                AttributeDefinition::optional("sst", AttributeType::Integer, None, "Slice/Service Type"),
                AttributeDefinition::optional("sd", AttributeType::String, None, "Slice Differentiator"),
            ],
            allowed_relationships: ["provides", "allocated_to"].iter().map(|s| (*s).to_string()).collect(),
            constraints: vec![],
        });

        // Add standard relationship schemas
        ontology.add_relationship_schema(RelationshipSchema {
            relationship_type: "attached_to".to_string(),
            allowed_sources: [EntityType::Ue].iter().copied().collect(),
            allowed_targets: [EntityType::Gnb, EntityType::Cell].iter().copied().collect(),
            attributes: vec![AttributeDefinition::optional(
                "attachment_time",
                AttributeType::Timestamp,
                None,
                "Time of attachment",
            )],
        });

        ontology.add_relationship_schema(RelationshipSchema {
            relationship_type: "serves".to_string(),
            allowed_sources: [EntityType::Gnb].iter().copied().collect(),
            allowed_targets: [EntityType::Ue, EntityType::Cell].iter().copied().collect(),
            attributes: vec![],
        });

        ontology
    }

    /// Adds an entity schema to the ontology
    pub fn add_entity_schema(&mut self, schema: EntitySchema) {
        self.entity_schemas.insert(schema.entity_type, schema);
    }

    /// Adds a relationship schema to the ontology
    pub fn add_relationship_schema(&mut self, schema: RelationshipSchema) {
        self.relationship_schemas
            .insert(schema.relationship_type.clone(), schema);
    }

    /// Validates an entity against the schema
    pub fn validate_entity(&self, entity: &Entity) -> Result<(), SchemaError> {
        let schema = self.entity_schemas.get(&entity.entity_type).ok_or_else(|| SchemaError::InvalidEntity {
            reason: format!("No schema defined for entity type {:?}", entity.entity_type),
        })?;

        // Check required attributes
        for attr_def in &schema.attributes {
            if attr_def.required && !entity.properties.contains_key(&attr_def.name) {
                return Err(SchemaError::MissingAttribute {
                    attribute: attr_def.name.clone(),
                });
            }
        }

        // Validate attribute types
        for (attr_name, attr_value) in &entity.properties {
            if let Some(attr_def) = schema.attributes.iter().find(|a| &a.name == attr_name) {
                self.validate_attribute_type(attr_value, &attr_def.attr_type)?;
            }
        }

        // Check constraints
        for constraint in &schema.constraints {
            self.validate_constraint(entity, constraint)?;
        }

        Ok(())
    }

    /// Validates a relationship against the schema
    pub fn validate_relationship(&self, relationship: &Relationship) -> Result<(), SchemaError> {
        let _schema = self
            .relationship_schemas
            .get(&relationship.relation_type)
            .ok_or_else(|| SchemaError::InvalidRelationship {
                reason: format!("No schema defined for relationship type '{}'", relationship.relation_type),
            })?;

        // Note: We would need entity type information to fully validate sources and targets
        // This is a simplified version

        Ok(())
    }

    /// Validates attribute type
    fn validate_attribute_type(&self, value: &str, expected_type: &AttributeType) -> Result<(), SchemaError> {
        match expected_type {
            AttributeType::String => Ok(()),
            AttributeType::Integer => value.parse::<i64>().map(|_| ()).map_err(|_| SchemaError::InvalidAttributeType {
                expected: "Integer".to_string(),
                actual: value.to_string(),
            }),
            AttributeType::Float => value.parse::<f64>().map(|_| ()).map_err(|_| SchemaError::InvalidAttributeType {
                expected: "Float".to_string(),
                actual: value.to_string(),
            }),
            AttributeType::Boolean => value.parse::<bool>().map(|_| ()).map_err(|_| SchemaError::InvalidAttributeType {
                expected: "Boolean".to_string(),
                actual: value.to_string(),
            }),
            AttributeType::Timestamp => value.parse::<u64>().map(|_| ()).map_err(|_| SchemaError::InvalidAttributeType {
                expected: "Timestamp".to_string(),
                actual: value.to_string(),
            }),
            _ => Ok(()), // Simplified
        }
    }

    /// Validates a constraint
    fn validate_constraint(&self, entity: &Entity, constraint: &Constraint) -> Result<(), SchemaError> {
        match constraint {
            Constraint::Unique { attribute: _ } => {
                // Would need to check against all entities in the store
                Ok(())
            }
            Constraint::Pattern { attribute, pattern: _ } => {
                if !entity.properties.contains_key(attribute) {
                    return Ok(()); // Optional attribute
                }
                // Would use regex matching here
                Ok(())
            }
            Constraint::Range { attribute, min, max } => {
                if let Some(value_str) = entity.properties.get(attribute) {
                    if let Ok(value) = value_str.parse::<f64>() {
                        if let Some(min_val) = min {
                            if value < *min_val {
                                return Err(SchemaError::ConstraintViolation {
                                    constraint: format!("{attribute} must be >= {min_val}"),
                                });
                            }
                        }
                        if let Some(max_val) = max {
                            if value > *max_val {
                                return Err(SchemaError::ConstraintViolation {
                                    constraint: format!("{attribute} must be <= {max_val}"),
                                });
                            }
                        }
                    }
                }
                Ok(())
            }
            Constraint::Dependency {
                source_attribute,
                required_attribute,
            } => {
                if entity.properties.contains_key(source_attribute) && !entity.properties.contains_key(required_attribute) {
                    return Err(SchemaError::ConstraintViolation {
                        constraint: format!("If {source_attribute} is set, {required_attribute} must also be set"),
                    });
                }
                Ok(())
            }
        }
    }

    /// Returns the schema for an entity type
    pub fn get_entity_schema(&self, entity_type: EntityType) -> Option<&EntitySchema> {
        self.entity_schemas.get(&entity_type)
    }

    /// Returns the schema for a relationship type
    pub fn get_relationship_schema(&self, relationship_type: &str) -> Option<&RelationshipSchema> {
        self.relationship_schemas.get(relationship_type)
    }

    /// Returns the ontology version
    pub fn version(&self) -> &str {
        &self.version
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ontology_creation() {
        let ontology = Ontology::new("1.0");
        assert_eq!(ontology.version(), "1.0");
    }

    #[test]
    fn test_3gpp_ontology() {
        let ontology = Ontology::new_3gpp_compliant();
        assert_eq!(ontology.version(), "3GPP-R18");

        let ue_schema = ontology.get_entity_schema(EntityType::Ue).unwrap();
        assert_eq!(ue_schema.entity_type, EntityType::Ue);
        assert!(ue_schema.attributes.iter().any(|a| a.name == "supi"));
    }

    #[test]
    fn test_validate_valid_entity() {
        let ontology = Ontology::new_3gpp_compliant();

        let mut entity = Entity {
            id: "ue1".to_string(),
            entity_type: EntityType::Ue,
            properties: HashMap::new(),
            embedding: None,
        };
        entity.properties.insert("supi".to_string(), "imsi-001".to_string());
        entity.properties.insert("imei".to_string(), "123456789".to_string());

        let result = ontology.validate_entity(&entity);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_missing_required_attribute() {
        let ontology = Ontology::new_3gpp_compliant();

        let entity = Entity {
            id: "ue1".to_string(),
            entity_type: EntityType::Ue,
            properties: HashMap::new(),
            embedding: None,
        };

        let result = ontology.validate_entity(&entity);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), SchemaError::MissingAttribute { .. }));
    }

    #[test]
    fn test_validate_invalid_type() {
        let mut ontology = Ontology::new("1.0");

        let schema = EntitySchema {
            entity_type: EntityType::Gnb,
            attributes: vec![AttributeDefinition::required("gnb_id", AttributeType::Integer, "gNB ID")],
            allowed_relationships: HashSet::new(),
            constraints: vec![],
        };
        ontology.add_entity_schema(schema);

        let mut entity = Entity {
            id: "gnb1".to_string(),
            entity_type: EntityType::Gnb,
            properties: HashMap::new(),
            embedding: None,
        };
        entity.properties.insert("gnb_id".to_string(), "not_a_number".to_string());

        let result = ontology.validate_entity(&entity);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), SchemaError::InvalidAttributeType { .. }));
    }

    #[test]
    fn test_range_constraint() {
        let mut ontology = Ontology::new("1.0");

        let schema = EntitySchema {
            entity_type: EntityType::Amf,
            attributes: vec![
                AttributeDefinition::required("amf_id", AttributeType::String, "AMF ID"),
                AttributeDefinition::optional("capacity", AttributeType::Integer, None, "Capacity"),
            ],
            allowed_relationships: HashSet::new(),
            constraints: vec![Constraint::Range {
                attribute: "capacity".to_string(),
                min: Some(0.0),
                max: Some(1000.0),
            }],
        };
        ontology.add_entity_schema(schema);

        let mut entity = Entity {
            id: "amf1".to_string(),
            entity_type: EntityType::Amf,
            properties: HashMap::new(),
            embedding: None,
        };
        entity.properties.insert("amf_id".to_string(), "amf-001".to_string());
        entity.properties.insert("capacity".to_string(), "1500".to_string());

        let result = ontology.validate_entity(&entity);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), SchemaError::ConstraintViolation { .. }));
    }
}
