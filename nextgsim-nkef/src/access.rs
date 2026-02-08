//! Access Control and Exposure Policies
//!
//! Implements Nnkef-style access control for knowledge graph data,
//! controlling which entities and relationships can be accessed by consumers.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::{Entity, EntityType};

/// Access level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessLevel {
    /// No access
    None,
    /// Read-only access
    Read,
    /// Read and write access
    ReadWrite,
    /// Full access (including delete)
    Admin,
}

/// Access policy for an entity or entity type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessPolicy {
    /// Policy identifier
    pub policy_id: String,
    /// Entity type this policy applies to (None = all types)
    pub entity_type: Option<EntityType>,
    /// Specific entity IDs this policy applies to
    pub entity_ids: HashSet<String>,
    /// Consumer/role identifier
    pub consumer_id: String,
    /// Access level granted
    pub access_level: AccessLevel,
    /// Property-level restrictions (list of allowed properties, empty = all)
    pub allowed_properties: HashSet<String>,
}

impl AccessPolicy {
    /// Creates a new access policy
    pub fn new(policy_id: String, consumer_id: String, access_level: AccessLevel) -> Self {
        Self {
            policy_id,
            entity_type: None,
            entity_ids: HashSet::new(),
            consumer_id,
            access_level,
            allowed_properties: HashSet::new(),
        }
    }

    /// Applies the policy to a specific entity type
    pub fn for_entity_type(mut self, entity_type: EntityType) -> Self {
        self.entity_type = Some(entity_type);
        self
    }

    /// Applies the policy to specific entity IDs
    pub fn for_entities(mut self, entity_ids: Vec<String>) -> Self {
        self.entity_ids = entity_ids.into_iter().collect();
        self
    }

    /// Restricts access to specific properties
    pub fn with_properties(mut self, properties: Vec<String>) -> Self {
        self.allowed_properties = properties.into_iter().collect();
        self
    }
}

/// Access control manager
#[derive(Debug)]
pub struct AccessController {
    /// Active policies indexed by policy ID
    policies: HashMap<String, AccessPolicy>,
    /// Default access level for unknown consumers
    default_access: AccessLevel,
}

impl AccessController {
    /// Creates a new access controller
    pub fn new(default_access: AccessLevel) -> Self {
        Self {
            policies: HashMap::new(),
            default_access,
        }
    }

    /// Adds an access policy
    pub fn add_policy(&mut self, policy: AccessPolicy) {
        self.policies.insert(policy.policy_id.clone(), policy);
    }

    /// Removes an access policy
    pub fn remove_policy(&mut self, policy_id: &str) -> Option<AccessPolicy> {
        self.policies.remove(policy_id)
    }

    /// Checks if a consumer can access an entity
    pub fn can_access(&self, consumer_id: &str, entity: &Entity) -> bool {
        let level = self.get_access_level(consumer_id, entity);
        matches!(level, AccessLevel::Read | AccessLevel::ReadWrite | AccessLevel::Admin)
    }

    /// Checks if a consumer can modify an entity
    pub fn can_modify(&self, consumer_id: &str, entity: &Entity) -> bool {
        let level = self.get_access_level(consumer_id, entity);
        matches!(level, AccessLevel::ReadWrite | AccessLevel::Admin)
    }

    /// Gets the access level for a consumer and entity
    pub fn get_access_level(&self, consumer_id: &str, entity: &Entity) -> AccessLevel {
        // Find matching policies
        let mut max_level = self.default_access;

        for policy in self.policies.values() {
            if policy.consumer_id != consumer_id {
                continue;
            }

            // Check if policy applies to this entity
            let applies = if !policy.entity_ids.is_empty() {
                policy.entity_ids.contains(&entity.id)
            } else if let Some(ref et) = policy.entity_type {
                *et == entity.entity_type
            } else {
                true // Policy applies to all entities
            };

            if applies && policy.access_level as u8 > max_level as u8 {
                max_level = policy.access_level;
            }
        }

        max_level
    }

    /// Filters entity properties based on access policy
    pub fn filter_properties(&self, consumer_id: &str, entity: &mut Entity) {
        // Find applicable policy with property restrictions
        for policy in self.policies.values() {
            if policy.consumer_id != consumer_id {
                continue;
            }

            let applies = if !policy.entity_ids.is_empty() {
                policy.entity_ids.contains(&entity.id)
            } else if let Some(ref et) = policy.entity_type {
                *et == entity.entity_type
            } else {
                true
            };

            if applies && !policy.allowed_properties.is_empty() {
                // Keep only allowed properties
                entity.properties.retain(|k, _| policy.allowed_properties.contains(k));
                return;
            }
        }
    }

    /// Returns the number of active policies
    pub fn policy_count(&self) -> usize {
        self.policies.len()
    }
}

impl Default for AccessController {
    fn default() -> Self {
        Self::new(AccessLevel::Read) // Default to read-only
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_access_controller_creation() {
        let controller = AccessController::new(AccessLevel::None);
        assert_eq!(controller.policy_count(), 0);
    }

    #[test]
    fn test_add_remove_policy() {
        let mut controller = AccessController::new(AccessLevel::None);
        let policy = AccessPolicy::new("p1".to_string(), "consumer-1".to_string(), AccessLevel::Read);

        controller.add_policy(policy);
        assert_eq!(controller.policy_count(), 1);

        controller.remove_policy("p1");
        assert_eq!(controller.policy_count(), 0);
    }

    #[test]
    fn test_can_access() {
        let mut controller = AccessController::new(AccessLevel::None);
        let policy = AccessPolicy::new("p1".to_string(), "consumer-1".to_string(), AccessLevel::Read)
            .for_entity_type(EntityType::Gnb);

        controller.add_policy(policy);

        let entity = Entity::new("gnb-1", EntityType::Gnb);
        assert!(controller.can_access("consumer-1", &entity));
        assert!(!controller.can_access("consumer-2", &entity));
    }

    #[test]
    fn test_can_modify() {
        let mut controller = AccessController::new(AccessLevel::None);
        let policy = AccessPolicy::new("p1".to_string(), "consumer-1".to_string(), AccessLevel::ReadWrite)
            .for_entity_type(EntityType::Gnb);

        controller.add_policy(policy);

        let entity = Entity::new("gnb-1", EntityType::Gnb);
        assert!(controller.can_modify("consumer-1", &entity));
        assert!(!controller.can_modify("consumer-2", &entity));
    }

    #[test]
    fn test_filter_properties() {
        let mut controller = AccessController::new(AccessLevel::Read);
        let policy = AccessPolicy::new("p1".to_string(), "consumer-1".to_string(), AccessLevel::Read)
            .for_entity_type(EntityType::Gnb)
            .with_properties(vec!["status".to_string()]);

        controller.add_policy(policy);

        let mut entity = Entity::new("gnb-1", EntityType::Gnb)
            .with_property("status", "active")
            .with_property("secret", "confidential");

        controller.filter_properties("consumer-1", &mut entity);

        assert!(entity.properties.contains_key("status"));
        assert!(!entity.properties.contains_key("secret"));
    }

    #[test]
    fn test_default_access() {
        let controller = AccessController::new(AccessLevel::Read);
        let entity = Entity::new("gnb-1", EntityType::Gnb);

        // No policies, should use default
        assert!(controller.can_access("anyone", &entity));
    }
}
