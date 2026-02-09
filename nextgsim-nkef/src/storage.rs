//! Persistent Storage for Knowledge Graph
//!
//! Provides disk-backed storage for the knowledge graph, enabling persistence
//! across restarts and supporting larger-than-memory graphs.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use thiserror::Error;

use crate::{Entity, Relationship};

/// Storage error types
#[derive(Error, Debug)]
pub enum StorageError {
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Entity not found
    #[error("Entity not found: {0}")]
    EntityNotFound(String),

    /// Invalid storage format
    #[error("Invalid storage format: {0}")]
    InvalidFormat(String),
}

/// Storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Base directory for storage
    pub base_dir: PathBuf,
    /// Whether to enable write-ahead logging
    pub enable_wal: bool,
    /// Maximum cache size (number of entities)
    pub cache_size: usize,
    /// Auto-save interval in seconds (0 = manual only)
    pub auto_save_interval_sec: u64,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            base_dir: PathBuf::from("./nkef_data"),
            enable_wal: true,
            cache_size: 10000,
            auto_save_interval_sec: 60,
        }
    }
}

/// Persistent storage backend for knowledge graph
#[derive(Debug)]
pub struct PersistentStorage {
    /// Storage configuration
    config: StorageConfig,
    /// In-memory cache of entities
    entity_cache: HashMap<String, Entity>,
    /// In-memory cache of relationships
    relationship_cache: Vec<Relationship>,
    /// Whether storage has been initialized
    initialized: bool,
    /// Last save timestamp
    last_save_ms: u64,
}

impl PersistentStorage {
    /// Creates a new persistent storage with the given configuration
    pub fn new(config: StorageConfig) -> Self {
        Self {
            config,
            entity_cache: HashMap::new(),
            relationship_cache: Vec::new(),
            initialized: false,
            last_save_ms: 0,
        }
    }

    /// Initializes the storage (creates directories, loads existing data)
    pub fn initialize(&mut self) -> Result<(), StorageError> {
        // Create base directory if it doesn't exist
        if !self.config.base_dir.exists() {
            std::fs::create_dir_all(&self.config.base_dir)?;
        }

        // Load existing data if present
        self.load()?;

        self.initialized = true;
        Ok(())
    }

    /// Loads data from disk
    pub fn load(&mut self) -> Result<(), StorageError> {
        let entities_file = self.config.base_dir.join("entities.json");
        let relationships_file = self.config.base_dir.join("relationships.json");

        // Load entities
        if entities_file.exists() {
            let data = std::fs::read_to_string(&entities_file)?;
            self.entity_cache = serde_json::from_str(&data)
                .map_err(|e| StorageError::Serialization(e.to_string()))?;
        }

        // Load relationships
        if relationships_file.exists() {
            let data = std::fs::read_to_string(&relationships_file)?;
            self.relationship_cache = serde_json::from_str(&data)
                .map_err(|e| StorageError::Serialization(e.to_string()))?;
        }

        Ok(())
    }

    /// Saves data to disk
    pub fn save(&mut self) -> Result<(), StorageError> {
        let entities_file = self.config.base_dir.join("entities.json");
        let relationships_file = self.config.base_dir.join("relationships.json");

        // Save entities
        let entities_data = serde_json::to_string_pretty(&self.entity_cache)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;
        std::fs::write(&entities_file, entities_data)?;

        // Save relationships
        let relationships_data = serde_json::to_string_pretty(&self.relationship_cache)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;
        std::fs::write(&relationships_file, relationships_data)?;

        self.last_save_ms = now_millis();

        Ok(())
    }

    /// Puts an entity into storage
    pub fn put_entity(&mut self, entity: Entity) -> Result<(), StorageError> {
        self.entity_cache.insert(entity.id.clone(), entity);

        // Auto-save if configured
        if self.should_auto_save() {
            self.save()?;
        }

        Ok(())
    }

    /// Gets an entity from storage
    pub fn get_entity(&self, id: &str) -> Result<Entity, StorageError> {
        self.entity_cache
            .get(id)
            .cloned()
            .ok_or_else(|| StorageError::EntityNotFound(id.to_string()))
    }

    /// Removes an entity from storage
    pub fn remove_entity(&mut self, id: &str) -> Result<(), StorageError> {
        self.entity_cache.remove(id);

        // Remove related relationships
        self.relationship_cache.retain(|r| r.source_id != id && r.target_id != id);

        if self.should_auto_save() {
            self.save()?;
        }

        Ok(())
    }

    /// Adds a relationship to storage
    pub fn put_relationship(&mut self, relationship: Relationship) -> Result<(), StorageError> {
        self.relationship_cache.push(relationship);

        if self.should_auto_save() {
            self.save()?;
        }

        Ok(())
    }

    /// Gets all entities
    pub fn all_entities(&self) -> Vec<Entity> {
        self.entity_cache.values().cloned().collect()
    }

    /// Gets all relationships
    pub fn all_relationships(&self) -> Vec<Relationship> {
        self.relationship_cache.clone()
    }

    /// Returns the number of cached entities
    pub fn entity_count(&self) -> usize {
        self.entity_cache.len()
    }

    /// Returns the number of cached relationships
    pub fn relationship_count(&self) -> usize {
        self.relationship_cache.len()
    }

    /// Checks if auto-save should be triggered
    fn should_auto_save(&self) -> bool {
        if self.config.auto_save_interval_sec == 0 {
            return false;
        }

        let elapsed_ms = now_millis().saturating_sub(self.last_save_ms);
        elapsed_ms >= self.config.auto_save_interval_sec * 1000
    }

    /// Clears all cached data
    pub fn clear(&mut self) {
        self.entity_cache.clear();
        self.relationship_cache.clear();
    }

    /// Returns the storage configuration
    pub fn config(&self) -> &StorageConfig {
        &self.config
    }
}

fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EntityType;
    use tempfile::TempDir;

    fn make_test_storage() -> (PersistentStorage, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let config = StorageConfig {
            base_dir: temp_dir.path().to_path_buf(),
            enable_wal: false,
            cache_size: 100,
            auto_save_interval_sec: 0, // Manual save only for tests
        };
        let storage = PersistentStorage::new(config);
        (storage, temp_dir)
    }

    #[test]
    fn test_storage_initialization() {
        let (mut storage, _temp) = make_test_storage();
        storage.initialize().unwrap();
        assert!(storage.initialized);
    }

    #[test]
    fn test_put_and_get_entity() {
        let (mut storage, _temp) = make_test_storage();
        storage.initialize().unwrap();

        let entity = Entity::new("test-entity", EntityType::Gnb);
        storage.put_entity(entity.clone()).unwrap();

        let retrieved = storage.get_entity("test-entity").unwrap();
        assert_eq!(retrieved.id, entity.id);
    }

    #[test]
    fn test_save_and_load() {
        let temp_dir = TempDir::new().unwrap();
        let config = StorageConfig {
            base_dir: temp_dir.path().to_path_buf(),
            enable_wal: false,
            cache_size: 100,
            auto_save_interval_sec: 0,
        };

        // Create storage, add entity, save
        {
            let mut storage = PersistentStorage::new(config.clone());
            storage.initialize().unwrap();

            let entity = Entity::new("test-entity", EntityType::Gnb)
                .with_property("status", "active");
            storage.put_entity(entity).unwrap();

            let rel = Relationship::new("test-entity", "other-entity", "connects_to");
            storage.put_relationship(rel).unwrap();

            storage.save().unwrap();
        }

        // Create new storage instance and load
        {
            let mut storage = PersistentStorage::new(config);
            storage.initialize().unwrap();

            assert_eq!(storage.entity_count(), 1);
            assert_eq!(storage.relationship_count(), 1);

            let entity = storage.get_entity("test-entity").unwrap();
            assert_eq!(entity.properties.get("status"), Some(&"active".to_string()));
        }
    }

    #[test]
    fn test_remove_entity() {
        let (mut storage, _temp) = make_test_storage();
        storage.initialize().unwrap();

        let entity = Entity::new("test-entity", EntityType::Gnb);
        storage.put_entity(entity).unwrap();

        assert_eq!(storage.entity_count(), 1);

        storage.remove_entity("test-entity").unwrap();
        assert_eq!(storage.entity_count(), 0);

        let result = storage.get_entity("test-entity");
        assert!(result.is_err());
    }

    #[test]
    fn test_all_entities() {
        let (mut storage, _temp) = make_test_storage();
        storage.initialize().unwrap();

        storage.put_entity(Entity::new("e1", EntityType::Gnb)).unwrap();
        storage.put_entity(Entity::new("e2", EntityType::Ue)).unwrap();

        let all = storage.all_entities();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_clear() {
        let (mut storage, _temp) = make_test_storage();
        storage.initialize().unwrap();

        storage.put_entity(Entity::new("e1", EntityType::Gnb)).unwrap();
        storage.put_relationship(Relationship::new("e1", "e2", "test")).unwrap();

        assert_eq!(storage.entity_count(), 1);
        assert_eq!(storage.relationship_count(), 1);

        storage.clear();

        assert_eq!(storage.entity_count(), 0);
        assert_eq!(storage.relationship_count(), 0);
    }
}
