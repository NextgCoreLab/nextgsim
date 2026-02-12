//! Model Versioning and Distribution (A17.3)
//!
//! Implements a version-controlled model store for federated learning,
//! allowing participants to fetch the correct model version and track
//! model evolution over time.

#![allow(missing_docs)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

use crate::AggregatedModel;

/// Error types for model store operations
#[derive(Debug, Error)]
pub enum ModelStoreError {
    /// Model version not found
    #[error("Model version {version} not found")]
    VersionNotFound { version: u64 },

    /// Storage error
    #[error("Storage error: {0}")]
    Storage(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),
}

/// Model metadata for tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model version
    pub version: u64,
    /// Creation timestamp
    pub timestamp_ms: u64,
    /// Number of participants that contributed
    pub num_participants: u32,
    /// Total samples used for training
    pub total_samples: u64,
    /// Average loss at this version
    pub avg_loss: f32,
    /// Model size in bytes (compressed)
    pub size_bytes: usize,
    /// Parent version (for tracking evolution)
    pub parent_version: Option<u64>,
    /// Tags for classification
    pub tags: Vec<String>,
}

/// Model store for versioning and distribution
pub struct ModelStore {
    /// All model versions
    models: HashMap<u64, AggregatedModel>,
    /// Model metadata
    metadata: HashMap<u64, ModelMetadata>,
    /// Latest version number
    latest_version: u64,
    /// Maximum versions to keep (for memory management)
    max_versions: usize,
}

impl ModelStore {
    /// Creates a new model store
    pub fn new(max_versions: usize) -> Self {
        Self {
            models: HashMap::new(),
            metadata: HashMap::new(),
            latest_version: 0,
            max_versions,
        }
    }

    /// Stores a new model version
    pub fn store(&mut self, model: AggregatedModel, tags: Vec<String>) -> Result<u64, ModelStoreError> {
        let version = model.version;

        // Create metadata
        let size_bytes = std::mem::size_of::<f32>() * model.weights.len()
            + std::mem::size_of::<AggregatedModel>();

        let metadata = ModelMetadata {
            version,
            timestamp_ms: model.timestamp_ms,
            num_participants: model.num_participants,
            total_samples: model.total_samples,
            avg_loss: model.avg_loss,
            size_bytes,
            parent_version: if version > 1 { Some(version - 1) } else { None },
            tags,
        };

        // Store model and metadata
        self.models.insert(version, model);
        self.metadata.insert(version, metadata);
        self.latest_version = self.latest_version.max(version);

        // Prune old versions if necessary
        self.prune_old_versions();

        Ok(version)
    }

    /// Retrieves a specific model version
    pub fn get(&self, version: u64) -> Result<&AggregatedModel, ModelStoreError> {
        self.models
            .get(&version)
            .ok_or(ModelStoreError::VersionNotFound { version })
    }

    /// Retrieves the latest model
    pub fn get_latest(&self) -> Option<&AggregatedModel> {
        self.models.get(&self.latest_version)
    }

    /// Retrieves metadata for a version
    pub fn get_metadata(&self, version: u64) -> Result<&ModelMetadata, ModelStoreError> {
        self.metadata
            .get(&version)
            .ok_or(ModelStoreError::VersionNotFound { version })
    }

    /// Lists all available versions
    pub fn list_versions(&self) -> Vec<u64> {
        let mut versions: Vec<u64> = self.models.keys().copied().collect();
        versions.sort();
        versions
    }

    /// Finds models by tag
    pub fn find_by_tag(&self, tag: &str) -> Vec<u64> {
        self.metadata
            .iter()
            .filter(|(_, meta)| meta.tags.contains(&tag.to_string()))
            .map(|(version, _)| *version)
            .collect()
    }

    /// Returns the latest version number
    pub fn latest_version(&self) -> u64 {
        self.latest_version
    }

    /// Prunes old versions to stay within `max_versions` limit
    fn prune_old_versions(&mut self) {
        if self.models.len() <= self.max_versions {
            return;
        }

        // Keep the latest versions, remove oldest
        let mut versions: Vec<u64> = self.models.keys().copied().collect();
        versions.sort();

        let to_remove = versions.len() - self.max_versions;
        for version in versions.iter().take(to_remove) {
            self.models.remove(version);
            self.metadata.remove(version);
        }
    }

    /// Exports a model version for distribution (serialized)
    pub fn export(&self, version: u64) -> Result<Vec<u8>, ModelStoreError> {
        let model = self.get(version)?;
        serde_json::to_vec(model)
            .map_err(|e| ModelStoreError::Serialization(e.to_string()))
    }

    /// Imports a model from serialized bytes
    pub fn import(&mut self, data: &[u8], tags: Vec<String>) -> Result<u64, ModelStoreError> {
        let model: AggregatedModel = serde_json::from_slice(data)
            .map_err(|e| ModelStoreError::Serialization(e.to_string()))?;

        self.store(model, tags)
    }

    /// Returns storage statistics
    pub fn stats(&self) -> StoreStats {
        StoreStats {
            num_versions: self.models.len(),
            latest_version: self.latest_version,
            total_size_bytes: self
                .metadata
                .values()
                .map(|m| m.size_bytes)
                .sum(),
        }
    }
}

impl Default for ModelStore {
    fn default() -> Self {
        Self::new(100) // Keep last 100 versions by default
    }
}

/// Model store statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreStats {
    /// Number of versions stored
    pub num_versions: usize,
    /// Latest version number
    pub latest_version: u64,
    /// Total storage size in bytes
    pub total_size_bytes: usize,
}

/// Model distribution manager for serving models to participants
pub struct DistributionManager {
    /// Model store
    store: ModelStore,
    /// Download statistics per participant
    downloads: HashMap<String, Vec<u64>>,
}

impl DistributionManager {
    /// Creates a new distribution manager
    pub fn new(store: ModelStore) -> Self {
        Self {
            store,
            downloads: HashMap::new(),
        }
    }

    /// Serves a model to a participant
    pub fn serve_model(
        &mut self,
        participant_id: String,
        version: Option<u64>,
    ) -> Result<&AggregatedModel, ModelStoreError> {
        let version = version.unwrap_or(self.store.latest_version());
        let model = self.store.get(version)?;

        // Track download
        self.downloads
            .entry(participant_id)
            .or_default()
            .push(version);

        Ok(model)
    }

    /// Returns download statistics for a participant
    pub fn get_download_history(&self, participant_id: &str) -> Option<&Vec<u64>> {
        self.downloads.get(participant_id)
    }

    /// Returns the model store
    pub fn store(&self) -> &ModelStore {
        &self.store
    }

    /// Returns mutable access to the model store
    pub fn store_mut(&mut self) -> &mut ModelStore {
        &mut self.store
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_model(version: u64) -> AggregatedModel {
        AggregatedModel {
            version,
            weights: vec![0.1 * version as f32; 100],
            num_participants: 10,
            total_samples: 1000,
            avg_loss: 0.5 / version as f32,
            timestamp_ms: version * 1000,
        }
    }

    #[test]
    fn test_model_store_creation() {
        let store = ModelStore::new(10);
        assert_eq!(store.latest_version(), 0);
        assert_eq!(store.list_versions().len(), 0);
    }

    #[test]
    fn test_store_and_retrieve() {
        let mut store = ModelStore::new(10);

        let model = create_test_model(1);
        let version = store.store(model.clone(), vec!["test".to_string()]).unwrap();

        assert_eq!(version, 1);
        assert_eq!(store.latest_version(), 1);

        let retrieved = store.get(1).unwrap();
        assert_eq!(retrieved.version, 1);
    }

    #[test]
    fn test_get_latest() {
        let mut store = ModelStore::new(10);

        store.store(create_test_model(1), vec![]).unwrap();
        store.store(create_test_model(2), vec![]).unwrap();
        store.store(create_test_model(3), vec![]).unwrap();

        let latest = store.get_latest().unwrap();
        assert_eq!(latest.version, 3);
    }

    #[test]
    fn test_list_versions() {
        let mut store = ModelStore::new(10);

        store.store(create_test_model(1), vec![]).unwrap();
        store.store(create_test_model(3), vec![]).unwrap();
        store.store(create_test_model(2), vec![]).unwrap();

        let versions = store.list_versions();
        assert_eq!(versions, vec![1, 2, 3]);
    }

    #[test]
    fn test_find_by_tag() {
        let mut store = ModelStore::new(10);

        store
            .store(create_test_model(1), vec!["baseline".to_string()])
            .unwrap();
        store
            .store(create_test_model(2), vec!["improved".to_string()])
            .unwrap();
        store
            .store(create_test_model(3), vec!["baseline".to_string()])
            .unwrap();

        let baseline_versions = store.find_by_tag("baseline");
        assert_eq!(baseline_versions.len(), 2);
    }

    #[test]
    fn test_pruning() {
        let mut store = ModelStore::new(3);

        for i in 1..=5 {
            store.store(create_test_model(i), vec![]).unwrap();
        }

        // Should keep only 3 latest versions
        assert_eq!(store.list_versions().len(), 3);
        assert_eq!(store.list_versions(), vec![3, 4, 5]);
    }

    #[test]
    fn test_export_import() {
        let mut store1 = ModelStore::new(10);
        let model = create_test_model(1);
        store1.store(model, vec!["exported".to_string()]).unwrap();

        let exported = store1.export(1).unwrap();

        let mut store2 = ModelStore::new(10);
        let version = store2.import(&exported, vec!["imported".to_string()]).unwrap();

        assert_eq!(version, 1);
        let imported_model = store2.get(1).unwrap();
        assert_eq!(imported_model.version, 1);
    }

    #[test]
    fn test_metadata() {
        let mut store = ModelStore::new(10);
        let model = create_test_model(1);
        store.store(model, vec!["test".to_string()]).unwrap();

        let metadata = store.get_metadata(1).unwrap();
        assert_eq!(metadata.version, 1);
        assert_eq!(metadata.num_participants, 10);
        assert!(metadata.tags.contains(&"test".to_string()));
    }

    #[test]
    fn test_distribution_manager() {
        let store = ModelStore::new(10);
        let mut manager = DistributionManager::new(store);

        manager
            .store_mut()
            .store(create_test_model(1), vec![])
            .unwrap();
        manager
            .store_mut()
            .store(create_test_model(2), vec![])
            .unwrap();

        let model = manager
            .serve_model("client1".to_string(), Some(1))
            .unwrap();
        assert_eq!(model.version, 1);

        let history = manager.get_download_history("client1").unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0], 1);
    }

    #[test]
    fn test_store_stats() {
        let mut store = ModelStore::new(10);
        store.store(create_test_model(1), vec![]).unwrap();
        store.store(create_test_model(2), vec![]).unwrap();

        let stats = store.stats();
        assert_eq!(stats.num_versions, 2);
        assert_eq!(stats.latest_version, 2);
        assert!(stats.total_size_bytes > 0);
    }
}
