//! Vector similarity search index
//!
//! Provides a `VectorIndex` that stores entity embeddings and supports
//! nearest-neighbor search using cosine similarity.

use std::collections::HashMap;

/// Result from a similarity search
#[derive(Debug, Clone)]
pub struct SimilarityResult {
    /// Entity ID
    pub id: String,
    /// Cosine similarity score in range [-1.0, 1.0] (higher = more similar)
    pub score: f32,
}

/// Vector index for embedding-based similarity search.
///
/// Stores entity embeddings and supports efficient nearest-neighbor
/// search using cosine similarity: `dot(a,b) / (||a|| * ||b||)`.
#[derive(Debug, Clone)]
pub struct VectorIndex {
    /// Embedding dimension
    dim: usize,
    /// Stored embeddings keyed by entity ID
    embeddings: HashMap<String, Vec<f32>>,
    /// Precomputed norms for each embedding (caching for performance)
    norms: HashMap<String, f32>,
}

impl VectorIndex {
    /// Creates a new vector index with the given embedding dimension.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            embeddings: HashMap::new(),
            norms: HashMap::new(),
        }
    }

    /// Returns the embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Returns the number of stored embeddings.
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    /// Returns true if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

    /// Inserts or updates an embedding for the given entity ID.
    ///
    /// The embedding must have the same dimension as the index.
    /// Returns `true` if the embedding was inserted (new), `false` if updated.
    pub fn upsert(&mut self, id: impl Into<String>, embedding: Vec<f32>) -> bool {
        let id = id.into();
        assert_eq!(
            embedding.len(),
            self.dim,
            "Embedding dimension mismatch: expected {}, got {}",
            self.dim,
            embedding.len()
        );

        let norm = l2_norm(&embedding);
        self.norms.insert(id.clone(), norm);
        self.embeddings.insert(id, embedding).is_none()
    }

    /// Removes an embedding by entity ID.
    ///
    /// Returns the removed embedding, or `None` if the ID was not found.
    pub fn remove(&mut self, id: &str) -> Option<Vec<f32>> {
        self.norms.remove(id);
        self.embeddings.remove(id)
    }

    /// Gets an embedding by entity ID.
    pub fn get(&self, id: &str) -> Option<&Vec<f32>> {
        self.embeddings.get(id)
    }

    /// Returns true if the index contains the given entity ID.
    pub fn contains(&self, id: &str) -> bool {
        self.embeddings.contains_key(id)
    }

    /// Finds the top-k most similar entities to the given query embedding.
    ///
    /// Returns results sorted by similarity score in descending order.
    /// Entities with zero-norm embeddings are excluded from results.
    pub fn search_topk(&self, query: &[f32], k: usize) -> Vec<SimilarityResult> {
        assert_eq!(
            query.len(),
            self.dim,
            "Query dimension mismatch: expected {}, got {}",
            self.dim,
            query.len()
        );

        let query_norm = l2_norm(query);
        if query_norm == 0.0 {
            return Vec::new();
        }

        let mut scores: Vec<SimilarityResult> = self
            .embeddings
            .iter()
            .filter_map(|(id, emb)| {
                let emb_norm = self.norms.get(id).copied().unwrap_or(0.0);
                if emb_norm == 0.0 {
                    return None;
                }
                let dot = dot_product(query, emb);
                let score = dot / (query_norm * emb_norm);
                Some(SimilarityResult {
                    id: id.clone(),
                    score,
                })
            })
            .collect();

        // Sort by score descending
        scores.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        scores.truncate(k);
        scores
    }

    /// Finds entities with similarity above a threshold to the query embedding.
    ///
    /// Returns results sorted by similarity score in descending order.
    pub fn search_threshold(&self, query: &[f32], threshold: f32) -> Vec<SimilarityResult> {
        assert_eq!(
            query.len(),
            self.dim,
            "Query dimension mismatch: expected {}, got {}",
            self.dim,
            query.len()
        );

        let query_norm = l2_norm(query);
        if query_norm == 0.0 {
            return Vec::new();
        }

        let mut scores: Vec<SimilarityResult> = self
            .embeddings
            .iter()
            .filter_map(|(id, emb)| {
                let emb_norm = self.norms.get(id).copied().unwrap_or(0.0);
                if emb_norm == 0.0 {
                    return None;
                }
                let dot = dot_product(query, emb);
                let score = dot / (query_norm * emb_norm);
                if score >= threshold {
                    Some(SimilarityResult {
                        id: id.clone(),
                        score,
                    })
                } else {
                    None
                }
            })
            .collect();

        scores.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        scores
    }

    /// Batch similarity search: finds top-k results for each query embedding.
    ///
    /// Returns one result vector per query, maintaining the same order.
    pub fn batch_search(&self, queries: &[Vec<f32>], k: usize) -> Vec<Vec<SimilarityResult>> {
        queries
            .iter()
            .map(|query| self.search_topk(query, k))
            .collect()
    }

    /// Returns all entity IDs stored in the index.
    pub fn entity_ids(&self) -> Vec<&str> {
        self.embeddings.keys().map(std::string::String::as_str).collect()
    }

    /// Clears all embeddings from the index.
    pub fn clear(&mut self) {
        self.embeddings.clear();
        self.norms.clear();
    }
}

/// Computes the dot product of two vectors.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Computes the L2 (Euclidean) norm of a vector.
#[inline]
pub fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Computes cosine similarity between two vectors.
///
/// Returns `dot(a,b) / (||a|| * ||b||)`.
/// Returns 0.0 if either vector has zero norm.
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let dot = dot_product(a, b);
    let norm_a = l2_norm(a);
    let norm_b = l2_norm(b);
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Normalizes a vector to unit length (L2 normalization).
///
/// Returns a zero vector if the input has zero norm.
pub fn normalize(v: &[f32]) -> Vec<f32> {
    let norm = l2_norm(v);
    if norm == 0.0 {
        return vec![0.0; v.len()];
    }
    v.iter().map(|x| x / norm).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_normalize() {
        let v = vec![3.0, 4.0];
        let n = normalize(&v);
        assert!((n[0] - 0.6).abs() < 1e-6);
        assert!((n[1] - 0.8).abs() < 1e-6);
        assert!((l2_norm(&n) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_zero() {
        let v = vec![0.0, 0.0, 0.0];
        let n = normalize(&v);
        assert_eq!(n, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_vector_index_upsert_and_get() {
        let mut index = VectorIndex::new(3);
        assert!(index.is_empty());

        let is_new = index.upsert("a", vec![1.0, 0.0, 0.0]);
        assert!(is_new);
        assert_eq!(index.len(), 1);
        assert!(index.contains("a"));

        let is_new = index.upsert("a", vec![0.0, 1.0, 0.0]);
        assert!(!is_new); // update
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_vector_index_remove() {
        let mut index = VectorIndex::new(3);
        index.upsert("a", vec![1.0, 0.0, 0.0]);
        assert_eq!(index.len(), 1);

        let removed = index.remove("a");
        assert!(removed.is_some());
        assert_eq!(index.len(), 0);
        assert!(!index.contains("a"));

        let removed = index.remove("nonexistent");
        assert!(removed.is_none());
    }

    #[test]
    fn test_vector_index_search_topk() {
        let mut index = VectorIndex::new(3);
        index.upsert("a", vec![1.0, 0.0, 0.0]);
        index.upsert("b", vec![0.9, 0.1, 0.0]);
        index.upsert("c", vec![0.0, 1.0, 0.0]);
        index.upsert("d", vec![-1.0, 0.0, 0.0]);

        let query = vec![1.0, 0.0, 0.0];
        let results = index.search_topk(&query, 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "a");
        assert!((results[0].score - 1.0).abs() < 1e-6);
        assert_eq!(results[1].id, "b");
        assert!(results[1].score > 0.9);
    }

    #[test]
    fn test_vector_index_search_threshold() {
        let mut index = VectorIndex::new(3);
        index.upsert("a", vec![1.0, 0.0, 0.0]);
        index.upsert("b", vec![0.7, 0.7, 0.0]);
        index.upsert("c", vec![0.0, 1.0, 0.0]);
        index.upsert("d", vec![-1.0, 0.0, 0.0]);

        let query = vec![1.0, 0.0, 0.0];
        let results = index.search_threshold(&query, 0.5);

        assert_eq!(results.len(), 2); // "a" (1.0) and "b" (~0.707)
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_vector_index_batch_search() {
        let mut index = VectorIndex::new(3);
        index.upsert("a", vec![1.0, 0.0, 0.0]);
        index.upsert("b", vec![0.0, 1.0, 0.0]);
        index.upsert("c", vec![0.0, 0.0, 1.0]);

        let queries = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ];
        let results = index.batch_search(&queries, 1);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 1);
        assert_eq!(results[0][0].id, "a");
        assert_eq!(results[1][0].id, "b");
    }

    #[test]
    fn test_vector_index_clear() {
        let mut index = VectorIndex::new(3);
        index.upsert("a", vec![1.0, 0.0, 0.0]);
        index.upsert("b", vec![0.0, 1.0, 0.0]);
        assert_eq!(index.len(), 2);

        index.clear();
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_vector_index_zero_query() {
        let mut index = VectorIndex::new(3);
        index.upsert("a", vec![1.0, 0.0, 0.0]);

        let results = index.search_topk(&[0.0, 0.0, 0.0], 10);
        assert!(results.is_empty());
    }
}
