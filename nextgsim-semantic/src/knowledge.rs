//! Knowledge Graph and Shared Background (A18.2)
//!
//! Implements a shared semantic context that can be referenced by both
//! transmitter and receiver to improve compression efficiency.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Knowledge graph node representing a semantic concept
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptNode {
    /// Unique concept identifier
    pub id: String,
    /// Human-readable label
    pub label: String,
    /// Embedding vector for this concept
    pub embedding: Vec<f32>,
    /// Related concept IDs (edges in the graph)
    pub relations: Vec<String>,
    /// Concept frequency (for importance weighting)
    pub frequency: u64,
}

impl ConceptNode {
    /// Creates a new concept node
    pub fn new(id: String, label: String, embedding: Vec<f32>) -> Self {
        Self {
            id,
            label,
            embedding,
            relations: Vec::new(),
            frequency: 0,
        }
    }

    /// Adds a relation to another concept
    pub fn add_relation(&mut self, related_id: String) {
        if !self.relations.contains(&related_id) {
            self.relations.push(related_id);
        }
    }

    /// Increments the frequency count
    pub fn increment_frequency(&mut self) {
        self.frequency += 1;
    }
}

/// Shared knowledge graph
pub struct KnowledgeGraph {
    /// All concept nodes indexed by ID
    concepts: HashMap<String, ConceptNode>,
    /// Reverse index: embedding dimension -> concept IDs
    /// (for fast nearest-neighbor search)
    embedding_dim: usize,
}

impl KnowledgeGraph {
    /// Creates a new empty knowledge graph
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            concepts: HashMap::new(),
            embedding_dim,
        }
    }

    /// Adds a concept to the knowledge graph
    pub fn add_concept(&mut self, concept: ConceptNode) -> Result<(), String> {
        if concept.embedding.len() != self.embedding_dim {
            return Err(format!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.embedding_dim,
                concept.embedding.len()
            ));
        }

        self.concepts.insert(concept.id.clone(), concept);
        Ok(())
    }

    /// Gets a concept by ID
    pub fn get_concept(&self, id: &str) -> Option<&ConceptNode> {
        self.concepts.get(id)
    }

    /// Gets a mutable concept by ID
    pub fn get_concept_mut(&mut self, id: &str) -> Option<&mut ConceptNode> {
        self.concepts.get_mut(id)
    }

    /// Adds a bidirectional relation between two concepts
    pub fn add_relation(&mut self, id1: &str, id2: &str) -> Result<(), String> {
        if !self.concepts.contains_key(id1) {
            return Err(format!("Concept not found: {id1}"));
        }
        if !self.concepts.contains_key(id2) {
            return Err(format!("Concept not found: {id2}"));
        }

        if let Some(concept) = self.concepts.get_mut(id1) {
            concept.add_relation(id2.to_string());
        }

        if let Some(concept) = self.concepts.get_mut(id2) {
            concept.add_relation(id1.to_string());
        }

        Ok(())
    }

    /// Finds the k nearest concepts to a given embedding
    pub fn find_nearest(&self, embedding: &[f32], k: usize) -> Vec<(String, f32)> {
        let mut scored: Vec<(String, f32)> = self
            .concepts
            .values()
            .map(|concept| {
                let similarity = cosine_similarity(embedding, &concept.embedding);
                (concept.id.clone(), similarity)
            })
            .collect();

        // Sort by similarity (descending)
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// Encodes data using the knowledge graph
    /// Returns concept IDs that best match the input embedding
    pub fn encode_with_knowledge(&self, embedding: &[f32], k: usize) -> Vec<String> {
        self.find_nearest(embedding, k)
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    }

    /// Decodes concept IDs back to an embedding
    /// Combines embeddings of referenced concepts
    pub fn decode_with_knowledge(&self, concept_ids: &[String]) -> Vec<f32> {
        if concept_ids.is_empty() {
            return vec![0.0; self.embedding_dim];
        }

        let mut result = vec![0.0f32; self.embedding_dim];
        let mut count = 0;

        for id in concept_ids {
            if let Some(concept) = self.get_concept(id) {
                for (i, &val) in concept.embedding.iter().enumerate() {
                    result[i] += val;
                }
                count += 1;
            }
        }

        if count > 0 {
            for val in &mut result {
                *val /= count as f32;
            }
        }

        result
    }

    /// Returns the number of concepts in the graph
    pub fn num_concepts(&self) -> usize {
        self.concepts.len()
    }

    /// Returns the most frequent concepts
    pub fn top_concepts(&self, k: usize) -> Vec<&ConceptNode> {
        let mut concepts: Vec<_> = self.concepts.values().collect();
        concepts.sort_by(|a, b| b.frequency.cmp(&a.frequency));
        concepts.truncate(k);
        concepts
    }
}

/// Shared background context for semantic communication
pub struct SharedContext {
    /// Knowledge graph
    pub knowledge: KnowledgeGraph,
    /// Common vocabulary (for NLP tasks)
    pub vocabulary: HashMap<String, u32>,
    /// Context-specific priors (e.g., common objects in a scene)
    pub priors: HashMap<String, f32>,
}

impl SharedContext {
    /// Creates a new shared context
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            knowledge: KnowledgeGraph::new(embedding_dim),
            vocabulary: HashMap::new(),
            priors: HashMap::new(),
        }
    }

    /// Adds a word to the vocabulary
    pub fn add_word(&mut self, word: String, frequency: u32) {
        self.vocabulary.insert(word, frequency);
    }

    /// Adds a prior probability for a concept
    pub fn add_prior(&mut self, concept: String, probability: f32) {
        self.priors.insert(concept, probability);
    }

    /// Gets the prior probability for a concept
    pub fn get_prior(&self, concept: &str) -> f32 {
        self.priors.get(concept).copied().unwrap_or(0.0)
    }

    /// Compresses data using shared knowledge
    /// Returns (`concept_ids`, residual) where residual is what can't be captured by concepts
    pub fn compress_with_context(
        &self,
        embedding: &[f32],
        k: usize,
    ) -> (Vec<String>, Vec<f32>) {
        let concept_ids = self.knowledge.encode_with_knowledge(embedding, k);
        let reconstructed = self.knowledge.decode_with_knowledge(&concept_ids);

        // Compute residual
        let residual: Vec<f32> = embedding
            .iter()
            .zip(reconstructed.iter())
            .map(|(e, r)| e - r)
            .collect();

        (concept_ids, residual)
    }

    /// Decompresses data using shared knowledge
    pub fn decompress_with_context(
        &self,
        concept_ids: &[String],
        residual: &[f32],
    ) -> Vec<f32> {
        let base = self.knowledge.decode_with_knowledge(concept_ids);

        // Add residual
        base.iter()
            .zip(residual.iter())
            .map(|(b, r)| b + r)
            .collect()
    }
}

/// Cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..len {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        (dot / denom).clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concept_node() {
        let mut node = ConceptNode::new(
            "cat".to_string(),
            "Cat".to_string(),
            vec![0.1, 0.2, 0.3],
        );

        assert_eq!(node.relations.len(), 0);
        node.add_relation("dog".to_string());
        assert_eq!(node.relations.len(), 1);

        node.increment_frequency();
        assert_eq!(node.frequency, 1);
    }

    #[test]
    fn test_knowledge_graph_creation() {
        let kg = KnowledgeGraph::new(3);
        assert_eq!(kg.num_concepts(), 0);
        assert_eq!(kg.embedding_dim, 3);
    }

    #[test]
    fn test_add_concept() {
        let mut kg = KnowledgeGraph::new(3);

        let concept = ConceptNode::new("cat".to_string(), "Cat".to_string(), vec![0.1, 0.2, 0.3]);
        kg.add_concept(concept).unwrap();

        assert_eq!(kg.num_concepts(), 1);
        assert!(kg.get_concept("cat").is_some());
    }

    #[test]
    fn test_add_relation() {
        let mut kg = KnowledgeGraph::new(3);

        kg.add_concept(ConceptNode::new(
            "cat".to_string(),
            "Cat".to_string(),
            vec![0.1, 0.2, 0.3],
        ))
        .unwrap();

        kg.add_concept(ConceptNode::new(
            "dog".to_string(),
            "Dog".to_string(),
            vec![0.15, 0.25, 0.35],
        ))
        .unwrap();

        kg.add_relation("cat", "dog").unwrap();

        let cat = kg.get_concept("cat").unwrap();
        assert!(cat.relations.contains(&"dog".to_string()));

        let dog = kg.get_concept("dog").unwrap();
        assert!(dog.relations.contains(&"cat".to_string()));
    }

    #[test]
    fn test_find_nearest() {
        let mut kg = KnowledgeGraph::new(3);

        kg.add_concept(ConceptNode::new(
            "cat".to_string(),
            "Cat".to_string(),
            vec![1.0, 0.0, 0.0],
        ))
        .unwrap();

        kg.add_concept(ConceptNode::new(
            "dog".to_string(),
            "Dog".to_string(),
            vec![0.9, 0.1, 0.0],
        ))
        .unwrap();

        kg.add_concept(ConceptNode::new(
            "bird".to_string(),
            "Bird".to_string(),
            vec![0.0, 1.0, 0.0],
        ))
        .unwrap();

        let query = vec![0.95, 0.05, 0.0];
        let nearest = kg.find_nearest(&query, 2);

        assert_eq!(nearest.len(), 2);
        // Should be most similar to cat and dog
        assert!(nearest[0].0 == "cat" || nearest[0].0 == "dog");
    }

    #[test]
    fn test_encode_decode_with_knowledge() {
        let mut kg = KnowledgeGraph::new(3);

        kg.add_concept(ConceptNode::new(
            "cat".to_string(),
            "Cat".to_string(),
            vec![1.0, 0.0, 0.0],
        ))
        .unwrap();

        kg.add_concept(ConceptNode::new(
            "dog".to_string(),
            "Dog".to_string(),
            vec![0.0, 1.0, 0.0],
        ))
        .unwrap();

        let embedding = vec![0.7, 0.3, 0.0];
        let concept_ids = kg.encode_with_knowledge(&embedding, 2);

        assert_eq!(concept_ids.len(), 2);

        let decoded = kg.decode_with_knowledge(&concept_ids);
        assert_eq!(decoded.len(), 3);
    }

    #[test]
    fn test_shared_context() {
        let mut context = SharedContext::new(3);

        context.knowledge.add_concept(ConceptNode::new(
            "cat".to_string(),
            "Cat".to_string(),
            vec![1.0, 0.0, 0.0],
        )).unwrap();

        context.add_word("cat".to_string(), 100);
        context.add_prior("cat".to_string(), 0.5);

        assert_eq!(context.vocabulary.len(), 1);
        assert_eq!(context.get_prior("cat"), 0.5);
    }

    #[test]
    fn test_compress_decompress_with_context() {
        let mut context = SharedContext::new(3);

        context.knowledge.add_concept(ConceptNode::new(
            "cat".to_string(),
            "Cat".to_string(),
            vec![1.0, 0.0, 0.0],
        )).unwrap();

        context.knowledge.add_concept(ConceptNode::new(
            "dog".to_string(),
            "Dog".to_string(),
            vec![0.0, 1.0, 0.0],
        )).unwrap();

        let embedding = vec![0.8, 0.2, 0.1];
        let (concept_ids, residual) = context.compress_with_context(&embedding, 1);

        assert_eq!(concept_ids.len(), 1);
        assert_eq!(residual.len(), 3);

        let reconstructed = context.decompress_with_context(&concept_ids, &residual);
        assert_eq!(reconstructed.len(), 3);

        // Should be close to original
        let error: f32 = embedding
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(error < 0.5);
    }

    #[test]
    fn test_top_concepts() {
        let mut kg = KnowledgeGraph::new(3);

        let mut cat = ConceptNode::new("cat".to_string(), "Cat".to_string(), vec![1.0, 0.0, 0.0]);
        cat.frequency = 100;
        kg.add_concept(cat).unwrap();

        let mut dog = ConceptNode::new("dog".to_string(), "Dog".to_string(), vec![0.0, 1.0, 0.0]);
        dog.frequency = 50;
        kg.add_concept(dog).unwrap();

        let top = kg.top_concepts(1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].id, "cat");
    }
}
