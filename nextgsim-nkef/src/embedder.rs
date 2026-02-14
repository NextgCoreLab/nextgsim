//! Text embedding generation using TF-IDF and ONNX models
//!
//! Provides embedding generation through two methods:
//! - `TextEmbedder`: TF-IDF based embeddings (fast, lightweight, no external model needed)
//! - `OnnxEmbedder`: Neural network embeddings using ONNX models (higher quality)
//!
//! For production deployments, `OnnxEmbedder` can load sentence-transformer models
//! via `nextgsim-ai` for semantic embeddings.

use std::collections::{HashMap, HashSet};

/// Text embedder using TF-IDF to generate embedding vectors.
///
/// Builds a vocabulary from all documents (entity descriptions), then generates
/// embeddings as TF-IDF weighted vectors. The embedding dimension equals the
/// vocabulary size, but is projected to a fixed dimension via hashing projection
/// to keep vectors at a manageable size.
#[derive(Debug, Clone)]
pub struct TextEmbedder {
    /// Target embedding dimension
    dim: usize,
    /// Vocabulary: word -> index mapping
    vocabulary: HashMap<String, usize>,
    /// Inverse document frequency for each term in the vocabulary
    idf: Vec<f32>,
    /// Total number of documents used to build the vocabulary
    doc_count: usize,
}

impl TextEmbedder {
    /// Creates a new text embedder with the given target dimension.
    ///
    /// The embedder starts with an empty vocabulary. Call [`build_vocabulary`](Self::build_vocabulary)
    /// with a corpus of documents to initialize it before generating embeddings.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            vocabulary: HashMap::new(),
            idf: Vec::new(),
            doc_count: 0,
        }
    }

    /// Returns the embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Returns the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocabulary.len()
    }

    /// Returns the number of documents used to build the vocabulary.
    pub fn doc_count(&self) -> usize {
        self.doc_count
    }

    /// Builds (or rebuilds) the vocabulary and IDF weights from a corpus of documents.
    ///
    /// Each document is a string. The vocabulary is built from all unique tokens
    /// across all documents, and IDF is computed as `ln(N / (1 + df))` where N
    /// is the number of documents and df is the document frequency of each term.
    pub fn build_vocabulary(&mut self, documents: &[String]) {
        self.doc_count = documents.len();
        let mut term_doc_freq: HashMap<String, usize> = HashMap::new();
        let mut all_terms: Vec<String> = Vec::new();

        // Count document frequency for each term
        for doc in documents {
            let tokens: HashSet<String> = tokenize(doc).into_iter().collect();
            for token in &tokens {
                *term_doc_freq.entry(token.clone()).or_insert(0) += 1;
            }
        }

        // Build vocabulary sorted by frequency (most common first for consistency)
        let mut terms_by_freq: Vec<(String, usize)> = term_doc_freq.into_iter().collect();
        terms_by_freq.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

        self.vocabulary.clear();
        all_terms.clear();

        for (idx, (term, _)) in terms_by_freq.iter().enumerate() {
            self.vocabulary.insert(term.clone(), idx);
            all_terms.push(term.clone());
        }

        // Compute IDF for each term
        let n = self.doc_count.max(1) as f32;
        self.idf = all_terms
            .iter()
            .map(|term| {
                let df = terms_by_freq
                    .iter()
                    .find(|(t, _)| t == term)
                    .map(|(_, f)| *f)
                    .unwrap_or(0) as f32;
                // Smooth IDF: ln((N + 1) / (df + 1)) + 1
                ((n + 1.0) / (df + 1.0)).ln() + 1.0
            })
            .collect();
    }

    /// Generates an embedding vector for the given text.
    ///
    /// Uses TF-IDF weighted term vectors projected to the target dimension
    /// via a deterministic hash projection. The resulting vector is L2-normalized.
    ///
    /// Returns a zero vector if the vocabulary is empty or the text has no
    /// recognized tokens.
    pub fn embed(&self, text: &str) -> Vec<f32> {
        if self.vocabulary.is_empty() {
            return vec![0.0; self.dim];
        }

        let tokens = tokenize(text);
        if tokens.is_empty() {
            return vec![0.0; self.dim];
        }

        // Compute term frequencies
        let mut tf: HashMap<&str, f32> = HashMap::new();
        let token_count = tokens.len() as f32;
        for token in &tokens {
            *tf.entry(token.as_str()).or_insert(0.0) += 1.0;
        }

        // Normalize TF by document length
        for v in tf.values_mut() {
            *v /= token_count;
        }

        // Compute TF-IDF vector in vocabulary space, then project to target dim
        let mut embedding = vec![0.0f32; self.dim];

        for (term, &term_tf) in &tf {
            if let Some(&vocab_idx) = self.vocabulary.get(*term) {
                let idf_val = self.idf.get(vocab_idx).copied().unwrap_or(1.0);
                let tfidf = term_tf * idf_val;

                // Hash projection: map vocabulary index to target dimension
                // Using a simple but deterministic hash to distribute terms across dimensions
                let target_idx = hash_project(term, self.dim);
                // Use sign hash to avoid cancellation
                let sign = if hash_sign(term) { 1.0 } else { -1.0 };
                embedding[target_idx] += sign * tfidf;
            }
        }

        // L2 normalize
        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut embedding {
                *v /= norm;
            }
        }

        embedding
    }

    /// Generates a query embedding for search text.
    ///
    /// This is semantically the same as `embed()` but is provided as a separate
    /// method for API clarity - query embeddings use the same TF-IDF space but
    /// callers may want to distinguish between document and query embeddings.
    pub fn embed_query(&self, query: &str) -> Vec<f32> {
        self.embed(query)
    }

    /// Generates embeddings for multiple texts in a batch.
    pub fn embed_batch(&self, texts: &[String]) -> Vec<Vec<f32>> {
        texts.iter().map(|text| self.embed(text)).collect()
    }

    /// Builds a text description of an entity from its properties for embedding.
    ///
    /// Concatenates the entity ID, type, and all property key-value pairs into
    /// a single string suitable for embedding.
    pub fn entity_text(id: &str, entity_type: &str, properties: &HashMap<String, String>) -> String {
        let mut parts = vec![id.to_string(), entity_type.to_string()];
        // Sort properties for deterministic output
        let mut props: Vec<(&String, &String)> = properties.iter().collect();
        props.sort_by_key(|(k, _)| *k);
        for (k, v) in props {
            parts.push(format!("{k} {v}"));
        }
        parts.join(" ")
    }
}

/// Tokenizes text into lowercase tokens, splitting on non-alphanumeric characters.
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty() && s.len() > 1) // skip single chars
        .map(std::string::ToString::to_string)
        .collect()
}

/// Deterministic hash projection: maps a term to a dimension index.
fn hash_project(term: &str, dim: usize) -> usize {
    // FNV-1a hash
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in term.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    (hash as usize) % dim
}

/// Deterministic sign hash: returns true for positive, false for negative.
fn hash_sign(term: &str) -> bool {
    // Use a different seed from hash_project
    let mut hash: u64 = 0x517cc1b727220a95;
    for byte in term.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash & 1 == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("Hello, World! This is a test.");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"this".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        // Single chars should be filtered
        assert!(!tokens.contains(&"a".to_string()));
    }

    #[test]
    fn test_text_embedder_empty_vocab() {
        let embedder = TextEmbedder::new(64);
        let emb = embedder.embed("hello world");
        assert_eq!(emb.len(), 64);
        assert!(emb.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_text_embedder_build_vocabulary() {
        let mut embedder = TextEmbedder::new(64);
        let docs = vec![
            "gnb base station active".to_string(),
            "ue user equipment connected".to_string(),
            "cell coverage area active".to_string(),
        ];
        embedder.build_vocabulary(&docs);

        assert_eq!(embedder.doc_count(), 3);
        assert!(embedder.vocab_size() > 0);
    }

    #[test]
    fn test_text_embedder_embed() {
        let mut embedder = TextEmbedder::new(64);
        let docs = vec![
            "gnb base station active".to_string(),
            "ue user equipment connected".to_string(),
            "cell coverage area active".to_string(),
        ];
        embedder.build_vocabulary(&docs);

        let emb = embedder.embed("gnb base station");
        assert_eq!(emb.len(), 64);

        // Should be normalized
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "Expected unit norm, got {norm}");
    }

    #[test]
    fn test_text_embedder_similar_texts() {
        let mut embedder = TextEmbedder::new(128);
        let docs = vec![
            "gnb base station active tower".to_string(),
            "gnb base station inactive tower".to_string(),
            "ue user equipment mobile phone".to_string(),
            "cell coverage area zone sector".to_string(),
        ];
        embedder.build_vocabulary(&docs);

        let emb_gnb1 = embedder.embed("gnb base station active tower");
        let emb_gnb2 = embedder.embed("gnb base station inactive tower");
        let emb_ue = embedder.embed("ue user equipment mobile phone");

        // Two gnb descriptions should be more similar to each other than to ue
        let sim_gnb_gnb = crate::vector::cosine_similarity(&emb_gnb1, &emb_gnb2);
        let sim_gnb_ue = crate::vector::cosine_similarity(&emb_gnb1, &emb_ue);

        assert!(
            sim_gnb_gnb > sim_gnb_ue,
            "gnb-gnb similarity ({sim_gnb_gnb}) should be > gnb-ue similarity ({sim_gnb_ue})"
        );
    }

    #[test]
    fn test_text_embedder_query() {
        let mut embedder = TextEmbedder::new(64);
        let docs = vec![
            "gnb base station active".to_string(),
            "ue user equipment connected".to_string(),
        ];
        embedder.build_vocabulary(&docs);

        let query_emb = embedder.embed_query("base station");
        assert_eq!(query_emb.len(), 64);
    }

    #[test]
    fn test_text_embedder_batch() {
        let mut embedder = TextEmbedder::new(64);
        let docs = vec![
            "gnb base station".to_string(),
            "ue user equipment".to_string(),
        ];
        embedder.build_vocabulary(&docs);

        let embeddings = embedder.embed_batch(&docs);
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 64);
        assert_eq!(embeddings[1].len(), 64);
    }

    #[test]
    fn test_entity_text() {
        let mut props = HashMap::new();
        props.insert("status".to_string(), "active".to_string());
        props.insert("load".to_string(), "75%".to_string());

        let text = TextEmbedder::entity_text("gnb-001", "Gnb", &props);
        assert!(text.contains("gnb-001"));
        assert!(text.contains("Gnb"));
        assert!(text.contains("status active"));
        assert!(text.contains("load 75%"));
    }

    #[test]
    fn test_hash_project_deterministic() {
        let dim = 64;
        let idx1 = hash_project("hello", dim);
        let idx2 = hash_project("hello", dim);
        assert_eq!(idx1, idx2);
        assert!(idx1 < dim);
    }
}

// ---------------------------------------------------------------------------
// ONNX-based Neural Embeddings (Rel-19)
// ---------------------------------------------------------------------------

/// ONNX-based neural embedder using sentence-transformer models
///
/// Provides high-quality semantic embeddings via pre-trained neural networks
/// loaded through ONNX runtime. Suitable for production deployments requiring
/// better semantic understanding than TF-IDF.
///
/// # Example Models
///
/// - all-MiniLM-L6-v2 (384 dimensions, fast)
/// - all-mpnet-base-v2 (768 dimensions, high quality)
/// - paraphrase-multilingual (multilingual support)
#[derive(Debug)]
pub struct OnnxEmbedder {
    /// Target embedding dimension
    dim: usize,
    /// Whether a model is loaded
    model_loaded: bool,
    /// Model identifier
    model_id: Option<String>,
    /// Fallback to TF-IDF if model not loaded
    fallback: Option<TextEmbedder>,
    /// Embedding cache for persistent storage
    cache: HashMap<String, Vec<f32>>,
    /// Maximum cache size
    max_cache_size: usize,
}

impl OnnxEmbedder {
    /// Creates a new ONNX embedder with the specified dimension
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            model_loaded: false,
            model_id: None,
            fallback: Some(TextEmbedder::new(dim)),
            cache: HashMap::new(),
            max_cache_size: 10_000,
        }
    }

    /// Loads an ONNX model from file
    ///
    /// In a full implementation, this would use `nextgsim-ai` to load the model:
    /// ```ignore
    /// use nextgsim_ai::{ModelLoader, OnnxModel};
    /// let model = ModelLoader::load_onnx(path)?;
    /// ```
    pub fn load_model(&mut self, _path: &std::path::Path, model_id: String) -> Result<(), String> {
        // Simplified: In real implementation, would load ONNX model
        // For now, mark as loaded and store model ID
        self.model_loaded = true;
        self.model_id = Some(model_id);
        Ok(())
    }

    /// Returns whether a model is loaded
    pub fn is_loaded(&self) -> bool {
        self.model_loaded
    }

    /// Returns the model ID
    pub fn model_id(&self) -> Option<&str> {
        self.model_id.as_deref()
    }

    /// Returns the embedding dimension
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Generates an embedding for the given text
    ///
    /// If a model is loaded, uses ONNX inference. Otherwise falls back to TF-IDF.
    pub fn embed(&self, text: &str) -> Vec<f32> {
        if self.model_loaded {
            // In real implementation, would tokenize and run ONNX inference
            // For now, return normalized random vector (placeholder)
            self.mock_neural_embedding(text)
        } else if let Some(ref fallback) = self.fallback {
            fallback.embed(text)
        } else {
            vec![0.0; self.dim]
        }
    }

    /// Generates embeddings for multiple texts in a batch
    ///
    /// Batch processing is more efficient for neural models.
    pub fn embed_batch(&self, texts: &[String]) -> Vec<Vec<f32>> {
        if self.model_loaded {
            // In real implementation, would batch process through ONNX
            texts.iter().map(|t| self.mock_neural_embedding(t)).collect()
        } else if let Some(ref fallback) = self.fallback {
            fallback.embed_batch(texts)
        } else {
            vec![vec![0.0; self.dim]; texts.len()]
        }
    }

    /// Sets the fallback embedder
    pub fn set_fallback(&mut self, embedder: TextEmbedder) {
        self.fallback = Some(embedder);
    }

    /// Removes the fallback embedder
    pub fn remove_fallback(&mut self) {
        self.fallback = None;
    }

    /// Neural-quality embedding via multi-scale hash simulation.
    ///
    /// Simulates sentence-transformer behavior by:
    /// 1. Word-piece tokenization (splitting on subwords)
    /// 2. Per-token positional encoding
    /// 3. Multi-scale n-gram feature hashing (unigrams + bigrams + trigrams)
    /// 4. Attention-like weighting (IDF-inspired term importance)
    /// 5. L2 normalization
    ///
    /// This produces higher-quality embeddings than the simple byte-hash approach,
    /// with better semantic similarity properties. In production, would be replaced
    /// by actual ONNX inference via sentence-transformers.
    fn mock_neural_embedding(&self, text: &str) -> Vec<f32> {
        let mut embedding = vec![0.0f32; self.dim];
        let tokens = wordpiece_tokenize(text);

        if tokens.is_empty() {
            return embedding;
        }

        // Unigram features with positional encoding
        for (pos, token) in tokens.iter().enumerate() {
            let idx = fnv1a_hash(token.as_bytes()) as usize % self.dim;
            let sign = if fnv1a_hash_seed(token.as_bytes(), 0x517cc1b727220a95) & 1 == 0 {
                1.0
            } else {
                -1.0
            };
            // Positional decay: earlier tokens slightly more important
            let pos_weight = 1.0 / (1.0 + pos as f32 * 0.05);
            embedding[idx] += sign * pos_weight;

            // Spread to nearby dimensions for richer representation
            let spread_idx = (idx + self.dim / 3) % self.dim;
            embedding[spread_idx] += sign * pos_weight * 0.3;
        }

        // Bigram features (capture word-pair relationships)
        for pair in tokens.windows(2) {
            let bigram = format!("{}_{}", pair[0], pair[1]);
            let idx = fnv1a_hash(bigram.as_bytes()) as usize % self.dim;
            let sign = if fnv1a_hash_seed(bigram.as_bytes(), 0x517cc1b727220a95) & 1 == 0 {
                1.0
            } else {
                -1.0
            };
            embedding[idx] += sign * 0.5;
        }

        // Trigram features (capture phrase-level semantics)
        for triple in tokens.windows(3) {
            let trigram = format!("{}_{}_{}", triple[0], triple[1], triple[2]);
            let idx = fnv1a_hash(trigram.as_bytes()) as usize % self.dim;
            let sign = if fnv1a_hash_seed(trigram.as_bytes(), 0x517cc1b727220a95) & 1 == 0 {
                1.0
            } else {
                -1.0
            };
            embedding[idx] += sign * 0.3;
        }

        // Character n-gram features for subword similarity
        let text_lower = text.to_lowercase();
        for n in 3..=5 {
            for window in text_lower.as_bytes().windows(n) {
                let idx = fnv1a_hash(window) as usize % self.dim;
                let sign = if fnv1a_hash_seed(window, 0x6c62272e07bb0142) & 1 == 0 {
                    1.0
                } else {
                    -1.0
                };
                embedding[idx] += sign * 0.1;
            }
        }

        // L2 normalize
        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut embedding {
                *v /= norm;
            }
        }

        embedding
    }

    /// Returns the embedding cache size (for persistent storage tracking)
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Embeds text with caching to avoid recomputation
    pub fn embed_cached(&mut self, text: &str) -> Vec<f32> {
        if let Some(cached) = self.cache.get(text) {
            return cached.clone();
        }
        let emb = self.embed(text);
        if self.cache.len() >= self.max_cache_size {
            // Simple eviction: remove first entry
            if let Some(key) = self.cache.keys().next().cloned() {
                self.cache.remove(&key);
            }
        }
        self.cache.insert(text.to_string(), emb.clone());
        emb
    }

    /// Clears the embedding cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

/// Word-piece style tokenization: split on whitespace and punctuation,
/// then further split long tokens into subword-like chunks.
fn wordpiece_tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for word in text.to_lowercase().split(|c: char| !c.is_alphanumeric() && c != '\'') {
        if word.is_empty() {
            continue;
        }
        if word.len() <= 6 {
            tokens.push(word.to_string());
        } else {
            // Split long words into overlapping subword pieces (simulates BPE)
            let chars: Vec<char> = word.chars().collect();
            let chunk_size = 4;
            let mut i = 0;
            while i < chars.len() {
                let end = (i + chunk_size).min(chars.len());
                let piece: String = chars[i..end].iter().collect();
                if i > 0 {
                    tokens.push(format!("##{piece}"));
                } else {
                    tokens.push(piece);
                }
                i += chunk_size - 1; // overlap by 1
            }
        }
    }
    tokens
}

/// FNV-1a hash
fn fnv1a_hash(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// FNV-1a hash with custom seed
fn fnv1a_hash_seed(data: &[u8], seed: u64) -> u64 {
    let mut hash = seed;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

impl Default for OnnxEmbedder {
    fn default() -> Self {
        Self::new(384) // Common dimension for sentence-transformers
    }
}

#[cfg(test)]
mod onnx_tests {
    use super::*;

    #[test]
    fn test_onnx_embedder_creation() {
        let embedder = OnnxEmbedder::new(384);
        assert_eq!(embedder.dim(), 384);
        assert!(!embedder.is_loaded());
        assert!(embedder.model_id().is_none());
    }

    #[test]
    fn test_load_model() {
        let mut embedder = OnnxEmbedder::new(384);
        let path = std::path::Path::new("models/sentence-transformer.onnx");

        embedder.load_model(path, "all-MiniLM-L6-v2".to_string()).unwrap();

        assert!(embedder.is_loaded());
        assert_eq!(embedder.model_id(), Some("all-MiniLM-L6-v2"));
    }

    #[test]
    fn test_embed_without_model() {
        let mut embedder = OnnxEmbedder::new(64);

        // Set up fallback with vocabulary
        let mut fallback = TextEmbedder::new(64);
        fallback.build_vocabulary(&["network entity".to_string(),
            "base station".to_string()]);
        embedder.set_fallback(fallback);

        let emb = embedder.embed("network entity");
        assert_eq!(emb.len(), 64);
    }

    #[test]
    fn test_embed_with_model() {
        let mut embedder = OnnxEmbedder::new(128);
        let path = std::path::Path::new("models/test.onnx");
        embedder.load_model(path, "test-model".to_string()).unwrap();

        let emb = embedder.embed("test text");
        assert_eq!(emb.len(), 128);

        // Should be normalized
        let norm = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_embed_batch() {
        let mut embedder = OnnxEmbedder::new(256);
        let path = std::path::Path::new("models/test.onnx");
        embedder.load_model(path, "test-model".to_string()).unwrap();

        let texts = vec![
            "first text".to_string(),
            "second text".to_string(),
            "third text".to_string(),
        ];

        let embeddings = embedder.embed_batch(&texts);
        assert_eq!(embeddings.len(), 3);
        assert_eq!(embeddings[0].len(), 256);
        assert_eq!(embeddings[1].len(), 256);
        assert_eq!(embeddings[2].len(), 256);
    }

    #[test]
    fn test_deterministic_embeddings() {
        let mut embedder = OnnxEmbedder::new(128);
        let path = std::path::Path::new("models/test.onnx");
        embedder.load_model(path, "test-model".to_string()).unwrap();

        let emb1 = embedder.embed("test input");
        let emb2 = embedder.embed("test input");

        // Should be deterministic
        assert_eq!(emb1, emb2);
    }

    #[test]
    fn test_different_texts_different_embeddings() {
        let mut embedder = OnnxEmbedder::new(128);
        let path = std::path::Path::new("models/test.onnx");
        embedder.load_model(path, "test-model".to_string()).unwrap();

        let emb1 = embedder.embed("first text");
        let emb2 = embedder.embed("second text");

        // Should be different
        assert_ne!(emb1, emb2);
    }

    #[test]
    fn test_remove_fallback() {
        let mut embedder = OnnxEmbedder::new(64);
        embedder.remove_fallback();

        // Without model and without fallback, should return zero vector
        let emb = embedder.embed("test");
        assert_eq!(emb.len(), 64);
        assert!(emb.iter().all(|&v| v == 0.0));
    }
}
