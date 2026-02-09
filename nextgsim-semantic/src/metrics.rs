//! Semantic similarity metrics
//!
//! Provides standard quality metrics for evaluating how well the semantic
//! communication pipeline preserves the original information:
//!
//! - **Cosine similarity**: measures directional alignment of feature vectors.
//! - **Mean Squared Error (MSE)**: average squared difference per element.
//! - **Peak Signal-to-Noise Ratio (PSNR)**: log-scale quality metric.
//! - **Top-k accuracy**: fraction of the k most important features preserved.

/// Result of a full quality evaluation between original and reconstructed vectors.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QualityMetrics {
    /// Cosine similarity in [-1, 1]. 1.0 = perfect alignment.
    pub cosine_similarity: f32,
    /// Mean Squared Error. 0.0 = perfect reconstruction.
    pub mse: f32,
    /// Peak Signal-to-Noise Ratio in dB. Higher = better.
    /// `None` if MSE is zero (perfect reconstruction, PSNR is infinite).
    pub psnr_db: Option<f32>,
    /// Top-k accuracy in [0, 1]. Fraction of the top-k original features
    /// that remain in the top-k of the reconstructed vector.
    pub top_k_accuracy: f32,
    /// The k used for top-k accuracy.
    pub top_k: usize,
}

impl std::fmt::Display for QualityMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "cosine={:.4}, MSE={:.6}, PSNR={}dB, top-{} acc={:.4}",
            self.cosine_similarity,
            self.mse,
            self.psnr_db
                .map(|v| format!("{v:.2}"))
                .unwrap_or_else(|| "inf".to_string()),
            self.top_k,
            self.top_k_accuracy,
        )
    }
}

/// Computes cosine similarity between two vectors.
///
/// Returns a value in `[-1, 1]`. Returns `0.0` if either vector has zero norm.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
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

/// Computes Mean Squared Error between two vectors.
///
/// If the vectors have different lengths, the shorter length is used.
pub fn mse(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }

    let sum_sq: f32 = a[..len]
        .iter()
        .zip(b[..len].iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum();

    sum_sq / len as f32
}

/// Computes Peak Signal-to-Noise Ratio in dB.
///
/// `max_val` is the maximum possible value of the signal (e.g. `1.0` for
/// normalised data, `255.0` for 8-bit images).
///
/// Returns `None` if MSE is zero (perfect reconstruction).
pub fn psnr(a: &[f32], b: &[f32], max_val: f32) -> Option<f32> {
    let m = mse(a, b);
    if m == 0.0 {
        None // infinite PSNR
    } else {
        Some(10.0 * (max_val * max_val / m).log10())
    }
}

/// Computes top-k accuracy: the fraction of indices that appear in both the
/// top-k of `original` and the top-k of `reconstructed`.
///
/// Features are ranked by absolute value (largest = most important).
/// `k` is clamped to `min(original.len(), reconstructed.len())`.
pub fn top_k_accuracy(original: &[f32], reconstructed: &[f32], k: usize) -> f32 {
    let len = original.len().min(reconstructed.len());
    if len == 0 || k == 0 {
        return 0.0;
    }
    let k = k.min(len);

    let top_k_indices = |data: &[f32]| -> Vec<usize> {
        let mut indices: Vec<usize> = (0..data.len().min(len)).collect();
        indices.sort_by(|&a, &b| {
            data[b]
                .abs()
                .partial_cmp(&data[a].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices.truncate(k);
        indices
    };

    let orig_top = top_k_indices(original);
    let recon_top = top_k_indices(reconstructed);

    let matches = orig_top
        .iter()
        .filter(|idx| recon_top.contains(idx))
        .count();

    matches as f32 / k as f32
}

/// Computes all quality metrics in one pass.
///
/// `max_val` is the maximum possible signal value (used for PSNR).
/// `k` is the k for top-k accuracy.
pub fn evaluate(original: &[f32], reconstructed: &[f32], max_val: f32, k: usize) -> QualityMetrics {
    QualityMetrics {
        cosine_similarity: cosine_similarity(original, reconstructed),
        mse: mse(original, reconstructed),
        psnr_db: psnr(original, reconstructed, max_val),
        top_k_accuracy: top_k_accuracy(original, reconstructed, k),
        top_k: k,
    }
}

// ---------------------------------------------------------------------------
// Perceptual Quality Metrics (A18.8)
// ---------------------------------------------------------------------------

/// Computes Structural Similarity Index (SSIM) for 1-D signals
///
/// SSIM measures the perceived quality by comparing luminance, contrast, and structure.
/// For images, this should be applied with a sliding window. Here we provide a simplified
/// version for 1-D feature vectors.
///
/// Returns a value in [0, 1] where 1 = perfect similarity.
pub fn ssim(original: &[f32], reconstructed: &[f32], window_size: usize) -> f32 {
    let len = original.len().min(reconstructed.len());
    if len < window_size {
        // Fall back to simple correlation
        return cosine_similarity(original, reconstructed).abs();
    }

    let mut ssim_sum = 0.0f32;
    let mut count = 0;

    // Slide window across the signal
    for i in 0..=len.saturating_sub(window_size) {
        let window_orig = &original[i..i + window_size];
        let window_recon = &reconstructed[i..i + window_size];

        let local_ssim = compute_ssim_window(window_orig, window_recon);
        ssim_sum += local_ssim;
        count += 1;
    }

    if count == 0 {
        0.0
    } else {
        ssim_sum / count as f32
    }
}

/// Computes SSIM for a single window
fn compute_ssim_window(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len() as f32;

    // Compute means
    let mu_x = x.iter().sum::<f32>() / n;
    let mu_y = y.iter().sum::<f32>() / n;

    // Compute variances
    let var_x = x.iter().map(|v| (v - mu_x).powi(2)).sum::<f32>() / n;
    let var_y = y.iter().map(|v| (v - mu_y).powi(2)).sum::<f32>() / n;

    // Compute covariance
    let cov_xy = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| (a - mu_x) * (b - mu_y))
        .sum::<f32>()
        / n;

    // SSIM constants (to avoid division by zero)
    let c1 = 0.01f32.powi(2);
    let c2 = 0.03f32.powi(2);

    // SSIM formula
    let luminance = (2.0 * mu_x * mu_y + c1) / (mu_x.powi(2) + mu_y.powi(2) + c1);
    let contrast = (2.0 * var_x.sqrt() * var_y.sqrt() + c2) / (var_x + var_y + c2);
    let structure = (cov_xy + c2 / 2.0) / (var_x.sqrt() * var_y.sqrt() + c2 / 2.0);

    luminance * contrast * structure
}

/// Computes Multi-Scale SSIM (MS-SSIM)
///
/// Evaluates SSIM at multiple scales by downsampling the signal.
/// More robust to different types of distortions than single-scale SSIM.
pub fn ms_ssim(original: &[f32], reconstructed: &[f32], num_scales: usize) -> f32 {
    let mut scores = Vec::new();
    let mut curr_orig = original.to_vec();
    let mut curr_recon = reconstructed.to_vec();

    for _ in 0..num_scales {
        if curr_orig.len() < 8 {
            break;
        }

        let score = ssim(&curr_orig, &curr_recon, 8.min(curr_orig.len()));
        scores.push(score);

        // Downsample by factor of 2
        curr_orig = downsample(&curr_orig, 2);
        curr_recon = downsample(&curr_recon, 2);
    }

    if scores.is_empty() {
        0.0
    } else {
        // Geometric mean of scales
        let product: f32 = scores.iter().product();
        product.powf(1.0 / scores.len() as f32)
    }
}

/// Downsamples a signal by a factor
fn downsample(signal: &[f32], factor: usize) -> Vec<f32> {
    if factor <= 1 {
        return signal.to_vec();
    }

    let new_len = signal.len() / factor;
    let mut downsampled = Vec::with_capacity(new_len);

    for i in 0..new_len {
        let start = i * factor;
        let end = ((i + 1) * factor).min(signal.len());
        let chunk = &signal[start..end];
        let mean = chunk.iter().sum::<f32>() / chunk.len() as f32;
        downsampled.push(mean);
    }

    downsampled
}

/// Computes perceptual distance using a weighted combination of metrics
///
/// Combines MSE (pixel accuracy) with SSIM (structural similarity) to better
/// reflect human perception of quality.
///
/// Returns a value where lower = better quality (0 = perfect).
pub fn perceptual_distance(original: &[f32], reconstructed: &[f32]) -> f32 {
    let mse_val = mse(original, reconstructed);
    let ssim_val = ssim(original, reconstructed, 8);

    // Weighted combination: balance pixel accuracy and structural similarity
    // Lower MSE and higher SSIM both indicate better quality
    let alpha = 0.5;
    alpha * mse_val + (1.0 - alpha) * (1.0 - ssim_val)
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
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_mse_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!(mse(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_mse_known_value() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 3.0, 4.0];
        // MSE = ((1)^2 + (1)^2 + (1)^2) / 3 = 1.0
        assert!((mse(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_psnr_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!(psnr(&a, &b, 1.0).is_none()); // Infinite PSNR
    }

    #[test]
    fn test_psnr_known_value() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 3.0, 4.0];
        // MSE = 1.0, max_val = 10.0
        // PSNR = 10 * log10(100 / 1) = 20 dB
        let p = psnr(&a, &b, 10.0).expect("Expected finite PSNR");
        assert!((p - 20.0).abs() < 1e-4);
    }

    #[test]
    fn test_top_k_accuracy_identical() {
        let a = vec![3.0, 1.0, 2.0, 0.5];
        let b = vec![3.0, 1.0, 2.0, 0.5];
        assert!((top_k_accuracy(&a, &b, 2) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_top_k_accuracy_no_overlap() {
        // Top-2 of a: indices 0, 2 (values 3.0, 2.0)
        let a = vec![3.0, 0.1, 2.0, 0.0];
        // Top-2 of b: indices 1, 3 (values 5.0, 4.0)
        let b = vec![0.0, 5.0, 0.0, 4.0];
        assert!(top_k_accuracy(&a, &b, 2).abs() < 1e-6);
    }

    #[test]
    fn test_top_k_accuracy_partial_overlap() {
        // Top-3 of a: indices 0, 2, 1 (values 5.0, 3.0, 2.0)
        let a = vec![5.0, 2.0, 3.0, 0.1];
        // Top-3 of b: indices 0, 3, 2 (values 5.0, 4.0, 3.0)
        let b = vec![5.0, 0.0, 3.0, 4.0];
        // Overlap: {0, 2} -> 2 out of 3
        let acc = top_k_accuracy(&a, &b, 3);
        assert!((acc - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_evaluate_comprehensive() {
        let original = vec![1.0, 0.5, 0.8, 0.2, 0.9];
        let reconstructed = vec![0.9, 0.6, 0.75, 0.25, 0.85];

        let m = evaluate(&original, &reconstructed, 1.0, 3);

        assert!(m.cosine_similarity > 0.99);
        assert!(m.mse < 0.01);
        assert!(m.psnr_db.is_some());
        assert!(m.top_k_accuracy >= 0.0 && m.top_k_accuracy <= 1.0);
        assert_eq!(m.top_k, 3);
    }

    #[test]
    fn test_empty_vectors() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
        assert_eq!(mse(&[], &[]), 0.0);
        assert!(psnr(&[], &[], 1.0).is_none());
        assert_eq!(top_k_accuracy(&[], &[], 5), 0.0);
    }
}
