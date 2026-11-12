//! Rate-adaptive feature encoding against a network-signalled Session-AMBR
//! (demo D444A, nextgsim#477). **Non-normative.**
//!
//! A UE application streams a 128-dimensional feature vector per sample over
//! its PDU session. When the network lowers the session's uplink Session-AMBR
//! (TS 24.501 §6.3.2.2, PDU SESSION MODIFICATION COMMAND), the application
//! keeps only as many features as the new rate carries
//! ([`keep_for_link_budget`]), most important first.
//!
//! What is here, and what it is not:
//! - [`FeatureGenerator`] is a **pinned synthetic task**: eight classes whose
//!   means lie along a decaying spectrum in a randomly rotated basis. Any
//!   accuracy gap between encoders comes from that spectrum. It is ground truth
//!   of the generator, not a semantic-communication result.
//! - [`PcaEncoder`] is a linear PCA fitted on generator samples (the "semantic"
//!   codec of the demo; swappable through [`RateAdaptiveEncoder`]).
//! - [`TopVarianceRaw`] (the fair baseline: the highest-variance raw
//!   dimensions) and [`MeanPool`] (an ablation) are scored at the **same keep,
//!   hence the same bytes**.
//! - [`NearestCentroid`] scores every encoder the same way.
//! - The wire datagram ([`FeatureDatagram`]) carries the sample's ground-truth
//!   label so the sink can score it. That label is a scoring aid, not payload.
//!
//! Everything is deterministic from one seed, so the UE and the sink rebuild
//! the same model independently and the sink can regenerate any sample by its
//! sequence number to score the baselines on exactly the samples it received.

use std::collections::HashMap;
use std::fmt;

/// The demo key (nextgsim#458 toggle convention).
pub const D444A_KEY: &str = "d444a-ambr-adapt";
/// Its modes. `semantic`: the UE application follows the uplink Session-AMBR
/// with the PCA codec (keep 128 -> 40). `raw`: the control; the application
/// ignores the Session-AMBR, sends all 128 raw features and is policed by the
/// UPF's QER.
pub const D444A_MODES: &[&str] = &["semantic", "raw"];

/// Features per sample.
pub const FEATURE_DIM: usize = 128;
/// Classes of the synthetic task.
pub const CLASSES: usize = 8;
/// IPv4 + UDP header bytes per datagram. The UPF's QER polices the inner IP
/// packet, so the link budget counts them.
pub const IP_UDP_OVERHEAD: usize = 28;
/// Bytes of the [`FeatureDatagram`] header.
pub const HEADER_LEN: usize = 16;
/// Bytes per transmitted feature (i16 fixed point with a per-datagram scale).
pub const BYTES_PER_FEATURE: usize = 2;
/// Fraction of the uplink Session-AMBR the application plans to use, leaving
/// room for the policer's burst behaviour and other traffic.
pub const HEADROOM: f64 = 0.8;
/// `keep` is a multiple of this.
pub const KEEP_STEP: usize = 8;
/// The smallest `keep` the application sends.
pub const MIN_KEEP: usize = 8;
/// The generator seed the shipped demo configs use.
pub const DEFAULT_SEED: u64 = 444;
/// Accuracy floor of the PCA encoder at `keep` 40 on the pinned generator
/// (`DEFAULT_SEED`): the acceptance check's "accuracy ≥ the pinned floor".
pub const PINNED_ACCURACY_FLOOR_KEEP40: f64 = 0.90;

/// Training samples for the PCA fit and each nearest-centroid fit.
const TRAIN_SAMPLES: u64 = 2048;
/// Training samples come from a sequence range disjoint from the stream's.
const TRAIN_SEQ_BASE: u64 = 1 << 40;
/// Amplitude of the class means along latent component 0.
const MEAN_AMPLITUDE: f64 = 1.6;
/// Per-component decay of the class-mean amplitude (the pinned spectrum).
const MEAN_DECAY: f64 = 0.93;
/// Within-class noise standard deviation (isotropic in the latent basis).
const NOISE_STD: f64 = 1.0;

// ---------------------------------------------------------------------------
// Deterministic randomness
// ---------------------------------------------------------------------------

/// SplitMix64: small, fast, and identical on every build, so the UE and the
/// sink draw the same numbers from the same seed.
#[derive(Debug, Clone)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in (0, 1].
    fn next_unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64
    }

    /// Standard normal (Box-Muller).
    fn gaussian(&mut self) -> f64 {
        let u1 = self.next_unit();
        let u2 = self.next_unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// A per-sample seed, so sample `seq` can be drawn without drawing the others.
fn sample_seed(seed: u64, seq: u64) -> u64 {
    SplitMix64::new(seed ^ seq.wrapping_mul(0xD6E8_FEB8_6659_FD93)).next_u64()
}

// ---------------------------------------------------------------------------
// The pinned synthetic task
// ---------------------------------------------------------------------------

/// Seeded synthetic classification task over [`FEATURE_DIM`] features.
///
/// Latent sample `z = mu[label] + n`, with `mu[c][k] = A * d^k * g[c][k]`
/// (`g` standard normal, so the discriminative energy decays along `k`) and
/// isotropic noise `n`. The transmitted features are `x = R z` for a random
/// orthonormal `R`, so no single raw dimension lines up with the spectrum.
#[derive(Debug, Clone)]
pub struct FeatureGenerator {
    seed: u64,
    /// Row-major `FEATURE_DIM x FEATURE_DIM` orthonormal rotation.
    rotation: Vec<f64>,
    /// `CLASSES x FEATURE_DIM` latent class means.
    means: Vec<Vec<f64>>,
}

impl FeatureGenerator {
    /// The generator for `seed`.
    pub fn new(seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let n = FEATURE_DIM;
        // Random Gaussian matrix, orthonormalised row by row (modified Gram-Schmidt).
        let mut rows: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..n).map(|_| rng.gaussian()).collect())
            .collect();
        for i in 0..n {
            for j in 0..i {
                let dot: f64 = (0..n).map(|k| rows[i][k] * rows[j][k]).sum();
                let (head, tail) = rows.split_at_mut(i);
                for k in 0..n {
                    tail[0][k] -= dot * head[j][k];
                }
            }
            let norm = rows[i].iter().map(|v| v * v).sum::<f64>().sqrt();
            for v in &mut rows[i] {
                *v /= norm;
            }
        }
        let rotation = rows.into_iter().flatten().collect();
        let means = (0..CLASSES)
            .map(|_| {
                (0..n)
                    .map(|k| MEAN_AMPLITUDE * MEAN_DECAY.powi(k as i32) * rng.gaussian())
                    .collect()
            })
            .collect();
        Self {
            seed,
            rotation,
            means,
        }
    }

    /// The seed this generator was built from.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Sample number `seq`: its ground-truth label and its features.
    pub fn sample(&self, seq: u64) -> (u8, Vec<f32>) {
        let mut rng = SplitMix64::new(sample_seed(self.seed, seq));
        let label = (rng.next_u64() % CLASSES as u64) as usize;
        let z: Vec<f64> = self.means[label]
            .iter()
            .map(|m| m + NOISE_STD * rng.gaussian())
            .collect();
        let n = FEATURE_DIM;
        let x = (0..n)
            .map(|i| {
                let row = &self.rotation[i * n..(i + 1) * n];
                row.iter().zip(&z).map(|(r, v)| r * v).sum::<f64>() as f32
            })
            .collect();
        (label as u8, x)
    }

    fn training_set(&self) -> impl Iterator<Item = (u8, Vec<f32>)> + '_ {
        (0..TRAIN_SAMPLES).map(|i| self.sample(TRAIN_SEQ_BASE + i))
    }
}

// ---------------------------------------------------------------------------
// Encoders
// ---------------------------------------------------------------------------

/// Which encoder produced a datagram (its wire code).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EncoderId {
    /// All 128 raw features in their original order (the `raw` mode control).
    Raw = 0,
    /// The fitted PCA (the `semantic` mode).
    Pca = 1,
    /// The `keep` highest-variance raw dimensions (the fair baseline).
    TopVarianceRaw = 2,
    /// `keep` contiguous mean-pooled bins (the ablation).
    MeanPool = 3,
}

impl EncoderId {
    /// From its wire code.
    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Raw),
            1 => Some(Self::Pca),
            2 => Some(Self::TopVarianceRaw),
            3 => Some(Self::MeanPool),
            _ => None,
        }
    }

    /// The label used in log rows.
    pub fn label(self) -> &'static str {
        match self {
            Self::Raw => "raw",
            Self::Pca => "pca",
            Self::TopVarianceRaw => "top-variance-raw",
            Self::MeanPool => "mean-pool",
        }
    }
}

/// An encoder whose output size is chosen per sample: `keep` features, most
/// important first. Implemented by the fitted PCA today; an ONNX model can
/// replace it once model loading (#440) is wired to this demo.
pub trait RateAdaptiveEncoder: Send + Sync {
    /// Its wire identity.
    fn id(&self) -> EncoderId;
    /// Encode `x` ([`FEATURE_DIM`] values) into `keep` values.
    fn encode(&self, x: &[f32], keep: usize) -> Vec<f32>;
}

/// The raw features, truncated to `keep` (the `raw` mode sends all 128).
#[derive(Debug, Clone, Copy, Default)]
pub struct Raw;

impl RateAdaptiveEncoder for Raw {
    fn id(&self) -> EncoderId {
        EncoderId::Raw
    }
    fn encode(&self, x: &[f32], keep: usize) -> Vec<f32> {
        x[..keep.min(x.len())].to_vec()
    }
}

/// Linear PCA fitted on generator samples.
#[derive(Debug, Clone)]
pub struct PcaEncoder {
    mean: Vec<f64>,
    /// Principal directions, largest eigenvalue first, each [`FEATURE_DIM`] long.
    components: Vec<Vec<f64>>,
    eigenvalues: Vec<f64>,
}

impl PcaEncoder {
    /// Fit on `samples`.
    pub fn fit<'a>(samples: impl Iterator<Item = &'a [f32]>) -> Self {
        let n = FEATURE_DIM;
        let data: Vec<&[f32]> = samples.collect();
        let count = data.len().max(1) as f64;
        let mut mean = vec![0.0; n];
        for x in &data {
            for (m, v) in mean.iter_mut().zip(x.iter()) {
                *m += f64::from(*v) / count;
            }
        }
        let mut cov = vec![0.0; n * n];
        for x in &data {
            let c: Vec<f64> = x
                .iter()
                .zip(&mean)
                .map(|(v, m)| f64::from(*v) - m)
                .collect();
            for i in 0..n {
                for j in i..n {
                    cov[i * n + j] += c[i] * c[j] / count;
                }
            }
        }
        for i in 0..n {
            for j in 0..i {
                cov[i * n + j] = cov[j * n + i];
            }
        }
        let (eigenvalues, vectors) = jacobi_eigen(cov, n);
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|a, b| eigenvalues[*b].total_cmp(&eigenvalues[*a]));
        let components = order
            .iter()
            .map(|&c| (0..n).map(|r| vectors[r * n + c]).collect())
            .collect();
        let eigenvalues = order.iter().map(|&c| eigenvalues[c]).collect();
        Self {
            mean,
            components,
            eigenvalues,
        }
    }

    /// Eigenvalues, largest first.
    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }
}

impl RateAdaptiveEncoder for PcaEncoder {
    fn id(&self) -> EncoderId {
        EncoderId::Pca
    }
    fn encode(&self, x: &[f32], keep: usize) -> Vec<f32> {
        self.components
            .iter()
            .take(keep)
            .map(|c| {
                c.iter()
                    .zip(x.iter().zip(&self.mean))
                    .map(|(w, (v, m))| w * (f64::from(*v) - m))
                    .sum::<f64>() as f32
            })
            .collect()
    }
}

/// The `keep` raw dimensions with the largest training variance.
#[derive(Debug, Clone)]
pub struct TopVarianceRaw {
    order: Vec<usize>,
}

impl TopVarianceRaw {
    /// Fit on `samples`.
    pub fn fit<'a>(samples: impl Iterator<Item = &'a [f32]>) -> Self {
        let n = FEATURE_DIM;
        let (mut sum, mut sq, mut count) = (vec![0.0f64; n], vec![0.0f64; n], 0.0f64);
        for x in samples {
            count += 1.0;
            for i in 0..n {
                let v = f64::from(x[i]);
                sum[i] += v;
                sq[i] += v * v;
            }
        }
        let var: Vec<f64> = (0..n)
            .map(|i| sq[i] / count - (sum[i] / count).powi(2))
            .collect();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|a, b| var[*b].total_cmp(&var[*a]));
        Self { order }
    }
}

impl RateAdaptiveEncoder for TopVarianceRaw {
    fn id(&self) -> EncoderId {
        EncoderId::TopVarianceRaw
    }
    fn encode(&self, x: &[f32], keep: usize) -> Vec<f32> {
        self.order.iter().take(keep).map(|&i| x[i]).collect()
    }
}

/// `keep` contiguous bins of the raw features, each averaged.
#[derive(Debug, Clone, Copy, Default)]
pub struct MeanPool;

impl RateAdaptiveEncoder for MeanPool {
    fn id(&self) -> EncoderId {
        EncoderId::MeanPool
    }
    fn encode(&self, x: &[f32], keep: usize) -> Vec<f32> {
        let n = x.len();
        let keep = keep.clamp(1, n);
        (0..keep)
            .map(|b| {
                let (lo, hi) = (b * n / keep, (b + 1) * n / keep);
                x[lo..hi].iter().sum::<f32>() / (hi - lo) as f32
            })
            .collect()
    }
}

/// Cyclic Jacobi eigen-decomposition of a symmetric `n x n` row-major matrix.
/// Returns the eigenvalues and the eigenvectors as the columns of a row-major
/// matrix.
fn jacobi_eigen(mut a: Vec<f64>, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let total: f64 = a.iter().map(|x| x * x).sum::<f64>().max(f64::MIN_POSITIVE);
    for _sweep in 0..64 {
        let off: f64 = (0..n)
            .flat_map(|i| (0..n).filter(move |j| *j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i * n + j] * a[i * n + j])
            .sum();
        if off <= 1e-22 * total {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p * n + k];
                    let aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    ((0..n).map(|i| a[i * n + i]).collect(), v)
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

/// Nearest-centroid classifier over encoded features.
#[derive(Debug, Clone)]
pub struct NearestCentroid {
    centroids: Vec<Vec<f32>>,
}

impl NearestCentroid {
    /// Fit on `(label, encoded)` pairs.
    pub fn fit(samples: impl Iterator<Item = (u8, Vec<f32>)>) -> Self {
        let mut sums: Vec<Vec<f64>> = vec![Vec::new(); CLASSES];
        let mut counts = vec![0usize; CLASSES];
        for (label, enc) in samples {
            let l = usize::from(label) % CLASSES;
            if sums[l].is_empty() {
                sums[l] = vec![0.0; enc.len()];
            }
            for (s, v) in sums[l].iter_mut().zip(&enc) {
                *s += f64::from(*v);
            }
            counts[l] += 1;
        }
        let centroids = sums
            .into_iter()
            .zip(counts)
            .map(|(s, c)| {
                s.into_iter()
                    .map(|v| (v / c.max(1) as f64) as f32)
                    .collect()
            })
            .collect();
        Self { centroids }
    }

    /// The predicted class of `enc`.
    pub fn predict(&self, enc: &[f32]) -> u8 {
        let mut best = (0usize, f32::INFINITY);
        for (c, centroid) in self.centroids.iter().enumerate() {
            if centroid.len() != enc.len() {
                continue;
            }
            let d: f32 = centroid
                .iter()
                .zip(enc)
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            if d < best.1 {
                best = (c, d);
            }
        }
        best.0 as u8
    }
}

/// The demo's model: the generator, the fitted encoders and a cache of
/// nearest-centroid classifiers per (encoder, keep). Built identically on the
/// UE and on the sink from the same seed.
pub struct D444aModel {
    generator: FeatureGenerator,
    pca: PcaEncoder,
    top_variance: TopVarianceRaw,
    classifiers: HashMap<(EncoderId, usize), NearestCentroid>,
}

impl fmt::Debug for D444aModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("D444aModel")
            .field("seed", &self.generator.seed())
            .finish_non_exhaustive()
    }
}

impl D444aModel {
    /// Build (and fit) the model for `seed`.
    pub fn new(seed: u64) -> Self {
        let generator = FeatureGenerator::new(seed);
        let train: Vec<Vec<f32>> = generator.training_set().map(|(_, x)| x).collect();
        let pca = PcaEncoder::fit(train.iter().map(Vec::as_slice));
        let top_variance = TopVarianceRaw::fit(train.iter().map(Vec::as_slice));
        Self {
            generator,
            pca,
            top_variance,
            classifiers: HashMap::new(),
        }
    }

    /// The generator.
    pub fn generator(&self) -> &FeatureGenerator {
        &self.generator
    }

    /// The encoder for `id`.
    pub fn encoder(&self, id: EncoderId) -> &dyn RateAdaptiveEncoder {
        match id {
            EncoderId::Raw => &Raw,
            EncoderId::Pca => &self.pca,
            EncoderId::TopVarianceRaw => &self.top_variance,
            EncoderId::MeanPool => &MeanPool,
        }
    }

    /// The fitted PCA.
    pub fn pca(&self) -> &PcaEncoder {
        &self.pca
    }

    /// Classify features `enc` produced by `id` at `keep = enc.len()`.
    pub fn classify(&mut self, id: EncoderId, enc: &[f32]) -> u8 {
        let keep = enc.len();
        if !self.classifiers.contains_key(&(id, keep)) {
            let encoder = self.encoder(id);
            let fitted = NearestCentroid::fit(
                self.generator
                    .training_set()
                    .map(|(l, x)| (l, encoder.encode(&x, keep))),
            );
            self.classifiers.insert((id, keep), fitted);
        }
        self.classifiers[&(id, keep)].predict(enc)
    }

    /// Accuracy of `id` at `keep` over samples `seqs`, through the wire's
    /// quantisation: what the sink would score.
    pub fn accuracy(&mut self, id: EncoderId, keep: usize, seqs: std::ops::Range<u64>) -> f64 {
        let total = seqs.end.saturating_sub(seqs.start).max(1) as f64;
        let mut correct = 0usize;
        for seq in seqs {
            let (label, x) = self.generator.sample(seq);
            let enc = quantise_roundtrip(&self.encoder(id).encode(&x, keep));
            if self.classify(id, &enc) == label {
                correct += 1;
            }
        }
        correct as f64 / total
    }
}

// ---------------------------------------------------------------------------
// Link budget
// ---------------------------------------------------------------------------

/// IP bytes of one datagram carrying `keep` features.
pub fn datagram_ip_bytes(keep: usize) -> usize {
    IP_UDP_OVERHEAD + HEADER_LEN + keep * BYTES_PER_FEATURE
}

/// The application's planned uplink rate for a Session-AMBR: [`HEADROOM`]
/// of it.
pub fn rate_target_bps(ul_ambr_bps: u64) -> u64 {
    (ul_ambr_bps as f64 * HEADROOM) as u64
}

/// How many features each datagram carries so that `rate_hz` datagrams per
/// second stay within [`rate_target_bps`] of the uplink Session-AMBR.
/// A multiple of [`KEEP_STEP`], within `MIN_KEEP..=FEATURE_DIM`.
///
/// 64 kbps at 50 Hz: 51 200 bit/s → 128 B per datagram → 42 features → 40.
pub fn keep_for_link_budget(ul_ambr_bps: u64, rate_hz: u32) -> usize {
    let per_datagram = rate_target_bps(ul_ambr_bps) / 8 / u64::from(rate_hz.max(1));
    let payload = (per_datagram as usize).saturating_sub(IP_UDP_OVERHEAD + HEADER_LEN);
    let keep = (payload / BYTES_PER_FEATURE).min(FEATURE_DIM);
    (keep - keep % KEEP_STEP).max(MIN_KEEP)
}

// ---------------------------------------------------------------------------
// Wire
// ---------------------------------------------------------------------------

/// Magic of a [`FeatureDatagram`].
pub const MAGIC: [u8; 2] = *b"D4";
/// Wire version.
pub const WIRE_VERSION: u8 = 1;

/// One UL feature datagram (non-normative application payload):
///
/// ```text
/// 0-1   magic "D4"        2  version (1)       3  encoder (EncoderId)
/// 4-7   seq (u32, BE)     8  label (ground truth, for scoring only)
/// 9     keep              10-11 reserved (0)  12-15 scale (f32, BE)
/// 16-   keep x i16 (BE), value = i16 * scale
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureDatagram {
    /// Which encoder produced the values.
    pub encoder: EncoderId,
    /// Sample sequence number (the generator's sample index).
    pub seq: u32,
    /// Ground-truth class.
    pub label: u8,
    /// The values (dequantised on decode).
    pub values: Vec<f32>,
}

/// A datagram that does not decode.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WireError {
    /// Shorter than its header or its declared `keep`.
    #[error("datagram truncated: {0} bytes")]
    Truncated(usize),
    /// Not a D444A datagram.
    #[error("bad magic or version")]
    BadMagic,
    /// Unknown encoder code.
    #[error("unknown encoder {0}")]
    UnknownEncoder(u8),
}

fn quantise(values: &[f32]) -> (f32, Vec<i16>) {
    let max = values.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let scale = if max > 0.0 {
        max / f32::from(i16::MAX)
    } else {
        1.0
    };
    let q = values
        .iter()
        .map(|v| (v / scale).round().clamp(-32767.0, 32767.0) as i16)
        .collect();
    (scale, q)
}

/// `values` as the receiver sees them after the wire's quantisation.
pub fn quantise_roundtrip(values: &[f32]) -> Vec<f32> {
    let (scale, q) = quantise(values);
    q.into_iter().map(|v| f32::from(v) * scale).collect()
}

impl FeatureDatagram {
    /// Encode to bytes ([`HEADER_LEN`] + 2 × keep).
    pub fn encode(&self) -> Vec<u8> {
        let (scale, q) = quantise(&self.values);
        let keep = u8::try_from(q.len()).unwrap_or(u8::MAX);
        let mut out = Vec::with_capacity(HEADER_LEN + q.len() * BYTES_PER_FEATURE);
        out.extend_from_slice(&MAGIC);
        out.push(WIRE_VERSION);
        out.push(self.encoder as u8);
        out.extend_from_slice(&self.seq.to_be_bytes());
        out.push(self.label);
        out.push(keep);
        out.extend_from_slice(&[0, 0]);
        out.extend_from_slice(&scale.to_be_bytes());
        for v in q.iter().take(usize::from(keep)) {
            out.extend_from_slice(&v.to_be_bytes());
        }
        out
    }

    /// Decode from bytes.
    pub fn decode(buf: &[u8]) -> Result<Self, WireError> {
        if buf.len() < HEADER_LEN {
            return Err(WireError::Truncated(buf.len()));
        }
        if buf[0..2] != MAGIC || buf[2] != WIRE_VERSION {
            return Err(WireError::BadMagic);
        }
        let encoder = EncoderId::from_code(buf[3]).ok_or(WireError::UnknownEncoder(buf[3]))?;
        let seq = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]);
        let label = buf[8];
        let keep = usize::from(buf[9]);
        let scale = f32::from_be_bytes([buf[12], buf[13], buf[14], buf[15]]);
        if buf.len() < HEADER_LEN + keep * BYTES_PER_FEATURE {
            return Err(WireError::Truncated(buf.len()));
        }
        let values = buf[HEADER_LEN..HEADER_LEN + keep * BYTES_PER_FEATURE]
            .chunks_exact(2)
            .map(|c| f32::from(i16::from_be_bytes([c[0], c[1]])) * scale)
            .collect();
        Ok(Self {
            encoder,
            seq,
            label,
            values,
        })
    }
}

// ---------------------------------------------------------------------------
// Per-epoch scoring at the sink
// ---------------------------------------------------------------------------

/// One epoch's row at the sink.
#[derive(Debug, Clone, PartialEq)]
pub struct EpochRow {
    /// Epoch number (`seq / epoch_datagrams`).
    pub epoch: u64,
    /// The demo mode the sink labels its rows with.
    pub mode: String,
    /// Encoder of the received datagrams (the last seen when it changed).
    pub encoder: EncoderId,
    /// Smallest and largest `keep` received.
    pub keep_min: usize,
    /// See `keep_min`.
    pub keep_max: usize,
    /// Datagrams received.
    pub received: u64,
    /// Datagrams of the epoch's sequence span that never arrived.
    pub lost: u64,
    /// IP bytes received ([`IP_UDP_OVERHEAD`] + datagram).
    pub ip_bytes: u64,
    /// Received IP rate over the epoch's nominal duration, kbit/s.
    pub ul_kbps: f64,
    /// Accuracy of what was received.
    pub accuracy: f64,
    /// Accuracy of the top-variance raw baseline on the same samples at the
    /// same keep (same bytes); `None` unless the epoch carried PCA datagrams.
    pub top_variance_raw_accuracy: Option<f64>,
    /// Same for the mean-pool ablation.
    pub mean_pool_accuracy: Option<f64>,
}

impl fmt::Display for EpochRow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let keep = if self.keep_min == self.keep_max {
            self.keep_max.to_string()
        } else {
            format!("{}..{}", self.keep_max, self.keep_min)
        };
        write!(
            f,
            "epoch={} mode={} encoder={} keep={} received={} lost={} ip_bytes={} ul_kbps={:.1} acc={:.3}",
            self.epoch,
            self.mode,
            self.encoder.label(),
            keep,
            self.received,
            self.lost,
            self.ip_bytes,
            self.ul_kbps,
            self.accuracy
        )?;
        if let (Some(t), Some(m)) = (self.top_variance_raw_accuracy, self.mean_pool_accuracy) {
            write!(
                f,
                " baseline top-variance-raw acc={t:.3} mean-pool acc={m:.3} (equal bytes) margin={:+.3}/{:+.3}",
                self.accuracy - t,
                self.accuracy - m
            )?;
        }
        Ok(())
    }
}

#[derive(Debug)]
struct EpochAcc {
    epoch: u64,
    first_seq: u64,
    encoder: EncoderId,
    keep_min: usize,
    keep_max: usize,
    received: u64,
    ip_bytes: u64,
    correct: u64,
    pca_samples: u64,
    top_variance_correct: u64,
    mean_pool_correct: u64,
}

/// Scores datagrams into per-epoch rows.
#[derive(Debug)]
pub struct EpochScorer {
    model: D444aModel,
    epoch_datagrams: u64,
    rate_hz: u32,
    first_seq_seen: Option<u64>,
    current: Option<EpochAcc>,
}

impl EpochScorer {
    /// A scorer over `model`, `epoch_datagrams` per epoch at `rate_hz`.
    pub fn new(model: D444aModel, epoch_datagrams: u64, rate_hz: u32) -> Self {
        Self {
            model,
            epoch_datagrams: epoch_datagrams.max(1),
            rate_hz: rate_hz.max(1),
            first_seq_seen: None,
            current: None,
        }
    }

    /// Score one datagram, labelled with `mode`. Returns the row of the epoch
    /// it closed, if it is the first of a later epoch.
    pub fn on_datagram(
        &mut self,
        d: &FeatureDatagram,
        wire_len: usize,
        mode: &str,
    ) -> Option<EpochRow> {
        let seq = u64::from(d.seq);
        let epoch = seq / self.epoch_datagrams;
        let mut closed = None;
        if self
            .current
            .as_ref()
            .is_some_and(|acc| epoch + 1 < acc.epoch)
        {
            // The sequence went back by more than one epoch: the sender
            // restarted. Close what was open and start over.
            closed = self.close(mode);
            self.first_seq_seen = None;
        }
        let first = *self.first_seq_seen.get_or_insert(seq);
        match &self.current {
            Some(acc) if acc.epoch == epoch => {}
            Some(acc) if epoch < acc.epoch => return closed, // late, out of order
            _ => {
                closed = closed.or(self.close(mode));
                self.current = Some(EpochAcc {
                    epoch,
                    // A full epoch's span starts at its boundary; the first,
                    // partial one at the first datagram seen.
                    first_seq: (epoch * self.epoch_datagrams).max(first),
                    encoder: d.encoder,
                    keep_min: usize::MAX,
                    keep_max: 0,
                    received: 0,
                    ip_bytes: 0,
                    correct: 0,
                    pca_samples: 0,
                    top_variance_correct: 0,
                    mean_pool_correct: 0,
                });
            }
        }
        let keep = d.values.len();
        let predicted = self.model.classify(d.encoder, &d.values);
        let baselines = (d.encoder == EncoderId::Pca).then(|| {
            let (_, x) = self.model.generator().sample(seq);
            let tv = quantise_roundtrip(
                &self
                    .model
                    .encoder(EncoderId::TopVarianceRaw)
                    .encode(&x, keep),
            );
            let mp = quantise_roundtrip(&MeanPool.encode(&x, keep));
            (
                self.model.classify(EncoderId::TopVarianceRaw, &tv) == d.label,
                self.model.classify(EncoderId::MeanPool, &mp) == d.label,
            )
        });
        let acc = self.current.as_mut()?;
        acc.encoder = d.encoder;
        acc.keep_min = acc.keep_min.min(keep);
        acc.keep_max = acc.keep_max.max(keep);
        acc.received += 1;
        acc.ip_bytes += (wire_len + IP_UDP_OVERHEAD) as u64;
        acc.correct += u64::from(predicted == d.label);
        if let Some((tv, mp)) = baselines {
            acc.pca_samples += 1;
            acc.top_variance_correct += u64::from(tv);
            acc.mean_pool_correct += u64::from(mp);
        }
        closed
    }

    /// Close the current epoch (at shutdown, or on a later epoch's first datagram).
    pub fn close(&mut self, mode: &str) -> Option<EpochRow> {
        let acc = self.current.take()?;
        let end = (acc.epoch + 1) * self.epoch_datagrams;
        let span = end.saturating_sub(acc.first_seq);
        let received = acc.received.max(1) as f64;
        let seconds = self.epoch_datagrams as f64 / f64::from(self.rate_hz);
        let pca = (acc.pca_samples > 0).then_some(acc.pca_samples as f64);
        Some(EpochRow {
            epoch: acc.epoch,
            mode: mode.to_string(),
            encoder: acc.encoder,
            keep_min: acc.keep_min,
            keep_max: acc.keep_max,
            received: acc.received,
            lost: span.saturating_sub(acc.received),
            ip_bytes: acc.ip_bytes,
            ul_kbps: acc.ip_bytes as f64 * 8.0 / seconds / 1000.0,
            accuracy: acc.correct as f64 / received,
            top_variance_raw_accuracy: pca.map(|n| acc.top_variance_correct as f64 / n),
            mean_pool_accuracy: pca.map(|n| acc.mean_pool_correct as f64 / n),
        })
    }
}

#[cfg(test)]
mod tests;
