//! Demo D444C (nextgsim#479): task-oriented classification vs raw at equal
//! bytes under an **emulated** AWGN sweep. Crate-level and **non-normative**.
//!
//! One seeded table of top-1, macro and worst-class accuracy per
//! (arm, keep, SNR), printed by `examples/task_vs_raw.rs`. What it is, and what
//! it is not:
//! - **Data:** D444A's pinned synthetic generator
//!   ([`FeatureGenerator`](crate::rate_adapt::FeatureGenerator), seed
//!   [`GENERATOR_SEED`]). The accuracy gap between arms is set by its pinned
//!   spectrum: ground truth of the generator, not a learned result.
//! - **Arms**, all at the same `keep`, hence the same bytes:
//!   - `task-encoder`: the fitted linear PCA of #477 (unsupervised, no labels,
//!     no neural network), keeping its `keep` largest-variance components. On
//!     this generator the within-class noise is isotropic, so those are also the
//!     discriminative directions;
//!   - `top-variance`: the `keep` raw dimensions with the largest training
//!     variance (the fair baseline);
//!   - `mean-pool`: `keep` contiguous mean-pooled bins (an ablation).
//! - **Channel:** [`AwgnChannel`] added in-process to the dequantised i16
//!   symbols, SNR per transmitted vector. It is an analog-channel abstraction
//!   with no coding and no network: a real PDU session would deliver the bytes
//!   bit-exact or drop them.
//! - **Classifier:** [`NearestCentroid`] fitted on the noiseless training
//!   encodings of each (arm, keep). Nothing is trained by gradient.
//!
//! The noise seed only drives the channel. Each sample's noise is drawn from
//! `ChaCha8Rng` (value-stable across `rand_chacha` versions) on stream
//! `keep << 32 | seq`, so it depends on (seed, keep, sample) only: every arm
//! and every SNR sees the same unit draws (common random numbers), and a
//! single-arm mode's rows equal the matching rows of `all`.

use std::fmt::Write as _;

use nextgsim_demo::{DemoConfigError, DemoRegistry, DemoRuntime, DemoSpec};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::metrics::{ClassificationScore, NearestCentroid};
use crate::rate_adapt::{
    quantise_roundtrip, D444aModel, EncoderId, BYTES_PER_FEATURE, CLASSES, DEFAULT_SEED,
    FEATURE_DIM,
};
use crate::AwgnChannel;

/// The demo key (nextgsim#458 toggle convention).
pub const KEY: &str = "d444c-task-vs-raw";
/// Its modes: `all` prints every arm and the deltas; the others print one arm.
pub const MODES: &[&str] = &["all", "task-encoder", "top-variance", "mean-pool"];
/// The generator seed: pinned, D444A's task.
pub const GENERATOR_SEED: u64 = DEFAULT_SEED;
/// The default noise seed of the example.
pub const DEFAULT_NOISE_SEED: u64 = 444;
/// Values kept per sample.
pub const KEEPS: &[usize] = &[8, 16, 32, 64];
/// The SNR sweep, dB.
pub const SNRS_DB: &[i32] = &[0, 5, 10, 15, 20];
/// Held-out samples per point (`seq 0..HELD_OUT`; training is at `seq >= 2^40`).
pub const HELD_OUT: u64 = 2000;

/// One arm of the comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arm {
    /// The fitted PCA.
    TaskEncoder,
    /// The top-variance raw dimensions.
    TopVariance,
    /// Mean-pooled bins.
    MeanPool,
}

impl Arm {
    /// Every arm, in table order.
    pub const ALL: [Arm; 3] = [Arm::TaskEncoder, Arm::TopVariance, Arm::MeanPool];

    /// Its name in the table (and its mode).
    pub fn name(self) -> &'static str {
        match self {
            Arm::TaskEncoder => "task-encoder",
            Arm::TopVariance => "top-variance",
            Arm::MeanPool => "mean-pool",
        }
    }

    fn encoder(self) -> EncoderId {
        match self {
            Arm::TaskEncoder => EncoderId::Pca,
            Arm::TopVariance => EncoderId::TopVarianceRaw,
            Arm::MeanPool => EncoderId::MeanPool,
        }
    }
}

/// What a mode prints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Every arm, then the deltas.
    All,
    /// One arm's rows.
    Arm(Arm),
}

impl Mode {
    /// From a registered mode name.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "all" => Some(Mode::All),
            other => Arm::ALL
                .into_iter()
                .find(|a| a.name() == other)
                .map(Mode::Arm),
        }
    }

    /// The arms it runs.
    pub fn arms(self) -> Vec<Arm> {
        match self {
            Mode::All => Arm::ALL.to_vec(),
            Mode::Arm(a) => vec![a],
        }
    }

    /// Its name.
    pub fn name(self) -> &'static str {
        match self {
            Mode::All => "all",
            Mode::Arm(a) => a.name(),
        }
    }
}

/// One table row.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    /// The arm.
    pub arm: Arm,
    /// Values kept.
    pub keep: usize,
    /// Bytes per sample on the wire, measured from the encoded vector.
    pub bytes: usize,
    /// The channel SNR, dB.
    pub snr_db: i32,
    /// The score over the held-out samples.
    pub score: ClassificationScore,
}

impl Row {
    /// The row as printed.
    pub fn render(&self) -> String {
        format!(
            "{:<12} {:>4} {:>5} {:>6} {:>6.4} {:>6.4} {:>6.4}",
            self.arm.name(),
            self.keep,
            self.bytes,
            self.snr_db,
            self.score.top1(),
            self.score.macro_accuracy(),
            self.score.worst_class_accuracy()
        )
    }
}

/// The sweep: the pinned model and the noise seed.
pub struct Sweep {
    model: D444aModel,
    noise_seed: u64,
}

impl std::fmt::Debug for Sweep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sweep")
            .field("noise_seed", &self.noise_seed)
            .finish_non_exhaustive()
    }
}

/// The RNG of sample `seq`'s noise at `keep`.
fn noise_rng(seed: u64, keep: usize, seq: u64) -> ChaCha8Rng {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(((keep as u64) << 32) | seq);
    rng
}

/// Score `samples`, each `(label, sent, received)`, with `classifier` on what
/// was received.
pub fn score(
    classifier: &NearestCentroid,
    samples: impl Iterator<Item = (u8, Vec<f32>, Vec<f32>)>,
) -> ClassificationScore {
    let mut score = ClassificationScore::new();
    // `_sent` is unused here on purpose: it is the hook of the 479-top-k-scorer
    // revert arm (scripts/demo/reverts/), which scores sent vs received with
    // `top_k_accuracy`. Removing it breaks that patch (ARM BROKEN).
    for (label, _sent, received) in samples {
        score.record(label, classifier.predict(&received));
    }
    score
}

impl Sweep {
    /// The sweep over the pinned generator, with channel noise from
    /// `noise_seed`.
    pub fn new(noise_seed: u64) -> Self {
        Self {
            model: D444aModel::new(GENERATOR_SEED),
            noise_seed,
        }
    }

    /// The nearest-centroid classifier of `arm` at `keep`, fitted on the
    /// noiseless training encodings.
    pub fn classifier(&self, arm: Arm, keep: usize) -> NearestCentroid {
        let encoder = self.model.encoder(arm.encoder());
        NearestCentroid::fit(
            self.model
                .generator()
                .training_set()
                .map(|(l, x)| (l, encoder.encode(&x, keep))),
        )
    }

    /// The held-out samples of `arm` at `keep` as sent: `(label, symbols)`,
    /// after the wire's i16 quantisation.
    pub fn held_out(&self, arm: Arm, keep: usize) -> Vec<(u8, Vec<f32>)> {
        let encoder = self.model.encoder(arm.encoder());
        (0..HELD_OUT)
            .map(|seq| {
                let (label, x) = self.model.generator().sample(seq);
                (label, quantise_roundtrip(&encoder.encode(&x, keep)))
            })
            .collect()
    }

    /// The rows of `arms`, in table order (arm, keep, SNR).
    pub fn run(&self, arms: &[Arm]) -> Vec<Row> {
        let mut rows = Vec::new();
        for &arm in arms {
            for &keep in KEEPS {
                let classifier = self.classifier(arm, keep);
                let sent = self.held_out(arm, keep);
                let bytes = sent[0].1.len() * BYTES_PER_FEATURE;
                assert!(
                    sent.iter()
                        .all(|(_, s)| s.len() * BYTES_PER_FEATURE == bytes),
                    "{} at keep {keep} sent a varying number of values",
                    arm.name()
                );
                for &snr_db in SNRS_DB {
                    let channel = AwgnChannel::new(snr_db as f32);
                    let samples = sent.iter().zip(0u64..).map(|((label, s), seq)| {
                        let mut rng = noise_rng(self.noise_seed, keep, seq);
                        (*label, s.clone(), channel.transmit_with_rng(s, &mut rng))
                    });
                    rows.push(Row {
                        arm,
                        keep,
                        bytes,
                        snr_db,
                        score: score(&classifier, samples),
                    });
                }
            }
        }
        rows
    }
}

/// The table header.
pub const HEADER: &str = "arm          keep bytes snr_db   top1  macro  worst";

/// The table for `mode`: three `#` lines naming what it is, the header, the
/// rows and, for `all`, the per-point deltas of the task encoder over each
/// baseline.
pub fn render(mode: Mode, noise_seed: u64, rows: &[Row]) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "# {KEY} mode={}: task encoder vs raw at equal bytes, emulated AWGN, non-normative",
        mode.name()
    );
    let _ = writeln!(
        out,
        "# data: synthetic, pinned generator seed {GENERATOR_SEED} ({CLASSES} classes, \
         {FEATURE_DIM} features); held-out {HELD_OUT} samples; noise seed {noise_seed}"
    );
    let _ = writeln!(
        out,
        "# channel: emulated AWGN on the dequantised i16 symbols, SNR per vector; \
         classifier: nearest centroid fitted noiseless"
    );
    let _ = writeln!(out, "{HEADER}");
    for row in rows {
        let _ = writeln!(out, "{}", row.render());
    }
    if mode == Mode::All {
        let _ = writeln!(
            out,
            "delta        keep bytes snr_db task-minus-top-variance task-minus-mean-pool"
        );
        let find = |arm: Arm, keep: usize, snr: i32| {
            rows.iter()
                .find(|r| r.arm == arm && r.keep == keep && r.snr_db == snr)
        };
        for &keep in KEEPS {
            for &snr in SNRS_DB {
                let (Some(t), Some(v), Some(p)) = (
                    find(Arm::TaskEncoder, keep, snr),
                    find(Arm::TopVariance, keep, snr),
                    find(Arm::MeanPool, keep, snr),
                ) else {
                    continue;
                };
                let _ = writeln!(
                    out,
                    "{:<12} {:>4} {:>5} {:>6} {:>+23.4} {:>+20.4}",
                    "delta",
                    keep,
                    t.bytes,
                    snr,
                    t.score.top1() - v.score.top1(),
                    t.score.top1() - p.score.top1()
                );
            }
        }
    }
    out
}

/// Why the example's config was refused. Every variant exits non-zero.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// Not YAML.
    #[error("config is not valid YAML: {0}")]
    Yaml(String),
    /// A root key other than `demo`.
    #[error("unknown config key {0:?} (task_vs_raw reads only the demo: block)")]
    UnknownField(String),
    /// The `demo:` block was refused.
    #[error(transparent)]
    Demo(#[from] DemoConfigError),
}

/// The example's demo registry: this demo only.
pub fn registry() -> DemoRegistry {
    DemoRegistry::new(&[DemoSpec {
        key: KEY,
        modes: MODES,
        requires: &[],
    }])
}

/// Read the example's config. `Ok((runtime, None))`: the demo is not active
/// (no `demo:` block, the kill switch off, or the key off), so the example
/// prints no table and no banner. A root that is not a mapping (a bare scalar
/// or list) is also "no demo", as `nextgsim_demo::DemoConfig::from_root` documents.
pub fn prepare(yaml: &str) -> Result<(DemoRuntime, Option<Mode>), ConfigError> {
    let root: serde_yaml::Value =
        serde_yaml::from_str(yaml).map_err(|e| ConfigError::Yaml(e.to_string()))?;
    if let Some(map) = root.as_mapping() {
        for k in map.keys() {
            if k.as_str() != Some(nextgsim_demo::DEMO_BLOCK) {
                return Err(ConfigError::UnknownField(format!("{k:?}")));
            }
        }
    }
    let runtime = DemoRuntime::from_root(&root, registry())?;
    let mode = runtime
        .mode(KEY)
        .map(|m| Mode::from_name(&m).expect("the registry admits only MODES"));
    Ok((runtime, mode))
}

#[cfg(test)]
mod tests;
