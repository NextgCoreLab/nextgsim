//! The pinned synthetic non-IID regression task (nextgsim#480).
//!
//! The data is declared synthetic: a fixed random "teacher" maps 16 inputs to one
//! target,
//!
//! `y = Σ_m a_m · tanh(u_m · x / 4) + 0.05 · ε`, with m = 1..4,
//!
//! and every client sees the SAME teacher on a DIFFERENT region of the input
//! space: client `c` draws `x ~ N(μ_c, σ² I)`, with `μ_c` a client-specific shift.
//! That is covariate-shift non-IID data: a model trained on one client's region
//! extrapolates poorly to the others, so federating helps measurably. The held-out
//! set mixes every client's region equally, and is what the AF scores on.
//!
//! The first [`CANARIES`] rows of every client are canaries. They are ordinary
//! training rows, but their exact bytes are known to anyone with the run seed, so
//! a scan of what crossed the network can prove no training row did
//! ([`contains_row`]).
//!
//! Everything is a pure function of `(seed, clients, samples)`.

use crate::mlp::IN;
use crate::seeded::{self, normal, Purpose};

/// Hidden units of the teacher.
const TEACHER_UNITS: usize = 4;
/// Distance of each client's region from the origin.
const SHIFT: f32 = 2.5;
/// Spread of each client's inputs around its region centre.
const SIGMA: f32 = 0.6;
/// Label noise.
const NOISE: f32 = 0.05;
/// Canary rows per client.
pub const CANARIES: usize = 4;

/// One dataset: `ys.len()` rows of `IN` inputs in `xs`, row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct Dataset {
    pub xs: Vec<f32>,
    pub ys: Vec<f32>,
}

impl Dataset {
    pub fn len(&self) -> usize {
        self.ys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ys.is_empty()
    }

    /// Row `n`'s inputs.
    pub fn row(&self, n: usize) -> &[f32] {
        &self.xs[n * IN..(n + 1) * IN]
    }

    /// This dataset's canary rows (the first [`CANARIES`]).
    pub fn canaries(&self) -> impl Iterator<Item = &[f32]> {
        (0..CANARIES.min(self.len())).map(|n| self.row(n))
    }

    /// Several datasets concatenated (the pooled baseline's data).
    pub fn concat<'a>(parts: impl IntoIterator<Item = &'a Dataset>) -> Self {
        let mut out = Dataset {
            xs: Vec::new(),
            ys: Vec::new(),
        };
        for p in parts {
            out.xs.extend_from_slice(&p.xs);
            out.ys.extend_from_slice(&p.ys);
        }
        out
    }
}

/// The teacher and the per-client regions of one run.
#[derive(Debug, Clone)]
pub struct Task {
    seed: u64,
    u: Vec<[f32; IN]>,
    a: Vec<f32>,
    centres: Vec<[f32; IN]>,
}

impl Task {
    /// The task of the run seeded by `seed`, with `clients` regions.
    pub fn new(seed: u64, clients: u32) -> Self {
        let mut rng = seeded::rng(seed, Purpose::Teacher);
        let u = (0..TEACHER_UNITS)
            .map(|_| std::array::from_fn(|_| normal(&mut rng)))
            .collect();
        let a = (0..TEACHER_UNITS).map(|_| normal(&mut rng)).collect();
        // Each region centre is a random unit direction scaled to SHIFT.
        let centres = (0..clients)
            .map(|_| {
                let d: [f32; IN] = std::array::from_fn(|_| normal(&mut rng));
                let norm = d.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
                d.map(|v| SHIFT * v / norm)
            })
            .collect();
        Self {
            seed,
            u,
            a,
            centres,
        }
    }

    pub fn clients(&self) -> u32 {
        self.centres.len() as u32
    }

    fn teacher(&self, x: &[f32]) -> f32 {
        self.u
            .iter()
            .zip(&self.a)
            .map(|(u, a)| a * (u.iter().zip(x).map(|(w, v)| w * v).sum::<f32>() / 4.0).tanh())
            .sum()
    }

    fn draw(&self, rng: &mut rand_chacha::ChaCha8Rng, centre: &[f32; IN], n: usize) -> Dataset {
        let mut xs = Vec::with_capacity(n * IN);
        let mut ys = Vec::with_capacity(n);
        for _ in 0..n {
            let row: [f32; IN] = std::array::from_fn(|i| centre[i] + SIGMA * normal(rng));
            ys.push(self.teacher(&row) + NOISE * normal(rng));
            xs.extend_from_slice(&row);
        }
        Dataset { xs, ys }
    }

    /// Client `c`'s training data: `n` rows from its own region.
    pub fn client_data(&self, c: u32, n: usize) -> Dataset {
        let mut rng = seeded::rng(self.seed, Purpose::ClientData(c));
        self.draw(&mut rng, &self.centres[c as usize], n)
    }

    /// The held-out set: `n_per_client` rows from every client's region.
    pub fn held_out(&self, n_per_client: usize) -> Dataset {
        let mut rng = seeded::rng(self.seed, Purpose::HeldOut);
        let parts: Vec<Dataset> = self
            .centres
            .iter()
            .map(|c| self.draw(&mut rng, c, n_per_client))
            .collect();
        Dataset::concat(&parts)
    }
}

/// Whether `bytes` contains `row` encoded as little-endian `f32`s, the encoding
/// every frame uses ([`crate::frame`]). The canary scan: any training row that
/// crossed the wire inside any frame field shows up here.
pub fn contains_row(bytes: &[u8], row: &[f32]) -> bool {
    let needle: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
    !needle.is_empty() && bytes.windows(needle.len()).any(|w| w == needle.as_slice())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_is_a_pure_function_of_the_seed() {
        let a = Task::new(480, 3);
        let b = Task::new(480, 3);
        assert_eq!(a.client_data(1, 50), b.client_data(1, 50));
        assert_eq!(a.held_out(20), b.held_out(20));
        assert_ne!(a.client_data(1, 50), Task::new(481, 3).client_data(1, 50));
        // Adding a client does not change the existing clients' data.
        assert_eq!(a.client_data(0, 50), Task::new(480, 5).client_data(0, 50));
    }

    /// Non-IID: each client's inputs sit around its own centre, far from the
    /// others'.
    #[test]
    fn clients_see_different_regions() {
        let t = Task::new(480, 3);
        let mean = |d: &Dataset| -> Vec<f32> {
            (0..IN)
                .map(|i| (0..d.len()).map(|n| d.row(n)[i]).sum::<f32>() / d.len() as f32)
                .collect()
        };
        let m: Vec<Vec<f32>> = (0..3).map(|c| mean(&t.client_data(c, 400))).collect();
        let dist = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f32>()
                .sqrt()
        };
        for i in 0..3 {
            for j in i + 1..3 {
                assert!(dist(&m[i], &m[j]) > 1.0, "clients {i} and {j} overlap");
            }
        }
        assert_eq!(
            t.held_out(10).len(),
            30,
            "the held-out set covers every region"
        );
    }

    #[test]
    fn the_canary_scan_finds_a_row_and_nothing_else() {
        let d = Task::new(480, 1).client_data(0, 20);
        let canary = d.canaries().next().expect("a canary").to_vec();
        let mut wire: Vec<u8> = vec![0xAA; 13];
        wire.extend(canary.iter().flat_map(|v| v.to_le_bytes()));
        wire.extend([0x55; 7]);
        assert!(contains_row(&wire, &canary), "a row on the wire is found");
        let other = d.row(CANARIES).to_vec();
        assert!(!contains_row(&wire, &other));
        assert_eq!(d.canaries().count(), CANARIES);
    }
}
