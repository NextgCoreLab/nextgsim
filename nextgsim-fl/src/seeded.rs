//! Seeded, reproducible randomness for FL runs (nextgsim#480).
//!
//! Every random draw in a demo run (data generation, model initialisation, the
//! shuffle of each local epoch) comes from a [`ChaCha8Rng`] derived from ONE run
//! seed plus a named purpose. Rerunning with the same seed reproduces the run bit
//! for bit, which is what the scorecard's "reruns are bit-identical" criterion
//! needs. ChaCha8 is used rather than `StdRng`, whose algorithm `rand` is free to
//! change between versions.
//!
//! Streams are independent by construction: each purpose selects its own ChaCha
//! stream (`set_stream`), so adding a draw to one purpose (say, one more shuffle)
//! never shifts the numbers another purpose sees.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// What a stream is for. The discriminant and the indices pick the stream, so a
/// purpose's numbers never depend on how many draws another purpose made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Purpose {
    /// The synthetic teacher function (shared by every client).
    Teacher,
    /// Client `c`'s training data.
    ClientData(u32),
    /// The held-out evaluation set.
    HeldOut,
    /// Model initialisation.
    Init,
    /// Client `c`'s shuffle in round `r`.
    Shuffle { client: u32, round: u32 },
}

impl Purpose {
    fn stream_id(self) -> u64 {
        match self {
            Self::Teacher => 1,
            Self::HeldOut => 2,
            Self::Init => 3,
            Self::ClientData(c) => (4 << 56) | u64::from(c),
            Self::Shuffle { client, round } => {
                (5 << 56) | (u64::from(client) << 24) | u64::from(round)
            }
        }
    }
}

/// The RNG for `purpose` in the run seeded by `seed`.
pub fn rng(seed: u64, purpose: Purpose) -> ChaCha8Rng {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(purpose.stream_id());
    rng
}

/// A standard normal sample (Box-Muller), from uniform draws of `rng`. Kept here
/// rather than taken from `rand_distr` so the transform, and therefore every
/// generated value, is pinned by this crate.
pub fn normal(rng: &mut ChaCha8Rng) -> f32 {
    use rand::Rng;
    // (0, 1] for u1 so ln never sees 0.
    let u1: f64 = 1.0 - rng.gen::<f64>();
    let u2: f64 = rng.gen::<f64>();
    ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn the_same_seed_and_purpose_give_the_same_numbers() {
        let a: Vec<u32> = (0..8).map(|_| rng(7, Purpose::Init).gen()).collect();
        let mut r1 = rng(7, Purpose::Init);
        let mut r2 = rng(7, Purpose::Init);
        let x: Vec<u32> = (0..8).map(|_| r1.gen()).collect();
        let y: Vec<u32> = (0..8).map(|_| r2.gen()).collect();
        assert_eq!(x, y);
        assert_eq!(a.len(), 8);
    }

    #[test]
    fn purposes_and_seeds_are_independent_streams() {
        let first = |seed, p| rng(seed, p).gen::<u64>();
        let values = [
            first(7, Purpose::Teacher),
            first(7, Purpose::HeldOut),
            first(7, Purpose::Init),
            first(7, Purpose::ClientData(0)),
            first(7, Purpose::ClientData(1)),
            first(
                7,
                Purpose::Shuffle {
                    client: 0,
                    round: 1,
                },
            ),
            first(
                7,
                Purpose::Shuffle {
                    client: 1,
                    round: 0,
                },
            ),
            first(8, Purpose::Teacher),
        ];
        for i in 0..values.len() {
            for j in i + 1..values.len() {
                assert_ne!(values[i], values[j], "streams {i} and {j} collide");
            }
        }
    }

    /// Pins the generator itself: a `rand_chacha` upgrade that changed the stream
    /// would silently change every demo run, so it must fail here first.
    #[test]
    fn the_chacha8_stream_is_pinned() {
        let mut r = rng(480, Purpose::Teacher);
        let got: Vec<u32> = (0..4).map(|_| r.gen()).collect();
        assert_eq!(
            got,
            PINNED.to_vec(),
            "update PINNED only on a deliberate change"
        );
    }

    // The first four `u32`s of seed 480's Teacher stream.
    const PINNED: [u32; 4] = [486142396, 3643128196, 4030196220, 821065123];

    #[test]
    fn normal_samples_have_unit_moments() {
        let mut r = rng(1, Purpose::HeldOut);
        let xs: Vec<f32> = (0..20_000).map(|_| normal(&mut r)).collect();
        let mean = xs.iter().sum::<f32>() / xs.len() as f32;
        let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / xs.len() as f32;
        assert!(mean.abs() < 0.03, "mean {mean}");
        assert!((var - 1.0).abs() < 0.05, "var {var}");
    }
}
