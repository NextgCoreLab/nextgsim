//! The demo model: a 16→4→32→1 multilayer perceptron for regression (nextgsim#480).
//!
//! Small on purpose: 261 parameters, so a model fits one frame and a round is
//! cheap, while the 4-unit bottleneck still makes the non-IID setting matter.
//! Parameters are ONE flat `Vec<f32>` in a fixed layout, which is exactly what is
//! exchanged and aggregated:
//!
//! `W1[4×16] b1[4] W2[32×4] b2[32] W3[1×32] b3[1]`, row-major, tanh on both hidden
//! layers, linear output, mean-squared-error loss.
//!
//! Everything here is plain sequential `f32` arithmetic in a fixed order, so the
//! same inputs give bit-identical outputs.

use crate::seeded::{self, Purpose};
use rand::seq::SliceRandom;

/// Input width.
pub const IN: usize = 16;
/// First hidden layer (the bottleneck).
pub const H1: usize = 4;
/// Second hidden layer.
pub const H2: usize = 32;
/// The number of parameters: 64 + 4 + 128 + 32 + 32 + 1.
pub const PARAMS: usize = H1 * IN + H1 + H2 * H1 + H2 + H2 + 1;

// Offsets of each block in the flat layout.
const W1: usize = 0;
const B1: usize = W1 + H1 * IN;
const W2: usize = B1 + H1;
const B2: usize = W2 + H2 * H1;
const W3: usize = B2 + H2;
const B3: usize = W3 + H2;

/// Xavier-uniform initial parameters, biases zero, drawn from the run's `Init` stream.
pub fn init(seed: u64) -> Vec<f32> {
    use rand::Rng;
    let mut rng = seeded::rng(seed, Purpose::Init);
    let mut p = vec![0.0f32; PARAMS];
    for (start, fan_in, fan_out) in [(W1, IN, H1), (W2, H1, H2), (W3, H2, 1)] {
        let bound = (6.0 / (fan_in + fan_out) as f32).sqrt();
        for w in &mut p[start..start + fan_in * fan_out] {
            *w = rng.gen_range(-bound..bound);
        }
    }
    p
}

/// Hidden activations of one forward pass, kept for the backward pass.
struct Trace {
    h1: [f32; H1],
    h2: [f32; H2],
    out: f32,
}

fn forward_trace(p: &[f32], x: &[f32]) -> Trace {
    let mut h1 = [0.0f32; H1];
    for (j, h) in h1.iter_mut().enumerate() {
        let mut a = p[B1 + j];
        for i in 0..IN {
            a += p[W1 + j * IN + i] * x[i];
        }
        *h = a.tanh();
    }
    let mut h2 = [0.0f32; H2];
    for (k, h) in h2.iter_mut().enumerate() {
        let mut a = p[B2 + k];
        for j in 0..H1 {
            a += p[W2 + k * H1 + j] * h1[j];
        }
        *h = a.tanh();
    }
    let mut out = p[B3];
    for k in 0..H2 {
        out += p[W3 + k] * h2[k];
    }
    Trace { h1, h2, out }
}

/// The model's prediction for one input row (`x.len() == IN`).
pub fn predict(p: &[f32], x: &[f32]) -> f32 {
    forward_trace(p, x).out
}

/// Mean squared error of `p` on `(xs, ys)`; `xs` is `ys.len()` rows of `IN`.
pub fn mse(p: &[f32], xs: &[f32], ys: &[f32]) -> f32 {
    if ys.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0f32;
    for (n, y) in ys.iter().enumerate() {
        let e = predict(p, &xs[n * IN..(n + 1) * IN]) - y;
        sum += e * e;
    }
    sum / ys.len() as f32
}

/// Add the MSE gradient of one sample to `g` (scaled by `scale`).
fn accumulate_grad(p: &[f32], x: &[f32], y: f32, scale: f32, g: &mut [f32]) {
    let t = forward_trace(p, x);
    // d(e²)/d out = 2e
    let d_out = 2.0 * (t.out - y) * scale;
    g[B3] += d_out;
    let mut d_h2 = [0.0f32; H2];
    for k in 0..H2 {
        g[W3 + k] += d_out * t.h2[k];
        d_h2[k] = d_out * p[W3 + k] * (1.0 - t.h2[k] * t.h2[k]);
    }
    let mut d_h1 = [0.0f32; H1];
    for k in 0..H2 {
        g[B2 + k] += d_h2[k];
        for j in 0..H1 {
            g[W2 + k * H1 + j] += d_h2[k] * t.h1[j];
            d_h1[j] += d_h2[k] * p[W2 + k * H1 + j];
        }
    }
    for j in 0..H1 {
        let d = d_h1[j] * (1.0 - t.h1[j] * t.h1[j]);
        g[B1 + j] += d;
        for i in 0..IN {
            g[W1 + j * IN + i] += d * x[i];
        }
    }
}

/// Local training settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainConfig {
    pub epochs: u32,
    pub batch: usize,
    pub lr: f32,
    /// The largest L2 norm of one mini-batch gradient: a larger one is scaled
    /// down to it before the step. `f32::INFINITY` turns clipping off.
    pub clip: f32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        // Chosen with the synthetic task (nextgsim#480). On the demo's seed 480
        // the 10 rounds give about 0.18× round 1, 0.47× local-only and 1.07×
        // pooled; those are seed-480 results (see `sim`). Clipping each batch
        // gradient to norm 1 keeps training finite across seeds: without it seed
        // 481 went NaN in round 1 at this lr (nextgsim#528 review).
        Self {
            epochs: 2,
            batch: 8,
            lr: 0.15,
            clip: 1.0,
        }
    }
}

/// Train `p` in place on `(xs, ys)` with mini-batch SGD. The sample order of each
/// epoch is a shuffle drawn from the `Shuffle { client, round }` stream, so the
/// result is a pure function of the inputs.
pub fn train(
    p: &mut [f32],
    xs: &[f32],
    ys: &[f32],
    cfg: TrainConfig,
    seed: u64,
    client: u32,
    round: u32,
) {
    let mut rng = seeded::rng(seed, Purpose::Shuffle { client, round });
    let mut order: Vec<usize> = (0..ys.len()).collect();
    let mut g = vec![0.0f32; PARAMS];
    for _ in 0..cfg.epochs {
        order.shuffle(&mut rng);
        for chunk in order.chunks(cfg.batch.max(1)) {
            g.iter_mut().for_each(|v| *v = 0.0);
            let scale = 1.0 / chunk.len() as f32;
            for &n in chunk {
                accumulate_grad(p, &xs[n * IN..(n + 1) * IN], ys[n], scale, &mut g);
            }
            // Gradient-norm clipping bounds every step, so one steep batch cannot
            // throw the weights into tanh saturation and on to inf/NaN.
            let norm = g.iter().map(|v| v * v).sum::<f32>().sqrt();
            let step = if norm > cfg.clip {
                cfg.lr * cfg.clip / norm
            } else {
                cfg.lr
            };
            for (w, d) in p.iter_mut().zip(&g) {
                *w -= step * d;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_layout_is_261_parameters() {
        assert_eq!(PARAMS, 261);
        assert_eq!(B3, PARAMS - 1);
        let p = init(1);
        assert_eq!(p.len(), PARAMS);
        assert!(p[B1..W2].iter().all(|b| *b == 0.0), "biases start at zero");
        assert!(p[W1..B1].iter().any(|w| *w != 0.0));
    }

    /// The analytic gradient matches a central finite difference, parameter by
    /// parameter, so the backward pass is the derivative of the forward pass.
    #[test]
    fn the_gradient_matches_finite_differences() {
        let p = init(3);
        let x: Vec<f32> = (0..IN).map(|i| (i as f32 * 0.37).sin()).collect();
        let y = 0.4f32;
        let mut g = vec![0.0f32; PARAMS];
        accumulate_grad(&p, &x, y, 1.0, &mut g);
        let loss = |q: &[f32]| {
            let e = predict(q, &x) - y;
            f64::from(e * e)
        };
        let h = 1e-3f32;
        let mut worst = 0.0f64;
        for i in 0..PARAMS {
            let mut a = p.clone();
            let mut b = p.clone();
            a[i] += h;
            b[i] -= h;
            let fd = (loss(&a) - loss(&b)) / (2.0 * f64::from(h));
            worst = worst.max((fd - f64::from(g[i])).abs());
        }
        assert!(worst < 2e-3, "largest gradient error {worst}");
    }

    #[test]
    fn training_reduces_the_loss_and_is_deterministic() {
        // y = sum of the first four inputs, scaled.
        let n = 256;
        let mut rng = seeded::rng(9, Purpose::ClientData(0));
        let xs: Vec<f32> = (0..n * IN).map(|_| seeded::normal(&mut rng)).collect();
        let ys: Vec<f32> = (0..n)
            .map(|r| 0.25 * xs[r * IN..r * IN + 4].iter().sum::<f32>())
            .collect();
        let mut a = init(9);
        let before = mse(&a, &xs, &ys);
        let cfg = TrainConfig {
            epochs: 20,
            ..Default::default()
        };
        train(&mut a, &xs, &ys, cfg, 9, 0, 0);
        let after = mse(&a, &xs, &ys);
        assert!(after < 0.2 * before, "{before} -> {after}");
        let mut b = init(9);
        train(&mut b, &xs, &ys, cfg, 9, 0, 0);
        assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "same inputs, same bits"
        );
    }
}
