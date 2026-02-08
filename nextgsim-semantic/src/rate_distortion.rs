//! Rate-distortion optimisation
//!
//! Implements a controller that selects the optimal compression ratio
//! for a given target distortion or target rate, taking the channel
//! quality into account.
//!
//! The approach is a simple analytical model based on the
//! rate-distortion function for a Gaussian source transmitted over
//! an AWGN channel. It can be replaced with a learned model by
//! loading an ONNX policy network.

use crate::ChannelQuality;

/// Operating mode for the rate-distortion controller.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RdMode {
    /// Target a maximum distortion (MSE). The controller picks the
    /// lowest rate that achieves the target.
    TargetDistortion {
        /// Maximum acceptable MSE
        max_mse: f32,
    },
    /// Target a maximum rate (bits per source symbol). The controller
    /// picks the compression ratio that stays within the budget.
    TargetRate {
        /// Maximum rate in bits per source symbol
        max_rate_bps: f32,
    },
}

/// Result of the rate-distortion optimisation.
#[derive(Debug, Clone, Copy)]
pub struct RdDecision {
    /// Selected compression ratio (>= 1.0; higher = more compression).
    pub compression_ratio: f32,
    /// Estimated distortion (MSE) at this operating point.
    pub estimated_mse: f32,
    /// Estimated rate in bits per source symbol.
    pub estimated_rate_bps: f32,
    /// Number of features to keep (after pruning).
    pub keep_features: usize,
}

impl std::fmt::Display for RdDecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "compression={:.2}x, rate={:.3} bps, MSE={:.6}, keep={}",
            self.compression_ratio,
            self.estimated_rate_bps,
            self.estimated_mse,
            self.keep_features,
        )
    }
}

/// Rate-distortion controller.
///
/// Given a source dimension, an operating mode, and the current channel
/// quality, the controller computes the best compression ratio.
pub struct RdController {
    /// Source feature dimension (before compression)
    source_dim: usize,
    /// Source variance estimate (default 1.0 for normalised features)
    source_variance: f32,
    /// Candidate compression ratios to search over
    candidates: Vec<f32>,
}

impl RdController {
    /// Creates a new controller for the given source dimension.
    pub fn new(source_dim: usize) -> Self {
        // Generate candidate compression ratios from 1x to 64x
        let candidates: Vec<f32> = (0..=6)
            .map(|i| 2.0f32.powi(i)) // 1, 2, 4, 8, 16, 32, 64
            .collect();

        Self {
            source_dim,
            source_variance: 1.0,
            candidates,
        }
    }

    /// Sets the source variance estimate.
    pub fn with_source_variance(mut self, variance: f32) -> Self {
        self.source_variance = variance.max(1e-6);
        self
    }

    /// Overrides the candidate compression ratios.
    pub fn with_candidates(mut self, candidates: Vec<f32>) -> Self {
        self.candidates = candidates;
        self
    }

    /// Returns the source dimension.
    pub fn source_dim(&self) -> usize {
        self.source_dim
    }

    /// Computes the rate-distortion decision for the given mode and channel.
    pub fn decide(&self, mode: RdMode, channel: &ChannelQuality) -> RdDecision {
        match mode {
            RdMode::TargetDistortion { max_mse } => self.decide_target_distortion(max_mse, channel),
            RdMode::TargetRate { max_rate_bps } => self.decide_target_rate(max_rate_bps, channel),
        }
    }

    /// Selects the lowest-rate operating point that achieves `max_mse`.
    fn decide_target_distortion(&self, max_mse: f32, channel: &ChannelQuality) -> RdDecision {
        let channel_capacity = self.channel_capacity_bps(channel);

        // Search candidates from highest compression (fewest bits) downward
        let mut best: Option<RdDecision> = None;

        for &cr in self.candidates.iter().rev() {
            let (rate, dist) = self.rd_point(cr, channel_capacity);
            if dist <= max_mse {
                let decision = RdDecision {
                    compression_ratio: cr,
                    estimated_mse: dist,
                    estimated_rate_bps: rate,
                    keep_features: (self.source_dim as f32 / cr).ceil() as usize,
                };
                match &best {
                    Some(prev) if prev.estimated_rate_bps <= decision.estimated_rate_bps => {}
                    _ => best = Some(decision),
                }
            }
        }

        // If no candidate meets the distortion target, pick the least-compressed
        best.unwrap_or_else(|| {
            let cr = self.candidates.first().copied().unwrap_or(1.0);
            let (rate, dist) = self.rd_point(cr, channel_capacity);
            RdDecision {
                compression_ratio: cr,
                estimated_mse: dist,
                estimated_rate_bps: rate,
                keep_features: (self.source_dim as f32 / cr).ceil() as usize,
            }
        })
    }

    /// Selects the operating point that maximises quality within the rate budget.
    fn decide_target_rate(&self, max_rate_bps: f32, channel: &ChannelQuality) -> RdDecision {
        let channel_capacity = self.channel_capacity_bps(channel);

        let mut best: Option<RdDecision> = None;

        for &cr in &self.candidates {
            let (rate, dist) = self.rd_point(cr, channel_capacity);
            if rate <= max_rate_bps {
                let decision = RdDecision {
                    compression_ratio: cr,
                    estimated_mse: dist,
                    estimated_rate_bps: rate,
                    keep_features: (self.source_dim as f32 / cr).ceil() as usize,
                };
                match &best {
                    Some(prev) if prev.estimated_mse <= decision.estimated_mse => {}
                    _ => best = Some(decision),
                }
            }
        }

        // If no candidate fits the rate budget, pick the most compressed
        best.unwrap_or_else(|| {
            let cr = self.candidates.last().copied().unwrap_or(64.0);
            let (rate, dist) = self.rd_point(cr, channel_capacity);
            RdDecision {
                compression_ratio: cr,
                estimated_mse: dist,
                estimated_rate_bps: rate,
                keep_features: (self.source_dim as f32 / cr).ceil() as usize,
            }
        })
    }

    /// Estimates the (rate, distortion) operating point for a given
    /// compression ratio, taking channel capacity into account.
    ///
    /// Uses the Gaussian source rate-distortion function:
    ///   D(R) = sigma^2 * 2^{-2R}
    ///   R(D) = 0.5 * log2(sigma^2 / D)
    ///
    /// The effective rate is bounded by the channel capacity.
    fn rd_point(&self, compression_ratio: f32, channel_capacity_bps: f32) -> (f32, f32) {
        // Number of compressed features
        let compressed_dim = (self.source_dim as f32 / compression_ratio).ceil().max(1.0);
        // Bits allocated per source dimension
        let rate = (compressed_dim / self.source_dim as f32) * channel_capacity_bps;
        let effective_rate = rate.min(channel_capacity_bps).max(0.0);

        // Distortion from the rate-distortion function
        let distortion = self.source_variance * 2.0f32.powf(-2.0 * effective_rate);

        (effective_rate, distortion)
    }

    /// Estimates the channel capacity in bits per channel use (Shannon formula).
    ///
    /// C = log2(1 + SNR_linear)
    fn channel_capacity_bps(&self, channel: &ChannelQuality) -> f32 {
        let snr_linear = 10.0f32.powf(channel.snr_db / 10.0);
        (1.0 + snr_linear).log2()
    }
}

/// Convenience function: given a source dimension and channel, pick a good
/// compression ratio using a target-distortion strategy.
pub fn auto_compression_ratio(
    source_dim: usize,
    channel: &ChannelQuality,
    max_mse: f32,
) -> f32 {
    let controller = RdController::new(source_dim);
    controller
        .decide(RdMode::TargetDistortion { max_mse }, channel)
        .compression_ratio
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ChannelQuality;

    #[test]
    fn test_rd_controller_basic() {
        let controller = RdController::new(256);
        let channel = ChannelQuality::new(20.0, 1000.0, 0.01);

        let decision = controller.decide(
            RdMode::TargetDistortion { max_mse: 0.01 },
            &channel,
        );

        assert!(decision.compression_ratio >= 1.0);
        assert!(decision.keep_features > 0);
        assert!(decision.keep_features <= 256);
    }

    #[test]
    fn test_rd_target_rate() {
        let controller = RdController::new(128);
        let channel = ChannelQuality::new(10.0, 500.0, 0.05);

        let decision = controller.decide(
            RdMode::TargetRate { max_rate_bps: 1.0 },
            &channel,
        );

        assert!(decision.estimated_rate_bps <= 1.0 + 0.01); // small tolerance
    }

    #[test]
    fn test_better_channel_allows_lower_compression() {
        let controller = RdController::new(256);
        let target = RdMode::TargetDistortion { max_mse: 0.001 };

        let good = ChannelQuality::new(25.0, 1000.0, 0.001);
        let poor = ChannelQuality::new(5.0, 100.0, 0.1);

        let good_decision = controller.decide(target, &good);
        let poor_decision = controller.decide(target, &poor);

        // Good channel should allow lower or equal compression
        assert!(good_decision.compression_ratio <= poor_decision.compression_ratio);
    }

    #[test]
    fn test_auto_compression_ratio() {
        let channel = ChannelQuality::new(15.0, 500.0, 0.02);
        let cr = auto_compression_ratio(256, &channel, 0.01);
        assert!(cr >= 1.0);
    }

    #[test]
    fn test_rd_decision_display() {
        let d = RdDecision {
            compression_ratio: 4.0,
            estimated_mse: 0.001,
            estimated_rate_bps: 2.5,
            keep_features: 64,
        };
        let s = format!("{d}");
        assert!(s.contains("4.00x"));
        assert!(s.contains("64"));
    }

    #[test]
    fn test_channel_capacity() {
        let controller = RdController::new(64);
        // SNR = 0 dB -> linear = 1 -> C = log2(2) = 1.0
        let ch = ChannelQuality::new(0.0, 100.0, 0.0);
        let cap = controller.channel_capacity_bps(&ch);
        assert!((cap - 1.0).abs() < 0.01);
    }
}
