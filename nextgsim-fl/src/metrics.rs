//! Training Convergence Metrics and Dashboard (A17.8)
//!
//! Tracks training progress, convergence, and participant contributions
//! for federated learning systems.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Training metrics for a single round
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundMetrics {
    /// Round number
    pub round: u64,
    /// Average training loss across participants
    pub avg_loss: f32,
    /// Minimum loss among participants
    pub min_loss: f32,
    /// Maximum loss among participants
    pub max_loss: f32,
    /// Number of participants
    pub num_participants: u32,
    /// Total samples trained on
    pub total_samples: u64,
    /// Round duration in milliseconds
    pub duration_ms: u64,
    /// Timestamp when round completed
    pub timestamp_ms: u64,
}

/// Per-participant contribution tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticipantContribution {
    /// Participant ID
    pub participant_id: String,
    /// Number of rounds participated in
    pub rounds_participated: u64,
    /// Total samples contributed
    pub total_samples: u64,
    /// Average loss across all rounds
    pub avg_loss: f32,
    /// Last seen timestamp
    pub last_seen_ms: u64,
    /// Reliability score (fraction of rounds completed on time)
    pub reliability: f32,
}

/// Convergence detector using moving statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceDetector {
    /// Window size for moving average
    window_size: usize,
    /// Recent loss values
    loss_history: VecDeque<f32>,
    /// Convergence threshold (relative change)
    threshold: f32,
}

impl ConvergenceDetector {
    /// Creates a new convergence detector
    pub fn new(window_size: usize, threshold: f32) -> Self {
        Self {
            window_size,
            loss_history: VecDeque::with_capacity(window_size),
            threshold,
        }
    }

    /// Records a new loss value
    pub fn record(&mut self, loss: f32) {
        if self.loss_history.len() >= self.window_size {
            self.loss_history.pop_front();
        }
        self.loss_history.push_back(loss);
    }

    /// Checks if training has converged
    /// Convergence is detected when the relative change in loss over the window
    /// is below the threshold
    pub fn has_converged(&self) -> bool {
        if self.loss_history.len() < self.window_size {
            return false;
        }

        let first_half: f32 = self.loss_history
            .iter()
            .take(self.window_size / 2)
            .sum::<f32>()
            / (self.window_size / 2) as f32;

        let second_half: f32 = self.loss_history
            .iter()
            .skip(self.window_size / 2)
            .sum::<f32>()
            / (self.window_size - self.window_size / 2) as f32;

        if first_half == 0.0 {
            return false;
        }

        let relative_change = ((first_half - second_half) / first_half).abs();
        relative_change < self.threshold
    }

    /// Returns the current moving average loss
    pub fn moving_average(&self) -> Option<f32> {
        if self.loss_history.is_empty() {
            None
        } else {
            Some(self.loss_history.iter().sum::<f32>() / self.loss_history.len() as f32)
        }
    }
}

/// Training dashboard for monitoring FL progress
pub struct TrainingDashboard {
    /// Historical round metrics
    round_history: Vec<RoundMetrics>,
    /// Participant contributions
    contributions: HashMap<String, ParticipantContribution>,
    /// Convergence detector
    convergence: ConvergenceDetector,
    /// Maximum history to keep
    max_history: usize,
}

impl TrainingDashboard {
    /// Creates a new training dashboard
    pub fn new(convergence_window: usize, convergence_threshold: f32, max_history: usize) -> Self {
        Self {
            round_history: Vec::new(),
            contributions: HashMap::new(),
            convergence: ConvergenceDetector::new(convergence_window, convergence_threshold),
            max_history,
        }
    }

    /// Records metrics for a completed round
    pub fn record_round(&mut self, metrics: RoundMetrics) {
        self.convergence.record(metrics.avg_loss);
        self.round_history.push(metrics);

        // Prune old history
        if self.round_history.len() > self.max_history {
            let to_remove = self.round_history.len() - self.max_history;
            self.round_history.drain(0..to_remove);
        }
    }

    /// Records a participant's contribution for a round
    pub fn record_contribution(
        &mut self,
        participant_id: String,
        samples: u64,
        loss: f32,
        completed: bool,
    ) {
        let contribution = self
            .contributions
            .entry(participant_id.clone())
            .or_insert_with(|| ParticipantContribution {
                participant_id,
                rounds_participated: 0,
                total_samples: 0,
                avg_loss: 0.0,
                last_seen_ms: 0,
                reliability: 1.0,
            });

        contribution.rounds_participated += 1;
        contribution.total_samples += samples;

        // Update moving average of loss
        let n = contribution.rounds_participated as f32;
        contribution.avg_loss = (contribution.avg_loss * (n - 1.0) + loss) / n;

        contribution.last_seen_ms = timestamp_now();

        // Update reliability (exponential moving average)
        let alpha = 0.1; // Smoothing factor
        contribution.reliability = alpha * if completed { 1.0 } else { 0.0 }
            + (1.0 - alpha) * contribution.reliability;
    }

    /// Returns whether training has converged
    pub fn has_converged(&self) -> bool {
        self.convergence.has_converged()
    }

    /// Returns the current loss trend
    pub fn loss_trend(&self) -> LossTrend {
        if self.round_history.len() < 2 {
            return LossTrend::Unknown;
        }

        let recent_rounds = self.round_history.len().min(10);
        let recent = &self.round_history[self.round_history.len() - recent_rounds..];

        let first_avg = recent.iter().take(recent_rounds / 2).map(|r| r.avg_loss).sum::<f32>()
            / (recent_rounds / 2) as f32;
        let second_avg = recent.iter().skip(recent_rounds / 2).map(|r| r.avg_loss).sum::<f32>()
            / (recent_rounds - recent_rounds / 2) as f32;

        if second_avg < first_avg * 0.95 {
            LossTrend::Decreasing
        } else if second_avg > first_avg * 1.05 {
            LossTrend::Increasing
        } else {
            LossTrend::Stable
        }
    }

    /// Returns top contributors by total samples
    pub fn top_contributors(&self, k: usize) -> Vec<ParticipantContribution> {
        let mut contributions: Vec<_> = self.contributions.values().cloned().collect();
        contributions.sort_by(|a, b| b.total_samples.cmp(&a.total_samples));
        contributions.truncate(k);
        contributions
    }

    /// Returns training summary statistics
    pub fn summary(&self) -> TrainingSummary {
        let total_rounds = self.round_history.len() as u64;
        let current_loss = self
            .round_history
            .last()
            .map(|r| r.avg_loss)
            .unwrap_or(0.0);

        let best_loss = self
            .round_history
            .iter()
            .map(|r| r.avg_loss)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        let total_samples: u64 = self.contributions.values().map(|c| c.total_samples).sum();
        let num_participants = self.contributions.len();
        let avg_reliability = if num_participants == 0 {
            0.0
        } else {
            self.contributions.values().map(|c| c.reliability).sum::<f32>()
                / num_participants as f32
        };

        TrainingSummary {
            total_rounds,
            current_loss,
            best_loss,
            loss_trend: self.loss_trend(),
            has_converged: self.has_converged(),
            total_samples,
            num_participants,
            avg_reliability,
        }
    }

    /// Returns the full round history
    pub fn round_history(&self) -> &[RoundMetrics] {
        &self.round_history
    }

    /// Returns participant contributions
    pub fn contributions(&self) -> &HashMap<String, ParticipantContribution> {
        &self.contributions
    }

    /// Exports dashboard data as JSON
    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        #[derive(Serialize)]
        struct DashboardExport<'a> {
            round_history: &'a [RoundMetrics],
            contributions: &'a HashMap<String, ParticipantContribution>,
            summary: TrainingSummary,
        }

        let export = DashboardExport {
            round_history: &self.round_history,
            contributions: &self.contributions,
            summary: self.summary(),
        };

        serde_json::to_string_pretty(&export)
    }
}

impl Default for TrainingDashboard {
    fn default() -> Self {
        Self::new(10, 0.01, 1000)
    }
}

/// Loss trend indicator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LossTrend {
    /// Loss is decreasing (training improving)
    Decreasing,
    /// Loss is stable (converged or stuck)
    Stable,
    /// Loss is increasing (diverging or overfitting)
    Increasing,
    /// Not enough data to determine trend
    Unknown,
}

/// Training summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSummary {
    /// Total number of rounds completed
    pub total_rounds: u64,
    /// Current (most recent) loss
    pub current_loss: f32,
    /// Best (minimum) loss achieved
    pub best_loss: f32,
    /// Current loss trend
    pub loss_trend: LossTrend,
    /// Whether training has converged
    pub has_converged: bool,
    /// Total samples contributed by all participants
    pub total_samples: u64,
    /// Number of unique participants
    pub num_participants: usize,
    /// Average participant reliability
    pub avg_reliability: f32,
}

impl std::fmt::Display for TrainingSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Rounds: {}, Current Loss: {:.6}, Best Loss: {:.6}, Trend: {:?}, Converged: {}, Participants: {}, Samples: {}",
            self.total_rounds,
            self.current_loss,
            self.best_loss,
            self.loss_trend,
            self.has_converged,
            self.num_participants,
            self.total_samples,
        )
    }
}

/// Gets current timestamp in milliseconds
fn timestamp_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convergence_detector() {
        let mut detector = ConvergenceDetector::new(4, 0.01);

        // Decreasing loss - should not converge yet
        detector.record(1.0);
        detector.record(0.9);
        detector.record(0.8);
        detector.record(0.7);

        assert!(!detector.has_converged());

        // Stable loss - should converge
        detector.record(0.705);
        detector.record(0.702);
        detector.record(0.701);
        detector.record(0.700);

        assert!(detector.has_converged());
    }

    #[test]
    fn test_convergence_moving_average() {
        let mut detector = ConvergenceDetector::new(3, 0.01);

        detector.record(1.0);
        detector.record(2.0);
        detector.record(3.0);

        let avg = detector.moving_average().unwrap();
        assert!((avg - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_training_dashboard() {
        let mut dashboard = TrainingDashboard::new(5, 0.01, 100);

        // Record some rounds
        for round in 1..=10 {
            let metrics = RoundMetrics {
                round,
                avg_loss: 1.0 / round as f32,
                min_loss: 0.8 / round as f32,
                max_loss: 1.2 / round as f32,
                num_participants: 5,
                total_samples: 500,
                duration_ms: 1000,
                timestamp_ms: round * 1000,
            };
            dashboard.record_round(metrics);
        }

        assert_eq!(dashboard.round_history().len(), 10);
        assert_eq!(dashboard.loss_trend(), LossTrend::Decreasing);
    }

    #[test]
    fn test_participant_contributions() {
        let mut dashboard = TrainingDashboard::default();

        dashboard.record_contribution("client1".to_string(), 100, 0.5, true);
        dashboard.record_contribution("client2".to_string(), 200, 0.4, true);
        dashboard.record_contribution("client1".to_string(), 150, 0.45, true);

        let contributions = dashboard.contributions();
        assert_eq!(contributions.len(), 2);

        let client1 = contributions.get("client1").unwrap();
        assert_eq!(client1.rounds_participated, 2);
        assert_eq!(client1.total_samples, 250);
    }

    #[test]
    fn test_top_contributors() {
        let mut dashboard = TrainingDashboard::default();

        dashboard.record_contribution("client1".to_string(), 1000, 0.5, true);
        dashboard.record_contribution("client2".to_string(), 2000, 0.4, true);
        dashboard.record_contribution("client3".to_string(), 500, 0.6, true);

        let top = dashboard.top_contributors(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].participant_id, "client2");
        assert_eq!(top[1].participant_id, "client1");
    }

    #[test]
    fn test_training_summary() {
        let mut dashboard = TrainingDashboard::new(5, 0.01, 100);

        for round in 1..=5 {
            let metrics = RoundMetrics {
                round,
                avg_loss: 1.0 - 0.1 * round as f32,
                min_loss: 0.8 - 0.1 * round as f32,
                max_loss: 1.2 - 0.1 * round as f32,
                num_participants: 3,
                total_samples: 300,
                duration_ms: 1000,
                timestamp_ms: round * 1000,
            };
            dashboard.record_round(metrics);
        }

        dashboard.record_contribution("client1".to_string(), 500, 0.5, true);

        let summary = dashboard.summary();
        assert_eq!(summary.total_rounds, 5);
        assert_eq!(summary.num_participants, 1);
        assert_eq!(summary.total_samples, 500);
    }

    #[test]
    fn test_reliability_tracking() {
        let mut dashboard = TrainingDashboard::default();

        // Client completes 3 rounds successfully
        dashboard.record_contribution("client1".to_string(), 100, 0.5, true);
        dashboard.record_contribution("client1".to_string(), 100, 0.5, true);
        dashboard.record_contribution("client1".to_string(), 100, 0.5, true);

        let contrib = dashboard.contributions().get("client1").unwrap();
        assert!(contrib.reliability > 0.9);

        // Client fails a round
        dashboard.record_contribution("client1".to_string(), 100, 0.5, false);

        let contrib = dashboard.contributions().get("client1").unwrap();
        assert!(contrib.reliability < 1.0);
    }

    #[test]
    fn test_export_json() {
        let mut dashboard = TrainingDashboard::new(5, 0.01, 100);

        let metrics = RoundMetrics {
            round: 1,
            avg_loss: 0.5,
            min_loss: 0.4,
            max_loss: 0.6,
            num_participants: 2,
            total_samples: 200,
            duration_ms: 1000,
            timestamp_ms: 1000,
        };
        dashboard.record_round(metrics);

        let json = dashboard.export_json().unwrap();
        assert!(json.contains("round_history"));
        assert!(json.contains("contributions"));
        assert!(json.contains("summary"));
    }
}
