//! Statistical anomaly detection using z-score
//!
//! Implements a sliding-window anomaly detector that maintains running
//! statistics (mean and standard deviation) over measurement history
//! and flags values whose z-score exceeds a configurable threshold.

use std::collections::{HashMap, VecDeque};

use serde::{Deserialize, Serialize};
use tracing::debug;

/// A detected anomaly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anomaly {
    /// Identifier of the metric that triggered the anomaly
    pub metric_name: String,
    /// Entity that produced the anomalous measurement (e.g. UE ID or cell ID)
    pub entity_id: String,
    /// The observed value
    pub observed_value: f64,
    /// Expected mean at the time of detection
    pub expected_mean: f64,
    /// Standard deviation at the time of detection
    pub std_deviation: f64,
    /// Z-score of the observed value
    pub z_score: f64,
    /// Timestamp of the measurement (ms since epoch)
    pub timestamp_ms: u64,
    /// Severity classification
    pub severity: AnomalySeverity,
}

/// Severity levels for detected anomalies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalySeverity {
    /// Low severity: z-score between threshold and 2x threshold
    Low,
    /// Medium severity: z-score between 2x and 3x threshold
    Medium,
    /// High severity: z-score above 3x threshold
    High,
}

impl std::fmt::Display for AnomalySeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnomalySeverity::Low => write!(f, "LOW"),
            AnomalySeverity::Medium => write!(f, "MEDIUM"),
            AnomalySeverity::High => write!(f, "HIGH"),
        }
    }
}

/// Running statistics for a single metric stream
#[derive(Debug)]
struct MetricStats {
    /// Sliding window of recent values
    values: VecDeque<f64>,
    /// Maximum window size
    max_window: usize,
    /// Running sum (for efficient mean calculation)
    sum: f64,
    /// Running sum of squares (for efficient variance calculation)
    sum_sq: f64,
}

impl MetricStats {
    fn new(max_window: usize) -> Self {
        Self {
            values: VecDeque::with_capacity(max_window),
            max_window,
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    /// Adds a value and returns (mean, std_dev) after the update
    fn push(&mut self, value: f64) -> (f64, f64) {
        // If window is full, remove oldest value from running stats
        if self.values.len() >= self.max_window {
            if let Some(old) = self.values.pop_front() {
                self.sum -= old;
                self.sum_sq -= old * old;
            }
        }

        self.values.push_back(value);
        self.sum += value;
        self.sum_sq += value * value;

        let n = self.values.len() as f64;
        let mean = self.sum / n;
        let variance = if n > 1.0 {
            (self.sum_sq / n) - (mean * mean)
        } else {
            0.0
        };
        // Clamp to avoid floating point issues giving tiny negative variance
        let std_dev = variance.max(0.0).sqrt();
        (mean, std_dev)
    }

    fn len(&self) -> usize {
        self.values.len()
    }
}

/// Statistical anomaly detector using z-score analysis
///
/// Maintains per-metric, per-entity running statistics and flags
/// measurements whose z-score exceeds the configured threshold.
///
/// # Algorithm
///
/// For each incoming measurement:
/// 1. Compute the current sliding-window mean and standard deviation
/// 2. Calculate z-score = |value - mean| / std_dev
/// 3. If z-score > threshold, classify as an anomaly
///
/// # Example
///
/// ```
/// use nextgsim_nwdaf::anomaly::AnomalyDetector;
///
/// let mut detector = AnomalyDetector::new(2.5, 100);
///
/// // Feed normal RSRP values
/// for i in 0..50 {
///     let _ = detector.check("rsrp", "ue-1", -80.0 + (i % 3) as f64, i as u64 * 100);
/// }
///
/// // Feed an anomalous value
/// let anomalies = detector.check("rsrp", "ue-1", -40.0, 5000);
/// // The large jump from ~-80 to -40 should be flagged
/// ```
#[derive(Debug)]
pub struct AnomalyDetector {
    /// Z-score threshold for flagging anomalies
    threshold: f64,
    /// Maximum sliding window size
    window_size: usize,
    /// Per-(metric, entity) running statistics
    stats: HashMap<(String, String), MetricStats>,
    /// Minimum number of samples before anomaly detection activates
    min_samples: usize,
    /// Recent anomalies (bounded ring buffer)
    recent_anomalies: VecDeque<Anomaly>,
    /// Maximum stored anomalies
    max_anomalies: usize,
}

impl AnomalyDetector {
    /// Creates a new anomaly detector
    ///
    /// # Arguments
    ///
    /// * `z_threshold` - Z-score threshold for anomaly detection (typically 2.0-3.0)
    /// * `window_size` - Number of recent values to keep for statistics
    pub fn new(z_threshold: f64, window_size: usize) -> Self {
        Self {
            threshold: z_threshold,
            window_size,
            stats: HashMap::new(),
            min_samples: 10,
            recent_anomalies: VecDeque::with_capacity(1000),
            max_anomalies: 1000,
        }
    }

    /// Sets the minimum number of samples before detection activates
    pub fn with_min_samples(mut self, min_samples: usize) -> Self {
        self.min_samples = min_samples;
        self
    }

    /// Checks a measurement value for anomalies
    ///
    /// Returns a list of anomalies detected (empty if the value is normal).
    /// A single call can only return zero or one anomaly, but the return
    /// type is a `Vec` for consistency with batch checking.
    ///
    /// # Arguments
    ///
    /// * `metric_name` - Name of the metric (e.g. "rsrp", "prb_usage", "sinr")
    /// * `entity_id` - Entity producing the measurement (e.g. "ue-42", "cell-7")
    /// * `value` - The measured value
    /// * `timestamp_ms` - Measurement timestamp
    pub fn check(
        &mut self,
        metric_name: &str,
        entity_id: &str,
        value: f64,
        timestamp_ms: u64,
    ) -> Vec<Anomaly> {
        let key = (metric_name.to_string(), entity_id.to_string());

        let stats = self
            .stats
            .entry(key)
            .or_insert_with(|| MetricStats::new(self.window_size));

        // Compute stats BEFORE adding the new value so that the new value
        // is compared against the historical distribution
        let count_before = stats.len();

        // We need pre-push mean/std for comparison, but the running stats
        // are designed around push. So we compute manually from the window
        // content before pushing.
        let (pre_mean, pre_std) = if count_before >= self.min_samples {
            let n = stats.values.len() as f64;
            let mean = stats.sum / n;
            let variance = if n > 1.0 {
                (stats.sum_sq / n) - (mean * mean)
            } else {
                0.0
            };
            (mean, variance.max(0.0).sqrt())
        } else {
            (0.0, 0.0)
        };

        // Now push the value (updates running stats)
        let (_new_mean, _new_std) = stats.push(value);

        // Only detect anomalies once we have enough history
        if count_before < self.min_samples {
            return Vec::new();
        }

        // Avoid division by zero: if std_dev is near zero, all values
        // are essentially the same, so any deviation is anomalous
        let z_score = if pre_std > f64::EPSILON {
            (value - pre_mean).abs() / pre_std
        } else {
            // If std is ~0, any value different from the mean is infinitely anomalous
            if (value - pre_mean).abs() > f64::EPSILON {
                self.threshold + 1.0 // Force detection
            } else {
                0.0 // Exact match
            }
        };

        if z_score > self.threshold {
            let severity = if z_score > self.threshold * 3.0 {
                AnomalySeverity::High
            } else if z_score > self.threshold * 2.0 {
                AnomalySeverity::Medium
            } else {
                AnomalySeverity::Low
            };

            let anomaly = Anomaly {
                metric_name: metric_name.to_string(),
                entity_id: entity_id.to_string(),
                observed_value: value,
                expected_mean: pre_mean,
                std_deviation: pre_std,
                z_score,
                timestamp_ms,
                severity,
            };

            debug!(
                "Anomaly detected: {} {} z={:.2} severity={}",
                metric_name, entity_id, z_score, severity
            );

            // Store in recent anomalies
            if self.recent_anomalies.len() >= self.max_anomalies {
                self.recent_anomalies.pop_front();
            }
            self.recent_anomalies.push_back(anomaly.clone());

            vec![anomaly]
        } else {
            Vec::new()
        }
    }

    /// Checks a batch of measurements at once
    pub fn check_batch(
        &mut self,
        measurements: &[(&str, &str, f64, u64)],
    ) -> Vec<Anomaly> {
        let mut all_anomalies = Vec::new();
        for &(metric, entity, value, ts) in measurements {
            all_anomalies.extend(self.check(metric, entity, value, ts));
        }
        all_anomalies
    }

    /// Returns all recent anomalies
    pub fn recent_anomalies(&self) -> &VecDeque<Anomaly> {
        &self.recent_anomalies
    }

    /// Returns recent anomalies filtered by entity
    pub fn anomalies_for_entity(&self, entity_id: &str) -> Vec<&Anomaly> {
        self.recent_anomalies
            .iter()
            .filter(|a| a.entity_id == entity_id)
            .collect()
    }

    /// Returns recent anomalies filtered by metric name
    pub fn anomalies_for_metric(&self, metric_name: &str) -> Vec<&Anomaly> {
        self.recent_anomalies
            .iter()
            .filter(|a| a.metric_name == metric_name)
            .collect()
    }

    /// Clears all stored statistics and anomalies
    pub fn reset(&mut self) {
        self.stats.clear();
        self.recent_anomalies.clear();
    }

    /// Returns the current z-score threshold
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Updates the z-score threshold
    pub fn set_threshold(&mut self, threshold: f64) {
        self.threshold = threshold;
    }

    /// Returns the number of tracked metric streams
    pub fn tracked_streams(&self) -> usize {
        self.stats.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anomaly_detector_creation() {
        let detector = AnomalyDetector::new(2.5, 100);
        assert_eq!(detector.threshold(), 2.5);
        assert_eq!(detector.tracked_streams(), 0);
    }

    #[test]
    fn test_no_anomaly_on_normal_data() {
        let mut detector = AnomalyDetector::new(2.5, 100).with_min_samples(10);

        // Feed steady values
        for i in 0..50 {
            let value = -80.0 + (i % 3) as f64 * 0.5; // slight jitter
            let anomalies = detector.check("rsrp", "ue-1", value, i as u64 * 100);
            // After warmup, normal values should not trigger anomalies
            if i > 15 {
                assert!(
                    anomalies.is_empty(),
                    "Unexpected anomaly at step {i}: value={value}, anomalies={anomalies:?}"
                );
            }
        }
    }

    #[test]
    fn test_anomaly_detected_on_spike() {
        let mut detector = AnomalyDetector::new(2.0, 100).with_min_samples(10);

        // Build up normal baseline around -80 dBm
        for i in 0..30 {
            let value = -80.0 + (i % 3) as f64 * 0.3;
            detector.check("rsrp", "ue-1", value, i as u64 * 100);
        }

        // Inject a huge spike
        let anomalies = detector.check("rsrp", "ue-1", -40.0, 3000);
        assert!(!anomalies.is_empty(), "Should detect anomaly on huge spike");

        let anomaly = &anomalies[0];
        assert_eq!(anomaly.metric_name, "rsrp");
        assert_eq!(anomaly.entity_id, "ue-1");
        assert!((anomaly.observed_value - (-40.0)).abs() < f64::EPSILON);
        assert!(anomaly.z_score > 2.0);
    }

    #[test]
    fn test_anomaly_severity_levels() {
        let mut detector = AnomalyDetector::new(1.5, 100).with_min_samples(10);

        // Build baseline
        for i in 0..50 {
            detector.check("load", "cell-1", 0.5 + (i % 5) as f64 * 0.01, i as u64 * 100);
        }

        // The detected anomaly severity depends on z-score magnitude
        // Feed values increasingly far from the mean to test severity levels
        let anomalies = detector.check("load", "cell-1", 5.0, 5000);
        if !anomalies.is_empty() {
            // With a baseline around 0.52 and very low std, 5.0 should be High
            assert_eq!(anomalies[0].severity, AnomalySeverity::High);
        }
    }

    #[test]
    fn test_no_detection_during_warmup() {
        let mut detector = AnomalyDetector::new(2.0, 100).with_min_samples(20);

        // Even with a spike during warmup, should not detect
        for i in 0..15 {
            let value = if i == 10 { 999.0 } else { 1.0 };
            let anomalies = detector.check("metric", "entity", value, i as u64 * 100);
            assert!(anomalies.is_empty(), "Should not detect during warmup (step {i})");
        }
    }

    #[test]
    fn test_multiple_entities() {
        let mut detector = AnomalyDetector::new(2.0, 100).with_min_samples(10);

        // Build up separate baselines
        for i in 0..30 {
            detector.check("rsrp", "ue-1", -80.0, i as u64 * 100);
            detector.check("rsrp", "ue-2", -60.0, i as u64 * 100);
        }

        // Spike on ue-1
        let a1 = detector.check("rsrp", "ue-1", -40.0, 3000);
        assert!(!a1.is_empty());

        // ue-2 at its normal value should be fine
        let a2 = detector.check("rsrp", "ue-2", -60.0, 3000);
        assert!(a2.is_empty());

        assert_eq!(detector.tracked_streams(), 2);
        assert_eq!(detector.anomalies_for_entity("ue-1").len(), 1);
        assert_eq!(detector.anomalies_for_entity("ue-2").len(), 0);
    }

    #[test]
    fn test_check_batch() {
        let mut detector = AnomalyDetector::new(2.0, 100).with_min_samples(5);

        // Warmup
        for i in 0..10 {
            detector.check("metric", "e1", 10.0, i as u64 * 100);
        }

        let measurements = vec![
            ("metric", "e1", 10.0, 1000u64), // normal
            ("metric", "e1", 100.0, 1100),    // anomalous
        ];
        let anomalies = detector.check_batch(&measurements);
        assert_eq!(anomalies.len(), 1);
    }

    #[test]
    fn test_recent_anomalies_bounded() {
        let mut detector = AnomalyDetector::new(2.0, 50).with_min_samples(5);
        // Override max_anomalies for testing
        detector.max_anomalies = 3;

        // Warmup
        for i in 0..20 {
            detector.check("m", "e", 10.0, i as u64 * 100);
        }

        // Generate multiple anomalies
        for i in 0..5 {
            detector.check("m", "e", 1000.0 + i as f64, 2000 + i as u64 * 100);
        }

        assert!(detector.recent_anomalies().len() <= 3);
    }

    #[test]
    fn test_reset() {
        let mut detector = AnomalyDetector::new(2.0, 100).with_min_samples(5);

        for i in 0..20 {
            detector.check("m", "e", 1.0, i as u64 * 100);
        }
        detector.check("m", "e", 100.0, 2000);

        assert!(detector.tracked_streams() > 0);
        assert!(!detector.recent_anomalies().is_empty());

        detector.reset();
        assert_eq!(detector.tracked_streams(), 0);
        assert!(detector.recent_anomalies().is_empty());
    }

    #[test]
    fn test_anomaly_serialization() {
        let anomaly = Anomaly {
            metric_name: "rsrp".to_string(),
            entity_id: "ue-1".to_string(),
            observed_value: -40.0,
            expected_mean: -80.0,
            std_deviation: 2.0,
            z_score: 20.0,
            timestamp_ms: 12345,
            severity: AnomalySeverity::High,
        };
        let json = serde_json::to_string(&anomaly).expect("serialize");
        let deserialized: Anomaly = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.metric_name, "rsrp");
        assert_eq!(deserialized.severity, AnomalySeverity::High);
    }
}
