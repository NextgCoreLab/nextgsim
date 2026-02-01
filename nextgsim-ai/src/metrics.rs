//! Inference metrics and monitoring
//!
//! This module provides structures for tracking inference performance metrics,
//! including latency, throughput, and resource utilization.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Inference metrics for a single model
#[derive(Debug)]
pub struct InferenceMetrics {
    /// Model name
    model_name: String,
    /// Total inference count
    inference_count: u64,
    /// Total errors
    error_count: u64,
    /// Latency samples (recent)
    latency_samples: VecDeque<Duration>,
    /// Maximum samples to keep
    max_samples: usize,
    /// Start time for throughput calculation
    start_time: Instant,
    /// Total bytes processed (input)
    total_input_bytes: u64,
    /// Total bytes produced (output)
    total_output_bytes: u64,
}

impl InferenceMetrics {
    /// Creates a new metrics tracker for a model
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            inference_count: 0,
            error_count: 0,
            latency_samples: VecDeque::with_capacity(1000),
            max_samples: 1000,
            start_time: Instant::now(),
            total_input_bytes: 0,
            total_output_bytes: 0,
        }
    }

    /// Records a successful inference
    pub fn record_inference(&mut self, latency: Duration, input_bytes: u64, output_bytes: u64) {
        self.inference_count += 1;
        self.total_input_bytes += input_bytes;
        self.total_output_bytes += output_bytes;

        if self.latency_samples.len() >= self.max_samples {
            self.latency_samples.pop_front();
        }
        self.latency_samples.push_back(latency);
    }

    /// Records an inference error
    pub fn record_error(&mut self) {
        self.error_count += 1;
    }

    /// Returns the model name
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Returns the total inference count
    pub fn inference_count(&self) -> u64 {
        self.inference_count
    }

    /// Returns the error count
    pub fn error_count(&self) -> u64 {
        self.error_count
    }

    /// Returns the success rate
    pub fn success_rate(&self) -> f64 {
        let total = self.inference_count + self.error_count;
        if total == 0 {
            1.0
        } else {
            self.inference_count as f64 / total as f64
        }
    }

    /// Returns the average latency
    pub fn avg_latency(&self) -> Option<Duration> {
        if self.latency_samples.is_empty() {
            return None;
        }
        let sum: Duration = self.latency_samples.iter().sum();
        Some(sum / self.latency_samples.len() as u32)
    }

    /// Returns the p50 (median) latency
    pub fn p50_latency(&self) -> Option<Duration> {
        self.percentile_latency(0.50)
    }

    /// Returns the p95 latency
    pub fn p95_latency(&self) -> Option<Duration> {
        self.percentile_latency(0.95)
    }

    /// Returns the p99 latency
    pub fn p99_latency(&self) -> Option<Duration> {
        self.percentile_latency(0.99)
    }

    /// Returns the min latency
    pub fn min_latency(&self) -> Option<Duration> {
        self.latency_samples.iter().min().copied()
    }

    /// Returns the max latency
    pub fn max_latency(&self) -> Option<Duration> {
        self.latency_samples.iter().max().copied()
    }

    /// Returns the throughput (inferences per second)
    pub fn throughput(&self) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed == 0.0 {
            0.0
        } else {
            self.inference_count as f64 / elapsed
        }
    }

    /// Returns the input bandwidth (bytes per second)
    pub fn input_bandwidth(&self) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed == 0.0 {
            0.0
        } else {
            self.total_input_bytes as f64 / elapsed
        }
    }

    /// Returns the output bandwidth (bytes per second)
    pub fn output_bandwidth(&self) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed == 0.0 {
            0.0
        } else {
            self.total_output_bytes as f64 / elapsed
        }
    }

    /// Resets all metrics
    pub fn reset(&mut self) {
        self.inference_count = 0;
        self.error_count = 0;
        self.latency_samples.clear();
        self.start_time = Instant::now();
        self.total_input_bytes = 0;
        self.total_output_bytes = 0;
    }

    /// Returns a percentile latency
    fn percentile_latency(&self, percentile: f64) -> Option<Duration> {
        if self.latency_samples.is_empty() {
            return None;
        }
        let mut sorted: Vec<Duration> = self.latency_samples.iter().copied().collect();
        sorted.sort();
        let index = ((sorted.len() as f64 * percentile) as usize).min(sorted.len() - 1);
        Some(sorted[index])
    }

    /// Returns a summary of the metrics
    pub fn summary(&self) -> MetricsSummary {
        MetricsSummary {
            model_name: self.model_name.clone(),
            inference_count: self.inference_count,
            error_count: self.error_count,
            success_rate: self.success_rate(),
            avg_latency_ms: self.avg_latency().map(|d| d.as_secs_f64() * 1000.0),
            p50_latency_ms: self.p50_latency().map(|d| d.as_secs_f64() * 1000.0),
            p95_latency_ms: self.p95_latency().map(|d| d.as_secs_f64() * 1000.0),
            p99_latency_ms: self.p99_latency().map(|d| d.as_secs_f64() * 1000.0),
            throughput: self.throughput(),
        }
    }
}

/// Summary of inference metrics
#[derive(Debug, Clone)]
pub struct MetricsSummary {
    /// Model name
    pub model_name: String,
    /// Total inference count
    pub inference_count: u64,
    /// Error count
    pub error_count: u64,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Average latency in milliseconds
    pub avg_latency_ms: Option<f64>,
    /// P50 latency in milliseconds
    pub p50_latency_ms: Option<f64>,
    /// P95 latency in milliseconds
    pub p95_latency_ms: Option<f64>,
    /// P99 latency in milliseconds
    pub p99_latency_ms: Option<f64>,
    /// Throughput in inferences per second
    pub throughput: f64,
}

impl std::fmt::Display for MetricsSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Model: {}", self.model_name)?;
        writeln!(f, "  Inferences: {}", self.inference_count)?;
        writeln!(f, "  Errors: {}", self.error_count)?;
        writeln!(f, "  Success Rate: {:.2}%", self.success_rate * 100.0)?;
        if let Some(avg) = self.avg_latency_ms {
            writeln!(f, "  Avg Latency: {:.2}ms", avg)?;
        }
        if let Some(p50) = self.p50_latency_ms {
            writeln!(f, "  P50 Latency: {:.2}ms", p50)?;
        }
        if let Some(p95) = self.p95_latency_ms {
            writeln!(f, "  P95 Latency: {:.2}ms", p95)?;
        }
        if let Some(p99) = self.p99_latency_ms {
            writeln!(f, "  P99 Latency: {:.2}ms", p99)?;
        }
        writeln!(f, "  Throughput: {:.2} inf/s", self.throughput)?;
        Ok(())
    }
}

/// Aggregate metrics for multiple models
#[derive(Debug, Default)]
pub struct ModelMetrics {
    /// Metrics per model
    metrics: std::collections::HashMap<String, InferenceMetrics>,
}

impl ModelMetrics {
    /// Creates a new ModelMetrics
    pub fn new() -> Self {
        Self::default()
    }

    /// Gets or creates metrics for a model
    pub fn get_or_create(&mut self, model_name: &str) -> &mut InferenceMetrics {
        self.metrics
            .entry(model_name.to_string())
            .or_insert_with(|| InferenceMetrics::new(model_name))
    }

    /// Gets metrics for a model
    pub fn get(&self, model_name: &str) -> Option<&InferenceMetrics> {
        self.metrics.get(model_name)
    }

    /// Returns summaries for all models
    pub fn summaries(&self) -> Vec<MetricsSummary> {
        self.metrics.values().map(|m| m.summary()).collect()
    }

    /// Resets all metrics
    pub fn reset_all(&mut self) {
        for metrics in self.metrics.values_mut() {
            metrics.reset();
        }
    }

    /// Returns total inference count across all models
    pub fn total_inference_count(&self) -> u64 {
        self.metrics.values().map(|m| m.inference_count()).sum()
    }

    /// Returns total error count across all models
    pub fn total_error_count(&self) -> u64 {
        self.metrics.values().map(|m| m.error_count()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_metrics_basic() {
        let mut metrics = InferenceMetrics::new("test_model");

        assert_eq!(metrics.model_name(), "test_model");
        assert_eq!(metrics.inference_count(), 0);
        assert_eq!(metrics.error_count(), 0);
        assert_eq!(metrics.success_rate(), 1.0);
    }

    #[test]
    fn test_inference_metrics_recording() {
        let mut metrics = InferenceMetrics::new("test_model");

        metrics.record_inference(Duration::from_millis(10), 1024, 512);
        metrics.record_inference(Duration::from_millis(15), 1024, 512);
        metrics.record_inference(Duration::from_millis(12), 1024, 512);

        assert_eq!(metrics.inference_count(), 3);
        assert!(metrics.avg_latency().is_some());

        let avg = metrics.avg_latency().unwrap();
        // Average of 10, 15, 12 = ~12.33ms
        assert!(avg.as_millis() >= 12 && avg.as_millis() <= 13);
    }

    #[test]
    fn test_inference_metrics_errors() {
        let mut metrics = InferenceMetrics::new("test_model");

        metrics.record_inference(Duration::from_millis(10), 100, 50);
        metrics.record_inference(Duration::from_millis(10), 100, 50);
        metrics.record_error();

        assert_eq!(metrics.inference_count(), 2);
        assert_eq!(metrics.error_count(), 1);
        assert!((metrics.success_rate() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_inference_metrics_percentiles() {
        let mut metrics = InferenceMetrics::new("test_model");

        // Add samples
        for i in 1..=100 {
            metrics.record_inference(Duration::from_millis(i), 100, 50);
        }

        let p50 = metrics.p50_latency().unwrap();
        let p95 = metrics.p95_latency().unwrap();
        let p99 = metrics.p99_latency().unwrap();

        assert!(p50.as_millis() >= 49 && p50.as_millis() <= 51);
        assert!(p95.as_millis() >= 94 && p95.as_millis() <= 96);
        assert!(p99.as_millis() >= 98 && p99.as_millis() <= 100);
    }

    #[test]
    fn test_inference_metrics_reset() {
        let mut metrics = InferenceMetrics::new("test_model");

        metrics.record_inference(Duration::from_millis(10), 100, 50);
        metrics.record_error();

        metrics.reset();

        assert_eq!(metrics.inference_count(), 0);
        assert_eq!(metrics.error_count(), 0);
        assert!(metrics.avg_latency().is_none());
    }

    #[test]
    fn test_metrics_summary() {
        let mut metrics = InferenceMetrics::new("test_model");
        metrics.record_inference(Duration::from_millis(10), 100, 50);

        let summary = metrics.summary();
        assert_eq!(summary.model_name, "test_model");
        assert_eq!(summary.inference_count, 1);
        assert!(summary.avg_latency_ms.is_some());
    }

    #[test]
    fn test_model_metrics() {
        let mut model_metrics = ModelMetrics::new();

        {
            let m1 = model_metrics.get_or_create("model1");
            m1.record_inference(Duration::from_millis(10), 100, 50);
        }
        {
            let m2 = model_metrics.get_or_create("model2");
            m2.record_inference(Duration::from_millis(20), 200, 100);
            m2.record_error();
        }

        assert_eq!(model_metrics.total_inference_count(), 2);
        assert_eq!(model_metrics.total_error_count(), 1);

        let summaries = model_metrics.summaries();
        assert_eq!(summaries.len(), 2);
    }
}
