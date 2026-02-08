//! SLA enforcement and QoS monitoring
//!
//! Implements Service Level Agreement violation detection and enforcement.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::workload::WorkloadId;

/// SLA metric type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SlaMetric {
    /// Latency (milliseconds)
    Latency,
    /// Throughput (requests per second)
    Throughput,
    /// Availability (percentage)
    Availability,
    /// Success rate (percentage)
    SuccessRate,
    /// Response time (milliseconds)
    ResponseTime,
}

impl std::fmt::Display for SlaMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SlaMetric::Latency => write!(f, "Latency"),
            SlaMetric::Throughput => write!(f, "Throughput"),
            SlaMetric::Availability => write!(f, "Availability"),
            SlaMetric::SuccessRate => write!(f, "Success Rate"),
            SlaMetric::ResponseTime => write!(f, "Response Time"),
        }
    }
}

/// SLA objective
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaObjective {
    /// Metric
    pub metric: SlaMetric,
    /// Target value
    pub target: f64,
    /// Threshold for violation (percentage below target)
    pub violation_threshold: f64,
    /// Is this a "less than" objective (e.g., latency < 10ms)?
    pub less_than: bool,
}

impl SlaObjective {
    /// Creates a latency objective (less than target)
    pub fn latency_ms(target_ms: f64, violation_threshold: f64) -> Self {
        Self {
            metric: SlaMetric::Latency,
            target: target_ms,
            violation_threshold,
            less_than: true,
        }
    }

    /// Creates a throughput objective (greater than target)
    pub fn throughput_rps(target_rps: f64, violation_threshold: f64) -> Self {
        Self {
            metric: SlaMetric::Throughput,
            target: target_rps,
            violation_threshold,
            less_than: false,
        }
    }

    /// Creates an availability objective (greater than target percentage)
    pub fn availability_percent(target_percent: f64, violation_threshold: f64) -> Self {
        Self {
            metric: SlaMetric::Availability,
            target: target_percent,
            violation_threshold,
            less_than: false,
        }
    }

    /// Checks if a value violates the SLA
    pub fn is_violated(&self, actual: f64) -> bool {
        if self.less_than {
            // For "less than" objectives (latency), violation is actual > threshold
            let threshold = self.target * (1.0 + self.violation_threshold);
            actual > threshold
        } else {
            // For "greater than" objectives (throughput, availability)
            let threshold = self.target * (1.0 - self.violation_threshold);
            actual < threshold
        }
    }

    /// Returns the violation severity (0.0 = no violation, 1.0 = severe)
    pub fn violation_severity(&self, actual: f64) -> f64 {
        if !self.is_violated(actual) {
            return 0.0;
        }

        if self.less_than {
            // How much over the threshold?
            let threshold = self.target * (1.0 + self.violation_threshold);
            ((actual - threshold) / threshold).max(0.0).min(1.0)
        } else {
            // How much under the threshold?
            let threshold = self.target * (1.0 - self.violation_threshold);
            ((threshold - actual) / threshold).max(0.0).min(1.0)
        }
    }
}

/// SLA violation event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaViolation {
    /// Workload ID (if specific to a workload)
    pub workload_id: Option<WorkloadId>,
    /// Objective that was violated
    pub objective: SlaObjective,
    /// Actual value measured
    pub actual_value: f64,
    /// Violation severity
    pub severity: f64,
    /// Timestamp (not serializable)
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
    /// Duration of violation in milliseconds
    pub duration_ms: Option<u64>,
}

impl SlaViolation {
    /// Creates a new SLA violation
    pub fn new(
        workload_id: Option<WorkloadId>,
        objective: SlaObjective,
        actual_value: f64,
    ) -> Self {
        let severity = objective.violation_severity(actual_value);

        Self {
            workload_id,
            objective,
            actual_value,
            severity,
            timestamp: Instant::now(),
            duration_ms: None,
        }
    }
}

/// SLA contract for a workload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaContract {
    /// Workload ID
    pub workload_id: WorkloadId,
    /// SLA objectives
    pub objectives: Vec<SlaObjective>,
    /// Contract start time (not serializable)
    #[serde(skip, default = "Instant::now")]
    pub start_time: Instant,
    /// Contract duration in milliseconds
    pub duration_ms: Option<u64>,
}

impl SlaContract {
    /// Creates a new SLA contract
    pub fn new(workload_id: WorkloadId, objectives: Vec<SlaObjective>) -> Self {
        Self {
            workload_id,
            objectives,
            start_time: Instant::now(),
            duration_ms: None,
        }
    }

    /// Creates a contract with a specific duration
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration_ms = Some(duration.as_millis() as u64);
        self
    }

    /// Checks if the contract is still active
    pub fn is_active(&self) -> bool {
        if let Some(duration_ms) = self.duration_ms {
            self.start_time.elapsed().as_millis() < duration_ms as u128
        } else {
            true
        }
    }

    /// Checks all objectives and returns violations
    pub fn check_violations(&self, measurements: &HashMap<SlaMetric, f64>) -> Vec<SlaViolation> {
        let mut violations = Vec::new();

        for objective in &self.objectives {
            if let Some(&actual) = measurements.get(&objective.metric) {
                if objective.is_violated(actual) {
                    violations.push(SlaViolation::new(
                        Some(self.workload_id),
                        objective.clone(),
                        actual,
                    ));
                }
            }
        }

        violations
    }
}

/// SLA monitor for tracking violations
#[derive(Debug)]
pub struct SlaMonitor {
    /// Active contracts by workload ID
    contracts: HashMap<WorkloadId, SlaContract>,
    /// Recent violations
    violations: Vec<SlaViolation>,
    /// Maximum violation history size
    max_violations: usize,
    /// Violation count by workload
    violation_counts: HashMap<WorkloadId, usize>,
}

impl SlaMonitor {
    /// Creates a new SLA monitor
    pub fn new(max_violations: usize) -> Self {
        Self {
            contracts: HashMap::new(),
            violations: Vec::new(),
            max_violations,
            violation_counts: HashMap::new(),
        }
    }

    /// Registers a new SLA contract
    pub fn register_contract(&mut self, contract: SlaContract) {
        let workload_id = contract.workload_id;
        self.contracts.insert(workload_id, contract);
        self.violation_counts.insert(workload_id, 0);
    }

    /// Removes a contract
    pub fn remove_contract(&mut self, workload_id: WorkloadId) -> Option<SlaContract> {
        self.violation_counts.remove(&workload_id);
        self.contracts.remove(&workload_id)
    }

    /// Checks a workload's measurements against its SLA
    pub fn check_workload(
        &mut self,
        workload_id: WorkloadId,
        measurements: &HashMap<SlaMetric, f64>,
    ) -> Vec<SlaViolation> {
        if let Some(contract) = self.contracts.get(&workload_id) {
            if !contract.is_active() {
                return Vec::new();
            }

            let violations = contract.check_violations(measurements);

            if !violations.is_empty() {
                *self.violation_counts.entry(workload_id).or_insert(0) += violations.len();

                for violation in &violations {
                    self.record_violation(violation.clone());
                }
            }

            violations
        } else {
            Vec::new()
        }
    }

    /// Records a violation
    fn record_violation(&mut self, violation: SlaViolation) {
        self.violations.push(violation);

        // Trim history if needed
        if self.violations.len() > self.max_violations {
            self.violations.drain(0..self.violations.len() - self.max_violations);
        }
    }

    /// Returns all recent violations
    pub fn recent_violations(&self) -> &[SlaViolation] {
        &self.violations
    }

    /// Returns violations for a specific workload
    pub fn violations_for_workload(&self, workload_id: WorkloadId) -> Vec<&SlaViolation> {
        self.violations
            .iter()
            .filter(|v| v.workload_id == Some(workload_id))
            .collect()
    }

    /// Returns the total violation count for a workload
    pub fn violation_count(&self, workload_id: WorkloadId) -> usize {
        *self.violation_counts.get(&workload_id).unwrap_or(&0)
    }

    /// Returns workloads with active SLA violations
    pub fn workloads_in_violation(&self) -> Vec<WorkloadId> {
        // Recent violations (within last minute)
        let cutoff = Instant::now() - Duration::from_secs(60);

        let mut violating: HashMap<WorkloadId, bool> = HashMap::new();

        for violation in self.violations.iter().rev() {
            if violation.timestamp < cutoff {
                break;
            }

            if let Some(wid) = violation.workload_id {
                violating.insert(wid, true);
            }
        }

        violating.keys().copied().collect()
    }

    /// Cleans up expired contracts
    pub fn cleanup_expired_contracts(&mut self) {
        let expired: Vec<WorkloadId> = self
            .contracts
            .iter()
            .filter(|(_, contract)| !contract.is_active())
            .map(|(id, _)| *id)
            .collect();

        for workload_id in expired {
            self.remove_contract(workload_id);
        }
    }

    /// Returns the number of active contracts
    pub fn active_contract_count(&self) -> usize {
        self.contracts.len()
    }
}

impl Default for SlaMonitor {
    fn default() -> Self {
        Self::new(1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sla_objective_latency() {
        let obj = SlaObjective::latency_ms(10.0, 0.2); // 10ms target, 20% tolerance

        // Within bounds
        assert!(!obj.is_violated(9.0));
        assert!(!obj.is_violated(10.0));
        assert!(!obj.is_violated(11.0));

        // Violation: > 12ms (10ms * 1.2)
        assert!(obj.is_violated(13.0));
        assert_eq!(obj.violation_severity(9.0), 0.0);
        assert!(obj.violation_severity(13.0) > 0.0);
    }

    #[test]
    fn test_sla_objective_throughput() {
        let obj = SlaObjective::throughput_rps(100.0, 0.1); // 100 rps, 10% tolerance

        // Within bounds
        assert!(!obj.is_violated(100.0));
        assert!(!obj.is_violated(95.0));

        // Violation: < 90 rps (100 * 0.9)
        assert!(obj.is_violated(85.0));
        assert_eq!(obj.violation_severity(100.0), 0.0);
        assert!(obj.violation_severity(85.0) > 0.0);
    }

    #[test]
    fn test_sla_contract() {
        let objectives = vec![
            SlaObjective::latency_ms(10.0, 0.2),
            SlaObjective::throughput_rps(100.0, 0.1),
        ];

        let contract = SlaContract::new(WorkloadId::new(1), objectives);

        let mut measurements = HashMap::new();
        measurements.insert(SlaMetric::Latency, 9.0);
        measurements.insert(SlaMetric::Throughput, 105.0);

        // No violations
        let violations = contract.check_violations(&measurements);
        assert_eq!(violations.len(), 0);

        // Add violation
        measurements.insert(SlaMetric::Latency, 15.0);
        let violations = contract.check_violations(&measurements);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].objective.metric, SlaMetric::Latency);
    }

    #[test]
    fn test_sla_monitor() {
        let mut monitor = SlaMonitor::new(100);

        let contract = SlaContract::new(
            WorkloadId::new(1),
            vec![SlaObjective::latency_ms(10.0, 0.2)],
        );

        monitor.register_contract(contract);

        assert_eq!(monitor.active_contract_count(), 1);

        let mut measurements = HashMap::new();
        measurements.insert(SlaMetric::Latency, 15.0);

        let violations = monitor.check_workload(WorkloadId::new(1), &measurements);
        assert_eq!(violations.len(), 1);
        assert_eq!(monitor.violation_count(WorkloadId::new(1)), 1);
        assert_eq!(monitor.recent_violations().len(), 1);
    }

    #[test]
    fn test_contract_expiry() {
        let contract = SlaContract::new(WorkloadId::new(1), vec![])
            .with_duration(Duration::from_millis(100));

        assert!(contract.is_active());

        std::thread::sleep(Duration::from_millis(150));
        assert!(!contract.is_active());
    }

    #[test]
    fn test_workloads_in_violation() {
        let mut monitor = SlaMonitor::new(100);

        let contract = SlaContract::new(
            WorkloadId::new(1),
            vec![SlaObjective::latency_ms(10.0, 0.2)],
        );
        monitor.register_contract(contract);

        let mut measurements = HashMap::new();
        measurements.insert(SlaMetric::Latency, 15.0);

        monitor.check_workload(WorkloadId::new(1), &measurements);

        let violating = monitor.workloads_in_violation();
        assert_eq!(violating.len(), 1);
        assert!(violating.contains(&WorkloadId::new(1)));
    }

    #[test]
    fn test_cleanup_expired() {
        let mut monitor = SlaMonitor::new(100);

        let contract = SlaContract::new(WorkloadId::new(1), vec![])
            .with_duration(Duration::from_millis(50));

        monitor.register_contract(contract);
        assert_eq!(monitor.active_contract_count(), 1);

        std::thread::sleep(Duration::from_millis(100));
        monitor.cleanup_expired_contracts();

        assert_eq!(monitor.active_contract_count(), 0);
    }
}
