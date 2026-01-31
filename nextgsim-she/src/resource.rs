//! Resource capacity and usage tracking for SHE

use serde::{Deserialize, Serialize};

/// Resource capacity specification
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct ResourceCapacity {
    /// Compute capacity in FLOPS
    pub compute_flops: u64,
    /// Memory capacity in bytes
    pub memory_bytes: u64,
    /// Number of GPUs
    pub gpu_count: u32,
}

impl ResourceCapacity {
    /// Creates a new resource capacity specification
    pub fn new(compute_flops: u64, memory_bytes: u64, gpu_count: u32) -> Self {
        Self {
            compute_flops,
            memory_bytes,
            gpu_count,
        }
    }

    /// Creates a capacity with compute FLOPS in TFLOPS
    pub fn with_tflops(tflops: u64) -> Self {
        Self {
            compute_flops: tflops * 1_000_000_000_000,
            ..Default::default()
        }
    }

    /// Sets the memory in GB
    pub fn with_memory_gb(mut self, gb: u64) -> Self {
        self.memory_bytes = gb * 1024 * 1024 * 1024;
        self
    }

    /// Sets the GPU count
    pub fn with_gpus(mut self, count: u32) -> Self {
        self.gpu_count = count;
        self
    }

    /// Returns compute in TFLOPS
    pub fn compute_tflops(&self) -> f64 {
        self.compute_flops as f64 / 1_000_000_000_000.0
    }

    /// Returns memory in GB
    pub fn memory_gb(&self) -> f64 {
        self.memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

impl std::fmt::Display for ResourceCapacity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:.2} TFLOPS, {:.2} GB, {} GPUs",
            self.compute_tflops(),
            self.memory_gb(),
            self.gpu_count
        )
    }
}

/// Current resource usage
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// Compute usage in FLOPS
    pub compute_flops: u64,
    /// Memory usage in bytes
    pub memory_bytes: u64,
    /// Number of active workloads
    pub active_workloads: u32,
}

impl ResourceUsage {
    /// Creates a new resource usage tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns compute usage in TFLOPS
    pub fn compute_tflops(&self) -> f64 {
        self.compute_flops as f64 / 1_000_000_000_000.0
    }

    /// Returns memory usage in GB
    pub fn memory_gb(&self) -> f64 {
        self.memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Calculates utilization against a capacity
    pub fn utilization(&self, capacity: &ResourceCapacity) -> ResourceUtilization {
        ResourceUtilization {
            compute: if capacity.compute_flops > 0 {
                self.compute_flops as f64 / capacity.compute_flops as f64
            } else {
                0.0
            },
            memory: if capacity.memory_bytes > 0 {
                self.memory_bytes as f64 / capacity.memory_bytes as f64
            } else {
                0.0
            },
        }
    }
}

impl std::fmt::Display for ResourceUsage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:.2} TFLOPS, {:.2} GB, {} workloads",
            self.compute_tflops(),
            self.memory_gb(),
            self.active_workloads
        )
    }
}

/// Resource utilization percentages
#[derive(Debug, Clone, Copy, Default)]
pub struct ResourceUtilization {
    /// Compute utilization (0.0 to 1.0)
    pub compute: f64,
    /// Memory utilization (0.0 to 1.0)
    pub memory: f64,
}

impl ResourceUtilization {
    /// Returns the maximum utilization across all resources
    pub fn max(&self) -> f64 {
        self.compute.max(self.memory)
    }

    /// Returns true if any resource is over-utilized (> 1.0)
    pub fn is_overloaded(&self) -> bool {
        self.compute > 1.0 || self.memory > 1.0
    }

    /// Returns true if utilization is within threshold
    pub fn is_within(&self, threshold: f64) -> bool {
        self.compute <= threshold && self.memory <= threshold
    }
}

impl std::fmt::Display for ResourceUtilization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "compute: {:.1}%, memory: {:.1}%",
            self.compute * 100.0,
            self.memory * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_capacity_builder() {
        let cap = ResourceCapacity::with_tflops(10)
            .with_memory_gb(64)
            .with_gpus(4);

        assert_eq!(cap.compute_flops, 10_000_000_000_000);
        assert_eq!(cap.memory_bytes, 64 * 1024 * 1024 * 1024);
        assert_eq!(cap.gpu_count, 4);
    }

    #[test]
    fn test_resource_capacity_display() {
        let cap = ResourceCapacity::with_tflops(10).with_memory_gb(64);
        let display = format!("{}", cap);
        assert!(display.contains("10.00 TFLOPS"));
        assert!(display.contains("64.00 GB"));
    }

    #[test]
    fn test_resource_usage_utilization() {
        let capacity = ResourceCapacity::with_tflops(10).with_memory_gb(64);
        let usage = ResourceUsage {
            compute_flops: 5_000_000_000_000, // 5 TFLOPS
            memory_bytes: 32 * 1024 * 1024 * 1024, // 32 GB
            active_workloads: 2,
        };

        let util = usage.utilization(&capacity);
        assert!((util.compute - 0.5).abs() < 0.01);
        assert!((util.memory - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_resource_utilization_checks() {
        let util = ResourceUtilization {
            compute: 0.8,
            memory: 0.6,
        };

        assert!(!util.is_overloaded());
        assert!(util.is_within(0.9));
        assert!(!util.is_within(0.7));
        assert!((util.max() - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_overloaded_utilization() {
        let util = ResourceUtilization {
            compute: 1.2,
            memory: 0.8,
        };

        assert!(util.is_overloaded());
    }
}
