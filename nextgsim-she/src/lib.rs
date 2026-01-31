//! Service Hosting Environment (SHE) for 6G Networks
//!
//! This crate implements the Service Hosting Environment per 3GPP TS 23.558,
//! providing a three-tier distributed compute platform for AI/ML workloads.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    Service Hosting Environment (SHE)                     │
//! │                                                                          │
//! │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
//! │  │   Local Edge    │  │  Regional Edge  │  │      Core Cloud         │  │
//! │  │   (< 10ms)      │  │   (< 20ms)      │  │   (No constraint)       │  │
//! │  │                 │  │                 │  │                         │  │
//! │  │  • Inference    │  │  • Inference    │  │  • Training             │  │
//! │  │  • Small models │  │  • Fine-tuning  │  │  • Large models         │  │
//! │  │  • UE-specific  │  │  • Cell-level   │  │  • Global aggregation   │  │
//! │  └─────────────────┘  └─────────────────┘  └─────────────────────────┘  │
//! │           │                    │                       │                 │
//! │           └────────────────────┴───────────────────────┘                 │
//! │                              Scheduler                                   │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Three-Tier Compute Model
//!
//! | Tier           | Latency | Capabilities | Use Cases |
//! |----------------|---------|--------------|-----------|
//! | Local Edge     | <10ms   | Inference    | Real-time prediction, UE-specific models |
//! | Regional Edge  | <20ms   | Fine-tuning  | Cell handover, load prediction |
//! | Core Cloud     | N/A     | Full training| Global model training, aggregation |
//!
//! # 3GPP Compliance
//!
//! This implementation aligns with:
//! - 3GPP TS 23.558: Architecture for enabling Edge Applications
//! - 3GPP TS 29.558: Application layer support for Edge Computing
//!
//! # Example Usage
//!
//! ```ignore
//! use nextgsim_she::{SheManager, ComputeTier, WorkloadRequirements, ComputeCapability};
//!
//! // Create SHE manager
//! let config = SheConfig::default();
//! let mut she = SheManager::new(config);
//!
//! // Submit a workload
//! let requirements = WorkloadRequirements::new()
//!     .with_latency_constraint_ms(10)
//!     .with_compute_flops(1_000_000_000)
//!     .with_capability(ComputeCapability::Inference);
//!
//! let placement = she.place_workload(workload_id, requirements)?;
//! ```

pub mod error;
pub mod messages;
pub mod resource;
pub mod scheduler;
pub mod task;
pub mod tier;
pub mod workload;

// Re-export main types
pub use error::SheError;
pub use messages::{SheMessage, SheResponse};
pub use resource::{ResourceCapacity, ResourceUsage};
pub use scheduler::{PlacementDecision, WorkloadScheduler};
pub use task::SheTask;
pub use tier::{ComputeCapability, ComputeNode, ComputeTier, TierManager};
pub use workload::{Workload, WorkloadId, WorkloadRequirements, WorkloadState};
