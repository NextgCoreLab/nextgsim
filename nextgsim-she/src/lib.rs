//! Service Hosting Environment (SHE) for 6G Networks
//!
//! This crate is a **research model** for distributed AI/ML compute placement,
//! inspired by the tiered-edge concepts of 3GPP TS 23.558 (EDGEAPP); it does
//! **not** implement TS 23.558. It provides a three-tier (LocalEdge/RegionalEdge/
//! CoreCloud) compute-placement scheduler for AI/ML workloads.
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
//! # Design references (non-conformant)
//!
//! Conceptual references only:
//! - 3GPP TS 23.558: Architecture for enabling Edge Applications
//! - 3GPP TS 29.558: Application layer support for Edge Computing
//!
//! No EDGEAPP entity (ECS/EES/EAS), reference point (EDGE-1..10) or SBI
//! (T8/N33) is implemented; this is a 6G research prototype aligned at the
//! concept level only. TR 22.870 is a Stage-1 study with no Stage-3 wire spec.
//!
//! # TR 22.870 mapping
//!
//! Prototypes the computing-service use cases of TR 22.870 clause 12 (e.g.
//! §12.1 computing service for XR gaming acceleration, §12.2 computing service
//! enabling personal AI Agent). TR 22.870 is a Stage-1 study; these are
//! potential requirements with no Stage-3 wire spec — this crate is a research
//! prototype, not a conformant implementation.
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

pub mod autoscale;
pub mod error;
pub mod messages;
pub mod resource;
pub mod scheduler;
pub mod security;
pub mod sla;
pub mod task;
pub mod tier;
pub mod workload;

// Re-export main types
pub use autoscale::{AutoScaleConfig, AutoScaler, ScalingAction, ScalingDecision, ScalingPolicy};
pub use error::SheError;
pub use messages::{SheMessage, SheResponse};
pub use resource::{AcceleratorType, ResourceCapacity, ResourceUsage};
pub use scheduler::{PlacementDecision, SchedulingPolicy, WorkloadScheduler};
pub use security::{
    AttestationEvidence, AttestationResult, SecurityContext, SecurityManager, TeeType,
};
pub use sla::{SlaContract, SlaMetric, SlaMonitor, SlaObjective, SlaViolation};
pub use task::SheTask;
pub use tier::{ComputeCapability, ComputeNode, ComputeTier, TierManager};
pub use workload::{Workload, WorkloadId, WorkloadRequirements, WorkloadState};
