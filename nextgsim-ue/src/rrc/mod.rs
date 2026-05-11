//! RRC (Radio Resource Control) Module for UE
//!
//! This module implements the RRC protocol handling for the UE, including:
//! - RRC state machine (Idle, Connected, Inactive)
//! - RRC connection management
//! - Cell selection/reselection per 3GPP TS 38.304
//! - Measurement and handover support
//!
//! # RRC State Machine (3GPP TS 38.331)
//!
//! ```text
//!                    ┌──────────┐
//!                    │   Idle   │◄────────────────────┐
//!                    └────┬─────┘                     │
//!                         │ RRC Setup                 │
//!                         ▼                           │
//!                    ┌──────────┐                     │
//!              ┌─────│Connected │─────┐               │
//!              │     └──────────┘     │               │
//!              │ RRC Suspend          │ RRC Release   │
//!              ▼                      └───────────────┘
//!         ┌──────────┐
//!         │ Inactive │
//!         └────┬─────┘
//!              │ RRC Resume / Release
//!              ▼
//!         ┌──────────┐
//!         │Connected │ or │Idle│
//!         └──────────┘
//! ```
//!
//! # Cell Selection (3GPP TS 38.304)
//!
//! Cell selection is performed in Idle and Inactive states:
//! - **Suitable cell**: Belongs to selected PLMN, not barred/reserved, TAI not forbidden
//! - **Acceptable cell**: Not barred/reserved, TAI not forbidden (any PLMN)
//!
//! # Reference
//!
//! Based on UERANSIM's UE RRC implementation from `src/ue/rrc/`.

pub mod cell_selection;
pub mod handover;
pub mod measurement;
pub mod redcap;
pub mod reestablishment;
pub mod resume;
pub mod state;
pub mod task;
pub mod uav;

// Re-export main types
pub use cell_selection::{
    ActiveCellInfo, CellCategory, CellChangeEvent, CellDescription, CellReselectionParams,
    CellSelectionReport, CellSelector, MibInfo, Plmn, Sib1Info, Tai, CELL_LOST_THRESHOLD_DBM,
    DEFAULT_Q_HYST_DB,
};
pub use handover::{
    build_reconfiguration_complete, parse_handover_command, HandoverCommand, HandoverFailureCause,
    HandoverManager, HandoverState, TargetCellInfo,
};
pub use measurement::{
    CellMeasResult, MeasConfig, MeasEventType, MeasQuantity, MeasurementManager, MeasurementReport,
    ReportTriggerConfig, ReportTriggerType,
};
pub use redcap::{RedCapMeasurementRestrictions, RedCapMode, RedCapRelease, ReducedMimoMode};
pub use reestablishment::{
    ReestablishmentCompleteParams, ReestablishmentError, ReestablishmentProcedure,
    ReestablishmentRequestParams, ReestablishmentState, ReestablishmentTrigger, RlfDetector,
    N310_DEFAULT, N311_DEFAULT, T301_DEFAULT_MS, T311_DEFAULT_MS,
};
pub use resume::{
    ResumeCause, ResumeCompleteParams, ResumeError, ResumeProcedure, ResumeProcedureState,
    ResumeRequestParams, T319_DEFAULT_MS,
};
pub use state::{RrcState, RrcStateError, RrcStateMachine, RrcStateTransition};
pub use task::RrcTask;
pub use uav::{
    C2LinkQuality, FlightPathConfig, FlightWaypoint, GeoPosition, RemoteIdBroadcast,
    UavAuthorizationState, UavIdentity, UavRegistrationContext,
};
