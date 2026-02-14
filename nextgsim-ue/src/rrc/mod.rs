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
pub mod state;
pub mod task;
pub mod redcap;
pub mod uav;

// Re-export main types
pub use cell_selection::{
    CellSelector, CellDescription, CellChangeEvent, CellCategory,
    ActiveCellInfo, CellSelectionReport, CellReselectionParams, MibInfo, Sib1Info,
    Plmn, Tai, CELL_LOST_THRESHOLD_DBM, DEFAULT_Q_HYST_DB,
};
pub use state::{RrcState, RrcStateMachine, RrcStateTransition, RrcStateError};
pub use task::RrcTask;
pub use measurement::{
    MeasQuantity, MeasEventType, ReportTriggerType, ReportTriggerConfig,
    MeasConfig, CellMeasResult, MeasurementReport, MeasurementManager,
};
pub use handover::{
    HandoverState, HandoverFailureCause, TargetCellInfo, HandoverCommand,
    HandoverManager, parse_handover_command, build_reconfiguration_complete,
};
pub use redcap::{
    RedCapMode, ReducedMimoMode, RedCapRelease, RedCapMeasurementRestrictions,
};
pub use uav::{
    UavIdentity, UavAuthorizationState, GeoPosition, FlightWaypoint, FlightPathConfig,
    RemoteIdBroadcast, C2LinkQuality, UavRegistrationContext,
};
