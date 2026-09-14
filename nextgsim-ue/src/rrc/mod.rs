//! RRC (Radio Resource Control) Module for UE
//!
//! This module implements the RRC protocol handling for the UE, including:
//! - RRC state machine (Idle, Connected, Inactive)
//! - RRC connection management
//! - Cell selection/reselection per 3GPP TS 38.304
//! - Measurement and handover support, including conditional handover
//!   (TS 38.331 §5.3.5.13)
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
//! # RRC_INACTIVE (3GPP TS 38.331 §5.3.8.3, §5.3.13; TS 38.304 §5.5)
//!
//! The Inactive state above is reachable in production since issue #38, and what
//! makes it reachable is worth naming because the state machine alone does not say it:
//!
//! - An `RRCRelease` **carrying a `suspendConfig`** drives Connected → Inactive; the
//!   same message without one drives Connected → Idle. [`inactive::InactiveContext`]
//!   holds the I-RNTI, the RAN Notification Area and the `t380` deadline.
//! - The UE leaves Inactive on its own initiative, by sending an
//!   `RRCResumeRequest1` whose `resumeMAC-I` it derives from the stored AS security
//!   context. Three production triggers reach it: an MO NAS message, a PCCH page, and
//!   an RNAU.
//! - An RNAU is performed on `t380` expiry **or** on reselecting a cell outside the
//!   RAN Notification Area, both with cause `rna-Update`
//!   ([`inactive::RnauTrigger`]).
//!
//! What Inactive does **not** preserve is the radio bearer configuration: a resumed UE
//! is given a fresh one with `fullConfig` set, so its DRBs need re-establishing. See
//! `nextgsim_rrc::procedures::rrc_resume::fresh_rrc_resume_params`.
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
pub mod conditional_handover;
pub mod handover;
pub mod inactive;
pub mod measurement;
pub mod redcap;
pub mod reestablishment;
pub mod resume;
pub mod security;
pub mod state;
pub mod task;
pub mod uav;

// Re-export main types
pub use cell_selection::{
    ActiveCellInfo, CellCategory, CellChangeEvent, CellDescription, CellReselectionParams,
    CellSelectionReport, CellSelector, MibInfo, Plmn, Sib1Info, Tai, CELL_LOST_THRESHOLD_DBM,
    DEFAULT_Q_HYST_DB,
};
pub use conditional_handover::{
    candidate_cell_id, handover_command_for, CondReconfigId, CondReconfigStore, TriggeredCandidate,
};
pub use handover::{
    build_reconfiguration_complete, parse_handover_command, HandoverCommand, HandoverFailureCause,
    HandoverManager, HandoverState, TargetCellInfo,
};
pub use measurement::{
    CellMeasResult, EutraCellKey, EutraMeasResult, MeasConfig, MeasEventType, MeasQuantity,
    MeasurementManager, MeasurementReport, ReportTriggerConfig, ReportTriggerType, TriggeringCell,
};
pub use redcap::{RedCapMeasurementRestrictions, RedCapMode, RedCapRelease, ReducedMimoMode};
pub use reestablishment::{
    ReestablishmentCompleteParams, ReestablishmentError, ReestablishmentProcedure,
    ReestablishmentRequestParams, ReestablishmentState, ReestablishmentTrigger, RlfDetector,
    N310_DEFAULT, N311_DEFAULT, T301_DEFAULT_MS, T311_DEFAULT_MS,
};
pub use resume::{
    ResumeCause, ResumeCompleteParams, ResumeError, ResumeIdentity, ResumeProcedure,
    ResumeProcedureState, ResumeRequestParams, T319_DEFAULT_MS,
};
pub use security::{
    as_security_enabled, compute_resume_mac_i, compute_short_mac_i, AsSecurityContext,
    AsSecurityError, CipheringAlgorithm, IntegrityAlgorithm, ShortMacError, DIRECTION_DOWNLINK,
    DIRECTION_UPLINK, SRB1_BEARER,
};
pub use state::{RrcState, RrcStateError, RrcStateMachine, RrcStateTransition};
pub use task::{RrcTask, Srb1Config};
pub use uav::{
    C2LinkQuality, FlightPathConfig, FlightWaypoint, GeoPosition, RemoteIdBroadcast,
    UavAuthorizationState, UavIdentity, UavRegistrationContext,
};
