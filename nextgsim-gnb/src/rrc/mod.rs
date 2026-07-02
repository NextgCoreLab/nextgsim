//! RRC (Radio Resource Control) Module for gNB
//!
//! This module implements the RRC protocol handling for the gNB, including:
//! - RRC connection management (Setup, Release, Reconfiguration)
//! - UE context tracking with RRC state machine
//! - RRC message handling (UL-CCCH, UL-DCCH, DL-CCCH, DL-DCCH)
//! - NAS message routing between UE (via RLS) and NGAP
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                        RRC Task                              │
//! │  ┌─────────────────┐  ┌─────────────────────────────────┐   │
//! │  │  UE Context     │  │  Connection Manager             │   │
//! │  │  Manager        │  │  - RRC Setup                    │   │
//! │  │  - RRC State    │  │  - RRC Release                  │   │
//! │  │  - C-RNTI       │  │  - RRC Reconfiguration          │   │
//! │  │  - Security     │  │  - Security Mode                │   │
//! │  └─────────────────┘  └─────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────┘
//!                    │                    │
//!                    ▼                    ▼
//!              RLS Task              NGAP Task
//!           (RRC transport)       (NAS routing)
//! ```
//!
//! # RRC State Machine
//!
//! ```text
//!                    ┌──────────┐
//!                    │   Idle   │
//!                    └────┬─────┘
//!                         │ RRC Setup
//!                         ▼
//!                    ┌──────────┐
//!              ┌─────│Connected │─────┐
//!              │     └──────────┘     │
//!              │ RRC Suspend          │ RRC Release
//!              ▼                      ▼
//!         ┌──────────┐           ┌──────────┐
//!         │ Inactive │           │   Idle   │
//!         └────┬─────┘           └──────────┘
//!              │ RRC Resume
//!              ▼
//!         ┌──────────┐
//!         │Connected │
//!         └──────────┘
//! ```
//!
//! # Message Flow
//!
//! ## RRC Setup Procedure
//! ```text
//! UE                    gNB (RRC)                 NGAP
//!  │                        │                       │
//!  │──RRCSetupRequest──────>│                       │
//!  │                        │                       │
//!  │<─────────RRCSetup──────│                       │
//!  │                        │                       │
//!  │──RRCSetupComplete─────>│                       │
//!  │     (NAS PDU)          │──InitialNasDelivery──>│
//!  │                        │                       │
//! ```
//!
//! ## Downlink NAS Delivery
//! ```text
//! NGAP                  gNB (RRC)                   UE
//!  │                        │                       │
//!  │──NasDelivery──────────>│                       │
//!  │                        │──DLInformationTransfer│
//!  │                        │     (NAS PDU)────────>│
//!  │                        │                       │
//! ```

pub mod connection;
pub mod energy_saving;
pub mod handover;
pub mod ntn_gnb;
pub mod redcap;
pub mod task;
pub mod transaction;
pub mod ue_context;

// Re-export main types
pub use connection::{
    RrcConnectionManager, RrcReleaseResult, RrcSetupCompleteResult, RrcSetupResult,
};
pub use energy_saving::{
    CellEnergyKpis, CellEnergyState, CellSleepMode, EnergySavingConfig, EnergySavingManager,
    EnergySavingPolicy,
};
pub use handover::{
    parse_measurement_report, GnbHandoverManager, HandoverCommand, HandoverConfig,
    HandoverDecision, MeasurementReport, NeighborMeasurement, PathSwitchRequest, UeHandoverState,
    XnHandoverAcknowledge, XnHandoverCause, XnHandoverRequest, XnPduSessionContext, XnUeContext,
};
pub use ntn_gnb::{
    BeamManagementMode, NtnBeamCell, NtnGnbConfig, NtnGnbManager, NtnHarqManager,
    NtnTimingAdvanceManager,
};
pub use redcap::{
    MimoRestriction, RedCapProcessor, RedCapRelease, RedCapRestrictions, RedCapRrcConfig,
    RedCapUeCapabilities,
};
pub use task::RrcTask;
pub use transaction::{
    RrcProcedure, RrcTransactionAllocator, TidVerification, C5_TYPED_DCCH_DISPATCH,
};
pub use ue_context::{RrcState, RrcUeContext, RrcUeContextManager};
