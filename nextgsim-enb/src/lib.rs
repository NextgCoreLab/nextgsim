//! nextgsim LTE eNB.
//!
//! nextgsim #186. Increment 1: the S1-MME control plane up to an operational
//! S1 interface -- SCTP association (TS 36.412 §7) and S1 Setup (TS 36.413
//! §8.7.3) with the §10 error handling. Increment 2: the E-UTRA cell over RLS
//! -- system information, RRC connection establishment and release
//! (TS 36.331), SRB1 PDCP (TS 36.323) -- and the UE-associated S1AP that
//! carries the UE's NAS (TS 36.413 §8.6, §8.3). AS security, DRBs and S1-U
//! follow; see `specs/s1ap-enb-s1-setup.md` and
//! `specs/lte-rrc-connection-and-nas-relay.md` in the nextg root repo.

pub mod config;
pub mod radio;
pub mod rrc;
pub mod s1_task;
pub mod s1ap;
pub mod s1ap_ue;

pub use config::{load_and_validate_enb_config, EnbConfig};
