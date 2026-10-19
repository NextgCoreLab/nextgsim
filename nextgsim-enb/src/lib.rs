//! nextgsim LTE eNB.
//!
//! nextgsim #186, increment 1: the S1-MME control plane up to an operational
//! S1 interface -- SCTP association (TS 36.412 §7) and S1 Setup (TS 36.413
//! §8.7.3) with the §10 error handling. The radio side (LTE RRC over RLS),
//! UE-associated S1AP and S1-U follow in later increments; see
//! `specs/s1ap-enb-s1-setup.md` in the nextg root repo.

pub mod config;
pub mod s1_task;
pub mod s1ap;

pub use config::{load_and_validate_enb_config, EnbConfig};
