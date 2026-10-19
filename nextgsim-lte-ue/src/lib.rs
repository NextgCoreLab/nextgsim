//! nextgsim LTE UE (nextgsim #186, increment 2).
//!
//! An E-UTRA UE over RLS: TS 36.304 cell selection, TS 36.331 system
//! information acquisition and RRC connection establishment/release, and SRB1
//! PDCP (TS 36.323), relaying NAS to and from a pluggable [`nas::NasLayer`].
//! EPS NAS (TS 24.301) and AS security are increment 3; see
//! `specs/lte-rrc-connection-and-nas-relay.md` in the nextg root repo.

pub mod config;
pub mod nas;
pub mod rrc;
pub mod task;
