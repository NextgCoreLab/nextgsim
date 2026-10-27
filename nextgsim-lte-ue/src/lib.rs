//! nextgsim LTE UE (nextgsim #186).
//!
//! An E-UTRA UE over RLS: TS 36.304 cell selection, TS 36.331 system
//! information acquisition and RRC connection establishment/release, and SRB1
//! PDCP (TS 36.323), relaying NAS to and from a pluggable [`nas::NasLayer`].
//! With a `subscriber` configured, the NAS is `nextgsim-eps-nas` (TS 24.301
//! attach, EPS AKA, NAS security) and RRC runs AS security on SRB1 (TS 33.401
//! §7.2.4.5) and the default bearer's RRCConnectionReconfiguration. See
//! `specs/lte-rrc-connection-and-nas-relay.md` and
//! `specs/eps-nas-attach-and-as-security.md` in the nextg root repo.

pub mod config;
pub mod control;
pub mod nas;
pub mod rrc;
pub mod task;
pub mod uplink_buffer;
