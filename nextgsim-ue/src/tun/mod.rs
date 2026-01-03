//! TUN interface module for UE user plane data
//!
//! This module provides TUN interface management for the UE, enabling
//! IP packet handling for PDU sessions. When a PDU session is established,
//! a TUN interface is created to handle user plane data.
//!
//! # Architecture
//!
//! - `TunInterface`: Wrapper around the TUN device for async I/O
//! - `TunTask`: Task that manages TUN interface lifecycle and packet handling
//! - IP packets read from TUN are sent to the App task for GTP encapsulation
//! - Downlink IP packets from GTP are written to the TUN interface
//!
//! # Reference
//!
//! Based on UERANSIM's TUN implementation from `src/ue/tun/`

pub mod config;
mod interface;
mod packet;
mod task;

pub use config::TunConfig;
pub use interface::{TunError, TunInterface};
pub use packet::{IpPacket, IpVersion};
pub use task::{TunTask, TunTaskConfig};
