//! RLS (Radio Link Simulation) Module for UE
//!
//! This module implements the RLS protocol handling for the UE, including:
//! - Cell search and discovery via heartbeat messages
//! - gNB connection management
//! - RRC message transport
//! - User plane data transport
//!
//! # Cell Search
//!
//! The UE performs cell search by sending heartbeat messages to all addresses
//! in the gNB search list. When a gNB responds with a heartbeat acknowledgment,
//! the UE discovers the cell and tracks its signal strength.
//!
//! # Reference
//!
//! Based on UERANSIM's UE RLS implementation from `src/ue/rls/`.

pub mod task;

// Re-export main types
pub use task::{RlsTask, RlsTaskConfig, DEFAULT_RLS_PORT};
