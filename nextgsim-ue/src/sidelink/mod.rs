//! Sidelink module for UE (Rel-17/18)
//!
//! Provides NR sidelink relay (UE-to-UE relay), sidelink discovery
//! procedures, PC5 link establishment, and sidelink positioning.

pub mod task;
pub mod pc5;

pub use task::SidelinkTask;
pub use pc5::{
    Pc5RrcConnection, Pc5RrcState, Pc5RadioBearerConfig,
    Pc5Discovery, Pc5DiscoveryMode, Pc5ResourceMode,
    Pc5DiscoveredPeer, Pc5HarqFeedback,
};
