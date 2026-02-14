//! Ambient IoT Fleet Management (Rel-18, TS 22.369)
//!
//! Provides fleet management capabilities for coordinating multiple ambient IoT devices.

pub mod fleet;

pub use fleet::{
    FleetManager, DeviceGroup, AmbientDevice, DeviceCommand, DeviceStatus,
    EnergyHarvestingStatus, FleetAnalytics, CommandResponse,
};
