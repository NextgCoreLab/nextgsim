//! Fleet Manager for Ambient IoT Devices (Rel-18, TS 22.369)
//!
//! Implements fleet-level management for ambient IoT devices:
//! - Device group management (create/join/leave groups)
//! - Bulk command dispatch (wake, read, configure)
//! - Energy harvesting status tracking per device
//! - Fleet-level analytics (response rates, energy levels, coverage)

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Ambient IoT device type (based on energy source).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmbientDeviceType {
    /// Type A: Energy harvesting, no battery (backscatter only)
    TypeA,
    /// Type B: Assisted energy harvesting with small capacitor
    TypeB,
    /// Type C: Battery-assisted with active Tx
    TypeC,
}

/// Energy harvesting status for an ambient IoT device.
#[derive(Debug, Clone, Copy)]
pub struct EnergyHarvestingStatus {
    /// Current energy level (0.0 - 1.0, where 1.0 = fully charged)
    pub energy_level: f64,
    /// Energy harvesting rate in microwatts
    pub harvesting_rate_uw: f64,
    /// Power consumption in microwatts
    pub consumption_uw: f64,
    /// Estimated time until depleted (seconds, None if harvesting > consumption)
    pub time_to_depletion_s: Option<u32>,
    /// Timestamp of last update
    pub timestamp_ms: u64,
}

impl EnergyHarvestingStatus {
    /// Creates a new energy harvesting status.
    pub fn new(energy_level: f64, harvesting_rate_uw: f64, consumption_uw: f64) -> Self {
        let time_to_depletion_s = if harvesting_rate_uw < consumption_uw && energy_level > 0.0 {
            // Calculate time to deplete based on net consumption
            let net_consumption_uw = consumption_uw - harvesting_rate_uw;
            // Assume total capacity proportional to energy_level
            let total_capacity_uj = energy_level * 1000.0; // Assume 1mJ capacity at full
            Some((total_capacity_uj / net_consumption_uw) as u32)
        } else {
            None
        };

        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            energy_level: energy_level.clamp(0.0, 1.0),
            harvesting_rate_uw,
            consumption_uw,
            time_to_depletion_s,
            timestamp_ms,
        }
    }

    /// Returns true if device has sufficient energy for operation.
    pub fn is_operational(&self) -> bool {
        self.energy_level > 0.1 // Need at least 10% energy
    }

    /// Returns true if device is in critical energy state.
    pub fn is_critical(&self) -> bool {
        self.energy_level < 0.05 || self.time_to_depletion_s.map(|t| t < 60).unwrap_or(false)
    }
}

/// Device status enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceStatus {
    /// Device is active and responding
    Active,
    /// Device is sleeping (low power mode)
    Sleeping,
    /// Device is not responding (out of range or depleted)
    Inactive,
    /// Device has critical energy level
    CriticalEnergy,
}

/// Command to send to ambient IoT devices.
#[derive(Debug, Clone)]
pub enum DeviceCommand {
    /// Wake up device from sleep
    Wake,
    /// Read sensor data from device
    Read {
        /// Sensor type identifier
        sensor_type: u8,
    },
    /// Configure device parameters
    Configure {
        /// Configuration data
        config_data: Vec<u8>,
    },
    /// Put device to sleep
    Sleep,
    /// Request energy status
    GetEnergyStatus,
}

/// Response from a device command.
#[derive(Debug, Clone)]
pub struct CommandResponse {
    /// Device ID that responded
    pub device_id: u64,
    /// Command that was executed
    pub command_type: String,
    /// Response data (if any)
    pub data: Option<Vec<u8>>,
    /// Success flag
    pub success: bool,
    /// Timestamp of response
    pub timestamp_ms: u64,
}

impl CommandResponse {
    /// Creates a new command response.
    pub fn new(device_id: u64, command_type: String, success: bool) -> Self {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            device_id,
            command_type,
            data: None,
            success,
            timestamp_ms,
        }
    }

    /// Sets the response data.
    pub fn with_data(mut self, data: Vec<u8>) -> Self {
        self.data = Some(data);
        self
    }
}

/// Ambient IoT device information.
#[derive(Debug, Clone)]
pub struct AmbientDevice {
    /// Unique device identifier
    pub device_id: u64,
    /// Device type
    pub device_type: AmbientDeviceType,
    /// Current status
    pub status: DeviceStatus,
    /// Energy harvesting status
    pub energy_status: Option<EnergyHarvestingStatus>,
    /// Groups this device belongs to
    pub groups: Vec<String>,
    /// Last seen timestamp
    pub last_seen_ms: u64,
    /// Signal strength (RSRP) in dBm
    pub rsrp_dbm: i32,
    /// Number of successful responses
    pub response_count: u32,
    /// Number of failed commands
    pub failure_count: u32,
}

impl AmbientDevice {
    /// Creates a new ambient IoT device.
    pub fn new(device_id: u64, device_type: AmbientDeviceType) -> Self {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            device_id,
            device_type,
            status: DeviceStatus::Active,
            energy_status: None,
            groups: Vec::new(),
            last_seen_ms: now_ms,
            rsrp_dbm: -100,
            response_count: 0,
            failure_count: 0,
        }
    }

    /// Updates energy status.
    pub fn update_energy(&mut self, energy_status: EnergyHarvestingStatus) {
        if energy_status.is_critical() {
            self.status = DeviceStatus::CriticalEnergy;
            tracing::warn!(
                "Ambient IoT device {}: CRITICAL energy level {:.1}%",
                self.device_id,
                energy_status.energy_level * 100.0
            );
        } else if !energy_status.is_operational() {
            self.status = DeviceStatus::Inactive;
        } else if self.status == DeviceStatus::CriticalEnergy || self.status == DeviceStatus::Inactive {
            self.status = DeviceStatus::Active;
        }

        self.energy_status = Some(energy_status);
    }

    /// Marks device as seen.
    pub fn mark_seen(&mut self, rsrp_dbm: i32) {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        self.last_seen_ms = now_ms;
        self.rsrp_dbm = rsrp_dbm;

        if self.status == DeviceStatus::Inactive {
            self.status = DeviceStatus::Active;
        }
    }

    /// Records a successful command response.
    pub fn record_response(&mut self) {
        self.response_count += 1;
    }

    /// Records a failed command.
    pub fn record_failure(&mut self) {
        self.failure_count += 1;
    }

    /// Joins a device group.
    pub fn join_group(&mut self, group_name: String) {
        if !self.groups.contains(&group_name) {
            self.groups.push(group_name);
        }
    }

    /// Leaves a device group.
    pub fn leave_group(&mut self, group_name: &str) {
        self.groups.retain(|g| g != group_name);
    }

    /// Returns the response rate (0.0 - 1.0).
    pub fn response_rate(&self) -> f64 {
        let total = self.response_count + self.failure_count;
        if total == 0 {
            0.0
        } else {
            self.response_count as f64 / total as f64
        }
    }

    /// Returns true if device is in the specified group.
    pub fn in_group(&self, group_name: &str) -> bool {
        self.groups.iter().any(|g| g == group_name)
    }
}

/// Device group for fleet organization.
#[derive(Debug, Clone)]
pub struct DeviceGroup {
    /// Group name
    pub name: String,
    /// Device IDs in this group
    pub device_ids: Vec<u64>,
    /// Group creation timestamp
    pub created_ms: u64,
}

impl DeviceGroup {
    /// Creates a new device group.
    pub fn new(name: String) -> Self {
        let created_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            name,
            device_ids: Vec::new(),
            created_ms,
        }
    }

    /// Adds a device to the group.
    pub fn add_device(&mut self, device_id: u64) {
        if !self.device_ids.contains(&device_id) {
            self.device_ids.push(device_id);
        }
    }

    /// Removes a device from the group.
    pub fn remove_device(&mut self, device_id: u64) {
        self.device_ids.retain(|&id| id != device_id);
    }

    /// Returns true if group contains the device.
    pub fn contains(&self, device_id: u64) -> bool {
        self.device_ids.contains(&device_id)
    }
}

/// Fleet-level analytics for ambient IoT devices.
#[derive(Debug, Clone)]
pub struct FleetAnalytics {
    /// Total number of devices in fleet
    pub total_devices: usize,
    /// Number of active devices
    pub active_devices: usize,
    /// Number of sleeping devices
    pub sleeping_devices: usize,
    /// Number of inactive devices
    pub inactive_devices: usize,
    /// Number of devices with critical energy
    pub critical_energy_devices: usize,
    /// Average energy level (0.0 - 1.0)
    pub avg_energy_level: f64,
    /// Average response rate (0.0 - 1.0)
    pub avg_response_rate: f64,
    /// Coverage area in square meters (based on device positions)
    pub coverage_area_m2: f64,
    /// Timestamp of analytics generation
    pub timestamp_ms: u64,
}

impl FleetAnalytics {
    /// Calculates analytics from a set of devices.
    pub fn from_devices(devices: &HashMap<u64, AmbientDevice>) -> Self {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let total_devices = devices.len();
        let mut active_devices = 0;
        let mut sleeping_devices = 0;
        let mut inactive_devices = 0;
        let mut critical_energy_devices = 0;
        let mut total_energy = 0.0;
        let mut total_response_rate = 0.0;

        for device in devices.values() {
            match device.status {
                DeviceStatus::Active => active_devices += 1,
                DeviceStatus::Sleeping => sleeping_devices += 1,
                DeviceStatus::Inactive => inactive_devices += 1,
                DeviceStatus::CriticalEnergy => critical_energy_devices += 1,
            }

            if let Some(energy) = &device.energy_status {
                total_energy += energy.energy_level;
            }

            total_response_rate += device.response_rate();
        }

        let avg_energy_level = if total_devices > 0 {
            total_energy / total_devices as f64
        } else {
            0.0
        };

        let avg_response_rate = if total_devices > 0 {
            total_response_rate / total_devices as f64
        } else {
            0.0
        };

        // Coverage area estimate (simplified)
        let coverage_area_m2 = (total_devices as f64) * 100.0; // Assume 100 m² per device

        Self {
            total_devices,
            active_devices,
            sleeping_devices,
            inactive_devices,
            critical_energy_devices,
            avg_energy_level,
            avg_response_rate,
            coverage_area_m2,
            timestamp_ms: now_ms,
        }
    }
}

/// Fleet manager for coordinating multiple ambient IoT devices.
#[derive(Debug, Clone)]
pub struct FleetManager {
    /// All devices in the fleet
    pub devices: HashMap<u64, AmbientDevice>,
    /// Device groups
    pub groups: HashMap<String, DeviceGroup>,
    /// Fleet name/identifier
    pub fleet_name: String,
}

impl FleetManager {
    /// Creates a new fleet manager.
    pub fn new(fleet_name: String) -> Self {
        Self {
            devices: HashMap::new(),
            groups: HashMap::new(),
            fleet_name,
        }
    }

    /// Adds a device to the fleet.
    pub fn add_device(&mut self, device: AmbientDevice) {
        tracing::info!(
            "Fleet '{}': Adding device {} (type={:?})",
            self.fleet_name,
            device.device_id,
            device.device_type
        );
        self.devices.insert(device.device_id, device);
    }

    /// Removes a device from the fleet.
    pub fn remove_device(&mut self, device_id: u64) {
        if self.devices.remove(&device_id).is_some() {
            tracing::info!("Fleet '{}': Removed device {}", self.fleet_name, device_id);

            // Remove from all groups
            for group in self.groups.values_mut() {
                group.remove_device(device_id);
            }
        }
    }

    /// Creates a new device group.
    pub fn create_group(&mut self, group_name: String) {
        if !self.groups.contains_key(&group_name) {
            tracing::info!("Fleet '{}': Created group '{}'", self.fleet_name, group_name);
            self.groups.insert(group_name.clone(), DeviceGroup::new(group_name));
        }
    }

    /// Adds a device to a group.
    pub fn add_to_group(&mut self, device_id: u64, group_name: &str) {
        if let Some(group) = self.groups.get_mut(group_name) {
            group.add_device(device_id);

            if let Some(device) = self.devices.get_mut(&device_id) {
                device.join_group(group_name.to_string());
            }

            tracing::debug!(
                "Fleet '{}': Added device {} to group '{}'",
                self.fleet_name,
                device_id,
                group_name
            );
        }
    }

    /// Removes a device from a group.
    pub fn remove_from_group(&mut self, device_id: u64, group_name: &str) {
        if let Some(group) = self.groups.get_mut(group_name) {
            group.remove_device(device_id);

            if let Some(device) = self.devices.get_mut(&device_id) {
                device.leave_group(group_name);
            }
        }
    }

    /// Sends a command to all devices in a group.
    pub fn dispatch_to_group(&self, group_name: &str, command: &DeviceCommand) -> Vec<u64> {
        let mut dispatched = Vec::new();

        if let Some(group) = self.groups.get(group_name) {
            for &device_id in &group.device_ids {
                if let Some(device) = self.devices.get(&device_id) {
                    if device.status == DeviceStatus::Active || device.status == DeviceStatus::CriticalEnergy {
                        tracing::debug!(
                            "Fleet '{}': Dispatching {:?} to device {} in group '{}'",
                            self.fleet_name,
                            command,
                            device_id,
                            group_name
                        );
                        dispatched.push(device_id);
                    }
                }
            }
        }

        tracing::info!(
            "Fleet '{}': Dispatched command to {} devices in group '{}'",
            self.fleet_name,
            dispatched.len(),
            group_name
        );

        dispatched
    }

    /// Sends a command to all devices in the fleet.
    pub fn dispatch_to_all(&self, command: &DeviceCommand) -> Vec<u64> {
        let mut dispatched = Vec::new();

        for (&device_id, device) in &self.devices {
            if device.status == DeviceStatus::Active || device.status == DeviceStatus::CriticalEnergy {
                tracing::debug!(
                    "Fleet '{}': Dispatching {:?} to device {}",
                    self.fleet_name,
                    command,
                    device_id
                );
                dispatched.push(device_id);
            }
        }

        tracing::info!(
            "Fleet '{}': Dispatched command to {} devices",
            self.fleet_name,
            dispatched.len()
        );

        dispatched
    }

    /// Gets devices with critical energy levels.
    pub fn get_critical_energy_devices(&self) -> Vec<u64> {
        self.devices
            .values()
            .filter(|d| d.status == DeviceStatus::CriticalEnergy)
            .map(|d| d.device_id)
            .collect()
    }

    /// Gets inactive devices (not seen recently).
    pub fn get_inactive_devices(&self, timeout_ms: u64) -> Vec<u64> {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        self.devices
            .values()
            .filter(|d| now_ms - d.last_seen_ms > timeout_ms)
            .map(|d| d.device_id)
            .collect()
    }

    /// Generates fleet-level analytics.
    pub fn get_analytics(&self) -> FleetAnalytics {
        let analytics = FleetAnalytics::from_devices(&self.devices);

        tracing::info!(
            "Fleet '{}': Analytics - {}/{} active, avg energy={:.1}%, avg response={:.1}%",
            self.fleet_name,
            analytics.active_devices,
            analytics.total_devices,
            analytics.avg_energy_level * 100.0,
            analytics.avg_response_rate * 100.0
        );

        analytics
    }

    /// Updates device energy status.
    pub fn update_device_energy(&mut self, device_id: u64, energy_status: EnergyHarvestingStatus) {
        if let Some(device) = self.devices.get_mut(&device_id) {
            device.update_energy(energy_status);
        }
    }

    /// Records a command response from a device.
    pub fn record_response(&mut self, device_id: u64, success: bool) {
        if let Some(device) = self.devices.get_mut(&device_id) {
            if success {
                device.record_response();
            } else {
                device.record_failure();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_harvesting_status() {
        let status = EnergyHarvestingStatus::new(0.8, 100.0, 50.0);
        assert!(status.is_operational());
        assert!(!status.is_critical());
        assert!(status.time_to_depletion_s.is_none()); // Harvesting > consumption

        let critical = EnergyHarvestingStatus::new(0.03, 10.0, 50.0);
        assert!(!critical.is_operational());
        assert!(critical.is_critical());
    }

    #[test]
    fn test_ambient_device_lifecycle() {
        let mut device = AmbientDevice::new(123, AmbientDeviceType::TypeA);
        assert_eq!(device.status, DeviceStatus::Active);

        device.update_energy(EnergyHarvestingStatus::new(0.02, 5.0, 50.0));
        assert_eq!(device.status, DeviceStatus::CriticalEnergy);

        device.record_response();
        device.record_response();
        device.record_failure();
        assert_eq!(device.response_count, 2);
        assert_eq!(device.failure_count, 1);
        assert!((device.response_rate() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_device_group_management() {
        let mut group = DeviceGroup::new("sensors".to_string());
        assert_eq!(group.device_ids.len(), 0);

        group.add_device(1);
        group.add_device(2);
        group.add_device(3);
        assert_eq!(group.device_ids.len(), 3);
        assert!(group.contains(2));

        group.remove_device(2);
        assert_eq!(group.device_ids.len(), 2);
        assert!(!group.contains(2));
    }

    #[test]
    fn test_fleet_manager() {
        let mut fleet = FleetManager::new("test-fleet".to_string());

        let device1 = AmbientDevice::new(1, AmbientDeviceType::TypeA);
        let device2 = AmbientDevice::new(2, AmbientDeviceType::TypeB);
        fleet.add_device(device1);
        fleet.add_device(device2);

        assert_eq!(fleet.devices.len(), 2);

        fleet.create_group("sensors".to_string());
        fleet.add_to_group(1, "sensors");
        fleet.add_to_group(2, "sensors");

        let dispatched = fleet.dispatch_to_group("sensors", &DeviceCommand::Wake);
        assert_eq!(dispatched.len(), 2);
    }

    #[test]
    fn test_fleet_analytics() {
        let mut devices = HashMap::new();

        let mut device1 = AmbientDevice::new(1, AmbientDeviceType::TypeA);
        device1.update_energy(EnergyHarvestingStatus::new(0.8, 100.0, 50.0));
        device1.response_count = 8;
        device1.failure_count = 2;

        let mut device2 = AmbientDevice::new(2, AmbientDeviceType::TypeB);
        device2.update_energy(EnergyHarvestingStatus::new(0.6, 80.0, 60.0));
        device2.response_count = 9;
        device2.failure_count = 1;

        devices.insert(1, device1);
        devices.insert(2, device2);

        let analytics = FleetAnalytics::from_devices(&devices);
        assert_eq!(analytics.total_devices, 2);
        assert_eq!(analytics.active_devices, 2);
        assert!((analytics.avg_energy_level - 0.7).abs() < 0.01);
        assert!(analytics.avg_response_rate > 0.85);
    }

    #[test]
    fn test_command_response() {
        let response = CommandResponse::new(123, "Wake".to_string(), true)
            .with_data(vec![1, 2, 3, 4]);

        assert_eq!(response.device_id, 123);
        assert_eq!(response.command_type, "Wake");
        assert!(response.success);
        assert_eq!(response.data, Some(vec![1, 2, 3, 4]));
    }
}
