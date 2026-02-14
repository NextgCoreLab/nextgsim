//! UAV (Unmanned Aerial Vehicle) Identification Protocol (Rel-18, TS 23.256)
//!
//! Implements UAV-specific RRC extensions for:
//! - UAV identity and registration with CAA (Civil Aviation Authority)
//! - Flight path reporting at configurable intervals
//! - Remote identification broadcast (Direct Remote ID per ASTM F3411)
//! - Command and control (C2) link quality monitoring

use std::time::{SystemTime, UNIX_EPOCH};

/// UAV identity information.
///
/// Based on 3GPP TS 23.256 and ASTM F3411 for remote ID.
#[derive(Debug, Clone)]
pub struct UavIdentity {
    /// UAV serial number (manufacturer assigned)
    pub serial_number: String,
    /// Civil Aviation Authority level ID (e.g., registration number)
    pub caa_level_id: Option<String>,
    /// Manufacturer name
    pub manufacturer: String,
    /// UAV model designation
    pub model: String,
}

impl UavIdentity {
    /// Creates a new UAV identity.
    pub fn new(serial_number: String, manufacturer: String, model: String) -> Self {
        Self {
            serial_number,
            caa_level_id: None,
            manufacturer,
            model,
        }
    }

    /// Sets the CAA level ID (registration number).
    pub fn with_caa_id(mut self, caa_level_id: String) -> Self {
        self.caa_level_id = Some(caa_level_id);
        self
    }
}

/// UAV authorization state for network registration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UavAuthorizationState {
    /// Not authorized for flight operations
    NotAuthorized,
    /// Authorization requested from network
    AuthorizationRequested,
    /// Authorized for flight in specified area
    Authorized,
    /// Authorization revoked (e.g., entering restricted airspace)
    Revoked,
}

/// Geographic position (WGS-84 coordinates).
#[derive(Debug, Clone, Copy)]
pub struct GeoPosition {
    /// Latitude in degrees (-90.0 to +90.0)
    pub latitude: f64,
    /// Longitude in degrees (-180.0 to +180.0)
    pub longitude: f64,
    /// Altitude in meters above sea level
    pub altitude_msl: f64,
}

impl GeoPosition {
    /// Creates a new geographic position.
    pub fn new(latitude: f64, longitude: f64, altitude_msl: f64) -> Self {
        Self {
            latitude,
            longitude,
            altitude_msl,
        }
    }

    /// Validates coordinates are within valid ranges.
    pub fn is_valid(&self) -> bool {
        self.latitude >= -90.0
            && self.latitude <= 90.0
            && self.longitude >= -180.0
            && self.longitude <= 180.0
    }
}

/// Flight path waypoint with timestamp.
#[derive(Debug, Clone)]
pub struct FlightWaypoint {
    /// Position at this waypoint
    pub position: GeoPosition,
    /// Timestamp in milliseconds since UNIX epoch
    pub timestamp_ms: u64,
    /// Ground speed in m/s
    pub ground_speed_ms: f64,
    /// Vertical speed in m/s (positive = climbing)
    pub vertical_speed_ms: f64,
}

impl FlightWaypoint {
    /// Creates a new flight waypoint with current timestamp.
    pub fn new(position: GeoPosition, ground_speed_ms: f64, vertical_speed_ms: f64) -> Self {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            position,
            timestamp_ms,
            ground_speed_ms,
            vertical_speed_ms,
        }
    }
}

/// Flight path reporting configuration.
#[derive(Debug, Clone)]
pub struct FlightPathConfig {
    /// Reporting interval in milliseconds
    pub reporting_interval_ms: u32,
    /// Maximum number of waypoints to retain
    pub max_waypoints: usize,
    /// Whether to report to network (vs. local storage only)
    pub report_to_network: bool,
}

impl Default for FlightPathConfig {
    fn default() -> Self {
        Self {
            reporting_interval_ms: 1000, // 1 Hz reporting
            max_waypoints: 100,
            report_to_network: true,
        }
    }
}

/// Direct Remote ID broadcast message (ASTM F3411).
///
/// This is broadcast over sidelink for local detection of UAVs.
#[derive(Debug, Clone)]
pub struct RemoteIdBroadcast {
    /// UAV serial number
    pub serial_number: String,
    /// Current position
    pub position: GeoPosition,
    /// Ground speed in m/s
    pub ground_speed_ms: f64,
    /// Heading in degrees (0-359, 0 = North)
    pub heading_deg: u16,
    /// Timestamp of this broadcast
    pub timestamp_ms: u64,
    /// Operator location (if available)
    pub operator_position: Option<GeoPosition>,
}

impl RemoteIdBroadcast {
    /// Creates a new remote ID broadcast message.
    pub fn new(
        serial_number: String,
        position: GeoPosition,
        ground_speed_ms: f64,
        heading_deg: u16,
    ) -> Self {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            serial_number,
            position,
            ground_speed_ms,
            heading_deg: heading_deg % 360,
            timestamp_ms,
            operator_position: None,
        }
    }

    /// Sets the operator position.
    pub fn with_operator_position(mut self, position: GeoPosition) -> Self {
        self.operator_position = Some(position);
        self
    }
}

/// Command and control (C2) link quality metrics.
#[derive(Debug, Clone, Copy)]
pub struct C2LinkQuality {
    /// Signal strength (RSRP) in dBm
    pub rsrp_dbm: i32,
    /// Signal quality (SINR) in dB
    pub sinr_db: f64,
    /// Packet loss rate (0.0 - 1.0)
    pub packet_loss_rate: f64,
    /// Round-trip latency in milliseconds
    pub latency_ms: u32,
    /// Timestamp of measurement
    pub timestamp_ms: u64,
}

impl C2LinkQuality {
    /// Creates a new C2 link quality measurement.
    pub fn new(rsrp_dbm: i32, sinr_db: f64, packet_loss_rate: f64, latency_ms: u32) -> Self {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            rsrp_dbm,
            sinr_db,
            packet_loss_rate,
            latency_ms,
            timestamp_ms,
        }
    }

    /// Returns true if C2 link quality is acceptable for safe flight.
    ///
    /// Based on typical requirements: RSRP > -110 dBm, packet loss < 10%, latency < 500ms.
    pub fn is_acceptable(&self) -> bool {
        self.rsrp_dbm > -110
            && self.packet_loss_rate < 0.1
            && self.latency_ms < 500
    }

    /// Returns true if C2 link quality is critical (fail-safe required).
    pub fn is_critical(&self) -> bool {
        self.rsrp_dbm < -120 || self.packet_loss_rate > 0.3 || self.latency_ms > 1000
    }
}

/// UAV registration context with state machine.
///
/// Manages the lifecycle of UAV registration with the network and CAA.
#[derive(Debug, Clone)]
pub struct UavRegistrationContext {
    /// UAV identity
    pub identity: UavIdentity,
    /// Current authorization state
    pub authorization_state: UavAuthorizationState,
    /// Flight path configuration
    pub flight_path_config: FlightPathConfig,
    /// Flight path history (waypoints)
    pub flight_path: Vec<FlightWaypoint>,
    /// Latest C2 link quality measurement
    pub c2_link_quality: Option<C2LinkQuality>,
    /// Time of last Remote ID broadcast (ms)
    pub last_remote_id_broadcast_ms: u64,
    /// Remote ID broadcast interval (ms)
    pub remote_id_interval_ms: u32,
}

impl UavRegistrationContext {
    /// Creates a new UAV registration context.
    pub fn new(identity: UavIdentity, flight_path_config: FlightPathConfig) -> Self {
        Self {
            identity,
            authorization_state: UavAuthorizationState::NotAuthorized,
            flight_path_config,
            flight_path: Vec::new(),
            c2_link_quality: None,
            last_remote_id_broadcast_ms: 0,
            remote_id_interval_ms: 1000, // 1 Hz broadcast per ASTM F3411
        }
    }

    /// Requests authorization from the network.
    pub fn request_authorization(&mut self) {
        tracing::info!(
            "UAV {}: Requesting authorization from network",
            self.identity.serial_number
        );
        self.authorization_state = UavAuthorizationState::AuthorizationRequested;
    }

    /// Grants authorization for flight operations.
    pub fn grant_authorization(&mut self) {
        tracing::info!(
            "UAV {}: Authorization granted",
            self.identity.serial_number
        );
        self.authorization_state = UavAuthorizationState::Authorized;
    }

    /// Revokes authorization (e.g., entering restricted airspace).
    pub fn revoke_authorization(&mut self) {
        tracing::warn!(
            "UAV {}: Authorization revoked - entering fail-safe mode",
            self.identity.serial_number
        );
        self.authorization_state = UavAuthorizationState::Revoked;
    }

    /// Returns true if UAV is authorized for flight.
    pub fn is_authorized(&self) -> bool {
        self.authorization_state == UavAuthorizationState::Authorized
    }

    /// Adds a flight path waypoint.
    pub fn add_waypoint(&mut self, waypoint: FlightWaypoint) {
        tracing::debug!(
            "UAV {}: Added waypoint at ({:.6}, {:.6}) alt={:.1}m",
            self.identity.serial_number,
            waypoint.position.latitude,
            waypoint.position.longitude,
            waypoint.position.altitude_msl
        );

        self.flight_path.push(waypoint);

        // Prune old waypoints if exceeding max
        if self.flight_path.len() > self.flight_path_config.max_waypoints {
            let excess = self.flight_path.len() - self.flight_path_config.max_waypoints;
            self.flight_path.drain(0..excess);
        }
    }

    /// Updates C2 link quality measurement.
    pub fn update_c2_link_quality(&mut self, quality: C2LinkQuality) {
        if quality.is_critical() {
            tracing::warn!(
                "UAV {}: C2 link quality CRITICAL - RSRP={}dBm, loss={:.1}%, latency={}ms",
                self.identity.serial_number,
                quality.rsrp_dbm,
                quality.packet_loss_rate * 100.0,
                quality.latency_ms
            );
        } else if !quality.is_acceptable() {
            tracing::warn!(
                "UAV {}: C2 link quality degraded - RSRP={}dBm, loss={:.1}%, latency={}ms",
                self.identity.serial_number,
                quality.rsrp_dbm,
                quality.packet_loss_rate * 100.0,
                quality.latency_ms
            );
        }

        self.c2_link_quality = Some(quality);
    }

    /// Checks if Remote ID broadcast is due.
    pub fn should_broadcast_remote_id(&self, now_ms: u64) -> bool {
        now_ms - self.last_remote_id_broadcast_ms >= self.remote_id_interval_ms as u64
    }

    /// Generates a Remote ID broadcast message from current state.
    pub fn generate_remote_id_broadcast(
        &mut self,
        current_position: GeoPosition,
        ground_speed_ms: f64,
        heading_deg: u16,
    ) -> RemoteIdBroadcast {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        self.last_remote_id_broadcast_ms = now_ms;

        RemoteIdBroadcast::new(
            self.identity.serial_number.clone(),
            current_position,
            ground_speed_ms,
            heading_deg,
        )
    }

    /// Returns the latest position from flight path.
    pub fn latest_position(&self) -> Option<&GeoPosition> {
        self.flight_path.last().map(|wp| &wp.position)
    }

    /// Returns the total flight distance in meters.
    pub fn total_flight_distance_m(&self) -> f64 {
        let mut total = 0.0;
        for i in 1..self.flight_path.len() {
            let p1 = &self.flight_path[i - 1].position;
            let p2 = &self.flight_path[i].position;
            total += haversine_distance(p1, p2);
        }
        total
    }
}

/// Calculates the haversine distance between two geographic positions.
///
/// Returns distance in meters.
fn haversine_distance(p1: &GeoPosition, p2: &GeoPosition) -> f64 {
    const EARTH_RADIUS_M: f64 = 6371000.0; // Earth radius in meters

    let lat1_rad = p1.latitude.to_radians();
    let lat2_rad = p2.latitude.to_radians();
    let delta_lat = (p2.latitude - p1.latitude).to_radians();
    let delta_lon = (p2.longitude - p1.longitude).to_radians();

    let a = (delta_lat / 2.0).sin().powi(2)
        + lat1_rad.cos() * lat2_rad.cos() * (delta_lon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

    EARTH_RADIUS_M * c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uav_identity_creation() {
        let identity = UavIdentity::new(
            "UAV-12345".to_string(),
            "DroneManufacturer".to_string(),
            "Model-X".to_string(),
        )
        .with_caa_id("FAA-N12345".to_string());

        assert_eq!(identity.serial_number, "UAV-12345");
        assert_eq!(identity.manufacturer, "DroneManufacturer");
        assert_eq!(identity.model, "Model-X");
        assert_eq!(identity.caa_level_id, Some("FAA-N12345".to_string()));
    }

    #[test]
    fn test_geo_position_validation() {
        let valid = GeoPosition::new(37.7749, -122.4194, 100.0); // San Francisco
        assert!(valid.is_valid());

        let invalid_lat = GeoPosition::new(95.0, -122.4194, 100.0);
        assert!(!invalid_lat.is_valid());

        let invalid_lon = GeoPosition::new(37.7749, -200.0, 100.0);
        assert!(!invalid_lon.is_valid());
    }

    #[test]
    fn test_c2_link_quality_assessment() {
        let good = C2LinkQuality::new(-80, 15.0, 0.01, 50);
        assert!(good.is_acceptable());
        assert!(!good.is_critical());

        let degraded = C2LinkQuality::new(-115, 5.0, 0.15, 600);
        assert!(!degraded.is_acceptable());
        assert!(!degraded.is_critical());

        let critical = C2LinkQuality::new(-125, -5.0, 0.4, 1500);
        assert!(!critical.is_acceptable());
        assert!(critical.is_critical());
    }

    #[test]
    fn test_uav_registration_state_machine() {
        let identity = UavIdentity::new(
            "UAV-TEST".to_string(),
            "TestManufacturer".to_string(),
            "TestModel".to_string(),
        );
        let config = FlightPathConfig::default();
        let mut context = UavRegistrationContext::new(identity, config);

        assert_eq!(context.authorization_state, UavAuthorizationState::NotAuthorized);
        assert!(!context.is_authorized());

        context.request_authorization();
        assert_eq!(
            context.authorization_state,
            UavAuthorizationState::AuthorizationRequested
        );

        context.grant_authorization();
        assert_eq!(context.authorization_state, UavAuthorizationState::Authorized);
        assert!(context.is_authorized());

        context.revoke_authorization();
        assert_eq!(context.authorization_state, UavAuthorizationState::Revoked);
        assert!(!context.is_authorized());
    }

    #[test]
    fn test_flight_path_waypoints() {
        let identity = UavIdentity::new(
            "UAV-TEST".to_string(),
            "TestManufacturer".to_string(),
            "TestModel".to_string(),
        );
        let config = FlightPathConfig {
            reporting_interval_ms: 1000,
            max_waypoints: 3,
            report_to_network: true,
        };
        let mut context = UavRegistrationContext::new(identity, config);

        let pos1 = GeoPosition::new(37.0, -122.0, 100.0);
        let pos2 = GeoPosition::new(37.1, -122.1, 150.0);
        let pos3 = GeoPosition::new(37.2, -122.2, 200.0);
        let pos4 = GeoPosition::new(37.3, -122.3, 250.0);

        context.add_waypoint(FlightWaypoint::new(pos1, 10.0, 2.0));
        context.add_waypoint(FlightWaypoint::new(pos2, 12.0, 1.5));
        context.add_waypoint(FlightWaypoint::new(pos3, 11.0, 1.0));
        assert_eq!(context.flight_path.len(), 3);

        // Adding 4th waypoint should prune the first
        context.add_waypoint(FlightWaypoint::new(pos4, 10.5, 0.5));
        assert_eq!(context.flight_path.len(), 3);

        // First waypoint should now be pos2
        assert!((context.flight_path[0].position.latitude - 37.1).abs() < 0.001);
    }

    #[test]
    fn test_remote_id_broadcast() {
        let position = GeoPosition::new(37.7749, -122.4194, 100.0);
        let broadcast = RemoteIdBroadcast::new(
            "UAV-12345".to_string(),
            position,
            15.0,
            90, // Heading East
        );

        assert_eq!(broadcast.serial_number, "UAV-12345");
        assert_eq!(broadcast.heading_deg, 90);
        assert!((broadcast.ground_speed_ms - 15.0).abs() < 0.001);
    }

    #[test]
    fn test_haversine_distance() {
        // San Francisco to Los Angeles (approx 559 km)
        let sf = GeoPosition::new(37.7749, -122.4194, 0.0);
        let la = GeoPosition::new(34.0522, -118.2437, 0.0);
        let distance = haversine_distance(&sf, &la);

        // Should be approximately 559 km (allow 1% error)
        assert!(distance > 550000.0 && distance < 570000.0);
    }
}
