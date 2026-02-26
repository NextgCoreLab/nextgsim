//! UAV (Unmanned Aerial Vehicle) UE Context (Rel-17/18, TS 23.256)
//!
//! Top-level UAV context for the UE, wrapping rrc/uav.rs RRC-level types
//! with NAS-level UAV authorization, C2 link management, geofencing,
//! and UTM (UAS Traffic Management) integration.

pub use crate::rrc::uav::{
    GeoPosition, RemoteIdBroadcast, UavAuthorizationState, UavIdentity, UavRegistrationContext,
};

use std::collections::VecDeque;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// C2 (Command and Control) link quality thresholds (per TS 23.256 §6.3.2)
const C2_LINK_LOSS_RSRP_DBM: i32 = -120;
const C2_LINK_WARNING_RSRP_DBM: i32 = -110;

/// C2 link status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C2LinkStatus {
    /// C2 link nominal
    Nominal,
    /// C2 link degraded (warning threshold crossed)
    Warning,
    /// C2 link lost (UAS must execute lost-link procedure)
    Lost,
}

/// UAV trajectory waypoint
#[derive(Debug, Clone, Copy)]
pub struct Waypoint {
    pub position: GeoPosition,
    /// Planned time at this waypoint (UNIX seconds)
    pub eta: u64,
    /// Airspeed in m/s
    pub airspeed_mps: f32,
}

/// Flight trajectory
#[derive(Debug, Clone, Default)]
pub struct FlightTrajectory {
    pub waypoints: Vec<Waypoint>,
    /// Whether this trajectory has been approved by UTM
    pub utm_approved: bool,
    /// UTM approval token (if approved)
    pub utm_token: Option<String>,
}

impl FlightTrajectory {
    /// Returns true if trajectory is complete (all waypoints visited)
    pub fn is_empty(&self) -> bool {
        self.waypoints.is_empty()
    }
}

/// Geofence (cylindrical) zone
#[derive(Debug, Clone)]
pub struct CylindricalGeofence {
    /// Center position
    pub center: GeoPosition,
    /// Radius in meters
    pub radius_m: f64,
    /// Floor altitude (HAE in meters)
    pub floor_m: f64,
    /// Ceiling altitude (HAE in meters)
    pub ceiling_m: f64,
    /// Whether this is a restricted (no-fly) zone
    pub restricted: bool,
}

impl CylindricalGeofence {
    /// Checks if a position is inside this geofence
    pub fn contains(&self, pos: &GeoPosition) -> bool {
        let dlat = (pos.latitude - self.center.latitude).to_radians();
        let dlon = (pos.longitude - self.center.longitude).to_radians();
        let lat1 = self.center.latitude.to_radians();
        let lat2 = pos.latitude.to_radians();
        // Haversine formula
        let a = (dlat / 2.0).sin().powi(2)
            + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
        let dist_m = 6_371_000.0 * c;
        let alt_ok = pos.altitude_msl >= self.floor_m && pos.altitude_msl <= self.ceiling_m;
        dist_m <= self.radius_m && alt_ok
    }
}

/// UAV NAS-level context for the UE
pub struct UavContext {
    /// UAV identity (ASTM F3411 remote ID)
    pub identity: UavIdentity,
    /// Current authorization state
    pub auth_state: UavAuthorizationState,
    /// Current position
    pub current_position: Option<GeoPosition>,
    /// Position history (rolling 100 samples)
    pub position_history: VecDeque<(GeoPosition, u64)>,
    /// Planned trajectory
    pub trajectory: FlightTrajectory,
    /// C2 link status
    pub c2_status: C2LinkStatus,
    /// Last measured C2 link RSRP (dBm)
    pub c2_rsrp_dbm: i32,
    /// Active geofences
    pub geofences: Vec<CylindricalGeofence>,
    /// Whether remote ID broadcast is active
    pub remote_id_active: bool,
    /// Remote ID broadcast interval (seconds)
    pub remote_id_interval_secs: u64,
    /// Last remote ID broadcast timestamp
    pub last_broadcast_at: Option<u64>,
}

impl UavContext {
    /// Creates a new UAV context with the given identity
    pub fn new(identity: UavIdentity) -> Self {
        Self {
            identity,
            auth_state: UavAuthorizationState::NotAuthorized,
            current_position: None,
            position_history: VecDeque::with_capacity(100),
            trajectory: FlightTrajectory::default(),
            c2_status: C2LinkStatus::Nominal,
            c2_rsrp_dbm: -90,
            geofences: Vec::new(),
            remote_id_active: false,
            remote_id_interval_secs: 3,
            last_broadcast_at: None,
        }
    }

    /// Updates position and checks geofence violations
    ///
    /// Returns Some(geofence_index) if a restricted zone is entered.
    pub fn update_position(&mut self, pos: GeoPosition) -> Option<usize> {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs();
        self.current_position = Some(pos);
        if self.position_history.len() >= 100 {
            self.position_history.pop_front();
        }
        self.position_history.push_back((pos, ts));

        // Check for restricted geofence violations
        self.geofences.iter().enumerate()
            .find(|(_, gf)| gf.restricted && gf.contains(&pos))
            .map(|(i, _)| i)
    }

    /// Updates C2 link quality
    pub fn update_c2_rsrp(&mut self, rsrp_dbm: i32) {
        self.c2_rsrp_dbm = rsrp_dbm;
        self.c2_status = if rsrp_dbm <= C2_LINK_LOSS_RSRP_DBM {
            C2LinkStatus::Lost
        } else if rsrp_dbm <= C2_LINK_WARNING_RSRP_DBM {
            C2LinkStatus::Warning
        } else {
            C2LinkStatus::Nominal
        };
    }

    /// Returns true if the UAV should broadcast remote ID now
    pub fn should_broadcast_remote_id(&self) -> bool {
        if !self.remote_id_active {
            return false;
        }
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs();
        self.last_broadcast_at
            .map(|t| now.saturating_sub(t) >= self.remote_id_interval_secs)
            .unwrap_or(true)
    }

    /// Records a remote ID broadcast
    pub fn record_broadcast(&mut self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs();
        self.last_broadcast_at = Some(now);
    }

    /// Adds a geofence
    pub fn add_geofence(&mut self, gf: CylindricalGeofence) {
        self.geofences.push(gf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_identity() -> UavIdentity {
        UavIdentity::new(
            "SN-12345678".into(),
            "AcmeDrones".into(),
            "HawkX-200".into(),
        )
    }

    #[test]
    fn test_uav_context_creation() {
        let ctx = UavContext::new(test_identity());
        assert_eq!(ctx.auth_state, UavAuthorizationState::NotAuthorized);
        assert_eq!(ctx.c2_status, C2LinkStatus::Nominal);
    }

    #[test]
    fn test_c2_link_status_transitions() {
        let mut ctx = UavContext::new(test_identity());
        ctx.update_c2_rsrp(-115); // -115 <= -110 → Warning
        assert_eq!(ctx.c2_status, C2LinkStatus::Warning);
        ctx.update_c2_rsrp(-125); // -125 <= -120 → Lost
        assert_eq!(ctx.c2_status, C2LinkStatus::Lost);
        ctx.update_c2_rsrp(-85); // -85 > -110 → Nominal
        assert_eq!(ctx.c2_status, C2LinkStatus::Nominal);
    }

    #[test]
    fn test_geofence_containment() {
        let gf = CylindricalGeofence {
            center: GeoPosition { latitude: 37.0, longitude: -122.0, altitude_msl: 0.0 },
            radius_m: 1000.0,
            floor_m: 0.0,
            ceiling_m: 400.0,
            restricted: true,
        };
        let inside = GeoPosition { latitude: 37.001, longitude: -122.0, altitude_msl: 100.0 };
        let outside = GeoPosition { latitude: 38.0, longitude: -122.0, altitude_msl: 100.0 };
        assert!(gf.contains(&inside));
        assert!(!gf.contains(&outside));
    }

    #[test]
    fn test_position_update_geofence_violation() {
        let mut ctx = UavContext::new(test_identity());
        ctx.add_geofence(CylindricalGeofence {
            center: GeoPosition { latitude: 37.0, longitude: -122.0, altitude_msl: 0.0 },
            radius_m: 1000.0,
            floor_m: 0.0,
            ceiling_m: 400.0,
            restricted: true,
        });
        let outside = GeoPosition { latitude: 38.0, longitude: -122.0, altitude_msl: 100.0 };
        assert!(ctx.update_position(outside).is_none());
        let inside = GeoPosition { latitude: 37.001, longitude: -122.0, altitude_msl: 100.0 };
        assert!(ctx.update_position(inside).is_some());
    }

    #[test]
    fn test_remote_id_broadcast_timing() {
        let mut ctx = UavContext::new(test_identity());
        ctx.remote_id_active = true;
        ctx.remote_id_interval_secs = 3;
        // No prior broadcast — should broadcast
        assert!(ctx.should_broadcast_remote_id());
        ctx.record_broadcast();
        // Just broadcast — should not broadcast again immediately
        assert!(!ctx.should_broadcast_remote_id());
    }

    #[test]
    fn test_position_history_bounded() {
        let mut ctx = UavContext::new(test_identity());
        for i in 0..150u64 {
            ctx.update_position(GeoPosition {
                latitude: 37.0 + i as f64 * 0.0001,
                longitude: -122.0,
                altitude_msl: 100.0,
            });
        }
        assert!(ctx.position_history.len() <= 100);
    }
}
