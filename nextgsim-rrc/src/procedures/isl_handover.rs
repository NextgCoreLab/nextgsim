//! Inter-Satellite Link (ISL) Handover for NTN
//!
//! Implements ISL handover procedures for Non-Terrestrial Networks where
//! satellite-to-satellite handovers occur with minimal UE involvement.
//!
//! # 3GPP Reference
//!
//! - TS 38.300: NR and NG-RAN Overall Description (NTN extensions)
//! - TS 38.331: RRC Protocol Specification (NTN procedures)
//! - TS 38.821: Solutions for NR to support non-terrestrial networks

/// ISL handover context
///
/// Tracks the state of an inter-satellite link handover
#[derive(Debug, Clone)]
pub struct IslHandoverContext {
    /// Source satellite identifier
    pub source_satellite: u32,
    /// Target satellite identifier
    pub target_satellite: u32,
    /// Feeder link identifier (satellite-to-gateway)
    pub feeder_link: Option<u32>,
    /// Service link identifier (satellite-to-UE)
    pub service_link: u32,
    /// Current handover state
    pub state: IslHandoverState,
    /// Handover trigger time (ms since epoch)
    pub trigger_time_ms: u64,
    /// Estimated completion time (ms since epoch)
    pub estimated_completion_ms: Option<u64>,
    /// ISL delay in microseconds
    pub isl_delay_us: Option<u64>,
}

/// ISL handover state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IslHandoverState {
    /// No handover in progress
    Idle,
    /// Handover preparation (target satellite configuration)
    Preparing,
    /// Handover execution (switching service link)
    Executing,
    /// Handover completed successfully
    Completed,
    /// Handover failed
    Failed,
}

impl IslHandoverContext {
    /// Creates a new ISL handover context
    pub fn new(
        source_satellite: u32,
        target_satellite: u32,
        service_link: u32,
        feeder_link: Option<u32>,
    ) -> Self {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            source_satellite,
            target_satellite,
            feeder_link,
            service_link,
            state: IslHandoverState::Idle,
            trigger_time_ms: now_ms,
            estimated_completion_ms: None,
            isl_delay_us: None,
        }
    }

    /// Initiates ISL handover preparation
    pub fn initiate_preparation(&mut self) {
        self.state = IslHandoverState::Preparing;
    }

    /// Executes ISL handover
    pub fn execute(&mut self) {
        self.state = IslHandoverState::Executing;
    }

    /// Completes ISL handover successfully
    pub fn complete(&mut self) {
        self.state = IslHandoverState::Completed;
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        self.estimated_completion_ms = Some(now_ms);
    }

    /// Marks ISL handover as failed
    pub fn fail(&mut self) {
        self.state = IslHandoverState::Failed;
    }

    /// Calculates ISL delay based on inter-satellite distance
    ///
    /// # Arguments
    /// * `distance_km` - Distance between satellites in kilometers
    pub fn calculate_isl_delay(&mut self, distance_km: f64) {
        // Speed of light: ~300,000 km/s
        const SPEED_OF_LIGHT_KM_PER_US: f64 = 0.3; // km/microsecond
        let delay_us = (distance_km / SPEED_OF_LIGHT_KM_PER_US) as u64;
        self.isl_delay_us = Some(delay_us);
    }

    /// Returns true if handover is in progress
    pub fn is_in_progress(&self) -> bool {
        matches!(
            self.state,
            IslHandoverState::Preparing | IslHandoverState::Executing
        )
    }

    /// Returns true if handover is completed
    pub fn is_completed(&self) -> bool {
        self.state == IslHandoverState::Completed
    }
}

/// ISL handover manager
///
/// Manages inter-satellite link handovers for NTN scenarios
pub struct IslHandoverManager {
    /// Active ISL handover context
    active_handover: Option<IslHandoverContext>,
    /// Satellite position cache for delay calculation
    satellite_positions: std::collections::HashMap<u32, SatellitePosition>,
}

/// Satellite position in 3D space
#[derive(Debug, Clone, Copy)]
pub struct SatellitePosition {
    /// X coordinate in km (ECEF)
    pub x_km: f64,
    /// Y coordinate in km (ECEF)
    pub y_km: f64,
    /// Z coordinate in km (ECEF)
    pub z_km: f64,
    /// Position timestamp (ms since epoch)
    pub timestamp_ms: u64,
}

impl SatellitePosition {
    /// Calculates distance to another satellite position in km
    pub fn distance_to(&self, other: &SatellitePosition) -> f64 {
        let dx = self.x_km - other.x_km;
        let dy = self.y_km - other.y_km;
        let dz = self.z_km - other.z_km;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

impl IslHandoverManager {
    /// Creates a new ISL handover manager
    pub fn new() -> Self {
        Self {
            active_handover: None,
            satellite_positions: std::collections::HashMap::new(),
        }
    }

    /// Updates satellite position
    pub fn update_satellite_position(&mut self, satellite_id: u32, position: SatellitePosition) {
        self.satellite_positions.insert(satellite_id, position);
    }

    /// Initiates an ISL handover
    pub fn initiate_isl_handover(
        &mut self,
        source_satellite: u32,
        target_satellite: u32,
        service_link: u32,
        feeder_link: Option<u32>,
    ) -> Result<(), IslHandoverError> {
        if self.active_handover.is_some() {
            return Err(IslHandoverError::HandoverInProgress);
        }

        let mut context =
            IslHandoverContext::new(source_satellite, target_satellite, service_link, feeder_link);

        // Calculate ISL delay if satellite positions are available
        if let (Some(source_pos), Some(target_pos)) = (
            self.satellite_positions.get(&source_satellite),
            self.satellite_positions.get(&target_satellite),
        ) {
            let distance_km = source_pos.distance_to(target_pos);
            context.calculate_isl_delay(distance_km);
        }

        context.initiate_preparation();
        self.active_handover = Some(context);

        Ok(())
    }

    /// Prepares target satellite for handover
    ///
    /// Pre-configures the target satellite with UE context before switching
    pub fn prepare_target_satellite(&mut self) -> Result<(), IslHandoverError> {
        let context = self
            .active_handover
            .as_mut()
            .ok_or(IslHandoverError::NoActiveHandover)?;

        if context.state != IslHandoverState::Preparing {
            return Err(IslHandoverError::InvalidState);
        }

        // In a real implementation, would send UE context to target satellite
        // For now, just transition to executing state
        context.execute();

        Ok(())
    }

    /// Executes ISL switch
    ///
    /// Switches the service link from source to target satellite
    pub fn execute_isl_switch(&mut self) -> Result<(), IslHandoverError> {
        let context = self
            .active_handover
            .as_mut()
            .ok_or(IslHandoverError::NoActiveHandover)?;

        if context.state != IslHandoverState::Executing {
            return Err(IslHandoverError::InvalidState);
        }

        // Seamless switch: UE continues using same service link
        // but it's now served by target satellite
        context.complete();

        Ok(())
    }

    /// Gets the active ISL handover context
    pub fn get_active_handover(&self) -> Option<&IslHandoverContext> {
        self.active_handover.as_ref()
    }

    /// Clears the active handover
    pub fn clear_active_handover(&mut self) {
        self.active_handover = None;
    }

    /// Gets ISL delay for active handover
    pub fn get_isl_delay_us(&self) -> Option<u64> {
        self.active_handover
            .as_ref()
            .and_then(|ctx| ctx.isl_delay_us)
    }
}

impl Default for IslHandoverManager {
    fn default() -> Self {
        Self::new()
    }
}

/// ISL handover error types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IslHandoverError {
    /// Another handover is already in progress
    HandoverInProgress,
    /// No active handover
    NoActiveHandover,
    /// Invalid handover state for the requested operation
    InvalidState,
    /// Satellite position not available
    SatellitePositionUnavailable,
}

impl std::fmt::Display for IslHandoverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IslHandoverError::HandoverInProgress => write!(f, "Handover already in progress"),
            IslHandoverError::NoActiveHandover => write!(f, "No active handover"),
            IslHandoverError::InvalidState => write!(f, "Invalid handover state"),
            IslHandoverError::SatellitePositionUnavailable => {
                write!(f, "Satellite position unavailable")
            }
        }
    }
}

impl std::error::Error for IslHandoverError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isl_handover_context() {
        let mut context = IslHandoverContext::new(100, 200, 1, Some(10));

        assert_eq!(context.source_satellite, 100);
        assert_eq!(context.target_satellite, 200);
        assert_eq!(context.state, IslHandoverState::Idle);
        assert!(!context.is_in_progress());

        context.initiate_preparation();
        assert_eq!(context.state, IslHandoverState::Preparing);
        assert!(context.is_in_progress());

        context.execute();
        assert_eq!(context.state, IslHandoverState::Executing);
        assert!(context.is_in_progress());

        context.complete();
        assert_eq!(context.state, IslHandoverState::Completed);
        assert!(!context.is_in_progress());
        assert!(context.is_completed());
    }

    #[test]
    fn test_isl_delay_calculation() {
        let mut context = IslHandoverContext::new(100, 200, 1, None);

        // 1000 km distance
        context.calculate_isl_delay(1000.0);

        // Delay should be approximately 1000 / 0.3 = 3333 microseconds
        assert!(context.isl_delay_us.is_some());
        let delay = context.isl_delay_us.unwrap();
        assert!(delay > 3300 && delay < 3400);
    }

    #[test]
    fn test_satellite_position_distance() {
        let pos1 = SatellitePosition {
            x_km: 0.0,
            y_km: 0.0,
            z_km: 0.0,
            timestamp_ms: 0,
        };

        let pos2 = SatellitePosition {
            x_km: 300.0,
            y_km: 400.0,
            z_km: 0.0,
            timestamp_ms: 0,
        };

        let distance = pos1.distance_to(&pos2);
        assert!((distance - 500.0).abs() < 0.01); // 3-4-5 triangle
    }

    #[test]
    fn test_isl_handover_manager() {
        let mut manager = IslHandoverManager::new();

        // Add satellite positions
        let pos1 = SatellitePosition {
            x_km: 0.0,
            y_km: 0.0,
            z_km: 6871.0, // LEO altitude ~500 km above Earth radius
            timestamp_ms: 0,
        };

        let pos2 = SatellitePosition {
            x_km: 1000.0,
            y_km: 0.0,
            z_km: 6871.0,
            timestamp_ms: 0,
        };

        manager.update_satellite_position(100, pos1);
        manager.update_satellite_position(200, pos2);

        // Initiate handover
        let result = manager.initiate_isl_handover(100, 200, 1, None);
        assert!(result.is_ok());

        let handover = manager.get_active_handover().unwrap();
        assert_eq!(handover.source_satellite, 100);
        assert_eq!(handover.target_satellite, 200);
        assert!(handover.isl_delay_us.is_some());

        // Prepare target
        let result = manager.prepare_target_satellite();
        assert!(result.is_ok());

        // Execute switch
        let result = manager.execute_isl_switch();
        assert!(result.is_ok());

        let handover = manager.get_active_handover().unwrap();
        assert!(handover.is_completed());
    }

    #[test]
    fn test_isl_handover_errors() {
        let mut manager = IslHandoverManager::new();

        // Try to prepare without initiating
        let result = manager.prepare_target_satellite();
        assert!(matches!(result, Err(IslHandoverError::NoActiveHandover)));

        // Initiate handover
        manager
            .initiate_isl_handover(100, 200, 1, None)
            .unwrap();

        // Try to initiate another handover
        let result = manager.initiate_isl_handover(100, 300, 1, None);
        assert!(matches!(result, Err(IslHandoverError::HandoverInProgress)));

        // Try to execute before preparing
        let result = manager.execute_isl_switch();
        assert!(matches!(result, Err(IslHandoverError::InvalidState)));
    }
}
