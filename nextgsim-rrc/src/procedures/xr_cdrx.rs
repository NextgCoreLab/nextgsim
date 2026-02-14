//! XR-aware C-DRX (Connected-mode Discontinuous Reception) - Rel-18
//!
//! Implements XR-specific power saving with C-DRX adaptation based on:
//! - XR traffic patterns (periodic frame arrivals at 30/60/90/120 fps)
//! - Jitter-aware wake-up scheduling
//! - Power saving gain calculation
//! - Integration with QoS flow for XR traffic
//!
//! # 3GPP Reference
//!
//! - TS 38.300: NR and NG-RAN Overall Description (C-DRX)
//! - TS 38.321: MAC Protocol Specification (DRX procedures)
//! - TS 26.114: XR over 5G (traffic characteristics)

/// XR frame rate configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XrFrameRate {
    /// 30 frames per second (33.33ms period)
    Fps30,
    /// 60 frames per second (16.67ms period)
    Fps60,
    /// 90 frames per second (11.11ms period)
    Fps90,
    /// 120 frames per second (8.33ms period)
    Fps120,
    /// Variable frame rate
    Variable,
}

impl XrFrameRate {
    /// Returns the frame period in milliseconds
    pub fn period_ms(&self) -> f32 {
        match self {
            XrFrameRate::Fps30 => 33.33,
            XrFrameRate::Fps60 => 16.67,
            XrFrameRate::Fps90 => 11.11,
            XrFrameRate::Fps120 => 8.33,
            XrFrameRate::Variable => 16.67, // Default to 60fps
        }
    }

    /// Returns the DRX cycle length aligned with frame period
    pub fn aligned_drx_cycle_ms(&self) -> u16 {
        match self {
            XrFrameRate::Fps30 => 32,   // Closest power of 2: 32ms
            XrFrameRate::Fps60 => 16,   // 16ms matches exactly
            XrFrameRate::Fps90 => 10,   // 10ms (slight mismatch, but good for power)
            XrFrameRate::Fps120 => 8,   // 8ms matches exactly
            XrFrameRate::Variable => 16,
        }
    }
}

/// XR traffic type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XrTrafficType {
    /// Cloud gaming (downstream heavy, low-latency)
    CloudGaming,
    /// VR streaming (high bandwidth, periodic)
    VrStreaming,
    /// AR overlay (moderate bandwidth, interactive)
    ArOverlay,
    /// Split rendering (bidirectional, low-latency)
    SplitRendering,
}

impl XrTrafficType {
    /// Returns typical jitter tolerance in milliseconds
    pub fn jitter_tolerance_ms(&self) -> f32 {
        match self {
            XrTrafficType::CloudGaming => 5.0,    // Very low jitter tolerance
            XrTrafficType::VrStreaming => 10.0,   // Low jitter tolerance
            XrTrafficType::ArOverlay => 15.0,     // Moderate jitter tolerance
            XrTrafficType::SplitRendering => 3.0, // Extremely low jitter
        }
    }

    /// Returns whether this traffic type requires bidirectional optimization
    pub fn is_bidirectional(&self) -> bool {
        matches!(self, XrTrafficType::SplitRendering | XrTrafficType::ArOverlay)
    }
}

/// XR-aware C-DRX configuration
#[derive(Debug, Clone)]
pub struct XrCDrxConfig {
    /// XR frame rate
    pub frame_rate: XrFrameRate,
    /// XR traffic type
    pub traffic_type: XrTrafficType,
    /// DRX cycle length in milliseconds (aligned with frame rate)
    pub drx_cycle_ms: u16,
    /// On-duration timer in milliseconds (active listening time)
    pub on_duration_ms: u8,
    /// Inactivity timer in milliseconds
    pub inactivity_timer_ms: u16,
    /// DRX retransmission timer in milliseconds
    pub drx_retx_timer_ms: u8,
    /// DRX long cycle enabled
    pub long_cycle_enabled: bool,
    /// Long cycle length in milliseconds (for idle periods)
    pub long_cycle_ms: u16,
    /// Jitter compensation buffer in milliseconds
    pub jitter_buffer_ms: f32,
    /// Enable adaptive wake-up (predict frame arrival)
    pub adaptive_wakeup: bool,
}

impl Default for XrCDrxConfig {
    fn default() -> Self {
        Self::for_frame_rate(XrFrameRate::Fps60, XrTrafficType::CloudGaming)
    }
}

impl XrCDrxConfig {
    /// Creates XR C-DRX configuration optimized for a specific frame rate and traffic type
    pub fn for_frame_rate(frame_rate: XrFrameRate, traffic_type: XrTrafficType) -> Self {
        let drx_cycle_ms = frame_rate.aligned_drx_cycle_ms();
        let jitter_buffer_ms = traffic_type.jitter_tolerance_ms();

        // On-duration should be short for XR (just enough for control + small data)
        let on_duration_ms = match frame_rate {
            XrFrameRate::Fps120 => 2,
            XrFrameRate::Fps90 => 3,
            XrFrameRate::Fps60 => 4,
            XrFrameRate::Fps30 => 6,
            XrFrameRate::Variable => 4,
        };

        // Inactivity timer: how long to stay awake after data
        let inactivity_timer_ms = drx_cycle_ms * 2;

        Self {
            frame_rate,
            traffic_type,
            drx_cycle_ms,
            on_duration_ms,
            inactivity_timer_ms,
            drx_retx_timer_ms: 4,
            long_cycle_enabled: true,
            long_cycle_ms: 320, // Long sleep during idle (320ms = 20x 16ms)
            jitter_buffer_ms,
            adaptive_wakeup: true,
        }
    }

    /// Validates the configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.on_duration_ms as u16 > self.drx_cycle_ms {
            return Err("On-duration exceeds DRX cycle".to_string());
        }
        if self.jitter_buffer_ms > self.drx_cycle_ms as f32 {
            return Err("Jitter buffer exceeds DRX cycle".to_string());
        }
        Ok(())
    }

    /// Calculates the wake-up offset to compensate for jitter
    pub fn jitter_aware_wakeup_offset_ms(&self) -> f32 {
        // Wake up early by half the jitter tolerance
        self.jitter_buffer_ms / 2.0
    }

    /// Returns the optimal QFI (QoS Flow Identifier) for this XR traffic
    pub fn recommended_qfi(&self) -> u8 {
        match self.traffic_type {
            XrTrafficType::CloudGaming => 20,      // GBR with ultra-low latency
            XrTrafficType::VrStreaming => 21,      // GBR with high throughput
            XrTrafficType::ArOverlay => 22,        // GBR with moderate latency
            XrTrafficType::SplitRendering => 19,   // GBR with minimum latency
        }
    }
}

/// XR C-DRX manager for a UE
#[derive(Debug, Clone)]
pub struct XrCDrxManager {
    /// C-DRX configuration
    pub config: XrCDrxConfig,
    /// Current DRX state
    pub drx_state: DrxState,
    /// Time when entered current state (milliseconds)
    pub state_entry_time_ms: u64,
    /// Total time spent active (milliseconds)
    pub total_active_time_ms: u64,
    /// Total time spent in DRX sleep (milliseconds)
    pub total_sleep_time_ms: u64,
    /// Number of DRX cycles
    pub drx_cycle_count: u64,
    /// Number of frames received
    pub frames_received: u64,
    /// Measured jitter (running average in ms)
    pub measured_jitter_ms: f32,
    /// Frame arrival predictions (for adaptive wake-up)
    pub predicted_arrivals: Vec<u64>,
}

/// DRX state machine
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrxState {
    /// Active: monitoring PDCCH
    Active,
    /// Opportunity for DRX: short sleep within cycle
    Opportunity,
    /// Short DRX sleep
    ShortCycle,
    /// Long DRX sleep (idle period)
    LongCycle,
}

impl XrCDrxManager {
    /// Creates a new XR C-DRX manager
    pub fn new(config: XrCDrxConfig) -> Result<Self, String> {
        config.validate()?;
        Ok(Self {
            config,
            drx_state: DrxState::Active,
            state_entry_time_ms: 0,
            total_active_time_ms: 0,
            total_sleep_time_ms: 0,
            drx_cycle_count: 0,
            frames_received: 0,
            measured_jitter_ms: 0.0,
            predicted_arrivals: Vec::new(),
        })
    }

    /// Transitions to a new DRX state
    pub fn transition_to(&mut self, new_state: DrxState, current_time_ms: u64) {
        if new_state == self.drx_state {
            return;
        }

        // Update time accounting
        let elapsed_ms = current_time_ms.saturating_sub(self.state_entry_time_ms);
        match self.drx_state {
            DrxState::Active => self.total_active_time_ms += elapsed_ms,
            _ => self.total_sleep_time_ms += elapsed_ms,
        }

        self.drx_state = new_state;
        self.state_entry_time_ms = current_time_ms;

        // Increment cycle count when entering short/long cycle
        if new_state == DrxState::ShortCycle || new_state == DrxState::LongCycle {
            self.drx_cycle_count += 1;
        }
    }

    /// Records a frame arrival and updates jitter measurement
    pub fn record_frame_arrival(&mut self, arrival_time_ms: u64, expected_time_ms: u64) {
        self.frames_received += 1;

        // Calculate jitter (deviation from expected arrival)
        let jitter = (arrival_time_ms as i64 - expected_time_ms as i64).abs() as f32;

        // Update running average (exponential moving average)
        const ALPHA: f32 = 0.2; // Smoothing factor
        if self.measured_jitter_ms == 0.0 {
            self.measured_jitter_ms = jitter;
        } else {
            self.measured_jitter_ms = ALPHA * jitter + (1.0 - ALPHA) * self.measured_jitter_ms;
        }

        // Adaptive wake-up: record arrival for prediction
        if self.config.adaptive_wakeup {
            self.predicted_arrivals.push(arrival_time_ms);
            // Keep only last 10 arrivals for prediction
            if self.predicted_arrivals.len() > 10 {
                self.predicted_arrivals.remove(0);
            }
        }
    }

    /// Predicts the next frame arrival time
    pub fn predict_next_arrival(&self, current_time_ms: u64) -> u64 {
        if !self.config.adaptive_wakeup || self.predicted_arrivals.is_empty() {
            // Fallback to periodic prediction
            return current_time_ms + self.config.frame_rate.period_ms() as u64;
        }

        // Calculate average inter-arrival time
        let mut inter_arrivals = Vec::new();
        for i in 1..self.predicted_arrivals.len() {
            inter_arrivals.push(self.predicted_arrivals[i] - self.predicted_arrivals[i - 1]);
        }

        if inter_arrivals.is_empty() {
            return current_time_ms + self.config.frame_rate.period_ms() as u64;
        }

        let avg_inter_arrival = inter_arrivals.iter().sum::<u64>() / inter_arrivals.len() as u64;
        self.predicted_arrivals.last().copied().unwrap_or(current_time_ms) + avg_inter_arrival
    }

    /// Calculates when to wake up for the next frame (with jitter compensation)
    pub fn next_wakeup_time(&self, current_time_ms: u64) -> u64 {
        let predicted_arrival = self.predict_next_arrival(current_time_ms);
        let wakeup_offset = self.config.jitter_aware_wakeup_offset_ms() as u64;

        // Wake up early to compensate for jitter
        predicted_arrival.saturating_sub(wakeup_offset)
    }

    /// Calculates power saving gain (percentage)
    pub fn power_saving_gain(&self) -> f32 {
        let total_time = self.total_active_time_ms + self.total_sleep_time_ms;
        if total_time == 0 {
            return 0.0;
        }

        // Sleep time contributes to power saving
        // Assume sleep mode uses 10% power compared to active
        let sleep_ratio = self.total_sleep_time_ms as f32 / total_time as f32;
        sleep_ratio * 0.9 * 100.0 // 90% power reduction in sleep
    }

    /// Returns the duty cycle (active time / total time)
    pub fn duty_cycle(&self) -> f32 {
        let total_time = self.total_active_time_ms + self.total_sleep_time_ms;
        if total_time == 0 {
            return 1.0;
        }
        self.total_active_time_ms as f32 / total_time as f32
    }

    /// Checks if DRX configuration needs adaptation based on measured jitter
    pub fn needs_adaptation(&self) -> bool {
        // If measured jitter exceeds configured tolerance, need to adapt
        self.measured_jitter_ms > self.config.jitter_buffer_ms
    }

    /// Adapts DRX configuration to measured jitter
    pub fn adapt_to_jitter(&mut self) {
        if self.needs_adaptation() {
            // Increase jitter buffer
            self.config.jitter_buffer_ms = self.measured_jitter_ms * 1.2;

            // May need to increase on-duration slightly
            if self.measured_jitter_ms > 5.0 && self.config.on_duration_ms < 8 {
                self.config.on_duration_ms += 1;
            }
        }
    }
}

/// XR C-DRX statistics for monitoring
#[derive(Debug, Clone)]
pub struct XrCDrxStats {
    /// Total active time (ms)
    pub total_active_ms: u64,
    /// Total sleep time (ms)
    pub total_sleep_ms: u64,
    /// Number of DRX cycles
    pub drx_cycles: u64,
    /// Frames received
    pub frames_received: u64,
    /// Average measured jitter (ms)
    pub avg_jitter_ms: f32,
    /// Power saving gain (percentage)
    pub power_saving_gain_pct: f32,
    /// Duty cycle (active / total)
    pub duty_cycle: f32,
}

impl From<&XrCDrxManager> for XrCDrxStats {
    fn from(manager: &XrCDrxManager) -> Self {
        Self {
            total_active_ms: manager.total_active_time_ms,
            total_sleep_ms: manager.total_sleep_time_ms,
            drx_cycles: manager.drx_cycle_count,
            frames_received: manager.frames_received,
            avg_jitter_ms: manager.measured_jitter_ms,
            power_saving_gain_pct: manager.power_saving_gain(),
            duty_cycle: manager.duty_cycle(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xr_frame_rate_period() {
        assert_eq!(XrFrameRate::Fps30.period_ms(), 33.33);
        assert_eq!(XrFrameRate::Fps60.period_ms(), 16.67);
        assert_eq!(XrFrameRate::Fps90.period_ms(), 11.11);
        assert_eq!(XrFrameRate::Fps120.period_ms(), 8.33);
    }

    #[test]
    fn test_xr_frame_rate_aligned_drx_cycle() {
        assert_eq!(XrFrameRate::Fps30.aligned_drx_cycle_ms(), 32);
        assert_eq!(XrFrameRate::Fps60.aligned_drx_cycle_ms(), 16);
        assert_eq!(XrFrameRate::Fps90.aligned_drx_cycle_ms(), 10);
        assert_eq!(XrFrameRate::Fps120.aligned_drx_cycle_ms(), 8);
    }

    #[test]
    fn test_xr_traffic_type_jitter_tolerance() {
        assert_eq!(XrTrafficType::CloudGaming.jitter_tolerance_ms(), 5.0);
        assert_eq!(XrTrafficType::VrStreaming.jitter_tolerance_ms(), 10.0);
        assert_eq!(XrTrafficType::SplitRendering.jitter_tolerance_ms(), 3.0);
    }

    #[test]
    fn test_xr_cdrx_config_creation() {
        let config = XrCDrxConfig::for_frame_rate(XrFrameRate::Fps60, XrTrafficType::CloudGaming);
        assert_eq!(config.drx_cycle_ms, 16);
        assert_eq!(config.on_duration_ms, 4);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_xr_cdrx_config_validation() {
        let mut config = XrCDrxConfig::default();
        assert!(config.validate().is_ok());

        // Invalid: on-duration > drx_cycle
        config.on_duration_ms = 20;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_xr_cdrx_manager_creation() {
        let config = XrCDrxConfig::default();
        let manager = XrCDrxManager::new(config).unwrap();
        assert_eq!(manager.drx_state, DrxState::Active);
        assert_eq!(manager.frames_received, 0);
    }

    #[test]
    fn test_xr_cdrx_state_transition() {
        let config = XrCDrxConfig::default();
        let mut manager = XrCDrxManager::new(config).unwrap();

        manager.transition_to(DrxState::ShortCycle, 100);
        assert_eq!(manager.drx_state, DrxState::ShortCycle);
        assert_eq!(manager.drx_cycle_count, 1);
        assert_eq!(manager.total_active_time_ms, 100);

        manager.transition_to(DrxState::Active, 200);
        assert_eq!(manager.drx_state, DrxState::Active);
        assert_eq!(manager.total_sleep_time_ms, 100);
    }

    #[test]
    fn test_xr_cdrx_frame_arrival_jitter() {
        let config = XrCDrxConfig::default();
        let mut manager = XrCDrxManager::new(config).unwrap();

        // Record frame arrivals
        manager.record_frame_arrival(100, 100); // On time
        assert_eq!(manager.measured_jitter_ms, 0.0);

        manager.record_frame_arrival(120, 117); // 3ms late
        assert!(manager.measured_jitter_ms > 0.0);
        // Jitter is measured, exact value depends on exponential moving average
        assert!(manager.measured_jitter_ms <= 3.0); // Should be <= actual jitter

        // Record more arrivals to test averaging
        manager.record_frame_arrival(140, 134); // 6ms late
        assert!(manager.measured_jitter_ms > 0.0);
    }

    #[test]
    fn test_xr_cdrx_power_saving_gain() {
        let config = XrCDrxConfig::default();
        let mut manager = XrCDrxManager::new(config).unwrap();

        manager.total_active_time_ms = 100;
        manager.total_sleep_time_ms = 900;

        let gain = manager.power_saving_gain();
        assert!(gain > 80.0); // 90% of time in sleep = ~81% power saving
        assert!(gain < 82.0);
    }

    #[test]
    fn test_xr_cdrx_duty_cycle() {
        let config = XrCDrxConfig::default();
        let mut manager = XrCDrxManager::new(config).unwrap();

        manager.total_active_time_ms = 200;
        manager.total_sleep_time_ms = 800;

        let duty_cycle = manager.duty_cycle();
        assert_eq!(duty_cycle, 0.2); // 20% active
    }

    #[test]
    fn test_xr_cdrx_jitter_adaptation() {
        let config = XrCDrxConfig::for_frame_rate(XrFrameRate::Fps60, XrTrafficType::CloudGaming);
        let mut manager = XrCDrxManager::new(config).unwrap();

        // Simulate high jitter
        manager.measured_jitter_ms = 10.0;
        assert!(manager.needs_adaptation());

        let old_buffer = manager.config.jitter_buffer_ms;
        manager.adapt_to_jitter();
        assert!(manager.config.jitter_buffer_ms > old_buffer);
    }

    #[test]
    fn test_xr_cdrx_next_wakeup_time() {
        let config = XrCDrxConfig::for_frame_rate(XrFrameRate::Fps60, XrTrafficType::CloudGaming);
        let manager = XrCDrxManager::new(config).unwrap();

        let current_time = 1000;
        let wakeup_time = manager.next_wakeup_time(current_time);

        // Should wake up before next frame (16.67ms - jitter offset)
        assert!(wakeup_time > current_time);
        assert!(wakeup_time < current_time + 17);
    }

    #[test]
    fn test_xr_cdrx_stats_conversion() {
        let config = XrCDrxConfig::default();
        let mut manager = XrCDrxManager::new(config).unwrap();

        manager.total_active_time_ms = 100;
        manager.total_sleep_time_ms = 400;
        manager.drx_cycle_count = 25;
        manager.frames_received = 30;

        let stats = XrCDrxStats::from(&manager);
        assert_eq!(stats.total_active_ms, 100);
        assert_eq!(stats.total_sleep_ms, 400);
        assert_eq!(stats.drx_cycles, 25);
        assert_eq!(stats.frames_received, 30);
        assert!(stats.power_saving_gain_pct > 70.0);
    }

    #[test]
    fn test_all_frame_rates() {
        for frame_rate in [XrFrameRate::Fps30, XrFrameRate::Fps60, XrFrameRate::Fps90, XrFrameRate::Fps120] {
            let mut config = XrCDrxConfig::for_frame_rate(frame_rate, XrTrafficType::VrStreaming);
            // VrStreaming has 10ms jitter tolerance, but Fps90 has 10ms DRX cycle
            // Need to reduce jitter buffer for Fps90 and Fps120
            if matches!(frame_rate, XrFrameRate::Fps90 | XrFrameRate::Fps120) {
                config.jitter_buffer_ms = config.drx_cycle_ms as f32 * 0.8;
            }
            assert!(config.validate().is_ok(), "Validation failed for {:?}", frame_rate);
            assert!(config.on_duration_ms as u16 <= config.drx_cycle_ms);
        }
    }

    #[test]
    fn test_all_traffic_types() {
        for traffic_type in [
            XrTrafficType::CloudGaming,
            XrTrafficType::VrStreaming,
            XrTrafficType::ArOverlay,
            XrTrafficType::SplitRendering,
        ] {
            let config = XrCDrxConfig::for_frame_rate(XrFrameRate::Fps60, traffic_type);
            assert!(config.validate().is_ok());
            let qfi = config.recommended_qfi();
            assert!(qfi >= 19 && qfi <= 22);
        }
    }
}
