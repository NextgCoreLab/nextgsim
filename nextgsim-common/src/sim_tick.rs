//! Unified simulation tick/timestep interface for synchronized simulation
//!
//! This module provides a common timestep interface that all 6G crates can use
//! for synchronized simulation runs.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Simulation tick counter
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SimulationTick(u64);

impl SimulationTick {
    /// Creates a new simulation tick
    pub fn new(tick: u64) -> Self {
        Self(tick)
    }

    /// Creates the initial tick (tick 0)
    pub fn initial() -> Self {
        Self(0)
    }

    /// Returns the tick value
    pub fn value(&self) -> u64 {
        self.0
    }

    /// Advances to the next tick
    pub fn next(&mut self) {
        self.0 += 1;
    }

    /// Returns the next tick without mutating
    pub fn next_tick(&self) -> Self {
        Self(self.0 + 1)
    }

    /// Advances by N ticks
    pub fn advance(&mut self, n: u64) {
        self.0 += n;
    }

    /// Returns a tick advanced by N ticks without mutating
    pub fn advanced_by(&self, n: u64) -> Self {
        Self(self.0 + n)
    }

    /// Returns true if this is the initial tick
    pub fn is_initial(&self) -> bool {
        self.0 == 0
    }

    /// Calculates the difference between two ticks
    pub fn diff(&self, other: &SimulationTick) -> u64 {
        self.0.abs_diff(other.0)
    }
}

impl std::fmt::Display for SimulationTick {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tick({})", self.0)
    }
}

impl From<u64> for SimulationTick {
    fn from(tick: u64) -> Self {
        Self::new(tick)
    }
}

impl From<SimulationTick> for u64 {
    fn from(tick: SimulationTick) -> u64 {
        tick.0
    }
}

/// Simulation time configuration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SimulationTimeConfig {
    /// Duration of each tick in milliseconds
    pub tick_duration_ms: u64,
    /// Total simulation duration in ticks
    pub total_ticks: u64,
    /// Real-time simulation (if true, wait for `tick_duration_ms` between ticks)
    pub real_time: bool,
}

impl Default for SimulationTimeConfig {
    fn default() -> Self {
        Self {
            tick_duration_ms: 100, // 100ms per tick (10 ticks per second)
            total_ticks: 1000,     // 100 seconds of simulation
            real_time: false,
        }
    }
}

impl SimulationTimeConfig {
    /// Creates a new simulation time configuration
    pub fn new(tick_duration_ms: u64, total_ticks: u64, real_time: bool) -> Self {
        Self {
            tick_duration_ms,
            total_ticks,
            real_time,
        }
    }

    /// Returns the total simulation duration in milliseconds
    pub fn total_duration_ms(&self) -> u64 {
        self.tick_duration_ms * self.total_ticks
    }

    /// Returns the total simulation duration as a Duration
    pub fn total_duration(&self) -> Duration {
        Duration::from_millis(self.total_duration_ms())
    }

    /// Returns the tick duration as a Duration
    pub fn tick_duration(&self) -> Duration {
        Duration::from_millis(self.tick_duration_ms)
    }

    /// Converts a tick to simulation time in milliseconds
    pub fn tick_to_ms(&self, tick: SimulationTick) -> u64 {
        tick.value() * self.tick_duration_ms
    }

    /// Converts simulation time in milliseconds to a tick
    pub fn ms_to_tick(&self, ms: u64) -> SimulationTick {
        SimulationTick::new(ms / self.tick_duration_ms)
    }
}

/// Trait for components that can be stepped forward in simulation
pub trait SimulationStepper {
    /// Steps the component forward by one tick
    ///
    /// # Arguments
    ///
    /// * `tick` - Current simulation tick
    ///
    /// # Returns
    ///
    /// Result indicating success or error
    fn step(&mut self, tick: SimulationTick) -> Result<(), String>;

    /// Resets the component to initial state
    fn reset(&mut self);
}

/// Simulation clock for coordinating timesteps
#[derive(Debug)]
pub struct SimulationClock {
    current_tick: SimulationTick,
    config: SimulationTimeConfig,
    start_time: std::time::Instant,
}

impl SimulationClock {
    /// Creates a new simulation clock
    pub fn new(config: SimulationTimeConfig) -> Self {
        Self {
            current_tick: SimulationTick::initial(),
            config,
            start_time: std::time::Instant::now(),
        }
    }

    /// Returns the current tick
    pub fn current_tick(&self) -> SimulationTick {
        self.current_tick
    }

    /// Returns the configuration
    pub fn config(&self) -> &SimulationTimeConfig {
        &self.config
    }

    /// Advances the clock by one tick
    pub fn tick(&mut self) {
        self.current_tick.next();
    }

    /// Returns true if the simulation is complete
    pub fn is_complete(&self) -> bool {
        self.current_tick.value() >= self.config.total_ticks
    }

    /// Returns the current simulation time in milliseconds
    pub fn current_time_ms(&self) -> u64 {
        self.config.tick_to_ms(self.current_tick)
    }

    /// Returns the elapsed real time since simulation start
    pub fn elapsed_real_time(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Resets the clock to initial state
    pub fn reset(&mut self) {
        self.current_tick = SimulationTick::initial();
        self.start_time = std::time::Instant::now();
    }

    /// Waits for the next tick if in real-time mode
    pub fn wait_for_next_tick(&self) {
        if self.config.real_time {
            let target_time = self.current_time_ms() + self.config.tick_duration_ms;
            let elapsed_ms = self.elapsed_real_time().as_millis() as u64;

            if target_time > elapsed_ms {
                let wait_time_ms = target_time - elapsed_ms;
                std::thread::sleep(Duration::from_millis(wait_time_ms));
            }
        }
    }
}

impl Default for SimulationClock {
    fn default() -> Self {
        Self::new(SimulationTimeConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulation_tick_creation() {
        let tick = SimulationTick::new(42);
        assert_eq!(tick.value(), 42);
        assert_eq!(format!("{tick}"), "Tick(42)");
    }

    #[test]
    fn test_simulation_tick_initial() {
        let tick = SimulationTick::initial();
        assert_eq!(tick.value(), 0);
        assert!(tick.is_initial());
    }

    #[test]
    fn test_simulation_tick_next() {
        let mut tick = SimulationTick::new(5);
        tick.next();
        assert_eq!(tick.value(), 6);

        let next_tick = tick.next_tick();
        assert_eq!(next_tick.value(), 7);
        assert_eq!(tick.value(), 6); // Original unchanged
    }

    #[test]
    fn test_simulation_tick_advance() {
        let mut tick = SimulationTick::new(10);
        tick.advance(5);
        assert_eq!(tick.value(), 15);

        let advanced = tick.advanced_by(10);
        assert_eq!(advanced.value(), 25);
        assert_eq!(tick.value(), 15); // Original unchanged
    }

    #[test]
    fn test_simulation_tick_diff() {
        let tick1 = SimulationTick::new(10);
        let tick2 = SimulationTick::new(25);

        assert_eq!(tick1.diff(&tick2), 15);
        assert_eq!(tick2.diff(&tick1), 15);
    }

    #[test]
    fn test_simulation_tick_from_u64() {
        let tick: SimulationTick = 100.into();
        assert_eq!(tick.value(), 100);

        let value: u64 = tick.into();
        assert_eq!(value, 100);
    }

    #[test]
    fn test_simulation_time_config_default() {
        let config = SimulationTimeConfig::default();
        assert_eq!(config.tick_duration_ms, 100);
        assert_eq!(config.total_ticks, 1000);
        assert!(!config.real_time);
    }

    #[test]
    fn test_simulation_time_config_duration() {
        let config = SimulationTimeConfig::new(50, 200, false);
        assert_eq!(config.total_duration_ms(), 10000); // 50ms * 200 = 10000ms
        assert_eq!(config.total_duration(), Duration::from_secs(10));
        assert_eq!(config.tick_duration(), Duration::from_millis(50));
    }

    #[test]
    fn test_simulation_time_config_conversion() {
        let config = SimulationTimeConfig::new(100, 1000, false);

        let tick = SimulationTick::new(50);
        assert_eq!(config.tick_to_ms(tick), 5000);

        let tick_from_ms = config.ms_to_tick(7500);
        assert_eq!(tick_from_ms.value(), 75);
    }

    #[test]
    fn test_simulation_clock_creation() {
        let config = SimulationTimeConfig::default();
        let clock = SimulationClock::new(config);

        assert_eq!(clock.current_tick().value(), 0);
        assert!(!clock.is_complete());
    }

    #[test]
    fn test_simulation_clock_tick() {
        let config = SimulationTimeConfig::new(100, 10, false);
        let mut clock = SimulationClock::new(config);

        assert_eq!(clock.current_tick().value(), 0);
        assert_eq!(clock.current_time_ms(), 0);

        clock.tick();
        assert_eq!(clock.current_tick().value(), 1);
        assert_eq!(clock.current_time_ms(), 100);

        clock.tick();
        assert_eq!(clock.current_tick().value(), 2);
        assert_eq!(clock.current_time_ms(), 200);
    }

    #[test]
    fn test_simulation_clock_completion() {
        let config = SimulationTimeConfig::new(100, 5, false);
        let mut clock = SimulationClock::new(config);

        assert!(!clock.is_complete());

        for _ in 0..5 {
            clock.tick();
        }

        assert!(clock.is_complete());
    }

    #[test]
    fn test_simulation_clock_reset() {
        let config = SimulationTimeConfig::new(100, 10, false);
        let mut clock = SimulationClock::new(config);

        clock.tick();
        clock.tick();
        clock.tick();
        assert_eq!(clock.current_tick().value(), 3);

        clock.reset();
        assert_eq!(clock.current_tick().value(), 0);
    }

    // Mock implementation of SimulationStepper for testing
    struct MockStepper {
        tick_count: u64,
    }

    impl MockStepper {
        fn new() -> Self {
            Self { tick_count: 0 }
        }
    }

    impl SimulationStepper for MockStepper {
        fn step(&mut self, _tick: SimulationTick) -> Result<(), String> {
            self.tick_count += 1;
            Ok(())
        }

        fn reset(&mut self) {
            self.tick_count = 0;
        }
    }

    #[test]
    fn test_simulation_stepper_trait() {
        let mut stepper = MockStepper::new();

        assert_eq!(stepper.tick_count, 0);

        stepper.step(SimulationTick::new(1)).unwrap();
        assert_eq!(stepper.tick_count, 1);

        stepper.step(SimulationTick::new(2)).unwrap();
        assert_eq!(stepper.tick_count, 2);

        stepper.reset();
        assert_eq!(stepper.tick_count, 0);
    }
}
