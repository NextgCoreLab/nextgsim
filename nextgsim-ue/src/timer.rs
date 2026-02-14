//! UE Timer Management
//!
//! This module implements timer management for the UE, supporting NAS timers
//! used in 5G mobility management (MM) and session management (SM) procedures.
//!
//! # Timer Types
//!
//! NAS timers are identified by their code (e.g., T3510, T3511, T3512) and
//! categorized as either MM timers or SM timers.
//!
//! # NAS Timer Definitions (3GPP TS 24.501)
//!
//! ## Mobility Management Timers
//! - T3346: Backoff timer for congestion control
//! - T3502: PLMN selection retry timer
//! - T3510: Registration procedure timer
//! - T3511: Registration retry timer
//! - T3512: Periodic registration update timer
//! - T3516: 5G-AKA authentication timer
//! - T3517: Service request timer
//! - T3519: SUCI storage timer
//! - T3520: Authentication failure timer
//! - T3521: Deregistration timer
//! - T3525: Service request retry timer
//! - T3540: Payload container timer
//!
//! ## Session Management Timers
//! - T3580: PDU session establishment timer
//! - T3581: PDU session modification timer
//! - T3582: PDU session release timer
//!
//! # GPRS Timer Formats
//!
//! The module supports starting timers with values from NAS IEs:
//! - GPRS Timer 2: Simple value in seconds
//! - GPRS Timer 3: Value with unit multiplier (2s, 30s, 1min, 10min, 1hr, 10hr, 320hr)

use std::time::{Duration, Instant};

// ============================================================================
// NAS Timer Constants (3GPP TS 24.501)
// ============================================================================

/// Timer code for T3346 (Backoff timer for congestion control)
pub const TIMER_T3346: u16 = 3346;
/// Timer code for T3502 (PLMN selection retry)
pub const TIMER_T3502: u16 = 3502;
/// Timer code for T3510 (Registration procedure)
pub const TIMER_T3510: u16 = 3510;
/// Timer code for T3511 (Registration retry)
pub const TIMER_T3511: u16 = 3511;
/// Timer code for T3512 (Periodic registration update)
pub const TIMER_T3512: u16 = 3512;
/// Timer code for T3516 (5G-AKA authentication)
pub const TIMER_T3516: u16 = 3516;
/// Timer code for T3517 (Service request)
pub const TIMER_T3517: u16 = 3517;
/// Timer code for T3519 (SUCI storage)
pub const TIMER_T3519: u16 = 3519;
/// Timer code for T3520 (Authentication failure)
pub const TIMER_T3520: u16 = 3520;
/// Timer code for T3521 (Deregistration)
pub const TIMER_T3521: u16 = 3521;
/// Timer code for T3525 (Service request retry)
pub const TIMER_T3525: u16 = 3525;
/// Timer code for T3540 (Payload container)
pub const TIMER_T3540: u16 = 3540;

/// Timer code for T3580 (PDU session establishment)
pub const TIMER_T3580: u16 = 3580;
/// Timer code for T3581 (PDU session modification)
pub const TIMER_T3581: u16 = 3581;
/// Timer code for T3582 (PDU session release)
pub const TIMER_T3582: u16 = 3582;

// Default timer values in seconds (3GPP TS 24.501 Table 10.2.1 and 10.3.1)

/// Default T3346 interval: 12 minutes (network controlled, this is max)
pub const DEFAULT_T3346_INTERVAL: u32 = 12 * 60;
/// Default T3502 interval: 12 minutes
pub const DEFAULT_T3502_INTERVAL: u32 = 12 * 60;
/// Default T3510 interval: 15 seconds
pub const DEFAULT_T3510_INTERVAL: u32 = 15;
/// Default T3511 interval: 10 seconds
pub const DEFAULT_T3511_INTERVAL: u32 = 10;
/// Default T3512 interval: 54 minutes (network controlled)
pub const DEFAULT_T3512_INTERVAL: u32 = 54 * 60;
/// Default T3516 interval: 30 seconds
pub const DEFAULT_T3516_INTERVAL: u32 = 30;
/// Default T3517 interval: 15 seconds
pub const DEFAULT_T3517_INTERVAL: u32 = 15;
/// Default T3519 interval: 60 seconds
pub const DEFAULT_T3519_INTERVAL: u32 = 60;
/// Default T3520 interval: 15 seconds
pub const DEFAULT_T3520_INTERVAL: u32 = 15;
/// Default T3521 interval: 15 seconds
pub const DEFAULT_T3521_INTERVAL: u32 = 15;
/// Default T3525 interval: 60 seconds
pub const DEFAULT_T3525_INTERVAL: u32 = 60;
/// Default T3540 interval: 10 seconds
pub const DEFAULT_T3540_INTERVAL: u32 = 10;

/// Default T3580 interval: 16 seconds
pub const DEFAULT_T3580_INTERVAL: u32 = 16;
/// Default T3581 interval: 16 seconds
pub const DEFAULT_T3581_INTERVAL: u32 = 16;
/// Default T3582 interval: 16 seconds
pub const DEFAULT_T3582_INTERVAL: u32 = 16;

/// Maximum retry count for T3510 (registration)
pub const MAX_T3510_RETRIES: u32 = 5;
/// Maximum retry count for T3521 (deregistration)
pub const MAX_T3521_RETRIES: u32 = 5;
/// Maximum retry count for T3580 (PDU session establishment)
pub const MAX_T3580_RETRIES: u32 = 5;
/// Maximum retry count for T3581 (PDU session modification)
pub const MAX_T3581_RETRIES: u32 = 5;
/// Maximum retry count for T3582 (PDU session release)
pub const MAX_T3582_RETRIES: u32 = 5;

/// GPRS Timer 3 unit values as defined in 3GPP TS 24.008
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GprsTimer3Unit {
    /// Multiples of 2 seconds
    MultiplesOf2Sec,
    /// Multiples of 1 minute
    MultiplesOf1Min,
    /// Multiples of 10 minutes (decihours)
    MultiplesOf10Min,
    /// Multiples of 1 hour
    MultiplesOf1Hour,
    /// Multiples of 10 hours
    MultiplesOf10Hour,
    /// Multiples of 30 seconds
    MultiplesOf30Sec,
    /// Multiples of 320 hours
    MultiplesOf320Hour,
    /// Timer deactivated
    Deactivated,
}

impl GprsTimer3Unit {
    /// Convert a unit value and timer value to seconds
    pub fn to_seconds(self, value: u8) -> u32 {
        let val = value as u32;
        match self {
            GprsTimer3Unit::MultiplesOf2Sec => val * 2,
            GprsTimer3Unit::MultiplesOf1Min => val * 60,
            GprsTimer3Unit::MultiplesOf10Min => val * 60 * 10,
            GprsTimer3Unit::MultiplesOf1Hour => val * 60 * 60,
            GprsTimer3Unit::MultiplesOf10Hour => val * 60 * 60 * 10,
            GprsTimer3Unit::MultiplesOf30Sec => val * 30,
            GprsTimer3Unit::MultiplesOf320Hour => val * 60 * 60 * 320,
            GprsTimer3Unit::Deactivated => 0,
        }
    }

    /// Create from the 3-bit unit field in GPRS Timer 3 IE
    pub fn from_bits(bits: u8) -> Self {
        match bits & 0x07 {
            0 => GprsTimer3Unit::MultiplesOf2Sec,
            1 => GprsTimer3Unit::MultiplesOf1Min,
            2 => GprsTimer3Unit::MultiplesOf10Min,
            3 => GprsTimer3Unit::MultiplesOf1Hour,
            4 => GprsTimer3Unit::MultiplesOf10Hour,
            5 => GprsTimer3Unit::MultiplesOf30Sec,
            6 => GprsTimer3Unit::MultiplesOf320Hour,
            _ => GprsTimer3Unit::Deactivated,
        }
    }
}

/// GPRS Timer 2 value (simple seconds value)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GprsTimer2 {
    /// Timer value in seconds
    pub value: u32,
}

impl GprsTimer2 {
    /// Create a new GPRS Timer 2 value
    pub fn new(value: u32) -> Self {
        Self { value }
    }
}

/// GPRS Timer 3 value (value with unit)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GprsTimer3 {
    /// Timer unit
    pub unit: GprsTimer3Unit,
    /// Timer value (5 bits, 0-31)
    pub timer_value: u8,
}

impl GprsTimer3 {
    /// Create a new GPRS Timer 3 value
    pub fn new(unit: GprsTimer3Unit, timer_value: u8) -> Self {
        Self {
            unit,
            timer_value: timer_value & 0x1F, // 5 bits
        }
    }

    /// Convert to seconds
    pub fn to_seconds(&self) -> u32 {
        self.unit.to_seconds(self.timer_value)
    }

    /// Create from raw IE byte (unit in bits 7-5, value in bits 4-0)
    pub fn from_byte(byte: u8) -> Self {
        let unit = GprsTimer3Unit::from_bits(byte >> 5);
        let timer_value = byte & 0x1F;
        Self { unit, timer_value }
    }
}

/// UE Timer for NAS procedures
///
/// Tracks timer state including start time, interval, running status,
/// and expiry count for retry logic.
#[derive(Debug)]
pub struct UeTimer {
    /// Timer code (e.g., 3510 for T3510)
    code: u16,
    /// Whether this is a mobility management timer
    is_mm: bool,
    /// Timer interval in seconds
    interval_secs: u32,
    /// When the timer was started
    start_time: Option<Instant>,
    /// Whether the timer is currently running
    is_running: bool,
    /// Number of times the timer has expired
    expiry_count: u32,
}

impl UeTimer {
    /// Create a new UE timer
    ///
    /// # Arguments
    /// * `code` - Timer code (e.g., 3510 for T3510)
    /// * `is_mm` - Whether this is a mobility management timer
    /// * `default_interval_secs` - Default interval in seconds
    pub fn new(code: u16, is_mm: bool, default_interval_secs: u32) -> Self {
        Self {
            code,
            is_mm,
            interval_secs: default_interval_secs,
            start_time: None,
            is_running: false,
            expiry_count: 0,
        }
    }

    /// Start the timer with the default interval
    ///
    /// # Arguments
    /// * `clear_expiry_count` - Whether to reset the expiry count
    pub fn start(&mut self, clear_expiry_count: bool) {
        if clear_expiry_count {
            self.reset_expiry_count();
        }
        self.start_time = Some(Instant::now());
        self.is_running = true;
    }

    /// Start the timer with a GPRS Timer 2 value
    ///
    /// # Arguments
    /// * `timer` - GPRS Timer 2 value
    /// * `clear_expiry_count` - Whether to reset the expiry count
    pub fn start_with_timer2(&mut self, timer: GprsTimer2, clear_expiry_count: bool) {
        if clear_expiry_count {
            self.reset_expiry_count();
        }
        self.interval_secs = timer.value;
        self.start_time = Some(Instant::now());
        self.is_running = true;
    }

    /// Start the timer with a GPRS Timer 3 value
    ///
    /// # Arguments
    /// * `timer` - GPRS Timer 3 value
    /// * `clear_expiry_count` - Whether to reset the expiry count
    pub fn start_with_timer3(&mut self, timer: GprsTimer3, clear_expiry_count: bool) {
        if clear_expiry_count {
            self.reset_expiry_count();
        }
        self.interval_secs = timer.to_seconds();
        self.start_time = Some(Instant::now());
        self.is_running = true;
    }

    /// Stop the timer
    ///
    /// # Arguments
    /// * `clear_expiry_count` - Whether to reset the expiry count
    pub fn stop(&mut self, clear_expiry_count: bool) {
        if clear_expiry_count {
            self.reset_expiry_count();
        }
        if self.is_running {
            self.start_time = None;
            self.is_running = false;
        }
    }

    /// Reset the expiry count to zero
    pub fn reset_expiry_count(&mut self) {
        self.expiry_count = 0;
    }

    /// Check if the timer has expired and update state
    ///
    /// This should be called periodically to check timer expiration.
    /// Returns `true` if the timer just expired on this tick.
    pub fn perform_tick(&mut self) -> bool {
        if self.is_running {
            if let Some(start) = self.start_time {
                let elapsed = start.elapsed();
                let interval = Duration::from_secs(self.interval_secs as u64);

                if elapsed >= interval {
                    // Timer expired
                    self.stop(false);
                    self.expiry_count += 1;
                    return true;
                }
            }
        }
        false
    }

    /// Check if the timer is currently running
    pub fn is_running(&self) -> bool {
        self.is_running
    }

    /// Get the timer code
    pub fn code(&self) -> u16 {
        self.code
    }

    /// Check if this is a mobility management timer
    pub fn is_mm_timer(&self) -> bool {
        self.is_mm
    }

    /// Get the timer interval in seconds
    pub fn interval(&self) -> u32 {
        self.interval_secs
    }

    /// Get the remaining time in seconds
    ///
    /// Returns 0 if the timer is not running or has expired.
    pub fn remaining(&self) -> u32 {
        if !self.is_running {
            return 0;
        }

        if let Some(start) = self.start_time {
            let elapsed_secs = start.elapsed().as_secs() as u32;
            if elapsed_secs >= self.interval_secs {
                return 0;
            }
            return self.interval_secs - elapsed_secs;
        }

        0
    }

    /// Get the number of times the timer has expired
    pub fn expiry_count(&self) -> u32 {
        self.expiry_count
    }
}

impl std::fmt::Display for UeTimer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_running {
            write!(
                f,
                "T{}: rem[{}] int[{}]",
                self.code,
                self.remaining(),
                self.interval_secs
            )
        } else {
            write!(f, "T{}: .", self.code)
        }
    }
}

// ============================================================================
// Timer Expiry Event
// ============================================================================

/// Event generated when a NAS timer expires.
///
/// This event is used to notify the NAS task about timer expirations
/// so it can take appropriate action based on the timer type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimerExpiryEvent {
    /// Timer code (e.g., 3510 for T3510)
    pub timer_code: u16,
    /// Whether this is a mobility management timer
    pub is_mm: bool,
    /// Number of times the timer has expired (for retry logic)
    pub expiry_count: u32,
}

impl TimerExpiryEvent {
    /// Create a new timer expiry event
    pub fn new(timer_code: u16, is_mm: bool, expiry_count: u32) -> Self {
        Self {
            timer_code,
            is_mm,
            expiry_count,
        }
    }
}

// ============================================================================
// NAS Timer Manager
// ============================================================================

/// Manager for all NAS timers used by the UE.
///
/// This struct manages the lifecycle of all MM and SM timers, providing
/// a centralized interface for starting, stopping, and checking timer
/// expirations.
///
/// # Example
///
/// ```
/// use nextgsim_ue::timer::{NasTimerManager, TIMER_T3510};
///
/// let mut timers = NasTimerManager::new();
///
/// // Start registration timer
/// timers.start(TIMER_T3510, true);
///
/// // Check for expirations (call periodically)
/// let expired = timers.perform_tick();
/// for event in expired {
///     println!("Timer T{} expired", event.timer_code);
/// }
/// ```
pub struct NasTimerManager {
    // Mobility Management timers
    /// T3346: Backoff timer for congestion control
    pub t3346: UeTimer,
    /// T3502: PLMN selection retry timer
    pub t3502: UeTimer,
    /// T3510: Registration procedure timer
    pub t3510: UeTimer,
    /// T3511: Registration retry timer
    pub t3511: UeTimer,
    /// T3512: Periodic registration update timer
    pub t3512: UeTimer,
    /// T3516: 5G-AKA authentication timer
    pub t3516: UeTimer,
    /// T3517: Service request timer
    pub t3517: UeTimer,
    /// T3519: SUCI storage timer
    pub t3519: UeTimer,
    /// T3520: Authentication failure timer
    pub t3520: UeTimer,
    /// T3521: Deregistration timer
    pub t3521: UeTimer,
    /// T3525: Service request retry timer
    pub t3525: UeTimer,
    /// T3540: Payload container timer
    pub t3540: UeTimer,

    // Session Management timers
    /// T3580: PDU session establishment timer
    pub t3580: UeTimer,
    /// T3581: PDU session modification timer
    pub t3581: UeTimer,
    /// T3582: PDU session release timer
    pub t3582: UeTimer,
}

impl NasTimerManager {
    /// Create a new NAS timer manager with default timer intervals.
    pub fn new() -> Self {
        Self {
            // MM timers
            t3346: UeTimer::new(TIMER_T3346, true, DEFAULT_T3346_INTERVAL),
            t3502: UeTimer::new(TIMER_T3502, true, DEFAULT_T3502_INTERVAL),
            t3510: UeTimer::new(TIMER_T3510, true, DEFAULT_T3510_INTERVAL),
            t3511: UeTimer::new(TIMER_T3511, true, DEFAULT_T3511_INTERVAL),
            t3512: UeTimer::new(TIMER_T3512, true, DEFAULT_T3512_INTERVAL),
            t3516: UeTimer::new(TIMER_T3516, true, DEFAULT_T3516_INTERVAL),
            t3517: UeTimer::new(TIMER_T3517, true, DEFAULT_T3517_INTERVAL),
            t3519: UeTimer::new(TIMER_T3519, true, DEFAULT_T3519_INTERVAL),
            t3520: UeTimer::new(TIMER_T3520, true, DEFAULT_T3520_INTERVAL),
            t3521: UeTimer::new(TIMER_T3521, true, DEFAULT_T3521_INTERVAL),
            t3525: UeTimer::new(TIMER_T3525, true, DEFAULT_T3525_INTERVAL),
            t3540: UeTimer::new(TIMER_T3540, true, DEFAULT_T3540_INTERVAL),
            // SM timers
            t3580: UeTimer::new(TIMER_T3580, false, DEFAULT_T3580_INTERVAL),
            t3581: UeTimer::new(TIMER_T3581, false, DEFAULT_T3581_INTERVAL),
            t3582: UeTimer::new(TIMER_T3582, false, DEFAULT_T3582_INTERVAL),
        }
    }

    /// Get a mutable reference to a timer by its code.
    ///
    /// Returns `None` if the timer code is not recognized.
    pub fn get_timer_mut(&mut self, code: u16) -> Option<&mut UeTimer> {
        match code {
            TIMER_T3346 => Some(&mut self.t3346),
            TIMER_T3502 => Some(&mut self.t3502),
            TIMER_T3510 => Some(&mut self.t3510),
            TIMER_T3511 => Some(&mut self.t3511),
            TIMER_T3512 => Some(&mut self.t3512),
            TIMER_T3516 => Some(&mut self.t3516),
            TIMER_T3517 => Some(&mut self.t3517),
            TIMER_T3519 => Some(&mut self.t3519),
            TIMER_T3520 => Some(&mut self.t3520),
            TIMER_T3521 => Some(&mut self.t3521),
            TIMER_T3525 => Some(&mut self.t3525),
            TIMER_T3540 => Some(&mut self.t3540),
            TIMER_T3580 => Some(&mut self.t3580),
            TIMER_T3581 => Some(&mut self.t3581),
            TIMER_T3582 => Some(&mut self.t3582),
            _ => None,
        }
    }

    /// Get an immutable reference to a timer by its code.
    ///
    /// Returns `None` if the timer code is not recognized.
    pub fn get_timer(&self, code: u16) -> Option<&UeTimer> {
        match code {
            TIMER_T3346 => Some(&self.t3346),
            TIMER_T3502 => Some(&self.t3502),
            TIMER_T3510 => Some(&self.t3510),
            TIMER_T3511 => Some(&self.t3511),
            TIMER_T3512 => Some(&self.t3512),
            TIMER_T3516 => Some(&self.t3516),
            TIMER_T3517 => Some(&self.t3517),
            TIMER_T3519 => Some(&self.t3519),
            TIMER_T3520 => Some(&self.t3520),
            TIMER_T3521 => Some(&self.t3521),
            TIMER_T3525 => Some(&self.t3525),
            TIMER_T3540 => Some(&self.t3540),
            TIMER_T3580 => Some(&self.t3580),
            TIMER_T3581 => Some(&self.t3581),
            TIMER_T3582 => Some(&self.t3582),
            _ => None,
        }
    }

    /// Start a timer by its code with default interval.
    ///
    /// # Arguments
    /// * `code` - Timer code (e.g., `TIMER_T3510`)
    /// * `clear_expiry_count` - Whether to reset the expiry count
    ///
    /// Returns `true` if the timer was found and started.
    pub fn start(&mut self, code: u16, clear_expiry_count: bool) -> bool {
        if let Some(timer) = self.get_timer_mut(code) {
            timer.start(clear_expiry_count);
            true
        } else {
            false
        }
    }

    /// Start a timer with a GPRS Timer 2 value.
    ///
    /// # Arguments
    /// * `code` - Timer code
    /// * `timer_value` - GPRS Timer 2 value
    /// * `clear_expiry_count` - Whether to reset the expiry count
    ///
    /// Returns `true` if the timer was found and started.
    pub fn start_with_timer2(
        &mut self,
        code: u16,
        timer_value: GprsTimer2,
        clear_expiry_count: bool,
    ) -> bool {
        if let Some(timer) = self.get_timer_mut(code) {
            timer.start_with_timer2(timer_value, clear_expiry_count);
            true
        } else {
            false
        }
    }

    /// Start a timer with a GPRS Timer 3 value.
    ///
    /// # Arguments
    /// * `code` - Timer code
    /// * `timer_value` - GPRS Timer 3 value
    /// * `clear_expiry_count` - Whether to reset the expiry count
    ///
    /// Returns `true` if the timer was found and started.
    pub fn start_with_timer3(
        &mut self,
        code: u16,
        timer_value: GprsTimer3,
        clear_expiry_count: bool,
    ) -> bool {
        if let Some(timer) = self.get_timer_mut(code) {
            timer.start_with_timer3(timer_value, clear_expiry_count);
            true
        } else {
            false
        }
    }

    /// Stop a timer by its code.
    ///
    /// # Arguments
    /// * `code` - Timer code
    /// * `clear_expiry_count` - Whether to reset the expiry count
    ///
    /// Returns `true` if the timer was found and stopped.
    pub fn stop(&mut self, code: u16, clear_expiry_count: bool) -> bool {
        if let Some(timer) = self.get_timer_mut(code) {
            timer.stop(clear_expiry_count);
            true
        } else {
            false
        }
    }

    /// Check if a timer is running.
    ///
    /// Returns `None` if the timer code is not recognized.
    pub fn is_running(&self, code: u16) -> Option<bool> {
        self.get_timer(code).map(UeTimer::is_running)
    }

    /// Get the expiry count for a timer.
    ///
    /// Returns `None` if the timer code is not recognized.
    pub fn expiry_count(&self, code: u16) -> Option<u32> {
        self.get_timer(code).map(UeTimer::expiry_count)
    }

    /// Perform a tick on all timers and return any expiry events.
    ///
    /// This should be called periodically (e.g., every second) to check
    /// for timer expirations.
    ///
    /// Returns a vector of `TimerExpiryEvent` for any timers that expired.
    pub fn perform_tick(&mut self) -> Vec<TimerExpiryEvent> {
        let mut events = Vec::new();

        // Check all MM timers
        let mm_timers = [
            &mut self.t3346,
            &mut self.t3502,
            &mut self.t3510,
            &mut self.t3511,
            &mut self.t3512,
            &mut self.t3516,
            &mut self.t3517,
            &mut self.t3519,
            &mut self.t3520,
            &mut self.t3521,
            &mut self.t3525,
            &mut self.t3540,
        ];

        for timer in mm_timers {
            if timer.perform_tick() {
                events.push(TimerExpiryEvent::new(
                    timer.code(),
                    timer.is_mm_timer(),
                    timer.expiry_count(),
                ));
            }
        }

        // Check all SM timers
        let sm_timers = [&mut self.t3580, &mut self.t3581, &mut self.t3582];

        for timer in sm_timers {
            if timer.perform_tick() {
                events.push(TimerExpiryEvent::new(
                    timer.code(),
                    timer.is_mm_timer(),
                    timer.expiry_count(),
                ));
            }
        }

        events
    }

    /// Stop all running timers.
    ///
    /// # Arguments
    /// * `clear_expiry_counts` - Whether to reset all expiry counts
    pub fn stop_all(&mut self, clear_expiry_counts: bool) {
        self.t3346.stop(clear_expiry_counts);
        self.t3502.stop(clear_expiry_counts);
        self.t3510.stop(clear_expiry_counts);
        self.t3511.stop(clear_expiry_counts);
        self.t3512.stop(clear_expiry_counts);
        self.t3516.stop(clear_expiry_counts);
        self.t3517.stop(clear_expiry_counts);
        self.t3519.stop(clear_expiry_counts);
        self.t3520.stop(clear_expiry_counts);
        self.t3521.stop(clear_expiry_counts);
        self.t3525.stop(clear_expiry_counts);
        self.t3540.stop(clear_expiry_counts);
        self.t3580.stop(clear_expiry_counts);
        self.t3581.stop(clear_expiry_counts);
        self.t3582.stop(clear_expiry_counts);
    }

    /// Get a list of all running timers.
    ///
    /// Returns a vector of timer codes for all currently running timers.
    pub fn running_timers(&self) -> Vec<u16> {
        let mut running = Vec::new();

        let all_timers: [&UeTimer; 15] = [
            &self.t3346,
            &self.t3502,
            &self.t3510,
            &self.t3511,
            &self.t3512,
            &self.t3516,
            &self.t3517,
            &self.t3519,
            &self.t3520,
            &self.t3521,
            &self.t3525,
            &self.t3540,
            &self.t3580,
            &self.t3581,
            &self.t3582,
        ];

        for timer in all_timers {
            if timer.is_running() {
                running.push(timer.code());
            }
        }

        running
    }

    /// Get status information for all timers.
    ///
    /// Returns a vector of (code, `is_running`, `remaining_secs`) tuples.
    pub fn status(&self) -> Vec<(u16, bool, u32)> {
        let all_timers: [&UeTimer; 15] = [
            &self.t3346,
            &self.t3502,
            &self.t3510,
            &self.t3511,
            &self.t3512,
            &self.t3516,
            &self.t3517,
            &self.t3519,
            &self.t3520,
            &self.t3521,
            &self.t3525,
            &self.t3540,
            &self.t3580,
            &self.t3581,
            &self.t3582,
        ];

        all_timers
            .iter()
            .map(|t| (t.code(), t.is_running(), t.remaining()))
            .collect()
    }
}

impl Default for NasTimerManager {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for NasTimerManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let running = self.running_timers();
        if running.is_empty() {
            write!(f, "NasTimerManager {{ no running timers }}")
        } else {
            write!(f, "NasTimerManager {{ running: {running:?} }}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    fn test_timer_creation() {
        let timer = UeTimer::new(3510, true, 15);
        assert_eq!(timer.code(), 3510);
        assert!(timer.is_mm_timer());
        assert_eq!(timer.interval(), 15);
        assert!(!timer.is_running());
        assert_eq!(timer.expiry_count(), 0);
    }

    #[test]
    fn test_timer_start_stop() {
        let mut timer = UeTimer::new(3511, true, 10);

        timer.start(true);
        assert!(timer.is_running());
        assert!(timer.remaining() <= 10);

        timer.stop(true);
        assert!(!timer.is_running());
        assert_eq!(timer.remaining(), 0);
    }

    #[test]
    fn test_timer_expiry() {
        let mut timer = UeTimer::new(3512, true, 1); // 1 second timer

        timer.start(true);
        assert!(timer.is_running());
        assert_eq!(timer.expiry_count(), 0);

        // Wait for timer to expire
        sleep(Duration::from_millis(1100));

        let expired = timer.perform_tick();
        assert!(expired);
        assert!(!timer.is_running());
        assert_eq!(timer.expiry_count(), 1);
    }

    #[test]
    fn test_timer_expiry_count_preserved() {
        let mut timer = UeTimer::new(3510, true, 1);

        // First expiry
        timer.start(true);
        sleep(Duration::from_millis(1100));
        timer.perform_tick();
        assert_eq!(timer.expiry_count(), 1);

        // Second expiry - don't clear count
        timer.start(false);
        sleep(Duration::from_millis(1100));
        timer.perform_tick();
        assert_eq!(timer.expiry_count(), 2);

        // Third start - clear count
        timer.start(true);
        assert_eq!(timer.expiry_count(), 0);
    }

    #[test]
    fn test_gprs_timer2() {
        let mut timer = UeTimer::new(3510, true, 15);
        let gprs_timer = GprsTimer2::new(30);

        timer.start_with_timer2(gprs_timer, true);
        assert!(timer.is_running());
        assert_eq!(timer.interval(), 30);
    }

    #[test]
    fn test_gprs_timer3_units() {
        // Test each unit type
        assert_eq!(GprsTimer3Unit::MultiplesOf2Sec.to_seconds(5), 10);
        assert_eq!(GprsTimer3Unit::MultiplesOf30Sec.to_seconds(2), 60);
        assert_eq!(GprsTimer3Unit::MultiplesOf1Min.to_seconds(5), 300);
        assert_eq!(GprsTimer3Unit::MultiplesOf10Min.to_seconds(3), 1800);
        assert_eq!(GprsTimer3Unit::MultiplesOf1Hour.to_seconds(2), 7200);
        assert_eq!(GprsTimer3Unit::MultiplesOf10Hour.to_seconds(1), 36000);
        assert_eq!(GprsTimer3Unit::MultiplesOf320Hour.to_seconds(1), 1152000);
        assert_eq!(GprsTimer3Unit::Deactivated.to_seconds(5), 0);
    }

    #[test]
    fn test_gprs_timer3() {
        let mut timer = UeTimer::new(3512, true, 15);
        let gprs_timer = GprsTimer3::new(GprsTimer3Unit::MultiplesOf1Min, 5);

        assert_eq!(gprs_timer.to_seconds(), 300);

        timer.start_with_timer3(gprs_timer, true);
        assert!(timer.is_running());
        assert_eq!(timer.interval(), 300);
    }

    #[test]
    fn test_gprs_timer3_from_byte() {
        // Unit = 1 (1 min), value = 10 -> byte = 0b001_01010 = 0x2A
        let timer = GprsTimer3::from_byte(0x2A);
        assert_eq!(timer.unit, GprsTimer3Unit::MultiplesOf1Min);
        assert_eq!(timer.timer_value, 10);
        assert_eq!(timer.to_seconds(), 600);
    }

    #[test]
    fn test_timer_display() {
        let mut timer = UeTimer::new(3510, true, 15);
        assert_eq!(format!("{timer}"), "T3510: .");

        timer.start(true);
        let display = format!("{timer}");
        assert!(display.starts_with("T3510: rem["));
        assert!(display.contains("] int[15]"));
    }

    #[test]
    fn test_timer_not_expired_before_interval() {
        let mut timer = UeTimer::new(3510, true, 10);
        timer.start(true);

        // Immediately check - should not be expired
        let expired = timer.perform_tick();
        assert!(!expired);
        assert!(timer.is_running());
        assert_eq!(timer.expiry_count(), 0);
    }

    #[test]
    fn test_sm_timer() {
        let timer = UeTimer::new(3580, false, 30);
        assert!(!timer.is_mm_timer());
        assert_eq!(timer.code(), 3580);
    }

    // ========================================================================
    // NAS Timer Manager Tests
    // ========================================================================

    #[test]
    fn test_timer_manager_creation() {
        let manager = NasTimerManager::new();

        // Verify all timers are created with correct codes
        assert_eq!(manager.t3346.code(), TIMER_T3346);
        assert_eq!(manager.t3510.code(), TIMER_T3510);
        assert_eq!(manager.t3512.code(), TIMER_T3512);
        assert_eq!(manager.t3580.code(), TIMER_T3580);

        // Verify default intervals
        assert_eq!(manager.t3510.interval(), DEFAULT_T3510_INTERVAL);
        assert_eq!(manager.t3512.interval(), DEFAULT_T3512_INTERVAL);
        assert_eq!(manager.t3580.interval(), DEFAULT_T3580_INTERVAL);

        // Verify MM vs SM classification
        assert!(manager.t3510.is_mm_timer());
        assert!(!manager.t3580.is_mm_timer());
    }

    #[test]
    fn test_timer_manager_get_timer() {
        let mut manager = NasTimerManager::new();

        // Test get_timer
        assert!(manager.get_timer(TIMER_T3510).is_some());
        assert!(manager.get_timer(TIMER_T3580).is_some());
        assert!(manager.get_timer(9999).is_none());

        // Test get_timer_mut
        assert!(manager.get_timer_mut(TIMER_T3510).is_some());
        assert!(manager.get_timer_mut(9999).is_none());
    }

    #[test]
    fn test_timer_manager_start_stop() {
        let mut manager = NasTimerManager::new();

        // Start timer
        assert!(manager.start(TIMER_T3510, true));
        assert_eq!(manager.is_running(TIMER_T3510), Some(true));

        // Stop timer
        assert!(manager.stop(TIMER_T3510, true));
        assert_eq!(manager.is_running(TIMER_T3510), Some(false));

        // Invalid timer code
        assert!(!manager.start(9999, true));
        assert!(!manager.stop(9999, true));
        assert_eq!(manager.is_running(9999), None);
    }

    #[test]
    fn test_timer_manager_start_with_gprs_timer2() {
        let mut manager = NasTimerManager::new();
        let gprs_timer = GprsTimer2::new(30);

        assert!(manager.start_with_timer2(TIMER_T3346, gprs_timer, true));
        assert_eq!(manager.is_running(TIMER_T3346), Some(true));
        assert_eq!(manager.t3346.interval(), 30);
    }

    #[test]
    fn test_timer_manager_start_with_gprs_timer3() {
        let mut manager = NasTimerManager::new();
        let gprs_timer = GprsTimer3::new(GprsTimer3Unit::MultiplesOf1Min, 5);

        assert!(manager.start_with_timer3(TIMER_T3512, gprs_timer, true));
        assert_eq!(manager.is_running(TIMER_T3512), Some(true));
        assert_eq!(manager.t3512.interval(), 300); // 5 minutes
    }

    #[test]
    fn test_timer_manager_running_timers() {
        let mut manager = NasTimerManager::new();

        // No timers running initially
        assert!(manager.running_timers().is_empty());

        // Start some timers
        manager.start(TIMER_T3510, true);
        manager.start(TIMER_T3512, true);
        manager.start(TIMER_T3580, true);

        let running = manager.running_timers();
        assert_eq!(running.len(), 3);
        assert!(running.contains(&TIMER_T3510));
        assert!(running.contains(&TIMER_T3512));
        assert!(running.contains(&TIMER_T3580));
    }

    #[test]
    fn test_timer_manager_stop_all() {
        let mut manager = NasTimerManager::new();

        // Start multiple timers
        manager.start(TIMER_T3510, true);
        manager.start(TIMER_T3512, true);
        manager.start(TIMER_T3580, true);
        assert_eq!(manager.running_timers().len(), 3);

        // Stop all
        manager.stop_all(true);
        assert!(manager.running_timers().is_empty());
    }

    #[test]
    fn test_timer_manager_status() {
        let mut manager = NasTimerManager::new();

        manager.start(TIMER_T3510, true);

        let status = manager.status();
        assert_eq!(status.len(), 15); // All 15 timers

        // Find T3510 in status
        let t3510_status = status.iter().find(|(code, _, _)| *code == TIMER_T3510);
        assert!(t3510_status.is_some());
        let (_, is_running, _) = t3510_status.unwrap();
        assert!(*is_running);
    }

    #[test]
    fn test_timer_manager_expiry_count() {
        let manager = NasTimerManager::new();

        assert_eq!(manager.expiry_count(TIMER_T3510), Some(0));
        assert_eq!(manager.expiry_count(9999), None);
    }

    #[test]
    fn test_timer_manager_perform_tick_no_expiry() {
        let mut manager = NasTimerManager::new();

        // Start a timer with long interval
        manager.start(TIMER_T3510, true);

        // Immediate tick should not expire
        let events = manager.perform_tick();
        assert!(events.is_empty());
    }

    #[test]
    fn test_timer_manager_perform_tick_with_expiry() {
        let mut manager = NasTimerManager::new();

        // Use a 1-second timer for testing
        manager.t3510 = UeTimer::new(TIMER_T3510, true, 1);
        manager.start(TIMER_T3510, true);

        // Wait for expiry
        sleep(Duration::from_millis(1100));

        let events = manager.perform_tick();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].timer_code, TIMER_T3510);
        assert!(events[0].is_mm);
        assert_eq!(events[0].expiry_count, 1);
    }

    #[test]
    fn test_timer_expiry_event() {
        let event = TimerExpiryEvent::new(TIMER_T3510, true, 2);
        assert_eq!(event.timer_code, TIMER_T3510);
        assert!(event.is_mm);
        assert_eq!(event.expiry_count, 2);
    }

    #[test]
    fn test_timer_manager_debug() {
        let mut manager = NasTimerManager::new();

        // No running timers
        let debug_str = format!("{manager:?}");
        assert!(debug_str.contains("no running timers"));

        // With running timers
        manager.start(TIMER_T3510, true);
        let debug_str = format!("{manager:?}");
        assert!(debug_str.contains("running"));
        assert!(debug_str.contains("3510"));
    }

    #[test]
    fn test_timer_manager_default() {
        let manager = NasTimerManager::default();
        assert_eq!(manager.t3510.code(), TIMER_T3510);
    }
}
