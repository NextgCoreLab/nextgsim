//! XR (Extended Reality) Traffic Modeling and `QoS` (Rel-18, TS 26.928)
//!
//! Implements XR-specific traffic patterns, PDU Set handling, `QoS` flow
//! management with 5QI 82-85, and C-DRX power saving for XR devices.

use std::collections::VecDeque;

// ============================================================================
// XR 5QI Definitions (3GPP TS 23.501 Table 5.7.4-1, Rel-18 additions)
// ============================================================================

/// XR-specific 5QI values added in Rel-18.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Xr5Qi {
    /// 5QI 82: XR cloud rendering DL video (GBR, 10ms PDB)
    CloudRenderingDl = 82,
    /// 5QI 83: XR cloud rendering UL pose/control (GBR, 5ms PDB)
    PoseControlUl = 83,
    /// 5QI 84: XR split rendering DL (GBR, 15ms PDB)
    SplitRenderingDl = 84,
    /// 5QI 85: XR haptic feedback (GBR, 5ms PDB)
    HapticFeedback = 85,
}

impl Xr5Qi {
    /// Returns the Packet Delay Budget in milliseconds.
    pub fn pdb_ms(&self) -> u32 {
        match self {
            Xr5Qi::CloudRenderingDl => 10,
            Xr5Qi::PoseControlUl => 5,
            Xr5Qi::SplitRenderingDl => 15,
            Xr5Qi::HapticFeedback => 5,
        }
    }

    /// Returns the Packet Error Rate.
    pub fn per(&self) -> f64 {
        match self {
            Xr5Qi::CloudRenderingDl => 1e-3,
            Xr5Qi::PoseControlUl => 1e-4,
            Xr5Qi::SplitRenderingDl => 1e-3,
            Xr5Qi::HapticFeedback => 1e-5,
        }
    }

    /// Returns the default MDBV (Maximum Data Burst Volume) in bytes.
    pub fn default_mdbv_bytes(&self) -> u32 {
        match self {
            Xr5Qi::CloudRenderingDl => 60_000,
            Xr5Qi::PoseControlUl => 1_500,
            Xr5Qi::SplitRenderingDl => 30_000,
            Xr5Qi::HapticFeedback => 500,
        }
    }

    /// Returns whether this 5QI is GBR (Guaranteed Bit Rate).
    pub fn is_gbr(&self) -> bool {
        true // All XR 5QIs are GBR
    }

    /// Returns the priority level (1 = highest).
    pub fn priority_level(&self) -> u8 {
        match self {
            Xr5Qi::HapticFeedback => 19,
            Xr5Qi::PoseControlUl => 20,
            Xr5Qi::CloudRenderingDl => 21,
            Xr5Qi::SplitRenderingDl => 22,
        }
    }

    /// Creates from a raw 5QI value.
    pub fn from_value(val: u8) -> Option<Self> {
        match val {
            82 => Some(Xr5Qi::CloudRenderingDl),
            83 => Some(Xr5Qi::PoseControlUl),
            84 => Some(Xr5Qi::SplitRenderingDl),
            85 => Some(Xr5Qi::HapticFeedback),
            _ => None,
        }
    }
}

// ============================================================================
// XR Traffic Model (periodic + jitter per TS 26.928)
// ============================================================================

/// XR frame descriptor with PDU Set information.
#[derive(Debug, Clone)]
pub struct XrFrame {
    /// Frame sequence number
    pub sequence: u64,
    /// Timestamp when the frame was generated (ms)
    pub timestamp_ms: u64,
    /// Frame size in bytes
    pub size_bytes: u32,
    /// PDU Set ID (groups PDUs belonging to same video frame)
    pub pdu_set_id: u32,
    /// Number of PDUs in this set
    pub pdu_count: u16,
    /// Whether this is an I-frame (keyframe)
    pub is_keyframe: bool,
    /// Frame importance (0.0-1.0, used for PDU Set-level `QoS`)
    pub importance: f32,
}

/// XR traffic generator with periodic + jitter model.
#[derive(Debug)]
pub struct XrTrafficModel {
    /// Target frame rate (fps)
    target_fps: u32,
    /// Frame interval (ms)
    frame_interval_ms: f64,
    /// Jitter standard deviation (ms)
    jitter_std_ms: f64,
    /// Mean frame size for I-frames (bytes)
    i_frame_size: u32,
    /// Mean frame size for P-frames (bytes)
    p_frame_size: u32,
    /// I-frame interval (every N frames)
    i_frame_interval: u32,
    /// Current frame sequence counter
    sequence: u64,
    /// Current PDU Set ID counter
    pdu_set_counter: u32,
    /// Current timestamp (ms)
    current_time_ms: u64,
    /// Frame buffer for scheduled frames
    #[allow(dead_code)]
    frame_buffer: VecDeque<XrFrame>,
}

impl XrTrafficModel {
    /// Creates a new XR traffic model.
    ///
    /// # Arguments
    /// * `fps` - Target frame rate (e.g., 60, 90, 120)
    /// * `jitter_ms` - Jitter standard deviation in ms
    /// * `i_frame_size` - Mean I-frame size in bytes
    /// * `p_frame_size` - Mean P-frame size in bytes
    pub fn new(fps: u32, jitter_ms: f64, i_frame_size: u32, p_frame_size: u32) -> Self {
        Self {
            target_fps: fps,
            frame_interval_ms: 1000.0 / fps as f64,
            jitter_std_ms: jitter_ms,
            i_frame_size,
            p_frame_size,
            i_frame_interval: fps, // One I-frame per second
            sequence: 0,
            pdu_set_counter: 0,
            current_time_ms: 0,
            frame_buffer: VecDeque::new(),
        }
    }

    /// Creates a cloud rendering XR traffic model (90fps, 60KB I-frame).
    pub fn cloud_rendering() -> Self {
        Self::new(90, 2.0, 60_000, 15_000)
    }

    /// Creates a split rendering XR traffic model (60fps, smaller frames).
    pub fn split_rendering() -> Self {
        Self::new(60, 3.0, 30_000, 8_000)
    }

    /// Creates a pose/control uplink traffic model (high rate, small packets).
    pub fn pose_control() -> Self {
        Self::new(120, 1.0, 200, 200)
    }

    /// Generates the next frame in the traffic model.
    pub fn generate_frame(&mut self) -> XrFrame {
        let is_keyframe = self.sequence % self.i_frame_interval as u64 == 0;
        let base_size = if is_keyframe {
            self.i_frame_size
        } else {
            self.p_frame_size
        };

        // Simple deterministic jitter based on sequence
        let jitter = self.jitter_std_ms
            * ((self.sequence as f64 * std::f64::consts::E).sin() * 0.5);
        let timestamp = self.current_time_ms
            + (self.frame_interval_ms + jitter).max(0.0) as u64;

        // Calculate PDU count based on frame size (assume 1400-byte MTU PDUs)
        let pdu_count = ((base_size as f64 / 1400.0).ceil() as u16).max(1);

        let frame = XrFrame {
            sequence: self.sequence,
            timestamp_ms: timestamp,
            size_bytes: base_size,
            pdu_set_id: self.pdu_set_counter,
            pdu_count,
            is_keyframe,
            importance: if is_keyframe { 1.0 } else { 0.5 },
        };

        self.sequence += 1;
        self.pdu_set_counter += 1;
        self.current_time_ms = timestamp;

        frame
    }

    /// Generates N frames and returns them.
    pub fn generate_frames(&mut self, count: usize) -> Vec<XrFrame> {
        (0..count).map(|_| self.generate_frame()).collect()
    }

    /// Returns the target frame rate.
    pub fn target_fps(&self) -> u32 {
        self.target_fps
    }

    /// Returns the frame interval in ms.
    pub fn frame_interval_ms(&self) -> f64 {
        self.frame_interval_ms
    }
}

// ============================================================================
// PDU Set Handler (groups PDUs for QoS treatment per TS 23.501 Rel-18)
// ============================================================================

/// PDU Set state for tracking frame-level `QoS`.
#[derive(Debug, Clone)]
pub struct PduSet {
    /// PDU Set identifier
    pub set_id: u32,
    /// Total PDUs expected in this set
    pub total_pdus: u16,
    /// PDUs received so far
    pub received_pdus: u16,
    /// Total bytes received
    pub received_bytes: u32,
    /// Timestamp of first PDU in set (ms)
    pub first_pdu_time_ms: u64,
    /// Timestamp of last PDU in set (ms)
    pub last_pdu_time_ms: u64,
    /// Set importance for dropping decisions
    pub importance: f32,
    /// Whether the set is complete
    pub complete: bool,
}

impl PduSet {
    /// Creates a new PDU Set tracker.
    pub fn new(set_id: u32, total_pdus: u16, importance: f32) -> Self {
        Self {
            set_id,
            total_pdus,
            received_pdus: 0,
            received_bytes: 0,
            first_pdu_time_ms: 0,
            last_pdu_time_ms: 0,
            importance,
            complete: false,
        }
    }

    /// Records a PDU arrival.
    pub fn record_pdu(&mut self, size_bytes: u32, timestamp_ms: u64) {
        if self.received_pdus == 0 {
            self.first_pdu_time_ms = timestamp_ms;
        }
        self.received_pdus += 1;
        self.received_bytes += size_bytes;
        self.last_pdu_time_ms = timestamp_ms;
        if self.received_pdus >= self.total_pdus {
            self.complete = true;
        }
    }

    /// Returns the PDU Set delivery delay (first to last PDU).
    pub fn delivery_delay_ms(&self) -> u64 {
        self.last_pdu_time_ms.saturating_sub(self.first_pdu_time_ms)
    }

    /// Returns the loss ratio within the set.
    pub fn loss_ratio(&self) -> f64 {
        if self.total_pdus == 0 {
            return 0.0;
        }
        1.0 - (self.received_pdus as f64 / self.total_pdus as f64)
    }
}

/// PDU Set manager for handling multiple concurrent PDU Sets.
#[derive(Debug)]
pub struct PduSetManager {
    /// Active PDU Sets being tracked
    active_sets: Vec<PduSet>,
    /// Maximum concurrent sets to track
    max_active: usize,
    /// Total sets completed
    pub completed_count: u64,
    /// Total sets that were incomplete (partial loss)
    pub incomplete_count: u64,
}

impl PduSetManager {
    /// Creates a new PDU Set manager.
    pub fn new(max_active: usize) -> Self {
        Self {
            active_sets: Vec::with_capacity(max_active),
            max_active,
            completed_count: 0,
            incomplete_count: 0,
        }
    }

    /// Starts tracking a new PDU Set.
    pub fn start_set(&mut self, set_id: u32, total_pdus: u16, importance: f32) {
        // Evict oldest if at capacity
        if self.active_sets.len() >= self.max_active {
            if let Some(evicted) = self.active_sets.first() {
                if evicted.complete {
                    self.completed_count += 1;
                } else {
                    self.incomplete_count += 1;
                }
            }
            self.active_sets.remove(0);
        }
        self.active_sets
            .push(PduSet::new(set_id, total_pdus, importance));
    }

    /// Records a PDU arrival for a given set.
    pub fn record_pdu(&mut self, set_id: u32, size_bytes: u32, timestamp_ms: u64) {
        if let Some(set) = self.active_sets.iter_mut().find(|s| s.set_id == set_id) {
            set.record_pdu(size_bytes, timestamp_ms);
            if set.complete {
                self.completed_count += 1;
            }
        }
    }

    /// Returns the number of active (incomplete) sets.
    pub fn active_count(&self) -> usize {
        self.active_sets.iter().filter(|s| !s.complete).count()
    }

    /// Returns average PDU Set delivery delay for completed sets.
    pub fn avg_delivery_delay_ms(&self) -> f64 {
        let completed: Vec<_> = self.active_sets.iter().filter(|s| s.complete).collect();
        if completed.is_empty() {
            return 0.0;
        }
        let total: u64 = completed.iter().map(|s| s.delivery_delay_ms()).sum();
        total as f64 / completed.len() as f64
    }
}

// ============================================================================
// C-DRX (Connected-mode DRX) for XR Power Saving
// ============================================================================

/// C-DRX state for XR power saving.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CdrxState {
    /// Active monitoring (radio on)
    Active,
    /// On-duration timer running (short monitoring window)
    OnDuration,
    /// Inactivity timer running (waiting for data)
    InactivityTimer,
    /// DRX sleep (radio off, power saving)
    Sleep,
}

/// C-DRX controller for XR-aware power saving.
#[derive(Debug)]
pub struct XrCdrxController {
    /// DRX cycle length (ms)
    cycle_ms: u32,
    /// On-duration timer length (ms)
    on_duration_ms: u32,
    /// Inactivity timer length (ms)
    inactivity_timer_ms: u32,
    /// Current state
    state: CdrxState,
    /// Time in current state (ms)
    state_time_ms: u32,
    /// XR frame-aligned wake-up enabled
    #[allow(dead_code)]
    frame_aligned: bool,
    /// Next expected frame time (ms, for frame-aligned wake-up)
    next_frame_time_ms: u64,
    /// Total active time (ms, for power metrics)
    total_active_ms: u64,
    /// Total sleep time (ms, for power metrics)
    total_sleep_ms: u64,
}

impl XrCdrxController {
    /// Creates a new C-DRX controller for XR.
    ///
    /// # Arguments
    /// * `cycle_ms` - DRX cycle length in ms
    /// * `on_duration_ms` - On-duration timer in ms
    /// * `inactivity_ms` - Inactivity timer in ms
    /// * `frame_aligned` - Whether to align wake-up with XR frame timing
    pub fn new(cycle_ms: u32, on_duration_ms: u32, inactivity_ms: u32, frame_aligned: bool) -> Self {
        Self {
            cycle_ms,
            on_duration_ms,
            inactivity_timer_ms: inactivity_ms,
            state: CdrxState::Active,
            state_time_ms: 0,
            frame_aligned,
            next_frame_time_ms: 0,
            total_active_ms: 0,
            total_sleep_ms: 0,
        }
    }

    /// Creates an XR-optimized C-DRX configuration.
    /// Uses short cycle aligned to typical XR frame timing.
    pub fn xr_optimized(fps: u32) -> Self {
        let frame_interval = 1000 / fps;
        Self::new(
            frame_interval, // Cycle = frame interval
            frame_interval / 2, // On-duration = half frame interval
            frame_interval / 4, // Inactivity = quarter frame interval
            true,
        )
    }

    /// Advances the C-DRX state machine by `delta_ms`.
    pub fn tick(&mut self, delta_ms: u32) {
        self.state_time_ms += delta_ms;

        match self.state {
            CdrxState::Active => {
                self.total_active_ms += delta_ms as u64;
            }
            CdrxState::OnDuration => {
                self.total_active_ms += delta_ms as u64;
                if self.state_time_ms >= self.on_duration_ms {
                    self.transition(CdrxState::Sleep);
                }
            }
            CdrxState::InactivityTimer => {
                self.total_active_ms += delta_ms as u64;
                if self.state_time_ms >= self.inactivity_timer_ms {
                    self.transition(CdrxState::Sleep);
                }
            }
            CdrxState::Sleep => {
                self.total_sleep_ms += delta_ms as u64;
                if self.state_time_ms >= self.cycle_ms {
                    self.transition(CdrxState::OnDuration);
                }
            }
        }
    }

    /// Notifies data arrival, restarting inactivity timer.
    pub fn on_data_arrival(&mut self) {
        match self.state {
            CdrxState::Sleep => {
                self.transition(CdrxState::Active);
            }
            CdrxState::OnDuration | CdrxState::InactivityTimer => {
                self.transition(CdrxState::InactivityTimer);
            }
            CdrxState::Active => {}
        }
    }

    /// Sets the next expected frame time for frame-aligned wake-up.
    pub fn set_next_frame_time(&mut self, time_ms: u64) {
        self.next_frame_time_ms = time_ms;
    }

    /// Triggers transition to no-data state (start DRX).
    pub fn start_drx(&mut self) {
        if self.state == CdrxState::Active {
            self.transition(CdrxState::InactivityTimer);
        }
    }

    fn transition(&mut self, new_state: CdrxState) {
        self.state = new_state;
        self.state_time_ms = 0;
    }

    /// Returns the current C-DRX state.
    pub fn state(&self) -> CdrxState {
        self.state
    }

    /// Returns the power saving ratio (sleep time / total time).
    pub fn power_saving_ratio(&self) -> f64 {
        let total = self.total_active_ms + self.total_sleep_ms;
        if total == 0 {
            return 0.0;
        }
        self.total_sleep_ms as f64 / total as f64
    }

    /// Returns whether the UE is currently monitoring PDCCH.
    pub fn is_monitoring(&self) -> bool {
        matches!(
            self.state,
            CdrxState::Active | CdrxState::OnDuration | CdrxState::InactivityTimer
        )
    }
}

// ============================================================================
// XR QoS Flow Manager
// ============================================================================

/// XR `QoS` flow with 5QI and PDU Set-aware scheduling.
#[derive(Debug)]
pub struct XrQosFlow {
    /// `QoS` Flow Identifier (QFI)
    pub qfi: u8,
    /// XR 5QI for this flow
    pub xr_5qi: Xr5Qi,
    /// Guaranteed Flow Bit Rate (kbps)
    pub gfbr_kbps: u64,
    /// Maximum Flow Bit Rate (kbps)
    pub mfbr_kbps: u64,
    /// PDU Set manager for this flow
    pub pdu_set_mgr: PduSetManager,
    /// Packets sent
    pub packets_sent: u64,
    /// Packets dropped (deadline missed)
    pub packets_dropped: u64,
    /// Total bytes sent
    pub bytes_sent: u64,
}

impl XrQosFlow {
    /// Creates a new XR `QoS` flow.
    pub fn new(qfi: u8, xr_5qi: Xr5Qi, gfbr_kbps: u64, mfbr_kbps: u64) -> Self {
        Self {
            qfi,
            xr_5qi,
            gfbr_kbps,
            mfbr_kbps,
            pdu_set_mgr: PduSetManager::new(32),
            packets_sent: 0,
            packets_dropped: 0,
            bytes_sent: 0,
        }
    }

    /// Creates a `QoS` flow for cloud rendering DL video.
    pub fn cloud_rendering_dl(qfi: u8) -> Self {
        Self::new(qfi, Xr5Qi::CloudRenderingDl, 30_000, 100_000)
    }

    /// Creates a `QoS` flow for pose/control UL data.
    pub fn pose_control_ul(qfi: u8) -> Self {
        Self::new(qfi, Xr5Qi::PoseControlUl, 500, 2_000)
    }

    /// Creates a `QoS` flow for haptic feedback.
    pub fn haptic(qfi: u8) -> Self {
        Self::new(qfi, Xr5Qi::HapticFeedback, 100, 500)
    }

    /// Records a packet transmission.
    pub fn record_tx(&mut self, size_bytes: u32) {
        self.packets_sent += 1;
        self.bytes_sent += size_bytes as u64;
    }

    /// Records a packet drop.
    pub fn record_drop(&mut self) {
        self.packets_dropped += 1;
    }

    /// Returns the packet loss rate.
    pub fn packet_loss_rate(&self) -> f64 {
        let total = self.packets_sent + self.packets_dropped;
        if total == 0 {
            return 0.0;
        }
        self.packets_dropped as f64 / total as f64
    }

    /// Returns true if the flow meets its `QoS` requirements.
    pub fn meets_qos(&self) -> bool {
        self.packet_loss_rate() <= self.xr_5qi.per()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xr_5qi_properties() {
        assert_eq!(Xr5Qi::CloudRenderingDl.pdb_ms(), 10);
        assert_eq!(Xr5Qi::PoseControlUl.pdb_ms(), 5);
        assert_eq!(Xr5Qi::HapticFeedback.pdb_ms(), 5);
        assert!(Xr5Qi::CloudRenderingDl.is_gbr());
        assert_eq!(Xr5Qi::from_value(82), Some(Xr5Qi::CloudRenderingDl));
        assert_eq!(Xr5Qi::from_value(99), None);
    }

    #[test]
    fn test_xr_traffic_model() {
        let mut model = XrTrafficModel::cloud_rendering();
        assert_eq!(model.target_fps(), 90);

        let frames = model.generate_frames(100);
        assert_eq!(frames.len(), 100);
        // First frame should be a keyframe (sequence 0 % 90 == 0)
        assert!(frames[0].is_keyframe);
        assert_eq!(frames[0].size_bytes, 60_000);
        // Second frame is P-frame
        assert!(!frames[1].is_keyframe);
        assert_eq!(frames[1].size_bytes, 15_000);
        // Sequence should be monotonic
        for i in 1..frames.len() {
            assert!(frames[i].sequence > frames[i - 1].sequence);
        }
    }

    #[test]
    fn test_pdu_set_tracking() {
        let mut set = PduSet::new(1, 10, 0.8);
        assert!(!set.complete);

        for i in 0..10 {
            set.record_pdu(1400, 100 + i);
        }
        assert!(set.complete);
        assert_eq!(set.received_pdus, 10);
        assert_eq!(set.received_bytes, 14_000);
        assert_eq!(set.delivery_delay_ms(), 9);
        assert_eq!(set.loss_ratio(), 0.0);
    }

    #[test]
    fn test_pdu_set_manager() {
        let mut mgr = PduSetManager::new(16);
        mgr.start_set(1, 5, 0.9);
        for i in 0..5 {
            mgr.record_pdu(1, 1400, 100 + i);
        }
        assert_eq!(mgr.completed_count, 1);
        assert_eq!(mgr.active_count(), 0);
    }

    #[test]
    fn test_cdrx_power_saving() {
        let mut cdrx = XrCdrxController::xr_optimized(90);
        assert_eq!(cdrx.state(), CdrxState::Active);
        assert!(cdrx.is_monitoring());

        // Start DRX
        cdrx.start_drx();
        assert_eq!(cdrx.state(), CdrxState::InactivityTimer);

        // Advance past inactivity timer
        cdrx.tick(10);
        assert_eq!(cdrx.state(), CdrxState::Sleep);
        assert!(!cdrx.is_monitoring());

        // Spend some time in sleep state
        cdrx.tick(5);
        assert_eq!(cdrx.state(), CdrxState::Sleep);

        // Data arrives during sleep
        cdrx.on_data_arrival();
        assert_eq!(cdrx.state(), CdrxState::Active);
        assert!(cdrx.is_monitoring());

        // Check power saving ratio is reasonable
        assert!(cdrx.power_saving_ratio() > 0.0);
    }

    #[test]
    fn test_xr_qos_flow() {
        let mut flow = XrQosFlow::cloud_rendering_dl(1);
        assert_eq!(flow.qfi, 1);
        assert_eq!(flow.xr_5qi, Xr5Qi::CloudRenderingDl);

        // Simulate some traffic
        for _ in 0..1000 {
            flow.record_tx(1400);
        }
        assert_eq!(flow.packets_sent, 1000);
        assert!(flow.meets_qos());

        // Drop a packet
        flow.record_drop();
        assert_eq!(flow.packets_dropped, 1);
        assert!(flow.packet_loss_rate() < 0.01);
    }
}
