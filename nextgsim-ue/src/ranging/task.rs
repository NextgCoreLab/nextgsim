//! Ranging Task for UE - UE-to-UE distance measurement and carrier phase positioning
//!
//! **Scaffold, not wired end-to-end.** Models the Rel-18 ranging concepts of
//! TS 23.586 (RTT ranging, carrier-phase measurement with multi-frequency
//! ambiguity resolution, and LMF result reporting), but no `RangingMessage`
//! producer, SL-PRS stimulus, or UE->LMF (SLPP/RSPP) transport is wired — so
//! the measurement handlers below are not reached at runtime.

use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::tasks::{RangingMessage, Task, TaskMessage, UeTaskBase};

/// The line the UE binary emits when it spawns the ranging task.
///
/// A constant rather than a literal at the `info!` site because the site is in
/// `main.rs`, inside a task closure no test can reach. Naming it here is what
/// lets `the_startup_lines_make_no_ts_23_586_compliance_claim` pin the
/// binary's wording too: the only way to change what the binary logs is to
/// change this string, and the test reads this string.
///
/// It must not claim TS 23.586 compliance. The pipeline behind it is dead —
/// see the module docs — so a spec citation here would be read as a
/// capability. Issue #55.
pub const SPAWN_LOG: &str =
    "Ranging task spawned (scaffold: no stimulus producer or UE->LMF transport)";

/// The line [`RangingTask::run`] emits on entry. Same no-compliance-claim
/// contract as [`SPAWN_LOG`], and unlike it this one is reached by a test.
pub const START_LOG: &str =
    "Ranging task started (scaffold: the sidelink-positioning pipeline is not wired end-to-end)";

/// Carrier phase measurement for a single frequency.
#[derive(Debug, Clone)]
struct CarrierPhaseMeasurement {
    /// Frequency in MHz
    frequency_mhz: f64,
    /// Measured phase in radians (0..2*PI)
    phase_rad: f64,
    /// Phase quality indicator (0.0 - 1.0)
    quality: f64,
}

/// Ranging session with a peer UE.
#[derive(Debug)]
struct RangingSession {
    /// Peer UE identifier
    peer_ue_id: u64,
    /// Last RTT measurement in nanoseconds
    last_rtt_ns: Option<u64>,
    /// Estimated distance in meters (from RTT)
    rtt_distance_m: Option<f64>,
    /// Carrier phase measurements for multi-frequency combining
    carrier_phases: Vec<CarrierPhaseMeasurement>,
    /// Carrier phase distance estimate (high precision)
    carrier_phase_distance_m: Option<f64>,
    /// Number of measurements taken
    measurement_count: u32,
    /// Timestamp of last measurement (ms)
    last_measurement_ms: u64,
}

impl RangingSession {
    fn new(peer_ue_id: u64) -> Self {
        Self {
            peer_ue_id,
            last_rtt_ns: None,
            rtt_distance_m: None,
            carrier_phases: Vec::new(),
            carrier_phase_distance_m: None,
            measurement_count: 0,
            last_measurement_ms: 0,
        }
    }

    /// Update RTT-based distance from round-trip time measurement.
    fn update_rtt(&mut self, rtt_ns: u64, timestamp_ms: u64) {
        self.last_rtt_ns = Some(rtt_ns);
        // distance = (RTT * c) / 2, c = 3e8 m/s, RTT in ns
        let distance_m = (rtt_ns as f64 * 0.3) / 2.0; // 0.3 m/ns = speed of light
        self.rtt_distance_m = Some(distance_m);
        self.measurement_count += 1;
        self.last_measurement_ms = timestamp_ms;
    }

    /// Add a carrier phase measurement and resolve ambiguity if enough frequencies.
    fn add_carrier_phase(&mut self, frequency_mhz: f64, phase_rad: f64, quality: f64) {
        self.carrier_phases.push(CarrierPhaseMeasurement {
            frequency_mhz,
            phase_rad,
            quality,
        });

        // Multi-frequency carrier phase combining for ambiguity resolution
        // Need at least 2 frequencies for widelane combination
        if self.carrier_phases.len() >= 2 {
            self.resolve_carrier_phase_ambiguity();
        }
    }

    /// Resolve carrier phase ambiguity using multi-frequency widelane combination.
    ///
    /// Uses the difference in phase measurements at two frequencies to create
    /// a widelane observable with a longer effective wavelength, which makes
    /// integer ambiguity resolution easier.
    fn resolve_carrier_phase_ambiguity(&mut self) {
        if self.carrier_phases.len() < 2 {
            return;
        }

        let m1 = &self.carrier_phases[0];
        let m2 = &self.carrier_phases[1];

        // Wavelength = c / f
        let lambda1 = 300.0 / m1.frequency_mhz; // meters (c in m/s / f in MHz = m)
        let _lambda2 = 300.0 / m2.frequency_mhz;

        // Widelane wavelength: lambda_w = c / (f1 - f2)
        let freq_diff = (m1.frequency_mhz - m2.frequency_mhz).abs();
        if freq_diff < 0.001 {
            return; // frequencies too close
        }
        let lambda_w = 300.0 / freq_diff;

        // Widelane phase difference
        let phase_diff = m1.phase_rad - m2.phase_rad;

        // Narrowlane for refinement using first frequency
        let distance_widelane = (phase_diff / (2.0 * std::f64::consts::PI)) * lambda_w;

        // Use RTT distance as initial estimate to resolve integer ambiguity
        if let Some(rtt_dist) = self.rtt_distance_m {
            // Find the integer N such that distance_widelane + N*lambda_w is closest to rtt_dist
            let n = ((rtt_dist - distance_widelane) / lambda_w).round();
            let resolved_distance = distance_widelane + n * lambda_w;

            // Refine with narrowlane using first frequency
            let n_narrow = ((resolved_distance
                - (m1.phase_rad / (2.0 * std::f64::consts::PI)) * lambda1)
                / lambda1)
                .round();
            let refined_distance =
                (m1.phase_rad / (2.0 * std::f64::consts::PI)) * lambda1 + n_narrow * lambda1;

            // Weight by quality
            let avg_quality = (m1.quality + m2.quality) / 2.0;
            if avg_quality > 0.3 {
                self.carrier_phase_distance_m = Some(refined_distance.abs());
            }
        }
    }

    /// Get the best distance estimate (carrier phase preferred, RTT as fallback).
    fn best_distance_m(&self) -> Option<f64> {
        self.carrier_phase_distance_m.or(self.rtt_distance_m)
    }
}

pub struct RangingTask {
    _task_base: UeTaskBase,
    /// Active ranging sessions by peer UE ID
    sessions: std::collections::HashMap<u64, RangingSession>,
}

impl RangingTask {
    pub fn new(task_base: UeTaskBase) -> Self {
        Self {
            _task_base: task_base,
            sessions: std::collections::HashMap::new(),
        }
    }
}

#[async_trait::async_trait]
impl Task for RangingTask {
    type Message = RangingMessage;

    async fn run(&mut self, mut rx: mpsc::Receiver<TaskMessage<Self::Message>>) {
        info!("{}", START_LOG);
        loop {
            match rx.recv().await {
                Some(TaskMessage::Message(msg)) => match msg {
                    RangingMessage::StartRanging { peer_ue_id, method } => {
                        debug!(
                            "Ranging: Start session with peer UE {} method={:?}",
                            peer_ue_id, method
                        );
                        self.sessions
                            .insert(peer_ue_id, RangingSession::new(peer_ue_id));
                    }
                    RangingMessage::StopRanging { peer_ue_id } => {
                        debug!("Ranging: Stop session with peer UE {}", peer_ue_id);
                        self.sessions.remove(&peer_ue_id);
                    }
                    RangingMessage::RttMeasurement {
                        peer_ue_id,
                        rtt_ns,
                        timestamp_ms,
                    } => {
                        if let Some(session) = self.sessions.get_mut(&peer_ue_id) {
                            session.update_rtt(rtt_ns, timestamp_ms);
                            debug!(
                                "Ranging: RTT measurement peer={} rtt={}ns distance={:.2}m",
                                peer_ue_id,
                                rtt_ns,
                                session.rtt_distance_m.unwrap_or(0.0)
                            );
                        } else {
                            warn!("Ranging: RTT measurement for unknown peer {}", peer_ue_id);
                        }
                    }
                    RangingMessage::CarrierPhaseMeasurement {
                        peer_ue_id,
                        frequency_mhz,
                        phase_rad,
                        quality,
                    } => {
                        if let Some(session) = self.sessions.get_mut(&peer_ue_id) {
                            session.add_carrier_phase(frequency_mhz, phase_rad, quality);
                            debug!(
                                "Ranging: Carrier phase peer={} freq={:.1}MHz phase={:.4}rad cp_dist={:?}",
                                peer_ue_id,
                                frequency_mhz,
                                phase_rad,
                                session.carrier_phase_distance_m
                            );
                        }
                    }
                    RangingMessage::ReportToLmf { response_tx } => {
                        let mut results = Vec::new();
                        for session in self.sessions.values() {
                            if let Some(distance) = session.best_distance_m() {
                                results.push(RangingResult {
                                    peer_ue_id: session.peer_ue_id,
                                    distance_m: distance,
                                    accuracy_m: if session.carrier_phase_distance_m.is_some() {
                                        0.01 // cm-level with carrier phase
                                    } else {
                                        1.0 // meter-level with RTT
                                    },
                                    measurement_count: session.measurement_count,
                                    method: if session.carrier_phase_distance_m.is_some() {
                                        "carrier_phase".to_string()
                                    } else {
                                        "rtt".to_string()
                                    },
                                });
                            }
                        }
                        debug!("Ranging: Report to LMF with {} results", results.len());
                        if let Some(tx) = response_tx {
                            let _ = tx.send(results);
                        }
                    }
                },
                Some(TaskMessage::Shutdown) => break,
                None => break,
            }
        }
        info!(
            "Ranging task stopped, {} active sessions",
            self.sessions.len()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::{AppMessage, NasMessage, RlsMessage, RrcMessage, TaskHandle};
    use nextgsim_common::config::UeConfig;
    use std::io::Write;
    use std::sync::{Arc, Mutex};

    /// A `MakeWriter` that appends every formatted log record to a shared
    /// buffer, so a test can assert on what the runtime actually emitted
    /// rather than on what the source appears to say.
    #[derive(Clone, Default)]
    struct CapturedLog(Arc<Mutex<Vec<u8>>>);

    impl CapturedLog {
        fn text(&self) -> String {
            String::from_utf8_lossy(&self.0.lock().expect("log buffer not poisoned")).into_owned()
        }
    }

    impl Write for CapturedLog {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0
                .lock()
                .expect("log buffer not poisoned")
                .extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl tracing_subscriber::fmt::MakeWriter<'_> for CapturedLog {
        type Writer = Self;

        fn make_writer(&self) -> Self::Writer {
            self.clone()
        }
    }

    fn task_base() -> UeTaskBase {
        // `UeConfig::default()` is the default (ranging-disabled) start: the
        // criterion is about what a UE that was NOT asked for ranging logs.
        // Read through the same `as_ref().is_some_and(..)` shape `main.rs`
        // uses, so "disabled" here means what it means at the gate.
        assert!(
            !UeConfig::default()
                .ranging_config
                .as_ref()
                .is_some_and(|c| c.enabled),
            "the default UE config must leave ranging disabled, or this test \
             is asserting about the wrong start"
        );
        let (app_tx, _app_rx) = mpsc::channel::<TaskMessage<AppMessage>>(1);
        let (nas_tx, _nas_rx) = mpsc::channel::<TaskMessage<NasMessage>>(1);
        let (rrc_tx, _rrc_rx) = mpsc::channel::<TaskMessage<RrcMessage>>(1);
        let (rls_tx, _rls_rx) = mpsc::channel::<TaskMessage<RlsMessage>>(1);
        UeTaskBase {
            config: Arc::new(UeConfig::default()),
            app_tx: TaskHandle::new(app_tx),
            nas_tx: TaskHandle::new(nas_tx),
            rrc_tx: TaskHandle::new(rrc_tx),
            rls_tx: TaskHandle::new(rls_tx),
            #[cfg(any(
                feature = "nextgsim-she",
                feature = "nextgsim-nwdaf",
                feature = "nextgsim-isac",
                feature = "nextgsim-fl",
                feature = "nextgsim-semantic",
            ))]
            sixg: None,
            rel18: None,
        }
    }

    /// Runs `RangingTask` to completion under a capturing subscriber and
    /// returns everything it logged.
    ///
    /// The task is driven directly rather than `tokio::spawn`ed, and on a
    /// current-thread runtime, because `with_default` installs the dispatcher
    /// in a *thread-local*: a spawned task could be polled on a worker thread
    /// where the capture is not installed, and the buffer would come back
    /// empty — which would satisfy the absence assertion for the wrong reason.
    fn run_ranging_task_capturing_logs() -> String {
        let captured = CapturedLog::default();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(captured.clone())
            .with_max_level(tracing::Level::TRACE)
            .with_ansi(false)
            .finish();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("current-thread runtime builds");

        tracing::subscriber::with_default(subscriber, || {
            runtime.block_on(async {
                let (tx, rx) = mpsc::channel::<TaskMessage<RangingMessage>>(1);
                // Dropping the sender is what ends `run`: `rx.recv()` yields
                // `None` and the loop breaks, so the task starts and stops
                // without needing a shutdown message.
                drop(tx);
                RangingTask::new(task_base()).run(rx).await;
            });
        });

        captured.text()
    }

    /// #55, criterion 3: a default (ranging-disabled) start must not log a
    /// TS 23.586 compliance claim, because the pipeline behind the claim is
    /// dead — no `RangingMessage` producer, no SL-PRS stimulus, no UE→LMF
    /// transport, and no ranging service at the LMF.
    #[test]
    fn a_default_ranging_disabled_start_logs_no_ts_23_586_claim() {
        let logged = run_ranging_task_capturing_logs();

        // Positive control FIRST. The assertion that matters is an absence,
        // and an absence is satisfied by every path that never arrives —
        // including a capture that was never installed or a task that never
        // started. Pinning the scaffold line proves the buffer holds this
        // task's own startup before anything is concluded from what is
        // missing.
        assert!(
            logged.contains(START_LOG),
            "the ranging task's startup line is missing from the capture, so \
             nothing can be concluded from what else is absent; captured: {logged:?}"
        );

        assert!(
            !logged.contains("TS 23.586"),
            "a default start claimed TS 23.586 compliance; captured: {logged:?}"
        );
        assert!(
            !logged.contains("Rel-18"),
            "a default start advertised the ranging scaffold as a Rel-18 \
             capability; captured: {logged:?}"
        );
    }

    /// The binary's spawn line is unreachable from a test (it is inside
    /// `main.rs`'s task closure), so what is pinned instead is the constant it
    /// logs. Changing the binary's wording means changing this string, and
    /// this test reads this string.
    #[test]
    fn the_startup_lines_make_no_ts_23_586_compliance_claim() {
        for line in [SPAWN_LOG, START_LOG] {
            assert!(
                line.contains("scaffold"),
                "{line:?} must say it is a scaffold, or a reader takes the \
                 spawn for a working feature"
            );
            assert!(
                !line.contains("TS 23.586"),
                "{line:?} cites TS 23.586, which reads as a compliance claim"
            );
            assert!(
                !line.contains("Rel-18"),
                "{line:?} advertises a Rel-18 capability the code does not deliver"
            );
        }
    }
}

/// Ranging result for LMF reporting.
#[derive(Debug, Clone)]
pub struct RangingResult {
    /// Peer UE identifier
    pub peer_ue_id: u64,
    /// Estimated distance in meters
    pub distance_m: f64,
    /// Estimated accuracy in meters
    pub accuracy_m: f64,
    /// Number of measurements used
    pub measurement_count: u32,
    /// Method used (rtt or carrier_phase)
    pub method: String,
}
