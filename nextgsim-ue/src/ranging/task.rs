//! Ranging Task for UE - UE-to-UE distance measurement and carrier phase positioning
//!
//! **Scaffold, not wired end-to-end.** Models the Rel-18 ranging concepts of
//! TS 23.586 (RTT ranging, carrier-phase measurement with multi-frequency
//! ambiguity resolution, and LMF result reporting).
//!
//! What is wired, since issue #136: SL-PRS occasions from the sidelink task
//! produce `RttMeasurement` and `CarrierPhaseMeasurement`, so the session maths
//! below runs on live input. The propagation delay behind those measurements is
//! MODELLED from configured geometry — there is no PC5 radio here — so the
//! accuracy they report is the model's, not a radio's.
//!
//! What is still missing: `ReportToLmf` ends at an in-process `oneshot` with no
//! caller, because there is no SLPP/RSPP UE->LMF transport (issue #137) and no
//! ranging service at the LMF to receive one (issue #138). Until both land and an
//! end-to-end run produces a range at an LMF (issue #139), the startup lines below
//! keep saying "scaffold".

use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::nas::lpp::{SidelinkRangingMethod, SidelinkRangingResult};
use crate::tasks::{NasMessage, RangingMessage, Task, TaskMessage, UeTaskBase};

/// Accuracy reported for a carrier-phase estimate, in metres.
///
/// Centimetre-level because the widelane combination resolves to a fraction of a
/// wavelength. It is the accuracy of the MODEL behind the measurement (see the
/// module docs), not of a radio.
const CARRIER_PHASE_ACCURACY_M: f64 = 0.01;

/// Accuracy reported for an RTT-only estimate, in metres. Metre-level: the RTT is
/// quantised to whole nanoseconds, which is 15 cm of range per count.
const RTT_ACCURACY_M: f64 = 1.0;

/// The widest PC5 Layer-2 identity the LPP IE can carry (TS 23.303 §8.2: 24 bits).
const LAYER2_ID_MAX: u32 = 0x00FF_FFFF;

/// The longest range the LPP IE can carry, in centimetres (10 km).
const RANGE_CM_MAX: u32 = 1_000_000;

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
    "Ranging task spawned (scaffold: SL-PRS measurements run, no UE->LMF transport)";

/// The line [`RangingTask::run`] emits on entry. Same no-compliance-claim
/// contract as [`SPAWN_LOG`], and unlike it this one is reached by a test.
pub const START_LOG: &str = "Ranging task started (scaffold: SL-PRS measurements reach the \
     session maths; the report to the LMF has no transport)";

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
    ///
    /// One entry per frequency: a repeat on a frequency already measured REPLACES
    /// it. Appending unconditionally was safe only while nothing produced
    /// measurements — with a live SL-PRS stimulus (issue #136) it would grow the
    /// vector without bound on a periodic path, and `resolve_carrier_phase_ambiguity`
    /// reads indices 0 and 1, so every later occasion would have resolved against
    /// the very first pair of phases and never moved.
    fn add_carrier_phase(&mut self, frequency_mhz: f64, phase_rad: f64, quality: f64) {
        let measurement = CarrierPhaseMeasurement {
            frequency_mhz,
            phase_rad,
            quality,
        };
        match self
            .carrier_phases
            .iter_mut()
            .find(|m| m.frequency_mhz == frequency_mhz)
        {
            Some(existing) => *existing = measurement,
            None => self.carrier_phases.push(measurement),
        }

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

        // Widelane wavelength: lambda_w = c / (f1 - f2), SIGNED.
        //
        // The sign matters and taking `.abs()` here was a defect: the widelane
        // observable is `phase1 - phase2` in cycles, whose sign follows
        // `f1 - f2`, so pairing a signed phase difference with an unsigned
        // wavelength makes the integer ambiguity resolve to the wrong cycle
        // whenever the first frequency is the LOWER one -- an error of up to a
        // whole widelane wavelength (3 m for the default 100 MHz spacing). It was
        // invisible while nothing produced measurements; the first end-to-end
        // occasion recovered 48.97 m for a 50 m geometry (issue #136).
        let freq_diff = m1.frequency_mhz - m2.frequency_mhz;
        if freq_diff.abs() < 0.001 {
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
    task_base: UeTaskBase,
    /// Active ranging sessions by peer UE ID
    sessions: std::collections::HashMap<u64, RangingSession>,
}

impl RangingTask {
    pub fn new(task_base: UeTaskBase) -> Self {
        Self {
            task_base,
            sessions: std::collections::HashMap::new(),
        }
    }
}

impl RangingTask {
    /// The current best estimate per session, in the shape an LMF report carries.
    fn collect_results(&self) -> Vec<RangingResult> {
        let mut results = Vec::new();
        for session in self.sessions.values() {
            if let Some(distance) = session.best_distance_m() {
                let by_carrier_phase = session.carrier_phase_distance_m.is_some();
                results.push(RangingResult {
                    peer_ue_id: session.peer_ue_id,
                    distance_m: distance,
                    accuracy_m: if by_carrier_phase {
                        CARRIER_PHASE_ACCURACY_M
                    } else {
                        RTT_ACCURACY_M
                    },
                    measurement_count: session.measurement_count,
                    method: if by_carrier_phase {
                        "carrier_phase".to_string()
                    } else {
                        "rtt".to_string()
                    },
                });
            }
        }
        results
    }

    /// Hand the report to the NAS task, which is where the LPP endpoint that
    /// reports to the LMF lives (TS 23.586 §5.3.3, issue #137).
    ///
    /// A result that cannot be represented in the LPP IE is DROPPED with a warning
    /// rather than clamped: a range clamped to 10 km or a peer identity truncated to
    /// 24 bits would be reported as a measurement, and a wrong range is worse than a
    /// missing one.
    async fn publish_to_nas(&self, results: &[RangingResult]) {
        let encodable: Vec<SidelinkRangingResult> = results
            .iter()
            .filter_map(|result| match sidelink_ranging_result(result) {
                Ok(encoded) => Some(encoded),
                Err(reason) => {
                    warn!(
                        "Ranging: not reporting peer {} to the LMF: {reason}",
                        result.peer_ue_id
                    );
                    None
                }
            })
            .collect();
        if encodable.is_empty() {
            return;
        }
        if let Err(e) = self
            .task_base
            .nas_tx
            .send(NasMessage::SidelinkRangingReport { results: encodable })
            .await
        {
            warn!("Ranging: could not hand the report to the NAS task: {e}");
        }
    }
}

/// Convert a ranging result into the LPP IE's units and bounds (issue #137).
///
/// # Errors
/// The reason the result cannot be carried, for the caller to log against the peer.
fn sidelink_ranging_result(result: &RangingResult) -> Result<SidelinkRangingResult, String> {
    let peer_layer2_id = u32::try_from(result.peer_ue_id)
        .ok()
        .filter(|id| *id <= LAYER2_ID_MAX)
        .ok_or_else(|| {
            format!(
                "the peer id {} does not fit a 24-bit PC5 Layer-2 identity",
                result.peer_ue_id
            )
        })?;
    if !result.distance_m.is_finite() || result.distance_m < 0.0 {
        return Err(format!(
            "the range {} m is not a distance",
            result.distance_m
        ));
    }
    let range_cm_f = (result.distance_m * 100.0).round();
    if range_cm_f > f64::from(RANGE_CM_MAX) {
        return Err(format!(
            "the range {:.2} m is beyond the {} m the IE can carry",
            result.distance_m,
            RANGE_CM_MAX / 100
        ));
    }
    let accuracy_cm_f = (result.accuracy_m * 100.0).round();
    let accuracy_cm = if accuracy_cm_f.is_finite() && (0.0..=65_535.0).contains(&accuracy_cm_f) {
        accuracy_cm_f as u16
    } else {
        return Err(format!(
            "the accuracy {} m is beyond the IE's range",
            result.accuracy_m
        ));
    };
    let method = match result.method.as_str() {
        "carrier_phase" => SidelinkRangingMethod::CarrierPhase,
        "rtt" => SidelinkRangingMethod::Rtt,
        other => return Err(format!("unknown ranging method {other:?}")),
    };
    Ok(SidelinkRangingResult {
        peer_layer2_id,
        range_cm: range_cm_f as u32,
        accuracy_cm,
        method,
        measurement_count: u16::try_from(result.measurement_count).unwrap_or(u16::MAX),
    })
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
                        let results = self.collect_results();
                        debug!("Ranging: Report to LMF with {} results", results.len());
                        // Push the report toward the LMF (issue #137). This is what
                        // the in-process oneshot used to be instead of: the LPP
                        // endpoint on the NAS task answers a
                        // RequestLocationInformation from what it holds, so the
                        // results have to reach it before the request does.
                        self.publish_to_nas(&results).await;
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
    use crate::test_support::{capture_task_logs, task_base};
    use nextgsim_common::config::UeConfig;

    /// Runs `RangingTask` to completion under a capturing subscriber and returns
    /// everything it logged.
    fn run_ranging_task_capturing_logs() -> String {
        // `UeConfig::default()` is the default (ranging-disabled) start: the
        // criterion is about what a UE that was NOT asked for ranging logs.
        // Read through the same `as_ref().is_some_and(..)` shape `main.rs` uses,
        // so "disabled" here means what it means at the gate.
        assert!(
            !UeConfig::default()
                .ranging_config
                .as_ref()
                .is_some_and(|c| c.enabled),
            "the default UE config must leave ranging disabled, or this test \
             is asserting about the wrong start"
        );
        capture_task_logs(RangingTask::new(task_base()))
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

    // ========================================================================
    // The SL-PRS stimulus, end to end (issue #136). The pipeline used to have no
    // producer at all: `RttMeasurement` and `CarrierPhaseMeasurement` were
    // unreachable, so the RTT and widelane maths above were dead code.
    // ========================================================================

    /// An SL-PRS occasion reaches the ranging session and the session acquires a
    /// distance. Driven through the real channels -- the sidelink task measures the
    /// configured anchor and reports to the ranging task -- and asserted on the
    /// COMPUTED DISTANCE the LMF report carries, not on a log line.
    ///
    /// The geometry is a 3-4-5 triangle scaled to 30/40/50 m, so the expected range
    /// is exactly 50 m and a wrong axis or a squared/unsquared slip would not
    /// coincide with it.
    #[cfg(feature = "sidelink")]
    #[test]
    fn an_sl_prs_occasion_gives_the_ranging_session_a_distance() {
        use crate::sidelink::SidelinkTask;
        use crate::tasks::{SidelinkMessage, TaskHandle, UeRel18Handles};
        use crate::test_support::{hold_capture_lock, task_base_with_config};
        use nextgsim_common::config::{RangingAnchor, RangingConfig};

        // This drives BOTH the ranging and sidelink tasks, whose startup lines the
        // honesty captures assert on. See `test_support`'s hazard 3.
        let _capture_guard = hold_capture_lock();

        const ANCHOR_UE_ID: u64 = 42;
        const EXPECTED_RANGE_M: f64 = 50.0;

        let config = UeConfig {
            ranging_config: Some(RangingConfig {
                enabled: true,
                own_position: [0.0, 0.0, 0.0],
                anchors: vec![RangingAnchor {
                    ue_id: ANCHOR_UE_ID,
                    position: [30.0, 40.0, 0.0],
                }],
                ..RangingConfig::default()
            }),
            ..UeConfig::default()
        };

        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("current-thread runtime builds");

        let results = runtime.block_on(async move {
            let (sidelink_tx, sidelink_rx) = mpsc::channel::<TaskMessage<SidelinkMessage>>(16);
            let (ranging_tx, ranging_rx) = mpsc::channel::<TaskMessage<RangingMessage>>(16);
            let (mint_tx, _mint_rx) = mpsc::channel(1);

            let sidelink_handle = TaskHandle::new(sidelink_tx);
            let ranging_handle = TaskHandle::new(ranging_tx);
            let rel18 = UeRel18Handles {
                ranging_tx: ranging_handle.clone(),
                mint_tx: TaskHandle::new(mint_tx),
                sidelink_tx: sidelink_handle.clone(),
            };

            let mut base = task_base_with_config(config);
            base.rel18 = Some(rel18);

            let mut sidelink = SidelinkTask::new(base.clone());
            assert_eq!(
                sidelink.sl_prs_resource_count(),
                1,
                "one SL-PRS resource per configured anchor, built at construction"
            );
            let mut ranging = RangingTask::new(base);

            let sidelink_run = tokio::spawn(async move { sidelink.run(sidelink_rx).await });
            let ranging_run = tokio::spawn(async move { ranging.run(ranging_rx).await });

            sidelink_handle
                .send(SidelinkMessage::SlPrsOccasion {
                    timestamp_ms: 1_000,
                })
                .await
                .expect("the sidelink task accepts the occasion");
            // Shut the producer down and JOIN it before asking for the report: its
            // measurements and the report travel the same channel, so without this
            // the report could overtake them and the test would race.
            sidelink_handle.shutdown().await.expect("shutdown accepted");
            sidelink_run.await.expect("the sidelink task exits cleanly");

            let (report_tx, report_rx) = tokio::sync::oneshot::channel();
            ranging_handle
                .send(RangingMessage::ReportToLmf {
                    response_tx: Some(report_tx),
                })
                .await
                .expect("the ranging task accepts the report request");
            let results = report_rx.await.expect("the report answers");
            ranging_handle.shutdown().await.expect("shutdown accepted");
            ranging_run.await.expect("the ranging task exits cleanly");
            results
        });

        assert_eq!(
            results.len(),
            1,
            "the occasion must produce exactly one ranging result"
        );
        let result = &results[0];
        assert_eq!(result.peer_ue_id, ANCHOR_UE_ID);
        assert!(
            (result.distance_m - EXPECTED_RANGE_M).abs() < 0.01,
            "the session must recover the 50 m geometry to centimetre level, got \
             {:.4}m -- the RTT estimate alone is 49.95m here (integer-nanosecond \
             quantisation), so a looser bound would not distinguish the widelane \
             result from the fallback",
            result.distance_m
        );
        assert_eq!(
            result.method, "carrier_phase",
            "two carrier frequencies are configured by default, so the widelane \
             combination must resolve and be preferred over the RTT estimate"
        );
        assert!(
            result.measurement_count >= 1,
            "and the RTT measurement must have been counted"
        );
    }

    /// The report the ranging task hands to NAS, end to end from an SL-PRS occasion
    /// (issue #137). This is the step that used to end at an in-process `oneshot` no
    /// caller invoked.
    #[cfg(feature = "sidelink")]
    #[test]
    fn a_measured_range_reaches_the_nas_task_in_the_lpp_ies_units() {
        use crate::sidelink::SidelinkTask;
        use crate::tasks::{SidelinkMessage, TaskHandle, UeRel18Handles};
        use crate::test_support::{hold_capture_lock, task_base_with_config};
        use nextgsim_common::config::{RangingAnchor, RangingConfig};

        // Drives both tasks, whose startup lines the honesty captures assert on.
        let _capture_guard = hold_capture_lock();

        let config = UeConfig {
            ranging_config: Some(RangingConfig {
                enabled: true,
                own_position: [0.0, 0.0, 0.0],
                anchors: vec![RangingAnchor {
                    ue_id: 0x00A5_A5A5,
                    position: [30.0, 40.0, 0.0],
                }],
                ..RangingConfig::default()
            }),
            ..UeConfig::default()
        };

        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("current-thread runtime builds");

        let reported = runtime.block_on(async move {
            let (sidelink_tx, sidelink_rx) = mpsc::channel::<TaskMessage<SidelinkMessage>>(16);
            let (ranging_tx, ranging_rx) = mpsc::channel::<TaskMessage<RangingMessage>>(16);
            let (mint_tx, _mint_rx) = mpsc::channel(1);
            // A LIVE NAS receiver: this is the boundary under test.
            let (nas_tx, mut nas_rx) = mpsc::channel::<TaskMessage<NasMessage>>(16);

            let sidelink_handle = TaskHandle::new(sidelink_tx);
            let ranging_handle = TaskHandle::new(ranging_tx);
            let mut base = task_base_with_config(config);
            base.nas_tx = TaskHandle::new(nas_tx);
            base.rel18 = Some(UeRel18Handles {
                ranging_tx: ranging_handle.clone(),
                mint_tx: TaskHandle::new(mint_tx),
                sidelink_tx: sidelink_handle.clone(),
            });

            let mut sidelink = SidelinkTask::new(base.clone());
            let mut ranging = RangingTask::new(base);
            let sidelink_run = tokio::spawn(async move { sidelink.run(sidelink_rx).await });
            let ranging_run = tokio::spawn(async move { ranging.run(ranging_rx).await });

            sidelink_handle
                .send(SidelinkMessage::SlPrsOccasion {
                    timestamp_ms: 1_000,
                })
                .await
                .expect("accepted");
            sidelink_handle.shutdown().await.expect("shutdown accepted");
            sidelink_run.await.expect("exits cleanly");

            ranging_handle
                .send(RangingMessage::ReportToLmf { response_tx: None })
                .await
                .expect("accepted");
            ranging_handle.shutdown().await.expect("shutdown accepted");
            ranging_run.await.expect("exits cleanly");

            let mut last = None;
            while let Ok(msg) = nas_rx.try_recv() {
                if let TaskMessage::Message(NasMessage::SidelinkRangingReport { results }) = msg {
                    last = Some(results);
                }
            }
            last.expect("the report must reach the NAS task")
        });

        assert_eq!(reported.len(), 1);
        let result = reported[0];
        assert_eq!(result.peer_layer2_id, 0x00A5_A5A5);
        assert_eq!(
            result.range_cm, 5_000,
            "50.00 m in centimetres -- metres would have thrown away the \
             centimetre-level estimate the widelane produced"
        );
        assert_eq!(result.method, SidelinkRangingMethod::CarrierPhase);
        assert_eq!(result.accuracy_cm, 1);
    }

    /// A result the IE cannot represent is DROPPED, not clamped: a range clamped to
    /// 10 km or an identity truncated to 24 bits would be reported as a measurement.
    #[test]
    fn a_result_the_lpp_ie_cannot_carry_is_refused_rather_than_clamped() {
        let ok = RangingResult {
            peer_ue_id: 7,
            distance_m: 12.34,
            accuracy_m: 0.01,
            measurement_count: 2,
            method: "carrier_phase".to_string(),
        };
        let converted = sidelink_ranging_result(&ok).expect("a plain result converts");
        assert_eq!(converted.range_cm, 1_234);
        assert_eq!(converted.peer_layer2_id, 7);

        for (label, bad) in [
            (
                "a peer id wider than 24 bits",
                RangingResult {
                    peer_ue_id: u64::from(LAYER2_ID_MAX) + 1,
                    ..ok.clone()
                },
            ),
            (
                "a range beyond 10 km",
                RangingResult {
                    distance_m: 10_001.0,
                    ..ok.clone()
                },
            ),
            (
                "a negative range",
                RangingResult {
                    distance_m: -1.0,
                    ..ok.clone()
                },
            ),
            (
                "a NaN range",
                RangingResult {
                    distance_m: f64::NAN,
                    ..ok.clone()
                },
            ),
            (
                "an unknown method",
                RangingResult {
                    method: "otdoa".to_string(),
                    ..ok.clone()
                },
            ),
        ] {
            assert!(
                sidelink_ranging_result(&bad).is_err(),
                "{label} must be refused, not carried"
            );
        }
    }

    /// The widelane combination must resolve the same distance whichever way the
    /// two frequencies are ordered. It did not: `lambda_w` took the ABSOLUTE
    /// frequency difference while the phase difference kept its sign, so an
    /// ascending pair resolved to the wrong cycle (issue #136).
    #[test]
    fn the_widelane_resolves_the_same_distance_for_either_frequency_ordering() {
        const RANGE_M: f64 = 50.0;
        // The measurement a phase detector actually gives: the fraction of a
        // wavelength, i.e. ambiguous modulo one cycle.
        let phase_for = |frequency_mhz: f64| {
            let wavelength_m = 300.0 / frequency_mhz;
            (RANGE_M / wavelength_m).fract() * std::f64::consts::TAU
        };
        // The RTT an integer-nanosecond clock would report for this range.
        let rtt_ns = ((2.0 * RANGE_M) / 0.3).round() as u64;

        for pair in [[3500.0, 3600.0], [3600.0, 3500.0]] {
            let mut session = RangingSession::new(1);
            session.update_rtt(rtt_ns, 0);
            for frequency_mhz in pair {
                session.add_carrier_phase(frequency_mhz, phase_for(frequency_mhz), 0.9);
            }
            let resolved = session
                .carrier_phase_distance_m
                .unwrap_or_else(|| panic!("{pair:?}: the widelane must resolve"));
            assert!(
                (resolved - RANGE_M).abs() < 0.01,
                "{pair:?}: expected {RANGE_M}m, got {resolved:.4}m"
            );
        }
    }

    /// Repeated occasions on the same frequencies must REPLACE their measurements,
    /// not accumulate. Two things break if they accumulate on a periodic path: the
    /// vector grows without bound, and `resolve_carrier_phase_ambiguity` reads
    /// indices 0 and 1, so every later occasion resolves against the first pair of
    /// phases and the estimate never moves (issue #136).
    #[test]
    fn repeated_occasions_replace_their_carrier_phases_rather_than_accumulating() {
        let phase_for = |range_m: f64, frequency_mhz: f64| {
            (range_m / (300.0 / frequency_mhz)).fract() * std::f64::consts::TAU
        };
        let rtt_for = |range_m: f64| ((2.0 * range_m) / 0.3).round() as u64;

        let mut session = RangingSession::new(1);
        // Occasion 1 at 50 m, occasion 2 with the peer moved to 80 m.
        for range_m in [50.0f64, 80.0] {
            session.update_rtt(rtt_for(range_m), 0);
            for frequency_mhz in [3500.0, 3600.0] {
                session.add_carrier_phase(frequency_mhz, phase_for(range_m, frequency_mhz), 0.9);
            }
        }

        assert_eq!(
            session.carrier_phases.len(),
            2,
            "two frequencies measured twice must leave two entries, not four"
        );
        let resolved = session
            .carrier_phase_distance_m
            .expect("the widelane still resolves");
        assert!(
            (resolved - 80.0).abs() < 0.01,
            "the estimate must follow the LATEST occasion (80m), got {resolved:.4}m"
        );
    }

    /// One frequency cannot form a widelane, so the estimate stays the RTT one --
    /// coarser, and reported as such. This is what makes the carrier-phase
    /// assertion above about the carrier phase.
    #[test]
    fn a_single_carrier_frequency_leaves_the_estimate_at_the_rtt_precision() {
        let mut session = RangingSession::new(1);
        let rtt_ns = ((2.0f64 * 50.0) / 0.3).round() as u64;
        session.update_rtt(rtt_ns, 0);
        session.add_carrier_phase(3500.0, 1.0, 0.9);

        assert!(
            session.carrier_phase_distance_m.is_none(),
            "a single frequency must not produce a carrier-phase distance"
        );
        let best = session.best_distance_m().expect("the RTT estimate stands");
        assert!(
            (best - 49.95).abs() < 0.001,
            "the RTT estimate is 49.95m at integer-nanosecond resolution, got {best:.4}m"
        );
    }

    /// The runtime gate: ranging disabled means an SL-PRS occasion measures
    /// nothing, so a default UE is unchanged.
    #[cfg(feature = "sidelink")]
    #[test]
    fn an_sl_prs_occasion_measures_nothing_while_ranging_is_disabled() {
        use crate::sidelink::SidelinkTask;
        use crate::tasks::{SidelinkMessage, TaskHandle, UeRel18Handles};
        use crate::test_support::{hold_capture_lock, task_base_with_config};
        use nextgsim_common::config::{RangingAnchor, RangingConfig};

        // Drives the sidelink task, whose startup line a capture asserts on.
        let _capture_guard = hold_capture_lock();

        let config = UeConfig {
            ranging_config: Some(RangingConfig {
                // The anchor is configured; only the flag is off.
                enabled: false,
                anchors: vec![RangingAnchor {
                    ue_id: 42,
                    position: [30.0, 40.0, 0.0],
                }],
                ..RangingConfig::default()
            }),
            ..UeConfig::default()
        };

        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("current-thread runtime builds");

        runtime.block_on(async move {
            let (sidelink_tx, sidelink_rx) = mpsc::channel::<TaskMessage<SidelinkMessage>>(16);
            let (ranging_tx, mut ranging_rx) = mpsc::channel::<TaskMessage<RangingMessage>>(16);
            let (mint_tx, _mint_rx) = mpsc::channel(1);

            let sidelink_handle = TaskHandle::new(sidelink_tx);
            let mut base = task_base_with_config(config);
            base.rel18 = Some(UeRel18Handles {
                ranging_tx: TaskHandle::new(ranging_tx),
                mint_tx: TaskHandle::new(mint_tx),
                sidelink_tx: sidelink_handle.clone(),
            });

            let mut sidelink = SidelinkTask::new(base);
            let sidelink_run = tokio::spawn(async move { sidelink.run(sidelink_rx).await });
            sidelink_handle
                .send(SidelinkMessage::SlPrsOccasion {
                    timestamp_ms: 1_000,
                })
                .await
                .expect("accepted");
            sidelink_handle.shutdown().await.expect("shutdown accepted");
            sidelink_run.await.expect("exits cleanly");

            assert!(
                ranging_rx.try_recv().is_err(),
                "a disabled ranging config must produce no measurement at all"
            );
        });
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
