//! Ranging Task for UE - UE-to-UE distance measurement and carrier phase positioning
//!
//! Implements Rel-18 ranging service per TS 23.586:
//! - Round-trip time (RTT) based ranging
//! - Carrier phase measurement for cm-level accuracy
//! - Phase ambiguity resolution via multi-frequency combining
//! - Ranging result reporting to LMF

use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::tasks::{RangingMessage, Task, TaskMessage, UeTaskBase};

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
            let n_narrow = ((resolved_distance - (m1.phase_rad / (2.0 * std::f64::consts::PI)) * lambda1) / lambda1).round();
            let refined_distance = (m1.phase_rad / (2.0 * std::f64::consts::PI)) * lambda1 + n_narrow * lambda1;

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
        info!("Ranging task started (Rel-18, TS 23.586)");
        loop {
            match rx.recv().await {
                Some(TaskMessage::Message(msg)) => match msg {
                    RangingMessage::StartRanging { peer_ue_id, method } => {
                        debug!("Ranging: Start session with peer UE {} method={:?}", peer_ue_id, method);
                        self.sessions.insert(peer_ue_id, RangingSession::new(peer_ue_id));
                    }
                    RangingMessage::StopRanging { peer_ue_id } => {
                        debug!("Ranging: Stop session with peer UE {}", peer_ue_id);
                        self.sessions.remove(&peer_ue_id);
                    }
                    RangingMessage::RttMeasurement { peer_ue_id, rtt_ns, timestamp_ms } => {
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
        info!("Ranging task stopped, {} active sessions", self.sessions.len());
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
