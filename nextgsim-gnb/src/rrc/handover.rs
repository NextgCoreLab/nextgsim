//! Handover Handling for gNB
//!
//! Implements handover procedures per 3GPP TS 38.300 and TS 38.331.
//!
//! # Handover Types
//!
//! ## Intra-gNB Handover
//! - Source and target cells are both managed by the same gNB
//! - Does not involve NGAP handover procedures
//! - Simpler coordination
//!
//! ## Inter-gNB Handover (Xn-based)
//! - Source and target gNBs are different
//! - Uses Xn interface for coordination
//! - Requires UE context transfer
//!
//! ## Inter-gNB Handover (N2-based)
//! - Uses AMF for coordination
//! - Falls back when Xn is not available
//!
//! # Handover Procedure (Intra-gNB)
//!
//! 1. gNB receives measurement report from UE (A3 event)
//! 2. gNB decides on handover based on measurements
//! 3. gNB prepares target cell
//! 4. gNB sends RRC Reconfiguration with mobility control info
//! 5. UE synchronizes with target cell
//! 6. UE sends RRC Reconfiguration Complete to target cell
//! 7. gNB updates UE context
//!
//! # Reference
//! - 3GPP TS 38.300: NR; Overall description
//! - 3GPP TS 38.331: NR; RRC protocol specification
//! - 3GPP TS 38.413: NGAP protocol

use std::time::{Duration, Instant};
use std::collections::HashMap;

use tracing::{debug, info, warn};

/// Handover decision result
#[derive(Debug, Clone)]
pub enum HandoverDecision {
    /// No handover needed
    NoHandover,
    /// Intra-gNB handover to specified cell
    IntraGnbHandover { target_cell_id: i32 },
    /// Inter-gNB handover (not implemented yet)
    InterGnbHandover { target_gnb_id: u32, target_cell_id: i32 },
}

/// Handover state for a UE
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UeHandoverState {
    /// No handover in progress
    Idle,
    /// Handover preparation
    Preparing,
    /// Waiting for UE to complete handover
    Executing,
    /// Handover complete
    Complete,
    /// Handover failed
    Failed,
}

impl Default for UeHandoverState {
    fn default() -> Self {
        Self::Idle
    }
}

/// Handover context for a UE
#[derive(Debug, Clone)]
pub struct UeHandoverContext {
    /// Current handover state
    pub state: UeHandoverState,
    /// Source cell ID
    pub source_cell_id: Option<i32>,
    /// Target cell ID
    pub target_cell_id: Option<i32>,
    /// Handover command transaction ID
    pub transaction_id: u8,
    /// Handover start time
    pub start_time: Option<Instant>,
    /// T304 expiry timeout
    pub t304_duration: Duration,
}

impl Default for UeHandoverContext {
    fn default() -> Self {
        Self {
            state: UeHandoverState::Idle,
            source_cell_id: None,
            target_cell_id: None,
            transaction_id: 0,
            start_time: None,
            t304_duration: Duration::from_millis(100),
        }
    }
}

/// Measurement report from UE
#[derive(Debug, Clone)]
pub struct MeasurementReport {
    /// Measurement ID
    pub meas_id: u8,
    /// Serving cell RSRP
    pub serving_rsrp: i32,
    /// Neighbor cell measurements
    pub neighbors: Vec<NeighborMeasurement>,
}

/// Neighbor cell measurement
#[derive(Debug, Clone)]
pub struct NeighborMeasurement {
    /// Physical cell ID
    pub pci: u32,
    /// RSRP measurement
    pub rsrp: i32,
}

/// Handover configuration parameters
#[derive(Debug, Clone)]
pub struct HandoverConfig {
    /// A3 offset threshold (dB)
    pub a3_offset: i32,
    /// Hysteresis (dB)
    pub hysteresis: i32,
    /// Time-to-trigger (ms)
    pub time_to_trigger: u64,
    /// T304 timer duration (ms)
    pub t304_duration: u64,
}

impl Default for HandoverConfig {
    fn default() -> Self {
        Self {
            a3_offset: 3,
            hysteresis: 1,
            time_to_trigger: 640,
            t304_duration: 100,
        }
    }
}

/// Handover manager for gNB
pub struct GnbHandoverManager {
    /// Configuration
    config: HandoverConfig,
    /// Handover context per UE (`ue_id` -> context)
    ue_contexts: HashMap<i32, UeHandoverContext>,
    /// Transaction ID counter
    transaction_counter: u8,
    /// Cell ID of this gNB (for future use in inter-gNB handover)
    #[allow(dead_code)]
    cell_id: i32,
}

impl GnbHandoverManager {
    pub fn new(cell_id: i32) -> Self {
        Self {
            config: HandoverConfig::default(),
            ue_contexts: HashMap::new(),
            transaction_counter: 0,
            cell_id,
        }
    }

    /// Set handover configuration
    pub fn set_config(&mut self, config: HandoverConfig) {
        self.config = config;
    }

    /// Get next transaction ID
    fn next_transaction_id(&mut self) -> u8 {
        self.transaction_counter = self.transaction_counter.wrapping_add(1);
        self.transaction_counter
    }

    /// Process measurement report from UE and decide on handover
    pub fn process_measurement_report(
        &mut self,
        ue_id: i32,
        report: &MeasurementReport,
    ) -> HandoverDecision {
        // Check if handover is already in progress
        if let Some(ctx) = self.ue_contexts.get(&ue_id) {
            if !matches!(ctx.state, UeHandoverState::Idle | UeHandoverState::Failed) {
                debug!("Ignoring measurement report - handover in progress for UE {}", ue_id);
                return HandoverDecision::NoHandover;
            }
        }

        // Find best neighbor that meets handover criteria
        let mut best_candidate: Option<(u32, i32)> = None;

        for neighbor in &report.neighbors {
            // Check A3 condition: neighbor > serving + offset - hysteresis
            let threshold = report.serving_rsrp + self.config.a3_offset - self.config.hysteresis;
            if neighbor.rsrp > threshold
                && best_candidate.is_none_or(|(_, rsrp)| neighbor.rsrp > rsrp) {
                    best_candidate = Some((neighbor.pci, neighbor.rsrp));
                }
        }

        if let Some((pci, rsrp)) = best_candidate {
            info!(
                "Handover decision for UE {}: target_pci={}, rsrp={} (serving={})",
                ue_id, pci, rsrp, report.serving_rsrp
            );
            // For now, use PCI as cell_id (in real impl, would lookup cell_id from PCI)
            HandoverDecision::IntraGnbHandover { target_cell_id: pci as i32 }
        } else {
            HandoverDecision::NoHandover
        }
    }

    /// Initiate handover for a UE
    pub fn initiate_handover(
        &mut self,
        ue_id: i32,
        source_cell_id: i32,
        target_cell_id: i32,
    ) -> Option<HandoverCommand> {
        let transaction_id = self.next_transaction_id();

        // Create handover context
        let ctx = UeHandoverContext {
            state: UeHandoverState::Preparing,
            source_cell_id: Some(source_cell_id),
            target_cell_id: Some(target_cell_id),
            transaction_id,
            start_time: Some(Instant::now()),
            t304_duration: Duration::from_millis(self.config.t304_duration),
        };
        self.ue_contexts.insert(ue_id, ctx);

        info!(
            "Initiating handover for UE {}: cell {} -> cell {}",
            ue_id, source_cell_id, target_cell_id
        );

        // Build handover command
        Some(HandoverCommand {
            ue_id,
            target_cell_id,
            target_pci: target_cell_id as u32, // Simplified: PCI = cell_id
            new_ue_id: None, // Same UE ID for intra-gNB handover
            transaction_id,
        })
    }

    /// Mark handover as executing (RRC Reconfiguration sent)
    pub fn mark_executing(&mut self, ue_id: i32) {
        if let Some(ctx) = self.ue_contexts.get_mut(&ue_id) {
            ctx.state = UeHandoverState::Executing;
        }
    }

    /// Complete handover (RRC Reconfiguration Complete received)
    pub fn complete_handover(&mut self, ue_id: i32) -> Option<i32> {
        if let Some(ctx) = self.ue_contexts.get_mut(&ue_id) {
            if ctx.state == UeHandoverState::Executing {
                ctx.state = UeHandoverState::Complete;

                if let Some(start) = ctx.start_time {
                    info!(
                        "Handover complete for UE {}: duration={:?}",
                        ue_id, start.elapsed()
                    );
                }

                let target = ctx.target_cell_id;

                // Reset context
                ctx.state = UeHandoverState::Idle;
                ctx.source_cell_id = None;
                ctx.target_cell_id = None;
                ctx.start_time = None;

                return target;
            }
        }
        None
    }

    /// Fail handover
    pub fn fail_handover(&mut self, ue_id: i32) -> Option<i32> {
        if let Some(ctx) = self.ue_contexts.get_mut(&ue_id) {
            warn!("Handover failed for UE {}", ue_id);
            ctx.state = UeHandoverState::Failed;

            let source = ctx.source_cell_id;
            ctx.source_cell_id = None;
            ctx.target_cell_id = None;
            ctx.start_time = None;

            return source;
        }
        None
    }

    /// Check for timed out handovers
    pub fn check_timeouts(&mut self) -> Vec<i32> {
        let mut timed_out = Vec::new();

        for (&ue_id, ctx) in self.ue_contexts.iter_mut() {
            if ctx.state == UeHandoverState::Executing {
                if let Some(start) = ctx.start_time {
                    if start.elapsed() >= ctx.t304_duration {
                        warn!("Handover timeout for UE {}", ue_id);
                        timed_out.push(ue_id);
                    }
                }
            }
        }

        // Mark timed out UEs as failed
        for ue_id in &timed_out {
            if let Some(ctx) = self.ue_contexts.get_mut(ue_id) {
                ctx.state = UeHandoverState::Failed;
            }
        }

        timed_out
    }

    /// Get handover state for a UE
    pub fn get_state(&self, ue_id: i32) -> UeHandoverState {
        self.ue_contexts.get(&ue_id)
            .map(|c| c.state)
            .unwrap_or(UeHandoverState::Idle)
    }

    /// Check if handover is in progress for a UE
    pub fn is_in_progress(&self, ue_id: i32) -> bool {
        self.ue_contexts.get(&ue_id)
            .map(|c| matches!(c.state, UeHandoverState::Preparing | UeHandoverState::Executing))
            .unwrap_or(false)
    }
}

// ============================================================================
// Inter-gNB Xn Handover Support
// ============================================================================

/// Xn handover request sent to target gNB
#[derive(Debug, Clone)]
pub struct XnHandoverRequest {
    /// UE ID at source gNB
    pub source_ue_id: i32,
    /// Source gNB ID
    pub source_gnb_id: u32,
    /// Target cell ID
    pub target_cell_id: i32,
    /// Cause of handover
    pub cause: XnHandoverCause,
    /// UE context to transfer (serialized)
    pub ue_context: XnUeContext,
}

/// Xn handover acknowledge from target gNB
#[derive(Debug, Clone)]
pub struct XnHandoverAcknowledge {
    /// UE ID allocated at target gNB
    pub target_ue_id: i32,
    /// Target gNB ID
    pub target_gnb_id: u32,
    /// Handover command (RRC Reconfiguration with mobility control) for the UE
    pub handover_command: Vec<u8>,
    /// Admitted PDU sessions
    pub admitted_pdu_sessions: Vec<i32>,
}

/// UE context transferred during Xn handover
#[derive(Debug, Clone)]
pub struct XnUeContext {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: Option<i64>,
    /// Security capabilities
    pub security_capabilities: u32,
    /// Active PDU sessions (session ID list)
    pub pdu_sessions: Vec<XnPduSessionContext>,
    /// RRC establishment cause
    pub establishment_cause: u8,
}

/// PDU session context for Xn transfer
#[derive(Debug, Clone)]
pub struct XnPduSessionContext {
    /// PDU session ID
    pub psi: i32,
    /// QoS flow ID
    pub qfi: u8,
    /// UPF tunnel endpoint (TEID)
    pub uplink_teid: u32,
    /// UPF address
    pub upf_address: std::net::IpAddr,
}

/// Cause for Xn handover
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XnHandoverCause {
    /// Radio resource management
    RadioResourceManagement,
    /// Resource optimization
    ResourceOptimization,
    /// Reduce load in serving cell
    ReduceLoadInServingCell,
}

/// Path Switch Request to AMF after Xn handover completion
#[derive(Debug, Clone)]
pub struct PathSwitchRequest {
    /// UE ID at target gNB
    pub ue_id: i32,
    /// Source AMF UE NGAP ID
    pub source_amf_ue_ngap_id: i64,
    /// Target gNB ID
    pub target_gnb_id: u32,
    /// User location information (TAI + NR CGI)
    pub tai: u32,
    /// PDU sessions switched
    pub pdu_sessions: Vec<i32>,
}

impl GnbHandoverManager {
    /// Process an inter-gNB handover decision and build an Xn Handover Request
    pub fn initiate_xn_handover(
        &mut self,
        ue_id: i32,
        source_cell_id: i32,
        target_gnb_id: u32,
        target_cell_id: i32,
        ue_context: XnUeContext,
    ) -> Option<XnHandoverRequest> {
        let transaction_id = self.next_transaction_id();

        let ctx = UeHandoverContext {
            state: UeHandoverState::Preparing,
            source_cell_id: Some(source_cell_id),
            target_cell_id: Some(target_cell_id),
            transaction_id,
            start_time: Some(Instant::now()),
            t304_duration: Duration::from_millis(self.config.t304_duration),
        };
        self.ue_contexts.insert(ue_id, ctx);

        info!(
            "Initiating Xn handover for UE {}: cell {} -> gnb {} cell {}",
            ue_id, source_cell_id, target_gnb_id, target_cell_id
        );

        Some(XnHandoverRequest {
            source_ue_id: ue_id,
            source_gnb_id: self.cell_id as u32,
            target_cell_id,
            cause: XnHandoverCause::RadioResourceManagement,
            ue_context,
        })
    }

    /// Handle incoming Xn Handover Request at target gNB
    /// Returns an XnHandoverAcknowledge if the target can accept the UE
    pub fn handle_xn_handover_request(
        &mut self,
        request: &XnHandoverRequest,
        new_ue_id: i32,
    ) -> Option<XnHandoverAcknowledge> {
        let transaction_id = self.next_transaction_id();

        // Create handover context at target
        let ctx = UeHandoverContext {
            state: UeHandoverState::Preparing,
            source_cell_id: None,
            target_cell_id: Some(request.target_cell_id),
            transaction_id,
            start_time: Some(Instant::now()),
            t304_duration: Duration::from_millis(self.config.t304_duration),
        };
        self.ue_contexts.insert(new_ue_id, ctx);

        info!(
            "Accepting Xn handover at target: source_ue={}, new_ue_id={}, target_cell={}",
            request.source_ue_id, new_ue_id, request.target_cell_id
        );

        // Build handover command for the UE
        let ho_cmd = HandoverCommand {
            ue_id: new_ue_id,
            target_cell_id: request.target_cell_id,
            target_pci: request.target_cell_id as u32,
            new_ue_id: Some(new_ue_id),
            transaction_id,
        };

        let admitted_pdu_sessions: Vec<i32> = request
            .ue_context
            .pdu_sessions
            .iter()
            .map(|s| s.psi)
            .collect();

        Some(XnHandoverAcknowledge {
            target_ue_id: new_ue_id,
            target_gnb_id: self.cell_id as u32,
            handover_command: ho_cmd.build_rrc_pdu(),
            admitted_pdu_sessions,
        })
    }

    /// Build a Path Switch Request after successful Xn handover at target gNB
    pub fn build_path_switch_request(
        &self,
        ue_id: i32,
        source_amf_ue_ngap_id: i64,
        tai: u32,
        pdu_sessions: Vec<i32>,
    ) -> PathSwitchRequest {
        PathSwitchRequest {
            ue_id,
            source_amf_ue_ngap_id,
            target_gnb_id: self.cell_id as u32,
            tai,
            pdu_sessions,
        }
    }

    /// Complete Xn handover at target gNB (UE has arrived)
    pub fn complete_xn_handover(&mut self, ue_id: i32) -> bool {
        if let Some(ctx) = self.ue_contexts.get_mut(&ue_id) {
            if matches!(ctx.state, UeHandoverState::Preparing | UeHandoverState::Executing) {
                if let Some(start) = ctx.start_time {
                    info!(
                        "Xn handover complete for UE {}: duration={:?}",
                        ue_id, start.elapsed()
                    );
                }
                ctx.state = UeHandoverState::Complete;
                ctx.state = UeHandoverState::Idle;
                ctx.start_time = None;
                return true;
            }
        }
        false
    }
}

impl Default for GnbHandoverManager {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Handover command to be sent to UE
#[derive(Debug, Clone)]
pub struct HandoverCommand {
    /// UE ID
    pub ue_id: i32,
    /// Target cell ID
    pub target_cell_id: i32,
    /// Target physical cell ID
    pub target_pci: u32,
    /// New UE ID (C-RNTI) for target cell
    pub new_ue_id: Option<i32>,
    /// Transaction ID
    pub transaction_id: u8,
}

impl HandoverCommand {
    /// Build RRC Reconfiguration PDU with handover command (simplified format)
    pub fn build_rrc_pdu(&self) -> Vec<u8> {
        // Simplified format (not real ASN.1):
        // [0] = message type (0x00 = RRC Reconfiguration with HO)
        // [1] = transaction_id
        // [2-3] = target PCI (big endian)
        // [4-7] = target cell ID (big endian)
        // [8] = flags (0x01 = has new_ue_id, 0x02 = full_config)
        // [9-12] = new_ue_id (if flag set)

        let mut pdu = Vec::with_capacity(13);
        pdu.push(0x00); // Message type
        pdu.push(self.transaction_id);
        pdu.extend_from_slice(&(self.target_pci as u16).to_be_bytes());
        pdu.extend_from_slice(&self.target_cell_id.to_be_bytes());

        let flags = if self.new_ue_id.is_some() { 0x01 } else { 0x00 };
        pdu.push(flags);

        if let Some(new_id) = self.new_ue_id {
            pdu.extend_from_slice(&new_id.to_be_bytes());
        }

        pdu
    }
}

/// Parse measurement report from UE (simplified format)
pub fn parse_measurement_report(pdu: &[u8]) -> Option<(i32, MeasurementReport)> {
    // Simplified format:
    // [0] = message type (0x0B = MeasurementReport)
    // [1] = meas_id
    // [2-3] = serving PCI
    // [4] = serving RSRP (signed)
    // [5] = num neighbors
    // [6...] = neighbor measurements (3 bytes each: PCI_hi, PCI_lo, RSRP)

    if pdu.len() < 6 || pdu[0] != 0x0B {
        return None;
    }

    let meas_id = pdu[1];
    let _serving_pci = u16::from_be_bytes([pdu[2], pdu[3]]);
    let serving_rsrp = pdu[4] as i8 as i32;
    let num_neighbors = pdu[5] as usize;

    let mut neighbors = Vec::new();
    let mut offset = 6;

    for _ in 0..num_neighbors {
        if offset + 3 > pdu.len() {
            break;
        }
        let pci = u16::from_be_bytes([pdu[offset], pdu[offset + 1]]) as u32;
        let rsrp = pdu[offset + 2] as i8 as i32;
        neighbors.push(NeighborMeasurement { pci, rsrp });
        offset += 3;
    }

    Some((
        meas_id as i32,
        MeasurementReport {
            meas_id,
            serving_rsrp,
            neighbors,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handover_manager_creation() {
        let manager = GnbHandoverManager::new(1);
        assert_eq!(manager.cell_id, 1);
    }

    #[test]
    fn test_measurement_report_processing() {
        let mut manager = GnbHandoverManager::new(1);
        manager.set_config(HandoverConfig {
            a3_offset: 3,
            hysteresis: 1,
            ..Default::default()
        });

        // Serving cell at -90 dBm
        let report = MeasurementReport {
            meas_id: 1,
            serving_rsrp: -90,
            neighbors: vec![
                NeighborMeasurement { pci: 2, rsrp: -80 }, // Better by 10 dB
                NeighborMeasurement { pci: 3, rsrp: -95 }, // Worse
            ],
        };

        let decision = manager.process_measurement_report(1, &report);
        match decision {
            HandoverDecision::IntraGnbHandover { target_cell_id } => {
                assert_eq!(target_cell_id, 2);
            }
            _ => panic!("Expected handover decision"),
        }
    }

    #[test]
    fn test_handover_lifecycle() {
        let mut manager = GnbHandoverManager::new(1);

        // Initiate handover
        let cmd = manager.initiate_handover(1, 1, 2).unwrap();
        assert_eq!(cmd.target_cell_id, 2);
        assert_eq!(manager.get_state(1), UeHandoverState::Preparing);

        // Mark executing
        manager.mark_executing(1);
        assert_eq!(manager.get_state(1), UeHandoverState::Executing);

        // Complete
        let target = manager.complete_handover(1);
        assert_eq!(target, Some(2));
        assert_eq!(manager.get_state(1), UeHandoverState::Idle);
    }

    #[test]
    fn test_handover_command_pdu() {
        let cmd = HandoverCommand {
            ue_id: 1,
            target_cell_id: 100,
            target_pci: 16,
            new_ue_id: None,
            transaction_id: 5,
        };

        let pdu = cmd.build_rrc_pdu();
        assert_eq!(pdu[0], 0x00); // Message type
        assert_eq!(pdu[1], 5); // Transaction ID
        // Target PCI = 16
        assert_eq!(u16::from_be_bytes([pdu[2], pdu[3]]), 16);
        // Target cell ID = 100
        assert_eq!(i32::from_be_bytes([pdu[4], pdu[5], pdu[6], pdu[7]]), 100);
    }

    #[test]
    fn test_parse_measurement_report() {
        let pdu = vec![
            0x0B, // Message type
            0x01, // meas_id
            0x00, 0x01, // serving PCI
            0xA6u8, // serving RSRP (-90 as signed byte)
            0x01, // 1 neighbor
            0x00, 0x02, // neighbor PCI = 2
            0xB0u8, // neighbor RSRP (-80 as signed byte)
        ];

        let (meas_id, report) = parse_measurement_report(&pdu).unwrap();
        assert_eq!(meas_id, 1);
        assert_eq!(report.serving_rsrp, -90);
        assert_eq!(report.neighbors.len(), 1);
        assert_eq!(report.neighbors[0].pci, 2);
        assert_eq!(report.neighbors[0].rsrp, -80);
    }
}
