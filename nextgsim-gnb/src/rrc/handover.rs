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

/// Path Switch Request to AMF after Xn handover completion.
///
/// Sent by target gNB to AMF to update the user plane path after successful handover.
/// Reference: 3GPP TS 38.413 Section 8.4.5
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
    /// PDU sessions switched with their tunnel endpoints
    pub pdu_sessions: Vec<PathSwitchPduSession>,
}

/// PDU session information for path switch request.
#[derive(Debug, Clone)]
pub struct PathSwitchPduSession {
    /// PDU session ID
    pub psi: i32,
    /// New GTP-U tunnel endpoint ID (target gNB)
    pub uplink_teid: u32,
    /// QoS flow list for this session
    pub qfi_list: Vec<u8>,
}

/// Path Switch Acknowledge from AMF.
///
/// Contains updated user plane tunnel information after AMF coordinates with UPF.
/// Reference: 3GPP TS 38.413 Section 8.4.6
#[derive(Debug, Clone)]
pub struct PathSwitchAcknowledge {
    /// UE ID at target gNB
    pub ue_id: i32,
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: i64,
    /// Updated PDU session list with new UPF tunnel endpoints
    pub pdu_sessions: Vec<PathSwitchAckPduSession>,
    /// Security context update (optional)
    pub security_context: Option<SecurityContext>,
}

/// PDU session info in Path Switch Acknowledge.
#[derive(Debug, Clone)]
pub struct PathSwitchAckPduSession {
    /// PDU session ID
    pub psi: i32,
    /// New downlink GTP-U tunnel endpoint (UPF to target gNB)
    pub downlink_teid: u32,
    /// UPF address
    pub upf_address: std::net::IpAddr,
    /// List of accepted QoS flows
    pub accepted_qfi_list: Vec<u8>,
}

/// Security context update (optional in Path Switch Ack).
#[derive(Debug, Clone)]
pub struct SecurityContext {
    /// Next hop chaining counter
    pub ncc: u8,
    /// Next hop (NH) value
    pub nh: [u8; 32],
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

    /// Builds a Path Switch Request after successful Xn handover at target gNB.
    ///
    /// Called when UE has completed handover and arrived at target cell.
    /// Sends request to AMF to update UPF tunnel endpoints.
    ///
    /// # Arguments
    /// * `ue_id` - UE identifier at target gNB
    /// * `source_amf_ue_ngap_id` - AMF UE NGAP ID from source
    /// * `tai` - Tracking Area Identity at target
    /// * `pdu_sessions` - List of PDU sessions with new tunnel info
    pub fn build_path_switch_request(
        &self,
        ue_id: i32,
        source_amf_ue_ngap_id: i64,
        tai: u32,
        pdu_sessions: Vec<PathSwitchPduSession>,
    ) -> PathSwitchRequest {
        info!(
            "Building Path Switch Request for UE {}: {} PDU sessions",
            ue_id, pdu_sessions.len()
        );

        PathSwitchRequest {
            ue_id,
            source_amf_ue_ngap_id,
            target_gnb_id: self.cell_id as u32,
            tai,
            pdu_sessions,
        }
    }

    /// Handles Path Switch Acknowledge from AMF.
    ///
    /// Updates local UPF tunnel information with new downlink TEIDs.
    /// Called after AMF coordinates with UPF to update user plane path.
    ///
    /// # Arguments
    /// * `ue_id` - UE identifier
    /// * `ack` - Path Switch Acknowledge message from AMF
    ///
    /// # Returns
    /// * `true` if path switch was successfully processed
    pub fn handle_path_switch_acknowledge(
        &mut self,
        ue_id: i32,
        ack: &PathSwitchAcknowledge,
    ) -> bool {
        info!(
            "Received Path Switch Acknowledge for UE {}: {} PDU sessions updated",
            ue_id, ack.pdu_sessions.len()
        );

        // Update UE context with new security context if provided
        if let Some(ref sec_ctx) = ack.security_context {
            debug!(
                "Updating security context for UE {}: NCC={}, NH present",
                ue_id, sec_ctx.ncc
            );
        }

        // In real implementation, would update:
        // 1. GTP-U tunnel endpoints for downlink (UPF -> target gNB)
        // 2. Security context (NH, NCC)
        // 3. QoS flow mappings

        true
    }

    /// Updates UPF tunnel endpoint after path switch.
    ///
    /// Called internally to update GTP-U tunnel configuration.
    ///
    /// # Arguments
    /// * `ue_id` - UE identifier
    /// * `psi` - PDU session ID
    /// * `new_downlink_teid` - New GTP-U TEID for downlink
    /// * `upf_address` - UPF address
    pub fn update_upf_tunnel(
        &mut self,
        ue_id: i32,
        psi: i32,
        new_downlink_teid: u32,
        upf_address: std::net::IpAddr,
    ) {
        debug!(
            "Updating UPF tunnel for UE {} session {}: DL_TEID={}, UPF={}",
            ue_id, psi, new_downlink_teid, upf_address
        );

        // In real implementation, would:
        // 1. Update GTP-U session with new downlink TEID
        // 2. Start forwarding DL data from UPF to target gNB
        // 3. Release old tunnel from source gNB
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

// ============================================================================
// DAPS (Dual Active Protocol Stack) Handover Support (Rel-16)
// ============================================================================

/// DAPS handover state tracking dual-active protocol stacks.
///
/// During DAPS handover, the UE maintains connections to both source and target cells
/// simultaneously, enabling make-before-break handover with zero interruption time.
///
/// Reference: 3GPP TS 38.300 Section 9.2.3.2.2
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DapsHandoverState {
    /// No DAPS handover in progress
    Inactive,
    /// Dual active mode - both source and target cells active
    DualActive,
    /// Switching data path from source to target
    Switching,
    /// Handover complete, releasing source cell
    Complete,
}

/// DAPS handover configuration containing cell and bearer information.
#[derive(Debug, Clone)]
pub struct DapsConfig {
    /// Source cell ID
    pub source_cell_id: i32,
    /// Target cell ID
    pub target_cell_id: i32,
    /// Source cell C-RNTI
    pub source_crnti: u16,
    /// Target cell C-RNTI
    pub target_crnti: u16,
    /// Bearer configurations to maintain during handover
    pub bearer_configs: Vec<DapsBearerConfig>,
    /// Data forwarding enabled from source to target
    pub data_forwarding: bool,
}

/// Per-bearer configuration for DAPS handover.
#[derive(Debug, Clone)]
pub struct DapsBearerConfig {
    /// Radio Bearer ID
    pub rb_id: u8,
    /// QoS Flow Identifier
    pub qfi: u8,
    /// Whether this bearer supports DAPS
    pub daps_capable: bool,
}

/// DAPS handover manager maintaining dual RLC entities.
///
/// Manages the dual-active protocol stack during DAPS handover:
/// - Maintains two RLC entities (source + target) simultaneously
/// - Forwards DL data from source cell while target cell is being prepared
/// - Switches data path once target cell confirms
pub struct DapsHandoverManager {
    /// Current DAPS handover state
    state: DapsHandoverState,
    /// DAPS configuration
    config: Option<DapsConfig>,
    /// DAPS handover start time
    start_time: Option<Instant>,
    /// T304daps timer duration (handover failure timer)
    t304_daps_duration: Duration,
    /// Source cell RLC data buffer size (bytes)
    source_rlc_buffer_bytes: usize,
    /// Target cell RLC data buffer size (bytes)
    target_rlc_buffer_bytes: usize,
}

impl DapsHandoverManager {
    /// Creates a new DAPS handover manager.
    pub fn new() -> Self {
        Self {
            state: DapsHandoverState::Inactive,
            config: None,
            start_time: None,
            t304_daps_duration: Duration::from_millis(150), // Per 3GPP TS 38.331
            source_rlc_buffer_bytes: 0,
            target_rlc_buffer_bytes: 0,
        }
    }

    /// Starts DAPS handover procedure.
    ///
    /// Transitions to DualActive state and begins maintaining dual protocol stacks.
    pub fn start_daps_handover(&mut self, config: DapsConfig) {
        info!(
            "Starting DAPS handover: source_cell={}, target_cell={}, source_crnti={}, target_crnti={}",
            config.source_cell_id, config.target_cell_id, config.source_crnti, config.target_crnti
        );

        self.config = Some(config);
        self.state = DapsHandoverState::DualActive;
        self.start_time = Some(Instant::now());
        self.source_rlc_buffer_bytes = 0;
        self.target_rlc_buffer_bytes = 0;
    }

    /// Forwards downlink data on the source cell RLC entity.
    ///
    /// Called during DualActive state to continue data delivery while target is preparing.
    pub fn forward_dl_data_source(&mut self, data_bytes: usize) -> bool {
        if self.state != DapsHandoverState::DualActive {
            return false;
        }

        self.source_rlc_buffer_bytes += data_bytes;
        debug!("DAPS: Forwarded {} bytes on source cell, total buffered: {}",
               data_bytes, self.source_rlc_buffer_bytes);
        true
    }

    /// Switches data path to target cell.
    ///
    /// Called when target cell confirms readiness (RRC Reconfiguration Complete received).
    /// Begins draining source RLC and routing new data to target.
    pub fn switch_to_target(&mut self) -> bool {
        if self.state != DapsHandoverState::DualActive {
            return false;
        }

        info!("DAPS: Switching data path to target cell");
        self.state = DapsHandoverState::Switching;
        true
    }

    /// Sends data on the target cell RLC entity.
    ///
    /// Used after switch to deliver new data via target cell.
    pub fn send_dl_data_target(&mut self, data_bytes: usize) -> bool {
        if self.state != DapsHandoverState::Switching {
            return false;
        }

        self.target_rlc_buffer_bytes += data_bytes;
        debug!("DAPS: Sent {} bytes on target cell, total buffered: {}",
               data_bytes, self.target_rlc_buffer_bytes);
        true
    }

    /// Releases source cell resources.
    ///
    /// Called after source RLC is drained and all data is delivered via target.
    /// Transitions to Complete state.
    pub fn release_source(&mut self) -> bool {
        if self.state != DapsHandoverState::Switching {
            return false;
        }

        if let Some(start) = self.start_time {
            info!("DAPS: Releasing source cell, handover duration: {:?}", start.elapsed());
        }

        self.state = DapsHandoverState::Complete;
        self.source_rlc_buffer_bytes = 0;
        true
    }

    /// Completes DAPS handover and resets state.
    pub fn complete(&mut self) {
        if let Some(start) = self.start_time {
            info!("DAPS handover complete: duration={:?}", start.elapsed());
        }

        self.state = DapsHandoverState::Inactive;
        self.config = None;
        self.start_time = None;
        self.source_rlc_buffer_bytes = 0;
        self.target_rlc_buffer_bytes = 0;
    }

    /// Checks if T304daps timer has expired.
    pub fn check_t304_daps_expired(&self) -> bool {
        if let Some(start) = self.start_time {
            if self.state != DapsHandoverState::Inactive {
                return start.elapsed() >= self.t304_daps_duration;
            }
        }
        false
    }

    /// Gets current DAPS handover state.
    pub fn state(&self) -> DapsHandoverState {
        self.state
    }

    /// Checks if DAPS handover is in progress.
    pub fn is_in_progress(&self) -> bool {
        self.state != DapsHandoverState::Inactive
    }

    /// Gets source RLC buffer size in bytes.
    pub fn source_buffer_bytes(&self) -> usize {
        self.source_rlc_buffer_bytes
    }

    /// Gets target RLC buffer size in bytes.
    pub fn target_buffer_bytes(&self) -> usize {
        self.target_rlc_buffer_bytes
    }
}

impl Default for DapsHandoverManager {
    fn default() -> Self {
        Self::new()
    }
}

/// DAPS-specific RRC Reconfiguration message.
///
/// Contains mobility control info for DAPS handover with dual active configuration.
#[derive(Debug, Clone)]
pub struct DapsRrcReconfiguration {
    /// Transaction ID
    pub transaction_id: u8,
    /// Source cell configuration (to maintain)
    pub source_cell_config: DapsCellConfig,
    /// Target cell configuration (to establish)
    pub target_cell_config: DapsCellConfig,
    /// T304daps timer value (ms)
    pub t304_daps_ms: u64,
    /// Data forwarding indicator
    pub data_forwarding_enabled: bool,
}

/// Cell configuration for DAPS handover.
#[derive(Debug, Clone)]
pub struct DapsCellConfig {
    /// Physical cell ID
    pub pci: u32,
    /// Cell ID
    pub cell_id: i32,
    /// C-RNTI
    pub crnti: u16,
    /// Radio bearer configurations
    pub bearer_configs: Vec<DapsBearerConfig>,
}

impl GnbHandoverManager {
    /// Builds a DAPS handover command (RRC Reconfiguration with DAPS config).
    ///
    /// # Arguments
    /// * `ue_id` - UE identifier
    /// * `daps_config` - DAPS configuration with source and target cell info
    ///
    /// # Returns
    /// * `DapsRrcReconfiguration` message to send to UE
    pub fn build_daps_handover_command(
        &mut self,
        ue_id: i32,
        daps_config: DapsConfig,
    ) -> DapsRrcReconfiguration {
        let transaction_id = self.next_transaction_id();

        info!(
            "Building DAPS handover command for UE {}: source_cell={}, target_cell={}",
            ue_id, daps_config.source_cell_id, daps_config.target_cell_id
        );

        // Create handover context
        let ctx = UeHandoverContext {
            state: UeHandoverState::Preparing,
            source_cell_id: Some(daps_config.source_cell_id),
            target_cell_id: Some(daps_config.target_cell_id),
            transaction_id,
            start_time: Some(Instant::now()),
            t304_duration: Duration::from_millis(self.config.t304_duration),
        };
        self.ue_contexts.insert(ue_id, ctx);

        DapsRrcReconfiguration {
            transaction_id,
            source_cell_config: DapsCellConfig {
                pci: daps_config.source_cell_id as u32,
                cell_id: daps_config.source_cell_id,
                crnti: daps_config.source_crnti,
                bearer_configs: daps_config.bearer_configs.clone(),
            },
            target_cell_config: DapsCellConfig {
                pci: daps_config.target_cell_id as u32,
                cell_id: daps_config.target_cell_id,
                crnti: daps_config.target_crnti,
                bearer_configs: daps_config.bearer_configs.clone(),
            },
            t304_daps_ms: 150, // Default T304daps per 3GPP TS 38.331
            data_forwarding_enabled: daps_config.data_forwarding,
        }
    }

    /// Handles DAPS handover completion from UE.
    ///
    /// Called when RRC Reconfiguration Complete is received during DAPS handover.
    /// Initiates data path switch from source to target cell.
    pub fn complete_daps_handover(&mut self, ue_id: i32) -> bool {
        if let Some(ctx) = self.ue_contexts.get_mut(&ue_id) {
            if matches!(ctx.state, UeHandoverState::Preparing | UeHandoverState::Executing) {
                if let Some(start) = ctx.start_time {
                    info!(
                        "DAPS handover complete for UE {}: duration={:?}",
                        ue_id, start.elapsed()
                    );
                }
                ctx.state = UeHandoverState::Complete;
                return true;
            }
        }
        false
    }
}

impl DapsRrcReconfiguration {
    /// Encodes the DAPS RRC Reconfiguration message to bytes (simplified format).
    ///
    /// Simplified format (not real ASN.1):
    /// [0] = message type (0x10 = RRC Reconfiguration with DAPS)
    /// [1] = transaction_id
    /// [2-3] = source PCI (big endian)
    /// [4-5] = source C-RNTI (big endian)
    /// [6-7] = target PCI (big endian)
    /// [8-9] = target C-RNTI (big endian)
    /// [10-11] = T304daps (big endian, ms)
    /// [12] = flags (0x01 = data_forwarding_enabled)
    pub fn encode(&self) -> Vec<u8> {
        let mut pdu = Vec::with_capacity(13);
        pdu.push(0x10); // Message type: DAPS RRC Reconfiguration
        pdu.push(self.transaction_id);
        pdu.extend_from_slice(&(self.source_cell_config.pci as u16).to_be_bytes());
        pdu.extend_from_slice(&self.source_cell_config.crnti.to_be_bytes());
        pdu.extend_from_slice(&(self.target_cell_config.pci as u16).to_be_bytes());
        pdu.extend_from_slice(&self.target_cell_config.crnti.to_be_bytes());
        pdu.extend_from_slice(&(self.t304_daps_ms as u16).to_be_bytes());
        let flags = if self.data_forwarding_enabled { 0x01 } else { 0x00 };
        pdu.push(flags);
        pdu
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
