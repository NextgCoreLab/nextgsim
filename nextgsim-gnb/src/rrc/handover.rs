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

use std::collections::HashMap;
use std::time::{Duration, Instant};

use nextgsim_rrc::procedures::measurement_report::{
    decode_measurement_report, rsrp_range_to_dbm, MeasurementReportData,
};
use nextgsim_rrc::procedures::rrc_reconfiguration::{
    encode_handover_command, HandoverCommandParams, MasterKeyUpdateParams,
};
use nextgsim_rrc::procedures::rrc_reestablishment::SIMULATED_C_RNTI;
use tracing::{debug, info, warn};

/// Handover decision result
#[derive(Debug, Clone)]
pub enum HandoverDecision {
    /// No handover needed
    NoHandover,
    /// Intra-gNB handover to specified cell
    IntraGnbHandover { target_cell_id: i32 },
    /// Inter-gNB handover (not implemented yet)
    InterGnbHandover {
        target_gnb_id: u32,
        target_cell_id: i32,
    },
}

/// Handover state for a UE
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UeHandoverState {
    /// No handover in progress
    #[default]
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
                debug!(
                    "Ignoring measurement report - handover in progress for UE {}",
                    ue_id
                );
                return HandoverDecision::NoHandover;
            }
        }

        // Find best neighbor that meets handover criteria
        let mut best_candidate: Option<(u32, i32)> = None;

        for neighbor in &report.neighbors {
            // Check A3 condition: neighbor > serving + offset - hysteresis
            let threshold = report.serving_rsrp + self.config.a3_offset - self.config.hysteresis;
            if neighbor.rsrp > threshold
                && best_candidate.is_none_or(|(_, rsrp)| neighbor.rsrp > rsrp)
            {
                best_candidate = Some((neighbor.pci, neighbor.rsrp));
            }
        }

        if let Some((pci, rsrp)) = best_candidate {
            info!(
                "Handover decision for UE {}: target_pci={}, rsrp={} (serving={})",
                ue_id, pci, rsrp, report.serving_rsrp
            );
            // For now, use PCI as cell_id (in real impl, would lookup cell_id from PCI)
            HandoverDecision::IntraGnbHandover {
                target_cell_id: pci as i32,
            }
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
            new_ue_id: None,                   // Same UE ID for intra-gNB handover
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
                        ue_id,
                        start.elapsed()
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
        self.ue_contexts
            .get(&ue_id)
            .map(|c| c.state)
            .unwrap_or(UeHandoverState::Idle)
    }

    /// Check if handover is in progress for a UE
    pub fn is_in_progress(&self, ue_id: i32) -> bool {
        self.ue_contexts
            .get(&ue_id)
            .map(|c| {
                matches!(
                    c.state,
                    UeHandoverState::Preparing | UeHandoverState::Executing
                )
            })
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

        // A command that cannot be encoded means the admit fails, not that an
        // unparseable PDU is handed to the source gNB to forward to a UE.
        let handover_command = ho_cmd.build_rrc_pdu()?;

        Some(XnHandoverAcknowledge {
            target_ue_id: new_ue_id,
            target_gnb_id: self.cell_id as u32,
            handover_command,
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
            ue_id,
            pdu_sessions.len()
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
            ue_id,
            ack.pdu_sessions.len()
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
            if matches!(
                ctx.state,
                UeHandoverState::Preparing | UeHandoverState::Executing
            ) {
                if let Some(start) = ctx.start_time {
                    info!(
                        "Xn handover complete for UE {}: duration={:?}",
                        ue_id,
                        start.elapsed()
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
        debug!(
            "DAPS: Forwarded {} bytes on source cell, total buffered: {}",
            data_bytes, self.source_rlc_buffer_bytes
        );
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
        debug!(
            "DAPS: Sent {} bytes on target cell, total buffered: {}",
            data_bytes, self.target_rlc_buffer_bytes
        );
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
            info!(
                "DAPS: Releasing source cell, handover duration: {:?}",
                start.elapsed()
            );
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
            if matches!(
                ctx.state,
                UeHandoverState::Preparing | UeHandoverState::Executing
            ) {
                if let Some(start) = ctx.start_time {
                    info!(
                        "DAPS handover complete for UE {}: duration={:?}",
                        ue_id,
                        start.elapsed()
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
    /// Encodes the DAPS reconfiguration as a real `RRCReconfiguration` carrying a
    /// `reconfigurationWithSync` toward the target (TS 38.331 §5.3.5.5.2,
    /// issue #107).
    ///
    /// Replaces a hand-rolled 13-byte format that the source itself flagged as
    /// not being genuine ASN.1.
    ///
    /// # What a DAPS reconfiguration is, and is not, here
    ///
    /// DAPS (Rel-16) keeps the SOURCE link up while the UE accesses the target, so
    /// the conformant message carries `daps-SourceRelease` and a per-DRB
    /// `daps-Config`. Neither used to exist in the vendored schema, which was
    /// Rel-15; #105 has since upgraded it to Rel-19, so the schema is no longer the
    /// reason they are absent. What goes on the wire is still the target half: a
    /// real handover command to the target cell with the DAPS T304. The source-link
    /// state (`source_cell_config`, `data_forwarding_enabled`) stays local to the
    /// gNB, which is where it already was — the old byte format carried it, but no
    /// UE ever read those bytes as anything.
    ///
    /// Stated rather than silently dropped: a reader comparing this to TS 38.331
    /// should know the DAPS-specific IEs are absent because nothing in this tree
    /// maintains a simultaneous source link for them to describe — not because they
    /// were forgotten, and no longer because the schema lacks them. Populating them
    /// needs the dual-connectivity behaviour they signal, which is separate work.
    pub fn encode(&self) -> Option<Vec<u8>> {
        encode_handover_command(&HandoverCommandParams {
            rrc_transaction_id: self.transaction_id.min(3),
            target_phys_cell_id: (self.target_cell_config.pci % 1008) as u16,
            new_ue_identity: self.target_cell_config.crnti,
            t304_ms: nearest_enumerated_t304_ms(self.t304_daps_ms),
            full_config: true,
            // `masterKeyUpdate` with `keySetChangeIndicator: false` — a HORIZONTAL
            // re-key from the UE's current `KgNB` (TS 38.331 §5.3.5.7, TS 33.501
            // §6.9.2.3.1, issue #39). Horizontal because an intra-gNB handover has no
            // fresh NH: a vertical one needs the AMF's `{NH, NCC}`, which only arrives
            // in a HANDOVER REQUEST or a PATH SWITCH REQUEST ACKNOWLEDGE.
            //
            // Present rather than omitted, which is the change: omitting it leaves the
            // UE on the source cell's keys, so a handover would silently keep a key the
            // source gNB holds — the forward-security loss this issue reports.
            master_key_update: Some(MasterKeyUpdateParams {
                key_set_change_indicator: false,
                next_hop_chaining_count: HORIZONTAL_NCC,
            }),
        })
        .ok()
    }
}

/// The `nextHopChainingCount` a horizontal re-key reports (TS 33.501 §6.9.2.3.1).
///
/// Zero, and that is the spec's own value rather than a placeholder: a horizontal
/// derivation does not consume a fresh NH, so the chaining count does not advance. The
/// UE compares it against the NCC it holds and derives horizontally when they match —
/// which is exactly what this gNB means.
const HORIZONTAL_NCC: u8 = 0;

/// The `t304` this gNB uses for a handover command, in milliseconds.
///
/// 1000 ms: long enough that a simulated target access does not time out on a
/// loaded host, and one of the values TS 38.331 enumerates.
const DEFAULT_T304_MS: u16 = 1000;

/// Rounds a configured DAPS T304 UP to the nearest value TS 38.331 enumerates.
///
/// Rounding **up** rather than to the nearest: T304 is a failure deadline, so a
/// configured 300 ms becoming 200 ms would declare failure sooner than the
/// operator asked, while 500 ms only gives the handover more time than requested.
/// A value above the largest enumerated one saturates at it.
fn nearest_enumerated_t304_ms(configured_ms: u64) -> u16 {
    const ENUMERATED: [u16; 8] = [50, 100, 150, 200, 500, 1000, 2000, 10000];
    ENUMERATED
        .iter()
        .copied()
        .find(|ms| u64::from(*ms) >= configured_ms)
        .unwrap_or(10000)
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
    /// Builds the handover command as a real `RRCReconfiguration` carrying a
    /// `reconfigurationWithSync` (TS 38.331 §5.3.5.5.2, issue #107).
    ///
    /// Replaces a hand-rolled byte format that the source itself flagged as not
    /// being genuine ASN.1:
    /// `[0x00, tid, pci_hi, pci_lo, cell_id x4, flags, new_ue_id x4]`.
    ///
    /// # The simulator's `target_cell_id` is not on the wire, and that is correct
    ///
    /// The old format carried it. `reconfigurationWithSync` has no field for it,
    /// because it is not a 3GPP identity — a target cell is identified by
    /// `physCellId` (plus a frequency, which is single-carrier here). The receiving
    /// UE resolves the PCI against the cells it can actually hear, using the same
    /// `phys_cell_id_from_nci` convention both ends already share for
    /// re-establishment. A UE handed a cell index it could not verify would trust a
    /// number the network invented; a UE handed a PCI it cannot find has learned
    /// something true, namely that it cannot reach the target.
    ///
    /// Returns `None` when the command cannot be encoded — a PCI past the ASN.1
    /// bound, say. `None` rather than a byte fallback: a handover command the UE
    /// cannot parse is worse than a handover that does not start.
    pub fn build_rrc_pdu(&self) -> Option<Vec<u8>> {
        // The C-RNTI the UE is to use in the target. `new_ue_id` is the
        // simulator's own UE index, which is what this gNB has; reduced into
        // `RNTI-Value` (0..65535) rather than masked, so the value stays a
        // function of the identity rather than of its low bits.
        let new_ue_identity = self
            .new_ue_id
            .map_or(SIMULATED_C_RNTI, |id| (id as u32 % 65536) as u16);
        encode_handover_command(&HandoverCommandParams {
            rrc_transaction_id: self.transaction_id.min(3),
            target_phys_cell_id: (self.target_pci % 1008) as u16,
            new_ue_identity,
            t304_ms: DEFAULT_T304_MS,
            // The UE releases its stored configuration: this gNB sends no
            // per-target delta, so a delta-on-nothing would leave the UE merging
            // into a configuration the target never described.
            full_config: true,
            // `masterKeyUpdate` with `keySetChangeIndicator: false` — a HORIZONTAL
            // re-key from the UE's current `KgNB` (TS 38.331 §5.3.5.7, TS 33.501
            // §6.9.2.3.1, issue #39). Horizontal because an intra-gNB handover has no
            // fresh NH: a vertical one needs the AMF's `{NH, NCC}`, which only arrives
            // in a HANDOVER REQUEST or a PATH SWITCH REQUEST ACKNOWLEDGE.
            //
            // Present rather than omitted, which is the change: omitting it leaves the
            // UE on the source cell's keys, so a handover would silently keep a key the
            // source gNB holds — the forward-security loss this issue reports.
            master_key_update: Some(MasterKeyUpdateParams {
                key_set_change_indicator: false,
                next_hop_chaining_count: HORIZONTAL_NCC,
            }),
        })
        .ok()
    }
}

/// Parse a UPER `MeasurementReport` from the UE (TS 38.331 §5.5.5, issue #107).
///
/// Replaces a hand-rolled byte format
/// (`[0x0B, meas_id, pci_hi, pci_lo, rsrp, n, ...]`) whose comment named it
/// "simplified format", while `nextgsim-rrc`'s complete `measurement_report`
/// codec sat with zero callers. The UE's sender flips in the **same change** — a
/// one-sided flip stops measurements arriving, which is silent because a gNB with
/// no measurement report simply never initiates handover.
///
/// Returns `None` for a PDU that is not a `MeasurementReport`, so the caller's
/// existing "not for me" path is unchanged.
///
/// # Quantities are converted back to dBm here
///
/// The wire carries `RSRP-Range` (`INTEGER (0..127)`, TS 38.133); `MeasurementReport`
/// downstream is in dBm, and the A3 comparison depends on it. `rsrp_range_to_dbm`
/// is the inverse of the `dbm_to_rsrp_range` the UE used — one mapping, in the
/// crate both ends share, because a second copy is how the two ends would disagree
/// about what -100 dBm is.
pub fn parse_measurement_report(pdu: &[u8]) -> Option<(i32, MeasurementReport)> {
    let data = decode_measurement_report(pdu).ok()?;
    Some(measurement_report_from(&data))
}

/// The same conversion, from an ALREADY-DECODED report (issue #162).
///
/// The typed UL-DCCH dispatch decodes the `UL-DCCH-Message` once and hands the parsed
/// `MeasurementReportData` on, so re-encoding it just to call the byte entry point
/// above would be two decodes and two chances to disagree. One implementation, two
/// entry points.
pub fn measurement_report_from(data: &MeasurementReportData) -> (i32, MeasurementReport) {
    // The serving cell's RSRP. A report whose serving entry carries no ssb-Results
    // is one where the UE had no measurement to give; treated as the weakest
    // representable level rather than dropped, because the neighbour results are
    // still actionable and discarding the whole report would lose them.
    let serving_rsrp = data
        .serv_freq_results
        .first()
        .and_then(|serv| {
            serv.meas_result_serving_cell
                .cell_results
                .ssb_results
                .as_ref()
        })
        .and_then(|r| r.rsrp)
        .map_or(MISSING_RSRP_DBM, |range| {
            rsrp_range_to_dbm(range).round() as i32
        });

    let mut neighbors = Vec::new();
    for freq in &data.neigh_freq_results {
        for cell in &freq.meas_result_list {
            // A neighbour with no PhysCellId cannot be a handover target: the
            // target is identified by PCI. Skipped rather than given a placeholder
            // PCI, which would name some other cell.
            let Some(pci) = cell.phys_cell_id else {
                continue;
            };
            let rsrp = cell
                .cell_results
                .ssb_results
                .as_ref()
                .and_then(|r| r.rsrp)
                .map_or(MISSING_RSRP_DBM, |range| {
                    rsrp_range_to_dbm(range).round() as i32
                });
            neighbors.push(NeighborMeasurement {
                pci: u32::from(pci),
                rsrp,
            });
        }
    }

    // Inter-RAT (B1/B2) results are decoded and NOT turned into handover
    // candidates: this gNB has no inter-RAT handover, so presenting an E-UTRA cell
    // as a `NeighborMeasurement` would offer the A3 evaluator a target it cannot
    // hand over to. Reported so the report is not silently half-read.
    if !data.eutra_neigh_results.is_empty() {
        debug!(
            "MeasurementReport meas_id={} carries {} E-UTRA result(s); recorded, not \
             offered as handover candidates (no inter-RAT handover here)",
            data.meas_id.0,
            data.eutra_neigh_results.len()
        );
    }

    (
        i32::from(data.meas_id.0),
        MeasurementReport {
            meas_id: data.meas_id.0,
            serving_rsrp,
            neighbors,
        },
    )
}

/// The dBm level a measurement the UE did not provide is reported as.
///
/// -156 dBm is the bottom of the NR `RSRP-Range` (TS 38.133), i.e. the weakest
/// representable measurement. Named rather than inlined because it is a
/// *substitution*: the UE said nothing, and this is what the A3 evaluator sees. It
/// is deliberately the weakest value, so an absent measurement never makes a cell
/// look like a handover target.
const MISSING_RSRP_DBM: i32 = -156;

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

    /// #107, criterion 3: the handover command is a real `RRCReconfiguration`
    /// carrying a `reconfigurationWithSync`.
    ///
    /// This test used to index the hand-rolled bytes
    /// (`pdu[0] == 0x00`, `pdu[2..4]` = PCI, `pdu[4..8]` = the simulator's cell
    /// id). Rewritten rather than deleted: the same facts are asserted, by decoding
    /// the message.
    ///
    /// Note the transaction id: the old format carried 5, which is outside
    /// `RRC-TransactionIdentifier`'s `INTEGER (0..3)`. The real encoder cannot put 5
    /// on the wire, so it is clamped — which is the hand-rolled format having been
    /// able to encode an illegal value.
    #[test]
    fn test_handover_command_pdu() {
        use nextgsim_rrc::procedures::rrc_reconfiguration::decode_handover_command;

        let cmd = HandoverCommand {
            ue_id: 1,
            target_cell_id: 100,
            target_pci: 16,
            new_ue_id: None,
            transaction_id: 5,
        };

        let pdu = cmd
            .build_rrc_pdu()
            .expect("the handover command must encode");
        let decoded = decode_handover_command(&pdu).expect("and must decode as a handover command");
        assert_eq!(
            decoded.target_phys_cell_id, 16,
            "the target is identified by physCellId"
        );
        assert_eq!(
            decoded.rrc_transaction_id, 3,
            "5 is outside RRC-TransactionIdentifier's INTEGER (0..3), so the real \
             encoder clamps it -- the hand-rolled format happily encoded an illegal \
             value"
        );
        assert!(decoded.full_config, "the UE applies a full configuration");
        assert_eq!(decoded.t304_ms, 1000);
    }

    /// A plain (DRB-establishing) `RRCReconfiguration` must NOT be read as a
    /// handover command. Both are DL-DCCH `RRCReconfiguration`s, so the receiver
    /// tells them apart by `reconfigurationWithSync` — and before this change it
    /// told them apart by a leading byte that no longer exists.
    #[test]
    fn a_plain_reconfiguration_is_not_a_handover_command() {
        use nextgsim_rrc::procedures::rrc_reconfiguration::{
            build_drb_reconfiguration_params, decode_handover_command, encode_rrc_reconfiguration,
        };

        let params = build_drb_reconfiguration_params(
            0,
            1,
            1,
            4,
            &[9],
            true,
            nextgsim_rrc::procedures::rrc_reconfiguration::DrbIntegrityProtection::Disabled,
            // With a measConfig, which is what the live `establish_drb` now sends
            // (issue #170): the distinguishing feature is `reconfigurationWithSync`,
            // and a measConfig must not be mistaken for one.
            crate::rrc::meas::a3_meas_config_params(&nextgsim_common::config::GnbConfig::default()),
        )
        .expect("a DRB reconfiguration builds");
        let pdu = encode_rrc_reconfiguration(&params).expect("and encodes");
        assert!(
            decode_handover_command(&pdu).is_none(),
            "a reconfiguration with no reconfigurationWithSync is not a handover \
             command; reading it as one would make the UE access a target cell it \
             was never given"
        );
    }

    /// The DAPS reconfiguration is a real `RRCReconfiguration` too, carrying the
    /// target half. Its DAPS-specific IEs are still not populated — no longer for
    /// want of a schema (#105 upgraded it to Rel-19) but because nothing here keeps
    /// a simultaneous source link for them to describe. That is stated at the
    /// encoder rather than left implicit.
    #[test]
    fn the_daps_reconfiguration_is_a_real_reconfiguration_with_sync() {
        use nextgsim_rrc::procedures::rrc_reconfiguration::decode_handover_command;

        let daps = DapsRrcReconfiguration {
            transaction_id: 1,
            source_cell_config: DapsCellConfig {
                cell_id: 1,
                pci: 7,
                crnti: 0x1111,
                bearer_configs: Vec::new(),
            },
            target_cell_config: DapsCellConfig {
                cell_id: 2,
                pci: 21,
                crnti: 0x2222,
                bearer_configs: Vec::new(),
            },
            // 300 ms is NOT an enumerated t304; it must round UP to 500, because
            // t304 is a failure deadline and rounding down would declare failure
            // sooner than configured.
            t304_daps_ms: 300,
            data_forwarding_enabled: true,
        };

        let pdu = daps.encode().expect("the DAPS reconfiguration must encode");
        let decoded = decode_handover_command(&pdu).expect("and must decode");
        assert_eq!(decoded.target_phys_cell_id, 21, "the TARGET's PCI");
        assert_eq!(
            decoded.new_ue_identity, 0x2222,
            "the C-RNTI for the target, not the source's"
        );
        assert_eq!(
            decoded.t304_ms, 500,
            "300 ms rounds UP to the next enumerated value, never down"
        );
    }

    /// #107, criterion 4: the parser reads a real UPER `MeasurementReport`.
    ///
    /// This test used to build the hand-rolled byte format
    /// (`[0x0B, meas_id, pci_hi, pci_lo, rsrp, n, ...]`) and assert it parsed. That
    /// made it a test OF the format being removed, so it is rewritten rather than
    /// deleted: the levels and the expectations are the same, the encoding is now
    /// the spec's.
    #[test]
    fn test_parse_measurement_report() {
        use nextgsim_rrc::procedures::measurement_report::{
            dbm_to_rsrp_range, encode_measurement_report, MeasCellResults, MeasResult2Nr,
            MeasResultCellNr, MeasResultNr, MeasResultServFreqNr, MeasurementReportParams,
        };

        fn cell(pci: u16, dbm: i32) -> MeasResultNr {
            MeasResultNr {
                phys_cell_id: Some(pci),
                cell_results: MeasResultCellNr {
                    ssb_results: Some(MeasCellResults {
                        rsrp: Some(dbm_to_rsrp_range(f64::from(dbm))),
                        rsrq: None,
                        sinr: None,
                    }),
                    csi_rs_results: None,
                },
                rs_index_results: None,
            }
        }

        let pdu = encode_measurement_report(&MeasurementReportParams {
            meas_id: 1,
            serv_freq_results: vec![MeasResultServFreqNr {
                serv_cell_index: 0,
                meas_result_serving_cell: cell(1, -90),
                meas_result_best_neigh_cell: None,
            }],
            neigh_freq_results: vec![MeasResult2Nr {
                ssb_frequency_arfcn: None,
                ref_freq_csi_rs: None,
                meas_result_list: vec![cell(2, -80)],
            }],
            eutra_neigh_results: Vec::new(),
            enhanced_quantities: None,
        })
        .expect("the report must encode");

        let (meas_id, report) = parse_measurement_report(&pdu).expect("and must parse");
        assert_eq!(meas_id, 1);
        assert_eq!(
            report.serving_rsrp, -90,
            "the dBm level must survive the dBm -> RSRP-Range -> dBm round trip; \
             the two converters are inverses in one crate for exactly this reason"
        );
        assert_eq!(report.neighbors.len(), 1);
        assert_eq!(report.neighbors[0].pci, 2);
        assert_eq!(report.neighbors[0].rsrp, -80);
    }

    /// The old byte format must NOT parse any more. Without this, a UE that was
    /// never flipped would keep sending the legacy PDU and the gNB would keep
    /// reading it, which is exactly the half-flipped state criterion 4 forbids.
    #[test]
    fn the_legacy_byte_format_measurement_report_no_longer_parses() {
        let legacy = vec![0x0B, 0x01, 0x00, 0x01, 0xA6u8, 0x01, 0x00, 0x02, 0xB0u8];
        assert!(
            parse_measurement_report(&legacy).is_none(),
            "the hand-rolled format must be gone from both ends, not tolerated on one"
        );
    }

    /// A serving entry with no ssb-Results still yields an actionable report: the
    /// neighbour results are the part that drives A3, and discarding the whole
    /// report would lose them.
    #[test]
    fn a_report_with_no_serving_measurement_still_yields_its_neighbours() {
        use nextgsim_rrc::procedures::measurement_report::{
            dbm_to_rsrp_range, encode_measurement_report, MeasCellResults, MeasResult2Nr,
            MeasResultCellNr, MeasResultNr, MeasResultServFreqNr, MeasurementReportParams,
        };

        let pdu = encode_measurement_report(&MeasurementReportParams {
            meas_id: 2,
            serv_freq_results: vec![MeasResultServFreqNr {
                serv_cell_index: 0,
                meas_result_serving_cell: MeasResultNr {
                    phys_cell_id: Some(1),
                    cell_results: MeasResultCellNr {
                        ssb_results: None,
                        csi_rs_results: None,
                    },
                    rs_index_results: None,
                },
                meas_result_best_neigh_cell: None,
            }],
            neigh_freq_results: vec![MeasResult2Nr {
                ssb_frequency_arfcn: None,
                ref_freq_csi_rs: None,
                meas_result_list: vec![MeasResultNr {
                    phys_cell_id: Some(7),
                    cell_results: MeasResultCellNr {
                        ssb_results: Some(MeasCellResults {
                            rsrp: Some(dbm_to_rsrp_range(-70.0)),
                            rsrq: None,
                            sinr: None,
                        }),
                        csi_rs_results: None,
                    },
                    rs_index_results: None,
                }],
            }],
            eutra_neigh_results: Vec::new(),
            enhanced_quantities: None,
        })
        .expect("encodes");

        let (_, report) = parse_measurement_report(&pdu).expect("parses");
        // The LITERAL -156, not `MISSING_RSRP_DBM`. Comparing against the constant
        // makes the assertion true by construction: a revert round that changed the
        // constant to -31 -- the STRONGEST level, which would make an unmeasured
        // serving cell beat every neighbour -- left this test green.
        assert_eq!(
            report.serving_rsrp, -156,
            "an absent serving measurement must substitute the WEAKEST representable \
             level (-156 dBm, the bottom of the NR RSRP-Range), so it can never make \
             a neighbour look worse than it is"
        );
        assert_eq!(
            MISSING_RSRP_DBM, -156,
            "and the constant must BE that value: pinned separately so the two \
             assertions cannot agree with each other while both drift"
        );
        assert_eq!(report.neighbors.len(), 1);
        assert_eq!(report.neighbors[0].pci, 7);
        assert_eq!(report.neighbors[0].rsrp, -70);
    }
}
