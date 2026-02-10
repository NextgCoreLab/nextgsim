//! RRC Task Implementation for UE
//!
//! This module implements the RRC (Radio Resource Control) task for the UE,
//! handling cell selection, RRC connection management, and handover.
//!
//! # Reference
//! - 3GPP TS 38.331: NR; RRC protocol specification
//! - 3GPP TS 38.304: UE procedures in Idle mode and RRC Inactive state
//! - UERANSIM: src/ue/rrc/task.cpp

use std::time::{Duration, Instant};

use tokio::sync::mpsc;
use tokio::time::interval;
use tracing::{debug, error, info, warn};

use crate::rrc::cell_selection::{
    CellChangeEvent, CellSelector, MibInfo, Plmn as CellPlmn, Sib1Info,
};
use crate::rrc::handover::{
    HandoverCommand, HandoverManager, parse_handover_command,
    build_reconfiguration_complete,
};
use crate::rrc::measurement::{MeasConfig, MeasurementManager, ReportTriggerType, MeasEventType, ReportTriggerConfig};
use crate::rrc::state::{RrcState, RrcStateMachine};
use crate::tasks::{
    NasMessage, RlsMessage, RlfCause, RrcMessage, Task, TaskMessage, UeTaskBase,
};
use nextgsim_common::OctetString;
use nextgsim_common::Plmn;
use nextgsim_rls::RrcChannel;

/// RRC cycle interval in milliseconds
const RRC_CYCLE_INTERVAL_MS: u64 = 2500;

/// Cell selection interval in milliseconds
const CELL_SELECTION_INTERVAL_MS: u64 = 1000;

/// UE-side NTN timing state
#[derive(Debug, Clone)]
pub struct UeNtnTiming {
    pub common_ta_us: u64,
    pub k_offset: u16,
    pub autonomous_ta: bool,
    pub max_doppler_hz: f64,
}

/// RRC Task for managing cell selection and RRC connections
pub struct RrcTask {
    task_base: UeTaskBase,
    /// RRC state machine
    state_machine: RrcStateMachine,
    /// Cell selector for cell selection/reselection
    cell_selector: CellSelector,
    /// Measurement manager for connected state measurements
    measurement_manager: MeasurementManager,
    /// Handover manager for mobility
    handover_manager: HandoverManager,
    /// PDU ID counter for RRC messages
    pdu_id_counter: u32,
    /// Current serving cell ID
    serving_cell_id: Option<i32>,
    /// Pending NAS PDU for initial message
    initial_nas_pdu: Option<OctetString>,
    /// RRC establishment cause
    establishment_cause: i64,
    /// Last cell selection attempt time
    last_cell_selection: Option<Instant>,
    /// NTN timing advance state (if operating via satellite)
    ntn_timing: Option<UeNtnTiming>,
}

impl RrcTask {
    pub fn new(task_base: UeTaskBase) -> Self {
        // Get HPLMN from config
        let hplmn = task_base.config.hplmn;
        let selected_plmn = CellPlmn::new(hplmn.mcc, hplmn.mnc, hplmn.long_mnc);

        let mut cell_selector = CellSelector::new();
        cell_selector.set_selected_plmn(Some(selected_plmn));

        Self {
            task_base,
            state_machine: RrcStateMachine::new(),
            cell_selector,
            measurement_manager: MeasurementManager::new(),
            handover_manager: HandoverManager::new(),
            pdu_id_counter: 0,
            serving_cell_id: None,
            initial_nas_pdu: None,
            establishment_cause: 3, // mo-Data
            last_cell_selection: None,
            ntn_timing: None,
        }
    }

    /// Get the next PDU ID for RRC message tracking
    fn next_pdu_id(&mut self) -> u32 {
        self.pdu_id_counter = self.pdu_id_counter.wrapping_add(1);
        if self.pdu_id_counter == 0 {
            self.pdu_id_counter = 1;
        }
        self.pdu_id_counter
    }

    /// Check if the given cell is the active serving cell
    fn is_active_cell(&self, cell_id: i32) -> bool {
        self.serving_cell_id == Some(cell_id)
    }

    /// Perform the RRC cycle (cell selection in idle, measurements in connected)
    async fn perform_cycle(&mut self) {
        match self.state_machine.state() {
            RrcState::Idle | RrcState::Inactive => {
                self.perform_cell_selection().await;
            }
            RrcState::Connected => {
                // In connected state, perform measurements for handover
                self.perform_measurements().await;
            }
        }
    }

    /// Perform cell selection
    async fn perform_cell_selection(&mut self) {
        // Check if enough time has passed since last selection
        if let Some(last) = self.last_cell_selection {
            if last.elapsed() < Duration::from_millis(CELL_SELECTION_INTERVAL_MS) {
                return;
            }
        }
        self.last_cell_selection = Some(Instant::now());

        // Run cell selection algorithm
        if let Some(selected_cell) = self.cell_selector.perform_cell_selection() {
            let old_cell = self.serving_cell_id;
            let old_tai = self.get_current_tai();

            // Update serving cell
            self.serving_cell_id = Some(selected_cell.cell_id);

            info!(
                "Cell selection complete: cell_id={}, plmn={}, tac={}, category={:?}",
                selected_cell.cell_id, selected_cell.plmn, selected_cell.tac, selected_cell.category
            );

            // Notify RLS of the new serving cell
            if let Err(e) = self
                .task_base
                .rls_tx
                .send(RlsMessage::AssignCurrentCell {
                    cell_id: selected_cell.cell_id,
                })
                .await
            {
                error!("Failed to notify RLS of cell change: {}", e);
            }

            // If cell changed, notify NAS
            if old_cell != Some(selected_cell.cell_id) {
                if let Err(e) = self
                    .task_base
                    .nas_tx
                    .send(NasMessage::ActiveCellChanged { previous_tai: old_tai })
                    .await
                {
                    error!("Failed to notify NAS of cell change: {}", e);
                }
            }
        }
    }

    /// Get the current TAI (Tracking Area Identity)
    fn get_current_tai(&self) -> nextgsim_common::types::Tai {
        if let Some(cell_id) = self.serving_cell_id {
            if let Some(cell) = self.cell_selector.get_cell(cell_id) {
                return nextgsim_common::types::Tai {
                    plmn: Plmn::new(
                        cell.sib1.plmn.mcc,
                        cell.sib1.plmn.mnc,
                        cell.sib1.plmn.long_mnc,
                    ),
                    tac: cell.sib1.tac,
                };
            }
        }
        nextgsim_common::types::Tai::default()
    }

    /// Perform measurements for handover (in connected state)
    async fn perform_measurements(&mut self) {
        // Update measurement manager with serving cell
        self.measurement_manager.set_serving_cell(self.serving_cell_id);

        // Update measurements from cell selector
        let cells = self.cell_selector.cells();
        for (&cell_id, cell) in cells.iter() {
            self.measurement_manager.update_measurement(cell_id, cell.dbm);
        }

        // Evaluate measurement events
        self.measurement_manager.evaluate_events();

        // Process any pending measurement reports
        let reports = self.measurement_manager.take_pending_reports();
        for report in reports {
            self.send_measurement_report(&report).await;
        }
    }

    /// Send measurement report to the network via RRC
    async fn send_measurement_report(&mut self, report: &crate::rrc::measurement::MeasurementReport) {
        // Build simplified measurement report message
        // In real implementation, this would be proper ASN.1 encoding
        let mut rrc_pdu = Vec::with_capacity(32);
        rrc_pdu.push(0x0B); // MeasurementReport message type
        rrc_pdu.push(report.meas_id);

        // Serving cell result
        rrc_pdu.push((report.serving_cell.pci >> 8) as u8);
        rrc_pdu.push(report.serving_cell.pci as u8);
        let rsrp = report.serving_cell.rsrp.unwrap_or(-120) as i8;
        rrc_pdu.push(rsrp as u8);

        // Number of neighbor cells
        rrc_pdu.push(report.neighbor_cells.len() as u8);

        // Neighbor cell results
        for neighbor in &report.neighbor_cells {
            rrc_pdu.push((neighbor.pci >> 8) as u8);
            rrc_pdu.push(neighbor.pci as u8);
            let rsrp = neighbor.rsrp.unwrap_or(-120) as i8;
            rrc_pdu.push(rsrp as u8);
        }

        let pdu = OctetString::from_slice(&rrc_pdu);
        info!(
            "Sending measurement report: meas_id={}, serving_rsrp={:?}, neighbors={}",
            report.meas_id,
            report.serving_cell.rsrp,
            report.neighbor_cells.len()
        );
        self.send_uplink_rrc(RrcChannel::UlDcch, pdu).await;
    }

    /// Configure measurements (called when receiving RRC Reconfiguration with measConfig)
    #[allow(dead_code)]
    fn configure_measurements(&mut self, config: MeasConfig) {
        info!(
            "Adding measurement config: meas_id={}, event={:?}",
            config.meas_id, config.trigger_config.trigger_type
        );
        self.measurement_manager.add_config(config);
    }

    /// Setup default A3 measurement for handover
    fn setup_default_measurements(&mut self) {
        // Default A3 event configuration for handover
        let config = MeasConfig {
            meas_id: 1,
            meas_object_id: 1,
            report_config_id: 1,
            quantity: crate::rrc::measurement::MeasQuantity::SsRsrp,
            trigger_config: ReportTriggerConfig {
                trigger_type: ReportTriggerType::Event(MeasEventType::A3),
                threshold: None,
                threshold1: None,
                threshold2: None,
                a3_offset: Some(3), // Neighbor 3dB better than serving
                hysteresis: 2,      // 1dB hysteresis
                time_to_trigger: 640, // 640ms
            },
            report_amount: 8,
            report_interval: 480,
            max_report_cells: 4,
        };
        self.measurement_manager.add_config(config);
    }

    /// Handle signal change from RLS
    async fn handle_signal_changed(&mut self, cell_id: i32, dbm: i32) {
        let event = self.cell_selector.handle_signal_change(cell_id, dbm);

        match event {
            CellChangeEvent::CellDetected(id) => {
                debug!("Cell detected: cell_id={}", id);
                // In simulation, we need to provide system info for the cell
                // For now, use default values that make the cell selectable
                self.provide_simulated_system_info(id);
            }
            CellChangeEvent::CellLost(id) => {
                debug!("Cell lost: cell_id={}", id);
                // Remove from measurement manager
                self.measurement_manager.remove_measurement(id);
            }
            CellChangeEvent::ActiveCellLost(cell_info) => {
                warn!(
                    "Active cell lost: cell_id={}, triggering cell selection",
                    cell_info.cell_id
                );
                self.serving_cell_id = None;
                self.measurement_manager.set_serving_cell(None);
                // Trigger immediate cell selection
                self.last_cell_selection = None;
            }
            CellChangeEvent::SignalUpdated(id, dbm) => {
                debug!("Signal updated: cell_id={}, dbm={}", id, dbm);
                // Update measurement manager
                self.measurement_manager.update_measurement(id, dbm);
            }
            CellChangeEvent::None => {}
        }
    }

    /// Provide simulated system information for a cell
    fn provide_simulated_system_info(&mut self, cell_id: i32) {
        // Get HPLMN from config to use for the cell
        let hplmn = self.task_base.config.hplmn;

        // Update MIB - cell not barred
        let mib = MibInfo {
            has_mib: true,
            is_barred: false,
            is_intra_freq_reselect_allowed: true,
        };
        self.cell_selector.update_mib(cell_id, mib);

        // Update SIB1 with PLMN matching our home network
        let sib1 = Sib1Info {
            has_sib1: true,
            is_reserved: false,
            nci: cell_id as i64,
            tac: 1, // Default TAC
            plmn: CellPlmn::new(hplmn.mcc, hplmn.mnc, hplmn.long_mnc),
            q_rx_lev_min: -70, // Reasonable minimum
            q_rx_lev_min_offset: None,
            q_qual_min: None,
        };
        self.cell_selector.update_sib1(cell_id, sib1);

        debug!(
            "Simulated system info provided for cell {}: plmn={}-{}",
            cell_id, hplmn.mcc, hplmn.mnc
        );
    }

    /// Handle downlink RRC message from RLS
    async fn handle_downlink_rrc(
        &mut self,
        cell_id: i32,
        channel: RrcChannel,
        pdu: OctetString,
    ) {
        if pdu.is_empty() {
            warn!("Empty downlink RRC PDU");
            return;
        }

        debug!(
            "Downlink RRC: cell_id={}, channel={:?}, len={}",
            cell_id,
            channel,
            pdu.len()
        );

        match channel {
            RrcChannel::DlCcch => {
                self.handle_dl_ccch_message(cell_id, &pdu).await;
            }
            RrcChannel::DlDcch => {
                self.handle_dl_dcch_message(cell_id, &pdu).await;
            }
            _ => {
                warn!("Unexpected downlink channel: {:?}", channel);
            }
        }
    }

    /// Handle DL-CCCH message (RRC Setup, RRC Reject)
    async fn handle_dl_ccch_message(&mut self, cell_id: i32, pdu: &OctetString) {
        let bytes = pdu.data();
        if bytes.is_empty() {
            return;
        }

        // Simplified parsing: check message type
        let msg_type = bytes[0] & 0x0F;

        match msg_type {
            0x00 => {
                // RRC Setup
                info!("Received RRC Setup from cell {}", cell_id);
                self.handle_rrc_setup(cell_id, pdu).await;
            }
            0x01 => {
                // RRC Reject
                warn!("Received RRC Reject from cell {}", cell_id);
                self.handle_rrc_reject(cell_id).await;
            }
            _ => {
                debug!("Unhandled DL-CCCH message type: {:#x}", msg_type);
            }
        }
    }

    /// Handle DL-DCCH message (DL Information Transfer, RRC Release, etc.)
    async fn handle_dl_dcch_message(&mut self, cell_id: i32, pdu: &OctetString) {
        if !self.is_active_cell(cell_id) {
            debug!("Ignoring DL-DCCH from non-active cell {}", cell_id);
            return;
        }

        let bytes = pdu.data();
        if bytes.is_empty() {
            return;
        }

        let msg_type = bytes[0] & 0x0F;

        match msg_type {
            0x04 => {
                // DL Information Transfer - forward NAS to NAS task
                if bytes.len() > 3 {
                    let nas_pdu = OctetString::from_slice(&bytes[3..]);
                    self.forward_nas_to_nas_task(nas_pdu).await;
                }
            }
            0x0D => {
                // RRC Release
                info!("Received RRC Release from cell {}", cell_id);
                self.handle_rrc_release().await;
            }
            0x00 => {
                // RRC Reconfiguration
                debug!("Received RRC Reconfiguration from cell {}", cell_id);
                self.handle_rrc_reconfiguration(cell_id, pdu).await;
            }
            _ => {
                // Check if this is a raw NAS PDU (EPD = 0x7E or 0x2E)
                if bytes.len() >= 2 && (bytes[0] == 0x7E || bytes[0] == 0x2E) {
                    debug!("Received raw NAS PDU, forwarding to NAS task");
                    self.forward_nas_to_nas_task(pdu.clone()).await;
                } else {
                    debug!("Unhandled DL-DCCH message type: {:#x}", msg_type);
                }
            }
        }
    }

    /// Handle RRC Setup message
    async fn handle_rrc_setup(&mut self, cell_id: i32, _pdu: &OctetString) {
        // Transition to connected state
        if let Err(e) = self.state_machine.on_rrc_setup() {
            warn!("Failed to transition to connected state: {}", e);
            return;
        }

        self.serving_cell_id = Some(cell_id);
        info!("RRC connection established with cell {}", cell_id);

        // Set up default measurements for handover
        self.setup_default_measurements();
        self.measurement_manager.set_serving_cell(Some(cell_id));

        // Send RRC Setup Complete with initial NAS
        self.send_rrc_setup_complete().await;

        // Notify NAS of connection setup
        if let Err(e) = self
            .task_base
            .nas_tx
            .send(NasMessage::RrcConnectionSetup)
            .await
        {
            error!("Failed to notify NAS of RRC setup: {}", e);
        }
    }

    /// Send RRC Setup Complete message
    async fn send_rrc_setup_complete(&mut self) {
        let nas_pdu = self.initial_nas_pdu.take().unwrap_or_default();

        // Build simplified RRC Setup Complete
        let mut rrc_pdu = Vec::with_capacity(nas_pdu.len() + 3);
        rrc_pdu.push(0x04); // RRCSetupComplete message type
        rrc_pdu.push(0x00); // Transaction ID
        rrc_pdu.push(0x01); // Selected PLMN Identity = 1
        rrc_pdu.extend_from_slice(nas_pdu.data());

        let pdu = OctetString::from_slice(&rrc_pdu);
        self.send_uplink_rrc(RrcChannel::UlDcch, pdu).await;
    }

    /// Handle RRC Reject message
    async fn handle_rrc_reject(&mut self, _cell_id: i32) {
        // Notify NAS of establishment failure
        if let Err(e) = self
            .task_base
            .nas_tx
            .send(NasMessage::RrcEstablishmentFailure)
            .await
        {
            error!("Failed to notify NAS of RRC reject: {}", e);
        }
    }

    /// Handle RRC Release message
    async fn handle_rrc_release(&mut self) {
        // Transition to idle state
        if let Err(e) = self.state_machine.on_rrc_release() {
            warn!("Failed to transition to idle state: {}", e);
        }

        info!("RRC connection released");

        // Notify NAS of connection release
        if let Err(e) = self
            .task_base
            .nas_tx
            .send(NasMessage::RrcConnectionRelease)
            .await
        {
            error!("Failed to notify NAS of RRC release: {}", e);
        }
    }

    /// Handle RRC Reconfiguration message (for handover)
    async fn handle_rrc_reconfiguration(&mut self, cell_id: i32, pdu: &OctetString) {
        let bytes = pdu.data();

        // Check if this is a handover reconfiguration
        if let Some(ho_command) = parse_handover_command(bytes) {
            info!(
                "Received handover command: target_cell={}, target_pci={}",
                ho_command.target_cell.cell_id, ho_command.target_cell.pci
            );
            self.handle_handover_command(cell_id, ho_command).await;
            return;
        }

        // Regular reconfiguration (no handover)
        debug!("RRC Reconfiguration (no handover) - sending RRC Reconfiguration Complete");

        // Extract transaction ID (simplified)
        let transaction_id = if bytes.len() > 1 { bytes[1] } else { 0 };

        // Build RRC Reconfiguration Complete
        let rrc_pdu = OctetString::from_slice(&build_reconfiguration_complete(transaction_id));
        self.send_uplink_rrc(RrcChannel::UlDcch, rrc_pdu).await;
    }

    /// Handle handover command from RRC Reconfiguration
    async fn handle_handover_command(&mut self, source_cell_id: i32, command: HandoverCommand) {
        let target_cell_id = command.target_cell.cell_id;
        let transaction_id = command.transaction_id;

        // Start handover in the handover manager
        self.handover_manager.start_handover(source_cell_id, command);

        // Check if we have signal to the target cell
        if self.cell_selector.has_signal_to_cell(target_cell_id) {
            // Start synchronization
            self.handover_manager.start_synchronization();

            // In simulation, we assume sync is instant
            self.handover_manager.sync_complete();

            // Complete handover
            if let Some(new_cell_id) = self.handover_manager.complete() {
                // Update serving cell
                let old_cell_id = self.serving_cell_id;
                self.serving_cell_id = Some(new_cell_id);

                // Update measurement manager
                self.measurement_manager.set_serving_cell(Some(new_cell_id));

                // Notify RLS of new serving cell
                if let Err(e) = self
                    .task_base
                    .rls_tx
                    .send(RlsMessage::AssignCurrentCell { cell_id: new_cell_id })
                    .await
                {
                    error!("Failed to notify RLS of handover: {}", e);
                }

                info!(
                    "Handover successful: {} -> {}",
                    old_cell_id.unwrap_or(-1), new_cell_id
                );

                // Send RRC Reconfiguration Complete
                let rrc_pdu = OctetString::from_slice(&build_reconfiguration_complete(transaction_id));
                self.send_uplink_rrc(RrcChannel::UlDcch, rrc_pdu).await;
            }
        } else {
            // Target cell not reachable - handover failure
            warn!("Handover failed: target cell {} not in coverage", target_cell_id);
            if let Some(source_cell) = self.handover_manager.fail(
                crate::rrc::handover::HandoverFailureCause::TargetCellUnreachable
            ) {
                // Stay on source cell
                self.serving_cell_id = Some(source_cell);
            }

            // Trigger RRC re-establishment
            self.handle_handover_failure().await;
        }
    }

    /// Handle handover failure - initiate re-establishment
    async fn handle_handover_failure(&mut self) {
        warn!("Initiating RRC re-establishment after handover failure");

        // Notify NAS of radio link failure
        if let Err(e) = self.task_base.nas_tx.send(NasMessage::RadioLinkFailure).await {
            error!("Failed to notify NAS of handover failure: {}", e);
        }
    }

    /// Forward NAS PDU to NAS task
    async fn forward_nas_to_nas_task(&self, pdu: OctetString) {
        if let Err(e) = self
            .task_base
            .nas_tx
            .send(NasMessage::NasDelivery { pdu })
            .await
        {
            error!("Failed to forward NAS to NAS task: {}", e);
        }
    }

    /// Handle uplink NAS delivery from NAS task
    async fn handle_uplink_nas_delivery(&mut self, pdu_id: u32, pdu: OctetString) {
        // If not connected, this is initial NAS - start connection establishment
        if self.state_machine.state() == RrcState::Idle {
            self.initial_nas_pdu = Some(pdu.clone());
            self.start_connection_establishment(pdu).await;
        } else if self.state_machine.state() == RrcState::Connected {
            // Build UL Information Transfer
            let mut rrc_pdu = Vec::with_capacity(pdu.len() + 2);
            rrc_pdu.push(0x08); // ULInformationTransfer message type
            rrc_pdu.push(0x00); // Critical extensions
            rrc_pdu.extend_from_slice(pdu.data());

            let ul_info = OctetString::from_slice(&rrc_pdu);
            self.send_uplink_rrc_with_id(RrcChannel::UlDcch, pdu_id, ul_info).await;
        }
    }

    /// Start RRC connection establishment
    async fn start_connection_establishment(&mut self, _nas_pdu: OctetString) {
        if self.state_machine.state() != RrcState::Idle {
            warn!("Cannot start connection establishment: not in idle state");
            self.handle_establishment_failure().await;
            return;
        }

        if self.serving_cell_id.is_none() {
            warn!("Cannot start connection establishment: no serving cell");
            self.handle_establishment_failure().await;
            return;
        }

        info!("Starting RRC connection establishment");

        // Build simplified RRC Setup Request
        // Format: [msg_type, initial_ue_id (5 bytes), establishment_cause]
        let mut rrc_pdu = Vec::with_capacity(8);
        rrc_pdu.push(0x00); // RRCSetupRequest message type + random value indicator
        // 5-byte random Initial UE Identity
        let random_id: u64 = rand::random::<u64>() & 0x7FFFFFFFFF; // 39-bit value
        rrc_pdu.extend_from_slice(&random_id.to_be_bytes()[3..8]);
        rrc_pdu.push(self.establishment_cause as u8);

        let pdu = OctetString::from_slice(&rrc_pdu);
        self.send_uplink_rrc(RrcChannel::UlCcch, pdu).await;
    }

    /// Handle establishment failure
    async fn handle_establishment_failure(&self) {
        if let Err(e) = self
            .task_base
            .nas_tx
            .send(NasMessage::RrcEstablishmentFailure)
            .await
        {
            error!("Failed to notify NAS of establishment failure: {}", e);
        }
    }

    /// Handle radio link failure
    async fn handle_radio_link_failure(&mut self, cause: RlfCause) {
        warn!("Radio link failure: {:?}", cause);

        // Transition to idle
        let _ = self.state_machine.on_rrc_release();
        self.serving_cell_id = None;

        // Notify NAS
        if let Err(e) = self.task_base.nas_tx.send(NasMessage::RadioLinkFailure).await {
            error!("Failed to notify NAS of radio link failure: {}", e);
        }
    }

    /// Handle local release connection request from NAS
    async fn handle_local_release_connection(&mut self, _treat_barred: bool) {
        info!("Local release connection requested");

        // Transition to idle
        let _ = self.state_machine.on_rrc_release();
        self.serving_cell_id = None;

        // Reset STI in RLS
        if let Err(e) = self.task_base.rls_tx.send(RlsMessage::ResetSti).await {
            error!("Failed to send ResetSti to RLS: {}", e);
        }
    }

    /// Send uplink RRC message
    async fn send_uplink_rrc(&mut self, channel: RrcChannel, pdu: OctetString) {
        let pdu_id = self.next_pdu_id();
        self.send_uplink_rrc_with_id(channel, pdu_id, pdu).await;
    }

    /// Send uplink RRC message with specific PDU ID
    async fn send_uplink_rrc_with_id(&self, channel: RrcChannel, pdu_id: u32, pdu: OctetString) {
        let msg = RlsMessage::RrcPduDelivery { channel, pdu_id, pdu };
        if let Err(e) = self.task_base.rls_tx.send(msg).await {
            error!("Failed to send RRC message to RLS: {}", e);
        }
    }
}

#[async_trait::async_trait]
impl Task for RrcTask {
    type Message = RrcMessage;

    async fn run(&mut self, mut rx: mpsc::Receiver<TaskMessage<Self::Message>>) {
        info!("RRC task started");

        let mut cycle_timer = interval(Duration::from_millis(RRC_CYCLE_INTERVAL_MS));

        loop {
            tokio::select! {
                Some(msg) = rx.recv() => {
                    match msg {
                        TaskMessage::Message(rrc_msg) => match rrc_msg {
                            RrcMessage::LocalReleaseConnection { treat_barred } => {
                                self.handle_local_release_connection(treat_barred).await;
                            }
                            RrcMessage::UplinkNasDelivery { pdu_id, pdu } => {
                                self.handle_uplink_nas_delivery(pdu_id, pdu).await;
                            }
                            RrcMessage::RrcNotify => {
                                debug!("RRC notify received");
                            }
                            RrcMessage::PerformUac { access_category, access_identities } => {
                                debug!("UAC check: category={}, identities={}", access_category, access_identities);
                                // For now, always allow access
                            }
                            RrcMessage::DownlinkRrcDelivery { cell_id, channel, pdu } => {
                                self.handle_downlink_rrc(cell_id, channel, pdu).await;
                            }
                            RrcMessage::SignalChanged { cell_id, dbm } => {
                                self.handle_signal_changed(cell_id, dbm).await;
                            }
                            RrcMessage::RadioLinkFailure { cause } => {
                                self.handle_radio_link_failure(cause).await;
                            }
                            RrcMessage::TriggerCycle => {
                                self.perform_cycle().await;
                            }
                            RrcMessage::NtnTimingAdvanceReceived {
                                common_ta_us, k_offset, autonomous_ta, max_doppler_hz,
                            } => {
                                info!(
                                    "UE RRC: NTN timing advance received: TA={}us, k_offset={}, autonomous={}, doppler={}Hz",
                                    common_ta_us, k_offset, autonomous_ta, max_doppler_hz
                                );
                                self.ntn_timing = Some(UeNtnTiming {
                                    common_ta_us,
                                    k_offset,
                                    autonomous_ta,
                                    max_doppler_hz,
                                });
                            }
                        },
                        TaskMessage::Shutdown => {
                            info!("RRC task received shutdown signal");
                            break;
                        }
                    }
                }
                _ = cycle_timer.tick() => {
                    self.perform_cycle().await;
                }
            }
        }

        info!(
            "RRC task stopped in {:?} state with {} cells",
            self.state_machine.state(),
            self.cell_selector.cells().len()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nextgsim_common::config::UeConfig;

    fn test_config() -> UeConfig {
        UeConfig::default()
    }

    #[test]
    fn test_rrc_task_creation() {
        let config = test_config();
        let (task_base, _app_rx, _nas_rx, _rrc_rx, _rls_rx) =
            UeTaskBase::new(config, 16);
        let task = RrcTask::new(task_base);
        assert_eq!(task.state_machine.state(), RrcState::Idle);
        assert!(task.serving_cell_id.is_none());
    }

    #[test]
    fn test_pdu_id_generation() {
        let config = test_config();
        let (task_base, _app_rx, _nas_rx, _rrc_rx, _rls_rx) =
            UeTaskBase::new(config, 16);
        let mut task = RrcTask::new(task_base);
        assert_eq!(task.next_pdu_id(), 1);
        assert_eq!(task.next_pdu_id(), 2);
        assert_eq!(task.next_pdu_id(), 3);
    }

    #[test]
    fn test_is_active_cell() {
        let config = test_config();
        let (task_base, _app_rx, _nas_rx, _rrc_rx, _rls_rx) =
            UeTaskBase::new(config, 16);
        let mut task = RrcTask::new(task_base);
        assert!(!task.is_active_cell(1));

        task.serving_cell_id = Some(1);
        assert!(task.is_active_cell(1));
        assert!(!task.is_active_cell(2));
    }
}
