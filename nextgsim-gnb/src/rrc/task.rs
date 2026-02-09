//! RRC Task Implementation

use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use crate::tasks::{
    GnbTaskBase, GutiMobileIdentity, NgapMessage, RlsMessage, RrcMessage, Task, TaskMessage,
};
use nextgsim_common::OctetString;
use nextgsim_rls::RrcChannel;

use super::connection::RrcConnectionManager;
use super::ue_context::RrcUeContextManager;

/// RRC Task for managing UE RRC connections
pub struct RrcTask {
    task_base: GnbTaskBase,
    ue_manager: RrcUeContextManager,
    connection_manager: RrcConnectionManager,
    pdu_id_counter: u32,
}

impl RrcTask {
    pub fn new(task_base: GnbTaskBase) -> Self {
        Self {
            task_base,
            ue_manager: RrcUeContextManager::new(),
            connection_manager: RrcConnectionManager::new(),
            pdu_id_counter: 0,
        }
    }

    fn next_pdu_id(&mut self) -> u32 {
        self.pdu_id_counter = self.pdu_id_counter.wrapping_add(1);
        if self.pdu_id_counter == 0 {
            self.pdu_id_counter = 1;
        }
        self.pdu_id_counter
    }

    fn handle_radio_power_on(&mut self) {
        info!("Radio power on - cell is now active");
        self.connection_manager.set_barred(false);
    }

    fn handle_signal_detected(&mut self, ue_id: i32) {
        debug!("Signal detected from UE[{}]", ue_id);
    }

    async fn handle_uplink_rrc(&mut self, ue_id: i32, channel: RrcChannel, data: OctetString) {
        debug!("Uplink RRC: ue_id={}, channel={:?}, len={}", ue_id, channel, data.len());

        match channel {
            RrcChannel::UlCcch | RrcChannel::UlCcch1 => {
                self.handle_ul_ccch_message(ue_id, &data).await;
            }
            RrcChannel::UlDcch => {
                self.handle_ul_dcch_message(ue_id, &data).await;
            }
            _ => warn!("Unexpected uplink channel: {:?}", channel),
        }
    }

    async fn handle_ul_ccch_message(&mut self, ue_id: i32, data: &OctetString) {
        if data.len() < 6 {
            warn!("UL-CCCH message too short: {} bytes", data.len());
            return;
        }

        let bytes = data.data();
        let initial_id = i64::from_be_bytes([
            0, 0, 0,
            bytes.get(1).copied().unwrap_or(0),
            bytes.get(2).copied().unwrap_or(0),
            bytes.get(3).copied().unwrap_or(0),
            bytes.get(4).copied().unwrap_or(0),
            bytes.get(5).copied().unwrap_or(0),
        ]) & 0x7FFFFFFFFF;
        
        let is_stmsi = (bytes.get(0).copied().unwrap_or(0) & 0x80) != 0;
        let establishment_cause = bytes.get(6).copied().unwrap_or(3) as i64;

        if let Some(result) = self.connection_manager.process_rrc_setup_request(
            &mut self.ue_manager, ue_id, initial_id, is_stmsi, establishment_cause,
        ) {
            self.send_rrc_message(result.ue_id, result.channel, result.rrc_setup_pdu).await;
        }
    }

    async fn handle_ul_dcch_message(&mut self, ue_id: i32, data: &OctetString) {
        if data.is_empty() {
            warn!("Empty UL-DCCH message");
            return;
        }

        let bytes = data.data();
        let message_type = bytes[0] & 0x0F;

        match message_type {
            0x04 => self.handle_rrc_setup_complete(ue_id, data).await,
            0x08 => self.handle_ul_information_transfer(ue_id, data).await,
            _ => {
                // Check if this looks like a raw NAS PDU
                // EPD = 0x7E for 5GMM (Mobility Management), 0x2E for 5GSM (Session Management)
                // Some UE simulators send NAS directly without proper RRC encapsulation
                let is_nas_pdu = bytes.len() >= 3 && (bytes[0] == 0x7E || bytes[0] == 0x2E);
                if is_nas_pdu {
                    // Check if UE context already exists (meaning Initial UE Message was already sent)
                    if let Some(ctx) = self.ue_manager.try_find_ue(ue_id) {
                        if ctx.is_connected() {
                            // UE is already connected, send as Uplink NAS Transport
                            info!("Received raw NAS PDU on UL-DCCH from connected UE, sending as Uplink NAS (ue_id={}, epd=0x{:02x})", ue_id, bytes[0]);
                            self.send_uplink_nas_delivery(ue_id, data.clone()).await;
                            return;
                        }
                    }

                    // First message - send as Initial UE Message
                    info!("Received raw NAS PDU on UL-DCCH, forwarding to NGAP as Initial UE (ue_id={})", ue_id);
                    // Create or get UE context
                    if self.ue_manager.try_find_ue(ue_id).is_none() {
                        // Auto-create UE context for this UE
                        let ctx = self.ue_manager.create_ue(ue_id);
                        ctx.on_setup_complete(); // Mark as connected
                    }
                    // Forward as Initial NAS
                    self.send_initial_nas_delivery(ue_id, data.clone(), 3, None).await; // cause=3 (mo-Data)
                } else {
                    debug!("Unhandled UL-DCCH message type: {:#x}", message_type);
                }
            }
        }
    }

    async fn handle_rrc_setup_complete(&mut self, ue_id: i32, data: &OctetString) {
        let bytes = data.data();
        if bytes.len() < 3 {
            warn!("RRC Setup Complete too short");
            return;
        }

        let transaction_id = bytes[1];
        let nas_pdu = if bytes.len() > 3 {
            OctetString::from_slice(&bytes[3..])
        } else {
            OctetString::new()
        };

        if let Some(result) = self.connection_manager.process_rrc_setup_complete(
            &mut self.ue_manager, ue_id, transaction_id, nas_pdu, None,
        ) {
            self.send_initial_nas_delivery(
                result.ue_id, result.nas_pdu, result.establishment_cause, result.s_tmsi,
            ).await;
        }
    }

    async fn handle_ul_information_transfer(&mut self, ue_id: i32, data: &OctetString) {
        let bytes = data.data();
        if bytes.len() <= 2 {
            warn!("UL Information Transfer too short");
            return;
        }
        let nas_pdu = OctetString::from_slice(&bytes[2..]);
        self.send_uplink_nas_delivery(ue_id, nas_pdu).await;
    }

    async fn handle_nas_delivery(&mut self, ue_id: i32, nas_pdu: OctetString) {
        let ctx = match self.ue_manager.try_find_ue(ue_id) {
            Some(c) => c,
            None => {
                warn!("NAS delivery for unknown UE[{}]", ue_id);
                return;
            }
        };

        if !ctx.is_connected() {
            warn!("NAS delivery for non-connected UE[{}]", ue_id);
            return;
        }

        let dl_info_transfer = self.build_dl_information_transfer(&nas_pdu);
        self.send_rrc_message(ue_id, RrcChannel::DlDcch, dl_info_transfer).await;
    }

    fn build_dl_information_transfer(&self, nas_pdu: &OctetString) -> OctetString {
        let mut pdu = Vec::with_capacity(nas_pdu.len() + 4);
        pdu.push(0x04);
        pdu.push(0x00);
        pdu.push(0x00);
        pdu.extend_from_slice(nas_pdu.data());
        OctetString::from_slice(&pdu)
    }

    async fn handle_an_release(&mut self, ue_id: i32) {
        if let Some(result) = self.connection_manager.initiate_rrc_release(&mut self.ue_manager, ue_id) {
            self.send_rrc_message(result.ue_id, result.channel, result.rrc_release_pdu).await;
        }
        self.ue_manager.delete_ue(ue_id);
    }

    async fn handle_paging(&mut self, _ue_paging_tmsi: Vec<u8>, _tai_list_for_paging: Vec<u8>) {
        debug!("Paging request received");
    }

    async fn send_rrc_message(&mut self, ue_id: i32, channel: RrcChannel, data: OctetString) {
        let pdu_id = self.next_pdu_id();
        let msg = RlsMessage::DownlinkRrc { ue_id, rrc_channel: channel, pdu_id, data };
        if let Err(e) = self.task_base.rls_tx.send(msg).await {
            error!("Failed to send RRC message to RLS: {}", e);
        }
    }

    async fn send_initial_nas_delivery(
        &self, ue_id: i32, pdu: OctetString, rrc_establishment_cause: i64, s_tmsi: Option<GutiMobileIdentity>,
    ) {
        let msg = NgapMessage::InitialNasDelivery { ue_id, pdu, rrc_establishment_cause, s_tmsi };
        if let Err(e) = self.task_base.ngap_tx.send(msg).await {
            error!("Failed to send Initial NAS to NGAP: {}", e);
        }
    }

    async fn send_uplink_nas_delivery(&self, ue_id: i32, pdu: OctetString) {
        let msg = NgapMessage::UplinkNasDelivery { ue_id, pdu };
        if let Err(e) = self.task_base.ngap_tx.send(msg).await {
            error!("Failed to send Uplink NAS to NGAP: {}", e);
        }
    }
}

#[async_trait::async_trait]
impl Task for RrcTask {
    type Message = RrcMessage;

    async fn run(&mut self, mut rx: mpsc::Receiver<TaskMessage<Self::Message>>) {
        info!("RRC task started");

        loop {
            tokio::select! {
                Some(msg) = rx.recv() => {
                    match msg {
                        TaskMessage::Message(rrc_msg) => match rrc_msg {
                            RrcMessage::RadioPowerOn => self.handle_radio_power_on(),
                            RrcMessage::SignalDetected { ue_id } => self.handle_signal_detected(ue_id),
                            RrcMessage::UplinkRrc { ue_id, rrc_channel, data } => {
                                self.handle_uplink_rrc(ue_id, rrc_channel, data).await;
                            }
                            RrcMessage::NasDelivery { ue_id, pdu } => {
                                self.handle_nas_delivery(ue_id, pdu).await;
                            }
                            RrcMessage::AnRelease { ue_id } => {
                                self.handle_an_release(ue_id).await;
                            }
                            RrcMessage::Paging { ue_paging_tmsi, tai_list_for_paging } => {
                                self.handle_paging(ue_paging_tmsi, tai_list_for_paging).await;
                            }
                        },
                        TaskMessage::Shutdown => {
                            info!("RRC task received shutdown signal");
                            break;
                        }
                    }
                }
                else => {
                    info!("RRC task channel closed");
                    break;
                }
            }
        }

        info!("RRC task stopped with {} UE contexts", self.ue_manager.count());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nextgsim_common::config::GnbConfig;
    use nextgsim_common::Plmn;

    fn test_config() -> GnbConfig {
        GnbConfig {
            nci: 0x000000010,
            gnb_id_length: 32,
            plmn: Plmn::new(001, 01, false),
            tac: 1,
            nssai: vec![],
            amf_configs: vec![],
            link_ip: "127.0.0.1".parse().unwrap(),
            ngap_ip: "127.0.0.1".parse().unwrap(),
            gtp_ip: "127.0.0.1".parse().unwrap(),
            gtp_advertise_ip: None,
            ignore_stream_ids: false, upf_addr: None, upf_port: 2152,
            pqc_config: nextgsim_common::config::PqcConfig::default(),
        }
    }

    #[test]
    fn test_rrc_task_creation() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, 16);
        let task = RrcTask::new(task_base);
        assert_eq!(task.ue_manager.count(), 0);
    }

    #[test]
    fn test_radio_power_on() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, 16);
        let mut task = RrcTask::new(task_base);
        assert!(task.connection_manager.is_barred());
        task.handle_radio_power_on();
        assert!(!task.connection_manager.is_barred());
    }

    #[test]
    fn test_pdu_id_generation() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, 16);
        let mut task = RrcTask::new(task_base);
        assert_eq!(task.next_pdu_id(), 1);
        assert_eq!(task.next_pdu_id(), 2);
        assert_eq!(task.next_pdu_id(), 3);
    }

    #[test]
    fn test_build_dl_information_transfer() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, 16);
        let task = RrcTask::new(task_base);
        let nas_pdu = OctetString::from_slice(&[0x7E, 0x00, 0x41]);
        let dl_info = task.build_dl_information_transfer(&nas_pdu);
        assert_eq!(dl_info.len(), 6);
        assert_eq!(dl_info.data()[0], 0x04);
    }
}
