//! RLS Task Implementation
//!
//! This module implements the RLS (Radio Link Simulation) task for the gNB,
//! handling UE discovery, RRC message relay, and user plane data relay.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use tokio::net::UdpSocket;
use tokio::sync::mpsc;
use tokio::time::interval;
use tracing::{debug, error, info, warn};

use crate::tasks::{
    GnbTaskBase, GtpMessage, RlsMessage, RrcMessage, Task, TaskMessage,
};
use nextgsim_common::OctetString;
use nextgsim_rls::{
    codec, GnbCellTracker, GnbTrackerEvent, PduType, RlsHeartbeatAck, RlsMessage as RlsProtocolMessage,
    RlsPduTransmission, RlsPduTransmissionAck, RrcChannel, Vector3,
};

/// Default RLS port for gNB
pub const DEFAULT_RLS_PORT: u16 = 4997;

/// Heartbeat check interval in milliseconds
const HEARTBEAT_CHECK_INTERVAL_MS: u64 = 500;

/// Maximum UDP receive buffer size
const UDP_BUFFER_SIZE: usize = 65535;

/// RLS Task for managing radio link simulation
pub struct RlsTask {
    /// Task base for inter-task communication
    task_base: GnbTaskBase,
    /// Cell tracker for UE discovery
    cell_tracker: GnbCellTracker,
    /// Mapping from UE ID to socket address
    ue_addresses: HashMap<i32, SocketAddr>,
    /// Mapping from STI to UE ID
    sti_to_ue_id: HashMap<u64, i32>,
    /// Pending acknowledgments per UE (UE ID -> list of PDU IDs)
    pending_acks: HashMap<i32, Vec<u32>>,
    /// gNB STI (Simulated Transmission Identifier)
    sti: u64,
    /// UDP socket for RLS communication
    socket: Option<Arc<UdpSocket>>,
    /// Local bind address
    bind_address: SocketAddr,
}

impl RlsTask {
    /// Creates a new RLS task
    pub fn new(task_base: GnbTaskBase) -> Self {
        // Generate STI from gNB NCI
        let sti = task_base.config.nci;
        let phy_location = Vector3::new(0, 0, 0);
        
        // Get bind address from config
        let bind_address = SocketAddr::new(task_base.config.link_ip, DEFAULT_RLS_PORT);

        Self {
            task_base,
            cell_tracker: GnbCellTracker::new(sti, phy_location),
            ue_addresses: HashMap::new(),
            sti_to_ue_id: HashMap::new(),
            pending_acks: HashMap::new(),
            sti,
            socket: None,
            bind_address,
        }
    }

    /// Creates a new RLS task with custom bind address
    pub fn with_bind_address(task_base: GnbTaskBase, bind_address: SocketAddr) -> Self {
        let sti = task_base.config.nci;
        let phy_location = Vector3::new(0, 0, 0);

        Self {
            task_base,
            cell_tracker: GnbCellTracker::new(sti, phy_location),
            ue_addresses: HashMap::new(),
            sti_to_ue_id: HashMap::new(),
            pending_acks: HashMap::new(),
            sti,
            socket: None,
            bind_address,
        }
    }

    /// Initializes the UDP socket
    async fn init_socket(&mut self) -> Result<(), std::io::Error> {
        let socket = UdpSocket::bind(self.bind_address).await?;
        info!("RLS task bound to {}", self.bind_address);
        self.socket = Some(Arc::new(socket));
        Ok(())
    }

    /// Handles a received RLS message from the network
    async fn handle_receive_rls_message(&mut self, data: OctetString, source: SocketAddr) {
        let bytes = Bytes::copy_from_slice(data.data());
        
        match codec::decode(&bytes) {
            Ok(msg) => {
                self.process_rls_message(msg, source).await;
            }
            Err(e) => {
                warn!("Failed to decode RLS message from {}: {}", source, e);
            }
        }
    }

    /// Processes a decoded RLS protocol message
    async fn process_rls_message(&mut self, msg: RlsProtocolMessage, source: SocketAddr) {
        match msg {
            RlsProtocolMessage::Heartbeat(heartbeat) => {
                self.handle_heartbeat(heartbeat.sti, source, &heartbeat).await;
            }
            RlsProtocolMessage::PduTransmission(pdu) => {
                self.handle_pdu_transmission(pdu.sti, source, &pdu).await;
            }
            RlsProtocolMessage::PduTransmissionAck(ack) => {
                self.handle_pdu_ack(&ack);
            }
            RlsProtocolMessage::HeartbeatAck(_) => {
                // gNB doesn't process heartbeat acks (it sends them)
                debug!("Ignoring heartbeat ack from {}", source);
            }
        }
    }

    /// Handles a heartbeat message from a UE
    async fn handle_heartbeat(
        &mut self,
        sti: u64,
        source: SocketAddr,
        heartbeat: &nextgsim_rls::RlsHeartbeat,
    ) {
        let (ack, events) = self.cell_tracker.process_heartbeat(sti, source, heartbeat);

        // Process tracker events
        for event in events {
            match event {
                GnbTrackerEvent::UeDetected { ue_id, sti } => {
                    info!("UE detected: ue_id={}, sti={}", ue_id, sti);
                    let ue_id_i32 = ue_id as i32;
                    self.ue_addresses.insert(ue_id_i32, source);
                    self.sti_to_ue_id.insert(sti, ue_id_i32);
                    
                    // Notify RRC task about signal detection
                    if let Err(e) = self.task_base.rrc_tx.send(RrcMessage::SignalDetected { 
                        ue_id: ue_id_i32 
                    }).await {
                        error!("Failed to send SignalDetected to RRC: {}", e);
                    }
                }
                GnbTrackerEvent::UeLost { ue_id } => {
                    info!("UE lost: ue_id={}", ue_id);
                    let ue_id_i32 = ue_id as i32;
                    self.ue_addresses.remove(&ue_id_i32);
                    // Note: sti_to_ue_id cleanup happens in check_lost_ues
                }
            }
        }

        // Send heartbeat acknowledgment
        if let Some(ack) = ack {
            self.send_heartbeat_ack(source, &ack).await;
        }
    }

    /// Sends a heartbeat acknowledgment to a UE
    async fn send_heartbeat_ack(&self, dest: SocketAddr, ack: &RlsHeartbeatAck) {
        let msg = RlsProtocolMessage::HeartbeatAck(ack.clone());
        self.send_rls_message(dest, &msg).await;
    }

    /// Handles a PDU transmission from a UE
    async fn handle_pdu_transmission(
        &mut self,
        sti: u64,
        source: SocketAddr,
        pdu: &RlsPduTransmission,
    ) {
        // Get UE ID from STI
        let ue_id = match self.sti_to_ue_id.get(&sti) {
            Some(&id) => id,
            None => {
                warn!("PDU from unknown STI {}", sti);
                return;
            }
        };

        // Update UE address if changed
        self.ue_addresses.insert(ue_id, source);

        // Queue acknowledgment if PDU ID is non-zero
        if pdu.pdu_id != 0 {
            self.pending_acks
                .entry(ue_id)
                .or_default()
                .push(pdu.pdu_id);
        }

        match pdu.pdu_type {
            PduType::Rrc => {
                self.handle_uplink_rrc(ue_id, pdu).await;
            }
            PduType::Data => {
                self.handle_uplink_data(ue_id, pdu).await;
            }
            PduType::Reserved => {
                warn!("Received reserved PDU type from UE[{}]", ue_id);
            }
        }
    }

    /// Handles uplink RRC message from UE
    async fn handle_uplink_rrc(&self, ue_id: i32, pdu: &RlsPduTransmission) {
        let channel = match RrcChannel::from_u32(pdu.payload) {
            Some(ch) => ch,
            None => {
                warn!("Invalid RRC channel {} from UE[{}]", pdu.payload, ue_id);
                return;
            }
        };

        debug!(
            "Uplink RRC: ue_id={}, channel={:?}, len={}",
            ue_id,
            channel,
            pdu.pdu.len()
        );

        // Forward to RRC task
        let data = OctetString::from_slice(&pdu.pdu);
        let msg = RrcMessage::UplinkRrc {
            ue_id,
            rrc_channel: channel,
            data,
        };

        if let Err(e) = self.task_base.rrc_tx.send(msg).await {
            error!("Failed to send uplink RRC to RRC task: {}", e);
        }
    }

    /// Handles uplink user plane data from UE
    async fn handle_uplink_data(&self, ue_id: i32, pdu: &RlsPduTransmission) {
        let psi = pdu.payload as i32;
        
        debug!(
            "Uplink data: ue_id={}, psi={}, len={}",
            ue_id,
            psi,
            pdu.pdu.len()
        );

        // Forward to GTP task
        let data = OctetString::from_slice(&pdu.pdu);
        let msg = GtpMessage::DataPduDelivery {
            ue_id,
            psi,
            pdu: data,
        };

        if let Err(e) = self.task_base.gtp_tx.send(msg).await {
            error!("Failed to send uplink data to GTP task: {}", e);
        }
    }

    /// Handles PDU transmission acknowledgment
    fn handle_pdu_ack(&mut self, ack: &RlsPduTransmissionAck) {
        debug!("PDU ack received: {} PDUs acknowledged", ack.pdu_ids.len());
        // PDU acknowledgment tracking can be extended here if needed
    }

    /// Handles downlink RRC message from RRC task
    async fn handle_downlink_rrc(
        &mut self,
        ue_id: i32,
        rrc_channel: RrcChannel,
        pdu_id: u32,
        data: OctetString,
    ) {
        let dest = match self.ue_addresses.get(&ue_id) {
            Some(&addr) => addr,
            None => {
                warn!("Downlink RRC for unknown UE[{}]", ue_id);
                return;
            }
        };

        debug!(
            "Downlink RRC: ue_id={}, channel={:?}, pdu_id={}, len={}",
            ue_id,
            rrc_channel,
            pdu_id,
            data.len()
        );

        let pdu = RlsPduTransmission {
            sti: self.sti,
            pdu_type: PduType::Rrc,
            pdu_id,
            payload: rrc_channel as u32,
            pdu: Bytes::copy_from_slice(data.data()),
        };

        let msg = RlsProtocolMessage::PduTransmission(pdu);
        self.send_rls_message(dest, &msg).await;
    }

    /// Handles downlink user plane data from GTP task
    async fn handle_downlink_data(&mut self, ue_id: i32, psi: i32, data: OctetString) {
        let dest = match self.ue_addresses.get(&ue_id) {
            Some(&addr) => addr,
            None => {
                warn!("Downlink data for unknown UE[{}]", ue_id);
                return;
            }
        };

        debug!(
            "Downlink data: ue_id={}, psi={}, len={}",
            ue_id,
            psi,
            data.len()
        );

        let pdu = RlsPduTransmission {
            sti: self.sti,
            pdu_type: PduType::Data,
            pdu_id: 0, // Data PDUs don't require acknowledgment
            payload: psi as u32,
            pdu: Bytes::copy_from_slice(data.data()),
        };

        let msg = RlsProtocolMessage::PduTransmission(pdu);
        self.send_rls_message(dest, &msg).await;
    }

    /// Sends an RLS message to a destination
    async fn send_rls_message(&self, dest: SocketAddr, msg: &RlsProtocolMessage) {
        let socket = match &self.socket {
            Some(s) => s,
            None => {
                error!("Cannot send RLS message: socket not initialized");
                return;
            }
        };

        let encoded = codec::encode(msg);
        if let Err(e) = socket.send_to(&encoded, dest).await {
            error!("Failed to send RLS message to {}: {}", dest, e);
        }
    }

    /// Sends pending acknowledgments to UEs
    async fn send_pending_acks(&mut self) {
        let pending = std::mem::take(&mut self.pending_acks);

        for (ue_id, pdu_ids) in pending {
            if pdu_ids.is_empty() {
                continue;
            }

            let dest = match self.ue_addresses.get(&ue_id) {
                Some(&addr) => addr,
                None => continue,
            };

            let ack = RlsPduTransmissionAck::with_pdu_ids(self.sti, pdu_ids);
            let msg = RlsProtocolMessage::PduTransmissionAck(ack);
            self.send_rls_message(dest, &msg).await;
        }
    }

    /// Checks for lost UEs and handles cleanup
    async fn check_lost_ues(&mut self) {
        let events = self.cell_tracker.check_lost_ues();

        for event in events {
            if let GnbTrackerEvent::UeLost { ue_id } = event {
                let ue_id_i32 = ue_id as i32;
                info!("UE[{}] lost due to heartbeat timeout", ue_id_i32);
                
                self.ue_addresses.remove(&ue_id_i32);
                self.pending_acks.remove(&ue_id_i32);
                
                // Find and remove STI mapping
                let sti_to_remove: Vec<u64> = self.sti_to_ue_id
                    .iter()
                    .filter(|(_, &id)| id == ue_id_i32)
                    .map(|(&sti, _)| sti)
                    .collect();
                
                for sti in sti_to_remove {
                    self.sti_to_ue_id.remove(&sti);
                }
            }
        }
    }

    /// Receives data from the UDP socket
    #[allow(dead_code)]
    async fn receive_udp(&self) -> Option<(OctetString, SocketAddr)> {
        let socket = self.socket.as_ref()?;
        let mut buf = vec![0u8; UDP_BUFFER_SIZE];
        
        match socket.recv_from(&mut buf).await {
            Ok((len, addr)) => {
                buf.truncate(len);
                Some((OctetString::from_slice(&buf), addr))
            }
            Err(e) => {
                error!("UDP receive error: {}", e);
                None
            }
        }
    }
}

#[async_trait::async_trait]
impl Task for RlsTask {
    type Message = RlsMessage;

    async fn run(&mut self, mut rx: mpsc::Receiver<TaskMessage<Self::Message>>) {
        info!("RLS task starting");

        // Initialize UDP socket
        if let Err(e) = self.init_socket().await {
            error!("Failed to initialize RLS socket: {}", e);
            return;
        }

        info!("RLS task started on {}", self.bind_address);

        let mut heartbeat_timer = interval(Duration::from_millis(HEARTBEAT_CHECK_INTERVAL_MS));

        loop {
            tokio::select! {
                // Handle messages from other tasks
                Some(msg) = rx.recv() => {
                    match msg {
                        TaskMessage::Message(rls_msg) => {
                            match rls_msg {
                                RlsMessage::ReceiveRlsMessage { data, source } => {
                                    self.handle_receive_rls_message(data, source).await;
                                }
                                RlsMessage::DownlinkRrc { ue_id, rrc_channel, pdu_id, data } => {
                                    self.handle_downlink_rrc(ue_id, rrc_channel, pdu_id, data).await;
                                }
                                RlsMessage::DownlinkData { ue_id, psi, pdu } => {
                                    self.handle_downlink_data(ue_id, psi, pdu).await;
                                }
                                RlsMessage::SignalDetected { ue_id } => {
                                    debug!("Signal detected notification for UE[{}]", ue_id);
                                }
                                RlsMessage::SignalLost { ue_id } => {
                                    debug!("Signal lost notification for UE[{}]", ue_id);
                                    self.ue_addresses.remove(&ue_id);
                                }
                                RlsMessage::UplinkRrc { ue_id, rrc_channel, data } => {
                                    // Internal uplink RRC (forwarded from receive handler)
                                    let msg = RrcMessage::UplinkRrc { ue_id, rrc_channel, data };
                                    if let Err(e) = self.task_base.rrc_tx.send(msg).await {
                                        error!("Failed to forward uplink RRC: {}", e);
                                    }
                                }
                                RlsMessage::UplinkData { ue_id, psi, pdu } => {
                                    // Internal uplink data (forwarded from receive handler)
                                    let msg = GtpMessage::DataPduDelivery { ue_id, psi, pdu };
                                    if let Err(e) = self.task_base.gtp_tx.send(msg).await {
                                        error!("Failed to forward uplink data: {}", e);
                                    }
                                }
                                RlsMessage::RadioLinkFailure { ue_id, cause } => {
                                    warn!("Radio link failure for UE[{}]: {:?}", ue_id, cause);
                                }
                                RlsMessage::TransmissionFailure { pdu_list } => {
                                    warn!("Transmission failure: {} PDUs failed", pdu_list.len());
                                }
                            }
                        }
                        TaskMessage::Shutdown => {
                            info!("RLS task received shutdown signal");
                            break;
                        }
                    }
                }

                // Handle incoming UDP packets
                result = async {
                    if let Some(socket) = &self.socket {
                        let mut buf = vec![0u8; UDP_BUFFER_SIZE];
                        socket.recv_from(&mut buf).await.ok().map(|(len, addr)| {
                            buf.truncate(len);
                            (OctetString::from_slice(&buf), addr)
                        })
                    } else {
                        None
                    }
                } => {
                    if let Some((data, source)) = result {
                        self.handle_receive_rls_message(data, source).await;
                    }
                }

                // Periodic heartbeat check and ack sending
                _ = heartbeat_timer.tick() => {
                    self.check_lost_ues().await;
                    self.send_pending_acks().await;
                }
            }
        }

        info!("RLS task stopped with {} tracked UEs", self.cell_tracker.ue_count());
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
            ntn_config: None,
            mbs_enabled: false,
            prose_enabled: false,
            lcs_enabled: false,
            snpn_config: None,
        }
    }

    #[test]
    fn test_rls_task_creation() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, 16);
        let task = RlsTask::new(task_base);
        
        assert_eq!(task.sti, 0x000000010);
        assert!(task.ue_addresses.is_empty());
        assert!(task.sti_to_ue_id.is_empty());
    }

    #[test]
    fn test_rls_task_with_custom_address() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, 16);
        
        let bind_addr: SocketAddr = "127.0.0.1:5000".parse().unwrap();
        let task = RlsTask::with_bind_address(task_base, bind_addr);
        
        assert_eq!(task.bind_address, bind_addr);
    }
}
