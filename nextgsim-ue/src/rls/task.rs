//! RLS Task Implementation for UE
//!
//! This module implements the RLS (Radio Link Simulation) task for the UE,
//! handling cell search, gNB connection, RRC message transport, and user plane data.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use tokio::net::UdpSocket;
use tokio::sync::mpsc;
use tokio::time::interval;
use tracing::{debug, error, info, warn};

use crate::tasks::{NasMessage, RlsMessage, RrcMessage, RlfCause, Task, TaskMessage, UeTaskBase};
use nextgsim_common::OctetString;
use nextgsim_rls::{
    codec, CellSearchEvent, RlsMessage as RlsProtocolMessage, RlsTransport, RrcChannel,
    TransportEvent, UeCellSearch,
};

/// Default RLS port for gNB
pub const DEFAULT_RLS_PORT: u16 = 4997;

const HEARTBEAT_INTERVAL_MS: u64 = 1000;
const LOST_CELL_CHECK_INTERVAL_MS: u64 = 500;
const UDP_BUFFER_SIZE: usize = 65535;

/// RLS task configuration
#[derive(Debug, Clone)]
pub struct RlsTaskConfig {
    pub gnb_search_list: Vec<SocketAddr>,
    pub bind_address: Option<SocketAddr>,
    pub heartbeat_interval: Duration,
    pub heartbeat_threshold: Duration,
}

impl Default for RlsTaskConfig {
    fn default() -> Self {
        Self {
            gnb_search_list: Vec::new(),
            bind_address: None,
            heartbeat_interval: Duration::from_millis(HEARTBEAT_INTERVAL_MS),
            heartbeat_threshold: Duration::from_millis(2000),
        }
    }
}

/// RLS Task for managing radio link simulation on UE side
pub struct RlsTask {
    task_base: UeTaskBase,
    cell_search: UeCellSearch,
    transport: RlsTransport,
    cell_addresses: HashMap<i32, SocketAddr>,
    serving_cell: Option<i32>,
    socket: Option<Arc<UdpSocket>>,
    config: RlsTaskConfig,
    sti: u64,
}

impl RlsTask {
    pub fn new(task_base: UeTaskBase, config: RlsTaskConfig) -> Self {
        let sti = rand::random::<u64>();
        let search_space: Vec<SocketAddr> = config.gnb_search_list.iter()
            .map(|addr| if addr.port() == 0 { SocketAddr::new(addr.ip(), DEFAULT_RLS_PORT) } else { *addr })
            .collect();

        let mut cell_search = UeCellSearch::new(sti, search_space);
        cell_search.set_heartbeat_interval(config.heartbeat_interval);
        cell_search.set_heartbeat_threshold(config.heartbeat_threshold);

        Self {
            task_base,
            cell_search,
            transport: RlsTransport::new(sti),
            cell_addresses: HashMap::new(),
            serving_cell: None,
            socket: None,
            config,
            sti,
        }
    }

    pub fn from_ue_config(task_base: UeTaskBase) -> Self {
        let gnb_search_list: Vec<SocketAddr> = task_base.config.gnb_search_list.iter()
            .filter_map(|s| s.parse::<std::net::IpAddr>().ok())
            .map(|ip| SocketAddr::new(ip, DEFAULT_RLS_PORT))
            .collect();
        let config = RlsTaskConfig { gnb_search_list, ..Default::default() };
        Self::new(task_base, config)
    }

    pub fn cell_count(&self) -> usize { self.cell_search.cell_count() }
    pub fn serving_cell(&self) -> Option<i32> { self.serving_cell }

    async fn init_socket(&mut self) -> Result<(), std::io::Error> {
        let bind_addr = self.config.bind_address.unwrap_or_else(|| "0.0.0.0:0".parse().unwrap());
        let socket = UdpSocket::bind(bind_addr).await?;
        info!("RLS task bound to {}", socket.local_addr()?);
        self.socket = Some(Arc::new(socket));
        Ok(())
    }

    async fn send_heartbeats(&mut self) {
        if !self.cell_search.should_send_heartbeats() { return; }
        for (addr, heartbeat) in self.cell_search.create_heartbeats() {
            self.send_rls_message(addr, &RlsProtocolMessage::Heartbeat(heartbeat)).await;
        }
    }

    async fn check_lost_cells(&mut self) {
        for event in self.cell_search.check_lost_cells() {
            if let CellSearchEvent::CellLost { cell_id } = event {
                info!("Cell lost: cell_id={}", cell_id);
                self.cell_addresses.remove(&(cell_id as i32));
                if self.serving_cell == Some(cell_id as i32) {
                    self.serving_cell = None;
                    self.transport.clear_serving_endpoint();
                    let _ = self.task_base.rrc_tx.send(RrcMessage::RadioLinkFailure {
                        cause: RlfCause::SignalLostToConnectedCell,
                    }).await;
                }
            }
        }
    }

    async fn send_pending_acks(&mut self) {
        for (endpoint_id, ack) in self.transport.create_pending_acks() {
            if let Some(&addr) = self.cell_addresses.get(&(endpoint_id as i32)) {
                self.send_rls_message(addr, &RlsProtocolMessage::PduTransmissionAck(ack)).await;
            }
        }
    }

    async fn check_expired_pdus(&mut self) {
        for event in self.transport.check_expired_pdus() {
            if let TransportEvent::TransmissionFailure { pdus } = event {
                warn!("Transmission failure: {} PDUs expired", pdus.len());
            }
        }
    }

    async fn send_rls_message(&self, dest: SocketAddr, msg: &RlsProtocolMessage) {
        if let Some(socket) = &self.socket {
            if let Err(e) = socket.send_to(&codec::encode(msg), dest).await {
                error!("Failed to send RLS message to {}: {}", dest, e);
            }
        }
    }

    async fn handle_receive_rls_message(&mut self, data: &[u8], source: SocketAddr) {
        match codec::decode(&Bytes::copy_from_slice(data)) {
            Ok(msg) => self.process_rls_message(msg, source).await,
            Err(e) => warn!("Failed to decode RLS message from {}: {}", source, e),
        }
    }

    async fn process_rls_message(&mut self, msg: RlsProtocolMessage, source: SocketAddr) {
        match msg {
            RlsProtocolMessage::HeartbeatAck(ack) => self.handle_heartbeat_ack(source, &ack).await,
            RlsProtocolMessage::PduTransmission(pdu) => self.handle_pdu_transmission(source, &pdu).await,
            RlsProtocolMessage::PduTransmissionAck(ack) => self.transport.process_pdu_ack(&ack),
            RlsProtocolMessage::Heartbeat(_) => debug!("Ignoring heartbeat from {}", source),
        }
    }


    async fn handle_heartbeat_ack(&mut self, source: SocketAddr, ack: &nextgsim_rls::RlsHeartbeatAck) {
        for event in self.cell_search.process_heartbeat_ack(ack.sti, source, ack) {
            match event {
                CellSearchEvent::CellDiscovered { cell_id, sti: _, dbm } => {
                    info!("Cell discovered: cell_id={}, dbm={}", cell_id, dbm);
                    self.cell_addresses.insert(cell_id as i32, source);
                    let _ = self.task_base.rrc_tx.send(RrcMessage::SignalChanged { cell_id: cell_id as i32, dbm }).await;
                }
                CellSearchEvent::SignalChanged { cell_id, old_dbm: _, new_dbm } => {
                    let _ = self.task_base.rrc_tx.send(RrcMessage::SignalChanged { cell_id: cell_id as i32, dbm: new_dbm }).await;
                }
                CellSearchEvent::CellLost { cell_id } => {
                    info!("Cell lost: cell_id={}", cell_id);
                    self.cell_addresses.remove(&(cell_id as i32));
                    if self.serving_cell == Some(cell_id as i32) {
                        self.serving_cell = None;
                        self.transport.clear_serving_endpoint();
                        let _ = self.task_base.rrc_tx.send(RrcMessage::RadioLinkFailure { cause: RlfCause::SignalLostToConnectedCell }).await;
                    }
                }
            }
        }
    }

    async fn handle_pdu_transmission(&mut self, source: SocketAddr, pdu: &nextgsim_rls::RlsPduTransmission) {
        let cell_id = match self.cell_addresses.iter().find(|(_, addr)| **addr == source).map(|(id, _)| *id) {
            Some(id) => id,
            None => { warn!("PDU from unknown source {}", source); return; }
        };

        for event in self.transport.process_pdu_transmission(cell_id as u32, pdu) {
            match event {
                TransportEvent::RrcReceived { channel, data, .. } => {
                    let pdu = OctetString::from_slice(&data);
                    let _ = self.task_base.rrc_tx.send(RrcMessage::DownlinkRrcDelivery { cell_id, channel, pdu }).await;
                }
                TransportEvent::DataReceived { psi, data } => {
                    let pdu = OctetString::from_slice(&data);
                    let _ = self.task_base.nas_tx.send(NasMessage::UplinkDataDelivery { psi: psi as i32, data: pdu }).await;
                }
                TransportEvent::TransmissionFailure { pdus } => warn!("Transmission failure: {} PDUs", pdus.len()),
                TransportEvent::RadioLinkFailure { cause } => {
                    let rlf_cause = match cause {
                        nextgsim_rls::RlfCause::PduIdExists => RlfCause::PduIdExists,
                        nextgsim_rls::RlfCause::PduIdFull => RlfCause::PduIdFull,
                        nextgsim_rls::RlfCause::SignalLostToConnectedCell => RlfCause::SignalLostToConnectedCell,
                    };
                    let _ = self.task_base.rrc_tx.send(RrcMessage::RadioLinkFailure { cause: rlf_cause }).await;
                }
            }
        }
    }

    fn handle_assign_current_cell(&mut self, cell_id: i32) {
        if self.cell_search.get_cell(cell_id as u32).is_some() {
            info!("Assigning serving cell: cell_id={}", cell_id);
            self.serving_cell = Some(cell_id);
            self.transport.set_serving_endpoint(cell_id as u32);
        } else {
            warn!("Cannot assign unknown cell as serving: cell_id={}", cell_id);
        }
    }

    fn handle_reset_sti(&mut self) {
        info!("Resetting STI");
        self.sti = rand::random::<u64>();
        self.serving_cell = None;
        self.cell_addresses.clear();
        self.transport = RlsTransport::new(self.sti);
        let search_space: Vec<SocketAddr> = self.config.gnb_search_list.clone();
        self.cell_search = UeCellSearch::new(self.sti, search_space);
        self.cell_search.set_heartbeat_interval(self.config.heartbeat_interval);
        self.cell_search.set_heartbeat_threshold(self.config.heartbeat_threshold);
    }

    async fn handle_rrc_pdu_delivery(&mut self, channel: RrcChannel, pdu_id: u32, pdu: OctetString) {
        let (cell_id, dest) = match (self.serving_cell, self.serving_cell.and_then(|id| self.cell_addresses.get(&id).copied())) {
            (Some(id), Some(addr)) => (id, addr),
            _ => { warn!("Cannot send uplink RRC: no serving cell"); return; }
        };

        debug!("Uplink RRC: cell_id={}, channel={:?}, pdu_id={}, len={}", cell_id, channel, pdu_id, pdu.len());
        let require_ack = pdu_id != 0;
        match self.transport.create_rrc_transmission(cell_id as u32, channel, Bytes::copy_from_slice(pdu.data()), require_ack) {
            Ok(transmission) => self.send_rls_message(dest, &RlsProtocolMessage::PduTransmission(transmission)).await,
            Err(cause) => {
                let rlf_cause = match cause {
                    nextgsim_rls::RlfCause::PduIdExists => RlfCause::PduIdExists,
                    nextgsim_rls::RlfCause::PduIdFull => RlfCause::PduIdFull,
                    nextgsim_rls::RlfCause::SignalLostToConnectedCell => RlfCause::SignalLostToConnectedCell,
                };
                let _ = self.task_base.rrc_tx.send(RrcMessage::RadioLinkFailure { cause: rlf_cause }).await;
            }
        }
    }

    async fn handle_data_pdu_delivery(&mut self, psi: i32, pdu: OctetString) {
        let dest = match self.serving_cell.and_then(|id| self.cell_addresses.get(&id).copied()) {
            Some(addr) => addr,
            None => { warn!("Cannot send uplink data: no serving cell"); return; }
        };
        debug!("Uplink data: psi={}, len={}", psi, pdu.len());
        let transmission = self.transport.create_data_transmission(psi as u32, Bytes::copy_from_slice(pdu.data()));
        self.send_rls_message(dest, &RlsProtocolMessage::PduTransmission(transmission)).await;
    }


    async fn handle_rls_message(&mut self, msg: RlsMessage) {
        match msg {
            RlsMessage::AssignCurrentCell { cell_id } => self.handle_assign_current_cell(cell_id),
            RlsMessage::RrcPduDelivery { channel, pdu_id, pdu } => self.handle_rrc_pdu_delivery(channel, pdu_id, pdu).await,
            RlsMessage::ResetSti => self.handle_reset_sti(),
            RlsMessage::DataPduDelivery { psi, pdu } => self.handle_data_pdu_delivery(psi, pdu).await,
            RlsMessage::ReceiveRlsMessage { data, .. } => debug!("Received internal RLS message, len={}", data.len()),
            RlsMessage::SignalChanged { cell_id, dbm } => debug!("Signal changed: cell_id={}, dbm={}", cell_id, dbm),
            RlsMessage::UplinkData { psi, data } => self.handle_data_pdu_delivery(psi, data).await,
            RlsMessage::UplinkRrc { channel, pdu_id, data, .. } => self.handle_rrc_pdu_delivery(channel, pdu_id, data).await,
            RlsMessage::DownlinkData { psi, data } => {
                let _ = self.task_base.nas_tx.send(NasMessage::UplinkDataDelivery { psi, data }).await;
            }
            RlsMessage::DownlinkRrc { cell_id, channel, data } => {
                let _ = self.task_base.rrc_tx.send(RrcMessage::DownlinkRrcDelivery { cell_id, channel, pdu: data }).await;
            }
            RlsMessage::RadioLinkFailure { cause } => warn!("Radio link failure: {:?}", cause),
            RlsMessage::TransmissionFailure { pdu_list } => warn!("Transmission failure: {} PDUs", pdu_list.len()),
        }
    }
}

#[async_trait::async_trait]
impl Task for RlsTask {
    type Message = RlsMessage;

    async fn run(&mut self, mut rx: mpsc::Receiver<TaskMessage<Self::Message>>) {
        info!("RLS task starting");
        if let Err(e) = self.init_socket().await {
            error!("Failed to initialize RLS socket: {}", e);
            return;
        }
        info!("RLS task started with {} gNBs in search list", self.config.gnb_search_list.len());

        let mut heartbeat_timer = interval(Duration::from_millis(HEARTBEAT_INTERVAL_MS));
        let mut lost_cell_timer = interval(Duration::from_millis(LOST_CELL_CHECK_INTERVAL_MS));

        loop {
            tokio::select! {
                Some(msg) = rx.recv() => {
                    match msg {
                        TaskMessage::Message(rls_msg) => self.handle_rls_message(rls_msg).await,
                        TaskMessage::Shutdown => { info!("RLS task received shutdown signal"); break; }
                    }
                }
                result = async {
                    if let Some(socket) = &self.socket {
                        let mut buf = vec![0u8; UDP_BUFFER_SIZE];
                        socket.recv_from(&mut buf).await.ok().map(|(len, addr)| { buf.truncate(len); (buf, addr) })
                    } else { None }
                } => {
                    if let Some((data, source)) = result {
                        self.handle_receive_rls_message(&data, source).await;
                    }
                }
                _ = heartbeat_timer.tick() => self.send_heartbeats().await,
                _ = lost_cell_timer.tick() => {
                    self.check_lost_cells().await;
                    self.send_pending_acks().await;
                    self.check_expired_pdus().await;
                }
            }
        }
        info!("RLS task stopped with {} discovered cells", self.cell_search.cell_count());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nextgsim_common::config::UeConfig;

    fn test_config() -> UeConfig {
        UeConfig {
            gnb_search_list: vec!["127.0.0.1".to_string()],
            ..Default::default()
        }
    }

    #[test]
    fn test_rls_task_config_default() {
        let config = RlsTaskConfig::default();
        assert!(config.gnb_search_list.is_empty());
        assert!(config.bind_address.is_none());
    }

    #[test]
    fn test_rls_task_creation() {
        let config = test_config();
        let (task_base, _app_rx, _nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(config, 16);
        let rls_config = RlsTaskConfig {
            gnb_search_list: vec!["127.0.0.1:4997".parse().unwrap()],
            ..Default::default()
        };
        let task = RlsTask::new(task_base, rls_config);
        assert!(task.serving_cell.is_none());
        assert_eq!(task.cell_count(), 0);
    }

    #[test]
    fn test_rls_task_from_ue_config() {
        let config = test_config();
        let (task_base, _app_rx, _nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(config, 16);
        let task = RlsTask::from_ue_config(task_base);
        assert!(task.serving_cell.is_none());
        assert_eq!(task.config.gnb_search_list.len(), 1);
    }

    #[test]
    fn test_assign_current_cell() {
        let config = test_config();
        let (task_base, _app_rx, _nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(config, 16);
        let rls_config = RlsTaskConfig {
            gnb_search_list: vec!["127.0.0.1:4997".parse().unwrap()],
            ..Default::default()
        };
        let mut task = RlsTask::new(task_base, rls_config);
        task.handle_assign_current_cell(1);
        assert!(task.serving_cell.is_none()); // Unknown cell
    }

    #[test]
    fn test_reset_sti() {
        let config = test_config();
        let (task_base, _app_rx, _nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(config, 16);
        let rls_config = RlsTaskConfig {
            gnb_search_list: vec!["127.0.0.1:4997".parse().unwrap()],
            ..Default::default()
        };
        let mut task = RlsTask::new(task_base, rls_config);
        let old_sti = task.sti;
        task.handle_reset_sti();
        assert_ne!(task.sti, old_sti);
        assert!(task.serving_cell.is_none());
        assert_eq!(task.cell_count(), 0);
    }
}
