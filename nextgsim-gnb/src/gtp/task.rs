//! GTP Task Implementation
//!
//! Implements the GTP-U task for user plane data forwarding.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use bytes::Bytes;
use tokio::net::UdpSocket;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use nextgsim_gtp::codec::{GtpHeader, GtpMessageType};
use nextgsim_gtp::tunnel::{GtpTunnel, PduSession, TunnelManager, GTP_U_PORT};

use crate::tasks::{
    GnbTaskBase, GtpMessage, GtpUeContextUpdate, PduSessionResource, RlsMessage, Task,
    TaskMessage,
};

/// GTP-U UE context
#[derive(Debug)]
struct GtpUeContext {
    /// UE ID
    ue_id: i32,
    /// AMF UE NGAP ID (if assigned)
    amf_ue_ngap_id: Option<i64>,
}

impl GtpUeContext {
    fn new(ue_id: i32) -> Self {
        Self {
            ue_id,
            amf_ue_ngap_id: None,
        }
    }
}

/// GTP Task
///
/// Handles GTP-U tunnel management and user plane data forwarding.
pub struct GtpTask {
    /// Task base with handles to other tasks
    task_base: GnbTaskBase,
    /// UDP socket for GTP-U
    udp_socket: Option<Arc<UdpSocket>>,
    /// UE contexts indexed by UE ID
    ue_contexts: HashMap<i32, GtpUeContext>,
    /// Tunnel manager for PDU sessions
    tunnel_manager: TunnelManager,
    /// Receive buffer size
    recv_buffer_size: usize,
}


impl GtpTask {
    /// Create a new GTP task
    pub fn new(task_base: GnbTaskBase) -> Self {
        Self {
            task_base,
            udp_socket: None,
            ue_contexts: HashMap::new(),
            tunnel_manager: TunnelManager::new(),
            recv_buffer_size: 65535,
        }
    }

    /// Initialize the UDP socket for GTP-U
    async fn init_udp_socket(&mut self) -> Result<(), std::io::Error> {
        let gtp_ip = self.task_base.config.gtp_ip;
        let bind_addr = SocketAddr::new(gtp_ip, GTP_U_PORT);

        let socket = UdpSocket::bind(bind_addr).await?;
        info!("GTP-U socket bound to {}", bind_addr);

        self.udp_socket = Some(Arc::new(socket));
        Ok(())
    }

    /// Handle UE context update from NGAP
    fn handle_ue_context_update(&mut self, ue_id: i32, update: GtpUeContextUpdate) {
        let context = self
            .ue_contexts
            .entry(ue_id)
            .or_insert_with(|| GtpUeContext::new(ue_id));

        if let Some(amf_id) = update.amf_ue_ngap_id {
            context.amf_ue_ngap_id = Some(amf_id);
        }

        debug!("UE context updated: ue_id={}", ue_id);
    }

    /// Handle UE context release from NGAP
    fn handle_ue_context_release(&mut self, ue_id: i32) {
        // Delete all PDU sessions for this UE
        let deleted = self.tunnel_manager.delete_sessions_for_ue(ue_id as u32);
        debug!(
            "Deleted {} PDU sessions for UE {}",
            deleted.len(),
            ue_id
        );

        // Remove UE context
        self.ue_contexts.remove(&ue_id);
        debug!("UE context released: ue_id={}", ue_id);
    }

    /// Handle PDU session create from NGAP
    fn handle_session_create(&mut self, ue_id: i32, resource: PduSessionResource) {
        if !self.ue_contexts.contains_key(&ue_id) {
            error!(
                "PDU session create failed: UE context not found for ue_id={}",
                ue_id
            );
            return;
        }

        let gtp_ip = self.task_base.config.gtp_ip;
        let upf_addr = SocketAddr::new(resource.upf_address, GTP_U_PORT);
        let gnb_addr = SocketAddr::new(gtp_ip, GTP_U_PORT);

        let session = PduSession::new(
            ue_id as u32,
            resource.psi as u8,
            GtpTunnel::new(resource.uplink_teid, upf_addr),
            GtpTunnel::new(resource.downlink_teid, gnb_addr),
        );

        let session = if let Some(qfi) = resource.qfi {
            session.with_qfi(qfi)
        } else {
            session
        };

        match self.tunnel_manager.create_session(session) {
            Ok(()) => {
                info!(
                    "PDU session created: ue_id={}, psi={}, ul_teid={:#x}, dl_teid={:#x}",
                    ue_id, resource.psi, resource.uplink_teid, resource.downlink_teid
                );
            }
            Err(e) => {
                error!("PDU session create failed: {}", e);
            }
        }
    }

    /// Handle PDU session release from NGAP
    fn handle_session_release(&mut self, ue_id: i32, psi: i32) {
        match self.tunnel_manager.delete_session(ue_id as u32, psi as u8) {
            Ok(_) => {
                info!("PDU session released: ue_id={}, psi={}", ue_id, psi);
            }
            Err(e) => {
                error!("PDU session release failed: {}", e);
            }
        }
    }


    /// Handle uplink data PDU from RLS (UE -> UPF)
    async fn handle_uplink_data(&self, ue_id: i32, psi: i32, pdu: Vec<u8>) {
        let socket = match &self.udp_socket {
            Some(s) => s,
            None => {
                error!("Uplink data failed: UDP socket not initialized");
                return;
            }
        };

        // Check if it's an IPv4 packet (version field in first nibble)
        if pdu.is_empty() || (pdu[0] >> 4) != 4 {
            debug!("Ignoring non-IPv4 packet");
            return;
        }

        // Encapsulate in GTP-U
        let payload = Bytes::from(pdu);
        match self
            .tunnel_manager
            .encapsulate_uplink(ue_id as u32, psi as u8, payload)
        {
            Ok((header, dest_addr)) => {
                let encoded = header.encode();
                if let Err(e) = socket.send_to(&encoded, dest_addr).await {
                    error!("Failed to send GTP-U uplink: {}", e);
                } else {
                    debug!(
                        "Sent uplink GTP-U: ue_id={}, psi={}, teid={:#x}, {} bytes",
                        ue_id,
                        psi,
                        header.teid,
                        encoded.len()
                    );
                }
            }
            Err(e) => {
                error!("Uplink encapsulation failed: {}", e);
            }
        }
    }

    /// Handle received GTP-U packet from network (UPF -> UE)
    async fn handle_udp_receive(&self, data: &[u8], _source: SocketAddr) {
        // Decode GTP-U header
        let header = match GtpHeader::decode(data) {
            Ok(h) => h,
            Err(e) => {
                error!("Failed to decode GTP-U: {}", e);
                return;
            }
        };

        match header.message_type {
            GtpMessageType::GPdu => {
                self.handle_downlink_gpdu(&header).await;
            }
            GtpMessageType::EchoRequest => {
                self.handle_echo_request(&header, _source).await;
            }
            other => {
                warn!("Unhandled GTP-U message type: {:?}", other);
            }
        }
    }

    /// Handle downlink G-PDU (user data from UPF)
    async fn handle_downlink_gpdu(&self, header: &GtpHeader) {
        match self.tunnel_manager.decapsulate_downlink(header) {
            Ok((ue_id, psi, payload)) => {
                // Forward to RLS task for delivery to UE
                let msg = RlsMessage::DownlinkData {
                    ue_id: ue_id as i32,
                    psi: psi as i32,
                    pdu: payload.to_vec().into(),
                };

                if let Err(e) = self.task_base.rls_tx.send(msg).await {
                    error!("Failed to send downlink data to RLS: {}", e);
                } else {
                    debug!(
                        "Forwarded downlink data: ue_id={}, psi={}, {} bytes",
                        ue_id,
                        psi,
                        payload.len()
                    );
                }
            }
            Err(e) => {
                error!("Downlink decapsulation failed: {}", e);
            }
        }
    }

    /// Handle GTP-U Echo Request
    async fn handle_echo_request(&self, request: &GtpHeader, source: SocketAddr) {
        let socket = match &self.udp_socket {
            Some(s) => s,
            None => return,
        };

        // Build Echo Response
        let mut response = GtpHeader::echo_response(request.teid);
        if let Some(seq) = request.sequence_number {
            response = response.with_sequence_number(seq);
        }
        // Add recovery IE (type 14, value 0)
        response.payload = Bytes::from_static(&[14, 0]);

        let encoded = response.encode();
        if let Err(e) = socket.send_to(&encoded, source).await {
            error!("Failed to send Echo Response: {}", e);
        } else {
            debug!("Sent Echo Response to {}", source);
        }
    }
}


#[async_trait::async_trait]
impl Task for GtpTask {
    type Message = GtpMessage;

    async fn run(&mut self, mut rx: mpsc::Receiver<TaskMessage<GtpMessage>>) {
        // Initialize UDP socket
        if let Err(e) = self.init_udp_socket().await {
            error!("Failed to initialize GTP-U socket: {}", e);
            return;
        }

        let socket = self.udp_socket.clone().unwrap();
        let mut recv_buf = vec![0u8; self.recv_buffer_size];

        info!("GTP task started");

        loop {
            tokio::select! {
                // Handle incoming messages from other tasks
                msg = rx.recv() => {
                    match msg {
                        Some(TaskMessage::Message(gtp_msg)) => {
                            match gtp_msg {
                                GtpMessage::UeContextUpdate { ue_id, update } => {
                                    self.handle_ue_context_update(ue_id, update);
                                }
                                GtpMessage::UeContextRelease { ue_id } => {
                                    self.handle_ue_context_release(ue_id);
                                }
                                GtpMessage::SessionCreate { ue_id, resource } => {
                                    self.handle_session_create(ue_id, resource);
                                }
                                GtpMessage::SessionRelease { ue_id, psi } => {
                                    self.handle_session_release(ue_id, psi);
                                }
                                GtpMessage::DataPduDelivery { ue_id, psi, pdu } => {
                                    self.handle_uplink_data(ue_id, psi, pdu.into_vec()).await;
                                }
                            }
                        }
                        Some(TaskMessage::Shutdown) | None => {
                            info!("GTP task shutting down");
                            break;
                        }
                    }
                }

                // Handle incoming UDP packets
                result = socket.recv_from(&mut recv_buf) => {
                    match result {
                        Ok((len, source)) => {
                            self.handle_udp_receive(&recv_buf[..len], source).await;
                        }
                        Err(e) => {
                            error!("UDP receive error: {}", e);
                        }
                    }
                }
            }
        }

        info!("GTP task stopped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nextgsim_common::config::GnbConfig;
    use nextgsim_common::Plmn;
    use std::net::IpAddr;

    fn test_config() -> GnbConfig {
        GnbConfig {
            nci: 0x000000010,
            gnb_id_length: 32,
            plmn: Plmn::new(1, 1, false),
            tac: 1,
            nssai: vec![],
            amf_configs: vec![],
            link_ip: "127.0.0.1".parse().unwrap(),
            ngap_ip: "127.0.0.1".parse().unwrap(),
            gtp_ip: "127.0.0.1".parse().unwrap(),
            gtp_advertise_ip: None,
            ignore_stream_ids: false,
        }
    }

    #[test]
    fn test_ue_context_create() {
        let context = GtpUeContext::new(1);
        assert_eq!(context.ue_id, 1);
        assert!(context.amf_ue_ngap_id.is_none());
    }

    #[test]
    fn test_gtp_task_new() {
        let (task_base, _, _, _, _, _, _) =
            GnbTaskBase::new(test_config(), 16);
        let task = GtpTask::new(task_base);

        assert!(task.udp_socket.is_none());
        assert!(task.ue_contexts.is_empty());
        assert_eq!(task.tunnel_manager.session_count(), 0);
    }

    #[test]
    fn test_handle_ue_context_update() {
        let (task_base, _, _, _, _, _, _) =
            GnbTaskBase::new(test_config(), 16);
        let mut task = GtpTask::new(task_base);

        let update = GtpUeContextUpdate {
            ue_id: 1,
            amf_ue_ngap_id: Some(12345),
        };

        task.handle_ue_context_update(1, update);

        assert!(task.ue_contexts.contains_key(&1));
        assert_eq!(task.ue_contexts[&1].amf_ue_ngap_id, Some(12345));
    }

    #[test]
    fn test_handle_session_create_without_context() {
        let (task_base, _, _, _, _, _, _) =
            GnbTaskBase::new(test_config(), 16);
        let mut task = GtpTask::new(task_base);

        let resource = PduSessionResource {
            psi: 1,
            qfi: Some(1),
            uplink_teid: 0x1000,
            downlink_teid: 0x2000,
            upf_address: IpAddr::from([10, 0, 0, 1]),
        };

        // Should fail because UE context doesn't exist
        task.handle_session_create(1, resource);
        assert_eq!(task.tunnel_manager.session_count(), 0);
    }

    #[test]
    fn test_handle_session_create_with_context() {
        let (task_base, _, _, _, _, _, _) =
            GnbTaskBase::new(test_config(), 16);
        let mut task = GtpTask::new(task_base);

        // First create UE context
        let update = GtpUeContextUpdate {
            ue_id: 1,
            amf_ue_ngap_id: None,
        };
        task.handle_ue_context_update(1, update);

        // Now create session
        let resource = PduSessionResource {
            psi: 1,
            qfi: Some(1),
            uplink_teid: 0x1000,
            downlink_teid: 0x2000,
            upf_address: IpAddr::from([10, 0, 0, 1]),
        };

        task.handle_session_create(1, resource);
        assert_eq!(task.tunnel_manager.session_count(), 1);
        assert!(task.tunnel_manager.has_session(1, 1));
    }

    #[test]
    fn test_handle_session_release() {
        let (task_base, _, _, _, _, _, _) =
            GnbTaskBase::new(test_config(), 16);
        let mut task = GtpTask::new(task_base);

        // Create UE context and session
        let update = GtpUeContextUpdate {
            ue_id: 1,
            amf_ue_ngap_id: None,
        };
        task.handle_ue_context_update(1, update);

        let resource = PduSessionResource {
            psi: 1,
            qfi: Some(1),
            uplink_teid: 0x1000,
            downlink_teid: 0x2000,
            upf_address: IpAddr::from([10, 0, 0, 1]),
        };
        task.handle_session_create(1, resource);

        // Release session
        task.handle_session_release(1, 1);
        assert_eq!(task.tunnel_manager.session_count(), 0);
    }

    #[test]
    fn test_handle_ue_context_release() {
        let (task_base, _, _, _, _, _, _) =
            GnbTaskBase::new(test_config(), 16);
        let mut task = GtpTask::new(task_base);

        // Create UE context and multiple sessions
        let update = GtpUeContextUpdate {
            ue_id: 1,
            amf_ue_ngap_id: None,
        };
        task.handle_ue_context_update(1, update);

        for psi in 1..=3 {
            let resource = PduSessionResource {
                psi,
                qfi: Some(1),
                uplink_teid: 0x1000 + psi as u32,
                downlink_teid: 0x2000 + psi as u32,
                upf_address: IpAddr::from([10, 0, 0, 1]),
            };
            task.handle_session_create(1, resource);
        }

        assert_eq!(task.tunnel_manager.session_count(), 3);

        // Release UE context (should delete all sessions)
        task.handle_ue_context_release(1);

        assert!(!task.ue_contexts.contains_key(&1));
        assert_eq!(task.tunnel_manager.session_count(), 0);
    }
}
