//! The operational eNB: one task serving the cell's RLS radio endpoint and
//! the S1-MME association together, executing what [`EnbRrc`] decides.
//!
//! RLS stands in for the physical layer and MAC exactly as on the NR side:
//! UE heartbeats are cell search and measurement, an RLS PDU transmission is
//! a transport block on a logical channel. Everything carried is real
//! protocol -- TS 36.331 RRC on CCCH, TS 36.323 PDCP PDUs on DCCH.
//!
//! One task rather than two linked by channels because every radio event that
//! matters ends in S1 traffic and vice versa, and both receive paths are
//! cancellation-safe (kernel SCTP waits for readiness and then reads
//! synchronously; UDP `recv_from` likewise), so a `select!` over them loses
//! nothing.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::Duration;

use bytes::Bytes;
use nextgsim_rls::{
    codec as rls_codec, GnbCellTracker, GnbTrackerEvent, PduType, RlsMessage, RlsPduTransmission,
    RlsPduTransmissionAck, RrcChannel, SimCoord,
};
use nextgsim_sctp::S1AP_PPID;
use tokio::net::UdpSocket;

use crate::config::EnbConfig;
use crate::rrc::{Action, EnbRrc};
use crate::s1_task::S1Transport;

/// How often lost UEs are looked for (heartbeat supervision).
const LOST_UE_CHECK_MS: u64 = 500;
/// One LTE radio frame, for advancing the SFN between SI broadcasts.
const RADIO_FRAME_MS: u64 = 10;

/// Why [`run_operational`] returned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stopped {
    /// The S1-MME association closed or failed; reconnect and set up again.
    S1Lost,
    /// The RLS socket failed.
    RadioFailed(String),
}

/// The eNB's RLS endpoint: the socket and the UEs heard on it.
pub struct RlsRadio {
    socket: UdpSocket,
    sti: u64,
    tracker: GnbCellTracker,
    ue_addr: HashMap<u32, SocketAddr>,
    next_pdu_id: u32,
}

impl RlsRadio {
    /// Bind the cell's RLS socket. The STI (RLS's sender identity) is the
    /// E-UTRAN Cell Identity, which is unique per cell.
    pub async fn bind(cfg: &EnbConfig) -> std::io::Result<Self> {
        let addr = SocketAddr::new(cfg.radio.link_ip, cfg.radio.rls_port);
        let socket = UdpSocket::bind(addr).await?;
        log::info!("RLS radio endpoint bound on {}", socket.local_addr()?);
        let sti = u64::from(cfg.cell_identity());
        Ok(Self {
            socket,
            sti,
            tracker: GnbCellTracker::new(sti, SimCoord::new(0, 0, 0)),
            ue_addr: HashMap::new(),
            next_pdu_id: 1,
        })
    }

    /// The bound address (useful when the port was 0).
    pub fn local_addr(&self) -> std::io::Result<SocketAddr> {
        self.socket.local_addr()
    }

    async fn send(&self, to: SocketAddr, msg: &RlsMessage) {
        if let Err(e) = self.socket.send_to(&rls_codec::encode(msg), to).await {
            log::warn!("RLS send to {to} failed: {e}");
        }
    }

    async fn send_rrc(&mut self, ue: u32, channel: RrcChannel, pdu: Vec<u8>) {
        let Some(&to) = self.ue_addr.get(&ue) else {
            log::warn!("RRC PDU for UE {ue} dropped: UE not heard on RLS");
            return;
        };
        let id = self.next_pdu_id;
        self.next_pdu_id = self.next_pdu_id.wrapping_add(1).max(1);
        let msg = RlsMessage::PduTransmission(RlsPduTransmission::rrc(
            self.sti,
            id,
            channel as u32,
            Bytes::from(pdu),
        ));
        self.send(to, &msg).await;
    }
}

async fn execute<T: S1Transport>(
    actions: Vec<Action>,
    radio: &mut RlsRadio,
    transport: &mut T,
) -> Result<(), Stopped> {
    for a in actions {
        match a {
            Action::Rrc { ue, channel, pdu } => radio.send_rrc(ue, channel, pdu).await,
            Action::S1 { stream, pdu } => {
                if let Err(e) = transport.send(stream, &pdu, S1AP_PPID).await {
                    log::warn!("S1-MME send failed: {e}");
                    return Err(Stopped::S1Lost);
                }
            }
        }
    }
    Ok(())
}

async fn broadcast_si(rrc: &mut EnbRrc, radio: &mut RlsRadio, ues: &[u32], sfn: u16) {
    let si = rrc.system_information(sfn);
    for &ue in ues {
        for (channel, pdu) in &si {
            radio.send_rrc(ue, *channel, pdu.clone()).await;
        }
    }
}

/// Serve the operational cell until the S1-MME association is lost. The RRC
/// state lives for exactly one S1 session: UE contexts do not survive the
/// loss of the S1 interface (TS 36.413 §8.7.1.2: a Reset or a new S1 Setup
/// clears UE-associated state on both sides).
pub async fn run_operational<T: S1Transport>(
    cfg: &EnbConfig,
    radio: &mut RlsRadio,
    transport: &mut T,
) -> Stopped {
    let mut rrc = EnbRrc::new(cfg, transport.outbound_streams());
    let mut si_timer =
        tokio::time::interval(Duration::from_millis(cfg.radio.si_broadcast_period_ms));
    let mut lost_timer = tokio::time::interval(Duration::from_millis(LOST_UE_CHECK_MS));
    let frames_per_si = (cfg.radio.si_broadcast_period_ms / RADIO_FRAME_MS).max(1) as u16;
    let mut sfn: u16 = 0;
    let mut buf = vec![0u8; 65535];
    log::info!("cell {:#09x} operational", cfg.cell_identity());
    loop {
        tokio::select! {
            r = transport.recv() => {
                let m = match r {
                    Ok(Some(m)) => m,
                    Ok(None) => { log::warn!("S1-MME association closed"); return Stopped::S1Lost; }
                    Err(e) => { log::warn!("S1-MME receive failed: {e}"); return Stopped::S1Lost; }
                };
                if m.ppid != S1AP_PPID {
                    log::warn!("discarding S1-MME message with PPID {} (S1AP is {S1AP_PPID})", m.ppid);
                    continue;
                }
                let acts = rrc.on_s1(&m.data);
                if let Err(s) = execute(acts, radio, transport).await { return s; }
            }
            r = radio.socket.recv_from(&mut buf) => {
                let (n, from) = match r {
                    Ok(v) => v,
                    Err(e) => return Stopped::RadioFailed(e.to_string()),
                };
                let msg = match rls_codec::decode(&buf[..n]) {
                    Ok(m) => m,
                    Err(e) => { log::debug!("undecodable RLS datagram from {from}: {e}"); continue; }
                };
                match msg {
                    RlsMessage::Heartbeat(hb) => {
                        let (ack, events) = radio.tracker.process_heartbeat(hb.sti, from, &hb);
                        if let Some(ack) = ack {
                            radio.send(from, &RlsMessage::HeartbeatAck(ack)).await;
                        }
                        for ev in events {
                            if let GnbTrackerEvent::UeDetected { ue_id, sti } = ev {
                                log::info!("UE {ue_id} detected on RLS (STI {sti:#x}, {from})");
                                radio.ue_addr.insert(ue_id, from);
                                // A newly heard UE gets the system information
                                // at once rather than at the next period.
                                broadcast_si(&mut rrc, radio, &[ue_id], sfn).await;
                            }
                        }
                        if let Some(ue) = radio.tracker.get_ue_by_sti(hb.sti) {
                            radio.ue_addr.insert(ue.ue_id, from);
                        }
                    }
                    RlsMessage::PduTransmission(pdu) => {
                        let Some(ue) = radio.tracker.get_ue_by_sti(pdu.sti).map(|u| u.ue_id) else {
                            log::debug!("RLS PDU from unknown STI {:#x}", pdu.sti);
                            continue;
                        };
                        // RLS acknowledges each transmission, as the NR side does.
                        radio.send(from, &RlsMessage::PduTransmissionAck(RlsPduTransmissionAck {
                            sti: radio.sti,
                            pdu_ids: vec![pdu.pdu_id],
                        })).await;
                        if pdu.pdu_type != PduType::Rrc {
                            log::debug!("UE {ue}: user-plane RLS PDU dropped (no DRBs in this build)");
                            continue;
                        }
                        let acts = match RrcChannel::from_u32(pdu.payload) {
                            Some(RrcChannel::UlCcch) => rrc.on_ul_ccch(ue, &pdu.pdu),
                            Some(RrcChannel::UlDcch) => rrc.on_ul_dcch(ue, &pdu.pdu),
                            other => {
                                log::warn!("UE {ue}: RRC PDU on {other:?} is not an E-UTRA uplink channel");
                                continue;
                            }
                        };
                        if let Err(s) = execute(acts, radio, transport).await { return s; }
                    }
                    RlsMessage::HeartbeatAck(_) | RlsMessage::PduTransmissionAck(_) => {}
                }
            }
            _ = si_timer.tick() => {
                let ues: Vec<u32> = radio.ue_addr.keys().copied().collect();
                broadcast_si(&mut rrc, radio, &ues, sfn).await;
                sfn = (sfn + frames_per_si) & 0x3FF;
            }
            _ = lost_timer.tick() => {
                for ev in radio.tracker.check_lost_ues() {
                    if let GnbTrackerEvent::UeLost { ue_id } = ev {
                        radio.ue_addr.remove(&ue_id);
                        let acts = rrc.on_ue_lost(ue_id);
                        if let Err(s) = execute(acts, radio, transport).await { return s; }
                    }
                }
            }
        }
    }
}
