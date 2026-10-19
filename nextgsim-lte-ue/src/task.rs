//! The LTE UE's RLS runtime: heartbeats to the search space (cell search and
//! measurement, as RLS models them), RRC PDUs to and from the serving cell,
//! and the [`UeRrc`] timers.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

use bytes::Bytes;
use nextgsim_rls::{
    codec as rls_codec, CellSearchEvent, PduType, RlsMessage, RlsPduTransmission,
    RlsPduTransmissionAck, RrcChannel, SimCoord, UeCellSearch,
};
use tokio::net::UdpSocket;
use tokio::sync::watch;

use crate::config::LteUeConfig;
use crate::nas::NasLayer;
use crate::rrc::{UeAction, UeRrc, UeRrcState};

/// RLS heartbeat interval (as the NR UE).
const HEARTBEAT_MS: u64 = nextgsim_rls::DEFAULT_HEARTBEAT_INTERVAL_MS;

/// Run the UE until `stop` fires. Every state change is published on the
/// returned watch channel, so a caller (a test, or a future CLI) can follow
/// the RRC state without polling the task.
pub async fn run(
    cfg: &LteUeConfig,
    nas: Box<dyn NasLayer>,
    socket: UdpSocket,
    state_tx: watch::Sender<UeRrcState>,
    mut stop: watch::Receiver<bool>,
) -> std::io::Result<()> {
    let sti = cfg.sti.unwrap_or_else(rand::random);
    let search = cfg
        .search_space()
        .map_err(|e| std::io::Error::other(e.to_string()))?;
    let mut cells = UeCellSearch::new(sti, search);
    cells.set_position(SimCoord::new(
        cfg.position[0],
        cfg.position[1],
        cfg.position[2],
    ));
    let mut rrc = UeRrc::new(cfg.plmn, nas, Box::new(rand::random::<u64>));
    let mut cell_addr: HashMap<u64, SocketAddr> = HashMap::new();
    let mut next_pdu_id: u32 = 1;
    let mut hb = tokio::time::interval(Duration::from_millis(HEARTBEAT_MS));
    let mut buf = vec![0u8; 65535];
    log::info!(
        "LTE UE STI {sti:#x}, PLMN {}-{}",
        cfg.plmn.mcc,
        cfg.plmn.mnc
    );

    loop {
        let deadline = rrc.next_deadline();
        let sleep = async {
            match deadline {
                Some(d) => tokio::time::sleep_until(tokio::time::Instant::from_std(d)).await,
                None => std::future::pending().await,
            }
        };
        let actions = tokio::select! {
            _ = stop.changed() => return Ok(()),
            _ = hb.tick() => {
                for (to, msg) in cells.create_heartbeats() {
                    let _ = socket.send_to(&rls_codec::encode(&RlsMessage::Heartbeat(msg)), to).await;
                }
                let mut acts = Vec::new();
                for ev in cells.check_lost_cells() {
                    if let CellSearchEvent::CellLost { cell_id } = ev {
                        let lost: Vec<u64> = cell_addr.keys().copied()
                            .filter(|s| cells.get_cell_by_sti(*s).is_none()).collect();
                        for s in lost {
                            cell_addr.remove(&s);
                            log::info!("cell {s:#x} (RLS cell {cell_id}) lost");
                            acts.extend(rrc.on_cell_lost(s, Instant::now()));
                        }
                    }
                }
                acts
            }
            _ = sleep => rrc.on_timer(Instant::now()),
            r = socket.recv_from(&mut buf) => {
                let (n, from) = r?;
                match rls_codec::decode(&buf[..n]) {
                    Ok(RlsMessage::HeartbeatAck(ack)) => {
                        cells.process_heartbeat_ack(ack.sti, from, &ack);
                        cell_addr.insert(ack.sti, from);
                        rrc.on_cell_measurement(ack.sti, ack.dbm, Instant::now())
                    }
                    Ok(RlsMessage::PduTransmission(pdu)) if pdu.pdu_type == PduType::Rrc => {
                        let _ = socket.send_to(&rls_codec::encode(&RlsMessage::PduTransmissionAck(
                            RlsPduTransmissionAck { sti, pdu_ids: vec![pdu.pdu_id] },
                        )), from).await;
                        match RrcChannel::from_u32(pdu.payload) {
                            Some(ch) if ch.is_downlink() => {
                                rrc.on_downlink(pdu.sti, ch, &pdu.pdu, Instant::now())
                            }
                            _ => Vec::new(),
                        }
                    }
                    Ok(_) => Vec::new(),
                    Err(e) => { log::debug!("undecodable RLS datagram from {from}: {e}"); Vec::new() }
                }
            }
        };
        for UeAction::Rrc { cell, channel, pdu } in actions {
            let Some(&to) = cell_addr.get(&cell) else {
                log::warn!("no RLS address for cell {cell:#x}; PDU dropped");
                continue;
            };
            let id = next_pdu_id;
            next_pdu_id = next_pdu_id.wrapping_add(1).max(1);
            let msg = RlsMessage::PduTransmission(RlsPduTransmission::rrc(
                sti,
                id,
                channel as u32,
                Bytes::from(pdu),
            ));
            let _ = socket.send_to(&rls_codec::encode(&msg), to).await;
        }
        state_tx.send_if_modified(|s| {
            let now = rrc.state();
            let changed = *s != now;
            *s = now;
            changed
        });
    }
}
