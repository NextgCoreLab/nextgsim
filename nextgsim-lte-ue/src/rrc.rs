//! The LTE UE's RRC (TS 36.331) and idle-mode cell selection (TS 36.304).
//!
//! Pure: cell measurements, received PDUs and the clock in; [`UeAction`]s out.
//! [`crate::task`] owns the RLS socket and the timers.
//!
//! - **Cell selection** (TS 36.304 §5.2.3): a cell is *suitable* when SIB1
//!   lists the selected PLMN, the cell is not barred, and the S criterion
//!   holds -- `Srxlev = Qrxlevmeas - (Qrxlevmin + Qrxlevminoffset) -
//!   Pcompensation > 0`. `Qrxlevmeas` is the RLS signal level; no `p-Max` is
//!   broadcast, so `Pcompensation` is 0. The UE camps on the strongest
//!   suitable cell.
//! - **Establishment** (TS 36.331 §5.3.3): only with valid MIB, SIB1 and SIB2
//!   (§5.3.3.2: "ensure having valid and up to date essential system
//!   information"); T300 from SIB2 supervises it (§5.3.3.3, §7.3).
//! - **SRB1** is a PDCP entity (TS 36.323) from RRCConnectionSetup on.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use nextgsim_common::Plmn;
use nextgsim_lte_rrc::msg::{
    self, BcchDlSch, CellInfo, DlCcch, DlDcch, InitialUeIdentity, MibInfo, PlmnId, Sib2Info,
};
use nextgsim_pdcp::LteSrbPdcp;
use nextgsim_rls::RrcChannel;

use crate::nas::{InitialNas, NasLayer, RrcIndication};

/// Something the UE must transmit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UeAction {
    /// Send an RRC PDU to `cell` (RLS STI) on `channel`.
    Rrc {
        /// The cell, by its RLS STI.
        cell: u64,
        /// Uplink logical channel.
        channel: RrcChannel,
        /// The PDU.
        pdu: Vec<u8>,
    },
}

/// The UE's RRC state, for observers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UeRrcState {
    /// RRC_IDLE, no suitable cell.
    NoCell,
    /// RRC_IDLE, camped on `cell`.
    Camped {
        /// The cell's RLS STI.
        cell: u64,
    },
    /// RRCConnectionRequest sent, T300 running.
    Connecting {
        /// The cell.
        cell: u64,
    },
    /// RRC_CONNECTED.
    Connected {
        /// The cell.
        cell: u64,
    },
}

#[derive(Debug, Default, Clone)]
struct CellView {
    dbm: i32,
    mib: Option<MibInfo>,
    sib1: Option<CellInfo>,
    sib2: Option<Sib2Info>,
}

#[derive(Debug)]
enum State {
    Idle,
    Connecting {
        cell: u64,
        t300_expiry: Instant,
        initial: InitialNas,
    },
    Connected {
        cell: u64,
        srb1: LteSrbPdcp,
    },
}

/// The LTE UE's RRC entity.
pub struct UeRrc {
    plmn: PlmnId,
    cells: HashMap<u64, CellView>,
    camped: Option<u64>,
    state: State,
    /// T302 (§5.3.3.8): no new establishment until this instant.
    barred_until: Option<Instant>,
    nas: Box<dyn NasLayer>,
    random: Box<dyn FnMut() -> u64 + Send>,
}

impl std::fmt::Debug for UeRrc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UeRrc")
            .field("state", &self.state())
            .field("cells", &self.cells.len())
            .finish()
    }
}

impl UeRrc {
    /// An RRC_IDLE UE looking for a suitable cell of `plmn`. `random` supplies
    /// the 40-bit `randomValue` of RRCConnectionRequest (§5.3.3.3).
    pub fn new(plmn: Plmn, nas: Box<dyn NasLayer>, random: Box<dyn FnMut() -> u64 + Send>) -> Self {
        Self {
            plmn: PlmnId {
                mcc: plmn.mcc,
                mnc: plmn.mnc,
                long_mnc: plmn.long_mnc,
            },
            cells: HashMap::new(),
            camped: None,
            state: State::Idle,
            barred_until: None,
            nas,
            random,
        }
    }

    /// The current state.
    pub fn state(&self) -> UeRrcState {
        match &self.state {
            State::Connecting { cell, .. } => UeRrcState::Connecting { cell: *cell },
            State::Connected { cell, .. } => UeRrcState::Connected { cell: *cell },
            State::Idle => self
                .camped
                .map_or(UeRrcState::NoCell, |cell| UeRrcState::Camped { cell }),
        }
    }

    /// The earliest timer deadline, if any (T300, T302).
    pub fn next_deadline(&self) -> Option<Instant> {
        let t300 = match &self.state {
            State::Connecting { t300_expiry, .. } => Some(*t300_expiry),
            _ => None,
        };
        [t300, self.barred_until].into_iter().flatten().min()
    }

    /// TS 36.304 §5.2.3.2.
    fn suitable(&self, v: &CellView) -> bool {
        let Some(sib1) = &v.sib1 else {
            return false;
        };
        v.mib.is_some()
            && !sib1.cell_barred
            && sib1.plmns.contains(&self.plmn)
            && v.dbm - i32::from(sib1.q_rx_lev_min_dbm) > 0
    }

    fn reselect(&mut self, now: Instant) -> Vec<UeAction> {
        if !matches!(self.state, State::Idle) {
            return vec![];
        }
        let best = self
            .cells
            .iter()
            .filter(|(_, v)| self.suitable(v))
            .max_by_key(|(sti, v)| (v.dbm, std::cmp::Reverse(**sti)))
            .map(|(sti, _)| *sti);
        if best != self.camped {
            match best {
                Some(c) => log::info!("camped on cell {c:#x} (TS 36.304 §5.2.3)"),
                None => log::info!("no suitable cell"),
            }
            self.camped = best;
        }
        self.try_establish(now)
    }

    fn try_establish(&mut self, now: Instant) -> Vec<UeAction> {
        let Some(cell) = self.camped else {
            return vec![];
        };
        if !matches!(self.state, State::Idle) {
            return vec![];
        }
        if self.barred_until.is_some_and(|t| now < t) {
            return vec![];
        }
        let Some(sib2) = self.cells.get(&cell).and_then(|v| v.sib2) else {
            return vec![]; // §5.3.3.2: SIB2 is essential system information.
        };
        let Some(initial) = self.nas.initial_nas() else {
            return vec![];
        };
        let identity = match initial.s_tmsi {
            Some((mmec, m_tmsi)) => InitialUeIdentity::STmsi { mmec, m_tmsi },
            None => InitialUeIdentity::Random((self.random)() & ((1 << 40) - 1)),
        };
        let pdu = match msg::encode_rrc_connection_request(identity, initial.cause) {
            Ok(p) => p,
            Err(e) => {
                log::error!("cannot encode RRCConnectionRequest: {e}");
                return vec![];
            }
        };
        let t300 = Duration::from_millis(u64::from(sib2.timers.t300_ms));
        log::info!(
            "RRCConnectionRequest ({:?}) to cell {cell:#x}; T300 {} ms",
            initial.cause,
            t300.as_millis()
        );
        self.state = State::Connecting {
            cell,
            t300_expiry: now + t300,
            initial,
        };
        vec![UeAction::Rrc {
            cell,
            channel: RrcChannel::UlCcch,
            pdu,
        }]
    }

    /// The signal level of `cell` (an RLS heartbeat acknowledgement).
    pub fn on_cell_measurement(&mut self, cell: u64, dbm: i32, now: Instant) -> Vec<UeAction> {
        self.cells.entry(cell).or_default().dbm = dbm;
        self.reselect(now)
    }

    /// `cell` is no longer heard.
    pub fn on_cell_lost(&mut self, cell: u64, now: Instant) -> Vec<UeAction> {
        self.cells.remove(&cell);
        let serving = match &self.state {
            State::Connecting { cell: c, .. } | State::Connected { cell: c, .. } => Some(*c),
            State::Idle => None,
        };
        if serving == Some(cell) {
            // §5.3.11.3: radio link failure before AS security activation
            // leaves RRC_CONNECTED with release cause "other"; during
            // establishment it is the §5.3.3.6 failure.
            let was_connecting = matches!(self.state, State::Connecting { .. });
            self.state = State::Idle;
            log::warn!("radio link to serving cell {cell:#x} lost");
            self.nas.on_rrc_indication(if was_connecting {
                RrcIndication::EstablishmentFailed { wait_time_s: None }
            } else {
                RrcIndication::Released
            });
        }
        if self.camped == Some(cell) {
            self.camped = None;
        }
        self.reselect(now)
    }

    /// Timers: T300 (§5.3.3.6) and T302 (§5.3.3.8).
    pub fn on_timer(&mut self, now: Instant) -> Vec<UeAction> {
        if let State::Connecting { t300_expiry, .. } = &self.state {
            if now >= *t300_expiry {
                log::warn!("T300 expired: RRC connection establishment failed (§5.3.3.6)");
                self.state = State::Idle;
                self.nas
                    .on_rrc_indication(RrcIndication::EstablishmentFailed { wait_time_s: None });
            }
        }
        if self.barred_until.is_some_and(|t| now >= t) {
            self.barred_until = None;
        }
        self.reselect(now)
    }

    /// An RRC PDU from `cell` on a downlink channel.
    pub fn on_downlink(
        &mut self,
        cell: u64,
        channel: RrcChannel,
        pdu: &[u8],
        now: Instant,
    ) -> Vec<UeAction> {
        match channel {
            RrcChannel::BcchBch => {
                match msg::decode_mib(pdu) {
                    Ok(m) => self.cells.entry(cell).or_default().mib = Some(m),
                    Err(e) => log::debug!("cell {cell:#x}: undecodable MIB: {e}"),
                }
                self.reselect(now)
            }
            RrcChannel::BcchDlSch => {
                match msg::decode_bcch_dl_sch(pdu) {
                    Ok(BcchDlSch::Sib1(s)) => self.cells.entry(cell).or_default().sib1 = Some(s),
                    Ok(BcchDlSch::SystemInformation { sib2: Some(s) }) => {
                        self.cells.entry(cell).or_default().sib2 = Some(s);
                    }
                    Ok(_) => {}
                    Err(e) => log::debug!("cell {cell:#x}: undecodable BCCH-DL-SCH: {e}"),
                }
                self.reselect(now)
            }
            RrcChannel::DlCcch => self.on_dl_ccch(cell, pdu, now),
            RrcChannel::DlDcch => self.on_dl_dcch(cell, pdu, now),
            other => {
                log::debug!("cell {cell:#x}: RRC PDU on {other:?} ignored");
                vec![]
            }
        }
    }

    fn on_dl_ccch(&mut self, cell: u64, pdu: &[u8], now: Instant) -> Vec<UeAction> {
        let State::Connecting { cell: serving, .. } = &self.state else {
            return vec![];
        };
        if *serving != cell {
            return vec![];
        }
        match msg::decode_dl_ccch(pdu) {
            Ok(DlCcch::ConnectionSetup {
                transaction_id,
                configures_srb1,
            }) => {
                if !configures_srb1 {
                    // §5.3.3.4 requires SRB1 to be established from this
                    // message; without it there is nowhere to send Complete.
                    log::warn!("RRCConnectionSetup without SRB1; ignored");
                    return vec![];
                }
                let State::Connecting { initial, .. } =
                    std::mem::replace(&mut self.state, State::Idle)
                else {
                    unreachable!("checked above");
                };
                // §5.3.3.4: stop T300, establish SRB1, send Complete with the
                // selected PLMN's 1-based index in SIB1's list.
                let index = self
                    .cells
                    .get(&cell)
                    .and_then(|v| v.sib1.as_ref())
                    .and_then(|s| s.plmns.iter().position(|p| *p == self.plmn))
                    .map_or(1, |i| i as u8 + 1);
                let mut srb1 = LteSrbPdcp::new();
                let complete = match msg::encode_rrc_connection_setup_complete(
                    transaction_id,
                    index,
                    &initial.nas,
                ) {
                    Ok(c) => c,
                    Err(e) => {
                        log::error!("cannot encode RRCConnectionSetupComplete: {e}");
                        return vec![];
                    }
                };
                let (pdcp, _) = srb1.transmit(&complete);
                self.state = State::Connected { cell, srb1 };
                log::info!("RRCConnectionSetup; RRCConnectionSetupComplete sent (RRC_CONNECTED)");
                self.nas.on_rrc_indication(RrcIndication::Connected);
                vec![UeAction::Rrc {
                    cell,
                    channel: RrcChannel::UlDcch,
                    pdu: pdcp,
                }]
            }
            Ok(DlCcch::ConnectionReject { wait_time_s }) => {
                // §5.3.3.8: stop T300, start T302 with waitTime, inform NAS.
                log::warn!("RRCConnectionReject, waitTime {wait_time_s} s");
                self.state = State::Idle;
                self.barred_until = Some(now + Duration::from_secs(u64::from(wait_time_s)));
                self.nas
                    .on_rrc_indication(RrcIndication::EstablishmentFailed {
                        wait_time_s: Some(wait_time_s),
                    });
                vec![]
            }
            Ok(DlCcch::Unsupported) => vec![],
            Err(e) => {
                log::warn!("undecodable DL-CCCH message: {e}");
                vec![]
            }
        }
    }

    fn on_dl_dcch(&mut self, cell: u64, pdu: &[u8], now: Instant) -> Vec<UeAction> {
        let State::Connected {
            cell: serving,
            srb1,
        } = &mut self.state
        else {
            return vec![];
        };
        if *serving != cell {
            return vec![];
        }
        let sdu = match srb1.receive(pdu) {
            Ok((sdu, _)) => sdu,
            Err(e) => {
                log::warn!("SRB1 PDCP: {e}");
                return vec![];
            }
        };
        match msg::decode_dl_dcch(&sdu) {
            Ok(DlDcch::InformationTransfer { nas }) => {
                let replies = self.nas.on_downlink_nas(&nas);
                self.send_ul_nas(cell, replies)
            }
            Ok(DlDcch::ConnectionRelease) => {
                // §5.3.8.3: leave RRC_CONNECTED, release cause "other".
                log::info!("RRCConnectionRelease: back to RRC_IDLE");
                self.state = State::Idle;
                self.nas.on_rrc_indication(RrcIndication::Released);
                self.reselect(now)
            }
            Ok(DlDcch::Unsupported) => {
                log::info!("DL-DCCH message not supported in this build, ignored");
                vec![]
            }
            Err(e) => {
                log::warn!("undecodable DL-DCCH message: {e}");
                vec![]
            }
        }
    }

    fn send_ul_nas(&mut self, cell: u64, nas_msgs: Vec<Vec<u8>>) -> Vec<UeAction> {
        let State::Connected { srb1, .. } = &mut self.state else {
            return vec![];
        };
        nas_msgs
            .into_iter()
            .filter_map(|nas| match msg::encode_ul_information_transfer(&nas) {
                Ok(rrc) => Some(UeAction::Rrc {
                    cell,
                    channel: RrcChannel::UlDcch,
                    pdu: srb1.transmit(&rrc).0,
                }),
                Err(e) => {
                    log::error!("cannot encode ULInformationTransfer: {e}");
                    None
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests;
