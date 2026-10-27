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
//! - **AS security** (§5.3.4.3): on SecurityModeCommand the UE derives
//!   `K_eNB` from NAS's `K_ASME` (TS 33.401 §7.2.6.2), `K_RRCint`, `K_RRCenc`
//!   and `K_UPenc` (Annex A.7), verifies the command's MAC-I, and switches
//!   SRB1's PDCP on in the order of TS 33.401 §7.2.4.5 -- the
//!   SecurityModeComplete goes out integrity protected and unciphered.
//! - **RRCConnectionReconfiguration** (§5.3.5.3) adds SRB2 and the DRBs of the
//!   default bearer and hands `dedicatedInfoNASList` (the ATTACH ACCEPT) to
//!   NAS. SRB2 runs with SRB1's AS security; uplink NAS moves onto it (TS
//!   36.331 §4.2.2: SRB2 is "for NAS messages"). Each DRB is a TS 36.323 PDCP
//!   entity ciphered with `K_UPenc` over TS 36.322 RLC UM; the UE refuses a
//!   DRB configuration it does not support by leaving RRC_CONNECTED
//!   (§5.3.5.5).
//! - **User plane**: IP packets in ([`UeRrc::send_ip`]) go on the DRB of the
//!   bearer the TFTs pick; RLC PDUs in ([`UeRrc::on_data`]) come out as
//!   [`UeAction::DeliverIp`]. Packets offered with no DRB, while NAS expects
//!   one (a service request, an attach), are held in a bounded
//!   [`UplinkBuffer`] and sent in order once the DRBs are set up; a failed
//!   service request discards them (#435).

use std::collections::HashMap;
use std::time::{Duration, Instant};

use nextgsim_common::Plmn;
use nextgsim_crypto::eps_kdf::{derive_algorithm_key, EpsAlgorithmType};
use nextgsim_eps_nas::{BearerCommand, BearerCommandRefusal};
use nextgsim_lte_rrc::msg::{
    self, BcchDlSch, CellInfo, DlCcch, DlDcch, DrbConfig, DrbRlc, InitialUeIdentity, MibInfo,
    PlmnId, Reconfiguration, Sib2Info,
};
use nextgsim_pdcp::lte_drb::drb_bearer;
use nextgsim_pdcp::lte_srb::{verify_mac_i, DIRECTION_DOWNLINK, DIRECTION_UPLINK, SRB1_BEARER};
use nextgsim_pdcp::{LteDrbPdcp, LteSrbPdcp, LteSrbReceived};
use nextgsim_rlc::lte_um::{LteUmRx, LteUmTx};
use nextgsim_rls::lte::{LCID_SRB1, LCID_SRB2};
use nextgsim_rls::RrcChannel;

use crate::nas::{InitialNas, NasLayer, RrcIndication};
use crate::uplink_buffer::UplinkBuffer;

/// BEARER of SRB2: RB identity 2 − 1 (TS 36.323 §5.6.0).
const SRB2_BEARER: u8 = 1;
/// `t-Reordering` of the DRBs' RLC UM receivers: the value this eNB sends
/// (`nextgsim_lte_rrc::msg`, ms35).
const DRB_T_REORDERING: Duration = Duration::from_millis(35);

/// Something the UE must do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UeAction {
    /// Send an RRC PDU to `cell` (RLS STI) on `channel`.
    Rrc {
        /// The cell, by its RLS STI.
        cell: u64,
        /// Uplink logical channel.
        channel: RrcChannel,
        /// LCID on DCCH (1 SRB1, 2 SRB2), 0 on CCCH.
        lcid: u8,
        /// The PDU.
        pdu: Vec<u8>,
    },
    /// Send an RLC PDU to `cell` on the DRB `lcid`.
    Data {
        /// The cell.
        cell: u64,
        /// The DRB's LCID.
        lcid: u8,
        /// The RLC UMD PDU.
        pdu: Vec<u8>,
    },
    /// Hand a received IP packet to the IP stack (the TUN interface).
    DeliverIp(Vec<u8>),
}

/// AS keys of the connection (TS 33.401 Annex A.7). Never printed.
#[derive(Clone)]
struct AsKeys {
    eea: u8,
    eia: u8,
    k_rrc_int: [u8; 16],
    k_rrc_enc: [u8; 16],
    k_up_enc: [u8; 16],
}

impl std::fmt::Debug for AsKeys {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EEA{}/EIA{} (keys redacted)", self.eea, self.eia)
    }
}

/// One DRB's user plane.
#[derive(Debug)]
struct UeDrb {
    ebi: u8,
    pdcp: LteDrbPdcp,
    tx: LteUmTx,
    rx: LteUmRx,
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
        /// SRB2, once a reconfiguration added it (boxed, like `keys`: the
        /// variant is otherwise several times the size of the others).
        srb2: Option<Box<LteSrbPdcp>>,
        /// AS security activated (§5.3.4.3).
        secured: bool,
        /// The AS keys, from the Security Mode Command on.
        keys: Option<Box<AsKeys>>,
        /// DRBs added by RRCConnectionReconfiguration.
        drbs: Vec<DrbConfig>,
        /// Their user plane, by LCID.
        user: HashMap<u8, UeDrb>,
    },
}

/// The LTE UE's RRC entity.
pub struct UeRrc {
    plmn: PlmnId,
    cells: HashMap<u64, CellView>,
    camped: Option<u64>,
    /// The tracking area code last reported to NAS (#356).
    reported_tac: Option<u16>,
    state: State,
    /// T302 (§5.3.3.8): no new establishment until this instant.
    barred_until: Option<Instant>,
    nas: Box<dyn NasLayer>,
    random: Box<dyn FnMut() -> u64 + Send>,
    /// Uplink IP packets waiting for the user plane (#435).
    held: UplinkBuffer,
    /// Held packets discarded because the service request failed.
    held_discarded: u64,
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
            reported_tac: None,
            state: State::Idle,
            barred_until: None,
            nas,
            random,
            held: UplinkBuffer::default(),
            held_discarded: 0,
        }
    }

    /// Uplink IP packets held for the user plane now (#435).
    pub fn held_uplink(&self) -> usize {
        self.held.len()
    }

    /// Held uplink packets dropped because the buffer was full (#435).
    pub fn held_uplink_dropped(&self) -> u64 {
        self.held.dropped()
    }

    /// Held uplink packets discarded because the procedure they waited for
    /// ended without the user plane (#435).
    pub fn held_uplink_discarded(&self) -> u64 {
        self.held_discarded
    }

    /// After every input that can move NAS or set up DRBs (#435): send what
    /// is held once a DRB exists, or discard it when the service request it
    /// waited for failed, or when, idle again, nothing will set up the user
    /// plane for it.
    fn settle(&mut self, mut acts: Vec<UeAction>) -> Vec<UeAction> {
        if let Some(why) = self.nas.take_service_request_failure() {
            self.discard_held(&format!("the service request failed ({why})"));
        }
        if self.held.is_empty() {
            return acts;
        }
        match &self.state {
            State::Connected { user, .. } if !user.is_empty() => {
                let held = self.held.drain();
                log::info!(
                    "user plane ready: sending {} uplink packet(s) held while it was down",
                    held.len()
                );
                for p in held {
                    acts.extend(self.transmit_ip(&p));
                }
            }
            State::Idle if !self.nas.awaits_user_plane() => {
                self.discard_held("the connection ended before the user plane was set up");
            }
            _ => {}
        }
        acts
    }

    fn discard_held(&mut self, why: &str) {
        if self.held.is_empty() {
            return;
        }
        let n = self.held.drain().len();
        self.held_discarded += n as u64;
        log::warn!("{n} held uplink packet(s) discarded: {why}");
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

    /// The earliest timer deadline, if any (T300, T302, NAS timers).
    pub fn next_deadline(&self) -> Option<Instant> {
        let t300 = match &self.state {
            State::Connecting { t300_expiry, .. } => Some(*t300_expiry),
            _ => None,
        };
        let drb = match &self.state {
            State::Connected { user, .. } => {
                user.values().filter_map(|d| d.rx.next_deadline()).min()
            }
            _ => None,
        };
        [t300, self.barred_until, self.nas.next_deadline(), drb]
            .into_iter()
            .flatten()
            .min()
    }

    /// Whether AS security is active on the connection.
    pub fn as_security_active(&self) -> bool {
        matches!(self.state, State::Connected { secured: true, .. })
    }

    /// The DRBs of the connection.
    pub fn drbs(&self) -> Vec<DrbConfig> {
        match &self.state {
            State::Connected { drbs, .. } => drbs.clone(),
            _ => Vec::new(),
        }
    }

    /// Leave RRC_CONNECTED (§5.3.12) and tell NAS the connection is gone.
    fn leave_connected(&mut self, why: &str, now: Instant) -> Vec<UeAction> {
        log::warn!("leaving RRC_CONNECTED: {why}");
        self.state = State::Idle;
        self.nas.on_rrc_indication(RrcIndication::Released);
        self.reselect(now)
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
        // TS 36.304 §5.2.3: camped, NAS learns the cell's tracking area (SIB1
        // trackingAreaCode) and may owe a tracking area update for it
        // (TS 24.301 §5.5.3.2.2 a), #356).
        let tac = self
            .camped
            .and_then(|c| self.cells.get(&c))
            .and_then(|v| v.sib1.as_ref())
            .map(|s| s.tac);
        if let Some(tac) = tac.filter(|t| Some(*t) != self.reported_tac) {
            self.reported_tac = Some(tac);
            self.nas.on_serving_cell(tac);
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
            lcid: 0,
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
        let acts = self.on_cell_lost_inner(cell, now);
        self.settle(acts)
    }

    fn on_cell_lost_inner(&mut self, cell: u64, now: Instant) -> Vec<UeAction> {
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

    /// Timers: T300 (§5.3.3.6), T302 (§5.3.3.8) and NAS's.
    pub fn on_timer(&mut self, now: Instant) -> Vec<UeAction> {
        let acts = self.on_timer_inner(now);
        self.settle(acts)
    }

    fn on_timer_inner(&mut self, now: Instant) -> Vec<UeAction> {
        let mut delivered = Vec::new();
        if let State::Connected { user, .. } = &mut self.state {
            for d in user.values_mut() {
                for sdu in d.rx.on_timer(now) {
                    match d.pdcp.receive(&sdu) {
                        Ok((ip, _)) => delivered.push(UeAction::DeliverIp(ip)),
                        Err(e) => log::warn!("DRB of EPS bearer {}: PDCP: {e}", d.ebi),
                    }
                }
            }
        }
        if !delivered.is_empty() {
            return delivered;
        }
        if self.nas.next_deadline().is_some_and(|t| now >= t) {
            let out = self.nas.on_timer(now);
            if out.release && matches!(self.state, State::Connected { .. }) {
                // Upper layers request the release (§5.3.12, "release cause
                // other").
                return self.leave_connected("NAS requested a local release", now);
            }
            if let State::Connected { cell, .. } = &self.state {
                let cell = *cell;
                let acts = self.send_ul_nas(cell, out.uplink);
                if !acts.is_empty() {
                    return acts;
                }
            }
        }
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
        lcid: u8,
        pdu: &[u8],
        now: Instant,
    ) -> Vec<UeAction> {
        let acts = self.on_downlink_inner(cell, channel, lcid, pdu, now);
        self.settle(acts)
    }

    fn on_downlink_inner(
        &mut self,
        cell: u64,
        channel: RrcChannel,
        lcid: u8,
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
            RrcChannel::DlDcch => self.on_dl_dcch(cell, lcid, pdu, now),
            RrcChannel::Pcch => self.on_paging(cell, pdu, now),
            other => {
                log::debug!("cell {cell:#x}: RRC PDU on {other:?} ignored");
                vec![]
            }
        }
    }

    /// A Paging message on PCCH (TS 36.331 §5.3.2.3, #351): in RRC_IDLE on the
    /// camped cell, each PS record is forwarded to NAS, which says whether the
    /// identity is the UE's and it wants a connection; the establishment then
    /// follows with NAS's initial message (a SERVICE REQUEST, cause mt-Access).
    fn on_paging(&mut self, cell: u64, pdu: &[u8], now: Instant) -> Vec<UeAction> {
        if !matches!(self.state, State::Idle) || self.camped != Some(cell) {
            return vec![];
        }
        let records = match msg::decode_pcch(pdu) {
            Ok(r) => r,
            Err(e) => {
                log::debug!("cell {cell:#x}: undecodable PCCH message: {e}");
                return vec![];
            }
        };
        let mut wanted = false;
        for r in records {
            if r.cn_domain == msg::PagingCnDomain::Cs {
                log::debug!("CS-domain paging record ignored: this UE has no CS fallback");
                continue;
            }
            if self.nas.on_paging(&r.identity) {
                log::info!("paged ({:?}) on cell {cell:#x}", r.identity);
                wanted = true;
            }
        }
        if wanted {
            self.try_establish(now)
        } else {
            vec![]
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
                self.state = State::Connected {
                    cell,
                    srb1,
                    srb2: None,
                    secured: false,
                    keys: None,
                    drbs: Vec::new(),
                    user: HashMap::new(),
                };
                log::info!("RRCConnectionSetup; RRCConnectionSetupComplete sent (RRC_CONNECTED)");
                self.nas.on_rrc_indication(RrcIndication::Connected);
                vec![UeAction::Rrc {
                    cell,
                    channel: RrcChannel::UlDcch,
                    lcid: LCID_SRB1,
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

    fn on_dl_dcch(&mut self, cell: u64, lcid: u8, pdu: &[u8], now: Instant) -> Vec<UeAction> {
        let State::Connected {
            cell: serving,
            srb1,
            srb2,
            ..
        } = &mut self.state
        else {
            return vec![];
        };
        if *serving != cell {
            return vec![];
        }
        let srb = match (lcid, srb2) {
            (LCID_SRB2, Some(s)) => &mut **s,
            (LCID_SRB2, None) => {
                log::warn!("PDU on SRB2 before SRB2 was established; discarded");
                return vec![];
            }
            _ => srb1,
        };
        let received = match srb.receive(pdu) {
            Ok(r) => r,
            Err(nextgsim_pdcp::LteSrbPdcpError::IntegrityCheckFailed(count)) => {
                // §5.3.7.2: an "integrity check failure indication from lower
                // layers concerning SRB1 or SRB2" initiates RRC connection
                // re-establishment, which this UE does not implement; it leaves
                // RRC_CONNECTED instead (§5.3.12).
                return self.leave_connected(
                    &format!("SRB{lcid} integrity check failure at COUNT {count}"),
                    now,
                );
            }
            Err(e) => {
                log::warn!("SRB{lcid} PDCP: {e}");
                return vec![];
            }
        };
        match msg::decode_dl_dcch(&received.sdu) {
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
            Ok(DlDcch::SecurityModeCommand {
                transaction_id,
                eea,
                eia,
            }) => self.on_security_mode_command(cell, &received, transaction_id, eea, eia),
            Ok(DlDcch::ConnectionReconfiguration(r)) => self.on_reconfiguration(cell, r, now),
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

    /// §5.3.4.3, with the PDCP ordering of TS 33.401 §7.2.4.5.
    fn on_security_mode_command(
        &mut self,
        cell: u64,
        received: &LteSrbReceived,
        transaction_id: u8,
        eea: u8,
        eia: u8,
    ) -> Vec<UeAction> {
        let k_enb = self.nas.k_enb();
        let State::Connected {
            srb1,
            secured,
            keys,
            ..
        } = &mut self.state
        else {
            return vec![];
        };
        let failure =
            |srb1: &mut LteSrbPdcp| match msg::encode_security_mode_failure(transaction_id) {
                // "neither apply integrity protection nor ciphering".
                Ok(f) => vec![UeAction::Rrc {
                    cell,
                    channel: RrcChannel::UlDcch,
                    lcid: LCID_SRB1,
                    pdu: srb1.transmit(&f).0,
                }],
                Err(e) => {
                    log::error!("cannot encode SecurityModeFailure: {e}");
                    vec![]
                }
            };
        let Some(k_enb) = k_enb else {
            log::warn!("SecurityModeCommand with no NAS security context; SecurityModeFailure");
            return failure(srb1);
        };
        if *secured {
            log::warn!("SecurityModeCommand with AS security already active; ignored");
            return vec![];
        }
        let k_rrc_int = derive_algorithm_key(&k_enb, EpsAlgorithmType::RrcInt, eia);
        if eia > 3
            || eea > 3
            || !verify_mac_i(received, eia, &k_rrc_int, SRB1_BEARER, DIRECTION_DOWNLINK)
        {
            log::warn!("SecurityModeCommand failed its integrity check; SecurityModeFailure");
            return failure(srb1);
        }
        let k_rrc_enc = derive_algorithm_key(&k_enb, EpsAlgorithmType::RrcEnc, eea);
        let complete = match msg::encode_security_mode_complete(transaction_id) {
            Ok(c) => c,
            Err(e) => {
                log::error!("cannot encode SecurityModeComplete: {e}");
                return vec![];
            }
        };
        let activated = srb1
            .activate_rx_integrity(eia, k_rrc_int)
            .and_then(|()| srb1.activate_rx_ciphering(eea, k_rrc_enc))
            .and_then(|()| srb1.activate_tx_integrity(eia, k_rrc_int));
        if let Err(e) = activated {
            log::error!("SRB1: {e}");
            return vec![];
        }
        let (pdu, _) = srb1.transmit(&complete);
        // "ciphering ... except for the SecurityModeComplete message".
        if let Err(e) = srb1.activate_tx_ciphering(eea, k_rrc_enc) {
            log::error!("SRB1: {e}");
        }
        *secured = true;
        *keys = Some(Box::new(AsKeys {
            eea,
            eia,
            k_rrc_int,
            k_rrc_enc,
            k_up_enc: derive_algorithm_key(&k_enb, EpsAlgorithmType::UpEnc, eea),
        }));
        log::info!("SecurityModeCommand EEA{eea}/EIA{eia}: AS security activated");
        vec![UeAction::Rrc {
            cell,
            channel: RrcChannel::UlDcch,
            lcid: LCID_SRB1,
            pdu,
        }]
    }

    /// §5.3.5.3: apply the radio resource configuration, hand the NAS list to
    /// upper layers, and complete.
    fn on_reconfiguration(&mut self, cell: u64, r: Reconfiguration, now: Instant) -> Vec<UeAction> {
        let State::Connected { secured, .. } = &mut self.state else {
            return vec![];
        };
        if r.mobility_control {
            log::warn!(
                "RRCConnectionReconfiguration with mobilityControlInfo: handover not supported"
            );
            return self.leave_connected("cannot comply with a handover command", now);
        }
        if !*secured && !r.drbs.is_empty() {
            // §5.3.5.2 permits DRB establishment only after AS security; a UE
            // that cannot comply leaves RRC_CONNECTED (§5.3.5.5).
            return self.leave_connected("DRB setup before AS security activation", now);
        }
        if let Some(d) = r.drbs.iter().find(|d| d.rlc != DrbRlc::UmSn10) {
            // This UE implements RLC UM with 10-bit SNs and PDCP with 12-bit
            // SNs for its DRBs; anything else it cannot comply with.
            let why = format!("DRB {} configured as {:?}", d.drb_identity, d.rlc);
            return self.leave_connected(&format!("cannot comply: {why}"), now);
        }
        let State::Connected {
            drbs,
            srb2,
            keys,
            user,
            ..
        } = &mut self.state
        else {
            return vec![];
        };
        // §5.3.10.2: release each DRB of drb-ToReleaseList that is part of the
        // current configuration, PDCP and RLC with it (#283)
        for id in &r.drbs_released {
            if let Some(pos) = drbs.iter().position(|d| d.drb_identity == *id) {
                let d = drbs.remove(pos);
                user.remove(&d.logical_channel_identity);
                log::info!(
                    "DRB {} released (EPS bearer {}, logical channel {})",
                    d.drb_identity,
                    d.eps_bearer_identity,
                    d.logical_channel_identity
                );
            }
        }
        if r.srb2 && srb2.is_none() {
            let Some(k) = keys.as_ref() else {
                return self.leave_connected("SRB2 setup without AS security", now);
            };
            let mut s = LteSrbPdcp::with_bearer(SRB2_BEARER, DIRECTION_UPLINK);
            let on = s
                .activate_tx_integrity(k.eia, k.k_rrc_int)
                .and_then(|()| s.activate_rx_integrity(k.eia, k.k_rrc_int))
                .and_then(|()| s.activate_tx_ciphering(k.eea, k.k_rrc_enc))
                .and_then(|()| s.activate_rx_ciphering(k.eea, k.k_rrc_enc));
            if let Err(e) = on {
                log::error!("SRB2: {e}");
                return vec![];
            }
            log::info!("SRB2 established");
            *srb2 = Some(Box::new(s));
        }
        for d in &r.drbs {
            let Some(k) = keys.as_ref() else {
                break;
            };
            let pdcp = match LteDrbPdcp::new(
                drb_bearer(d.logical_channel_identity),
                DIRECTION_UPLINK,
                k.eea,
                k.k_up_enc,
            ) {
                Ok(p) => p,
                Err(e) => {
                    log::error!("DRB {}: {e}", d.drb_identity);
                    continue;
                }
            };
            log::info!(
                "DRB {} configured for EPS bearer {} on logical channel {} (RLC UM SN10, PDCP SN12)",
                d.drb_identity,
                d.eps_bearer_identity,
                d.logical_channel_identity
            );
            user.insert(
                d.logical_channel_identity,
                UeDrb {
                    ebi: d.eps_bearer_identity,
                    pdcp,
                    tx: LteUmTx::new(),
                    rx: LteUmRx::new(DRB_T_REORDERING),
                },
            );
            drbs.retain(|x| x.drb_identity != d.drb_identity);
            drbs.push(*d);
        }
        let complete = match msg::encode_rrc_connection_reconfiguration_complete(r.transaction_id) {
            Ok(c) => c,
            Err(e) => {
                log::error!("cannot encode RRCConnectionReconfigurationComplete: {e}");
                return vec![];
            }
        };
        let mut replies = Vec::new();
        for nas in &r.nas {
            replies.extend(self.nas.on_downlink_nas(nas));
        }
        if !r.drbs.is_empty() {
            self.nas
                .on_rrc_indication(RrcIndication::RadioBearersEstablished);
        }
        let State::Connected { srb1, .. } = &mut self.state else {
            return vec![];
        };
        let mut acts = vec![UeAction::Rrc {
            cell,
            channel: RrcChannel::UlDcch,
            lcid: LCID_SRB1,
            pdu: srb1.transmit(&complete).0,
        }];
        acts.extend(self.send_ul_nas(cell, replies));
        acts
    }

    /// An RLC PDU from `cell` on the DRB `lcid`: the IP packets it completes.
    pub fn on_data(&mut self, cell: u64, lcid: u8, pdu: &[u8], now: Instant) -> Vec<UeAction> {
        let State::Connected {
            cell: serving,
            user,
            ..
        } = &mut self.state
        else {
            return vec![];
        };
        if *serving != cell {
            return vec![];
        }
        let Some(d) = user.get_mut(&lcid) else {
            log::debug!("data PDU on LCID {lcid} with no DRB; discarded");
            return vec![];
        };
        let sdus = match d.rx.receive(pdu, now) {
            Ok(s) => s,
            Err(e) => {
                log::warn!("LCID {lcid}: RLC UM: {e}");
                return vec![];
            }
        };
        sdus.into_iter()
            .filter_map(|sdu| match d.pdcp.receive(&sdu) {
                Ok((ip, _)) => Some(UeAction::DeliverIp(ip)),
                Err(e) => {
                    log::warn!("LCID {lcid}: PDCP: {e}");
                    None
                }
            })
            .collect()
    }

    /// A UE-initiated detach (TS 24.301 §5.5.2.2, #365).
    ///
    /// Connected, the DETACH REQUEST goes out now in a ULInformationTransfer.
    /// In RRC_IDLE, NAS holds it as its initial NAS message and the UE
    /// establishes a connection for it, as it does for pending uplink data.
    /// While a connection is being established nothing is sent: NAS already
    /// handed RRC its initial message, and the detach is asked for again once
    /// connected (the caller retries).
    pub fn detach(&mut self, switch_off: bool, now: Instant) -> Vec<UeAction> {
        let acts = self.detach_inner(switch_off, now);
        self.settle(acts)
    }

    fn detach_inner(&mut self, switch_off: bool, now: Instant) -> Vec<UeAction> {
        match &self.state {
            State::Connected { cell, .. } => {
                let cell = *cell;
                let out = self.nas.detach(switch_off, true);
                let mut acts = self.send_ul_nas(cell, out.uplink);
                if out.release {
                    acts.extend(self.leave_connected("NAS requested a local release", now));
                }
                acts
            }
            State::Idle => {
                let out = self.nas.detach(switch_off, false);
                if out.uplink.is_empty() {
                    self.try_establish(now)
                } else {
                    Vec::new()
                }
            }
            State::Connecting { .. } => {
                log::info!("detach asked for while an RRC connection is being set up; not now");
                Vec::new()
            }
        }
    }

    /// A UE-requested bearer resource modification (TS 24.301 §6.5.4, #434).
    ///
    /// Connected, NAS sends the BEARER RESOURCE MODIFICATION REQUEST now in a
    /// ULInformationTransfer and the PTI is returned. In RRC_IDLE NAS asks for
    /// a connection (a SERVICE REQUEST) and the UE establishes one, and while
    /// one is being set up nothing is sent: both answer
    /// [`BearerCommandRefusal::NotConnected`], and the caller asks again.
    pub fn bearer_command(
        &mut self,
        cmd: BearerCommand,
        now: Instant,
    ) -> (Vec<UeAction>, Result<u8, BearerCommandRefusal>) {
        let (acts, r) = self.bearer_command_inner(cmd, now);
        (self.settle(acts), r)
    }

    fn bearer_command_inner(
        &mut self,
        cmd: BearerCommand,
        now: Instant,
    ) -> (Vec<UeAction>, Result<u8, BearerCommandRefusal>) {
        match &self.state {
            State::Connected { cell, .. } => {
                let cell = *cell;
                match self.nas.bearer_command(cmd, true) {
                    Ok((pti, out)) => (self.send_ul_nas(cell, out.uplink), Ok(pti)),
                    Err(e) => (Vec::new(), Err(e)),
                }
            }
            State::Idle => {
                let r = self.nas.bearer_command(cmd, false).map(|(pti, _)| pti);
                let acts = if r == Err(BearerCommandRefusal::NotConnected) {
                    self.try_establish(now)
                } else {
                    Vec::new()
                };
                (acts, r)
            }
            State::Connecting { .. } => {
                // Refusals still answer; a request is not sent yet.
                let r = self.nas.bearer_command(cmd, false).map(|(pti, _)| pti);
                (Vec::new(), r)
            }
        }
    }

    /// An IP packet from the IP stack, for the DRB of the bearer the TFTs
    /// pick (the lowest LCID when NAS names no default bearer).
    ///
    /// With no DRB set up, in RRC_IDLE NAS is told uplink data is pending,
    /// which starts a service request (TS 24.301 §5.6.1.1 b). The packet is
    /// held while NAS expects the user plane radio bearers (§5.6.1.4), or
    /// while earlier packets are held, and goes out first, in order, once
    /// they are set up (#435); otherwise it is dropped.
    pub fn send_ip(&mut self, packet: &[u8], now: Instant) -> Vec<UeAction> {
        let acts = self.send_ip_inner(packet, now);
        self.settle(acts)
    }

    fn send_ip_inner(&mut self, packet: &[u8], now: Instant) -> Vec<UeAction> {
        // `settle` sends what is held as soon as a DRB exists, so with one,
        // nothing held is waiting ahead of this packet.
        if matches!(&self.state, State::Connected { user, .. } if !user.is_empty()) {
            return self.transmit_ip(packet).into_iter().collect();
        }
        let idle = matches!(self.state, State::Idle);
        if idle {
            self.nas.on_uplink_pending();
        }
        if self.nas.awaits_user_plane() || !self.held.is_empty() {
            self.held.hold(packet);
            log::debug!(
                "no user plane yet: uplink packet held ({} held, {} dropped)",
                self.held.len(),
                self.held.dropped()
            );
        } else {
            log::debug!("uplink packet discarded: no DRB and no procedure to set one up");
        }
        if idle {
            self.try_establish(now)
        } else {
            Vec::new()
        }
    }

    /// `packet` through PDCP and RLC on the DRB the TFTs pick, if any.
    fn transmit_ip(&mut self, packet: &[u8]) -> Option<UeAction> {
        // The DRB of the bearer the TFTs pick (TS 23.401 §5.4.1, #283). With
        // a default bearer, no pick means the packet is discarded; so does a
        // pick whose DRB is not set up -- another bearer's DRB would carry it
        // against its TFT. Without one (a NAS that keeps no bearers), the
        // lowest DRB.
        let ebi = self.nas.uplink_bearer_for(packet);
        let has_default = self.nas.default_bearer_ebi().is_some();
        let State::Connected { cell, user, .. } = &mut self.state else {
            return None;
        };
        let lcid = match ebi {
            Some(e) => user.iter().find(|(_, d)| d.ebi == e).map(|(l, _)| *l),
            None if has_default => None,
            None => user.keys().min().copied(),
        };
        let Some(lcid) = lcid else {
            log::debug!("uplink packet discarded: no DRB for EPS bearer {ebi:?}");
            return None;
        };
        let d = user.get_mut(&lcid).expect("found above");
        let pdcp = d.pdcp.transmit(packet).0;
        Some(UeAction::Data {
            cell: *cell,
            lcid,
            pdu: d.tx.transmit(&pdcp),
        })
    }

    fn send_ul_nas(&mut self, cell: u64, nas_msgs: Vec<Vec<u8>>) -> Vec<UeAction> {
        let State::Connected { srb1, srb2, .. } = &mut self.state else {
            return vec![];
        };
        // TS 36.331 §4.2.2: NAS goes on SRB2 once it is established.
        let (lcid, srb): (u8, &mut LteSrbPdcp) = match srb2 {
            Some(s) => (LCID_SRB2, s),
            None => (LCID_SRB1, srb1),
        };
        nas_msgs
            .into_iter()
            .filter_map(|nas| match msg::encode_ul_information_transfer(&nas) {
                Ok(rrc) => Some(UeAction::Rrc {
                    cell,
                    channel: RrcChannel::UlDcch,
                    lcid,
                    pdu: srb.transmit(&rrc).0,
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
