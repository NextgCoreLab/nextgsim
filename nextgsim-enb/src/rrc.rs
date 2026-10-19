//! The eNB's E-UTRA RRC: system information and UE contexts from RRC
//! connection establishment (TS 36.331 §5.3.3) through NAS relay (§5.6.1,
//! §5.6.2) to release (§5.3.8), tied to UE-associated S1AP (TS 36.413 §8.6,
//! §8.3).
//!
//! Pure: radio and S1 inputs in, [`Action`]s out. [`crate::radio`] owns the
//! RLS socket and the S1-MME association and executes the actions.
//!
//! # Signalling radio bearers over RLS
//!
//! RLS replaces the physical layer and MAC (as on the NR side). The logical
//! channels map onto RLS RRC channels: SRB0 is CCCH and carries bare RRC
//! messages (TM RLC, no PDCP, TS 36.331 §4.2.2); SRB1 is DCCH and carries
//! PDCP Data PDUs (TS 36.323 §6.2.2, via [`LteSrbPdcp`]).

use std::collections::{HashMap, VecDeque};

use nextgsim_common::Plmn;
use nextgsim_lte_rrc::msg::{self, CellInfo, InitialUeIdentity, MibInfo, PlmnId, Sib2Info};
use nextgsim_pdcp::LteSrbPdcp;
use nextgsim_rls::RrcChannel;
use nextgsim_s1ap::codec::{self as s1codec, CauseRadioNetwork, S1AP_PDU};

use crate::config::EnbConfig;
use crate::s1_task::NON_UE_ASSOCIATED_STREAM;
use crate::s1ap_ue::{self, ReleaseIds, S1Inbound, UeLocation, MAX_ENB_UE_S1AP_ID};

/// How many UE contexts the cell admits before answering RRCConnectionRequest
/// with RRCConnectionReject (§5.3.3.8).
pub const MAX_UE_CONTEXTS: usize = 1024;
/// `waitTime` sent in RRCConnectionReject, seconds (§6.2.2: 1..16).
pub const REJECT_WAIT_TIME_S: u8 = 10;
/// UL NAS messages buffered for a UE whose MME UE S1AP ID is not yet known.
const MAX_PENDING_UL_NAS: usize = 8;

/// Something the radio task must do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// Send an RRC PDU to one UE on `channel`.
    Rrc {
        /// RLS UE id.
        ue: u32,
        /// Logical channel.
        channel: RrcChannel,
        /// The PDU (an RRC message on CCCH, a PDCP PDU on DCCH).
        pdu: Vec<u8>,
    },
    /// Send an S1AP PDU to the MME on `stream`.
    S1 {
        /// SCTP stream.
        stream: u16,
        /// Encoded S1AP PDU.
        pdu: Vec<u8>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum UeState {
    /// RRCConnectionSetup sent; waiting for RRCConnectionSetupComplete with
    /// this transaction id.
    AwaitingSetupComplete { transaction_id: u8 },
    /// RRC connected; the UE-associated S1 connection exists once the first
    /// DOWNLINK NAS TRANSPORT supplies the MME UE S1AP ID.
    Connected,
}

#[derive(Debug, Clone)]
struct UeContext {
    enb_ue_s1ap_id: u32,
    mme_ue_s1ap_id: Option<u32>,
    state: UeState,
    srb1: LteSrbPdcp,
    identity: InitialUeIdentity,
    cause: msg::EstablishmentCauseValue,
    location: Option<UeLocation>,
    next_transaction_id: u8,
    pending_ul_nas: VecDeque<Vec<u8>>,
}

impl UeContext {
    fn next_tid(&mut self) -> u8 {
        let t = self.next_transaction_id;
        self.next_transaction_id = (t + 1) & 3;
        t
    }
}

/// The eNB's RRC entity for its one cell.
#[derive(Debug)]
pub struct EnbRrc {
    mib: MibInfo,
    cell: CellInfo,
    sib2: Sib2Info,
    /// The cell's PLMNs in SIB1 order, for mapping `selectedPLMN-Identity`.
    plmns: Vec<Plmn>,
    ues: HashMap<u32, UeContext>,
    by_enb_id: HashMap<u32, u32>,
    next_enb_ue_s1ap_id: u32,
    outbound_streams: u16,
}

fn to_rrc_plmn(p: &Plmn) -> PlmnId {
    PlmnId {
        mcc: p.mcc,
        mnc: p.mnc,
        long_mnc: p.long_mnc,
    }
}

fn encode_s1(pdu: &S1AP_PDU) -> Option<Vec<u8>> {
    match s1codec::encode_s1ap_pdu(pdu) {
        Ok(b) => Some(b),
        Err(e) => {
            log::error!("cannot encode S1AP PDU: {e}");
            None
        }
    }
}

impl EnbRrc {
    /// Build the cell from a validated config. `outbound_streams` is the
    /// S1-MME association's outbound stream count (TS 36.412 §7).
    pub fn new(cfg: &EnbConfig, outbound_streams: u16) -> Self {
        let plmns = cfg
            .effective_supported_tas()
            .into_iter()
            .find(|ta| ta.tac == cfg.tac)
            .map_or_else(|| vec![cfg.plmn], |ta| ta.broadcast_plmns);
        let r = &cfg.radio;
        let dl_bandwidth = match r.dl_bandwidth {
            crate::config::DlBandwidthRb::N6 => msg::DlBandwidth::N6,
            crate::config::DlBandwidthRb::N15 => msg::DlBandwidth::N15,
            crate::config::DlBandwidthRb::N25 => msg::DlBandwidth::N25,
            crate::config::DlBandwidthRb::N50 => msg::DlBandwidth::N50,
            crate::config::DlBandwidthRb::N75 => msg::DlBandwidth::N75,
            crate::config::DlBandwidthRb::N100 => msg::DlBandwidth::N100,
        };
        let paging = match cfg.paging_drx {
            crate::config::PagingDrx::V32 => msg::PagingCycle::Rf32,
            crate::config::PagingDrx::V64 => msg::PagingCycle::Rf64,
            crate::config::PagingDrx::V128 => msg::PagingCycle::Rf128,
            crate::config::PagingDrx::V256 => msg::PagingCycle::Rf256,
        };
        Self {
            mib: MibInfo {
                dl_bandwidth,
                sfn: 0,
            },
            cell: CellInfo {
                plmns: plmns.iter().map(to_rrc_plmn).collect(),
                tac: cfg.tac,
                cell_identity: cfg.cell_identity(),
                cell_barred: false,
                q_rx_lev_min_dbm: r.q_rx_lev_min_dbm,
                freq_band: r.freq_band,
                system_info_value_tag: 0,
            },
            sib2: Sib2Info {
                timers: msg::UeTimers {
                    t300_ms: r.t300_ms,
                    ..msg::UeTimers::default()
                },
                // TS 36.413 §9.2.1.16: the Default Paging DRX sent in S1 Setup
                // is the cell's broadcast default paging cycle.
                default_paging_cycle: paging,
            },
            plmns,
            ues: HashMap::new(),
            by_enb_id: HashMap::new(),
            next_enb_ue_s1ap_id: 1,
            outbound_streams,
        }
    }

    /// Number of UE contexts.
    pub fn ue_count(&self) -> usize {
        self.ues.len()
    }

    /// The eNB UE S1AP ID of the RLS UE `ue`, if it has a context.
    pub fn enb_ue_s1ap_id(&self, ue: u32) -> Option<u32> {
        self.ues.get(&ue).map(|c| c.enb_ue_s1ap_id)
    }

    /// The broadcast PDUs: MIB on BCCH-BCH, SIB1 and the SIB2
    /// SystemInformation on BCCH-DL-SCH (§5.2.1.2). `sfn` is the current
    /// system frame number.
    pub fn system_information(&mut self, sfn: u16) -> Vec<(RrcChannel, Vec<u8>)> {
        self.mib.sfn = sfn & 0x3FF;
        let mut out = Vec::with_capacity(3);
        match msg::encode_mib(&self.mib) {
            Ok(b) => out.push((RrcChannel::BcchBch, b)),
            Err(e) => log::error!("cannot encode MIB: {e}"),
        }
        match msg::encode_sib1(&self.cell) {
            Ok(b) => out.push((RrcChannel::BcchDlSch, b)),
            Err(e) => log::error!("cannot encode SIB1: {e}"),
        }
        match msg::encode_sib2_si(&self.sib2) {
            Ok(b) => out.push((RrcChannel::BcchDlSch, b)),
            Err(e) => log::error!("cannot encode SIB2: {e}"),
        }
        out
    }

    /// TS 36.412 §7: one stream per UE-associated signalling, never stream 0
    /// (reserved for non UE-associated signalling) and never changed for the
    /// life of the UE's signalling. With a single outbound stream there is no
    /// stream left to reserve, so stream 0 is used and the deviation logged.
    fn ue_stream(&self, enb_ue_s1ap_id: u32) -> u16 {
        if self.outbound_streams < 2 {
            return NON_UE_ASSOCIATED_STREAM;
        }
        let ue_streams = u32::from(self.outbound_streams - 1);
        1 + (enb_ue_s1ap_id % ue_streams) as u16
    }

    fn allocate_enb_ue_s1ap_id(&mut self) -> Option<u32> {
        // §9.2.3.4: unique within the eNB. Allocated sequentially, skipping
        // IDs in use, wrapping at 2^24.
        for _ in 0..=MAX_ENB_UE_S1AP_ID {
            let id = self.next_enb_ue_s1ap_id;
            self.next_enb_ue_s1ap_id = if id >= MAX_ENB_UE_S1AP_ID { 0 } else { id + 1 };
            if !self.by_enb_id.contains_key(&id) {
                return Some(id);
            }
        }
        None
    }

    fn remove(&mut self, ue: u32) -> Option<UeContext> {
        let ctx = self.ues.remove(&ue)?;
        self.by_enb_id.remove(&ctx.enb_ue_s1ap_id);
        Some(ctx)
    }

    /// A PDU from `ue` on UL-CCCH (SRB0): RRCConnectionRequest (§5.3.3.3).
    pub fn on_ul_ccch(&mut self, ue: u32, pdu: &[u8]) -> Vec<Action> {
        let req = match msg::decode_ul_ccch(pdu) {
            Ok(r) => r,
            Err(e) => {
                log::warn!("UE {ue}: undecodable UL-CCCH message: {e}");
                return vec![];
            }
        };
        let msg::UlCcch::ConnectionRequest { identity, cause } = req else {
            log::info!("UE {ue}: UL-CCCH message not supported in this build, ignored");
            return vec![];
        };
        // A new request from a UE that already has a context means the UE
        // restarted its RRC connection; the old context cannot be resumed.
        if let Some(old) = self.remove(ue) {
            log::info!(
                "UE {ue}: new RRCConnectionRequest; dropping eNB UE S1AP ID {}",
                old.enb_ue_s1ap_id
            );
            let mut acts = Vec::new();
            if let Some(mme) = old.mme_ue_s1ap_id {
                self.push_release_request(&mut acts, mme, old.enb_ue_s1ap_id);
            }
            return acts
                .into_iter()
                .chain(self.admit(ue, identity, cause))
                .collect();
        }
        self.admit(ue, identity, cause)
    }

    fn admit(
        &mut self,
        ue: u32,
        identity: InitialUeIdentity,
        cause: msg::EstablishmentCauseValue,
    ) -> Vec<Action> {
        let id = (self.ues.len() < MAX_UE_CONTEXTS)
            .then(|| self.allocate_enb_ue_s1ap_id())
            .flatten();
        let Some(enb_ue_s1ap_id) = id else {
            log::warn!(
                "UE {ue}: no capacity; RRCConnectionReject (waitTime {REJECT_WAIT_TIME_S}s)"
            );
            return match msg::encode_rrc_connection_reject(REJECT_WAIT_TIME_S) {
                Ok(pdu) => vec![Action::Rrc {
                    ue,
                    channel: RrcChannel::DlCcch,
                    pdu,
                }],
                Err(e) => {
                    log::error!("cannot encode RRCConnectionReject: {e}");
                    vec![]
                }
            };
        };
        let mut ctx = UeContext {
            enb_ue_s1ap_id,
            mme_ue_s1ap_id: None,
            state: UeState::Connected,
            // SRB1 is established when RRCConnectionSetup is sent (§5.3.3.4).
            srb1: LteSrbPdcp::new(),
            identity,
            cause,
            location: None,
            next_transaction_id: 0,
            pending_ul_nas: VecDeque::new(),
        };
        let transaction_id = ctx.next_tid();
        ctx.state = UeState::AwaitingSetupComplete { transaction_id };
        let pdu = match msg::encode_rrc_connection_setup(transaction_id) {
            Ok(p) => p,
            Err(e) => {
                log::error!("cannot encode RRCConnectionSetup: {e}");
                return vec![];
            }
        };
        log::info!(
            "UE {ue}: RRCConnectionRequest ({cause:?}); RRCConnectionSetup, eNB UE S1AP ID {enb_ue_s1ap_id}"
        );
        self.by_enb_id.insert(enb_ue_s1ap_id, ue);
        self.ues.insert(ue, ctx);
        vec![Action::Rrc {
            ue,
            channel: RrcChannel::DlCcch,
            pdu,
        }]
    }

    /// A PDCP PDU from `ue` on UL-DCCH (SRB1).
    pub fn on_ul_dcch(&mut self, ue: u32, pdcp_pdu: &[u8]) -> Vec<Action> {
        let Some(ctx) = self.ues.get_mut(&ue) else {
            log::warn!("UE {ue}: UL-DCCH PDU with no RRC connection, discarded");
            return vec![];
        };
        let sdu = match ctx.srb1.receive(pdcp_pdu) {
            Ok((sdu, _count)) => sdu,
            Err(e) => {
                log::warn!("UE {ue}: SRB1 PDCP: {e}");
                return vec![];
            }
        };
        let m = match msg::decode_ul_dcch(&sdu) {
            Ok(m) => m,
            Err(e) => {
                log::warn!("UE {ue}: undecodable UL-DCCH message: {e}");
                return vec![];
            }
        };
        match m {
            msg::UlDcch::ConnectionSetupComplete {
                transaction_id,
                selected_plmn_index,
                nas,
            } => self.on_setup_complete(ue, transaction_id, selected_plmn_index, nas),
            msg::UlDcch::InformationTransfer { nas } => self.on_ul_nas(ue, nas),
            msg::UlDcch::Unsupported => {
                log::info!("UE {ue}: UL-DCCH message not supported in this build, ignored");
                vec![]
            }
        }
    }

    fn on_setup_complete(
        &mut self,
        ue: u32,
        transaction_id: u8,
        selected_plmn_index: u8,
        nas: Vec<u8>,
    ) -> Vec<Action> {
        let plmn = usize::from(selected_plmn_index)
            .checked_sub(1)
            .and_then(|i| self.plmns.get(i))
            .copied();
        let location = plmn.map(|plmn| UeLocation {
            plmn,
            tac: self.cell.tac,
            cell_identity: self.cell.cell_identity,
        });
        let Some(ctx) = self.ues.get_mut(&ue) else {
            return vec![];
        };
        if ctx.state != (UeState::AwaitingSetupComplete { transaction_id }) {
            log::warn!(
                "UE {ue}: RRCConnectionSetupComplete (transaction {transaction_id}) not expected in {:?}; ignored",
                ctx.state
            );
            return vec![];
        }
        let Some(location) = location else {
            log::warn!(
                "UE {ue}: selectedPLMN-Identity {selected_plmn_index} is not in SIB1's list of {}",
                self.plmns.len()
            );
            return vec![];
        };
        ctx.state = UeState::Connected;
        ctx.location = Some(location);
        let s_tmsi = match ctx.identity {
            InitialUeIdentity::STmsi { mmec, m_tmsi } => Some((mmec, m_tmsi)),
            InitialUeIdentity::Random(_) => None,
        };
        let enb_id = ctx.enb_ue_s1ap_id;
        let cause = ctx.cause.index();
        let stream = self.ue_stream(enb_id);
        log::info!(
            "UE {ue}: RRCConnectionSetupComplete; INITIAL UE MESSAGE ({} octets of NAS) on stream {stream}",
            nas.len()
        );
        encode_s1(&s1ap_ue::build_initial_ue_message(
            enb_id, &nas, &location, cause, s_tmsi,
        ))
        .map(|pdu| vec![Action::S1 { stream, pdu }])
        .unwrap_or_default()
    }

    fn on_ul_nas(&mut self, ue: u32, nas: Vec<u8>) -> Vec<Action> {
        let Some(ctx) = self.ues.get_mut(&ue) else {
            return vec![];
        };
        let (Some(mme), Some(location)) = (ctx.mme_ue_s1ap_id, ctx.location) else {
            // UPLINK NAS TRANSPORT needs the MME UE S1AP ID (§9.1.7.3), which
            // only the first downlink message supplies (§8.6.2.2).
            if ctx.pending_ul_nas.len() < MAX_PENDING_UL_NAS {
                ctx.pending_ul_nas.push_back(nas);
            } else {
                log::warn!("UE {ue}: UL NAS buffer full before MME UE S1AP ID; dropped");
            }
            return vec![];
        };
        let enb = ctx.enb_ue_s1ap_id;
        let stream = self.ue_stream(enb);
        encode_s1(&s1ap_ue::build_uplink_nas_transport(
            mme, enb, &nas, &location,
        ))
        .map(|pdu| vec![Action::S1 { stream, pdu }])
        .unwrap_or_default()
    }

    fn push_release_request(&self, acts: &mut Vec<Action>, mme: u32, enb: u32) {
        if let Some(pdu) = encode_s1(&s1ap_ue::build_ue_context_release_request(
            mme,
            enb,
            CauseRadioNetwork::RADIO_CONNECTION_WITH_UE_LOST,
        )) {
            acts.push(Action::S1 {
                stream: self.ue_stream(enb),
                pdu,
            });
        }
    }

    /// The RLS link to `ue` is gone (heartbeats stopped). The eNB releases the
    /// RRC context locally and, when the UE has an S1 connection, asks the
    /// MME to release it (TS 36.413 §8.3.2, cause "Radio Connection With UE
    /// Lost").
    pub fn on_ue_lost(&mut self, ue: u32) -> Vec<Action> {
        let Some(ctx) = self.remove(ue) else {
            return vec![];
        };
        log::info!(
            "UE {ue}: radio link lost (eNB UE S1AP ID {})",
            ctx.enb_ue_s1ap_id
        );
        let mut acts = Vec::new();
        if let Some(mme) = ctx.mme_ue_s1ap_id {
            self.push_release_request(&mut acts, mme, ctx.enb_ue_s1ap_id);
        }
        acts
    }

    /// An S1AP PDU from the MME on an operational S1 interface.
    pub fn on_s1(&mut self, bytes: &[u8]) -> Vec<Action> {
        match s1ap_ue::handle_s1_pdu(bytes) {
            S1Inbound::DownlinkNas {
                mme_ue_s1ap_id,
                enb_ue_s1ap_id,
                nas,
                error_indication,
            } => {
                let mut acts = Vec::new();
                if let Some(ei) = error_indication {
                    acts.push(Action::S1 {
                        stream: self.ue_stream(enb_ue_s1ap_id),
                        pdu: ei,
                    });
                }
                acts.extend(self.on_downlink_nas(mme_ue_s1ap_id, enb_ue_s1ap_id, nas));
                acts
            }
            S1Inbound::ReleaseCommand { ids } => self.on_release_command(ids),
            S1Inbound::ErrorIndication(pdu) => vec![Action::S1 {
                stream: NON_UE_ASSOCIATED_STREAM,
                pdu,
            }],
            S1Inbound::NotHandled(what) => {
                log::info!("S1AP {what}: not handled in this build");
                vec![]
            }
        }
    }

    fn on_downlink_nas(&mut self, mme: u32, enb: u32, nas: Vec<u8>) -> Vec<Action> {
        let Some(&ue) = self.by_enb_id.get(&enb) else {
            log::warn!("DOWNLINK NAS TRANSPORT for unknown eNB UE S1AP ID {enb}");
            return s1ap_ue::unknown_enb_ue_s1ap_id_error(mme, enb)
                .map(|pdu| {
                    vec![Action::S1 {
                        stream: NON_UE_ASSOCIATED_STREAM,
                        pdu,
                    }]
                })
                .unwrap_or_default();
        };
        let stream = self.ue_stream(enb);
        let ctx = self.ues.get_mut(&ue).expect("by_enb_id and ues agree");
        match ctx.mme_ue_s1ap_id {
            // §8.6.2.2: the first DOWNLINK NAS TRANSPORT establishes the
            // UE-associated logical S1 connection; the eNB stores the ID.
            None => ctx.mme_ue_s1ap_id = Some(mme),
            Some(stored) if stored != mme => {
                // §10.6: inconsistent UE S1AP IDs.
                log::warn!("UE {ue}: MME UE S1AP ID {mme} differs from stored {stored}");
                return s1ap_ue::unknown_enb_ue_s1ap_id_error(mme, enb)
                    .map(|pdu| vec![Action::S1 { stream, pdu }])
                    .unwrap_or_default();
            }
            Some(_) => {}
        }
        let tid = ctx.next_tid();
        let mut acts = Vec::new();
        match msg::encode_dl_information_transfer(tid, &nas) {
            Ok(rrc) => {
                let (pdcp, _) = ctx.srb1.transmit(&rrc);
                acts.push(Action::Rrc {
                    ue,
                    channel: RrcChannel::DlDcch,
                    pdu: pdcp,
                });
            }
            Err(e) => log::error!("cannot encode DLInformationTransfer: {e}"),
        }
        // UL NAS that arrived before the MME UE S1AP ID was known.
        let pending: Vec<_> = ctx.pending_ul_nas.drain(..).collect();
        for nas in pending {
            acts.extend(self.on_ul_nas(ue, nas));
        }
        acts
    }

    fn on_release_command(&mut self, ids: ReleaseIds) -> Vec<Action> {
        let found = match ids {
            ReleaseIds::Pair { mme: _, enb } => self.by_enb_id.get(&enb).copied(),
            ReleaseIds::Mme(mme) => self
                .ues
                .iter()
                .find(|(_, c)| c.mme_ue_s1ap_id == Some(mme))
                .map(|(ue, _)| *ue),
        };
        let (mme, enb) = match ids {
            ReleaseIds::Pair { mme, enb } => (mme, Some(enb)),
            ReleaseIds::Mme(mme) => (mme, None),
        };
        let Some(ue) = found else {
            // §8.3.3.2: the eNB releases what it has and replies; with no
            // context there is nothing to release, so it confirms at once.
            log::info!("UE CONTEXT RELEASE COMMAND for a UE with no context; completing");
            return enb
                .and_then(|enb| encode_s1(&s1ap_ue::build_ue_context_release_complete(mme, enb)))
                .map(|pdu| {
                    vec![Action::S1 {
                        stream: NON_UE_ASSOCIATED_STREAM,
                        pdu,
                    }]
                })
                .unwrap_or_default();
        };
        let mut ctx = self.remove(ue).expect("found above");
        let stream = self.ue_stream(ctx.enb_ue_s1ap_id);
        let mut acts = Vec::new();
        // TS 36.331 §5.3.8: release the RRC connection.
        match msg::encode_rrc_connection_release(ctx.next_tid()) {
            Ok(rrc) => {
                let (pdcp, _) = ctx.srb1.transmit(&rrc);
                acts.push(Action::Rrc {
                    ue,
                    channel: RrcChannel::DlDcch,
                    pdu: pdcp,
                });
            }
            Err(e) => log::error!("cannot encode RRCConnectionRelease: {e}"),
        }
        log::info!(
            "UE {ue}: UE CONTEXT RELEASE COMMAND; RRCConnectionRelease, UE CONTEXT RELEASE COMPLETE"
        );
        if let Some(pdu) = encode_s1(&s1ap_ue::build_ue_context_release_complete(
            mme,
            ctx.enb_ue_s1ap_id,
        )) {
            acts.push(Action::S1 { stream, pdu });
        }
        acts
    }
}

#[cfg(test)]
mod tests;
