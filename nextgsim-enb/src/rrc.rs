//! The eNB's E-UTRA RRC: system information and UE contexts from RRC
//! connection establishment (TS 36.331 §5.3.3) through NAS relay (§5.6.1,
//! §5.6.2), initial security activation (§5.3.4) and the RRC connection
//! reconfiguration of Initial Context Setup (§5.3.5; TS 36.413 §8.3.1) to
//! release (§5.3.8), tied to UE-associated S1AP (TS 36.413 §8.6, §8.3).
//!
//! # Initial Context Setup (TS 36.413 §8.3.1.2; TS 23.401 §5.3.2.1 steps 17-21)
//!
//! INITIAL CONTEXT SETUP REQUEST brings `K_eNB` and the UE's security
//! capabilities. The eNB picks the highest-priority EIA and EEA of its own
//! lists that the UE supports (TS 33.401 §7.2.4.2.1), derives `K_RRCint`,
//! `K_RRCenc` and `K_UPenc` (Annex A.7), and runs the AS Security Mode Command
//! with SRB1's PDCP switches set in the order of TS 33.401 §7.2.4.5. Once the
//! UE completes it, an RRCConnectionReconfiguration sets up one DRB per E-RAB
//! and carries the E-RABs' NAS-PDUs (the ATTACH ACCEPT); its completion is
//! answered with INITIAL CONTEXT SETUP RESPONSE naming the eNB's S1-U
//! endpoints, and hands each E-RAB's DRB and S1-U tunnel to the user plane
//! ([`crate::userplane`]) as an [`Action::UserPlane`].
//!
//! The same reconfiguration adds **SRB2** (TS 36.331 §4.2.2: "configured by
//! E-UTRAN after security activation"), with SRB1's AS security; NAS moves
//! onto it once the UE has completed the reconfiguration. RLS tells SRB1,
//! SRB2 and the DRBs apart by LCID ([`nextgsim_rls::lte`]).
//!
//! **UE Capability Enquiry** is not run: over RLS no
//! configuration this eNB sends depends on a UE capability, and forwarding one
//! to the MME (UE CAPABILITY INFO INDICATION) needs the
//! `EUTRA-InterNodeDefinitions` module the RRC codec does not generate yet.
//!
//! Pure: radio and S1 inputs in, [`Action`]s out. [`crate::radio`] owns the
//! RLS socket and the S1-MME association and executes the actions.
//!
//! # Signalling radio bearers over RLS
//!
//! RLS replaces the physical layer and MAC (as on the NR side). The logical
//! channels map onto RLS RRC channels: SRB0 is CCCH and carries bare RRC
//! messages (TM RLC, no PDCP, TS 36.331 §4.2.2); SRB1 and SRB2 are DCCH, LCID
//! 1 and 2, and carry PDCP Data PDUs (TS 36.323 §6.2.2, via [`LteSrbPdcp`]).

use std::collections::{HashMap, VecDeque};
use std::net::IpAddr;
use std::time::{Duration, Instant};

use nextgsim_common::Plmn;
use nextgsim_crypto::eps_kdf::{derive_algorithm_key, EpsAlgorithmType};
use nextgsim_lte_rrc::msg::{
    self, CellInfo, DrbConfig, DrbRlc, InitialUeIdentity, MibInfo, PlmnId, Sib2Info,
};
use nextgsim_pdcp::lte_srb::{DIRECTION_DOWNLINK, SRB1_BEARER};

/// BEARER of SRB2: RB identity 2 − 1 (TS 36.323 §5.6.0).
const SRB2_BEARER: u8 = 1;
use nextgsim_pdcp::LteSrbPdcp;
use nextgsim_rls::lte::{LCID_SRB1, LCID_SRB2};
use nextgsim_rls::RrcChannel;
use nextgsim_s1ap::codec::{self as s1codec, Cause, CauseRadioNetwork, CauseTransport, S1AP_PDU};

use crate::config::{AsSecurityConfig, EnbConfig};
use crate::s1_task::NON_UE_ASSOCIATED_STREAM;
use crate::s1ap::DiagnosedIe;
use crate::s1ap_ue::{
    self, ErabSetup, ErabToSetup, InitialContextSetup, ReleaseIds, S1Inbound, UeLocation,
    MAX_ENB_UE_S1AP_ID,
};
use crate::userplane::{sgw_endpoint, UpBearer};

/// DRBs one RRCConnectionReconfiguration can add here: logical channel
/// identities 3..=10 (TS 36.331 §6.3.2 `DRB-ToAddMod`).
pub const MAX_DRBS: usize = 8;

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
        /// LCID on DCCH (1 SRB1, 2 SRB2), 0 otherwise ([`nextgsim_rls::lte`]).
        lcid: u8,
        /// The PDU (an RRC message on CCCH, a PDCP PDU on DCCH).
        pdu: Vec<u8>,
    },
    /// Establish the user plane of `ue`'s E-RABs (their DRBs and S1-U
    /// tunnels): the UE has completed the RRCConnectionReconfiguration that
    /// set the DRBs up (TS 36.331 §5.3.5.3).
    UserPlane {
        /// RLS UE id.
        ue: u32,
        /// The E-RABs.
        bearers: Vec<UpBearer>,
    },
    /// Release the DRBs on `lcids` of `ue` and their S1-U tunnels (E-RAB
    /// RELEASE, TS 36.413 §8.2.3; #283).
    ReleaseUserPlane {
        /// RLS UE id.
        ue: u32,
        /// The DRBs' LCIDs.
        lcids: Vec<u8>,
    },
    /// Broadcast an RRC PDU on a common channel to every UE the cell hears
    /// (PCCH paging, TS 36.331 §5.3.2). RLS has no paging occasions, so it goes
    /// out at once.
    Broadcast {
        /// Logical channel.
        channel: RrcChannel,
        /// The RRC message.
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

/// An Initial Context Setup in progress (TS 36.413 §8.3.1.2).
#[derive(Clone)]
struct PendingIcs {
    erabs: Vec<ErabToSetup>,
    notify: Vec<DiagnosedIe>,
    /// The ciphering algorithm and `K_RRCenc`, for switching on uplink
    /// deciphering once SecurityModeComplete verifies (TS 33.401 §7.2.4.5).
    rx_ciphering: (u8, [u8; 16]),
    smc_transaction_id: u8,
    reconfiguration_transaction_id: Option<u8>,
    teids: Vec<(u8, u32)>,
}

impl std::fmt::Debug for PendingIcs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingIcs")
            .field("erabs", &self.erabs)
            .field("smc_transaction_id", &self.smc_transaction_id)
            .field(
                "reconfiguration_transaction_id",
                &self.reconfiguration_transaction_id,
            )
            .finish_non_exhaustive()
    }
}

/// An E-RAB management request waiting for, or running, its
/// RRCConnectionReconfiguration (TS 36.413 §8.2; #283).
#[derive(Debug, Clone)]
enum ErabRequest {
    Setup(s1ap_ue::ErabSetupRequest),
    Modify(s1ap_ue::ErabModifyRequest),
    Release(s1ap_ue::ErabReleaseCommand),
}

/// What the running E-RAB procedure answers once the UE has completed its
/// RRCConnectionReconfiguration.
#[derive(Debug, Clone)]
enum ErabOutcome {
    /// E-RABs set up: item, eNB TEID, DRB.
    Setup(Vec<(ErabToSetup, u32, DrbConfig)>),
    /// E-RABs modified, with their new QCIs.
    Modify(Vec<(u8, u8)>),
    /// E-RABs released, with their DRBs' LCIDs.
    Release(Vec<(u8, u8)>),
}

#[derive(Debug, Clone)]
struct PendingErab {
    mme: u32,
    enb: u32,
    transaction_id: u8,
    outcome: ErabOutcome,
    failed: Vec<(u8, Cause)>,
}

/// AS security of a UE (TS 33.401 §7.2.4). Keys never printed.
#[derive(Clone)]
struct AsSecurity {
    eea: u8,
    eia: u8,
    /// `K_RRCint` and `K_RRCenc`, for SRB2 (the same AS security as SRB1).
    k_rrc_int: [u8; 16],
    k_rrc_enc: [u8; 16],
    /// `K_UPenc`, for the DRBs' PDCP.
    k_up_enc: [u8; 16],
}

impl std::fmt::Debug for AsSecurity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EEA{}/EIA{} (keys redacted)", self.eea, self.eia)
    }
}

#[derive(Debug, Clone)]
struct UeContext {
    enb_ue_s1ap_id: u32,
    mme_ue_s1ap_id: Option<u32>,
    state: UeState,
    srb1: LteSrbPdcp,
    /// SRB2 (TS 36.331 §4.2.2), from the RRCConnectionReconfiguration of
    /// Initial Context Setup on; NAS moves onto it once the UE has completed
    /// that reconfiguration (`srb2_ready`).
    srb2: Option<LteSrbPdcp>,
    srb2_ready: bool,
    identity: InitialUeIdentity,
    cause: msg::EstablishmentCauseValue,
    location: Option<UeLocation>,
    next_transaction_id: u8,
    pending_ul_nas: VecDeque<Vec<u8>>,
    security: Option<AsSecurity>,
    ics: Option<PendingIcs>,
    drbs: Vec<DrbConfig>,
    /// The QCI of each E-RAB set up, by E-RAB ID (#283).
    erab_qci: HashMap<u8, u8>,
    /// The E-RAB procedure whose RRCConnectionReconfiguration is out (#283).
    erab: Option<PendingErab>,
    /// E-RAB procedures that arrived while another reconfiguration ran (#283).
    erab_queue: VecDeque<ErabRequest>,
    /// The last user data or signalling to or from the UE, for the
    /// user-inactivity release (#350).
    last_activity: Instant,
    /// When a UE CONTEXT RELEASE REQUEST (user-inactivity) went out (#350). It
    /// is not sent again while the MME's command is awaited, but a request is
    /// re-armed when activity follows it (the MME kept the UE connected) or when
    /// no command came within one more inactivity period (the request or the
    /// command was lost), so a UE is never left connected for good (#353 review).
    inactivity_release_requested_at: Option<Instant>,
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
    security: AsSecurityConfig,
    gtp_ip: IpAddr,
    next_teid: u32,
    /// UEs whose context was removed since [`EnbRrc::take_released`].
    released: Vec<u32>,
    /// `inactivity_timer_secs` (#350): `None` disables the release.
    inactivity: Option<Duration>,
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
            security: cfg.security.clone(),
            gtp_ip: cfg.gtp_ip.unwrap_or(cfg.s1ap_ip),
            next_teid: 1,
            released: Vec::new(),
            inactivity: cfg.inactivity_timer_secs.map(Duration::from_secs),
        }
    }

    /// User data went to or came from `ue` (#350): the inactivity timer
    /// restarts.
    pub fn on_user_plane_activity(&mut self, ue: u32, now: Instant) {
        if let Some(ctx) = self.ues.get_mut(&ue) {
            ctx.last_activity = now;
        }
    }

    /// The user-inactivity release (#350, TS 36.413 §8.3.2): every UE with an
    /// S1 connection that has been silent -- no user data, no signalling -- for
    /// the configured time gets one UE CONTEXT RELEASE REQUEST with cause
    /// user-inactivity. The release itself follows the MME's UE CONTEXT RELEASE
    /// COMMAND (§8.3.3), so the UE keeps its context until then.
    pub fn check_inactivity(&mut self, now: Instant) -> Vec<Action> {
        let Some(limit) = self.inactivity else {
            return vec![];
        };
        let due: Vec<(u32, u32, u32)> = self
            .ues
            .iter()
            .filter(|(_, c)| {
                c.state == UeState::Connected
                    && now.saturating_duration_since(c.last_activity) >= limit
                    && c.inactivity_release_requested_at.is_none_or(|at| {
                        c.last_activity > at || now.saturating_duration_since(at) >= limit
                    })
            })
            .filter_map(|(ue, c)| c.mme_ue_s1ap_id.map(|mme| (*ue, mme, c.enb_ue_s1ap_id)))
            .collect();
        let mut acts = Vec::new();
        for (ue, mme, enb) in due {
            log::info!(
                "UE {ue}: no activity for {}s; UE CONTEXT RELEASE REQUEST (user-inactivity)",
                limit.as_secs()
            );
            self.push_release_request_with(&mut acts, mme, enb, CauseRadioNetwork::USER_INACTIVITY);
            if let Some(c) = self.ues.get_mut(&ue) {
                c.inactivity_release_requested_at = Some(now);
            }
        }
        acts
    }

    /// The AS algorithms a UE context runs with, `(EEA, EIA)`, once its
    /// security is active.
    pub fn as_algorithms(&self, ue: u32) -> Option<(u8, u8)> {
        self.ues
            .get(&ue)
            .and_then(|c| c.security.as_ref())
            .map(|s| (s.eea, s.eia))
    }

    /// The DRBs configured for a UE.
    pub fn drbs(&self, ue: u32) -> Vec<DrbConfig> {
        self.ues
            .get(&ue)
            .map(|c| c.drbs.clone())
            .unwrap_or_default()
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
        self.released.push(ue);
        Some(ctx)
    }

    /// The UEs whose RRC context ended since the last call: their user plane
    /// must be released with it.
    pub fn take_released(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.released)
    }

    /// This eNB's S1-U address.
    pub fn gtp_ip(&self) -> IpAddr {
        self.gtp_ip
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
                    lcid: 0,
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
            // SRB1 is established when RRCConnectionSetup is sent (§5.3.3.4);
            // the eNB transmits downlink on it.
            srb1: LteSrbPdcp::with_bearer(SRB1_BEARER, DIRECTION_DOWNLINK),
            srb2: None,
            srb2_ready: false,
            identity,
            cause,
            location: None,
            next_transaction_id: 0,
            pending_ul_nas: VecDeque::new(),
            security: None,
            ics: None,
            drbs: Vec::new(),
            erab_qci: HashMap::new(),
            erab: None,
            erab_queue: VecDeque::new(),
            last_activity: Instant::now(),
            inactivity_release_requested_at: None,
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
            lcid: 0,
            pdu,
        }]
    }

    /// A PDCP PDU from `ue` on UL-DCCH, on SRB1 or SRB2 (`lcid`).
    pub fn on_ul_dcch(&mut self, ue: u32, lcid: u8, pdcp_pdu: &[u8]) -> Vec<Action> {
        let Some(ctx) = self.ues.get_mut(&ue) else {
            log::warn!("UE {ue}: UL-DCCH PDU with no RRC connection, discarded");
            return vec![];
        };
        // Signalling is activity too (#350): a UE mid-procedure is not idle.
        ctx.last_activity = Instant::now();
        if lcid == LCID_SRB2 {
            let Some(srb2) = ctx.srb2.as_mut() else {
                log::warn!("UE {ue}: PDU on SRB2 before SRB2 was established, discarded");
                return vec![];
            };
            return match srb2.receive(pdcp_pdu).map(|r| msg::decode_ul_dcch(&r.sdu)) {
                // TS 36.331 §4.2.2: SRB2 carries NAS (and logged
                // measurements, which this build does not request).
                Ok(Ok(msg::UlDcch::InformationTransfer { nas })) => self.on_ul_nas(ue, nas),
                Ok(Ok(other)) => {
                    log::warn!("UE {ue}: {other:?} on SRB2, which carries NAS only here");
                    vec![]
                }
                Ok(Err(e)) => {
                    log::warn!("UE {ue}: undecodable UL-DCCH message on SRB2: {e}");
                    vec![]
                }
                Err(e) => {
                    log::warn!("UE {ue}: SRB2 PDCP: {e}; discarded");
                    vec![]
                }
            };
        }
        let sdu = match ctx.srb1.receive(pdcp_pdu) {
            Ok(r) => r.sdu,
            Err(nextgsim_pdcp::LteSrbPdcpError::IntegrityCheckFailed(_))
                if ctx
                    .ics
                    .as_ref()
                    .is_some_and(|i| i.reconfiguration_transaction_id.is_none()) =>
            {
                // TS 33.401 §7.2.4.5: a UE whose checks of the AS SMC fail
                // replies with an UNPROTECTED SecurityModeFailure, so while the
                // command is outstanding that one message is accepted without a
                // MAC; anything else that fails verification is still dropped.
                let Ok(r) = ctx.srb1.clone().receive_unprotected(pdcp_pdu) else {
                    return vec![];
                };
                match msg::decode_ul_dcch(&r.sdu) {
                    Ok(msg::UlDcch::SecurityModeFailure { transaction_id }) => {
                        let _ = ctx.srb1.receive_unprotected(pdcp_pdu);
                        return self.on_security_mode_failure(ue, transaction_id);
                    }
                    _ => {
                        log::warn!("UE {ue}: SRB1 integrity verification failed; discarded");
                        return vec![];
                    }
                }
            }
            Err(e) => {
                // TS 36.323 §5.7: an integrity failure is indicated to RRC and
                // the PDU discarded.
                log::warn!("UE {ue}: SRB1 PDCP: {e}; discarded");
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
            msg::UlDcch::SecurityModeComplete { transaction_id } => {
                self.on_security_mode_complete(ue, transaction_id)
            }
            msg::UlDcch::SecurityModeFailure { transaction_id } => {
                self.on_security_mode_failure(ue, transaction_id)
            }
            msg::UlDcch::ReconfigurationComplete { transaction_id } => {
                self.on_reconfiguration_complete(ue, transaction_id)
            }
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
        self.push_release_request_with(
            acts,
            mme,
            enb,
            CauseRadioNetwork::RADIO_CONNECTION_WITH_UE_LOST,
        );
    }

    fn push_release_request_with(&self, acts: &mut Vec<Action>, mme: u32, enb: u32, cause: u8) {
        if let Some(pdu) = encode_s1(&s1ap_ue::build_ue_context_release_request(mme, enb, cause)) {
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
            S1Inbound::InitialContextSetup(ics) => self.on_initial_context_setup(*ics),
            S1Inbound::InitialContextSetupFailure {
                enb_ue_s1ap_id,
                pdu,
            } => vec![Action::S1 {
                stream: self.ue_stream(enb_ue_s1ap_id),
                pdu,
            }],
            S1Inbound::Paging(p) => self.on_paging(&p),
            S1Inbound::ErabSetup(r) => self.on_erab_request(ErabRequest::Setup(*r)),
            S1Inbound::ErabModify(r) => self.on_erab_request(ErabRequest::Modify(*r)),
            S1Inbound::ErabRelease(r) => self.on_erab_request(ErabRequest::Release(*r)),
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

    /// S1AP PAGING (TS 36.413 §8.5.2): when one of the listed TAIs is this
    /// cell's (one of its broadcast PLMNs and its TAC), broadcast an RRC Paging
    /// with the UE's paging record (TS 36.331 §5.3.2.2). A PAGING for other TAs
    /// is not this cell's to send.
    fn on_paging(&mut self, p: &s1ap_ue::PagingRequest) -> Vec<Action> {
        let ours = p
            .tais
            .iter()
            .any(|(plmn, tac)| *tac == self.cell.tac && self.plmns.contains(plmn));
        if !ours {
            log::info!(
                "PAGING for TAIs {:?}: not this cell's (TAC {})",
                p.tais,
                self.cell.tac
            );
            return vec![];
        }
        let identity = match &p.identity {
            s1ap_ue::PagingIdentity::STmsi { mmec, m_tmsi } => msg::PagedUeIdentity::STmsi {
                mmec: *mmec,
                m_tmsi: *m_tmsi,
            },
            s1ap_ue::PagingIdentity::Imsi(d) => msg::PagedUeIdentity::Imsi(d.clone()),
        };
        let record = msg::PagingRecordInfo {
            identity,
            cn_domain: if p.cs_domain {
                msg::PagingCnDomain::Cs
            } else {
                msg::PagingCnDomain::Ps
            },
        };
        match msg::encode_paging(std::slice::from_ref(&record)) {
            Ok(pdu) => {
                log::info!(
                    "PAGING ({:?}, {}): RRC Paging on PCCH",
                    record.identity,
                    if p.cs_domain { "CS" } else { "PS" }
                );
                vec![Action::Broadcast {
                    channel: RrcChannel::Pcch,
                    pdu,
                }]
            }
            Err(e) => {
                log::warn!("PAGING not broadcast: {e}");
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
        if let Some(c) = self.ues.get_mut(&ue) {
            c.last_activity = Instant::now();
        }
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
                // TS 36.331 §4.2.2: NAS goes on SRB2 once it is established;
                // before that (attach, before Initial Context Setup) on SRB1.
                let (lcid, pdcp) = match (&mut ctx.srb2, ctx.srb2_ready) {
                    (Some(srb2), true) => (LCID_SRB2, srb2.transmit(&rrc).0),
                    _ => (LCID_SRB1, ctx.srb1.transmit(&rrc).0),
                };
                acts.push(Action::Rrc {
                    ue,
                    channel: RrcChannel::DlDcch,
                    lcid,
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

    fn ics_failure(&self, mme: u32, enb: u32, cause: Cause) -> Vec<Action> {
        encode_s1(&s1ap_ue::build_initial_context_setup_failure(
            mme,
            enb,
            cause,
            &[],
        ))
        .map(|pdu| {
            vec![Action::S1 {
                stream: self.ue_stream(enb),
                pdu,
            }]
        })
        .unwrap_or_default()
    }

    /// INITIAL CONTEXT SETUP REQUEST (TS 36.413 §8.3.1.2): take the security
    /// context into use with the AS Security Mode Command (TS 36.331 §5.3.4).
    fn on_initial_context_setup(&mut self, ics: InitialContextSetup) -> Vec<Action> {
        let (mme, enb) = (ics.mme_ue_s1ap_id, ics.enb_ue_s1ap_id);
        let Some(&ue) = self.by_enb_id.get(&enb) else {
            log::warn!("INITIAL CONTEXT SETUP REQUEST for unknown eNB UE S1AP ID {enb}");
            return s1ap_ue::unknown_enb_ue_s1ap_id_error(mme, enb)
                .map(|pdu| {
                    vec![Action::S1 {
                        stream: NON_UE_ASSOCIATED_STREAM,
                        pdu,
                    }]
                })
                .unwrap_or_default();
        };
        let ctx = self.ues.get(&ue).expect("by_enb_id and ues agree");
        if ctx.mme_ue_s1ap_id.is_some_and(|m| m != mme) {
            log::warn!("UE {ue}: INITIAL CONTEXT SETUP for MME UE S1AP ID {mme}, stored differs");
            return s1ap_ue::unknown_enb_ue_s1ap_id_error(mme, enb)
                .map(|pdu| {
                    vec![Action::S1 {
                        stream: self.ue_stream(enb),
                        pdu,
                    }]
                })
                .unwrap_or_default();
        }
        if ctx.security.is_some() || ctx.ics.is_some() || ctx.state != UeState::Connected {
            log::warn!("UE {ue}: INITIAL CONTEXT SETUP while one is active or in progress");
            return self.ics_failure(
                mme,
                enb,
                Cause::RadioNetwork(CauseRadioNetwork(CauseRadioNetwork::UNSPECIFIED)),
            );
        }
        if ics.erabs.is_empty() || ics.erabs.len() > MAX_DRBS {
            log::warn!(
                "UE {ue}: {} E-RABs; this eNB sets up 1..={MAX_DRBS}",
                ics.erabs.len()
            );
            return self.ics_failure(
                mme,
                enb,
                Cause::RadioNetwork(CauseRadioNetwork(
                    CauseRadioNetwork::RADIO_RESOURCES_NOT_AVAILABLE,
                )),
            );
        }
        if let Some(e) = ics
            .erabs
            .iter()
            .find(|e| sgw_endpoint(&e.sgw_address).is_none())
        {
            // §9.2.2.1: 32, 128 or 160 bits. Anything else names no SGW.
            log::warn!(
                "UE {ue}: E-RAB {} Transport Layer Address of {} octets",
                e.e_rab_id,
                e.sgw_address.len()
            );
            return self.ics_failure(
                mme,
                enb,
                Cause::Transport(CauseTransport(CauseTransport::UNSPECIFIED)),
            );
        }
        // TS 33.401 §7.2.4.2.1: the highest-priority algorithm of the eNB's
        // list that is also in the UE's security capabilities.
        let eia = self
            .security
            .integrity
            .iter()
            .copied()
            .find(|a| ics.ue_supports_eia(*a));
        let eea = self
            .security
            .ciphering
            .iter()
            .copied()
            .find(|a| ics.ue_supports_eea(*a));
        let (Some(eia), Some(eea)) = (eia, eea) else {
            log::warn!(
                "UE {ue}: no AS algorithm in common (UE EEA {:#06x}, EIA {:#06x})",
                ics.eea_bitmap,
                ics.eia_bitmap
            );
            return self.ics_failure(
                mme,
                enb,
                Cause::RadioNetwork(CauseRadioNetwork(
                    CauseRadioNetwork::ENCRYPTION_AND_OR_INTEGRITY_PROTECTION_ALGORITHMS_NOT_SUPPORTED,
                )),
            );
        };
        let k_enb = ics.security_key;
        let k_rrc_int = derive_algorithm_key(&k_enb, EpsAlgorithmType::RrcInt, eia);
        let k_rrc_enc = derive_algorithm_key(&k_enb, EpsAlgorithmType::RrcEnc, eea);
        let k_up_enc = derive_algorithm_key(&k_enb, EpsAlgorithmType::UpEnc, eea);
        let ctx = self.ues.get_mut(&ue).expect("present");
        ctx.mme_ue_s1ap_id = Some(mme);
        let tid = ctx.next_tid();
        let smc = match msg::encode_security_mode_command(tid, eea, eia) {
            Ok(b) => b,
            Err(e) => {
                log::error!("cannot encode SecurityModeCommand: {e}");
                return vec![];
            }
        };
        // TS 33.401 §7.2.4.5: the command is integrity protected; uplink is
        // verified from the complete on; downlink ciphering starts after the
        // command is sent; uplink deciphering after the complete verifies.
        if let Err(e) = ctx
            .srb1
            .activate_tx_integrity(eia, k_rrc_int)
            .and_then(|()| ctx.srb1.activate_rx_integrity(eia, k_rrc_int))
        {
            log::error!("UE {ue}: {e}");
            return vec![];
        }
        let (pdcp, _) = ctx.srb1.transmit(&smc);
        if let Err(e) = ctx.srb1.activate_tx_ciphering(eea, k_rrc_enc) {
            log::error!("UE {ue}: {e}");
        }
        ctx.security = Some(AsSecurity {
            eea,
            eia,
            k_rrc_int,
            k_rrc_enc,
            k_up_enc,
        });
        log::info!(
            "UE {ue}: INITIAL CONTEXT SETUP REQUEST ({} E-RAB(s)); SecurityModeCommand EEA{eea}/EIA{eia}",
            ics.erabs.len()
        );
        ctx.ics = Some(PendingIcs {
            erabs: ics.erabs,
            notify: ics.notify,
            rx_ciphering: (eea, k_rrc_enc),
            smc_transaction_id: tid,
            reconfiguration_transaction_id: None,
            teids: Vec::new(),
        });
        vec![Action::Rrc {
            ue,
            channel: RrcChannel::DlDcch,
            lcid: LCID_SRB1,
            pdu: pdcp,
        }]
    }

    fn on_security_mode_complete(&mut self, ue: u32, transaction_id: u8) -> Vec<Action> {
        let Some(ctx) = self.ues.get_mut(&ue) else {
            return vec![];
        };
        let Some(ics) = ctx.ics.as_mut() else {
            log::warn!("UE {ue}: SecurityModeComplete with no security activation running");
            return vec![];
        };
        if ics.smc_transaction_id != transaction_id || ics.reconfiguration_transaction_id.is_some()
        {
            log::warn!("UE {ue}: SecurityModeComplete (transaction {transaction_id}) not expected");
            return vec![];
        }
        // It passed PDCP integrity verification to get here.
        let (eea, k) = ics.rx_ciphering;
        if let Err(e) = ctx.srb1.activate_rx_ciphering(eea, k) {
            log::error!("UE {ue}: {e}");
            return vec![];
        }
        let mut drbs = Vec::new();
        let mut nas = Vec::new();
        let mut teids = Vec::new();
        for (i, e) in ics.erabs.iter().enumerate() {
            drbs.push(DrbConfig {
                eps_bearer_identity: e.e_rab_id,
                drb_identity: i as u8 + 1,
                logical_channel_identity: i as u8 + 3,
                rlc: DrbRlc::UmSn10,
            });
            if let Some(n) = &e.nas {
                nas.push(n.clone());
            }
            teids.push(e.e_rab_id);
        }
        let teids: Vec<(u8, u32)> = teids
            .into_iter()
            .map(|e| {
                let t = self.next_teid;
                self.next_teid = self.next_teid.wrapping_add(1).max(1);
                (e, t)
            })
            .collect();
        let ctx = self.ues.get_mut(&ue).expect("present");
        let ics = ctx.ics.as_mut().expect("present");
        ics.teids = teids;
        let tid = {
            let t = ctx.next_transaction_id;
            ctx.next_transaction_id = (t + 1) & 3;
            t
        };
        ics.reconfiguration_transaction_id = Some(tid);
        // SRB2 is added in the same reconfiguration (TS 36.331 §4.2.2, after
        // security activation), with SRB1's AS security from the start.
        let mut srb2 = LteSrbPdcp::with_bearer(SRB2_BEARER, DIRECTION_DOWNLINK);
        if let Some(sec) = &ctx.security {
            let on = srb2
                .activate_tx_integrity(sec.eia, sec.k_rrc_int)
                .and_then(|()| srb2.activate_rx_integrity(sec.eia, sec.k_rrc_int))
                .and_then(|()| srb2.activate_tx_ciphering(sec.eea, sec.k_rrc_enc))
                .and_then(|()| srb2.activate_rx_ciphering(sec.eea, sec.k_rrc_enc));
            if let Err(e) = on {
                log::error!("UE {ue}: SRB2: {e}");
                return vec![];
            }
        }
        ctx.srb2 = Some(srb2);
        let reconf = match msg::encode_rrc_connection_reconfiguration(tid, true, &drbs, &nas) {
            Ok(b) => b,
            Err(e) => {
                log::error!("cannot encode RRCConnectionReconfiguration: {e}");
                return vec![];
            }
        };
        log::info!(
            "UE {ue}: SecurityModeComplete; AS security active. RRCConnectionReconfiguration: {} DRB(s), {} NAS PDU(s)",
            drbs.len(),
            nas.len()
        );
        ctx.drbs = drbs;
        ctx.erab_qci = ics.erabs.iter().map(|e| (e.e_rab_id, e.qci)).collect();
        let (pdcp, _) = ctx.srb1.transmit(&reconf);
        vec![Action::Rrc {
            ue,
            channel: RrcChannel::DlDcch,
            lcid: LCID_SRB1,
            pdu: pdcp,
        }]
    }

    fn on_security_mode_failure(&mut self, ue: u32, transaction_id: u8) -> Vec<Action> {
        let Some(ctx) = self.ues.get_mut(&ue) else {
            return vec![];
        };
        if ctx.ics.as_ref().map(|i| i.smc_transaction_id) != Some(transaction_id) {
            return vec![];
        }
        // TS 36.331 §5.3.4.3: the UE continues without security; so does SRB1.
        log::warn!("UE {ue}: SecurityModeFailure; INITIAL CONTEXT SETUP FAILURE");
        ctx.srb1.deactivate_security();
        ctx.security = None;
        ctx.ics = None;
        let (Some(mme), enb) = (ctx.mme_ue_s1ap_id, ctx.enb_ue_s1ap_id) else {
            return vec![];
        };
        self.ics_failure(
            mme,
            enb,
            Cause::RadioNetwork(CauseRadioNetwork(
                CauseRadioNetwork::FAILURE_IN_RADIO_INTERFACE_PROCEDURE,
            )),
        )
    }

    fn on_reconfiguration_complete(&mut self, ue: u32, transaction_id: u8) -> Vec<Action> {
        let Some(ctx) = self.ues.get_mut(&ue) else {
            return vec![];
        };
        let Some(ics) = ctx.ics.as_ref() else {
            if ctx.erab.as_ref().map(|p| p.transaction_id) == Some(transaction_id) {
                return self.on_erab_reconfiguration_complete(ue);
            }
            log::info!("UE {ue}: RRCConnectionReconfigurationComplete");
            return vec![];
        };
        if ics.reconfiguration_transaction_id != Some(transaction_id) {
            log::warn!("UE {ue}: RRCConnectionReconfigurationComplete (transaction {transaction_id}) not expected");
            return vec![];
        }
        let ics = ctx.ics.take().expect("checked");
        // The UE has applied the configuration (§5.3.5.3): SRB2 carries NAS
        // from here, and the DRBs' user plane can run.
        ctx.srb2_ready = true;
        let (Some(mme), enb) = (ctx.mme_ue_s1ap_id, ctx.enb_ue_s1ap_id) else {
            return vec![];
        };
        let bearers: Vec<UpBearer> = match &ctx.security {
            Some(sec) => ics
                .erabs
                .iter()
                .zip(&ics.teids)
                .filter_map(|(e, (_, teid))| {
                    let lcid = ctx
                        .drbs
                        .iter()
                        .find(|d| d.eps_bearer_identity == e.e_rab_id)?
                        .logical_channel_identity;
                    Some(UpBearer {
                        e_rab_id: e.e_rab_id,
                        lcid,
                        enb_teid: *teid,
                        sgw: sgw_endpoint(&e.sgw_address)?,
                        sgw_teid: e.sgw_teid,
                        eea: sec.eea,
                        k_up_enc: sec.k_up_enc,
                    })
                })
                .collect(),
            None => Vec::new(),
        };
        let gtp_ip = self.gtp_ip;
        let erabs: Vec<ErabSetup> = ics
            .teids
            .iter()
            .map(|(e, t)| ErabSetup {
                e_rab_id: *e,
                address: gtp_ip,
                teid: *t,
            })
            .collect();
        log::info!(
            "UE {ue}: RRCConnectionReconfigurationComplete; INITIAL CONTEXT SETUP RESPONSE ({} E-RAB(s))",
            erabs.len()
        );
        let mut acts = vec![Action::UserPlane { ue, bearers }];
        if let Some(pdu) = encode_s1(&s1ap_ue::build_initial_context_setup_response(
            mme,
            enb,
            &erabs,
            &ics.notify,
        )) {
            acts.push(Action::S1 {
                stream: self.ue_stream(enb),
                pdu,
            });
        }
        // E-RAB procedures that waited for the Initial Context Setup
        acts.extend(self.next_erab_request(ue));
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
                    lcid: LCID_SRB1,
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

// ---------------------------------------------------------------------------
// E-RAB management (TS 36.413 §8.2; TS 36.331 §5.3.5, §5.3.10; #283)
// ---------------------------------------------------------------------------

fn radio_network(v: u8) -> Cause {
    Cause::RadioNetwork(CauseRadioNetwork(v))
}

/// Whether `qci` is of GBR resource type (TS 23.203 Table 6.1.7-A; the
/// standardized values are those of TS 23.501 Table 5.7.4-1): an E-RAB with
/// one needs the GBR QoS Information IE (TS 36.413 §8.2.1.4, §8.2.2.4).
fn is_gbr_qci(qci: u8) -> bool {
    matches!(qci, 1..=4 | 65..=67 | 71..=76 | 82..=90)
}

/// E-RAB IDs that occur more than once in a request (§8.2.1.4, §8.2.2.4).
fn duplicated(ids: &[u8]) -> Vec<u8> {
    let mut seen = std::collections::HashSet::new();
    let mut dup = Vec::new();
    for id in ids {
        if !seen.insert(*id) && !dup.contains(id) {
            dup.push(*id);
        }
    }
    dup
}

impl EnbRrc {
    /// An E-RAB SETUP / MODIFY REQUEST or E-RAB RELEASE COMMAND. One E-RAB
    /// procedure runs per UE at a time, and none during an Initial Context
    /// Setup: their RRCConnectionReconfigurations are told apart by transaction
    /// id, and the radio configuration they change is the one the previous one
    /// left. Later requests wait in order.
    fn on_erab_request(&mut self, req: ErabRequest) -> Vec<Action> {
        let (mme, enb) = match &req {
            ErabRequest::Setup(r) => (r.mme_ue_s1ap_id, r.enb_ue_s1ap_id),
            ErabRequest::Modify(r) => (r.mme_ue_s1ap_id, r.enb_ue_s1ap_id),
            ErabRequest::Release(r) => (r.mme_ue_s1ap_id, r.enb_ue_s1ap_id),
        };
        let found = self
            .by_enb_id
            .get(&enb)
            .copied()
            .filter(|ue| self.ues[ue].mme_ue_s1ap_id == Some(mme));
        let Some(ue) = found else {
            log::warn!("E-RAB procedure for unknown UE (MME {mme}, eNB {enb})");
            return s1ap_ue::unknown_enb_ue_s1ap_id_error(mme, enb)
                .map(|pdu| {
                    vec![Action::S1 {
                        stream: NON_UE_ASSOCIATED_STREAM,
                        pdu,
                    }]
                })
                .unwrap_or_default();
        };
        self.ues
            .get_mut(&ue)
            .expect("present")
            .erab_queue
            .push_back(req);
        self.next_erab_request(ue)
    }

    /// Start the next queued E-RAB procedure of `ue`, if none runs.
    fn next_erab_request(&mut self, ue: u32) -> Vec<Action> {
        let mut acts = Vec::new();
        loop {
            let Some(ctx) = self.ues.get_mut(&ue) else {
                return acts;
            };
            if ctx.erab.is_some() || ctx.ics.is_some() {
                if !ctx.erab_queue.is_empty() {
                    log::info!(
                        "UE {ue}: {} E-RAB procedure(s) wait for the reconfiguration in progress",
                        ctx.erab_queue.len()
                    );
                }
                return acts;
            }
            let Some(req) = ctx.erab_queue.pop_front() else {
                return acts;
            };
            let started = match req {
                ErabRequest::Setup(r) => self.start_erab_setup(ue, r),
                ErabRequest::Modify(r) => self.start_erab_modify(ue, r),
                ErabRequest::Release(r) => self.start_erab_release(ue, r),
            };
            acts.extend(started);
        }
    }

    /// Send the RRCConnectionReconfiguration of an E-RAB procedure on SRB1
    /// and record what its completion answers.
    #[allow(clippy::too_many_arguments)]
    fn send_erab_reconfiguration(
        &mut self,
        ue: u32,
        mme: u32,
        enb: u32,
        add: &[DrbConfig],
        release: &[u8],
        nas: &[Vec<u8>],
        outcome: ErabOutcome,
        failed: Vec<(u8, Cause)>,
    ) -> Vec<Action> {
        let ctx = self.ues.get_mut(&ue).expect("present");
        let tid = ctx.next_tid();
        let reconf = match msg::encode_rrc_connection_reconfiguration_with_release(
            tid, false, add, release, nas,
        ) {
            Ok(b) => b,
            Err(e) => {
                log::error!("UE {ue}: cannot encode RRCConnectionReconfiguration: {e}");
                return vec![];
            }
        };
        let (pdcp, _) = ctx.srb1.transmit(&reconf);
        ctx.erab = Some(PendingErab {
            mme,
            enb,
            transaction_id: tid,
            outcome,
            failed,
        });
        vec![Action::Rrc {
            ue,
            channel: RrcChannel::DlDcch,
            lcid: LCID_SRB1,
            pdu: pdcp,
        }]
    }

    fn s1_to_ue(&self, enb: u32, pdu: &S1AP_PDU) -> Vec<Action> {
        encode_s1(pdu)
            .map(|pdu| {
                vec![Action::S1 {
                    stream: self.ue_stream(enb),
                    pdu,
                }]
            })
            .unwrap_or_default()
    }

    /// E-RAB SETUP REQUEST (TS 36.413 §8.2.1.2): one DRB per E-RAB that can be
    /// set up, its NAS-PDU to the UE in the reconfiguration; §8.2.1.4: an
    /// E-RAB ID given twice, or of an active E-RAB, fails "Multiple E-RAB ID
    /// instances".
    fn start_erab_setup(&mut self, ue: u32, r: s1ap_ue::ErabSetupRequest) -> Vec<Action> {
        let (mme, enb) = (r.mme_ue_s1ap_id, r.enb_ue_s1ap_id);
        let ctx = &self.ues[&ue];
        let ids: Vec<u8> = r.erabs.iter().map(|e| e.e_rab_id).collect();
        let dup = duplicated(&ids);
        let mut failed = Vec::new();
        let mut ok = Vec::new();
        let secured = ctx.security.is_some();
        let mut used: Vec<u8> = ctx.drbs.iter().map(|d| d.drb_identity).collect();
        for e in r.erabs {
            let cause = if !secured {
                // §8.2.1: E-RABs go on an established UE context, after Initial
                // Context Setup
                Some(radio_network(CauseRadioNetwork::UNSPECIFIED))
            } else if dup.contains(&e.e_rab_id)
                || ctx.drbs.iter().any(|d| d.eps_bearer_identity == e.e_rab_id)
            {
                Some(radio_network(
                    CauseRadioNetwork::MULTIPLE_E_RAB_ID_INSTANCES,
                ))
            } else if is_gbr_qci(e.qci) && !e.gbr_info {
                // §8.2.1.4: a GBR QCI without GBR QoS Information
                Some(radio_network(CauseRadioNetwork::INVALID_QOS_COMBINATION))
            } else if sgw_endpoint(&e.sgw_address).is_none() {
                Some(Cause::Transport(CauseTransport(
                    CauseTransport::UNSPECIFIED,
                )))
            } else {
                None
            };
            if let Some(c) = cause {
                // A repeated E-RAB ID fails once
                if !failed.iter().any(|(f, _)| *f == e.e_rab_id) {
                    failed.push((e.e_rab_id, c));
                }
                continue;
            }
            // The lowest free DRB identity; LCID = DRB identity + 2, as the
            // Initial Context Setup assigns them
            let Some(drb_identity) = (1..=MAX_DRBS as u8).find(|d| !used.contains(d)) else {
                failed.push((
                    e.e_rab_id,
                    radio_network(CauseRadioNetwork::RADIO_RESOURCES_NOT_AVAILABLE),
                ));
                continue;
            };
            used.push(drb_identity);
            let drb = DrbConfig {
                eps_bearer_identity: e.e_rab_id,
                drb_identity,
                logical_channel_identity: drb_identity + 2,
                rlc: DrbRlc::UmSn10,
            };
            ok.push((e, drb));
        }
        if ok.is_empty() {
            log::warn!("UE {ue}: E-RAB SETUP REQUEST: no E-RAB set up");
            return self.s1_to_ue(
                enb,
                &s1ap_ue::build_erab_setup_response(mme, enb, &[], &failed),
            );
        }
        let setup: Vec<(ErabToSetup, u32, DrbConfig)> = ok
            .into_iter()
            .map(|(e, d)| {
                let t = self.next_teid;
                self.next_teid = self.next_teid.wrapping_add(1).max(1);
                (e, t, d)
            })
            .collect();
        let add: Vec<DrbConfig> = setup.iter().map(|(_, _, d)| *d).collect();
        let nas: Vec<Vec<u8>> = setup.iter().filter_map(|(e, _, _)| e.nas.clone()).collect();
        log::info!(
            "UE {ue}: E-RAB SETUP REQUEST; RRCConnectionReconfiguration adds {} DRB(s), {} failed",
            add.len(),
            failed.len()
        );
        self.send_erab_reconfiguration(
            ue,
            mme,
            enb,
            &add,
            &[],
            &nas,
            ErabOutcome::Setup(setup),
            failed,
        )
    }

    /// E-RAB MODIFY REQUEST (TS 36.413 §8.2.2.2): the NAS-PDUs to the UE in
    /// a reconfiguration; this eNB's DRBs carry no QoS of their own over RLS,
    /// so the radio configuration does not change. §8.2.2.4: an unknown or
    /// repeated E-RAB ID fails.
    fn start_erab_modify(&mut self, ue: u32, r: s1ap_ue::ErabModifyRequest) -> Vec<Action> {
        let (mme, enb) = (r.mme_ue_s1ap_id, r.enb_ue_s1ap_id);
        let ctx = &self.ues[&ue];
        let ids: Vec<u8> = r.erabs.iter().map(|e| e.e_rab_id).collect();
        let dup = duplicated(&ids);
        let mut failed = Vec::new();
        let mut modified = Vec::new();
        let mut nas = Vec::new();
        for e in r.erabs {
            if dup.contains(&e.e_rab_id) {
                if !failed.iter().any(|(f, _)| *f == e.e_rab_id) {
                    failed.push((
                        e.e_rab_id,
                        radio_network(CauseRadioNetwork::MULTIPLE_E_RAB_ID_INSTANCES),
                    ));
                }
            } else if !ctx.drbs.iter().any(|d| d.eps_bearer_identity == e.e_rab_id) {
                failed.push((
                    e.e_rab_id,
                    radio_network(CauseRadioNetwork::UNKNOWN_E_RAB_ID),
                ));
            } else if is_gbr_qci(e.qci)
                && !e.gbr_info
                && !ctx
                    .erab_qci
                    .get(&e.e_rab_id)
                    .copied()
                    .is_some_and(is_gbr_qci)
            {
                // §8.2.2.4: non-GBR to GBR without GBR QoS Information
                failed.push((
                    e.e_rab_id,
                    radio_network(CauseRadioNetwork::INVALID_QOS_COMBINATION),
                ));
            } else {
                modified.push((e.e_rab_id, e.qci));
                nas.push(e.nas);
            }
        }
        if modified.is_empty() {
            log::warn!("UE {ue}: E-RAB MODIFY REQUEST: no E-RAB modified");
            return self.s1_to_ue(
                enb,
                &s1ap_ue::build_erab_modify_response(mme, enb, &[], &failed),
            );
        }
        log::info!("UE {ue}: E-RAB MODIFY REQUEST for {modified:?}; RRCConnectionReconfiguration");
        self.send_erab_reconfiguration(
            ue,
            mme,
            enb,
            &[],
            &[],
            &nas,
            ErabOutcome::Modify(modified),
            failed,
        )
    }

    /// E-RAB RELEASE COMMAND (TS 36.413 §8.2.3.2.1): the DRBs go in
    /// `drb-ToReleaseList`, with the NAS-PDU; the S1-U tunnels go once the UE
    /// has completed it. §8.2.3.3: an E-RAB ID given more than once releases
    /// one E-RAB; an unknown one fails.
    fn start_erab_release(&mut self, ue: u32, r: s1ap_ue::ErabReleaseCommand) -> Vec<Action> {
        let (mme, enb) = (r.mme_ue_s1ap_id, r.enb_ue_s1ap_id);
        let ctx = &self.ues[&ue];
        let mut failed: Vec<(u8, Cause)> = Vec::new();
        let mut released = Vec::new();
        let mut release = Vec::new();
        for (i, &id) in r.erabs.iter().enumerate() {
            if r.erabs[..i].contains(&id) {
                continue;
            }
            match ctx.drbs.iter().find(|d| d.eps_bearer_identity == id) {
                Some(d) => {
                    released.push((id, d.logical_channel_identity));
                    release.push(d.drb_identity);
                }
                None => failed.push((id, radio_network(CauseRadioNetwork::UNKNOWN_E_RAB_ID))),
            }
        }
        if released.is_empty() {
            log::warn!("UE {ue}: E-RAB RELEASE COMMAND: no E-RAB released");
            let mut acts = self.s1_to_ue(
                enb,
                &s1ap_ue::build_erab_release_response(mme, enb, &[], &failed),
            );
            // "If a NAS-PDU IE is contained in the message, the eNB shall pass
            // it to the UE"
            if let Some(nas) = r.nas {
                acts.extend(self.on_downlink_nas(mme, enb, nas));
            }
            return acts;
        }
        let nas: Vec<Vec<u8>> = r.nas.into_iter().collect();
        log::info!(
            "UE {ue}: E-RAB RELEASE COMMAND for {:?}; RRCConnectionReconfiguration releases DRB(s) {release:?}",
            released.iter().map(|(e, _)| e).collect::<Vec<_>>()
        );
        self.send_erab_reconfiguration(
            ue,
            mme,
            enb,
            &[],
            &release,
            &nas,
            ErabOutcome::Release(released),
            failed,
        )
    }

    /// The UE completed an E-RAB procedure's RRCConnectionReconfiguration
    /// (TS 36.331 §5.3.5.3): set up or release the user plane and answer the
    /// MME, then start the next queued procedure.
    fn on_erab_reconfiguration_complete(&mut self, ue: u32) -> Vec<Action> {
        let gtp_ip = self.gtp_ip;
        let ctx = self.ues.get_mut(&ue).expect("present");
        let p = ctx.erab.take().expect("checked by the caller");
        let mut acts = Vec::new();
        let response = match p.outcome {
            ErabOutcome::Setup(setup) => {
                let sec = ctx.security.clone().expect("checked at the request");
                let bearers: Vec<UpBearer> = setup
                    .iter()
                    .filter_map(|(e, teid, d)| {
                        Some(UpBearer {
                            e_rab_id: e.e_rab_id,
                            lcid: d.logical_channel_identity,
                            enb_teid: *teid,
                            sgw: sgw_endpoint(&e.sgw_address)?,
                            sgw_teid: e.sgw_teid,
                            eea: sec.eea,
                            k_up_enc: sec.k_up_enc,
                        })
                    })
                    .collect();
                ctx.drbs.extend(setup.iter().map(|(_, _, d)| *d));
                ctx.erab_qci
                    .extend(setup.iter().map(|(e, _, _)| (e.e_rab_id, e.qci)));
                let erabs: Vec<ErabSetup> = setup
                    .iter()
                    .map(|(e, t, _)| ErabSetup {
                        e_rab_id: e.e_rab_id,
                        address: gtp_ip,
                        teid: *t,
                    })
                    .collect();
                log::info!(
                    "UE {ue}: RRCConnectionReconfigurationComplete; E-RAB SETUP RESPONSE ({} set up, {} failed)",
                    erabs.len(),
                    p.failed.len()
                );
                acts.push(Action::UserPlane { ue, bearers });
                s1ap_ue::build_erab_setup_response(p.mme, p.enb, &erabs, &p.failed)
            }
            ErabOutcome::Modify(modified) => {
                ctx.erab_qci.extend(modified.iter().copied());
                log::info!("UE {ue}: RRCConnectionReconfigurationComplete; E-RAB MODIFY RESPONSE");
                let ids: Vec<u8> = modified.iter().map(|(e, _)| *e).collect();
                s1ap_ue::build_erab_modify_response(p.mme, p.enb, &ids, &p.failed)
            }
            ErabOutcome::Release(released) => {
                ctx.drbs
                    .retain(|d| !released.iter().any(|(e, _)| *e == d.eps_bearer_identity));
                for (e, _) in &released {
                    ctx.erab_qci.remove(e);
                }
                log::info!("UE {ue}: RRCConnectionReconfigurationComplete; E-RAB RELEASE RESPONSE");
                acts.push(Action::ReleaseUserPlane {
                    ue,
                    lcids: released.iter().map(|(_, l)| *l).collect(),
                });
                let ids: Vec<u8> = released.iter().map(|(e, _)| *e).collect();
                s1ap_ue::build_erab_release_response(p.mme, p.enb, &ids, &p.failed)
            }
        };
        acts.extend(self.s1_to_ue(p.enb, &response));
        acts.extend(self.next_erab_request(ue));
        acts
    }
}

#[cfg(test)]
mod tests;
