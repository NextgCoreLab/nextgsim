//! The seam between the UE's RRC and its NAS.
//!
//! TS 36.331 §5.3.3.2: RRC establishes a connection when "upper layers
//! request establishment of an RRC connection", handing it the initial NAS
//! message (§5.3.3.4: carried in RRCConnectionSetupComplete
//! `dedicatedInfoNAS`), and relays NAS both ways afterwards (§5.6.1, §5.6.2).
//! For the AS Security Mode Command RRC asks NAS for `K_eNB` (§5.3.4.3: "derive
//! the KeNB key, as specified in TS 33.401").
//!
//! The product UE runs [`EpsNas`] -- TS 24.301 EMM/ESM from
//! `nextgsim-eps-nas` -- when its configuration carries a subscription, and
//! [`NoNas`] (camp only) when it does not; tests plug in a scripted NAS.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextgsim_eps_nas::{
    BearerCommand, BearerCommandRefusal, BearerResourceOutcome, DefaultBearer, EmmState,
    EpsNasConfig, EpsNasUe, PagedWith, ServiceRequestFailure,
};
use nextgsim_lte_rrc::msg::{EstablishmentCauseValue, PagedUeIdentity};

/// What NAS hands RRC when it needs a connection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InitialNas {
    /// The initial NAS message (e.g. ATTACH REQUEST).
    pub nas: Vec<u8>,
    /// RRC establishment cause (TS 24.301 Annex D maps NAS procedures to it).
    pub cause: EstablishmentCauseValue,
    /// The S-TMSI, when NAS holds a valid GUTI (§5.3.3.3: used as the
    /// `ue-Identity`); otherwise RRC draws a random value.
    pub s_tmsi: Option<(u8, u32)>,
}

/// What RRC tells NAS (TS 36.331's "inform upper layers").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RrcIndication {
    /// RRCConnectionSetupComplete sent: the connection exists (§5.3.3.4).
    Connected,
    /// Establishment failed: T300 expiry (§5.3.3.6) or RRCConnectionReject
    /// (§5.3.3.8, with its wait time).
    EstablishmentFailed {
        /// The wait time, when rejected.
        wait_time_s: Option<u8>,
    },
    /// The connection is gone: RRCConnectionRelease (§5.3.8.3) or radio link
    /// failure before AS security (§5.3.11.3, release cause "other").
    Released,
    /// An RRCConnectionReconfiguration set up the DRBs: "the user plane radio
    /// bearers are set up" (TS 24.301 §5.6.1.4).
    RadioBearersEstablished,
}

/// What a NAS timer asks of RRC.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NasTimerOutcome {
    /// NAS messages to send on the existing connection.
    pub uplink: Vec<Vec<u8>>,
    /// Release the connection locally (e.g. TS 24.301 §5.5.1.2.6 c, T3410).
    pub release: bool,
}

/// The UE's NAS as RRC sees it.
pub trait NasLayer: Send {
    /// Called when RRC is camped and idle: the initial NAS message, if NAS
    /// wants a connection now.
    fn initial_nas(&mut self) -> Option<InitialNas>;
    /// A NAS message from the network (DLInformationTransfer, or the
    /// `dedicatedInfoNASList` of RRCConnectionReconfiguration). Returns the
    /// NAS messages to send back, in order (ULInformationTransfer).
    fn on_downlink_nas(&mut self, nas: &[u8]) -> Vec<Vec<u8>>;
    /// An RRC indication.
    fn on_rrc_indication(&mut self, ind: RrcIndication);
    /// An uplink IP packet found no DRB because the UE is in RRC_IDLE: NAS
    /// may want a connection for it (TS 24.301 §5.6.1.1 b).
    fn on_uplink_pending(&mut self) {}
    /// Whether NAS expects the user plane radio bearers to be set up by a
    /// procedure under way or about to start (a service request, an attach),
    /// so RRC holds uplink IP packets that have no DRB yet (#435).
    fn awaits_user_plane(&self) -> bool {
        false
    }
    /// How the last service request ended without the user plane, once
    /// (#435): RRC then discards the uplink it held for it.
    fn take_service_request_failure(&mut self) -> Option<ServiceRequestFailure> {
        None
    }
    /// RRC camps on a cell of tracking area `tac` of the selected PLMN
    /// (SIB1 `trackingAreaCode`): NAS compares its TAI with the TAI list
    /// (TS 24.301 §5.5.3.2.2 a), #356).
    fn on_serving_cell(&mut self, _tac: u16) {}
    /// `K_eNB` for the AS Security Mode Command, when NAS has a security
    /// context (TS 33.401 §7.2.6.2).
    fn k_enb(&mut self) -> Option<[u8; 32]> {
        None
    }
    /// The EPS bearer identity of the default bearer, once activated: the
    /// bearer uplink IP packets no TFT picks go on (TS 23.401 §4.7.2).
    fn default_bearer_ebi(&self) -> Option<u8> {
        None
    }
    /// The EPS bearer an uplink IP packet goes on, by the TFTs' uplink packet
    /// filters (TS 23.401 §5.4.1, TS 24.301 §6.4.2; #283); `None` to discard
    /// it while a default bearer exists.
    fn uplink_bearer_for(&self, _packet: &[u8]) -> Option<u8> {
        self.default_bearer_ebi()
    }
    /// The earliest NAS timer deadline.
    fn next_deadline(&self) -> Option<Instant> {
        None
    }
    /// NAS timers that are due.
    fn on_timer(&mut self, _now: Instant) -> NasTimerOutcome {
        NasTimerOutcome::default()
    }
    /// A PCCH paging record (TS 36.331 §5.3.2.3: forwarded "to the upper
    /// layers" when the identity matches one the UE was allocated). Returns
    /// whether NAS now wants a connection (#351). Only PS-domain records reach
    /// here; this UE has no CS fallback.
    fn on_paging(&mut self, _identity: &PagedUeIdentity) -> bool {
        false
    }
    /// A UE-initiated detach (TS 24.301 §5.5.2.2, #365). `connected` says
    /// whether there is a NAS signalling connection to send the DETACH REQUEST
    /// on now. The returned uplink goes on it; without one, NAS holds the
    /// request as its next initial NAS message.
    fn detach(&mut self, _switch_off: bool, _connected: bool) -> NasTimerOutcome {
        NasTimerOutcome::default()
    }
    /// A UE-requested bearer resource modification (TS 24.301 §6.5.4, #434).
    /// `connected` as for [`NasLayer::detach`]. Returns the procedure's PTI and
    /// the uplink to send now; [`BearerCommandRefusal::NotConnected`] asks for
    /// a connection (NAS then has a SERVICE REQUEST as its initial message)
    /// and for the command again once connected. A NAS without EPS bearers
    /// refuses as not EMM-REGISTERED.
    fn bearer_command(
        &mut self,
        _cmd: BearerCommand,
        _connected: bool,
    ) -> Result<(u8, NasTimerOutcome), BearerCommandRefusal> {
        Err(BearerCommandRefusal::NotRegistered(EmmState::Deregistered))
    }
}

/// A NAS that never requests a connection: the UE selects a cell and camps
/// (TS 36.304 §5.2), which is all an RRC without EMM can honestly do.
#[derive(Debug, Default)]
pub struct NoNas;

impl NasLayer for NoNas {
    fn initial_nas(&mut self) -> Option<InitialNas> {
        None
    }
    fn on_downlink_nas(&mut self, _nas: &[u8]) -> Vec<Vec<u8>> {
        Vec::new()
    }
    fn on_rrc_indication(&mut self, _ind: RrcIndication) {}
}

/// What an observer of the UE's NAS can see (a CLI, a test).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpsNasStatus {
    /// The EMM state.
    pub state: EmmState,
    /// The default EPS bearer, once activated.
    pub bearer: Option<DefaultBearer>,
    /// The tracking area updating attempt counter (TS 24.301 §5.5.3.1, #356):
    /// 0 after a TRACKING AREA UPDATE ACCEPT.
    pub tau_attempts: u8,
    /// The M-TMSI of the GUTI, once assigned.
    pub m_tmsi: Option<u32>,
    /// How the recent UE-requested bearer resource modifications ended, by
    /// PTI (#434).
    pub bearer_outcomes: Vec<(u8, BearerResourceOutcome)>,
}

impl EpsNasStatus {
    /// How the UE-requested bearer resource modification on `pti` ended;
    /// `None` while it runs.
    pub fn bearer_outcome(&self, pti: u8) -> Option<BearerResourceOutcome> {
        self.bearer_outcomes
            .iter()
            .rev()
            .find(|(p, _)| *p == pti)
            .map(|(_, o)| *o)
    }
}

/// TS 24.301 EPS NAS for the RRC: [`EpsNasUe`] behind the [`NasLayer`] seam.
#[derive(Debug)]
pub struct EpsNas {
    ue: EpsNasUe,
    status: Arc<Mutex<EpsNasStatus>>,
}

impl EpsNas {
    /// A UE NAS for `cfg`, EMM-DEREGISTERED with an attach due.
    pub fn new(cfg: EpsNasConfig) -> Self {
        Self::from_ue(EpsNasUe::new(cfg))
    }

    /// The layer around an existing EMM entity (tests drive one through an
    /// attach first).
    pub(crate) fn from_ue(ue: EpsNasUe) -> Self {
        let status = Arc::new(Mutex::new(EpsNasStatus {
            state: ue.state(),
            bearer: None,
            tau_attempts: 0,
            m_tmsi: None,
            bearer_outcomes: Vec::new(),
        }));
        Self { ue, status }
    }

    /// The EMM entity, for observers.
    pub fn ue(&self) -> &EpsNasUe {
        &self.ue
    }

    /// A handle that follows the NAS state after the layer is handed to RRC.
    pub fn status(&self) -> Arc<Mutex<EpsNasStatus>> {
        Arc::clone(&self.status)
    }

    fn publish(&self) {
        if let Ok(mut s) = self.status.lock() {
            s.state = self.ue.state();
            s.bearer = self.ue.default_bearer().cloned();
            s.tau_attempts = self.ue.tau_attempts();
            s.m_tmsi = self.ue.guti().map(|g| g.m_tmsi);
            s.bearer_outcomes = self.ue.bearer_resource_outcomes().to_vec();
        }
    }
}

impl NasLayer for EpsNas {
    fn initial_nas(&mut self) -> Option<InitialNas> {
        let nas = self.ue.initial_message(Instant::now());
        self.publish();
        let nas = nas?;
        // TS 24.301 Annex D Table D.1.1: attach and tracking area update are
        // "mo-Signalling"; a service request for uplink user data is
        // "mo-Data".
        // A service request (#351) or a tracking area update (#356 review)
        // answering a paging is "mt-Access".
        let cause = if self.ue.initial_message_answers_paging() {
            EstablishmentCauseValue::MtAccess
        } else if self.ue.state() == EmmState::ServiceRequestInitiated {
            EstablishmentCauseValue::MoData
        } else {
            EstablishmentCauseValue::MoSignalling
        };
        Some(InitialNas {
            nas,
            cause,
            s_tmsi: self.ue.s_tmsi(),
        })
    }

    fn on_downlink_nas(&mut self, nas: &[u8]) -> Vec<Vec<u8>> {
        // A release asked for here is not acted on: no downlink NAS message of
        // the procedures implemented makes the UE release the connection.
        let out = self.ue.on_downlink(nas, Instant::now()).uplink;
        self.publish();
        out
    }

    fn on_rrc_indication(&mut self, ind: RrcIndication) {
        match ind {
            RrcIndication::Connected => self.ue.on_connected(),
            RrcIndication::EstablishmentFailed { .. } | RrcIndication::Released => {
                self.ue.on_released(Instant::now());
            }
            RrcIndication::RadioBearersEstablished => self.ue.on_user_plane_established(),
        }
        self.publish();
    }

    fn on_uplink_pending(&mut self) {
        self.ue.on_uplink_data_pending();
    }

    fn awaits_user_plane(&self) -> bool {
        self.ue.user_plane_awaited()
    }

    fn take_service_request_failure(&mut self) -> Option<ServiceRequestFailure> {
        self.ue.take_service_request_failure()
    }

    fn on_serving_cell(&mut self, tac: u16) {
        self.ue.on_serving_cell(tac);
        self.publish();
    }

    fn on_paging(&mut self, identity: &PagedUeIdentity) -> bool {
        // TS 36.331 §5.3.2.3: a record is for this UE when its ue-Identity
        // matches the S-TMSI, or the IMSI, it holds.
        let with = match identity {
            PagedUeIdentity::STmsi { mmec, m_tmsi } => {
                if self.ue.s_tmsi() != Some((*mmec, *m_tmsi)) {
                    return false;
                }
                PagedWith::STmsi
            }
            PagedUeIdentity::Imsi(digits) => {
                let ours: Vec<u8> = self
                    .ue
                    .imsi()
                    .bytes()
                    .filter(u8::is_ascii_digit)
                    .map(|b| b - b'0')
                    .collect();
                if *digits != ours {
                    return false;
                }
                PagedWith::Imsi
            }
        };
        let wants = self.ue.on_paging(with);
        self.publish();
        wants
    }

    fn k_enb(&mut self) -> Option<[u8; 32]> {
        self.ue.k_enb()
    }

    fn detach(&mut self, switch_off: bool, connected: bool) -> NasTimerOutcome {
        let out = self.ue.detach(switch_off, connected, Instant::now());
        self.publish();
        NasTimerOutcome {
            uplink: out.uplink,
            release: out.release,
        }
    }

    fn bearer_command(
        &mut self,
        cmd: BearerCommand,
        connected: bool,
    ) -> Result<(u8, NasTimerOutcome), BearerCommandRefusal> {
        let started = self.ue.bearer_command(cmd, connected, Instant::now());
        self.publish();
        let (pti, out) = started?;
        Ok((
            pti,
            NasTimerOutcome {
                uplink: out.uplink,
                release: out.release,
            },
        ))
    }

    fn default_bearer_ebi(&self) -> Option<u8> {
        self.ue.default_bearer().map(|b| b.ebi)
    }

    fn uplink_bearer_for(&self, packet: &[u8]) -> Option<u8> {
        self.ue.uplink_bearer_for(packet)
    }

    fn next_deadline(&self) -> Option<Instant> {
        self.ue.next_deadline()
    }

    fn on_timer(&mut self, now: Instant) -> NasTimerOutcome {
        let out = self.ue.on_timer(now);
        self.publish();
        NasTimerOutcome {
            uplink: out.uplink,
            release: out.release,
        }
    }
}

#[cfg(test)]
mod tests {
    //! #351 review: the paging match and the response cause on the REAL
    //! [`EpsNas`], driven through an attach against the eps-nas scripted MME.
    use super::*;
    use nextgsim_eps_nas::testing::{registered_idle, IMSI};

    /// The GUTI the scripted MME's ATTACH ACCEPT assigns: MMEC 0x01,
    /// M-TMSI 0xC0000001.
    const OUR_S_TMSI: PagedUeIdentity = PagedUeIdentity::STmsi {
        mmec: 0x01,
        m_tmsi: 0xC000_0001,
    };

    fn registered() -> EpsNas {
        let (ue, _mme, _now) = registered_idle();
        let nas = EpsNas {
            status: Arc::new(Mutex::new(EpsNasStatus {
                state: ue.state(),
                bearer: None,
                tau_attempts: 0,
                m_tmsi: None,
                bearer_outcomes: Vec::new(),
            })),
            ue,
        };
        assert_eq!(nas.ue.s_tmsi(), Some((0x01, 0xC000_0001)));
        nas
    }

    fn digits(imsi: &str) -> Vec<u8> {
        imsi.bytes().map(|b| b - b'0').collect()
    }

    /// TS 36.331 §5.3.2.3 + TS 24.301 §5.6.2.2.1 and Annex D: this UE's S-TMSI
    /// starts a connection whose initial NAS is a SERVICE REQUEST with cause
    /// mt-Access.
    #[test]
    fn our_s_tmsi_is_answered_with_an_mt_access_service_request() {
        let mut nas = registered();
        assert!(nas.on_paging(&OUR_S_TMSI));
        let initial = nas.initial_nas().expect("the paging response");
        assert_eq!(initial.cause, EstablishmentCauseValue::MtAccess);
        assert_eq!(
            initial.nas[0], 0xC7,
            "SERVICE REQUEST, security header type 12"
        );
        assert_eq!(initial.s_tmsi, Some((0x01, 0xC000_0001)));
    }

    /// #356 review, TS 24.301 Annex D Table D.1.1: a TRACKING AREA UPDATE
    /// REQUEST that answers a paging goes with cause mt-Access; one that
    /// does not, mo-Signalling.
    #[test]
    fn a_tracking_area_update_answering_a_paging_is_mt_access() {
        let mut nas = registered();
        nas.on_serving_cell(2);
        assert!(nas.on_paging(&OUR_S_TMSI));
        let initial = nas.initial_nas().expect("the paging response");
        assert_eq!(initial.cause, EstablishmentCauseValue::MtAccess);
        assert_eq!(initial.nas[0], 0x17, "integrity protected, not ciphered");
        assert_eq!(initial.nas[7], 0x48, "TRACKING AREA UPDATE REQUEST");

        let mut nas = registered();
        nas.on_serving_cell(2);
        let initial = nas.initial_nas().expect("the TA update");
        assert_eq!(initial.nas[7], 0x48);
        assert_eq!(initial.cause, EstablishmentCauseValue::MoSignalling);
    }

    /// Another UE's S-TMSI is not this UE's paging: no connection, nothing owed.
    #[test]
    fn another_s_tmsi_is_not_answered() {
        for other in [
            PagedUeIdentity::STmsi {
                mmec: 0x01,
                m_tmsi: 0xC000_0002,
            },
            PagedUeIdentity::STmsi {
                mmec: 0x02,
                m_tmsi: 0xC000_0001,
            },
        ] {
            let mut nas = registered();
            assert!(!nas.on_paging(&other), "{other:?}");
            assert!(nas.initial_nas().is_none(), "{other:?}");
            assert_eq!(nas.ue.state(), EmmState::Registered);
        }
    }

    /// TS 24.301 §5.6.2.2.2: a paging with this UE's IMSI detaches locally and
    /// attaches again (mo-Signalling, Annex D), the GUTI gone.
    #[test]
    fn our_imsi_detaches_and_reattaches() {
        let mut nas = registered();
        assert!(nas.on_paging(&PagedUeIdentity::Imsi(digits(IMSI))));
        assert_eq!(nas.ue.s_tmsi(), None, "the GUTI is deleted");
        let initial = nas.initial_nas().expect("ATTACH REQUEST");
        assert_eq!(&initial.nas[..2], &[0x07, 0x41], "plain ATTACH REQUEST");
        assert_eq!(initial.cause, EstablishmentCauseValue::MoSignalling);
    }

    /// #365: the real EpsNas holds an idle detach as its initial NAS message,
    /// with cause mo-Signalling (Annex D), and says it is detaching.
    #[test]
    fn an_idle_detach_is_the_next_initial_nas_message() {
        let mut nas = registered();
        let out = nas.detach(false, false);
        assert!(out.uplink.is_empty());
        let initial = nas.initial_nas().expect("the DETACH REQUEST");
        assert_eq!(initial.cause, EstablishmentCauseValue::MoSignalling);
        assert_eq!(initial.nas[0] >> 4, 1, "integrity protected");
        assert_eq!(nas.ue.state(), EmmState::DeregisteredInitiated);
        assert_eq!(
            nas.status.lock().unwrap().state,
            EmmState::DeregisteredInitiated
        );
    }

    /// #434: the real EpsNas runs a bearer release and publishes how it ended,
    /// which is what the control socket's answer waits on.
    #[test]
    fn a_bearer_release_is_sent_and_its_outcome_published() {
        use nextgsim_eps_nas::testing::attach;
        let (ue, mut mme, _) = attach();
        let mut nas = EpsNas {
            status: Arc::new(Mutex::new(EpsNasStatus {
                state: ue.state(),
                bearer: None,
                tau_attempts: 0,
                m_tmsi: None,
                bearer_outcomes: Vec::new(),
            })),
            ue,
        };
        // ACTIVATE DEDICATED EPS BEARER CONTEXT REQUEST: EBI 6, linked 5, QCI 1,
        // one uplink UDP filter (id 1)
        let act = [
            0x62, 0x00, 0xC5, 0x05, 0x01, 0x01, 0x06, 0x21, 0x21, 0x10, 0x02, 0x30, 0x11,
        ];
        let reply = nas.on_downlink_nas(&mme.protect(&act, 2));
        assert_eq!(mme.unprotect(&reply[0]), vec![0x62, 0x00, 0xC6]);

        let cmd = BearerCommand::Release { ebi: 6 };
        assert_eq!(
            nas.bearer_command(BearerCommand::Release { ebi: 5 }, true),
            Err(BearerCommandRefusal::DefaultBearer(5))
        );
        let (pti, out) = nas.bearer_command(cmd, true).expect("started");
        assert_eq!(
            mme.unprotect(&out.uplink[0]),
            vec![0x02, pti, 0xD6, 0x06, 0x02, 0xA1, 0x01, 0x58, 36]
        );
        assert_eq!(nas.status.lock().unwrap().bearer_outcome(pti), None);
        let reply = nas.on_downlink_nas(&mme.protect(&[0x62, pti, 0xCD, 36], 2));
        assert_eq!(mme.unprotect(&reply[0]), vec![0x62, pti, 0xCE]);
        let outcome = nas.status.lock().unwrap().bearer_outcome(pti);
        assert_eq!(outcome, Some(BearerResourceOutcome::Deactivated));
        assert_eq!(cmd.answer(outcome.unwrap()), Ok("released"));
    }

    /// Another UE's IMSI must not wipe this UE's registration.
    #[test]
    fn another_imsi_leaves_the_registration_alone() {
        let mut other = digits(IMSI);
        *other.last_mut().unwrap() = 2;
        for paged in [other, digits("99970000000000")] {
            let mut nas = registered();
            assert!(
                !nas.on_paging(&PagedUeIdentity::Imsi(paged.clone())),
                "{paged:?}"
            );
            assert_eq!(nas.ue.s_tmsi(), Some((0x01, 0xC000_0001)), "GUTI kept");
            assert_eq!(nas.ue.state(), EmmState::Registered);
            assert!(nas.initial_nas().is_none());
        }
    }
}
