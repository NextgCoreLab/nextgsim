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

use nextgsim_eps_nas::{DefaultBearer, EmmState, EpsNasConfig, EpsNasUe, PagedWith};
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
    /// `K_eNB` for the AS Security Mode Command, when NAS has a security
    /// context (TS 33.401 §7.2.6.2).
    fn k_enb(&mut self) -> Option<[u8; 32]> {
        None
    }
    /// The EPS bearer identity of the default bearer, once activated: the
    /// bearer uplink IP packets go on (TS 23.401 §4.7.2; no TFTs here).
    fn default_bearer_ebi(&self) -> Option<u8> {
        None
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
        let ue = EpsNasUe::new(cfg);
        let status = Arc::new(Mutex::new(EpsNasStatus {
            state: ue.state(),
            bearer: None,
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
        }
    }
}

impl NasLayer for EpsNas {
    fn initial_nas(&mut self) -> Option<InitialNas> {
        let nas = self.ue.initial_message(Instant::now());
        self.publish();
        let nas = nas?;
        // TS 24.301 Annex D Table D.1.1: attach is "mo-Signalling"; a service
        // request for uplink user data is "mo-Data".
        // A service request answering a paging is "mt-Access" (#351).
        let cause = if self.ue.service_request_answers_paging() {
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

    fn default_bearer_ebi(&self) -> Option<u8> {
        self.ue.default_bearer().map(|b| b.ebi)
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
