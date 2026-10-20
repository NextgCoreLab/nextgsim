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

use nextgsim_eps_nas::{DefaultBearer, EmmState, EpsNasConfig, EpsNasUe};
use nextgsim_lte_rrc::msg::EstablishmentCauseValue;

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
        Some(InitialNas {
            nas,
            // TS 24.301 Annex D Table D.1.1: attach is "mo-Signalling".
            cause: EstablishmentCauseValue::MoSignalling,
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
        }
        self.publish();
    }

    fn k_enb(&mut self) -> Option<[u8; 32]> {
        self.ue.k_enb()
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
