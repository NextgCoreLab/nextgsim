//! The seam between the UE's RRC and its NAS.
//!
//! TS 36.331 §5.3.3.2: RRC establishes a connection when "upper layers
//! request establishment of an RRC connection", handing it the initial NAS
//! message (§5.3.3.4: carried in RRCConnectionSetupComplete
//! `dedicatedInfoNAS`), and relays NAS both ways afterwards (§5.6.1, §5.6.2).
//! EPS NAS itself (EMM/ESM, TS 24.301) is nextgsim #186 increment 3; until
//! then the product UE runs [`NoNas`], which never requests a connection, and
//! tests plug in a scripted NAS.

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

/// The UE's NAS as RRC sees it.
pub trait NasLayer: Send {
    /// Called when RRC is camped and idle: the initial NAS message, if NAS
    /// wants a connection now.
    fn initial_nas(&mut self) -> Option<InitialNas>;
    /// A NAS message from the network (DLInformationTransfer). Returns the
    /// NAS messages to send back, in order (ULInformationTransfer).
    fn on_downlink_nas(&mut self, nas: &[u8]) -> Vec<Vec<u8>>;
    /// An RRC indication.
    fn on_rrc_indication(&mut self, ind: RrcIndication);
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
