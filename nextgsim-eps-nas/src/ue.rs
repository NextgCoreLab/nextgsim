//! The UE's EMM and ESM entities for attach (TS 24.301 §5.5.1), the common
//! procedures attach runs -- identification (§5.4.4), authentication (§5.4.2)
//! and security mode control (§5.4.3) -- and the default EPS bearer it sets up
//! (§6.4.1).
//!
//! Pure: downlink NAS PDUs, lower-layer indications and the clock in; uplink
//! NAS PDUs and a "release the connection" flag out. The LTE UE's RRC owns the
//! connection and the radio; this module owns NAS state.
//!
//! # What is in scope
//!
//! - **Attach** with IMSI, or with the GUTI an earlier ATTACH ACCEPT assigned
//!   (§5.5.1.2.2), carrying a PDN CONNECTIVITY REQUEST; ATTACH ACCEPT ->
//!   ACTIVATE DEFAULT EPS BEARER CONTEXT ACCEPT inside ATTACH COMPLETE
//!   (§5.5.1.2.4, §6.4.1.3); ATTACH REJECT per §5.5.1.2.5; the abnormal cases of
//!   §5.5.1.2.6 with T3410, T3411, T3402 and the attach attempt counter.
//! - **EPS AKA** with MILENAGE ([`crate::aka`]), including MAC failure (#20),
//!   synch failure (#21, AUTS) and the AMF separation bit (#26).
//! - **NAS SMC**: key derivation for the selected EEA/EIA, the replayed UE
//!   security capability check (§5.4.3.3, #23), the IMEISV request, and from
//!   then on protected uplink and verified downlink (§4.4.4).
//! - `K_eNB` for the AS Security Mode Command (TS 33.401 §7.2.6.2, A.3).
//! - **Service request** (§5.6.1) for an EMM-REGISTERED UE in EMM-IDLE with
//!   pending uplink user data (§5.6.1.1 case b): SERVICE REQUEST under T3417,
//!   completed by the lower layers' "user plane radio bearers set up"
//!   indication (§5.6.1.4); SERVICE REJECT per §5.6.1.5 -- #9, #10 and #40
//!   lead to a new attach -- and the abnormal cases of §5.6.1.6 that apply
//!   (T3417 expiry, lower layer failure).
//! - **Network-initiated detach** (§5.5.2.3): DETACH REQUEST is answered with
//!   DETACH ACCEPT and the EPS bearer contexts are deactivated locally; with
//!   "re-attach required" (or "re-attach not required" without a cause that
//!   bars service) the UE is EMM-DEREGISTERED.NORMAL-SERVICE and attaches
//!   again once the NAS signalling connection is released (T3440 releasing it
//!   locally if the network does not).
//!
//! # What is not (and is said so where it would be reached)
//!
//! Tracking area update, UE-initiated detach and paging are not implemented. A network
//! message of a procedure this UE does not implement is answered with EMM
//! STATUS #97 (§7.4), which is what the clause asks of a UE that does not
//! implement a message type.

use std::time::{Duration, Instant};

use nextgsim_common::Plmn;

use crate::aka::{self, AkaOutcome, SqnState, UsimCredentials};
use crate::ie::{
    EpsMobileIdentity, Guti, MobileIdentity, NasKsi, PdnAddress, PdnType, Tai, UeNetworkCapability,
};
use crate::msg::{
    self, emm_cause, AttachRequest, DownlinkEmm, DownlinkEsm, EpsAttachType,
    PdnConnectivityRequest, PD_EMM, PD_ESM,
};
use crate::security::{header_type, security_header_type, EpsSecurityContext};

/// T3410: attach supervision (§10.2, Table 10.2.1).
pub const T3410: Duration = Duration::from_secs(15);
/// T3411: attach retry (§10.2).
pub const T3411: Duration = Duration::from_secs(10);
/// T3416: lifetime of the stored RAND and RES (§5.4.2.3, Table 10.2.1).
pub const T3416: Duration = Duration::from_secs(30);
/// T3402 default: 12 minutes (§10.2).
pub const T3402_DEFAULT: Duration = Duration::from_secs(12 * 60);
/// T3417: service request supervision (§10.2, Table 10.2.1).
pub const T3417: Duration = Duration::from_secs(5);
/// T3440: wait for the network to release the NAS signalling connection
/// after a reject that ends the UE's registration (§10.2, Table 10.2.1); on
/// expiry the UE releases it locally.
pub const T3440: Duration = Duration::from_secs(10);
/// Attach attempt counter limit (§5.5.1.2.6: "equal to 5").
pub const MAX_ATTACH_ATTEMPTS: u8 = 5;

/// ESM cause #81 "invalid PTI value" (§9.9.4.4).
const ESM_INVALID_PTI: u8 = 81;
/// ESM cause #95 "semantically incorrect message".
const ESM_SEMANTICALLY_INCORRECT: u8 = 95;
/// ESM cause #97 "message type non-existent or not implemented".
const ESM_MESSAGE_TYPE_NOT_IMPLEMENTED: u8 = 97;

/// The subscription and capabilities this UE attaches with.
#[derive(Debug, Clone)]
pub struct EpsNasConfig {
    /// IMSI digits.
    pub imsi: String,
    /// The PLMN the UE selected: the serving network whose identity keys
    /// `K_ASME` (TS 33.401 A.2) -- the UE has no other source for it.
    pub plmn: Plmn,
    /// K and OPc.
    pub credentials: UsimCredentials,
    /// The USIM's highest accepted SQN at power-on.
    pub initial_sqn: u64,
    /// IMEISV digits, sent when a SECURITY MODE COMMAND requests it.
    pub imeisv: Option<String>,
    /// Supported EEA identities (EEA0 included or not, by configuration).
    pub eea: Vec<u8>,
    /// Supported EIA identities.
    pub eia: Vec<u8>,
    /// APN to request; `None` asks for the subscription's default APN.
    pub apn: Option<String>,
    /// PDN type to request.
    pub pdn_type: PdnType,
}

/// The EMM state (§5.1.3.2), the substates this UE distinguishes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmmState {
    /// EMM-DEREGISTERED.NORMAL-SERVICE: an attach is due.
    Deregistered,
    /// EMM-REGISTERED-INITIATED: ATTACH REQUEST sent, T3410 running.
    RegisteredInitiated,
    /// EMM-REGISTERED.
    Registered,
    /// EMM-SERVICE-REQUEST-INITIATED: SERVICE REQUEST sent, T3417 running.
    ServiceRequestInitiated,
    /// EMM-DEREGISTERED.ATTEMPTING-TO-ATTACH: T3411 or T3402 running.
    AttemptingToAttach,
    /// EMM-DEREGISTERED with no further attach: the USIM is invalid for EPS
    /// or the network refused this PLMN/TA (the EMM cause).
    NoService(u8),
}

/// The default EPS bearer context (§6.4.1.3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DefaultBearer {
    /// EPS bearer identity.
    pub ebi: u8,
    /// APN.
    pub apn: String,
    /// The PDN address the network assigned.
    pub pdn_address: PdnAddress,
    /// QCI.
    pub qci: u8,
    /// IPv4 DNS servers the network gave in the PCO.
    pub dns_ipv4: Vec<std::net::Ipv4Addr>,
}

/// What a NAS event asks of the layers below.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NasOutput {
    /// Uplink NAS PDUs to send, in order.
    pub uplink: Vec<Vec<u8>>,
    /// Release the NAS signalling connection locally (§5.5.1.2.6 c).
    pub release: bool,
}

impl NasOutput {
    fn send(pdu: Vec<u8>) -> Self {
        Self {
            uplink: vec![pdu],
            release: false,
        }
    }
}

/// The UE's EPS NAS.
#[derive(Debug)]
pub struct EpsNasUe {
    cfg: EpsNasConfig,
    state: EmmState,
    capability: UeNetworkCapability,
    /// The current EPS security context, after a NAS SMC.
    ctx: Option<EpsSecurityContext>,
    /// `K_ASME` of a completed AKA, waiting for the SMC that takes it into use.
    pending: Option<(NasKsi, [u8; 32])>,
    sqn: SqnState,
    guti: Option<Guti>,
    tai_list: Vec<Tai>,
    t3410: Option<Instant>,
    retry_at: Option<Instant>,
    attempts: u8,
    t3402: Duration,
    next_pti: u8,
    attach_pti: Option<u8>,
    bearer: Option<DefaultBearer>,
    /// The RAND of the last successful challenge with its RES and `K_ASME`,
    /// kept so a retransmitted AUTHENTICATION REQUEST is answered without
    /// the USIM (§5.4.2.3), and the T3416 deadline that bounds it.
    stored_auth: Option<StoredAuth>,
    /// Whether the secure exchange of NAS messages is established on the
    /// current NAS signalling connection (§4.4.5). Only a SECURITY MODE
    /// CONTROL procedure on this connection sets it, and the connection's
    /// release clears it; until then a current context integrity-protects
    /// but does not cipher (§4.4.4.2), so an MME that no longer holds the
    /// context can still read the message (§4.4.4.3 NOTE 3).
    secure_exchange: bool,
    /// Uplink user data is waiting for a connection (§5.6.1.1 b).
    uplink_pending: bool,
    t3417: Option<Instant>,
    t3440: Option<Instant>,
    /// T3346 from a SERVICE REJECT #22: no service request before it expires
    /// (§5.6.1.5).
    t3346: Option<Instant>,
}

/// RAND, RES and `K_ASME` held in the ME's volatile memory (§5.4.2.3).
#[derive(Debug, Clone)]
struct StoredAuth {
    rand: [u8; 16],
    res: Vec<u8>,
    kasme: [u8; 32],
    t3416: Instant,
}

impl EpsNasUe {
    /// A powered-on UE with a USIM: EMM-DEREGISTERED, an attach due.
    pub fn new(cfg: EpsNasConfig) -> Self {
        let capability = UeNetworkCapability::from_algorithms(&cfg.eea, &cfg.eia);
        let sqn = SqnState {
            highest: cfg.initial_sqn,
        };
        Self {
            cfg,
            state: EmmState::Deregistered,
            capability,
            ctx: None,
            pending: None,
            sqn,
            guti: None,
            tai_list: Vec::new(),
            t3410: None,
            retry_at: None,
            attempts: 0,
            t3402: T3402_DEFAULT,
            next_pti: 1,
            attach_pti: None,
            bearer: None,
            stored_auth: None,
            secure_exchange: false,
            uplink_pending: false,
            t3417: None,
            t3440: None,
            t3346: None,
        }
    }

    /// The EMM state.
    pub fn state(&self) -> EmmState {
        self.state
    }

    /// The default EPS bearer, once activated.
    pub fn default_bearer(&self) -> Option<&DefaultBearer> {
        self.bearer.as_ref()
    }

    /// The GUTI, once assigned.
    pub fn guti(&self) -> Option<Guti> {
        self.guti
    }

    /// The TAI list of the last ATTACH ACCEPT.
    pub fn tai_list(&self) -> &[Tai] {
        &self.tai_list
    }

    /// `K_eNB` for the AS Security Mode Command, when a NAS security context
    /// exists.
    pub fn k_enb(&self) -> Option<[u8; 32]> {
        self.ctx.as_ref().map(EpsSecurityContext::k_enb)
    }

    /// The S-TMSI (MME code, M-TMSI) RRC identifies the UE with, when it
    /// holds a GUTI (TS 36.331 §5.3.3.3).
    pub fn s_tmsi(&self) -> Option<(u8, u32)> {
        self.guti.map(|g| (g.mme_code, g.m_tmsi))
    }

    fn alloc_pti(&mut self) -> u8 {
        let p = self.next_pti;
        // §9.4: 1..254 assigned, 0 and 255 reserved.
        self.next_pti = if p >= 254 { 1 } else { p + 1 };
        p
    }

    /// Uplink user data is waiting and there is no connection to send it on.
    /// An EMM-REGISTERED UE then starts the service request procedure on the
    /// next [`Self::initial_message`] (§5.6.1.1 b); in any other state the
    /// data cannot be served and nothing is remembered.
    pub fn on_uplink_data_pending(&mut self) {
        if self.state == EmmState::Registered {
            self.uplink_pending = true;
        }
    }

    /// The lower layers indicate that the user plane radio bearers are set
    /// up: the service request completed successfully (§5.6.1.4).
    pub fn on_user_plane_established(&mut self) {
        if self.state == EmmState::ServiceRequestInitiated {
            log::info!("EMM: user plane radio bearers set up; service request complete (§5.6.1.4)");
            self.t3417 = None;
            self.state = EmmState::Registered;
        }
    }

    /// SERVICE REQUEST (§5.6.1.2), when EMM-REGISTERED with uplink data
    /// pending and not barred by T3346.
    fn service_request(&mut self, now: Instant) -> Option<Vec<u8>> {
        if !self.uplink_pending || self.t3346.is_some_and(|t| now < t) {
            return None;
        }
        self.t3346 = None;
        let Some(ctx) = self.ctx.as_mut() else {
            // A registered UE always holds the context the attach's SMC
            // created; without one there is nothing to compute the short MAC.
            log::warn!("EMM: uplink data pending but no EPS security context; no SERVICE REQUEST");
            self.uplink_pending = false;
            return None;
        };
        let pdu = ctx.service_request();
        self.uplink_pending = false;
        self.state = EmmState::ServiceRequestInitiated;
        self.t3417 = Some(now + T3417);
        log::info!("EMM: SERVICE REQUEST (uplink data pending); T3417 started");
        Some(pdu)
    }

    /// The NAS message that needs an RRC connection now, if any: ATTACH
    /// REQUEST when an attach is due and no retry timer is running, SERVICE
    /// REQUEST when registered with uplink data pending.
    pub fn initial_message(&mut self, now: Instant) -> Option<Vec<u8>> {
        if self.state == EmmState::Registered {
            return self.service_request(now);
        }
        if self.state != EmmState::Deregistered {
            return None;
        }
        if self.retry_at.is_some_and(|t| now < t) {
            return None;
        }
        self.retry_at = None;
        let pti = self.alloc_pti();
        self.attach_pti = Some(pti);
        let esm = PdnConnectivityRequest {
            pti,
            pdn_type: self.cfg.pdn_type,
            apn: self.cfg.apn.clone(),
        }
        .encode();
        // §5.5.1.2.2: a valid GUTI is the identity when the UE has one, else the
        // IMSI; the KSI is the current context's, or "no key is available".
        let identity = match self.guti {
            Some(g) => EpsMobileIdentity::Guti(g),
            None => EpsMobileIdentity::Imsi(self.cfg.imsi.clone()),
        };
        let last_visited_tai = self.guti.and_then(|_| self.tai_list.first().copied());
        let req = AttachRequest {
            ksi: self
                .ctx
                .as_ref()
                .map_or(NasKsi::NO_KEY, EpsSecurityContext::ksi),
            attach_type: EpsAttachType::Eps,
            identity,
            ue_network_capability: self.capability,
            esm_message: esm,
            last_visited_tai,
        }
        .encode();
        // §5.5.1.2.2 / §4.4.4.2: integrity protected (not ciphered) when a
        // current EPS security context exists, plain otherwise.
        let pdu = match &mut self.ctx {
            Some(c) => c.protect(&req, header_type::INTEGRITY),
            None => req,
        };
        self.state = EmmState::RegisteredInitiated;
        self.t3410 = Some(now + T3410);
        log::info!(
            "EMM: ATTACH REQUEST ({}), PTI {pti}; T3410 started",
            if self.guti.is_some() { "GUTI" } else { "IMSI" }
        );
        Some(pdu)
    }

    /// The earliest NAS timer deadline.
    pub fn next_deadline(&self) -> Option<Instant> {
        [
            self.t3410,
            self.t3417,
            self.t3440,
            self.t3346,
            self.retry_at,
            self.stored_auth.as_ref().map(|a| a.t3416),
        ]
        .into_iter()
        .flatten()
        .min()
    }

    /// Timer expiry.
    pub fn on_timer(&mut self, now: Instant) -> NasOutput {
        if self.stored_auth.as_ref().is_some_and(|a| now >= a.t3416) {
            // §5.4.2.3: the stored RAND and RES are deleted on T3416 expiry.
            self.stored_auth = None;
        }
        if self.t3410.is_some_and(|t| now >= t) {
            // §5.5.1.2.6 c): abort the attach, release the NAS signalling
            // connection locally, and handle the attempt counter.
            log::warn!("EMM: T3410 expired; attach aborted (§5.5.1.2.6 c)");
            self.t3410 = None;
            self.attach_failed(now, None);
            self.secure_exchange = false;
            return NasOutput {
                uplink: vec![],
                release: true,
            };
        }
        if self.t3417.is_some_and(|t| now >= t) {
            // §5.6.1.6 c): back to EMM-REGISTERED, the procedure aborted and
            // the resources allocated for it released locally. The next
            // uplink data starts a new service request.
            log::warn!("EMM: T3417 expired; service request aborted (§5.6.1.6 c)");
            self.t3417 = None;
            self.state = EmmState::Registered;
            self.secure_exchange = false;
            return NasOutput {
                uplink: vec![],
                release: true,
            };
        }
        if self.t3440.is_some_and(|t| now >= t) {
            // Table 10.2.1: the network did not release the connection.
            log::warn!("EMM: T3440 expired; NAS signalling connection released locally");
            self.t3440 = None;
            self.secure_exchange = false;
            return NasOutput {
                uplink: vec![],
                release: true,
            };
        }
        if self.t3346.is_some_and(|t| now >= t) {
            self.t3346 = None;
        }
        if self.retry_at.is_some_and(|t| now >= t) && self.state == EmmState::AttemptingToAttach {
            log::info!("EMM: retry timer expired; attach due again");
            self.retry_at = None;
            self.state = EmmState::Deregistered;
        }
        NasOutput::default()
    }

    /// The abnormal-case attempt counter of §5.5.1.2.6: below 5, T3411 and
    /// retry; at 5, T3402 (or the network's value). `wait` overrides both, for
    /// T3346 (#22 congestion).
    fn attach_failed(&mut self, now: Instant, wait: Option<Duration>) {
        // EMM-DEREGISTERED.ATTEMPTING-TO-ATTACH: §5.4.2.3 deletes the stored
        // RAND and RES on entering EMM-DEREGISTERED.
        self.stored_auth = None;
        self.attempts = self.attempts.saturating_add(1);
        let t = wait.unwrap_or(if self.attempts < MAX_ATTACH_ATTEMPTS {
            T3411
        } else {
            self.t3402
        });
        if self.attempts >= MAX_ATTACH_ATTEMPTS {
            self.attempts = 0;
        }
        self.state = EmmState::AttemptingToAttach;
        self.retry_at = Some(now + t);
        log::info!("EMM: next attach attempt in {} s", t.as_secs());
    }

    /// The RRC connection is up.
    pub fn on_connected(&mut self) {}

    /// The RRC connection was released or failed. Before ATTACH ACCEPT or
    /// REJECT this aborts the attach (§5.5.1.2.6 a).
    pub fn on_released(&mut self, now: Instant) {
        if self.state == EmmState::RegisteredInitiated {
            log::warn!("EMM: NAS signalling connection lost during attach (§5.5.1.2.6 a)");
            self.t3410 = None;
            self.attach_failed(now, None);
        }
        if self.state == EmmState::ServiceRequestInitiated {
            // §5.6.1.6 a): lower layer failure or release before the user
            // plane radio bearers are set up aborts the procedure.
            log::warn!("EMM: NAS signalling connection lost during service request (§5.6.1.6 a)");
            self.t3417 = None;
            self.state = EmmState::Registered;
        }
        // Table 10.2.1: T3440 stops when the connection is released.
        self.t3440 = None;
        // AKA results that no SMC took into use do not survive the connection,
        // and §5.4.2.3 deletes the stored RAND and RES on entering EMM-IDLE.
        self.pending = None;
        self.stored_auth = None;
        // §4.4.5: ciphering lasts "until the NAS signalling connection is
        // released"; the next connection needs its own SMC.
        self.secure_exchange = false;
    }

    /// Wrap an uplink EMM message: protected with the current context when
    /// one exists (§4.4.4.2), plain otherwise; ciphered only once the secure
    /// exchange is established on this connection (§4.4.5).
    fn send_emm(&mut self, plain: Vec<u8>) -> NasOutput {
        let ht = if self.secure_exchange {
            header_type::INTEGRITY_CIPHERED
        } else {
            header_type::INTEGRITY
        };
        let pdu = match &mut self.ctx {
            Some(c) => c.protect(&plain, ht),
            None => plain,
        };
        NasOutput::send(pdu)
    }

    /// A downlink NAS PDU.
    pub fn on_downlink(&mut self, pdu: &[u8], now: Instant) -> NasOutput {
        let Some(&first) = pdu.first() else {
            return NasOutput::default();
        };
        match first & 0x0F {
            PD_EMM => {}
            PD_ESM => {
                // §4.4.4.2: an ESM message is only processed inside a
                // security protected EMM message.
                log::warn!("NAS: plain ESM message discarded (§4.4.4.2)");
                return NasOutput::default();
            }
            other => {
                log::warn!("NAS: protocol discriminator {other} discarded");
                return NasOutput::default();
            }
        }
        let ht = security_header_type(pdu).unwrap_or(0);
        match ht {
            header_type::PLAIN => self.on_plain_emm(pdu, now),
            header_type::INTEGRITY_NEW_CONTEXT | header_type::INTEGRITY_CIPHERED_NEW_CONTEXT => {
                self.on_security_mode_command(pdu)
            }
            header_type::INTEGRITY | header_type::INTEGRITY_CIPHERED => {
                let Some(ctx) = self.ctx.as_mut() else {
                    log::warn!("NAS: protected message with no EPS security context; discarded");
                    return NasOutput::default();
                };
                match ctx.unprotect(pdu) {
                    Ok(plain) => self.on_protected_plain(&plain, now),
                    Err(e) => {
                        // §4.4.4.3: discard.
                        log::warn!("NAS: {e}; message discarded (§4.4.4.3)");
                        NasOutput::default()
                    }
                }
            }
            other => {
                log::warn!("NAS: security header type {other} not supported; discarded");
                NasOutput::default()
            }
        }
    }

    fn on_plain_emm(&mut self, pdu: &[u8], now: Instant) -> NasOutput {
        let m = match msg::decode_downlink_emm(pdu) {
            Ok(m) => m,
            Err(e) => {
                log::warn!("EMM: undecodable message: {e}");
                return NasOutput::default();
            }
        };
        // §4.4.4.2: what the UE processes without integrity protection.
        match m {
            DownlinkEmm::IdentityRequest(_)
            | DownlinkEmm::AuthenticationRequest { .. }
            | DownlinkEmm::AuthenticationReject
            | DownlinkEmm::AttachReject(_)
            | DownlinkEmm::ServiceReject(_) => self.on_emm(m, now),
            other => {
                log::warn!("EMM: unprotected {other:?} discarded (§4.4.4.2)");
                NasOutput::default()
            }
        }
    }

    fn on_protected_plain(&mut self, plain: &[u8], now: Instant) -> NasOutput {
        match plain.first().map(|b| b & 0x0F) {
            Some(PD_ESM) => self.on_esm(plain),
            Some(PD_EMM) => match msg::decode_downlink_emm(plain) {
                Ok(m) => self.on_emm(m, now),
                Err(e) => {
                    log::warn!("EMM: undecodable message: {e}");
                    NasOutput::default()
                }
            },
            _ => NasOutput::default(),
        }
    }

    fn on_emm(&mut self, m: DownlinkEmm, now: Instant) -> NasOutput {
        match m {
            DownlinkEmm::IdentityRequest(ty) => {
                // §5.4.4.3; identity type 2 values (§9.9.3.17).
                let id = match ty {
                    1 => MobileIdentity::Imsi(self.cfg.imsi.clone()),
                    3 => self
                        .cfg
                        .imeisv
                        .clone()
                        .map_or(MobileIdentity::NoIdentity, MobileIdentity::Imeisv),
                    2 => self
                        .cfg
                        .imeisv
                        .as_ref()
                        .map(|sv| MobileIdentity::Imei(imei_of(sv)))
                        .unwrap_or(MobileIdentity::NoIdentity),
                    _ => MobileIdentity::NoIdentity,
                };
                log::info!("EMM: IDENTITY REQUEST (type {ty}); IDENTITY RESPONSE");
                self.send_emm(msg::encode_identity_response(&id))
            }
            DownlinkEmm::AuthenticationRequest { ksi, rand, autn } => {
                self.on_authentication_request(ksi, &rand, &autn, now)
            }
            DownlinkEmm::AuthenticationReject => {
                self.stored_auth = None;
                // §5.4.2.5: the USIM is invalid for EPS services; GUTI, TAI
                // list and KSI are deleted; no further attach.
                log::warn!("EMM: AUTHENTICATION REJECT; USIM invalid for EPS (§5.4.2.5)");
                self.invalidate(emm_cause::ILLEGAL_UE);
                NasOutput::default()
            }
            DownlinkEmm::AttachAccept(a) => self.on_attach_accept(a, now),
            DownlinkEmm::AttachReject(r) => {
                self.on_attach_reject(r.cause, r.t3402_s, r.t3346_s, now);
                NasOutput::default()
            }
            DownlinkEmm::ServiceReject(r) => {
                self.on_service_reject(r, now);
                NasOutput::default()
            }
            DownlinkEmm::DetachRequest { detach_type, cause } => {
                self.on_detach_request(detach_type, cause, now)
            }
            DownlinkEmm::SecurityModeCommand(_) => {
                // A SMC with header type 1/2 does not establish a context;
                // §5.4.3.2 has the network send it with header type 3.
                log::warn!("EMM: SECURITY MODE COMMAND without new context header; discarded");
                NasOutput::default()
            }
            DownlinkEmm::EmmInformation => {
                log::info!("EMM: EMM INFORMATION received");
                NasOutput::default()
            }
            DownlinkEmm::EmmStatus(cause) => {
                log::warn!("EMM: EMM STATUS #{cause} from the network");
                NasOutput::default()
            }
            DownlinkEmm::Other(ty) => {
                log::warn!("EMM: message type 0x{ty:02x} not implemented; EMM STATUS #97 (§7.4)");
                self.send_emm(msg::encode_emm_status(
                    emm_cause::MESSAGE_TYPE_NOT_IMPLEMENTED,
                ))
            }
        }
    }

    fn on_authentication_request(
        &mut self,
        ksi: NasKsi,
        rand: &[u8; 16],
        autn: &[u8; 16],
        now: Instant,
    ) -> NasOutput {
        // §5.4.2.3: a RAND equal to the stored one is not passed to the USIM --
        // its SQN is no longer fresh, so the USIM would answer synch failure --
        // and the stored RES is sent again.
        if let Some(stored) = self.stored_auth.as_ref().filter(|a| &a.rand == rand) {
            let res = stored.res.clone();
            self.pending = Some((ksi, stored.kasme));
            log::info!("EMM: retransmitted challenge; stored RES re-sent (§5.4.2.3)");
            return self.send_emm(msg::encode_authentication_response(&res));
        }
        let sn_id = self.cfg.plmn.encode();
        match aka::authenticate(&self.cfg.credentials, &mut self.sqn, rand, autn, &sn_id) {
            AkaOutcome::Success { res, kasme } => {
                // §5.4.2.3: store K_ASME under the KSI; it is taken into use by
                // the SECURITY MODE COMMAND that names that KSI. RAND and RES
                // override any stored pair, and T3416 (re)starts.
                self.pending = Some((ksi, kasme));
                self.stored_auth = Some(StoredAuth {
                    rand: *rand,
                    res: res.to_vec(),
                    kasme,
                    t3416: now + T3416,
                });
                log::info!(
                    "EMM: network authenticated (KSI {}); AUTHENTICATION RESPONSE",
                    ksi.value
                );
                self.send_emm(msg::encode_authentication_response(&res))
            }
            AkaOutcome::MacFailure => {
                log::warn!("EMM: AUTN MAC failure; AUTHENTICATION FAILURE #20");
                self.send_emm(msg::encode_authentication_failure(
                    emm_cause::MAC_FAILURE,
                    None,
                ))
            }
            AkaOutcome::SynchFailure { auts } => {
                log::warn!("EMM: SQN out of range; AUTHENTICATION FAILURE #21 with AUTS");
                self.send_emm(msg::encode_authentication_failure(
                    emm_cause::SYNCH_FAILURE,
                    Some(&auts),
                ))
            }
            AkaOutcome::NonEpsAuthenticationUnacceptable => {
                log::warn!("EMM: AMF separation bit 0; AUTHENTICATION FAILURE #26");
                self.send_emm(msg::encode_authentication_failure(
                    emm_cause::NON_EPS_AUTHENTICATION_UNACCEPTABLE,
                    None,
                ))
            }
        }
    }

    /// SECURITY MODE COMMAND (§5.4.3.3), which arrives with security header
    /// type 3 and is verified with the context it creates.
    fn on_security_mode_command(&mut self, pdu: &[u8]) -> NasOutput {
        // §5.4.2.3: the stored RAND and RES are deleted on receipt of a
        // SECURITY MODE COMMAND.
        self.stored_auth = None;
        if pdu.len() < 6 {
            return NasOutput::default();
        }
        let smc = match msg::decode_downlink_emm(&pdu[6..]) {
            Ok(DownlinkEmm::SecurityModeCommand(s)) => s,
            Ok(other) => {
                log::warn!("EMM: new-context header on {other:?}; discarded");
                return NasOutput::default();
            }
            Err(e) => {
                log::warn!("EMM: undecodable SECURITY MODE COMMAND: {e}");
                return NasOutput::default();
            }
        };
        // The K_ASME the command's KSI names: the pending one from this
        // connection's AKA, or the current context's.
        let kasme = match &self.pending {
            Some((ksi, k)) if ksi.value == smc.ksi.value => Some(*k),
            _ => None,
        };
        let Some(kasme) = kasme else {
            log::warn!(
                "EMM: SECURITY MODE COMMAND names KSI {} with no K_ASME; SECURITY MODE REJECT #24",
                smc.ksi.value
            );
            return NasOutput::send(msg::encode_security_mode_reject(
                emm_cause::SECURITY_MODE_REJECTED_UNSPECIFIED,
            ));
        };
        let supported_eea = self.capability.supports_eea(smc.eea);
        let supported_eia = self.capability.supports_eia(smc.eia);
        let mut candidate = EpsSecurityContext::new(smc.ksi, kasme, smc.eea, smc.eia);
        if !supported_eea || !supported_eia {
            log::warn!(
                "EMM: SECURITY MODE COMMAND selects EEA{}/EIA{} this UE does not support; #24",
                smc.eea,
                smc.eia
            );
            return NasOutput::send(msg::encode_security_mode_reject(
                emm_cause::SECURITY_MODE_REJECTED_UNSPECIFIED,
            ));
        }
        if candidate.unprotect(pdu).is_err() {
            // §5.4.3.5 reject; the integrity check is the command's own.
            log::warn!("EMM: SECURITY MODE COMMAND failed its integrity check; #24");
            return NasOutput::send(msg::encode_security_mode_reject(
                emm_cause::SECURITY_MODE_REJECTED_UNSPECIFIED,
            ));
        }
        // §5.4.3.3: the replayed capabilities must be the ones the UE sent.
        let sent = self.capability.encode();
        if smc.replayed_capabilities.get(..2) != Some(&sent[..]) {
            log::warn!(
                "EMM: replayed UE security capabilities {:02x?} differ from {sent:02x?}; #23",
                smc.replayed_capabilities
            );
            return NasOutput::send(msg::encode_security_mode_reject(
                emm_cause::UE_SECURITY_CAPABILITIES_MISMATCH,
            ));
        }
        let imeisv = if smc.imeisv_requested {
            Some(
                self.cfg
                    .imeisv
                    .clone()
                    .map_or(MobileIdentity::NoIdentity, MobileIdentity::Imeisv),
            )
        } else {
            None
        };
        candidate.mark_security_mode_complete();
        let complete = candidate.protect(
            &msg::encode_security_mode_complete(imeisv.as_ref()),
            header_type::INTEGRITY_CIPHERED_NEW_CONTEXT,
        );
        log::info!(
            "EMM: SECURITY MODE COMMAND EEA{}/EIA{} accepted; SECURITY MODE COMPLETE",
            smc.eea,
            smc.eia
        );
        self.ctx = Some(candidate);
        self.pending = None;
        // §4.4.5: the secure exchange is established on this connection.
        self.secure_exchange = true;
        NasOutput::send(complete)
    }

    fn on_attach_accept(&mut self, a: msg::AttachAccept, now: Instant) -> NasOutput {
        if self.state != EmmState::RegisteredInitiated {
            log::warn!("EMM: ATTACH ACCEPT in {:?}; ignored", self.state);
            return NasOutput::default();
        }
        self.t3410 = None;
        if let Some(g) = a.guti {
            self.guti = Some(g);
        }
        self.tai_list = a.tai_list.clone();
        if let Some(s) = a.t3402_s {
            self.t3402 = Duration::from_secs(s);
        }
        // §5.5.1.2.4 / §6.4.1.3: the ESM container carries ACTIVATE DEFAULT EPS
        // BEARER CONTEXT REQUEST; its accept or reject goes in ATTACH COMPLETE.
        let esm_reply = match msg::decode_downlink_esm(&a.esm_message) {
            Ok(DownlinkEsm::ActivateDefaultBearerRequest(b)) => {
                if Some(b.pti) != self.attach_pti {
                    log::warn!(
                        "ESM: default bearer request with PTI {} not ours; #81",
                        b.pti
                    );
                    msg::encode_activate_default_reject(b.ebi, b.pti, ESM_INVALID_PTI)
                } else {
                    log::info!(
                        "ESM: default EPS bearer {} activated, APN {:?}, PDN address {:?}, QCI {}",
                        b.ebi,
                        b.apn,
                        b.pdn_address,
                        b.qci
                    );
                    let reply = msg::encode_activate_default_accept(b.ebi, b.pti);
                    self.bearer = Some(DefaultBearer {
                        ebi: b.ebi,
                        apn: b.apn,
                        pdn_address: b.pdn_address,
                        qci: b.qci,
                        dns_ipv4: b.dns_ipv4,
                    });
                    reply
                }
            }
            Ok(other) => {
                log::warn!("ESM: ATTACH ACCEPT carries {other:?}; #95");
                msg::encode_esm_status(0, 0, ESM_SEMANTICALLY_INCORRECT)
            }
            Err(e) => {
                log::warn!("ESM: undecodable ESM container: {e}; #95");
                msg::encode_esm_status(0, 0, ESM_SEMANTICALLY_INCORRECT)
            }
        };
        self.state = EmmState::Registered;
        self.attempts = 0;
        self.retry_at = None;
        self.attach_pti = None;
        log::info!(
            "EMM: ATTACH ACCEPT; ATTACH COMPLETE (EMM-REGISTERED){}",
            self.guti
                .map_or(String::new(), |g| format!(", M-TMSI {:#010x}", g.m_tmsi))
        );
        let _ = now;
        self.send_emm(msg::encode_attach_complete(&esm_reply))
    }

    fn invalidate(&mut self, cause: u8) {
        self.stored_auth = None;
        self.guti = None;
        self.tai_list.clear();
        self.ctx = None;
        self.pending = None;
        self.t3410 = None;
        self.retry_at = None;
        self.state = EmmState::NoService(cause);
    }

    fn on_attach_reject(
        &mut self,
        cause: u8,
        t3402_s: Option<u64>,
        t3346_s: Option<u64>,
        now: Instant,
    ) {
        self.t3410 = None;
        if let Some(s) = t3402_s {
            self.t3402 = Duration::from_secs(s);
        }
        log::warn!("EMM: ATTACH REJECT #{cause}");
        match cause {
            // §5.5.1.2.5: the USIM is invalid for EPS services until switched
            // off or the UICC removed; GUTI, TAI list and KSI deleted.
            emm_cause::ILLEGAL_UE
            | emm_cause::ILLEGAL_ME
            | emm_cause::EPS_SERVICES_NOT_ALLOWED
            | emm_cause::EPS_AND_NON_EPS_NOT_ALLOWED => self.invalidate(cause),
            // §5.5.1.2.5: forbidden PLMN / TA lists -- no further attach on this
            // PLMN or in this TA, which for this single-PLMN UE is "no service".
            emm_cause::PLMN_NOT_ALLOWED
            | emm_cause::TA_NOT_ALLOWED
            | emm_cause::ROAMING_NOT_ALLOWED_IN_TA
            | emm_cause::EPS_NOT_ALLOWED_IN_PLMN
            | emm_cause::NO_SUITABLE_CELLS_IN_TA
            | emm_cause::NOT_AUTHORIZED_FOR_CSG
            | emm_cause::SERVICE_OPTION_NOT_AUTHORIZED_IN_PLMN => {
                self.guti = None;
                self.ctx = None;
                self.state = EmmState::NoService(cause);
            }
            // §5.5.1.2.5: #22 congestion with T3346 -> wait that long.
            22 => self.attach_failed(now, t3346_s.map(Duration::from_secs)),
            // Any other cause is abnormal case §5.5.1.2.6 d).
            _ => self.attach_failed(now, None),
        }
    }

    /// SERVICE REJECT (§5.6.1.5).
    fn on_service_reject(&mut self, r: msg::ServiceReject, now: Instant) {
        if self.state != EmmState::ServiceRequestInitiated {
            log::warn!(
                "EMM: SERVICE REJECT #{} in {:?}; ignored",
                r.cause,
                self.state
            );
            return;
        }
        self.t3417 = None;
        log::warn!("EMM: SERVICE REJECT #{}", r.cause);
        match r.cause {
            emm_cause::ILLEGAL_UE
            | emm_cause::ILLEGAL_ME
            | emm_cause::EPS_SERVICES_NOT_ALLOWED
            | emm_cause::EPS_AND_NON_EPS_NOT_ALLOWED => {
                self.bearer = None;
                self.invalidate(r.cause);
                self.t3440 = Some(now + T3440);
            }
            emm_cause::UE_IDENTITY_CANNOT_BE_DERIVED => {
                // #9: delete the GUTI, last visited TAI, TAI list and KSI, enter
                // EMM-DEREGISTERED, deactivate the EPS bearer contexts locally,
                // and attach again.
                log::info!("EMM: identity unknown to the network; GUTI deleted, re-attach");
                self.guti = None;
                self.tai_list.clear();
                self.ctx = None;
                self.local_detach();
                self.t3440 = Some(now + T3440);
            }
            emm_cause::IMPLICITLY_DETACHED | emm_cause::NO_EPS_BEARER_CONTEXT_ACTIVATED => {
                // #10: EMM-DEREGISTERED.NORMAL-SERVICE, EPS bearer contexts
                // deactivated locally, a new attach. #40: the same, as the UE
                // is left with no EPS bearer context.
                log::info!("EMM: EPS bearer contexts deactivated locally, re-attach");
                self.local_detach();
                self.t3440 = Some(now + T3440);
            }
            emm_cause::PLMN_NOT_ALLOWED
            | emm_cause::TA_NOT_ALLOWED
            | emm_cause::ROAMING_NOT_ALLOWED_IN_TA
            | emm_cause::EPS_NOT_ALLOWED_IN_PLMN
            | emm_cause::NO_SUITABLE_CELLS_IN_TA
            | emm_cause::NOT_AUTHORIZED_FOR_CSG
            | emm_cause::SERVICE_OPTION_NOT_AUTHORIZED_IN_PLMN => {
                self.bearer = None;
                self.guti = None;
                self.ctx = None;
                self.state = EmmState::NoService(r.cause);
                self.t3440 = Some(now + T3440);
            }
            emm_cause::CONGESTION => {
                // #22: abort, stay EMM-REGISTERED; no new service request
                // while T3346 runs.
                self.state = EmmState::Registered;
                self.t3346 = r.t3346_s.map(|s| now + Duration::from_secs(s));
            }
            // §5.6.1.6: any other cause aborts the procedure.
            _ => self.state = EmmState::Registered,
        }
    }

    /// DETACH REQUEST from the network (§5.5.2.3.2).
    fn on_detach_request(&mut self, detach_type: u8, cause: Option<u8>, now: Instant) -> NasOutput {
        const REATTACH_REQUIRED: u8 = 1;
        const IMSI_DETACH: u8 = 3;
        if detach_type == IMSI_DETACH {
            // Non-EPS services only; this UE attaches for EPS services alone,
            // so its EPS state is untouched.
            log::info!("EMM: DETACH REQUEST (IMSI detach); DETACH ACCEPT");
            return self.send_emm(msg::encode_detach_accept());
        }
        log::warn!(
            "EMM: DETACH REQUEST from the network (detach type {detach_type}, cause {cause:?}) in \
             {:?}; EPS bearer contexts deactivated locally, DETACH ACCEPT",
            self.state
        );
        // §5.5.2.3.4 e): a detach colliding with the UE's own attach or
        // service request is progressed; those procedures are aborted.
        self.t3410 = None;
        self.t3417 = None;
        // DETACH ACCEPT goes out under the context the request came with,
        // before a cause below deletes that context.
        let out = self.send_emm(msg::encode_detach_accept());
        // "re-attach required": the EMM cause, if any, is ignored.
        let cause = if detach_type == REATTACH_REQUIRED {
            None
        } else {
            cause
        };
        match cause {
            Some(
                c @ (emm_cause::ILLEGAL_UE
                | emm_cause::ILLEGAL_ME
                | emm_cause::EPS_SERVICES_NOT_ALLOWED
                | emm_cause::EPS_AND_NON_EPS_NOT_ALLOWED),
            ) => {
                self.bearer = None;
                self.invalidate(c);
            }
            Some(
                c @ (emm_cause::PLMN_NOT_ALLOWED
                | emm_cause::TA_NOT_ALLOWED
                | emm_cause::ROAMING_NOT_ALLOWED_IN_TA
                | emm_cause::EPS_NOT_ALLOWED_IN_PLMN
                | emm_cause::NO_SUITABLE_CELLS_IN_TA
                | emm_cause::NOT_AUTHORIZED_FOR_CSG
                | emm_cause::SERVICE_OPTION_NOT_AUTHORIZED_IN_PLMN),
            ) => {
                self.bearer = None;
                self.guti = None;
                self.ctx = None;
                self.state = EmmState::NoService(c);
            }
            // Re-attach required, or not required with no (or another) cause:
            // EMM-DEREGISTERED.NORMAL-SERVICE, so an attach follows the
            // release of this connection.
            _ => self.local_detach(),
        }
        self.t3440 = Some(now + T3440);
        out
    }

    /// Deactivate the EPS bearer contexts locally and enter
    /// EMM-DEREGISTERED.NORMAL-SERVICE, so an attach is due.
    fn local_detach(&mut self) {
        self.bearer = None;
        self.uplink_pending = false;
        self.state = EmmState::Deregistered;
        self.attempts = 0;
        self.retry_at = None;
    }

    fn on_esm(&mut self, plain: &[u8]) -> NasOutput {
        match msg::decode_downlink_esm(plain) {
            Ok(DownlinkEsm::EsmInformationRequest { pti }) => {
                // §6.6.1.2: answer with the APN (or none, for the default).
                log::info!("ESM: ESM INFORMATION REQUEST; ESM INFORMATION RESPONSE");
                let apn = self.cfg.apn.clone();
                self.send_emm(msg::encode_esm_information_response(pti, apn.as_deref()))
            }
            Ok(DownlinkEsm::PdnConnectivityReject { pti, cause }) => {
                log::warn!("ESM: PDN CONNECTIVITY REJECT #{cause} (PTI {pti})");
                NasOutput::default()
            }
            Ok(DownlinkEsm::ActivateDefaultBearerRequest(b)) => {
                // Outside ATTACH ACCEPT a default bearer belongs to a UE-requested
                // PDN connectivity procedure this UE does not run (§6.5.1).
                log::warn!("ESM: unsolicited default bearer request; #81");
                self.send_emm(msg::encode_activate_default_reject(
                    b.ebi,
                    b.pti,
                    ESM_INVALID_PTI,
                ))
            }
            Ok(DownlinkEsm::Other(ebi, pti, ty)) => {
                log::warn!("ESM: message type 0x{ty:02x} not implemented; ESM STATUS #97");
                self.send_emm(msg::encode_esm_status(
                    ebi,
                    pti,
                    ESM_MESSAGE_TYPE_NOT_IMPLEMENTED,
                ))
            }
            Err(e) => {
                log::warn!("ESM: undecodable message: {e}");
                NasOutput::default()
            }
        }
    }
}

/// The IMEI of an IMEISV: its first 14 digits plus the Luhn check digit
/// (TS 23.003 §6.2.1-6.2.2).
fn imei_of(imeisv: &str) -> String {
    let body: Vec<u32> = imeisv
        .chars()
        .take(14)
        .filter_map(|c| c.to_digit(10))
        .collect();
    let sum: u32 = body
        .iter()
        .rev()
        .enumerate()
        .map(|(i, d)| {
            if i % 2 == 0 {
                let x = d * 2;
                x / 10 + x % 10
            } else {
                *d
            }
        })
        .sum();
    let check = (10 - sum % 10) % 10;
    let mut s: String = body
        .iter()
        .map(|d| char::from_digit(*d, 10).unwrap_or('0'))
        .collect();
    s.push(char::from_digit(check, 10).unwrap_or('0'));
    s
}

#[cfg(test)]
mod tests;
