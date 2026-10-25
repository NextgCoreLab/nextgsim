//! The UE's EMM and ESM entities for attach (TS 24.301 §5.5.1) and tracking
//! area updating (§5.5.3), the common procedures they run -- identification
//! (§5.4.4), authentication (§5.4.2) and security mode control (§5.4.3) --
//! and the EPS bearer contexts (§6.4).
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
//! - **UE-initiated detach** (§5.5.2.2, #365): DETACH REQUEST ("EPS detach",
//!   normal or switch off) under T3421 with four retransmissions; DETACH
//!   ACCEPT, a lost connection or the fifth expiry leave the UE
//!   EMM-DEREGISTERED with no further attach.
//!
//! - **Tracking area updating** (§5.5.3.2, #356, in `ue/tau.rs`): periodic on
//!   T3412 (§5.3.5) and normal on entering a TA outside the TAI list, with the
//!   EPS bearer context status IE and the "active" flag for pending uplink
//!   data or a paging; TRACKING AREA UPDATE ACCEPT (TAI list, GUTI with TAU
//!   COMPLETE, T3412, bearer reconciliation), REJECT per cause (§5.5.3.2.5) and
//!   the abnormal cases of §5.5.3.2.6 with T3430, T3411, T3402 and the
//!   tracking area updating attempt counter. Combined (EPS/IMSI) updating is
//!   not: this UE attaches for EPS services only.
//!
//! A network message of a procedure this UE does not implement is answered
//! with EMM STATUS #97 (§7.4), which is what the clause asks of a UE that does
//! not implement a message type.

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
/// T3421: UE-initiated detach supervision (§10.2, Table 10.2.1).
pub const T3421: Duration = Duration::from_secs(15);
/// DETACH REQUEST retransmissions on T3421 expiry: "On the first four expiries
/// of the timer, the UE shall retransmit the DETACH REQUEST message"
/// (§5.5.2.2.4 c).
pub const MAX_DETACH_RETRANSMISSIONS: u8 = 4;
/// Type of detach "IMSI detach" in the network-to-UE direction (§9.9.3.7: bits
/// 3-1 = 011).
const NETWORK_DETACH_TYPE_IMSI: u8 = 0b011;
/// Attach attempt counter limit (§5.5.1.2.6: "equal to 5").
pub const MAX_ATTACH_ATTEMPTS: u8 = 5;
/// T3430: tracking area update supervision (§10.2, Table 10.2.1).
pub const T3430: Duration = Duration::from_secs(15);
/// Tracking area updating attempt counter limit (§5.5.3.2.6: "equal to 5").
pub const MAX_TAU_ATTEMPTS: u8 = 5;

/// ESM cause #81 "invalid PTI value" (§9.9.4.4).
const ESM_INVALID_PTI: u8 = 81;
/// ESM cause #31 "request rejected, unspecified".
const ESM_REQUEST_REJECTED_UNSPECIFIED: u8 = 31;
/// ESM cause #36 "regular deactivation".
const ESM_REGULAR_DEACTIVATION: u8 = 36;
/// T3481, the UE-requested bearer resource modification's timer (Table
/// 10.3.2: 8 s).
pub const T3481: Duration = Duration::from_secs(8);
/// BEARER RESOURCE MODIFICATION REQUEST retransmissions before the procedure
/// is aborted on the fifth T3481 expiry (§6.5.4.5 a)).
pub const MAX_BEARER_RESOURCE_MODIFICATION_RETRANSMISSIONS: u8 = 4;
/// ESM cause #43 invalid EPS bearer identity.
const ESM_INVALID_EPS_BEARER_IDENTITY: u8 = 43;
/// ESM cause #95 "semantically incorrect message".
const ESM_SEMANTICALLY_INCORRECT: u8 = 95;
/// ESM cause #96 "invalid mandatory information" (TS 24.301 §9.9.4.4).
const ESM_INVALID_MANDATORY_INFORMATION: u8 = 96;
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
    /// EMM-DEREGISTERED-INITIATED: a UE-initiated DETACH REQUEST is sent (or
    /// waits for a connection), T3421 running (§5.5.2.2.1, #365).
    DeregisteredInitiated,
    /// EMM-DEREGISTERED after a UE-initiated detach (§5.5.2.2.2): the EPS
    /// bearer contexts are gone and no attach follows until asked for.
    Detached,
    /// EMM-TRACKING-AREA-UPDATING-INITIATED: TRACKING AREA UPDATE REQUEST
    /// sent, T3430 running (§5.5.3.2.2, #356).
    TrackingAreaUpdatingInitiated,
    /// EMM-REGISTERED.ATTEMPTING-TO-UPDATE (§5.2.3.2.2): a tracking area
    /// update failed; no user data; the update is tried again on T3411, T3402
    /// or T3346 expiry, or in a new TA.
    AttemptingToUpdate,
    /// EMM-REGISTERED.LIMITED-SERVICE or .PLMN-SEARCH after TRACKING AREA
    /// UPDATE REJECT #13, #15 (§5.5.3.2.5): the bearer contexts stay, the TA is
    /// forbidden, and a TA that is not starts a tracking area update.
    RegisteredLimited(u8),
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

/// A dedicated EPS bearer context (TS 24.301 §6.4.2, #283).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DedicatedBearer {
    /// EPS bearer identity.
    pub ebi: u8,
    /// The default bearer of its PDN connection.
    pub linked_ebi: u8,
    /// QCI.
    pub qci: u8,
    /// The packet filters of its TFT.
    pub filters: Vec<crate::tft::PacketFilter>,
}

/// A UE-requested bearer resource modification in progress (TS 24.301
/// §6.5.4, #408): PROCEDURE TRANSACTION PENDING on `pti`, T3481 running.
#[derive(Debug, Clone)]
struct BearerResourceModification {
    pti: u8,
    /// The EPS bearer of the traffic flow aggregate.
    ebi: u8,
    /// The BEARER RESOURCE MODIFICATION REQUEST, plain, for retransmission.
    request: Vec<u8>,
    /// The request releases every traffic flow of the bearer, which a fifth
    /// T3481 expiry or a REJECT #31 then deactivates locally (§6.5.4.5 a), d)).
    releases_all: bool,
    retransmissions: u8,
    t3481: Instant,
}

/// The outcome of the §6.4.2.4 d) 2) / §6.4.3.4 d) 2) precedence check of a
/// TFT a bearer gets.
enum PrecedenceClash {
    /// An old filter of the default bearer has a precedence of the new TFT:
    /// the PDN connection is released. Nothing was changed.
    DefaultBearer,
    /// The old filters of other dedicated bearers with those precedences were
    /// deleted, as (EBI, packet filter identifiers, whether none is left).
    Deleted(Vec<(u8, Vec<u8>, bool)>),
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
    /// The packet filters of the default bearer's TFT, when a MODIFY gave it
    /// one; empty is match-all (§6.4.3.4 a) 3)).
    default_filters: Vec<crate::tft::PacketFilter>,
    /// Dedicated EPS bearer contexts (#283).
    dedicated: Vec<DedicatedBearer>,
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
    /// Paged with the S-TMSI and not answered yet (§5.6.2.2.1, #351).
    paging_response_pending: bool,
    /// The service request in progress answers a paging: the RRC establishment
    /// cause is mt-Access (Annex D, #351).
    service_request_for_paging: bool,
    /// A UE-initiated detach waiting for a connection: the DETACH REQUEST is the
    /// next [`EpsNasUe::initial_message`] (#365). Holds the switch-off bit.
    detach_pending: Option<bool>,
    /// The DETACH REQUEST in progress (unprotected) and how often it has been
    /// retransmitted (§5.5.2.2.4 c), for T3421.
    detach_request: Option<(Vec<u8>, u8)>,
    t3421: Option<Instant>,
    /// T3412 as the network last set it (§5.3.5); `None` is deactivated.
    t3412_value: Option<Duration>,
    /// The running T3412 (#356).
    t3412: Option<Instant>,
    /// T3430, while a TRACKING AREA UPDATE REQUEST is unanswered.
    t3430: Option<Instant>,
    /// T3411 or T3402 before the next tracking area update attempt
    /// (§5.5.3.2.6).
    tau_retry_at: Option<Instant>,
    /// The tracking area updating attempt counter (§5.5.3.1).
    tau_attempts: u8,
    /// T3412 expired: a periodic update is owed (§5.3.5: delayed until
    /// EMM-REGISTERED.NORMAL-SERVICE).
    tau_periodic_due: bool,
    /// A normal ("TA updating") update is owed: a TA outside the TAI list
    /// (§5.5.3.2.2 a), or the retry of one that failed.
    tau_ta_due: bool,
    /// The TRACKING AREA UPDATE REQUEST in progress: its update type and
    /// "active" flag.
    tau_in_progress: Option<(msg::EpsUpdateType, bool)>,
    /// The TRACKING AREA UPDATE REQUEST in progress answers a paging: the
    /// RRC establishment cause is MT access (Annex D Table D.1.1).
    tau_answers_paging: bool,
    /// EPS update status EU1 UPDATED (§5.1.3.3); false is EU2 NOT UPDATED or
    /// EU3 ROAMING NOT ALLOWED.
    updated: bool,
    /// The TAI of the cell the UE camps on, as the lower layers report it.
    current_tai: Option<Tai>,
    /// The last visited registered TAI (§5.5.3.2.2, IEI 52).
    last_visited_tai: Option<Tai>,
    /// The lists of forbidden tracking areas "for roaming" and "for regional
    /// provision of service" (§5.5.3.2.5), merged: this UE treats both alike.
    forbidden_tais: Vec<Tai>,
    /// A UE-initiated detach (not switch-off) asked for during a tracking area
    /// update, sent once it completes (§5.5.3.2.6 m).
    detach_after_tau: bool,
    /// The detach in progress is the "detach and re-attach" of §6.4.2.4 d) 2)
    /// (#408): it ends in EMM-DEREGISTERED.NORMAL-SERVICE, with an attach due.
    reattach_after_detach: bool,
    /// UE-requested bearer resource modifications in progress (#408).
    bearer_resource_modifications: Vec<BearerResourceModification>,
}

/// How the network paged the UE (TS 24.301 §5.6.2.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PagedWith {
    /// The S-TMSI (§5.6.2.2.1): answer with a SERVICE REQUEST.
    STmsi,
    /// The IMSI (§5.6.2.2.2): the network lost the UE's context.
    Imsi,
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
            default_filters: Vec::new(),
            dedicated: Vec::new(),
            stored_auth: None,
            secure_exchange: false,
            uplink_pending: false,
            t3417: None,
            t3440: None,
            t3346: None,
            paging_response_pending: false,
            service_request_for_paging: false,
            detach_pending: None,
            detach_request: None,
            t3421: None,
            t3412_value: None,
            t3412: None,
            t3430: None,
            tau_retry_at: None,
            tau_attempts: 0,
            tau_periodic_due: false,
            tau_ta_due: false,
            tau_in_progress: None,
            tau_answers_paging: false,
            updated: false,
            current_tai: None,
            last_visited_tai: None,
            forbidden_tais: Vec::new(),
            detach_after_tau: false,
            reattach_after_detach: false,
            bearer_resource_modifications: Vec::new(),
        }
    }

    /// UE-initiated detach for EPS services (TS 24.301 §5.5.2.2, #365).
    ///
    /// - Not registered: nothing to detach from. A UE between attach attempts
    ///   stops attaching (EMM-DEREGISTERED, [`EmmState::Detached`]).
    /// - Registered with a NAS signalling connection (`connected`): the DETACH
    ///   REQUEST is returned now.
    /// - Registered in EMM-IDLE: the DETACH REQUEST is the next
    ///   [`Self::initial_message`], which the RRC sends when it establishes a
    ///   connection for it.
    ///
    /// "If the detach is not due to switch off and the UE is in the state
    /// EMM-REGISTERED or EMM-REGISTERED-INITIATED, timer T3421 shall be started
    /// in the UE after the DETACH REQUEST message has been sent", and the UE
    /// enters EMM-DEREGISTERED-INITIATED (§5.5.2.2.1). On a switch-off detach
    /// the network sends no DETACH ACCEPT (§5.5.2.2.2), so the UE deactivates
    /// its bearer contexts and is EMM-DEREGISTERED as soon as the request is out.
    pub fn detach(&mut self, switch_off: bool, connected: bool, now: Instant) -> NasOutput {
        match self.state {
            EmmState::Registered
            | EmmState::ServiceRequestInitiated
            | EmmState::AttemptingToUpdate
            | EmmState::RegisteredLimited(_) => {}
            EmmState::TrackingAreaUpdatingInitiated if switch_off => {
                // §5.5.3.2.6 m): "Detach due to ... switch off: The tracking
                // area updating procedure shall be aborted, and the UE
                // initiated detach procedure shall be performed", with the
                // GUTI the TRACKING AREA UPDATE REQUEST carried.
                log::info!("EMM: switch-off detach during a tracking area update; TAU aborted");
                self.t3430 = None;
                self.tau_in_progress = None;
            }
            EmmState::TrackingAreaUpdatingInitiated => {
                // §5.5.3.2.6 m): any other detach "shall be initiated after
                // successful completion of the tracking area updating
                // procedure".
                log::info!("EMM: detach requested during a tracking area update; after it");
                self.detach_after_tau = true;
                return NasOutput::default();
            }
            EmmState::RegisteredInitiated if connected => {
                // §5.5.2.2.1: the DETACH REQUEST goes, with T3421, "in the state
                // EMM-REGISTERED or EMM-REGISTERED-INITIATED". The attach is
                // abandoned for it, so the MME is told rather than left with an
                // attach in progress (#369 review).
                log::info!("EMM: detach requested during attach; T3410 stopped, DETACH REQUEST");
                self.t3410 = None;
                self.attach_pti = None;
                return self.send_detach_request(switch_off, now);
            }
            EmmState::RegisteredInitiated => {
                // No connection to send it on: the attach ends locally.
                log::info!(
                    "EMM: detach requested during attach without a connection; attach aborted"
                );
                self.t3410 = None;
                self.attach_pti = None;
                self.enter_detached();
                return NasOutput {
                    uplink: vec![],
                    release: true,
                };
            }
            EmmState::Deregistered | EmmState::AttemptingToAttach => {
                log::info!("EMM: detach requested while not attached; no further attach");
                self.enter_detached();
                return NasOutput::default();
            }
            EmmState::DeregisteredInitiated if self.reattach_after_detach => {
                // The detach-and-re-attach of a precedence clash is in progress
                // (#408): the user's detach wins over the re-attach (#417
                // review). A switch-off goes as a switch-off DETACH REQUEST
                // (§5.5.2.2.1), after which the UE is EMM-DEREGISTERED.
                log::info!("EMM: detach requested during the re-attach detach; no re-attach");
                self.reattach_after_detach = false;
                if !switch_off {
                    return NasOutput::default();
                }
                if connected {
                    return self.send_detach_request(true, now);
                }
                self.enter_detached();
                return NasOutput {
                    uplink: vec![],
                    release: true,
                };
            }
            EmmState::DeregisteredInitiated | EmmState::Detached | EmmState::NoService(_) => {
                return NasOutput::default();
            }
        }
        self.t3417 = None;
        self.uplink_pending = false;
        self.paging_response_pending = false;
        if connected {
            self.send_detach_request(switch_off, now)
        } else {
            log::info!("EMM: detach requested in EMM-IDLE; DETACH REQUEST on the next connection");
            self.detach_pending = Some(switch_off);
            self.state = EmmState::DeregisteredInitiated;
            NasOutput::default()
        }
    }

    /// Whether a UE-initiated detach is waiting for a connection (#365).
    pub fn detach_pending(&self) -> bool {
        self.detach_pending.is_some()
    }

    /// Build, protect and record the DETACH REQUEST (§5.5.2.2.1).
    fn detach_request_pdu(&mut self, switch_off: bool) -> Vec<u8> {
        // "If the UE has a valid GUTI, the UE shall populate the EPS mobile
        // identity IE with the valid GUTI. If the UE does not have a valid GUTI,
        // the UE shall populate the EPS mobile identity IE with its IMSI."
        let identity = match self.guti {
            Some(g) => EpsMobileIdentity::Guti(g),
            None => EpsMobileIdentity::Imsi(self.cfg.imsi.clone()),
        };
        let ksi = self
            .ctx
            .as_ref()
            .map_or(NasKsi::NO_KEY, EpsSecurityContext::ksi);
        msg::encode_detach_request(switch_off, ksi, &identity)
    }

    fn send_detach_request(&mut self, switch_off: bool, now: Instant) -> NasOutput {
        let plain = self.detach_request_pdu(switch_off);
        let out = self.send_emm(plain.clone());
        if switch_off {
            log::info!("EMM: DETACH REQUEST (switch off); EPS bearer contexts deactivated");
            self.enter_detached();
        } else {
            log::info!("EMM: DETACH REQUEST (EPS detach); T3421 started");
            self.state = EmmState::DeregisteredInitiated;
            self.detach_request = Some((plain, 0));
            self.t3421 = Some(now + T3421);
        }
        out
    }

    /// EMM-DEREGISTERED after a UE-initiated detach: the EPS bearer contexts are
    /// deactivated locally (§5.5.2.2.2) and no attach follows.
    fn enter_detached(&mut self) {
        self.clear_bearers();
        self.stop_tau();
        self.uplink_pending = false;
        self.paging_response_pending = false;
        self.detach_pending = None;
        self.detach_request = None;
        self.t3421 = None;
        self.retry_at = None;
        self.state = EmmState::Detached;
        if std::mem::take(&mut self.reattach_after_detach) {
            // §6.4.2.4 d) 2): "the UE shall detach and re-attach to the
            // network": EMM-DEREGISTERED.NORMAL-SERVICE, so the attach is the
            // next initial message (#408).
            log::info!("EMM: detached to re-attach; attach due");
            self.state = EmmState::Deregistered;
            self.attempts = 0;
        }
    }

    /// The UE was paged for the PS domain (TS 24.301 §5.6.2.2, #351). Returns
    /// whether NAS now needs an RRC connection.
    ///
    /// - S-TMSI, EMM-REGISTERED in EMM-IDLE (§5.6.2.2.1): "Upon reception of a
    ///   paging indication, ... the UE shall stop the timer T3346, if running,
    ///   and shall initiate ... a service request procedure to respond to the
    ///   paging". It follows on the next [`Self::initial_message`], with
    ///   establishment cause mt-Access.
    /// - IMSI (§5.6.2.2.2): "the UE shall stop timer T3346, if it is running,
    ///   locally deactivate any EPS bearer context(s), if any, and locally
    ///   detach from EPS. Additionally, the UE shall delete the following
    ///   parameters: last visited registered TAI, TAI list, GUTI and KSI_(ASME)",
    ///   then attaches. A UE-initiated EMM procedure in progress is aborted.
    ///   A UE with no EPS service (an invalid USIM, a refused PLMN) stays as it
    ///   is.
    pub fn on_paging(&mut self, with: PagedWith) -> bool {
        match with {
            PagedWith::STmsi if self.state == EmmState::AttemptingToUpdate => {
                // §5.5.3.2.2 r): "upon reception of a paging indication using
                // S-TMSI and the UE is in state
                // EMM-REGISTERED.ATTEMPTING-TO-UPDATE" a tracking area update,
                // with the "active" flag set.
                log::info!(
                    "EMM: paged with the S-TMSI in ATTEMPTING-TO-UPDATE; TAU (§5.5.3.2.2 r)"
                );
                self.t3346 = None;
                self.paging_response_pending = true;
                self.tau_ta_due = true;
                true
            }
            PagedWith::STmsi => {
                if self.state != EmmState::Registered {
                    log::info!(
                        "EMM: paged with the S-TMSI in {:?}; not answered",
                        self.state
                    );
                    return false;
                }
                log::info!("EMM: paged with the S-TMSI; SERVICE REQUEST (§5.6.2.2.1)");
                self.t3346 = None;
                self.paging_response_pending = true;
                true
            }
            PagedWith::Imsi => {
                if matches!(self.state, EmmState::NoService(_)) {
                    return false;
                }
                log::info!("EMM: paged with the IMSI; local detach, then attach (§5.6.2.2.2)");
                self.t3346 = None;
                self.t3410 = None;
                self.t3417 = None;
                self.attach_pti = None;
                self.local_detach();
                self.guti = None;
                self.tai_list.clear();
                self.ctx = None;
                self.secure_exchange = false;
                true
            }
        }
    }

    /// Whether the service request in progress answers a paging (#351): the
    /// RRC establishment cause is then mt-Access (Annex D Table D.1.1).
    pub fn service_request_answers_paging(&self) -> bool {
        self.state == EmmState::ServiceRequestInitiated && self.service_request_for_paging
    }

    /// Whether the initial NAS message in progress answers a paging: a
    /// SERVICE REQUEST (#351) or a TRACKING AREA UPDATE REQUEST (#356 review).
    /// Annex D Table D.1.1: the RRC establishment cause is then MT access.
    pub fn initial_message_answers_paging(&self) -> bool {
        self.service_request_answers_paging()
            || (self.state == EmmState::TrackingAreaUpdatingInitiated && self.tau_answers_paging)
    }

    /// The EMM state.
    pub fn state(&self) -> EmmState {
        self.state
    }

    /// The default EPS bearer, once activated.
    pub fn default_bearer(&self) -> Option<&DefaultBearer> {
        self.bearer.as_ref()
    }

    /// The active dedicated EPS bearers (#283).
    pub fn dedicated_bearers(&self) -> &[DedicatedBearer] {
        &self.dedicated
    }

    /// The EPS bearer an uplink IP packet goes on (TS 23.401 §5.4.1, #283):
    /// the bearer of the matching uplink packet filter with the lowest
    /// evaluation precedence among all TFTs, the default bearer's included;
    /// with no match, the bearer that has no uplink packet filter. `None`
    /// when every bearer has one -- the packet is discarded -- or there is no
    /// default bearer.
    pub fn uplink_bearer_for(&self, packet: &[u8]) -> Option<u8> {
        let default = self.bearer.as_ref()?.ebi;
        let tfts = || {
            std::iter::once((default, &self.default_filters))
                .chain(self.dedicated.iter().map(|b| (b.ebi, &b.filters)))
        };
        let matched = crate::tft::PacketKey::parse(packet).and_then(|key| {
            tfts()
                .flat_map(|(ebi, fs)| fs.iter().map(move |f| (f, ebi)))
                .filter(|(f, _)| f.matches_uplink(&key))
                .min_by_key(|(f, _)| f.precedence)
                .map(|(_, ebi)| ebi)
        });
        matched.or_else(|| {
            tfts()
                .find(|(_, fs)| !fs.iter().any(|f| f.direction.applies_uplink()))
                .map(|(ebi, _)| ebi)
        })
    }

    /// Drop every EPS bearer context, default and dedicated.
    fn clear_bearers(&mut self) {
        self.bearer = None;
        self.default_filters.clear();
        self.dedicated.clear();
        self.bearer_resource_modifications.clear();
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
    /// The UE's IMSI, as configured (#351: an IMSI paging record is compared
    /// with it).
    pub fn imsi(&self) -> &str {
        &self.cfg.imsi
    }

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
    ///
    /// On a connection a SERVICE REQUEST set up, this indication is also what
    /// establishes the secure exchange of NAS messages (§4.4.2.3 item 2): from
    /// here on every uplink NAS message is ciphered (§4.4.5), and a receiver
    /// discards one that should have been but was not (#318).
    pub fn on_user_plane_established(&mut self) {
        if self.state == EmmState::ServiceRequestInitiated {
            log::info!("EMM: user plane radio bearers set up; service request complete (§5.6.1.4)");
            self.t3417 = None;
            self.state = EmmState::Registered;
            if self.ctx.is_some() {
                self.secure_exchange = true;
            }
        }
    }

    /// SERVICE REQUEST (§5.6.1.2), when EMM-REGISTERED with uplink data
    /// pending and not barred by T3346.
    fn service_request(&mut self, now: Instant) -> Option<Vec<u8>> {
        let paging = self.paging_response_pending;
        // T3346 bars a data-triggered service request (§5.6.1.5); a paging
        // response already stopped it (§5.6.2.2.1).
        if !paging && (!self.uplink_pending || self.t3346.is_some_and(|t| now < t)) {
            return None;
        }
        self.t3346 = None;
        self.paging_response_pending = false;
        self.service_request_for_paging = paging;
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
        log::info!(
            "EMM: SERVICE REQUEST ({}); T3417 started",
            if paging {
                "paging response"
            } else {
                "uplink data pending"
            }
        );
        Some(pdu)
    }

    /// The NAS message that needs an RRC connection now, if any: ATTACH
    /// REQUEST when an attach is due and no retry timer is running, SERVICE
    /// REQUEST when registered with uplink data pending.
    pub fn initial_message(&mut self, now: Instant) -> Option<Vec<u8>> {
        if let Some(switch_off) = self.detach_pending.take() {
            // §4.4.4.2: integrity protected, not ciphered, as an initial NAS
            // message (the secure exchange is not established yet).
            return self
                .send_detach_request(switch_off, now)
                .uplink
                .into_iter()
                .next();
        }
        if let Some(tau) = self.tau_request(now) {
            return Some(tau);
        }
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
        let last_visited_tai = self.guti.and_then(|_| {
            self.last_visited_tai
                .or_else(|| self.tai_list.first().copied())
        });
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
            self.t3421,
            self.t3412,
            self.t3430,
            self.tau_retry_at,
            self.retry_at,
            self.bearer_resource_modifications
                .iter()
                .map(|m| m.t3481)
                .min(),
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
        if self.t3421.is_some_and(|t| now >= t) {
            // §5.5.2.2.4 c): "On the first four expiries of the timer, the UE
            // shall retransmit the DETACH REQUEST message and shall reset and
            // restart timer T3421. On the fifth expiry of timer T3421, the
            // detach procedure shall be aborted" and the UE enters
            // EMM-DEREGISTERED.
            if let Some((plain, sent)) = self.detach_request.take() {
                if sent < MAX_DETACH_RETRANSMISSIONS {
                    log::warn!(
                        "EMM: T3421 expired; DETACH REQUEST retransmitted ({})",
                        sent + 1
                    );
                    self.detach_request = Some((plain.clone(), sent + 1));
                    self.t3421 = Some(now + T3421);
                    return self.send_emm(plain);
                }
            }
            log::warn!(
                "EMM: T3421 expired a fifth time; detach aborted, EMM-DEREGISTERED (§5.5.2.2.4 c)"
            );
            self.enter_detached();
            return NasOutput {
                uplink: vec![],
                release: true,
            };
        }
        if self.t3430.is_some_and(|t| now >= t) {
            // §5.5.3.2.6 c): "The UE shall abort the procedure. The NAS
            // signalling connection, if any, shall be released locally."
            log::warn!("EMM: T3430 expired; tracking area update aborted (§5.5.3.2.6 c)");
            self.tau_failed(now, false);
            self.secure_exchange = false;
            return NasOutput {
                uplink: vec![],
                release: true,
            };
        }
        self.on_tau_timers(now);
        if self.t3346.is_some_and(|t| now >= t) {
            self.t3346 = None;
        }
        if self
            .bearer_resource_modifications
            .iter()
            .any(|m| now >= m.t3481)
        {
            return self.on_t3481(now);
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

    /// The RRC connection is up: EMM-CONNECTED, so T3412 stops (§5.3.5).
    pub fn on_connected(&mut self) {
        self.t3412 = None;
    }

    /// The RRC connection was released or failed. Before ATTACH ACCEPT or
    /// REJECT this aborts the attach (§5.5.1.2.6 a).
    pub fn on_released(&mut self, now: Instant) {
        if self.state == EmmState::RegisteredInitiated {
            log::warn!("EMM: NAS signalling connection lost during attach (§5.5.1.2.6 a)");
            self.t3410 = None;
            self.attach_failed(now, None);
        }
        if self.state == EmmState::DeregisteredInitiated && self.detach_request.is_some() {
            // §5.5.2.2.4 b): "Lower layer failure or release of the NAS
            // signalling connection before reception of DETACH ACCEPT message":
            // the detach is aborted and, for "EPS detach", the UE enters
            // EMM-DEREGISTERED.
            log::warn!("EMM: NAS signalling connection lost before DETACH ACCEPT; EMM-DEREGISTERED (§5.5.2.2.4 b)");
            self.enter_detached();
        }
        if self.state == EmmState::TrackingAreaUpdatingInitiated {
            // §5.5.3.2.6 b): lower layer failure, or release of the NAS
            // signalling connection, before TRACKING AREA UPDATE ACCEPT or
            // REJECT aborts the procedure.
            log::warn!(
                "EMM: NAS signalling connection lost during a tracking area update (§5.5.3.2.6 b)"
            );
            self.tau_failed(now, false);
        }
        if self.state == EmmState::ServiceRequestInitiated {
            // §5.6.1.6 a): lower layer failure or release before the user
            // plane radio bearers are set up aborts the procedure.
            log::warn!("EMM: NAS signalling connection lost during service request (§5.6.1.6 a)");
            self.t3417 = None;
            self.state = EmmState::Registered;
        }
        // A paging response that never went out (the establishment failed or
        // the connection went before the SERVICE REQUEST) is not owed any
        // more: the network pages again on T3413 (§5.6.2.2.1). Kept, it would
        // label the next data-triggered SERVICE REQUEST mt-Access and let it
        // past T3346 (#351 review).
        self.paging_response_pending = false;
        // Table 10.2.1: T3440 stops when the connection is released.
        self.t3440 = None;
        // AKA results that no SMC took into use do not survive the connection,
        // and §5.4.2.3 deletes the stored RAND and RES on entering EMM-IDLE.
        self.pending = None;
        self.stored_auth = None;
        // §4.4.5: ciphering lasts "until the NAS signalling connection is
        // released"; the next connection needs its own SMC.
        self.secure_exchange = false;
        // §5.3.5: "Timer T3412 is reset and started with its initial value,
        // when the UE changes from EMM-CONNECTED ... to EMM-IDLE mode."
        self.start_t3412(now);
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
        // §4.4.4.2: "Once the secure exchange of NAS messages has been
        // established, the receiving EMM or ESM entity in the UE shall not
        // process any NAS signalling messages unless they have been
        // successfully integrity checked". The allow-list below is for a
        // connection without one; on a secure exchange a plain message is
        // discarded unread, and is not answered either (#400 review).
        if self.secure_exchange {
            log::warn!("EMM: plain message on a secure exchange; discarded (§4.4.4.2)");
            return NasOutput::default();
        }
        let m = match msg::decode_downlink_emm(pdu) {
            Ok(m) => m,
            Err(e) => {
                // §7.5.1: a mandatory-IE error is answered with EMM STATUS #96,
                // but only for a message the UE would process unprotected
                // (§4.4.4.2); any other plain message is discarded unread (#382).
                let allowed = matches!(
                    pdu.get(1).copied(),
                    Some(
                        msg::emm_type::IDENTITY_REQUEST
                            | msg::emm_type::AUTHENTICATION_REQUEST
                            | msg::emm_type::AUTHENTICATION_REJECT
                            | msg::emm_type::ATTACH_REJECT
                            | msg::emm_type::TAU_REJECT
                            | msg::emm_type::SERVICE_REJECT
                    )
                );
                if allowed {
                    log::warn!("EMM: undecodable message: {e}; EMM STATUS #96 (§7.5.1)");
                    return self.emm_status_96();
                }
                log::warn!("EMM: undecodable unprotected message: {e}; discarded (§4.4.4.2)");
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
            // §4.4.4.2: "TRACKING AREA UPDATE REJECT (if the EMM cause is not
            // #25 or #78)"; §5.5.3.2.5: one with #25 or #78 "received without
            // integrity protection" is discarded.
            DownlinkEmm::TauReject(r) if !matches!(r.cause, 25 | 78) => {
                self.on_tau_reject(r, false, now)
            }
            other => {
                log::warn!("EMM: unprotected {other:?} discarded (§4.4.4.2)");
                NasOutput::default()
            }
        }
    }

    fn on_protected_plain(&mut self, plain: &[u8], now: Instant) -> NasOutput {
        match plain.first().map(|b| b & 0x0F) {
            Some(PD_ESM) => self.on_esm(plain, now),
            Some(PD_EMM) => match msg::decode_downlink_emm(plain) {
                Ok(m) => self.on_emm(m, now),
                Err(e) => {
                    // §7.5.1, §7.5.2 (no EMM exception): EMM STATUS #96 (#382)
                    log::warn!("EMM: undecodable message: {e}; EMM STATUS #96 (§7.5.1)");
                    self.emm_status_96()
                }
            },
            _ => NasOutput::default(),
        }
    }

    /// EMM STATUS with cause #96 "invalid mandatory information" (TS 24.301
    /// §7.5.1, §8.2.14): the answer to an EMM message with an imperative-part
    /// error, a missing or syntactically incorrect mandatory IE, or an unknown
    /// "comprehension required" IE. §7.5.2 lists no EMM exception, so the
    /// procedure the message belonged to is left as it was. Protected like any
    /// other uplink when a context is active. Before #382 the UE logged the
    /// error and sent nothing.
    fn emm_status_96(&mut self) -> NasOutput {
        self.send_emm(msg::encode_emm_status(
            emm_cause::INVALID_MANDATORY_INFORMATION,
        ))
    }

    /// The ESM answer to an ESM message with a mandatory-IE error (TS 24.301
    /// §7.5.1, §7.5.3): the REJECT #96 of an ACTIVATE DEFAULT / ACTIVATE
    /// DEDICATED / MODIFY EPS BEARER CONTEXT REQUEST (§7.5.3 a)-c)); for a
    /// DEACTIVATE EPS BEARER CONTEXT REQUEST, the bearer is released and the
    /// ACCEPT returned (d), #283); ESM STATUS #96 otherwise. The EBI and PTI
    /// are read from the header octets when they are there.
    fn esm_invalid_mandatory_reply(&mut self, esm: &[u8]) -> Vec<u8> {
        use msg::esm_type;
        let ebi = esm.first().map_or(0, |b| b >> 4);
        let pti = esm.get(1).copied().unwrap_or(0);
        let cause = ESM_INVALID_MANDATORY_INFORMATION;
        match esm.get(2).copied() {
            Some(esm_type::ACTIVATE_DEFAULT_REQUEST) => {
                msg::encode_activate_default_reject(ebi, pti, cause)
            }
            Some(esm_type::ACTIVATE_DEDICATED_REQUEST) => {
                msg::encode_activate_dedicated_reject(ebi, pti, cause)
            }
            Some(esm_type::MODIFY_REQUEST) => msg::encode_modify_reject(ebi, pti, cause),
            Some(esm_type::DEACTIVATE_REQUEST) => {
                self.on_deactivate_bearer(ebi, cause);
                msg::encode_deactivate_accept(ebi, pti)
            }
            _ => msg::encode_esm_status(ebi, pti, cause),
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
            DownlinkEmm::TauAccept(a) => self.on_tau_accept(a, now),
            DownlinkEmm::TauReject(r) => self.on_tau_reject(r, true, now),
            DownlinkEmm::Other(msg::emm_type::GUTI_REALLOCATION_COMMAND)
                if self.state == EmmState::TrackingAreaUpdatingInitiated =>
            {
                // §5.5.3.2.6 g): ignored, the tracking area update progresses.
                log::info!("EMM: GUTI REALLOCATION COMMAND during a tracking area update; ignored");
                NasOutput::default()
            }
            DownlinkEmm::DetachAccept => {
                if self.state == EmmState::DeregisteredInitiated && self.detach_request.is_some() {
                    // §5.5.2.2.2: "The UE, when receiving the DETACH ACCEPT
                    // message, shall stop timer T3421", deactivates its EPS
                    // bearer contexts locally and enters EMM-DEREGISTERED.
                    log::info!("EMM: DETACH ACCEPT; detached, EMM-DEREGISTERED (§5.5.2.2.2)");
                    self.enter_detached();
                } else {
                    log::warn!("EMM: DETACH ACCEPT with no detach in progress; ignored");
                }
                NasOutput::default()
            }
            DownlinkEmm::DetachRequest {
                detach_type: NETWORK_DETACH_TYPE_IMSI,
                cause,
            } if self.state == EmmState::DeregisteredInitiated && self.detach_request.is_some() => {
                // §5.5.2.2.4 d): a network "IMSI detach" while this UE runs an
                // "EPS detach": "the UE shall progress both procedures". The
                // network's is answered; this UE's own detach goes on, T3421 with
                // it (#369 review). This UE has no CS domain to detach from.
                log::info!(
                    "EMM: network IMSI DETACH REQUEST (cause {cause:?}) during this UE's detach; \
                     DETACH ACCEPT, own detach continues (§5.5.2.2.4 d)"
                );
                self.send_emm(msg::encode_detach_accept())
            }
            DownlinkEmm::DetachRequest { detach_type, cause }
                if self.state == EmmState::DeregisteredInitiated
                    && self.detach_request.is_some() =>
            {
                // §5.5.2.2.4 d), a network detach colliding with this UE's
                // "EPS detach": it is treated per §5.5.2.3.2, except that "the UE
                // need not initiate the attach". The UE answers DETACH ACCEPT
                // and is detached.
                log::info!(
                    "EMM: network DETACH REQUEST (type {detach_type}, cause {cause:?}) during \
                     this UE's detach; DETACH ACCEPT, detached (§5.5.2.2.4 d)"
                );
                // "the UE need not initiate the attach": the network's detach
                // overrides a re-attach this UE's own detach was for (#417
                // review).
                self.reattach_after_detach = false;
                self.enter_detached();
                self.send_emm(msg::encode_detach_accept())
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
                // §7.5.1 / §7.5.2: EMM STATUS #96, as for any EMM message (#382)
                log::warn!("EMM: undecodable SECURITY MODE COMMAND: {e}; EMM STATUS #96");
                return self.emm_status_96();
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
        // §5.3.5: T3412 from ATTACH ACCEPT; deactivated or zero, no periodic
        // update.
        // §5.5.1.2.4: the T3412 extended value, when present, is T3412;
        // otherwise the T3412 value (#356 review), as for TAU ACCEPT.
        self.t3412_value = match a.t3412_extended_s {
            Some(extended) => extended,
            None => a.t3412_s,
        }
        .filter(|s| *s > 0)
        .map(Duration::from_secs);
        self.registered_in_current_ta();
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
                // §7.5.2: the container itself is checked for presence and length
                // only; its ESM message gets the §7.5.3 a) treatment: an
                // undecodable ACTIVATE DEFAULT EPS BEARER CONTEXT REQUEST is
                // rejected with #96 in ATTACH COMPLETE (#382)
                log::warn!("ESM: undecodable ESM container: {e}; #96 (§7.5.3 a))");
                self.esm_invalid_mandatory_reply(&a.esm_message)
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
        self.stop_tau();
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
                self.clear_bearers();
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
                self.clear_bearers();
                self.stop_tau();
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
        if detach_type == IMSI_DETACH && self.state == EmmState::TrackingAreaUpdatingInitiated {
            // §5.5.3.2.6 f): "the DETACH REQUEST message shall be ignored and
            // tracking area updating procedure shall be progressed".
            log::info!("EMM: IMSI DETACH REQUEST during a tracking area update; ignored");
            return NasOutput::default();
        }
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
        // service request is progressed; those procedures are aborted, and so
        // is a tracking area update (§5.5.3.2.6 f).
        self.t3410 = None;
        self.t3417 = None;
        self.t3430 = None;
        self.tau_in_progress = None;
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
                self.clear_bearers();
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
                self.clear_bearers();
                self.stop_tau();
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
        self.clear_bearers();
        self.stop_tau();
        self.uplink_pending = false;
        self.state = EmmState::Deregistered;
        self.attempts = 0;
        self.retry_at = None;
    }

    fn on_esm(&mut self, plain: &[u8], now: Instant) -> NasOutput {
        let decoded = msg::decode_downlink_esm(plain);
        if let Ok(
            DownlinkEsm::ActivateDedicatedBearerRequest(msg::ActivateDedicatedBearerRequest {
                pti,
                ..
            })
            | DownlinkEsm::ModifyBearerRequest(msg::ModifyBearerRequest { pti, .. })
            | DownlinkEsm::DeactivateBearerRequest { pti, .. },
        ) = &decoded
        {
            // §6.5.4.3: the network's answer to a UE-requested bearer resource
            // modification carries its PTI; T3481 stops (#408).
            if *pti != 0 {
                self.bearer_resource_modifications.retain(|m| {
                    let answered = m.pti == *pti;
                    if answered {
                        log::info!(
                            "ESM: bearer resource modification PTI {} answered; T3481 stopped",
                            m.pti
                        );
                    }
                    !answered
                });
            }
        }
        match decoded {
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
            Ok(DownlinkEsm::ActivateDedicatedBearerRequest(req)) => {
                let answer = self.on_activate_dedicated(req);
                self.esm_answer(answer, now)
            }
            Ok(DownlinkEsm::ModifyBearerRequest(req)) => {
                let answer = self.on_modify_bearer(req);
                self.esm_answer(answer, now)
            }
            Ok(DownlinkEsm::DeactivateBearerRequest { ebi, pti, cause }) => {
                // §6.5.4.5 c): a deactivation of a bearer this UE asked to
                // modify aborts the modification (T3481 stopped).
                self.bearer_resource_modifications.retain(|m| m.ebi != ebi);
                self.on_deactivate_bearer(ebi, cause);
                self.send_emm(msg::encode_deactivate_accept(ebi, pti))
            }
            Ok(DownlinkEsm::BearerResourceModificationReject { pti, cause }) => {
                self.on_bearer_resource_modification_reject(pti, cause);
                NasOutput::default()
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
                // §7.5.1 / §7.5.3 (#382)
                log::warn!("ESM: undecodable message: {e}; #96 (§7.5.3)");
                let reply = self.esm_invalid_mandatory_reply(plain);
                self.send_emm(reply)
            }
        }
    }
}

impl EpsNasUe {
    /// ACTIVATE DEDICATED EPS BEARER CONTEXT REQUEST (TS 24.301 §6.4.2.3-5):
    /// the ACCEPT or REJECT to send.
    fn on_activate_dedicated(&mut self, req: msg::ActivateDedicatedBearerRequest) -> EsmAnswer {
        use crate::tft::{Tft, TftError, TftOperation};
        let reject = |cause: u8| {
            log::warn!(
                "ESM: dedicated bearer {} (linked {}) rejected, #{cause}",
                req.ebi,
                req.linked_ebi
            );
            EsmAnswer::reply(msg::encode_activate_dedicated_reject(
                req.ebi, req.pti, cause,
            ))
        };
        // §6.4.2.5 c): no default bearer with the linked EBI
        if self.bearer.as_ref().map(|b| b.ebi) != Some(req.linked_ebi) {
            return reject(ESM_INVALID_EPS_BEARER_IDENTITY);
        }
        // §6.4.2.4 a)-d)
        let tft = match Tft::decode(&req.tft) {
            Ok(t) if t.operation != TftOperation::CreateNew => {
                return reject(TftError::SemanticOperation.esm_cause())
            }
            Ok(t) => t,
            Err(e) => return reject(e.esm_cause()),
        };
        if !tft.filters.iter().any(|f| f.direction.applies_uplink()) {
            return reject(TftError::SemanticFilter.esm_cause());
        }
        // §6.4.2.5 a): the EBI of the active default bearer deactivates it and
        // its dedicated bearers locally, and the activation proceeds; b): that
        // of a dedicated one deactivates that one.
        if self.bearer.as_ref().map(|b| b.ebi) == Some(req.ebi) {
            log::warn!(
                "ESM: dedicated bearer request with the default bearer's EBI {}: the \
                 default bearer and its dedicated bearers are deactivated locally",
                req.ebi
            );
            self.clear_bearers();
        }
        // §6.4.2.4 d) 2): a precedence of the default bearer's releases the
        // PDN connection, and the activation with it.
        let PrecedenceClash::Deleted(deleted) = self.drop_precedence_clashes(req.ebi, &tft.filters)
        else {
            return EsmAnswer::ReleasePdnConnection;
        };
        self.dedicated.retain(|b| b.ebi != req.ebi);
        log::info!(
            "ESM: dedicated EPS bearer {} activated (linked {}), QCI {}, {} packet filter(s)",
            req.ebi,
            req.linked_ebi,
            req.qci,
            tft.filters.len()
        );
        self.dedicated.push(DedicatedBearer {
            ebi: req.ebi,
            linked_ebi: req.linked_ebi,
            qci: req.qci,
            filters: tft.filters,
        });
        EsmAnswer {
            reply: Some(msg::encode_activate_dedicated_accept(req.ebi, req.pti)),
            deleted,
        }
    }

    /// TS 24.301 §6.4.2.4 d) 2) / §6.4.3.4 d) 2): the precedences of
    /// `filters`, the TFT bearer `ebi` gets, against the old filters of the
    /// other bearers of the PDN connection.
    /// - An old filter of the default bearer with one of them: the PDN
    ///   connection is to be released ([`PrecedenceClash::DefaultBearer`]);
    ///   nothing is changed here.
    /// - Old filters of other dedicated bearers with one of them are deleted
    ///   and listed, for the UE-requested bearer resource modification that
    ///   deletes them in the network too (#408).
    fn drop_precedence_clashes(
        &mut self,
        ebi: u8,
        filters: &[crate::tft::PacketFilter],
    ) -> PrecedenceClash {
        let clashes =
            |old: &crate::tft::PacketFilter| filters.iter().any(|f| f.precedence == old.precedence);
        let default = self.bearer.as_ref().map(|b| b.ebi);
        if default != Some(ebi) && self.default_filters.iter().any(clashes) {
            log::warn!(
                "ESM: EPS bearer {ebi}'s TFT reuses a precedence of the default bearer's; \
                 the PDN connection is released (§6.4.2.4 d) 2))"
            );
            return PrecedenceClash::DefaultBearer;
        }
        let mut deleted = Vec::new();
        // `ebi`'s own old filters are replaced by `filters` anyway
        for b in self.dedicated.iter_mut().filter(|b| b.ebi != ebi) {
            let ids: Vec<u8> = b
                .filters
                .iter()
                .filter(|o| clashes(o))
                .map(|o| o.id)
                .collect();
            if ids.is_empty() {
                continue;
            }
            b.filters.retain(|old| !clashes(old));
            log::info!(
                "ESM: EPS bearer {}'s packet filter(s) {ids:?} share a precedence with EPS \
                 bearer {ebi}'s new TFT; deleted (§6.4.2.4 d) 2))",
                b.ebi
            );
            deleted.push((b.ebi, ids, b.filters.is_empty()));
        }
        PrecedenceClash::Deleted(deleted)
    }

    /// MODIFY EPS BEARER CONTEXT REQUEST (TS 24.301 §6.4.3.3-4): the ACCEPT or
    /// REJECT to send.
    fn on_modify_bearer(&mut self, req: msg::ModifyBearerRequest) -> EsmAnswer {
        use crate::tft::{apply_operation, Tft};
        let reject = |cause: u8| {
            log::warn!(
                "ESM: modification of EPS bearer {} rejected, #{cause}",
                req.ebi
            );
            EsmAnswer::reply(msg::encode_modify_reject(req.ebi, req.pti, cause))
        };
        let is_default = self.bearer.as_ref().map(|b| b.ebi) == Some(req.ebi);
        let dedicated_idx = self.dedicated.iter().position(|b| b.ebi == req.ebi);
        if !is_default && dedicated_idx.is_none() {
            return reject(ESM_INVALID_EPS_BEARER_IDENTITY);
        }
        let new_filters = match &req.tft {
            None => None,
            Some(v) => {
                let tft = match Tft::decode(v) {
                    Ok(t) => t,
                    Err(e) => return reject(e.esm_cause()),
                };
                let current = match dedicated_idx {
                    Some(i) => &self.dedicated[i].filters,
                    None => &self.default_filters,
                };
                match apply_operation(current, &tft, dedicated_idx.is_some()) {
                    Ok(f) => Some(f),
                    Err(e) => return reject(e.esm_cause()),
                }
            }
        };
        let deleted = match &new_filters {
            None => Vec::new(),
            Some(f) => match self.drop_precedence_clashes(req.ebi, f) {
                PrecedenceClash::Deleted(d) => d,
                // §6.4.3.4 d) 2): the PDN connection is released
                PrecedenceClash::DefaultBearer => return EsmAnswer::ReleasePdnConnection,
            },
        };
        match dedicated_idx {
            Some(i) => {
                let b = &mut self.dedicated[i];
                if let Some(q) = req.qci {
                    b.qci = q;
                }
                if let Some(f) = new_filters {
                    b.filters = f;
                }
            }
            None => {
                if let (Some(q), Some(b)) = (req.qci, self.bearer.as_mut()) {
                    b.qci = q;
                }
                if let Some(f) = new_filters {
                    self.default_filters = f;
                }
            }
        }
        log::info!("ESM: EPS bearer {} modified", req.ebi);
        EsmAnswer {
            reply: Some(msg::encode_modify_accept(req.ebi, req.pti)),
            deleted,
        }
    }

    /// DEACTIVATE EPS BEARER CONTEXT REQUEST (TS 24.301 §6.4.4.3): a dedicated
    /// bearer goes; the default bearer takes its PDN connection, dedicated
    /// bearers included, with it. An EBI that is not active is accepted too.
    fn on_deactivate_bearer(&mut self, ebi: u8, cause: u8) {
        if self.bearer.as_ref().map(|b| b.ebi) == Some(ebi) {
            log::info!(
                "ESM: default EPS bearer {ebi} deactivated (#{cause}), with its PDN connection"
            );
            self.clear_bearers();
        } else if self.dedicated.iter().any(|b| b.ebi == ebi) {
            log::info!("ESM: dedicated EPS bearer {ebi} deactivated (#{cause})");
            self.dedicated.retain(|b| b.ebi != ebi);
        } else {
            log::info!("ESM: deactivation of EPS bearer {ebi}, which is not active; accepted");
        }
    }
}

/// What an ESM request is answered with (#408).
struct EsmAnswer {
    /// The ACCEPT or REJECT.
    reply: Option<Vec<u8>>,
    /// Packet filters of other dedicated bearers deleted for a precedence
    /// clash, as (EBI, identifiers, whether none is left): each bearer's are
    /// deleted in the network with a UE-requested bearer resource
    /// modification (§6.4.2.4 d) 2)).
    deleted: Vec<(u8, Vec<u8>, bool)>,
}

#[allow(non_upper_case_globals)]
impl EsmAnswer {
    /// §6.4.2.4 d) 2) / §6.4.3.4 d) 2): an old filter of the default bearer
    /// clashes; the PDN connection is released instead of answering.
    const ReleasePdnConnection: Self = Self {
        reply: None,
        deleted: Vec::new(),
    };

    fn reply(pdu: Vec<u8>) -> Self {
        Self {
            reply: Some(pdu),
            deleted: Vec::new(),
        }
    }
}

impl EpsNasUe {
    /// Send an [`EsmAnswer`]: the reply, then a BEARER RESOURCE MODIFICATION
    /// REQUEST per bearer whose filters were deleted. No reply means the PDN
    /// connection is released.
    fn esm_answer(&mut self, answer: EsmAnswer, now: Instant) -> NasOutput {
        let Some(reply) = answer.reply else {
            return self.release_pdn_connection(now);
        };
        let mut out = self.send_emm(reply);
        for (ebi, ids, releases_all) in answer.deleted {
            let req = self.start_bearer_resource_modification(ebi, ids, releases_all, now);
            out.uplink.extend(req.uplink);
        }
        out
    }

    /// §6.4.2.4 d) 2) / §6.4.3.4 d) 2): "the UE shall release the relevant
    /// PDN connection. If the relevant PDN connection is the last one that the
    /// UE has and EMM-REGISTERED without PDN connection is not supported by
    /// the UE or the MME, the UE shall detach and re-attach to the network."
    /// This UE has one PDN connection, and its UE network capability (§9.9.3.34,
    /// octets 3-4 only) does not indicate EMM-REGISTERED without PDN
    /// connection, so it detaches ("EPS detach", not switched off) and the
    /// attach follows once the detach ends (#408).
    fn release_pdn_connection(&mut self, now: Instant) -> NasOutput {
        log::warn!(
            "ESM: the PDN connection is released, and it is the last one: detach and \
             re-attach (§6.4.2.4 d) 2))"
        );
        self.reattach_after_detach = true;
        let out = self.detach(false, true, now);
        if self.state != EmmState::DeregisteredInitiated {
            // No detach was started; the flag must not outlive it
            self.reattach_after_detach = false;
        }
        out
    }

    /// BEARER RESOURCE MODIFICATION REQUEST deleting `ids` from EPS bearer
    /// `ebi`'s traffic flow aggregate (§6.5.4.2): TFT operation "Delete packet
    /// filters from existing TFT" with the packet filter identifiers, ESM cause
    /// #36 "regular deactivation" (the release of bearer resources). T3481
    /// starts, PROCEDURE TRANSACTION PENDING.
    fn start_bearer_resource_modification(
        &mut self,
        ebi: u8,
        ids: Vec<u8>,
        releases_all: bool,
        now: Instant,
    ) -> NasOutput {
        use crate::tft::{Tft, TftOperation};
        let pti = self.alloc_pti();
        let aggregate = Tft {
            operation: TftOperation::DeleteFilters,
            filters: Vec::new(),
            deleted_ids: ids.clone(),
        }
        .encode();
        let request = msg::encode_bearer_resource_modification_request(
            pti,
            ebi,
            &aggregate,
            Some(ESM_REGULAR_DEACTIVATION),
        );
        log::info!(
            "ESM: BEARER RESOURCE MODIFICATION REQUEST, PTI {pti}: delete packet filter(s) \
             {ids:?} of EPS bearer {ebi}; T3481 started (§6.5.4.2)"
        );
        self.bearer_resource_modifications
            .push(BearerResourceModification {
                pti,
                ebi,
                request: request.clone(),
                releases_all,
                retransmissions: 0,
                t3481: now + T3481,
            });
        self.send_emm(request)
    }

    /// T3481 expiry (§6.5.4.5 a)): the request is retransmitted on the first
    /// four expiries; on the fifth the procedure is aborted and its PTI
    /// released, and a bearer whose every traffic flow it released is
    /// deactivated locally.
    fn on_t3481(&mut self, now: Instant) -> NasOutput {
        let mut out = NasOutput::default();
        let mut aborted = Vec::new();
        let mut resend = Vec::new();
        for m in &mut self.bearer_resource_modifications {
            if now < m.t3481 {
                continue;
            }
            if m.retransmissions < MAX_BEARER_RESOURCE_MODIFICATION_RETRANSMISSIONS {
                m.retransmissions += 1;
                m.t3481 = now + T3481;
                log::warn!(
                    "ESM: T3481 expired; BEARER RESOURCE MODIFICATION REQUEST PTI {} \
                     retransmitted ({})",
                    m.pti,
                    m.retransmissions
                );
                resend.push(m.request.clone());
            } else {
                aborted.push((m.pti, m.ebi, m.releases_all));
            }
        }
        for pdu in resend {
            out.uplink.extend(self.send_emm(pdu).uplink);
        }
        for (pti, ebi, releases_all) in aborted {
            log::warn!(
                "ESM: T3481 expired a fifth time; bearer resource modification PTI {pti} \
                 aborted (§6.5.4.5 a))"
            );
            self.bearer_resource_modifications.retain(|m| m.pti != pti);
            if releases_all {
                self.deactivate_dedicated_locally(ebi);
            }
        }
        out
    }

    /// BEARER RESOURCE MODIFICATION REJECT (§6.5.4.4.1): T3481 stops and the
    /// PTI is released. #43 deactivates the bearer locally (§6.5.4.5 b)), and
    /// so does #31 when the request released every traffic flow of it
    /// (§6.5.4.5 d)). A PTI not in use is ignored (§7.3.1 d)).
    fn on_bearer_resource_modification_reject(&mut self, pti: u8, cause: u8) {
        let Some(i) = self
            .bearer_resource_modifications
            .iter()
            .position(|m| m.pti == pti)
        else {
            log::warn!(
                "ESM: BEARER RESOURCE MODIFICATION REJECT for PTI {pti} not in use; ignored"
            );
            return;
        };
        let m = self.bearer_resource_modifications.remove(i);
        log::warn!("ESM: BEARER RESOURCE MODIFICATION REJECT #{cause}, PTI {pti}; T3481 stopped");
        if cause == ESM_INVALID_EPS_BEARER_IDENTITY
            || (cause == ESM_REQUEST_REJECTED_UNSPECIFIED && m.releases_all)
        {
            self.deactivate_dedicated_locally(m.ebi);
        }
    }

    /// Deactivate dedicated EPS bearer `ebi` without peer-to-peer signalling.
    fn deactivate_dedicated_locally(&mut self, ebi: u8) {
        log::info!("ESM: EPS bearer {ebi} deactivated locally");
        self.dedicated.retain(|b| b.ebi != ebi);
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

mod tau;

#[cfg(test)]
mod tests;
