//! 5GMM Procedure Orchestrator (3GPP TS 24.501 Section 5)
//!
//! This module owns the UE-side mobility-management procedure state:
//!
//! - Registration procedure (initial, mobility, periodic) with timers
//!   T3510 (procedure guard), T3511 (retry), T3502 (attempt-counter
//!   exhaustion), T3346 (congestion backoff) and T3512 (periodic update).
//! - Authentication procedure including the abnormal cases of
//!   TS 24.501 Section 5.4.1.3.6: MAC failure (cause #20), non-5G
//!   authentication (cause #21 separation-bit check via cause #26) and
//!   synch failure (cause #21 with a real Milenage f1*/f5* AUTS).
//! - Security mode control (TS 24.501 Section 5.4.2 / TS 33.501 Section
//!   6.7.2): NAS-MAC verification *before* key activation, bidding-down
//!   check of the replayed UE security capabilities, KAMF derivation from
//!   the network-signalled ABBA and the UE's SUPI.
//! - NAS message protection/unprotection (TS 24.501 Section 4.4): outer
//!   security header handling, downlink MAC verification, deciphering,
//!   uplink integrity protection and ciphering.
//!
//! The orchestrator is sans-IO: callers feed it downlink PDUs, trigger
//! procedures and drive `tick()` once a second; it returns [`MmOutput`]
//! values describing NAS PDUs to transmit and procedure outcomes.

use std::collections::HashMap;

use tracing::{debug, info, warn};

use nextgsim_common::config::{OpType, UeConfig};
use nextgsim_crypto::kdf::{derive_kamf, derive_kausf, derive_kgnb, derive_kseaf, derive_res_star};
use nextgsim_crypto::milenage::{compute_opc, Milenage};
use nextgsim_nas::enums::{MmMessageType, SecurityHeaderType};
use nextgsim_nas::ies::ie1::{
    FollowOnRequest, Ie5gsRegistrationType, IeServiceType, RegistrationType, ServiceType,
};
use nextgsim_nas::messages::mm::ue_policy::{
    ManageUePolicyCommand, ManageUePolicyCommandReject, ManageUePolicyComplete, PlmnId,
    UePolicyError, UePolicyPart, UePolicyPartType, UePolicyResult, UePolicySectionManagementResult,
    UePolicySectionManagementSubresult, UrspRule, CAUSE_PROTOCOL_ERROR_UNSPECIFIED, PCF_PTI_MAX,
    PCF_PTI_MIN, UPDP_MSG_MANAGE_UE_POLICY_COMMAND,
};
use nextgsim_nas::messages::mm::{
    AuthenticationFailure, AuthenticationRequest, AuthenticationResponse,
    DeregistrationRequestUeOriginating, DeregistrationRequestUeTerminated, Ie5gsMobileIdentity,
    MmCause, MobileIdentityType, RegistrationAccept, RegistrationComplete, RegistrationReject,
    RegistrationRequest, SecurityModeCommand, SecurityModeComplete, SecurityModeReject,
    ServiceReject, ServiceRequest,
};
use nextgsim_nas::security::{
    compute_nas_mac, nas_cipher, verify_nas_mac, CipheringAlgorithm, IntegrityAlgorithm, NasCount,
    NasDirection, NasKeySetIdentifier, NasSecurityContext, NAS_BEARER,
};

use crate::timer::{
    GprsTimer2, GprsTimer3, NasTimerManager, TimerExpiryEvent, TIMER_T3346, TIMER_T3502,
    TIMER_T3510, TIMER_T3511, TIMER_T3512, TIMER_T3516, TIMER_T3517, TIMER_T3519, TIMER_T3520,
    TIMER_T3521,
};

use super::deregistration::DeregistrationProcedure;
use super::state::{CmState, MmStateMachine, MmSubState, UpdateStatus};

/// Maximum registration attempts before falling back to T3502
/// (TS 24.501 Section 5.5.1.2.7)
pub const MAX_REGISTRATION_ATTEMPTS: u32 = 5;

/// Maximum consecutive authentication failures before the UE treats the
/// network as failed authentication (TS 24.501 Section 5.4.1.3.7)
pub const MAX_AUTH_FAILURES: u32 = 3;

/// Maximum deregistration retransmissions on T3521 expiry
/// (TS 24.501 Section 5.5.2.2.6)
pub const MAX_DEREGISTRATION_ATTEMPTS: u32 = 5;

/// SQN acceptance window delta (TS 33.102 Annex C.2.1)
pub const SQN_DELTA: u64 = 1 << 28;

/// Default IMEISV used when the configuration does not provide one
/// (same default as UERANSIM)
const DEFAULT_IMEISV: &str = "4370816125816151";

/// Sim-private message type for the UAV tracking report.
///
/// NOT 3GPP-conformant (Wave 4, T4.3 — honest reframe). Verified against
/// TS 24.501 v18 §8.2.10 / §9.11.3.40: **there is no UE-originated UAV
/// position/tracking NAS message** — real-time UAV tracking / Remote-ID is an
/// **application-layer** function (USS/UTM "Network Remote ID" over the user
/// plane), and network-based UAV location would use LCS (TS 23.273). This
/// simulator models the telemetry over an unassigned 5GMM message-type code for
/// a self-contained geofence demo; it has no conformant NAS equivalent and a
/// real network would not accept it. (Conformant UAS *registration/UUAA* uses
/// the Service-level-AA container — IEI 0x72 — which IS implemented.) Kept in
/// sync with `nextgcore-amfd`'s `gmm_build::message_type::UAV_TRACKING_REPORT`.
/// See `.context/WAVE6-DOWNGRADED-FEATURES.md` §6.
pub const UAV_TRACKING_REPORT_MSG_TYPE: u8 = 0x6A;

/// Decode a GPRS Timer 2 IE value octet (3GPP TS 24.008 Section 10.5.7.4)
/// into seconds. Returns 0 when the timer is deactivated.
pub fn gprs_timer2_seconds(byte: u8) -> u32 {
    let value = (byte & 0x1F) as u32;
    match byte >> 5 {
        0b000 => value * 2,
        0b001 => value * 60,
        0b010 => value * 6 * 60, // decihours
        0b111 => 0,              // deactivated
        _ => value * 60,         // other values interpreted as 1 minute
    }
}

/// Outputs produced by the orchestrator for the caller (NAS task) to act on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MmOutput {
    /// A NAS PDU ready for transmission (already security protected when
    /// the security context is active)
    SendNasPdu(Vec<u8>),
    /// The registration procedure completed successfully
    RegistrationSucceeded,
    /// The equivalent-PLMN list was (re)signalled by the network in the
    /// Registration Accept (TS 24.501 §5.5.1.2.4). Each entry is a 3-octet
    /// BCD-encoded PLMN in the network-supplied priority order; an empty
    /// vector means the list was deleted. The caller wires this into the
    /// TS 23.122 PLMN selector so subsequent selection honours it.
    EquivalentPlmnsUpdated(Vec<[u8; 3]>),
    /// The registration procedure failed terminally with the given cause
    RegistrationFailed(MmCause),
    /// The network rejected authentication; the USIM is considered invalid
    AuthenticationRejected,
    /// The UE should perform PLMN selection (forbidden PLMN / TA)
    PlmnSearchNeeded,
    /// A plain (deciphered) 5GMM PDU the orchestrator does not own;
    /// the caller should dispatch it (identity, transport, config update,
    /// session management piggybacks, ...)
    NotHandled(Vec<u8>),
    /// KgNB for AS-security activation, derived from KAMF and the uplink NAS
    /// COUNT (TS 33.501 §6.9.4.1). Emitted once NAS security is active so the
    /// caller hands it to the RRC plane (`RrcMessage::AsSecurityKey`) to derive
    /// K_RRCint/K_RRCenc when the AS SecurityModeCommand arrives (Wave-6 I5).
    /// Only produced when the `I5_UE_AS_SECURITY` wire gate is on.
    AsSecurityKgnb([u8; 32]),
}

/// UE identity and credential material used by the MM procedures.
#[derive(Debug, Clone)]
pub struct MmUeIdentity {
    /// SUPI value (IMSI digits, no prefix) as signalled to the network
    pub supi: String,
    /// Pre-built SUCI bytes for the 5GS mobile identity IE
    pub suci: Vec<u8>,
    /// Subscriber key K
    pub k: [u8; 16],
    /// Operator key OPc (already converted from OP if needed)
    pub opc: [u8; 16],
    /// Serving network name ("5G:mnc...mcc....3gppnetwork.org")
    pub sn_name: String,
    /// IMEISV digits (16) for the Security Mode Complete
    pub imei_sv: String,
    /// Advertised 5G-EA capability octet (bit 7 = EA0 ... bit 0 = EA7)
    pub ea_cap: u8,
    /// Advertised 5G-IA capability octet (bit 7 = IA0 ... bit 0 = IA7)
    pub ia_cap: u8,
    /// Requested NSSAI IE value (encoded S-NSSAI list), if configured
    pub requested_nssai: Option<Vec<u8>>,
    /// Home PLMN encoded as BCD (3 octets, TS 24.501 Section 9.11.3.4)
    pub plmn_bcd: [u8; 3],
    /// SNPN NID (Rel-17, TS 23.501 §5.30) when the UE is configured to access
    /// a Standalone Non-Public Network; included in the Registration Request.
    pub snpn_nid: Option<String>,
    /// MINT disaster-roaming indication (Rel-18, TS 23.761 §4.2). Set on the
    /// identity of a MINT secondary subscription so its Registration Request
    /// carries the disaster-roaming indication. Independent of the SNPN field.
    pub disaster_roaming: bool,
    /// UAV indication (Rel-18, TS 23.256). `Some(caa_id)` when the UE registers
    /// as an aerial UE; the Registration Request then advertises the aerial-UE
    /// flag plus the UAV CAA-level ID so the AMF creates the UAV flight
    /// authorization (geofence) context.
    pub uav_indication: Option<String>,
    /// RedCap (Reduced Capability) indication (Rel-17, TS 38.101). Set from the
    /// UE config so the Registration Request carries the reduced-capability NAS
    /// indication and the AMF/SMF apply a reduced session-AMBR.
    pub redcap: bool,
}

impl MmUeIdentity {
    /// Build the MM identity from the UE configuration and a pre-computed
    /// SUCI (the SUCI scheme output is built by the caller).
    pub fn from_config(config: &UeConfig, suci: Vec<u8>) -> Self {
        let opc = match config.op_type {
            OpType::Opc => config.op,
            OpType::Op => compute_opc(&config.key, &config.op),
        };

        let mcc = config.hplmn.mcc;
        let mnc = config.hplmn.mnc;
        let sn_name = format!("5G:mnc{mnc:03}.mcc{mcc:03}.3gppnetwork.org");

        let supi = if let Some(ref supi) = config.supi {
            supi.value.clone()
        } else {
            format!("{mcc:03}{mnc:02}0000000001")
        };

        // Advertised security capabilities (EA0 always supported; IA0 is
        // null integrity and must not be advertised for normal service).
        let algs = &config.supported_algs;
        let ea_cap = 0x80
            | (u8::from(algs.nea1) << 6)
            | (u8::from(algs.nea2) << 5)
            | (u8::from(algs.nea3) << 4);
        let ia_cap =
            (u8::from(algs.nia1) << 6) | (u8::from(algs.nia2) << 5) | (u8::from(algs.nia3) << 4);

        // Requested NSSAI value: list of S-NSSAI entries (TS 24.501 9.11.3.37)
        let requested_nssai = if config.configured_nssai.slices.is_empty() {
            None
        } else {
            let mut value = Vec::new();
            for s in &config.configured_nssai.slices {
                if let Some(sd) = s.sd {
                    value.push(4);
                    value.push(s.sst);
                    value.extend_from_slice(&sd);
                } else {
                    value.push(1);
                    value.push(s.sst);
                }
            }
            Some(value)
        };

        let imei_sv = config
            .imei_sv
            .clone()
            .unwrap_or_else(|| DEFAULT_IMEISV.to_string());

        Self {
            supi,
            suci,
            k: config.key,
            opc,
            sn_name,
            imei_sv,
            ea_cap,
            ia_cap,
            requested_nssai,
            plmn_bcd: encode_plmn_bcd(mcc, mnc, config.hplmn.long_mnc),
            // SNPN access (Rel-17, TS 23.501 §5.30): carry the configured NID
            // so the Registration Request advertises the SNPN being accessed.
            snpn_nid: config.snpn_config.as_ref().map(|s| s.nid.clone()),
            // The primary subscription never sets disaster-roaming; MINT
            // secondary identities opt in via `with_supi` (TS 23.761 §4.2).
            disaster_roaming: false,
            // UAV (Rel-18, TS 23.256): aerial UE registers with its CAA-level
            // ID so the AMF grants/tracks flight authorization. Inert (None)
            // unless the UE config enables the UAV feature.
            uav_indication: config
                .uav_config
                .as_ref()
                .filter(|u| u.is_aerial_ue)
                .map(|u| u.uav_id.clone().unwrap_or_default()),
            // RedCap (Rel-17, TS 38.101): carry the reduced-capability flag from
            // config so the Registration Request signals it to the AMF.
            redcap: config.redcap,
        }
    }

    /// Derive a MINT secondary-subscription identity from the UE configuration
    /// but for a different SUPI value (TS 23.761): the home PLMN, keys and
    /// security capabilities are shared with the primary, only the signalled
    /// SUPI and the pre-built SUCI differ. `disaster_roaming` controls the
    /// disaster-roaming indication in this subscription's Registration Request.
    pub fn for_secondary_supi(
        config: &UeConfig,
        suci: Vec<u8>,
        supi_value: &str,
        disaster_roaming: bool,
    ) -> Self {
        let mut id = Self::from_config(config, suci);
        id.supi = supi_value.to_string();
        id.disaster_roaming = disaster_roaming;
        // A MINT secondary subscription is not the aerial-UE registration; the
        // UAV indication belongs only to the primary subscription's identity.
        id.uav_indication = None;
        id
    }
}

/// Encode MCC/MNC as the 3-octet BCD PLMN field of TS 24.501 9.11.3.4.
pub(crate) fn encode_plmn_bcd(mcc: u16, mnc: u16, long_mnc: bool) -> [u8; 3] {
    let mcc_d = [
        (mcc / 100 % 10) as u8,
        (mcc / 10 % 10) as u8,
        (mcc % 10) as u8,
    ];
    let (mnc_d3, mnc_d) = if long_mnc || mnc > 99 {
        (
            (mnc % 10) as u8,
            [(mnc / 100 % 10) as u8, (mnc / 10 % 10) as u8],
        )
    } else {
        (0x0F, [(mnc / 10 % 10) as u8, (mnc % 10) as u8])
    };
    [
        (mcc_d[1] << 4) | mcc_d[0],
        (mnc_d3 << 4) | mcc_d[2],
        (mnc_d[1] << 4) | mnc_d[0],
    ]
}

/// Build an IMEISV 5GS mobile identity (TS 24.501 Section 9.11.3.4).
fn build_imeisv_identity(imei_sv: &str) -> Ie5gsMobileIdentity {
    let digits: Vec<u8> = imei_sv
        .chars()
        .filter_map(|c| c.to_digit(10).map(|d| d as u8))
        .collect();
    let mut data = Vec::with_capacity(9);
    let first_digit = digits.first().copied().unwrap_or(0);
    // Even number of digits (16 for IMEISV): odd indicator = 0
    let odd = u8::from(digits.len() % 2 == 1);
    data.push((first_digit << 4) | (odd << 3) | (MobileIdentityType::ImeiSv as u8));
    let mut i = 1;
    while i < digits.len() {
        let low = digits[i];
        let high = digits.get(i + 1).copied().unwrap_or(0x0F);
        data.push((high << 4) | low);
        i += 2;
    }
    Ie5gsMobileIdentity::new(MobileIdentityType::ImeiSv, data)
}

/// Check whether the algorithm with the given identifier (0-7) is contained
/// in the advertised capability octet (bit 7 = algorithm 0).
fn alg_in_capability(cap_octet: u8, alg_id: u8) -> bool {
    if alg_id > 7 {
        return false;
    }
    cap_octet & (0x80 >> alg_id) != 0
}

/// Convert a 48-bit big-endian SQN to an integer.
fn sqn_to_u64(sqn: &[u8; 6]) -> u64 {
    let mut v = 0u64;
    for b in sqn {
        v = (v << 8) | u64::from(*b);
    }
    v
}

/// A stored UE policy section: the UE policy parts installed for one
/// `(PLMN, UPSC)` key (TS 24.501 D.2.1.3 / D.3). Owned by the MM plane
/// (never the SM orchestrator, which owns PDU-session logic).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredUePolicySection {
    /// The UE policy parts of the section (URSP, ANDSP, ...), as delivered.
    pub parts: Vec<UePolicyPart>,
}

impl StoredUePolicySection {
    /// The decoded URSP rules of the section's URSP part, if it carries one.
    pub fn ursp_rules(&self) -> Option<Vec<UrspRule>> {
        self.parts
            .iter()
            .find(|p| p.part_type == UePolicyPartType::Ursp)
            .and_then(|p| p.decode_ursp().ok())
    }
}

/// Result of handling a received "UE policy container": the UE policy delivery
/// service message content to carry uplink (or nothing).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UePolicyReaction {
    /// Encoded MANAGE UE POLICY COMPLETE / COMMAND REJECT content to send in a
    /// UL NAS TRANSPORT with payload container type "UE policy container"
    /// (0b0101).
    Reply(Vec<u8>),
    /// The command was too malformed to even echo a PCF PTI: drop it without
    /// replying (never a panic — remote-DoS caution).
    Ignore,
}

/// 5GMM procedure orchestrator (UE side).
pub struct MmOrchestrator {
    identity: MmUeIdentity,
    state: MmStateMachine,
    timers: NasTimerManager,
    sec: NasSecurityContext,

    // -- authentication state --
    /// ABBA parameter received in the last Authentication Request
    /// (TS 33.501 Annex A.7 input to KAMF derivation)
    abba: Vec<u8>,
    /// Highest accepted SQN (SQN_MS, TS 33.102 Annex C)
    sqn_ms: [u8; 6],
    /// Stored RES* (deleted on T3516 expiry, TS 24.501 Section 5.4.1.3.3)
    res_star: Option<[u8; 16]>,
    /// Consecutive authentication failure counter (Section 5.4.1.3.7)
    auth_failure_count: u32,

    // -- registration state --
    /// Plain encoding of the last Registration Request (for the Security
    /// Mode Complete NAS message container and retransmission)
    last_registration_request: Option<Vec<u8>>,
    pending_reg_type: RegistrationType,
    registration_attempt_counter: u32,
    stored_guti: Option<Ie5gsMobileIdentity>,
    tai_list: Option<Vec<u8>>,
    allowed_nssai: Option<Vec<u8>>,
    current_tai: Option<[u8; 6]>,
    /// Equivalent PLMN list signalled in the last Registration Accept
    /// (TS 24.501 §5.5.1.2.4 / IE 9.11.3.45). Each entry is a 3-octet
    /// BCD-encoded PLMN, in the network-supplied priority order. Replaced
    /// on every Registration Accept (deleted when the IE is absent).
    equivalent_plmns: Vec<[u8; 3]>,
    /// T3502 value signalled by the network (used at attempt exhaustion)
    t3502_value: Option<GprsTimer3>,

    // -- forbidden lists (TS 24.501 Section 5.3.13 / annexed lists) --
    forbidden_plmns: Vec<[u8; 3]>,
    forbidden_tais_roaming: Vec<[u8; 6]>,
    forbidden_tais_service: Vec<[u8; 6]>,
    snpn_temporarily_forbidden: bool,
    snpn_permanently_forbidden: bool,

    /// Set when the network signalled the USIM must be considered invalid
    usim_invalid: bool,
    /// Set when N1 mode was disabled by the network (cause #27)
    n1_mode_disabled: bool,

    // -- deregistration state --
    dereg_pdu: Option<Vec<u8>>,

    // -- SUCI freshness (TS 24.501 §5.4.3.3 / §5.5.1.2.2) --
    /// The last SUCI sent to the network. Reused while T3519 is running instead
    /// of generating a fresh ECIES concealment on every IDENTITY/REGISTRATION
    /// request (TS 33.501 §6.12.4 frequency limit).
    stored_suci: Option<Vec<u8>>,

    // -- UE policy delivery service (UPDP, TS 24.501 Annex D) --
    /// UE policy sections stored by `(PLMN, UPSC)` (TS 24.501 D.2.1.3 / D.3).
    ue_policy_sections: HashMap<(PlmnId, u16), StoredUePolicySection>,
}

impl MmOrchestrator {
    /// Create a new orchestrator for the given UE identity.
    pub fn new(identity: MmUeIdentity) -> Self {
        Self {
            identity,
            state: MmStateMachine::new(),
            timers: NasTimerManager::new(),
            sec: NasSecurityContext::new_3gpp(),
            abba: vec![0x00, 0x00],
            sqn_ms: [0u8; 6],
            res_star: None,
            auth_failure_count: 0,
            last_registration_request: None,
            pending_reg_type: RegistrationType::InitialRegistration,
            registration_attempt_counter: 0,
            stored_guti: None,
            tai_list: None,
            allowed_nssai: None,
            current_tai: None,
            equivalent_plmns: Vec::new(),
            t3502_value: None,
            forbidden_plmns: Vec::new(),
            forbidden_tais_roaming: Vec::new(),
            forbidden_tais_service: Vec::new(),
            snpn_temporarily_forbidden: false,
            snpn_permanently_forbidden: false,
            usim_invalid: false,
            n1_mode_disabled: false,
            dereg_pdu: None,
            stored_suci: None,
            ue_policy_sections: HashMap::new(),
        }
    }

    /// TS 24.501 §5.4.3.3 b): while T3519 is running the UE must reuse the SUCI
    /// it last sent instead of generating a fresh concealment. Returns the
    /// stored SUCI to reuse, or `None` when the caller must generate (and then
    /// store, via [`Self::store_suci_and_arm_t3519`]) a fresh one.
    pub fn stored_suci_while_t3519_running(&self) -> Option<Vec<u8>> {
        if self.timers.is_running(TIMER_T3519) == Some(true) {
            self.stored_suci.clone()
        } else {
            None
        }
    }

    /// Store a freshly generated SUCI and arm T3519 so subsequent
    /// IDENTITY/REGISTRATION requests reuse it until the timer expires
    /// (TS 24.501 §5.4.3.3 / §5.5.1.2.2, TS 33.501 §6.12.4).
    pub fn store_suci_and_arm_t3519(&mut self, suci: Vec<u8>) {
        self.stored_suci = Some(suci);
        self.timers.start(TIMER_T3519, true);
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    /// Immutable access to the MM state machine
    pub fn state(&self) -> &MmStateMachine {
        &self.state
    }

    /// Mutable access to the MM state machine (CM transitions are driven
    /// by the RRC layer)
    pub fn state_mut(&mut self) -> &mut MmStateMachine {
        &mut self.state
    }

    /// Immutable access to the NAS timers
    pub fn timers(&self) -> &NasTimerManager {
        &self.timers
    }

    /// Immutable access to the NAS security context
    pub fn security_context(&self) -> &NasSecurityContext {
        &self.sec
    }

    /// The 5G-GUTI assigned by the network, if any
    pub fn stored_guti(&self) -> Option<&Ie5gsMobileIdentity> {
        self.stored_guti.as_ref()
    }

    /// The SUPI this orchestrator registers (used by MINT to report a
    /// secondary subscription's identity, TS 23.761).
    pub fn supi(&self) -> &str {
        &self.identity.supi
    }

    /// Registration area (TAI list) assigned by the network, if any
    pub fn tai_list(&self) -> Option<&[u8]> {
        self.tai_list.as_deref()
    }

    /// Equivalent PLMN list (TS 24.501 §5.5.1.2.4) signalled in the last
    /// Registration Accept, as 3-octet BCD PLMNs in priority order. Empty when
    /// the network never signalled one (or deleted it).
    pub fn equivalent_plmns(&self) -> &[[u8; 3]] {
        &self.equivalent_plmns
    }

    /// Allowed NSSAI assigned by the network, if any
    pub fn allowed_nssai(&self) -> Option<&[u8]> {
        self.allowed_nssai.as_deref()
    }

    /// Whether the USIM is considered invalid (after cause #3/#6/#7 or
    /// Authentication Reject)
    pub fn is_usim_invalid(&self) -> bool {
        self.usim_invalid
    }

    /// Stored RES* (kept until T3516 expiry)
    pub fn res_star(&self) -> Option<&[u8; 16]> {
        self.res_star.as_ref()
    }

    /// Forbidden PLMN list
    pub fn forbidden_plmns(&self) -> &[[u8; 3]] {
        &self.forbidden_plmns
    }

    /// "5GS forbidden tracking areas for roaming" list
    pub fn forbidden_tais_roaming(&self) -> &[[u8; 6]] {
        &self.forbidden_tais_roaming
    }

    /// "5GS forbidden tracking areas for regional provision of service" list
    pub fn forbidden_tais_service(&self) -> &[[u8; 6]] {
        &self.forbidden_tais_service
    }

    /// Registration attempt counter (TS 24.501 Section 5.5.1.2.7)
    pub fn registration_attempt_counter(&self) -> u32 {
        self.registration_attempt_counter
    }

    // ========================================================================
    // Timer driving
    // ========================================================================

    /// Drive all NAS timers; call roughly once a second.
    pub fn tick(&mut self) -> Vec<MmOutput> {
        let events = self.timers.perform_tick();
        let mut outs = Vec::new();
        for ev in events {
            outs.extend(self.handle_timer_expiry(ev));
        }
        outs
    }

    /// Handle a single timer expiry event.
    pub fn handle_timer_expiry(&mut self, ev: TimerExpiryEvent) -> Vec<MmOutput> {
        match ev.timer_code {
            TIMER_T3510 => {
                // TS 24.501 5.5.1.2.7 case c: T3510 expiry -> abort the
                // procedure, abnormal case handling
                warn!("T3510 expired: registration procedure timed out");
                self.abnormal_registration_failure()
            }
            TIMER_T3511 => {
                if self.may_retry_registration() {
                    info!("T3511 expired: retrying registration");
                    self.build_and_send_registration(self.pending_reg_type)
                } else {
                    Vec::new()
                }
            }
            TIMER_T3502 => {
                if self.may_retry_registration() {
                    info!("T3502 expired: restarting registration, attempt counter reset");
                    self.registration_attempt_counter = 0;
                    self.build_and_send_registration(self.pending_reg_type)
                } else {
                    Vec::new()
                }
            }
            TIMER_T3512 => {
                if self.state.is_registered() {
                    info!("T3512 expired: starting periodic registration update");
                    self.build_and_send_registration(RegistrationType::PeriodicRegistrationUpdating)
                } else {
                    Vec::new()
                }
            }
            TIMER_T3346 => {
                if !self.state.is_registered() && !self.usim_invalid {
                    info!("T3346 expired: congestion backoff over, retrying registration");
                    self.registration_attempt_counter = 0;
                    self.build_and_send_registration(self.pending_reg_type)
                } else {
                    Vec::new()
                }
            }
            TIMER_T3516 => {
                // TS 24.501 5.4.1.3.3: delete the stored RES* on expiry
                debug!("T3516 expired: deleting stored RES*");
                self.res_star = None;
                Vec::new()
            }
            TIMER_T3519 => {
                // TS 24.501 §5.4.3.3: once T3519 expires the stored SUCI may be
                // regenerated, so drop it and let the next request build a fresh
                // concealment.
                debug!("T3519 expired: clearing stored SUCI");
                self.stored_suci = None;
                Vec::new()
            }
            TIMER_T3520 => {
                // TS 24.501 5.4.1.3.7: network failed to react to the
                // authentication failure in time
                warn!("T3520 expired: authentication failure procedure timed out");
                self.auth_failure_count = 0;
                Vec::new()
            }
            TIMER_T3521 => {
                if ev.expiry_count < MAX_DEREGISTRATION_ATTEMPTS {
                    if let Some(pdu) = self.dereg_pdu.clone() {
                        info!(
                            "T3521 expired ({} of {}): retransmitting Deregistration Request",
                            ev.expiry_count, MAX_DEREGISTRATION_ATTEMPTS
                        );
                        self.timers.start(TIMER_T3521, false);
                        let protected = self.protect_if_active(pdu);
                        return vec![MmOutput::SendNasPdu(protected)];
                    }
                    Vec::new()
                } else {
                    // TS 24.501 5.5.2.2.6: abort and deregister locally
                    warn!("T3521 exhausted: aborting deregistration, local detach");
                    self.dereg_pdu = None;
                    self.state
                        .switch_mm_state(MmSubState::DeregisteredNormalService);
                    Vec::new()
                }
            }
            TIMER_T3517 => {
                // TS 24.501 5.6.1.7: abort the service request procedure
                warn!("T3517 expired: service request procedure timed out");
                if self.state.mm_substate() == MmSubState::ServiceRequestInitiated {
                    self.state
                        .switch_mm_state(MmSubState::RegisteredNormalService);
                }
                Vec::new()
            }
            _ => Vec::new(),
        }
    }

    fn may_retry_registration(&self) -> bool {
        !self.usim_invalid
            && !self.n1_mode_disabled
            && !self.state.is_registered()
            && !self.timers.t3346.is_running()
    }

    // ========================================================================
    // Registration procedure (TS 24.501 Section 5.5.1)
    // ========================================================================

    /// Start a registration procedure of the given type.
    pub fn start_registration(&mut self, reg_type: RegistrationType) -> Vec<MmOutput> {
        if self.usim_invalid {
            warn!("Registration not started: USIM considered invalid");
            return Vec::new();
        }
        if self.n1_mode_disabled {
            warn!("Registration not started: N1 mode disabled by the network");
            return Vec::new();
        }
        if self.timers.t3346.is_running() && reg_type != RegistrationType::EmergencyRegistration {
            info!(
                "Registration deferred: T3346 congestion backoff running ({}s left)",
                self.timers.t3346.remaining()
            );
            self.pending_reg_type = reg_type;
            self.state
                .switch_mm_state(MmSubState::DeregisteredAttemptingRegistration);
            return Vec::new();
        }
        if self.state.mm_substate() == MmSubState::RegisteredInitiated {
            debug!("Registration already in progress");
            return Vec::new();
        }
        self.build_and_send_registration(reg_type)
    }

    /// Build, record and emit a Registration Request; arms T3510.
    fn build_and_send_registration(&mut self, reg_type: RegistrationType) -> Vec<MmOutput> {
        let ng_ksi = if self.sec.is_active() {
            self.sec.nas_ksi()
        } else {
            NasKeySetIdentifier::no_key()
        };

        // TS 24.501 5.5.1.2.2: use the valid 5G-GUTI when available,
        // otherwise the SUCI. A SUCI sent here is stored and T3519 armed so the
        // identity/registration paths reuse it until the timer expires.
        let mobile_identity = if let Some(ref guti) = self.stored_guti {
            guti.clone()
        } else {
            let suci = self.identity.suci.clone();
            self.store_suci_and_arm_t3519(suci.clone());
            Ie5gsMobileIdentity::new(MobileIdentityType::Suci, suci)
        };

        let mut req = RegistrationRequest::new(
            Ie5gsRegistrationType::new(FollowOnRequest::NoPending, reg_type),
            ng_ksi,
            mobile_identity.clone(),
        );
        // Mandatory-for-security IEs: UE security capability (replayed by
        // the network in the Security Mode Command) and requested NSSAI
        req.ue_security_capability = Some(vec![self.identity.ea_cap, self.identity.ia_cap]);
        req.requested_nssai = self.identity.requested_nssai.clone();
        req.last_visited_tai = self.current_tai;
        // SNPN access (Rel-17, TS 23.501 §5.30): when configured, advertise the
        // NID so the AMF can validate SNPN authorization (TS 24.501).
        if let Some(ref nid) = self.identity.snpn_nid {
            req.snpn_nid = Some(nid.clone());
            info!("SNPN registration: including NID={nid}");
        }
        // MINT disaster-roaming indication (Rel-18, TS 23.761 §4.2): a MINT
        // secondary subscription advertises disaster-roaming so the AMF can
        // apply minimization-of-service-interruption handling.
        if self.identity.disaster_roaming {
            req.disaster_roaming = true;
            info!(
                "MINT registration: including disaster-roaming indication (SUPI={})",
                self.identity.supi
            );
        }
        // UAV indication (Rel-18, TS 23.256): an aerial UE advertises the
        // aerial-UE flag plus its CAA-level ID so the AMF creates the UAV
        // flight-authorization context for this registration.
        if let Some(ref caa_id) = self.identity.uav_indication {
            req.uav_indication = Some(caa_id.clone());
            info!("UAV registration: including aerial-UE indication (CAA-ID={caa_id})");
        }
        // RedCap (Rel-17, TS 38.101): a reduced-capability UE advertises the
        // indication so the AMF caps UE-AMBR and the SMF reduces session-AMBR.
        // (The RRCSetupComplete AS signalling is modelled in the lib RrcTask but
        // the runtime binary's RRC path does not exercise it; this NAS
        // indication is what actually reaches the core.)
        if self.identity.redcap {
            req.redcap = true;
            info!("RedCap registration: including reduced-capability indication");
        }

        let mut plain = Vec::new();
        req.encode(&mut plain);

        // The FULL request (all IEs) is replayed in the Security Mode Complete
        // NAS message container once security is up (TS 24.501 §4.4.6).
        self.last_registration_request = Some(plain.clone());
        self.pending_reg_type = reg_type;
        self.state.switch_mm_state(MmSubState::RegisteredInitiated);
        self.timers.start(TIMER_T3510, true);
        self.timers.stop(TIMER_T3511, true);

        // TS 24.501 §4.4.6: an initial registration request sent WITHOUT
        // integrity protection must carry cleartext IEs only. Non-cleartext IEs
        // (requested NSSAI, last visited TAI, the Rel-17/18 indications) are
        // deferred to the Security Mode Complete container above; a real AMF
        // (Open5GS) rejects them in the clear with 5GMM cause #95 "semantically
        // incorrect message". UE security capability and the NID *are* cleartext
        // IEs, so they stay. When a security context is already active the full
        // request is sent integrity-protected and all IEs are allowed.
        let wire = if self.sec.is_active() {
            plain
        } else {
            let mut ct = RegistrationRequest::new(
                Ie5gsRegistrationType::new(FollowOnRequest::NoPending, reg_type),
                NasKeySetIdentifier::no_key(),
                mobile_identity,
            );
            ct.ue_security_capability = Some(vec![self.identity.ea_cap, self.identity.ia_cap]);
            if let Some(ref nid) = self.identity.snpn_nid {
                ct.snpn_nid = Some(nid.clone());
            }
            // ue-05: "UE determined PLMN with disaster condition" is a permitted
            // cleartext IE (TS 24.501 §4.4.6), so when MINT disaster roaming is
            // active it stays in the unprotected initial request rather than
            // being deferred to the Security Mode Complete container.
            if self.identity.disaster_roaming {
                ct.disaster_roaming = true;
            }
            let mut ct_bytes = Vec::new();
            ct.encode(&mut ct_bytes);
            ct_bytes
        };

        info!(
            "Sending Registration Request ({:?}), len={}, cleartext_only={}, T3510 started",
            reg_type,
            wire.len(),
            !self.sec.is_active()
        );

        let pdu = self.protect_if_active(wire);
        vec![MmOutput::SendNasPdu(pdu)]
    }

    /// Build a UAV tracking report uplink NAS PDU (Rel-18, TS 23.256).
    ///
    /// The aerial UE periodically reports its position (Remote ID / flight
    /// position) to the network so the AMF can run the geofence and enforce
    /// flight authorization. This is carried as a plain 5GMM message with the
    /// AMF-private message type [`UAV_TRACKING_REPORT_MSG_TYPE`]; the value is a
    /// fixed layout of latitude/longitude/altitude (fixed point) plus a flight
    /// status octet. It is protected with the active NAS security context (the
    /// gNB relays it to the AMF as an Uplink NAS Transport), and returns `None`
    /// before security is established so it is never sent in the clear.
    ///
    /// `caa_id` is the UAV CAA-level identifier; `lat`/`lon` are decimal
    /// degrees and `alt` is metres; `flight_status` is 0=grounded, 1=flying,
    /// 2=emergency.
    pub fn build_uav_tracking_report(
        &mut self,
        caa_id: &str,
        lat: f64,
        lon: f64,
        alt: f64,
        flight_status: u8,
    ) -> Option<Vec<u8>> {
        if !self.sec.is_active() {
            warn!("UAV tracking report suppressed: NAS security not established");
            return None;
        }
        let mut plain = Vec::with_capacity(3 + 1 + caa_id.len() + 13);
        plain.push(0x7E); // 5GMM extended protocol discriminator
        plain.push(0x00); // plain NAS message (security applied below)
        plain.push(UAV_TRACKING_REPORT_MSG_TYPE);
        // CAA-level ID (length-prefixed)
        plain.push(caa_id.len() as u8);
        plain.extend_from_slice(caa_id.as_bytes());
        // Position: lat/lon as 1e-7 degree fixed point (i32), alt as 0.1 m (i32)
        plain.extend_from_slice(&((lat * 1e7) as i32).to_be_bytes());
        plain.extend_from_slice(&((lon * 1e7) as i32).to_be_bytes());
        plain.extend_from_slice(&((alt * 10.0) as i32).to_be_bytes());
        plain.push(flight_status);
        info!(
            "UAV tracking report: CAA-ID={caa_id}, pos=({lat:.6}, {lon:.6}), \
             alt={alt:.1}m, status={flight_status}"
        );
        Some(self.protect_if_active(plain))
    }

    /// Registration Accept handling (TS 24.501 Section 5.5.1.2.4)
    fn handle_registration_accept(&mut self, plain: &[u8]) -> Vec<MmOutput> {
        let acc = match RegistrationAccept::decode(&mut &plain[3..]) {
            Ok(acc) => acc,
            Err(e) => {
                warn!("Failed to decode Registration Accept: {e:?}");
                return Vec::new();
            }
        };

        self.timers.stop(TIMER_T3510, true);
        self.timers.stop(TIMER_T3346, true);
        // TS 24.501 5.4.1.3.3: stop T3516 and delete RES* on registration accept
        self.timers.stop(TIMER_T3516, true);
        self.res_star = None;
        self.registration_attempt_counter = 0;

        self.state
            .switch_mm_state(MmSubState::RegisteredNormalService);
        self.state.switch_update_status(UpdateStatus::Updated);
        info!("Received Registration Accept: UE is now {}", self.state);

        // Mandatory registration area / slice assignment from the network
        if let Some(ref tai_list) = acc.tai_list {
            self.current_tai = parse_first_tai(tai_list);
            self.tai_list = Some(tai_list.clone());
        } else {
            warn!("Registration Accept without TAI list (network not strict)");
        }
        if let Some(ref nssai) = acc.allowed_nssai {
            self.allowed_nssai = Some(nssai.clone());
        } else {
            warn!("Registration Accept without Allowed NSSAI (network not strict)");
        }

        // Equivalent PLMN list (TS 24.501 §5.5.1.2.4 / IE 9.11.3.45). The list
        // is replaced on every Registration Accept; an absent IE deletes it.
        // The IE value is a sequence of 3-octet BCD PLMNs.
        let new_equivalent_plmns: Vec<[u8; 3]> = acc
            .equivalent_plmns
            .as_deref()
            .map(parse_equivalent_plmns)
            .unwrap_or_default();
        let equivalent_plmns_changed = new_equivalent_plmns != self.equivalent_plmns;
        if equivalent_plmns_changed {
            self.equivalent_plmns = new_equivalent_plmns;
            info!(
                "Registration Accept signalled {} equivalent PLMN(s)",
                self.equivalent_plmns.len()
            );
        }

        // Periodic registration timer (T3512, GPRS timer 3)
        if let Some(t3512_byte) = acc.t3512_value {
            let timer3 = GprsTimer3::from_byte(t3512_byte);
            let secs = timer3.to_seconds();
            if secs > 0 {
                self.timers.start_with_timer3(TIMER_T3512, timer3, true);
                info!("T3512 (periodic registration) started: {secs}s");
            } else {
                self.timers.stop(TIMER_T3512, true);
            }
        }
        // T3502 value for attempt-counter exhaustion handling
        if let Some(t3502_byte) = acc.t3502_value {
            self.t3502_value = Some(GprsTimer3::from_byte(t3502_byte));
        }

        let mut outs = Vec::new();
        // TS 24.501 5.5.1.2.4: the UE acknowledges with Registration Complete
        // whenever the Accept assigns a new 5G-GUTI, carries a Pending NSSAI IE,
        // a Network slicing indication IE with the subscription-change indication
        // set to "changed", or negotiated DRX parameters. The AMF runs T3550
        // awaiting the acknowledgement, so gating the Complete on the GUTI alone
        // would leave that timer to expire on any of these other Accepts.
        let needs_ack = acc.guti.is_some()
            || acc.pending_nssai.is_some()
            || acc.network_slicing_subscription_change == Some(true)
            || acc.negotiated_drx_parameters.is_some();
        if let Some(guti) = acc.guti {
            info!(
                "Registration Accept assigned a new 5G-GUTI ({} bytes)",
                guti.data.len()
            );
            self.stored_guti = Some(guti);
        }
        if needs_ack {
            let mut complete = Vec::new();
            RegistrationComplete::new().encode(&mut complete);
            let pdu = self.protect_if_active(complete);
            info!("Sending Registration Complete");
            outs.push(MmOutput::SendNasPdu(pdu));
        }
        // Surface the equivalent-PLMN list so the caller can wire it into the
        // TS 23.122 PLMN selector. Emitted only when the list changed (a list
        // becomes non-empty, or a previously signalled list is cleared by a
        // list-less Registration Accept), so a UE that never receives the IE
        // produces no spurious output.
        if equivalent_plmns_changed {
            outs.push(MmOutput::EquivalentPlmnsUpdated(
                self.equivalent_plmns.clone(),
            ));
        }
        outs.push(MmOutput::RegistrationSucceeded);
        outs
    }

    /// Registration Reject handling with the full cause table of
    /// TS 24.501 Sections 5.5.1.2.5 and 5.5.1.3.5.
    fn handle_registration_reject(&mut self, plain: &[u8]) -> Vec<MmOutput> {
        let rej = match RegistrationReject::decode(&mut &plain[3..]) {
            Ok(rej) => rej,
            Err(e) => {
                warn!("Failed to decode Registration Reject: {e:?}");
                return Vec::new();
            }
        };
        let cause = rej.mm_cause.value;
        warn!("Registration Reject: cause {:?} (#{})", cause, cause as u8);

        self.timers.stop(TIMER_T3510, true);
        self.timers.stop(TIMER_T3516, true);
        self.res_star = None;
        if let Some(t3502_byte) = rej.t3502_value {
            self.t3502_value = Some(GprsTimer3::from_byte(t3502_byte));
        }

        match cause {
            MmCause::IllegalUe | MmCause::IllegalMe | MmCause::FiveGsServicesNotAllowed => {
                // 5U3, delete 5G-GUTI/TAI list/ngKSI, USIM invalid, no retry
                self.state
                    .switch_update_status(UpdateStatus::RoamingNotAllowed);
                self.delete_registration_context();
                self.usim_invalid = true;
                self.state.switch_mm_state(MmSubState::Deregistered);
                vec![MmOutput::RegistrationFailed(cause)]
            }
            MmCause::PlmnNotAllowed => {
                self.state
                    .switch_update_status(UpdateStatus::RoamingNotAllowed);
                self.delete_registration_context();
                self.forbidden_plmns.push(self.identity.plmn_bcd);
                self.registration_attempt_counter = 0;
                self.state
                    .switch_mm_state(MmSubState::DeregisteredPlmnSearch);
                vec![
                    MmOutput::PlmnSearchNeeded,
                    MmOutput::RegistrationFailed(cause),
                ]
            }
            MmCause::TrackingAreaNotAllowed => {
                self.state
                    .switch_update_status(UpdateStatus::RoamingNotAllowed);
                self.delete_registration_context();
                if let Some(tai) = self.current_tai {
                    self.forbidden_tais_service.push(tai);
                }
                self.registration_attempt_counter = 0;
                self.state
                    .switch_mm_state(MmSubState::DeregisteredLimitedService);
                vec![MmOutput::RegistrationFailed(cause)]
            }
            MmCause::RoamingNotAllowedInTa => {
                self.state
                    .switch_update_status(UpdateStatus::RoamingNotAllowed);
                self.delete_registration_context();
                if let Some(tai) = self.current_tai {
                    self.forbidden_tais_roaming.push(tai);
                }
                self.registration_attempt_counter = 0;
                self.state
                    .switch_mm_state(MmSubState::DeregisteredPlmnSearch);
                vec![
                    MmOutput::PlmnSearchNeeded,
                    MmOutput::RegistrationFailed(cause),
                ]
            }
            MmCause::NoSuitableCellsInTa => {
                self.state
                    .switch_update_status(UpdateStatus::RoamingNotAllowed);
                if let Some(tai) = self.current_tai {
                    self.forbidden_tais_roaming.push(tai);
                }
                self.registration_attempt_counter = 0;
                self.state
                    .switch_mm_state(MmSubState::DeregisteredLimitedService);
                vec![MmOutput::RegistrationFailed(cause)]
            }
            MmCause::Congestion => {
                // TS 24.501 5.5.1.2.5 #22: start T3346 with the signalled
                // value; without a usable value treat as abnormal case
                let secs = rej.t3346_value.map(gprs_timer2_seconds).unwrap_or(0);
                if secs > 0 {
                    self.state.switch_update_status(UpdateStatus::NotUpdated);
                    self.registration_attempt_counter = 0;
                    self.timers
                        .start_with_timer2(TIMER_T3346, GprsTimer2::new(secs), true);
                    self.state
                        .switch_mm_state(MmSubState::DeregisteredAttemptingRegistration);
                    info!("T3346 (congestion backoff) started: {secs}s");
                    vec![MmOutput::RegistrationFailed(cause)]
                } else {
                    self.abnormal_registration_failure()
                }
            }
            MmCause::N1ModeNotAllowed => {
                self.state
                    .switch_update_status(UpdateStatus::RoamingNotAllowed);
                self.delete_registration_context();
                self.n1_mode_disabled = true;
                self.state.switch_mm_state(MmSubState::Deregistered);
                vec![MmOutput::RegistrationFailed(cause)]
            }
            MmCause::NoNetworkSlicesAvailable => {
                // #62: delete the allowed NSSAI, limited service
                self.allowed_nssai = None;
                self.registration_attempt_counter = 0;
                self.state
                    .switch_mm_state(MmSubState::DeregisteredLimitedService);
                vec![MmOutput::RegistrationFailed(cause)]
            }
            MmCause::Non3gppAccessTo5gcnNotAllowed => {
                // #72 applies to non-3GPP access; over 3GPP access this is
                // an abnormal case (TS 24.501 5.5.1.2.7)
                self.abnormal_registration_failure()
            }
            MmCause::ServingNetworkNotAuthorized => {
                self.state
                    .switch_update_status(UpdateStatus::RoamingNotAllowed);
                self.delete_registration_context();
                self.forbidden_plmns.push(self.identity.plmn_bcd);
                self.registration_attempt_counter = 0;
                self.state
                    .switch_mm_state(MmSubState::DeregisteredPlmnSearch);
                vec![
                    MmOutput::PlmnSearchNeeded,
                    MmOutput::RegistrationFailed(cause),
                ]
            }
            MmCause::TemporarilyNotAuthorizedForSnpn => {
                self.delete_registration_context();
                self.snpn_temporarily_forbidden = true;
                self.registration_attempt_counter = 0;
                self.state
                    .switch_mm_state(MmSubState::DeregisteredPlmnSearch);
                vec![
                    MmOutput::PlmnSearchNeeded,
                    MmOutput::RegistrationFailed(cause),
                ]
            }
            MmCause::PermanentlyNotAuthorizedForSnpn => {
                self.delete_registration_context();
                self.snpn_permanently_forbidden = true;
                self.registration_attempt_counter = 0;
                self.state
                    .switch_mm_state(MmSubState::DeregisteredPlmnSearch);
                vec![
                    MmOutput::PlmnSearchNeeded,
                    MmOutput::RegistrationFailed(cause),
                ]
            }
            _ => self.abnormal_registration_failure(),
        }
    }

    /// Abnormal registration failure (TS 24.501 Section 5.5.1.2.7):
    /// increment the attempt counter; below the limit retry via T3511,
    /// at the limit fall back to T3502.
    fn abnormal_registration_failure(&mut self) -> Vec<MmOutput> {
        self.registration_attempt_counter += 1;
        if self.registration_attempt_counter < MAX_REGISTRATION_ATTEMPTS {
            info!(
                "Registration attempt {} of {} failed; starting T3511",
                self.registration_attempt_counter, MAX_REGISTRATION_ATTEMPTS
            );
            self.state
                .switch_mm_state(MmSubState::DeregisteredAttemptingRegistration);
            self.timers.start(TIMER_T3511, true);
        } else {
            warn!(
                "Registration attempt counter reached {}; starting T3502",
                MAX_REGISTRATION_ATTEMPTS
            );
            self.state.switch_update_status(UpdateStatus::NotUpdated);
            self.state
                .switch_mm_state(MmSubState::DeregisteredAttemptingRegistration);
            if let Some(timer3) = self.t3502_value {
                self.timers.start_with_timer3(TIMER_T3502, timer3, true);
            } else {
                self.timers.start(TIMER_T3502, true);
            }
        }
        Vec::new()
    }

    /// Delete the stored registration context (5G-GUTI, TAI list, ngKSI)
    fn delete_registration_context(&mut self) {
        self.stored_guti = None;
        self.tai_list = None;
        self.sec.reset();
    }

    // ========================================================================
    // Authentication procedure (TS 24.501 Section 5.4.1, TS 33.501 6.1.3)
    // ========================================================================

    fn handle_authentication_request(&mut self, plain: &[u8]) -> Vec<MmOutput> {
        let req = match AuthenticationRequest::decode(&mut &plain[3..]) {
            Ok(req) => req,
            Err(e) => {
                warn!("Failed to decode Authentication Request: {e:?}");
                return Vec::new();
            }
        };

        // A new authentication request stops T3520 (TS 24.501 5.4.1.3.6 h)
        self.timers.stop(TIMER_T3520, true);

        // Store the network-signalled ABBA (TS 33.501 Annex A.7)
        self.abba = req.abba.value.clone();

        let (rand_ie, autn_ie) = match (&req.rand, &req.autn) {
            (Some(r), Some(a)) => (r, a),
            _ => {
                // EAP-AKA' primary auth (TS 24.501 5.4.1.2, TS 33.501 6.1.3.1):
                // RAND/AUTN are carried inside the EAP-Request/AKA'-Challenge.
                if let Some(ref eap_ie) = req.eap_message {
                    return self.handle_eap_aka_prime_challenge(&eap_ie.data, req.ng_ksi);
                }
                warn!("Authentication Request without RAND/AUTN and without EAP message: ignored");
                return Vec::new();
            }
        };
        let rand = &rand_ie.value;
        let autn = &autn_ie.value;
        if autn.len() != 16 {
            warn!("AUTN has invalid length {} (expected 16)", autn.len());
            return self.send_authentication_failure(MmCause::MacFailure, None);
        }

        let m = Milenage::new(&self.identity.k, &self.identity.opc);
        let ak = m.f5(rand);

        // AUTN = (SQN xor AK) || AMF || MAC
        let mut sqn_xor_ak = [0u8; 6];
        sqn_xor_ak.copy_from_slice(&autn[0..6]);
        let mut amf_autn = [0u8; 2];
        amf_autn.copy_from_slice(&autn[6..8]);
        let mac_autn = &autn[8..16];

        let mut sqn = [0u8; 6];
        for i in 0..6 {
            sqn[i] = sqn_xor_ak[i] ^ ak[i];
        }

        // 1) MAC check (TS 24.501 5.4.1.3.6 c: cause #20)
        let expected_mac = m.f1(rand, &sqn, &amf_autn);
        if expected_mac != mac_autn {
            warn!(
                "AUTN MAC failure: expected {:02x?}, got {:02x?}",
                expected_mac, mac_autn
            );
            self.auth_failure_count += 1;
            let mut outs = self.send_authentication_failure(MmCause::MacFailure, None);
            if self.auth_failure_count >= MAX_AUTH_FAILURES {
                // TS 24.501 5.4.1.3.7: after the third consecutive failure
                // the UE may treat the network as failed and reselect
                warn!("{MAX_AUTH_FAILURES} consecutive authentication failures: network considered invalid");
                outs.push(MmOutput::PlmnSearchNeeded);
            }
            return outs;
        }
        // Log the success case too. This is where the UE decides the network
        // proved knowledge of K (TS 33.501 §6.1.3.2), so it is as diagnostic as
        // the failure branch above -- and previously only failures were visible,
        // leaving a successful authentication with no positive marker.
        info!("AUTN MAC verified");

        // 2) AMF separation bit check (TS 33.501 6.1.3.2: bit must be 1
        // for 5G authentication; cause #26 otherwise)
        if amf_autn[0] & 0x80 == 0 {
            warn!("AMF separation bit not set: non-5G authentication unacceptable");
            self.auth_failure_count += 1;
            return self
                .send_authentication_failure(MmCause::Non5gAuthenticationUnacceptable, None);
        }

        // 3) SQN range check (TS 33.102 Annex C.2: cause #21 with AUTS)
        if !self.sqn_in_range(&sqn) {
            warn!(
                "SQN out of range (received {:02x?}, SQN_MS {:02x?}): synch failure",
                sqn, self.sqn_ms
            );
            self.auth_failure_count += 1;
            let auts = self.compute_auts(&m, rand);
            return self.send_authentication_failure(MmCause::SynchFailure, Some(auts));
        }

        // Authentication accepted
        self.sqn_ms = sqn;
        self.auth_failure_count = 0;

        let res = m.f2(rand);
        let ck = m.f3(rand);
        let ik = m.f4(rand);

        let sn_name = self.identity.sn_name.as_bytes();
        let res_star = derive_res_star(&ck, &ik, sn_name, rand, &res);
        self.res_star = Some(res_star);
        // TS 24.501 5.4.1.3.3: RES* is kept while T3516 runs
        self.timers.start(TIMER_T3516, true);

        // Key hierarchy (TS 33.501 Annex A): KAUSF -> KSEAF -> KAMF with
        // the received ABBA and the UE's SUPI
        let kausf = derive_kausf(&ck, &ik, sn_name, &sqn_xor_ak);
        let kseaf = derive_kseaf(&kausf, sn_name);
        let kamf = derive_kamf(&kseaf, self.identity.supi.as_bytes(), &self.abba);
        self.sec.keys_mut().set_kausf(&kausf);
        self.sec.keys_mut().set_kseaf(&kseaf);
        self.sec.keys_mut().set_kamf(&kamf);
        self.sec.keys_mut().abba = self.abba.clone();
        self.sec.set_nas_ksi(req.ng_ksi);
        self.sec.begin_establishing();

        info!(
            "5G-AKA succeeded: RES* computed, key hierarchy derived (ABBA {:02x?}, ngKSI {})",
            self.abba, req.ng_ksi.ksi
        );

        let response = AuthenticationResponse::with_res_star(res_star.to_vec());
        let mut pdu = Vec::new();
        response.encode(&mut pdu);
        let pdu = self.protect_if_active(pdu);
        info!(
            "Sending Authentication Response (RES* {} bytes)",
            res_star.len()
        );
        vec![MmOutput::SendNasPdu(pdu)]
    }

    /// EAP-AKA' primary authentication (TS 24.501 5.4.1.2, TS 33.501 6.1.3.1,
    /// RFC 9048): called when an AUTHENTICATION REQUEST carries the EAP-message
    /// IE instead of RAND/AUTN. Extracts AT_RAND/AT_AUTN/AT_KDF_INPUT from the
    /// EAP-Request/AKA'-Challenge, verifies the AUTN MAC, checks the SQN, builds
    /// an EAP-Response/AKA'-Challenge with AT_RES and AT_MAC, and derives the
    /// KAUSF/KSEAF/KAMF key hierarchy.
    fn handle_eap_aka_prime_challenge(
        &mut self,
        eap_bytes: &[u8],
        ng_ksi: NasKeySetIdentifier,
    ) -> Vec<MmOutput> {
        use nextgsim_crypto::eap_aka_prime::run_eap_aka_prime;
        use nextgsim_crypto::kdf::hmac_sha256;
        use nextgsim_nas::{
            decode_eap, encode_eap_to_vec, Eap, EapAkaPrime, EapAkaSubType, EapCode,
        };

        let Ok(eap) = decode_eap(&mut &eap_bytes[..]) else {
            warn!("EAP-AKA': failed to decode EAP message");
            return Vec::new();
        };
        let Eap::AkaPrime(ch) = eap else {
            warn!("EAP-AKA': EAP message is not AKA'");
            return Vec::new();
        };
        if ch.sub_type != EapAkaSubType::AkaChallenge {
            warn!("EAP-AKA': expected AKA-Challenge, got {:?}", ch.sub_type);
            return Vec::new();
        }
        let (Some(rand_v), Some(autn_v), Some(net_name)) = (
            ch.attributes.get_rand(),
            ch.attributes.get_autn(),
            ch.attributes.get_kdf_input(),
        ) else {
            warn!("EAP-AKA' Challenge missing AT_RAND/AT_AUTN/AT_KDF_INPUT");
            return Vec::new();
        };
        if rand_v.len() != 16 || autn_v.len() != 16 {
            warn!(
                "EAP-AKA' RAND/AUTN length mismatch: rand={}, autn={}",
                rand_v.len(),
                autn_v.len()
            );
            return Vec::new();
        }
        let mut rand = [0u8; 16];
        rand.copy_from_slice(&rand_v);
        let autn = &autn_v;

        let m = Milenage::new(&self.identity.k, &self.identity.opc);
        let ak = m.f5(&rand);
        let mut sqn = [0u8; 6];
        for i in 0..6 {
            sqn[i] = autn[i] ^ ak[i];
        }
        let mut amf_autn = [0u8; 2];
        amf_autn.copy_from_slice(&autn[6..8]);
        let mac_autn = &autn[8..16];

        if m.f1(&rand, &sqn, &amf_autn) != mac_autn {
            warn!("EAP-AKA' AUTN MAC failure");
            let rej = EapAkaPrime::authentication_reject(EapCode::Response, ch.id);
            let resp =
                AuthenticationResponse::with_eap_message(encode_eap_to_vec(&Eap::AkaPrime(rej)));
            let mut pdu = Vec::new();
            resp.encode(&mut pdu);
            return vec![MmOutput::SendNasPdu(self.protect_if_active(pdu))];
        }

        if !self.sqn_in_range(&sqn) {
            warn!(
                "EAP-AKA' SQN out of range (received {:02x?}, SQN_MS {:02x?}): synch failure",
                sqn, self.sqn_ms
            );
            let auts = self.compute_auts(&m, &rand);
            let mut sync = EapAkaPrime::synchronization_failure(EapCode::Response, ch.id);
            sync.attributes.put_auts(auts);
            let resp =
                AuthenticationResponse::with_eap_message(encode_eap_to_vec(&Eap::AkaPrime(sync)));
            let mut pdu = Vec::new();
            resp.encode(&mut pdu);
            return vec![MmOutput::SendNasPdu(self.protect_if_active(pdu))];
        }

        self.sqn_ms = sqn;
        let res = m.f2(&rand);
        let ck = m.f3(&rand);
        let ik = m.f4(&rand);

        // SQN xor AK is the first 6 octets of AUTN (before MAC computation)
        let mut sqn_xor_ak = [0u8; 6];
        sqn_xor_ak.copy_from_slice(&autn[0..6]);

        let (keys, kausf) = run_eap_aka_prime(
            &ck,
            &ik,
            &net_name,
            &sqn_xor_ak,
            self.identity.supi.as_bytes(),
        );

        // Build EAP-Response/AKA'-Challenge with AT_RES and AT_MAC (RFC 9048 §9.4)
        let mut resp_eap = EapAkaPrime::new(EapCode::Response, ch.id, EapAkaSubType::AkaChallenge);
        resp_eap.attributes.put_res(&res);
        resp_eap.attributes.put_mac(&[0u8; 16]); // placeholder for MAC computation
        let mut eap_msg = Eap::AkaPrime(resp_eap);
        // MAC covers the full EAP packet with AT_MAC zeroed, keyed by K_aut (RFC 9048 §6.3)
        let full_mac = hmac_sha256(&keys.k_aut, &encode_eap_to_vec(&eap_msg));
        if let Eap::AkaPrime(ref mut a) = eap_msg {
            a.attributes.replace_mac(&full_mac[..16]);
        }
        let eap_response = encode_eap_to_vec(&eap_msg);

        let kseaf = derive_kseaf(&kausf, &net_name);
        let kamf = derive_kamf(&kseaf, self.identity.supi.as_bytes(), &self.abba);
        self.sec.keys_mut().set_kausf(&kausf);
        self.sec.keys_mut().set_kseaf(&kseaf);
        self.sec.keys_mut().set_kamf(&kamf);
        self.sec.keys_mut().abba = self.abba.clone();
        self.sec.set_nas_ksi(ng_ksi);
        self.sec.begin_establishing();
        self.auth_failure_count = 0;

        info!(
            "EAP-AKA' challenge accepted; EAP-Response/AKA'-Challenge built, key hierarchy derived"
        );
        let resp = AuthenticationResponse::with_eap_message(eap_response);
        let mut pdu = Vec::new();
        resp.encode(&mut pdu);
        vec![MmOutput::SendNasPdu(self.protect_if_active(pdu))]
    }

    /// SQN acceptance check (TS 33.102 Annex C.2.1): the received SQN must
    /// be greater than the highest previously accepted SQN and within the
    /// wrap-around window delta.
    fn sqn_in_range(&self, sqn: &[u8; 6]) -> bool {
        let received = sqn_to_u64(sqn);
        let current = sqn_to_u64(&self.sqn_ms);
        received > current && (received - current) < SQN_DELTA
    }

    /// Compute the AUTS resynchronisation parameter
    /// (TS 33.102 Section 6.3.3): AUTS = (SQN_MS xor AK*) || MAC-S with
    /// AK* = f5*(RAND) and MAC-S = f1*(K, RAND, SQN_MS, AMF*) where AMF*
    /// is all zeros (TS 33.102 Annex C.2.2).
    fn compute_auts(&self, m: &Milenage, rand: &[u8; 16]) -> Vec<u8> {
        let ak_star = m.f5_star(rand);
        let amf_star = [0u8, 0u8];
        let mac_s = m.f1_star(rand, &self.sqn_ms, &amf_star);

        let mut auts = Vec::with_capacity(14);
        for (sqn_byte, ak_byte) in self.sqn_ms.iter().zip(ak_star.iter()) {
            auts.push(sqn_byte ^ ak_byte);
        }
        auts.extend_from_slice(&mac_s);
        auts
    }

    fn send_authentication_failure(
        &mut self,
        cause: MmCause,
        auts: Option<Vec<u8>>,
    ) -> Vec<MmOutput> {
        let failure = match auts {
            Some(auts) => AuthenticationFailure::synch_failure(auts),
            None => AuthenticationFailure::with_cause(cause),
        };
        let mut pdu = Vec::new();
        failure.encode(&mut pdu);
        // TS 24.501 5.4.1.3.6: start T3520 when sending Authentication Failure
        self.timers.start(TIMER_T3520, true);
        info!("Sending Authentication Failure (cause #{})", cause as u8);
        let pdu = self.protect_if_active(pdu);
        vec![MmOutput::SendNasPdu(pdu)]
    }

    /// Authentication Reject handling (TS 24.501 Section 5.4.1.3.5)
    fn handle_authentication_reject(&mut self) -> Vec<MmOutput> {
        warn!("Authentication Reject received: USIM considered invalid");
        self.timers.stop(TIMER_T3510, true);
        self.timers.stop(TIMER_T3516, true);
        self.timers.stop(TIMER_T3517, true);
        self.timers.stop(TIMER_T3520, true);
        self.timers.stop(TIMER_T3521, true);
        self.res_star = None;
        self.usim_invalid = true;
        self.state
            .switch_update_status(UpdateStatus::RoamingNotAllowed);
        self.delete_registration_context();
        self.state.switch_mm_state(MmSubState::Deregistered);
        vec![MmOutput::AuthenticationRejected]
    }

    // ========================================================================
    // Security mode control (TS 24.501 Section 5.4.2, TS 33.501 6.7.2)
    // ========================================================================

    /// Whether the UE is operating in an emergency-services context, in which
    /// the network is permitted to select NIA0 / NEA0 (TS 33.501 5.5.2).
    fn in_emergency_context(&self) -> bool {
        self.pending_reg_type == RegistrationType::EmergencyRegistration
    }

    /// Handle a Security Mode Command received with a "new security
    /// context" security header (SHT 0x03/0x04). The NAS-MAC is verified
    /// with the *new* keys before the context is activated.
    fn handle_security_mode_command(
        &mut self,
        payload: &[u8],
        sqn: u8,
        mac: [u8; 4],
    ) -> Vec<MmOutput> {
        let smc = match SecurityModeCommand::decode(&mut &payload[3..]) {
            Ok(smc) => smc,
            Err(e) => {
                warn!("Failed to decode Security Mode Command: {e:?}");
                return self.send_security_mode_reject(MmCause::InvalidMandatoryInformation);
            }
        };

        // 1) Bidding-down check (TS 24.501 5.4.2.3): the replayed UE
        // security capabilities must match what the UE sent
        let mut replayed = Vec::new();
        smc.replayed_ue_security_capabilities.encode(&mut replayed);
        // encode() output is [len, EA octet, IA octet, ...]
        if replayed.len() < 3
            || replayed[1] != self.identity.ea_cap
            || replayed[2] != self.identity.ia_cap
        {
            warn!(
                "SMC replayed UE security capabilities mismatch: sent [{:02x} {:02x}], replayed {:02x?}",
                self.identity.ea_cap,
                self.identity.ia_cap,
                &replayed[1..]
            );
            return self.send_security_mode_reject(MmCause::UeSecurityCapabilitiesMismatch);
        }

        // 2) Selected algorithms must be valid and within the UE's
        // advertised capabilities
        let algs = &smc.selected_nas_security_algorithms;
        let Ok(cipher_alg) = CipheringAlgorithm::try_from(algs.ciphering) else {
            warn!(
                "SMC selected unknown ciphering algorithm {}",
                algs.ciphering
            );
            return self.send_security_mode_reject(MmCause::UeSecurityCapabilitiesMismatch);
        };
        let Ok(integ_alg) = IntegrityAlgorithm::try_from(algs.integrity) else {
            warn!(
                "SMC selected unknown integrity algorithm {}",
                algs.integrity
            );
            return self.send_security_mode_reject(MmCause::UeSecurityCapabilitiesMismatch);
        };
        // TS 33.501 5.5.2 / TS 24.501 5.4.2.3: NIA0 (null integrity) may be
        // selected only for an emergency-services / unauthenticated UE. Outside an
        // emergency context an SMC selecting NIA0 is out-of-capability and must be
        // rejected (otherwise it would silently drop NAS integrity protection).
        if integ_alg == IntegrityAlgorithm::Nia0 && !self.in_emergency_context() {
            warn!("SMC selected NIA0 (null integrity) outside an emergency context: rejecting");
            return self.send_security_mode_reject(MmCause::UeSecurityCapabilitiesMismatch);
        }
        if !alg_in_capability(self.identity.ea_cap, cipher_alg as u8)
            || (integ_alg != IntegrityAlgorithm::Nia0
                && !alg_in_capability(self.identity.ia_cap, integ_alg as u8))
        {
            warn!(
                "SMC selected algorithms outside UE capabilities: NEA{} / NIA{}",
                cipher_alg as u8, integ_alg as u8
            );
            return self.send_security_mode_reject(MmCause::UeSecurityCapabilitiesMismatch);
        }

        // 3) A partial security context from authentication is required
        if !self.sec.keys().has_kamf() {
            warn!("SMC received without a partial security context (no KAMF)");
            return self.send_security_mode_reject(MmCause::SecurityModeRejectedUnspecified);
        }

        // 4) If the SMC signals a (different) ABBA, re-derive KAMF with it
        // (TS 33.501 Annex A.7: ABBA is a KAMF derivation input)
        if let Some(ref abba) = smc.abba {
            if *abba != self.abba {
                let Some(kseaf) = self.sec.keys().kseaf().copied() else {
                    warn!("SMC carries new ABBA but no KSEAF stored");
                    return self
                        .send_security_mode_reject(MmCause::SecurityModeRejectedUnspecified);
                };
                info!("SMC carries new ABBA {abba:02x?}: re-deriving KAMF");
                self.abba = abba.clone();
                let kamf = derive_kamf(&kseaf, self.identity.supi.as_bytes(), &self.abba);
                self.sec.keys_mut().set_kamf(&kamf);
                self.sec.keys_mut().abba = self.abba.clone();
            }
        }

        // 5) Derive the NAS keys and verify the SMC NAS-MAC *before*
        // activating the new context (TS 33.501 6.7.2)
        if let Err(e) = self.sec.derive_nas_keys(cipher_alg, integ_alg) {
            warn!("NAS key derivation failed: {e}");
            return self.send_security_mode_reject(MmCause::SecurityModeRejectedUnspecified);
        }
        self.sec.reset_counts();

        let count = NasCount::new(0, sqn);
        let knas_int = *self.sec.keys().knas_int().expect("knas_int derived above");
        if let Err(e) = verify_nas_mac(
            integ_alg,
            &knas_int,
            &count,
            NasDirection::Downlink,
            sqn,
            payload,
            &mac,
        ) {
            warn!("SMC NAS-MAC verification failed ({e}): rejecting, keys NOT activated");
            return self.send_security_mode_reject(MmCause::SecurityModeRejectedUnspecified);
        }

        // MAC verified: activate the new security context
        self.sec.set_nas_ksi(smc.ng_ksi);
        self.sec.update_downlink_count(&count);
        self.sec.activate();
        // TS 24.501 5.4.1.3.3: stop T3516 on Security Mode Command
        self.timers.stop(TIMER_T3516, true);
        info!(
            "Security Mode Command accepted: security context activated (NEA{} / NIA{}, ngKSI {})",
            cipher_alg as u8, integ_alg as u8, smc.ng_ksi.ksi
        );

        // Build the Security Mode Complete
        let mut complete = SecurityModeComplete::new();
        if smc.imeisv_request.is_some() {
            complete.imeisv = Some(build_imeisv_identity(&self.identity.imei_sv));
        }
        // The strict core requires the replayed initial NAS message in the
        // NAS message container (RINMR, TS 24.501 5.4.2.3)
        complete.nas_message_container = self.last_registration_request.clone();

        let mut plain = Vec::new();
        complete.encode(&mut plain);
        info!(
            "Sending Security Mode Complete (container={} bytes)",
            complete.nas_message_container.as_ref().map_or(0, Vec::len)
        );
        let pdu = self.protect_uplink(
            plain,
            SecurityHeaderType::IntegrityProtectedAndCipheredWithNewSecurityContext,
        );
        let mut outs = vec![MmOutput::SendNasPdu(pdu)];
        // Wave-6 I5: once NAS security is active, hand the derived KgNB to the
        // RRC plane so it can derive K_RRCint/K_RRCenc when the AS
        // SecurityModeCommand arrives (TS 33.501 §6.9.4.1). Gated behind the
        // AS-security wire knob; the exact uplink NAS COUNT the AMF uses for the
        // KgNB derivation is a cross-stack detail verified in the docker E2E
        // sign-off (hazard #269, host-only).
        if crate::rrc::security::I5_UE_AS_SECURITY {
            if let Some(kgnb) = self.derive_kgnb_for_as_security() {
                outs.push(MmOutput::AsSecurityKgnb(kgnb));
            }
        }
        outs
    }

    /// Derive KgNB for AS security from the active NAS context (TS 33.501
    /// §6.9.4.1: `KgNB = KDF(KAMF, uplink NAS COUNT, 3GPP-access)`). Returns
    /// `None` when no KAMF is available (NAS security not established). The gNB
    /// receives the same KgNB from the AMF in the NGAP InitialContextSetupRequest
    /// SecurityKey IE, so both ends derive identical K_RRCint/K_RRCenc.
    pub fn derive_kgnb_for_as_security(&self) -> Option<[u8; 32]> {
        let kamf = self.sec.keys().kamf()?;
        let ul_count = self.sec.uplink_count().to_u32();
        Some(derive_kgnb(kamf, ul_count, 0x01))
    }

    fn send_security_mode_reject(&mut self, cause: MmCause) -> Vec<MmOutput> {
        let reject = SecurityModeReject::new(cause);
        let mut pdu = Vec::new();
        reject.encode(&mut pdu);
        info!("Sending Security Mode Reject (cause #{})", cause as u8);
        let pdu = self.protect_if_active(pdu);
        vec![MmOutput::SendNasPdu(pdu)]
    }

    // ========================================================================
    // Service request procedure (TS 24.501 Section 5.6.1)
    // ========================================================================

    /// Start a service request procedure. Requires a stored 5G-GUTI
    /// (the 5G-S-TMSI is derived from it).
    pub fn start_service_request(
        &mut self,
        service_type: ServiceType,
        uplink_data_status: Option<u16>,
    ) -> Vec<MmOutput> {
        if !self.state.is_registered() {
            warn!("Service Request not started: UE not registered");
            return Vec::new();
        }
        let Some(s_tmsi) = self.build_s_tmsi() else {
            warn!("Service Request not started: no stored 5G-GUTI");
            return Vec::new();
        };

        let mut req =
            ServiceRequest::new(self.sec.nas_ksi(), IeServiceType::new(service_type), s_tmsi);
        req.uplink_data_status = uplink_data_status;

        let mut plain = Vec::new();
        req.encode(&mut plain);

        self.state
            .switch_mm_state(MmSubState::ServiceRequestInitiated);
        self.timers.start(TIMER_T3517, true);
        info!("Sending Service Request ({service_type:?}), T3517 started");
        let pdu = self.protect_if_active(plain);
        vec![MmOutput::SendNasPdu(pdu)]
    }

    /// Derive the 5G-S-TMSI mobile identity from the stored 5G-GUTI
    /// (TS 24.501 Section 9.11.3.4: AMF set ID + AMF pointer + 5G-TMSI).
    fn build_s_tmsi(&self) -> Option<Ie5gsMobileIdentity> {
        let guti = self.stored_guti.as_ref()?;
        // GUTI layout: type(1) + PLMN(3) + AMF region(1) + AMF set/ptr(2) + TMSI(4)
        if guti.data.len() < 11 {
            return None;
        }
        let mut data = Vec::with_capacity(7);
        data.push(0xF0 | (MobileIdentityType::Tmsi as u8));
        data.extend_from_slice(&guti.data[5..11]);
        Some(Ie5gsMobileIdentity::new(MobileIdentityType::Tmsi, data))
    }

    fn handle_service_accept(&mut self) -> Vec<MmOutput> {
        info!("Service Accept received");
        self.timers.stop(TIMER_T3517, true);
        self.timers.stop(TIMER_T3516, true);
        self.res_star = None;
        self.state
            .switch_mm_state(MmSubState::RegisteredNormalService);
        self.state.switch_cm_state(CmState::Connected);
        Vec::new()
    }

    fn handle_service_reject(&mut self, plain: &[u8]) -> Vec<MmOutput> {
        let rej = match ServiceReject::decode(&mut &plain[3..]) {
            Ok(rej) => rej,
            Err(e) => {
                warn!("Failed to decode Service Reject: {e:?}");
                return Vec::new();
            }
        };
        let cause = rej.mm_cause.value;
        warn!("Service Reject: cause {:?} (#{})", cause, cause as u8);
        self.timers.stop(TIMER_T3517, true);

        match cause {
            MmCause::IllegalUe | MmCause::IllegalMe | MmCause::FiveGsServicesNotAllowed => {
                // Same terminal handling as for registration (5.6.1.5)
                self.state
                    .switch_update_status(UpdateStatus::RoamingNotAllowed);
                self.delete_registration_context();
                self.usim_invalid = true;
                self.state.switch_mm_state(MmSubState::Deregistered);
                vec![MmOutput::RegistrationFailed(cause)]
            }
            MmCause::UeIdentityCannotBeDerived | MmCause::ImplicitlyDeregistered => {
                // TS 24.501 5.6.1.5: delete the GUTI and perform a fresh
                // initial registration
                self.delete_registration_context();
                self.state
                    .switch_mm_state(MmSubState::DeregisteredNormalService);
                self.build_and_send_registration(RegistrationType::InitialRegistration)
            }
            MmCause::Congestion => {
                let secs = rej.t3346_value.map(gprs_timer2_seconds).unwrap_or(0);
                if secs > 0 {
                    self.timers
                        .start_with_timer2(TIMER_T3346, GprsTimer2::new(secs), true);
                    info!("T3346 (congestion backoff) started: {secs}s");
                }
                self.state
                    .switch_mm_state(MmSubState::RegisteredNormalService);
                Vec::new()
            }
            _ => {
                self.state
                    .switch_mm_state(MmSubState::RegisteredNormalService);
                Vec::new()
            }
        }
    }

    // ========================================================================
    // Deregistration procedure (TS 24.501 Section 5.5.2)
    // ========================================================================

    /// Start a UE-originating deregistration.
    pub fn start_deregistration(&mut self, switch_off: bool) -> Vec<MmOutput> {
        use nextgsim_nas::ies::ie1::{
            DeRegistrationAccessType, IeDeRegistrationType, ReRegistrationRequired,
            SwitchOff as NasSwitchOff,
        };

        let dereg_type = IeDeRegistrationType::new(
            DeRegistrationAccessType::ThreeGppAccess,
            ReRegistrationRequired::NotRequired,
            if switch_off {
                NasSwitchOff::SwitchOff
            } else {
                NasSwitchOff::NormalDeRegistration
            },
        );
        let mobile_identity = if let Some(ref guti) = self.stored_guti {
            guti.clone()
        } else {
            // §5.5.2.2.2: store the SUCI and arm T3519 under the same rule.
            let suci = self.identity.suci.clone();
            self.store_suci_and_arm_t3519(suci.clone());
            Ie5gsMobileIdentity::new(MobileIdentityType::Suci, suci)
        };
        let req = DeregistrationRequestUeOriginating::new(
            dereg_type,
            self.sec.nas_ksi(),
            mobile_identity,
        );
        let mut plain = Vec::new();
        req.encode(&mut plain);

        self.state
            .switch_mm_state(MmSubState::DeregisteredInitiated);
        if switch_off {
            // No response expected on switch-off (5.5.2.2.4)
            self.dereg_pdu = None;
        } else {
            self.dereg_pdu = Some(plain.clone());
            self.timers.start(TIMER_T3521, true);
        }
        info!("Sending Deregistration Request (switch_off={switch_off})");
        let pdu = self.protect_if_active(plain);
        vec![MmOutput::SendNasPdu(pdu)]
    }

    fn handle_deregistration_accept(&mut self) -> Vec<MmOutput> {
        info!("Deregistration Accept received");
        self.timers.stop(TIMER_T3521, true);
        self.timers.stop(TIMER_T3512, true);
        self.dereg_pdu = None;
        self.state
            .switch_mm_state(MmSubState::DeregisteredNormalService);
        Vec::new()
    }

    /// Handle a network-initiated DEREGISTRATION REQUEST (UE terminated, `0x47`).
    ///
    /// TS 24.501 §5.5.2.3.2: for a de-registration that is not "switch off" the
    /// UE sends a DEREGISTRATION ACCEPT and enters `5GMM-DEREGISTERED`; when the
    /// de-registration type indicates re-registration is required the UE locally
    /// releases its resources and initiates a fresh registration. `plain` is the
    /// deciphered 5GMM PDU (`plain[0..3]` = header, `plain[3..]` = the message
    /// body). The message reaches this arm only when integrity protected — a
    /// plain `0x47` is rejected earlier by the TS 24.501 §4.4.4.2 allow-list.
    fn handle_network_deregistration(&mut self, plain: &[u8]) -> Vec<MmOutput> {
        let mut body = &plain[3..];
        let msg = match DeregistrationRequestUeTerminated::decode(&mut body) {
            Ok(m) => m,
            Err(e) => {
                warn!("Malformed network DEREGISTRATION REQUEST: {e}");
                return Vec::new();
            }
        };

        // Drive the TS 24.501 §5.5.2.3.2 receive-side procedure with the current
        // registration state.
        let rm_state = self.state.rm_state();
        let mm_state = self.state.mm_state();
        let mut proc = DeregistrationProcedure::new();
        let result = match proc.receive_deregistration_request(&msg, rm_state, mm_state) {
            Ok(r) => r,
            Err(e) => {
                warn!("Network de-registration not processed: {e}");
                return Vec::new();
            }
        };

        // Apply the network-signalled context clearing.
        if result.clear_guti {
            self.stored_guti = None;
        }
        if result.clear_tai_list {
            self.tai_list = None;
            self.current_tai = None;
        }
        if result.clear_equivalent_plmn {
            self.equivalent_plmns.clear();
        }
        if result.invalidate_usim {
            self.usim_invalid = true;
        }
        if let Some(status) = result.new_update_status {
            self.state.switch_update_status(status);
        }

        let mut outs = Vec::new();

        // Send DEREGISTRATION ACCEPT unless the procedure suppressed it (e.g. a
        // non-3GPP-access request, where an MM status is sent instead).
        if let Some(accept) = result.accept {
            let mut pdu = Vec::new();
            accept.encode(&mut pdu);
            info!("Network-initiated de-registration: sending Deregistration Accept");
            outs.push(MmOutput::SendNasPdu(self.protect_if_active(pdu)));
        }

        if let Some(sub) = result.new_sub_state {
            self.state.switch_mm_state(sub);
        }
        self.dereg_pdu = None;

        // §5.5.2.3.2: re-registration required — release local resources and
        // start a fresh initial registration (mirrors the implicit-dereg path).
        if result.re_registration_required {
            let _ = proc.perform_local_deregistration();
            self.delete_registration_context();
            self.state
                .switch_mm_state(MmSubState::DeregisteredNormalService);
            outs.extend(self.build_and_send_registration(RegistrationType::InitialRegistration));
        }

        outs
    }

    // ========================================================================
    // Downlink NAS handling (TS 24.501 Section 4.4)
    // ========================================================================

    /// Process a downlink 5GMM NAS PDU (raw, possibly security protected).
    ///
    /// Verifies/unwraps the outer security header, dispatches procedure
    /// messages owned by the orchestrator and returns
    /// [`MmOutput::NotHandled`] with the deciphered plain PDU for
    /// everything else.
    pub fn handle_downlink(&mut self, raw: &[u8]) -> Vec<MmOutput> {
        if raw.len() < 3 || raw[0] != 0x7E {
            warn!("Downlink NAS PDU too short or not 5GMM");
            return Vec::new();
        }
        let sht_byte = raw[1] & 0x0F;
        let Ok(sht) = SecurityHeaderType::try_from(sht_byte) else {
            warn!("Unknown security header type 0x{sht_byte:02x}");
            return Vec::new();
        };

        if sht == SecurityHeaderType::NotProtected {
            // TS 24.501 4.4.4.2 (last paragraph): once a secure exchange of NAS
            // messages has been established, the UE shall discard any NAS
            // signalling message received without integrity protection.
            if self.sec.is_active() {
                warn!(
                    "Unprotected NAS message received after security activation: discarded (TS 24.501 4.4.4.2)"
                );
                return Vec::new();
            }
            return self.handle_plain(raw.to_vec(), false);
        }

        if raw.len() < 8 {
            warn!("Secured NAS PDU too short: {} bytes", raw.len());
            return Vec::new();
        }
        let mut mac = [0u8; 4];
        mac.copy_from_slice(&raw[2..6]);
        let sqn = raw[6];
        let payload = &raw[7..];

        // A Security Mode Command with a new security context is verified
        // with the keys signalled inside the message itself
        if sht.is_new_security_context()
            && payload.len() >= 3
            && payload[0] == 0x7E
            && payload[2] == u8::from(MmMessageType::SecurityModeCommand)
        {
            return self.handle_security_mode_command(payload, sqn, mac);
        }

        if !self.sec.is_active() {
            warn!(
                "Protected NAS message (SHT 0x{sht_byte:02x}) received without an active security context: discarded"
            );
            return Vec::new();
        }

        // Estimate the downlink count and verify the MAC over SQN||payload
        // (the payload is still ciphered at this point, matching the
        // sender's cipher-then-MAC order)
        let count = self.sec.estimate_downlink_count(sqn);
        let knas_int = match self.sec.keys().knas_int() {
            Some(k) => *k,
            None => {
                warn!("Active security context without KNASint: discarding message");
                return Vec::new();
            }
        };
        if let Err(e) = verify_nas_mac(
            self.sec.integrity_algorithm(),
            &knas_int,
            &count,
            NasDirection::Downlink,
            sqn,
            payload,
            &mac,
        ) {
            // TS 24.501 4.4.4.2: discard messages failing the integrity check
            warn!("Downlink NAS-MAC verification failed ({e}): message discarded");
            return Vec::new();
        }
        // TS 24.501 4.4.3.1 / 4.4.4.2: after a successful integrity check the
        // estimated downlink NAS COUNT must be strictly greater than the last
        // accepted one; a replayed or non-monotonic COUNT is discarded.
        if !self.sec.is_acceptable_downlink_count(&count) {
            warn!(
                "Downlink NAS COUNT {} not greater than last accepted {}: replay / non-monotonic, discarded",
                count.to_u32(),
                self.sec.downlink_count().to_u32()
            );
            return Vec::new();
        }
        self.sec.update_downlink_count(&count);

        let mut plain = payload.to_vec();
        if sht.is_ciphered() {
            if let Some(key) = self.sec.keys().knas_enc() {
                nas_cipher(
                    self.sec.ciphering_algorithm(),
                    key,
                    &count,
                    NAS_BEARER,
                    NasDirection::Downlink,
                    &mut plain,
                );
            }
        }

        self.handle_plain(plain, true)
    }

    /// Dispatch a plain (inner) 5GMM message.
    fn handle_plain(&mut self, plain: Vec<u8>, protected: bool) -> Vec<MmOutput> {
        if plain.len() < 3 {
            warn!("Plain NAS PDU too short: {} bytes", plain.len());
            return Vec::new();
        }
        let Ok(msg_type) = MmMessageType::try_from(plain[2]) else {
            warn!("Unknown 5GMM message type 0x{:02x}", plain[2]);
            return Vec::new();
        };
        debug!("5GMM message {:?} (protected={protected})", msg_type);

        // TS 24.501 4.4.4.2: before a secure exchange is established, only a
        // closed allow-list of messages may be processed without integrity
        // protection. Anything else (e.g. REGISTRATION ACCEPT, SERVICE ACCEPT,
        // CONFIGURATION UPDATE COMMAND) is discarded when received in the clear.
        if !protected && !self.allowed_when_plain(msg_type, &plain) {
            warn!(
                "Plain (unprotected) {msg_type:?} not on the TS 24.501 4.4.4.2 allow-list: discarded"
            );
            return Vec::new();
        }

        match msg_type {
            MmMessageType::AuthenticationRequest => self.handle_authentication_request(&plain),
            MmMessageType::AuthenticationReject => self.handle_authentication_reject(),
            MmMessageType::RegistrationAccept => self.handle_registration_accept(&plain),
            MmMessageType::RegistrationReject => self.handle_registration_reject(&plain),
            MmMessageType::ServiceAccept => self.handle_service_accept(),
            MmMessageType::ServiceReject => self.handle_service_reject(&plain),
            MmMessageType::DeregistrationAcceptUeOriginating => self.handle_deregistration_accept(),
            MmMessageType::DeregistrationRequestUeTerminated => {
                self.handle_network_deregistration(&plain)
            }
            MmMessageType::SecurityModeCommand => {
                // TS 24.501 4.4.4.2: an SMC must arrive integrity protected
                // with a new-context security header; a plain SMC is discarded
                warn!("Security Mode Command without integrity protection: discarded");
                Vec::new()
            }
            _ => vec![MmOutput::NotHandled(plain)],
        }
    }

    /// TS 24.501 4.4.4.2 allow-list: messages the UE may process when received
    /// without integrity protection (before a secure exchange is established).
    ///
    /// `plain` is the inner 5GMM message (`plain[2]` = message type, `plain[3]`
    /// = the first mandatory IE octet — the 5GMM cause for reject messages, or
    /// the 5GS identity type for IDENTITY REQUEST).
    fn allowed_when_plain(&self, msg_type: MmMessageType, plain: &[u8]) -> bool {
        match msg_type {
            // Authentication exchange runs before security is established.
            MmMessageType::AuthenticationRequest
            | MmMessageType::AuthenticationReject
            | MmMessageType::AuthenticationResult => true,
            // IDENTITY REQUEST is acceptable in the clear only when it requests
            // the SUCI (5GS identity type 0b001); any other requested identity
            // would leak a persistent identifier and must be protected.
            MmMessageType::IdentityRequest => plain.get(3).is_some_and(|b| b & 0x0F == 0x01),
            // A UE-originating DEREGISTRATION ACCEPT (non switch-off) is allowed.
            MmMessageType::DeregistrationAcceptUeOriginating => true,
            // REGISTRATION REJECT is processable in the clear only if its 5GMM
            // cause is not #76/#78/#81/#82 (those drive persistent forbidden-list
            // / SNPN state and must be integrity protected).
            MmMessageType::RegistrationReject => !matches!(plain.get(3), Some(76 | 78 | 81 | 82)),
            // SERVICE REJECT is processable in the clear only if its 5GMM cause
            // is not #76/#78.
            MmMessageType::ServiceReject => !matches!(plain.get(3), Some(76 | 78)),
            // Everything else (REGISTRATION ACCEPT, SERVICE ACCEPT, CONFIGURATION
            // UPDATE COMMAND, SECURITY MODE COMMAND, ...) must be protected.
            _ => false,
        }
    }

    // ========================================================================
    // UE policy delivery service (UPDP) — TS 24.501 Annex D (Wave-6 E8)
    // ========================================================================

    /// Number of UE policy sections currently stored (test/observability hook).
    pub fn ue_policy_section_count(&self) -> usize {
        self.ue_policy_sections.len()
    }

    /// The stored UE policy section for `(plmn, upsc)`, if any (test hook).
    pub fn ue_policy_section(&self, plmn: PlmnId, upsc: u16) -> Option<&StoredUePolicySection> {
        self.ue_policy_sections.get(&(plmn, upsc))
    }

    /// Handle a received "UE policy container" payload (the MANAGE UE POLICY
    /// COMMAND content carried in a DL NAS TRANSPORT, TS 24.501 D.2.1.3).
    ///
    /// Decodes the command, installs/replaces/deletes the addressed sections
    /// keyed by `(PLMN, UPSC)` and returns the UE policy delivery service
    /// message content to send back uplink:
    ///
    /// - all instructions stored → MANAGE UE POLICY COMPLETE (echoing the PTI);
    /// - one or more instructions unstorable → MANAGE UE POLICY COMMAND REJECT
    ///   with a per-instruction D.6.3 result (fail-closed: the UE never
    ///   COMPLETEs a section it did not store);
    /// - a malformed command still carrying a recoverable PCF PTI → REJECT
    ///   (never a panic — remote-DoS caution mirroring the SM plane);
    /// - a command too short to carry a PTI → [`UePolicyReaction::Ignore`].
    pub fn handle_ue_policy_command(&mut self, container: &[u8]) -> UePolicyReaction {
        match ManageUePolicyCommand::decode(container) {
            Ok(cmd) => self.apply_ue_policy_command(&cmd),
            Err(e) => {
                warn!("MANAGE UE POLICY COMMAND decode failed ({e}); rejecting");
                self.reject_unparseable_command(container)
            }
        }
    }

    /// Install/replace/delete the sections of a decoded command and build the
    /// COMPLETE or COMMAND REJECT reply.
    fn apply_ue_policy_command(&mut self, cmd: &ManageUePolicyCommand) -> UePolicyReaction {
        let mut subresults: Vec<UePolicySectionManagementSubresult> = Vec::new();
        for sublist in &cmd.sublists {
            let mut failed: Vec<UePolicyResult> = Vec::new();
            for (idx, instr) in sublist.instructions.iter().enumerate() {
                // 1-based instruction order within the sublist contents (D.6.3).
                let order = (idx + 1) as u16;
                if instr.parts.is_empty() {
                    // D.2.1.3: an instruction with no parts deletes the section.
                    self.ue_policy_sections
                        .remove(&(sublist.plmn_id, instr.upsc));
                    debug!(
                        "UE policy: deleted section (PLMN {:?}, UPSC {})",
                        sublist.plmn_id, instr.upsc
                    );
                    continue;
                }
                // Fail-closed: validate every URSP part BEFORE storing anything
                // for this instruction — never store a half-decoded section.
                let storable = instr.parts.iter().all(|part| {
                    if part.part_type == UePolicyPartType::Ursp {
                        match part.decode_ursp() {
                            Ok(_) => true,
                            Err(e) => {
                                warn!(
                                    "UE policy: URSP part of (PLMN {:?}, UPSC {}) is \
                                     undecodable: {e}",
                                    sublist.plmn_id, instr.upsc
                                );
                                false
                            }
                        }
                    } else {
                        // Non-URSP part types are stored opaquely (framing was
                        // already validated by the command decoder).
                        true
                    }
                });
                if storable {
                    self.ue_policy_sections.insert(
                        (sublist.plmn_id, instr.upsc),
                        StoredUePolicySection {
                            parts: instr.parts.clone(),
                        },
                    );
                    debug!(
                        "UE policy: stored section (PLMN {:?}, UPSC {}, {} part(s))",
                        sublist.plmn_id,
                        instr.upsc,
                        instr.parts.len()
                    );
                } else {
                    failed.push(UePolicyResult {
                        upsc: instr.upsc,
                        failed_instruction_order: order,
                        cause: CAUSE_PROTOCOL_ERROR_UNSPECIFIED,
                    });
                }
            }
            if !failed.is_empty() {
                subresults.push(UePolicySectionManagementSubresult {
                    plmn_id: sublist.plmn_id,
                    results: failed,
                });
            }
        }

        if subresults.is_empty() {
            self.encode_updp_reply(
                ManageUePolicyComplete { pti: cmd.pti }.encode(),
                "MANAGE UE POLICY COMPLETE",
            )
        } else {
            self.encode_updp_reply(
                ManageUePolicyCommandReject {
                    pti: cmd.pti,
                    result: UePolicySectionManagementResult { subresults },
                }
                .encode(),
                "MANAGE UE POLICY COMMAND REJECT",
            )
        }
    }

    /// Best-effort REJECT for a command that failed to frame-decode. Recovers
    /// the PTI (octet 1) only when the message is at least PTI + message type,
    /// is a MANAGE UE POLICY COMMAND and carries a PCF-allocated PTI; otherwise
    /// a spec-valid (PTI 80H-FEH) REJECT cannot be formed and the command is
    /// dropped.
    fn reject_unparseable_command(&self, container: &[u8]) -> UePolicyReaction {
        if container.len() < 2
            || container[1] != UPDP_MSG_MANAGE_UE_POLICY_COMMAND
            || !(PCF_PTI_MIN..=PCF_PTI_MAX).contains(&container[0])
        {
            return UePolicyReaction::Ignore;
        }
        let pti = container[0];
        // The specific failed instruction cannot be identified in an
        // unparseable command; report the home PLMN, UPSC 0, order 1.
        let Ok(plmn_id) = PlmnId::from_bcd(self.identity.plmn_bcd) else {
            return UePolicyReaction::Ignore;
        };
        let reject = ManageUePolicyCommandReject {
            pti,
            result: UePolicySectionManagementResult {
                subresults: vec![UePolicySectionManagementSubresult {
                    plmn_id,
                    results: vec![UePolicyResult {
                        upsc: 0,
                        failed_instruction_order: 1,
                        cause: CAUSE_PROTOCOL_ERROR_UNSPECIFIED,
                    }],
                }],
            },
        };
        self.encode_updp_reply(reject.encode(), "MANAGE UE POLICY COMMAND REJECT")
    }

    /// Wrap an encoder result into a [`UePolicyReaction`]; an (unexpected)
    /// encode failure drops the reply rather than sending malformed bytes.
    fn encode_updp_reply(
        &self,
        encoded: Result<Vec<u8>, UePolicyError>,
        what: &str,
    ) -> UePolicyReaction {
        match encoded {
            Ok(bytes) => UePolicyReaction::Reply(bytes),
            Err(e) => {
                warn!("UE policy: failed to encode {what}: {e}");
                UePolicyReaction::Ignore
            }
        }
    }

    // ========================================================================
    // Uplink NAS protection (TS 24.501 Section 4.4)
    // ========================================================================

    /// Protect an uplink plain NAS message when the security context is
    /// active (integrity protected + ciphered); pass through otherwise.
    pub fn protect_if_active(&mut self, plain: Vec<u8>) -> Vec<u8> {
        if self.sec.is_active() {
            self.protect_uplink(plain, SecurityHeaderType::IntegrityProtectedAndCiphered)
        } else {
            plain
        }
    }

    /// Apply uplink ciphering and integrity protection with the given
    /// security header type (cipher first, then MAC over SQN || payload,
    /// matching TS 24.501 Section 4.4).
    fn protect_uplink(&mut self, plain: Vec<u8>, sht: SecurityHeaderType) -> Vec<u8> {
        let count = *self.sec.uplink_count();
        let mut payload = plain;

        if sht.is_ciphered() {
            if let Some(key) = self.sec.keys().knas_enc() {
                nas_cipher(
                    self.sec.ciphering_algorithm(),
                    key,
                    &count,
                    NAS_BEARER,
                    NasDirection::Uplink,
                    &mut payload,
                );
            }
        }

        let mac = match self.sec.keys().knas_int() {
            Some(key) => compute_nas_mac(
                self.sec.integrity_algorithm(),
                key,
                &count,
                NasDirection::Uplink,
                count.sqn,
                &payload,
            ),
            None => [0u8; 4],
        };

        if self.sec.increment_uplink_count().is_err() {
            warn!("Uplink NAS COUNT overflow: security context must be re-established");
        }

        let mut out = Vec::with_capacity(7 + payload.len());
        out.push(0x7E);
        out.push(u8::from(sht));
        out.extend_from_slice(&mac);
        out.push(count.sqn);
        out.extend_from_slice(&payload);
        out
    }
}

/// Parse the first TAI from a TAI list IE value (list type 00,
/// TS 24.501 Section 9.11.3.9): PLMN (3 octets) + TAC (3 octets).
fn parse_first_tai(tai_list: &[u8]) -> Option<[u8; 6]> {
    if tai_list.len() < 7 {
        return None;
    }
    let mut tai = [0u8; 6];
    tai.copy_from_slice(&tai_list[1..7]);
    Some(tai)
}

/// Parse the Equivalent PLMNs IE value (TS 24.501 §9.11.3.45): a sequence of
/// 3-octet BCD-encoded PLMNs. A trailing partial octet group is ignored.
fn parse_equivalent_plmns(value: &[u8]) -> Vec<[u8; 3]> {
    value.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nextgsim_nas::messages::mm::{
        Abba, Ie5gMmCause, Ie5gsRegistrationResult, IeNasSecurityAlgorithms,
        IeUeSecurityCapability, RegistrationResultValue, SmsOverNasAllowed,
    };
    use nextgsim_nas::security::SecurityContextState;

    /// 3GPP TS 35.207 Test Set 1 credentials (real Milenage vectors)
    const TEST_K: [u8; 16] = [
        0x46, 0x5b, 0x5c, 0xe8, 0xb1, 0x99, 0xb4, 0x9f, 0xaa, 0x5f, 0x0a, 0x2e, 0xe2, 0x38, 0xa6,
        0xbc,
    ];
    const TEST_OP: [u8; 16] = [
        0xcd, 0xc2, 0x02, 0xd5, 0x12, 0x3e, 0x20, 0xf6, 0x2b, 0x6d, 0x67, 0x6a, 0xc7, 0x2c, 0xb3,
        0x18,
    ];
    const TEST_RAND: [u8; 16] = [
        0x23, 0x55, 0x3c, 0xbe, 0x96, 0x37, 0xa8, 0x9d, 0x21, 0x8a, 0xe6, 0x4d, 0xae, 0x47, 0xbf,
        0x35,
    ];
    /// Expected AK from TS 35.207 Test Set 1 (f5)
    const TEST_AK: [u8; 6] = [0xaa, 0x68, 0x9c, 0x64, 0x83, 0x70];
    /// Expected AK* from TS 35.207 Test Set 1 (f5*)
    const TEST_AK_STAR: [u8; 6] = [0x45, 0x1e, 0x8b, 0xec, 0xa4, 0x3b];

    fn test_identity() -> MmUeIdentity {
        MmUeIdentity {
            supi: "999700000000001".to_string(),
            suci: vec![
                0x01, 0x09, 0xF9, 0x07, 0x00, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10,
            ],
            k: TEST_K,
            opc: compute_opc(&TEST_K, &TEST_OP),
            sn_name: "5G:mnc070.mcc999.3gppnetwork.org".to_string(),
            imei_sv: DEFAULT_IMEISV.to_string(),
            ea_cap: 0xF0, // EA0-EA3
            ia_cap: 0x70, // IA1-IA3 (no IA0)
            requested_nssai: Some(vec![0x01, 0x01]),
            plmn_bcd: encode_plmn_bcd(999, 70, false),
            snpn_nid: None,
            disaster_roaming: false,
            uav_indication: None,
            redcap: false,
        }
    }

    fn new_orch() -> MmOrchestrator {
        MmOrchestrator::new(test_identity())
    }

    /// Build a valid AUTN for the test credentials with the given SQN and AMF
    fn build_autn(sqn: [u8; 6], amf: [u8; 2]) -> Vec<u8> {
        let m = Milenage::new(&TEST_K, &compute_opc(&TEST_K, &TEST_OP));
        let mac = m.f1(&TEST_RAND, &sqn, &amf);
        let mut autn = Vec::with_capacity(16);
        for i in 0..6 {
            autn.push(sqn[i] ^ TEST_AK[i]);
        }
        autn.extend_from_slice(&amf);
        autn.extend_from_slice(&mac);
        autn
    }

    fn build_auth_request_pdu(sqn: [u8; 6], amf: [u8; 2]) -> Vec<u8> {
        let req = AuthenticationRequest::for_5g_aka(
            NasKeySetIdentifier::new(nextgsim_nas::security::SecurityContextType::Native, 0),
            Abba::new(vec![0x00, 0x00]),
            TEST_RAND,
            build_autn(sqn, amf),
        );
        let mut pdu = Vec::new();
        req.encode(&mut pdu);
        pdu
    }

    fn first_sent_pdu(outs: &[MmOutput]) -> &[u8] {
        for out in outs {
            if let MmOutput::SendNasPdu(pdu) = out {
                return pdu;
            }
        }
        panic!("no SendNasPdu in outputs: {outs:?}");
    }

    /// Drive a full successful registration + authentication + SMC sequence;
    /// returns the orchestrator with an active security context.
    fn establish_security_context() -> MmOrchestrator {
        let mut orch = new_orch();
        let outs = orch.start_registration(RegistrationType::InitialRegistration);
        assert_eq!(outs.len(), 1);
        assert!(orch.timers().t3510.is_running());

        // Authentication (valid AUTN, separation bit set)
        let auth_pdu = build_auth_request_pdu([0, 0, 0, 0, 0, 1], [0x80, 0x00]);
        let outs = orch.handle_downlink(&auth_pdu);
        let resp = first_sent_pdu(&outs);
        assert_eq!(resp[2], u8::from(MmMessageType::AuthenticationResponse));
        assert!(orch.res_star().is_some());
        assert!(orch.timers().t3516.is_running());

        // Security Mode Command, integrity protected with new context
        let smc_pdu = build_protected_smc(&orch, 2, 2, None);
        let outs = orch.handle_downlink(&smc_pdu);
        let complete = first_sent_pdu(&outs);
        assert_eq!(complete[1], 0x04, "SMC Complete must use SHT 0x04");
        assert!(orch.security_context().is_active());
        orch
    }

    /// Build an SMC the way the strict core does: plain inner SMC,
    /// ciphering NOT applied (SHT 0x03), MAC computed with the new keys.
    fn build_protected_smc(
        orch: &MmOrchestrator,
        enc_alg: u8,
        int_alg: u8,
        bad_mac: Option<[u8; 4]>,
    ) -> Vec<u8> {
        let smc = SecurityModeCommand::new(
            IeNasSecurityAlgorithms::new(enc_alg, int_alg),
            NasKeySetIdentifier::new(nextgsim_nas::security::SecurityContextType::Native, 0),
            replayed_caps(orch),
        );
        let mut inner = Vec::new();
        smc.encode(&mut inner);

        // Derive the same KNASint the UE will derive
        let kamf = *orch.security_context().keys().kamf().expect("kamf");
        let knas_int = nextgsim_crypto::kdf::derive_knas_int(&kamf, int_alg);

        let count = NasCount::new(0, 0);
        let mac = bad_mac.unwrap_or_else(|| {
            compute_nas_mac(
                IntegrityAlgorithm::try_from(int_alg).unwrap(),
                &knas_int,
                &count,
                NasDirection::Downlink,
                0,
                &inner,
            )
        });

        let mut out = vec![0x7E, 0x03];
        out.extend_from_slice(&mac);
        out.push(0);
        out.extend_from_slice(&inner);
        out
    }

    fn replayed_caps(orch: &MmOrchestrator) -> IeUeSecurityCapability {
        // Decode the UE's advertised capability octets into the IE
        let raw = vec![2u8, orch.identity.ea_cap, orch.identity.ia_cap];
        IeUeSecurityCapability::decode(&mut raw.as_slice()).unwrap()
    }

    /// Protect a downlink plain message the way the core does
    /// (cipher, then MAC over SQN || ciphered payload).
    fn protect_downlink(orch: &MmOrchestrator, plain: &[u8], sqn: u8) -> Vec<u8> {
        let sec = orch.security_context();
        let count = sec.estimate_downlink_count(sqn);
        let mut payload = plain.to_vec();
        nas_cipher(
            sec.ciphering_algorithm(),
            sec.keys().knas_enc().unwrap(),
            &count,
            NAS_BEARER,
            NasDirection::Downlink,
            &mut payload,
        );
        let mac = compute_nas_mac(
            sec.integrity_algorithm(),
            sec.keys().knas_int().unwrap(),
            &count,
            NasDirection::Downlink,
            sqn,
            &payload,
        );
        let mut out = vec![0x7E, 0x02];
        out.extend_from_slice(&mac);
        out.push(sqn);
        out.extend_from_slice(&payload);
        out
    }

    fn build_registration_accept_pdu(with_guti: bool) -> Vec<u8> {
        let mut acc = RegistrationAccept::new(Ie5gsRegistrationResult::new(
            SmsOverNasAllowed::NotAllowed,
            RegistrationResultValue::ThreeGppAccess,
        ));
        // TAI list: type 00, one entry, PLMN 999/70, TAC 1
        acc.tai_list = Some(vec![0x00, 0x99, 0xF9, 0x07, 0x00, 0x00, 0x01]);
        acc.allowed_nssai = Some(vec![0x01, 0x01]);
        acc.t3512_value = Some(0x49); // unit 010 = 10 hours, value 9 (as sent by the core)
        acc.t3502_value = Some(0xAC); // unit 101 = 1 minute, value 12
        if with_guti {
            let mut guti_data = vec![0xF2]; // type GUTI
            guti_data.extend_from_slice(&[0x99, 0xF9, 0x07]); // PLMN
            guti_data.push(0x01); // AMF region
            guti_data.extend_from_slice(&[0x00, 0x40]); // AMF set/ptr
            guti_data.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]); // TMSI
            acc.guti = Some(Ie5gsMobileIdentity::new(
                MobileIdentityType::Guti,
                guti_data,
            ));
        }
        let mut pdu = Vec::new();
        acc.encode(&mut pdu);
        pdu
    }

    fn build_registration_reject_pdu(cause: MmCause, t3346: Option<u8>) -> Vec<u8> {
        let mut rej = RegistrationReject::new(Ie5gMmCause::new(cause));
        rej.t3346_value = t3346;
        let mut pdu = Vec::new();
        rej.encode(&mut pdu);
        pdu
    }

    // ========================================================================
    // Registration procedure tests
    // ========================================================================

    #[test]
    fn test_registration_request_arms_t3510_and_includes_capabilities() {
        let mut orch = new_orch();
        let outs = orch.start_registration(RegistrationType::InitialRegistration);
        let pdu = first_sent_pdu(&outs);
        assert_eq!(pdu[0], 0x7E);
        assert_eq!(pdu[1], 0x00, "initial registration must be plain");
        assert_eq!(pdu[2], u8::from(MmMessageType::RegistrationRequest));
        assert!(orch.timers().t3510.is_running());
        assert_eq!(orch.state().mm_substate(), MmSubState::RegisteredInitiated);

        // TS 24.501 §4.4.6: the initial *unprotected* request carries cleartext
        // IEs only. UE security capability is a cleartext IE (present); the
        // requested NSSAI is non-cleartext and must NOT appear here (it is
        // deferred to the Security Mode Complete NAS message container).
        let decoded = RegistrationRequest::decode(&mut &pdu[3..]).unwrap();
        assert_eq!(decoded.ue_security_capability, Some(vec![0xF0, 0x70]));
        assert_eq!(
            decoded.requested_nssai, None,
            "requested NSSAI is non-cleartext; it must not be sent in the clear"
        );
        assert_eq!(
            decoded.mobile_identity.identity_type,
            MobileIdentityType::Suci
        );

        // The FULL request (including the non-cleartext IEs) is stashed for
        // replay inside the Security Mode Complete NAS message container.
        let full = orch
            .last_registration_request
            .as_ref()
            .expect("full registration request stashed for the SMC container");
        let full_decoded = RegistrationRequest::decode(&mut &full[3..]).unwrap();
        assert_eq!(full_decoded.requested_nssai, Some(vec![0x01, 0x01]));
        assert_eq!(full_decoded.ue_security_capability, Some(vec![0xF0, 0x70]));
    }

    #[test]
    fn test_registration_accept_with_guti_sends_registration_complete() {
        let mut orch = establish_security_context();
        let plain_acc = build_registration_accept_pdu(true);
        let protected = protect_downlink(&orch, &plain_acc, 1);

        let outs = orch.handle_downlink(&protected);
        // First output: Registration Complete; second: RegistrationSucceeded
        let complete = first_sent_pdu(&outs);
        assert_eq!(complete[0], 0x7E);
        assert_eq!(complete[1], 0x02, "Registration Complete must be protected");
        assert!(outs.contains(&MmOutput::RegistrationSucceeded));

        assert!(orch.stored_guti().is_some());
        assert!(orch.tai_list().is_some());
        assert!(orch.allowed_nssai().is_some());
        assert!(!orch.timers().t3510.is_running());
        assert!(orch.timers().t3512.is_running());
        // 0x49: TS 24.008 10.5.7.4a unit 010 = 10 hours, value 9
        assert_eq!(orch.timers().t3512.interval(), 9 * 10 * 3600);
        assert_eq!(
            orch.state().mm_substate(),
            MmSubState::RegisteredNormalService
        );
        assert_eq!(orch.state().update_status(), UpdateStatus::Updated);
    }

    #[test]
    fn test_registration_accept_without_guti_no_complete() {
        let mut orch = establish_security_context();
        let plain_acc = build_registration_accept_pdu(false);
        let protected = protect_downlink(&orch, &plain_acc, 1);

        let outs = orch.handle_downlink(&protected);
        assert_eq!(outs, vec![MmOutput::RegistrationSucceeded]);
        assert!(orch.stored_guti().is_none());
    }

    /// Build a Registration Accept (no GUTI) carrying an Equivalent PLMNs IE
    /// with the given 3-octet BCD PLMNs.
    fn build_registration_accept_with_equivalent_plmns(plmns: &[[u8; 3]]) -> Vec<u8> {
        let mut acc = RegistrationAccept::new(Ie5gsRegistrationResult::new(
            SmsOverNasAllowed::NotAllowed,
            RegistrationResultValue::ThreeGppAccess,
        ));
        acc.tai_list = Some(vec![0x00, 0x99, 0xF9, 0x07, 0x00, 0x00, 0x01]);
        acc.allowed_nssai = Some(vec![0x01, 0x01]);
        let mut eq = Vec::new();
        for p in plmns {
            eq.extend_from_slice(p);
        }
        acc.equivalent_plmns = Some(eq);
        let mut pdu = Vec::new();
        acc.encode(&mut pdu);
        pdu
    }

    #[test]
    fn test_registration_accept_equivalent_plmns_parsed_and_emitted() {
        let mut orch = establish_security_context();
        // 999/71 BCD = [0x99, 0xF9, 0x17]; 310/410 (3-digit MNC) = [0x13, 0x00, 0x14]
        let plmn_a = [0x99u8, 0xF9, 0x17];
        let plmn_b = [0x13u8, 0x00, 0x14];
        let plain_acc = build_registration_accept_with_equivalent_plmns(&[plmn_a, plmn_b]);
        let protected = protect_downlink(&orch, &plain_acc, 1);

        let outs = orch.handle_downlink(&protected);

        // The orchestrator stored the decoded equivalent-PLMN list...
        assert_eq!(orch.equivalent_plmns(), &[plmn_a, plmn_b]);
        // ...and surfaced it to the caller before RegistrationSucceeded.
        assert!(outs.contains(&MmOutput::EquivalentPlmnsUpdated(vec![plmn_a, plmn_b])));
        assert!(outs.contains(&MmOutput::RegistrationSucceeded));
        let eq_idx = outs
            .iter()
            .position(|o| matches!(o, MmOutput::EquivalentPlmnsUpdated(_)));
        let ok_idx = outs
            .iter()
            .position(|o| matches!(o, MmOutput::RegistrationSucceeded));
        assert!(eq_idx < ok_idx, "equivalent PLMNs must precede success");
    }

    #[test]
    fn test_registration_accept_without_equivalent_plmns_no_emit() {
        // A Registration Accept without the IE on a fresh UE leaves the list
        // empty and produces no EquivalentPlmnsUpdated output.
        let mut orch = establish_security_context();
        let plain_acc = build_registration_accept_pdu(false);
        let protected = protect_downlink(&orch, &plain_acc, 1);
        let outs = orch.handle_downlink(&protected);
        assert!(orch.equivalent_plmns().is_empty());
        assert!(!outs
            .iter()
            .any(|o| matches!(o, MmOutput::EquivalentPlmnsUpdated(_))));
    }

    #[test]
    fn test_equivalent_plmn_from_accept_is_honored_by_selection() {
        // End-to-end of T3.7a: an equivalent PLMN signalled in Registration
        // Accept, parsed by the orchestrator, decoded into a Plmn and fed to
        // the TS 23.122 PLMN selector, is chosen ahead of the HPLMN when the
        // RPLMN is unavailable (TS 23.122 §4.4.3.1.1 step 1).
        use crate::rrc::cell_selection::{plmn_from_bcd, Plmn, PlmnSelector};

        let mut orch = establish_security_context();
        // RPLMN = 999/71 (the visited network), equivalent PLMN = 310/410.
        let rplmn_bcd = [0x99u8, 0xF9, 0x17]; // 999/71
        let equiv_bcd = [0x13u8, 0x00, 0x14]; // 310/410
        let plain_acc = build_registration_accept_with_equivalent_plmns(&[equiv_bcd]);
        let protected = protect_downlink(&orch, &plain_acc, 1);
        let outs = orch.handle_downlink(&protected);

        // Extract the equivalent-PLMN list exactly as main.rs does.
        let bcd_plmns = outs
            .iter()
            .find_map(|o| match o {
                MmOutput::EquivalentPlmnsUpdated(p) => Some(p.clone()),
                _ => None,
            })
            .expect("EquivalentPlmnsUpdated output present");

        let home = Plmn::new(999, 70, false);
        let rplmn = plmn_from_bcd(&rplmn_bcd);
        let equiv = plmn_from_bcd(&equiv_bcd);
        assert_eq!(equiv, Plmn::new(310, 410, true));

        let mut selector = PlmnSelector::new(home);
        selector.set_registered_plmn(Some(rplmn));
        selector.set_equivalent_plmns(bcd_plmns.iter().map(plmn_from_bcd).collect());

        // RPLMN not in coverage; HPLMN and the equivalent PLMN are. The
        // equivalent PLMN (of the RPLMN) outranks the HPLMN.
        assert_eq!(selector.select(&[home, equiv]), Some(equiv));
    }

    #[test]
    fn test_registration_reject_cause_3_terminal() {
        let mut orch = new_orch();
        orch.start_registration(RegistrationType::InitialRegistration);
        let pdu = build_registration_reject_pdu(MmCause::IllegalUe, None);
        let outs = orch.handle_downlink(&pdu);
        assert_eq!(outs, vec![MmOutput::RegistrationFailed(MmCause::IllegalUe)]);
        assert!(orch.is_usim_invalid());
        assert_eq!(orch.state().mm_substate(), MmSubState::Deregistered);
        assert_eq!(
            orch.state().update_status(),
            UpdateStatus::RoamingNotAllowed
        );
        assert!(!orch.timers().t3510.is_running());
        // No retry possible
        assert!(orch
            .start_registration(RegistrationType::InitialRegistration)
            .is_empty());
    }

    #[test]
    fn test_registration_reject_cause_11_forbidden_plmn() {
        let mut orch = new_orch();
        orch.start_registration(RegistrationType::InitialRegistration);
        let pdu = build_registration_reject_pdu(MmCause::PlmnNotAllowed, None);
        let outs = orch.handle_downlink(&pdu);
        assert!(outs.contains(&MmOutput::PlmnSearchNeeded));
        assert_eq!(orch.forbidden_plmns().len(), 1);
        assert_eq!(
            orch.state().mm_substate(),
            MmSubState::DeregisteredPlmnSearch
        );
    }

    // ---- TS 24.501 4.4.4.2 downlink integrity-protection gating (ue-01) ----

    #[test]
    fn test_plain_registration_accept_discarded() {
        // REGISTRATION ACCEPT is not on the 4.4.4.2 plain allow-list; received
        // unprotected it must be discarded (no Registration Complete emitted).
        let mut orch = new_orch();
        orch.start_registration(RegistrationType::InitialRegistration);
        let plain_accept = build_registration_accept_pdu(true);
        let outs = orch.handle_downlink(&plain_accept);
        assert!(
            outs.is_empty(),
            "plain RegistrationAccept must be discarded"
        );
    }

    #[test]
    fn test_plain_registration_reject_cause76_discarded() {
        // 4.4.4.2: a plain REGISTRATION REJECT with 5GMM cause #76 is discarded.
        let mut orch = new_orch();
        orch.start_registration(RegistrationType::InitialRegistration);
        let pdu = build_registration_reject_pdu(MmCause::NotAuthorizedForCag, None);
        assert_eq!(pdu[3], 76, "cause octet should be #76");
        let outs = orch.handle_downlink(&pdu);
        assert!(
            outs.is_empty(),
            "plain RegistrationReject #76 must be discarded"
        );
        assert!(!outs.contains(&MmOutput::PlmnSearchNeeded));
    }

    #[test]
    fn test_plain_registration_reject_cause11_processed() {
        // 4.4.4.2: cause #11 (PLMN not allowed) is not in the excluded set, so a
        // plain REGISTRATION REJECT carrying it is still processed.
        let mut orch = new_orch();
        orch.start_registration(RegistrationType::InitialRegistration);
        let pdu = build_registration_reject_pdu(MmCause::PlmnNotAllowed, None);
        assert_eq!(pdu[3], 11);
        let outs = orch.handle_downlink(&pdu);
        assert!(outs.contains(&MmOutput::PlmnSearchNeeded));
    }

    #[test]
    fn test_unprotected_discarded_after_security_activation() {
        // 4.4.4.2 (last paragraph): after a secure exchange is established, ANY
        // unprotected NAS message is discarded - even one otherwise on the plain
        // allow-list (e.g. AUTHENTICATION REQUEST).
        let mut orch = establish_security_context();
        let plain_auth = build_auth_request_pdu([0, 0, 0, 0, 0, 2], [0x80, 0x00]);
        assert!(orch.handle_downlink(&plain_auth).is_empty());
        let plain_accept = build_registration_accept_pdu(true);
        assert!(orch.handle_downlink(&plain_accept).is_empty());
    }

    #[test]
    fn test_plain_auth_request_still_processed() {
        // 4.4.4.2: AUTHENTICATION REQUEST stays processable before security is
        // established (it is on the plain allow-list).
        let mut orch = new_orch();
        orch.start_registration(RegistrationType::InitialRegistration);
        let auth = build_auth_request_pdu([0, 0, 0, 0, 0, 1], [0x80, 0x00]);
        let outs = orch.handle_downlink(&auth);
        let resp = first_sent_pdu(&outs);
        assert_eq!(resp[2], u8::from(MmMessageType::AuthenticationResponse));
    }

    #[test]
    fn test_duplicate_downlink_pdu_replay_rejected() {
        // TS 24.501 4.4.3.1 / 4.4.4.2 (ue-02): a replayed (duplicated) protected
        // downlink message is discarded on the second delivery (non-monotonic
        // NAS COUNT), producing no further MmOutput.
        let mut orch = establish_security_context();
        let accept = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        let first = orch.handle_downlink(&accept);
        assert!(!first.is_empty(), "first delivery must be processed");
        let second = orch.handle_downlink(&accept);
        assert!(second.is_empty(), "replayed downlink PDU must be discarded");
    }

    #[test]
    fn test_smc_nia0_rejected_outside_emergency() {
        // ue-04: an SMC selecting NIA0 (null integrity) in a normal registration
        // is rejected and does not activate the context (TS 33.501 5.5.2).
        let mut orch = new_orch();
        orch.start_registration(RegistrationType::InitialRegistration);
        let auth_pdu = build_auth_request_pdu([0, 0, 0, 0, 0, 1], [0x80, 0x00]);
        orch.handle_downlink(&auth_pdu);
        let smc = build_protected_smc(&orch, 2, 0, None); // NEA2 / NIA0
        let outs = orch.handle_downlink(&smc);
        let resp = first_sent_pdu(&outs);
        assert_eq!(resp[2], u8::from(MmMessageType::SecurityModeReject));
        assert!(!orch.security_context().is_active());
    }

    #[test]
    fn test_smc_nia0_accepted_in_emergency() {
        // ue-04: NIA0 is permitted during an emergency registration.
        let mut orch = new_orch();
        orch.start_registration(RegistrationType::EmergencyRegistration);
        let auth_pdu = build_auth_request_pdu([0, 0, 0, 0, 0, 1], [0x80, 0x00]);
        orch.handle_downlink(&auth_pdu);
        let smc = build_protected_smc(&orch, 2, 0, None); // NEA2 / NIA0
        let outs = orch.handle_downlink(&smc);
        let resp = first_sent_pdu(&outs);
        // The SMC Complete is returned protected with the new context (SHT 0x04);
        // an active context proves the NIA0 selection was accepted.
        assert_eq!(resp[1], 0x04, "SMC Complete must use SHT 0x04");
        assert!(orch.security_context().is_active());
    }

    #[test]
    fn test_registration_complete_on_drx_without_guti() {
        // ue-03: a Registration Accept that carries negotiated DRX parameters
        // but assigns no new 5G-GUTI still triggers Registration Complete
        // (TS 24.501 5.5.1.2.4); previously only a GUTI did.
        let mut orch = establish_security_context();
        let mut acc = RegistrationAccept::new(Ie5gsRegistrationResult::new(
            SmsOverNasAllowed::NotAllowed,
            RegistrationResultValue::ThreeGppAccess,
        ));
        acc.tai_list = Some(vec![0x00, 0x99, 0xF9, 0x07, 0x00, 0x00, 0x01]);
        acc.allowed_nssai = Some(vec![0x01, 0x01]);
        acc.negotiated_drx_parameters = Some(0x01);
        let mut plain = Vec::new();
        acc.encode(&mut plain);
        let protected = protect_downlink(&orch, &plain, 1);
        let outs = orch.handle_downlink(&protected);
        assert!(
            outs.iter().any(|o| matches!(o, MmOutput::SendNasPdu(_))),
            "DRX-only Accept must trigger Registration Complete"
        );
    }

    #[test]
    fn test_no_registration_complete_without_ack_ie() {
        // ue-03: an Accept with neither a new GUTI nor any ack-triggering IE
        // sends no Registration Complete.
        let mut orch = establish_security_context();
        let plain = build_registration_accept_pdu(false); // no GUTI, no DRX
        let protected = protect_downlink(&orch, &plain, 1);
        let outs = orch.handle_downlink(&protected);
        assert!(
            !outs.iter().any(|o| matches!(o, MmOutput::SendNasPdu(_))),
            "Accept without an ack-triggering IE must not send Registration Complete"
        );
    }

    #[test]
    fn test_registration_complete_on_pending_nssai_without_guti() {
        // ue-03-tail: a Registration Accept carrying a Pending NSSAI IE
        // (TS 24.501 Table 8.2.7.1.1, IEI 0x39) but no new 5G-GUTI must still
        // trigger Registration Complete (TS 24.501 5.5.1.2.4).
        let mut orch = establish_security_context();
        let mut acc = RegistrationAccept::new(Ie5gsRegistrationResult::new(
            SmsOverNasAllowed::NotAllowed,
            RegistrationResultValue::ThreeGppAccess,
        ));
        acc.tai_list = Some(vec![0x00, 0x99, 0xF9, 0x07, 0x00, 0x00, 0x01]);
        acc.allowed_nssai = Some(vec![0x01, 0x01]);
        // One S-NSSAI (length 2: length-of-SST + SST)
        acc.pending_nssai = Some(vec![0x02, 0x01, 0x01]);
        let mut plain = Vec::new();
        acc.encode(&mut plain);
        // The IE survives a round-trip (correct IEI framing, trailing IEs intact)
        let decoded = RegistrationAccept::decode(&mut &plain[3..]).unwrap();
        assert_eq!(decoded.pending_nssai, Some(vec![0x02, 0x01, 0x01]));
        let protected = protect_downlink(&orch, &plain, 1);
        let outs = orch.handle_downlink(&protected);
        assert!(
            outs.iter().any(|o| matches!(o, MmOutput::SendNasPdu(_))),
            "Pending-NSSAI Accept must trigger Registration Complete"
        );
    }

    #[test]
    fn test_registration_complete_on_network_slicing_subscription_change() {
        // ue-03-tail: a Network slicing indication IE with NSSCI = "changed"
        // (TS 24.501 9.11.3.36, IEI nibble 0x9, octet-1 bit 1) triggers
        // Registration Complete (TS 24.501 5.5.1.2.4 item a).
        let mut orch = establish_security_context();
        let mut acc = RegistrationAccept::new(Ie5gsRegistrationResult::new(
            SmsOverNasAllowed::NotAllowed,
            RegistrationResultValue::ThreeGppAccess,
        ));
        acc.tai_list = Some(vec![0x00, 0x99, 0xF9, 0x07, 0x00, 0x00, 0x01]);
        acc.allowed_nssai = Some(vec![0x01, 0x01]);
        acc.network_slicing_subscription_change = Some(true);
        let mut plain = Vec::new();
        acc.encode(&mut plain);
        // 0x91 = IEI nibble 0x9 in the high nibble + NSSCI bit set
        assert!(plain.contains(&0x91), "encoded NSSCI octet must be 0x91");
        let decoded = RegistrationAccept::decode(&mut &plain[3..]).unwrap();
        assert_eq!(decoded.network_slicing_subscription_change, Some(true));
        let protected = protect_downlink(&orch, &plain, 1);
        let outs = orch.handle_downlink(&protected);
        assert!(
            outs.iter().any(|o| matches!(o, MmOutput::SendNasPdu(_))),
            "NSSCI=changed Accept must trigger Registration Complete"
        );
    }

    #[test]
    fn test_no_registration_complete_on_slicing_indication_not_changed() {
        // ue-03-tail: a Network slicing indication IE present but with NSSCI = 0
        // ("not changed") and no other ack-triggering IE must NOT send a
        // Registration Complete.
        let mut orch = establish_security_context();
        let mut acc = RegistrationAccept::new(Ie5gsRegistrationResult::new(
            SmsOverNasAllowed::NotAllowed,
            RegistrationResultValue::ThreeGppAccess,
        ));
        acc.tai_list = Some(vec![0x00, 0x99, 0xF9, 0x07, 0x00, 0x00, 0x01]);
        acc.allowed_nssai = Some(vec![0x01, 0x01]);
        acc.network_slicing_subscription_change = Some(false);
        let mut plain = Vec::new();
        acc.encode(&mut plain);
        let decoded = RegistrationAccept::decode(&mut &plain[3..]).unwrap();
        assert_eq!(decoded.network_slicing_subscription_change, Some(false));
        let protected = protect_downlink(&orch, &plain, 1);
        let outs = orch.handle_downlink(&protected);
        assert!(
            !outs.iter().any(|o| matches!(o, MmOutput::SendNasPdu(_))),
            "NSSCI=not-changed alone must not send Registration Complete"
        );
    }

    /// Locate a contiguous byte subsequence; returns the start index if found.
    fn find_subseq(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        haystack.windows(needle.len()).position(|w| w == needle)
    }

    #[test]
    fn test_cleartext_registration_omits_disaster_ie_by_default() {
        // ue-05 baseline: with disaster roaming inactive, the unprotected
        // initial Registration Request carries no disaster-condition IE.
        let mut orch = new_orch();
        let outs = orch.start_registration(RegistrationType::InitialRegistration);
        let wire = first_sent_pdu(&outs).to_vec();
        // Not security-protected yet, so the wire is the raw cleartext request.
        assert_eq!(wire[1], 0x00, "initial request must be plain (SHT 0)");
        let decoded = RegistrationRequest::decode(&mut &wire[3..]).unwrap();
        assert!(!decoded.disaster_roaming);
        // MINT disaster-roaming IE (IEI 0xA7, TLV) must be absent.
        assert!(
            find_subseq(&wire, &[0xA7, 0x01, 0x01]).is_none(),
            "no disaster-condition IE expected by default"
        );
    }

    #[test]
    fn test_cleartext_registration_includes_disaster_ie_when_active() {
        // ue-05: when MINT disaster roaming is active the "UE determined PLMN
        // with disaster condition" IE is a permitted cleartext IE (TS 24.501
        // §4.4.6) and must appear in the unprotected initial request, which
        // must remain decodable.
        let mut id = test_identity();
        id.disaster_roaming = true;
        let mut orch = MmOrchestrator::new(id);
        let outs = orch.start_registration(RegistrationType::InitialRegistration);
        let wire = first_sent_pdu(&outs).to_vec();
        assert_eq!(wire[1], 0x00, "initial request must be plain (SHT 0)");
        let decoded = RegistrationRequest::decode(&mut &wire[3..]).unwrap();
        assert!(
            decoded.disaster_roaming,
            "decoded request must carry the IE"
        );
        let ie_pos = find_subseq(&wire, &[0xA7, 0x01, 0x01])
            .expect("disaster-condition IE must be present in the cleartext request");

        // "Byte-identical to today" guarantee: the ONLY difference from the
        // disaster-inactive cleartext request is the inserted 3-byte IE.
        let mut baseline = new_orch();
        let baseline_wire =
            first_sent_pdu(&baseline.start_registration(RegistrationType::InitialRegistration))
                .to_vec();
        let mut without_ie = wire.clone();
        without_ie.drain(ie_pos..ie_pos + 3);
        assert_eq!(
            without_ie, baseline_wire,
            "disaster IE must be the sole addition to the cleartext request"
        );
    }

    #[test]
    fn test_registration_reject_cause_12_13_15_forbidden_tais() {
        // The UE must know its TAI to populate the forbidden lists:
        // run a full attach first so the TAI list is stored
        let mut orch = establish_security_context();
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);

        // #12 Tracking area not allowed
        let pdu = build_registration_reject_pdu(MmCause::TrackingAreaNotAllowed, None);
        let protected = protect_downlink(&orch, &pdu, 2);
        orch.handle_downlink(&protected);
        assert_eq!(orch.forbidden_tais_service().len(), 1);
        assert_eq!(
            orch.state().mm_substate(),
            MmSubState::DeregisteredLimitedService
        );

        // #13 / #15 populate the roaming list
        let mut orch = establish_security_context();
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);
        let pdu = build_registration_reject_pdu(MmCause::RoamingNotAllowedInTa, None);
        let protected = protect_downlink(&orch, &pdu, 2);
        let outs = orch.handle_downlink(&protected);
        assert!(outs.contains(&MmOutput::PlmnSearchNeeded));
        assert_eq!(orch.forbidden_tais_roaming().len(), 1);
    }

    #[test]
    fn test_registration_reject_cause_22_arms_t3346() {
        let mut orch = new_orch();
        orch.start_registration(RegistrationType::InitialRegistration);
        // T3346 = unit 1 minute, value 2 -> 120s (GPRS timer 2: 0b001_00010)
        let pdu = build_registration_reject_pdu(MmCause::Congestion, Some(0x22));
        orch.handle_downlink(&pdu);
        assert!(orch.timers().t3346.is_running());
        assert_eq!(orch.timers().t3346.interval(), 120);
        assert_eq!(orch.state().update_status(), UpdateStatus::NotUpdated);
        assert_eq!(
            orch.state().mm_substate(),
            MmSubState::DeregisteredAttemptingRegistration
        );
        // While T3346 runs, new registrations are deferred
        assert!(orch
            .start_registration(RegistrationType::InitialRegistration)
            .is_empty());
    }

    #[test]
    fn test_registration_reject_cause_22_without_timer_is_abnormal() {
        let mut orch = new_orch();
        orch.start_registration(RegistrationType::InitialRegistration);
        let pdu = build_registration_reject_pdu(MmCause::Congestion, None);
        orch.handle_downlink(&pdu);
        assert!(!orch.timers().t3346.is_running());
        assert!(orch.timers().t3511.is_running());
        assert_eq!(orch.registration_attempt_counter(), 1);
    }

    #[test]
    fn test_registration_reject_cause_27_disables_n1() {
        let mut orch = new_orch();
        orch.start_registration(RegistrationType::InitialRegistration);
        let pdu = build_registration_reject_pdu(MmCause::N1ModeNotAllowed, None);
        let outs = orch.handle_downlink(&pdu);
        assert_eq!(
            outs,
            vec![MmOutput::RegistrationFailed(MmCause::N1ModeNotAllowed)]
        );
        assert!(orch
            .start_registration(RegistrationType::InitialRegistration)
            .is_empty());
    }

    #[test]
    fn test_registration_reject_cause_62_deletes_allowed_nssai() {
        let mut orch = establish_security_context();
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);
        assert!(orch.allowed_nssai().is_some());

        let pdu = build_registration_reject_pdu(MmCause::NoNetworkSlicesAvailable, None);
        let protected = protect_downlink(&orch, &pdu, 2);
        orch.handle_downlink(&protected);
        assert!(orch.allowed_nssai().is_none());
        assert_eq!(
            orch.state().mm_substate(),
            MmSubState::DeregisteredLimitedService
        );
    }

    #[test]
    fn test_registration_reject_snpn_causes() {
        for cause in [
            MmCause::TemporarilyNotAuthorizedForSnpn,
            MmCause::PermanentlyNotAuthorizedForSnpn,
            MmCause::ServingNetworkNotAuthorized,
        ] {
            let mut orch = new_orch();
            orch.start_registration(RegistrationType::InitialRegistration);
            let pdu = build_registration_reject_pdu(cause, None);
            let outs = orch.handle_downlink(&pdu);
            assert!(
                outs.contains(&MmOutput::PlmnSearchNeeded),
                "cause {cause:?} must trigger PLMN search"
            );
            assert_eq!(
                orch.state().mm_substate(),
                MmSubState::DeregisteredPlmnSearch
            );
        }
    }

    #[test]
    fn test_registration_reject_unknown_cause_abnormal_retry() {
        let mut orch = new_orch();
        orch.start_registration(RegistrationType::InitialRegistration);
        let pdu = build_registration_reject_pdu(MmCause::ProtocolErrorUnspecified, None);
        orch.handle_downlink(&pdu);
        assert!(orch.timers().t3511.is_running());
        assert_eq!(orch.registration_attempt_counter(), 1);

        // T3511 expiry retries the registration
        let outs = orch.handle_timer_expiry(TimerExpiryEvent::new(TIMER_T3511, true, 1));
        let pdu = first_sent_pdu(&outs);
        assert_eq!(pdu[2], u8::from(MmMessageType::RegistrationRequest));
        assert!(orch.timers().t3510.is_running());
    }

    #[test]
    fn test_registration_attempt_exhaustion_starts_t3502() {
        let mut orch = new_orch();
        orch.start_registration(RegistrationType::InitialRegistration);
        for i in 1..=MAX_REGISTRATION_ATTEMPTS {
            let pdu = build_registration_reject_pdu(MmCause::ProtocolErrorUnspecified, None);
            orch.handle_downlink(&pdu);
            if i < MAX_REGISTRATION_ATTEMPTS {
                assert!(orch.timers().t3511.is_running());
                // retry
                orch.handle_timer_expiry(TimerExpiryEvent::new(TIMER_T3511, true, i));
            }
        }
        assert_eq!(
            orch.registration_attempt_counter(),
            MAX_REGISTRATION_ATTEMPTS
        );
        assert!(orch.timers().t3502.is_running());
        assert!(!orch.timers().t3511.is_running());
        assert_eq!(orch.state().update_status(), UpdateStatus::NotUpdated);

        // T3502 expiry resets the counter and retries
        let outs = orch.handle_timer_expiry(TimerExpiryEvent::new(TIMER_T3502, true, 1));
        assert!(!outs.is_empty());
        assert_eq!(orch.registration_attempt_counter(), 0);
    }

    #[test]
    fn test_t3510_expiry_is_abnormal_case() {
        let mut orch = new_orch();
        orch.start_registration(RegistrationType::InitialRegistration);
        let outs = orch.handle_timer_expiry(TimerExpiryEvent::new(TIMER_T3510, true, 1));
        assert!(outs.is_empty());
        assert!(orch.timers().t3511.is_running());
        assert_eq!(orch.registration_attempt_counter(), 1);
    }

    #[test]
    fn test_t3512_expiry_triggers_periodic_registration_with_guti() {
        let mut orch = establish_security_context();
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);
        assert!(orch.timers().t3512.is_running());

        let outs = orch.handle_timer_expiry(TimerExpiryEvent::new(TIMER_T3512, true, 1));
        let pdu = first_sent_pdu(&outs);
        // Periodic registration must be security protected (active ctx)
        assert_eq!(pdu[1], 0x02);
        assert!(orch.timers().t3510.is_running());
    }

    #[test]
    fn test_t3346_expiry_retries_registration() {
        let mut orch = new_orch();
        orch.start_registration(RegistrationType::InitialRegistration);
        let pdu = build_registration_reject_pdu(MmCause::Congestion, Some(0x22));
        orch.handle_downlink(&pdu);
        assert!(orch.timers().t3346.is_running());

        let outs = orch.handle_timer_expiry(TimerExpiryEvent::new(TIMER_T3346, true, 1));
        let pdu = first_sent_pdu(&outs);
        assert_eq!(pdu[2], u8::from(MmMessageType::RegistrationRequest));
    }

    // ========================================================================
    // Authentication tests
    // ========================================================================

    #[test]
    fn test_authentication_success_with_real_vectors() {
        let mut orch = new_orch();
        orch.start_registration(RegistrationType::InitialRegistration);
        let pdu = build_auth_request_pdu([0, 0, 0, 0, 0, 1], [0x80, 0x00]);
        let outs = orch.handle_downlink(&pdu);
        let resp = first_sent_pdu(&outs);
        assert_eq!(resp[2], u8::from(MmMessageType::AuthenticationResponse));

        // Round-trip the Authentication Response and check RES* presence
        let decoded = AuthenticationResponse::decode(&mut &resp[3..]).unwrap();
        let res_star = decoded.auth_response_parameter.unwrap().value;
        assert_eq!(res_star.len(), 16);
        assert_eq!(&res_star[..], orch.res_star().unwrap());

        assert!(orch.timers().t3516.is_running());
        assert!(orch.security_context().keys().has_kamf());
        assert_eq!(
            orch.security_context().state(),
            SecurityContextState::Establishing
        );
    }

    #[test]
    fn test_authentication_mac_failure_sends_cause_20() {
        let mut orch = new_orch();
        // Corrupt the AUTN MAC
        let mut autn = build_autn([0, 0, 0, 0, 0, 1], [0x80, 0x00]);
        autn[15] ^= 0xFF;
        let req = AuthenticationRequest::for_5g_aka(
            NasKeySetIdentifier::new(nextgsim_nas::security::SecurityContextType::Native, 0),
            Abba::new(vec![0x00, 0x00]),
            TEST_RAND,
            autn,
        );
        let mut pdu = Vec::new();
        req.encode(&mut pdu);

        let outs = orch.handle_downlink(&pdu);
        let failure = first_sent_pdu(&outs);
        assert_eq!(failure[2], u8::from(MmMessageType::AuthenticationFailure));
        let decoded = AuthenticationFailure::decode(&mut &failure[3..]).unwrap();
        assert_eq!(decoded.mm_cause.value, MmCause::MacFailure);
        assert!(decoded.auth_failure_parameter.is_none());
        assert!(orch.timers().t3520.is_running());
    }

    #[test]
    fn test_authentication_wrong_amf_separation_bit_cause_26() {
        let mut orch = new_orch();
        // AMF separation bit (bit 7 of octet 0) cleared -> non-5G vector
        let pdu = build_auth_request_pdu([0, 0, 0, 0, 0, 1], [0x00, 0x00]);
        let outs = orch.handle_downlink(&pdu);
        let failure = first_sent_pdu(&outs);
        let decoded = AuthenticationFailure::decode(&mut &failure[3..]).unwrap();
        assert_eq!(
            decoded.mm_cause.value,
            MmCause::Non5gAuthenticationUnacceptable
        );
    }

    #[test]
    fn test_authentication_sqn_replay_sends_auts_cause_21() {
        let mut orch = new_orch();

        // First authentication succeeds with SQN = 5
        let pdu = build_auth_request_pdu([0, 0, 0, 0, 0, 5], [0x80, 0x00]);
        let outs = orch.handle_downlink(&pdu);
        assert_eq!(
            first_sent_pdu(&outs)[2],
            u8::from(MmMessageType::AuthenticationResponse)
        );

        // Replay with SQN = 3 (not greater than SQN_MS) -> synch failure
        let pdu = build_auth_request_pdu([0, 0, 0, 0, 0, 3], [0x80, 0x00]);
        let outs = orch.handle_downlink(&pdu);
        let failure = first_sent_pdu(&outs);
        let decoded = AuthenticationFailure::decode(&mut &failure[3..]).unwrap();
        assert_eq!(decoded.mm_cause.value, MmCause::SynchFailure);

        // Verify AUTS against the TS 35.207 f1*/f5* vectors:
        // AUTS = (SQN_MS xor AK*) || MAC-S
        let auts = decoded.auth_failure_parameter.unwrap().value;
        assert_eq!(auts.len(), 14);
        let sqn_ms = [0u8, 0, 0, 0, 0, 5];
        let m = Milenage::new(&TEST_K, &compute_opc(&TEST_K, &TEST_OP));
        let expected_mac_s = m.f1_star(&TEST_RAND, &sqn_ms, &[0x00, 0x00]);
        for i in 0..6 {
            assert_eq!(auts[i], sqn_ms[i] ^ TEST_AK_STAR[i], "AUTS SQN byte {i}");
        }
        assert_eq!(&auts[6..14], &expected_mac_s, "AUTS MAC-S");
    }

    #[test]
    fn test_authentication_sqn_too_far_ahead_is_synch_failure() {
        let mut orch = new_orch();
        // SQN jump beyond the delta window (2^28) -> out of range
        let huge_sqn = [0x10, 0x00, 0x00, 0x00, 0x00, 0x00];
        let pdu = build_auth_request_pdu(huge_sqn, [0x80, 0x00]);
        let outs = orch.handle_downlink(&pdu);
        let failure = first_sent_pdu(&outs);
        let decoded = AuthenticationFailure::decode(&mut &failure[3..]).unwrap();
        assert_eq!(decoded.mm_cause.value, MmCause::SynchFailure);
    }

    #[test]
    fn test_three_consecutive_mac_failures_trigger_plmn_search() {
        let mut orch = new_orch();
        let mut autn = build_autn([0, 0, 0, 0, 0, 1], [0x80, 0x00]);
        autn[15] ^= 0xFF;
        for i in 1..=MAX_AUTH_FAILURES {
            let req = AuthenticationRequest::for_5g_aka(
                NasKeySetIdentifier::new(nextgsim_nas::security::SecurityContextType::Native, 0),
                Abba::new(vec![0x00, 0x00]),
                TEST_RAND,
                autn.clone(),
            );
            let mut pdu = Vec::new();
            req.encode(&mut pdu);
            let outs = orch.handle_downlink(&pdu);
            if i == MAX_AUTH_FAILURES {
                assert!(outs.contains(&MmOutput::PlmnSearchNeeded));
            } else {
                assert!(!outs.contains(&MmOutput::PlmnSearchNeeded));
            }
        }
    }

    #[test]
    fn test_authentication_reject_invalidates_usim() {
        let mut orch = new_orch();
        orch.start_registration(RegistrationType::InitialRegistration);
        let pdu = build_auth_request_pdu([0, 0, 0, 0, 0, 1], [0x80, 0x00]);
        orch.handle_downlink(&pdu);

        let reject = nextgsim_nas::messages::mm::AuthenticationReject::new();
        let mut pdu = Vec::new();
        reject.encode(&mut pdu);
        let outs = orch.handle_downlink(&pdu);
        assert_eq!(outs, vec![MmOutput::AuthenticationRejected]);
        assert!(orch.is_usim_invalid());
        assert!(orch.res_star().is_none());
        assert!(!orch.timers().t3516.is_running());
        assert_eq!(orch.state().mm_substate(), MmSubState::Deregistered);
    }

    #[test]
    fn test_t3516_expiry_deletes_res_star() {
        let mut orch = new_orch();
        let pdu = build_auth_request_pdu([0, 0, 0, 0, 0, 1], [0x80, 0x00]);
        orch.handle_downlink(&pdu);
        assert!(orch.res_star().is_some());

        orch.handle_timer_expiry(TimerExpiryEvent::new(TIMER_T3516, true, 1));
        assert!(orch.res_star().is_none());
    }

    // ========================================================================
    // Security mode control tests
    // ========================================================================

    #[test]
    fn test_smc_success_activates_and_replays_registration_request() {
        let orch = establish_security_context();
        assert!(orch.security_context().is_active());
        assert_eq!(
            orch.security_context().ciphering_algorithm(),
            CipheringAlgorithm::Nea2
        );
        assert_eq!(
            orch.security_context().integrity_algorithm(),
            IntegrityAlgorithm::Nia2
        );
    }

    #[test]
    fn test_smc_complete_round_trip_contains_container_and_imeisv() {
        let mut orch = new_orch();
        orch.start_registration(RegistrationType::InitialRegistration);
        let auth = build_auth_request_pdu([0, 0, 0, 0, 0, 1], [0x80, 0x00]);
        orch.handle_downlink(&auth);

        // SMC with IMEISV request
        let mut smc = SecurityModeCommand::new(
            IeNasSecurityAlgorithms::new(2, 2),
            NasKeySetIdentifier::new(nextgsim_nas::security::SecurityContextType::Native, 0),
            replayed_caps(&orch),
        );
        smc.imeisv_request = Some(nextgsim_nas::ies::ie1::IeImeiSvRequest::new(
            nextgsim_nas::ies::ie1::ImeiSvRequest::Requested,
        ));
        let mut inner = Vec::new();
        smc.encode(&mut inner);
        let kamf = *orch.security_context().keys().kamf().unwrap();
        let knas_int = nextgsim_crypto::kdf::derive_knas_int(&kamf, 2);
        let mac = compute_nas_mac(
            IntegrityAlgorithm::Nia2,
            &knas_int,
            &NasCount::new(0, 0),
            NasDirection::Downlink,
            0,
            &inner,
        );
        let mut pdu = vec![0x7E, 0x03];
        pdu.extend_from_slice(&mac);
        pdu.push(0);
        pdu.extend_from_slice(&inner);

        let outs = orch.handle_downlink(&pdu);
        let complete_pdu = first_sent_pdu(&outs);
        assert_eq!(complete_pdu[1], 0x04);

        // Decipher the SMC Complete the way the core would (count 0, uplink)
        let mut payload = complete_pdu[7..].to_vec();
        let knas_enc = nextgsim_crypto::kdf::derive_knas_enc(&kamf, 2);
        nas_cipher(
            CipheringAlgorithm::Nea2,
            &knas_enc,
            &NasCount::new(0, 0),
            NAS_BEARER,
            NasDirection::Uplink,
            &mut payload,
        );
        assert_eq!(payload[2], u8::from(MmMessageType::SecurityModeComplete));
        let complete = SecurityModeComplete::decode(&mut &payload[3..]).unwrap();
        let container = complete.nas_message_container.expect("NAS msg container");
        assert_eq!(
            container[2],
            u8::from(MmMessageType::RegistrationRequest),
            "container must replay the Registration Request"
        );
        let imeisv = complete.imeisv.expect("IMEISV");
        assert_eq!(imeisv.identity_type, MobileIdentityType::ImeiSv);

        // Verify the MAC of the SMC Complete with the same keys
        let mut mac_in = [0u8; 4];
        mac_in.copy_from_slice(&complete_pdu[2..6]);
        verify_nas_mac(
            IntegrityAlgorithm::Nia2,
            &knas_int,
            &NasCount::new(0, 0),
            NasDirection::Uplink,
            0,
            &complete_pdu[7..],
            &mac_in,
        )
        .expect("SMC Complete MAC must verify");
    }

    #[test]
    fn test_smc_with_bad_mac_is_rejected_and_keys_not_activated() {
        let mut orch = new_orch();
        orch.start_registration(RegistrationType::InitialRegistration);
        let auth = build_auth_request_pdu([0, 0, 0, 0, 0, 1], [0x80, 0x00]);
        orch.handle_downlink(&auth);

        let pdu = build_protected_smc(&orch, 2, 2, Some([0xDE, 0xAD, 0xBE, 0xEF]));
        let outs = orch.handle_downlink(&pdu);
        let reject = first_sent_pdu(&outs);
        assert_eq!(reject[2], u8::from(MmMessageType::SecurityModeReject));
        let decoded = SecurityModeReject::decode(&mut &reject[3..]).unwrap();
        assert_eq!(
            decoded.mm_cause.value,
            MmCause::SecurityModeRejectedUnspecified
        );
        assert!(
            !orch.security_context().is_active(),
            "keys must NOT be activated on MAC failure"
        );
    }

    #[test]
    fn test_smc_bidding_down_detected() {
        let mut orch = new_orch();
        orch.start_registration(RegistrationType::InitialRegistration);
        let auth = build_auth_request_pdu([0, 0, 0, 0, 0, 1], [0x80, 0x00]);
        orch.handle_downlink(&auth);

        // Replayed capabilities differ from what the UE advertised
        let weak_caps = IeUeSecurityCapability::decode(&mut [2u8, 0x80, 0x00].as_slice()).unwrap();
        let smc = SecurityModeCommand::new(
            IeNasSecurityAlgorithms::new(0, 0),
            NasKeySetIdentifier::new(nextgsim_nas::security::SecurityContextType::Native, 0),
            weak_caps,
        );
        let mut inner = Vec::new();
        smc.encode(&mut inner);
        let mut pdu = vec![0x7E, 0x03, 0, 0, 0, 0, 0];
        pdu.extend_from_slice(&inner);

        let outs = orch.handle_downlink(&pdu);
        let reject = first_sent_pdu(&outs);
        let decoded = SecurityModeReject::decode(&mut &reject[3..]).unwrap();
        assert_eq!(
            decoded.mm_cause.value,
            MmCause::UeSecurityCapabilitiesMismatch
        );
        assert!(!orch.security_context().is_active());
    }

    #[test]
    fn test_smc_unsupported_algorithm_rejected() {
        let mut orch = new_orch();
        orch.start_registration(RegistrationType::InitialRegistration);
        let auth = build_auth_request_pdu([0, 0, 0, 0, 0, 1], [0x80, 0x00]);
        orch.handle_downlink(&auth);

        // The test identity does not advertise NIA0 for integrity, and the
        // algorithm id 5 is undefined
        let smc = SecurityModeCommand::new(
            IeNasSecurityAlgorithms::new(5, 5),
            NasKeySetIdentifier::new(nextgsim_nas::security::SecurityContextType::Native, 0),
            replayed_caps(&orch),
        );
        let mut inner = Vec::new();
        smc.encode(&mut inner);
        let mut pdu = vec![0x7E, 0x03, 0, 0, 0, 0, 0];
        pdu.extend_from_slice(&inner);

        let outs = orch.handle_downlink(&pdu);
        let reject = first_sent_pdu(&outs);
        let decoded = SecurityModeReject::decode(&mut &reject[3..]).unwrap();
        assert_eq!(
            decoded.mm_cause.value,
            MmCause::UeSecurityCapabilitiesMismatch
        );
    }

    #[test]
    fn test_smc_without_authentication_rejected() {
        let mut orch = new_orch();
        // No authentication ran: no KAMF available
        let smc = SecurityModeCommand::new(
            IeNasSecurityAlgorithms::new(2, 2),
            NasKeySetIdentifier::new(nextgsim_nas::security::SecurityContextType::Native, 0),
            replayed_caps(&orch),
        );
        let mut inner = Vec::new();
        smc.encode(&mut inner);
        let mut pdu = vec![0x7E, 0x03, 0, 0, 0, 0, 0];
        pdu.extend_from_slice(&inner);

        let outs = orch.handle_downlink(&pdu);
        let reject = first_sent_pdu(&outs);
        let decoded = SecurityModeReject::decode(&mut &reject[3..]).unwrap();
        assert_eq!(
            decoded.mm_cause.value,
            MmCause::SecurityModeRejectedUnspecified
        );
    }

    #[test]
    fn test_plain_smc_is_discarded() {
        let mut orch = new_orch();
        let auth = build_auth_request_pdu([0, 0, 0, 0, 0, 1], [0x80, 0x00]);
        orch.handle_downlink(&auth);

        let smc = SecurityModeCommand::new(
            IeNasSecurityAlgorithms::new(2, 2),
            NasKeySetIdentifier::new(nextgsim_nas::security::SecurityContextType::Native, 0),
            replayed_caps(&orch),
        );
        let mut pdu = Vec::new();
        smc.encode(&mut pdu);
        // Plain SMC (SHT 0x00) must be discarded, not answered
        let outs = orch.handle_downlink(&pdu);
        assert!(outs.is_empty());
        assert!(!orch.security_context().is_active());
    }

    #[test]
    fn test_smc_with_abba_rederives_kamf() {
        let mut orch = new_orch();
        orch.start_registration(RegistrationType::InitialRegistration);
        let auth = build_auth_request_pdu([0, 0, 0, 0, 0, 1], [0x80, 0x00]);
        orch.handle_downlink(&auth);
        let kamf_before = *orch.security_context().keys().kamf().unwrap();

        // SMC signalling a different ABBA: the MAC must be computed with
        // the re-derived KAMF for the UE to accept it
        let new_abba = vec![0x00, 0x01];
        let mut smc = SecurityModeCommand::new(
            IeNasSecurityAlgorithms::new(2, 2),
            NasKeySetIdentifier::new(nextgsim_nas::security::SecurityContextType::Native, 0),
            replayed_caps(&orch),
        );
        smc.abba = Some(new_abba.clone());
        let mut inner = Vec::new();
        smc.encode(&mut inner);

        let kseaf = *orch.security_context().keys().kseaf().unwrap();
        let kamf_new = derive_kamf(&kseaf, orch.identity.supi.as_bytes(), &new_abba);
        assert_ne!(kamf_before, kamf_new, "different ABBA must change KAMF");
        let knas_int = nextgsim_crypto::kdf::derive_knas_int(&kamf_new, 2);
        let mac = compute_nas_mac(
            IntegrityAlgorithm::Nia2,
            &knas_int,
            &NasCount::new(0, 0),
            NasDirection::Downlink,
            0,
            &inner,
        );
        let mut pdu = vec![0x7E, 0x03];
        pdu.extend_from_slice(&mac);
        pdu.push(0);
        pdu.extend_from_slice(&inner);

        let outs = orch.handle_downlink(&pdu);
        let complete = first_sent_pdu(&outs);
        assert_eq!(complete[1], 0x04);
        assert!(orch.security_context().is_active());
        assert_eq!(
            orch.security_context().keys().kamf().unwrap(),
            &kamf_new,
            "KAMF must be re-derived from the signalled ABBA"
        );
    }

    // ========================================================================
    // Downlink protection / unprotection tests
    // ========================================================================

    #[test]
    fn test_protected_downlink_with_bad_mac_is_discarded() {
        let mut orch = establish_security_context();
        let plain_acc = build_registration_accept_pdu(true);
        let mut protected = protect_downlink(&orch, &plain_acc, 1);
        protected[2] ^= 0xFF; // corrupt the MAC

        let outs = orch.handle_downlink(&protected);
        assert!(outs.is_empty(), "tampered message must be discarded");
        assert!(orch.stored_guti().is_none());
        assert!(!orch.state().is_registered());
    }

    #[test]
    fn test_protected_message_without_context_is_discarded() {
        let mut orch = new_orch();
        let plain_acc = build_registration_accept_pdu(true);
        let mut pdu = vec![0x7E, 0x02, 0, 0, 0, 0, 0];
        pdu.extend_from_slice(&plain_acc);
        let outs = orch.handle_downlink(&pdu);
        assert!(outs.is_empty());
    }

    #[test]
    fn test_unhandled_protected_message_is_deciphered_and_forwarded() {
        let mut orch = establish_security_context();
        // DL NAS Transport (0x68) is not owned by the orchestrator
        let inner = vec![0x7E, 0x00, 0x68, 0x01, 0x00, 0x01, 0xAA];
        let protected = protect_downlink(&orch, &inner, 1);
        let outs = orch.handle_downlink(&protected);
        assert_eq!(outs, vec![MmOutput::NotHandled(inner)]);
    }

    // ========================================================================
    // Service request / deregistration tests
    // ========================================================================

    #[test]
    fn test_service_request_uses_stored_tmsi() {
        let mut orch = establish_security_context();
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);
        // Simulate CM-IDLE
        orch.state_mut().switch_cm_state(CmState::Idle);

        let outs = orch.start_service_request(ServiceType::Data, Some(0x2000));
        let pdu = first_sent_pdu(&outs);
        assert_eq!(pdu[1], 0x02, "service request must be protected");
        assert!(orch.timers().t3517.is_running());
        assert_eq!(
            orch.state().mm_substate(),
            MmSubState::ServiceRequestInitiated
        );

        // Decipher and verify the 5G-S-TMSI is derived from the GUTI
        let sec = orch.security_context();
        let count = NasCount::new(0, sec.uplink_count().sqn.wrapping_sub(1));
        let mut payload = pdu[7..].to_vec();
        nas_cipher(
            sec.ciphering_algorithm(),
            sec.keys().knas_enc().unwrap(),
            &count,
            NAS_BEARER,
            NasDirection::Uplink,
            &mut payload,
        );
        let req = ServiceRequest::decode(&mut &payload[3..]).unwrap();
        // GUTI TMSI bytes were 0xDEADBEEF; the 5G-S-TMSI carries set/ptr + TMSI
        assert_eq!(&req.tmsi.data[3..7], &[0xDE, 0xAD, 0xBE, 0xEF]);
        assert_eq!(req.uplink_data_status, Some(0x2000));
    }

    #[test]
    fn test_service_request_without_guti_refused() {
        let mut orch = establish_security_context();
        // Registered but no GUTI stored
        orch.state_mut()
            .switch_mm_state(MmSubState::RegisteredNormalService);
        let outs = orch.start_service_request(ServiceType::Data, None);
        assert!(outs.is_empty());
    }

    #[test]
    fn test_service_reject_implicit_dereg_triggers_reregistration() {
        let mut orch = establish_security_context();
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);
        orch.start_service_request(ServiceType::Data, None);

        let mut rej_pdu = Vec::new();
        ServiceReject::new(MmCause::ImplicitlyDeregistered).encode(&mut rej_pdu);
        let protected = protect_downlink(&orch, &rej_pdu, 2);
        let outs = orch.handle_downlink(&protected);
        // A fresh registration must be initiated (GUTI deleted -> SUCI used;
        // security context was reset so the request is plain)
        let pdu = first_sent_pdu(&outs);
        assert_eq!(pdu[2], u8::from(MmMessageType::RegistrationRequest));
        assert!(!orch.timers().t3517.is_running());
        assert!(orch.stored_guti().is_none());
    }

    #[test]
    fn test_deregistration_t3521_retransmission_and_exhaustion() {
        let mut orch = establish_security_context();
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);

        let outs = orch.start_deregistration(false);
        assert_eq!(outs.len(), 1);
        assert!(orch.timers().t3521.is_running());

        // Retransmissions up to the limit
        for i in 1..MAX_DEREGISTRATION_ATTEMPTS {
            let outs = orch.handle_timer_expiry(TimerExpiryEvent::new(TIMER_T3521, true, i));
            assert_eq!(outs.len(), 1, "retransmission {i} expected");
        }
        // Exhaustion: local deregistration
        let outs = orch.handle_timer_expiry(TimerExpiryEvent::new(
            TIMER_T3521,
            true,
            MAX_DEREGISTRATION_ATTEMPTS,
        ));
        assert!(outs.is_empty());
        assert_eq!(
            orch.state().mm_substate(),
            MmSubState::DeregisteredNormalService
        );
    }

    #[test]
    fn test_deregistration_accept_stops_t3521() {
        let mut orch = establish_security_context();
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);
        orch.start_deregistration(false);
        assert!(orch.timers().t3521.is_running());

        let mut accept = Vec::new();
        nextgsim_nas::messages::mm::DeregistrationAcceptUeOriginating::new().encode(&mut accept);
        let protected = protect_downlink(&orch, &accept, 2);
        orch.handle_downlink(&protected);
        assert!(!orch.timers().t3521.is_running());
        assert_eq!(
            orch.state().mm_substate(),
            MmSubState::DeregisteredNormalService
        );
    }

    // TS 24.501 §5.4.3.3: the UE reuses a stored SUCI while T3519 runs and only
    // regenerates once the timer has expired.
    #[test]
    fn test_suci_reuse_while_t3519_running() {
        use crate::timer::{TimerExpiryEvent, TIMER_T3519};
        let mut orch = new_orch();

        // T3519 is not running initially, so there is nothing to reuse.
        assert_eq!(orch.timers().is_running(TIMER_T3519), Some(false));
        assert!(orch.stored_suci_while_t3519_running().is_none());

        // First SUCI: store it and arm T3519.
        let first = vec![0x01u8, 0x02, 0x03, 0x04];
        orch.store_suci_and_arm_t3519(first.clone());
        assert_eq!(orch.timers().is_running(TIMER_T3519), Some(true));

        // While T3519 runs the stored SUCI is reused byte-for-byte.
        assert_eq!(orch.stored_suci_while_t3519_running(), Some(first.clone()));
        assert_eq!(orch.stored_suci_while_t3519_running(), Some(first));

        // On T3519 expiry the stored SUCI is cleared -> caller regenerates.
        let out = orch.handle_timer_expiry(TimerExpiryEvent {
            timer_code: TIMER_T3519,
            is_mm: true,
            expiry_count: 1,
        });
        assert!(out.is_empty());
        assert!(orch.stored_suci_while_t3519_running().is_none());
    }

    // TS 24.501 §5.5.2.3.2: a network-initiated (non switch-off) DEREGISTRATION
    // REQUEST drives a DEREGISTRATION ACCEPT and moves the UE to
    // 5GMM-DEREGISTERED. Driven through the MM dispatch entry point.
    #[test]
    fn test_network_deregistration_sends_accept_and_deregisters() {
        use nextgsim_nas::ies::ie1::{
            DeRegistrationAccessType, IeDeRegistrationType, ReRegistrationRequired, SwitchOff,
        };
        let mut orch = establish_security_context();
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);
        assert!(orch.state().is_registered());

        let dereg_type = IeDeRegistrationType::new(
            DeRegistrationAccessType::ThreeGppAccess,
            ReRegistrationRequired::NotRequired,
            SwitchOff::NormalDeRegistration,
        );
        let mut req = Vec::new();
        nextgsim_nas::messages::mm::DeregistrationRequestUeTerminated::new(dereg_type)
            .encode(&mut req);
        let protected = protect_downlink(&orch, &req, 2);
        let outs = orch.handle_downlink(&protected);

        // Exactly one DEREGISTRATION ACCEPT is emitted.
        let sent: Vec<&Vec<u8>> = outs
            .iter()
            .filter_map(|o| match o {
                MmOutput::SendNasPdu(p) => Some(p),
                _ => None,
            })
            .collect();
        assert_eq!(sent.len(), 1, "expected exactly one Deregistration Accept");

        // Decipher the (protected) accept and confirm its message type.
        let sec = orch.security_context();
        let count = NasCount::new(0, sec.uplink_count().sqn.wrapping_sub(1));
        let mut payload = sent[0][7..].to_vec();
        nas_cipher(
            sec.ciphering_algorithm(),
            sec.keys().knas_enc().unwrap(),
            &count,
            NAS_BEARER,
            NasDirection::Uplink,
            &mut payload,
        );
        assert_eq!(
            payload[2],
            u8::from(MmMessageType::DeregistrationAcceptUeTerminated)
        );
        // Entered 5GMM-DEREGISTERED. A request with no 5GMM cause is the
        // §5.5.2.3.4 item b) abnormal case, which lands in PLMN-SEARCH.
        assert!(!orch.state().is_registered());
        assert_eq!(
            orch.state().mm_substate(),
            MmSubState::DeregisteredPlmnSearch
        );
    }

    // TS 24.501 §5.5.2.3.2: when re-registration is required the UE releases its
    // context and initiates a fresh registration in addition to the accept.
    #[test]
    fn test_network_deregistration_re_registration_required() {
        use nextgsim_nas::ies::ie1::{
            DeRegistrationAccessType, IeDeRegistrationType, ReRegistrationRequired, SwitchOff,
        };
        let mut orch = establish_security_context();
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);

        let dereg_type = IeDeRegistrationType::new(
            DeRegistrationAccessType::ThreeGppAccess,
            ReRegistrationRequired::Required,
            SwitchOff::NormalDeRegistration,
        );
        let mut req = Vec::new();
        nextgsim_nas::messages::mm::DeregistrationRequestUeTerminated::new(dereg_type)
            .encode(&mut req);
        let protected = protect_downlink(&orch, &req, 2);
        let outs = orch.handle_downlink(&protected);

        // A fresh registration is initiated: the context is reset first, so the
        // REGISTRATION REQUEST goes out unprotected.
        let re_registers = outs.iter().any(|o| match o {
            MmOutput::SendNasPdu(p) => {
                p.len() >= 3 && p[1] == 0x00 && p[2] == u8::from(MmMessageType::RegistrationRequest)
            }
            _ => false,
        });
        assert!(re_registers, "re-registration must be initiated");
        assert_eq!(orch.state().mm_substate(), MmSubState::RegisteredInitiated);
    }

    // ========================================================================
    // Helper tests
    // ========================================================================

    #[test]
    fn test_gprs_timer2_decoding() {
        assert_eq!(gprs_timer2_seconds(0x02), 4); // 2 * 2s
        assert_eq!(gprs_timer2_seconds(0x22), 120); // 2 * 1min
        assert_eq!(gprs_timer2_seconds(0x42), 2 * 6 * 60); // 2 decihours
        assert_eq!(gprs_timer2_seconds(0xE5), 0); // deactivated
    }

    #[test]
    fn test_encode_plmn_bcd() {
        // MCC 999 / MNC 70 (2-digit): 99 F9 07
        assert_eq!(encode_plmn_bcd(999, 70, false), [0x99, 0xF9, 0x07]);
        // MCC 001 / MNC 001 (3-digit): 00 11 00
        // (octet 2 = MNC digit 3 << 4 | MCC digit 3 = 1 << 4 | 1)
        assert_eq!(encode_plmn_bcd(1, 1, true), [0x00, 0x11, 0x00]);
    }

    #[test]
    fn test_build_imeisv_identity() {
        let id = build_imeisv_identity(DEFAULT_IMEISV);
        assert_eq!(id.identity_type, MobileIdentityType::ImeiSv);
        // 16 digits -> 1 + 8 octets (last high nibble is the 16th digit)
        assert_eq!(id.data.len(), 9);
        assert_eq!(id.data[0] & 0x07, MobileIdentityType::ImeiSv as u8);
        assert_eq!(id.data[0] & 0x08, 0, "even digit count -> odd bit clear");
        assert_eq!(id.data[0] >> 4, 4, "first IMEISV digit");
    }

    #[test]
    fn test_sqn_helpers() {
        assert_eq!(sqn_to_u64(&[0, 0, 0, 0, 0, 5]), 5);
        assert_eq!(sqn_to_u64(&[0, 0, 0, 1, 0, 0]), 0x10000);
        let orch = new_orch();
        assert!(orch.sqn_in_range(&[0, 0, 0, 0, 0, 1]));
        assert!(!orch.sqn_in_range(&[0, 0, 0, 0, 0, 0]));
    }

    #[test]
    fn test_alg_in_capability() {
        // ea_cap 0xF0 -> EA0..EA3
        assert!(alg_in_capability(0xF0, 0));
        assert!(alg_in_capability(0xF0, 3));
        assert!(!alg_in_capability(0xF0, 4));
        // ia_cap 0x70 -> IA1..IA3, no IA0
        assert!(!alg_in_capability(0x70, 0));
        assert!(alg_in_capability(0x70, 2));
    }

    // ========================================================================
    // EAP-AKA' authentication tests
    // ========================================================================

    /// Build an EAP-Request/AKA'-Challenge using the TS 35.207 Test Set 1
    /// credentials. The AUTN is computed so it passes the f1 MAC check.
    fn build_eap_aka_challenge(sqn: [u8; 6], amf: [u8; 2], net_name: &[u8]) -> Vec<u8> {
        use nextgsim_nas::{
            encode_eap_to_vec, Eap, EapAkaPrime, EapAkaSubType, EapAttributeType, EapCode,
        };

        let autn_v = build_autn(sqn, amf);
        let mut ch = EapAkaPrime::new(EapCode::Request, 42, EapAkaSubType::AkaChallenge);

        // AT_RAND: 2 reserved bytes + 16 RAND bytes (RFC 4187 §10.6)
        let mut rand_attr = vec![0u8, 0u8];
        rand_attr.extend_from_slice(&TEST_RAND);
        ch.attributes
            .put_raw_attribute(EapAttributeType::AtRand, rand_attr);

        // AT_AUTN: 2 reserved bytes + 16 AUTN bytes (RFC 4187 §10.7)
        let mut autn_attr = vec![0u8, 0u8];
        autn_attr.extend_from_slice(&autn_v);
        ch.attributes
            .put_raw_attribute(EapAttributeType::AtAutn, autn_attr);

        // AT_KDF_INPUT: 2-byte big-endian length + network name bytes (RFC 9048 §6.3.1)
        let name_len = (net_name.len() as u16).to_be_bytes();
        let mut kdf_input = vec![name_len[0], name_len[1]];
        kdf_input.extend_from_slice(net_name);
        ch.attributes
            .put_raw_attribute(EapAttributeType::AtKdfInput, kdf_input);

        // AT_KDF = 1 (RFC 9048 §3.3 key derivation function)
        ch.attributes.put_kdf(1);

        // AT_MAC: zero placeholder (UE implementation does not verify challenge AT_MAC)
        ch.attributes.put_mac(&[0u8; 16]);

        encode_eap_to_vec(&Eap::AkaPrime(ch))
    }

    #[test]
    fn test_eap_aka_prime_challenge_success() {
        use nextgsim_crypto::eap_aka_prime::run_eap_aka_prime;
        use nextgsim_crypto::kdf::hmac_sha256;
        use nextgsim_nas::{
            decode_eap, encode_eap_to_vec, Eap, EapAkaPrime, EapAkaSubType, EapAttributeType,
            EapCode,
        };

        let mut orch = new_orch();
        orch.start_registration(RegistrationType::InitialRegistration);

        let sqn: [u8; 6] = [0, 0, 0, 0, 0, 1];
        let amf: [u8; 2] = [0x80, 0x00]; // separation bit must be set
        let net_name = b"5G:mnc070.mcc999.3gppnetwork.org";
        let eap_challenge = build_eap_aka_challenge(sqn, amf, net_name);

        let req = AuthenticationRequest::for_eap_aka(
            NasKeySetIdentifier::new(nextgsim_nas::security::SecurityContextType::Native, 0),
            Abba::new(vec![0x00, 0x00]),
            eap_challenge,
        );
        let mut pdu = Vec::new();
        req.encode(&mut pdu);

        let outs = orch.handle_downlink(&pdu);
        let resp_bytes = first_sent_pdu(&outs);

        // Response must be an AuthenticationResponse
        assert_eq!(
            resp_bytes[2],
            u8::from(MmMessageType::AuthenticationResponse),
            "expected AuthenticationResponse message type"
        );

        let decoded_resp = AuthenticationResponse::decode(&mut &resp_bytes[3..]).unwrap();
        let eap_resp_data = decoded_resp
            .eap_message
            .expect("EAP message missing in response")
            .data;

        // Decode the EAP-Response/AKA'-Challenge
        let resp_eap = decode_eap(&mut &eap_resp_data[..]).unwrap();
        let Eap::AkaPrime(resp_aka) = resp_eap else {
            panic!("expected EAP AKA' response");
        };
        assert_eq!(
            resp_aka.code,
            EapCode::Response,
            "EAP code must be Response"
        );
        assert_eq!(
            resp_aka.sub_type,
            EapAkaSubType::AkaChallenge,
            "EAP subtype must be AKA-Challenge"
        );

        // Derive expected values independently from TS 35.207 Test Set 1
        let m = Milenage::new(&TEST_K, &compute_opc(&TEST_K, &TEST_OP));
        let expected_res = m.f2(&TEST_RAND);
        let ck = m.f3(&TEST_RAND);
        let ik = m.f4(&TEST_RAND);
        let ak = m.f5(&TEST_RAND);
        let mut sqn_xor_ak = [0u8; 6];
        for i in 0..6 {
            sqn_xor_ak[i] = sqn[i] ^ ak[i];
        }
        let supi = "999700000000001";
        let (keys, _kausf) = run_eap_aka_prime(&ck, &ik, net_name, &sqn_xor_ak, supi.as_bytes());

        // AT_RES: put_res stores 2-byte bit-length prefix then the RES bytes
        let (_, res_raw) = resp_aka
            .attributes
            .iter_ordered()
            .find(|(k, _)| *k == EapAttributeType::AtRes)
            .expect("AT_RES missing in EAP response");
        assert_eq!(
            &res_raw[2..],
            &expected_res[..],
            "AT_RES must equal Milenage f2(RAND)"
        );

        // AT_MAC: rebuild the response with MAC zeroed (same attribute order as impl)
        // and recompute HMAC-SHA-256, truncated to 128 bits (RFC 9048 §6.3)
        let mut verify_msg =
            EapAkaPrime::new(EapCode::Response, resp_aka.id, EapAkaSubType::AkaChallenge);
        verify_msg.attributes.put_res(&expected_res);
        verify_msg.attributes.put_mac(&[0u8; 16]);
        let zeroed_bytes = encode_eap_to_vec(&Eap::AkaPrime(verify_msg));
        let computed_mac = hmac_sha256(&keys.k_aut, &zeroed_bytes);

        // get_mac() strips the 2-byte reserved prefix and returns the 16-byte MAC
        let received_mac = resp_aka
            .attributes
            .get_mac()
            .expect("AT_MAC missing in EAP response");
        assert_eq!(
            received_mac,
            &computed_mac[..16],
            "AT_MAC must be HMAC-SHA-256(K_aut, zeroed_packet)[..16]"
        );

        // Key hierarchy must be derived after EAP-AKA'
        assert!(
            orch.security_context().keys().has_kamf(),
            "KAMF must be derived after EAP-AKA' challenge"
        );
    }

    // ------------------------------------------------------------------------
    // UE policy delivery service (UPDP) — Wave-6 E8
    // ------------------------------------------------------------------------

    /// E1 vector (f): complete MANAGE UE POLICY COMMAND content, re-cited
    /// verbatim from the hand-derived E1 golden vectors in
    /// nextgcore/src/libs/nextgcore-nas/tests/ue_policy_golden_vectors/data.rs.
    /// These bytes are the exact output of the INDEPENDENT nextgcore encoder,
    /// so decoding them here is the cross-stack strict-peer oracle: the two
    /// codecs must agree byte-for-byte. (PTI 0x80; one PLMN 001/01 sublist; one
    /// UPSC-1 instruction; one URSP part = catch-all rule → internet.)
    const E1_VEC_F_COMMAND: &[u8] = &[
        0x80, 0x01, 0x00, 0x2B, 0x00, 0x29, 0x00, 0xF1, 0x10, 0x00, 0x24, 0x00, 0x01, 0x00, 0x20,
        0x01, 0x00, 0x1D, 0xFF, 0x00, 0x01, 0x01, 0x00, 0x17, 0x00, 0x15, 0xFF, 0x00, 0x12, 0x01,
        0x01, 0x02, 0x01, 0x01, 0x04, 0x09, 0x08, 0x69, 0x6E, 0x74, 0x65, 0x72, 0x6E, 0x65, 0x74,
        0x08, 0x03,
    ];

    fn plmn_001_01() -> PlmnId {
        PlmnId {
            mcc: [0, 0, 1],
            mnc: [0, 1, 0],
            mnc_len: 2,
        }
    }

    #[test]
    fn test_ue_policy_command_stores_and_completes() {
        use nextgsim_nas::messages::mm::ue_policy::TrafficDescriptorComponent;
        let mut orch = new_orch();
        let reaction = orch.handle_ue_policy_command(E1_VEC_F_COMMAND);
        // COMPLETE echoes the command PTI (0x80) + message type 0x02.
        assert_eq!(reaction, UePolicyReaction::Reply(vec![0x80, 0x02]));
        // Exactly one section stored, at (PLMN 001/01, UPSC 1).
        assert_eq!(orch.ue_policy_section_count(), 1);
        let section = orch
            .ue_policy_section(plmn_001_01(), 1)
            .expect("section (001/01, UPSC 1) must be stored");
        let rules = section.ursp_rules().expect("URSP rules present");
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].precedence, 255);
        assert_eq!(
            rules[0].traffic_descriptor,
            vec![TrafficDescriptorComponent::MatchAll]
        );
    }

    #[test]
    fn test_ue_policy_complete_pti_is_echoed() {
        // A different PCF PTI must be echoed verbatim in the COMPLETE.
        let mut cmd = E1_VEC_F_COMMAND.to_vec();
        cmd[0] = 0xC7;
        let mut orch = new_orch();
        assert_eq!(
            orch.handle_ue_policy_command(&cmd),
            UePolicyReaction::Reply(vec![0xC7, 0x02])
        );
    }

    #[test]
    fn test_ue_policy_malformed_command_rejects_not_crash() {
        let mut orch = new_orch();
        // Truncate the command anywhere past its PTI/type/length header: the
        // declared list length no longer matches the bytes present, so the UE
        // must answer a REJECT (never panic) and must NOT store anything.
        for cut in 4..E1_VEC_F_COMMAND.len() {
            match orch.handle_ue_policy_command(&E1_VEC_F_COMMAND[..cut]) {
                UePolicyReaction::Reply(bytes) => {
                    assert_eq!(bytes[0], 0x80, "REJECT must echo the PTI");
                    assert_eq!(bytes[1], 0x03, "message type must be COMMAND REJECT");
                }
                UePolicyReaction::Ignore => {}
            }
        }
        assert_eq!(
            orch.ue_policy_section_count(),
            0,
            "a rejected command stores nothing (fail-closed)"
        );
    }

    #[test]
    fn test_ue_policy_delete_section() {
        let mut orch = new_orch();
        assert_eq!(
            orch.handle_ue_policy_command(E1_VEC_F_COMMAND),
            UePolicyReaction::Reply(vec![0x80, 0x02])
        );
        assert_eq!(orch.ue_policy_section_count(), 1);
        // A command with a single instruction carrying UPSC 1 and NO parts
        // deletes the section (TS 24.501 D.2.1.3).
        let delete_cmd = [
            0x80u8, 0x01, 0x00, 0x09, 0x00, 0x07, 0x00, 0xF1, 0x10, 0x00, 0x02, 0x00, 0x01,
        ];
        assert_eq!(
            orch.handle_ue_policy_command(&delete_cmd),
            UePolicyReaction::Reply(vec![0x80, 0x02])
        );
        assert_eq!(
            orch.ue_policy_section_count(),
            0,
            "an empty-parts instruction deletes the section"
        );
    }

    #[test]
    fn test_ue_policy_too_short_is_ignored() {
        let mut orch = new_orch();
        // < 2 octets: no PTI can be recovered → dropped (no reply, no crash).
        assert_eq!(
            orch.handle_ue_policy_command(&[0x80]),
            UePolicyReaction::Ignore
        );
        assert_eq!(orch.handle_ue_policy_command(&[]), UePolicyReaction::Ignore);
        assert_eq!(orch.ue_policy_section_count(), 0);
    }

    // ------------------------------------------------------------------------
    // Wave-6 I5 — NAS-plane KgNB derivation for AS security
    // ------------------------------------------------------------------------

    #[test]
    fn test_derive_kgnb_for_as_security() {
        use nextgsim_crypto::kdf::derive_kgnb;
        let mut orch = new_orch();
        // No KAMF yet → None (NAS security not established).
        assert!(orch.derive_kgnb_for_as_security().is_none());

        // Seed a KAMF; KgNB must equal derive_kgnb(KAMF, uplink NAS COUNT,
        // 3GPP-access) — the same value the AMF sends the gNB as SecurityKey
        // (TS 33.501 §6.9.4.1).
        let kamf = [0x22u8; 32];
        orch.sec.keys_mut().set_kamf(&kamf);
        let ul = orch.sec.uplink_count().to_u32();
        assert_eq!(
            orch.derive_kgnb_for_as_security(),
            Some(derive_kgnb(&kamf, ul, 0x01))
        );
    }
}
