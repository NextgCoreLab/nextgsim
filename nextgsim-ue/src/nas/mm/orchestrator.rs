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
    FollowOnRequest, Ie5gsRegistrationType, IeServiceType, PayloadContainerType, RegistrationType,
    ServiceType,
};
use nextgsim_nas::messages::mm::ue_policy::{
    self as ue_policy, ManageUePolicyCommand, ManageUePolicyCommandReject, ManageUePolicyComplete,
    PlmnId, UePolicyClassmark, UePolicyError, UePolicyPart, UePolicyPartType, UePolicyResult,
    UePolicySectionManagementResult, UePolicySectionManagementSubresult, UeStateIndication,
    UpsiSublist, UrspRule, CAUSE_PROTOCOL_ERROR_UNSPECIFIED, PCF_PTI_MAX, PCF_PTI_MIN,
    UPDP_MSG_MANAGE_UE_POLICY_COMMAND,
};
use nextgsim_nas::messages::mm::{
    AuthenticationFailure, AuthenticationRequest, AuthenticationResponse,
    DeregistrationRequestUeOriginating, DeregistrationRequestUeTerminated, Ie5gsMobileIdentity,
    MmCause, MobileIdentityType, RegistrationAccept, RegistrationComplete, RegistrationReject,
    RegistrationRequest, SecurityModeCommand, SecurityModeComplete, SecurityModeReject,
    ServiceAccept, ServiceReject, ServiceRequest,
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

use super::config_update::ConfigUpdateResult;
use super::deregistration::DeregistrationProcedure;
use super::persistence::{SecurityContextSnapshot, UeStateSnapshot};
use super::state::{CmState, MmStateMachine, MmSubState, UpdateStatus};
use super::uuaa::{UuaaProcedure, UuaaReaction, UuaaState};
use crate::nas::lpp::{LppEndpoint, LppUplink, ServingCellMeasurements, SidelinkRangingResult};
use crate::uav::{UavAuthorizationState, UavContext, UavIdentity};

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

/// Access type distinguisher for 3GPP access, the `P1` of the KgNB derivation
/// (TS 33.501 Annex A.9, table A.9-1). `0x02` is non-3GPP access and yields
/// K_N3IWF/K_TNGF instead, so this is not a free choice: a UE resuming over NR
/// that derived with `0x02` would hold a different KgNB from the AMF's.
const ACCESS_TYPE_3GPP: u8 = 0x01;

/// Sim-private message type for the UAV tracking report.
///
/// NOT 3GPP-conformant (Wave 4, T4.3 — honest reframe). Verified against
/// TS 24.501 v18 §8.2.10 / §9.11.3.40: **there is no UE-originated UAV
/// position/tracking NAS message** — real-time UAV tracking / Remote-ID is an
/// **application-layer** function (USS/UTM "Network Remote ID" over the user
/// plane), and network-based UAV location would use LCS (TS 23.273). This
/// simulator models the telemetry over an unassigned 5GMM message-type code for
/// a self-contained geofence demo; it has no conformant NAS equivalent and a
/// real network would not accept it. (Conformant UAS *registration* uses the
/// Service-level-AA container — IEI 0x72 — and conformant UUAA-MM uses the same
/// container over UL/DL NAS TRANSPORT with payload container type `0b1001`
/// (#58, [`super::UuaaProcedure`]); both are implemented.) Kept in sync with
/// `nextgcore-amfd`'s `gmm_build::message_type::UAV_TRACKING_REPORT`.
/// See §6 of `WAVE6-DOWNGRADED-FEATURES.md`, which lives in the `6g_docs`
/// repository's `.context/` rather than this one's — the old bare
/// `.context/...` path resolved to nothing from inside nextgsim.
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

/// Octets of a 5G-S-TMSI (TS 23.003 §2.10.1): AMF Set ID (10 bits) and AMF
/// Pointer (6 bits) packed into two octets, then the 32-bit 5G-TMSI.
pub const FIVE_G_S_TMSI_OCTETS: usize = 6;

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
    ///
    /// Produced whenever NAS security is active, in EVERY configuration (issue
    /// #201). It used to be gated on `UeConfig::as_security_enabled`, which starved
    /// the RRC plane of the key an RRC resume's `resumeMAC-I` needs
    /// (TS 38.331 §5.3.13.3) under both shipped configs. `KgNB` never crosses the
    /// air — both ends derive it from KAMF (TS 33.501 §6.2) — so this is in-process
    /// only and carrying it changes nothing on the wire. The wire gate still
    /// governs what RRC applies with it.
    AsSecurityKgnb([u8; 32]),
    /// The network signalled its view of PDU session state in a PDU session
    /// status IE on REGISTRATION ACCEPT or SERVICE ACCEPT (TS 24.501
    /// §5.5.1.3.4, §5.6.1.4.2 a): bit *n* set means PSI *n* is active in the
    /// network.
    ///
    /// The UE must locally release every session that is active on the UE and
    /// **not** set here. The diff is not computed in this orchestrator because
    /// session state lives in the SM orchestrator; the caller hands the bitmap
    /// to [`crate::nas::sm::SmOrchestrator::reconcile_pdu_session_status`],
    /// which owns both the state and the teardown.
    NetworkPduSessionStatus(u16),
    /// The UE is de-registered over 3GPP access by the network, so it releases
    /// all its PDU sessions locally (TS 24.501 §5.5.2.1, §5.5.2.3.2). Session
    /// state lives in the SM orchestrator, so the caller does the release
    /// ([`crate::nas::sm::SmOrchestrator::release_all_locally`]) and the TUN
    /// teardown (nextgsim #272).
    ReleasePduSessionsLocally,
    /// Release the NAS signalling connection locally (TS 24.501 §5.5.1.2.7 c),
    /// §5.5.1.3.7 c)): the caller has RRC drop the connection without
    /// signalling (`RrcMessage::LocalReleaseConnection`). The orchestrator is
    /// already in CM-IDLE when this is emitted, so the next NAS message it
    /// builds is an initial NAS message on a fresh RRC connection.
    ReleaseConnectionLocally,
    /// The "5GS forbidden tracking areas for roaming" or "... for regional
    /// provision of service" list changed (TS 24.501 §5.3.13, §5.5.1.2.5,
    /// §5.6.1.5). Both lists, whole, as 6-octet TAIs (PLMN BCD + TAC). The caller
    /// hands them to RRC cell selection, which must not treat a cell in one of
    /// those TAs as suitable (TS 38.304 §4.5), so the UE leaves the TA it was
    /// told it may not use (nextgsim #322).
    ForbiddenTaisChanged {
        /// "5GS forbidden tracking areas for roaming".
        roaming: Vec<[u8; 6]>,
        /// "5GS forbidden tracking areas for regional provision of service".
        service: Vec<[u8; 6]>,
    },
}

/// The Uplink data status bit for `psi` (TS 24.501 §9.11.3.44).
///
/// Bit position *n* of the little-endian IE value corresponds to PSI *n*, so
/// PSI 1 is `0x0002` and PSI 13 is `0x2000`. PSI 0 is spare and PSIs above 15
/// do not exist (TS 24.501 §9.4), so both yield an empty bitmap rather than a
/// wrapped or panicking shift.
pub fn uplink_data_status_bit(psi: u8) -> u16 {
    if psi == 0 || psi > 15 {
        return 0;
    }
    1u16 << psi
}

/// The Uplink data status bitmap advertising pending uplink data for every PSI
/// in `psis` (TS 24.501 §5.6.1.2).
///
/// Returns `None` when no in-range PSI was given, so the caller omits the IE
/// entirely instead of sending an all-zero bitmap: zero means "no session has
/// pending data", which contradicts the SERVICE REQUEST it would ride on.
pub fn uplink_data_status(psis: &[u8]) -> Option<u16> {
    let bitmap = psis
        .iter()
        .fold(0u16, |acc, &psi| acc | uplink_data_status_bit(psi));
    (bitmap != 0).then_some(bitmap)
}

/// Why the network rejected an S-NSSAI (TS 24.501 §9.11.3.46, cause value in
/// bits 4-1 of a rejected S-NSSAI's first octet).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectedSNssaiCause {
    /// 0000: S-NSSAI not available in the current PLMN or SNPN. Scoped to the
    /// PLMN, so it outlives a change of registration area.
    NotAvailableInPlmn,
    /// 0001: S-NSSAI not available in the current registration area. Dropped
    /// when the registration area changes.
    NotAvailableInRegistrationArea,
    /// 0010: not available because network slice-specific authentication and
    /// authorization failed or was revoked.
    NssaaFailedOrRevoked,
    /// A value the spec reserves. Kept (so the S-NSSAI is still excluded) but
    /// never treated as registration-area scoped, because guessing the scope of
    /// an unknown cause could re-request a slice the network refused.
    Reserved(u8),
}

impl RejectedSNssaiCause {
    /// Decodes the 4-bit cause value.
    pub fn from_bits(bits: u8) -> Self {
        match bits & 0x0F {
            0 => Self::NotAvailableInPlmn,
            1 => Self::NotAvailableInRegistrationArea,
            2 => Self::NssaaFailedOrRevoked,
            other => Self::Reserved(other),
        }
    }
}

/// One entry of a Rejected NSSAI IE (TS 24.501 §9.11.3.46).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RejectedSNssai {
    /// Why it was rejected
    pub cause: RejectedSNssaiCause,
    /// Slice/Service Type
    pub sst: u8,
    /// Slice Differentiator, when the entry carried one
    pub sd: Option<[u8; 3]>,
}

/// Parses a Rejected NSSAI IE value (TS 24.501 §9.11.3.46).
///
/// Each entry is one octet carrying the length of the rejected S-NSSAI in bits
/// 8-5 and the cause value in bits 4-1, then the SST and (when the length says
/// so) the 3-octet SD. A truncated trailing entry ends the parse rather than
/// yielding a half-read S-NSSAI: an S-NSSAI read short is a *different* slice,
/// and excluding the wrong one would silently drop a slice the network allows.
pub fn parse_rejected_nssai(value: &[u8]) -> Vec<RejectedSNssai> {
    let mut entries = Vec::new();
    let mut index = 0;
    while index < value.len() {
        let header = value[index];
        let len = usize::from(header >> 4);
        let cause = RejectedSNssaiCause::from_bits(header);
        index += 1;
        if len == 0 || index + len > value.len() {
            break;
        }
        let contents = &value[index..index + len];
        index += len;
        let sd = if len >= 4 {
            Some([contents[1], contents[2], contents[3]])
        } else {
            None
        };
        entries.push(RejectedSNssai {
            cause,
            sst: contents[0],
            sd,
        });
    }
    entries
}

/// One entry of a length-prefixed NSSAI IE (Requested / Allowed / Configured
/// NSSAI: TS 24.501 §9.11.3.37, §9.11.3.28, §9.11.3.15), kept with its raw
/// octets so a filtered list can be rebuilt without re-encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
struct NssaiEntry {
    sst: u8,
    sd: Option<[u8; 3]>,
    /// The entry exactly as it appeared, length octet included
    raw: Vec<u8>,
}

/// Parses a length-prefixed NSSAI IE value into its entries.
///
/// An S-NSSAI's contents are 1, 2, 4, 5 or 8 octets (TS 24.501 §9.11.2.8): SST
/// alone, SST plus a mapped HPLMN SST, SST plus SD, and the combinations with
/// mapped values. The SD is present exactly when the contents are 4 octets or
/// more, and any mapped values that follow are carried through untouched.
fn parse_nssai_entries(value: &[u8]) -> Vec<NssaiEntry> {
    let mut entries = Vec::new();
    let mut index = 0;
    while index < value.len() {
        let len = usize::from(value[index]);
        let start = index;
        index += 1;
        if len == 0 || index + len > value.len() {
            break;
        }
        let contents = &value[index..index + len];
        index += len;
        let sd = if len >= 4 {
            Some([contents[1], contents[2], contents[3]])
        } else {
            None
        };
        entries.push(NssaiEntry {
            sst: contents[0],
            sd,
            raw: value[start..index].to_vec(),
        });
    }
    entries
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
    /// The UE is configured for high priority access in the selected PLMN
    /// (`UeConfig::high_priority_access`; TS 24.501 §5.6.1.7 c) 4)).
    pub high_priority_access: bool,
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
            high_priority_access: config.high_priority_access,
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

/// First wait before a data-triggered SERVICE REQUEST is retried after a SERVICE
/// REJECT that named no wait (no T3346) and left the UE registered (#313).
/// Doubled per consecutive reject up to [`SERVICE_REJECT_BACKOFF_MAX`].
pub const SERVICE_REJECT_BACKOFF_BASE: std::time::Duration = std::time::Duration::from_secs(5);

/// Ceiling of the Service Reject backoff.
pub const SERVICE_REJECT_BACKOFF_MAX: std::time::Duration = std::time::Duration::from_secs(80);

/// 5GMM procedure orchestrator (UE side).
pub struct MmOrchestrator {
    identity: MmUeIdentity,
    state: MmStateMachine,
    timers: NasTimerManager,
    sec: NasSecurityContext,
    /// Whether a secure exchange of NAS messages has been established FOR THE
    /// CURRENT NAS signalling connection (TS 24.501 §4.4.4.2, nextgsim #286).
    ///
    /// Distinct from `sec.is_active()`, which survives CM-IDLE: a UE that
    /// holds a security context and opens a new connection with a Service
    /// Request has no secure exchange on that connection until a message passes
    /// its integrity check. Until then the §4.4.4.2 allow-list applies, which is
    /// what lets a new AMF incarnation that knows nothing of the UE reject it
    /// with an unprotected SERVICE REJECT #9 (§5.6.1.5). Treating the stored
    /// context as an established exchange made the UE discard that reject and
    /// sit out T3517 instead of re-registering.
    secure_exchange: bool,
    /// The user-plane resources of this UE's PDU sessions were lost with the
    /// last NAS signalling connection, and no Service Request has re-established
    /// them yet (nextgsim #288).
    ///
    /// Set when a registered UE's connection is released (RRC release or radio
    /// link failure); cleared by SERVICE ACCEPT, or by a SERVICE REJECT whose
    /// cause releases the PDU sessions (#3/#6/#7, #9/#10). Any other reject
    /// leaves it set: no user plane came back (#313). A UE that
    /// lost its cell and re-registered on a new one (mobility registration) is
    /// CM-CONNECTED again with no user plane, and only a Service Request asks
    /// the network to establish it (TS 24.501 §5.6.1.1). Without this flag the
    /// UE treated "CM-CONNECTED" as "user plane up" and sent uplink on a DRB the
    /// new gNB never set up, forever.
    up_reactivation_pending: bool,
    /// Consecutive SERVICE REJECTs that neither carried a T3346 value nor ended
    /// the UE's registration (#313). Sets [`Self::sr_backoff_until`]; cleared by
    /// SERVICE ACCEPT.
    sr_reject_count: u32,
    /// No data-triggered SERVICE REQUEST before this instant (#313). After a
    /// reject that leaves the UE registered with its user plane still down,
    /// asking again on the very next uplink packet would loop the AMF through
    /// reject after reject; see [`SERVICE_REJECT_BACKOFF_BASE`].
    sr_backoff_until: Option<std::time::Instant>,
    /// Uplink data waiting for a data SERVICE REQUEST that has not yet
    /// succeeded, as an Uplink data status bitmap (TS 24.501 §9.11.3.44; `0`
    /// when no PSI was named). The TUN forwarder holds the packets themselves.
    /// When the bar on the request lifts (T3346 expiry, the end of the backoff)
    /// the request is started from here: §5.6.1.5 #22 starts it "if still
    /// necessary, when timer T3346 expires", and a one-shot sender sends nothing
    /// more that could trigger it (#313 review). Cleared by SERVICE ACCEPT and by
    /// the rejects that release the PDU sessions.
    held_uplink_data_status: Option<u16>,
    /// Service type of the last SERVICE REQUEST sent (§5.6.1.5 #28 depends on it).
    pending_service_type: Option<ServiceType>,
    /// The Uplink data status the last registration update carried (#313 review):
    /// the sessions whose user plane the Registration Accept is to bring back.
    registration_uplink_data_status: Option<u16>,
    /// Whether this UE is configured for high priority access
    /// (`UeConfig::high_priority_access`). §5.6.1.7 c) 4) exempts such a UE from
    /// T3346, and §5.3.5 from the non-allowed-area bar.
    high_priority_access: bool,

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
    /// The paging identity last handed to the AS
    /// ([`MmOrchestrator::take_paging_identity_update`]). Kept separately from
    /// `stored_guti` so a change can be detected by difference rather than by
    /// every GUTI-mutating path remembering to signal one.
    signalled_paging_identity: Option<[u8; FIVE_G_S_TMSI_OCTETS]>,
    tai_list: Option<Vec<u8>>,
    allowed_nssai: Option<Vec<u8>>,
    /// S-NSSAIs the network has rejected (TS 24.501 §9.11.3.46), excluded from
    /// the Requested NSSAI of subsequent registrations
    rejected_nssai: Vec<RejectedSNssai>,
    /// Pending NSSAI (TS 24.501 §9.11.3.39): S-NSSAIs awaiting network
    /// slice-specific authentication and authorization. Stored as signalled;
    /// NSSAA itself is not implemented, so this is what the UE knows is pending
    /// rather than a procedure state.
    pending_nssai: Option<Vec<u8>>,
    /// Network-assigned UE radio capability ID (RACS, TS 24.501 §9.11.3.68),
    /// only kept when `racs_store_assigned_id` is configured
    racs_id: Option<String>,
    /// Whether an assigned UE radio capability ID is kept at all
    /// (`UeConfig::racs_store_assigned_id`)
    racs_store_assigned_id: bool,
    current_tai: Option<[u8; 6]>,
    /// TAI of the cell the UE is camped on (PLMN + TAC as in the TAI list IE),
    /// reported by RRC on every cell change. Unlike `current_tai` (the first
    /// entry of the TAI list), this is where the UE actually is (#277).
    serving_tai: Option<[u8; 6]>,
    /// The serving TAI when the last registration update was sent (#277 review).
    /// While its retry is scheduled (T3511/T3502), a cell change triggers a new
    /// update only if the TAI differs from this one (TS 24.501 §5.2.3.2.3 d)).
    update_attempt_tai: Option<[u8; 6]>,
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
    /// A network-initiated de-registration said "re-registration required"
    /// while the NAS signalling connection was up: the initial registration
    /// waits for that connection's release (TS 24.501 §5.5.2.3.2, issue #265).
    reregistration_on_release: bool,
    /// A SERVICE REJECT #28 asked for a registration for mobility and periodic
    /// update once the N1 NAS signalling connection is released (TS 24.501
    /// §5.6.1.5 #28, #313).
    mobility_registration_on_release: bool,

    // -- SUCI freshness (TS 24.501 §5.4.3.3 / §5.5.1.2.2) --
    /// The last SUCI sent to the network. Reused while T3519 is running instead
    /// of generating a fresh ECIES concealment on every IDENTITY/REGISTRATION
    /// request (TS 33.501 §6.12.4 frequency limit).
    stored_suci: Option<Vec<u8>>,

    // -- UE policy delivery service (UPDP, TS 24.501 Annex D) --
    /// UE policy sections stored by `(PLMN, UPSC)` (TS 24.501 D.2.1.3 / D.3).
    ue_policy_sections: HashMap<(PlmnId, u16), StoredUePolicySection>,
    /// Last UE-initiated UPDP PTI used (TS 24.501 D.1.2 range 01H-77H). `0` means
    /// none allocated yet, which is why it is not a valid PTI to send.
    updp_pti: u8,

    // -- non-volatile 5GMM parameters (TS 24.501 Annex C.1) --
    /// Where the Annex C.1 parameters are stored across restarts, when the
    /// operator configured a path. `None` disables persistence entirely.
    ///
    /// This is the one field that makes the orchestrator touch the filesystem.
    /// It is here rather than at the caller because the two moments that matter
    /// -- capturing on deregistration, and INVALIDATING when the registration
    /// context is deleted -- are both inside `delete_registration_context`,
    /// which is reached from a dozen reject paths that do not return outputs.
    /// Surfacing it as an `MmOutput` would mean threading a return value
    /// through all of them, and any one missed would leave a stale GUTI on disk
    /// that a restart would then present to the network.
    state_file: Option<std::path::PathBuf>,

    // -- LPP positioning (TS 37.355, over the NAS payload container) --
    /// The UE's LPP endpoint, which answers the LMF's capability and
    /// location-information requests (issue #46).
    lpp: LppEndpoint,
    /// The serving cell's latest measurements, as the RRC layer last reported them.
    ///
    /// Cached rather than fetched, because an LPP request arrives on the NAS task and
    /// the measurements live on the RRC task. The cost is staleness bounded by the
    /// RRC cycle (`RRC_CYCLE_INTERVAL_MS`, 2500 ms) -- a real E-CID report is a
    /// snapshot too, and one that old is still a measurement rather than a guess.
    serving_cell_measurements: ServingCellMeasurements,
    /// The sidelink ranging results the LPP endpoint reports to the LMF, pushed by
    /// the ranging task (issue #137). Empty until it has measured something, which
    /// is what a UE with no sidelink peers honestly holds.
    sidelink_ranging_results: Vec<SidelinkRangingResult>,

    // -- UAS service-level authentication (UUAA-MM, TS 23.256 §5.2.2) --
    /// The UE half of the UUAA exchange over UL/DL NAS TRANSPORT.
    uuaa: UuaaProcedure,
    /// NAS-level UAV context, present only for an aerial UE. This is what the
    /// UUAA result is *applied to*: without it the outcome would live in a
    /// private field and nothing about the aircraft would change.
    uav: Option<UavContext>,
}

impl MmOrchestrator {
    /// Create a new orchestrator for the given UE identity.
    pub fn new(identity: MmUeIdentity) -> Self {
        let high_priority_access = identity.high_priority_access;
        Self {
            high_priority_access,
            identity,
            state: MmStateMachine::new(),
            timers: NasTimerManager::new(),
            sec: NasSecurityContext::new_3gpp(),
            secure_exchange: false,
            up_reactivation_pending: false,
            sr_reject_count: 0,
            sr_backoff_until: None,
            held_uplink_data_status: None,
            pending_service_type: None,
            registration_uplink_data_status: None,
            abba: vec![0x00, 0x00],
            sqn_ms: [0u8; 6],
            res_star: None,
            auth_failure_count: 0,
            last_registration_request: None,
            pending_reg_type: RegistrationType::InitialRegistration,
            registration_attempt_counter: 0,
            stored_guti: None,
            signalled_paging_identity: None,
            tai_list: None,
            allowed_nssai: None,
            rejected_nssai: Vec::new(),
            pending_nssai: None,
            racs_id: None,
            racs_store_assigned_id: false,
            current_tai: None,
            serving_tai: None,
            update_attempt_tai: None,
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
            reregistration_on_release: false,
            mobility_registration_on_release: false,
            stored_suci: None,
            ue_policy_sections: HashMap::new(),
            updp_pti: 0,
            state_file: None,
            lpp: LppEndpoint::new(),
            sidelink_ranging_results: Vec::new(),
            // Zero until RRC reports a camped cell. A location request answered
            // before then carries no RSRP at all rather than a made-up one -- see
            // `ServingCellMeasurements`.
            serving_cell_measurements: ServingCellMeasurements::default(),
            // No payload and no UAV context until `from_config` sees an aerial
            // UE: a non-UAV UE that answered a UUAA pending indication would be
            // claiming a UAS identity it does not have.
            uuaa: UuaaProcedure::with_initiation(None, cfg!(feature = "uuaa-mm")),
            uav: None,
        }
    }

    // ========================================================================
    // Non-volatile 5GMM parameters (TS 24.501 Annex C.1)
    // ========================================================================

    /// Build an orchestrator for `config`, rehydrating the Annex C.1 parameters
    /// from `config.state_file` when one is configured and readable.
    ///
    /// With no `state_file` this is exactly [`Self::new`]: no file is read, no
    /// file will be written, and the UE registers with a SUCI as before.
    ///
    /// A state file that exists but does not parse is reported and IGNORED
    /// rather than fatal: a UE that cannot start because of a corrupt cache is
    /// worse than one that registers from scratch, which is a spec-valid
    /// procedure the network completes normally. The distinction that matters is
    /// that it is never silently *replaced* -- it stays on disk for inspection
    /// until the next successful registration overwrites it.
    pub fn from_config(identity: MmUeIdentity, config: &UeConfig) -> Self {
        let mut orch = Self::new(identity);
        orch.racs_store_assigned_id = config.racs_store_assigned_id;
        orch.configure_uav(config);
        let Some(ref path) = config.state_file else {
            return orch;
        };
        orch.state_file = Some(path.clone());
        match UeStateSnapshot::load(path) {
            Ok(Some(snapshot)) => orch.restore_stored_state(snapshot),
            Ok(None) => info!(
                "no stored 5GMM state at {} yet; registering with SUCI",
                path.display()
            ),
            Err(e) => warn!(
                "stored 5GMM state at {} not usable ({e}); registering with SUCI",
                path.display()
            ),
        }
        orch
    }

    // ========================================================================
    // UAS service-level authentication (UUAA-MM, TS 23.256 §5.2.2)
    // ========================================================================

    /// Build the NAS-level UAV context and the UUAA procedure for an aerial UE.
    ///
    /// Only an aerial UE gets either. The UUAA payload is an opaque blob from
    /// configuration because TS 23.256 puts the credential exchange in the
    /// application layer, between the UE's UAS application and the USS: this
    /// simulator has no such application, so it carries what the operator
    /// configured rather than synthesising something credential-shaped.
    fn configure_uav(&mut self, config: &UeConfig) {
        let Some(uav_config) = config.uav_config.as_ref().filter(|u| u.is_aerial_ue) else {
            return;
        };

        // The UE config models the CAA-level ID only. The serial number,
        // manufacturer and model are ASTM F3411 remote-ID fields with no
        // configuration source in this tree, so they are left empty rather than
        // invented -- an invented serial number is a different aircraft.
        let mut identity = UavIdentity::new(String::new(), String::new(), String::new());
        identity.caa_level_id = uav_config.uav_id.clone();
        self.uav = Some(UavContext::new(identity));

        let payload = uav_config.uuaa_payload();
        if payload.is_none() && uav_config.uuaa_payload_hex.is_some() {
            warn!(
                "UAV config carries a uuaa_payload_hex that is not valid hex; UUAA uplinks will \
                 not be sent"
            );
        }
        self.uuaa = UuaaProcedure::with_initiation(payload, cfg!(feature = "uuaa-mm"));
    }

    /// Handle a Service-level-AA container that arrived in DL NAS TRANSPORT
    /// (payload container type `0b1001`, TS 24.501 §9.11.3.40).
    ///
    /// The network's outcome is applied to [`UavContext::auth_state`] here and
    /// not by the caller, so every path that consumes a container updates the
    /// aircraft's authorization -- a caller that forgot would leave a revoked
    /// UAV reading as authorized.
    pub fn handle_service_level_aa_container(&mut self, container: &[u8]) -> UuaaReaction {
        let reaction = self.uuaa.handle_downlink(container);
        let authorization = self.uuaa.authorization_state();
        if let Some(uav) = self.uav.as_mut() {
            if uav.auth_state != authorization {
                info!(
                    "UAV authorization state: {:?} -> {:?} (UUAA-MM)",
                    uav.auth_state, authorization
                );
                uav.auth_state = authorization;
            }
        } else {
            // A container addressed to a UE that is not configured as an aerial
            // UE. Recorded rather than acted on: there is no UAV to authorize.
            warn!(
                "UUAA-MM: Service-level-AA container received but this UE is not configured as an \
                 aerial UE; state recorded only"
            );
        }
        reaction
    }

    /// Handle an LPP payload container that arrived in DL NAS TRANSPORT (payload
    /// container type `0b0011`, TS 24.501 §5.4.5.3) and return the UL NAS TRANSPORT
    /// that answers it.
    ///
    /// Before issue #46 this container had no consumer at all: `main.rs` matched only
    /// the UE-policy and Service-level-AA types and let everything else fall through
    /// to the SM orchestrator, which owns N1 SM information and returns nothing for
    /// anything else -- so the LMF's transaction timed out with no reply and no error.
    pub fn handle_lpp_container(&mut self, container: &[u8]) -> LppUplink {
        self.lpp.handle_nas_container(
            container,
            &self.serving_cell_measurements,
            &self.sidelink_ranging_results,
        )
    }

    /// Record the sidelink ranging results, as reported by the ranging task
    /// (TS 23.586 §5.3.3, issue #137).
    ///
    /// Held rather than fetched for the same reason the serving-cell measurements
    /// are: the LPP endpoint is synchronous, and a location request has to be
    /// answered from what has already arrived.
    pub fn set_sidelink_ranging_results(&mut self, results: Vec<SidelinkRangingResult>) {
        self.sidelink_ranging_results = results;
    }

    /// The sidelink ranging results this UE would report to the LMF.
    pub fn sidelink_ranging_results(&self) -> &[SidelinkRangingResult] {
        &self.sidelink_ranging_results
    }

    /// Record the serving cell's measurements, as reported by the RRC layer.
    ///
    /// This is what an E-CID report is *made of*, so without it a location request
    /// could only be answered with invented values.
    pub fn set_serving_cell_measurements(&mut self, measurements: ServingCellMeasurements) {
        self.serving_cell_measurements = measurements;
    }

    /// The serving cell's measurements as the LPP endpoint would report them.
    pub fn serving_cell_measurements(&self) -> ServingCellMeasurements {
        self.serving_cell_measurements
    }

    /// How many LPP replies this UE has sent, for tests and status reporting.
    pub fn lpp_replies_sent(&self) -> u32 {
        self.lpp.replies_sent()
    }

    /// The UUAA exchange's state, for tests and status reporting.
    pub fn uuaa_state(&self) -> UuaaState {
        self.uuaa.state()
    }

    /// The UAV's authorization state, when this UE is an aerial UE.
    pub fn uav_authorization_state(&self) -> Option<UavAuthorizationState> {
        self.uav.as_ref().map(|uav| uav.auth_state)
    }

    /// Install a stored snapshot's 5GMM parameters (TS 24.501 Annex C.1).
    ///
    /// The security context is restored only if it restores CLEANLY. A GUTI
    /// without a usable security context is still installed, and is still worth
    /// having: §5.5.1.2.2 wants the GUTI presented whenever the UE has a valid
    /// one, and the network then runs a fresh authentication -- which is the
    /// pre-existing behaviour, not a regression.
    pub fn restore_stored_state(&mut self, snapshot: UeStateSnapshot) {
        if let Some(guti) = snapshot.guti_ie() {
            info!(
                "restored 5G-GUTI from non-volatile state: registration will be GUTI-first \
                 (TS 24.501 §5.5.1.2.2)"
            );
            self.stored_guti = Some(guti);
        }
        if let Some(tai_list) = snapshot.tai_list.clone() {
            self.current_tai = parse_first_tai(&tai_list);
            self.tai_list = Some(tai_list);
        }
        if let Some(tai) = snapshot.last_visited_tai {
            // The last visited registered TAI outranks whatever the TAI list's
            // first entry happened to be: Annex C.1 stores it separately
            // precisely because it is where the UE actually was.
            self.current_tai = Some(tai);
        }
        self.state
            .switch_update_status(UpdateStatus::from(snapshot.update_status));
        if let Some(ref security) = snapshot.security {
            match security.restore() {
                Ok(sec) => {
                    info!(
                        "restored native 5G NAS security context (ngKSI {}) with NAS COUNT \
                         continuity (TS 24.501 §4.4.2.1.3)",
                        sec.ng_ksi()
                    );
                    self.sec = sec;
                }
                Err(e) => warn!(
                    "stored 5G NAS security context not usable ({e}); a fresh primary \
                     authentication will run"
                ),
            }
        }
    }

    /// Capture the current Annex C.1 parameters.
    pub fn state_snapshot(&self) -> UeStateSnapshot {
        UeStateSnapshot {
            guti: self.stored_guti.as_ref().map(|g| g.data.clone()),
            tai_list: self.tai_list.clone(),
            last_visited_tai: self.current_tai,
            update_status: self.state.update_status().into(),
            security: SecurityContextSnapshot::capture(&self.sec),
        }
    }

    /// Persist the Annex C.1 parameters, if a state file is configured.
    ///
    /// Call on graceful shutdown and after deregistration. A snapshot with
    /// neither a GUTI nor a security context is not written: the next start
    /// would take the SUCI branch either way.
    pub fn persist_state(&self) {
        let Some(ref path) = self.state_file else {
            return;
        };
        let snapshot = self.state_snapshot();
        if snapshot.is_empty() {
            return;
        }
        match snapshot.store(path) {
            Ok(()) => info!("5GMM state persisted to {}", path.display()),
            Err(e) => warn!("could not persist 5GMM state to {}: {e}", path.display()),
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

    /// The S-NSSAIs the network has rejected, with their cause
    /// (TS 24.501 §9.11.3.46). Excluded from the Requested NSSAI of subsequent
    /// registrations.
    pub fn rejected_nssai(&self) -> &[RejectedSNssai] {
        &self.rejected_nssai
    }

    /// The Pending NSSAI IE value last signalled by the network
    /// (TS 24.501 §9.11.3.39), if any.
    pub fn pending_nssai(&self) -> Option<&[u8]> {
        self.pending_nssai.as_deref()
    }

    /// The network-assigned UE radio capability ID (RACS), when one was
    /// assigned and `racs_store_assigned_id` is configured.
    pub fn racs_id(&self) -> Option<&str> {
        self.racs_id.as_deref()
    }

    /// Stores the Rejected NSSAI IE value the network signalled
    /// (TS 24.501 §9.11.3.46).
    ///
    /// The signalled set REPLACES the stored one for the causes it mentions and
    /// is merged with entries of other causes, rather than replacing everything:
    /// the registration-area and PLMN sets have different lifetimes (§4.6.2.2),
    /// so a Registration Accept that rejects one slice for the current
    /// registration area must not resurrect a slice the PLMN rejected earlier.
    fn store_rejected_nssai(&mut self, value: &[u8]) {
        let signalled = parse_rejected_nssai(value);
        if signalled.is_empty() {
            return;
        }
        for entry in signalled {
            if !self.rejected_nssai.contains(&entry) {
                // A slice can only be rejected for one reason at a time: drop a
                // previous entry for the same S-NSSAI so the newest cause (and
                // therefore the newest lifetime) is the one that applies.
                self.rejected_nssai
                    .retain(|stored| !(stored.sst == entry.sst && stored.sd == entry.sd));
                self.rejected_nssai.push(entry);
            }
        }
        info!(
            "rejected NSSAI now holds {} S-NSSAI(s); they are excluded from the next requested NSSAI",
            self.rejected_nssai.len()
        );
    }

    /// Drops from the rejected set every S-NSSAI the network has just allowed
    /// (TS 24.501 §4.6.2.2 a) 3): a slice in the new allowed NSSAI is no longer
    /// rejected, and leaving it in would suppress it from every later Requested
    /// NSSAI even though the network is offering it.
    ///
    /// Mapped S-NSSAIs are not modelled here, so the qualification about
    /// entries associated with mapped S-NSSAIs does not apply.
    fn drop_rejected_now_allowed(&mut self, allowed_nssai: &[u8]) {
        let allowed = parse_nssai_entries(allowed_nssai);
        if allowed.is_empty() {
            return;
        }
        let before = self.rejected_nssai.len();
        self.rejected_nssai.retain(|rejected| {
            !allowed
                .iter()
                .any(|entry| entry.sst == rejected.sst && entry.sd == rejected.sd)
        });
        if self.rejected_nssai.len() != before {
            info!(
                "dropped {} rejected S-NSSAI(s) that the new allowed NSSAI includes",
                before - self.rejected_nssai.len()
            );
        }
    }

    /// Drops the registration-area-scoped rejections when the registration area
    /// changes (TS 24.501 §4.6.2.2: those entries are associated with the
    /// tracking areas where the S-NSSAI is unavailable).
    ///
    /// PLMN-scoped and NSSAA rejections survive, because the PLMN and the
    /// authorization outcome have not changed.
    fn drop_registration_area_rejections(&mut self) {
        let before = self.rejected_nssai.len();
        self.rejected_nssai
            .retain(|entry| entry.cause != RejectedSNssaiCause::NotAvailableInRegistrationArea);
        if self.rejected_nssai.len() != before {
            info!(
                "registration area changed: {} registration-area-scoped rejection(s) dropped",
                before - self.rejected_nssai.len()
            );
        }
    }

    /// The Requested NSSAI to send in the next REGISTRATION REQUEST: the
    /// configured requested NSSAI minus every S-NSSAI in the rejected NSSAI
    /// (TS 24.501 §4.6.2.2).
    ///
    /// Returns `None` when every configured S-NSSAI has been rejected — the IE
    /// is then omitted rather than sent empty, since an empty Requested NSSAI
    /// would ask the network for a slice set it just refused.
    fn requested_nssai_for_registration(&self) -> Option<Vec<u8>> {
        let configured = self.identity.requested_nssai.as_ref()?;
        if self.rejected_nssai.is_empty() {
            return Some(configured.clone());
        }
        let kept: Vec<u8> = parse_nssai_entries(configured)
            .into_iter()
            .filter(|entry| {
                !self
                    .rejected_nssai
                    .iter()
                    .any(|rejected| rejected.sst == entry.sst && rejected.sd == entry.sd)
            })
            .flat_map(|entry| entry.raw)
            .collect();
        if kept.is_empty() {
            warn!(
                "every configured S-NSSAI is in the rejected NSSAI: omitting the requested NSSAI"
            );
            return None;
        }
        Some(kept)
    }

    /// Applies a Generic UE Configuration Update (TS 24.501 §5.4.4.2): stores
    /// the parameters the network updated and restarts T3512 when a new value
    /// came with it.
    ///
    /// Before this the decoded parameters were logged and dropped, so a
    /// network-initiated 5G-GUTI reallocation, registration-area change or
    /// allowed-NSSAI update had no effect and the next REGISTRATION REQUEST
    /// still carried the old identity.
    ///
    /// The Annex C.1 snapshot is rewritten when the 5G-GUTI or the registration
    /// area changed, because those are exactly the parameters it exists to keep
    /// (a no-op unless a `state_file` is configured).
    pub fn apply_config_update(&mut self, result: &ConfigUpdateResult) {
        let mut persist = false;

        if let Some(ref guti) = result.new_guti {
            info!(
                "Configuration Update assigned a new 5G-GUTI ({} bytes)",
                guti.data.len()
            );
            self.stored_guti = Some(guti.clone());
            persist = true;
        }

        if let Some(ref tai_list) = result.new_tai_list {
            let area_changed = self.tai_list.as_ref() != Some(tai_list);
            self.current_tai = parse_first_tai(tai_list);
            self.tai_list = Some(tai_list.clone());
            if area_changed {
                self.drop_registration_area_rejections();
            }
            persist = true;
        }

        if let Some(ref nssai) = result.new_allowed_nssai {
            self.allowed_nssai = Some(nssai.clone());
            self.drop_rejected_now_allowed(nssai);
        }

        if let Some(ref nssai) = result.new_rejected_nssai {
            self.store_rejected_nssai(nssai);
        }

        if let Some(secs) = result.new_t3512_secs {
            if secs > 0 {
                self.timers.start_with_interval(TIMER_T3512, secs, true);
                info!("T3512 (periodic registration) restarted by Configuration Update: {secs}s");
            } else {
                self.timers.stop(TIMER_T3512, true);
                info!("T3512 deactivated by Configuration Update");
            }
        }

        // RACS (TS 24.501 §5.4.4.2): the deletion indication is applied before
        // an assignment in the same command, so a command that deletes the old
        // IDs and assigns a new one ends with the new one.
        if let Some(deletion) = result.racs_deletion {
            if deletion.deletes_network_assigned() {
                if self.racs_id.take().is_some() {
                    info!(
                        "Configuration Update deleted the network-assigned UE radio capability ID"
                    );
                }
            } else {
                debug!("UE radio capability ID deletion indication: {deletion:?}");
            }
        }
        if let Some(ref racs_id) = result.new_racs_id {
            if self.racs_store_assigned_id {
                info!("Configuration Update assigned UE radio capability ID {racs_id}");
                self.racs_id = Some(racs_id.clone());
            } else {
                debug!(
                    "UE radio capability ID {racs_id} assigned but not kept \
                     (racs_store_assigned_id is off)"
                );
            }
        }

        if persist {
            self.persist_state();
        }
    }

    /// The 5G-S-TMSI the access stratum must match PCCH paging records against,
    /// derived from the stored 5G-GUTI (TS 24.501 §9.11.3.4, TS 23.003
    /// §2.10.1): AMF Set ID (10 bits) + AMF Pointer (6 bits) in two octets,
    /// then the 32-bit 5G-TMSI.
    ///
    /// `None` when the UE holds no 5G-GUTI, in which case it has no paging
    /// identity at all and no paging record can be for it.
    pub fn paging_identity(&self) -> Option<[u8; FIVE_G_S_TMSI_OCTETS]> {
        let guti = self.stored_guti.as_ref()?;
        // GUTI layout: type(1) + PLMN(3) + AMF region(1) + AMF set/ptr(2) + TMSI(4)
        guti.data.get(5..11)?.try_into().ok()
    }

    /// Reports the paging identity to hand to the access stratum when it has
    /// CHANGED since the last call, otherwise `None`.
    ///
    /// The outer `Option` is "did it change"; the inner one is the identity,
    /// where `None` means *delete* — a UE whose GUTI was removed by
    /// deregistration must stop answering paging, and leaving the old identity
    /// installed in the AS would have it start a service request for an
    /// identity it no longer owns.
    ///
    /// Detection is by difference against what was last signalled rather than
    /// by each GUTI-mutating path emitting an output, so Registration Accept, a
    /// restored non-volatile snapshot, network deregistration and a local
    /// context delete are all covered without a per-site hook to forget.
    pub fn take_paging_identity_update(&mut self) -> Option<Option<[u8; FIVE_G_S_TMSI_OCTETS]>> {
        let current = self.paging_identity();
        if current == self.signalled_paging_identity {
            return None;
        }
        self.signalled_paging_identity = current;
        Some(current)
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

    /// Record the TAI of the cell the UE is now camped on (#277).
    pub fn set_serving_tai(&mut self, tai: [u8; 6]) {
        self.serving_tai = Some(tai);
    }

    /// Whether a cell change (RRC's `ActiveCellChanged`, the serving TAI already
    /// recorded) starts a registration for mobility and periodic registration
    /// update (TS 24.501 §5.5.1.3.2 a), §5.2.3.2.3).
    ///
    /// - Not registered: no (the initial registration has its own trigger).
    /// - 5GMM-REGISTERED.ATTEMPTING-REGISTRATION-UPDATE with T3346 running: no.
    ///   §5.2.3.2.3 d) starts the update on a TAI change only "if timer T3346 is
    ///   not running", and its expiry retries the update.
    /// - A failed update's retry scheduled (T3511 or T3502): only if the serving
    ///   TAI differs from the one the failed update was sent in. A re-camp in
    ///   the same TA must not bypass the retry timer (#275); a move to another TA
    ///   must not wait it out (§5.2.3.2.3 d), up to 12 minutes of T3502).
    /// - Otherwise: yes, as before.
    pub fn registration_update_on_cell_change(&self) -> bool {
        if !self.state.is_registered() {
            return false;
        }
        if self.state.mm_substate() == MmSubState::RegisteredAttemptingRegistrationUpdate
            && self.timers.t3346.is_running()
        {
            return false;
        }
        if self.timers.t3511.is_running() || self.timers.t3502.is_running() {
            return match (self.serving_tai, self.update_attempt_tai) {
                (Some(now), Some(then)) => now != then,
                // The TA of the failed attempt is unknown, or the serving one is:
                // nothing shows the TAI changed.
                _ => false,
            };
        }
        true
    }

    /// Whether the serving cell's TAI is in the stored TAI list. False when
    /// either is unknown: nothing then shows the UE is inside its registration
    /// area (#277).
    fn serving_tai_in_tai_list(&self) -> bool {
        match (&self.serving_tai, &self.tai_list) {
            (Some(tai), Some(list)) => tai_list_contains(list, tai),
            _ => false,
        }
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

    /// Whether the serving cell's TAI is in either forbidden-TA list, so the UE
    /// may not register or ask for service there (TS 24.501 §5.3.13). False when
    /// the serving TAI is unknown.
    pub fn serving_tai_forbidden(&self) -> bool {
        self.serving_tai.is_some_and(|tai| {
            self.forbidden_tais_roaming.contains(&tai) || self.forbidden_tais_service.contains(&tai)
        })
    }

    /// The TAI a reject's "current TAI" means: the serving cell's, as RRC last
    /// reported it, else the first TAI of the stored list.
    fn current_tai_for_reject(&self) -> Option<[u8; 6]> {
        self.serving_tai.or(self.current_tai)
    }

    /// Add `tai` to a forbidden-TA list unless already there. True if it was added.
    fn add_forbidden_tai(list: &mut Vec<[u8; 6]>, tai: [u8; 6]) -> bool {
        if list.contains(&tai) {
            return false;
        }
        list.push(tai);
        true
    }

    /// The current forbidden-TA lists, for RRC cell selection.
    fn forbidden_tais_output(&self) -> MmOutput {
        MmOutput::ForbiddenTaisChanged {
            roaming: self.forbidden_tais_roaming.clone(),
            service: self.forbidden_tais_service.clone(),
        }
    }

    /// Remove `tai` from the stored TAI list if present (TS 24.501 §5.6.1.5 #13,
    /// #15). The list is rebuilt as "list of TAIs belonging to different PLMNs"
    /// partial lists (§9.11.3.9 type 10); a list left empty is deleted.
    fn remove_tai_from_tai_list(&mut self, tai: &[u8; 6]) {
        let Some(list) = self.tai_list.as_ref() else {
            return;
        };
        if !tai_list_contains(list, tai) {
            return;
        }
        let kept: Vec<[u8; 6]> = tai_list_entries(list)
            .into_iter()
            .filter(|t| t != tai)
            .collect();
        self.tai_list = encode_tai_list(&kept);
        info!(
            "TAI {tai:02x?} removed from the stored TAI list ({} left)",
            kept.len()
        );
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
        // The end of the local Service Reject backoff (#313) is noticed here,
        // once: the deadline is cleared (the reject count, which sets the next
        // backoff, is not) and held uplink asks again.
        if self
            .sr_backoff_until
            .is_some_and(|until| std::time::Instant::now() >= until)
        {
            self.sr_backoff_until = None;
            outs.extend(self.retry_held_uplink_service_request("Service Reject backoff over"));
        }
        outs
    }

    /// Start the data SERVICE REQUEST for uplink still held after a rejected one,
    /// once nothing bars it (#313 review). Nothing happens when no uplink is
    /// held, the user plane is up, or a request is already in flight.
    fn retry_held_uplink_service_request(&mut self, why: &str) -> Vec<MmOutput> {
        let Some(status) = self.held_uplink_data_status else {
            return Vec::new();
        };
        let user_plane_down = self.state.is_idle() || self.up_reactivation_pending;
        if !self.state.is_registered() || !user_plane_down || self.timers.t3517.is_running() {
            return Vec::new();
        }
        if self.service_request_barred(ServiceType::Data).is_some() {
            return Vec::new();
        }
        info!("{why}: starting the Service Request for held uplink data");
        self.start_service_request(ServiceType::Data, (status != 0).then_some(status))
    }

    /// Handle a single timer expiry event.
    pub fn handle_timer_expiry(&mut self, ev: TimerExpiryEvent) -> Vec<MmOutput> {
        match ev.timer_code {
            TIMER_T3510 => {
                // TS 24.501 5.5.1.2.7 case c: T3510 expiry -> abort the
                // procedure, abnormal case handling
                warn!("T3510 expired: registration procedure timed out");
                let mut outs = Vec::new();
                // TS 24.501 §5.5.1.2.7 c), §5.5.1.3.7 c): abort the procedure
                // and release the NAS signalling connection locally (unless the
                // request is for emergency services). Keeping it sent the T3511
                // retry as an uplink NAS transport on a connection the AMF never
                // answered, so the gNB, holding no AMF UE NGAP ID, dropped every
                // retry (nextgsim #275).
                if self.pending_reg_type != RegistrationType::EmergencyRegistration {
                    info!("T3510 expired: releasing the NAS signalling connection locally");
                    self.state.switch_cm_state(CmState::Idle);
                    self.secure_exchange = false;
                    outs.push(MmOutput::ReleaseConnectionLocally);
                }
                outs.extend(self.abnormal_registration_failure());
                outs
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
                let update_waiting = self.state.mm_substate()
                    == MmSubState::RegisteredAttemptingRegistrationUpdate
                    && self.pending_is_registration_update();
                if (!self.state.is_registered() || update_waiting) && !self.usim_invalid {
                    info!("T3346 expired: congestion backoff over, retrying registration");
                    self.registration_attempt_counter = 0;
                    self.build_and_send_registration(self.pending_reg_type)
                } else {
                    // §5.6.1.5 #22, §5.6.1.7 c): "The service request procedure is
                    // started, if still necessary, when timer T3346 expires".
                    self.retry_held_uplink_service_request("T3346 expired")
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
            // A failed mobility/periodic update is retried from 5GMM-REGISTERED
            // (TS 24.501 §5.5.1.3.7, nextgsim #277).
            && (!self.state.is_registered() || self.pending_is_registration_update())
            && !self.timers.t3346.is_running()
    }

    /// Whether the pending registration is a mobility or periodic registration
    /// update (§5.5.1.3) rather than an initial or emergency registration.
    fn pending_is_registration_update(&self) -> bool {
        matches!(
            self.pending_reg_type,
            RegistrationType::MobilityRegistrationUpdating
                | RegistrationType::PeriodicRegistrationUpdating
        )
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
            if self.state.is_registered() && self.pending_is_registration_update() {
                // A deferred mobility or periodic update leaves the UE
                // registered: it is still known to the network, just not at its
                // current location (TS 24.501 §5.5.1.3.7, #277). T3346 expiry
                // runs the update.
                self.state.switch_update_status(UpdateStatus::NotUpdated);
                self.state
                    .switch_mm_state(MmSubState::RegisteredAttemptingRegistrationUpdate);
            } else {
                self.state
                    .switch_mm_state(MmSubState::DeregisteredAttemptingRegistration);
            }
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
        if matches!(
            reg_type,
            RegistrationType::MobilityRegistrationUpdating
                | RegistrationType::PeriodicRegistrationUpdating
        ) {
            self.update_attempt_tai = self.serving_tai;
        }
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

        // TS 24.501 §5.5.1.3.2: a registration update sent with uplink user data
        // pending names those PDU sessions in the Uplink data status IE, with
        // the follow-on request pending, so the network brings their user plane
        // back with the Registration Accept (#313 review: after SERVICE REJECT
        // #28 the held uplink otherwise waited for another packet).
        //
        // Two of §5.5.1.3.2's limits apply (#313 re-review):
        // - "For all cases except case b ... the UE may include the Uplink data
        //   status IE": case b is the periodic update on T3512 expiry in
        //   5GMM-IDLE, so a periodic update never carries it;
        // - "If the UE is in a non-allowed area or is not in an allowed area ...
        //   the UE shall not include the Uplink data status IE except for
        //   emergency services or for high priority access": the update after
        //   #28 (5GMM-REGISTERED.NON-ALLOWED-SERVICE) carries it only for a UE
        //   configured for high priority access. Its held uplink waits for an
        //   allowed area, or for a Service Request after the Accept.
        let non_allowed = self.state.mm_substate() == MmSubState::RegisteredNonAllowedService
            && !self.high_priority_access;
        let uplink_data_status = match reg_type {
            RegistrationType::MobilityRegistrationUpdating
                if !non_allowed && (self.state.is_idle() || self.up_reactivation_pending) =>
            {
                self.held_uplink_data_status.filter(|status| *status != 0)
            }
            _ => None,
        };
        self.registration_uplink_data_status = uplink_data_status;
        let follow_on = if uplink_data_status.is_some() {
            FollowOnRequest::Pending
        } else {
            FollowOnRequest::NoPending
        };
        let mut req = RegistrationRequest::new(
            Ie5gsRegistrationType::new(follow_on, reg_type),
            ng_ksi,
            mobile_identity.clone(),
        );
        req.uplink_data_status = uplink_data_status;
        // Mandatory-for-security IEs: UE security capability (replayed by
        // the network in the Security Mode Command) and requested NSSAI
        req.ue_security_capability = Some(vec![self.identity.ea_cap, self.identity.ia_cap]);
        // TS 24.501 §4.6.2.2: an S-NSSAI in the rejected NSSAI must not be
        // requested again, so the configured set is filtered rather than sent
        // verbatim.
        req.requested_nssai = self.requested_nssai_for_registration();
        req.last_visited_tai = self.current_tai;
        // TS 24.501 §5.5.1.2.2: report the stored UE policy sections so the PCF
        // knows what this UE already holds. Before this the sections existed in
        // `ue_policy_sections` and were never signalled back, so a PCF that
        // optimises delivery on the reported UPSI list either re-pushed
        // everything or withheld it.
        if let Some(indication) = self.build_ue_state_indication() {
            req.payload_container_type = Some(PayloadContainerType::UePolicyContainer);
            req.payload_container = Some(indication);
        }
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
        // RACS (Rel-16, TS 23.501 §5.4.4.1a / TS 24.501 §8.2.6): present the
        // network-assigned UE radio capability ID so the AMF can resolve the
        // capability set from the UCMF instead of asking for the full one.
        //
        // NON-CLEARTEXT (§4.4.6): this is set on `req` only, which is the full
        // request replayed inside the SECURITY MODE COMPLETE NAS message
        // container. The unprotected `ct` request built below does not carry it.
        // The ID identifies the subscriber's device capabilities, so sending it in
        // the clear would leak exactly what §4.4.6 exists to protect -- the trap
        // #48 hit with the UE STATE INDICATION.
        if let Some(racs_id) = self.racs_id() {
            req.ue_radio_capability_id = Some(racs_id.to_string());
            info!("RACS registration: presenting UE radio capability ID {racs_id}");
        }
        // The 5GMM capability's RACS bit (TS 24.501 §9.11.3.1 octet 4 bit 8) is
        // what entitles the AMF to send a UE radio capability ID or its deletion
        // indication at all (§5.5.1.2.4); without it the network-side half of RACS
        // can never reach this UE. Every other capability bit is 0: this UE
        // supports none of them. Non-cleartext (§4.4.6), so `req` only.
        if self.racs_store_assigned_id {
            req.five_gmm_capability = Some(vec![
                0x00,
                nextgsim_nas::messages::mm::FIVE_GMM_CAPABILITY_OCTET4_RACS,
            ]);
        }

        let mut plain = Vec::new();
        req.encode(&mut plain);

        // The FULL request (all IEs) is replayed in the Security Mode Complete
        // NAS message container once security is up (TS 24.501 §4.4.6).
        self.last_registration_request = Some(plain.clone());
        self.pending_reg_type = reg_type;
        // TS 24.501 §5.6.1.7: a registration procedure for mobility or periodic
        // update triggered while a service request is in progress aborts it and
        // stops T3517 (#316). No other registration can start while one runs: a
        // service request is only started from 5GMM-REGISTERED.
        self.abort_service_request("registration procedure started");
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
        //
        // TS 24.501 §4.4.6 b) / §4.4.5: a request sent from 5GMM-IDLE WITH a
        // valid context (mobility or periodic registration update) is also an
        // initial NAS message. It is integrity protected but NOT ciphered, carries
        // the cleartext IEs only (with the current ngKSI, so the AMF can find the
        // context the MAC is checked under), and the full request travels ciphered
        // in the NAS message container. Ciphering the whole message instead hides
        // the 5G-GUTI the AMF needs to locate that context (nextgsim #258). From
        // 5GMM-CONNECTED the request rides an existing NAS signalling connection
        // and is ciphered as usual.
        let cleartext_request = |ksi: NasKeySetIdentifier| {
            let mut ct = RegistrationRequest::new(
                // The registration type IE is cleartext; its follow-on bit is the
                // full request's.
                Ie5gsRegistrationType::new(follow_on, reg_type),
                ksi,
                mobile_identity.clone(),
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
            ct
        };
        let initial_protected = self.sec.is_active() && self.state.is_idle();

        info!(
            "Sending Registration Request ({:?}), len={}, cleartext_only={}, initial_protected={}, T3510 started",
            reg_type,
            plain.len(),
            !self.sec.is_active(),
            initial_protected
        );

        let pdu = if !self.sec.is_active() {
            let mut ct_bytes = Vec::new();
            cleartext_request(NasKeySetIdentifier::no_key()).encode(&mut ct_bytes);
            self.protect_if_active(ct_bytes)
        } else if initial_protected {
            let mut outer = cleartext_request(ng_ksi);
            let mut cleartext = Vec::new();
            outer.encode(&mut cleartext);
            self.protect_initial_nas(plain, cleartext, |container| {
                outer.nas_message_container = Some(container);
                let mut out = Vec::new();
                outer.encode(&mut out);
                out
            })
        } else {
            self.protect_if_active(plain)
        };
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
        // A retry scheduled by an earlier failed attempt is moot once one is
        // accepted; left running, its expiry would re-run a registration update
        // from 5GMM-REGISTERED (#277).
        self.timers.stop(TIMER_T3511, true);
        self.timers.stop(TIMER_T3502, true);
        // §5.5.1.3.4: for an update that carried the Uplink data status IE the
        // AMF re-establishes the user plane and "include[s] PDU session
        // reactivation result IE in the REGISTRATION ACCEPT message to indicate
        // the user-plane resources re-establishment result". A bit of 0 for a
        // requested session means its user plane "is successful" (§9.11.3.42);
        // 1, that it is not. An Accept WITHOUT the IE says nothing was
        // re-established: the requested sessions stay held, and the Service
        // Request below asks for them. Treating the absence as success left the
        // uplink held for good with nothing to ask for it (seen against amfd
        // after a gNB restart, #337 run).
        if let Some(requested) = self.registration_uplink_data_status.take() {
            let failed = match acc.pdu_session_reactivation_result {
                Some(result) => result & requested,
                None => {
                    warn!(
                        "Registration Accept carries no PDU session reactivation result for \
                         the requested {requested:#06x}: the user plane is not re-established"
                    );
                    requested
                }
            };
            if failed == 0 {
                info!("Registration Accept re-established the user plane for held uplink");
                self.up_reactivation_pending = false;
                self.held_uplink_data_status = None;
            } else {
                warn!(
                    "Registration Accept: user-plane reactivation failed for {failed:#06x}; \
                     a Service Request will ask again"
                );
                self.held_uplink_data_status = Some(failed);
            }
        }
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
            let area_changed = self.tai_list.as_ref() != Some(tai_list);
            self.current_tai = parse_first_tai(tai_list);
            self.tai_list = Some(tai_list.clone());
            if area_changed {
                self.drop_registration_area_rejections();
            }
        } else {
            warn!("Registration Accept without TAI list (network not strict)");
        }
        if let Some(ref nssai) = acc.allowed_nssai {
            self.allowed_nssai = Some(nssai.clone());
            // TS 24.501 §4.6.2.2 a) 3): a slice the network now allows is no
            // longer rejected.
            self.drop_rejected_now_allowed(nssai);
        } else {
            warn!("Registration Accept without Allowed NSSAI (network not strict)");
        }
        // Slice-selection state the UE must keep (TS 24.501 §5.5.1.2.4): the
        // rejected NSSAI decides what a later Requested NSSAI may ask for, and
        // the pending NSSAI records what is awaiting NSSAA.
        if let Some(ref nssai) = acc.rejected_nssai {
            self.store_rejected_nssai(nssai);
        }
        if let Some(ref nssai) = acc.pending_nssai {
            info!(
                "Registration Accept signalled a pending NSSAI ({} octets): NSSAA outstanding",
                nssai.len()
            );
            self.pending_nssai = Some(nssai.clone());
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
        // §5.5.1.2.4: a UE radio capability ID IE or its deletion indication IE
        // is acknowledged with REGISTRATION COMPLETE too (the AMF runs T3550).
        let racs_signalled =
            acc.ue_radio_capability_id.is_some() || acc.ue_radio_capability_id_deletion.is_some();
        let needs_ack = acc.guti.is_some()
            || acc.pending_nssai.is_some()
            || acc.network_slicing_subscription_change == Some(true)
            || acc.negotiated_drx_parameters.is_some()
            || racs_signalled;
        // §5.5.1.2.4 a)/b): delete on the deletion indication (then re-register
        // after this registration completes), store a newly assigned ID.
        let mut racs_reregister = false;
        if acc.ue_radio_capability_id_deletion == Some(1) {
            if self.racs_id.take().is_some() {
                info!("Registration Accept deleted the network-assigned UE radio capability ID");
            }
            racs_reregister = true;
        }
        if let Some(ref id) = acc.ue_radio_capability_id {
            if self.racs_store_assigned_id {
                info!("Registration Accept assigned UE radio capability ID {id}");
                self.racs_id = Some(id.clone());
            }
        }
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
        if racs_reregister && self.racs_store_assigned_id {
            // §5.5.1.2.4 a): "after the completion of the ongoing registration
            // procedure, initiate a registration procedure for mobility and periodic
            // registration update ... over the existing N1 NAS signalling connection".
            info!("RACS ID deleted: starting a mobility registration update (§5.5.1.2.4 a))");
            outs.extend(self.start_registration(RegistrationType::MobilityRegistrationUpdating));
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
        // TS 24.501 §5.5.1.3.4: on a Registration Accept carrying a PDU session
        // status IE the UE locally releases the sessions that are active on the
        // UE but indicated as inactive by the network. Emitted BEFORE
        // RegistrationSucceeded so the caller reconciles away the stale sessions
        // before that arm establishes the configured default ones -- otherwise
        // the establish gate (`active_sessions().is_empty()`) still sees the
        // sessions the network just told us are gone, and the UE would neither
        // release them nor re-establish anything.
        if let Some(status) = acc.pdu_session_status {
            outs.push(MmOutput::NetworkPduSessionStatus(status));
        }
        outs.push(MmOutput::RegistrationSucceeded);
        // The Accept stopped T3346 (above). §5.6.1.5 #22 / §5.6.1.7 c): "The
        // service request procedure is started, if still necessary, when timer
        // T3346 expires or is stopped" -- uplink still held asks now (#313
        // re-review).
        outs.extend(self.retry_held_uplink_service_request("After the Registration Accept"));
        outs
    }

    /// PDU session `psi` was released (#313 re-review): it has no uplink data
    /// pending any more, so its bit leaves the held Uplink data status
    /// (§5.6.1.2 names only sessions with pending data).
    pub fn on_pdu_session_released(&mut self, psi: u8) {
        let bit = uplink_data_status_bit(psi);
        if let Some(status) = self.held_uplink_data_status.as_mut() {
            *status &= !bit;
            if *status == 0 {
                self.held_uplink_data_status = None;
            }
        }
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
        // A REGISTRATION REJECT may carry the rejected NSSAI too
        // (TS 24.501 §8.2.7). Storing it matters most here: the registration
        // that failed is the one whose requested NSSAI should not be repeated.
        if let Some(ref nssai) = rej.rejected_nssai {
            self.store_rejected_nssai(nssai);
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
                    Self::add_forbidden_tai(&mut self.forbidden_tais_service, tai);
                }
                self.registration_attempt_counter = 0;
                self.state
                    .switch_mm_state(MmSubState::DeregisteredLimitedService);
                vec![
                    self.forbidden_tais_output(),
                    MmOutput::RegistrationFailed(cause),
                ]
            }
            MmCause::RoamingNotAllowedInTa => {
                self.state
                    .switch_update_status(UpdateStatus::RoamingNotAllowed);
                self.delete_registration_context();
                if let Some(tai) = self.current_tai {
                    Self::add_forbidden_tai(&mut self.forbidden_tais_roaming, tai);
                }
                self.registration_attempt_counter = 0;
                self.state
                    .switch_mm_state(MmSubState::DeregisteredPlmnSearch);
                vec![
                    self.forbidden_tais_output(),
                    MmOutput::PlmnSearchNeeded,
                    MmOutput::RegistrationFailed(cause),
                ]
            }
            MmCause::NoSuitableCellsInTa => {
                self.state
                    .switch_update_status(UpdateStatus::RoamingNotAllowed);
                if let Some(tai) = self.current_tai {
                    Self::add_forbidden_tai(&mut self.forbidden_tais_roaming, tai);
                }
                self.registration_attempt_counter = 0;
                self.state
                    .switch_mm_state(MmSubState::DeregisteredLimitedService);
                vec![
                    self.forbidden_tais_output(),
                    MmOutput::RegistrationFailed(cause),
                ]
            }
            MmCause::Congestion => {
                // TS 24.501 5.5.1.2.5 #22: start T3346 with the signalled
                // value; without a usable value treat as abnormal case.
                // §5.5.1.3.5 #22: for a mobility or periodic update the UE stays
                // registered, in 5GMM-REGISTERED.ATTEMPTING-REGISTRATION-UPDATE
                // (#277).
                let secs = rej.t3346_value.map(gprs_timer2_seconds).unwrap_or(0);
                if secs > 0 {
                    self.state.switch_update_status(UpdateStatus::NotUpdated);
                    self.registration_attempt_counter = 0;
                    self.timers
                        .start_with_timer2(TIMER_T3346, GprsTimer2::new(secs), true);
                    self.state
                        .switch_mm_state(if self.pending_is_registration_update() {
                            MmSubState::RegisteredAttemptingRegistrationUpdate
                        } else {
                            MmSubState::DeregisteredAttemptingRegistration
                        });
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
            // TS 24.501 §5.5.1.3.5: a MOBILITY or PERIODIC registration rejected
            // with #9 / #10 (a restarted AMF that has no context for this UE)
            // deletes the 5G-GUTI and starts an initial registration. It used to
            // fall through to the abnormal case, which retried the same GUTI five
            // times and then waited out T3502.
            MmCause::UeIdentityCannotBeDerived | MmCause::ImplicitlyDeregistered
                if self.pending_reg_type != RegistrationType::InitialRegistration
                    && self.pending_reg_type != RegistrationType::EmergencyRegistration =>
            {
                self.state.switch_update_status(UpdateStatus::NotUpdated);
                self.delete_registration_context();
                self.registration_attempt_counter = 0;
                self.state
                    .switch_mm_state(MmSubState::DeregisteredNormalService);
                self.reregister_after_identity_loss()
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

    /// The network lost this UE's identity (#9 / #10): the UE is now
    /// 5GMM-DEREGISTERED, which holds no PDU sessions, so they are released
    /// locally BEFORE the new initial registration (TS 24.501 §5.6.1.5,
    /// §5.5.1.3.5). The default sessions then come back after the Registration
    /// Accept, the #272 path.
    ///
    /// Keeping them was the devtest1 failure after an AMF restart: the new AMF
    /// had no SM context for PSI 1, traffic only flowed on the untouched old
    /// tunnels, and the first Service Request that tried to re-activate PSI 1
    /// (after a gNB restart) found nothing to activate.
    fn reregister_after_identity_loss(&mut self) -> Vec<MmOutput> {
        let mut outs = vec![MmOutput::ReleasePduSessionsLocally];
        outs.extend(self.build_and_send_registration(RegistrationType::InitialRegistration));
        outs
    }

    /// Abnormal registration failure (TS 24.501 Section 5.5.1.2.7):
    /// increment the attempt counter; below the limit retry via T3511,
    /// at the limit fall back to T3502.
    ///
    /// A mobility or periodic registration update follows §5.5.1.3.7 instead
    /// (nextgsim #277): the UE stays registered.
    fn abnormal_registration_failure(&mut self) -> Vec<MmOutput> {
        if self.pending_is_registration_update() {
            return self.abnormal_registration_update_failure();
        }
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

    /// Abnormal failure of a mobility or periodic registration update
    /// (TS 24.501 §5.5.1.3.7). Below the attempt limit, T3511 runs. The UE
    /// re-enters 5GMM-REGISTERED.NORMAL-SERVICE when its update status is still
    /// 5U1 UPDATED and the current TAI is in its TAI list. Otherwise it goes to
    /// 5U2 NOT UPDATED and 5GMM-REGISTERED.ATTEMPTING-REGISTRATION-UPDATE. At
    /// the limit: T3502, 5U2, ATTEMPTING-REGISTRATION-UPDATE.
    ///
    /// "The TAI is in the list" is the serving cell's TAI, which RRC reports on
    /// every cell change, against the stored TAI list.
    fn abnormal_registration_update_failure(&mut self) -> Vec<MmOutput> {
        if self.registration_attempt_counter < MAX_REGISTRATION_ATTEMPTS {
            self.registration_attempt_counter += 1;
        }
        if self.registration_attempt_counter < MAX_REGISTRATION_ATTEMPTS {
            if self.serving_tai_in_tai_list() && self.state.update_status() == UpdateStatus::Updated
            {
                self.state
                    .switch_mm_state(MmSubState::RegisteredNormalService);
            } else {
                self.state.switch_update_status(UpdateStatus::NotUpdated);
                self.state
                    .switch_mm_state(MmSubState::RegisteredAttemptingRegistrationUpdate);
            }
            info!(
                "Registration update attempt {} of {} failed; {}, starting T3511",
                self.registration_attempt_counter,
                MAX_REGISTRATION_ATTEMPTS,
                self.state.mm_substate()
            );
            self.timers.start(TIMER_T3511, true);
        } else {
            warn!(
                "Registration attempt counter reached {}; starting T3502",
                MAX_REGISTRATION_ATTEMPTS
            );
            self.state.switch_update_status(UpdateStatus::NotUpdated);
            self.state
                .switch_mm_state(MmSubState::RegisteredAttemptingRegistrationUpdate);
            // A network T3502 of zero starts a zero-length timer, which expires
            // on the next tick: "the actions defined for the expiry of the timer
            // T3502", as §5.5.1.3.7 asks.
            if let Some(timer3) = self.t3502_value {
                self.timers.start_with_timer3(TIMER_T3502, timer3, true);
            } else {
                self.timers.start(TIMER_T3502, true);
            }
            // §5.5.1.3.7: at 5 attempts the list of equivalent PLMNs goes too
            // (single access: never registered over both accesses).
            if !self.equivalent_plmns.is_empty() {
                self.equivalent_plmns.clear();
                return vec![MmOutput::EquivalentPlmnsUpdated(Vec::new())];
            }
        }
        Vec::new()
    }

    /// Delete the stored registration context (5G-GUTI, TAI list, ngKSI)
    fn delete_registration_context(&mut self) {
        self.stored_guti = None;
        self.tai_list = None;
        // De-registered: no session has uplink waiting for a service request.
        self.held_uplink_data_status = None;
        self.registration_uplink_data_status = None;
        self.sec.reset();
        // TS 24.501 Annex C.1: the stored copy goes with the in-memory one.
        // Leaving the file behind would mean a restart presents a 5G-GUTI and a
        // security context the network has just told this UE to forget -- which
        // is worse than not persisting at all, because the UE would look
        // registered to itself and be rejected on every attempt.
        if let Some(ref path) = self.state_file {
            if let Err(e) = UeStateSnapshot::remove(path) {
                warn!(
                    "registration context deleted but stored state at {} survives ({e}): a \
                     restart may present a stale 5G-GUTI",
                    path.display()
                );
            }
        }
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
        self.secure_exchange = true;
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
        // The uplink NAS COUNT this Security Mode Complete is SENT under, captured
        // BEFORE `protect_uplink` consumes and advances it. That value — not the one
        // left behind afterwards — is the KgNB freshness parameter (issue #203; see
        // `derive_kgnb_for_as_security`).
        let smc_complete_ul_count = self.sec.uplink_count().to_u32();
        let pdu = self.protect_uplink(
            plain,
            SecurityHeaderType::IntegrityProtectedAndCipheredWithNewSecurityContext,
        );
        let mut outs = vec![MmOutput::SendNasPdu(pdu)];
        // Wave-6 I5: once NAS security is active, hand the derived KgNB to the
        // RRC plane so it can derive K_RRCint/K_RRCenc when the AS
        // SecurityModeCommand arrives (TS 33.501 §6.9.4.1).
        //
        // UNGATED since issue #201. This used to be behind `as_security_enabled`,
        // which made it the third and innermost of three gates that together made
        // RRC_INACTIVE unresumable in every shipped configuration: with the flag
        // off the RRC plane never received a KgNB, so it could derive no K_RRCint,
        // so `initiate_resume` had no `resumeMAC-I` to present and fell back to
        // RRC_IDLE — while the gNB, which derives the same key with no such check
        // (`activate_as_security`, nextgsim-gnb), had already committed to holding
        // an RRC_INACTIVE context.
        //
        // Ungating it protects nothing less: `KgNB` never crosses the air in either
        // direction (TS 33.501 §6.2 derives it independently at both ends from
        // KAMF), so this message is in-process only and changes no byte on the
        // wire. What the RRC plane DOES with it is still gated — see
        // `store_inactive_as_context` (nextgsim-ue/src/rrc/task.rs), which records
        // the keys for the `resumeMAC-I` of §5.3.13.3 without activating SRB1
        // protection.
        if let Some(kgnb) = self.derive_kgnb_for_as_security(smc_complete_ul_count) {
            outs.push(MmOutput::AsSecurityKgnb(kgnb));
        }
        outs
    }

    /// Derive KgNB for AS security from the active NAS context (TS 33.501 Annex
    /// A.9: `KgNB = KDF(KAMF, uplink NAS COUNT, access type distinguisher)`, with
    /// the distinguisher `0x01` for 3GPP access per table A.9-1). Returns `None`
    /// when no KAMF is available (NAS security not established).
    ///
    /// # Why the COUNT is a parameter (issue #203)
    ///
    /// It used to read `self.sec.uplink_count()` itself, and the sole caller
    /// invoked it AFTER `protect_uplink` had already sent the Security Mode
    /// Complete — and `protect_uplink` calls `increment_uplink_count`. So the
    /// derivation ran on the count of the NEXT uplink message rather than the one
    /// the SMComplete was protected with: a UE whose SMComplete went out at COUNT 0
    /// derived its KgNB at COUNT 1.
    ///
    /// TS 33.501 §6.8.1.1.2.2 requires "the uplink NAS COUNT of the most recent NAS
    /// Security Mode Complete", and §6.8.1.1.2.3 says "the NAS SMC complete message
    /// shall include the start value of the uplink NAS COUNT that is used as
    /// freshness parameter in the KgNB derivation" — i.e. the value carried in that
    /// message's own sequence number, which is the pre-increment one. §6.8.1.2.2
    /// repeats it for the CM-IDLE→CM-CONNECTED case. So the off-by-one was the UE's:
    /// the AMF derives from the count it accepted off the SMComplete's header.
    ///
    /// KgNB never crosses the air (§6.2 derives it independently at both ends from
    /// KAMF), so nothing on the wire reveals the skew. It stayed latent until #202
    /// made the RRC_INACTIVE resume reachable, because that made `K_RRCint` the key
    /// a `resumeMAC-I` is verified with — the first thing to compare the two ends'
    /// copies. Every AS key derived from a skewed KgNB differs, so the
    /// `resumeMAC-I` of TS 38.331 §5.3.13.3 could never verify and the gNB answered
    /// with silence, leaving the UE to fall back (§5.3.13.5).
    ///
    /// Taking the COUNT as an argument rather than re-reading it is the point: the
    /// caller must capture it before the send advances it, and a signature that
    /// forces that cannot silently regress to reading post-increment state.
    pub fn derive_kgnb_for_as_security(&self, uplink_nas_count: u32) -> Option<[u8; 32]> {
        let kamf = self.sec.keys().kamf()?;
        Some(derive_kgnb(kamf, uplink_nas_count, ACCESS_TYPE_3GPP))
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
        // Recorded first, so uplink for any PSI is remembered even when the
        // request itself does not go out now (in progress, or barred).
        if service_type == ServiceType::Data {
            self.held_uplink_data_status =
                Some(self.held_uplink_data_status.unwrap_or(0) | uplink_data_status.unwrap_or(0));
        }
        // TS 24.501 §5.6.1.2: a service request is initiated when the UE is not
        // already in 5GMM-SERVICE-REQUEST-INITIATED; while T3517 runs the
        // procedure is in progress. Before #317 every uplink packet that arrived
        // between the first SERVICE REQUEST and the RRC connection setup built
        // and sent another one, each with a new UL NAS COUNT and a restarted
        // T3517, which the AMF then saw as duplicates.
        if self.service_request_in_progress() {
            debug!("Service Request ({service_type:?}) not started: one is already in progress");
            return Vec::new();
        }
        if let Some(reason) = self.service_request_barred(service_type) {
            info!("Service Request ({service_type:?}) not started: {reason}");
            return Vec::new();
        }
        // 5GMM-REGISTERED.ATTEMPTING-REGISTRATION-UPDATE (TS 24.501 §5.2.3.2.3):
        // the registration area is not up to date, so the UE does not request
        // service for its own data until the update succeeds; answering a page,
        // emergency services and high priority access are still allowed (#277).
        if self.state.mm_substate() == MmSubState::RegisteredAttemptingRegistrationUpdate
            && !matches!(
                service_type,
                ServiceType::MobileTerminatedServices
                    | ServiceType::EmergencyServices
                    | ServiceType::EmergencyServicesFallback
                    | ServiceType::HighPriorityAccess
            )
        {
            info!(
                "Service Request ({service_type:?}) not started: registration update pending \
                 (5GMM-REGISTERED.ATTEMPTING-REGISTRATION-UPDATE)"
            );
            return Vec::new();
        }
        if let Some(reason) = self.service_request_barred(service_type) {
            info!("Service Request ({service_type:?}) not started: {reason}");
            return Vec::new();
        }
        let Some(s_tmsi) = self.build_s_tmsi() else {
            warn!("Service Request not started: no stored 5G-GUTI");
            return Vec::new();
        };

        let ksi = self.sec.nas_ksi();
        let mut req = ServiceRequest::new(ksi, IeServiceType::new(service_type), s_tmsi.clone());
        req.uplink_data_status = uplink_data_status;

        let mut plain = Vec::new();
        req.encode(&mut plain);

        // TS 24.501 §5.6.1.2.1 / §4.4.6 b): a SERVICE REQUEST sent from
        // 5GMM-IDLE is an initial NAS message. With a valid context it is sent
        // integrity protected but NOT ciphered (§4.4.5), carrying the cleartext
        // IEs only; any non-cleartext IE (Uplink data status, PDU session
        // status, Allowed PDU session status) travels inside the NAS message
        // container, whose value part is ciphered. Sending the whole message
        // ciphered (security header type 2) hides the 5G-S-TMSI the AMF needs to
        // find the UE's context, so the network can never answer it.
        let mut cleartext_only = ServiceRequest::new(ksi, IeServiceType::new(service_type), s_tmsi);
        let mut cleartext = Vec::new();
        cleartext_only.encode(&mut cleartext);
        let initial = self.state.is_idle();

        self.pending_service_type = Some(service_type);
        self.state
            .switch_mm_state(MmSubState::ServiceRequestInitiated);
        self.timers.start(TIMER_T3517, true);
        info!("Sending Service Request ({service_type:?}), T3517 started");
        let pdu = if initial && self.sec.is_active() {
            self.protect_initial_nas(plain, cleartext, |container| {
                cleartext_only.nas_message_container = Some(container);
                let mut out = Vec::new();
                cleartext_only.encode(&mut out);
                out
            })
        } else {
            self.protect_if_active(plain)
        };
        vec![MmOutput::SendNasPdu(pdu)]
    }

    /// Protect an initial NAS message sent from 5GMM-IDLE with a valid 5G NAS
    /// security context (TS 24.501 §4.4.6 b), §4.4.5).
    ///
    /// `full` is the message with every IE; `cleartext` the same message with
    /// the cleartext IEs only. When they are equal the message is sent as is
    /// (§4.4.6 b) 3)); otherwise `full` is ciphered and handed to
    /// `with_container` to be wrapped in the NAS message container of the
    /// cleartext message (§4.4.6 b) 1)). Either way the result is integrity
    /// protected but not ciphered: security header type 1. The container is
    /// ciphered with the same uplink NAS COUNT the MAC is computed over
    /// (TS 33.501 §6.4.6), so it is ciphered BEFORE the count advances.
    fn protect_initial_nas(
        &mut self,
        full: Vec<u8>,
        cleartext: Vec<u8>,
        with_container: impl FnOnce(Vec<u8>) -> Vec<u8>,
    ) -> Vec<u8> {
        let outer = if full == cleartext {
            full
        } else {
            let count = *self.sec.uplink_count();
            let mut value = full;
            if let Some(key) = self.sec.keys().knas_enc() {
                nas_cipher(
                    self.sec.ciphering_algorithm(),
                    key,
                    &count,
                    NAS_BEARER,
                    NasDirection::Uplink,
                    &mut value,
                );
            }
            with_container(value)
        };
        self.protect_uplink(outer, SecurityHeaderType::IntegrityProtected)
    }

    /// Derive the 5G-S-TMSI mobile identity from the stored 5G-GUTI
    /// (TS 24.501 Section 9.11.3.4: AMF set ID + AMF pointer + 5G-TMSI).
    fn build_s_tmsi(&self) -> Option<Ie5gsMobileIdentity> {
        let s_tmsi = self.paging_identity()?;
        let mut data = Vec::with_capacity(1 + FIVE_G_S_TMSI_OCTETS);
        data.push(0xF0 | (MobileIdentityType::Tmsi as u8));
        data.extend_from_slice(&s_tmsi);
        Some(Ie5gsMobileIdentity::new(MobileIdentityType::Tmsi, data))
    }

    /// Whether a service request procedure is in progress: T3517 running, or
    /// 5GMM-SERVICE-REQUEST-INITIATED (TS 24.501 §5.6.1.2, #317).
    pub fn service_request_in_progress(&self) -> bool {
        self.timers.t3517.is_running()
            || self.state.mm_substate() == MmSubState::ServiceRequestInitiated
    }

    /// Abort a service request procedure in progress: stop T3517 and leave
    /// 5GMM-SERVICE-REQUEST-INITIATED for 5GMM-REGISTERED.NORMAL-SERVICE
    /// (TS 24.501 §5.6.1.7, #316). Returns whether one was in progress.
    fn abort_service_request(&mut self, why: &str) -> bool {
        if !self.service_request_in_progress() {
            return false;
        }
        info!("Service request procedure aborted ({why}): T3517 stopped");
        self.timers.stop(TIMER_T3517, true);
        if self.state.mm_substate() == MmSubState::ServiceRequestInitiated {
            self.state
                .switch_mm_state(MmSubState::RegisteredNormalService);
        }
        true
    }

    /// Whether uplink user data can be sent now: a NAS signalling connection
    /// is up and no SERVICE REQUEST is awaiting its answer (T3517 stopped).
    ///
    /// From CM-IDLE the user plane comes back only once the service request
    /// procedure completes (TS 24.501 §5.6.1.2, TS 23.502 §4.2.3.2 steps
    /// 12-13: the AMF's Initial Context Setup re-establishes the N3 tunnel the
    /// Service Accept rides with). Data written before that -- including the
    /// packet that triggered the procedure -- has nowhere to go and must be
    /// held (issue #267).
    pub fn uplink_user_plane_ready(&self) -> bool {
        self.state.is_connected()
            && !self.timers.t3517.is_running()
            && !self.up_reactivation_pending
            // §5.2.3.2.3 a): in ATTEMPTING-REGISTRATION-UPDATE the UE "shall not
            // send any user data", even over a user plane still up (#277 review).
            && self.state.mm_substate() != MmSubState::RegisteredAttemptingRegistrationUpdate
    }

    /// Whether uplink data in CM-CONNECTED must first trigger a Service Request
    /// to re-establish the user plane lost with an earlier connection (#288).
    ///
    /// False while a new data Service Request is barred (#313): under T3346 after
    /// a #22 reject, in 5GMM-REGISTERED.NON-ALLOWED-SERVICE after #28, or inside
    /// the backoff after another reject. The user plane stays down meanwhile, so
    /// [`Self::uplink_user_plane_ready`] keeps holding the uplink.
    pub fn user_plane_reactivation_needed(&self) -> bool {
        self.up_reactivation_pending
            && self.state.is_registered()
            && self.state.is_connected()
            && !self.timers.t3517.is_running()
            && self.service_request_barred(ServiceType::Data).is_none()
    }

    /// Why a SERVICE REQUEST of `service_type` may not be initiated now, if it
    /// may not (#313).
    ///
    /// - T3346 running (congestion, TS 24.501 §5.3.9, §5.6.1.5 #22) and
    ///   5GMM-REGISTERED.NON-ALLOWED-SERVICE (§5.3.5, §5.6.1.5 #28) both bar every
    ///   request except those the spec exempts: responding to paging (mobile
    ///   terminated services), emergency services, emergency services fallback
    ///   and high priority access.
    /// - The local backoff after another reject bars only the requests uplink data
    ///   triggers (data, signalling): it is this UE's choice, not the network's,
    ///   and must not stop the UE answering a page.
    fn service_request_barred(&self, service_type: ServiceType) -> Option<&'static str> {
        // §5.2.3.2.4-5: 5GMM-REGISTERED.LIMITED-SERVICE and PLMN-SEARCH offer no
        // normal service; the UE reselects (or selects a PLMN) and only an
        // emergency request may go out. After #27 N1 mode is disabled (§4.9).
        let emergency = matches!(
            service_type,
            ServiceType::EmergencyServices | ServiceType::EmergencyServicesFallback
        );
        if !emergency
            && matches!(
                self.state.mm_substate(),
                MmSubState::RegisteredLimitedService | MmSubState::RegisteredPlmnSearch
            )
        {
            return Some("5GMM-REGISTERED.LIMITED-SERVICE or PLMN-SEARCH (no normal service)");
        }
        if self.n1_mode_disabled {
            return Some("N1 mode is disabled (5GMM cause #27 or #31)");
        }
        // §5.6.1.7 c) 4) exempts "a UE configured for high priority access in
        // selected PLMN": the UE's configuration, not a request's service type.
        let exempt = matches!(
            service_type,
            ServiceType::MobileTerminatedServices
                | ServiceType::EmergencyServices
                | ServiceType::EmergencyServicesFallback
        ) || self.high_priority_access;
        // §5.6.1.7 c) 7): elevated signalling is also exempt from T3346.
        if !exempt
            && service_type != ServiceType::ElevatedSignalling
            && self.timers.t3346.is_running()
        {
            return Some("T3346 (congestion backoff) is running");
        }
        if exempt {
            return None;
        }
        // A configured high-priority UE passed the `exempt` check above, so it
        // is not held here. §5.6.1.5 #28 withdraws that exemption when the
        // network indicates "High priority access UEs are not exempt from
        // service area restrictions"; that indication is not parsed yet, so
        // such a UE is always treated as exempt.
        if self.state.mm_substate() == MmSubState::RegisteredNonAllowedService {
            return Some("5GMM-REGISTERED.NON-ALLOWED-SERVICE (restricted service area)");
        }
        if self
            .sr_backoff_until
            .is_some_and(|until| std::time::Instant::now() < until)
        {
            return Some("backing off after a Service Reject");
        }
        None
    }

    /// Start the local backoff after a SERVICE REJECT that left the user plane
    /// down and named no wait of its own (#313).
    fn start_service_reject_backoff(&mut self) {
        self.sr_reject_count = self.sr_reject_count.saturating_add(1);
        let doublings = (self.sr_reject_count - 1).min(16);
        let wait = SERVICE_REJECT_BACKOFF_BASE
            .saturating_mul(1u32 << doublings)
            .min(SERVICE_REJECT_BACKOFF_MAX);
        info!(
            "Service Request backoff {wait:?} after {} consecutive reject(s)",
            self.sr_reject_count
        );
        self.sr_backoff_until = Some(std::time::Instant::now() + wait);
    }

    fn handle_service_accept(&mut self, plain: &[u8]) -> Vec<MmOutput> {
        info!("Service Accept received");
        self.up_reactivation_pending = false;
        self.held_uplink_data_status = None;
        self.sr_reject_count = 0;
        self.sr_backoff_until = None;
        self.timers.stop(TIMER_T3517, true);
        self.timers.stop(TIMER_T3516, true);
        self.res_star = None;
        self.state
            .switch_mm_state(MmSubState::RegisteredNormalService);
        self.state.switch_cm_state(CmState::Connected);

        // TS 24.501 §5.6.1.4.2 a): the UE shall locally release every PDU
        // session that is active on the UE but indicated as inactive in the
        // PDU session status IE. Before this the handler took no message bytes
        // at all, so the decoded IE was discarded and a session the network had
        // released stayed up on the UE with a live TUN device, black-holing
        // whatever was written to it.
        //
        // A decode failure must not undo the state transitions above: the
        // Service Accept has already been accepted and integrity-verified by
        // the time it reaches here, so the procedure completed. Only the
        // optional reconciliation is lost.
        let mut outs = Vec::new();
        match ServiceAccept::decode(&mut &plain[3..]) {
            Ok(acc) => {
                if let Some(status) = acc.pdu_session_status {
                    outs.push(MmOutput::NetworkPduSessionStatus(status));
                }
                // The reactivation result says which of the sessions the UE
                // ASKED to reactivate the network actually brought up. It is
                // reported rather than acted on: a session the network refused
                // is still active on the UE per §5.6.1.4.2, and the PDU session
                // status IE above is the member that decides releases.
                if let Some(result) = acc.pdu_session_reactivation_result {
                    info!("Service Accept PDU session reactivation result: {result:#06x}");
                }
            }
            Err(e) => {
                warn!(
                    "Service Accept accepted but its body did not decode ({e:?}); \
                     PDU session status not reconciled"
                );
            }
        }
        outs
    }

    fn handle_service_reject(&mut self, plain: &[u8], protected: bool) -> Vec<MmOutput> {
        let rej = match ServiceReject::decode(&mut &plain[3..]) {
            Ok(rej) => rej,
            Err(e) => {
                warn!("Failed to decode Service Reject: {e:?}");
                return Vec::new();
            }
        };
        // The cause octet as sent: #78 has no MmCause variant, so the decoded
        // value cannot tell it apart from any other unknown cause.
        let raw_cause = plain.get(3).copied().unwrap_or(0);
        // TS 24.501 §5.6.1.5: a SERVICE REJECT with #76 or #78 that is not
        // integrity protected is discarded, before anything else acts on it.
        if !protected && matches!(raw_cause, 76 | 78) {
            warn!("Service Reject with cause #{raw_cause} without integrity protection: discarded");
            return Vec::new();
        }
        let cause = rej.mm_cause.value;
        warn!("Service Reject: cause {:?} (#{raw_cause})", cause);
        self.timers.stop(TIMER_T3517, true);
        // `up_reactivation_pending` is NOT cleared here (#313). Only the arms
        // that release the PDU sessions clear it; after any other reject the
        // user plane is still down, so uplink stays held and a new request is
        // gated on T3346, NON-ALLOWED-SERVICE, LIMITED-SERVICE, PLMN-SEARCH or
        // the backoff instead.

        let mut outs = Vec::new();
        // §5.6.1.5: an integrity-protected reject's PDU session status IE says
        // which sessions the AMF still has; the UE releases the others locally.
        // From an unprotected one it is ignored.
        if let Some(status) = rej.pdu_session_status {
            if protected {
                outs.push(MmOutput::NetworkPduSessionStatus(status));
            } else {
                warn!("Service Reject PDU session status ignored: not integrity protected");
            }
        }
        // §5.6.1.5, regardless of the cause: the forbidden-TAI IEs.
        let mut tais_changed = self.store_service_reject_forbidden_tais(&rej);
        let snpn = self.identity.snpn_nid.is_some();

        match cause {
            MmCause::IllegalUe | MmCause::IllegalMe | MmCause::FiveGsServicesNotAllowed => {
                // Same terminal handling as for registration (5.6.1.5)
                self.up_reactivation_pending = false;
                self.held_uplink_data_status = None;
                self.state
                    .switch_update_status(UpdateStatus::RoamingNotAllowed);
                self.delete_registration_context();
                if cause != MmCause::FiveGsServicesNotAllowed {
                    // #3/#6 also delete the equivalent PLMNs.
                    self.equivalent_plmns.clear();
                }
                self.usim_invalid = true;
                self.state.switch_mm_state(MmSubState::Deregistered);
                // 5GMM-DEREGISTERED holds no PDU sessions (TS 24.501 §5.6.1.5).
                outs.push(MmOutput::ReleasePduSessionsLocally);
                outs.push(MmOutput::RegistrationFailed(cause));
            }
            MmCause::UeIdentityCannotBeDerived | MmCause::ImplicitlyDeregistered => {
                // TS 24.501 5.6.1.5: delete the GUTI, enter 5GMM-DEREGISTERED and
                // perform a fresh initial registration. #9 also sets 5U2.
                self.up_reactivation_pending = false;
                if cause == MmCause::UeIdentityCannotBeDerived {
                    self.state.switch_update_status(UpdateStatus::NotUpdated);
                }
                self.held_uplink_data_status = None;
                self.delete_registration_context();
                self.state
                    .switch_mm_state(MmSubState::DeregisteredNormalService);
                outs.extend(self.reregister_after_identity_loss());
            }
            // #11 PLMN not allowed, #73 serving network not authorized: leave
            // the PLMN. From an SNPN cell both are abnormal (§5.6.1.7).
            MmCause::PlmnNotAllowed | MmCause::ServingNetworkNotAuthorized if !snpn => {
                self.up_reactivation_pending = false;
                self.state
                    .switch_update_status(UpdateStatus::RoamingNotAllowed);
                self.delete_registration_context();
                self.equivalent_plmns.clear();
                if !self.forbidden_plmns.contains(&self.identity.plmn_bcd) {
                    self.forbidden_plmns.push(self.identity.plmn_bcd);
                }
                self.state
                    .switch_mm_state(MmSubState::DeregisteredPlmnSearch);
                outs.push(MmOutput::ReleasePduSessionsLocally);
                outs.push(MmOutput::PlmnSearchNeeded);
                outs.push(MmOutput::RegistrationFailed(cause));
            }
            // #12 tracking area not allowed: the TA goes on the "regional
            // provision of service" list and the UE de-registers.
            MmCause::TrackingAreaNotAllowed => {
                self.up_reactivation_pending = false;
                self.state
                    .switch_update_status(UpdateStatus::RoamingNotAllowed);
                let tai = self.current_tai_for_reject();
                self.delete_registration_context();
                if let Some(tai) = tai {
                    tais_changed |= Self::add_forbidden_tai(&mut self.forbidden_tais_service, tai);
                }
                self.state
                    .switch_mm_state(MmSubState::DeregisteredLimitedService);
                outs.push(MmOutput::ReleasePduSessionsLocally);
                outs.push(MmOutput::RegistrationFailed(cause));
            }
            // #13 roaming not allowed in this TA: the UE stays registered, the TA
            // goes on the "roaming" list and out of the TAI list, PLMN selection.
            MmCause::RoamingNotAllowedInTa => {
                self.state
                    .switch_update_status(UpdateStatus::RoamingNotAllowed);
                if let Some(tai) = self.current_tai_for_reject() {
                    tais_changed |= Self::add_forbidden_tai(&mut self.forbidden_tais_roaming, tai);
                    self.remove_tai_from_tai_list(&tai);
                }
                self.state.switch_mm_state(MmSubState::RegisteredPlmnSearch);
                outs.push(MmOutput::PlmnSearchNeeded);
            }
            // #15 no suitable cells in TA: REGISTERED.LIMITED-SERVICE and a
            // suitable cell in another TA. The current TAI is not stored when
            // the reject names the forbidden TAIs itself, or carries an access
            // technology utilization control IE under integrity protection.
            MmCause::NoSuitableCellsInTa => {
                let store_current = rej.forbidden_tais_roaming.is_none()
                    && !(rej.access_tech_utilization_control.is_some() && protected);
                if store_current {
                    if let Some(tai) = self.current_tai_for_reject() {
                        tais_changed |=
                            Self::add_forbidden_tai(&mut self.forbidden_tais_roaming, tai);
                        self.remove_tai_from_tai_list(&tai);
                    }
                }
                self.state
                    .switch_mm_state(MmSubState::RegisteredLimitedService);
            }
            // #27 N1 mode not allowed.
            MmCause::N1ModeNotAllowed => {
                self.state
                    .switch_update_status(UpdateStatus::RoamingNotAllowed);
                self.n1_mode_disabled = true;
                self.state
                    .switch_mm_state(MmSubState::RegisteredLimitedService);
            }
            // #74/#75 apply only from an SNPN cell (otherwise abnormal).
            MmCause::TemporarilyNotAuthorizedForSnpn | MmCause::PermanentlyNotAuthorizedForSnpn
                if snpn =>
            {
                self.up_reactivation_pending = false;
                self.state
                    .switch_update_status(UpdateStatus::RoamingNotAllowed);
                self.delete_registration_context();
                if cause == MmCause::TemporarilyNotAuthorizedForSnpn {
                    self.snpn_temporarily_forbidden = true;
                } else {
                    self.snpn_permanently_forbidden = true;
                }
                self.state
                    .switch_mm_state(MmSubState::DeregisteredPlmnSearch);
                outs.push(MmOutput::ReleasePduSessionsLocally);
                outs.push(MmOutput::PlmnSearchNeeded);
                outs.push(MmOutput::RegistrationFailed(cause));
            }
            MmCause::Congestion => {
                // §5.6.1.5 #22: with a T3346 value the UE starts T3346 and may
                // initiate the service request again once it expires. Without
                // one (or a zero value) nothing tells it when, so the local
                // backoff applies.
                let secs = rej.t3346_value.map(gprs_timer2_seconds).unwrap_or(0);
                if secs > 0 {
                    // An unprotected reject's value is not trusted: T3346 then
                    // runs for a random value from TS 24.008's default range,
                    // 15 to 30 minutes (§5.6.1.5 #22).
                    let secs = if protected {
                        secs
                    } else {
                        use rand::Rng;
                        rand::thread_rng().gen_range(15 * 60..=30 * 60)
                    };
                    self.timers
                        .start_with_timer2(TIMER_T3346, GprsTimer2::new(secs), true);
                    info!("T3346 (congestion backoff) started: {secs}s");
                } else {
                    // No usable T3346 value: an abnormal case (§5.6.1.7 i):
                    // REGISTERED, procedure aborted. The backoff is this UE's.
                    self.start_service_reject_backoff();
                }
                self.state
                    .switch_mm_state(MmSubState::RegisteredNormalService);
            }
            MmCause::RestrictedServiceArea => {
                // §5.6.1.5 #28: enter 5GMM-REGISTERED.NON-ALLOWED-SERVICE, wait
                // for the release of the N1 NAS signalling connection and perform
                // the registration procedure for mobility and periodic update
                // (unless the request was for elevated signalling, which then
                // waits for an allowed area). The PDU sessions stay; their user
                // plane cannot be activated meanwhile (§5.3.5).
                self.state
                    .switch_mm_state(MmSubState::RegisteredNonAllowedService);
                if self.pending_service_type != Some(ServiceType::ElevatedSignalling) {
                    if self.state.is_connected() {
                        self.mobility_registration_on_release = true;
                    } else {
                        outs.extend(self.build_and_send_registration(
                            RegistrationType::MobilityRegistrationUpdating,
                        ));
                    }
                }
            }
            // Every other cause, and those §5.6.1.5 itself calls abnormal here:
            // #11/#73 from an SNPN cell, #31 (this UE indicates no CIoT
            // optimization), #36 (not an IAB-node; no MmCause variant, so it
            // lands here), #72 over 3GPP access, #74/#75 from a non-SNPN cell,
            // #76 (this UE does not support CAG, so it holds no CAG lists to
            // update), #77 (not wireline), #78 (not a satellite cell). §5.6.1.7
            // i): 5GMM-REGISTERED, procedure aborted, resources released. Nothing
            // says when to retry, so the backoff is this UE's own.
            _ => {
                self.start_service_reject_backoff();
                self.state
                    .switch_mm_state(MmSubState::RegisteredNormalService);
            }
        }
        if tais_changed {
            outs.push(self.forbidden_tais_output());
        }
        outs
    }

    /// Store the TAIs of a SERVICE REJECT's forbidden-TAI IEs (TS 24.501
    /// §5.6.1.5, "regardless of the 5GMM cause value"), if not already stored.
    /// Only TAIs of the serving PLMN or an equivalent PLMN are kept; the rest
    /// are ignored. True if a list changed.
    ///
    /// Rel-19 scopes these IEs to a reject received via satellite NG-RAN, and
    /// the AMF includes them only then. This NAS layer does not know whether
    /// the cell is a satellite one, so it honours them whenever they arrive: an
    /// AMF that sends them means them.
    fn store_service_reject_forbidden_tais(&mut self, rej: &ServiceReject) -> bool {
        let serving_plmn: [u8; 3] = match self.serving_tai {
            Some(t) => [t[0], t[1], t[2]],
            None => self.identity.plmn_bcd,
        };
        let ours = |tai: &[u8; 6]| {
            let plmn = [tai[0], tai[1], tai[2]];
            plmn == serving_plmn || self.equivalent_plmns.contains(&plmn)
        };
        let mut roaming: Vec<[u8; 6]> = Vec::new();
        let mut service: Vec<[u8; 6]> = Vec::new();
        if let Some(ref v) = rej.forbidden_tais_roaming {
            roaming = tai_list_entries(v)
                .into_iter()
                .filter(|t| ours(t))
                .collect();
        }
        if let Some(ref v) = rej.forbidden_tais_service {
            service = tai_list_entries(v)
                .into_iter()
                .filter(|t| ours(t))
                .collect();
        }
        let mut changed = false;
        for tai in roaming {
            changed |= Self::add_forbidden_tai(&mut self.forbidden_tais_roaming, tai);
        }
        for tai in service {
            changed |= Self::add_forbidden_tai(&mut self.forbidden_tais_service, tai);
        }
        changed
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
        if switch_off {
            // TS 24.501 §5.5.2.2.1: a switched-off UE may power down as soon as
            // the request is sent, and the AMF answers a switch-off with no
            // DEREGISTRATION ACCEPT (§5.5.2.2.2), so nothing would ever move the
            // UE out of 5GMM-DEREGISTERED-INITIATED. 5GS services are now
            // disabled: 5GMM-NULL (§5.1.3.2.1.1). Switched after protecting, so
            // the request still goes out under the active context. T3512 is
            // stopped as the DEREGISTRATION ACCEPT path stops it: left running,
            // a switched-off UE started a periodic registration update 9
            // minutes later against an AMF that had deleted its context.
            self.timers.stop(TIMER_T3512, true);
            self.state.switch_mm_state(MmSubState::Null);
        }
        vec![MmOutput::SendNasPdu(pdu)]
    }

    fn handle_deregistration_accept(&mut self) -> Vec<MmOutput> {
        self.held_uplink_data_status = None;
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
            // De-registered over 3GPP access: the PDU sessions go with it, so the
            // default session is re-established after the next registration
            // (nextgsim #272). The UE-initiated path already did this.
            outs.push(MmOutput::ReleasePduSessionsLocally);
        }

        if let Some(sub) = result.new_sub_state {
            self.state.switch_mm_state(sub);
        }
        self.dereg_pdu = None;

        // §5.5.2.3.2: re-registration required — release local resources and
        // initiate an initial registration "after the completion of the
        // de-registration procedure and the release of the existing NAS
        // signalling connection". Sending it at once put the REGISTRATION
        // REQUEST on the connection the AMF was already releasing after the
        // DEREGISTRATION ACCEPT: amfd took it into a UE context it then
        // released, nothing answered, and every such re-registration waited
        // out T3510 before the retry succeeded (issue #265).
        if result.re_registration_required {
            let _ = proc.perform_local_deregistration();
            self.delete_registration_context();
            self.state
                .switch_mm_state(MmSubState::DeregisteredNormalService);
            if self.state.is_connected() {
                info!(
                    "Re-registration required: initial registration deferred until the \
                     NAS signalling connection is released (TS 24.501 §5.5.2.3.2)"
                );
                self.reregistration_on_release = true;
            } else {
                outs.extend(
                    self.build_and_send_registration(RegistrationType::InitialRegistration),
                );
            }
        }

        outs
    }

    /// The lower layers failed to establish the RRC connection a NAS message
    /// was waiting for (T300 expiry, RRC reject). A service request in progress
    /// is aborted: T3517 stopped, 5GMM-REGISTERED (TS 24.501 §5.6.1.7 l), #316).
    /// A registration in progress is left to T3510.
    pub fn on_rrc_establishment_failure(&mut self) {
        self.abort_service_request("RRC connection establishment failed");
    }

    /// The NAS signalling connection was released (RRC release or radio link
    /// failure): the UE is CM-IDLE. Starts the initial registration a
    /// network-initiated de-registration deferred to this point
    /// (TS 24.501 §5.5.2.3.2), unless one has already been started.
    pub fn on_nas_signalling_connection_released(&mut self) -> Vec<MmOutput> {
        self.state.switch_cm_state(CmState::Idle);
        // TS 24.501 §5.6.1.7: a lower-layer release or failure before the
        // service request completes aborts the procedure; the UE stops T3517
        // and enters 5GMM-REGISTERED. Left running, T3517 kept the uplink held
        // and refused a new Service Request for up to 15 s after the connection
        // it belonged to was gone (#316).
        self.abort_service_request("NAS signalling connection released");
        // A secure exchange belongs to one connection (§4.4.4.2).
        self.secure_exchange = false;
        // So do the user-plane resources (#288).
        if self.state.is_registered() {
            self.up_reactivation_pending = true;
        }
        if std::mem::take(&mut self.mobility_registration_on_release) && self.state.is_registered()
        {
            info!(
                "NAS signalling connection released after SERVICE REJECT #28: registration \
                 for mobility and periodic update (TS 24.501 §5.6.1.5)"
            );
            return self
                .build_and_send_registration(RegistrationType::MobilityRegistrationUpdating);
        }
        if !std::mem::take(&mut self.reregistration_on_release) {
            return Vec::new();
        }
        if !self.state.is_deregistered()
            || self.state.mm_substate() == MmSubState::RegisteredInitiated
        {
            debug!("Deferred re-registration dropped: a registration already ran");
            return Vec::new();
        }
        info!("NAS signalling connection released: initiating the deferred initial registration");
        self.build_and_send_registration(RegistrationType::InitialRegistration)
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
            // messages has been established FOR THIS NAS signalling connection,
            // the UE shall discard any NAS signalling message received without
            // integrity protection. Before that, `handle_plain`'s allow-list
            // decides (e.g. SERVICE REJECT #9 from an AMF that lost the UE).
            if self.sec.is_active() && self.secure_exchange {
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
        self.secure_exchange = true;

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
            MmMessageType::ServiceAccept => self.handle_service_accept(&plain),
            MmMessageType::ServiceReject => self.handle_service_reject(&plain, protected),
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

    /// Every URSP rule across all stored UE policy sections (TS 24.526 §5.2).
    ///
    /// The rules were decoded and stored and had no reader other than a unit
    /// test, so a PCF's URSP had no runtime effect at all (#47). This is what
    /// hands them to the SM orchestrator's evaluation engine.
    ///
    /// Rules from ALL sections are returned together, not per section: a section
    /// is a delivery/versioning unit (its UPSC is how the PCF replaces or deletes
    /// it), while the URSP the UE evaluates is the union of what it holds. §5.2
    /// orders by precedence across the whole policy, not within a section.
    pub fn all_ursp_rules(&self) -> Vec<UrspRule> {
        self.ue_policy_sections
            .values()
            .filter_map(|section| section.ursp_rules())
            .flatten()
            .collect()
    }

    /// Number of UE policy sections currently stored (test/observability hook).
    pub fn ue_policy_section_count(&self) -> usize {
        self.ue_policy_sections.len()
    }

    /// The stored UE policy section for `(plmn, upsc)`, if any (test hook).
    pub fn ue_policy_section(&self, plmn: PlmnId, upsc: u16) -> Option<&StoredUePolicySection> {
        self.ue_policy_sections.get(&(plmn, upsc))
    }

    /// The next UE-initiated UPDP PTI (TS 24.501 D.1.2: 01H-77H).
    ///
    /// D.2.2.2 a) wants "a PTI value currently not used", so this cycles the
    /// whole range rather than reusing one value. **Not zero:** the issue that
    /// asked for this suggested PTI 0, but D.1.2 reserves 01H-77H for the UE and
    /// 0 means "no procedure transaction identity assigned".
    fn next_ue_updp_pti(&mut self) -> u8 {
        // Cycle 0x01..=0x77. UE STATE INDICATION draws no response (D.2.2.3), so
        // nothing outstanding can collide; cycling is about not repeating a
        // value the PCF has just seen.
        self.updp_pti = if self.updp_pti >= ue_policy::UE_PTI_MAX {
            ue_policy::UE_PTI_MIN
        } else {
            self.updp_pti + 1
        };
        self.updp_pti
    }

    /// The UE policy classmark this UE advertises (TS 24.501 D.6.5).
    ///
    /// ANDSP is claimed because the UPDP codec decodes and stores an ANDSP part
    /// (`UePolicyPartType::Andsp`). The other three are **not**: there is no EPS
    /// URSP path, no VPS URSP, and nothing reports URSP rule enforcement in this
    /// tree, and claiming a capability the UE does not have is worse than not
    /// claiming it — a PCF that trusts the bit withholds or mis-targets policy.
    fn ue_policy_classmark(&self) -> UePolicyClassmark {
        UePolicyClassmark {
            andsp: true,
            ursp_in_eps: false,
            vps_ursp: false,
            report_ursp_rule_enforcement: false,
        }
    }

    /// Build the UE STATE INDICATION to carry in the REGISTRATION REQUEST
    /// Payload container (TS 24.501 §5.5.1.2.2, D.5.4), or `None` when there is
    /// nothing to report.
    ///
    /// Returns `None` — and so omits the Payload container entirely — when no
    /// section is stored for the selected PLMN. §5.5.1.2.2's second paragraph
    /// does allow a section-less indication, but only when "the UE needs to send
    /// a UE policy container to the network", and this UE has no other reason
    /// to: its classmark has not changed and it reports no OS Id. Sending one
    /// anyway would make every clean first attach carry a container it has
    /// nothing to say in.
    fn build_ue_state_indication(&mut self) -> Option<Vec<u8>> {
        // §5.5.1.2.2: sections identified by a UPSI whose PLMN ID part is the
        // HPLMN or the selected PLMN. In this simulator the camped cell
        // broadcasts the configured PLMN, so those are the same value.
        let selected = PlmnId::from_bcd(self.identity.plmn_bcd).ok()?;
        let mut upscs: Vec<u16> = self
            .ue_policy_sections
            .keys()
            .filter(|(plmn, _)| *plmn == selected)
            .map(|(_, upsc)| *upsc)
            .collect();
        if upscs.is_empty() {
            return None;
        }
        // HashMap iteration order is arbitrary; sort so the encoded bytes are a
        // function of the stored state and a test can assert them.
        upscs.sort_unstable();

        let indication = UeStateIndication {
            pti: self.next_ue_updp_pti(),
            upsi_list: vec![UpsiSublist {
                plmn_id: selected,
                upscs,
            }],
            classmark: self.ue_policy_classmark(),
        };
        match indication.encode() {
            Ok(bytes) => {
                info!(
                    "REGISTRATION REQUEST will report {} stored UE policy section(s) in a UE \
                     STATE INDICATION (TS 24.501 §5.5.1.2.2)",
                    indication
                        .upsi_list
                        .iter()
                        .map(|s| s.upscs.len())
                        .sum::<usize>()
                );
                Some(bytes)
            }
            Err(e) => {
                warn!("Cannot encode UE STATE INDICATION ({e}); omitting the Payload container");
                None
            }
        }
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

/// Whether `tai` (PLMN + TAC, 6 octets) is in a 5GS tracking area identity list
/// IE value (TS 24.501 §9.11.3.9).
///
/// The value is a sequence of partial lists. Each starts with an octet whose
/// bits 7-6 give the type and bits 5-1 the number of elements minus one:
/// - `00`: one PLMN, then that many TACs;
/// - `01`: one PLMN and one TAC, standing for that many consecutive TACs;
/// - `10`: that many PLMN + TAC pairs.
///
/// A truncated or unknown partial list ends the scan: what was parsed so far
/// still counts, and nothing past it is guessed at.
fn tai_list_contains(list: &[u8], tai: &[u8; 6]) -> bool {
    let (plmn, tac) = (&tai[..3], &tai[3..]);
    let tac_value = |t: &[u8]| (u32::from(t[0]) << 16) | (u32::from(t[1]) << 8) | u32::from(t[2]);
    let mut rest = list;
    while let Some((&head, body)) = rest.split_first() {
        let kind = (head >> 5) & 0x03;
        let n = usize::from(head & 0x1F) + 1;
        match kind {
            0b00 => {
                let Some(part) = body.get(..3 + 3 * n) else {
                    return false;
                };
                if &part[..3] == plmn && part[3..].chunks_exact(3).any(|t| t == tac) {
                    return true;
                }
                rest = &body[3 + 3 * n..];
            }
            0b01 => {
                let Some(part) = body.get(..6) else {
                    return false;
                };
                if &part[..3] == plmn {
                    let first = tac_value(&part[3..6]);
                    let want = tac_value(tac);
                    // n consecutive TACs from `first`; u64 so the end cannot wrap.
                    if want >= first && u64::from(want) < u64::from(first) + n as u64 {
                        return true;
                    }
                }
                rest = &body[6..];
            }
            0b10 => {
                let Some(part) = body.get(..6 * n) else {
                    return false;
                };
                if part.chunks_exact(6).any(|t| t == tai) {
                    return true;
                }
                rest = &body[6 * n..];
            }
            _ => return false,
        }
    }
    false
}

/// Every TAI in a 5GS tracking area identity list IE value (TS 24.501
/// §9.11.3.9), in order. A type `01` partial list (consecutive TACs) is
/// expanded. A truncated or unknown partial list ends the walk, as in
/// [`tai_list_contains`].
fn tai_list_entries(list: &[u8]) -> Vec<[u8; 6]> {
    let mut out = Vec::new();
    let tai_of = |plmn: &[u8], tac: u32| {
        let mut t = [0u8; 6];
        t[..3].copy_from_slice(plmn);
        t[3..].copy_from_slice(&tac.to_be_bytes()[1..]);
        t
    };
    let mut rest = list;
    while let Some((&head, body)) = rest.split_first() {
        let n = usize::from(head & 0x1F) + 1;
        match (head >> 5) & 0x03 {
            0b00 => {
                let Some(part) = body.get(..3 + 3 * n) else {
                    break;
                };
                for tac in part[3..].chunks_exact(3) {
                    out.push(tai_of(
                        &part[..3],
                        u32::from_be_bytes([0, tac[0], tac[1], tac[2]]),
                    ));
                }
                rest = &body[3 + 3 * n..];
            }
            0b01 => {
                let Some(part) = body.get(..6) else {
                    break;
                };
                let first = u32::from_be_bytes([0, part[3], part[4], part[5]]);
                // TACs are 24 bits; a run past 0xFFFFFF does not exist.
                for i in 0..n as u32 {
                    let Some(tac) = first.checked_add(i).filter(|t| *t <= 0x00FF_FFFF) else {
                        break;
                    };
                    out.push(tai_of(&part[..3], tac));
                }
                rest = &body[6..];
            }
            0b10 => {
                let Some(part) = body.get(..6 * n) else {
                    break;
                };
                for t in part.chunks_exact(6) {
                    out.push([t[0], t[1], t[2], t[3], t[4], t[5]]);
                }
                rest = &body[6 * n..];
            }
            _ => break,
        }
    }
    out
}

/// Encode `tais` as a 5GS tracking area identity list IE value of type `10`
/// partial lists (TS 24.501 §9.11.3.9), at most 16 elements each. `None` for no
/// TAIs: an empty list is not a list.
fn encode_tai_list(tais: &[[u8; 6]]) -> Option<Vec<u8>> {
    if tais.is_empty() {
        return None;
    }
    let mut out = Vec::new();
    for chunk in tais.chunks(16) {
        out.push(0x40 | (chunk.len() as u8 - 1));
        for t in chunk {
            out.extend_from_slice(t);
        }
    }
    Some(out)
}

/// Parse the Equivalent PLMNs IE value (TS 24.501 §9.11.3.45): a sequence of
/// 3-octet BCD-encoded PLMNs. A trailing partial octet group is ignored.
fn parse_equivalent_plmns(value: &[u8]) -> Vec<[u8; 3]> {
    value.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nas::mm::config_update::{
        ConfigUpdateProcedure, ConfigurationUpdateCommand, RacsDeletionRequest,
    };
    use nextgsim_nas::ies::service_level_aa::{
        ServiceLevelAaContainer, ServiceLevelAaResponse, ServiceLevelAaResult,
    };
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
            high_priority_access: false,
            disaster_roaming: false,
            uav_indication: None,
            redcap: false,
        }
    }

    fn new_orch() -> MmOrchestrator {
        MmOrchestrator::new(test_identity())
    }

    // ========================================================================
    // Non-volatile 5GMM parameters (TS 24.501 Annex C.1) — #53
    // ========================================================================

    /// A UE config pointing at a private state file. PID- and
    /// nanosecond-tagged: these tests run concurrently in one process, and a
    /// shared path would have them read each other's snapshots.
    fn state_file_config(name: &str) -> (UeConfig, std::path::PathBuf) {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir()
            .join(format!(
                "nextgsim-mm-state-{}-{name}-{nanos}",
                std::process::id()
            ))
            .join("state.json");
        let mut config = UeConfig::default();
        config.state_file = Some(path.clone());
        (config, path)
    }

    /// The identity a UE presents in its Registration Request, decoded off the
    /// PDU it actually emits, plus whether that PDU was security protected.
    ///
    /// Both forms have to be handled, and the difference is itself a result of
    /// #53: a UE with no stored context sends a PLAIN request (there is no key),
    /// while one that restored a native context sends an INTEGRITY-PROTECTED and
    /// ciphered one. Decoding only the plain shape would have made the restored
    /// case unreadable rather than wrong.
    fn registration_identity(orch: &mut MmOrchestrator) -> (MobileIdentityType, u8, bool) {
        let outs = orch.start_registration(RegistrationType::InitialRegistration);
        let pdu = first_sent_pdu(&outs).to_vec();
        let protected = pdu[1] != 0x00;
        let (_, req) = open_registration_request(orch, &pdu);
        (req.mobile_identity.identity_type, req.ng_ksi.ksi, protected)
    }

    /// #53: a UE that completed registration, was persisted and restarted
    /// registers GUTI-FIRST with the stored ngKSI (TS 24.501 Annex C.1,
    /// §5.5.1.2.2), instead of falling back to SUCI as it did before.
    ///
    /// "Restart" is a genuinely fresh orchestrator built through
    /// `from_config` — the same path `main` uses — so this exercises the load,
    /// not an in-memory copy.
    #[test]
    fn a_restarted_ue_registers_guti_first_with_the_stored_ngksi() {
        let (config, path) = state_file_config("guti-first");

        // First life: register, receive a 5G-GUTI, persist.
        let fresh = MmOrchestrator::from_config(test_identity(), &config);
        assert!(
            fresh.stored_guti().is_none(),
            "precondition: nothing stored yet"
        );
        let orch = {
            let mut orch = establish_security_context();
            orch.state_file = Some(path.clone());
            let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
            orch.handle_downlink(&acc);
            assert!(orch.stored_guti().is_some(), "the Accept assigned a GUTI");
            orch.persist_state();
            orch
        };
        let stored_ksi = orch.security_context().ng_ksi();
        assert!(path.exists(), "persist_state must have written the file");

        // Second life: a fresh orchestrator loading that file.
        let mut restarted = MmOrchestrator::from_config(test_identity(), &config);
        assert!(
            restarted.stored_guti().is_some(),
            "the 5G-GUTI must be rehydrated BEFORE the first registration"
        );
        assert!(
            restarted.security_context().is_active(),
            "the native security context must come back active, or the ngKSI is meaningless"
        );

        let (identity_type, ng_ksi, protected) = registration_identity(&mut restarted);
        assert_eq!(
            identity_type,
            MobileIdentityType::Guti,
            "a restarted UE with a stored GUTI must not take the SUCI fallback"
        );
        assert_eq!(
            ng_ksi, stored_ksi,
            "the request must carry the STORED key set identifier, not 'no key'"
        );
        assert!(
            protected,
            "with a restored native context the request is integrity protected, not plain -- \
             which is the point of storing the context rather than only the GUTI"
        );

        UeStateSnapshot::remove(&path).unwrap();
    }

    /// #53: with no state file the behaviour is unchanged — SUCI, no key.
    ///
    /// This is the default path, so it is the one that must not regress.
    #[test]
    fn without_a_state_file_the_ue_still_registers_with_suci() {
        let mut orch = MmOrchestrator::from_config(test_identity(), &UeConfig::default());
        let (identity_type, ng_ksi, protected) = registration_identity(&mut orch);
        assert_eq!(identity_type, MobileIdentityType::Suci);
        assert_eq!(ng_ksi, NasKeySetIdentifier::NO_KEY_AVAILABLE);
        assert!(!protected, "no stored context means a plain request");
    }

    /// #53: a corrupt state file is ignored, not fatal, and the UE registers
    /// cleanly with a SUCI.
    ///
    /// A UE that cannot start because of a damaged cache is worse than one that
    /// re-registers, which is a spec-valid procedure the network completes.
    #[test]
    fn a_corrupt_state_file_falls_back_to_suci_without_panicking() {
        let (config, path) = state_file_config("corrupt");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"{\"guti\": ").unwrap();

        let mut orch = MmOrchestrator::from_config(test_identity(), &config);
        assert!(orch.stored_guti().is_none());
        let (identity_type, _, _) = registration_identity(&mut orch);
        assert_eq!(identity_type, MobileIdentityType::Suci);

        UeStateSnapshot::remove(&path).unwrap();
    }

    /// #53: deleting the registration context removes the stored copy, so a
    /// restart cannot present a 5G-GUTI the network has told this UE to forget.
    ///
    /// Driven through a real reject (#3 ILLEGAL UE) rather than by calling the
    /// private helper, so the wiring is what is under test.
    #[test]
    fn a_deleted_registration_context_takes_the_stored_state_with_it() {
        let (config, path) = state_file_config("invalidate");

        let mut orch = establish_security_context();
        orch.state_file = Some(path.clone());
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);
        orch.persist_state();
        assert!(path.exists(), "precondition: state was persisted");

        // ILLEGAL UE is one of the causes whose handling deletes the
        // registration context (TS 24.501 §5.5.1.2.5).
        let rej = build_registration_reject_pdu(MmCause::IllegalUe, None);
        orch.handle_downlink(&protect_downlink(&orch, &rej, 2));

        assert!(orch.stored_guti().is_none(), "in-memory GUTI is gone");
        assert!(
            !path.exists(),
            "the stored copy must go with it, or a restart presents a stale 5G-GUTI"
        );
        // And a fresh start really does fall back to SUCI.
        let mut restarted = MmOrchestrator::from_config(test_identity(), &config);
        let (identity_type, _, _) = registration_identity(&mut restarted);
        assert_eq!(identity_type, MobileIdentityType::Suci);
    }

    /// #53: NAS COUNT continuity across the restart.
    ///
    /// Asserted separately from the GUTI because it is the half that has a
    /// security consequence: a restored context whose uplink COUNT went back to
    /// zero reuses the keystream for COUNTs the network has already seen.
    #[test]
    fn a_restored_context_keeps_its_nas_counts() {
        let (config, path) = state_file_config("counts");

        let mut orch = establish_security_context();
        orch.state_file = Some(path.clone());
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);
        let uplink = *orch.security_context().uplink_count();
        let downlink = *orch.security_context().downlink_count();
        assert!(
            uplink.to_u32() > 0,
            "the fixture must have sent something, or continuity is untestable"
        );
        orch.persist_state();

        let restarted = MmOrchestrator::from_config(test_identity(), &config);
        assert_eq!(restarted.security_context().uplink_count(), &uplink);
        assert_eq!(restarted.security_context().downlink_count(), &downlink);

        UeStateSnapshot::remove(&path).unwrap();
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

    /// Opens a REGISTRATION REQUEST PDU in whichever of the three forms the UE
    /// sends it (nextgsim #258) and returns `(outer, full)`:
    /// - security header type 0: no context, plain request (`outer == full`);
    /// - type 1: an initial NAS message from 5GMM-IDLE with a valid context
    ///   (TS 24.501 §4.4.6 b)) -- the MAC is checked, the outer request carries
    ///   the cleartext IEs only, and `full` is the NAS message container
    ///   deciphered with the uplink COUNT the MAC was computed over (or the outer
    ///   request itself when it needed no container, §4.4.6 b) 3));
    /// - type 2: from 5GMM-CONNECTED, the whole message ciphered.
    fn open_registration_request(
        orch: &MmOrchestrator,
        pdu: &[u8],
    ) -> (RegistrationRequest, RegistrationRequest) {
        assert_eq!(pdu[0], 0x7E);
        let decipher = |v: &mut Vec<u8>| {
            let sec = orch.security_context();
            let count = NasCount::new(0, sec.uplink_count().sqn.wrapping_sub(1));
            nas_cipher(
                sec.ciphering_algorithm(),
                sec.keys().knas_enc().unwrap(),
                &count,
                NAS_BEARER,
                NasDirection::Uplink,
                v,
            );
        };
        match pdu[1] {
            0x00 => {
                let r = RegistrationRequest::decode(&mut &pdu[3..]).unwrap();
                (r.clone(), r)
            }
            0x01 => {
                let sec = orch.security_context();
                let count = NasCount::new(0, sec.uplink_count().sqn.wrapping_sub(1));
                assert_eq!(pdu[6], count.sqn, "sequence number");
                let mac = compute_nas_mac(
                    sec.integrity_algorithm(),
                    sec.keys().knas_int().unwrap(),
                    &count,
                    NasDirection::Uplink,
                    count.sqn,
                    &pdu[7..],
                );
                assert_eq!(&pdu[2..6], &mac, "MAC over the cleartext request");
                let outer = RegistrationRequest::decode(&mut &pdu[10..]).unwrap();
                let full = match outer.nas_message_container.as_ref() {
                    Some(c) => {
                        let mut v = c.clone();
                        decipher(&mut v);
                        RegistrationRequest::decode(&mut &v[3..]).unwrap()
                    }
                    None => outer.clone(),
                };
                (outer, full)
            }
            0x02 => {
                let mut v = pdu[7..].to_vec();
                decipher(&mut v);
                let r = RegistrationRequest::decode(&mut &v[3..]).unwrap();
                (r.clone(), r)
            }
            other => panic!("unexpected security header type {other:#04x}"),
        }
    }

    /// Opens an initial SERVICE REQUEST sent from 5GMM-IDLE (TS 24.501
    /// §4.4.6 b)): asserts security header type 1 (integrity protected, NOT
    /// ciphered) and a valid MAC, decodes the cleartext outer message, and
    /// deciphers the NAS message container, if present, with the same uplink
    /// COUNT the MAC was computed over.
    fn open_initial_service_request(
        orch: &MmOrchestrator,
        pdu: &[u8],
    ) -> (ServiceRequest, Option<ServiceRequest>) {
        assert_eq!(pdu[0], 0x7E);
        assert_eq!(
            pdu[1], 0x01,
            "an initial NAS message from 5GMM-IDLE is integrity protected, not ciphered"
        );
        let sec = orch.security_context();
        let count = NasCount::new(0, sec.uplink_count().sqn.wrapping_sub(1));
        assert_eq!(pdu[6], count.sqn, "sequence number");
        let mac = compute_nas_mac(
            sec.integrity_algorithm(),
            sec.keys().knas_int().unwrap(),
            &count,
            NasDirection::Uplink,
            count.sqn,
            &pdu[7..],
        );
        assert_eq!(&pdu[2..6], &mac, "MAC over the cleartext message");
        let outer = ServiceRequest::decode(&mut &pdu[10..]).unwrap();
        let inner = outer.nas_message_container.as_ref().map(|c| {
            let mut v = c.clone();
            nas_cipher(
                sec.ciphering_algorithm(),
                sec.keys().knas_enc().unwrap(),
                &count,
                NAS_BEARER,
                NasDirection::Uplink,
                &mut v,
            );
            ServiceRequest::decode(&mut &v[3..]).unwrap()
        });
        (outer, inner)
    }

    /// Drive a full successful registration + authentication + SMC sequence;
    /// returns the orchestrator with an active security context.
    fn establish_security_context() -> MmOrchestrator {
        establish_security_context_with(MmOrchestrator::new(test_identity()))
    }

    /// As [`establish_security_context`], for an orchestrator built with a
    /// different identity (e.g. a multi-slice requested NSSAI).
    fn establish_security_context_with(mut orch: MmOrchestrator) -> MmOrchestrator {
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

    /// The TAI the test registration accept's TAI list holds: PLMN 999/70, TAC 1.
    const TAI_IN_LIST: [u8; 6] = [0x99, 0xF9, 0x07, 0x00, 0x00, 0x01];
    /// A TAI the list does not hold: same PLMN, TAC 2.
    const TAI_NOT_IN_LIST: [u8; 6] = [0x99, 0xF9, 0x07, 0x00, 0x00, 0x02];

    /// A registered UE, camped on `serving_tai`, whose registration update of
    /// `reg_type` is in flight.
    fn registered_with_update_in_flight(
        reg_type: RegistrationType,
        serving_tai: [u8; 6],
    ) -> MmOrchestrator {
        let mut orch = establish_security_context();
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);
        assert!(orch.state().is_registered());
        orch.set_serving_tai(serving_tai);
        assert!(!orch.start_registration(reg_type).is_empty());
        assert!(orch.timers().t3510.is_running());
        orch
    }

    /// #277 review, TS 24.501 §5.2.3.2.3 d): while a failed update's retry is
    /// scheduled, a cell change re-registers at once only if the TAI changed; a
    /// re-camp in the same TA waits for the timer. With T3346 running in
    /// ATTEMPTING-REGISTRATION-UPDATE, not even a TAI change does.
    #[test]
    fn a_tai_change_bypasses_a_scheduled_update_retry_but_a_recamp_does_not() {
        let mut orch = registered_with_update_in_flight(
            RegistrationType::MobilityRegistrationUpdating,
            TAI_NOT_IN_LIST,
        );
        orch.handle_timer_expiry(TimerExpiryEvent::new(TIMER_T3510, true, 1));
        assert!(
            orch.timers().t3511.is_running(),
            "precondition: retry scheduled"
        );

        orch.set_serving_tai(TAI_NOT_IN_LIST);
        assert!(
            !orch.registration_update_on_cell_change(),
            "same TA: T3511 decides"
        );
        orch.set_serving_tai(TAI_IN_LIST);
        assert!(
            orch.registration_update_on_cell_change(),
            "a new TA: the update starts now"
        );

        // T3502 stage: the same rule.
        orch.timers.stop(TIMER_T3511, true);
        orch.timers.start(TIMER_T3502, true);
        orch.set_serving_tai(TAI_NOT_IN_LIST);
        assert!(!orch.registration_update_on_cell_change());
        orch.set_serving_tai(TAI_IN_LIST);
        assert!(orch.registration_update_on_cell_change());

        // T3346 in ATTEMPTING-REGISTRATION-UPDATE: no update on any cell change.
        orch.timers.stop(TIMER_T3502, true);
        orch.state_mut()
            .switch_mm_state(MmSubState::RegisteredAttemptingRegistrationUpdate);
        orch.timers
            .start_with_timer2(TIMER_T3346, GprsTimer2::new(60), true);
        assert!(!orch.registration_update_on_cell_change());
        orch.timers.stop(TIMER_T3346, true);
        assert!(
            orch.registration_update_on_cell_change(),
            "no timer: as before"
        );
    }

    /// #277 review, §5.2.3.2.3 a): no user data in ATTEMPTING-REGISTRATION-UPDATE,
    /// even with the user plane still up; back in NORMAL-SERVICE it flows again.
    #[test]
    fn no_user_data_in_attempting_registration_update() {
        let mut orch = establish_security_context();
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);
        orch.state_mut().switch_cm_state(CmState::Connected);
        assert!(orch.uplink_user_plane_ready(), "precondition");
        orch.state_mut()
            .switch_mm_state(MmSubState::RegisteredAttemptingRegistrationUpdate);
        assert!(!orch.uplink_user_plane_ready());
        orch.state_mut()
            .switch_mm_state(MmSubState::RegisteredNormalService);
        assert!(orch.uplink_user_plane_ready());
    }

    /// A UE that is not registered never starts an update on a cell change.
    #[test]
    fn a_deregistered_ue_starts_no_update_on_a_cell_change() {
        let orch = registered_orch();
        assert!(!orch.registration_update_on_cell_change());
    }

    /// #277 review, §5.5.1.3.7: at five failed update attempts the equivalent
    /// PLMN list is deleted, and PLMN selection is told.
    #[test]
    fn the_fifth_failed_update_deletes_the_equivalent_plmns() {
        let mut orch = registered_with_update_in_flight(
            RegistrationType::MobilityRegistrationUpdating,
            TAI_IN_LIST,
        );
        orch.equivalent_plmns = vec![[0x00, 0xF1, 0x10]];
        orch.registration_attempt_counter = MAX_REGISTRATION_ATTEMPTS - 1;
        let outs = orch.handle_timer_expiry(TimerExpiryEvent::new(TIMER_T3510, true, 1));
        assert!(orch.timers().t3502.is_running());
        assert!(orch.equivalent_plmns().is_empty());
        assert!(
            outs.contains(&MmOutput::EquivalentPlmnsUpdated(Vec::new())),
            "{outs:?}"
        );
    }

    /// nextgsim #277, TS 24.501 §5.5.1.3.7: a failed periodic update with the UE
    /// still 5U1 UPDATED and its serving TAI in the TAI list returns to
    /// 5GMM-REGISTERED.NORMAL-SERVICE, and T3511 retries the same update. Before,
    /// the UE dropped to 5GMM-DEREGISTERED.
    #[test]
    fn a_failed_periodic_update_keeps_the_ue_registered_and_retries_it() {
        let mut orch = registered_with_update_in_flight(
            RegistrationType::PeriodicRegistrationUpdating,
            TAI_IN_LIST,
        );
        orch.handle_timer_expiry(TimerExpiryEvent::new(TIMER_T3510, true, 1));

        assert!(
            orch.state().is_registered(),
            "{}",
            orch.state().mm_substate()
        );
        assert_eq!(
            orch.state().mm_substate(),
            MmSubState::RegisteredNormalService
        );
        assert_eq!(orch.state().update_status(), UpdateStatus::Updated);
        assert!(orch.timers().t3511.is_running());

        let outs = orch.handle_timer_expiry(TimerExpiryEvent::new(TIMER_T3511, true, 1));
        let pdu = first_sent_pdu(&outs);
        let (_outer, full) = open_registration_request(&orch, pdu);
        assert_eq!(
            full.registration_type.registration_type,
            RegistrationType::PeriodicRegistrationUpdating
        );
    }

    /// nextgsim #277: a failed update whose serving TAI is NOT in the TAI list
    /// goes to 5U2 NOT UPDATED and 5GMM-REGISTERED.ATTEMPTING-REGISTRATION-UPDATE,
    /// still registered. At the attempt limit T3502 runs, and its expiry retries
    /// the update from that state.
    #[test]
    fn a_failed_mobility_update_attempts_the_update_while_registered() {
        let mut orch = registered_with_update_in_flight(
            RegistrationType::MobilityRegistrationUpdating,
            TAI_NOT_IN_LIST,
        );
        for i in 1..=MAX_REGISTRATION_ATTEMPTS {
            orch.handle_timer_expiry(TimerExpiryEvent::new(TIMER_T3510, true, i));
            assert!(orch.state().is_registered(), "attempt {i}");
            assert_eq!(
                orch.state().mm_substate(),
                MmSubState::RegisteredAttemptingRegistrationUpdate
            );
            assert_eq!(orch.state().update_status(), UpdateStatus::NotUpdated);
            if i < MAX_REGISTRATION_ATTEMPTS {
                assert!(orch.timers().t3511.is_running());
                let outs = orch.handle_timer_expiry(TimerExpiryEvent::new(TIMER_T3511, true, i));
                assert!(!outs.is_empty(), "T3511 retries the update");
            }
        }
        assert_eq!(
            orch.registration_attempt_counter(),
            MAX_REGISTRATION_ATTEMPTS
        );
        assert!(orch.timers().t3502.is_running());
        assert!(!orch.timers().t3511.is_running());

        let outs = orch.handle_timer_expiry(TimerExpiryEvent::new(TIMER_T3502, true, 1));
        let pdu = first_sent_pdu(&outs);
        let (_outer, full) = open_registration_request(&orch, pdu);
        assert_eq!(
            full.registration_type.registration_type,
            RegistrationType::MobilityRegistrationUpdating
        );
    }

    /// #277: the serving TAI decides the substate, not the update's type. A
    /// mobility update (sent for another reason) that fails inside the TAI list
    /// with 5U1 keeps NORMAL-SERVICE; a periodic one outside it does not.
    #[test]
    fn the_serving_tai_decides_the_substate_after_a_failed_update() {
        let mut orch = registered_with_update_in_flight(
            RegistrationType::MobilityRegistrationUpdating,
            TAI_IN_LIST,
        );
        orch.handle_timer_expiry(TimerExpiryEvent::new(TIMER_T3510, true, 1));
        assert_eq!(
            orch.state().mm_substate(),
            MmSubState::RegisteredNormalService
        );

        let mut orch = registered_with_update_in_flight(
            RegistrationType::PeriodicRegistrationUpdating,
            TAI_NOT_IN_LIST,
        );
        orch.handle_timer_expiry(TimerExpiryEvent::new(TIMER_T3510, true, 1));
        assert_eq!(
            orch.state().mm_substate(),
            MmSubState::RegisteredAttemptingRegistrationUpdate
        );
    }

    /// #277: an initial registration that fails still follows §5.5.1.2.7, i.e.
    /// 5GMM-DEREGISTERED.ATTEMPTING-REGISTRATION.
    #[test]
    fn a_failed_initial_registration_still_deregisters() {
        let mut orch = new_orch();
        orch.start_registration(RegistrationType::InitialRegistration);
        orch.handle_timer_expiry(TimerExpiryEvent::new(TIMER_T3510, true, 1));
        assert!(!orch.state().is_registered());
        assert_eq!(
            orch.state().mm_substate(),
            MmSubState::DeregisteredAttemptingRegistration
        );
    }

    /// #277, §5.5.1.3.5 #22: a congestion reject of a registration update keeps
    /// the UE registered (ATTEMPTING-REGISTRATION-UPDATE, 5U2) under T3346, and
    /// T3346 expiry runs the update.
    #[test]
    fn a_congestion_reject_of_an_update_keeps_the_ue_registered() {
        let mut orch = registered_with_update_in_flight(
            RegistrationType::MobilityRegistrationUpdating,
            TAI_NOT_IN_LIST,
        );
        let rej = protect_downlink(
            &orch,
            &build_registration_reject_pdu(MmCause::Congestion, Some(0x05)),
            2,
        );
        orch.handle_downlink(&rej);
        assert!(orch.state().is_registered());
        assert_eq!(
            orch.state().mm_substate(),
            MmSubState::RegisteredAttemptingRegistrationUpdate
        );
        assert_eq!(orch.state().update_status(), UpdateStatus::NotUpdated);
        assert!(orch.timers().t3346.is_running());

        orch.timers.stop(TIMER_T3346, true);
        let outs = orch.handle_timer_expiry(TimerExpiryEvent::new(TIMER_T3346, true, 1));
        let pdu = first_sent_pdu(&outs);
        let (_outer, full) = open_registration_request(&orch, pdu);
        assert_eq!(
            full.registration_type.registration_type,
            RegistrationType::MobilityRegistrationUpdating
        );
    }

    /// #277, §5.2.3.2.3: in ATTEMPTING-REGISTRATION-UPDATE no data Service Request
    /// is started, but a paging response is.
    #[test]
    fn attempting_registration_update_bars_data_service_requests() {
        let mut orch = registered_with_update_in_flight(
            RegistrationType::MobilityRegistrationUpdating,
            TAI_NOT_IN_LIST,
        );
        orch.handle_timer_expiry(TimerExpiryEvent::new(TIMER_T3510, true, 1));
        assert_eq!(
            orch.state().mm_substate(),
            MmSubState::RegisteredAttemptingRegistrationUpdate
        );
        assert!(orch
            .start_service_request(ServiceType::Data, uplink_data_status(&[1]))
            .is_empty());
        assert!(!orch
            .start_service_request(ServiceType::MobileTerminatedServices, None)
            .is_empty());
    }

    /// #277: an accepted registration stops a retry timer an earlier failure left
    /// running, so it cannot re-run an update from 5GMM-REGISTERED.
    #[test]
    fn a_registration_accept_stops_t3511() {
        let mut orch = registered_with_update_in_flight(
            RegistrationType::PeriodicRegistrationUpdating,
            TAI_IN_LIST,
        );
        orch.handle_timer_expiry(TimerExpiryEvent::new(TIMER_T3510, true, 1));
        assert!(orch.timers().t3511.is_running());
        let outs = orch.handle_timer_expiry(TimerExpiryEvent::new(TIMER_T3511, true, 1));
        assert!(!outs.is_empty());
        orch.timers.start(TIMER_T3511, true);
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 2);
        orch.handle_downlink(&acc);
        assert!(!orch.timers().t3511.is_running());
        assert!(!orch.timers().t3502.is_running());
    }

    /// The 5GS TAI list parser, all three partial-list types (TS 24.501 §9.11.3.9).
    #[test]
    fn tai_list_contains_reads_every_partial_list_type() {
        let plmn = [0x99, 0xF9, 0x07];
        let tai = |tac: u32| {
            let mut t = [0u8; 6];
            t[..3].copy_from_slice(&plmn);
            t[3..].copy_from_slice(&tac.to_be_bytes()[1..]);
            t
        };
        // Type 00, two TACs (count-1 = 1): 5 and 9.
        let t00 = [0x01, 0x99, 0xF9, 0x07, 0, 0, 5, 0, 0, 9];
        assert!(tai_list_contains(&t00, &tai(5)));
        assert!(tai_list_contains(&t00, &tai(9)));
        assert!(!tai_list_contains(&t00, &tai(7)));
        // Type 01, three consecutive TACs from 0x0000FF: FF, 100, 101.
        let t01 = [0x22, 0x99, 0xF9, 0x07, 0, 0, 0xFF];
        assert!(tai_list_contains(&t01, &tai(0xFF)));
        assert!(tai_list_contains(&t01, &tai(0x101)));
        assert!(!tai_list_contains(&t01, &tai(0x102)));
        assert!(!tai_list_contains(&t01, &tai(0xFE)));
        // Type 10, two TAIs with different PLMNs; then a type 00 partial list.
        let mut t10 = vec![0x41, 0x00, 0xF1, 0x10, 0, 0, 1];
        t10.extend_from_slice(&tai(3));
        t10.extend_from_slice(&[0x00, 0x99, 0xF9, 0x07, 0, 0, 8]);
        assert!(tai_list_contains(&t10, &tai(3)));
        assert!(tai_list_contains(&t10, &tai(8)));
        assert!(
            !tai_list_contains(&t10, &tai(1)),
            "TAC 1 is in another PLMN"
        );
        // Truncated and empty input: no panic, no match.
        assert!(!tai_list_contains(&t00[..6], &tai(9)));
        assert!(!tai_list_contains(&[], &tai(1)));
        assert!(
            !tai_list_contains(&[0x60, 1, 2, 3], &tai(1)),
            "type 11 is reserved"
        );
        // The last TAC of a 32-element type 01 list at the top of the TAC range.
        let top = [0x3F, 0x99, 0xF9, 0x07, 0xFF, 0xFF, 0xE0];
        assert!(tai_list_contains(&top, &tai(0xFF_FFFF)));
    }

    #[test]
    fn test_t3510_expiry_is_abnormal_case() {
        let mut orch = new_orch();
        orch.start_registration(RegistrationType::InitialRegistration);
        let outs = orch.handle_timer_expiry(TimerExpiryEvent::new(TIMER_T3510, true, 1));
        assert_eq!(outs, vec![MmOutput::ReleaseConnectionLocally]);
        assert!(orch.timers().t3511.is_running());
        assert_eq!(orch.registration_attempt_counter(), 1);
    }

    /// nextgsim #275, TS 24.501 §5.5.1.2.7 c): T3510 expiry releases the NAS
    /// signalling connection locally, so the T3511 retry is a new initial NAS
    /// message (integrity protected, not ciphered: SHT 0x01) rather than an
    /// uplink NAS transport on the connection the AMF never answered.
    #[test]
    fn t3510_expiry_releases_the_connection_and_the_retry_is_an_initial_nas_message() {
        let mut orch = establish_security_context();
        // The RRC connection carrying the request is up (main.rs sets this on
        // RRC connection setup).
        orch.state_mut().switch_cm_state(CmState::Connected);
        assert!(orch.timers().t3510.is_running());

        let outs = orch.handle_timer_expiry(TimerExpiryEvent::new(TIMER_T3510, true, 1));
        assert!(
            outs.contains(&MmOutput::ReleaseConnectionLocally),
            "T3510 expiry must release the connection locally: {outs:?}"
        );
        assert!(orch.state().is_idle());
        assert!(orch.timers().t3511.is_running());

        let outs = orch.handle_timer_expiry(TimerExpiryEvent::new(TIMER_T3511, true, 1));
        let pdu = first_sent_pdu(&outs);
        assert_eq!(
            pdu[1], 0x01,
            "the retry is an initial NAS message: integrity protected, not ciphered"
        );
        let (_outer, full) = open_registration_request(&orch, pdu);
        assert_eq!(
            full.registration_type.registration_type,
            RegistrationType::InitialRegistration
        );
    }

    /// The release is skipped for an emergency registration (§5.5.1.2.7 c).
    #[test]
    fn t3510_expiry_keeps_the_connection_for_an_emergency_registration() {
        let mut orch = new_orch();
        orch.start_registration(RegistrationType::EmergencyRegistration);
        orch.state_mut().switch_cm_state(CmState::Connected);
        let outs = orch.handle_timer_expiry(TimerExpiryEvent::new(TIMER_T3510, true, 1));
        assert!(!outs.contains(&MmOutput::ReleaseConnectionLocally));
        assert!(!orch.state().is_idle());
    }

    #[test]
    fn test_t3512_expiry_triggers_periodic_registration_with_guti() {
        let mut orch = establish_security_context();
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);
        assert!(orch.timers().t3512.is_running());

        let outs = orch.handle_timer_expiry(TimerExpiryEvent::new(TIMER_T3512, true, 1));
        let pdu = first_sent_pdu(&outs).to_vec();
        // A periodic registration is sent from 5GMM-IDLE with the active
        // context: an initial NAS message, so integrity protected but NOT
        // ciphered (TS 24.501 §4.4.6 b), §4.4.5; nextgsim #258). This assertion
        // used to pin header type 2, which hid the 5G-GUTI from the AMF.
        assert_eq!(pdu[1], 0x01);
        let (outer, full) = open_registration_request(&orch, &pdu);
        assert_eq!(
            outer.mobile_identity.identity_type,
            MobileIdentityType::Guti,
            "the 5G-GUTI is cleartext, so the AMF can find the context"
        );
        assert_eq!(
            full.registration_type.registration_type,
            RegistrationType::PeriodicRegistrationUpdating
        );
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

    /// #252: a paged UE's SERVICE REQUEST from 5GMM-IDLE, byte for byte.
    ///
    /// Hand-derived from TS 24.501: security header type 1 (§4.4.5 -- integrity
    /// protected, not ciphered, because it is an initial NAS message) around
    /// the plain message
    ///   7E          extended protocol discriminator (5GMM, 9.2)
    ///   00          security header type 0 + spare half octet
    ///   4C          SERVICE REQUEST (9.7)
    ///   ..          ngKSI 0 (native) and service type 2 = "mobile
    ///               terminated services" (9.11.3.50) -- nibble order below
    ///   00 07       5GS mobile identity length (LV-E, 9.11.3.4)
    ///   F4          1111 | even | type 100 = 5G-S-TMSI
    ///   00 40       AMF set ID (10 bits) + AMF pointer (6 bits) from the GUTI
    ///   DE AD BE EF 5G-TMSI
    /// with no NAS message container, since an MT request with no pending
    /// uplink data has no non-cleartext IE (§4.4.6 b) 3)).
    #[test]
    fn a_paged_service_request_matches_hand_derived_bytes() {
        let mut orch = establish_security_context();
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);
        orch.state_mut().switch_cm_state(CmState::Idle);

        let outs = orch.start_service_request(ServiceType::MobileTerminatedServices, None);
        let pdu = first_sent_pdu(&outs).to_vec();
        let (req, _) = open_initial_service_request(&orch, &pdu); // header type 1 + MAC
        assert_eq!(&pdu[7..10], &[0x7E, 0x00, 0x4C]);
        // Octet 4: ngKSI (native, 0) in bits 1-4 and service type "mobile
        // terminated services" (0010) in bits 5-8 -- TS 24.501 §8.1 b),
        // Table 8.2.16.1.1 (nextgsim #257 / nextgcore #537).
        assert_eq!(pdu[10], 0x20);
        assert_eq!(
            req.service_type.service_type,
            ServiceType::MobileTerminatedServices
        );
        assert_eq!(
            &pdu[11..],
            &[0x00, 0x07, 0xF4, 0x00, 0x40, 0xDE, 0xAD, 0xBE, 0xEF]
        );
    }

    /// nextgsim #258: a mobility registration update sent from 5GMM-IDLE with a
    /// valid context is an initial NAS message (TS 24.501 §4.4.6 b)): integrity
    /// protected but not ciphered, the cleartext IEs only outside, and the full
    /// request -- here carrying the non-cleartext Requested NSSAI and Last
    /// visited TAI -- inside the ciphered NAS message container.
    ///
    /// The outer octets are derived by hand from Table 8.2.6.1.1 and §8.1 b):
    /// `7E 00 41` (EPD, plain header, REGISTRATION REQUEST); `02` = 5GS
    /// registration type "mobility registration updating" (FOR 0) in bits 1-4,
    /// the first half-octet IE, and ngKSI native/0 in bits 5-8; the 5G-GUTI as
    /// LV-E `00 0B F2 99 F9 07 01 00 40 DE AD BE EF`; UE security capability
    /// `2E 02 F0 70`; then the NAS message container `71 LL LL ...`.
    #[test]
    fn an_idle_mode_registration_is_integrity_protected_with_a_ciphered_container() {
        let mut orch = establish_security_context();
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);
        orch.state_mut().switch_cm_state(CmState::Idle);

        let outs = orch.start_registration(RegistrationType::MobilityRegistrationUpdating);
        let pdu = first_sent_pdu(&outs).to_vec();
        assert_eq!(
            pdu[1], 0x01,
            "header type 1: integrity protected, not ciphered"
        );
        let (outer, full) = open_registration_request(&orch, &pdu); // checks the MAC

        let head = [
            0x7E, 0x00, 0x41, 0x02, 0x00, 0x0B, 0xF2, 0x99, 0xF9, 0x07, 0x01, 0x00, 0x40, 0xDE,
            0xAD, 0xBE, 0xEF, 0x2E, 0x02, 0xF0, 0x70, 0x71,
        ];
        assert_eq!(&pdu[7..7 + head.len()], &head);
        assert!(
            outer.requested_nssai.is_none() && outer.last_visited_tai.is_none(),
            "non-cleartext IEs must not appear outside the container"
        );
        let container = outer.nas_message_container.as_ref().expect("container");
        let len = u16::from_be_bytes([pdu[7 + head.len()], pdu[8 + head.len()]]) as usize;
        assert_eq!(len, container.len());
        assert_eq!(
            pdu.len(),
            7 + head.len() + 2 + len,
            "the container is the last IE"
        );
        assert_eq!(
            full.registration_type.registration_type,
            RegistrationType::MobilityRegistrationUpdating
        );
        assert!(
            full.requested_nssai.is_some(),
            "the full request is in the container"
        );
        assert_eq!(full.mobile_identity.identity_type, MobileIdentityType::Guti);
        // The container is ciphered: the plaintext full request must not be
        // readable in the outer PDU.
        let mut plain_full = Vec::new();
        full.encode(&mut plain_full);
        assert_ne!(container, &plain_full);
    }

    /// From 5GMM-CONNECTED a REGISTRATION REQUEST rides the existing NAS
    /// signalling connection, so it is ciphered as usual (TS 24.501 §4.4.5):
    /// only the idle case changed for #258.
    #[test]
    fn a_registration_from_connected_mode_is_ciphered() {
        let mut orch = establish_security_context();
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);
        orch.state_mut().switch_cm_state(CmState::Connected);
        let outs = orch.start_registration(RegistrationType::MobilityRegistrationUpdating);
        let pdu = first_sent_pdu(&outs).to_vec();
        assert_eq!(pdu[1], 0x02);
        let (_, full) = open_registration_request(&orch, &pdu);
        assert!(full.requested_nssai.is_some());
    }

    /// From 5GMM-CONNECTED a SERVICE REQUEST rides an existing NAS signalling
    /// connection, so it is not an initial NAS message and is ciphered as
    /// usual (TS 24.501 §4.4.5): only the idle case changed for #252.
    #[test]
    fn a_service_request_from_connected_mode_is_ciphered() {
        let mut orch = establish_security_context();
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);
        orch.state_mut().switch_cm_state(CmState::Connected);
        let outs = orch.start_service_request(ServiceType::Data, uplink_data_status(&[1]));
        assert_eq!(first_sent_pdu(&outs)[1], 0x02);
    }

    #[test]
    fn test_service_request_uses_stored_tmsi() {
        let mut orch = establish_security_context();
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);
        // Simulate CM-IDLE
        orch.state_mut().switch_cm_state(CmState::Idle);

        // PSI 3, so the expected IE value (0x0008) cannot coincide with the
        // 0x2000 (PSI 13) constant this path used to hardcode, nor with the
        // PSI-1 case the dedicated bitmap tests below cover.
        let outs = orch.start_service_request(ServiceType::Data, uplink_data_status(&[3]));
        let pdu = first_sent_pdu(&outs).to_vec();
        assert!(orch.timers().t3517.is_running());
        assert_eq!(
            orch.state().mm_substate(),
            MmSubState::ServiceRequestInitiated
        );

        // TS 24.501 §4.4.6 b) 1): the 5G-S-TMSI is a cleartext IE, readable
        // without the ciphering key, so the AMF can find the context; the
        // Uplink data status is not, so it rides only in the ciphered container.
        let (outer, inner) = open_initial_service_request(&orch, &pdu);
        // GUTI TMSI bytes were 0xDEADBEEF; the 5G-S-TMSI carries set/ptr + TMSI
        assert_eq!(&outer.tmsi.data[3..7], &[0xDE, 0xAD, 0xBE, 0xEF]);
        assert_eq!(
            outer.uplink_data_status, None,
            "a non-cleartext IE must not be sent in the clear"
        );
        let inner = inner.expect("non-cleartext IEs travel in the NAS message container");
        assert_eq!(&inner.tmsi.data[3..7], &[0xDE, 0xAD, 0xBE, 0xEF]);
        assert_eq!(inner.uplink_data_status, Some(0x0008), "PSI 3 is bit 3");
        assert_eq!(inner.nas_message_container, None);
    }

    // ========================================================================
    // Generic UE configuration update + slice-selection state (#19)
    // ========================================================================

    /// A configured requested NSSAI with TWO slices: SST 1 without an SD and
    /// SST 2 with SD AABBCC. Two entries so a filtering test can assert that the
    /// rejected one is dropped AND the other survives — a single-slice fixture
    /// passes against an implementation that drops everything.
    const TWO_SLICE_REQUESTED_NSSAI: [u8; 7] = [0x01, 0x01, 0x04, 0x02, 0xAA, 0xBB, 0xCC];

    /// A registered UE with a two-slice configured requested NSSAI and an active
    /// security context, so the network messages under test (REGISTRATION
    /// ACCEPT, REGISTRATION REJECT) can be integrity protected — an
    /// unprotected one is discarded per TS 24.501 §4.4.4.2, which would make
    /// every assertion below vacuous.
    fn orch_with_two_slices() -> MmOrchestrator {
        let mut identity = test_identity();
        identity.requested_nssai = Some(TWO_SLICE_REQUESTED_NSSAI.to_vec());
        establish_security_context_with(MmOrchestrator::new(identity))
    }

    /// The requested NSSAI the next REGISTRATION REQUEST carries.
    ///
    /// Read from the stashed FULL request rather than the emitted PDU: the
    /// requested NSSAI is not a cleartext IE (TS 24.501 §4.4.6), so the plain
    /// wire copy of an unprotected request omits it by design.
    fn requested_nssai_in_the_next_request(orch: &mut MmOrchestrator) -> Option<Vec<u8>> {
        orch.start_registration(RegistrationType::InitialRegistration);
        let full = orch
            .last_registration_request
            .as_ref()
            .expect("a registration request must have been built");
        RegistrationRequest::decode(&mut &full[3..])
            .unwrap()
            .requested_nssai
    }

    /// Runs the real procedure over a command, so the tests exercise the
    /// message → result mapping rather than a hand-built result.
    fn config_update_result(cmd: ConfigurationUpdateCommand) -> ConfigUpdateResult {
        ConfigUpdateProcedure::process_command(&cmd)
    }

    fn command_with_guti(guti_data: Vec<u8>) -> ConfigurationUpdateCommand {
        let mut cmd = ConfigurationUpdateCommand::new();
        cmd.guti = Some(Ie5gsMobileIdentity::new(
            MobileIdentityType::Guti,
            guti_data,
        ));
        cmd
    }

    /// The 5G-GUTI a Configuration Update Command reallocates, distinct from the
    /// Registration Accept fixture's GUTI in every octet that matters.
    const CONFIG_UPDATE_GUTI: [u8; 11] = [
        0xF2, 0x99, 0xF9, 0x07, 0x02, 0x00, 0x80, 0xC0, 0xFF, 0xEE, 0x01,
    ];

    #[test]
    fn a_config_update_guti_is_stored_and_used_by_the_next_registration_request() {
        let mut orch = new_orch();
        assert!(orch.stored_guti().is_none(), "precondition: no GUTI yet");

        orch.apply_config_update(&config_update_result(command_with_guti(
            CONFIG_UPDATE_GUTI.to_vec(),
        )));

        assert_eq!(
            orch.stored_guti().map(|g| g.data.clone()),
            Some(CONFIG_UPDATE_GUTI.to_vec()),
            "the reallocated 5G-GUTI must be stored, not logged and dropped"
        );
        let (identity_type, _, _) = registration_identity(&mut orch);
        assert_eq!(
            identity_type,
            MobileIdentityType::Guti,
            "the next Registration Request must identify the UE by the new GUTI"
        );
    }

    #[test]
    fn a_config_update_t3512_restarts_the_periodic_registration_timer() {
        let mut orch = new_orch();
        let mut cmd = ConfigurationUpdateCommand::new();
        // Unit 000 (10 minutes), value 3 → 1800 s. Not a value any default
        // could coincide with.
        cmd.t3512_value_secs = Some(decode_config_update_timer(0b000_00011));
        orch.apply_config_update(&config_update_result(cmd));

        assert!(orch.timers().t3512.is_running());
        assert_eq!(orch.timers().t3512.interval(), 1800);
    }

    #[test]
    fn a_config_update_deactivating_t3512_stops_the_timer() {
        let mut orch = new_orch();
        let mut start = ConfigurationUpdateCommand::new();
        start.t3512_value_secs = Some(1800);
        orch.apply_config_update(&config_update_result(start));
        assert!(orch.timers().t3512.is_running());

        let mut stop = ConfigurationUpdateCommand::new();
        stop.t3512_value_secs = Some(0); // unit 111: deactivated
        orch.apply_config_update(&config_update_result(stop));

        assert!(
            !orch.timers().t3512.is_running(),
            "a deactivated T3512 must stop the timer, not restart it at 0s"
        );
    }

    /// Same decoder the message path uses, so the fixture cannot drift from it.
    fn decode_config_update_timer(byte: u8) -> u32 {
        GprsTimer3::from_byte(byte).to_seconds()
    }

    #[test]
    fn a_config_update_tai_list_updates_the_registration_area() {
        let mut orch = new_orch();
        let mut cmd = ConfigurationUpdateCommand::new();
        // TAI list: type 00, one entry, PLMN 999/70, TAC 0x000007
        cmd.tai_list = Some(vec![0x00, 0x99, 0xF9, 0x07, 0x00, 0x00, 0x07]);
        orch.apply_config_update(&config_update_result(cmd));

        assert_eq!(
            orch.current_tai,
            Some([0x99, 0xF9, 0x07, 0x00, 0x00, 0x07]),
            "the new registration area must be recorded for the Last visited TAI"
        );
    }

    #[test]
    fn a_config_update_allowed_nssai_replaces_the_stored_one() {
        let mut orch = new_orch();
        let mut cmd = ConfigurationUpdateCommand::new();
        cmd.allowed_nssai = Some(vec![0x04, 0x02, 0xAA, 0xBB, 0xCC]);
        orch.apply_config_update(&config_update_result(cmd));

        assert_eq!(
            orch.allowed_nssai(),
            Some(&vec![0x04, 0x02, 0xAA, 0xBB, 0xCC][..])
        );
    }

    #[test]
    fn a_rejected_nssai_ie_parses_every_entry_with_its_cause() {
        // SST 1 rejected for the PLMN, SST 2/AABBCC for the registration area,
        // SST 3 for a failed NSSAA.
        let value = [0x10, 0x01, 0x41, 0x02, 0xAA, 0xBB, 0xCC, 0x12, 0x03];
        assert_eq!(
            parse_rejected_nssai(&value),
            vec![
                RejectedSNssai {
                    cause: RejectedSNssaiCause::NotAvailableInPlmn,
                    sst: 1,
                    sd: None,
                },
                RejectedSNssai {
                    cause: RejectedSNssaiCause::NotAvailableInRegistrationArea,
                    sst: 2,
                    sd: Some([0xAA, 0xBB, 0xCC]),
                },
                RejectedSNssai {
                    cause: RejectedSNssaiCause::NssaaFailedOrRevoked,
                    sst: 3,
                    sd: None,
                },
            ]
        );
    }

    /// A truncated trailing entry is DROPPED, not half-read: an S-NSSAI read
    /// short is a different slice, and excluding the wrong one would suppress a
    /// slice the network allows.
    #[test]
    fn a_truncated_rejected_nssai_entry_is_dropped_rather_than_half_read() {
        // First entry complete (SST 1); second claims an SD but supplies 2 of 3
        // octets.
        let value = [0x10, 0x01, 0x40, 0x02, 0xAA, 0xBB];
        assert_eq!(
            parse_rejected_nssai(&value),
            vec![RejectedSNssai {
                cause: RejectedSNssaiCause::NotAvailableInPlmn,
                sst: 1,
                sd: None,
            }]
        );
    }

    #[test]
    fn a_rejected_s_nssai_is_omitted_from_the_next_requested_nssai() {
        let mut orch = orch_with_two_slices();
        // Reject SST 1 (no SD) for the current PLMN: entry length 1, cause 0000.
        let acc = accept_with_rejected_nssai(&orch, &[0x10, 0x01]);
        orch.handle_downlink(&acc);

        assert_eq!(
            orch.rejected_nssai(),
            &[RejectedSNssai {
                cause: RejectedSNssaiCause::NotAvailableInPlmn,
                sst: 1,
                sd: None,
            }]
        );
        assert_eq!(
            requested_nssai_in_the_next_request(&mut orch),
            Some(vec![0x04, 0x02, 0xAA, 0xBB, 0xCC]),
            "the rejected SST 1 must be gone and the other slice must remain"
        );
    }

    /// An S-NSSAI with an SD is a DIFFERENT slice from the same SST without one,
    /// so rejecting one must not exclude the other.
    #[test]
    fn a_rejection_matches_on_the_whole_s_nssai_not_just_the_sst() {
        let mut orch = orch_with_two_slices();
        // Reject SST 2 with a DIFFERENT SD (112233), not the configured AABBCC.
        let acc = accept_with_rejected_nssai(&orch, &[0x40, 0x02, 0x11, 0x22, 0x33]);
        orch.handle_downlink(&acc);

        assert_eq!(
            requested_nssai_in_the_next_request(&mut orch),
            Some(TWO_SLICE_REQUESTED_NSSAI.to_vec()),
            "a rejection for another SD must leave both configured slices requested"
        );
    }

    #[test]
    fn every_configured_slice_rejected_omits_the_requested_nssai_ie() {
        let mut orch = orch_with_two_slices();
        let acc = accept_with_rejected_nssai(&orch, &[0x10, 0x01, 0x40, 0x02, 0xAA, 0xBB, 0xCC]);
        orch.handle_downlink(&acc);

        assert_eq!(
            requested_nssai_in_the_next_request(&mut orch),
            None,
            "an empty requested NSSAI would ask for the slice set just refused"
        );
    }

    /// TS 24.501 §4.6.2.2 a) 3): a slice the network now allows is no longer
    /// rejected. Without this a stale rejection suppresses the slice forever.
    #[test]
    fn a_newly_allowed_s_nssai_leaves_the_rejected_set() {
        let mut orch = orch_with_two_slices();
        let acc = accept_with_rejected_nssai(&orch, &[0x10, 0x01]);
        orch.handle_downlink(&acc);
        assert_eq!(orch.rejected_nssai().len(), 1);

        // A later Accept allows SST 1 again.
        let mut acc = RegistrationAccept::new(Ie5gsRegistrationResult::new(
            SmsOverNasAllowed::NotAllowed,
            RegistrationResultValue::ThreeGppAccess,
        ));
        acc.allowed_nssai = Some(vec![0x01, 0x01]);
        let mut plain = Vec::new();
        acc.encode(&mut plain);
        let pdu = protect_downlink(&orch, &plain, 2);
        orch.handle_downlink(&pdu);

        assert!(
            orch.rejected_nssai().is_empty(),
            "the allowed NSSAI must clear the matching rejection"
        );
        assert_eq!(
            requested_nssai_in_the_next_request(&mut orch),
            Some(TWO_SLICE_REQUESTED_NSSAI.to_vec())
        );
    }

    /// Registration-area rejections are scoped to the area (§4.6.2.2); PLMN ones
    /// are not. A move must drop exactly the first kind.
    #[test]
    fn a_registration_area_change_drops_only_the_area_scoped_rejections() {
        let mut orch = orch_with_two_slices();
        // SST 1 rejected for the registration area (cause 0001), SST 2/AABBCC
        // rejected for the PLMN (cause 0000).
        let acc = accept_with_rejected_nssai(&orch, &[0x11, 0x01, 0x40, 0x02, 0xAA, 0xBB, 0xCC]);
        orch.handle_downlink(&acc);
        assert_eq!(orch.rejected_nssai().len(), 2);

        // A second Accept with a different TAI list: the UE moved.
        let mut acc = RegistrationAccept::new(Ie5gsRegistrationResult::new(
            SmsOverNasAllowed::NotAllowed,
            RegistrationResultValue::ThreeGppAccess,
        ));
        acc.tai_list = Some(vec![0x00, 0x99, 0xF9, 0x07, 0x00, 0x00, 0x09]);
        let mut plain = Vec::new();
        acc.encode(&mut plain);
        let pdu = protect_downlink(&orch, &plain, 2);
        orch.handle_downlink(&pdu);

        assert_eq!(
            orch.rejected_nssai(),
            &[RejectedSNssai {
                cause: RejectedSNssaiCause::NotAvailableInPlmn,
                sst: 2,
                sd: Some([0xAA, 0xBB, 0xCC]),
            }],
            "the PLMN-scoped rejection survives a move; the area-scoped one does not"
        );
    }

    #[test]
    fn a_rejected_nssai_in_a_registration_reject_is_stored() {
        let mut orch = orch_with_two_slices();
        let mut rej = RegistrationReject::new(Ie5gMmCause::new(MmCause::Congestion));
        rej.rejected_nssai = Some(vec![0x10, 0x01]);
        let mut plain = Vec::new();
        rej.encode(&mut plain);
        let pdu = protect_downlink(&orch, &plain, 1);
        orch.handle_downlink(&pdu);

        assert_eq!(
            orch.rejected_nssai().len(),
            1,
            "the registration that failed is the one whose NSSAI must not be repeated"
        );
    }

    #[test]
    fn a_pending_nssai_is_stored_from_a_registration_accept() {
        let mut orch = establish_security_context();
        let mut acc = RegistrationAccept::new(Ie5gsRegistrationResult::new(
            SmsOverNasAllowed::NotAllowed,
            RegistrationResultValue::ThreeGppAccess,
        ));
        acc.pending_nssai = Some(vec![0x01, 0x03]);
        let mut plain = Vec::new();
        acc.encode(&mut plain);
        let pdu = protect_downlink(&orch, &plain, 1);
        orch.handle_downlink(&pdu);

        assert_eq!(orch.pending_nssai(), Some(&[0x01, 0x03][..]));
    }

    /// An integrity-protected REGISTRATION ACCEPT carrying a rejected NSSAI.
    /// Protected because an unprotected Accept is discarded by TS 24.501
    /// §4.4.4.2, which would make every assertion that follows vacuous.
    fn accept_with_rejected_nssai(orch: &MmOrchestrator, rejected: &[u8]) -> Vec<u8> {
        let mut acc = RegistrationAccept::new(Ie5gsRegistrationResult::new(
            SmsOverNasAllowed::NotAllowed,
            RegistrationResultValue::ThreeGppAccess,
        ));
        acc.rejected_nssai = Some(rejected.to_vec());
        let mut plain = Vec::new();
        acc.encode(&mut plain);
        protect_downlink(orch, &plain, 1)
    }

    // ---- RACS ------------------------------------------------------------

    fn config_with_racs(store: bool) -> UeConfig {
        UeConfig {
            racs_store_assigned_id: store,
            ..UeConfig::default()
        }
    }

    fn command_with_racs_id(id: &str, octets: Vec<u8>) -> ConfigurationUpdateCommand {
        let mut cmd = ConfigurationUpdateCommand::new();
        let _ = octets;
        cmd.ue_radio_capability_id = Some(id.to_string());
        cmd
    }

    #[test]
    fn an_assigned_ue_radio_capability_id_is_kept_when_configured() {
        let mut orch = MmOrchestrator::from_config(test_identity(), &config_with_racs(true));
        orch.apply_config_update(&config_update_result(command_with_racs_id(
            "123456",
            vec![],
        )));
        assert_eq!(orch.racs_id(), Some("123456"));
    }

    /// Kept by default since #101: the ID has a consumer now, so the shipped
    /// default stores it.
    #[test]
    fn an_assigned_ue_radio_capability_id_is_kept_by_default() {
        let mut orch = MmOrchestrator::from_config(test_identity(), &UeConfig::default());
        orch.apply_config_update(&config_update_result(command_with_racs_id(
            "123456",
            vec![],
        )));
        assert_eq!(
            orch.racs_id(),
            Some("123456"),
            "racs_store_assigned_id defaults to true"
        );
    }

    /// With the switch off the ID is decoded (so the command is still processed
    /// and acknowledged) and discarded.
    #[test]
    fn an_assigned_ue_radio_capability_id_is_discarded_when_the_switch_is_off() {
        let mut orch = MmOrchestrator::from_config(test_identity(), &config_with_racs(false));
        let result = config_update_result(command_with_racs_id("123456", vec![]));
        orch.apply_config_update(&result);
        assert_eq!(orch.racs_id(), None);
        assert_eq!(
            result.new_racs_id.as_deref(),
            Some("123456"),
            "the IE is still decoded and reported to the caller"
        );
    }

    /// The point of #101: a stored ID is presented back to the network in the UE
    /// radio capability ID IE, with the low-nibble-first packing of TS 24.501
    /// §9.11.3.68 asserted **on the wire octets**.
    ///
    /// "123456" packs as 0x21 0x43 0x65: digit 1 in the low nibble of the first
    /// octet, digit 2 in its high nibble, and so on.
    #[test]
    fn a_stored_ue_radio_capability_id_is_signalled_in_the_registration_request() {
        let mut orch = MmOrchestrator::from_config(test_identity(), &UeConfig::default());
        orch.apply_config_update(&config_update_result(command_with_racs_id(
            "123456",
            vec![],
        )));
        assert_eq!(orch.racs_id(), Some("123456"));

        orch.start_registration(RegistrationType::InitialRegistration);
        let full = orch
            .last_registration_request
            .as_ref()
            .expect("a registration request must have been built");

        // The IE on the wire: IEI 0x67, length 3, then the packed digits.
        let ie = [0x67u8, 0x03, 0x21, 0x43, 0x65];
        assert!(
            full.windows(ie.len()).any(|w| w == ie),
            "the full request must carry the UE radio capability ID IE as \
             {ie:02X?}; got {full:02X?}"
        );

        // And it decodes back to the same identifier.
        assert_eq!(
            RegistrationRequest::decode(&mut &full[3..])
                .expect("decode")
                .ue_radio_capability_id
                .as_deref(),
            Some("123456")
        );
    }

    /// TS 24.501 §9.11.3.1: the full request carries a 5GMM capability IE with
    /// the RACS bit (octet 4, bit 8) -- `10 02 00 80` -- and the plain cleartext
    /// request does not (§4.4.6), and a UE with RACS switched off sets no bit.
    #[test]
    fn the_registration_request_advertises_racs_in_the_5gmm_capability() {
        let mut orch = MmOrchestrator::from_config(test_identity(), &UeConfig::default());
        let outputs = orch.start_registration(RegistrationType::InitialRegistration);
        let full = orch
            .last_registration_request
            .clone()
            .expect("full request");
        let ie = [0x10u8, 0x02, 0x00, 0x80];
        assert!(full.windows(4).any(|w| w == ie), "got {full:02X?}");
        let plain = first_sent_pdu(&outputs);
        assert!(
            !plain.windows(4).any(|w| w == ie),
            "non-cleartext IE leaked: {plain:02X?}"
        );

        let mut off = MmOrchestrator::from_config(test_identity(), &config_with_racs(false));
        off.start_registration(RegistrationType::InitialRegistration);
        let full = off.last_registration_request.clone().expect("full request");
        assert_eq!(
            RegistrationRequest::decode(&mut &full[3..])
                .unwrap()
                .five_gmm_capability,
            None
        );
    }

    fn racs_accept(id: Option<&str>, deletion: Option<u8>) -> RegistrationAccept {
        let mut acc = RegistrationAccept::new(Ie5gsRegistrationResult::new(
            SmsOverNasAllowed::NotAllowed,
            RegistrationResultValue::ThreeGppAccess,
        ));
        acc.tai_list = Some(vec![0x00, 0x99, 0xF9, 0x07, 0x00, 0x00, 0x01]);
        acc.allowed_nssai = Some(vec![0x01, 0x01]);
        acc.ue_radio_capability_id = id.map(str::to_string);
        acc.ue_radio_capability_id_deletion = deletion;
        acc
    }

    /// The two RACS IEs on the wire (Table 8.2.7.1.1): `67 03 21 43 65` and the
    /// Type 1 `E1`, and back.
    #[test]
    fn the_registration_accept_racs_ies_round_trip() {
        let mut plain = Vec::new();
        racs_accept(Some("123456"), Some(1)).encode(&mut plain);
        assert!(plain
            .windows(5)
            .any(|w| w == [0x67, 0x03, 0x21, 0x43, 0x65]));
        assert!(plain.contains(&0xE1));
        let decoded = RegistrationAccept::decode(&mut &plain[3..]).unwrap();
        assert_eq!(decoded.ue_radio_capability_id.as_deref(), Some("123456"));
        assert_eq!(decoded.ue_radio_capability_id_deletion, Some(1));
    }

    /// §5.5.1.2.4 b) and the T3550 rule: an assigned ID is stored and acknowledged
    /// with REGISTRATION COMPLETE even with no new 5G-GUTI.
    #[test]
    fn a_registration_accept_assigning_an_id_stores_it_and_completes() {
        let mut orch = establish_security_context();
        orch.racs_store_assigned_id = true;
        let mut plain = Vec::new();
        racs_accept(Some("abcde1"), None).encode(&mut plain);
        let outs = orch.handle_downlink(&protect_downlink(&orch, &plain, 1));
        assert_eq!(orch.racs_id(), Some("abcde1"));
        assert!(
            outs.iter().any(|o| matches!(o, MmOutput::SendNasPdu(_))),
            "the ID must be acknowledged with Registration Complete"
        );
    }

    /// §5.5.1.2.4 a): the deletion indication deletes the network-assigned ID,
    /// is acknowledged, and is followed by a mobility registration update.
    #[test]
    fn a_registration_accept_deleting_the_id_completes_then_re_registers() {
        let mut orch = establish_security_context();
        orch.racs_store_assigned_id = true;
        orch.racs_id = Some("123456".to_string());
        let mut plain = Vec::new();
        racs_accept(None, Some(1)).encode(&mut plain);
        let outs = orch.handle_downlink(&protect_downlink(&orch, &plain, 1));
        assert_eq!(orch.racs_id(), None);
        let sent = outs
            .iter()
            .filter(|o| matches!(o, MmOutput::SendNasPdu(_)))
            .count();
        assert!(
            sent >= 2,
            "Registration Complete and a new Registration Request: {outs:?}"
        );
        let req = orch
            .last_registration_request
            .clone()
            .expect("re-registration");
        let decoded = RegistrationRequest::decode(&mut &req[3..]).unwrap();
        assert_eq!(
            decoded.registration_type.registration_type,
            RegistrationType::MobilityRegistrationUpdating
        );
        assert_eq!(
            decoded.ue_radio_capability_id, None,
            "the deleted ID is not presented"
        );
    }

    /// §4.4.6: the ID is a NON-cleartext IE, so it must be absent from the plain
    /// unprotected initial REGISTRATION REQUEST and present in the full request
    /// replayed inside the SECURITY MODE COMPLETE container. Both halves asserted,
    /// because either alone passes against a wrong implementation.
    #[test]
    fn the_ue_radio_capability_id_is_not_a_cleartext_ie() {
        let mut orch = MmOrchestrator::from_config(test_identity(), &UeConfig::default());
        orch.apply_config_update(&config_update_result(command_with_racs_id(
            "123456",
            vec![],
        )));

        let outputs = orch.start_registration(RegistrationType::InitialRegistration);

        // The emitted PDU is the cleartext-only request (no security context).
        let wire = outputs
            .iter()
            .find_map(|out| match out {
                MmOutput::SendNasPdu(pdu) => Some(pdu.clone()),
                _ => None,
            })
            .expect("a NAS PDU is emitted");
        assert!(
            RegistrationRequest::decode(&mut &wire[3..])
                .expect("decode")
                .ue_radio_capability_id
                .is_none(),
            "the UE radio capability ID must not travel in the clear"
        );
        assert!(
            !wire.contains(&0x67u8) || {
                // Belt and braces: the IEI byte may occur inside another IE's
                // value, so also check no 0x67 starts a well-formed RACS IE.
                !wire.windows(5).any(|w| w == [0x67, 0x03, 0x21, 0x43, 0x65])
            },
            "no RACS IE may appear in the cleartext request"
        );

        // The stashed FULL request does carry it.
        let full = orch
            .last_registration_request
            .as_ref()
            .expect("the full request is stashed for the SMC container");
        assert_eq!(
            RegistrationRequest::decode(&mut &full[3..])
                .expect("decode")
                .ue_radio_capability_id
                .as_deref(),
            Some("123456"),
            "the full request replayed in SECURITY MODE COMPLETE carries it"
        );
    }

    /// With no ID stored, no IE is emitted — a UE that has never been assigned one
    /// must not present an empty or placeholder identifier.
    #[test]
    fn no_stored_id_means_no_ue_radio_capability_id_ie() {
        let mut orch = MmOrchestrator::from_config(test_identity(), &UeConfig::default());
        assert_eq!(orch.racs_id(), None);

        orch.start_registration(RegistrationType::InitialRegistration);
        let full = orch.last_registration_request.as_ref().expect("built");

        assert!(RegistrationRequest::decode(&mut &full[3..])
            .expect("decode")
            .ue_radio_capability_id
            .is_none());
    }

    /// With the switch off nothing is stored, so nothing is signalled either —
    /// the pre-#101 wire, byte for byte.
    #[test]
    fn the_switch_off_signals_no_ue_radio_capability_id() {
        let mut orch = MmOrchestrator::from_config(test_identity(), &config_with_racs(false));
        orch.apply_config_update(&config_update_result(command_with_racs_id(
            "123456",
            vec![],
        )));

        orch.start_registration(RegistrationType::InitialRegistration);
        let full = orch.last_registration_request.as_ref().expect("built");

        assert!(RegistrationRequest::decode(&mut &full[3..])
            .expect("decode")
            .ue_radio_capability_id
            .is_none());
    }

    /// A deletion indication stops the ID being signalled, not just stored.
    ///
    /// Two orchestrators rather than one registering twice: the second
    /// `start_registration` on one orchestrator is a no-op while T3510 runs, so
    /// the stashed request would be the first one and the test would pass or fail
    /// for the wrong reason. Both are driven from the same assignment so the only
    /// difference is the deletion.
    #[test]
    fn a_deleted_ue_radio_capability_id_is_no_longer_signalled() {
        let signalled_id = |delete: bool| {
            let mut orch = MmOrchestrator::from_config(test_identity(), &UeConfig::default());
            orch.apply_config_update(&config_update_result(command_with_racs_id(
                "123456",
                vec![],
            )));
            if delete {
                let mut deletion = ConfigurationUpdateCommand::new();
                deletion.ue_radio_capability_id_deletion =
                    Some(RacsDeletionRequest::NetworkAssigned);
                orch.apply_config_update(&config_update_result(deletion));
                assert_eq!(orch.racs_id(), None, "the deletion took effect");
            }
            orch.start_registration(RegistrationType::InitialRegistration);
            RegistrationRequest::decode(
                &mut &orch.last_registration_request.as_ref().expect("built")[3..],
            )
            .expect("decode")
            .ue_radio_capability_id
        };

        assert_eq!(
            signalled_id(false).as_deref(),
            Some("123456"),
            "without the deletion the ID is presented"
        );
        assert!(
            signalled_id(true).is_none(),
            "a deleted ID must not keep being presented"
        );
    }

    #[test]
    fn a_deletion_indication_removes_the_stored_ue_radio_capability_id() {
        let mut orch = MmOrchestrator::from_config(test_identity(), &config_with_racs(true));
        orch.apply_config_update(&config_update_result(command_with_racs_id(
            "123456",
            vec![],
        )));
        assert_eq!(orch.racs_id(), Some("123456"));

        let mut cmd = ConfigurationUpdateCommand::new();
        cmd.ue_radio_capability_id_deletion = Some(RacsDeletionRequest::NetworkAssigned);
        orch.apply_config_update(&config_update_result(cmd));

        assert_eq!(orch.racs_id(), None);
    }

    /// A command that deletes and reassigns in one message ends with the NEW ID:
    /// applying the deletion after the assignment would leave the UE with none.
    #[test]
    fn a_command_that_deletes_and_reassigns_keeps_the_new_id() {
        let mut orch = MmOrchestrator::from_config(test_identity(), &config_with_racs(true));
        orch.apply_config_update(&config_update_result(command_with_racs_id(
            "aaaaaa",
            vec![],
        )));

        let mut cmd = ConfigurationUpdateCommand::new();
        cmd.ue_radio_capability_id_deletion = Some(RacsDeletionRequest::NetworkAssigned);
        cmd.ue_radio_capability_id = Some("bbbbbb".to_string());
        orch.apply_config_update(&config_update_result(cmd));

        assert_eq!(orch.racs_id(), Some("bbbbbb"));
    }

    // ========================================================================
    // Paging identity handed to the AS (#35, TS 38.331 §5.3.2.3)
    // ========================================================================

    /// The 5G-S-TMSI of the fixture GUTI: AMF set/pointer octets 0x0040 then
    /// the 5G-TMSI 0xDEADBEEF (`build_registration_accept_pdu`).
    const FIXTURE_PAGING_IDENTITY: [u8; FIVE_G_S_TMSI_OCTETS] =
        [0x00, 0x40, 0xDE, 0xAD, 0xBE, 0xEF];

    #[test]
    fn a_ue_without_a_5g_guti_has_no_paging_identity_to_signal() {
        let mut orch = new_orch();
        assert_eq!(orch.paging_identity(), None);
        assert_eq!(
            orch.take_paging_identity_update(),
            None,
            "nothing to signal: the AS starts with no identity either"
        );
    }

    #[test]
    fn a_registration_accept_assigning_a_guti_signals_the_paging_identity_once() {
        let mut orch = establish_security_context();
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);

        assert_eq!(
            orch.paging_identity(),
            Some(FIXTURE_PAGING_IDENTITY),
            "the 5G-S-TMSI is the GUTI's AMF set/pointer octets plus the 5G-TMSI"
        );
        assert_eq!(
            orch.take_paging_identity_update(),
            Some(Some(FIXTURE_PAGING_IDENTITY)),
            "the assignment must be reported to the caller for the AS"
        );
        assert_eq!(
            orch.take_paging_identity_update(),
            None,
            "an unchanged identity must not be re-signalled on every output batch"
        );
    }

    /// A UE the network deregisters no longer owns its 5G-GUTI, so the AS must
    /// be told to delete the paging identity — otherwise the UE keeps answering
    /// paging for an identity it has given up.
    #[test]
    fn a_network_deregistration_signals_the_paging_identity_deletion() {
        use nextgsim_nas::ies::ie1::{
            DeRegistrationAccessType, IeDeRegistrationType, ReRegistrationRequired, SwitchOff,
        };

        let mut orch = establish_security_context();
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);
        assert_eq!(
            orch.take_paging_identity_update(),
            Some(Some(FIXTURE_PAGING_IDENTITY))
        );

        let dereg_type = IeDeRegistrationType::new(
            DeRegistrationAccessType::ThreeGppAccess,
            ReRegistrationRequired::NotRequired,
            SwitchOff::NormalDeRegistration,
        );
        let mut req = Vec::new();
        nextgsim_nas::messages::mm::DeregistrationRequestUeTerminated::new(dereg_type)
            .encode(&mut req);
        let protected = protect_downlink(&orch, &req, 2);
        orch.handle_downlink(&protected);

        assert_eq!(orch.paging_identity(), None, "the GUTI is gone");
        assert_eq!(
            orch.take_paging_identity_update(),
            Some(None),
            "the deletion must be signalled, not merely the assignment"
        );
    }

    /// TS 24.501 §5.6.1.1: paging in 5GMM-IDLE starts a service request with
    /// the mobile-terminated service type — the message the paging pipeline
    /// exists to produce.
    #[test]
    fn a_paged_ue_sends_a_service_request_with_the_mt_service_type() {
        let mut orch = establish_security_context();
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);
        orch.state_mut().switch_cm_state(CmState::Idle);
        assert!(orch.state().is_registered() && orch.state().is_idle());

        let outs = orch.start_service_request(ServiceType::MobileTerminatedServices, None);
        let pdu = first_sent_pdu(&outs).to_vec();

        // No non-cleartext IE to send, so no container (§4.4.6 b) 3)).
        let (req, inner) = open_initial_service_request(&orch, &pdu);
        assert!(
            inner.is_none(),
            "an MT request with nothing non-cleartext carries no container"
        );

        assert_eq!(
            req.service_type.service_type,
            ServiceType::MobileTerminatedServices,
            "an MT service request must not be sent as mo-data"
        );
        assert_eq!(
            &req.tmsi.data[3..7],
            &[0xDE, 0xAD, 0xBE, 0xEF],
            "the request identifies the UE by the 5G-S-TMSI it was paged with"
        );
        assert!(orch.timers().t3517.is_running());
    }

    /// #51: PSI *n* maps to bit *n* of the Uplink data status IE
    /// (TS 24.501 §9.11.3.44).
    ///
    /// Asserted over the whole domain rather than on one value, because the
    /// defect being fixed was a CONSTANT: any single-value test passes against a
    /// function that ignores its argument.
    #[test]
    fn uplink_data_status_maps_psi_n_to_bit_n() {
        assert_eq!(uplink_data_status_bit(1), 0x0002);
        assert_eq!(uplink_data_status_bit(3), 0x0008);
        assert_eq!(uplink_data_status_bit(8), 0x0100);
        assert_eq!(uplink_data_status_bit(13), 0x2000);
        assert_eq!(uplink_data_status_bit(15), 0x8000);
        // PSI 0 is spare and 16+ do not exist (TS 24.501 §9.4). Both must yield
        // nothing rather than a wrapped shift -- `1u16 << 16` panics in debug.
        assert_eq!(uplink_data_status_bit(0), 0);
        assert_eq!(uplink_data_status_bit(16), 0);
        assert_eq!(uplink_data_status_bit(u8::MAX), 0);

        // The IE is omitted, not zeroed, when nothing has pending data: an
        // all-zero bitmap would tell the network no session needs reactivating
        // while asking it to reactivate one.
        assert_eq!(uplink_data_status(&[]), None);
        assert_eq!(uplink_data_status(&[0]), None);
        assert_eq!(uplink_data_status(&[1]), Some(0x0002));
        assert_eq!(uplink_data_status(&[1, 3]), Some(0x000A));
    }

    /// #51: a data-triggered SERVICE REQUEST for PSI 1 puts bit 1 on the wire.
    ///
    /// Decoded from the ciphered PDU rather than read back off the struct, so it
    /// pins what the network actually receives.
    #[test]
    fn a_service_request_for_psi_1_advertises_bit_1_on_the_wire() {
        let mut orch = establish_security_context();
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);
        orch.state_mut().switch_cm_state(CmState::Idle);

        let outs = orch.start_service_request(ServiceType::Data, uplink_data_status(&[1]));
        let pdu = first_sent_pdu(&outs).to_vec();

        let (_outer, inner) = open_initial_service_request(&orch, &pdu);
        let req = inner.expect("the Uplink data status rides in the NAS message container");
        assert_eq!(
            req.uplink_data_status,
            Some(0x0002),
            "PSI 1 is bit 1; 0x2000 was bit 13, a PSI no session ever allocates"
        );
    }

    /// #51: a SERVICE ACCEPT carrying a PDU session status IE surfaces it, so the
    /// caller can locally release the sessions the network reports inactive
    /// (TS 24.501 §5.6.1.4.2 a).
    ///
    /// Before this the handler took no message bytes at all, so the decoded IE
    /// was unreachable — which is why the assertion is on the emitted output and
    /// not on a flag: the output is the only thing that can reach the SM
    /// orchestrator.
    /// Issue #267: the user plane is not ready from the moment uplink data
    /// triggers a SERVICE REQUEST from CM-IDLE -- through the RRC connection
    /// coming up -- until the Service Accept; then it is.
    #[test]
    fn the_uplink_user_plane_is_ready_only_once_the_service_request_completes() {
        let mut orch = establish_security_context();
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);
        orch.state_mut().switch_cm_state(CmState::Connected);
        assert!(orch.uplink_user_plane_ready(), "registered and connected");

        orch.state_mut().switch_cm_state(CmState::Idle);
        assert!(!orch.uplink_user_plane_ready(), "CM-IDLE: no user plane");

        orch.start_service_request(ServiceType::Data, uplink_data_status(&[1]));
        // The RRC connection the request rides on is up; the N3 tunnel is not.
        orch.state_mut().switch_cm_state(CmState::Connected);
        assert!(!orch.uplink_user_plane_ready(), "service request in flight");

        let mut plain = Vec::new();
        ServiceAccept::new().encode(&mut plain);
        orch.handle_downlink(&protect_downlink(&orch, &plain, 2));
        assert!(
            orch.uplink_user_plane_ready(),
            "Service Accept: user plane back"
        );
    }

    /// nextgsim #288: a UE that lost its connection (RLF on a gNB that
    /// restarted) and is CM-CONNECTED again after a mobility registration on
    /// the new gNB has NO user plane. Uplink must be held and a Service Request
    /// sent; the user plane counts as up only after Service Accept. Before,
    /// CM-CONNECTED alone read as "user plane up" and uplink went to a DRB the
    /// new gNB never set up.
    #[test]
    fn after_a_lost_connection_cm_connected_alone_is_not_a_user_plane() {
        let mut orch = establish_security_context();
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);
        orch.state_mut().switch_cm_state(CmState::Connected);
        assert!(orch.uplink_user_plane_ready());
        assert!(!orch.user_plane_reactivation_needed());

        // Radio link failure, then a new connection on the new gNB.
        orch.on_nas_signalling_connection_released();
        orch.state_mut().switch_cm_state(CmState::Connected);
        assert!(
            !orch.uplink_user_plane_ready(),
            "connected, but the user plane went with the old connection"
        );
        assert!(orch.user_plane_reactivation_needed());

        let outs = orch.start_service_request(ServiceType::Data, uplink_data_status(&[1]));
        assert!(
            !outs.is_empty(),
            "a Service Request is sent from CM-CONNECTED"
        );
        assert!(
            !orch.user_plane_reactivation_needed(),
            "one request in flight, not one per packet"
        );

        let mut plain = Vec::new();
        ServiceAccept::new().encode(&mut plain);
        orch.handle_downlink(&protect_downlink(&orch, &plain, 2));
        assert!(
            orch.uplink_user_plane_ready(),
            "Service Accept: user plane back"
        );
        assert!(!orch.user_plane_reactivation_needed());
    }

    /// A registered UE that lost its connection and is CM-CONNECTED again with no
    /// user plane, with a data Service Request for PSI 1 in flight (#313).
    fn connected_without_user_plane_with_sr_in_flight() -> MmOrchestrator {
        let mut orch = establish_security_context();
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);
        orch.on_nas_signalling_connection_released();
        orch.state_mut().switch_cm_state(CmState::Connected);
        assert!(orch.user_plane_reactivation_needed());
        let outs = orch.start_service_request(ServiceType::Data, uplink_data_status(&[1]));
        assert!(!outs.is_empty());
        orch
    }

    /// Deliver a protected SERVICE REJECT with `cause` and an optional T3346 value.
    fn reject_service_request(
        orch: &mut MmOrchestrator,
        cause: MmCause,
        t3346: Option<u8>,
        dl_count: u8,
    ) -> Vec<MmOutput> {
        let mut rej = ServiceReject::new(cause);
        rej.t3346_value = t3346;
        let mut plain = Vec::new();
        rej.encode(&mut plain);
        let protected = protect_downlink(orch, &plain, dl_count);
        orch.handle_downlink(&protected)
    }

    /// #313, TS 24.501 §5.6.1.5 #22: a congestion reject with T3346 does NOT bring
    /// the user plane back, so uplink stays held and no new data Service Request
    /// goes out while T3346 runs; once it stops, the next uplink asks again. Before,
    /// any reject cleared the flag: uplink was released onto a DRB that did not
    /// exist and the UE never asked again.
    #[test]
    fn a_congestion_reject_keeps_the_user_plane_down_until_t3346_expires() {
        let mut orch = connected_without_user_plane_with_sr_in_flight();
        // 0x05: GPRS timer 2, unit 2 s, value 5 -> 10 s.
        reject_service_request(&mut orch, MmCause::Congestion, Some(0x05), 2);
        assert!(orch.timers().t3346.is_running());
        assert!(!orch.timers().t3517.is_running());
        assert!(
            !orch.uplink_user_plane_ready(),
            "the reject brought no user plane back"
        );
        assert!(
            !orch.user_plane_reactivation_needed(),
            "no new data Service Request while T3346 runs"
        );
        assert!(
            orch.start_service_request(ServiceType::Data, uplink_data_status(&[1]))
                .is_empty(),
            "a data Service Request is barred under T3346"
        );
        assert!(
            !orch
                .start_service_request(ServiceType::MobileTerminatedServices, None)
                .is_empty(),
            "answering a page is exempt from T3346"
        );
        // Back to the state before the page, with T3346 over.
        orch.timers.stop(TIMER_T3517, true);
        orch.state_mut()
            .switch_mm_state(MmSubState::RegisteredNormalService);
        orch.timers.stop(TIMER_T3346, true);
        assert!(
            orch.user_plane_reactivation_needed(),
            "T3346 over: the next uplink re-asks"
        );
        let outs = orch.start_service_request(ServiceType::Data, uplink_data_status(&[1]));
        assert!(!outs.is_empty(), "a new Service Request is sent");
    }

    /// #313 review, §5.6.1.5 #22: "The service request procedure is started, if
    /// still necessary, when timer T3346 expires". Held uplink gets its Service
    /// Request on the expiry itself, with no new packet to trigger it, and with
    /// the PSI the rejected request named.
    #[test]
    fn t3346_expiry_starts_the_service_request_for_held_uplink() {
        for idle in [false, true] {
            let mut orch = connected_without_user_plane_with_sr_in_flight();
            reject_service_request(&mut orch, MmCause::Congestion, Some(0x05), 2);
            if idle {
                orch.on_nas_signalling_connection_released();
            }
            assert!(orch.held_uplink_data_status.is_some());
            orch.timers.stop(TIMER_T3346, true);
            let outs = orch.handle_timer_expiry(TimerExpiryEvent::new(TIMER_T3346, true, 1));
            assert!(
                outs.iter().any(|o| matches!(o, MmOutput::SendNasPdu(_))),
                "idle={idle}: a Service Request on expiry: {outs:?}"
            );
            assert!(orch.timers().t3517.is_running(), "idle={idle}");
            assert_eq!(orch.pending_service_type, Some(ServiceType::Data));
            assert_eq!(
                orch.held_uplink_data_status,
                uplink_data_status(&[1]),
                "the PSI the rejected request named"
            );
        }
    }

    /// The complement: with no uplink held (the rejected request was signalling),
    /// T3346 expiry starts nothing.
    #[test]
    fn t3346_expiry_without_held_uplink_starts_nothing() {
        let mut orch = establish_security_context();
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);
        orch.on_nas_signalling_connection_released();
        assert!(!orch
            .start_service_request(ServiceType::Signalling, None)
            .is_empty());
        reject_service_request(&mut orch, MmCause::Congestion, Some(0x05), 2);
        orch.timers.stop(TIMER_T3346, true);
        let outs = orch.handle_timer_expiry(TimerExpiryEvent::new(TIMER_T3346, true, 1));
        assert!(outs.is_empty(), "{outs:?}");
        assert!(!orch.timers().t3517.is_running());
    }

    /// #313 review: the end of the local backoff is noticed by the 1 s tick, once,
    /// and held uplink asks again then; a Service Accept ends the holding.
    #[test]
    fn the_end_of_the_backoff_starts_the_service_request_for_held_uplink() {
        let mut orch = connected_without_user_plane_with_sr_in_flight();
        reject_service_request(&mut orch, MmCause::ProtocolErrorUnspecified, None, 2);
        assert!(
            !orch
                .tick()
                .iter()
                .any(|o| matches!(o, MmOutput::SendNasPdu(_))),
            "not inside the backoff"
        );
        orch.sr_backoff_until = Some(std::time::Instant::now());
        let outs = orch.tick();
        assert!(
            outs.iter().any(|o| matches!(o, MmOutput::SendNasPdu(_))),
            "{outs:?}"
        );
        assert!(orch.sr_backoff_until.is_none(), "noticed once");
        assert_eq!(orch.sr_reject_count, 1, "the next backoff still doubles");

        let mut plain = Vec::new();
        ServiceAccept::new().encode(&mut plain);
        orch.handle_downlink(&protect_downlink(&orch, &plain, 3));
        assert!(orch.held_uplink_data_status.is_none());
    }

    /// #313, §5.6.1.5 #28: restricted service area. The UE enters
    /// 5GMM-REGISTERED.NON-ALLOWED-SERVICE with 5U1 UPDATED; its sessions stay but
    /// their user plane cannot be activated, so uplink is held and only exempt
    /// requests (a paging response) may go out.
    #[test]
    fn a_restricted_service_area_reject_enters_non_allowed_service() {
        let mut orch = connected_without_user_plane_with_sr_in_flight();
        let outs = reject_service_request(&mut orch, MmCause::RestrictedServiceArea, None, 2);
        assert!(
            !outs.contains(&MmOutput::ReleasePduSessionsLocally),
            "#28 keeps the PDU sessions"
        );
        assert!(
            outs.is_empty(),
            "the registration waits for the release: {outs:?}"
        );
        assert_eq!(
            orch.state().mm_substate(),
            MmSubState::RegisteredNonAllowedService
        );
        assert!(!orch.uplink_user_plane_ready());
        assert!(!orch.user_plane_reactivation_needed());
        assert!(orch
            .start_service_request(ServiceType::Data, uplink_data_status(&[1]))
            .is_empty());

        // §5.6.1.5 #28: once the N1 NAS signalling connection is released, a
        // registration for mobility and periodic update.
        let outs = orch.on_nas_signalling_connection_released();
        let pdu = first_sent_pdu(&outs);
        let (_outer, full) = open_registration_request(&orch, pdu);
        assert_eq!(
            full.registration_type.registration_type,
            RegistrationType::MobilityRegistrationUpdating
        );
        // §5.5.1.3.2 (#313 re-review): in a non-allowed area the UE "shall not
        // include the Uplink data status IE except for emergency services or
        // for high priority access".
        assert_eq!(full.uplink_data_status, None);
        assert_eq!(
            full.registration_type.follow_on_request_pending,
            FollowOnRequest::NoPending
        );
    }

    /// #313 re-review: a UE configured for high priority access does name its
    /// held uplink in the #28 update.
    #[test]
    fn a_high_priority_ue_names_held_uplink_in_the_update_after_28() {
        let mut orch = connected_without_user_plane_with_sr_in_flight();
        orch.high_priority_access = true;
        reject_service_request(&mut orch, MmCause::RestrictedServiceArea, None, 2);
        let outs = orch.on_nas_signalling_connection_released();
        let (_outer, full) = open_registration_request(&orch, first_sent_pdu(&outs));
        assert_eq!(full.uplink_data_status, uplink_data_status(&[1]));
        assert_eq!(
            full.registration_type.follow_on_request_pending,
            FollowOnRequest::Pending
        );
    }

    /// The UE held after another reject, now idle.
    fn idle_with_held_uplink() -> MmOrchestrator {
        let mut orch = connected_without_user_plane_with_sr_in_flight();
        reject_service_request(&mut orch, MmCause::ProtocolErrorUnspecified, None, 2);
        orch.on_nas_signalling_connection_released();
        assert_eq!(orch.held_uplink_data_status, uplink_data_status(&[1]));
        orch
    }

    /// #313 re-review, §5.5.1.3.2: a mobility update names the held session; a
    /// periodic one (case b, T3512 expiry in 5GMM-IDLE) never does.
    #[test]
    fn only_a_mobility_update_names_held_uplink() {
        for (reg_type, named) in [
            (RegistrationType::MobilityRegistrationUpdating, true),
            (RegistrationType::PeriodicRegistrationUpdating, false),
        ] {
            let mut orch = idle_with_held_uplink();
            let outs = orch.start_registration(reg_type);
            let (_outer, full) = open_registration_request(&orch, first_sent_pdu(&outs));
            assert_eq!(
                full.uplink_data_status.is_some(),
                named,
                "{reg_type:?}: {:?}",
                full.uplink_data_status
            );
        }
    }

    /// #313 re-review: a released session leaves the held status, and losing
    /// the registration clears it.
    #[test]
    fn held_uplink_is_cleared_by_release_and_deregistration() {
        let mut orch = idle_with_held_uplink();
        orch.on_pdu_session_released(1);
        assert_eq!(orch.held_uplink_data_status, None);
        let mut orch = idle_with_held_uplink();
        orch.delete_registration_context();
        assert_eq!(orch.held_uplink_data_status, None);
    }

    /// An Accept to an update that named held uplink but carries NO PDU session
    /// reactivation result re-established nothing (§5.5.1.3.4: the AMF includes
    /// the IE "to indicate the user-plane resources re-establishment result").
    /// The uplink stays held and, connected, the UE asks with a Service Request.
    #[test]
    fn an_accept_without_a_reactivation_result_restores_nothing() {
        let mut orch = idle_with_held_uplink();
        orch.sr_backoff_until = None;
        orch.start_registration(RegistrationType::MobilityRegistrationUpdating);
        assert_eq!(
            orch.registration_uplink_data_status,
            uplink_data_status(&[1])
        );
        let mut acc = RegistrationAccept::new(Ie5gsRegistrationResult::new(
            SmsOverNasAllowed::NotAllowed,
            RegistrationResultValue::ThreeGppAccess,
        ));
        acc.tai_list = Some(vec![0x00, 0x99, 0xF9, 0x07, 0x00, 0x00, 0x01]);
        let mut plain = Vec::new();
        acc.encode(&mut plain);
        orch.state_mut().switch_cm_state(CmState::Connected);
        let outs = orch.handle_downlink(&protect_downlink(&orch, &plain, 3));
        assert!(orch.up_reactivation_pending, "the user plane is still down");
        assert_eq!(orch.held_uplink_data_status, uplink_data_status(&[1]));
        assert!(
            orch.timers().t3517.is_running(),
            "a Service Request asks for it: {outs:?}"
        );
    }

    /// #313 review, §5.5.1.3.4: the Registration Accept to an update that named
    /// held uplink brings its user plane back, unless the PDU session
    /// reactivation result marks the session as failed (bit set = failed).
    #[test]
    fn a_registration_accept_restores_the_user_plane_the_update_asked_for() {
        for (failed, restored) in [(Some(0x0000), true), (Some(0x0002), false)] {
            let mut orch = idle_with_held_uplink();
            orch.start_registration(RegistrationType::MobilityRegistrationUpdating);
            assert_eq!(
                orch.registration_uplink_data_status,
                uplink_data_status(&[1])
            );
            let mut acc = RegistrationAccept::new(Ie5gsRegistrationResult::new(
                SmsOverNasAllowed::NotAllowed,
                RegistrationResultValue::ThreeGppAccess,
            ));
            acc.tai_list = Some(vec![0x00, 0x99, 0xF9, 0x07, 0x00, 0x00, 0x01]);
            acc.pdu_session_reactivation_result = failed;
            let mut plain = Vec::new();
            acc.encode(&mut plain);
            orch.state_mut().switch_cm_state(CmState::Connected);
            orch.handle_downlink(&protect_downlink(&orch, &plain, 3));
            assert_eq!(!orch.up_reactivation_pending, restored, "failed={failed:?}");
            assert_eq!(
                orch.held_uplink_data_status,
                failed.filter(|f| *f != 0),
                "what failed is still held for a Service Request"
            );
        }
    }

    /// #331 review, §5.6.1.5 #22 / §5.6.1.7 c): "The service request procedure
    /// is started, if still necessary, when timer T3346 expires or is stopped."
    /// A Registration Accept stops T3346, and uplink still held then gets its
    /// Service Request. With nothing held, nothing is asked.
    #[test]
    fn a_registration_accept_that_stops_t3346_retries_held_uplink() {
        for held in [true, false] {
            let mut orch = connected_without_user_plane_with_sr_in_flight();
            reject_service_request(&mut orch, MmCause::Congestion, Some(0x05), 2);
            orch.on_nas_signalling_connection_released();
            assert!(orch.timers().t3346.is_running());
            if !held {
                orch.held_uplink_data_status = None;
            }
            let mut acc = RegistrationAccept::new(Ie5gsRegistrationResult::new(
                SmsOverNasAllowed::NotAllowed,
                RegistrationResultValue::ThreeGppAccess,
            ));
            acc.tai_list = Some(vec![0x00, 0x99, 0xF9, 0x07, 0x00, 0x00, 0x01]);
            let mut plain = Vec::new();
            acc.encode(&mut plain);
            orch.state_mut().switch_cm_state(CmState::Connected);
            let outs = orch.handle_downlink(&protect_downlink(&orch, &plain, 3));
            assert!(!orch.timers().t3346.is_running(), "the Accept stops T3346");
            assert_eq!(
                orch.timers().t3517.is_running(),
                held,
                "held={held}: a Service Request after the Accept: {outs:?}"
            );
            if held {
                assert_eq!(orch.pending_service_type, Some(ServiceType::Data));
            }
        }
    }

    /// #313 review, §5.6.1.7 c) 4): the T3346 exemption belongs to a UE
    /// CONFIGURED for high priority access, whatever the request's service type.
    #[test]
    fn the_high_priority_exemption_follows_the_configuration() {
        for configured in [false, true] {
            let mut orch = connected_without_user_plane_with_sr_in_flight();
            orch.high_priority_access = configured;
            reject_service_request(&mut orch, MmCause::Congestion, Some(0x05), 2);
            assert!(orch.timers().t3346.is_running());
            let sent = !orch
                .start_service_request(ServiceType::Data, uplink_data_status(&[1]))
                .is_empty();
            assert_eq!(sent, configured, "configured={configured}");
        }
        let mut orch = connected_without_user_plane_with_sr_in_flight();
        reject_service_request(&mut orch, MmCause::Congestion, Some(0x05), 2);
        assert!(
            orch.start_service_request(ServiceType::HighPriorityAccess, None)
                .is_empty(),
            "the service type alone exempts nothing"
        );
    }

    /// #313, §5.6.1.5 #22: an UNPROTECTED congestion reject's T3346 value is not
    /// used; T3346 runs for a random 15-30 minutes (TS 24.008 default range).
    #[test]
    fn an_unprotected_congestion_reject_uses_the_default_t3346_range() {
        let mut orch = establish_security_context();
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);
        orch.on_nas_signalling_connection_released();
        assert!(!orch
            .start_service_request(ServiceType::Data, uplink_data_status(&[1]))
            .is_empty());
        let mut rej = ServiceReject::new(MmCause::Congestion);
        rej.t3346_value = Some(0x05); // 10 s, which must NOT be used
        let mut plain = Vec::new();
        rej.encode(&mut plain);
        orch.handle_downlink(&plain);
        assert!(orch.timers().t3346.is_running());
        let left = orch.timers().t3346.remaining();
        assert!((15 * 60 - 1..=30 * 60).contains(&left), "T3346 {left}s");
    }

    /// #313, §5.3.5: in NON-ALLOWED-SERVICE a paging response is still allowed.
    #[test]
    fn non_allowed_service_still_answers_paging() {
        let mut orch = connected_without_user_plane_with_sr_in_flight();
        reject_service_request(&mut orch, MmCause::RestrictedServiceArea, None, 2);
        assert!(!orch
            .start_service_request(ServiceType::MobileTerminatedServices, None)
            .is_empty());
    }

    /// #313: any other non-terminal reject (here #111) also leaves the user plane
    /// down. The UE does not re-ask on the very next packet -- that would loop the
    /// AMF through reject after reject -- but after a backoff that doubles per
    /// consecutive reject, and a Service Accept resets it.
    #[test]
    fn another_reject_keeps_the_user_plane_down_and_backs_off() {
        let mut orch = connected_without_user_plane_with_sr_in_flight();
        reject_service_request(&mut orch, MmCause::ProtocolErrorUnspecified, None, 2);
        assert!(orch.state().is_registered());
        assert!(!orch.uplink_user_plane_ready());
        assert!(!orch.user_plane_reactivation_needed(), "inside the backoff");
        let until = orch.sr_backoff_until.expect("a backoff");
        let first = until.saturating_duration_since(std::time::Instant::now());
        assert!(first <= SERVICE_REJECT_BACKOFF_BASE && first > SERVICE_REJECT_BACKOFF_BASE / 2);

        // Backoff over: the next uplink re-asks, and a second reject doubles it.
        orch.sr_backoff_until = Some(std::time::Instant::now());
        assert!(orch.user_plane_reactivation_needed());
        assert!(!orch
            .start_service_request(ServiceType::Data, uplink_data_status(&[1]))
            .is_empty());
        reject_service_request(&mut orch, MmCause::ProtocolErrorUnspecified, None, 3);
        let second = orch
            .sr_backoff_until
            .unwrap()
            .saturating_duration_since(std::time::Instant::now());
        assert!(second > SERVICE_REJECT_BACKOFF_BASE, "doubled: {second:?}");

        // A Service Accept ends the backoff and the user-plane gap.
        orch.sr_backoff_until = Some(std::time::Instant::now());
        orch.start_service_request(ServiceType::Data, uplink_data_status(&[1]));
        let mut plain = Vec::new();
        ServiceAccept::new().encode(&mut plain);
        orch.handle_downlink(&protect_downlink(&orch, &plain, 4));
        assert!(orch.uplink_user_plane_ready());
        assert_eq!(orch.sr_reject_count, 0);
        assert!(orch.sr_backoff_until.is_none());
    }

    /// #313: the causes that release the PDU sessions DO end the user-plane gap --
    /// there is nothing left to reactivate.
    #[test]
    fn a_reject_that_releases_the_sessions_clears_the_reactivation() {
        for cause in [MmCause::IllegalUe, MmCause::UeIdentityCannotBeDerived] {
            let mut orch = connected_without_user_plane_with_sr_in_flight();
            let outs = reject_service_request(&mut orch, cause, None, 2);
            assert!(
                outs.contains(&MmOutput::ReleasePduSessionsLocally),
                "{cause:?}: {outs:?}"
            );
            assert!(!orch.up_reactivation_pending, "{cause:?}");
        }
    }

    // ---- #322: SERVICE REJECT cause by cause (TS 24.501 §5.6.1.5) ----

    /// The fixture's registration-area TAI: PLMN 999/70, TAC 1.
    const FIXTURE_TAI: [u8; 6] = [0x99, 0xF9, 0x07, 0x00, 0x00, 0x01];

    /// Deliver a protected SERVICE REJECT built by `build`.
    fn reject_service_request_with(
        orch: &mut MmOrchestrator,
        cause: MmCause,
        dl_count: u8,
        build: impl FnOnce(&mut ServiceReject),
    ) -> Vec<MmOutput> {
        let mut rej = ServiceReject::new(cause);
        build(&mut rej);
        let mut plain = Vec::new();
        rej.encode(&mut plain);
        let protected = protect_downlink(orch, &plain, dl_count);
        orch.handle_downlink(&protected)
    }

    /// #11 and #73: 5U3, the registration context and the equivalent PLMNs go,
    /// the PLMN is forbidden, 5GMM-DEREGISTERED.PLMN-SEARCH, PLMN selection, and
    /// the PDU sessions are released (DEREGISTERED holds none).
    #[test]
    fn a_plmn_not_allowed_service_reject_leaves_the_plmn() {
        for cause in [
            MmCause::PlmnNotAllowed,
            MmCause::ServingNetworkNotAuthorized,
        ] {
            let mut orch = connected_without_user_plane_with_sr_in_flight();
            orch.equivalent_plmns = vec![[0x00, 0xF1, 0x10]];
            let outs = reject_service_request(&mut orch, cause, None, 2);
            for want in [
                MmOutput::ReleasePduSessionsLocally,
                MmOutput::PlmnSearchNeeded,
                MmOutput::RegistrationFailed(cause),
            ] {
                assert!(
                    outs.contains(&want),
                    "{cause:?}: {want:?} missing from {outs:?}"
                );
            }
            assert_eq!(
                orch.state().mm_substate(),
                MmSubState::DeregisteredPlmnSearch,
                "{cause:?}"
            );
            assert_eq!(
                orch.state().update_status(),
                UpdateStatus::RoamingNotAllowed
            );
            assert!(orch.stored_guti().is_none() && orch.tai_list().is_none());
            assert!(orch.equivalent_plmns().is_empty(), "{cause:?}");
            assert_eq!(orch.forbidden_plmns(), &[orch.identity.plmn_bcd]);
            assert!(!orch.up_reactivation_pending);
        }
    }

    /// §5.6.1.5: #11/#73 from an SNPN cell, and #74/#75 from a non-SNPN cell,
    /// are abnormal (§5.6.1.7 i): the UE stays registered.
    #[test]
    fn snpn_mismatched_causes_are_abnormal() {
        for (cause, snpn) in [
            (MmCause::PlmnNotAllowed, true),
            (MmCause::ServingNetworkNotAuthorized, true),
            (MmCause::TemporarilyNotAuthorizedForSnpn, false),
            (MmCause::PermanentlyNotAuthorizedForSnpn, false),
        ] {
            let mut orch = connected_without_user_plane_with_sr_in_flight();
            orch.identity.snpn_nid = snpn.then(|| "000000000001".to_string());
            let outs = reject_service_request(&mut orch, cause, None, 2);
            assert!(outs.is_empty(), "{cause:?}: {outs:?}");
            assert_eq!(
                orch.state().mm_substate(),
                MmSubState::RegisteredNormalService,
                "{cause:?}"
            );
            assert!(orch.stored_guti().is_some(), "{cause:?}");
            assert!(orch.forbidden_plmns().is_empty(), "{cause:?}");
            assert!(orch.sr_backoff_until.is_some(), "{cause:?}");
        }
    }

    /// #74/#75 from an SNPN cell: the SNPN is forbidden, DEREGISTERED.PLMN-SEARCH.
    #[test]
    fn an_snpn_not_authorized_service_reject_leaves_the_snpn() {
        for cause in [
            MmCause::TemporarilyNotAuthorizedForSnpn,
            MmCause::PermanentlyNotAuthorizedForSnpn,
        ] {
            let mut orch = connected_without_user_plane_with_sr_in_flight();
            orch.identity.snpn_nid = Some("000000000001".to_string());
            let outs = reject_service_request(&mut orch, cause, None, 2);
            assert!(outs.contains(&MmOutput::PlmnSearchNeeded), "{outs:?}");
            assert!(outs.contains(&MmOutput::ReleasePduSessionsLocally));
            assert_eq!(
                orch.state().mm_substate(),
                MmSubState::DeregisteredPlmnSearch
            );
            assert_eq!(
                (
                    orch.snpn_temporarily_forbidden,
                    orch.snpn_permanently_forbidden
                ),
                (
                    cause == MmCause::TemporarilyNotAuthorizedForSnpn,
                    cause == MmCause::PermanentlyNotAuthorizedForSnpn
                )
            );
            assert!(orch.stored_guti().is_none());
        }
    }

    /// #12: the current TAI goes on the "regional provision of service" list,
    /// the registration context goes, 5GMM-DEREGISTERED.LIMITED-SERVICE, and
    /// RRC is told the new list.
    #[test]
    fn a_tracking_area_not_allowed_service_reject_forbids_the_ta_and_deregisters() {
        let mut orch = connected_without_user_plane_with_sr_in_flight();
        let outs = reject_service_request(&mut orch, MmCause::TrackingAreaNotAllowed, None, 2);
        assert_eq!(orch.forbidden_tais_service(), &[FIXTURE_TAI]);
        assert!(orch.forbidden_tais_roaming().is_empty());
        assert_eq!(
            orch.state().mm_substate(),
            MmSubState::DeregisteredLimitedService
        );
        assert_eq!(
            orch.state().update_status(),
            UpdateStatus::RoamingNotAllowed
        );
        assert!(orch.stored_guti().is_none() && orch.tai_list().is_none());
        assert!(outs.contains(&MmOutput::ReleasePduSessionsLocally));
        assert!(outs.contains(&MmOutput::ForbiddenTaisChanged {
            roaming: vec![],
            service: vec![FIXTURE_TAI],
        }));
    }

    /// #13: REGISTERED.PLMN-SEARCH, the current TAI on the "roaming" list and out
    /// of the TAI list, PLMN selection. The 5G-GUTI and the PDU sessions stay;
    /// no normal Service Request goes out.
    #[test]
    fn a_roaming_not_allowed_in_ta_service_reject_keeps_the_registration() {
        let mut orch = connected_without_user_plane_with_sr_in_flight();
        let outs = reject_service_request(&mut orch, MmCause::RoamingNotAllowedInTa, None, 2);
        assert_eq!(orch.state().mm_substate(), MmSubState::RegisteredPlmnSearch);
        assert_eq!(
            orch.state().update_status(),
            UpdateStatus::RoamingNotAllowed
        );
        assert_eq!(orch.forbidden_tais_roaming(), &[FIXTURE_TAI]);
        assert!(orch.serving_tai_forbidden() || orch.serving_tai.is_none());
        assert!(orch.tai_list().is_none(), "its only TAI was removed");
        assert!(orch.stored_guti().is_some());
        assert!(outs.contains(&MmOutput::PlmnSearchNeeded), "{outs:?}");
        assert!(!outs.contains(&MmOutput::ReleasePduSessionsLocally));
        assert!(outs.contains(&MmOutput::ForbiddenTaisChanged {
            roaming: vec![FIXTURE_TAI],
            service: vec![],
        }));
        assert!(orch
            .start_service_request(ServiceType::Data, uplink_data_status(&[1]))
            .is_empty());
        assert!(orch
            .start_service_request(ServiceType::MobileTerminatedServices, None)
            .is_empty());
    }

    /// #13 removes only the current TAI: the rest of the registration area stays.
    #[test]
    fn a_roaming_not_allowed_in_ta_reject_removes_only_the_current_tai() {
        let mut orch = connected_without_user_plane_with_sr_in_flight();
        // Type 00, two TACs (1 and 2) of 999/70; camped in TAC 2.
        orch.tai_list = Some(vec![0x01, 0x99, 0xF9, 0x07, 0, 0, 1, 0, 0, 2]);
        let tac2 = [0x99, 0xF9, 0x07, 0, 0, 2];
        orch.set_serving_tai(tac2);
        reject_service_request(&mut orch, MmCause::RoamingNotAllowedInTa, None, 2);
        let list = orch.tai_list().expect("TAC 1 is left");
        assert_eq!(tai_list_entries(list), vec![FIXTURE_TAI]);
        assert_eq!(orch.forbidden_tais_roaming(), &[tac2]);
    }

    /// #15: REGISTERED.LIMITED-SERVICE, the current TAI on the "roaming" list.
    #[test]
    fn a_no_suitable_cells_service_reject_enters_limited_service() {
        let mut orch = connected_without_user_plane_with_sr_in_flight();
        let outs = reject_service_request(&mut orch, MmCause::NoSuitableCellsInTa, None, 2);
        assert_eq!(
            orch.state().mm_substate(),
            MmSubState::RegisteredLimitedService
        );
        assert_eq!(orch.forbidden_tais_roaming(), &[FIXTURE_TAI]);
        assert!(orch.tai_list().is_none());
        assert!(orch.stored_guti().is_some());
        assert!(!outs.contains(&MmOutput::ReleasePduSessionsLocally));
        assert!(outs
            .iter()
            .any(|o| matches!(o, MmOutput::ForbiddenTaisChanged { .. })));
        assert!(orch
            .start_service_request(ServiceType::Data, uplink_data_status(&[1]))
            .is_empty());
    }

    /// #15 (Rel-19): the current TAI is NOT stored when the reject names the
    /// forbidden TAIs itself, or carries an integrity-protected access
    /// technology utilization control IE. The named TAIs are stored instead.
    #[test]
    fn a_no_suitable_cells_reject_naming_tais_does_not_forbid_the_current_one() {
        let other = [0x99, 0xF9, 0x07, 0x00, 0x00, 0x09];
        let mut orch = connected_without_user_plane_with_sr_in_flight();
        reject_service_request_with(&mut orch, MmCause::NoSuitableCellsInTa, 2, |r| {
            r.forbidden_tais_roaming = encode_tai_list(&[other]);
        });
        assert_eq!(orch.forbidden_tais_roaming(), &[other]);
        assert!(
            orch.tai_list().is_some(),
            "the current TAI stays in the list"
        );

        let mut orch = connected_without_user_plane_with_sr_in_flight();
        reject_service_request_with(&mut orch, MmCause::NoSuitableCellsInTa, 2, |r| {
            r.access_tech_utilization_control = Some(vec![0x01]);
        });
        assert!(orch.forbidden_tais_roaming().is_empty());
        assert_eq!(
            orch.state().mm_substate(),
            MmSubState::RegisteredLimitedService
        );
    }

    /// §5.6.1.5, regardless of the cause: the forbidden-TAI IEs are stored, but
    /// only TAIs of the serving or an equivalent PLMN; the rest are ignored.
    /// Stored once, however often they are repeated.
    #[test]
    fn forbidden_tai_ies_are_stored_for_the_serving_and_equivalent_plmns_only() {
        let ours = [0x99, 0xF9, 0x07, 0x00, 0x00, 0x05];
        let equivalent = [0x00, 0xF1, 0x10, 0x00, 0x00, 0x05];
        let foreign = [0x21, 0xF3, 0x54, 0x00, 0x00, 0x05];
        let mut orch = connected_without_user_plane_with_sr_in_flight();
        orch.equivalent_plmns = vec![[0x00, 0xF1, 0x10]];
        let outs =
            reject_service_request_with(&mut orch, MmCause::ProtocolErrorUnspecified, 2, |r| {
                r.forbidden_tais_roaming = encode_tai_list(&[ours, foreign, ours]);
                r.forbidden_tais_service = encode_tai_list(&[equivalent]);
            });
        assert_eq!(orch.forbidden_tais_roaming(), &[ours]);
        assert_eq!(orch.forbidden_tais_service(), &[equivalent]);
        assert!(outs.contains(&MmOutput::ForbiddenTaisChanged {
            roaming: vec![ours],
            service: vec![equivalent],
        }));
        // The cause is still handled: #111 is abnormal.
        assert_eq!(
            orch.state().mm_substate(),
            MmSubState::RegisteredNormalService
        );
    }

    /// #27: 5U3, REGISTERED.LIMITED-SERVICE, N1 mode disabled: no Service
    /// Request, no registration.
    #[test]
    fn an_n1_mode_not_allowed_service_reject_disables_n1_mode() {
        let mut orch = connected_without_user_plane_with_sr_in_flight();
        reject_service_request(&mut orch, MmCause::N1ModeNotAllowed, None, 2);
        assert!(orch.n1_mode_disabled);
        assert_eq!(
            orch.state().mm_substate(),
            MmSubState::RegisteredLimitedService
        );
        assert_eq!(
            orch.state().update_status(),
            UpdateStatus::RoamingNotAllowed
        );
        assert!(orch
            .start_service_request(ServiceType::Data, uplink_data_status(&[1]))
            .is_empty());
        assert!(orch
            .start_registration(RegistrationType::MobilityRegistrationUpdating)
            .is_empty());
    }

    /// The causes §5.6.1.5 makes abnormal for this UE -- #31 (no CIoT support
    /// indicated), #72 (3GPP access), #76 (no CAG support), #77 (not wireline)
    /// -- and #62, which §5.6.1.5 does not list: 5GMM-REGISTERED, the
    /// registration kept, the local backoff.
    #[test]
    fn causes_that_are_abnormal_here_keep_the_ue_registered() {
        for cause in [
            MmCause::RedirectionToEpcRequired,
            MmCause::Non3gppAccessTo5gcnNotAllowed,
            MmCause::NotAuthorizedForCag,
            MmCause::WirelineAccessAreaNotAllowed,
            MmCause::NoNetworkSlicesAvailable,
        ] {
            let mut orch = connected_without_user_plane_with_sr_in_flight();
            let outs = reject_service_request(&mut orch, cause, None, 2);
            assert!(outs.is_empty(), "{cause:?}: {outs:?}");
            assert_eq!(
                orch.state().mm_substate(),
                MmSubState::RegisteredNormalService,
                "{cause:?}"
            );
            assert_eq!(
                orch.state().update_status(),
                UpdateStatus::Updated,
                "{cause:?}"
            );
            assert!(orch.stored_guti().is_some(), "{cause:?}");
            assert!(!orch.n1_mode_disabled, "{cause:?}");
            assert!(!orch.timers().t3517.is_running(), "{cause:?}");
            assert!(orch.sr_backoff_until.is_some(), "{cause:?}");
        }
    }

    /// §5.6.1.5: a SERVICE REJECT with #76 or #78 that is not integrity
    /// protected is discarded: T3517 keeps running, nothing changes.
    #[test]
    fn an_unprotected_service_reject_76_or_78_is_discarded() {
        for raw in [76u8, 78] {
            let mut orch = establish_security_context();
            let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
            orch.handle_downlink(&acc);
            orch.on_nas_signalling_connection_released();
            assert!(!orch
                .start_service_request(ServiceType::Data, uplink_data_status(&[1]))
                .is_empty());
            let mut plain = Vec::new();
            ServiceReject::new(MmCause::NotAuthorizedForCag).encode(&mut plain);
            plain[3] = raw;
            let outs = orch.handle_downlink(&plain);
            assert!(outs.is_empty(), "#{raw}: {outs:?}");
            assert!(orch.timers().t3517.is_running(), "#{raw}: not processed");
            assert!(orch.sr_backoff_until.is_none(), "#{raw}");
        }
    }

    /// The control: an unprotected reject with another cause on a new
    /// connection IS processed (§4.4.4.2), so the test above isolates #76/#78.
    #[test]
    fn an_unprotected_service_reject_with_another_cause_is_processed() {
        let mut orch = establish_security_context();
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);
        orch.on_nas_signalling_connection_released();
        orch.start_service_request(ServiceType::Data, uplink_data_status(&[1]));
        let mut plain = Vec::new();
        ServiceReject::new(MmCause::ProtocolErrorUnspecified).encode(&mut plain);
        orch.handle_downlink(&plain);
        assert!(!orch.timers().t3517.is_running());
    }

    /// §5.6.1.5: the PDU session status IE of an integrity-protected reject
    /// reaches the SM reconciliation; from an unprotected reject it is ignored.
    #[test]
    fn a_service_reject_pdu_session_status_is_used_only_when_protected() {
        let mut orch = connected_without_user_plane_with_sr_in_flight();
        let outs =
            reject_service_request_with(&mut orch, MmCause::ProtocolErrorUnspecified, 2, |r| {
                r.pdu_session_status = Some(0x0002);
            });
        assert!(
            outs.contains(&MmOutput::NetworkPduSessionStatus(0x0002)),
            "{outs:?}"
        );

        let mut orch = establish_security_context();
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);
        orch.on_nas_signalling_connection_released();
        orch.start_service_request(ServiceType::Data, uplink_data_status(&[1]));
        let mut rej = ServiceReject::new(MmCause::ProtocolErrorUnspecified);
        rej.pdu_session_status = Some(0x0000);
        let mut plain = Vec::new();
        rej.encode(&mut plain);
        let outs = orch.handle_downlink(&plain);
        assert!(
            !outs
                .iter()
                .any(|o| matches!(o, MmOutput::NetworkPduSessionStatus(_))),
            "{outs:?}"
        );
    }

    /// #9 sets 5U2 NOT UPDATED (§5.6.1.5).
    #[test]
    fn a_ue_identity_cannot_be_derived_service_reject_sets_5u2() {
        let mut orch = connected_without_user_plane_with_sr_in_flight();
        reject_service_request(&mut orch, MmCause::UeIdentityCannotBeDerived, None, 2);
        assert_eq!(orch.state().update_status(), UpdateStatus::NotUpdated);
    }

    /// A registration reject's #12/#13/#15 also reach RRC cell selection, and a
    /// repeated reject does not duplicate the TAI.
    #[test]
    fn a_registration_reject_forbidding_a_ta_tells_rrc_once() {
        for already_stored in [false, true] {
            let mut orch = registered_orch();
            orch.current_tai = Some(FIXTURE_TAI);
            if already_stored {
                orch.forbidden_tais_roaming = vec![FIXTURE_TAI];
            }
            // A new connection, so the unprotected reject is accepted (§4.4.4.2).
            orch.on_nas_signalling_connection_released();
            orch.start_registration(RegistrationType::InitialRegistration);
            let outs = orch.handle_downlink(&build_registration_reject_pdu(
                MmCause::NoSuitableCellsInTa,
                None,
            ));
            assert_eq!(orch.forbidden_tais_roaming(), &[FIXTURE_TAI]);
            assert!(outs.contains(&MmOutput::ForbiddenTaisChanged {
                roaming: vec![FIXTURE_TAI],
                service: vec![],
            }));
        }
    }

    /// §9.11.3.9: every partial-list type expands to its TAIs, and the type-10
    /// re-encoding reads back the same.
    #[test]
    fn tai_list_entries_expand_every_partial_list_type() {
        let list = [
            // 00: 999/70, TACs 1 and 2
            0x01, 0x99, 0xF9, 0x07, 0, 0, 1, 0, 0, 2, //
            // 01: 001/01, three consecutive TACs from 0x10
            0x22, 0x00, 0xF1, 0x10, 0, 0, 0x10, //
            // 10: one PLMN+TAC pair
            0x40, 0x21, 0xF3, 0x54, 0, 1, 0,
        ];
        let entries = tai_list_entries(&list);
        assert_eq!(entries.len(), 6);
        assert_eq!(entries[1], [0x99, 0xF9, 0x07, 0, 0, 2]);
        assert_eq!(entries[4], [0x00, 0xF1, 0x10, 0, 0, 0x12]);
        assert_eq!(entries[5], [0x21, 0xF3, 0x54, 0, 1, 0]);
        let encoded = encode_tai_list(&entries).unwrap();
        assert_eq!(tai_list_entries(&encoded), entries);
        for t in &entries {
            assert!(tai_list_contains(&encoded, t));
        }
        assert_eq!(encode_tai_list(&[]), None);
        // 17 TAIs need two partial lists of at most 16.
        let many: Vec<[u8; 6]> = (0..17u8).map(|i| [0x99, 0xF9, 0x07, 0, 0, i]).collect();
        let encoded = encode_tai_list(&many).unwrap();
        assert_eq!(encoded[0], 0x4F);
        assert_eq!(tai_list_entries(&encoded), many);
        // A truncated list yields what parsed, and never panics.
        assert_eq!(tai_list_entries(&list[..12]).len(), 2);
        // A type-01 run cannot pass the 24-bit TAC space.
        assert_eq!(
            tai_list_entries(&[0x3F, 0x00, 0xF1, 0x10, 0xFF, 0xFF, 0xFE]).len(),
            2
        );
    }

    #[test]
    fn a_service_accept_surfaces_the_network_pdu_session_status() {
        let mut orch = establish_security_context();
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);
        orch.state_mut().switch_cm_state(CmState::Idle);
        orch.start_service_request(ServiceType::Data, uplink_data_status(&[1]));

        // PSI 1 active, PSI 3 NOT: a UE holding both must release 3.
        let mut accept = ServiceAccept::new();
        accept.pdu_session_status = Some(0x0002);
        let mut plain = Vec::new();
        accept.encode(&mut plain);
        let outs = orch.handle_downlink(&protect_downlink(&orch, &plain, 2));

        assert!(
            outs.contains(&MmOutput::NetworkPduSessionStatus(0x0002)),
            "the PDU session status IE must reach the caller, got {outs:?}"
        );
        // The procedure itself still completed.
        assert_eq!(
            orch.state().mm_substate(),
            MmSubState::RegisteredNormalService
        );
        assert!(!orch.timers().t3517.is_running(), "T3517 must be stopped");
    }

    /// #51: a SERVICE ACCEPT with no PDU session status IE reconciles nothing.
    ///
    /// The negative case matters here: emitting a zero bitmap for an absent IE
    /// would read as "the network says every session is inactive" and release
    /// every session the UE holds.
    #[test]
    fn a_service_accept_without_the_ie_reconciles_nothing() {
        let mut orch = establish_security_context();
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);
        orch.state_mut().switch_cm_state(CmState::Idle);
        orch.start_service_request(ServiceType::Data, None);

        let mut plain = Vec::new();
        ServiceAccept::new().encode(&mut plain);
        let outs = orch.handle_downlink(&protect_downlink(&orch, &plain, 2));

        assert!(
            !outs
                .iter()
                .any(|o| matches!(o, MmOutput::NetworkPduSessionStatus(_))),
            "an absent IE must produce no reconciliation, got {outs:?}"
        );
        assert_eq!(
            orch.state().mm_substate(),
            MmSubState::RegisteredNormalService,
            "the Service Accept is still accepted"
        );
    }

    /// #51: a REGISTRATION ACCEPT carrying a PDU session status IE surfaces it
    /// too (TS 24.501 §5.5.1.3.4), and does so BEFORE `RegistrationSucceeded`.
    ///
    /// The ordering is load-bearing, not stylistic: the caller's
    /// `RegistrationSucceeded` arm establishes the configured default sessions
    /// only when `active_sessions()` is empty, so a reconciliation arriving after
    /// it would leave the stale sessions counted, and the UE would neither
    /// release them nor establish anything.
    #[test]
    fn a_registration_accept_surfaces_the_status_before_success() {
        let mut orch = establish_security_context();
        let mut acc = RegistrationAccept::new(Ie5gsRegistrationResult::new(
            SmsOverNasAllowed::NotAllowed,
            RegistrationResultValue::ThreeGppAccess,
        ));
        acc.tai_list = Some(vec![0x00, 0x99, 0xF9, 0x07, 0x00, 0x00, 0x01]);
        acc.pdu_session_status = Some(0x0002);
        let mut plain = Vec::new();
        acc.encode(&mut plain);

        let outs = orch.handle_downlink(&protect_downlink(&orch, &plain, 1));

        let status_at = outs
            .iter()
            .position(|o| matches!(o, MmOutput::NetworkPduSessionStatus(0x0002)))
            .expect("the PDU session status IE must reach the caller");
        let success_at = outs
            .iter()
            .position(|o| matches!(o, MmOutput::RegistrationSucceeded))
            .expect("registration must still succeed");
        assert!(
            status_at < success_at,
            "reconciliation must precede RegistrationSucceeded, got {outs:?}"
        );
    }

    /// #317, TS 24.501 §5.6.1.2: a second trigger while a service request is in
    /// progress (T3517 running, SERVICE-REQUEST-INITIATED) sends nothing -- one
    /// SERVICE REQUEST PDU, one UL NAS COUNT, T3517 not restarted.
    #[test]
    fn two_triggers_send_one_service_request() {
        let mut orch = establish_security_context();
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);
        orch.state_mut().switch_cm_state(CmState::Idle);

        let first = orch.start_service_request(ServiceType::Data, uplink_data_status(&[1]));
        assert_eq!(first.len(), 1, "{first:?}");
        let count_after_first = orch.sec.uplink_count().to_u32();
        assert!(orch.service_request_in_progress());

        for _ in 0..10 {
            let again = orch.start_service_request(ServiceType::Data, uplink_data_status(&[1]));
            assert!(again.is_empty(), "no duplicate SERVICE REQUEST: {again:?}");
        }
        assert_eq!(
            orch.sec.uplink_count().to_u32(),
            count_after_first,
            "no UL NAS COUNT spent on duplicates"
        );
        assert_eq!(
            orch.state().mm_substate(),
            MmSubState::ServiceRequestInitiated
        );
    }

    /// #316, TS 24.501 §5.6.1.7: a lower-layer release before the Service Accept
    /// aborts the procedure -- T3517 stopped, 5GMM-REGISTERED -- so after the UE
    /// reconnects (here: an RLF and a mobility registration elsewhere) it can ask
    /// for its user plane at once instead of waiting out T3517.
    #[test]
    fn a_release_during_a_service_request_stops_t3517() {
        let mut orch = establish_security_context();
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);
        orch.state_mut().switch_cm_state(CmState::Idle);
        assert!(!orch
            .start_service_request(ServiceType::Data, uplink_data_status(&[1]))
            .is_empty());
        assert!(orch.timers().t3517.is_running());

        orch.on_nas_signalling_connection_released();

        assert!(!orch.timers().t3517.is_running(), "T3517 stopped");
        assert_eq!(
            orch.state().mm_substate(),
            MmSubState::RegisteredNormalService
        );
        assert!(!orch.service_request_in_progress());
        // Reconnected without a user plane: a new Service Request may go out now.
        orch.state_mut().switch_cm_state(CmState::Connected);
        assert!(orch.user_plane_reactivation_needed());
        assert!(!orch
            .start_service_request(ServiceType::Data, uplink_data_status(&[1]))
            .is_empty());
    }

    /// #316, §5.6.1.7 l): an RRC establishment failure (the SERVICE REQUEST from
    /// CM-IDLE never got a connection) aborts the procedure too.
    #[test]
    fn an_rrc_establishment_failure_aborts_the_service_request() {
        let mut orch = establish_security_context();
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);
        orch.state_mut().switch_cm_state(CmState::Idle);
        assert!(!orch
            .start_service_request(ServiceType::Data, uplink_data_status(&[1]))
            .is_empty());
        orch.on_rrc_establishment_failure();
        assert!(!orch.timers().t3517.is_running());
        assert_eq!(
            orch.state().mm_substate(),
            MmSubState::RegisteredNormalService
        );
        assert!(
            !orch
                .start_service_request(ServiceType::Data, uplink_data_status(&[1]))
                .is_empty(),
            "the next uplink may try again"
        );
    }

    /// #316, TS 24.501 §5.6.1.7: a mobility registration triggered while a service
    /// request is in progress aborts it and stops T3517; the registration runs
    /// under T3510.
    #[test]
    fn a_mobility_registration_stops_t3517() {
        let mut orch = establish_security_context();
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);
        orch.state_mut().switch_cm_state(CmState::Connected);
        assert!(!orch
            .start_service_request(ServiceType::Data, uplink_data_status(&[1]))
            .is_empty());
        assert!(orch.timers().t3517.is_running());

        let outs = orch.start_registration(RegistrationType::MobilityRegistrationUpdating);

        assert!(!outs.is_empty(), "the registration request is sent");
        assert!(!orch.timers().t3517.is_running(), "T3517 stopped");
        assert!(orch.timers().t3510.is_running());
        assert_eq!(orch.state().mm_substate(), MmSubState::RegisteredInitiated);
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

    /// nextgsim #286: an AMF that restarted knows nothing of the UE. The UE,
    /// CM-IDLE with a stored security context, sends a Service Request on a NEW
    /// NAS signalling connection; the AMF can only answer with an UNPROTECTED
    /// SERVICE REJECT #9. No secure exchange exists on this connection yet, so
    /// §4.4.4.2's allow-list applies, the reject is processed, and the UE
    /// re-registers from scratch (§5.6.1.5). The UE used to discard it and sit
    /// out T3517, never recovering.
    #[test]
    fn an_unprotected_service_reject_9_on_a_new_connection_triggers_reregistration() {
        let mut orch = establish_security_context();
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);
        assert!(orch.stored_guti().is_some());
        orch.on_nas_signalling_connection_released();

        orch.start_service_request(ServiceType::Data, None);
        let mut rej_pdu = Vec::new();
        ServiceReject::new(MmCause::UeIdentityCannotBeDerived).encode(&mut rej_pdu);
        assert_eq!(rej_pdu[1] & 0x0F, 0, "sent in the clear");
        let outs = orch.handle_downlink(&rej_pdu);

        let pdu = first_sent_pdu(&outs);
        assert_eq!(pdu[2], u8::from(MmMessageType::RegistrationRequest));
        assert!(!orch.timers().t3517.is_running());
        assert!(orch.stored_guti().is_none());
        // 5GMM-DEREGISTERED holds no PDU sessions (§5.6.1.5): they are released
        // locally BEFORE the new registration, so the defaults come back after
        // its accept instead of lingering without an SM context at the new AMF.
        let release = outs
            .iter()
            .position(|o| *o == MmOutput::ReleasePduSessionsLocally);
        let send = outs
            .iter()
            .position(|o| matches!(o, MmOutput::SendNasPdu(_)));
        assert!(
            matches!((release, send), (Some(r), Some(s)) if r < s),
            "sessions released locally before the registration: {outs:?}"
        );
    }

    /// TS 24.501 §5.5.1.3.5: a MOBILITY registration rejected #9 (an AMF that
    /// restarted and lost the UE) deletes the 5G-GUTI, releases the PDU
    /// sessions locally and starts an initial registration with the SUCI. It
    /// used to be an abnormal failure: the same GUTI was retried five times,
    /// then T3502.
    #[test]
    fn a_mobility_registration_rejected_9_re_registers_from_scratch() {
        let mut orch = establish_security_context();
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);
        orch.on_nas_signalling_connection_released();
        orch.start_registration(RegistrationType::MobilityRegistrationUpdating);

        let rej = build_registration_reject_pdu(MmCause::UeIdentityCannotBeDerived, None);
        let outs = orch.handle_downlink(&rej);

        assert_eq!(
            outs.first(),
            Some(&MmOutput::ReleasePduSessionsLocally),
            "{outs:?}"
        );
        let pdu = first_sent_pdu(&outs);
        assert_eq!(pdu[2], u8::from(MmMessageType::RegistrationRequest));
        assert!(orch.stored_guti().is_none());
        assert!(
            !orch.timers().t3511.is_running(),
            "not the abnormal-case retry"
        );
    }

    /// The complement: on the connection where the secure exchange WAS
    /// established, the same unprotected reject is still discarded (§4.4.4.2).
    #[test]
    fn an_unprotected_service_reject_on_a_secured_connection_is_discarded() {
        let mut orch = establish_security_context();
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);
        let mut rej_pdu = Vec::new();
        ServiceReject::new(MmCause::UeIdentityCannotBeDerived).encode(&mut rej_pdu);
        assert!(orch.handle_downlink(&rej_pdu).is_empty());
        assert!(orch.stored_guti().is_some());
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

    /// Issue #266: a switch-off de-registration gets no DEREGISTRATION ACCEPT
    /// (TS 24.501 §5.5.2.2.2), so the UE must not wait in
    /// 5GMM-DEREGISTERED-INITIATED -- and still REGISTERED -- for one.
    #[test]
    fn a_switch_off_deregistration_leaves_the_ue_deregistered_without_an_accept() {
        let mut orch = establish_security_context();
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);
        assert!(orch.state().is_registered());

        let outs = orch.start_deregistration(true);
        assert_eq!(outs.len(), 1);
        assert!(matches!(outs[0], MmOutput::SendNasPdu(_)));
        assert!(!orch.timers().t3521.is_running(), "no answer is awaited");
        assert!(
            !orch.timers().t3512.is_running(),
            "no periodic registration update from a switched-off UE"
        );
        assert_eq!(orch.state().mm_substate(), MmSubState::Null);
        assert!(orch.state().is_deregistered());
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
        // Issue #272 (TS 24.501 §5.5.2.1): de-registered over 3GPP access, the UE
        // releases its PDU sessions locally, after the accept.
        assert!(
            matches!(
                outs.as_slice(),
                [MmOutput::SendNasPdu(_), MmOutput::ReleasePduSessionsLocally]
            ),
            "{outs:?}"
        );

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

    /// Issue #272: a de-registration for non-3GPP access only leaves the 3GPP
    /// sessions alone.
    #[test]
    fn a_non_3gpp_network_deregistration_keeps_the_pdu_sessions() {
        use nextgsim_nas::ies::ie1::{
            DeRegistrationAccessType, IeDeRegistrationType, ReRegistrationRequired, SwitchOff,
        };
        let mut orch = establish_security_context();
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);
        let mut req = Vec::new();
        nextgsim_nas::messages::mm::DeregistrationRequestUeTerminated::new(
            IeDeRegistrationType::new(
                DeRegistrationAccessType::NonThreeGppAccess,
                ReRegistrationRequired::NotRequired,
                SwitchOff::NormalDeRegistration,
            ),
        )
        .encode(&mut req);
        let protected = protect_downlink(&orch, &req, 2);
        let outs = orch.handle_downlink(&protected);
        assert!(
            !outs.contains(&MmOutput::ReleasePduSessionsLocally),
            "{outs:?}"
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
        // The request arrives on a NAS signalling connection.
        orch.state_mut().switch_cm_state(CmState::Connected);
        let protected = protect_downlink(&orch, &req, 2);
        let outs = orch.handle_downlink(&protected);

        let is_registration_request = |o: &MmOutput| match o {
            MmOutput::SendNasPdu(p) => {
                p.len() >= 3 && p[1] == 0x00 && p[2] == u8::from(MmMessageType::RegistrationRequest)
            }
            _ => false,
        };
        // Issue #265: only the DEREGISTRATION ACCEPT goes out on the connection
        // the AMF is about to release; a REGISTRATION REQUEST sent on it was
        // lost and the UE waited out T3510.
        assert_eq!(outs.len(), 2, "only the accept is sent now: {outs:?}");
        // Issue #272: the sessions are released locally with the de-registration.
        assert_eq!(outs[1], MmOutput::ReleasePduSessionsLocally);
        assert!(!outs.iter().any(is_registration_request));
        assert_eq!(
            orch.state().mm_substate(),
            MmSubState::DeregisteredNormalService
        );
        assert!(!orch.timers().t3510.is_running());

        // The release of that connection starts the initial registration, with
        // the context reset first, so it goes out unprotected.
        let outs = orch.on_nas_signalling_connection_released();
        assert!(
            outs.iter().any(is_registration_request),
            "re-registration must be initiated"
        );
        assert_eq!(orch.state().mm_substate(), MmSubState::RegisteredInitiated);
        assert!(orch.state().is_idle());

        // Once only: a later release starts nothing.
        assert!(orch.on_nas_signalling_connection_released().is_empty());
    }

    /// A release with no deferred re-registration pending only enters CM-IDLE.
    #[test]
    fn a_plain_connection_release_starts_no_registration() {
        let mut orch = establish_security_context();
        let acc = protect_downlink(&orch, &build_registration_accept_pdu(true), 1);
        orch.handle_downlink(&acc);
        orch.state_mut().switch_cm_state(CmState::Connected);
        assert!(orch.on_nas_signalling_connection_released().is_empty());
        assert!(orch.state().is_idle());
        assert!(orch.state().is_registered());
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

    // ========================================================================
    // UE STATE INDICATION in REGISTRATION REQUEST (TS 24.501 §5.5.1.2.2) — #48
    // ========================================================================

    /// The Payload container on the WIRE Registration Request, if any.
    ///
    /// Deciphers when the request is security protected, which is the case that
    /// matters here: the Payload container is NOT a cleartext IE (TS 24.501
    /// §4.4.6), so it only reaches the wire once a security context is active.
    fn registration_payload_container(
        orch: &mut MmOrchestrator,
    ) -> (Option<PayloadContainerType>, Option<Vec<u8>>) {
        let outs = orch.start_registration(RegistrationType::InitialRegistration);
        let pdu = first_sent_pdu(&outs).to_vec();
        let (_, req) = open_registration_request(orch, &pdu);
        (req.payload_container_type, req.payload_container)
    }

    /// The Payload container on the FULL request stashed for replay inside the
    /// SECURITY MODE COMPLETE NAS message container (TS 24.501 §4.4.6 a) 1)).
    fn stashed_payload_container(orch: &MmOrchestrator) -> Option<Vec<u8>> {
        let full = orch
            .last_registration_request
            .as_ref()
            .expect("the full request is stashed for the SMC container");
        RegistrationRequest::decode(&mut &full[3..])
            .expect("decodes")
            .payload_container
    }

    /// An orchestrator with an ACTIVE security context, ready to start a fresh
    /// registration -- which is what a RE-registration is, and the only case in
    /// which the Payload container reaches the wire (§4.4.6).
    ///
    /// `establish_security_context` leaves the state machine mid-procedure, so it
    /// is returned to DEREGISTERED; otherwise `start_registration` short-circuits
    /// on its "already in progress" guard and emits nothing.
    fn registered_orch() -> MmOrchestrator {
        let mut orch = establish_security_context();
        orch.state_mut().switch_mm_state(MmSubState::Deregistered);
        orch
    }

    /// Store a UE policy section for THIS UE's PLMN (999/70), so §5.5.1.2.2's
    /// "HPLMN or selected PLMN" condition is satisfied.
    ///
    /// The shipped E1 fixture stores at PLMN 001/01, which is deliberately NOT
    /// this UE's PLMN — see `a_section_for_another_plmn_is_not_reported`.
    fn store_section_for_own_plmn(orch: &mut MmOrchestrator, upsc: u16) {
        let mut cmd = E1_VEC_F_COMMAND.to_vec();
        // The command's sublist PLMN sits after PTI, type, list length (2) and
        // sublist length (2): octets 6..9.
        let own = encode_plmn_bcd(999, 70, false);
        cmd[6..9].copy_from_slice(&own);
        // ... and the instruction's UPSC after the PLMN: octets 11..13
        // (instruction length is at 9..11).
        cmd[11..13].copy_from_slice(&upsc.to_be_bytes());
        let reaction = orch.handle_ue_policy_command(&cmd);
        assert!(
            matches!(reaction, UePolicyReaction::Reply(_)),
            "the fixture must be accepted, got {reaction:?}"
        );
    }

    /// #48: a REGISTRATION REQUEST from a UE holding a policy section for its own
    /// PLMN carries the UE STATE INDICATION in the Payload container, with the
    /// Payload container type set to "UE policy container" (§5.5.1.2.2).
    #[test]
    fn a_stored_policy_section_is_reported_in_the_registration_request() {
        let mut orch = registered_orch();
        store_section_for_own_plmn(&mut orch, 4);
        assert_eq!(orch.ue_policy_section_count(), 1, "precondition");

        let (container_type, container) = registration_payload_container(&mut orch);
        assert_eq!(
            container_type,
            Some(PayloadContainerType::UePolicyContainer),
            "§8.2.6.17A NOTE: the only value this IE takes in a REGISTRATION REQUEST"
        );
        let container = container.expect("the Payload container must be present");

        // Decoded rather than byte-compared, so the assertion is about WHAT is
        // reported; the byte layout is pinned in nextgsim-nas's own tests.
        let indication =
            UeStateIndication::decode(&container).expect("a valid UE STATE INDICATION");
        assert!(
            (ue_policy::UE_PTI_MIN..=ue_policy::UE_PTI_MAX).contains(&indication.pti),
            "PTI {} must be in the UE-initiated range 01H-77H (D.1.2); 0 is not a PTI",
            indication.pti
        );
        assert_eq!(indication.upsi_list.len(), 1);
        assert_eq!(
            indication.upsi_list[0].plmn_id,
            PlmnId::from_bcd(encode_plmn_bcd(999, 70, false)).unwrap()
        );
        assert_eq!(
            indication.upsi_list[0].upscs,
            vec![4],
            "the UPSC the PCF assigned must be the one reported"
        );
        assert!(
            indication.classmark.andsp,
            "the UPDP codec stores an ANDSP part, so ANDSP is claimed"
        );
        assert!(
            !indication.classmark.ursp_in_eps
                && !indication.classmark.vps_ursp
                && !indication.classmark.report_ursp_rule_enforcement,
            "capabilities this tree does not have must not be claimed"
        );
    }

    /// #48: a clean first attach carries NO Payload container.
    ///
    /// This is the default path and the one that must not regress: §5.5.1.2.2's
    /// obligation is conditional on holding a stored section, and a container
    /// sent unconditionally would make every registration carry an IE with
    /// nothing to report.
    #[test]
    fn a_clean_first_attach_carries_no_payload_container() {
        let mut orch = registered_orch();
        assert_eq!(orch.ue_policy_section_count(), 0, "precondition");
        let (container_type, container) = registration_payload_container(&mut orch);
        assert_eq!(container, None, "no stored section, no container");
        assert_eq!(
            container_type, None,
            "§8.2.6.17A: the type IE exists only to describe a container"
        );
    }

    /// #48: a section stored for a DIFFERENT PLMN is not reported.
    ///
    /// §5.5.1.2.2 scopes the report to sections whose UPSI names the HPLMN or the
    /// selected PLMN. Reporting a foreign PLMN's UPSC would tell this PCF the UE
    /// holds a section it did not issue — and the shipped E1 fixture stores at
    /// 001/01 while this UE is on 999/70, so an unfiltered implementation passes
    /// the positive test above and fails here.
    #[test]
    fn a_section_for_another_plmn_is_not_reported() {
        let mut orch = registered_orch();
        orch.handle_ue_policy_command(E1_VEC_F_COMMAND);
        assert_eq!(
            orch.ue_policy_section_count(),
            1,
            "precondition: a section IS stored, just not for this PLMN"
        );
        assert!(
            orch.ue_policy_section(plmn_001_01(), 1).is_some(),
            "precondition: stored at 001/01"
        );

        let (_, container) = registration_payload_container(&mut orch);
        assert_eq!(
            container, None,
            "only HPLMN / selected-PLMN sections are reportable (§5.5.1.2.2)"
        );
    }

    /// #48: two sections for this PLMN are reported as two UPSCs in one sublist,
    /// in a deterministic order.
    ///
    /// The store is a `HashMap`, so without an explicit sort the encoded bytes
    /// would vary run to run — which would make any byte-level assertion, here or
    /// at a peer, flaky rather than wrong.
    #[test]
    fn multiple_sections_are_reported_in_a_stable_order() {
        let mut orch = registered_orch();
        // FOUR sections, stored in DESCENDING order. Two would make this guard a
        // coin flip -- an unsorted HashMap walk yields ascending order half the
        // time -- so it is four, where the odds of accidentally passing are 1/24.
        for upsc in [9u16, 7, 4, 2] {
            store_section_for_own_plmn(&mut orch, upsc);
        }
        assert_eq!(orch.ue_policy_section_count(), 4, "precondition");

        let (_, container) = registration_payload_container(&mut orch);
        let indication =
            UeStateIndication::decode(&container.expect("container")).expect("valid indication");
        assert_eq!(indication.upsi_list.len(), 1, "one PLMN, one sublist");
        assert_eq!(
            indication.upsi_list[0].upscs,
            vec![2, 4, 7, 9],
            "ascending UPSC: the encoded bytes must be a function of the stored state, not \
             of HashMap iteration order"
        );
    }

    /// #48: with NO security context the indication is NOT on the wire, but IS in
    /// the full request stashed for the SECURITY MODE COMPLETE container.
    ///
    /// This is where §5.5.1.2.2 and §4.4.6 meet, and the answer is not obvious.
    /// §5.5.1.2.2 says the *initial* REGISTRATION REQUEST shall carry the UE
    /// STATE INDICATION; §4.4.6 enumerates the cleartext IEs of a REGISTRATION
    /// REQUEST and **the Payload container is not among them**, so a UE with no
    /// valid 5G NAS security context may not send it in the clear. §4.4.6 a) 1)
    /// resolves it: the entire request, non-cleartext IEs included, is replayed
    /// inside the NAS message container of SECURITY MODE COMPLETE.
    ///
    /// So the obligation is met without ever putting a UPSI list — which names
    /// what policy this subscriber holds — on an unprotected radio link.
    #[test]
    fn without_security_the_indication_travels_in_the_smc_container_not_in_the_clear() {
        let mut orch = new_orch();
        store_section_for_own_plmn(&mut orch, 4);
        assert!(
            !orch.security_context().is_active(),
            "precondition: no security context"
        );

        let (container_type, container) = registration_payload_container(&mut orch);
        assert_eq!(
            container, None,
            "the Payload container is not a cleartext IE (§4.4.6): it must not be sent \
             unprotected"
        );
        assert_eq!(container_type, None);

        let stashed = stashed_payload_container(&orch)
            .expect("the full request must carry it for the SMC container (§4.4.6 a) 1))");
        let indication = UeStateIndication::decode(&stashed).expect("a valid indication");
        assert_eq!(indication.upsi_list[0].upscs, vec![4]);
    }

    /// #48: successive registrations use DIFFERENT PTIs (D.2.2.2 a): "allocate a
    /// PTI value currently not used").
    #[test]
    fn successive_indications_use_different_ptis() {
        let mut orch = registered_orch();
        store_section_for_own_plmn(&mut orch, 1);

        let first =
            UeStateIndication::decode(&registration_payload_container(&mut orch).1.unwrap())
                .unwrap()
                .pti;
        // Return to a state from which another registration can be started
        // (the same "already in progress" guard as in `registered_orch`).
        orch.state_mut().switch_mm_state(MmSubState::Deregistered);
        let second =
            UeStateIndication::decode(&registration_payload_container(&mut orch).1.unwrap())
                .unwrap()
                .pti;
        assert_ne!(first, second, "a reused PTI is not 'currently not used'");
        for pti in [first, second] {
            assert!((ue_policy::UE_PTI_MIN..=ue_policy::UE_PTI_MAX).contains(&pti));
        }
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
        assert!(orch.derive_kgnb_for_as_security(0).is_none());

        // Seed a KAMF; KgNB must equal derive_kgnb(KAMF, the COUNT handed in,
        // 3GPP-access) — the same value the AMF sends the gNB as SecurityKey
        // (TS 33.501 Annex A.9).
        let kamf = [0x22u8; 32];
        orch.sec.keys_mut().set_kamf(&kamf);
        assert_eq!(
            orch.derive_kgnb_for_as_security(0),
            Some(derive_kgnb(&kamf, 0, 0x01))
        );
        // And the COUNT is genuinely load-bearing: a different one yields a
        // different key, which is the whole reason issue #203's off-by-one was
        // fatal to the resumeMAC-I.
        assert_ne!(
            orch.derive_kgnb_for_as_security(0),
            orch.derive_kgnb_for_as_security(1)
        );
    }

    /// **Issue #203.** The KgNB the NAS plane hands RRC must be derived from the
    /// uplink NAS COUNT the Security Mode Complete was SENT under — not the
    /// post-increment value left behind once `protect_uplink` has consumed it.
    ///
    /// TS 33.501 §6.8.1.1.2.3: "The NAS SMC complete message shall include the
    /// start value of the uplink NAS COUNT that is used as freshness parameter in
    /// the KgNB derivation". The AMF derives from the COUNT it accepts off that
    /// message's own header (its `ul_count` is committed to the verified candidate),
    /// so the UE must use the same one.
    ///
    /// Asserted through the real `handle_downlink` SMC path, and positively: the
    /// handed-out key is compared against `derive_kgnb` at the COUNT actually
    /// carried in the emitted SMComplete's sequence-number octet, which is read off
    /// the PDU rather than assumed. Before the fix the key matched COUNT 1 while the
    /// PDU carried SQN 0.
    #[test]
    fn the_kgnb_uses_the_count_the_security_mode_complete_was_sent_under() {
        use nextgsim_crypto::kdf::derive_kgnb;

        let mut orch = MmOrchestrator::new(test_identity());
        let outs = orch.start_registration(RegistrationType::InitialRegistration);
        assert_eq!(outs.len(), 1);
        let auth_pdu = build_auth_request_pdu([0, 0, 0, 0, 0, 1], [0x80, 0x00]);
        let _ = orch.handle_downlink(&auth_pdu);

        let smc_pdu = build_protected_smc(&orch, 2, 2, None);
        let outs = orch.handle_downlink(&smc_pdu);

        // The SMComplete as it goes on the wire. Octet 6 is the sequence number
        // (TS 24.501 §9.1.1: EPD, SHT, 4-octet MAC, SQN), i.e. the low 8 bits of the
        // uplink NAS COUNT this message was protected with.
        let complete = first_sent_pdu(&outs);
        assert_eq!(
            complete[1], 0x04,
            "the SMComplete must use SHT 0x04 (new security context)"
        );
        let wire_sqn = complete[6];

        let handed = outs
            .iter()
            .find_map(|o| match o {
                MmOutput::AsSecurityKgnb(k) => Some(*k),
                _ => None,
            })
            .expect(
                "the SMComplete must hand the RRC plane a KgNB; without one the resume \
                 path has no K_RRCint at all (issue #201)",
            );

        let kamf = *orch.security_context().keys().kamf().expect("kamf");
        assert_eq!(
            handed,
            derive_kgnb(&kamf, u32::from(wire_sqn), 0x01),
            "the KgNB must be derived from the uplink NAS COUNT the SMComplete went out \
             under (SQN {wire_sqn} on the wire), per TS 33.501 §6.8.1.1.2.3. Matching \
             COUNT {} instead is issue #203's off-by-one: protect_uplink had already \
             incremented, so every AS key differed from the AMF's and the resumeMAC-I \
             of TS 38.331 §5.3.13.3 could never verify",
            u32::from(wire_sqn) + 1
        );

        // The post-send COUNT really is different, so the assertion above is not
        // vacuously satisfied by the two values happening to coincide.
        assert_ne!(
            orch.security_context().uplink_count().to_u32(),
            u32::from(wire_sqn),
            "protect_uplink must advance the COUNT, or there would be no off-by-one to \
             get wrong and this test would prove nothing"
        );
    }

    // ------------------------------------------------------------------------
    // UUAA-MM (#58, TS 23.256 §5.2.2 / TS 24.501 §9.11.2.10)
    // ------------------------------------------------------------------------

    /// A UE configured as an aerial UE with a UUAA payload available.
    fn aerial_ue_config() -> UeConfig {
        let mut config = UeConfig::default();
        config.uav_config = Some(nextgsim_common::config::UavConfig {
            is_aerial_ue: true,
            uav_id: Some("FAA-N12345".to_string()),
            uuaa_payload_hex: Some("a1b2c3".to_string()),
            ..Default::default()
        });
        config
    }

    fn slaa_response_container(slar: ServiceLevelAaResult) -> Vec<u8> {
        ServiceLevelAaContainer {
            response: Some(ServiceLevelAaResponse {
                slar,
                c2ar: ServiceLevelAaResult::NoInformation,
            }),
            ..Default::default()
        }
        .encode()
    }

    #[test]
    fn a_uuaa_success_reaches_the_uav_context_authorization_state() {
        // The wiring the issue asks for: a DL NAS TRANSPORT Service-level-AA
        // response must land on UavContext::auth_state, not in a private field.
        let mut orch = MmOrchestrator::from_config(test_identity(), &aerial_ue_config());
        assert_eq!(
            orch.uav_authorization_state(),
            Some(UavAuthorizationState::NotAuthorized)
        );

        orch.handle_service_level_aa_container(&slaa_response_container(
            ServiceLevelAaResult::Successful,
        ));

        assert_eq!(orch.uuaa_state(), UuaaState::Authorized);
        assert_eq!(
            orch.uav_authorization_state(),
            Some(UavAuthorizationState::Authorized)
        );
    }

    #[test]
    fn a_uuaa_revocation_reaches_the_uav_context_authorization_state() {
        // The direction that matters for safety: an authorized UAV told its
        // authorization is revoked must stop reading as authorized.
        let mut orch = MmOrchestrator::from_config(test_identity(), &aerial_ue_config());
        orch.handle_service_level_aa_container(&slaa_response_container(
            ServiceLevelAaResult::Successful,
        ));
        assert_eq!(
            orch.uav_authorization_state(),
            Some(UavAuthorizationState::Authorized)
        );

        orch.handle_service_level_aa_container(&slaa_response_container(
            ServiceLevelAaResult::NotSuccessfulOrRevoked,
        ));

        assert_eq!(orch.uuaa_state(), UuaaState::Rejected);
        assert_eq!(
            orch.uav_authorization_state(),
            Some(UavAuthorizationState::Revoked)
        );
    }

    #[test]
    fn a_non_aerial_ue_has_no_uav_context_and_no_uuaa_payload() {
        // A UE that is not a UAV must not answer a UUAA pending indication: it
        // would be claiming a UAS identity it does not have.
        let mut orch = MmOrchestrator::from_config(test_identity(), &UeConfig::default());
        assert_eq!(orch.uav_authorization_state(), None);

        let pending = ServiceLevelAaContainer {
            pending_indication: Some(true),
            ..Default::default()
        }
        .encode();
        assert_eq!(
            orch.handle_service_level_aa_container(&pending),
            UuaaReaction::Nothing
        );
    }

    #[test]
    fn an_aerial_ue_with_an_unparsable_uuaa_payload_sends_nothing() {
        // "a1b2c" is odd-length hex: not a payload. The UE must not send a
        // half-decoded or empty one.
        let mut config = aerial_ue_config();
        config.uav_config.as_mut().unwrap().uuaa_payload_hex = Some("a1b2c".to_string());
        let mut orch = MmOrchestrator::from_config(test_identity(), &config);

        let pending = ServiceLevelAaContainer {
            pending_indication: Some(true),
            ..Default::default()
        }
        .encode();
        assert_eq!(
            orch.handle_service_level_aa_container(&pending),
            UuaaReaction::Nothing
        );
        assert_eq!(orch.uuaa_state(), UuaaState::Pending);
        // The UAV is not authorized while an unanswerable request is pending.
        assert_eq!(
            orch.uav_authorization_state(),
            Some(UavAuthorizationState::AuthorizationRequested)
        );
    }

    /// The `uuaa-mm` gate, asserted from the live orchestrator in whichever
    /// configuration is being built. Both arms exist so neither is compiled
    /// without being checked -- the CI `uuaa-mm` job runs the enabled one.
    #[cfg(feature = "uuaa-mm")]
    #[test]
    fn with_uuaa_mm_enabled_a_pending_indication_produces_an_uplink() {
        let mut orch = MmOrchestrator::from_config(test_identity(), &aerial_ue_config());
        let pending = ServiceLevelAaContainer {
            pending_indication: Some(true),
            ..Default::default()
        }
        .encode();

        let UuaaReaction::SendUplink(pdu) = orch.handle_service_level_aa_container(&pending) else {
            panic!("with uuaa-mm on, a pending indication must produce an uplink");
        };
        // Plain UL NAS TRANSPORT with payload container type 9, carrying the
        // configured payload.
        assert_eq!(&pdu[..4], &[0x7E, 0x00, 0x67, 0x09]);
        assert_eq!(
            ServiceLevelAaContainer::decode(&pdu[6..])
                .payload
                .as_deref(),
            Some(&[0xA1, 0xB2, 0xC3][..])
        );
        assert_eq!(orch.uuaa_state(), UuaaState::Requested);
    }

    #[cfg(not(feature = "uuaa-mm"))]
    #[test]
    fn without_uuaa_mm_a_pending_indication_is_recorded_and_not_answered() {
        let mut orch = MmOrchestrator::from_config(test_identity(), &aerial_ue_config());
        let pending = ServiceLevelAaContainer {
            pending_indication: Some(true),
            ..Default::default()
        }
        .encode();

        assert_eq!(
            orch.handle_service_level_aa_container(&pending),
            UuaaReaction::Nothing
        );
        // Recorded, so the default build is honest about having been asked.
        assert_eq!(orch.uuaa_state(), UuaaState::Pending);
        assert_eq!(
            orch.uav_authorization_state(),
            Some(UavAuthorizationState::AuthorizationRequested)
        );
    }

    #[test]
    fn the_configured_caa_id_reaches_the_uav_identity() {
        let orch = MmOrchestrator::from_config(test_identity(), &aerial_ue_config());
        assert_eq!(
            orch.uav.as_ref().unwrap().identity.caa_level_id.as_deref(),
            Some("FAA-N12345")
        );
    }

    #[test]
    fn a_uuaa_payload_hex_round_trips_through_the_config_accessor() {
        let config = aerial_ue_config();
        let uav = config.uav_config.as_ref().unwrap();
        assert_eq!(uav.uuaa_payload(), Some(vec![0xA1, 0xB2, 0xC3]));

        // Absent, empty, odd-length and non-hex all yield no payload rather
        // than a partial one.
        for hex in [None, Some(""), Some("a1b2c"), Some("zz")] {
            let candidate = nextgsim_common::config::UavConfig {
                uuaa_payload_hex: hex.map(str::to_string),
                ..Default::default()
            };
            assert_eq!(candidate.uuaa_payload(), None, "hex={hex:?}");
        }
    }
    // ========================================================================
    // LPP positioning over the NAS payload container (#46, TS 37.355)
    // ========================================================================

    /// The LMF's own hand-derived reference vector (see `nas::lpp`): a
    /// `RequestLocationInformation` asking for RSRP and RSRQ, transaction 0.
    const LMF_LOCATION_REQUEST: &[u8] = &[0x90, 0x01, 0x20, 0x09, 0x30];

    #[test]
    fn an_lpp_container_produces_a_ul_nas_transport_from_the_orchestrator() {
        // The wiring the issue asks for: a DL NAS TRANSPORT LPP container must be
        // answered here rather than falling through to the SM orchestrator, which
        // returns nothing for it.
        use crate::nas::lpp::{LppBody, LppMessage, LppUplink, ServingCellMeasurements};
        use nextgsim_nas::ies::ie1::PayloadContainerType;
        use nextgsim_nas::messages::mm::UlNasTransport;

        let mut orch = new_orch();
        orch.set_serving_cell_measurements(ServingCellMeasurements {
            phys_cell_id: 7,
            arfcn: 1850,
            system_frame_number: Some(100),
            rsrp_result: Some(60),
            rsrq_result: None,
        });

        let LppUplink::Send(pdu) = orch.handle_lpp_container(LMF_LOCATION_REQUEST) else {
            panic!("an LPP location request must be answered");
        };
        assert_eq!(orch.lpp_replies_sent(), 1);

        let ul = UlNasTransport::decode(&mut &pdu[3..]).expect("decodes");
        assert_eq!(ul.payload_container_type, PayloadContainerType::LppMessage);
        let reply = LppMessage::decode(&ul.payload_container).expect("decodes");
        assert_eq!(reply.transaction_id.map(|t| t.transaction_number), Some(0));
        assert!(reply.end_transaction);
        match reply.body {
            Some(LppBody::ProvideLocationInformation { measurements, .. }) => {
                let element = measurements.primary_cell.expect("primary");
                assert_eq!(element.phys_cell_id, 7, "the cell RRC reported");
                assert_eq!(element.arfcn, 1850);
                assert_eq!(element.system_frame_number, Some(100));
                assert_eq!(element.rsrp_result, Some(60));
            }
            other => panic!("expected ProvideLocationInformation, got {other:?}"),
        }
    }

    /// The sidelink ranging report LEAVES THE UE (issue #137): it rides the LPP
    /// payload container of a real UL NAS TRANSPORT, which is what replaced the
    /// in-process `oneshot` behind `RangingMessage::ReportToLmf`.
    #[test]
    fn a_sidelink_ranging_report_leaves_the_ue_in_an_ul_nas_transport() {
        use crate::nas::lpp::{
            LppBody, LppMessage, LppUplink, ServingCellMeasurements, SidelinkRangingMethod,
            SidelinkRangingReport, SidelinkRangingResult,
        };
        use nextgsim_nas::ies::ie1::PayloadContainerType;
        use nextgsim_nas::messages::mm::UlNasTransport;

        let mut orch = new_orch();
        orch.set_serving_cell_measurements(ServingCellMeasurements {
            phys_cell_id: 7,
            arfcn: 1850,
            system_frame_number: None,
            rsrp_result: Some(60),
            rsrq_result: None,
        });
        let measured = SidelinkRangingResult {
            peer_layer2_id: 0x00A5_A5A5,
            range_cm: 5_000,
            accuracy_cm: 1,
            method: SidelinkRangingMethod::CarrierPhase,
            measurement_count: 3,
        };
        orch.set_sidelink_ranging_results(vec![measured]);

        let LppUplink::Send(pdu) = orch.handle_lpp_container(LMF_LOCATION_REQUEST) else {
            panic!("an LPP location request must be answered");
        };

        // A real NAS message, announced as an LPP container: an AMF routes on that
        // type, so a report in the wrong container would reach the wrong consumer.
        let ul = UlNasTransport::decode(&mut &pdu[3..]).expect("decodes");
        assert_eq!(ul.payload_container_type, PayloadContainerType::LppMessage);
        match LppMessage::decode(&ul.payload_container)
            .expect("decodes")
            .body
        {
            Some(LppBody::ProvideLocationInformation {
                measurements,
                sidelink,
            }) => {
                assert_eq!(
                    sidelink,
                    Some(SidelinkRangingReport {
                        results: vec![measured]
                    }),
                    "the measured range must survive the whole transport"
                );
                // And the E-CID half is still there: a UE that ranges over sidelink
                // also has a serving cell.
                assert_eq!(measurements.primary_cell.expect("primary").phys_cell_id, 7);
            }
            other => panic!("expected ProvideLocationInformation, got {other:?}"),
        }
    }

    /// With nothing measured the reply carries no report at all, rather than an empty
    /// one -- and the encoding is what it was before sidelink existed.
    #[test]
    fn a_ue_with_no_ranging_results_sends_the_same_report_it_always_did() {
        use crate::nas::lpp::{LppBody, LppMessage, LppUplink, ServingCellMeasurements};
        use nextgsim_nas::messages::mm::UlNasTransport;

        let mut orch = new_orch();
        orch.set_serving_cell_measurements(ServingCellMeasurements {
            phys_cell_id: 7,
            arfcn: 1850,
            system_frame_number: None,
            rsrp_result: Some(60),
            rsrq_result: None,
        });
        assert!(
            orch.sidelink_ranging_results().is_empty(),
            "precondition: nothing measured"
        );

        let LppUplink::Send(pdu) = orch.handle_lpp_container(LMF_LOCATION_REQUEST) else {
            panic!("answered");
        };
        let ul = UlNasTransport::decode(&mut &pdu[3..]).expect("decodes");
        assert!(
            matches!(
                LppMessage::decode(&ul.payload_container)
                    .expect("decodes")
                    .body,
                Some(LppBody::ProvideLocationInformation { sidelink: None, .. })
            ),
            "no report, not an empty one"
        );
    }

    #[test]
    fn the_lpp_report_carries_the_measurements_rrc_last_reported() {
        // The cache is what makes the report a measurement rather than a guess, so a
        // later report must supersede an earlier one.
        use crate::nas::lpp::{LppBody, LppMessage, LppUplink, ServingCellMeasurements};
        use nextgsim_nas::messages::mm::UlNasTransport;

        let mut orch = new_orch();
        let mut reported = Vec::new();
        for (pci, rsrp) in [(1u16, 30u8), (2, 90)] {
            orch.set_serving_cell_measurements(ServingCellMeasurements {
                phys_cell_id: pci,
                arfcn: 1850,
                system_frame_number: None,
                rsrp_result: Some(rsrp),
                rsrq_result: None,
            });
            let LppUplink::Send(pdu) = orch.handle_lpp_container(LMF_LOCATION_REQUEST) else {
                panic!("answered");
            };
            let ul = UlNasTransport::decode(&mut &pdu[3..]).expect("decodes");
            match LppMessage::decode(&ul.payload_container)
                .expect("decodes")
                .body
            {
                Some(LppBody::ProvideLocationInformation { measurements, .. }) => {
                    let element = measurements.primary_cell.expect("primary");
                    reported.push((element.phys_cell_id, element.rsrp_result));
                }
                other => panic!("expected ProvideLocationInformation, got {other:?}"),
            }
        }
        assert_eq!(reported, vec![(1, Some(30)), (2, Some(90))]);
    }

    #[test]
    fn an_lpp_capability_request_is_answered_before_any_measurement_arrives() {
        // A capability transfer does not depend on the radio, so it must not wait for
        // one -- TS 37.355 §5.1.3 has the server ask this first.
        use crate::nas::lpp::{LppBody, LppMessage, LppUplink};
        use nextgsim_nas::messages::mm::UlNasTransport;

        let mut orch = new_orch();
        assert_eq!(
            orch.serving_cell_measurements(),
            crate::nas::lpp::ServingCellMeasurements::default(),
            "nothing has been measured yet"
        );
        // requestCapabilities (c1 index 0) with the E-CID body, transaction 0.
        let request = {
            use crate::nas::lpp::UperWriter;
            let mut w = UperWriter::new();
            w.write_sequence_preamble(None, &[true, false, false, true]);
            w.write_sequence_preamble(Some(false), &[]);
            w.write_extensible_enumerated(0, 1).expect("root");
            w.write_constrained(0, 0, 255).expect("in range");
            w.write_bit(false);
            w.write_choice_index(0, 2).expect("c1");
            w.write_choice_index(0, 16).expect("requestCapabilities");
            w.write_choice_index(0, 2).expect("critExt c1");
            w.write_choice_index(0, 4).expect("r9");
            w.write_sequence_preamble(Some(false), &[false, false, false, true, false]);
            w.write_sequence_preamble(Some(false), &[]);
            w.into_bytes()
        };
        let LppUplink::Send(pdu) = orch.handle_lpp_container(&request) else {
            panic!("a capability request must be answered");
        };
        let ul = UlNasTransport::decode(&mut &pdu[3..]).expect("decodes");
        assert!(matches!(
            LppMessage::decode(&ul.payload_container)
                .expect("decodes")
                .body,
            Some(LppBody::ProvideCapabilities { .. })
        ));
    }

    #[test]
    fn an_undecodable_lpp_container_sends_nothing() {
        use crate::nas::lpp::LppUplink;
        let mut orch = new_orch();
        assert_eq!(orch.handle_lpp_container(&[0xFF; 6]), LppUplink::Nothing);
        assert_eq!(orch.lpp_replies_sent(), 0);
    }
}
