//! The EPS NAS messages of attach, tracking area update, identification,
//! authentication, security mode control and EPS bearer context management
//! (TS 24.301 §8.2, §8.3), as a UE sends and receives them.
//!
//! Uplink messages are built from values; downlink messages are parsed into
//! values. The plain message is what these functions see -- security
//! protection (§9.3, the 6-octet security header) is [`crate::security`]'s.

use crate::ie::{
    decode_apn, decode_pdn_address, decode_tai_list, encode_apn, gprs_timer3_seconds,
    gprs_timer_seconds, put_lv, put_lv_e, put_tlv, EpsBearerContextStatus, EpsMobileIdentity, Guti,
    MobileIdentity, NasDecodeError, NasKsi, PdnAddress, PdnType, Reader, Tai, UeNetworkCapability,
};

/// EPS mobility management protocol discriminator (TS 24.007 §11.2.3.1.1).
pub const PD_EMM: u8 = 0x07;
/// EPS session management protocol discriminator.
pub const PD_ESM: u8 = 0x02;

/// EMM message types (§9.8, Table 9.8.1).
pub mod emm_type {
    /// ATTACH REQUEST.
    pub const ATTACH_REQUEST: u8 = 0x41;
    /// ATTACH ACCEPT.
    pub const ATTACH_ACCEPT: u8 = 0x42;
    /// ATTACH COMPLETE.
    pub const ATTACH_COMPLETE: u8 = 0x43;
    /// ATTACH REJECT.
    pub const ATTACH_REJECT: u8 = 0x44;
    /// DETACH REQUEST.
    pub const DETACH_REQUEST: u8 = 0x45;
    /// DETACH ACCEPT.
    pub const DETACH_ACCEPT: u8 = 0x46;
    /// TRACKING AREA UPDATE REQUEST.
    pub const TAU_REQUEST: u8 = 0x48;
    /// TRACKING AREA UPDATE ACCEPT.
    pub const TAU_ACCEPT: u8 = 0x49;
    /// TRACKING AREA UPDATE COMPLETE.
    pub const TAU_COMPLETE: u8 = 0x4A;
    /// TRACKING AREA UPDATE REJECT.
    pub const TAU_REJECT: u8 = 0x4B;
    /// SERVICE REJECT.
    pub const SERVICE_REJECT: u8 = 0x4E;
    /// GUTI REALLOCATION COMMAND.
    pub const GUTI_REALLOCATION_COMMAND: u8 = 0x50;
    /// AUTHENTICATION REQUEST.
    pub const AUTHENTICATION_REQUEST: u8 = 0x52;
    /// AUTHENTICATION RESPONSE.
    pub const AUTHENTICATION_RESPONSE: u8 = 0x53;
    /// AUTHENTICATION REJECT.
    pub const AUTHENTICATION_REJECT: u8 = 0x54;
    /// IDENTITY REQUEST.
    pub const IDENTITY_REQUEST: u8 = 0x55;
    /// IDENTITY RESPONSE.
    pub const IDENTITY_RESPONSE: u8 = 0x56;
    /// AUTHENTICATION FAILURE.
    pub const AUTHENTICATION_FAILURE: u8 = 0x5C;
    /// SECURITY MODE COMMAND.
    pub const SECURITY_MODE_COMMAND: u8 = 0x5D;
    /// SECURITY MODE COMPLETE.
    pub const SECURITY_MODE_COMPLETE: u8 = 0x5E;
    /// SECURITY MODE REJECT.
    pub const SECURITY_MODE_REJECT: u8 = 0x5F;
    /// EMM STATUS.
    pub const EMM_STATUS: u8 = 0x60;
    /// EMM INFORMATION.
    pub const EMM_INFORMATION: u8 = 0x61;
    /// DOWNLINK NAS TRANSPORT.
    pub const DOWNLINK_NAS_TRANSPORT: u8 = 0x62;
}

/// ESM message types (§9.8, Table 9.8.2).
pub mod esm_type {
    /// ACTIVATE DEFAULT EPS BEARER CONTEXT REQUEST.
    pub const ACTIVATE_DEFAULT_REQUEST: u8 = 0xC1;
    /// ACTIVATE DEFAULT EPS BEARER CONTEXT ACCEPT.
    pub const ACTIVATE_DEFAULT_ACCEPT: u8 = 0xC2;
    /// ACTIVATE DEFAULT EPS BEARER CONTEXT REJECT.
    pub const ACTIVATE_DEFAULT_REJECT: u8 = 0xC3;
    /// ACTIVATE DEDICATED EPS BEARER CONTEXT REQUEST.
    pub const ACTIVATE_DEDICATED_REQUEST: u8 = 0xC5;
    /// ACTIVATE DEDICATED EPS BEARER CONTEXT ACCEPT.
    pub const ACTIVATE_DEDICATED_ACCEPT: u8 = 0xC6;
    /// ACTIVATE DEDICATED EPS BEARER CONTEXT REJECT.
    pub const ACTIVATE_DEDICATED_REJECT: u8 = 0xC7;
    /// MODIFY EPS BEARER CONTEXT REQUEST.
    pub const MODIFY_REQUEST: u8 = 0xC9;
    /// MODIFY EPS BEARER CONTEXT ACCEPT.
    pub const MODIFY_ACCEPT: u8 = 0xCA;
    /// MODIFY EPS BEARER CONTEXT REJECT.
    pub const MODIFY_REJECT: u8 = 0xCB;
    /// DEACTIVATE EPS BEARER CONTEXT REQUEST.
    pub const DEACTIVATE_REQUEST: u8 = 0xCD;
    /// DEACTIVATE EPS BEARER CONTEXT ACCEPT.
    pub const DEACTIVATE_ACCEPT: u8 = 0xCE;
    /// PDN CONNECTIVITY REQUEST.
    pub const PDN_CONNECTIVITY_REQUEST: u8 = 0xD0;
    /// PDN CONNECTIVITY REJECT.
    pub const PDN_CONNECTIVITY_REJECT: u8 = 0xD1;
    /// BEARER RESOURCE MODIFICATION REQUEST (Table 9.8.2: 1101 0110).
    pub const BEARER_RESOURCE_MODIFICATION_REQUEST: u8 = 0xD6;
    /// BEARER RESOURCE MODIFICATION REJECT (Table 9.8.2: 1101 0111).
    pub const BEARER_RESOURCE_MODIFICATION_REJECT: u8 = 0xD7;
    /// ESM INFORMATION REQUEST.
    pub const ESM_INFORMATION_REQUEST: u8 = 0xD9;
    /// ESM INFORMATION RESPONSE.
    pub const ESM_INFORMATION_RESPONSE: u8 = 0xDA;
    /// ESM STATUS.
    pub const ESM_STATUS: u8 = 0xE8;
}

/// EMM causes the UE sends or acts on (§9.9.3.9, Annex A).
pub mod emm_cause {
    /// #3 Illegal UE.
    pub const ILLEGAL_UE: u8 = 3;
    /// #6 Illegal ME.
    pub const ILLEGAL_ME: u8 = 6;
    /// #7 EPS services not allowed.
    pub const EPS_SERVICES_NOT_ALLOWED: u8 = 7;
    /// #8 EPS services and non-EPS services not allowed.
    pub const EPS_AND_NON_EPS_NOT_ALLOWED: u8 = 8;
    /// #9 UE identity cannot be derived by the network.
    pub const UE_IDENTITY_CANNOT_BE_DERIVED: u8 = 9;
    /// #10 Implicitly detached.
    pub const IMPLICITLY_DETACHED: u8 = 10;
    /// #11 PLMN not allowed.
    pub const PLMN_NOT_ALLOWED: u8 = 11;
    /// #12 Tracking area not allowed.
    pub const TA_NOT_ALLOWED: u8 = 12;
    /// #13 Roaming not allowed in this tracking area.
    pub const ROAMING_NOT_ALLOWED_IN_TA: u8 = 13;
    /// #14 EPS services not allowed in this PLMN.
    pub const EPS_NOT_ALLOWED_IN_PLMN: u8 = 14;
    /// #15 No suitable cells in tracking area.
    pub const NO_SUITABLE_CELLS_IN_TA: u8 = 15;
    /// #20 MAC failure.
    pub const MAC_FAILURE: u8 = 20;
    /// #21 Synch failure.
    pub const SYNCH_FAILURE: u8 = 21;
    /// #22 Congestion.
    pub const CONGESTION: u8 = 22;
    /// #23 UE security capabilities mismatch.
    pub const UE_SECURITY_CAPABILITIES_MISMATCH: u8 = 23;
    /// #24 Security mode rejected, unspecified.
    pub const SECURITY_MODE_REJECTED_UNSPECIFIED: u8 = 24;
    /// #25 Not authorized for this CSG.
    pub const NOT_AUTHORIZED_FOR_CSG: u8 = 25;
    /// #26 Non-EPS authentication unacceptable.
    pub const NON_EPS_AUTHENTICATION_UNACCEPTABLE: u8 = 26;
    /// #35 Requested service option not authorized in this PLMN.
    pub const SERVICE_OPTION_NOT_AUTHORIZED_IN_PLMN: u8 = 35;
    /// #40 No EPS bearer context activated.
    pub const NO_EPS_BEARER_CONTEXT_ACTIVATED: u8 = 40;
    /// #96 Invalid mandatory information.
    pub const INVALID_MANDATORY_INFORMATION: u8 = 96;
    /// #97 Message type non-existent or not implemented.
    pub const MESSAGE_TYPE_NOT_IMPLEMENTED: u8 = 97;
}

/// The EPS attach type (§9.9.3.11).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpsAttachType {
    /// EPS attach (1).
    Eps,
    /// Combined EPS/IMSI attach (2).
    CombinedEpsImsi,
    /// EPS emergency attach (6).
    Emergency,
}

impl EpsAttachType {
    fn value(self) -> u8 {
        match self {
            Self::Eps => 1,
            Self::CombinedEpsImsi => 2,
            Self::Emergency => 6,
        }
    }
}

/// ATTACH REQUEST (§8.2.4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttachRequest {
    /// NAS key set identifier ([`NasKsi::NO_KEY`] without a security context).
    pub ksi: NasKsi,
    /// EPS attach type.
    pub attach_type: EpsAttachType,
    /// IMSI, or the GUTI when the UE holds one (§5.5.1.2.2).
    pub identity: EpsMobileIdentity,
    /// UE network capability.
    pub ue_network_capability: UeNetworkCapability,
    /// The piggybacked PDN CONNECTIVITY REQUEST.
    pub esm_message: Vec<u8>,
    /// Last visited registered TAI (IEI 52), with a GUTI.
    pub last_visited_tai: Option<Tai>,
}

impl AttachRequest {
    /// Encode the plain message.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = vec![PD_EMM, emm_type::ATTACH_REQUEST];
        // NAS key set identifier in bits 8-5, EPS attach type in bits 4-1.
        out.push((self.ksi.nibble() << 4) | self.attach_type.value());
        put_lv(&mut out, &self.identity.encode());
        put_lv(&mut out, &self.ue_network_capability.encode());
        put_lv_e(&mut out, &self.esm_message);
        if let Some(tai) = self.last_visited_tai {
            // IEI 52, TV 6 (§9.9.3.32).
            out.push(0x52);
            out.extend_from_slice(&tai.plmn.encode());
            out.extend_from_slice(&tai.tac.to_be_bytes());
        }
        out
    }
}

/// The EPS update type value (§9.9.3.14, bits 3-1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpsUpdateType {
    /// TA updating (000).
    TaUpdating,
    /// Periodic updating (011).
    Periodic,
}

/// TRACKING AREA UPDATE REQUEST (§8.2.29), the IEs this UE sends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrackingAreaUpdateRequest {
    /// EPS update type.
    pub update_type: EpsUpdateType,
    /// The "active" flag (§9.9.3.14 bit 4): bearer establishment requested.
    pub active: bool,
    /// NAS key set identifier.
    pub ksi: NasKsi,
    /// Old GUTI (§5.5.3.2.2: the valid native GUTI).
    pub old_guti: Guti,
    /// UE network capability (IEI 58); §8.2.29.7: not on a periodic update.
    pub ue_network_capability: Option<UeNetworkCapability>,
    /// Last visited registered TAI (IEI 52).
    pub last_visited_tai: Option<Tai>,
    /// EPS bearer context status (IEI 57).
    pub eps_bearer_context_status: Option<EpsBearerContextStatus>,
}

impl TrackingAreaUpdateRequest {
    /// Encode the plain message. Optional IEs go in Table 8.2.29.1 order; the
    /// Old GUTI type (IEI E-, §9.9.3.45: "native GUTI") is always included
    /// (§8.2.29.20).
    pub fn encode(&self) -> Vec<u8> {
        let mut out = vec![PD_EMM, emm_type::TAU_REQUEST];
        let update_type = match self.update_type {
            EpsUpdateType::TaUpdating => 0b000,
            EpsUpdateType::Periodic => 0b011,
        };
        // NAS key set identifier in bits 8-5; the EPS update type, the first
        // half-octet IE, in bits 4-1 with the "active" flag in bit 4.
        out.push((self.ksi.nibble() << 4) | (u8::from(self.active) << 3) | update_type);
        put_lv(&mut out, &EpsMobileIdentity::Guti(self.old_guti).encode());
        if let Some(c) = &self.ue_network_capability {
            put_tlv(&mut out, 0x58, &c.encode());
        }
        if let Some(tai) = self.last_visited_tai {
            out.push(0x52);
            out.extend_from_slice(&tai.plmn.encode());
            out.extend_from_slice(&tai.tac.to_be_bytes());
        }
        if let Some(s) = self.eps_bearer_context_status {
            put_tlv(&mut out, 0x57, &s.encode());
        }
        out.push(0xE0);
        out
    }
}

/// TRACKING AREA UPDATE COMPLETE (§8.2.27).
pub fn encode_tau_complete() -> Vec<u8> {
    vec![PD_EMM, emm_type::TAU_COMPLETE]
}

/// ATTACH COMPLETE (§8.2.2).
pub fn encode_attach_complete(esm_message: &[u8]) -> Vec<u8> {
    let mut out = vec![PD_EMM, emm_type::ATTACH_COMPLETE];
    put_lv_e(&mut out, esm_message);
    out
}

/// IDENTITY RESPONSE (§8.2.19).
pub fn encode_identity_response(identity: &MobileIdentity) -> Vec<u8> {
    let mut out = vec![PD_EMM, emm_type::IDENTITY_RESPONSE];
    put_lv(&mut out, &identity.encode());
    out
}

/// AUTHENTICATION RESPONSE (§8.2.8).
pub fn encode_authentication_response(res: &[u8]) -> Vec<u8> {
    let mut out = vec![PD_EMM, emm_type::AUTHENTICATION_RESPONSE];
    put_lv(&mut out, res);
    out
}

/// AUTHENTICATION FAILURE (§8.2.5): the cause, and for #21 the AUTS in the
/// authentication failure parameter (IEI 30, TS 24.008 §10.5.3.2.2).
pub fn encode_authentication_failure(cause: u8, auts: Option<&[u8; 14]>) -> Vec<u8> {
    let mut out = vec![PD_EMM, emm_type::AUTHENTICATION_FAILURE, cause];
    if let Some(auts) = auts {
        put_tlv(&mut out, 0x30, auts);
    }
    out
}

/// SECURITY MODE COMPLETE (§8.2.21), with the IMEISV (IEI 23) when the
/// command requested it.
pub fn encode_security_mode_complete(imeisv: Option<&MobileIdentity>) -> Vec<u8> {
    let mut out = vec![PD_EMM, emm_type::SECURITY_MODE_COMPLETE];
    if let Some(id) = imeisv {
        put_tlv(&mut out, 0x23, &id.encode());
    }
    out
}

/// DETACH ACCEPT, UE to network (§8.2.10.1).
pub fn encode_detach_accept() -> Vec<u8> {
    vec![PD_EMM, emm_type::DETACH_ACCEPT]
}

/// DETACH REQUEST, UE to network (§8.2.11.1, #365): "EPS detach" (Type of
/// detach 001, §9.9.3.7), with the switch-off bit (bit 4) as asked. The detach
/// type is the first half-octet IE, so it takes bits 4 to 1 of octet 3, and
/// the NAS key set identifier takes bits 8 to 5. Then the EPS mobile identity
/// (LV, §9.9.3.12).
pub fn encode_detach_request(
    switch_off: bool,
    ksi: NasKsi,
    identity: &EpsMobileIdentity,
) -> Vec<u8> {
    let detach_type = 0x01 | if switch_off { 0x08 } else { 0x00 };
    let mut out = vec![
        PD_EMM,
        emm_type::DETACH_REQUEST,
        (ksi.nibble() << 4) | detach_type,
    ];
    put_lv(&mut out, &identity.encode());
    out
}

/// SECURITY MODE REJECT (§8.2.22).
pub fn encode_security_mode_reject(cause: u8) -> Vec<u8> {
    vec![PD_EMM, emm_type::SECURITY_MODE_REJECT, cause]
}

/// EMM STATUS (§8.2.14).
pub fn encode_emm_status(cause: u8) -> Vec<u8> {
    vec![PD_EMM, emm_type::EMM_STATUS, cause]
}

/// PDN CONNECTIVITY REQUEST (§8.3.20).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PdnConnectivityRequest {
    /// Procedure transaction identity (1..254, §9.4).
    pub pti: u8,
    /// Requested PDN type.
    pub pdn_type: PdnType,
    /// Access point name (IEI 28); absent for the default APN.
    pub apn: Option<String>,
}

/// Protocol configuration options container identifiers the UE requests
/// (TS 24.008 §10.5.6.3, Table 10.5.154).
pub mod pco {
    /// DNS Server IPv4 Address Request (MS to network) / DNS Server IPv4
    /// Address (network to MS).
    pub const DNS_SERVER_IPV4: u16 = 0x000D;
    /// IP address allocation via NAS signalling.
    pub const IP_ADDRESS_ALLOCATION_VIA_NAS: u16 = 0x000A;
}

/// The PCO a UE sends with its PDN CONNECTIVITY REQUEST (IEI 27, §9.9.4.11 ->
/// TS 24.008 §10.5.6.3): configuration protocol PPP (octet 3 = 1000 0000) and,
/// as MS-to-network containers with no contents, "DNS Server IPv4 Address
/// Request" and "IP address allocation via NAS signalling" -- what a UE needs
/// to be told its resolver and to take the PDN address from NAS rather than
/// DHCP.
pub fn ue_pco() -> Vec<u8> {
    let mut v = vec![0x80];
    for id in [pco::DNS_SERVER_IPV4, pco::IP_ADDRESS_ALLOCATION_VIA_NAS] {
        v.extend_from_slice(&id.to_be_bytes());
        v.push(0);
    }
    v
}

/// The IPv4 DNS servers in a network-to-MS PCO value.
pub fn pco_dns_ipv4(v: &[u8]) -> Vec<std::net::Ipv4Addr> {
    let mut out = Vec::new();
    let mut r = Reader::new(v.get(1..).unwrap_or_default());
    while r.remaining() >= 3 {
        let (Ok(id), Ok(len)) = (r.bytes(2), r.u8()) else {
            break;
        };
        let Ok(c) = r.bytes(usize::from(len)) else {
            break;
        };
        if u16::from_be_bytes([id[0], id[1]]) == pco::DNS_SERVER_IPV4 && c.len() == 4 {
            out.push(std::net::Ipv4Addr::new(c[0], c[1], c[2], c[3]));
        }
    }
    out
}

impl PdnConnectivityRequest {
    /// Encode the plain message. Request type is "initial request" (1,
    /// §9.9.4.14), the request type of a PDN connection set up by attach; the
    /// PCO is [`ue_pco`].
    pub fn encode(&self) -> Vec<u8> {
        // EPS bearer identity 0 ("no EPS bearer identity assigned", §9.3.2).
        let mut out = vec![PD_ESM, self.pti, esm_type::PDN_CONNECTIVITY_REQUEST];
        // PDN type in bits 8-5, request type in bits 4-1: one octet together.
        out.push((self.pdn_type.value() << 4) | 0x01);
        if let Some(apn) = &self.apn {
            put_tlv(&mut out, 0x28, &encode_apn(apn));
        }
        put_tlv(&mut out, 0x27, &ue_pco());
        out
    }
}

/// ACTIVATE DEFAULT EPS BEARER CONTEXT ACCEPT (§8.3.4).
pub fn encode_activate_default_accept(ebi: u8, pti: u8) -> Vec<u8> {
    vec![(ebi << 4) | PD_ESM, pti, esm_type::ACTIVATE_DEFAULT_ACCEPT]
}

/// ACTIVATE DEFAULT EPS BEARER CONTEXT REJECT (§8.3.5).
pub fn encode_activate_default_reject(ebi: u8, pti: u8, esm_cause: u8) -> Vec<u8> {
    vec![
        (ebi << 4) | PD_ESM,
        pti,
        esm_type::ACTIVATE_DEFAULT_REJECT,
        esm_cause,
    ]
}

/// ACTIVATE DEDICATED EPS BEARER CONTEXT REQUEST (§8.3.3), the parts the UE
/// acts on (#283).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivateDedicatedBearerRequest {
    /// The new bearer's EPS bearer identity.
    pub ebi: u8,
    /// PTI: 0 for a network-initiated activation.
    pub pti: u8,
    /// The default bearer of the PDN connection it belongs to.
    pub linked_ebi: u8,
    /// QCI (first octet of the EPS QoS).
    pub qci: u8,
    /// The TFT IE value, decoded by [`crate::tft::Tft::decode`].
    pub tft: Vec<u8>,
}

/// MODIFY EPS BEARER CONTEXT REQUEST (§8.3.18), the parts the UE acts on
/// (#283).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModifyBearerRequest {
    /// The bearer to modify.
    pub ebi: u8,
    /// PTI.
    pub pti: u8,
    /// New QCI (IEI 5B, New EPS QoS).
    pub qci: Option<u8>,
    /// The TFT IE value (IEI 36).
    pub tft: Option<Vec<u8>>,
}

/// ACTIVATE DEDICATED EPS BEARER CONTEXT ACCEPT (§8.3.1).
pub fn encode_activate_dedicated_accept(ebi: u8, pti: u8) -> Vec<u8> {
    vec![
        (ebi << 4) | PD_ESM,
        pti,
        esm_type::ACTIVATE_DEDICATED_ACCEPT,
    ]
}

/// ACTIVATE DEDICATED EPS BEARER CONTEXT REJECT (§8.3.2).
pub fn encode_activate_dedicated_reject(ebi: u8, pti: u8, esm_cause: u8) -> Vec<u8> {
    vec![
        (ebi << 4) | PD_ESM,
        pti,
        esm_type::ACTIVATE_DEDICATED_REJECT,
        esm_cause,
    ]
}

/// MODIFY EPS BEARER CONTEXT ACCEPT (§8.3.16).
pub fn encode_modify_accept(ebi: u8, pti: u8) -> Vec<u8> {
    vec![(ebi << 4) | PD_ESM, pti, esm_type::MODIFY_ACCEPT]
}

/// MODIFY EPS BEARER CONTEXT REJECT (§8.3.17).
pub fn encode_modify_reject(ebi: u8, pti: u8, esm_cause: u8) -> Vec<u8> {
    vec![(ebi << 4) | PD_ESM, pti, esm_type::MODIFY_REJECT, esm_cause]
}

/// BEARER RESOURCE MODIFICATION REQUEST (§8.3.10, #408): EPS bearer
/// identity 0 in the header (§9.3.2: a UE-requested procedure), the PTI, the
/// EPS bearer identity for packet filter in bits 4-1 with a spare half octet
/// in bits 8-5, the LV Traffic flow aggregate description (§9.9.4.15, coded
/// as the TFT of TS 24.008 §10.5.6.12), and the optional ESM cause (TV 58).
pub fn encode_bearer_resource_modification_request(
    pti: u8,
    ebi_for_packet_filter: u8,
    traffic_flow_aggregate: &[u8],
    esm_cause: Option<u8>,
) -> Vec<u8> {
    let mut out = vec![
        PD_ESM,
        pti,
        esm_type::BEARER_RESOURCE_MODIFICATION_REQUEST,
        ebi_for_packet_filter & 0x0F,
        traffic_flow_aggregate.len() as u8,
    ];
    out.extend_from_slice(traffic_flow_aggregate);
    if let Some(c) = esm_cause {
        out.extend_from_slice(&[0x58, c]);
    }
    out
}

/// DEACTIVATE EPS BEARER CONTEXT ACCEPT (§8.3.11).
pub fn encode_deactivate_accept(ebi: u8, pti: u8) -> Vec<u8> {
    vec![(ebi << 4) | PD_ESM, pti, esm_type::DEACTIVATE_ACCEPT]
}

/// ESM INFORMATION RESPONSE (§8.3.14), with the APN (IEI 28) when one is
/// configured.
pub fn encode_esm_information_response(pti: u8, apn: Option<&str>) -> Vec<u8> {
    let mut out = vec![PD_ESM, pti, esm_type::ESM_INFORMATION_RESPONSE];
    if let Some(apn) = apn {
        put_tlv(&mut out, 0x28, &encode_apn(apn));
    }
    out
}

/// ESM STATUS (§8.3.15).
pub fn encode_esm_status(ebi: u8, pti: u8, esm_cause: u8) -> Vec<u8> {
    vec![(ebi << 4) | PD_ESM, pti, esm_type::ESM_STATUS, esm_cause]
}

/// ATTACH ACCEPT (§8.2.1), the parts the UE acts on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttachAccept {
    /// EPS attach result (§9.9.3.10): 1 EPS only, 2 combined.
    pub result: u8,
    /// T3412, seconds; `None` = deactivated.
    pub t3412_s: Option<u64>,
    /// TAI list.
    pub tai_list: Vec<Tai>,
    /// ESM message container: the ACTIVATE DEFAULT EPS BEARER CONTEXT REQUEST.
    pub esm_message: Vec<u8>,
    /// GUTI (IEI 50).
    pub guti: Option<Guti>,
    /// EMM cause (IEI 53), e.g. for a combined attach accepted for EPS only.
    pub emm_cause: Option<u8>,
    /// T3402 (IEI 17), seconds.
    pub t3402_s: Option<u64>,
    /// T3412 extended value (IEI 5E, GPRS timer 3, §8.2.1.12): `Some(None)` =
    /// deactivated; unit 111 leaves it `None` (TS 24.008 §10.5.7.4a NOTE 2).
    pub t3412_extended_s: Option<Option<u64>>,
}

/// TRACKING AREA UPDATE ACCEPT (§8.2.26), the IEs this UE acts on.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TrackingAreaUpdateAccept {
    /// EPS update result (§9.9.3.13, bits 3-1).
    pub result: u8,
    /// T3412 value (IEI 5A): `Some(None)` = deactivated.
    pub t3412_s: Option<Option<u64>>,
    /// T3412 extended value (IEI 5E, GPRS timer 3): `Some(None)` =
    /// deactivated; unit 111 leaves it `None` (TS 24.008 §10.5.7.4a NOTE 2).
    pub t3412_extended_s: Option<Option<u64>>,
    /// GUTI (IEI 50).
    pub guti: Option<Guti>,
    /// TAI list (IEI 54).
    pub tai_list: Option<Vec<Tai>>,
    /// EPS bearer context status (IEI 57).
    pub eps_bearer_context_status: Option<EpsBearerContextStatus>,
    /// EMM cause (IEI 53).
    pub emm_cause: Option<u8>,
    /// T3402 value (IEI 17), seconds.
    pub t3402_s: Option<u64>,
}

/// TRACKING AREA UPDATE REJECT (§8.2.28).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrackingAreaUpdateReject {
    /// EMM cause.
    pub cause: u8,
    /// T3346 value (IEI 5F, GPRS timer 2): `Some(None)` = deactivated.
    pub t3346_s: Option<Option<u64>>,
}

/// ATTACH REJECT (§8.2.3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttachReject {
    /// EMM cause.
    pub cause: u8,
    /// T3402 (IEI 16, GPRS timer 2), seconds.
    pub t3402_s: Option<u64>,
    /// T3346 (IEI 5F, GPRS timer 2), seconds.
    pub t3346_s: Option<u64>,
}

/// SECURITY MODE COMMAND (§8.2.20).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SecurityModeCommand {
    /// Selected EEA (§9.9.3.23 bits 7-5).
    pub eea: u8,
    /// Selected EIA (§9.9.3.23 bits 3-1).
    pub eia: u8,
    /// NAS key set identifier.
    pub ksi: NasKsi,
    /// Replayed UE security capabilities (§9.9.3.36): octet 3 EEA, octet 4
    /// EIA, then optional UMTS/GPRS octets.
    pub replayed_capabilities: Vec<u8>,
    /// IMEISV request (IEI C-, value 1 = requested).
    pub imeisv_requested: bool,
}

/// SERVICE REJECT (§8.2.24).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceReject {
    /// EMM cause.
    pub cause: u8,
    /// T3442 (IEI 5B), seconds; for #39 only.
    pub t3442_s: Option<u64>,
    /// T3346 (IEI 5F), seconds; for #22.
    pub t3346_s: Option<u64>,
}

/// A downlink EMM message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DownlinkEmm {
    /// ATTACH ACCEPT.
    AttachAccept(AttachAccept),
    /// ATTACH REJECT.
    AttachReject(AttachReject),
    /// SERVICE REJECT.
    ServiceReject(ServiceReject),
    /// TRACKING AREA UPDATE ACCEPT.
    TauAccept(TrackingAreaUpdateAccept),
    /// TRACKING AREA UPDATE REJECT.
    TauReject(TrackingAreaUpdateReject),
    /// DETACH ACCEPT, network to UE (§8.2.10.2): the answer to a UE-initiated
    /// detach (#365).
    DetachAccept,
    /// DETACH REQUEST, network to UE (§8.2.11.2).
    DetachRequest {
        /// Detach type (§9.9.3.7): 1 re-attach required, 2 re-attach not
        /// required, 3 IMSI detach.
        detach_type: u8,
        /// EMM cause (IEI 53).
        cause: Option<u8>,
    },
    /// AUTHENTICATION REQUEST (§8.2.7).
    AuthenticationRequest {
        /// NAS key set identifier ASME.
        ksi: NasKsi,
        /// RAND.
        rand: [u8; 16],
        /// AUTN.
        autn: [u8; 16],
    },
    /// AUTHENTICATION REJECT (§8.2.6).
    AuthenticationReject,
    /// IDENTITY REQUEST (§8.2.18): the identity type (§9.9.3.17).
    IdentityRequest(u8),
    /// SECURITY MODE COMMAND.
    SecurityModeCommand(SecurityModeCommand),
    /// EMM INFORMATION (§8.2.13), not acted on.
    EmmInformation,
    /// EMM STATUS (§8.2.14).
    EmmStatus(u8),
    /// Any other EMM message type.
    Other(u8),
}

fn timer2(v: &[u8]) -> Option<u64> {
    v.first().and_then(|o| gprs_timer_seconds(*o))
}

/// Decode a plain downlink EMM message (protocol discriminator 7, security
/// header type 0).
pub fn decode_downlink_emm(msg: &[u8]) -> Result<DownlinkEmm, NasDecodeError> {
    let mut r = Reader::new(msg);
    let pd = r.u8()?;
    if pd != PD_EMM {
        return Err(NasDecodeError::Invalid("not a plain EMM message"));
    }
    let ty = r.u8()?;
    Ok(match ty {
        emm_type::ATTACH_ACCEPT => {
            let result = r.u8()? & 0x07;
            let t3412_s = gprs_timer_seconds(r.u8()?);
            let tai_list = decode_tai_list(r.lv()?)?;
            let esm_message = r.lv_e()?.to_vec();
            let mut a = AttachAccept {
                result,
                t3412_s,
                tai_list,
                esm_message,
                guti: None,
                emm_cause: None,
                t3402_s: None,
                t3412_extended_s: None,
            };
            for (iei, v) in r.optional_ies(&[(0x13, 6), (0x53, 2), (0x17, 2), (0x59, 2)])? {
                match iei {
                    0x50 => match EpsMobileIdentity::decode(v)? {
                        EpsMobileIdentity::Guti(g) => a.guti = Some(g),
                        _ => return Err(NasDecodeError::Invalid("GUTI IE")),
                    },
                    0x53 => a.emm_cause = v.first().copied(),
                    0x17 => a.t3402_s = timer2(v),
                    0x5E => {
                        a.t3412_extended_s = v
                            .first()
                            .filter(|o| *o >> 5 != 0b111)
                            .map(|o| gprs_timer3_seconds(*o))
                    }
                    _ => {}
                }
            }
            DownlinkEmm::AttachAccept(a)
        }
        emm_type::ATTACH_REJECT => {
            let cause = r.u8()?;
            let mut rej = AttachReject {
                cause,
                t3402_s: None,
                t3346_s: None,
            };
            for (iei, v) in r.optional_ies(&[])? {
                match iei {
                    0x16 => rej.t3402_s = timer2(v),
                    0x5F => rej.t3346_s = timer2(v),
                    _ => {}
                }
            }
            DownlinkEmm::AttachReject(rej)
        }
        emm_type::SERVICE_REJECT => {
            let cause = r.u8()?;
            let mut rej = ServiceReject {
                cause,
                t3442_s: None,
                t3346_s: None,
            };
            for (iei, v) in r.optional_ies(&[(0x5B, 2)])? {
                match iei {
                    0x5B => rej.t3442_s = timer2(v),
                    0x5F => rej.t3346_s = timer2(v),
                    _ => {}
                }
            }
            DownlinkEmm::ServiceReject(rej)
        }
        emm_type::TAU_ACCEPT => {
            let mut a = TrackingAreaUpdateAccept {
                result: r.u8()? & 0x07,
                ..TrackingAreaUpdateAccept::default()
            };
            for (iei, v) in
                r.optional_ies(&[(0x5A, 2), (0x13, 6), (0x53, 2), (0x17, 2), (0x59, 2)])?
            {
                match iei {
                    0x5A => a.t3412_s = v.first().map(|o| gprs_timer_seconds(*o)),
                    0x50 => match EpsMobileIdentity::decode(v)? {
                        EpsMobileIdentity::Guti(g) => a.guti = Some(g),
                        _ => return Err(NasDecodeError::Invalid("GUTI IE")),
                    },
                    0x54 => a.tai_list = Some(decode_tai_list(v)?),
                    0x57 => a.eps_bearer_context_status = Some(EpsBearerContextStatus::decode(v)?),
                    0x53 => a.emm_cause = v.first().copied(),
                    0x17 => a.t3402_s = timer2(v),
                    0x5E => {
                        a.t3412_extended_s = v
                            .first()
                            .filter(|o| *o >> 5 != 0b111)
                            .map(|o| gprs_timer3_seconds(*o))
                    }
                    _ => {}
                }
            }
            DownlinkEmm::TauAccept(a)
        }
        emm_type::TAU_REJECT => {
            let cause = r.u8()?;
            let mut rej = TrackingAreaUpdateReject {
                cause,
                t3346_s: None,
            };
            for (iei, v) in r.optional_ies(&[])? {
                if iei == 0x5F {
                    rej.t3346_s = v.first().map(|o| gprs_timer_seconds(*o));
                }
            }
            DownlinkEmm::TauReject(rej)
        }
        emm_type::DETACH_ACCEPT => DownlinkEmm::DetachAccept,
        emm_type::DETACH_REQUEST => {
            // Octet 3: spare half octet, then the detach type (§9.9.3.7).
            let detach_type = r.u8()? & 0x07;
            let mut cause = None;
            for (iei, v) in r.optional_ies(&[(0x53, 2)])? {
                if iei == 0x53 {
                    cause = v.first().copied();
                }
            }
            DownlinkEmm::DetachRequest { detach_type, cause }
        }
        emm_type::AUTHENTICATION_REQUEST => {
            let ksi = NasKsi::from_nibble(r.u8()? & 0x0F);
            let rand: [u8; 16] = r.bytes(16)?.try_into().expect("16 octets");
            let autn = r.lv()?;
            let autn: [u8; 16] = autn
                .try_into()
                .map_err(|_| NasDecodeError::Invalid("AUTN length"))?;
            DownlinkEmm::AuthenticationRequest { ksi, rand, autn }
        }
        emm_type::AUTHENTICATION_REJECT => DownlinkEmm::AuthenticationReject,
        emm_type::IDENTITY_REQUEST => DownlinkEmm::IdentityRequest(r.u8()? & 0x07),
        emm_type::SECURITY_MODE_COMMAND => {
            let algs = r.u8()?;
            let ksi = NasKsi::from_nibble(r.u8()? & 0x0F);
            let replayed = r.lv()?;
            if replayed.len() < 2 {
                return Err(NasDecodeError::Invalid("replayed UE security capabilities"));
            }
            let mut smc = SecurityModeCommand {
                eea: (algs >> 4) & 0x07,
                eia: algs & 0x07,
                ksi,
                replayed_capabilities: replayed.to_vec(),
                imeisv_requested: false,
            };
            for (iei, v) in r.optional_ies(&[(0x55, 5), (0x56, 5)])? {
                if iei == 0xC0 {
                    smc.imeisv_requested = v[0] & 0x07 == 1;
                }
            }
            DownlinkEmm::SecurityModeCommand(smc)
        }
        emm_type::EMM_INFORMATION => DownlinkEmm::EmmInformation,
        emm_type::EMM_STATUS => DownlinkEmm::EmmStatus(r.u8()?),
        other => DownlinkEmm::Other(other),
    })
}

/// ACTIVATE DEFAULT EPS BEARER CONTEXT REQUEST (§8.3.6), the parts the UE
/// keeps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivateDefaultBearerRequest {
    /// EPS bearer identity (5..15).
    pub ebi: u8,
    /// Procedure transaction identity.
    pub pti: u8,
    /// QCI (§9.9.4.3 octet 3).
    pub qci: u8,
    /// Access point name.
    pub apn: String,
    /// PDN address.
    pub pdn_address: PdnAddress,
    /// APN-AMBR (IEI 5E), the raw value.
    pub apn_ambr: Option<Vec<u8>>,
    /// ESM cause (IEI 58), e.g. #50 "PDN type IPv4 only allowed".
    pub esm_cause: Option<u8>,
    /// IPv4 DNS servers from the PCO (IEI 27).
    pub dns_ipv4: Vec<std::net::Ipv4Addr>,
}

/// A downlink ESM message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DownlinkEsm {
    /// ACTIVATE DEFAULT EPS BEARER CONTEXT REQUEST.
    ActivateDefaultBearerRequest(ActivateDefaultBearerRequest),
    /// PDN CONNECTIVITY REJECT (§8.3.19).
    PdnConnectivityReject {
        /// PTI.
        pti: u8,
        /// ESM cause.
        cause: u8,
    },
    /// ESM INFORMATION REQUEST (§8.3.13).
    EsmInformationRequest {
        /// PTI.
        pti: u8,
    },
    /// ACTIVATE DEDICATED EPS BEARER CONTEXT REQUEST (§8.3.3, #283).
    ActivateDedicatedBearerRequest(ActivateDedicatedBearerRequest),
    /// MODIFY EPS BEARER CONTEXT REQUEST (§8.3.18, #283).
    ModifyBearerRequest(ModifyBearerRequest),
    /// DEACTIVATE EPS BEARER CONTEXT REQUEST (§8.3.12, #283).
    DeactivateBearerRequest {
        /// The EPS bearer to deactivate.
        ebi: u8,
        /// PTI.
        pti: u8,
        /// ESM cause.
        cause: u8,
    },
    /// BEARER RESOURCE MODIFICATION REJECT (§8.3.9, #408).
    BearerResourceModificationReject {
        /// PTI.
        pti: u8,
        /// ESM cause.
        cause: u8,
    },
    /// Any other ESM message: (EBI, PTI, type).
    Other(u8, u8, u8),
}

/// Decode a plain ESM message.
pub fn decode_downlink_esm(msg: &[u8]) -> Result<DownlinkEsm, NasDecodeError> {
    let mut r = Reader::new(msg);
    let first = r.u8()?;
    if first & 0x0F != PD_ESM {
        return Err(NasDecodeError::Invalid("not an ESM message"));
    }
    let ebi = first >> 4;
    let pti = r.u8()?;
    let ty = r.u8()?;
    Ok(match ty {
        esm_type::ACTIVATE_DEFAULT_REQUEST => {
            let qos = r.lv()?;
            let qci = *qos.first().ok_or(NasDecodeError::Invalid("EPS QoS"))?;
            let apn = decode_apn(r.lv()?);
            let pdn_address = decode_pdn_address(r.lv()?)?;
            let mut req = ActivateDefaultBearerRequest {
                ebi,
                pti,
                qci,
                apn,
                pdn_address,
                apn_ambr: None,
                esm_cause: None,
                dns_ipv4: Vec::new(),
            };
            for (iei, v) in r.optional_ies(&[(0x32, 2), (0x58, 2)])? {
                match iei {
                    0x5E => req.apn_ambr = Some(v.to_vec()),
                    0x58 => req.esm_cause = v.first().copied(),
                    0x27 => req.dns_ipv4 = pco_dns_ipv4(v),
                    _ => {}
                }
            }
            DownlinkEsm::ActivateDefaultBearerRequest(req)
        }
        esm_type::PDN_CONNECTIVITY_REJECT => DownlinkEsm::PdnConnectivityReject {
            pti,
            cause: r.u8()?,
        },
        esm_type::ESM_INFORMATION_REQUEST => DownlinkEsm::EsmInformationRequest { pti },
        esm_type::ACTIVATE_DEDICATED_REQUEST => {
            // Linked EPS bearer identity in bits 4-1, spare half octet in 8-5
            let linked_ebi = r.u8()? & 0x0F;
            let qos = r.lv()?;
            let qci = *qos.first().ok_or(NasDecodeError::Invalid("EPS QoS"))?;
            let tft = r.lv()?.to_vec();
            // TV IEs: 32 Negotiated LLC SAPI (2). Type 1: 8- radio priority,
            // C- WLAN offload acceptability.
            let _ = r.optional_ies(&[(0x32, 2)])?;
            DownlinkEsm::ActivateDedicatedBearerRequest(ActivateDedicatedBearerRequest {
                ebi,
                pti,
                linked_ebi,
                qci,
                tft,
            })
        }
        esm_type::MODIFY_REQUEST => {
            let mut req = ModifyBearerRequest {
                ebi,
                pti,
                qci: None,
                tft: None,
            };
            for (iei, v) in r.optional_ies(&[(0x32, 2)])? {
                match iei {
                    0x5B => req.qci = v.first().copied(),
                    0x36 => req.tft = Some(v.to_vec()),
                    _ => {}
                }
            }
            DownlinkEsm::ModifyBearerRequest(req)
        }
        esm_type::DEACTIVATE_REQUEST => DownlinkEsm::DeactivateBearerRequest {
            ebi,
            pti,
            cause: r.u8()?,
        },
        esm_type::BEARER_RESOURCE_MODIFICATION_REJECT => {
            DownlinkEsm::BearerResourceModificationReject {
                pti,
                cause: r.u8()?,
            }
        }
        other => DownlinkEsm::Other(ebi, pti, other),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nextgsim_common::Plmn;

    #[test]
    fn attach_request_matches_its_24301_layout() {
        // Hand-derived from Table 8.2.4.1: 07 41 | KSI 7, type 1 | LV IMSI |
        // LV UE network capability | LV-E ESM container.
        let esm = PdnConnectivityRequest {
            pti: 1,
            pdn_type: PdnType::Ipv4,
            apn: None,
        }
        .encode();
        // 02 01 D0 | PDN type IPv4 / request type initial | 27 07 80 000D 00 000A 00
        assert_eq!(
            esm,
            [0x02, 0x01, 0xD0, 0x11, 0x27, 0x07, 0x80, 0x00, 0x0D, 0x00, 0x00, 0x0A, 0x00]
        );
        let m = AttachRequest {
            ksi: NasKsi::NO_KEY,
            attach_type: EpsAttachType::Eps,
            identity: EpsMobileIdentity::Imsi("999700000000001".into()),
            ue_network_capability: UeNetworkCapability::from_algorithms(&[0, 1, 2], &[0, 1, 2]),
            esm_message: esm,
            last_visited_tai: None,
        }
        .encode();
        assert_eq!(
            m,
            [
                0x07, 0x41, 0x71, 0x08, 0x99, 0x99, 0x07, 0x00, 0x00, 0x00, 0x00, 0x10, 0x02, 0xE0,
                0xE0, 0x00, 0x0D, 0x02, 0x01, 0xD0, 0x11, 0x27, 0x07, 0x80, 0x00, 0x0D, 0x00, 0x00,
                0x0A, 0x00
            ]
        );
    }

    #[test]
    fn attach_complete_and_bearer_accept() {
        let esm = encode_activate_default_accept(5, 1);
        assert_eq!(esm, [0x52, 0x01, 0xC2]);
        assert_eq!(
            encode_attach_complete(&esm),
            [0x07, 0x43, 0x00, 0x03, 0x52, 0x01, 0xC2]
        );
    }

    #[test]
    fn authentication_failure_carries_auts_only_for_synch_failure() {
        assert_eq!(
            encode_authentication_failure(emm_cause::MAC_FAILURE, None),
            [0x07, 0x5C, 20]
        );
        let m = encode_authentication_failure(emm_cause::SYNCH_FAILURE, Some(&[0xAB; 14]));
        assert_eq!(&m[..5], &[0x07, 0x5C, 21, 0x30, 14]);
        assert_eq!(m.len(), 5 + 14);
    }

    #[test]
    fn authentication_request_decodes() {
        let mut m = vec![0x07, 0x52, 0x01];
        m.extend_from_slice(&[0x11; 16]);
        m.push(16);
        m.extend_from_slice(&[0x22; 16]);
        assert_eq!(
            decode_downlink_emm(&m).unwrap(),
            DownlinkEmm::AuthenticationRequest {
                ksi: NasKsi {
                    mapped: false,
                    value: 1
                },
                rand: [0x11; 16],
                autn: [0x22; 16],
            }
        );
    }

    #[test]
    fn security_mode_command_decodes_algorithms_ksi_and_imeisv_request() {
        // 0x02 = EEA0 / EIA2, KSI 1, replayed E0 E0, IMEISV request C1.
        let m = [0x07, 0x5D, 0x02, 0x01, 0x02, 0xE0, 0xE0, 0xC1];
        assert_eq!(
            decode_downlink_emm(&m).unwrap(),
            DownlinkEmm::SecurityModeCommand(SecurityModeCommand {
                eea: 0,
                eia: 2,
                ksi: NasKsi {
                    mapped: false,
                    value: 1
                },
                replayed_capabilities: vec![0xE0, 0xE0],
                imeisv_requested: true,
            })
        );
    }

    #[test]
    fn attach_accept_decodes_mandatory_and_guti() {
        let esm = [
            0x52, 0x01, 0xC1, 0x01, 0x09, 0x01, 0x00, 0x05, 0x01, 10, 45, 0, 2,
        ];
        let mut m = vec![
            0x07, 0x42, 0x01, 0x49, 0x06, 0x00, 0x99, 0xF9, 0x07, 0x00, 0x01,
        ];
        m.extend_from_slice(&(esm.len() as u16).to_be_bytes());
        m.extend_from_slice(&esm);
        m.extend_from_slice(&[
            0x50, 0x0B, 0xF6, 0x99, 0xF9, 0x07, 0x00, 0x02, 0x01, 0xC0, 0x00, 0x00, 0x01,
        ]);
        let DownlinkEmm::AttachAccept(a) = decode_downlink_emm(&m).unwrap() else {
            panic!("not an attach accept");
        };
        assert_eq!(a.result, 1);
        assert_eq!(a.t3412_s, Some(9 * 360));
        assert_eq!(a.tai_list[0].tac, 1);
        assert_eq!(a.esm_message, esm);
        assert_eq!(a.guti.unwrap().m_tmsi, 0xC000_0001);
        let DownlinkEsm::ActivateDefaultBearerRequest(b) = decode_downlink_esm(&esm).unwrap()
        else {
            panic!("not a default bearer request");
        };
        assert_eq!((b.ebi, b.pti, b.qci), (5, 1, 9));
        assert_eq!(b.apn, "");
        assert_eq!(
            b.pdn_address,
            PdnAddress::Ipv4(std::net::Ipv4Addr::new(10, 45, 0, 2))
        );
        let _ = Plmn::new(999, 70, false);
    }

    #[test]
    fn dns_servers_are_read_from_the_network_pco() {
        // 80 | 000D 04 08.08.08.08 | 0010 02 05 DC (IPv4 link MTU, ignored)
        let v = [
            0x80, 0x00, 0x0D, 0x04, 8, 8, 8, 8, 0x00, 0x10, 0x02, 0x05, 0xDC,
        ];
        assert_eq!(pco_dns_ipv4(&v), vec![std::net::Ipv4Addr::new(8, 8, 8, 8)]);
    }

    #[test]
    fn attach_reject_reads_t3402() {
        let m = [0x07, 0x44, 0x11, 0x16, 0x01, 0x21];
        assert_eq!(
            decode_downlink_emm(&m).unwrap(),
            DownlinkEmm::AttachReject(AttachReject {
                cause: 17,
                t3402_s: Some(60),
                t3346_s: None
            })
        );
    }
}
