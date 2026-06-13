//! Path Switch Request Procedure
//!
//! Implements the Path Switch Request procedure as defined in 3GPP TS 38.413
//! Section 8.4.4. After an Xn handover the target NG-RAN node requests the
//! 5GC to switch the DL GTP-U termination point towards itself:
//!
//! - `PathSwitchRequest` (gNB → AMF, initiating)
//! - `PathSwitchRequestAcknowledge` (AMF → gNB, successful outcome)
//! - `PathSwitchRequestFailure` (AMF → gNB, unsuccessful outcome)
//!
//! The per-session `PathSwitchRequestTransfer` (TS 38.413 §9.3.4.8) carries
//! the new DL GTP-U F-TEID allocated by the target gNB plus the accepted QoS
//! flows; it is encoded with real APER via the generated ASN.1 types.

use bitvec::prelude::*;

use crate::codec::generated::*;
use crate::codec::{decode_aper, decode_ngap_pdu, encode_aper, encode_ngap_pdu, NgapCodecError};
use crate::procedures::initial_ue_message::{build_nr_cgi, build_tai, NrCgi, Tai};
use crate::procedures::ng_setup::NgSetupFailureCause;
use crate::procedures::transfer::GtpTunnelInfo;
use crate::procedures::ue_context_release::parse_cause;
use thiserror::Error;

/// Errors that can occur during Path Switch procedures
#[derive(Debug, Error)]
pub enum PathSwitchError {
    /// Codec error during encoding/decoding
    #[error("Codec error: {0}")]
    CodecError(#[from] NgapCodecError),

    /// Transfer container error
    #[error("Transfer error: {0}")]
    TransferError(#[from] crate::procedures::transfer::TransferError),

    /// Invalid message type received
    #[error("Invalid message type: expected {expected}, got {actual}")]
    InvalidMessageType {
        /// Expected message type
        expected: String,
        /// Actual message type received
        actual: String,
    },

    /// Missing mandatory IE
    #[error("Missing mandatory IE: {0}")]
    MissingMandatoryIe(&'static str),
}

// ============================================================================
// Path Switch Request Transfer (TS 38.413 §9.3.4.8)
// ============================================================================

/// Per-session data carried in the PathSwitchRequest
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathSwitchSessionItem {
    /// PDU Session ID
    pub pdu_session_id: u8,
    /// New DL GTP-U tunnel endpoint allocated by the target gNB
    pub dl_tunnel: GtpTunnelInfo,
    /// QFIs accepted by the target gNB (at least one)
    pub accepted_qfis: Vec<u8>,
}

/// Encode a `PathSwitchRequestTransfer` for one PDU session
pub fn encode_path_switch_request_transfer(
    item: &PathSwitchSessionItem,
) -> Result<Vec<u8>, PathSwitchError> {
    if item.accepted_qfis.is_empty() {
        return Err(PathSwitchError::MissingMandatoryIe("QosFlowAcceptedList"));
    }
    let transfer = PathSwitchRequestTransfer {
        dl_ngu_up_tnl_information: tunnel_to_asn(item.dl_tunnel),
        dl_ngu_tnl_information_reused: None,
        user_plane_security_information: None,
        qos_flow_accepted_list: QosFlowAcceptedList(
            item.accepted_qfis
                .iter()
                .map(|&qfi| QosFlowAcceptedItem {
                    qos_flow_identifier: QosFlowIdentifier(qfi),
                    ie_extensions: None,
                })
                .collect(),
        ),
        ie_extensions: None,
    };
    Ok(encode_aper(&transfer)?)
}

/// Decode a `PathSwitchRequestTransfer`
pub fn decode_path_switch_request_transfer(
    pdu_session_id: u8,
    bytes: &[u8],
) -> Result<PathSwitchSessionItem, PathSwitchError> {
    let transfer: PathSwitchRequestTransfer = decode_aper(bytes)?;
    Ok(PathSwitchSessionItem {
        pdu_session_id,
        dl_tunnel: tunnel_from_asn(&transfer.dl_ngu_up_tnl_information)?,
        accepted_qfis: transfer
            .qos_flow_accepted_list
            .0
            .iter()
            .map(|i| i.qos_flow_identifier.0)
            .collect(),
    })
}

fn tunnel_to_asn(tunnel: GtpTunnelInfo) -> UPTransportLayerInformation {
    let mut bits: BitVec<u8, Msb0> = BitVec::new();
    let push_octets = |bits: &mut BitVec<u8, Msb0>, octets: &[u8]| {
        for byte in octets {
            for i in (0..8).rev() {
                bits.push((byte >> i) & 1 == 1);
            }
        }
    };
    match tunnel.address {
        std::net::IpAddr::V4(v4) => push_octets(&mut bits, &v4.octets()),
        std::net::IpAddr::V6(v6) => push_octets(&mut bits, &v6.octets()),
    }
    UPTransportLayerInformation::GTPTunnel(GTPTunnel {
        transport_layer_address: TransportLayerAddress(bits),
        gtp_teid: GTP_TEID(tunnel.teid.to_be_bytes().to_vec()),
        ie_extensions: None,
    })
}

fn tunnel_from_asn(
    info: &UPTransportLayerInformation,
) -> Result<GtpTunnelInfo, PathSwitchError> {
    let tunnel = match info {
        UPTransportLayerInformation::GTPTunnel(t) => t,
        UPTransportLayerInformation::Choice_Extensions(_) => {
            return Err(PathSwitchError::InvalidMessageType {
                expected: "GTPTunnel".to_string(),
                actual: "choice-Extensions".to_string(),
            })
        }
    };
    let bits = &tunnel.transport_layer_address.0;
    let address = match bits.len() {
        32 => {
            let mut octets = [0u8; 4];
            for (i, octet) in octets.iter_mut().enumerate() {
                for b in 0..8 {
                    if bits[i * 8 + b] {
                        *octet |= 1 << (7 - b);
                    }
                }
            }
            std::net::IpAddr::V4(octets.into())
        }
        128 => {
            let mut octets = [0u8; 16];
            for (i, octet) in octets.iter_mut().enumerate() {
                for b in 0..8 {
                    if bits[i * 8 + b] {
                        *octet |= 1 << (7 - b);
                    }
                }
            }
            std::net::IpAddr::V6(octets.into())
        }
        _ => {
            return Err(PathSwitchError::InvalidMessageType {
                expected: "IPv4/IPv6 TransportLayerAddress".to_string(),
                actual: format!("{} bits", bits.len()),
            })
        }
    };
    if tunnel.gtp_teid.0.len() != 4 {
        return Err(PathSwitchError::MissingMandatoryIe("GTP-TEID"));
    }
    let teid = u32::from_be_bytes([
        tunnel.gtp_teid.0[0],
        tunnel.gtp_teid.0[1],
        tunnel.gtp_teid.0[2],
        tunnel.gtp_teid.0[3],
    ]);
    Ok(GtpTunnelInfo { address, teid })
}

// ============================================================================
// Path Switch Request
// ============================================================================

/// UE security capabilities advertised in the PathSwitchRequest
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UeSecurityCapabilityBits {
    /// NR encryption algorithms bitmap (bit 0 = NEA1, bit 1 = NEA2, bit 2 = NEA3)
    pub nr_encryption: u16,
    /// NR integrity algorithms bitmap (bit 0 = NIA1, bit 1 = NIA2, bit 2 = NIA3)
    pub nr_integrity: u16,
    /// E-UTRA encryption algorithms bitmap
    pub eutra_encryption: u16,
    /// E-UTRA integrity algorithms bitmap
    pub eutra_integrity: u16,
}

impl Default for UeSecurityCapabilityBits {
    fn default() -> Self {
        // NEA1/NEA2 and NIA1/NIA2 supported (bits are MSB-first per TS 38.413)
        Self {
            nr_encryption: 0xC000,
            nr_integrity: 0xC000,
            eutra_encryption: 0,
            eutra_integrity: 0,
        }
    }
}

fn bits16(value: u16) -> BitVec<u8, Msb0> {
    let mut bv: BitVec<u8, Msb0> = BitVec::with_capacity(16);
    for i in (0..16).rev() {
        bv.push((value >> i) & 1 == 1);
    }
    bv
}

fn bits16_to_u16(bits: &BitVec<u8, Msb0>) -> u16 {
    let mut value = 0u16;
    for (i, bit) in bits.iter().take(16).enumerate() {
        if *bit {
            value |= 1 << (15 - i);
        }
    }
    value
}

/// Parameters for building a Path Switch Request
#[derive(Debug, Clone)]
pub struct PathSwitchRequestParams {
    /// RAN UE NGAP ID allocated by the target gNB
    pub ran_ue_ngap_id: u32,
    /// AMF UE NGAP ID as known to the source gNB (sourceAMF-UE-NGAP-ID)
    pub source_amf_ue_ngap_id: u64,
    /// User location at the target cell
    pub nr_cgi: NrCgi,
    /// TAI of the target cell
    pub tai: Tai,
    /// UE security capabilities
    pub ue_security_capabilities: UeSecurityCapabilityBits,
    /// PDU sessions to be switched (at least one)
    pub sessions: Vec<PathSwitchSessionItem>,
}

/// Parsed Path Switch Request data
#[derive(Debug, Clone)]
pub struct PathSwitchRequestData {
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
    /// Source AMF UE NGAP ID
    pub source_amf_ue_ngap_id: u64,
    /// UE security capabilities
    pub ue_security_capabilities: UeSecurityCapabilityBits,
    /// PDU sessions to be switched
    pub sessions: Vec<PathSwitchSessionItem>,
}

/// Build a Path Switch Request PDU
pub fn build_path_switch_request(
    params: &PathSwitchRequestParams,
) -> Result<NGAP_PDU, PathSwitchError> {
    if params.sessions.is_empty() {
        return Err(PathSwitchError::MissingMandatoryIe(
            "PDUSessionResourceToBeSwitchedDLList",
        ));
    }

    let mut protocol_ies = Vec::new();

    // IE: RAN-UE-NGAP-ID (mandatory)
    protocol_ies.push(PathSwitchRequestProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_RAN_UE_NGAP_ID),
        criticality: Criticality(Criticality::REJECT),
        value: PathSwitchRequestProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(RAN_UE_NGAP_ID(
            params.ran_ue_ngap_id,
        )),
    });

    // IE: SourceAMF-UE-NGAP-ID (mandatory)
    protocol_ies.push(PathSwitchRequestProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_SOURCE_AMF_UE_NGAP_ID),
        criticality: Criticality(Criticality::REJECT),
        value: PathSwitchRequestProtocolIEs_EntryValue::Id_SourceAMF_UE_NGAP_ID(AMF_UE_NGAP_ID(
            params.source_amf_ue_ngap_id,
        )),
    });

    // IE: UserLocationInformation (mandatory)
    let uli = UserLocationInformation::UserLocationInformationNR(UserLocationInformationNR {
        nr_cgi: build_nr_cgi(&params.nr_cgi),
        tai: build_tai(&params.tai),
        time_stamp: None,
        ie_extensions: None,
    });
    protocol_ies.push(PathSwitchRequestProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_USER_LOCATION_INFORMATION),
        criticality: Criticality(Criticality::IGNORE),
        value: PathSwitchRequestProtocolIEs_EntryValue::Id_UserLocationInformation(uli),
    });

    // IE: UESecurityCapabilities (mandatory)
    let caps = &params.ue_security_capabilities;
    protocol_ies.push(PathSwitchRequestProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_UE_SECURITY_CAPABILITIES),
        criticality: Criticality(Criticality::IGNORE),
        value: PathSwitchRequestProtocolIEs_EntryValue::Id_UESecurityCapabilities(
            UESecurityCapabilities {
                n_rencryption_algorithms: NRencryptionAlgorithms(bits16(caps.nr_encryption)),
                n_rintegrity_protection_algorithms: NRintegrityProtectionAlgorithms(bits16(
                    caps.nr_integrity,
                )),
                eutr_aencryption_algorithms: EUTRAencryptionAlgorithms(bits16(
                    caps.eutra_encryption,
                )),
                eutr_aintegrity_protection_algorithms: EUTRAintegrityProtectionAlgorithms(bits16(
                    caps.eutra_integrity,
                )),
                ie_extensions: None,
            },
        ),
    });

    // IE: PDUSessionResourceToBeSwitchedDLList (mandatory)
    let mut items = Vec::with_capacity(params.sessions.len());
    for session in &params.sessions {
        items.push(PDUSessionResourceToBeSwitchedDLItem {
            pdu_session_id: PDUSessionID(session.pdu_session_id),
            path_switch_request_transfer: PDUSessionResourceToBeSwitchedDLItemPathSwitchRequestTransfer(
                encode_path_switch_request_transfer(session)?,
            ),
            ie_extensions: None,
        });
    }
    protocol_ies.push(PathSwitchRequestProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_PDU_SESSION_RESOURCE_TO_BE_SWITCHED_DL_LIST),
        criticality: Criticality(Criticality::REJECT),
        value: PathSwitchRequestProtocolIEs_EntryValue::Id_PDUSessionResourceToBeSwitchedDLList(
            PDUSessionResourceToBeSwitchedDLList(items),
        ),
    });

    Ok(NGAP_PDU::InitiatingMessage(InitiatingMessage {
        procedure_code: ProcedureCode(ID_PATH_SWITCH_REQUEST),
        criticality: Criticality(Criticality::REJECT),
        value: InitiatingMessageValue::Id_PathSwitchRequest(PathSwitchRequest {
            protocol_i_es: PathSwitchRequestProtocolIEs(protocol_ies),
        }),
    }))
}

/// Parse a Path Switch Request from an NGAP PDU
pub fn parse_path_switch_request(
    pdu: &NGAP_PDU,
) -> Result<PathSwitchRequestData, PathSwitchError> {
    let initiating = match pdu {
        NGAP_PDU::InitiatingMessage(msg) => msg,
        _ => {
            return Err(PathSwitchError::InvalidMessageType {
                expected: "InitiatingMessage".to_string(),
                actual: format!("{pdu:?}"),
            })
        }
    };
    let request = match &initiating.value {
        InitiatingMessageValue::Id_PathSwitchRequest(req) => req,
        _ => {
            return Err(PathSwitchError::InvalidMessageType {
                expected: "PathSwitchRequest".to_string(),
                actual: format!("{:?}", initiating.value),
            })
        }
    };

    let mut ran_ue_ngap_id = None;
    let mut source_amf_ue_ngap_id = None;
    let mut ue_security_capabilities = None;
    let mut sessions: Option<Vec<PathSwitchSessionItem>> = None;

    for ie in &request.protocol_i_es.0 {
        match &ie.value {
            PathSwitchRequestProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(id) => {
                ran_ue_ngap_id = Some(id.0);
            }
            PathSwitchRequestProtocolIEs_EntryValue::Id_SourceAMF_UE_NGAP_ID(id) => {
                source_amf_ue_ngap_id = Some(id.0);
            }
            PathSwitchRequestProtocolIEs_EntryValue::Id_UESecurityCapabilities(caps) => {
                ue_security_capabilities = Some(UeSecurityCapabilityBits {
                    nr_encryption: bits16_to_u16(&caps.n_rencryption_algorithms.0),
                    nr_integrity: bits16_to_u16(&caps.n_rintegrity_protection_algorithms.0),
                    eutra_encryption: bits16_to_u16(&caps.eutr_aencryption_algorithms.0),
                    eutra_integrity: bits16_to_u16(&caps.eutr_aintegrity_protection_algorithms.0),
                });
            }
            PathSwitchRequestProtocolIEs_EntryValue::Id_PDUSessionResourceToBeSwitchedDLList(
                list,
            ) => {
                let mut parsed = Vec::with_capacity(list.0.len());
                for item in &list.0 {
                    parsed.push(decode_path_switch_request_transfer(
                        item.pdu_session_id.0,
                        &item.path_switch_request_transfer.0,
                    )?);
                }
                sessions = Some(parsed);
            }
            _ => {}
        }
    }

    Ok(PathSwitchRequestData {
        ran_ue_ngap_id: ran_ue_ngap_id
            .ok_or(PathSwitchError::MissingMandatoryIe("RAN-UE-NGAP-ID"))?,
        source_amf_ue_ngap_id: source_amf_ue_ngap_id
            .ok_or(PathSwitchError::MissingMandatoryIe("SourceAMF-UE-NGAP-ID"))?,
        ue_security_capabilities: ue_security_capabilities
            .ok_or(PathSwitchError::MissingMandatoryIe("UESecurityCapabilities"))?,
        sessions: sessions.ok_or(PathSwitchError::MissingMandatoryIe(
            "PDUSessionResourceToBeSwitchedDLList",
        ))?,
    })
}

// ============================================================================
// Path Switch Request Acknowledge
// ============================================================================

/// Per-session data in the PathSwitchRequestAcknowledge
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwitchedSessionItem {
    /// PDU Session ID
    pub pdu_session_id: u8,
    /// New UL NG-U tunnel from the 5GC (optional in the transfer)
    pub ul_tunnel: Option<GtpTunnelInfo>,
}

/// Parameters for building a Path Switch Request Acknowledge
#[derive(Debug, Clone)]
pub struct PathSwitchRequestAcknowledgeParams {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
    /// Next Hop Chaining Count (0..7)
    pub next_hop_chaining_count: u8,
    /// Next Hop NH key (256 bits)
    pub next_hop_nh: [u8; 32],
    /// Switched PDU sessions
    pub switched_sessions: Vec<SwitchedSessionItem>,
}

/// Parsed Path Switch Request Acknowledge data
#[derive(Debug, Clone)]
pub struct PathSwitchRequestAcknowledgeData {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
    /// Next Hop Chaining Count
    pub next_hop_chaining_count: u8,
    /// Next Hop NH key (256 bits)
    pub next_hop_nh: [u8; 32],
    /// Switched PDU sessions
    pub switched_sessions: Vec<SwitchedSessionItem>,
}

fn encode_path_switch_ack_transfer(
    item: &SwitchedSessionItem,
) -> Result<Vec<u8>, PathSwitchError> {
    let transfer = PathSwitchRequestAcknowledgeTransfer {
        ul_ngu_up_tnl_information: item.ul_tunnel.map(tunnel_to_asn),
        security_indication: None,
        ie_extensions: None,
    };
    Ok(encode_aper(&transfer)?)
}

fn decode_path_switch_ack_transfer(
    pdu_session_id: u8,
    bytes: &[u8],
) -> Result<SwitchedSessionItem, PathSwitchError> {
    let transfer: PathSwitchRequestAcknowledgeTransfer = decode_aper(bytes)?;
    Ok(SwitchedSessionItem {
        pdu_session_id,
        ul_tunnel: transfer
            .ul_ngu_up_tnl_information
            .as_ref()
            .map(tunnel_from_asn)
            .transpose()?,
    })
}

/// Build a Path Switch Request Acknowledge PDU
pub fn build_path_switch_request_acknowledge(
    params: &PathSwitchRequestAcknowledgeParams,
) -> Result<NGAP_PDU, PathSwitchError> {
    let mut protocol_ies = Vec::new();

    protocol_ies.push(PathSwitchRequestAcknowledgeProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_AMF_UE_NGAP_ID),
        criticality: Criticality(Criticality::IGNORE),
        value: PathSwitchRequestAcknowledgeProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(
            AMF_UE_NGAP_ID(params.amf_ue_ngap_id),
        ),
    });
    protocol_ies.push(PathSwitchRequestAcknowledgeProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_RAN_UE_NGAP_ID),
        criticality: Criticality(Criticality::IGNORE),
        value: PathSwitchRequestAcknowledgeProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(
            RAN_UE_NGAP_ID(params.ran_ue_ngap_id),
        ),
    });

    // IE: SecurityContext (mandatory)
    let mut nh_bits: BitVec<u8, Msb0> = BitVec::with_capacity(256);
    for byte in params.next_hop_nh {
        for i in (0..8).rev() {
            nh_bits.push((byte >> i) & 1 == 1);
        }
    }
    protocol_ies.push(PathSwitchRequestAcknowledgeProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_SECURITY_CONTEXT),
        criticality: Criticality(Criticality::REJECT),
        value: PathSwitchRequestAcknowledgeProtocolIEs_EntryValue::Id_SecurityContext(
            SecurityContext {
                next_hop_chaining_count: NextHopChainingCount(params.next_hop_chaining_count),
                next_hop_nh: SecurityKey(nh_bits),
                ie_extensions: None,
            },
        ),
    });

    // IE: PDUSessionResourceSwitchedList (optional but expected on success)
    if !params.switched_sessions.is_empty() {
        let mut items = Vec::with_capacity(params.switched_sessions.len());
        for session in &params.switched_sessions {
            items.push(PDUSessionResourceSwitchedItem {
                pdu_session_id: PDUSessionID(session.pdu_session_id),
                path_switch_request_acknowledge_transfer:
                    PDUSessionResourceSwitchedItemPathSwitchRequestAcknowledgeTransfer(
                        encode_path_switch_ack_transfer(session)?,
                    ),
                ie_extensions: None,
            });
        }
        protocol_ies.push(PathSwitchRequestAcknowledgeProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_PDU_SESSION_RESOURCE_SWITCHED_LIST),
            criticality: Criticality(Criticality::IGNORE),
            value:
                PathSwitchRequestAcknowledgeProtocolIEs_EntryValue::Id_PDUSessionResourceSwitchedList(
                    PDUSessionResourceSwitchedList(items),
                ),
        });
    }

    Ok(NGAP_PDU::SuccessfulOutcome(SuccessfulOutcome {
        procedure_code: ProcedureCode(ID_PATH_SWITCH_REQUEST),
        criticality: Criticality(Criticality::REJECT),
        value: SuccessfulOutcomeValue::Id_PathSwitchRequest(PathSwitchRequestAcknowledge {
            protocol_i_es: PathSwitchRequestAcknowledgeProtocolIEs(protocol_ies),
        }),
    }))
}

/// Parse a Path Switch Request Acknowledge from an NGAP PDU
pub fn parse_path_switch_request_acknowledge(
    pdu: &NGAP_PDU,
) -> Result<PathSwitchRequestAcknowledgeData, PathSwitchError> {
    let outcome = match pdu {
        NGAP_PDU::SuccessfulOutcome(outcome) => outcome,
        _ => {
            return Err(PathSwitchError::InvalidMessageType {
                expected: "SuccessfulOutcome".to_string(),
                actual: format!("{pdu:?}"),
            })
        }
    };
    let ack = match &outcome.value {
        SuccessfulOutcomeValue::Id_PathSwitchRequest(ack) => ack,
        _ => {
            return Err(PathSwitchError::InvalidMessageType {
                expected: "PathSwitchRequestAcknowledge".to_string(),
                actual: format!("{:?}", outcome.value),
            })
        }
    };

    let mut amf_ue_ngap_id = None;
    let mut ran_ue_ngap_id = None;
    let mut security_context: Option<(u8, [u8; 32])> = None;
    let mut switched_sessions = Vec::new();

    for ie in &ack.protocol_i_es.0 {
        match &ie.value {
            PathSwitchRequestAcknowledgeProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(id) => {
                amf_ue_ngap_id = Some(id.0);
            }
            PathSwitchRequestAcknowledgeProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(id) => {
                ran_ue_ngap_id = Some(id.0);
            }
            PathSwitchRequestAcknowledgeProtocolIEs_EntryValue::Id_SecurityContext(ctx) => {
                let mut nh = [0u8; 32];
                for (i, byte) in nh.iter_mut().enumerate() {
                    for b in 0..8 {
                        let idx = i * 8 + b;
                        if idx < ctx.next_hop_nh.0.len() && ctx.next_hop_nh.0[idx] {
                            *byte |= 1 << (7 - b);
                        }
                    }
                }
                security_context = Some((ctx.next_hop_chaining_count.0, nh));
            }
            PathSwitchRequestAcknowledgeProtocolIEs_EntryValue::Id_PDUSessionResourceSwitchedList(
                list,
            ) => {
                for item in &list.0 {
                    switched_sessions.push(decode_path_switch_ack_transfer(
                        item.pdu_session_id.0,
                        &item.path_switch_request_acknowledge_transfer.0,
                    )?);
                }
            }
            _ => {}
        }
    }

    let (next_hop_chaining_count, next_hop_nh) =
        security_context.ok_or(PathSwitchError::MissingMandatoryIe("SecurityContext"))?;

    Ok(PathSwitchRequestAcknowledgeData {
        amf_ue_ngap_id: amf_ue_ngap_id
            .ok_or(PathSwitchError::MissingMandatoryIe("AMF-UE-NGAP-ID"))?,
        ran_ue_ngap_id: ran_ue_ngap_id
            .ok_or(PathSwitchError::MissingMandatoryIe("RAN-UE-NGAP-ID"))?,
        next_hop_chaining_count,
        next_hop_nh,
        switched_sessions,
    })
}

// ============================================================================
// Path Switch Request Failure
// ============================================================================

/// Parsed Path Switch Request Failure data
#[derive(Debug, Clone)]
pub struct PathSwitchRequestFailureData {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
    /// PDU sessions released with their causes (from the released list transfers)
    pub released_sessions: Vec<(u8, NgSetupFailureCause)>,
}

/// Parse a Path Switch Request Failure from an NGAP PDU.
///
/// Per TS 38.413 the failure carries the released-session list whose
/// `PathSwitchRequestUnsuccessfulTransfer` holds the cause for each session.
pub fn parse_path_switch_request_failure(
    pdu: &NGAP_PDU,
) -> Result<PathSwitchRequestFailureData, PathSwitchError> {
    let outcome = match pdu {
        NGAP_PDU::UnsuccessfulOutcome(outcome) => outcome,
        _ => {
            return Err(PathSwitchError::InvalidMessageType {
                expected: "UnsuccessfulOutcome".to_string(),
                actual: format!("{pdu:?}"),
            })
        }
    };
    let failure = match &outcome.value {
        UnsuccessfulOutcomeValue::Id_PathSwitchRequest(failure) => failure,
        _ => {
            return Err(PathSwitchError::InvalidMessageType {
                expected: "PathSwitchRequestFailure".to_string(),
                actual: format!("{:?}", outcome.value),
            })
        }
    };

    let mut amf_ue_ngap_id = None;
    let mut ran_ue_ngap_id = None;
    let mut released_sessions = Vec::new();

    for ie in &failure.protocol_i_es.0 {
        match &ie.value {
            PathSwitchRequestFailureProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(id) => {
                amf_ue_ngap_id = Some(id.0);
            }
            PathSwitchRequestFailureProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(id) => {
                ran_ue_ngap_id = Some(id.0);
            }
            PathSwitchRequestFailureProtocolIEs_EntryValue::Id_PDUSessionResourceReleasedListPSFail(
                list,
            ) => {
                for item in &list.0 {
                    let transfer: PathSwitchRequestUnsuccessfulTransfer =
                        decode_aper(&item.path_switch_request_unsuccessful_transfer.0)?;
                    released_sessions
                        .push((item.pdu_session_id.0, parse_cause(&transfer.cause)));
                }
            }
            _ => {}
        }
    }

    Ok(PathSwitchRequestFailureData {
        amf_ue_ngap_id: amf_ue_ngap_id
            .ok_or(PathSwitchError::MissingMandatoryIe("AMF-UE-NGAP-ID"))?,
        ran_ue_ngap_id: ran_ue_ngap_id
            .ok_or(PathSwitchError::MissingMandatoryIe("RAN-UE-NGAP-ID"))?,
        released_sessions,
    })
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Build and encode a Path Switch Request to bytes
pub fn encode_path_switch_request(
    params: &PathSwitchRequestParams,
) -> Result<Vec<u8>, PathSwitchError> {
    let pdu = build_path_switch_request(params)?;
    Ok(encode_ngap_pdu(&pdu)?)
}

/// Decode and parse a Path Switch Request from bytes
pub fn decode_path_switch_request(
    bytes: &[u8],
) -> Result<PathSwitchRequestData, PathSwitchError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_path_switch_request(&pdu)
}

/// Build and encode a Path Switch Request Acknowledge to bytes
pub fn encode_path_switch_request_acknowledge(
    params: &PathSwitchRequestAcknowledgeParams,
) -> Result<Vec<u8>, PathSwitchError> {
    let pdu = build_path_switch_request_acknowledge(params)?;
    Ok(encode_ngap_pdu(&pdu)?)
}

/// Decode and parse a Path Switch Request Acknowledge from bytes
pub fn decode_path_switch_request_acknowledge(
    bytes: &[u8],
) -> Result<PathSwitchRequestAcknowledgeData, PathSwitchError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_path_switch_request_acknowledge(&pdu)
}

/// Decode and parse a Path Switch Request Failure from bytes
pub fn decode_path_switch_request_failure(
    bytes: &[u8],
) -> Result<PathSwitchRequestFailureData, PathSwitchError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_path_switch_request_failure(&pdu)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_session() -> PathSwitchSessionItem {
        PathSwitchSessionItem {
            pdu_session_id: 5,
            dl_tunnel: GtpTunnelInfo {
                address: "192.168.55.10".parse().unwrap(),
                teid: 0xAABB_CCDD,
            },
            accepted_qfis: vec![1, 2],
        }
    }

    fn sample_request_params() -> PathSwitchRequestParams {
        PathSwitchRequestParams {
            ran_ue_ngap_id: 7,
            source_amf_ue_ngap_id: 4242,
            nr_cgi: NrCgi {
                plmn_identity: [0x00, 0xF1, 0x10],
                nr_cell_identity: 0x10,
            },
            tai: Tai {
                plmn_identity: [0x00, 0xF1, 0x10],
                tac: [0, 0, 1],
            },
            ue_security_capabilities: UeSecurityCapabilityBits::default(),
            sessions: vec![sample_session()],
        }
    }

    #[test]
    fn test_path_switch_request_transfer_roundtrip() {
        let session = sample_session();
        let bytes = encode_path_switch_request_transfer(&session).unwrap();
        let decoded = decode_path_switch_request_transfer(5, &bytes).unwrap();
        assert_eq!(decoded, session);
    }

    #[test]
    fn test_path_switch_request_transfer_rejects_empty_flows() {
        let mut session = sample_session();
        session.accepted_qfis.clear();
        assert!(encode_path_switch_request_transfer(&session).is_err());
    }

    #[test]
    fn test_path_switch_request_roundtrip() {
        let params = sample_request_params();
        let bytes = encode_path_switch_request(&params).unwrap();
        let decoded = decode_path_switch_request(&bytes).unwrap();
        assert_eq!(decoded.ran_ue_ngap_id, params.ran_ue_ngap_id);
        assert_eq!(decoded.source_amf_ue_ngap_id, params.source_amf_ue_ngap_id);
        assert_eq!(
            decoded.ue_security_capabilities,
            params.ue_security_capabilities
        );
        assert_eq!(decoded.sessions, params.sessions);
    }

    #[test]
    fn test_path_switch_request_rejects_empty_sessions() {
        let mut params = sample_request_params();
        params.sessions.clear();
        assert!(build_path_switch_request(&params).is_err());
    }

    #[test]
    fn test_path_switch_request_acknowledge_roundtrip() {
        let params = PathSwitchRequestAcknowledgeParams {
            amf_ue_ngap_id: 4242,
            ran_ue_ngap_id: 7,
            next_hop_chaining_count: 2,
            next_hop_nh: [0x5A; 32],
            switched_sessions: vec![SwitchedSessionItem {
                pdu_session_id: 5,
                ul_tunnel: Some(GtpTunnelInfo {
                    address: "10.45.0.1".parse().unwrap(),
                    teid: 0x1234,
                }),
            }],
        };
        let bytes = encode_path_switch_request_acknowledge(&params).unwrap();
        let decoded = decode_path_switch_request_acknowledge(&bytes).unwrap();
        assert_eq!(decoded.amf_ue_ngap_id, params.amf_ue_ngap_id);
        assert_eq!(decoded.ran_ue_ngap_id, params.ran_ue_ngap_id);
        assert_eq!(
            decoded.next_hop_chaining_count,
            params.next_hop_chaining_count
        );
        assert_eq!(decoded.next_hop_nh, params.next_hop_nh);
        assert_eq!(decoded.switched_sessions, params.switched_sessions);
    }

    #[test]
    fn test_path_switch_ack_missing_security_context_rejected() {
        // Build an acknowledge without the mandatory SecurityContext IE
        let protocol_ies = vec![
            PathSwitchRequestAcknowledgeProtocolIEs_Entry {
                id: ProtocolIE_ID(ID_AMF_UE_NGAP_ID),
                criticality: Criticality(Criticality::IGNORE),
                value: PathSwitchRequestAcknowledgeProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(
                    AMF_UE_NGAP_ID(1),
                ),
            },
            PathSwitchRequestAcknowledgeProtocolIEs_Entry {
                id: ProtocolIE_ID(ID_RAN_UE_NGAP_ID),
                criticality: Criticality(Criticality::IGNORE),
                value: PathSwitchRequestAcknowledgeProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(
                    RAN_UE_NGAP_ID(2),
                ),
            },
        ];
        let pdu = NGAP_PDU::SuccessfulOutcome(SuccessfulOutcome {
            procedure_code: ProcedureCode(ID_PATH_SWITCH_REQUEST),
            criticality: Criticality(Criticality::REJECT),
            value: SuccessfulOutcomeValue::Id_PathSwitchRequest(PathSwitchRequestAcknowledge {
                protocol_i_es: PathSwitchRequestAcknowledgeProtocolIEs(protocol_ies),
            }),
        });
        let bytes = encode_ngap_pdu(&pdu).unwrap();
        let result = decode_path_switch_request_acknowledge(&bytes);
        assert!(matches!(
            result,
            Err(PathSwitchError::MissingMandatoryIe("SecurityContext"))
        ));
    }

    #[test]
    fn test_path_switch_request_truncated_rejected() {
        let params = sample_request_params();
        let bytes = encode_path_switch_request(&params).unwrap();
        assert!(decode_path_switch_request(&bytes[..bytes.len() / 2]).is_err());
    }
}
