//! PDU Session Resource Transfer Containers (TS 38.413 Section 9.3.4)
//!
//! Real APER encode/decode for the N2 SM transfer containers that are carried
//! as opaque OCTET STRINGs inside the PDU Session Resource procedures:
//!
//! - `PDUSessionResourceSetupRequestTransfer` (SMF → gNB, decode)
//! - `PDUSessionResourceSetupResponseTransfer` (gNB → SMF, encode)
//! - `PDUSessionResourceSetupUnsuccessfulTransfer` (gNB → SMF, encode)
//! - `PDUSessionResourceModifyRequestTransfer` (SMF → gNB, decode)
//! - `PDUSessionResourceModifyResponseTransfer` (gNB → SMF, encode)
//! - `PDUSessionResourceReleaseCommandTransfer` (SMF → gNB, decode)
//! - `PDUSessionResourceReleaseResponseTransfer` (gNB → SMF, encode)
//!
//! # Wire-format note (interop with the nextgcore peer)
//!
//! The request transfers (`...SetupRequestTransfer`, `...ModifyRequestTransfer`)
//! are extensible SEQUENCEs (TS 38.413 §9.3.4.x, `...`), so their conformant
//! X.691 Aligned-PER encoding begins with one outer-SEQUENCE extension-presence
//! bit. As of ngap-04 the core's `ogs-ngap` codec emits that bit, so we decode
//! (and encode) the full outer transfer SEQUENCE here, matching the core bit
//! for bit. The response-direction transfers are likewise plain extensible
//! SEQUENCEs whose encoding matches the core's decoder.

use std::net::IpAddr;

use bitvec::prelude::*;

use crate::codec::generated::*;
use crate::codec::{decode_aper, encode_aper, NgapCodecError};
use crate::procedures::ng_setup::NgSetupFailureCause;
use crate::procedures::ue_context_release::{build_cause, parse_cause};
use thiserror::Error;

/// Errors that can occur while encoding/decoding transfer containers
#[derive(Debug, Error)]
pub enum TransferError {
    /// Codec error during encoding/decoding
    #[error("Codec error: {0}")]
    CodecError(#[from] NgapCodecError),

    /// Missing mandatory IE
    #[error("Missing mandatory IE: {0}")]
    MissingMandatoryIe(&'static str),

    /// Invalid IE value
    #[error("Invalid IE value: {0}")]
    InvalidIeValue(String),
}

// ============================================================================
// GTP tunnel helpers
// ============================================================================

/// A GTP-U tunnel endpoint (transport address + TEID)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GtpTunnelInfo {
    /// Transport layer address of the tunnel endpoint
    pub address: IpAddr,
    /// GTP Tunnel Endpoint Identifier
    pub teid: u32,
}

impl GtpTunnelInfo {
    /// Builds the generated `UPTransportLayerInformation` for this tunnel
    fn to_asn(self) -> UPTransportLayerInformation {
        let mut bits: BitVec<u8, Msb0> = BitVec::new();
        match self.address {
            IpAddr::V4(v4) => {
                for byte in v4.octets() {
                    for i in (0..8).rev() {
                        bits.push((byte >> i) & 1 == 1);
                    }
                }
            }
            IpAddr::V6(v6) => {
                for byte in v6.octets() {
                    for i in (0..8).rev() {
                        bits.push((byte >> i) & 1 == 1);
                    }
                }
            }
        }
        UPTransportLayerInformation::GTPTunnel(GTPTunnel {
            transport_layer_address: TransportLayerAddress(bits),
            gtp_teid: GTP_TEID(self.teid.to_be_bytes().to_vec()),
            ie_extensions: None,
        })
    }

    /// Extracts tunnel info from the generated `UPTransportLayerInformation`
    fn from_asn(info: &UPTransportLayerInformation) -> Result<Self, TransferError> {
        let tunnel = match info {
            UPTransportLayerInformation::GTPTunnel(t) => t,
            UPTransportLayerInformation::Choice_Extensions(_) => {
                return Err(TransferError::InvalidIeValue(
                    "Unsupported UPTransportLayerInformation choice extension".to_string(),
                ))
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
                IpAddr::V4(octets.into())
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
                IpAddr::V6(octets.into())
            }
            len => {
                return Err(TransferError::InvalidIeValue(format!(
                    "Unsupported TransportLayerAddress length: {len} bits"
                )))
            }
        };

        if tunnel.gtp_teid.0.len() != 4 {
            return Err(TransferError::InvalidIeValue(format!(
                "Invalid GTP-TEID length: {}",
                tunnel.gtp_teid.0.len()
            )));
        }
        let teid = u32::from_be_bytes([
            tunnel.gtp_teid.0[0],
            tunnel.gtp_teid.0[1],
            tunnel.gtp_teid.0[2],
            tunnel.gtp_teid.0[3],
        ]);

        Ok(Self { address, teid })
    }
}

// ============================================================================
// User-plane security policy (SecurityIndication, TS 38.413 §9.3.1.27)
// ============================================================================

/// How strongly the SMF wants one kind of user-plane protection
/// (TS 38.413 §9.3.1.27; TS 33.501 §5.10.3, §6.6.1).
///
/// The three values are a real hierarchy of obligation, not a preference scale,
/// and the NG-RAN node's duties differ at each one:
///
/// - `Required` — the gNB **must** protect the DRB, and must **reject** the PDU
///   session if it cannot.
/// - `Preferred` — protect if able; establish the session either way.
/// - `NotNeeded` — the gNB should not protect the DRB.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpProtectionPolicy {
    /// `required`: protection is mandatory, and the session fails without it.
    Required,
    /// `preferred`: protect when possible.
    Preferred,
    /// `not-needed`: do not protect.
    NotNeeded,
}

impl UpProtectionPolicy {
    /// Whether a gNB unable to provide this protection must refuse the session.
    pub fn is_mandatory(self) -> bool {
        matches!(self, Self::Required)
    }

    /// Whether the gNB should turn this protection on when it can.
    pub fn wants_protection(self) -> bool {
        matches!(self, Self::Required | Self::Preferred)
    }
}

/// The maximum rate the UE will integrity-protect at
/// (`MaximumIntegrityProtectedDataRate`, TS 38.413 §9.3.1.72).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaxIntegrityProtectedDataRate {
    /// `bitrate64kbs`: the UE can only integrity-protect a signalling-rate flow.
    Bitrate64kbs,
    /// `maximum-UE-rate`: no rate restriction beyond the UE's own capability.
    MaximumUeRate,
}

/// The `SecurityIndication` IE: the SMF's user-plane security policy for one PDU
/// session (TS 38.413 §9.3.1.27).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UpSecurityPolicy {
    /// Integrity protection policy for the session's DRBs.
    pub integrity: UpProtectionPolicy,
    /// Confidentiality protection policy for the session's DRBs.
    pub confidentiality: UpProtectionPolicy,
    /// Present when integrity protection is not `not-needed`; the rate ceiling the
    /// UE indicated it can integrity-protect at.
    pub max_integrity_protected_data_rate: Option<MaxIntegrityProtectedDataRate>,
}

impl UpSecurityPolicy {
    /// The policy a session with **no** `SecurityIndication` IE runs under.
    ///
    /// TS 33.501 §5.10.3: the IE is optional, and when the SMF omits it the
    /// NG-RAN node applies its locally configured policy. `Preferred` for both is
    /// that policy here — it protects when the negotiated algorithms allow and
    /// never refuses a session, which is what a session the SMF said nothing about
    /// should get. `Required` would refuse sessions no SMF asked to have refused,
    /// and `NotNeeded` would silently leave user traffic in the clear.
    pub fn locally_configured_default() -> Self {
        Self {
            integrity: UpProtectionPolicy::Preferred,
            confidentiality: UpProtectionPolicy::Preferred,
            max_integrity_protected_data_rate: None,
        }
    }
}

/// Decode a `SecurityIndication`.
///
/// An unknown enumeration value — one past this schema's extension marker — is
/// read as `Preferred` rather than refused. It is the only value that cannot be
/// wrong in a dangerous direction: `Required` would fail sessions on an IE we did
/// not understand, and `NotNeeded` would drop protection on the SMF's say-so
/// without knowing that is what it said.
pub(crate) fn up_security_policy_from_asn(ind: &SecurityIndication) -> UpSecurityPolicy {
    let integrity = match ind.integrity_protection_indication.0 {
        IntegrityProtectionIndication::REQUIRED => UpProtectionPolicy::Required,
        IntegrityProtectionIndication::NOT_NEEDED => UpProtectionPolicy::NotNeeded,
        _ => UpProtectionPolicy::Preferred,
    };
    let confidentiality = match ind.confidentiality_protection_indication.0 {
        ConfidentialityProtectionIndication::REQUIRED => UpProtectionPolicy::Required,
        ConfidentialityProtectionIndication::NOT_NEEDED => UpProtectionPolicy::NotNeeded,
        _ => UpProtectionPolicy::Preferred,
    };
    let max_integrity_protected_data_rate = ind
        .maximum_integrity_protected_data_rate_ul
        .as_ref()
        .map(|rate| match rate.0 {
            MaximumIntegrityProtectedDataRate::BITRATE64KBS => {
                MaxIntegrityProtectedDataRate::Bitrate64kbs
            }
            _ => MaxIntegrityProtectedDataRate::MaximumUeRate,
        });
    UpSecurityPolicy {
        integrity,
        confidentiality,
        max_integrity_protected_data_rate,
    }
}

/// Encode a `SecurityIndication`.
///
/// `maximumIntegrityProtectedDataRate-UL` is **conditional**, not optional
/// (TS 38.413 §9.3.1.27): it is present when integrity protection is not
/// `not-needed`. So an encoder handed `None` with integrity wanted substitutes
/// `maximum-UE-rate`, which is the "no restriction" value and the only one that
/// does not invent a ceiling the UE never indicated.
pub(crate) fn up_security_policy_to_asn(policy: &UpSecurityPolicy) -> SecurityIndication {
    let integrity = IntegrityProtectionIndication(match policy.integrity {
        UpProtectionPolicy::Required => IntegrityProtectionIndication::REQUIRED,
        UpProtectionPolicy::Preferred => IntegrityProtectionIndication::PREFERRED,
        UpProtectionPolicy::NotNeeded => IntegrityProtectionIndication::NOT_NEEDED,
    });
    let confidentiality = ConfidentialityProtectionIndication(match policy.confidentiality {
        UpProtectionPolicy::Required => ConfidentialityProtectionIndication::REQUIRED,
        UpProtectionPolicy::Preferred => ConfidentialityProtectionIndication::PREFERRED,
        UpProtectionPolicy::NotNeeded => ConfidentialityProtectionIndication::NOT_NEEDED,
    });
    let rate = if policy.integrity == UpProtectionPolicy::NotNeeded {
        None
    } else {
        Some(MaximumIntegrityProtectedDataRate(
            match policy.max_integrity_protected_data_rate {
                Some(MaxIntegrityProtectedDataRate::Bitrate64kbs) => {
                    MaximumIntegrityProtectedDataRate::BITRATE64KBS
                }
                _ => MaximumIntegrityProtectedDataRate::MAXIMUM_UE_RATE,
            },
        ))
    };
    SecurityIndication {
        integrity_protection_indication: integrity,
        confidentiality_protection_indication: confidentiality,
        maximum_integrity_protected_data_rate_ul: rate,
        ie_extensions: None,
    }
}

// ============================================================================
// PDU Session Resource Setup Request Transfer (decode, TS 38.413 §9.3.4.1)
// ============================================================================

/// A QoS flow requested in a Setup Request Transfer
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QosFlowSetupInfo {
    /// QoS Flow Identifier (0..63)
    pub qfi: u8,
    /// 5QI when the flow uses standardized (non-dynamic) characteristics
    pub five_qi: Option<u16>,
    /// ARP priority level (1..15)
    pub arp_priority_level: u8,
}

/// Decoded PDU Session Resource Setup Request Transfer
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetupRequestTransferData {
    /// PDU Session AMBR downlink in bits/s (conditional)
    pub ambr_dl: Option<u64>,
    /// PDU Session AMBR uplink in bits/s (conditional)
    pub ambr_ul: Option<u64>,
    /// UL NG-U UP TNL information — the UPF N3 endpoint (mandatory)
    pub ul_tunnel: GtpTunnelInfo,
    /// PDU Session Type (mandatory)
    pub pdu_session_type: u8,
    /// QoS flows to set up (mandatory, at least one)
    pub qos_flows: Vec<QosFlowSetupInfo>,
    /// The SMF's user-plane security policy for this session (optional IE).
    ///
    /// `None` means the SMF sent no `SecurityIndication`; see
    /// [`UpSecurityPolicy::locally_configured_default`] for what a gNB then does.
    /// Kept as `Option` rather than defaulted here so a consumer can tell "the SMF
    /// asked for `preferred`" from "the SMF said nothing".
    pub security_indication: Option<UpSecurityPolicy>,
}

/// Decode a PDU Session Resource Setup Request Transfer.
///
/// The bytes are the full extensible outer SEQUENCE (TS 38.413 §9.3.4.1): a
/// leading APER extension-presence bit followed by the `ProtocolIE-Container`
/// (see module docs).
pub fn decode_setup_request_transfer(
    bytes: &[u8],
) -> Result<SetupRequestTransferData, TransferError> {
    let transfer: PDUSessionResourceSetupRequestTransfer = decode_aper(bytes)?;
    let container = transfer.protocol_i_es;

    let mut ambr_dl = None;
    let mut ambr_ul = None;
    let mut ul_tunnel = None;
    let mut pdu_session_type = None;
    let mut qos_flows: Option<Vec<QosFlowSetupInfo>> = None;
    let mut security_indication = None;

    for entry in &container.0 {
        match &entry.value {
            PDUSessionResourceSetupRequestTransferProtocolIEs_EntryValue::Id_PDUSessionAggregateMaximumBitRate(ambr) => {
                ambr_dl = Some(ambr.pdu_session_aggregate_maximum_bit_rate_dl.0);
                ambr_ul = Some(ambr.pdu_session_aggregate_maximum_bit_rate_ul.0);
            }
            PDUSessionResourceSetupRequestTransferProtocolIEs_EntryValue::Id_UL_NGU_UP_TNLInformation(info) => {
                ul_tunnel = Some(GtpTunnelInfo::from_asn(info)?);
            }
            PDUSessionResourceSetupRequestTransferProtocolIEs_EntryValue::Id_PDUSessionType(ty) => {
                pdu_session_type = Some(ty.0);
            }
            PDUSessionResourceSetupRequestTransferProtocolIEs_EntryValue::Id_QosFlowSetupRequestList(list) => {
                qos_flows = Some(
                    list.0
                        .iter()
                        .map(|item| QosFlowSetupInfo {
                            qfi: item.qos_flow_identifier.0,
                            five_qi: match &item.qos_flow_level_qos_parameters.qos_characteristics {
                                QosCharacteristics::NonDynamic5QI(d) => Some(d.five_qi.0 as u16),
                                QosCharacteristics::Dynamic5QI(d) => {
                                    d.five_qi.as_ref().map(|q| q.0 as u16)
                                }
                                QosCharacteristics::Choice_Extensions(_) => None,
                            },
                            arp_priority_level: item
                                .qos_flow_level_qos_parameters
                                .allocation_and_retention_priority
                                .priority_level_arp
                                .0,
                        })
                        .collect(),
                );
            }
            PDUSessionResourceSetupRequestTransferProtocolIEs_EntryValue::Id_SecurityIndication(ind) => {
                security_indication = Some(up_security_policy_from_asn(ind));
            }
            _ => {} // Optional IEs we do not act on (criticality handled by sender)
        }
    }

    let ul_tunnel = ul_tunnel.ok_or(TransferError::MissingMandatoryIe(
        "UL-NGU-UP-TNLInformation",
    ))?;
    let pdu_session_type =
        pdu_session_type.ok_or(TransferError::MissingMandatoryIe("PDUSessionType"))?;
    let qos_flows =
        qos_flows.ok_or(TransferError::MissingMandatoryIe("QosFlowSetupRequestList"))?;
    if qos_flows.is_empty() {
        return Err(TransferError::InvalidIeValue(
            "QosFlowSetupRequestList must contain at least one flow".to_string(),
        ));
    }

    Ok(SetupRequestTransferData {
        ambr_dl,
        ambr_ul,
        ul_tunnel,
        pdu_session_type,
        qos_flows,
        security_indication,
    })
}

/// Encode a PDU Session Resource Setup Request Transfer (used by tests and by
/// peer simulations acting as the SMF side).
pub fn encode_setup_request_transfer(
    data: &SetupRequestTransferData,
) -> Result<Vec<u8>, TransferError> {
    let mut entries = Vec::new();

    if let (Some(dl), Some(ul)) = (data.ambr_dl, data.ambr_ul) {
        entries.push(PDUSessionResourceSetupRequestTransferProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_PDU_SESSION_AGGREGATE_MAXIMUM_BIT_RATE),
            criticality: Criticality(Criticality::REJECT),
            value: PDUSessionResourceSetupRequestTransferProtocolIEs_EntryValue::Id_PDUSessionAggregateMaximumBitRate(
                PDUSessionAggregateMaximumBitRate {
                    pdu_session_aggregate_maximum_bit_rate_dl: BitRate(dl),
                    pdu_session_aggregate_maximum_bit_rate_ul: BitRate(ul),
                    ie_extensions: None,
                },
            ),
        });
    }

    entries.push(PDUSessionResourceSetupRequestTransferProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_UL_NGU_UP_TNL_INFORMATION),
        criticality: Criticality(Criticality::REJECT),
        value: PDUSessionResourceSetupRequestTransferProtocolIEs_EntryValue::Id_UL_NGU_UP_TNLInformation(
            data.ul_tunnel.to_asn(),
        ),
    });

    entries.push(PDUSessionResourceSetupRequestTransferProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_PDU_SESSION_TYPE),
        criticality: Criticality(Criticality::REJECT),
        value: PDUSessionResourceSetupRequestTransferProtocolIEs_EntryValue::Id_PDUSessionType(
            PDUSessionType(data.pdu_session_type),
        ),
    });

    let items: Vec<QosFlowSetupRequestItem> = data
        .qos_flows
        .iter()
        .map(|flow| QosFlowSetupRequestItem {
            qos_flow_identifier: QosFlowIdentifier(flow.qfi),
            qos_flow_level_qos_parameters: QosFlowLevelQosParameters {
                qos_characteristics: QosCharacteristics::NonDynamic5QI(NonDynamic5QIDescriptor {
                    five_qi: FiveQI(flow.five_qi.unwrap_or(9) as u8),
                    priority_level_qos: None,
                    averaging_window: None,
                    maximum_data_burst_volume: None,
                    ie_extensions: None,
                }),
                allocation_and_retention_priority: AllocationAndRetentionPriority {
                    priority_level_arp: PriorityLevelARP(flow.arp_priority_level),
                    pre_emption_capability: Pre_emptionCapability(
                        Pre_emptionCapability::SHALL_NOT_TRIGGER_PRE_EMPTION,
                    ),
                    pre_emption_vulnerability: Pre_emptionVulnerability(
                        Pre_emptionVulnerability::NOT_PRE_EMPTABLE,
                    ),
                    ie_extensions: None,
                },
                gbr_qos_information: None,
                reflective_qos_attribute: None,
                additional_qos_flow_information: None,
                ie_extensions: None,
            },
            e_rab_id: None,
            ie_extensions: None,
        })
        .collect();

    entries.push(PDUSessionResourceSetupRequestTransferProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_QOS_FLOW_SETUP_REQUEST_LIST),
        criticality: Criticality(Criticality::REJECT),
        value: PDUSessionResourceSetupRequestTransferProtocolIEs_EntryValue::Id_QosFlowSetupRequestList(
            QosFlowSetupRequestList(items),
        ),
    });

    // Appended last so a transfer with no policy is byte-identical to what this
    // encoder produced before issue #32, which is what keeps the existing
    // cross-decode tests against the core's codec meaningful.
    if let Some(policy) = &data.security_indication {
        entries.push(PDUSessionResourceSetupRequestTransferProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_SECURITY_INDICATION),
            // REJECT, per TS 38.413 §9.3.4.1: a gNB that cannot act on the SMF's
            // security policy must fail the session rather than establish it
            // unprotected.
            criticality: Criticality(Criticality::REJECT),
            value:
                PDUSessionResourceSetupRequestTransferProtocolIEs_EntryValue::Id_SecurityIndication(
                    up_security_policy_to_asn(policy),
                ),
        });
    }

    Ok(encode_aper(&PDUSessionResourceSetupRequestTransfer {
        protocol_i_es: PDUSessionResourceSetupRequestTransferProtocolIEs(entries),
    })?)
}

// ============================================================================
// PDU Session Resource Setup Response Transfer (encode, TS 38.413 §9.3.4.2)
// ============================================================================

/// A QoS flow that failed to set up, with its cause
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FailedQosFlow {
    /// QoS Flow Identifier
    pub qfi: u8,
    /// Failure cause
    pub cause: NgSetupFailureCause,
}

/// Parameters for the PDU Session Resource Setup Response Transfer
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetupResponseTransferParams {
    /// DL NG-U tunnel endpoint allocated by the gNB (mandatory)
    pub dl_tunnel: GtpTunnelInfo,
    /// QFIs accepted on the DL tunnel (mandatory, at least one)
    pub accepted_qfis: Vec<u8>,
    /// QoS flows that failed to set up (optional)
    pub failed_qos_flows: Vec<FailedQosFlow>,
}

/// Encode a PDU Session Resource Setup Response Transfer per TS 38.413 §9.3.4.2
pub fn encode_setup_response_transfer(
    params: &SetupResponseTransferParams,
) -> Result<Vec<u8>, TransferError> {
    if params.accepted_qfis.is_empty() {
        return Err(TransferError::InvalidIeValue(
            "AssociatedQosFlowList must contain at least one flow".to_string(),
        ));
    }

    let associated: Vec<AssociatedQosFlowItem> = params
        .accepted_qfis
        .iter()
        .map(|&qfi| AssociatedQosFlowItem {
            qos_flow_identifier: QosFlowIdentifier(qfi),
            qos_flow_mapping_indication: None,
            ie_extensions: None,
        })
        .collect();

    let failed = if params.failed_qos_flows.is_empty() {
        None
    } else {
        Some(QosFlowListWithCause(
            params
                .failed_qos_flows
                .iter()
                .map(|f| QosFlowWithCauseItem {
                    qos_flow_identifier: QosFlowIdentifier(f.qfi),
                    cause: build_cause(&f.cause),
                    ie_extensions: None,
                })
                .collect(),
        ))
    };

    let transfer = PDUSessionResourceSetupResponseTransfer {
        dl_qos_flow_per_tnl_information: QosFlowPerTNLInformation {
            up_transport_layer_information: params.dl_tunnel.to_asn(),
            associated_qos_flow_list: AssociatedQosFlowList(associated),
            ie_extensions: None,
        },
        additional_dl_qos_flow_per_tnl_information: None,
        security_result: None,
        qos_flow_failed_to_setup_list: failed,
        ie_extensions: None,
    };

    Ok(encode_aper(&transfer)?)
}

/// Decoded PDU Session Resource Setup Response Transfer
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetupResponseTransferData {
    /// DL NG-U tunnel endpoint
    pub dl_tunnel: GtpTunnelInfo,
    /// Accepted QFIs
    pub accepted_qfis: Vec<u8>,
    /// QoS flows that failed to set up
    pub failed_qos_flows: Vec<FailedQosFlow>,
}

/// Decode a PDU Session Resource Setup Response Transfer
pub fn decode_setup_response_transfer(
    bytes: &[u8],
) -> Result<SetupResponseTransferData, TransferError> {
    let transfer: PDUSessionResourceSetupResponseTransfer = decode_aper(bytes)?;

    let dl_tunnel = GtpTunnelInfo::from_asn(
        &transfer
            .dl_qos_flow_per_tnl_information
            .up_transport_layer_information,
    )?;

    let accepted_qfis = transfer
        .dl_qos_flow_per_tnl_information
        .associated_qos_flow_list
        .0
        .iter()
        .map(|item| item.qos_flow_identifier.0)
        .collect();

    let failed_qos_flows = transfer
        .qos_flow_failed_to_setup_list
        .map(|list| {
            list.0
                .iter()
                .map(|item| FailedQosFlow {
                    qfi: item.qos_flow_identifier.0,
                    cause: parse_cause(&item.cause),
                })
                .collect()
        })
        .unwrap_or_default();

    Ok(SetupResponseTransferData {
        dl_tunnel,
        accepted_qfis,
        failed_qos_flows,
    })
}

// ============================================================================
// PDU Session Resource Setup Unsuccessful Transfer (TS 38.413 §9.3.4.16)
// ============================================================================

/// Encode a PDU Session Resource Setup Unsuccessful Transfer with the given cause
pub fn encode_setup_unsuccessful_transfer(
    cause: &NgSetupFailureCause,
) -> Result<Vec<u8>, TransferError> {
    let transfer = PDUSessionResourceSetupUnsuccessfulTransfer {
        cause: build_cause(cause),
        criticality_diagnostics: None,
        ie_extensions: None,
    };
    Ok(encode_aper(&transfer)?)
}

/// Decode a PDU Session Resource Setup Unsuccessful Transfer, returning the cause
pub fn decode_setup_unsuccessful_transfer(
    bytes: &[u8],
) -> Result<NgSetupFailureCause, TransferError> {
    let transfer: PDUSessionResourceSetupUnsuccessfulTransfer = decode_aper(bytes)?;
    Ok(parse_cause(&transfer.cause))
}

// ============================================================================
// Handover Required Transfer (encode, TS 38.413 §9.3.4.12)
// ============================================================================

/// Encode a `PDUSessionResourceInformationItem`'s `HandoverRequiredTransfer`.
///
/// Its only member is `directForwardingPathAvailability`, so an all-absent transfer is
/// the truthful encoding for a node that forwards nothing during a handover — which is
/// this one. `vec![0x00]`, which this replaces, is not a decodable transfer at all
/// (issue #39).
pub fn encode_handover_required_transfer(
    direct_forwarding_available: bool,
) -> Result<Vec<u8>, TransferError> {
    let transfer = HandoverRequiredTransfer {
        direct_forwarding_path_availability: direct_forwarding_available.then_some(
            DirectForwardingPathAvailability(
                DirectForwardingPathAvailability::DIRECT_PATH_AVAILABLE,
            ),
        ),
        ie_extensions: None,
    };
    Ok(encode_aper(&transfer)?)
}

/// Decode one, reporting whether a direct forwarding path was offered.
pub fn decode_handover_required_transfer(bytes: &[u8]) -> Result<bool, TransferError> {
    let transfer: HandoverRequiredTransfer = decode_aper(bytes)?;
    Ok(transfer.direct_forwarding_path_availability.is_some())
}

// ============================================================================
// Handover Request Acknowledge Transfer (encode, TS 38.413 §9.3.4.13)
// ============================================================================

/// What the target NG-RAN node actually applied to a session's DRBs, reported back
/// per admitted session (`SecurityResult`, TS 38.413 §9.3.1.59).
///
/// The counterpart of the `SecurityIndication` the SMF sent (issue #32): that states
/// an obligation and this states an outcome. Two booleans rather than a reuse of
/// `UpSecurityPolicy`, because `performed`/`not-performed` has no third value — a
/// target does not report "preferred".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct UpSecurityResult {
    /// `integrityProtectionResult`: whether the DRBs are integrity protected.
    pub integrity_performed: bool,
    /// `confidentialityProtectionResult`: whether the DRBs are ciphered.
    pub confidentiality_performed: bool,
}

/// Parameters for a per-session Handover Request Acknowledge Transfer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HandoverAckTransferParams {
    /// The DL NG-U tunnel the **target** allocated. Mandatory: this is the endpoint
    /// the UPF will switch downlink traffic to, and it is the whole reason the AMF
    /// asked.
    pub dl_tunnel: GtpTunnelInfo,
    /// QFIs the target admitted, with data forwarding accepted for each.
    pub admitted_qfis: Vec<u8>,
    /// QoS flows the target could not admit, with causes.
    pub failed_qos_flows: Vec<FailedQosFlow>,
    /// What user-plane security the target applied, when it applied any.
    ///
    /// `None` omits the IE, which is what a target with no user-plane security to
    /// report should send — claiming `not-performed` for both would be a *decision* the
    /// target had not actually made.
    pub security_result: Option<UpSecurityResult>,
}

/// A decoded Handover Request Acknowledge Transfer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HandoverAckTransferData {
    /// The DL NG-U tunnel the target allocated.
    pub dl_tunnel: GtpTunnelInfo,
    /// QFIs the target admitted.
    pub admitted_qfis: Vec<u8>,
    /// What user-plane security the target applied, if it said.
    pub security_result: Option<UpSecurityResult>,
}

/// Encode a `PDUSessionResourceHandoverRequestAckTransfer` (TS 38.413 §9.3.4.13).
///
/// This is what the target NG-RAN node returns per **admitted** session, and it is
/// what makes the admission mean something: before issue #39 the gNB sent
/// `vec![0x00]`, which is not a decodable transfer at all — so an AMF learned nothing
/// about where to switch the downlink tunnel, and the data plane went nowhere.
pub fn encode_handover_ack_transfer(
    params: &HandoverAckTransferParams,
) -> Result<Vec<u8>, TransferError> {
    if params.admitted_qfis.is_empty() {
        return Err(TransferError::InvalidIeValue(
            "QosFlowListWithDataForwarding must contain at least one flow: a session \
             admitted with no QoS flow carries nothing"
                .to_string(),
        ));
    }
    let admitted: Vec<QosFlowItemWithDataForwarding> = params
        .admitted_qfis
        .iter()
        .map(|&qfi| QosFlowItemWithDataForwarding {
            qos_flow_identifier: QosFlowIdentifier(qfi),
            // The flow is admitted, so forwarding of its data to this target is
            // accepted. Omitting the IE would leave the source unable to tell whether
            // in-flight downlink data should be forwarded, and it would hold it.
            data_forwarding_accepted: Some(DataForwardingAccepted(
                DataForwardingAccepted::DATA_FORWARDING_ACCEPTED,
            )),
            ie_extensions: None,
        })
        .collect();

    let failed = if params.failed_qos_flows.is_empty() {
        None
    } else {
        Some(QosFlowListWithCause(
            params
                .failed_qos_flows
                .iter()
                .map(|f| QosFlowWithCauseItem {
                    qos_flow_identifier: QosFlowIdentifier(f.qfi),
                    cause: build_cause(&f.cause),
                    ie_extensions: None,
                })
                .collect(),
        ))
    };

    let transfer = HandoverRequestAcknowledgeTransfer {
        dl_ngu_up_tnl_information: params.dl_tunnel.to_asn(),
        // No separate data-forwarding tunnel: this simulator forwards nothing during a
        // handover, and advertising an endpoint it would not read from is worse than
        // advertising none.
        dl_forwarding_up_tnl_information: None,
        security_result: params.security_result.map(|r| SecurityResult {
            integrity_protection_result: IntegrityProtectionResult(if r.integrity_performed {
                IntegrityProtectionResult::PERFORMED
            } else {
                IntegrityProtectionResult::NOT_PERFORMED
            }),
            confidentiality_protection_result: ConfidentialityProtectionResult(
                if r.confidentiality_performed {
                    ConfidentialityProtectionResult::PERFORMED
                } else {
                    ConfidentialityProtectionResult::NOT_PERFORMED
                },
            ),
            ie_extensions: None,
        }),
        qos_flow_setup_response_list: QosFlowListWithDataForwarding(admitted),
        qos_flow_failed_to_setup_list: failed,
        data_forwarding_response_drb_list: None,
        ie_extensions: None,
    };
    Ok(encode_aper(&transfer)?)
}

/// Decode a `PDUSessionResourceHandoverRequestAckTransfer`.
pub fn decode_handover_ack_transfer(
    bytes: &[u8],
) -> Result<HandoverAckTransferData, TransferError> {
    let transfer: HandoverRequestAcknowledgeTransfer = decode_aper(bytes)?;
    Ok(HandoverAckTransferData {
        dl_tunnel: GtpTunnelInfo::from_asn(&transfer.dl_ngu_up_tnl_information)?,
        admitted_qfis: transfer
            .qos_flow_setup_response_list
            .0
            .iter()
            .map(|item| item.qos_flow_identifier.0)
            .collect(),
        security_result: transfer.security_result.as_ref().map(|r| UpSecurityResult {
            integrity_performed: r.integrity_protection_result.0
                == IntegrityProtectionResult::PERFORMED,
            confidentiality_performed: r.confidentiality_protection_result.0
                == ConfidentialityProtectionResult::PERFORMED,
        }),
    })
}

// ============================================================================
// PDU Session Resource Modify Request Transfer (decode, TS 38.413 §9.3.4.3)
// ============================================================================

/// Decoded PDU Session Resource Modify Request Transfer
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ModifyRequestTransferData {
    /// PDU Session AMBR downlink in bits/s (optional)
    pub ambr_dl: Option<u64>,
    /// PDU Session AMBR uplink in bits/s (optional)
    pub ambr_ul: Option<u64>,
    /// New UL NG-U tunnel from the UPF (optional, from UL-NGU-UP-TNLModifyList)
    pub new_ul_tunnel: Option<GtpTunnelInfo>,
    /// QFIs to add or modify (optional)
    pub qos_flows_add_or_modify: Vec<u8>,
    /// QFIs to release with their causes (optional)
    pub qos_flows_to_release: Vec<FailedQosFlow>,
}

/// Decode a PDU Session Resource Modify Request Transfer.
///
/// The bytes are the full extensible outer SEQUENCE (TS 38.413 §9.3.4.3): a
/// leading APER extension-presence bit followed by the `ProtocolIE-Container`
/// (see module docs). The core's shared `ogs-ngap` encode/decode helper emits
/// this bit for the Modify transfer too (ngap-04).
pub fn decode_modify_request_transfer(
    bytes: &[u8],
) -> Result<ModifyRequestTransferData, TransferError> {
    let transfer: PDUSessionResourceModifyRequestTransfer = decode_aper(bytes)?;
    let container = transfer.protocol_i_es;

    let mut data = ModifyRequestTransferData::default();

    for entry in &container.0 {
        match &entry.value {
            PDUSessionResourceModifyRequestTransferProtocolIEs_EntryValue::Id_PDUSessionAggregateMaximumBitRate(ambr) => {
                data.ambr_dl = Some(ambr.pdu_session_aggregate_maximum_bit_rate_dl.0);
                data.ambr_ul = Some(ambr.pdu_session_aggregate_maximum_bit_rate_ul.0);
            }
            PDUSessionResourceModifyRequestTransferProtocolIEs_EntryValue::Id_UL_NGU_UP_TNLModifyList(list) => {
                if let Some(item) = list.0.first() {
                    data.new_ul_tunnel =
                        Some(GtpTunnelInfo::from_asn(&item.ul_ngu_up_tnl_information)?);
                }
            }
            PDUSessionResourceModifyRequestTransferProtocolIEs_EntryValue::Id_QosFlowAddOrModifyRequestList(list) => {
                data.qos_flows_add_or_modify =
                    list.0.iter().map(|item| item.qos_flow_identifier.0).collect();
            }
            PDUSessionResourceModifyRequestTransferProtocolIEs_EntryValue::Id_QosFlowToReleaseList(list) => {
                data.qos_flows_to_release = list
                    .0
                    .iter()
                    .map(|item| FailedQosFlow {
                        qfi: item.qos_flow_identifier.0,
                        cause: parse_cause(&item.cause),
                    })
                    .collect();
            }
            _ => {}
        }
    }

    Ok(data)
}

// ============================================================================
// PDU Session Resource Modify Response Transfer (encode, TS 38.413 §9.3.4.4)
// ============================================================================

/// Parameters for the PDU Session Resource Modify Response Transfer
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ModifyResponseTransferParams {
    /// New DL NG-U tunnel allocated by the gNB (optional)
    pub dl_tunnel: Option<GtpTunnelInfo>,
    /// UL NG-U tunnel acknowledged by the gNB (optional)
    pub ul_tunnel: Option<GtpTunnelInfo>,
    /// QFIs successfully added or modified (optional)
    pub modified_qfis: Vec<u8>,
    /// QoS flows that failed to add or modify (optional)
    pub failed_qos_flows: Vec<FailedQosFlow>,
}

/// Encode a PDU Session Resource Modify Response Transfer per TS 38.413 §9.3.4.4
pub fn encode_modify_response_transfer(
    params: &ModifyResponseTransferParams,
) -> Result<Vec<u8>, TransferError> {
    let response_list = if params.modified_qfis.is_empty() {
        None
    } else {
        Some(QosFlowAddOrModifyResponseList(
            params
                .modified_qfis
                .iter()
                .map(|&qfi| QosFlowAddOrModifyResponseItem {
                    qos_flow_identifier: QosFlowIdentifier(qfi),
                    ie_extensions: None,
                })
                .collect(),
        ))
    };

    let failed = if params.failed_qos_flows.is_empty() {
        None
    } else {
        Some(QosFlowListWithCause(
            params
                .failed_qos_flows
                .iter()
                .map(|f| QosFlowWithCauseItem {
                    qos_flow_identifier: QosFlowIdentifier(f.qfi),
                    cause: build_cause(&f.cause),
                    ie_extensions: None,
                })
                .collect(),
        ))
    };

    let transfer = PDUSessionResourceModifyResponseTransfer {
        dl_ngu_up_tnl_information: params.dl_tunnel.map(GtpTunnelInfo::to_asn),
        ul_ngu_up_tnl_information: params.ul_tunnel.map(GtpTunnelInfo::to_asn),
        qos_flow_add_or_modify_response_list: response_list,
        additional_dl_qos_flow_per_tnl_information: None,
        qos_flow_failed_to_add_or_modify_list: failed,
        ie_extensions: None,
    };

    Ok(encode_aper(&transfer)?)
}

/// Decode a PDU Session Resource Modify Response Transfer
pub fn decode_modify_response_transfer(
    bytes: &[u8],
) -> Result<ModifyResponseTransferParams, TransferError> {
    let transfer: PDUSessionResourceModifyResponseTransfer = decode_aper(bytes)?;

    let dl_tunnel = transfer
        .dl_ngu_up_tnl_information
        .as_ref()
        .map(GtpTunnelInfo::from_asn)
        .transpose()?;
    let ul_tunnel = transfer
        .ul_ngu_up_tnl_information
        .as_ref()
        .map(GtpTunnelInfo::from_asn)
        .transpose()?;

    let modified_qfis = transfer
        .qos_flow_add_or_modify_response_list
        .map(|list| {
            list.0
                .iter()
                .map(|item| item.qos_flow_identifier.0)
                .collect()
        })
        .unwrap_or_default();

    let failed_qos_flows = transfer
        .qos_flow_failed_to_add_or_modify_list
        .map(|list| {
            list.0
                .iter()
                .map(|item| FailedQosFlow {
                    qfi: item.qos_flow_identifier.0,
                    cause: parse_cause(&item.cause),
                })
                .collect()
        })
        .unwrap_or_default();

    Ok(ModifyResponseTransferParams {
        dl_tunnel,
        ul_tunnel,
        modified_qfis,
        failed_qos_flows,
    })
}

// ============================================================================
// PDU Session Resource Modify Unsuccessful Transfer (TS 38.413 §9.3.4.17)
// ============================================================================

/// Encode a PDU Session Resource Modify Unsuccessful Transfer with the given cause
pub fn encode_modify_unsuccessful_transfer(
    cause: &NgSetupFailureCause,
) -> Result<Vec<u8>, TransferError> {
    let transfer = PDUSessionResourceModifyUnsuccessfulTransfer {
        cause: build_cause(cause),
        criticality_diagnostics: None,
        ie_extensions: None,
    };
    Ok(encode_aper(&transfer)?)
}

/// Decode a PDU Session Resource Modify Unsuccessful Transfer, returning the cause
pub fn decode_modify_unsuccessful_transfer(
    bytes: &[u8],
) -> Result<NgSetupFailureCause, TransferError> {
    let transfer: PDUSessionResourceModifyUnsuccessfulTransfer = decode_aper(bytes)?;
    Ok(parse_cause(&transfer.cause))
}

// ============================================================================
// PDU Session Resource Release Command/Response Transfer (§9.3.4.11/12)
// ============================================================================

/// Decode a PDU Session Resource Release Command Transfer, returning the cause
pub fn decode_release_command_transfer(bytes: &[u8]) -> Result<NgSetupFailureCause, TransferError> {
    let transfer: PDUSessionResourceReleaseCommandTransfer = decode_aper(bytes)?;
    Ok(parse_cause(&transfer.cause))
}

/// Encode a PDU Session Resource Release Response Transfer.
///
/// The Rel-15 type only carries optional iE-Extensions, so the strict encoding
/// is a single preamble octet (`0x00`).
pub fn encode_release_response_transfer() -> Result<Vec<u8>, TransferError> {
    let transfer = PDUSessionResourceReleaseResponseTransfer {
        ie_extensions: None,
    };
    Ok(encode_aper(&transfer)?)
}

/// Decode a PDU Session Resource Release Response Transfer
pub fn decode_release_response_transfer(bytes: &[u8]) -> Result<(), TransferError> {
    let _transfer: PDUSessionResourceReleaseResponseTransfer = decode_aper(bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::procedures::ng_setup::{NasCause, ProtocolCause, RadioNetworkCause};

    fn sample_tunnel() -> GtpTunnelInfo {
        GtpTunnelInfo {
            address: "10.45.0.1".parse().unwrap(),
            teid: 0x0000_1234,
        }
    }

    // ------------------------------------------------------------------
    // Setup Response Transfer
    // ------------------------------------------------------------------

    /// Byte-level conformance vector mirroring the strict nextgcore decoder
    /// layout (`libs/ogs-ngap/src/transfer.rs`): minimal response transfer with
    /// tunnel 10.45.0.1 / TEID 0x00001234 and the single QFI 1.
    ///
    /// Layout (X.691 APER):
    /// - preamble: ext + 4 option bits, all 0 (5 bits)
    /// - QosFlowPerTNLInformation: ext + opt (2 bits)
    /// - UPTransportLayerInformation choice index (1 bit) => first byte 0x00
    /// - GTPTunnel: ext + opt (2 bits)
    /// - TransportLayerAddress: ext bit + 8-bit length (31 = 32 bits - 1),
    ///   pad to octet boundary => 0x03 0xE0
    /// - address octets 0A 2D 00 01, TEID 00 00 12 34 (octet aligned)
    /// - AssociatedQosFlowList length (6 bits, value 0 = 1 item)
    /// - item preamble (3 bits) + QFI ext bit + 6-bit value 1 => 0x00 0x01
    const SETUP_RESPONSE_VECTOR: [u8; 13] = [
        0x00, 0x03, 0xE0, 0x0A, 0x2D, 0x00, 0x01, 0x00, 0x00, 0x12, 0x34, 0x00, 0x01,
    ];

    #[test]
    fn test_setup_response_transfer_matches_strict_peer_layout() {
        let params = SetupResponseTransferParams {
            dl_tunnel: sample_tunnel(),
            accepted_qfis: vec![1],
            failed_qos_flows: vec![],
        };
        let bytes = encode_setup_response_transfer(&params).unwrap();
        assert_eq!(bytes, SETUP_RESPONSE_VECTOR);
    }

    #[test]
    fn test_setup_response_transfer_roundtrip() {
        let params = SetupResponseTransferParams {
            dl_tunnel: sample_tunnel(),
            accepted_qfis: vec![1, 5],
            failed_qos_flows: vec![FailedQosFlow {
                qfi: 9,
                cause: NgSetupFailureCause::RadioNetwork(RadioNetworkCause::UnkownQosFlowId),
            }],
        };
        let bytes = encode_setup_response_transfer(&params).unwrap();
        let decoded = decode_setup_response_transfer(&bytes).unwrap();
        assert_eq!(decoded.dl_tunnel, params.dl_tunnel);
        assert_eq!(decoded.accepted_qfis, params.accepted_qfis);
        assert_eq!(decoded.failed_qos_flows.len(), 1);
        assert_eq!(decoded.failed_qos_flows[0].qfi, 9);
    }

    #[test]
    fn test_setup_response_transfer_ipv6_roundtrip() {
        let params = SetupResponseTransferParams {
            dl_tunnel: GtpTunnelInfo {
                address: "2001:db8::1".parse().unwrap(),
                teid: 0xDEAD_BEEF,
            },
            accepted_qfis: vec![63],
            failed_qos_flows: vec![],
        };
        let bytes = encode_setup_response_transfer(&params).unwrap();
        let decoded = decode_setup_response_transfer(&bytes).unwrap();
        assert_eq!(decoded.dl_tunnel, params.dl_tunnel);
        assert_eq!(decoded.accepted_qfis, vec![63]);
    }

    #[test]
    fn test_setup_response_transfer_rejects_empty_qfi_list() {
        let params = SetupResponseTransferParams {
            dl_tunnel: sample_tunnel(),
            accepted_qfis: vec![],
            failed_qos_flows: vec![],
        };
        assert!(encode_setup_response_transfer(&params).is_err());
    }

    #[test]
    fn test_setup_response_transfer_rejects_truncated() {
        let params = SetupResponseTransferParams {
            dl_tunnel: sample_tunnel(),
            accepted_qfis: vec![1],
            failed_qos_flows: vec![],
        };
        let bytes = encode_setup_response_transfer(&params).unwrap();
        assert!(decode_setup_response_transfer(&bytes[..bytes.len() / 2]).is_err());
    }

    // ------------------------------------------------------------------
    // Setup Request Transfer
    // ------------------------------------------------------------------

    fn sample_request() -> SetupRequestTransferData {
        SetupRequestTransferData {
            ambr_dl: Some(1_000_000_000),
            ambr_ul: Some(500_000_000),
            ul_tunnel: sample_tunnel(),
            pdu_session_type: PDUSessionType::IPV4,
            qos_flows: vec![QosFlowSetupInfo {
                qfi: 1,
                five_qi: Some(9),
                arp_priority_level: 8,
            }],
            security_indication: None,
        }
    }

    /// #39, criterion 2: the Handover Request Acknowledge Transfer is a real decodable
    /// `PDUSessionResourceHandoverRequestAckTransfer`, not `vec![0x00]`.
    #[test]
    fn a_handover_ack_transfer_round_trips() {
        let params = HandoverAckTransferParams {
            dl_tunnel: GtpTunnelInfo {
                address: "10.45.0.7".parse().unwrap(),
                teid: 0xDEAD_BEEF,
            },
            admitted_qfis: vec![1, 9],
            failed_qos_flows: vec![],
            security_result: Some(UpSecurityResult {
                integrity_performed: true,
                confidentiality_performed: false,
            }),
        };
        let bytes = encode_handover_ack_transfer(&params).expect("encode");
        assert!(
            bytes.len() > 1,
            "the old placeholder was a single 0x00 byte, which decodes as nothing"
        );
        let data = decode_handover_ack_transfer(&bytes).expect("decode");
        assert_eq!(
            data.dl_tunnel, params.dl_tunnel,
            "the DL tunnel is the whole point: it is where the UPF switches traffic to"
        );
        assert_eq!(data.admitted_qfis, vec![1, 9]);
        assert_eq!(
            data.security_result,
            Some(UpSecurityResult {
                integrity_performed: true,
                confidentiality_performed: false
            }),
            "the two protections are reported independently, as the SMF asked for them \
             independently (issue #32)"
        );
    }

    /// The old placeholder is refused by the decoder, so "it decodes" is a real check.
    #[test]
    fn the_old_placeholder_ack_transfer_does_not_decode() {
        assert!(
            decode_handover_ack_transfer(&[0x00]).is_err(),
            "vec![0x00] must NOT decode, or criterion 2's round trip proves nothing"
        );
        assert!(decode_handover_ack_transfer(&[]).is_err());
    }

    /// An absent `securityResult` stays absent, and a session admitted with no QoS flow
    /// is refused rather than encoded as an empty list.
    #[test]
    fn an_absent_security_result_stays_absent_and_an_empty_flow_list_is_refused() {
        let mut params = HandoverAckTransferParams {
            dl_tunnel: GtpTunnelInfo {
                address: "10.45.0.7".parse().unwrap(),
                teid: 1,
            },
            admitted_qfis: vec![1],
            failed_qos_flows: vec![],
            security_result: None,
        };
        let data =
            decode_handover_ack_transfer(&encode_handover_ack_transfer(&params).unwrap()).unwrap();
        assert_eq!(
            data.security_result, None,
            "a target with nothing to report must not claim `not-performed` for both: \
             that is a decision it did not make"
        );
        params.admitted_qfis.clear();
        assert!(
            encode_handover_ack_transfer(&params).is_err(),
            "a session admitted with no QoS flow carries nothing"
        );
    }

    #[test]
    fn test_setup_request_transfer_roundtrip() {
        let data = sample_request();
        let bytes = encode_setup_request_transfer(&data).unwrap();
        let decoded = decode_setup_request_transfer(&bytes).unwrap();
        assert_eq!(decoded, data);
    }

    /// #32, criterion 1: every `SecurityIndication` the SMF can send survives a
    /// round trip onto the parsed setup item.
    #[test]
    fn every_security_indication_round_trips_onto_the_parsed_item() {
        use UpProtectionPolicy::*;
        for integrity in [Required, Preferred, NotNeeded] {
            for confidentiality in [Required, Preferred, NotNeeded] {
                for rate in [
                    None,
                    Some(MaxIntegrityProtectedDataRate::Bitrate64kbs),
                    Some(MaxIntegrityProtectedDataRate::MaximumUeRate),
                ] {
                    let mut data = sample_request();
                    data.security_indication = Some(UpSecurityPolicy {
                        integrity,
                        confidentiality,
                        max_integrity_protected_data_rate: rate,
                    });
                    let bytes = encode_setup_request_transfer(&data).unwrap();
                    let decoded = decode_setup_request_transfer(&bytes)
                        .expect("a transfer carrying a SecurityIndication must decode");
                    let got = decoded
                        .security_indication
                        .expect("the policy must reach the parsed item");
                    assert_eq!(got.integrity, integrity);
                    assert_eq!(got.confidentiality, confidentiality);
                    // The rate IE is conditional on integrity not being
                    // `not-needed`, so it is absent exactly then -- and present
                    // otherwise even when the caller supplied none.
                    if integrity == NotNeeded {
                        assert_eq!(
                            got.max_integrity_protected_data_rate, None,
                            "the rate IE must be absent when integrity is not needed"
                        );
                    } else {
                        assert_eq!(
                            got.max_integrity_protected_data_rate,
                            Some(rate.unwrap_or(MaxIntegrityProtectedDataRate::MaximumUeRate)),
                            "an absent rate must encode as maximum-UE-rate, not vanish"
                        );
                    }
                }
            }
        }
    }

    /// A transfer with no `SecurityIndication` yields `None`, distinguishable from
    /// an SMF that explicitly asked for `preferred`.
    #[test]
    fn an_absent_security_indication_is_none_and_not_a_default() {
        let bytes = encode_setup_request_transfer(&sample_request()).unwrap();
        let decoded = decode_setup_request_transfer(&bytes).unwrap();
        assert_eq!(decoded.security_indication, None);
        assert_ne!(
            decoded.security_indication,
            Some(UpSecurityPolicy::locally_configured_default()),
            "\"the SMF said nothing\" must not be confused with \"the SMF said preferred\""
        );
        // And the local default is what a gNB then applies.
        let fallback = UpSecurityPolicy::locally_configured_default();
        assert!(fallback.integrity.wants_protection());
        assert!(fallback.confidentiality.wants_protection());
        assert!(
            !fallback.integrity.is_mandatory() && !fallback.confidentiality.is_mandatory(),
            "the local default must never refuse a session the SMF did not ask to refuse"
        );
    }

    #[test]
    fn test_setup_request_transfer_missing_mandatory_rejected() {
        // Container holding only the PDU session type: missing UL TNL info
        // and QoS flow list must be rejected.
        let entries = vec![PDUSessionResourceSetupRequestTransferProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_PDU_SESSION_TYPE),
            criticality: Criticality(Criticality::REJECT),
            value: PDUSessionResourceSetupRequestTransferProtocolIEs_EntryValue::Id_PDUSessionType(
                PDUSessionType(PDUSessionType::IPV4),
            ),
        }];
        let bytes = encode_aper(&PDUSessionResourceSetupRequestTransfer {
            protocol_i_es: PDUSessionResourceSetupRequestTransferProtocolIEs(entries),
        })
        .unwrap();
        let result = decode_setup_request_transfer(&bytes);
        assert!(matches!(
            result,
            Err(TransferError::MissingMandatoryIe(
                "UL-NGU-UP-TNLInformation"
            ))
        ));
    }

    #[test]
    fn test_setup_request_transfer_rejects_garbage() {
        assert!(decode_setup_request_transfer(&[0xFF]).is_err());
        assert!(decode_setup_request_transfer(&[]).is_err());
    }

    // ------------------------------------------------------------------
    // Setup Unsuccessful Transfer
    // ------------------------------------------------------------------

    #[test]
    fn test_setup_unsuccessful_transfer_roundtrip() {
        let cause =
            NgSetupFailureCause::RadioNetwork(RadioNetworkCause::RadioResourcesNotAvailable);
        let bytes = encode_setup_unsuccessful_transfer(&cause).unwrap();
        let decoded = decode_setup_unsuccessful_transfer(&bytes).unwrap();
        assert_eq!(decoded, cause);

        let cause = NgSetupFailureCause::Protocol(ProtocolCause::TransferSyntaxError);
        let bytes = encode_setup_unsuccessful_transfer(&cause).unwrap();
        let decoded = decode_setup_unsuccessful_transfer(&bytes).unwrap();
        assert_eq!(decoded, cause);
    }

    #[test]
    fn test_setup_unsuccessful_transfer_rejects_truncated() {
        let cause = NgSetupFailureCause::Nas(NasCause::NormalRelease);
        let bytes = encode_setup_unsuccessful_transfer(&cause).unwrap();
        assert!(decode_setup_unsuccessful_transfer(&bytes[..0]).is_err());
    }

    // ------------------------------------------------------------------
    // Modify Request/Response Transfer
    // ------------------------------------------------------------------

    #[test]
    fn test_modify_request_transfer_decode() {
        // Build a modify request container with a new UL tunnel and one flow
        let entries = vec![
            PDUSessionResourceModifyRequestTransferProtocolIEs_Entry {
                id: ProtocolIE_ID(ID_UL_NGU_UP_TNL_MODIFY_LIST),
                criticality: Criticality(Criticality::REJECT),
                value: PDUSessionResourceModifyRequestTransferProtocolIEs_EntryValue::Id_UL_NGU_UP_TNLModifyList(
                    UL_NGU_UP_TNLModifyList(vec![UL_NGU_UP_TNLModifyItem {
                        ul_ngu_up_tnl_information: sample_tunnel().to_asn(),
                        dl_ngu_up_tnl_information: sample_tunnel().to_asn(),
                        ie_extensions: None,
                    }]),
                ),
            },
            PDUSessionResourceModifyRequestTransferProtocolIEs_Entry {
                id: ProtocolIE_ID(ID_QOS_FLOW_ADD_OR_MODIFY_REQUEST_LIST),
                criticality: Criticality(Criticality::REJECT),
                value: PDUSessionResourceModifyRequestTransferProtocolIEs_EntryValue::Id_QosFlowAddOrModifyRequestList(
                    QosFlowAddOrModifyRequestList(vec![QosFlowAddOrModifyRequestItem {
                        qos_flow_identifier: QosFlowIdentifier(2),
                        qos_flow_level_qos_parameters: None,
                        e_rab_id: None,
                        ie_extensions: None,
                    }]),
                ),
            },
        ];
        let bytes = encode_aper(&PDUSessionResourceModifyRequestTransfer {
            protocol_i_es: PDUSessionResourceModifyRequestTransferProtocolIEs(entries),
        })
        .unwrap();
        let decoded = decode_modify_request_transfer(&bytes).unwrap();
        assert_eq!(decoded.new_ul_tunnel, Some(sample_tunnel()));
        assert_eq!(decoded.qos_flows_add_or_modify, vec![2]);
    }

    #[test]
    fn test_modify_response_transfer_roundtrip() {
        let params = ModifyResponseTransferParams {
            dl_tunnel: Some(sample_tunnel()),
            ul_tunnel: None,
            modified_qfis: vec![2],
            failed_qos_flows: vec![],
        };
        let bytes = encode_modify_response_transfer(&params).unwrap();
        let decoded = decode_modify_response_transfer(&bytes).unwrap();
        assert_eq!(decoded, params);
    }

    #[test]
    fn test_modify_request_transfer_rejects_garbage() {
        assert!(decode_modify_request_transfer(&[0xFF, 0xFF, 0xFF]).is_err());
    }

    #[test]
    fn test_modify_unsuccessful_transfer_roundtrip() {
        let cause = NgSetupFailureCause::RadioNetwork(RadioNetworkCause::UnknownPduSessionId);
        let bytes = encode_modify_unsuccessful_transfer(&cause).unwrap();
        let decoded = decode_modify_unsuccessful_transfer(&bytes).unwrap();
        assert_eq!(decoded, cause);
    }

    // ------------------------------------------------------------------
    // Release Command/Response Transfer
    // ------------------------------------------------------------------

    #[test]
    fn test_release_command_transfer_roundtrip() {
        // Encode with the same layout as the strict peer: ext + opt bits + Cause
        let transfer = PDUSessionResourceReleaseCommandTransfer {
            cause: Cause::Nas(CauseNas(CauseNas::NORMAL_RELEASE)),
            ie_extensions: None,
        };
        let bytes = encode_aper(&transfer).unwrap();
        let cause = decode_release_command_transfer(&bytes).unwrap();
        assert_eq!(cause, NgSetupFailureCause::Nas(NasCause::NormalRelease));
    }

    #[test]
    fn test_release_response_transfer_is_single_preamble_octet() {
        // Mirrors the strict peer's expectation: exactly one 0x00 octet
        let bytes = encode_release_response_transfer().unwrap();
        assert_eq!(bytes, vec![0x00]);
        decode_release_response_transfer(&bytes).unwrap();
    }

    #[test]
    fn test_release_command_transfer_rejects_empty() {
        assert!(decode_release_command_transfer(&[]).is_err());
    }
}
