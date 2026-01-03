//! NG Setup Procedure
//!
//! Implements the NG Setup procedure as defined in 3GPP TS 38.413 Section 8.7.1.
//! This procedure is used to exchange application-level data needed for the NG-RAN node
//! and the AMF to correctly interoperate on the NG-C interface.

use crate::codec::generated::*;
use crate::codec::{decode_ngap_pdu, encode_ngap_pdu, NgapCodecError};
use bitvec::prelude::*;
use thiserror::Error;

/// Errors that can occur during NG Setup procedures
#[derive(Debug, Error)]
pub enum NgSetupError {
    /// Codec error during encoding/decoding
    #[error("Codec error: {0}")]
    CodecError(#[from] NgapCodecError),

    /// Invalid message type received
    #[error("Invalid message type: expected {expected}, got {actual}")]
    InvalidMessageType { expected: String, actual: String },

    /// Missing mandatory IE
    #[error("Missing mandatory IE: {0}")]
    MissingMandatoryIe(String),

    /// Invalid IE value
    #[error("Invalid IE value: {0}")]
    InvalidIeValue(String),
}

/// Paging DRX values as defined in 3GPP TS 38.413
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PagingDrx {
    /// 32 radio frames
    V32,
    /// 64 radio frames
    V64,
    /// 128 radio frames
    V128,
    /// 256 radio frames
    V256,
}

impl From<PagingDrx> for PagingDRX {
    fn from(drx: PagingDrx) -> Self {
        match drx {
            PagingDrx::V32 => PagingDRX(PagingDRX::V32),
            PagingDrx::V64 => PagingDRX(PagingDRX::V64),
            PagingDrx::V128 => PagingDRX(PagingDRX::V128),
            PagingDrx::V256 => PagingDRX(PagingDRX::V256),
        }
    }
}

impl TryFrom<PagingDRX> for PagingDrx {
    type Error = NgSetupError;

    fn try_from(drx: PagingDRX) -> Result<Self, Self::Error> {
        match drx.0 {
            PagingDRX::V32 => Ok(PagingDrx::V32),
            PagingDRX::V64 => Ok(PagingDrx::V64),
            PagingDRX::V128 => Ok(PagingDrx::V128),
            PagingDRX::V256 => Ok(PagingDrx::V256),
            _ => Err(NgSetupError::InvalidIeValue(format!(
                "Unknown PagingDRX value: {}",
                drx.0
            ))),
        }
    }
}

/// S-NSSAI (Single Network Slice Selection Assistance Information)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SNssai {
    /// Slice/Service Type (SST) - 1 byte
    pub sst: u8,
    /// Slice Differentiator (SD) - 3 bytes, optional
    pub sd: Option<[u8; 3]>,
}

/// Broadcast PLMN item for NG Setup Request
#[derive(Debug, Clone)]
pub struct BroadcastPlmnItem {
    /// PLMN Identity (3 bytes)
    pub plmn_identity: [u8; 3],
    /// List of supported S-NSSAIs for this PLMN
    pub slice_support_list: Vec<SNssai>,
}

/// Supported TA (Tracking Area) item for NG Setup Request
#[derive(Debug, Clone)]
pub struct SupportedTaItem {
    /// Tracking Area Code (3 bytes)
    pub tac: [u8; 3],
    /// List of broadcast PLMNs for this TA
    pub broadcast_plmn_list: Vec<BroadcastPlmnItem>,
}

/// Parameters for building an NG Setup Request
#[derive(Debug, Clone)]
pub struct NgSetupRequestParams {
    /// Global RAN Node ID - gNB ID with PLMN
    pub gnb_id: GnbId,
    /// RAN Node Name (optional)
    pub ran_node_name: Option<String>,
    /// List of supported TAs
    pub supported_ta_list: Vec<SupportedTaItem>,
    /// Default Paging DRX
    pub default_paging_drx: PagingDrx,
}

/// gNB ID with PLMN identity
#[derive(Debug, Clone)]
pub struct GnbId {
    /// PLMN Identity (3 bytes)
    pub plmn_identity: [u8; 3],
    /// gNB ID value (22-32 bits)
    pub gnb_id_value: u32,
    /// gNB ID bit length (22-32)
    pub gnb_id_length: u8,
}

/// GUAMI (Globally Unique AMF Identifier)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Guami {
    /// PLMN Identity (3 bytes)
    pub plmn_identity: [u8; 3],
    /// AMF Region ID (1 byte)
    pub amf_region_id: u8,
    /// AMF Set ID (10 bits)
    pub amf_set_id: u16,
    /// AMF Pointer (6 bits)
    pub amf_pointer: u8,
}

/// Served GUAMI item from NG Setup Response
#[derive(Debug, Clone)]
pub struct ServedGuamiItem {
    /// GUAMI
    pub guami: Guami,
    /// Backup AMF Name (optional)
    pub backup_amf_name: Option<String>,
}

/// PLMN Support item from NG Setup Response
#[derive(Debug, Clone)]
pub struct PlmnSupportItem {
    /// PLMN Identity (3 bytes)
    pub plmn_identity: [u8; 3],
    /// List of supported S-NSSAIs
    pub slice_support_list: Vec<SNssai>,
}

/// Parsed NG Setup Response data
#[derive(Debug, Clone)]
pub struct NgSetupResponseData {
    /// AMF Name
    pub amf_name: String,
    /// List of served GUAMIs
    pub served_guami_list: Vec<ServedGuamiItem>,
    /// Relative AMF Capacity (0-255)
    pub relative_amf_capacity: u8,
    /// List of supported PLMNs
    pub plmn_support_list: Vec<PlmnSupportItem>,
}

/// Cause values for NG Setup Failure
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NgSetupFailureCause {
    /// Radio Network Layer cause
    RadioNetwork(RadioNetworkCause),
    /// Transport Layer cause
    Transport(TransportCause),
    /// NAS cause
    Nas(NasCause),
    /// Protocol cause
    Protocol(ProtocolCause),
    /// Miscellaneous cause
    Misc(MiscCause),
}

/// Radio Network Layer causes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadioNetworkCause {
    Unspecified,
    TxnrelocoverallExpiry,
    SuccessfulHandover,
    ReleaseDueToNgranGeneratedReason,
    ReleaseDueTo5gcGeneratedReason,
    HandoverCancelled,
    PartialHandover,
    HoFailureInTarget5gcNgranNodeOrTargetSystem,
    HoTargetNotAllowed,
    TngrelocOverallExpiry,
    TngrelocPrepExpiry,
    CellNotAvailable,
    UnknownTargetId,
    NoRadioResourcesAvailableInTargetCell,
    UnknownLocalUeNgapId,
    InconsistentRemoteUeNgapId,
    HandoverDesirableForRadioReason,
    TimeCriticalHandover,
    ResourceOptimisationHandover,
    ReduceLoadInServingCell,
    UserInactivity,
    RadioConnectionWithUeLost,
    RadioResourcesNotAvailable,
    InvalidQosCombination,
    FailureInRadioInterfaceProcedure,
    InteractionWithOtherProcedure,
    UnknownPduSessionId,
    UnkownQosFlowId,
    MultipleQosFlowIdInstances,
    UnknownMappedUeNgapId,
    Other(u8),
}

/// Transport Layer causes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportCause {
    TransportResourceUnavailable,
    Unspecified,
    Other(u8),
}

/// NAS causes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NasCause {
    NormalRelease,
    AuthenticationFailure,
    Deregister,
    Unspecified,
    Other(u8),
}

/// Protocol causes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProtocolCause {
    TransferSyntaxError,
    AbstractSyntaxErrorReject,
    AbstractSyntaxErrorIgnoreAndNotify,
    MessageNotCompatibleWithReceiverState,
    SemanticError,
    AbstractSyntaxErrorFalselyConstructedMessage,
    Unspecified,
    Other(u8),
}

/// Miscellaneous causes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiscCause {
    ControlProcessingOverload,
    NotEnoughUserPlaneProcessingResources,
    HardwareFailure,
    OmIntervention,
    UnknownPlmnOrSnpn,
    Unspecified,
    Other(u8),
}

/// Parsed NG Setup Failure data
#[derive(Debug, Clone)]
pub struct NgSetupFailureData {
    /// Cause of failure
    pub cause: NgSetupFailureCause,
    /// Time to wait before retrying (optional)
    pub time_to_wait: Option<TimeToWaitValue>,
}

/// Time to wait values
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeToWaitValue {
    V1s,
    V2s,
    V5s,
    V10s,
    V20s,
    V60s,
}

// ============================================================================
// NG Setup Request Builder
// ============================================================================

/// Build an NG Setup Request PDU
///
/// # Arguments
/// * `params` - Parameters for the NG Setup Request
///
/// # Returns
/// * `Ok(NGAP_PDU)` - The constructed PDU
/// * `Err(NgSetupError)` - If construction fails
pub fn build_ng_setup_request(params: &NgSetupRequestParams) -> Result<NGAP_PDU, NgSetupError> {
    let mut protocol_ies = Vec::new();

    // IE: GlobalRANNodeID (mandatory)
    let global_ran_node_id = build_global_ran_node_id(&params.gnb_id);
    protocol_ies.push(NGSetupRequestProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_GLOBAL_RAN_NODE_ID),
        criticality: Criticality(Criticality::REJECT),
        value: NGSetupRequestProtocolIEs_EntryValue::Id_GlobalRANNodeID(global_ran_node_id),
    });

    // IE: RANNodeName (optional)
    if let Some(ref name) = params.ran_node_name {
        protocol_ies.push(NGSetupRequestProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_RAN_NODE_NAME),
            criticality: Criticality(Criticality::IGNORE),
            value: NGSetupRequestProtocolIEs_EntryValue::Id_RANNodeName(RANNodeName(
                name.clone(),
            )),
        });
    }

    // IE: SupportedTAList (mandatory)
    let supported_ta_list = build_supported_ta_list(&params.supported_ta_list);
    protocol_ies.push(NGSetupRequestProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_SUPPORTED_TA_LIST),
        criticality: Criticality(Criticality::REJECT),
        value: NGSetupRequestProtocolIEs_EntryValue::Id_SupportedTAList(supported_ta_list),
    });

    // IE: DefaultPagingDRX (mandatory)
    protocol_ies.push(NGSetupRequestProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_DEFAULT_PAGING_DRX),
        criticality: Criticality(Criticality::IGNORE),
        value: NGSetupRequestProtocolIEs_EntryValue::Id_DefaultPagingDRX(
            params.default_paging_drx.into(),
        ),
    });

    let ng_setup_request = NGSetupRequest {
        protocol_i_es: NGSetupRequestProtocolIEs(protocol_ies),
    };

    let initiating_message = InitiatingMessage {
        procedure_code: ProcedureCode(ID_NG_SETUP),
        criticality: Criticality(Criticality::REJECT),
        value: InitiatingMessageValue::Id_NGSetup(ng_setup_request),
    };

    Ok(NGAP_PDU::InitiatingMessage(initiating_message))
}

fn build_global_ran_node_id(gnb_id: &GnbId) -> GlobalRANNodeID {
    // Build the gNB-ID bit string
    // The gNB ID is 22-32 bits, we need to create a BitVec
    let mut bv: BitVec<u8, Msb0> = BitVec::new();
    
    // Add the bits from the gnb_id_value (MSB first)
    for i in (0..gnb_id.gnb_id_length).rev() {
        bv.push((gnb_id.gnb_id_value >> i) & 1 == 1);
    }

    let gnb_id_bitstring = GNB_ID::GNB_ID(GNB_ID_gNB_ID(bv));

    let global_gnb_id = GlobalGNB_ID {
        plmn_identity: PLMNIdentity(gnb_id.plmn_identity.to_vec()),
        gnb_id: gnb_id_bitstring,
        ie_extensions: None,
    };

    GlobalRANNodeID::GlobalGNB_ID(global_gnb_id)
}

fn build_supported_ta_list(ta_list: &[SupportedTaItem]) -> SupportedTAList {
    let items: Vec<SupportedTAItem> = ta_list
        .iter()
        .map(|ta| {
            let broadcast_plmn_list: Vec<BroadcastPLMNItem> = ta
                .broadcast_plmn_list
                .iter()
                .map(|bp| {
                    let slice_support_list: Vec<SliceSupportItem> = bp
                        .slice_support_list
                        .iter()
                        .map(|snssai| {
                            let sd = snssai.sd.map(|sd_bytes| SD(sd_bytes.to_vec()));
                            SliceSupportItem {
                                s_nssai: S_NSSAI {
                                    sst: SST(vec![snssai.sst]),
                                    sd,
                                    ie_extensions: None,
                                },
                                ie_extensions: None,
                            }
                        })
                        .collect();

                    BroadcastPLMNItem {
                        plmn_identity: PLMNIdentity(bp.plmn_identity.to_vec()),
                        tai_slice_support_list: SliceSupportList(slice_support_list),
                        ie_extensions: None,
                    }
                })
                .collect();

            SupportedTAItem {
                tac: TAC(ta.tac.to_vec()),
                broadcast_plmn_list: BroadcastPLMNList(broadcast_plmn_list),
                ie_extensions: None,
            }
        })
        .collect();

    SupportedTAList(items)
}

// ============================================================================
// NG Setup Response Parser
// ============================================================================

/// Parse an NG Setup Response from an NGAP PDU
///
/// # Arguments
/// * `pdu` - The NGAP PDU to parse
///
/// # Returns
/// * `Ok(NgSetupResponseData)` - The parsed response data
/// * `Err(NgSetupError)` - If parsing fails
pub fn parse_ng_setup_response(pdu: &NGAP_PDU) -> Result<NgSetupResponseData, NgSetupError> {
    let successful_outcome = match pdu {
        NGAP_PDU::SuccessfulOutcome(outcome) => outcome,
        _ => {
            return Err(NgSetupError::InvalidMessageType {
                expected: "SuccessfulOutcome".to_string(),
                actual: format!("{:?}", pdu),
            })
        }
    };

    let ng_setup_response = match &successful_outcome.value {
        SuccessfulOutcomeValue::Id_NGSetup(response) => response,
        _ => {
            return Err(NgSetupError::InvalidMessageType {
                expected: "NGSetupResponse".to_string(),
                actual: format!("{:?}", successful_outcome.value),
            })
        }
    };

    let mut amf_name: Option<String> = None;
    let mut served_guami_list: Option<Vec<ServedGuamiItem>> = None;
    let mut relative_amf_capacity: Option<u8> = None;
    let mut plmn_support_list: Option<Vec<PlmnSupportItem>> = None;

    for ie in &ng_setup_response.protocol_i_es.0 {
        match &ie.value {
            NGSetupResponseProtocolIEs_EntryValue::Id_AMFName(name) => {
                amf_name = Some(name.0.clone());
            }
            NGSetupResponseProtocolIEs_EntryValue::Id_ServedGUAMIList(list) => {
                served_guami_list = Some(parse_served_guami_list(list));
            }
            NGSetupResponseProtocolIEs_EntryValue::Id_RelativeAMFCapacity(capacity) => {
                relative_amf_capacity = Some(capacity.0);
            }
            NGSetupResponseProtocolIEs_EntryValue::Id_PLMNSupportList(list) => {
                plmn_support_list = Some(parse_plmn_support_list(list));
            }
            _ => {
                // Ignore other IEs
            }
        }
    }

    Ok(NgSetupResponseData {
        amf_name: amf_name.ok_or_else(|| NgSetupError::MissingMandatoryIe("AMFName".to_string()))?,
        served_guami_list: served_guami_list
            .ok_or_else(|| NgSetupError::MissingMandatoryIe("ServedGUAMIList".to_string()))?,
        relative_amf_capacity: relative_amf_capacity
            .ok_or_else(|| NgSetupError::MissingMandatoryIe("RelativeAMFCapacity".to_string()))?,
        plmn_support_list: plmn_support_list
            .ok_or_else(|| NgSetupError::MissingMandatoryIe("PLMNSupportList".to_string()))?,
    })
}

fn parse_served_guami_list(list: &ServedGUAMIList) -> Vec<ServedGuamiItem> {
    list.0
        .iter()
        .map(|item| {
            let guami = parse_guami(&item.guami);
            let backup_amf_name = item
                .backup_amf_name
                .as_ref()
                .map(|name| name.0.clone());

            ServedGuamiItem { guami, backup_amf_name }
        })
        .collect()
}

fn parse_guami(guami: &GUAMI) -> Guami {
    let plmn_identity: [u8; 3] = guami
        .plmn_identity
        .0
        .as_slice()
        .try_into()
        .unwrap_or([0, 0, 0]);

    // AMFRegionID is a BitVec of 8 bits
    let amf_region_id = if !guami.amf_region_id.0.is_empty() {
        guami.amf_region_id.0.as_raw_slice().first().copied().unwrap_or(0)
    } else {
        0
    };

    // AMF Set ID is 10 bits
    let amf_set_id = if guami.amf_set_id.0.len() >= 10 {
        let raw = guami.amf_set_id.0.as_raw_slice();
        if raw.len() >= 2 {
            ((raw[0] as u16) << 2) | ((raw[1] as u16) >> 6)
        } else if !raw.is_empty() {
            (raw[0] as u16) << 2
        } else {
            0
        }
    } else {
        0
    };

    // AMF Pointer is 6 bits
    let amf_pointer = if !guami.amf_pointer.0.is_empty() {
        let raw = guami.amf_pointer.0.as_raw_slice();
        if !raw.is_empty() {
            raw[0] >> 2
        } else {
            0
        }
    } else {
        0
    };

    Guami {
        plmn_identity,
        amf_region_id,
        amf_set_id,
        amf_pointer,
    }
}

fn parse_plmn_support_list(list: &PLMNSupportList) -> Vec<PlmnSupportItem> {
    list.0
        .iter()
        .map(|item| {
            let plmn_identity: [u8; 3] = item
                .plmn_identity
                .0
                .as_slice()
                .try_into()
                .unwrap_or([0, 0, 0]);

            let slice_support_list: Vec<SNssai> = item
                .slice_support_list
                .0
                .iter()
                .map(|slice_item| {
                    let sst = slice_item.s_nssai.sst.0.first().copied().unwrap_or(0);
                    let sd = slice_item.s_nssai.sd.as_ref().and_then(|sd| {
                        sd.0.as_slice().try_into().ok()
                    });
                    SNssai { sst, sd }
                })
                .collect();

            PlmnSupportItem { plmn_identity, slice_support_list }
        })
        .collect()
}

// ============================================================================
// NG Setup Failure Parser
// ============================================================================

/// Parse an NG Setup Failure from an NGAP PDU
///
/// # Arguments
/// * `pdu` - The NGAP PDU to parse
///
/// # Returns
/// * `Ok(NgSetupFailureData)` - The parsed failure data
/// * `Err(NgSetupError)` - If parsing fails
pub fn parse_ng_setup_failure(pdu: &NGAP_PDU) -> Result<NgSetupFailureData, NgSetupError> {
    let unsuccessful_outcome = match pdu {
        NGAP_PDU::UnsuccessfulOutcome(outcome) => outcome,
        _ => {
            return Err(NgSetupError::InvalidMessageType {
                expected: "UnsuccessfulOutcome".to_string(),
                actual: format!("{:?}", pdu),
            })
        }
    };

    let ng_setup_failure = match &unsuccessful_outcome.value {
        UnsuccessfulOutcomeValue::Id_NGSetup(failure) => failure,
        _ => {
            return Err(NgSetupError::InvalidMessageType {
                expected: "NGSetupFailure".to_string(),
                actual: format!("{:?}", unsuccessful_outcome.value),
            })
        }
    };

    let mut cause: Option<NgSetupFailureCause> = None;
    let mut time_to_wait: Option<TimeToWaitValue> = None;

    for ie in &ng_setup_failure.protocol_i_es.0 {
        match &ie.value {
            NGSetupFailureProtocolIEs_EntryValue::Id_Cause(c) => {
                cause = Some(parse_cause(c));
            }
            NGSetupFailureProtocolIEs_EntryValue::Id_TimeToWait(ttw) => {
                time_to_wait = Some(parse_time_to_wait(ttw));
            }
            _ => {
                // Ignore other IEs
            }
        }
    }

    Ok(NgSetupFailureData {
        cause: cause.ok_or_else(|| NgSetupError::MissingMandatoryIe("Cause".to_string()))?,
        time_to_wait,
    })
}

fn parse_cause(cause: &Cause) -> NgSetupFailureCause {
    match cause {
        Cause::RadioNetwork(rn) => {
            NgSetupFailureCause::RadioNetwork(parse_radio_network_cause(rn))
        }
        Cause::Transport(t) => NgSetupFailureCause::Transport(parse_transport_cause(t)),
        Cause::Nas(n) => NgSetupFailureCause::Nas(parse_nas_cause(n)),
        Cause::Protocol(p) => NgSetupFailureCause::Protocol(parse_protocol_cause(p)),
        Cause::Misc(m) => NgSetupFailureCause::Misc(parse_misc_cause(m)),
        Cause::Choice_Extensions(_) => {
            NgSetupFailureCause::Misc(MiscCause::Unspecified)
        }
    }
}

fn parse_radio_network_cause(cause: &CauseRadioNetwork) -> RadioNetworkCause {
    match cause.0 {
        CauseRadioNetwork::UNSPECIFIED => RadioNetworkCause::Unspecified,
        CauseRadioNetwork::TXNRELOCOVERALL_EXPIRY => RadioNetworkCause::TxnrelocoverallExpiry,
        CauseRadioNetwork::SUCCESSFUL_HANDOVER => RadioNetworkCause::SuccessfulHandover,
        CauseRadioNetwork::RELEASE_DUE_TO_NGRAN_GENERATED_REASON => {
            RadioNetworkCause::ReleaseDueToNgranGeneratedReason
        }
        CauseRadioNetwork::RELEASE_DUE_TO_5GC_GENERATED_REASON => {
            RadioNetworkCause::ReleaseDueTo5gcGeneratedReason
        }
        CauseRadioNetwork::HANDOVER_CANCELLED => RadioNetworkCause::HandoverCancelled,
        CauseRadioNetwork::PARTIAL_HANDOVER => RadioNetworkCause::PartialHandover,
        CauseRadioNetwork::HO_FAILURE_IN_TARGET_5GC_NGRAN_NODE_OR_TARGET_SYSTEM => {
            RadioNetworkCause::HoFailureInTarget5gcNgranNodeOrTargetSystem
        }
        CauseRadioNetwork::HO_TARGET_NOT_ALLOWED => RadioNetworkCause::HoTargetNotAllowed,
        CauseRadioNetwork::TNGRELOCOVERALL_EXPIRY => RadioNetworkCause::TngrelocOverallExpiry,
        CauseRadioNetwork::TNGRELOCPREP_EXPIRY => RadioNetworkCause::TngrelocPrepExpiry,
        CauseRadioNetwork::CELL_NOT_AVAILABLE => RadioNetworkCause::CellNotAvailable,
        CauseRadioNetwork::UNKNOWN_TARGET_ID => RadioNetworkCause::UnknownTargetId,
        CauseRadioNetwork::NO_RADIO_RESOURCES_AVAILABLE_IN_TARGET_CELL => {
            RadioNetworkCause::NoRadioResourcesAvailableInTargetCell
        }
        CauseRadioNetwork::UNKNOWN_LOCAL_UE_NGAP_ID => RadioNetworkCause::UnknownLocalUeNgapId,
        CauseRadioNetwork::INCONSISTENT_REMOTE_UE_NGAP_ID => {
            RadioNetworkCause::InconsistentRemoteUeNgapId
        }
        CauseRadioNetwork::HANDOVER_DESIRABLE_FOR_RADIO_REASON => {
            RadioNetworkCause::HandoverDesirableForRadioReason
        }
        CauseRadioNetwork::TIME_CRITICAL_HANDOVER => RadioNetworkCause::TimeCriticalHandover,
        CauseRadioNetwork::RESOURCE_OPTIMISATION_HANDOVER => {
            RadioNetworkCause::ResourceOptimisationHandover
        }
        CauseRadioNetwork::REDUCE_LOAD_IN_SERVING_CELL => {
            RadioNetworkCause::ReduceLoadInServingCell
        }
        CauseRadioNetwork::USER_INACTIVITY => RadioNetworkCause::UserInactivity,
        CauseRadioNetwork::RADIO_CONNECTION_WITH_UE_LOST => {
            RadioNetworkCause::RadioConnectionWithUeLost
        }
        CauseRadioNetwork::RADIO_RESOURCES_NOT_AVAILABLE => {
            RadioNetworkCause::RadioResourcesNotAvailable
        }
        CauseRadioNetwork::INVALID_QOS_COMBINATION => RadioNetworkCause::InvalidQosCombination,
        CauseRadioNetwork::FAILURE_IN_RADIO_INTERFACE_PROCEDURE => {
            RadioNetworkCause::FailureInRadioInterfaceProcedure
        }
        CauseRadioNetwork::INTERACTION_WITH_OTHER_PROCEDURE => {
            RadioNetworkCause::InteractionWithOtherProcedure
        }
        CauseRadioNetwork::UNKNOWN_PDU_SESSION_ID => RadioNetworkCause::UnknownPduSessionId,
        CauseRadioNetwork::UNKOWN_QOS_FLOW_ID => RadioNetworkCause::UnkownQosFlowId,
        CauseRadioNetwork::MULTIPLE_QOS_FLOW_ID_INSTANCES => {
            RadioNetworkCause::MultipleQosFlowIdInstances
        }
        other => RadioNetworkCause::Other(other),
    }
}

fn parse_transport_cause(cause: &CauseTransport) -> TransportCause {
    match cause.0 {
        CauseTransport::TRANSPORT_RESOURCE_UNAVAILABLE => {
            TransportCause::TransportResourceUnavailable
        }
        CauseTransport::UNSPECIFIED => TransportCause::Unspecified,
        other => TransportCause::Other(other),
    }
}

fn parse_nas_cause(cause: &CauseNas) -> NasCause {
    match cause.0 {
        CauseNas::NORMAL_RELEASE => NasCause::NormalRelease,
        CauseNas::AUTHENTICATION_FAILURE => NasCause::AuthenticationFailure,
        CauseNas::DEREGISTER => NasCause::Deregister,
        CauseNas::UNSPECIFIED => NasCause::Unspecified,
        other => NasCause::Other(other),
    }
}

fn parse_protocol_cause(cause: &CauseProtocol) -> ProtocolCause {
    match cause.0 {
        CauseProtocol::TRANSFER_SYNTAX_ERROR => ProtocolCause::TransferSyntaxError,
        CauseProtocol::ABSTRACT_SYNTAX_ERROR_REJECT => ProtocolCause::AbstractSyntaxErrorReject,
        CauseProtocol::ABSTRACT_SYNTAX_ERROR_IGNORE_AND_NOTIFY => {
            ProtocolCause::AbstractSyntaxErrorIgnoreAndNotify
        }
        CauseProtocol::MESSAGE_NOT_COMPATIBLE_WITH_RECEIVER_STATE => {
            ProtocolCause::MessageNotCompatibleWithReceiverState
        }
        CauseProtocol::SEMANTIC_ERROR => ProtocolCause::SemanticError,
        CauseProtocol::ABSTRACT_SYNTAX_ERROR_FALSELY_CONSTRUCTED_MESSAGE => {
            ProtocolCause::AbstractSyntaxErrorFalselyConstructedMessage
        }
        CauseProtocol::UNSPECIFIED => ProtocolCause::Unspecified,
        other => ProtocolCause::Other(other),
    }
}

fn parse_misc_cause(cause: &CauseMisc) -> MiscCause {
    match cause.0 {
        CauseMisc::CONTROL_PROCESSING_OVERLOAD => MiscCause::ControlProcessingOverload,
        CauseMisc::NOT_ENOUGH_USER_PLANE_PROCESSING_RESOURCES => {
            MiscCause::NotEnoughUserPlaneProcessingResources
        }
        CauseMisc::HARDWARE_FAILURE => MiscCause::HardwareFailure,
        CauseMisc::OM_INTERVENTION => MiscCause::OmIntervention,
        CauseMisc::UNKNOWN_PLMN_OR_SNPN => MiscCause::UnknownPlmnOrSnpn,
        CauseMisc::UNSPECIFIED => MiscCause::Unspecified,
        other => MiscCause::Other(other),
    }
}

fn parse_time_to_wait(ttw: &TimeToWait) -> TimeToWaitValue {
    match ttw.0 {
        TimeToWait::V1S => TimeToWaitValue::V1s,
        TimeToWait::V2S => TimeToWaitValue::V2s,
        TimeToWait::V5S => TimeToWaitValue::V5s,
        TimeToWait::V10S => TimeToWaitValue::V10s,
        TimeToWait::V20S => TimeToWaitValue::V20s,
        TimeToWait::V60S => TimeToWaitValue::V60s,
        _ => TimeToWaitValue::V1s, // Default fallback
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Build and encode an NG Setup Request to bytes
///
/// # Arguments
/// * `params` - Parameters for the NG Setup Request
///
/// # Returns
/// * `Ok(Vec<u8>)` - The encoded bytes
/// * `Err(NgSetupError)` - If building or encoding fails
pub fn encode_ng_setup_request(params: &NgSetupRequestParams) -> Result<Vec<u8>, NgSetupError> {
    let pdu = build_ng_setup_request(params)?;
    Ok(encode_ngap_pdu(&pdu)?)
}

/// Decode and parse an NG Setup Response from bytes
///
/// # Arguments
/// * `bytes` - The encoded bytes
///
/// # Returns
/// * `Ok(NgSetupResponseData)` - The parsed response data
/// * `Err(NgSetupError)` - If decoding or parsing fails
pub fn decode_ng_setup_response(bytes: &[u8]) -> Result<NgSetupResponseData, NgSetupError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_ng_setup_response(&pdu)
}

/// Decode and parse an NG Setup Failure from bytes
///
/// # Arguments
/// * `bytes` - The encoded bytes
///
/// # Returns
/// * `Ok(NgSetupFailureData)` - The parsed failure data
/// * `Err(NgSetupError)` - If decoding or parsing fails
pub fn decode_ng_setup_failure(bytes: &[u8]) -> Result<NgSetupFailureData, NgSetupError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_ng_setup_failure(&pdu)
}

/// Check if an NGAP PDU is an NG Setup Response
pub fn is_ng_setup_response(pdu: &NGAP_PDU) -> bool {
    matches!(
        pdu,
        NGAP_PDU::SuccessfulOutcome(outcome)
            if matches!(outcome.value, SuccessfulOutcomeValue::Id_NGSetup(_))
    )
}

/// Check if an NGAP PDU is an NG Setup Failure
pub fn is_ng_setup_failure(pdu: &NGAP_PDU) -> bool {
    matches!(
        pdu,
        NGAP_PDU::UnsuccessfulOutcome(outcome)
            if matches!(outcome.value, UnsuccessfulOutcomeValue::Id_NGSetup(_))
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_params() -> NgSetupRequestParams {
        NgSetupRequestParams {
            gnb_id: GnbId {
                plmn_identity: [0x00, 0xF1, 0x10], // MCC=001, MNC=01
                gnb_id_value: 1,
                gnb_id_length: 22,
            },
            ran_node_name: Some("nextgsim-gnb".to_string()),
            supported_ta_list: vec![SupportedTaItem {
                tac: [0x00, 0x00, 0x01], // TAC = 1
                broadcast_plmn_list: vec![BroadcastPlmnItem {
                    plmn_identity: [0x00, 0xF1, 0x10],
                    slice_support_list: vec![
                        SNssai { sst: 1, sd: None },
                        SNssai { sst: 1, sd: Some([0x00, 0x00, 0x01]) },
                    ],
                }],
            }],
            default_paging_drx: PagingDrx::V128,
        }
    }

    #[test]
    fn test_build_ng_setup_request() {
        let params = create_test_params();
        let result = build_ng_setup_request(&params);
        assert!(result.is_ok());

        let pdu = result.unwrap();
        match pdu {
            NGAP_PDU::InitiatingMessage(msg) => {
                assert_eq!(msg.procedure_code.0, ID_NG_SETUP);
            }
            _ => panic!("Expected InitiatingMessage"),
        }
    }

    #[test]
    fn test_encode_ng_setup_request() {
        let params = create_test_params();
        let result = encode_ng_setup_request(&params);
        assert!(result.is_ok());

        let bytes = result.unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_ng_setup_request_roundtrip() {
        let params = create_test_params();

        // Build and encode
        let pdu = build_ng_setup_request(&params).expect("Failed to build request");
        let encoded = encode_ngap_pdu(&pdu).expect("Failed to encode");

        // Decode
        let decoded_pdu = decode_ngap_pdu(&encoded).expect("Failed to decode");

        // Verify structure
        match decoded_pdu {
            NGAP_PDU::InitiatingMessage(msg) => {
                assert_eq!(msg.procedure_code.0, ID_NG_SETUP);
                match msg.value {
                    InitiatingMessageValue::Id_NGSetup(req) => {
                        // Verify we have the expected IEs
                        assert!(req.protocol_i_es.0.len() >= 3);
                    }
                    _ => panic!("Expected NGSetupRequest"),
                }
            }
            _ => panic!("Expected InitiatingMessage"),
        }
    }

    #[test]
    fn test_paging_drx_conversion() {
        // Test From<PagingDrx> for PagingDRX
        let drx: PagingDRX = PagingDrx::V32.into();
        assert_eq!(drx.0, PagingDRX::V32);

        let drx: PagingDRX = PagingDrx::V64.into();
        assert_eq!(drx.0, PagingDRX::V64);

        let drx: PagingDRX = PagingDrx::V128.into();
        assert_eq!(drx.0, PagingDRX::V128);

        let drx: PagingDRX = PagingDrx::V256.into();
        assert_eq!(drx.0, PagingDRX::V256);

        // Test TryFrom<PagingDRX> for PagingDrx
        assert_eq!(PagingDrx::try_from(PagingDRX(PagingDRX::V32)).unwrap(), PagingDrx::V32);
        assert_eq!(PagingDrx::try_from(PagingDRX(PagingDRX::V64)).unwrap(), PagingDrx::V64);
        assert_eq!(PagingDrx::try_from(PagingDRX(PagingDRX::V128)).unwrap(), PagingDrx::V128);
        assert_eq!(PagingDrx::try_from(PagingDRX(PagingDRX::V256)).unwrap(), PagingDrx::V256);
    }

    #[test]
    fn test_snssai_with_sd() {
        let snssai = SNssai {
            sst: 1,
            sd: Some([0x00, 0x00, 0x01]),
        };
        assert_eq!(snssai.sst, 1);
        assert_eq!(snssai.sd, Some([0x00, 0x00, 0x01]));
    }

    #[test]
    fn test_snssai_without_sd() {
        let snssai = SNssai { sst: 1, sd: None };
        assert_eq!(snssai.sst, 1);
        assert!(snssai.sd.is_none());
    }

    #[test]
    fn test_gnb_id_encoding() {
        let gnb_id = GnbId {
            plmn_identity: [0x00, 0xF1, 0x10],
            gnb_id_value: 1,
            gnb_id_length: 22,
        };

        let global_ran_node_id = build_global_ran_node_id(&gnb_id);
        match global_ran_node_id {
            GlobalRANNodeID::GlobalGNB_ID(global_gnb_id) => {
                assert_eq!(global_gnb_id.plmn_identity.0, vec![0x00, 0xF1, 0x10]);
            }
            _ => panic!("Expected GlobalGNB_ID"),
        }
    }
}
