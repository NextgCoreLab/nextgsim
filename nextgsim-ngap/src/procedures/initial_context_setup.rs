//! Initial Context Setup Procedure
//!
//! Implements the Initial Context Setup procedure as defined in 3GPP TS 38.413 Section 8.3.1.
//! This procedure is used by the AMF to establish the necessary overall initial UE context
//! at the NG-RAN node, including PDU session resources if available.

use crate::codec::generated::*;
use crate::codec::{decode_ngap_pdu, encode_ngap_pdu, NgapCodecError};
use crate::procedures::ng_setup::{
    MiscCause, NasCause, NgSetupFailureCause, ProtocolCause, RadioNetworkCause, TransportCause,
};
use bitvec::prelude::*;
use thiserror::Error;

/// Errors that can occur during Initial Context Setup procedures
#[derive(Debug, Error)]
pub enum InitialContextSetupError {
    /// Codec error during encoding/decoding
    #[error("Codec error: {0}")]
    CodecError(#[from] NgapCodecError),

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
    MissingMandatoryIe(String),

    /// Invalid IE value
    #[error("Invalid IE value: {0}")]
    InvalidIeValue(String),
}

/// GUAMI (Globally Unique AMF Identifier)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuamiValue {
    /// PLMN Identity (3 bytes)
    pub plmn_identity: [u8; 3],
    /// AMF Region ID (1 byte)
    pub amf_region_id: u8,
    /// AMF Set ID (10 bits)
    pub amf_set_id: u16,
    /// AMF Pointer (6 bits)
    pub amf_pointer: u8,
}

/// UE Aggregate Maximum Bit Rate
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UeAggregateMaxBitRate {
    /// Maximum bit rate for downlink (bits per second)
    pub dl: u64,
    /// Maximum bit rate for uplink (bits per second)
    pub ul: u64,
}

/// UE Security Capabilities
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UeSecurityCapabilitiesValue {
    /// NR encryption algorithms (16 bits)
    pub nr_encryption_algorithms: u16,
    /// NR integrity protection algorithms (16 bits)
    pub nr_integrity_algorithms: u16,
    /// E-UTRA encryption algorithms (16 bits, optional)
    pub eutra_encryption_algorithms: Option<u16>,
    /// E-UTRA integrity protection algorithms (16 bits, optional)
    pub eutra_integrity_algorithms: Option<u16>,
}


/// Allowed S-NSSAI item
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllowedSnssaiValue {
    /// Slice/Service Type (SST) - 1 byte
    pub sst: u8,
    /// Slice Differentiator (SD) - 3 bytes, optional
    pub sd: Option<[u8; 3]>,
}

/// Parsed Initial Context Setup Request data
#[derive(Debug, Clone)]
pub struct InitialContextSetupRequestData {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
    /// Old AMF name (optional)
    pub old_amf: Option<String>,
    /// UE Aggregate Maximum Bit Rate (conditional)
    pub ue_aggregate_max_bit_rate: Option<UeAggregateMaxBitRate>,
    /// GUAMI
    pub guami: GuamiValue,
    /// Allowed NSSAI
    pub allowed_nssai: Vec<AllowedSnssaiValue>,
    /// UE Security Capabilities
    pub ue_security_capabilities: UeSecurityCapabilitiesValue,
    /// Security Key (256 bits)
    pub security_key: [u8; 32],
    /// NAS PDU (optional)
    pub nas_pdu: Option<Vec<u8>>,
}

/// Parameters for building an Initial Context Setup Response
#[derive(Debug, Clone)]
pub struct InitialContextSetupResponseParams {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
}

/// Parsed Initial Context Setup Response data
#[derive(Debug, Clone)]
pub struct InitialContextSetupResponseData {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
}

/// Cause values for Initial Context Setup Failure
pub type InitialContextSetupFailureCause = NgSetupFailureCause;

/// Parameters for building an Initial Context Setup Failure
#[derive(Debug, Clone)]
pub struct InitialContextSetupFailureParams {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
    /// Cause of failure
    pub cause: InitialContextSetupFailureCause,
}

/// Parsed Initial Context Setup Failure data
#[derive(Debug, Clone)]
pub struct InitialContextSetupFailureData {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
    /// Cause of failure
    pub cause: InitialContextSetupFailureCause,
}

// ============================================================================
// Initial Context Setup Request Parser
// ============================================================================

/// Parse an Initial Context Setup Request from an NGAP PDU
///
/// # Arguments
/// * `pdu` - The NGAP PDU to parse
///
/// # Returns
/// * `Ok(InitialContextSetupRequestData)` - The parsed request data
/// * `Err(InitialContextSetupError)` - If parsing fails
pub fn parse_initial_context_setup_request(
    pdu: &NGAP_PDU,
) -> Result<InitialContextSetupRequestData, InitialContextSetupError> {
    let initiating_message = match pdu {
        NGAP_PDU::InitiatingMessage(msg) => msg,
        _ => {
            return Err(InitialContextSetupError::InvalidMessageType {
                expected: "InitiatingMessage".to_string(),
                actual: format!("{:?}", pdu),
            })
        }
    };

    let request = match &initiating_message.value {
        InitiatingMessageValue::Id_InitialContextSetup(req) => req,
        _ => {
            return Err(InitialContextSetupError::InvalidMessageType {
                expected: "InitialContextSetupRequest".to_string(),
                actual: format!("{:?}", initiating_message.value),
            })
        }
    };

    let mut amf_ue_ngap_id: Option<u64> = None;
    let mut ran_ue_ngap_id: Option<u32> = None;
    let mut old_amf: Option<String> = None;
    let mut ue_aggregate_max_bit_rate: Option<UeAggregateMaxBitRate> = None;
    let mut guami: Option<GuamiValue> = None;
    let mut allowed_nssai: Option<Vec<AllowedSnssaiValue>> = None;
    let mut ue_security_capabilities: Option<UeSecurityCapabilitiesValue> = None;
    let mut security_key: Option<[u8; 32]> = None;
    let mut nas_pdu: Option<Vec<u8>> = None;

    for ie in &request.protocol_i_es.0 {
        match &ie.value {
            InitialContextSetupRequestProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(id) => {
                amf_ue_ngap_id = Some(id.0);
            }
            InitialContextSetupRequestProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(id) => {
                ran_ue_ngap_id = Some(id.0);
            }
            InitialContextSetupRequestProtocolIEs_EntryValue::Id_OldAMF(name) => {
                old_amf = Some(name.0.clone());
            }
            InitialContextSetupRequestProtocolIEs_EntryValue::Id_UEAggregateMaximumBitRate(rate) => {
                ue_aggregate_max_bit_rate = Some(parse_ue_aggregate_max_bit_rate(rate));
            }
            InitialContextSetupRequestProtocolIEs_EntryValue::Id_GUAMI(g) => {
                guami = Some(parse_guami(g));
            }
            InitialContextSetupRequestProtocolIEs_EntryValue::Id_AllowedNSSAI(nssai) => {
                allowed_nssai = Some(parse_allowed_nssai(nssai));
            }
            InitialContextSetupRequestProtocolIEs_EntryValue::Id_UESecurityCapabilities(caps) => {
                ue_security_capabilities = Some(parse_ue_security_capabilities(caps));
            }
            InitialContextSetupRequestProtocolIEs_EntryValue::Id_SecurityKey(key) => {
                security_key = Some(parse_security_key(key));
            }
            InitialContextSetupRequestProtocolIEs_EntryValue::Id_NAS_PDU(pdu) => {
                nas_pdu = Some(pdu.0.clone());
            }
            _ => {
                // Ignore other IEs
            }
        }
    }

    Ok(InitialContextSetupRequestData {
        amf_ue_ngap_id: amf_ue_ngap_id.ok_or_else(|| {
            InitialContextSetupError::MissingMandatoryIe("AMF-UE-NGAP-ID".to_string())
        })?,
        ran_ue_ngap_id: ran_ue_ngap_id.ok_or_else(|| {
            InitialContextSetupError::MissingMandatoryIe("RAN-UE-NGAP-ID".to_string())
        })?,
        old_amf,
        ue_aggregate_max_bit_rate,
        guami: guami
            .ok_or_else(|| InitialContextSetupError::MissingMandatoryIe("GUAMI".to_string()))?,
        allowed_nssai: allowed_nssai.ok_or_else(|| {
            InitialContextSetupError::MissingMandatoryIe("AllowedNSSAI".to_string())
        })?,
        ue_security_capabilities: ue_security_capabilities.ok_or_else(|| {
            InitialContextSetupError::MissingMandatoryIe("UESecurityCapabilities".to_string())
        })?,
        security_key: security_key.ok_or_else(|| {
            InitialContextSetupError::MissingMandatoryIe("SecurityKey".to_string())
        })?,
        nas_pdu,
    })
}


fn parse_ue_aggregate_max_bit_rate(rate: &UEAggregateMaximumBitRate) -> UeAggregateMaxBitRate {
    UeAggregateMaxBitRate {
        dl: rate.ue_aggregate_maximum_bit_rate_dl.0,
        ul: rate.ue_aggregate_maximum_bit_rate_ul.0,
    }
}

fn parse_guami(guami: &GUAMI) -> GuamiValue {
    let plmn_identity: [u8; 3] = guami
        .plmn_identity
        .0
        .as_slice()
        .try_into()
        .unwrap_or([0, 0, 0]);

    // AMFRegionID is a BitVec of 8 bits
    let amf_region_id = if !guami.amf_region_id.0.is_empty() {
        guami
            .amf_region_id
            .0
            .as_raw_slice()
            .first()
            .copied()
            .unwrap_or(0)
    } else {
        0
    };

    // AMF Set ID is 10 bits
    let amf_set_id = if guami.amf_set_id.0.len() >= 10 {
        let mut value: u16 = 0;
        for (i, bit) in guami.amf_set_id.0.iter().take(10).enumerate() {
            if *bit {
                value |= 1 << (9 - i);
            }
        }
        value
    } else {
        0
    };

    // AMF Pointer is 6 bits
    let amf_pointer = if guami.amf_pointer.0.len() >= 6 {
        let mut value: u8 = 0;
        for (i, bit) in guami.amf_pointer.0.iter().take(6).enumerate() {
            if *bit {
                value |= 1 << (5 - i);
            }
        }
        value
    } else {
        0
    };

    GuamiValue {
        plmn_identity,
        amf_region_id,
        amf_set_id,
        amf_pointer,
    }
}

fn parse_allowed_nssai(nssai: &AllowedNSSAI) -> Vec<AllowedSnssaiValue> {
    nssai
        .0
        .iter()
        .map(|item| {
            let sst = item.s_nssai.sst.0.first().copied().unwrap_or(0);
            let sd = item
                .s_nssai
                .sd
                .as_ref()
                .and_then(|sd| sd.0.as_slice().try_into().ok());
            AllowedSnssaiValue { sst, sd }
        })
        .collect()
}

fn parse_ue_security_capabilities(caps: &UESecurityCapabilities) -> UeSecurityCapabilitiesValue {
    // NR encryption algorithms is 16 bits
    let nr_encryption_algorithms = parse_16bit_bitvec(&caps.n_rencryption_algorithms.0);

    // NR integrity algorithms is 16 bits
    let nr_integrity_algorithms = parse_16bit_bitvec(&caps.n_rintegrity_protection_algorithms.0);

    // E-UTRA encryption algorithms is 16 bits
    let eutra_encryption_algorithms = Some(parse_16bit_bitvec(&caps.eutr_aencryption_algorithms.0));

    // E-UTRA integrity algorithms is 16 bits
    let eutra_integrity_algorithms =
        Some(parse_16bit_bitvec(&caps.eutr_aintegrity_protection_algorithms.0));

    UeSecurityCapabilitiesValue {
        nr_encryption_algorithms,
        nr_integrity_algorithms,
        eutra_encryption_algorithms,
        eutra_integrity_algorithms,
    }
}

fn parse_16bit_bitvec(bv: &BitVec<u8, Msb0>) -> u16 {
    if bv.len() >= 16 {
        let mut value: u16 = 0;
        for (i, bit) in bv.iter().take(16).enumerate() {
            if *bit {
                value |= 1 << (15 - i);
            }
        }
        value
    } else {
        0
    }
}

fn parse_security_key(key: &SecurityKey) -> [u8; 32] {
    // SecurityKey is 256 bits
    let mut result = [0u8; 32];
    let raw = key.0.as_raw_slice();
    let len = raw.len().min(32);
    result[..len].copy_from_slice(&raw[..len]);
    result
}

// ============================================================================
// Initial Context Setup Response Builder
// ============================================================================

/// Build an Initial Context Setup Response PDU
///
/// # Arguments
/// * `params` - Parameters for the Initial Context Setup Response
///
/// # Returns
/// * `Ok(NGAP_PDU)` - The constructed PDU
/// * `Err(InitialContextSetupError)` - If construction fails
pub fn build_initial_context_setup_response(
    params: &InitialContextSetupResponseParams,
) -> Result<NGAP_PDU, InitialContextSetupError> {
    let mut protocol_ies = Vec::new();

    // IE: AMF-UE-NGAP-ID (mandatory)
    protocol_ies.push(InitialContextSetupResponseProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_AMF_UE_NGAP_ID),
        criticality: Criticality(Criticality::IGNORE),
        value: InitialContextSetupResponseProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(
            AMF_UE_NGAP_ID(params.amf_ue_ngap_id),
        ),
    });

    // IE: RAN-UE-NGAP-ID (mandatory)
    protocol_ies.push(InitialContextSetupResponseProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_RAN_UE_NGAP_ID),
        criticality: Criticality(Criticality::IGNORE),
        value: InitialContextSetupResponseProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(
            RAN_UE_NGAP_ID(params.ran_ue_ngap_id),
        ),
    });

    let response = InitialContextSetupResponse {
        protocol_i_es: InitialContextSetupResponseProtocolIEs(protocol_ies),
    };

    let successful_outcome = SuccessfulOutcome {
        procedure_code: ProcedureCode(ID_INITIAL_CONTEXT_SETUP),
        criticality: Criticality(Criticality::REJECT),
        value: SuccessfulOutcomeValue::Id_InitialContextSetup(response),
    };

    Ok(NGAP_PDU::SuccessfulOutcome(successful_outcome))
}


/// Parse an Initial Context Setup Response from an NGAP PDU
///
/// # Arguments
/// * `pdu` - The NGAP PDU to parse
///
/// # Returns
/// * `Ok(InitialContextSetupResponseData)` - The parsed response data
/// * `Err(InitialContextSetupError)` - If parsing fails
pub fn parse_initial_context_setup_response(
    pdu: &NGAP_PDU,
) -> Result<InitialContextSetupResponseData, InitialContextSetupError> {
    let successful_outcome = match pdu {
        NGAP_PDU::SuccessfulOutcome(outcome) => outcome,
        _ => {
            return Err(InitialContextSetupError::InvalidMessageType {
                expected: "SuccessfulOutcome".to_string(),
                actual: format!("{:?}", pdu),
            })
        }
    };

    let response = match &successful_outcome.value {
        SuccessfulOutcomeValue::Id_InitialContextSetup(resp) => resp,
        _ => {
            return Err(InitialContextSetupError::InvalidMessageType {
                expected: "InitialContextSetupResponse".to_string(),
                actual: format!("{:?}", successful_outcome.value),
            })
        }
    };

    let mut amf_ue_ngap_id: Option<u64> = None;
    let mut ran_ue_ngap_id: Option<u32> = None;

    for ie in &response.protocol_i_es.0 {
        match &ie.value {
            InitialContextSetupResponseProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(id) => {
                amf_ue_ngap_id = Some(id.0);
            }
            InitialContextSetupResponseProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(id) => {
                ran_ue_ngap_id = Some(id.0);
            }
            _ => {
                // Ignore other IEs
            }
        }
    }

    Ok(InitialContextSetupResponseData {
        amf_ue_ngap_id: amf_ue_ngap_id.ok_or_else(|| {
            InitialContextSetupError::MissingMandatoryIe("AMF-UE-NGAP-ID".to_string())
        })?,
        ran_ue_ngap_id: ran_ue_ngap_id.ok_or_else(|| {
            InitialContextSetupError::MissingMandatoryIe("RAN-UE-NGAP-ID".to_string())
        })?,
    })
}

// ============================================================================
// Initial Context Setup Failure Builder
// ============================================================================

/// Build an Initial Context Setup Failure PDU
///
/// # Arguments
/// * `params` - Parameters for the Initial Context Setup Failure
///
/// # Returns
/// * `Ok(NGAP_PDU)` - The constructed PDU
/// * `Err(InitialContextSetupError)` - If construction fails
pub fn build_initial_context_setup_failure(
    params: &InitialContextSetupFailureParams,
) -> Result<NGAP_PDU, InitialContextSetupError> {
    let mut protocol_ies = Vec::new();

    // IE: AMF-UE-NGAP-ID (mandatory)
    protocol_ies.push(InitialContextSetupFailureProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_AMF_UE_NGAP_ID),
        criticality: Criticality(Criticality::IGNORE),
        value: InitialContextSetupFailureProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(AMF_UE_NGAP_ID(
            params.amf_ue_ngap_id,
        )),
    });

    // IE: RAN-UE-NGAP-ID (mandatory)
    protocol_ies.push(InitialContextSetupFailureProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_RAN_UE_NGAP_ID),
        criticality: Criticality(Criticality::IGNORE),
        value: InitialContextSetupFailureProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(RAN_UE_NGAP_ID(
            params.ran_ue_ngap_id,
        )),
    });

    // IE: Cause (mandatory)
    let cause = build_cause(&params.cause);
    protocol_ies.push(InitialContextSetupFailureProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_CAUSE),
        criticality: Criticality(Criticality::IGNORE),
        value: InitialContextSetupFailureProtocolIEs_EntryValue::Id_Cause(cause),
    });

    let failure = InitialContextSetupFailure {
        protocol_i_es: InitialContextSetupFailureProtocolIEs(protocol_ies),
    };

    let unsuccessful_outcome = UnsuccessfulOutcome {
        procedure_code: ProcedureCode(ID_INITIAL_CONTEXT_SETUP),
        criticality: Criticality(Criticality::REJECT),
        value: UnsuccessfulOutcomeValue::Id_InitialContextSetup(failure),
    };

    Ok(NGAP_PDU::UnsuccessfulOutcome(unsuccessful_outcome))
}

fn build_cause(cause: &InitialContextSetupFailureCause) -> Cause {
    match cause {
        NgSetupFailureCause::RadioNetwork(rn) => Cause::RadioNetwork(build_radio_network_cause(rn)),
        NgSetupFailureCause::Transport(t) => Cause::Transport(build_transport_cause(t)),
        NgSetupFailureCause::Nas(n) => Cause::Nas(build_nas_cause(n)),
        NgSetupFailureCause::Protocol(p) => Cause::Protocol(build_protocol_cause(p)),
        NgSetupFailureCause::Misc(m) => Cause::Misc(build_misc_cause(m)),
    }
}

fn build_radio_network_cause(cause: &RadioNetworkCause) -> CauseRadioNetwork {
    let value = match cause {
        RadioNetworkCause::Unspecified => CauseRadioNetwork::UNSPECIFIED,
        RadioNetworkCause::TxnrelocoverallExpiry => CauseRadioNetwork::TXNRELOCOVERALL_EXPIRY,
        RadioNetworkCause::SuccessfulHandover => CauseRadioNetwork::SUCCESSFUL_HANDOVER,
        RadioNetworkCause::ReleaseDueToNgranGeneratedReason => {
            CauseRadioNetwork::RELEASE_DUE_TO_NGRAN_GENERATED_REASON
        }
        RadioNetworkCause::ReleaseDueTo5gcGeneratedReason => {
            CauseRadioNetwork::RELEASE_DUE_TO_5GC_GENERATED_REASON
        }
        RadioNetworkCause::HandoverCancelled => CauseRadioNetwork::HANDOVER_CANCELLED,
        RadioNetworkCause::PartialHandover => CauseRadioNetwork::PARTIAL_HANDOVER,
        RadioNetworkCause::HoFailureInTarget5gcNgranNodeOrTargetSystem => {
            CauseRadioNetwork::HO_FAILURE_IN_TARGET_5GC_NGRAN_NODE_OR_TARGET_SYSTEM
        }
        RadioNetworkCause::HoTargetNotAllowed => CauseRadioNetwork::HO_TARGET_NOT_ALLOWED,
        RadioNetworkCause::TngrelocOverallExpiry => CauseRadioNetwork::TNGRELOCOVERALL_EXPIRY,
        RadioNetworkCause::TngrelocPrepExpiry => CauseRadioNetwork::TNGRELOCPREP_EXPIRY,
        RadioNetworkCause::CellNotAvailable => CauseRadioNetwork::CELL_NOT_AVAILABLE,
        RadioNetworkCause::UnknownTargetId => CauseRadioNetwork::UNKNOWN_TARGET_ID,
        RadioNetworkCause::NoRadioResourcesAvailableInTargetCell => {
            CauseRadioNetwork::NO_RADIO_RESOURCES_AVAILABLE_IN_TARGET_CELL
        }
        RadioNetworkCause::UnknownLocalUeNgapId => CauseRadioNetwork::UNKNOWN_LOCAL_UE_NGAP_ID,
        RadioNetworkCause::InconsistentRemoteUeNgapId => {
            CauseRadioNetwork::INCONSISTENT_REMOTE_UE_NGAP_ID
        }
        RadioNetworkCause::HandoverDesirableForRadioReason => {
            CauseRadioNetwork::HANDOVER_DESIRABLE_FOR_RADIO_REASON
        }
        RadioNetworkCause::TimeCriticalHandover => CauseRadioNetwork::TIME_CRITICAL_HANDOVER,
        RadioNetworkCause::ResourceOptimisationHandover => {
            CauseRadioNetwork::RESOURCE_OPTIMISATION_HANDOVER
        }
        RadioNetworkCause::ReduceLoadInServingCell => {
            CauseRadioNetwork::REDUCE_LOAD_IN_SERVING_CELL
        }
        RadioNetworkCause::UserInactivity => CauseRadioNetwork::USER_INACTIVITY,
        RadioNetworkCause::RadioConnectionWithUeLost => {
            CauseRadioNetwork::RADIO_CONNECTION_WITH_UE_LOST
        }
        RadioNetworkCause::RadioResourcesNotAvailable => {
            CauseRadioNetwork::RADIO_RESOURCES_NOT_AVAILABLE
        }
        RadioNetworkCause::InvalidQosCombination => CauseRadioNetwork::INVALID_QOS_COMBINATION,
        RadioNetworkCause::FailureInRadioInterfaceProcedure => {
            CauseRadioNetwork::FAILURE_IN_RADIO_INTERFACE_PROCEDURE
        }
        RadioNetworkCause::InteractionWithOtherProcedure => {
            CauseRadioNetwork::INTERACTION_WITH_OTHER_PROCEDURE
        }
        RadioNetworkCause::UnknownPduSessionId => CauseRadioNetwork::UNKNOWN_PDU_SESSION_ID,
        RadioNetworkCause::UnkownQosFlowId => CauseRadioNetwork::UNKOWN_QOS_FLOW_ID,
        RadioNetworkCause::MultipleQosFlowIdInstances => {
            CauseRadioNetwork::MULTIPLE_QOS_FLOW_ID_INSTANCES
        }
        RadioNetworkCause::UnknownMappedUeNgapId => CauseRadioNetwork::UNSPECIFIED, // Fallback
        RadioNetworkCause::Other(v) => *v,
    };
    CauseRadioNetwork(value)
}

fn build_transport_cause(cause: &TransportCause) -> CauseTransport {
    let value = match cause {
        TransportCause::TransportResourceUnavailable => {
            CauseTransport::TRANSPORT_RESOURCE_UNAVAILABLE
        }
        TransportCause::Unspecified => CauseTransport::UNSPECIFIED,
        TransportCause::Other(v) => *v,
    };
    CauseTransport(value)
}

fn build_nas_cause(cause: &NasCause) -> CauseNas {
    let value = match cause {
        NasCause::NormalRelease => CauseNas::NORMAL_RELEASE,
        NasCause::AuthenticationFailure => CauseNas::AUTHENTICATION_FAILURE,
        NasCause::Deregister => CauseNas::DEREGISTER,
        NasCause::Unspecified => CauseNas::UNSPECIFIED,
        NasCause::Other(v) => *v,
    };
    CauseNas(value)
}

fn build_protocol_cause(cause: &ProtocolCause) -> CauseProtocol {
    let value = match cause {
        ProtocolCause::TransferSyntaxError => CauseProtocol::TRANSFER_SYNTAX_ERROR,
        ProtocolCause::AbstractSyntaxErrorReject => CauseProtocol::ABSTRACT_SYNTAX_ERROR_REJECT,
        ProtocolCause::AbstractSyntaxErrorIgnoreAndNotify => {
            CauseProtocol::ABSTRACT_SYNTAX_ERROR_IGNORE_AND_NOTIFY
        }
        ProtocolCause::MessageNotCompatibleWithReceiverState => {
            CauseProtocol::MESSAGE_NOT_COMPATIBLE_WITH_RECEIVER_STATE
        }
        ProtocolCause::SemanticError => CauseProtocol::SEMANTIC_ERROR,
        ProtocolCause::AbstractSyntaxErrorFalselyConstructedMessage => {
            CauseProtocol::ABSTRACT_SYNTAX_ERROR_FALSELY_CONSTRUCTED_MESSAGE
        }
        ProtocolCause::Unspecified => CauseProtocol::UNSPECIFIED,
        ProtocolCause::Other(v) => *v,
    };
    CauseProtocol(value)
}

fn build_misc_cause(cause: &MiscCause) -> CauseMisc {
    let value = match cause {
        MiscCause::ControlProcessingOverload => CauseMisc::CONTROL_PROCESSING_OVERLOAD,
        MiscCause::NotEnoughUserPlaneProcessingResources => {
            CauseMisc::NOT_ENOUGH_USER_PLANE_PROCESSING_RESOURCES
        }
        MiscCause::HardwareFailure => CauseMisc::HARDWARE_FAILURE,
        MiscCause::OmIntervention => CauseMisc::OM_INTERVENTION,
        MiscCause::UnknownPlmnOrSnpn => CauseMisc::UNKNOWN_PLMN_OR_SNPN,
        MiscCause::Unspecified => CauseMisc::UNSPECIFIED,
        MiscCause::Other(v) => *v,
    };
    CauseMisc(value)
}


/// Parse an Initial Context Setup Failure from an NGAP PDU
///
/// # Arguments
/// * `pdu` - The NGAP PDU to parse
///
/// # Returns
/// * `Ok(InitialContextSetupFailureData)` - The parsed failure data
/// * `Err(InitialContextSetupError)` - If parsing fails
pub fn parse_initial_context_setup_failure(
    pdu: &NGAP_PDU,
) -> Result<InitialContextSetupFailureData, InitialContextSetupError> {
    let unsuccessful_outcome = match pdu {
        NGAP_PDU::UnsuccessfulOutcome(outcome) => outcome,
        _ => {
            return Err(InitialContextSetupError::InvalidMessageType {
                expected: "UnsuccessfulOutcome".to_string(),
                actual: format!("{:?}", pdu),
            })
        }
    };

    let failure = match &unsuccessful_outcome.value {
        UnsuccessfulOutcomeValue::Id_InitialContextSetup(fail) => fail,
        _ => {
            return Err(InitialContextSetupError::InvalidMessageType {
                expected: "InitialContextSetupFailure".to_string(),
                actual: format!("{:?}", unsuccessful_outcome.value),
            })
        }
    };

    let mut amf_ue_ngap_id: Option<u64> = None;
    let mut ran_ue_ngap_id: Option<u32> = None;
    let mut cause: Option<InitialContextSetupFailureCause> = None;

    for ie in &failure.protocol_i_es.0 {
        match &ie.value {
            InitialContextSetupFailureProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(id) => {
                amf_ue_ngap_id = Some(id.0);
            }
            InitialContextSetupFailureProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(id) => {
                ran_ue_ngap_id = Some(id.0);
            }
            InitialContextSetupFailureProtocolIEs_EntryValue::Id_Cause(c) => {
                cause = Some(parse_cause(c));
            }
            _ => {
                // Ignore other IEs
            }
        }
    }

    Ok(InitialContextSetupFailureData {
        amf_ue_ngap_id: amf_ue_ngap_id.ok_or_else(|| {
            InitialContextSetupError::MissingMandatoryIe("AMF-UE-NGAP-ID".to_string())
        })?,
        ran_ue_ngap_id: ran_ue_ngap_id.ok_or_else(|| {
            InitialContextSetupError::MissingMandatoryIe("RAN-UE-NGAP-ID".to_string())
        })?,
        cause: cause
            .ok_or_else(|| InitialContextSetupError::MissingMandatoryIe("Cause".to_string()))?,
    })
}

fn parse_cause(cause: &Cause) -> InitialContextSetupFailureCause {
    match cause {
        Cause::RadioNetwork(rn) => {
            NgSetupFailureCause::RadioNetwork(parse_radio_network_cause(rn))
        }
        Cause::Transport(t) => NgSetupFailureCause::Transport(parse_transport_cause(t)),
        Cause::Nas(n) => NgSetupFailureCause::Nas(parse_nas_cause(n)),
        Cause::Protocol(p) => NgSetupFailureCause::Protocol(parse_protocol_cause(p)),
        Cause::Misc(m) => NgSetupFailureCause::Misc(parse_misc_cause(m)),
        Cause::Choice_Extensions(_) => NgSetupFailureCause::Misc(MiscCause::Unspecified),
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

// ============================================================================
// Convenience Functions
// ============================================================================

/// Build and encode an Initial Context Setup Response to bytes
///
/// # Arguments
/// * `params` - Parameters for the Initial Context Setup Response
///
/// # Returns
/// * `Ok(Vec<u8>)` - The encoded bytes
/// * `Err(InitialContextSetupError)` - If building or encoding fails
pub fn encode_initial_context_setup_response(
    params: &InitialContextSetupResponseParams,
) -> Result<Vec<u8>, InitialContextSetupError> {
    let pdu = build_initial_context_setup_response(params)?;
    Ok(encode_ngap_pdu(&pdu)?)
}

/// Build and encode an Initial Context Setup Failure to bytes
///
/// # Arguments
/// * `params` - Parameters for the Initial Context Setup Failure
///
/// # Returns
/// * `Ok(Vec<u8>)` - The encoded bytes
/// * `Err(InitialContextSetupError)` - If building or encoding fails
pub fn encode_initial_context_setup_failure(
    params: &InitialContextSetupFailureParams,
) -> Result<Vec<u8>, InitialContextSetupError> {
    let pdu = build_initial_context_setup_failure(params)?;
    Ok(encode_ngap_pdu(&pdu)?)
}

/// Decode and parse an Initial Context Setup Request from bytes
///
/// # Arguments
/// * `bytes` - The encoded bytes
///
/// # Returns
/// * `Ok(InitialContextSetupRequestData)` - The parsed request data
/// * `Err(InitialContextSetupError)` - If decoding or parsing fails
pub fn decode_initial_context_setup_request(
    bytes: &[u8],
) -> Result<InitialContextSetupRequestData, InitialContextSetupError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_initial_context_setup_request(&pdu)
}

/// Decode and parse an Initial Context Setup Response from bytes
///
/// # Arguments
/// * `bytes` - The encoded bytes
///
/// # Returns
/// * `Ok(InitialContextSetupResponseData)` - The parsed response data
/// * `Err(InitialContextSetupError)` - If decoding or parsing fails
pub fn decode_initial_context_setup_response(
    bytes: &[u8],
) -> Result<InitialContextSetupResponseData, InitialContextSetupError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_initial_context_setup_response(&pdu)
}

/// Decode and parse an Initial Context Setup Failure from bytes
///
/// # Arguments
/// * `bytes` - The encoded bytes
///
/// # Returns
/// * `Ok(InitialContextSetupFailureData)` - The parsed failure data
/// * `Err(InitialContextSetupError)` - If decoding or parsing fails
pub fn decode_initial_context_setup_failure(
    bytes: &[u8],
) -> Result<InitialContextSetupFailureData, InitialContextSetupError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_initial_context_setup_failure(&pdu)
}

/// Check if an NGAP PDU is an Initial Context Setup Request
pub fn is_initial_context_setup_request(pdu: &NGAP_PDU) -> bool {
    matches!(
        pdu,
        NGAP_PDU::InitiatingMessage(msg)
            if matches!(msg.value, InitiatingMessageValue::Id_InitialContextSetup(_))
    )
}

/// Check if an NGAP PDU is an Initial Context Setup Response
pub fn is_initial_context_setup_response(pdu: &NGAP_PDU) -> bool {
    matches!(
        pdu,
        NGAP_PDU::SuccessfulOutcome(outcome)
            if matches!(outcome.value, SuccessfulOutcomeValue::Id_InitialContextSetup(_))
    )
}

/// Check if an NGAP PDU is an Initial Context Setup Failure
pub fn is_initial_context_setup_failure(pdu: &NGAP_PDU) -> bool {
    matches!(
        pdu,
        NGAP_PDU::UnsuccessfulOutcome(outcome)
            if matches!(outcome.value, UnsuccessfulOutcomeValue::Id_InitialContextSetup(_))
    )
}


#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_response_params() -> InitialContextSetupResponseParams {
        InitialContextSetupResponseParams {
            amf_ue_ngap_id: 12345,
            ran_ue_ngap_id: 1,
        }
    }

    fn create_test_failure_params() -> InitialContextSetupFailureParams {
        InitialContextSetupFailureParams {
            amf_ue_ngap_id: 12345,
            ran_ue_ngap_id: 1,
            cause: NgSetupFailureCause::RadioNetwork(RadioNetworkCause::Unspecified),
        }
    }

    #[test]
    fn test_build_initial_context_setup_response() {
        let params = create_test_response_params();
        let result = build_initial_context_setup_response(&params);
        assert!(result.is_ok());

        let pdu = result.unwrap();
        match pdu {
            NGAP_PDU::SuccessfulOutcome(outcome) => {
                assert_eq!(outcome.procedure_code.0, ID_INITIAL_CONTEXT_SETUP);
            }
            _ => panic!("Expected SuccessfulOutcome"),
        }
    }

    #[test]
    fn test_encode_initial_context_setup_response() {
        let params = create_test_response_params();
        let result = encode_initial_context_setup_response(&params);
        assert!(result.is_ok());

        let bytes = result.unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_initial_context_setup_response_roundtrip() {
        let params = create_test_response_params();

        // Build and encode
        let pdu = build_initial_context_setup_response(&params).expect("Failed to build response");
        let encoded = encode_ngap_pdu(&pdu).expect("Failed to encode");

        // Decode
        let decoded_pdu = decode_ngap_pdu(&encoded).expect("Failed to decode");

        // Verify structure
        match decoded_pdu {
            NGAP_PDU::SuccessfulOutcome(outcome) => {
                assert_eq!(outcome.procedure_code.0, ID_INITIAL_CONTEXT_SETUP);
                match outcome.value {
                    SuccessfulOutcomeValue::Id_InitialContextSetup(resp) => {
                        // Verify we have the expected IEs (2 mandatory)
                        assert!(resp.protocol_i_es.0.len() >= 2);
                    }
                    _ => panic!("Expected InitialContextSetupResponse"),
                }
            }
            _ => panic!("Expected SuccessfulOutcome"),
        }
    }

    #[test]
    fn test_initial_context_setup_response_parse_roundtrip() {
        let params = create_test_response_params();

        // Build, encode, decode, and parse
        let encoded =
            encode_initial_context_setup_response(&params).expect("Failed to encode");
        let parsed =
            decode_initial_context_setup_response(&encoded).expect("Failed to decode and parse");

        // Verify parsed data matches original params
        assert_eq!(parsed.amf_ue_ngap_id, params.amf_ue_ngap_id);
        assert_eq!(parsed.ran_ue_ngap_id, params.ran_ue_ngap_id);
    }

    #[test]
    fn test_build_initial_context_setup_failure() {
        let params = create_test_failure_params();
        let result = build_initial_context_setup_failure(&params);
        assert!(result.is_ok());

        let pdu = result.unwrap();
        match pdu {
            NGAP_PDU::UnsuccessfulOutcome(outcome) => {
                assert_eq!(outcome.procedure_code.0, ID_INITIAL_CONTEXT_SETUP);
            }
            _ => panic!("Expected UnsuccessfulOutcome"),
        }
    }

    #[test]
    fn test_encode_initial_context_setup_failure() {
        let params = create_test_failure_params();
        let result = encode_initial_context_setup_failure(&params);
        assert!(result.is_ok());

        let bytes = result.unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_initial_context_setup_failure_roundtrip() {
        let params = create_test_failure_params();

        // Build and encode
        let pdu = build_initial_context_setup_failure(&params).expect("Failed to build failure");
        let encoded = encode_ngap_pdu(&pdu).expect("Failed to encode");

        // Decode
        let decoded_pdu = decode_ngap_pdu(&encoded).expect("Failed to decode");

        // Verify structure
        match decoded_pdu {
            NGAP_PDU::UnsuccessfulOutcome(outcome) => {
                assert_eq!(outcome.procedure_code.0, ID_INITIAL_CONTEXT_SETUP);
                match outcome.value {
                    UnsuccessfulOutcomeValue::Id_InitialContextSetup(fail) => {
                        // Verify we have the expected IEs (3 mandatory)
                        assert!(fail.protocol_i_es.0.len() >= 3);
                    }
                    _ => panic!("Expected InitialContextSetupFailure"),
                }
            }
            _ => panic!("Expected UnsuccessfulOutcome"),
        }
    }

    #[test]
    fn test_initial_context_setup_failure_parse_roundtrip() {
        let params = create_test_failure_params();

        // Build, encode, decode, and parse
        let encoded =
            encode_initial_context_setup_failure(&params).expect("Failed to encode");
        let parsed =
            decode_initial_context_setup_failure(&encoded).expect("Failed to decode and parse");

        // Verify parsed data matches original params
        assert_eq!(parsed.amf_ue_ngap_id, params.amf_ue_ngap_id);
        assert_eq!(parsed.ran_ue_ngap_id, params.ran_ue_ngap_id);
        assert_eq!(parsed.cause, params.cause);
    }

    #[test]
    fn test_initial_context_setup_failure_with_different_causes() {
        let causes = [
            NgSetupFailureCause::RadioNetwork(RadioNetworkCause::Unspecified),
            NgSetupFailureCause::RadioNetwork(RadioNetworkCause::CellNotAvailable),
            NgSetupFailureCause::Transport(TransportCause::TransportResourceUnavailable),
            NgSetupFailureCause::Nas(NasCause::NormalRelease),
            NgSetupFailureCause::Protocol(ProtocolCause::SemanticError),
            NgSetupFailureCause::Misc(MiscCause::HardwareFailure),
        ];

        for cause in causes {
            let params = InitialContextSetupFailureParams {
                amf_ue_ngap_id: 12345,
                ran_ue_ngap_id: 1,
                cause: cause.clone(),
            };

            let encoded =
                encode_initial_context_setup_failure(&params).expect("Failed to encode");
            let parsed =
                decode_initial_context_setup_failure(&encoded).expect("Failed to decode and parse");

            assert_eq!(parsed.cause, cause);
        }
    }

    #[test]
    fn test_is_initial_context_setup_response() {
        let params = create_test_response_params();
        let pdu = build_initial_context_setup_response(&params).expect("Failed to build response");
        assert!(is_initial_context_setup_response(&pdu));
        assert!(!is_initial_context_setup_failure(&pdu));
        assert!(!is_initial_context_setup_request(&pdu));
    }

    #[test]
    fn test_is_initial_context_setup_failure() {
        let params = create_test_failure_params();
        let pdu = build_initial_context_setup_failure(&params).expect("Failed to build failure");
        assert!(is_initial_context_setup_failure(&pdu));
        assert!(!is_initial_context_setup_response(&pdu));
        assert!(!is_initial_context_setup_request(&pdu));
    }

    #[test]
    fn test_guami_value() {
        let guami = GuamiValue {
            plmn_identity: [0x00, 0xF1, 0x10],
            amf_region_id: 1,
            amf_set_id: 1,
            amf_pointer: 0,
        };
        assert_eq!(guami.plmn_identity, [0x00, 0xF1, 0x10]);
        assert_eq!(guami.amf_region_id, 1);
        assert_eq!(guami.amf_set_id, 1);
        assert_eq!(guami.amf_pointer, 0);
    }

    #[test]
    fn test_ue_aggregate_max_bit_rate() {
        let rate = UeAggregateMaxBitRate {
            dl: 1_000_000_000, // 1 Gbps
            ul: 500_000_000,   // 500 Mbps
        };
        assert_eq!(rate.dl, 1_000_000_000);
        assert_eq!(rate.ul, 500_000_000);
    }

    #[test]
    fn test_ue_security_capabilities_value() {
        let caps = UeSecurityCapabilitiesValue {
            nr_encryption_algorithms: 0xF000,    // NEA0, NEA1, NEA2, NEA3
            nr_integrity_algorithms: 0xF000,     // NIA0, NIA1, NIA2, NIA3
            eutra_encryption_algorithms: Some(0xF000),
            eutra_integrity_algorithms: Some(0xF000),
        };
        assert_eq!(caps.nr_encryption_algorithms, 0xF000);
        assert_eq!(caps.nr_integrity_algorithms, 0xF000);
        assert_eq!(caps.eutra_encryption_algorithms, Some(0xF000));
        assert_eq!(caps.eutra_integrity_algorithms, Some(0xF000));
    }

    #[test]
    fn test_allowed_snssai_value() {
        let snssai = AllowedSnssaiValue {
            sst: 1,
            sd: Some([0x00, 0x00, 0x01]),
        };
        assert_eq!(snssai.sst, 1);
        assert_eq!(snssai.sd, Some([0x00, 0x00, 0x01]));

        let snssai_no_sd = AllowedSnssaiValue { sst: 1, sd: None };
        assert_eq!(snssai_no_sd.sst, 1);
        assert!(snssai_no_sd.sd.is_none());
    }
}
