//! RAN Configuration Update Procedure
//!
//! Implements the RAN Configuration Update procedure as defined in 3GPP TS 38.413 Section 8.7.3.
//! This is a Class 1 (request/response) procedure initiated by the NG-RAN node to update
//! its configuration information at the AMF.
//!
//! The procedure consists of:
//! 1. RANConfigurationUpdate - NG-RAN node -> AMF: Request to update configuration
//! 2. RANConfigurationUpdateAcknowledge - AMF -> NG-RAN node: Successful response
//! 3. RANConfigurationUpdateFailure - AMF -> NG-RAN node: Failure response

use crate::codec::generated::*;
use crate::codec::{decode_ngap_pdu, encode_ngap_pdu, NgapCodecError};
use crate::procedures::ng_setup::{
    BroadcastPlmnItem, MiscCause, NasCause, NgSetupFailureCause, PagingDrx, ProtocolCause,
    RadioNetworkCause, SNssai, SupportedTaItem, TimeToWaitValue, TransportCause,
};
use thiserror::Error;

/// Errors that can occur during RAN Configuration Update procedures
#[derive(Debug, Error)]
pub enum RanConfigurationUpdateError {
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

/// Parameters for building a RAN Configuration Update message
#[derive(Debug, Clone)]
pub struct RanConfigurationUpdateParams {
    /// RAN Node Name (optional)
    pub ran_node_name: Option<String>,
    /// Supported TA List (optional)
    pub supported_ta_list: Option<Vec<SupportedTaItem>>,
    /// Default Paging DRX (optional)
    pub default_paging_drx: Option<PagingDrx>,
}

/// Parsed RAN Configuration Update data
#[derive(Debug, Clone)]
pub struct RanConfigurationUpdateData {
    /// RAN Node Name (optional)
    pub ran_node_name: Option<String>,
    /// Supported TA List (optional)
    pub supported_ta_list: Option<Vec<SupportedTaItem>>,
    /// Default Paging DRX (optional)
    pub default_paging_drx: Option<PagingDrx>,
}

/// Parsed RAN Configuration Update Acknowledge data
#[derive(Debug, Clone)]
pub struct RanConfigurationUpdateAcknowledgeData {
    /// Placeholder - acknowledge has minimal mandatory IEs
    _private: (),
}

/// Parsed RAN Configuration Update Failure data
#[derive(Debug, Clone)]
pub struct RanConfigurationUpdateFailureData {
    /// Cause of failure
    pub cause: NgSetupFailureCause,
    /// Time to wait before retrying (optional)
    pub time_to_wait: Option<TimeToWaitValue>,
}

// ============================================================================
// RAN Configuration Update Builder
// ============================================================================

/// Build a RAN Configuration Update PDU
pub fn build_ran_configuration_update(
    params: &RanConfigurationUpdateParams,
) -> Result<NGAP_PDU, RanConfigurationUpdateError> {
    let mut protocol_ies = Vec::new();

    // IE: RANNodeName (optional)
    if let Some(ref name) = params.ran_node_name {
        protocol_ies.push(RANConfigurationUpdateProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_RAN_NODE_NAME),
            criticality: Criticality(Criticality::IGNORE),
            value: RANConfigurationUpdateProtocolIEs_EntryValue::Id_RANNodeName(RANNodeName(
                name.clone(),
            )),
        });
    }

    // IE: SupportedTAList (optional)
    if let Some(ref ta_list) = params.supported_ta_list {
        let supported_ta_list = build_supported_ta_list(ta_list);
        protocol_ies.push(RANConfigurationUpdateProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_SUPPORTED_TA_LIST),
            criticality: Criticality(Criticality::REJECT),
            value: RANConfigurationUpdateProtocolIEs_EntryValue::Id_SupportedTAList(
                supported_ta_list,
            ),
        });
    }

    // IE: DefaultPagingDRX (optional)
    if let Some(drx) = params.default_paging_drx {
        protocol_ies.push(RANConfigurationUpdateProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_DEFAULT_PAGING_DRX),
            criticality: Criticality(Criticality::IGNORE),
            value: RANConfigurationUpdateProtocolIEs_EntryValue::Id_DefaultPagingDRX(drx.into()),
        });
    }

    let ran_configuration_update = RANConfigurationUpdate {
        protocol_i_es: RANConfigurationUpdateProtocolIEs(protocol_ies),
    };

    let initiating_message = InitiatingMessage {
        procedure_code: ProcedureCode(ID_RAN_CONFIGURATION_UPDATE),
        criticality: Criticality(Criticality::REJECT),
        value: InitiatingMessageValue::Id_RANConfigurationUpdate(ran_configuration_update),
    };

    Ok(NGAP_PDU::InitiatingMessage(initiating_message))
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

/// Parse a RAN Configuration Update from an NGAP PDU
pub fn parse_ran_configuration_update(
    pdu: &NGAP_PDU,
) -> Result<RanConfigurationUpdateData, RanConfigurationUpdateError> {
    let initiating_message = match pdu {
        NGAP_PDU::InitiatingMessage(msg) => msg,
        _ => {
            return Err(RanConfigurationUpdateError::InvalidMessageType {
                expected: "InitiatingMessage".to_string(),
                actual: format!("{pdu:?}"),
            })
        }
    };

    let request = match &initiating_message.value {
        InitiatingMessageValue::Id_RANConfigurationUpdate(req) => req,
        _ => {
            return Err(RanConfigurationUpdateError::InvalidMessageType {
                expected: "RANConfigurationUpdate".to_string(),
                actual: format!("{:?}", initiating_message.value),
            })
        }
    };

    let mut ran_node_name: Option<String> = None;
    let mut supported_ta_list: Option<Vec<SupportedTaItem>> = None;
    let mut default_paging_drx: Option<PagingDrx> = None;

    for ie in &request.protocol_i_es.0 {
        match &ie.value {
            RANConfigurationUpdateProtocolIEs_EntryValue::Id_RANNodeName(name) => {
                ran_node_name = Some(name.0.clone());
            }
            RANConfigurationUpdateProtocolIEs_EntryValue::Id_SupportedTAList(list) => {
                supported_ta_list = Some(parse_supported_ta_list(list));
            }
            RANConfigurationUpdateProtocolIEs_EntryValue::Id_DefaultPagingDRX(drx) => {
                default_paging_drx = parse_paging_drx(drx);
            }
            _ => {}
        }
    }

    Ok(RanConfigurationUpdateData {
        ran_node_name,
        supported_ta_list,
        default_paging_drx,
    })
}

fn parse_supported_ta_list(list: &SupportedTAList) -> Vec<SupportedTaItem> {
    list.0
        .iter()
        .map(|ta| {
            let tac: [u8; 3] = ta.tac.0.as_slice().try_into().unwrap_or([0, 0, 0]);

            let broadcast_plmn_list: Vec<BroadcastPlmnItem> = ta
                .broadcast_plmn_list
                .0
                .iter()
                .map(|bp| {
                    let plmn_identity: [u8; 3] = bp
                        .plmn_identity
                        .0
                        .as_slice()
                        .try_into()
                        .unwrap_or([0, 0, 0]);

                    let slice_support_list: Vec<SNssai> = bp
                        .tai_slice_support_list
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

                    BroadcastPlmnItem {
                        plmn_identity,
                        slice_support_list,
                    }
                })
                .collect();

            SupportedTaItem {
                tac,
                broadcast_plmn_list,
            }
        })
        .collect()
}

fn parse_paging_drx(drx: &PagingDRX) -> Option<PagingDrx> {
    match drx.0 {
        PagingDRX::V32 => Some(PagingDrx::V32),
        PagingDRX::V64 => Some(PagingDrx::V64),
        PagingDRX::V128 => Some(PagingDrx::V128),
        PagingDRX::V256 => Some(PagingDrx::V256),
        _ => None,
    }
}

/// Parse a RAN Configuration Update Failure from an NGAP PDU
pub fn parse_ran_configuration_update_failure(
    pdu: &NGAP_PDU,
) -> Result<RanConfigurationUpdateFailureData, RanConfigurationUpdateError> {
    let unsuccessful_outcome = match pdu {
        NGAP_PDU::UnsuccessfulOutcome(outcome) => outcome,
        _ => {
            return Err(RanConfigurationUpdateError::InvalidMessageType {
                expected: "UnsuccessfulOutcome".to_string(),
                actual: format!("{pdu:?}"),
            })
        }
    };

    let failure = match &unsuccessful_outcome.value {
        UnsuccessfulOutcomeValue::Id_RANConfigurationUpdate(fail) => fail,
        _ => {
            return Err(RanConfigurationUpdateError::InvalidMessageType {
                expected: "RANConfigurationUpdateFailure".to_string(),
                actual: format!("{:?}", unsuccessful_outcome.value),
            })
        }
    };

    let mut cause: Option<NgSetupFailureCause> = None;
    let mut time_to_wait: Option<TimeToWaitValue> = None;

    for ie in &failure.protocol_i_es.0 {
        match &ie.value {
            RANConfigurationUpdateFailureProtocolIEs_EntryValue::Id_Cause(c) => {
                cause = Some(parse_cause(c));
            }
            RANConfigurationUpdateFailureProtocolIEs_EntryValue::Id_TimeToWait(ttw) => {
                time_to_wait = Some(parse_time_to_wait(ttw));
            }
            _ => {}
        }
    }

    Ok(RanConfigurationUpdateFailureData {
        cause: cause.ok_or_else(|| {
            RanConfigurationUpdateError::MissingMandatoryIe("Cause".to_string())
        })?,
        time_to_wait,
    })
}

fn parse_cause(cause: &Cause) -> NgSetupFailureCause {
    match cause {
        Cause::RadioNetwork(rn) => {
            let rc = match rn.0 {
                CauseRadioNetwork::UNSPECIFIED => RadioNetworkCause::Unspecified,
                other => RadioNetworkCause::Other(other),
            };
            NgSetupFailureCause::RadioNetwork(rc)
        }
        Cause::Transport(t) => {
            let tc = match t.0 {
                CauseTransport::TRANSPORT_RESOURCE_UNAVAILABLE => {
                    TransportCause::TransportResourceUnavailable
                }
                CauseTransport::UNSPECIFIED => TransportCause::Unspecified,
                other => TransportCause::Other(other),
            };
            NgSetupFailureCause::Transport(tc)
        }
        Cause::Nas(n) => {
            let nc = match n.0 {
                CauseNas::UNSPECIFIED => NasCause::Unspecified,
                other => NasCause::Other(other),
            };
            NgSetupFailureCause::Nas(nc)
        }
        Cause::Protocol(p) => {
            let pc = match p.0 {
                CauseProtocol::UNSPECIFIED => ProtocolCause::Unspecified,
                other => ProtocolCause::Other(other),
            };
            NgSetupFailureCause::Protocol(pc)
        }
        Cause::Misc(m) => {
            let mc = match m.0 {
                CauseMisc::UNSPECIFIED => MiscCause::Unspecified,
                other => MiscCause::Other(other),
            };
            NgSetupFailureCause::Misc(mc)
        }
        Cause::Choice_Extensions(_) => NgSetupFailureCause::Misc(MiscCause::Unspecified),
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
        _ => TimeToWaitValue::V1s,
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Build and encode a RAN Configuration Update to bytes
pub fn encode_ran_configuration_update(
    params: &RanConfigurationUpdateParams,
) -> Result<Vec<u8>, RanConfigurationUpdateError> {
    let pdu = build_ran_configuration_update(params)?;
    Ok(encode_ngap_pdu(&pdu)?)
}

/// Decode and parse a RAN Configuration Update from bytes
pub fn decode_ran_configuration_update(
    bytes: &[u8],
) -> Result<RanConfigurationUpdateData, RanConfigurationUpdateError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_ran_configuration_update(&pdu)
}

/// Check if an NGAP PDU is a RAN Configuration Update
pub fn is_ran_configuration_update(pdu: &NGAP_PDU) -> bool {
    matches!(
        pdu,
        NGAP_PDU::InitiatingMessage(msg)
            if matches!(msg.value, InitiatingMessageValue::Id_RANConfigurationUpdate(_))
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_params() -> RanConfigurationUpdateParams {
        RanConfigurationUpdateParams {
            ran_node_name: Some("nextgsim-gnb-updated".to_string()),
            supported_ta_list: Some(vec![SupportedTaItem {
                tac: [0x00, 0x00, 0x02],
                broadcast_plmn_list: vec![BroadcastPlmnItem {
                    plmn_identity: [0x00, 0xF1, 0x10],
                    slice_support_list: vec![SNssai { sst: 1, sd: None }],
                }],
            }]),
            default_paging_drx: Some(PagingDrx::V128),
        }
    }

    #[test]
    fn test_build_ran_configuration_update() {
        let params = create_test_params();
        let result = build_ran_configuration_update(&params);
        assert!(result.is_ok());

        let pdu = result.unwrap();
        assert!(is_ran_configuration_update(&pdu));
    }

    #[test]
    fn test_encode_ran_configuration_update() {
        let params = create_test_params();
        let result = encode_ran_configuration_update(&params);
        assert!(result.is_ok());

        let bytes = result.unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_ran_configuration_update_roundtrip() {
        let params = create_test_params();

        let pdu = build_ran_configuration_update(&params).expect("Failed to build");
        let encoded = encode_ngap_pdu(&pdu).expect("Failed to encode");
        let decoded_pdu = decode_ngap_pdu(&encoded).expect("Failed to decode");
        let parsed = parse_ran_configuration_update(&decoded_pdu).expect("Failed to parse");

        assert_eq!(parsed.ran_node_name, params.ran_node_name);
        assert!(parsed.supported_ta_list.is_some());
        assert!(parsed.default_paging_drx.is_some());
    }

    #[test]
    fn test_ran_configuration_update_minimal() {
        let params = RanConfigurationUpdateParams {
            ran_node_name: None,
            supported_ta_list: None,
            default_paging_drx: None,
        };

        let encoded = encode_ran_configuration_update(&params).expect("Failed to encode");
        let parsed = decode_ran_configuration_update(&encoded).expect("Failed to decode");

        assert!(parsed.ran_node_name.is_none());
        assert!(parsed.supported_ta_list.is_none());
        assert!(parsed.default_paging_drx.is_none());
    }

    #[test]
    fn test_ran_configuration_update_name_only() {
        let params = RanConfigurationUpdateParams {
            ran_node_name: Some("new-gnb-name".to_string()),
            supported_ta_list: None,
            default_paging_drx: None,
        };

        let encoded = encode_ran_configuration_update(&params).expect("Failed to encode");
        let parsed = decode_ran_configuration_update(&encoded).expect("Failed to decode");

        assert_eq!(parsed.ran_node_name, Some("new-gnb-name".to_string()));
    }
}
