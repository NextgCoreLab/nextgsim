//! NG Reset and AMF Configuration Update (receive side)
//!
//! Implements the gNB receive side of two AMF-initiated NGAP procedures:
//!
//! * **NG Reset** (TS 38.413 §8.7.4.2): the AMF requests the NG-RAN node to
//!   release either all UE-associated logical NG-connections (`NG Interface`)
//!   or a specified subset (`Part of NG Interface`). The NG-RAN node replies
//!   with an **NG Reset Acknowledge** that lists the UE associations it
//!   actually released.
//! * **AMF Configuration Update** (TS 38.413 §8.7.3): the AMF announces updated
//!   configuration (served GUAMI list, relative capacity, PLMN support, AMF
//!   name, TNL associations). The NG-RAN node updates its stored AMF state and
//!   replies with an **AMF Configuration Update Acknowledge**.
//!
//! Only the *decode-request / build-acknowledge* direction is provided here —
//! these are the messages the gNB consumes/answers. Both Acknowledge builders
//! round-trip through strict APER.

use crate::codec::generated::*;
use crate::codec::{decode_ngap_pdu, encode_ngap_pdu, NgapCodecError};
use crate::procedures::ng_setup::{
    parse_cause, parse_plmn_support_list, parse_served_guami_list, NgSetupFailureCause,
    PlmnSupportItem, ServedGuamiItem,
};
use thiserror::Error;

/// Errors that can occur during NG Reset / AMF Configuration Update handling.
#[derive(Debug, Error)]
pub enum NgResetError {
    /// Codec error during encoding/decoding.
    #[error("Codec error: {0}")]
    CodecError(#[from] NgapCodecError),

    /// Invalid message type received.
    #[error("Invalid message type: expected {expected}, got {actual}")]
    InvalidMessageType {
        /// Expected message type.
        expected: String,
        /// Actual message type received.
        actual: String,
    },

    /// Missing mandatory IE.
    #[error("Missing mandatory IE: {0}")]
    MissingMandatoryIe(String),
}

// ============================================================================
// NG Reset (receive)
// ============================================================================

/// A single UE-associated logical NG-connection referenced by an NG Reset
/// (or echoed in the Acknowledge). Both identifiers are optional per ASN.1.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UeAssociation {
    /// AMF UE NGAP ID, if present.
    pub amf_ue_ngap_id: Option<u64>,
    /// RAN UE NGAP ID, if present.
    pub ran_ue_ngap_id: Option<u32>,
}

/// The scope of an NG Reset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NgResetScope {
    /// Reset the entire NG interface (release all UE-associated connections).
    All,
    /// Reset only the listed UE-associated logical NG-connections.
    Part(Vec<UeAssociation>),
}

/// Parsed NG Reset request data.
#[derive(Debug, Clone)]
pub struct NgResetData {
    /// Cause of the reset (mandatory in the ASN.1; informational here).
    pub cause: Option<NgSetupFailureCause>,
    /// Reset scope: all connections or a specific subset.
    pub scope: NgResetScope,
}

/// Parameters for building an NG Reset Acknowledge.
#[derive(Debug, Clone, Default)]
pub struct NgResetAcknowledgeParams {
    /// UE-associated logical NG-connections actually released. When `Some`,
    /// even an empty list is encoded (echoes "released nothing"); when `None`
    /// the IE is omitted (used for a full reset-all acknowledge).
    pub released: Option<Vec<UeAssociation>>,
}

/// Decode an NG Reset message from APER bytes.
pub fn decode_ng_reset(bytes: &[u8]) -> Result<NgResetData, NgResetError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_ng_reset(&pdu)
}

/// Parse an NG Reset from an already-decoded NGAP PDU.
pub fn parse_ng_reset(pdu: &NGAP_PDU) -> Result<NgResetData, NgResetError> {
    let initiating = match pdu {
        NGAP_PDU::InitiatingMessage(msg) => msg,
        _ => {
            return Err(NgResetError::InvalidMessageType {
                expected: "InitiatingMessage".to_string(),
                actual: format!("{pdu:?}"),
            })
        }
    };

    let reset = match &initiating.value {
        InitiatingMessageValue::Id_NGReset(r) => r,
        _ => {
            return Err(NgResetError::InvalidMessageType {
                expected: "NGReset".to_string(),
                actual: format!("{:?}", initiating.value),
            })
        }
    };

    let mut cause: Option<NgSetupFailureCause> = None;
    let mut scope: Option<NgResetScope> = None;

    for ie in &reset.protocol_i_es.0 {
        match &ie.value {
            NGResetProtocolIEs_EntryValue::Id_Cause(c) => {
                cause = Some(parse_cause(c));
            }
            NGResetProtocolIEs_EntryValue::Id_ResetType(rt) => {
                scope = Some(match rt {
                    ResetType::NG_Interface(_) => NgResetScope::All,
                    ResetType::PartOfNG_Interface(list) => {
                        NgResetScope::Part(parse_ue_association_list(list))
                    }
                    ResetType::Choice_Extensions(_) => NgResetScope::All,
                });
            }
        }
    }

    Ok(NgResetData {
        cause,
        scope: scope.ok_or_else(|| NgResetError::MissingMandatoryIe("ResetType".to_string()))?,
    })
}

fn parse_ue_association_list(list: &UE_associatedLogicalNG_connectionList) -> Vec<UeAssociation> {
    list.0
        .iter()
        .map(|item| UeAssociation {
            amf_ue_ngap_id: item.amf_ue_ngap_id.as_ref().map(|v| v.0),
            ran_ue_ngap_id: item.ran_ue_ngap_id.as_ref().map(|v| v.0),
        })
        .collect()
}

fn build_ue_association_list(
    associations: &[UeAssociation],
) -> UE_associatedLogicalNG_connectionList {
    UE_associatedLogicalNG_connectionList(
        associations
            .iter()
            .map(|a| UE_associatedLogicalNG_connectionItem {
                amf_ue_ngap_id: a.amf_ue_ngap_id.map(AMF_UE_NGAP_ID),
                ran_ue_ngap_id: a.ran_ue_ngap_id.map(RAN_UE_NGAP_ID),
                ie_extensions: None,
            })
            .collect(),
    )
}

/// Build an NG Reset Acknowledge PDU.
pub fn build_ng_reset_acknowledge(
    params: &NgResetAcknowledgeParams,
) -> Result<NGAP_PDU, NgResetError> {
    let mut protocol_ies = Vec::new();

    if let Some(ref released) = params.released {
        protocol_ies.push(NGResetAcknowledgeProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_UE_ASSOCIATED_LOGICAL_NG_CONNECTION_LIST),
            criticality: Criticality(Criticality::IGNORE),
            value:
                NGResetAcknowledgeProtocolIEs_EntryValue::Id_UE_associatedLogicalNG_connectionList(
                    build_ue_association_list(released),
                ),
        });
    }

    let ack = NGResetAcknowledge {
        protocol_i_es: NGResetAcknowledgeProtocolIEs(protocol_ies),
    };

    let successful_outcome = SuccessfulOutcome {
        procedure_code: ProcedureCode(ID_NG_RESET),
        criticality: Criticality(Criticality::REJECT),
        value: SuccessfulOutcomeValue::Id_NGReset(ack),
    };

    Ok(NGAP_PDU::SuccessfulOutcome(successful_outcome))
}

/// Build and APER-encode an NG Reset Acknowledge.
pub fn encode_ng_reset_acknowledge(
    params: &NgResetAcknowledgeParams,
) -> Result<Vec<u8>, NgResetError> {
    let pdu = build_ng_reset_acknowledge(params)?;
    Ok(encode_ngap_pdu(&pdu)?)
}

/// Parse an NG Reset Acknowledge from an NGAP PDU (used in tests / source side).
pub fn parse_ng_reset_acknowledge(
    pdu: &NGAP_PDU,
) -> Result<NgResetAcknowledgeParams, NgResetError> {
    let outcome = match pdu {
        NGAP_PDU::SuccessfulOutcome(o) => o,
        _ => {
            return Err(NgResetError::InvalidMessageType {
                expected: "SuccessfulOutcome".to_string(),
                actual: format!("{pdu:?}"),
            })
        }
    };

    let ack = match &outcome.value {
        SuccessfulOutcomeValue::Id_NGReset(a) => a,
        _ => {
            return Err(NgResetError::InvalidMessageType {
                expected: "NGResetAcknowledge".to_string(),
                actual: format!("{:?}", outcome.value),
            })
        }
    };

    let mut released = None;
    for ie in &ack.protocol_i_es.0 {
        if let NGResetAcknowledgeProtocolIEs_EntryValue::Id_UE_associatedLogicalNG_connectionList(
            list,
        ) = &ie.value
        {
            released = Some(parse_ue_association_list(list));
        }
    }

    Ok(NgResetAcknowledgeParams { released })
}

/// Decode an NG Reset Acknowledge from APER bytes.
pub fn decode_ng_reset_acknowledge(bytes: &[u8]) -> Result<NgResetAcknowledgeParams, NgResetError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_ng_reset_acknowledge(&pdu)
}

// ============================================================================
// AMF Configuration Update (receive)
// ============================================================================

/// Parsed AMF Configuration Update data (subset relevant to the NG-RAN node).
#[derive(Debug, Clone, Default)]
pub struct AmfConfigurationUpdateData {
    /// Updated AMF name, if present.
    pub amf_name: Option<String>,
    /// Updated served-GUAMI list (empty if the IE is absent).
    pub served_guami_list: Vec<ServedGuamiItem>,
    /// Relative AMF capacity (0..255), if present.
    pub relative_amf_capacity: Option<u8>,
    /// Updated PLMN support list (empty if the IE is absent).
    pub plmn_support_list: Vec<PlmnSupportItem>,
}

/// Parameters for building an AMF Configuration Update Acknowledge.
///
/// All IEs of the Acknowledge are OPTIONAL in the ASN.1; a bare acknowledge
/// (no IEs) is the conformant minimal positive response.
#[derive(Debug, Clone, Default)]
pub struct AmfConfigurationUpdateAcknowledgeParams {
    /// Placeholder for future TNL-association setup reporting. Currently the
    /// gNB sends a bare acknowledge.
    pub _reserved: (),
}

/// Decode an AMF Configuration Update message from APER bytes.
pub fn decode_amf_configuration_update(
    bytes: &[u8],
) -> Result<AmfConfigurationUpdateData, NgResetError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_amf_configuration_update(&pdu)
}

/// Parse an AMF Configuration Update from an NGAP PDU.
pub fn parse_amf_configuration_update(
    pdu: &NGAP_PDU,
) -> Result<AmfConfigurationUpdateData, NgResetError> {
    let initiating = match pdu {
        NGAP_PDU::InitiatingMessage(msg) => msg,
        _ => {
            return Err(NgResetError::InvalidMessageType {
                expected: "InitiatingMessage".to_string(),
                actual: format!("{pdu:?}"),
            })
        }
    };

    let update = match &initiating.value {
        InitiatingMessageValue::Id_AMFConfigurationUpdate(u) => u,
        _ => {
            return Err(NgResetError::InvalidMessageType {
                expected: "AMFConfigurationUpdate".to_string(),
                actual: format!("{:?}", initiating.value),
            })
        }
    };

    let mut data = AmfConfigurationUpdateData::default();
    for ie in &update.protocol_i_es.0 {
        match &ie.value {
            AMFConfigurationUpdateProtocolIEs_EntryValue::Id_AMFName(name) => {
                data.amf_name = Some(name.0.clone());
            }
            AMFConfigurationUpdateProtocolIEs_EntryValue::Id_ServedGUAMIList(list) => {
                data.served_guami_list = parse_served_guami_list(list);
            }
            AMFConfigurationUpdateProtocolIEs_EntryValue::Id_RelativeAMFCapacity(cap) => {
                data.relative_amf_capacity = Some(cap.0);
            }
            AMFConfigurationUpdateProtocolIEs_EntryValue::Id_PLMNSupportList(list) => {
                data.plmn_support_list = parse_plmn_support_list(list);
            }
            _ => {}
        }
    }

    Ok(data)
}

/// Build an AMF Configuration Update Acknowledge PDU (bare positive response).
pub fn build_amf_configuration_update_acknowledge(
    _params: &AmfConfigurationUpdateAcknowledgeParams,
) -> Result<NGAP_PDU, NgResetError> {
    let ack = AMFConfigurationUpdateAcknowledge {
        protocol_i_es: AMFConfigurationUpdateAcknowledgeProtocolIEs(Vec::new()),
    };

    let successful_outcome = SuccessfulOutcome {
        procedure_code: ProcedureCode(ID_AMF_CONFIGURATION_UPDATE),
        criticality: Criticality(Criticality::REJECT),
        value: SuccessfulOutcomeValue::Id_AMFConfigurationUpdate(ack),
    };

    Ok(NGAP_PDU::SuccessfulOutcome(successful_outcome))
}

/// Build and APER-encode an AMF Configuration Update Acknowledge.
pub fn encode_amf_configuration_update_acknowledge(
    params: &AmfConfigurationUpdateAcknowledgeParams,
) -> Result<Vec<u8>, NgResetError> {
    let pdu = build_amf_configuration_update_acknowledge(params)?;
    Ok(encode_ngap_pdu(&pdu)?)
}

/// Decode an AMF Configuration Update Acknowledge (used in round-trip tests).
pub fn decode_amf_configuration_update_acknowledge(bytes: &[u8]) -> Result<(), NgResetError> {
    let pdu = decode_ngap_pdu(bytes)?;
    match pdu {
        NGAP_PDU::SuccessfulOutcome(o) => match o.value {
            SuccessfulOutcomeValue::Id_AMFConfigurationUpdate(_) => Ok(()),
            other => Err(NgResetError::InvalidMessageType {
                expected: "AMFConfigurationUpdateAcknowledge".to_string(),
                actual: format!("{other:?}"),
            }),
        },
        other => Err(NgResetError::InvalidMessageType {
            expected: "SuccessfulOutcome".to_string(),
            actual: format!("{other:?}"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_ng_reset_pdu(reset_type: ResetType) -> Vec<u8> {
        let reset = NGReset {
            protocol_i_es: NGResetProtocolIEs(vec![
                NGResetProtocolIEs_Entry {
                    id: ProtocolIE_ID(ID_CAUSE),
                    criticality: Criticality(Criticality::IGNORE),
                    value: NGResetProtocolIEs_EntryValue::Id_Cause(Cause::Misc(CauseMisc(
                        CauseMisc::OM_INTERVENTION,
                    ))),
                },
                NGResetProtocolIEs_Entry {
                    id: ProtocolIE_ID(ID_RESET_TYPE),
                    criticality: Criticality(Criticality::REJECT),
                    value: NGResetProtocolIEs_EntryValue::Id_ResetType(reset_type),
                },
            ]),
        };
        let pdu = NGAP_PDU::InitiatingMessage(InitiatingMessage {
            procedure_code: ProcedureCode(ID_NG_RESET),
            criticality: Criticality(Criticality::REJECT),
            value: InitiatingMessageValue::Id_NGReset(reset),
        });
        encode_ngap_pdu(&pdu).expect("encode NG Reset")
    }

    #[test]
    fn test_decode_ng_reset_all() {
        let bytes = build_ng_reset_pdu(ResetType::NG_Interface(ResetAll(ResetAll::RESET_ALL)));
        let data = decode_ng_reset(&bytes).expect("decode");
        assert_eq!(data.scope, NgResetScope::All);
        assert!(data.cause.is_some());
    }

    #[test]
    fn test_decode_ng_reset_partial() {
        let list = UE_associatedLogicalNG_connectionList(vec![
            UE_associatedLogicalNG_connectionItem {
                amf_ue_ngap_id: Some(AMF_UE_NGAP_ID(42)),
                ran_ue_ngap_id: Some(RAN_UE_NGAP_ID(7)),
                ie_extensions: None,
            },
            UE_associatedLogicalNG_connectionItem {
                amf_ue_ngap_id: Some(AMF_UE_NGAP_ID(99)),
                ran_ue_ngap_id: None,
                ie_extensions: None,
            },
        ]);
        let bytes = build_ng_reset_pdu(ResetType::PartOfNG_Interface(list));
        let data = decode_ng_reset(&bytes).expect("decode");
        match data.scope {
            NgResetScope::Part(items) => {
                assert_eq!(items.len(), 2);
                assert_eq!(items[0].amf_ue_ngap_id, Some(42));
                assert_eq!(items[0].ran_ue_ngap_id, Some(7));
                assert_eq!(items[1].amf_ue_ngap_id, Some(99));
                assert_eq!(items[1].ran_ue_ngap_id, None);
            }
            other => panic!("expected Part, got {other:?}"),
        }
    }

    #[test]
    fn test_ng_reset_acknowledge_roundtrip_with_list() {
        let params = NgResetAcknowledgeParams {
            released: Some(vec![
                UeAssociation {
                    amf_ue_ngap_id: Some(42),
                    ran_ue_ngap_id: Some(7),
                },
                UeAssociation {
                    amf_ue_ngap_id: Some(99),
                    ran_ue_ngap_id: Some(8),
                },
            ]),
        };
        let bytes = encode_ng_reset_acknowledge(&params).expect("encode");
        let decoded = decode_ng_reset_acknowledge(&bytes).expect("decode");
        let released = decoded.released.expect("list present");
        assert_eq!(released.len(), 2);
        assert_eq!(released[0].amf_ue_ngap_id, Some(42));
        assert_eq!(released[1].ran_ue_ngap_id, Some(8));
    }

    #[test]
    fn test_ng_reset_acknowledge_roundtrip_empty() {
        // reset-all acknowledge: no UE-association list IE.
        let params = NgResetAcknowledgeParams { released: None };
        let bytes = encode_ng_reset_acknowledge(&params).expect("encode");
        let decoded = decode_ng_reset_acknowledge(&bytes).expect("decode");
        assert!(decoded.released.is_none());
    }

    fn build_amf_config_update_pdu() -> Vec<u8> {
        let update = AMFConfigurationUpdate {
            protocol_i_es: AMFConfigurationUpdateProtocolIEs(vec![
                AMFConfigurationUpdateProtocolIEs_Entry {
                    id: ProtocolIE_ID(ID_AMF_NAME),
                    criticality: Criticality(Criticality::REJECT),
                    value: AMFConfigurationUpdateProtocolIEs_EntryValue::Id_AMFName(AMFName(
                        "test-amf".to_string(),
                    )),
                },
                AMFConfigurationUpdateProtocolIEs_Entry {
                    id: ProtocolIE_ID(ID_RELATIVE_AMF_CAPACITY),
                    criticality: Criticality(Criticality::IGNORE),
                    value: AMFConfigurationUpdateProtocolIEs_EntryValue::Id_RelativeAMFCapacity(
                        RelativeAMFCapacity(200),
                    ),
                },
            ]),
        };
        let pdu = NGAP_PDU::InitiatingMessage(InitiatingMessage {
            procedure_code: ProcedureCode(ID_AMF_CONFIGURATION_UPDATE),
            criticality: Criticality(Criticality::REJECT),
            value: InitiatingMessageValue::Id_AMFConfigurationUpdate(update),
        });
        encode_ngap_pdu(&pdu).expect("encode AMF Config Update")
    }

    #[test]
    fn test_decode_amf_configuration_update() {
        let bytes = build_amf_config_update_pdu();
        let data = decode_amf_configuration_update(&bytes).expect("decode");
        assert_eq!(data.amf_name.as_deref(), Some("test-amf"));
        assert_eq!(data.relative_amf_capacity, Some(200));
    }

    #[test]
    fn test_amf_configuration_update_acknowledge_roundtrip() {
        let params = AmfConfigurationUpdateAcknowledgeParams::default();
        let bytes = encode_amf_configuration_update_acknowledge(&params).expect("encode");
        decode_amf_configuration_update_acknowledge(&bytes).expect("decode");
    }
}
