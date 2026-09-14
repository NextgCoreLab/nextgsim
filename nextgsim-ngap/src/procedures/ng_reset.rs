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
    /// TNL associations the AMF asks the NG-RAN node to ADD
    /// (`AMF-TNLAssociationToAddList`, TS 38.413 §9.2.6.5 / §9.3.3.20).
    pub tnla_to_add: Vec<AmfTnlAssociationToAdd>,
    /// TNL associations the AMF asks the NG-RAN node to REMOVE.
    pub tnla_to_remove: Vec<AmfTnlAssociationAddress>,
    /// TNL associations whose usage or weight the AMF asks to UPDATE.
    pub tnla_to_update: Vec<AmfTnlAssociationToUpdate>,
}

/// What a TNL association may be used for (TS 38.413 §9.3.3.22
/// `TNLAssociationUsage`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TnlAssociationUsage {
    /// UE-associated signalling only
    Ue,
    /// Non-UE-associated signalling only
    NonUe,
    /// Both
    Both,
}

impl TnlAssociationUsage {
    fn from_asn(value: &TNLAssociationUsage) -> Option<Self> {
        match value.0 {
            TNLAssociationUsage::UE => Some(Self::Ue),
            TNLAssociationUsage::NON_UE => Some(Self::NonUe),
            TNLAssociationUsage::BOTH => Some(Self::Both),
            // An unknown usage is dropped rather than guessed: `TNLAssociationUsage`
            // is extensible, and defaulting an unrecognised value to `Both` would
            // let UE traffic onto an association the AMF reserved for something
            // else.
            _ => None,
        }
    }
}

/// A TNL association address as the AMF gave it.
///
/// `CPTransportLayerInformation` is a CHOICE whose only Rel-15 alternative is an
/// `endpointIPAddress` — a `TransportLayerAddress` BIT STRING of 32 or 128 bits.
/// Decoded to an `IpAddr` here, following the same bit-by-bit conversion the
/// GTP-tunnel path in `transfer.rs` uses, because a consumer that has to open an
/// SCTP association needs an address rather than a bit string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AmfTnlAssociationAddress {
    /// The endpoint the AMF named
    pub endpoint: std::net::IpAddr,
}

impl AmfTnlAssociationAddress {
    /// `None` when the address is one this codec cannot turn into an endpoint:
    /// a choice extension (a transport not modelled here) or a bit string that is
    /// neither 32 nor 128 bits.
    ///
    /// Absent rather than a placeholder address, deliberately: a consumer handed
    /// `0.0.0.0` would try to connect to it and report a transport failure, when
    /// the truth is that the gNB never understood the request.
    fn from_asn(info: &CPTransportLayerInformation) -> Option<Self> {
        let CPTransportLayerInformation::EndpointIPAddress(addr) = info else {
            return None;
        };
        let bits = &addr.0;
        match bits.len() {
            32 => {
                let mut octets = [0u8; 4];
                for (i, octet) in octets.iter_mut().enumerate() {
                    for b in 0..8 {
                        if bits[i * 8 + b] {
                            *octet |= 1 << (7 - b);
                        }
                    }
                }
                Some(Self {
                    endpoint: std::net::IpAddr::V4(octets.into()),
                })
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
                Some(Self {
                    endpoint: std::net::IpAddr::V6(octets.into()),
                })
            }
            _ => None,
        }
    }

    /// Encodes this address back into a `CPTransportLayerInformation`, for the
    /// Acknowledge's setup / failed-to-setup lists.
    fn to_asn(self) -> CPTransportLayerInformation {
        let mut bits = bitvec::vec::BitVec::<u8, bitvec::order::Msb0>::new();
        match self.endpoint {
            std::net::IpAddr::V4(v4) => {
                for octet in v4.octets() {
                    for b in 0..8 {
                        bits.push(octet & (1 << (7 - b)) != 0);
                    }
                }
            }
            std::net::IpAddr::V6(v6) => {
                for octet in v6.octets() {
                    for b in 0..8 {
                        bits.push(octet & (1 << (7 - b)) != 0);
                    }
                }
            }
        }
        CPTransportLayerInformation::EndpointIPAddress(TransportLayerAddress(bits))
    }
}

/// A TNL association the AMF asks the NG-RAN node to add.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AmfTnlAssociationToAdd {
    /// Where to connect
    pub address: AmfTnlAssociationAddress,
    /// What the association may carry, when the AMF said
    pub usage: Option<TnlAssociationUsage>,
    /// `TNLAddressWeightFactor` (0..255): the share of traffic this association
    /// should take relative to the AMF's others.
    pub weight_factor: u8,
}

/// A TNL association whose usage or weight the AMF asks to update.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AmfTnlAssociationToUpdate {
    /// Which association
    pub address: AmfTnlAssociationAddress,
    /// New usage, when the AMF said
    pub usage: Option<TnlAssociationUsage>,
    /// New weight factor, when the AMF said
    pub weight_factor: Option<u8>,
}

/// Parameters for building an AMF Configuration Update Acknowledge.
///
/// All IEs of the Acknowledge are OPTIONAL in the ASN.1; a bare acknowledge
/// (no IEs) is the conformant minimal positive response.
#[derive(Debug, Clone, Default)]
pub struct AmfConfigurationUpdateAcknowledgeParams {
    /// TNL associations the NG-RAN node DID set up, reported back in
    /// `AMF-TNLAssociationSetupList` (TS 38.413 §9.2.6.6).
    pub tnla_setup: Vec<AmfTnlAssociationAddress>,
    /// TNL associations the NG-RAN node could NOT set up, reported in
    /// `AMF-TNLAssociationFailedToSetupList` with a cause.
    ///
    /// Populated rather than left empty when a requested association cannot be
    /// established: a bare acknowledge tells the AMF nothing went wrong, and the
    /// AMF would then balance traffic onto an association that does not exist.
    pub tnla_failed_to_setup: Vec<AmfTnlAssociationAddress>,
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
            AMFConfigurationUpdateProtocolIEs_EntryValue::Id_AMF_TNLAssociationToAddList(list) => {
                for item in &list.0 {
                    // An item whose address this codec cannot model is SKIPPED, not
                    // represented with an empty address: the consumer would try to
                    // connect to it.
                    if let Some(address) =
                        AmfTnlAssociationAddress::from_asn(&item.amf_tnl_association_address)
                    {
                        data.tnla_to_add.push(AmfTnlAssociationToAdd {
                            address,
                            usage: item
                                .tnl_association_usage
                                .as_ref()
                                .and_then(TnlAssociationUsage::from_asn),
                            weight_factor: item.tnl_address_weight_factor.0,
                        });
                    }
                }
            }
            AMFConfigurationUpdateProtocolIEs_EntryValue::Id_AMF_TNLAssociationToRemoveList(
                list,
            ) => {
                for item in &list.0 {
                    if let Some(address) =
                        AmfTnlAssociationAddress::from_asn(&item.amf_tnl_association_address)
                    {
                        data.tnla_to_remove.push(address);
                    }
                }
            }
            AMFConfigurationUpdateProtocolIEs_EntryValue::Id_AMF_TNLAssociationToUpdateList(
                list,
            ) => {
                for item in &list.0 {
                    if let Some(address) =
                        AmfTnlAssociationAddress::from_asn(&item.amf_tnl_association_address)
                    {
                        data.tnla_to_update.push(AmfTnlAssociationToUpdate {
                            address,
                            usage: item
                                .tnl_association_usage
                                .as_ref()
                                .and_then(TnlAssociationUsage::from_asn),
                            weight_factor: item.tnl_address_weight_factor.as_ref().map(|w| w.0),
                        });
                    }
                }
            }
            _ => {}
        }
    }

    Ok(data)
}

/// Build an AMF Configuration Update Acknowledge PDU (bare positive response).
pub fn build_amf_configuration_update_acknowledge(
    params: &AmfConfigurationUpdateAcknowledgeParams,
) -> Result<NGAP_PDU, NgResetError> {
    let mut ies = Vec::new();

    // Both lists are SIZE (1..32): an empty one is not encodable, so an empty
    // request omits the IE. A bare acknowledge stays the conformant minimal
    // positive response when the AMF asked for no TNLA change.
    if !params.tnla_setup.is_empty() {
        ies.push(AMFConfigurationUpdateAcknowledgeProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_AMF_TNL_ASSOCIATION_SETUP_LIST),
            criticality: Criticality(Criticality::IGNORE),
            value:
                AMFConfigurationUpdateAcknowledgeProtocolIEs_EntryValue::Id_AMF_TNLAssociationSetupList(
                    AMF_TNLAssociationSetupList(
                        params
                            .tnla_setup
                            .iter()
                            .map(|a| AMF_TNLAssociationSetupItem {
                                amf_tnl_association_address: a.to_asn(),
                                ie_extensions: None,
                            })
                            .collect(),
                    ),
                ),
        });
    }

    if !params.tnla_failed_to_setup.is_empty() {
        ies.push(AMFConfigurationUpdateAcknowledgeProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_AMF_TNL_ASSOCIATION_FAILED_TO_SETUP_LIST),
            criticality: Criticality(Criticality::IGNORE),
            value:
                AMFConfigurationUpdateAcknowledgeProtocolIEs_EntryValue::Id_AMF_TNLAssociationFailedToSetupList(
                    TNLAssociationList(
                        params
                            .tnla_failed_to_setup
                            .iter()
                            .map(|a| TNLAssociationItem {
                                tnl_association_address: a.to_asn(),
                                // transport-resource-unavailable: the NG-RAN node
                                // could not bring the association up. A specific
                                // cause rather than `unspecified`, because "the
                                // transport did not come up" is what actually
                                // happened and the AMF can act on it.
                                cause: Cause::Transport(CauseTransport(
                                    CauseTransport::TRANSPORT_RESOURCE_UNAVAILABLE,
                                )),
                                ie_extensions: None,
                            })
                            .collect(),
                    ),
                ),
        });
    }

    let ack = AMFConfigurationUpdateAcknowledge {
        protocol_i_es: AMFConfigurationUpdateAcknowledgeProtocolIEs(ies),
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

/// What an AMF Configuration Update Acknowledge reported.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AmfConfigurationUpdateAcknowledgeData {
    /// TNL associations the NG-RAN node set up
    pub tnla_setup: Vec<AmfTnlAssociationAddress>,
    /// TNL associations it could not set up
    pub tnla_failed_to_setup: Vec<AmfTnlAssociationAddress>,
}

/// Decode an AMF Configuration Update Acknowledge, returning what it reported.
pub fn parse_amf_configuration_update_acknowledge(
    bytes: &[u8],
) -> Result<AmfConfigurationUpdateAcknowledgeData, NgResetError> {
    let pdu = decode_ngap_pdu(bytes)?;
    let NGAP_PDU::SuccessfulOutcome(outcome) = pdu else {
        return Err(NgResetError::InvalidMessageType {
            expected: "SuccessfulOutcome".to_string(),
            actual: format!("{pdu:?}"),
        });
    };
    let SuccessfulOutcomeValue::Id_AMFConfigurationUpdate(ack) = outcome.value else {
        return Err(NgResetError::InvalidMessageType {
            expected: "AMFConfigurationUpdateAcknowledge".to_string(),
            actual: format!("{:?}", outcome.value),
        });
    };

    let mut data = AmfConfigurationUpdateAcknowledgeData::default();
    for ie in &ack.protocol_i_es.0 {
        match &ie.value {
            AMFConfigurationUpdateAcknowledgeProtocolIEs_EntryValue::Id_AMF_TNLAssociationSetupList(
                list,
            ) => {
                data.tnla_setup = list
                    .0
                    .iter()
                    .filter_map(|i| AmfTnlAssociationAddress::from_asn(&i.amf_tnl_association_address))
                    .collect();
            }
            AMFConfigurationUpdateAcknowledgeProtocolIEs_EntryValue::Id_AMF_TNLAssociationFailedToSetupList(
                list,
            ) => {
                data.tnla_failed_to_setup = list
                    .0
                    .iter()
                    .filter_map(|i| AmfTnlAssociationAddress::from_asn(&i.tnl_association_address))
                    .collect();
            }
            AMFConfigurationUpdateAcknowledgeProtocolIEs_EntryValue::Id_CriticalityDiagnostics(_) => {}
        }
    }
    Ok(data)
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

    // ========================================================================
    // TNL association lists (issue #41)
    // ========================================================================

    /// The three TNLA lists must actually be PARSED. Before this they were IEs the
    /// decoder's catch-all arm dropped, so an AMF asking the gNB to add an
    /// association got a bare acknowledge that reads as "done".
    #[test]
    fn the_tnl_association_lists_round_trip_through_an_amf_configuration_update() {
        use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

        let add_v4 = IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1));
        let add_v6 = IpAddr::V6(Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 1));
        let remove = IpAddr::V4(Ipv4Addr::new(10, 0, 0, 2));
        let update = IpAddr::V4(Ipv4Addr::new(10, 0, 0, 3));

        // Built with the generated types directly, because this crate encodes the
        // ACKNOWLEDGE and only DECODES the update -- there is no gNB-side builder
        // for an AMF-originated message to round trip through.
        let ies = vec![
            AMFConfigurationUpdateProtocolIEs_Entry {
                id: ProtocolIE_ID(ID_AMF_TNL_ASSOCIATION_TO_ADD_LIST),
                criticality: Criticality(Criticality::IGNORE),
                value: AMFConfigurationUpdateProtocolIEs_EntryValue::Id_AMF_TNLAssociationToAddList(
                    AMF_TNLAssociationToAddList(vec![
                        AMF_TNLAssociationToAddItem {
                            amf_tnl_association_address: AmfTnlAssociationAddress {
                                endpoint: add_v4,
                            }
                            .to_asn(),
                            tnl_association_usage: Some(TNLAssociationUsage(
                                TNLAssociationUsage::BOTH,
                            )),
                            tnl_address_weight_factor: TNLAddressWeightFactor(50),
                            ie_extensions: None,
                        },
                        AMF_TNLAssociationToAddItem {
                            amf_tnl_association_address: AmfTnlAssociationAddress {
                                endpoint: add_v6,
                            }
                            .to_asn(),
                            tnl_association_usage: Some(TNLAssociationUsage(
                                TNLAssociationUsage::NON_UE,
                            )),
                            tnl_address_weight_factor: TNLAddressWeightFactor(10),
                            ie_extensions: None,
                        },
                    ]),
                ),
            },
            AMFConfigurationUpdateProtocolIEs_Entry {
                id: ProtocolIE_ID(ID_AMF_TNL_ASSOCIATION_TO_REMOVE_LIST),
                criticality: Criticality(Criticality::IGNORE),
                value:
                    AMFConfigurationUpdateProtocolIEs_EntryValue::Id_AMF_TNLAssociationToRemoveList(
                        AMF_TNLAssociationToRemoveList(vec![AMF_TNLAssociationToRemoveItem {
                            amf_tnl_association_address: AmfTnlAssociationAddress {
                                endpoint: remove,
                            }
                            .to_asn(),
                            ie_extensions: None,
                        }]),
                    ),
            },
            AMFConfigurationUpdateProtocolIEs_Entry {
                id: ProtocolIE_ID(ID_AMF_TNL_ASSOCIATION_TO_UPDATE_LIST),
                criticality: Criticality(Criticality::IGNORE),
                value:
                    AMFConfigurationUpdateProtocolIEs_EntryValue::Id_AMF_TNLAssociationToUpdateList(
                        AMF_TNLAssociationToUpdateList(vec![AMF_TNLAssociationToUpdateItem {
                            amf_tnl_association_address: AmfTnlAssociationAddress {
                                endpoint: update,
                            }
                            .to_asn(),
                            tnl_association_usage: Some(TNLAssociationUsage(
                                TNLAssociationUsage::UE,
                            )),
                            tnl_address_weight_factor: Some(TNLAddressWeightFactor(200)),
                            ie_extensions: None,
                        }]),
                    ),
            },
        ];
        let pdu = NGAP_PDU::InitiatingMessage(InitiatingMessage {
            procedure_code: ProcedureCode(ID_AMF_CONFIGURATION_UPDATE),
            criticality: Criticality(Criticality::REJECT),
            value: InitiatingMessageValue::Id_AMFConfigurationUpdate(AMFConfigurationUpdate {
                protocol_i_es: AMFConfigurationUpdateProtocolIEs(ies),
            }),
        });
        let bytes = encode_ngap_pdu(&pdu).expect("encodes");
        let data = decode_amf_configuration_update(&bytes).expect("decodes");

        assert_eq!(data.tnla_to_add.len(), 2);
        assert_eq!(data.tnla_to_add[0].address.endpoint, add_v4);
        assert_eq!(data.tnla_to_add[0].usage, Some(TnlAssociationUsage::Both));
        assert_eq!(data.tnla_to_add[0].weight_factor, 50);
        assert_eq!(
            data.tnla_to_add[1].address.endpoint, add_v6,
            "an IPv6 endpoint must survive: the BIT STRING is 128 bits, not 32"
        );
        assert_eq!(data.tnla_to_add[1].usage, Some(TnlAssociationUsage::NonUe));

        assert_eq!(data.tnla_to_remove.len(), 1);
        assert_eq!(data.tnla_to_remove[0].endpoint, remove);

        assert_eq!(data.tnla_to_update.len(), 1);
        assert_eq!(data.tnla_to_update[0].address.endpoint, update);
        assert_eq!(data.tnla_to_update[0].usage, Some(TnlAssociationUsage::Ue));
        assert_eq!(data.tnla_to_update[0].weight_factor, Some(200));
    }

    /// The Acknowledge's setup and failed-to-setup lists round trip, and an
    /// Acknowledge with nothing to report is still the bare positive response.
    #[test]
    fn the_acknowledge_reports_setup_and_failed_tnl_associations() {
        use std::net::{IpAddr, Ipv4Addr};

        let ok = AmfTnlAssociationAddress {
            endpoint: IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)),
        };
        let bad = AmfTnlAssociationAddress {
            endpoint: IpAddr::V4(Ipv4Addr::new(10, 0, 0, 9)),
        };

        let bytes =
            encode_amf_configuration_update_acknowledge(&AmfConfigurationUpdateAcknowledgeParams {
                tnla_setup: vec![ok],
                tnla_failed_to_setup: vec![bad],
            })
            .expect("encodes");
        let data = parse_amf_configuration_update_acknowledge(&bytes).expect("decodes");
        assert_eq!(data.tnla_setup, vec![ok]);
        assert_eq!(
            data.tnla_failed_to_setup,
            vec![bad],
            "a failure must be reported as a FAILURE; a bare acknowledge tells the \
             AMF nothing went wrong and it would balance traffic onto an \
             association that does not exist"
        );

        // Nothing to report: still a valid bare acknowledge, and the lists are
        // OMITTED rather than encoded empty (both are SIZE (1..32)).
        let bytes = encode_amf_configuration_update_acknowledge(
            &AmfConfigurationUpdateAcknowledgeParams::default(),
        )
        .expect("encodes");
        let data = parse_amf_configuration_update_acknowledge(&bytes).expect("decodes");
        assert!(data.tnla_setup.is_empty() && data.tnla_failed_to_setup.is_empty());
        assert!(decode_amf_configuration_update_acknowledge(&bytes).is_ok());
    }
}
