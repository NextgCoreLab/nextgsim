//! UE radio capability procedures used by RACS (TS 23.501 §5.4.4.1a).
//!
//! * **UE Radio Capability ID Mapping** (TS 38.413 §8.14.3): Class 1,
//!   non-UE-associated. The NG-RAN node asks the AMF for the UE radio capability
//!   that a UE Radio Capability ID stands for. §8.14.3.3 defines no unsuccessful
//!   outcome and §8.14.3.4 no abnormal conditions.
//! * **UE Radio Capability Info Indication** (TS 38.413 §8.14.1): Class 2,
//!   UE-associated. The NG-RAN node reports the capability it retrieved from the UE
//!   (the §8.3.1.2 "neither the UE Radio Capability IE nor the UE Radio Capability ID
//!   IE" case), which is what lets an AMF obtain a capability to have a UCMF assign an
//!   ID for.
//!
//! The UE Radio Capability ID is carried as an opaque OCTET STRING
//! (TS 38.413 §9.3.1.142, "Defined in TS 23.003"). TS 38.413 does not fix how the
//! TS 23.003 §29.2 hexadecimal digits are packed into octets, so this module never
//! interprets the bytes: the NG-RAN node only compares IDs for equality.

use crate::codec::generated::*;
use crate::codec::{decode_ngap_pdu, encode_ngap_pdu, NgapCodecError};
use thiserror::Error;

/// Errors from the UE radio capability procedures.
#[derive(Debug, Error)]
pub enum UeRadioCapabilityError {
    /// Codec error during encoding/decoding
    #[error("Codec error: {0}")]
    CodecError(#[from] NgapCodecError),
    /// The PDU is not the expected message
    #[error("Invalid message type: expected {expected}")]
    InvalidMessageType {
        /// Expected message
        expected: &'static str,
    },
    /// A mandatory IE is absent
    #[error("Missing mandatory IE: {0}")]
    MissingMandatoryIe(&'static str),
}

// ============================================================================
// UE Radio Capability ID Mapping (TS 38.413 §8.14.3)
// ============================================================================

/// Build a UE RADIO CAPABILITY ID MAPPING REQUEST (TS 38.413 §9.2.15.4).
pub fn build_ue_radio_capability_id_mapping_request(ue_radio_capability_id: &[u8]) -> NGAP_PDU {
    let ies = vec![UERadioCapabilityIDMappingRequestProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_UE_RADIO_CAPABILITY_ID),
        criticality: Criticality(Criticality::REJECT),
        value: UERadioCapabilityIDMappingRequestProtocolIEs_EntryValue::Id_UERadioCapabilityID(
            UERadioCapabilityID(ue_radio_capability_id.to_vec()),
        ),
    }];
    NGAP_PDU::InitiatingMessage(InitiatingMessage {
        procedure_code: ProcedureCode(ID_UE_RADIO_CAPABILITY_ID_MAPPING),
        criticality: Criticality(Criticality::REJECT),
        value: InitiatingMessageValue::Id_UERadioCapabilityIDMapping(
            UERadioCapabilityIDMappingRequest {
                protocol_i_es: UERadioCapabilityIDMappingRequestProtocolIEs(ies),
            },
        ),
    })
}

/// Encode a UE RADIO CAPABILITY ID MAPPING REQUEST to APER bytes.
pub fn encode_ue_radio_capability_id_mapping_request(
    ue_radio_capability_id: &[u8],
) -> Result<Vec<u8>, UeRadioCapabilityError> {
    Ok(encode_ngap_pdu(
        &build_ue_radio_capability_id_mapping_request(ue_radio_capability_id),
    )?)
}

/// Decode a UE RADIO CAPABILITY ID MAPPING REQUEST, returning the requested ID.
pub fn decode_ue_radio_capability_id_mapping_request(
    bytes: &[u8],
) -> Result<Vec<u8>, UeRadioCapabilityError> {
    let pdu = decode_ngap_pdu(bytes)?;
    let NGAP_PDU::InitiatingMessage(InitiatingMessage {
        value: InitiatingMessageValue::Id_UERadioCapabilityIDMapping(req),
        ..
    }) = pdu
    else {
        return Err(UeRadioCapabilityError::InvalidMessageType {
            expected: "UERadioCapabilityIDMappingRequest",
        });
    };
    req.protocol_i_es
        .0
        .into_iter()
        .map(|ie| match ie.value {
            UERadioCapabilityIDMappingRequestProtocolIEs_EntryValue::Id_UERadioCapabilityID(id) => {
                id.0
            }
        })
        .next()
        .ok_or(UeRadioCapabilityError::MissingMandatoryIe(
            "UERadioCapabilityID",
        ))
}

/// A UE RADIO CAPABILITY ID MAPPING RESPONSE (TS 38.413 §9.2.15.5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UeRadioCapabilityIdMappingResponseData {
    /// The ID the response answers
    pub ue_radio_capability_id: Vec<u8>,
    /// The UE radio capability mapped to it: an encoded TS 38.331
    /// `UERadioAccessCapabilityInformation` (TS 38.413 §9.3.1.74)
    pub ue_radio_capability: Vec<u8>,
}

/// Build a UE RADIO CAPABILITY ID MAPPING RESPONSE. The AMF side of the procedure;
/// nextgsim uses it for its stand-in AMF in tests.
pub fn build_ue_radio_capability_id_mapping_response(
    data: &UeRadioCapabilityIdMappingResponseData,
) -> NGAP_PDU {
    let ies = vec![
        UERadioCapabilityIDMappingResponseProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_UE_RADIO_CAPABILITY_ID),
            criticality: Criticality(Criticality::REJECT),
            value: UERadioCapabilityIDMappingResponseProtocolIEs_EntryValue::Id_UERadioCapabilityID(
                UERadioCapabilityID(data.ue_radio_capability_id.clone()),
            ),
        },
        UERadioCapabilityIDMappingResponseProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_UE_RADIO_CAPABILITY),
            criticality: Criticality(Criticality::IGNORE),
            value: UERadioCapabilityIDMappingResponseProtocolIEs_EntryValue::Id_UERadioCapability(
                UERadioCapability(data.ue_radio_capability.clone()),
            ),
        },
    ];
    NGAP_PDU::SuccessfulOutcome(SuccessfulOutcome {
        procedure_code: ProcedureCode(ID_UE_RADIO_CAPABILITY_ID_MAPPING),
        criticality: Criticality(Criticality::REJECT),
        value: SuccessfulOutcomeValue::Id_UERadioCapabilityIDMapping(
            UERadioCapabilityIDMappingResponse {
                protocol_i_es: UERadioCapabilityIDMappingResponseProtocolIEs(ies),
            },
        ),
    })
}

/// Encode a UE RADIO CAPABILITY ID MAPPING RESPONSE to APER bytes.
pub fn encode_ue_radio_capability_id_mapping_response(
    data: &UeRadioCapabilityIdMappingResponseData,
) -> Result<Vec<u8>, UeRadioCapabilityError> {
    Ok(encode_ngap_pdu(
        &build_ue_radio_capability_id_mapping_response(data),
    )?)
}

/// Decode a UE RADIO CAPABILITY ID MAPPING RESPONSE.
pub fn decode_ue_radio_capability_id_mapping_response(
    bytes: &[u8],
) -> Result<UeRadioCapabilityIdMappingResponseData, UeRadioCapabilityError> {
    let pdu = decode_ngap_pdu(bytes)?;
    let NGAP_PDU::SuccessfulOutcome(SuccessfulOutcome {
        value: SuccessfulOutcomeValue::Id_UERadioCapabilityIDMapping(resp),
        ..
    }) = pdu
    else {
        return Err(UeRadioCapabilityError::InvalidMessageType {
            expected: "UERadioCapabilityIDMappingResponse",
        });
    };
    let mut id = None;
    let mut capability = None;
    for ie in resp.protocol_i_es.0 {
        match ie.value {
            UERadioCapabilityIDMappingResponseProtocolIEs_EntryValue::Id_UERadioCapabilityID(v) => {
                id = Some(v.0)
            }
            UERadioCapabilityIDMappingResponseProtocolIEs_EntryValue::Id_UERadioCapability(v) => {
                capability = Some(v.0)
            }
            UERadioCapabilityIDMappingResponseProtocolIEs_EntryValue::Id_CriticalityDiagnostics(
                _,
            ) => {}
        }
    }
    Ok(UeRadioCapabilityIdMappingResponseData {
        ue_radio_capability_id: id.ok_or(UeRadioCapabilityError::MissingMandatoryIe(
            "UERadioCapabilityID",
        ))?,
        ue_radio_capability: capability.ok_or(UeRadioCapabilityError::MissingMandatoryIe(
            "UERadioCapability",
        ))?,
    })
}

// ============================================================================
// UE Radio Capability Info Indication (TS 38.413 §8.14.1)
// ============================================================================

/// A UE RADIO CAPABILITY INFO INDICATION (TS 38.413 §9.2.15.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UeRadioCapabilityInfoIndicationData {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
    /// An encoded TS 38.331 `UERadioAccessCapabilityInformation`
    pub ue_radio_capability: Vec<u8>,
}

/// Build a UE RADIO CAPABILITY INFO INDICATION.
///
/// The optional UE Radio Capability for Paging and E-UTRA-format IEs are not built:
/// this NG-RAN node is mode of operation B (TS 23.501 §5.4.4.1a), providing the
/// TS 38.331 format only, and derives no paging subset.
pub fn build_ue_radio_capability_info_indication(
    data: &UeRadioCapabilityInfoIndicationData,
) -> NGAP_PDU {
    let ies = vec![
        UERadioCapabilityInfoIndicationProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_AMF_UE_NGAP_ID),
            criticality: Criticality(Criticality::REJECT),
            value: UERadioCapabilityInfoIndicationProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(
                AMF_UE_NGAP_ID(data.amf_ue_ngap_id),
            ),
        },
        UERadioCapabilityInfoIndicationProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_RAN_UE_NGAP_ID),
            criticality: Criticality(Criticality::REJECT),
            value: UERadioCapabilityInfoIndicationProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(
                RAN_UE_NGAP_ID(data.ran_ue_ngap_id),
            ),
        },
        UERadioCapabilityInfoIndicationProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_UE_RADIO_CAPABILITY),
            criticality: Criticality(Criticality::IGNORE),
            value: UERadioCapabilityInfoIndicationProtocolIEs_EntryValue::Id_UERadioCapability(
                UERadioCapability(data.ue_radio_capability.clone()),
            ),
        },
    ];
    NGAP_PDU::InitiatingMessage(InitiatingMessage {
        procedure_code: ProcedureCode(ID_UE_RADIO_CAPABILITY_INFO_INDICATION),
        criticality: Criticality(Criticality::IGNORE),
        value: InitiatingMessageValue::Id_UERadioCapabilityInfoIndication(
            UERadioCapabilityInfoIndication {
                protocol_i_es: UERadioCapabilityInfoIndicationProtocolIEs(ies),
            },
        ),
    })
}

/// Encode a UE RADIO CAPABILITY INFO INDICATION to APER bytes.
pub fn encode_ue_radio_capability_info_indication(
    data: &UeRadioCapabilityInfoIndicationData,
) -> Result<Vec<u8>, UeRadioCapabilityError> {
    Ok(encode_ngap_pdu(
        &build_ue_radio_capability_info_indication(data),
    )?)
}

/// Decode a UE RADIO CAPABILITY INFO INDICATION.
pub fn decode_ue_radio_capability_info_indication(
    bytes: &[u8],
) -> Result<UeRadioCapabilityInfoIndicationData, UeRadioCapabilityError> {
    let pdu = decode_ngap_pdu(bytes)?;
    let NGAP_PDU::InitiatingMessage(InitiatingMessage {
        value: InitiatingMessageValue::Id_UERadioCapabilityInfoIndication(ind),
        ..
    }) = pdu
    else {
        return Err(UeRadioCapabilityError::InvalidMessageType {
            expected: "UERadioCapabilityInfoIndication",
        });
    };
    let (mut amf, mut ran, mut cap) = (None, None, None);
    for ie in ind.protocol_i_es.0 {
        match ie.value {
            UERadioCapabilityInfoIndicationProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(v) => {
                amf = Some(v.0)
            }
            UERadioCapabilityInfoIndicationProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(v) => {
                ran = Some(v.0)
            }
            UERadioCapabilityInfoIndicationProtocolIEs_EntryValue::Id_UERadioCapability(v) => {
                cap = Some(v.0)
            }
            _ => {}
        }
    }
    Ok(UeRadioCapabilityInfoIndicationData {
        amf_ue_ngap_id: amf.ok_or(UeRadioCapabilityError::MissingMandatoryIe("AMF-UE-NGAP-ID"))?,
        ran_ue_ngap_id: ran.ok_or(UeRadioCapabilityError::MissingMandatoryIe("RAN-UE-NGAP-ID"))?,
        ue_radio_capability: cap.ok_or(UeRadioCapabilityError::MissingMandatoryIe(
            "UERadioCapability",
        ))?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-derived from X.691 APER (TS 38.413 §9.4):
    /// `00` InitiatingMessage (CHOICE index 0, extensible root, no ext bit set)
    /// `3c` procedureCode 60 (INTEGER 0..255, one octet)
    /// `00` criticality reject (ENUMERATED 0..2, 2 bits) + 6 bits padding
    /// `0b` open type length 11
    ///   `00` SEQUENCE extension bit 0 + padding
    ///   `00 01` container count 1 (SIZE 0..65535, two octets)
    ///   `01 08` IE id 264
    ///   `00` criticality reject + padding
    ///   `04` open type length 4
    ///     `03` OCTET STRING length 3, then `12 34 56`
    #[test]
    fn a_mapping_request_matches_hand_derived_octets() {
        let bytes = encode_ue_radio_capability_id_mapping_request(&[0x12, 0x34, 0x56]).unwrap();
        assert_eq!(
            bytes,
            vec![
                0x00, 0x3c, 0x00, 0x0b, 0x00, 0x00, 0x01, 0x01, 0x08, 0x00, 0x04, 0x03, 0x12, 0x34,
                0x56
            ]
        );
        assert_eq!(
            decode_ue_radio_capability_id_mapping_request(&bytes).unwrap(),
            vec![0x12, 0x34, 0x56]
        );
    }

    #[test]
    fn a_mapping_response_round_trips() {
        let data = UeRadioCapabilityIdMappingResponseData {
            ue_radio_capability_id: vec![0x10, 0x20],
            ue_radio_capability: vec![0xAA; 300],
        };
        let bytes = encode_ue_radio_capability_id_mapping_response(&data).unwrap();
        assert_eq!(bytes[0], 0x20, "SuccessfulOutcome");
        assert_eq!(bytes[1], 60, "procedure code 60");
        assert_eq!(
            decode_ue_radio_capability_id_mapping_response(&bytes).unwrap(),
            data
        );
    }

    #[test]
    fn an_info_indication_round_trips_with_procedure_44_ignore() {
        let data = UeRadioCapabilityInfoIndicationData {
            amf_ue_ngap_id: 7,
            ran_ue_ngap_id: 9,
            ue_radio_capability: vec![1, 2, 3],
        };
        let bytes = encode_ue_radio_capability_info_indication(&data).unwrap();
        assert_eq!(
            &bytes[..3],
            &[0x00, 44, 0x40],
            "initiating, code 44, ignore"
        );
        assert_eq!(
            decode_ue_radio_capability_info_indication(&bytes).unwrap(),
            data
        );
    }

    #[test]
    fn a_mapping_request_is_not_a_response() {
        let bytes = encode_ue_radio_capability_id_mapping_request(&[1]).unwrap();
        assert!(decode_ue_radio_capability_id_mapping_response(&bytes).is_err());
    }
}
