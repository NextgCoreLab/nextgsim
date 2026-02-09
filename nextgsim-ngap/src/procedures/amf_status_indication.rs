//! AMF Status Indication Procedure
//!
//! Implements the AMF Status Indication procedure as defined in 3GPP TS 38.413 Section 8.7.6.
//! This is a Class 2 (no response) procedure sent by an AMF to notify other AMFs
//! or NG-RAN nodes of status changes, containing the UnavailableGUAMIList IE.

use crate::codec::generated::*;
use crate::codec::{decode_ngap_pdu, encode_ngap_pdu, NgapCodecError};
use crate::procedures::initial_context_setup::GuamiValue;
use bitvec::prelude::*;
use thiserror::Error;

/// Errors that can occur during AMF Status Indication procedures
#[derive(Debug, Error)]
pub enum AmfStatusIndicationError {
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

/// Unavailable GUAMI item
#[derive(Debug, Clone)]
pub struct UnavailableGuamiItem {
    /// GUAMI value
    pub guami: GuamiValue,
    /// Backup AMF name (optional)
    pub backup_amf_name: Option<String>,
}

/// Parameters for building an AMF Status Indication message
#[derive(Debug, Clone)]
pub struct AmfStatusIndicationParams {
    /// List of unavailable GUAMIs
    pub unavailable_guami_list: Vec<UnavailableGuamiItem>,
}

/// Parsed AMF Status Indication data
#[derive(Debug, Clone)]
pub struct AmfStatusIndicationData {
    /// List of unavailable GUAMIs
    pub unavailable_guami_list: Vec<UnavailableGuamiItem>,
}

// ============================================================================
// AMF Status Indication Builder
// ============================================================================

/// Build an AMF Status Indication PDU
pub fn build_amf_status_indication(
    params: &AmfStatusIndicationParams,
) -> Result<NGAP_PDU, AmfStatusIndicationError> {
    let mut protocol_ies = Vec::new();

    // IE: UnavailableGUAMIList (mandatory)
    let unavailable_guami_list = build_unavailable_guami_list(&params.unavailable_guami_list);
    protocol_ies.push(AMFStatusIndicationProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_UNAVAILABLE_GUAMI_LIST),
        criticality: Criticality(Criticality::REJECT),
        value: AMFStatusIndicationProtocolIEs_EntryValue::Id_UnavailableGUAMIList(
            unavailable_guami_list,
        ),
    });

    let amf_status_indication = AMFStatusIndication {
        protocol_i_es: AMFStatusIndicationProtocolIEs(protocol_ies),
    };

    let initiating_message = InitiatingMessage {
        procedure_code: ProcedureCode(ID_AMF_STATUS_INDICATION),
        criticality: Criticality(Criticality::IGNORE),
        value: InitiatingMessageValue::Id_AMFStatusIndication(amf_status_indication),
    };

    Ok(NGAP_PDU::InitiatingMessage(initiating_message))
}

fn build_unavailable_guami_list(
    items: &[UnavailableGuamiItem],
) -> UnavailableGUAMIList {
    let list: Vec<UnavailableGUAMIItem> = items
        .iter()
        .map(|item| {
            let guami = build_guami(&item.guami);
            let backup_amf_name = item.backup_amf_name.as_ref().map(|name| AMFName(name.clone()));

            UnavailableGUAMIItem {
                guami,
                timer_approach_for_guami_removal: None,
                backup_amf_name,
                ie_extensions: None,
            }
        })
        .collect();
    UnavailableGUAMIList(list)
}

fn build_guami(guami: &GuamiValue) -> GUAMI {
    // Build AMF Region ID (8 bits)
    let mut amf_region_id_bv: BitVec<u8, Msb0> = BitVec::with_capacity(8);
    for i in (0..8).rev() {
        amf_region_id_bv.push((guami.amf_region_id >> i) & 1 == 1);
    }

    // Build AMF Set ID (10 bits)
    let mut amf_set_id_bv: BitVec<u8, Msb0> = BitVec::with_capacity(10);
    for i in (0..10).rev() {
        amf_set_id_bv.push((guami.amf_set_id >> i) & 1 == 1);
    }

    // Build AMF Pointer (6 bits)
    let mut amf_pointer_bv: BitVec<u8, Msb0> = BitVec::with_capacity(6);
    for i in (0..6).rev() {
        amf_pointer_bv.push((guami.amf_pointer >> i) & 1 == 1);
    }

    GUAMI {
        plmn_identity: PLMNIdentity(guami.plmn_identity.to_vec()),
        amf_region_id: AMFRegionID(amf_region_id_bv),
        amf_set_id: AMFSetID(amf_set_id_bv),
        amf_pointer: AMFPointer(amf_pointer_bv),
        ie_extensions: None,
    }
}

/// Parse an AMF Status Indication from an NGAP PDU
pub fn parse_amf_status_indication(
    pdu: &NGAP_PDU,
) -> Result<AmfStatusIndicationData, AmfStatusIndicationError> {
    let initiating_message = match pdu {
        NGAP_PDU::InitiatingMessage(msg) => msg,
        _ => {
            return Err(AmfStatusIndicationError::InvalidMessageType {
                expected: "InitiatingMessage".to_string(),
                actual: format!("{pdu:?}"),
            })
        }
    };

    let amf_status_indication = match &initiating_message.value {
        InitiatingMessageValue::Id_AMFStatusIndication(msg) => msg,
        _ => {
            return Err(AmfStatusIndicationError::InvalidMessageType {
                expected: "AMFStatusIndication".to_string(),
                actual: format!("{:?}", initiating_message.value),
            })
        }
    };

    let mut unavailable_guami_list: Option<Vec<UnavailableGuamiItem>> = None;

    for ie in &amf_status_indication.protocol_i_es.0 {
        #[allow(unreachable_patterns)]
        if let AMFStatusIndicationProtocolIEs_EntryValue::Id_UnavailableGUAMIList(list) = &ie.value {
            unavailable_guami_list = Some(parse_unavailable_guami_list(list));
        }
    }

    Ok(AmfStatusIndicationData {
        unavailable_guami_list: unavailable_guami_list.ok_or_else(|| {
            AmfStatusIndicationError::MissingMandatoryIe("UnavailableGUAMIList".to_string())
        })?,
    })
}

fn parse_unavailable_guami_list(
    list: &UnavailableGUAMIList,
) -> Vec<UnavailableGuamiItem> {
    list.0
        .iter()
        .map(|item| {
            let guami = parse_guami(&item.guami);
            let backup_amf_name = item.backup_amf_name.as_ref().map(|name| name.0.clone());
            UnavailableGuamiItem {
                guami,
                backup_amf_name,
            }
        })
        .collect()
}

fn parse_guami(guami: &GUAMI) -> GuamiValue {
    let plmn_identity: [u8; 3] = guami
        .plmn_identity
        .0
        .as_slice()
        .try_into()
        .unwrap_or([0, 0, 0]);

    let amf_region_id = if !guami.amf_region_id.0.is_empty() {
        guami.amf_region_id.0.as_raw_slice().first().copied().unwrap_or(0)
    } else {
        0
    };

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

    GuamiValue {
        plmn_identity,
        amf_region_id,
        amf_set_id,
        amf_pointer,
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Build and encode an AMF Status Indication to bytes
pub fn encode_amf_status_indication(
    params: &AmfStatusIndicationParams,
) -> Result<Vec<u8>, AmfStatusIndicationError> {
    let pdu = build_amf_status_indication(params)?;
    Ok(encode_ngap_pdu(&pdu)?)
}

/// Decode and parse an AMF Status Indication from bytes
pub fn decode_amf_status_indication(
    bytes: &[u8],
) -> Result<AmfStatusIndicationData, AmfStatusIndicationError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_amf_status_indication(&pdu)
}

/// Check if an NGAP PDU is an AMF Status Indication
pub fn is_amf_status_indication(pdu: &NGAP_PDU) -> bool {
    matches!(
        pdu,
        NGAP_PDU::InitiatingMessage(msg)
            if matches!(msg.value, InitiatingMessageValue::Id_AMFStatusIndication(_))
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_params() -> AmfStatusIndicationParams {
        AmfStatusIndicationParams {
            unavailable_guami_list: vec![UnavailableGuamiItem {
                guami: GuamiValue {
                    plmn_identity: [0x00, 0xF1, 0x10],
                    amf_region_id: 1,
                    amf_set_id: 1,
                    amf_pointer: 0,
                },
                backup_amf_name: None,
            }],
        }
    }

    #[test]
    fn test_build_amf_status_indication() {
        let params = create_test_params();
        let result = build_amf_status_indication(&params);
        assert!(result.is_ok());

        let pdu = result.unwrap();
        assert!(is_amf_status_indication(&pdu));
    }

    #[test]
    fn test_encode_amf_status_indication() {
        let params = create_test_params();
        let result = encode_amf_status_indication(&params);
        assert!(result.is_ok());

        let bytes = result.unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_amf_status_indication_roundtrip() {
        let params = create_test_params();

        let pdu = build_amf_status_indication(&params).expect("Failed to build");
        let encoded = encode_ngap_pdu(&pdu).expect("Failed to encode");
        let decoded_pdu = decode_ngap_pdu(&encoded).expect("Failed to decode");
        let parsed = parse_amf_status_indication(&decoded_pdu).expect("Failed to parse");

        assert_eq!(parsed.unavailable_guami_list.len(), 1);
        assert_eq!(
            parsed.unavailable_guami_list[0].guami.plmn_identity,
            [0x00, 0xF1, 0x10]
        );
    }

    #[test]
    fn test_amf_status_indication_with_backup_amf() {
        let params = AmfStatusIndicationParams {
            unavailable_guami_list: vec![UnavailableGuamiItem {
                guami: GuamiValue {
                    plmn_identity: [0x00, 0xF1, 0x10],
                    amf_region_id: 2,
                    amf_set_id: 3,
                    amf_pointer: 1,
                },
                backup_amf_name: Some("backup-amf-1".to_string()),
            }],
        };

        let encoded = encode_amf_status_indication(&params).expect("Failed to encode");
        let parsed = decode_amf_status_indication(&encoded).expect("Failed to decode");

        assert_eq!(parsed.unavailable_guami_list.len(), 1);
        assert_eq!(
            parsed.unavailable_guami_list[0].backup_amf_name,
            Some("backup-amf-1".to_string())
        );
    }

    #[test]
    fn test_amf_status_indication_multiple_guamis() {
        let params = AmfStatusIndicationParams {
            unavailable_guami_list: vec![
                UnavailableGuamiItem {
                    guami: GuamiValue {
                        plmn_identity: [0x00, 0xF1, 0x10],
                        amf_region_id: 1,
                        amf_set_id: 1,
                        amf_pointer: 0,
                    },
                    backup_amf_name: None,
                },
                UnavailableGuamiItem {
                    guami: GuamiValue {
                        plmn_identity: [0x00, 0xF1, 0x10],
                        amf_region_id: 2,
                        amf_set_id: 2,
                        amf_pointer: 1,
                    },
                    backup_amf_name: None,
                },
            ],
        };

        let encoded = encode_amf_status_indication(&params).expect("Failed to encode");
        let parsed = decode_amf_status_indication(&encoded).expect("Failed to decode");

        assert_eq!(parsed.unavailable_guami_list.len(), 2);
    }
}
