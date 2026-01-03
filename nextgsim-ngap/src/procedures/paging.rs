//! Paging Procedure
//!
//! Implements the Paging procedure as defined in 3GPP TS 38.413 Section 8.7.2.
//! This procedure is used by the AMF to request the NG-RAN node to page a UE.
//! The Paging message is sent from AMF to gNB to initiate paging for a UE.

use crate::codec::generated::*;
use crate::codec::{decode_ngap_pdu, encode_ngap_pdu, NgapCodecError};
use crate::procedures::initial_ue_message::{build_tai, parse_tai, FiveGSTmsi, Tai};
use bitvec::prelude::*;
use thiserror::Error;

/// Errors that can occur during Paging procedures
#[derive(Debug, Error)]
pub enum PagingError {
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

/// Paging Priority values as defined in 3GPP TS 38.413
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PagingPriorityValue {
    /// Priority level 1 (highest)
    PrioLevel1,
    /// Priority level 2
    PrioLevel2,
    /// Priority level 3
    PrioLevel3,
    /// Priority level 4
    PrioLevel4,
    /// Priority level 5
    PrioLevel5,
    /// Priority level 6
    PrioLevel6,
    /// Priority level 7
    PrioLevel7,
    /// Priority level 8 (lowest)
    PrioLevel8,
}


impl From<PagingPriorityValue> for PagingPriority {
    fn from(priority: PagingPriorityValue) -> Self {
        let value = match priority {
            PagingPriorityValue::PrioLevel1 => PagingPriority::PRIOLEVEL1,
            PagingPriorityValue::PrioLevel2 => PagingPriority::PRIOLEVEL2,
            PagingPriorityValue::PrioLevel3 => PagingPriority::PRIOLEVEL3,
            PagingPriorityValue::PrioLevel4 => PagingPriority::PRIOLEVEL4,
            PagingPriorityValue::PrioLevel5 => PagingPriority::PRIOLEVEL5,
            PagingPriorityValue::PrioLevel6 => PagingPriority::PRIOLEVEL6,
            PagingPriorityValue::PrioLevel7 => PagingPriority::PRIOLEVEL7,
            PagingPriorityValue::PrioLevel8 => PagingPriority::PRIOLEVEL8,
        };
        PagingPriority(value)
    }
}

impl TryFrom<PagingPriority> for PagingPriorityValue {
    type Error = PagingError;

    fn try_from(priority: PagingPriority) -> Result<Self, Self::Error> {
        match priority.0 {
            PagingPriority::PRIOLEVEL1 => Ok(PagingPriorityValue::PrioLevel1),
            PagingPriority::PRIOLEVEL2 => Ok(PagingPriorityValue::PrioLevel2),
            PagingPriority::PRIOLEVEL3 => Ok(PagingPriorityValue::PrioLevel3),
            PagingPriority::PRIOLEVEL4 => Ok(PagingPriorityValue::PrioLevel4),
            PagingPriority::PRIOLEVEL5 => Ok(PagingPriorityValue::PrioLevel5),
            PagingPriority::PRIOLEVEL6 => Ok(PagingPriorityValue::PrioLevel6),
            PagingPriority::PRIOLEVEL7 => Ok(PagingPriorityValue::PrioLevel7),
            PagingPriority::PRIOLEVEL8 => Ok(PagingPriorityValue::PrioLevel8),
            _ => Err(PagingError::InvalidIeValue(format!(
                "Unknown PagingPriority value: {}",
                priority.0
            ))),
        }
    }
}

/// Paging Origin values as defined in 3GPP TS 38.413
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PagingOriginValue {
    /// Paging originated from non-3GPP access
    Non3gpp,
}

impl From<PagingOriginValue> for PagingOrigin {
    fn from(_: PagingOriginValue) -> Self {
        PagingOrigin(PagingOrigin::NON_3GPP)
    }
}

impl TryFrom<PagingOrigin> for PagingOriginValue {
    type Error = PagingError;

    fn try_from(origin: PagingOrigin) -> Result<Self, Self::Error> {
        match origin.0 {
            PagingOrigin::NON_3GPP => Ok(PagingOriginValue::Non3gpp),
            _ => Err(PagingError::InvalidIeValue(format!(
                "Unknown PagingOrigin value: {}",
                origin.0
            ))),
        }
    }
}

/// Paging DRX values as defined in 3GPP TS 38.413
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PagingDrxValue {
    /// 32 radio frames
    V32,
    /// 64 radio frames
    V64,
    /// 128 radio frames
    V128,
    /// 256 radio frames
    V256,
}

impl From<PagingDrxValue> for PagingDRX {
    fn from(drx: PagingDrxValue) -> Self {
        match drx {
            PagingDrxValue::V32 => PagingDRX(PagingDRX::V32),
            PagingDrxValue::V64 => PagingDRX(PagingDRX::V64),
            PagingDrxValue::V128 => PagingDRX(PagingDRX::V128),
            PagingDrxValue::V256 => PagingDRX(PagingDRX::V256),
        }
    }
}

impl TryFrom<PagingDRX> for PagingDrxValue {
    type Error = PagingError;

    fn try_from(drx: PagingDRX) -> Result<Self, Self::Error> {
        match drx.0 {
            PagingDRX::V32 => Ok(PagingDrxValue::V32),
            PagingDRX::V64 => Ok(PagingDrxValue::V64),
            PagingDRX::V128 => Ok(PagingDrxValue::V128),
            PagingDRX::V256 => Ok(PagingDrxValue::V256),
            _ => Err(PagingError::InvalidIeValue(format!(
                "Unknown PagingDRX value: {}",
                drx.0
            ))),
        }
    }
}


/// UE Paging Identity - identifies the UE to be paged
#[derive(Debug, Clone)]
pub enum UePagingIdentityValue {
    /// 5G-S-TMSI based identity
    FiveGSTmsi(FiveGSTmsi),
}

/// Parameters for building a Paging message
#[derive(Debug, Clone)]
pub struct PagingParams {
    /// UE Paging Identity (mandatory)
    pub ue_paging_identity: UePagingIdentityValue,
    /// TAI List for Paging (mandatory) - list of TAIs where the UE should be paged
    pub tai_list_for_paging: Vec<Tai>,
    /// Paging DRX (optional)
    pub paging_drx: Option<PagingDrxValue>,
    /// Paging Priority (optional)
    pub paging_priority: Option<PagingPriorityValue>,
    /// Paging Origin (optional)
    pub paging_origin: Option<PagingOriginValue>,
}

/// Parsed Paging message data
#[derive(Debug, Clone)]
pub struct PagingData {
    /// UE Paging Identity
    pub ue_paging_identity: UePagingIdentityValue,
    /// TAI List for Paging
    pub tai_list_for_paging: Vec<Tai>,
    /// Paging DRX (optional)
    pub paging_drx: Option<PagingDrxValue>,
    /// Paging Priority (optional)
    pub paging_priority: Option<PagingPriorityValue>,
    /// Paging Origin (optional)
    pub paging_origin: Option<PagingOriginValue>,
}

// ============================================================================
// Paging Message Builder
// ============================================================================

/// Build a Paging PDU
///
/// # Arguments
/// * `params` - Parameters for the Paging message
///
/// # Returns
/// * `Ok(NGAP_PDU)` - The constructed PDU
/// * `Err(PagingError)` - If construction fails
pub fn build_paging(params: &PagingParams) -> Result<NGAP_PDU, PagingError> {
    let mut protocol_ies = Vec::new();

    // IE: UEPagingIdentity (mandatory)
    let ue_paging_identity = build_ue_paging_identity(&params.ue_paging_identity);
    protocol_ies.push(PagingProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_UE_PAGING_IDENTITY),
        criticality: Criticality(Criticality::IGNORE),
        value: PagingProtocolIEs_EntryValue::Id_UEPagingIdentity(ue_paging_identity),
    });

    // IE: PagingDRX (optional)
    if let Some(drx) = params.paging_drx {
        protocol_ies.push(PagingProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_PAGING_DRX),
            criticality: Criticality(Criticality::IGNORE),
            value: PagingProtocolIEs_EntryValue::Id_PagingDRX(drx.into()),
        });
    }

    // IE: TAIListForPaging (mandatory)
    let tai_list = build_tai_list_for_paging(&params.tai_list_for_paging);
    protocol_ies.push(PagingProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_TAI_LIST_FOR_PAGING),
        criticality: Criticality(Criticality::IGNORE),
        value: PagingProtocolIEs_EntryValue::Id_TAIListForPaging(tai_list),
    });

    // IE: PagingPriority (optional)
    if let Some(priority) = params.paging_priority {
        protocol_ies.push(PagingProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_PAGING_PRIORITY),
            criticality: Criticality(Criticality::IGNORE),
            value: PagingProtocolIEs_EntryValue::Id_PagingPriority(priority.into()),
        });
    }

    // IE: PagingOrigin (optional)
    if let Some(origin) = params.paging_origin {
        protocol_ies.push(PagingProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_PAGING_ORIGIN),
            criticality: Criticality(Criticality::IGNORE),
            value: PagingProtocolIEs_EntryValue::Id_PagingOrigin(origin.into()),
        });
    }

    let paging = Paging {
        protocol_i_es: PagingProtocolIEs(protocol_ies),
    };

    let initiating_message = InitiatingMessage {
        procedure_code: ProcedureCode(ID_PAGING),
        criticality: Criticality(Criticality::IGNORE),
        value: InitiatingMessageValue::Id_Paging(paging),
    };

    Ok(NGAP_PDU::InitiatingMessage(initiating_message))
}


fn build_ue_paging_identity(identity: &UePagingIdentityValue) -> UEPagingIdentity {
    match identity {
        UePagingIdentityValue::FiveGSTmsi(tmsi) => {
            let five_g_s_tmsi = build_five_g_s_tmsi(tmsi);
            UEPagingIdentity::FiveG_S_TMSI(five_g_s_tmsi)
        }
    }
}

fn build_five_g_s_tmsi(tmsi: &FiveGSTmsi) -> FiveG_S_TMSI {
    // AMF Set ID is 10 bits
    let mut amf_set_id_bv: BitVec<u8, Msb0> = BitVec::new();
    for i in (0..10).rev() {
        amf_set_id_bv.push((tmsi.amf_set_id >> i) & 1 == 1);
    }

    // AMF Pointer is 6 bits
    let mut amf_pointer_bv: BitVec<u8, Msb0> = BitVec::new();
    for i in (0..6).rev() {
        amf_pointer_bv.push((tmsi.amf_pointer >> i) & 1 == 1);
    }

    FiveG_S_TMSI {
        amf_set_id: AMFSetID(amf_set_id_bv),
        amf_pointer: AMFPointer(amf_pointer_bv),
        five_g_tmsi: FiveG_TMSI(tmsi.five_g_tmsi.to_vec()),
        ie_extensions: None,
    }
}

fn build_tai_list_for_paging(tai_list: &[Tai]) -> TAIListForPaging {
    let items: Vec<TAIListForPagingItem> = tai_list
        .iter()
        .map(|tai| TAIListForPagingItem {
            tai: build_tai(tai),
            ie_extensions: None,
        })
        .collect();

    TAIListForPaging(items)
}

// ============================================================================
// Paging Message Parser
// ============================================================================

/// Parse a Paging message from an NGAP PDU
///
/// # Arguments
/// * `pdu` - The NGAP PDU to parse
///
/// # Returns
/// * `Ok(PagingData)` - The parsed message data
/// * `Err(PagingError)` - If parsing fails
pub fn parse_paging(pdu: &NGAP_PDU) -> Result<PagingData, PagingError> {
    let initiating_message = match pdu {
        NGAP_PDU::InitiatingMessage(msg) => msg,
        _ => {
            return Err(PagingError::InvalidMessageType {
                expected: "InitiatingMessage".to_string(),
                actual: format!("{:?}", pdu),
            })
        }
    };

    let paging = match &initiating_message.value {
        InitiatingMessageValue::Id_Paging(msg) => msg,
        _ => {
            return Err(PagingError::InvalidMessageType {
                expected: "Paging".to_string(),
                actual: format!("{:?}", initiating_message.value),
            })
        }
    };

    let mut ue_paging_identity: Option<UePagingIdentityValue> = None;
    let mut tai_list_for_paging: Option<Vec<Tai>> = None;
    let mut paging_drx: Option<PagingDrxValue> = None;
    let mut paging_priority: Option<PagingPriorityValue> = None;
    let mut paging_origin: Option<PagingOriginValue> = None;

    for ie in &paging.protocol_i_es.0 {
        match &ie.value {
            PagingProtocolIEs_EntryValue::Id_UEPagingIdentity(identity) => {
                ue_paging_identity = Some(parse_ue_paging_identity(identity)?);
            }
            PagingProtocolIEs_EntryValue::Id_TAIListForPaging(list) => {
                tai_list_for_paging = Some(parse_tai_list_for_paging(list));
            }
            PagingProtocolIEs_EntryValue::Id_PagingDRX(drx) => {
                paging_drx = Some(drx.clone().try_into()?);
            }
            PagingProtocolIEs_EntryValue::Id_PagingPriority(priority) => {
                paging_priority = Some(priority.clone().try_into()?);
            }
            PagingProtocolIEs_EntryValue::Id_PagingOrigin(origin) => {
                paging_origin = Some(origin.clone().try_into()?);
            }
            _ => {
                // Ignore other IEs
            }
        }
    }

    Ok(PagingData {
        ue_paging_identity: ue_paging_identity
            .ok_or_else(|| PagingError::MissingMandatoryIe("UEPagingIdentity".to_string()))?,
        tai_list_for_paging: tai_list_for_paging
            .ok_or_else(|| PagingError::MissingMandatoryIe("TAIListForPaging".to_string()))?,
        paging_drx,
        paging_priority,
        paging_origin,
    })
}


fn parse_ue_paging_identity(identity: &UEPagingIdentity) -> Result<UePagingIdentityValue, PagingError> {
    match identity {
        UEPagingIdentity::FiveG_S_TMSI(tmsi) => {
            Ok(UePagingIdentityValue::FiveGSTmsi(parse_five_g_s_tmsi(tmsi)))
        }
        _ => Err(PagingError::InvalidIeValue(
            "Unsupported UEPagingIdentity type".to_string(),
        )),
    }
}

fn parse_five_g_s_tmsi(tmsi: &FiveG_S_TMSI) -> FiveGSTmsi {
    // AMF Set ID is 10 bits
    let amf_set_id = if tmsi.amf_set_id.0.len() >= 10 {
        let mut value: u16 = 0;
        for (i, bit) in tmsi.amf_set_id.0.iter().take(10).enumerate() {
            if *bit {
                value |= 1 << (9 - i);
            }
        }
        value
    } else {
        0
    };

    // AMF Pointer is 6 bits
    let amf_pointer = if tmsi.amf_pointer.0.len() >= 6 {
        let mut value: u8 = 0;
        for (i, bit) in tmsi.amf_pointer.0.iter().take(6).enumerate() {
            if *bit {
                value |= 1 << (5 - i);
            }
        }
        value
    } else {
        0
    };

    let five_g_tmsi: [u8; 4] = tmsi
        .five_g_tmsi
        .0
        .as_slice()
        .try_into()
        .unwrap_or([0, 0, 0, 0]);

    FiveGSTmsi {
        amf_set_id,
        amf_pointer,
        five_g_tmsi,
    }
}

fn parse_tai_list_for_paging(list: &TAIListForPaging) -> Vec<Tai> {
    list.0
        .iter()
        .map(|item| parse_tai(&item.tai))
        .collect()
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Build and encode a Paging message to bytes
///
/// # Arguments
/// * `params` - Parameters for the Paging message
///
/// # Returns
/// * `Ok(Vec<u8>)` - The encoded bytes
/// * `Err(PagingError)` - If building or encoding fails
pub fn encode_paging(params: &PagingParams) -> Result<Vec<u8>, PagingError> {
    let pdu = build_paging(params)?;
    Ok(encode_ngap_pdu(&pdu)?)
}

/// Decode and parse a Paging message from bytes
///
/// # Arguments
/// * `bytes` - The encoded bytes
///
/// # Returns
/// * `Ok(PagingData)` - The parsed message data
/// * `Err(PagingError)` - If decoding or parsing fails
pub fn decode_paging(bytes: &[u8]) -> Result<PagingData, PagingError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_paging(&pdu)
}

/// Check if an NGAP PDU is a Paging message
pub fn is_paging(pdu: &NGAP_PDU) -> bool {
    matches!(
        pdu,
        NGAP_PDU::InitiatingMessage(msg)
            if matches!(msg.value, InitiatingMessageValue::Id_Paging(_))
    )
}


#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_params() -> PagingParams {
        PagingParams {
            ue_paging_identity: UePagingIdentityValue::FiveGSTmsi(FiveGSTmsi {
                amf_set_id: 1,
                amf_pointer: 0,
                five_g_tmsi: [0x00, 0x00, 0x00, 0x01],
            }),
            tai_list_for_paging: vec![Tai {
                plmn_identity: [0x00, 0xF1, 0x10], // MCC=001, MNC=01
                tac: [0x00, 0x00, 0x01],           // TAC = 1
            }],
            paging_drx: None,
            paging_priority: None,
            paging_origin: None,
        }
    }

    #[test]
    fn test_build_paging() {
        let params = create_test_params();
        let result = build_paging(&params);
        assert!(result.is_ok());

        let pdu = result.unwrap();
        match pdu {
            NGAP_PDU::InitiatingMessage(msg) => {
                assert_eq!(msg.procedure_code.0, ID_PAGING);
            }
            _ => panic!("Expected InitiatingMessage"),
        }
    }

    #[test]
    fn test_encode_paging() {
        let params = create_test_params();
        let result = encode_paging(&params);
        assert!(result.is_ok());

        let bytes = result.unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_paging_roundtrip() {
        let params = create_test_params();

        // Build and encode
        let pdu = build_paging(&params).expect("Failed to build message");
        let encoded = encode_ngap_pdu(&pdu).expect("Failed to encode");

        // Decode
        let decoded_pdu = decode_ngap_pdu(&encoded).expect("Failed to decode");

        // Verify structure
        match decoded_pdu {
            NGAP_PDU::InitiatingMessage(msg) => {
                assert_eq!(msg.procedure_code.0, ID_PAGING);
                match msg.value {
                    InitiatingMessageValue::Id_Paging(paging_msg) => {
                        // Verify we have the expected IEs (at least 2 mandatory)
                        assert!(paging_msg.protocol_i_es.0.len() >= 2);
                    }
                    _ => panic!("Expected Paging"),
                }
            }
            _ => panic!("Expected InitiatingMessage"),
        }
    }

    #[test]
    fn test_paging_parse_roundtrip() {
        let params = create_test_params();

        // Build, encode, decode, and parse
        let encoded = encode_paging(&params).expect("Failed to encode");
        let parsed = decode_paging(&encoded).expect("Failed to decode and parse");

        // Verify parsed data matches original params
        match (&parsed.ue_paging_identity, &params.ue_paging_identity) {
            (
                UePagingIdentityValue::FiveGSTmsi(parsed_tmsi),
                UePagingIdentityValue::FiveGSTmsi(params_tmsi),
            ) => {
                assert_eq!(parsed_tmsi.amf_set_id, params_tmsi.amf_set_id);
                assert_eq!(parsed_tmsi.amf_pointer, params_tmsi.amf_pointer);
                assert_eq!(parsed_tmsi.five_g_tmsi, params_tmsi.five_g_tmsi);
            }
        }

        assert_eq!(parsed.tai_list_for_paging.len(), params.tai_list_for_paging.len());
        assert_eq!(
            parsed.tai_list_for_paging[0].plmn_identity,
            params.tai_list_for_paging[0].plmn_identity
        );
        assert_eq!(
            parsed.tai_list_for_paging[0].tac,
            params.tai_list_for_paging[0].tac
        );
    }

    #[test]
    fn test_paging_with_optional_ies() {
        let mut params = create_test_params();
        params.paging_drx = Some(PagingDrxValue::V128);
        params.paging_priority = Some(PagingPriorityValue::PrioLevel1);
        params.paging_origin = Some(PagingOriginValue::Non3gpp);

        let encoded = encode_paging(&params).expect("Failed to encode");
        let parsed = decode_paging(&encoded).expect("Failed to decode and parse");

        assert_eq!(parsed.paging_drx, Some(PagingDrxValue::V128));
        assert_eq!(parsed.paging_priority, Some(PagingPriorityValue::PrioLevel1));
        assert_eq!(parsed.paging_origin, Some(PagingOriginValue::Non3gpp));
    }

    #[test]
    fn test_paging_with_multiple_tais() {
        let mut params = create_test_params();
        params.tai_list_for_paging = vec![
            Tai {
                plmn_identity: [0x00, 0xF1, 0x10],
                tac: [0x00, 0x00, 0x01],
            },
            Tai {
                plmn_identity: [0x00, 0xF1, 0x10],
                tac: [0x00, 0x00, 0x02],
            },
            Tai {
                plmn_identity: [0x00, 0xF1, 0x10],
                tac: [0x00, 0x00, 0x03],
            },
        ];

        let encoded = encode_paging(&params).expect("Failed to encode");
        let parsed = decode_paging(&encoded).expect("Failed to decode and parse");

        assert_eq!(parsed.tai_list_for_paging.len(), 3);
        assert_eq!(parsed.tai_list_for_paging[0].tac, [0x00, 0x00, 0x01]);
        assert_eq!(parsed.tai_list_for_paging[1].tac, [0x00, 0x00, 0x02]);
        assert_eq!(parsed.tai_list_for_paging[2].tac, [0x00, 0x00, 0x03]);
    }

    #[test]
    fn test_paging_priority_conversion() {
        // Test all priority values
        let priorities = [
            PagingPriorityValue::PrioLevel1,
            PagingPriorityValue::PrioLevel2,
            PagingPriorityValue::PrioLevel3,
            PagingPriorityValue::PrioLevel4,
            PagingPriorityValue::PrioLevel5,
            PagingPriorityValue::PrioLevel6,
            PagingPriorityValue::PrioLevel7,
            PagingPriorityValue::PrioLevel8,
        ];

        for priority in priorities {
            let paging_priority: PagingPriority = priority.into();
            let converted_back: PagingPriorityValue = paging_priority.try_into().unwrap();
            assert_eq!(priority, converted_back);
        }
    }

    #[test]
    fn test_paging_drx_conversion() {
        // Test all DRX values
        let drx_values = [
            PagingDrxValue::V32,
            PagingDrxValue::V64,
            PagingDrxValue::V128,
            PagingDrxValue::V256,
        ];

        for drx in drx_values {
            let paging_drx: PagingDRX = drx.into();
            let converted_back: PagingDrxValue = paging_drx.try_into().unwrap();
            assert_eq!(drx, converted_back);
        }
    }

    #[test]
    fn test_is_paging() {
        let params = create_test_params();
        let pdu = build_paging(&params).expect("Failed to build message");
        assert!(is_paging(&pdu));
    }
}
