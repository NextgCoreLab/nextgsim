//! Downlink/Uplink NAS Transport Procedures
//!
//! Implements the NAS Transport procedures as defined in 3GPP TS 38.413 Section 8.6.
//! - Downlink NAS Transport (Section 8.6.2): AMF to gNB, carries NAS PDU to UE
//! - Uplink NAS Transport (Section 8.6.3): gNB to AMF, carries NAS PDU from UE

use crate::codec::generated::*;
use crate::codec::{decode_ngap_pdu, encode_ngap_pdu, NgapCodecError};
use crate::procedures::initial_ue_message::{
    build_nr_cgi, build_tai, parse_nr_cgi, parse_tai, UserLocationInfoNr,
};
use thiserror::Error;

/// Errors that can occur during NAS Transport procedures
#[derive(Debug, Error)]
pub enum NasTransportError {
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

// ============================================================================
// Downlink NAS Transport
// ============================================================================

/// Parameters for building a Downlink NAS Transport message
#[derive(Debug, Clone)]
pub struct DownlinkNasTransportParams {
    /// AMF UE NGAP ID (allocated by AMF)
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID (allocated by gNB)
    pub ran_ue_ngap_id: u32,
    /// NAS PDU (contains the NAS message for UE)
    pub nas_pdu: Vec<u8>,
    /// Old AMF name (optional, for inter-AMF handover)
    pub old_amf: Option<String>,
    /// RAN Paging Priority (optional, 1-256)
    pub ran_paging_priority: Option<u16>,
    /// Index to RFSP (optional)
    pub index_to_rfsp: Option<u16>,
    /// UE Aggregate Maximum Bit Rate (optional)
    pub ue_ambr: Option<UeAmbrParams>,
    /// Allowed NSSAI (optional)
    pub allowed_nssai: Option<Vec<AllowedSnssaiItem>>,
}

/// UE Aggregate Maximum Bit Rate parameters
#[derive(Debug, Clone)]
pub struct UeAmbrParams {
    /// Downlink bit rate in bits per second
    pub dl_bit_rate: u64,
    /// Uplink bit rate in bits per second
    pub ul_bit_rate: u64,
}

/// Allowed S-NSSAI item for NAS Transport
#[derive(Debug, Clone)]
pub struct AllowedSnssaiItem {
    /// Slice/Service Type (SST) - 1 byte
    pub sst: u8,
    /// Slice Differentiator (SD) - 3 bytes, optional
    pub sd: Option<[u8; 3]>,
}

/// Parsed Downlink NAS Transport data
#[derive(Debug, Clone)]
pub struct DownlinkNasTransportData {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
    /// NAS PDU
    pub nas_pdu: Vec<u8>,
    /// Old AMF name (optional)
    pub old_amf: Option<String>,
    /// RAN Paging Priority (optional, 1-256)
    pub ran_paging_priority: Option<u16>,
    /// Index to RFSP (optional)
    pub index_to_rfsp: Option<u16>,
    /// Allowed NSSAI (optional)
    pub allowed_nssai: Option<Vec<AllowedSnssaiItem>>,
}

/// Build a Downlink NAS Transport PDU
///
/// # Arguments
/// * `params` - Parameters for the Downlink NAS Transport message
///
/// # Returns
/// * `Ok(NGAP_PDU)` - The constructed PDU
/// * `Err(NasTransportError)` - If construction fails
pub fn build_downlink_nas_transport(
    params: &DownlinkNasTransportParams,
) -> Result<NGAP_PDU, NasTransportError> {
    let mut protocol_ies = Vec::new();

    // IE: AMF-UE-NGAP-ID (mandatory)
    protocol_ies.push(DownlinkNASTransportProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_AMF_UE_NGAP_ID),
        criticality: Criticality(Criticality::REJECT),
        value: DownlinkNASTransportProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(AMF_UE_NGAP_ID(
            params.amf_ue_ngap_id,
        )),
    });

    // IE: RAN-UE-NGAP-ID (mandatory)
    protocol_ies.push(DownlinkNASTransportProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_RAN_UE_NGAP_ID),
        criticality: Criticality(Criticality::REJECT),
        value: DownlinkNASTransportProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(RAN_UE_NGAP_ID(
            params.ran_ue_ngap_id,
        )),
    });

    // IE: OldAMF (optional)
    if let Some(ref old_amf) = params.old_amf {
        protocol_ies.push(DownlinkNASTransportProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_OLD_AMF),
            criticality: Criticality(Criticality::REJECT),
            value: DownlinkNASTransportProtocolIEs_EntryValue::Id_OldAMF(AMFName(old_amf.clone())),
        });
    }

    // IE: RANPagingPriority (optional)
    if let Some(priority) = params.ran_paging_priority {
        protocol_ies.push(DownlinkNASTransportProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_RAN_PAGING_PRIORITY),
            criticality: Criticality(Criticality::IGNORE),
            value: DownlinkNASTransportProtocolIEs_EntryValue::Id_RANPagingPriority(
                RANPagingPriority(priority),
            ),
        });
    }

    // IE: NAS-PDU (mandatory)
    protocol_ies.push(DownlinkNASTransportProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_NAS_PDU),
        criticality: Criticality(Criticality::REJECT),
        value: DownlinkNASTransportProtocolIEs_EntryValue::Id_NAS_PDU(NAS_PDU(
            params.nas_pdu.clone(),
        )),
    });

    // IE: IndexToRFSP (optional)
    if let Some(index) = params.index_to_rfsp {
        protocol_ies.push(DownlinkNASTransportProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_INDEX_TO_RFSP),
            criticality: Criticality(Criticality::IGNORE),
            value: DownlinkNASTransportProtocolIEs_EntryValue::Id_IndexToRFSP(IndexToRFSP(index)),
        });
    }

    // IE: UEAggregateMaximumBitRate (optional)
    if let Some(ref ambr) = params.ue_ambr {
        protocol_ies.push(DownlinkNASTransportProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_UE_AGGREGATE_MAXIMUM_BIT_RATE),
            criticality: Criticality(Criticality::IGNORE),
            value: DownlinkNASTransportProtocolIEs_EntryValue::Id_UEAggregateMaximumBitRate(
                UEAggregateMaximumBitRate {
                    ue_aggregate_maximum_bit_rate_dl: BitRate(ambr.dl_bit_rate),
                    ue_aggregate_maximum_bit_rate_ul: BitRate(ambr.ul_bit_rate),
                    ie_extensions: None,
                },
            ),
        });
    }

    // IE: AllowedNSSAI (optional)
    if let Some(ref allowed_nssai) = params.allowed_nssai {
        let items: Vec<AllowedNSSAI_Item> = allowed_nssai
            .iter()
            .map(|snssai| {
                let sd = snssai.sd.map(|sd_bytes| SD(sd_bytes.to_vec()));
                AllowedNSSAI_Item {
                    s_nssai: S_NSSAI {
                        sst: SST(vec![snssai.sst]),
                        sd,
                        ie_extensions: None,
                    },
                    ie_extensions: None,
                }
            })
            .collect();
        protocol_ies.push(DownlinkNASTransportProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_ALLOWED_NSSAI),
            criticality: Criticality(Criticality::REJECT),
            value: DownlinkNASTransportProtocolIEs_EntryValue::Id_AllowedNSSAI(AllowedNSSAI(items)),
        });
    }

    let downlink_nas_transport = DownlinkNASTransport {
        protocol_i_es: DownlinkNASTransportProtocolIEs(protocol_ies),
    };

    let initiating_message = InitiatingMessage {
        procedure_code: ProcedureCode(ID_DOWNLINK_NAS_TRANSPORT),
        criticality: Criticality(Criticality::IGNORE),
        value: InitiatingMessageValue::Id_DownlinkNASTransport(downlink_nas_transport),
    };

    Ok(NGAP_PDU::InitiatingMessage(initiating_message))
}

/// Parse a Downlink NAS Transport message from an NGAP PDU
///
/// # Arguments
/// * `pdu` - The NGAP PDU to parse
///
/// # Returns
/// * `Ok(DownlinkNasTransportData)` - The parsed message data
/// * `Err(NasTransportError)` - If parsing fails
pub fn parse_downlink_nas_transport(
    pdu: &NGAP_PDU,
) -> Result<DownlinkNasTransportData, NasTransportError> {
    let initiating_message = match pdu {
        NGAP_PDU::InitiatingMessage(msg) => msg,
        _ => {
            return Err(NasTransportError::InvalidMessageType {
                expected: "InitiatingMessage".to_string(),
                actual: format!("{pdu:?}"),
            })
        }
    };

    let downlink_nas_transport = match &initiating_message.value {
        InitiatingMessageValue::Id_DownlinkNASTransport(msg) => msg,
        _ => {
            return Err(NasTransportError::InvalidMessageType {
                expected: "DownlinkNASTransport".to_string(),
                actual: format!("{:?}", initiating_message.value),
            })
        }
    };

    let mut amf_ue_ngap_id: Option<u64> = None;
    let mut ran_ue_ngap_id: Option<u32> = None;
    let mut nas_pdu: Option<Vec<u8>> = None;
    let mut old_amf: Option<String> = None;
    let mut ran_paging_priority: Option<u16> = None;
    let mut index_to_rfsp: Option<u16> = None;
    let mut allowed_nssai: Option<Vec<AllowedSnssaiItem>> = None;

    for ie in &downlink_nas_transport.protocol_i_es.0 {
        match &ie.value {
            DownlinkNASTransportProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(id) => {
                amf_ue_ngap_id = Some(id.0);
            }
            DownlinkNASTransportProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(id) => {
                ran_ue_ngap_id = Some(id.0);
            }
            DownlinkNASTransportProtocolIEs_EntryValue::Id_NAS_PDU(pdu) => {
                nas_pdu = Some(pdu.0.clone());
            }
            DownlinkNASTransportProtocolIEs_EntryValue::Id_OldAMF(name) => {
                old_amf = Some(name.0.clone());
            }
            DownlinkNASTransportProtocolIEs_EntryValue::Id_RANPagingPriority(priority) => {
                ran_paging_priority = Some(priority.0);
            }
            DownlinkNASTransportProtocolIEs_EntryValue::Id_IndexToRFSP(index) => {
                index_to_rfsp = Some(index.0);
            }
            DownlinkNASTransportProtocolIEs_EntryValue::Id_AllowedNSSAI(nssai) => {
                allowed_nssai = Some(parse_allowed_nssai(nssai));
            }
            _ => {
                // Ignore other IEs
            }
        }
    }

    Ok(DownlinkNasTransportData {
        amf_ue_ngap_id: amf_ue_ngap_id
            .ok_or_else(|| NasTransportError::MissingMandatoryIe("AMF-UE-NGAP-ID".to_string()))?,
        ran_ue_ngap_id: ran_ue_ngap_id
            .ok_or_else(|| NasTransportError::MissingMandatoryIe("RAN-UE-NGAP-ID".to_string()))?,
        nas_pdu: nas_pdu
            .ok_or_else(|| NasTransportError::MissingMandatoryIe("NAS-PDU".to_string()))?,
        old_amf,
        ran_paging_priority,
        index_to_rfsp,
        allowed_nssai,
    })
}

fn parse_allowed_nssai(nssai: &AllowedNSSAI) -> Vec<AllowedSnssaiItem> {
    nssai
        .0
        .iter()
        .map(|item| {
            let sst = item.s_nssai.sst.0.first().copied().unwrap_or(0);
            let sd = item.s_nssai.sd.as_ref().and_then(|sd| sd.0.as_slice().try_into().ok());
            AllowedSnssaiItem { sst, sd }
        })
        .collect()
}

// ============================================================================
// Uplink NAS Transport
// ============================================================================

/// Parameters for building an Uplink NAS Transport message
#[derive(Debug, Clone)]
pub struct UplinkNasTransportParams {
    /// AMF UE NGAP ID (allocated by AMF)
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID (allocated by gNB)
    pub ran_ue_ngap_id: u32,
    /// NAS PDU (contains the NAS message from UE)
    pub nas_pdu: Vec<u8>,
    /// User Location Information
    pub user_location_info: UserLocationInfoNr,
}

/// Parsed Uplink NAS Transport data
#[derive(Debug, Clone)]
pub struct UplinkNasTransportData {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
    /// NAS PDU
    pub nas_pdu: Vec<u8>,
    /// User Location Information
    pub user_location_info: UserLocationInfoNr,
}

/// Build an Uplink NAS Transport PDU
///
/// # Arguments
/// * `params` - Parameters for the Uplink NAS Transport message
///
/// # Returns
/// * `Ok(NGAP_PDU)` - The constructed PDU
/// * `Err(NasTransportError)` - If construction fails
pub fn build_uplink_nas_transport(
    params: &UplinkNasTransportParams,
) -> Result<NGAP_PDU, NasTransportError> {
    let mut protocol_ies = Vec::new();

    // IE: AMF-UE-NGAP-ID (mandatory)
    protocol_ies.push(UplinkNASTransportProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_AMF_UE_NGAP_ID),
        criticality: Criticality(Criticality::REJECT),
        value: UplinkNASTransportProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(AMF_UE_NGAP_ID(
            params.amf_ue_ngap_id,
        )),
    });

    // IE: RAN-UE-NGAP-ID (mandatory)
    protocol_ies.push(UplinkNASTransportProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_RAN_UE_NGAP_ID),
        criticality: Criticality(Criticality::REJECT),
        value: UplinkNASTransportProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(RAN_UE_NGAP_ID(
            params.ran_ue_ngap_id,
        )),
    });

    // IE: NAS-PDU (mandatory)
    protocol_ies.push(UplinkNASTransportProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_NAS_PDU),
        criticality: Criticality(Criticality::REJECT),
        value: UplinkNASTransportProtocolIEs_EntryValue::Id_NAS_PDU(NAS_PDU(
            params.nas_pdu.clone(),
        )),
    });

    // IE: UserLocationInformation (mandatory)
    let user_location_info = build_user_location_info(&params.user_location_info);
    protocol_ies.push(UplinkNASTransportProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_USER_LOCATION_INFORMATION),
        criticality: Criticality(Criticality::IGNORE),
        value: UplinkNASTransportProtocolIEs_EntryValue::Id_UserLocationInformation(
            user_location_info,
        ),
    });

    let uplink_nas_transport = UplinkNASTransport {
        protocol_i_es: UplinkNASTransportProtocolIEs(protocol_ies),
    };

    let initiating_message = InitiatingMessage {
        procedure_code: ProcedureCode(ID_UPLINK_NAS_TRANSPORT),
        criticality: Criticality(Criticality::IGNORE),
        value: InitiatingMessageValue::Id_UplinkNASTransport(uplink_nas_transport),
    };

    Ok(NGAP_PDU::InitiatingMessage(initiating_message))
}

fn build_user_location_info(info: &UserLocationInfoNr) -> UserLocationInformation {
    let nr_cgi = build_nr_cgi(&info.nr_cgi);
    let tai = build_tai(&info.tai);
    let time_stamp = info.time_stamp.map(|ts| TimeStamp(ts.to_vec()));

    let user_location_info_nr = UserLocationInformationNR {
        nr_cgi,
        tai,
        time_stamp,
        ie_extensions: None,
    };

    UserLocationInformation::UserLocationInformationNR(user_location_info_nr)
}

/// Parse an Uplink NAS Transport message from an NGAP PDU
///
/// # Arguments
/// * `pdu` - The NGAP PDU to parse
///
/// # Returns
/// * `Ok(UplinkNasTransportData)` - The parsed message data
/// * `Err(NasTransportError)` - If parsing fails
pub fn parse_uplink_nas_transport(
    pdu: &NGAP_PDU,
) -> Result<UplinkNasTransportData, NasTransportError> {
    let initiating_message = match pdu {
        NGAP_PDU::InitiatingMessage(msg) => msg,
        _ => {
            return Err(NasTransportError::InvalidMessageType {
                expected: "InitiatingMessage".to_string(),
                actual: format!("{pdu:?}"),
            })
        }
    };

    let uplink_nas_transport = match &initiating_message.value {
        InitiatingMessageValue::Id_UplinkNASTransport(msg) => msg,
        _ => {
            return Err(NasTransportError::InvalidMessageType {
                expected: "UplinkNASTransport".to_string(),
                actual: format!("{:?}", initiating_message.value),
            })
        }
    };

    let mut amf_ue_ngap_id: Option<u64> = None;
    let mut ran_ue_ngap_id: Option<u32> = None;
    let mut nas_pdu: Option<Vec<u8>> = None;
    let mut user_location_info: Option<UserLocationInfoNr> = None;

    for ie in &uplink_nas_transport.protocol_i_es.0 {
        #[allow(unreachable_patterns)]
        match &ie.value {
            UplinkNASTransportProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(id) => {
                amf_ue_ngap_id = Some(id.0);
            }
            UplinkNASTransportProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(id) => {
                ran_ue_ngap_id = Some(id.0);
            }
            UplinkNASTransportProtocolIEs_EntryValue::Id_NAS_PDU(pdu) => {
                nas_pdu = Some(pdu.0.clone());
            }
            UplinkNASTransportProtocolIEs_EntryValue::Id_UserLocationInformation(info) => {
                user_location_info = Some(parse_user_location_info(info)?);
            }
            _ => {
                // Ignore other IEs
            }
        }
    }

    Ok(UplinkNasTransportData {
        amf_ue_ngap_id: amf_ue_ngap_id
            .ok_or_else(|| NasTransportError::MissingMandatoryIe("AMF-UE-NGAP-ID".to_string()))?,
        ran_ue_ngap_id: ran_ue_ngap_id
            .ok_or_else(|| NasTransportError::MissingMandatoryIe("RAN-UE-NGAP-ID".to_string()))?,
        nas_pdu: nas_pdu
            .ok_or_else(|| NasTransportError::MissingMandatoryIe("NAS-PDU".to_string()))?,
        user_location_info: user_location_info.ok_or_else(|| {
            NasTransportError::MissingMandatoryIe("UserLocationInformation".to_string())
        })?,
    })
}

fn parse_user_location_info(
    info: &UserLocationInformation,
) -> Result<UserLocationInfoNr, NasTransportError> {
    match info {
        UserLocationInformation::UserLocationInformationNR(nr_info) => {
            let nr_cgi = parse_nr_cgi(&nr_info.nr_cgi);
            let tai = parse_tai(&nr_info.tai);
            let time_stamp = nr_info
                .time_stamp
                .as_ref()
                .and_then(|ts| ts.0.as_slice().try_into().ok());

            Ok(UserLocationInfoNr {
                nr_cgi,
                tai,
                time_stamp,
            })
        }
        _ => Err(NasTransportError::InvalidIeValue(
            "Expected UserLocationInformationNR".to_string(),
        )),
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Build and encode a Downlink NAS Transport message to bytes
///
/// # Arguments
/// * `params` - Parameters for the Downlink NAS Transport message
///
/// # Returns
/// * `Ok(Vec<u8>)` - The encoded bytes
/// * `Err(NasTransportError)` - If building or encoding fails
pub fn encode_downlink_nas_transport(
    params: &DownlinkNasTransportParams,
) -> Result<Vec<u8>, NasTransportError> {
    let pdu = build_downlink_nas_transport(params)?;
    Ok(encode_ngap_pdu(&pdu)?)
}

/// Decode and parse a Downlink NAS Transport message from bytes
///
/// # Arguments
/// * `bytes` - The encoded bytes
///
/// # Returns
/// * `Ok(DownlinkNasTransportData)` - The parsed message data
/// * `Err(NasTransportError)` - If decoding or parsing fails
pub fn decode_downlink_nas_transport(
    bytes: &[u8],
) -> Result<DownlinkNasTransportData, NasTransportError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_downlink_nas_transport(&pdu)
}

/// Build and encode an Uplink NAS Transport message to bytes
///
/// # Arguments
/// * `params` - Parameters for the Uplink NAS Transport message
///
/// # Returns
/// * `Ok(Vec<u8>)` - The encoded bytes
/// * `Err(NasTransportError)` - If building or encoding fails
pub fn encode_uplink_nas_transport(
    params: &UplinkNasTransportParams,
) -> Result<Vec<u8>, NasTransportError> {
    let pdu = build_uplink_nas_transport(params)?;
    Ok(encode_ngap_pdu(&pdu)?)
}

/// Decode and parse an Uplink NAS Transport message from bytes
///
/// # Arguments
/// * `bytes` - The encoded bytes
///
/// # Returns
/// * `Ok(UplinkNasTransportData)` - The parsed message data
/// * `Err(NasTransportError)` - If decoding or parsing fails
pub fn decode_uplink_nas_transport(
    bytes: &[u8],
) -> Result<UplinkNasTransportData, NasTransportError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_uplink_nas_transport(&pdu)
}

/// Check if an NGAP PDU is a Downlink NAS Transport message
pub fn is_downlink_nas_transport(pdu: &NGAP_PDU) -> bool {
    matches!(
        pdu,
        NGAP_PDU::InitiatingMessage(msg)
            if matches!(msg.value, InitiatingMessageValue::Id_DownlinkNASTransport(_))
    )
}

/// Check if an NGAP PDU is an Uplink NAS Transport message
pub fn is_uplink_nas_transport(pdu: &NGAP_PDU) -> bool {
    matches!(
        pdu,
        NGAP_PDU::InitiatingMessage(msg)
            if matches!(msg.value, InitiatingMessageValue::Id_UplinkNASTransport(_))
    )
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::procedures::initial_ue_message::{NrCgi, Tai};

    fn create_downlink_test_params() -> DownlinkNasTransportParams {
        DownlinkNasTransportParams {
            amf_ue_ngap_id: 1,
            ran_ue_ngap_id: 1,
            nas_pdu: vec![0x7e, 0x00, 0x56, 0x01, 0x02], // Sample NAS PDU
            old_amf: None,
            ran_paging_priority: None,
            index_to_rfsp: None,
            ue_ambr: None,
            allowed_nssai: None,
        }
    }

    fn create_uplink_test_params() -> UplinkNasTransportParams {
        UplinkNasTransportParams {
            amf_ue_ngap_id: 1,
            ran_ue_ngap_id: 1,
            nas_pdu: vec![0x7e, 0x00, 0x67, 0x01, 0x02], // Sample NAS PDU
            user_location_info: UserLocationInfoNr {
                nr_cgi: NrCgi {
                    plmn_identity: [0x00, 0xF1, 0x10], // MCC=001, MNC=01
                    nr_cell_identity: 0x000000001,     // Cell ID = 1
                },
                tai: Tai {
                    plmn_identity: [0x00, 0xF1, 0x10],
                    tac: [0x00, 0x00, 0x01], // TAC = 1
                },
                time_stamp: None,
            },
        }
    }

    #[test]
    fn test_build_downlink_nas_transport() {
        let params = create_downlink_test_params();
        let result = build_downlink_nas_transport(&params);
        assert!(result.is_ok());

        let pdu = result.unwrap();
        match pdu {
            NGAP_PDU::InitiatingMessage(msg) => {
                assert_eq!(msg.procedure_code.0, ID_DOWNLINK_NAS_TRANSPORT);
            }
            _ => panic!("Expected InitiatingMessage"),
        }
    }

    #[test]
    fn test_encode_downlink_nas_transport() {
        let params = create_downlink_test_params();
        let result = encode_downlink_nas_transport(&params);
        assert!(result.is_ok());

        let bytes = result.unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_downlink_nas_transport_roundtrip() {
        let params = create_downlink_test_params();

        // Build and encode
        let pdu = build_downlink_nas_transport(&params).expect("Failed to build message");
        let encoded = encode_ngap_pdu(&pdu).expect("Failed to encode");

        // Decode
        let decoded_pdu = decode_ngap_pdu(&encoded).expect("Failed to decode");

        // Verify structure
        match decoded_pdu {
            NGAP_PDU::InitiatingMessage(msg) => {
                assert_eq!(msg.procedure_code.0, ID_DOWNLINK_NAS_TRANSPORT);
                match msg.value {
                    InitiatingMessageValue::Id_DownlinkNASTransport(dl_msg) => {
                        // Verify we have the expected IEs (3 mandatory)
                        assert!(dl_msg.protocol_i_es.0.len() >= 3);
                    }
                    _ => panic!("Expected DownlinkNASTransport"),
                }
            }
            _ => panic!("Expected InitiatingMessage"),
        }
    }

    #[test]
    fn test_downlink_nas_transport_parse_roundtrip() {
        let params = create_downlink_test_params();

        // Build, encode, decode, and parse
        let encoded = encode_downlink_nas_transport(&params).expect("Failed to encode");
        let parsed = decode_downlink_nas_transport(&encoded).expect("Failed to decode and parse");

        // Verify parsed data matches original params
        assert_eq!(parsed.amf_ue_ngap_id, params.amf_ue_ngap_id);
        assert_eq!(parsed.ran_ue_ngap_id, params.ran_ue_ngap_id);
        assert_eq!(parsed.nas_pdu, params.nas_pdu);
        assert!(parsed.old_amf.is_none());
        assert!(parsed.ran_paging_priority.is_none());
    }

    #[test]
    fn test_downlink_nas_transport_with_optional_ies() {
        let mut params = create_downlink_test_params();
        params.old_amf = Some("old-amf-name".to_string());
        params.ran_paging_priority = Some(5);
        params.index_to_rfsp = Some(10);
        params.allowed_nssai = Some(vec![
            AllowedSnssaiItem { sst: 1, sd: None },
            AllowedSnssaiItem {
                sst: 1,
                sd: Some([0x00, 0x00, 0x01]),
            },
        ]);

        let encoded = encode_downlink_nas_transport(&params).expect("Failed to encode");
        let parsed = decode_downlink_nas_transport(&encoded).expect("Failed to decode and parse");

        assert_eq!(parsed.old_amf, Some("old-amf-name".to_string()));
        assert_eq!(parsed.ran_paging_priority, Some(5));
        assert_eq!(parsed.index_to_rfsp, Some(10));
        assert!(parsed.allowed_nssai.is_some());
        let nssai = parsed.allowed_nssai.unwrap();
        assert_eq!(nssai.len(), 2);
        assert_eq!(nssai[0].sst, 1);
        assert!(nssai[0].sd.is_none());
        assert_eq!(nssai[1].sst, 1);
        assert_eq!(nssai[1].sd, Some([0x00, 0x00, 0x01]));
    }

    #[test]
    fn test_build_uplink_nas_transport() {
        let params = create_uplink_test_params();
        let result = build_uplink_nas_transport(&params);
        assert!(result.is_ok());

        let pdu = result.unwrap();
        match pdu {
            NGAP_PDU::InitiatingMessage(msg) => {
                assert_eq!(msg.procedure_code.0, ID_UPLINK_NAS_TRANSPORT);
            }
            _ => panic!("Expected InitiatingMessage"),
        }
    }

    #[test]
    fn test_encode_uplink_nas_transport() {
        let params = create_uplink_test_params();
        let result = encode_uplink_nas_transport(&params);
        assert!(result.is_ok());

        let bytes = result.unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_uplink_nas_transport_roundtrip() {
        let params = create_uplink_test_params();

        // Build and encode
        let pdu = build_uplink_nas_transport(&params).expect("Failed to build message");
        let encoded = encode_ngap_pdu(&pdu).expect("Failed to encode");

        // Decode
        let decoded_pdu = decode_ngap_pdu(&encoded).expect("Failed to decode");

        // Verify structure
        match decoded_pdu {
            NGAP_PDU::InitiatingMessage(msg) => {
                assert_eq!(msg.procedure_code.0, ID_UPLINK_NAS_TRANSPORT);
                match msg.value {
                    InitiatingMessageValue::Id_UplinkNASTransport(ul_msg) => {
                        // Verify we have the expected IEs (4 mandatory)
                        assert!(ul_msg.protocol_i_es.0.len() >= 4);
                    }
                    _ => panic!("Expected UplinkNASTransport"),
                }
            }
            _ => panic!("Expected InitiatingMessage"),
        }
    }

    #[test]
    fn test_uplink_nas_transport_parse_roundtrip() {
        let params = create_uplink_test_params();

        // Build, encode, decode, and parse
        let encoded = encode_uplink_nas_transport(&params).expect("Failed to encode");
        let parsed = decode_uplink_nas_transport(&encoded).expect("Failed to decode and parse");

        // Verify parsed data matches original params
        assert_eq!(parsed.amf_ue_ngap_id, params.amf_ue_ngap_id);
        assert_eq!(parsed.ran_ue_ngap_id, params.ran_ue_ngap_id);
        assert_eq!(parsed.nas_pdu, params.nas_pdu);
        assert_eq!(
            parsed.user_location_info.nr_cgi.plmn_identity,
            params.user_location_info.nr_cgi.plmn_identity
        );
        assert_eq!(
            parsed.user_location_info.nr_cgi.nr_cell_identity,
            params.user_location_info.nr_cgi.nr_cell_identity
        );
        assert_eq!(
            parsed.user_location_info.tai.plmn_identity,
            params.user_location_info.tai.plmn_identity
        );
        assert_eq!(
            parsed.user_location_info.tai.tac,
            params.user_location_info.tai.tac
        );
    }

    #[test]
    fn test_is_downlink_nas_transport() {
        let params = create_downlink_test_params();
        let pdu = build_downlink_nas_transport(&params).expect("Failed to build message");
        assert!(is_downlink_nas_transport(&pdu));
        assert!(!is_uplink_nas_transport(&pdu));
    }

    #[test]
    fn test_is_uplink_nas_transport() {
        let params = create_uplink_test_params();
        let pdu = build_uplink_nas_transport(&params).expect("Failed to build message");
        assert!(is_uplink_nas_transport(&pdu));
        assert!(!is_downlink_nas_transport(&pdu));
    }

    #[test]
    fn test_downlink_nas_transport_with_ue_ambr() {
        let mut params = create_downlink_test_params();
        params.ue_ambr = Some(UeAmbrParams {
            dl_bit_rate: 1_000_000_000, // 1 Gbps
            ul_bit_rate: 500_000_000,   // 500 Mbps
        });

        let encoded = encode_downlink_nas_transport(&params).expect("Failed to encode");
        // Just verify it encodes successfully - AMBR is not parsed back in our implementation
        assert!(!encoded.is_empty());
    }
}
