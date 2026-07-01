//! RRC Reconfiguration Procedure
//!
//! Implements the RRC Reconfiguration procedure as defined in 3GPP TS 38.331 Section 5.3.5.
//! This procedure is used to modify an RRC connection, including radio bearer configuration,
//! measurement configuration, and cell group configuration.
//!
//! The procedure consists of two messages:
//! 1. `RRCReconfiguration` - gNB → UE: Network request to modify RRC connection
//! 2. `RRCReconfigurationComplete` - UE → gNB: Confirmation of reconfiguration

use crate::codec::generated::*;
use crate::codec::{decode_rrc, encode_rrc, RrcCodecError};
use thiserror::Error;

/// Errors that can occur during RRC Reconfiguration procedures
#[derive(Debug, Error)]
pub enum RrcReconfigurationError {
    /// Codec error during encoding/decoding
    #[error("Codec error: {0}")]
    CodecError(#[from] RrcCodecError),

    /// Invalid message type received
    #[error("Invalid message type: expected {expected}, got {actual}")]
    InvalidMessageType {
        /// Expected message type
        expected: String,
        /// Actual message type received
        actual: String,
    },

    /// Invalid field value
    #[error("Invalid field value: {0}")]
    InvalidFieldValue(String),
}

// ============================================================================
// RRC Reconfiguration
// ============================================================================

/// Parameters for building an RRC Reconfiguration message
#[derive(Debug, Clone)]
pub struct RrcReconfigurationParams {
    /// RRC Transaction Identifier (0-3)
    pub rrc_transaction_id: u8,
    /// Radio Bearer Configuration (encoded as bytes, optional)
    pub radio_bearer_config: Option<Vec<u8>>,
    /// Secondary Cell Group Configuration (encoded as bytes, optional)
    pub secondary_cell_group: Option<Vec<u8>>,
    /// Master Cell Group Configuration (encoded as bytes, optional)
    pub master_cell_group: Option<Vec<u8>>,
    /// Full configuration indicator
    pub full_config: bool,
}

/// Parsed RRC Reconfiguration data
#[derive(Debug, Clone)]
pub struct RrcReconfigurationData {
    /// RRC Transaction Identifier
    pub rrc_transaction_id: u8,
    /// Radio Bearer Configuration (raw bytes)
    pub radio_bearer_config: Option<Vec<u8>>,
    /// Secondary Cell Group Configuration (raw bytes)
    pub secondary_cell_group: Option<Vec<u8>>,
    /// Master Cell Group Configuration (raw bytes)
    pub master_cell_group: Option<Vec<u8>>,
    /// Full configuration indicator
    pub full_config: bool,
}

/// Build an RRC Reconfiguration message
pub fn build_rrc_reconfiguration(
    params: &RrcReconfigurationParams,
) -> Result<DL_DCCH_Message, RrcReconfigurationError> {
    if params.rrc_transaction_id > 3 {
        return Err(RrcReconfigurationError::InvalidFieldValue(
            "RRC Transaction ID must be 0-3".to_string(),
        ));
    }

    // Decode radio bearer config if provided
    let radio_bearer_config = if let Some(ref bytes) = params.radio_bearer_config {
        Some(decode_rrc::<RadioBearerConfig>(bytes)?)
    } else {
        None
    };

    // Build the IEs
    let rrc_reconfiguration_ies = RRCReconfiguration_IEs {
        radio_bearer_config,
        secondary_cell_group: params
            .secondary_cell_group
            .as_ref()
            .map(|b| RRCReconfiguration_IEsSecondaryCellGroup(b.clone())),
        meas_config: None, // Simplified - not including MeasConfig for now
        late_non_critical_extension: None,
        non_critical_extension: if params.master_cell_group.is_some() || params.full_config {
            Some(build_v1530_extension(params))
        } else {
            None
        },
    };

    let rrc_reconfiguration = RRCReconfiguration {
        rrc_transaction_identifier: RRC_TransactionIdentifier(params.rrc_transaction_id),
        critical_extensions: RRCReconfigurationCriticalExtensions::RrcReconfiguration(
            rrc_reconfiguration_ies,
        ),
    };

    let message_type = DL_DCCH_MessageType::C1(DL_DCCH_MessageType_c1::RrcReconfiguration(
        rrc_reconfiguration,
    ));

    Ok(DL_DCCH_Message {
        message: message_type,
    })
}

/// Build the v1530 extension for RRC Reconfiguration
fn build_v1530_extension(params: &RrcReconfigurationParams) -> RRCReconfiguration_v1530_IEs {
    RRCReconfiguration_v1530_IEs {
        master_cell_group: params
            .master_cell_group
            .as_ref()
            .map(|b| RRCReconfiguration_v1530_IEsMasterCellGroup(b.clone())),
        full_config: if params.full_config {
            Some(RRCReconfiguration_v1530_IEsFullConfig(
                RRCReconfiguration_v1530_IEsFullConfig::TRUE,
            ))
        } else {
            None
        },
        dedicated_nas_message_list: None,
        master_key_update: None,
        dedicated_sib1_delivery: None,
        dedicated_system_information_delivery: None,
        other_config: None,
        non_critical_extension: None,
    }
}

// ============================================================================
// amfg-04: structured RadioBearerConfig / DRB / SDAP / CellGroupConfig builders
// ============================================================================
//
// TS 38.331 §5.3.5.6.5/§6.3.2: an RRCReconfiguration that establishes a PDU
// session's user plane carries a RadioBearerConfig with one DRB-ToAddMod per
// PDU session; the DRB's cn-Association is an SDAP-Config keyed to the PDU
// session id with the accepted QoS flows (QFIs) mapped via
// mappedQoS-FlowsToAdd, plus a matching CellGroupConfig (one RLC bearer per
// DRB). Previously these were passed as opaque pre-encoded bytes and only the
// first QoS flow was honoured; these constructors build the real structured
// config from the full accepted-QFI set.

/// Build a `RadioBearerConfig` carrying a single DRB for one PDU session.
///
/// * `pdu_session_id` — the 5GS PDU session id the DRB serves.
/// * `drb_id` — the data radio bearer identity (1..=32).
/// * `qfis` — every accepted QoS Flow Identifier to map in SDAP
///   (`mappedQoS-FlowsToAdd`); when empty the field is omitted (the ASN.1
///   SEQUENCE-OF has a lower bound of 1).
/// * `default_drb` — whether this DRB is the SDAP default DRB for the session.
///
/// The SDAP header for DL and UL is set to PRESENT (3-byte SDAP header), which
/// is required when QoS flow remapping/QFI carriage is in use.
pub fn build_drb_radio_bearer_config(
    pdu_session_id: u8,
    drb_id: u8,
    qfis: &[u8],
    default_drb: bool,
) -> RadioBearerConfig {
    let mapped_qo_s_flows_to_add = if qfis.is_empty() {
        None
    } else {
        Some(SDAP_ConfigMappedQoS_FlowsToAdd(
            qfis.iter().map(|q| QFI(*q)).collect(),
        ))
    };

    let sdap_config = SDAP_Config {
        pdu_session: PDU_SessionID(pdu_session_id),
        sdap_header_dl: SDAP_ConfigSdap_HeaderDL(SDAP_ConfigSdap_HeaderDL::PRESENT),
        sdap_header_ul: SDAP_ConfigSdap_HeaderUL(SDAP_ConfigSdap_HeaderUL::PRESENT),
        default_drb: SDAP_ConfigDefaultDRB(default_drb),
        mapped_qo_s_flows_to_add,
        mapped_qo_s_flows_to_release: None,
    };

    // Minimal but valid PDCP-Config (all optional sub-fields absent).
    let pdcp_config = PDCP_Config {
        drb: None,
        more_than_one_rlc: None,
        t_reordering: None,
    };

    let drb = DRB_ToAddMod {
        cn_association: Some(DRB_ToAddModCnAssociation::Sdap_Config(sdap_config)),
        drb_identity: DRB_Identity(drb_id),
        reestablish_pdcp: None,
        recover_pdcp: None,
        pdcp_config: Some(pdcp_config),
    };

    RadioBearerConfig {
        srb_to_add_mod_list: None,
        srb3_to_release: None,
        drb_to_add_mod_list: Some(DRB_ToAddModList(vec![drb])),
        drb_to_release_list: None,
        security_config: None,
    }
}

/// Build a `CellGroupConfig` (master cell group) with one RLC bearer for the
/// given DRB. `lcid` is the logical channel identity carrying the DRB.
pub fn build_cell_group_config(drb_id: u8, lcid: u8) -> CellGroupConfig {
    let rlc_bearer = RLC_BearerConfig {
        logical_channel_identity: LogicalChannelIdentity(lcid),
        served_radio_bearer: Some(RLC_BearerConfigServedRadioBearer::Drb_Identity(
            DRB_Identity(drb_id),
        )),
        reestablish_rlc: None,
        rlc_config: None,
        mac_logical_channel_config: None,
    };

    CellGroupConfig {
        cell_group_id: CellGroupId(0),
        rlc_bearer_to_add_mod_list: Some(CellGroupConfigRlc_BearerToAddModList(vec![rlc_bearer])),
        rlc_bearer_to_release_list: None,
        mac_cell_group_config: None,
        physical_cell_group_config: None,
        sp_cell_config: None,
        s_cell_to_add_mod_list: None,
        s_cell_to_release_list: None,
    }
}

/// amfg-04: build a fully-structured `RrcReconfigurationParams` for bringing up
/// one PDU session's DRB. The structured `RadioBearerConfig` and
/// `CellGroupConfig` are UPER-encoded into the existing byte-carrying param
/// fields (`radio_bearer_config` is decoded back and embedded structurally by
/// `build_rrc_reconfiguration`; `master_cell_group` is carried as the
/// `OCTET STRING (CONTAINING CellGroupConfig)` masterCellGroup IE). The opaque
/// byte path is preserved for callers that already have pre-encoded config.
pub fn build_drb_reconfiguration_params(
    rrc_transaction_id: u8,
    pdu_session_id: u8,
    drb_id: u8,
    lcid: u8,
    qfis: &[u8],
    default_drb: bool,
) -> Result<RrcReconfigurationParams, RrcReconfigurationError> {
    let rbc = build_drb_radio_bearer_config(pdu_session_id, drb_id, qfis, default_drb);
    let cgc = build_cell_group_config(drb_id, lcid);

    let radio_bearer_config = encode_rrc(&rbc)?;
    let master_cell_group = encode_rrc(&cgc)?;

    Ok(RrcReconfigurationParams {
        rrc_transaction_id,
        radio_bearer_config: Some(radio_bearer_config),
        secondary_cell_group: None,
        master_cell_group: Some(master_cell_group),
        full_config: false,
    })
}

/// Parse an RRC Reconfiguration from a DL-DCCH message
pub fn parse_rrc_reconfiguration(
    msg: &DL_DCCH_Message,
) -> Result<RrcReconfigurationData, RrcReconfigurationError> {
    let rrc_reconfiguration = match &msg.message {
        DL_DCCH_MessageType::C1(c1) => match c1 {
            DL_DCCH_MessageType_c1::RrcReconfiguration(reconfig) => reconfig,
            _ => {
                return Err(RrcReconfigurationError::InvalidMessageType {
                    expected: "RRCReconfiguration".to_string(),
                    actual: "other c1 message".to_string(),
                })
            }
        },
        _ => {
            return Err(RrcReconfigurationError::InvalidMessageType {
                expected: "c1".to_string(),
                actual: "messageClassExtension".to_string(),
            })
        }
    };

    let ies = match &rrc_reconfiguration.critical_extensions {
        RRCReconfigurationCriticalExtensions::RrcReconfiguration(ies) => ies,
        RRCReconfigurationCriticalExtensions::CriticalExtensionsFuture(_) => {
            return Err(RrcReconfigurationError::InvalidMessageType {
                expected: "rrcReconfiguration".to_string(),
                actual: "criticalExtensionsFuture".to_string(),
            })
        }
    };

    // Encode radio bearer config back to bytes if present
    let radio_bearer_config = if let Some(ref config) = ies.radio_bearer_config {
        Some(encode_rrc(config)?)
    } else {
        None
    };

    // Extract secondary cell group
    let secondary_cell_group = ies.secondary_cell_group.as_ref().map(|scg| scg.0.clone());

    // Extract master cell group and full_config from v1530 extension
    let (master_cell_group, full_config) = if let Some(ref ext) = ies.non_critical_extension {
        let mcg = ext.master_cell_group.as_ref().map(|m| m.0.clone());
        let fc = ext.full_config.is_some();
        (mcg, fc)
    } else {
        (None, false)
    };

    Ok(RrcReconfigurationData {
        rrc_transaction_id: rrc_reconfiguration.rrc_transaction_identifier.0,
        radio_bearer_config,
        secondary_cell_group,
        master_cell_group,
        full_config,
    })
}

// ============================================================================
// RRC Reconfiguration Complete
// ============================================================================

/// Parameters for building an RRC Reconfiguration Complete message
#[derive(Debug, Clone)]
pub struct RrcReconfigurationCompleteParams {
    /// RRC Transaction Identifier (0-3)
    pub rrc_transaction_id: u8,
}

/// Parsed RRC Reconfiguration Complete data
#[derive(Debug, Clone)]
pub struct RrcReconfigurationCompleteData {
    /// RRC Transaction Identifier
    pub rrc_transaction_id: u8,
}

/// Build an RRC Reconfiguration Complete message
pub fn build_rrc_reconfiguration_complete(
    params: &RrcReconfigurationCompleteParams,
) -> Result<UL_DCCH_Message, RrcReconfigurationError> {
    if params.rrc_transaction_id > 3 {
        return Err(RrcReconfigurationError::InvalidFieldValue(
            "RRC Transaction ID must be 0-3".to_string(),
        ));
    }

    let rrc_reconfiguration_complete_ies = RRCReconfigurationComplete_IEs {
        late_non_critical_extension: None,
        non_critical_extension: None,
    };

    let rrc_reconfiguration_complete = RRCReconfigurationComplete {
        rrc_transaction_identifier: RRC_TransactionIdentifier(params.rrc_transaction_id),
        critical_extensions:
            RRCReconfigurationCompleteCriticalExtensions::RrcReconfigurationComplete(
                rrc_reconfiguration_complete_ies,
            ),
    };

    let message_type = UL_DCCH_MessageType::C1(UL_DCCH_MessageType_c1::RrcReconfigurationComplete(
        rrc_reconfiguration_complete,
    ));

    Ok(UL_DCCH_Message {
        message: message_type,
    })
}

/// Parse an RRC Reconfiguration Complete from a UL-DCCH message
pub fn parse_rrc_reconfiguration_complete(
    msg: &UL_DCCH_Message,
) -> Result<RrcReconfigurationCompleteData, RrcReconfigurationError> {
    let rrc_reconfiguration_complete = match &msg.message {
        UL_DCCH_MessageType::C1(c1) => match c1 {
            UL_DCCH_MessageType_c1::RrcReconfigurationComplete(complete) => complete,
            _ => {
                return Err(RrcReconfigurationError::InvalidMessageType {
                    expected: "RRCReconfigurationComplete".to_string(),
                    actual: "other c1 message".to_string(),
                })
            }
        },
        _ => {
            return Err(RrcReconfigurationError::InvalidMessageType {
                expected: "c1".to_string(),
                actual: "messageClassExtension".to_string(),
            })
        }
    };

    // Verify we have the expected critical extensions variant
    match &rrc_reconfiguration_complete.critical_extensions {
        RRCReconfigurationCompleteCriticalExtensions::RrcReconfigurationComplete(_) => {}
        RRCReconfigurationCompleteCriticalExtensions::CriticalExtensionsFuture(_) => {
            return Err(RrcReconfigurationError::InvalidMessageType {
                expected: "rrcReconfigurationComplete".to_string(),
                actual: "criticalExtensionsFuture".to_string(),
            })
        }
    };

    Ok(RrcReconfigurationCompleteData {
        rrc_transaction_id: rrc_reconfiguration_complete.rrc_transaction_identifier.0,
    })
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Build and encode an RRC Reconfiguration to bytes
pub fn encode_rrc_reconfiguration(
    params: &RrcReconfigurationParams,
) -> Result<Vec<u8>, RrcReconfigurationError> {
    let msg = build_rrc_reconfiguration(params)?;
    Ok(encode_rrc(&msg)?)
}

/// Decode and parse an RRC Reconfiguration from bytes
pub fn decode_rrc_reconfiguration(
    bytes: &[u8],
) -> Result<RrcReconfigurationData, RrcReconfigurationError> {
    let msg: DL_DCCH_Message = decode_rrc(bytes)?;
    parse_rrc_reconfiguration(&msg)
}

/// Build and encode an RRC Reconfiguration Complete to bytes
pub fn encode_rrc_reconfiguration_complete(
    params: &RrcReconfigurationCompleteParams,
) -> Result<Vec<u8>, RrcReconfigurationError> {
    let msg = build_rrc_reconfiguration_complete(params)?;
    Ok(encode_rrc(&msg)?)
}

/// Decode and parse an RRC Reconfiguration Complete from bytes
pub fn decode_rrc_reconfiguration_complete(
    bytes: &[u8],
) -> Result<RrcReconfigurationCompleteData, RrcReconfigurationError> {
    let msg: UL_DCCH_Message = decode_rrc(bytes)?;
    parse_rrc_reconfiguration_complete(&msg)
}

/// Check if a DL-DCCH message is an RRC Reconfiguration
pub fn is_rrc_reconfiguration(msg: &DL_DCCH_Message) -> bool {
    matches!(
        &msg.message,
        DL_DCCH_MessageType::C1(DL_DCCH_MessageType_c1::RrcReconfiguration(_))
    )
}

/// Check if a UL-DCCH message is an RRC Reconfiguration Complete
pub fn is_rrc_reconfiguration_complete(msg: &UL_DCCH_Message) -> bool {
    matches!(
        &msg.message,
        UL_DCCH_MessageType::C1(UL_DCCH_MessageType_c1::RrcReconfigurationComplete(_))
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // RRC Reconfiguration Tests
    // ========================================================================

    fn create_test_reconfiguration_params() -> RrcReconfigurationParams {
        RrcReconfigurationParams {
            rrc_transaction_id: 0,
            radio_bearer_config: None,
            secondary_cell_group: None,
            master_cell_group: Some(vec![0x00, 0x01, 0x02]), // Sample cell group config
            full_config: false,
        }
    }

    #[test]
    fn test_build_rrc_reconfiguration() {
        let params = create_test_reconfiguration_params();
        let result = build_rrc_reconfiguration(&params);
        assert!(result.is_ok());

        let msg = result.unwrap();
        assert!(is_rrc_reconfiguration(&msg));
    }

    #[test]
    fn test_parse_rrc_reconfiguration() {
        let params = create_test_reconfiguration_params();
        let msg = build_rrc_reconfiguration(&params).unwrap();
        let result = parse_rrc_reconfiguration(&msg);
        assert!(result.is_ok());

        let data = result.unwrap();
        assert_eq!(data.rrc_transaction_id, params.rrc_transaction_id);
        assert_eq!(data.master_cell_group, params.master_cell_group);
        assert_eq!(data.full_config, params.full_config);
    }

    #[test]
    fn test_rrc_reconfiguration_with_full_config() {
        let params = RrcReconfigurationParams {
            rrc_transaction_id: 2,
            radio_bearer_config: None,
            secondary_cell_group: None,
            master_cell_group: Some(vec![0xAA, 0xBB]),
            full_config: true,
        };

        let msg = build_rrc_reconfiguration(&params).unwrap();
        let data = parse_rrc_reconfiguration(&msg).unwrap();

        assert_eq!(data.rrc_transaction_id, 2);
        assert!(data.full_config);
        assert_eq!(data.master_cell_group, Some(vec![0xAA, 0xBB]));
    }

    #[test]
    fn test_encode_decode_rrc_reconfiguration() {
        let params = create_test_reconfiguration_params();
        let encoded = encode_rrc_reconfiguration(&params);
        assert!(encoded.is_ok());

        let bytes = encoded.unwrap();
        assert!(!bytes.is_empty());

        let decoded = decode_rrc_reconfiguration(&bytes);
        assert!(decoded.is_ok());

        let data = decoded.unwrap();
        assert_eq!(data.rrc_transaction_id, params.rrc_transaction_id);
    }

    #[test]
    fn test_invalid_rrc_transaction_id_reconfiguration() {
        let params = RrcReconfigurationParams {
            rrc_transaction_id: 5, // Invalid: must be 0-3
            radio_bearer_config: None,
            secondary_cell_group: None,
            master_cell_group: None,
            full_config: false,
        };

        let result = build_rrc_reconfiguration(&params);
        assert!(result.is_err());
    }

    // ========================================================================
    // RRC Reconfiguration Complete Tests
    // ========================================================================

    fn create_test_reconfiguration_complete_params() -> RrcReconfigurationCompleteParams {
        RrcReconfigurationCompleteParams {
            rrc_transaction_id: 0,
        }
    }

    #[test]
    fn test_build_rrc_reconfiguration_complete() {
        let params = create_test_reconfiguration_complete_params();
        let result = build_rrc_reconfiguration_complete(&params);
        assert!(result.is_ok());

        let msg = result.unwrap();
        assert!(is_rrc_reconfiguration_complete(&msg));
    }

    #[test]
    fn test_parse_rrc_reconfiguration_complete() {
        let params = create_test_reconfiguration_complete_params();
        let msg = build_rrc_reconfiguration_complete(&params).unwrap();
        let result = parse_rrc_reconfiguration_complete(&msg);
        assert!(result.is_ok());

        let data = result.unwrap();
        assert_eq!(data.rrc_transaction_id, params.rrc_transaction_id);
    }

    #[test]
    fn test_encode_decode_rrc_reconfiguration_complete() {
        let params = create_test_reconfiguration_complete_params();
        let encoded = encode_rrc_reconfiguration_complete(&params);
        assert!(encoded.is_ok());

        let bytes = encoded.unwrap();
        assert!(!bytes.is_empty());

        let decoded = decode_rrc_reconfiguration_complete(&bytes);
        assert!(decoded.is_ok());

        let data = decoded.unwrap();
        assert_eq!(data.rrc_transaction_id, params.rrc_transaction_id);
    }

    #[test]
    fn test_invalid_rrc_transaction_id_complete() {
        let params = RrcReconfigurationCompleteParams {
            rrc_transaction_id: 4, // Invalid: must be 0-3
        };

        let result = build_rrc_reconfiguration_complete(&params);
        assert!(result.is_err());
    }

    #[test]
    fn test_rrc_reconfiguration_complete_all_transaction_ids() {
        // Test all valid transaction IDs (0-3)
        for id in 0..=3 {
            let params = RrcReconfigurationCompleteParams {
                rrc_transaction_id: id,
            };
            let msg = build_rrc_reconfiguration_complete(&params).unwrap();
            let data = parse_rrc_reconfiguration_complete(&msg).unwrap();
            assert_eq!(data.rrc_transaction_id, id);
        }
    }

    // ========================================================================
    // amfg-04: structured DRB / SDAP / CellGroup tests
    // ========================================================================

    #[test]
    fn test_build_drb_radio_bearer_config_maps_all_qfis() {
        let qfis = [1u8, 5, 9];
        let rbc = build_drb_radio_bearer_config(2, 1, &qfis, true);

        let drb_list = rbc
            .drb_to_add_mod_list
            .as_ref()
            .expect("DRB-ToAddModList present");
        assert_eq!(drb_list.0.len(), 1);
        let drb = &drb_list.0[0];
        assert_eq!(drb.drb_identity.0, 1);

        let sdap = match drb.cn_association.as_ref().expect("cn-Association present") {
            DRB_ToAddModCnAssociation::Sdap_Config(s) => s,
            _ => panic!("expected SDAP-Config cn-Association"),
        };
        assert_eq!(sdap.pdu_session.0, 2);
        assert!(sdap.default_drb.0);
        let mapped = sdap
            .mapped_qo_s_flows_to_add
            .as_ref()
            .expect("mappedQoS-FlowsToAdd present");
        let mapped_vals: Vec<u8> = mapped.0.iter().map(|q| q.0).collect();
        assert_eq!(mapped_vals, vec![1, 5, 9]);
    }

    #[test]
    fn test_build_drb_radio_bearer_config_empty_qfis_omits_mapping() {
        let rbc = build_drb_radio_bearer_config(1, 1, &[], false);
        let drb = &rbc.drb_to_add_mod_list.as_ref().unwrap().0[0];
        let sdap = match drb.cn_association.as_ref().unwrap() {
            DRB_ToAddModCnAssociation::Sdap_Config(s) => s,
            _ => panic!("expected SDAP-Config"),
        };
        assert!(sdap.mapped_qo_s_flows_to_add.is_none());
    }

    #[test]
    fn test_radio_bearer_config_uper_roundtrip() {
        let qfis = [1u8, 2, 3];
        let rbc = build_drb_radio_bearer_config(4, 2, &qfis, true);
        let bytes = encode_rrc(&rbc).expect("encode RadioBearerConfig");
        let decoded: RadioBearerConfig = decode_rrc(&bytes).expect("decode RadioBearerConfig");
        assert_eq!(decoded, rbc);

        // Confirm the decoded SDAP carries the same QFIs.
        let drb = &decoded.drb_to_add_mod_list.unwrap().0[0];
        let sdap = match drb.cn_association.as_ref().unwrap() {
            DRB_ToAddModCnAssociation::Sdap_Config(s) => s,
            _ => panic!("expected SDAP-Config"),
        };
        let mapped: Vec<u8> = sdap
            .mapped_qo_s_flows_to_add
            .as_ref()
            .unwrap()
            .0
            .iter()
            .map(|q| q.0)
            .collect();
        assert_eq!(mapped, vec![1, 2, 3]);
    }

    #[test]
    fn test_cell_group_config_uper_roundtrip() {
        let cgc = build_cell_group_config(2, 4);
        let bytes = encode_rrc(&cgc).expect("encode CellGroupConfig");
        let decoded: CellGroupConfig = decode_rrc(&bytes).expect("decode CellGroupConfig");
        assert_eq!(decoded, cgc);

        let rlc_list = decoded
            .rlc_bearer_to_add_mod_list
            .expect("RLC bearer list present");
        assert_eq!(rlc_list.0.len(), 1);
        assert_eq!(rlc_list.0[0].logical_channel_identity.0, 4);
        match rlc_list.0[0].served_radio_bearer.as_ref().unwrap() {
            RLC_BearerConfigServedRadioBearer::Drb_Identity(d) => assert_eq!(d.0, 2),
            _ => panic!("expected served DRB identity"),
        }
    }

    #[test]
    fn test_build_drb_reconfiguration_params_full_message_roundtrip() {
        let qfis = [1u8, 6, 7];
        let params = build_drb_reconfiguration_params(1, 5, 1, 4, &qfis, true)
            .expect("build structured reconfig params");

        // Build the full RRCReconfiguration and round-trip through UPER.
        let encoded = encode_rrc_reconfiguration(&params).expect("encode RRCReconfiguration");
        let data = decode_rrc_reconfiguration(&encoded).expect("decode RRCReconfiguration");

        // The carried radio_bearer_config decodes back to the structured DRB.
        let rbc_bytes = data
            .radio_bearer_config
            .expect("radio_bearer_config present after roundtrip");
        let rbc: RadioBearerConfig = decode_rrc(&rbc_bytes).expect("decode RBC");
        let drb_list = rbc.drb_to_add_mod_list.expect("DRB list");
        assert_eq!(drb_list.0.len(), 1);
        let sdap = match drb_list.0[0].cn_association.as_ref().unwrap() {
            DRB_ToAddModCnAssociation::Sdap_Config(s) => s,
            _ => panic!("expected SDAP-Config"),
        };
        assert_eq!(sdap.pdu_session.0, 5);
        let mapped: Vec<u8> = sdap
            .mapped_qo_s_flows_to_add
            .as_ref()
            .unwrap()
            .0
            .iter()
            .map(|q| q.0)
            .collect();
        assert_eq!(mapped, vec![1, 6, 7]);

        // The masterCellGroup octet string decodes back to the structured cell group.
        let mcg_bytes = data.master_cell_group.expect("master_cell_group present");
        let cgc: CellGroupConfig = decode_rrc(&mcg_bytes).expect("decode CGC");
        assert_eq!(
            cgc.rlc_bearer_to_add_mod_list.unwrap().0[0]
                .logical_channel_identity
                .0,
            4
        );
    }
}
