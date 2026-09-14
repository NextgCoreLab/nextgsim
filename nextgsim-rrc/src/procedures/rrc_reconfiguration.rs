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
    // Handover command golden bytes (issue #107, criterion 6)
    // ========================================================================

    /// Hand-derived handover command: `RRCReconfiguration`, tid 0, whose
    /// `masterCellGroup` carries a `reconfigurationWithSync` to physCellId 16 with
    /// `newUE-Identity` 1 and `t304` 1000 ms, and `fullConfig` set.
    ///
    /// Derivation of the framing from `tools/rrc-15.6.0.asn1`:
    ///
    /// ```text
    /// bit 0      DL-DCCH-MessageType CHOICE, 2 alternatives -> 1 bit. c1 = 0
    /// bits 1..4  c1 CHOICE, 16 alternatives -> 4 bits.
    ///            rrcReconfiguration = 0 -> 0000
    /// bits 5..6  RRC-TransactionIdentifier, INTEGER (0..3) -> 2 bits. tid 0 -> 00
    /// bit 7      RRCReconfiguration criticalExtensions CHOICE -> 1 bit.
    ///            rrcReconfiguration = 0
    ///            => byte 0 = 0b0_0000_00_0 = 0x00
    /// ```
    ///
    /// The tail is the `RRCReconfiguration-IEs` preamble, the v1530 extension
    /// carrying `masterCellGroup` and `fullConfig`, and the embedded
    /// `CellGroupConfig`. Those inner layers are pinned by this literal and
    /// cross-checked by `golden_handover_command_cross_decode` rather than derived
    /// bit by bit — stated plainly, because a wrong derivation in a comment is worse
    /// than an honest "cross-checked".
    const GOLDEN_HANDOVER_COMMAND_TID0: [u8; 13] = [
        0x00, 0x0E, 0x00, 0x48, 0x20, 0x42, 0x40, 0x00, 0x20, 0x50, 0x00, 0x03, 0x40,
    ];

    fn golden_handover_params(rrc_transaction_id: u8) -> HandoverCommandParams {
        HandoverCommandParams {
            rrc_transaction_id,
            target_phys_cell_id: 16,
            new_ue_identity: 1,
            t304_ms: 1000,
            full_config: true,
        }
    }

    #[test]
    fn golden_handover_command_bytes() {
        let bytes = encode_handover_command(&golden_handover_params(0)).expect("encode");
        assert_eq!(
            bytes,
            GOLDEN_HANDOVER_COMMAND_TID0.to_vec(),
            "the handover command must match the hand-derived UPER bytes, not a \
             round trip through this codec"
        );

        // The tid occupies bits 5..6 of byte 0 only: tid 2 -> 0b0_0000_10_0 = 0x04.
        let bytes_tid2 = encode_handover_command(&golden_handover_params(2)).expect("encode");
        let mut expected = GOLDEN_HANDOVER_COMMAND_TID0;
        expected[0] = 0x04;
        assert_eq!(bytes_tid2, expected.to_vec());
    }

    #[test]
    fn golden_handover_command_cross_decode() {
        let data = decode_handover_command(&GOLDEN_HANDOVER_COMMAND_TID0)
            .expect("the golden must decode as a handover command");
        assert_eq!(data.rrc_transaction_id, 0);
        assert_eq!(data.target_phys_cell_id, 16);
        assert_eq!(data.new_ue_identity, 1);
        assert_eq!(data.t304_ms, 1000);
        assert!(data.full_config);

        // And the reconfigurationWithSync really is there, which is what makes this
        // a handover command rather than a plain reconfiguration.
        let reconf = decode_rrc_reconfiguration(&GOLDEN_HANDOVER_COMMAND_TID0).expect("decode");
        let cgc: CellGroupConfig = decode_rrc(
            reconf
                .master_cell_group
                .as_ref()
                .expect("masterCellGroup present"),
        )
        .expect("decode CellGroupConfig");
        let sync = cgc
            .sp_cell_config
            .expect("spCellConfig present")
            .reconfiguration_with_sync
            .expect("reconfigurationWithSync present");
        assert_eq!(
            sync.sp_cell_config_common
                .expect("spCellConfigCommon present")
                .phys_cell_id
                .expect("physCellId present")
                .0,
            16
        );
    }

    /// A target PCI past the ASN.1 bound is refused rather than encoded.
    #[test]
    fn an_out_of_range_target_phys_cell_id_is_refused() {
        let mut params = golden_handover_params(0);
        params.target_phys_cell_id = 1008;
        assert!(encode_handover_command(&params).is_err());
    }

    /// `t304` enumerates non-uniform values, so a value outside the set is refused
    /// rather than rounded: it is the timer that decides when the UE declares
    /// handover failure.
    #[test]
    fn a_non_enumerated_t304_is_refused_not_rounded() {
        for legal in [50u16, 100, 150, 200, 500, 1000, 2000, 10000] {
            let mut params = golden_handover_params(0);
            params.t304_ms = legal;
            assert!(
                encode_handover_command(&params).is_ok(),
                "t304 {legal} ms is enumerated and must encode"
            );
            assert_eq!(
                t304_ms(t304_index(legal).expect("index")).expect("ms"),
                legal
            );
        }
        for illegal in [0u16, 300, 250, 1500, 9999] {
            let mut params = golden_handover_params(0);
            params.t304_ms = illegal;
            assert!(
                encode_handover_command(&params).is_err(),
                "t304 {illegal} ms is NOT enumerated and must be refused, not \
                 rounded to a deadline nobody configured"
            );
        }
    }

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

    // ========================================================================
    // Wave-6 C5 — hand-derived golden byte vectors (TS 38.331 §6.2.1/§5.3.5,
    // UPER per X.691). Derived BY HAND from tools/rrc-15.6.0.asn1, NOT produced
    // by the encoder — the reviewer re-derives every bit below.
    // ========================================================================

    /// Minimal RRCReconfiguration on DL-DCCH, tid 0, all IEs absent, 13 bits:
    ///
    /// ```text
    /// DL-DCCH-MessageType CHOICE {c1, mce} — 1 bit, c1                       0
    ///   c1 CHOICE (16 alts) — 4 bits, rrcReconfiguration = index 0        0000
    /// RRCReconfiguration ::= SEQUENCE { tid, criticalExtensions } (no ext)
    ///   rrc-TransactionIdentifier — 2 bits, 0                               00
    ///   criticalExtensions CHOICE {rrcReconfiguration, future} — 1 bit       0
    /// RRCReconfiguration-IEs ::= SEQUENCE { 5 OPTIONAL fields } (no ext)
    ///   presence (radioBearer|secondaryCG|measConfig|lateNC|nonCrit)     00000
    /// = 0 0000 00 0 00000  (13 bits) + 3 pad = 0000 0000 | 0000 0000 -> 0x00 0x00
    /// ```
    const GOLDEN_RECONFIG_MINIMAL_TID0: [u8; 2] = [0x00, 0x00];

    /// RRCReconfigurationComplete on UL-DCCH, tid 0, 10 bits:
    ///
    /// ```text
    /// UL-DCCH-MessageType CHOICE {c1, mce} — 1 bit, c1                       0
    ///   c1 CHOICE (16 alts) — 4 bits, rrcReconfigurationComplete = index 1 0001
    /// RRCReconfigurationComplete ::= SEQUENCE { tid, critExt } (no ext)
    ///   rrc-TransactionIdentifier — 2 bits, 0                               00
    ///   criticalExtensions CHOICE {rrcReconfigurationComplete, future} 1 bit 0
    /// RRCReconfigurationComplete-IEs ::= SEQUENCE { 2 OPTIONAL fields }(no ext)
    ///   presence (lateNonCritical|nonCritical)                              00
    /// = 0 0001 00 0 00  (10 bits) + 6 pad = 0000 1000 | 0000 0000 -> 0x08 0x00
    /// ```
    const GOLDEN_RECONFIG_COMPLETE_TID0: [u8; 2] = [0x08, 0x00];

    #[test]
    fn golden_rrc_reconfiguration_minimal_bytes() {
        let bytes = encode_rrc_reconfiguration(&RrcReconfigurationParams {
            rrc_transaction_id: 0,
            radio_bearer_config: None,
            secondary_cell_group: None,
            master_cell_group: None,
            full_config: false,
        })
        .expect("encode minimal RRCReconfiguration");
        assert_eq!(
            bytes,
            GOLDEN_RECONFIG_MINIMAL_TID0.to_vec(),
            "minimal RRCReconfiguration(tid 0) must match the hand-derived UPER bytes"
        );
    }

    #[test]
    fn golden_rrc_reconfiguration_complete_bytes() {
        let bytes = encode_rrc_reconfiguration_complete(&RrcReconfigurationCompleteParams {
            rrc_transaction_id: 0,
        })
        .expect("encode RRCReconfigurationComplete");
        assert_eq!(
            bytes,
            GOLDEN_RECONFIG_COMPLETE_TID0.to_vec(),
            "RRCReconfigurationComplete(tid 0) must match the hand-derived UPER bytes"
        );
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

    // ========================================================================
    // Wave-6 H6 — hand-derived golden UPER byte vectors for the LIVE DRB
    // RRCReconfiguration path (TS 38.331 §5.3.5 / §6.2.2 / §6.3.2, UPER per
    // ITU-T X.691). Authored with the H5 dual-derivation method
    // (`.context/GOLDEN-VECTOR-METHOD.md`):
    //
    //   * Derivation A = the per-byte bit tables in the doc comments below,
    //     hand-derived from `tools/rrc-15.6.0.asn1`.
    //   * Derivation B = the independent `bderive_*` UPER recompute below — a
    //     clean-room bit writer built directly from X.691 + the ASN.1, NOT the
    //     production `encode_rrc`. `golden_h6_derivation_b_matches_frozen`
    //     asserts B reproduces every frozen array (catches a symmetric
    //     encoder/decoder bug the round-trip tests cannot see).
    //   * The production encoder is the THIRD, independent check (tier-1
    //     `golden_*_bytes` tests).
    //
    // Exact production shape: `establish_drb` (nextgsim-gnb ngap/task.rs) for a
    // PDU session with id 1 uses drb_id = psi.clamp(1,32) = 1, lcid =
    // (3+drb_id).min(32) = 4, tid pinned 0, one accepted QFI. It calls
    // `build_drb_reconfiguration_params(0, 1, 1, 4, &[qfi], true)`, which
    // `encode_rrc`s the RadioBearerConfig and CellGroupConfig standalone (those
    // two standalone encodings are the `params.radio_bearer_config` and
    // masterCellGroup octet-string contents) before encoding the whole message.
    // ========================================================================

    /// Independent UNALIGNED-PER bit writer for derivation B (H5 method).
    /// Deliberately NOT the generated encoder — it emits the bitstream directly
    /// from the X.691 rules so that agreeing with `encode_rrc` proves the
    /// golden bytes are spec-derived, not a self-consistent codec artefact.
    struct BitW {
        bits: Vec<u8>,
    }
    impl BitW {
        fn new() -> Self {
            Self { bits: Vec::new() }
        }
        /// One raw bit.
        fn bit(&mut self, b: u8) {
            self.bits.push(b & 1);
        }
        /// Non-negative binary integer into `n` bits, MSB first (X.691 3.7.19).
        fn nbits(&mut self, value: u64, n: u32) {
            for i in (0..n).rev() {
                self.bits.push(((value >> i) & 1) as u8);
            }
        }
        /// X.691 §13.2 constrained whole number, UNALIGNED variant: a
        /// minimal-width bit field of `ceil(log2(range))` bits, never aligned;
        /// a range of 1 emits nothing (§13.2.1).
        fn cint(&mut self, value: u64, lo: u64, hi: u64) {
            let range = hi - lo + 1;
            if range == 1 {
                return;
            }
            let width = u64::BITS - (range - 1).leading_zeros();
            self.nbits(value - lo, width);
        }
        /// Unconstrained OCTET STRING, UNALIGNED (X.691 §11.9 + §17): a general
        /// length determinant (single octet `0nnnnnnn` for len < 128, no
        /// alignment) then the octets placed directly bit-by-bit.
        fn octet_string(&mut self, data: &[u8]) {
            assert!(data.len() < 128, "vectors use the single-octet length form");
            self.nbits(data.len() as u64, 8);
            for &b in data {
                self.nbits(u64::from(b), 8);
            }
        }
        /// Pad the trailing partial octet with zero bits (final message only).
        fn bytes(&self) -> Vec<u8> {
            let mut out = Vec::new();
            let mut i = 0;
            while i < self.bits.len() {
                let mut v = 0u8;
                for j in 0..8 {
                    v <<= 1;
                    if i + j < self.bits.len() {
                        v |= self.bits[i + j];
                    }
                }
                out.push(v);
                i += 8;
            }
            out
        }
    }

    /// Derivation B — RadioBearerConfig carrying one DRB (SDAP), inline body.
    fn bderive_drb_radio_bearer_config(w: &mut BitW, psi: u64, drb_id: u64, qfis: &[u64]) {
        // RadioBearerConfig ::= SEQUENCE {5 OPTIONAL, ...} (extensible)
        w.bit(0); // extension bit
        w.nbits(0b0_0100, 5); // presence: only drb-ToAddModList
                              // DRB-ToAddModList ::= SEQUENCE (SIZE(1..maxDRB=29)) OF
        w.cint(1, 1, 29); // one element
                          // DRB-ToAddMod ::= SEQUENCE {4 OPTIONAL, ...} (extensible)
        w.bit(0); // extension bit
        w.nbits(0b1001, 4); // presence: cnAssociation + pdcp-Config
                            // cnAssociation CHOICE {eps-BearerIdentity, sdap-Config} (2 alts)
        w.cint(1, 0, 1); // sdap-Config = index 1
                         // SDAP-Config ::= SEQUENCE {2 OPTIONAL, ...} (extensible)
        w.bit(0); // extension bit
        w.nbits(0b10, 2); // presence: only mappedQoS-FlowsToAdd
        w.cint(psi, 0, 255); // pdu-Session (PDU-SessionID 0..255)
        w.cint(0, 0, 1); // sdap-HeaderDL: present = index 0
        w.cint(0, 0, 1); // sdap-HeaderUL: present = index 0
        w.bit(1); // defaultDRB = TRUE (BOOLEAN)
                  // mappedQoS-FlowsToAdd ::= SEQUENCE (SIZE(1..maxNrofQFIs=64)) OF QFI
        w.cint(qfis.len() as u64, 1, 64);
        for &q in qfis {
            w.cint(q, 0, 63); // QFI (0..maxQFI=63)
        }
        w.cint(drb_id, 1, 32); // drb-Identity (mandatory, after the OPTIONAL cnAssociation)
                               // pdcp-Config ::= SEQUENCE {3 OPTIONAL, ...} (extensible)
        w.bit(0); // extension bit
        w.nbits(0b000, 3); // presence: drb / moreThanOneRLC / t-Reordering all absent
    }

    /// Derivation B — CellGroupConfig with one RLC bearer, inline body.
    fn bderive_cell_group_config(w: &mut BitW, lcid: u64, served_is_drb: bool, served_id: u64) {
        // CellGroupConfig ::= SEQUENCE {7 OPTIONAL, ..., ext-group} (extensible)
        w.bit(0); // extension bit
        w.nbits(0b100_0000, 7); // presence: only rlc-BearerToAddModList
        w.cint(0, 0, 3); // cellGroupId (0..maxSecondaryCellGroups=3)
                         // rlc-BearerToAddModList ::= SEQUENCE (SIZE(1..maxLC-ID=32)) OF
        w.cint(1, 1, 32); // one element
                          // RLC-BearerConfig ::= SEQUENCE {4 OPTIONAL, ...} (extensible)
        w.bit(0); // extension bit
        w.nbits(0b1000, 4); // presence: only servedRadioBearer
        w.cint(lcid, 1, 32); // logicalChannelIdentity (1..32)
                             // servedRadioBearer CHOICE {srb-Identity, drb-Identity} (2 alts)
        w.cint(u64::from(served_is_drb), 0, 1);
        if served_is_drb {
            w.cint(served_id, 1, 32); // DRB-Identity (1..32)
        } else {
            w.cint(served_id, 1, 3); // SRB-Identity (1..3)
        }
    }

    fn bderive_drb_reconfiguration(
        tid: u64,
        psi: u64,
        drb_id: u64,
        lcid: u64,
        qfis: &[u64],
    ) -> Vec<u8> {
        let mut w = BitW::new();
        // DL-DCCH-MessageType CHOICE {c1, messageClassExtension} (2 alts)
        w.cint(0, 0, 1); // c1 = index 0
                         // c1 CHOICE (16 alternatives, spare-padded, not extensible)
        w.cint(0, 0, 15); // rrcReconfiguration = index 0
                          // RRCReconfiguration ::= SEQUENCE {tid, criticalExtensions}
        w.cint(tid, 0, 3); // rrc-TransactionIdentifier (0..3)
                           // criticalExtensions CHOICE {rrcReconfiguration, future} (2 alts)
        w.cint(0, 0, 1); // rrcReconfiguration = index 0
                         // RRCReconfiguration-IEs ::= SEQUENCE {5 OPTIONAL} (NOT extensible)
        w.nbits(0b1_0001, 5); // radioBearerConfig + nonCriticalExtension present
        bderive_drb_radio_bearer_config(&mut w, psi, drb_id, qfis);
        // nonCriticalExtension = RRCReconfiguration-v1530-IEs ::= SEQUENCE {8 OPTIONAL}
        w.nbits(0b1000_0000, 8); // only masterCellGroup present
                                 // masterCellGroup ::= OCTET STRING (CONTAINING CellGroupConfig)
        let mut cg = BitW::new();
        bderive_cell_group_config(&mut cg, lcid, true, drb_id);
        w.octet_string(&cg.bytes());
        w.bytes()
    }

    fn bderive_srb1_radio_bearer_config() -> Vec<u8> {
        let mut w = BitW::new();
        // RadioBearerConfig ::= SEQUENCE {5 OPTIONAL, ...} (extensible)
        w.bit(0);
        w.nbits(0b1_0000, 5); // only srb-ToAddModList
                              // SRB-ToAddModList ::= SEQUENCE (SIZE(1..2)) OF
        w.cint(1, 1, 2); // one element
                         // SRB-ToAddMod ::= SEQUENCE {3 OPTIONAL, ...} (extensible)
        w.bit(0);
        w.nbits(0b000, 3);
        w.cint(1, 1, 3); // srb-Identity = 1
        w.bytes()
    }

    /// DRB RadioBearerConfig standalone (the live `params.radio_bearer_config`),
    /// UPER, 52 bits (psi=1, drb_id=1, one QFI=1):
    ///
    /// ```text
    /// RadioBearerConfig ::= SEQUENCE {5 OPT, ...}   ext=0, presence 0 0100
    /// DRB-ToAddModList SIZE(1..29) — 5-bit len, count 1 -> 0 0000
    /// DRB-ToAddMod ::= SEQUENCE {4 OPT, ...}        ext=0, presence 1 0 0 1
    ///   cnAssociation CHOICE {eps,sdap} — 1 bit, sdap = idx 1        1
    ///   SDAP-Config ::= SEQUENCE {2 OPT, ...}       ext=0, presence 1 0
    ///     pdu-Session INTEGER(0..255) — 8 bits, 1        0000 0001
    ///     sdap-HeaderDL {present,absent} — 1 bit, present            0
    ///     sdap-HeaderUL {present,absent} — 1 bit, present            0
    ///     defaultDRB BOOLEAN — 1 bit, TRUE                           1
    ///     mappedQoS-FlowsToAdd SIZE(1..64) — 6-bit len, count 1 000000
    ///       QFI INTEGER(0..63) — 6 bits, value 1              000001
    ///   drb-Identity INTEGER(1..32) — 5 bits, value 1 -> off 0  0 0000
    ///   PDCP-Config ::= SEQUENCE {3 OPT, ...}       ext=0, presence 000
    /// = 0001 0000 0000 1001 1010 0000 0001 0010 0000 0000 0010 0000
    ///   + 4 pad -> 10 09 A0 12 00 20 00
    /// ```
    const GOLDEN_DRB_RADIO_BEARER_CONFIG_PSI1: [u8; 7] = [0x10, 0x09, 0xA0, 0x12, 0x00, 0x20, 0x00];

    /// DRB CellGroupConfig standalone (the live masterCellGroup octet-string
    /// contents), UPER, 31 bits (drb_id=1, lcid=4):
    ///
    /// ```text
    /// CellGroupConfig ::= SEQUENCE {7 OPT, ...}   ext=0, presence 1 000000
    /// cellGroupId INTEGER(0..3) — 2 bits, 0                          0 0
    /// rlc-BearerToAddModList SIZE(1..32) — 5-bit len, count 1  0 0000
    /// RLC-BearerConfig ::= SEQUENCE {4 OPT, ...}  ext=0, presence 1 000
    ///   logicalChannelIdentity INTEGER(1..32) — 5 bits, 4 -> off 3 00011
    ///   servedRadioBearer CHOICE {srb,drb} — 1 bit, drb = idx 1        1
    ///     drb-Identity INTEGER(1..32) — 5 bits, 1 -> off 0        0 0000
    /// = 0100 0000 0000 0000 1000 0001 1100 000 + 1 pad -> 40 00 81 C0
    /// ```
    const GOLDEN_DRB_CELL_GROUP_CONFIG_PSI1: [u8; 4] = [0x40, 0x00, 0x81, 0xC0];

    /// Full DL-DCCH RRCReconfiguration establishing one DRB, tid 0, 113 bits —
    /// the exact `establish_drb(psi=1)` shape (drb_id 1, lcid 4, QFI 1):
    ///
    /// ```text
    /// DL-DCCH-MessageType CHOICE {c1, mce} — 1 bit, c1                    0
    /// c1 CHOICE (16 alts) — 4 bits, rrcReconfiguration = idx 0        0000
    /// rrc-TransactionIdentifier INTEGER(0..3) — 2 bits, 0               0 0
    /// criticalExtensions CHOICE {rrcReconfiguration, future} — 1 bit      0
    /// RRCReconfiguration-IEs ::= SEQUENCE {5 OPT} (NOT ext)
    ///   presence (radioBearer|secondaryCG|meas|lateNC|nonCrit)     1 0001
    /// radioBearerConfig (inline) = the 52 GOLDEN_DRB_RADIO_BEARER_CONFIG_PSI1
    ///   bits, UNPADDED:              0 00100 00000 0 1001 1 0 10 00000001
    ///                                0 0 1 000000 000001 00000 0 000
    /// nonCriticalExtension = RRCReconfiguration-v1530-IEs {8 OPT} (NOT ext)
    ///   presence (mcg|full|nas|mku|sib1|sysinfo|other|nonCrit)  1 0000000
    /// masterCellGroup OCTET STRING (CONTAINING CellGroupConfig):
    ///   length determinant, 1 octet (<128), value 4          0000 0100
    ///   contents = GOLDEN_DRB_CELL_GROUP_CONFIG_PSI1 (4 octets, unaligned)
    ///                          0100 0000 0000 0000 1000 0001 1100 0000
    /// = 113 bits + 7 pad ->
    ///   00 88 80 4D 00 90 01 00 40 02 20 00 40 E0 00
    /// ```
    const GOLDEN_RECONFIG_DRB_PSI1: [u8; 15] = [
        0x00, 0x88, 0x80, 0x4D, 0x00, 0x90, 0x01, 0x00, 0x40, 0x02, 0x20, 0x00, 0x40, 0xE0, 0x00,
    ];

    /// Derivation B reproduces every H6 frozen vector (independent of both the
    /// hand bit tables above AND the production `encode_rrc`).
    #[test]
    fn golden_h6_derivation_b_matches_frozen() {
        assert_eq!(
            bderive_srb1_radio_bearer_config(),
            [0x40, 0x00],
            "derivation B: SRB1 RadioBearerConfig"
        );
        assert_eq!(
            {
                let mut w = BitW::new();
                bderive_cell_group_config(&mut w, 4, true, 1);
                w.bytes()
            },
            GOLDEN_DRB_CELL_GROUP_CONFIG_PSI1.to_vec(),
            "derivation B: DRB CellGroupConfig"
        );
        assert_eq!(
            {
                let mut w = BitW::new();
                bderive_drb_radio_bearer_config(&mut w, 1, 1, &[1]);
                w.bytes()
            },
            GOLDEN_DRB_RADIO_BEARER_CONFIG_PSI1.to_vec(),
            "derivation B: DRB RadioBearerConfig"
        );
        assert_eq!(
            bderive_drb_reconfiguration(0, 1, 1, 4, &[1]),
            GOLDEN_RECONFIG_DRB_PSI1.to_vec(),
            "derivation B: DRB RRCReconfiguration"
        );
    }

    /// Tier 1 — encoder golden: the production `encode_rrc` output for the
    /// standalone DRB RadioBearerConfig must equal the frozen bytes.
    #[test]
    fn golden_drb_radio_bearer_config_bytes() {
        let bytes = encode_rrc(&build_drb_radio_bearer_config(1, 1, &[1], true)).expect("encode");
        assert_eq!(
            bytes,
            GOLDEN_DRB_RADIO_BEARER_CONFIG_PSI1.to_vec(),
            "DRB RadioBearerConfig must match the hand-derived UPER bytes"
        );
    }

    /// Tier 2 — decoder golden: the frozen DRB RadioBearerConfig bytes decode
    /// to exactly the builder's struct (guards decoder drift independently).
    #[test]
    fn golden_drb_radio_bearer_config_cross_decode() {
        let decoded: RadioBearerConfig =
            decode_rrc(&GOLDEN_DRB_RADIO_BEARER_CONFIG_PSI1).expect("decode RBC");
        assert_eq!(decoded, build_drb_radio_bearer_config(1, 1, &[1], true));
    }

    /// Tier 1 — encoder golden for the standalone DRB CellGroupConfig.
    #[test]
    fn golden_drb_cell_group_config_bytes() {
        let bytes = encode_rrc(&build_cell_group_config(1, 4)).expect("encode");
        assert_eq!(
            bytes,
            GOLDEN_DRB_CELL_GROUP_CONFIG_PSI1.to_vec(),
            "DRB CellGroupConfig must match the hand-derived UPER bytes"
        );
    }

    /// Tier 2 — decoder golden for the standalone DRB CellGroupConfig.
    #[test]
    fn golden_drb_cell_group_config_cross_decode() {
        let decoded: CellGroupConfig =
            decode_rrc(&GOLDEN_DRB_CELL_GROUP_CONFIG_PSI1).expect("decode CGC");
        assert_eq!(decoded, build_cell_group_config(1, 4));
    }

    /// Tier 1 — encoder golden for the FULL live DRB RRCReconfiguration
    /// message (`establish_drb(psi=1)`): flipping any encoder bit fails here.
    #[test]
    fn golden_drb_rrc_reconfiguration_bytes() {
        let params =
            build_drb_reconfiguration_params(0, 1, 1, 4, &[1], true).expect("build DRB params");
        let bytes = encode_rrc_reconfiguration(&params).expect("encode RRCReconfiguration");
        assert_eq!(
            bytes,
            GOLDEN_RECONFIG_DRB_PSI1.to_vec(),
            "DRB RRCReconfiguration must match the hand-derived UPER bytes"
        );
    }

    /// Tier 2 — decoder golden for the full DRB RRCReconfiguration: the frozen
    /// bytes decode to the same DL-DCCH message the builder produces, and the
    /// SDAP mapping / masterCellGroup survive intact.
    #[test]
    fn golden_drb_rrc_reconfiguration_cross_decode() {
        let params =
            build_drb_reconfiguration_params(0, 1, 1, 4, &[1], true).expect("build DRB params");
        let expected = build_rrc_reconfiguration(&params).expect("build message");
        let decoded: DL_DCCH_Message =
            decode_rrc(&GOLDEN_RECONFIG_DRB_PSI1).expect("decode DL-DCCH");
        assert_eq!(decoded, expected);

        // Spec-shape assertions on the decoded message (TS 38.331 §5.3.5.6).
        let data = parse_rrc_reconfiguration(&decoded).expect("parse");
        assert_eq!(data.rrc_transaction_id, 0);
        let rbc: RadioBearerConfig =
            decode_rrc(&data.radio_bearer_config.expect("radioBearerConfig")).expect("RBC");
        let drb = &rbc.drb_to_add_mod_list.expect("DRB list").0[0];
        assert_eq!(drb.drb_identity.0, 1);
        let sdap = match drb.cn_association.as_ref().expect("cn-Association") {
            DRB_ToAddModCnAssociation::Sdap_Config(s) => s,
            _ => panic!("expected SDAP-Config"),
        };
        assert_eq!(sdap.pdu_session.0, 1);
        let mapped: Vec<u8> = sdap
            .mapped_qo_s_flows_to_add
            .as_ref()
            .expect("mappedQoS-FlowsToAdd")
            .0
            .iter()
            .map(|q| q.0)
            .collect();
        assert_eq!(mapped, vec![1]);
        let cgc: CellGroupConfig =
            decode_rrc(&data.master_cell_group.expect("masterCellGroup")).expect("CGC");
        let bearer = &cgc.rlc_bearer_to_add_mod_list.expect("RLC list").0[0];
        assert_eq!(bearer.logical_channel_identity.0, 4);
        match bearer
            .served_radio_bearer
            .as_ref()
            .expect("servedRadioBearer")
        {
            RLC_BearerConfigServedRadioBearer::Drb_Identity(d) => assert_eq!(d.0, 1),
            _ => panic!("expected served DRB identity"),
        }
    }
}

// ============================================================================
// Handover command: RRCReconfiguration with reconfigurationWithSync
// (TS 38.331 §5.3.5.5.2, §6.3.2 — issue #107, criterion 3)
// ============================================================================

/// `t304` in milliseconds, and the ASN.1 enumeration index carrying it
/// (TS 38.331 §6.3.2 `ReconfigurationWithSync.t304`).
///
/// A table rather than arithmetic: the values are not a uniform step (50, 100,
/// 150, 200, 500, 1000, 2000, 10000), so any computed mapping is wrong above 200.
const T304_MS_TO_INDEX: [(u16, u8); 8] = [
    (50, 0),
    (100, 1),
    (150, 2),
    (200, 3),
    (500, 4),
    (1000, 5),
    (2000, 6),
    (10000, 7),
];

/// Maps a `t304` in milliseconds onto its enumeration index, refusing a value the
/// enumeration does not contain.
///
/// Refused rather than rounded to the nearest legal value: `t304` is the timer
/// that decides when the UE declares handover failure, and a silently adjusted
/// one makes a handover-failure test fire at a time nobody configured.
pub fn t304_index(ms: u16) -> Result<u8, RrcReconfigurationError> {
    T304_MS_TO_INDEX
        .iter()
        .find(|(value, _)| *value == ms)
        .map(|(_, index)| *index)
        .ok_or_else(|| {
            RrcReconfigurationError::InvalidFieldValue(format!(
                "t304 {ms} ms is not one of the values TS 38.331 enumerates \
                 (50, 100, 150, 200, 500, 1000, 2000, 10000)"
            ))
        })
}

/// The inverse of [`t304_index`], in milliseconds.
pub fn t304_ms(index: u8) -> Result<u16, RrcReconfigurationError> {
    T304_MS_TO_INDEX
        .iter()
        .find(|(_, value)| *value == index)
        .map(|(ms, _)| *ms)
        .ok_or_else(|| {
            RrcReconfigurationError::InvalidFieldValue(format!(
                "t304 index {index} is out of range"
            ))
        })
}

/// `ss-PBCH-BlockPower` broadcast in a `ServingCellConfigCommon`.
///
/// A PHY parameter this simulator does not model (there is no PBCH). -20 dBm is
/// inside the `INTEGER (-60..50)` range and unremarkable; documented here rather
/// than made a knob nothing reads, the same treatment `mib_params` gives the
/// MIB's physical-layer fields.
const SS_PBCH_BLOCK_POWER_DBM: i8 = -20;

/// Parameters for a handover command (issue #107, criterion 3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HandoverCommandParams {
    /// RRC-TransactionIdentifier (0..3)
    pub rrc_transaction_id: u8,
    /// `physCellId` of the target cell (0..1007). This is how the target is
    /// identified on the wire — the simulator's own cell index has no IE, so the
    /// receiving UE resolves the PCI against the cells it can hear.
    pub target_phys_cell_id: u16,
    /// `newUE-Identity`: the C-RNTI the UE is to use in the target cell.
    pub new_ue_identity: u16,
    /// `t304` in milliseconds; must be one of the enumerated values.
    pub t304_ms: u16,
    /// `fullConfig`: the UE releases its stored configuration and applies this one
    /// whole.
    pub full_config: bool,
}

/// What a handover command carried.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HandoverCommandData {
    /// RRC-TransactionIdentifier
    pub rrc_transaction_id: u8,
    /// `physCellId` of the target cell
    pub target_phys_cell_id: u16,
    /// `newUE-Identity` (C-RNTI in the target)
    pub new_ue_identity: u16,
    /// `t304` in milliseconds
    pub t304_ms: u16,
    /// Whether `fullConfig` was set
    pub full_config: bool,
}

/// Builds the `CellGroupConfig` carrying a `reconfigurationWithSync` for a
/// handover to `target_phys_cell_id` (TS 38.331 §6.3.2).
///
/// `reconfigurationWithSync` is what makes an `RRCReconfiguration` a **handover
/// command** rather than a plain reconfiguration: TS 38.331 §5.3.5.5.2 has the UE
/// perform the synchronous reconfiguration — reset MAC, re-establish RLC/PDCP and
/// access the target cell — only when this IE is present.
pub fn build_handover_cell_group_config(
    target_phys_cell_id: u16,
    new_ue_identity: u16,
    t304_ms: u16,
) -> Result<CellGroupConfig, RrcReconfigurationError> {
    if target_phys_cell_id > 1007 {
        return Err(RrcReconfigurationError::InvalidFieldValue(format!(
            "physCellId is INTEGER (0..1007), got {target_phys_cell_id}"
        )));
    }

    let sp_cell_config_common = ServingCellConfigCommon {
        phys_cell_id: Some(PhysCellId(target_phys_cell_id)),
        // The target's frequency configuration is deliberately absent. This
        // simulator's RLS presents one carrier, so a target cell is always on the
        // UE's current frequency and `downlinkConfigCommon` would carry a
        // `frequencyInfoDL` the UE already has. Absent is honest; a fabricated
        // ARFCN would be a value the UE might act on.
        downlink_config_common: None,
        uplink_config_common: None,
        supplementary_uplink_config: None,
        n_timing_advance_offset: None,
        ssb_positions_in_burst: None,
        ssb_periodicity_serving_cell: None,
        // Mandatory PHY fields with no counterpart in a simulator with no PHY,
        // fixed at the values a 30 kHz FR1 cell would use — the same treatment
        // `nextgsim-gnb`'s `mib_params` gives the MIB's physical-layer fields, and
        // documented for the same reason.
        dmrs_type_a_position: ServingCellConfigCommonDmrs_TypeA_Position(
            ServingCellConfigCommonDmrs_TypeA_Position::POS2,
        ),
        lte_crs_to_match_around: None,
        rate_match_pattern_to_add_mod_list: None,
        rate_match_pattern_to_release_list: None,
        ssb_subcarrier_spacing: None,
        tdd_ul_dl_configuration_common: None,
        ss_pbch_block_power: ServingCellConfigCommonSs_PBCH_BlockPower(SS_PBCH_BLOCK_POWER_DBM),
    };

    let sp_cell_config = SpCellConfig {
        serv_cell_index: None,
        reconfiguration_with_sync: Some(ReconfigurationWithSync {
            sp_cell_config_common: Some(sp_cell_config_common),
            new_ue_identity: RNTI_Value(new_ue_identity),
            t304: ReconfigurationWithSyncT304(t304_index(t304_ms)?),
            // No dedicated RACH preamble: this simulator has no RACH, so a
            // contention-free preamble would name a resource that does not exist.
            rach_config_dedicated: None,
        }),
        rlf_timers_and_constants: None,
        rlm_in_sync_out_of_sync_threshold: None,
        sp_cell_config_dedicated: None,
    };

    Ok(CellGroupConfig {
        cell_group_id: CellGroupId(0),
        rlc_bearer_to_add_mod_list: None,
        rlc_bearer_to_release_list: None,
        mac_cell_group_config: None,
        physical_cell_group_config: None,
        sp_cell_config: Some(sp_cell_config),
        s_cell_to_add_mod_list: None,
        s_cell_to_release_list: None,
    })
}

/// Builds the `RrcReconfigurationParams` for a handover command.
pub fn build_handover_command_params(
    params: &HandoverCommandParams,
) -> Result<RrcReconfigurationParams, RrcReconfigurationError> {
    let cgc = build_handover_cell_group_config(
        params.target_phys_cell_id,
        params.new_ue_identity,
        params.t304_ms,
    )?;
    Ok(RrcReconfigurationParams {
        rrc_transaction_id: params.rrc_transaction_id,
        radio_bearer_config: None,
        secondary_cell_group: None,
        master_cell_group: Some(encode_rrc(&cgc)?),
        full_config: params.full_config,
    })
}

/// Builds and encodes a handover command to UPER bytes.
pub fn encode_handover_command(
    params: &HandoverCommandParams,
) -> Result<Vec<u8>, RrcReconfigurationError> {
    encode_rrc_reconfiguration(&build_handover_command_params(params)?)
}

/// Decodes a handover command, i.e. an `RRCReconfiguration` whose
/// `masterCellGroup` carries a `reconfigurationWithSync`.
///
/// Returns `None` for an `RRCReconfiguration` that is **not** a handover command —
/// one with no `masterCellGroup`, or one whose cell group has no
/// `reconfigurationWithSync`. That is not an error: a DRB-establishing
/// reconfiguration is a perfectly good message that simply is not a handover, and
/// a receiver has to be able to tell them apart.
pub fn decode_handover_command(bytes: &[u8]) -> Option<HandoverCommandData> {
    let data = decode_rrc_reconfiguration(bytes).ok()?;
    let master_cell_group = data.master_cell_group.as_ref()?;
    let cgc: CellGroupConfig = decode_rrc(master_cell_group).ok()?;
    let sync = cgc.sp_cell_config?.reconfiguration_with_sync?;
    let target_phys_cell_id = sync.sp_cell_config_common?.phys_cell_id?.0;
    Some(HandoverCommandData {
        rrc_transaction_id: data.rrc_transaction_id,
        target_phys_cell_id,
        new_ue_identity: sync.new_ue_identity.0,
        // A t304 index outside the enumeration cannot occur from a successful
        // decode (the ASN.1 type bounds it), so an unmappable one is treated as
        // "not a usable handover command" rather than silently defaulted to a
        // timer the network never set.
        t304_ms: t304_ms(sync.t304.0).ok()?,
        full_config: data.full_config,
    })
}
