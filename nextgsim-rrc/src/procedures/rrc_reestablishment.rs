//! RRC Reestablishment Procedure
//!
//! Implements the RRC Reestablishment procedure as defined in 3GPP TS 38.331 Section 5.3.7.
//! This procedure is used when the UE needs to recover from radio link failure,
//! handover failure, integrity check failure, or RRC reconfiguration failure.
//!
//! The procedure consists of three messages:
//! 1. `RRCReestablishmentRequest` - UE -> gNB: Request to reestablish RRC connection
//! 2. `RRCReestablishment` - gNB -> UE: Network response with configuration
//! 3. `RRCReestablishmentComplete` - UE -> gNB: Confirmation of reestablishment

use crate::codec::generated::*;
use crate::codec::{decode_rrc, encode_rrc, RrcCodecError, PHYS_CELL_ID_MAX};
use bitvec::prelude::*;
use nextgsim_crypto::nia::{nia1_compute_mac, nia2_compute_mac, nia3_compute_mac};
use thiserror::Error;

/// Errors that can occur during RRC Reestablishment procedures
#[derive(Debug, Error)]
pub enum RrcReestablishmentError {
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

    /// Missing mandatory field
    #[error("Missing mandatory field: {0}")]
    MissingMandatoryField(String),

    /// Invalid field value
    #[error("Invalid field value: {0}")]
    InvalidFieldValue(String),
}

/// Reestablishment cause
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReestablishmentCauseValue {
    /// Reconfiguration failure
    ReconfigurationFailure,
    /// Handover failure
    HandoverFailure,
    /// Other failure
    OtherFailure,
}

/// UE Identity for Reestablishment
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReestablishmentUeIdentity {
    /// C-RNTI value (16 bits)
    pub c_rnti: u16,
    /// Physical Cell ID (0-1007)
    pub phys_cell_id: u16,
    /// ShortMAC-I (16 bits)
    pub short_mac_i: u16,
}

// ============================================================================
// RRC Reestablishment Request
// ============================================================================

/// Parameters for building an RRC Reestablishment Request
#[derive(Debug, Clone)]
pub struct RrcReestablishmentRequestParams {
    /// UE Identity for reestablishment
    pub ue_identity: ReestablishmentUeIdentity,
    /// Reestablishment cause
    pub reestablishment_cause: ReestablishmentCauseValue,
}

/// Parsed RRC Reestablishment Request data
#[derive(Debug, Clone)]
pub struct RrcReestablishmentRequestData {
    /// UE Identity for reestablishment
    pub ue_identity: ReestablishmentUeIdentity,
    /// Reestablishment cause
    pub reestablishment_cause: ReestablishmentCauseValue,
}

/// Build an RRC Reestablishment Request message
pub fn build_rrc_reestablishment_request(
    params: &RrcReestablishmentRequestParams,
) -> Result<UL_CCCH_Message, RrcReestablishmentError> {
    // Build C-RNTI (u16)
    let c_rnti = RNTI_Value(params.ue_identity.c_rnti);

    // Build PhysCellId (0-1007)
    let phys_cell_id = PhysCellId(params.ue_identity.phys_cell_id);

    // Build ShortMAC-I (16 bits)
    let mut short_mac_i_bv: BitVec<u8, Msb0> = BitVec::with_capacity(16);
    for i in (0..16).rev() {
        short_mac_i_bv.push((params.ue_identity.short_mac_i >> i) & 1 == 1);
    }

    let reestab_ue_identity = ReestabUE_Identity {
        c_rnti,
        phys_cell_id,
        short_mac_i: ShortMAC_I(short_mac_i_bv),
    };

    let reestablishment_cause = match params.reestablishment_cause {
        ReestablishmentCauseValue::ReconfigurationFailure => {
            ReestablishmentCause(ReestablishmentCause::RECONFIGURATION_FAILURE)
        }
        ReestablishmentCauseValue::HandoverFailure => {
            ReestablishmentCause(ReestablishmentCause::HANDOVER_FAILURE)
        }
        ReestablishmentCauseValue::OtherFailure => {
            ReestablishmentCause(ReestablishmentCause::OTHER_FAILURE)
        }
    };

    // Build spare bit (1 bit)
    let mut spare_bv: BitVec<u8, Msb0> = BitVec::new();
    spare_bv.push(false);

    let rrc_reestablishment_request_ies = RRCReestablishmentRequest_IEs {
        ue_identity: reestab_ue_identity,
        reestablishment_cause,
        spare: RRCReestablishmentRequest_IEsSpare(spare_bv),
    };

    let rrc_reestablishment_request = RRCReestablishmentRequest {
        rrc_reestablishment_request: rrc_reestablishment_request_ies,
    };

    let message_type = UL_CCCH_MessageType::C1(UL_CCCH_MessageType_c1::RrcReestablishmentRequest(
        rrc_reestablishment_request,
    ));

    Ok(UL_CCCH_Message {
        message: message_type,
    })
}

/// Parse an RRC Reestablishment Request from a UL-CCCH message
pub fn parse_rrc_reestablishment_request(
    msg: &UL_CCCH_Message,
) -> Result<RrcReestablishmentRequestData, RrcReestablishmentError> {
    let request = match &msg.message {
        UL_CCCH_MessageType::C1(c1) => match c1 {
            UL_CCCH_MessageType_c1::RrcReestablishmentRequest(req) => req,
            _ => {
                return Err(RrcReestablishmentError::InvalidMessageType {
                    expected: "RRCReestablishmentRequest".to_string(),
                    actual: "other c1 message".to_string(),
                })
            }
        },
        _ => {
            return Err(RrcReestablishmentError::InvalidMessageType {
                expected: "c1".to_string(),
                actual: "messageClassExtension".to_string(),
            })
        }
    };

    let ies = &request.rrc_reestablishment_request;

    // Parse C-RNTI
    let c_rnti = ies.ue_identity.c_rnti.0;

    // Parse PhysCellId
    let phys_cell_id = ies.ue_identity.phys_cell_id.0;

    // Parse ShortMAC-I
    let short_mac_i = bitvec_to_u16(&ies.ue_identity.short_mac_i.0);

    // Parse reestablishment cause
    let reestablishment_cause = match ies.reestablishment_cause.0 {
        ReestablishmentCause::RECONFIGURATION_FAILURE => {
            ReestablishmentCauseValue::ReconfigurationFailure
        }
        ReestablishmentCause::HANDOVER_FAILURE => ReestablishmentCauseValue::HandoverFailure,
        ReestablishmentCause::OTHER_FAILURE => ReestablishmentCauseValue::OtherFailure,
        _ => ReestablishmentCauseValue::OtherFailure,
    };

    Ok(RrcReestablishmentRequestData {
        ue_identity: ReestablishmentUeIdentity {
            c_rnti,
            phys_cell_id,
            short_mac_i,
        },
        reestablishment_cause,
    })
}

fn bitvec_to_u16(bv: &BitVec<u8, Msb0>) -> u16 {
    let mut value: u16 = 0;
    for bit in bv.iter() {
        value = (value << 1) | (*bit as u16);
    }
    value
}

// ============================================================================
// RRC Reestablishment Complete
// ============================================================================

/// Parameters for building an RRC Reestablishment Complete message
#[derive(Debug, Clone)]
pub struct RrcReestablishmentCompleteParams {
    /// RRC Transaction Identifier (0-3)
    pub rrc_transaction_id: u8,
}

/// Parsed RRC Reestablishment Complete data
#[derive(Debug, Clone)]
pub struct RrcReestablishmentCompleteData {
    /// RRC Transaction Identifier
    pub rrc_transaction_id: u8,
}

/// Build an RRC Reestablishment Complete message
pub fn build_rrc_reestablishment_complete(
    params: &RrcReestablishmentCompleteParams,
) -> Result<UL_DCCH_Message, RrcReestablishmentError> {
    if params.rrc_transaction_id > 3 {
        return Err(RrcReestablishmentError::InvalidFieldValue(
            "RRC Transaction ID must be 0-3".to_string(),
        ));
    }

    let ies = RRCReestablishmentComplete_IEs {
        late_non_critical_extension: None,
        non_critical_extension: None,
    };

    let rrc_reestablishment_complete = RRCReestablishmentComplete {
        rrc_transaction_identifier: RRC_TransactionIdentifier(params.rrc_transaction_id),
        critical_extensions:
            RRCReestablishmentCompleteCriticalExtensions::RrcReestablishmentComplete(ies),
    };

    let message_type = UL_DCCH_MessageType::C1(UL_DCCH_MessageType_c1::RrcReestablishmentComplete(
        rrc_reestablishment_complete,
    ));

    Ok(UL_DCCH_Message {
        message: message_type,
    })
}

/// Parse an RRC Reestablishment Complete from a UL-DCCH message
pub fn parse_rrc_reestablishment_complete(
    msg: &UL_DCCH_Message,
) -> Result<RrcReestablishmentCompleteData, RrcReestablishmentError> {
    let complete = match &msg.message {
        UL_DCCH_MessageType::C1(c1) => match c1 {
            UL_DCCH_MessageType_c1::RrcReestablishmentComplete(c) => c,
            _ => {
                return Err(RrcReestablishmentError::InvalidMessageType {
                    expected: "RRCReestablishmentComplete".to_string(),
                    actual: "other c1 message".to_string(),
                })
            }
        },
        _ => {
            return Err(RrcReestablishmentError::InvalidMessageType {
                expected: "c1".to_string(),
                actual: "messageClassExtension".to_string(),
            })
        }
    };

    match &complete.critical_extensions {
        RRCReestablishmentCompleteCriticalExtensions::RrcReestablishmentComplete(_) => {}
        RRCReestablishmentCompleteCriticalExtensions::CriticalExtensionsFuture(_) => {
            return Err(RrcReestablishmentError::InvalidMessageType {
                expected: "rrcReestablishmentComplete".to_string(),
                actual: "criticalExtensionsFuture".to_string(),
            })
        }
    };

    Ok(RrcReestablishmentCompleteData {
        rrc_transaction_id: complete.rrc_transaction_identifier.0,
    })
}

// ============================================================================
// RRC Reestablishment (network -> UE)
// ============================================================================

/// `NextHopChainingCount ::= INTEGER (0..7)` — TS 38.331 §6.3.2.
pub const NEXT_HOP_CHAINING_COUNT_MAX: u8 = 7;

/// Parameters for building an `RRCReestablishment` (TS 38.331 §6.2.2).
///
/// A **DL-DCCH / SRB1** message, unlike the `RRCReestablishmentRequest` that
/// precedes it on SRB0 / UL-CCCH.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RrcReestablishmentParams {
    /// RRC Transaction Identifier (0-3)
    pub rrc_transaction_id: u8,
    /// `nextHopChainingCount` from the AS security context (0-7). TS 33.501
    /// §6.9.4.1: the UE derives the new KgNB from the NH/NCC pair this names.
    pub next_hop_chaining_count: u8,
}

/// Parsed `RRCReestablishment` data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RrcReestablishmentData {
    /// RRC Transaction Identifier (0-3)
    pub rrc_transaction_id: u8,
    /// `nextHopChainingCount` (0-7)
    pub next_hop_chaining_count: u8,
}

/// Build an `RRCReestablishment` DL-DCCH message.
pub fn build_rrc_reestablishment(
    params: &RrcReestablishmentParams,
) -> Result<DL_DCCH_Message, RrcReestablishmentError> {
    if params.rrc_transaction_id > 3 {
        return Err(RrcReestablishmentError::InvalidFieldValue(format!(
            "rrc-TransactionIdentifier {} is outside INTEGER (0..3)",
            params.rrc_transaction_id
        )));
    }
    if params.next_hop_chaining_count > NEXT_HOP_CHAINING_COUNT_MAX {
        return Err(RrcReestablishmentError::InvalidFieldValue(format!(
            "nextHopChainingCount {} is outside INTEGER (0..{NEXT_HOP_CHAINING_COUNT_MAX})",
            params.next_hop_chaining_count
        )));
    }

    let ies = RRCReestablishment_IEs {
        next_hop_chaining_count: NextHopChainingCount(params.next_hop_chaining_count),
        late_non_critical_extension: None,
        non_critical_extension: None,
    };

    Ok(DL_DCCH_Message {
        message: DL_DCCH_MessageType::C1(DL_DCCH_MessageType_c1::RrcReestablishment(
            RRCReestablishment {
                rrc_transaction_identifier: RRC_TransactionIdentifier(params.rrc_transaction_id),
                critical_extensions: RRCReestablishmentCriticalExtensions::RrcReestablishment(ies),
            },
        )),
    })
}

/// Parse an `RRCReestablishment` from a DL-DCCH message.
pub fn parse_rrc_reestablishment(
    msg: &DL_DCCH_Message,
) -> Result<RrcReestablishmentData, RrcReestablishmentError> {
    let reestablishment = match &msg.message {
        DL_DCCH_MessageType::C1(DL_DCCH_MessageType_c1::RrcReestablishment(r)) => r,
        DL_DCCH_MessageType::C1(_) => {
            return Err(RrcReestablishmentError::InvalidMessageType {
                expected: "RRCReestablishment".to_string(),
                actual: "other c1 message".to_string(),
            })
        }
        _ => {
            return Err(RrcReestablishmentError::InvalidMessageType {
                expected: "c1".to_string(),
                actual: "messageClassExtension".to_string(),
            })
        }
    };

    let ies = match &reestablishment.critical_extensions {
        RRCReestablishmentCriticalExtensions::RrcReestablishment(ies) => ies,
        RRCReestablishmentCriticalExtensions::CriticalExtensionsFuture(_) => {
            return Err(RrcReestablishmentError::InvalidMessageType {
                expected: "rrcReestablishment critical extension".to_string(),
                actual: "criticalExtensionsFuture".to_string(),
            })
        }
    };

    Ok(RrcReestablishmentData {
        rrc_transaction_id: reestablishment.rrc_transaction_identifier.0,
        next_hop_chaining_count: ies.next_hop_chaining_count.0,
    })
}

// ============================================================================
// The simulator's ReestabUE-Identity conventions
// ============================================================================

/// The C-RNTI both ends use for the `ReestabUE-Identity` of TS 38.331 §6.2.2.
///
/// A C-RNTI is allocated by MAC in the Random Access Response (TS 38.321
/// §5.1.4), and this simulator has no MAC layer, so there is no allocation to
/// signal. Both ends therefore use this well-known constant, which is what makes
/// a `(C-RNTI, PCI)` lookup possible at all.
///
/// **Consequence, because it changes what the lookup means:** every UE presents
/// the same C-RNTI, so `(C-RNTI, PCI)` identifies a *set* of contexts rather than
/// one. The `shortMAC-I` verification is what resolves the set, and it can:
/// each UE has a different `K_RRCint`, so at most one candidate reproduces the
/// presented MAC. That is also the order TS 38.331 §5.3.7.2 puts them in —
/// retrieve by identity, then authenticate.
pub const SIMULATED_C_RNTI: u16 = 0x4601;

/// The `physCellId` a cell with NR Cell Identity `nci` presents.
///
/// A real PCI comes from the SSB the UE detects (TS 38.211 §7.4.2), which this
/// simulator does not model — the UE's own `cell_id` is a locally allocated index
/// into the cells it has heard, so it is not a value the network could hold.
///
/// This is the convention that closes that gap: the PCI is the cell's NR Cell
/// Identity reduced into `PhysCellId ::= INTEGER (0..1007)`. SIB1 broadcasts the
/// NCI, so a UE that has decoded a real SIB1 derives the same number the gNB
/// does from its configuration.
///
/// It degrades honestly rather than silently: a UE that never decoded a real SIB1
/// falls back to its fabricated system information, derives a PCI the gNB does
/// not hold, and its re-establishment is answered with an `RRCSetup` — which is
/// precisely §5.3.3.1's behaviour for an identity the network cannot resolve.
pub fn phys_cell_id_from_nci(nci: u64) -> u16 {
    // Modulo rather than a 10-bit mask: the low 10 bits reach 1023 and PhysCellId
    // stops at 1007, so masking would produce values the ASN.1 type rejects.
    (nci % (u64::from(PHYS_CELL_ID_MAX) + 1)) as u16
}

// ============================================================================
// ShortMAC-I (TS 38.331 §5.3.7.4)
// ============================================================================

/// Compute the `shortMAC-I` of TS 38.331 §5.3.7.4.
///
/// > set the shortMAC-I to the 16 least significant bits of the MAC-I calculated
/// > over the ASN.1 encoded VarShortMAC-Input, with the KRRCint key and integrity
/// > protection algorithm used in the source PCell, and with all input bits for
/// > COUNT, BEARER and DIRECTION set to binary ones.
///
/// This lives here, in the crate that owns `VarShortMAC-Input`, because **both
/// ends have to compute the same number**: the UE sets it and the network
/// verifies it. Two implementations of the same formula is a defect waiting to
/// happen, so the UE's `compute_short_mac_i` and the gNB's verification both
/// call this.
///
/// `integrity_alg_id` is the NIA identity (0 = NIA0 … 3 = NIA3); an unknown
/// identity is an error rather than a silent fall back to NIA0, whose MAC is all
/// zeros and would make every verification succeed.
pub fn compute_short_mac_i(
    k_rrc_int: &[u8; 16],
    integrity_alg_id: u8,
    source_c_rnti: u16,
    source_phys_cell_id: u16,
    target_cell_identity: u64,
) -> Result<u16, RrcReestablishmentError> {
    let mut cell_id_bv: BitVec<u8, Msb0> = BitVec::with_capacity(36);
    for i in (0..36).rev() {
        cell_id_bv.push((target_cell_identity >> i) & 1 == 1);
    }

    let input = VarShortMAC_Input {
        source_phys_cell_id: PhysCellId(source_phys_cell_id),
        target_cell_identity: CellIdentity(cell_id_bv),
        source_c_rnti: RNTI_Value(source_c_rnti),
    };
    let encoded = encode_rrc(&input)?;

    mac_i_lsb16(k_rrc_int, integrity_alg_id, &encoded)
}

/// The 16 least significant bits of the MAC-I over `data`, with COUNT, BEARER
/// and DIRECTION all set to binary ones (TS 38.331 §5.3.7.4, §5.3.13.3).
pub fn mac_i_lsb16(
    k_rrc_int: &[u8; 16],
    integrity_alg_id: u8,
    data: &[u8],
) -> Result<u16, RrcReestablishmentError> {
    // COUNT, BEARER and DIRECTION all set to binary ones
    const COUNT: u32 = 0xFFFF_FFFF;
    const BEARER: u8 = 0x1F;
    const DIRECTION: u8 = 0x01;

    let mac = match integrity_alg_id {
        // NIA0 produces an all-zero MAC (TS 33.501 Annex D.1)
        0 => [0u8; 4],
        1 => nia1_compute_mac(COUNT, BEARER, DIRECTION, k_rrc_int, data),
        2 => nia2_compute_mac(COUNT, BEARER, DIRECTION, k_rrc_int, data),
        3 => nia3_compute_mac(COUNT, BEARER, DIRECTION, k_rrc_int, data),
        other => {
            return Err(RrcReestablishmentError::InvalidFieldValue(format!(
                "integrity algorithm identity {other} is not one of NIA0..NIA3"
            )))
        }
    };

    // 16 least significant bits of the 32-bit MAC-I
    Ok(u16::from_be_bytes([mac[2], mac[3]]))
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Build and encode an `RRCReestablishment` to bytes.
pub fn encode_rrc_reestablishment(
    params: &RrcReestablishmentParams,
) -> Result<Vec<u8>, RrcReestablishmentError> {
    let msg = build_rrc_reestablishment(params)?;
    Ok(encode_rrc(&msg)?)
}

/// Decode and parse an `RRCReestablishment` from bytes.
pub fn decode_rrc_reestablishment(
    bytes: &[u8],
) -> Result<RrcReestablishmentData, RrcReestablishmentError> {
    let msg: DL_DCCH_Message = decode_rrc(bytes)?;
    parse_rrc_reestablishment(&msg)
}

/// Build and encode an RRC Reestablishment Request to bytes
pub fn encode_rrc_reestablishment_request(
    params: &RrcReestablishmentRequestParams,
) -> Result<Vec<u8>, RrcReestablishmentError> {
    let msg = build_rrc_reestablishment_request(params)?;
    Ok(encode_rrc(&msg)?)
}

/// Decode and parse an RRC Reestablishment Request from bytes
pub fn decode_rrc_reestablishment_request(
    bytes: &[u8],
) -> Result<RrcReestablishmentRequestData, RrcReestablishmentError> {
    let msg: UL_CCCH_Message = decode_rrc(bytes)?;
    parse_rrc_reestablishment_request(&msg)
}

/// Build and encode an RRC Reestablishment Complete to bytes
pub fn encode_rrc_reestablishment_complete(
    params: &RrcReestablishmentCompleteParams,
) -> Result<Vec<u8>, RrcReestablishmentError> {
    let msg = build_rrc_reestablishment_complete(params)?;
    Ok(encode_rrc(&msg)?)
}

/// Decode and parse an RRC Reestablishment Complete from bytes
pub fn decode_rrc_reestablishment_complete(
    bytes: &[u8],
) -> Result<RrcReestablishmentCompleteData, RrcReestablishmentError> {
    let msg: UL_DCCH_Message = decode_rrc(bytes)?;
    parse_rrc_reestablishment_complete(&msg)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_request_params() -> RrcReestablishmentRequestParams {
        RrcReestablishmentRequestParams {
            ue_identity: ReestablishmentUeIdentity {
                c_rnti: 0x1234,
                phys_cell_id: 100,
                short_mac_i: 0xABCD,
            },
            reestablishment_cause: ReestablishmentCauseValue::HandoverFailure,
        }
    }

    #[test]
    fn test_build_rrc_reestablishment_request() {
        let params = create_test_request_params();
        let result = build_rrc_reestablishment_request(&params);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_rrc_reestablishment_request() {
        let params = create_test_request_params();
        let msg = build_rrc_reestablishment_request(&params).unwrap();
        let data = parse_rrc_reestablishment_request(&msg).unwrap();

        assert_eq!(data.ue_identity.c_rnti, params.ue_identity.c_rnti);
        assert_eq!(
            data.ue_identity.phys_cell_id,
            params.ue_identity.phys_cell_id
        );
        assert_eq!(data.ue_identity.short_mac_i, params.ue_identity.short_mac_i);
        assert_eq!(data.reestablishment_cause, params.reestablishment_cause);
    }

    #[test]
    fn test_encode_decode_rrc_reestablishment_request() {
        let params = create_test_request_params();
        let encoded = encode_rrc_reestablishment_request(&params).expect("Failed to encode");
        assert!(!encoded.is_empty());

        let decoded = decode_rrc_reestablishment_request(&encoded).expect("Failed to decode");
        assert_eq!(decoded.ue_identity.c_rnti, params.ue_identity.c_rnti);
        assert_eq!(decoded.reestablishment_cause, params.reestablishment_cause);
    }

    #[test]
    fn test_build_rrc_reestablishment_complete() {
        let params = RrcReestablishmentCompleteParams {
            rrc_transaction_id: 1,
        };
        let result = build_rrc_reestablishment_complete(&params);
        assert!(result.is_ok());
    }

    #[test]
    fn test_encode_decode_rrc_reestablishment_complete() {
        let params = RrcReestablishmentCompleteParams {
            rrc_transaction_id: 2,
        };
        let encoded = encode_rrc_reestablishment_complete(&params).expect("Failed to encode");
        let decoded = decode_rrc_reestablishment_complete(&encoded).expect("Failed to decode");
        assert_eq!(decoded.rrc_transaction_id, 2);
    }

    #[test]
    fn test_invalid_transaction_id() {
        let params = RrcReestablishmentCompleteParams {
            rrc_transaction_id: 5,
        };
        assert!(build_rrc_reestablishment_complete(&params).is_err());
    }

    #[test]
    fn test_all_reestablishment_causes() {
        let causes = [
            ReestablishmentCauseValue::ReconfigurationFailure,
            ReestablishmentCauseValue::HandoverFailure,
            ReestablishmentCauseValue::OtherFailure,
        ];

        for cause in causes {
            let params = RrcReestablishmentRequestParams {
                ue_identity: ReestablishmentUeIdentity {
                    c_rnti: 0x1000,
                    phys_cell_id: 50,
                    short_mac_i: 0x0001,
                },
                reestablishment_cause: cause,
            };
            let msg = build_rrc_reestablishment_request(&params).unwrap();
            let data = parse_rrc_reestablishment_request(&msg).unwrap();
            assert_eq!(data.reestablishment_cause, cause);
        }
    }

    // ========================================================================
    // RRCReestablishment (DL-DCCH) and the shortMAC-I (issue #37)
    // ========================================================================

    const TEST_K_RRC_INT: [u8; 16] = [
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E,
        0x0F,
    ];

    /// Every `nextHopChainingCount` the ASN.1 type allows survives a byte round
    /// trip, so the reply can carry a real value rather than a hardcoded 0.
    #[test]
    fn an_rrc_reestablishment_round_trips_every_next_hop_chaining_count() {
        for ncc in 0..=NEXT_HOP_CHAINING_COUNT_MAX {
            for tid in 0..=3 {
                let params = RrcReestablishmentParams {
                    rrc_transaction_id: tid,
                    next_hop_chaining_count: ncc,
                };
                let bytes = encode_rrc_reestablishment(&params).expect("encode");
                let decoded = decode_rrc_reestablishment(&bytes).expect("decode");
                assert_eq!(decoded.rrc_transaction_id, tid);
                assert_eq!(decoded.next_hop_chaining_count, ncc);
            }
        }
    }

    #[test]
    fn an_out_of_range_next_hop_chaining_count_is_rejected() {
        let err = encode_rrc_reestablishment(&RrcReestablishmentParams {
            rrc_transaction_id: 0,
            next_hop_chaining_count: 8,
        })
        .expect_err("NextHopChainingCount is INTEGER (0..7)");
        assert!(err.to_string().contains("nextHopChainingCount"));
    }

    /// `RRCReestablishment` (DL-DCCH) and `RRCReestablishmentComplete` (UL-DCCH)
    /// are **not** distinguished by their bytes — they are separate top-level
    /// ASN.1 types and the logical channel is what says which one is on the wire.
    ///
    /// Pinned deliberately, because it is a trap: a dispatcher that tried both
    /// decoders on one buffer would route by luck. Each direction's dispatcher
    /// must only try its own message class, which is what the gNB (UL) and UE (DL)
    /// handlers do.
    #[test]
    fn the_reply_and_the_completion_are_told_apart_by_direction_not_by_bytes() {
        let reply = encode_rrc_reestablishment(&RrcReestablishmentParams {
            rrc_transaction_id: 1,
            next_hop_chaining_count: 3,
        })
        .expect("encode reply");

        assert!(
            decode_rrc_reestablishment_complete(&reply).is_ok(),
            "the DL reply's bytes also parse as the UL completion: the channel, \
             not the content, is the discriminator"
        );
    }

    /// The UE's DL-DCCH dispatcher tries the typed `RRCReestablishment` decode
    /// before its legacy nibble matcher, so none of the gNB's bespoke DL-DCCH
    /// framings may decode as one — otherwise a NAS delivery or a reconfiguration
    /// would be routed into the re-establishment handler.
    #[test]
    fn no_legacy_dl_dcch_framing_decodes_as_an_rrc_reestablishment() {
        // The gNB's bespoke DL-DCCH leading bytes, with plausible payloads.
        let framings: [(&str, Vec<u8>); 6] = [
            ("DLInformationTransfer", vec![0x04, 0x00, 0x00, 0x7E, 0x00]),
            ("RRCRelease", vec![0x0D, 0x01, 0x00]),
            ("RRCReconfiguration", vec![0x00, 0x02, 0x00, 0x00]),
            ("UECapabilityEnquiry", vec![0x06, 0x00, 0x00]),
            ("RRCResume", vec![0x28, 0x00, 0x00]),
            ("SCell reconfiguration", vec![0x0E, 0x01, 0x00, 0x40, 0x02]),
        ];

        for (name, bytes) in framings {
            assert!(
                decode_rrc_reestablishment(&bytes).is_err(),
                "the {name} framing must not be routed to the re-establishment \
                 handler"
            );
        }

        // And the real encoding does route, in the c1-index-3 leading-byte range
        // no legacy framing occupies.
        let reply = encode_rrc_reestablishment(&RrcReestablishmentParams {
            rrc_transaction_id: 0,
            next_hop_chaining_count: 0,
        })
        .expect("encode");
        assert!(decode_rrc_reestablishment(&reply).is_ok());
        assert!(
            (0x18..=0x1F).contains(&reply[0]),
            "DL-DCCH c1 index 3 puts the leading byte in 0x18..=0x1F, got {:#04x}",
            reply[0]
        );
    }

    /// The whole point of the shared derivation: the same inputs give the same
    /// number, which is what lets the network verify what the UE set.
    #[test]
    fn the_short_mac_i_is_a_function_of_its_inputs() {
        let base = compute_short_mac_i(&TEST_K_RRC_INT, 2, 0x4601, 1, 0x10).expect("compute");
        assert_eq!(
            base,
            compute_short_mac_i(&TEST_K_RRC_INT, 2, 0x4601, 1, 0x10).expect("compute"),
            "deterministic"
        );

        // Each input changes it.
        let mut other_key = TEST_K_RRC_INT;
        other_key[0] ^= 0xFF;
        assert_ne!(
            base,
            compute_short_mac_i(&other_key, 2, 0x4601, 1, 0x10).expect("compute"),
            "a different K_RRCint gives a different MAC -- which is what makes \
             candidate disambiguation work"
        );
        assert_ne!(
            base,
            compute_short_mac_i(&TEST_K_RRC_INT, 1, 0x4601, 1, 0x10).expect("compute"),
            "a different integrity algorithm"
        );
        assert_ne!(
            base,
            compute_short_mac_i(&TEST_K_RRC_INT, 2, 0x4602, 1, 0x10).expect("compute"),
            "a different C-RNTI"
        );
        assert_ne!(
            base,
            compute_short_mac_i(&TEST_K_RRC_INT, 2, 0x4601, 2, 0x10).expect("compute"),
            "a different source PCI"
        );
        assert_ne!(
            base,
            compute_short_mac_i(&TEST_K_RRC_INT, 2, 0x4601, 1, 0x11).expect("compute"),
            "a different target cell identity"
        );
    }

    /// NIA0's MAC is all zeros (TS 33.501 Annex D.1), so the shortMAC-I is 0 —
    /// worth pinning, because a verifier that silently fell back to NIA0 would
    /// accept a zero MAC from anyone.
    #[test]
    fn the_short_mac_i_is_zero_under_nia0() {
        assert_eq!(
            compute_short_mac_i(&TEST_K_RRC_INT, 0, 0x4601, 1, 0x10).expect("compute"),
            0
        );
    }

    /// An unknown integrity identity is an error, not a fall back to NIA0.
    #[test]
    fn an_unknown_integrity_algorithm_is_rejected() {
        for alg in [4u8, 255] {
            let err = compute_short_mac_i(&TEST_K_RRC_INT, alg, 0x4601, 1, 0x10)
                .expect_err("only NIA0..NIA3 exist");
            assert!(err.to_string().contains("NIA0..NIA3"), "{err}");
        }
    }

    /// The PCI convention must land inside `PhysCellId ::= INTEGER (0..1007)` for
    /// every NCI, including the ones whose low ten bits exceed 1007 — which a
    /// bitmask would not.
    #[test]
    fn the_derived_phys_cell_id_is_always_encodable() {
        for nci in [0u64, 1, 0x10, 1007, 1008, 1023, 0xF_FFFF_FFFF, u64::MAX] {
            let pci = phys_cell_id_from_nci(nci);
            assert!(
                pci <= PHYS_CELL_ID_MAX,
                "nci {nci:#x} gave physCellId {pci}, past the ASN.1 bound"
            );
            // And it encodes, which is the property that actually matters.
            build_rrc_reestablishment_request(&RrcReestablishmentRequestParams {
                ue_identity: ReestablishmentUeIdentity {
                    c_rnti: SIMULATED_C_RNTI,
                    phys_cell_id: pci,
                    short_mac_i: 0,
                },
                reestablishment_cause: ReestablishmentCauseValue::OtherFailure,
            })
            .unwrap_or_else(|e| panic!("nci {nci:#x} -> physCellId {pci} must encode: {e}"));
        }
    }

    /// A UL-CCCH `RRCReestablishmentRequest` and a UL-CCCH `RRCSetupRequest` must
    /// not be confusable, because that is exactly what went wrong: the bespoke
    /// re-establishment framing `[0x05, …]` decoded as an `RRCSetupRequest`, so
    /// the gNB answered with an `RRCSetup` and never saw the `shortMAC-I`.
    #[test]
    fn a_reestablishment_request_does_not_decode_as_a_setup_request() {
        use crate::procedures::rrc_setup::decode_rrc_setup_request;

        let real = encode_rrc_reestablishment_request(&RrcReestablishmentRequestParams {
            ue_identity: ReestablishmentUeIdentity {
                c_rnti: SIMULATED_C_RNTI,
                phys_cell_id: 16,
                short_mac_i: 0xABCD,
            },
            reestablishment_cause: ReestablishmentCauseValue::OtherFailure,
        })
        .expect("encode");

        assert!(
            decode_rrc_setup_request(&real).is_err(),
            "the real UPER encoding is discriminated by its UL-CCCH c1 index"
        );
        assert!(decode_rrc_reestablishment_request(&real).is_ok());

        // The framing that used to be sent, for the record.
        let bespoke = [0x05u8, 0x03, 0x46, 0x01, 0x00, 0x01, 0xAB, 0xCD];
        assert!(
            decode_rrc_setup_request(&bespoke).is_ok(),
            "the bespoke re-establishment framing decoded as an RRCSetupRequest, \
             which is why the network never ran the procedure"
        );
        assert!(decode_rrc_reestablishment_request(&bespoke).is_err());
    }
}
