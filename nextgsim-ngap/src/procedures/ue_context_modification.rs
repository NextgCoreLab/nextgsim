//! UE Context Modification Procedure
//!
//! Implements the UE Context Modification procedure defined in 3GPP TS 38.413
//! Section 8.3.4. This is a mandatory class-1 procedure by which the AMF modifies
//! an established UE context in the NG-RAN node. It can carry:
//!
//! - a new **Security Key** (KgNB) IE that triggers AS re-keying (TS 33.501
//!   Section 6.9.2),
//! - updated **UE Security Capabilities**, and
//! - a replaced **UE Aggregate Maximum Bit Rate** (UE-AMBR).
//!
//! The NG-RAN node answers with a UE CONTEXT MODIFICATION RESPONSE
//! (SuccessfulOutcome) on success, or a UE CONTEXT MODIFICATION FAILURE
//! (UnsuccessfulOutcome) carrying a Cause on error — never with an Error
//! Indication.
//!
//! Only AMF-UE-NGAP-ID and RAN-UE-NGAP-ID are mandatory on the request
//! (TS 38.413 Section 9.2.2.7); the Security Key, UE Security Capabilities and
//! UE-AMBR are all optional and are surfaced as `Option`s below.

use crate::codec::generated::*;
use crate::codec::{decode_ngap_pdu, encode_ngap_pdu, NgapCodecError};
use crate::procedures::initial_context_setup::{
    UeAggregateMaxBitRate, UeSecurityCapabilitiesValue,
};
use crate::procedures::ng_setup::NgSetupFailureCause;
use crate::procedures::ue_context_release::build_cause;
use bitvec::prelude::*;
use thiserror::Error;

/// Cause values for a UE Context Modification Failure. Reuses the shared typed
/// cause hierarchy (see [`NgSetupFailureCause`]).
pub type UeContextModificationFailureCause = NgSetupFailureCause;

/// Errors that can occur while handling UE Context Modification messages.
#[derive(Debug, Error)]
pub enum UeContextModificationError {
    /// Underlying APER codec error.
    #[error("Codec error: {0}")]
    CodecError(#[from] NgapCodecError),
    /// The PDU was not the expected envelope / procedure.
    #[error("Invalid message type: expected {expected}, got {actual}")]
    InvalidMessageType {
        /// What was expected.
        expected: String,
        /// What was actually decoded.
        actual: String,
    },
    /// A mandatory IE was absent.
    #[error("Missing mandatory IE: {0}")]
    MissingMandatoryIe(String),
}

// ============================================================================
// UE Context Modification Request (decoded)
// ============================================================================

/// Parsed UE Context Modification Request data.
#[derive(Debug, Clone)]
pub struct UeContextModificationRequestData {
    /// AMF UE NGAP ID (mandatory).
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID (mandatory).
    pub ran_ue_ngap_id: u32,
    /// Replaced UE Aggregate Maximum Bit Rate (optional).
    pub ue_aggregate_max_bit_rate: Option<UeAggregateMaxBitRate>,
    /// Updated UE Security Capabilities (optional).
    pub ue_security_capabilities: Option<UeSecurityCapabilitiesValue>,
    /// New NR key (KgNB), BIT STRING(256) rendered as 32 bytes (optional).
    /// When present, triggers AS re-keying.
    pub security_key: Option<[u8; 32]>,
}

/// Parse a UE Context Modification Request from an already-decoded NGAP PDU.
pub fn parse_ue_context_modification_request(
    pdu: &NGAP_PDU,
) -> Result<UeContextModificationRequestData, UeContextModificationError> {
    let initiating_message = match pdu {
        NGAP_PDU::InitiatingMessage(msg) => msg,
        _ => {
            return Err(UeContextModificationError::InvalidMessageType {
                expected: "InitiatingMessage".to_string(),
                actual: format!("{pdu:?}"),
            })
        }
    };

    let request = match &initiating_message.value {
        InitiatingMessageValue::Id_UEContextModification(req) => req,
        _ => {
            return Err(UeContextModificationError::InvalidMessageType {
                expected: "UEContextModificationRequest".to_string(),
                actual: format!("{:?}", initiating_message.value),
            })
        }
    };

    let mut amf_ue_ngap_id: Option<u64> = None;
    let mut ran_ue_ngap_id: Option<u32> = None;
    let mut ue_aggregate_max_bit_rate: Option<UeAggregateMaxBitRate> = None;
    let mut ue_security_capabilities: Option<UeSecurityCapabilitiesValue> = None;
    let mut security_key: Option<[u8; 32]> = None;

    for ie in &request.protocol_i_es.0 {
        match &ie.value {
            UEContextModificationRequestProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(id) => {
                amf_ue_ngap_id = Some(id.0);
            }
            UEContextModificationRequestProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(id) => {
                ran_ue_ngap_id = Some(id.0);
            }
            UEContextModificationRequestProtocolIEs_EntryValue::Id_UEAggregateMaximumBitRate(
                rate,
            ) => {
                ue_aggregate_max_bit_rate = Some(parse_ue_aggregate_max_bit_rate(rate));
            }
            UEContextModificationRequestProtocolIEs_EntryValue::Id_UESecurityCapabilities(caps) => {
                ue_security_capabilities = Some(parse_ue_security_capabilities(caps));
            }
            UEContextModificationRequestProtocolIEs_EntryValue::Id_SecurityKey(key) => {
                security_key = Some(parse_security_key(key));
            }
            // Other IEs (New AMF-UE-NGAP-ID, New GUAMI, IndexToRFSP,
            // EmergencyFallbackIndicator, ...) are not modelled here.
            _ => {}
        }
    }

    Ok(UeContextModificationRequestData {
        amf_ue_ngap_id: amf_ue_ngap_id.ok_or_else(|| {
            UeContextModificationError::MissingMandatoryIe("AMF-UE-NGAP-ID".to_string())
        })?,
        ran_ue_ngap_id: ran_ue_ngap_id.ok_or_else(|| {
            UeContextModificationError::MissingMandatoryIe("RAN-UE-NGAP-ID".to_string())
        })?,
        ue_aggregate_max_bit_rate,
        ue_security_capabilities,
        security_key,
    })
}

/// Decode + parse a UE Context Modification Request from bytes.
pub fn decode_ue_context_modification_request(
    bytes: &[u8],
) -> Result<UeContextModificationRequestData, UeContextModificationError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_ue_context_modification_request(&pdu)
}

// ---- IE decode helpers ----------------------------------------------------
// These mirror the private helpers in `initial_context_setup.rs`. They are kept
// private to this module (rather than importing the ICS versions) so that making
// them re-exportable does not introduce `ambiguous_glob_reexports` across the
// procedure modules.

fn parse_ue_aggregate_max_bit_rate(rate: &UEAggregateMaximumBitRate) -> UeAggregateMaxBitRate {
    UeAggregateMaxBitRate {
        dl: rate.ue_aggregate_maximum_bit_rate_dl.0,
        ul: rate.ue_aggregate_maximum_bit_rate_ul.0,
    }
}

fn parse_ue_security_capabilities(caps: &UESecurityCapabilities) -> UeSecurityCapabilitiesValue {
    UeSecurityCapabilitiesValue {
        nr_encryption_algorithms: parse_16bit_bitvec(&caps.n_rencryption_algorithms.0),
        nr_integrity_algorithms: parse_16bit_bitvec(&caps.n_rintegrity_protection_algorithms.0),
        eutra_encryption_algorithms: Some(parse_16bit_bitvec(&caps.eutr_aencryption_algorithms.0)),
        eutra_integrity_algorithms: Some(parse_16bit_bitvec(
            &caps.eutr_aintegrity_protection_algorithms.0,
        )),
    }
}

fn parse_16bit_bitvec(bv: &BitVec<u8, Msb0>) -> u16 {
    if bv.len() >= 16 {
        let mut value: u16 = 0;
        for (i, bit) in bv.iter().take(16).enumerate() {
            if *bit {
                value |= 1 << (15 - i);
            }
        }
        value
    } else {
        0
    }
}

fn parse_security_key(key: &SecurityKey) -> [u8; 32] {
    let mut result = [0u8; 32];
    let raw = key.0.as_raw_slice();
    let len = raw.len().min(32);
    result[..len].copy_from_slice(&raw[..len]);
    result
}

/// Pack a `u16` into a 16-bit, MSB-first BIT STRING (the wire form for the
/// per-family algorithm bitmaps).
fn build_16bit_bitvec_from_u16(value: u16) -> BitVec<u8, Msb0> {
    let mut bv: BitVec<u8, Msb0> = BitVec::with_capacity(16);
    for i in (0..16).rev() {
        bv.push((value >> i) & 1 == 1);
    }
    bv
}

// ============================================================================
// UE Context Modification Request (encode) — the AMF/peer side. The gNB is the
// receiver in production, but a full codec (build + parse) is provided for
// symmetry with the other procedures and for interop/testing.
// ============================================================================

/// Build a UE Context Modification Request PDU from parsed request data.
pub fn build_ue_context_modification_request(
    data: &UeContextModificationRequestData,
) -> Result<NGAP_PDU, UeContextModificationError> {
    let mut protocol_ies = vec![
        UEContextModificationRequestProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_AMF_UE_NGAP_ID),
            criticality: Criticality(Criticality::REJECT),
            value: UEContextModificationRequestProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(
                AMF_UE_NGAP_ID(data.amf_ue_ngap_id),
            ),
        },
        UEContextModificationRequestProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_RAN_UE_NGAP_ID),
            criticality: Criticality(Criticality::REJECT),
            value: UEContextModificationRequestProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(
                RAN_UE_NGAP_ID(data.ran_ue_ngap_id),
            ),
        },
    ];

    if let Some(ambr) = &data.ue_aggregate_max_bit_rate {
        protocol_ies.push(UEContextModificationRequestProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_UE_AGGREGATE_MAXIMUM_BIT_RATE),
            criticality: Criticality(Criticality::IGNORE),
            value: UEContextModificationRequestProtocolIEs_EntryValue::Id_UEAggregateMaximumBitRate(
                UEAggregateMaximumBitRate {
                    ue_aggregate_maximum_bit_rate_dl: BitRate(ambr.dl),
                    ue_aggregate_maximum_bit_rate_ul: BitRate(ambr.ul),
                    ie_extensions: None,
                },
            ),
        });
    }

    if let Some(caps) = &data.ue_security_capabilities {
        protocol_ies.push(UEContextModificationRequestProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_UE_SECURITY_CAPABILITIES),
            criticality: Criticality(Criticality::REJECT),
            value: UEContextModificationRequestProtocolIEs_EntryValue::Id_UESecurityCapabilities(
                UESecurityCapabilities {
                    n_rencryption_algorithms: NRencryptionAlgorithms(build_16bit_bitvec_from_u16(
                        caps.nr_encryption_algorithms,
                    )),
                    n_rintegrity_protection_algorithms: NRintegrityProtectionAlgorithms(
                        build_16bit_bitvec_from_u16(caps.nr_integrity_algorithms),
                    ),
                    eutr_aencryption_algorithms: EUTRAencryptionAlgorithms(
                        build_16bit_bitvec_from_u16(caps.eutra_encryption_algorithms.unwrap_or(0)),
                    ),
                    eutr_aintegrity_protection_algorithms: EUTRAintegrityProtectionAlgorithms(
                        build_16bit_bitvec_from_u16(caps.eutra_integrity_algorithms.unwrap_or(0)),
                    ),
                    ie_extensions: None,
                },
            ),
        });
    }

    if let Some(key) = &data.security_key {
        protocol_ies.push(UEContextModificationRequestProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_SECURITY_KEY),
            criticality: Criticality(Criticality::REJECT),
            value: UEContextModificationRequestProtocolIEs_EntryValue::Id_SecurityKey(SecurityKey(
                BitVec::<u8, Msb0>::from_slice(key),
            )),
        });
    }

    let request = UEContextModificationRequest {
        protocol_i_es: UEContextModificationRequestProtocolIEs(protocol_ies),
    };

    let initiating_message = InitiatingMessage {
        procedure_code: ProcedureCode(ID_UE_CONTEXT_MODIFICATION),
        criticality: Criticality(Criticality::REJECT),
        value: InitiatingMessageValue::Id_UEContextModification(request),
    };

    Ok(NGAP_PDU::InitiatingMessage(initiating_message))
}

/// Build + encode a UE Context Modification Request to bytes.
pub fn encode_ue_context_modification_request(
    data: &UeContextModificationRequestData,
) -> Result<Vec<u8>, UeContextModificationError> {
    let pdu = build_ue_context_modification_request(data)?;
    Ok(encode_ngap_pdu(&pdu)?)
}

// ============================================================================
// UE Context Modification Response (SuccessfulOutcome)
// ============================================================================

/// Build a UE Context Modification Response PDU carrying the mandatory
/// AMF-UE-NGAP-ID and RAN-UE-NGAP-ID.
pub fn build_ue_context_modification_response(
    amf_ue_id: u64,
    ran_ue_id: u32,
) -> Result<NGAP_PDU, UeContextModificationError> {
    let protocol_ies = vec![
        UEContextModificationResponseProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_AMF_UE_NGAP_ID),
            criticality: Criticality(Criticality::IGNORE),
            value: UEContextModificationResponseProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(
                AMF_UE_NGAP_ID(amf_ue_id),
            ),
        },
        UEContextModificationResponseProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_RAN_UE_NGAP_ID),
            criticality: Criticality(Criticality::IGNORE),
            value: UEContextModificationResponseProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(
                RAN_UE_NGAP_ID(ran_ue_id),
            ),
        },
    ];

    let response = UEContextModificationResponse {
        protocol_i_es: UEContextModificationResponseProtocolIEs(protocol_ies),
    };

    let successful_outcome = SuccessfulOutcome {
        procedure_code: ProcedureCode(ID_UE_CONTEXT_MODIFICATION),
        criticality: Criticality(Criticality::REJECT),
        value: SuccessfulOutcomeValue::Id_UEContextModification(response),
    };

    Ok(NGAP_PDU::SuccessfulOutcome(successful_outcome))
}

/// Build + encode a UE Context Modification Response to bytes.
pub fn encode_ue_context_modification_response(
    amf_ue_id: u64,
    ran_ue_id: u32,
) -> Result<Vec<u8>, UeContextModificationError> {
    let pdu = build_ue_context_modification_response(amf_ue_id, ran_ue_id)?;
    Ok(encode_ngap_pdu(&pdu)?)
}

// ============================================================================
// UE Context Modification Failure (UnsuccessfulOutcome)
// ============================================================================

/// Build a UE Context Modification Failure PDU carrying AMF-UE-NGAP-ID,
/// RAN-UE-NGAP-ID and a Cause.
pub fn build_ue_context_modification_failure(
    amf_ue_id: u64,
    ran_ue_id: u32,
    cause: &UeContextModificationFailureCause,
) -> Result<NGAP_PDU, UeContextModificationError> {
    let protocol_ies = vec![
        UEContextModificationFailureProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_AMF_UE_NGAP_ID),
            criticality: Criticality(Criticality::IGNORE),
            value: UEContextModificationFailureProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(
                AMF_UE_NGAP_ID(amf_ue_id),
            ),
        },
        UEContextModificationFailureProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_RAN_UE_NGAP_ID),
            criticality: Criticality(Criticality::IGNORE),
            value: UEContextModificationFailureProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(
                RAN_UE_NGAP_ID(ran_ue_id),
            ),
        },
        UEContextModificationFailureProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_CAUSE),
            criticality: Criticality(Criticality::IGNORE),
            value: UEContextModificationFailureProtocolIEs_EntryValue::Id_Cause(build_cause(cause)),
        },
    ];

    let failure = UEContextModificationFailure {
        protocol_i_es: UEContextModificationFailureProtocolIEs(protocol_ies),
    };

    let unsuccessful_outcome = UnsuccessfulOutcome {
        procedure_code: ProcedureCode(ID_UE_CONTEXT_MODIFICATION),
        criticality: Criticality(Criticality::REJECT),
        value: UnsuccessfulOutcomeValue::Id_UEContextModification(failure),
    };

    Ok(NGAP_PDU::UnsuccessfulOutcome(unsuccessful_outcome))
}

/// Build + encode a UE Context Modification Failure to bytes.
pub fn encode_ue_context_modification_failure(
    amf_ue_id: u64,
    ran_ue_id: u32,
    cause: &UeContextModificationFailureCause,
) -> Result<Vec<u8>, UeContextModificationError> {
    let pdu = build_ue_context_modification_failure(amf_ue_id, ran_ue_id, cause)?;
    Ok(encode_ngap_pdu(&pdu)?)
}

// ============================================================================
// Type predicates
// ============================================================================

/// Returns `true` if the PDU is a UE Context Modification Request.
pub fn is_ue_context_modification_request(pdu: &NGAP_PDU) -> bool {
    matches!(
        pdu,
        NGAP_PDU::InitiatingMessage(msg)
            if matches!(msg.value, InitiatingMessageValue::Id_UEContextModification(_))
    )
}

/// Returns `true` if the PDU is a UE Context Modification Response.
pub fn is_ue_context_modification_response(pdu: &NGAP_PDU) -> bool {
    matches!(
        pdu,
        NGAP_PDU::SuccessfulOutcome(msg)
            if matches!(msg.value, SuccessfulOutcomeValue::Id_UEContextModification(_))
    )
}

/// Returns `true` if the PDU is a UE Context Modification Failure.
pub fn is_ue_context_modification_failure(pdu: &NGAP_PDU) -> bool {
    matches!(
        pdu,
        NGAP_PDU::UnsuccessfulOutcome(msg)
            if matches!(msg.value, UnsuccessfulOutcomeValue::Id_UEContextModification(_))
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::procedures::ng_setup::RadioNetworkCause;

    /// Build a UE Context Modification Request PDU with the mandatory IDs plus an
    /// optional Security Key, UE Security Capabilities and UE-AMBR, via the
    /// public builder.
    fn build_test_request_pdu(
        amf_ue_ngap_id: u64,
        ran_ue_ngap_id: u32,
        security_key: Option<[u8; 32]>,
        caps: Option<(u16, u16)>,
        ambr: Option<(u64, u64)>,
    ) -> NGAP_PDU {
        let data = UeContextModificationRequestData {
            amf_ue_ngap_id,
            ran_ue_ngap_id,
            ue_aggregate_max_bit_rate: ambr.map(|(dl, ul)| UeAggregateMaxBitRate { dl, ul }),
            ue_security_capabilities: caps.map(|(nea, nia)| UeSecurityCapabilitiesValue {
                nr_encryption_algorithms: nea,
                nr_integrity_algorithms: nia,
                eutra_encryption_algorithms: Some(0),
                eutra_integrity_algorithms: Some(0),
            }),
            security_key,
        };
        build_ue_context_modification_request(&data).expect("build request")
    }

    #[test]
    fn request_roundtrip_carries_all_ies() {
        let key: [u8; 32] = core::array::from_fn(|i| i as u8 + 1);
        let pdu = build_test_request_pdu(
            12345,
            7,
            Some(key),
            Some((0xE000, 0xE000)),
            Some((1000, 2000)),
        );

        let encoded = encode_ngap_pdu(&pdu).expect("encode request");
        assert!(!encoded.is_empty());

        let parsed = decode_ue_context_modification_request(&encoded).expect("decode request");
        assert_eq!(parsed.amf_ue_ngap_id, 12345);
        assert_eq!(parsed.ran_ue_ngap_id, 7);
        assert_eq!(parsed.security_key, Some(key));
        let caps = parsed.ue_security_capabilities.expect("caps present");
        assert_eq!(caps.nr_encryption_algorithms, 0xE000);
        assert_eq!(caps.nr_integrity_algorithms, 0xE000);
        let ambr = parsed.ue_aggregate_max_bit_rate.expect("ambr present");
        assert_eq!(ambr.dl, 1000);
        assert_eq!(ambr.ul, 2000);
    }

    #[test]
    fn request_roundtrip_minimal_ids_only() {
        let pdu = build_test_request_pdu(1, 2, None, None, None);
        let encoded = encode_ngap_pdu(&pdu).expect("encode request");
        let parsed = decode_ue_context_modification_request(&encoded).expect("decode request");
        assert_eq!(parsed.amf_ue_ngap_id, 1);
        assert_eq!(parsed.ran_ue_ngap_id, 2);
        assert!(parsed.security_key.is_none());
        assert!(parsed.ue_security_capabilities.is_none());
        assert!(parsed.ue_aggregate_max_bit_rate.is_none());
    }

    #[test]
    fn request_aper_byte_level_roundtrip() {
        let key: [u8; 32] = core::array::from_fn(|i| (0xA0 + i) as u8);
        let pdu = build_test_request_pdu(9, 3, Some(key), Some((0xC000, 0xC000)), None);
        let encoded = encode_ngap_pdu(&pdu).expect("encode");
        let decoded = decode_ngap_pdu(&encoded).expect("decode");
        let re_encoded = encode_ngap_pdu(&decoded).expect("re-encode");
        assert_eq!(
            encoded, re_encoded,
            "APER re-encoding must be deterministic"
        );
    }

    #[test]
    fn request_missing_mandatory_ie_errors() {
        // A request with only RAN-UE-NGAP-ID (no AMF-UE-NGAP-ID) must fail to parse.
        let pdu = NGAP_PDU::InitiatingMessage(InitiatingMessage {
            procedure_code: ProcedureCode(ID_UE_CONTEXT_MODIFICATION),
            criticality: Criticality(Criticality::REJECT),
            value: InitiatingMessageValue::Id_UEContextModification(UEContextModificationRequest {
                protocol_i_es: UEContextModificationRequestProtocolIEs(vec![
                    UEContextModificationRequestProtocolIEs_Entry {
                        id: ProtocolIE_ID(ID_RAN_UE_NGAP_ID),
                        criticality: Criticality(Criticality::REJECT),
                        value:
                            UEContextModificationRequestProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(
                                RAN_UE_NGAP_ID(2),
                            ),
                    },
                ]),
            }),
        });
        let encoded = encode_ngap_pdu(&pdu).expect("encode");
        let err = decode_ue_context_modification_request(&encoded).unwrap_err();
        assert!(matches!(
            err,
            UeContextModificationError::MissingMandatoryIe(_)
        ));
    }

    #[test]
    fn response_roundtrip() {
        let bytes = encode_ue_context_modification_response(42, 5).expect("encode response");
        let pdu = decode_ngap_pdu(&bytes).expect("decode response");
        assert!(is_ue_context_modification_response(&pdu));
        match pdu {
            NGAP_PDU::SuccessfulOutcome(o) => {
                assert_eq!(o.procedure_code.0, ID_UE_CONTEXT_MODIFICATION);
            }
            other => panic!("expected SuccessfulOutcome, got {other:?}"),
        }
    }

    #[test]
    fn failure_roundtrip_carries_cause() {
        let cause = UeContextModificationFailureCause::RadioNetwork(
            RadioNetworkCause::UnknownLocalUeNgapId,
        );
        let bytes = encode_ue_context_modification_failure(42, 5, &cause).expect("encode failure");
        let pdu = decode_ngap_pdu(&bytes).expect("decode failure");
        assert!(is_ue_context_modification_failure(&pdu));
        match pdu {
            NGAP_PDU::UnsuccessfulOutcome(o) => {
                assert_eq!(o.procedure_code.0, ID_UE_CONTEXT_MODIFICATION);
                // Cause IE must be present.
                let has_cause = o.value.clone();
                match has_cause {
                    UnsuccessfulOutcomeValue::Id_UEContextModification(f) => {
                        assert!(f.protocol_i_es.0.iter().any(|ie| matches!(
                            ie.value,
                            UEContextModificationFailureProtocolIEs_EntryValue::Id_Cause(_)
                        )));
                    }
                    other => panic!("expected UEContextModification failure, got {other:?}"),
                }
            }
            other => panic!("expected UnsuccessfulOutcome, got {other:?}"),
        }
    }
}
