//! Overload Start / Overload Stop Procedures
//!
//! Implements the AMF-overload signalling of 3GPP TS 38.413 Section 8.7.7
//! (Overload Start) and Section 8.7.8 (Overload Stop). Both are AMF-initiated;
//! the NG-RAN node only ever receives them, so a parse/decode path is the
//! primary surface. Build/encode helpers are also provided for tests and interop
//! symmetry.
//!
//! OVERLOAD START may carry an AMF Traffic Load Reduction Indication (a
//! percentage) and per-slice overload information; OVERLOAD STOP carries no IEs.

use crate::codec::generated::*;
use crate::codec::{decode_ngap_pdu, encode_ngap_pdu, NgapCodecError};
use thiserror::Error;

/// Errors that can occur while handling Overload messages.
#[derive(Debug, Error)]
pub enum OverloadError {
    /// Underlying APER codec error.
    #[error("Codec error: {0}")]
    CodecError(#[from] NgapCodecError),
    /// The PDU was not the expected Overload procedure.
    #[error("Invalid message type: expected {expected}, got {actual}")]
    InvalidMessageType {
        /// What was expected.
        expected: String,
        /// What was actually decoded.
        actual: String,
    },
}

/// Parsed OVERLOAD START data.
#[derive(Debug, Clone, Default)]
pub struct OverloadData {
    /// AMF Traffic Load Reduction Indication (percentage, 0..=100) if present.
    pub traffic_load_reduction: Option<u8>,
}

/// Parse an OVERLOAD START from an already-decoded NGAP PDU.
pub fn parse_overload_start(pdu: &NGAP_PDU) -> Result<OverloadData, OverloadError> {
    let initiating = match pdu {
        NGAP_PDU::InitiatingMessage(m) => m,
        _ => {
            return Err(OverloadError::InvalidMessageType {
                expected: "InitiatingMessage".to_string(),
                actual: format!("{pdu:?}"),
            })
        }
    };
    let start = match &initiating.value {
        InitiatingMessageValue::Id_OverloadStart(s) => s,
        _ => {
            return Err(OverloadError::InvalidMessageType {
                expected: "OverloadStart".to_string(),
                actual: format!("{:?}", initiating.value),
            })
        }
    };

    let mut traffic_load_reduction = None;
    for ie in &start.protocol_i_es.0 {
        if let OverloadStartProtocolIEs_EntryValue::Id_AMFTrafficLoadReductionIndication(t) =
            &ie.value
        {
            traffic_load_reduction = Some(t.0);
        }
    }
    Ok(OverloadData {
        traffic_load_reduction,
    })
}

/// Decode + parse an OVERLOAD START from bytes.
pub fn decode_overload_start(bytes: &[u8]) -> Result<OverloadData, OverloadError> {
    parse_overload_start(&decode_ngap_pdu(bytes)?)
}

/// Validate that a decoded PDU is an OVERLOAD STOP (which carries no IEs).
pub fn parse_overload_stop(pdu: &NGAP_PDU) -> Result<(), OverloadError> {
    match pdu {
        NGAP_PDU::InitiatingMessage(m)
            if matches!(m.value, InitiatingMessageValue::Id_OverloadStop(_)) =>
        {
            Ok(())
        }
        _ => Err(OverloadError::InvalidMessageType {
            expected: "OverloadStop".to_string(),
            actual: format!("{pdu:?}"),
        }),
    }
}

/// Decode + validate an OVERLOAD STOP from bytes.
pub fn decode_overload_stop(bytes: &[u8]) -> Result<(), OverloadError> {
    parse_overload_stop(&decode_ngap_pdu(bytes)?)
}

/// Returns `true` if the PDU is an OVERLOAD START.
pub fn is_overload_start(pdu: &NGAP_PDU) -> bool {
    matches!(
        pdu,
        NGAP_PDU::InitiatingMessage(m)
            if matches!(m.value, InitiatingMessageValue::Id_OverloadStart(_))
    )
}

/// Returns `true` if the PDU is an OVERLOAD STOP.
pub fn is_overload_stop(pdu: &NGAP_PDU) -> bool {
    matches!(
        pdu,
        NGAP_PDU::InitiatingMessage(m)
            if matches!(m.value, InitiatingMessageValue::Id_OverloadStop(_))
    )
}

/// Build an OVERLOAD START PDU (AMF/peer side; provided for tests and interop).
pub fn build_overload_start(traffic_load_reduction: Option<u8>) -> Result<NGAP_PDU, OverloadError> {
    let mut ies = Vec::new();
    if let Some(t) = traffic_load_reduction {
        ies.push(OverloadStartProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_AMF_TRAFFIC_LOAD_REDUCTION_INDICATION),
            criticality: Criticality(Criticality::IGNORE),
            value: OverloadStartProtocolIEs_EntryValue::Id_AMFTrafficLoadReductionIndication(
                TrafficLoadReductionIndication(t),
            ),
        });
    }
    let start = OverloadStart {
        protocol_i_es: OverloadStartProtocolIEs(ies),
    };
    Ok(NGAP_PDU::InitiatingMessage(InitiatingMessage {
        procedure_code: ProcedureCode(ID_OVERLOAD_START),
        criticality: Criticality(Criticality::IGNORE),
        value: InitiatingMessageValue::Id_OverloadStart(start),
    }))
}

/// Build + encode an OVERLOAD START to bytes.
pub fn encode_overload_start(traffic_load_reduction: Option<u8>) -> Result<Vec<u8>, OverloadError> {
    Ok(encode_ngap_pdu(&build_overload_start(
        traffic_load_reduction,
    )?)?)
}

/// Build an OVERLOAD STOP PDU (AMF/peer side; provided for tests and interop).
pub fn build_overload_stop() -> Result<NGAP_PDU, OverloadError> {
    let stop = OverloadStop {
        protocol_i_es: OverloadStopProtocolIEs(Vec::new()),
    };
    Ok(NGAP_PDU::InitiatingMessage(InitiatingMessage {
        procedure_code: ProcedureCode(ID_OVERLOAD_STOP),
        criticality: Criticality(Criticality::IGNORE),
        value: InitiatingMessageValue::Id_OverloadStop(stop),
    }))
}

/// Build + encode an OVERLOAD STOP to bytes.
pub fn encode_overload_stop() -> Result<Vec<u8>, OverloadError> {
    Ok(encode_ngap_pdu(&build_overload_stop()?)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overload_start_roundtrip_carries_reduction() {
        let bytes = encode_overload_start(Some(80)).expect("encode");
        let pdu = decode_ngap_pdu(&bytes).expect("decode");
        assert!(is_overload_start(&pdu));
        assert!(!is_overload_stop(&pdu));
        let data = parse_overload_start(&pdu).expect("parse");
        assert_eq!(data.traffic_load_reduction, Some(80));
        // Byte-level determinism.
        let re = encode_ngap_pdu(&pdu).expect("re-encode");
        assert_eq!(bytes, re);
    }

    #[test]
    fn overload_start_without_reduction() {
        let data = decode_overload_start(&encode_overload_start(None).unwrap()).unwrap();
        assert_eq!(data.traffic_load_reduction, None);
    }

    #[test]
    fn overload_stop_roundtrip() {
        let bytes = encode_overload_stop().expect("encode");
        let pdu = decode_ngap_pdu(&bytes).expect("decode");
        assert!(is_overload_stop(&pdu));
        assert!(decode_overload_stop(&bytes).is_ok());
        // An OVERLOAD START must not parse as STOP and vice-versa.
        assert!(decode_overload_start(&bytes).is_err());
    }
}
