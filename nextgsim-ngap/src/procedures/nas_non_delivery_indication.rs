//! NAS Non Delivery Indication Procedure
//!
//! Implements the NAS Non Delivery Indication procedure defined in 3GPP TS 38.413
//! Section 8.6.4. When the NG-RAN node cannot deliver a NAS message received in a
//! DOWNLINK NAS TRANSPORT, it reports the non-delivery to the AMF by returning the
//! original NAS-PDU together with an appropriate Cause value.
//!
//! This is a gNB-initiated `InitiatingMessage` (procedure code 19). Only a build/
//! encode path is provided: the gNB is always the sender.

use crate::codec::generated::*;
use crate::codec::{encode_ngap_pdu, NgapCodecError};
use crate::procedures::ue_context_release::{build_cause, UeContextReleaseCause};
use thiserror::Error;

/// Errors that can occur while building a NAS Non Delivery Indication.
#[derive(Debug, Error)]
pub enum NasNonDeliveryError {
    /// Underlying APER codec error.
    #[error("Codec error: {0}")]
    CodecError(#[from] NgapCodecError),
}

/// Parameters for a NAS Non Delivery Indication.
#[derive(Debug, Clone)]
pub struct NasNonDeliveryIndicationParams {
    /// AMF UE NGAP ID of the affected UE-associated connection.
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID of the affected UE-associated connection.
    pub ran_ue_ngap_id: u32,
    /// The original, undelivered NAS PDU (echoed back to the AMF).
    pub nas_pdu: Vec<u8>,
    /// Cause of the non-delivery.
    pub cause: UeContextReleaseCause,
}

/// Build a NAS Non Delivery Indication PDU.
pub fn build_nas_non_delivery_indication(
    params: &NasNonDeliveryIndicationParams,
) -> Result<NGAP_PDU, NasNonDeliveryError> {
    let ies = vec![
        NASNonDeliveryIndicationProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_AMF_UE_NGAP_ID),
            criticality: Criticality(Criticality::REJECT),
            value: NASNonDeliveryIndicationProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(
                AMF_UE_NGAP_ID(params.amf_ue_ngap_id),
            ),
        },
        NASNonDeliveryIndicationProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_RAN_UE_NGAP_ID),
            criticality: Criticality(Criticality::REJECT),
            value: NASNonDeliveryIndicationProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(
                RAN_UE_NGAP_ID(params.ran_ue_ngap_id),
            ),
        },
        NASNonDeliveryIndicationProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_NAS_PDU),
            criticality: Criticality(Criticality::REJECT),
            value: NASNonDeliveryIndicationProtocolIEs_EntryValue::Id_NAS_PDU(NAS_PDU(
                params.nas_pdu.clone(),
            )),
        },
        NASNonDeliveryIndicationProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_CAUSE),
            criticality: Criticality(Criticality::IGNORE),
            value: NASNonDeliveryIndicationProtocolIEs_EntryValue::Id_Cause(build_cause(
                &params.cause,
            )),
        },
    ];

    let msg = NASNonDeliveryIndication {
        protocol_i_es: NASNonDeliveryIndicationProtocolIEs(ies),
    };
    let initiating_message = InitiatingMessage {
        procedure_code: ProcedureCode(ID_NAS_NON_DELIVERY_INDICATION),
        criticality: Criticality(Criticality::IGNORE),
        value: InitiatingMessageValue::Id_NASNonDeliveryIndication(msg),
    };
    Ok(NGAP_PDU::InitiatingMessage(initiating_message))
}

/// Build + encode a NAS Non Delivery Indication to bytes.
pub fn encode_nas_non_delivery_indication(
    params: &NasNonDeliveryIndicationParams,
) -> Result<Vec<u8>, NasNonDeliveryError> {
    let pdu = build_nas_non_delivery_indication(params)?;
    Ok(encode_ngap_pdu(&pdu)?)
}

/// Returns `true` if the PDU is a NAS Non Delivery Indication.
pub fn is_nas_non_delivery_indication(pdu: &NGAP_PDU) -> bool {
    matches!(
        pdu,
        NGAP_PDU::InitiatingMessage(msg)
            if matches!(msg.value, InitiatingMessageValue::Id_NASNonDeliveryIndication(_))
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::decode_ngap_pdu;
    use crate::procedures::ng_setup::RadioNetworkCause;

    #[test]
    fn nas_non_delivery_roundtrip() {
        let nas_pdu = vec![0x7e, 0x00, 0x42, 0x01, 0x02];
        let params = NasNonDeliveryIndicationParams {
            amf_ue_ngap_id: 12345,
            ran_ue_ngap_id: 7,
            nas_pdu: nas_pdu.clone(),
            cause: UeContextReleaseCause::RadioNetwork(
                RadioNetworkCause::RadioConnectionWithUeLost,
            ),
        };
        let bytes = encode_nas_non_delivery_indication(&params).expect("encode");
        let pdu = decode_ngap_pdu(&bytes).expect("decode");
        assert!(is_nas_non_delivery_indication(&pdu));

        match pdu {
            NGAP_PDU::InitiatingMessage(m) => {
                assert_eq!(m.procedure_code.0, ID_NAS_NON_DELIVERY_INDICATION);
                let inner = match m.value {
                    InitiatingMessageValue::Id_NASNonDeliveryIndication(x) => x,
                    other => panic!("wrong variant: {other:?}"),
                };
                // All four IEs present, and the NAS-PDU + IDs round-trip.
                assert_eq!(inner.protocol_i_es.0.len(), 4);
                let mut saw_nas = false;
                let mut saw_amf = false;
                let mut saw_ran = false;
                let mut saw_cause = false;
                for ie in &inner.protocol_i_es.0 {
                    match &ie.value {
                        NASNonDeliveryIndicationProtocolIEs_EntryValue::Id_NAS_PDU(p) => {
                            assert_eq!(p.0, nas_pdu);
                            saw_nas = true;
                        }
                        NASNonDeliveryIndicationProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(id) => {
                            assert_eq!(id.0, 12345);
                            saw_amf = true;
                        }
                        NASNonDeliveryIndicationProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(id) => {
                            assert_eq!(id.0, 7);
                            saw_ran = true;
                        }
                        NASNonDeliveryIndicationProtocolIEs_EntryValue::Id_Cause(_) => {
                            saw_cause = true;
                        }
                    }
                }
                assert!(saw_nas && saw_amf && saw_ran && saw_cause);
            }
            other => panic!("expected InitiatingMessage, got {other:?}"),
        }
    }

    #[test]
    fn nas_non_delivery_byte_level_roundtrip() {
        let params = NasNonDeliveryIndicationParams {
            amf_ue_ngap_id: 1,
            ran_ue_ngap_id: 2,
            nas_pdu: vec![0xAA, 0xBB],
            cause: UeContextReleaseCause::RadioNetwork(RadioNetworkCause::UnknownLocalUeNgapId),
        };
        let bytes = encode_nas_non_delivery_indication(&params).expect("encode");
        let decoded = decode_ngap_pdu(&bytes).expect("decode");
        let re = encode_ngap_pdu(&decoded).expect("re-encode");
        assert_eq!(bytes, re, "APER re-encoding must be deterministic");
    }
}
