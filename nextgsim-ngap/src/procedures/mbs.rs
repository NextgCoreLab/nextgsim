//! MBS (Multicast/Broadcast Services) NGAP procedures, TS 38.413 §9.2.9
//!
//! Decoders for the three AMF-initiated MBS messages the gNB must recognise on
//! the NG-C association, and the encoder for the one outcome it must send back:
//!
//! | Procedure code | Message | Direction |
//! |---|---|---|
//! | 71 `id-MulticastSessionActivation`   | `MulticastSessionActivationRequest` / `...Response` | AMF -> RAN / RAN -> AMF |
//! | 72 `id-MulticastSessionDeactivation` | `MulticastSessionDeactivationRequest` | AMF -> RAN |
//! | 74 `id-MulticastGroupPaging`         | `MulticastGroupPaging` | AMF -> RAN |
//!
//! # Why the `MBS-SessionID` appears twice on the wire
//!
//! `MulticastSessionActivationRequest` carries exactly two mandatory IEs
//! (TS 38.413 §9.2.9.1): `id-MBS-SessionID` (299) and
//! `id-MulticastSessionActivationRequestTransfer` (304), the latter an
//! `OCTET STRING (CONTAINING MulticastSessionActivationRequestTransfer)` whose
//! body restates the same `MBS-SessionID`. That duplication is the spec's shape,
//! not redundancy to collapse: the transfer is the opaque payload the AMF
//! relays from the MB-SMF, so it is encoded independently of the outer IE.
//! Deactivation (§9.2.9.3) is the same shape under IE 305.
//!
//! This module decodes the OUTER IE 299 as the authoritative session id and
//! additionally verifies that the transfer's inner copy agrees, because a
//! mismatch means the AMF relayed a transfer belonging to a different session
//! and the activation must not be applied to the outer one.
//!
//! # What the activation request does NOT carry
//!
//! No S-NSSAI, no TEID, no transport address, no TAC list. Multicast transport
//! is established by the Distribution Setup procedures (69/70) against the
//! MB-UPF, not here. A decoder that expects tunnel information in this message
//! is reading the wrong procedure.
//!
//! # TMGI octet order
//!
//! `MBS-SessionID.tMGI` is `OCTET STRING (SIZE(6))` "encoded as defined in
//! TS 23.003" (TS 38.413 §9.3.1.206). TS 23.003 §30.2 composes the TMGI as MBS
//! Service ID (3 octets) followed by MCC/MNC, i.e. **service id first, PLMN
//! second**. This module therefore exposes the 6 octets verbatim
//! ([`MbsSessionId::tmgi`]) and never reorders them: the bytes that arrive are
//! the bytes that go back out in the response, so a peer's TMGI always
//! round-trips regardless of which half it considers first.

use crate::codec::generated::*;
use crate::codec::{decode_ngap_pdu, encode_ngap_pdu, NgapCodecError};
use thiserror::Error;

/// Errors that can occur during MBS procedures
#[derive(Debug, Error)]
pub enum MbsError {
    /// Codec error during encoding/decoding
    #[error("Codec error: {0}")]
    CodecError(#[from] NgapCodecError),

    /// The PDU is not the MBS message this decoder was asked for
    #[error("Invalid message type: expected {expected}, got {actual}")]
    InvalidMessageType {
        /// Expected message type
        expected: String,
        /// Actual message type received
        actual: String,
    },

    /// A mandatory IE was absent
    #[error("Missing mandatory IE: {0}")]
    MissingMandatoryIe(String),

    /// An IE was present but carried a value this decoder cannot accept
    #[error("Invalid IE value: {0}")]
    InvalidIeValue(String),
}

/// An `MBS-SessionID` (TS 38.413 §9.3.1.206).
///
/// `SEQUENCE { tMGI TMGI, nID NID OPTIONAL, ... }` with
/// `TMGI ::= OCTET STRING (SIZE(6))` and `NID ::= BIT STRING (SIZE(44))`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MbsSessionId {
    /// The 6 TMGI octets exactly as they appeared on the wire.
    ///
    /// Held verbatim rather than split into PLMN and service id so that an
    /// echoed response reproduces the peer's TMGI byte-for-byte. See the module
    /// docs on TMGI octet order.
    pub tmgi: [u8; 6],
    /// SNPN Network Identifier, 44 significant bits, when the session belongs
    /// to a stand-alone non-public network (TS 23.003 §30.2).
    pub nid: Option<u64>,
}

impl MbsSessionId {
    /// Creates a session id from the 6 TMGI octets, with no NID.
    pub fn new(tmgi: [u8; 6]) -> Self {
        Self { tmgi, nid: None }
    }

    /// The TMGI as an uppercase hex string, for logging.
    pub fn tmgi_hex(&self) -> String {
        self.tmgi.iter().map(|b| format!("{b:02X}")).collect()
    }
}

/// A decoded `MulticastSessionActivationRequest` (TS 38.413 §9.2.9.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MulticastSessionActivationRequestData {
    /// The session id from the outer `id-MBS-SessionID` IE (299).
    pub mbs_session_id: MbsSessionId,
}

/// A decoded `MulticastSessionDeactivationRequest` (TS 38.413 §9.2.9.3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MulticastSessionDeactivationRequestData {
    /// The session id from the outer `id-MBS-SessionID` IE (299).
    pub mbs_session_id: MbsSessionId,
}

/// A tracking area the AMF asked the gNB to page in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MbsPagingTai {
    /// PLMN Identity (3 octets)
    pub plmn_identity: [u8; 3],
    /// Tracking Area Code (3 octets)
    pub tac: [u8; 3],
}

/// A decoded `MulticastGroupPaging` (TS 38.413 §9.2.9.5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MulticastGroupPagingData {
    /// The session id whose group is being paged.
    pub mbs_session_id: MbsSessionId,
    /// Flattened `MBS-AreaTAIList` across every `MulticastGroupPagingAreaItem`.
    ///
    /// The per-item grouping carries no information the gNB acts on -- it pages
    /// in a TAI if that TAI appears in any area -- so the areas are flattened
    /// here rather than preserved as a list of lists.
    pub paging_tais: Vec<MbsPagingTai>,
}

/// Converts a generated `MBS-SessionID` into the flat form above.
fn parse_mbs_session_id(id: &MBS_SessionID) -> Result<MbsSessionId, MbsError> {
    let tmgi: [u8; 6] = id.tmgi.0.as_slice().try_into().map_err(|_| {
        // TMGI is OCTET STRING (SIZE(6)); a decoder that produced any other
        // length means the APER size constraint was not enforced upstream.
        MbsError::InvalidIeValue(format!(
            "TMGI is {} octets, expected 6 (TS 38.413 §9.3.1.206)",
            id.tmgi.0.len()
        ))
    })?;

    // NID is BIT STRING (SIZE(44)); pack the bits MSB-first into a u64.
    let nid = id.nid.as_ref().map(|nid| {
        nid.0
            .iter()
            .take(44)
            .fold(0u64, |acc, bit| (acc << 1) | u64::from(*bit))
    });

    Ok(MbsSessionId { tmgi, nid })
}

/// Builds a generated `MBS-SessionID` from the flat form above.
fn build_mbs_session_id(id: &MbsSessionId) -> MBS_SessionID {
    MBS_SessionID {
        tmgi: TMGI(id.tmgi.to_vec()),
        nid: id.nid.map(|nid| {
            // Emit the 44 significant bits MSB-first, matching the parse above.
            let mut bits = bitvec::vec::BitVec::<u8, bitvec::order::Msb0>::with_capacity(44);
            for i in (0..44).rev() {
                bits.push((nid >> i) & 1 == 1);
            }
            NID(bits)
        }),
        ie_extensions: None,
    }
}

/// Decodes the `MBS-SessionID` restated inside an activation or deactivation
/// transfer, so it can be checked against the outer IE.
///
/// Both transfers are
/// `SEQUENCE { mBS-SessionID, iE-Extensions OPTIONAL, ... }` (TS 38.413
/// §9.3.4.31 / §9.3.4.32) carried as an `OCTET STRING (CONTAINING ...)`.
fn parse_activation_transfer(bytes: &[u8]) -> Result<MbsSessionId, MbsError> {
    let transfer: MulticastSessionActivationRequestTransfer = crate::codec::decode_aper(bytes)
        .map_err(|e| {
            MbsError::InvalidIeValue(format!(
                "MulticastSessionActivationRequestTransfer (IE 304) did not decode: {e}"
            ))
        })?;
    parse_mbs_session_id(&transfer.mbs_session_id)
}

/// The deactivation counterpart of [`parse_activation_transfer`] (IE 305).
fn parse_deactivation_transfer(bytes: &[u8]) -> Result<MbsSessionId, MbsError> {
    let transfer: MulticastSessionDeactivationRequestTransfer = crate::codec::decode_aper(bytes)
        .map_err(|e| {
            MbsError::InvalidIeValue(format!(
                "MulticastSessionDeactivationRequestTransfer (IE 305) did not decode: {e}"
            ))
        })?;
    parse_mbs_session_id(&transfer.mbs_session_id)
}

/// Rejects a transfer whose inner session id disagrees with the outer IE 299.
///
/// A mismatch means the AMF relayed an MB-SMF transfer belonging to a different
/// MBS session; applying it to the outer session would activate or tear down
/// the wrong group, so the decode fails rather than silently preferring one.
fn require_matching_session_id(
    outer: &MbsSessionId,
    inner: &MbsSessionId,
    ie: &str,
) -> Result<(), MbsError> {
    if outer.tmgi != inner.tmgi || outer.nid != inner.nid {
        return Err(MbsError::InvalidIeValue(format!(
            "{ie} restates TMGI {} but id-MBS-SessionID (299) carries {}",
            inner.tmgi_hex(),
            outer.tmgi_hex()
        )));
    }
    Ok(())
}

/// Extracts the `MulticastSessionActivationRequest` from a decoded PDU.
pub fn parse_multicast_session_activation_request(
    pdu: &NGAP_PDU,
) -> Result<MulticastSessionActivationRequestData, MbsError> {
    let msg = match pdu {
        NGAP_PDU::InitiatingMessage(m) => match &m.value {
            InitiatingMessageValue::Id_MulticastSessionActivation(req) => req,
            other => {
                return Err(MbsError::InvalidMessageType {
                    expected: "MulticastSessionActivationRequest".to_string(),
                    actual: format!("{other:?}"),
                })
            }
        },
        other => {
            return Err(MbsError::InvalidMessageType {
                expected: "InitiatingMessage".to_string(),
                actual: format!("{other:?}"),
            })
        }
    };

    let mut outer: Option<MbsSessionId> = None;
    let mut inner: Option<MbsSessionId> = None;

    for ie in &msg.protocol_i_es.0 {
        match &ie.value {
            MulticastSessionActivationRequestProtocolIEs_EntryValue::Id_MBS_SessionID(id) => {
                outer = Some(parse_mbs_session_id(id)?);
            }
            MulticastSessionActivationRequestProtocolIEs_EntryValue::Id_MulticastSessionActivationRequestTransfer(t) => {
                inner = Some(parse_activation_transfer(&t.0)?);
            }
        }
    }

    // Both IEs are mandatory and criticality reject (TS 38.413 §9.2.9.1), so an
    // absent one fails the procedure rather than being treated as a default.
    let outer =
        outer.ok_or_else(|| MbsError::MissingMandatoryIe("id-MBS-SessionID (299)".into()))?;
    let inner = inner.ok_or_else(|| {
        MbsError::MissingMandatoryIe("id-MulticastSessionActivationRequestTransfer (304)".into())
    })?;
    require_matching_session_id(
        &outer,
        &inner,
        "id-MulticastSessionActivationRequestTransfer (304)",
    )?;

    Ok(MulticastSessionActivationRequestData {
        mbs_session_id: outer,
    })
}

/// Extracts the `MulticastSessionDeactivationRequest` from a decoded PDU.
pub fn parse_multicast_session_deactivation_request(
    pdu: &NGAP_PDU,
) -> Result<MulticastSessionDeactivationRequestData, MbsError> {
    let msg = match pdu {
        NGAP_PDU::InitiatingMessage(m) => match &m.value {
            InitiatingMessageValue::Id_MulticastSessionDeactivation(req) => req,
            other => {
                return Err(MbsError::InvalidMessageType {
                    expected: "MulticastSessionDeactivationRequest".to_string(),
                    actual: format!("{other:?}"),
                })
            }
        },
        other => {
            return Err(MbsError::InvalidMessageType {
                expected: "InitiatingMessage".to_string(),
                actual: format!("{other:?}"),
            })
        }
    };

    let mut outer: Option<MbsSessionId> = None;
    let mut inner: Option<MbsSessionId> = None;

    for ie in &msg.protocol_i_es.0 {
        match &ie.value {
            MulticastSessionDeactivationRequestProtocolIEs_EntryValue::Id_MBS_SessionID(id) => {
                outer = Some(parse_mbs_session_id(id)?);
            }
            MulticastSessionDeactivationRequestProtocolIEs_EntryValue::Id_MulticastSessionDeactivationRequestTransfer(t) => {
                inner = Some(parse_deactivation_transfer(&t.0)?);
            }
        }
    }

    let outer =
        outer.ok_or_else(|| MbsError::MissingMandatoryIe("id-MBS-SessionID (299)".into()))?;
    let inner = inner.ok_or_else(|| {
        MbsError::MissingMandatoryIe("id-MulticastSessionDeactivationRequestTransfer (305)".into())
    })?;
    require_matching_session_id(
        &outer,
        &inner,
        "id-MulticastSessionDeactivationRequestTransfer (305)",
    )?;

    Ok(MulticastSessionDeactivationRequestData {
        mbs_session_id: outer,
    })
}

/// Extracts the `MulticastGroupPaging` from a decoded PDU.
pub fn parse_multicast_group_paging(pdu: &NGAP_PDU) -> Result<MulticastGroupPagingData, MbsError> {
    let msg = match pdu {
        NGAP_PDU::InitiatingMessage(m) => match &m.value {
            InitiatingMessageValue::Id_MulticastGroupPaging(p) => p,
            other => {
                return Err(MbsError::InvalidMessageType {
                    expected: "MulticastGroupPaging".to_string(),
                    actual: format!("{other:?}"),
                })
            }
        },
        other => {
            return Err(MbsError::InvalidMessageType {
                expected: "InitiatingMessage".to_string(),
                actual: format!("{other:?}"),
            })
        }
    };

    let mut mbs_session_id: Option<MbsSessionId> = None;
    let mut paging_tais = Vec::new();

    for ie in &msg.protocol_i_es.0 {
        match &ie.value {
            MulticastGroupPagingProtocolIEs_EntryValue::Id_MBS_SessionID(id) => {
                mbs_session_id = Some(parse_mbs_session_id(id)?);
            }
            MulticastGroupPagingProtocolIEs_EntryValue::Id_MulticastGroupPagingAreaList(list) => {
                for item in &list.0 {
                    for tai in &item.multicast_group_paging_area.mbs_area_tai_list.0 {
                        paging_tais.push(MbsPagingTai {
                            plmn_identity: tai
                                .plmn_identity
                                .0
                                .as_slice()
                                .try_into()
                                .unwrap_or([0, 0, 0]),
                            tac: tai.tac.0.as_slice().try_into().unwrap_or([0, 0, 0]),
                        });
                    }
                }
            }
            // id-MBS-ServiceArea (298) is optional and criticality ignore
            // (TS 38.413 §9.2.9.5): the service area constrains where the
            // session may be delivered, not where this paging goes, so a gNB
            // that pages per the area list alone is conformant.
            MulticastGroupPagingProtocolIEs_EntryValue::Id_MBS_ServiceArea(_) => {}
        }
    }

    let mbs_session_id = mbs_session_id
        .ok_or_else(|| MbsError::MissingMandatoryIe("id-MBS-SessionID (299)".into()))?;

    Ok(MulticastGroupPagingData {
        mbs_session_id,
        paging_tais,
    })
}

/// Builds a `MulticastSessionActivationResponse` (TS 38.413 §9.2.9.2).
///
/// The response's only mandatory IE is `id-MBS-SessionID` (299), which per the
/// procedure text echoes the session the request named -- that is how the AMF
/// correlates the outcome, since MBS session procedures are not UE-associated
/// and carry no NGAP UE IDs. `id-CriticalityDiagnostics` (19) is optional and
/// omitted on a clean success.
pub fn build_multicast_session_activation_response(
    mbs_session_id: &MbsSessionId,
) -> Result<NGAP_PDU, MbsError> {
    let protocol_ies = vec![MulticastSessionActivationResponseProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_MBS_SESSION_ID),
        criticality: Criticality(Criticality::REJECT),
        value: MulticastSessionActivationResponseProtocolIEs_EntryValue::Id_MBS_SessionID(
            build_mbs_session_id(mbs_session_id),
        ),
    }];

    Ok(NGAP_PDU::SuccessfulOutcome(SuccessfulOutcome {
        procedure_code: ProcedureCode(ID_MULTICAST_SESSION_ACTIVATION),
        criticality: Criticality(Criticality::REJECT),
        value: SuccessfulOutcomeValue::Id_MulticastSessionActivation(
            MulticastSessionActivationResponse {
                protocol_i_es: MulticastSessionActivationResponseProtocolIEs(protocol_ies),
            },
        ),
    }))
}

/// Builds a `MulticastSessionDeactivationResponse` (TS 38.413 §9.2.9.4).
///
/// Same shape as the activation response: `id-MBS-SessionID` echoed back.
pub fn build_multicast_session_deactivation_response(
    mbs_session_id: &MbsSessionId,
) -> Result<NGAP_PDU, MbsError> {
    let protocol_ies = vec![MulticastSessionDeactivationResponseProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_MBS_SESSION_ID),
        criticality: Criticality(Criticality::REJECT),
        value: MulticastSessionDeactivationResponseProtocolIEs_EntryValue::Id_MBS_SessionID(
            build_mbs_session_id(mbs_session_id),
        ),
    }];

    Ok(NGAP_PDU::SuccessfulOutcome(SuccessfulOutcome {
        procedure_code: ProcedureCode(ID_MULTICAST_SESSION_DEACTIVATION),
        criticality: Criticality(Criticality::REJECT),
        value: SuccessfulOutcomeValue::Id_MulticastSessionDeactivation(
            MulticastSessionDeactivationResponse {
                protocol_i_es: MulticastSessionDeactivationResponseProtocolIEs(protocol_ies),
            },
        ),
    }))
}

/// Decodes a `MulticastSessionActivationRequest` from wire bytes.
pub fn decode_multicast_session_activation_request(
    bytes: &[u8],
) -> Result<MulticastSessionActivationRequestData, MbsError> {
    parse_multicast_session_activation_request(&decode_ngap_pdu(bytes)?)
}

/// Decodes a `MulticastSessionDeactivationRequest` from wire bytes.
pub fn decode_multicast_session_deactivation_request(
    bytes: &[u8],
) -> Result<MulticastSessionDeactivationRequestData, MbsError> {
    parse_multicast_session_deactivation_request(&decode_ngap_pdu(bytes)?)
}

/// Decodes a `MulticastGroupPaging` from wire bytes.
pub fn decode_multicast_group_paging(bytes: &[u8]) -> Result<MulticastGroupPagingData, MbsError> {
    parse_multicast_group_paging(&decode_ngap_pdu(bytes)?)
}

/// A decoded `MulticastSessionActivationResponse` (TS 38.413 §9.2.9.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MulticastSessionActivationResponseData {
    /// The session id echoed from the request's `id-MBS-SessionID` (299).
    pub mbs_session_id: MbsSessionId,
}

/// Extracts the `MulticastSessionActivationResponse` from a decoded PDU.
///
/// The AMF side needs this to correlate the outcome; it also lets a test assert
/// the echoed `MBS-SessionID` against the wire bytes rather than against the
/// struct that produced them.
pub fn parse_multicast_session_activation_response(
    pdu: &NGAP_PDU,
) -> Result<MulticastSessionActivationResponseData, MbsError> {
    let msg = match pdu {
        NGAP_PDU::SuccessfulOutcome(o) => match &o.value {
            SuccessfulOutcomeValue::Id_MulticastSessionActivation(resp) => resp,
            other => {
                return Err(MbsError::InvalidMessageType {
                    expected: "MulticastSessionActivationResponse".to_string(),
                    actual: format!("{other:?}"),
                })
            }
        },
        other => {
            return Err(MbsError::InvalidMessageType {
                expected: "SuccessfulOutcome".to_string(),
                actual: format!("{other:?}"),
            })
        }
    };

    let mut mbs_session_id = None;
    for ie in &msg.protocol_i_es.0 {
        if let MulticastSessionActivationResponseProtocolIEs_EntryValue::Id_MBS_SessionID(id) =
            &ie.value
        {
            mbs_session_id = Some(parse_mbs_session_id(id)?);
        }
        // id-CriticalityDiagnostics (19) is optional and criticality ignore; the
        // outcome is a success either way, so it is not surfaced here.
    }

    Ok(MulticastSessionActivationResponseData {
        mbs_session_id: mbs_session_id
            .ok_or_else(|| MbsError::MissingMandatoryIe("id-MBS-SessionID (299)".into()))?,
    })
}

/// Decodes a `MulticastSessionActivationResponse` from wire bytes.
pub fn decode_multicast_session_activation_response(
    bytes: &[u8],
) -> Result<MulticastSessionActivationResponseData, MbsError> {
    parse_multicast_session_activation_response(&decode_ngap_pdu(bytes)?)
}

/// Encodes a `MulticastSessionActivationResponse` to wire bytes.
pub fn encode_multicast_session_activation_response(
    mbs_session_id: &MbsSessionId,
) -> Result<Vec<u8>, MbsError> {
    Ok(encode_ngap_pdu(
        &build_multicast_session_activation_response(mbs_session_id)?,
    )?)
}

/// Encodes a `MulticastSessionDeactivationResponse` to wire bytes.
pub fn encode_multicast_session_deactivation_response(
    mbs_session_id: &MbsSessionId,
) -> Result<Vec<u8>, MbsError> {
    Ok(encode_ngap_pdu(
        &build_multicast_session_deactivation_response(mbs_session_id)?,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `MulticastSessionActivationRequest` as built by a real nextgcore AMF.
    ///
    /// Captured from `nextgcore_ngap::builder::build_multicast_session_activation_request`
    /// with `MbsSessionId::new([0x00, 0xF1, 0x10], [0x01, 0x02, 0x03])` -- i.e.
    /// an independent hand-written APER encoder, not this crate's generated one,
    /// so the vector pins cross-stack interoperability rather than a
    /// self-consistent round trip.
    ///
    /// Byte 0 = 0x00 is the `NGAP-PDU` CHOICE index (initiatingMessage) and
    /// byte 1 = 0x47 = 71 is `id-MulticastSessionActivation`. A builder that
    /// wrote the procedure code as a u16 would put 0x00 in byte 1 and decode as
    /// procedure 0; that this vector decodes as 71 is the guard against it.
    const NEXTGCORE_ACTIVATION_REQUEST: &[u8] = &[
        0x00, 0x47, 0x00, 0x1A, 0x00, 0x00, 0x02, 0x01, 0x2B, 0x00, 0x07, 0x00, 0x00, 0xF1, 0x10,
        0x01, 0x02, 0x03, 0x01, 0x30, 0x00, 0x08, 0x07, 0x00, 0x00, 0xF1, 0x10, 0x01, 0x02, 0x03,
    ];

    /// The deactivation counterpart, same session id (procedure 72 = 0x48).
    const NEXTGCORE_DEACTIVATION_REQUEST: &[u8] = &[
        0x00, 0x48, 0x00, 0x1A, 0x00, 0x00, 0x02, 0x01, 0x2B, 0x00, 0x07, 0x00, 0x00, 0xF1, 0x10,
        0x01, 0x02, 0x03, 0x01, 0x31, 0x00, 0x08, 0x07, 0x00, 0x00, 0xF1, 0x10, 0x01, 0x02, 0x03,
    ];

    /// `MulticastGroupPaging` for the same session over TAC 000001 under PLMN
    /// 00F110 (procedure 74 = 0x4A).
    const NEXTGCORE_GROUP_PAGING: &[u8] = &[
        0x00, 0x4A, 0x40, 0x1B, 0x00, 0x00, 0x02, 0x01, 0x2B, 0x00, 0x07, 0x00, 0x00, 0xF1, 0x10,
        0x01, 0x02, 0x03, 0x01, 0x33, 0x40, 0x09, 0x00, 0x00, 0x00, 0x00, 0xF1, 0x10, 0x00, 0x00,
        0x01,
    ];

    /// The TMGI those vectors carry: PLMN 00F110 then service id 010203.
    const EXPECTED_TMGI: [u8; 6] = [0x00, 0xF1, 0x10, 0x01, 0x02, 0x03];

    /// Criterion 4: the gNB decodes a PDU built by nextgcore's
    /// `build_multicast_session_activation_request` and the TMGI survives.
    ///
    /// Asserts the decoded TMGI equals the encoded one -- a value only
    /// reachable by decoding IE 299 correctly. Before IE 304 was generated this
    /// failed at `decode_ngap_pdu` with `Key 304 Not Found`, so the assertion
    /// also pins the codec fix.
    #[test]
    fn the_tmgi_survives_a_real_amf_activation_request() {
        let data = decode_multicast_session_activation_request(NEXTGCORE_ACTIVATION_REQUEST)
            .expect("a real AMF's activation request must decode");
        assert_eq!(data.mbs_session_id.tmgi, EXPECTED_TMGI);
        assert_eq!(data.mbs_session_id.tmgi_hex(), "00F110010203");
        assert_eq!(data.mbs_session_id.nid, None);
    }

    /// The same for deactivation (procedure 72, IE 305).
    #[test]
    fn the_tmgi_survives_a_real_amf_deactivation_request() {
        let data = decode_multicast_session_deactivation_request(NEXTGCORE_DEACTIVATION_REQUEST)
            .expect("a real AMF's deactivation request must decode");
        assert_eq!(data.mbs_session_id.tmgi, EXPECTED_TMGI);
    }

    /// `MulticastGroupPaging` yields both the session id and the paging area.
    #[test]
    fn a_real_amf_group_paging_yields_its_tmgi_and_tai() {
        let data = decode_multicast_group_paging(NEXTGCORE_GROUP_PAGING)
            .expect("a real AMF's group paging must decode");
        assert_eq!(data.mbs_session_id.tmgi, EXPECTED_TMGI);
        assert_eq!(
            data.paging_tais,
            vec![MbsPagingTai {
                plmn_identity: [0x00, 0xF1, 0x10],
                tac: [0x00, 0x00, 0x01],
            }]
        );
    }

    /// The PDU really is procedure 71 as an initiating message, not procedure 0.
    #[test]
    fn the_activation_request_is_procedure_71() {
        let pdu = decode_ngap_pdu(NEXTGCORE_ACTIVATION_REQUEST).expect("must decode");
        match pdu {
            NGAP_PDU::InitiatingMessage(m) => {
                assert_eq!(m.procedure_code.0, ID_MULTICAST_SESSION_ACTIVATION);
                assert_eq!(m.procedure_code.0, 71);
            }
            other => panic!("expected InitiatingMessage, got {other:?}"),
        }
    }

    /// Criterion 3: the response's `MBS-SessionID` echoes the request's.
    ///
    /// Encodes the response, decodes it back, and asserts the TMGI equals the
    /// one the request carried -- so the echo is verified through the wire
    /// format rather than by inspecting the struct we just built.
    #[test]
    fn the_activation_response_echoes_the_requests_session_id() {
        let request = decode_multicast_session_activation_request(NEXTGCORE_ACTIVATION_REQUEST)
            .expect("must decode");
        let bytes = encode_multicast_session_activation_response(&request.mbs_session_id)
            .expect("must encode");

        let pdu = decode_ngap_pdu(&bytes).expect("our own response must decode");
        match pdu {
            NGAP_PDU::SuccessfulOutcome(o) => {
                assert_eq!(o.procedure_code.0, ID_MULTICAST_SESSION_ACTIVATION);
                match o.value {
                    SuccessfulOutcomeValue::Id_MulticastSessionActivation(resp) => {
                        let echoed = resp
                            .protocol_i_es
                            .0
                            .iter()
                            .find_map(|ie| match &ie.value {
                                MulticastSessionActivationResponseProtocolIEs_EntryValue::Id_MBS_SessionID(id) => Some(id),
                                _ => None,
                            })
                            .expect("id-MBS-SessionID (299) is mandatory in the response");
                        assert_eq!(echoed.tmgi.0, EXPECTED_TMGI.to_vec());
                    }
                    other => panic!("expected MulticastSessionActivationResponse, got {other:?}"),
                }
            }
            other => panic!("expected SuccessfulOutcome, got {other:?}"),
        }
    }

    /// The deactivation response echoes its session id too (procedure 72).
    #[test]
    fn the_deactivation_response_echoes_the_requests_session_id() {
        let request = decode_multicast_session_deactivation_request(NEXTGCORE_DEACTIVATION_REQUEST)
            .expect("must decode");
        let bytes = encode_multicast_session_deactivation_response(&request.mbs_session_id)
            .expect("must encode");
        let pdu = decode_ngap_pdu(&bytes).expect("must decode");
        match pdu {
            NGAP_PDU::SuccessfulOutcome(o) => {
                assert_eq!(o.procedure_code.0, ID_MULTICAST_SESSION_DEACTIVATION);
            }
            other => panic!("expected SuccessfulOutcome, got {other:?}"),
        }
    }

    /// A session id carrying an SNPN NID round-trips through both directions,
    /// including the 44-bit BIT STRING packing (TS 23.003 §30.2).
    #[test]
    fn an_snpn_nid_round_trips_through_the_response() {
        let id = MbsSessionId {
            tmgi: EXPECTED_TMGI,
            // 44 bits, with both the top and bottom bit set so a shift error
            // or a truncation to 32 bits changes the value.
            nid: Some(0x0800_0000_0001),
        };
        let bytes = encode_multicast_session_activation_response(&id).expect("must encode");
        let pdu = decode_ngap_pdu(&bytes).expect("must decode");
        let NGAP_PDU::SuccessfulOutcome(o) = pdu else {
            panic!("expected SuccessfulOutcome");
        };
        let SuccessfulOutcomeValue::Id_MulticastSessionActivation(resp) = o.value else {
            panic!("expected MulticastSessionActivationResponse");
        };
        let echoed = resp
            .protocol_i_es
            .0
            .iter()
            .find_map(|ie| match &ie.value {
                MulticastSessionActivationResponseProtocolIEs_EntryValue::Id_MBS_SessionID(x) => {
                    Some(x)
                }
                _ => None,
            })
            .expect("mandatory IE");
        assert_eq!(parse_mbs_session_id(echoed).expect("valid").nid, id.nid);
    }

    /// A transfer restating a DIFFERENT session id is rejected rather than
    /// silently applied to the outer one.
    ///
    /// Built by taking the real AMF vector and flipping one service-id octet
    /// inside the IE-304 transfer body only, leaving the outer IE 299 intact.
    #[test]
    fn a_transfer_naming_another_session_is_rejected() {
        let mut bytes = NEXTGCORE_ACTIVATION_REQUEST.to_vec();
        // Last octet of the transfer's restated TMGI (service id byte 3).
        let last = bytes.len() - 1;
        assert_eq!(bytes[last], 0x03, "vector layout changed");
        bytes[last] = 0x04;

        let err = decode_multicast_session_activation_request(&bytes)
            .expect_err("a mismatched transfer must not be accepted");
        assert!(
            matches!(err, MbsError::InvalidIeValue(ref m) if m.contains("restates TMGI")),
            "expected a session-id mismatch, got {err:?}"
        );
    }

    /// An activation request missing its mandatory IE 304 fails the procedure.
    ///
    /// Both IEs are criticality reject (TS 38.413 §9.2.9.1), so a partial
    /// message must not be half-applied.
    #[test]
    fn an_activation_request_without_the_transfer_is_rejected() {
        let pdu = NGAP_PDU::InitiatingMessage(InitiatingMessage {
            procedure_code: ProcedureCode(ID_MULTICAST_SESSION_ACTIVATION),
            criticality: Criticality(Criticality::REJECT),
            value: InitiatingMessageValue::Id_MulticastSessionActivation(
                MulticastSessionActivationRequest {
                    protocol_i_es: MulticastSessionActivationRequestProtocolIEs(vec![
                        MulticastSessionActivationRequestProtocolIEs_Entry {
                            id: ProtocolIE_ID(ID_MBS_SESSION_ID),
                            criticality: Criticality(Criticality::REJECT),
                            value: MulticastSessionActivationRequestProtocolIEs_EntryValue::Id_MBS_SessionID(
                                build_mbs_session_id(&MbsSessionId::new(EXPECTED_TMGI)),
                            ),
                        },
                    ]),
                },
            ),
        });

        let err = parse_multicast_session_activation_request(&pdu)
            .expect_err("a request without IE 304 must be rejected");
        assert!(
            matches!(err, MbsError::MissingMandatoryIe(ref m) if m.contains("304")),
            "expected the missing-IE-304 error, got {err:?}"
        );
    }

    /// Asking the activation decoder for a paging PDU is a type error, not a
    /// silent success on the wrong procedure.
    #[test]
    fn the_activation_decoder_rejects_a_paging_pdu() {
        let err = decode_multicast_session_activation_request(NEXTGCORE_GROUP_PAGING)
            .expect_err("procedure 74 is not an activation request");
        assert!(
            matches!(err, MbsError::InvalidMessageType { .. }),
            "got {err:?}"
        );
    }
}
