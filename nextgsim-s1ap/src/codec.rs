//! S1AP APER codec over the types generated from TS 36.413.

use asn1_codecs::aper::AperCodec;
use asn1_codecs::PerCodecData;
use thiserror::Error;

/// The S1AP types generated from the TS 36.413 V19.2.0 ASN.1.
#[allow(
    missing_docs,
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    unused,
    non_camel_case_types,
    non_snake_case
)]
pub mod generated {
    include!(concat!(env!("OUT_DIR"), "/s1ap.rs"));
}

pub use generated::*;

/// S1AP codec errors.
#[derive(Debug, Error)]
pub enum S1apCodecError {
    /// APER encoding failed.
    #[error("APER encoding error: {0}")]
    EncodeError(String),
    /// APER decoding failed.
    #[error("APER decoding error: {0}")]
    DecodeError(String),
}

/// Encode an S1AP PDU with APER (TS 36.413 §9.4).
pub fn encode_s1ap_pdu(pdu: &S1AP_PDU) -> Result<Vec<u8>, S1apCodecError> {
    let mut data = PerCodecData::new_aper();
    pdu.aper_encode(&mut data)
        .map_err(|e| S1apCodecError::EncodeError(format!("{e:?}")))?;
    Ok(data.into_bytes())
}

/// Decode an S1AP PDU with APER, failing on any IE the schema does not know.
pub fn decode_s1ap_pdu(bytes: &[u8]) -> Result<S1AP_PDU, S1apCodecError> {
    let mut data = PerCodecData::from_slice_aper(bytes);
    S1AP_PDU::aper_decode(&mut data).map_err(|e| S1apCodecError::DecodeError(format!("{e:?}")))
}

/// A protocol IE (or IE extension) whose id this schema does not know, skipped
/// by [`decode_s1ap_pdu_with_unknown_ies`].
///
/// TS 36.413 §10.3.4.1 makes the receiver's action depend on `criticality`:
/// `reject` -- reject the procedure; `notify` -- ignore the IE and report it;
/// `ignore` -- ignore it. That decision is the caller's, so the IE is returned.
#[derive(Debug, Clone, PartialEq)]
pub struct UnknownIe {
    /// The ProtocolIE-ID (or ProtocolExtensionID) the schema has no entry for.
    pub id: i128,
    /// Its criticality as sent.
    pub criticality: Option<Criticality>,
    /// The IE value's octets, exactly as framed on the wire.
    pub value: Vec<u8>,
}

/// Decode an S1AP PDU, skipping protocol IEs whose id the schema does not know
/// and returning them with their criticality (TS 36.413 §10.3.4, nextgsim #214).
///
/// The caller MUST inspect the returned IEs: one marked `reject` still means the
/// procedure is to be rejected. Only an entry of a protocol-IE or extension
/// container is skippable; an unknown id anywhere else still fails the decode.
pub fn decode_s1ap_pdu_with_unknown_ies(
    bytes: &[u8],
) -> Result<(S1AP_PDU, Vec<UnknownIe>), S1apCodecError> {
    let mut data = PerCodecData::from_slice_aper(bytes);
    data.set_skip_unknown_open_types(true);
    let pdu = S1AP_PDU::aper_decode(&mut data)
        .map_err(|e| S1apCodecError::DecodeError(format!("{e:?}")))?;
    let unknown = data
        .take_unknown_open_types()
        .into_iter()
        .map(|u| UnknownIe {
            id: u.key,
            criticality: u
                .criticality
                .and_then(|c| u8::try_from(c).ok())
                .map(Criticality),
            value: u.value,
        })
        .collect();
    Ok((pdu, unknown))
}

/// Encode any generated S1AP type with APER (standalone IEs, transparent
/// containers).
pub fn encode_aper<T: AperCodec>(value: &T) -> Result<Vec<u8>, S1apCodecError> {
    let mut data = PerCodecData::new_aper();
    value
        .aper_encode(&mut data)
        .map_err(|e| S1apCodecError::EncodeError(format!("{e:?}")))?;
    Ok(data.into_bytes())
}

/// Decode any generated S1AP type with APER.
pub fn decode_aper<T: AperCodec<Output = T>>(bytes: &[u8]) -> Result<T, S1apCodecError> {
    let mut data = PerCodecData::from_slice_aper(bytes);
    T::aper_decode(&mut data).map_err(|e| S1apCodecError::DecodeError(format!("{e:?}")))
}
