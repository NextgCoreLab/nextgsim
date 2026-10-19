//! E-UTRA RRC UPER codec over the types generated from TS 36.331.

use asn1_codecs::uper::UperCodec;
use asn1_codecs::PerCodecData;
use thiserror::Error;

/// The E-UTRA RRC types generated from the TS 36.331 V19.4.0 ASN.1.
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
    include!(concat!(env!("OUT_DIR"), "/lte_rrc.rs"));
}

pub use generated::*;

/// E-UTRA RRC codec errors.
#[derive(Debug, Error)]
pub enum LteRrcCodecError {
    /// UPER encoding failed.
    #[error("UPER encoding error: {0}")]
    EncodeError(String),
    /// UPER decoding failed.
    #[error("UPER decoding error: {0}")]
    DecodeError(String),
}

/// Encode an RRC message with UPER (TS 36.331 §8.1).
///
/// X.691 §11.1 pads the outermost value to a whole octet; `into_bytes` does
/// that, and TS 36.331 §8.5 requires the padding bits to be 0.
pub fn encode<T: UperCodec>(msg: &T) -> Result<Vec<u8>, LteRrcCodecError> {
    let mut data = PerCodecData::new_uper();
    msg.uper_encode(&mut data)
        .map_err(|e| LteRrcCodecError::EncodeError(format!("{e:?}")))?;
    Ok(data.into_bytes())
}

/// Decode an RRC message with UPER.
pub fn decode<T: UperCodec<Output = T>>(bytes: &[u8]) -> Result<T, LteRrcCodecError> {
    let mut data = PerCodecData::from_slice_uper(bytes);
    T::uper_decode(&mut data).map_err(|e| LteRrcCodecError::DecodeError(format!("{e:?}")))
}
