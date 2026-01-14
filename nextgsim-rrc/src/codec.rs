//! RRC ASN.1 UPER codec
//!
//! This module provides RRC message encoding and decoding using Unaligned PER (UPER).
//! The Rust types are generated from the 3GPP RRC ASN.1 schema at compile time.

use asn1_codecs::uper::UperCodec;
use asn1_codecs::PerCodecData;
use thiserror::Error;

/// Include the generated RRC types from ASN.1 schema
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
    include!(concat!(env!("OUT_DIR"), "/rrc.rs"));
}

// Re-export commonly used types at the module level
pub use generated::*;

/// RRC codec error types
#[derive(Debug, Error)]
pub enum RrcCodecError {
    /// Error during UPER encoding
    #[error("UPER encoding error: {0}")]
    EncodeError(String),

    /// Error during UPER decoding
    #[error("UPER decoding error: {0}")]
    DecodeError(String),
}

/// Encode an RRC message to bytes using UPER
///
/// # Arguments
/// * `msg` - The RRC message to encode (any type implementing UperCodec)
///
/// # Returns
/// * `Ok(Vec<u8>)` - The encoded bytes
/// * `Err(RrcCodecError)` - If encoding fails
pub fn encode_rrc<T: UperCodec>(msg: &T) -> Result<Vec<u8>, RrcCodecError> {
    let mut data = PerCodecData::new_uper();
    msg.uper_encode(&mut data)
        .map_err(|e| RrcCodecError::EncodeError(format!("{:?}", e)))?;
    Ok(data.into_bytes())
}

/// Decode an RRC message from bytes using UPER
///
/// # Arguments
/// * `bytes` - The bytes to decode
///
/// # Returns
/// * `Ok(T)` - The decoded message
/// * `Err(RrcCodecError)` - If decoding fails
pub fn decode_rrc<T: UperCodec<Output = T>>(bytes: &[u8]) -> Result<T, RrcCodecError> {
    let mut data = PerCodecData::from_slice_uper(bytes);
    T::uper_decode(&mut data).map_err(|e| RrcCodecError::DecodeError(format!("{:?}", e)))
}

#[cfg(test)]
mod tests {
    // Note: Specific RRC message tests will be added once the generated types
    // are available and we can identify commonly used message types.
    // The generated code from rrc-15.6.0.asn1 includes all RRC message types.

    #[test]
    fn test_codec_module_compiles() {
        // Basic test to ensure the module compiles with generated code
        // More specific tests will be added for individual RRC messages
        assert!(true);
    }
}
