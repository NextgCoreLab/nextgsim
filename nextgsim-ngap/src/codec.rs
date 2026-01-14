//! NGAP ASN.1 PER codec
//!
//! This module provides NGAP message encoding and decoding using Aligned PER (APER).
//! The Rust types are generated from the 3GPP NGAP ASN.1 schema at compile time.

use asn1_codecs::aper::AperCodec;
use asn1_codecs::PerCodecData;
use thiserror::Error;

/// Include the generated NGAP types from ASN.1 schema
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
    include!(concat!(env!("OUT_DIR"), "/ngap.rs"));
}

// Re-export commonly used types at the module level
pub use generated::*;

/// NGAP codec error types
#[derive(Debug, Error)]
pub enum NgapCodecError {
    /// Error during APER encoding
    #[error("APER encoding error: {0}")]
    EncodeError(String),

    /// Error during APER decoding
    #[error("APER decoding error: {0}")]
    DecodeError(String),
}

/// Encode an NGAP PDU to bytes using APER
///
/// # Arguments
/// * `pdu` - The NGAP PDU to encode
///
/// # Returns
/// * `Ok(Vec<u8>)` - The encoded bytes
/// * `Err(NgapCodecError)` - If encoding fails
pub fn encode_ngap_pdu(pdu: &NGAP_PDU) -> Result<Vec<u8>, NgapCodecError> {
    let mut data = PerCodecData::new_aper();
    pdu.aper_encode(&mut data)
        .map_err(|e| NgapCodecError::EncodeError(format!("{:?}", e)))?;
    Ok(data.into_bytes())
}

/// Decode an NGAP PDU from bytes using APER
///
/// # Arguments
/// * `bytes` - The bytes to decode
///
/// # Returns
/// * `Ok(NGAP_PDU)` - The decoded PDU
/// * `Err(NgapCodecError)` - If decoding fails
pub fn decode_ngap_pdu(bytes: &[u8]) -> Result<NGAP_PDU, NgapCodecError> {
    let mut data = PerCodecData::from_slice_aper(bytes);
    NGAP_PDU::aper_decode(&mut data)
        .map_err(|e| NgapCodecError::DecodeError(format!("{:?}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amf_ue_ngap_id_roundtrip() {
        // Test encoding and decoding of AMF_UE_NGAP_ID (constrained integer 0..2^40-1)
        let id = AMF_UE_NGAP_ID(12345);

        let mut data = PerCodecData::new_aper();
        id.aper_encode(&mut data).expect("Encoding should succeed");
        let encoded = data.into_bytes();

        let mut decode_data = PerCodecData::from_slice_aper(&encoded);
        let decoded =
            AMF_UE_NGAP_ID::aper_decode(&mut decode_data).expect("Decoding should succeed");

        assert_eq!(id.0, decoded.0);
    }

    #[test]
    fn test_plmn_identity_roundtrip() {
        // Test encoding and decoding of PLMNIdentity (OCTET STRING SIZE 3)
        // PLMN for MCC=001, MNC=01 encoded as per 3GPP TS 24.501
        let plmn = PLMNIdentity(vec![0x00, 0xF1, 0x10]);

        let mut data = PerCodecData::new_aper();
        plmn.aper_encode(&mut data).expect("Encoding should succeed");
        let encoded = data.into_bytes();

        let mut decode_data = PerCodecData::from_slice_aper(&encoded);
        let decoded =
            PLMNIdentity::aper_decode(&mut decode_data).expect("Decoding should succeed");

        assert_eq!(plmn.0, decoded.0);
    }
}
