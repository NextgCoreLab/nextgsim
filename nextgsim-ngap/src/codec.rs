//! NGAP ASN.1 PER codec
//!
//! This module provides NGAP message encoding and decoding using Aligned PER (APER).
//! The Rust types are generated from the 3GPP NGAP ASN.1 schema at compile time.

use asn1_codecs::aper::AperCodec;
use asn1_codecs::PerCodecData;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU16, Ordering};
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
        .map_err(|e| NgapCodecError::EncodeError(format!("{e:?}")))?;
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
        .map_err(|e| NgapCodecError::DecodeError(format!("{e:?}")))
}

// ============================================================================
// Multi-SCTP Stream Management
// ============================================================================

/// SCTP Stream ID type for identifying SCTP streams
///
/// In NGAP, stream 0 is reserved for non-UE-associated signalling.
/// UE-associated signalling should be distributed across other streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StreamId(pub u16);

impl StreamId {
    /// Stream 0 is reserved for non-UE-associated signalling (NG Setup, Reset, etc.)
    pub const NON_UE_ASSOCIATED: StreamId = StreamId(0);

    /// Create a new StreamId
    pub fn new(id: u16) -> Self {
        StreamId(id)
    }

    /// Check if this stream is for non-UE-associated signalling
    pub fn is_non_ue_associated(&self) -> bool {
        self.0 == 0
    }
}

impl std::fmt::Display for StreamId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "StreamId({})", self.0)
    }
}

/// UE context key for stream allocation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UeContextKey {
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
    /// AMF UE NGAP ID (optional, may not be assigned yet)
    pub amf_ue_ngap_id: Option<u64>,
}

/// Stream allocation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamAllocationStrategy {
    /// Round-robin allocation across available streams
    RoundRobin,
    /// Hash-based allocation using UE ID
    HashBased,
    /// Least-loaded stream allocation
    LeastLoaded,
}

/// SCTP stream manager for managing multi-stream NGAP connections
///
/// Supports parallel procedure handling via multiple SCTP streams as defined
/// in 3GPP TS 38.412 (NGAP transport).
#[derive(Debug)]
pub struct SctpStreamManager {
    /// Total number of available streams (excluding stream 0)
    num_streams: u16,
    /// Stream allocation strategy
    strategy: StreamAllocationStrategy,
    /// Map from UE context to assigned stream
    ue_stream_map: HashMap<UeContextKey, StreamId>,
    /// Number of active UEs per stream (for load balancing)
    stream_load: HashMap<StreamId, u32>,
    /// Round-robin counter for stream allocation
    next_stream: AtomicU16,
}

impl SctpStreamManager {
    /// Create a new SCTP stream manager
    ///
    /// # Arguments
    /// * `num_streams` - Total number of SCTP streams (including stream 0)
    /// * `strategy` - Stream allocation strategy
    ///
    /// # Panics
    /// Panics if `num_streams` is less than 2 (need at least stream 0 + 1 UE stream)
    pub fn new(num_streams: u16, strategy: StreamAllocationStrategy) -> Self {
        assert!(
            num_streams >= 2,
            "Need at least 2 streams (stream 0 + 1 UE stream)"
        );

        let mut stream_load = HashMap::new();
        for i in 1..num_streams {
            stream_load.insert(StreamId(i), 0);
        }

        SctpStreamManager {
            num_streams,
            strategy,
            ue_stream_map: HashMap::new(),
            stream_load,
            next_stream: AtomicU16::new(1),
        }
    }

    /// Get the stream ID for non-UE-associated signalling
    pub fn non_ue_stream(&self) -> StreamId {
        StreamId::NON_UE_ASSOCIATED
    }

    /// Allocate a stream for a UE context
    ///
    /// If the UE already has an assigned stream, returns that stream.
    /// Otherwise, allocates a new stream based on the configured strategy.
    pub fn allocate_stream(&mut self, ue_key: &UeContextKey) -> StreamId {
        // Check if UE already has a stream assigned
        if let Some(&stream) = self.ue_stream_map.get(ue_key) {
            return stream;
        }

        // Allocate a new stream based on strategy
        let stream = match self.strategy {
            StreamAllocationStrategy::RoundRobin => self.allocate_round_robin(),
            StreamAllocationStrategy::HashBased => self.allocate_hash_based(ue_key),
            StreamAllocationStrategy::LeastLoaded => self.allocate_least_loaded(),
        };

        // Record the allocation
        self.ue_stream_map.insert(*ue_key, stream);
        if let Some(load) = self.stream_load.get_mut(&stream) {
            *load += 1;
        }

        stream
    }

    /// Release a UE's stream allocation
    pub fn release_stream(&mut self, ue_key: &UeContextKey) {
        if let Some(stream) = self.ue_stream_map.remove(ue_key) {
            if let Some(load) = self.stream_load.get_mut(&stream) {
                *load = load.saturating_sub(1);
            }
        }
    }

    /// Get the stream assigned to a UE, if any
    pub fn get_stream(&self, ue_key: &UeContextKey) -> Option<StreamId> {
        self.ue_stream_map.get(ue_key).copied()
    }

    /// Get the number of active UEs on a stream
    pub fn stream_load(&self, stream: &StreamId) -> u32 {
        self.stream_load.get(stream).copied().unwrap_or(0)
    }

    /// Get the total number of active UE allocations
    pub fn total_allocations(&self) -> usize {
        self.ue_stream_map.len()
    }

    /// Get the number of available UE streams (excluding stream 0)
    pub fn num_ue_streams(&self) -> u16 {
        self.num_streams - 1
    }

    fn allocate_round_robin(&self) -> StreamId {
        let current = self.next_stream.fetch_add(1, Ordering::Relaxed);
        let stream_id = (current % (self.num_streams - 1)) + 1;
        StreamId(stream_id)
    }

    fn allocate_hash_based(&self, ue_key: &UeContextKey) -> StreamId {
        // Simple hash: use ran_ue_ngap_id modulo number of UE streams
        let hash = ue_key.ran_ue_ngap_id;
        let stream_id = (hash % (self.num_streams as u32 - 1)) as u16 + 1;
        StreamId(stream_id)
    }

    fn allocate_least_loaded(&self) -> StreamId {
        let mut min_load = u32::MAX;
        let mut min_stream = StreamId(1);

        for (&stream, &load) in &self.stream_load {
            if load < min_load {
                min_load = load;
                min_stream = stream;
            }
        }

        min_stream
    }
}

/// Encode an NGAP PDU with stream information
///
/// Returns the encoded bytes and the recommended stream ID for sending.
pub fn encode_ngap_pdu_with_stream(
    pdu: &NGAP_PDU,
    stream_manager: &mut SctpStreamManager,
    ue_key: Option<&UeContextKey>,
) -> Result<(Vec<u8>, StreamId), NgapCodecError> {
    let encoded = encode_ngap_pdu(pdu)?;
    let stream = match ue_key {
        Some(key) => stream_manager.allocate_stream(key),
        None => stream_manager.non_ue_stream(),
    };
    Ok((encoded, stream))
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

    // ========================================================================
    // SCTP Stream Management Tests
    // ========================================================================

    #[test]
    fn test_stream_id_non_ue_associated() {
        let stream = StreamId::NON_UE_ASSOCIATED;
        assert_eq!(stream.0, 0);
        assert!(stream.is_non_ue_associated());
    }

    #[test]
    fn test_stream_id_ue_associated() {
        let stream = StreamId::new(1);
        assert_eq!(stream.0, 1);
        assert!(!stream.is_non_ue_associated());
    }

    #[test]
    fn test_stream_manager_creation() {
        let manager = SctpStreamManager::new(4, StreamAllocationStrategy::RoundRobin);
        assert_eq!(manager.num_ue_streams(), 3);
        assert_eq!(manager.total_allocations(), 0);
    }

    #[test]
    #[should_panic]
    fn test_stream_manager_too_few_streams() {
        SctpStreamManager::new(1, StreamAllocationStrategy::RoundRobin);
    }

    #[test]
    fn test_round_robin_allocation() {
        let mut manager = SctpStreamManager::new(4, StreamAllocationStrategy::RoundRobin);

        let ue1 = UeContextKey {
            ran_ue_ngap_id: 1,
            amf_ue_ngap_id: Some(100),
        };
        let ue2 = UeContextKey {
            ran_ue_ngap_id: 2,
            amf_ue_ngap_id: Some(200),
        };
        let ue3 = UeContextKey {
            ran_ue_ngap_id: 3,
            amf_ue_ngap_id: Some(300),
        };

        let s1 = manager.allocate_stream(&ue1);
        let s2 = manager.allocate_stream(&ue2);
        let s3 = manager.allocate_stream(&ue3);

        // All streams should be UE-associated (> 0)
        assert!(!s1.is_non_ue_associated());
        assert!(!s2.is_non_ue_associated());
        assert!(!s3.is_non_ue_associated());

        assert_eq!(manager.total_allocations(), 3);
    }

    #[test]
    fn test_stream_reuse_for_same_ue() {
        let mut manager = SctpStreamManager::new(4, StreamAllocationStrategy::RoundRobin);

        let ue = UeContextKey {
            ran_ue_ngap_id: 1,
            amf_ue_ngap_id: Some(100),
        };

        let s1 = manager.allocate_stream(&ue);
        let s2 = manager.allocate_stream(&ue);

        assert_eq!(s1, s2);
        assert_eq!(manager.total_allocations(), 1);
    }

    #[test]
    fn test_stream_release() {
        let mut manager = SctpStreamManager::new(4, StreamAllocationStrategy::RoundRobin);

        let ue = UeContextKey {
            ran_ue_ngap_id: 1,
            amf_ue_ngap_id: Some(100),
        };

        let stream = manager.allocate_stream(&ue);
        assert_eq!(manager.stream_load(&stream), 1);
        assert_eq!(manager.total_allocations(), 1);

        manager.release_stream(&ue);
        assert_eq!(manager.stream_load(&stream), 0);
        assert_eq!(manager.total_allocations(), 0);
    }

    #[test]
    fn test_hash_based_allocation() {
        let mut manager = SctpStreamManager::new(4, StreamAllocationStrategy::HashBased);

        let ue = UeContextKey {
            ran_ue_ngap_id: 5,
            amf_ue_ngap_id: Some(500),
        };

        let stream = manager.allocate_stream(&ue);
        assert!(!stream.is_non_ue_associated());
        assert!(stream.0 >= 1 && stream.0 <= 3);
    }

    #[test]
    fn test_least_loaded_allocation() {
        let mut manager = SctpStreamManager::new(4, StreamAllocationStrategy::LeastLoaded);

        // Allocate several UEs
        for i in 0..6 {
            let ue = UeContextKey {
                ran_ue_ngap_id: i,
                amf_ue_ngap_id: Some(i as u64 * 100),
            };
            let stream = manager.allocate_stream(&ue);
            assert!(!stream.is_non_ue_associated());
        }

        assert_eq!(manager.total_allocations(), 6);

        // Each stream should have approximately 2 UEs (6 UEs / 3 streams)
        let mut total_load = 0;
        for i in 1..4 {
            total_load += manager.stream_load(&StreamId(i));
        }
        assert_eq!(total_load, 6);
    }

    #[test]
    fn test_get_stream() {
        let mut manager = SctpStreamManager::new(4, StreamAllocationStrategy::RoundRobin);

        let ue = UeContextKey {
            ran_ue_ngap_id: 1,
            amf_ue_ngap_id: Some(100),
        };

        assert!(manager.get_stream(&ue).is_none());

        let stream = manager.allocate_stream(&ue);
        assert_eq!(manager.get_stream(&ue), Some(stream));
    }

    #[test]
    fn test_non_ue_stream() {
        let manager = SctpStreamManager::new(4, StreamAllocationStrategy::RoundRobin);
        let stream = manager.non_ue_stream();
        assert_eq!(stream, StreamId::NON_UE_ASSOCIATED);
        assert!(stream.is_non_ue_associated());
    }
}

// ============================================================================
// 6G Extension Procedure Codes (Rel-20 Research)
// ============================================================================

/// 6G NGAP extension procedure codes.
///
/// These are provisional procedure codes for 6G extensions not yet
/// standardized in 3GPP TS 38.413. They use the vendor extension range
/// to avoid conflicts with standard NGAP procedures.
pub mod sixg_procedure_codes {
    /// ISAC Measurement Configuration (gNB ↔ AMF)
    pub const ISAC_MEASUREMENT_CONFIG: u16 = 0xF000;
    /// ISAC Measurement Report (gNB → AMF)
    pub const ISAC_MEASUREMENT_REPORT: u16 = 0xF001;
    /// AI Model Transfer (AMF ↔ gNB)
    pub const AI_MODEL_TRANSFER: u16 = 0xF010;
    /// AI Inference Request (AMF → gNB)
    pub const AI_INFERENCE_REQUEST: u16 = 0xF011;
    /// AI Inference Response (gNB → AMF)
    pub const AI_INFERENCE_RESPONSE: u16 = 0xF012;
    /// NTN Timing Info Update (AMF → gNB)
    pub const NTN_TIMING_INFO_UPDATE: u16 = 0xF020;
    /// NTN Cell Info (gNB → AMF)
    pub const NTN_CELL_INFO: u16 = 0xF021;
}

/// 6G RRC extension message types (embedded in RRC non-critical extensions).
pub mod sixg_rrc_extensions {
    /// AI/ML model configuration in RRCReconfiguration
    pub const AI_ML_CONFIG: u16 = 0xF100;
    /// ISAC measurement configuration in RRCReconfiguration
    pub const ISAC_MEAS_CONFIG: u16 = 0xF101;
    /// NTN timing advance update in RRCReconfiguration
    pub const NTN_TIMING_ADVANCE: u16 = 0xF102;
    /// Sub-THz band configuration in RRCReconfiguration
    pub const SUB_THZ_BAND_CONFIG: u16 = 0xF103;
}

/// Wrap a 6G extension payload into an NGAP IE container format.
///
/// Format: procedure_code (2 bytes) + length (4 bytes) + payload
pub fn encode_sixg_ngap_extension(procedure_code: u16, payload: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(6 + payload.len());
    result.extend_from_slice(&procedure_code.to_be_bytes());
    result.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    result.extend_from_slice(payload);
    result
}

/// Decode a 6G extension from an NGAP IE container format.
///
/// Returns (procedure_code, payload).
pub fn decode_sixg_ngap_extension(data: &[u8]) -> Result<(u16, Vec<u8>), NgapCodecError> {
    if data.len() < 6 {
        return Err(NgapCodecError::DecodeError(
            "6G extension too short".to_string(),
        ));
    }
    let procedure_code = u16::from_be_bytes([data[0], data[1]]);
    let length = u32::from_be_bytes([data[2], data[3], data[4], data[5]]) as usize;
    if data.len() < 6 + length {
        return Err(NgapCodecError::DecodeError(
            "6G extension payload truncated".to_string(),
        ));
    }
    Ok((procedure_code, data[6..6 + length].to_vec()))
}

/// Wrap a 6G extension into an RRC non-critical extension format.
///
/// Format: extension_type (2 bytes) + length (2 bytes) + payload
pub fn encode_sixg_rrc_extension(extension_type: u16, payload: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(4 + payload.len());
    result.extend_from_slice(&extension_type.to_be_bytes());
    result.extend_from_slice(&(payload.len() as u16).to_be_bytes());
    result.extend_from_slice(payload);
    result
}

/// Decode a 6G extension from an RRC non-critical extension format.
///
/// Returns (extension_type, payload).
pub fn decode_sixg_rrc_extension(data: &[u8]) -> Result<(u16, Vec<u8>), NgapCodecError> {
    if data.len() < 4 {
        return Err(NgapCodecError::DecodeError(
            "6G RRC extension too short".to_string(),
        ));
    }
    let extension_type = u16::from_be_bytes([data[0], data[1]]);
    let length = u16::from_be_bytes([data[2], data[3]]) as usize;
    if data.len() < 4 + length {
        return Err(NgapCodecError::DecodeError(
            "6G RRC extension payload truncated".to_string(),
        ));
    }
    Ok((extension_type, data[4..4 + length].to_vec()))
}
