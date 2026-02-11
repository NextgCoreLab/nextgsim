//! Type 4 Information Elements (variable length, TLV)
//!
//! Type 4 IEs have a variable length with a 1-byte length field (TLV format).
//! They are used for encoding/decoding NAS message fields.
//!
//! Based on 3GPP TS 24.501 specification.

use bytes::{Buf, BufMut};
use thiserror::Error;

use crate::codec::{CodecError, CodecResult, NasDecode, NasEncode};

/// Error type for Type 4 IE decoding
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum Ie4Error {
    /// Buffer too short for decoding
    #[error("Buffer too short: expected at least {expected} bytes, got {actual}")]
    BufferTooShort {
        /// Expected minimum bytes
        expected: usize,
        /// Actual bytes available
        actual: usize,
    },
    /// Invalid value for the IE type
    #[error("Invalid value: {0}")]
    InvalidValue(String),
}


// ============================================================================
// UE Security Capability IE (3GPP TS 24.501 Section 9.11.3.54)
// ============================================================================

/// UE Security Capability IE (Type 4, TLV, min 4 bytes value)
///
/// Contains the UE's security algorithm capabilities, including
/// supported 5GS encryption algorithms (EA0-EA7) and 5GS integrity
/// algorithms (IA0-IA7).
///
/// 3GPP TS 24.501 Section 9.11.3.54
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IeUeSecurityCapability {
    /// 5GS encryption algorithms supported (EA0 = bit 7, EA7 = bit 0)
    pub ea: u8,
    /// 5GS integrity algorithms supported (IA0 = bit 7, IA7 = bit 0)
    pub ia: u8,
    /// EPS encryption algorithms supported (optional, EEA0 = bit 7, EEA7 = bit 0)
    pub eea: Option<u8>,
    /// EPS integrity algorithms supported (optional, EIA0 = bit 7, EIA7 = bit 0)
    pub eia: Option<u8>,
}

impl Default for IeUeSecurityCapability {
    fn default() -> Self {
        Self {
            ea: 0x80,  // EA0 supported by default
            ia: 0x80,  // IA0 supported by default
            eea: None,
            eia: None,
        }
    }
}

impl IeUeSecurityCapability {
    /// Create a new UE Security Capability IE
    pub fn new(ea: u8, ia: u8) -> Self {
        Self {
            ea,
            ia,
            eea: None,
            eia: None,
        }
    }

    /// Create with EPS capabilities
    pub fn with_eps(ea: u8, ia: u8, eea: u8, eia: u8) -> Self {
        Self {
            ea,
            ia,
            eea: Some(eea),
            eia: Some(eia),
        }
    }

    /// Check if a specific 5GS encryption algorithm is supported (0=EA0, 7=EA7)
    pub fn supports_ea(&self, alg: u8) -> bool {
        if alg > 7 {
            return false;
        }
        (self.ea >> (7 - alg)) & 0x01 == 1
    }

    /// Check if a specific 5GS integrity algorithm is supported (0=IA0, 7=IA7)
    pub fn supports_ia(&self, alg: u8) -> bool {
        if alg > 7 {
            return false;
        }
        (self.ia >> (7 - alg)) & 0x01 == 1
    }

    /// Decode from bytes (without IEI, with 1-byte length)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, Ie4Error> {
        if buf.remaining() < 1 {
            return Err(Ie4Error::BufferTooShort {
                expected: 1,
                actual: buf.remaining(),
            });
        }

        let length = buf.get_u8() as usize;
        if length < 2 {
            return Err(Ie4Error::BufferTooShort {
                expected: 2,
                actual: length,
            });
        }
        if buf.remaining() < length {
            return Err(Ie4Error::BufferTooShort {
                expected: length,
                actual: buf.remaining(),
            });
        }

        let ea = buf.get_u8();
        let ia = buf.get_u8();

        let mut cap = Self::new(ea, ia);

        if length >= 3 {
            cap.eea = Some(buf.get_u8());
        }
        if length >= 4 {
            cap.eia = Some(buf.get_u8());
        }
        // Skip any remaining extension bytes
        if length > 4 {
            buf.advance(length - 4);
        }

        Ok(cap)
    }

    /// Encode to bytes (without IEI, with 1-byte length)
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        let length = if self.eia.is_some() {
            4
        } else if self.eea.is_some() {
            3
        } else {
            2
        };

        buf.put_u8(length);
        buf.put_u8(self.ea);
        buf.put_u8(self.ia);

        if let Some(eea) = self.eea {
            buf.put_u8(eea);
        }
        if let Some(eia) = self.eia {
            buf.put_u8(eia);
        }
    }

    /// Get the encoded length (including 1-byte length field)
    pub fn encoded_len(&self) -> usize {
        let value_len = if self.eia.is_some() {
            4
        } else if self.eea.is_some() {
            3
        } else {
            2
        };
        1 + value_len // length byte + value
    }
}

impl NasEncode for IeUeSecurityCapability {
    fn nas_encode<B: BufMut>(&self, buf: &mut B) -> CodecResult<()> {
        self.encode(buf);
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        self.encoded_len()
    }
}

impl NasDecode for IeUeSecurityCapability {
    fn nas_decode<B: Buf>(buf: &mut B) -> CodecResult<Self> {
        IeUeSecurityCapability::decode(buf).map_err(|e| CodecError::InvalidValue(e.to_string()))
    }
}


// ============================================================================
// AI/ML Capability IE (6G extension)
// ============================================================================

/// AI/ML Capability IE (Type 4, TLV)
///
/// Indicates the UE's AI/ML capabilities for 6G systems.
/// This is a 6G extension IE not yet standardized in 3GPP TS 24.501.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct IeAiMlCapability {
    /// AI/ML capability flags
    ///
    /// Bit 0: Federated learning supported
    /// Bit 1: Model inference at UE supported
    /// Bit 2: Model training at UE supported
    /// Bit 3: AI/ML-based beam management supported
    /// Bit 4: AI/ML-based CSI feedback supported
    /// Bit 5: AI/ML-based positioning supported
    /// Bits 6-7: Reserved
    pub capability_flags: u8,
    /// Maximum model size supported (in KB, 0 = not applicable)
    pub max_model_size_kb: u16,
}

impl IeAiMlCapability {
    /// Create a new AI/ML Capability IE
    pub fn new(capability_flags: u8, max_model_size_kb: u16) -> Self {
        Self {
            capability_flags,
            max_model_size_kb,
        }
    }

    /// Check if federated learning is supported
    pub fn supports_federated_learning(&self) -> bool {
        self.capability_flags & 0x01 != 0
    }

    /// Check if model inference at UE is supported
    pub fn supports_model_inference(&self) -> bool {
        self.capability_flags & 0x02 != 0
    }

    /// Check if model training at UE is supported
    pub fn supports_model_training(&self) -> bool {
        self.capability_flags & 0x04 != 0
    }

    /// Decode from bytes (without IEI, with 1-byte length)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, Ie4Error> {
        if buf.remaining() < 1 {
            return Err(Ie4Error::BufferTooShort {
                expected: 1,
                actual: buf.remaining(),
            });
        }

        let length = buf.get_u8() as usize;
        if length < 3 {
            return Err(Ie4Error::BufferTooShort {
                expected: 3,
                actual: length,
            });
        }
        if buf.remaining() < length {
            return Err(Ie4Error::BufferTooShort {
                expected: length,
                actual: buf.remaining(),
            });
        }

        let capability_flags = buf.get_u8();
        let max_model_size_kb = buf.get_u16();

        if length > 3 {
            buf.advance(length - 3);
        }

        Ok(Self {
            capability_flags,
            max_model_size_kb,
        })
    }

    /// Encode to bytes (without IEI, with 1-byte length)
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        buf.put_u8(3); // length: flags(1) + max_model_size(2)
        buf.put_u8(self.capability_flags);
        buf.put_u16(self.max_model_size_kb);
    }

    /// Get the encoded length (including 1-byte length field)
    pub fn encoded_len(&self) -> usize {
        4 // 1 byte length + 3 bytes value
    }
}

impl NasEncode for IeAiMlCapability {
    fn nas_encode<B: BufMut>(&self, buf: &mut B) -> CodecResult<()> {
        self.encode(buf);
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        self.encoded_len()
    }
}

impl NasDecode for IeAiMlCapability {
    fn nas_decode<B: Buf>(buf: &mut B) -> CodecResult<Self> {
        IeAiMlCapability::decode(buf).map_err(|e| CodecError::InvalidValue(e.to_string()))
    }
}


// ============================================================================
// ISAC Parameter IE (6G extension)
// ============================================================================

/// ISAC (Integrated Sensing and Communication) Parameter IE (Type 4, TLV)
///
/// Contains ISAC parameters for 6G NTN and terrestrial systems.
/// This is a 6G extension IE.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct IeIsacParameter {
    /// ISAC mode flags
    ///
    /// Bit 0: Monostatic sensing supported
    /// Bit 1: Bistatic sensing supported
    /// Bit 2: Communication-assisted sensing supported
    /// Bit 3: Sensing-assisted communication supported
    /// Bits 4-7: Reserved
    pub mode_flags: u8,
    /// Sensing resolution (encoded value, units TBD)
    pub sensing_resolution: u8,
    /// Maximum sensing range (encoded value, units TBD)
    pub max_sensing_range: u8,
}

impl IeIsacParameter {
    /// Create a new ISAC Parameter IE
    pub fn new(mode_flags: u8, sensing_resolution: u8, max_sensing_range: u8) -> Self {
        Self {
            mode_flags,
            sensing_resolution,
            max_sensing_range,
        }
    }

    /// Decode from bytes (without IEI, with 1-byte length)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, Ie4Error> {
        if buf.remaining() < 1 {
            return Err(Ie4Error::BufferTooShort {
                expected: 1,
                actual: buf.remaining(),
            });
        }

        let length = buf.get_u8() as usize;
        if length < 3 {
            return Err(Ie4Error::BufferTooShort {
                expected: 3,
                actual: length,
            });
        }
        if buf.remaining() < length {
            return Err(Ie4Error::BufferTooShort {
                expected: length,
                actual: buf.remaining(),
            });
        }

        let mode_flags = buf.get_u8();
        let sensing_resolution = buf.get_u8();
        let max_sensing_range = buf.get_u8();

        if length > 3 {
            buf.advance(length - 3);
        }

        Ok(Self {
            mode_flags,
            sensing_resolution,
            max_sensing_range,
        })
    }

    /// Encode to bytes (without IEI, with 1-byte length)
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        buf.put_u8(3); // length
        buf.put_u8(self.mode_flags);
        buf.put_u8(self.sensing_resolution);
        buf.put_u8(self.max_sensing_range);
    }

    /// Get the encoded length (including 1-byte length field)
    pub fn encoded_len(&self) -> usize {
        4 // 1 byte length + 3 bytes value
    }
}

impl NasEncode for IeIsacParameter {
    fn nas_encode<B: BufMut>(&self, buf: &mut B) -> CodecResult<()> {
        self.encode(buf);
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        self.encoded_len()
    }
}

impl NasDecode for IeIsacParameter {
    fn nas_decode<B: Buf>(buf: &mut B) -> CodecResult<Self> {
        IeIsacParameter::decode(buf).map_err(|e| CodecError::InvalidValue(e.to_string()))
    }
}


// ============================================================================
// Semantic Communication Parameter IE (6G extension)
// ============================================================================

/// Semantic Communication Parameter IE (Type 4, TLV)
///
/// Indicates semantic communication capabilities for 6G systems.
/// This is a 6G extension IE.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct IeSemanticCommParameter {
    /// Semantic communication capability flags
    ///
    /// Bit 0: Semantic extraction supported
    /// Bit 1: Semantic encoding supported
    /// Bit 2: Semantic decoding supported
    /// Bit 3: Task-oriented communication supported
    /// Bits 4-7: Reserved
    pub capability_flags: u8,
    /// Supported semantic codec type(s)
    pub codec_type: u8,
}

impl IeSemanticCommParameter {
    /// Create a new Semantic Communication Parameter IE
    pub fn new(capability_flags: u8, codec_type: u8) -> Self {
        Self {
            capability_flags,
            codec_type,
        }
    }

    /// Decode from bytes (without IEI, with 1-byte length)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, Ie4Error> {
        if buf.remaining() < 1 {
            return Err(Ie4Error::BufferTooShort {
                expected: 1,
                actual: buf.remaining(),
            });
        }

        let length = buf.get_u8() as usize;
        if length < 2 {
            return Err(Ie4Error::BufferTooShort {
                expected: 2,
                actual: length,
            });
        }
        if buf.remaining() < length {
            return Err(Ie4Error::BufferTooShort {
                expected: length,
                actual: buf.remaining(),
            });
        }

        let capability_flags = buf.get_u8();
        let codec_type = buf.get_u8();

        if length > 2 {
            buf.advance(length - 2);
        }

        Ok(Self {
            capability_flags,
            codec_type,
        })
    }

    /// Encode to bytes (without IEI, with 1-byte length)
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        buf.put_u8(2); // length
        buf.put_u8(self.capability_flags);
        buf.put_u8(self.codec_type);
    }

    /// Get the encoded length (including 1-byte length field)
    pub fn encoded_len(&self) -> usize {
        3 // 1 byte length + 2 bytes value
    }
}

impl NasEncode for IeSemanticCommParameter {
    fn nas_encode<B: BufMut>(&self, buf: &mut B) -> CodecResult<()> {
        self.encode(buf);
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        self.encoded_len()
    }
}

impl NasDecode for IeSemanticCommParameter {
    fn nas_decode<B: Buf>(buf: &mut B) -> CodecResult<Self> {
        IeSemanticCommParameter::decode(buf).map_err(|e| CodecError::InvalidValue(e.to_string()))
    }
}


// ============================================================================
// Sub-THz Band Parameter IE (6G extension)
// ============================================================================

/// Sub-THz Band Parameter IE (Type 4, TLV)
///
/// Contains sub-THz band parameters for 6G systems operating
/// in the sub-THz frequency range (above 100 GHz).
/// This is a 6G extension IE.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct IeSubThzBandParameter {
    /// Supported sub-THz band flags
    ///
    /// Bit 0: 100-200 GHz band supported
    /// Bit 1: 200-300 GHz band supported
    /// Bit 2: 300-450 GHz band supported
    /// Bits 3-7: Reserved
    pub band_flags: u8,
    /// Maximum supported bandwidth (encoded, in MHz)
    pub max_bandwidth_mhz: u16,
    /// Minimum required beam tracking interval (in ms)
    pub beam_tracking_interval_ms: u8,
}

impl IeSubThzBandParameter {
    /// Create a new Sub-THz Band Parameter IE
    pub fn new(band_flags: u8, max_bandwidth_mhz: u16, beam_tracking_interval_ms: u8) -> Self {
        Self {
            band_flags,
            max_bandwidth_mhz,
            beam_tracking_interval_ms,
        }
    }

    /// Decode from bytes (without IEI, with 1-byte length)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, Ie4Error> {
        if buf.remaining() < 1 {
            return Err(Ie4Error::BufferTooShort {
                expected: 1,
                actual: buf.remaining(),
            });
        }

        let length = buf.get_u8() as usize;
        if length < 4 {
            return Err(Ie4Error::BufferTooShort {
                expected: 4,
                actual: length,
            });
        }
        if buf.remaining() < length {
            return Err(Ie4Error::BufferTooShort {
                expected: length,
                actual: buf.remaining(),
            });
        }

        let band_flags = buf.get_u8();
        let max_bandwidth_mhz = buf.get_u16();
        let beam_tracking_interval_ms = buf.get_u8();

        if length > 4 {
            buf.advance(length - 4);
        }

        Ok(Self {
            band_flags,
            max_bandwidth_mhz,
            beam_tracking_interval_ms,
        })
    }

    /// Encode to bytes (without IEI, with 1-byte length)
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        buf.put_u8(4); // length
        buf.put_u8(self.band_flags);
        buf.put_u16(self.max_bandwidth_mhz);
        buf.put_u8(self.beam_tracking_interval_ms);
    }

    /// Get the encoded length (including 1-byte length field)
    pub fn encoded_len(&self) -> usize {
        5 // 1 byte length + 4 bytes value
    }
}

impl NasEncode for IeSubThzBandParameter {
    fn nas_encode<B: BufMut>(&self, buf: &mut B) -> CodecResult<()> {
        self.encode(buf);
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        self.encoded_len()
    }
}

impl NasDecode for IeSubThzBandParameter {
    fn nas_decode<B: Buf>(buf: &mut B) -> CodecResult<Self> {
        IeSubThzBandParameter::decode(buf).map_err(|e| CodecError::InvalidValue(e.to_string()))
    }
}


// ============================================================================
// NTN Timing Advance IE (6G extension)
// ============================================================================

/// NTN (Non-Terrestrial Network) Timing Advance IE (Type 4, TLV)
///
/// Contains NTN timing advance information for satellite-based 6G systems.
/// This is a 6G extension IE.
///
/// The timing advance compensates for the propagation delay in
/// satellite communications.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct IeNtnTimingAdvance {
    /// Timing advance value (in units of 0.5 microseconds)
    pub timing_advance: u32,
    /// UE-specific timing advance valid flag
    pub ta_valid: bool,
}

impl IeNtnTimingAdvance {
    /// Create a new NTN Timing Advance IE
    pub fn new(timing_advance: u32, ta_valid: bool) -> Self {
        Self {
            timing_advance,
            ta_valid,
        }
    }

    /// Get timing advance in microseconds
    pub fn timing_advance_us(&self) -> f64 {
        self.timing_advance as f64 * 0.5
    }

    /// Decode from bytes (without IEI, with 1-byte length)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, Ie4Error> {
        if buf.remaining() < 1 {
            return Err(Ie4Error::BufferTooShort {
                expected: 1,
                actual: buf.remaining(),
            });
        }

        let length = buf.get_u8() as usize;
        if length < 5 {
            return Err(Ie4Error::BufferTooShort {
                expected: 5,
                actual: length,
            });
        }
        if buf.remaining() < length {
            return Err(Ie4Error::BufferTooShort {
                expected: length,
                actual: buf.remaining(),
            });
        }

        let timing_advance = buf.get_u32();
        let flags = buf.get_u8();
        let ta_valid = (flags & 0x01) != 0;

        if length > 5 {
            buf.advance(length - 5);
        }

        Ok(Self {
            timing_advance,
            ta_valid,
        })
    }

    /// Encode to bytes (without IEI, with 1-byte length)
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        buf.put_u8(5); // length
        buf.put_u32(self.timing_advance);
        buf.put_u8(if self.ta_valid { 0x01 } else { 0x00 });
    }

    /// Get the encoded length (including 1-byte length field)
    pub fn encoded_len(&self) -> usize {
        6 // 1 byte length + 5 bytes value
    }
}

impl NasEncode for IeNtnTimingAdvance {
    fn nas_encode<B: BufMut>(&self, buf: &mut B) -> CodecResult<()> {
        self.encode(buf);
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        self.encoded_len()
    }
}

impl NasDecode for IeNtnTimingAdvance {
    fn nas_decode<B: Buf>(buf: &mut B) -> CodecResult<Self> {
        IeNtnTimingAdvance::decode(buf).map_err(|e| CodecError::InvalidValue(e.to_string()))
    }
}


// ============================================================================
// NTN Access Barring IE (6G extension)
// ============================================================================

/// NTN Access Barring IE (Type 4, TLV)
///
/// Contains NTN-specific access barring information for satellite-based
/// 6G systems.
/// This is a 6G extension IE.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct IeNtnAccessBarring {
    /// Access barring factor (0-100, percentage)
    pub barring_factor: u8,
    /// Access barring time (in seconds, encoded)
    pub barring_time_seconds: u16,
    /// Access class barring flags (bits 0-9 for AC 0-9, bits 10-15 for AC 11-15)
    pub ac_barring_flags: u16,
}

impl IeNtnAccessBarring {
    /// Create a new NTN Access Barring IE
    pub fn new(barring_factor: u8, barring_time_seconds: u16, ac_barring_flags: u16) -> Self {
        Self {
            barring_factor,
            barring_time_seconds,
            ac_barring_flags,
        }
    }

    /// Check if a specific access class is barred
    pub fn is_ac_barred(&self, access_class: u8) -> bool {
        if access_class > 15 {
            return false;
        }
        (self.ac_barring_flags >> access_class) & 0x01 == 1
    }

    /// Decode from bytes (without IEI, with 1-byte length)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, Ie4Error> {
        if buf.remaining() < 1 {
            return Err(Ie4Error::BufferTooShort {
                expected: 1,
                actual: buf.remaining(),
            });
        }

        let length = buf.get_u8() as usize;
        if length < 5 {
            return Err(Ie4Error::BufferTooShort {
                expected: 5,
                actual: length,
            });
        }
        if buf.remaining() < length {
            return Err(Ie4Error::BufferTooShort {
                expected: length,
                actual: buf.remaining(),
            });
        }

        let barring_factor = buf.get_u8();
        let barring_time_seconds = buf.get_u16();
        let ac_barring_flags = buf.get_u16();

        if length > 5 {
            buf.advance(length - 5);
        }

        Ok(Self {
            barring_factor,
            barring_time_seconds,
            ac_barring_flags,
        })
    }

    /// Encode to bytes (without IEI, with 1-byte length)
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        buf.put_u8(5); // length
        buf.put_u8(self.barring_factor);
        buf.put_u16(self.barring_time_seconds);
        buf.put_u16(self.ac_barring_flags);
    }

    /// Get the encoded length (including 1-byte length field)
    pub fn encoded_len(&self) -> usize {
        6 // 1 byte length + 5 bytes value
    }
}

impl NasEncode for IeNtnAccessBarring {
    fn nas_encode<B: BufMut>(&self, buf: &mut B) -> CodecResult<()> {
        self.encode(buf);
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        self.encoded_len()
    }
}

impl NasDecode for IeNtnAccessBarring {
    fn nas_decode<B: Buf>(buf: &mut B) -> CodecResult<Self> {
        IeNtnAccessBarring::decode(buf).map_err(|e| CodecError::InvalidValue(e.to_string()))
    }
}


// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // UE Security Capability Tests
    // ========================================================================

    #[test]
    fn test_ue_security_capability_new() {
        let cap = IeUeSecurityCapability::new(0xF0, 0xF0);
        assert_eq!(cap.ea, 0xF0);
        assert_eq!(cap.ia, 0xF0);
        assert!(cap.eea.is_none());
        assert!(cap.eia.is_none());
    }

    #[test]
    fn test_ue_security_capability_supports_ea() {
        let cap = IeUeSecurityCapability::new(0xE0, 0x00); // EA0, EA1, EA2
        assert!(cap.supports_ea(0)); // EA0
        assert!(cap.supports_ea(1)); // EA1
        assert!(cap.supports_ea(2)); // EA2
        assert!(!cap.supports_ea(3)); // EA3
        assert!(!cap.supports_ea(7)); // EA7
    }

    #[test]
    fn test_ue_security_capability_supports_ia() {
        let cap = IeUeSecurityCapability::new(0x00, 0x80); // IA0 only
        assert!(cap.supports_ia(0)); // IA0
        assert!(!cap.supports_ia(1)); // IA1
    }

    #[test]
    fn test_ue_security_capability_encode_decode_minimal() {
        let cap = IeUeSecurityCapability::new(0xE0, 0xE0);
        let mut buf = Vec::new();
        cap.encode(&mut buf);

        // Length(1) + EA(1) + IA(1) = 3 bytes
        assert_eq!(buf.len(), 3);
        assert_eq!(buf[0], 2); // length = 2

        let decoded = IeUeSecurityCapability::decode(&mut &buf[..]).unwrap();
        assert_eq!(decoded.ea, 0xE0);
        assert_eq!(decoded.ia, 0xE0);
        assert!(decoded.eea.is_none());
        assert!(decoded.eia.is_none());
    }

    #[test]
    fn test_ue_security_capability_encode_decode_with_eps() {
        let cap = IeUeSecurityCapability::with_eps(0xE0, 0xE0, 0xF0, 0xF0);
        let mut buf = Vec::new();
        cap.encode(&mut buf);

        assert_eq!(buf.len(), 5); // length(1) + ea(1) + ia(1) + eea(1) + eia(1)
        assert_eq!(buf[0], 4); // length = 4

        let decoded = IeUeSecurityCapability::decode(&mut &buf[..]).unwrap();
        assert_eq!(decoded.ea, 0xE0);
        assert_eq!(decoded.ia, 0xE0);
        assert_eq!(decoded.eea, Some(0xF0));
        assert_eq!(decoded.eia, Some(0xF0));
    }

    #[test]
    fn test_ue_security_capability_decode_too_short() {
        let buf: &[u8] = &[];
        let result = IeUeSecurityCapability::decode(&mut &buf[..]);
        assert!(result.is_err());
    }

    // ========================================================================
    // AI/ML Capability Tests
    // ========================================================================

    #[test]
    fn test_ai_ml_capability_new() {
        let cap = IeAiMlCapability::new(0x07, 1024);
        assert_eq!(cap.capability_flags, 0x07);
        assert_eq!(cap.max_model_size_kb, 1024);
    }

    #[test]
    fn test_ai_ml_capability_flags() {
        let cap = IeAiMlCapability::new(0x07, 0);
        assert!(cap.supports_federated_learning());
        assert!(cap.supports_model_inference());
        assert!(cap.supports_model_training());
    }

    #[test]
    fn test_ai_ml_capability_encode_decode() {
        let cap = IeAiMlCapability::new(0x0F, 2048);
        let mut buf = Vec::new();
        cap.encode(&mut buf);

        assert_eq!(buf.len(), 4); // length(1) + flags(1) + size(2)

        let decoded = IeAiMlCapability::decode(&mut &buf[..]).unwrap();
        assert_eq!(decoded.capability_flags, 0x0F);
        assert_eq!(decoded.max_model_size_kb, 2048);
    }

    // ========================================================================
    // ISAC Parameter Tests
    // ========================================================================

    #[test]
    fn test_isac_parameter_new() {
        let param = IeIsacParameter::new(0x03, 10, 200);
        assert_eq!(param.mode_flags, 0x03);
        assert_eq!(param.sensing_resolution, 10);
        assert_eq!(param.max_sensing_range, 200);
    }

    #[test]
    fn test_isac_parameter_encode_decode() {
        let param = IeIsacParameter::new(0x0F, 50, 255);
        let mut buf = Vec::new();
        param.encode(&mut buf);

        assert_eq!(buf.len(), 4); // length(1) + flags(1) + res(1) + range(1)

        let decoded = IeIsacParameter::decode(&mut &buf[..]).unwrap();
        assert_eq!(decoded.mode_flags, 0x0F);
        assert_eq!(decoded.sensing_resolution, 50);
        assert_eq!(decoded.max_sensing_range, 255);
    }

    // ========================================================================
    // Semantic Communication Parameter Tests
    // ========================================================================

    #[test]
    fn test_semantic_comm_parameter_new() {
        let param = IeSemanticCommParameter::new(0x0F, 0x01);
        assert_eq!(param.capability_flags, 0x0F);
        assert_eq!(param.codec_type, 0x01);
    }

    #[test]
    fn test_semantic_comm_parameter_encode_decode() {
        let param = IeSemanticCommParameter::new(0x05, 0x02);
        let mut buf = Vec::new();
        param.encode(&mut buf);

        assert_eq!(buf.len(), 3); // length(1) + flags(1) + codec(1)

        let decoded = IeSemanticCommParameter::decode(&mut &buf[..]).unwrap();
        assert_eq!(decoded.capability_flags, 0x05);
        assert_eq!(decoded.codec_type, 0x02);
    }

    // ========================================================================
    // Sub-THz Band Parameter Tests
    // ========================================================================

    #[test]
    fn test_sub_thz_band_parameter_new() {
        let param = IeSubThzBandParameter::new(0x03, 5000, 10);
        assert_eq!(param.band_flags, 0x03);
        assert_eq!(param.max_bandwidth_mhz, 5000);
        assert_eq!(param.beam_tracking_interval_ms, 10);
    }

    #[test]
    fn test_sub_thz_band_parameter_encode_decode() {
        let param = IeSubThzBandParameter::new(0x07, 10000, 5);
        let mut buf = Vec::new();
        param.encode(&mut buf);

        assert_eq!(buf.len(), 5); // length(1) + flags(1) + bw(2) + interval(1)

        let decoded = IeSubThzBandParameter::decode(&mut &buf[..]).unwrap();
        assert_eq!(decoded.band_flags, 0x07);
        assert_eq!(decoded.max_bandwidth_mhz, 10000);
        assert_eq!(decoded.beam_tracking_interval_ms, 5);
    }

    // ========================================================================
    // NTN Timing Advance Tests
    // ========================================================================

    #[test]
    fn test_ntn_timing_advance_new() {
        let ta = IeNtnTimingAdvance::new(1000, true);
        assert_eq!(ta.timing_advance, 1000);
        assert!(ta.ta_valid);
    }

    #[test]
    fn test_ntn_timing_advance_us() {
        let ta = IeNtnTimingAdvance::new(1000, true);
        assert!((ta.timing_advance_us() - 500.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ntn_timing_advance_encode_decode() {
        let ta = IeNtnTimingAdvance::new(50000, true);
        let mut buf = Vec::new();
        ta.encode(&mut buf);

        assert_eq!(buf.len(), 6); // length(1) + ta(4) + flags(1)

        let decoded = IeNtnTimingAdvance::decode(&mut &buf[..]).unwrap();
        assert_eq!(decoded.timing_advance, 50000);
        assert!(decoded.ta_valid);
    }

    #[test]
    fn test_ntn_timing_advance_encode_decode_not_valid() {
        let ta = IeNtnTimingAdvance::new(0, false);
        let mut buf = Vec::new();
        ta.encode(&mut buf);

        let decoded = IeNtnTimingAdvance::decode(&mut &buf[..]).unwrap();
        assert_eq!(decoded.timing_advance, 0);
        assert!(!decoded.ta_valid);
    }

    // ========================================================================
    // NTN Access Barring Tests
    // ========================================================================

    #[test]
    fn test_ntn_access_barring_new() {
        let ab = IeNtnAccessBarring::new(50, 300, 0x03FF);
        assert_eq!(ab.barring_factor, 50);
        assert_eq!(ab.barring_time_seconds, 300);
        assert_eq!(ab.ac_barring_flags, 0x03FF);
    }

    #[test]
    fn test_ntn_access_barring_is_ac_barred() {
        let ab = IeNtnAccessBarring::new(50, 300, 0x0005); // AC 0 and AC 2 barred
        assert!(ab.is_ac_barred(0));
        assert!(!ab.is_ac_barred(1));
        assert!(ab.is_ac_barred(2));
        assert!(!ab.is_ac_barred(3));
    }

    #[test]
    fn test_ntn_access_barring_encode_decode() {
        let ab = IeNtnAccessBarring::new(75, 600, 0xFFFF);
        let mut buf = Vec::new();
        ab.encode(&mut buf);

        assert_eq!(buf.len(), 6); // length(1) + factor(1) + time(2) + flags(2)

        let decoded = IeNtnAccessBarring::decode(&mut &buf[..]).unwrap();
        assert_eq!(decoded.barring_factor, 75);
        assert_eq!(decoded.barring_time_seconds, 600);
        assert_eq!(decoded.ac_barring_flags, 0xFFFF);
    }

    #[test]
    fn test_ntn_access_barring_decode_too_short() {
        let buf: &[u8] = &[0x02, 0x50, 0x01]; // length 2, but need 5
        let result = IeNtnAccessBarring::decode(&mut &buf[..]);
        assert!(result.is_err());
    }
}
