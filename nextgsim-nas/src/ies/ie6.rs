//! Type 6 Information Elements (variable length, TLV-E)
//!
//! Type 6 IEs have a variable length with a 2-byte length field (TLV-E format).
//! They are used for encoding/decoding NAS message fields that may be larger
//! than 255 bytes.
//!
//! Based on 3GPP TS 24.501 specification.

use bytes::{Buf, BufMut};
use thiserror::Error;

use crate::codec::{CodecError, CodecResult, NasDecode, NasEncode};

/// Error type for Type 6 IE decoding
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum Ie6Error {
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
// LADN Entry (3GPP TS 24.501 Section 9.11.3.30)
// ============================================================================

/// A single LADN (Local Area Data Network) entry
///
/// Each entry contains a DNN and a TAI (Tracking Area Identity) list.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LadnEntry {
    /// DNN value (encoded as length-prefixed labels, e.g. "\x08internet")
    pub dnn: Vec<u8>,
    /// TAI list (raw encoded bytes per 3GPP TS 24.501 Section 9.11.3.9)
    pub tai_list: Vec<u8>,
}

impl LadnEntry {
    /// Create a new LADN entry
    pub fn new(dnn: Vec<u8>, tai_list: Vec<u8>) -> Self {
        Self { dnn, tai_list }
    }

    /// Decode a single LADN entry from bytes
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, Ie6Error> {
        // DNN: 1-byte length + DNN value
        if buf.remaining() < 1 {
            return Err(Ie6Error::BufferTooShort {
                expected: 1,
                actual: buf.remaining(),
            });
        }
        let dnn_len = buf.get_u8() as usize;
        if buf.remaining() < dnn_len {
            return Err(Ie6Error::BufferTooShort {
                expected: dnn_len,
                actual: buf.remaining(),
            });
        }
        let mut dnn = vec![0u8; dnn_len];
        buf.copy_to_slice(&mut dnn);

        // TAI list: 2-byte length + TAI list value
        if buf.remaining() < 2 {
            return Err(Ie6Error::BufferTooShort {
                expected: 2,
                actual: buf.remaining(),
            });
        }
        let tai_len = buf.get_u16() as usize;
        if buf.remaining() < tai_len {
            return Err(Ie6Error::BufferTooShort {
                expected: tai_len,
                actual: buf.remaining(),
            });
        }
        let mut tai_list = vec![0u8; tai_len];
        buf.copy_to_slice(&mut tai_list);

        Ok(Self { dnn, tai_list })
    }

    /// Encode a single LADN entry to bytes
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        // DNN: 1-byte length + value
        buf.put_u8(self.dnn.len() as u8);
        buf.put_slice(&self.dnn);

        // TAI list: 2-byte length + value
        buf.put_u16(self.tai_list.len() as u16);
        buf.put_slice(&self.tai_list);
    }

    /// Get the encoded length of this entry
    pub fn encoded_len(&self) -> usize {
        1 + self.dnn.len() + 2 + self.tai_list.len()
    }
}


// ============================================================================
// LADN Information IE (3GPP TS 24.501 Section 9.11.3.30)
// ============================================================================

/// LADN Information IE (Type 6, TLV-E)
///
/// Contains a list of Local Area Data Network (LADN) entries.
/// Each entry specifies a DNN and associated tracking areas where
/// the LADN is available.
///
/// 3GPP TS 24.501 Section 9.11.3.30
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct IeLadnInformation {
    /// List of LADN entries
    pub entries: Vec<LadnEntry>,
}

impl IeLadnInformation {
    /// Create a new LADN Information IE
    pub fn new(entries: Vec<LadnEntry>) -> Self {
        Self { entries }
    }

    /// Create an empty LADN Information IE
    pub fn empty() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Decode from bytes (without IEI, with 2-byte length)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, Ie6Error> {
        if buf.remaining() < 2 {
            return Err(Ie6Error::BufferTooShort {
                expected: 2,
                actual: buf.remaining(),
            });
        }

        let total_len = buf.get_u16() as usize;
        if buf.remaining() < total_len {
            return Err(Ie6Error::BufferTooShort {
                expected: total_len,
                actual: buf.remaining(),
            });
        }

        let mut entries = Vec::new();
        let mut remaining = total_len;

        // Create a sub-slice to limit reading
        let mut sub_buf = &buf.chunk()[..remaining];

        while sub_buf.remaining() > 0 {
            let entry = LadnEntry::decode(&mut sub_buf)?;
            remaining -= entry.encoded_len();
            entries.push(entry);
        }

        // Advance the original buffer
        buf.advance(total_len);

        Ok(Self { entries })
    }

    /// Encode to bytes (without IEI, with 2-byte length)
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        // Calculate total value length
        let value_len: usize = self.entries.iter().map(LadnEntry::encoded_len).sum();
        buf.put_u16(value_len as u16);

        for entry in &self.entries {
            entry.encode(buf);
        }
    }

    /// Get the encoded length (including 2-byte length field)
    pub fn encoded_len(&self) -> usize {
        let value_len: usize = self.entries.iter().map(LadnEntry::encoded_len).sum();
        2 + value_len
    }
}

impl NasEncode for IeLadnInformation {
    fn nas_encode<B: BufMut>(&self, buf: &mut B) -> CodecResult<()> {
        self.encode(buf);
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        self.encoded_len()
    }
}

impl NasDecode for IeLadnInformation {
    fn nas_decode<B: Buf>(buf: &mut B) -> CodecResult<Self> {
        IeLadnInformation::decode(buf).map_err(|e| CodecError::InvalidValue(e.to_string()))
    }
}


// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ladn_entry_new() {
        let entry = LadnEntry::new(
            vec![0x08, 0x69, 0x6E, 0x74, 0x65, 0x72, 0x6E, 0x65, 0x74],
            vec![0x00, 0x13, 0x10, 0xF4, 0x00, 0x00, 0x01],
        );
        assert_eq!(entry.dnn.len(), 9);
        assert_eq!(entry.tai_list.len(), 7);
    }

    #[test]
    fn test_ladn_entry_encode_decode() {
        let entry = LadnEntry::new(
            vec![0x08, 0x69, 0x6E, 0x74, 0x65, 0x72, 0x6E, 0x65, 0x74],
            vec![0x00, 0x13, 0x10, 0xF4, 0x00, 0x00, 0x01],
        );

        let mut buf = Vec::new();
        entry.encode(&mut buf);

        let decoded = LadnEntry::decode(&mut &buf[..]).unwrap();
        assert_eq!(decoded.dnn, entry.dnn);
        assert_eq!(decoded.tai_list, entry.tai_list);
    }

    #[test]
    fn test_ladn_information_empty() {
        let info = IeLadnInformation::empty();
        assert!(info.entries.is_empty());
    }

    #[test]
    fn test_ladn_information_encode_decode_empty() {
        let info = IeLadnInformation::empty();
        let mut buf = Vec::new();
        info.encode(&mut buf);

        // 2-byte length = 0
        assert_eq!(buf.len(), 2);
        assert_eq!(buf[0], 0x00);
        assert_eq!(buf[1], 0x00);

        let decoded = IeLadnInformation::decode(&mut &buf[..]).unwrap();
        assert!(decoded.entries.is_empty());
    }

    #[test]
    fn test_ladn_information_encode_decode_single_entry() {
        let entry = LadnEntry::new(
            vec![0x04, 0x74, 0x65, 0x73, 0x74], // "test"
            vec![0x00, 0x01, 0x02],              // dummy TAI
        );
        let info = IeLadnInformation::new(vec![entry]);

        let mut buf = Vec::new();
        info.encode(&mut buf);

        let decoded = IeLadnInformation::decode(&mut &buf[..]).unwrap();
        assert_eq!(decoded.entries.len(), 1);
        assert_eq!(decoded.entries[0].dnn, vec![0x04, 0x74, 0x65, 0x73, 0x74]);
        assert_eq!(decoded.entries[0].tai_list, vec![0x00, 0x01, 0x02]);
    }

    #[test]
    fn test_ladn_information_encode_decode_multiple_entries() {
        let entry1 = LadnEntry::new(vec![0x01, 0x41], vec![0x01, 0x02]);
        let entry2 = LadnEntry::new(vec![0x01, 0x42], vec![0x03, 0x04]);
        let info = IeLadnInformation::new(vec![entry1, entry2]);

        let mut buf = Vec::new();
        info.encode(&mut buf);

        let decoded = IeLadnInformation::decode(&mut &buf[..]).unwrap();
        assert_eq!(decoded.entries.len(), 2);
        assert_eq!(decoded.entries[0].dnn, vec![0x01, 0x41]);
        assert_eq!(decoded.entries[0].tai_list, vec![0x01, 0x02]);
        assert_eq!(decoded.entries[1].dnn, vec![0x01, 0x42]);
        assert_eq!(decoded.entries[1].tai_list, vec![0x03, 0x04]);
    }

    #[test]
    fn test_ladn_information_decode_too_short() {
        let buf: &[u8] = &[0x00]; // Only 1 byte, need 2 for length
        let result = IeLadnInformation::decode(&mut &buf[..]);
        assert!(result.is_err());
    }

    #[test]
    fn test_ladn_information_encoded_len() {
        let entry = LadnEntry::new(vec![0x01, 0x41], vec![0x01, 0x02]);
        let info = IeLadnInformation::new(vec![entry]);

        // 2 (outer length) + 1 (dnn len) + 2 (dnn) + 2 (tai len) + 2 (tai) = 9
        assert_eq!(info.encoded_len(), 9);
    }
}
