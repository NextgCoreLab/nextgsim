//! RLC PDU types and encoding/decoding — TS 38.322 §6.2

use crate::error::RlcError;

// ── Segmentation Info ────────────────────────────────────────────────────────

/// SI (Segmentation Info) field — 2-bit field present in every UM/AM PDU
/// header (TS 38.322 §6.2.3.3, Table 6.2.3.3-1 / §6.2.3.5).
///
/// Encodes whether the payload is a complete SDU or a segment thereof.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentationInfo {
    /// Complete SDU — no Segment Offset field present
    FullSdu = 0b00,
    /// First segment — no SO field; this is the head of the SDU
    FirstSegment = 0b01,
    /// Last segment — SO field present
    LastSegment = 0b10,
    /// Middle segment — SO field present
    MiddleSegment = 0b11,
}

impl SegmentationInfo {
    /// Decode a raw 2-bit SI value
    pub fn from_u8(v: u8) -> Result<Self, RlcError> {
        match v & 0b11 {
            0b00 => Ok(Self::FullSdu),
            0b01 => Ok(Self::FirstSegment),
            0b10 => Ok(Self::LastSegment),
            0b11 => Ok(Self::MiddleSegment),
            _ => Err(RlcError::InvalidSi(v)),
        }
    }

    /// Returns `true` if a Segment Offset field follows the fixed header
    pub fn has_so(self) -> bool {
        matches!(self, Self::LastSegment | Self::MiddleSegment)
    }

    /// Returns `true` if this segment is the start (or entirety) of an SDU
    pub fn is_first(self) -> bool {
        matches!(self, Self::FullSdu | Self::FirstSegment)
    }

    /// Returns `true` if this segment is the end (or entirety) of an SDU
    pub fn is_last(self) -> bool {
        matches!(self, Self::FullSdu | Self::LastSegment)
    }
}

// ── UM PDU ────────────────────────────────────────────────────────────────────

/// RLC UM PDU — TS 38.322 §6.2.3.3
///
/// Wire layout (6-bit SN):
/// ```text
///  0        1
/// ┌──┬──┬──────┐
/// │R │SI│  SN  │  (1 byte when SI=00 or SI=01)
/// ├──┴──┴──────┤
/// │    SO hi   │  (2 bytes, only when SI has SO)
/// │    SO lo   │
/// └────────────┘
/// ```
///
/// Wire layout (12-bit SN):
/// ```text
///  0        1        2        3
/// ┌──┬──┬──────────────────────┐
/// │R │SI│      SN[11:0]        │  (2 bytes)
/// ├──┴──┴──────────────────────┤
/// │        SO[15:8]            │  (2 bytes, only when SI has SO)
/// │        SO[7:0]             │
/// └────────────────────────────┘
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RlcUmPdu {
    /// Segmentation Info
    pub si: SegmentationInfo,
    /// Sequence number (6 or 12 bits depending on configuration)
    pub sn: u16,
    /// Segment offset — present only when `si` has SO
    pub so: Option<u16>,
    /// Payload bytes
    pub data: Vec<u8>,
}

impl RlcUmPdu {
    /// Encode this PDU into bytes using a 6-bit SN field.
    ///
    /// Header size: 1 byte (+ 2 bytes SO when present).
    pub fn encode_sn6(&self) -> Vec<u8> {
        let header_len = 1 + if self.si.has_so() { 2 } else { 0 };
        let mut out = Vec::with_capacity(header_len + self.data.len());

        // Byte 0: R(1) | SI(2) | SN(5 of 6 — but 6-bit SN uses bits [5:0])
        // Layout: [R][SI1][SI0][SN5][SN4][SN3][SN2][SN1][SN0]
        // Packed into 1 byte: bits 7..6 = SI, bits 5..0 = SN
        let si_bits = (self.si as u8) << 6;
        let sn_bits = (self.sn as u8) & 0x3F;
        out.push(si_bits | sn_bits);

        if let Some(so) = self.so {
            out.push((so >> 8) as u8);
            out.push(so as u8);
        }

        out.extend_from_slice(&self.data);
        out
    }

    /// Encode this PDU into bytes using a 12-bit SN field.
    ///
    /// Header size: 2 bytes (+ 2 bytes SO when present).
    pub fn encode_sn12(&self) -> Vec<u8> {
        let header_len = 2 + if self.si.has_so() { 2 } else { 0 };
        let mut out = Vec::with_capacity(header_len + self.data.len());

        // Byte 0: [R][R][SI1][SI0][SN11][SN10][SN9][SN8]
        // Byte 1: [SN7][SN6][SN5][SN4][SN3][SN2][SN1][SN0]
        let si_bits = (self.si as u8) << 4;
        let sn_hi = ((self.sn >> 8) as u8) & 0x0F;
        let sn_lo = (self.sn & 0xFF) as u8;
        out.push(si_bits | sn_hi);
        out.push(sn_lo);

        if let Some(so) = self.so {
            out.push((so >> 8) as u8);
            out.push(so as u8);
        }

        out.extend_from_slice(&self.data);
        out
    }

    /// Decode a UM PDU from bytes using a 6-bit SN field.
    pub fn decode_sn6(buf: &[u8]) -> Result<Self, RlcError> {
        if buf.is_empty() {
            return Err(RlcError::PduTooShort { need: 1, got: 0 });
        }
        let si = SegmentationInfo::from_u8(buf[0] >> 6)?;
        let sn = (buf[0] & 0x3F) as u16;

        let (so, data_start) = if si.has_so() {
            if buf.len() < 3 {
                return Err(RlcError::PduTooShort {
                    need: 3,
                    got: buf.len(),
                });
            }
            let so = ((buf[1] as u16) << 8) | buf[2] as u16;
            (Some(so), 3)
        } else {
            (None, 1)
        };

        Ok(Self {
            si,
            sn,
            so,
            data: buf[data_start..].to_vec(),
        })
    }

    /// Decode a UM PDU from bytes using a 12-bit SN field.
    pub fn decode_sn12(buf: &[u8]) -> Result<Self, RlcError> {
        if buf.len() < 2 {
            return Err(RlcError::PduTooShort {
                need: 2,
                got: buf.len(),
            });
        }
        let si = SegmentationInfo::from_u8(buf[0] >> 4)?;
        let sn = (((buf[0] & 0x0F) as u16) << 8) | buf[1] as u16;

        let (so, data_start) = if si.has_so() {
            if buf.len() < 4 {
                return Err(RlcError::PduTooShort {
                    need: 4,
                    got: buf.len(),
                });
            }
            let so = ((buf[2] as u16) << 8) | buf[3] as u16;
            (Some(so), 4)
        } else {
            (None, 2)
        };

        Ok(Self {
            si,
            sn,
            so,
            data: buf[data_start..].to_vec(),
        })
    }
}

// ── AM PDU ────────────────────────────────────────────────────────────────────

/// RLC AM Data PDU — TS 38.322 §6.2.3.5
///
/// Wire layout (12-bit SN):
/// ```text
/// Byte 0: [D/C][P][SI1][SI0][SN11][SN10][SN9][SN8]
/// Byte 1: [SN7][SN6][SN5][SN4][SN3][SN2][SN1][SN0]
/// Byte 2-3: SO (optional, 16 bits)
/// ```
///
/// Wire layout (18-bit SN):
/// ```text
/// Byte 0: [D/C][P][SI1][SI0][R][R][SN17][SN16]
/// Byte 1: [SN15..SN8]
/// Byte 2: [SN7..SN0]
/// Byte 3-4: SO (optional, 16 bits)
/// ```
///
/// The D/C bit distinguishes Data PDUs (`dc=true`) from Control PDUs (`dc=false`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RlcAmPdu {
    /// D/C flag: `true` = Data PDU, `false` = Control PDU (STATUS)
    pub dc: bool,
    /// Poll bit — requests a STATUS report from the peer
    pub p: bool,
    /// Segmentation info
    pub si: SegmentationInfo,
    /// Sequence number (12 or 18 bits)
    pub sn: u32,
    /// Segment offset (present when SI has SO)
    pub so: Option<u16>,
    /// Payload bytes
    pub data: Vec<u8>,
}

impl RlcAmPdu {
    /// Encode using a 12-bit SN field.
    pub fn encode_sn12(&self) -> Vec<u8> {
        let header_len = 2 + if self.si.has_so() { 2 } else { 0 };
        let mut out = Vec::with_capacity(header_len + self.data.len());

        let dc_bit = if self.dc { 0x80u8 } else { 0 };
        let p_bit = if self.p { 0x40u8 } else { 0 };
        let si_bits = (self.si as u8) << 4;
        let sn_hi = ((self.sn >> 8) as u8) & 0x0F;
        out.push(dc_bit | p_bit | si_bits | sn_hi);
        out.push((self.sn & 0xFF) as u8);

        if let Some(so) = self.so {
            out.push((so >> 8) as u8);
            out.push(so as u8);
        }
        out.extend_from_slice(&self.data);
        out
    }

    /// Encode using an 18-bit SN field.
    pub fn encode_sn18(&self) -> Vec<u8> {
        let header_len = 3 + if self.si.has_so() { 2 } else { 0 };
        let mut out = Vec::with_capacity(header_len + self.data.len());

        let dc_bit = if self.dc { 0x80u8 } else { 0 };
        let p_bit = if self.p { 0x40u8 } else { 0 };
        let si_bits = (self.si as u8) << 4;
        let sn_hi2 = ((self.sn >> 16) as u8) & 0x03;
        let sn_mid = ((self.sn >> 8) & 0xFF) as u8;
        let sn_lo = (self.sn & 0xFF) as u8;
        out.push(dc_bit | p_bit | si_bits | sn_hi2);
        out.push(sn_mid);
        out.push(sn_lo);

        if let Some(so) = self.so {
            out.push((so >> 8) as u8);
            out.push(so as u8);
        }
        out.extend_from_slice(&self.data);
        out
    }

    /// Decode using a 12-bit SN field.
    pub fn decode_sn12(buf: &[u8]) -> Result<Self, RlcError> {
        if buf.len() < 2 {
            return Err(RlcError::PduTooShort {
                need: 2,
                got: buf.len(),
            });
        }
        let dc = (buf[0] & 0x80) != 0;
        let p = (buf[0] & 0x40) != 0;
        let si = SegmentationInfo::from_u8((buf[0] >> 4) & 0x03)?;
        let sn = (((buf[0] & 0x0F) as u32) << 8) | buf[1] as u32;

        let (so, data_start) = if si.has_so() {
            if buf.len() < 4 {
                return Err(RlcError::PduTooShort {
                    need: 4,
                    got: buf.len(),
                });
            }
            let so = ((buf[2] as u16) << 8) | buf[3] as u16;
            (Some(so), 4)
        } else {
            (None, 2)
        };

        Ok(Self {
            dc,
            p,
            si,
            sn,
            so,
            data: buf[data_start..].to_vec(),
        })
    }

    /// Decode using an 18-bit SN field.
    pub fn decode_sn18(buf: &[u8]) -> Result<Self, RlcError> {
        if buf.len() < 3 {
            return Err(RlcError::PduTooShort {
                need: 3,
                got: buf.len(),
            });
        }
        let dc = (buf[0] & 0x80) != 0;
        let p = (buf[0] & 0x40) != 0;
        let si = SegmentationInfo::from_u8((buf[0] >> 4) & 0x03)?;
        let sn = (((buf[0] & 0x03) as u32) << 16) | ((buf[1] as u32) << 8) | buf[2] as u32;

        let (so, data_start) = if si.has_so() {
            if buf.len() < 5 {
                return Err(RlcError::PduTooShort {
                    need: 5,
                    got: buf.len(),
                });
            }
            let so = ((buf[3] as u16) << 8) | buf[4] as u16;
            (Some(so), 5)
        } else {
            (None, 3)
        };

        Ok(Self {
            dc,
            p,
            si,
            sn,
            so,
            data: buf[data_start..].to_vec(),
        })
    }
}

// ── STATUS PDU (AM control) ───────────────────────────────────────────────────

/// RLC AM STATUS PDU (Control PDU) — TS 38.322 §6.2.3.6
///
/// Minimal implementation: ACK_SN only, no NACK list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RlcStatusPdu {
    /// Highest SN that the receiver has received in sequence (next expected)
    pub ack_sn: u32,
}

impl RlcStatusPdu {
    /// Create a new STATUS PDU acknowledging up to (but not including) `ack_sn`.
    pub fn new(ack_sn: u32) -> Self {
        Self { ack_sn }
    }

    /// Encode as 3 bytes (D/C=0, CPT=000, ACK_SN 12-bit, E1=0, padding).
    ///
    /// Wire (12-bit SN):
    /// ```text
    /// Byte 0: [D/C=0][CPT=000][ACK_SN[11:8]]
    /// Byte 1: [ACK_SN[7:0]]
    /// Byte 2: [E1=0][padding=0000000]
    /// ```
    pub fn encode_sn12(&self) -> Vec<u8> {
        let sn = self.ack_sn & 0x0FFF;
        vec![
            ((sn >> 8) as u8) & 0x0F, // D/C=0, CPT=000, SN[11:8]
            (sn & 0xFF) as u8,
            0x00, // E1=0, no NACKs
        ]
    }

    /// Decode from bytes (12-bit SN STATUS PDU).
    pub fn decode_sn12(buf: &[u8]) -> Result<Self, RlcError> {
        if buf.len() < 3 {
            return Err(RlcError::PduTooShort {
                need: 3,
                got: buf.len(),
            });
        }
        let ack_sn = (((buf[0] & 0x0F) as u32) << 8) | buf[1] as u32;
        Ok(Self { ack_sn })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── SegmentationInfo ──────────────────────────────────────────────────────

    #[test]
    fn test_si_roundtrip() {
        for &(bits, expected) in &[
            (0b00u8, SegmentationInfo::FullSdu),
            (0b01, SegmentationInfo::FirstSegment),
            (0b10, SegmentationInfo::LastSegment),
            (0b11, SegmentationInfo::MiddleSegment),
        ] {
            let si = SegmentationInfo::from_u8(bits).unwrap();
            assert_eq!(si, expected);
        }
    }

    #[test]
    fn test_si_has_so() {
        assert!(!SegmentationInfo::FullSdu.has_so());
        assert!(!SegmentationInfo::FirstSegment.has_so());
        assert!(SegmentationInfo::LastSegment.has_so());
        assert!(SegmentationInfo::MiddleSegment.has_so());
    }

    // ── UM PDU (6-bit SN) ─────────────────────────────────────────────────────

    #[test]
    fn test_um_sn6_full_sdu_roundtrip() {
        let pdu = RlcUmPdu {
            si: SegmentationInfo::FullSdu,
            sn: 42,
            so: None,
            data: vec![1, 2, 3, 4],
        };
        let encoded = pdu.encode_sn6();
        assert_eq!(encoded.len(), 1 + 4);
        let decoded = RlcUmPdu::decode_sn6(&encoded).unwrap();
        assert_eq!(decoded, pdu);
    }

    #[test]
    fn test_um_sn6_middle_segment_roundtrip() {
        let pdu = RlcUmPdu {
            si: SegmentationInfo::MiddleSegment,
            sn: 63,
            so: Some(128),
            data: vec![0xAB, 0xCD],
        };
        let encoded = pdu.encode_sn6();
        assert_eq!(encoded.len(), 1 + 2 + 2); // header + SO + data
        let decoded = RlcUmPdu::decode_sn6(&encoded).unwrap();
        assert_eq!(decoded, pdu);
    }

    #[test]
    fn test_um_sn12_full_sdu_roundtrip() {
        let pdu = RlcUmPdu {
            si: SegmentationInfo::FullSdu,
            sn: 0xABC,
            so: None,
            data: vec![0xFF; 10],
        };
        let encoded = pdu.encode_sn12();
        assert_eq!(encoded.len(), 2 + 10);
        let decoded = RlcUmPdu::decode_sn12(&encoded).unwrap();
        assert_eq!(decoded, pdu);
    }

    #[test]
    fn test_um_sn12_last_segment_roundtrip() {
        let pdu = RlcUmPdu {
            si: SegmentationInfo::LastSegment,
            sn: 4095,
            so: Some(256),
            data: vec![0x11, 0x22, 0x33],
        };
        let encoded = pdu.encode_sn12();
        assert_eq!(encoded.len(), 2 + 2 + 3);
        let decoded = RlcUmPdu::decode_sn12(&encoded).unwrap();
        assert_eq!(decoded, pdu);
    }

    // ── AM PDU (12-bit SN) ────────────────────────────────────────────────────

    #[test]
    fn test_am_sn12_data_pdu_roundtrip() {
        let pdu = RlcAmPdu {
            dc: true,
            p: true,
            si: SegmentationInfo::FullSdu,
            sn: 100,
            so: None,
            data: vec![0xDE, 0xAD, 0xBE, 0xEF],
        };
        let encoded = pdu.encode_sn12();
        assert_eq!(encoded.len(), 2 + 4);
        let decoded = RlcAmPdu::decode_sn12(&encoded).unwrap();
        assert_eq!(decoded, pdu);
    }

    #[test]
    fn test_am_sn12_segment_roundtrip() {
        let pdu = RlcAmPdu {
            dc: true,
            p: false,
            si: SegmentationInfo::FirstSegment,
            sn: 200,
            so: None,
            data: vec![1, 2, 3],
        };
        let encoded = pdu.encode_sn12();
        let decoded = RlcAmPdu::decode_sn12(&encoded).unwrap();
        assert_eq!(decoded, pdu);
    }

    #[test]
    fn test_am_sn18_roundtrip() {
        let pdu = RlcAmPdu {
            dc: true,
            p: false,
            si: SegmentationInfo::FullSdu,
            sn: 0x1_ABCD,
            so: None,
            data: vec![0x42; 8],
        };
        let encoded = pdu.encode_sn18();
        assert_eq!(encoded.len(), 3 + 8);
        let decoded = RlcAmPdu::decode_sn18(&encoded).unwrap();
        assert_eq!(decoded, pdu);
    }

    // ── STATUS PDU ────────────────────────────────────────────────────────────

    #[test]
    fn test_status_pdu_roundtrip() {
        let status = RlcStatusPdu::new(42);
        let encoded = status.encode_sn12();
        assert_eq!(encoded.len(), 3);
        let decoded = RlcStatusPdu::decode_sn12(&encoded).unwrap();
        assert_eq!(decoded.ack_sn, 42);
    }
}
