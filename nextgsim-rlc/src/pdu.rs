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
    /// Whether this PDU's header carries an SN field.
    ///
    /// TS 38.322 §6.2.2.3: only a **segmented** SDU does. A complete SDU needs no
    /// SN because there is nothing to reassemble it with, and the receiver delivers
    /// it immediately (§5.2.2.2.2's first branch).
    pub fn carries_sn(&self) -> bool {
        self.si != SegmentationInfo::FullSdu
    }

    /// Header length in octets for a 12-bit-SN UMD PDU with the given `si`.
    ///
    /// A free function of `si` alone so the transmitter can size a header before it
    /// has a PDU to encode -- which it must, because whether the SDU fits depends on
    /// the header length and the header length depends on whether it fits.
    pub fn header_len_sn12(si: SegmentationInfo) -> usize {
        let base = if si == SegmentationInfo::FullSdu {
            1
        } else {
            2
        };
        base + if si.has_so() { 2 } else { 0 }
    }

    /// Header length in octets for a 6-bit-SN UMD PDU with the given `si`.
    ///
    /// Always one octet plus the SO, because a 6-bit SN shares its octet with the
    /// SI: for a complete SDU those six bits become reserved rather than absent.
    pub fn header_len_sn6(si: SegmentationInfo) -> usize {
        1 + if si.has_so() { 2 } else { 0 }
    }

    /// Encode this PDU into bytes using a 6-bit SN field.
    ///
    /// Header size: 1 byte (+ 2 bytes SO when present).
    ///
    /// A PDU carrying a **complete** SDU has no SN field at all
    /// (TS 38.322 §6.2.2.3: *"An UMD PDU header contains the SN field only when
    /// the corresponding RLC SDU is segmented"*). For a 6-bit SN the header stays
    /// one octet and the SN bits become reserved zeros, so the difference is in the
    /// contents rather than the length.
    pub fn encode_sn6(&self) -> Vec<u8> {
        let header_len = 1 + if self.si.has_so() { 2 } else { 0 };
        let mut out = Vec::with_capacity(header_len + self.data.len());

        // Byte 0: [SI1][SI0][SN5][SN4][SN3][SN2][SN1][SN0], and for a complete SDU
        // the six SN bits are reserved and coded as zero.
        let si_bits = (self.si as u8) << 6;
        let sn_bits = if self.carries_sn() {
            (self.sn as u8) & 0x3F
        } else {
            0
        };
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
    /// Header size: **1** byte for a complete SDU, 2 bytes for a segment
    /// (+ 2 bytes SO when present).
    ///
    /// TS 38.322 §6.2.2.3: *"When an UMD PDU contains a complete RLC SDU, the UMD
    /// PDU header only contains the SI and R fields"*, and *"An UMD PDU header
    /// contains the SN field only when the corresponding RLC SDU is segmented"*.
    /// This tree used to emit the SN unconditionally, so a complete-SDU PDU was two
    /// octets of header where a conformant one is one -- and a conformant peer read
    /// the extra octet as the first byte of the SDU (issue #103).
    pub fn encode_sn12(&self) -> Vec<u8> {
        let header_len = Self::header_len_sn12(self.si);
        let mut out = Vec::with_capacity(header_len + self.data.len());

        // Byte 0: [R][R][SI1][SI0][SN11][SN10][SN9][SN8]
        // Byte 1: [SN7][SN6][SN5][SN4][SN3][SN2][SN1][SN0]
        //
        // For a complete SDU byte 0's low nibble is reserved and byte 1 is absent.
        let si_bits = (self.si as u8) << 4;
        if self.carries_sn() {
            let sn_hi = ((self.sn >> 8) as u8) & 0x0F;
            let sn_lo = (self.sn & 0xFF) as u8;
            out.push(si_bits | sn_hi);
            out.push(sn_lo);
        } else {
            out.push(si_bits);
        }

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
        // No SN field on a complete SDU (TS 38.322 §6.2.2.3), so those bits are
        // reserved. Reported as 0 rather than as whatever the reserved bits hold:
        // a receiver that used them would be keying its window on padding.
        let sn = if si == SegmentationInfo::FullSdu {
            0
        } else {
            (buf[0] & 0x3F) as u16
        };

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
        // One octet is enough for a complete SDU, which has no SN field
        // (TS 38.322 §6.2.2.3); a segment needs two.
        if buf.is_empty() {
            return Err(RlcError::PduTooShort { need: 1, got: 0 });
        }
        let si = SegmentationInfo::from_u8(buf[0] >> 4)?;
        if si != SegmentationInfo::FullSdu && buf.len() < 2 {
            return Err(RlcError::PduTooShort {
                need: 2,
                got: buf.len(),
            });
        }
        let sn = if si == SegmentationInfo::FullSdu {
            0
        } else {
            (((buf[0] & 0x0F) as u16) << 8) | buf[1] as u16
        };

        let (so, data_start) = if si.has_so() {
            if buf.len() < 4 {
                return Err(RlcError::PduTooShort {
                    need: 4,
                    got: buf.len(),
                });
            }
            let so = ((buf[2] as u16) << 8) | buf[3] as u16;
            (Some(so), 4)
        } else if si == SegmentationInfo::FullSdu {
            (None, 1)
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

/// One negative acknowledgement inside a STATUS PDU (TS 38.322 §6.2.2.5).
///
/// `so_range` and `nack_range` are the optional refinements the E2 and E3 bits
/// announce. This tree's receiver only ever emits whole-SDU NACKs, but a
/// conformant peer may send either, and a decoder that skipped them would read
/// the following NACK_SN out of the middle of an SOstart field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RlcStatusNack {
    /// SN of the RLC SDU (or SDU segment) detected as lost
    pub nack_sn: u32,
    /// `(SOstart, SOend)` when the NACK covers a byte range of a partly
    /// received SDU (E2 = 1)
    pub so_range: Option<(u16, u16)>,
    /// Number of consecutively lost SDUs starting at `nack_sn` (E3 = 1)
    pub nack_range: Option<u8>,
}

impl RlcStatusNack {
    /// A whole-SDU NACK: no byte range, no run length.
    pub fn sdu(nack_sn: u32) -> Self {
        Self {
            nack_sn,
            so_range: None,
            nack_range: None,
        }
    }
}

/// RLC AM STATUS PDU (Control PDU) — TS 38.322 §6.2.2.5
///
/// Wire layout, bit-packed and padded to an octet boundary:
///
/// ```text
/// D/C(1) CPT(3) ACK_SN(12|18) E1(1)
///   then, for each NACK: NACK_SN(12|18) E1(1) E2(1) E3(1)
///                        [SOstart(16) SOend(16) if E2] [NACK range(8) if E3]
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RlcStatusPdu {
    /// SN of the next not-received RLC SDU that is not reported as missing:
    /// everything below it is acknowledged, except the NACKed SNs
    /// (TS 38.322 §6.2.3.10)
    pub ack_sn: u32,
    /// Negatively acknowledged SNs, in increasing SN order
    pub nacks: Vec<RlcStatusNack>,
}

/// Control PDU type for a STATUS PDU (TS 38.322 §6.2.3.9)
const CPT_STATUS: u8 = 0b000;

impl RlcStatusPdu {
    /// Create a STATUS PDU acknowledging up to (but not including) `ack_sn`,
    /// with nothing negatively acknowledged.
    pub fn new(ack_sn: u32) -> Self {
        Self {
            ack_sn,
            nacks: Vec::new(),
        }
    }

    /// Create a STATUS PDU with a NACK list.
    pub fn with_nacks(ack_sn: u32, nacks: Vec<RlcStatusNack>) -> Self {
        Self { ack_sn, nacks }
    }

    /// Encode with a 12-bit SN.
    pub fn encode_sn12(&self) -> Vec<u8> {
        self.encode(12)
    }

    /// Encode with an 18-bit SN.
    pub fn encode_sn18(&self) -> Vec<u8> {
        self.encode(18)
    }

    /// Decode a 12-bit-SN STATUS PDU.
    pub fn decode_sn12(buf: &[u8]) -> Result<Self, RlcError> {
        Self::decode(buf, 12)
    }

    /// Decode an 18-bit-SN STATUS PDU.
    pub fn decode_sn18(buf: &[u8]) -> Result<Self, RlcError> {
        Self::decode(buf, 18)
    }

    fn encode(&self, sn_bits: u32) -> Vec<u8> {
        let mut out = BitWriter::new();
        out.push_bits(0, 1); // D/C = 0 (control PDU)
        out.push_bits(u32::from(CPT_STATUS), 3);
        out.push_bits(self.ack_sn, sn_bits);

        // One E1 after ACK_SN announces whether a NACK follows at all; after that
        // each NACK carries its own E1 for the next one. Emitting a leading E1
        // per NACK inserts an extra bit between entries, which a single-NACK
        // fixture cannot see (there it coincides with the trailing E1 = 0).
        out.push_bits(u32::from(!self.nacks.is_empty()), 1);
        for (index, nack) in self.nacks.iter().enumerate() {
            out.push_bits(nack.nack_sn, sn_bits);
            let more = index + 1 < self.nacks.len();
            out.push_bits(u32::from(more), 1); // E1: another NACK follows
            out.push_bits(u32::from(nack.so_range.is_some()), 1); // E2
            out.push_bits(u32::from(nack.nack_range.is_some()), 1); // E3
            if let Some((so_start, so_end)) = nack.so_range {
                out.push_bits(u32::from(so_start), 16);
                out.push_bits(u32::from(so_end), 16);
            }
            if let Some(range) = nack.nack_range {
                out.push_bits(u32::from(range), 8);
            }
        }
        out.finish()
    }

    fn decode(buf: &[u8], sn_bits: u32) -> Result<Self, RlcError> {
        let mut bits = BitReader::new(buf);
        // D/C and CPT are consumed by the caller's dispatch but must still be
        // stepped over here; a non-STATUS CPT is refused rather than parsed as
        // one, because its payload has a different shape entirely.
        bits.take(1)?;
        let cpt = bits.take(3)? as u8;
        if cpt != CPT_STATUS {
            return Err(RlcError::InvalidSi(cpt));
        }
        let ack_sn = bits.take(sn_bits)?;

        let mut nacks = Vec::new();
        let mut more = bits.take(1)? == 1;
        while more {
            let nack_sn = bits.take(sn_bits)?;
            more = bits.take(1)? == 1;
            let has_so = bits.take(1)? == 1;
            let has_range = bits.take(1)? == 1;
            let so_range = if has_so {
                let start = bits.take(16)? as u16;
                let end = bits.take(16)? as u16;
                Some((start, end))
            } else {
                None
            };
            let nack_range = if has_range {
                Some(bits.take(8)? as u8)
            } else {
                None
            };
            nacks.push(RlcStatusNack {
                nack_sn,
                so_range,
                nack_range,
            });
        }

        Ok(Self { ack_sn, nacks })
    }
}

/// Minimal MSB-first bit writer for the STATUS PDU's non-octet-aligned fields.
struct BitWriter {
    out: Vec<u8>,
    /// Bits already written into the last octet of `out`
    used: u32,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            out: Vec::new(),
            used: 8,
        }
    }

    fn push_bits(&mut self, value: u32, bits: u32) {
        for shift in (0..bits).rev() {
            if self.used == 8 {
                self.out.push(0);
                self.used = 0;
            }
            let bit = (value >> shift) & 1;
            if bit == 1 {
                let last = self.out.len() - 1;
                self.out[last] |= 1 << (7 - self.used);
            }
            self.used += 1;
        }
    }

    /// The encoded octets, the trailing partial octet zero-padded (the R bits of
    /// TS 38.322 figure 6.2.2.5-1).
    fn finish(self) -> Vec<u8> {
        self.out
    }
}

/// Minimal MSB-first bit reader, refusing to read past the buffer rather than
/// returning zeros — a truncated STATUS PDU must be an error, not an ACK_SN of 0.
struct BitReader<'a> {
    buf: &'a [u8],
    position: u32,
}

impl<'a> BitReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, position: 0 }
    }

    fn take(&mut self, bits: u32) -> Result<u32, RlcError> {
        let available = self.buf.len() as u32 * 8;
        if self.position + bits > available {
            return Err(RlcError::PduTooShort {
                need: (self.position as usize + bits as usize).div_ceil(8),
                got: self.buf.len(),
            });
        }
        let mut value = 0u32;
        for _ in 0..bits {
            let byte = self.buf[(self.position / 8) as usize];
            let bit = (byte >> (7 - (self.position % 8))) & 1;
            value = (value << 1) | u32::from(bit);
            self.position += 1;
        }
        Ok(value)
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
    /// FLIPPED by issue #103. This round-tripped a complete-SDU PDU with SN 42 and
    /// expected the SN back, which only worked while the header carried one in
    /// violation of TS 38.322 §6.2.2.3.
    ///
    /// A complete SDU has no SN field, so the six bits it used to occupy are
    /// reserved and the SN reads back as 0. Asserted **on the wire bytes**, because
    /// a round trip cannot catch a header-length or field-placement error that both
    /// halves of the codec make.
    #[test]
    fn a_complete_sdu_sn6_header_carries_no_sn() {
        let pdu = RlcUmPdu {
            si: SegmentationInfo::FullSdu,
            sn: 42,
            so: None,
            data: vec![1, 2, 3, 4],
        };
        let encoded = pdu.encode_sn6();

        // One octet: SI in bits 8-7, the six SN bits reserved and zero.
        assert_eq!(encoded.len(), 1 + 4, "a 6-bit-SN header stays one octet");
        assert_eq!(
            encoded[0],
            (SegmentationInfo::FullSdu as u8) << 6,
            "the SN bits must be reserved zeros, not SN 42"
        );
        assert_eq!(&encoded[1..], &[1, 2, 3, 4]);

        let decoded = RlcUmPdu::decode_sn6(&encoded).unwrap();
        assert_eq!(decoded.si, SegmentationInfo::FullSdu);
        assert_eq!(decoded.sn, 0, "there was no SN to recover");
        assert_eq!(decoded.so, None);
        assert_eq!(decoded.data, vec![1, 2, 3, 4]);
    }

    /// And a SEGMENT still carries its SN, on the wire, in the same octet.
    #[test]
    fn a_segmented_sn6_header_still_carries_its_sn() {
        let pdu = RlcUmPdu {
            si: SegmentationInfo::FirstSegment,
            sn: 42,
            so: None,
            data: vec![1, 2, 3, 4],
        };
        let encoded = pdu.encode_sn6();
        assert_eq!(encoded.len(), 1 + 4);
        assert_eq!(
            encoded[0],
            ((SegmentationInfo::FirstSegment as u8) << 6) | 42,
            "a segment's SN is in the low six bits"
        );
        assert_eq!(RlcUmPdu::decode_sn6(&encoded).unwrap(), pdu);
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
    /// FLIPPED by issue #103. This expected a 2-octet header on a complete-SDU PDU;
    /// TS 38.322 §6.2.2.3 gives it **one**, containing only the SI and R fields.
    ///
    /// The extra octet was the interop defect: a conformant peer read it as the
    /// first byte of the SDU. Asserted on the wire bytes for that reason -- a round
    /// trip is blind to a length both halves agree on.
    #[test]
    fn a_complete_sdu_sn12_header_is_one_octet_with_no_sn() {
        let pdu = RlcUmPdu {
            si: SegmentationInfo::FullSdu,
            sn: 0xABC,
            so: None,
            data: vec![0xFF; 10],
        };
        let encoded = pdu.encode_sn12();

        assert_eq!(
            encoded.len(),
            1 + 10,
            "one octet of header, not two -- the second was the defect"
        );
        assert_eq!(
            encoded[0],
            (SegmentationInfo::FullSdu as u8) << 4,
            "SI only; the SN nibble is reserved and zero, not 0xA"
        );
        assert_eq!(&encoded[1..], &[0xFF; 10]);

        let decoded = RlcUmPdu::decode_sn12(&encoded).unwrap();
        assert_eq!(decoded.si, SegmentationInfo::FullSdu);
        assert_eq!(decoded.sn, 0, "there was no SN to recover");
        assert_eq!(decoded.data, vec![0xFF; 10]);
    }

    /// And a SEGMENT still gets its two octets with the SN across both.
    #[test]
    fn a_segmented_sn12_header_still_carries_its_sn_across_two_octets() {
        let pdu = RlcUmPdu {
            si: SegmentationInfo::FirstSegment,
            sn: 0xABC,
            so: None,
            data: vec![0xFF; 10],
        };
        let encoded = pdu.encode_sn12();
        assert_eq!(encoded.len(), 2 + 10);
        assert_eq!(
            encoded[0],
            ((SegmentationInfo::FirstSegment as u8) << 4) | 0x0A,
            "SI plus the SN's high nibble"
        );
        assert_eq!(encoded[1], 0xBC, "the SN's low octet");
        assert_eq!(RlcUmPdu::decode_sn12(&encoded).unwrap(), pdu);
    }

    /// A one-octet complete-SDU PDU must decode, where the old codec demanded two.
    #[test]
    fn a_one_octet_complete_sdu_pdu_decodes() {
        // SI = FullSdu, no SN, no payload at all.
        let bytes = [(SegmentationInfo::FullSdu as u8) << 4];
        let decoded = RlcUmPdu::decode_sn12(&bytes).expect("one octet is a whole PDU");
        assert_eq!(decoded.si, SegmentationInfo::FullSdu);
        assert!(decoded.data.is_empty());

        // A SEGMENT in one octet is still too short, because its SN needs the second.
        let segment = [(SegmentationInfo::FirstSegment as u8) << 4];
        assert!(RlcUmPdu::decode_sn12(&segment).is_err());
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
        assert!(decoded.nacks.is_empty());
    }

    // ── STATUS PDU with a NACK list (#15) ─────────────────────────────────────

    /// The header bits are not octet-aligned, so the layout is asserted on the
    /// wire bytes: D/C=0, CPT=000, ACK_SN(12), E1=1, then NACK_SN(12) E1 E2 E3.
    #[test]
    fn a_status_pdu_with_one_nack_matches_the_ts_38_322_bit_layout() {
        let status = RlcStatusPdu::with_nacks(0x123, vec![RlcStatusNack::sdu(0x456)]);
        let encoded = status.encode_sn12();

        // byte 0: D/C=0 CPT=000 | ACK_SN[11:8] = 0001          -> 0x01
        // byte 1: ACK_SN[7:0] = 0x23                             -> 0x23
        // byte 2: E1=1 | NACK_SN[11:5] = 0100010                 -> 0xA2
        // byte 3: NACK_SN[4:0] = 10110 | E1=0 E2=0 E3=0          -> 0xB0
        assert_eq!(encoded, vec![0x01, 0x23, 0xA2, 0xB0]);
        assert_eq!(RlcStatusPdu::decode_sn12(&encoded).unwrap(), status);
    }

    #[test]
    fn a_status_pdu_round_trips_a_multi_nack_list() {
        let status = RlcStatusPdu::with_nacks(
            2048,
            vec![
                RlcStatusNack::sdu(7),
                RlcStatusNack::sdu(9),
                RlcStatusNack::sdu(4095),
            ],
        );
        let decoded = RlcStatusPdu::decode_sn12(&status.encode_sn12()).unwrap();
        assert_eq!(decoded, status, "every NACK must survive, in order");
    }

    /// The E2 and E3 refinements must survive too: a decoder that skipped them
    /// would read the next NACK_SN out of the middle of an SOstart field.
    #[test]
    fn a_status_pdu_round_trips_so_ranges_and_nack_ranges() {
        let status = RlcStatusPdu::with_nacks(
            100,
            vec![
                RlcStatusNack {
                    nack_sn: 10,
                    so_range: Some((16, 47)),
                    nack_range: None,
                },
                RlcStatusNack {
                    nack_sn: 20,
                    so_range: None,
                    nack_range: Some(5),
                },
                RlcStatusNack {
                    nack_sn: 30,
                    so_range: Some((0, 0xFFFF)),
                    nack_range: Some(3),
                },
                RlcStatusNack::sdu(40),
            ],
        );
        assert_eq!(
            RlcStatusPdu::decode_sn12(&status.encode_sn12()).unwrap(),
            status
        );
    }

    #[test]
    fn a_status_pdu_round_trips_with_an_18_bit_sn() {
        let status = RlcStatusPdu::with_nacks(
            0x3_FFFF,
            vec![RlcStatusNack::sdu(0x2_ABCD), RlcStatusNack::sdu(1)],
        );
        let decoded = RlcStatusPdu::decode_sn18(&status.encode_sn18()).unwrap();
        assert_eq!(decoded, status);
    }

    /// A truncated STATUS PDU must be an error, not an ACK_SN of 0 — which the
    /// sender would read as "nothing acknowledged" and act on.
    #[test]
    fn a_truncated_status_pdu_is_an_error() {
        let full = RlcStatusPdu::with_nacks(5, vec![RlcStatusNack::sdu(2)]).encode_sn12();
        for truncated in 0..full.len() {
            assert!(
                RlcStatusPdu::decode_sn12(&full[..truncated]).is_err(),
                "{truncated} octets must not decode"
            );
        }
    }

    /// A control PDU of another type must be refused rather than parsed as a
    /// STATUS report: its payload has a different shape entirely.
    #[test]
    fn a_control_pdu_of_another_type_is_refused() {
        let mut bytes = RlcStatusPdu::new(5).encode_sn12();
        bytes[0] |= 0b0001_0000; // CPT = 001, reserved
        assert!(RlcStatusPdu::decode_sn12(&bytes).is_err());
    }
}
