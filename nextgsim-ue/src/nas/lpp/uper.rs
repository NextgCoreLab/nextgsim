//! The UNALIGNED PER primitives the LPP codec needs (X.691).
//!
//! LPP is specified with the **unaligned** PER variant, so nothing here aligns to
//! an octet boundary except the final padding of a complete encoding. That is the
//! whole reason this exists rather than reusing the APER machinery the RRC and NGAP
//! codecs generate: an aligned encoder produces different bytes for the same
//! abstract value.
//!
//! Only the constructs the LPP subset in this module uses are implemented, and each
//! is named for the X.691 clause it comes from. A construct that is not here is
//! absent because nothing needs it, not because it was forgotten — the encoders
//! that would need one return an error rather than guessing.
//!
//! ## Byte agreement with the peer
//!
//! The bit layouts mirror `nextgcore`'s `nextgcore-asn1c` LPP codec, which is the
//! LMF side of every exchange this UE will have. Where a layout decision could go
//! either way, the comment says which clause fixes it — because "the two agree"
//! is only worth anything if both are right, and two implementations that agree
//! wrongly is the failure mode a shared enum would have hidden.

use std::fmt;

/// Why an encode or decode failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UperError {
    /// The reader ran out of bits.
    OutOfBits {
        /// Bits requested.
        needed: usize,
        /// Bits left.
        available: usize,
    },
    /// A value fell outside its ASN.1 constraint.
    OutOfRange {
        /// The value.
        value: i64,
        /// Lower bound.
        low: i64,
        /// Upper bound.
        high: i64,
    },
    /// A CHOICE index named an alternative that does not exist.
    InvalidChoiceIndex {
        /// The index read.
        index: usize,
        /// The highest valid index.
        max: usize,
    },
    /// A length determinant fell outside its constraint.
    InvalidLength(usize),
    /// A construct this codec deliberately does not implement.
    Unsupported(&'static str),
}

impl fmt::Display for UperError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfBits { needed, available } => {
                write!(f, "out of bits: needed {needed}, {available} available")
            }
            Self::OutOfRange { value, low, high } => {
                write!(f, "value {value} outside constraint ({low}..{high})")
            }
            Self::InvalidChoiceIndex { index, max } => {
                write!(f, "CHOICE index {index} exceeds the maximum {max}")
            }
            Self::InvalidLength(len) => write!(f, "length {len} outside its constraint"),
            Self::Unsupported(what) => write!(f, "unsupported LPP construct: {what}"),
        }
    }
}

impl std::error::Error for UperError {}

/// Result alias for this codec.
pub type UperResult<T> = Result<T, UperError>;

/// Number of bits needed to carry `0..=range_max_minus_min` (X.691 11.5.7.4).
///
/// A range of exactly one value needs **zero** bits, which is the case a naive
/// `log2` gets wrong and which appears in real ASN.1 wherever a constraint pins a
/// single value. The loop below handles it without a special case — `1 << 0` is 1,
/// which is not `<= 0`, so it never runs. An explicit early return for it was here
/// and removed: the revert harness proved it dead.
fn bits_for_range(range: u64) -> usize {
    let mut bits = 0;
    while (1u64 << bits) <= range {
        bits += 1;
    }
    bits
}

/// Bit-level writer.
#[derive(Debug, Default)]
pub struct UperWriter {
    bytes: Vec<u8>,
    /// Bits already written into the last byte, 0..8.
    bit: usize,
}

impl UperWriter {
    /// A new, empty writer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Write one bit.
    pub fn write_bit(&mut self, bit: bool) {
        if self.bit == 0 {
            self.bytes.push(0);
        }
        if bit {
            let last = self.bytes.len() - 1;
            self.bytes[last] |= 0x80 >> self.bit;
        }
        self.bit = (self.bit + 1) % 8;
    }

    /// Write the `count` least significant bits of `value`, most significant first.
    pub fn write_bits(&mut self, value: u64, count: usize) {
        for index in (0..count).rev() {
            self.write_bit((value >> index) & 1 == 1);
        }
    }

    /// How many bits have been written.
    pub fn bit_len(&self) -> usize {
        if self.bit == 0 {
            self.bytes.len() * 8
        } else {
            (self.bytes.len() - 1) * 8 + self.bit
        }
    }

    /// Finish, padding the final octet with zeros (X.691 11.1.2).
    ///
    /// The same clause makes a zero-bit encoding **one** zero octet rather than none:
    /// a peer handed zero octets has nothing to read at all, so it would report a
    /// truncated message instead of an empty one.
    pub fn into_bytes(mut self) -> Vec<u8> {
        if self.bytes.is_empty() {
            self.bytes.push(0);
        }
        self.bytes
    }

    /// A constrained whole number (X.691 13.2.2): the offset from the lower bound
    /// in exactly as many bits as the range needs, with no alignment.
    ///
    /// # Errors
    /// [`UperError::OutOfRange`] when `value` is outside `low..=high`. Refused
    /// rather than clamped: a clamped value is a different abstract value, and the
    /// peer would decode it without any sign that something was wrong.
    pub fn write_constrained(&mut self, value: i64, low: i64, high: i64) -> UperResult<()> {
        if value < low || value > high {
            return Err(UperError::OutOfRange { value, low, high });
        }
        let range = (high - low) as u64;
        self.write_bits((value - low) as u64, bits_for_range(range));
        Ok(())
    }

    /// A SEQUENCE preamble (X.691 18.1-18.2): the extension bit when the type is
    /// extensible, then one presence bit per OPTIONAL/DEFAULT root member, in
    /// declaration order.
    pub fn write_sequence_preamble(&mut self, extensible: Option<bool>, optionals: &[bool]) {
        if let Some(additions_follow) = extensible {
            self.write_bit(additions_follow);
        }
        for &present in optionals {
            self.write_bit(present);
        }
    }

    /// A CHOICE index (X.691 23).
    ///
    /// # Errors
    /// [`UperError::InvalidChoiceIndex`] when the index names no alternative.
    pub fn write_choice_index(&mut self, index: usize, alternatives: usize) -> UperResult<()> {
        if index >= alternatives {
            return Err(UperError::InvalidChoiceIndex {
                index,
                max: alternatives.saturating_sub(1),
            });
        }
        self.write_constrained(index as i64, 0, (alternatives - 1) as i64)
    }

    /// An ENUMERATED value of an extensible enumeration (X.691 14): the extension
    /// bit, then the index as a constrained whole number over the root.
    ///
    /// # Errors
    /// [`UperError::OutOfRange`] when the value is not a root value.
    pub fn write_extensible_enumerated(&mut self, value: i64, root_max: i64) -> UperResult<()> {
        self.write_bit(false); // a root value, not an extension addition
        self.write_constrained(value, 0, root_max)
    }

    /// A BIT STRING with a constrained size (X.691 16): the length as a constrained
    /// whole number, then the bits. `low == high` writes the bits alone.
    ///
    /// # Errors
    /// [`UperError::InvalidLength`] when the bit count is outside `low..=high`.
    pub fn write_bit_string(&mut self, bits: &[bool], low: usize, high: usize) -> UperResult<()> {
        if bits.len() < low || bits.len() > high {
            return Err(UperError::InvalidLength(bits.len()));
        }
        if low != high {
            self.write_constrained(bits.len() as i64, low as i64, high as i64)?;
        }
        for &bit in bits {
            self.write_bit(bit);
        }
        Ok(())
    }
}

/// Bit-level reader.
#[derive(Debug)]
pub struct UperReader<'a> {
    bytes: &'a [u8],
    /// Absolute bit position.
    position: usize,
}

impl<'a> UperReader<'a> {
    /// A reader over `bytes`.
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    /// Bits not yet consumed.
    pub fn remaining_bits(&self) -> usize {
        (self.bytes.len() * 8).saturating_sub(self.position)
    }

    /// Read one bit.
    ///
    /// # Errors
    /// [`UperError::OutOfBits`] at the end of the buffer.
    pub fn read_bit(&mut self) -> UperResult<bool> {
        if self.remaining_bits() == 0 {
            return Err(UperError::OutOfBits {
                needed: 1,
                available: 0,
            });
        }
        let byte = self.bytes[self.position / 8];
        let bit = (byte >> (7 - (self.position % 8))) & 1 == 1;
        self.position += 1;
        Ok(bit)
    }

    /// Read `count` bits, most significant first.
    ///
    /// # Errors
    /// [`UperError::OutOfBits`] when fewer than `count` bits remain.
    pub fn read_bits(&mut self, count: usize) -> UperResult<u64> {
        if self.remaining_bits() < count {
            return Err(UperError::OutOfBits {
                needed: count,
                available: self.remaining_bits(),
            });
        }
        let mut value = 0u64;
        for _ in 0..count {
            value = (value << 1) | u64::from(self.read_bit()?);
        }
        Ok(value)
    }

    /// A constrained whole number.
    ///
    /// # Errors
    /// [`UperError::OutOfBits`] when the buffer is short.
    pub fn read_constrained(&mut self, low: i64, high: i64) -> UperResult<i64> {
        let range = (high - low) as u64;
        let offset = self.read_bits(bits_for_range(range))?;
        Ok(low + offset as i64)
    }

    /// A SEQUENCE preamble. Returns `(additions_follow, presence_bits)`.
    ///
    /// # Errors
    /// [`UperError::OutOfBits`] when the buffer is short.
    pub fn read_sequence_preamble(
        &mut self,
        extensible: bool,
        optionals: usize,
    ) -> UperResult<(bool, Vec<bool>)> {
        let additions_follow = if extensible { self.read_bit()? } else { false };
        let mut presence = Vec::with_capacity(optionals);
        for _ in 0..optionals {
            presence.push(self.read_bit()?);
        }
        Ok((additions_follow, presence))
    }

    /// A CHOICE index.
    ///
    /// # Errors
    /// [`UperError::OutOfBits`] when the buffer is short.
    pub fn read_choice_index(&mut self, alternatives: usize) -> UperResult<usize> {
        Ok(self.read_constrained(0, (alternatives - 1) as i64)? as usize)
    }

    /// An ENUMERATED value of an extensible enumeration.
    ///
    /// # Errors
    /// [`UperError::Unsupported`] when the peer sent an extension addition, which
    /// this codec cannot interpret — reported rather than read as a root value,
    /// because reading it as one would silently substitute a different enumerator.
    pub fn read_extensible_enumerated(&mut self, root_max: i64) -> UperResult<i64> {
        if self.read_bit()? {
            return Err(UperError::Unsupported("ENUMERATED extension addition"));
        }
        self.read_constrained(0, root_max)
    }

    /// A BIT STRING with a constrained size.
    ///
    /// # Errors
    /// [`UperError::OutOfBits`] or [`UperError::InvalidLength`].
    pub fn read_bit_string(&mut self, low: usize, high: usize) -> UperResult<Vec<bool>> {
        let len = if low == high {
            low
        } else {
            self.read_constrained(low as i64, high as i64)? as usize
        };
        if len < low || len > high {
            return Err(UperError::InvalidLength(len));
        }
        let mut bits = Vec::with_capacity(len);
        for _ in 0..len {
            bits.push(self.read_bit()?);
        }
        Ok(bits)
    }

    /// Consume an extension-addition block (X.691 18.7-18.9) without interpreting
    /// it: the normally-small bit-map length, the presence bits, then each present
    /// addition as an open type (length determinant + that many octets).
    ///
    /// Skipping rather than parsing is what forward compatibility means here: a
    /// newer peer's additions must not make the root fields unreadable.
    ///
    /// # Errors
    /// [`UperError::Unsupported`] for a fragmented or long-form open type, which
    /// this codec cannot skip correctly and must not pretend to.
    pub fn skip_extension_additions(&mut self) -> UperResult<()> {
        // Normally-small non-negative number (X.691 11.6): a leading 0 bit then 6
        // bits of (count - 1).
        if self.read_bit()? {
            return Err(UperError::Unsupported(
                "extension-addition bit-map length above 64",
            ));
        }
        let count = self.read_bits(6)? as usize + 1;
        let mut present = Vec::with_capacity(count);
        for _ in 0..count {
            present.push(self.read_bit()?);
        }
        for is_present in present {
            if !is_present {
                continue;
            }
            // Open-type length determinant (X.691 11.9): the short form only, which
            // covers additions below 128 octets.
            let length = self.read_bits(8)?;
            if length & 0x80 != 0 {
                return Err(UperError::Unsupported(
                    "long-form or fragmented open-type length",
                ));
            }
            let bits = (length as usize) * 8;
            if self.remaining_bits() < bits {
                return Err(UperError::OutOfBits {
                    needed: bits,
                    available: self.remaining_bits(),
                });
            }
            self.position += bits;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_single_value_range_costs_no_bits() {
        // X.691 11.5.7.4. The case a naive log2 gets wrong, and it appears wherever
        // a constraint pins one value.
        assert_eq!(bits_for_range(0), 0);
        let mut w = UperWriter::new();
        w.write_constrained(5, 5, 5).expect("in range");
        assert_eq!(w.bit_len(), 0, "no bits for a one-value range");
    }

    #[test]
    fn an_empty_encoding_is_one_zero_octet_and_not_zero_octets() {
        // X.691 11.1.2. Nothing in the LPP subset produces a zero-bit encoding today
        // -- the shortest message is five bits -- so this is the writer's contract
        // being kept rather than a case the codec reaches. Tested directly because
        // the revert harness showed no LPP-level test could reach it.
        assert_eq!(UperWriter::new().into_bytes(), vec![0u8]);

        // One bit still rounds up to one octet, so the rule is about EMPTINESS and
        // not about padding in general.
        let mut w = UperWriter::new();
        w.write_bit(false);
        assert_eq!(w.into_bytes(), vec![0u8]);
    }

    #[test]
    fn a_constrained_number_uses_exactly_the_bits_its_range_needs() {
        // 0..255 is 8 bits; 0..503 is 9; 0..1 is 1; 0..2 is 2 (not 1).
        for (low, high, bits) in [
            (0i64, 1i64, 1usize),
            (0, 2, 2),
            (0, 3, 2),
            (0, 4, 3),
            (0, 34, 6),
            (0, 97, 7),
            (0, 255, 8),
            (0, 503, 9),
            (0, 4095, 12),
            (0, 65535, 16),
        ] {
            let mut w = UperWriter::new();
            w.write_constrained(low, low, high).expect("in range");
            assert_eq!(w.bit_len(), bits, "range {low}..{high}");
        }
    }

    #[test]
    fn a_constrained_number_round_trips_at_both_bounds_and_between() {
        for (low, high) in [(0i64, 255i64), (0, 503), (1, 32), (0, 97), (-5, 5)] {
            for value in [low, low + 1, (low + high) / 2, high] {
                let mut w = UperWriter::new();
                w.write_constrained(value, low, high).expect("in range");
                let bytes = w.into_bytes();
                let mut r = UperReader::new(&bytes);
                assert_eq!(
                    r.read_constrained(low, high).expect("readable"),
                    value,
                    "{value} in {low}..{high}"
                );
            }
        }
    }

    #[test]
    fn an_out_of_range_value_is_refused_rather_than_clamped() {
        // A clamped value is a DIFFERENT abstract value, and the peer would decode
        // it with no sign anything was wrong.
        let mut w = UperWriter::new();
        assert_eq!(
            w.write_constrained(300, 0, 255),
            Err(UperError::OutOfRange {
                value: 300,
                low: 0,
                high: 255
            })
        );
        assert!(w.write_constrained(-1, 0, 255).is_err());
    }

    #[test]
    fn bits_are_written_most_significant_first_and_padded_at_the_end() {
        let mut w = UperWriter::new();
        w.write_bits(0b101, 3);
        assert_eq!(w.bit_len(), 3);
        // 101 then five zero pad bits.
        assert_eq!(w.into_bytes(), vec![0b1010_0000]);
    }

    #[test]
    fn a_sequence_preamble_writes_the_extension_bit_before_the_presence_bits() {
        let mut w = UperWriter::new();
        w.write_sequence_preamble(Some(false), &[true, false, true]);
        let bytes = w.into_bytes();
        let mut r = UperReader::new(&bytes);
        let (ext, opts) = r.read_sequence_preamble(true, 3).expect("readable");
        assert!(!ext);
        assert_eq!(opts, vec![true, false, true]);

        // A non-extensible SEQUENCE writes no extension bit at all, so the same
        // presence bits occupy one bit less.
        let mut w = UperWriter::new();
        w.write_sequence_preamble(None, &[true, false, true]);
        assert_eq!(w.bit_len(), 3);
    }

    #[test]
    fn a_choice_index_round_trips_and_an_invalid_one_is_refused() {
        for (index, alternatives) in [(0usize, 2usize), (1, 2), (0, 16), (5, 16), (15, 16)] {
            let mut w = UperWriter::new();
            w.write_choice_index(index, alternatives).expect("valid");
            let bytes = w.into_bytes();
            let mut r = UperReader::new(&bytes);
            assert_eq!(r.read_choice_index(alternatives).expect("readable"), index);
        }
        let mut w = UperWriter::new();
        assert_eq!(
            w.write_choice_index(2, 2),
            Err(UperError::InvalidChoiceIndex { index: 2, max: 1 })
        );
    }

    #[test]
    fn an_extensible_enumerated_round_trips_and_an_addition_is_reported() {
        let mut w = UperWriter::new();
        w.write_extensible_enumerated(1, 1).expect("root value");
        let bytes = w.into_bytes();
        let mut r = UperReader::new(&bytes);
        assert_eq!(r.read_extensible_enumerated(1).expect("readable"), 1);

        // An extension addition must be REPORTED, not read as a root value --
        // reading it as one silently substitutes a different enumerator.
        let mut w = UperWriter::new();
        w.write_bit(true);
        w.write_bits(0, 7);
        let bytes = w.into_bytes();
        let mut r = UperReader::new(&bytes);
        assert_eq!(
            r.read_extensible_enumerated(1),
            Err(UperError::Unsupported("ENUMERATED extension addition"))
        );
    }

    #[test]
    fn a_constrained_bit_string_carries_its_length_unless_it_is_fixed() {
        // SIZE(1..8): a length field then the bits.
        let bits = vec![true, false, true];
        let mut w = UperWriter::new();
        w.write_bit_string(&bits, 1, 8).expect("in range");
        let bytes = w.into_bytes();
        let mut r = UperReader::new(&bytes);
        assert_eq!(r.read_bit_string(1, 8).expect("readable"), bits);

        // SIZE(10): the bits alone, so exactly 10 bits.
        let sfn = vec![true; 10];
        let mut w = UperWriter::new();
        w.write_bit_string(&sfn, 10, 10).expect("in range");
        assert_eq!(w.bit_len(), 10, "a fixed size carries no length field");

        // And a wrong length for a fixed size is refused.
        let mut w = UperWriter::new();
        assert_eq!(
            w.write_bit_string(&[true; 9], 10, 10),
            Err(UperError::InvalidLength(9))
        );
    }

    #[test]
    fn a_short_buffer_is_reported_rather_than_read_as_zeros() {
        let mut r = UperReader::new(&[]);
        assert!(r.read_bit().is_err());
        let mut r = UperReader::new(&[0xFF]);
        assert_eq!(
            r.read_bits(9),
            Err(UperError::OutOfBits {
                needed: 9,
                available: 8
            })
        );
    }

    #[test]
    fn an_extension_addition_block_is_skipped_leaving_the_reader_positioned_after_it() {
        // One addition of 2 octets, then a marker bit this test reads back.
        let mut w = UperWriter::new();
        w.write_bit(false); // normally-small form
        w.write_bits(0, 6); // count - 1 = 0, i.e. one addition
        w.write_bit(true); // that addition is present
        w.write_bits(2, 8); // open-type length: 2 octets
        w.write_bits(0xABCD, 16);
        w.write_bit(true); // the marker
        let bytes = w.into_bytes();

        let mut r = UperReader::new(&bytes);
        r.skip_extension_additions().expect("skippable");
        assert!(r.read_bit().expect("the marker survives"));
    }

    #[test]
    fn a_long_form_open_type_is_reported_rather_than_mis_skipped() {
        // Skipping the wrong number of octets would leave every later field
        // misaligned, which is worse than refusing.
        let mut w = UperWriter::new();
        w.write_bit(false);
        w.write_bits(0, 6);
        w.write_bit(true);
        w.write_bits(0x80, 8); // long form
        let bytes = w.into_bytes();
        let mut r = UperReader::new(&bytes);
        assert_eq!(
            r.skip_extension_additions(),
            Err(UperError::Unsupported(
                "long-form or fragmented open-type length"
            ))
        );
    }
}
