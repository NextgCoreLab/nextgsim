//! BitString type for bit-level operations.
//!
//! This module provides the `BitString` type for building and manipulating
//! sequences of bits, commonly used in ASN.1 encoding and protocol messages.

use std::fmt;

/// A variable-length sequence of bits.
///
/// `BitString` provides methods for writing individual bits and multi-bit values.
/// It automatically manages the underlying byte storage and tracks the current
/// bit position.
///
/// Bits are written in big-endian order (MSB first within each byte).
#[derive(Clone, PartialEq, Eq, Hash, Default)]
pub struct BitString {
    /// Underlying byte storage
    data: Vec<u8>,
    /// Current bit index (total bits written)
    bit_index: usize,
}

impl BitString {
    /// Creates a new empty `BitString`.
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            bit_index: 0,
        }
    }

    /// Creates a `BitString` with pre-allocated capacity for the given number of bits.
    pub fn with_capacity(bits: usize) -> Self {
        let bytes = (bits + 7) / 8;
        Self {
            data: vec![0u8; bytes],
            bit_index: 0,
        }
    }

    /// Creates a `BitString` from existing bytes and bit length.
    ///
    /// # Arguments
    /// * `data` - The byte data
    /// * `bit_length` - The number of valid bits in the data
    pub fn from_bytes(data: Vec<u8>, bit_length: usize) -> Self {
        Self {
            data,
            bit_index: bit_length,
        }
    }

    /// Writes a single bit.
    ///
    /// # Arguments
    /// * `bit` - The bit value to write (true = 1, false = 0)
    pub fn write(&mut self, bit: bool) {
        let octet_index = self.bit_index / 8;
        let bit_offset = self.bit_index % 8;

        // Ensure we have enough space
        if octet_index >= self.data.len() {
            self.data.push(0);
        }

        if bit {
            // Set the bit (MSB first within byte)
            self.data[octet_index] |= 1 << (7 - bit_offset);
        } else {
            // Clear the bit
            self.data[octet_index] &= !(1 << (7 - bit_offset));
        }

        self.bit_index += 1;
    }

    /// Writes multiple bits from an integer value.
    ///
    /// Bits are written MSB first. For example, `write_bits(0b101, 3)` writes
    /// bits 1, 0, 1 in that order.
    ///
    /// # Arguments
    /// * `value` - The integer value containing the bits to write
    /// * `len` - The number of bits to write (1-32)
    ///
    /// # Panics
    /// Panics if `len` is 0 or greater than 32.
    pub fn write_bits(&mut self, value: u32, len: usize) {
        if len == 0 {
            return;
        }
        assert!(len <= 32, "Cannot write more than 32 bits at once");

        for i in 0..len {
            let bit = (value >> (len - 1 - i)) & 1;
            self.write(bit != 0);
        }
    }

    /// Writes multiple bits from a 64-bit integer value.
    ///
    /// # Arguments
    /// * `value` - The 64-bit integer value containing the bits to write
    /// * `len` - The number of bits to write (1-64)
    ///
    /// # Panics
    /// Panics if `len` is 0 or greater than 64.
    pub fn write_bits_u64(&mut self, value: u64, len: usize) {
        if len == 0 {
            return;
        }
        assert!(len <= 64, "Cannot write more than 64 bits at once");

        for i in 0..len {
            let bit = (value >> (len - 1 - i)) & 1;
            self.write(bit != 0);
        }
    }

    /// Reads a single bit at the given index.
    ///
    /// # Arguments
    /// * `index` - The bit index to read
    ///
    /// # Returns
    /// The bit value (true = 1, false = 0)
    ///
    /// # Panics
    /// Panics if `index >= bit_length()`.
    pub fn read(&self, index: usize) -> bool {
        assert!(index < self.bit_index, "Bit index out of bounds");
        let octet_index = index / 8;
        let bit_offset = index % 8;
        (self.data[octet_index] >> (7 - bit_offset)) & 1 != 0
    }

    /// Reads multiple bits starting at the given index.
    ///
    /// # Arguments
    /// * `index` - The starting bit index
    /// * `len` - The number of bits to read (1-32)
    ///
    /// # Returns
    /// The bits as a u32 value
    ///
    /// # Panics
    /// Panics if the read would exceed the bit length.
    pub fn read_bits(&self, index: usize, len: usize) -> u32 {
        assert!(len <= 32, "Cannot read more than 32 bits at once");
        assert!(
            index + len <= self.bit_index,
            "Read would exceed bit length"
        );

        let mut result = 0u32;
        for i in 0..len {
            result <<= 1;
            if self.read(index + i) {
                result |= 1;
            }
        }
        result
    }

    /// Returns the total number of bits written.
    pub fn bit_length(&self) -> usize {
        self.bit_index
    }

    /// Returns the number of octets needed to store the bits.
    ///
    /// This is the ceiling of `bit_length() / 8`.
    pub fn octet_length(&self) -> usize {
        (self.bit_index + 7) / 8
    }

    /// Returns a reference to the underlying byte data.
    pub fn data(&self) -> &[u8] {
        &self.data[..self.octet_length()]
    }

    /// Returns the underlying byte data, consuming the `BitString`.
    pub fn into_bytes(mut self) -> Vec<u8> {
        self.data.truncate(self.octet_length());
        self.data
    }

    /// Returns true if no bits have been written.
    pub fn is_empty(&self) -> bool {
        self.bit_index == 0
    }

    /// Clears all bits and resets the bit index.
    pub fn clear(&mut self) {
        self.data.clear();
        self.bit_index = 0;
    }

    /// Aligns the bit index to the next octet boundary by writing zero bits.
    pub fn octet_align(&mut self) {
        let remainder = self.bit_index % 8;
        if remainder != 0 {
            let padding = 8 - remainder;
            for _ in 0..padding {
                self.write(false);
            }
        }
    }

    /// Returns the number of unused bits in the last octet.
    ///
    /// This is useful for ASN.1 BIT STRING encoding where the number of
    /// unused bits must be indicated.
    pub fn unused_bits(&self) -> usize {
        if self.bit_index == 0 {
            0
        } else {
            (8 - (self.bit_index % 8)) % 8
        }
    }
}

impl fmt::Debug for BitString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BitString({} bits: ", self.bit_index)?;
        for i in 0..self.bit_index {
            write!(f, "{}", if self.read(i) { '1' } else { '0' })?;
        }
        write!(f, ")")
    }
}

impl fmt::Display for BitString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.bit_index {
            write!(f, "{}", if self.read(i) { '1' } else { '0' })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let bs = BitString::new();
        assert!(bs.is_empty());
        assert_eq!(bs.bit_length(), 0);
        assert_eq!(bs.octet_length(), 0);
    }

    #[test]
    fn test_write_single_bit() {
        let mut bs = BitString::new();
        bs.write(true);
        assert_eq!(bs.bit_length(), 1);
        assert_eq!(bs.octet_length(), 1);
        assert!(bs.read(0));
    }

    #[test]
    fn test_write_multiple_bits() {
        let mut bs = BitString::new();
        bs.write(true);
        bs.write(false);
        bs.write(true);
        bs.write(true);
        assert_eq!(bs.bit_length(), 4);
        assert!(bs.read(0)); // 1
        assert!(!bs.read(1)); // 0
        assert!(bs.read(2)); // 1
        assert!(bs.read(3)); // 1
    }

    #[test]
    fn test_write_bits() {
        let mut bs = BitString::new();
        bs.write_bits(0b10110, 5);
        assert_eq!(bs.bit_length(), 5);
        assert!(bs.read(0)); // 1
        assert!(!bs.read(1)); // 0
        assert!(bs.read(2)); // 1
        assert!(bs.read(3)); // 1
        assert!(!bs.read(4)); // 0
    }

    #[test]
    fn test_write_bits_full_byte() {
        let mut bs = BitString::new();
        bs.write_bits(0xAB, 8); // 10101011
        assert_eq!(bs.bit_length(), 8);
        assert_eq!(bs.octet_length(), 1);
        assert_eq!(bs.data()[0], 0xAB);
    }

    #[test]
    fn test_write_bits_multiple_bytes() {
        let mut bs = BitString::new();
        bs.write_bits(0x1234, 16);
        assert_eq!(bs.bit_length(), 16);
        assert_eq!(bs.octet_length(), 2);
        assert_eq!(bs.data(), &[0x12, 0x34]);
    }

    #[test]
    fn test_read_bits() {
        let mut bs = BitString::new();
        bs.write_bits(0b10110011, 8);
        assert_eq!(bs.read_bits(0, 4), 0b1011);
        assert_eq!(bs.read_bits(4, 4), 0b0011);
        assert_eq!(bs.read_bits(2, 4), 0b1100);
    }

    #[test]
    fn test_octet_length() {
        let mut bs = BitString::new();
        assert_eq!(bs.octet_length(), 0);

        bs.write(true);
        assert_eq!(bs.octet_length(), 1);

        for _ in 0..7 {
            bs.write(false);
        }
        assert_eq!(bs.octet_length(), 1);

        bs.write(true);
        assert_eq!(bs.octet_length(), 2);
    }

    #[test]
    fn test_octet_align() {
        let mut bs = BitString::new();
        bs.write_bits(0b101, 3);
        assert_eq!(bs.bit_length(), 3);

        bs.octet_align();
        assert_eq!(bs.bit_length(), 8);
        assert_eq!(bs.data()[0], 0b10100000);
    }

    #[test]
    fn test_octet_align_already_aligned() {
        let mut bs = BitString::new();
        bs.write_bits(0xFF, 8);
        bs.octet_align();
        assert_eq!(bs.bit_length(), 8);
    }

    #[test]
    fn test_unused_bits() {
        let mut bs = BitString::new();
        assert_eq!(bs.unused_bits(), 0);

        bs.write(true);
        assert_eq!(bs.unused_bits(), 7);

        bs.write_bits(0, 7);
        assert_eq!(bs.unused_bits(), 0);

        bs.write(true);
        assert_eq!(bs.unused_bits(), 7);
    }

    #[test]
    fn test_clear() {
        let mut bs = BitString::new();
        bs.write_bits(0xFFFF, 16);
        assert_eq!(bs.bit_length(), 16);

        bs.clear();
        assert!(bs.is_empty());
        assert_eq!(bs.bit_length(), 0);
    }

    #[test]
    fn test_from_bytes() {
        let bs = BitString::from_bytes(vec![0xAB, 0xCD], 12);
        assert_eq!(bs.bit_length(), 12);
        assert_eq!(bs.octet_length(), 2);
        assert!(bs.read(0)); // 1
        assert!(!bs.read(1)); // 0
    }

    #[test]
    fn test_into_bytes() {
        let mut bs = BitString::new();
        bs.write_bits(0xABCD, 16);
        let bytes = bs.into_bytes();
        assert_eq!(bytes, vec![0xAB, 0xCD]);
    }

    #[test]
    fn test_display() {
        let mut bs = BitString::new();
        bs.write_bits(0b10110, 5);
        assert_eq!(format!("{}", bs), "10110");
    }

    #[test]
    fn test_equality() {
        let mut bs1 = BitString::new();
        bs1.write_bits(0xAB, 8);

        let mut bs2 = BitString::new();
        bs2.write_bits(0xAB, 8);

        let mut bs3 = BitString::new();
        bs3.write_bits(0xAC, 8);

        assert_eq!(bs1, bs2);
        assert_ne!(bs1, bs3);
    }

    #[test]
    fn test_write_bits_u64() {
        let mut bs = BitString::new();
        bs.write_bits_u64(0x123456789ABCDEF0, 64);
        assert_eq!(bs.bit_length(), 64);
        assert_eq!(bs.octet_length(), 8);
        assert_eq!(
            bs.data(),
            &[0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0]
        );
    }
}
