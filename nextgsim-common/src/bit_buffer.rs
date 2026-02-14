//! `BitBuffer` - A bit-level read/write buffer for ASN.1 encoding.
//!
//! This module provides `BitBuffer`, which allows reading and writing
//! individual bits and multi-bit values. It's commonly used for ASN.1
//! PER/UPER encoding in NGAP and RRC protocols.

use std::cell::Cell;

/// A bit-level read/write buffer.
///
/// `BitBuffer` wraps a mutable byte slice and provides methods for reading
/// and writing individual bits and multi-bit values. Bits are read/written
/// in MSB-first order within each byte.
///
/// # Example
/// ```
/// use nextgsim_common::BitBuffer;
///
/// let mut data = [0u8; 4];
/// let mut buffer = BitBuffer::new(&mut data);
///
/// // Write some bits
/// buffer.write_bits(0b1010, 4);
/// buffer.write_bits(0b1100, 4);
///
/// // Seek back and read
/// buffer.seek(0);
/// assert_eq!(buffer.read_bits(4), 0b1010);
/// assert_eq!(buffer.read_bits(4), 0b1100);
/// ```
pub struct BitBuffer<'a> {
    data: &'a mut [u8],
    index: Cell<usize>, // bit index
}

impl<'a> BitBuffer<'a> {
    /// Creates a new `BitBuffer` from a mutable byte slice.
    pub fn new(data: &'a mut [u8]) -> Self {
        Self {
            data,
            index: Cell::new(0),
        }
    }

    /// Seeks to the specified bit index.
    #[inline]
    pub fn seek(&self, index: usize) {
        self.index.set(index);
    }

    /// Returns the current bit index.
    #[inline]
    pub fn current_index(&self) -> usize {
        self.index.get()
    }

    /// Peeks at the current bit without advancing the index.
    ///
    /// Returns 0 or 1.
    #[inline]
    pub fn peek(&self) -> i32 {
        let octet_index = self.index.get() / 8;
        let bit_index = self.index.get() % 8;
        ((self.data[octet_index] >> (7 - bit_index)) & 1) as i32
    }

    /// Reads a single bit and advances the index.
    ///
    /// Returns 0 or 1.
    #[inline]
    pub fn read(&self) -> i32 {
        let octet_index = self.index.get() / 8;
        let bit_index = self.index.get() % 8;
        self.index.set(self.index.get() + 1);
        ((self.data[octet_index] >> (7 - bit_index)) & 1) as i32
    }

    /// Reads multiple bits and returns them as an i32.
    ///
    /// # Arguments
    /// * `len` - Number of bits to read (1-31)
    ///
    /// # Panics
    /// Panics if `len` is 0 or >= 32.
    #[inline]
    pub fn read_bits(&self, len: usize) -> i32 {
        assert!(len > 0 && len < 32, "len must be between 1 and 31");

        let mut result = 0i32;
        for _ in 0..len {
            result <<= 1;
            result |= self.read();
        }
        result
    }

    /// Reads multiple bits and returns them as an i64.
    ///
    /// # Arguments
    /// * `len` - Number of bits to read (1-64)
    ///
    /// # Panics
    /// Panics if `len` is 0 or > 64.
    #[inline]
    pub fn read_bits_long(&self, len: usize) -> i64 {
        assert!(len > 0 && len <= 64, "len must be between 1 and 64");

        let mut result = 0i64;
        for _ in 0..len {
            result <<= 1;
            result |= self.read() as i64;
        }
        result
    }

    /// Writes a single bit.
    ///
    /// # Arguments
    /// * `bit` - The bit value (true = 1, false = 0)
    #[inline]
    pub fn write(&mut self, bit: bool) {
        let octet_index = self.index.get() / 8;
        let bit_index = self.index.get() % 8;

        if bit {
            self.data[octet_index] |= 1 << (7 - bit_index);
        } else {
            self.data[octet_index] &= !(1 << (7 - bit_index));
        }
        self.index.set(self.index.get() + 1);
    }

    /// Writes multiple bits from an i32 value.
    ///
    /// # Arguments
    /// * `value` - The value containing the bits to write
    /// * `len` - Number of bits to write (0-32)
    ///
    /// # Panics
    /// Panics if `len` > 32.
    #[inline]
    pub fn write_bits(&mut self, value: i32, len: usize) {
        if len == 0 {
            return;
        }
        assert!(len <= 32, "len must be <= 32");

        for i in 0..len {
            let bit = ((value >> (len - 1 - i)) & 1) != 0;
            self.write(bit);
        }
    }

    /// Writes multiple bits from an i64 value.
    ///
    /// # Arguments
    /// * `value` - The value containing the bits to write
    /// * `len` - Number of bits to write (0-64)
    ///
    /// # Panics
    /// Panics if `len` > 64.
    #[inline]
    pub fn write_bits_long(&mut self, value: i64, len: usize) {
        if len == 0 {
            return;
        }
        assert!(len <= 64, "len must be <= 64");

        for i in 0..len {
            let bit = ((value >> (len - 1 - i)) & 1) != 0;
            self.write(bit);
        }
    }

    /// Returns the total number of octets written.
    ///
    /// This rounds up to the nearest byte boundary.
    #[inline]
    pub fn written_octets(&self) -> usize {
        let idx = self.index.get();
        idx.div_ceil(8)
    }

    /// Aligns the buffer to the next octet boundary by writing zero bits.
    #[inline]
    pub fn octet_align(&mut self) {
        let remainder = self.index.get() % 8;
        if remainder != 0 {
            self.write_bits(0, 8 - remainder);
        }
    }

    /// Returns the number of bits until the next octet boundary.
    #[inline]
    pub fn bits_to_octet_boundary(&self) -> usize {
        let remainder = self.index.get() % 8;
        if remainder == 0 {
            0
        } else {
            8 - remainder
        }
    }

    /// Returns a reference to the underlying data.
    #[inline]
    pub fn data(&self) -> &[u8] {
        self.data
    }

    /// Returns the total capacity in bits.
    #[inline]
    pub fn capacity_bits(&self) -> usize {
        self.data.len() * 8
    }

    /// Returns the remaining bits that can be written.
    #[inline]
    pub fn remaining_bits(&self) -> usize {
        self.capacity_bits().saturating_sub(self.index.get())
    }
}

/// A read-only bit buffer for parsing bit-level data.
///
/// Unlike `BitBuffer`, this doesn't require mutable access to the underlying data.
pub struct BitBufferReader<'a> {
    data: &'a [u8],
    index: Cell<usize>, // bit index
}

impl<'a> BitBufferReader<'a> {
    /// Creates a new `BitBufferReader` from a byte slice.
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            index: Cell::new(0),
        }
    }

    /// Seeks to the specified bit index.
    #[inline]
    pub fn seek(&self, index: usize) {
        self.index.set(index);
    }

    /// Returns the current bit index.
    #[inline]
    pub fn current_index(&self) -> usize {
        self.index.get()
    }

    /// Peeks at the current bit without advancing the index.
    #[inline]
    pub fn peek(&self) -> i32 {
        let octet_index = self.index.get() / 8;
        let bit_index = self.index.get() % 8;
        ((self.data[octet_index] >> (7 - bit_index)) & 1) as i32
    }

    /// Reads a single bit and advances the index.
    #[inline]
    pub fn read(&self) -> i32 {
        let octet_index = self.index.get() / 8;
        let bit_index = self.index.get() % 8;
        self.index.set(self.index.get() + 1);
        ((self.data[octet_index] >> (7 - bit_index)) & 1) as i32
    }

    /// Reads multiple bits and returns them as an i32.
    #[inline]
    pub fn read_bits(&self, len: usize) -> i32 {
        assert!(len > 0 && len < 32, "len must be between 1 and 31");

        let mut result = 0i32;
        for _ in 0..len {
            result <<= 1;
            result |= self.read();
        }
        result
    }

    /// Reads multiple bits and returns them as an i64.
    #[inline]
    pub fn read_bits_long(&self, len: usize) -> i64 {
        assert!(len > 0 && len <= 64, "len must be between 1 and 64");

        let mut result = 0i64;
        for _ in 0..len {
            result <<= 1;
            result |= self.read() as i64;
        }
        result
    }

    /// Skips to the next octet boundary.
    #[inline]
    pub fn octet_align(&self) {
        let remainder = self.index.get() % 8;
        if remainder != 0 {
            self.index.set(self.index.get() + (8 - remainder));
        }
    }

    /// Returns the total capacity in bits.
    #[inline]
    pub fn capacity_bits(&self) -> usize {
        self.data.len() * 8
    }

    /// Returns the remaining bits that can be read.
    #[inline]
    pub fn remaining_bits(&self) -> usize {
        self.capacity_bits().saturating_sub(self.index.get())
    }

    /// Returns true if there are more bits to read.
    #[inline]
    pub fn has_next(&self) -> bool {
        self.index.get() < self.capacity_bits()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_and_read_single_bits() {
        let mut data = [0u8; 2];
        {
            let mut buffer = BitBuffer::new(&mut data);

            buffer.write(true);
            buffer.write(false);
            buffer.write(true);
            buffer.write(false);
            buffer.write(true);
            buffer.write(false);
            buffer.write(true);
            buffer.write(false);
        }

        assert_eq!(data[0], 0b10101010);

        // Read back
        let buffer = BitBuffer::new(&mut data);
        buffer.seek(0);
        assert_eq!(buffer.read(), 1);
        assert_eq!(buffer.read(), 0);
        assert_eq!(buffer.read(), 1);
        assert_eq!(buffer.read(), 0);
    }

    #[test]
    fn test_write_and_read_bits() {
        let mut data = [0u8; 4];
        {
            let mut buffer = BitBuffer::new(&mut data);

            buffer.write_bits(0b1010, 4);
            buffer.write_bits(0b1100, 4);
            buffer.write_bits(0b11110000, 8);
        }

        assert_eq!(data[0], 0b10101100);
        assert_eq!(data[1], 0b11110000);

        // Read back
        let buffer = BitBuffer::new(&mut data);
        buffer.seek(0);
        assert_eq!(buffer.read_bits(4), 0b1010);
        assert_eq!(buffer.read_bits(4), 0b1100);
        assert_eq!(buffer.read_bits(8), 0b11110000);
    }

    #[test]
    fn test_write_bits_long() {
        let mut data = [0u8; 8];
        let mut buffer = BitBuffer::new(&mut data);

        let value: i64 = 0x123456789ABCDEF0u64 as i64;
        buffer.write_bits_long(value, 64);

        buffer.seek(0);
        assert_eq!(buffer.read_bits_long(64) as u64, 0x123456789ABCDEF0);
    }

    #[test]
    fn test_octet_align() {
        let mut data = [0u8; 4];
        let mut buffer = BitBuffer::new(&mut data);

        buffer.write_bits(0b101, 3);
        assert_eq!(buffer.current_index(), 3);

        buffer.octet_align();
        assert_eq!(buffer.current_index(), 8);

        // Already aligned, should not change
        buffer.octet_align();
        assert_eq!(buffer.current_index(), 8);
    }

    #[test]
    fn test_written_octets() {
        let mut data = [0u8; 4];
        let mut buffer = BitBuffer::new(&mut data);

        assert_eq!(buffer.written_octets(), 0);

        buffer.write_bits(0b1, 1);
        assert_eq!(buffer.written_octets(), 1);

        buffer.write_bits(0b1111111, 7);
        assert_eq!(buffer.written_octets(), 1);

        buffer.write_bits(0b1, 1);
        assert_eq!(buffer.written_octets(), 2);
    }

    #[test]
    fn test_peek() {
        let mut data = [0b10101010u8];
        let buffer = BitBuffer::new(&mut data);

        assert_eq!(buffer.peek(), 1);
        assert_eq!(buffer.current_index(), 0);

        buffer.seek(1);
        assert_eq!(buffer.peek(), 0);
    }

    #[test]
    fn test_bits_to_octet_boundary() {
        let mut data = [0u8; 4];
        let buffer = BitBuffer::new(&mut data);

        assert_eq!(buffer.bits_to_octet_boundary(), 0);

        buffer.seek(1);
        assert_eq!(buffer.bits_to_octet_boundary(), 7);

        buffer.seek(7);
        assert_eq!(buffer.bits_to_octet_boundary(), 1);

        buffer.seek(8);
        assert_eq!(buffer.bits_to_octet_boundary(), 0);
    }

    #[test]
    fn test_bit_buffer_reader() {
        let data = [0b10101100u8, 0b11110000u8];
        let reader = BitBufferReader::new(&data);

        assert_eq!(reader.read_bits(4), 0b1010);
        assert_eq!(reader.read_bits(4), 0b1100);
        assert_eq!(reader.read_bits(8), 0b11110000);
    }

    #[test]
    fn test_bit_buffer_reader_octet_align() {
        let data = [0u8; 4];
        let reader = BitBufferReader::new(&data);

        reader.seek(3);
        reader.octet_align();
        assert_eq!(reader.current_index(), 8);

        reader.octet_align();
        assert_eq!(reader.current_index(), 8);
    }

    #[test]
    fn test_remaining_bits() {
        let mut data = [0u8; 2];
        let buffer = BitBuffer::new(&mut data);

        assert_eq!(buffer.remaining_bits(), 16);
        buffer.seek(5);
        assert_eq!(buffer.remaining_bits(), 11);
    }

    #[test]
    fn test_cross_byte_boundary() {
        let mut data = [0u8; 4];
        let mut buffer = BitBuffer::new(&mut data);

        // Write 12 bits that cross byte boundary
        buffer.write_bits(0b111100001111, 12);

        buffer.seek(0);
        assert_eq!(buffer.read_bits(12), 0b111100001111);
    }

    #[test]
    fn test_write_zero_bits() {
        let mut data = [0xFFu8; 4];
        {
            let mut buffer = BitBuffer::new(&mut data);
            buffer.write_bits(0, 0); // Should do nothing
            assert_eq!(buffer.current_index(), 0);
        }
        assert_eq!(data[0], 0xFF);
    }
}
