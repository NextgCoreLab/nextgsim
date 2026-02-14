//! `OctetView` - A read-only view over byte slices for parsing protocol messages.
//!
//! This module provides `OctetView`, which allows sequential reading of bytes
//! from a byte slice without copying. It's commonly used for parsing 5G NAS
//! and NGAP protocol messages.

use crate::OctetString;
use std::cell::Cell;

/// A read-only view over a byte slice for sequential parsing.
///
/// `OctetView` maintains an internal read index that advances as data is read.
/// It provides methods for reading single bytes and multi-byte values in
/// big-endian order.
///
/// # Example
/// ```
/// use nextgsim_common::OctetView;
///
/// let data = [0x12, 0x34, 0x56, 0x78];
/// let view = OctetView::new(&data);
///
/// assert_eq!(view.read(), 0x12);
/// assert_eq!(view.read_u16(), 0x3456);
/// assert_eq!(view.read(), 0x78);
/// assert!(!view.has_next());
/// ```
#[derive(Debug)]
pub struct OctetView<'a> {
    data: &'a [u8],
    index: Cell<usize>,
}

impl<'a> OctetView<'a> {
    /// Creates a new `OctetView` from a byte slice.
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            index: Cell::new(0),
        }
    }

    /// Creates a new `OctetView` from an `OctetString`.
    pub fn from_octet_string(data: &'a OctetString) -> Self {
        Self::new(data.data())
    }

    // --- Peek methods (don't advance index) ---

    /// Peeks at the current byte without advancing the index.
    ///
    /// # Panics
    /// Panics if there are no more bytes to read.
    #[inline]
    pub fn peek(&self) -> u8 {
        self.data[self.index.get()]
    }

    /// Peeks at a byte at the given offset from the current position.
    ///
    /// # Panics
    /// Panics if `current_index + offset >= len`.
    #[inline]
    pub fn peek_at(&self, offset: usize) -> u8 {
        self.data[self.index.get() + offset]
    }

    /// Peeks at the current byte as an i32.
    #[inline]
    pub fn peek_i(&self) -> i32 {
        self.peek() as i32
    }

    /// Peeks at a byte at the given offset as an i32.
    #[inline]
    pub fn peek_i_at(&self, offset: usize) -> i32 {
        self.peek_at(offset) as i32
    }

    // --- Read methods (advance index) ---

    /// Reads a single byte and advances the index.
    ///
    /// # Panics
    /// Panics if there are no more bytes to read.
    #[inline]
    pub fn read(&self) -> u8 {
        let idx = self.index.get();
        self.index.set(idx + 1);
        self.data[idx]
    }

    /// Reads a single byte as an i32.
    #[inline]
    pub fn read_i(&self) -> i32 {
        self.read() as i32
    }

    /// Reads a 16-bit value in big-endian order.
    #[inline]
    pub fn read_u16(&self) -> u16 {
        let b0 = self.read() as u16;
        let b1 = self.read() as u16;
        (b0 << 8) | b1
    }

    /// Reads a 16-bit value as an i32.
    #[inline]
    pub fn read_u16_i(&self) -> i32 {
        self.read_u16() as i32
    }

    /// Reads a 24-bit value in big-endian order.
    #[inline]
    pub fn read_u24(&self) -> u32 {
        let b0 = self.read() as u32;
        let b1 = self.read() as u32;
        let b2 = self.read() as u32;
        (b0 << 16) | (b1 << 8) | b2
    }

    /// Reads a 24-bit value as an i32.
    #[inline]
    pub fn read_u24_i(&self) -> i32 {
        self.read_u24() as i32
    }

    /// Reads a 32-bit value in big-endian order.
    #[inline]
    pub fn read_u32(&self) -> u32 {
        let b0 = self.read() as u32;
        let b1 = self.read() as u32;
        let b2 = self.read() as u32;
        let b3 = self.read() as u32;
        (b0 << 24) | (b1 << 16) | (b2 << 8) | b3
    }

    /// Reads a 32-bit value as an i32.
    #[inline]
    pub fn read_u32_i(&self) -> i32 {
        self.read_u32() as i32
    }

    /// Reads a 64-bit value in big-endian order.
    #[inline]
    pub fn read_u64(&self) -> u64 {
        let b0 = self.read() as u64;
        let b1 = self.read() as u64;
        let b2 = self.read() as u64;
        let b3 = self.read() as u64;
        let b4 = self.read() as u64;
        let b5 = self.read() as u64;
        let b6 = self.read() as u64;
        let b7 = self.read() as u64;
        (b0 << 56) | (b1 << 48) | (b2 << 40) | (b3 << 32) | (b4 << 24) | (b5 << 16) | (b6 << 8) | b7
    }

    /// Reads a 64-bit value as an i64.
    #[inline]
    pub fn read_i64(&self) -> i64 {
        self.read_u64() as i64
    }

    /// Reads an `OctetString` of the specified length.
    ///
    /// # Panics
    /// Panics if there are not enough bytes remaining.
    pub fn read_octet_string(&self, length: usize) -> OctetString {
        if length == 0 {
            return OctetString::new();
        }
        let idx = self.index.get();
        let result = OctetString::from_slice(&self.data[idx..idx + length]);
        self.index.set(idx + length);
        result
    }

    /// Reads the remaining bytes as an `OctetString`.
    pub fn read_remaining_octet_string(&self) -> OctetString {
        let remaining = self.remaining();
        self.read_octet_string(remaining)
    }

    /// Reads a UTF-8 string of the specified length.
    ///
    /// # Panics
    /// Panics if there are not enough bytes remaining or if the bytes are not valid UTF-8.
    pub fn read_utf8_string(&self, length: usize) -> String {
        let idx = self.index.get();
        let result = String::from_utf8(self.data[idx..idx + length].to_vec())
            .expect("Invalid UTF-8 sequence");
        self.index.set(idx + length);
        result
    }

    /// Reads a UTF-8 string of the specified length, returning an error if invalid.
    pub fn try_read_utf8_string(&self, length: usize) -> Result<String, std::string::FromUtf8Error> {
        let idx = self.index.get();
        let result = String::from_utf8(self.data[idx..idx + length].to_vec())?;
        self.index.set(idx + length);
        Ok(result)
    }

    // --- Position and state methods ---

    /// Returns the current read index.
    #[inline]
    pub fn current_index(&self) -> usize {
        self.index.get()
    }

    /// Returns the total length of the underlying data.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the underlying data is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the number of bytes remaining to be read.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.index.get())
    }

    /// Returns true if there are more bytes to read.
    #[inline]
    pub fn has_next(&self) -> bool {
        self.index.get() < self.data.len()
    }

    /// Skips the specified number of bytes.
    #[inline]
    pub fn skip(&self, count: usize) {
        self.index.set(self.index.get() + count);
    }

    /// Seeks to the specified index.
    #[inline]
    pub fn seek(&self, index: usize) {
        self.index.set(index);
    }

    /// Returns a reference to the underlying data slice.
    #[inline]
    pub fn data(&self) -> &[u8] {
        self.data
    }

    /// Returns a slice of the remaining unread data.
    #[inline]
    pub fn remaining_data(&self) -> &[u8] {
        &self.data[self.index.get()..]
    }
}

impl<'a> From<&'a [u8]> for OctetView<'a> {
    fn from(data: &'a [u8]) -> Self {
        Self::new(data)
    }
}

impl<'a> From<&'a OctetString> for OctetView<'a> {
    fn from(data: &'a OctetString) -> Self {
        Self::from_octet_string(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let data = [0x01, 0x02, 0x03];
        let view = OctetView::new(&data);
        assert_eq!(view.len(), 3);
        assert_eq!(view.current_index(), 0);
        assert!(view.has_next());
    }

    #[test]
    fn test_peek() {
        let data = [0x12, 0x34, 0x56];
        let view = OctetView::new(&data);

        assert_eq!(view.peek(), 0x12);
        assert_eq!(view.peek_at(1), 0x34);
        assert_eq!(view.peek_at(2), 0x56);
        assert_eq!(view.current_index(), 0); // Index unchanged
    }

    #[test]
    fn test_read() {
        let data = [0x12, 0x34, 0x56];
        let view = OctetView::new(&data);

        assert_eq!(view.read(), 0x12);
        assert_eq!(view.current_index(), 1);
        assert_eq!(view.read(), 0x34);
        assert_eq!(view.current_index(), 2);
        assert_eq!(view.read(), 0x56);
        assert_eq!(view.current_index(), 3);
        assert!(!view.has_next());
    }

    #[test]
    fn test_read_u16() {
        let data = [0x12, 0x34, 0x56, 0x78];
        let view = OctetView::new(&data);

        assert_eq!(view.read_u16(), 0x1234);
        assert_eq!(view.read_u16(), 0x5678);
    }

    #[test]
    fn test_read_u24() {
        let data = [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC];
        let view = OctetView::new(&data);

        assert_eq!(view.read_u24(), 0x123456);
        assert_eq!(view.read_u24(), 0x789ABC);
    }

    #[test]
    fn test_read_u32() {
        let data = [0x12, 0x34, 0x56, 0x78];
        let view = OctetView::new(&data);

        assert_eq!(view.read_u32(), 0x12345678);
    }

    #[test]
    fn test_read_u64() {
        let data = [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0];
        let view = OctetView::new(&data);

        assert_eq!(view.read_u64(), 0x123456789ABCDEF0);
    }

    #[test]
    fn test_read_octet_string() {
        let data = [0x01, 0x02, 0x03, 0x04, 0x05];
        let view = OctetView::new(&data);

        view.read(); // Skip first byte
        let os = view.read_octet_string(3);
        assert_eq!(os.data(), &[0x02, 0x03, 0x04]);
        assert_eq!(view.current_index(), 4);
    }

    #[test]
    fn test_read_remaining_octet_string() {
        let data = [0x01, 0x02, 0x03, 0x04, 0x05];
        let view = OctetView::new(&data);

        view.read(); // Skip first byte
        view.read(); // Skip second byte
        let os = view.read_remaining_octet_string();
        assert_eq!(os.data(), &[0x03, 0x04, 0x05]);
        assert!(!view.has_next());
    }

    #[test]
    fn test_read_utf8_string() {
        let data = b"Hello, World!";
        let view = OctetView::new(data);

        let s = view.read_utf8_string(5);
        assert_eq!(s, "Hello");
        assert_eq!(view.current_index(), 5);
    }

    #[test]
    fn test_skip_and_seek() {
        let data = [0x01, 0x02, 0x03, 0x04, 0x05];
        let view = OctetView::new(&data);

        view.skip(2);
        assert_eq!(view.current_index(), 2);
        assert_eq!(view.read(), 0x03);

        view.seek(0);
        assert_eq!(view.current_index(), 0);
        assert_eq!(view.read(), 0x01);
    }

    #[test]
    fn test_remaining() {
        let data = [0x01, 0x02, 0x03, 0x04, 0x05];
        let view = OctetView::new(&data);

        assert_eq!(view.remaining(), 5);
        view.read();
        assert_eq!(view.remaining(), 4);
        view.skip(2);
        assert_eq!(view.remaining(), 2);
    }

    #[test]
    fn test_from_octet_string() {
        let os = OctetString::from_hex("DEADBEEF").unwrap();
        let view = OctetView::from_octet_string(&os);

        assert_eq!(view.read(), 0xDE);
        assert_eq!(view.read(), 0xAD);
        assert_eq!(view.read(), 0xBE);
        assert_eq!(view.read(), 0xEF);
    }

    #[test]
    fn test_empty_octet_string_read() {
        let data = [0x01, 0x02];
        let view = OctetView::new(&data);

        let os = view.read_octet_string(0);
        assert!(os.is_empty());
        assert_eq!(view.current_index(), 0);
    }

    #[test]
    fn test_remaining_data() {
        let data = [0x01, 0x02, 0x03, 0x04, 0x05];
        let view = OctetView::new(&data);

        view.skip(2);
        assert_eq!(view.remaining_data(), &[0x03, 0x04, 0x05]);
    }
}
