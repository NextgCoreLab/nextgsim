//! `OctetString` type for variable-length byte sequences.
//!
//! This module provides the `OctetString` type, which is a wrapper around `Vec<u8>`
//! with convenient methods for protocol encoding/decoding operations.

use std::fmt;

/// A variable-length sequence of octets (bytes).
///
/// `OctetString` provides methods for building and parsing byte sequences
/// commonly used in 5G NAS and NGAP protocols. It supports:
/// - Appending single bytes and multi-byte values (big-endian)
/// - Reading values at specific indices
/// - Hex string conversion
/// - XOR operations
/// - Subslice copying
#[derive(Clone, PartialEq, Eq, Hash, Default)]
pub struct OctetString {
    data: Vec<u8>,
}

impl OctetString {
    /// Creates a new empty `OctetString`.
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Creates an `OctetString` from a `Vec<u8>`.
    pub fn from_vec(data: Vec<u8>) -> Self {
        Self { data }
    }

    /// Creates an `OctetString` from a byte slice.
    pub fn from_slice(data: &[u8]) -> Self {
        Self {
            data: data.to_vec(),
        }
    }

    /// Creates an empty `OctetString`.
    pub fn empty() -> Self {
        Self::new()
    }

    /// Creates an `OctetString` filled with zeros of the given length.
    pub fn from_spare(length: usize) -> Self {
        Self {
            data: vec![0u8; length],
        }
    }

    /// Creates an `OctetString` from a single byte.
    pub fn from_octet(value: u8) -> Self {
        Self { data: vec![value] }
    }

    /// Creates an `OctetString` from a 16-bit value (big-endian).
    pub fn from_u16(value: u16) -> Self {
        Self {
            data: vec![(value >> 8) as u8, (value & 0xFF) as u8],
        }
    }

    /// Creates an `OctetString` from a 32-bit value (big-endian).
    pub fn from_u32(value: u32) -> Self {
        Self {
            data: vec![
                (value >> 24) as u8,
                (value >> 16) as u8,
                (value >> 8) as u8,
                (value & 0xFF) as u8,
            ],
        }
    }

    /// Creates an `OctetString` from a 64-bit value (big-endian).
    pub fn from_u64(value: u64) -> Self {
        Self {
            data: vec![
                (value >> 56) as u8,
                (value >> 48) as u8,
                (value >> 40) as u8,
                (value >> 32) as u8,
                (value >> 24) as u8,
                (value >> 16) as u8,
                (value >> 8) as u8,
                (value & 0xFF) as u8,
            ],
        }
    }

    /// Creates an `OctetString` from a hex string.
    ///
    /// # Arguments
    /// * `hex` - A hex string (e.g., "DEADBEEF" or "dead beef")
    ///
    /// # Returns
    /// `Some(OctetString)` if the hex string is valid, `None` otherwise.
    pub fn from_hex(hex: &str) -> Option<Self> {
        let hex: String = hex.chars().filter(|c| !c.is_whitespace()).collect();
        if hex.len() % 2 != 0 {
            return None;
        }

        let mut data = Vec::with_capacity(hex.len() / 2);
        for i in (0..hex.len()).step_by(2) {
            let byte = u8::from_str_radix(&hex[i..i + 2], 16).ok()?;
            data.push(byte);
        }
        Some(Self { data })
    }

    /// Creates an `OctetString` from an ASCII string.
    pub fn from_ascii(ascii: &str) -> Self {
        Self {
            data: ascii.as_bytes().to_vec(),
        }
    }

    // --- Append methods ---

    /// Appends another `OctetString` to this one.
    pub fn append(&mut self, other: &OctetString) {
        self.data.extend_from_slice(&other.data);
    }

    /// Appends a UTF-8 string as bytes.
    pub fn append_utf8(&mut self, s: &str) {
        self.data.extend_from_slice(s.as_bytes());
    }

    /// Appends a single byte.
    pub fn append_octet(&mut self, value: u8) {
        self.data.push(value);
    }

    /// Appends a byte constructed from two 4-bit nibbles.
    ///
    /// # Arguments
    /// * `high_nibble` - The high 4 bits (0-15)
    /// * `low_nibble` - The low 4 bits (0-15)
    pub fn append_octet_nibbles(&mut self, high_nibble: u8, low_nibble: u8) {
        self.data
            .push(((high_nibble & 0x0F) << 4) | (low_nibble & 0x0F));
    }

    /// Appends a 16-bit value in big-endian order.
    pub fn append_u16(&mut self, value: u16) {
        self.data.push((value >> 8) as u8);
        self.data.push((value & 0xFF) as u8);
    }

    /// Appends a 24-bit value in big-endian order.
    pub fn append_u24(&mut self, value: u32) {
        self.data.push((value >> 16) as u8);
        self.data.push((value >> 8) as u8);
        self.data.push((value & 0xFF) as u8);
    }

    /// Appends a 32-bit value in big-endian order.
    pub fn append_u32(&mut self, value: u32) {
        self.data.push((value >> 24) as u8);
        self.data.push((value >> 16) as u8);
        self.data.push((value >> 8) as u8);
        self.data.push((value & 0xFF) as u8);
    }

    /// Appends a 64-bit value in big-endian order.
    pub fn append_u64(&mut self, value: u64) {
        self.data.push((value >> 56) as u8);
        self.data.push((value >> 48) as u8);
        self.data.push((value >> 40) as u8);
        self.data.push((value >> 32) as u8);
        self.data.push((value >> 24) as u8);
        self.data.push((value >> 16) as u8);
        self.data.push((value >> 8) as u8);
        self.data.push((value & 0xFF) as u8);
    }

    /// Appends `length` zero bytes.
    pub fn append_padding(&mut self, length: usize) {
        self.data.extend(std::iter::repeat_n(0u8, length));
    }

    // --- Accessor methods ---

    /// Returns a reference to the underlying byte slice.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Returns a mutable reference to the underlying byte slice.
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Returns the length in bytes.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the string is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Gets a single byte at the given index.
    ///
    /// # Panics
    /// Panics if `index >= len()`.
    pub fn get(&self, index: usize) -> u8 {
        self.data[index]
    }

    /// Gets a 16-bit value at the given index (big-endian).
    ///
    /// # Panics
    /// Panics if `index + 2 > len()`.
    pub fn get_u16(&self, index: usize) -> u16 {
        ((self.data[index] as u16) << 8) | (self.data[index + 1] as u16)
    }

    /// Gets a 24-bit value at the given index (big-endian).
    ///
    /// # Panics
    /// Panics if `index + 3 > len()`.
    pub fn get_u24(&self, index: usize) -> u32 {
        ((self.data[index] as u32) << 16)
            | ((self.data[index + 1] as u32) << 8)
            | (self.data[index + 2] as u32)
    }

    /// Gets a 32-bit value at the given index (big-endian).
    ///
    /// # Panics
    /// Panics if `index + 4 > len()`.
    pub fn get_u32(&self, index: usize) -> u32 {
        ((self.data[index] as u32) << 24)
            | ((self.data[index + 1] as u32) << 16)
            | ((self.data[index + 2] as u32) << 8)
            | (self.data[index + 3] as u32)
    }

    /// Gets a 64-bit value at the given index (big-endian).
    ///
    /// # Panics
    /// Panics if `index + 8 > len()`.
    pub fn get_u64(&self, index: usize) -> u64 {
        ((self.data[index] as u64) << 56)
            | ((self.data[index + 1] as u64) << 48)
            | ((self.data[index + 2] as u64) << 40)
            | ((self.data[index + 3] as u64) << 32)
            | ((self.data[index + 4] as u64) << 24)
            | ((self.data[index + 5] as u64) << 16)
            | ((self.data[index + 6] as u64) << 8)
            | (self.data[index + 7] as u64)
    }

    // --- Conversion methods ---

    /// Converts to a hex string (uppercase).
    pub fn to_hex_string(&self) -> String {
        self.data
            .iter()
            .map(|b| format!("{b:02X}"))
            .collect::<String>()
    }

    /// Creates a copy of this `OctetString`.
    pub fn copy(&self) -> Self {
        self.clone()
    }

    /// Creates a copy of the substring from `index` to the end.
    pub fn sub_copy(&self, index: usize) -> Self {
        self.sub_copy_len(index, self.len() - index)
    }

    /// Creates a copy of the substring from `index` with the given `length`.
    pub fn sub_copy_len(&self, index: usize, length: usize) -> Self {
        Self {
            data: self.data[index..index + length].to_vec(),
        }
    }

    /// Consumes self and returns the underlying `Vec<u8>`.
    pub fn into_vec(self) -> Vec<u8> {
        self.data
    }

    // --- Static operations ---

    /// Concatenates two `OctetString`s.
    pub fn concat(a: &OctetString, b: &OctetString) -> Self {
        let mut result = a.clone();
        result.append(b);
        result
    }

    /// XORs two `OctetString`s.
    ///
    /// The result has the length of the shorter string.
    pub fn xor(a: &OctetString, b: &OctetString) -> Self {
        let min_len = a.len().min(b.len());
        let mut result = Vec::with_capacity(min_len);
        for i in 0..min_len {
            result.push(a.data[i] ^ b.data[i]);
        }
        Self { data: result }
    }
}

impl fmt::Debug for OctetString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OctetString({})", self.to_hex_string())
    }
}

impl fmt::Display for OctetString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_hex_string())
    }
}

impl From<Vec<u8>> for OctetString {
    fn from(data: Vec<u8>) -> Self {
        Self { data }
    }
}

impl From<&[u8]> for OctetString {
    fn from(data: &[u8]) -> Self {
        Self {
            data: data.to_vec(),
        }
    }
}

impl AsRef<[u8]> for OctetString {
    fn as_ref(&self) -> &[u8] {
        &self.data
    }
}

impl std::ops::Index<usize> for OctetString {
    type Output = u8;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_and_empty() {
        let os = OctetString::new();
        assert!(os.is_empty());
        assert_eq!(os.len(), 0);

        let os2 = OctetString::empty();
        assert!(os2.is_empty());
    }

    #[test]
    fn test_from_spare() {
        let os = OctetString::from_spare(5);
        assert_eq!(os.len(), 5);
        assert_eq!(os.data(), &[0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_from_octet() {
        let os = OctetString::from_octet(0xAB);
        assert_eq!(os.len(), 1);
        assert_eq!(os.get(0), 0xAB);
    }

    #[test]
    fn test_from_u16() {
        let os = OctetString::from_u16(0x1234);
        assert_eq!(os.len(), 2);
        assert_eq!(os.data(), &[0x12, 0x34]);
    }

    #[test]
    fn test_from_u32() {
        let os = OctetString::from_u32(0x12345678);
        assert_eq!(os.len(), 4);
        assert_eq!(os.data(), &[0x12, 0x34, 0x56, 0x78]);
    }

    #[test]
    fn test_from_u64() {
        let os = OctetString::from_u64(0x123456789ABCDEF0);
        assert_eq!(os.len(), 8);
        assert_eq!(
            os.data(),
            &[0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0]
        );
    }

    #[test]
    fn test_from_hex() {
        let os = OctetString::from_hex("DEADBEEF").unwrap();
        assert_eq!(os.data(), &[0xDE, 0xAD, 0xBE, 0xEF]);

        let os2 = OctetString::from_hex("de ad be ef").unwrap();
        assert_eq!(os2.data(), &[0xDE, 0xAD, 0xBE, 0xEF]);

        assert!(OctetString::from_hex("DEA").is_none()); // Odd length
        assert!(OctetString::from_hex("GHIJ").is_none()); // Invalid hex
    }

    #[test]
    fn test_from_ascii() {
        let os = OctetString::from_ascii("ABC");
        assert_eq!(os.data(), &[0x41, 0x42, 0x43]);
    }

    #[test]
    fn test_append_octet() {
        let mut os = OctetString::new();
        os.append_octet(0x12);
        os.append_octet(0x34);
        assert_eq!(os.data(), &[0x12, 0x34]);
    }

    #[test]
    fn test_append_octet_nibbles() {
        let mut os = OctetString::new();
        os.append_octet_nibbles(0x0A, 0x0B);
        assert_eq!(os.get(0), 0xAB);
    }

    #[test]
    fn test_append_u16() {
        let mut os = OctetString::new();
        os.append_u16(0x1234);
        assert_eq!(os.data(), &[0x12, 0x34]);
    }

    #[test]
    fn test_append_u24() {
        let mut os = OctetString::new();
        os.append_u24(0x123456);
        assert_eq!(os.data(), &[0x12, 0x34, 0x56]);
    }

    #[test]
    fn test_append_u32() {
        let mut os = OctetString::new();
        os.append_u32(0x12345678);
        assert_eq!(os.data(), &[0x12, 0x34, 0x56, 0x78]);
    }

    #[test]
    fn test_append_u64() {
        let mut os = OctetString::new();
        os.append_u64(0x123456789ABCDEF0);
        assert_eq!(
            os.data(),
            &[0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0]
        );
    }

    #[test]
    fn test_append_padding() {
        let mut os = OctetString::new();
        os.append_octet(0xFF);
        os.append_padding(3);
        assert_eq!(os.data(), &[0xFF, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_get_u16() {
        let os = OctetString::from_hex("12345678").unwrap();
        assert_eq!(os.get_u16(0), 0x1234);
        assert_eq!(os.get_u16(1), 0x3456);
    }

    #[test]
    fn test_get_u24() {
        let os = OctetString::from_hex("12345678").unwrap();
        assert_eq!(os.get_u24(0), 0x123456);
        assert_eq!(os.get_u24(1), 0x345678);
    }

    #[test]
    fn test_get_u32() {
        let os = OctetString::from_hex("12345678").unwrap();
        assert_eq!(os.get_u32(0), 0x12345678);
    }

    #[test]
    fn test_get_u64() {
        let os = OctetString::from_hex("123456789ABCDEF0").unwrap();
        assert_eq!(os.get_u64(0), 0x123456789ABCDEF0);
    }

    #[test]
    fn test_to_hex_string() {
        let os = OctetString::from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);
        assert_eq!(os.to_hex_string(), "DEADBEEF");
    }

    #[test]
    fn test_sub_copy() {
        let os = OctetString::from_hex("0102030405").unwrap();
        let sub = os.sub_copy(2);
        assert_eq!(sub.data(), &[0x03, 0x04, 0x05]);

        let sub2 = os.sub_copy_len(1, 2);
        assert_eq!(sub2.data(), &[0x02, 0x03]);
    }

    #[test]
    fn test_concat() {
        let a = OctetString::from_hex("0102").unwrap();
        let b = OctetString::from_hex("0304").unwrap();
        let c = OctetString::concat(&a, &b);
        assert_eq!(c.data(), &[0x01, 0x02, 0x03, 0x04]);
    }

    #[test]
    fn test_xor() {
        let a = OctetString::from_hex("FF00FF00").unwrap();
        let b = OctetString::from_hex("0F0F0F0F").unwrap();
        let c = OctetString::xor(&a, &b);
        assert_eq!(c.data(), &[0xF0, 0x0F, 0xF0, 0x0F]);
    }

    #[test]
    fn test_xor_different_lengths() {
        let a = OctetString::from_hex("FF00FF").unwrap();
        let b = OctetString::from_hex("0F0F0F0F0F").unwrap();
        let c = OctetString::xor(&a, &b);
        assert_eq!(c.len(), 3);
        assert_eq!(c.data(), &[0xF0, 0x0F, 0xF0]);
    }

    #[test]
    fn test_equality() {
        let a = OctetString::from_hex("DEADBEEF").unwrap();
        let b = OctetString::from_hex("DEADBEEF").unwrap();
        let c = OctetString::from_hex("DEADBEEE").unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_index() {
        let os = OctetString::from_hex("0102030405").unwrap();
        assert_eq!(os[0], 0x01);
        assert_eq!(os[2], 0x03);
        assert_eq!(os[4], 0x05);
    }
}
