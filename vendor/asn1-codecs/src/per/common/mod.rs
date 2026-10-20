pub(crate) mod encode;

pub(crate) mod decode;

// FIXME: Remove the pub(crate) when `decode` also pulled uner `common`.
pub(crate) fn bytes_needed_for_range(range: i128) -> u8 {
    let bits_needed: u8 = 128 - (range - 1).leading_zeros() as u8;
    let mut bytes_needed = bits_needed / 8;
    if bits_needed % 8 != 0 {
        bytes_needed += 1
    }
    bytes_needed
}

// See X.690 section 8.5.9 for details on special REAL values
const NEGATIVE_ZERO: u8 = 0b0100_0011;
const INFINITY: u8 = 0b0100_0000;
const NEGATIVE_INFINITY: u8 = 0b0100_0001;
const NOT_A_NUMBER: u8 = 0b0100_0010;
const BASE_10_NR1: u8 = 0b0000_0001;
const BASE_10_NR2: u8 = 0b0000_0010;
const BASE_10_NR3: u8 = 0b0000_0011;

/// Whether the content of a BIT STRING or OCTET STRING goes in a bit-field
/// "octet-aligned in the ALIGNED variant" (X.691 §16.9-16.11, §17.6-17.8).
///
/// Only a FIXED-size string (`lb == ub`, below 64K units) of at most
/// `fixed_unaligned_max` units -- 16 bits, or 2 octets -- is placed unaligned
/// with no length determinant (§16.9, §17.6). Every other string -- variable
/// size, fixed at 64K or more, or an out-of-root extension value -- is placed
/// octet-aligned after its length determinant whatever its length (§16.11,
/// §17.8); the fork used to apply the fixed-size threshold to those too.
/// Callers only ask for non-empty content: an empty string has no bit-field.
pub(crate) fn string_content_aligned(
    lb: Option<i128>,
    ub: Option<i128>,
    count: usize,
    fixed_unaligned_max: usize,
) -> bool {
    let fixed = matches!((lb, ub), (Some(l), Some(u)) if l == u && u < 65_536);
    !fixed || count > fixed_unaligned_max
}

#[cfg(test)]
mod fragment_tests;

#[cfg(test)]
mod bitstring_align_tests;
