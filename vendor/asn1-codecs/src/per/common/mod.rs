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

/// Whether the characters of a known-multiplier character string are placed
/// "octet-aligned in the ALIGNED variant" (X.691 §30.5.6-30.5.8).
///
/// The decision is the TYPE's, not the value's: with `aub` the upper bound and
/// `b` the bits per character, the characters are aligned when `aub × b > 16`,
/// and always when there is no upper bound below 64K (the fragmenting /
/// unconstrained case, §30.5.8). A fixed size of at most 16 bits is the only
/// unaligned case (§30.5.6). The fork used to compare the value's own length
/// instead, so a 1- or 2-character `SIZE(1..150)` string went unaligned.
/// Callers only ask for non-empty content.
pub(crate) fn charstring_content_aligned(ub: Option<i128>, bits_per_char: usize) -> bool {
    match ub {
        Some(u) if u < 65_536 => u * bits_per_char as i128 > 16,
        _ => true,
    }
}

#[cfg(test)]
mod fragment_tests;

#[cfg(test)]
mod bitstring_align_tests;

#[cfg(test)]
mod charstring_align_tests;

#[cfg(test)]
mod integer_ext_tests;
