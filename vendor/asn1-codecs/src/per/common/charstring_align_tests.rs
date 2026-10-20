//! Character strings in PER, X.691 §30.5 (known-multiplier types) and §30.6
//! (the others).
//!
//! Two rules the fork got wrong:
//!
//! 1. A known-multiplier string (PrintableString, VisibleString, …) is placed
//!    octet-aligned in the ALIGNED variant according to its UPPER BOUND, not its
//!    actual length: with `aub` the upper bound and `b` the bits per character,
//!    the characters are aligned when `aub × b > 16` (§30.5.7). The fork aligned
//!    only when the value itself had more than 2 characters, so a 1- or
//!    2-character `AMFName ::= PrintableString (SIZE(1..150, ...))` was packed
//!    straight after its length bits.
//! 2. A UTF8String is not a known-multiplier type, so its size constraint is not
//!    PER-visible (X.691 §10.9.3.3, §30.6): it has no extension bit, an
//!    unconstrained length in octets, and octet-aligned content. The fork applied
//!    `SIZE(1..150, ...)` with an extension bit and a constrained 8-bit length.
//!
//! Every expected byte string below is derived by hand from X.691, with the
//! derivation beside it; none is captured from the encoder. asn1tools 0.166.0
//! (`per.py`, `KnownMultiplierStringType` / `UTF8String`) encodes the same.

use crate::per::{aper, PerCodecData};

fn enc(f: impl FnOnce(&mut PerCodecData)) -> Vec<u8> {
    let mut d = PerCodecData::new_aper();
    f(&mut d);
    d.into_bytes()
}

fn s(v: &str) -> String {
    v.to_string()
}

const LB: Option<i128> = Some(1);
const UB: Option<i128> = Some(150);

/// `PrintableString (SIZE(1..150, ...))`, APER (b = 8, aub × b = 1200 > 16):
///   ext bit 0, length − 1 in 8 bits (range 150, a bit-field), pad, characters.
#[test]
fn a_short_constrained_printable_string_is_octet_aligned() {
    // "A":   0 | 00000000 | pad(7) -> 00 00 | 41
    let a =
        enc(|d| aper::encode::encode_printable_string(d, LB, UB, true, &s("A"), false).unwrap());
    assert_eq!(a, [0x00, 0x00, 0x41]);
    // "AB":  0 | 00000001 | pad    -> 00 80 | 41 42
    let ab =
        enc(|d| aper::encode::encode_printable_string(d, LB, UB, true, &s("AB"), false).unwrap());
    assert_eq!(ab, [0x00, 0x80, 0x41, 0x42]);
    // "ABC" was already aligned: 0 | 00000010 | pad -> 01 00 | 41 42 43
    let abc =
        enc(|d| aper::encode::encode_printable_string(d, LB, UB, true, &s("ABC"), false).unwrap());
    assert_eq!(abc, [0x01, 0x00, 0x41, 0x42, 0x43]);

    for (bytes, want) in [(a, "A"), (ab, "AB"), (abc, "ABC")] {
        let mut d = PerCodecData::from_slice_aper(&bytes);
        let got = aper::decode::decode_printable_string(&mut d, LB, UB, true).unwrap();
        assert_eq!(got, want);
    }
}

#[test]
fn a_short_constrained_visible_string_is_octet_aligned() {
    let a = enc(|d| aper::encode::encode_visible_string(d, LB, UB, true, &s("A"), false).unwrap());
    assert_eq!(a, [0x00, 0x00, 0x41]);
    let mut d = PerCodecData::from_slice_aper(&a);
    assert_eq!(
        aper::decode::decode_visible_string(&mut d, LB, UB, true).unwrap(),
        "A"
    );
}

/// `UTF8String (SIZE(1..150, ...))`: the constraint is not PER-visible, so
/// "A" is an unconstrained length octet then the octet: 01 41 (no ext bit).
#[test]
fn a_utf8_string_ignores_its_size_constraint() {
    let a = enc(|d| aper::encode::encode_utf8_string(d, LB, UB, true, &s("A"), false).unwrap());
    assert_eq!(a, [0x01, 0x41]);
    // "é" is two UTF-8 octets: length 2, C3 A9.
    let e = enc(|d| aper::encode::encode_utf8_string(d, LB, UB, true, &s("é"), false).unwrap());
    assert_eq!(e, [0x02, 0xC3, 0xA9]);
    for (bytes, want) in [(a, "A"), (e, "é")] {
        let mut d = PerCodecData::from_slice_aper(&bytes);
        assert_eq!(
            aper::decode::decode_utf8_string(&mut d, LB, UB, true).unwrap(),
            want
        );
    }
}

/// A fixed-size string with `aub × b ≤ 16` stays unaligned (§30.5.6): after a
/// `1` bit, `PrintableString (SIZE(2))` "AB" is 1 | 01000001 01000010 ->
/// A0 A1 00.
#[test]
fn a_two_character_fixed_size_string_stays_unaligned() {
    let mut d = PerCodecData::new_aper();
    d.encode_bool(true);
    aper::encode::encode_printable_string(&mut d, Some(2), Some(2), false, &s("AB"), false)
        .unwrap();
    assert_eq!(d.into_bytes(), [0xA0, 0xA1, 0x00]);
}
