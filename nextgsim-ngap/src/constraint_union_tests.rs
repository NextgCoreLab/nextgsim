//! A root constraint that mixes a range and single values takes its PER upper
//! bound from the largest member, not the smallest (asn1-compiler
//! `ConstraintValues::max`).
//!
//! `ExpectedActivityPeriod ::= INTEGER (1..30|40|50|60|80|100|120|150|180|181, ...)`
//! (TS 38.413 §9.3.1.94; `ExpectedIdlePeriod` has the same constraint). Its
//! PER-visible root is the union, so the effective range is 1..181 (range 181).
//! The compiler took `min(30, 181)` and generated `ub = 30`, so every root value
//! above 30 (40, 50, ... 181) was encoded as an EXTENSION value (`1` bit plus an
//! unconstrained whole number), and 1..30 were sent in 5 bits instead of 8. Every
//! expected byte string below is derived by hand from X.691 beside it.

use asn1_codecs::aper::AperCodec;
use asn1_codecs::PerCodecData;

use crate::codec::generated::{ExpectedActivityPeriod, ExpectedIdlePeriod};

fn enc<T: AperCodec>(v: &T) -> Vec<u8> {
    let mut d = PerCodecData::new_aper();
    v.aper_encode(&mut d).expect("encode");
    d.into_bytes()
}

fn dec<T: AperCodec<Output = T>>(bytes: &[u8]) -> T {
    let mut d = PerCodecData::from_slice_aper(bytes);
    T::aper_decode(&mut d).expect("decode")
}

/// Extensible root 1..181: ext bit 0, then the offset from 1 as a constrained
/// whole number of range 181 — a non-aligned 8-bit bit-field (X.691 §11.5.7.2,
/// range <= 255). So 9 bits, padded to two octets.
#[test]
fn expected_activity_period_encodes_root_values_up_to_181_in_eight_bits() {
    // 1  -> 0 | 00000000           -> 0000 0000 0... -> 00 00
    assert_eq!(enc(&ExpectedActivityPeriod(1)), [0x00, 0x00]);
    // 30 -> 0 | 00011101 (29)      -> 0000 1110 1... -> 0E 80   (was 74: 5 bits)
    assert_eq!(enc(&ExpectedActivityPeriod(30)), [0x0E, 0x80]);
    // 40 -> 0 | 00100111 (39)      -> 0001 0011 1... -> 13 80   (was 80 01 28: extension)
    assert_eq!(enc(&ExpectedActivityPeriod(40)), [0x13, 0x80]);
    // 181 -> 0 | 10110100 (180)    -> 0101 1010 0... -> 5A 00
    assert_eq!(enc(&ExpectedActivityPeriod(181)), [0x5A, 0x00]);
    for v in [1u8, 30, 40, 50, 60, 80, 100, 120, 150, 180, 181] {
        assert_eq!(
            dec::<ExpectedActivityPeriod>(&enc(&ExpectedActivityPeriod(v))).0,
            v
        );
    }
}

#[test]
fn expected_idle_period_has_the_same_bound() {
    assert_eq!(enc(&ExpectedIdlePeriod(181)), [0x5A, 0x00]);
    assert_eq!(enc(&ExpectedIdlePeriod(40)), [0x13, 0x80]);
}
