//! Extensible INTEGER values outside the extension root, X.691 §12.1, in both
//! PER variants (#231).
//!
//! §12.1: for an extensible constrained INTEGER, a value inside the root is
//! preceded by a `0` bit and encoded per §12.2 with the root bounds; a value
//! outside the root is preceded by a `1` bit and encoded as an UNCONSTRAINED
//! whole number (§12.2.6 -> §11.8): a length determinant giving the number of
//! octets, then the minimal two's-complement octets. The fork refused the
//! second case outright ("Encode of extended integer not yet implemented").
//!
//! Every expected byte string below is derived by hand from X.691, with the
//! derivation beside it; none is captured from the encoder.

use crate::per::{aper, uper, PerCodecData};

/// `MaximumDataBurstVolume ::= INTEGER (0..4095, ..., 4096..2000000)`
/// (TS 38.413): root 0..4095, so a PER-visible lb/ub of 0/4095.
const MDBV: (Option<i128>, Option<i128>) = (Some(0), Some(4095));

fn enc(aligned: bool, lb: Option<i128>, ub: Option<i128>, ext: bool, v: i128) -> Vec<u8> {
    let mut d = if aligned {
        PerCodecData::new_aper()
    } else {
        PerCodecData::new_uper()
    };
    let extended = ext && (lb.is_some_and(|l| v < l) || ub.is_some_and(|u| v > u));
    if aligned {
        aper::encode::encode_integer(&mut d, lb, ub, ext, v, extended).unwrap();
    } else {
        uper::encode::encode_integer(&mut d, lb, ub, ext, v, extended).unwrap();
    }
    d.into_bytes()
}

fn dec(aligned: bool, lb: Option<i128>, ub: Option<i128>, ext: bool, bytes: &[u8]) -> (i128, bool) {
    let mut d = if aligned {
        PerCodecData::from_slice_aper(bytes)
    } else {
        PerCodecData::from_slice_uper(bytes)
    };
    if aligned {
        aper::decode::decode_integer(&mut d, lb, ub, ext).unwrap()
    } else {
        uper::decode::decode_integer(&mut d, lb, ub, ext).unwrap()
    }
}

/// (value, expected APER, expected UPER, outside the root?)
#[allow(clippy::type_complexity)]
fn mdbv_cases() -> Vec<(i128, Vec<u8>, Vec<u8>, bool)> {
    vec![
        // ROOT values (unchanged by #231; pinned so the fix cannot move them).
        // APER: ext bit 0, then range 4096 (>256, <=64K) is the two-octet
        // aligned case (§11.5.7.3): 0 + 7 pad bits = 00, then 00 00.
        // UPER: ext bit 0, then 12 bits (range 4096): 0 000000000000 + pad.
        (0, vec![0x00, 0x00, 0x00], vec![0x00, 0x00], false),
        // 4095: APER 00 | 0F FF. UPER 0 111111111111 + 000 = 0111 1111 1111 1000.
        (4095, vec![0x00, 0x0F, 0xFF], vec![0x7F, 0xF8], false),
        // EXTENSION values: ext bit 1, then an unconstrained whole number.
        // 4096 = 0x1000: minimal two's complement is 2 octets (top bit of 0x10
        // is 0). APER: 1 + 7 pad = 80, length 02 (aligned, already on a
        // boundary), 10 00.  UPER: 1 | 00000010 | 00010000 | 00000000 + pad
        //   = 1000 0001  0000 1000  0000 0000  0(000 0000) -> 81 08 00 00.
        (
            4096,
            vec![0x80, 0x02, 0x10, 0x00],
            vec![0x81, 0x08, 0x00, 0x00],
            true,
        ),
        // 65535 = 0xFFFF: top bit set, so a leading 00 octet keeps it positive:
        // 3 octets 00 FF FF. APER 80 | 03 | 00 FF FF.
        // UPER 1 | 00000011 | 00000000 | 11111111 | 11111111 + pad
        //   = 1000 0001  1000 0000  0111 1111  1111 1111  1(000 0000)
        //   -> 81 80 7F FF 80.
        (
            65535,
            vec![0x80, 0x03, 0x00, 0xFF, 0xFF],
            vec![0x81, 0x80, 0x7F, 0xFF, 0x80],
            true,
        ),
        // 2000000 = 0x1E8480: 3 octets (top bit of 0x1E is 0).
        // APER 80 | 03 | 1E 84 80.
        // UPER 1 | 00000011 | 00011110 | 10000100 | 10000000 + pad
        //   = 1000 0001  1000 1111  0100 0010  0100 0000  0(000 0000)
        //   -> 81 8F 42 40 00.
        (
            2_000_000,
            vec![0x80, 0x03, 0x1E, 0x84, 0x80],
            vec![0x81, 0x8F, 0x42, 0x40, 0x00],
            true,
        ),
    ]
}

#[test]
fn mdbv_root_and_extension_values_encode_per_x691_12_1() {
    let (lb, ub) = MDBV;
    for (v, want_aper, want_uper, _) in mdbv_cases() {
        assert_eq!(enc(true, lb, ub, true, v), want_aper, "APER encode of {v}");
        assert_eq!(enc(false, lb, ub, true, v), want_uper, "UPER encode of {v}");
    }
}

#[test]
fn mdbv_root_and_extension_values_decode_per_x691_12_1() {
    let (lb, ub) = MDBV;
    for (v, aper_bytes, uper_bytes, outside) in mdbv_cases() {
        assert_eq!(
            dec(true, lb, ub, true, &aper_bytes),
            (v, outside),
            "APER decode of {v}"
        );
        assert_eq!(
            dec(false, lb, ub, true, &uper_bytes),
            (v, outside),
            "UPER decode of {v}"
        );
    }
}

/// A signed root, so an extension value can lie on either side of it.
/// `INTEGER (-10..10, ...)`: root range 21 -> 5 bits.
#[test]
fn values_below_and_above_a_signed_root_encode_as_unconstrained() {
    let (lb, ub) = (Some(-10), Some(10));
    // -11 = 0xF5 in one two's-complement octet. APER 80 | 01 | F5.
    // UPER 1 | 00000001 | 11110101 + pad = 1000 0000  1111 1010  1(000 0000)
    //   -> 80 FA 80.
    assert_eq!(enc(true, lb, ub, true, -11), vec![0x80, 0x01, 0xF5]);
    assert_eq!(enc(false, lb, ub, true, -11), vec![0x80, 0xFA, 0x80]);
    // 200 = 0xC8: top bit set, so 2 octets 00 C8. APER 80 | 02 | 00 C8.
    assert_eq!(enc(true, lb, ub, true, 200), vec![0x80, 0x02, 0x00, 0xC8]);
    // A root value keeps the root encoding: -10 is offset 0 in 5 bits.
    // APER: ext 0, 5 bits 00000 (range 21 <= 255: bit-field, not aligned)
    //   = 0000 00(00) -> 00.
    assert_eq!(enc(true, lb, ub, true, -10), vec![0x00]);
    assert_eq!(dec(true, lb, ub, true, &[0x80, 0x01, 0xF5]), (-11, true));
    assert_eq!(dec(false, lb, ub, true, &[0x80, 0xFA, 0x80]), (-11, true));
    assert_eq!(
        dec(true, lb, ub, true, &[0x80, 0x02, 0x00, 0xC8]),
        (200, true)
    );
}

/// Out-of-range values of a NON-extensible INTEGER are still refused, and so is
/// asking for the extension encoding of a type that has no extension marker.
#[test]
fn a_non_extensible_integer_still_refuses_out_of_range_values() {
    let mut d = PerCodecData::new_aper();
    assert!(aper::encode::encode_integer(&mut d, Some(0), Some(4095), false, 4096, false).is_err());
    let mut d = PerCodecData::new_aper();
    assert!(aper::encode::encode_integer(&mut d, Some(0), Some(4095), false, 4096, true).is_err());
    let mut d = PerCodecData::new_uper();
    assert!(uper::encode::encode_integer(&mut d, Some(0), Some(4095), false, 4096, true).is_err());
}
