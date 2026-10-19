//! Octet alignment of BIT STRING and OCTET STRING content, X.691 §16.9-16.11
//! and §17.6-17.8, in both PER variants.
//!
//! The rule the fork got wrong: only a FIXED-size string is placed without
//! alignment when it is short (≤16 bits, ≤2 octets — §16.9, §17.6). Every
//! other string — variable size, or fixed at 64K or more — goes in "a bit-field
//! (octet-aligned in the ALIGNED variant)" after its length determinant
//! (§16.11, §17.8), whatever its length. The fork applied the fixed-size
//! threshold to variable sizes too, so a 1..16-bit TransportLayerAddress
//! (`SIZE (1..160, ...)`) or a 1..2-octet variable OCTET STRING was packed
//! straight after its length bits. An empty string has no bit-field to align.
//!
//! Every expected byte string below is derived by hand from X.691, with the
//! derivation beside it; none is captured from the encoder. UPER never aligns
//! (§16, §17: "octet-aligned in the ALIGNED variant" only), so its cases pin
//! that it is untouched.

use bitvec::prelude::*;

use crate::per::{aper, uper, PerCodecData};

const BEEF: [u8; 2] = [0xBE, 0xEF];

/// The first `n` bits of `bytes`.
fn bits(bytes: &[u8], n: usize) -> BitVec<u8, Msb0> {
    bytes.view_bits::<Msb0>()[..n].to_bitvec()
}

/// Encode one BIT STRING, optionally preceded by a single `1` bit so the string
/// does not start on an octet boundary.
fn enc_bits(
    aligned: bool,
    prefix: bool,
    lb: Option<i128>,
    ub: Option<i128>,
    ext: bool,
    value: &BitSlice<u8, Msb0>,
) -> Vec<u8> {
    let mut d = if aligned {
        PerCodecData::new_aper()
    } else {
        PerCodecData::new_uper()
    };
    if prefix {
        d.encode_bool(true);
    }
    if aligned {
        aper::encode::encode_bitstring(&mut d, lb, ub, ext, value, false).unwrap();
    } else {
        uper::encode::encode_bitstring(&mut d, lb, ub, ext, value, false).unwrap();
    }
    d.into_bytes()
}

fn dec_bits(
    aligned: bool,
    prefix: bool,
    lb: Option<i128>,
    ub: Option<i128>,
    ext: bool,
    bytes: &[u8],
) -> BitVec<u8, Msb0> {
    let mut d = if aligned {
        PerCodecData::from_slice_aper(bytes)
    } else {
        PerCodecData::from_slice_uper(bytes)
    };
    if prefix {
        assert!(d.decode_bool().unwrap());
    }
    if aligned {
        aper::decode::decode_bitstring(&mut d, lb, ub, ext).unwrap()
    } else {
        uper::decode::decode_bitstring(&mut d, lb, ub, ext).unwrap()
    }
}

fn enc_octets(
    aligned: bool,
    prefix: bool,
    lb: Option<i128>,
    ub: Option<i128>,
    value: &[u8],
) -> Vec<u8> {
    let mut d = if aligned {
        PerCodecData::new_aper()
    } else {
        PerCodecData::new_uper()
    };
    if prefix {
        d.encode_bool(true);
    }
    let v = value.to_vec();
    if aligned {
        aper::encode::encode_octetstring(&mut d, lb, ub, false, &v, false).unwrap();
    } else {
        uper::encode::encode_octetstring(&mut d, lb, ub, false, &v, false).unwrap();
    }
    d.into_bytes()
}

fn dec_octets(
    aligned: bool,
    prefix: bool,
    lb: Option<i128>,
    ub: Option<i128>,
    bytes: &[u8],
) -> Vec<u8> {
    let mut d = if aligned {
        PerCodecData::from_slice_aper(bytes)
    } else {
        PerCodecData::from_slice_uper(bytes)
    };
    if prefix {
        assert!(d.decode_bool().unwrap());
    }
    if aligned {
        aper::decode::decode_octetstring(&mut d, lb, ub, false).unwrap()
    } else {
        uper::decode::decode_octetstring(&mut d, lb, ub, false).unwrap()
    }
}

struct BitCase {
    name: &'static str,
    aligned: bool,
    prefix: bool,
    lb: Option<i128>,
    ub: Option<i128>,
    ext: bool,
    value: BitVec<u8, Msb0>,
    wire: &'static [u8],
}

fn bit_cases() -> Vec<BitCase> {
    let tla = (Some(1), Some(160), true); // TransportLayerAddress, TS 38.413 / 36.413
    vec![
        // ---- variable size, APER: content octet-aligned whatever its length (§16.11)
        BitCase {
            // ext 0 | len-1 = 0 in 8 bits (range 160, §10.5.7.1) | pad 7 | '1' pad 7
            name: "TLA 1 bit",
            aligned: true,
            prefix: false,
            lb: tla.0,
            ub: tla.1,
            ext: tla.2,
            value: bits(&[0x80], 1),
            wire: &[0x00, 0x00, 0x80],
        },
        BitCase {
            // ext 0 | 00000111 | pad 7 | A5  =  0000_0011 1000_0000 A5
            name: "TLA 8 bits",
            aligned: true,
            prefix: false,
            lb: tla.0,
            ub: tla.1,
            ext: tla.2,
            value: bits(&[0xA5], 8),
            wire: &[0x03, 0x80, 0xA5],
        },
        BitCase {
            // ext 0 | 00001110 | pad 7 | 15 bits of BEEF (1011111011101110) pad 1
            name: "TLA 15 bits",
            aligned: true,
            prefix: false,
            lb: tla.0,
            ub: tla.1,
            ext: tla.2,
            value: bits(&BEEF, 15),
            wire: &[0x07, 0x00, 0xBE, 0xEE],
        },
        BitCase {
            // ext 0 | 00001111 | pad 7 | BE EF
            name: "TLA 16 bits",
            aligned: true,
            prefix: false,
            lb: tla.0,
            ub: tla.1,
            ext: tla.2,
            value: bits(&BEEF, 16),
            wire: &[0x07, 0x80, 0xBE, 0xEF],
        },
        BitCase {
            // ext 0 | 00010000 | pad 7 | BE EF '1' pad 7  (aligned before and after the fix)
            name: "TLA 17 bits",
            aligned: true,
            prefix: false,
            lb: tla.0,
            ub: tla.1,
            ext: tla.2,
            value: bits(&[0xBE, 0xEF, 0x80], 17),
            wire: &[0x08, 0x00, 0xBE, 0xEF, 0x80],
        },
        BitCase {
            // ext 0 | 00011111 | pad 7 | 4 octets — an IPv4 address
            name: "TLA 32 bits",
            aligned: true,
            prefix: false,
            lb: tla.0,
            ub: tla.1,
            ext: tla.2,
            value: bits(&[10, 0, 0, 1], 32),
            wire: &[0x0F, 0x80, 10, 0, 0, 1],
        },
        BitCase {
            // SIZE (0..16): length 0 in 5 bits (range 17); no bit-field, so no pad
            name: "SIZE(0..16) empty",
            aligned: true,
            prefix: false,
            lb: Some(0),
            ub: Some(16),
            ext: false,
            value: BitVec::new(),
            wire: &[0x00],
        },
        BitCase {
            // 10000 (5 bits) | pad 3 | BE EF
            name: "SIZE(0..16) 16 bits",
            aligned: true,
            prefix: false,
            lb: Some(0),
            ub: Some(16),
            ext: false,
            value: bits(&BEEF, 16),
            wire: &[0x80, 0xBE, 0xEF],
        },
        BitCase {
            // prefix 1 | 00001 (5 bits) | pad 2 | '1' pad 7
            name: "SIZE(0..16) 1 bit after a prefix",
            aligned: true,
            prefix: true,
            lb: Some(0),
            ub: Some(16),
            ext: false,
            value: bits(&[0x80], 1),
            wire: &[0x84, 0x80],
        },
        BitCase {
            // SIZE (1..64): length-1 = 63 in 6 bits (range 64) | pad 2 | 8 octets
            name: "SIZE(1..64) 64 bits",
            aligned: true,
            prefix: false,
            lb: Some(1),
            ub: Some(64),
            ext: false,
            value: bits(&[1, 2, 3, 4, 5, 6, 7, 8], 64),
            wire: &[0xFC, 1, 2, 3, 4, 5, 6, 7, 8],
        },
        BitCase {
            // unconstrained: prefix 1 | pad 7 | length 0x08 (octet, §11.9.3.6) | A5
            name: "unconstrained 8 bits after a prefix",
            aligned: true,
            prefix: true,
            lb: None,
            ub: None,
            ext: false,
            value: bits(&[0xA5], 8),
            wire: &[0x80, 0x08, 0xA5],
        },
        // ---- fixed size, APER: unchanged by the fix (§16.8-16.10)
        BitCase {
            // SIZE (0): not encoded at all; only the prefix bit
            name: "fixed 0",
            aligned: true,
            prefix: true,
            lb: Some(0),
            ub: Some(0),
            ext: false,
            value: BitVec::new(),
            wire: &[0x80],
        },
        BitCase {
            // prefix 1 | '1' — no length, no pad (§16.9)
            name: "fixed 1",
            aligned: true,
            prefix: true,
            lb: Some(1),
            ub: Some(1),
            ext: false,
            value: bits(&[0x80], 1),
            wire: &[0xC0],
        },
        BitCase {
            // prefix 1 | 101111101110111
            name: "fixed 15",
            aligned: true,
            prefix: true,
            lb: Some(15),
            ub: Some(15),
            ext: false,
            value: bits(&BEEF, 15),
            wire: &[0xDF, 0x77],
        },
        BitCase {
            // prefix 1 | BEEF unaligned = 1101_1111 0111_0111 1000_0000
            name: "fixed 16",
            aligned: true,
            prefix: true,
            lb: Some(16),
            ub: Some(16),
            ext: false,
            value: bits(&BEEF, 16),
            wire: &[0xDF, 0x77, 0x80],
        },
        BitCase {
            // extensible SIZE (16, ...) in the root (NRencryptionAlgorithms):
            // ext 0 | 16 bits, no pad
            name: "fixed 16 extensible",
            aligned: true,
            prefix: false,
            lb: Some(16),
            ub: Some(16),
            ext: true,
            value: bits(&BEEF, 16),
            wire: &[0x5F, 0x77, 0x80],
        },
        BitCase {
            // prefix 1 | pad 7 | BE EF '1' (§16.10)
            name: "fixed 17",
            aligned: true,
            prefix: true,
            lb: Some(17),
            ub: Some(17),
            ext: false,
            value: bits(&[0xBE, 0xEF, 0x80], 17),
            wire: &[0x80, 0xBE, 0xEF, 0x80],
        },
        BitCase {
            // prefix 1 | pad 7 | 4 octets
            name: "fixed 32",
            aligned: true,
            prefix: true,
            lb: Some(32),
            ub: Some(32),
            ext: false,
            value: bits(&[1, 2, 3, 4], 32),
            wire: &[0x80, 1, 2, 3, 4],
        },
        // ---- UPER: never aligned, before or after the fix
        BitCase {
            // ext 0 | 00000111 | A5  =  0000_0011 1101_0010 1000_0000
            name: "UPER TLA 8 bits",
            aligned: false,
            prefix: false,
            lb: tla.0,
            ub: tla.1,
            ext: tla.2,
            value: bits(&[0xA5], 8),
            wire: &[0x03, 0xD2, 0x80],
        },
        BitCase {
            // ext 0 | 00001111 | BEEF  =  0000_0111 1101_1111 0111_0111 1000_0000
            name: "UPER TLA 16 bits",
            aligned: false,
            prefix: false,
            lb: tla.0,
            ub: tla.1,
            ext: tla.2,
            value: bits(&BEEF, 16),
            wire: &[0x07, 0xDF, 0x77, 0x80],
        },
        BitCase {
            // ext 0 | 00010000 | 17 bits, no pad
            name: "UPER TLA 17 bits",
            aligned: false,
            prefix: false,
            lb: tla.0,
            ub: tla.1,
            ext: tla.2,
            value: bits(&[0xBE, 0xEF, 0x80], 17),
            wire: &[0x08, 0x5F, 0x77, 0xC0],
        },
        BitCase {
            // prefix 1 | 17 bits
            name: "UPER fixed 17",
            aligned: false,
            prefix: true,
            lb: Some(17),
            ub: Some(17),
            ext: false,
            value: bits(&[0xBE, 0xEF, 0x80], 17),
            wire: &[0xDF, 0x77, 0xC0],
        },
    ]
}

#[test]
fn bit_string_alignment_matches_x691() {
    let mut failures = Vec::new();
    for c in bit_cases() {
        let got = enc_bits(c.aligned, c.prefix, c.lb, c.ub, c.ext, &c.value);
        if got != c.wire {
            failures.push(format!(
                "{}: encode {:02X?}, X.691 says {:02X?}",
                c.name, got, c.wire
            ));
            continue;
        }
        let back = dec_bits(c.aligned, c.prefix, c.lb, c.ub, c.ext, c.wire);
        if back != c.value {
            failures.push(format!("{}: decode gave {} bits", c.name, back.len()));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

struct OctetCase {
    name: &'static str,
    aligned: bool,
    prefix: bool,
    lb: Option<i128>,
    ub: Option<i128>,
    value: &'static [u8],
    wire: &'static [u8],
}

const OCTET_CASES: &[OctetCase] = &[
    // ---- variable size, APER: content octet-aligned whatever its length (§17.8)
    OctetCase {
        // SIZE (1..8): length-1 = 0 in 3 bits | pad 5 | 5A
        name: "SIZE(1..8) 1 octet",
        aligned: true,
        prefix: false,
        lb: Some(1),
        ub: Some(8),
        value: &[0x5A],
        wire: &[0x00, 0x5A],
    },
    OctetCase {
        // 001 | pad 5 | 5A C3
        name: "SIZE(1..8) 2 octets",
        aligned: true,
        prefix: false,
        lb: Some(1),
        ub: Some(8),
        value: &[0x5A, 0xC3],
        wire: &[0x20, 0x5A, 0xC3],
    },
    OctetCase {
        // 010 | pad 5 | 5A C3 7E  (aligned before and after the fix)
        name: "SIZE(1..8) 3 octets",
        aligned: true,
        prefix: false,
        lb: Some(1),
        ub: Some(8),
        value: &[0x5A, 0xC3, 0x7E],
        wire: &[0x40, 0x5A, 0xC3, 0x7E],
    },
    OctetCase {
        // SIZE (0..2): length 0 in 2 bits; nothing to align
        name: "SIZE(0..2) empty",
        aligned: true,
        prefix: false,
        lb: Some(0),
        ub: Some(2),
        value: &[],
        wire: &[0x00],
    },
    OctetCase {
        // prefix 1 | 01 | pad 5 | 5A
        name: "SIZE(0..2) 1 octet after a prefix",
        aligned: true,
        prefix: true,
        lb: Some(0),
        ub: Some(2),
        value: &[0x5A],
        wire: &[0xA0, 0x5A],
    },
    // ---- fixed size, APER: unchanged (§17.6-17.7)
    OctetCase {
        // prefix 1 | 5A unaligned = 1010_1101 0000_0000
        name: "fixed 1 (SST)",
        aligned: true,
        prefix: true,
        lb: Some(1),
        ub: Some(1),
        value: &[0x5A],
        wire: &[0xAD, 0x00],
    },
    OctetCase {
        // prefix 1 | 5A C3 unaligned
        name: "fixed 2",
        aligned: true,
        prefix: true,
        lb: Some(2),
        ub: Some(2),
        value: &[0x5A, 0xC3],
        wire: &[0xAD, 0x61, 0x80],
    },
    OctetCase {
        // prefix 1 | pad 7 | 5A C3 7E
        name: "fixed 3 (PLMN, TAC)",
        aligned: true,
        prefix: true,
        lb: Some(3),
        ub: Some(3),
        value: &[0x5A, 0xC3, 0x7E],
        wire: &[0x80, 0x5A, 0xC3, 0x7E],
    },
    // ---- UPER: never aligned
    OctetCase {
        // 000 | 5A  =  0000_1011 0100_0000
        name: "UPER SIZE(1..8) 1 octet",
        aligned: false,
        prefix: false,
        lb: Some(1),
        ub: Some(8),
        value: &[0x5A],
        wire: &[0x0B, 0x40],
    },
];

#[test]
fn octet_string_alignment_matches_x691() {
    let mut failures = Vec::new();
    for c in OCTET_CASES {
        let got = enc_octets(c.aligned, c.prefix, c.lb, c.ub, c.value);
        if got != c.wire {
            failures.push(format!(
                "{}: encode {:02X?}, X.691 says {:02X?}",
                c.name, got, c.wire
            ));
            continue;
        }
        let back = dec_octets(c.aligned, c.prefix, c.lb, c.ub, c.wire);
        if back != c.value {
            failures.push(format!("{}: decode gave {:02X?}", c.name, back));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// An extension (out-of-root) BIT STRING is decoded as an unconstrained one
/// (§16.7): ext 1 | pad 7 | length octet | content. A 9-bit value outside
/// `SIZE (16, ...)` is the shape a newer peer would send.
#[test]
fn an_out_of_root_bit_string_decodes_as_unconstrained() {
    // 1 pad7 | 0x09 | 1011_1110 1(000_0000)
    let wire = [0x80, 0x09, 0xBE, 0x80];
    let got = dec_bits(true, false, Some(16), Some(16), true, &wire);
    assert_eq!(got, bits(&[0xBE, 0x80], 9));
}
