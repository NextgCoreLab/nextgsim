//! #253: SEQUENCE extension additions (X.691 §19.7-§19.9) through the derive.
//!
//! A hand-annotated type, written exactly as the asn1-compiler now generates
//! one:
//!
//! ```text
//! S ::= SEQUENCE {
//!     a INTEGER (0..7),
//!     b BOOLEAN OPTIONAL,
//!     ...,
//!     c INTEGER (0..255),                        -- a lone extension addition
//!     [[ d BOOLEAN, e INTEGER (0..3) OPTIONAL ]] -- an extension addition group
//! }
//! ```
//!
//! Every expected octet string is derived by hand from X.691 beside it.

#![allow(non_camel_case_types)]

use asn1_codecs::aper::AperCodec;
use asn1_codecs::uper::UperCodec;
use asn1_codecs::PerCodecData;
use asn1_codecs_derive::{AperCodec, UperCodec};

#[derive(Debug, Clone, PartialEq, AperCodec, UperCodec)]
#[asn(type = "INTEGER", lb = "0", ub = "7")]
struct A(u8);

#[derive(Debug, Clone, PartialEq, AperCodec, UperCodec)]
#[asn(type = "BOOLEAN")]
struct B(bool);

#[derive(Debug, Clone, PartialEq, AperCodec, UperCodec)]
#[asn(type = "INTEGER", lb = "0", ub = "255")]
struct C(u8);

#[derive(Debug, Clone, PartialEq, AperCodec, UperCodec)]
#[asn(type = "INTEGER", lb = "0", ub = "3")]
struct E(u8);

#[derive(Debug, Clone, PartialEq, AperCodec, UperCodec)]
#[asn(type = "SEQUENCE", extensible = true, optional_fields = 1)]
struct S {
    a: A,
    #[asn(optional_idx = 0)]
    b: Option<B>,
    #[asn(extension_additions = true)]
    extensions: S_ExtAdditions,
}

#[derive(Debug, Clone, PartialEq, Default, AperCodec, UperCodec)]
#[asn(type = "SEQUENCE-EXTENSIONS")]
struct S_ExtAdditions {
    #[asn(extension_idx = 0)]
    c: Option<C>,
    #[asn(extension_idx = 1, extension_group = true)]
    d: Option<B>,
    #[asn(extension_idx = 1, extension_group = true, group_optional_idx = 0)]
    e: Option<E>,
}

// The same type as an older peer knows it: no extension additions at all.
// (Plain comments: the derive rejects non-`asn` attributes such as doc.)
#[derive(Debug, Clone, PartialEq, AperCodec, UperCodec)]
#[asn(type = "SEQUENCE", extensible = true, optional_fields = 1)]
struct S_Root {
    a: A,
    #[asn(optional_idx = 0)]
    b: Option<B>,
}

// An older peer that knows only the lone addition `c`, not the group.
#[derive(Debug, Clone, PartialEq, AperCodec, UperCodec)]
#[asn(type = "SEQUENCE", extensible = true, optional_fields = 1)]
struct S_V1 {
    a: A,
    #[asn(optional_idx = 0)]
    b: Option<B>,
    #[asn(extension_additions = true)]
    extensions: S_V1_ExtAdditions,
}

#[derive(Debug, Clone, PartialEq, Default, AperCodec, UperCodec)]
#[asn(type = "SEQUENCE-EXTENSIONS")]
struct S_V1_ExtAdditions {
    #[asn(extension_idx = 0)]
    c: Option<C>,
}

fn full() -> S {
    S {
        a: A(5),
        b: None,
        extensions: S_ExtAdditions {
            c: Some(C(200)),
            d: Some(B(true)),
            e: Some(E(2)),
        },
    }
}

fn uper<T: UperCodec>(v: &T) -> Vec<u8> {
    let mut d = PerCodecData::new_uper();
    v.uper_encode(&mut d).expect("UPER encode");
    d.into_bytes()
}

fn aper<T: AperCodec>(v: &T) -> Vec<u8> {
    let mut d = PerCodecData::new_aper();
    v.aper_encode(&mut d).expect("APER encode");
    d.into_bytes()
}

fn from_uper<T: UperCodec<Output = T>>(b: &[u8]) -> T {
    T::uper_decode(&mut PerCodecData::from_slice_uper(b)).expect("UPER decode")
}

fn from_aper<T: AperCodec<Output = T>>(b: &[u8]) -> T {
    T::aper_decode(&mut PerCodecData::from_slice_aper(b)).expect("APER decode")
}

/// UPER, a=5, b absent, c=200, d=TRUE, e=2:
///
/// ```text
/// 1            extension bit: additions present (§19.1, §19.6)
/// 0            b absent (root optional bitmap)
/// 101          a = 5, INTEGER (0..7) in 3 bits
/// 0 000001     normally small length n = 2 additions (§19.7, §11.9.3.4: n-1)
/// 11           presence bitmap: c and the group both present (§19.8)
/// 00000001     open type length, 1 octet (§19.9, §11.2)
/// 11001000     c = 200, INTEGER (0..255) in 8 bits
/// 00000001     open type length, 1 octet
/// 1 1 10 0000  the group as SEQUENCE { d, e OPTIONAL }: e present, d TRUE,
///              e = 2 in 2 bits, padded to an octet (§11.2.1)
/// 00           final padding
/// ```
///
/// = `A8 1C 07 20 07 80`.
const UPER_FULL: [u8; 6] = [0xA8, 0x1C, 0x07, 0x20, 0x07, 0x80];

/// APER, same value. Up to and including the presence bitmap the bits match
/// UPER (bit-fields and the normally small length are not aligned); each open
/// type's length determinant is octet-aligned (§11.9.3.5 via §11.2), so two
/// padding bits follow the bitmap, and the contents start on an octet:
///
/// ```text
/// 10101000 00011100   ext=1, b absent, a=101, n-1=000001, bitmap 11, pad 00
/// 00000001 11001000   length 1, c = 200 (range 256: one aligned octet, §11.5.7.2)
/// 00000001 11100000   length 1, the group: e present, d TRUE, e = 2, padded
/// ```
///
/// = `A8 1C 01 C8 01 E0`.
const APER_FULL: [u8; 6] = [0xA8, 0x1C, 0x01, 0xC8, 0x01, 0xE0];

#[test]
fn a_lone_addition_and_a_group_encode_to_the_x691_bytes() {
    assert_eq!(uper(&full()), UPER_FULL, "UPER");
    assert_eq!(aper(&full()), APER_FULL, "APER");
}

#[test]
fn a_lone_addition_and_a_group_decode_from_the_x691_bytes() {
    assert_eq!(from_uper::<S>(&UPER_FULL), full(), "UPER");
    assert_eq!(from_aper::<S>(&APER_FULL), full(), "APER");
}

/// No additions present: the extension bit is 0 and nothing follows the root
/// (§19.6), so the bytes are what a root-only type produces.
#[test]
fn without_additions_the_wire_form_is_the_root_form() {
    let root_only = S {
        a: A(5),
        b: Some(B(false)),
        extensions: S_ExtAdditions::default(),
    };
    let old = S_Root {
        a: A(5),
        b: Some(B(false)),
    };
    // 0 ext, 1 b present, 101 a, 0 b FALSE -> 0110 1000 = 0x68.
    assert_eq!(uper(&root_only), vec![0x68]);
    assert_eq!(uper(&root_only), uper(&old));
    assert_eq!(aper(&root_only), aper(&old));
    assert_eq!(from_uper::<S>(&[0x68]), root_only);
}

/// A group's OPTIONAL member may be absent: its bit in the group's own bitmap
/// is 0 and only the mandatory member is encoded.
#[test]
fn a_group_without_its_optional_member() {
    let v = S {
        a: A(0),
        b: None,
        extensions: S_ExtAdditions {
            c: None,
            d: Some(B(false)),
            e: None,
        },
    };
    // UPER: 1 0 000 | 0 000001 | 01 | 00000001 | 0 0 000000
    //  -> 10000000 00010100 00000100 0000(0000) = 80 14 04 00
    assert_eq!(uper(&v), vec![0x80, 0x14, 0x04, 0x00]);
    assert_eq!(from_uper::<S>(&uper(&v)), v);
    assert_eq!(from_aper::<S>(&aper(&v)), v);
}

/// A group present with its mandatory member missing is a value the type
/// cannot carry; the encoder refuses it instead of emitting a malformed group.
#[test]
fn a_group_missing_its_mandatory_member_is_refused() {
    let v = S {
        a: A(0),
        b: None,
        extensions: S_ExtAdditions {
            c: None,
            d: None,
            e: Some(E(1)),
        },
    };
    let mut d = PerCodecData::new_uper();
    assert!(v.uper_encode(&mut d).is_err());
}

/// A peer that knows fewer additions skips the ones it does not know by their
/// open-type length (§19.9) and still reads the root and what it knows.
#[test]
fn an_older_peer_skips_unknown_additions() {
    let v1 = from_uper::<S_V1>(&UPER_FULL);
    assert_eq!(v1.a, A(5));
    assert_eq!(v1.extensions.c, Some(C(200)));
    let root = from_uper::<S_Root>(&UPER_FULL);
    assert_eq!(root.a, A(5));
    assert_eq!(from_aper::<S_V1>(&APER_FULL).extensions.c, Some(C(200)));
    assert_eq!(from_aper::<S_Root>(&APER_FULL).a, A(5));
}

/// What follows the SEQUENCE is read from the right place after additions
/// were decoded: the open-type frames are consumed exactly.
#[test]
fn decoding_stops_exactly_after_the_additions() {
    let mut bytes = UPER_FULL.to_vec();
    bytes.push(0xFF);
    let mut d = PerCodecData::from_slice_uper(&bytes);
    assert_eq!(S::uper_decode(&mut d).expect("decode"), full());
    assert_eq!(
        d.decode_offset_bits(),
        46,
        "the 46 bits of the value, not the padding"
    );
}

/// The presence bitmap's length is a normally small length (X.691 §11.9.3.4):
/// up to 64 it is a 0 bit and n-1 in six bits. The encoder used to switch to
/// the long form above 32, which the decoder (correctly) does not expect. A
/// SEQUENCE with 33..=64 extension additions -- common in TS 38.331 -- would
/// not round-trip. n = 40, only the first addition present:
///
/// ```text
/// 0 100111   n-1 = 39
/// 1 0{39}    the 40 presence bits
/// 0          padding
/// ```
///
/// = `4F 00 00 00 00 00`.
#[test]
fn a_presence_bitmap_of_33_to_64_additions_uses_the_short_length_form() {
    use bitvec::prelude::*;
    let mut presence = bitvec![u8, Msb0; 0; 40];
    presence.set(0, true);
    let mut d = PerCodecData::new_uper();
    asn1_codecs::encode_sequence_extension_additions_bitmap(&mut d, &presence, false)
        .expect("encode");
    assert_eq!(d.into_bytes(), vec![0x4F, 0x00, 0x00, 0x00, 0x00, 0x00]);

    for n in [1_usize, 2, 32, 33, 63, 64, 65, 100] {
        for aligned in [false, true] {
            let mut p = bitvec![u8, Msb0; 0; n];
            p.set(n - 1, true);
            let mut d = if aligned {
                PerCodecData::new_aper()
            } else {
                PerCodecData::new_uper()
            };
            asn1_codecs::encode_sequence_extension_additions_bitmap(&mut d, &p, aligned)
                .expect("encode");
            let bytes = d.into_bytes();
            let mut r = if aligned {
                PerCodecData::from_slice_aper(&bytes)
            } else {
                PerCodecData::from_slice_uper(&bytes)
            };
            let back = asn1_codecs::decode_sequence_extension_additions_bitmap(&mut r, aligned)
                .expect("decode");
            assert_eq!(back, p, "n = {n}, aligned = {aligned}");
        }
    }
}
