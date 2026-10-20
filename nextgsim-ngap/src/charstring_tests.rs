//! Character strings through the GENERATED NGAP types (X.691 §30.5.7, §30.6).
//!
//! `AMFName ::= PrintableString (SIZE(1..150, ...))` aligns its characters
//! because of its upper bound (150 × 8 > 16), whatever the value's length; the
//! fork once left a 1- or 2-character name unaligned. `AMFNameUTF8String ::=
//! UTF8String (SIZE(1..150, ...))` has no PER-visible constraint: an
//! unconstrained length then the octets. Bytes derived by hand from X.691.

use asn1_codecs::aper::AperCodec;
use asn1_codecs::PerCodecData;

use crate::codec::generated::{AMFName, AMFNameUTF8String, RANNodeName};

fn enc<T: AperCodec>(v: &T) -> Vec<u8> {
    let mut d = PerCodecData::new_aper();
    v.aper_encode(&mut d).expect("encode");
    d.into_bytes()
}

#[test]
fn short_names_are_octet_aligned() {
    // ext 0 | length-1 = 0 in 8 bits | pad -> 00 00 | 41
    assert_eq!(enc(&AMFName("A".into())), [0x00, 0x00, 0x41]);
    // ext 0 | length-1 = 1 | pad -> 00 80 | 67 4E
    assert_eq!(enc(&RANNodeName("gN".into())), [0x00, 0x80, 0x67, 0x4E]);
}

#[test]
fn a_utf8_name_has_an_unconstrained_length_and_no_extension_bit() {
    assert_eq!(enc(&AMFNameUTF8String("A".into())), [0x01, 0x41]);
}
