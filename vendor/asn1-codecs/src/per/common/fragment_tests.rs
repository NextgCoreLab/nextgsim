//! X.691 §11.9.3.8 length fragmentation, for every length site, in both PER
//! variants.
//!
//! Expected bytes come from [`frag_octets`] / [`frag_bits`], a deliberately
//! naive transcription of §11.9.3.8 that shares no code with the codec, and the
//! transcription is itself pinned to literal header octets in
//! `the_reference_fragmenter_matches_x691_literally`. Every site is driven from
//! an octet-aligned start, so APER and UPER produce identical bytes for the
//! unconstrained cases (APER's alignment pads are all zero-length there); the
//! UPER-specific test starts mid-octet, where the two differ.

use bitvec::prelude::*;

use crate::per::{aper, uper, PerCodecData, PerCodecErrorCause};

const K16: usize = 16384;

/// The lengths the task names: both sides of every 16K boundary, the 2-octet
/// remainder case nextgcore saw fail (16,384 + 3,729), exact multiples (which
/// end in a zero-length remainder), and a > 64K multi-fragment length.
const LENGTHS: &[usize] = &[
    0,
    1,
    127,
    128,
    16_383,
    16_384,
    16_385,
    16_384 + 127,
    16_384 + 128,
    16_384 + 3_729,
    32_767,
    32_768,
    49_152,
    65_535,
    65_536,
    65_536 + 16_384 + 5,
    200_000,
];

/// One unconstrained determinant below 16K (§11.9.3.6-7).
fn short_determinant(n: usize) -> Vec<u8> {
    assert!(n < K16);
    if n < 128 {
        vec![n as u8]
    } else {
        vec![0x80 | (n >> 8) as u8, (n & 0xFF) as u8]
    }
}

/// Reference fragmenter for octet-unit content: 64K/48K/32K/16K pieces, each
/// headed by `0b11` + six-bit m, then the remainder with an ordinary
/// determinant (a single 0x00 when it is zero).
fn frag_octets(content: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut rest = content;
    while rest.len() >= K16 {
        let m = (rest.len() / K16).min(4);
        out.push(0xC0 | m as u8);
        out.extend_from_slice(&rest[..m * K16]);
        rest = &rest[m * K16..];
    }
    out.extend(short_determinant(rest.len()));
    out.extend_from_slice(rest);
    out
}

/// The same for bit-unit content (BIT STRING): the unit is a bit, so a 16K
/// fragment is 2,048 octets. `bits.len()` is chosen a multiple of 8 so each
/// piece ends on an octet boundary and the expectation stays octet-shaped.
fn frag_bits(bits: &BitSlice<u8, Msb0>) -> Vec<u8> {
    assert_eq!(bits.len() % 8, 0);
    let mut out: BitVec<u8, Msb0> = BitVec::new();
    let mut rest = bits;
    while rest.len() >= K16 {
        let m = (rest.len() / K16).min(4);
        out.extend_from_bitslice((0xC0_u8 | m as u8).view_bits::<Msb0>());
        out.extend_from_bitslice(&rest[..m * K16]);
        rest = &rest[m * K16..];
    }
    for b in short_determinant(rest.len()) {
        out.extend_from_bitslice(b.view_bits::<Msb0>());
    }
    out.extend_from_bitslice(rest);
    out.into_vec()
}

fn octets(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i * 31 + 7) as u8).collect()
}

fn new_data(aligned: bool) -> PerCodecData {
    if aligned {
        PerCodecData::new_aper()
    } else {
        PerCodecData::new_uper()
    }
}

fn read_data(bytes: &[u8], aligned: bool) -> PerCodecData {
    if aligned {
        PerCodecData::from_slice_aper(bytes)
    } else {
        PerCodecData::from_slice_uper(bytes)
    }
}

fn first_diff(a: &[u8], b: &[u8]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y).or_else(|| {
        if a.len() == b.len() {
            None
        } else {
            Some(a.len().min(b.len()))
        }
    })
}

#[test]
fn the_reference_fragmenter_matches_x691_literally() {
    assert_eq!(frag_octets(&[]), [0x00]);
    assert_eq!(frag_octets(&[1; 127])[0], 0x7F);
    assert_eq!(frag_octets(&[1; 128])[..2], [0x80, 0x80]);
    assert_eq!(frag_octets(&[1; 16_383])[..2], [0xBF, 0xFF]);
    let f = frag_octets(&[1; 16_384]);
    assert_eq!((f[0], f.len(), *f.last().unwrap()), (0xC1, 16_386, 0x00));
    let f = frag_octets(&[1; 16_384 + 3_729]);
    assert_eq!((f[0], f[16_385], f[16_386]), (0xC1, 0x8E, 0x91));
    let f = frag_octets(&[1; 32_768]);
    assert_eq!((f[0], f[32_769]), (0xC2, 0x00));
    let f = frag_octets(&[1; 49_152]);
    assert_eq!((f[0], f[49_153]), (0xC3, 0x00));
    let f = frag_octets(&[1; 65_535]);
    // 48K then 16,383: m is the largest multiple that fits, capped at 4.
    assert_eq!((f[0], f[49_153], f[49_154]), (0xC3, 0xBF, 0xFF));
    let f = frag_octets(&[1; 65_536 + 16_384 + 5]);
    assert_eq!((f[0], f[65_537], f[65_537 + 16_385]), (0xC4, 0xC1, 0x05));
    let f = frag_octets(&[1; 200_000]);
    // 3 x 64K = 196,608; remainder 3,392 = 0x0D40.
    let h = 65_537;
    assert_eq!(
        (f[0], f[h], f[2 * h], f[3 * h], f[3 * h + 1]),
        (0xC4, 0xC4, 0xC4, 0x8D, 0x40)
    );
}

#[test]
fn an_unconstrained_octet_string_fragments_and_round_trips() {
    for aligned in [true, false] {
        for &len in LENGTHS {
            let value = octets(len);
            let mut data = new_data(aligned);
            if aligned {
                aper::encode::encode_octetstring(&mut data, None, None, false, &value, false)
            } else {
                uper::encode::encode_octetstring(&mut data, None, None, false, &value, false)
            }
            .unwrap_or_else(|e| {
                panic!(
                    "len {len} aligned={aligned}: encode: {e}",
                    len = len,
                    aligned = aligned,
                    e = e
                )
            });
            let bytes = data.into_bytes();
            let expected = frag_octets(&value);
            assert!(
                bytes == expected,
                "len {len} aligned={aligned}: first difference at {:?}",
                first_diff(&bytes, &expected),
                len = len,
                aligned = aligned
            );

            let mut read = read_data(&bytes, aligned);
            let decoded = if aligned {
                aper::decode::decode_octetstring(&mut read, None, None, false)
            } else {
                uper::decode::decode_octetstring(&mut read, None, None, false)
            }
            .unwrap_or_else(|e| {
                panic!(
                    "len {len} aligned={aligned}: decode: {e}",
                    len = len,
                    aligned = aligned,
                    e = e
                )
            });
            assert!(
                decoded == value,
                "len {len} aligned={aligned}: content",
                len = len,
                aligned = aligned
            );
            assert_eq!(
                read.decode_offset_bits(),
                bytes.len() * 8,
                "len {len}",
                len = len
            );
        }
    }
}

/// `SIZE (0..20000)`: `ub` below 64K is a CONSTRAINED length (§11.9.3.3), so
/// 16K..20K octets are legal, take a 16-bit length and are NOT fragmented. The
/// encoder used to refuse them; the decoder used to read the 16K-or-more
/// length as a fragment header and look for a second determinant.
#[test]
fn a_constrained_length_of_16k_or_more_is_not_a_fragment() {
    for &len in &[16_383_usize, 16_384, 16_385, 20_000] {
        let value = octets(len);
        let mut data = PerCodecData::new_aper();
        aper::encode::encode_octetstring(&mut data, Some(0), Some(20_000), false, &value, false)
            .unwrap_or_else(|e| panic!("len {len}: {e}", len = len, e = e));
        let bytes = data.into_bytes();
        // Range 20,001 > 256: an aligned 16-bit constrained number, then the
        // aligned content.
        assert_eq!(
            bytes[..2],
            [(len >> 8) as u8, (len & 0xFF) as u8],
            "len {len}",
            len = len
        );
        assert_eq!(bytes.len(), 2 + len);

        let mut read = PerCodecData::from_slice_aper(&bytes);
        let decoded = aper::decode::decode_octetstring(&mut read, Some(0), Some(20_000), false)
            .unwrap_or_else(|e| panic!("len {len}: decode {e}", len = len, e = e));
        assert!(decoded == value, "len {len}", len = len);
    }
}

#[test]
fn an_unconstrained_bit_string_fragments_in_bits_and_round_trips() {
    for aligned in [true, false] {
        for &len in LENGTHS {
            // BIT STRING lengths count bits; keep them whole octets (see frag_bits).
            let nbits = len - len % 8;
            let store = octets(nbits / 8);
            let value = &store.view_bits::<Msb0>()[..nbits];
            let mut data = new_data(aligned);
            if aligned {
                aper::encode::encode_bitstring(&mut data, None, None, false, value, false)
            } else {
                uper::encode::encode_bitstring(&mut data, None, None, false, value, false)
            }
            .unwrap_or_else(|e| {
                panic!(
                    "bits {nbits} aligned={aligned}: encode: {e}",
                    nbits = nbits,
                    aligned = aligned,
                    e = e
                )
            });
            let bytes = data.into_bytes();
            let expected = frag_bits(value);
            assert!(
                bytes == expected,
                "bits {nbits} aligned={aligned}: first difference at {:?}",
                first_diff(&bytes, &expected),
                nbits = nbits,
                aligned = aligned
            );

            let mut read = read_data(&bytes, aligned);
            let decoded = if aligned {
                aper::decode::decode_bitstring(&mut read, None, None, false)
            } else {
                uper::decode::decode_bitstring(&mut read, None, None, false)
            }
            .unwrap_or_else(|e| {
                panic!(
                    "bits {nbits} aligned={aligned}: decode: {e}",
                    nbits = nbits,
                    aligned = aligned,
                    e = e
                )
            });
            assert!(
                decoded.as_bitslice() == value,
                "bits {nbits} aligned={aligned}",
                nbits = nbits,
                aligned = aligned
            );
        }
    }
}

/// APER PrintableString (8 bits per character here): the length counts
/// characters (§30.5.7), which are octets.
#[test]
fn an_unconstrained_aper_character_string_fragments_and_round_trips() {
    for &len in LENGTHS {
        let value: String = (0..len).map(|i| (b'A' + (i % 26) as u8) as char).collect();
        let mut data = PerCodecData::new_aper();
        aper::encode::encode_printable_string(&mut data, None, None, false, &value, false)
            .unwrap_or_else(|e| panic!("len {len}: encode {e}", len = len, e = e));
        let bytes = data.into_bytes();
        let expected = frag_octets(value.as_bytes());
        assert!(
            bytes == expected,
            "len {len}: first difference at {:?}",
            first_diff(&bytes, &expected),
            len = len
        );
        let mut read = PerCodecData::from_slice_aper(&bytes);
        let decoded = aper::decode::decode_printable_string(&mut read, None, None, false)
            .unwrap_or_else(|e| panic!("len {len}: decode {e}", len = len, e = e));
        assert!(decoded == value, "len {len}", len = len);
    }
}

/// UPER PrintableString at seven bits per character: the fragment unit is
/// still 16K CHARACTERS (§30.5.7), i.e. 16K x 7 bits of content per piece.
#[test]
fn an_unconstrained_uper_seven_bit_string_fragments_in_characters() {
    for &len in LENGTHS {
        let value: String = (0..len).map(|i| (b'a' + (i % 26) as u8) as char).collect();
        let mut data = PerCodecData::new_uper();
        uper::encode::encode_printable_string(&mut data, None, None, false, &value, false)
            .unwrap_or_else(|e| panic!("len {len}: encode {e}", len = len, e = e));
        let bytes = data.into_bytes();

        // Expected: the octet fragmenter's headers, with each character as 7 bits.
        let mut expected: BitVec<u8, Msb0> = BitVec::new();
        let chars = value.as_bytes();
        let mut rest = chars;
        let push_chars = |out: &mut BitVec<u8, Msb0>, cs: &[u8]| {
            for c in cs {
                out.extend_from_bitslice(&c.view_bits::<Msb0>()[1..]);
            }
        };
        while rest.len() >= K16 {
            let m = (rest.len() / K16).min(4);
            expected.extend_from_bitslice((0xC0_u8 | m as u8).view_bits::<Msb0>());
            push_chars(&mut expected, &rest[..m * K16]);
            rest = &rest[m * K16..];
        }
        for b in short_determinant(rest.len()) {
            expected.extend_from_bitslice(b.view_bits::<Msb0>());
        }
        push_chars(&mut expected, rest);
        let expected = expected.into_vec();
        assert!(
            bytes == expected,
            "len {len}: first difference at {:?}",
            first_diff(&bytes, &expected),
            len = len
        );

        let mut read = PerCodecData::from_slice_uper(&bytes);
        let decoded = uper::decode::decode_printable_string(&mut read, None, None, false)
            .unwrap_or_else(|e| panic!("len {len}: decode {e}", len = len, e = e));
        assert!(decoded == value, "len {len}", len = len);
    }
}

/// SEQUENCE OF with an unconstrained size, components of one octet each (an
/// INTEGER (0..255), aligned): the COUNT fragments at 16K components and the
/// components of each fragment follow its header.
#[test]
fn an_unconstrained_sequence_of_fragments_its_count_and_round_trips() {
    for aligned in [true, false] {
        for &count in LENGTHS {
            let items = octets(count);
            let mut data = new_data(aligned);
            let emit = |data: &mut PerCodecData, start: usize, n: usize| {
                for v in &items[start..start + n] {
                    if aligned {
                        aper::encode::encode_integer(
                            data,
                            Some(0),
                            Some(255),
                            false,
                            *v as i128,
                            false,
                        )?;
                    } else {
                        uper::encode::encode_integer(
                            data,
                            Some(0),
                            Some(255),
                            false,
                            *v as i128,
                            false,
                        )?;
                    }
                }
                Ok(())
            };
            if aligned {
                aper::encode::encode_sequence_of(&mut data, None, None, false, count, emit)
            } else {
                uper::encode::encode_sequence_of(&mut data, None, None, false, count, emit)
            }
            .unwrap_or_else(|e| {
                panic!(
                    "count {count} aligned={aligned}: encode {e}",
                    count = count,
                    aligned = aligned,
                    e = e
                )
            });
            let bytes = data.into_bytes();
            // Each component is exactly one octet, so the expectation is the
            // octet fragmenter over the component values.
            let expected = frag_octets(&items);
            assert!(
                bytes == expected,
                "count {count} aligned={aligned}: first difference at {:?}",
                first_diff(&bytes, &expected),
                count = count,
                aligned = aligned
            );

            let mut read = read_data(&bytes, aligned);
            let mut decoded = Vec::new();
            let collect = |data: &mut PerCodecData, n: usize| {
                for _ in 0..n {
                    let (v, _) = if aligned {
                        aper::decode::decode_integer(data, Some(0), Some(255), false)?
                    } else {
                        uper::decode::decode_integer(data, Some(0), Some(255), false)?
                    };
                    decoded.push(v as u8);
                }
                Ok(())
            };
            if aligned {
                aper::decode::decode_sequence_of(&mut read, None, None, false, collect)
            } else {
                uper::decode::decode_sequence_of(&mut read, None, None, false, collect)
            }
            .unwrap_or_else(|e| {
                panic!(
                    "count {count} aligned={aligned}: decode {e}",
                    count = count,
                    aligned = aligned,
                    e = e
                )
            });
            assert!(
                decoded == items,
                "count {count} aligned={aligned}",
                count = count,
                aligned = aligned
            );
        }
    }
}

/// A SIZE-constrained SEQUENCE OF keeps its constrained count and never
/// fragments, however many components (NGAP's largest, 1..65535, is here).
#[test]
fn a_constrained_sequence_of_count_does_not_fragment() {
    let items = octets(20_000);
    let mut data = PerCodecData::new_aper();
    aper::encode::encode_sequence_of(
        &mut data,
        Some(1),
        Some(65_535),
        false,
        items.len(),
        |d, s, n| {
            for v in &items[s..s + n] {
                aper::encode::encode_integer(d, Some(0), Some(255), false, *v as i128, false)?;
            }
            Ok(())
        },
    )
    .unwrap();
    let bytes = data.into_bytes();
    // count - lb = 19,999 = 0x4E1F in an aligned 16-bit field.
    assert_eq!(bytes[..2], [0x4E, 0x1F]);
    assert_eq!(bytes.len(), 2 + 20_000);
}

/// Open type: the derive's framing, driven directly. The inner encoding is a
/// run of octets; the frame is the octet fragmenter over it. Decoding must
/// leave the content CONTIGUOUS at the returned offset (the derive decodes the
/// inner value in place) and the data after the frame where it was.
#[test]
fn an_open_type_fragments_and_reassembles_in_place() {
    for aligned in [true, false] {
        for &len in LENGTHS {
            let content = octets(len);
            let mut inner = new_data(aligned);
            inner.append_bits(content.view_bits());
            let mut data = new_data(aligned);
            if aligned {
                aper::encode::encode_open_type(&mut data, &mut inner)
            } else {
                uper::encode::encode_open_type(&mut data, &mut inner)
            }
            .unwrap_or_else(|e| {
                panic!(
                    "len {len} aligned={aligned}: encode {e}",
                    len = len,
                    aligned = aligned,
                    e = e
                )
            });
            // A trailing field after the frame, which must survive reassembly.
            data.append_bits(0xA5_u8.view_bits::<Msb0>());
            let bytes = data.into_bytes();
            let mut expected = frag_octets(&content);
            expected.push(0xA5);
            assert!(
                bytes == expected,
                "len {len} aligned={aligned}: first difference at {:?}",
                first_diff(&bytes, &expected),
                len = len,
                aligned = aligned
            );

            let mut read = read_data(&bytes, aligned);
            let length = if aligned {
                aper::decode::decode_open_type_length(&mut read)
            } else {
                uper::decode::decode_open_type_length(&mut read)
            }
            .unwrap_or_else(|e| {
                panic!(
                    "len {len} aligned={aligned}: decode {e}",
                    len = len,
                    aligned = aligned,
                    e = e
                )
            });
            assert_eq!(length, len, "aligned={aligned}", aligned = aligned);
            let got = read.get_bytes(length).expect("contiguous content");
            assert!(
                got == content,
                "len {len} aligned={aligned}: content",
                len = len,
                aligned = aligned
            );
            assert_eq!(
                read.get_bytes(1).unwrap(),
                vec![0xA5],
                "len {len}: trailer",
                len = len
            );
        }
    }
}

/// UPER open type starting MID-OCTET: unaligned PER pads nothing, so the
/// fragment headers sit at odd bit offsets and reassembly must cut bit ranges,
/// not octets. Three leading bits `101` then a 16,384 + 3,729 frame.
#[test]
fn a_uper_open_type_fragments_from_an_unaligned_offset() {
    let len = 16_384 + 3_729;
    let content = octets(len);
    let mut inner = PerCodecData::new_uper();
    inner.append_bits(content.view_bits());
    let mut data = PerCodecData::new_uper();
    data.append_bits(bits![u8, Msb0; 1, 0, 1]);
    uper::encode::encode_open_type(&mut data, &mut inner).unwrap();
    let bytes = data.into_bytes();

    let mut expected: BitVec<u8, Msb0> = bitvec![u8, Msb0; 1, 0, 1];
    expected.extend_from_bitslice(frag_octets(&content).view_bits::<Msb0>());
    let expected = expected.into_vec();
    assert!(
        bytes == expected,
        "first difference at {:?}",
        first_diff(&bytes, &expected)
    );

    let mut read = PerCodecData::from_slice_uper(&bytes);
    read.advance_maybe_err(3, false).unwrap();
    let length = uper::decode::decode_open_type_length(&mut read).unwrap();
    assert_eq!(length, len);
    let got = read.get_bytes(len).unwrap();
    assert!(got == content);
}

/// A fragment multiplier outside 1..=4 is malformed (§11.9.3.8.1), and a frame
/// shorter than its headers claim is a buffer error -- neither a panic nor a
/// silent short read.
#[test]
fn malformed_fragments_are_errors() {
    for m in [0x00_u8, 0x05, 0x3F] {
        let bytes = [0xC0 | m, 0x00];
        let mut read = PerCodecData::from_slice_aper(&bytes);
        let err = aper::decode::decode_open_type_length(&mut read).expect_err("bad multiplier");
        assert!(
            err.to_string().contains("1 to 4"),
            "m={m}: {err}",
            m = m,
            err = err
        );
        let mut read = PerCodecData::from_slice_aper(&bytes);
        assert!(
            aper::decode::decode_octetstring(&mut read, None, None, false).is_err(),
            "m={m}",
            m = m
        );
    }

    // A 16K header with only 100 octets behind it.
    let mut truncated = vec![0xC1];
    truncated.extend(vec![0u8; 100]);
    let mut read = PerCodecData::from_slice_aper(&truncated);
    let err = aper::decode::decode_open_type_length(&mut read).expect_err("truncated");
    assert!(
        matches!(err.cause, PerCodecErrorCause::BufferTooShort),
        "{err}",
        err = err
    );
    let mut read = PerCodecData::from_slice_aper(&truncated);
    let err =
        aper::decode::decode_octetstring(&mut read, None, None, false).expect_err("truncated");
    assert!(
        matches!(err.cause, PerCodecErrorCause::BufferTooShort),
        "{err}",
        err = err
    );
}

/// Below 16K nothing changes: these are the exact bytes the codec produced
/// before fragmentation existed (single determinant, then content).
#[test]
fn lengths_below_16k_keep_their_wire_form() {
    for &len in &[0_usize, 1, 2, 3, 127, 128, 16_383] {
        let value = octets(len);
        let mut data = PerCodecData::new_aper();
        aper::encode::encode_octetstring(&mut data, None, None, false, &value, false).unwrap();
        let mut expected = short_determinant(len);
        expected.extend(&value);
        assert_eq!(data.into_bytes(), expected, "len {len}", len = len);
    }
}

/// The SIZE bounds are still enforced on the fragmenting path.
#[test]
fn a_large_upper_bound_is_still_a_bound_when_fragmenting() {
    let value = octets(70_001);
    let mut data = PerCodecData::new_aper();
    let err =
        aper::encode::encode_octetstring(&mut data, Some(0), Some(70_000), false, &value, false)
            .expect_err("over ub");
    assert!(
        err.to_string().contains("greater than upper bound 70000"),
        "{err}",
        err = err
    );

    let value = octets(70_000);
    let mut data = PerCodecData::new_aper();
    aper::encode::encode_octetstring(&mut data, Some(0), Some(70_000), false, &value, false)
        .unwrap();
    let bytes = data.into_bytes();
    assert_eq!(
        bytes,
        frag_octets(&value),
        "ub >= 64K fragments like unconstrained"
    );
}
