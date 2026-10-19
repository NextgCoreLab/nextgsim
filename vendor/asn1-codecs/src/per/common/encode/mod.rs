//! ASN.1 PER Encoding common functions

use bitvec::prelude::*;

use crate::per::{PerCodecData, PerCodecError, PerCodecErrorCause};

mod encode_internal;

#[allow(unused)]
use encode_internal::*;

pub(crate) use encode_internal::{encode_fragmented_common, length_fragments};

// Functions defined in this module are called by the respective API functions in the codecs. For
// example, `crate::aper::encode::encode_choice_index` would call `encode_choice_index_common`
// with `aligned` as `true`.

// Common function to encode a Choice Index
//
// X.691 §23.5-23.6. For an extensible CHOICE the extension bit comes first, then
// the index. Which index encoding applies depends on that bit:
//
//   - root arm (§23.6): a constrained whole number over the root's `lb..=ub`.
//   - extension arm (§23.6): a **normally small non-negative whole number**,
//     numbered from 0 in the order the extension additions are declared -- NOT
//     continuing the root's numbering, and not constrained by `ub`.
//
// The extension arm's *value* is then wrapped as an open type (§23.8), which this
// function deliberately does not do: it writes the index only, and the caller
// appends the value. Keeping the split here means the generated code frames the
// value exactly the way it already frames an ASN.1 OPEN TYPE, rather than this
// function needing to know how to encode an arbitrary caller-side type.
pub(crate) fn encode_choice_idx_common(
    data: &mut PerCodecData,
    lb: i128,
    ub: i128,
    is_extensible: bool,
    idx: i128,
    extended: bool,
    aligned: bool,
) -> Result<(), PerCodecError> {
    if extended && !is_extensible {
        return Err(PerCodecError::new(
            PerCodecErrorCause::Generic,
            "An extension arm was selected in a CHOICE that has no extension marker",
        ));
    }

    if is_extensible {
        data.encode_bool(extended);
    }

    if extended {
        // §23.6: the extension index is normally small and independent of the
        // root's bounds, so `lb`/`ub` are deliberately not consulted here.
        encode_normally_small_non_negative_whole_number_common(data, idx, aligned)
    } else {
        encode_integer_common(data, Some(lb), Some(ub), false, idx, false, aligned)
    }
}

// Common function to encode a sequence header.
pub(crate) fn encode_sequence_header_common(
    data: &mut PerCodecData,
    is_extensible: bool,
    optionals: &BitSlice<u8, Msb0>,
    extended: bool,
    _aligned: bool,
) -> Result<(), PerCodecError> {
    if extended {
        return Err(PerCodecError::new(
            PerCodecErrorCause::EncodeNotSupported,
            "Encode of extended sequence not yet implemented",
        ));
    }

    if is_extensible {
        data.encode_bool(extended);
    }

    data.append_bits(optionals);

    data.dump_encode();

    Ok(())
}

// Common function to encode an integer
pub(crate) fn encode_integer_common(
    data: &mut PerCodecData,
    lb: Option<i128>,
    ub: Option<i128>,
    is_extensible: bool,
    value: i128,
    extended: bool,
    aligned: bool,
) -> Result<(), PerCodecError> {
    if extended {
        return Err(PerCodecError::new(
            PerCodecErrorCause::EncodeNotSupported,
            "Encode of extended integer not yet implemented",
        ));
    }

    if is_extensible {
        data.encode_bool(extended);
    }

    match (lb, ub) {
        (None, _) => encode_unconstrained_whole_number_common(data, value, aligned)?,
        (Some(lb), None) => encode_semi_constrained_whole_number_common(data, lb, value, aligned)?,
        (Some(lb), Some(ub)) => {
            encode_constrained_whole_number_common(data, lb, ub, value, aligned)?
        }
    };

    data.dump_encode();

    Ok(())
}

// Common function to encode a real value
// Refer to ITU X.691 section 15 and ITU X.690 section 8.5
pub(crate) fn encode_real_common(
    data: &mut PerCodecData,
    value: f64,
    aligned: bool,
) -> Result<(), PerCodecError> {
    // Extract the value to encode
    let mut encoded_data = BitVec::<u8, Msb0>::new();
    // Encoding process is detailed in X.690 section 8.5
    if value == 0.0 {
        // -0.0 uses a reserved value, and +0.0 uses no data bits
        if f64::is_sign_negative(value) {
            encoded_data
                .extend_from_bitslice::<u8, Msb0>(super::NEGATIVE_ZERO.to_be_bytes().as_bits());
        } else {
            // This is +0.0, so there is no need to append any data bits
        }
    } else if value == f64::INFINITY {
        encoded_data.extend_from_bitslice::<u8, Msb0>(super::INFINITY.to_be_bytes().as_bits());
    } else if value == f64::NEG_INFINITY {
        encoded_data
            .extend_from_bitslice::<u8, Msb0>(super::NEGATIVE_INFINITY.to_be_bytes().as_bits());
    } else if value.is_nan() {
        encoded_data.extend_from_bitslice::<u8, Msb0>(super::NOT_A_NUMBER.to_be_bytes().as_bits());
    } else {
        // This is a standard non-zero value. For simplicity, always encode
        // using base 10 encoding based on ISO 6093 NR3.
        // TODO: add in support for binary encoding, which can improve space usage.
        encoded_data.extend_from_bitslice::<u8, Msb0>(super::BASE_10_NR3.to_be_bytes().as_bits());
        let encoded_value = format!("{:e}", value);
        encoded_data.extend_from_bitslice::<u8, Msb0>(encoded_value.as_bits());
    }

    // Set the length (X.691 section 15)
    encode_length_determinent_common(data, None, None, false, encoded_data.len() / 8, aligned)?;

    // Set the value
    data.append_bits(&encoded_data);

    data.dump_encode();
    Ok(())
}

// Common function to encode a BOOLEAN Value
pub(crate) fn encode_bool_common(
    data: &mut PerCodecData,
    value: bool,
    _aligned: bool,
) -> Result<(), PerCodecError> {
    data.encode_bool(value);

    data.dump_encode();
    Ok(())
}

// Common function to encode an ENUMERATED Value
//
// X.691 §14. For a ROOT value (`extended == false`) `value` is the root
// enumeration index, encoded as a constrained whole number over `lb..=ub`
// (after the extension bit, if the type is extensible).
//
// For an EXTENSION ADDITION (`extended == true`) `value` is the index WITHIN the
// extension additions, numbered from 0 in declaration order -- not the
// root-continuing number the generated constants use. §14.3: the extension bit
// is set and that index is encoded as a normally small non-negative whole number
// (§11.6), exactly the layout `decode_enumerated_common` already read. Until
// #213 this branch returned `EncodeNotSupported`, so an extension value such as
// NGAP CauseRadioNetwork n26-interface-not-available could not be sent at all.
pub(crate) fn encode_enumerated_common(
    data: &mut PerCodecData,
    lb: Option<i128>,
    ub: Option<i128>,
    is_extensible: bool,
    value: i128,
    extended: bool,
    aligned: bool,
) -> Result<(), PerCodecError> {
    if extended {
        if !is_extensible {
            return Err(PerCodecError::new(
                PerCodecErrorCause::Generic,
                "An extension value was selected in an ENUMERATED that has no extension marker",
            ));
        }
        data.encode_bool(true);
        encode_normally_small_non_negative_whole_number_common(data, value, aligned)?;
        data.dump_encode();
        return Ok(());
    }

    if is_extensible {
        data.encode_bool(extended);
    }

    encode_integer_common(data, lb, ub, false, value, false, aligned)?;

    data.dump();

    Ok(())
}

// Common function to encode a bitstring
// Refer to Section 15.
pub(crate) fn encode_bitstring_common(
    data: &mut PerCodecData,
    lb: Option<i128>,
    ub: Option<i128>,
    is_extensible: bool,
    bit_string: &BitSlice<u8, Msb0>,
    extended: bool,
    aligned: bool,
) -> Result<(), PerCodecError> {
    if extended {
        return Err(PerCodecError::new(
            PerCodecErrorCause::EncodeNotSupported,
            "Encode of extended bitstring not yet implemented",
        ));
    }

    if is_extensible {
        data.encode_bool(extended);
    }

    let length = bit_string.len();
    // X.691 §16.11: content longer than 16 bits is octet-aligned in APER. Per
    // piece, so a fragmented string aligns each fragment (already aligned by its
    // header) exactly as an unfragmented one aligns its whole content.
    let emit = |data: &mut PerCodecData, start: usize, count: usize| {
        if count > 0 {
            if count > 16 && aligned {
                data.align();
            }
            data.append_bits(&bit_string[start..start + count]);
        }
        Ok(())
    };
    if length_fragments(ub) {
        check_length_bounds(lb, ub, length)?;
        encode_fragmented_common(data, length, aligned, emit)?;
    } else {
        encode_length_determinent_common(data, lb, ub, false, length, aligned)?;
        emit(data, 0, length)?;
    }

    // TODO: Not sure if 15.11 is handled correctly?
    data.dump_encode();

    Ok(())
}

// Common function to encode an OCTET STRING
pub(crate) fn encode_octet_string_common(
    data: &mut PerCodecData,
    lb: Option<i128>,
    ub: Option<i128>,
    is_extensible: bool,
    octet_string: &[u8],
    extended: bool,
    aligned: bool,
) -> Result<(), PerCodecError> {
    if extended {
        return Err(PerCodecError::new(
            PerCodecErrorCause::EncodeNotSupported,
            "Encode of extended octetstring not yet implemented",
        ));
    }

    if is_extensible {
        data.encode_bool(extended);
    }

    let length = octet_string.len();
    // X.691 §17.8: content longer than two octets is octet-aligned in APER.
    let emit = |data: &mut PerCodecData, start: usize, count: usize| {
        if count > 0 {
            if count > 2 && aligned {
                data.align();
            }
            data.append_bits(octet_string[start..start + count].view_bits());
        }
        Ok(())
    };
    if length_fragments(ub) {
        check_length_bounds(lb, ub, length)?;
        encode_fragmented_common(data, length, aligned, emit)?;
    } else {
        // A `ub` below 64K is a constrained length (§11.9.3.3): it never
        // fragments, so 16K..ub octets are legal here and were wrongly refused.
        encode_length_determinent_common(data, lb, ub, false, length, aligned)?;
        emit(data, 0, length)?;
    }

    data.dump_encode();
    Ok(())
}

// Encode a Length Determinent
pub(crate) fn encode_length_determinent_common(
    data: &mut PerCodecData,
    lb: Option<i128>,
    ub: Option<i128>,
    normally_small: bool,
    value: usize,
    aligned: bool,
) -> Result<(), PerCodecError> {
    if normally_small {
        encode_normally_small_length_determinent_common(data, value, aligned)?;
        data.dump_encode();

        return Ok(());
    }

    match ub {
        Some(ub) if ub < 65_536 => encode_constrained_whole_number_common(
            data,
            lb.unwrap_or(0),
            ub,
            value as i128,
            aligned,
        )?,
        _ => {
            if let Some(u) = ub {
                if value > u as usize {
                    return Err(PerCodecError::new(
                        PerCodecErrorCause::Generic,
                        format!(
                            "Cannot encode length determinent {} - greater than upper bound {}",
                            value, u,
                        ),
                    ));
                }
            }

            if let Some(l) = lb {
                if value < l as usize {
                    return Err(PerCodecError::new(
                        PerCodecErrorCause::Generic,
                        format!(
                            "Cannot encode length determinent {} - less than lower bound {}",
                            value, l,
                        ),
                    ));
                }
            }

            encode_indefinite_length_determinent_common(data, value, aligned)?
        }
    };

    data.dump_encode();

    Ok(())
}

// Common function to encode the length of a SIZE-constrained SEQUENCE OF / SET OF.
//
// X.691 §20.4-20.6 (2021: §20; SEQUENCE OF reuses the §11.9 length determinant).
// Two things the plain length determinant cannot do, and the reason this exists
// (#212):
//
//   - an EXTENSIBLE size constraint (`SIZE (lb..ub, ...)`) is preceded by one
//     bit saying whether the count lies in the root `lb..=ub`. Inside the root
//     the length is constrained as usual; outside it the length is encoded as a
//     semi-constrained (unbounded) length determinant (§20.4, §11.9.4.2).
//   - a NON-extensible constraint must REFUSE a count outside `lb..=ub`. The
//     derive used to discard this function's result, so an empty
//     `SIZE (1..n)` list or an over-long one encoded as a PDU with no length
//     field at all -- silently, and unreadable by any peer.
pub(crate) fn encode_sequence_of_length_common(
    data: &mut PerCodecData,
    lb: Option<i128>,
    ub: Option<i128>,
    is_extensible: bool,
    value: usize,
    aligned: bool,
) -> Result<(), PerCodecError> {
    let count = value as i128;
    let in_root = lb.is_none_or(|l| count >= l) && ub.is_none_or(|u| count <= u);

    if is_extensible {
        data.encode_bool(!in_root);
        if !in_root {
            return encode_length_determinent_common(data, None, None, false, value, aligned);
        }
    } else if !in_root {
        return Err(PerCodecError::new(
            PerCodecErrorCause::Generic,
            format!(
                "Cannot encode SEQUENCE OF with {} elements - outside SIZE ({}..{})",
                value,
                lb.map_or("MIN".to_string(), |l| l.to_string()),
                ub.map_or("MAX".to_string(), |u| u.to_string()),
            ),
        ));
    }

    encode_length_determinent_common(data, lb, ub, false, value, aligned)
}

// Common function to encode string value.
pub(crate) fn encode_string_common(
    data: &mut PerCodecData,
    lb: Option<i128>,
    ub: Option<i128>,
    is_extensible: bool,
    value: &String,
    extended: bool,
    aligned: bool,
) -> Result<(), PerCodecError> {
    if extended {
        return Err(PerCodecError::new(
            PerCodecErrorCause::EncodeNotSupported,
            "Encode of extended visible string not yet implemented",
        ));
    }

    if is_extensible {
        data.encode_bool(extended);
    }
    // Eight bits per character (the APER encoding of these types here), so a
    // character is an octet and the §30.5.7 fragment unit of 16K characters is
    // 16K octets.
    let bytes = value.as_bytes();
    let length = bytes.len();
    let emit = |data: &mut PerCodecData, start: usize, count: usize| {
        if count > 2 && aligned {
            data.align();
        }
        data.append_bits(bytes[start..start + count].view_bits());
        Ok(())
    };
    if length_fragments(ub) {
        check_length_bounds(lb, ub, length)?;
        encode_fragmented_common(data, length, aligned, emit)?;
    } else {
        encode_length_determinent_common(data, lb, ub, false, length, aligned)?;
        emit(data, 0, length)?;
    }

    data.dump_encode();
    Ok(())
}

// Refuse a length outside `lb..=ub` before writing anything (§11.9.4.2 only
// carries the length; the bound is the type's).
pub(crate) fn check_length_bounds(
    lb: Option<i128>,
    ub: Option<i128>,
    value: usize,
) -> Result<(), PerCodecError> {
    if let Some(u) = ub {
        if value as i128 > u {
            return Err(PerCodecError::new(
                PerCodecErrorCause::Generic,
                format!(
                    "Cannot encode length determinent {} - greater than upper bound {}",
                    value, u,
                ),
            ));
        }
    }
    if let Some(l) = lb {
        if (value as i128) < l {
            return Err(PerCodecError::new(
                PerCodecErrorCause::Generic,
                format!(
                    "Cannot encode length determinent {} - less than lower bound {}",
                    value, l,
                ),
            ));
        }
    }
    Ok(())
}

// Common function to encode a SEQUENCE OF / SET OF: its count AND its
// components, because once the count reaches 16K in an unconstrained (or
// `ub >= 64K`) size the components are interleaved with fragment headers
// (X.691 §20.6 via §11.9.3.8): 16K..64K components, then the next header.
// `emit(data, start, count)` encodes components `start..start + count`.
//
// For every count the plain length could already express, the bytes are
// exactly those of `encode_sequence_of_length_common` followed by the
// components.
pub(crate) fn encode_sequence_of_common<F>(
    data: &mut PerCodecData,
    lb: Option<i128>,
    ub: Option<i128>,
    is_extensible: bool,
    count: usize,
    aligned: bool,
    mut emit: F,
) -> Result<(), PerCodecError>
where
    F: FnMut(&mut PerCodecData, usize, usize) -> Result<(), PerCodecError>,
{
    let n = count as i128;
    let in_root = lb.is_none_or(|l| n >= l) && ub.is_none_or(|u| n <= u);

    if is_extensible {
        data.encode_bool(!in_root);
        if !in_root {
            // Outside the root the count is semi-constrained (§20.4).
            return encode_fragmented_common(data, count, aligned, emit);
        }
    } else if !in_root {
        return Err(PerCodecError::new(
            PerCodecErrorCause::Generic,
            format!(
                "Cannot encode SEQUENCE OF with {} elements - outside SIZE ({}..{})",
                count,
                lb.map_or("MIN".to_string(), |l| l.to_string()),
                ub.map_or("MAX".to_string(), |u| u.to_string()),
            ),
        ));
    }

    if length_fragments(ub) {
        encode_fragmented_common(data, count, aligned, emit)
    } else {
        encode_length_determinent_common(data, lb, ub, false, count, aligned)?;
        emit(data, 0, count)
    }
}

// Common function to encode an open type (X.691 §11.2): the complete
// encoding of `inner`, padded to a whole number of octets, framed by an
// unconstrained length in octets -- fragmented at 16K like any other.
//
// `align_outer` pads THIS buffer before each piece of content. That is right
// for APER, where the determinant has already aligned and the pad is a no-op,
// and wrong for UPER (see `append_open_type_unaligned`); callers pass `aligned`.
pub(crate) fn encode_open_type_common(
    data: &mut PerCodecData,
    inner: &mut PerCodecData,
    align_outer: bool,
    aligned: bool,
) -> Result<(), PerCodecError> {
    inner.align();
    let octets: Vec<u8> = inner.bits.as_raw_slice().to_vec();
    let total = inner.length_in_bytes();
    encode_fragmented_common(data, total, aligned, |data, start, count| {
        if align_outer {
            data.align();
        }
        data.append_bits(octets[start..start + count].view_bits());
        Ok(())
    })?;
    data.dump_encode();
    Ok(())
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::per::common::decode::decode_choice_idx_common;

    use crate::per::common::decode::decode_enumerated_common;

    /// #213: an ENUMERATED extension addition encodes (X.691 §14.3) and the
    /// shipped decoder reads back the same within-additions index, flagged as
    /// extended. 63/64 straddles the six-bit short form of the normally small
    /// number, where an off-by-one would hide.
    #[test]
    fn an_extension_enumerated_value_round_trips_for_both_codecs() {
        for idx in [0_i128, 1, 2, 62, 63, 64, 200] {
            for aligned in [true, false] {
                let mut data = if aligned {
                    PerCodecData::new_aper()
                } else {
                    PerCodecData::new_uper()
                };
                encode_enumerated_common(&mut data, Some(0), Some(44), true, idx, true, aligned)
                    .expect("an extension value must encode");
                let bytes = data.into_bytes();
                assert_eq!(bytes[0] & 0x80, 0x80, "the extension bit must be set");

                let mut read = if aligned {
                    PerCodecData::from_slice_aper(&bytes)
                } else {
                    PerCodecData::from_slice_uper(&bytes)
                };
                let (decoded, extended) =
                    decode_enumerated_common(&mut read, Some(0), Some(44), true, aligned)
                        .expect("decode");
                assert!(
                    extended,
                    "idx {} aligned={} must decode as extended",
                    idx, aligned
                );
                assert_eq!(decoded, idx, "idx {idx} aligned={aligned}");
            }
        }
    }

    /// A root value keeps its pre-#213 wire form: extension bit 0, then the
    /// constrained index. For lb 0 / ub 44 (range 45, six bits) value 44 is
    /// 0 + 101100, padded: 0b0101_1000.
    #[test]
    fn a_root_enumerated_value_keeps_its_wire_form() {
        let mut data = PerCodecData::new_aper();
        encode_enumerated_common(&mut data, Some(0), Some(44), true, 44, false, true)
            .expect("encode");
        assert_eq!(data.into_bytes(), vec![0b0101_1000]);
    }

    /// An extension value in a type without an extension marker is a caller
    /// error: there is no extension bit to set, so it would read back as a root
    /// value.
    #[test]
    fn an_extension_value_in_a_non_extensible_enumerated_is_rejected() {
        let mut data = PerCodecData::new_aper();
        let err = encode_enumerated_common(&mut data, Some(0), Some(3), false, 0, true, true)
            .expect_err("no extension marker");
        assert!(err.to_string().contains("no extension marker"), "{}", err);
    }

    /// The core X.691 §23.6 guarantee, asserted against the DECODER rather than
    /// against hand-computed bytes: for every extension index the decoder must
    /// read back exactly what the encoder wrote, and must report it as extended.
    ///
    /// Driving it off the decoder is deliberate. The decoder was already correct
    /// and shipped, so it is the authority on the bit layout; a test against bytes
    /// I derived myself would only prove the encoder agrees with my reading of the
    /// spec. Both codecs are exercised because the length/index encodings differ
    /// between aligned and unaligned PER.
    #[test]
    fn an_extended_choice_index_round_trips_for_both_codecs() {
        // 63/64 is the boundary of the six-bit short form (§11.6.1 vs §11.6.2),
        // which is where an off-by-one would hide.
        for idx in [0_i128, 1, 5, 62, 63, 64, 65, 200] {
            for aligned in [false, true] {
                let mut data = if aligned {
                    PerCodecData::new_aper()
                } else {
                    PerCodecData::new_uper()
                };
                encode_choice_idx_common(&mut data, 0, 3, true, idx, true, aligned)
                    .expect("an extended choice index must encode");

                let bytes = data.into_bytes();
                let mut read = if aligned {
                    PerCodecData::from_slice_aper(&bytes)
                } else {
                    PerCodecData::from_slice_uper(&bytes)
                };
                let (decoded, extended) = decode_choice_idx_common(&mut read, 0, 3, true, aligned)
                    .expect("and decode back");

                assert!(
                    extended,
                    "idx {idx} (aligned={aligned}) must decode as an extension arm"
                );
                assert_eq!(
                    decoded, idx,
                    "extension index {idx} (aligned={aligned}) did not survive the round trip"
                );
            }
        }
    }

    /// A root index must still round trip as a ROOT index, so adding the extension
    /// path did not disturb the arm that already worked. Asserting `!extended` is
    /// the point: an encoder that set the extension bit unconditionally would pass
    /// an index-only comparison.
    #[test]
    fn a_root_choice_index_still_round_trips_as_a_root_index() {
        for idx in [0_i128, 1, 2, 3] {
            let mut data = PerCodecData::new_uper();
            encode_choice_idx_common(&mut data, 0, 3, true, idx, false, false).expect("encode");

            let bytes = data.into_bytes();
            let mut read = PerCodecData::from_slice_uper(&bytes);
            let (decoded, extended) =
                decode_choice_idx_common(&mut read, 0, 3, true, false).expect("decode");

            assert!(
                !extended,
                "root index {idx} must not be flagged as extended"
            );
            assert_eq!(decoded, idx);
        }
    }

    /// Selecting an extension arm in a CHOICE that has no extension marker is a
    /// caller error, not something to encode as if it were a root index: there is
    /// no extension bit in the encoding, so the index would be read back as a root
    /// index and silently resolve to the wrong variant.
    #[test]
    fn an_extension_arm_in_a_non_extensible_choice_is_rejected() {
        let mut data = PerCodecData::new_uper();
        let err = encode_choice_idx_common(&mut data, 0, 3, false, 0, true, false)
            .expect_err("an extension arm needs an extension marker");
        assert!(
            err.to_string().contains("no extension marker"),
            "unexpected error: {err}"
        );
    }

    /// The §11.6 normally-small encoder must NOT be confused with the length
    /// determinant next door, which encodes `n - 1`. Index 0 is the case that
    /// exposes it: the length helper would underflow, and every index would be off
    /// by one. Pinned as bytes because this is an absolute claim about a layout,
    /// not a relative one: a leading `0` bit then six bits of value, seven bits
    /// total, which `into_bytes` pads with one trailing zero.
    #[test]
    fn a_normally_small_number_encodes_the_value_not_the_value_minus_one() {
        let mut data = PerCodecData::new_uper();
        encode_normally_small_non_negative_whole_number_common(&mut data, 0, false)
            .expect("encode");
        // 0b0_000000 + one pad bit.
        assert_eq!(data.into_bytes(), vec![0b0000_0000]);

        let mut data = PerCodecData::new_uper();
        encode_normally_small_non_negative_whole_number_common(&mut data, 1, false)
            .expect("encode");
        // 0b0_000001 + one pad bit = 0x02. A length determinant would have written
        // `1 - 1 == 0` here, i.e. the same bytes as the line above -- which is the
        // confusion this test exists to catch.
        assert_eq!(data.into_bytes(), vec![0b0000_0010]);

        let mut data = PerCodecData::new_uper();
        encode_normally_small_non_negative_whole_number_common(&mut data, 63, false)
            .expect("encode");
        // 0b0_111111 + one pad bit: the top of the six-bit short form.
        assert_eq!(data.into_bytes(), vec![0b0111_1110]);
    }

    use crate::per::common::decode::decode_sequence_of_length_common;

    fn seq_of_len_round_trip(
        lb: Option<i128>,
        ub: Option<i128>,
        ext: bool,
        n: usize,
        aligned: bool,
    ) -> Result<(Vec<u8>, usize), PerCodecError> {
        let mut data = if aligned {
            PerCodecData::new_aper()
        } else {
            PerCodecData::new_uper()
        };
        encode_sequence_of_length_common(&mut data, lb, ub, ext, n, aligned)?;
        let bytes = data.into_bytes();
        let mut read = if aligned {
            PerCodecData::from_slice_aper(&bytes)
        } else {
            PerCodecData::from_slice_uper(&bytes)
        };
        let decoded = decode_sequence_of_length_common(&mut read, lb, ub, ext, aligned)?;
        Ok((bytes, decoded))
    }

    /// #212: a non-extensible SIZE refuses a count below the lower bound and
    /// above the upper bound, and still accepts both exact bounds. The bounds
    /// are NGAP's PLMNSupportList, SIZE (1..maxnoofPLMNs = 12).
    #[test]
    fn a_sequence_of_count_outside_a_fixed_size_is_refused() {
        for aligned in [true, false] {
            for n in [0usize, 13, 1000] {
                let err = seq_of_len_round_trip(Some(1), Some(12), false, n, aligned)
                    .expect_err("an out-of-SIZE count must not encode");
                assert!(
                    err.to_string().contains("outside SIZE (1..12)"),
                    "n={} aligned={}: unexpected error {}",
                    n,
                    aligned,
                    err
                );
            }
            for n in [1usize, 12] {
                let (_, decoded) = seq_of_len_round_trip(Some(1), Some(12), false, n, aligned)
                    .expect("an exact bound must encode");
                assert_eq!(decoded, n, "n={n} aligned={aligned}");
            }
        }
    }

    /// The bound check also covers the large-ub path (ub >= 64K), which goes
    /// through the unconstrained length determinant rather than a constrained
    /// whole number.
    #[test]
    fn a_large_upper_bound_is_still_a_bound() {
        let err = seq_of_len_round_trip(Some(1), Some(70_000), false, 0, true)
            .expect_err("below the lower bound");
        assert!(
            err.to_string().contains("outside SIZE (1..70000)"),
            "{}",
            err
        );
        // (Counts >= 16K need fragmentation, which this codec does not implement.)
        let (_, decoded) =
            seq_of_len_round_trip(Some(1), Some(70_000), false, 300, true).expect("in range");
        assert_eq!(decoded, 300);
    }

    /// X.691 §20.4: an extensible SIZE leads with one bit -- 0 for a count in the
    /// root, 1 for a count outside it -- and an out-of-root count round trips
    /// instead of being refused. Pinned on the first bit as well as the round
    /// trip, because a codec that dropped the bit on both sides would still
    /// round trip.
    #[test]
    fn an_extensible_size_flags_and_carries_an_out_of_root_count() {
        for aligned in [true, false] {
            let (bytes, decoded) =
                seq_of_len_round_trip(Some(1), Some(4), true, 3, aligned).expect("in root");
            assert_eq!(decoded, 3);
            assert_eq!(
                bytes[0] & 0x80,
                0,
                "in-root count must clear the extension bit"
            );

            for n in [0usize, 5, 200] {
                let (bytes, decoded) =
                    seq_of_len_round_trip(Some(1), Some(4), true, n, aligned).expect("ext");
                assert_eq!(decoded, n, "n={n} aligned={aligned}");
                assert_eq!(
                    bytes[0] & 0x80,
                    0x80,
                    "out-of-root count {n} must set the extension bit"
                );
            }
        }
    }

    /// The wire form of an in-root, non-extensible length is unchanged by #212:
    /// SIZE (1..12) with 3 elements is the constrained whole number 2 in 4 bits.
    #[test]
    fn a_fixed_size_length_keeps_its_wire_form() {
        let (bytes, _) = seq_of_len_round_trip(Some(1), Some(12), false, 3, true).expect("ok");
        assert_eq!(bytes, vec![0b0010_0000]);
    }

    #[test]
    fn test_encode_real_base_10_nr3() {
        let expected_data = &[
            0x0A,
            super::super::BASE_10_NR3,
            b'1',
            b'.',
            b'5',
            b'6',
            b'2',
            b'5',
            b'e',
            b'-',
            b'1',
        ];
        let value = 0.15625f64;
        let mut data = PerCodecData::new_aper();
        let result = encode_real_common(&mut data, value, true);
        assert!(result.is_ok(), "{:#?}", result.err().unwrap());
        assert_eq!(data.into_bytes(), expected_data);
    }
}
