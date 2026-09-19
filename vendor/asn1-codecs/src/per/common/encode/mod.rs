//! ASN.1 PER Encoding common functions

use bitvec::prelude::*;

use crate::per::{PerCodecData, PerCodecError, PerCodecErrorCause};

mod encode_internal;

#[allow(unused)]
use encode_internal::*;

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
        return Err(PerCodecError::new(
            PerCodecErrorCause::EncodeNotSupported,
            "Encode of extended enumerated not yet implemented",
        ));
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
    if length >= 16384 {
        return Err(PerCodecError::new(
            PerCodecErrorCause::EncodeNotSupported,
            "Encode of fragmented bitstring not yet implemented",
        ));
    }

    encode_length_determinent_common(data, lb, ub, false, length, aligned)?;
    if length > 0 {
        if length > 16 && aligned {
            data.align();
        }
        data.append_bits(bit_string);
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
    if length >= 16384 {
        return Err(PerCodecError::new(
            PerCodecErrorCause::EncodeNotSupported,
            "Encode of fragmented octetstring not yet implemented",
        ));
    }

    encode_length_determinent_common(data, lb, ub, false, length, aligned)?;

    if length > 0 {
        if length > 2 && aligned {
            data.align();
        }
        data.append_bits(octet_string.view_bits());
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
    encode_length_determinent_common(data, lb, ub, false, value.len(), aligned)?;
    if value.len() > 2 && aligned {
        data.align();
    }
    data.append_bits(value.as_bits());

    data.dump_encode();
    Ok(())
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::per::common::decode::decode_choice_idx_common;

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
