//! ASN.1 Aper Encoder module.

use bitvec::prelude::*;

use crate::per::PerCodecData;

#[allow(unused)]
use crate::per::common::encode::*;

use crate::PerCodecError;

/// Encode a Choice Index
///
/// During Encoding a 'CHOICE' Type to help decoding, the 'CHOICE' Index is encoded first, followed
/// by the actual encoding of the 'CHOICE' variant.
pub fn encode_choice_idx(
    data: &mut PerCodecData,
    lb: i128,
    ub: i128,
    is_extensible: bool,
    idx: i128,
    extended: bool,
) -> Result<(), PerCodecError> {
    log::trace!(
        "encode_choice_idx: lb: {}, ub: {}, is_extensible: {}, idx: {}, extended: {}",
        lb,
        ub,
        is_extensible,
        idx,
        extended
    );

    encode_choice_idx_common(data, lb, ub, is_extensible, idx, extended, true)
}

/// Encode sequence header
pub fn encode_sequence_header(
    data: &mut PerCodecData,
    is_extensible: bool,
    optionals: &BitSlice<u8, Msb0>,
    extended: bool,
) -> Result<(), PerCodecError> {
    log::trace!(
        "encode_sequence_header: is_extensible: {}, optional_fields: {:?}, extended: {}",
        is_extensible,
        optionals,
        extended
    );

    encode_sequence_header_common(data, is_extensible, optionals, extended, true)
}

/// Encode an INTEGER
///
/// This API is also used by other `encode` functions to encode an integer value.
///
/// Note: The maximum (and minimum) value to be decoded is limited to an `i128` value. For the
/// protocols that are currently supported this limit is acceptable.
pub fn encode_integer(
    data: &mut PerCodecData,
    lb: Option<i128>,
    ub: Option<i128>,
    is_extensible: bool,
    value: i128,
    extended: bool,
) -> Result<(), PerCodecError> {
    log::trace!(
        "encode_integer: lb: {:?}, ub: {:?}, is_extensible: {}, value: {}, extended: {}",
        lb,
        ub,
        is_extensible,
        value,
        extended
    );

    encode_integer_common(data, lb, ub, is_extensible, value, extended, true)
}

/// Encode a BOOLEAN Value
///
/// Encodes a boolean value into the passed `PerCodecData` structure.
pub fn encode_bool(data: &mut PerCodecData, value: bool) -> Result<(), PerCodecError> {
    log::trace!("encode_bool: {}", value);

    encode_bool_common(data, value, true)
}

/// Encode a REAL Value
///
/// Encodes a boolean value into the passed `PerCodecData` structure.
pub fn encode_real(data: &mut PerCodecData, value: f64) -> Result<(), PerCodecError> {
    log::trace!("encode_real: {}", value);
    encode_real_common(data, value, true)
}

/// Encode an ENUMERATED Value
pub fn encode_enumerated(
    data: &mut PerCodecData,
    lb: Option<i128>,
    ub: Option<i128>,
    is_extensible: bool,
    value: i128,
    extended: bool,
) -> Result<(), PerCodecError> {
    log::trace!(
        "encode_enumerated: lb: {:?}, ub: {:?}, is_extensible: {}, value: {}, extended: {}",
        lb,
        ub,
        is_extensible,
        value,
        extended
    );

    encode_enumerated_common(data, lb, ub, is_extensible, value, extended, true)
}

/// Encode a Bit String
pub fn encode_bitstring(
    data: &mut PerCodecData,
    lb: Option<i128>,
    ub: Option<i128>,
    is_extensible: bool,
    bit_string: &BitSlice<u8, Msb0>,
    extended: bool,
) -> Result<(), PerCodecError> {
    log::trace!(
        "encode_bitstring: lb: {:?}, ub: {:?}, is_extensible: {}, bits: {:?}, extended: {}",
        lb,
        ub,
        is_extensible,
        bit_string,
        extended
    );

    encode_bitstring_common(data, lb, ub, is_extensible, bit_string, extended, true)
}

/// Encode an OCTET STRING
pub fn encode_octetstring(
    data: &mut PerCodecData,
    lb: Option<i128>,
    ub: Option<i128>,
    is_extensible: bool,
    octet_string: &Vec<u8>,
    extended: bool,
) -> Result<(), PerCodecError> {
    log::trace!(
        "encode_octetstring: lb: {:?}, ub: {:?}, is_extensible: {}, bytes: {:?}, extended: {}",
        lb,
        ub,
        is_extensible,
        octet_string,
        extended
    );

    encode_octet_string_common(data, lb, ub, is_extensible, octet_string, extended, true)
}

// Encode a Length Determinent
pub fn encode_length_determinent(
    data: &mut PerCodecData,
    lb: Option<i128>,
    ub: Option<i128>,
    normally_small: bool,
    value: usize,
) -> Result<(), PerCodecError> {
    log::trace!(
        "encode_length_determinent: lb: {:?}, ub: {:?}, normally_small: {}, value: {}",
        lb,
        ub,
        normally_small,
        value
    );

    encode_length_determinent_common(data, lb, ub, normally_small, value, true)
}

/// Encode the length of a SIZE-constrained SEQUENCE OF (X.691 §20.4).
///
/// Unlike [`encode_length_determinent`], this honours an extensible size
/// constraint and refuses a count outside a non-extensible one.
pub fn encode_sequence_of_length(
    data: &mut PerCodecData,
    lb: Option<i128>,
    ub: Option<i128>,
    is_extensible: bool,
    value: usize,
) -> Result<(), PerCodecError> {
    log::trace!(
        "encode_sequence_of_length: lb: {:?}, ub: {:?}, is_extensible: {}, value: {}",
        lb,
        ub,
        is_extensible,
        value
    );

    encode_sequence_of_length_common(data, lb, ub, is_extensible, value, true)
}

/// Encode a SEQUENCE OF / SET OF: its count and, through `emit`, its
/// components (X.691 §20.6).
///
/// `emit(data, start, count)` must encode components `start..start + count`.
/// It is called once per piece, so a count of 16K or more in an unconstrained
/// size is fragmented (§11.9.3.8) with the headers interleaved. Prefer this to
/// [`encode_sequence_of_length`], which can only write a single length.
pub fn encode_sequence_of<F>(
    data: &mut PerCodecData,
    lb: Option<i128>,
    ub: Option<i128>,
    is_extensible: bool,
    count: usize,
    emit: F,
) -> Result<(), PerCodecError>
where
    F: FnMut(&mut PerCodecData, usize, usize) -> Result<(), PerCodecError>,
{
    log::trace!(
        "encode_sequence_of: lb: {:?}, ub: {:?}, is_extensible: {}, count: {}",
        lb,
        ub,
        is_extensible,
        count
    );

    encode_sequence_of_common(data, lb, ub, is_extensible, count, true, emit)
}

/// Encode an open type (X.691 §11.2): `inner`'s complete encoding, padded to
/// whole octets and framed by a length in octets that fragments at 16K
/// (§11.9.3.8). Prefer this to [`encode_length_determinent`] plus an append,
/// which cannot frame 16K octets or more.
pub fn encode_open_type(
    data: &mut PerCodecData,
    inner: &mut PerCodecData,
) -> Result<(), PerCodecError> {
    encode_open_type_common(data, inner, true, true)
}

/// Encode a VisibleString CharacterString Type.
pub fn encode_visible_string(
    data: &mut PerCodecData,
    lb: Option<i128>,
    ub: Option<i128>,
    is_extensible: bool,
    value: &String,
    extended: bool,
) -> Result<(), PerCodecError> {
    log::trace!(
        "encode_visible_string: lb: {:?}, ub: {:?}, is_extensible: {}, value: {}, extended: {}",
        lb,
        ub,
        is_extensible,
        value,
        extended
    );

    encode_string_common(data, lb, ub, is_extensible, value, extended, true)
}

/// Encode a PrintableString CharacterString Type.
pub fn encode_printable_string(
    data: &mut PerCodecData,
    lb: Option<i128>,
    ub: Option<i128>,
    is_extensible: bool,
    value: &String,
    extended: bool,
) -> Result<(), PerCodecError> {
    log::trace!(
        "encode_printable_string: lb: {:?}, ub: {:?}, is_extensible: {}, value: {}, extended: {}",
        lb,
        ub,
        is_extensible,
        value,
        extended
    );

    encode_string_common(data, lb, ub, is_extensible, value, extended, true)
}

/// Encode a UTF8String CharacterString Type.
pub fn encode_utf8_string(
    data: &mut PerCodecData,
    lb: Option<i128>,
    ub: Option<i128>,
    is_extensible: bool,
    value: &String,
    extended: bool,
) -> Result<(), PerCodecError> {
    log::trace!(
        "encode_utf8_string: lb: {:?}, ub: {:?}, is_extensible: {}, value: {}, extended: {}",
        lb,
        ub,
        is_extensible,
        value,
        extended
    );

    // X.691 §30.6 / §10.9.3.3: UTF8String is not a known-multiplier type, so
    // its size constraint and extensibility are not PER-visible: no extension
    // bit, an unconstrained length in octets, octet-aligned content.
    let _ = (lb, ub, is_extensible, extended);
    encode_string_common(data, None, None, false, value, false, true)
}

/// Encode a UTCTime CharacterString Type.
pub fn encode_utc_time(
    data: &mut PerCodecData,
    lb: Option<i128>,
    ub: Option<i128>,
    is_extensible: bool,
    value: &String,
    extended: bool,
) -> Result<(), PerCodecError> {
    log::trace!(
        "encode_utc_time: lb: {:?}, ub: {:?}, is_extensible: {}, value: {}, extended: {}",
        lb,
        ub,
        is_extensible,
        value,
        extended
    );

    encode_string_common(data, lb, ub, is_extensible, value, extended, true)
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn encode_bool_always_success() {
        let mut data = PerCodecData::new_aper();

        let result = encode_bool(&mut data, true);
        assert!(result.is_ok());
        assert_eq!(data.bits.len(), 1);
        assert_eq!(data.bits[0], true);
    }

    #[test]
    fn int_too_small() {
        assert!(encode_integer(
            &mut PerCodecData::new_aper(),
            Some(1),
            None,
            false,
            0,
            false
        )
        .is_err());
    }

    #[test]
    fn int_too_big() {
        assert!(encode_integer(
            &mut PerCodecData::new_aper(),
            Some(-1),
            Some(0),
            false,
            1,
            false
        )
        .is_err());
    }

    #[test]
    fn octetstring_too_small() {
        assert!(encode_octetstring(
            &mut PerCodecData::new_aper(),
            Some(2),
            None,
            false,
            &vec![0],
            false
        )
        .is_err());
    }
    #[test]
    fn octetstring_too_big() {
        assert!(encode_octetstring(
            &mut PerCodecData::new_aper(),
            None,
            Some(1),
            false,
            &vec![0, 0],
            false
        )
        .is_err());
    }

    #[test]
    fn string_too_small() {
        assert!(encode_visible_string(
            &mut PerCodecData::new_aper(),
            Some(2),
            None,
            false,
            &"a".to_string(),
            false
        )
        .is_err());
    }

    #[test]
    fn string_too_big() {
        assert!(encode_visible_string(
            &mut PerCodecData::new_aper(),
            None,
            Some(1),
            false,
            &"aa".to_string(),
            false
        )
        .is_err());
    }

    #[test]
    fn length_too_small() {
        assert!(
            encode_length_determinent(&mut PerCodecData::new_aper(), Some(2), None, false, 1,)
                .is_err()
        );
    }
    #[test]
    fn length_too_big() {
        assert!(
            encode_length_determinent(&mut PerCodecData::new_aper(), None, Some(1), false, 2,)
                .is_err()
        );
    }

    #[test]
    fn big_length_too_big() {
        assert!(encode_length_determinent(
            &mut PerCodecData::new_aper(),
            None,
            Some(65536),
            false,
            65537,
        )
        .is_err());
    }

    #[test]
    fn bitstring_too_small() {
        assert!(encode_bitstring(
            &mut PerCodecData::new_aper(),
            Some(2),
            None,
            false,
            bits![u8, Msb0; 0],
            false
        )
        .is_err());
    }

    #[test]
    fn bitstring_too_big() {
        assert!(encode_bitstring(
            &mut PerCodecData::new_aper(),
            None,
            Some(1),
            false,
            bits![u8, Msb0; 0, 0],
            false
        )
        .is_err());
    }
}
