//! Extensible INTEGERs past their root, through the GENERATED NGAP and NRPPa
//! types (#231).
//!
//! X.691 §12.1: a value outside the root of an extensible INTEGER is a `1` bit
//! followed by an unconstrained whole number (a length determinant, then the
//! minimal two's-complement octets); a root value is a `0` bit followed by the
//! root encoding. Before #231 the derive always encoded as if the value were in
//! the root (so 4096 of `0..4095, ...` was refused), sized the Rust type from
//! the root alone (so `MaximumDataBurstVolume` was `u16` and 65536..2000000 could
//! not exist), and decoded with an `as` cast (so an out-of-type value wrapped
//! silently). Every expected byte string is derived by hand from X.691 beside it.

use asn1_codecs::aper::AperCodec;
use asn1_codecs::PerCodecData;

use crate::codec::generated::{AveragingWindow, ExtendedPacketDelayBudget, MaximumDataBurstVolume};
use crate::codec::nrppa_generated::UE_Measurement_ID;

fn enc<T: AperCodec>(v: &T) -> Vec<u8> {
    let mut d = PerCodecData::new_aper();
    v.aper_encode(&mut d).expect("encode");
    d.into_bytes()
}

fn dec<T: AperCodec<Output = T>>(bytes: &[u8]) -> Result<T, asn1_codecs::PerCodecError> {
    let mut d = PerCodecData::from_slice_aper(bytes);
    T::aper_decode(&mut d)
}

/// `MaximumDataBurstVolume ::= INTEGER (0..4095, ..., 4096..2000000)`.
#[test]
fn maximum_data_burst_volume_carries_its_extension_range() {
    // Root, unchanged: ext 0 + 7 pad = 00, then range 4096 is the two-octet
    // aligned case (§11.5.7.3).
    assert_eq!(enc(&MaximumDataBurstVolume(0)), [0x00, 0x00, 0x00]);
    assert_eq!(enc(&MaximumDataBurstVolume(4095)), [0x00, 0x0F, 0xFF]);
    // Extension: 1 + 7 pad = 80, length, two's complement.
    // 4096 = 0x1000 -> 02 10 00.  2000000 = 0x1E8480 -> 03 1E 84 80.
    assert_eq!(enc(&MaximumDataBurstVolume(4096)), [0x80, 0x02, 0x10, 0x00]);
    assert_eq!(
        enc(&MaximumDataBurstVolume(2_000_000)),
        [0x80, 0x03, 0x1E, 0x84, 0x80]
    );
    for v in [0, 4095, 4096, 65_535, 65_536, 2_000_000] {
        let bytes = enc(&MaximumDataBurstVolume(v));
        assert_eq!(
            dec::<MaximumDataBurstVolume>(&bytes).unwrap(),
            MaximumDataBurstVolume(v)
        );
    }
}

/// `ExtendedPacketDelayBudget ::= INTEGER (1..65535, ..., 65536..109999)`.
#[test]
fn extended_packet_delay_budget_carries_its_extension_range() {
    // Root: offset from lb 1; range 65535 is two-octet aligned.
    // 65535 -> offset 65534 = FF FE.
    assert_eq!(enc(&ExtendedPacketDelayBudget(1)), [0x00, 0x00, 0x00]);
    assert_eq!(enc(&ExtendedPacketDelayBudget(65_535)), [0x00, 0xFF, 0xFE]);
    // 109999 = 0x01ADAF -> 03 01 AD AF.
    assert_eq!(
        enc(&ExtendedPacketDelayBudget(109_999)),
        [0x80, 0x03, 0x01, 0xAD, 0xAF]
    );
    assert_eq!(
        dec::<ExtendedPacketDelayBudget>(&[0x80, 0x03, 0x01, 0xAD, 0xAF]).unwrap(),
        ExtendedPacketDelayBudget(109_999)
    );
}

/// NRPPa `UE-Measurement-ID ::= INTEGER (1..15, ..., 16..256)` (TS 38.455).
#[test]
fn ue_measurement_id_carries_16_to_256() {
    // Root: 15 -> offset 14 in 4 bits (range 15, a bit-field, not aligned):
    // ext 0 + 1110 + pad = 0111 0000 -> 70.
    assert_eq!(enc(&UE_Measurement_ID(15)), [0x70]);
    // Extension: 16 = 0x10 -> 01 10. 256 = 0x0100 -> 02 01 00.
    assert_eq!(enc(&UE_Measurement_ID(16)), [0x80, 0x01, 0x10]);
    assert_eq!(enc(&UE_Measurement_ID(256)), [0x80, 0x02, 0x01, 0x00]);
    for v in [1, 15, 16, 255, 256] {
        assert_eq!(
            dec::<UE_Measurement_ID>(&enc(&UE_Measurement_ID(v))).unwrap(),
            UE_Measurement_ID(v)
        );
    }
}

/// A bare `...` has no PER-visible additions, so the type keeps its root width
/// (`AveragingWindow ::= INTEGER (0..4095, ...)` stays `u16`). A value that fits
/// the type still takes the extension encoding; one that does not is REFUSED on
/// decode — before #231 an `as` cast wrapped 65536 silently to 0.
#[test]
fn a_root_sized_type_refuses_an_extension_value_it_cannot_hold() {
    assert_eq!(enc(&AveragingWindow(4096)), [0x80, 0x02, 0x10, 0x00]);
    assert_eq!(
        dec::<AveragingWindow>(&[0x80, 0x02, 0x10, 0x00]).unwrap(),
        AveragingWindow(4096)
    );
    // 65536 = 0x010000 -> 03 01 00 00.
    assert!(dec::<AveragingWindow>(&[0x80, 0x03, 0x01, 0x00, 0x00]).is_err());
}
