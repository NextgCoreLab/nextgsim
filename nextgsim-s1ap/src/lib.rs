//! S1AP (S1 Application Protocol, TS 36.413) for the nextgsim eNB.
//!
//! The codec is generated at build time from the TS 36.413 V19.2.0 ASN.1
//! (`tools/s1ap-19.2.0.asn`, see `tools/S1AP-PROVENANCE.md`). It is kept
//! independent of nextgcore's S1AP implementation on purpose: nextgsim is the
//! peer that tests nextgcore, and an independent codec is what catches a
//! misreading on either side (nextgsim #45, #205).

pub mod codec;

/// `ExpectedActivityPeriod ::= INTEGER (1..30|40|50|60|80|100|120|150|180|181,...)`
/// (TS 36.413 §9.2.1.96): the PER root is the union 1..181, so a root value
/// above 30 is a root encoding (0 bit + 8-bit offset), not an extension. The
/// compiler once bounded it at 30. Bytes derived by hand from X.691 §11.5.7.2.
#[cfg(test)]
mod constraint_union_tests {
    use crate::codec::generated::{ExpectedActivityPeriod, ExpectedIdlePeriod};
    use asn1_codecs::aper::AperCodec;
    use asn1_codecs::PerCodecData;

    fn enc<T: AperCodec>(v: &T) -> Vec<u8> {
        let mut d = PerCodecData::new_aper();
        v.aper_encode(&mut d).expect("encode");
        d.into_bytes()
    }

    #[test]
    fn root_values_above_30_are_root_encodings() {
        assert_eq!(enc(&ExpectedActivityPeriod(30)), [0x0E, 0x80]);
        assert_eq!(enc(&ExpectedActivityPeriod(40)), [0x13, 0x80]);
        assert_eq!(enc(&ExpectedActivityPeriod(181)), [0x5A, 0x00]);
        assert_eq!(enc(&ExpectedIdlePeriod(181)), [0x5A, 0x00]);
    }
}
