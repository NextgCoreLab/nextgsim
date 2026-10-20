//! X.691 §11.9.3.8 length fragmentation, end to end through a real NGAP PDU.
//!
//! A DOWNLINK NAS TRANSPORT whose NAS-PDU is 16K octets or more is the case
//! that matters in practice: the NAS-PDU OCTET STRING, the IE's open type and the
//! initiatingMessage's open type each cross the 16K boundary at slightly
//! different NAS lengths, so one message exercises all three fragmenting sites.
//!
//! The expected bytes are NOT produced by the codec under test. They are built
//! here from X.691 by hand: every fixed octet below carries its derivation, and
//! the fragmentation is done by [`frag`], a deliberately naive transcription of
//! §11.9.3.8 that shares no code with the codec. Before the fix the codec could
//! not encode any of these PDUs ("Length determinent >= 16384 not implemented"),
//! and decoding the hand-built bytes read the 16K fragment header of the
//! initiatingMessage open type as if it were content.

use crate::procedures::nas_transport::{
    decode_downlink_nas_transport, encode_downlink_nas_transport, DownlinkNasTransportParams,
};

/// X.691 §11.9.3.8, aligned and unaligned alike when the field starts on an
/// octet boundary (which every open type and every NGAP OCTET STRING here
/// does): emit 64K/48K/32K/16K fragments, largest first, each headed by one
/// octet `0b11` followed by the six-bit multiplier m (1..=4); then the
/// remainder `n mod 16K` with an ordinary determinant -- one octet below 128,
/// two octets `0b10` + 14 bits below 16K -- which is a single `0x00` when the
/// remainder is zero.
fn frag(content: &[u8]) -> Vec<u8> {
    const K16: usize = 16384;
    let mut out = Vec::new();
    let mut rest = content;
    while rest.len() >= K16 {
        let m = (rest.len() / K16).min(4);
        out.push(0xC0 | m as u8);
        out.extend_from_slice(&rest[..m * K16]);
        rest = &rest[m * K16..];
    }
    let n = rest.len();
    if n < 128 {
        out.push(n as u8);
    } else {
        out.push(0x80 | (n >> 8) as u8);
        out.push((n & 0xFF) as u8);
    }
    out.extend_from_slice(rest);
    out
}

/// The DOWNLINK NAS TRANSPORT, built by hand from TS 38.413 and X.691.
fn hand_built(amf: u8, ran: u8, nas: &[u8]) -> Vec<u8> {
    // IE AMF-UE-NGAP-ID: id 10 (16 bits, constrained 0..65535, aligned),
    // criticality reject (2 bits, then padding: the open-type length aligns),
    // value INTEGER (0..2^40-1) = 1..255: range > 64K, so a length in octets
    // constrained to 1..5 (3 bits, value len-1 = 0), align, one value octet.
    let ie_amf = [0x00, 0x0A, 0x00, 0x02, 0x00, amf];
    // IE RAN-UE-NGAP-ID: id 85, reject, INTEGER (0..2^32-1): length 1..4 in
    // 2 bits (0), align, one value octet.
    let ie_ran = [0x00, 0x55, 0x00, 0x02, 0x00, ran];
    // IE NAS-PDU: id 38, reject, then the open type whose content is the
    // unconstrained OCTET STRING: its own (fragmented) length and octets.
    let mut ie_nas = vec![0x00, 0x26, 0x00];
    ie_nas.extend(frag(&frag(nas)));

    // DownlinkNASTransport ::= SEQUENCE { protocolIEs, ... }: extension bit 0,
    // then ProtocolIE-Container SIZE (0..65535): a 16-bit aligned count, so
    // the 7 padding bits complete octet 0, then 00 03.
    let mut value = vec![0x00, 0x00, 0x03];
    value.extend_from_slice(&ie_amf);
    value.extend_from_slice(&ie_ran);
    value.extend(ie_nas);

    // NGAP-PDU CHOICE initiatingMessage: extension bit 0 + index 0 of 3 (2
    // bits), padded; procedureCode 4 (id-DownlinkNASTransport, 0..255 = one
    // aligned octet); criticality ignore (2 bits 01, padded); open-type value.
    let mut pdu = vec![0x00, 0x04, 0x40];
    pdu.extend(frag(&value));
    pdu
}

fn params(nas: Vec<u8>) -> DownlinkNasTransportParams {
    DownlinkNasTransportParams {
        amf_ue_ngap_id: 7,
        ran_ue_ngap_id: 9,
        nas_pdu: nas,
        old_amf: None,
        ran_paging_priority: None,
        index_to_rfsp: None,
        ue_ambr: None,
        allowed_nssai: None,
        ue_radio_capability: None,
        ue_capability_info_request: false,
        ue_radio_capability_id: None,
    }
}

/// NAS lengths straddling every 16K boundary each layer has: the NAS OCTET
/// STRING itself, its IE open type (NAS + 2 or 3 length octets) and the
/// message open type (that + 21 octets of header and other IEs).
const NAS_LENGTHS: &[usize] = &[
    0,
    1,
    127,
    128,
    16_350,
    16_358,
    16_359,
    16_360,
    16_380,
    16_381,
    16_383,
    16_384,
    16_385,
    16_384 + 109,
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

fn nas_of(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i * 7 + 3) as u8).collect()
}

#[test]
fn a_large_downlink_nas_transport_encodes_to_the_x691_fragmented_form() {
    for &len in NAS_LENGTHS {
        let nas = nas_of(len);
        let encoded = encode_downlink_nas_transport(&params(nas.clone()))
            .unwrap_or_else(|e| panic!("NAS length {len} must encode: {e:?}"));
        let expected = hand_built(7, 9, &nas);
        assert!(
            encoded == expected,
            "NAS length {len}: encoding differs from the X.691 hand derivation \
             (got {} octets, expected {}; first difference at octet {:?})",
            encoded.len(),
            expected.len(),
            encoded.iter().zip(&expected).position(|(a, b)| a != b),
        );
    }
}

#[test]
fn a_hand_built_fragmented_downlink_nas_transport_decodes() {
    for &len in NAS_LENGTHS {
        let nas = nas_of(len);
        let decoded = decode_downlink_nas_transport(&hand_built(7, 9, &nas))
            .unwrap_or_else(|e| panic!("NAS length {len} must decode: {e:?}"));
        assert_eq!(decoded.amf_ue_ngap_id, 7, "NAS length {len}");
        assert_eq!(decoded.ran_ue_ngap_id, 9, "NAS length {len}");
        assert!(
            decoded.nas_pdu == nas,
            "NAS length {len}: NAS-PDU corrupted"
        );
    }
}

/// The fixture helper itself, pinned against the literal octets X.691 gives,
/// so the two tests above are not merely self-consistent.
#[test]
fn the_hand_fragmenter_matches_x691_literally() {
    assert_eq!(frag(&[]), vec![0x00]);
    assert_eq!(frag(&[0xAB; 127])[..1], [0x7F]);
    assert_eq!(frag(&[0xAB; 128])[..2], [0x80, 0x80]);
    assert_eq!(frag(&[0xAB; 16383])[..2], [0xBF, 0xFF]);
    let f = frag(&[0xAB; 16384]);
    assert_eq!((f[0], f.len(), f[f.len() - 1]), (0xC1, 1 + 16384 + 1, 0x00));
    let f = frag(&[0xAB; 16384 + 3729]);
    assert_eq!((f[0], f[16385], f[16386]), (0xC1, 0x8E, 0x91));
    let f = frag(&[0xAB; 65536 + 16384 + 5]);
    assert_eq!((f[0], f[65537], f[65537 + 16385]), (0xC4, 0xC1, 0x05));
    let f = frag(&[0xAB; 200_000]);
    // 3 x 64K = 196,608; remainder 3,392 = 0x0D40.
    assert_eq!(
        (f[0], f[65537], f[2 * 65537], f[3 * 65537], f[3 * 65537 + 1]),
        (0xC4, 0xC4, 0xC4, 0x8D, 0x40)
    );
}
