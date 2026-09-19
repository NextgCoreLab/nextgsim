//! SDAP: the Service Data Adaptation Protocol header (3GPP TS 37.324).
//!
//! # Why this lives in `nextgsim-pdcp`
//!
//! SDAP sits directly **above** PDCP (TS 37.324 §4.2), and like the two modules
//! beside it — `srb_security` and `up_security` — it is code the gNB and the UE
//! must agree on **byte for byte** or the receiver reads a header octet as
//! payload. That shared-layout property, not the crate's name, is what decides
//! where it belongs: a second copy on each endpoint is how the two ends drift.
//!
//! # The header is ONE octet, not three
//!
//! Issue #44's criterion 2 asks for "a 3-byte DL SDAP header carrying the QFI and
//! RQI", and it inherited that number from a comment in
//! `nextgsim-rrc/src/procedures/rrc_reconfiguration.rs` which this change also
//! corrects. **TS 37.324 §6.2.2 defines one octet**, and the field widths make it
//! plain:
//!
//! ```text
//! Downlink SDAP Data PDU with SDAP header (§6.2.2.2):
//!   bit 8 (MSB)  D/C   1 = Data PDU, 0 = Control PDU
//!   bit 7        RQI   Reflective QoS Indication
//!   bits 6..1    QFI   QoS Flow Identifier (6 bits)
//!
//! Uplink SDAP Data PDU with SDAP header (§6.2.2.3):
//!   bit 8 (MSB)  D/C   1 = Data PDU
//!   bit 7        R     Reserved, set to 0
//!   bits 6..1    QFI   QoS Flow Identifier (6 bits)
//! ```
//!
//! A six-bit QFI is why `QosFlowIdentifier ::= INTEGER (0..63)` in TS 38.413 and
//! `QFI ::= INTEGER (1..maxNrofQFIs)` in TS 38.331. Implementing three bytes would
//! produce **exactly the interop failure criterion 2 cites as its motivation** — a
//! real UE would read two octets of payload as header. So one octet it is, and the
//! deviation is deliberate.

use thiserror::Error;

/// Errors decoding an SDAP header.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SdapError {
    /// The PDU is shorter than the one-octet header.
    #[error("SDAP PDU is {0} octets; a Data PDU with a header needs at least 1")]
    TooShort(usize),
    /// The D/C bit says this is a Control PDU, which carries no user data.
    ///
    /// Not an error the data path can act on: an end-marker Control PDU
    /// (§6.2.3) has no payload to deliver.
    #[error("SDAP Control PDU (D/C = 0) carries no user data")]
    ControlPdu,
    /// A QFI outside `INTEGER (0..63)` was offered for encoding.
    #[error("QFI {0} does not fit the 6-bit SDAP QFI field (0..63)")]
    QfiOutOfRange(u8),
}

/// The one-octet SDAP header, TS 37.324 §6.2.2.
pub const SDAP_HEADER_LEN: usize = 1;

/// `D/C` bit mask: bit 8, the MSB. `1` is a Data PDU (§6.2.2.2).
const DC_DATA: u8 = 0b1000_0000;

/// `RQI` bit mask: bit 7. On the uplink this bit is `R`, reserved and zero.
const RQI: u8 = 0b0100_0000;

/// `QFI` field mask: bits 6..1.
const QFI_MASK: u8 = 0b0011_1111;

/// The largest QFI the six-bit field can carry.
pub const MAX_QFI: u8 = 63;

/// A decoded SDAP Data PDU header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SdapHeader {
    /// `QFI`: which QoS flow the SDU belongs to (0..=63).
    pub qfi: u8,
    /// `RQI`: reflective QoS is indicated for this flow (TS 23.501 §5.7.5).
    ///
    /// Downlink only. Always `false` on an uplink header, where the bit is the
    /// reserved `R`.
    pub rqi: bool,
}

/// Encode a **downlink** SDAP Data PDU header octet (§6.2.2.2).
pub fn encode_dl_header(header: SdapHeader) -> Result<u8, SdapError> {
    if header.qfi > MAX_QFI {
        return Err(SdapError::QfiOutOfRange(header.qfi));
    }
    let mut octet = DC_DATA | (header.qfi & QFI_MASK);
    if header.rqi {
        octet |= RQI;
    }
    Ok(octet)
}

/// Encode an **uplink** SDAP Data PDU header octet (§6.2.2.3).
///
/// Takes a QFI and not an [`SdapHeader`]: bit 7 is the reserved `R` on the
/// uplink, so there is no RQI to pass and a signature that accepted one would
/// invite a caller to set a bit the spec reserves.
pub fn encode_ul_header(qfi: u8) -> Result<u8, SdapError> {
    if qfi > MAX_QFI {
        return Err(SdapError::QfiOutOfRange(qfi));
    }
    Ok(DC_DATA | (qfi & QFI_MASK))
}

/// Decode a **downlink** SDAP Data PDU, returning its header and the SDU.
///
/// The SDU is the remainder after the single header octet. A zero-length
/// remainder is returned as an empty slice rather than refused: SDAP does not
/// constrain the SDU's length, and a receiver that rejected one would be
/// enforcing a rule the spec does not have.
pub fn decode_dl_pdu(pdu: &[u8]) -> Result<(SdapHeader, &[u8]), SdapError> {
    let &octet = pdu.first().ok_or(SdapError::TooShort(pdu.len()))?;
    if octet & DC_DATA == 0 {
        return Err(SdapError::ControlPdu);
    }
    Ok((
        SdapHeader {
            qfi: octet & QFI_MASK,
            rqi: octet & RQI != 0,
        },
        &pdu[SDAP_HEADER_LEN..],
    ))
}

/// Decode an **uplink** SDAP Data PDU, returning its QFI and the SDU.
///
/// The reserved `R` bit is **ignored**, not validated: TS 37.324 §6.2.2.3 has the
/// receiver disregard it, and a decoder that refused a set `R` would reject a
/// conformant peer that padded it.
pub fn decode_ul_pdu(pdu: &[u8]) -> Result<(u8, &[u8]), SdapError> {
    let &octet = pdu.first().ok_or(SdapError::TooShort(pdu.len()))?;
    if octet & DC_DATA == 0 {
        return Err(SdapError::ControlPdu);
    }
    Ok((octet & QFI_MASK, &pdu[SDAP_HEADER_LEN..]))
}

/// Prepend a downlink SDAP header to an SDU, producing the SDAP Data PDU.
pub fn build_dl_pdu(header: SdapHeader, sdu: &[u8]) -> Result<Vec<u8>, SdapError> {
    let octet = encode_dl_header(header)?;
    let mut pdu = Vec::with_capacity(SDAP_HEADER_LEN + sdu.len());
    pdu.push(octet);
    pdu.extend_from_slice(sdu);
    Ok(pdu)
}

/// Prepend an uplink SDAP header to an SDU, producing the SDAP Data PDU.
pub fn build_ul_pdu(qfi: u8, sdu: &[u8]) -> Result<Vec<u8>, SdapError> {
    let octet = encode_ul_header(qfi)?;
    let mut pdu = Vec::with_capacity(SDAP_HEADER_LEN + sdu.len());
    pdu.push(octet);
    pdu.extend_from_slice(sdu);
    Ok(pdu)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ====================================================================
    // Absolute layout. These are hand-computed byte assertions, which is the
    // right tool here: there is no independent decoder to drive a round trip
    // against for the CLAIM "the QFI occupies bits 6..1" -- a round trip
    // through this module's own pair would pass for any consistent bit
    // assignment, including a wrong one.
    // ====================================================================

    /// The exact octets TS 37.324 §6.2.2.2 specifies, computed by hand from the
    /// field positions.
    #[test]
    fn the_downlink_header_octet_matches_the_hand_derived_layout() {
        // D/C=1, RQI=0, QFI=0     -> 1 0 000000 = 0x80
        assert_eq!(
            encode_dl_header(SdapHeader { qfi: 0, rqi: false }),
            Ok(0x80),
            "D/C set, everything else clear"
        );
        // D/C=1, RQI=1, QFI=0     -> 1 1 000000 = 0xC0
        assert_eq!(
            encode_dl_header(SdapHeader { qfi: 0, rqi: true }),
            Ok(0xC0),
            "RQI is bit 7"
        );
        // D/C=1, RQI=0, QFI=1     -> 1 0 000001 = 0x81
        assert_eq!(
            encode_dl_header(SdapHeader { qfi: 1, rqi: false }),
            Ok(0x81)
        );
        // D/C=1, RQI=0, QFI=63    -> 1 0 111111 = 0xBF  (the six-bit maximum)
        assert_eq!(
            encode_dl_header(SdapHeader {
                qfi: MAX_QFI,
                rqi: false
            }),
            Ok(0xBF),
            "QFI 63 fills bits 6..1 and must NOT bleed into RQI"
        );
        // D/C=1, RQI=1, QFI=63    -> 1 1 111111 = 0xFF
        assert_eq!(
            encode_dl_header(SdapHeader {
                qfi: MAX_QFI,
                rqi: true
            }),
            Ok(0xFF)
        );
        // QFI 9, the default non-GBR flow this simulator uses most
        // -> 1 0 001001 = 0x89
        assert_eq!(
            encode_dl_header(SdapHeader { qfi: 9, rqi: false }),
            Ok(0x89)
        );
    }

    /// The uplink octet has NO RQI: bit 7 is the reserved `R` and must be zero,
    /// so the same QFI produces a different octet from a DL header with RQI set.
    #[test]
    fn the_uplink_header_octet_leaves_the_reserved_bit_clear() {
        assert_eq!(encode_ul_header(0), Ok(0x80));
        assert_eq!(encode_ul_header(1), Ok(0x81));
        assert_eq!(encode_ul_header(MAX_QFI), Ok(0xBF));
        // The whole point: no value of the UL QFI can set bit 7.
        for qfi in 0..=MAX_QFI {
            let octet = encode_ul_header(qfi).expect("in range");
            assert_eq!(
                octet & RQI,
                0,
                "UL QFI {qfi} must not set the reserved R bit"
            );
        }
    }

    /// The header is ONE octet. Asserted on a length, because this is the number
    /// issue #44's criterion 2 got wrong and the source comment it came from said
    /// three.
    #[test]
    fn the_header_is_one_octet() {
        assert_eq!(SDAP_HEADER_LEN, 1, "TS 37.324 §6.2.2");
        let pdu = build_dl_pdu(SdapHeader { qfi: 5, rqi: false }, &[0xAA, 0xBB]).expect("build");
        assert_eq!(
            pdu.len(),
            3,
            "one header octet plus a two-octet SDU -- NOT 3 + 2"
        );
        assert_eq!(pdu, vec![0x85, 0xAA, 0xBB]);
    }

    // ====================================================================
    // Round trips, over a spread including the six-bit boundary.
    // ====================================================================

    /// Every QFI and both RQI values survive a downlink round trip, and the SDU
    /// comes back byte-identical. 0..=63 exhaustively, so a masking error at any
    /// bit position shows up.
    #[test]
    fn every_downlink_header_round_trips() {
        let sdu = [0xDE, 0xAD, 0xBE, 0xEF];
        for qfi in 0..=MAX_QFI {
            for rqi in [false, true] {
                let header = SdapHeader { qfi, rqi };
                let pdu = build_dl_pdu(header, &sdu).expect("build");
                let (decoded, payload) = decode_dl_pdu(&pdu).expect("decode");
                assert_eq!(decoded, header, "QFI {qfi}, RQI {rqi}");
                assert_eq!(payload, sdu, "the SDU must survive QFI {qfi}");
            }
        }
    }

    /// The uplink half, exhaustively. Separate from the DL because the two
    /// headers differ in bit 7 and a shared test would not notice if one
    /// direction started encoding the other's layout.
    #[test]
    fn every_uplink_header_round_trips() {
        let sdu = [0x01, 0x02];
        for qfi in 0..=MAX_QFI {
            let pdu = build_ul_pdu(qfi, &sdu).expect("build");
            let (decoded, payload) = decode_ul_pdu(&pdu).expect("decode");
            assert_eq!(decoded, qfi, "QFI {qfi}");
            assert_eq!(payload, sdu);
        }
    }

    /// 64 is the first QFI the six-bit field cannot hold, and it is REFUSED
    /// rather than truncated to 0 — which is what a bare `& 0x3F` would do, and
    /// would silently reassign the flow to QFI 0.
    #[test]
    fn a_qfi_past_the_six_bit_field_is_refused() {
        assert_eq!(
            encode_dl_header(SdapHeader {
                qfi: 64,
                rqi: false
            }),
            Err(SdapError::QfiOutOfRange(64)),
            "64 is one past the field, and truncating would make it QFI 0"
        );
        assert_eq!(encode_ul_header(64), Err(SdapError::QfiOutOfRange(64)));
        assert_eq!(encode_ul_header(255), Err(SdapError::QfiOutOfRange(255)));
        // And 63 is accepted, so the boundary is the boundary and not an
        // off-by-one.
        assert!(encode_ul_header(MAX_QFI).is_ok());
    }

    /// A PDU too short to hold the header is refused rather than indexed into.
    #[test]
    fn an_empty_pdu_is_refused() {
        assert_eq!(decode_dl_pdu(&[]), Err(SdapError::TooShort(0)));
        assert_eq!(decode_ul_pdu(&[]), Err(SdapError::TooShort(0)));
    }

    /// A header with no SDU behind it decodes to an empty payload, not an error:
    /// SDAP does not constrain the SDU length.
    #[test]
    fn a_header_with_no_sdu_decodes_to_an_empty_payload() {
        let (header, payload) = decode_dl_pdu(&[0x89]).expect("decode");
        assert_eq!(header, SdapHeader { qfi: 9, rqi: false });
        assert!(payload.is_empty());
    }

    /// D/C = 0 is a Control PDU (§6.2.3) and carries no user data, so it is
    /// reported rather than delivered as a zero-QFI data PDU.
    #[test]
    fn a_control_pdu_is_not_read_as_data() {
        // 0x09: D/C clear, QFI bits set -- exactly the shape that would decode as
        // "QFI 9 data" if the D/C bit were ignored.
        assert_eq!(decode_dl_pdu(&[0x09, 0xAA]), Err(SdapError::ControlPdu));
        assert_eq!(decode_ul_pdu(&[0x09, 0xAA]), Err(SdapError::ControlPdu));
    }

    /// The reserved `R` bit is ignored on receive, so a peer that sets it is
    /// still understood — and its QFI is not corrupted by the stray bit.
    #[test]
    fn a_set_reserved_bit_is_ignored_on_the_uplink() {
        // D/C=1, R=1, QFI=9 -> 0xC9. A conformant sender clears R, but §6.2.2.3
        // has the receiver disregard it.
        let (qfi, payload) = decode_ul_pdu(&[0xC9, 0x77]).expect("decode");
        assert_eq!(
            qfi, 9,
            "the QFI must be read from bits 6..1 regardless of R"
        );
        assert_eq!(payload, [0x77]);
    }

    /// The DL and UL headers for the same QFI are the same octet when RQI is
    /// clear — which is what makes the two directions interoperable — and differ
    /// when it is set.
    #[test]
    fn the_two_directions_differ_only_in_the_rqi_bit() {
        for qfi in 0..=MAX_QFI {
            assert_eq!(
                encode_dl_header(SdapHeader { qfi, rqi: false }),
                encode_ul_header(qfi),
                "with RQI clear the two directions must agree for QFI {qfi}"
            );
            assert_ne!(
                encode_dl_header(SdapHeader { qfi, rqi: true }),
                encode_ul_header(qfi),
                "with RQI set they must not, for QFI {qfi}"
            );
        }
    }
}
