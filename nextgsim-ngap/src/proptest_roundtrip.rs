//! Property-based round-trip tests for the NGAP codec (#23).
//!
//! The generated `NGAP_PDU` tree is impractical to generate directly, so these
//! drive the **builder params** the tree already uses (`NgSetupRequestParams`,
//! `InitialUeMessageParams`) and assert two properties over generated inputs:
//!
//! 1. **Encode → decode → encode is a fixpoint.** Re-encoding a decoded PDU must
//!    reproduce the same octets. This is the property that catches a field APER
//!    drops or reorders, which a parse-and-compare test cannot see when the parser
//!    and the builder share the same blind spot.
//! 2. **The parsed view matches what was built** for the fields the crate's own
//!    parsers expose, so a fixpoint that preserves the wrong value still fails.
//!
//! Case count is 32 by default (`PROPTEST_CASES` overrides it): APER encoding of
//! these PDUs is heavier than the NAS codecs, and the whole workspace suite runs on
//! every push.

use proptest::prelude::*;

use crate::codec::{decode_ngap_pdu, encode_ngap_pdu};
use crate::procedures::initial_ue_message::{
    build_initial_ue_message, parse_initial_ue_message, FiveGSTmsi, InitialUeMessageParams, NrCgi,
    RrcEstablishmentCauseValue, Tai, UserLocationInfoNr,
};
use crate::procedures::ng_setup::{
    build_ng_setup_request, BroadcastPlmnItem, GnbId, NgSetupRequestParams, PagingDrx, SNssai,
    SupportedTaItem,
};

/// Number of cases per property unless `PROPTEST_CASES` says otherwise.
const CASES: u32 = 32;

/// A BCD-encoded PLMN identity. Generated from digits rather than arbitrary
/// octets: the encoder packs nibbles, and an octet with an out-of-range nibble is
/// not a PLMN any network could broadcast.
fn plmn_identity() -> impl Strategy<Value = [u8; 3]> {
    prop::collection::vec(0u8..=9, 6..=6)
        .prop_map(|d| [(d[1] << 4) | d[0], (0xF << 4) | d[2], (d[5] << 4) | d[4]])
}

fn s_nssai() -> impl Strategy<Value = SNssai> {
    (any::<u8>(), proptest::option::of(any::<[u8; 3]>())).prop_map(|(sst, sd)| SNssai { sst, sd })
}

fn gnb_id() -> impl Strategy<Value = GnbId> {
    // The gNB ID is 22-32 bits; a value wider than its length would be truncated
    // on encode, so the value is masked to the generated length.
    (plmn_identity(), any::<u32>(), 22u8..=32).prop_map(|(plmn_identity, value, gnb_id_length)| {
        let mask = if gnb_id_length >= 32 {
            u32::MAX
        } else {
            (1u32 << gnb_id_length) - 1
        };
        GnbId {
            plmn_identity,
            gnb_id_value: value & mask,
            gnb_id_length,
        }
    })
}

fn ng_setup_request_params() -> impl Strategy<Value = NgSetupRequestParams> {
    (
        gnb_id(),
        proptest::option::of("[a-zA-Z0-9-]{1,12}"),
        prop::collection::vec(
            (
                any::<[u8; 3]>(),
                prop::collection::vec(
                    (plmn_identity(), prop::collection::vec(s_nssai(), 1..=2)),
                    1..=2,
                ),
            ),
            1..=2,
        ),
        prop_oneof![
            Just(PagingDrx::V32),
            Just(PagingDrx::V64),
            Just(PagingDrx::V128),
            Just(PagingDrx::V256),
        ],
    )
        .prop_map(
            |(gnb_id, ran_node_name, tas, default_paging_drx)| NgSetupRequestParams {
                gnb_id,
                ran_node_name,
                supported_ta_list: tas
                    .into_iter()
                    .map(|(tac, plmns)| SupportedTaItem {
                        tac,
                        broadcast_plmn_list: plmns
                            .into_iter()
                            .map(|(plmn_identity, slice_support_list)| BroadcastPlmnItem {
                                plmn_identity,
                                slice_support_list,
                            })
                            .collect(),
                    })
                    .collect(),
                default_paging_drx,
            },
        )
}

fn initial_ue_message_params() -> impl Strategy<Value = InitialUeMessageParams> {
    (
        any::<u32>(),
        prop::collection::vec(any::<u8>(), 3..=32),
        // The NR Cell Identity is 36 bits.
        (plmn_identity(), any::<[u8; 3]>(), 0u64..(1u64 << 36)),
        prop_oneof![
            Just(RrcEstablishmentCauseValue::Emergency),
            Just(RrcEstablishmentCauseValue::MoSignalling),
            Just(RrcEstablishmentCauseValue::MoData),
            Just(RrcEstablishmentCauseValue::MpsHighPriorityAccess),
        ],
        proptest::option::of((0u16..=0x3FF, 0u8..=0x3F, any::<[u8; 4]>())),
    )
        .prop_map(
            |(
                ran_ue_ngap_id,
                nas_pdu,
                (plmn_identity, tac, nr_cell_identity),
                rrc_establishment_cause,
                s_tmsi,
            )| InitialUeMessageParams {
                ran_ue_ngap_id,
                nas_pdu,
                user_location_info: UserLocationInfoNr {
                    nr_cgi: NrCgi {
                        plmn_identity,
                        nr_cell_identity,
                    },
                    tai: Tai { plmn_identity, tac },
                    time_stamp: None,
                },
                rrc_establishment_cause,
                five_g_s_tmsi: s_tmsi.map(|(amf_set_id, amf_pointer, five_g_tmsi)| FiveGSTmsi {
                    amf_set_id,
                    amf_pointer,
                    five_g_tmsi,
                }),
                // AMF Set ID selection, UE context request and allowed NSSAI are
                // driven by the gNB's own policy rather than by the UE's initial
                // message; the parser this asserts against does not surface them,
                // so generating them would widen the input without widening the
                // assertion.
                amf_set_id: None,
                ue_context_request: None,
                allowed_nssai: None,
            },
        )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(CASES))]

    /// An NG Setup Request re-encodes to the same octets after a decode.
    ///
    /// Fixpoint only: this crate has no `parse_ng_setup_request` — the request is
    /// what the gNB *sends*, and the AMF that parses it lives in nextgcore. The
    /// fixpoint is still the property that matters for an encoder, and the
    /// cross-codec byte vectors in `capture_tests` pin the wire form against that
    /// AMF's own codec.
    #[test]
    fn an_ng_setup_request_is_an_encode_decode_fixpoint(params in ng_setup_request_params()) {
        let pdu = build_ng_setup_request(&params).expect("the params must build");
        let first = encode_ngap_pdu(&pdu).expect("APER encode");
        let decoded = decode_ngap_pdu(&first).expect("APER decode");
        let second = encode_ngap_pdu(&decoded).expect("APER re-encode");
        prop_assert_eq!(first, second, "re-encoding a decoded PDU must be byte-identical");
    }

    /// An Initial UE Message re-encodes identically and keeps the NAS PDU and the
    /// UE's location intact — the two things the AMF acts on.
    #[test]
    fn an_initial_ue_message_is_an_encode_decode_fixpoint(params in initial_ue_message_params()) {
        let pdu = build_initial_ue_message(&params).expect("the params must build");
        let first = encode_ngap_pdu(&pdu).expect("APER encode");
        let decoded = decode_ngap_pdu(&first).expect("APER decode");
        let second = encode_ngap_pdu(&decoded).expect("APER re-encode");
        prop_assert_eq!(first, second, "re-encoding a decoded PDU must be byte-identical");

        let parsed = parse_initial_ue_message(&decoded).expect("the crate's own parser");
        prop_assert_eq!(parsed.ran_ue_ngap_id, params.ran_ue_ngap_id);
        prop_assert_eq!(&parsed.nas_pdu, &params.nas_pdu);
        prop_assert_eq!(
            parsed.user_location_info.nr_cgi.nr_cell_identity,
            params.user_location_info.nr_cgi.nr_cell_identity
        );
        prop_assert_eq!(
            parsed.user_location_info.tai.tac,
            params.user_location_info.tai.tac
        );
        prop_assert_eq!(parsed.rrc_establishment_cause, params.rrc_establishment_cause);
    }
}
