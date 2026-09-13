//! Property-based round-trip tests for the RRC codecs (#23).
//!
//! Driven through the builder params (`RrcSetupRequestParams`,
//! `RrcSetupCompleteParams`) rather than the generated ASN.1 tree, asserting both
//! properties that matter for a UPER codec:
//!
//! 1. **Encode → decode → encode is a fixpoint**: re-encoding a decoded message
//!    reproduces the same octets, which is what catches a field the codec drops or
//!    a bit it misplaces in a way both directions agree on.
//! 2. **The decoded view equals what was built**, so a fixpoint that faithfully
//!    preserves the wrong value still fails.
//!
//! Case count is 32 by default (`PROPTEST_CASES` overrides it).

use proptest::prelude::*;

use crate::procedures::rrc_setup::{
    decode_rrc_setup_complete, decode_rrc_setup_request, encode_rrc_setup_complete,
    encode_rrc_setup_request, GuamiType, Ng5gSTmsiValue, RegisteredAmfParams, RegisteredAmfPlmn,
    RrcEstablishmentCause, RrcSetupCompleteParams, RrcSetupRequestParams, SNssai, UeIdentity,
};

/// Number of cases per property unless `PROPTEST_CASES` says otherwise.
const CASES: u32 = 32;

fn establishment_cause() -> impl Strategy<Value = RrcEstablishmentCause> {
    prop_oneof![
        Just(RrcEstablishmentCause::Emergency),
        Just(RrcEstablishmentCause::HighPriorityAccess),
        Just(RrcEstablishmentCause::MtAccess),
        Just(RrcEstablishmentCause::MoSignalling),
        Just(RrcEstablishmentCause::MoData),
        Just(RrcEstablishmentCause::MoVoiceCall),
        Just(RrcEstablishmentCause::MoVideoCall),
        Just(RrcEstablishmentCause::MoSms),
        Just(RrcEstablishmentCause::MpsPriorityAccess),
        Just(RrcEstablishmentCause::McsPriorityAccess),
    ]
}

/// A UE identity within its 39-bit field (TS 38.331 `InitialUE-Identity`): a wider
/// value would be truncated on encode, so the generator masks it rather than
/// asserting on a value the wire cannot carry.
fn ue_identity() -> impl Strategy<Value = UeIdentity> {
    const MASK_39: u64 = (1u64 << 39) - 1;
    prop_oneof![
        any::<u64>().prop_map(|v| UeIdentity::Ng5gSTmsiPart1(v & MASK_39)),
        any::<u64>().prop_map(|v| UeIdentity::RandomValue(v & MASK_39)),
    ]
}

fn rrc_setup_request_params() -> impl Strategy<Value = RrcSetupRequestParams> {
    (ue_identity(), establishment_cause()).prop_map(|(ue_identity, establishment_cause)| {
        RrcSetupRequestParams {
            ue_identity,
            establishment_cause,
        }
    })
}

fn s_nssai() -> impl Strategy<Value = SNssai> {
    // The SD is 24 bits.
    (any::<u8>(), proptest::option::of(0u32..(1 << 24))).prop_map(|(sst, sd)| SNssai { sst, sd })
}

fn registered_amf() -> impl Strategy<Value = RegisteredAmfParams> {
    // AMF Set ID is 10 bits and the AMF Pointer 6 (TS 23.003 §2.10.1). The PLMN
    // identity is digit-based: the MCC may be omitted, and the MNC is 2 or 3
    // digits — a count that must survive, since 001-01 and 001-001 are different
    // networks.
    (
        proptest::option::of((
            proptest::option::of(prop::collection::vec(0u8..=9, 3..=3)),
            prop::collection::vec(0u8..=9, 2..=3),
        )),
        any::<u8>(),
        0u16..=0x3FF,
        0u8..=0x3F,
    )
        .prop_map(
            |(plmn, amf_region_id, amf_set_id, amf_pointer)| RegisteredAmfParams {
                plmn: plmn.map(|(mcc, mnc)| RegisteredAmfPlmn {
                    mcc: mcc.map(|d| [d[0], d[1], d[2]]),
                    mnc,
                }),
                amf_region_id,
                amf_set_id,
                amf_pointer,
            },
        )
}

fn rrc_setup_complete_params() -> impl Strategy<Value = RrcSetupCompleteParams> {
    (
        // rrc-TransactionIdentifier is 0..3 and selectedPLMN-Identity 1..12.
        0u8..=3,
        1u8..=12,
        proptest::option::of(registered_amf()),
        proptest::option::of(prop_oneof![
            Just(GuamiType::Native),
            Just(GuamiType::Mapped)
        ]),
        proptest::option::of(prop::collection::vec(s_nssai(), 1..=4)),
        prop::collection::vec(any::<u8>(), 1..=48),
        proptest::option::of(prop_oneof![
            any::<u64>().prop_map(|v| Ng5gSTmsiValue::Full(v & ((1u64 << 48) - 1))),
            any::<u16>().prop_map(|v| Ng5gSTmsiValue::Part2(v & 0x1FF)),
        ]),
        any::<bool>(),
    )
        .prop_map(
            |(
                rrc_transaction_id,
                selected_plmn_identity,
                registered_amf,
                guami_type,
                s_nssai_list,
                dedicated_nas_message,
                ng_5g_s_tmsi_value,
                redcap_indication,
            )| RrcSetupCompleteParams {
                rrc_transaction_id,
                selected_plmn_identity,
                registered_amf,
                guami_type,
                s_nssai_list,
                dedicated_nas_message,
                ng_5g_s_tmsi_value,
                redcap_indication,
            },
        )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(CASES))]

    #[test]
    fn an_rrc_setup_request_survives_a_round_trip(params in rrc_setup_request_params()) {
        let first = encode_rrc_setup_request(&params).expect("UPER encode");
        let decoded = decode_rrc_setup_request(&first).expect("UPER decode");

        prop_assert_eq!(&decoded.ue_identity, &params.ue_identity);
        prop_assert_eq!(decoded.establishment_cause, params.establishment_cause);

        let second = encode_rrc_setup_request(&RrcSetupRequestParams {
            ue_identity: decoded.ue_identity.clone(),
            establishment_cause: decoded.establishment_cause,
        })
        .expect("UPER re-encode");
        prop_assert_eq!(first, second, "re-encoding a decoded message must be byte-identical");
    }

    #[test]
    fn an_rrc_setup_complete_survives_a_round_trip(params in rrc_setup_complete_params()) {
        let first = encode_rrc_setup_complete(&params).expect("UPER encode");
        let decoded = decode_rrc_setup_complete(&first).expect("UPER decode");

        prop_assert_eq!(decoded.rrc_transaction_id, params.rrc_transaction_id);
        prop_assert_eq!(decoded.selected_plmn_identity, params.selected_plmn_identity);
        prop_assert_eq!(&decoded.dedicated_nas_message, &params.dedicated_nas_message);
        prop_assert_eq!(&decoded.s_nssai_list, &params.s_nssai_list);
        prop_assert_eq!(decoded.guami_type, params.guami_type);
        prop_assert_eq!(&decoded.registered_amf, &params.registered_amf);
        prop_assert_eq!(&decoded.ng_5g_s_tmsi_value, &params.ng_5g_s_tmsi_value);
        prop_assert_eq!(decoded.redcap_indication, params.redcap_indication);

        let second = encode_rrc_setup_complete(&RrcSetupCompleteParams {
            rrc_transaction_id: decoded.rrc_transaction_id,
            selected_plmn_identity: decoded.selected_plmn_identity,
            registered_amf: decoded.registered_amf.clone(),
            guami_type: decoded.guami_type,
            s_nssai_list: decoded.s_nssai_list.clone(),
            dedicated_nas_message: decoded.dedicated_nas_message.clone(),
            ng_5g_s_tmsi_value: decoded.ng_5g_s_tmsi_value.clone(),
            redcap_indication: decoded.redcap_indication,
        })
        .expect("UPER re-encode");
        prop_assert_eq!(first, second, "re-encoding a decoded message must be byte-identical");
    }
}
