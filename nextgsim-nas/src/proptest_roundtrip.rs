//! Property-based round-trip tests for the NAS codecs (#23).
//!
//! The hand-written round trips in [`crate::capture_tests`] check a handful of
//! fixed messages. These check the same property — `decode(encode(m)) == m` — over
//! generated field combinations, so a codec that happens to work for the one
//! value a fixture chose is not mistaken for one that works.
//!
//! # What the generators do and do not cover
//!
//! Each strategy generates the fields the message's own encoder writes, in the
//! ranges the wire format allows, and the assertions compare exactly those. Where
//! a field is deliberately excluded the reason is stated at the strategy, because
//! an unstated exclusion reads as coverage.
//!
//! Case count is 64 by default (`PROPTEST_CASES` overrides it) so
//! `cargo test --workspace` stays fast.

use proptest::prelude::*;

use crate::messages::mm::{
    Abba, AuthenticationParameterAutn, AuthenticationParameterRand, AuthenticationRequest,
    Ie5gsMobileIdentity, IeNasSecurityAlgorithms, IeUeSecurityCapability, MobileIdentityType,
    RegistrationRequest, SecurityModeCommand,
};
use crate::security::NasKeySetIdentifier;
use crate::{FollowOnRequest, Ie5gsRegistrationType, RegistrationType, SecurityContextType};

/// Number of cases per property unless `PROPTEST_CASES` says otherwise.
const CASES: u32 = 64;

fn ng_ksi() -> impl Strategy<Value = NasKeySetIdentifier> {
    // ksi is 3 bits (0..=7); 7 means "no key" and is a legal value to signal.
    (
        prop_oneof![
            Just(SecurityContextType::Native),
            Just(SecurityContextType::Mapped),
        ],
        0u8..=7,
    )
        .prop_map(|(context, ksi)| NasKeySetIdentifier::new(context, ksi))
}

fn registration_type() -> impl Strategy<Value = Ie5gsRegistrationType> {
    (
        prop_oneof![
            Just(FollowOnRequest::NoPending),
            Just(FollowOnRequest::Pending),
        ],
        prop_oneof![
            Just(RegistrationType::InitialRegistration),
            Just(RegistrationType::MobilityRegistrationUpdating),
            Just(RegistrationType::PeriodicRegistrationUpdating),
            Just(RegistrationType::EmergencyRegistration),
        ],
    )
        .prop_map(|(follow_on, kind)| Ie5gsRegistrationType::new(follow_on, kind))
}

/// A mobile identity whose first octet agrees with its declared type.
///
/// The decoder reads the type back out of `data[0] & 0x07` (TS 24.501 §9.11.3.4),
/// so a generator that chose the two independently would produce messages no
/// encoder in the tree can emit and fail for that reason alone.
fn mobile_identity() -> impl Strategy<Value = Ie5gsMobileIdentity> {
    (
        prop_oneof![
            Just(MobileIdentityType::Suci),
            Just(MobileIdentityType::Guti),
            Just(MobileIdentityType::Imei),
            Just(MobileIdentityType::Tmsi),
            Just(MobileIdentityType::ImeiSv),
        ],
        prop::collection::vec(any::<u8>(), 0..12),
    )
        .prop_map(|(identity_type, tail)| {
            let mut data = Vec::with_capacity(tail.len() + 1);
            data.push(0xF0 | (identity_type as u8));
            data.extend_from_slice(&tail);
            Ie5gsMobileIdentity::new(identity_type, data)
        })
}

/// A REGISTRATION REQUEST with generated mandatory fields and a generated subset
/// of the optional IEs.
///
/// Excluded on purpose: the 6G extension IEs (`ai_ml_capability` and friends) and
/// the Service-level-AA / payload containers, whose values are structured
/// sub-messages with their own codecs — generating raw bytes for them would test
/// this encoder's TLV framing while asserting on a value the sub-codec is free to
/// normalise. They keep their own fixed-input tests.
fn registration_request() -> impl Strategy<Value = RegistrationRequest> {
    (
        registration_type(),
        ng_ksi(),
        mobile_identity(),
        proptest::option::of(ng_ksi()),
        // UE security capability is 2-8 octets (TS 24.501 §9.11.3.54).
        proptest::option::of(prop::collection::vec(any::<u8>(), 2..=8)),
        // Requested NSSAI: at least one S-NSSAI, up to the IE's 74-octet maximum.
        proptest::option::of(prop::collection::vec(any::<u8>(), 1..=32)),
        proptest::option::of(any::<[u8; 6]>()),
        proptest::option::of(any::<u8>()),
        proptest::option::of(any::<u16>()),
        proptest::option::of(any::<u16>()),
        any::<bool>(),
    )
        .prop_map(
            |(
                registration_type,
                ng_ksi,
                mobile_identity,
                non_current_ng_ksi,
                ue_security_capability,
                requested_nssai,
                last_visited_tai,
                ue_status,
                uplink_data_status,
                pdu_session_status,
                disaster_roaming,
            )| {
                let mut msg = RegistrationRequest::new(registration_type, ng_ksi, mobile_identity);
                msg.non_current_ng_ksi = non_current_ng_ksi;
                msg.ue_security_capability = ue_security_capability;
                msg.requested_nssai = requested_nssai;
                msg.last_visited_tai = last_visited_tai;
                msg.ue_status = ue_status;
                msg.uplink_data_status = uplink_data_status;
                msg.pdu_session_status = pdu_session_status;
                msg.disaster_roaming = disaster_roaming;
                msg
            },
        )
}

fn authentication_request() -> impl Strategy<Value = AuthenticationRequest> {
    (
        ng_ksi(),
        // ABBA is at least 2 octets (TS 24.501 §9.11.3.10).
        prop::collection::vec(any::<u8>(), 2..=6),
        proptest::option::of(any::<[u8; 16]>()),
        proptest::option::of(prop::collection::vec(any::<u8>(), 16..=16)),
    )
        .prop_map(|(ng_ksi, abba, rand, autn)| AuthenticationRequest {
            ng_ksi,
            abba: Abba::new(abba),
            rand: rand.map(AuthenticationParameterRand::new),
            autn: autn.map(AuthenticationParameterAutn::new),
            // An EAP message and a RAND/AUTN pair are alternatives, and the EAP
            // body is its own protocol: excluded here, covered by the EAP tests.
            eap_message: None,
        })
}

/// A replayed UE security capability with generated per-algorithm support flags.
///
/// The IE is a pair of bitmaps (TS 24.501 §9.11.3.54); generating the flags rather
/// than two bytes means a bit the codec maps to the wrong position shows up as a
/// mismatched flag rather than as an equal byte.
fn ue_security_capability() -> impl Strategy<Value = IeUeSecurityCapability> {
    (
        prop::collection::vec(any::<bool>(), 16..=16),
        // The EPS algorithm octets are present as a pair or not at all: the
        // encoder writes them only when both are `Some` (length 4 vs 2), so a
        // generator that set one alone would produce a message it then reads back
        // as neither.
        proptest::option::of((any::<u8>(), any::<u8>())),
    )
        .prop_map(|(bits, eps)| IeUeSecurityCapability {
            ea0: bits[0],
            ea1_128: bits[1],
            ea2_128: bits[2],
            ea3_128: bits[3],
            ea4: bits[4],
            ea5: bits[5],
            ea6: bits[6],
            ea7: bits[7],
            ia0: bits[8],
            ia1_128: bits[9],
            ia2_128: bits[10],
            ia3_128: bits[11],
            ia4: bits[12],
            ia5: bits[13],
            ia6: bits[14],
            ia7: bits[15],
            eps_ea: eps.map(|(ea, _)| ea),
            eps_ia: eps.map(|(_, ia)| ia),
        })
}

fn security_mode_command() -> impl Strategy<Value = SecurityModeCommand> {
    (
        // Ciphering and integrity algorithm identities are 4 bits each.
        (0u8..=15, 0u8..=15),
        ng_ksi(),
        ue_security_capability(),
        proptest::option::of(any::<u8>()),
        proptest::option::of(prop::collection::vec(any::<u8>(), 1..=4)),
        proptest::option::of(prop::collection::vec(any::<u8>(), 2..=6)),
    )
        .prop_map(
            |(
                (ciphering, integrity),
                ng_ksi,
                replayed_ue_security_capabilities,
                selected_eps_nas_security_algorithms,
                additional_5g_security_information,
                abba,
            )| {
                let mut msg = SecurityModeCommand::new(
                    IeNasSecurityAlgorithms::new(ciphering, integrity),
                    ng_ksi,
                    replayed_ue_security_capabilities,
                );
                msg.selected_eps_nas_security_algorithms = selected_eps_nas_security_algorithms;
                msg.additional_5g_security_information = additional_5g_security_information;
                msg.abba = abba;
                msg
            },
        )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(CASES))]

    /// Every generated REGISTRATION REQUEST survives encode → decode unchanged in
    /// the fields the encoder writes.
    #[test]
    fn a_registration_request_survives_a_round_trip(msg in registration_request()) {
        let mut buf = Vec::new();
        msg.encode(&mut buf);
        let decoded = RegistrationRequest::decode(&mut &buf[3..])
            .expect("a message this encoder produced must decode");

        prop_assert_eq!(decoded.registration_type, msg.registration_type);
        prop_assert_eq!(decoded.ng_ksi, msg.ng_ksi);
        prop_assert_eq!(&decoded.mobile_identity, &msg.mobile_identity);
        prop_assert_eq!(decoded.non_current_ng_ksi, msg.non_current_ng_ksi);
        prop_assert_eq!(&decoded.ue_security_capability, &msg.ue_security_capability);
        prop_assert_eq!(&decoded.requested_nssai, &msg.requested_nssai);
        prop_assert_eq!(decoded.last_visited_tai, msg.last_visited_tai);
        prop_assert_eq!(decoded.ue_status, msg.ue_status);
        prop_assert_eq!(decoded.uplink_data_status, msg.uplink_data_status);
        prop_assert_eq!(decoded.pdu_session_status, msg.pdu_session_status);
        prop_assert_eq!(decoded.disaster_roaming, msg.disaster_roaming);
    }

    /// The encoding is a fixpoint: re-encoding what was decoded reproduces the
    /// same octets. This catches a field the decoder drops silently, which a
    /// field-by-field comparison of only the fields the test names cannot.
    #[test]
    fn a_registration_request_re_encodes_to_the_same_bytes(msg in registration_request()) {
        let mut first = Vec::new();
        msg.encode(&mut first);
        let decoded = RegistrationRequest::decode(&mut &first[3..]).expect("decode");
        let mut second = Vec::new();
        decoded.encode(&mut second);
        prop_assert_eq!(first, second);
    }

    #[test]
    fn an_authentication_request_survives_a_round_trip(msg in authentication_request()) {
        let mut buf = Vec::new();
        msg.encode(&mut buf);
        let decoded = AuthenticationRequest::decode(&mut &buf[3..]).expect("decode");

        prop_assert_eq!(decoded.ng_ksi, msg.ng_ksi);
        prop_assert_eq!(&decoded.abba.value, &msg.abba.value);
        prop_assert_eq!(
            decoded.rand.as_ref().map(|r| r.value),
            msg.rand.as_ref().map(|r| r.value)
        );
        prop_assert_eq!(
            decoded.autn.as_ref().map(|a| a.value.clone()),
            msg.autn.as_ref().map(|a| a.value.clone())
        );
    }

    #[test]
    fn an_authentication_request_re_encodes_to_the_same_bytes(msg in authentication_request()) {
        let mut first = Vec::new();
        msg.encode(&mut first);
        let decoded = AuthenticationRequest::decode(&mut &first[3..]).expect("decode");
        let mut second = Vec::new();
        decoded.encode(&mut second);
        prop_assert_eq!(first, second);
    }

    #[test]
    fn a_security_mode_command_survives_a_round_trip(msg in security_mode_command()) {
        let mut buf = Vec::new();
        msg.encode(&mut buf);
        let decoded = SecurityModeCommand::decode(&mut &buf[3..]).expect("decode");

        prop_assert_eq!(
            decoded.selected_nas_security_algorithms,
            msg.selected_nas_security_algorithms
        );
        prop_assert_eq!(decoded.ng_ksi, msg.ng_ksi);
        prop_assert_eq!(
            decoded.replayed_ue_security_capabilities,
            msg.replayed_ue_security_capabilities
        );
        prop_assert_eq!(
            decoded.selected_eps_nas_security_algorithms,
            msg.selected_eps_nas_security_algorithms
        );
        prop_assert_eq!(
            &decoded.additional_5g_security_information,
            &msg.additional_5g_security_information
        );
        prop_assert_eq!(&decoded.abba, &msg.abba);
    }

    #[test]
    fn a_security_mode_command_re_encodes_to_the_same_bytes(msg in security_mode_command()) {
        let mut first = Vec::new();
        msg.encode(&mut first);
        let decoded = SecurityModeCommand::decode(&mut &first[3..]).expect("decode");
        let mut second = Vec::new();
        decoded.encode(&mut second);
        prop_assert_eq!(first, second);
    }
}
