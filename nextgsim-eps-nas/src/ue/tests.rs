//! The attach procedure against a scripted MME. The MME side is written here
//! from TS 24.301 / TS 33.401 directly -- its protection and AUTN are built
//! with the crypto primitives, not with [`EpsSecurityContext`] -- so a
//! symmetric misreading in `security.rs` cannot pass.

use super::*;
use crate::testing::*;
use nextgsim_crypto::eps_kdf::derive_kenb;

#[test]
fn attach_completes_and_activates_the_default_bearer() {
    let (ue, _mme, complete) = attach();
    assert_eq!(&complete[..2], &[0x07, 0x43]);
    // ESM container: ACTIVATE DEFAULT EPS BEARER CONTEXT ACCEPT, EBI 5.
    assert_eq!(&complete[2..4], &[0x00, 0x03]);
    assert_eq!(complete[4], 0x52);
    assert_eq!(complete[6], 0xC2);
    assert_eq!(ue.state(), EmmState::Registered);
    let b = ue.default_bearer().unwrap();
    assert_eq!(b.ebi, 5);
    assert_eq!(b.apn, "internet");
    assert_eq!(
        b.pdn_address,
        PdnAddress::Ipv4(std::net::Ipv4Addr::new(10, 45, 0, 2))
    );
    assert_eq!(ue.guti().unwrap().m_tmsi, 0xC000_0001);
    assert_eq!(ue.s_tmsi(), Some((1, 0xC000_0001)));
}

#[test]
fn kenb_is_the_mmes_with_the_security_mode_complete_count() {
    let (ue, mme, _) = attach();
    // TS 33.401 §7.2.6.2: after a NAS SMC, the UL NAS COUNT of the SECURITY
    // MODE COMPLETE -- the first message of the new context, COUNT 0.
    assert_eq!(ue.k_enb(), Some(derive_kenb(&mme.kasme, 0)));
}

#[test]
fn a_tampered_attach_accept_is_discarded() {
    let now = Instant::now();
    let mut ue = EpsNasUe::new(cfg());
    let req = ue.initial_message(now).unwrap();
    let pti = req[18];
    let mut mme = Mme::new(0x21);
    ue.on_downlink(&mme.authentication_request(), now);
    ue.on_downlink(&mme.security_mode_command([0xE0, 0xE0], false), now);
    let mut accept = mme.attach_accept(pti);
    accept[10] ^= 0x01;
    assert_eq!(ue.on_downlink(&accept, now), NasOutput::default());
    assert_eq!(ue.state(), EmmState::RegisteredInitiated);
}

#[test]
fn an_unprotected_attach_accept_is_discarded() {
    let now = Instant::now();
    let mut ue = EpsNasUe::new(cfg());
    ue.initial_message(now).unwrap();
    let plain = [
        0x07, 0x42, 0x01, 0x49, 0x06, 0x00, 0x99, 0xF9, 0x07, 0x00, 0x01, 0x00, 0x00,
    ];
    assert_eq!(ue.on_downlink(&plain, now), NasOutput::default());
    assert_eq!(ue.state(), EmmState::RegisteredInitiated);
}

#[test]
fn a_bad_autn_gets_mac_failure_and_a_stale_one_synch_failure() {
    let now = Instant::now();
    let mut ue = EpsNasUe::new(cfg());
    ue.initial_message(now).unwrap();
    let mme = Mme::new(0x21);
    let mut bad = mme.authentication_request();
    let last = bad.len() - 1;
    bad[last] ^= 1;
    assert_eq!(ue.on_downlink(&bad, now).uplink, vec![vec![0x07, 0x5C, 20]]);
    // Authenticate once at SQN 0x21; a new challenge (different RAND) at a
    // stale SQN then reaches the USIM and fails synchronisation.
    ue.on_downlink(&mme.authentication_request(), now);
    let stale = Mme::with_rand(0x21, [0x24; 16]);
    let out = ue.on_downlink(&stale.authentication_request(), now);
    assert_eq!(&out.uplink[0][..5], &[0x07, 0x5C, 21, 0x30, 14]);
}

/// A retransmitted AUTHENTICATION REQUEST (same RAND, e.g. after the MME's
/// T3460 fires) is answered with the stored RES, not passed to the USIM
/// again, where its SQN is no longer fresh (TS 24.301 §5.4.2.3, #237).
fn first_challenge() -> (EpsNasUe, Mme, Instant) {
    let now = Instant::now();
    let mut ue = EpsNasUe::new(cfg());
    ue.initial_message(now).unwrap();
    let mme = Mme::new(0x21);
    let out = ue.on_downlink(&mme.authentication_request(), now);
    assert_eq!(&out.uplink[0][3..], &mme.xres);
    (ue, mme, now)
}

fn is_synch_failure(out: &NasOutput) -> bool {
    out.uplink
        .first()
        .is_some_and(|p| p[..3] == [0x07, 0x5C, 21])
}

#[test]
fn a_retransmitted_challenge_gets_the_stored_res() {
    let (mut ue, mut mme, now) = first_challenge();
    let out = ue.on_downlink(&mme.authentication_request(), now);
    assert_eq!(out.uplink.len(), 1);
    assert_eq!(
        &out.uplink[0][..3],
        &[0x07, 0x53, 8],
        "AUTHENTICATION RESPONSE"
    );
    assert_eq!(&out.uplink[0][3..], &mme.xres, "the stored RES, unchanged");
    // The K_ASME of that challenge is still the one the SMC takes into use.
    let out = ue.on_downlink(&mme.security_mode_command([0xE0, 0xE0], false), now);
    assert_eq!(out.uplink[0][0], 0x47, "SECURITY MODE COMPLETE");
}

#[test]
fn a_new_rand_replaces_the_stored_pair() {
    let (mut ue, mme, now) = first_challenge();
    let second = Mme::with_rand(0x22, [0x24; 16]);
    let out = ue.on_downlink(&second.authentication_request(), now);
    assert_eq!(&out.uplink[0][3..], &second.xres);
    // The first RAND is no longer stored: it goes to the USIM and is stale.
    assert!(is_synch_failure(
        &ue.on_downlink(&mme.authentication_request(), now)
    ));
}

#[test]
fn the_stored_pair_is_deleted_on_security_mode_command() {
    let (mut ue, mut mme, now) = first_challenge();
    let complete = ue.on_downlink(&mme.security_mode_command([0xE0, 0xE0], false), now);
    mme.unprotect(&complete.uplink[0]);
    // On the secure exchange the challenge must be integrity protected
    // (§4.4.4.2: a plain one is discarded, #400 review), and it is answered
    // inside the new security context, so unprotect before reading.
    let challenge = mme.authentication_request();
    let out = ue.on_downlink(&mme.protect(&challenge, 2), now);
    let plain = mme.unprotect(&out.uplink[0]);
    assert_eq!(&plain[..3], &[0x07, 0x5C, 21], "the RAND went to the USIM");
}

#[test]
fn the_stored_pair_is_deleted_on_t3416_expiry() {
    let (mut ue, mme, now) = first_challenge();
    // T3416 is armed alongside T3410. T3410 (15 s) would otherwise abort the
    // attach first, which deletes the pair on entering EMM-DEREGISTERED; stop
    // it so T3416 is the timer under test.
    assert!(ue
        .stored_auth
        .as_ref()
        .is_some_and(|a| a.t3416 == now + T3416));
    ue.t3410 = None;
    assert_eq!(ue.next_deadline(), Some(now + T3416));
    ue.on_timer(now + T3416 - Duration::from_millis(1));
    assert!(!is_synch_failure(
        &ue.on_downlink(&mme.authentication_request(), now)
    ));
    ue.on_timer(now + T3416);
    assert!(is_synch_failure(
        &ue.on_downlink(&mme.authentication_request(), now)
    ));
}

#[test]
fn the_stored_pair_is_deleted_on_entering_emm_idle() {
    let (mut ue, mme, now) = first_challenge();
    ue.on_released(now);
    assert!(is_synch_failure(
        &ue.on_downlink(&mme.authentication_request(), now)
    ));
}

#[test]
fn the_stored_pair_is_deleted_on_authentication_reject() {
    let (mut ue, mme, now) = first_challenge();
    ue.on_downlink(&[0x07, 0x54], now);
    assert!(is_synch_failure(
        &ue.on_downlink(&mme.authentication_request(), now)
    ));
}

#[test]
fn a_mismatched_replayed_capability_is_rejected_with_23() {
    let now = Instant::now();
    let mut ue = EpsNasUe::new(cfg());
    ue.initial_message(now).unwrap();
    let mut mme = Mme::new(0x21);
    ue.on_downlink(&mme.authentication_request(), now);
    let out = ue.on_downlink(&mme.security_mode_command([0xF0, 0xE0], false), now);
    assert_eq!(out.uplink, vec![vec![0x07, 0x5F, 23]]);
    assert_eq!(ue.k_enb(), None, "no context was taken into use");
}

#[test]
fn a_forged_security_mode_command_is_rejected_with_24() {
    let now = Instant::now();
    let mut ue = EpsNasUe::new(cfg());
    ue.initial_message(now).unwrap();
    let mut mme = Mme::new(0x21);
    ue.on_downlink(&mme.authentication_request(), now);
    let mut smc = mme.security_mode_command([0xE0, 0xE0], false);
    smc[2] ^= 0x10;
    assert_eq!(ue.on_downlink(&smc, now).uplink, vec![vec![0x07, 0x5F, 24]]);
}

#[test]
fn attach_reject_17_retries_after_t3411_and_t3402_after_five() {
    let t0 = Instant::now();
    let mut ue = EpsNasUe::new(cfg());
    let mut now = t0;
    for attempt in 1..=MAX_ATTACH_ATTEMPTS {
        assert!(ue.initial_message(now).is_some(), "attempt {attempt}");
        ue.on_downlink(&[0x07, 0x44, 17], now);
        assert_eq!(ue.state(), EmmState::AttemptingToAttach);
        assert!(
            ue.initial_message(now).is_none(),
            "barred while the timer runs"
        );
        let wait = if attempt < MAX_ATTACH_ATTEMPTS {
            T3411
        } else {
            T3402_DEFAULT
        };
        assert_eq!(ue.next_deadline(), Some(now + wait), "attempt {attempt}");
        now += wait;
        ue.on_timer(now);
        assert_eq!(ue.state(), EmmState::Deregistered);
    }
}

#[test]
fn attach_reject_3_invalidates_the_usim() {
    let now = Instant::now();
    let mut ue = EpsNasUe::new(cfg());
    ue.initial_message(now).unwrap();
    ue.on_downlink(&[0x07, 0x44, 3], now);
    assert_eq!(ue.state(), EmmState::NoService(3));
    assert!(ue.initial_message(now + T3402_DEFAULT).is_none());
}

#[test]
fn t3410_expiry_aborts_and_releases() {
    let now = Instant::now();
    let mut ue = EpsNasUe::new(cfg());
    ue.initial_message(now).unwrap();
    let out = ue.on_timer(now + T3410);
    assert!(out.release);
    assert_eq!(ue.state(), EmmState::AttemptingToAttach);
}

#[test]
fn a_second_attach_uses_the_guti_and_integrity_protects() {
    let (mut ue, _, _) = attach();
    // Force a new attach (the network-initiated detach is not implemented, so
    // reach the state directly), as after a power cycle with stored context.
    ue.state = EmmState::Deregistered;
    let req = ue.initial_message(Instant::now()).unwrap();
    assert_eq!(req[0], 0x17, "header type 1: integrity protected");
    let plain = &req[6..];
    assert_eq!(&plain[..2], &[0x07, 0x41]);
    assert_eq!(plain[2] >> 4, 1, "KSI of the current context");
    assert_eq!(plain[4], 0xF6, "GUTI identity");
}

#[test]
fn imei_check_digit_is_luhn() {
    // TS 23.003 Annex B example: IMEI 35-209900-176148-1.
    assert_eq!(imei_of("3520990017614800"), "352099001761481");
}

/// After a release, the next connection's messages are integrity protected
/// with the kept context but not ciphered until that connection's SMC
/// (TS 24.301 §4.4.5); the release-then-re-attach case of #238.
#[test]
fn a_new_connection_integrity_protects_but_does_not_cipher_until_its_smc() {
    let (mut ue, mut old, _) = attach();
    let now = Instant::now();
    // The MME released the connection (e.g. after an ATTACH REJECT, nextgcore
    // #473); the UE keeps its current EPS security context (§5.5.1.2.5).
    ue.state = EmmState::Deregistered;
    ue.on_released(now);

    let req = ue.initial_message(now).expect("re-attach due");
    assert_eq!(req[0], 0x17, "ATTACH REQUEST: header type 1");
    let plain = old.unprotect(&req);
    assert_eq!(&plain[..2], &[0x07, 0x41]);

    // A fresh challenge on the new connection.
    let mut mme = Mme::with_rand(0x22, [0x5A; 16]);
    let out = ue.on_downlink(&mme.authentication_request(), now);
    let resp = &out.uplink[0];
    assert_eq!(
        resp[0], 0x17,
        "AUTHENTICATION RESPONSE before this connection's SMC: header type 1, not 2"
    );
    // Not ciphered: the NAS message is readable after the 6-octet header, and
    // the MAC verifies under the kept context.
    assert_eq!(&resp[6..9], &[0x07, 0x53, 8]);
    assert_eq!(&resp[9..], &mme.xres);
    let plain = old.unprotect(resp);
    assert_eq!(&plain[..3], &[0x07, 0x53, 8]);

    // This connection's SMC establishes the secure exchange: ciphered again.
    let out = ue.on_downlink(&mme.security_mode_command([0xE0, 0xE0], false), now);
    assert_eq!(
        out.uplink[0][0], 0x47,
        "SECURITY MODE COMPLETE: header type 4"
    );
    mme.unprotect(&out.uplink[0]);
    let pti = ue.attach_pti.expect("the re-attach's PTI");
    let out = ue.on_downlink(&mme.attach_accept(pti), now);
    assert_eq!(
        out.uplink[0][0], 0x27,
        "ATTACH COMPLETE after the SMC: header type 2"
    );
    let complete = mme.unprotect(&out.uplink[0]);
    assert_eq!(&complete[..2], &[0x07, 0x43]);
}

/// §5.6.1.1 b / §5.6.1.2: only pending uplink data makes a registered idle UE
/// ask for a connection, and it asks with a SERVICE REQUEST under T3417 whose
/// COUNT keys K_eNB (TS 33.401 A.3).
#[test]
fn pending_uplink_data_starts_a_service_request() {
    let (mut ue, mut mme, now) = registered_idle();
    assert_eq!(ue.initial_message(now), None, "no data, no connection");
    ue.on_uplink_data_pending();
    let sr = ue.initial_message(now).expect("SERVICE REQUEST");
    let count = mme.verify_service_request(&sr);
    assert_eq!(ue.state(), EmmState::ServiceRequestInitiated);
    assert_eq!(ue.next_deadline(), Some(now + T3417));
    assert_eq!(ue.k_enb(), Some(derive_kenb(&mme.kasme, count)));
    assert_eq!(ue.initial_message(now), None, "one SERVICE REQUEST");
    // §5.6.1.4: the DRBs set up complete it.
    ue.on_user_plane_established();
    assert_eq!(ue.state(), EmmState::Registered);
    assert_eq!(ue.next_deadline(), None);
    assert!(
        ue.default_bearer().is_some(),
        "the bearer survives idle mode"
    );
}

/// §5.6.1.5 #9 (what a restarted MME answers): GUTI, TAI list and KSI deleted,
/// bearers deactivated locally, EMM-DEREGISTERED; the network (or T3440)
/// releases the connection and the UE attaches again with its IMSI.
#[test]
fn service_reject_9_reattaches_with_the_imsi() {
    let (mut ue, mut mme, now) = registered_idle();
    ue.on_uplink_data_pending();
    let sr = ue.initial_message(now).unwrap();
    mme.verify_service_request(&sr);
    let out = ue.on_downlink(&[0x07, 0x4E, 9], now);
    assert!(out.uplink.is_empty());
    assert_eq!(ue.state(), EmmState::Deregistered);
    assert_eq!(ue.guti(), None);
    assert_eq!(ue.default_bearer(), None);
    assert!(ue.tai_list().is_empty());
    assert_eq!(ue.k_enb(), None, "KSI deleted");
    // No network release: T3440 releases locally.
    assert_eq!(ue.next_deadline(), Some(now + T3440));
    assert!(ue.on_timer(now + T3440).release);
    ue.on_released(now + T3440);
    let req = ue.initial_message(now + T3440).expect("re-attach");
    assert_eq!(&req[..3], &[0x07, 0x41, 0x71], "plain, no KSI, EPS attach");
    assert_eq!(ue.state(), EmmState::RegisteredInitiated);
}

/// §5.6.1.5 #10: bearers deactivated locally, EMM-DEREGISTERED, a new attach
/// -- with the GUTI and the kept security context, so integrity protected.
#[test]
fn service_reject_10_reattaches_with_the_guti() {
    let (mut ue, mut mme, now) = registered_idle();
    ue.on_uplink_data_pending();
    mme.verify_service_request(&ue.initial_message(now).unwrap());
    ue.on_downlink(&[0x07, 0x4E, 10], now);
    assert_eq!(ue.state(), EmmState::Deregistered);
    assert_eq!(ue.default_bearer(), None);
    assert!(ue.guti().is_some());
    ue.on_released(now);
    assert_eq!(ue.next_deadline(), None, "the release stops T3440");
    let req = ue.initial_message(now).expect("re-attach");
    assert_eq!(req[0], 0x17, "integrity protected with the kept context");
    let plain = mme.unprotect(&req);
    assert_eq!(&plain[..2], &[0x07, 0x41]);
    assert_eq!(plain[4], 0xF6, "GUTI identity");
}

/// §5.6.1.5 #40: no EPS bearer context at the network; the same re-attach.
#[test]
fn service_reject_40_reattaches() {
    let (mut ue, mut mme, now) = registered_idle();
    ue.on_uplink_data_pending();
    mme.verify_service_request(&ue.initial_message(now).unwrap());
    let out = ue.on_downlink(&mme.protect(&[0x07, 0x4E, 40], 2), now);
    assert!(out.uplink.is_empty());
    assert_eq!(ue.state(), EmmState::Deregistered);
    assert_eq!(ue.default_bearer(), None);
}

/// §5.6.1.6 c: T3417 expiry (an MME that accepts without setting up the
/// bearers) aborts the procedure and releases locally; the UE stays
/// registered and the next uplink data tries again.
#[test]
fn t3417_expiry_aborts_and_the_next_data_retries() {
    let (mut ue, mut mme, now) = registered_idle();
    ue.on_uplink_data_pending();
    mme.verify_service_request(&ue.initial_message(now).unwrap());
    let out = ue.on_timer(now + T3417);
    assert!(out.release);
    assert_eq!(ue.state(), EmmState::Registered);
    ue.on_released(now + T3417);
    ue.on_uplink_data_pending();
    let sr = ue
        .initial_message(now + T3417)
        .expect("a new SERVICE REQUEST");
    mme.verify_service_request(&sr);
}

/// §5.6.1.6 a: a connection lost before the bearers are set up aborts it.
#[test]
fn a_release_during_the_service_request_aborts_it() {
    let (mut ue, mut mme, now) = registered_idle();
    ue.on_uplink_data_pending();
    mme.verify_service_request(&ue.initial_message(now).unwrap());
    ue.on_released(now);
    assert_eq!(ue.state(), EmmState::Registered);
    assert_eq!(ue.next_deadline(), None);
}

/// §5.6.1.5 #22 with T3346: no new SERVICE REQUEST until T3346 expires.
#[test]
fn service_reject_22_bars_service_requests_for_t3346() {
    let (mut ue, mut mme, now) = registered_idle();
    ue.on_uplink_data_pending();
    mme.verify_service_request(&ue.initial_message(now).unwrap());
    // T3346 TLV: IEI 5F, length 1, GPRS timer 2 s (unit 2 s, value 1).
    ue.on_downlink(&[0x07, 0x4E, 22, 0x5F, 0x01, 0x01], now);
    assert_eq!(ue.state(), EmmState::Registered);
    ue.on_released(now);
    ue.on_uplink_data_pending();
    assert_eq!(ue.initial_message(now), None, "barred by T3346");
    let later = now + Duration::from_secs(2);
    ue.on_timer(later);
    assert!(ue.initial_message(later).is_some(), "T3346 expired");
}

/// A SERVICE REJECT outside a service request is not acted on.
#[test]
fn an_unsolicited_service_reject_is_ignored() {
    let (mut ue, _mme, now) = registered_idle();
    ue.on_downlink(&[0x07, 0x4E, 9], now);
    assert_eq!(ue.state(), EmmState::Registered);
    assert!(ue.guti().is_some());
}

/// DETACH REQUEST (network to UE) as the MME sends it: detach type, then the
/// optional EMM cause (IEI 53).
fn detach_request(detach_type: u8, cause: Option<u8>) -> Vec<u8> {
    let mut m = vec![0x07, 0x45, detach_type];
    if let Some(c) = cause {
        m.extend_from_slice(&[0x53, c]);
    }
    m
}

/// §5.5.2.3.2 "re-attach required" (what nextgcore mmed sends after a PGW-U
/// or SGW restart, with cause #10): DETACH ACCEPT under the current context,
/// bearers deactivated locally, EMM-DEREGISTERED; the attach waits for the
/// release of the connection, then goes out with the GUTI.
#[test]
fn detach_reattach_required_accepts_and_reattaches_after_the_release() {
    let (mut ue, mut mme, _) = attach();
    let now = Instant::now();
    let out = ue.on_downlink(&mme.protect(&detach_request(1, Some(10)), 2), now);
    assert_eq!(out.uplink.len(), 1);
    assert_eq!(out.uplink[0][0], 0x27, "integrity protected and ciphered");
    assert_eq!(
        mme.unprotect(&out.uplink[0]),
        vec![0x07, 0x46],
        "DETACH ACCEPT"
    );
    assert_eq!(ue.state(), EmmState::Deregistered);
    assert_eq!(ue.default_bearer(), None);
    assert!(
        ue.guti().is_some(),
        "the cause is ignored for re-attach required"
    );
    // The UE's RRC only asks NAS for an initial message in RRC_IDLE; the
    // network releases (here) or T3440 does.
    assert_eq!(ue.next_deadline(), Some(now + T3440));
    ue.on_released(now);
    assert_eq!(ue.next_deadline(), None);
    let req = ue.initial_message(now).expect("re-attach");
    assert_eq!(req[0], 0x17, "integrity protected with the kept context");
    let plain = mme.unprotect(&req);
    assert_eq!(&plain[..2], &[0x07, 0x41]);
    assert_eq!(plain[4], 0xF6, "GUTI identity");
}

/// T3440: a network that does not release after the detach.
#[test]
fn detach_without_a_network_release_releases_locally_on_t3440() {
    let (mut ue, mut mme, _) = attach();
    let now = Instant::now();
    ue.on_downlink(&mme.protect(&detach_request(1, None), 2), now);
    assert!(!ue.on_timer(now + T3440 - Duration::from_millis(1)).release);
    assert!(ue.on_timer(now + T3440).release);
}

/// §5.5.2.3.2: "re-attach not required" with #7 -- the USIM is invalid for
/// EPS services; no further attach.
#[test]
fn detach_not_required_with_7_leaves_no_service() {
    let (mut ue, mut mme, _) = attach();
    let now = Instant::now();
    let out = ue.on_downlink(&mme.protect(&detach_request(2, Some(7)), 2), now);
    assert_eq!(mme.unprotect(&out.uplink[0]), vec![0x07, 0x46]);
    assert_eq!(ue.state(), EmmState::NoService(7));
    assert_eq!(ue.default_bearer(), None);
    ue.on_released(now);
    assert_eq!(ue.initial_message(now), None);
}

/// "re-attach not required" with no cause: EMM-DEREGISTERED.NORMAL-SERVICE.
#[test]
fn detach_not_required_without_a_cause_deregisters() {
    let (mut ue, mut mme, _) = attach();
    let now = Instant::now();
    ue.on_downlink(&mme.protect(&detach_request(2, None), 2), now);
    assert_eq!(ue.state(), EmmState::Deregistered);
    assert_eq!(ue.default_bearer(), None);
}

/// IMSI detach concerns non-EPS services only: accepted, EPS state kept.
#[test]
fn imsi_detach_keeps_the_eps_registration() {
    let (mut ue, mut mme, _) = attach();
    let now = Instant::now();
    let out = ue.on_downlink(&mme.protect(&detach_request(3, None), 2), now);
    assert_eq!(mme.unprotect(&out.uplink[0]), vec![0x07, 0x46]);
    assert_eq!(ue.state(), EmmState::Registered);
    assert!(ue.default_bearer().is_some());
}

/// A DETACH REQUEST must be integrity protected (§4.4.4.2): a plain one is
/// discarded.
#[test]
fn a_plain_detach_request_is_discarded() {
    let (mut ue, _mme, _) = attach();
    let out = ue.on_downlink(&detach_request(1, None), Instant::now());
    assert!(out.uplink.is_empty());
    assert_eq!(ue.state(), EmmState::Registered);
}

/// A detach that collides with the UE's service request wins (§5.5.2.3.4).
#[test]
fn a_detach_during_a_service_request_aborts_it() {
    let (mut ue, mut mme, now) = registered_idle();
    ue.on_uplink_data_pending();
    mme.verify_service_request(&ue.initial_message(now).unwrap());
    let out = ue.on_downlink(&mme.protect(&detach_request(1, None), 2), now);
    assert_eq!(mme.unprotect(&out.uplink[0]), vec![0x07, 0x46]);
    assert_eq!(ue.state(), EmmState::Deregistered);
    assert_eq!(ue.next_deadline(), Some(now + T3440), "T3417 stopped");
}

/// §4.4.2.3 item 2: on a connection a SERVICE REQUEST set up, the lower layers'
/// "user plane radio bearers set up" indication establishes the secure exchange,
/// so the next uplink EMM message -- here the DETACH ACCEPT answering a network
/// DETACH REQUEST -- is integrity protected AND ciphered (header type 2, octet
/// 0x27), not integrity protected only (0x17) (#318).
#[test]
fn uplink_nas_is_ciphered_once_a_service_request_set_up_the_bearers() {
    let (mut ue, mut mme, now) = registered_idle();
    ue.on_uplink_data_pending();
    mme.verify_service_request(&ue.initial_message(now).unwrap());
    ue.on_user_plane_established();
    assert_eq!(ue.state(), EmmState::Registered);
    let out = ue.on_downlink(&mme.protect(&detach_request(1, None), 2), now);
    let accept = &out.uplink[0];
    assert_eq!(
        accept[0], 0x27,
        "DETACH ACCEPT: integrity protected and ciphered"
    );
    assert_ne!(
        &accept[6..],
        &[0x07, 0x46],
        "the message body is not in clear"
    );
    assert_eq!(mme.unprotect(accept), vec![0x07, 0x46]);
}

/// The secure exchange is per connection (§4.4.5): the release that ends the
/// service-request connection ends it too, and the next SERVICE REQUEST
/// connection is integrity only until its own bearers are set up.
#[test]
fn the_service_request_secure_exchange_ends_with_the_connection() {
    let (mut ue, mut mme, now) = registered_idle();
    ue.on_uplink_data_pending();
    mme.verify_service_request(&ue.initial_message(now).unwrap());
    ue.on_user_plane_established();
    ue.on_released(now);
    ue.on_uplink_data_pending();
    mme.verify_service_request(&ue.initial_message(now).unwrap());
    // Bearers not yet up: a DETACH ACCEPT here is integrity protected only.
    let out = ue.on_downlink(&mme.protect(&detach_request(1, None), 2), now);
    assert_eq!(
        out.uplink[0][0], 0x17,
        "no secure exchange before the bearers"
    );
    assert_eq!(mme.unprotect(&out.uplink[0]), vec![0x07, 0x46]);
}

/// #351, TS 24.301 §5.6.2.2.1: paged with the S-TMSI, a registered idle UE
/// answers with a SERVICE REQUEST (no uplink data needed) whose RRC cause is
/// mt-Access; T3346 is stopped by the paging.
#[test]
fn an_s_tmsi_paging_is_answered_with_a_service_request() {
    let (mut ue, mut mme, now) = registered_idle();
    assert_eq!(ue.initial_message(now), None);
    assert!(ue.on_paging(PagedWith::STmsi));
    let sr = ue
        .initial_message(now)
        .expect("SERVICE REQUEST for the paging");
    mme.verify_service_request(&sr);
    assert_eq!(ue.state(), EmmState::ServiceRequestInitiated);
    assert!(ue.service_request_answers_paging());
    assert_eq!(ue.initial_message(now), None, "one SERVICE REQUEST");

    // A data-triggered one is not mt-Access.
    ue.on_user_plane_established();
    ue.on_released(now);
    ue.on_uplink_data_pending();
    ue.initial_message(now).unwrap();
    assert!(!ue.service_request_answers_paging());
}

/// #351 review: a paging response that never went out does not outlive the
/// connection attempt. Kept, it would label the next SERVICE REQUEST a paging
/// response (mt-Access) and send one with no uplink data pending.
#[test]
fn an_unanswered_paging_is_dropped_when_the_connection_goes() {
    let (mut ue, mut mme, now) = registered_idle();
    assert!(ue.on_paging(PagedWith::STmsi));
    // The RRC establishment fails before the SERVICE REQUEST is built.
    ue.on_released(now);
    assert_eq!(
        ue.initial_message(now),
        None,
        "no paging response owed any more"
    );
    ue.on_uplink_data_pending();
    mme.verify_service_request(&ue.initial_message(now).expect("data SERVICE REQUEST"));
    assert!(
        !ue.service_request_answers_paging(),
        "a data-triggered request is not a paging response"
    );
}

/// §5.6.2.2.1: T3346 does not hold back a paging response; it is stopped.
#[test]
fn a_paging_is_answered_under_t3346() {
    let (mut ue, mut mme, now) = registered_idle();
    ue.on_uplink_data_pending();
    mme.verify_service_request(&ue.initial_message(now).unwrap());
    ue.on_downlink(&[0x07, 0x4E, 22, 0x5F, 0x01, 0x01], now);
    ue.on_released(now);
    ue.on_uplink_data_pending();
    assert_eq!(ue.initial_message(now), None, "barred by T3346");
    assert!(ue.on_paging(PagedWith::STmsi));
    assert!(ue.initial_message(now).is_some(), "the paging is answered");
}

/// A UE that is not EMM-REGISTERED (here: attaching) does not answer an S-TMSI
/// paging.
#[test]
fn an_s_tmsi_paging_is_not_answered_outside_emm_registered() {
    let mut ue = EpsNasUe::new(cfg());
    let now = Instant::now();
    ue.initial_message(now).unwrap();
    assert_eq!(ue.state(), EmmState::RegisteredInitiated);
    assert!(!ue.on_paging(PagedWith::STmsi));
}

/// #351, §5.6.2.2.2: paged with the IMSI, the UE detaches locally (bearers,
/// GUTI, TAI list and KSI gone) and attaches again, with the IMSI.
#[test]
fn an_imsi_paging_detaches_locally_and_reattaches() {
    let (mut ue, _mme, now) = registered_idle();
    assert!(ue.on_paging(PagedWith::Imsi));
    assert_eq!(ue.state(), EmmState::Deregistered);
    assert_eq!(ue.guti(), None);
    assert_eq!(ue.default_bearer(), None);
    assert!(ue.tai_list().is_empty());
    assert_eq!(ue.k_enb(), None, "KSI deleted");
    let req = ue.initial_message(now).expect("re-attach");
    assert_eq!(&req[..3], &[0x07, 0x41, 0x71], "plain, no KSI, EPS attach");
}

// ---- #365: UE-initiated detach (TS 24.301 §5.5.2.2) ----

/// The plain DETACH REQUEST inside a protected uplink PDU.
fn detach_request_body(mme: &mut Mme, pdu: &[u8]) -> Vec<u8> {
    let plain = mme.unprotect(pdu);
    assert_eq!(&plain[..2], &[0x07, 0x45], "DETACH REQUEST");
    plain
}

/// §5.5.2.2.1 / §8.2.11.1, connected: an integrity-protected DETACH REQUEST
/// with "EPS detach" and the GUTI (detach type in bits 4-1 of octet 3, the KSI
/// in bits 8-5), EMM-DEREGISTERED-INITIATED under T3421. The DETACH ACCEPT then
/// stops T3421, deactivates the default bearer and leaves the UE
/// EMM-DEREGISTERED with no attach to follow (§5.5.2.2.2).
#[test]
fn a_connected_detach_is_accepted_and_the_ue_is_deregistered() {
    let (mut ue, mut mme, _) = attach();
    let now = Instant::now();
    let out = ue.detach(false, true, now);
    let plain = detach_request_body(&mut mme, &out.uplink[0]);
    assert_eq!(plain[2] & 0x0F, 0x01, "EPS detach, not switch off");
    assert_eq!(plain[2] >> 4, 1, "the KSI of the current context");
    assert_eq!(plain[3], 11, "EPS mobile identity LV: a GUTI");
    assert_eq!(plain[4] & 0x07, 6, "type of identity GUTI");
    assert_eq!(ue.state(), EmmState::DeregisteredInitiated);
    assert_eq!(ue.next_deadline(), Some(now + T3421));

    let accept = mme.protect(&[0x07, 0x46], 2);
    let out = ue.on_downlink(&accept, now);
    assert!(out.uplink.is_empty());
    assert_eq!(ue.state(), EmmState::Detached);
    assert!(ue.default_bearer().is_none(), "bearer contexts deactivated");
    assert_eq!(ue.next_deadline(), None, "T3421 stopped");
    assert_eq!(ue.initial_message(now), None, "no attach follows a detach");
}

/// In EMM-IDLE the DETACH REQUEST waits for a connection: it is the next
/// initial NAS message, integrity protected and not ciphered (§4.4.4.2).
#[test]
fn an_idle_detach_goes_out_as_the_initial_message() {
    let (mut ue, mut mme, now) = registered_idle();
    let out = ue.detach(false, false, now);
    assert!(out.uplink.is_empty(), "nothing to send it on yet");
    assert!(ue.detach_pending());
    let pdu = ue.initial_message(now).expect("the DETACH REQUEST");
    assert_eq!(pdu[0] >> 4, 1, "integrity protected, not ciphered");
    detach_request_body(&mut mme, &pdu);
    assert_eq!(ue.state(), EmmState::DeregisteredInitiated);
    assert!(!ue.detach_pending());
}

/// §5.5.2.2.4 c): four retransmissions on T3421, then the procedure is aborted
/// and the UE is EMM-DEREGISTERED.
#[test]
fn t3421_retransmits_four_times_then_gives_up() {
    let (mut ue, mut mme, _) = attach();
    let mut now = Instant::now();
    let first = ue.detach(false, true, now);
    let first = detach_request_body(&mut mme, &first.uplink[0]);
    // §5.5.2.2.4 c): "On the first four expiries of the timer, the UE shall
    // retransmit the DETACH REQUEST message". The literal, not the constant, so
    // a changed constant fails here (#369 review).
    assert_eq!(MAX_DETACH_RETRANSMISSIONS, 4);
    for n in 1..=4 {
        now += T3421;
        let out = ue.on_timer(now);
        assert_eq!(out.uplink.len(), 1, "retransmission {n}");
        assert_eq!(detach_request_body(&mut mme, &out.uplink[0]), first);
        assert_eq!(ue.state(), EmmState::DeregisteredInitiated);
    }
    now += T3421;
    let out = ue.on_timer(now);
    assert!(out.uplink.is_empty());
    assert!(out.release);
    assert_eq!(ue.state(), EmmState::Detached);
}

/// A switch-off detach gets no DETACH ACCEPT (§5.5.2.2.2): the UE is
/// EMM-DEREGISTERED as soon as the request is out, with no T3421.
#[test]
fn a_switch_off_detach_needs_no_accept() {
    let (mut ue, mut mme, _) = attach();
    let now = Instant::now();
    let out = ue.detach(true, true, now);
    let plain = detach_request_body(&mut mme, &out.uplink[0]);
    assert_eq!(plain[2] & 0x0F, 0x09, "switch off, EPS detach");
    assert_eq!(ue.state(), EmmState::Detached);
    assert_eq!(ue.next_deadline(), None);
}

/// §5.5.2.2.4 b): the connection lost before DETACH ACCEPT aborts the detach;
/// for "EPS detach" the UE enters EMM-DEREGISTERED.
#[test]
fn a_lost_connection_before_the_accept_still_detaches() {
    let (mut ue, _mme, _) = attach();
    let now = Instant::now();
    ue.detach(false, true, now);
    ue.on_released(now);
    assert_eq!(ue.state(), EmmState::Detached);
    assert_eq!(ue.next_deadline(), None);
}

/// §5.5.2.2.4 d): a network DETACH REQUEST (not switch off) during this UE's
/// detach is answered with DETACH ACCEPT, and "the UE need not initiate the
/// attach".
#[test]
fn a_colliding_network_detach_is_accepted_without_reattach() {
    let (mut ue, mut mme, _) = attach();
    let now = Instant::now();
    let out = ue.detach(false, true, now);
    detach_request_body(&mut mme, &out.uplink[0]);
    // DETACH REQUEST, network to UE: re-attach required.
    let req = mme.protect(&[0x07, 0x45, 0x01], 2);
    let out = ue.on_downlink(&req, now);
    let accept = mme.unprotect(&out.uplink[0]);
    assert_eq!(&accept[..2], &[0x07, 0x46], "DETACH ACCEPT");
    assert_eq!(ue.state(), EmmState::Detached);
    assert_eq!(ue.initial_message(now), None, "no re-attach");
}

/// Not attached: nothing to detach from, and no attach follows.
#[test]
fn a_detach_while_not_attached_stops_attaching() {
    let mut ue = EpsNasUe::new(cfg());
    let out = ue.detach(false, false, Instant::now());
    assert!(out.uplink.is_empty());
    assert_eq!(ue.state(), EmmState::Detached);
    assert_eq!(ue.initial_message(Instant::now()), None);
}

/// #369 review, §5.5.2.2.4 d): a network "IMSI detach" during this UE's "EPS
/// detach" is answered, and the UE's own detach goes on: still
/// EMM-DEREGISTERED-INITIATED, T3421 running, completed by its DETACH ACCEPT.
#[test]
fn a_network_imsi_detach_during_the_ues_detach_progresses_both() {
    let (mut ue, mut mme, _) = attach();
    let now = Instant::now();
    let out = ue.detach(false, true, now);
    detach_request_body(&mut mme, &out.uplink[0]);
    let req = mme.protect(&[0x07, 0x45, 0x03], 2);
    let out = ue.on_downlink(&req, now);
    let accept = mme.unprotect(&out.uplink[0]);
    assert_eq!(
        &accept[..2],
        &[0x07, 0x46],
        "DETACH ACCEPT for the IMSI detach"
    );
    assert_eq!(
        ue.state(),
        EmmState::DeregisteredInitiated,
        "own detach continues"
    );
    assert_eq!(ue.next_deadline(), Some(now + T3421));
    let out = ue.on_downlink(&mme.protect(&[0x07, 0x46], 2), now);
    assert!(out.uplink.is_empty());
    assert_eq!(ue.state(), EmmState::Detached);
}

/// #369 review, §5.5.2.2.1: a detach during an attach sends the DETACH REQUEST,
/// with T3421, on the attach's connection.
#[test]
fn a_detach_during_an_attach_sends_the_request() {
    let now = Instant::now();
    let mut ue = EpsNasUe::new(cfg());
    ue.initial_message(now).expect("ATTACH REQUEST");
    assert_eq!(ue.state(), EmmState::RegisteredInitiated);
    let out = ue.detach(false, true, now);
    assert_eq!(
        &out.uplink[0][..2],
        &[0x07, 0x45],
        "plain DETACH REQUEST: no context yet"
    );
    assert_eq!(ue.state(), EmmState::DeregisteredInitiated);
    assert_eq!(ue.next_deadline(), Some(now + T3421), "T3421, not T3410");
}

// ---- #382: mandatory-IE errors (TS 24.301 §7.5.1, §7.5.3) ----

/// The exact EMM STATUS #96 octets (§8.2.14): PD 7 / plain header, type 0x60,
/// EMM cause 96.
const EMM_STATUS_96: [u8; 3] = [0x07, 0x60, 0x60];

/// #382, §7.5.1 / §7.5.2: an integrity-protected EMM message whose mandatory
/// part does not decode is answered with EMM STATUS #96, protected and ciphered
/// like any uplink on the secure exchange. The procedure state is unchanged.
#[test]
fn a_protected_emm_message_with_a_bad_mandatory_ie_is_answered_with_status_96() {
    for (name, plain) in [
        ("IDENTITY REQUEST", vec![0x07, 0x55]),
        ("DETACH REQUEST", vec![0x07, 0x45]),
        ("ATTACH REJECT", vec![0x07, 0x44]),
        ("SERVICE REJECT", vec![0x07, 0x4E]),
        ("AUTHENTICATION REQUEST", vec![0x07, 0x52, 0x01]),
    ] {
        let (mut ue, mut mme, _) = attach();
        let now = Instant::now();
        let out = ue.on_downlink(&mme.protect(&plain, 2), now);
        assert_eq!(out.uplink.len(), 1, "{name}: one uplink");
        assert_eq!(
            out.uplink[0][0] >> 4,
            2,
            "{name}: integrity protected and ciphered"
        );
        assert_eq!(mme.unprotect(&out.uplink[0]), EMM_STATUS_96, "{name}");
        assert_eq!(ue.state(), EmmState::Registered, "{name}: state unchanged");
    }
}

/// #382, §7.5.1 with §4.4.4.2: a plain message the UE processes unprotected
/// gets EMM STATUS #96 (plain: no context yet); a plain message off the
/// allow-list is discarded unread.
#[test]
fn a_plain_emm_message_with_a_bad_mandatory_ie_is_answered_only_when_allowed() {
    let now = Instant::now();
    let mut ue = EpsNasUe::new(cfg());
    ue.initial_message(now).expect("ATTACH REQUEST");
    let out = ue.on_downlink(&[0x07, 0x55], now);
    assert_eq!(out.uplink, vec![EMM_STATUS_96.to_vec()], "IDENTITY REQUEST");
    assert_eq!(ue.state(), EmmState::RegisteredInitiated);
    assert_eq!(
        ue.on_downlink(&[0x07, 0x45], now),
        NasOutput::default(),
        "a plain DETACH REQUEST is not on the allow-list"
    );
}

/// #382, §7.5.1 / §7.5.2 (as #384 for 5GS): a new-context SECURITY MODE
/// COMMAND whose mandatory part does not decode is answered with EMM STATUS
/// #96 -- plain, since no context exists yet.
#[test]
fn an_undecodable_security_mode_command_is_answered_with_status_96() {
    let now = Instant::now();
    let mut ue = EpsNasUe::new(cfg());
    ue.initial_message(now).expect("ATTACH REQUEST");
    let mut mme = Mme::new(0x21);
    ue.on_downlink(&mme.authentication_request(), now);
    let out = ue.on_downlink(&mme.protect(&[0x07, 0x5D], 3), now);
    assert_eq!(out.uplink, vec![EMM_STATUS_96.to_vec()]);
    assert_eq!(ue.state(), EmmState::RegisteredInitiated);
}

/// #382, §7.5.1 / §7.5.3: a protected ESM message with a mandatory-IE error is
/// answered with ESM STATUS #96, except that an ACTIVATE DEFAULT EPS BEARER
/// CONTEXT REQUEST gets ACTIVATE DEFAULT EPS BEARER CONTEXT REJECT #96
/// (§7.5.3 a)), both with the message's EBI and PTI.
#[test]
fn a_protected_esm_message_with_a_bad_mandatory_ie_is_answered_with_96() {
    for (name, esm, reply) in [
        // PDN CONNECTIVITY REJECT without its ESM cause
        (
            "PDN CONNECTIVITY REJECT",
            vec![0x02, 0x07, 0xD1],
            vec![0x02, 0x07, 0xE8, 0x60],
        ),
        // ACTIVATE DEFAULT EPS BEARER CONTEXT REQUEST, EBI 6, without its EPS QoS
        (
            "ACTIVATE DEFAULT EPS BEARER CONTEXT REQUEST",
            vec![0x62, 0x00, 0xC1],
            vec![0x62, 0x00, 0xC3, 0x60],
        ),
        // An imperative-part error: the message type is missing
        (
            "truncated header",
            vec![0x52, 0x03],
            vec![0x52, 0x03, 0xE8, 0x60],
        ),
    ] {
        let (mut ue, mut mme, _) = attach();
        let now = Instant::now();
        let out = ue.on_downlink(&mme.protect(&esm, 2), now);
        assert_eq!(out.uplink.len(), 1, "{name}");
        assert_eq!(mme.unprotect(&out.uplink[0]), reply, "{name}");
        assert_eq!(ue.default_bearer().unwrap().ebi, 5, "{name}: bearer kept");
    }
}

/// #382, §7.5.2 / §7.5.3 a): an ATTACH ACCEPT whose ESM container is an
/// undecodable ACTIVATE DEFAULT EPS BEARER CONTEXT REQUEST is completed with
/// ATTACH COMPLETE carrying ACTIVATE DEFAULT EPS BEARER CONTEXT REJECT #96 (it
/// was ESM STATUS #95), and no bearer is activated.
#[test]
fn an_attach_accept_with_an_undecodable_default_bearer_request_rejects_it_with_96() {
    let now = Instant::now();
    let mut ue = EpsNasUe::new(cfg());
    let req = ue.initial_message(now).expect("attach due");
    let pti = req[18];
    let mut mme = Mme::new(0x21);
    ue.on_downlink(&mme.authentication_request(), now);
    let out = ue.on_downlink(&mme.security_mode_command([0xE0, 0xE0], false), now);
    mme.unprotect(&out.uplink[0]);
    // ESM: EBI 5, the PTI, 0xC1 -- and nothing of its mandatory part
    let esm = [0x52, pti, 0xC1];
    let mut m = vec![
        0x07, 0x42, 0x01, 0x49, 0x06, 0x00, 0x99, 0xF9, 0x07, 0x00, 0x01,
    ];
    m.extend_from_slice(&(esm.len() as u16).to_be_bytes());
    m.extend_from_slice(&esm);
    let out = ue.on_downlink(&mme.protect(&m, 2), now);
    let complete = mme.unprotect(&out.uplink[0]);
    assert_eq!(&complete[..2], &[0x07, 0x43], "ATTACH COMPLETE");
    assert_eq!(&complete[2..], &[0x00, 0x04, 0x52, pti, 0xC3, 0x60]);
    assert!(ue.default_bearer().is_none());
}

/// #400 review, §4.4.4.2: once the secure exchange is established, a plain
/// message is discarded unread -- neither answered with EMM STATUS #96 when it
/// does not decode, nor processed when it does (an AUTHENTICATION REJECT here,
/// which would otherwise invalidate the USIM).
#[test]
fn a_plain_message_on_a_secure_exchange_is_discarded() {
    let (mut ue, _mme, _) = attach();
    let now = Instant::now();
    assert_eq!(ue.on_downlink(&[0x07, 0x55], now), NasOutput::default());
    assert_eq!(ue.on_downlink(&[0x07, 0x54], now), NasOutput::default());
    assert_eq!(
        ue.state(),
        EmmState::Registered,
        "the AUTHENTICATION REJECT was not processed"
    );
}

// ============================================================================
// Dedicated EPS bearers (TS 24.301 §6.4.2-6.4.4, nextgsim #283)
// ============================================================================

/// The TFT of a SIP-like dedicated bearer: one uplink filter, precedence
/// 0x10, UDP to 10.45.0.1/32 port 5060 (TS 24.008 Table 10.5.162).
fn sip_tft() -> Vec<u8> {
    vec![
        0x21, 0x21, 0x10, 0x0E, 0x10, 0x0A, 0x2D, 0x00, 0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0x30, 0x11,
        0x50, 0x13, 0xC4,
    ]
}

/// ACTIVATE DEDICATED EPS BEARER CONTEXT REQUEST (Table 8.3.3.1), hand-built:
/// EBI, PTI 0, C5, linked EBI (bits 4-1), LV EPS QoS (QCI 1), LV TFT.
fn activate_dedicated(ebi: u8, linked: u8, tft: &[u8]) -> Vec<u8> {
    let mut m = vec![(ebi << 4) | 0x02, 0x00, 0xC5, linked & 0x0F, 0x01, 0x01];
    m.push(tft.len() as u8);
    m.extend_from_slice(tft);
    m
}

fn udp4(dst: [u8; 4], dport: u16) -> Vec<u8> {
    let mut p = vec![
        0x45, 0x00, 0x00, 0x1C, 0, 0, 0, 0, 64, 17, 0, 0, 10, 45, 0, 2,
    ];
    p.extend_from_slice(&dst);
    p.extend_from_slice(&40000u16.to_be_bytes());
    p.extend_from_slice(&dport.to_be_bytes());
    p.extend_from_slice(&[0, 8, 0, 0]);
    p
}

/// The UE's (unprotected) reply to a protected ESM request from the MME.
fn esm_reply(ue: &mut EpsNasUe, mme: &mut Mme, esm: &[u8]) -> Vec<u8> {
    let out = ue.on_downlink(&mme.protect(esm, 2), Instant::now());
    assert_eq!(out.uplink.len(), 1, "one reply");
    mme.unprotect(&out.uplink[0])
}

/// #283, §6.4.2.3: the UE accepts a dedicated bearer linked to its default
/// bearer, keeps its TFT, and classifies uplink by it: a packet that matches
/// the uplink filter goes on the dedicated bearer, anything else on the
/// default one.
#[test]
fn a_dedicated_bearer_is_accepted_and_classifies_uplink_by_its_tft() {
    let (mut ue, mut mme, _) = attach();
    let reply = esm_reply(&mut ue, &mut mme, &activate_dedicated(6, 5, &sip_tft()));
    assert_eq!(reply, vec![0x62, 0x00, 0xC6], "ACCEPT, EBI 6, PTI 0");
    let b = &ue.dedicated_bearers()[0];
    assert_eq!((b.ebi, b.linked_ebi, b.qci, b.filters.len()), (6, 5, 1, 1));
    assert_eq!(ue.uplink_bearer_for(&udp4([10, 45, 0, 1], 5060)), Some(6));
    assert_eq!(ue.uplink_bearer_for(&udp4([10, 45, 0, 1], 80)), Some(5));
    assert_eq!(ue.uplink_bearer_for(&[0x00, 0x01]), Some(5), "not IP");
}

/// #283 with #382, §7.5.3 b)-d): a dedicated activation or a modification
/// whose mandatory part does not decode is rejected with #96; a deactivation
/// releases the bearer and is accepted.
#[test]
fn dedicated_bearer_messages_with_a_bad_mandatory_ie_follow_7_5_3() {
    let (mut ue, mut mme, _) = attach();
    // ACTIVATE DEDICATED, EBI 6, linked 5, without its EPS QoS and TFT
    let reply = esm_reply(&mut ue, &mut mme, &[0x62, 0x00, 0xC5, 0x05]);
    assert_eq!(reply, vec![0x62, 0x00, 0xC7, 0x60]);
    assert!(ue.dedicated_bearers().is_empty());

    esm_reply(&mut ue, &mut mme, &activate_dedicated(6, 5, &sip_tft()));
    // MODIFY with a TFT IE whose length runs past the message
    let reply = esm_reply(&mut ue, &mut mme, &[0x62, 0x00, 0xC9, 0x36, 0x05, 0x21]);
    assert_eq!(reply, vec![0x62, 0x00, 0xCB, 0x60]);
    assert_eq!(ue.dedicated_bearers().len(), 1, "kept");
    // DEACTIVATE without its ESM cause
    let reply = esm_reply(&mut ue, &mut mme, &[0x62, 0x00, 0xCD]);
    assert_eq!(reply, vec![0x62, 0x00, 0xCE]);
    assert!(ue.dedicated_bearers().is_empty(), "released");
    assert_eq!(ue.default_bearer().unwrap().ebi, 5);
}

/// TS 23.401 §5.3.3 / TS 24.008 §10.5.6.12: among the uplink filters that
/// match, the lowest precedence value picks the bearer, across bearers.
#[test]
fn the_matching_filter_with_the_lowest_precedence_value_wins() {
    let (mut ue, mut mme, _) = attach();
    // EBI 7 first: any UDP, precedence 0x20.
    let udp = [0x21, 0x21, 0x20, 0x02, 0x30, 0x11];
    esm_reply(&mut ue, &mut mme, &activate_dedicated(7, 5, &udp));
    esm_reply(&mut ue, &mut mme, &activate_dedicated(6, 5, &sip_tft()));
    assert_eq!(ue.uplink_bearer_for(&udp4([10, 45, 0, 1], 5060)), Some(6));
    assert_eq!(ue.uplink_bearer_for(&udp4([10, 45, 0, 1], 53)), Some(7));
}

/// #283, §6.4.2.4 / §6.4.2.5 c): REJECT with the causes the spec names.
#[test]
fn a_dedicated_bearer_request_is_rejected_per_6_4_2_4() {
    let (mut ue, mut mme, _) = attach();
    // c) no default bearer with the linked EBI: #43
    let reply = esm_reply(&mut ue, &mut mme, &activate_dedicated(6, 9, &sip_tft()));
    assert_eq!(reply, vec![0x62, 0x00, 0xC7, 43]);
    // a) an operation other than "create new TFT": #41
    let mut add = sip_tft();
    add[0] = 0x61;
    assert_eq!(
        esm_reply(&mut ue, &mut mme, &activate_dedicated(6, 5, &add))[3],
        41
    );
    // b) create with an empty packet filter list: #42
    assert_eq!(
        esm_reply(&mut ue, &mut mme, &activate_dedicated(6, 5, &[0x20]))[3],
        42
    );
    // c) 2) no uplink filter: #44
    let mut down = sip_tft();
    down[1] = 0x11;
    assert_eq!(
        esm_reply(&mut ue, &mut mme, &activate_dedicated(6, 5, &down))[3],
        44
    );
    // d) 3) a reserved component type: #45
    let bad = [0x21, 0x21, 0x00, 0x02, 0x99, 0x00];
    assert_eq!(
        esm_reply(&mut ue, &mut mme, &activate_dedicated(6, 5, &bad))[3],
        45
    );
    assert!(ue.dedicated_bearers().is_empty());
}

/// #283, §6.4.2.5 b): a request for the EBI of an active dedicated bearer
/// replaces it; §6.4.2.4 d) 2): a filter of another dedicated bearer with the
/// same precedence is deleted.
#[test]
fn a_reused_dedicated_ebi_replaces_the_bearer() {
    let (mut ue, mut mme, _) = attach();
    esm_reply(&mut ue, &mut mme, &activate_dedicated(6, 5, &sip_tft()));
    // EBI 7 with the same precedence 0x10, TCP
    let tcp = [0x21, 0x21, 0x10, 0x02, 0x30, 0x06];
    esm_reply(&mut ue, &mut mme, &activate_dedicated(7, 5, &tcp));
    assert!(
        ue.dedicated_bearers()[0].filters.is_empty(),
        "EBI 6's filter with precedence 0x10 is deleted"
    );
    esm_reply(&mut ue, &mut mme, &activate_dedicated(7, 5, &sip_tft()));
    assert_eq!(ue.dedicated_bearers().len(), 2);
    let seven = ue.dedicated_bearers().iter().find(|b| b.ebi == 7).unwrap();
    assert_eq!(
        seven.filters,
        crate::tft::Tft::decode(&sip_tft()).unwrap().filters
    );
}

/// #283, §6.4.3: MODIFY EPS BEARER CONTEXT REQUEST updates the QoS and the TFT
/// and is accepted (CA); an unknown EBI is rejected #43 (CB); a TFT operation
/// §6.4.3.4 refuses is rejected with its cause.
#[test]
fn a_bearer_modification_follows_6_4_3() {
    let (mut ue, mut mme, _) = attach();
    esm_reply(&mut ue, &mut mme, &activate_dedicated(6, 5, &sip_tft()));
    // New EPS QoS (5B) QCI 2, TFT (36): add filter 2, uplink, TCP to port 80
    let add = [0x61, 0x22, 0x20, 0x05, 0x30, 0x06, 0x50, 0x00, 0x50];
    let mut modify = vec![0x62, 0x00, 0xC9, 0x5B, 0x01, 0x02, 0x36, add.len() as u8];
    modify.extend_from_slice(&add);
    assert_eq!(
        esm_reply(&mut ue, &mut mme, &modify),
        vec![0x62, 0x00, 0xCA]
    );
    let b = &ue.dedicated_bearers()[0];
    assert_eq!((b.qci, b.filters.len()), (2, 2));
    let mut tcp80 = udp4([1, 1, 1, 1], 80);
    tcp80[9] = 6;
    assert_eq!(ue.uplink_bearer_for(&tcp80), Some(6));

    // Delete existing TFT on a dedicated bearer: #41
    let modify = [0x62, 0x00, 0xC9, 0x36, 0x01, 0x40];
    assert_eq!(
        esm_reply(&mut ue, &mut mme, &modify),
        vec![0x62, 0x00, 0xCB, 41]
    );
    // An EBI that is not active: #43
    let modify = [0x82, 0x00, 0xC9];
    assert_eq!(
        esm_reply(&mut ue, &mut mme, &modify),
        vec![0x82, 0x00, 0xCB, 43]
    );
}

/// #283, §6.4.4.3: DEACTIVATE EPS BEARER CONTEXT REQUEST of a dedicated bearer
/// removes it and is accepted (CE); uplink falls back to the default bearer.
/// Deactivating the default bearer takes its dedicated bearers with it, and
/// an EBI that is not active is accepted as well.
#[test]
fn a_bearer_deactivation_follows_6_4_4() {
    let (mut ue, mut mme, _) = attach();
    esm_reply(&mut ue, &mut mme, &activate_dedicated(6, 5, &sip_tft()));
    assert_eq!(
        esm_reply(&mut ue, &mut mme, &[0x62, 0x00, 0xCD, 36]),
        vec![0x62, 0x00, 0xCE]
    );
    assert!(ue.dedicated_bearers().is_empty());
    assert_eq!(ue.uplink_bearer_for(&udp4([10, 45, 0, 1], 5060)), Some(5));
    assert_eq!(
        esm_reply(&mut ue, &mut mme, &[0x92, 0x00, 0xCD, 36]),
        vec![0x92, 0x00, 0xCE],
        "not active: accepted"
    );
    esm_reply(&mut ue, &mut mme, &activate_dedicated(6, 5, &sip_tft()));
    esm_reply(&mut ue, &mut mme, &[0x52, 0x00, 0xCD, 36]);
    assert_eq!(ue.default_bearer(), None);
    assert!(ue.dedicated_bearers().is_empty(), "with its PDN connection");
}

/// #283: losing the registration drops the dedicated bearers too.
#[test]
fn a_detach_drops_the_dedicated_bearers() {
    let (mut ue, mut mme, _) = attach();
    esm_reply(&mut ue, &mut mme, &activate_dedicated(6, 5, &sip_tft()));
    ue.on_downlink(&mme.protect(&detach_request(1, None), 2), Instant::now());
    assert!(ue.dedicated_bearers().is_empty());
}
