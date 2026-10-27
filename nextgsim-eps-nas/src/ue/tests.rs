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
    // EMM-CONNECTED stops T3412 (§5.3.5, #356).
    ue.on_connected();
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
    ue.on_connected();
    ue.on_released(now);
    assert_eq!(ue.state(), EmmState::Registered);
    assert_eq!(
        ue.next_deadline(),
        Some(now + ue.t3412().unwrap()),
        "only T3412, restarted in EMM-IDLE (#356)"
    );
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
/// Anything after it can only be a BEARER RESOURCE MODIFICATION REQUEST for
/// filters a precedence clash deleted (#408).
fn esm_reply(ue: &mut EpsNasUe, mme: &mut Mme, esm: &[u8]) -> Vec<u8> {
    let out = ue.on_downlink(&mme.protect(esm, 2), Instant::now());
    assert!(!out.uplink.is_empty(), "a reply");
    let reply = mme.unprotect(&out.uplink[0]);
    for extra in &out.uplink[1..] {
        assert_eq!(
            mme.unprotect(extra)[2],
            0xD6,
            "only a bearer resource modification follows"
        );
    }
    reply
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

fn modify_tft(ebi: u8, tft: &[u8]) -> Vec<u8> {
    let mut m = vec![(ebi << 4) | 0x02, 0x00, 0xC9, 0x36, tft.len() as u8];
    m.extend_from_slice(tft);
    m
}

/// #407 review, TS 23.401 §5.4.1: the default bearer's uplink filters take
/// part in the precedence order like any other; with no match the packet
/// goes on the bearer without uplink filters, and with none such it is
/// discarded.
#[test]
fn every_tft_counts_and_an_unmatched_packet_needs_a_bearer_without_filters() {
    let (mut ue, mut mme, _) = attach();
    esm_reply(&mut ue, &mut mme, &activate_dedicated(6, 5, &sip_tft()));
    let icmp = [
        0x45, 0, 0, 20, 0, 0, 0, 0, 64, 1, 0, 0, 10, 45, 0, 2, 8, 8, 8, 8,
    ];
    assert_eq!(
        ue.uplink_bearer_for(&icmp),
        Some(5),
        "the default has no filter"
    );
    // The default bearer gets an uplink UDP filter at precedence 1.
    let udp = [0x21, 0x21, 0x01, 0x02, 0x30, 0x11];
    let reply = esm_reply(&mut ue, &mut mme, &modify_tft(5, &udp));
    assert_eq!(reply, vec![0x52, 0x00, 0xCA]);
    assert_eq!(ue.uplink_bearer_for(&udp4([10, 45, 0, 1], 5060)), Some(5));
    assert_eq!(
        ue.uplink_bearer_for(&icmp),
        None,
        "every bearer filters: discarded"
    );
    assert_eq!(ue.uplink_bearer_for(&[0x00, 0x01]), None, "not IP");
}

/// #407 review, §6.4.3.4 d) 2): a modification that gives a dedicated bearer
/// the precedence of another dedicated bearer's filter deletes that filter.
#[test]
fn a_modification_deletes_the_filters_whose_precedence_it_takes() {
    let (mut ue, mut mme, _) = attach();
    esm_reply(&mut ue, &mut mme, &activate_dedicated(6, 5, &sip_tft()));
    let tcp = [0x21, 0x21, 0x20, 0x02, 0x30, 0x06];
    esm_reply(&mut ue, &mut mme, &activate_dedicated(7, 5, &tcp));
    // Add to EBI 7 a filter (id 2) at EBI 6's precedence 0x10.
    let add = [0x61, 0x22, 0x10, 0x02, 0x30, 0x11];
    let reply = esm_reply(&mut ue, &mut mme, &modify_tft(7, &add));
    assert_eq!(reply, vec![0x72, 0x00, 0xCA]);
    let b = |e| {
        ue.dedicated_bearers()
            .iter()
            .find(|b| b.ebi == e)
            .unwrap()
            .filters
            .len()
    };
    assert_eq!((b(6), b(7)), (0, 2));
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

// ---- #408: precedence clashes (§6.4.2.4 d) 2), §6.4.3.4 d) 2)) and the
// UE-requested bearer resource modification (§6.5.4) ----

/// The plain uplink PDUs of `out`.
fn plain_uplinks(mme: &mut Mme, out: &NasOutput) -> Vec<Vec<u8>> {
    out.uplink.iter().map(|p| mme.unprotect(p)).collect()
}

/// An attached UE with dedicated bearers 6 (SIP, filter 1 at precedence 0x10)
/// and the TCP bearer 7 given a filter 1 at the same precedence 0x10: EBI 6's
/// filter is deleted. Returns the plain uplinks of the activation of 7.
fn clash_on_dedicated() -> (EpsNasUe, Mme, Vec<Vec<u8>>, Instant) {
    let (mut ue, mut mme, _) = attach();
    let now = Instant::now();
    esm_reply(&mut ue, &mut mme, &activate_dedicated(6, 5, &sip_tft()));
    let tcp = [0x21, 0x21, 0x10, 0x02, 0x30, 0x06];
    let out = ue.on_downlink(&mme.protect(&activate_dedicated(7, 5, &tcp), 2), now);
    let sent = plain_uplinks(&mut mme, &out);
    (ue, mme, sent, now)
}

/// #408, §6.4.2.4 d) 2) and §6.5.4.2 / §8.3.10: the deletion of another
/// dedicated bearer's filters is followed by a BEARER RESOURCE MODIFICATION
/// REQUEST that deletes them in the network: EBI 0 and its PTI in the header,
/// type D6, EBI for packet filter 6, Traffic flow aggregate "Delete packet
/// filters from existing TFT" (101, one filter) with identifier 1 (TS 24.008
/// §10.5.6.12), ESM cause #36. T3481 runs; the network's DEACTIVATE EPS
/// BEARER CONTEXT REQUEST with that PTI stops it (§6.5.4.3).
#[test]
fn a_deleted_dedicated_filter_is_deleted_in_the_network_too() {
    let (mut ue, mut mme, sent, now) = clash_on_dedicated();
    assert_eq!(sent.len(), 2, "{sent:02x?}");
    assert_eq!(sent[0], vec![0x72, 0x00, 0xC6], "EBI 7's ACCEPT first");
    let pti = sent[1][1];
    assert_ne!(pti, 0, "an assigned PTI");
    assert_eq!(
        sent[1],
        vec![0x02, pti, 0xD6, 0x06, 0x02, 0xA1, 0x01, 0x58, 36],
        "BEARER RESOURCE MODIFICATION REQUEST"
    );
    assert_eq!(ue.next_deadline(), Some(now + T3481));

    // The network deactivates EBI 6, which has no filter left
    let deactivate = [0x62, pti, 0xCD, 36];
    assert_eq!(
        esm_reply(&mut ue, &mut mme, &deactivate),
        vec![0x62, pti, 0xCE]
    );
    assert_eq!(ue.next_deadline(), None, "T3481 stopped");
    assert_eq!(ue.dedicated_bearers().len(), 1);
}

/// #408, §6.5.4.5 a): T3481 retransmits the identical request four times; on
/// the fifth expiry the procedure is aborted and, as it released every
/// traffic flow of EBI 6, EBI 6 is deactivated locally.
#[test]
fn t3481_retransmits_four_times_then_releases_the_bearer_locally() {
    let (mut ue, mut mme, sent, mut now) = clash_on_dedicated();
    for n in 1..=4 {
        now += T3481;
        let out = ue.on_timer(now);
        assert_eq!(plain_uplinks(&mut mme, &out), vec![sent[1].clone()], "#{n}");
    }
    now += T3481;
    assert!(ue.on_timer(now).uplink.is_empty());
    assert_eq!(ue.next_deadline(), None);
    assert!(
        ue.dedicated_bearers().iter().all(|b| b.ebi != 6),
        "EBI 6 deactivated locally"
    );
}

/// #408, §6.5.4.4.1 / §6.5.4.5 b), §7.3.1 d): a BEARER RESOURCE MODIFICATION
/// REJECT for a PTI not in use is ignored; with the PTI and #43 it stops T3481
/// and EBI 6 is deactivated locally. #26 only ends the procedure.
#[test]
fn a_bearer_resource_modification_reject_ends_the_procedure() {
    let (mut ue, mut mme, sent, _) = clash_on_dedicated();
    let pti = sent[1][1];
    let reject = |pti: u8, cause: u8| vec![0x02, pti, 0xD7, cause];
    let out = ue.on_downlink(&mme.protect(&reject(pti ^ 0x40, 43), 2), Instant::now());
    assert!(out.uplink.is_empty());
    assert!(ue.next_deadline().is_some(), "another PTI: ignored");
    let out = ue.on_downlink(&mme.protect(&reject(pti, 43), 2), Instant::now());
    assert!(out.uplink.is_empty());
    assert_eq!(ue.next_deadline(), None, "T3481 stopped");
    assert!(ue.dedicated_bearers().iter().all(|b| b.ebi != 6));

    let (mut ue, mut mme, sent, _) = clash_on_dedicated();
    ue.on_downlink(&mme.protect(&reject(sent[1][1], 26), 2), Instant::now());
    assert_eq!(ue.next_deadline(), None);
    assert!(ue.dedicated_bearers().iter().any(|b| b.ebi == 6), "kept");
}

/// The attached UE whose default bearer (EBI 5) has an uplink UDP filter at
/// precedence 1.
fn default_bearer_with_a_filter() -> (EpsNasUe, Mme) {
    let (mut ue, mut mme, _) = attach();
    let udp = [0x21, 0x21, 0x01, 0x02, 0x30, 0x11];
    assert_eq!(
        esm_reply(&mut ue, &mut mme, &modify_tft(5, &udp)),
        vec![0x52, 0x00, 0xCA]
    );
    (ue, mme)
}

/// The DETACH REQUEST of a release of the last PDN connection, then the
/// DETACH ACCEPT, the release, and the re-attach.
fn detaches_and_reattaches(ue: &mut EpsNasUe, mme: &mut Mme, out: &NasOutput, now: Instant) {
    assert_eq!(
        out.uplink.len(),
        1,
        "the DETACH REQUEST only, no ESM answer"
    );
    let plain = detach_request_body(mme, &out.uplink[0]);
    assert_eq!(plain[2] & 0x0F, 0x01, "EPS detach, not switch off");
    assert_eq!(ue.state(), EmmState::DeregisteredInitiated);
    ue.on_downlink(&mme.protect(&[0x07, 0x46], 2), now);
    assert_eq!(ue.state(), EmmState::Deregistered, "an attach is due");
    assert!(ue.default_bearer().is_none());
    ue.on_released(now);
    let req = ue.initial_message(now).expect("the re-attach");
    assert_eq!(&mme.unprotect(&req)[..2], &[0x07, 0x41], "ATTACH REQUEST");
}

/// #408, §6.4.2.4 d) 2): a dedicated bearer whose TFT takes a precedence of a
/// default-bearer filter releases the PDN connection. It is the UE's last,
/// and the UE does not support EMM-REGISTERED without PDN connection, so it
/// detaches and re-attaches.
#[test]
fn an_activation_clashing_with_the_default_bearer_detaches_and_reattaches() {
    let (mut ue, mut mme) = default_bearer_with_a_filter();
    let now = Instant::now();
    let tcp = [0x21, 0x21, 0x01, 0x02, 0x30, 0x06];
    let out = ue.on_downlink(&mme.protect(&activate_dedicated(6, 5, &tcp), 2), now);
    assert!(ue.dedicated_bearers().is_empty(), "not activated");
    detaches_and_reattaches(&mut ue, &mut mme, &out, now);
}

/// #408, §6.4.3.4 d) 2): the same for a modification of a dedicated bearer.
#[test]
fn a_modification_clashing_with_the_default_bearer_detaches_and_reattaches() {
    let (mut ue, mut mme) = default_bearer_with_a_filter();
    let now = Instant::now();
    esm_reply(&mut ue, &mut mme, &activate_dedicated(6, 5, &sip_tft()));
    // Add to EBI 6 a filter (id 2) at the default bearer's precedence 1
    let add = [0x61, 0x22, 0x01, 0x02, 0x30, 0x06];
    let out = ue.on_downlink(&mme.protect(&modify_tft(6, &add), 2), now);
    detaches_and_reattaches(&mut ue, &mut mme, &out, now);
}

/// #408, §6.5.4.3 ii): a MODIFY EPS BEARER CONTEXT REQUEST with the request's
/// PTI is the network's answer too: T3481 stops, the bearer stays.
#[test]
fn a_modification_with_the_requests_pti_stops_t3481() {
    let (mut ue, mut mme, sent, _) = clash_on_dedicated();
    let pti = sent[1][1];
    // EBI 6, the request's PTI, new EPS QoS (5B) QCI 1
    let modify = [0x62, pti, 0xC9, 0x5B, 0x01, 0x01];
    assert_eq!(esm_reply(&mut ue, &mut mme, &modify), vec![0x62, pti, 0xCA]);
    assert_eq!(ue.next_deadline(), None, "T3481 stopped");
    assert!(ue.dedicated_bearers().iter().any(|b| b.ebi == 6));
}

/// #417 review, §6.5.4.5 c): a DEACTIVATE EPS BEARER CONTEXT REQUEST of the
/// bearer under modification aborts the procedure even without its PTI:
/// T3481 stops.
#[test]
fn a_deactivation_of_the_bearer_aborts_its_bearer_resource_modification() {
    let (mut ue, mut mme, _, _) = clash_on_dedicated();
    assert!(ue.next_deadline().is_some(), "T3481 running");
    assert_eq!(
        esm_reply(&mut ue, &mut mme, &[0x62, 0x00, 0xCD, 36]),
        vec![0x62, 0x00, 0xCE]
    );
    assert_eq!(ue.next_deadline(), None, "T3481 stopped");
}

/// #417 review, §6.5.4.5 d): REJECT #31 deactivates the bearer locally only
/// when the request released every traffic flow of it. EBI 6 keeps a second
/// filter here, so it stays.
#[test]
fn a_reject_31_keeps_a_bearer_with_filters_left() {
    let (mut ue, mut mme, _) = attach();
    let now = Instant::now();
    esm_reply(&mut ue, &mut mme, &activate_dedicated(6, 5, &sip_tft()));
    // EBI 6 gets a second filter (id 2) at precedence 0x30
    let add = [0x61, 0x22, 0x30, 0x02, 0x30, 0x06];
    esm_reply(&mut ue, &mut mme, &modify_tft(6, &add));
    let tcp = [0x21, 0x21, 0x10, 0x02, 0x30, 0x06];
    let out = ue.on_downlink(&mme.protect(&activate_dedicated(7, 5, &tcp), 2), now);
    let sent = plain_uplinks(&mut mme, &out);
    let pti = sent[1][1];
    assert_eq!(&sent[1][5..7], &[0xA1, 0x01], "only filter 1 of EBI 6");
    ue.on_downlink(&mme.protect(&[0x02, pti, 0xD7, 31], 2), now);
    assert_eq!(ue.next_deadline(), None, "T3481 stopped");
    let six = ue.dedicated_bearers().iter().find(|b| b.ebi == 6);
    assert_eq!(six.map(|b| b.filters.len()), Some(1), "kept, with filter 2");

    // With every flow released, #31 deactivates it
    let (mut ue, mut mme, sent, _) = clash_on_dedicated();
    ue.on_downlink(&mme.protect(&[0x02, sent[1][1], 0xD7, 31], 2), now);
    assert!(ue.dedicated_bearers().iter().all(|b| b.ebi != 6));
}

/// #417 review: the user's switch-off during the detach-and-re-attach of a
/// default-bearer clash wins: a switch-off DETACH REQUEST goes, and no attach
/// follows.
#[test]
fn a_switch_off_during_the_reattach_detach_does_not_reattach() {
    let (mut ue, mut mme) = default_bearer_with_a_filter();
    let now = Instant::now();
    let tcp = [0x21, 0x21, 0x01, 0x02, 0x30, 0x06];
    let out = ue.on_downlink(&mme.protect(&activate_dedicated(6, 5, &tcp), 2), now);
    detach_request_body(&mut mme, &out.uplink[0]);
    assert_eq!(ue.state(), EmmState::DeregisteredInitiated);

    let out = ue.detach(true, true, now);
    let plain = detach_request_body(&mut mme, &out.uplink[0]);
    assert_eq!(plain[2] & 0x08, 0x08, "switch off");
    assert_eq!(ue.state(), EmmState::Detached);
    ue.on_released(now);
    assert_eq!(ue.initial_message(now), None, "no re-attach");
}

/// #417 review, §5.5.2.2.4 d): a network DETACH REQUEST "re-attach not
/// required" crossing the detach-and-re-attach wins: "the UE need not
/// initiate the attach", and it does not.
#[test]
fn a_crossing_network_detach_cancels_the_reattach() {
    let (mut ue, mut mme) = default_bearer_with_a_filter();
    let now = Instant::now();
    let tcp = [0x21, 0x21, 0x01, 0x02, 0x30, 0x06];
    let out = ue.on_downlink(&mme.protect(&activate_dedicated(6, 5, &tcp), 2), now);
    detach_request_body(&mut mme, &out.uplink[0]);
    let out = ue.on_downlink(&mme.protect(&detach_request(2, Some(7)), 2), now);
    assert_eq!(
        mme.unprotect(&out.uplink[0]),
        vec![0x07, 0x46],
        "DETACH ACCEPT"
    );
    ue.on_released(now);
    assert_eq!(ue.initial_message(now), None, "no re-attach");
}

// ============================================================================
// UE-requested bearer resource modification on command (TS 24.301 §6.5.4,
// nextgsim #434)
// ============================================================================

/// An attached, connected UE with dedicated bearer 6 (QCI 1, SIP filter 1).
fn with_dedicated_6() -> (EpsNasUe, Mme, Instant) {
    let (mut ue, mut mme, _) = attach();
    esm_reply(&mut ue, &mut mme, &activate_dedicated(6, 5, &sip_tft()));
    (ue, mme, Instant::now())
}

/// Run `cmd` connected; its PTI and the plain BEARER RESOURCE MODIFICATION
/// REQUEST.
fn command(ue: &mut EpsNasUe, mme: &mut Mme, cmd: BearerCommand, now: Instant) -> (u8, Vec<u8>) {
    let (pti, out) = ue.bearer_command(cmd, true, now).expect("started");
    let sent = plain_uplinks(mme, &out);
    assert_eq!(sent.len(), 1, "{sent:02x?}");
    assert_eq!(sent[0][1], pti, "the request carries the PTI returned");
    (pti, sent[0].clone())
}

/// #434, §6.5.4.2 / Table 8.3.10.1: `bearer-modify 6 1000` is a BEARER
/// RESOURCE MODIFICATION REQUEST with EBI 0 and its PTI in the header, type
/// D6, EBI for packet filter 6, Traffic flow aggregate "No TFT operation"
/// (110, E bit, no packet filter, TS 24.008 §10.5.6.12) naming filter 1 in
/// parameter 03H, and Required traffic flow
/// QoS (5B, §9.9.4.3): QCI 1, MBR UL/DL and GBR UL/DL 1000 kbit/s, coded
/// 0x87 (576 + 7 x 64 = 1024, the next coded value). No ESM cause. T3481
/// runs; a MODIFY EPS BEARER CONTEXT REQUEST with the PTI, accepted, ends it
/// as "modified".
#[test]
fn bearer_modify_sends_no_tft_operation_with_the_required_qos_and_ends_modified() {
    let (mut ue, mut mme, now) = with_dedicated_6();
    let cmd = BearerCommand::Modify {
        ebi: 6,
        gbr_kbps: 1000,
    };
    let (pti, req) = command(&mut ue, &mut mme, cmd, now);
    assert_ne!(pti, 0);
    assert_eq!(
        req,
        vec![
            0x02, pti, 0xD6, 0x06, 0x04, 0xD0, 0x03, 0x01, 0x01, 0x5B, 0x05, 0x01, 0x87, 0x87,
            0x87, 0x87
        ]
    );
    assert_eq!(ue.next_deadline(), Some(now + T3481));
    assert_eq!(ue.bearer_resource_outcome(pti), None, "pending");

    let modify = [0x62, pti, 0xC9, 0x5B, 0x05, 0x01, 0x87, 0x87, 0x87, 0x87];
    assert_eq!(esm_reply(&mut ue, &mut mme, &modify), vec![0x62, pti, 0xCA]);
    assert_eq!(ue.next_deadline(), None, "T3481 stopped");
    let outcome = ue.bearer_resource_outcome(pti).expect("ended");
    assert_eq!(outcome, BearerResourceOutcome::Modified);
    assert_eq!(cmd.answer(outcome), Ok("modified"));
    assert_eq!(ue.dedicated_bearers()[0].ebi, 6, "kept");
}

/// #434: the Required traffic flow QoS takes the bearer's QCI and grows the
/// extended octets past 8640 kbit/s: 20 Mbit/s is base 0xFE, extended 0x4E
/// (16 + 4 Mbit/s), 9 octets of contents.
#[test]
fn bearer_modify_codes_a_high_rate_in_the_extended_octets() {
    let (mut ue, mut mme, _) = attach();
    let mut act = activate_dedicated(6, 5, &sip_tft());
    act[5] = 2; // QCI 2
    esm_reply(&mut ue, &mut mme, &act);
    let cmd = BearerCommand::Modify {
        ebi: 6,
        gbr_kbps: 20_000,
    };
    let (_, req) = command(&mut ue, &mut mme, cmd, Instant::now());
    assert_eq!(
        &req[9..],
        &[0x5B, 0x09, 0x02, 0xFE, 0xFE, 0xFE, 0xFE, 0x4E, 0x4E, 0x4E, 0x4E]
    );
}

/// #434, §6.5.4.2: `bearer-release 6` reuses the precedence-clash request:
/// "Delete packet filters from existing TFT" (101) naming every filter of the
/// bearer, ESM cause #36. The network's DEACTIVATE EPS BEARER CONTEXT REQUEST
/// for EBI 6, accepted, ends it as "released"; the bearer is gone.
#[test]
fn bearer_release_deletes_every_filter_and_ends_released() {
    let (mut ue, mut mme, now) = with_dedicated_6();
    // A second filter (id 2) on EBI 6, so the list has two identifiers
    let add = [0x61, 0x22, 0x30, 0x02, 0x30, 0x06];
    esm_reply(&mut ue, &mut mme, &modify_tft(6, &add));
    let cmd = BearerCommand::Release { ebi: 6 };
    let (pti, req) = command(&mut ue, &mut mme, cmd, now);
    assert_eq!(
        req,
        vec![0x02, pti, 0xD6, 0x06, 0x03, 0xA2, 0x01, 0x02, 0x58, 36]
    );
    assert_eq!(ue.next_deadline(), Some(now + T3481));
    assert_eq!(ue.bearer_resource_outcome(pti), None, "pending");

    let deactivate = [0x62, pti, 0xCD, 36];
    assert_eq!(
        esm_reply(&mut ue, &mut mme, &deactivate),
        vec![0x62, pti, 0xCE]
    );
    let outcome = ue.bearer_resource_outcome(pti).expect("ended");
    assert_eq!(cmd.answer(outcome), Ok("released"));
    assert!(ue.dedicated_bearers().is_empty());
    assert_eq!(ue.next_deadline(), None);
}

/// #434, §6.5.4.4: a BEARER RESOURCE MODIFICATION REJECT ends the command in
/// an error naming the cause; the bearer stays for #30.
#[test]
fn a_rejected_bearer_command_answers_the_cause() {
    let (mut ue, mut mme, now) = with_dedicated_6();
    let cmd = BearerCommand::Modify {
        ebi: 6,
        gbr_kbps: 64,
    };
    let (pti, _) = command(&mut ue, &mut mme, cmd, now);
    ue.on_downlink(&mme.protect(&[0x02, pti, 0xD7, 30], 2), now);
    let outcome = ue.bearer_resource_outcome(pti).expect("ended");
    assert_eq!(outcome, BearerResourceOutcome::Rejected(30));
    let err = cmd.answer(outcome).unwrap_err();
    assert!(err.contains("REJECT") && err.contains("#30"), "{err}");
    assert_eq!(ue.dedicated_bearers().len(), 1);
}

/// #434, §6.5.4.5 a): four retransmissions, then the fifth T3481 expiry aborts
/// the command; the answer says so.
#[test]
fn a_bearer_command_t3481_gives_up_after_the_fifth_expiry() {
    let (mut ue, mut mme, mut now) = with_dedicated_6();
    let cmd = BearerCommand::Modify {
        ebi: 6,
        gbr_kbps: 128,
    };
    let (pti, req) = command(&mut ue, &mut mme, cmd, now);
    for n in 1..=4 {
        now += T3481;
        let out = ue.on_timer(now);
        assert_eq!(plain_uplinks(&mut mme, &out), vec![req.clone()], "#{n}");
        assert_eq!(ue.bearer_resource_outcome(pti), None);
    }
    now += T3481;
    assert!(ue.on_timer(now).uplink.is_empty());
    let outcome = ue.bearer_resource_outcome(pti).expect("ended");
    assert_eq!(outcome, BearerResourceOutcome::Aborted);
    assert!(cmd.answer(outcome).unwrap_err().contains("T3481"));
    assert_eq!(ue.dedicated_bearers().len(), 1, "a modify keeps the bearer");
}

/// #434: a MODIFY with the PTI that the UE itself rejects (§6.4.3.4 a) 3):
/// it would empty the dedicated bearer's TFT, #41) is not "modified"; a release
/// answered by a MODIFY is not "released", nor a modify answered by a
/// DEACTIVATE "modified".
#[test]
fn a_bearer_command_answered_otherwise_is_an_error() {
    let (mut ue, mut mme, now) = with_dedicated_6();
    let modify = BearerCommand::Modify {
        ebi: 6,
        gbr_kbps: 64,
    };
    let (pti, _) = command(&mut ue, &mut mme, modify, now);
    // Delete the only filter of the dedicated bearer: §6.4.3.4 a) 3), #41
    let bad = [0x62, pti, 0xC9, 0x36, 0x02, 0xA1, 0x01];
    assert_eq!(
        esm_reply(&mut ue, &mut mme, &bad),
        vec![0x62, pti, 0xCB, 41]
    );
    let outcome = ue.bearer_resource_outcome(pti).unwrap();
    assert_eq!(outcome, BearerResourceOutcome::ModifyNotAccepted(Some(41)));
    assert!(modify.answer(outcome).is_err());

    let release = BearerCommand::Release { ebi: 6 };
    let (pti, _) = command(&mut ue, &mut mme, release, now);
    esm_reply(&mut ue, &mut mme, &[0x62, pti, 0xC9, 0x5B, 0x01, 0x01]);
    let outcome = ue.bearer_resource_outcome(pti).unwrap();
    assert!(release.answer(outcome).unwrap_err().contains("instead"));

    let (pti, _) = command(&mut ue, &mut mme, modify, now);
    esm_reply(&mut ue, &mut mme, &[0x62, 0x00, 0xCD, 36]);
    let outcome = ue.bearer_resource_outcome(pti).unwrap();
    assert_eq!(outcome, BearerResourceOutcome::Deactivated, "§6.5.4.5 c)");
    assert!(modify.answer(outcome).unwrap_err().contains("instead"));
}

/// #434: refused, with nothing sent and no PTI used: the default bearer, an
/// EBI with no bearer, a rate the EPS QoS IE cannot code, a GBR for a
/// non-GBR bearer, a second command
/// on a bearer under modification, and a UE that is not EMM-REGISTERED.
#[test]
fn bearer_commands_are_refused_for_the_default_bearer_an_unknown_ebi_and_no_registration() {
    let (mut ue, mut mme, now) = with_dedicated_6();
    let modify = |ebi, gbr_kbps| BearerCommand::Modify { ebi, gbr_kbps };
    use BearerCommandRefusal as R;
    for (cmd, refusal) in [
        (modify(5, 64), R::DefaultBearer(5)),
        (BearerCommand::Release { ebi: 5 }, R::DefaultBearer(5)),
        (modify(9, 64), R::UnknownBearer(9)),
        (BearerCommand::Release { ebi: 9 }, R::UnknownBearer(9)),
        (modify(6, 10_000_001), R::RateOutOfRange(10_000_001)),
    ] {
        assert_eq!(ue.bearer_command(cmd, true, now), Err(refusal), "{cmd:?}");
    }
    assert_eq!(ue.next_deadline(), None, "nothing started");
    assert!(
        R::DefaultBearer(5).to_string().contains("default bearer"),
        "the error line names it"
    );

    // A non-GBR dedicated bearer (QCI 9) has no GBR to modify; it can be
    // released
    let mut act = activate_dedicated(7, 5, &[0x21, 0x22, 0x20, 0x02, 0x30, 0x06]);
    act[5] = 9;
    esm_reply(&mut ue, &mut mme, &act);
    assert_eq!(
        ue.bearer_command(modify(7, 64), true, now),
        Err(R::NonGbrBearer(7, 9))
    );

    command(&mut ue, &mut mme, modify(6, 10_000_000), now);
    assert_eq!(
        ue.bearer_command(BearerCommand::Release { ebi: 6 }, true, now),
        Err(R::InProgress(6))
    );

    let mut fresh = EpsNasUe::new(cfg());
    assert_eq!(
        fresh.bearer_command(modify(6, 64), true, now),
        Err(R::NotRegistered(EmmState::Deregistered))
    );
    let out = ue.detach(false, true, now);
    assert!(!out.uplink.is_empty());
    assert_eq!(
        ue.bearer_command(modify(6, 64), true, now),
        Err(R::NotRegistered(EmmState::DeregisteredInitiated))
    );
}

/// #434: EMM-REGISTERED in EMM-IDLE, the command is not sent; it asks for a
/// connection, and the UE's next initial message is a SERVICE REQUEST. Once
/// the service request is done, the command goes.
#[test]
fn an_idle_bearer_command_starts_a_service_request_first() {
    let (mut ue, mut mme, now) = with_dedicated_6();
    ue.on_released(now);
    let cmd = BearerCommand::Release { ebi: 6 };
    assert_eq!(
        ue.bearer_command(cmd, false, now),
        Err(BearerCommandRefusal::NotConnected)
    );
    let sr = ue.initial_message(now).expect("SERVICE REQUEST");
    mme.verify_service_request(&sr);
    assert_eq!(
        ue.bearer_command(cmd, true, now),
        Err(BearerCommandRefusal::NotConnected),
        "not before the service request completes"
    );
    ue.on_user_plane_established();
    command(&mut ue, &mut mme, cmd, now);
}

/// #434: a detach while a command runs ends it: the bearer contexts go.
#[test]
fn a_detach_ends_a_running_bearer_command() {
    let (mut ue, mut mme, now) = with_dedicated_6();
    let cmd = BearerCommand::Release { ebi: 6 };
    let (pti, _) = command(&mut ue, &mut mme, cmd, now);
    ue.detach(true, true, now);
    let outcome = ue.bearer_resource_outcome(pti).expect("ended");
    assert_eq!(outcome, BearerResourceOutcome::Cleared);
    assert!(cmd.answer(outcome).is_err());
}

/// #435: the user plane is awaited from pending uplink data to the end of
/// the service request, and each way it ends without the user plane is
/// reported once.
#[test]
fn a_service_request_reports_once_how_it_failed() {
    let (mut ue, mut mme, now) = registered_idle();
    assert!(!ue.user_plane_awaited());
    ue.on_uplink_data_pending();
    assert!(ue.user_plane_awaited(), "pending uplink data");
    mme.verify_service_request(&ue.initial_message(now).unwrap());
    assert!(ue.user_plane_awaited(), "SERVICE REQUEST sent");
    ue.on_timer(now + T3417);
    assert!(!ue.user_plane_awaited());
    assert_eq!(
        ue.take_service_request_failure(),
        Some(ServiceRequestFailure::T3417Expired)
    );
    assert_eq!(ue.take_service_request_failure(), None, "once");
    ue.on_released(now + T3417);
    assert_eq!(ue.take_service_request_failure(), None);

    let (mut ue, mut mme, now) = registered_idle();
    ue.on_uplink_data_pending();
    mme.verify_service_request(&ue.initial_message(now).unwrap());
    ue.on_downlink(&[0x07, 0x4E, 22, 0x5F, 0x01, 0x01], now);
    assert_eq!(
        ue.take_service_request_failure(),
        Some(ServiceRequestFailure::Rejected(22))
    );

    let (mut ue, mut mme, now) = registered_idle();
    ue.on_uplink_data_pending();
    mme.verify_service_request(&ue.initial_message(now).unwrap());
    ue.on_released(now);
    assert_eq!(
        ue.take_service_request_failure(),
        Some(ServiceRequestFailure::LowerLayerFailure)
    );

    // A detach with uplink data pending, before the SERVICE REQUEST went.
    let (mut ue, _mme, now) = registered_idle();
    ue.on_uplink_data_pending();
    ue.detach(false, false, now);
    assert_eq!(
        ue.take_service_request_failure(),
        Some(ServiceRequestFailure::Detach)
    );

    // The user plane set up: no failure.
    let (mut ue, mut mme, now) = registered_idle();
    ue.on_uplink_data_pending();
    mme.verify_service_request(&ue.initial_message(now).unwrap());
    ue.on_user_plane_established();
    assert!(!ue.user_plane_awaited());
    ue.on_released(now);
    assert_eq!(ue.take_service_request_failure(), None);
}
