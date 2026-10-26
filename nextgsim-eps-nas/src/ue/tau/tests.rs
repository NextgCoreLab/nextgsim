//! Tracking area updating against the scripted MME (#356). The expected
//! octets are written from TS 24.301 §8.2.26-§8.2.29 and §9.9 by hand.

use super::super::*;
use crate::testing::*;

/// T3412 of the scripted ATTACH ACCEPT: 0x49, unit 010 (decihours) x 9.
const T3412_ATTACH: Duration = Duration::from_secs(54 * 60);

/// The Old GUTI LV of the scripted ATTACH ACCEPT's GUTI (§9.9.3.12): 0xF6,
/// PLMN 999/70, MME group 0x0002, MME code 0x01, M-TMSI 0xC0000001.
const OLD_GUTI: [u8; 12] = [
    0x0B, 0xF6, 0x99, 0xF9, 0x07, 0x00, 0x02, 0x01, 0xC0, 0x00, 0x00, 0x01,
];

/// Last visited registered TAI 999/70 TAC 1 (IEI 52, §9.9.3.32).
const LAST_VISITED_TA1: [u8; 6] = [0x52, 0x99, 0xF9, 0x07, 0x00, 0x01];

/// A new GUTI (IEI 50): M-TMSI 0xC0000002.
const NEW_GUTI: [u8; 13] = [
    0x50, 0x0B, 0xF6, 0x99, 0xF9, 0x07, 0x00, 0x02, 0x01, 0xC0, 0x00, 0x00, 0x02,
];

/// TAI list (IEI 54) 999/70 TACs 1 and 2 (§9.9.3.33 type 00, two TACs).
const TAI_LIST_1_2: [u8; 10] = [0x54, 0x08, 0x01, 0x99, 0xF9, 0x07, 0x00, 0x01, 0x00, 0x02];

/// EMM-REGISTERED in EMM-IDLE, camped in TA 1 (in the TAI list).
fn idle_in_ta1() -> (EpsNasUe, Mme, Instant) {
    let (mut ue, mme, now) = registered_idle();
    ue.on_serving_cell(1);
    (ue, mme, now)
}

/// The TRACKING AREA UPDATE REQUEST the UE asks a connection for, checked
/// integrity protected and unciphered (§4.4.5), as the MME reads it; the
/// connection then comes up.
fn tau_request(ue: &mut EpsNasUe, mme: &mut Mme, now: Instant) -> Vec<u8> {
    let pdu = ue
        .initial_message(now)
        .expect("TRACKING AREA UPDATE REQUEST");
    assert_eq!(
        pdu[0], 0x17,
        "security header type 1: integrity protected only"
    );
    assert_eq!(ue.state(), EmmState::TrackingAreaUpdatingInitiated);
    assert_eq!(ue.next_deadline(), Some(now + T3430), "T3430");
    ue.on_connected();
    mme.unprotect(&pdu)
}

/// TRACKING AREA UPDATE ACCEPT, "TA updated", with `ies` (§8.2.26.1).
fn accept(mme: &mut Mme, ies: &[u8]) -> Vec<u8> {
    let mut m = vec![0x07, 0x49, 0x00];
    m.extend_from_slice(ies);
    mme.protect(&m, 2)
}

/// TRACKING AREA UPDATE REJECT with `cause` and `ies` (§8.2.28.1).
fn reject(mme: &mut Mme, cause: u8, ies: &[u8]) -> Vec<u8> {
    let mut m = vec![0x07, 0x4B, cause];
    m.extend_from_slice(ies);
    mme.protect(&m, 2)
}

/// #356, §5.3.5 / §5.5.3.2.2 b): T3412 runs in EMM-IDLE; its expiry owes a
/// periodic update, sent as "periodic updating" without the UE network
/// capability (§8.2.29.7), with the last visited TAI, the EPS bearer context
/// status of bearer 5 and the Old GUTI type. The accept ends it: EMM-REGISTERED,
/// T3440 for the network's release (§5.3.1.2.1 b), and T3412 again in idle.
#[test]
fn t3412_expiry_sends_a_periodic_update() {
    let (mut ue, mut mme, now) = idle_in_ta1();
    assert_eq!(ue.t3412(), Some(T3412_ATTACH));
    assert_eq!(ue.next_deadline(), Some(now + T3412_ATTACH));
    assert_eq!(ue.initial_message(now), None, "nothing owed yet");
    let t = now + T3412_ATTACH;
    assert_eq!(ue.on_timer(t), NasOutput::default());

    let req = tau_request(&mut ue, &mut mme, t);
    let mut expect = vec![0x07, 0x48, 0x13];
    expect.extend_from_slice(&OLD_GUTI);
    expect.extend_from_slice(&LAST_VISITED_TA1);
    expect.extend_from_slice(&[0x57, 0x02, 0x20, 0x00, 0xE0]);
    assert_eq!(req, expect, "KSI 1, periodic (011), no active flag");

    let out = ue.on_downlink(&accept(&mut mme, &[]), t);
    assert_eq!(out, NasOutput::default(), "no GUTI, no TAU COMPLETE");
    assert_eq!(ue.state(), EmmState::Registered);
    assert_eq!(ue.next_deadline(), Some(t + T3440), "T3440: no active flag");
    ue.on_released(t);
    assert_eq!(ue.next_deadline(), Some(t + T3412_ATTACH), "T3412 again");
    assert_eq!(ue.initial_message(t), None, "nothing owed after the update");
}

/// #356, §5.5.3.2.2 a): a TA outside the TAI list owes "TA updating", with
/// the UE network capability, and with pending uplink data the "active" flag.
/// The accept's GUTI is acknowledged with TRACKING AREA UPDATE COMPLETE,
/// ciphered on the now secured exchange; its TAI list replaces the old one.
#[test]
fn a_new_ta_sends_a_ta_update_with_the_active_flag() {
    let (mut ue, mut mme, now) = idle_in_ta1();
    ue.on_serving_cell(1);
    assert_eq!(ue.initial_message(now), None, "TA 1 is in the list");
    ue.on_serving_cell(2);
    ue.on_uplink_data_pending();
    let req = tau_request(&mut ue, &mut mme, now);
    assert_eq!(req[2], 0x18, "KSI 1, active flag, TA updating (000)");
    assert_eq!(&req[3..15], &OLD_GUTI);
    assert_eq!(
        &req[15..19],
        &[0x58, 0x02, 0xE0, 0xE0],
        "UE network capability"
    );
    assert_eq!(&req[19..25], &LAST_VISITED_TA1);

    let mut ies = NEW_GUTI.to_vec();
    ies.extend_from_slice(&TAI_LIST_1_2);
    let out = ue.on_downlink(&accept(&mut mme, &ies), now);
    assert_eq!(out.uplink.len(), 1, "TAU COMPLETE");
    assert_eq!(out.uplink[0][0] >> 4, 2, "integrity protected and ciphered");
    assert_eq!(mme.unprotect(&out.uplink[0]), vec![0x07, 0x4A]);
    assert_eq!(ue.guti().unwrap().m_tmsi, 0xC000_0002);
    assert_eq!(
        ue.tai_list().iter().map(|t| t.tac).collect::<Vec<_>>(),
        [1, 2]
    );
    assert_eq!(
        ue.next_deadline(),
        None,
        "active flag: no T3440, no T3412 connected"
    );

    // The next update names TA 2 as the last visited registered TAI.
    ue.on_released(now);
    ue.on_serving_cell(3);
    let req = tau_request(&mut ue, &mut mme, now);
    let lvt = req.windows(2).position(|w| w == [0x52, 0x99]).unwrap();
    assert_eq!(&req[lvt + 4..lvt + 6], &[0x00, 0x02]);
}

/// #356, §5.5.3.2.4: bearer contexts the accept marks inactive are
/// deactivated locally; the request listed both (EBI 5 and 6, `60 00`).
#[test]
fn the_accept_deactivates_the_bearers_the_mme_does_not_list() {
    let (mut ue, mut mme, _) = attach();
    let now = Instant::now();
    ue.on_serving_cell(1);
    // ACTIVATE DEDICATED EPS BEARER CONTEXT REQUEST: EBI 6, linked 5, QCI 1,
    // one uplink UDP filter.
    let tft = [0x21, 0x21, 0x10, 0x02, 0x30, 0x11];
    let mut esm = vec![0x62, 0x00, 0xC5, 0x05, 0x01, 0x01, tft.len() as u8];
    esm.extend_from_slice(&tft);
    let out = ue.on_downlink(&mme.protect(&esm, 2), now);
    assert_eq!(mme.unprotect(&out.uplink[0]), vec![0x62, 0x00, 0xC6]);
    ue.on_released(now);
    ue.on_serving_cell(2);

    let req = tau_request(&mut ue, &mut mme, now);
    let ebcs = req.windows(2).position(|w| w == [0x57, 0x02]).unwrap();
    assert_eq!(
        &req[ebcs + 2..ebcs + 4],
        &[0x60, 0x00],
        "EBI 5 and 6 active"
    );
    ue.on_downlink(&accept(&mut mme, &[0x57, 0x02, 0x20, 0x00]), now);
    assert!(
        ue.dedicated_bearers().is_empty(),
        "EBI 6 deactivated locally"
    );
    assert_eq!(ue.default_bearer().unwrap().ebi, 5);
    assert_eq!(ue.state(), EmmState::Registered);
}

/// #356: an accept without the default bearer takes the only PDN connection;
/// EMM-REGISTERED without PDN connection is not supported, so the UE detaches
/// locally and attaches again (with its GUTI).
#[test]
fn an_accept_without_the_default_bearer_leads_to_a_new_attach() {
    let (mut ue, mut mme, now) = idle_in_ta1();
    ue.on_serving_cell(2);
    tau_request(&mut ue, &mut mme, now);
    let out = ue.on_downlink(&accept(&mut mme, &[0x57, 0x02, 0x00, 0x00]), now);
    assert_eq!(out, NasOutput::default());
    assert!(ue.default_bearer().is_none());
    assert_eq!(ue.state(), EmmState::Deregistered);
    assert_eq!(ue.next_deadline(), Some(now + T3440));
    ue.on_released(now);
    let attach = ue.initial_message(now).expect("ATTACH REQUEST");
    assert_eq!(attach[0] >> 4, 1, "integrity protected, with the GUTI");
}

/// #356, §5.5.3.2.4 / §5.3.5: the accept's T3412 extended value wins over its
/// T3412 value; a zero T3412 deactivates the periodic update.
#[test]
fn the_accept_sets_t3412() {
    let (mut ue, mut mme, now) = idle_in_ta1();
    ue.on_serving_cell(2);
    tau_request(&mut ue, &mut mme, now);
    // T3412 value 0x21 (1 min) and T3412 extended value 0x22 (unit 001, 2 h).
    ue.on_downlink(&accept(&mut mme, &[0x5A, 0x21, 0x5E, 0x01, 0x22]), now);
    assert_eq!(ue.t3412(), Some(Duration::from_secs(2 * 3600)));
    ue.on_released(now);
    ue.on_serving_cell(3);
    tau_request(&mut ue, &mut mme, now);
    ue.on_downlink(&accept(&mut mme, &[0x5A, 0x21]), now);
    assert_eq!(ue.t3412(), Some(Duration::from_secs(60)));
    ue.on_released(now);
    ue.on_serving_cell(4);
    tau_request(&mut ue, &mut mme, now);
    ue.on_downlink(&accept(&mut mme, &[0x5A, 0x00]), now);
    assert_eq!(ue.t3412(), None, "zero: deactivated");
    ue.on_released(now);
    assert_eq!(ue.next_deadline(), None, "no T3412 in idle");
}

/// The UE after TAU REJECT `cause`, from a periodic update in TA 1.
fn rejected(cause: u8, ies: &[u8]) -> (EpsNasUe, Mme, Instant) {
    let (mut ue, mut mme, now) = idle_in_ta1();
    let now = now + T3412_ATTACH;
    ue.on_timer(now);
    tau_request(&mut ue, &mut mme, now);
    let out = ue.on_downlink(&reject(&mut mme, cause, ies), now);
    assert_eq!(out, NasOutput::default(), "#{cause}: nothing sent");
    (ue, mme, now)
}

/// #356, §5.5.3.2.5 #3, #6, #7, #8: the USIM is invalid for EPS; GUTI, TAI
/// list and eKSI deleted; EMM-DEREGISTERED with no further attach.
#[test]
fn reject_3_6_7_8_invalidate_the_usim() {
    for c in [3, 6, 7, 8] {
        let (mut ue, _, now) = rejected(c, &[]);
        assert_eq!(ue.state(), EmmState::NoService(c));
        assert_eq!(ue.guti(), None);
        assert!(ue.tai_list().is_empty() && ue.default_bearer().is_none() && ue.k_enb().is_none());
        assert_eq!(ue.next_deadline(), Some(now + T3440));
        ue.on_released(now);
        assert_eq!(ue.initial_message(now), None, "#{c}: no attach");
    }
}

/// #356, §5.5.3.2.5 #9: GUTI, TAI list and eKSI deleted, EMM-DEREGISTERED,
/// attach -- plain, with the IMSI. #10 and #40: the bearer contexts go and
/// the UE attaches again with the GUTI it keeps.
#[test]
fn reject_9_10_40_lead_to_a_new_attach() {
    let (mut ue, _, now) = rejected(9, &[]);
    assert_eq!(
        (ue.state(), ue.guti(), ue.k_enb()),
        (EmmState::Deregistered, None, None)
    );
    ue.on_released(now);
    let a = ue.initial_message(now).unwrap();
    assert_eq!(&a[..3], &[0x07, 0x41, 0x71], "plain, no KSI, IMSI");
    for c in [10, 40] {
        let (mut ue, _, now) = rejected(c, &[]);
        assert_eq!(ue.state(), EmmState::Deregistered, "#{c}");
        assert!(ue.default_bearer().is_none(), "#{c}");
        assert_eq!(ue.next_deadline(), Some(now + T3440), "#{c}");
        ue.on_released(now);
        let a = ue.initial_message(now).unwrap();
        assert_eq!(a[0] >> 4, 1, "#{c}: integrity protected GUTI attach");
    }
}

/// #356, §5.5.3.2.5 #11, #14, #35: GUTI, TAI list and eKSI deleted and
/// EMM-DEREGISTERED.PLMN-SEARCH -- for this single-PLMN UE, no service.
/// #12: the same in LIMITED-SERVICE with the TA forbidden; in another TA the
/// UE attaches.
#[test]
fn reject_11_12_14_35_deregister() {
    for c in [11, 14, 35] {
        let (mut ue, _, now) = rejected(c, &[]);
        assert_eq!((ue.state(), ue.guti()), (EmmState::NoService(c), None));
        assert_eq!(ue.next_deadline(), Some(now + T3440), "#{c}");
        ue.on_serving_cell(2);
        assert_eq!(ue.initial_message(now), None, "#{c}");
    }
    let (mut ue, _, now) = rejected(12, &[]);
    assert_eq!(ue.state(), EmmState::NoService(12));
    ue.on_serving_cell(1);
    assert_eq!(ue.state(), EmmState::NoService(12), "TA 1 is forbidden");
    ue.on_serving_cell(2);
    assert_eq!(ue.state(), EmmState::Deregistered);
    ue.on_released(now);
    assert_eq!(&ue.initial_message(now).unwrap()[..3], &[0x07, 0x41, 0x71]);
}

/// #356, §5.5.3.2.5 #13 and #15: EMM-REGISTERED with no service in this TA:
/// the bearer contexts and GUTI stay, the TA leaves the TAI list and is
/// forbidden; a TA that is not starts a "TA updating" update.
#[test]
fn reject_13_15_keep_the_registration_until_another_ta() {
    for c in [13, 15] {
        let (mut ue, mut mme, now) = rejected(c, &[]);
        assert_eq!(ue.state(), EmmState::RegisteredLimited(c));
        assert!(ue.default_bearer().is_some() && ue.guti().is_some(), "#{c}");
        assert!(ue.tai_list().is_empty(), "#{c}: TA 1 removed");
        ue.on_released(now);
        assert_eq!(ue.initial_message(now), None, "#{c}");
        ue.on_serving_cell(1);
        assert_eq!(ue.initial_message(now), None, "#{c}: TA 1 forbidden");
        ue.on_serving_cell(2);
        let req = tau_request(&mut ue, &mut mme, now);
        assert_eq!(req[2] & 0x07, 0, "#{c}: TA updating");
    }
}

/// #356, §5.5.3.2.5 #22: with T3346 the update waits for it, the network's
/// value when the reject is protected, 15 to 30 minutes when plain; without
/// T3346 it is abnormal case d).
#[test]
fn reject_22_waits_for_t3346() {
    let (mut ue, mut mme, now) = rejected(22, &[0x5F, 0x01, 0x22]);
    assert_eq!(ue.state(), EmmState::AttemptingToUpdate);
    ue.on_released(now);
    let t3346 = now + Duration::from_secs(2 * 60);
    assert_eq!(ue.initial_message(t3346 - Duration::from_secs(1)), None);
    ue.on_timer(t3346);
    let req = tau_request(&mut ue, &mut mme, t3346);
    assert_eq!(req[2] & 0x07, 0b011, "still the periodic update");

    let (mut ue, mut mme, now) = idle_in_ta1();
    let now = now + T3412_ATTACH;
    ue.on_timer(now);
    tau_request(&mut ue, &mut mme, now);
    ue.on_downlink(&[0x07, 0x4B, 22, 0x5F, 0x01, 0x22], now);
    assert_eq!(ue.state(), EmmState::AttemptingToUpdate);
    ue.on_released(now);
    assert_eq!(
        ue.initial_message(now + Duration::from_secs(15 * 60 - 1)),
        None,
        "plain: at least 15 minutes"
    );
    let later = now + Duration::from_secs(30 * 60);
    ue.on_timer(later);
    assert!(ue.initial_message(later).is_some(), "at most 30 minutes");

    let (ue, _, _) = rejected(22, &[]);
    assert_eq!(ue.tau_attempts(), 1, "abnormal case d)");
}

/// #356, §5.5.3.2.5 #42: GUTI, TAI list and eKSI deleted, the counter at 5,
/// E-UTRA off for 2 T (TS 23.122: T = 60 min), then an attach.
#[test]
fn reject_42_attaches_after_two_hours() {
    let (mut ue, _, now) = rejected(42, &[]);
    assert_eq!(
        (ue.state(), ue.guti()),
        (EmmState::AttemptingToAttach, None)
    );
    ue.on_released(now);
    let two_t = now + Duration::from_secs(2 * 3600);
    ue.on_timer(two_t - Duration::from_secs(1));
    assert_eq!(ue.initial_message(two_t - Duration::from_secs(1)), None);
    ue.on_timer(two_t);
    assert!(ue.initial_message(two_t).is_some(), "attach");
}

/// #356, §4.4.4.2 / §5.5.3.2.5: a plain reject is processed unless its cause
/// is #25 or #78, which is discarded; #25 off a CSG cell is abnormal case d).
#[test]
fn plain_rejects_and_25_78() {
    for c in [25, 78] {
        let (mut ue, mut mme, now) = idle_in_ta1();
        ue.on_serving_cell(2);
        tau_request(&mut ue, &mut mme, now);
        assert_eq!(ue.on_downlink(&[0x07, 0x4B, c], now), NasOutput::default());
        assert_eq!(
            ue.state(),
            EmmState::TrackingAreaUpdatingInitiated,
            "#{c} discarded"
        );
    }
    let (ue, _, _) = rejected(25, &[]);
    assert_eq!(ue.tau_attempts(), 1);
    let (mut ue, mut mme, now) = idle_in_ta1();
    ue.on_serving_cell(2);
    tau_request(&mut ue, &mut mme, now);
    ue.on_downlink(&[0x07, 0x4B, 12], now);
    assert_eq!(
        ue.state(),
        EmmState::NoService(12),
        "plain #12 is processed"
    );
}

/// #356, §5.5.3.2.6 c), d) and 3) to 5): T3430 expiry releases the
/// connection; below 5 attempts, in the TAI list and EU1 UPDATED the UE stays
/// EMM-REGISTERED with T3411 and updates again on its expiry; the fifth
/// failure starts T3402 in ATTEMPTING-TO-UPDATE, whose expiry resets the
/// counter. #111 sets the counter to 5 at once.
#[test]
fn failed_updates_retry_on_t3411_then_t3402() {
    let (mut ue, mut mme, now) = idle_in_ta1();
    let mut t = now + T3412_ATTACH;
    ue.on_timer(t);
    for attempt in 1..=4 {
        tau_request(&mut ue, &mut mme, t);
        t += T3430;
        let out = ue.on_timer(t);
        assert!(out.release, "attempt {attempt}: released locally");
        ue.on_released(t);
        assert_eq!(ue.tau_attempts(), attempt);
        assert_eq!(ue.state(), EmmState::Registered, "3): NORMAL-SERVICE");
        assert_eq!(ue.initial_message(t), None, "T3411 runs");
        t += T3411;
        ue.on_timer(t);
    }
    tau_request(&mut ue, &mut mme, t);
    t += T3430;
    ue.on_timer(t);
    ue.on_released(t);
    assert_eq!(
        (ue.tau_attempts(), ue.state()),
        (5, EmmState::AttemptingToUpdate)
    );
    assert_eq!(ue.initial_message(t + T3411), None, "T3402, not T3411");
    t += T3402_DEFAULT;
    ue.on_timer(t);
    assert_eq!(ue.tau_attempts(), 0, "T3402 expiry resets the counter");
    let req = tau_request(&mut ue, &mut mme, t);
    assert_eq!(req[2] & 0x07, 0b011, "the periodic update, again");

    let (ue, _, _) = rejected(111, &[]);
    assert_eq!(
        (ue.tau_attempts(), ue.state()),
        (5, EmmState::AttemptingToUpdate)
    );
}

/// #356, §5.5.3.2.6 b) and 4): a release before the answer, for a TA outside
/// the list, is ATTEMPTING-TO-UPDATE with EU2 and T3411; a paging there is
/// answered with an update with the "active" flag (§5.5.3.2.2 r), T3411 or
/// not, and a new TA resets the counter and updates at once.
#[test]
fn a_lost_ta_update_attempts_again_and_answers_paging() {
    let (mut ue, mut mme, now) = idle_in_ta1();
    ue.on_serving_cell(2);
    tau_request(&mut ue, &mut mme, now);
    ue.on_released(now);
    assert_eq!(
        (ue.tau_attempts(), ue.state()),
        (1, EmmState::AttemptingToUpdate)
    );
    ue.on_uplink_data_pending();
    assert_eq!(ue.initial_message(now), None, "no user data, T3411 runs");
    assert!(ue.on_paging(PagedWith::STmsi));
    let req = tau_request(&mut ue, &mut mme, now);
    assert_eq!(req[2], 0x18, "TA updating with the active flag");
    assert!(
        ue.initial_message_answers_paging(),
        "Annex D Table D.1.1: MT access"
    );
    ue.on_released(now);
    assert_eq!(ue.tau_attempts(), 2);
    ue.on_serving_cell(3);
    assert_eq!(ue.tau_attempts(), 0, "a new TA resets the counter");
    tau_request(&mut ue, &mut mme, now);
    assert!(
        !ue.initial_message_answers_paging(),
        "not paged: originating signalling"
    );
}

/// #356 review, §5.5.1.2.4: the T3412 extended value IE (5E, GPRS timer 3)
/// of the ATTACH ACCEPT is T3412, over the T3412 value; a unit of 111
/// leaves the T3412 value in force, as on the TAU ACCEPT path.
#[test]
fn the_attach_accepts_t3412_extended_value_is_t3412() {
    // Unit 001 (1 h), value 2: 2 h
    let (ue, _, _) = crate::testing::attach_with(&[0x5E, 0x01, 0x22]);
    assert_eq!(ue.t3412(), Some(Duration::from_secs(2 * 3600)));
    let (ue, _, _) = crate::testing::attach_with(&[0x5E, 0x01, 0xE2]);
    assert_eq!(ue.t3412(), Some(T3412_ATTACH), "unit 111: the T3412 value");
    let (ue, _, _) = crate::testing::attach_with(&[]);
    assert_eq!(ue.t3412(), Some(T3412_ATTACH));
}

/// #356, §5.5.3.2.6 l): no update while T3346 runs; for a TA outside the list
/// the UE is ATTEMPTING-TO-UPDATE with EU2 until it expires.
#[test]
fn t3346_holds_the_update_back() {
    let (mut ue, mut mme, now) = idle_in_ta1();
    // SERVICE REJECT #22 with T3346 = 2 min.
    ue.on_uplink_data_pending();
    mme.verify_service_request(&ue.initial_message(now).unwrap());
    ue.on_connected();
    ue.on_downlink(&mme.protect(&[0x07, 0x4E, 22, 0x5F, 0x01, 0x22], 2), now);
    ue.on_released(now);
    ue.on_serving_cell(2);
    assert_eq!(ue.initial_message(now), None);
    assert_eq!(ue.state(), EmmState::AttemptingToUpdate);
    let t = now + Duration::from_secs(120);
    ue.on_timer(t);
    tau_request(&mut ue, &mut mme, t);
}

/// #356, §5.5.3.2.6 f) and g): during the update an IMSI DETACH REQUEST and a
/// GUTI REALLOCATION COMMAND are ignored; a "re-attach required" detach
/// aborts the update and is answered.
#[test]
fn detach_and_guti_reallocation_during_the_update() {
    let (mut ue, mut mme, now) = idle_in_ta1();
    ue.on_serving_cell(2);
    tau_request(&mut ue, &mut mme, now);
    assert_eq!(
        ue.on_downlink(&mme.protect(&[0x07, 0x45, 0x03], 2), now),
        NasOutput::default()
    );
    assert_eq!(
        ue.on_downlink(&mme.protect(&[0x07, 0x50], 2), now),
        NasOutput::default()
    );
    assert_eq!(ue.state(), EmmState::TrackingAreaUpdatingInitiated);
    let out = ue.on_downlink(&mme.protect(&[0x07, 0x45, 0x01], 2), now);
    assert_eq!(
        mme.unprotect(&out.uplink[0]),
        vec![0x07, 0x46],
        "DETACH ACCEPT"
    );
    assert_eq!(ue.state(), EmmState::Deregistered);
    assert_eq!(ue.next_deadline(), Some(now + T3440), "T3430 stopped");
}

/// #356, §5.5.3.2.6 m): a detach that is not switch-off waits for the update
/// to complete; switch-off aborts it.
#[test]
fn a_ue_detach_during_the_update() {
    let (mut ue, mut mme, now) = idle_in_ta1();
    ue.on_serving_cell(2);
    tau_request(&mut ue, &mut mme, now);
    assert_eq!(ue.detach(false, true, now), NasOutput::default());
    let out = ue.on_downlink(&accept(&mut mme, &NEW_GUTI), now);
    assert_eq!(out.uplink.len(), 2);
    assert_eq!(mme.unprotect(&out.uplink[0]), vec![0x07, 0x4A]);
    let detach = mme.unprotect(&out.uplink[1]);
    assert_eq!(&detach[..3], &[0x07, 0x45, 0x11], "EPS detach, KSI 1");
    assert_eq!(&detach[4..], &NEW_GUTI[2..], "the new GUTI");
    assert_eq!(ue.state(), EmmState::DeregisteredInitiated);

    let (mut ue, mut mme, now) = idle_in_ta1();
    ue.on_serving_cell(2);
    tau_request(&mut ue, &mut mme, now);
    let out = ue.detach(true, true, now);
    let detach = mme.unprotect(&out.uplink[0]);
    assert_eq!(&detach[..3], &[0x07, 0x45, 0x19], "switch off");
    assert_eq!(&detach[3..], &OLD_GUTI, "the GUTI of the aborted update");
    assert_eq!(ue.state(), EmmState::Detached);
}
