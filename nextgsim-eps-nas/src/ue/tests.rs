//! The attach procedure against a scripted MME. The MME side is written here
//! from TS 24.301 / TS 33.401 directly -- its protection and AUTN are built
//! with the crypto primitives, not with [`EpsSecurityContext`] -- so a
//! symmetric misreading in `security.rs` cannot pass.

use super::*;
use nextgsim_crypto::eps_kdf::{derive_algorithm_key, derive_kasme, derive_kenb, EpsAlgorithmType};
use nextgsim_crypto::milenage::Milenage;
use nextgsim_crypto::nea::nea2_encrypt;
use nextgsim_crypto::nia::nia2_compute_mac;

const K: [u8; 16] = [
    0x46, 0x5B, 0x5C, 0xE8, 0xB1, 0x99, 0xB4, 0x9F, 0xAA, 0x5F, 0x0A, 0x2E, 0xE2, 0x38, 0xA6, 0xBC,
];
const OPC: [u8; 16] = [
    0xE8, 0xED, 0x28, 0x9D, 0xEB, 0xA9, 0x52, 0xE4, 0x28, 0x3B, 0x54, 0xE8, 0x8E, 0x6D, 0x83, 0x4D,
];
const IMSI: &str = "999700000000001";

fn cfg() -> EpsNasConfig {
    EpsNasConfig {
        imsi: IMSI.into(),
        plmn: Plmn::new(999, 70, false),
        credentials: UsimCredentials { k: K, opc: OPC },
        initial_sqn: 0,
        imeisv: Some("3534900698733019".into()),
        eea: vec![0, 1, 2],
        eia: vec![0, 1, 2],
        apn: None,
        pdn_type: PdnType::Ipv4,
    }
}

/// The MME's view: one authentication vector and the NAS context it keys.
struct Mme {
    rand: [u8; 16],
    autn: [u8; 16],
    xres: [u8; 8],
    kasme: [u8; 32],
    knas_int: [u8; 16],
    knas_enc: [u8; 16],
    dl: u32,
    ul: u32,
}

impl Mme {
    /// An AV for SQN `sqn` as the HSS builds it (TS 33.102 §6.3.2, TS 33.401
    /// §6.1.1 with the separation bit set in AMF 0x8000).
    fn new(sqn: u64) -> Self {
        Self::with_rand(sqn, [0x23; 16])
    }

    /// An AV for SQN `sqn` with a chosen RAND.
    fn with_rand(sqn: u64, rand: [u8; 16]) -> Self {
        let m = Milenage::new(&K, &OPC);
        let sqn_b: [u8; 6] = sqn.to_be_bytes()[2..].try_into().unwrap();
        let ak = m.f5(&rand);
        let mut autn = [0u8; 16];
        for i in 0..6 {
            autn[i] = sqn_b[i] ^ ak[i];
        }
        autn[6..8].copy_from_slice(&[0x80, 0x00]);
        autn[8..].copy_from_slice(&m.f1(&rand, &sqn_b, &[0x80, 0x00]));
        let kasme = derive_kasme(
            &m.f3(&rand),
            &m.f4(&rand),
            &Plmn::new(999, 70, false).encode(),
            autn[..6].try_into().unwrap(),
        );
        Self {
            rand,
            autn,
            xres: m.f2(&rand),
            kasme,
            knas_int: derive_algorithm_key(&kasme, EpsAlgorithmType::NasInt, 2),
            knas_enc: derive_algorithm_key(&kasme, EpsAlgorithmType::NasEnc, 2),
            dl: 0,
            ul: 0,
        }
    }

    fn authentication_request(&self) -> Vec<u8> {
        let mut m = vec![0x07, 0x52, 0x01];
        m.extend_from_slice(&self.rand);
        m.push(16);
        m.extend_from_slice(&self.autn);
        m
    }

    /// TS 24.301 §9.3.1 with EEA2/EIA2, DIRECTION 1.
    fn protect(&mut self, plain: &[u8], ht: u8) -> Vec<u8> {
        let count = self.dl;
        let mut body = plain.to_vec();
        if ht == 2 || ht == 4 {
            nea2_encrypt(count, 0, 1, &self.knas_enc, &mut body);
        }
        let mut input = vec![(count & 0xFF) as u8];
        input.extend_from_slice(&body);
        let mac = nia2_compute_mac(count, 0, 1, &self.knas_int, &input);
        self.dl += 1;
        let mut out = vec![(ht << 4) | 7];
        out.extend_from_slice(&mac);
        out.extend_from_slice(&input);
        out
    }

    /// Verify and decipher an uplink protected message.
    fn unprotect(&mut self, pdu: &[u8]) -> Vec<u8> {
        let count = self.ul;
        let mac = nia2_compute_mac(count, 0, 0, &self.knas_int, &pdu[5..]);
        assert_eq!(&pdu[1..5], &mac, "uplink MAC at COUNT {count}");
        assert_eq!(pdu[5], (count & 0xFF) as u8);
        let mut body = pdu[6..].to_vec();
        let ht = pdu[0] >> 4;
        if ht == 2 || ht == 4 {
            nea2_encrypt(count, 0, 0, &self.knas_enc, &mut body);
        }
        self.ul += 1;
        body
    }

    fn security_mode_command(&mut self, replayed: [u8; 2], imeisv: bool) -> Vec<u8> {
        let mut m = vec![0x07, 0x5D, 0x22, 0x01, 0x02, replayed[0], replayed[1]];
        if imeisv {
            m.push(0xC1);
        }
        self.dl = 0;
        self.ul = 0;
        self.protect(&m, 3)
    }

    fn attach_accept(&mut self, pti: u8) -> Vec<u8> {
        // ESM: EBI 5, PTI, 0xC1, QoS LV {QCI 9}, APN LV "internet", PDN address
        // LV {IPv4 10.45.0.2}.
        let mut esm = vec![0x52, pti, 0xC1, 0x01, 0x09];
        let apn = crate::ie::encode_apn("internet");
        esm.push(apn.len() as u8);
        esm.extend_from_slice(&apn);
        esm.extend_from_slice(&[0x05, 0x01, 10, 45, 0, 2]);
        let mut m = vec![
            0x07, 0x42, 0x01, 0x49, 0x06, 0x00, 0x99, 0xF9, 0x07, 0x00, 0x01,
        ];
        m.extend_from_slice(&(esm.len() as u16).to_be_bytes());
        m.extend_from_slice(&esm);
        m.extend_from_slice(&[
            0x50, 0x0B, 0xF6, 0x99, 0xF9, 0x07, 0x00, 0x02, 0x01, 0xC0, 0x00, 0x00, 0x01,
        ]);
        self.protect(&m, 2)
    }
}

/// Run the attach up to and including ATTACH COMPLETE; returns the UE, the
/// MME and the decoded ATTACH COMPLETE.
fn attach() -> (EpsNasUe, Mme, Vec<u8>) {
    let now = Instant::now();
    let mut ue = EpsNasUe::new(cfg());
    let req = ue.initial_message(now).expect("attach due");
    assert_eq!(&req[..3], &[0x07, 0x41, 0x71], "plain, KSI 7, EPS attach");
    assert_eq!(ue.state(), EmmState::RegisteredInitiated);
    // PTI from the piggybacked PDN CONNECTIVITY REQUEST: octet 2 of the ESM
    // container, which starts after the 2-octet LV-E length at octet 15.
    let pti = req[18];
    assert_eq!(req[17], 0x02, "ESM container starts with PD 2, EBI 0");
    let mut mme = Mme::new(0x21);

    let out = ue.on_downlink(&mme.authentication_request(), now);
    assert_eq!(out.uplink.len(), 1);
    let resp = &out.uplink[0];
    assert_eq!(
        &resp[..3],
        &[0x07, 0x53, 8],
        "plain AUTHENTICATION RESPONSE, RES of 8"
    );
    assert_eq!(&resp[3..], &mme.xres, "RES = XRES");

    let out = ue.on_downlink(&mme.security_mode_command([0xE0, 0xE0], true), now);
    let smc_complete = &out.uplink[0];
    assert_eq!(
        smc_complete[0], 0x47,
        "header type 4: integrity, ciphered, new context"
    );
    let plain = mme.unprotect(smc_complete);
    assert_eq!(&plain[..2], &[0x07, 0x5E]);
    assert_eq!(plain[2], 0x23, "IMEISV requested, IMEISV sent");

    let out = ue.on_downlink(&mme.attach_accept(pti), now);
    let complete = mme.unprotect(&out.uplink[0]);
    (ue, mme, complete)
}

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
    // Answered inside the new security context, so unprotect before reading.
    let out = ue.on_downlink(&mme.authentication_request(), now);
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
