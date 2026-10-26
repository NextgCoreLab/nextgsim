//! A scripted MME for tests of the UE side (#351 review): the attach and the
//! MME's NAS protection, written from TS 24.301 / TS 33.401 directly -- its
//! protection and AUTN are built with the crypto primitives, not with
//! [`EpsSecurityContext`](crate::security::EpsSecurityContext) -- so a
//! symmetric misreading in `security.rs` cannot pass.
//!
//! Built for this crate's tests and, with the `testing` feature, for crates that
//! drive an [`EpsNasUe`] (nextgsim-lte-ue). Not part of the production API.

#![allow(clippy::missing_panics_doc, missing_docs)]

use std::time::Instant;

use nextgsim_common::Plmn;
use nextgsim_crypto::eps_kdf::{derive_algorithm_key, derive_kasme, EpsAlgorithmType};
use nextgsim_crypto::milenage::Milenage;
use nextgsim_crypto::nea::nea2_encrypt;
use nextgsim_crypto::nia::nia2_compute_mac;

use crate::aka::UsimCredentials;
use crate::ie::PdnType;
use crate::{EmmState, EpsNasConfig, EpsNasUe};

pub const K: [u8; 16] = [
    0x46, 0x5B, 0x5C, 0xE8, 0xB1, 0x99, 0xB4, 0x9F, 0xAA, 0x5F, 0x0A, 0x2E, 0xE2, 0x38, 0xA6, 0xBC,
];
pub const OPC: [u8; 16] = [
    0xE8, 0xED, 0x28, 0x9D, 0xEB, 0xA9, 0x52, 0xE4, 0x28, 0x3B, 0x54, 0xE8, 0x8E, 0x6D, 0x83, 0x4D,
];
pub const IMSI: &str = "999700000000001";

pub fn cfg() -> EpsNasConfig {
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
pub struct Mme {
    pub rand: [u8; 16],
    pub autn: [u8; 16],
    pub xres: [u8; 8],
    pub kasme: [u8; 32],
    pub knas_int: [u8; 16],
    pub knas_enc: [u8; 16],
    pub dl: u32,
    pub ul: u32,
}

impl Mme {
    /// An AV for SQN `sqn` as the HSS builds it (TS 33.102 §6.3.2, TS 33.401
    /// §6.1.1 with the separation bit set in AMF 0x8000).
    pub fn new(sqn: u64) -> Self {
        Self::with_rand(sqn, [0x23; 16])
    }

    /// An AV for SQN `sqn` with a chosen RAND.
    pub fn with_rand(sqn: u64, rand: [u8; 16]) -> Self {
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

    pub fn authentication_request(&self) -> Vec<u8> {
        let mut m = vec![0x07, 0x52, 0x01];
        m.extend_from_slice(&self.rand);
        m.push(16);
        m.extend_from_slice(&self.autn);
        m
    }

    /// TS 24.301 §9.3.1 with EEA2/EIA2, DIRECTION 1.
    pub fn protect(&mut self, plain: &[u8], ht: u8) -> Vec<u8> {
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
    pub fn unprotect(&mut self, pdu: &[u8]) -> Vec<u8> {
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

    /// Verify a SERVICE REQUEST (§8.2.25): KSI and the low 5 bits of the
    /// uplink NAS COUNT, and the short MAC -- the 2 low octets of the EIA2 MAC
    /// over the first two octets (§9.9.3.28). Returns the COUNT it used.
    pub fn verify_service_request(&mut self, sr: &[u8]) -> u32 {
        let count = self.ul;
        assert_eq!(sr.len(), 4, "SERVICE REQUEST is 4 octets");
        assert_eq!(sr[0], 0xC7, "security header type 12, EMM");
        assert_eq!(
            sr[1],
            (1 << 5) | (count & 0x1F) as u8,
            "KSI 1, sequence number"
        );
        let mac = nia2_compute_mac(count, 0, 0, &self.knas_int, &sr[..2]);
        assert_eq!(&sr[2..], &mac[2..], "short MAC at COUNT {count}");
        self.ul += 1;
        count
    }

    pub fn security_mode_command(&mut self, replayed: [u8; 2], imeisv: bool) -> Vec<u8> {
        let mut m = vec![0x07, 0x5D, 0x22, 0x01, 0x02, replayed[0], replayed[1]];
        if imeisv {
            m.push(0xC1);
        }
        self.dl = 0;
        self.ul = 0;
        self.protect(&m, 3)
    }

    pub fn attach_accept(&mut self, pti: u8) -> Vec<u8> {
        self.attach_accept_with(pti, &[])
    }

    /// ATTACH ACCEPT with `extra` optional IEs after the GUTI.
    pub fn attach_accept_with(&mut self, pti: u8, extra: &[u8]) -> Vec<u8> {
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
        m.extend_from_slice(extra);
        self.protect(&m, 2)
    }
}

/// Run the attach up to and including ATTACH COMPLETE; returns the UE, the
/// MME and the decoded ATTACH COMPLETE.
pub fn attach() -> (EpsNasUe, Mme, Vec<u8>) {
    attach_with(&[])
}

/// [`attach`], with `extra` optional IEs in the ATTACH ACCEPT.
pub fn attach_with(extra: &[u8]) -> (EpsNasUe, Mme, Vec<u8>) {
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

    let out = ue.on_downlink(&mme.attach_accept_with(pti, extra), now);
    let complete = mme.unprotect(&out.uplink[0]);
    (ue, mme, complete)
}

/// Attach, then lose the connection (an MME restart takes the cell down):
/// EMM-REGISTERED in EMM-IDLE.
pub fn registered_idle() -> (EpsNasUe, Mme, Instant) {
    let (mut ue, mme, _) = attach();
    let now = Instant::now();
    ue.on_released(now);
    assert_eq!(ue.state(), EmmState::Registered);
    (ue, mme, now)
}
