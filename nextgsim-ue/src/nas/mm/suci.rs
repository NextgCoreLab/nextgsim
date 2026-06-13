//! SUCI (Subscription Concealed Identifier) construction
//!
//! Builds the 5GS mobile identity IE contents for a SUCI per TS 24.501
//! Section 9.11.3.4:
//!
//! ```text
//! octet 1      : SUPI format (bits 7-5) | spare | type of identity (001 = SUCI)
//! octets 2-4   : home network PLMN (MCC/MNC, TBCD)
//! octets 5-6   : routing indicator (TBCD, 0xF filler)
//! octet 7      : protection scheme identifier (bits 4-1)
//! octet 8      : home network public key identifier
//! octets 9..   : scheme output
//! ```
//!
//! The scheme output is the cleartext MSIN (TBCD) for the null scheme, or the
//! ECIES output `ephemeral public key || ciphertext || MAC tag` for Profile A
//! (X25519) and Profile B (secp256r1) per TS 33.501 Annex C.

use nextgsim_common::config::UeConfig;
use nextgsim_crypto::ecies::{
    generate_suci_profile_a, generate_suci_profile_b, parse_p256_public_key, X25519_KEY_SIZE,
};
use thiserror::Error;

use super::orchestrator::encode_plmn_bcd;

/// Null protection scheme (TS 33.501 Annex C.2)
pub const PROTECTION_SCHEME_NULL: u8 = 0;
/// ECIES Profile A, X25519 (TS 33.501 Annex C.3.4.1)
pub const PROTECTION_SCHEME_PROFILE_A: u8 = 1;
/// ECIES Profile B, secp256r1 (TS 33.501 Annex C.3.4.2)
pub const PROTECTION_SCHEME_PROFILE_B: u8 = 2;

/// Errors raised when the SUCI cannot be built from the UE configuration.
///
/// There is deliberately no fallback to the null scheme: a UE configured for
/// ECIES concealment must not silently reveal its SUPI (TS 33.501 Section 6.12).
#[derive(Debug, Error)]
pub enum SuciBuildError {
    /// The configured protection scheme is not supported
    #[error(
        "unsupported SUCI protection scheme {0} (supported: 0=null, 1=Profile A, 2=Profile B)"
    )]
    UnsupportedScheme(u8),
    /// ECIES schemes require a non-zero home network public key identifier
    #[error("home network public key id {key_id} is invalid for protection scheme {scheme} (must be 1-255)")]
    InvalidKeyId {
        /// Configured protection scheme
        scheme: u8,
        /// Configured home network public key identifier
        key_id: u8,
    },
    /// The configured home network public key is missing or malformed
    #[error("invalid home network public key for protection scheme {scheme}: {reason}")]
    InvalidHomeNetworkKey {
        /// Configured protection scheme
        scheme: u8,
        /// Why the key was rejected
        reason: String,
    },
    /// The configured routing indicator is not 1-4 decimal digits
    #[error("invalid routing indicator '{0}' (must be 1-4 decimal digits)")]
    InvalidRoutingIndicator(String),
    /// The ECIES concealment itself failed
    #[error("ECIES concealment failed: {0}")]
    Concealment(String),
}

/// Build the SUCI 5GS mobile identity IE contents from the UE configuration.
///
/// Selects the protection scheme from `config.protection_scheme` and conceals
/// the MSIN with a fresh ephemeral key pair per call for the ECIES profiles.
///
/// # Errors
/// Returns a [`SuciBuildError`] if the scheme is unsupported, the home network
/// public key id/key material is invalid, or concealment fails. The caller
/// must treat this as fatal for identity signalling; there is no null-scheme
/// fallback.
pub fn build_suci(config: &UeConfig) -> Result<Vec<u8>, SuciBuildError> {
    let msin_tbcd = encode_msin_tbcd(&msin_from_config(config));

    let mut data = Vec::with_capacity(8 + msin_tbcd.len());
    // SUPI format IMSI (000), type of identity SUCI (001)
    data.push(0x01);
    // Home network PLMN (TBCD, 3 octets)
    data.extend_from_slice(&encode_plmn_bcd(
        config.hplmn.mcc,
        config.hplmn.mnc,
        config.hplmn.long_mnc,
    ));
    // Routing indicator (TBCD, 2 octets); "0000" when not configured
    let routing_indicator = config.routing_indicator.as_deref().unwrap_or("0000");
    data.extend_from_slice(&encode_routing_indicator(routing_indicator)?);

    let scheme = config.protection_scheme;
    let key_id = config.home_network_public_key_id;
    match scheme {
        PROTECTION_SCHEME_NULL => {
            // Protection scheme id and home network public key id are
            // separate octets; both are 0 for the null scheme.
            data.push(PROTECTION_SCHEME_NULL);
            data.push(0x00);
            data.extend_from_slice(&msin_tbcd);
        }
        PROTECTION_SCHEME_PROFILE_A => {
            if key_id == 0 {
                return Err(SuciBuildError::InvalidKeyId { scheme, key_id });
            }
            let hn_pub: [u8; X25519_KEY_SIZE] = config
                .home_network_public_key
                .as_slice()
                .try_into()
                .map_err(|_| SuciBuildError::InvalidHomeNetworkKey {
                    scheme,
                    reason: format!(
                        "expected {} bytes (X25519), got {}",
                        X25519_KEY_SIZE,
                        config.home_network_public_key.len()
                    ),
                })?;
            let scheme_output = generate_suci_profile_a(&msin_tbcd, &hn_pub)
                .map_err(|e| SuciBuildError::Concealment(e.to_string()))?;
            data.push(PROTECTION_SCHEME_PROFILE_A);
            data.push(key_id);
            data.extend_from_slice(&scheme_output);
        }
        PROTECTION_SCHEME_PROFILE_B => {
            if key_id == 0 {
                return Err(SuciBuildError::InvalidKeyId { scheme, key_id });
            }
            let hn_pub = parse_p256_public_key(&config.home_network_public_key).map_err(|e| {
                SuciBuildError::InvalidHomeNetworkKey {
                    scheme,
                    reason: e.to_string(),
                }
            })?;
            let scheme_output = generate_suci_profile_b(&msin_tbcd, &hn_pub)
                .map_err(|e| SuciBuildError::Concealment(e.to_string()))?;
            data.push(PROTECTION_SCHEME_PROFILE_B);
            data.push(key_id);
            data.extend_from_slice(&scheme_output);
        }
        other => return Err(SuciBuildError::UnsupportedScheme(other)),
    }

    Ok(data)
}

/// Extract the MSIN digits from the configured SUPI (IMSI = MCC | MNC | MSIN).
fn msin_from_config(config: &UeConfig) -> String {
    let mcc_len = 3;
    let mnc_len = if config.hplmn.long_mnc || config.hplmn.mnc > 99 {
        3
    } else {
        2
    };
    let plmn_len = mcc_len + mnc_len;
    if let Some(ref supi) = config.supi {
        if supi.value.len() > plmn_len {
            return supi.value[plmn_len..].to_string();
        }
    }
    "0000000001".to_string()
}

/// Encode MSIN digits as TBCD (first digit in the low nibble, 0xF filler in
/// the high nibble of the last octet for an odd number of digits).
fn encode_msin_tbcd(msin: &str) -> Vec<u8> {
    let digits: Vec<u8> = msin
        .chars()
        .filter_map(|c| c.to_digit(10).map(|d| d as u8))
        .collect();
    digits
        .chunks(2)
        .map(|chunk| {
            let low = chunk[0];
            let high = chunk.get(1).copied().unwrap_or(0xF);
            (high << 4) | low
        })
        .collect()
}

/// Encode the routing indicator as 2 TBCD octets (TS 24.501 Section 9.11.3.4:
/// 1-4 digits, unused half-octets coded as 0xF).
fn encode_routing_indicator(ri: &str) -> Result<[u8; 2], SuciBuildError> {
    if ri.is_empty() || ri.len() > 4 || !ri.bytes().all(|b| b.is_ascii_digit()) {
        return Err(SuciBuildError::InvalidRoutingIndicator(ri.to_string()));
    }
    let digit = |i: usize| -> u8 { ri.as_bytes().get(i).map_or(0xF, |b| b - b'0') };
    Ok([(digit(1) << 4) | digit(0), (digit(3) << 4) | digit(2)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use nextgsim_common::config::OpType;
    use nextgsim_common::types::{Plmn, Supi, SupiType};
    use nextgsim_crypto::ecies::{
        decode_suci_profile_a, decode_suci_profile_b, EciesProfileBKeyPair,
    };

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    /// TS 33.501 Annex C.4.3 home network key pair (Profile A / X25519)
    const HN_PRIV_A: &str = "c53c22208b61860b06c62e5406a7b330c2b577aa5558981510d128247d38bd1d";
    const HN_PUB_A: &str = "5a8d38864820197c3394b92613b20b91633cbd897119273bf8e4a6f4eec0a650";
    /// TS 33.501 Annex C.4.4 home network key pair (Profile B / secp256r1)
    const HN_PRIV_B: &str = "f1ab1074477ebcc7f554ea1c5fc368b1616730155e0041ac447d6301975fecda";
    const HN_PUB_B: &str = "0272da71976234ce833a6907425867b82e074d44ef907dfb4b3e21c1c2256ebcd1";

    fn test_config(scheme: u8, key_id: u8, key: Vec<u8>) -> UeConfig {
        UeConfig {
            hplmn: Plmn {
                mcc: 999,
                mnc: 70,
                long_mnc: false,
            },
            supi: Some(Supi {
                supi_type: SupiType::Imsi,
                value: "999700000000001".to_string(),
            }),
            protection_scheme: scheme,
            home_network_public_key_id: key_id,
            home_network_public_key: key,
            routing_indicator: None,
            key: [0x11; 16],
            op: [0x22; 16],
            op_type: OpType::Opc,
            ..Default::default()
        }
    }

    #[test]
    fn test_null_scheme_exact_bytes() {
        let suci = build_suci(&test_config(0, 0, Vec::new())).expect("build");
        // type(1) | PLMN 999/70(3) | RI 0000(2) | scheme 0(1) | key id 0(1) | MSIN(5)
        assert_eq!(
            suci,
            vec![
                0x01, // SUPI format IMSI, type SUCI
                0x99, 0xF9, 0x07, // PLMN 999/70 (TBCD)
                0x00, 0x00, // routing indicator "0000"
                0x00, // protection scheme: null
                0x00, // home network public key id: 0
                0x00, 0x00, 0x00, 0x00, 0x10, // MSIN 0000000001 (TBCD)
            ]
        );
    }

    #[test]
    fn test_null_scheme_msin_nibble_order() {
        // MSIN "1234567890": digit 1 in the low nibble of the first octet
        let mut config = test_config(0, 0, Vec::new());
        config.supi = Some(Supi {
            supi_type: SupiType::Imsi,
            value: "999701234567890".to_string(),
        });
        let suci = build_suci(&config).expect("build");
        assert_eq!(&suci[8..], &[0x21, 0x43, 0x65, 0x87, 0x09]);
    }

    #[test]
    fn test_null_scheme_odd_msin_filler() {
        // 9-digit MSIN: last octet has 0xF filler in the high nibble
        let mut config = test_config(0, 0, Vec::new());
        config.supi = Some(Supi {
            supi_type: SupiType::Imsi,
            value: "99970123456789".to_string(),
        });
        let suci = build_suci(&config).expect("build");
        assert_eq!(&suci[8..], &[0x21, 0x43, 0x65, 0x87, 0xF9]);
    }

    #[test]
    fn test_routing_indicator_from_config() {
        let mut config = test_config(0, 0, Vec::new());
        config.routing_indicator = Some("12".to_string());
        let suci = build_suci(&config).expect("build");
        assert_eq!(&suci[4..6], &[0x21, 0xFF]);

        config.routing_indicator = Some("1234".to_string());
        let suci = build_suci(&config).expect("build");
        assert_eq!(&suci[4..6], &[0x21, 0x43]);
    }

    #[test]
    fn test_routing_indicator_invalid() {
        for bad in ["", "12345", "12a4"] {
            let mut config = test_config(0, 0, Vec::new());
            config.routing_indicator = Some(bad.to_string());
            assert!(
                matches!(
                    build_suci(&config),
                    Err(SuciBuildError::InvalidRoutingIndicator(_))
                ),
                "routing indicator '{bad}' must be rejected"
            );
        }
    }

    #[test]
    fn test_profile_a_conceals_and_deconceals() {
        let config = test_config(1, 1, unhex(HN_PUB_A));
        let suci = build_suci(&config).expect("build");

        // Header fields per TS 24.501 9.11.3.4
        assert_eq!(suci[0], 0x01);
        assert_eq!(&suci[1..4], &[0x99, 0xF9, 0x07]);
        assert_eq!(&suci[4..6], &[0x00, 0x00]);
        assert_eq!(suci[6], PROTECTION_SCHEME_PROFILE_A);
        assert_eq!(suci[7], 1);

        // Scheme output: eph pub (32) || ciphertext (5) || MAC (8)
        let scheme_output = &suci[8..];
        assert_eq!(scheme_output.len(), 32 + 5 + 8);

        // Deconceal with the Annex C.4.3 home network private key
        let hn_priv: [u8; 32] = unhex(HN_PRIV_A).try_into().unwrap();
        let msin = decode_suci_profile_a(scheme_output, &hn_priv).expect("deconceal");
        assert_eq!(msin, unhex("0000000010")); // MSIN 0000000001 in TBCD
    }

    #[test]
    fn test_profile_b_conceals_and_deconceals() {
        let config = test_config(2, 2, unhex(HN_PUB_B));
        let suci = build_suci(&config).expect("build");

        assert_eq!(suci[6], PROTECTION_SCHEME_PROFILE_B);
        assert_eq!(suci[7], 2);

        // Scheme output: compressed eph pub (33) || ciphertext (5) || MAC (8)
        let scheme_output = &suci[8..];
        assert_eq!(scheme_output.len(), 33 + 5 + 8);

        let hn_priv: [u8; 32] = unhex(HN_PRIV_B).try_into().unwrap();
        let hn_keypair = EciesProfileBKeyPair::from_secret_bytes(&hn_priv).expect("hn key");
        let msin =
            decode_suci_profile_b(scheme_output, hn_keypair.secret_key()).expect("deconceal");
        assert_eq!(msin, unhex("0000000010"));
    }

    #[test]
    fn test_profile_a_fresh_ephemeral_per_concealment() {
        let config = test_config(1, 1, unhex(HN_PUB_A));
        let suci1 = build_suci(&config).expect("build 1");
        let suci2 = build_suci(&config).expect("build 2");
        // Fresh ephemeral key pair per concealment: scheme outputs must differ
        assert_ne!(&suci1[8..], &suci2[8..]);
        // ... but both must still deconceal to the same MSIN
        let hn_priv: [u8; 32] = unhex(HN_PRIV_A).try_into().unwrap();
        assert_eq!(
            decode_suci_profile_a(&suci1[8..], &hn_priv).expect("deconceal 1"),
            decode_suci_profile_a(&suci2[8..], &hn_priv).expect("deconceal 2"),
        );
    }

    #[test]
    fn test_tampered_mac_fails_deconcealment() {
        let config = test_config(1, 1, unhex(HN_PUB_A));
        let mut suci = build_suci(&config).expect("build");
        let last = suci.len() - 1;
        suci[last] ^= 0xFF; // flip a MAC tag bit
        let hn_priv: [u8; 32] = unhex(HN_PRIV_A).try_into().unwrap();
        assert!(decode_suci_profile_a(&suci[8..], &hn_priv).is_err());
    }

    #[test]
    fn test_wrong_home_network_key_fails_deconcealment() {
        let config = test_config(1, 1, unhex(HN_PUB_A));
        let suci = build_suci(&config).expect("build");
        let wrong_priv = [0x42u8; 32];
        assert!(decode_suci_profile_a(&suci[8..], &wrong_priv).is_err());
    }

    #[test]
    fn test_unsupported_scheme_rejected() {
        let config = test_config(3, 1, unhex(HN_PUB_A));
        assert!(matches!(
            build_suci(&config),
            Err(SuciBuildError::UnsupportedScheme(3))
        ));
    }

    #[test]
    fn test_ecies_key_id_zero_rejected() {
        for scheme in [1u8, 2u8] {
            let key = if scheme == 1 {
                unhex(HN_PUB_A)
            } else {
                unhex(HN_PUB_B)
            };
            let config = test_config(scheme, 0, key);
            assert!(
                matches!(
                    build_suci(&config),
                    Err(SuciBuildError::InvalidKeyId { key_id: 0, .. })
                ),
                "key id 0 must be rejected for scheme {scheme}"
            );
        }
    }

    #[test]
    fn test_bad_key_material_rejected() {
        // Profile A: wrong length
        let config = test_config(1, 1, vec![0xAA; 16]);
        assert!(matches!(
            build_suci(&config),
            Err(SuciBuildError::InvalidHomeNetworkKey { scheme: 1, .. })
        ));
        // Profile A: missing key
        let config = test_config(1, 1, Vec::new());
        assert!(matches!(
            build_suci(&config),
            Err(SuciBuildError::InvalidHomeNetworkKey { scheme: 1, .. })
        ));
        // Profile B: not a curve point
        let config = test_config(2, 2, vec![0xFF; 33]);
        assert!(matches!(
            build_suci(&config),
            Err(SuciBuildError::InvalidHomeNetworkKey { scheme: 2, .. })
        ));
    }

    #[test]
    fn test_three_digit_mnc_plmn_encoding() {
        let mut config = test_config(0, 0, Vec::new());
        config.hplmn = Plmn {
            mcc: 1,
            mnc: 1,
            long_mnc: true,
        };
        config.supi = Some(Supi {
            supi_type: SupiType::Imsi,
            value: "0010010000000099".to_string(),
        });
        let suci = build_suci(&config).expect("build");
        // MCC 001 / MNC 001 with 3-digit MNC: [0x00, 0x11, 0x00]
        assert_eq!(&suci[1..4], &[0x00, 0x11, 0x00]);
        assert_eq!(&suci[8..], &[0x00, 0x00, 0x00, 0x00, 0x99]);
    }
}
