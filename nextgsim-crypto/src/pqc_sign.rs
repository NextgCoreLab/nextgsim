//! ML-DSA (CRYSTALS-Dilithium) post-quantum digital signature scheme
//!
//! Implements ML-DSA (Module-Lattice-Based Digital Signature Algorithm)
//! as standardized in FIPS 204, supporting three security levels:
//! - ML-DSA-44: NIST Level 2 security
//! - ML-DSA-65: NIST Level 3 security (recommended)
//! - ML-DSA-87: NIST Level 5 security
//!
//! ML-DSA provides quantum-resistant digital signatures for authentication
//! and integrity in 5G/6G security protocols.

use ml_dsa::signature::Signer;
use ml_dsa::signature::Verifier;
use ml_dsa::{MlDsa44, MlDsa65, MlDsa87};
use rand::rngs::OsRng;
use rand::RngCore;
use thiserror::Error;

/// ML-DSA security level parameter sets
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlDsaLevel {
    /// ML-DSA-44: NIST Level 2 security
    MlDsa44,
    /// ML-DSA-65: NIST Level 3 security (recommended)
    MlDsa65,
    /// ML-DSA-87: NIST Level 5 security (highest security)
    MlDsa87,
}

/// ML-DSA error types
#[derive(Debug, Error)]
pub enum MlDsaError {
    /// Signing failed
    #[error("Signing failed: {0}")]
    SigningFailed(String),
    /// Verification failed
    #[error("Verification failed: {0}")]
    VerificationFailed(String),
    /// Invalid key data
    #[error("Invalid key data: {0}")]
    InvalidKeyData(String),
    /// Invalid signature data
    #[error("Invalid signature data: {0}")]
    InvalidSignature(String),
}

/// Result type for ML-DSA operations
pub type MlDsaResult<T> = Result<T, MlDsaError>;

/// ML-DSA key pair wrapping signing and verifying keys
///
/// Stores a 32-byte seed for the signing key (from which the full signing key
/// can be deterministically derived via `from_seed`) and the encoded verifying key.
#[derive(Clone)]
pub struct MlDsaKeyPair {
    /// The security level of this key pair
    level: MlDsaLevel,
    /// 32-byte seed for signing key reconstruction
    signing_key_seed: Vec<u8>,
    /// Serialized verifying (public) key
    verifying_key: Vec<u8>,
}

impl MlDsaKeyPair {
    /// Get the security level of this key pair
    pub fn level(&self) -> MlDsaLevel {
        self.level
    }

    /// Get the signing key seed bytes
    pub fn signing_key(&self) -> &[u8] {
        &self.signing_key_seed
    }

    /// Get the verifying (public) key bytes
    pub fn verifying_key(&self) -> &[u8] {
        &self.verifying_key
    }
}

/// Generate an ML-DSA key pair at the specified security level
///
/// # Arguments
/// * `level` - The ML-DSA security level to use
///
/// # Returns
/// A new ML-DSA key pair
pub fn ml_dsa_generate_keypair(level: MlDsaLevel) -> MlDsaKeyPair {
    // Generate a random 32-byte seed
    let mut seed = [0u8; 32];
    OsRng.fill_bytes(&mut seed);

    let vk_bytes = match level {
        MlDsaLevel::MlDsa44 => {
            let sk = ml_dsa::SigningKey::<MlDsa44>::from_seed((&seed).into());
            let vk = sk.verifying_key();
            vk.encode().to_vec()
        }
        MlDsaLevel::MlDsa65 => {
            let sk = ml_dsa::SigningKey::<MlDsa65>::from_seed((&seed).into());
            let vk = sk.verifying_key();
            vk.encode().to_vec()
        }
        MlDsaLevel::MlDsa87 => {
            let sk = ml_dsa::SigningKey::<MlDsa87>::from_seed((&seed).into());
            let vk = sk.verifying_key();
            vk.encode().to_vec()
        }
    };

    MlDsaKeyPair {
        level,
        signing_key_seed: seed.to_vec(),
        verifying_key: vk_bytes,
    }
}

/// Sign a message using an ML-DSA signing key
///
/// # Arguments
/// * `level` - The ML-DSA security level
/// * `signing_key` - The signing key seed bytes (32 bytes)
/// * `message` - The message to sign
///
/// # Returns
/// The digital signature bytes
///
/// # Errors
/// Returns an error if the signing key is invalid.
pub fn ml_dsa_sign(level: MlDsaLevel, signing_key: &[u8], message: &[u8]) -> MlDsaResult<Vec<u8>> {
    let seed: &[u8; 32] = signing_key.try_into().map_err(|_| {
        MlDsaError::InvalidKeyData(format!(
            "Invalid signing key seed length: {} (expected 32)",
            signing_key.len()
        ))
    })?;

    match level {
        MlDsaLevel::MlDsa44 => {
            let sk = ml_dsa::SigningKey::<MlDsa44>::from_seed(seed.into());
            let sig = sk.sign(message);
            Ok(sig.encode().to_vec())
        }
        MlDsaLevel::MlDsa65 => {
            let sk = ml_dsa::SigningKey::<MlDsa65>::from_seed(seed.into());
            let sig = sk.sign(message);
            Ok(sig.encode().to_vec())
        }
        MlDsaLevel::MlDsa87 => {
            let sk = ml_dsa::SigningKey::<MlDsa87>::from_seed(seed.into());
            let sig = sk.sign(message);
            Ok(sig.encode().to_vec())
        }
    }
}

/// Verify a signature using an ML-DSA verifying key
///
/// # Arguments
/// * `level` - The ML-DSA security level
/// * `verifying_key` - The verifying (public) key bytes
/// * `message` - The original message
/// * `signature` - The signature to verify
///
/// # Returns
/// `true` if the signature is valid, `false` otherwise
///
/// # Errors
/// Returns an error if the verifying key or signature format is invalid.
pub fn ml_dsa_verify(
    level: MlDsaLevel,
    verifying_key: &[u8],
    message: &[u8],
    signature: &[u8],
) -> MlDsaResult<bool> {
    match level {
        MlDsaLevel::MlDsa44 => {
            let vk_encoded = verifying_key.try_into().map_err(|_| {
                MlDsaError::InvalidKeyData(format!(
                    "Invalid ML-DSA-44 verifying key length: {}",
                    verifying_key.len()
                ))
            })?;
            let vk = ml_dsa::VerifyingKey::<MlDsa44>::decode(vk_encoded);
            let sig_encoded = signature.try_into().map_err(|_| {
                MlDsaError::InvalidSignature(format!(
                    "Invalid ML-DSA-44 signature length: {}",
                    signature.len()
                ))
            })?;
            let sig = ml_dsa::Signature::<MlDsa44>::decode(sig_encoded).ok_or_else(|| {
                MlDsaError::InvalidSignature("Failed to decode ML-DSA-44 signature".to_string())
            })?;
            match vk.verify(message, &sig) {
                Ok(()) => Ok(true),
                Err(_) => Ok(false),
            }
        }
        MlDsaLevel::MlDsa65 => {
            let vk_encoded = verifying_key.try_into().map_err(|_| {
                MlDsaError::InvalidKeyData(format!(
                    "Invalid ML-DSA-65 verifying key length: {}",
                    verifying_key.len()
                ))
            })?;
            let vk = ml_dsa::VerifyingKey::<MlDsa65>::decode(vk_encoded);
            let sig_encoded = signature.try_into().map_err(|_| {
                MlDsaError::InvalidSignature(format!(
                    "Invalid ML-DSA-65 signature length: {}",
                    signature.len()
                ))
            })?;
            let sig = ml_dsa::Signature::<MlDsa65>::decode(sig_encoded).ok_or_else(|| {
                MlDsaError::InvalidSignature("Failed to decode ML-DSA-65 signature".to_string())
            })?;
            match vk.verify(message, &sig) {
                Ok(()) => Ok(true),
                Err(_) => Ok(false),
            }
        }
        MlDsaLevel::MlDsa87 => {
            let vk_encoded = verifying_key.try_into().map_err(|_| {
                MlDsaError::InvalidKeyData(format!(
                    "Invalid ML-DSA-87 verifying key length: {}",
                    verifying_key.len()
                ))
            })?;
            let vk = ml_dsa::VerifyingKey::<MlDsa87>::decode(vk_encoded);
            let sig_encoded = signature.try_into().map_err(|_| {
                MlDsaError::InvalidSignature(format!(
                    "Invalid ML-DSA-87 signature length: {}",
                    signature.len()
                ))
            })?;
            let sig = ml_dsa::Signature::<MlDsa87>::decode(sig_encoded).ok_or_else(|| {
                MlDsaError::InvalidSignature("Failed to decode ML-DSA-87 signature".to_string())
            })?;
            match vk.verify(message, &sig) {
                Ok(()) => Ok(true),
                Err(_) => Ok(false),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pqc_vectors as v;

    /// One ACVP signature-verification case: `(level, pk, message, signature)`.
    type SignatureCase = (MlDsaLevel, Vec<u8>, Vec<u8>, Vec<u8>);

    // ---------------------------------------------------------------------
    // FIPS 204 known-answer tests.
    //
    // Sources and pinned commits are recorded per constant in `pqc_vectors.rs`.
    // No expected value here was captured from this implementation.
    //
    // There is deliberately NO signature-generation KAT, and the reason is not
    // that `ml_dsa_sign` is randomized -- it is not. `ml_dsa_sign` calls the
    // `Signer` impl, which in `ml-dsa 0.1.0-rc.7` routes to
    // `raw_sign_deterministic`, so the same seed and message always give the same
    // signature. (The issue that asked for this work assumed hedged signing; that
    // is wrong for this crate version, and `test_ml_dsa_sign_is_deterministic`
    // below pins the actual behaviour so a future crate bump that makes signing
    // hedged fails here rather than silently.)
    //
    // The real obstacle is that ACVP's sigGen vectors supply an EXPANDED signing
    // key (`sk`, 2560 bytes at ML-DSA-44) and no seed, while `ml_dsa_sign` takes
    // only a 32-byte seed. `MlDsaKeyPair` stores the seed by design -- FIPS 204
    // §3.6.3 prefers seed-form private keys -- and `ml-dsa`'s expanded-key import
    // is `#[deprecated]` and documented as able to panic on malformed input. Adding
    // a public expanded-key entry point purely to satisfy a test would widen the API
    // toward the form the standard steers away from. ACVP publishes no
    // seed-to-signature vector: its keyGen and sigGen files share no key material
    // (verified: zero `sk`/`pk` intersection), so the chain seed -> sk -> signature
    // cannot be assembled from published data. Signature GENERATION is therefore
    // covered transitively -- our signatures verify under our verifier, and our
    // verifier is itself pinned to ACVP signatures below.
    // ---------------------------------------------------------------------

    /// ML-DSA.KeyGen (FIPS 204 Algorithm 1) plus `pkEncode` (Algorithm 22) at all
    /// three parameter sets: a fixed 32-byte seed must produce exactly the ACVP
    /// verifying key.
    ///
    /// `ml_dsa_generate_keypair` draws its own seed, so it cannot be held to a
    /// vector; this drives `SigningKey::from_seed` through the same path that
    /// function uses to derive `verifying_key`, which is the part the standard
    /// fixes.
    #[test]
    fn test_ml_dsa_keygen_kat_acvp() {
        let cases: [(MlDsaLevel, &str, &str, usize); 3] = [
            (
                MlDsaLevel::MlDsa44,
                v::MLDSA44_KEYGEN_SEED,
                v::MLDSA44_KEYGEN_PK,
                1312,
            ),
            (
                MlDsaLevel::MlDsa65,
                v::MLDSA65_KEYGEN_SEED,
                v::MLDSA65_KEYGEN_PK,
                1952,
            ),
            (
                MlDsaLevel::MlDsa87,
                v::MLDSA87_KEYGEN_SEED,
                v::MLDSA87_KEYGEN_PK,
                2592,
            ),
        ];

        for (level, seed, want_pk, pk_len) in cases {
            let got = verifying_key_from_seed(level, &v::unhex32(seed));
            let want = v::unhex(want_pk);

            // FIPS 204 Table 2 sizes.
            assert_eq!(want.len(), pk_len, "{level:?} vector pk length");
            assert_eq!(
                got, want,
                "{level:?} verifying key does not match the ACVP vector"
            );
        }
    }

    /// The same derivation held to a SECOND, independent authority: the IETF LAMPS
    /// dilithium-certificates example for ML-DSA-44, generated by CIRCL rather than
    /// by ACVP's reference implementation.
    ///
    /// Two unrelated implementations agreeing on `seed -> vk` rules out both a bug
    /// in ours and a transcription error in either vector -- the same argument
    /// `eap_aka_prime.rs` records for its RFC 9048 constants.
    #[test]
    fn test_ml_dsa_44_keygen_kat_lamps() {
        let got = verifying_key_from_seed(MlDsaLevel::MlDsa44, &v::unhex32(v::MLDSA44_LAMPS_SEED));
        assert_eq!(
            got,
            v::unhex(v::MLDSA44_LAMPS_PK),
            "verifying key does not match the IETF LAMPS example"
        );
    }

    /// Derive a verifying key from a fixed seed, mirroring what
    /// `ml_dsa_generate_keypair` does after drawing its seed from the RNG.
    fn verifying_key_from_seed(level: MlDsaLevel, seed: &[u8; 32]) -> Vec<u8> {
        match level {
            MlDsaLevel::MlDsa44 => ml_dsa::SigningKey::<MlDsa44>::from_seed(seed.into())
                .verifying_key()
                .encode()
                .to_vec(),
            MlDsaLevel::MlDsa65 => ml_dsa::SigningKey::<MlDsa65>::from_seed(seed.into())
                .verifying_key()
                .encode()
                .to_vec(),
            MlDsaLevel::MlDsa87 => ml_dsa::SigningKey::<MlDsa87>::from_seed(seed.into())
                .verifying_key()
                .encode()
                .to_vec(),
        }
    }

    /// ML-DSA.Verify (FIPS 204 Algorithm 3, pure, empty context): a verifying key,
    /// message and signature all published by ACVP must be ACCEPTED.
    ///
    /// This is the assertion that pins the `sigEncode` (Algorithm 26) signature
    /// layout and the µ computation including the domain separator. A round-trip
    /// test cannot reach it: if our encoder and decoder shared a mistake, a round
    /// trip would still pass while nothing we produced would interoperate.
    #[test]
    fn test_ml_dsa_verify_kat_acvp() {
        for (level, pk, msg, sig) in acvp_signature_cases() {
            let valid = ml_dsa_verify(level, &pk, &msg, &sig).expect("verify the ACVP vector");
            assert!(
                valid,
                "{level:?}: a valid ACVP signature was rejected -- our signature \
                 encoding or message hashing disagrees with FIPS 204"
            );
        }
    }

    /// A single flipped bit anywhere in an otherwise-valid ACVP signature must be
    /// rejected, at every covered parameter set and in every one of the signature's
    /// three encoded fields.
    ///
    /// `sigEncode` lays out c~ || z || h (FIPS 204 Algorithm 26), and the three
    /// offsets below land in each in turn, so a verifier that skipped checking any
    /// one field would fail here. The `wrong_message` and `wrong_key` tests further
    /// down perturb the INPUTS; this perturbs the signature itself, which is the
    /// forgery an attacker actually attempts.
    #[test]
    fn test_ml_dsa_bit_flipped_signature_rejected_kat_acvp() {
        for (level, pk, msg, sig) in acvp_signature_cases() {
            // Sanity: the unmodified signature must verify, or this test would
            // "pass" by rejecting something that was already invalid.
            assert!(
                ml_dsa_verify(level, &pk, &msg, &sig).expect("verify"),
                "{level:?}: baseline ACVP signature must be valid"
            );

            // c~ is first (32/48/64 bytes), then z, then the hint h at the tail.
            let offsets = [0usize, sig.len() / 2, sig.len() - 1];
            for offset in offsets {
                for bit in [0u8, 3, 7] {
                    let mut tampered = sig.clone();
                    tampered[offset] ^= 1 << bit;
                    // A flipped bit may make the signature undecodable (an out-of-range
                    // z coefficient, or a hint that no longer parses), which FIPS 204
                    // §5.3 also treats as invalid. Either outcome is a rejection; what
                    // must never happen is acceptance.
                    let accepted = ml_dsa_verify(level, &pk, &msg, &tampered).unwrap_or(false);
                    assert!(
                        !accepted,
                        "{level:?}: signature with bit {bit} of byte {offset} flipped \
                         was ACCEPTED"
                    );
                }
            }
        }
    }

    /// ACVP `(pk, message, signature)` triples for the external/pure interface with
    /// an empty context -- the interface `ml_dsa_verify` implements.
    ///
    /// ML-DSA-65 is absent on purpose: ACVP's ML-DSA-65 external/pure groups carry
    /// no empty-context case, and inventing one would mean re-deriving the
    /// signature ourselves, which is exactly the self-reference these tests exist
    /// to avoid. ML-DSA-65 key generation IS covered above.
    fn acvp_signature_cases() -> Vec<SignatureCase> {
        vec![
            (
                MlDsaLevel::MlDsa44,
                v::unhex(v::MLDSA44_SIGVER_PK),
                v::unhex(v::MLDSA44_SIGVER_MSG),
                v::unhex(v::MLDSA44_SIGVER_SIG),
            ),
            (
                MlDsaLevel::MlDsa87,
                v::unhex(v::MLDSA87_SIGVER_PK),
                v::unhex(v::MLDSA87_SIGVER_MSG),
                v::unhex(v::MLDSA87_SIGVER_SIG),
            ),
        ]
    }

    /// `ml_dsa_sign` is DETERMINISTIC in `ml-dsa 0.1.0-rc.7`: the `Signer` impl
    /// routes to `raw_sign_deterministic` (FIPS 204 Algorithm 2 with rnd = 0), not
    /// to the hedged variant.
    ///
    /// Pinned as a test because callers may reasonably assume otherwise, and
    /// because a future crate version that switches `Signer` to hedged signing
    /// would change our observable behaviour. Better that it fails here than in a
    /// caller that had come to depend on stable signature bytes.
    #[test]
    fn test_ml_dsa_sign_is_deterministic() {
        let seed = v::unhex32(v::MLDSA44_KEYGEN_SEED);
        let message = b"determinism is a property worth pinning";

        let first = ml_dsa_sign(MlDsaLevel::MlDsa44, &seed, message).expect("sign once");
        let second = ml_dsa_sign(MlDsaLevel::MlDsa44, &seed, message).expect("sign twice");

        assert_eq!(
            first, second,
            "signing became non-deterministic -- `ml-dsa`'s `Signer` impl no longer \
             uses the deterministic variant"
        );
    }

    #[test]
    fn test_ml_dsa_44_keygen() {
        let kp = ml_dsa_generate_keypair(MlDsaLevel::MlDsa44);
        assert_eq!(kp.level(), MlDsaLevel::MlDsa44);
        assert!(!kp.signing_key().is_empty());
        assert!(!kp.verifying_key().is_empty());
    }

    #[test]
    fn test_ml_dsa_65_keygen() {
        let kp = ml_dsa_generate_keypair(MlDsaLevel::MlDsa65);
        assert_eq!(kp.level(), MlDsaLevel::MlDsa65);
        assert!(!kp.signing_key().is_empty());
        assert!(!kp.verifying_key().is_empty());
    }

    #[test]
    fn test_ml_dsa_87_keygen() {
        let kp = ml_dsa_generate_keypair(MlDsaLevel::MlDsa87);
        assert_eq!(kp.level(), MlDsaLevel::MlDsa87);
        assert!(!kp.signing_key().is_empty());
        assert!(!kp.verifying_key().is_empty());
    }

    #[test]
    fn test_ml_dsa_44_sign_verify() {
        let kp = ml_dsa_generate_keypair(MlDsaLevel::MlDsa44);
        let message = b"Test message for ML-DSA-44";

        let sig = ml_dsa_sign(MlDsaLevel::MlDsa44, kp.signing_key(), message).expect("sign");
        let valid =
            ml_dsa_verify(MlDsaLevel::MlDsa44, kp.verifying_key(), message, &sig).expect("verify");

        assert!(valid);
    }

    #[test]
    fn test_ml_dsa_65_sign_verify() {
        let kp = ml_dsa_generate_keypair(MlDsaLevel::MlDsa65);
        let message = b"Test message for ML-DSA-65";

        let sig = ml_dsa_sign(MlDsaLevel::MlDsa65, kp.signing_key(), message).expect("sign");
        let valid =
            ml_dsa_verify(MlDsaLevel::MlDsa65, kp.verifying_key(), message, &sig).expect("verify");

        assert!(valid);
    }

    #[test]
    fn test_ml_dsa_87_sign_verify() {
        let kp = ml_dsa_generate_keypair(MlDsaLevel::MlDsa87);
        let message = b"Test message for ML-DSA-87";

        let sig = ml_dsa_sign(MlDsaLevel::MlDsa87, kp.signing_key(), message).expect("sign");
        let valid =
            ml_dsa_verify(MlDsaLevel::MlDsa87, kp.verifying_key(), message, &sig).expect("verify");

        assert!(valid);
    }

    #[test]
    fn test_ml_dsa_65_wrong_message_fails() {
        let kp = ml_dsa_generate_keypair(MlDsaLevel::MlDsa65);
        let message = b"Original message";
        let wrong_message = b"Tampered message";

        let sig = ml_dsa_sign(MlDsaLevel::MlDsa65, kp.signing_key(), message).expect("sign");
        let valid = ml_dsa_verify(MlDsaLevel::MlDsa65, kp.verifying_key(), wrong_message, &sig)
            .expect("verify");

        assert!(!valid);
    }

    #[test]
    fn test_ml_dsa_65_wrong_key_fails() {
        let kp1 = ml_dsa_generate_keypair(MlDsaLevel::MlDsa65);
        let kp2 = ml_dsa_generate_keypair(MlDsaLevel::MlDsa65);
        let message = b"Test message";

        let sig = ml_dsa_sign(MlDsaLevel::MlDsa65, kp1.signing_key(), message).expect("sign");
        let valid =
            ml_dsa_verify(MlDsaLevel::MlDsa65, kp2.verifying_key(), message, &sig).expect("verify");

        assert!(!valid);
    }

    #[test]
    fn test_ml_dsa_invalid_signing_key() {
        let result = ml_dsa_sign(MlDsaLevel::MlDsa65, &[0u8; 10], b"test");
        assert!(result.is_err());
    }

    #[test]
    fn test_ml_dsa_invalid_verifying_key() {
        let result = ml_dsa_verify(MlDsaLevel::MlDsa65, &[0u8; 10], b"test", &[0u8; 10]);
        assert!(result.is_err());
    }

    #[test]
    fn test_ml_dsa_empty_message() {
        let kp = ml_dsa_generate_keypair(MlDsaLevel::MlDsa65);
        let message = b"";

        let sig = ml_dsa_sign(MlDsaLevel::MlDsa65, kp.signing_key(), message).expect("sign");
        let valid =
            ml_dsa_verify(MlDsaLevel::MlDsa65, kp.verifying_key(), message, &sig).expect("verify");

        assert!(valid);
    }
}
