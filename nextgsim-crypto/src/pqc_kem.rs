//! ML-KEM (CRYSTALS-Kyber) post-quantum key encapsulation mechanism
//!
//! Implements ML-KEM (Module-Lattice-Based Key Encapsulation Mechanism)
//! as standardized in FIPS 203, supporting three security levels:
//! - ML-KEM-512: NIST Level 1 security
//! - ML-KEM-768: NIST Level 3 security (recommended)
//! - ML-KEM-1024: NIST Level 5 security
//!
//! ML-KEM provides quantum-resistant key encapsulation for establishing
//! shared secrets between parties.

use ml_kem::kem::{Decapsulate, Encapsulate};
use ml_kem::{
    EncapsulateDeterministic, EncodedSizeUser, KemCore, MlKem1024, MlKem512, MlKem768, B32,
};
use rand::rngs::OsRng;
use thiserror::Error;

/// Size in bytes of each of the two ML-KEM key-generation seeds, `d` and `z`
/// (FIPS 203 §7.1, `ML-KEM.KeyGen`).
pub const ML_KEM_SEED_SIZE: usize = 32;

/// ML-KEM security level parameter sets
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlKemLevel {
    /// ML-KEM-512: NIST Level 1 security (smallest keys and ciphertexts)
    MlKem512,
    /// ML-KEM-768: NIST Level 3 security (recommended for most applications)
    MlKem768,
    /// ML-KEM-1024: NIST Level 5 security (highest security)
    MlKem1024,
}

/// ML-KEM error types
#[derive(Debug, Error)]
pub enum MlKemError {
    /// Encapsulation failed
    #[error("Encapsulation failed: {0}")]
    EncapsulationFailed(String),
    /// Decapsulation failed
    #[error("Decapsulation failed: {0}")]
    DecapsulationFailed(String),
    /// Invalid key data
    #[error("Invalid key data: {0}")]
    InvalidKeyData(String),
}

/// Result type for ML-KEM operations
pub type MlKemResult<T> = Result<T, MlKemError>;

/// ML-KEM key pair wrapping encapsulation and decapsulation keys
///
/// Stores the key pair as serialized bytes so it can work across
/// all three parameter sets (512, 768, 1024).
#[derive(Clone)]
pub struct MlKemKeyPair {
    /// The security level of this key pair
    level: MlKemLevel,
    /// Serialized encapsulation (public) key
    encapsulation_key: Vec<u8>,
    /// Serialized decapsulation (secret) key
    decapsulation_key: Vec<u8>,
}

impl MlKemKeyPair {
    /// Get the security level of this key pair
    pub fn level(&self) -> MlKemLevel {
        self.level
    }

    /// Get the encapsulation (public) key bytes
    pub fn encapsulation_key(&self) -> &[u8] {
        &self.encapsulation_key
    }

    /// Get the decapsulation (secret) key bytes
    pub fn decapsulation_key(&self) -> &[u8] {
        &self.decapsulation_key
    }
}

/// Concrete encapsulation/decapsulation using a macro to avoid generic trait bound issues
macro_rules! ml_kem_ops {
    ($kem_type:ty, $ek_bytes:expr, encapsulate) => {{
        let ek_encoded = $ek_bytes.try_into().map_err(|_| {
            MlKemError::InvalidKeyData(format!(
                "Invalid encapsulation key length: {}",
                $ek_bytes.len()
            ))
        })?;
        let ek = <$kem_type as KemCore>::EncapsulationKey::from_bytes(ek_encoded);
        let (ct, ss) = ek
            .encapsulate(&mut OsRng)
            .map_err(|e| MlKemError::EncapsulationFailed(format!("{e:?}")))?;
        let ct_slice: &[u8] = ct.as_ref();
        let ss_slice: &[u8] = ss.as_ref();
        Ok((ct_slice.to_vec(), ss_slice.to_vec()))
    }};
    ($kem_type:ty, $ek_bytes:expr, $m:expr, encapsulate_deterministic) => {{
        let ek_encoded = $ek_bytes.try_into().map_err(|_| {
            MlKemError::InvalidKeyData(format!(
                "Invalid encapsulation key length: {}",
                $ek_bytes.len()
            ))
        })?;
        let ek = <$kem_type as KemCore>::EncapsulationKey::from_bytes(ek_encoded);
        let (ct, ss) = ek
            .encapsulate_deterministic($m)
            .map_err(|e| MlKemError::EncapsulationFailed(format!("{e:?}")))?;
        let ct_slice: &[u8] = ct.as_ref();
        let ss_slice: &[u8] = ss.as_ref();
        Ok((ct_slice.to_vec(), ss_slice.to_vec()))
    }};
    ($kem_type:ty, $dk_bytes:expr, $ct_bytes:expr, decapsulate) => {{
        let dk_encoded = $dk_bytes.try_into().map_err(|_| {
            MlKemError::InvalidKeyData(format!(
                "Invalid decapsulation key length: {}",
                $dk_bytes.len()
            ))
        })?;
        let dk = <$kem_type as KemCore>::DecapsulationKey::from_bytes(dk_encoded);
        let ct_encoded = $ct_bytes.try_into().map_err(|_| {
            MlKemError::InvalidKeyData(format!("Invalid ciphertext length: {}", $ct_bytes.len()))
        })?;
        let ss = dk
            .decapsulate(ct_encoded)
            .map_err(|e| MlKemError::DecapsulationFailed(format!("{e:?}")))?;
        let ss_slice: &[u8] = ss.as_ref();
        Ok(ss_slice.to_vec())
    }};
}

/// Generate an ML-KEM key pair at the specified security level
///
/// # Arguments
/// * `level` - The ML-KEM security level to use
///
/// # Returns
/// A new ML-KEM key pair
pub fn ml_kem_generate_keypair(level: MlKemLevel) -> MlKemKeyPair {
    match level {
        MlKemLevel::MlKem512 => {
            let (dk, ek) = MlKem512::generate(&mut OsRng);
            MlKemKeyPair {
                level,
                encapsulation_key: ek.as_bytes().to_vec(),
                decapsulation_key: dk.as_bytes().to_vec(),
            }
        }
        MlKemLevel::MlKem768 => {
            let (dk, ek) = MlKem768::generate(&mut OsRng);
            MlKemKeyPair {
                level,
                encapsulation_key: ek.as_bytes().to_vec(),
                decapsulation_key: dk.as_bytes().to_vec(),
            }
        }
        MlKemLevel::MlKem1024 => {
            let (dk, ek) = MlKem1024::generate(&mut OsRng);
            MlKemKeyPair {
                level,
                encapsulation_key: ek.as_bytes().to_vec(),
                decapsulation_key: dk.as_bytes().to_vec(),
            }
        }
    }
}

/// Generate an ML-KEM key pair deterministically from the FIPS 203 seeds `(d, z)`
///
/// This is `ML-KEM.KeyGen_internal` (FIPS 203 Algorithm 16), the entry point
/// `ml_kem_generate_keypair` reaches with random seeds. It exists as a public
/// function for exactly one reason: it is the only way to hold key generation to a
/// known-answer vector. With the RNG in charge, the only property a test can check
/// is that the pair round-trips with itself, which would pass even if our encoding
/// of `ek`/`dk` disagreed with the standard.
///
/// Do not call this with seeds from anywhere but a test vector or a cryptographic
/// RNG: FIPS 203 §3.3 requires `d` and `z` to be freshly random, and reusing them
/// reproduces a key pair exactly.
///
/// # Arguments
/// * `level` - The ML-KEM security level to use
/// * `d` - The 32-byte `d` seed
/// * `z` - The 32-byte `z` implicit-rejection seed
///
/// # Returns
/// The key pair those seeds determine
pub fn ml_kem_generate_keypair_deterministic(
    level: MlKemLevel,
    d: &[u8; ML_KEM_SEED_SIZE],
    z: &[u8; ML_KEM_SEED_SIZE],
) -> MlKemKeyPair {
    let d: &B32 = d.into();
    let z: &B32 = z.into();
    match level {
        MlKemLevel::MlKem512 => {
            let (dk, ek) = MlKem512::generate_deterministic(d, z);
            MlKemKeyPair {
                level,
                encapsulation_key: ek.as_bytes().to_vec(),
                decapsulation_key: dk.as_bytes().to_vec(),
            }
        }
        MlKemLevel::MlKem768 => {
            let (dk, ek) = MlKem768::generate_deterministic(d, z);
            MlKemKeyPair {
                level,
                encapsulation_key: ek.as_bytes().to_vec(),
                decapsulation_key: dk.as_bytes().to_vec(),
            }
        }
        MlKemLevel::MlKem1024 => {
            let (dk, ek) = MlKem1024::generate_deterministic(d, z);
            MlKemKeyPair {
                level,
                encapsulation_key: ek.as_bytes().to_vec(),
                decapsulation_key: dk.as_bytes().to_vec(),
            }
        }
    }
}

/// Encapsulate deterministically from a fixed message `m` (FIPS 203 Algorithm 17)
///
/// `ML-KEM.Encaps_internal`. As with
/// [`ml_kem_generate_keypair_deterministic`], this is public so that
/// encapsulation can be held to a vector: it is the only direction that produces a
/// CIPHERTEXT, so without it no test pins the FIPS 203 §7.2 ciphertext encoding at
/// all -- a decapsulation KAT consumes a ciphertext it was handed.
///
/// Do not call this outside a known-answer test. FIPS 203 requires `m` to be
/// freshly random; a fixed `m` makes the ciphertext and shared secret predictable
/// and destroys the KEM's security.
///
/// # Arguments
/// * `level` - The ML-KEM security level
/// * `encapsulation_key` - The recipient's encapsulation key bytes
/// * `m` - The 32-byte message randomness to use instead of drawing from an RNG
///
/// # Returns
/// A tuple of (`ciphertext_bytes`, `shared_secret_bytes`)
///
/// # Errors
/// Returns an error if the encapsulation key is invalid.
pub fn ml_kem_encapsulate_deterministic(
    level: MlKemLevel,
    encapsulation_key: &[u8],
    m: &[u8; ML_KEM_SEED_SIZE],
) -> MlKemResult<(Vec<u8>, Vec<u8>)> {
    let m: &B32 = m.into();
    match level {
        MlKemLevel::MlKem512 => {
            ml_kem_ops!(MlKem512, encapsulation_key, m, encapsulate_deterministic)
        }
        MlKemLevel::MlKem768 => {
            ml_kem_ops!(MlKem768, encapsulation_key, m, encapsulate_deterministic)
        }
        MlKemLevel::MlKem1024 => {
            ml_kem_ops!(MlKem1024, encapsulation_key, m, encapsulate_deterministic)
        }
    }
}

/// Encapsulate a shared secret using an ML-KEM public key
///
/// Generates a ciphertext and shared secret from the given encapsulation key.
///
/// # Arguments
/// * `level` - The ML-KEM security level
/// * `encapsulation_key` - The recipient's encapsulation (public) key bytes
///
/// # Returns
/// A tuple of (`ciphertext_bytes`, `shared_secret_bytes`)
///
/// # Errors
/// Returns an error if the encapsulation key is invalid.
pub fn ml_kem_encapsulate(
    level: MlKemLevel,
    encapsulation_key: &[u8],
) -> MlKemResult<(Vec<u8>, Vec<u8>)> {
    match level {
        MlKemLevel::MlKem512 => {
            ml_kem_ops!(MlKem512, encapsulation_key, encapsulate)
        }
        MlKemLevel::MlKem768 => {
            ml_kem_ops!(MlKem768, encapsulation_key, encapsulate)
        }
        MlKemLevel::MlKem1024 => {
            ml_kem_ops!(MlKem1024, encapsulation_key, encapsulate)
        }
    }
}

/// Decapsulate a shared secret using an ML-KEM secret key
///
/// # Arguments
/// * `level` - The ML-KEM security level
/// * `decapsulation_key` - The recipient's decapsulation (secret) key bytes
/// * `ciphertext` - The ciphertext from encapsulation
///
/// # Returns
/// The shared secret bytes (32 bytes)
///
/// # Errors
/// Returns an error if the decapsulation key or ciphertext is invalid.
pub fn ml_kem_decapsulate(
    level: MlKemLevel,
    decapsulation_key: &[u8],
    ciphertext: &[u8],
) -> MlKemResult<Vec<u8>> {
    match level {
        MlKemLevel::MlKem512 => {
            ml_kem_ops!(MlKem512, decapsulation_key, ciphertext, decapsulate)
        }
        MlKemLevel::MlKem768 => {
            ml_kem_ops!(MlKem768, decapsulation_key, ciphertext, decapsulate)
        }
        MlKemLevel::MlKem1024 => {
            ml_kem_ops!(MlKem1024, decapsulation_key, ciphertext, decapsulate)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pqc_vectors as v;

    // ---------------------------------------------------------------------
    // FIPS 203 known-answer tests.
    //
    // Every expected value below comes from the NIST ACVP internal-projection
    // files at the pinned commit recorded in `pqc_vectors.rs` -- never from this
    // implementation's own output. The round-trip tests further down prove the
    // three operations agree with EACH OTHER; only these prove they agree with
    // the standard.
    // ---------------------------------------------------------------------

    /// ML-KEM.KeyGen_internal (FIPS 203 Algorithm 16) at all three parameter sets:
    /// a fixed `(d, z)` must produce exactly the ACVP encapsulation and
    /// decapsulation keys, which pins the §7.1 key encodings byte for byte.
    #[test]
    fn test_ml_kem_keygen_kat_acvp() {
        let cases: [(MlKemLevel, &str, &str, &str, &str, usize, usize); 3] = [
            (
                MlKemLevel::MlKem512,
                v::MLKEM512_KEYGEN_D,
                v::MLKEM512_KEYGEN_Z,
                v::MLKEM512_KEYGEN_EK,
                v::MLKEM512_KEYGEN_DK,
                800,
                1632,
            ),
            (
                MlKemLevel::MlKem768,
                v::MLKEM768_KEYGEN_D,
                v::MLKEM768_KEYGEN_Z,
                v::MLKEM768_KEYGEN_EK,
                v::MLKEM768_KEYGEN_DK,
                1184,
                2400,
            ),
            (
                MlKemLevel::MlKem1024,
                v::MLKEM1024_KEYGEN_D,
                v::MLKEM1024_KEYGEN_Z,
                v::MLKEM1024_KEYGEN_EK,
                v::MLKEM1024_KEYGEN_DK,
                1568,
                3168,
            ),
        ];

        for (level, d, z, want_ek, want_dk, ek_len, dk_len) in cases {
            let kp = ml_kem_generate_keypair_deterministic(level, &v::unhex32(d), &v::unhex32(z));
            let want_ek = v::unhex(want_ek);
            let want_dk = v::unhex(want_dk);

            // FIPS 203 Table 3 sizes, asserted so a length mismatch reports as a
            // size error rather than as an opaque byte diff.
            assert_eq!(want_ek.len(), ek_len, "{level:?} vector ek length");
            assert_eq!(want_dk.len(), dk_len, "{level:?} vector dk length");

            assert_eq!(
                kp.encapsulation_key(),
                &want_ek[..],
                "{level:?} encapsulation key does not match the ACVP vector"
            );
            assert_eq!(
                kp.decapsulation_key(),
                &want_dk[..],
                "{level:?} decapsulation key does not match the ACVP vector"
            );
        }
    }

    /// ML-KEM.Encaps_internal (FIPS 203 Algorithm 17) for ML-KEM-768: a fixed `ek`
    /// and message `m` must produce exactly the ACVP ciphertext and shared secret.
    ///
    /// This is the only test that pins the §7.2 CIPHERTEXT encoding, since it is
    /// the only direction that produces one.
    #[test]
    fn test_ml_kem_768_encapsulate_kat_acvp() {
        let (ct, ss) = ml_kem_encapsulate_deterministic(
            MlKemLevel::MlKem768,
            &v::unhex(v::MLKEM768_ENCAP_EK),
            &v::unhex32(v::MLKEM768_ENCAP_M),
        )
        .expect("encapsulate deterministically");

        let want_ct = v::unhex(v::MLKEM768_ENCAP_C);
        assert_eq!(want_ct.len(), 1088, "vector ciphertext length");
        assert_eq!(ct, want_ct, "ciphertext does not match the ACVP vector");
        assert_eq!(
            ss,
            v::unhex(v::MLKEM768_ENCAP_K),
            "shared secret does not match the ACVP vector"
        );
    }

    /// ML-KEM.Decaps (FIPS 203 Algorithm 18) at all three parameter sets: a fixed
    /// `dk` and ciphertext must yield exactly the ACVP 32-byte shared secret.
    #[test]
    fn test_ml_kem_decapsulate_kat_acvp() {
        let cases: [(MlKemLevel, &str, &str, &str); 3] = [
            (
                MlKemLevel::MlKem512,
                v::MLKEM512_DECAP_DK,
                v::MLKEM512_DECAP_C,
                v::MLKEM512_DECAP_K,
            ),
            (
                MlKemLevel::MlKem768,
                v::MLKEM768_DECAP_DK,
                v::MLKEM768_DECAP_C,
                v::MLKEM768_DECAP_K,
            ),
            (
                MlKemLevel::MlKem1024,
                v::MLKEM1024_DECAP_DK,
                v::MLKEM1024_DECAP_C,
                v::MLKEM1024_DECAP_K,
            ),
        ];

        for (level, dk, ct, want_k) in cases {
            let ss = ml_kem_decapsulate(level, &v::unhex(dk), &v::unhex(ct))
                .expect("decapsulate the ACVP ciphertext");
            assert_eq!(
                ss,
                v::unhex(want_k),
                "{level:?} shared secret does not match the ACVP vector"
            );
        }
    }

    /// FIPS 203 §6.3 implicit rejection: decapsulating a CORRUPTED ciphertext must
    /// not error and must not return the honest key -- it must return the specific
    /// key derived from `z`, which ACVP publishes.
    ///
    /// This is the behaviour an ML-KEM implementation is most likely to get wrong
    /// in a way a round-trip test cannot see, because a round trip never presents a
    /// corrupted ciphertext. The `test_ml_kem_wrong_key_*` tests below only check
    /// that the result DIFFERS; this checks it is the right different answer.
    #[test]
    fn test_ml_kem_768_implicit_rejection_kat_acvp() {
        let dk = v::unhex(v::MLKEM768_DECAP_DK);

        let rejected =
            ml_kem_decapsulate(MlKemLevel::MlKem768, &dk, &v::unhex(v::MLKEM768_REJECT_C))
                .expect("implicit rejection returns a key rather than an error");

        assert_eq!(
            rejected,
            v::unhex(v::MLKEM768_REJECT_K),
            "implicit-rejection key does not match the ACVP vector"
        );
        // And it is genuinely the rejection branch, not the honest one: the same dk
        // decapsulating the UNMODIFIED ciphertext gives a different answer.
        assert_ne!(
            rejected,
            v::unhex(v::MLKEM768_DECAP_K),
            "rejection key must differ from the honest shared secret"
        );
    }

    #[test]
    fn test_ml_kem_512_keygen() {
        let kp = ml_kem_generate_keypair(MlKemLevel::MlKem512);
        assert_eq!(kp.level(), MlKemLevel::MlKem512);
        assert!(!kp.encapsulation_key().is_empty());
        assert!(!kp.decapsulation_key().is_empty());
    }

    #[test]
    fn test_ml_kem_768_keygen() {
        let kp = ml_kem_generate_keypair(MlKemLevel::MlKem768);
        assert_eq!(kp.level(), MlKemLevel::MlKem768);
        assert!(!kp.encapsulation_key().is_empty());
        assert!(!kp.decapsulation_key().is_empty());
    }

    #[test]
    fn test_ml_kem_1024_keygen() {
        let kp = ml_kem_generate_keypair(MlKemLevel::MlKem1024);
        assert_eq!(kp.level(), MlKemLevel::MlKem1024);
        assert!(!kp.encapsulation_key().is_empty());
        assert!(!kp.decapsulation_key().is_empty());
    }

    #[test]
    fn test_ml_kem_768_roundtrip() {
        let kp = ml_kem_generate_keypair(MlKemLevel::MlKem768);
        let (ct, ss_enc) =
            ml_kem_encapsulate(MlKemLevel::MlKem768, kp.encapsulation_key()).expect("encapsulate");
        let ss_dec = ml_kem_decapsulate(MlKemLevel::MlKem768, kp.decapsulation_key(), &ct)
            .expect("decapsulate");

        assert_eq!(ss_enc, ss_dec);
        assert_eq!(ss_enc.len(), 32);
    }

    #[test]
    fn test_ml_kem_512_roundtrip() {
        let kp = ml_kem_generate_keypair(MlKemLevel::MlKem512);
        let (ct, ss_enc) =
            ml_kem_encapsulate(MlKemLevel::MlKem512, kp.encapsulation_key()).expect("encapsulate");
        let ss_dec = ml_kem_decapsulate(MlKemLevel::MlKem512, kp.decapsulation_key(), &ct)
            .expect("decapsulate");

        assert_eq!(ss_enc, ss_dec);
        assert_eq!(ss_enc.len(), 32);
    }

    #[test]
    fn test_ml_kem_1024_roundtrip() {
        let kp = ml_kem_generate_keypair(MlKemLevel::MlKem1024);
        let (ct, ss_enc) =
            ml_kem_encapsulate(MlKemLevel::MlKem1024, kp.encapsulation_key()).expect("encapsulate");
        let ss_dec = ml_kem_decapsulate(MlKemLevel::MlKem1024, kp.decapsulation_key(), &ct)
            .expect("decapsulate");

        assert_eq!(ss_enc, ss_dec);
        assert_eq!(ss_enc.len(), 32);
    }

    #[test]
    fn test_ml_kem_different_keypairs_different_secrets() {
        let kp1 = ml_kem_generate_keypair(MlKemLevel::MlKem768);
        let kp2 = ml_kem_generate_keypair(MlKemLevel::MlKem768);

        let (_, ss1) =
            ml_kem_encapsulate(MlKemLevel::MlKem768, kp1.encapsulation_key()).expect("encapsulate");
        let (_, ss2) =
            ml_kem_encapsulate(MlKemLevel::MlKem768, kp2.encapsulation_key()).expect("encapsulate");

        assert_ne!(ss1, ss2);
    }

    #[test]
    fn test_ml_kem_invalid_encapsulation_key() {
        let result = ml_kem_encapsulate(MlKemLevel::MlKem768, &[0u8; 10]);
        assert!(result.is_err());
    }

    #[test]
    fn test_ml_kem_invalid_decapsulation_key() {
        let kp = ml_kem_generate_keypair(MlKemLevel::MlKem768);
        let (ct, _) =
            ml_kem_encapsulate(MlKemLevel::MlKem768, kp.encapsulation_key()).expect("encapsulate");
        let result = ml_kem_decapsulate(MlKemLevel::MlKem768, &[0u8; 10], &ct);
        assert!(result.is_err());
    }
}
