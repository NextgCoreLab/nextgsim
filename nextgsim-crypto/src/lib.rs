//! Cryptographic algorithms for nextgsim
//!
//! Implements 5G/6G security algorithms:
//! - Milenage (5G-AKA, TS 35.206)
//! - TUAK (alternative 5G-AKA set on Keccak, TS 35.231)
//! - SNOW3G (NEA1/NIA1)
//! - ZUC (NEA3/NIA3)
//! - ZUC-256 (256-bit security variant)
//! - AES-based (NEA2/NIA2)
//! - Key derivation functions (5G key hierarchy, TS 33.501 Annex A)
//! - EAP-AKA' key derivation (RFC 9048 / TS 33.501 §6.1.3.1)
//! - ECIES for SUPI concealment (Profile A and Profile B)
//! - SNOW5G (next-gen stream cipher, placeholder)
//!
//! Behind the off-by-default `pqc` feature:
//! - ML-KEM (CRYSTALS-Kyber) post-quantum key encapsulation (FIPS 203)
//! - ML-DSA (CRYSTALS-Dilithium) post-quantum digital signatures (FIPS 204)
//! - Hybrid key exchange (X25519 + ML-KEM-768)
//!
//! # Why the post-quantum surface is gated off
//!
//! 3GPP has frozen no normative post-quantum specification: SA3's migration work
//! is study-phase (TR 33.871), so no 5G or 6G procedure in this simulator has a
//! standardised place to call ML-KEM or ML-DSA from, and none does. What the
//! modules offer today is FIPS 203/204 primitives validated against NIST ACVP
//! known-answer vectors -- a correct foundation, not an integrated feature. Gating
//! them keeps two pre-1.0 release-candidate crates out of the default build of
//! every crate that depends on `nextgsim-crypto` (NAS, PDCP, RRC, gNB, UE) while
//! the primitives wait for a procedure to serve.

pub mod aes;
pub mod eap_aka_prime;
pub mod ecies;
#[cfg(feature = "pqc")]
pub mod hybrid;
pub mod kdf;
pub mod milenage;
pub mod nea;
pub mod nia;
#[cfg(feature = "pqc")]
pub mod pqc_kem;
#[cfg(feature = "pqc")]
pub mod pqc_sign;
#[cfg(all(test, feature = "pqc"))]
mod pqc_vectors;
pub mod snow3g;
pub mod snow5g;
pub mod tuak;
pub mod zuc;
pub mod zuc256;
