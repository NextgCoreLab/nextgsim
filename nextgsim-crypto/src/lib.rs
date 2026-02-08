//! Cryptographic algorithms for nextgsim
//!
//! Implements 5G/6G security algorithms:
//! - Milenage (5G-AKA)
//! - SNOW3G (NEA1/NIA1)
//! - ZUC (NEA3/NIA3)
//! - ZUC-256 (256-bit security variant)
//! - AES-based (NEA2/NIA2)
//! - Key derivation functions
//! - ECIES for SUPI concealment (Profile A and Profile B)
//! - ML-KEM (CRYSTALS-Kyber) post-quantum key encapsulation
//! - ML-DSA (CRYSTALS-Dilithium) post-quantum digital signatures
//! - Hybrid key exchange (X25519 + ML-KEM-768)
//! - SNOW5G (next-gen stream cipher, placeholder)

pub mod aes;
pub mod ecies;
pub mod hybrid;
pub mod kdf;
pub mod milenage;
pub mod nea;
pub mod nia;
pub mod pqc_kem;
pub mod pqc_sign;
pub mod snow3g;
pub mod snow5g;
pub mod zuc;
pub mod zuc256;
