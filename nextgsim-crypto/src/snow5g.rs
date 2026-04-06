//! SNOW5G next-generation stream cipher (not yet implemented)
//!
//! SNOW5G is a next-generation stream cipher being developed for 6G security
//! algorithms. It is the successor to SNOW3G (used in 4G/5G) with enhanced
//! security properties including:
//! - 256-bit key size
//! - 128-bit IV
//! - Higher throughput for future network speeds
//!
//! **Status:** SNOW5G is still being standardized. All public functions in this
//! module **panic** at runtime so that any accidental call-site is caught
//! immediately rather than silently producing wrong ciphertext.
//!
//! Use NEA1 (SNOW3G), NEA2 (AES-128-CTR), or NEA3 (ZUC) for production
//! ciphering/integrity operations.
//!
//! Reference: 3GPP future specifications (pending)

/// SNOW5G key size in bytes (256 bits)
pub const KEY_SIZE: usize = 32;

/// SNOW5G IV size in bytes (128 bits)
pub const IV_SIZE: usize = 16;

/// SNOW5G encryption
///
/// **NOT IMPLEMENTED.** SNOW5G is still being standardized; calling this
/// function will panic. Use NEA1 (SNOW3G), NEA2 (AES), or NEA3 (ZUC) instead.
///
/// # Panics
/// Always panics — SNOW5G is not yet implemented.
#[deprecated(note = "SNOW5G is not yet implemented — use NEA1/NEA2/NEA3 instead")]
pub fn snow5g_encrypt(_key: &[u8; KEY_SIZE], _iv: &[u8; IV_SIZE], _data: &[u8]) -> Vec<u8> {
    panic!("SNOW5G not yet implemented — use NEA1/NEA2/NEA3 instead");
}

/// SNOW5G decryption
///
/// **NOT IMPLEMENTED.** SNOW5G is still being standardized; calling this
/// function will panic. Use NEA1 (SNOW3G), NEA2 (AES), or NEA3 (ZUC) instead.
///
/// # Panics
/// Always panics — SNOW5G is not yet implemented.
#[deprecated(note = "SNOW5G is not yet implemented — use NEA1/NEA2/NEA3 instead")]
pub fn snow5g_decrypt(_key: &[u8; KEY_SIZE], _iv: &[u8; IV_SIZE], _data: &[u8]) -> Vec<u8> {
    panic!("SNOW5G not yet implemented — use NEA1/NEA2/NEA3 instead");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that calling snow5g_encrypt panics loudly rather than silently
    /// producing wrong output with the old SHA-256 placeholder.
    #[test]
    #[should_panic(expected = "SNOW5G not yet implemented")]
    #[allow(deprecated)]
    fn test_snow5g_encrypt_panics() {
        let key = [0x42u8; KEY_SIZE];
        let iv = [0x13u8; IV_SIZE];
        let _ = snow5g_encrypt(&key, &iv, b"should panic");
    }

    /// Verify that calling snow5g_decrypt panics loudly.
    #[test]
    #[should_panic(expected = "SNOW5G not yet implemented")]
    #[allow(deprecated)]
    fn test_snow5g_decrypt_panics() {
        let key = [0x42u8; KEY_SIZE];
        let iv = [0x13u8; IV_SIZE];
        let _ = snow5g_decrypt(&key, &iv, b"should panic");
    }
}
