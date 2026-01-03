//! Error types for nextgsim

use thiserror::Error;

/// Error types for the nextgsim library.
#[derive(Debug, Error)]
pub enum Error {
    /// Configuration-related errors.
    #[error("Configuration error: {0}")]
    Config(String),

    /// Protocol-related errors.
    #[error("Protocol error: {0}")]
    Protocol(String),

    /// Network I/O errors.
    #[error("Network error: {0}")]
    Network(#[from] std::io::Error),

    /// ASN.1 encoding errors.
    #[error("ASN.1 encoding error: {0}")]
    Asn1Encode(String),

    /// ASN.1 decoding errors.
    #[error("ASN.1 decoding error: {0}")]
    Asn1Decode(String),

    /// Cryptographic operation errors.
    #[error("Crypto error: {0}")]
    Crypto(String),

    /// State machine errors.
    #[error("State machine error: {0}")]
    StateMachine(String),

    /// YAML parsing errors.
    #[error("YAML parse error: {0}")]
    YamlParse(#[from] serde_yaml::Error),
}
