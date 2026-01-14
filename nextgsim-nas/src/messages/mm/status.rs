//! 5GMM Status Message (3GPP TS 24.501 Section 8.2.29)
//!
//! This module implements the 5GMM Status message which is used to report
//! error conditions in the 5G Mobility Management layer.

use bytes::{Buf, BufMut};
use thiserror::Error;

use crate::enums::MmMessageType;
use crate::header::PlainMmHeader;

use super::registration::{Ie5gMmCause, MmCause};

/// Error type for 5GMM Status message encoding/decoding
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum StatusError {
    /// Buffer too short for decoding
    #[error("Buffer too short: expected at least {expected} bytes, got {actual}")]
    BufferTooShort {
        /// Expected minimum bytes
        expected: usize,
        /// Actual bytes available
        actual: usize,
    },
    /// Invalid IE value
    #[error("Invalid IE value: {0}")]
    InvalidIeValue(String),
    /// Header decoding error
    #[error("Header error: {0}")]
    HeaderError(#[from] crate::header::HeaderError),
}

// ============================================================================
// 5GMM Status Message (3GPP TS 24.501 Section 8.2.29)
// ============================================================================

/// 5GMM Status message
///
/// This message is sent by the UE or the network to report error conditions
/// in the 5GMM sublayer.
///
/// 3GPP TS 24.501 Section 8.2.29
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FiveGMmStatus {
    /// 5GMM cause (mandatory, Type 3)
    pub mm_cause: Ie5gMmCause,
}

impl Default for FiveGMmStatus {
    fn default() -> Self {
        Self {
            mm_cause: Ie5gMmCause::new(MmCause::ProtocolErrorUnspecified),
        }
    }
}

impl FiveGMmStatus {
    /// Create a new 5GMM Status message with the specified cause
    pub fn new(cause: MmCause) -> Self {
        Self {
            mm_cause: Ie5gMmCause::new(cause),
        }
    }

    /// Create a 5GMM Status message from an Ie5gMmCause
    pub fn from_ie(mm_cause: Ie5gMmCause) -> Self {
        Self { mm_cause }
    }

    /// Decode from bytes (after header has been parsed)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, StatusError> {
        // 5GMM cause (mandatory, Type 3 - 1 byte)
        let mm_cause = Ie5gMmCause::decode(buf)
            .map_err(|e| StatusError::InvalidIeValue(e.to_string()))?;

        Ok(Self { mm_cause })
    }

    /// Encode to bytes (including header)
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        // Header
        let header = PlainMmHeader::new(MmMessageType::FiveGMmStatus);
        header.encode(buf);

        // 5GMM cause (mandatory)
        self.mm_cause.encode(buf);
    }

    /// Get the message type
    pub fn message_type() -> MmMessageType {
        MmMessageType::FiveGMmStatus
    }

    /// Get the cause value
    pub fn cause(&self) -> MmCause {
        self.mm_cause.value
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_5gmm_status_new() {
        let status = FiveGMmStatus::new(MmCause::IllegalUe);
        assert_eq!(status.cause(), MmCause::IllegalUe);
    }

    #[test]
    fn test_5gmm_status_default() {
        let status = FiveGMmStatus::default();
        assert_eq!(status.cause(), MmCause::ProtocolErrorUnspecified);
    }

    #[test]
    fn test_5gmm_status_encode_decode() {
        let status = FiveGMmStatus::new(MmCause::MessageNotCompatible);

        let mut buf = Vec::new();
        status.encode(&mut buf);

        // Expected: EPD (1) + Security Header (1) + Message Type (1) + Cause (1) = 4 bytes
        assert_eq!(buf.len(), 4);

        // Skip header (3 bytes) and decode the message body
        let decoded = FiveGMmStatus::decode(&mut &buf[3..]).unwrap();
        assert_eq!(decoded.cause(), MmCause::MessageNotCompatible);
    }

    #[test]
    fn test_5gmm_status_message_type() {
        assert_eq!(FiveGMmStatus::message_type(), MmMessageType::FiveGMmStatus);
    }

    #[test]
    fn test_5gmm_status_decode_empty_buffer() {
        let buf: &[u8] = &[];
        let result = FiveGMmStatus::decode(&mut &buf[..]);
        assert!(matches!(result, Err(StatusError::InvalidIeValue(_))));
    }
}
