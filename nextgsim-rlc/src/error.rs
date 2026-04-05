//! RLC error types

use thiserror::Error;

/// Errors that can occur during RLC operations
#[derive(Debug, Error)]
pub enum RlcError {
    /// The PDU byte slice is too short to contain a valid header
    #[error("RLC PDU too short: need at least {need} bytes, got {got}")]
    PduTooShort {
        /// Minimum bytes required
        need: usize,
        /// Bytes actually available
        got: usize,
    },

    /// Reserved SI field value encountered
    #[error("invalid segmentation info field value: {0}")]
    InvalidSi(u8),

    /// Sequence number in received PDU is outside the reassembly window
    #[error("sequence number {sn} is outside the reassembly window")]
    SnOutsideWindow {
        /// The out-of-window sequence number
        sn: u32,
    },

    /// Duplicate segment received (same SN + SO)
    #[error("duplicate segment: sn={sn}, so={so}")]
    DuplicateSegment {
        /// Sequence number of the duplicate
        sn: u32,
        /// Segment offset of the duplicate
        so: u16,
    },

    /// SDU submitted to a TM entity is empty
    #[error("empty SDU submitted")]
    EmptySdu,

    /// Attempted operation is not valid for this RLC mode
    #[error("operation not supported in {mode} mode")]
    WrongMode {
        /// Name of the current mode
        mode: &'static str,
    },
}
