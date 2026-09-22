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

    /// A SRAP header named a bearer identity outside the range TS 38.331 allows
    /// (issue #190, `SL-RemoteUE-RB-Identity-r17`).
    ///
    /// Carries `is_drb` because the SRB and DRB number spaces differ — `INTEGER (0..3)`
    /// minus SRB3, against `DRB-Identity`'s `1..=32` — so the identity alone does not say
    /// which range was violated.
    #[error("SRAP bearer identity {id} is not a valid {}", if *is_drb { "DRB" } else { "SRB" })]
    SrapInvalidBearerId {
        /// The offending bearer identity.
        id: u8,
        /// Whether it was carried as a DRB.
        is_drb: bool,
    },

    /// A SRAP PDU named a local Remote UE ID this SRAP entity is not configured for
    /// (issue #190, TS 38.300 §16.12.2.1).
    #[error("no SRAP mapping is configured for local Remote UE ID {local_remote_ue_id}")]
    SrapUnknownRemoteUe {
        /// The local Remote UE ID that is not served here.
        local_remote_ue_id: u8,
    },
}
