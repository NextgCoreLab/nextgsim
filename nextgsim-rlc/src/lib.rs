//! Radio Link Control (RLC) layer — TS 38.322
//!
//! RLC sits between MAC (below) and PDCP (above) in the 3GPP NR protocol
//! stack. It provides three modes of operation:
//!
//! - **TM** (Transparent Mode): no RLC overhead; used for BCCH/PCCH/CCCH.
//! - **UM** (Unacknowledged Mode): segmentation and reassembly, no ARQ;
//!   used for DTCH and some DCCH bearers.
//! - **AM** (Acknowledged Mode): segmentation, reassembly, and ARQ with
//!   retransmission; used for DCCH and reliable DTCH bearers.
//!
//! # Usage
//!
//! ```rust
//! use nextgsim_rlc::{RlcEntity, RlcMode, SnSize};
//!
//! // Create a UM entity with 12-bit sequence numbers
//! let mut tx = RlcEntity::new(RlcMode::UnacknowledgedMode, SnSize::Sn12);
//! let mut rx = RlcEntity::new(RlcMode::UnacknowledgedMode, SnSize::Sn12);
//!
//! // PDCP submits an SDU that is larger than the MAC grant
//! tx.submit_sdu(vec![0u8; 100]);
//!
//! // MAC polls for a PDU fitting in 60 bytes
//! let pdu1 = tx.build_pdu(60).expect("first segment");
//! let pdu2 = tx.build_pdu(60).expect("second segment");
//!
//! // Receiver processes both PDUs
//! rx.receive_pdu(&pdu1);
//! rx.receive_pdu(&pdu2);
//!
//! // PDCP collects the reassembled SDU
//! let sdu = rx.poll_reassembled().expect("reassembled SDU");
//! assert_eq!(sdu.len(), 100);
//! ```

#![warn(missing_docs)]

pub mod entity;
pub mod error;
pub mod pdu;

pub use entity::{RlcEntity, RlcSegment};
pub use error::RlcError;
pub use pdu::{RlcAmPdu, RlcUmPdu, SegmentationInfo};

/// RLC operating mode (TS 38.322 §4.2)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RlcMode {
    /// TM — no RLC header; used for BCCH, PCCH, CCCH (SRB0)
    TransparentMode,
    /// UM — segmentation without retransmission; used for DRB/DTCH
    UnacknowledgedMode,
    /// AM — segmentation plus ARQ retransmission; used for SRB1/SRB2/DRB
    AcknowledgedMode,
}

/// Sequence number field size (TS 38.322 §6.2.3)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnSize {
    /// 6-bit SN (UM only — TS 38.322 §6.2.3.3)
    Sn6,
    /// 12-bit SN (UM or AM — TS 38.322 §6.2.3.3 / §6.2.3.5)
    Sn12,
    /// 18-bit SN (AM only — TS 38.322 §6.2.3.5)
    Sn18,
}

impl SnSize {
    /// Maximum sequence number for this SN size (modulus)
    pub fn modulus(self) -> u32 {
        match self {
            Self::Sn6 => 1 << 6,
            Self::Sn12 => 1 << 12,
            Self::Sn18 => 1 << 18,
        }
    }

    /// Header bytes consumed by the fixed part of a UM PDU header
    pub fn um_header_bytes(self) -> usize {
        match self {
            // 6-bit SN fits in 1 byte together with SI bits
            Self::Sn6 => 1,
            // 12-bit SN needs 2 bytes; SO (if present) adds 2 more
            Self::Sn12 => 2,
            Self::Sn18 => unreachable!("18-bit SN is AM only"),
        }
    }
}
