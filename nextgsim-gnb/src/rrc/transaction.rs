//! Per-UE RRC transaction identifier allocation (Wave-6 C4-final).
//!
//! TS 38.331 §6.3.2: `RRC-TransactionIdentifier ::= INTEGER (0..3)`. The tid is
//! carried in the downlink procedure message (RRCSetup, SecurityModeCommand,
//! RRCReconfiguration, UECapabilityEnquiry) and **echoed** by the UE in the
//! matching uplink Complete/Information message so both peers can pair a
//! response to its request (TS 38.331 §5.3.4 / §5.3.5 / §5.6.1).
//!
//! ## Why this replaces the gNB-global counter
//!
//! The pre-C4-final gNB kept a **single** `next_tid` counter shared across
//! every UE (`RrcConnectionManager::next_tid`, cycling 0..3). Two defects
//! followed:
//!
//! 1. one UE's procedures shifted another UE's tids (nothing is scoped to a
//!    UE), so a reviewer could not reason about the tid a given UE will see;
//! 2. for RRCSetup specifically the shared cycler emitted tid 1/3 on the 2nd/4th
//!    connection, whose UPER leading byte (0x28/0x38, low nibble 0x8) had **no
//!    arm** in the UE's legacy DL-CCCH nibble dispatcher — every 2nd/4th
//!    RRCSetup was silently dropped (the C4-interim `{0,2}` pin was a stop-gap
//!    for exactly this).
//!
//! C4-final moves allocation to a per-UE-context [`RrcTransactionAllocator`]:
//! each UE owns its own 0..3 cycle and its own per-procedure outstanding-tid
//! record, independent of every other UE. Because each UE context is fresh, its
//! first allocation is 0 — so RRCSetup is deterministically nibble-safe without
//! the parity hack, and the latent multi-connection drop is gone.
//!
//! ## Wire-safety gate ([`C5_TYPED_DCCH_DISPATCH`])
//!
//! Full 0..3 cycling on the **wire** for the gNB→UE messages is only safe once
//! C5 replaces the UE's legacy nibble dispatchers (`bytes[0] & 0x0F` on both
//! DL-CCCH and DL-DCCH) with a typed ASN.1 `DL-DCCH-Message` / `DL-CCCH-Message`
//! decode. Until then a non-zero tid shuffles the leading-byte nibble the UE
//! routes on and mis-delivers or drops the message (TS 38.331 §6.2). This
//! module therefore emits the allocated value on the wire **only** when
//! [`C5_TYPED_DCCH_DISPATCH`] is `true`; while it is `false` every gNB→UE tid is
//! pinned to 0 (still recorded as outstanding, so the echoed-tid verification
//! runs). Flipping the single const below to `true` in the C5 commit turns on
//! full per-UE cycling at every sender site with no other change — see
//! `.context/remediation/wave6/WS-C-gnb-srb1.md` item C4.

/// Number of RRC procedures tracked independently per UE.
const PROC_COUNT: usize = 4;

/// Wave-6 C4-final wire-safety gate.
///
/// `false` until C5 (typed ASN.1 DL-DCCH/DL-CCCH dispatch at the UE): gNB→UE
/// transaction ids are pinned to 0 on the wire because the UE's legacy nibble
/// dispatcher only routes correctly on tid 0. Flip to `true` in the C5 commit
/// to enable full per-UE 0..3 cycling at every sender site.
///
/// See the module docs and `WS-C-gnb-srb1.md` (C4 / C5 sequencing:
/// "Do not reorder C4-final before C5 — non-zero tids break both nibble
/// dispatchers by construction").
pub const C5_TYPED_DCCH_DISPATCH: bool = false;

/// The RRC procedures that allocate a downlink transaction id and expect the UE
/// to echo it in the matching uplink message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RrcProcedure {
    /// RRCSetup ↔ RRCSetupComplete (TS 38.331 §5.3.3, DL-CCCH).
    Setup,
    /// SecurityModeCommand ↔ SecurityModeComplete (TS 38.331 §5.3.4).
    SecurityMode,
    /// RRCReconfiguration ↔ RRCReconfigurationComplete (TS 38.331 §5.3.5).
    Reconfiguration,
    /// UECapabilityEnquiry ↔ UECapabilityInformation (TS 38.331 §5.6.1).
    UeCapability,
}

impl RrcProcedure {
    #[inline]
    fn index(self) -> usize {
        match self {
            RrcProcedure::Setup => 0,
            RrcProcedure::SecurityMode => 1,
            RrcProcedure::Reconfiguration => 2,
            RrcProcedure::UeCapability => 3,
        }
    }
}

/// Result of checking a UE-echoed transaction id against the outstanding
/// transaction the gNB allocated for a procedure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TidVerification {
    /// The echoed tid matches the outstanding transaction; it has been cleared.
    Match,
    /// The echoed tid does not match the outstanding transaction — the caller
    /// MUST discard the message (fail-closed, TS 38.331 procedure matching).
    /// The outstanding transaction is left pending (not cleared).
    Mismatch {
        /// The tid the gNB is actually waiting for.
        expected: u8,
    },
    /// No transaction is outstanding for this procedure (e.g. the gNB never
    /// sent the request, or the context was auto-created on a raw-NAS uplink).
    /// The caller proceeds tolerantly — there is nothing to reject against.
    NoOutstanding,
}

/// Per-UE RRC transaction-identifier allocator (TS 38.331 §6.3.2).
///
/// Lives on the gNB UE context (RRC-task `RrcUeContext`, NGAP-task
/// `NgapUeContext`) — never as a gNB-global counter. Cycles 0..3 for this UE
/// and records one outstanding tid per procedure for echoed-tid verification.
#[derive(Debug, Clone)]
pub struct RrcTransactionAllocator {
    /// Next tid to hand out for this UE (cycles 0..3).
    next: u8,
    /// Outstanding tid per procedure, awaiting the UE's echoed reply.
    outstanding: [Option<u8>; PROC_COUNT],
}

impl Default for RrcTransactionAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl RrcTransactionAllocator {
    /// Creates a fresh per-UE allocator (first allocation will be tid 0).
    pub const fn new() -> Self {
        Self {
            next: 0,
            outstanding: [None; PROC_COUNT],
        }
    }

    /// Allocates the wire transaction id for `proc` and records it as
    /// outstanding.
    ///
    /// Honours the [`C5_TYPED_DCCH_DISPATCH`] wire-safety gate: full per-UE
    /// 0..3 cycling when C5 has landed, otherwise a pinned 0 (still recorded,
    /// so [`Self::verify`] runs against it). This is the method the four gNB
    /// sender sites call.
    pub fn allocate(&mut self, proc: RrcProcedure) -> u8 {
        if C5_TYPED_DCCH_DISPATCH {
            self.allocate_cycling(proc)
        } else {
            self.pin_zero(proc)
        }
    }

    /// Allocates the next per-UE tid cycling 0..3, unconditionally, and records
    /// it as outstanding. This is the C5-enabled behavior; [`Self::allocate`]
    /// gates it behind [`C5_TYPED_DCCH_DISPATCH`].
    pub fn allocate_cycling(&mut self, proc: RrcProcedure) -> u8 {
        let tid = self.next;
        self.next = (self.next + 1) % 4;
        self.outstanding[proc.index()] = Some(tid);
        tid
    }

    /// Records a pinned tid 0 as the outstanding transaction for `proc` and
    /// returns 0, without advancing the cycle. Used while
    /// [`C5_TYPED_DCCH_DISPATCH`] is `false` so the gNB→UE wire byte stays
    /// nibble-safe for the UE's legacy dispatcher.
    pub fn pin_zero(&mut self, proc: RrcProcedure) -> u8 {
        self.outstanding[proc.index()] = Some(0);
        0
    }

    /// Verifies a UE-echoed transaction id against the outstanding transaction
    /// for `proc`. Fail-closed: a mismatch leaves the transaction pending and
    /// the caller MUST discard the message. Clears the transaction on a match.
    pub fn verify(&mut self, proc: RrcProcedure, echoed: u8) -> TidVerification {
        match self.outstanding[proc.index()] {
            Some(expected) if expected == echoed => {
                self.outstanding[proc.index()] = None;
                TidVerification::Match
            }
            Some(expected) => TidVerification::Mismatch { expected },
            None => TidVerification::NoOutstanding,
        }
    }

    /// Returns the outstanding tid for `proc`, if any.
    pub fn outstanding(&self, proc: RrcProcedure) -> Option<u8> {
        self.outstanding[proc.index()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cycling_allocation_wraps_0_1_2_3_0() {
        let mut a = RrcTransactionAllocator::new();
        // allocate_cycling is the C5-enabled behavior; assert the 0..3 wrap
        // regardless of the compile-time C5 gate.
        assert_eq!(a.allocate_cycling(RrcProcedure::Setup), 0);
        assert_eq!(a.allocate_cycling(RrcProcedure::SecurityMode), 1);
        assert_eq!(a.allocate_cycling(RrcProcedure::Reconfiguration), 2);
        assert_eq!(a.allocate_cycling(RrcProcedure::UeCapability), 3);
        assert_eq!(a.allocate_cycling(RrcProcedure::Setup), 0);
    }

    #[test]
    fn allocators_are_independent_per_ue() {
        let mut ue1 = RrcTransactionAllocator::new();
        let mut ue2 = RrcTransactionAllocator::new();

        // UE1 advances twice.
        assert_eq!(ue1.allocate_cycling(RrcProcedure::Setup), 0);
        assert_eq!(ue1.allocate_cycling(RrcProcedure::SecurityMode), 1);

        // UE2 is untouched by UE1's allocations — its first tid is still 0.
        assert_eq!(ue2.allocate_cycling(RrcProcedure::Setup), 0);
        assert_eq!(ue2.allocate_cycling(RrcProcedure::SecurityMode), 1);

        // UE1 continues its own independent cycle.
        assert_eq!(ue1.allocate_cycling(RrcProcedure::Reconfiguration), 2);
    }

    #[test]
    fn outstanding_is_tracked_per_procedure() {
        let mut a = RrcTransactionAllocator::new();
        a.allocate_cycling(RrcProcedure::Setup); // 0
        a.allocate_cycling(RrcProcedure::SecurityMode); // 1
        assert_eq!(a.outstanding(RrcProcedure::Setup), Some(0));
        assert_eq!(a.outstanding(RrcProcedure::SecurityMode), Some(1));
        assert_eq!(a.outstanding(RrcProcedure::Reconfiguration), None);
    }

    #[test]
    fn verify_matches_and_clears() {
        let mut a = RrcTransactionAllocator::new();
        let tid = a.allocate_cycling(RrcProcedure::SecurityMode); // 0
        assert_eq!(
            a.verify(RrcProcedure::SecurityMode, tid),
            TidVerification::Match
        );
        // Once matched, the transaction is cleared.
        assert_eq!(a.outstanding(RrcProcedure::SecurityMode), None);
        assert_eq!(
            a.verify(RrcProcedure::SecurityMode, tid),
            TidVerification::NoOutstanding
        );
    }

    #[test]
    fn verify_mismatch_is_fail_closed_and_keeps_pending() {
        let mut a = RrcTransactionAllocator::new();
        a.allocate_cycling(RrcProcedure::Setup); // 0
        a.allocate_cycling(RrcProcedure::SecurityMode); // 1 (Setup outstanding = 0)
        assert_eq!(
            a.verify(RrcProcedure::Setup, 1),
            TidVerification::Mismatch { expected: 0 }
        );
        // Fail-closed: the mismatch does NOT consume the outstanding tid.
        assert_eq!(a.outstanding(RrcProcedure::Setup), Some(0));
        // The correct echo still verifies afterwards.
        assert_eq!(a.verify(RrcProcedure::Setup, 0), TidVerification::Match);
    }

    #[test]
    fn verify_no_outstanding_when_never_allocated() {
        let mut a = RrcTransactionAllocator::new();
        assert_eq!(
            a.verify(RrcProcedure::Reconfiguration, 0),
            TidVerification::NoOutstanding
        );
    }

    #[test]
    fn allocate_pins_zero_while_c5_off() {
        // The production `allocate` honours the wire-safety gate. This test pins
        // the C5-off behavior (every gNB→UE tid is 0, so the UE's legacy nibble
        // dispatcher keeps routing); it does not apply once
        // `C5_TYPED_DCCH_DISPATCH` is flipped on (allocate then cycles 0..3).
        if C5_TYPED_DCCH_DISPATCH {
            return;
        }
        let mut a = RrcTransactionAllocator::new();
        assert_eq!(a.allocate(RrcProcedure::Setup), 0);
        assert_eq!(a.allocate(RrcProcedure::SecurityMode), 0);
        assert_eq!(a.allocate(RrcProcedure::Reconfiguration), 0);
        // All recorded as outstanding 0 so verification still runs.
        assert_eq!(a.outstanding(RrcProcedure::Setup), Some(0));
        assert_eq!(a.outstanding(RrcProcedure::SecurityMode), Some(0));
        // A UE echoing 0 (as it does until it learns to echo the real tid in
        // C5) verifies cleanly against the pinned transaction.
        assert_eq!(a.verify(RrcProcedure::Setup, 0), TidVerification::Match);
    }

    #[test]
    fn pin_zero_records_zero_without_advancing() {
        let mut a = RrcTransactionAllocator::new();
        assert_eq!(a.pin_zero(RrcProcedure::Setup), 0);
        assert_eq!(a.pin_zero(RrcProcedure::SecurityMode), 0);
        assert_eq!(a.outstanding(RrcProcedure::Setup), Some(0));
        // pin_zero must not advance the cycle: a subsequent cycling allocation
        // still starts at 0.
        assert_eq!(a.allocate_cycling(RrcProcedure::Reconfiguration), 0);
    }
}
