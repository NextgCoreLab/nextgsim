//! What the UE remembers while it is in RRC_INACTIVE (TS 38.331 §5.3.8.3,
//! §5.3.13; TS 38.304 §5.5).
//!
//! Before issue #38 the UE had a `RrcState::Inactive` it could never reach: no code
//! parsed a `suspendConfig`, `on_rrc_suspend` had no production caller, and there was
//! nowhere to keep the I-RNTI a resume would need. This module is that somewhere.
//!
//! # The two RNAU triggers, and why both are needed
//!
//! TS 38.304 §5.5 gives a UE in RRC_INACTIVE two reasons to come back and say where
//! it is:
//!
//! - **Periodic**, on `t380` expiry. Without it a stationary UE would never be heard
//!   from again, and the network could not tell it apart from one that had left.
//! - **RNA crossing**, on reselecting a cell outside the configured area. Without it
//!   the network would page a UE in cells it is no longer near.
//!
//! Either alone leaves a real gap, so [`InactiveContext`] implements both and
//! [`RnauTrigger`] names which fired — the log line is the only way to tell a
//! stationary UE's periodic update from a moving one's crossing.
//!
//! # Time is a parameter
//!
//! `t380` is evaluated from a `now` the caller passes, never from a clock read here.
//! A timer that read the clock itself would be untestable at exactly the boundary
//! that decides whether it expired, which is the boundary that matters.

use std::time::{Duration, Instant};

use nextgsim_rrc::procedures::suspend_config::{RanNotificationArea, SuspendConfigData};

/// Why an RNAU is being performed (TS 38.304 §5.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RnauTrigger {
    /// `t380` expired: the periodic update.
    Periodic,
    /// The UE reselected a cell outside its RAN Notification Area.
    RnaCrossing {
        /// The 36-bit NR Cell Identity the UE moved to.
        cell_identity: u64,
    },
}

impl std::fmt::Display for RnauTrigger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Periodic => write!(f, "t380 expiry (periodic RNAU)"),
            Self::RnaCrossing { cell_identity } => {
                write!(f, "RNA crossing into cell {cell_identity:#x}")
            }
        }
    }
}

/// The suspend configuration the UE is living under, plus the `t380` deadline.
///
/// Created from an `RRCRelease` carrying a `suspendConfig` and dropped when the UE
/// leaves RRC_INACTIVE — in either direction, because a resumed *or* released UE must
/// not keep an I-RNTI the network has reassigned.
#[derive(Debug, Clone)]
pub struct InactiveContext {
    /// The configuration as signalled.
    config: SuspendConfigData,
    /// The physical cell identity of the PCell the `suspendConfig` arrived in.
    ///
    /// Stored rather than re-read at resume time: the `resumeMAC-I` covers the
    /// **source** PCI (TS 38.331 §5.3.13.3), and by the time the UE resumes it may
    /// have reselected, so reading the current cell would fail every verification.
    source_phys_cell_id: u16,
    /// When `t380` expires, if one was configured.
    t380_deadline: Option<Instant>,
}

impl InactiveContext {
    /// Enter RRC_INACTIVE under `config`, arming `t380` from `now`.
    pub fn new(config: SuspendConfigData, source_phys_cell_id: u16, now: Instant) -> Self {
        let t380_deadline = config
            .t380_minutes
            .map(|minutes| now + Duration::from_secs(u64::from(minutes) * 60));
        Self {
            config,
            source_phys_cell_id,
            t380_deadline,
        }
    }

    /// The full 40-bit I-RNTI.
    pub fn full_i_rnti(&self) -> u64 {
        self.config.full_i_rnti
    }

    /// The short 24-bit I-RNTI.
    pub fn short_i_rnti(&self) -> u32 {
        self.config.short_i_rnti
    }

    /// The PCI of the cell the `suspendConfig` arrived in — the `resumeMAC-I`'s
    /// source PCI.
    pub fn source_phys_cell_id(&self) -> u16 {
        self.source_phys_cell_id
    }

    /// `nextHopChainingCount` from the `suspendConfig`.
    pub fn next_hop_chaining_count(&self) -> u8 {
        self.config.next_hop_chaining_count
    }

    /// `ran-PagingCycle` in radio frames.
    pub fn ran_paging_cycle_rf(&self) -> u16 {
        self.config.ran_paging_cycle_rf
    }

    /// `t380` in minutes, if configured.
    pub fn t380_minutes(&self) -> Option<u16> {
        self.config.t380_minutes
    }

    /// The RAN Notification Area, if configured.
    pub fn ran_notification_area(&self) -> Option<&RanNotificationArea> {
        self.config.ran_notification_area.as_ref()
    }

    /// Whether `t380` has expired by `now`.
    ///
    /// `false` when no `t380` was configured, which is not the same as "not yet": a UE
    /// given no periodic timer must never fire one, and treating an absent timer as
    /// expired would have it resume immediately on every tick.
    pub fn t380_expired(&self, now: Instant) -> bool {
        self.t380_deadline.is_some_and(|deadline| now >= deadline)
    }

    /// Re-arm `t380` from `now`, as a completed RNAU does (TS 38.304 §5.5: the timer
    /// restarts when the UE has reported in, whatever the reason).
    pub fn restart_t380(&mut self, now: Instant) {
        self.t380_deadline = self
            .config
            .t380_minutes
            .map(|minutes| now + Duration::from_secs(u64::from(minutes) * 60));
    }

    /// Whether camping on `cell_identity` is an RNA crossing.
    ///
    /// With **no** area configured every cell other than the one the UE was suspended
    /// in is a crossing — TS 38.304 §5.5 makes the serving cell the area by default —
    /// so `suspended_in_cell` is needed to answer at all.
    pub fn is_rna_crossing(&self, cell_identity: u64, suspended_in_cell: u64) -> bool {
        match self.config.ran_notification_area.as_ref() {
            Some(area) => !area.contains(cell_identity),
            None => cell_identity != suspended_in_cell,
        }
    }

    /// The RNAU trigger that applies now, if any.
    ///
    /// Checks the crossing first: a UE that has both moved out of its area *and* run
    /// its timer down should say it moved, because that is the fact the network cannot
    /// infer. A periodic update from a cell the network does not expect looks like a
    /// stationary UE and would have it page the wrong cells.
    pub fn rnau_trigger(
        &self,
        now: Instant,
        current_cell_identity: Option<u64>,
        suspended_in_cell: u64,
    ) -> Option<RnauTrigger> {
        if let Some(cell) = current_cell_identity {
            if self.is_rna_crossing(cell, suspended_in_cell) {
                return Some(RnauTrigger::RnaCrossing {
                    cell_identity: cell,
                });
            }
        }
        if self.t380_expired(now) {
            return Some(RnauTrigger::Periodic);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(t380: Option<u16>, area: Option<Vec<u64>>) -> SuspendConfigData {
        SuspendConfigData {
            full_i_rnti: 0x12_3456_789A,
            short_i_rnti: 0x0056_789A,
            ran_paging_cycle_rf: 64,
            ran_notification_area: area.map(RanNotificationArea::CellList),
            t380_minutes: t380,
            next_hop_chaining_count: 3,
        }
    }

    /// #38, criterion 7: `t380` expires at the configured minute and not before.
    #[test]
    fn t380_expires_at_the_configured_minute() {
        let base = Instant::now();
        let ctx = InactiveContext::new(config(Some(30), None), 1, base);
        assert!(
            !ctx.t380_expired(base),
            "not expired the instant it started"
        );
        assert!(
            !ctx.t380_expired(base + Duration::from_secs(30 * 60 - 1)),
            "not expired one second early"
        );
        assert!(
            ctx.t380_expired(base + Duration::from_secs(30 * 60)),
            "expired exactly on the boundary"
        );
    }

    /// An absent `t380` never expires. Pinned because treating "no timer" as "expired"
    /// would make a UE with no periodic RNAU resume on the very next tick.
    #[test]
    fn an_absent_t380_never_expires() {
        let base = Instant::now();
        let ctx = InactiveContext::new(config(None, None), 1, base);
        assert!(!ctx.t380_expired(base));
        assert!(!ctx.t380_expired(base + Duration::from_secs(720 * 60 * 10)));
        assert_eq!(
            ctx.rnau_trigger(base + Duration::from_secs(10_000_000), None, 0x10),
            None,
            "and it must trigger no periodic RNAU either"
        );
    }

    /// A completed RNAU re-arms the timer, so a UE does not immediately fire again.
    #[test]
    fn a_completed_rnau_restarts_t380() {
        let base = Instant::now();
        let mut ctx = InactiveContext::new(config(Some(5), None), 1, base);
        let expiry = base + Duration::from_secs(5 * 60);
        assert!(ctx.t380_expired(expiry));
        ctx.restart_t380(expiry);
        assert!(
            !ctx.t380_expired(expiry),
            "restarting must clear the expiry, or the UE resumes on every tick"
        );
        assert!(ctx.t380_expired(expiry + Duration::from_secs(5 * 60)));
    }

    /// #38, criterion 8: a cell outside the configured area is a crossing, and one
    /// inside it is not.
    #[test]
    fn a_cell_outside_the_notification_area_is_a_crossing() {
        let base = Instant::now();
        let ctx = InactiveContext::new(config(Some(30), Some(vec![0x10, 0x11])), 1, base);
        assert!(!ctx.is_rna_crossing(0x10, 0x10), "the suspending cell");
        assert!(!ctx.is_rna_crossing(0x11, 0x10), "another area cell");
        assert!(ctx.is_rna_crossing(0x12, 0x10), "a cell outside the area");
        assert_eq!(
            ctx.rnau_trigger(base, Some(0x12), 0x10),
            Some(RnauTrigger::RnaCrossing {
                cell_identity: 0x12
            })
        );
        assert_eq!(
            ctx.rnau_trigger(base, Some(0x11), 0x10),
            None,
            "moving inside the area must NOT trigger an RNAU, or the area is pointless"
        );
    }

    /// With no area configured the serving cell IS the area (TS 38.304 §5.5), so any
    /// other cell is a crossing.
    #[test]
    fn with_no_area_the_suspending_cell_is_the_area() {
        let base = Instant::now();
        let ctx = InactiveContext::new(config(Some(30), None), 1, base);
        assert!(!ctx.is_rna_crossing(0x10, 0x10));
        assert!(ctx.is_rna_crossing(0x11, 0x10));
        assert_eq!(
            ctx.rnau_trigger(base, Some(0x11), 0x10),
            Some(RnauTrigger::RnaCrossing {
                cell_identity: 0x11
            })
        );
    }

    /// A crossing takes precedence over an expiry, because the network cannot infer a
    /// move from a periodic update.
    #[test]
    fn a_crossing_takes_precedence_over_a_periodic_expiry() {
        let base = Instant::now();
        let ctx = InactiveContext::new(config(Some(5), Some(vec![0x10])), 1, base);
        let expired = base + Duration::from_secs(5 * 60);
        assert!(ctx.t380_expired(expired), "precondition: t380 has expired");
        assert_eq!(
            ctx.rnau_trigger(expired, Some(0x99), 0x10),
            Some(RnauTrigger::RnaCrossing {
                cell_identity: 0x99
            }),
            "the crossing is the fact the network cannot work out for itself"
        );
        // And with no move, the same instant gives the periodic trigger — the positive
        // control, without which a `rnau_trigger` that only ever reported crossings
        // would pass.
        assert_eq!(
            ctx.rnau_trigger(expired, Some(0x10), 0x10),
            Some(RnauTrigger::Periodic)
        );
    }

    /// An unknown current cell cannot be judged a crossing, but the periodic timer
    /// still applies: a UE that has lost coverage must still report in when it can.
    #[test]
    fn an_unknown_current_cell_leaves_only_the_periodic_trigger() {
        let base = Instant::now();
        let ctx = InactiveContext::new(config(Some(5), Some(vec![0x10])), 1, base);
        assert_eq!(ctx.rnau_trigger(base, None, 0x10), None);
        assert_eq!(
            ctx.rnau_trigger(base + Duration::from_secs(5 * 60), None, 0x10),
            Some(RnauTrigger::Periodic)
        );
    }

    /// The stored source PCI is the one the config arrived in, which is what the
    /// `resumeMAC-I` covers.
    #[test]
    fn the_source_pci_is_the_suspending_cells_and_is_kept() {
        let ctx = InactiveContext::new(config(Some(30), None), 407, Instant::now());
        assert_eq!(
            ctx.source_phys_cell_id(),
            407,
            "the resumeMAC-I covers the SOURCE PCI; reading the current cell at resume \
             time would fail every verification after a reselection"
        );
        assert_eq!(ctx.full_i_rnti(), 0x12_3456_789A);
        assert_eq!(ctx.short_i_rnti(), 0x0056_789A);
        assert_eq!(ctx.next_hop_chaining_count(), 3);
        assert_eq!(ctx.ran_paging_cycle_rf(), 64);
    }
}
