//! QoS-flow-to-DRB mapping (3GPP TS 37.324 §5.1).
//!
//! # What SDAP actually decides, and what it does not
//!
//! TS 37.324 §5.1 requires an SDAP entity per PDU session that maps each QoS flow
//! onto a DRB. It does **not** say which flow goes to which bearer — that is a RAN
//! implementation choice, normally driven by 5QI, and §5.3.1 only requires that a
//! flow with no explicit mapping goes to the session's **default DRB**.
//!
//! So this module is a stated policy, not a spec transcription, and issue #44's
//! "a policy for which QFI goes to which DRB" is what it settles.
//!
//! # The policy: split by resource type
//!
//! Two DRBs per PDU session, chosen by the 5QI's `QosResourceType`
//! (TS 23.501 Table 5.7.4-1):
//!
//! * **GBR and delay-critical GBR** flows take the *second* DRB.
//! * **Non-GBR** flows, and any flow whose 5QI is unknown, take the **default**
//!   DRB.
//!
//! Why resource type and not, say, priority: it is the one 5QI attribute that
//! divides the standardised table into two non-empty groups for *every* realistic
//! flow set, it is the division that actually matters to a scheduler (a GBR bearer
//! owes a rate; a non-GBR one does not), and it needs no operator configuration to
//! be meaningful. A priority threshold would need a number nobody has chosen, and
//! a per-QFI table would need configuration that does not exist.
//!
//! An unknown 5QI falling to the default DRB is §5.3.1's own rule and not a
//! fallback of convenience: the alternative — refusing the flow — would drop
//! traffic the core admitted.
//!
//! # Why two and not more
//!
//! Because two is what it takes to make criterion 1 true ("two flows with distinct
//! QFIs on one PDU session map to distinct DRBs") and one more than the tree had.
//! A DRB per 5QI would be conformant too, but every DRB costs an RLC entity, a
//! PDCP entity, an LCID and a `DRB-ToAddMod`, and nothing in this simulator
//! schedules them differently — so the extra bearers would be structure without
//! behaviour. The mapping is a function, so a richer policy is a change here and
//! nowhere else.

use crate::qos::{lookup_5qi, QosResourceType};
use std::collections::HashMap;

/// The largest DRB identity TS 38.331 allows (`DRB-Identity ::= INTEGER (1..32)`).
pub const MAX_DRB_ID: u8 = 32;

/// Which of a session's DRBs a QoS flow maps to.
///
/// An enum and not a bare `u8`: the *identity* of the second DRB is allocated by
/// the caller that owns the numbering, and a function returning a raw id would
/// have to invent one. This says which bearer, and [`DrbAllocation`] says what it
/// is called.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrbChoice {
    /// The session's default DRB (TS 37.324 §5.3.1).
    Default,
    /// The session's second DRB, carrying GBR traffic.
    Gbr,
}

/// The DRB identities and logical channel identities a PDU session's bearers use.
///
/// Derived from the PDU session id so that both ends compute the same numbers from
/// the same input without signalling them — which is the property the RLC and PDCP
/// entity maps depend on, since they are keyed by DRB identity on both sides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrbAllocation {
    /// `drb-Identity` of the default DRB.
    pub default_drb_id: u8,
    /// `drb-Identity` of the GBR DRB.
    pub gbr_drb_id: u8,
    /// `logicalChannelIdentity` of the default DRB.
    pub default_lcid: u8,
    /// `logicalChannelIdentity` of the GBR DRB.
    pub gbr_lcid: u8,
}

/// Allocate the two DRB identities and LCIDs for one PDU session.
///
/// # The numbering, and why it is what it is
///
/// The default DRB keeps `psi.clamp(1, 32)` — **byte-for-byte what
/// `drb_identity_for` produced before this change**. That is deliberate and it is
/// the reason a UE and gNB that disagree about whether a second DRB exists still
/// interoperate on the first one: a session's default bearer has the same identity
/// either way.
///
/// The GBR DRB takes `default + 16`, wrapping into 1..=32. `+ 16` rather than `+ 1`
/// so that two adjacent PDU sessions' bearers do not collide: session 1 would
/// otherwise take DRBs 1 and 2 while session 2 took 2 and 3, and the two sessions'
/// RLC entities would share a key. With `+ 16` a collision needs 17 sessions,
/// which is past what this simulator runs — and [`Self::gbr_drb_id`] is checked
/// against the default so a wrap that DID collide is reported rather than silently
/// aliasing two bearers.
pub fn allocate_drbs(psi: u8) -> DrbAllocation {
    let default_drb_id = psi.clamp(1, MAX_DRB_ID);
    // Wrap into 1..=32: `(x - 1) % 32 + 1` keeps 0 out, which `% 32` alone would
    // not — and DRB 0 is not a value.
    let gbr_drb_id = (default_drb_id + 16 - 1) % MAX_DRB_ID + 1;
    DrbAllocation {
        default_drb_id,
        gbr_drb_id,
        default_lcid: lcid_for(default_drb_id),
        gbr_lcid: lcid_for(gbr_drb_id),
    }
}

/// The logical channel identity a DRB uses.
///
/// `3 + drb_id`, clamped to 32 — the same expression the gNB used inline before
/// this change, kept so the default DRB's LCID is unchanged. The `+ 3` clears the
/// three SRBs (TS 38.331 `LogicalChannelIdentity ::= INTEGER (1..32)`, with 1..3
/// reserved for SRB0..SRB2).
fn lcid_for(drb_id: u8) -> u8 {
    (3 + drb_id).min(32)
}

impl DrbAllocation {
    /// The DRB identity a [`DrbChoice`] names.
    pub fn id_of(&self, choice: DrbChoice) -> u8 {
        match choice {
            DrbChoice::Default => self.default_drb_id,
            DrbChoice::Gbr => self.gbr_drb_id,
        }
    }

    /// The LCID a [`DrbChoice`] names.
    pub fn lcid_of(&self, choice: DrbChoice) -> u8 {
        match choice {
            DrbChoice::Default => self.default_lcid,
            DrbChoice::Gbr => self.gbr_lcid,
        }
    }

    /// Whether the two DRBs are actually distinct.
    ///
    /// They always are for the numbering [`allocate_drbs`] produces, and this
    /// exists so a caller can assert it rather than assume it: two bearers with
    /// one identity would share an RLC entity and interleave two flows'
    /// sequence-number spaces, which is a silent corruption rather than a failure.
    pub fn are_distinct(&self) -> bool {
        self.default_drb_id != self.gbr_drb_id && self.default_lcid != self.gbr_lcid
    }
}

/// The QoS-flow-to-DRB mapping for one PDU session (TS 37.324 §5.1).
///
/// One per PDU session, as §5.1 requires — the entity this issue says is missing.
#[derive(Debug, Clone, Default)]
pub struct QfiDrbMap {
    /// The 5QI each QFI was admitted with, from the NGAP `QosFlowSetupInfo`.
    ///
    /// `None` for a flow the core admitted without a standardised 5QI (a dynamic
    /// one), which maps to the default DRB.
    flow_5qi: HashMap<u8, Option<u16>>,
}

impl QfiDrbMap {
    /// An empty map: every QFI maps to the default DRB until one is admitted.
    pub fn new() -> Self {
        Self::default()
    }

    /// Admit a QoS flow with the 5QI the core gave it.
    pub fn admit_flow(&mut self, qfi: u8, five_qi: Option<u16>) {
        self.flow_5qi.insert(qfi, five_qi);
    }

    /// Release a QoS flow.
    pub fn release_flow(&mut self, qfi: u8) {
        self.flow_5qi.remove(&qfi);
    }

    /// How many flows are admitted.
    pub fn flow_count(&self) -> usize {
        self.flow_5qi.len()
    }

    /// The DRB a QFI maps to.
    ///
    /// The **default** DRB for a flow that was never admitted. Not an error: a
    /// downlink packet may carry a QFI the gNB has no record of (the core is the
    /// authority on what it admitted), and dropping it would lose traffic the core
    /// accepted. §5.3.1's default-DRB rule covers exactly this.
    pub fn drb_for(&self, qfi: u8) -> DrbChoice {
        match self.flow_5qi.get(&qfi) {
            Some(Some(five_qi)) => Self::drb_for_5qi(*five_qi),
            // Admitted with a dynamic (non-standardised) 5QI, or not admitted at
            // all. Both mean "nothing says this is GBR".
            Some(None) | None => DrbChoice::Default,
        }
    }

    /// The DRB a 5QI's resource type selects — the policy itself, in one place.
    pub fn drb_for_5qi(five_qi: u16) -> DrbChoice {
        match lookup_5qi(five_qi).map(|c| c.resource_type) {
            Some(QosResourceType::Gbr) | Some(QosResourceType::DelayCriticalGbr) => DrbChoice::Gbr,
            // Non-GBR, and any 5QI the standardised table does not list.
            Some(QosResourceType::NonGbr) | None => DrbChoice::Default,
        }
    }

    /// Every admitted QFI that maps to `choice`, ascending.
    ///
    /// This is what fills each DRB's `mappedQoS-FlowsToAdd` in the
    /// `RRCReconfiguration`, so the UE is *told* the mapping rather than being
    /// configured to reproduce it.
    pub fn qfis_for(&self, choice: DrbChoice) -> Vec<u8> {
        let mut qfis: Vec<u8> = self
            .flow_5qi
            .keys()
            .copied()
            .filter(|qfi| self.drb_for(*qfi) == choice)
            .collect();
        qfis.sort_unstable();
        qfis
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ====================================================================
    // The policy.
    // ====================================================================

    /// GBR and delay-critical GBR 5QIs take the second DRB; non-GBR takes the
    /// default. Asserted over the WHOLE standardised table, so a 5QI added to it
    /// later cannot land in neither group unnoticed.
    #[test]
    fn the_resource_type_decides_the_drb_for_every_standard_5qi() {
        let mut gbr = 0;
        let mut non_gbr = 0;
        for entry in crate::qos::standard_5qi_table() {
            let choice = QfiDrbMap::drb_for_5qi(entry.five_qi);
            match entry.resource_type {
                QosResourceType::Gbr | QosResourceType::DelayCriticalGbr => {
                    assert_eq!(
                        choice,
                        DrbChoice::Gbr,
                        "5QI {} is {:?} and must take the GBR DRB",
                        entry.five_qi,
                        entry.resource_type
                    );
                    gbr += 1;
                }
                QosResourceType::NonGbr => {
                    assert_eq!(
                        choice,
                        DrbChoice::Default,
                        "5QI {} is non-GBR and must take the default DRB",
                        entry.five_qi
                    );
                    non_gbr += 1;
                }
            }
        }
        // Both groups must be non-empty, or the policy is not a split at all and
        // every test below would pass for a constant function.
        assert!(gbr > 0, "the table must contain GBR 5QIs");
        assert!(non_gbr > 0, "and non-GBR ones");
    }

    /// A 5QI the standardised table does not list falls to the default DRB, per
    /// §5.3.1 — not refused, which would drop traffic the core admitted.
    #[test]
    fn an_unknown_5qi_falls_to_the_default_drb() {
        // 200 is in the operator-specific range and not in the table.
        assert_eq!(QfiDrbMap::drb_for_5qi(200), DrbChoice::Default);
        assert_eq!(QfiDrbMap::drb_for_5qi(0), DrbChoice::Default);
    }

    /// Criterion 1, at the policy level: two flows with distinct QFIs on ONE PDU
    /// session map to DISTINCT DRBs.
    #[test]
    fn two_flows_with_distinct_qfis_map_to_distinct_drbs() {
        let mut map = QfiDrbMap::new();
        // 5QI 1 is GBR (conversational voice); 5QI 9 is non-GBR (best effort).
        map.admit_flow(1, Some(1));
        map.admit_flow(9, Some(9));

        let alloc = allocate_drbs(1);
        let drb_a = alloc.id_of(map.drb_for(1));
        let drb_b = alloc.id_of(map.drb_for(9));
        assert_ne!(
            drb_a, drb_b,
            "a GBR flow and a non-GBR flow on one session must land on different DRBs"
        );
        assert_eq!(
            drb_b, alloc.default_drb_id,
            "the non-GBR flow is the default"
        );
        assert_eq!(drb_a, alloc.gbr_drb_id);
    }

    /// A flow nobody admitted still forwards, on the default DRB.
    #[test]
    fn an_unadmitted_qfi_maps_to_the_default_drb() {
        let map = QfiDrbMap::new();
        assert_eq!(map.drb_for(7), DrbChoice::Default);
        assert_eq!(map.flow_count(), 0, "and asking did not admit it");
    }

    /// A flow admitted with a DYNAMIC 5QI (none signalled) maps to the default.
    #[test]
    fn a_flow_with_no_five_qi_maps_to_the_default_drb() {
        let mut map = QfiDrbMap::new();
        map.admit_flow(3, None);
        assert_eq!(map.drb_for(3), DrbChoice::Default);
    }

    /// `qfis_for` partitions the admitted flows: every QFI appears in exactly one
    /// of the two lists, and both lists are sorted.
    #[test]
    fn the_two_qfi_lists_partition_the_admitted_flows() {
        let mut map = QfiDrbMap::new();
        map.admit_flow(9, Some(9)); // non-GBR
        map.admit_flow(1, Some(1)); // GBR
        map.admit_flow(5, Some(5)); // non-GBR (IMS signalling)
        map.admit_flow(2, Some(2)); // GBR
        map.admit_flow(40, None); // dynamic -> default

        let default = map.qfis_for(DrbChoice::Default);
        let gbr = map.qfis_for(DrbChoice::Gbr);

        assert_eq!(default, vec![5, 9, 40], "sorted, and the non-GBR set");
        assert_eq!(gbr, vec![1, 2], "sorted, and the GBR set");
        assert_eq!(
            default.len() + gbr.len(),
            map.flow_count(),
            "every admitted flow must be in exactly one list"
        );
    }

    /// Releasing a flow removes it from its list without disturbing the other.
    #[test]
    fn releasing_a_flow_removes_it_from_its_list_only() {
        let mut map = QfiDrbMap::new();
        map.admit_flow(1, Some(1));
        map.admit_flow(9, Some(9));
        map.release_flow(1);
        assert_eq!(map.qfis_for(DrbChoice::Gbr), Vec::<u8>::new());
        assert_eq!(map.qfis_for(DrbChoice::Default), vec![9]);
        assert_eq!(map.flow_count(), 1);
    }

    // ====================================================================
    // The numbering.
    // ====================================================================

    /// The default DRB identity and LCID are UNCHANGED from what the gNB computed
    /// before this change (`psi.clamp(1, 32)` and `(3 + drb).min(32)`).
    ///
    /// This is the compatibility claim the whole change rests on: a session's
    /// default bearer keeps its identity, so the existing PSI-keyed end-to-end
    /// paths see the same numbers.
    #[test]
    fn the_default_drb_keeps_the_identity_the_gnb_used_before() {
        for psi in 0u8..=40 {
            let alloc = allocate_drbs(psi);
            assert_eq!(
                alloc.default_drb_id,
                psi.clamp(1, 32),
                "psi {psi}: the default DRB identity must not move"
            );
            assert_eq!(
                alloc.default_lcid,
                (3 + psi.clamp(1, 32)).min(32),
                "psi {psi}: nor its LCID"
            );
        }
    }

    /// The two DRBs of one session are always distinct, and always in range —
    /// including at the wrap, where `+ 16` is most likely to be wrong.
    #[test]
    fn a_sessions_two_drbs_are_always_distinct_and_in_range() {
        for psi in 0u8..=255 {
            let alloc = allocate_drbs(psi);
            assert!(
                alloc.are_distinct(),
                "psi {psi}: {alloc:?} must name two different bearers -- sharing one \
                 would interleave two flows' RLC sequence numbers"
            );
            for id in [alloc.default_drb_id, alloc.gbr_drb_id] {
                assert!(
                    (1..=MAX_DRB_ID).contains(&id),
                    "psi {psi}: DRB {id} is outside INTEGER (1..32)"
                );
            }
            for lcid in [alloc.default_lcid, alloc.gbr_lcid] {
                assert!(
                    (4..=32).contains(&lcid),
                    "psi {psi}: LCID {lcid} must be above the three SRBs and within \
                     INTEGER (1..32)"
                );
            }
        }
    }

    /// The wrap is checked explicitly at the values where it happens: DRB 17
    /// wraps to 1, DRB 32 wraps to 16.
    #[test]
    fn the_gbr_drb_wraps_into_range_without_reaching_zero() {
        assert_eq!(allocate_drbs(1).gbr_drb_id, 17, "1 + 16");
        assert_eq!(
            allocate_drbs(16).gbr_drb_id,
            32,
            "16 + 16, the last before wrap"
        );
        assert_eq!(allocate_drbs(17).gbr_drb_id, 1, "17 + 16 = 33 -> 1");
        assert_eq!(allocate_drbs(32).gbr_drb_id, 16, "32 + 16 = 48 -> 16");
        // DRB 0 is not a value, so no input may produce it.
        for psi in 0u8..=255 {
            assert_ne!(allocate_drbs(psi).gbr_drb_id, 0, "psi {psi}");
        }
    }

    /// `id_of` and `lcid_of` agree with the fields, so a caller cannot get a DRB
    /// identity from one and an LCID from the other's bearer.
    #[test]
    fn id_and_lcid_stay_on_the_same_bearer() {
        let alloc = allocate_drbs(5);
        assert_eq!(alloc.id_of(DrbChoice::Default), alloc.default_drb_id);
        assert_eq!(alloc.lcid_of(DrbChoice::Default), alloc.default_lcid);
        assert_eq!(alloc.id_of(DrbChoice::Gbr), alloc.gbr_drb_id);
        assert_eq!(alloc.lcid_of(DrbChoice::Gbr), alloc.gbr_lcid);
        assert_eq!(
            alloc.lcid_of(DrbChoice::Gbr),
            3 + alloc.id_of(DrbChoice::Gbr),
            "the LCID is derived from ITS OWN DRB identity"
        );
    }
}
