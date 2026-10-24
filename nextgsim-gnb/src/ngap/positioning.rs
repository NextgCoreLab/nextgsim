//! The positioning SRS this gNB configures for its UEs, for Multi-RTT over
//! NRPPa (TS 38.455 §8.2.6, §8.2.9, §8.2.10, §8.5.1; nextgsim #347).
//!
//! A Positioning Information Request configures a UE's SRS. A Positioning
//! Activation starts a semi-persistent or aperiodic one, and a Positioning
//! Deactivation stops it. A Measurement Request names no UE: it names TRPs and
//! carries the SRS Configuration this node gave the LMF, so the measured UE is
//! the one whose active SRS that configuration names.

use std::collections::HashMap;

use nextgsim_ngap::procedures::nrppa::{
    NrppaRadioCause, SrsActivation, SrsAllocation, SrsDeactivation, SrsResourceType,
};

/// `possrsResourceSetID ::= INTEGER (0..15)`: at most 16 UEs hold one at once.
const RESOURCE_SET_IDS: u8 = 16;

#[derive(Debug, Clone, Copy)]
struct UeSrs {
    srs: SrsAllocation,
    /// The UE is transmitting it: a periodic set from configuration, a
    /// semi-persistent one between activation and deactivation, an aperiodic
    /// one once triggered.
    active: bool,
}

/// The positioning SRS of each UE, keyed by the NGAP task's UE id.
#[derive(Debug, Default)]
pub(crate) struct PositioningSrsTable {
    by_ue: HashMap<i32, UeSrs>,
}

impl PositioningSrsTable {
    /// Forgets the SRS of every UE `live` no longer holds.
    fn prune(&mut self, live: impl Fn(i32) -> bool) {
        self.by_ue.retain(|ue_id, _| live(*ue_id));
    }

    /// Configures `ue_id`'s positioning SRS of `resource_type`. A UE keeps its
    /// resource set id across a reconfiguration; a new UE gets the lowest id
    /// no live UE holds.
    pub(crate) fn configure(
        &mut self,
        ue_id: i32,
        resource_type: SrsResourceType,
        pci: u16,
        point_a: u32,
        live: impl Fn(i32) -> bool,
    ) -> Result<SrsAllocation, NrppaRadioCause> {
        self.prune(live);
        let resource_set_id = match self.by_ue.get(&ue_id) {
            Some(existing) => existing.srs.resource_set_id,
            None => (0..RESOURCE_SET_IDS)
                .find(|id| self.by_ue.values().all(|u| u.srs.resource_set_id != *id))
                .ok_or(NrppaRadioCause::RequestedItemTemporarilyNotAvailable)?,
        };
        let srs = SrsAllocation {
            resource_set_id,
            resource_type,
            // The UE id keeps two UEs' sequences apart.
            sequence_id: (ue_id as u32 & 0xFFFF) as u16,
            pci,
            point_a,
        };
        self.by_ue.insert(
            ue_id,
            UeSrs {
                srs,
                active: resource_type == SrsResourceType::Periodic,
            },
        );
        Ok(srs)
    }

    /// Activates `ue_id`'s SRS (§8.2.9). It must be the configured set, of
    /// the activated type.
    pub(crate) fn activate(
        &mut self,
        ue_id: i32,
        activation: SrsActivation,
    ) -> Result<(), NrppaRadioCause> {
        let entry = self
            .by_ue
            .get_mut(&ue_id)
            .ok_or(NrppaRadioCause::RequestedItemTemporarilyNotAvailable)?;
        let matches = match activation {
            SrsActivation::SemiPersistent { resource_set_id } => {
                entry.srs.resource_type == SrsResourceType::SemiPersistent
                    && entry.srs.resource_set_id == resource_set_id
            }
            SrsActivation::Aperiodic => entry.srs.resource_type == SrsResourceType::Aperiodic,
        };
        if !matches {
            return Err(NrppaRadioCause::RequestedItemNotSupported);
        }
        entry.active = true;
        Ok(())
    }

    /// Deactivates (§8.2.10): the named set stops, or every resource of the UE
    /// is released.
    pub(crate) fn deactivate(&mut self, ue_id: i32, deactivation: SrsDeactivation) {
        match deactivation {
            SrsDeactivation::ResourceSet(id) => {
                if let Some(entry) = self.by_ue.get_mut(&ue_id) {
                    if entry.srs.resource_set_id == id
                        && entry.srs.resource_type != SrsResourceType::Periodic
                    {
                        entry.active = false;
                    }
                }
            }
            SrsDeactivation::ReleaseAll => {
                self.by_ue.remove(&ue_id);
            }
        }
    }

    /// The UE a Measurement Request measures: the one whose active SRS the
    /// request's SRS Configuration names (`sets`, each with its carrier's PCI).
    /// A request with no SRS Configuration measures the one UE with an active
    /// SRS, if there is exactly one.
    pub(crate) fn measured_ue(
        &mut self,
        sets: &[(u8, Option<u16>)],
        live: impl Fn(i32) -> bool,
    ) -> Result<i32, NrppaRadioCause> {
        self.prune(live);
        let mut active = self.by_ue.iter().filter(|(_, u)| u.active);
        let found = if sets.is_empty() {
            match (active.next(), active.next()) {
                (Some((ue_id, _)), None) => Some(*ue_id),
                _ => None,
            }
        } else {
            active
                .find(|(_, u)| {
                    sets.iter().any(|(id, pci)| {
                        *id == u.srs.resource_set_id && pci.is_none_or(|p| p == u.srs.pci)
                    })
                })
                .map(|(ue_id, _)| *ue_id)
        };
        found.ok_or(NrppaRadioCause::RequestedItemTemporarilyNotAvailable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PCI: u16 = 42;

    fn all_live(_: i32) -> bool {
        true
    }

    #[test]
    fn each_ue_gets_its_own_resource_set_and_keeps_it() {
        let mut t = PositioningSrsTable::default();
        let a = t
            .configure(1, SrsResourceType::SemiPersistent, PCI, 0, all_live)
            .unwrap();
        let b = t
            .configure(2, SrsResourceType::SemiPersistent, PCI, 0, all_live)
            .unwrap();
        assert_ne!(a.resource_set_id, b.resource_set_id);
        assert_ne!(a.sequence_id, b.sequence_id);
        let again = t
            .configure(1, SrsResourceType::SemiPersistent, PCI, 0, all_live)
            .unwrap();
        assert_eq!(again.resource_set_id, a.resource_set_id);
    }

    #[test]
    fn a_seventeenth_ue_waits_until_a_set_is_freed() {
        let mut t = PositioningSrsTable::default();
        for ue in 0..16 {
            t.configure(ue, SrsResourceType::SemiPersistent, PCI, 0, all_live)
                .unwrap();
        }
        assert_eq!(
            t.configure(16, SrsResourceType::SemiPersistent, PCI, 0, all_live),
            Err(NrppaRadioCause::RequestedItemTemporarilyNotAvailable)
        );
        // UE 3 is gone: its set is free again.
        let srs = t
            .configure(16, SrsResourceType::SemiPersistent, PCI, 0, |ue| ue != 3)
            .unwrap();
        assert_eq!(srs.resource_set_id, 3);
    }

    #[test]
    fn a_semi_persistent_srs_is_measured_only_between_activation_and_deactivation() {
        let mut t = PositioningSrsTable::default();
        let srs = t
            .configure(5, SrsResourceType::SemiPersistent, PCI, 0, all_live)
            .unwrap();
        let sets = [(srs.resource_set_id, Some(PCI))];
        assert!(
            t.measured_ue(&sets, all_live).is_err(),
            "configured, not transmitting"
        );
        t.activate(
            5,
            SrsActivation::SemiPersistent {
                resource_set_id: srs.resource_set_id,
            },
        )
        .unwrap();
        assert_eq!(t.measured_ue(&sets, all_live), Ok(5));
        t.deactivate(5, SrsDeactivation::ResourceSet(srs.resource_set_id));
        assert!(t.measured_ue(&sets, all_live).is_err());
        // Released: no longer even configured.
        t.deactivate(5, SrsDeactivation::ReleaseAll);
        assert_eq!(
            t.activate(
                5,
                SrsActivation::SemiPersistent {
                    resource_set_id: srs.resource_set_id
                }
            ),
            Err(NrppaRadioCause::RequestedItemTemporarilyNotAvailable)
        );
    }

    #[test]
    fn an_activation_must_name_the_configured_set_and_type() {
        let mut t = PositioningSrsTable::default();
        let srs = t
            .configure(5, SrsResourceType::SemiPersistent, PCI, 0, all_live)
            .unwrap();
        let wrong = (srs.resource_set_id + 1) % RESOURCE_SET_IDS;
        assert_eq!(
            t.activate(
                5,
                SrsActivation::SemiPersistent {
                    resource_set_id: wrong
                }
            ),
            Err(NrppaRadioCause::RequestedItemNotSupported)
        );
        assert_eq!(
            t.activate(5, SrsActivation::Aperiodic),
            Err(NrppaRadioCause::RequestedItemNotSupported)
        );
        assert_eq!(
            t.activate(6, SrsActivation::Aperiodic),
            Err(NrppaRadioCause::RequestedItemTemporarilyNotAvailable),
            "no SRS configured"
        );
    }

    #[test]
    fn the_measured_ue_is_the_one_the_srs_configuration_names() {
        let mut t = PositioningSrsTable::default();
        for ue in [1, 2] {
            let srs = t
                .configure(ue, SrsResourceType::SemiPersistent, PCI, 0, all_live)
                .unwrap();
            t.activate(
                ue,
                SrsActivation::SemiPersistent {
                    resource_set_id: srs.resource_set_id,
                },
            )
            .unwrap();
        }
        let set_of_2 = t.by_ue[&2].srs.resource_set_id;
        assert_eq!(t.measured_ue(&[(set_of_2, Some(PCI))], all_live), Ok(2));
        assert!(
            t.measured_ue(&[(set_of_2, Some(PCI + 1))], all_live)
                .is_err(),
            "another cell's set"
        );
        assert!(
            t.measured_ue(&[], all_live).is_err(),
            "two candidates, none named"
        );
        assert_eq!(
            t.measured_ue(&[], |ue| ue == 1),
            Ok(1),
            "the only active UE"
        );
    }

    #[test]
    fn a_periodic_srs_is_transmitting_from_configuration() {
        let mut t = PositioningSrsTable::default();
        let srs = t
            .configure(9, SrsResourceType::Periodic, PCI, 0, all_live)
            .unwrap();
        assert_eq!(
            t.measured_ue(&[(srs.resource_set_id, None)], all_live),
            Ok(9)
        );
    }
}
