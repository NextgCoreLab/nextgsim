//! 5GMM (5G Mobility Management) Procedures
//!
//! This module implements the UE-side 5GMM procedures as defined in 3GPP TS 24.501:
//! - Registration procedure (initial, mobility, periodic)
//! - Deregistration procedure (UE-initiated, network-initiated)
//! - Service request procedure
//! - Authentication procedure
//! - Security mode control procedure
//!
//! # State Machine
//!
//! The MM state machine follows 3GPP TS 24.501 Section 5.1.3:
//! - RM states: RM-DEREGISTERED, RM-REGISTERED
//! - CM states: CM-IDLE, CM-CONNECTED
//! - MM states: MM-NULL, MM-DEREGISTERED, MM-REGISTERED-INITIATED, etc.
//! - U-states: U1-UPDATED, U2-NOT-UPDATED, U3-ROAMING-NOT-ALLOWED
//!
//! The `MmStateMachine` struct manages all state transitions and provides
//! callbacks for state change events.
//!
//! # Reference
//!
//! Based on UERANSIM's `src/ue/nas/mm/` implementation.

mod config_update;
mod deregistration;
mod emergency;
mod mint;
mod orchestrator;
mod persistence;
mod service_request;
mod state;
mod suci;
mod uuaa;

pub use config_update::*;
pub use deregistration::*;
pub use emergency::*;
pub use mint::*;
pub use orchestrator::*;
pub use persistence::*;
pub use service_request::*;
pub use state::*;
pub use suci::*;
pub use uuaa::*;

/// Which registration a UE may attempt on a cell of the given category
/// (TS 38.304 §4.4, TS 23.122 §3.3, issue #50).
///
/// A **suitable** cell allows normal service, so initial registration. An
/// **acceptable** cell allows only limited service — emergency calls, ETWS,
/// CMAS — so only emergency registration. Attempting normal registration on an
/// acceptable-only cell claims a service the cell cannot give.
///
/// A free function rather than logic inside `main.rs`'s NAS task closure, so the
/// policy is testable: the closure that consumes `ActiveCellChanged` is only
/// reachable by starting the binary, and that is where this decision used to
/// live implicitly (as an unconditional initial registration).
///
/// [`CellCategory::None`] maps to emergency as well. It is not reachable from a
/// successful camp — cell selection sets a category on every cell it returns —
/// but normal service needs POSITIVE evidence of suitability, so an unknown
/// category must not be read as suitable.
pub fn registration_type_for_cell_category(
    category: crate::rrc::cell_selection::CellCategory,
) -> nextgsim_nas::ies::RegistrationType {
    use crate::rrc::cell_selection::CellCategory;
    use nextgsim_nas::ies::RegistrationType;

    match category {
        CellCategory::SuitableCell => RegistrationType::InitialRegistration,
        CellCategory::AcceptableCell | CellCategory::None => {
            RegistrationType::EmergencyRegistration
        }
    }
}

#[cfg(test)]
mod cell_category_registration_tests {
    use super::registration_type_for_cell_category;
    use crate::rrc::cell_selection::CellCategory;
    use nextgsim_nas::ies::RegistrationType;

    /// #50, criterion 4: camping on an acceptable-only cell must lead to
    /// EMERGENCY registration, not normal initial registration.
    #[test]
    fn an_acceptable_only_cell_permits_only_emergency_registration() {
        assert_eq!(
            registration_type_for_cell_category(CellCategory::AcceptableCell),
            RegistrationType::EmergencyRegistration,
            "an acceptable cell gives limited service only (TS 38.304 §4.4), so \
             normal initial registration claims a service it cannot give"
        );
    }

    /// The other half of the pair. Without it the test above would still pass if
    /// every category mapped to emergency, which would break normal attach
    /// entirely.
    #[test]
    fn a_suitable_cell_permits_normal_initial_registration() {
        assert_eq!(
            registration_type_for_cell_category(CellCategory::SuitableCell),
            RegistrationType::InitialRegistration
        );
    }

    /// An unknown category is treated as limited service, because normal service
    /// needs positive evidence of suitability.
    #[test]
    fn an_unknown_category_is_treated_as_limited_service() {
        assert_eq!(
            registration_type_for_cell_category(CellCategory::None),
            RegistrationType::EmergencyRegistration
        );
    }
}
