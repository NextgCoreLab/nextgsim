//! NAS (Non-Access Stratum) Protocol Handling
//!
//! This module implements the UE-side NAS protocol handling including:
//! - 5GMM (Mobility Management) procedures
//! - 5GSM (Session Management) procedures
//!
//! # Architecture
//!
//! The NAS layer is organized into:
//! - `mm`: Mobility Management procedures (registration, deregistration, authentication, etc.)
//! - `sm`: Session Management procedures (PDU session establishment, modification, release)
//! - `lpp`: LPP positioning procedures carried in the NAS payload container
//!
//! # Reference
//!
//! Based on 3GPP TS 24.501 and UERANSIM's `src/ue/nas/` implementation.

pub mod lpp;
pub mod mm;
pub mod sm;

/// Drive the re-attempt of configured PDU sessions whose establishment failed
/// (#379, TS 24.501 §6.4.1.4.2); call once a second after the SM tick.
///
/// A due re-attempt goes out once the UE is in 5GMM-REGISTERED.NORMAL-SERVICE
/// with a NAS signalling connection. From 5GMM-IDLE it first asks for one with
/// a SERVICE REQUEST for signalling (TS 24.501 §5.6.1.2), which the MM
/// orchestrator holds back while T3346 or another bar is in force (§5.6.1.7);
/// the next call after T3346 expires asks again. Returns the MM and SM outputs
/// to deliver, in that order.
pub fn drive_session_retries(
    mm: &mut mm::MmOrchestrator,
    sm: &mut sm::SmOrchestrator,
) -> (Vec<mm::MmOutput>, Vec<sm::SmOutput>) {
    use nextgsim_nas::ies::ie1::ServiceType;

    if !sm.has_due_retry() || mm.state().mm_substate() != mm::MmSubState::RegisteredNormalService {
        return (Vec::new(), Vec::new());
    }
    if mm.state().is_connected() {
        return (Vec::new(), sm.retry_wanted_sessions());
    }
    (
        mm.start_service_request(ServiceType::Signalling, None),
        Vec::new(),
    )
}
