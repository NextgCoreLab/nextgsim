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

/// The SM timers of a subscription with its own NAS signalling connection
/// (the primary), once a second (#409 follow-up).
///
/// A procedure timer's retransmission (TS 24.501 §6.4.1.6 a), §6.4.2.6 a),
/// §6.4.3.6 a)) goes in UL NAS TRANSPORT, which needs 5GMM-CONNECTED. In
/// 5GMM-IDLE the SM orchestrator holds it, and the UE in
/// 5GMM-REGISTERED.NORMAL-SERVICE sends a SERVICE REQUEST for the pending
/// uplink signalling (§5.6.1.1, service type "signalling", §5.6.1.2); the MM
/// orchestrator holds that back while T3346 or another bar is in force. The
/// held retransmission goes on the first call with the connection up. Before,
/// a retransmission from 5GMM-IDLE went to RRC as if it were an initial NAS
/// message. Returns the MM and SM outputs to deliver, in that order.
pub fn drive_sm_timers(
    mm: &mut mm::MmOrchestrator,
    sm: &mut sm::SmOrchestrator,
) -> (Vec<mm::MmOutput>, Vec<sm::SmOutput>) {
    use nextgsim_nas::ies::ie1::ServiceType;

    let connected = mm.state().is_connected();
    let sm_outs = sm.tick_with_connection(sm_transport_ready(mm));
    let mm_outs = if !connected
        && sm.has_held_retransmission()
        && mm.state().mm_substate() == mm::MmSubState::RegisteredNormalService
    {
        mm.start_service_request(ServiceType::Signalling, None)
    } else {
        Vec::new()
    };
    (mm_outs, sm_outs)
}

/// Whether a 5GSM message can go now: a NAS signalling connection in
/// 5GMM-REGISTERED.NORMAL-SERVICE, so not while the SERVICE REQUEST that
/// asked for the connection is still unanswered (#420 review).
fn sm_transport_ready(mm: &mm::MmOrchestrator) -> bool {
    mm.state().is_connected() && mm.state().mm_substate() == mm::MmSubState::RegisteredNormalService
}

/// The held 5GSM retransmissions, once the connection a SERVICE REQUEST asked
/// for is up (#420 review): call after each downlink NAS message, so they go
/// on the Service Accept rather than on the next tick.
pub fn flush_held_sm(mm: &mm::MmOrchestrator, sm: &mut sm::SmOrchestrator) -> Vec<sm::SmOutput> {
    if sm_transport_ready(mm) && sm.has_held_retransmission() {
        sm.take_held_retransmissions()
    } else {
        Vec::new()
    }
}

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
