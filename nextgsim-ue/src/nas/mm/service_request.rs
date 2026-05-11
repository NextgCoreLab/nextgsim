//! Service Request Procedure
//!
//! This module implements the UE-side service request procedure as defined in
//! 3GPP TS 24.501 Section 5.6.1.
//!
//! # Purpose
//!
//! The service request procedure is used by the UE to:
//! - Transition from CM-IDLE to CM-CONNECTED
//! - Trigger paging response (MT service)
//! - Reactivate PDU sessions that are suspended
//!
//! # Trigger Conditions
//!
//! The UE initiates service request when:
//! 1. UE has uplink data to send (MO data)
//! 2. UE receives a paging message (MT services)
//! 3. UE needs emergency services (Emergency)
//!
//! # Timer T3517
//!
//! Timer T3517 governs the service request procedure:
//! - Started when Service Request is sent
//! - Stopped when Service Accept or Service Reject is received
//! - On expiry: retransmit (up to max attempts)
//!
//! # Reference
//!
//! Based on UERANSIM's service request handling in `src/ue/nas/mm/`.

use thiserror::Error;

use nextgsim_nas::ies::ie1::{IeServiceType, ServiceType};
use nextgsim_nas::messages::mm::{
    Ie5gsMobileIdentity, MmCause, ServiceAccept, ServiceReject, ServiceRequest,
};
use nextgsim_nas::security::NasKeySetIdentifier;

use super::state::{CmState, MmState, MmSubState, RmState, UpdateStatus};
use crate::timer::UeTimer;

// ============================================================================
// Constants
// ============================================================================

/// Timer T3517 code (Service Request timer)
pub const T3517_CODE: u16 = 3517;

/// Default T3517 interval: 30 seconds per 3GPP TS 24.501 Table 10.2.1
pub const T3517_DEFAULT_INTERVAL_SECS: u32 = 30;

/// Maximum T3517 retransmission count
pub const T3517_MAX_RETRANSMISSION: u32 = 5;

// ============================================================================
// Service Request Cause
// ============================================================================

/// Reason for initiating the service request procedure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceRequestCause {
    /// UE has MO (mobile-originated) data to send
    MoData,
    /// UE has MO signalling to send
    MoSignalling,
    /// UE received a paging message (MT services)
    MtServices,
    /// Emergency services request
    EmergencyServices,
    /// High-priority access
    HighPriorityAccess,
}

impl ServiceRequestCause {
    /// Maps the cause to the 5G NAS `ServiceType` IE value.
    pub fn to_service_type(self) -> ServiceType {
        match self {
            ServiceRequestCause::MoData => ServiceType::Data,
            ServiceRequestCause::MoSignalling => ServiceType::Signalling,
            ServiceRequestCause::MtServices => ServiceType::MobileTerminatedServices,
            ServiceRequestCause::EmergencyServices => ServiceType::EmergencyServices,
            ServiceRequestCause::HighPriorityAccess => ServiceType::HighPriorityAccess,
        }
    }
}

impl std::fmt::Display for ServiceRequestCause {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ServiceRequestCause::MoData => write!(f, "mo-data"),
            ServiceRequestCause::MoSignalling => write!(f, "mo-signalling"),
            ServiceRequestCause::MtServices => write!(f, "mt-services"),
            ServiceRequestCause::EmergencyServices => write!(f, "emergency-services"),
            ServiceRequestCause::HighPriorityAccess => write!(f, "high-priority-access"),
        }
    }
}

// ============================================================================
// Error Type
// ============================================================================

/// Error type for the service request procedure.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ServiceRequestError {
    /// UE is not in RM-REGISTERED state
    #[error("UE is not registered (RM state: {0:?})")]
    NotRegistered(RmState),
    /// UE is not in CM-IDLE state (already connected)
    #[error("UE is already in CM-CONNECTED state")]
    AlreadyConnected,
    /// Service request already in progress
    #[error("Service request already in progress")]
    AlreadyInProgress,
    /// No 5G-S-TMSI available (needed for Service Request)
    #[error("No 5G-S-TMSI available for Service Request")]
    NoTmsi,
    /// Invalid state for service request
    #[error("Invalid state: {0:?}")]
    InvalidState(MmState),
}

// ============================================================================
// Service Request Accept Result
// ============================================================================

/// Result of processing a Service Accept message.
#[derive(Debug, Clone)]
pub struct ServiceAcceptResult {
    /// New MM sub-state after accept
    pub new_sub_state: MmSubState,
    /// New CM state after accept
    pub new_cm_state: CmState,
    /// PDU session status bitmask (if present, indicates active sessions)
    pub pdu_session_status: Option<u16>,
}

// ============================================================================
// Service Request Reject Result
// ============================================================================

/// Result of processing a Service Reject message.
#[derive(Debug, Clone)]
pub struct ServiceRejectResult {
    /// New MM sub-state after reject
    pub new_sub_state: MmSubState,
    /// New Update Status (if changed)
    pub new_update_status: Option<UpdateStatus>,
    /// Backoff timer value in seconds (from T3346 if present)
    pub backoff_secs: Option<u8>,
    /// Whether to clear stored GUTI
    pub clear_guti: bool,
}

// ============================================================================
// Procedure Handler
// ============================================================================

/// Manages the UE-side service request procedure.
///
/// Tracks procedure state, T3517 timer, and last request for retransmission.
#[derive(Debug)]
pub struct ServiceRequestProcedure {
    /// Timer T3517
    t3517: UeTimer,
    /// Last service request sent (for retransmission)
    last_request: Option<ServiceRequest>,
    /// Last cause (for context)
    last_cause: Option<ServiceRequestCause>,
}

impl Default for ServiceRequestProcedure {
    fn default() -> Self {
        Self::new()
    }
}

impl ServiceRequestProcedure {
    /// Creates a new service request procedure handler.
    pub fn new() -> Self {
        Self {
            t3517: UeTimer::new(T3517_CODE, true, T3517_DEFAULT_INTERVAL_SECS),
            last_request: None,
            last_cause: None,
        }
    }

    /// Returns a reference to the T3517 timer.
    pub fn t3517(&self) -> &UeTimer {
        &self.t3517
    }

    /// Returns a mutable reference to the T3517 timer.
    pub fn t3517_mut(&mut self) -> &mut UeTimer {
        &mut self.t3517
    }

    /// Returns the last service request cause.
    pub fn last_cause(&self) -> Option<ServiceRequestCause> {
        self.last_cause
    }

    // ========================================================================
    // Initiation
    // ========================================================================

    /// Initiates the service request procedure.
    ///
    /// Per 3GPP TS 24.501 Section 5.6.1.3:
    /// - UE must be RM-REGISTERED
    /// - UE must be CM-IDLE (otherwise no need for service request)
    /// - UE needs a 5G-S-TMSI to include in the request
    ///
    /// # Arguments
    /// * `cause` - Reason for the service request
    /// * `rm_state` - Current RM state
    /// * `cm_state` - Current CM state
    /// * `mm_state` - Current MM main state
    /// * `ng_ksi` - Current NAS key set identifier
    /// * `tmsi` - 5G-S-TMSI (mandatory for Service Request)
    /// * `uplink_data_status` - Bitmask of PDU sessions with pending UL data
    /// * `pdu_session_status` - Bitmask of active PDU sessions
    ///
    /// # Returns
    /// * `Ok((request, new_sub_state))` - Message to send + new sub-state
    /// * `Err(error)` - If procedure cannot be initiated
    #[allow(clippy::too_many_arguments)]
    pub fn initiate(
        &mut self,
        cause: ServiceRequestCause,
        rm_state: RmState,
        cm_state: CmState,
        mm_state: MmState,
        ng_ksi: NasKeySetIdentifier,
        tmsi: Ie5gsMobileIdentity,
        uplink_data_status: Option<u16>,
        pdu_session_status: Option<u16>,
    ) -> Result<(ServiceRequest, MmSubState), ServiceRequestError> {
        // Must be registered
        if rm_state != RmState::Registered {
            return Err(ServiceRequestError::NotRegistered(rm_state));
        }

        // Must be CM-IDLE (if already connected, no need to request)
        if cm_state == CmState::Connected {
            return Err(ServiceRequestError::AlreadyConnected);
        }

        // Cannot be already in service-request-initiated state
        if mm_state == MmState::ServiceRequestInitiated {
            return Err(ServiceRequestError::AlreadyInProgress);
        }

        let service_type = IeServiceType::new(cause.to_service_type());

        let mut request = ServiceRequest::new(ng_ksi, service_type, tmsi);
        request.uplink_data_status = uplink_data_status;
        request.pdu_session_status = pdu_session_status;

        // Store for potential retransmission
        self.last_request = Some(request.clone());
        self.last_cause = Some(cause);

        // Start T3517
        self.t3517.start(true);

        tracing::debug!(
            "Service Request initiated: cause={}, uplink_status={:?}",
            cause,
            uplink_data_status
        );

        Ok((request, MmSubState::ServiceRequestInitiated))
    }

    // ========================================================================
    // Response Handling
    // ========================================================================

    /// Handles reception of a Service Accept message.
    ///
    /// Per 3GPP TS 24.501 Section 5.6.1.4:
    /// - Stop T3517
    /// - Transition to REGISTERED.NORMAL-SERVICE / CM-CONNECTED
    /// - Reactivate PDU sessions per pdu_session_status if provided
    pub fn receive_service_accept(
        &mut self,
        msg: &ServiceAccept,
        mm_state: MmState,
    ) -> Result<ServiceAcceptResult, ServiceRequestError> {
        if mm_state != MmState::ServiceRequestInitiated {
            return Err(ServiceRequestError::InvalidState(mm_state));
        }

        // Stop timer
        self.t3517.stop(true);
        self.last_request = None;
        self.last_cause = None;

        tracing::debug!("Service Accept received — transitioning to CM-CONNECTED");

        Ok(ServiceAcceptResult {
            new_sub_state: MmSubState::RegisteredNormalService,
            new_cm_state: CmState::Connected,
            pdu_session_status: msg.pdu_session_status,
        })
    }

    /// Handles reception of a Service Reject message.
    ///
    /// Per 3GPP TS 24.501 Section 5.6.1.5:
    /// - Stop T3517
    /// - Determine new state based on 5GMM cause
    pub fn receive_service_reject(
        &mut self,
        msg: &ServiceReject,
        mm_state: MmState,
    ) -> Result<ServiceRejectResult, ServiceRequestError> {
        if mm_state != MmState::ServiceRequestInitiated {
            return Err(ServiceRequestError::InvalidState(mm_state));
        }

        // Stop timer
        self.t3517.stop(true);
        self.last_request = None;
        self.last_cause = None;

        let result = Self::process_reject_cause(msg.mm_cause.value, msg.t3346_value);

        tracing::warn!(
            "Service Reject received: cause={:?}, new_state={:?}",
            msg.mm_cause.value,
            result.new_sub_state
        );

        Ok(result)
    }

    // ========================================================================
    // Timer Handling
    // ========================================================================

    /// Handles T3517 expiry.
    ///
    /// Returns `Some(request)` if the request should be retransmitted,
    /// or `None` if max retransmissions have been reached (local release).
    pub fn handle_t3517_expiry(&mut self) -> Option<ServiceRequest> {
        if self.t3517.expiry_count() >= T3517_MAX_RETRANSMISSION {
            self.last_request = None;
            tracing::warn!("T3517 expired max times — aborting service request");
            return None;
        }

        if let Some(ref req) = self.last_request {
            self.t3517.start(false);
            tracing::debug!(
                "T3517 expired — retransmitting service request (attempt {})",
                self.t3517.expiry_count()
            );
            return Some(req.clone());
        }

        None
    }

    // ========================================================================
    // Helper Methods
    // ========================================================================

    /// Processes the 5GMM cause from a Service Reject and returns state changes.
    fn process_reject_cause(cause: MmCause, t3346: Option<u8>) -> ServiceRejectResult {
        match cause {
            MmCause::IllegalUe | MmCause::IllegalMe | MmCause::FiveGsServicesNotAllowed => {
                ServiceRejectResult {
                    new_sub_state: MmSubState::DeregisteredNormalService,
                    new_update_status: Some(UpdateStatus::RoamingNotAllowed),
                    backoff_secs: None,
                    clear_guti: true,
                }
            }
            MmCause::PlmnNotAllowed => ServiceRejectResult {
                new_sub_state: MmSubState::DeregisteredPlmnSearch,
                new_update_status: Some(UpdateStatus::RoamingNotAllowed),
                backoff_secs: None,
                clear_guti: true,
            },
            MmCause::TrackingAreaNotAllowed | MmCause::NoSuitableCellsInTa => ServiceRejectResult {
                new_sub_state: MmSubState::DeregisteredLimitedService,
                new_update_status: Some(UpdateStatus::RoamingNotAllowed),
                backoff_secs: None,
                clear_guti: false,
            },
            MmCause::Congestion => ServiceRejectResult {
                new_sub_state: MmSubState::RegisteredAttemptingRegistrationUpdate,
                new_update_status: Some(UpdateStatus::NotUpdated),
                backoff_secs: t3346,
                clear_guti: false,
            },
            MmCause::ImplicitlyDeregistered => ServiceRejectResult {
                new_sub_state: MmSubState::DeregisteredNormalService,
                new_update_status: None,
                backoff_secs: None,
                clear_guti: false,
            },
            MmCause::RestrictedServiceArea => ServiceRejectResult {
                new_sub_state: MmSubState::RegisteredLimitedService,
                new_update_status: None,
                backoff_secs: None,
                clear_guti: false,
            },
            // Other causes - stay registered, retry later
            _ => ServiceRejectResult {
                new_sub_state: MmSubState::RegisteredNormalService,
                new_update_status: Some(UpdateStatus::NotUpdated),
                backoff_secs: None,
                clear_guti: false,
            },
        }
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use nextgsim_nas::messages::mm::MobileIdentityType;

    fn make_tmsi() -> Ie5gsMobileIdentity {
        Ie5gsMobileIdentity::new(MobileIdentityType::Tmsi, vec![0xF4, 0x01, 0x02, 0x03, 0x04])
    }

    fn make_ng_ksi() -> NasKeySetIdentifier {
        use nextgsim_nas::security::SecurityContextType;
        NasKeySetIdentifier::new(SecurityContextType::Native, 1)
    }

    #[test]
    fn test_initiate_service_request_success() {
        let mut proc = ServiceRequestProcedure::new();
        let result = proc.initiate(
            ServiceRequestCause::MoData,
            RmState::Registered,
            CmState::Idle,
            MmState::Registered,
            make_ng_ksi(),
            make_tmsi(),
            Some(0x0001),
            None,
        );

        assert!(result.is_ok());
        let (req, new_state) = result.unwrap();
        assert_eq!(new_state, MmSubState::ServiceRequestInitiated);
        assert_eq!(req.service_type.service_type, ServiceType::Data);
        assert_eq!(req.uplink_data_status, Some(0x0001));
        assert!(proc.t3517.is_running());
    }

    #[test]
    fn test_initiate_service_request_not_registered() {
        let mut proc = ServiceRequestProcedure::new();
        let result = proc.initiate(
            ServiceRequestCause::MoSignalling,
            RmState::Deregistered,
            CmState::Idle,
            MmState::Deregistered,
            make_ng_ksi(),
            make_tmsi(),
            None,
            None,
        );

        assert!(matches!(result, Err(ServiceRequestError::NotRegistered(_))));
    }

    #[test]
    fn test_initiate_service_request_already_connected() {
        let mut proc = ServiceRequestProcedure::new();
        let result = proc.initiate(
            ServiceRequestCause::MoData,
            RmState::Registered,
            CmState::Connected, // Already connected
            MmState::Registered,
            make_ng_ksi(),
            make_tmsi(),
            None,
            None,
        );

        assert!(matches!(result, Err(ServiceRequestError::AlreadyConnected)));
    }

    #[test]
    fn test_initiate_service_request_already_in_progress() {
        let mut proc = ServiceRequestProcedure::new();
        let result = proc.initiate(
            ServiceRequestCause::MoData,
            RmState::Registered,
            CmState::Idle,
            MmState::ServiceRequestInitiated, // Already in progress
            make_ng_ksi(),
            make_tmsi(),
            None,
            None,
        );

        assert!(matches!(
            result,
            Err(ServiceRequestError::AlreadyInProgress)
        ));
    }

    #[test]
    fn test_receive_service_accept() {
        let mut proc = ServiceRequestProcedure::new();

        // First initiate
        let _ = proc.initiate(
            ServiceRequestCause::MoData,
            RmState::Registered,
            CmState::Idle,
            MmState::Registered,
            make_ng_ksi(),
            make_tmsi(),
            None,
            None,
        );

        let accept = ServiceAccept::new();
        let result = proc.receive_service_accept(&accept, MmState::ServiceRequestInitiated);

        assert!(result.is_ok());
        let accept_result = result.unwrap();
        assert_eq!(
            accept_result.new_sub_state,
            MmSubState::RegisteredNormalService
        );
        assert_eq!(accept_result.new_cm_state, CmState::Connected);
        assert!(!proc.t3517.is_running());
    }

    #[test]
    fn test_receive_service_accept_wrong_state() {
        let mut proc = ServiceRequestProcedure::new();
        let accept = ServiceAccept::new();
        let result = proc.receive_service_accept(&accept, MmState::Registered);

        assert!(matches!(result, Err(ServiceRequestError::InvalidState(_))));
    }

    #[test]
    fn test_receive_service_reject_congestion() {
        let mut proc = ServiceRequestProcedure::new();

        let _ = proc.initiate(
            ServiceRequestCause::MoData,
            RmState::Registered,
            CmState::Idle,
            MmState::Registered,
            make_ng_ksi(),
            make_tmsi(),
            None,
            None,
        );

        let mut reject = ServiceReject::new(MmCause::Congestion);
        reject.t3346_value = Some(0x1E); // 30 seconds

        let result = proc.receive_service_reject(&reject, MmState::ServiceRequestInitiated);
        assert!(result.is_ok());
        let reject_result = result.unwrap();
        assert_eq!(
            reject_result.new_sub_state,
            MmSubState::RegisteredAttemptingRegistrationUpdate
        );
        assert_eq!(reject_result.backoff_secs, Some(0x1E));
        assert!(!proc.t3517.is_running());
    }

    #[test]
    fn test_receive_service_reject_illegal_ue() {
        let mut proc = ServiceRequestProcedure::new();

        let _ = proc.initiate(
            ServiceRequestCause::MoData,
            RmState::Registered,
            CmState::Idle,
            MmState::Registered,
            make_ng_ksi(),
            make_tmsi(),
            None,
            None,
        );

        let reject = ServiceReject::new(MmCause::IllegalUe);
        let result = proc.receive_service_reject(&reject, MmState::ServiceRequestInitiated);
        assert!(result.is_ok());
        let reject_result = result.unwrap();
        assert_eq!(
            reject_result.new_sub_state,
            MmSubState::DeregisteredNormalService
        );
        assert_eq!(
            reject_result.new_update_status,
            Some(UpdateStatus::RoamingNotAllowed)
        );
        assert!(reject_result.clear_guti);
    }

    #[test]
    fn test_handle_t3517_expiry_retransmit() {
        let mut proc = ServiceRequestProcedure::new();

        let _ = proc.initiate(
            ServiceRequestCause::MoSignalling,
            RmState::Registered,
            CmState::Idle,
            MmState::Registered,
            make_ng_ksi(),
            make_tmsi(),
            None,
            None,
        );

        // Should retransmit before reaching max
        let retransmit = proc.handle_t3517_expiry();
        assert!(retransmit.is_some());
    }

    #[test]
    fn test_handle_t3517_expiry_no_request_returns_none() {
        // When no request is stored (e.g., after accept/reject clears it),
        // handle_t3517_expiry returns None.
        let mut proc = ServiceRequestProcedure::new();
        // No initiate called - no stored request
        let result = proc.handle_t3517_expiry();
        assert!(result.is_none());
    }

    #[test]
    fn test_handle_t3517_max_retransmission_constant() {
        // Verify the constant is sensible (matches 3GPP TS 24.501)
        assert_eq!(T3517_MAX_RETRANSMISSION, 5);
    }

    #[test]
    fn test_service_request_cause_to_service_type() {
        assert_eq!(
            ServiceRequestCause::MoData.to_service_type(),
            ServiceType::Data
        );
        assert_eq!(
            ServiceRequestCause::MoSignalling.to_service_type(),
            ServiceType::Signalling
        );
        assert_eq!(
            ServiceRequestCause::MtServices.to_service_type(),
            ServiceType::MobileTerminatedServices
        );
        assert_eq!(
            ServiceRequestCause::EmergencyServices.to_service_type(),
            ServiceType::EmergencyServices
        );
    }

    #[test]
    fn test_service_request_cause_display() {
        assert_eq!(format!("{}", ServiceRequestCause::MoData), "mo-data");
        assert_eq!(
            format!("{}", ServiceRequestCause::EmergencyServices),
            "emergency-services"
        );
    }
}
