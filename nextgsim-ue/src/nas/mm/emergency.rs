//! Emergency Registration Procedure
//!
//! This module implements the UE-side emergency registration procedure as defined in
//! 3GPP TS 24.501 Section 5.5.1.2.7 and 3GPP TS 24.501 Annex C.
//!
//! Emergency registration is a variant of the initial registration procedure with:
//! - Registration type set to `EmergencyRegistration` (0b100)
//! - No USIM required (can use IMEI as identity)
//! - Limited service - only emergency bearer services allowed
//!
//! # Key Differences from Normal Registration
//!
//! - `registration_type` is `RegistrationType::EmergencyRegistration`
//! - UE may send IMEI instead of SUCI if no USIM is present
//! - Network may respond with a restricted Registration Accept
//! - T3510 still applies as the registration timer
//!
//! # Reference
//!
//! Based on UERANSIM's `src/ue/nas/mm/access.cpp` emergency registration handling.

use thiserror::Error;

use nextgsim_nas::messages::mm::{
    Ie5gsMobileIdentity, MobileIdentityType, RegistrationRequest,
};
use nextgsim_nas::ies::ie1::{
    FollowOnRequest, Ie5gsRegistrationType, RegistrationType,
};
use nextgsim_nas::security::NasKeySetIdentifier;

use super::state::{MmState, MmSubState, RmState};

// ============================================================================
// Constants
// ============================================================================

/// Timer T3510 code (used for emergency registration too)
pub const T3510_CODE: u16 = 3510;

/// Default T3510 interval: 15 seconds per 3GPP TS 24.501 Table 10.2.1
pub const T3510_DEFAULT_INTERVAL_SECS: u32 = 15;

/// Maximum T3510 expiry count (retransmission limit)
pub const T3510_MAX_RETRANSMISSION: u32 = 5;

// ============================================================================
// Error Type
// ============================================================================

/// Error type for emergency registration procedure.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum EmergencyRegistrationError {
    /// UE is already registered
    #[error("UE is already registered (current state: {0:?})")]
    AlreadyRegistered(RmState),
    /// Registration already in progress
    #[error("Emergency registration already in progress")]
    AlreadyInProgress,
    /// Invalid state for emergency registration
    #[error("Invalid state for emergency registration: {0:?}")]
    InvalidState(MmState),
}

// ============================================================================
// Emergency Registration Procedure
// ============================================================================

/// Handles the emergency registration procedure for the UE.
///
/// Emergency registration follows the same flow as initial registration but
/// with `RegistrationType::EmergencyRegistration` and uses IMEI if no USIM.
#[derive(Debug)]
pub struct EmergencyRegistrationProcedure {
    /// Last request sent (for retransmission on T3510 expiry)
    last_request: Option<RegistrationRequest>,
}

impl Default for EmergencyRegistrationProcedure {
    fn default() -> Self {
        Self::new()
    }
}

impl EmergencyRegistrationProcedure {
    /// Creates a new emergency registration procedure handler.
    pub fn new() -> Self {
        Self {
            last_request: None,
        }
    }

    /// Returns the last emergency registration request (for retransmission).
    pub fn last_request(&self) -> Option<&RegistrationRequest> {
        self.last_request.as_ref()
    }

    // ========================================================================
    // Initiation
    // ========================================================================

    /// Initiates an emergency registration procedure.
    ///
    /// Builds a `RegistrationRequest` with `EmergencyRegistration` type.
    /// The UE uses IMEI as identity if no USIM is available, otherwise SUCI.
    ///
    /// # Arguments
    /// * `mm_state` - Current MM main state
    /// * `mm_sub_state` - Current MM sub-state
    /// * `ng_ksi` - NAS key set identifier (no-key if no security context)
    /// * `mobile_identity` - UE identity (SUCI or IMEI)
    /// * `follow_on` - Whether to indicate follow-on request pending
    ///
    /// # Returns
    /// * `Ok((request, new_sub_state))` - Request to send + new sub-state
    /// * `Err(error)` - If procedure cannot be initiated
    pub fn initiate(
        &mut self,
        mm_state: MmState,
        mm_sub_state: MmSubState,
        ng_ksi: Option<NasKeySetIdentifier>,
        mobile_identity: Ie5gsMobileIdentity,
        follow_on: bool,
    ) -> Result<(RegistrationRequest, MmSubState), EmergencyRegistrationError> {
        // Cannot initiate if already in REGISTERED-INITIATED state
        if mm_state == MmState::RegisteredInitiated {
            return Err(EmergencyRegistrationError::AlreadyInProgress);
        }

        // Emergency registration is allowed from DEREGISTERED substates only
        // (3GPP TS 24.501 Section 5.5.1.2.7)
        match mm_sub_state {
            MmSubState::DeregisteredNormalService
            | MmSubState::DeregisteredLimitedService
            | MmSubState::DeregisteredNoCellAvailable
            | MmSubState::DeregisteredAttemptingRegistration
            | MmSubState::DeregisteredPlmnSearch
            | MmSubState::DeregisteredEcallInactive
            | MmSubState::Deregistered => {
                // OK - proceed
            }
            _ => {
                return Err(EmergencyRegistrationError::InvalidState(mm_state));
            }
        }

        let follow_on_req = if follow_on {
            FollowOnRequest::Pending
        } else {
            FollowOnRequest::NoPending
        };

        let registration_type = Ie5gsRegistrationType::new(
            follow_on_req,
            RegistrationType::EmergencyRegistration,
        );

        let ksi = ng_ksi.unwrap_or_else(NasKeySetIdentifier::no_key);

        let request = RegistrationRequest::new(registration_type, ksi, mobile_identity);

        self.last_request = Some(request.clone());

        Ok((request, MmSubState::RegisteredInitiated))
    }

    /// Handles T3510 expiry for emergency registration.
    ///
    /// Returns `Some(request)` to retransmit, or `None` if max retries reached.
    pub fn handle_t3510_expiry(
        &self,
        expiry_count: u32,
    ) -> Option<&RegistrationRequest> {
        if expiry_count >= T3510_MAX_RETRANSMISSION {
            return None;
        }
        self.last_request.as_ref()
    }

    /// Clears the stored request (called after registration completes or fails).
    pub fn clear(&mut self) {
        self.last_request = None;
    }

    // ========================================================================
    // Helper
    // ========================================================================

    /// Creates an IMEI-based mobile identity for emergency registration when
    /// no USIM is present.
    ///
    /// The IMEI is BCD-encoded in the standard 5GS mobile identity format.
    /// Bytes: [identity_type | odd/even indicator, BCD digit pairs...]
    ///
    /// # Arguments
    /// * `imei_digits` - Exactly 15 decimal digits as ASCII bytes or raw digit values
    pub fn build_imei_identity(imei_digits: &[u8; 15]) -> Ie5gsMobileIdentity {
        // IMEI identity type byte: bits[2:0] = 0b011 (IMEI), bit[3] = odd indicator (1 for 15 digits)
        // Octet 1: identity type (IMEI=3) in bits 2-0, odd/even in bit 3
        let mut data = Vec::with_capacity(8);
        // First octet: odd length (bit3=1) + identity type IMEI (0b011)
        data.push(0b0000_1011u8); // odd=1, identity=IMEI(3)

        // Pack 15 BCD digits into 7 bytes + 1 filler nibble
        // Digits 1..8 pack into 4 bytes (digit1 low, digit2 high, etc.)
        // Digit 15 low nibble, filler 0xF high nibble
        let mut i = 0usize;
        while i + 1 < 15 {
            let lo = imei_digits[i] & 0x0F;
            let hi = imei_digits[i + 1] & 0x0F;
            data.push(lo | (hi << 4));
            i += 2;
        }
        // Last digit (15th) in low nibble, filler 0xF in high nibble
        data.push((imei_digits[14] & 0x0F) | 0xF0);

        Ie5gsMobileIdentity::new(MobileIdentityType::Imei, data)
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_suci_identity() -> Ie5gsMobileIdentity {
        // Minimal SUCI: identity_type byte + home network public key ID + ...
        Ie5gsMobileIdentity::new(
            MobileIdentityType::Suci,
            vec![0x01, 0x02, 0xF0, 0x10, 0x00, 0x00, 0x00],
        )
    }

    #[test]
    fn test_initiate_emergency_from_limited_service() {
        let mut proc = EmergencyRegistrationProcedure::new();
        let result = proc.initiate(
            MmState::Deregistered,
            MmSubState::DeregisteredLimitedService,
            None,
            make_suci_identity(),
            false,
        );

        assert!(result.is_ok());
        let (req, new_state) = result.unwrap();
        assert_eq!(new_state, MmSubState::RegisteredInitiated);
        assert_eq!(
            req.registration_type.registration_type,
            RegistrationType::EmergencyRegistration
        );
        assert_eq!(
            req.registration_type.follow_on_request_pending,
            FollowOnRequest::NoPending
        );
    }

    #[test]
    fn test_initiate_emergency_from_normal_service() {
        let mut proc = EmergencyRegistrationProcedure::new();
        let result = proc.initiate(
            MmState::Deregistered,
            MmSubState::DeregisteredNormalService,
            None,
            make_suci_identity(),
            true, // follow-on
        );

        assert!(result.is_ok());
        let (req, _) = result.unwrap();
        assert_eq!(
            req.registration_type.registration_type,
            RegistrationType::EmergencyRegistration
        );
        assert_eq!(
            req.registration_type.follow_on_request_pending,
            FollowOnRequest::Pending
        );
    }

    #[test]
    fn test_initiate_emergency_already_in_progress() {
        let mut proc = EmergencyRegistrationProcedure::new();
        let result = proc.initiate(
            MmState::RegisteredInitiated,
            MmSubState::RegisteredInitiated,
            None,
            make_suci_identity(),
            false,
        );

        assert!(matches!(
            result,
            Err(EmergencyRegistrationError::AlreadyInProgress)
        ));
    }

    #[test]
    fn test_initiate_emergency_invalid_state() {
        let mut proc = EmergencyRegistrationProcedure::new();
        // Cannot start emergency reg from REGISTERED state
        let result = proc.initiate(
            MmState::Registered,
            MmSubState::RegisteredNormalService,
            None,
            make_suci_identity(),
            false,
        );

        assert!(matches!(
            result,
            Err(EmergencyRegistrationError::InvalidState(_))
        ));
    }

    #[test]
    fn test_handle_t3510_expiry_retransmit() {
        let mut proc = EmergencyRegistrationProcedure::new();
        let _ = proc.initiate(
            MmState::Deregistered,
            MmSubState::DeregisteredNormalService,
            None,
            make_suci_identity(),
            false,
        );

        // Should retransmit for counts 0..4
        for count in 0..T3510_MAX_RETRANSMISSION {
            assert!(proc.handle_t3510_expiry(count).is_some());
        }

        // Should give up at max
        assert!(proc.handle_t3510_expiry(T3510_MAX_RETRANSMISSION).is_none());
    }

    #[test]
    fn test_handle_t3510_expiry_no_request() {
        let proc = EmergencyRegistrationProcedure::new();
        // No request stored
        assert!(proc.handle_t3510_expiry(0).is_none());
    }

    #[test]
    fn test_clear_resets_state() {
        let mut proc = EmergencyRegistrationProcedure::new();
        let _ = proc.initiate(
            MmState::Deregistered,
            MmSubState::DeregisteredLimitedService,
            None,
            make_suci_identity(),
            false,
        );
        assert!(proc.last_request().is_some());

        proc.clear();
        assert!(proc.last_request().is_none());
    }

    #[test]
    fn test_build_imei_identity() {
        // 15-digit IMEI: 356938035643809
        let digits: [u8; 15] = [3, 5, 6, 9, 3, 8, 0, 3, 5, 6, 4, 3, 8, 0, 9];
        let identity = EmergencyRegistrationProcedure::build_imei_identity(&digits);
        assert_eq!(identity.identity_type, MobileIdentityType::Imei);
        // First octet: odd indicator set, IMEI type
        assert_eq!(identity.data[0], 0b0000_1011);
        // Data: 1 type byte + 7 pairs (digits 1-14) + 1 byte (digit 15 + filler 0xF) = 9 bytes
        assert_eq!(identity.data.len(), 9);
    }

    #[test]
    fn test_emergency_registration_uses_no_key_when_no_ksi() {
        let mut proc = EmergencyRegistrationProcedure::new();
        let result = proc.initiate(
            MmState::Deregistered,
            MmSubState::DeregisteredNormalService,
            None,
            make_suci_identity(),
            false,
        );
        let (req, _) = result.unwrap();
        // No KSI means no_key (KSI=7)
        assert_eq!(req.ng_ksi.ksi, 7);
    }
}
