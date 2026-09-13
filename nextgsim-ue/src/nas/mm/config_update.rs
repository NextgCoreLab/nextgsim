//! Configuration Update Command Procedure
//!
//! This module implements the UE-side handling of the Configuration Update Command
//! procedure as defined in 3GPP TS 24.501 Section 5.4.4.
//!
//! # Overview
//!
//! The AMF may send a `ConfigurationUpdateCommand` to the UE at any time while the
//! UE is in CM-CONNECTED state to update:
//! - GUTI (5G-GUTI reallocation)
//! - TAI list
//! - Allowed NSSAI
//! - Configured NSSAI
//! - Service area list
//! - Full name for network
//! - Short name for network
//! - T3512 value
//! - Operator-defined access category definitions
//!
//! # Acknowledgement
//!
//! When the `Configuration Update Indication` IE has the ACK bit set, the UE shall
//! send a `ConfigurationUpdateComplete` message in response.
//!
//! When the RED (Registration Requested) bit is set, the UE should initiate a
//! registration procedure after processing the command.
//!
//! # Reference
//!
//! Based on UERANSIM's `src/ue/nas/mm/config.cpp` implementation.

use bytes::{Buf, BufMut};
use thiserror::Error;

use nextgsim_nas::enums::MmMessageType;
use nextgsim_nas::header::PlainMmHeader;
use nextgsim_nas::ies::ie1::{
    Acknowledgement, IeConfigurationUpdateIndication, InformationElement1, RegistrationRequested,
};
use nextgsim_nas::messages::mm::Ie5gsMobileIdentity;

use super::state::MmSubState;
use crate::timer::GprsTimer3;

// ============================================================================
// IEI constants for ConfigurationUpdateCommand optional IEs
// (3GPP TS 24.501 Section 8.2.4)
// ============================================================================

/// IEI values for Configuration Update Command optional IEs
mod config_update_iei {
    /// Configuration update indication (Type 1, IEI 0xD)
    pub const CONFIG_UPDATE_INDICATION: u8 = 0xD0;
    /// 5G-GUTI (Type 6, IEI 0x77)
    pub const GUTI: u8 = 0x77;
    /// TAI list (Type 4, IEI 0x54)
    pub const TAI_LIST: u8 = 0x54;
    /// Allowed NSSAI (Type 4, IEI 0x15)
    pub const ALLOWED_NSSAI: u8 = 0x15;
    /// Service area list (Type 4, IEI 0x27)
    pub const SERVICE_AREA_LIST: u8 = 0x27;
    /// Full name for network (Type 4, IEI 0x43)
    pub const FULL_NAME_FOR_NETWORK: u8 = 0x43;
    /// Short name for network (Type 4, IEI 0x45)
    pub const SHORT_NAME_FOR_NETWORK: u8 = 0x45;
    /// T3512 value (Type 4, IEI 0x5E)
    pub const T3512_VALUE: u8 = 0x5E;
    /// Operator-defined access category definitions (Type 6, IEI 0x76)
    pub const OPERATOR_DEFINED_ACCESS_CATEGORY: u8 = 0x76;
    /// Configured NSSAI (Type 4, IEI 0x31)
    pub const CONFIGURED_NSSAI: u8 = 0x31;
    /// Rejected NSSAI (Type 4, IEI 0x11)
    pub const REJECTED_NSSAI: u8 = 0x11;
    /// UE radio capability ID (Type 4, IEI 0x67) — RACS, TS 24.501 §9.11.3.68
    pub const UE_RADIO_CAPABILITY_ID: u8 = 0x67;
    /// UE radio capability ID deletion indication (Type 1, IEI 0xA)
    /// — TS 24.501 §9.11.3.69
    pub const UE_RADIO_CAPABILITY_ID_DELETION: u8 = 0xA0;
}

/// UE radio capability ID deletion indication (TS 24.501 §9.11.3.69).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RacsDeletionRequest {
    /// Bits 3-1 = 000: no deletion requested
    NotRequested,
    /// Bits 3-1 = 001: delete the network-assigned UE radio capability IDs
    NetworkAssigned,
    /// A value the spec reserves. Reported rather than treated as a deletion,
    /// because acting on an unknown code would delete state the network may not
    /// have asked to lose.
    Reserved(u8),
}

impl RacsDeletionRequest {
    /// Decodes the 3-bit deletion-request field of the Type 1 IE.
    pub fn from_bits(bits: u8) -> Self {
        match bits & 0x07 {
            0 => Self::NotRequested,
            1 => Self::NetworkAssigned,
            other => Self::Reserved(other),
        }
    }

    /// Whether the UE must delete its network-assigned UE radio capability IDs.
    pub fn deletes_network_assigned(&self) -> bool {
        matches!(self, Self::NetworkAssigned)
    }
}

// ============================================================================
// ConfigurationUpdateCommand Message
// ============================================================================

/// Configuration Update Command message (network to UE).
///
/// 3GPP TS 24.501 Section 8.2.4
#[derive(Debug, Clone, Default)]
pub struct ConfigurationUpdateCommand {
    /// Configuration update indication (optional, Type 1, IEI 0xD)
    /// Contains ACK bit (acknowledgement required) and RED bit (re-registration required).
    pub config_update_indication: Option<IeConfigurationUpdateIndication>,
    /// New 5G-GUTI (optional, Type 6, IEI 0x77)
    pub guti: Option<Ie5gsMobileIdentity>,
    /// TAI list (optional, Type 4, IEI 0x54)
    pub tai_list: Option<Vec<u8>>,
    /// Allowed NSSAI (optional, Type 4, IEI 0x15)
    pub allowed_nssai: Option<Vec<u8>>,
    /// Service area list (optional, Type 4, IEI 0x27)
    pub service_area_list: Option<Vec<u8>>,
    /// Full name for network (optional, Type 4, IEI 0x43)
    pub full_name_for_network: Option<Vec<u8>>,
    /// Short name for network (optional, Type 4, IEI 0x45)
    pub short_name_for_network: Option<Vec<u8>>,
    /// T3512 value in seconds (optional, decoded from GPRS Timer 3 IE).
    ///
    /// `Some(0)` means the network deactivated the timer, matching
    /// [`GprsTimer3::to_seconds`] and the REGISTRATION ACCEPT path.
    pub t3512_value_secs: Option<u32>,
    /// Configured NSSAI (optional, Type 4, IEI 0x31)
    pub configured_nssai: Option<Vec<u8>>,
    /// Rejected NSSAI (optional, Type 4, IEI 0x11) — TS 24.501 §9.11.3.46
    pub rejected_nssai: Option<Vec<u8>>,
    /// Network-assigned UE radio capability ID (optional, Type 4, IEI 0x67).
    ///
    /// Decoded to its hexadecimal-digit string: the IE packs each digit into a
    /// nibble, LOW nibble first (TS 24.501 §9.11.3.68), so the octets are not
    /// the ID.
    pub ue_radio_capability_id: Option<String>,
    /// UE radio capability ID deletion indication (optional, Type 1, IEI 0xA)
    pub ue_radio_capability_id_deletion: Option<RacsDeletionRequest>,
}

/// Error type for Configuration Update message encoding/decoding.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ConfigUpdateError {
    /// Buffer too short for decoding
    #[error("Buffer too short: expected {expected} bytes, got {actual}")]
    BufferTooShort {
        /// Expected minimum bytes
        expected: usize,
        /// Actual bytes available
        actual: usize,
    },
    /// Invalid IE value
    #[error("Invalid IE value: {0}")]
    InvalidIeValue(String),
}

impl ConfigurationUpdateCommand {
    /// Create a new (empty) Configuration Update Command.
    pub fn new() -> Self {
        Self::default()
    }

    /// Decode from bytes (after the 3-byte NAS header has been stripped).
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, ConfigUpdateError> {
        let mut msg = Self::new();

        while buf.remaining() > 0 {
            let iei = buf.chunk()[0];

            // Type 1 IEs have IEI in the high nibble; 0xD0 mask covers config update indication
            if iei & 0xF0 == config_update_iei::CONFIG_UPDATE_INDICATION {
                buf.advance(1);
                let indication = IeConfigurationUpdateIndication::decode(iei & 0x0F)
                    .map_err(|e| ConfigUpdateError::InvalidIeValue(e.to_string()))?;
                msg.config_update_indication = Some(indication);
                continue;
            }

            // Type 1 IE: UE radio capability ID deletion indication (IEI 0xA in
            // the high nibble, deletion request in bits 3-1).
            if iei & 0xF0 == config_update_iei::UE_RADIO_CAPABILITY_ID_DELETION {
                buf.advance(1);
                msg.ue_radio_capability_id_deletion =
                    Some(RacsDeletionRequest::from_bits(iei & 0x07));
                continue;
            }

            // Type 4 / Type 6 IEs
            match iei {
                config_update_iei::GUTI => {
                    buf.advance(1);
                    if buf.remaining() < 2 {
                        break;
                    }
                    let len = buf.get_u16() as usize;
                    if buf.remaining() < len {
                        break;
                    }
                    // Reconstruct 2-byte length + data for Ie5gsMobileIdentity::decode
                    let data: Vec<u8> = buf.copy_to_bytes(len).to_vec();
                    // Reconstruct the LV buffer so Ie5gsMobileIdentity::decode works
                    let mut lv_buf = Vec::with_capacity(2 + len);
                    lv_buf.extend_from_slice(&(len as u16).to_be_bytes());
                    lv_buf.extend_from_slice(&data);
                    if let Ok(guti) = Ie5gsMobileIdentity::decode(&mut lv_buf.as_slice()) {
                        msg.guti = Some(guti);
                    }
                }
                config_update_iei::TAI_LIST => {
                    buf.advance(1);
                    if buf.remaining() < 1 {
                        break;
                    }
                    let len = buf.get_u8() as usize;
                    if buf.remaining() < len {
                        break;
                    }
                    msg.tai_list = Some(buf.copy_to_bytes(len).to_vec());
                }
                config_update_iei::ALLOWED_NSSAI => {
                    buf.advance(1);
                    if buf.remaining() < 1 {
                        break;
                    }
                    let len = buf.get_u8() as usize;
                    if buf.remaining() < len {
                        break;
                    }
                    msg.allowed_nssai = Some(buf.copy_to_bytes(len).to_vec());
                }
                config_update_iei::SERVICE_AREA_LIST => {
                    buf.advance(1);
                    if buf.remaining() < 1 {
                        break;
                    }
                    let len = buf.get_u8() as usize;
                    if buf.remaining() < len {
                        break;
                    }
                    msg.service_area_list = Some(buf.copy_to_bytes(len).to_vec());
                }
                config_update_iei::FULL_NAME_FOR_NETWORK => {
                    buf.advance(1);
                    if buf.remaining() < 1 {
                        break;
                    }
                    let len = buf.get_u8() as usize;
                    if buf.remaining() < len {
                        break;
                    }
                    msg.full_name_for_network = Some(buf.copy_to_bytes(len).to_vec());
                }
                config_update_iei::SHORT_NAME_FOR_NETWORK => {
                    buf.advance(1);
                    if buf.remaining() < 1 {
                        break;
                    }
                    let len = buf.get_u8() as usize;
                    if buf.remaining() < len {
                        break;
                    }
                    msg.short_name_for_network = Some(buf.copy_to_bytes(len).to_vec());
                }
                config_update_iei::T3512_VALUE => {
                    buf.advance(1);
                    if buf.remaining() < 1 {
                        break;
                    }
                    let len = buf.get_u8() as usize;
                    if buf.remaining() < len || len < 1 {
                        break;
                    }
                    let gprs_timer_byte = buf.get_u8();
                    if len > 1 {
                        buf.advance(len - 1);
                    }
                    msg.t3512_value_secs = Some(decode_gprs_timer3(gprs_timer_byte));
                }
                config_update_iei::CONFIGURED_NSSAI => {
                    buf.advance(1);
                    if buf.remaining() < 1 {
                        break;
                    }
                    let len = buf.get_u8() as usize;
                    if buf.remaining() < len {
                        break;
                    }
                    msg.configured_nssai = Some(buf.copy_to_bytes(len).to_vec());
                }
                config_update_iei::REJECTED_NSSAI => {
                    buf.advance(1);
                    if buf.remaining() < 1 {
                        break;
                    }
                    let len = buf.get_u8() as usize;
                    if buf.remaining() < len {
                        break;
                    }
                    msg.rejected_nssai = Some(buf.copy_to_bytes(len).to_vec());
                }
                config_update_iei::UE_RADIO_CAPABILITY_ID => {
                    buf.advance(1);
                    if buf.remaining() < 1 {
                        break;
                    }
                    let len = buf.get_u8() as usize;
                    if buf.remaining() < len {
                        break;
                    }
                    let octets = buf.copy_to_bytes(len).to_vec();
                    msg.ue_radio_capability_id = Some(decode_racs_id(&octets));
                }
                config_update_iei::OPERATOR_DEFINED_ACCESS_CATEGORY => {
                    buf.advance(1);
                    if buf.remaining() < 2 {
                        break;
                    }
                    let len = buf.get_u16() as usize;
                    if buf.remaining() < len {
                        break;
                    }
                    buf.advance(len); // Not stored — parsed but ignored in happy path
                }
                _ => {
                    // Unknown IE: skip (try to consume length)
                    buf.advance(1);
                    if buf.remaining() > 0 {
                        let len = buf.get_u8() as usize;
                        if buf.remaining() >= len {
                            buf.advance(len);
                        }
                    }
                }
            }
        }

        Ok(msg)
    }

    /// Returns `true` if the network requires the UE to send a
    /// `ConfigurationUpdateComplete` in response.
    pub fn acknowledgement_required(&self) -> bool {
        self.config_update_indication
            .as_ref()
            .is_some_and(|ind| ind.ack == Acknowledgement::Requested)
    }

    /// Returns `true` if the network requests the UE to initiate a new
    /// registration procedure (RED bit set).
    pub fn registration_requested(&self) -> bool {
        self.config_update_indication
            .as_ref()
            .is_some_and(|ind| ind.red == RegistrationRequested::Requested)
    }
}

// ============================================================================
// ConfigurationUpdateComplete Message
// ============================================================================

/// Configuration Update Complete message (UE to network).
///
/// 3GPP TS 24.501 Section 8.2.5
/// Sent by UE in response to `ConfigurationUpdateCommand` when ACK bit is set.
#[derive(Debug, Clone, Default)]
pub struct ConfigurationUpdateComplete;

impl ConfigurationUpdateComplete {
    /// Create a new Configuration Update Complete message.
    pub fn new() -> Self {
        Self
    }

    /// Encode to bytes (including the 3-byte NAS header).
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        let header = PlainMmHeader::new(MmMessageType::ConfigurationUpdateComplete);
        header.encode(buf);
        // No IEs in this message body
    }

    /// Get the message type.
    pub fn message_type() -> MmMessageType {
        MmMessageType::ConfigurationUpdateComplete
    }
}

// ============================================================================
// Procedure Handler
// ============================================================================

/// Result of processing a Configuration Update Command.
#[derive(Debug, Clone)]
pub struct ConfigUpdateResult {
    /// New GUTI to store (if provided by network)
    pub new_guti: Option<Ie5gsMobileIdentity>,
    /// New TAI list bytes (if provided)
    pub new_tai_list: Option<Vec<u8>>,
    /// New allowed NSSAI bytes (if provided)
    pub new_allowed_nssai: Option<Vec<u8>>,
    /// New T3512 value in seconds (if provided); 0 means deactivated
    pub new_t3512_secs: Option<u32>,
    /// New configured NSSAI bytes (if provided)
    pub new_configured_nssai: Option<Vec<u8>>,
    /// Rejected NSSAI bytes (if provided) — TS 24.501 §9.11.3.46
    pub new_rejected_nssai: Option<Vec<u8>>,
    /// Network-assigned UE radio capability ID (if provided), as its
    /// hexadecimal-digit string
    pub new_racs_id: Option<String>,
    /// UE radio capability ID deletion request (if the IE was present)
    pub racs_deletion: Option<RacsDeletionRequest>,
    /// Whether UE must send ConfigurationUpdateComplete
    pub send_complete: bool,
    /// Whether UE must initiate a new registration procedure
    pub re_register: bool,
    /// New MM sub-state to transition to (if any)
    pub new_sub_state: Option<MmSubState>,
}

/// Handles the Configuration Update Command procedure.
///
/// Call `process_command` when `ConfigurationUpdateCommand` is received.
/// The result tells the caller what actions to take.
pub struct ConfigUpdateProcedure;

impl ConfigUpdateProcedure {
    /// Processes a received Configuration Update Command.
    ///
    /// Per 3GPP TS 24.501 Section 5.4.4.2:
    /// 1. Store any updated parameters (GUTI, TAI list, NSSAI, …)
    /// 2. If ACK bit set, send ConfigurationUpdateComplete
    /// 3. If RED bit set, initiate registration after sending complete
    pub fn process_command(cmd: &ConfigurationUpdateCommand) -> ConfigUpdateResult {
        let send_complete = cmd.acknowledgement_required();
        let re_register = cmd.registration_requested();

        // When re-registration is needed the UE should enter
        // REGISTERED.UPDATE-NEEDED substate until registration completes.
        let new_sub_state = if re_register {
            Some(MmSubState::RegisteredUpdateNeeded)
        } else {
            None
        };

        ConfigUpdateResult {
            new_guti: cmd.guti.clone(),
            new_tai_list: cmd.tai_list.clone(),
            new_allowed_nssai: cmd.allowed_nssai.clone(),
            new_t3512_secs: cmd.t3512_value_secs,
            new_configured_nssai: cmd.configured_nssai.clone(),
            new_rejected_nssai: cmd.rejected_nssai.clone(),
            new_racs_id: cmd.ue_radio_capability_id.clone(),
            racs_deletion: cmd.ue_radio_capability_id_deletion,
            send_complete,
            re_register,
            new_sub_state,
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Decodes a GPRS Timer 3 IE byte into a duration in seconds.
///
/// Delegates to [`GprsTimer3`], which carries the TS 24.008 §10.5.7.4a unit
/// table (`000` = 10 minutes, `001` = 1 hour, `010` = 10 hours, `011` = 2
/// seconds, `100` = 30 seconds, `101` = 1 minute, `110` = 320 hours, `111` =
/// deactivated → 0 seconds). This module previously carried its own table with
/// the units in ascending order, so every unit but 320 hours decoded to the
/// wrong duration — a T3512 of `0x49` (unit `010`, value 9: 90 hours, which is
/// what the core sends) came out as 540 seconds. One decoder, and it is the one
/// the REGISTRATION ACCEPT path already uses.
fn decode_gprs_timer3(byte: u8) -> u32 {
    GprsTimer3::from_byte(byte).to_seconds()
}

/// Read the UE radio capability ID out of the IE octets (TS 24.501 §9.11.3.68).
///
/// Delegates to `nextgsim-nas`, which owns both halves of the packing: the UE
/// decodes the ID a CONFIGURATION UPDATE COMMAND assigns and encodes the same ID
/// back into a REGISTRATION REQUEST, so a divergence between the two would be
/// invisible until an interop run (issue #101).
fn decode_racs_id(octets: &[u8]) -> String {
    nextgsim_nas::messages::mm::decode_ue_radio_capability_id(octets)
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use nextgsim_nas::ies::ie1::{Acknowledgement, RegistrationRequested};

    fn build_config_update_command_bytes(ack: bool, red: bool, include_tai: bool) -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::new();

        // Type 1 IE for config update indication (IEI 0xD in high nibble)
        let ack_bit: u8 = if ack { 1 } else { 0 };
        let red_bit: u8 = if red { 1 } else { 0 };
        let indication_byte = 0xD0 | ack_bit | (red_bit << 1);
        buf.push(indication_byte);

        if include_tai {
            // TAI list IE: IEI 0x54, length 6, dummy TAI bytes
            buf.push(config_update_iei::TAI_LIST);
            buf.push(6); // length
            buf.extend_from_slice(&[0x00, 0x01, 0xF1, 0x10, 0x00, 0x01]);
        }

        buf
    }

    #[test]
    fn test_decode_config_update_command_ack_set() {
        let bytes = build_config_update_command_bytes(true, false, false);
        let cmd = ConfigurationUpdateCommand::decode(&mut bytes.as_slice()).unwrap();
        assert!(cmd.acknowledgement_required());
        assert!(!cmd.registration_requested());
    }

    #[test]
    fn test_decode_config_update_command_red_set() {
        let bytes = build_config_update_command_bytes(false, true, false);
        let cmd = ConfigurationUpdateCommand::decode(&mut bytes.as_slice()).unwrap();
        assert!(!cmd.acknowledgement_required());
        assert!(cmd.registration_requested());
    }

    #[test]
    fn test_decode_config_update_command_both_bits() {
        let bytes = build_config_update_command_bytes(true, true, false);
        let cmd = ConfigurationUpdateCommand::decode(&mut bytes.as_slice()).unwrap();
        assert!(cmd.acknowledgement_required());
        assert!(cmd.registration_requested());
    }

    #[test]
    fn test_decode_config_update_command_with_tai_list() {
        let bytes = build_config_update_command_bytes(true, false, true);
        let cmd = ConfigurationUpdateCommand::decode(&mut bytes.as_slice()).unwrap();
        assert!(cmd.tai_list.is_some());
        assert_eq!(cmd.tai_list.unwrap().len(), 6);
    }

    #[test]
    fn test_decode_empty_command() {
        let bytes: Vec<u8> = vec![];
        let cmd = ConfigurationUpdateCommand::decode(&mut bytes.as_slice()).unwrap();
        assert!(cmd.config_update_indication.is_none());
        assert!(cmd.guti.is_none());
        assert!(cmd.tai_list.is_none());
        assert!(!cmd.acknowledgement_required());
    }

    #[test]
    fn test_process_command_ack_required() {
        let mut cmd = ConfigurationUpdateCommand::new();
        cmd.config_update_indication = Some(IeConfigurationUpdateIndication::new(
            Acknowledgement::Requested,
            RegistrationRequested::NotRequested,
        ));

        let result = ConfigUpdateProcedure::process_command(&cmd);
        assert!(result.send_complete);
        assert!(!result.re_register);
        assert!(result.new_sub_state.is_none());
    }

    #[test]
    fn test_process_command_re_register_required() {
        let mut cmd = ConfigurationUpdateCommand::new();
        cmd.config_update_indication = Some(IeConfigurationUpdateIndication::new(
            Acknowledgement::NotRequested,
            RegistrationRequested::Requested,
        ));

        let result = ConfigUpdateProcedure::process_command(&cmd);
        assert!(!result.send_complete);
        assert!(result.re_register);
        assert_eq!(
            result.new_sub_state,
            Some(MmSubState::RegisteredUpdateNeeded)
        );
    }

    #[test]
    fn test_process_command_with_guti() {
        use nextgsim_nas::messages::mm::MobileIdentityType;
        let mut cmd = ConfigurationUpdateCommand::new();
        cmd.guti = Some(Ie5gsMobileIdentity::new(
            MobileIdentityType::Guti,
            vec![
                0x02, 0xF8, 0x39, 0xCA, 0xFE, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
            ],
        ));

        let result = ConfigUpdateProcedure::process_command(&cmd);
        assert!(result.new_guti.is_some());
    }

    #[test]
    fn test_configuration_update_complete_encode() {
        let complete = ConfigurationUpdateComplete::new();
        let mut buf = Vec::new();
        complete.encode(&mut buf);

        // Should encode: EPD (0x7E) + Security Header (0x00) + MsgType (0x55)
        assert_eq!(buf.len(), 3);
        assert_eq!(buf[0], 0x7E); // EPD: Mobility Management
        assert_eq!(buf[1], 0x00); // Plain NAS
        assert_eq!(buf[2], MmMessageType::ConfigurationUpdateComplete as u8);
    }

    /// The GPRS Timer 3 unit field is NOT an ascending list of durations
    /// (TS 24.008 §10.5.7.4a table 10.5.163a): `000` is 10 minutes, `011` is 2
    /// seconds. The previous version of this test pinned an ascending table
    /// (`000` = 2 s, `001` = 30 s, `010` = 1 min, …), which is the defect it was
    /// written against: it made a T3512 of `0x49` — unit `010`, value 9, i.e.
    /// the 90 hours the core actually sends — decode as 540 seconds.
    #[test]
    fn a_gprs_timer3_byte_decodes_with_the_ts_24_008_unit_table() {
        // Unit 000 = multiples of 10 minutes
        assert_eq!(decode_gprs_timer3(0b000_00101), 5 * 600);
        // Unit 001 = multiples of 1 hour
        assert_eq!(decode_gprs_timer3(0b001_00110), 6 * 3600);
        // Unit 010 = multiples of 10 hours: 0x49 is what the core sends for T3512
        assert_eq!(decode_gprs_timer3(0x49), 9 * 10 * 3600);
        // Unit 011 = multiples of 2 seconds
        assert_eq!(decode_gprs_timer3(0b011_00101), 10);
        // Unit 100 = multiples of 30 seconds
        assert_eq!(decode_gprs_timer3(0b100_00110), 180);
        // Unit 101 = multiples of 1 minute
        assert_eq!(decode_gprs_timer3(0b101_00011), 180);
        // Unit 110 = multiples of 320 hours
        assert_eq!(decode_gprs_timer3(0b110_00001), 320 * 3600);
        // Unit 111 = deactivated, reported as 0 seconds like every other caller
        // of GprsTimer3 sees it
        assert_eq!(decode_gprs_timer3(0b111_00001), 0);
    }

    /// The T3512 IE of a Configuration Update Command goes through the same
    /// decoder, so a command carrying the core's own byte yields the same value
    /// a Registration Accept would.
    #[test]
    fn a_config_update_t3512_ie_decodes_to_the_same_value_as_a_registration_accept() {
        let bytes = vec![0xD1, config_update_iei::T3512_VALUE, 0x01, 0x49];
        let cmd = ConfigurationUpdateCommand::decode(&mut bytes.as_slice()).unwrap();
        assert_eq!(cmd.t3512_value_secs, Some(9 * 10 * 3600));
        assert_eq!(
            cmd.t3512_value_secs,
            Some(GprsTimer3::from_byte(0x49).to_seconds())
        );
    }

    // ========================================================================
    // RACS: UE radio capability ID (#19)
    // ========================================================================

    /// TS 24.501 §9.11.3.68 packs the ID's hexadecimal digits one per nibble,
    /// LOW nibble first, so reading the octets as-is reverses every digit pair.
    #[test]
    fn a_ue_radio_capability_id_decodes_low_nibble_first() {
        // Digits "123456": 0x21, 0x43, 0x65
        assert_eq!(decode_racs_id(&[0x21, 0x43, 0x65]), "123456");
    }

    /// An odd digit count fills the last high nibble with 1111, which is not
    /// part of the ID.
    #[test]
    fn an_odd_length_ue_radio_capability_id_drops_the_filler_nibble() {
        // Digits "abc": 0xBA, 0xFC
        assert_eq!(decode_racs_id(&[0xBA, 0xFC]), "abc");
    }

    /// An `f` that is a real digit rather than the filler must survive: `1111`
    /// is only a filler in the HIGH nibble of the LAST octet.
    #[test]
    fn an_f_digit_inside_the_id_is_not_mistaken_for_the_filler() {
        // Digits "afb" (odd): 'a' then 'f' in the first octet, 'b' plus the
        // filler in the second. The first octet's high nibble is also 1111, so a
        // decoder that checks for the filler in every octet loses the real 'f'.
        assert_eq!(decode_racs_id(&[0xFA, 0xFB]), "afb");
    }

    #[test]
    fn a_command_carrying_a_ue_radio_capability_id_decodes_it() {
        let mut bytes = vec![0xD1]; // config update indication, ACK requested
        bytes.push(config_update_iei::UE_RADIO_CAPABILITY_ID);
        bytes.push(3); // length
        bytes.extend_from_slice(&[0x21, 0x43, 0x65]);

        let cmd = ConfigurationUpdateCommand::decode(&mut bytes.as_slice()).unwrap();
        assert_eq!(cmd.ue_radio_capability_id.as_deref(), Some("123456"));
        assert!(
            cmd.acknowledgement_required(),
            "the ID must not consume the indication that precedes it"
        );
    }

    #[test]
    fn a_ue_radio_capability_id_deletion_indication_decodes() {
        // Type 1 IE: IEI 0xA in the high nibble, deletion request 001
        let bytes = vec![0xA1];
        let cmd = ConfigurationUpdateCommand::decode(&mut bytes.as_slice()).unwrap();
        assert_eq!(
            cmd.ue_radio_capability_id_deletion,
            Some(RacsDeletionRequest::NetworkAssigned)
        );
        assert!(cmd
            .ue_radio_capability_id_deletion
            .unwrap()
            .deletes_network_assigned());
    }

    /// Deletion request 000 is "not requested": present in the message, but it
    /// must not delete anything.
    #[test]
    fn a_zero_deletion_request_deletes_nothing() {
        let bytes = vec![0xA0];
        let cmd = ConfigurationUpdateCommand::decode(&mut bytes.as_slice()).unwrap();
        assert_eq!(
            cmd.ue_radio_capability_id_deletion,
            Some(RacsDeletionRequest::NotRequested)
        );
        assert!(!cmd
            .ue_radio_capability_id_deletion
            .unwrap()
            .deletes_network_assigned());
    }

    /// A reserved code is reported and treated as "no deletion": acting on an
    /// unknown code would delete IDs the network may not have asked to lose.
    #[test]
    fn a_reserved_deletion_request_is_not_treated_as_a_deletion() {
        let bytes = vec![0xA5];
        let cmd = ConfigurationUpdateCommand::decode(&mut bytes.as_slice()).unwrap();
        assert_eq!(
            cmd.ue_radio_capability_id_deletion,
            Some(RacsDeletionRequest::Reserved(5))
        );
        assert!(!cmd
            .ue_radio_capability_id_deletion
            .unwrap()
            .deletes_network_assigned());
    }

    #[test]
    fn a_command_carrying_a_rejected_nssai_decodes_it() {
        let mut bytes = vec![0xD1];
        bytes.push(config_update_iei::REJECTED_NSSAI);
        bytes.push(2); // length of the IE value
        bytes.extend_from_slice(&[0x10, 0x01]); // one entry: len 1, cause 0, SST 1

        let cmd = ConfigurationUpdateCommand::decode(&mut bytes.as_slice()).unwrap();
        assert_eq!(cmd.rejected_nssai, Some(vec![0x10, 0x01]));
    }

    /// Every new IE reaches the caller through the procedure result, not just
    /// the decoded message.
    #[test]
    fn process_command_carries_the_new_ies_to_the_caller() {
        let mut cmd = ConfigurationUpdateCommand::new();
        cmd.rejected_nssai = Some(vec![0x10, 0x01]);
        cmd.configured_nssai = Some(vec![0x01, 0x02]);
        cmd.ue_radio_capability_id = Some("123456".to_string());
        cmd.ue_radio_capability_id_deletion = Some(RacsDeletionRequest::NetworkAssigned);

        let result = ConfigUpdateProcedure::process_command(&cmd);
        assert_eq!(result.new_rejected_nssai, Some(vec![0x10, 0x01]));
        assert_eq!(result.new_configured_nssai, Some(vec![0x01, 0x02]));
        assert_eq!(result.new_racs_id.as_deref(), Some("123456"));
        assert_eq!(
            result.racs_deletion,
            Some(RacsDeletionRequest::NetworkAssigned)
        );
    }
}
