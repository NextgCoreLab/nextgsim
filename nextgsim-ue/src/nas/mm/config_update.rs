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
    /// T3512 value in seconds (optional, decoded from GPRS Timer 3 IE)
    pub t3512_value_secs: Option<u32>,
    /// Configured NSSAI (optional, Type 4, IEI 0x31)
    pub configured_nssai: Option<Vec<u8>>,
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
    /// New T3512 value in seconds (if provided)
    pub new_t3512_secs: Option<u32>,
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
            send_complete,
            re_register,
            new_sub_state,
        }
    }
}

// ============================================================================
// Helper: GPRS Timer 3 decoder
// ============================================================================

/// Decodes a GPRS Timer 3 IE byte into a duration in seconds.
///
/// Bits 7-5: unit, bits 4-0: value.
/// Units: 0=2s, 1=30s, 2=1min, 3=10min, 4=1hr, 5=10hr, 6=320hr, 7=deactivated
fn decode_gprs_timer3(byte: u8) -> u32 {
    let unit = (byte >> 5) & 0x07;
    let value = (byte & 0x1F) as u32;

    let multiplier: u32 = match unit {
        0 => 2,
        1 => 30,
        2 => 60,
        3 => 600,
        4 => 3600,
        5 => 36_000,
        6 => 1_152_000, // 320 hours
        _ => return u32::MAX, // deactivated
    };

    value * multiplier
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use nextgsim_nas::ies::ie1::{Acknowledgement, RegistrationRequested};

    fn build_config_update_command_bytes(
        ack: bool,
        red: bool,
        include_tai: bool,
    ) -> Vec<u8> {
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
        assert_eq!(result.new_sub_state, Some(MmSubState::RegisteredUpdateNeeded));
    }

    #[test]
    fn test_process_command_with_guti() {
        use nextgsim_nas::messages::mm::MobileIdentityType;
        let mut cmd = ConfigurationUpdateCommand::new();
        cmd.guti = Some(Ie5gsMobileIdentity::new(
            MobileIdentityType::Guti,
            vec![0x02, 0xF8, 0x39, 0xCA, 0xFE, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00],
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

    #[test]
    fn test_decode_gprs_timer3() {
        // Unit 0 (2s), value 5 -> 10 seconds
        assert_eq!(decode_gprs_timer3(0b000_00101), 10);
        // Unit 1 (30s), value 6 -> 180 seconds
        assert_eq!(decode_gprs_timer3(0b001_00110), 180);
        // Unit 2 (1min=60s), value 3 -> 180 seconds
        assert_eq!(decode_gprs_timer3(0b010_00011), 180);
        // Unit 7 (deactivated) -> u32::MAX
        assert_eq!(decode_gprs_timer3(0b111_00001), u32::MAX);
    }
}
