//! Typed UL-DCCH-Message dispatch (Wave-6 C5).
//!
//! TS 38.331 §6.2.1: every uplink dedicated-control message is a
//! `UL-DCCH-Message` whose `message.c1` CHOICE selects the concrete message
//! type. The historical gNB dispatcher routed on `bytes[0] & 0x0F` (a bespoke
//! nibble scheme) which only matched the UE's non-conformant fallback framing
//! and dropped every real UPER message. This module replaces that with a single
//! typed decode: [`dispatch_ul_dcch`] decodes the `UL-DCCH-Message` once and
//! returns a [`UlDcchMessage`] the caller matches on — no byte-offset NAS
//! extraction, no nibble arithmetic.
//!
//! ## Why the caller gates this behind `C5_TYPED_DCCH_DISPATCH`
//!
//! The matched-sim UE still emits bespoke framing whose leading bytes COLLIDE
//! with real UPER leading bytes: e.g. its uplink NAS `[0x08, 0x00, NAS…]`
//! decodes as a *well-formed* `RRCReconfigurationComplete` (UL-DCCH c1 index 1,
//! tid 0) and would silently swallow the NAS. So the broad typed dispatch is
//! only safe once the peer UE is itself a conformant typed peer; until then the
//! caller keeps the legacy nibble arms. The narrow `RRCSetupComplete`-only
//! check (Wave-6 C3) is collision-free and stays always-on separately.

use crate::codec::generated::*;
use crate::codec::{decode_rrc, RrcCodecError};

use super::information_transfer::{parse_ul_information_transfer, UlInformationTransferData};
use super::rrc_reconfiguration::{
    parse_rrc_reconfiguration_complete, RrcReconfigurationCompleteData,
};
use super::rrc_setup::{parse_rrc_setup_complete, RrcSetupCompleteData};
use super::security_mode::{parse_security_mode_complete, SecurityModeCompleteData};
use super::ue_capability::{parse_ue_capability_information, UeCapabilityInformationData};

/// A decoded UL-DCCH-Message, resolved to the concrete c1 message type the gNB
/// dispatches on (TS 38.331 §6.2.1). Variants the gNB does not act on collapse
/// to [`UlDcchMessage::Unsupported`].
#[derive(Debug, Clone)]
pub enum UlDcchMessage {
    /// `rrcSetupComplete` (c1 index 2) — carries the registration NAS.
    RrcSetupComplete(RrcSetupCompleteData),
    /// `securityModeComplete` (c1 index 5) — AS security activation confirmed.
    SecurityModeComplete(SecurityModeCompleteData),
    /// `rrcReconfigurationComplete` (c1 index 1).
    RrcReconfigurationComplete(RrcReconfigurationCompleteData),
    /// `ulInformationTransfer` (c1 index 7) — carries an uplink NAS message.
    UlInformationTransfer(UlInformationTransferData),
    /// `ueCapabilityInformation` (c1 index 9).
    UeCapabilityInformation(UeCapabilityInformationData),
    /// A well-formed UL-DCCH-Message the gNB does not dispatch on (e.g.
    /// measurementReport, securityModeFailure, messageClassExtension).
    Unsupported,
}

/// Decode a UL-DCCH-Message and resolve it to the typed message the gNB acts on
/// (TS 38.331 §6.2.1). Returns `Err` only when the bytes are not a decodable
/// UL-DCCH-Message; a decodable message of an unhandled type resolves to
/// [`UlDcchMessage::Unsupported`] (i.e. `Ok`).
pub fn dispatch_ul_dcch(bytes: &[u8]) -> Result<UlDcchMessage, RrcCodecError> {
    let msg: UL_DCCH_Message = decode_rrc(bytes)?;

    let c1 = match &msg.message {
        UL_DCCH_MessageType::C1(c1) => c1,
        _ => return Ok(UlDcchMessage::Unsupported),
    };

    let dispatched = match c1 {
        UL_DCCH_MessageType_c1::RrcSetupComplete(_) => {
            UlDcchMessage::RrcSetupComplete(parse_error(parse_rrc_setup_complete(&msg))?)
        }
        UL_DCCH_MessageType_c1::SecurityModeComplete(_) => {
            UlDcchMessage::SecurityModeComplete(parse_error(parse_security_mode_complete(&msg))?)
        }
        UL_DCCH_MessageType_c1::RrcReconfigurationComplete(_) => {
            UlDcchMessage::RrcReconfigurationComplete(parse_error(
                parse_rrc_reconfiguration_complete(&msg),
            )?)
        }
        UL_DCCH_MessageType_c1::UlInformationTransfer(_) => {
            UlDcchMessage::UlInformationTransfer(parse_error(parse_ul_information_transfer(&msg))?)
        }
        UL_DCCH_MessageType_c1::UeCapabilityInformation(_) => {
            UlDcchMessage::UeCapabilityInformation(parse_error(parse_ue_capability_information(
                &msg,
            ))?)
        }
        _ => UlDcchMessage::Unsupported,
    };

    Ok(dispatched)
}

/// Maps any per-procedure parse error into a codec decode error so the caller
/// gets one uniform error type.
fn parse_error<T, E: std::fmt::Debug>(r: Result<T, E>) -> Result<T, RrcCodecError> {
    r.map_err(|e| RrcCodecError::DecodeError(format!("{e:?}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::procedures::information_transfer::{
        encode_ul_information_transfer, UlInformationTransferParams,
    };
    use crate::procedures::rrc_reconfiguration::{
        encode_rrc_reconfiguration_complete, RrcReconfigurationCompleteParams,
    };
    use crate::procedures::rrc_setup::{encode_rrc_setup_complete, RrcSetupCompleteParams};
    use crate::procedures::security_mode::{
        encode_security_mode_complete, SecurityModeCompleteParams,
    };

    #[test]
    fn dispatch_rrc_setup_complete() {
        let nas = vec![0x7E, 0x00, 0x41];
        let bytes = encode_rrc_setup_complete(&RrcSetupCompleteParams {
            registered_amf: None,
            rrc_transaction_id: 0,
            selected_plmn_identity: 1,
            guami_type: None,
            s_nssai_list: None,
            dedicated_nas_message: nas.clone(),
            ng_5g_s_tmsi_value: None,
            redcap_indication: false,
        })
        .unwrap();
        match dispatch_ul_dcch(&bytes).unwrap() {
            UlDcchMessage::RrcSetupComplete(d) => {
                assert_eq!(d.rrc_transaction_id, 0);
                assert_eq!(d.dedicated_nas_message, nas);
            }
            other => panic!("expected RrcSetupComplete, got {other:?}"),
        }
    }

    #[test]
    fn dispatch_security_mode_complete() {
        let bytes = encode_security_mode_complete(&SecurityModeCompleteParams {
            rrc_transaction_id: 2,
        })
        .unwrap();
        match dispatch_ul_dcch(&bytes).unwrap() {
            UlDcchMessage::SecurityModeComplete(d) => assert_eq!(d.rrc_transaction_id, 2),
            other => panic!("expected SecurityModeComplete, got {other:?}"),
        }
    }

    #[test]
    fn dispatch_rrc_reconfiguration_complete() {
        let bytes = encode_rrc_reconfiguration_complete(&RrcReconfigurationCompleteParams {
            rrc_transaction_id: 1,
        })
        .unwrap();
        match dispatch_ul_dcch(&bytes).unwrap() {
            UlDcchMessage::RrcReconfigurationComplete(d) => assert_eq!(d.rrc_transaction_id, 1),
            other => panic!("expected RrcReconfigurationComplete, got {other:?}"),
        }
    }

    #[test]
    fn dispatch_ul_information_transfer() {
        let nas = vec![0x7E, 0x00, 0x42];
        let bytes = encode_ul_information_transfer(&UlInformationTransferParams {
            dedicated_nas_message: Some(nas.clone()),
        })
        .unwrap();
        match dispatch_ul_dcch(&bytes).unwrap() {
            UlDcchMessage::UlInformationTransfer(d) => {
                assert_eq!(d.dedicated_nas_message, Some(nas));
            }
            other => panic!("expected UlInformationTransfer, got {other:?}"),
        }
    }

    /// The collision the C5 gate guards against: the matched-sim UE's bespoke
    /// uplink NAS framing `[0x08, 0x00, NAS…]` decodes as a *well-formed*
    /// RRCReconfigurationComplete (c1 index 1, tid 0) and silently drops the
    /// NAS. This is exactly why the caller must NOT run the broad typed
    /// dispatch until the UE is a conformant typed peer.
    #[test]
    fn bespoke_ul_info_transfer_misdecodes_as_reconfig_complete() {
        let bespoke = [0x08u8, 0x00, 0x7E, 0x00, 0x41];
        match dispatch_ul_dcch(&bespoke).unwrap() {
            UlDcchMessage::RrcReconfigurationComplete(d) => {
                assert_eq!(d.rrc_transaction_id, 0);
            }
            other => panic!("collision expected RrcReconfigurationComplete, got {other:?}"),
        }
    }

    #[test]
    fn dispatch_ue_capability_information() {
        use crate::procedures::ue_capability::{
            encode_ue_capability_information, RatType, UeCapabilityInformationParams,
            UeCapabilityRatContainer,
        };
        let bytes = encode_ue_capability_information(&UeCapabilityInformationParams {
            rrc_transaction_id: 3,
            containers: vec![UeCapabilityRatContainer {
                rat_type: RatType::Nr,
                container: vec![0x00],
            }],
        })
        .unwrap();
        match dispatch_ul_dcch(&bytes).unwrap() {
            UlDcchMessage::UeCapabilityInformation(d) => {
                assert_eq!(d.rrc_transaction_id, 3);
                assert_eq!(d.containers.len(), 1);
            }
            other => panic!("expected UeCapabilityInformation, got {other:?}"),
        }
    }

    #[test]
    fn message_class_extension_is_unsupported() {
        // Leading bit 1 selects the messageClassExtension arm (empty SEQUENCE),
        // a well-formed but non-dispatched UL-DCCH-Message.
        assert!(matches!(
            dispatch_ul_dcch(&[0x80]).unwrap(),
            UlDcchMessage::Unsupported
        ));
    }

    #[test]
    fn truncated_message_errors() {
        // A ulInformationTransfer that claims a NAS payload but is truncated
        // mid-content is not decodable.
        let full = encode_ul_information_transfer(&UlInformationTransferParams {
            dedicated_nas_message: Some(vec![0x7E, 0x00, 0x41, 0x02, 0x03]),
        })
        .unwrap();
        let truncated = &full[..2];
        assert!(dispatch_ul_dcch(truncated).is_err());
    }
}
