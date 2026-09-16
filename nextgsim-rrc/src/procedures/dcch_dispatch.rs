//! Typed DCCH/CCCH-Message dispatch, both directions (Wave-6 C5).
//!
//! TS 38.331 §6.2.1: every dedicated-control message is a `UL-DCCH-Message` or
//! `DL-DCCH-Message` whose `message.c1` CHOICE selects the concrete message
//! type. Both historical dispatchers routed on `bytes[0] & 0x0F` (a bespoke
//! nibble scheme) which only matched the peer simulator's non-conformant
//! fallback framing and misrouted or dropped real UPER messages. This module
//! replaces both with a single typed decode per direction:
//! [`dispatch_ul_dcch`], [`dispatch_dl_dcch`] and [`dispatch_dl_ccch`] each
//! decode the message once and return an enum the caller matches on — no
//! byte-offset NAS extraction, no nibble arithmetic.
//!
//! ## Why a nibble cannot work
//!
//! The leading byte of a UPER `DL-DCCH-Message` is `0b0_iiii_ttt_x`: bit 0
//! selects `c1`, bits 1–4 are the message index from the schema's declaration
//! order, bits 5–6 are the `rrc-TransactionIdentifier` and bit 7 begins
//! `criticalExtensions`. So the low nibble is a function of the message index
//! AND the transaction identifier, and two different messages share a nibble
//! while one message occupies several. `DLInformationTransfer` (index 5, tid 0)
//! leads with `0x28`, whose low nibble `0x8` is the one the UE's old dispatcher
//! read as `RRCResume` — downlink NAS was delivered to the resume handler and
//! silently dropped (issue #151). The same arithmetic pinned every gNB→UE
//! transaction identifier to 0, because a non-zero tid moves bits 5–6 into the
//! routed nibble.
//!
//! DL-CCCH is the same shape with a 2-bit index (`rrcReject` = 0,
//! `rrcSetup` = 1): a real `RRCReject` leads with `0x00`, which the old
//! dispatcher read as `RRCSetup`.
//!
//! ## Simulator framing is matched separately, not here
//!
//! These functions decode real UPER only. The two hand-rolled envelopes this
//! simulator still uses on DL-DCCH — [`SIM_RECONFIGURATION_WITH_CHO`] and
//! [`SIM_RECONFIGURATION_WITH_SCELL`] — are matched by the caller BEFORE this
//! dispatch, on the full first byte rather than a nibble, and documented at the
//! call site as simulator framing. Both now live in the `messageClassExtension`
//! space so they cannot collide with a real message (see
//! [`SIM_RECONFIGURATION_WITH_SCELL`]); issue #107 covers replacing them, and the
//! CHO container also needs a Rel-16 schema (issue #105). A third envelope,
//! `0x06` for UE capability transfer, is gone entirely: it was also a conformant
//! `RRCReconfiguration` at tid 3.

use crate::codec::generated::*;
use crate::codec::{decode_rrc, RrcCodecError};

/// Simulator-private DL-DCCH framing code for an `RRCReconfiguration` carrying a
/// conditional-reconfiguration container: `[0x8D][transaction id][container]`.
///
/// See [`SIM_RECONFIGURATION_WITH_SCELL`] for why the top bit is set.
pub const SIM_RECONFIGURATION_WITH_CHO: u8 = 0x8D;

/// Simulator-private DL-DCCH framing code for an `RRCReconfiguration` carrying a
/// secondary-cell configuration: `[0x8E][transaction id][UPER CellGroupConfig]`.
///
/// ## Why the top bit is set, and why these two codes moved (issue #151)
///
/// TS 38.331 §6.2.2's `DL-DCCH-Message` is a CHOICE of `c1` and
/// `messageClassExtension`; a leading bit of **1** selects
/// `messageClassExtension`, an empty SEQUENCE. So every byte with the top bit
/// set is a well-formed DL-DCCH-Message carrying nothing, and **no `c1` message
/// can ever lead with one**. That makes it the only space a private envelope can
/// occupy without colliding with a real message.
///
/// The previous codes were `0x0D` and `0x0E`, and they DID collide. The leading
/// byte of a `c1` message is `(index << 3) | (tid << 1) | criticalExtensions`,
/// so once transaction identifiers stopped being pinned to 0 a conformant
/// `RRCResume` (index 1) with tid 3 leads with `0x0E` and would have been eaten
/// by the secondary-cell arm, and one with tid 2 and `criticalExtensionsFuture`
/// leads with `0x0D`. The retired `0x06` UE-capability envelope collided the
/// same way with an `RRCReconfiguration` at tid 3.
///
/// This is simulator framing, not 3GPP: a real peer sending
/// `messageClassExtension` is answered as unsupported, which is the honest
/// outcome for a message class this release does not define. Issue #107 covers
/// replacing both envelopes with real UPER — the CHO container additionally
/// needs a Rel-16 schema (issue #105).
pub const SIM_RECONFIGURATION_WITH_SCELL: u8 = 0x8E;

use super::information_transfer::{
    parse_dl_information_transfer, parse_ul_information_transfer, DlInformationTransferData,
    UlInformationTransferData,
};
use super::measurement_report::{parse_measurement_report, MeasurementReportData};
use super::rrc_reconfiguration::{
    parse_rrc_reconfiguration, parse_rrc_reconfiguration_complete, RrcReconfigurationCompleteData,
    RrcReconfigurationData,
};
use super::rrc_reestablishment::{
    parse_rrc_reestablishment, parse_rrc_reestablishment_complete, RrcReestablishmentCompleteData,
    RrcReestablishmentData,
};
use super::rrc_release::{parse_rrc_release, RrcReleaseData};
use super::rrc_resume::{
    parse_rrc_resume, parse_rrc_resume_complete, RrcResumeCompleteData, RrcResumeData,
};
use super::rrc_setup::{
    parse_rrc_setup, parse_rrc_setup_complete, RrcSetupCompleteData, RrcSetupData,
};
use super::security_mode::{
    parse_security_mode_command, parse_security_mode_complete, SecurityModeCommandData,
    SecurityModeCompleteData,
};
use super::ue_capability::{
    parse_ue_capability_enquiry, parse_ue_capability_information, UeCapabilityEnquiryData,
    UeCapabilityInformationData,
};

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
    /// `rrcReestablishmentComplete` (c1 index 3).
    RrcReestablishmentComplete(RrcReestablishmentCompleteData),
    /// `rrcResumeComplete` (c1 index 4) — may carry a piggybacked NAS message.
    RrcResumeComplete(RrcResumeCompleteData),
    /// `measurementReport` (c1 index 0) — the measurements a handover decision is
    /// made from (TS 38.331 §5.5.5).
    MeasurementReport(MeasurementReportData),
    /// `securityModeFailure` (c1 index 6) — the UE refused the
    /// SecurityModeCommand with this transaction identifier (TS 38.331 §5.3.4.4).
    SecurityModeFailure {
        /// The `rrc-TransactionIdentifier` being refused.
        rrc_transaction_id: u8,
    },
    /// A well-formed UL-DCCH-Message the gNB does not dispatch on (e.g.
    /// locationMeasurementIndication, messageClassExtension).
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
        UL_DCCH_MessageType_c1::RrcReestablishmentComplete(_) => {
            UlDcchMessage::RrcReestablishmentComplete(parse_error(
                parse_rrc_reestablishment_complete(&msg),
            )?)
        }
        UL_DCCH_MessageType_c1::RrcResumeComplete(_) => {
            UlDcchMessage::RrcResumeComplete(parse_error(parse_rrc_resume_complete(&msg))?)
        }
        UL_DCCH_MessageType_c1::MeasurementReport(_) => {
            UlDcchMessage::MeasurementReport(parse_error(parse_measurement_report(&msg))?)
        }
        UL_DCCH_MessageType_c1::SecurityModeFailure(failure) => {
            UlDcchMessage::SecurityModeFailure {
                rrc_transaction_id: failure.rrc_transaction_identifier.0,
            }
        }
        _ => UlDcchMessage::Unsupported,
    };

    Ok(dispatched)
}

/// A decoded DL-DCCH-Message, resolved to the concrete c1 message type the UE
/// dispatches on (TS 38.331 §6.2.1). Variants the UE does not act on collapse
/// to [`DlDcchMessage::Unsupported`] — which is the point: an unhandled message
/// must be *named* as unhandled rather than fall into a sibling's arm, which is
/// what the nibble dispatcher did (issue #151).
#[derive(Debug, Clone)]
pub enum DlDcchMessage {
    /// `rrcReconfiguration` (c1 index 0).
    RrcReconfiguration(RrcReconfigurationData),
    /// `rrcResume` (c1 index 1).
    RrcResume(RrcResumeData),
    /// `rrcRelease` (c1 index 2).
    RrcRelease(RrcReleaseData),
    /// `rrcReestablishment` (c1 index 3).
    RrcReestablishment(RrcReestablishmentData),
    /// `securityModeCommand` (c1 index 4).
    SecurityModeCommand(SecurityModeCommandData),
    /// `dlInformationTransfer` (c1 index 5) — carries a downlink NAS message.
    DlInformationTransfer(DlInformationTransferData),
    /// `ueCapabilityEnquiry` (c1 index 6).
    UeCapabilityEnquiry(UeCapabilityEnquiryData),
    /// A well-formed DL-DCCH-Message the UE does not dispatch on: `counterCheck`
    /// (index 7), `mobilityFromNRCommand` (index 8), a spare, or
    /// `messageClassExtension`.
    Unsupported,
}

/// Decode a DL-DCCH-Message and resolve it to the typed message the UE acts on
/// (TS 38.331 §6.2.1). Returns `Err` only when the bytes are not a decodable
/// DL-DCCH-Message; a decodable message of an unhandled type resolves to
/// [`DlDcchMessage::Unsupported`] (i.e. `Ok`).
pub fn dispatch_dl_dcch(bytes: &[u8]) -> Result<DlDcchMessage, RrcCodecError> {
    let msg: DL_DCCH_Message = decode_rrc(bytes)?;

    let c1 = match &msg.message {
        DL_DCCH_MessageType::C1(c1) => c1,
        _ => return Ok(DlDcchMessage::Unsupported),
    };

    let dispatched = match c1 {
        DL_DCCH_MessageType_c1::RrcReconfiguration(_) => {
            DlDcchMessage::RrcReconfiguration(parse_error(parse_rrc_reconfiguration(&msg))?)
        }
        DL_DCCH_MessageType_c1::RrcResume(_) => {
            DlDcchMessage::RrcResume(parse_error(parse_rrc_resume(&msg))?)
        }
        DL_DCCH_MessageType_c1::RrcRelease(_) => {
            DlDcchMessage::RrcRelease(parse_error(parse_rrc_release(&msg))?)
        }
        DL_DCCH_MessageType_c1::RrcReestablishment(_) => {
            DlDcchMessage::RrcReestablishment(parse_error(parse_rrc_reestablishment(&msg))?)
        }
        DL_DCCH_MessageType_c1::SecurityModeCommand(_) => {
            DlDcchMessage::SecurityModeCommand(parse_error(parse_security_mode_command(&msg))?)
        }
        DL_DCCH_MessageType_c1::DlInformationTransfer(_) => {
            DlDcchMessage::DlInformationTransfer(parse_error(parse_dl_information_transfer(&msg))?)
        }
        DL_DCCH_MessageType_c1::UeCapabilityEnquiry(_) => {
            DlDcchMessage::UeCapabilityEnquiry(parse_error(parse_ue_capability_enquiry(&msg))?)
        }
        _ => DlDcchMessage::Unsupported,
    };

    Ok(dispatched)
}

/// A decoded DL-CCCH-Message, resolved to the concrete c1 message type the UE
/// dispatches on (TS 38.331 §6.2.1).
///
/// The nibble dispatcher this replaces read a real `RRCReject` (leading byte
/// `0x00`) as an `RRCSetup`, and would have misrouted `RRCSetup` itself for two
/// of the four legal transaction identifiers once tids stopped being pinned to 0
/// (tid 1 → `0x28`, tid 3 → `0x38`, neither with low nibble `0x0`).
#[derive(Debug, Clone)]
pub enum DlCcchMessage {
    /// `rrcSetup` (c1 index 1).
    RrcSetup(RrcSetupData),
    /// `rrcReject` (c1 index 0). No `rrc-TransactionIdentifier`: TS 38.331
    /// §6.2.2 gives `RRCReject` only `criticalExtensions`.
    RrcReject,
    /// A well-formed DL-CCCH-Message the UE does not dispatch on (a spare, or
    /// `messageClassExtension`).
    Unsupported,
}

/// Decode a DL-CCCH-Message and resolve it to the typed message the UE acts on
/// (TS 38.331 §6.2.1). Returns `Err` only when the bytes are not a decodable
/// DL-CCCH-Message.
pub fn dispatch_dl_ccch(bytes: &[u8]) -> Result<DlCcchMessage, RrcCodecError> {
    let msg: DL_CCCH_Message = decode_rrc(bytes)?;

    let c1 = match &msg.message {
        DL_CCCH_MessageType::C1(c1) => c1,
        _ => return Ok(DlCcchMessage::Unsupported),
    };

    let dispatched = match c1 {
        DL_CCCH_MessageType_c1::RrcSetup(_) => {
            DlCcchMessage::RrcSetup(parse_error(parse_rrc_setup(&msg))?)
        }
        DL_CCCH_MessageType_c1::RrcReject(_) => DlCcchMessage::RrcReject,
        _ => DlCcchMessage::Unsupported,
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

    // ---------------------------------------------------------------------
    // Downlink (issue #151). The property under test is DISCRIMINATION: each
    // message type must resolve to its OWN variant and to no other. A misroute
    // shows up here as a different variant, which is exactly the failure the
    // nibble dispatcher had and no encode/decode round trip could catch.
    // ---------------------------------------------------------------------

    /// The whole DL-DCCH set the UE acts on, each built as real UPER and each
    /// asserted to reach its own variant. Table-driven so a new c1 arm cannot be
    /// added without a row.
    #[test]
    fn every_dl_dcch_message_type_dispatches_to_its_own_variant() {
        use crate::procedures::information_transfer::{
            encode_dl_information_transfer, DlInformationTransferParams,
        };
        use crate::procedures::rrc_reconfiguration::{
            encode_rrc_reconfiguration, RrcReconfigurationParams,
        };
        use crate::procedures::rrc_reestablishment::{
            encode_rrc_reestablishment, RrcReestablishmentParams,
        };
        use crate::procedures::rrc_release::{encode_rrc_release, RrcReleaseParams};
        use crate::procedures::rrc_resume::{encode_rrc_resume, fresh_rrc_resume_params};
        use crate::procedures::security_mode::{
            encode_security_mode_command, CipheringAlgorithmType, IntegrityAlgorithmType,
            SecurityAlgorithms, SecurityModeCommandParams,
        };
        use crate::procedures::ue_capability::{
            encode_ue_capability_enquiry, RatType, UeCapabilityEnquiryParams,
        };

        let reconfiguration = encode_rrc_reconfiguration(&RrcReconfigurationParams {
            rrc_transaction_id: 1,
            radio_bearer_config: None,
            secondary_cell_group: None,
            master_cell_group: None,
            full_config: false,
            master_key_update: None,
        })
        .expect("encode RRCReconfiguration");
        let resume = encode_rrc_resume(&fresh_rrc_resume_params(2).expect("resume params"))
            .expect("encode RRCResume");
        let release = encode_rrc_release(&RrcReleaseParams {
            rrc_transaction_id: 3,
            redirected_carrier_info: None,
            cell_reselection_priorities: None,
            suspend_config: None,
            deprioritisation_req: None,
            wait_time: None,
        })
        .expect("encode RRCRelease");
        let reestablishment = encode_rrc_reestablishment(&RrcReestablishmentParams {
            rrc_transaction_id: 1,
            next_hop_chaining_count: 2,
        })
        .expect("encode RRCReestablishment");
        let smc = encode_security_mode_command(&SecurityModeCommandParams {
            rrc_transaction_id: 2,
            security_algorithms: SecurityAlgorithms {
                ciphering_algorithm: CipheringAlgorithmType::Nea0,
                integrity_algorithm: Some(IntegrityAlgorithmType::Nia2),
            },
        })
        .expect("encode SecurityModeCommand");
        let dl_info = encode_dl_information_transfer(&DlInformationTransferParams {
            rrc_transaction_id: 1,
            dedicated_nas_message: Some(vec![0x7E, 0x00, 0x42]),
        })
        .expect("encode DLInformationTransfer");
        let enquiry = encode_ue_capability_enquiry(&UeCapabilityEnquiryParams {
            rrc_transaction_id: 3,
            rat_types: vec![RatType::Nr],
        })
        .expect("encode UECapabilityEnquiry");

        /// (label, encoded bytes, the ONE variant those bytes may resolve to).
        type DispatchCase = (&'static str, Vec<u8>, fn(&DlDcchMessage) -> bool);

        let cases: Vec<DispatchCase> = vec![
            ("rrcReconfiguration", reconfiguration, |m| {
                matches!(m, DlDcchMessage::RrcReconfiguration(_))
            }),
            ("rrcResume", resume, |m| {
                matches!(m, DlDcchMessage::RrcResume(_))
            }),
            ("rrcRelease", release, |m| {
                matches!(m, DlDcchMessage::RrcRelease(_))
            }),
            ("rrcReestablishment", reestablishment, |m| {
                matches!(m, DlDcchMessage::RrcReestablishment(_))
            }),
            ("securityModeCommand", smc, |m| {
                matches!(m, DlDcchMessage::SecurityModeCommand(_))
            }),
            ("dlInformationTransfer", dl_info, |m| {
                matches!(m, DlDcchMessage::DlInformationTransfer(_))
            }),
            ("ueCapabilityEnquiry", enquiry, |m| {
                matches!(m, DlDcchMessage::UeCapabilityEnquiry(_))
            }),
        ];

        for (label, bytes, is_expected) in &cases {
            let dispatched = dispatch_dl_dcch(bytes)
                .unwrap_or_else(|e| panic!("{label} must decode as a DL-DCCH-Message: {e:?}"));
            assert!(
                is_expected(&dispatched),
                "{label} (leading byte {:#04x}) dispatched to the WRONG variant: {dispatched:?}",
                bytes[0]
            );
        }

        // And the cross product: no other message's bytes may satisfy a given
        // row's predicate. This is the "and no other" half made explicit.
        for (label, _, is_expected) in &cases {
            for (other_label, other_bytes, _) in &cases {
                if label == other_label {
                    continue;
                }
                let dispatched = dispatch_dl_dcch(other_bytes).expect("decodes");
                assert!(
                    !is_expected(&dispatched),
                    "{other_label} was dispatched as {label}"
                );
            }
        }
    }

    /// The live interop defect from issue #151, named: a conformant
    /// `DLInformationTransfer` leads with `0x28`, whose low nibble `0x8` was the
    /// UE's `RRCResume` arm — so downlink NAS was handed to the resume procedure
    /// and dropped. The leading-byte assertion is what pins the collision; the
    /// dispatch assertion is what proves it no longer decides the route.
    #[test]
    fn a_conformant_dl_information_transfer_is_not_dispatched_as_a_resume() {
        use crate::procedures::information_transfer::{
            encode_dl_information_transfer, DlInformationTransferParams,
        };
        let nas = vec![0x7E, 0x00, 0x42, 0x01];
        let bytes = encode_dl_information_transfer(&DlInformationTransferParams {
            rrc_transaction_id: 0,
            dedicated_nas_message: Some(nas.clone()),
        })
        .expect("encode DLInformationTransfer");

        assert_eq!(
            bytes[0], 0x28,
            "DL-DCCH c1 index 5, tid 0 -> leading byte 0x28 (low nibble 0x8, \
             the old RRCResume arm)"
        );
        match dispatch_dl_dcch(&bytes).expect("decodes") {
            DlDcchMessage::DlInformationTransfer(d) => {
                assert_eq!(d.dedicated_nas_message, Some(nas));
            }
            other => panic!("expected DlInformationTransfer, got {other:?}"),
        }
    }

    /// A real `UECapabilityEnquiry` dispatches on its own encoding, and the
    /// retired `0x06` envelope framing does NOT — so the envelope cannot be
    /// quietly reintroduced as the thing that makes capability transfer work.
    #[test]
    fn a_ue_capability_enquiry_dispatches_without_the_legacy_envelope_byte() {
        use crate::procedures::ue_capability::{
            encode_ue_capability_enquiry, RatType, UeCapabilityEnquiryParams,
        };
        let bytes = encode_ue_capability_enquiry(&UeCapabilityEnquiryParams {
            rrc_transaction_id: 2,
            rat_types: vec![RatType::Nr],
        })
        .expect("encode UECapabilityEnquiry");

        match dispatch_dl_dcch(&bytes).expect("decodes") {
            DlDcchMessage::UeCapabilityEnquiry(d) => assert_eq!(d.rrc_transaction_id, 2),
            other => panic!("expected UeCapabilityEnquiry, got {other:?}"),
        }

        let mut enveloped = Vec::with_capacity(bytes.len() + 1);
        enveloped.push(0x06);
        enveloped.extend_from_slice(&bytes);
        assert!(
            !matches!(
                dispatch_dl_dcch(&enveloped),
                Ok(DlDcchMessage::UeCapabilityEnquiry(_))
            ),
            "the 0x06-enveloped framing is simulator framing, not a DL-DCCH-Message: \
             it must not dispatch here"
        );
    }

    /// Every legal `rrc-TransactionIdentifier` (INTEGER(0..3), TS 38.331 §6.3.2)
    /// round-trips, and the leading byte MOVES with the tid — which is why the
    /// nibble dispatcher pinned tids to 0 and why the typed one need not.
    #[test]
    fn a_non_zero_transaction_identifier_round_trips_and_moves_the_leading_byte() {
        use crate::procedures::security_mode::{
            encode_security_mode_command, CipheringAlgorithmType, IntegrityAlgorithmType,
            SecurityAlgorithms, SecurityModeCommandParams,
        };
        let mut leading = Vec::new();
        for tid in 0u8..=3 {
            let bytes = encode_security_mode_command(&SecurityModeCommandParams {
                rrc_transaction_id: tid,
                security_algorithms: SecurityAlgorithms {
                    ciphering_algorithm: CipheringAlgorithmType::Nea0,
                    integrity_algorithm: Some(IntegrityAlgorithmType::Nia2),
                },
            })
            .expect("encode SecurityModeCommand");
            leading.push(bytes[0]);
            match dispatch_dl_dcch(&bytes).expect("decodes") {
                DlDcchMessage::SecurityModeCommand(d) => assert_eq!(
                    d.rrc_transaction_id, tid,
                    "tid {tid} must survive the round trip"
                ),
                other => panic!("tid {tid}: expected SecurityModeCommand, got {other:?}"),
            }
        }
        // Bits 5-6 carry the tid, so all four leading bytes differ: a nibble
        // dispatcher cannot route all four to one arm.
        assert_eq!(leading, vec![0x20, 0x22, 0x24, 0x26], "tid moves bits 5-6");
    }

    /// `counterCheck` and `mobilityFromNRCommand` have no UE handler. They must
    /// be NAMED unsupported rather than fall into a sibling's arm — the same
    /// property, for the messages we deliberately do not act on.
    #[test]
    fn a_non_dispatched_dl_dcch_message_is_named_unsupported() {
        // Leading bit 1 selects messageClassExtension (an empty SEQUENCE).
        assert!(matches!(
            dispatch_dl_dcch(&[0x80]).expect("decodes"),
            DlDcchMessage::Unsupported
        ));
    }

    #[test]
    fn a_truncated_dl_dcch_message_errors() {
        use crate::procedures::information_transfer::{
            encode_dl_information_transfer, DlInformationTransferParams,
        };
        let full = encode_dl_information_transfer(&DlInformationTransferParams {
            rrc_transaction_id: 0,
            dedicated_nas_message: Some(vec![0x7E, 0x00, 0x41, 0x02, 0x03]),
        })
        .expect("encode DLInformationTransfer");
        assert!(dispatch_dl_dcch(&full[..2]).is_err());
    }

    /// A `RRCResumeComplete` carrying a piggybacked NAS must dispatch WITH that
    /// NAS. The parser used to drop `dedicatedNAS-Message` on the floor, which only
    /// went unnoticed because the peer recovered it by slicing bespoke framing
    /// (issue #151).
    #[test]
    fn a_resume_complete_dispatches_with_its_piggybacked_nas() {
        use crate::procedures::rrc_resume::{encode_rrc_resume_complete, RrcResumeCompleteParams};
        let nas = vec![0x7E, 0x00, 0x4C, 0x01];
        let bytes = encode_rrc_resume_complete(&RrcResumeCompleteParams {
            rrc_transaction_id: 2,
            dedicated_nas_message: Some(nas.clone()),
            selected_plmn_identity: Some(1),
        })
        .expect("encode RRCResumeComplete");

        match dispatch_ul_dcch(&bytes).expect("decodes") {
            UlDcchMessage::RrcResumeComplete(d) => {
                assert_eq!(d.rrc_transaction_id, 2);
                assert_eq!(
                    d.dedicated_nas_message,
                    Some(nas),
                    "the piggybacked NAS must survive the round trip"
                );
                assert_eq!(d.selected_plmn_identity, Some(1));
            }
            other => panic!("expected RrcResumeComplete, got {other:?}"),
        }
    }

    /// A `SecurityModeFailure` must be resolved to its OWN variant, not collapsed
    /// into `Unsupported` -- otherwise a UE that refuses AS security is
    /// indistinguishable from one that says nothing.
    #[test]
    fn a_security_mode_failure_is_named_and_not_unsupported() {
        use crate::procedures::security_mode::{
            encode_security_mode_failure, SecurityModeFailureParams,
        };
        let bytes = encode_security_mode_failure(&SecurityModeFailureParams {
            rrc_transaction_id: 3,
        })
        .expect("encode SecurityModeFailure");
        assert!(matches!(
            dispatch_ul_dcch(&bytes).expect("decodes"),
            UlDcchMessage::SecurityModeFailure {
                rrc_transaction_id: 3
            }
        ));
    }

    /// The reason the two simulator envelope codes moved out of `0x0D`/`0x0E`:
    /// a CONFORMANT `RRCResume` at tid 3 leads with `0x0E`, so the old
    /// secondary-cell arm would have swallowed it the moment tids stopped being
    /// pinned to 0. Asserted on the real encoder, not on arithmetic in a comment.
    #[test]
    fn the_old_simulator_envelope_codes_collided_with_a_conformant_rrc_resume() {
        use crate::procedures::rrc_resume::{encode_rrc_resume, fresh_rrc_resume_params};
        let resume = encode_rrc_resume(&fresh_rrc_resume_params(3).expect("resume params"))
            .expect("encode RRCResume");
        assert_eq!(
            resume[0], 0x0E,
            "RRCResume (c1 index 1) at tid 3 leads with 0x0E — the byte the \
             secondary-cell envelope used to claim"
        );
        assert!(matches!(
            dispatch_dl_dcch(&resume).expect("decodes"),
            DlDcchMessage::RrcResume(_)
        ));

        // And the replacement codes cannot collide with anything: the top bit
        // selects messageClassExtension, which no c1 message can reach.
        for code in [SIM_RECONFIGURATION_WITH_CHO, SIM_RECONFIGURATION_WITH_SCELL] {
            assert_ne!(code & 0x80, 0, "{code:#04x} must set the top bit");
            assert!(
                matches!(
                    dispatch_dl_dcch(&[code]).expect("decodes"),
                    DlDcchMessage::Unsupported
                ),
                "{code:#04x} must resolve to Unsupported, never to a c1 message"
            );
        }
    }

    /// DL-CCCH half of the same defect: `RRCReject` carries no transaction
    /// identifier, so a real one leads with `0x00` — the nibble the old
    /// dispatcher routed to `RRCSetup`. A rejected UE was told it was set up.
    #[test]
    fn a_conformant_rrc_reject_is_not_dispatched_as_an_rrc_setup() {
        // `[0x00]`: bit 0 selects c1, bits 1-2 select rrcReject (index 0), the
        // next bit selects the rrcReject criticalExtensions arm, and the three
        // OPTIONAL RRCReject-IEs members are all absent -> 7 bits, padded.
        let reject = [0x00u8];
        assert!(
            matches!(
                dispatch_dl_ccch(&reject).expect("decodes"),
                DlCcchMessage::RrcReject
            ),
            "a real RRCReject must dispatch as a reject, not as a setup"
        );
    }

    /// `RRCSetup` dispatches for every legal tid. Two of the four (1 and 3) have
    /// a non-zero low nibble, which the old DL-CCCH nibble dispatcher would have
    /// dropped outright — this is the DL-CCCH reason tids were pinned to 0.
    #[test]
    fn an_rrc_setup_dispatches_for_every_legal_transaction_identifier() {
        use crate::procedures::rrc_setup::{encode_rrc_setup, srb1_rrc_setup_params};
        for tid in 0u8..=3 {
            let bytes = encode_rrc_setup(&srb1_rrc_setup_params(tid).expect("setup params"))
                .expect("encode RRCSetup");
            match dispatch_dl_ccch(&bytes).expect("decodes") {
                DlCcchMessage::RrcSetup(d) => {
                    assert_eq!(d.rrc_transaction_id, tid, "tid {tid} must survive");
                }
                other => panic!("tid {tid}: expected RrcSetup, got {other:?}"),
            }
        }
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
