//! `HandoverPreparationInformation`: what the source node tells the target about the
//! UE (TS 38.331 §11.2.2, §6.2.2; TS 38.413 §9.3.1.20).
//!
//! This is the RRC message carried in the `RRCContainer` of the NGAP
//! `SourceNGRANNode-ToTargetNGRANNode-TransparentContainer`. The AMF passes it through
//! opaquely; the **target NG-RAN node** decodes it, which is why it lives here rather
//! than in either endpoint (issue #39).
//!
//! Before this the source sent `vec![0x00]` as its whole transparent container, so a
//! target learned nothing about the UE it was being handed — not even its capabilities,
//! which is what decides the configuration the target can give it.
//!
//! # What is carried, and what is deliberately not
//!
//! `ue-CapabilityRAT-List` is **mandatory** and is the point of the message: the target
//! must know what the UE supports before it configures anything. The source gNB already
//! holds the real `UE-CapabilityRAT-Container` bytes from the UE's
//! `UECapabilityInformation`, so this carries those rather than a fabrication.
//!
//! `sourceConfig` (`AS-Config`), `rrm-Config` and `as-Context` are optional and are
//! omitted. `AS-Config` would be the source's full `RRCReconfiguration` for delta
//! signalling, and this simulator's target sends a **fresh** configuration with
//! `fullConfig` — the same decision `fresh_rrc_resume_params` records. Sending a source
//! configuration the target then ignores would invite a peer to build a delta against
//! it.

use crate::codec::generated::*;
use crate::codec::{decode_rrc, encode_rrc, RrcCodecError};
use thiserror::Error;

/// Errors from building or reading a `HandoverPreparationInformation`.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum HandoverPreparationError {
    /// Codec error.
    #[error("Codec error: {0}")]
    CodecError(String),

    /// The message carried no NR capability container.
    #[error("HandoverPreparationInformation carries no NR UE capability container")]
    NoNrCapability,

    /// The critical extensions were a spare or future arm.
    #[error("HandoverPreparationInformation uses a critical extension this codec cannot read")]
    UnsupportedCriticalExtension,
}

impl From<RrcCodecError> for HandoverPreparationError {
    fn from(e: RrcCodecError) -> Self {
        Self::CodecError(e.to_string())
    }
}

/// What a source node states about the UE it is handing over.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HandoverPreparationParams {
    /// The UE's `UE-CapabilityRAT-Container` for NR, as the source collected it from
    /// the UE's `UECapabilityInformation`.
    ///
    /// Empty is legal on the wire (`UE-CapabilityRAT-ContainerList` has a lower bound
    /// of 0) and means the source never asked the UE what it supports — which is a real
    /// state this simulator can be in, and one the target should be able to see rather
    /// than have papered over.
    pub nr_capability: Option<Vec<u8>>,
}

/// What a target node reads out of one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HandoverPreparationData {
    /// The UE's NR capability container, if the source had one.
    pub nr_capability: Option<Vec<u8>>,
}

/// Build a `HandoverPreparationInformation`.
pub fn build_handover_preparation_information(
    params: &HandoverPreparationParams,
) -> HandoverPreparationInformation {
    let containers = match &params.nr_capability {
        Some(bytes) => vec![UE_CapabilityRAT_Container {
            rat_type: RAT_Type(RAT_Type::NR),
            ue_capability_rat_container: UE_CapabilityRAT_ContainerUe_CapabilityRAT_Container(
                bytes.clone(),
            ),
        }],
        None => Vec::new(),
    };
    HandoverPreparationInformation {
        critical_extensions: HandoverPreparationInformationCriticalExtensions::C1(
            HandoverPreparationInformationCriticalExtensions_c1::HandoverPreparationInformation(
                HandoverPreparationInformation_IEs {
                    ue_capability_rat_list: UE_CapabilityRAT_ContainerList(containers),
                    // Omitted deliberately -- see the module docs.
                    source_config: None,
                    rrm_config: None,
                    as_context: None,
                    non_critical_extension: None,
                },
            ),
        ),
    }
}

/// Read a `HandoverPreparationInformation`.
pub fn parse_handover_preparation_information(
    msg: &HandoverPreparationInformation,
) -> Result<HandoverPreparationData, HandoverPreparationError> {
    let ies = match &msg.critical_extensions {
        HandoverPreparationInformationCriticalExtensions::C1(
            HandoverPreparationInformationCriticalExtensions_c1::HandoverPreparationInformation(
                ies,
            ),
        ) => ies,
        _ => return Err(HandoverPreparationError::UnsupportedCriticalExtension),
    };
    // The NR container, specifically: the list may hold an E-UTRA one too, and a target
    // that took the first entry whatever its RAT would configure an NR cell from
    // E-UTRA capabilities.
    let nr_capability = ies
        .ue_capability_rat_list
        .0
        .iter()
        .find(|c| c.rat_type.0 == RAT_Type::NR)
        .map(|c| c.ue_capability_rat_container.0.clone());
    Ok(HandoverPreparationData { nr_capability })
}

/// Build and UPER-encode a `HandoverPreparationInformation` to the bytes the NGAP
/// `RRCContainer` carries.
pub fn encode_handover_preparation_information(
    params: &HandoverPreparationParams,
) -> Result<Vec<u8>, HandoverPreparationError> {
    Ok(encode_rrc(&build_handover_preparation_information(params))?)
}

/// Decode and read one from those bytes.
pub fn decode_handover_preparation_information(
    bytes: &[u8],
) -> Result<HandoverPreparationData, HandoverPreparationError> {
    let msg: HandoverPreparationInformation = decode_rrc(bytes)?;
    parse_handover_preparation_information(&msg)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #39: the UE's real capability container survives the round trip, so a target
    /// learns what the UE supports.
    #[test]
    fn the_ue_capability_container_survives_the_round_trip() {
        let capability = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02];
        let bytes = encode_handover_preparation_information(&HandoverPreparationParams {
            nr_capability: Some(capability.clone()),
        })
        .expect("encode");
        assert!(
            bytes.len() > 1,
            "the old container was a single 0x00 byte and carried nothing"
        );
        let data = decode_handover_preparation_information(&bytes).expect("decode");
        assert_eq!(
            data.nr_capability,
            Some(capability),
            "the target must receive the capabilities the source actually collected"
        );
    }

    /// A source that never asked the UE for its capabilities says so, rather than
    /// sending an empty container the target would read as "supports nothing".
    #[test]
    fn an_absent_capability_stays_absent() {
        let bytes = encode_handover_preparation_information(&HandoverPreparationParams {
            nr_capability: None,
        })
        .expect("encode");
        let data = decode_handover_preparation_information(&bytes).expect("decode");
        assert_eq!(data.nr_capability, None);
        // And the two encodings differ, or the capability is on the wire nowhere.
        assert_ne!(
            bytes,
            encode_handover_preparation_information(&HandoverPreparationParams {
                nr_capability: Some(vec![0x01]),
            })
            .unwrap()
        );
    }

    /// The **NR** container is selected, not the first entry.
    ///
    /// Pinned because with one entry the two are indistinguishable, and a target that
    /// configured an NR cell from E-UTRA capabilities would do so silently.
    #[test]
    fn the_nr_container_is_selected_and_not_merely_the_first() {
        let msg = HandoverPreparationInformation {
            critical_extensions: HandoverPreparationInformationCriticalExtensions::C1(
                HandoverPreparationInformationCriticalExtensions_c1::HandoverPreparationInformation(
                    HandoverPreparationInformation_IEs {
                        ue_capability_rat_list: UE_CapabilityRAT_ContainerList(vec![
                            UE_CapabilityRAT_Container {
                                rat_type: RAT_Type(RAT_Type::EUTRA),
                                ue_capability_rat_container:
                                    UE_CapabilityRAT_ContainerUe_CapabilityRAT_Container(vec![
                                        0xEE, 0xEE,
                                    ]),
                            },
                            UE_CapabilityRAT_Container {
                                rat_type: RAT_Type(RAT_Type::NR),
                                ue_capability_rat_container:
                                    UE_CapabilityRAT_ContainerUe_CapabilityRAT_Container(vec![
                                        0x11, 0x22,
                                    ]),
                            },
                        ]),
                        source_config: None,
                        rrm_config: None,
                        as_context: None,
                        non_critical_extension: None,
                    },
                ),
            ),
        };
        assert_eq!(
            parse_handover_preparation_information(&msg)
                .expect("parse")
                .nr_capability,
            Some(vec![0x11, 0x22]),
            "the E-UTRA container comes first on the wire; the NR one is what an NR \
             target must configure from"
        );
    }

    /// A future critical extension is refused rather than read as an empty message.
    #[test]
    fn an_unsupported_critical_extension_is_refused() {
        let msg = HandoverPreparationInformation {
            critical_extensions:
                HandoverPreparationInformationCriticalExtensions::CriticalExtensionsFuture(
                    HandoverPreparationInformationCriticalExtensions_criticalExtensionsFuture {},
                ),
        };
        assert_eq!(
            parse_handover_preparation_information(&msg).err(),
            Some(HandoverPreparationError::UnsupportedCriticalExtension),
            "a message this codec cannot read must not look like a UE with no capabilities"
        );
    }

    /// The old placeholder does not decode, so "it decodes" is a real check.
    #[test]
    fn the_old_placeholder_container_does_not_decode() {
        assert!(decode_handover_preparation_information(&[0x00]).is_err());
        assert!(decode_handover_preparation_information(&[]).is_err());
    }
}
