//! `SidelinkUEInformationNR` and `sl-ConfigDedicatedNR`: the RRC half of sidelink —
//! the UE's sidelink resource request and the network's dedicated sidelink
//! configuration (TS 38.331 §5.8.3, §6.2.2, §6.3.5).
//!
//! # Why this module exists (issue #141)
//!
//! `grep SidelinkUEInformation` over this tree returned nothing. A UE with the sidelink
//! feature on requested no sidelink resources, and a gNB therefore allocated none: PC5
//! transmissions were modelled entirely UE-local, with the network unaware sidelink was
//! in use at all. TS 38.331 §5.8.3.2 makes this the UE's obligation — it "shall initiate
//! the procedure ... to indicate it is interested to receive or transmit NR sidelink
//! communication".
//!
//! # The blocker this issue recorded is DEAD, and the measurement is recorded here
//!
//! Issue #141 was parked `decision`/`needs-human` on a verified blocker: the vendored
//! RRC schema was `tools/rrc-15.6.0.asn1`, Rel-15, and sidelink is a Rel-16 feature, so
//! **no sidelink IE existed to encode**. That was true when it was written.
//!
//! Issue #105 (PR #184) then vendored `tools/rrc-19.3.0.asn1`, Rel-19. The schema now
//! carries 1095 lines matching `SL-`/`sl-Config`/`SidelinkUEInformation`, and all three
//! types this module needs are present AND generated:
//! `SidelinkUEInformationNR_r16`, `SL_ConfigDedicatedNR_r16` and the
//! `RRCReconfiguration_v1610_IEs.sl_config_dedicated_nr_r16` field that carries it. So
//! the blocker is gone and this module is what the schema upgrade unblocked.
//!
//! # Both encode paths were probed BEFORE this was written, and both are clear
//!
//! Two known ceilings in the vendored codec and compiler could have blocked this, and
//! neither does. Recorded because the *reason* is non-obvious and a future agent
//! extending this module will hit them:
//!
//! **1. The extended-SEQUENCE encode gap does not bite.** `vendor/asn1-codecs`'s
//! `encode_sequence_header_common` still returns `EncodeNotSupported` for an extended
//! SEQUENCE carrying an extension addition (issue #117 implemented the extended-CHOICE
//! case only). `SL-ConfigDedicatedNR-r16` **is** an extensible SEQUENCE — it has `...`
//! and five `[[ ]]` addition groups. But every field in every one of those groups is
//! `OPTIONAL`, so a root-only value writes extension bit 0 and never reaches the
//! unimplemented path. Measured: a `SL_ConfigDedicatedNR_r16` carrying only `t400-r16`
//! encodes to `[0x02, 0x80]` — bit 0 clear (no extension), then the 6-bit optional
//! bitmap `000001`, then the 3-bit `t400` index. This is the same shape the #56 agent
//! found for `SIB19-r17`.
//!
//! **2. `SidelinkUEInformationNR-r16` rides a ROOT arm, not an extension arm.** In
//! `UL-DCCH-MessageType`, it is reached through `messageClassExtension.c2` — and within
//! `c2` it is `key = 4, extended = false`, a root arm of a non-extensible CHOICE. So it
//! encodes by exactly the same machinery that already carries `dedicatedSIBRequest-r16`
//! in `dcch_dispatch`. No vendored change was needed for this module, and none was made.
//!
//! **The ceiling that DOES remain**, stated so it is not rediscovered as a bug: the
//! vendored `asn1-compiler` silently drops SEQUENCE extension-addition fields (the same
//! class of silent drop that issue #185 fixed for information-object-class fields). So
//! `SL-ConfigDedicatedNR-r16`'s Rel-17/18/19 additions — `sl-DiscConfig-r17` and the
//! `sl-DiscConfig-v18xx` chain — **do not exist in the generated struct at all**. A
//! network wanting to signal PC5 discovery resources through RRC cannot, here. That does
//! not affect this module: the PC5 discovery of TS 23.304 §6.3.1.2 is a PC5-S procedure
//! (see `nextgsim-ue::sidelink::discovery`) and runs without RRC-signalled discovery
//! pools, using the UE-autonomous resource selection of Mode 2.
//!
//! # What is signalled, and why only this much
//!
//! The request carries `sl-RxInterestedFreqList` and `sl-TxResourceReqList`, which are
//! the two IEs §5.8.3.3 makes the substance of the procedure: which carriers the UE
//! wants to receive on, and per destination, what it wants to transmit. The
//! configuration carries `sl-PHY-MAC-RLC-Config` and `t400`.
//!
//! Not signalled: `sl-FailureList`, `sl-MeasConfigInfo*`, `sl-RadioBearer*`. Each would
//! be an IE no code in this tree reads — there is no PC5 measurement reporting, no PC5
//! RLF detection and no SLRB manager to configure — and an unread IE is exactly the
//! defect issue #141 exists to remove.

use crate::codec::generated::*;
use crate::codec::{decode_rrc, encode_rrc, RrcCodecError};
use thiserror::Error;

/// Errors building or reading a sidelink RRC message.
#[derive(Debug, Error, PartialEq)]
pub enum SidelinkRrcError {
    /// A field value TS 38.331 does not allow.
    #[error("invalid sidelink field: {0}")]
    InvalidFieldValue(String),
    /// The codec refused the message.
    #[error("sidelink RRC codec error: {0}")]
    Codec(String),
}

impl From<RrcCodecError> for SidelinkRrcError {
    fn from(e: RrcCodecError) -> Self {
        Self::Codec(format!("{e}"))
    }
}

/// `maxNrofFreqSL-r16` is 8, and `SL-InterestedFreqList-r16` entries are
/// `INTEGER (1..maxNrofFreqSL-r16)` — so a carrier index is 1-based, not 0-based.
///
/// Named because an off-by-one here produces a value that encodes fine and means the
/// wrong carrier.
const SL_FREQ_INDEX_RANGE: std::ops::RangeInclusive<u8> = 1..=8;

/// `maxNrofSL-Dest-r16` is 32: the most destinations one `sl-TxResourceReqList` may
/// carry.
const MAX_SL_DESTINATIONS: usize = 32;

/// A `SL-DestinationIdentity-r16` is a `BIT STRING (SIZE (24))` — the 24-bit ProSe
/// Layer-2 ID of TS 23.304 §5.8.2.1, as RRC carries it.
const SL_DESTINATION_IDENTITY_BITS: usize = 24;

/// What cast type a sidelink transmission-resource request is for
/// (`SL-TxResourceReq-r16.sl-CastType-r16`, TS 38.331 §6.3.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlCastType {
    /// Broadcast.
    Broadcast,
    /// Groupcast.
    Groupcast,
    /// Unicast — what a PC5 unicast link established per TS 23.304 §6.4.3.1 needs.
    Unicast,
}

impl SlCastType {
    /// The generated ENUMERATED index this cast type encodes to.
    fn to_asn(self) -> SL_TxResourceReq_r16Sl_CastType_r16 {
        SL_TxResourceReq_r16Sl_CastType_r16(match self {
            Self::Broadcast => SL_TxResourceReq_r16Sl_CastType_r16::BROADCAST,
            Self::Groupcast => SL_TxResourceReq_r16Sl_CastType_r16::GROUPCAST,
            Self::Unicast => SL_TxResourceReq_r16Sl_CastType_r16::UNICAST,
        })
    }

    /// Reads the cast type back.
    ///
    /// `spare1` maps to [`SlCastType::Broadcast`]: the value is reserved and has no
    /// meaning yet, and treating it as the least-privileged cast type is safer than
    /// failing the whole request over a spare code point a future release may assign.
    fn from_asn(value: &SL_TxResourceReq_r16Sl_CastType_r16) -> Self {
        match value.0 {
            SL_TxResourceReq_r16Sl_CastType_r16::UNICAST => Self::Unicast,
            SL_TxResourceReq_r16Sl_CastType_r16::GROUPCAST => Self::Groupcast,
            _ => Self::Broadcast,
        }
    }
}

/// One destination's sidelink transmission-resource request
/// (`SL-TxResourceReq-r16`, TS 38.331 §6.3.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlTxResourceRequest {
    /// The peer's 24-bit ProSe Layer-2 ID (TS 23.304 §5.8.2.1).
    ///
    /// The SAME identity the PC5-S handshake uses, deliberately: the whole point of the
    /// request is to tell the network which destination the UE is about to transmit to,
    /// and a second identity space would make the RRC request unrelatable to the PC5
    /// link it is for.
    pub destination_l2_id: u32,
    /// Which cast type those resources are for.
    pub cast_type: SlCastType,
}

/// The sidelink resource request a UE sends (TS 38.331 §5.8.3, `SidelinkUEInformationNR`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SidelinkUeInformationParams {
    /// `sl-RxInterestedFreqList-r16`: the carrier indices the UE wants to receive
    /// sidelink on, each `1..=8`.
    ///
    /// Empty means the UE is not interested in receiving — which §5.8.3.3 signals by
    /// *omitting* the IE, not by sending an empty list (the ASN.1 lower bound is 1).
    pub rx_interested_freqs: Vec<u8>,
    /// `sl-TxResourceReqList-r16`: one entry per destination the UE wants to transmit
    /// to.
    pub tx_resource_requests: Vec<SlTxResourceRequest>,
}

/// The network's dedicated sidelink configuration
/// (TS 38.331 §6.3.2 `SL-ConfigDedicatedNR-r16`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SlConfigDedicatedParams {
    /// `t400-r16`: the sidelink RRC reconfiguration guard timer, in milliseconds.
    ///
    /// Only the eight values the ENUMERATED allows are signallable: 100, 200, 300, 400,
    /// 600, 1000, 1500, 2000. Anything else is refused at the builder rather than
    /// silently rounded, because a timer that is not the one the network configured is
    /// worse than no timer.
    pub t400_ms: Option<u16>,
    /// Whether to include an `sl-PHY-MAC-RLC-Config-r16`.
    ///
    /// The config is included empty (every field within it is `OPTIONAL` and this tree
    /// has no PC5 scheduler to configure), so it carries presence and nothing more.
    /// Present because its presence is what tells the UE the network has granted
    /// sidelink at all: TS 38.331 §5.3.5.3 has the UE apply `sl-ConfigDedicatedNR` on
    /// reception, and a `setup` with no PHY/MAC/RLC config is the "sidelink is
    /// permitted, select your own Mode 2 resources" case.
    pub phy_mac_rlc_config: bool,
}

/// The eight `t400-r16` values TS 38.331 allows, in ENUMERATED index order.
///
/// A table rather than arithmetic because the values are not evenly spaced — there is no
/// `ms500`, and the step changes from 100 to 200 to 400 to 500 — so any formula would be
/// wrong somewhere.
const T400_VALUES_MS: [u16; 8] = [100, 200, 300, 400, 600, 1000, 1500, 2000];

/// Builds a `SidelinkUEInformationNR` as a complete UL-DCCH message
/// (TS 38.331 §5.8.3, §6.2.1).
///
/// The message rides `messageClassExtension.c2`, arm 4 — a ROOT arm, which is why this
/// encodes at all (see the module docs).
pub fn build_sidelink_ue_information(
    params: &SidelinkUeInformationParams,
) -> Result<UL_DCCH_Message, SidelinkRrcError> {
    for freq in &params.rx_interested_freqs {
        if !SL_FREQ_INDEX_RANGE.contains(freq) {
            return Err(SidelinkRrcError::InvalidFieldValue(format!(
                "sl-RxInterestedFreqList entry {freq} is outside 1..=8"
            )));
        }
    }
    if params.tx_resource_requests.len() > MAX_SL_DESTINATIONS {
        return Err(SidelinkRrcError::InvalidFieldValue(format!(
            "sl-TxResourceReqList has {} entries, more than maxNrofSL-Dest-r16 ({MAX_SL_DESTINATIONS})",
            params.tx_resource_requests.len()
        )));
    }

    // An empty list is OMITTED rather than encoded, because the ASN.1 size constraint is
    // `SIZE (1..maxNrof...)`: a zero-length SEQUENCE OF is not a legal value, and
    // encoding one would produce a PDU no conformant peer can decode.
    let rx_list = if params.rx_interested_freqs.is_empty() {
        None
    } else {
        Some(SL_InterestedFreqList_r16(
            params
                .rx_interested_freqs
                .iter()
                .map(|f| SL_InterestedFreqList_r16_Entry(*f))
                .collect(),
        ))
    };

    let tx_list = if params.tx_resource_requests.is_empty() {
        None
    } else {
        Some(SL_TxResourceReqList_r16(
            params
                .tx_resource_requests
                .iter()
                .map(|req| SL_TxResourceReq_r16 {
                    sl_destination_identity_r16: destination_identity(req.destination_l2_id),
                    sl_cast_type_r16: req.cast_type.to_asn(),
                    sl_rlc_mode_indication_list_r16: None,
                    sl_qo_s_info_list_r16: None,
                    sl_type_tx_sync_list_r16: None,
                    sl_tx_interested_freq_list_r16: None,
                    sl_capability_information_sidelink_r16: None,
                })
                .collect(),
        ))
    };

    let ies = SidelinkUEInformationNR_r16_IEs {
        sl_rx_interested_freq_list_r16: rx_list,
        sl_tx_resource_req_list_r16: tx_list,
        sl_failure_list_r16: None,
        late_non_critical_extension: None,
        non_critical_extension: None,
    };

    Ok(UL_DCCH_Message {
        message: UL_DCCH_MessageType::MessageClassExtension(
            UL_DCCH_MessageType_messageClassExtension::C2(
                UL_DCCH_MessageType_messageClassExtension_c2::SidelinkUEInformationNR_r16(
                    SidelinkUEInformationNR_r16 {
                        critical_extensions:
                            SidelinkUEInformationNR_r16CriticalExtensions::SidelinkUEInformationNR_r16(
                                ies,
                            ),
                    },
                ),
            ),
        ),
    })
}

/// The 24-bit `SL-DestinationIdentity-r16` bit string for a ProSe Layer-2 ID.
///
/// Most significant bit first, which is what `Msb0` ordering and a
/// `BIT STRING (SIZE (24))` together mean: bit 0 of the encoding is bit 23 of the ID.
fn destination_identity(l2_id: u32) -> SL_DestinationIdentity_r16 {
    let mut bits =
        bitvec::vec::BitVec::<u8, bitvec::order::Msb0>::repeat(false, SL_DESTINATION_IDENTITY_BITS);
    for i in 0..SL_DESTINATION_IDENTITY_BITS {
        // Bit i of the string is bit (23 - i) of the value.
        let shift = SL_DESTINATION_IDENTITY_BITS - 1 - i;
        bits.set(i, (l2_id >> shift) & 1 == 1);
    }
    SL_DestinationIdentity_r16(bits)
}

/// Reads a `SL-DestinationIdentity-r16` back to a ProSe Layer-2 ID.
///
/// A bit string of the wrong length yields `None` rather than a truncated ID: a
/// destination identity that is not 24 bits is not a Layer-2 ID, and silently
/// reinterpreting it would address the wrong peer.
fn read_destination_identity(id: &SL_DestinationIdentity_r16) -> Option<u32> {
    if id.0.len() != SL_DESTINATION_IDENTITY_BITS {
        return None;
    }
    let mut value = 0u32;
    for bit in id.0.iter() {
        value = (value << 1) | u32::from(*bit);
    }
    Some(value)
}

/// Reads a `SidelinkUEInformationNR` back out of a UL-DCCH message
/// (TS 38.331 §5.8.3).
///
/// `None` for any UL-DCCH message that is not one — including a `c2` message of another
/// type — rather than an error, because the gNB's dispatcher sees every uplink message
/// and a `measurementReport` is not a malformed sidelink request.
pub fn read_sidelink_ue_information(msg: &UL_DCCH_Message) -> Option<SidelinkUeInformationParams> {
    let UL_DCCH_MessageType::MessageClassExtension(UL_DCCH_MessageType_messageClassExtension::C2(
        UL_DCCH_MessageType_messageClassExtension_c2::SidelinkUEInformationNR_r16(sui),
    )) = &msg.message
    else {
        return None;
    };
    let SidelinkUEInformationNR_r16CriticalExtensions::SidelinkUEInformationNR_r16(ies) =
        &sui.critical_extensions
    else {
        // `criticalExtensionsFuture`: a later release's encoding this decoder cannot
        // read. Not an error — the message is well-formed, just not for us.
        return None;
    };

    Some(SidelinkUeInformationParams {
        rx_interested_freqs: ies
            .sl_rx_interested_freq_list_r16
            .as_ref()
            .map(|list| list.0.iter().map(|e| e.0).collect())
            .unwrap_or_default(),
        tx_resource_requests: ies
            .sl_tx_resource_req_list_r16
            .as_ref()
            .map(|list| {
                list.0
                    .iter()
                    .filter_map(|req| {
                        Some(SlTxResourceRequest {
                            destination_l2_id: read_destination_identity(
                                &req.sl_destination_identity_r16,
                            )?,
                            cast_type: SlCastType::from_asn(&req.sl_cast_type_r16),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default(),
    })
}

/// Builds an `SL-ConfigDedicatedNR-r16` (TS 38.331 §6.3.2).
///
/// Only root fields are set, which is what keeps this inside the vendored codec's
/// encodable subset — see the module docs for the measurement.
pub fn build_sl_config_dedicated(
    params: &SlConfigDedicatedParams,
) -> Result<SL_ConfigDedicatedNR_r16, SidelinkRrcError> {
    let t400 = match params.t400_ms {
        None => None,
        Some(ms) => {
            let index = T400_VALUES_MS
                .iter()
                .position(|v| *v == ms)
                .ok_or_else(|| {
                    SidelinkRrcError::InvalidFieldValue(format!(
                        "t400 {ms} ms is not one of the signallable values {T400_VALUES_MS:?}"
                    ))
                })?;
            Some(SL_ConfigDedicatedNR_r16T400_r16(index as u8))
        }
    };

    Ok(SL_ConfigDedicatedNR_r16 {
        sl_phy_mac_rlc_config_r16: if params.phy_mac_rlc_config {
            Some(empty_phy_mac_rlc_config())
        } else {
            None
        },
        sl_radio_bearer_to_release_list_r16: None,
        sl_radio_bearer_to_add_mod_list_r16: None,
        sl_meas_config_info_to_release_list_r16: None,
        sl_meas_config_info_to_add_mod_list_r16: None,
        t400_r16: t400,
    })
}

/// An `SL-PHY-MAC-RLC-Config-r16` with every optional field absent.
///
/// Its presence is the signal (see [`SlConfigDedicatedParams::phy_mac_rlc_config`]);
/// there is no PC5 scheduler in this tree for its contents to configure, and inventing
/// scheduling pools no code reads is the defect issue #141 removes.
fn empty_phy_mac_rlc_config() -> SL_PHY_MAC_RLC_Config_r16 {
    SL_PHY_MAC_RLC_Config_r16 {
        sl_scheduled_config_r16: None,
        sl_ue_selected_config_r16: None,
        sl_freq_info_to_release_list_r16: None,
        sl_freq_info_to_add_mod_list_r16: None,
        sl_rlc_bearer_to_release_list_r16: None,
        sl_rlc_bearer_to_add_mod_list_r16: None,
        sl_max_num_consecutive_dtx_r16: None,
        sl_csi_acquisition_r16: None,
        sl_csi_scheduling_request_id_r16: None,
        sl_ssb_priority_nr_r16: None,
        network_controlled_sync_tx_r16: None,
    }
}

/// Reads an `SL-ConfigDedicatedNR-r16` back to the parameters that built it.
pub fn read_sl_config_dedicated(config: &SL_ConfigDedicatedNR_r16) -> SlConfigDedicatedParams {
    SlConfigDedicatedParams {
        t400_ms: config
            .t400_r16
            .as_ref()
            .and_then(|t| T400_VALUES_MS.get(usize::from(t.0)).copied()),
        phy_mac_rlc_config: config.sl_phy_mac_rlc_config_r16.is_some(),
    }
}

/// Encodes a `SidelinkUEInformationNR` to the UPER bytes that leave the UE.
pub fn encode_sidelink_ue_information(
    params: &SidelinkUeInformationParams,
) -> Result<Vec<u8>, SidelinkRrcError> {
    Ok(encode_rrc(&build_sidelink_ue_information(params)?)?)
}

/// Decodes the bytes a gNB received, returning the request if they carry one.
pub fn decode_sidelink_ue_information(
    bytes: &[u8],
) -> Result<Option<SidelinkUeInformationParams>, SidelinkRrcError> {
    let msg: UL_DCCH_Message = decode_rrc(bytes)?;
    Ok(read_sidelink_ue_information(&msg))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A destination Layer-2 ID with a bit set in every octet, so a byte-order slip or a
    /// sign extension cannot produce it by accident.
    const PEER_L2_ID: u32 = 0x00_A5_3C_71 & 0x00FF_FFFF;

    /// The whole criterion-3 round trip through the REAL generated codec and real UPER
    /// bytes: the UE builds a request, it is encoded, and the values come back out.
    ///
    /// This is the test the issue's blocker made impossible to write: before #105 there
    /// was no `SidelinkUEInformationNR` type in the generated tree at all.
    #[test]
    fn a_sidelink_ue_information_round_trips_through_real_uper() {
        let params = SidelinkUeInformationParams {
            rx_interested_freqs: vec![1, 3, 8],
            tx_resource_requests: vec![
                SlTxResourceRequest {
                    destination_l2_id: PEER_L2_ID,
                    cast_type: SlCastType::Unicast,
                },
                SlTxResourceRequest {
                    destination_l2_id: 0x00_00_00_02,
                    cast_type: SlCastType::Groupcast,
                },
            ],
        };

        let bytes = encode_sidelink_ue_information(&params).expect("encode");
        assert!(
            !bytes.is_empty(),
            "a SidelinkUEInformationNR must produce bytes"
        );

        let decoded = decode_sidelink_ue_information(&bytes)
            .expect("decode")
            .expect("the bytes carry a SidelinkUEInformationNR");

        // Positive assertions on values only reachable by having decoded real UPER.
        assert_eq!(decoded.rx_interested_freqs, vec![1, 3, 8]);
        assert_eq!(decoded.tx_resource_requests.len(), 2);
        assert_eq!(
            decoded.tx_resource_requests[0].destination_l2_id,
            PEER_L2_ID
        );
        assert_eq!(
            decoded.tx_resource_requests[0].cast_type,
            SlCastType::Unicast
        );
        assert_eq!(
            decoded.tx_resource_requests[1].cast_type,
            SlCastType::Groupcast
        );
        assert_eq!(decoded, params);
    }

    /// The 24-bit destination identity survives the bit string, most significant bit
    /// first. A reversed order would still round-trip through THIS codec, so the octets
    /// are pinned rather than only the value.
    #[test]
    fn a_destination_identity_is_24_bits_most_significant_first() {
        let id = destination_identity(0x00_A5_3C_71 & 0x00FF_FFFF);
        assert_eq!(id.0.len(), 24);
        // 0xA53C71 = 1010 0101 0011 1100 0111 0001
        let bits: Vec<bool> = id.0.iter().map(|b| *b).collect();
        assert_eq!(
            &bits[0..8],
            &[true, false, true, false, false, true, false, true]
        );
        assert_eq!(
            read_destination_identity(&id),
            Some(0x00_A5_3C_71 & 0x00FF_FFFF)
        );
    }

    /// A bit string that is not 24 bits is refused rather than reinterpreted, so a
    /// malformed identity cannot address the wrong peer.
    #[test]
    fn a_destination_identity_of_the_wrong_length_is_refused() {
        let short = SL_DestinationIdentity_r16(
            bitvec::vec::BitVec::<u8, bitvec::order::Msb0>::repeat(true, 16),
        );
        assert_eq!(read_destination_identity(&short), None);
    }

    /// An empty interest list omits the IE rather than encoding a zero-length
    /// SEQUENCE OF, which the `SIZE (1..8)` constraint forbids.
    #[test]
    fn an_empty_request_omits_both_lists_rather_than_encoding_empty_ones() {
        let params = SidelinkUeInformationParams::default();
        let bytes = encode_sidelink_ue_information(&params).expect("encode");
        let decoded = decode_sidelink_ue_information(&bytes)
            .expect("decode")
            .expect("still a SidelinkUEInformationNR");
        assert!(decoded.rx_interested_freqs.is_empty());
        assert!(decoded.tx_resource_requests.is_empty());
    }

    /// A carrier index outside `1..=8` is refused at the builder, not encoded to a
    /// value that means a different carrier.
    #[test]
    fn a_carrier_index_outside_the_asn1_range_is_refused() {
        for bad in [0u8, 9, 255] {
            let params = SidelinkUeInformationParams {
                rx_interested_freqs: vec![bad],
                tx_resource_requests: Vec::new(),
            };
            assert!(
                build_sidelink_ue_information(&params).is_err(),
                "carrier index {bad} must be refused"
            );
        }
        // And the boundary values ARE allowed.
        for good in [1u8, 8] {
            let params = SidelinkUeInformationParams {
                rx_interested_freqs: vec![good],
                tx_resource_requests: Vec::new(),
            };
            assert!(build_sidelink_ue_information(&params).is_ok());
        }
    }

    /// More destinations than `maxNrofSL-Dest-r16` is refused rather than truncated.
    #[test]
    fn more_destinations_than_the_asn1_maximum_are_refused() {
        let params = SidelinkUeInformationParams {
            rx_interested_freqs: Vec::new(),
            tx_resource_requests: (0..=MAX_SL_DESTINATIONS as u32)
                .map(|i| SlTxResourceRequest {
                    destination_l2_id: i,
                    cast_type: SlCastType::Unicast,
                })
                .collect(),
        };
        assert!(build_sidelink_ue_information(&params).is_err());
    }

    /// `sl-ConfigDedicatedNR` round trips through real UPER — and this is the case the
    /// module docs' encode probe covers: an extensible SEQUENCE whose every extension
    /// addition is OPTIONAL encodes root-only without reaching the vendored codec's
    /// unimplemented extended-SEQUENCE path.
    #[test]
    fn an_sl_config_dedicated_round_trips_through_real_uper() {
        let params = SlConfigDedicatedParams {
            t400_ms: Some(400),
            phy_mac_rlc_config: true,
        };
        let config = build_sl_config_dedicated(&params).expect("build");
        let bytes = encode_rrc(&config).expect(
            "an extensible SEQUENCE with only root fields set must encode; if this \
             fails, the vendored codec's extended-SEQUENCE gap has started to bite",
        );
        let back: SL_ConfigDedicatedNR_r16 = decode_rrc(&bytes).expect("decode");
        assert_eq!(read_sl_config_dedicated(&back), params);
        assert_eq!(read_sl_config_dedicated(&back).t400_ms, Some(400));
    }

    /// The exact bytes of a minimal `SL-ConfigDedicatedNR-r16`, pinned.
    ///
    /// This is the measurement the module docs quote, and pinning it is what makes the
    /// claim "the extension bit is 0, so the unimplemented path is never reached"
    /// checkable rather than assertible. If the vendored codec ever starts emitting an
    /// extension bit here, this test fails and says so.
    #[test]
    fn a_root_only_sl_config_writes_extension_bit_zero() {
        let config = build_sl_config_dedicated(&SlConfigDedicatedParams {
            t400_ms: Some(300),
            phy_mac_rlc_config: false,
        })
        .expect("build");
        let bytes = encode_rrc(&config).expect("encode");
        // bit 0 = extension marker, and it must be clear.
        assert_eq!(
            bytes[0] & 0b1000_0000,
            0,
            "the extension bit is set, so this value would reach the vendored codec's \
             unimplemented extended-SEQUENCE encoder; bytes = {bytes:02x?}"
        );
        // t400 index 2 = 300 ms, the third of the eight values.
        assert_eq!(bytes, vec![0x02, 0x80]);
    }

    /// Every one of the eight signallable `t400` values round trips to itself, and
    /// nothing else is accepted. The table is not evenly spaced, so an index/value
    /// formula would be wrong somewhere and this is what would catch it.
    #[test]
    fn every_signallable_t400_round_trips_and_others_are_refused() {
        for ms in T400_VALUES_MS {
            let config = build_sl_config_dedicated(&SlConfigDedicatedParams {
                t400_ms: Some(ms),
                phy_mac_rlc_config: false,
            })
            .expect("build");
            let bytes = encode_rrc(&config).expect("encode");
            let back: SL_ConfigDedicatedNR_r16 = decode_rrc(&bytes).expect("decode");
            assert_eq!(
                read_sl_config_dedicated(&back).t400_ms,
                Some(ms),
                "t400 {ms} ms did not survive the round trip"
            );
        }
        // 500 ms looks plausible and is NOT in the table.
        assert!(build_sl_config_dedicated(&SlConfigDedicatedParams {
            t400_ms: Some(500),
            phy_mac_rlc_config: false,
        })
        .is_err());
    }

    /// The `sl-PHY-MAC-RLC-Config` presence bit is what tells the UE sidelink was
    /// granted, so present and absent must be distinguishable after a round trip.
    #[test]
    fn the_phy_mac_rlc_config_presence_survives_the_round_trip() {
        for present in [true, false] {
            let config = build_sl_config_dedicated(&SlConfigDedicatedParams {
                t400_ms: None,
                phy_mac_rlc_config: present,
            })
            .expect("build");
            let bytes = encode_rrc(&config).expect("encode");
            let back: SL_ConfigDedicatedNR_r16 = decode_rrc(&bytes).expect("decode");
            assert_eq!(read_sl_config_dedicated(&back).phy_mac_rlc_config, present);
        }
    }

    /// A UL-DCCH message that is not a sidelink request reads as `None`, not as an
    /// error: the gNB's dispatcher sees every uplink message.
    #[test]
    fn another_ul_dcch_message_is_not_read_as_a_sidelink_request() {
        let other = UL_DCCH_Message {
            message: UL_DCCH_MessageType::MessageClassExtension(
                UL_DCCH_MessageType_messageClassExtension::C2(
                    UL_DCCH_MessageType_messageClassExtension_c2::DedicatedSIBRequest_r16(
                        DedicatedSIBRequest_r16 {
                            critical_extensions:
                                DedicatedSIBRequest_r16CriticalExtensions::CriticalExtensionsFuture(
                                    DedicatedSIBRequest_r16CriticalExtensions_criticalExtensionsFuture {},
                                ),
                        },
                    ),
                ),
            ),
        };
        assert_eq!(read_sidelink_ue_information(&other), None);
        // And through the bytes, so the dispatcher's real input is covered.
        let bytes = encode_rrc(&other).expect("encode");
        assert_eq!(
            decode_sidelink_ue_information(&bytes).expect("decode"),
            None
        );
    }
}
