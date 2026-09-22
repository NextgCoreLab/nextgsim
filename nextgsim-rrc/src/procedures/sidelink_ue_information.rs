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
//!
//! # What issue #190 added here
//!
//! Three fields, each of which was generated and unused before it:
//!
//! * **`ue-Type-r17`** on the request ([`SidelinkUeInformationParams::ue_type`]), in the
//!   `v1700` non-critical extension. This is how a UE asks for L2 UE-to-Network relay
//!   resources, and it is what gives the gNB's relay grant a production trigger instead of
//!   a network-side configuration flag.
//! * **`sl-ScheduledConfig-r16.sl-RNTI-r16`** on the grant
//!   ([`SlConfigDedicatedParams::sl_rnti`]) — Mode 1, but the *identity* only. The
//!   resource pools stay absent for #141's unchanged reason; that field's docs carry the
//!   full decision, which issue #190's criterion 3 asked to be made explicitly.
//! * **`sl-RLC-BearerToAddModList-r16`** on the grant
//!   ([`SlConfigDedicatedParams::sl_rlc_bearers`]), the relay RLC channels a relay's SRAP
//!   mappings may name. Before #190 `grep -rn sl_rlc_bearer_to_add_mod_list_r16` over the
//!   tree found exactly one hit: the literal `None` in this file.
//!
//! All three ride plain `OPTIONAL` fields of non-critical extensions rather than `[[ ]]`
//! extension additions, so none of them reaches the vendored codec's unimplemented
//! extended-SEQUENCE encoder. See `sidelink_relay_config`'s module docs for the measured
//! bytes.

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

/// What role a UE declares for itself in L2 UE-to-Network relay
/// (`SidelinkUEInformationNR-v1700-IEs.ue-Type-r17`, TS 38.331 §6.2.2; issue #190).
///
/// This is how the gNB learns a UE wants relay resources at all, and it is what makes the
/// relay grant *requested* rather than configured out of band: without it the gNB would
/// have to guess which of its UEs is a relay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlUeType {
    /// This UE is offering itself as an L2 UE-to-Network relay.
    RelayUe,
    /// This UE wants to reach the network through a relay.
    RemoteUe,
}

impl SlUeType {
    /// The generated ENUMERATED index this role encodes to.
    fn to_asn(self) -> SidelinkUEInformationNR_v1700_IEsUe_Type_r17 {
        SidelinkUEInformationNR_v1700_IEsUe_Type_r17(match self {
            Self::RelayUe => SidelinkUEInformationNR_v1700_IEsUe_Type_r17::RELAY_UE,
            Self::RemoteUe => SidelinkUEInformationNR_v1700_IEsUe_Type_r17::REMOTE_UE,
        })
    }

    /// Reads the role back.
    ///
    /// Anything that is not `remoteUE` reads as `RelayUe`, matching the ENUMERATED's two
    /// code points. The ASN.1 is not extensible here, so there is no third value a
    /// conformant peer can send.
    fn from_asn(value: &SidelinkUEInformationNR_v1700_IEsUe_Type_r17) -> Self {
        match value.0 {
            SidelinkUEInformationNR_v1700_IEsUe_Type_r17::REMOTE_UE => Self::RemoteUe,
            _ => Self::RelayUe,
        }
    }
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
    /// `ue-Type-r17`: the L2 UE-to-Network relay role this UE declares (issue #190).
    ///
    /// `None` is a plain PC5 UE, which is what issue #141's requests were and still are.
    /// `Some` is what asks the gNB for relay resources, and is why the gNB's relay grant
    /// has a production trigger rather than needing a configuration flag on the network
    /// side.
    ///
    /// Carried in the `v1700` non-critical extension, so a request that sets it builds
    /// that extension and a request that does not is byte-for-byte what #141 sent.
    pub ue_type: Option<SlUeType>,
}

/// One sidelink relay RLC channel the network configures on the relay's own Uu interface
/// (`SL-RLC-BearerConfig-r16`, TS 38.331 §6.3.2; issue #190).
///
/// This is the IE issue #190's criterion 2 names. Before that issue it was generated and
/// unused: `grep -rn sl_rlc_bearer_to_add_mod_list_r16` found one hit, the literal `None`
/// in the old `empty_phy_mac_rlc_config`. A relay needs it because TS 38.300 §16.12.2.1 maps
/// remote-UE bearers onto *the relay's own* relay RLC channels, and the relay cannot know
/// which channels exist unless the network says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlRlcBearerConfig {
    /// `sl-RLC-BearerConfigIndex-r16`, which identifies this bearer configuration.
    pub index: u16,
    /// `sl-ServedRadioBearer-r16`: the `SLRB-Uu-ConfigIndex-r16` this RLC channel serves.
    pub served_radio_bearer: Option<u16>,
}

/// The network's dedicated sidelink configuration
/// (TS 38.331 §6.3.2 `SL-ConfigDedicatedNR-r16`).
///
/// No longer `Copy` since issue #190: [`Self::sl_rlc_bearers`] is a list, and the
/// alternative — a fixed-size array sized to `maxSL-LCID-r16` — would carry 512 slots to
/// describe the one or two channels a relay actually gets.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
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
    /// Its presence is what tells the UE the network has granted sidelink at all:
    /// TS 38.331 §5.3.5.3 has the UE apply `sl-ConfigDedicatedNR` on reception, and a
    /// `setup` whose PHY/MAC/RLC config carries neither a scheduled config nor any RLC
    /// bearer is the "sidelink is permitted, select your own Mode 2 resources" case.
    pub phy_mac_rlc_config: bool,
    /// `sl-ScheduledConfig-r16.sl-RNTI-r16`: the SL-RNTI for **Mode 1** scheduled
    /// sidelink resource allocation (TS 38.300 §16.9.3.2; issue #190's criterion 3).
    ///
    /// # The explicit decision criterion 3 asks for
    ///
    /// Issue #141 granted Mode 2 only, and was right to: `sl-ScheduledConfig-r16`'s
    /// **resource pools** (`sl-ConfiguredGrantConfigList-r16`) would describe slots in a
    /// PC5 physical layer this simulator does not have, and signalling a pool nothing
    /// consults is the unread-IE defect #141 exists to remove.
    ///
    /// That reasoning applies to the *pools*, and **not** to the SL-RNTI. An RNTI is an
    /// identity, not a slot: it is how the gNB names this UE when it schedules sidelink,
    /// and it is meaningful with or without a PHY — the same way a C-RNTI is meaningful in
    /// this tree. So this carries the SL-RNTI and **not** a configured-grant list, and
    /// `sl-PSFCH-ToPUCCH` and `mac-MainConfigSL` stay absent for the pool reason.
    ///
    /// `None` keeps #141's Mode-2 behaviour, which is still right for a plain PC5 UE.
    /// `Some` is the relay case: TS 38.300 §16.12.2.1 has the relay carry remote-UE
    /// traffic on channels the *network* configured, so a relay whose resources were
    /// UE-autonomous would be selecting its own resources for traffic the network is
    /// scheduling.
    pub sl_rnti: Option<u16>,
    /// `sl-RLC-BearerToAddModList-r16`: the relay RLC channels on this UE's own interface
    /// (issue #190's criterion 2).
    ///
    /// Empty for a plain PC5 UE, which needs no relay RLC channel. For a relay, these are
    /// the channels its SRAP bearer mappings may name — and the UE **validates** its SRAP
    /// mappings against them rather than trusting the mapping alone, so a mapping naming a
    /// channel the network never configured is rejected rather than used.
    pub sl_rlc_bearers: Vec<SlRlcBearerConfig>,
}

/// The eight `t400-r16` values TS 38.331 allows, in ENUMERATED index order.
///
/// A table rather than arithmetic because the values are not evenly spaced — there is no
/// `ms500`, and the step changes from 100 to 200 to 400 to 500 — so any formula would be
/// wrong somewhere.
const T400_VALUES_MS: [u16; 8] = [100, 200, 300, 400, 600, 1000, 1500, 2000];

/// `SL-RLC-BearerConfigIndex-r16 ::= INTEGER (1..maxSL-LCID-r16)`, and `maxSL-LCID-r16`
/// is 512 (`tools/rrc-19.3.0.asn1`).
const SL_RLC_BEARER_CONFIG_INDEX_RANGE: std::ops::RangeInclusive<u16> = 1..=512;

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
        // The v1700 extension, built only to carry `ue-Type-r17` (issue #190). A request
        // that declares no relay role omits the whole extension, so it stays byte-for-byte
        // what issue #141 sent.
        non_critical_extension: params.ue_type.map(|ue_type| {
            SidelinkUEInformationNR_v1700_IEs {
                sl_tx_resource_req_list_v1700: None,
                sl_rx_drx_report_list_v1700: None,
                sl_rx_interested_gc_bc_dest_list_r17: None,
                sl_rx_interested_freq_list_disc_r17: None,
                sl_tx_resource_req_list_disc_r17: None,
                // `sl-TxResourceReqListCommRelay-r17` carries the per-destination relay
                // transmission request. Absent because this UE asks for relay resources by
                // declaring its ROLE, and the destinations are already in the root
                // `sl-TxResourceReqList-r16` above -- repeating them in a Rel-17 list the
                // gNB would have to reconcile would be two statements of one fact.
                sl_tx_resource_req_list_comm_relay_r17: None,
                ue_type_r17: Some(ue_type.to_asn()),
                sl_source_identity_remote_ue_r17: None,
                non_critical_extension: None,
            }
        }),
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
        // The declared relay role, from the v1700 extension (issue #190). A request that
        // carries no extension yields `None`, which is a plain PC5 UE.
        ue_type: ies
            .non_critical_extension
            .as_ref()
            .and_then(|v1700| v1700.ue_type_r17.as_ref())
            .map(SlUeType::from_asn),
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
            Some(phy_mac_rlc_config(params)?)
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

/// An `SL-PHY-MAC-RLC-Config-r16` carrying whatever this grant configures.
///
/// For a plain PC5 UE that is nothing at all — every field is `OPTIONAL` and its presence
/// alone is the grant (see [`SlConfigDedicatedParams::phy_mac_rlc_config`]). For a relay
/// it carries the Mode-1 SL-RNTI and the relay RLC channels; see
/// [`SlConfigDedicatedParams::sl_rnti`] for why the RNTI is signalled and the resource
/// pools still are not.
fn phy_mac_rlc_config(
    params: &SlConfigDedicatedParams,
) -> Result<SL_PHY_MAC_RLC_Config_r16, SidelinkRrcError> {
    let mut bearers = Vec::with_capacity(params.sl_rlc_bearers.len());
    for bearer in &params.sl_rlc_bearers {
        if !SL_RLC_BEARER_CONFIG_INDEX_RANGE.contains(&bearer.index) {
            return Err(SidelinkRrcError::InvalidFieldValue(format!(
                "sl-RLC-BearerConfigIndex {} is outside 1..={} (maxSL-LCID-r16)",
                bearer.index,
                SL_RLC_BEARER_CONFIG_INDEX_RANGE.end()
            )));
        }
        bearers.push(SL_RLC_BearerConfig_r16 {
            sl_rlc_bearer_config_index_r16: SL_RLC_BearerConfigIndex_r16(bearer.index),
            sl_served_radio_bearer_r16: bearer.served_radio_bearer.map(SLRB_Uu_ConfigIndex_r16),
            // The RLC mode and logical-channel priority of the relay RLC channel. Absent
            // for the reason the resource pools are: there is no PC5 MAC scheduler here to
            // apply a priority, and a configured value nothing reads is an unread IE.
            sl_rlc_config_r16: None,
            sl_mac_logical_channel_config_r16: None,
        });
    }

    Ok(SL_PHY_MAC_RLC_Config_r16 {
        // Mode 1, when the network scheduled this UE's sidelink (issue #190).
        sl_scheduled_config_r16: params.sl_rnti.map(|rnti| {
            SL_PHY_MAC_RLC_Config_r16Sl_ScheduledConfig_r16::Setup(SL_ScheduledConfig_r16 {
                sl_rnti_r16: RNTI_Value(rnti),
                // Deliberately absent -- these are the resource POOLS and the PSFCH
                // timing, which would describe slots in a PC5 physical layer this
                // simulator does not have. See `SlConfigDedicatedParams::sl_rnti`.
                mac_main_config_sl_r16: None,
                sl_cs_rnti_r16: None,
                sl_psfch_to_pucch_r16: None,
                sl_configured_grant_config_list_r16: None,
            })
        }),
        sl_ue_selected_config_r16: None,
        sl_freq_info_to_release_list_r16: None,
        sl_freq_info_to_add_mod_list_r16: None,
        sl_rlc_bearer_to_release_list_r16: None,
        // An empty list is OMITTED rather than encoded: `SIZE (1..maxSL-LCID-r16)` makes a
        // zero-length SEQUENCE OF illegal.
        sl_rlc_bearer_to_add_mod_list_r16: if bearers.is_empty() {
            None
        } else {
            Some(SL_PHY_MAC_RLC_Config_r16Sl_RLC_BearerToAddModList_r16(
                bearers,
            ))
        },
        sl_max_num_consecutive_dtx_r16: None,
        sl_csi_acquisition_r16: None,
        sl_csi_scheduling_request_id_r16: None,
        sl_ssb_priority_nr_r16: None,
        network_controlled_sync_tx_r16: None,
    })
}

/// Reads an `SL-ConfigDedicatedNR-r16` back to the parameters that built it.
pub fn read_sl_config_dedicated(config: &SL_ConfigDedicatedNR_r16) -> SlConfigDedicatedParams {
    let phy_mac_rlc = config.sl_phy_mac_rlc_config_r16.as_ref();
    SlConfigDedicatedParams {
        t400_ms: config
            .t400_r16
            .as_ref()
            .and_then(|t| T400_VALUES_MS.get(usize::from(t.0)).copied()),
        phy_mac_rlc_config: phy_mac_rlc.is_some(),
        // Only a `setup` yields an SL-RNTI: a `release` withdraws Mode 1, which reads back
        // as `None`, the same as never having been granted it. Both mean "this UE selects
        // its own resources", which is the UE's actual behaviour in each case.
        sl_rnti: phy_mac_rlc
            .and_then(|c| c.sl_scheduled_config_r16.as_ref())
            .and_then(|s| match s {
                SL_PHY_MAC_RLC_Config_r16Sl_ScheduledConfig_r16::Setup(cfg) => {
                    Some(cfg.sl_rnti_r16.0)
                }
                SL_PHY_MAC_RLC_Config_r16Sl_ScheduledConfig_r16::Release(_) => None,
            }),
        sl_rlc_bearers: phy_mac_rlc
            .and_then(|c| c.sl_rlc_bearer_to_add_mod_list_r16.as_ref())
            .map(|list| {
                list.0
                    .iter()
                    .map(|b| SlRlcBearerConfig {
                        index: b.sl_rlc_bearer_config_index_r16.0,
                        served_radio_bearer: b.sl_served_radio_bearer_r16.as_ref().map(|s| s.0),
                    })
                    .collect()
            })
            .unwrap_or_default(),
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
            ue_type: None,
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
                ue_type: None,
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
                ue_type: None,
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
            ue_type: None,
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
            ..Default::default()
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

    /// Issue #190's criterion 3: a **Mode-1** grant round trips, carrying the SL-RNTI and
    /// the relay RLC channels a relay's SRAP mappings may name.
    ///
    /// Both values are positive observables only reachable by having decoded real UPER:
    /// before #190 `sl_rlc_bearer_to_add_mod_list_r16` was a hardcoded `None` and
    /// `sl_scheduled_config_r16` was never built at all.
    #[test]
    fn a_mode_1_relay_grant_round_trips_with_its_rnti_and_rlc_bearers() {
        const SL_RNTI: u16 = 0x4602;
        let params = SlConfigDedicatedParams {
            t400_ms: Some(400),
            phy_mac_rlc_config: true,
            sl_rnti: Some(SL_RNTI),
            sl_rlc_bearers: vec![
                SlRlcBearerConfig {
                    index: 1,
                    served_radio_bearer: Some(1),
                },
                SlRlcBearerConfig {
                    index: 2,
                    served_radio_bearer: None,
                },
            ],
        };
        let config = build_sl_config_dedicated(&params).expect("build");
        let bytes = encode_rrc(&config).expect(
            "a Mode-1 grant must encode; if this fails, the vendored codec's \
             extended-SEQUENCE gap has started to bite on SL-ScheduledConfig",
        );
        let back: SL_ConfigDedicatedNR_r16 = decode_rrc(&bytes).expect("decode");
        let read = read_sl_config_dedicated(&back);

        assert_eq!(
            read.sl_rnti,
            Some(SL_RNTI),
            "the SL-RNTI must survive: it is how the gNB names this UE when scheduling \
             sidelink"
        );
        assert_eq!(read.sl_rlc_bearers.len(), 2);
        assert_eq!(read.sl_rlc_bearers[0].index, 1);
        assert_eq!(read.sl_rlc_bearers[0].served_radio_bearer, Some(1));
        assert_eq!(read.sl_rlc_bearers[1].index, 2);
        assert_eq!(read.sl_rlc_bearers[1].served_radio_bearer, None);
        assert_eq!(read, params);
    }

    /// A Mode-2 grant carries NO SL-RNTI and no relay RLC channel, so Mode 1 and Mode 2
    /// are distinguishable on the wire. Without this, the Mode-1 test above would pass
    /// against a builder that always emitted a scheduled config.
    #[test]
    fn a_mode_2_grant_carries_no_rnti_and_no_rlc_bearers() {
        let params = SlConfigDedicatedParams {
            t400_ms: Some(400),
            phy_mac_rlc_config: true,
            ..Default::default()
        };
        let config = build_sl_config_dedicated(&params).expect("build");
        let bytes = encode_rrc(&config).expect("encode");
        let back: SL_ConfigDedicatedNR_r16 = decode_rrc(&bytes).expect("decode");
        let read = read_sl_config_dedicated(&back);
        assert_eq!(read.sl_rnti, None);
        assert!(read.sl_rlc_bearers.is_empty());
    }

    /// An `sl-RLC-BearerConfigIndex` outside `1..=512` is refused at the builder rather
    /// than encoded to a value naming a different channel.
    #[test]
    fn an_rlc_bearer_index_outside_the_asn1_range_is_refused() {
        for bad in [0u16, 513, u16::MAX] {
            let params = SlConfigDedicatedParams {
                phy_mac_rlc_config: true,
                sl_rlc_bearers: vec![SlRlcBearerConfig {
                    index: bad,
                    served_radio_bearer: None,
                }],
                ..Default::default()
            };
            assert!(
                build_sl_config_dedicated(&params).is_err(),
                "sl-RLC-BearerConfigIndex {bad} must be refused"
            );
        }
        for good in [1u16, 512] {
            let params = SlConfigDedicatedParams {
                phy_mac_rlc_config: true,
                sl_rlc_bearers: vec![SlRlcBearerConfig {
                    index: good,
                    served_radio_bearer: None,
                }],
                ..Default::default()
            };
            assert!(build_sl_config_dedicated(&params).is_ok());
        }
    }

    /// Issue #190: a UE declaring itself a relay carries `ue-Type-r17`, and it survives a
    /// real UPER round trip through the `v1700` non-critical extension.
    ///
    /// This is the IE that gives the gNB's relay grant a production trigger, so it has to
    /// reach the gNB rather than merely exist.
    #[test]
    fn a_declared_relay_role_survives_the_round_trip() {
        for (role, other) in [
            (SlUeType::RelayUe, SlUeType::RemoteUe),
            (SlUeType::RemoteUe, SlUeType::RelayUe),
        ] {
            let params = SidelinkUeInformationParams {
                rx_interested_freqs: vec![1],
                tx_resource_requests: Vec::new(),
                ue_type: Some(role),
            };
            let bytes = encode_sidelink_ue_information(&params).expect("encode");
            let decoded = decode_sidelink_ue_information(&bytes)
                .expect("decode")
                .expect("still a SidelinkUEInformationNR");
            assert_eq!(
                decoded.ue_type,
                Some(role),
                "the declared role must survive, and must not become {other:?}"
            );
            assert_ne!(decoded.ue_type, Some(other));
        }
    }

    /// A request that declares no role omits the whole `v1700` extension, so a plain PC5
    /// UE's request is unchanged by issue #190.
    ///
    /// The byte comparison is the load-bearing part: it proves the extension is *absent*
    /// rather than present-and-empty, which is what "unchanged" has to mean for a peer
    /// that decodes it.
    #[test]
    fn a_request_declaring_no_role_omits_the_v1700_extension() {
        let plain = SidelinkUeInformationParams {
            rx_interested_freqs: vec![1],
            tx_resource_requests: Vec::new(),
            ue_type: None,
        };
        let with_role = SidelinkUeInformationParams {
            ue_type: Some(SlUeType::RelayUe),
            ..plain.clone()
        };
        let plain_bytes = encode_sidelink_ue_information(&plain).expect("encode");
        let role_bytes = encode_sidelink_ue_information(&with_role).expect("encode");

        assert_eq!(
            decode_sidelink_ue_information(&plain_bytes)
                .expect("decode")
                .expect("present")
                .ue_type,
            None
        );
        assert!(
            plain_bytes.len() < role_bytes.len(),
            "declaring a role must cost bytes; without the v1700 extension the two \
             encodings would be identical and the role unreachable: {plain_bytes:02x?} \
             vs {role_bytes:02x?}"
        );
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
            ..Default::default()
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
                ..Default::default()
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
            ..Default::default()
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
                ..Default::default()
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
