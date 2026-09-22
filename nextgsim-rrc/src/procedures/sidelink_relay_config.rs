//! `sl-L2RelayUE-Config` and `sl-L2RemoteUE-Config`: the RRC half of L2 UE-to-Network
//! relay (TS 38.331 §5.3.5.17, §6.3.2; TS 38.300 §16.12.2.1; TS 38.351).
//!
//! # Why this module exists (issue #190)
//!
//! Issue #141 landed the PC5 procedures and UE-to-UE relay, and left UE-to-Network relay
//! to #190. Two of its criteria are RRC-shaped and are what this module closes:
//!
//! * the relay must be **told** which remote-UE bearers map onto which of its own RLC
//!   channels — `SL-SRAP-Config-r17`'s `sl-MappingToAddModList-r17`;
//! * the remote UE must be told the **local Remote UE ID** the gNB assigned it, which
//!   TS 38.300 §16.12.2.1 says it *"obtains ... from the gNB via Uu RRC messages"*
//!   (`38300-j30.txt:14749`).
//!
//! Before this, `grep -rn sl_rlc_bearer_to_add_mod_list_r16` over the tree found exactly
//! one hit — `sl_rlc_bearer_to_add_mod_list_r16: None` in `sidelink_ue_information.rs` —
//! so the IE existed in the generated types and nothing built or read one.
//!
//! # The encode path was probed BEFORE this was written, and it is clear
//!
//! Issue #141 recorded a ceiling aimed squarely at this module: the vendored
//! `asn1-codecs` still returns `EncodeNotSupported` for an extensible SEQUENCE that
//! actually carries an extension addition, and the vendored `asn1-compiler` silently
//! drops `[[ ]]` extension-addition fields. That made `sl-DiscConfig-r17` unreachable and
//! was called "the first thing #190 will hit".
//!
//! **It does not bite, for a structural reason worth recording.** `sl-L2RelayUE-Config-r17`
//! and `sl-L2RemoteUE-Config-r17` are not `[[ ]]` extension additions of
//! `RRCReconfiguration`: they are plain `OPTIONAL` fields of
//! `RRCReconfiguration-v1700-IEs` (`tools/rrc-19.3.0.asn1:1050-1051`), which is reached
//! through the ordinary `nonCriticalExtension` chain that issue #141 already walks to
//! v1610. A non-critical extension is a nested SEQUENCE, not an extension addition, so
//! the unimplemented encoder is never reached.
//!
//! Measured, root-only, on this branch. Every one encodes and round trips; the leading
//! extension bit is 0 in each, which is why:
//!
//! | IE | UPER bytes |
//! |---|---|
//! | `SL-SRAP-Config-r17` (local id `0x2A`, DRB5 → Uu channel 7) | `62 a0 24 86` |
//! | `SL-L2RelayUE-Config-r17` | `40 7f ff ff d0 1c` |
//! | `SL-L2RemoteUE-Config-r17` | `68 12 8c 02` |
//! | `SL-ScheduledConfig-r16` | `02 30 10` |
//! | `SL-PHY-MAC-RLC-Config-r16` with a Mode-1 grant and one RLC bearer | `84 10 23 01 00 10 00 01` |
//! | `RRCReconfiguration-v1700-IEs` carrying the relay config | 11 octets |
//!
//! `a_srap_relay_config_writes_extension_bit_zero` pins the first of these **and derives
//! it field by field from the ASN.1**, so a change is attributable to a field rather than
//! merely visible — the same method issue #141 used for `SL-ConfigDedicatedNR-r16`'s
//! `[0x02, 0x80]`.
//!
//! **The #141 ceiling is unchanged and still real, just not on this path.**
//! `SL-MappingToAddMod-r17`'s own Rel-19 additions — `sl-EgressRLC-Channel-UL-r19` and
//! `sl-EgressRLC-Channel-DL-r19` — ARE in a `[[ ]]` group, and the compiler drops them:
//! the generated struct has `optional_fields = 2`, covering only the two root optional
//! fields. Those two exist for the *multi-hop* intermediate relay case
//! (`38331-j30.txt:141886`), which this module does not implement, so nothing here needs
//! them. A future multi-hop issue does, and would have to fix the compiler first.
//!
//! # No vendored crate was touched
//!
//! Stated because the alternative was in scope and precedented (#117, #185, #45): it was
//! not necessary, so it was not done.
//!
//! # What is signalled, and what is deliberately not
//!
//! Signalled: the relay's per-remote-UE `sl-SRAP-ConfigRelay-r17` (local Remote UE ID plus
//! the bearer→egress-channel mappings), and the remote UE's `sl-SRAP-ConfigRemote-r17`
//! (its local Remote UE ID, and its `sl-UEIdentityRemote-r17` C-RNTI).
//!
//! Not signalled: the Rel-18 U2U and multi-path (`N3C`) chains, and
//! `sl-SRAP-ConfigRelay-ToAddModList-r19`. Each is a different relay architecture — L2
//! UE-to-UE relay (§16.12.2.2) and multi-path relay (§16.21) — and an IE for an
//! architecture this tree does not implement is the unread-IE defect #141 exists to
//! remove.

use crate::codec::generated::*;
use crate::codec::{decode_rrc, encode_rrc, RrcCodecError};
use nextgsim_rlc::srap::{EgressChannel, RemoteBearerId, RemoteUeMapping};
use thiserror::Error;

/// Errors building or reading an L2 relay RRC configuration.
#[derive(Debug, Error, PartialEq)]
pub enum SidelinkRelayRrcError {
    /// A field value TS 38.331 does not allow.
    #[error("invalid L2 relay field: {0}")]
    InvalidFieldValue(String),
    /// The codec refused the message.
    #[error("L2 relay RRC codec error: {0}")]
    Codec(String),
}

impl From<RrcCodecError> for SidelinkRelayRrcError {
    fn from(e: RrcCodecError) -> Self {
        Self::Codec(format!("{e}"))
    }
}

/// `maxNrofRemoteUE-r17` is 32: the most L2 U2N remote UEs one relay may serve
/// (`tools/rrc-19.3.0.asn1:29789`).
pub const MAX_REMOTE_UES: usize = 32;

/// `maxLC-ID` is 32, and `Uu-RelayRLC-ChannelID-r17 ::= INTEGER (1..maxLC-ID)`
/// (`tools/rrc-19.3.0.asn1:16825`, `:29451`).
const UU_RELAY_RLC_CHANNEL_RANGE: std::ops::RangeInclusive<u8> = 1..=32;

/// `maxSL-LCID-r16`, the bound on `SL-RLC-ChannelID-r17`
/// (`tools/rrc-19.3.0.asn1:28834`).
const SL_RLC_CHANNEL_RANGE: std::ops::RangeInclusive<u16> = 1..=512;

/// A `SL-DestinationIdentity-r16` is a `BIT STRING (SIZE (24))` — the remote UE's 24-bit
/// ProSe Layer-2 ID as RRC carries it (TS 23.304 §5.8.2.1).
const SL_DESTINATION_IDENTITY_BITS: usize = 24;

/// One remote UE as the relay's `sl-L2RelayUE-Config` describes it
/// (`SL-RemoteUE-ToAddMod-r17`, TS 38.331 §6.3.2).
///
/// Pairs the remote UE's PC5 Layer-2 ID — how the relay addresses it over PC5 — with the
/// SRAP mapping the relay applies to its traffic. Both are needed and neither substitutes
/// for the other: §16.12.2.1 is explicit that *"The serving gNB can perform local Remote
/// UE ID update independent of the PC5 unicast link L2 ID update procedure"*
/// (`38300-j30.txt:14761`), so the two identities are separate spaces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelayRemoteUeConfig {
    /// The remote UE's 24-bit ProSe Layer-2 ID (`sl-L2IdentityRemote-r17`).
    pub remote_l2_id: u32,
    /// The SRAP mapping for this remote UE: its local Remote UE ID and its per-bearer
    /// egress channels.
    ///
    /// The SAME type `nextgsim_rlc::srap` applies, deliberately: the mapping this gNB
    /// signals and the mapping the relay's SRAP entity enforces must be one thing, or a
    /// bearer could be signalled that the header writer cannot express.
    pub srap: RemoteUeMapping,
}

/// The relay UE's L2 relay configuration (`SL-L2RelayUE-Config-r17`, TS 38.331 §6.3.2).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct L2RelayUeConfigParams {
    /// The remote UEs this relay serves, to add or modify.
    pub remote_ues: Vec<RelayRemoteUeConfig>,
    /// Remote UEs to stop serving, by PC5 Layer-2 ID
    /// (`sl-RemoteUE-ToReleaseList-r17`).
    pub release_remote_l2_ids: Vec<u32>,
}

/// The remote UE's own L2 configuration (`SL-L2RemoteUE-Config-r17`, TS 38.331 §6.3.2).
///
/// This is how a remote UE learns the identity it must put in its own SRAP headers:
/// §16.12.2.1's *"L2 U2N Remote UE obtains the local Remote ID from the gNB via Uu RRC
/// messages"*.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct L2RemoteUeConfigParams {
    /// The local Remote UE ID the gNB assigned, and this remote UE's own bearer mappings
    /// onto its egress PC5 relay RLC channels (`sl-SRAP-ConfigRemote-r17`).
    ///
    /// `None` means the IE is omitted, which TS 38.331 §9.2.5's default covers: SRAP PDUs
    /// with any local identity go to the SRB1 PDCP entity.
    pub srap: Option<RemoteUeMapping>,
    /// The C-RNTI the remote UE uses on the relay's cell
    /// (`sl-UEIdentityRemote-r17`, `RNTI-Value`).
    pub remote_ue_c_rnti: Option<u16>,
}

/// Builds an `SL-SRAP-Config-r17` from a SRAP mapping (TS 38.331 §6.3.2; TS 38.351).
///
/// Only root fields are set, which is what keeps this inside the vendored codec's
/// encodable subset — see the module docs for the measurement.
pub fn build_srap_config(
    mapping: &RemoteUeMapping,
) -> Result<SL_SRAP_Config_r17, SidelinkRelayRrcError> {
    let mut list = Vec::with_capacity(mapping.mapped_bearer_count());
    for (bearer, egress) in mapping.bearers() {
        let (uu, pc5) = match egress {
            EgressChannel::Uu(channel) => {
                if !UU_RELAY_RLC_CHANNEL_RANGE.contains(&channel) {
                    return Err(SidelinkRelayRrcError::InvalidFieldValue(format!(
                        "Uu relay RLC channel {channel} is outside 1..=32 (maxLC-ID)"
                    )));
                }
                (Some(Uu_RelayRLC_ChannelID_r17(channel)), None)
            }
            EgressChannel::Pc5(channel) => {
                if !SL_RLC_CHANNEL_RANGE.contains(&channel) {
                    return Err(SidelinkRelayRrcError::InvalidFieldValue(format!(
                        "PC5 relay RLC channel {channel} is outside 1..=512 (maxSL-LCID-r16)"
                    )));
                }
                (None, Some(SL_RLC_ChannelID_r17(channel)))
            }
        };
        list.push(SL_MappingToAddMod_r17 {
            sl_remote_ue_rb_identity_r17: bearer_identity(bearer),
            sl_egress_rlc_channel_uu_r17: uu,
            sl_egress_rlc_channel_pc5_r17: pc5,
        });
    }

    Ok(SL_SRAP_Config_r17 {
        sl_local_identity_r17: Some(SL_SRAP_Config_r17Sl_LocalIdentity_r17(
            mapping.local_remote_ue_id,
        )),
        // An empty list is OMITTED rather than encoded: the ASN.1 size constraint is
        // `SIZE (1..maxLC-ID)`, so a zero-length SEQUENCE OF is not a legal value and
        // encoding one would produce a PDU no conformant peer can decode.
        sl_mapping_to_add_mod_list_r17: if list.is_empty() {
            None
        } else {
            Some(SL_SRAP_Config_r17Sl_MappingToAddModList_r17(list))
        },
        sl_mapping_to_release_list_r17: None,
    })
}

/// The `SL-RemoteUE-RB-Identity-r17` CHOICE arm for a SRAP bearer identity.
///
/// The SRB/DRB distinction is carried by the CHOICE, which is why
/// [`RemoteBearerId`] is an enum: collapsing the two would make SRB *n* and DRB *n*
/// the same bearer, and the ASN.1 says they are not.
fn bearer_identity(bearer: RemoteBearerId) -> SL_RemoteUE_RB_Identity_r17 {
    match bearer {
        RemoteBearerId::Srb(id) => SL_RemoteUE_RB_Identity_r17::Srb_Identity_r17(
            SL_RemoteUE_RB_Identity_r17_srb_Identity_r17(id),
        ),
        RemoteBearerId::Drb(id) => SL_RemoteUE_RB_Identity_r17::Drb_Identity_r17(DRB_Identity(id)),
    }
}

/// Reads an `SL-SRAP-Config-r17` back into a SRAP mapping.
///
/// A mapping entry whose bearer identity or egress channel the ASN.1 admits but
/// [`RemoteBearerId`] does not — SRB3, say, which TS 38.331's field description excludes —
/// is **skipped rather than accepted**, and the rest of the configuration still applies.
/// Dropping one unusable mapping is better than refusing a whole reconfiguration, because
/// the alternative leaves every other bearer of that remote UE unconfigured too.
///
/// An entry naming neither egress channel is likewise skipped: both fields are `OPTIONAL`
/// in the ASN.1, so "no egress channel" is encodable and means nothing this code can act
/// on.
pub fn read_srap_config(config: &SL_SRAP_Config_r17) -> Option<RemoteUeMapping> {
    let local_id = config.sl_local_identity_r17.as_ref()?.0;
    let mut mapping = RemoteUeMapping::new(local_id);
    if let Some(list) = config.sl_mapping_to_add_mod_list_r17.as_ref() {
        for entry in &list.0 {
            let Some(bearer) = read_bearer_identity(&entry.sl_remote_ue_rb_identity_r17) else {
                continue;
            };
            // Uu first: TS 38.331 makes `sl-EgressRLC-ChannelUu-r17` the
            // `Cond L2RelayUE` field, i.e. the one a relay UE is given, and a value
            // carrying both would be describing a hop this single-hop code does not have.
            let egress = match (
                entry.sl_egress_rlc_channel_uu_r17.as_ref(),
                entry.sl_egress_rlc_channel_pc5_r17.as_ref(),
            ) {
                (Some(uu), _) => EgressChannel::Uu(uu.0),
                (None, Some(pc5)) => EgressChannel::Pc5(pc5.0),
                (None, None) => continue,
            };
            mapping.map_bearer(bearer, egress);
        }
    }
    Some(mapping)
}

/// Reads a `SL-RemoteUE-RB-Identity-r17` back, or `None` if it names a bearer TS 38.331
/// does not allow (SRB3, or a DRB outside `1..=32`).
fn read_bearer_identity(id: &SL_RemoteUE_RB_Identity_r17) -> Option<RemoteBearerId> {
    match id {
        SL_RemoteUE_RB_Identity_r17::Srb_Identity_r17(srb) => RemoteBearerId::srb(srb.0),
        SL_RemoteUE_RB_Identity_r17::Drb_Identity_r17(drb) => RemoteBearerId::drb(drb.0),
    }
}

/// Builds an `SL-L2RelayUE-Config-r17` (TS 38.331 §6.3.2).
pub fn build_l2_relay_ue_config(
    params: &L2RelayUeConfigParams,
) -> Result<SL_L2RelayUE_Config_r17, SidelinkRelayRrcError> {
    if params.remote_ues.len() > MAX_REMOTE_UES {
        return Err(SidelinkRelayRrcError::InvalidFieldValue(format!(
            "{} remote UEs is more than maxNrofRemoteUE-r17 ({MAX_REMOTE_UES})",
            params.remote_ues.len()
        )));
    }
    if params.release_remote_l2_ids.len() > MAX_REMOTE_UES {
        return Err(SidelinkRelayRrcError::InvalidFieldValue(format!(
            "{} remote UEs to release is more than maxNrofRemoteUE-r17 ({MAX_REMOTE_UES})",
            params.release_remote_l2_ids.len()
        )));
    }

    let mut add_mod = Vec::with_capacity(params.remote_ues.len());
    for remote in &params.remote_ues {
        add_mod.push(SL_RemoteUE_ToAddMod_r17 {
            sl_l2_identity_remote_r17: destination_identity(remote.remote_l2_id),
            sl_srap_config_relay_r17: Some(build_srap_config(&remote.srap)?),
        });
    }

    Ok(SL_L2RelayUE_Config_r17 {
        sl_remote_ue_to_add_mod_list_r17: if add_mod.is_empty() {
            None
        } else {
            Some(SL_L2RelayUE_Config_r17Sl_RemoteUE_ToAddModList_r17(add_mod))
        },
        sl_remote_ue_to_release_list_r17: if params.release_remote_l2_ids.is_empty() {
            None
        } else {
            Some(SL_L2RelayUE_Config_r17Sl_RemoteUE_ToReleaseList_r17(
                params
                    .release_remote_l2_ids
                    .iter()
                    .map(|id| destination_identity(*id))
                    .collect(),
            ))
        },
    })
}

/// Reads an `SL-L2RelayUE-Config-r17` back to the parameters that built it.
///
/// A remote UE whose `sl-SRAP-ConfigRelay-r17` is absent or unreadable is skipped: a
/// remote UE with no SRAP mapping is one the relay cannot adapt any traffic for, and
/// admitting it would put an entry in the relay's SRAP entity that every `adapt` call
/// would then refuse.
pub fn read_l2_relay_ue_config(config: &SL_L2RelayUE_Config_r17) -> L2RelayUeConfigParams {
    L2RelayUeConfigParams {
        remote_ues: config
            .sl_remote_ue_to_add_mod_list_r17
            .as_ref()
            .map(|list| {
                list.0
                    .iter()
                    .filter_map(|entry| {
                        Some(RelayRemoteUeConfig {
                            remote_l2_id: read_destination_identity(
                                &entry.sl_l2_identity_remote_r17,
                            )?,
                            srap: read_srap_config(entry.sl_srap_config_relay_r17.as_ref()?)?,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default(),
        release_remote_l2_ids: config
            .sl_remote_ue_to_release_list_r17
            .as_ref()
            .map(|list| {
                list.0
                    .iter()
                    .filter_map(read_destination_identity)
                    .collect()
            })
            .unwrap_or_default(),
    }
}

/// Builds an `SL-L2RemoteUE-Config-r17` (TS 38.331 §6.3.2).
pub fn build_l2_remote_ue_config(
    params: &L2RemoteUeConfigParams,
) -> Result<SL_L2RemoteUE_Config_r17, SidelinkRelayRrcError> {
    Ok(SL_L2RemoteUE_Config_r17 {
        sl_srap_config_remote_r17: match params.srap.as_ref() {
            Some(mapping) => Some(build_srap_config(mapping)?),
            None => None,
        },
        sl_ue_identity_remote_r17: params.remote_ue_c_rnti.map(RNTI_Value),
    })
}

/// Reads an `SL-L2RemoteUE-Config-r17` back to the parameters that built it.
pub fn read_l2_remote_ue_config(config: &SL_L2RemoteUE_Config_r17) -> L2RemoteUeConfigParams {
    L2RemoteUeConfigParams {
        srap: config
            .sl_srap_config_remote_r17
            .as_ref()
            .and_then(read_srap_config),
        remote_ue_c_rnti: config.sl_ue_identity_remote_r17.as_ref().map(|r| r.0),
    }
}

/// The 24-bit `SL-DestinationIdentity-r16` bit string for a ProSe Layer-2 ID.
///
/// Most significant bit first, matching `sidelink_ue_information`'s encoding of the same
/// ASN.1 type: bit 0 of the string is bit 23 of the value. The two must agree, because a
/// relay's `sl-L2IdentityRemote-r17` names the same peer a
/// `SidelinkUEInformation`'s `sl-DestinationIdentity-r16` does.
fn destination_identity(l2_id: u32) -> SL_DestinationIdentity_r16 {
    let mut bits =
        bitvec::vec::BitVec::<u8, bitvec::order::Msb0>::repeat(false, SL_DESTINATION_IDENTITY_BITS);
    for i in 0..SL_DESTINATION_IDENTITY_BITS {
        let shift = SL_DESTINATION_IDENTITY_BITS - 1 - i;
        bits.set(i, (l2_id >> shift) & 1 == 1);
    }
    SL_DestinationIdentity_r16(bits)
}

/// Reads a `SL-DestinationIdentity-r16` back to a ProSe Layer-2 ID.
///
/// A bit string of the wrong length yields `None` rather than a truncated ID: an identity
/// that is not 24 bits is not a Layer-2 ID, and reinterpreting it would address the wrong
/// remote UE.
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

/// Encodes an `SL-SRAP-Config-r17` to UPER, for the round-trip tests and the encode probe.
pub fn encode_srap_config(mapping: &RemoteUeMapping) -> Result<Vec<u8>, SidelinkRelayRrcError> {
    Ok(encode_rrc(&build_srap_config(mapping)?)?)
}

/// Decodes an `SL-SRAP-Config-r17` from UPER.
pub fn decode_srap_config(bytes: &[u8]) -> Result<Option<RemoteUeMapping>, SidelinkRelayRrcError> {
    let config: SL_SRAP_Config_r17 = decode_rrc(bytes)?;
    Ok(read_srap_config(&config))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A remote UE Layer-2 ID with a bit set in every octet, so a byte-order slip cannot
    /// produce it by accident.
    const REMOTE_L2_ID: u32 = 0x00_A5_3C_71 & 0x00FF_FFFF;
    const LOCAL_ID: u8 = 0x2A;
    const UU_CHANNEL: u8 = 7;

    fn drb5() -> RemoteBearerId {
        RemoteBearerId::drb(5).expect("DRB5")
    }

    fn mapping() -> RemoteUeMapping {
        let mut m = RemoteUeMapping::new(LOCAL_ID);
        m.map_bearer(drb5(), EgressChannel::Uu(UU_CHANNEL));
        m
    }

    /// The criterion-2 round trip: a SRAP bearer mapping survives real UPER, and the
    /// bearer→channel pair comes back out.
    ///
    /// This is the encode probe the module docs describe, as a standing test: if the
    /// vendored codec's extended-SEQUENCE gap ever starts to bite here, this fails.
    #[test]
    fn a_srap_config_round_trips_through_real_uper() {
        let bytes = encode_srap_config(&mapping()).expect(
            "an extensible SEQUENCE with only root fields set must encode; if this fails, \
             the vendored codec's extended-SEQUENCE gap has started to bite",
        );
        let back = decode_srap_config(&bytes)
            .expect("decode")
            .expect("the bytes carry an SL-SRAP-Config");

        // Positive assertions on values only reachable by having decoded real UPER.
        assert_eq!(back.local_remote_ue_id, LOCAL_ID);
        assert_eq!(back.mapped_bearer_count(), 1);
        assert_eq!(back.egress_for(drb5()), Some(EgressChannel::Uu(UU_CHANNEL)));
        assert_eq!(back, mapping());
    }

    /// The exact bytes of a minimal `SL-SRAP-Config-r17`, pinned.
    ///
    /// The measurement the module docs quote. Pinning it makes "the extension bit is 0, so
    /// the unimplemented encoder is never reached" checkable rather than assertible — the
    /// method issue #141 used for `SL-ConfigDedicatedNR-r16`.
    ///
    /// The 32 bits, derived from the ASN.1 rather than merely observed — so that a change
    /// to any one field is attributable rather than just "the bytes moved":
    ///
    /// ```text
    /// 0            extension bit, CLEAR — this is the whole point of the test
    /// 110          optional bitmap: local identity present, add-mod list present,
    ///              release list absent
    /// 0010 1010    sl-LocalIdentity-r17 = 0x2A, INTEGER (0..255), 8 bits
    /// 00000        add-mod list length 1, SIZE (1..32) as a 5-bit count of (n-1)
    /// 0            SL-MappingToAddMod-r17 extension bit, clear
    /// 10           its optional bitmap: egress Uu present, egress PC5 absent
    /// 0            SL-RemoteUE-RB-Identity-r17 CHOICE extension bit, clear
    /// 1            CHOICE index 1 = drb-Identity
    /// 0 0100       DRB-Identity 5, INTEGER (1..32) as 5 bits of (5-1)
    /// 0 0110       Uu-RelayRLC-ChannelID-r17 7, INTEGER (1..32) as 5 bits of (7-1)
    /// ```
    #[test]
    fn a_srap_relay_config_writes_extension_bit_zero() {
        let bytes = encode_srap_config(&mapping()).expect("encode");
        assert_eq!(
            bytes[0] & 0b1000_0000,
            0,
            "the extension bit is set, so this value would reach the vendored codec's \
             unimplemented extended-SEQUENCE encoder; bytes = {bytes:02x?}"
        );
        assert_eq!(bytes, vec![0x62, 0xa0, 0x24, 0x86]);
    }

    /// An SRB mapping and a DRB mapping with the same number are two entries, and survive
    /// as two — the CHOICE distinction has to make it through the codec.
    #[test]
    fn an_srb_and_a_drb_mapping_with_the_same_number_both_survive() {
        let srb2 = RemoteBearerId::srb(2).expect("SRB2");
        let drb2 = RemoteBearerId::drb(2).expect("DRB2");
        let mut m = RemoteUeMapping::new(LOCAL_ID);
        m.map_bearer(srb2, EgressChannel::Uu(1));
        m.map_bearer(drb2, EgressChannel::Uu(2));

        let bytes = encode_srap_config(&m).expect("encode");
        let back = decode_srap_config(&bytes)
            .expect("decode")
            .expect("present");
        assert_eq!(back.mapped_bearer_count(), 2, "SRB2 and DRB2 are distinct");
        assert_eq!(back.egress_for(srb2), Some(EgressChannel::Uu(1)));
        assert_eq!(back.egress_for(drb2), Some(EgressChannel::Uu(2)));
    }

    /// A PC5 egress channel round trips as a PC5 channel, not as a Uu one: the two are
    /// different hops, and confusing them would send downlink traffic towards the gNB.
    #[test]
    fn a_pc5_egress_channel_does_not_become_a_uu_one() {
        let mut m = RemoteUeMapping::new(LOCAL_ID);
        m.map_bearer(drb5(), EgressChannel::Pc5(12));
        let bytes = encode_srap_config(&m).expect("encode");
        let back = decode_srap_config(&bytes)
            .expect("decode")
            .expect("present");
        assert_eq!(back.egress_for(drb5()), Some(EgressChannel::Pc5(12)));
    }

    /// The whole relay configuration round trips, with the remote UE's Layer-2 ID and its
    /// SRAP mapping both intact.
    #[test]
    fn an_l2_relay_ue_config_round_trips_through_real_uper() {
        let params = L2RelayUeConfigParams {
            remote_ues: vec![RelayRemoteUeConfig {
                remote_l2_id: REMOTE_L2_ID,
                srap: mapping(),
            }],
            release_remote_l2_ids: Vec::new(),
        };
        let config = build_l2_relay_ue_config(&params).expect("build");
        let bytes = encode_rrc(&config).expect("encode");
        let back: SL_L2RelayUE_Config_r17 = decode_rrc(&bytes).expect("decode");
        let read = read_l2_relay_ue_config(&back);

        assert_eq!(read.remote_ues.len(), 1);
        assert_eq!(read.remote_ues[0].remote_l2_id, REMOTE_L2_ID);
        assert_eq!(read.remote_ues[0].srap.local_remote_ue_id, LOCAL_ID);
        assert_eq!(
            read.remote_ues[0].srap.egress_for(drb5()),
            Some(EgressChannel::Uu(UU_CHANNEL))
        );
        assert_eq!(read, params);
    }

    /// The remote UE's own configuration round trips, carrying the local Remote UE ID it
    /// must put in its SRAP headers and its C-RNTI.
    #[test]
    fn an_l2_remote_ue_config_round_trips_through_real_uper() {
        let params = L2RemoteUeConfigParams {
            srap: Some(mapping()),
            remote_ue_c_rnti: Some(0x4601),
        };
        let config = build_l2_remote_ue_config(&params).expect("build");
        let bytes = encode_rrc(&config).expect("encode");
        let back: SL_L2RemoteUE_Config_r17 = decode_rrc(&bytes).expect("decode");
        let read = read_l2_remote_ue_config(&back);

        assert_eq!(read.remote_ue_c_rnti, Some(0x4601));
        assert_eq!(
            read.srap.as_ref().expect("srap").local_remote_ue_id,
            LOCAL_ID,
            "the remote UE must learn the local Remote UE ID the gNB assigned it \
             (TS 38.300 §16.12.2.1)"
        );
        assert_eq!(read, params);
    }

    /// A release list round trips, so a gNB can stop a relay serving a remote UE.
    #[test]
    fn a_release_list_round_trips() {
        let params = L2RelayUeConfigParams {
            remote_ues: Vec::new(),
            release_remote_l2_ids: vec![REMOTE_L2_ID, 0x00_00_02],
        };
        let config = build_l2_relay_ue_config(&params).expect("build");
        let bytes = encode_rrc(&config).expect("encode");
        let back: SL_L2RelayUE_Config_r17 = decode_rrc(&bytes).expect("decode");
        assert_eq!(
            read_l2_relay_ue_config(&back).release_remote_l2_ids,
            vec![REMOTE_L2_ID, 0x00_00_02]
        );
    }

    /// More remote UEs than `maxNrofRemoteUE-r17` is refused rather than truncated: a
    /// truncated list would silently leave a remote UE unconfigured.
    #[test]
    fn more_remote_ues_than_the_asn1_maximum_are_refused() {
        let params = L2RelayUeConfigParams {
            remote_ues: (0..=MAX_REMOTE_UES as u32)
                .map(|i| RelayRemoteUeConfig {
                    remote_l2_id: i,
                    srap: RemoteUeMapping::new(i as u8),
                })
                .collect(),
            release_remote_l2_ids: Vec::new(),
        };
        assert!(build_l2_relay_ue_config(&params).is_err());
    }

    /// A Uu relay RLC channel outside `1..=32` is refused at the builder, not encoded to a
    /// value meaning a different channel.
    #[test]
    fn a_uu_relay_rlc_channel_outside_the_asn1_range_is_refused() {
        for bad in [0u8, 33, 255] {
            let mut m = RemoteUeMapping::new(LOCAL_ID);
            m.map_bearer(drb5(), EgressChannel::Uu(bad));
            assert!(
                build_srap_config(&m).is_err(),
                "Uu relay RLC channel {bad} must be refused"
            );
        }
        for good in [1u8, 32] {
            let mut m = RemoteUeMapping::new(LOCAL_ID);
            m.map_bearer(drb5(), EgressChannel::Uu(good));
            assert!(build_srap_config(&m).is_ok());
        }
    }

    /// A mapping with no bearers omits the list rather than encoding a zero-length
    /// SEQUENCE OF, which `SIZE (1..maxLC-ID)` forbids.
    #[test]
    fn a_mapping_with_no_bearers_omits_the_list() {
        let bytes = encode_srap_config(&RemoteUeMapping::new(LOCAL_ID)).expect("encode");
        let back = decode_srap_config(&bytes)
            .expect("decode")
            .expect("present");
        assert_eq!(back.local_remote_ue_id, LOCAL_ID);
        assert_eq!(back.mapped_bearer_count(), 0);
    }

    /// A received mapping naming SRB3 — which the ASN.1 admits and TS 38.331's field
    /// description excludes — is skipped, and the rest of the configuration still applies.
    ///
    /// Built by hand, because `build_srap_config` cannot produce one: `RemoteBearerId::srb`
    /// refuses SRB3. So this covers a conformant-looking PDU from a peer that did encode
    /// it.
    #[test]
    fn a_received_srb3_mapping_is_skipped_and_the_rest_still_applies() {
        let config = SL_SRAP_Config_r17 {
            sl_local_identity_r17: Some(SL_SRAP_Config_r17Sl_LocalIdentity_r17(LOCAL_ID)),
            sl_mapping_to_add_mod_list_r17: Some(SL_SRAP_Config_r17Sl_MappingToAddModList_r17(
                vec![
                    // SRB3, which TS 38.331 says is not supported.
                    SL_MappingToAddMod_r17 {
                        sl_remote_ue_rb_identity_r17: SL_RemoteUE_RB_Identity_r17::Srb_Identity_r17(
                            SL_RemoteUE_RB_Identity_r17_srb_Identity_r17(3),
                        ),
                        sl_egress_rlc_channel_uu_r17: Some(Uu_RelayRLC_ChannelID_r17(1)),
                        sl_egress_rlc_channel_pc5_r17: None,
                    },
                    // And a perfectly good DRB after it.
                    SL_MappingToAddMod_r17 {
                        sl_remote_ue_rb_identity_r17: SL_RemoteUE_RB_Identity_r17::Drb_Identity_r17(
                            DRB_Identity(5),
                        ),
                        sl_egress_rlc_channel_uu_r17: Some(Uu_RelayRLC_ChannelID_r17(UU_CHANNEL)),
                        sl_egress_rlc_channel_pc5_r17: None,
                    },
                ],
            )),
            sl_mapping_to_release_list_r17: None,
        };
        let read = read_srap_config(&config).expect("a local identity is present");
        assert_eq!(
            read.mapped_bearer_count(),
            1,
            "SRB3 must be skipped, not accepted"
        );
        assert_eq!(read.egress_for(drb5()), Some(EgressChannel::Uu(UU_CHANNEL)));
    }

    /// A mapping entry naming NEITHER egress channel is skipped: both fields are
    /// `OPTIONAL`, so it is encodable and means nothing this code can act on.
    #[test]
    fn a_mapping_with_no_egress_channel_is_skipped() {
        let config = SL_SRAP_Config_r17 {
            sl_local_identity_r17: Some(SL_SRAP_Config_r17Sl_LocalIdentity_r17(LOCAL_ID)),
            sl_mapping_to_add_mod_list_r17: Some(SL_SRAP_Config_r17Sl_MappingToAddModList_r17(
                vec![SL_MappingToAddMod_r17 {
                    sl_remote_ue_rb_identity_r17: SL_RemoteUE_RB_Identity_r17::Drb_Identity_r17(
                        DRB_Identity(5),
                    ),
                    sl_egress_rlc_channel_uu_r17: None,
                    sl_egress_rlc_channel_pc5_r17: None,
                }],
            )),
            sl_mapping_to_release_list_r17: None,
        };
        assert_eq!(
            read_srap_config(&config)
                .expect("present")
                .mapped_bearer_count(),
            0
        );
    }

    /// A configuration with no `sl-LocalIdentity` yields no mapping: without the local
    /// Remote UE ID there is no identity to put in a SRAP header, and TS 38.331 §9.2.5's
    /// default sends such PDUs to SRB1 rather than to a mapped bearer.
    #[test]
    fn a_config_without_a_local_identity_yields_no_mapping() {
        let config = SL_SRAP_Config_r17 {
            sl_local_identity_r17: None,
            sl_mapping_to_add_mod_list_r17: None,
            sl_mapping_to_release_list_r17: None,
        };
        assert_eq!(read_srap_config(&config), None);
    }

    /// The remote UE's 24-bit identity is most significant bit first, matching
    /// `sidelink_ue_information`'s encoding of the same ASN.1 type. The octets are pinned,
    /// because a reversed order would still round trip through this codec.
    #[test]
    fn a_remote_ue_identity_is_24_bits_most_significant_first() {
        let id = destination_identity(REMOTE_L2_ID);
        assert_eq!(id.0.len(), 24);
        // 0xA53C71 = 1010 0101 ...
        let bits: Vec<bool> = id.0.iter().map(|b| *b).collect();
        assert_eq!(
            &bits[0..8],
            &[true, false, true, false, false, true, false, true]
        );
        assert_eq!(read_destination_identity(&id), Some(REMOTE_L2_ID));
    }

    /// An identity that is not 24 bits is refused rather than reinterpreted.
    #[test]
    fn a_remote_ue_identity_of_the_wrong_length_is_refused() {
        let short = SL_DestinationIdentity_r16(
            bitvec::vec::BitVec::<u8, bitvec::order::Msb0>::repeat(true, 16),
        );
        assert_eq!(read_destination_identity(&short), None);
    }
}
