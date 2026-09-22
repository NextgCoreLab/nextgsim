//! SRAP — Sidelink Relay Adaptation Protocol (TS 38.351), the L2 UE-to-Network relay
//! adaptation layer.
//!
//! # Why this module exists (issue #190)
//!
//! Issue #141 implemented the PC5 procedures and **UE-to-UE** relay forwarding, and
//! deliberately carved UE-to-Network relay out: a remote UE reaching the 5GC *through* a
//! relay UE's Uu connection needs an adaptation sublayer that did not exist anywhere in
//! this tree, in either `nextgsim-rlc` or `nextgsim-pdcp`. This is that sublayer.
//!
//! # Where it sits, and why it is in this crate
//!
//! TS 38.300 §16.12.2.1: *"The SRAP sublayer is placed above the RLC sublayer for both CP
//! and UP at both PC5 interface and Uu interface"* (`38300-j30.txt:14663`). SRAP is
//! therefore an RLC-adjacent sublayer, which is why it lives beside [`crate::entity`]
//! rather than in `nextgsim-pdcp`: the end-to-end PDCP entities it multiplexes belong to
//! the **remote** UE and the gNB, and are terminated at neither hop SRAP runs on.
//!
//! The protocol stack this implements, for single-hop L2 U2N relay:
//!
//! ```text
//!  Remote UE                  Relay UE                       gNB
//!  ┌─────────┐                                          ┌─────────┐
//!  │  SDAP   │ ═══════════ end to end ═════════════════ │  SDAP   │
//!  │  PDCP   │ ═══════════ end to end ═════════════════ │  PDCP   │
//!  ├─────────┤            ┌──────┬──────┐               ├─────────┤
//!  │  SRAP   │──── PC5 ───│ SRAP │ SRAP │───── Uu ──────│  SRAP   │
//!  │  RLC    │            │ RLC  │ RLC  │               │  RLC    │
//!  └─────────┘            └──────┴──────┘               └─────────┘
//! ```
//!
//! # What the header carries, and the one thing that is a local assignment
//!
//! §16.12.2.1: *"The identity information of L2 U2N Remote UE end-to-end Uu Radio Bearer
//! and a local Remote UE ID are included in the Uu SRAP header at UL in order for gNB to
//! correlate the received packets for the specific PDCP entity associated with the right
//! end-to-end Uu Radio Bearer of the L2 U2N Remote UE"* (`38300-j30.txt:14710`). So the
//! header carries exactly two fields, and this module encodes exactly those two:
//!
//! * the **local Remote UE ID**, which TS 38.331's `SL-SRAP-Config-r17.sl-LocalIdentity-r17`
//!   bounds to `INTEGER (0..255)` — one octet, verified at
//!   `tools/rrc-19.3.0.asn1:28936`;
//! * the **bearer identity**, `SL-RemoteUE-RB-Identity-r17`, a CHOICE of
//!   `srb-Identity-r17 INTEGER (0..3)` or `drb-Identity DRB-Identity`, the latter
//!   `INTEGER (1..32)` (`tools/rrc-19.3.0.asn1:28954`, `:8304`).
//!
//! **TS 38.351 is NOT in the vendored spec text.** `ls ../../6g_docs/specs/ | grep 38351`
//! is empty; the 38-series there stops at 38.331/38.413/38.455. So the *fields* of this
//! header and their value ranges are taken from TS 38.300 §16.12.2.1 and from the TS
//! 38.331 ASN.1 that configures them — both vendored, both cited above — but the **octet
//! layout within the header is this simulator's own assignment**, not a transcription of
//! TS 38.351 §6.2. This is the same honesty position issue #141 recorded for the PC5-S
//! message-type octets when TS 24.554 turned out not to be vendored: both ends here read
//! the same [`SrapHeader::encode`]/[`SrapHeader::decode`] pair, so the layout is
//! self-consistent and one function changes it, but it is not claimed to interoperate
//! with a real TS 38.351 implementation.
//!
//! What is NOT invented is the part that matters for correctness: which identities the
//! header must carry, their legal ranges, and the bearer-mapping behaviour — all of which
//! come from the two vendored specs.
//!
//! # SRB3 is refused rather than encoded
//!
//! TS 38.331's `SL-SRAP-Config` field description for `sl-RemoteUE-RB-Identity`:
//! *"The value 3 for the field srb-identity-r17 (i.e., for configuring SRB3) is not
//! supported in this version of the specification"* (`38331-j30.txt:141881`). So a value
//! the ASN.1 range admits is nonetheless not signallable, and
//! [`RemoteBearerId::srb`] refuses it rather than encoding a bearer the network cannot
//! have configured.

use crate::error::RlcError;
use std::collections::BTreeMap;

/// The width of the local Remote UE ID field, in octets.
///
/// One, because `SL-SRAP-Config-r17.sl-LocalIdentity-r17` is `INTEGER (0..255)`
/// (`tools/rrc-19.3.0.asn1:28936`) — the configured value cannot exceed one octet, so a
/// wider field would carry bits the network can never set.
const LOCAL_ID_OCTETS: usize = 1;

/// The width of the bearer-identity field, in octets.
const BEARER_ID_OCTETS: usize = 1;

/// The SRAP header length, in octets.
pub const SRAP_HEADER_LEN: usize = LOCAL_ID_OCTETS + BEARER_ID_OCTETS;

/// The bit in the bearer-identity octet that distinguishes an SRB from a DRB.
///
/// `SL-RemoteUE-RB-Identity-r17` is a CHOICE, so the two arms overlap numerically: SRB 1
/// and DRB 1 are different bearers with the same value. One discriminator bit is the
/// minimum that keeps them distinguishable, and without it a DRB's traffic would be
/// delivered to an SRB's PDCP entity.
const BEARER_IS_DRB: u8 = 0b1000_0000;

/// The mask covering the bearer identity itself, below [`BEARER_IS_DRB`].
const BEARER_ID_MASK: u8 = 0b0111_1111;

/// `DRB-Identity ::= INTEGER (1..32)` (`tools/rrc-19.3.0.asn1:8304`).
const DRB_ID_RANGE: std::ops::RangeInclusive<u8> = 1..=32;

/// `srb-Identity-r17 INTEGER (0..3)`, minus SRB3 — see the module docs.
const SRB_ID_RANGE: std::ops::RangeInclusive<u8> = 0..=2;

/// The SRB identity TS 38.331 §9.2.5 names as the default SRAP bearer.
///
/// *"`>sl-RemoteUE-RB-Identity`  SRB1"* (`38331-j30.txt:149881`): with no
/// `sl-LocalIdentity` configured, SRAP PDUs are submitted to the SRB1 PDCP entity.
pub const DEFAULT_SRAP_SRB: u8 = 1;

/// Which end-to-end Uu radio bearer of the remote UE a SRAP PDU belongs to
/// (`SL-RemoteUE-RB-Identity-r17`, TS 38.331; TS 38.351 via TS 38.300 §16.12.2.1).
///
/// An enum rather than a bare `u8` because the ASN.1 is a CHOICE: the SRB and DRB number
/// spaces overlap, and TS 38.300 §16.12.2.1 forbids mapping an end-to-end DRB and an
/// end-to-end SRB into the same relay RLC channel — a rule that cannot be stated, let
/// alone enforced, if the two are the same type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RemoteBearerId {
    /// An end-to-end signalling radio bearer of the remote UE.
    Srb(u8),
    /// An end-to-end data radio bearer of the remote UE.
    Drb(u8),
}

impl RemoteBearerId {
    /// An end-to-end SRB of the remote UE, or `None` if TS 38.331 does not allow it.
    ///
    /// SRB3 is refused even though `INTEGER (0..3)` admits it: TS 38.331's field
    /// description says value 3 "is not supported in this version of the specification"
    /// (`38331-j30.txt:141881`). Refused at construction rather than at encode, so an
    /// unsignallable bearer cannot be held in a mapping table and then fail later.
    pub fn srb(id: u8) -> Option<Self> {
        SRB_ID_RANGE.contains(&id).then_some(Self::Srb(id))
    }

    /// An end-to-end DRB of the remote UE, or `None` if outside `DRB-Identity`'s `1..=32`.
    pub fn drb(id: u8) -> Option<Self> {
        DRB_ID_RANGE.contains(&id).then_some(Self::Drb(id))
    }

    /// Whether this is a data radio bearer.
    pub fn is_drb(self) -> bool {
        matches!(self, Self::Drb(_))
    }

    /// The bearer number, without the SRB/DRB distinction.
    pub fn id(self) -> u8 {
        match self {
            Self::Srb(id) | Self::Drb(id) => id,
        }
    }

    /// The single octet this bearer identity occupies in the SRAP header.
    fn to_octet(self) -> u8 {
        match self {
            Self::Srb(id) => id & BEARER_ID_MASK,
            Self::Drb(id) => BEARER_IS_DRB | (id & BEARER_ID_MASK),
        }
    }

    /// Reads a bearer identity back out of a header octet.
    ///
    /// A value outside the ASN.1 range is an error rather than a clamped bearer: a PDU
    /// claiming DRB 40 is not a PDU for DRB 32, and delivering it to the wrong PDCP
    /// entity would corrupt a bearer that is working.
    fn from_octet(octet: u8) -> Result<Self, RlcError> {
        let id = octet & BEARER_ID_MASK;
        if octet & BEARER_IS_DRB != 0 {
            Self::drb(id).ok_or(RlcError::SrapInvalidBearerId { id, is_drb: true })
        } else {
            Self::srb(id).ok_or(RlcError::SrapInvalidBearerId { id, is_drb: false })
        }
    }
}

impl std::fmt::Display for RemoteBearerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Srb(id) => write!(f, "SRB{id}"),
            Self::Drb(id) => write!(f, "DRB{id}"),
        }
    }
}

/// The SRAP adaptation-layer header (TS 38.300 §16.12.2.1; TS 38.351).
///
/// Carries the two identities §16.12.2.1 requires: the local Remote UE ID the gNB
/// assigned, and which end-to-end Uu radio bearer of that remote UE the payload belongs
/// to. Together they are what lets the receiving end "correlate the received packets for
/// the specific PDCP entity" (`38300-j30.txt:14713`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SrapHeader {
    /// The local Remote UE ID (`sl-LocalIdentity-r17`, `INTEGER (0..255)`).
    ///
    /// Assigned by the gNB, which TS 38.300 §16.12.2.1 makes responsible for avoiding
    /// collisions: *"It is the gNB responsibility to avoid collision on the usage of local
    /// Remote UE ID"* (`38300-j30.txt:14757`).
    pub local_remote_ue_id: u8,
    /// Which end-to-end Uu radio bearer of that remote UE this is.
    pub bearer: RemoteBearerId,
}

impl SrapHeader {
    /// A header for `bearer` of the remote UE known locally as `local_remote_ue_id`.
    pub fn new(local_remote_ue_id: u8, bearer: RemoteBearerId) -> Self {
        Self {
            local_remote_ue_id,
            bearer,
        }
    }

    /// Encodes the header and prepends it to `payload`, giving a SRAP PDU.
    ///
    /// The payload is an end-to-end PDCP PDU of the remote UE, which this sublayer does
    /// not inspect: PDCP is terminated at the remote UE and the gNB, not at either hop
    /// SRAP runs on (TS 38.300 §16.12.2.1), so treating it as opaque is what "end to end"
    /// means here rather than a shortcut.
    pub fn encode(&self, payload: &[u8]) -> Vec<u8> {
        let mut pdu = Vec::with_capacity(SRAP_HEADER_LEN + payload.len());
        pdu.push(self.local_remote_ue_id);
        pdu.push(self.bearer.to_octet());
        pdu.extend_from_slice(payload);
        pdu
    }

    /// Decodes a SRAP PDU into its header and the end-to-end payload it carries.
    ///
    /// A PDU with a header but no payload is refused: a SRAP PDU exists to carry an
    /// end-to-end PDCP PDU, and an empty one would consume a relay RLC channel to deliver
    /// nothing to a PDCP entity.
    pub fn decode(pdu: &[u8]) -> Result<(Self, &[u8]), RlcError> {
        if pdu.len() <= SRAP_HEADER_LEN {
            return Err(RlcError::PduTooShort {
                need: SRAP_HEADER_LEN + 1,
                got: pdu.len(),
            });
        }
        Ok((
            Self {
                local_remote_ue_id: pdu[0],
                bearer: RemoteBearerId::from_octet(pdu[1])?,
            },
            &pdu[SRAP_HEADER_LEN..],
        ))
    }
}

/// Which egress relay RLC channel a remote UE's end-to-end bearer is mapped onto.
///
/// TS 38.300 §16.12.2.1 gives the relay UE two egress interfaces, and TS 38.331's
/// `SL-MappingToAddMod-r17` a separate optional field for each
/// (`tools/rrc-19.3.0.asn1:28942`): `sl-EgressRLC-ChannelUu-r17` for the Uu hop, which its
/// field description scopes to *"uplink transmissions at the L2 U2N Relay UE"*
/// (`38331-j30.txt:141896`), and `sl-EgressRLC-ChannelPC5-r17` for the PC5 hop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EgressChannel {
    /// A Uu relay RLC channel (`Uu-RelayRLC-ChannelID-r17`, `INTEGER (1..maxLC-ID)`,
    /// i.e. `1..=32` — `tools/rrc-19.3.0.asn1:16825`, `:29451`). The uplink direction
    /// towards the gNB.
    Uu(u8),
    /// A PC5 relay RLC channel (`SL-RLC-ChannelID-r17`). The downlink direction, towards
    /// the remote UE.
    Pc5(u16),
}

/// One remote UE's bearer mapping, as `sl-SRAP-ConfigRelay` configures it
/// (`SL-SRAP-Config-r17`, TS 38.331 §5.3.5.17; TS 38.351).
///
/// Holds the local Remote UE ID the gNB assigned and the per-bearer egress mapping. The
/// map is keyed by [`RemoteBearerId`] rather than by a bare number precisely so that an
/// end-to-end SRB and an end-to-end DRB with the same number are two entries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteUeMapping {
    /// The local Remote UE ID, which goes in every SRAP header for this remote UE.
    pub local_remote_ue_id: u8,
    /// Which egress channel each of this remote UE's end-to-end bearers maps onto.
    mappings: BTreeMap<RemoteBearerId, EgressChannel>,
}

impl RemoteUeMapping {
    /// A mapping for the remote UE the gNB assigned `local_remote_ue_id` to, with no
    /// bearers mapped yet.
    pub fn new(local_remote_ue_id: u8) -> Self {
        Self {
            local_remote_ue_id,
            mappings: BTreeMap::new(),
        }
    }

    /// Maps one end-to-end bearer onto an egress relay RLC channel
    /// (`sl-MappingToAddModList-r17`).
    ///
    /// Replaces an existing mapping for the same bearer, which is what
    /// "AddMod" means: TS 38.331 §5.3.5.17 has the UE *"modify the configuration in
    /// accordance with the sl-SRAP-ConfigRelay"* for an entry already present
    /// (`38331-j30.txt:14485`).
    pub fn map_bearer(&mut self, bearer: RemoteBearerId, egress: EgressChannel) {
        self.mappings.insert(bearer, egress);
    }

    /// Releases one bearer's mapping (`sl-MappingToReleaseList-r17`), reporting whether
    /// one was there.
    pub fn release_bearer(&mut self, bearer: RemoteBearerId) -> bool {
        self.mappings.remove(&bearer).is_some()
    }

    /// The egress channel for one of this remote UE's bearers, if it is mapped.
    pub fn egress_for(&self, bearer: RemoteBearerId) -> Option<EgressChannel> {
        self.mappings.get(&bearer).copied()
    }

    /// How many of this remote UE's bearers are mapped.
    pub fn mapped_bearer_count(&self) -> usize {
        self.mappings.len()
    }

    /// Every mapped bearer, in a deterministic order.
    pub fn bearers(&self) -> impl Iterator<Item = (RemoteBearerId, EgressChannel)> + '_ {
        self.mappings.iter().map(|(b, e)| (*b, *e))
    }
}

/// What a SRAP entity decided to do with a PDU.
///
/// A decision type rather than an `Option`, for the reason
/// `sidelink::relay::RelayForwardDecision` is one: a SRAP entity declines for several
/// distinguishable reasons, and collapsing them would make "the remote UE is not served
/// here" indistinguishable from "that bearer is not mapped" — two different
/// misconfigurations with different fixes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SrapDecision {
    /// Submit `pdu` to the egress relay RLC channel `egress`.
    Submit {
        /// The egress relay RLC channel.
        egress: EgressChannel,
        /// The SRAP PDU, header included.
        pdu: Vec<u8>,
    },
    /// No remote UE with that local ID is served by this entity.
    UnknownRemoteUe {
        /// The local Remote UE ID that is not served.
        local_remote_ue_id: u8,
    },
    /// The remote UE is served, but that end-to-end bearer has no egress mapping.
    ///
    /// TS 38.331 §5.3.5.17 configures the mapping per bearer, so a bearer the network has
    /// not mapped has nowhere to go — and inventing a channel for it would send a remote
    /// UE's traffic onto a channel the gNB is not reading for it.
    BearerNotMapped {
        /// The remote UE the bearer belongs to.
        local_remote_ue_id: u8,
        /// The unmapped bearer.
        bearer: RemoteBearerId,
    },
    /// An empty payload, which would carry nothing to a PDCP entity.
    EmptyPayload,
}

impl SrapDecision {
    /// Whether the PDU is being submitted to an egress channel.
    pub fn is_submitted(&self) -> bool {
        matches!(self, Self::Submit { .. })
    }

    /// The egress channel, when the PDU is being submitted.
    pub fn egress(&self) -> Option<EgressChannel> {
        match self {
            Self::Submit { egress, .. } => Some(*egress),
            _ => None,
        }
    }
}

/// A SRAP entity: the adaptation sublayer of one UE or gNB
/// (TS 38.351; TS 38.300 §16.12.2.1).
///
/// TS 38.331 §5.3.5.17 establishes *"a SRAP entity as specified in TS 38.351"*
/// (`38331-j30.txt:14448`) when the network configures relay or remote-UE operation, and
/// configures its per-remote-UE parameters into it. One entity therefore serves every
/// remote UE this node relays for, which is what multiplexing "different L2 U2N Remote
/// UEs ... over the same egress Uu/PC5 Relay RLC channel" (`38300-j30.txt:14705`)
/// requires.
#[derive(Debug, Default)]
pub struct SrapEntity {
    /// Every remote UE this entity is configured for, keyed by local Remote UE ID.
    remote_ues: BTreeMap<u8, RemoteUeMapping>,
    /// SRAP PDUs submitted to an egress channel.
    ///
    /// The observable that proves adaptation happened, incremented only on the submitting
    /// path — the same reason `RelayForwarder` counts octets rather than logging.
    submitted_pdus: u64,
    /// Octets of end-to-end payload carried, header excluded.
    submitted_payload_octets: u64,
    /// PDUs that could not be submitted because the bearer had no mapping.
    dropped_unmapped: u64,
}

impl SrapEntity {
    /// A SRAP entity serving no remote UEs yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// Configures one remote UE's mapping into this entity
    /// (`sl-SRAP-ConfigRelay`/`sl-SRAP-ConfigRemote`, TS 38.331 §5.3.5.17).
    ///
    /// Keyed by the mapping's own local Remote UE ID, so the gNB's assignment is the
    /// single source of that identity rather than a separate argument that could disagree
    /// with the header the entity then writes.
    pub fn configure_remote_ue(&mut self, mapping: RemoteUeMapping) {
        self.remote_ues.insert(mapping.local_remote_ue_id, mapping);
    }

    /// Releases a remote UE's whole mapping (`sl-RemoteUE-ToReleaseList-r17`), reporting
    /// whether one was configured.
    pub fn release_remote_ue(&mut self, local_remote_ue_id: u8) -> bool {
        self.remote_ues.remove(&local_remote_ue_id).is_some()
    }

    /// The mapping configured for one remote UE, if any.
    pub fn remote_ue(&self, local_remote_ue_id: u8) -> Option<&RemoteUeMapping> {
        self.remote_ues.get(&local_remote_ue_id)
    }

    /// Mutable access to a configured remote UE's mapping, for an
    /// add/modify that changes only some bearers.
    pub fn remote_ue_mut(&mut self, local_remote_ue_id: u8) -> Option<&mut RemoteUeMapping> {
        self.remote_ues.get_mut(&local_remote_ue_id)
    }

    /// How many remote UEs this entity is configured for.
    pub fn remote_ue_count(&self) -> usize {
        self.remote_ues.len()
    }

    /// SRAP PDUs submitted to an egress relay RLC channel.
    pub fn submitted_pdus(&self) -> u64 {
        self.submitted_pdus
    }

    /// End-to-end payload octets carried, SRAP headers excluded.
    pub fn submitted_payload_octets(&self) -> u64 {
        self.submitted_payload_octets
    }

    /// PDUs dropped because the bearer had no egress mapping.
    pub fn dropped_unmapped(&self) -> u64 {
        self.dropped_unmapped
    }

    /// **Adapts an end-to-end PDCP PDU into a SRAP PDU** and says which egress relay RLC
    /// channel it goes to (TS 38.300 §16.12.2.1's bearer mapping).
    ///
    /// This is the uplink direction at a relay UE and the downlink direction at a gNB —
    /// the same operation, because §16.12.2.1 describes the same two header fields for
    /// both and the direction is expressed by which [`EgressChannel`] the bearer is
    /// mapped to.
    ///
    /// The counters advance here rather than in a caller, so "SRAP carried this" cannot
    /// be claimed by a path that did not build the header.
    pub fn adapt(
        &mut self,
        local_remote_ue_id: u8,
        bearer: RemoteBearerId,
        payload: &[u8],
    ) -> SrapDecision {
        let Some(mapping) = self.remote_ues.get(&local_remote_ue_id) else {
            return SrapDecision::UnknownRemoteUe { local_remote_ue_id };
        };
        if payload.is_empty() {
            return SrapDecision::EmptyPayload;
        }
        let Some(egress) = mapping.egress_for(bearer) else {
            self.dropped_unmapped += 1;
            return SrapDecision::BearerNotMapped {
                local_remote_ue_id,
                bearer,
            };
        };

        let pdu = SrapHeader::new(local_remote_ue_id, bearer).encode(payload);
        self.submitted_pdus += 1;
        self.submitted_payload_octets += payload.len() as u64;
        SrapDecision::Submit { egress, pdu }
    }

    /// **Reads a received SRAP PDU**, correlating it to the remote UE and end-to-end
    /// bearer it belongs to (TS 38.300 §16.12.2.1's DL correlation at the remote UE and
    /// UL correlation at the gNB).
    ///
    /// Returns the header and the end-to-end payload to hand to that bearer's PDCP
    /// entity. A PDU naming a remote UE this entity is not configured for is an error
    /// rather than a silently accepted one: §16.12.2.1 makes the local Remote UE ID the
    /// key that selects the PDCP entity, so accepting an unknown one would deliver a
    /// remote UE's traffic to whatever entity happened to be there.
    pub fn deliver<'a>(&self, pdu: &'a [u8]) -> Result<(SrapHeader, &'a [u8]), RlcError> {
        let (header, payload) = SrapHeader::decode(pdu)?;
        if !self.remote_ues.contains_key(&header.local_remote_ue_id) {
            return Err(RlcError::SrapUnknownRemoteUe {
                local_remote_ue_id: header.local_remote_ue_id,
            });
        }
        Ok((header, payload))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A local Remote UE ID with bits set across the octet, so a truncation or a swap
    /// with the bearer octet cannot produce it by accident.
    const LOCAL_ID: u8 = 0xA5;
    const UU_CHANNEL: u8 = 7;

    fn drb5() -> RemoteBearerId {
        RemoteBearerId::drb(5).expect("DRB 5 is within 1..=32")
    }

    fn mapping_with_drb5_on_uu() -> RemoteUeMapping {
        let mut mapping = RemoteUeMapping::new(LOCAL_ID);
        mapping.map_bearer(drb5(), EgressChannel::Uu(UU_CHANNEL));
        mapping
    }

    /// Criterion 1's codec round trip: the header's two identities survive, and the
    /// end-to-end payload comes back byte for byte.
    ///
    /// The payload equality is the load-bearing assertion — SRAP must not touch the PDCP
    /// PDU it carries, because PDCP is terminated at the remote UE and the gNB.
    #[test]
    fn a_srap_header_round_trips_with_its_payload_intact() {
        const PAYLOAD: &[u8] = &[0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0xFF];
        let header = SrapHeader::new(LOCAL_ID, drb5());
        let pdu = header.encode(PAYLOAD);

        assert_eq!(pdu.len(), SRAP_HEADER_LEN + PAYLOAD.len());
        let (back, payload) = SrapHeader::decode(&pdu).expect("decode");
        assert_eq!(back, header);
        assert_eq!(back.local_remote_ue_id, LOCAL_ID);
        assert_eq!(back.bearer, drb5());
        assert_eq!(
            payload, PAYLOAD,
            "SRAP must carry PDCP end to end, unaltered"
        );
    }

    /// The exact octets of a SRAP header, pinned.
    ///
    /// Pinned rather than only round-tripped because a round trip passes against any
    /// self-consistent layout, including one that put the two fields in the other order.
    /// Since TS 38.351 is not vendored (see the module docs) this layout is a local
    /// assignment, and pinning it is what makes a change to it visible rather than
    /// silent.
    #[test]
    fn the_srap_header_octets_are_the_local_id_then_the_bearer() {
        let pdu = SrapHeader::new(0x2A, drb5()).encode(&[0x99]);
        // local id, then the DRB discriminator bit set over bearer 5, then the payload.
        assert_eq!(pdu, vec![0x2A, 0x85, 0x99]);

        // And an SRB leaves the discriminator clear, so the two arms differ on the wire.
        let srb = SrapHeader::new(0x2A, RemoteBearerId::srb(1).expect("SRB1")).encode(&[0x99]);
        assert_eq!(srb, vec![0x2A, 0x01, 0x99]);
    }

    /// SRB *n* and DRB *n* are different bearers, and must not decode to each other —
    /// otherwise a data bearer's traffic would reach a signalling bearer's PDCP entity.
    #[test]
    fn an_srb_and_a_drb_with_the_same_number_are_distinguishable() {
        let srb2 = RemoteBearerId::srb(2).expect("SRB2");
        let drb2 = RemoteBearerId::drb(2).expect("DRB2");
        assert_ne!(srb2, drb2);

        let (srb_back, _) = SrapHeader::decode(&SrapHeader::new(1, srb2).encode(&[0x01]))
            .expect("the SRB PDU decodes");
        let (drb_back, _) = SrapHeader::decode(&SrapHeader::new(1, drb2).encode(&[0x01]))
            .expect("the DRB PDU decodes");
        assert_eq!(srb_back.bearer, srb2);
        assert_eq!(drb_back.bearer, drb2);
        assert!(drb_back.bearer.is_drb() && !srb_back.bearer.is_drb());
    }

    /// SRB3 is refused, as TS 38.331's field description requires, even though the ASN.1
    /// `INTEGER (0..3)` range admits it.
    #[test]
    fn srb3_is_refused_because_ts_38_331_does_not_support_it() {
        assert_eq!(RemoteBearerId::srb(3), None);
        // And the values that ARE supported are accepted, so this is not refusing
        // everything.
        for ok in [0u8, 1, 2] {
            assert!(RemoteBearerId::srb(ok).is_some(), "SRB{ok} must be allowed");
        }
        assert_eq!(DEFAULT_SRAP_SRB, 1, "TS 38.331 §9.2.5 defaults to SRB1");
    }

    /// A DRB identity outside `DRB-Identity`'s `1..=32` is refused rather than clamped.
    #[test]
    fn a_drb_identity_outside_the_asn1_range_is_refused() {
        for bad in [0u8, 33, 255] {
            assert_eq!(RemoteBearerId::drb(bad), None, "DRB {bad} must be refused");
        }
        for ok in [1u8, 32] {
            assert!(RemoteBearerId::drb(ok).is_some());
        }
    }

    /// A header octet naming a bearer outside the ASN.1 range is an error, not a bearer
    /// with a clamped identity.
    #[test]
    fn a_pdu_naming_an_out_of_range_bearer_is_refused() {
        // Discriminator set, bearer 0 — DRB 0 does not exist (`1..=32`).
        let bad_drb = vec![LOCAL_ID, BEARER_IS_DRB, 0xAA];
        assert!(SrapHeader::decode(&bad_drb).is_err());
        // Discriminator clear, bearer 3 — SRB3, which TS 38.331 excludes.
        let srb3 = vec![LOCAL_ID, 0x03, 0xAA];
        assert!(SrapHeader::decode(&srb3).is_err());
    }

    /// A PDU with a header and no payload is refused: SRAP exists to carry an end-to-end
    /// PDCP PDU, and an empty one delivers nothing.
    #[test]
    fn a_pdu_with_no_payload_is_refused() {
        assert!(SrapHeader::decode(&[LOCAL_ID, 0x85]).is_err());
        assert!(SrapHeader::decode(&[LOCAL_ID]).is_err());
        assert!(SrapHeader::decode(&[]).is_err());
    }

    /// The whole adaptation step: a configured bearer is adapted onto its egress channel,
    /// and the counters move only on that path.
    #[test]
    fn a_mapped_bearer_is_adapted_onto_its_egress_channel() {
        const PAYLOAD: &[u8] = &[0x01, 0x02, 0x03, 0x04];
        let mut srap = SrapEntity::new();
        srap.configure_remote_ue(mapping_with_drb5_on_uu());
        assert_eq!(srap.remote_ue_count(), 1);

        let decision = srap.adapt(LOCAL_ID, drb5(), PAYLOAD);
        assert!(decision.is_submitted());
        assert_eq!(decision.egress(), Some(EgressChannel::Uu(UU_CHANNEL)));

        let SrapDecision::Submit { pdu, .. } = decision else {
            panic!("a mapped bearer must be submitted");
        };
        // The PDU really carries the payload, and the header addresses the right bearer.
        let (header, carried) = SrapHeader::decode(&pdu).expect("the built PDU decodes");
        assert_eq!(header.local_remote_ue_id, LOCAL_ID);
        assert_eq!(header.bearer, drb5());
        assert_eq!(carried, PAYLOAD);

        // Positive counters, reachable only through `adapt`.
        assert_eq!(srap.submitted_pdus(), 1);
        assert_eq!(srap.submitted_payload_octets(), PAYLOAD.len() as u64);
        assert_eq!(srap.dropped_unmapped(), 0);
    }

    /// A bearer the network has not mapped is not adapted, and is distinguishable from an
    /// unknown remote UE.
    #[test]
    fn an_unmapped_bearer_is_not_adapted_and_is_counted() {
        let mut srap = SrapEntity::new();
        srap.configure_remote_ue(mapping_with_drb5_on_uu());

        let other = RemoteBearerId::drb(9).expect("DRB9");
        assert_eq!(
            srap.adapt(LOCAL_ID, other, &[0xAA]),
            SrapDecision::BearerNotMapped {
                local_remote_ue_id: LOCAL_ID,
                bearer: other,
            }
        );
        assert_eq!(srap.dropped_unmapped(), 1);
        assert_eq!(srap.submitted_pdus(), 0);
    }

    /// A remote UE this entity is not configured for is refused, distinguishably.
    #[test]
    fn an_unknown_remote_ue_is_refused() {
        let mut srap = SrapEntity::new();
        srap.configure_remote_ue(mapping_with_drb5_on_uu());
        assert_eq!(
            srap.adapt(LOCAL_ID.wrapping_add(1), drb5(), &[0xAA]),
            SrapDecision::UnknownRemoteUe {
                local_remote_ue_id: LOCAL_ID.wrapping_add(1),
            }
        );
        assert_eq!(srap.submitted_pdus(), 0);
        // And nothing was counted as an unmapped bearer: the two are different faults.
        assert_eq!(srap.dropped_unmapped(), 0);
    }

    /// An empty payload is not adapted: it would carry nothing to a PDCP entity.
    #[test]
    fn an_empty_payload_is_not_adapted() {
        let mut srap = SrapEntity::new();
        srap.configure_remote_ue(mapping_with_drb5_on_uu());
        assert_eq!(
            srap.adapt(LOCAL_ID, drb5(), &[]),
            SrapDecision::EmptyPayload
        );
        assert_eq!(srap.submitted_pdus(), 0);
    }

    /// Releasing a remote UE's mapping stops the adaptation. The guard that proves
    /// `adapt` consults the live configuration rather than a table written once.
    #[test]
    fn releasing_a_remote_ue_stops_the_adaptation() {
        let mut srap = SrapEntity::new();
        srap.configure_remote_ue(mapping_with_drb5_on_uu());
        assert!(srap.adapt(LOCAL_ID, drb5(), &[0xAA]).is_submitted());

        assert!(srap.release_remote_ue(LOCAL_ID), "a mapping was configured");
        assert_eq!(
            srap.adapt(LOCAL_ID, drb5(), &[0xAA]),
            SrapDecision::UnknownRemoteUe {
                local_remote_ue_id: LOCAL_ID
            }
        );
        // The first submission still counted; the second did not.
        assert_eq!(srap.submitted_pdus(), 1);
        assert!(!srap.release_remote_ue(LOCAL_ID), "already released");
    }

    /// Releasing ONE bearer leaves the remote UE's others mapped
    /// (`sl-MappingToReleaseList-r17` is per bearer, not per UE).
    #[test]
    fn releasing_one_bearer_leaves_the_others_mapped() {
        let drb6 = RemoteBearerId::drb(6).expect("DRB6");
        let mut mapping = mapping_with_drb5_on_uu();
        mapping.map_bearer(drb6, EgressChannel::Uu(8));
        assert_eq!(mapping.mapped_bearer_count(), 2);

        let mut srap = SrapEntity::new();
        srap.configure_remote_ue(mapping);
        assert!(srap
            .remote_ue_mut(LOCAL_ID)
            .expect("configured")
            .release_bearer(drb5()));

        assert!(matches!(
            srap.adapt(LOCAL_ID, drb5(), &[0xAA]),
            SrapDecision::BearerNotMapped { .. }
        ));
        // DRB6 still goes.
        assert_eq!(
            srap.adapt(LOCAL_ID, drb6, &[0xAA]).egress(),
            Some(EgressChannel::Uu(8))
        );
    }

    /// Two remote UEs multiplex onto the SAME egress relay RLC channel, which is what
    /// TS 38.300 §16.12.2.1 requires: *"different end-to-end Uu Radio Bearers (SRBs or
    /// DRBs) of the same L2 U2N Remote UE and/or different L2 U2N Remote UEs can be
    /// multiplexed over the same egress Uu/PC5 Relay RLC channel"*.
    ///
    /// The local Remote UE ID in each header is what keeps them apart at the far end, so
    /// that is what is asserted.
    #[test]
    fn two_remote_ues_multiplex_onto_one_egress_channel_and_stay_distinguishable() {
        const OTHER_LOCAL_ID: u8 = 0x5A;
        let mut srap = SrapEntity::new();
        srap.configure_remote_ue(mapping_with_drb5_on_uu());
        let mut other = RemoteUeMapping::new(OTHER_LOCAL_ID);
        // The SAME Uu channel: this is the multiplexing.
        other.map_bearer(drb5(), EgressChannel::Uu(UU_CHANNEL));
        srap.configure_remote_ue(other);

        let first = srap.adapt(LOCAL_ID, drb5(), &[0x11]);
        let second = srap.adapt(OTHER_LOCAL_ID, drb5(), &[0x22]);
        assert_eq!(first.egress(), second.egress(), "one egress channel");

        let (SrapDecision::Submit { pdu: a, .. }, SrapDecision::Submit { pdu: b, .. }) =
            (first, second)
        else {
            panic!("both must be submitted");
        };
        // Same bearer number, same channel, different remote UE: only the local ID
        // distinguishes them, and it must.
        let (ha, pa) = srap.deliver(&a).expect("deliver a");
        let (hb, pb) = srap.deliver(&b).expect("deliver b");
        assert_eq!(ha.local_remote_ue_id, LOCAL_ID);
        assert_eq!(hb.local_remote_ue_id, OTHER_LOCAL_ID);
        assert_ne!(ha.local_remote_ue_id, hb.local_remote_ue_id);
        assert_eq!(pa, &[0x11]);
        assert_eq!(pb, &[0x22]);
        assert_eq!(srap.submitted_pdus(), 2);
    }

    /// A PDU for a remote UE this entity does not serve is refused on delivery, rather
    /// than handed to whatever PDCP entity happened to be configured.
    #[test]
    fn delivering_a_pdu_for_an_unserved_remote_ue_is_refused() {
        let mut srap = SrapEntity::new();
        srap.configure_remote_ue(mapping_with_drb5_on_uu());
        // Built by hand for a remote UE that is not configured here.
        let foreign = SrapHeader::new(LOCAL_ID.wrapping_add(1), drb5()).encode(&[0xAA]);
        assert!(srap.deliver(&foreign).is_err());
        // The served one still delivers, so this is not refusing everything.
        let served = SrapHeader::new(LOCAL_ID, drb5()).encode(&[0xAA]);
        assert!(srap.deliver(&served).is_ok());
    }

    /// Re-configuring a bearer replaces its egress channel rather than adding a second
    /// mapping — which is what `sl-MappingToAddModList`'s "Mod" means.
    #[test]
    fn re_mapping_a_bearer_replaces_its_egress_channel() {
        let mut mapping = mapping_with_drb5_on_uu();
        mapping.map_bearer(drb5(), EgressChannel::Pc5(12));
        assert_eq!(mapping.mapped_bearer_count(), 1, "replaced, not added");
        assert_eq!(mapping.egress_for(drb5()), Some(EgressChannel::Pc5(12)));
    }
}
