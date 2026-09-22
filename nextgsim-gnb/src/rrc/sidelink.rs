//! The gNB's answer to a UE's sidelink resource request, and its L2 UE-to-Network relay
//! configuration (TS 38.331 §5.8.3, §5.3.5.3, §5.3.5.17; TS 38.300 §16.12.2.1;
//! issues #141, #190).
//!
//! # What this closes
//!
//! Nothing in this tree produced or consumed a `SidelinkUEInformation`, and
//! `RRCReconfiguration` had no `sl-ConfigDedicatedNR` field at all. A UE running PC5
//! therefore did so with no network grant: the gNB never learned sidelink was in use and
//! allocated nothing for it. TS 38.331 §5.8.3.2 makes the request the UE's obligation and
//! §5.3.5.3 makes applying the grant the UE's action on receiving one — so both halves
//! were missing, not just one.
//!
//! This is the network half: a request arrives, and the gNB decides what to grant.
//!
//! # Two grants, because there are two kinds of UE asking
//!
//! [`sl_config_for_request`] answers a **plain PC5 UE**: an `sl-ConfigDedicatedNR-r16`
//! `setup` carrying an `sl-PHY-MAC-RLC-Config` and a `t400`. Its presence is the substance
//! — it tells the UE the network permits sidelink on the requested carriers, which is the
//! **Mode-2** (UE-autonomous resource selection) case of TS 38.300 §16.9. Unchanged since
//! issue #141.
//!
//! [`sl_config_for_relay`] answers a UE that declared `ue-Type-r17`, i.e. one asking for L2
//! UE-to-Network relay resources (issue #190). That grant additionally carries a **Mode-1**
//! `sl-ScheduledConfig-r16` SL-RNTI and a relay RLC channel. Issue #190's criterion 3 asked
//! for that choice to be made explicitly; the full reasoning — including why the SL-RNTI is
//! now signalled and why the resource **pools** still are not — is on
//! [`sl_config_for_relay`], and the short version is that an RNTI is an identity while a
//! pool is a slot, and this simulator has no PC5 PHY for a slot to mean anything to.
//!
//! # Why the gNB checks the request rather than granting unconditionally
//!
//! A UE may ask for carriers the cell does not operate. Granting those would tell the UE
//! it may transmit on a frequency the network has no allocation for, and the UE would
//! believe it. [`sl_config_for_request`] grants only when the request is one this cell
//! can honour, and says why when it declines. A declared relay role does **not** bypass
//! that check — `a_relay_asking_for_ungrantable_carriers_is_still_refused` is the guard.
//!
//! # The local Remote UE ID is allocated here, because §16.12.2.1 says it must be
//!
//! *"It is the gNB responsibility to avoid collision on the usage of local Remote UE ID"*
//! (`38300-j30.txt:14757`). [`LocalRemoteUeIdAllocator`] is that responsibility, and
//! [`relay_configs_for_remote_ue`] builds the relay's and the remote UE's halves **together**
//! so the ID the relay is told to expect cannot drift from the one the remote UE is told to
//! write — a mismatch that would drop every relayed packet with nothing to say why.

use nextgsim_rlc::srap::{EgressChannel, RemoteBearerId, RemoteUeMapping, DEFAULT_SRAP_SRB};
use nextgsim_rrc::procedures::sidelink_relay_config::{
    L2RelayUeConfigParams, L2RemoteUeConfigParams, RelayRemoteUeConfig, MAX_REMOTE_UES,
};
use nextgsim_rrc::procedures::sidelink_ue_information::{
    SidelinkUeInformationParams, SlConfigDedicatedParams, SlRlcBearerConfig,
};
use tracing::{debug, info};

/// The `t400` this gNB configures, in milliseconds.
///
/// 400 ms, one of the eight values `SL-ConfigDedicatedNR-r16.t400-r16` allows. T400 is
/// the sidelink RRC reconfiguration guard timer: it bounds how long a UE waits for a peer
/// to answer an `RRCReconfigurationSidelink` before declaring the PC5 RRC procedure
/// failed. Chosen at the middle of the range rather than at an extreme — the two ends
/// here are in one process, so no configured value is derivable from a measured
/// round-trip time, and picking a boundary value would imply one was.
pub const SL_T400_MS: u16 = 400;

/// The highest sidelink carrier index this cell will grant
/// (`maxNrofFreqSL-r16` is 8, and `SL-InterestedFreqList-r16` entries are `1..=8`).
///
/// This cell operates one sidelink carrier, index 1. A UE asking for index 5 is asking
/// for a carrier this cell does not have, and is not granted it.
pub const SL_MAX_GRANTED_FREQ_INDEX: u8 = 1;

/// Why a sidelink request was not granted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlGrantRefusal {
    /// The UE asked for no resources at all — neither to receive nor to transmit.
    ///
    /// TS 38.331 §5.8.3.2 also uses an empty `SidelinkUEInformation` to signal that the
    /// UE is **no longer** interested, so this is not malformed: it is a withdrawal, and
    /// the right answer is to grant nothing rather than to grant a default.
    NoResourcesRequested,
    /// Every carrier the UE asked to receive on is one this cell does not operate.
    NoGrantableCarrier {
        /// What the UE asked for.
        requested: Vec<u8>,
        /// The highest index this cell can grant.
        max_granted: u8,
    },
}

impl std::fmt::Display for SlGrantRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoResourcesRequested => {
                write!(f, "the UE requested no sidelink resources")
            }
            Self::NoGrantableCarrier {
                requested,
                max_granted,
            } => write!(
                f,
                "none of the requested sidelink carriers {requested:?} is within this \
                 cell's 1..={max_granted}"
            ),
        }
    }
}

/// Decides the `sl-ConfigDedicatedNR` this gNB grants for a UE's request, or why it does
/// not (TS 38.331 §5.8.3, §6.3.2).
///
/// A request is granted when the UE asked for something this cell can honour: either a
/// receive carrier within this cell's range, or transmission resources for at least one
/// destination. The transmit side is not range-checked against carriers, because
/// `SL-TxResourceReq-r16.sl-TxInterestedFreqList` is optional and this tree's UE does not
/// populate it — a UE that named no transmit carrier is asking for the cell's, which is
/// index 1.
pub fn sl_config_for_request(
    request: &SidelinkUeInformationParams,
) -> Result<SlConfigDedicatedParams, SlGrantRefusal> {
    if request.rx_interested_freqs.is_empty() && request.tx_resource_requests.is_empty() {
        return Err(SlGrantRefusal::NoResourcesRequested);
    }

    // A receive interest naming only carriers this cell does not operate is refused.
    // Checked only when the UE named some: a transmit-only request names none, and
    // refusing that would decline a UE that asked for exactly what this cell has.
    if !request.rx_interested_freqs.is_empty()
        && !request
            .rx_interested_freqs
            .iter()
            .any(|f| (1..=SL_MAX_GRANTED_FREQ_INDEX).contains(f))
    {
        return Err(SlGrantRefusal::NoGrantableCarrier {
            requested: request.rx_interested_freqs.clone(),
            max_granted: SL_MAX_GRANTED_FREQ_INDEX,
        });
    }

    info!(
        "Granting sidelink: {} receive carrier(s), {} transmit destination(s), t400={SL_T400_MS}ms",
        request.rx_interested_freqs.len(),
        request.tx_resource_requests.len()
    );
    Ok(SlConfigDedicatedParams {
        t400_ms: Some(SL_T400_MS),
        // The presence of the PHY/MAC/RLC config is the grant (see the module docs).
        phy_mac_rlc_config: true,
        // Mode 2 for a plain PC5 UE: no SL-RNTI, no relay RLC channel. Issue #190's
        // criterion 3 asked this to be decided explicitly, and this is half the answer —
        // see `sl_config_for_relay` for the other half and the reasoning.
        sl_rnti: None,
        sl_rlc_bearers: Vec::new(),
    })
}

/// The first Uu relay RLC channel index this cell assigns to a relay
/// (`Uu-RelayRLC-ChannelID-r17`, `INTEGER (1..32)`).
///
/// 1 rather than 0 because the ASN.1 range is 1-based; a 0 here would be unencodable.
pub const FIRST_UU_RELAY_RLC_CHANNEL: u8 = 1;

/// The `sl-RLC-BearerConfigIndex-r16` this cell gives a relay's first relay RLC channel.
pub const FIRST_SL_RLC_BEARER_INDEX: u16 = 1;

/// The end-to-end bearer of a remote UE this cell relays.
///
/// SRB1, which TS 38.331 §9.2.5 names as the default SRAP bearer: *"`>sl-RemoteUE-RB-Identity`
/// SRB1"* (`38331-j30.txt:149881`). A constant rather than a per-UE choice because this gNB
/// configures one end-to-end bearer per remote UE; a DRB would need a PDU session for the
/// remote UE, which is the part deliberately split out (see the PR).
pub const DEFAULT_SRAP_BEARER: RemoteBearerId = RemoteBearerId::Srb(DEFAULT_SRAP_SRB);

/// The SL-RNTI this gNB schedules a relay UE's sidelink with (TS 38.300 §16.9.3.2).
///
/// Derived from the UE id rather than allocated from a separate pool: this simulator
/// already identifies a UE by `ue_id` everywhere, and a second allocator could disagree
/// with the first about which UE a grant belongs to — a bug whose symptom would be a grant
/// applied by the wrong UE.
///
/// `RNTI-Value` is `INTEGER (0..65535)`, so the id is masked to 16 bits. Offset above the
/// reserved range so that an SL-RNTI cannot be 0: TS 38.321 reserves 0 as an invalid RNTI,
/// and a grant naming RNTI 0 would be one no UE can match.
pub fn sl_rnti_for_ue(ue_id: i32) -> u16 {
    const SL_RNTI_BASE: u16 = 0x4600;
    SL_RNTI_BASE.wrapping_add((ue_id as u16) & 0x00FF)
}

/// Decides the `sl-ConfigDedicatedNR` for a UE acting as an **L2 UE-to-Network relay**
/// (TS 38.300 §16.9.3.2, §16.12.2.1; TS 38.331 §6.3.2; issue #190's criterion 3).
///
/// # The Mode-1 decision, made explicitly
///
/// Issue #141 granted Mode 2 only and said why: `sl-ScheduledConfig-r16`'s resource pools
/// would describe slots in a PC5 physical layer this simulator does not have, and
/// signalling a pool nothing consults is the unread-IE defect that issue existed to
/// remove. Issue #190's criterion 3 asks whether that changes for the relay case.
///
/// **It changes for the identity, not for the pools.** Two parts, decided separately:
///
/// * **`sl-RNTI-r16` IS now granted for a relay.** TS 38.300 §16.12.2.1 has the relay
///   carry remote-UE traffic on relay RLC channels *the network configured*, so the gNB is
///   scheduling that traffic and needs a name for the UE it is scheduling. An RNTI is an
///   identity rather than a slot, so it is meaningful without a PHY — exactly as the
///   C-RNTI already is throughout this tree. And the relay's SRAP mappings are checked
///   against the channels granted here, so this is an IE that is *read*.
/// * **The resource pools are still NOT granted.** `sl-ConfiguredGrantConfigList-r16`,
///   `sl-PSFCH-ToPUCCH-r16` and `mac-MainConfigSL-r16` stay absent, because #141's
///   reasoning about them is untouched by this issue: there is still no PC5 PHY for a pool
///   to mean anything to.
///
/// So the honest description is **Mode 1 identity with nominal pools**, and that is what
/// the docs-book row says rather than claiming full Mode-1 scheduling.
///
/// One relay RLC channel is granted, which is what a single-hop relay needs: TS 38.300
/// §16.12.2.1 permits many end-to-end bearers to multiplex onto one egress channel, and
/// this cell has no reason to split them until something differentiates their treatment.
pub fn sl_config_for_relay(
    request: &SidelinkUeInformationParams,
    sl_rnti: u16,
) -> Result<SlConfigDedicatedParams, SlGrantRefusal> {
    let mut granted = sl_config_for_request(request)?;
    granted.sl_rnti = Some(sl_rnti);
    granted.sl_rlc_bearers = vec![SlRlcBearerConfig {
        index: FIRST_SL_RLC_BEARER_INDEX,
        // The `SLRB-Uu-ConfigIndex-r16` this relay RLC channel serves. Set to the same
        // index: this cell configures one relay RLC channel carrying one sidelink radio
        // bearer configuration, so a second number would imply a mapping that does not
        // exist.
        served_radio_bearer: Some(FIRST_SL_RLC_BEARER_INDEX),
    }];
    info!(
        "Granting L2 UE-to-Network relay resources: SL-RNTI {:#06x}, {} relay RLC channel(s)",
        sl_rnti,
        granted.sl_rlc_bearers.len()
    );
    Ok(granted)
}

/// Logs a refusal at the level its cause warrants.
///
/// A withdrawal is routine and logged at `debug`; an ungrantable carrier is a
/// configuration mismatch between UE and cell and is worth an `info`. Split out so the
/// caller's match arm does not have to make that judgement inline.
pub fn log_refusal(ue_id: i32, refusal: &SlGrantRefusal) {
    match refusal {
        SlGrantRefusal::NoResourcesRequested => {
            debug!("UE[{ue_id}] withdrew its sidelink interest; granting nothing");
        }
        SlGrantRefusal::NoGrantableCarrier { .. } => {
            info!("Not granting sidelink to UE[{ue_id}]: {refusal}");
        }
    }
}

/// Assigns the local Remote UE IDs a relay uses in its SRAP headers, avoiding collisions
/// (TS 38.300 §16.12.2.1; issue #190).
///
/// # Why the gNB owns this
///
/// §16.12.2.1 is explicit: *"It is the gNB responsibility to avoid collision on the usage
/// of local Remote UE ID"* (`38300-j30.txt:14757`). A collision would make two remote UEs
/// indistinguishable in the SRAP header, so the relay would deliver one remote UE's
/// traffic to the other's PDCP entity — and neither UE could detect it.
///
/// One allocator per relay, because the identity space is per relay: §16.12.2.1 scopes the
/// local Remote UE ID to the SRAP header between one relay and the gNB, so two different
/// relays may each use ID 1 for different remote UEs without ambiguity.
#[derive(Debug, Default)]
pub struct LocalRemoteUeIdAllocator {
    /// Which remote UE (by PC5 Layer-2 ID) holds which local Remote UE ID.
    ///
    /// Keyed by Layer-2 ID so that re-requesting for a remote UE already served returns
    /// the SAME local ID rather than allocating a second one — otherwise a relay
    /// reconfigured twice would hold two mappings for one remote UE and the gNB would not
    /// know which header to expect.
    assigned: std::collections::BTreeMap<u32, u8>,
}

impl LocalRemoteUeIdAllocator {
    /// An allocator with nothing assigned.
    pub fn new() -> Self {
        Self::default()
    }

    /// The local Remote UE ID for `remote_l2_id`, allocating one if it has none.
    ///
    /// `None` when every ID is taken — `sl-LocalIdentity-r17` is `INTEGER (0..255)` and
    /// `maxNrofRemoteUE-r17` is 32, so this cannot happen below the 32-remote-UE limit, but
    /// it is returned rather than wrapped: a wrapped ID would be a collision, which is the
    /// one thing §16.12.2.1 makes the gNB responsible for preventing.
    pub fn assign(&mut self, remote_l2_id: u32) -> Option<u8> {
        if let Some(existing) = self.assigned.get(&remote_l2_id) {
            return Some(*existing);
        }
        if self.assigned.len() >= MAX_REMOTE_UES {
            return None;
        }
        // The lowest unused ID, so the assignment is deterministic and a test can state
        // which ID a given remote UE gets.
        let next =
            (0u8..=u8::MAX).find(|candidate| !self.assigned.values().any(|v| v == candidate));
        if let Some(id) = next {
            self.assigned.insert(remote_l2_id, id);
        }
        next
    }

    /// The local Remote UE ID assigned to `remote_l2_id`, without allocating one.
    pub fn assigned_id(&self, remote_l2_id: u32) -> Option<u8> {
        self.assigned.get(&remote_l2_id).copied()
    }

    /// Releases a remote UE's local ID, so a later remote UE may reuse it.
    pub fn release(&mut self, remote_l2_id: u32) -> Option<u8> {
        self.assigned.remove(&remote_l2_id)
    }

    /// How many remote UEs hold a local ID.
    pub fn len(&self) -> usize {
        self.assigned.len()
    }

    /// Whether no remote UE holds a local ID.
    pub fn is_empty(&self) -> bool {
        self.assigned.is_empty()
    }
}

/// Builds the relay's `sl-L2RelayUE-Config` and the remote UE's `sl-L2RemoteUE-Config`
/// for one remote UE reaching the network through one relay
/// (TS 38.300 §16.12.2.1; TS 38.331 §5.3.5.17; issue #190).
///
/// Returns the pair because they must agree: the local Remote UE ID the relay is told to
/// expect in the header is the one the remote UE is told to write, and building them apart
/// would let the two drift. That agreement is the whole correctness condition of the
/// adaptation layer — a relay expecting ID 3 and a remote UE writing ID 4 drops every
/// packet.
///
/// `bearer` is the remote UE's end-to-end Uu bearer being relayed, and it is mapped onto
/// the Uu relay RLC channel [`FIRST_UU_RELAY_RLC_CHANNEL`] that
/// [`sl_config_for_relay`] granted — so the mapping names a channel the relay was actually
/// configured with, rather than one invented here.
pub fn relay_configs_for_remote_ue(
    remote_l2_id: u32,
    local_remote_ue_id: u8,
    bearer: RemoteBearerId,
    remote_ue_c_rnti: u16,
) -> (L2RelayUeConfigParams, L2RemoteUeConfigParams) {
    // The relay's side: adapt this remote UE's bearer onto the relay's egress Uu channel,
    // which is the uplink direction of §16.12.2.1's bearer mapping.
    let mut relay_mapping = RemoteUeMapping::new(local_remote_ue_id);
    relay_mapping.map_bearer(bearer, EgressChannel::Uu(FIRST_UU_RELAY_RLC_CHANNEL));

    // The remote UE's side: the same bearer, but its egress is the PC5 hop towards the
    // relay. The local ID is the SAME value, which is the point of returning both here.
    let mut remote_mapping = RemoteUeMapping::new(local_remote_ue_id);
    remote_mapping.map_bearer(
        bearer,
        EgressChannel::Pc5(u16::from(FIRST_UU_RELAY_RLC_CHANNEL)),
    );

    (
        L2RelayUeConfigParams {
            remote_ues: vec![RelayRemoteUeConfig {
                remote_l2_id,
                srap: relay_mapping,
            }],
            release_remote_l2_ids: Vec::new(),
        },
        L2RemoteUeConfigParams {
            srap: Some(remote_mapping),
            remote_ue_c_rnti: Some(remote_ue_c_rnti),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use nextgsim_rrc::procedures::sidelink_ue_information::{
        SlCastType, SlTxResourceRequest, SlUeType,
    };

    fn request(rx: Vec<u8>, destinations: Vec<u32>) -> SidelinkUeInformationParams {
        SidelinkUeInformationParams {
            rx_interested_freqs: rx,
            tx_resource_requests: destinations
                .into_iter()
                .map(|destination_l2_id| SlTxResourceRequest {
                    destination_l2_id,
                    cast_type: SlCastType::Unicast,
                })
                .collect(),
            ue_type: None,
        }
    }

    /// A relay's request: the same, plus the declared `ue-Type-r17` role.
    fn relay_request(destinations: Vec<u32>) -> SidelinkUeInformationParams {
        SidelinkUeInformationParams {
            ue_type: Some(SlUeType::RelayUe),
            ..request(vec![1], destinations)
        }
    }

    /// A request this cell can honour is granted, with a `t400` the ASN.1 can carry and
    /// the PHY/MAC/RLC presence that constitutes the grant.
    #[test]
    fn a_grantable_request_is_granted_with_a_signallable_t400() {
        let granted = sl_config_for_request(&request(vec![1], vec![0x00_00_02])).expect("granted");
        assert_eq!(granted.t400_ms, Some(SL_T400_MS));
        assert!(granted.phy_mac_rlc_config);
        // And the value the gNB picked must actually be signallable, or the
        // reconfiguration that carries it fails to build.
        assert!(
            nextgsim_rrc::procedures::sidelink_ue_information::build_sl_config_dedicated(&granted)
                .is_ok(),
            "the gNB's configured t400 must be one the ASN.1 ENUMERATED can carry"
        );
    }

    /// An empty request is a withdrawal (TS 38.331 §5.8.3.2), not a malformed message:
    /// nothing is granted, and the reason says so.
    #[test]
    fn an_empty_request_is_a_withdrawal_and_grants_nothing() {
        assert_eq!(
            sl_config_for_request(&request(Vec::new(), Vec::new())),
            Err(SlGrantRefusal::NoResourcesRequested)
        );
    }

    /// A UE asking only for carriers this cell does not operate is refused, rather than
    /// being told it may transmit on a frequency the network has not allocated.
    #[test]
    fn a_request_for_only_ungrantable_carriers_is_refused() {
        let refusal = sl_config_for_request(&request(vec![5, 7], vec![0x00_00_02]))
            .expect_err("carriers 5 and 7 are outside this cell's 1..=1");
        assert_eq!(
            refusal,
            SlGrantRefusal::NoGrantableCarrier {
                requested: vec![5, 7],
                max_granted: SL_MAX_GRANTED_FREQ_INDEX,
            }
        );
        // The message names the mismatch, so an operator can see which side to fix.
        assert!(refusal.to_string().contains("[5, 7]"), "{refusal}");
    }

    /// A request that names ONE grantable carrier among several is granted: the cell
    /// honours what it can rather than refusing the whole request.
    #[test]
    fn a_request_naming_one_grantable_carrier_among_several_is_granted() {
        assert!(sl_config_for_request(&request(vec![4, 1, 6], Vec::new())).is_ok());
    }

    /// A transmit-only request names no receive carrier, and must not be refused for it:
    /// the UE is asking for this cell's carrier by not naming one.
    #[test]
    fn a_transmit_only_request_is_granted() {
        let granted =
            sl_config_for_request(&request(Vec::new(), vec![0x00_00_02])).expect("granted");
        assert!(granted.phy_mac_rlc_config);
    }

    // ── Issue #190: the L2 UE-to-Network relay grant ─────────────────────────────

    const REMOTE_L2_ID: u32 = 0x00_0A_00_01 & 0x00FF_FFFF;
    const OTHER_REMOTE_L2_ID: u32 = 0x00_0B_00_02 & 0x00FF_FFFF;
    const SL_RNTI: u16 = 0x4601;

    /// Criterion 3, the decided half: a relay's grant carries a Mode-1 SL-RNTI and a relay
    /// RLC channel, where a plain PC5 UE's carries neither.
    ///
    /// The contrast with `a_plain_pc5_grant_stays_mode_2` is what makes this meaningful:
    /// asserting the relay grant alone would pass against a gNB that gave everyone Mode 1.
    #[test]
    fn a_relay_grant_carries_a_mode_1_rnti_and_a_relay_rlc_channel() {
        let granted =
            sl_config_for_relay(&relay_request(vec![REMOTE_L2_ID]), SL_RNTI).expect("granted");
        assert_eq!(granted.sl_rnti, Some(SL_RNTI));
        assert_eq!(granted.sl_rlc_bearers.len(), 1);
        assert_eq!(granted.sl_rlc_bearers[0].index, FIRST_SL_RLC_BEARER_INDEX);
        // And it must still be encodable, or the reconfiguration carrying it fails to
        // build and the relay is granted nothing at all.
        assert!(
            nextgsim_rrc::procedures::sidelink_ue_information::build_sl_config_dedicated(&granted)
                .is_ok(),
            "the relay grant this gNB chose must be encodable"
        );
    }

    /// A plain PC5 UE stays on Mode 2, so issue #141's behaviour is unchanged for
    /// non-relay UEs.
    #[test]
    fn a_plain_pc5_grant_stays_mode_2() {
        let granted =
            sl_config_for_request(&request(vec![1], vec![REMOTE_L2_ID])).expect("granted");
        assert_eq!(
            granted.sl_rnti, None,
            "a plain PC5 UE must not be given a scheduled-mode RNTI"
        );
        assert!(granted.sl_rlc_bearers.is_empty());
    }

    /// A relay whose carriers this cell cannot honour is refused, exactly as a plain UE is:
    /// declaring a relay role does not bypass the carrier check.
    #[test]
    fn a_relay_asking_for_ungrantable_carriers_is_still_refused() {
        let ungrantable = SidelinkUeInformationParams {
            ue_type: Some(SlUeType::RelayUe),
            ..request(vec![5, 7], vec![REMOTE_L2_ID])
        };
        assert!(sl_config_for_relay(&ungrantable, SL_RNTI).is_err());
    }

    /// **The correctness condition of the whole adaptation layer**: the local Remote UE ID
    /// the relay is told to expect is the one the remote UE is told to write.
    ///
    /// A relay expecting ID 3 and a remote UE writing ID 4 drops every packet, and neither
    /// end can tell why — so this is asserted directly rather than left to the E2E.
    #[test]
    fn the_relay_and_the_remote_ue_are_told_the_same_local_id() {
        const LOCAL_ID: u8 = 7;
        let (relay, remote) =
            relay_configs_for_remote_ue(REMOTE_L2_ID, LOCAL_ID, DEFAULT_SRAP_BEARER, SL_RNTI);

        assert_eq!(relay.remote_ues.len(), 1);
        assert_eq!(relay.remote_ues[0].remote_l2_id, REMOTE_L2_ID);
        assert_eq!(relay.remote_ues[0].srap.local_remote_ue_id, LOCAL_ID);
        assert_eq!(
            remote.srap.as_ref().expect("srap").local_remote_ue_id,
            LOCAL_ID,
            "the remote UE must be told the SAME local ID the relay expects"
        );
        // And each side's mapping names the hop it actually transmits on: the relay's
        // egress is Uu (towards the gNB), the remote UE's is PC5 (towards the relay).
        assert_eq!(
            relay.remote_ues[0].srap.egress_for(DEFAULT_SRAP_BEARER),
            Some(EgressChannel::Uu(FIRST_UU_RELAY_RLC_CHANNEL))
        );
        assert!(matches!(
            remote
                .srap
                .as_ref()
                .expect("srap")
                .egress_for(DEFAULT_SRAP_BEARER),
            Some(EgressChannel::Pc5(_))
        ));
        assert_eq!(remote.remote_ue_c_rnti, Some(SL_RNTI));
    }

    /// Two remote UEs get DIFFERENT local Remote UE IDs. TS 38.300 §16.12.2.1 makes
    /// collision avoidance the gNB's responsibility, and a collision would make the two
    /// indistinguishable in the SRAP header.
    #[test]
    fn two_remote_ues_get_different_local_ids() {
        let mut allocator = LocalRemoteUeIdAllocator::new();
        let first = allocator.assign(REMOTE_L2_ID).expect("assigned");
        let second = allocator.assign(OTHER_REMOTE_L2_ID).expect("assigned");
        assert_ne!(
            first, second,
            "two remote UEs sharing a local ID would be indistinguishable in the SRAP header"
        );
        assert_eq!(allocator.len(), 2);
    }

    /// Re-requesting for a remote UE already served returns the SAME local ID, rather than
    /// allocating a second one — otherwise a relay reconfigured twice would hold two
    /// mappings for one remote UE and the gNB would not know which header to expect.
    #[test]
    fn re_requesting_for_a_served_remote_ue_returns_the_same_local_id() {
        let mut allocator = LocalRemoteUeIdAllocator::new();
        let first = allocator.assign(REMOTE_L2_ID).expect("assigned");
        assert_eq!(allocator.assign(REMOTE_L2_ID), Some(first));
        assert_eq!(allocator.len(), 1, "no second ID may be allocated");
        assert_eq!(allocator.assigned_id(REMOTE_L2_ID), Some(first));
    }

    /// A released local ID becomes reusable, and the released remote UE no longer holds
    /// one.
    #[test]
    fn releasing_a_local_id_frees_it_for_reuse() {
        let mut allocator = LocalRemoteUeIdAllocator::new();
        let first = allocator.assign(REMOTE_L2_ID).expect("assigned");
        assert_eq!(allocator.release(REMOTE_L2_ID), Some(first));
        assert!(allocator.is_empty());
        assert_eq!(allocator.assigned_id(REMOTE_L2_ID), None);
        // The next remote UE may now take the freed value.
        assert_eq!(allocator.assign(OTHER_REMOTE_L2_ID), Some(first));
    }

    /// Past `maxNrofRemoteUE-r17` the allocator refuses rather than wrapping: a wrapped ID
    /// would be a collision, which is the one thing §16.12.2.1 makes the gNB prevent.
    #[test]
    fn the_allocator_refuses_past_the_asn1_maximum_rather_than_colliding() {
        let mut allocator = LocalRemoteUeIdAllocator::new();
        for i in 0..MAX_REMOTE_UES as u32 {
            assert!(
                allocator.assign(0x10_0000 + i).is_some(),
                "remote UE {i} is within maxNrofRemoteUE-r17 and must be assigned"
            );
        }
        assert_eq!(allocator.len(), MAX_REMOTE_UES);
        assert_eq!(
            allocator.assign(0xFF_FFFF),
            None,
            "the 33rd remote UE must be refused, not given a colliding ID"
        );
        // And an already-served remote UE is still served, even at the limit.
        assert!(allocator.assign(0x10_0000).is_some());
    }

    /// The SL-RNTI is never 0 (TS 38.321 reserves it) and differs between UEs, so a grant
    /// cannot be matched by the wrong UE.
    #[test]
    fn the_sl_rnti_is_never_zero_and_differs_per_ue() {
        assert_ne!(sl_rnti_for_ue(0), 0);
        assert_ne!(sl_rnti_for_ue(1), sl_rnti_for_ue(2));
        assert_eq!(
            sl_rnti_for_ue(1),
            sl_rnti_for_ue(1),
            "the same UE must always get the same SL-RNTI"
        );
    }
}
