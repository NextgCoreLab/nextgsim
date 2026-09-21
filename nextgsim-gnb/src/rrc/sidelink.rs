//! The gNB's answer to a UE's sidelink resource request
//! (TS 38.331 §5.8.3, §5.3.5.3; issue #141).
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
//! # What "granting" means here, and what it deliberately does not
//!
//! The grant is an `sl-ConfigDedicatedNR-r16` `setup` carrying an
//! `sl-PHY-MAC-RLC-Config` and a `t400`. Its presence is the substance: it tells the UE
//! the network permits sidelink on the requested carriers, which is the
//! Mode-2 (UE-autonomous resource selection) case of TS 38.300 §16.9.
//!
//! It does **not** carry a Mode-1 scheduled grant — `sl-ScheduledConfig-r16` with
//! concrete resource pools. There is no PC5 physical layer in this simulator for a pool
//! to mean anything to: the sidelink transmissions modelled here go over in-process
//! channels, not over a numbered slot in a configured pool. Signalling pools no
//! transmission consults would be an IE nothing reads, which is the defect issue #141
//! exists to remove. Mode 2 is the honest description of what actually happens.
//!
//! # Why the gNB checks the request rather than granting unconditionally
//!
//! A UE may ask for carriers the cell does not operate. Granting those would tell the UE
//! it may transmit on a frequency the network has no allocation for, and the UE would
//! believe it. [`sl_config_for_request`] grants only when the request is one this cell
//! can honour, and says why when it declines.

use nextgsim_rrc::procedures::sidelink_ue_information::{
    SidelinkUeInformationParams, SlConfigDedicatedParams,
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
    })
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

#[cfg(test)]
mod tests {
    use super::*;
    use nextgsim_rrc::procedures::sidelink_ue_information::{SlCastType, SlTxResourceRequest};

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
}
