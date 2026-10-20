//! E-UTRA logical channels on RLS.
//!
//! RLS replaces the physical layer and MAC, so it also takes over MAC's job of
//! telling logical channels apart -- the LCID of a MAC subheader (TS 36.321
//! §6.2.1, Table 6.2.1-1: 0 CCCH, 1..=2 SRB1/SRB2, 3..=10 the DRBs'
//! `logicalChannelIdentity`). The wire format does not change: the existing
//! 32-bit `payload` of [`crate::RlsPduTransmission`] already means "which
//! channel" (the RRC channel of an RRC PDU, the DRB of a data PDU on the NR
//! side), and E-UTRA uses it as follows:
//!
//! - **RRC PDU** (`PduType::Rrc`): bits 0..7 are the [`RrcChannel`] as before,
//!   bits 8..15 the LCID -- 1 or 2 on DCCH, 0 on BCCH, PCCH and CCCH (which
//!   have no SRB beyond SRB0). Bits 16..31 are 0.
//! - **Data PDU** (`PduType::Data`): the payload is the DRB's LCID (3..=10).
//!
//! The NR side never sets bits 8..31 of an RRC payload, so its meaning is
//! unchanged, and the RLS version stays as it is.

use crate::transport::RrcChannel;

/// LCID of SRB1 (TS 36.321 Table 6.2.1-1).
pub const LCID_SRB1: u8 = 1;
/// LCID of SRB2.
pub const LCID_SRB2: u8 = 2;
/// The DRBs' LCID range (TS 36.331 §6.3.2 `DRB-ToAddMod`).
pub const LCID_DRB_RANGE: std::ops::RangeInclusive<u8> = 3..=10;

/// The `payload` of an E-UTRA RRC PDU on `channel` and `lcid`.
pub fn rrc_payload(channel: RrcChannel, lcid: u8) -> u32 {
    channel as u32 | (u32::from(lcid) << 8)
}

/// The channel and LCID of an E-UTRA RRC PDU's `payload`, if well formed:
/// a known channel, no bits above 15, and an LCID of 1 or 2 on DCCH only.
pub fn split_rrc_payload(payload: u32) -> Option<(RrcChannel, u8)> {
    if payload >> 16 != 0 {
        return None;
    }
    let channel = RrcChannel::from_u32(payload & 0xFF)?;
    let lcid = (payload >> 8) as u8;
    let dcch = matches!(channel, RrcChannel::DlDcch | RrcChannel::UlDcch);
    match (dcch, lcid) {
        (true, LCID_SRB1 | LCID_SRB2) | (false, 0) => Some((channel, lcid)),
        _ => None,
    }
}

/// The LCID of an E-UTRA data PDU's `payload`, if it names a DRB.
pub fn data_lcid(payload: u32) -> Option<u8> {
    u8::try_from(payload)
        .ok()
        .filter(|l| LCID_DRB_RANGE.contains(l))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payloads_by_hand() {
        // UL-DCCH is 7: SRB2 → 0x0207, SRB1 → 0x0107; UL-CCCH (5) stays 5.
        assert_eq!(rrc_payload(RrcChannel::UlDcch, LCID_SRB2), 0x0207);
        assert_eq!(rrc_payload(RrcChannel::UlDcch, LCID_SRB1), 0x0107);
        assert_eq!(rrc_payload(RrcChannel::UlCcch, 0), 5);
        assert_eq!(split_rrc_payload(0x0207), Some((RrcChannel::UlDcch, 2)));
        assert_eq!(split_rrc_payload(5), Some((RrcChannel::UlCcch, 0)));
    }

    #[test]
    fn malformed_payloads_are_refused() {
        // DCCH without an SRB, CCCH with one, an SRB3, high bits.
        assert_eq!(split_rrc_payload(RrcChannel::DlDcch as u32), None);
        assert_eq!(split_rrc_payload(0x0102), None);
        assert_eq!(split_rrc_payload(0x0303), None);
        assert_eq!(split_rrc_payload(0x1_0107), None);
        assert_eq!(data_lcid(2), None);
        assert_eq!(data_lcid(3), Some(3));
        assert_eq!(data_lcid(11), None);
    }
}
