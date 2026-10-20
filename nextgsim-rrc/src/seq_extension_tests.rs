//! #253: SEQUENCE extension addition groups in the generated NR RRC.
//!
//! Before the fix the generator dropped every `[[ ... ]]` group, so a Rel-16+
//! field such as `servedRadioBearerSRB4-r17` was unrepresentable and a received
//! one decoded silently to nothing. Expected bytes are derived by hand from
//! X.691 (UNALIGNED) beside them.

use asn1_codecs::uper::UperCodec;
use asn1_codecs::PerCodecData;

use crate::codec::generated::{
    LogicalChannelIdentity, RLC_BearerConfig, RLC_BearerConfig_ExtAdditions, SRB_Identity_v1700,
};

/// `RLC-BearerConfig` (TS 38.331 §6.3.2) with logicalChannelIdentity 4 and,
/// from its second extension addition group, `servedRadioBearerSRB4-r17 = 4`:
///
/// ```text
/// 1          extension bit: additions present (X.691 §19.1)
/// 0000       servedRadioBearer, reestablishRLC, rlc-Config, mac-... absent
/// 00011      logicalChannelIdentity 4, INTEGER (1..32): offset 3 in 5 bits
/// 0 000011   normally small length: 4 extension additions (§19.7, n-1 = 3)
/// 0100       presence bitmap: only the second group (§19.8)
/// 00000001   open type length, 1 octet (§19.9, §11.2)
/// 0001 0000  the group as a SEQUENCE of four OPTIONALs: only srb4 present;
///            SRB-Identity-v1700 ::= INTEGER (4) has one value, so 0 bits;
///            padded to an octet (§11.2.1)
/// 000        final padding
/// ```
///
/// 37 bits: `10000000 11000001 10100|000 00001|000 10000|000`
/// = `80 C1 A0 08 80`.
const SRB4_BEARER: [u8; 5] = [0x80, 0xC1, 0xA0, 0x08, 0x80];

fn srb4_bearer() -> RLC_BearerConfig {
    RLC_BearerConfig {
        logical_channel_identity: LogicalChannelIdentity(4),
        served_radio_bearer: None,
        reestablish_rlc: None,
        rlc_config: None,
        mac_logical_channel_config: None,
        extensions: RLC_BearerConfig_ExtAdditions {
            served_radio_bearer_srb4_r17: Some(SRB_Identity_v1700(4)),
            ..Default::default()
        },
    }
}

#[test]
fn an_addition_group_member_encodes_to_the_x691_bytes() {
    let mut d = PerCodecData::new_uper();
    srb4_bearer().uper_encode(&mut d).expect("encode");
    assert_eq!(d.into_bytes(), SRB4_BEARER);
}

#[test]
fn an_addition_group_member_decodes_from_the_x691_bytes() {
    let v = RLC_BearerConfig::uper_decode(&mut PerCodecData::from_slice_uper(&SRB4_BEARER))
        .expect("decode");
    assert_eq!(v, srb4_bearer());
    assert_eq!(
        v.extensions.served_radio_bearer_srb4_r17,
        Some(SRB_Identity_v1700(4)),
        "the Rel-17 field survives the decode"
    );
}
