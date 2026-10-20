//! #253: SEQUENCE extension addition groups in the generated E-UTRA RRC.
//!
//! Before the fix every `[[ ... ]]` group was dropped by the generator. The
//! expected bytes are derived by hand from X.691 (UNALIGNED) beside them.

use asn1_codecs::uper::UperCodec;
use asn1_codecs::PerCodecData;

use crate::codec::generated::{
    LogicalChannelConfig, LogicalChannelConfigLogicalChannelSR_Prohibit_r12,
    LogicalChannelConfig_ExtAdditions,
};

/// `LogicalChannelConfig` (TS 36.331 §6.3.2) with no ul-SpecificParameters and
/// `logicalChannelSR-Prohibit-r12 = TRUE`, the second of its six extension
/// addition groups:
///
/// ```text
/// 1          extension bit (X.691 §19.1)
/// 0          ul-SpecificParameters absent
/// 0 000101   normally small length: 6 extension additions (§19.7, n-1 = 5)
/// 010000     presence bitmap: only the r12 group (§19.8)
/// 00000001   open type length, 1 octet (§19.9)
/// 1 1 000000 the group: its one OPTIONAL present, BOOLEAN TRUE, padded
/// 0          final padding
/// ```
///
/// = `82 A0 03 80`.
const SR_PROHIBIT: [u8; 4] = [0x82, 0xA0, 0x03, 0x80];

fn sr_prohibit() -> LogicalChannelConfig {
    LogicalChannelConfig {
        ul_specific_parameters: None,
        extensions: LogicalChannelConfig_ExtAdditions {
            logical_channel_sr_prohibit_r12: Some(
                LogicalChannelConfigLogicalChannelSR_Prohibit_r12(true),
            ),
            ..Default::default()
        },
    }
}

#[test]
fn an_addition_group_encodes_to_the_x691_bytes() {
    let mut d = PerCodecData::new_uper();
    sr_prohibit().uper_encode(&mut d).expect("encode");
    assert_eq!(d.into_bytes(), SR_PROHIBIT);
}

#[test]
fn an_addition_group_decodes_from_the_x691_bytes() {
    let v = LogicalChannelConfig::uper_decode(&mut PerCodecData::from_slice_uper(&SR_PROHIBIT))
        .expect("decode");
    assert_eq!(v, sr_prohibit());
}
