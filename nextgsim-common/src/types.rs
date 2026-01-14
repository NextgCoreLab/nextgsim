//! Core 5G types: PLMN, TAI, S-NSSAI, SUPI, GUTI, etc.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Public Land Mobile Network identifier.
///
/// A PLMN uniquely identifies a mobile network and consists of:
/// - MCC (Mobile Country Code): 3 decimal digits (001-999)
/// - MNC (Mobile Network Code): 2 or 3 decimal digits
///
/// The `long_mnc` field indicates whether the MNC uses 3 digits (true) or 2 digits (false).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Plmn {
    /// Mobile Country Code (3 digits, range 0-999)
    pub mcc: u16,
    /// Mobile Network Code (2-3 digits, range 0-999)
    pub mnc: u16,
    /// True if MNC is 3 digits, false if 2 digits
    pub long_mnc: bool,
}

impl Plmn {
    /// Creates a new PLMN with the given MCC and MNC.
    ///
    /// # Arguments
    /// * `mcc` - Mobile Country Code (3 digits)
    /// * `mnc` - Mobile Network Code (2-3 digits)
    /// * `long_mnc` - Whether MNC is 3 digits
    pub const fn new(mcc: u16, mnc: u16, long_mnc: bool) -> Self {
        Self { mcc, mnc, long_mnc }
    }

    /// Returns true if this PLMN has valid values set.
    pub fn has_value(&self) -> bool {
        self.mcc > 0 || self.mnc > 0
    }

    /// Encodes the PLMN to 3GPP format (3 bytes).
    ///
    /// The encoding follows 3GPP TS 24.008 format:
    /// - Byte 0: MCC digit 2 (high nibble) | MCC digit 1 (low nibble)
    /// - Byte 1: MNC digit 3 or 0xF (high nibble) | MCC digit 3 (low nibble)
    /// - Byte 2: MNC digit 2 (high nibble) | MNC digit 1 (low nibble)
    pub fn encode(&self) -> [u8; 3] {
        let mcc = self.mcc;
        let mcc3 = (mcc % 10) as u8;
        let mcc2 = ((mcc % 100) / 10) as u8;
        let mcc1 = ((mcc % 1000) / 100) as u8;

        let mnc = self.mnc;
        let (mnc1, mnc2, mnc3) = if self.long_mnc {
            (
                ((mnc % 1000) / 100) as u8,
                ((mnc % 100) / 10) as u8,
                (mnc % 10) as u8,
            )
        } else {
            (((mnc % 100) / 10) as u8, (mnc % 10) as u8, 0x0F)
        };

        let octet1 = (mcc2 << 4) | mcc1;
        let octet2 = (mnc3 << 4) | mcc3;
        let octet3 = (mnc2 << 4) | mnc1;

        [octet1, octet2, octet3]
    }

    /// Decodes a PLMN from 3GPP format (3 bytes).
    ///
    /// # Arguments
    /// * `bytes` - 3-byte array in 3GPP PLMN encoding format
    ///
    /// # Returns
    /// The decoded PLMN
    pub fn decode(bytes: [u8; 3]) -> Self {
        let octet1 = bytes[0];
        let octet2 = bytes[1];
        let octet3 = bytes[2];

        // Decode MCC
        let mcc1 = (octet1 & 0x0F) as u16;
        let mcc2 = ((octet1 >> 4) & 0x0F) as u16;
        let mcc3 = (octet2 & 0x0F) as u16;
        let mcc = 100 * mcc1 + 10 * mcc2 + mcc3;

        // Decode MNC
        let mnc3 = (octet2 >> 4) & 0x0F;
        let mnc1 = (octet3 & 0x0F) as u16;
        let mnc2 = ((octet3 >> 4) & 0x0F) as u16;

        let (mnc, long_mnc) =
            if mnc3 != 0x0F || (octet1 == 0xFF && octet2 == 0xFF && octet3 == 0xFF) {
                // 3-digit MNC
                (10 * (10 * mnc1 + mnc2) + mnc3 as u16, true)
            } else {
                // 2-digit MNC
                (10 * mnc1 + mnc2, false)
            };

        Self { mcc, mnc, long_mnc }
    }
}

impl fmt::Debug for Plmn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.long_mnc {
            write!(f, "Plmn({:03}-{:03})", self.mcc, self.mnc)
        } else {
            write!(f, "Plmn({:03}-{:02})", self.mcc, self.mnc)
        }
    }
}

impl fmt::Display for Plmn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.long_mnc {
            write!(f, "{:03}{:03}", self.mcc, self.mnc)
        } else {
            write!(f, "{:03}{:02}", self.mcc, self.mnc)
        }
    }
}

impl Default for Plmn {
    fn default() -> Self {
        Self {
            mcc: 0,
            mnc: 0,
            long_mnc: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plmn_new() {
        let plmn = Plmn::new(310, 410, false);
        assert_eq!(plmn.mcc, 310);
        assert_eq!(plmn.mnc, 410);
        assert!(!plmn.long_mnc);
    }

    #[test]
    fn test_plmn_encode_2digit_mnc() {
        // MCC=310, MNC=41 (2-digit)
        let plmn = Plmn::new(310, 41, false);
        let encoded = plmn.encode();
        // MCC: 3-1-0 -> mcc1=3, mcc2=1, mcc3=0
        // MNC: 4-1 -> mnc1=4, mnc2=1, mnc3=0xF
        // octet1 = mcc2<<4 | mcc1 = 0x13
        // octet2 = mnc3<<4 | mcc3 = 0xF0
        // octet3 = mnc2<<4 | mnc1 = 0x14
        assert_eq!(encoded, [0x13, 0xF0, 0x14]);
    }

    #[test]
    fn test_plmn_encode_3digit_mnc() {
        // MCC=310, MNC=410 (3-digit)
        let plmn = Plmn::new(310, 410, true);
        let encoded = plmn.encode();
        // MCC: 3-1-0 -> mcc1=3, mcc2=1, mcc3=0
        // MNC: 4-1-0 -> mnc1=4, mnc2=1, mnc3=0
        // octet1 = mcc2<<4 | mcc1 = 0x13
        // octet2 = mnc3<<4 | mcc3 = 0x00
        // octet3 = mnc2<<4 | mnc1 = 0x14
        assert_eq!(encoded, [0x13, 0x00, 0x14]);
    }

    #[test]
    fn test_plmn_decode_2digit_mnc() {
        let bytes = [0x13, 0xF0, 0x14];
        let plmn = Plmn::decode(bytes);
        assert_eq!(plmn.mcc, 310);
        assert_eq!(plmn.mnc, 41);
        assert!(!plmn.long_mnc);
    }

    #[test]
    fn test_plmn_decode_3digit_mnc() {
        let bytes = [0x13, 0x00, 0x14];
        let plmn = Plmn::decode(bytes);
        assert_eq!(plmn.mcc, 310);
        assert_eq!(plmn.mnc, 410);
        assert!(plmn.long_mnc);
    }

    #[test]
    fn test_plmn_roundtrip_2digit() {
        let original = Plmn::new(234, 15, false);
        let encoded = original.encode();
        let decoded = Plmn::decode(encoded);
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_plmn_roundtrip_3digit() {
        let original = Plmn::new(234, 150, true);
        let encoded = original.encode();
        let decoded = Plmn::decode(encoded);
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_plmn_display_2digit() {
        let plmn = Plmn::new(310, 41, false);
        assert_eq!(format!("{}", plmn), "31041");
    }

    #[test]
    fn test_plmn_display_3digit() {
        let plmn = Plmn::new(310, 410, true);
        assert_eq!(format!("{}", plmn), "310410");
    }

    #[test]
    fn test_plmn_debug_2digit() {
        let plmn = Plmn::new(310, 41, false);
        assert_eq!(format!("{:?}", plmn), "Plmn(310-41)");
    }

    #[test]
    fn test_plmn_debug_3digit() {
        let plmn = Plmn::new(310, 410, true);
        assert_eq!(format!("{:?}", plmn), "Plmn(310-410)");
    }

    #[test]
    fn test_plmn_has_value() {
        let empty = Plmn::default();
        assert!(!empty.has_value());

        let with_mcc = Plmn::new(310, 0, false);
        assert!(with_mcc.has_value());

        let with_mnc = Plmn::new(0, 41, false);
        assert!(with_mnc.has_value());
    }

    #[test]
    fn test_plmn_equality() {
        let plmn1 = Plmn::new(310, 410, true);
        let plmn2 = Plmn::new(310, 410, true);
        let plmn3 = Plmn::new(310, 410, false);
        assert_eq!(plmn1, plmn2);
        assert_ne!(plmn1, plmn3);
    }

    // TAI tests

    #[test]
    fn test_tai_new() {
        let plmn = Plmn::new(310, 410, true);
        let tai = Tai::new(plmn, 0x123456);
        assert_eq!(tai.plmn, plmn);
        assert_eq!(tai.tac, 0x123456);
    }

    #[test]
    fn test_tai_from_parts() {
        let tai = Tai::from_parts(310, 410, true, 0x123456);
        assert_eq!(tai.plmn.mcc, 310);
        assert_eq!(tai.plmn.mnc, 410);
        assert!(tai.plmn.long_mnc);
        assert_eq!(tai.tac, 0x123456);
    }

    #[test]
    fn test_tai_has_value() {
        let empty = Tai::default();
        assert!(!empty.has_value());

        let with_plmn = Tai::new(Plmn::new(310, 0, false), 0);
        assert!(with_plmn.has_value());

        let with_tac = Tai::new(Plmn::default(), 1);
        assert!(with_tac.has_value());

        let full = Tai::from_parts(310, 410, true, 0x123456);
        assert!(full.has_value());
    }

    #[test]
    fn test_tai_encode() {
        // MCC=310, MNC=410 (3-digit), TAC=0x123456
        let tai = Tai::from_parts(310, 410, true, 0x123456);
        let encoded = tai.encode();
        // PLMN encoding: [0x13, 0x00, 0x14]
        // TAC encoding: [0x12, 0x34, 0x56]
        assert_eq!(encoded, [0x13, 0x00, 0x14, 0x12, 0x34, 0x56]);
    }

    #[test]
    fn test_tai_encode_2digit_mnc() {
        // MCC=310, MNC=41 (2-digit), TAC=0x000001
        let tai = Tai::from_parts(310, 41, false, 1);
        let encoded = tai.encode();
        // PLMN encoding: [0x13, 0xF0, 0x14]
        // TAC encoding: [0x00, 0x00, 0x01]
        assert_eq!(encoded, [0x13, 0xF0, 0x14, 0x00, 0x00, 0x01]);
    }

    #[test]
    fn test_tai_decode() {
        let bytes = [0x13, 0x00, 0x14, 0x12, 0x34, 0x56];
        let tai = Tai::decode(bytes);
        assert_eq!(tai.plmn.mcc, 310);
        assert_eq!(tai.plmn.mnc, 410);
        assert!(tai.plmn.long_mnc);
        assert_eq!(tai.tac, 0x123456);
    }

    #[test]
    fn test_tai_decode_2digit_mnc() {
        let bytes = [0x13, 0xF0, 0x14, 0x00, 0x00, 0x01];
        let tai = Tai::decode(bytes);
        assert_eq!(tai.plmn.mcc, 310);
        assert_eq!(tai.plmn.mnc, 41);
        assert!(!tai.plmn.long_mnc);
        assert_eq!(tai.tac, 1);
    }

    #[test]
    fn test_tai_roundtrip() {
        let original = Tai::from_parts(234, 150, true, 0xABCDEF);
        let encoded = original.encode();
        let decoded = Tai::decode(encoded);
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_tai_roundtrip_2digit() {
        let original = Tai::from_parts(234, 15, false, 0x000FFF);
        let encoded = original.encode();
        let decoded = Tai::decode(encoded);
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_tai_display() {
        let tai = Tai::from_parts(310, 410, true, 0x123456);
        assert_eq!(format!("{}", tai), "310410-1193046");
    }

    #[test]
    fn test_tai_debug() {
        let tai = Tai::from_parts(310, 410, true, 0x123456);
        assert_eq!(format!("{:?}", tai), "Tai(Plmn(310-410), tac=1193046)");
    }

    #[test]
    fn test_tai_equality() {
        let tai1 = Tai::from_parts(310, 410, true, 0x123456);
        let tai2 = Tai::from_parts(310, 410, true, 0x123456);
        let tai3 = Tai::from_parts(310, 410, true, 0x123457);
        let tai4 = Tai::from_parts(310, 410, false, 0x123456);
        assert_eq!(tai1, tai2);
        assert_ne!(tai1, tai3);
        assert_ne!(tai1, tai4);
    }

    #[test]
    fn test_tai_max_tac() {
        // TAC is 24-bit, max value is 0xFFFFFF (16777215)
        let tai = Tai::from_parts(999, 999, true, 0xFFFFFF);
        let encoded = tai.encode();
        let decoded = Tai::decode(encoded);
        assert_eq!(decoded.tac, 0xFFFFFF);
    }

    #[test]
    fn test_tai_default() {
        let tai = Tai::default();
        assert_eq!(tai.plmn.mcc, 0);
        assert_eq!(tai.plmn.mnc, 0);
        assert!(!tai.plmn.long_mnc);
        assert_eq!(tai.tac, 0);
    }

    // SNssai tests

    #[test]
    fn test_snssai_new() {
        let snssai = SNssai::new(1);
        assert_eq!(snssai.sst, 1);
        assert!(snssai.sd.is_none());
    }

    #[test]
    fn test_snssai_with_sd() {
        let snssai = SNssai::with_sd(1, [0x01, 0x02, 0x03]);
        assert_eq!(snssai.sst, 1);
        assert_eq!(snssai.sd, Some([0x01, 0x02, 0x03]));
    }

    #[test]
    fn test_snssai_with_sd_u32() {
        let snssai = SNssai::with_sd_u32(1, 0x010203);
        assert_eq!(snssai.sst, 1);
        assert_eq!(snssai.sd, Some([0x01, 0x02, 0x03]));
    }

    #[test]
    fn test_snssai_sd_as_u32() {
        let snssai = SNssai::with_sd(1, [0x01, 0x02, 0x03]);
        assert_eq!(snssai.sd_as_u32(), Some(0x010203));

        let snssai_no_sd = SNssai::new(1);
        assert_eq!(snssai_no_sd.sd_as_u32(), None);
    }

    #[test]
    fn test_snssai_has_value() {
        let empty = SNssai::default();
        assert!(!empty.has_value());

        let with_sst = SNssai::new(1);
        assert!(with_sst.has_value());

        let with_sd = SNssai::with_sd(0, [0x00, 0x00, 0x01]);
        assert!(with_sd.has_value());
    }

    #[test]
    fn test_snssai_encode_without_sd() {
        let snssai = SNssai::new(1);
        let encoded = snssai.encode();
        assert_eq!(encoded, vec![1]);
    }

    #[test]
    fn test_snssai_encode_with_sd() {
        let snssai = SNssai::with_sd(1, [0x01, 0x02, 0x03]);
        let encoded = snssai.encode();
        assert_eq!(encoded, vec![1, 0x01, 0x02, 0x03]);
    }

    #[test]
    fn test_snssai_decode_without_sd() {
        let decoded = SNssai::decode(&[1]).unwrap();
        assert_eq!(decoded.sst, 1);
        assert!(decoded.sd.is_none());
    }

    #[test]
    fn test_snssai_decode_with_sd() {
        let decoded = SNssai::decode(&[1, 0x01, 0x02, 0x03]).unwrap();
        assert_eq!(decoded.sst, 1);
        assert_eq!(decoded.sd, Some([0x01, 0x02, 0x03]));
    }

    #[test]
    fn test_snssai_decode_invalid() {
        assert!(SNssai::decode(&[]).is_none());
        assert!(SNssai::decode(&[1, 2]).is_none());
        assert!(SNssai::decode(&[1, 2, 3]).is_none());
        assert!(SNssai::decode(&[1, 2, 3, 4, 5]).is_none());
    }

    #[test]
    fn test_snssai_roundtrip_without_sd() {
        let original = SNssai::new(1);
        let encoded = original.encode();
        let decoded = SNssai::decode(&encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_snssai_roundtrip_with_sd() {
        let original = SNssai::with_sd(1, [0xAB, 0xCD, 0xEF]);
        let encoded = original.encode();
        let decoded = SNssai::decode(&encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_snssai_display_without_sd() {
        let snssai = SNssai::new(1);
        assert_eq!(format!("{}", snssai), "1");
    }

    #[test]
    fn test_snssai_display_with_sd() {
        let snssai = SNssai::with_sd(1, [0x01, 0x02, 0x03]);
        assert_eq!(format!("{}", snssai), "1-010203");
    }

    #[test]
    fn test_snssai_debug_without_sd() {
        let snssai = SNssai::new(1);
        assert_eq!(format!("{:?}", snssai), "SNssai(sst=1)");
    }

    #[test]
    fn test_snssai_debug_with_sd() {
        let snssai = SNssai::with_sd(1, [0x01, 0x02, 0x03]);
        assert_eq!(format!("{:?}", snssai), "SNssai(sst=1, sd=010203)");
    }

    #[test]
    fn test_snssai_equality() {
        let s1 = SNssai::new(1);
        let s2 = SNssai::new(1);
        let s3 = SNssai::new(2);
        let s4 = SNssai::with_sd(1, [0x01, 0x02, 0x03]);
        assert_eq!(s1, s2);
        assert_ne!(s1, s3);
        assert_ne!(s1, s4);
    }

    #[test]
    fn test_snssai_default() {
        let snssai = SNssai::default();
        assert_eq!(snssai.sst, 0);
        assert!(snssai.sd.is_none());
    }

    // NetworkSlice tests

    #[test]
    fn test_network_slice_new() {
        let ns = NetworkSlice::new();
        assert!(ns.is_empty());
        assert_eq!(ns.len(), 0);
    }

    #[test]
    fn test_network_slice_from_slices() {
        let slices = vec![SNssai::new(1), SNssai::new(2)];
        let ns = NetworkSlice::from_slices(slices);
        assert_eq!(ns.len(), 2);
    }

    #[test]
    fn test_network_slice_add_if_not_exists() {
        let mut ns = NetworkSlice::new();
        assert!(ns.add_if_not_exists(SNssai::new(1)));
        assert_eq!(ns.len(), 1);

        // Adding same slice should return false
        assert!(!ns.add_if_not_exists(SNssai::new(1)));
        assert_eq!(ns.len(), 1);

        // Adding different slice should return true
        assert!(ns.add_if_not_exists(SNssai::new(2)));
        assert_eq!(ns.len(), 2);
    }

    #[test]
    fn test_network_slice_contains() {
        let mut ns = NetworkSlice::new();
        ns.add_if_not_exists(SNssai::new(1));

        assert!(ns.contains(&SNssai::new(1)));
        assert!(!ns.contains(&SNssai::new(2)));
    }

    #[test]
    fn test_network_slice_iter() {
        let slices = vec![SNssai::new(1), SNssai::new(2)];
        let ns = NetworkSlice::from_slices(slices.clone());

        let collected: Vec<_> = ns.iter().cloned().collect();
        assert_eq!(collected, slices);
    }

    #[test]
    fn test_network_slice_default() {
        let ns = NetworkSlice::default();
        assert!(ns.is_empty());
    }

    #[test]
    fn test_network_slice_equality() {
        let ns1 = NetworkSlice::from_slices(vec![SNssai::new(1), SNssai::new(2)]);
        let ns2 = NetworkSlice::from_slices(vec![SNssai::new(1), SNssai::new(2)]);
        let ns3 = NetworkSlice::from_slices(vec![SNssai::new(1)]);
        assert_eq!(ns1, ns2);
        assert_ne!(ns1, ns3);
    }

    // Supi tests

    #[test]
    fn test_supi_new() {
        let supi = Supi::new(SupiType::Imsi, "310410123456789");
        assert_eq!(supi.supi_type, SupiType::Imsi);
        assert_eq!(supi.value, "310410123456789");
    }

    #[test]
    fn test_supi_imsi() {
        let supi = Supi::imsi("310410123456789");
        assert_eq!(supi.supi_type, SupiType::Imsi);
        assert_eq!(supi.value, "310410123456789");
    }

    #[test]
    fn test_supi_nai() {
        let supi = Supi::nai("user@example.com");
        assert_eq!(supi.supi_type, SupiType::Nai);
        assert_eq!(supi.value, "user@example.com");
    }

    #[test]
    fn test_supi_parse_imsi() {
        let supi = Supi::parse("imsi-310410123456789").unwrap();
        assert_eq!(supi.supi_type, SupiType::Imsi);
        assert_eq!(supi.value, "310410123456789");
    }

    #[test]
    fn test_supi_parse_nai() {
        let supi = Supi::parse("nai-user@example.com").unwrap();
        assert_eq!(supi.supi_type, SupiType::Nai);
        assert_eq!(supi.value, "user@example.com");
    }

    #[test]
    fn test_supi_parse_case_insensitive() {
        let supi = Supi::parse("IMSI-310410123456789").unwrap();
        assert_eq!(supi.supi_type, SupiType::Imsi);
    }

    #[test]
    fn test_supi_parse_invalid() {
        assert!(Supi::parse("invalid").is_none());
        assert!(Supi::parse("unknown-123").is_none());
        assert!(Supi::parse("").is_none());
    }

    #[test]
    fn test_supi_has_value() {
        let supi = Supi::imsi("310410123456789");
        assert!(supi.has_value());

        let empty = Supi::imsi("");
        assert!(!empty.has_value());
    }

    #[test]
    fn test_supi_display() {
        let supi = Supi::imsi("310410123456789");
        assert_eq!(format!("{}", supi), "imsi-310410123456789");

        let nai = Supi::nai("user@example.com");
        assert_eq!(format!("{}", nai), "nai-user@example.com");
    }

    #[test]
    fn test_supi_debug() {
        let supi = Supi::imsi("310410123456789");
        assert_eq!(format!("{:?}", supi), "Supi(imsi-310410123456789)");
    }

    #[test]
    fn test_supi_equality() {
        let s1 = Supi::imsi("310410123456789");
        let s2 = Supi::imsi("310410123456789");
        let s3 = Supi::imsi("310410123456780");
        let s4 = Supi::nai("310410123456789");
        assert_eq!(s1, s2);
        assert_ne!(s1, s3);
        assert_ne!(s1, s4);
    }

    #[test]
    fn test_supi_type_prefix() {
        assert_eq!(SupiType::Imsi.prefix(), "imsi");
        assert_eq!(SupiType::Nai.prefix(), "nai");
    }

    // Guti tests

    #[test]
    fn test_guti_new() {
        let plmn = Plmn::new(310, 410, true);
        let guti = Guti::new(plmn, 0x12, 0x123, 0x15, 0xABCDEF01);
        assert_eq!(guti.plmn, plmn);
        assert_eq!(guti.amf_region_id, 0x12);
        assert_eq!(guti.amf_set_id, 0x123);
        assert_eq!(guti.amf_pointer, 0x15);
        assert_eq!(guti.tmsi, 0xABCDEF01);
    }

    #[test]
    fn test_guti_from_parts() {
        let guti = Guti::from_parts(310, 410, true, 0x12, 0x123, 0x15, 0xABCDEF01);
        assert_eq!(guti.plmn.mcc, 310);
        assert_eq!(guti.plmn.mnc, 410);
        assert!(guti.plmn.long_mnc);
        assert_eq!(guti.amf_region_id, 0x12);
        assert_eq!(guti.amf_set_id, 0x123);
        assert_eq!(guti.amf_pointer, 0x15);
        assert_eq!(guti.tmsi, 0xABCDEF01);
    }

    #[test]
    fn test_guti_amf_set_id_masking() {
        // AMF Set ID is 10-bit, should be masked
        let guti = Guti::from_parts(310, 410, true, 0x12, 0xFFF, 0x15, 0xABCDEF01);
        assert_eq!(guti.amf_set_id, 0x3FF); // Masked to 10 bits
    }

    #[test]
    fn test_guti_amf_pointer_masking() {
        // AMF Pointer is 6-bit, should be masked
        let guti = Guti::from_parts(310, 410, true, 0x12, 0x123, 0xFF, 0xABCDEF01);
        assert_eq!(guti.amf_pointer, 0x3F); // Masked to 6 bits
    }

    #[test]
    fn test_guti_has_value() {
        let empty = Guti::default();
        assert!(!empty.has_value());

        let with_plmn = Guti::from_parts(310, 0, false, 0, 0, 0, 0);
        assert!(with_plmn.has_value());

        let with_region = Guti::from_parts(0, 0, false, 1, 0, 0, 0);
        assert!(with_region.has_value());

        let with_tmsi = Guti::from_parts(0, 0, false, 0, 0, 0, 1);
        assert!(with_tmsi.has_value());
    }

    #[test]
    fn test_guti_amf_id() {
        let guti = Guti::from_parts(310, 410, true, 0x12, 0x123, 0x15, 0xABCDEF01);
        // AMF ID = region(8) | set(10) | pointer(6) = 0x12 << 16 | 0x123 << 6 | 0x15
        let expected = (0x12 << 16) | (0x123 << 6) | 0x15;
        assert_eq!(guti.amf_id(), expected);
    }

    #[test]
    fn test_guti_from_amf_id() {
        let plmn = Plmn::new(310, 410, true);
        let amf_id = (0x12 << 16) | (0x123 << 6) | 0x15;
        let guti = Guti::from_amf_id(plmn, amf_id, 0xABCDEF01);
        assert_eq!(guti.amf_region_id, 0x12);
        assert_eq!(guti.amf_set_id, 0x123);
        assert_eq!(guti.amf_pointer, 0x15);
        assert_eq!(guti.tmsi, 0xABCDEF01);
    }

    #[test]
    fn test_guti_s_tmsi() {
        let guti = Guti::from_parts(310, 410, true, 0x12, 0x123, 0x15, 0xABCDEF01);
        // S-TMSI = set(10) | pointer(6) | tmsi(32)
        let amf_set_pointer = ((0x123 as u64) << 6) | 0x15;
        let expected = (amf_set_pointer << 32) | 0xABCDEF01;
        assert_eq!(guti.s_tmsi(), expected);
    }

    #[test]
    fn test_guti_from_s_tmsi() {
        let amf_set_pointer = ((0x123 as u64) << 6) | 0x15;
        let s_tmsi = (amf_set_pointer << 32) | 0xABCDEF01;
        let guti = Guti::from_s_tmsi(s_tmsi);
        assert_eq!(guti.amf_set_id, 0x123);
        assert_eq!(guti.amf_pointer, 0x15);
        assert_eq!(guti.tmsi, 0xABCDEF01);
        // PLMN and region should be default
        assert_eq!(guti.plmn, Plmn::default());
        assert_eq!(guti.amf_region_id, 0);
    }

    #[test]
    fn test_guti_encode() {
        let guti = Guti::from_parts(310, 410, true, 0x12, 0x123, 0x15, 0xABCDEF01);
        let encoded = guti.encode();
        // PLMN: [0x13, 0x00, 0x14]
        // AMF Region ID: 0x12
        // AMF Set ID (10) | AMF Pointer (6): (0x123 << 6) | 0x15 = 0x48D5 -> [0x48, 0xD5]
        // TMSI: [0xAB, 0xCD, 0xEF, 0x01]
        assert_eq!(encoded[0..3], [0x13, 0x00, 0x14]); // PLMN
        assert_eq!(encoded[3], 0x12); // AMF Region ID
        let amf_set_pointer = ((0x123 as u16) << 6) | 0x15;
        assert_eq!(encoded[4], (amf_set_pointer >> 8) as u8);
        assert_eq!(encoded[5], (amf_set_pointer & 0xFF) as u8);
        assert_eq!(encoded[6..10], [0xAB, 0xCD, 0xEF, 0x01]); // TMSI
    }

    #[test]
    fn test_guti_decode() {
        let plmn_bytes = Plmn::new(310, 410, true).encode();
        let amf_set_pointer: u16 = (0x123 << 6) | 0x15;
        let bytes = [
            plmn_bytes[0],
            plmn_bytes[1],
            plmn_bytes[2],
            0x12,
            (amf_set_pointer >> 8) as u8,
            (amf_set_pointer & 0xFF) as u8,
            0xAB,
            0xCD,
            0xEF,
            0x01,
        ];
        let guti = Guti::decode(bytes);
        assert_eq!(guti.plmn.mcc, 310);
        assert_eq!(guti.plmn.mnc, 410);
        assert!(guti.plmn.long_mnc);
        assert_eq!(guti.amf_region_id, 0x12);
        assert_eq!(guti.amf_set_id, 0x123);
        assert_eq!(guti.amf_pointer, 0x15);
        assert_eq!(guti.tmsi, 0xABCDEF01);
    }

    #[test]
    fn test_guti_roundtrip() {
        let original = Guti::from_parts(234, 150, true, 0xAB, 0x2AA, 0x2A, 0x12345678);
        let encoded = original.encode();
        let decoded = Guti::decode(encoded);
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_guti_roundtrip_2digit_mnc() {
        let original = Guti::from_parts(234, 15, false, 0x01, 0x001, 0x01, 0x00000001);
        let encoded = original.encode();
        let decoded = Guti::decode(encoded);
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_guti_display() {
        let guti = Guti::from_parts(310, 410, true, 0x12, 0x123, 0x15, 0xABCDEF01);
        // Format: PLMN-RegionID-SetID-Pointer-TMSI
        assert_eq!(format!("{}", guti), "310410-12-123-15-ABCDEF01");
    }

    #[test]
    fn test_guti_debug() {
        let guti = Guti::from_parts(310, 410, true, 0x12, 0x123, 0x15, 0xABCDEF01);
        let debug_str = format!("{:?}", guti);
        assert!(debug_str.contains("Guti"));
        assert!(debug_str.contains("310"));
        assert!(debug_str.contains("410"));
    }

    #[test]
    fn test_guti_equality() {
        let g1 = Guti::from_parts(310, 410, true, 0x12, 0x123, 0x15, 0xABCDEF01);
        let g2 = Guti::from_parts(310, 410, true, 0x12, 0x123, 0x15, 0xABCDEF01);
        let g3 = Guti::from_parts(310, 410, true, 0x12, 0x123, 0x15, 0xABCDEF02);
        let g4 = Guti::from_parts(310, 410, false, 0x12, 0x123, 0x15, 0xABCDEF01);
        assert_eq!(g1, g2);
        assert_ne!(g1, g3);
        assert_ne!(g1, g4);
    }

    #[test]
    fn test_guti_default() {
        let guti = Guti::default();
        assert_eq!(guti.plmn, Plmn::default());
        assert_eq!(guti.amf_region_id, 0);
        assert_eq!(guti.amf_set_id, 0);
        assert_eq!(guti.amf_pointer, 0);
        assert_eq!(guti.tmsi, 0);
    }

    #[test]
    fn test_guti_max_values() {
        let guti = Guti::from_parts(999, 999, true, 0xFF, 0x3FF, 0x3F, 0xFFFFFFFF);
        let encoded = guti.encode();
        let decoded = Guti::decode(encoded);
        assert_eq!(decoded.amf_region_id, 0xFF);
        assert_eq!(decoded.amf_set_id, 0x3FF);
        assert_eq!(decoded.amf_pointer, 0x3F);
        assert_eq!(decoded.tmsi, 0xFFFFFFFF);
    }
}

/// Tracking Area Identity (TAI)
///
/// A TAI uniquely identifies a tracking area within a PLMN and consists of:
/// - PLMN: The Public Land Mobile Network identifier
/// - TAC: Tracking Area Code (24-bit value, range 0-16777215)
///
/// TAI is used in 5G networks for mobility management and paging.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Tai {
    /// Public Land Mobile Network identifier
    pub plmn: Plmn,
    /// Tracking Area Code (24-bit, range 0-16777215)
    pub tac: u32,
}

impl Tai {
    /// Creates a new TAI with the given PLMN and TAC.
    ///
    /// # Arguments
    /// * `plmn` - The PLMN identifier
    /// * `tac` - Tracking Area Code (24-bit value)
    pub const fn new(plmn: Plmn, tac: u32) -> Self {
        Self { plmn, tac }
    }

    /// Creates a new TAI from individual MCC, MNC, and TAC values.
    ///
    /// # Arguments
    /// * `mcc` - Mobile Country Code (3 digits)
    /// * `mnc` - Mobile Network Code (2-3 digits)
    /// * `long_mnc` - Whether MNC is 3 digits
    /// * `tac` - Tracking Area Code (24-bit value)
    pub const fn from_parts(mcc: u16, mnc: u16, long_mnc: bool, tac: u32) -> Self {
        Self {
            plmn: Plmn::new(mcc, mnc, long_mnc),
            tac,
        }
    }

    /// Returns true if this TAI has valid values set.
    ///
    /// A TAI is considered to have a value if either the PLMN has a value
    /// or the TAC is non-zero.
    pub fn has_value(&self) -> bool {
        self.plmn.has_value() || self.tac > 0
    }

    /// Encodes the TAI to 3GPP format (6 bytes).
    ///
    /// The encoding follows 3GPP TS 24.501 format:
    /// - Bytes 0-2: PLMN in 3GPP encoding
    /// - Bytes 3-5: TAC in big-endian format (24-bit)
    pub fn encode(&self) -> [u8; 6] {
        let plmn_bytes = self.plmn.encode();
        let tac_bytes = [
            ((self.tac >> 16) & 0xFF) as u8,
            ((self.tac >> 8) & 0xFF) as u8,
            (self.tac & 0xFF) as u8,
        ];

        [
            plmn_bytes[0],
            plmn_bytes[1],
            plmn_bytes[2],
            tac_bytes[0],
            tac_bytes[1],
            tac_bytes[2],
        ]
    }

    /// Decodes a TAI from 3GPP format (6 bytes).
    ///
    /// # Arguments
    /// * `bytes` - 6-byte array in 3GPP TAI encoding format
    ///
    /// # Returns
    /// The decoded TAI
    pub fn decode(bytes: [u8; 6]) -> Self {
        let plmn = Plmn::decode([bytes[0], bytes[1], bytes[2]]);
        let tac = ((bytes[3] as u32) << 16) | ((bytes[4] as u32) << 8) | (bytes[5] as u32);

        Self { plmn, tac }
    }
}

impl fmt::Debug for Tai {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tai({:?}, tac={})", self.plmn, self.tac)
    }
}

impl fmt::Display for Tai {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-{}", self.plmn, self.tac)
    }
}

impl Default for Tai {
    fn default() -> Self {
        Self {
            plmn: Plmn::default(),
            tac: 0,
        }
    }
}

/// Single Network Slice Selection Assistance Information (S-NSSAI)
///
/// S-NSSAI identifies a network slice and consists of:
/// - SST (Slice/Service Type): 8-bit value identifying the slice type
/// - SD (Slice Differentiator): Optional 24-bit value for further differentiation
///
/// Standard SST values (3GPP TS 23.501):
/// - 1: eMBB (enhanced Mobile Broadband)
/// - 2: URLLC (Ultra-Reliable Low-Latency Communications)
/// - 3: MIoT (Massive IoT)
/// - 4: V2X (Vehicle-to-Everything)
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SNssai {
    /// Slice/Service Type (8-bit)
    pub sst: u8,
    /// Slice Differentiator (optional 24-bit value)
    pub sd: Option<[u8; 3]>,
}

impl SNssai {
    /// Creates a new S-NSSAI with only SST (no SD).
    ///
    /// # Arguments
    /// * `sst` - Slice/Service Type
    pub const fn new(sst: u8) -> Self {
        Self { sst, sd: None }
    }

    /// Creates a new S-NSSAI with SST and SD.
    ///
    /// # Arguments
    /// * `sst` - Slice/Service Type
    /// * `sd` - Slice Differentiator (24-bit value as 3 bytes)
    pub const fn with_sd(sst: u8, sd: [u8; 3]) -> Self {
        Self { sst, sd: Some(sd) }
    }

    /// Creates a new S-NSSAI with SST and SD from a u32 value.
    ///
    /// # Arguments
    /// * `sst` - Slice/Service Type
    /// * `sd` - Slice Differentiator as u32 (only lower 24 bits used)
    pub const fn with_sd_u32(sst: u8, sd: u32) -> Self {
        Self {
            sst,
            sd: Some([
                ((sd >> 16) & 0xFF) as u8,
                ((sd >> 8) & 0xFF) as u8,
                (sd & 0xFF) as u8,
            ]),
        }
    }

    /// Returns the SD as a u32 value, or None if SD is not set.
    pub fn sd_as_u32(&self) -> Option<u32> {
        self.sd.map(|sd| ((sd[0] as u32) << 16) | ((sd[1] as u32) << 8) | (sd[2] as u32))
    }

    /// Returns true if this S-NSSAI has a valid SST value set.
    pub fn has_value(&self) -> bool {
        self.sst > 0 || self.sd.is_some()
    }

    /// Encodes the S-NSSAI to 3GPP format.
    ///
    /// The encoding follows 3GPP TS 24.501:
    /// - 1 byte: SST
    /// - 3 bytes (optional): SD in big-endian format
    ///
    /// Returns 1 byte if SD is None, 4 bytes if SD is present.
    pub fn encode(&self) -> Vec<u8> {
        match self.sd {
            Some(sd) => vec![self.sst, sd[0], sd[1], sd[2]],
            None => vec![self.sst],
        }
    }

    /// Decodes an S-NSSAI from 3GPP format.
    ///
    /// # Arguments
    /// * `bytes` - Byte slice containing the encoded S-NSSAI (1 or 4 bytes)
    ///
    /// # Returns
    /// The decoded S-NSSAI, or None if the input is invalid
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        match bytes.len() {
            1 => Some(Self {
                sst: bytes[0],
                sd: None,
            }),
            4 => Some(Self {
                sst: bytes[0],
                sd: Some([bytes[1], bytes[2], bytes[3]]),
            }),
            _ => None,
        }
    }
}

impl fmt::Debug for SNssai {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.sd {
            Some(sd) => {
                let sd_val = ((sd[0] as u32) << 16) | ((sd[1] as u32) << 8) | (sd[2] as u32);
                write!(f, "SNssai(sst={}, sd={:06X})", self.sst, sd_val)
            }
            None => write!(f, "SNssai(sst={})", self.sst),
        }
    }
}

impl fmt::Display for SNssai {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.sd {
            Some(sd) => {
                let sd_val = ((sd[0] as u32) << 16) | ((sd[1] as u32) << 8) | (sd[2] as u32);
                write!(f, "{}-{:06X}", self.sst, sd_val)
            }
            None => write!(f, "{}", self.sst),
        }
    }
}

impl Default for SNssai {
    fn default() -> Self {
        Self { sst: 0, sd: None }
    }
}

/// Network Slice configuration containing multiple S-NSSAIs.
///
/// This represents a collection of network slices that a UE or gNB supports.
/// Duplicate slices are automatically prevented when using `add_if_not_exists`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NetworkSlice {
    /// List of S-NSSAIs in this network slice configuration
    pub slices: Vec<SNssai>,
}

impl NetworkSlice {
    /// Creates a new empty NetworkSlice.
    pub const fn new() -> Self {
        Self { slices: Vec::new() }
    }

    /// Creates a NetworkSlice from a vector of S-NSSAIs.
    pub fn from_slices(slices: Vec<SNssai>) -> Self {
        Self { slices }
    }

    /// Adds an S-NSSAI if it doesn't already exist in the collection.
    ///
    /// # Arguments
    /// * `slice` - The S-NSSAI to add
    ///
    /// # Returns
    /// `true` if the slice was added, `false` if it already existed
    pub fn add_if_not_exists(&mut self, slice: SNssai) -> bool {
        if !self.slices.contains(&slice) {
            self.slices.push(slice);
            true
        } else {
            false
        }
    }

    /// Returns true if the collection contains the given S-NSSAI.
    pub fn contains(&self, slice: &SNssai) -> bool {
        self.slices.contains(slice)
    }

    /// Returns the number of S-NSSAIs in the collection.
    pub fn len(&self) -> usize {
        self.slices.len()
    }

    /// Returns true if the collection is empty.
    pub fn is_empty(&self) -> bool {
        self.slices.is_empty()
    }

    /// Returns an iterator over the S-NSSAIs.
    pub fn iter(&self) -> impl Iterator<Item = &SNssai> {
        self.slices.iter()
    }
}

/// SUPI type enumeration.
///
/// Defines the type of Subscription Permanent Identifier per 3GPP TS 23.003.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SupiType {
    /// International Mobile Subscriber Identity (IMSI-based SUPI)
    Imsi,
    /// Network Access Identifier (NAI-based SUPI)
    Nai,
}

impl SupiType {
    /// Returns the string prefix for this SUPI type.
    pub fn prefix(&self) -> &'static str {
        match self {
            SupiType::Imsi => "imsi",
            SupiType::Nai => "nai",
        }
    }
}

impl fmt::Display for SupiType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.prefix())
    }
}

/// Subscription Permanent Identifier (SUPI).
///
/// SUPI is the permanent identity of a subscriber in 5G networks.
/// It can be either IMSI-based or NAI-based per 3GPP TS 23.003.
///
/// Format: `<type>-<value>` (e.g., "imsi-310410123456789")
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Supi {
    /// The type of SUPI (IMSI or NAI)
    pub supi_type: SupiType,
    /// The SUPI value (e.g., "310410123456789" for IMSI)
    pub value: String,
}

impl Supi {
    /// Creates a new SUPI with the given type and value.
    ///
    /// # Arguments
    /// * `supi_type` - The type of SUPI (IMSI or NAI)
    /// * `value` - The SUPI value string
    pub fn new(supi_type: SupiType, value: impl Into<String>) -> Self {
        Self {
            supi_type,
            value: value.into(),
        }
    }

    /// Creates a new IMSI-based SUPI.
    ///
    /// # Arguments
    /// * `value` - The IMSI value (e.g., "310410123456789")
    pub fn imsi(value: impl Into<String>) -> Self {
        Self::new(SupiType::Imsi, value)
    }

    /// Creates a new NAI-based SUPI.
    ///
    /// # Arguments
    /// * `value` - The NAI value
    pub fn nai(value: impl Into<String>) -> Self {
        Self::new(SupiType::Nai, value)
    }

    /// Parses a SUPI from a string in the format "type-value".
    ///
    /// # Arguments
    /// * `s` - The SUPI string (e.g., "imsi-310410123456789")
    ///
    /// # Returns
    /// The parsed SUPI, or None if the format is invalid
    pub fn parse(s: &str) -> Option<Self> {
        let (type_str, value) = s.split_once('-')?;
        let supi_type = match type_str.to_lowercase().as_str() {
            "imsi" => SupiType::Imsi,
            "nai" => SupiType::Nai,
            _ => return None,
        };
        Some(Self::new(supi_type, value))
    }

    /// Returns true if this SUPI has a non-empty value.
    pub fn has_value(&self) -> bool {
        !self.value.is_empty()
    }
}

impl fmt::Debug for Supi {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Supi({}-{})", self.supi_type, self.value)
    }
}

impl fmt::Display for Supi {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-{}", self.supi_type, self.value)
    }
}

/// 5G Globally Unique Temporary Identifier (5G-GUTI).
///
/// GUTI is a temporary identity assigned to a UE by the AMF.
/// It consists of:
/// - PLMN: Public Land Mobile Network identifier
/// - AMF Region ID: 8-bit identifier for the AMF region
/// - AMF Set ID: 10-bit identifier for the AMF set within the region
/// - AMF Pointer: 6-bit identifier for the AMF within the set
/// - 5G-TMSI: 32-bit Temporary Mobile Subscriber Identity
///
/// Per 3GPP TS 23.003 Section 2.10.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Guti {
    /// Public Land Mobile Network identifier
    pub plmn: Plmn,
    /// AMF Region ID (8-bit)
    pub amf_region_id: u8,
    /// AMF Set ID (10-bit, range 0-1023)
    pub amf_set_id: u16,
    /// AMF Pointer (6-bit, range 0-63)
    pub amf_pointer: u8,
    /// 5G Temporary Mobile Subscriber Identity (32-bit)
    pub tmsi: u32,
}

impl Guti {
    /// Maximum value for AMF Set ID (10-bit)
    pub const MAX_AMF_SET_ID: u16 = 0x3FF;
    /// Maximum value for AMF Pointer (6-bit)
    pub const MAX_AMF_POINTER: u8 = 0x3F;

    /// Creates a new GUTI with the given values.
    ///
    /// # Arguments
    /// * `plmn` - Public Land Mobile Network identifier
    /// * `amf_region_id` - AMF Region ID (8-bit)
    /// * `amf_set_id` - AMF Set ID (10-bit, will be masked to 10 bits)
    /// * `amf_pointer` - AMF Pointer (6-bit, will be masked to 6 bits)
    /// * `tmsi` - 5G-TMSI (32-bit)
    pub fn new(plmn: Plmn, amf_region_id: u8, amf_set_id: u16, amf_pointer: u8, tmsi: u32) -> Self {
        Self {
            plmn,
            amf_region_id,
            amf_set_id: amf_set_id & Self::MAX_AMF_SET_ID,
            amf_pointer: amf_pointer & Self::MAX_AMF_POINTER,
            tmsi,
        }
    }

    /// Creates a new GUTI from individual PLMN components.
    ///
    /// # Arguments
    /// * `mcc` - Mobile Country Code
    /// * `mnc` - Mobile Network Code
    /// * `long_mnc` - Whether MNC is 3 digits
    /// * `amf_region_id` - AMF Region ID (8-bit)
    /// * `amf_set_id` - AMF Set ID (10-bit)
    /// * `amf_pointer` - AMF Pointer (6-bit)
    /// * `tmsi` - 5G-TMSI (32-bit)
    pub fn from_parts(
        mcc: u16,
        mnc: u16,
        long_mnc: bool,
        amf_region_id: u8,
        amf_set_id: u16,
        amf_pointer: u8,
        tmsi: u32,
    ) -> Self {
        Self::new(
            Plmn::new(mcc, mnc, long_mnc),
            amf_region_id,
            amf_set_id,
            amf_pointer,
            tmsi,
        )
    }

    /// Returns true if this GUTI has valid values set.
    pub fn has_value(&self) -> bool {
        self.plmn.has_value() || self.amf_region_id > 0 || self.amf_set_id > 0 || self.tmsi > 0
    }

    /// Returns the AMF Identifier (AMFI) as a 24-bit value.
    ///
    /// AMFI = AMF Region ID (8 bits) | AMF Set ID (10 bits) | AMF Pointer (6 bits)
    pub fn amf_id(&self) -> u32 {
        ((self.amf_region_id as u32) << 16)
            | ((self.amf_set_id as u32 & 0x3FF) << 6)
            | (self.amf_pointer as u32 & 0x3F)
    }

    /// Creates a GUTI from PLMN and AMF ID.
    ///
    /// # Arguments
    /// * `plmn` - Public Land Mobile Network identifier
    /// * `amf_id` - 24-bit AMF Identifier
    /// * `tmsi` - 5G-TMSI (32-bit)
    pub fn from_amf_id(plmn: Plmn, amf_id: u32, tmsi: u32) -> Self {
        let amf_region_id = ((amf_id >> 16) & 0xFF) as u8;
        let amf_set_id = ((amf_id >> 6) & 0x3FF) as u16;
        let amf_pointer = (amf_id & 0x3F) as u8;
        Self::new(plmn, amf_region_id, amf_set_id, amf_pointer, tmsi)
    }

    /// Returns the 5G-S-TMSI as a 48-bit value.
    ///
    /// 5G-S-TMSI = AMF Set ID (10 bits) | AMF Pointer (6 bits) | 5G-TMSI (32 bits)
    pub fn s_tmsi(&self) -> u64 {
        let amf_set_pointer =
            ((self.amf_set_id as u64 & 0x3FF) << 6) | (self.amf_pointer as u64 & 0x3F);
        (amf_set_pointer << 32) | (self.tmsi as u64)
    }

    /// Creates a partial GUTI from 5G-S-TMSI value.
    ///
    /// Note: This creates a GUTI with default PLMN and AMF Region ID,
    /// as those are not included in the S-TMSI.
    ///
    /// # Arguments
    /// * `s_tmsi` - 48-bit 5G-S-TMSI value
    pub fn from_s_tmsi(s_tmsi: u64) -> Self {
        let amf_set_id = ((s_tmsi >> 38) & 0x3FF) as u16;
        let amf_pointer = ((s_tmsi >> 32) & 0x3F) as u8;
        let tmsi = (s_tmsi & 0xFFFFFFFF) as u32;
        Self::new(Plmn::default(), 0, amf_set_id, amf_pointer, tmsi)
    }

    /// Encodes the GUTI to 3GPP format (10 bytes).
    ///
    /// The encoding follows 3GPP TS 24.501:
    /// - Bytes 0-2: PLMN in 3GPP encoding
    /// - Byte 3: AMF Region ID
    /// - Bytes 4-5: AMF Set ID (10 bits) | AMF Pointer (6 bits)
    /// - Bytes 6-9: 5G-TMSI in big-endian format
    pub fn encode(&self) -> [u8; 10] {
        let plmn_bytes = self.plmn.encode();
        let amf_set_pointer =
            ((self.amf_set_id as u16 & 0x3FF) << 6) | (self.amf_pointer as u16 & 0x3F);

        [
            plmn_bytes[0],
            plmn_bytes[1],
            plmn_bytes[2],
            self.amf_region_id,
            (amf_set_pointer >> 8) as u8,
            (amf_set_pointer & 0xFF) as u8,
            ((self.tmsi >> 24) & 0xFF) as u8,
            ((self.tmsi >> 16) & 0xFF) as u8,
            ((self.tmsi >> 8) & 0xFF) as u8,
            (self.tmsi & 0xFF) as u8,
        ]
    }

    /// Decodes a GUTI from 3GPP format (10 bytes).
    ///
    /// # Arguments
    /// * `bytes` - 10-byte array in 3GPP GUTI encoding format
    ///
    /// # Returns
    /// The decoded GUTI
    pub fn decode(bytes: [u8; 10]) -> Self {
        let plmn = Plmn::decode([bytes[0], bytes[1], bytes[2]]);
        let amf_region_id = bytes[3];
        let amf_set_pointer = ((bytes[4] as u16) << 8) | (bytes[5] as u16);
        let amf_set_id = (amf_set_pointer >> 6) & 0x3FF;
        let amf_pointer = (amf_set_pointer & 0x3F) as u8;
        let tmsi = ((bytes[6] as u32) << 24)
            | ((bytes[7] as u32) << 16)
            | ((bytes[8] as u32) << 8)
            | (bytes[9] as u32);

        Self {
            plmn,
            amf_region_id,
            amf_set_id,
            amf_pointer,
            tmsi,
        }
    }
}

impl fmt::Debug for Guti {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Guti({:?}, region={}, set={}, ptr={}, tmsi={:08X})",
            self.plmn, self.amf_region_id, self.amf_set_id, self.amf_pointer, self.tmsi
        )
    }
}

impl fmt::Display for Guti {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}-{:02X}-{:03X}-{:02X}-{:08X}",
            self.plmn, self.amf_region_id, self.amf_set_id, self.amf_pointer, self.tmsi
        )
    }
}

impl Default for Guti {
    fn default() -> Self {
        Self {
            plmn: Plmn::default(),
            amf_region_id: 0,
            amf_set_id: 0,
            amf_pointer: 0,
            tmsi: 0,
        }
    }
}
