//! UE AS (Access Stratum) security context and MAC-I derivation
//!
//! Implements the ShortMAC-I computation of TS 38.331 §5.3.7.4 (RRC
//! re-establishment) and the resumeMAC-I computation of §5.3.13.3 (RRC
//! resume):
//!
//! > set the shortMAC-I to the 16 least significant bits of the MAC-I
//! > calculated over the ASN.1 encoded VarShortMAC-Input, with the KRRCint
//! > key and integrity protection algorithm used in the source PCell, and
//! > with all input bits for COUNT, BEARER and DIRECTION set to binary ones.
//!
//! The resumeMAC-I is derived identically over `VarResumeMAC-Input` (same
//! sourcePhysCellId / targetCellIdentity / source-c-RNTI fields) per
//! §5.3.13.3.
//!
//! The `VarShortMAC-Input` / `VarResumeMAC-Input` (sourcePhysCellId,
//! targetCellIdentity, source-c-RNTI) are UPER-encoded with the generated
//! ASN.1 types from `nextgsim-rrc`, and the MAC-I is computed with the NIA
//! algorithms from `nextgsim-crypto`.

use bitvec::prelude::*;

use nextgsim_crypto::nia::{nia1_compute_mac, nia2_compute_mac, nia3_compute_mac};
use nextgsim_rrc::codec::generated::{
    CellIdentity, PhysCellId, RNTI_Value, VarResumeMAC_Input, VarShortMAC_Input,
};
use nextgsim_rrc::codec::{encode_rrc, RrcCodecError};

/// 5G AS integrity protection algorithm (TS 33.501 §5.11.1)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegrityAlgorithm {
    /// NIA0 — null integrity (MAC-I is all zeros)
    Nia0,
    /// 128-NIA1 — SNOW3G based
    Nia1,
    /// 128-NIA2 — AES-CMAC based
    Nia2,
    /// 128-NIA3 — ZUC based
    Nia3,
}

/// AS security context established by the AS Security Mode Command
#[derive(Debug, Clone)]
pub struct AsSecurityContext {
    /// KRRCint — RRC integrity protection key (128 bits)
    pub k_rrc_int: [u8; 16],
    /// Integrity protection algorithm of the source PCell
    pub integrity_algorithm: IntegrityAlgorithm,
    /// C-RNTI allocated by the source PCell
    pub c_rnti: u16,
}

/// Errors during ShortMAC-I derivation
#[derive(Debug, thiserror::Error)]
pub enum ShortMacError {
    /// VarShortMAC-Input encoding failed
    #[error("VarShortMAC-Input encoding error: {0}")]
    EncodeError(#[from] RrcCodecError),
}

/// Computes the ShortMAC-I per TS 38.331 §5.3.7.4.
///
/// # Arguments
/// * `ctx` - AS security context of the source PCell
/// * `source_pci` - Physical cell identity of the source PCell (0..1007)
/// * `target_cell_identity` - 36-bit NR Cell Identity of the target cell
///
/// # Returns
/// The 16 least significant bits of the MAC-I computed over the UPER-encoded
/// `VarShortMAC-Input` with COUNT, BEARER and DIRECTION set to binary ones.
pub fn compute_short_mac_i(
    ctx: &AsSecurityContext,
    source_pci: u16,
    target_cell_identity: u64,
) -> Result<u16, ShortMacError> {
    // Build the 36-bit target cell identity
    let mut cell_id_bv: BitVec<u8, Msb0> = BitVec::with_capacity(36);
    for i in (0..36).rev() {
        cell_id_bv.push((target_cell_identity >> i) & 1 == 1);
    }

    let input = VarShortMAC_Input {
        source_phys_cell_id: PhysCellId(source_pci),
        target_cell_identity: CellIdentity(cell_id_bv),
        source_c_rnti: RNTI_Value(ctx.c_rnti),
    };
    let encoded = encode_rrc(&input)?;

    Ok(mac_i_lsb16(ctx, &encoded))
}

/// Computes the resumeMAC-I per TS 38.331 §5.3.13.3.
///
/// > set the resumeMAC-I to the 16 least significant bits of the MAC-I
/// > calculated over the ASN.1 encoded as per clause 8 (i.e a multiple of 8
/// > bits) VarResumeMAC-Input, with the KRRCint key in the used AS security
/// > context and the previously configured integrity protection algorithm, and
/// > with all input bits for COUNT, BEARER and DIRECTION set to binary ones.
///
/// The `VarResumeMAC-Input` carries `sourcePhysCellId` (PCI of the source
/// PCell in which the UE received the last RRCRelease with suspendConfig),
/// `targetCellIdentity` (the 36-bit NR Cell Identity of the cell the UE is
/// resuming on) and `source-c-RNTI` (the C-RNTI used in the source PCell).
///
/// # Arguments
/// * `ctx` - AS security context of the source PCell (KRRCint, NIA, C-RNTI)
/// * `source_pci` - Physical cell identity of the source PCell (0..1007)
/// * `target_cell_identity` - 36-bit NR Cell Identity of the target cell
///
/// # Returns
/// The 16 least significant bits of the MAC-I computed over the UPER-encoded
/// `VarResumeMAC-Input` with COUNT, BEARER and DIRECTION set to binary ones.
pub fn compute_resume_mac_i(
    ctx: &AsSecurityContext,
    source_pci: u16,
    target_cell_identity: u64,
) -> Result<u16, ShortMacError> {
    // Build the 36-bit target cell identity
    let mut cell_id_bv: BitVec<u8, Msb0> = BitVec::with_capacity(36);
    for i in (0..36).rev() {
        cell_id_bv.push((target_cell_identity >> i) & 1 == 1);
    }

    let input = VarResumeMAC_Input {
        source_phys_cell_id: PhysCellId(source_pci),
        target_cell_identity: CellIdentity(cell_id_bv),
        source_c_rnti: RNTI_Value(ctx.c_rnti),
    };
    let encoded = encode_rrc(&input)?;

    Ok(mac_i_lsb16(ctx, &encoded))
}

/// Computes the 16 least significant bits of the 32-bit MAC-I over the
/// UPER-encoded `VarShortMAC-Input` / `VarResumeMAC-Input` message, with the
/// KRRCint key and the source PCell integrity algorithm, with COUNT, BEARER
/// and DIRECTION all set to binary ones (TS 38.331 §5.3.7.4 / §5.3.13.3).
fn mac_i_lsb16(ctx: &AsSecurityContext, encoded: &[u8]) -> u16 {
    // COUNT, BEARER and DIRECTION all set to binary ones
    const COUNT: u32 = 0xFFFF_FFFF;
    const BEARER: u8 = 0x1F;
    const DIRECTION: u8 = 0x01;

    let mac = match ctx.integrity_algorithm {
        // NIA0 produces an all-zero MAC (TS 33.501 D.1)
        IntegrityAlgorithm::Nia0 => [0u8; 4],
        IntegrityAlgorithm::Nia1 => {
            nia1_compute_mac(COUNT, BEARER, DIRECTION, &ctx.k_rrc_int, encoded)
        }
        IntegrityAlgorithm::Nia2 => {
            nia2_compute_mac(COUNT, BEARER, DIRECTION, &ctx.k_rrc_int, encoded)
        }
        IntegrityAlgorithm::Nia3 => {
            nia3_compute_mac(COUNT, BEARER, DIRECTION, &ctx.k_rrc_int, encoded)
        }
    };

    // 16 least significant bits of the 32-bit MAC-I
    u16::from_be_bytes([mac[2], mac[3]])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_ctx(alg: IntegrityAlgorithm) -> AsSecurityContext {
        AsSecurityContext {
            k_rrc_int: [
                0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D,
                0x0E, 0x0F,
            ],
            integrity_algorithm: alg,
            c_rnti: 0x1234,
        }
    }

    #[test]
    fn test_short_mac_i_deterministic() {
        let ctx = test_ctx(IntegrityAlgorithm::Nia2);
        let a = compute_short_mac_i(&ctx, 100, 0x10).unwrap();
        let b = compute_short_mac_i(&ctx, 100, 0x10).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_short_mac_i_depends_on_inputs() {
        let ctx = test_ctx(IntegrityAlgorithm::Nia2);
        let base = compute_short_mac_i(&ctx, 100, 0x10).unwrap();

        // Different source PCI changes the MAC
        assert_ne!(base, compute_short_mac_i(&ctx, 101, 0x10).unwrap());
        // Different target cell changes the MAC
        assert_ne!(base, compute_short_mac_i(&ctx, 100, 0x11).unwrap());

        // Different C-RNTI changes the MAC
        let mut ctx2 = test_ctx(IntegrityAlgorithm::Nia2);
        ctx2.c_rnti = 0x4321;
        assert_ne!(base, compute_short_mac_i(&ctx2, 100, 0x10).unwrap());

        // Different key changes the MAC
        let mut ctx3 = test_ctx(IntegrityAlgorithm::Nia2);
        ctx3.k_rrc_int = [0xFF; 16];
        assert_ne!(base, compute_short_mac_i(&ctx3, 100, 0x10).unwrap());
    }

    #[test]
    fn test_short_mac_i_depends_on_algorithm() {
        let nia1 = compute_short_mac_i(&test_ctx(IntegrityAlgorithm::Nia1), 100, 0x10).unwrap();
        let nia2 = compute_short_mac_i(&test_ctx(IntegrityAlgorithm::Nia2), 100, 0x10).unwrap();
        let nia3 = compute_short_mac_i(&test_ctx(IntegrityAlgorithm::Nia3), 100, 0x10).unwrap();
        // The three real algorithms must not agree on the same input
        assert!(!(nia1 == nia2 && nia2 == nia3));
    }

    #[test]
    fn test_short_mac_i_nia0_is_zero() {
        let mac = compute_short_mac_i(&test_ctx(IntegrityAlgorithm::Nia0), 100, 0x10).unwrap();
        assert_eq!(mac, 0);
    }

    #[test]
    fn test_resume_mac_i_deterministic() {
        let ctx = test_ctx(IntegrityAlgorithm::Nia2);
        let a = compute_resume_mac_i(&ctx, 100, 0x10).unwrap();
        let b = compute_resume_mac_i(&ctx, 100, 0x10).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_resume_mac_i_depends_on_inputs() {
        let ctx = test_ctx(IntegrityAlgorithm::Nia2);
        let base = compute_resume_mac_i(&ctx, 100, 0x10).unwrap();

        // Different source PCI changes the MAC
        assert_ne!(base, compute_resume_mac_i(&ctx, 101, 0x10).unwrap());
        // Different target cell changes the MAC
        assert_ne!(base, compute_resume_mac_i(&ctx, 100, 0x11).unwrap());

        // Different C-RNTI changes the MAC
        let mut ctx2 = test_ctx(IntegrityAlgorithm::Nia2);
        ctx2.c_rnti = 0x4321;
        assert_ne!(base, compute_resume_mac_i(&ctx2, 100, 0x10).unwrap());

        // Different key changes the MAC
        let mut ctx3 = test_ctx(IntegrityAlgorithm::Nia2);
        ctx3.k_rrc_int = [0xFF; 16];
        assert_ne!(base, compute_resume_mac_i(&ctx3, 100, 0x10).unwrap());
    }

    #[test]
    fn test_resume_mac_i_depends_on_algorithm() {
        let nia1 = compute_resume_mac_i(&test_ctx(IntegrityAlgorithm::Nia1), 100, 0x10).unwrap();
        let nia2 = compute_resume_mac_i(&test_ctx(IntegrityAlgorithm::Nia2), 100, 0x10).unwrap();
        let nia3 = compute_resume_mac_i(&test_ctx(IntegrityAlgorithm::Nia3), 100, 0x10).unwrap();
        // The three real algorithms must not agree on the same input
        assert!(!(nia1 == nia2 && nia2 == nia3));
    }

    #[test]
    fn test_resume_mac_i_nia0_is_zero() {
        let mac = compute_resume_mac_i(&test_ctx(IntegrityAlgorithm::Nia0), 100, 0x10).unwrap();
        assert_eq!(mac, 0);
    }

    #[test]
    fn test_resume_and_short_mac_i_share_input_layout() {
        // VarResumeMAC-Input and VarShortMAC-Input have identical SEQUENCE
        // layout (sourcePhysCellId, targetCellIdentity, source-c-RNTI), so for
        // the same security context and inputs the derived 16-bit MAC matches.
        let ctx = test_ctx(IntegrityAlgorithm::Nia2);
        let resume = compute_resume_mac_i(&ctx, 100, 0x10).unwrap();
        let short = compute_short_mac_i(&ctx, 100, 0x10).unwrap();
        assert_eq!(resume, short);
    }
}
