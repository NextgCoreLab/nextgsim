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

use nextgsim_crypto::kdf::{derive_rrc_up_key, AlgorithmTypeDistinguisher};
use nextgsim_crypto::nea::{nea1_encrypt, nea2_encrypt};
use nextgsim_crypto::nia::{nia1_compute_mac, nia2_compute_mac, nia3_compute_mac};
use nextgsim_rrc::codec::generated::{
    CellIdentity, PhysCellId, RNTI_Value, VarResumeMAC_Input, VarShortMAC_Input,
};
use nextgsim_rrc::codec::{encode_rrc, RrcCodecError};
use nextgsim_rrc::procedures::security_mode::{CipheringAlgorithmType, IntegrityAlgorithmType};

/// Wire-safety gate for UE AS-security (Wave-6 residue I5, TS 38.331 §5.3.4 /
/// TS 33.501 §6.7). When `false` (the default) the UE keeps its legacy DL-DCCH
/// handling verbatim, so the matched-sim registration/PDU/ping E2E is
/// byte-for-byte unchanged: the gNB's RRC SecurityModeCommand is handled by the
/// pre-I5 nibble dispatcher exactly as before.
///
/// Flipping this single const to `true` turns on real AS-security activation on
/// the UE: typed decode of the AS SecurityModeCommand, `KgNB`→K_RRCint/K_RRCenc
/// derivation, and PDCP SRB integrity/ciphering enforcement. The flip is a
/// MANUAL host gate — it requires the docker matched-sim A/B E2E sign-off
/// (hazard #269) because the paired gNB SMC sender and the AMF `KgNB`/uplink
/// NAS COUNT must agree end-to-end. It mirrors the gNB's `C5_TYPED_DCCH_DISPATCH`
/// wire-safety gate.
pub const I5_UE_AS_SECURITY: bool = false;

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

impl IntegrityAlgorithm {
    /// 3GPP algorithm identity (TS 33.501 §5.11.1): 0=NIA0 .. 3=NIA3. This is
    /// the value fed as the algorithm-id KDF parameter of the K_RRCint
    /// derivation (Annex A.8) and MUST match the gNB's `integrity_alg_id`.
    #[must_use]
    pub fn id(self) -> u8 {
        match self {
            IntegrityAlgorithm::Nia0 => 0,
            IntegrityAlgorithm::Nia1 => 1,
            IntegrityAlgorithm::Nia2 => 2,
            IntegrityAlgorithm::Nia3 => 3,
        }
    }

    /// Map a 3GPP integrity algorithm identity (0..3) to the enum. Returns
    /// `None` for reserved / unknown identities (fail-closed).
    #[must_use]
    pub fn from_id(id: u8) -> Option<Self> {
        match id {
            0 => Some(IntegrityAlgorithm::Nia0),
            1 => Some(IntegrityAlgorithm::Nia1),
            2 => Some(IntegrityAlgorithm::Nia2),
            3 => Some(IntegrityAlgorithm::Nia3),
            _ => None,
        }
    }
}

/// 5G AS ciphering algorithm (TS 33.501 §5.11.1)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CipheringAlgorithm {
    /// NEA0 — null ciphering (RRC PDUs are not enciphered)
    Nea0,
    /// 128-NEA1 — SNOW3G based
    Nea1,
    /// 128-NEA2 — AES-CTR based
    Nea2,
    /// 128-NEA3 — ZUC based
    Nea3,
}

impl CipheringAlgorithm {
    /// 3GPP algorithm identity (TS 33.501 §5.11.1): 0=NEA0 .. 3=NEA3. Fed as
    /// the algorithm-id KDF parameter of the K_RRCenc derivation (Annex A.8);
    /// MUST match the gNB's `ciphering_alg_id`.
    #[must_use]
    pub fn id(self) -> u8 {
        match self {
            CipheringAlgorithm::Nea0 => 0,
            CipheringAlgorithm::Nea1 => 1,
            CipheringAlgorithm::Nea2 => 2,
            CipheringAlgorithm::Nea3 => 3,
        }
    }

    /// Map a 3GPP ciphering algorithm identity (0..3) to the enum. Returns
    /// `None` for reserved / unknown identities (fail-closed).
    #[must_use]
    pub fn from_id(id: u8) -> Option<Self> {
        match id {
            0 => Some(CipheringAlgorithm::Nea0),
            1 => Some(CipheringAlgorithm::Nea1),
            2 => Some(CipheringAlgorithm::Nea2),
            3 => Some(CipheringAlgorithm::Nea3),
            _ => None,
        }
    }
}

impl From<IntegrityAlgorithmType> for IntegrityAlgorithm {
    /// Map the RRC-codec integrity algorithm (as decoded from the AS
    /// SecurityModeCommand's `SecurityAlgorithmConfig`) to the AS-security enum.
    fn from(alg: IntegrityAlgorithmType) -> Self {
        match alg {
            IntegrityAlgorithmType::Nia0 => IntegrityAlgorithm::Nia0,
            IntegrityAlgorithmType::Nia1 => IntegrityAlgorithm::Nia1,
            IntegrityAlgorithmType::Nia2 => IntegrityAlgorithm::Nia2,
            IntegrityAlgorithmType::Nia3 => IntegrityAlgorithm::Nia3,
        }
    }
}

impl From<CipheringAlgorithmType> for CipheringAlgorithm {
    /// Map the RRC-codec ciphering algorithm (as decoded from the AS
    /// SecurityModeCommand's `SecurityAlgorithmConfig`) to the AS-security enum.
    fn from(alg: CipheringAlgorithmType) -> Self {
        match alg {
            CipheringAlgorithmType::Nea0 => CipheringAlgorithm::Nea0,
            CipheringAlgorithmType::Nea1 => CipheringAlgorithm::Nea1,
            CipheringAlgorithmType::Nea2 => CipheringAlgorithm::Nea2,
            CipheringAlgorithmType::Nea3 => CipheringAlgorithm::Nea3,
        }
    }
}

/// AS security context established by the AS Security Mode Command
/// (TS 38.331 §5.3.4, TS 33.501 §6.7). Carries the RRC keys derived from
/// `KgNB` and the algorithms selected by the gNB, plus the source-PCell
/// C-RNTI used for the re-establishment/resume ShortMAC-I.
#[derive(Clone)]
pub struct AsSecurityContext {
    /// KRRCint — RRC integrity protection key (128 bits), derived from KgNB
    /// with `AlgorithmTypeDistinguisher::RrcInt` (TS 33.501 Annex A.8).
    pub k_rrc_int: [u8; 16],
    /// KRRCenc — RRC ciphering key (128 bits), derived from KgNB with
    /// `AlgorithmTypeDistinguisher::RrcEnc` (TS 33.501 Annex A.8).
    pub k_rrc_enc: [u8; 16],
    /// Integrity protection algorithm of the source PCell
    pub integrity_algorithm: IntegrityAlgorithm,
    /// Ciphering algorithm of the source PCell
    pub ciphering_algorithm: CipheringAlgorithm,
    /// C-RNTI allocated by the source PCell
    pub c_rnti: u16,
}

impl std::fmt::Debug for AsSecurityContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never print key material.
        f.debug_struct("AsSecurityContext")
            .field("integrity_algorithm", &self.integrity_algorithm)
            .field("ciphering_algorithm", &self.ciphering_algorithm)
            .field("c_rnti", &self.c_rnti)
            .finish_non_exhaustive()
    }
}

/// BEARER identity for SRB1 fed to the PDCP security algorithms: the radio
/// bearer identity minus one (TS 33.501 §6.5 referencing TS 33.401 §7); SRB1
/// (SRB-Identity 1) → 0.
pub const SRB1_BEARER: u8 = 0;
/// DIRECTION bit for uplink (UE→gNB) PDCP security inputs (TS 33.501 §6.5).
pub const DIRECTION_UPLINK: u8 = 0;
/// DIRECTION bit for downlink (gNB→UE) PDCP security inputs (TS 33.501 §6.5).
pub const DIRECTION_DOWNLINK: u8 = 1;

/// Errors from PDCP SRB integrity/ciphering enforcement (TS 38.323 §5.8/§5.9,
/// TS 33.501 §6.5). All are fail-closed: the plaintext is never surfaced.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum AsSecurityError {
    /// The received PDCP MAC-I did not match the recomputed MAC-I.
    #[error("PDCP integrity check failed (MAC-I mismatch)")]
    IntegrityCheckFailed,
    /// The received PDCP payload is shorter than the 32-bit MAC-I trailer.
    #[error("PDCP payload shorter than the 32-bit MAC-I")]
    PduTooShort,
    /// NEA3 ciphering has no keystream implementation in `nextgsim-crypto`
    /// (only NIA3 integrity exists), so an SRB ciphered with NEA3 cannot be
    /// processed — the caller must fail closed rather than pass it in clear.
    #[error("ciphering algorithm not supported (NEA3 keystream unavailable)")]
    UnsupportedCipheringAlgorithm,
}

/// Constant-time equality of a computed 4-byte MAC-I against a received one:
/// folds all byte differences so the comparison time does not depend on where
/// the first mismatch is (TS 33.501 §6.5 discard-on-failure must not leak a
/// timing oracle).
fn ct_eq_mac(computed: &[u8; 4], received: &[u8]) -> bool {
    if received.len() != 4 {
        return false;
    }
    let mut diff = 0u8;
    for i in 0..4 {
        diff |= computed[i] ^ received[i];
    }
    diff == 0
}

impl AsSecurityContext {
    /// Derive the AS security context from `KgNB` and the algorithms the gNB
    /// selected in the AS SecurityModeCommand (TS 33.501 §6.7 / Annex A.8,
    /// TS 38.331 §5.3.4.3).
    ///
    /// `K_RRCenc = KDF(KgNB, RRC-enc-alg-distinguisher, ciphering-alg-id)` and
    /// `K_RRCint = KDF(KgNB, RRC-int-alg-distinguisher, integrity-alg-id)`,
    /// taking the least-significant 128 bits (done by `derive_rrc_up_key`).
    /// These are byte-identical to the gNB's own `derive_rrc_up_key` calls, so
    /// the two ends share the same SRB keys.
    #[must_use]
    pub fn derive_from_kgnb(
        kgnb: &[u8; 32],
        ciphering_algorithm: CipheringAlgorithm,
        integrity_algorithm: IntegrityAlgorithm,
        c_rnti: u16,
    ) -> Self {
        Self {
            k_rrc_int: derive_rrc_up_key(
                kgnb,
                AlgorithmTypeDistinguisher::RrcInt,
                integrity_algorithm.id(),
            ),
            k_rrc_enc: derive_rrc_up_key(
                kgnb,
                AlgorithmTypeDistinguisher::RrcEnc,
                ciphering_algorithm.id(),
            ),
            integrity_algorithm,
            ciphering_algorithm,
            c_rnti,
        }
    }

    /// Compute the 32-bit PDCP MAC-I over an SRB RRC message (TS 38.323 §5.9,
    /// TS 33.501 §6.5 / Annex D) with `K_RRCint` and the negotiated NIA.
    /// `message` is the data unit being integrity-protected (the RRC PDU
    /// carried on the SRB); COUNT/BEARER/DIRECTION are the PDCP security inputs.
    /// NIA0 yields an all-zero MAC-I (TS 33.501 Annex D.1).
    #[must_use]
    pub fn compute_rrc_mac_i(
        &self,
        count: u32,
        bearer: u8,
        direction: u8,
        message: &[u8],
    ) -> [u8; 4] {
        match self.integrity_algorithm {
            IntegrityAlgorithm::Nia0 => [0u8; 4],
            IntegrityAlgorithm::Nia1 => {
                nia1_compute_mac(count, bearer, direction, &self.k_rrc_int, message)
            }
            IntegrityAlgorithm::Nia2 => {
                nia2_compute_mac(count, bearer, direction, &self.k_rrc_int, message)
            }
            IntegrityAlgorithm::Nia3 => {
                nia3_compute_mac(count, bearer, direction, &self.k_rrc_int, message)
            }
        }
    }

    /// Apply the SRB ciphering keystream in place with `K_RRCenc` and the
    /// negotiated NEA (TS 38.323 §5.8). NEA0 is a no-op. NEA is a stream
    /// cipher (CTR), so this same routine deciphers. NEA3 has no keystream in
    /// `nextgsim-crypto` and is reported as unsupported so the caller fails
    /// closed.
    fn apply_ciphering(
        &self,
        count: u32,
        bearer: u8,
        direction: u8,
        data: &mut [u8],
    ) -> Result<(), AsSecurityError> {
        match self.ciphering_algorithm {
            CipheringAlgorithm::Nea0 => Ok(()),
            CipheringAlgorithm::Nea1 => {
                nea1_encrypt(count, bearer, direction, &self.k_rrc_enc, data);
                Ok(())
            }
            CipheringAlgorithm::Nea2 => {
                nea2_encrypt(count, bearer, direction, &self.k_rrc_enc, data);
                Ok(())
            }
            CipheringAlgorithm::Nea3 => Err(AsSecurityError::UnsupportedCipheringAlgorithm),
        }
    }

    /// PDCP-protect an SRB RRC message for transmission (TS 38.323 §5.8/§5.9):
    /// compute the 32-bit MAC-I over the plaintext, append it, then cipher the
    /// concatenation `data || MAC-I` (NEA0 leaves it in the clear). Returns the
    /// PDCP payload to place on the SRB.
    pub fn protect_srb(
        &self,
        count: u32,
        bearer: u8,
        direction: u8,
        plaintext: &[u8],
    ) -> Result<Vec<u8>, AsSecurityError> {
        let mac = self.compute_rrc_mac_i(count, bearer, direction, plaintext);
        let mut out = Vec::with_capacity(plaintext.len() + 4);
        out.extend_from_slice(plaintext);
        out.extend_from_slice(&mac);
        self.apply_ciphering(count, bearer, direction, &mut out)?;
        Ok(out)
    }

    /// PDCP-unprotect a received SRB PDCP payload (TS 38.323 §5.8/§5.9):
    /// decipher `data || MAC-I`, then verify the 32-bit MAC-I in constant time.
    /// Fail-closed: a MAC mismatch, a too-short payload or an unsupported
    /// cipher returns `Err` and the plaintext is never surfaced (TS 33.501
    /// §6.5: a PDU failing integrity is discarded).
    pub fn unprotect_srb(
        &self,
        count: u32,
        bearer: u8,
        direction: u8,
        protected: &[u8],
    ) -> Result<Vec<u8>, AsSecurityError> {
        if protected.len() < 4 {
            return Err(AsSecurityError::PduTooShort);
        }
        let mut buf = protected.to_vec();
        self.apply_ciphering(count, bearer, direction, &mut buf)?;
        let split = buf.len() - 4;
        let (data, mac_recv) = buf.split_at(split);
        let mac_calc = self.compute_rrc_mac_i(count, bearer, direction, data);
        if ct_eq_mac(&mac_calc, mac_recv) {
            Ok(data.to_vec())
        } else {
            Err(AsSecurityError::IntegrityCheckFailed)
        }
    }
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
            k_rrc_enc: [
                0xF0, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFB, 0xFC, 0xFD,
                0xFE, 0xFF,
            ],
            integrity_algorithm: alg,
            ciphering_algorithm: CipheringAlgorithm::Nea2,
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

    // ------------------------------------------------------------------------
    // Wave-6 I5 — KgNB→K_RRCint/K_RRCenc derivation + PDCP SRB enforcement
    // (TS 33.501 §6.5/§6.7 Annex A.8/D, TS 38.323 §5.8/§5.9). Golden literals
    // are frozen outputs of the KAT-anchored nextgsim-crypto NIA2/NEA2
    // primitives; each is cross-checked against the primitive directly, and a
    // roundtrip + fail-closed twin guards the enforcement path.
    // ------------------------------------------------------------------------

    /// Golden PDCP MAC-I: NIA2 (128-NIA2 = AES-CMAC, RFC 4493-anchored) over the
    /// RRC PDU [0x20,0x08,0x10] (the golden SecurityModeCommand) with
    /// K_RRCint = 0001..0F, COUNT=0, BEARER=0 (SRB1 = srb-id−1), DIRECTION=1
    /// (downlink).
    const GOLDEN_SMC_MAC_I_NIA2: [u8; 4] = [0xDD, 0x1B, 0x43, 0xA0];

    /// Golden NEA2-protected SRB payload: AES-CTR (128-NEA2) of `msg || MAC-I`
    /// with K_RRCenc = F0F1..FF, COUNT=1, BEARER=0, DIRECTION=0 (uplink),
    /// msg = [0x20,0x08,0x10].
    const GOLDEN_SRB_NEA2_UL_COUNT1: [u8; 7] = [0x0B, 0x91, 0x1F, 0x9B, 0x04, 0x58, 0x54];

    #[test]
    fn test_pdcp_mac_i_golden_nia2() {
        let ctx = test_ctx(IntegrityAlgorithm::Nia2);
        let msg = [0x20u8, 0x08, 0x10];
        let mac = ctx.compute_rrc_mac_i(0, SRB1_BEARER, DIRECTION_DOWNLINK, &msg);
        assert_eq!(mac, GOLDEN_SMC_MAC_I_NIA2);
        // Dual check: the PDCP layer delegates to the KAT-anchored NIA2
        // primitive with exactly these security inputs.
        assert_eq!(
            mac,
            nextgsim_crypto::nia::nia2_compute_mac(
                0,
                SRB1_BEARER,
                DIRECTION_DOWNLINK,
                &ctx.k_rrc_int,
                &msg
            )
        );
    }

    #[test]
    fn test_pdcp_protect_srb_golden_nea2() {
        let ctx = test_ctx(IntegrityAlgorithm::Nia2); // ciphering = NEA2
        let msg = [0x20u8, 0x08, 0x10];
        let prot = ctx
            .protect_srb(1, SRB1_BEARER, DIRECTION_UPLINK, &msg)
            .unwrap();
        assert_eq!(prot.as_slice(), GOLDEN_SRB_NEA2_UL_COUNT1);
        assert_eq!(
            prot.len(),
            msg.len() + 4,
            "SRB PDCP appends a 4-octet MAC-I"
        );
    }

    #[test]
    fn test_pdcp_srb_roundtrip_nea2_nia2() {
        let ctx = test_ctx(IntegrityAlgorithm::Nia2);
        let msg = b"\x2e\x01\x02\x03rrc-payload";
        let prot = ctx
            .protect_srb(7, SRB1_BEARER, DIRECTION_DOWNLINK, msg)
            .unwrap();
        let recovered = ctx
            .unprotect_srb(7, SRB1_BEARER, DIRECTION_DOWNLINK, &prot)
            .unwrap();
        assert_eq!(recovered, msg);
    }

    #[test]
    fn test_pdcp_srb_roundtrip_nea0_leaves_plaintext() {
        // NEA0 (the matched-sim ciphering choice): the RRC PDU stays in clear,
        // only the MAC-I is appended.
        let mut ctx = test_ctx(IntegrityAlgorithm::Nia2);
        ctx.ciphering_algorithm = CipheringAlgorithm::Nea0;
        let msg = [0x20u8, 0x08, 0x10];
        let prot = ctx
            .protect_srb(0, SRB1_BEARER, DIRECTION_DOWNLINK, &msg)
            .unwrap();
        assert_eq!(&prot[..msg.len()], &msg, "NEA0 leaves the RRC PDU in clear");
        assert_eq!(&prot[msg.len()..], &GOLDEN_SMC_MAC_I_NIA2);
        assert_eq!(
            ctx.unprotect_srb(0, SRB1_BEARER, DIRECTION_DOWNLINK, &prot)
                .unwrap(),
            msg
        );
    }

    #[test]
    fn test_pdcp_unprotect_fails_closed_on_tamper() {
        let ctx = test_ctx(IntegrityAlgorithm::Nia2);
        let msg = [0x20u8, 0x08, 0x10];
        let mut prot = ctx
            .protect_srb(0, SRB1_BEARER, DIRECTION_DOWNLINK, &msg)
            .unwrap();
        prot[0] ^= 0x01; // flip one ciphertext bit
        assert_eq!(
            ctx.unprotect_srb(0, SRB1_BEARER, DIRECTION_DOWNLINK, &prot),
            Err(AsSecurityError::IntegrityCheckFailed)
        );
    }

    #[test]
    fn test_pdcp_unprotect_wrong_count_fails_closed() {
        let ctx = test_ctx(IntegrityAlgorithm::Nia2);
        let msg = [0x20u8, 0x08, 0x10];
        let prot = ctx
            .protect_srb(5, SRB1_BEARER, DIRECTION_DOWNLINK, &msg)
            .unwrap();
        // Verifying at a different COUNT must fail (anti-replay at MAC level).
        assert!(ctx
            .unprotect_srb(6, SRB1_BEARER, DIRECTION_DOWNLINK, &prot)
            .is_err());
    }

    #[test]
    fn test_pdcp_too_short_fails_closed() {
        let ctx = test_ctx(IntegrityAlgorithm::Nia2);
        assert_eq!(
            ctx.unprotect_srb(0, SRB1_BEARER, DIRECTION_DOWNLINK, &[0x00, 0x01, 0x02]),
            Err(AsSecurityError::PduTooShort)
        );
    }

    #[test]
    fn test_pdcp_nea3_unsupported_fails_closed() {
        let mut ctx = test_ctx(IntegrityAlgorithm::Nia2);
        ctx.ciphering_algorithm = CipheringAlgorithm::Nea3;
        assert_eq!(
            ctx.protect_srb(0, SRB1_BEARER, DIRECTION_UPLINK, &[0x20, 0x08, 0x10]),
            Err(AsSecurityError::UnsupportedCipheringAlgorithm)
        );
    }

    #[test]
    fn test_derive_from_kgnb_matches_gnb_derivation() {
        // The UE's AS-key derivation MUST be byte-identical to the gNB's own
        // derive_rrc_up_key calls (nextgsim-gnb activate_as_security), so both
        // ends share the SRB keys (TS 33.501 Annex A.8).
        use nextgsim_crypto::kdf::{derive_rrc_up_key, AlgorithmTypeDistinguisher};
        let kgnb = [0x11u8; 32];
        let ctx = AsSecurityContext::derive_from_kgnb(
            &kgnb,
            CipheringAlgorithm::Nea0,
            IntegrityAlgorithm::Nia2,
            0x4601,
        );
        assert_eq!(
            ctx.k_rrc_int,
            derive_rrc_up_key(&kgnb, AlgorithmTypeDistinguisher::RrcInt, 2)
        );
        assert_eq!(
            ctx.k_rrc_enc,
            derive_rrc_up_key(&kgnb, AlgorithmTypeDistinguisher::RrcEnc, 0)
        );
    }

    #[test]
    fn test_algorithm_type_conversions() {
        assert_eq!(
            IntegrityAlgorithm::from(IntegrityAlgorithmType::Nia2),
            IntegrityAlgorithm::Nia2
        );
        assert_eq!(
            CipheringAlgorithm::from(CipheringAlgorithmType::Nea0),
            CipheringAlgorithm::Nea0
        );
        assert_eq!(
            CipheringAlgorithm::from(CipheringAlgorithmType::Nea3),
            CipheringAlgorithm::Nea3
        );
    }
}
