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
use nextgsim_pdcp::srb_security::{SrbSecurity, SrbSecurityError};
use nextgsim_rrc::codec::generated::{CellIdentity, PhysCellId, RNTI_Value, VarResumeMAC_Input};
use nextgsim_rrc::codec::{encode_rrc, RrcCodecError};
use nextgsim_rrc::procedures::rrc_reestablishment::{
    compute_short_mac_i as rrc_compute_short_mac_i, mac_i_lsb16 as rrc_mac_i_lsb16,
    RrcReestablishmentError,
};
use nextgsim_rrc::procedures::security_mode::{CipheringAlgorithmType, IntegrityAlgorithmType};
/// AS security activation is controlled by `UeConfig::as_security_enabled`
/// (issue #31), not by this constant.
///
/// This used to be `pub const I5_UE_AS_SECURITY: bool = false;` — a compile-time
/// gate, so **no shipping build could activate AS security at all**, whatever the
/// operator configured. Criterion 1 of #31 is exactly that: the gate must be
/// configuration, not a constant.
///
/// Kept as a named helper rather than an inline field read so the three call sites
/// share one predicate and a reader searching for the old constant lands here.
///
/// See `UeConfig::as_security_enabled` for why the default is `false` and what
/// enabling it requires.
#[must_use]
pub fn as_security_enabled(config: &nextgsim_common::config::UeConfig) -> bool {
    config.as_security_enabled
}

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
    /// KUPint — user-plane (DRB) integrity key (128 bits), derived from KgNB with
    /// `AlgorithmTypeDistinguisher::UpInt` (TS 33.501 Annex A.8, issue #32).
    pub k_up_int: [u8; 16],
    /// KUPenc — user-plane (DRB) ciphering key (128 bits), derived from KgNB with
    /// `AlgorithmTypeDistinguisher::UpEnc` (TS 33.501 Annex A.8, issue #32).
    pub k_up_enc: [u8; 16],
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
/// The PDCP COUNT the SecurityModeCommand is integrity-protected with.
///
/// Zero: AS security activation is the first protected PDU on SRB1, so the DL
/// PDCP COUNT is still 0 (TS 38.323 §5.9 takes COUNT from the PDCP entity, which
/// has just been keyed). Named rather than inlined because BOTH ends must use the
/// same value and a mismatch fails the MAC with no other symptom (issue #31).
pub const SMC_PDCP_COUNT: u32 = 0;

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
            // The UP keys use different distinguishers and therefore differ from the
            // RRC pair even though the algorithms are the same (issue #32). The gNB
            // makes the same four calls in `activate_as_security`, so both ends hold
            // identical keys without either signalling them.
            k_up_int: derive_rrc_up_key(
                kgnb,
                AlgorithmTypeDistinguisher::UpInt,
                integrity_algorithm.id(),
            ),
            k_up_enc: derive_rrc_up_key(
                kgnb,
                AlgorithmTypeDistinguisher::UpEnc,
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
        // Delegated so the gNB computes the SAME MAC: the layer lives in
        // `nextgsim-pdcp` because both ends must agree (issue #31). An all-zero MAC
        // for an unknown algorithm cannot arise here -- `id()` only yields 0..=3 --
        // and NIA0's MAC is legitimately all zeros anyway.
        match self.srb_security() {
            Ok(sec) => sec.compute_mac_i(count, bearer, direction, message),
            Err(_) => [0u8; 4],
        }
    }

    /// The shared SRB PDCP security state (TS 38.323 §5.8/§5.9).
    ///
    /// Built on demand rather than stored, so this context stays a plain
    /// key-and-algorithm record and there is no second copy of the keys to keep in
    /// step. The layer itself lives in `nextgsim-pdcp` because the **gNB verifies
    /// what this produces**, and two implementations of one MAC-and-cipher layout
    /// is a defect waiting to happen (issue #31).
    ///
    /// Fails only for an algorithm identity outside 0..=3, which cannot arise from
    /// the enums above — kept as a `Result` because the shared constructor is
    /// fail-closed and swallowing that here would defeat it.
    pub fn srb_security(&self) -> Result<SrbSecurity, AsSecurityError> {
        SrbSecurity::new(
            self.k_rrc_enc,
            self.k_rrc_int,
            self.ciphering_algorithm.id(),
            self.integrity_algorithm.id(),
        )
        .map_err(|_| AsSecurityError::UnsupportedCipheringAlgorithm)
    }

    /// PDCP-protect an SRB RRC message for transmission (TS 38.323 §5.8/§5.9).
    ///
    /// Delegates to the shared layer. **NEA3 works now**: this used to return
    /// `UnsupportedCipheringAlgorithm` with a comment claiming NEA3 "has no
    /// keystream in `nextgsim-crypto`", which was never true —
    /// `zuc::nea3_encrypt` is complete and `nextgsim-nas` has been using it for
    /// NAS ciphering all along (issue #31).
    pub fn protect_srb(
        &self,
        count: u32,
        bearer: u8,
        direction: u8,
        plaintext: &[u8],
    ) -> Result<Vec<u8>, AsSecurityError> {
        Ok(self
            .srb_security()?
            .protect(count, bearer, direction, plaintext))
    }

    /// PDCP-unprotect a received SRB PDCP payload (TS 38.323 §5.8/§5.9).
    ///
    /// Fail-closed: a MAC mismatch or a too-short payload returns `Err` and the
    /// plaintext is never surfaced (TS 33.501 §6.5).
    pub fn unprotect_srb(
        &self,
        count: u32,
        bearer: u8,
        direction: u8,
        protected: &[u8],
    ) -> Result<Vec<u8>, AsSecurityError> {
        self.srb_security()?
            .unprotect(count, bearer, direction, protected)
            .map_err(|e| match e {
                SrbSecurityError::PduTooShort => AsSecurityError::PduTooShort,
                SrbSecurityError::IntegrityCheckFailed => AsSecurityError::IntegrityCheckFailed,
                SrbSecurityError::UnknownAlgorithm(_) => {
                    AsSecurityError::UnsupportedCipheringAlgorithm
                }
            })
    }
}

/// Errors during ShortMAC-I derivation
#[derive(Debug, thiserror::Error)]
pub enum ShortMacError {
    /// VarShortMAC-Input encoding failed
    #[error("VarShortMAC-Input encoding error: {0}")]
    EncodeError(#[from] RrcCodecError),
    /// The shared derivation in `nextgsim-rrc` rejected the inputs
    #[error("ShortMAC-I derivation error: {0}")]
    DerivationError(#[from] RrcReestablishmentError),
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

    let _ = cell_id_bv;
    // The one implementation lives in nextgsim-rrc, the crate that owns
    // VarShortMAC-Input, because the UE sets this value and the network verifies
    // it -- two copies of the formula is a defect waiting to happen (issue #37).
    Ok(rrc_compute_short_mac_i(
        &ctx.k_rrc_int,
        ctx.integrity_algorithm.id(),
        ctx.c_rnti,
        source_pci,
        target_cell_identity,
    )?)
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

    let _ = (COUNT, BEARER, DIRECTION);
    // Delegated for the same reason as compute_short_mac_i: one formula. The
    // identity is always one of NIA0..NIA3 here, since `IntegrityAlgorithm` has
    // no other variant, so the shared function cannot reject it.
    rrc_mac_i_lsb16(&ctx.k_rrc_int, ctx.integrity_algorithm.id(), encoded)
        .expect("IntegrityAlgorithm::id() is always 0..=3")
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
            // Distinct from the RRC pair, as `derive_from_kgnb` produces them:
            // a test that reused the RRC keys here would not notice a UP path keyed
            // with the wrong one.
            k_up_int: [
                0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D,
                0x1E, 0x1F,
            ],
            k_up_enc: [
                0xE0, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xEB, 0xEC, 0xED,
                0xEE, 0xEF,
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

    /// #31, criterion 6: **NEA3 works.** This test used to be
    /// `test_pdcp_nea3_unsupported_fails_closed`, asserting that NEA3 was refused
    /// — it pinned the defect AS the requirement, on the strength of a comment
    /// claiming NEA3 "has no keystream in `nextgsim-crypto`".
    ///
    /// That premise was false: `zuc::nea3_encrypt` is a complete ZUC
    /// implementation and `nextgsim-nas` has used it for NAS ciphering all along.
    /// Replaced rather than deleted, because the algorithm it names is the point.
    #[test]
    fn test_pdcp_nea3_round_trips_and_ciphers() {
        let mut ctx = test_ctx(IntegrityAlgorithm::Nia2);
        ctx.ciphering_algorithm = CipheringAlgorithm::Nea3;
        let msg = [0x20u8, 0x08, 0x10];

        let protected = ctx
            .protect_srb(0, SRB1_BEARER, DIRECTION_UPLINK, &msg)
            .expect("NEA3 must protect, not fail closed");
        assert_eq!(protected.len(), msg.len() + 4, "the MAC-I must be appended");
        assert_ne!(
            &protected[..msg.len()],
            &msg[..],
            "NEA3 must actually cipher: an apply_ciphering that silently did \
             nothing would still round trip"
        );
        assert_eq!(
            ctx.unprotect_srb(0, SRB1_BEARER, DIRECTION_UPLINK, &protected)
                .expect("and must verify"),
            msg.to_vec()
        );
    }

    /// The whole point of hoisting the layer: what the UE protects is what the
    /// **gNB's** verifier accepts, because there is one implementation.
    #[test]
    fn the_ue_and_the_shared_gnb_layer_agree_on_every_algorithm() {
        use nextgsim_pdcp::srb_security::SrbSecurity;

        for (integ, ciph) in [
            (IntegrityAlgorithm::Nia1, CipheringAlgorithm::Nea1),
            (IntegrityAlgorithm::Nia2, CipheringAlgorithm::Nea2),
            (IntegrityAlgorithm::Nia3, CipheringAlgorithm::Nea3),
            (IntegrityAlgorithm::Nia2, CipheringAlgorithm::Nea0),
        ] {
            let mut ctx = test_ctx(integ);
            ctx.ciphering_algorithm = ciph;
            let msg = [0x20u8, 0x40, 0x00, 0x22];

            // The UE protects uplink; the gNB's layer, built from the SAME keys and
            // algorithm identities, must verify it.
            let uplink = ctx
                .protect_srb(3, SRB1_BEARER, DIRECTION_UPLINK, &msg)
                .expect("protect");
            let gnb = SrbSecurity::new(ctx.k_rrc_enc, ctx.k_rrc_int, ciph.id(), integ.id())
                .expect("the gNB builds the same state");
            assert_eq!(
                gnb.unprotect(3, SRB1_BEARER, DIRECTION_UPLINK, &uplink)
                    .unwrap_or_else(|e| panic!("{ciph:?}/{integ:?} must verify at the gNB: {e}")),
                msg.to_vec()
            );

            // And the reverse direction: the gNB protects downlink, the UE verifies.
            let downlink = gnb.protect(3, SRB1_BEARER, DIRECTION_DOWNLINK, &msg);
            assert_eq!(
                ctx.unprotect_srb(3, SRB1_BEARER, DIRECTION_DOWNLINK, &downlink)
                    .expect("must verify at the UE"),
                msg.to_vec()
            );
        }
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
