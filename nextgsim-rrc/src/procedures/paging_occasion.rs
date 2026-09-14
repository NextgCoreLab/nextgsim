//! Paging Frame and Paging Occasion derivation (3GPP TS 38.304 §7.1)
//!
//! A UE in RRC_IDLE or RRC_INACTIVE monitors **one** paging occasion per DRX
//! cycle, so the network must transmit its paging in that occasion and the UE
//! need only listen there. TS 38.304 §7.1 gives the two formulae verbatim:
//!
//! ```text
//! SFN for the PF:   (SFN + PF_offset) mod T = (T div N) * (UE_ID mod N)
//! PO index:         i_s = floor(UE_ID / N) mod Ns
//! ```
//!
//! with, from the same clause:
//!
//! | Term        | Meaning                                    |
//! |-------------|--------------------------------------------|
//! | `T`         | DRX cycle of the UE, in radio frames        |
//! | `N`         | number of total paging frames in `T`        |
//! | `Ns`        | number of paging occasions for a PF         |
//! | `PF_offset` | offset used for PF determination           |
//! | `UE_ID`     | `5G-S-TMSI mod 1024` (non-eDRX)            |
//!
//! This module is the *pure* half: no clock, no transmission, no state. The
//! frame counter it is evaluated against lives in
//! `nextgsim_common::frame_clock`, and the scheduling that uses both lives in
//! the gNB and UE RRC tasks.
//!
//! ## What is modelled and what is not
//!
//! The PF is a radio **frame**, and that is the granularity here. `i_s` selects
//! a PO *within* the frame, which in a real network means a set of PDCCH
//! monitoring occasions — sub-frame timing this simulator does not have. `i_s` is
//! therefore computed and carried (it is part of the spec's answer, and a
//! consumer that gains slot timing will need it) but it does not gate a
//! transmission. Said here rather than left for a reader to discover, because a
//! computed-and-unused value is the shape this tree keeps finding defects behind.

/// The DRX cycle values `PCCH-Config.defaultPagingCycle` can take
/// (TS 38.331: `rf32`, `rf64`, `rf128`, `rf256`), in radio frames.
pub const VALID_DRX_CYCLES: [u16; 4] = [32, 64, 128, 256];

/// The paging-occasion counts `PCCH-Config.ns` can take (TS 38.331: `four`,
/// `two`, `one`).
pub const VALID_NS: [u8; 3] = [1, 2, 4];

/// `UE_ID` is the 5G-S-TMSI modulo this (TS 38.304 §7.1, non-eDRX).
pub const UE_ID_MODULUS: u32 = 1024;

/// Why a paging-cycle configuration was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PagingCycleError {
    /// `T` is not one of the `defaultPagingCycle` values.
    InvalidDrxCycle(u16),
    /// `N` is zero, exceeds `T`, or does not divide `T`.
    ///
    /// TS 38.331 restricts `N` to `T`, `T/2`, `T/4`, `T/8` or `T/16` via
    /// `nAndPagingFrameOffset`, all of which divide `T`. A non-divisor would
    /// make `T div N` a truncating division and put the derived PF outside the
    /// residue class the UE computes, so the two ends would disagree about the
    /// occasion — silently, since each would be self-consistent.
    InvalidN { t: u16, n: u16 },
    /// `Ns` is not 1, 2 or 4.
    InvalidNs(u8),
    /// `PF_offset` is not smaller than `T`, so it cannot be an offset within the
    /// cycle.
    InvalidPfOffset { t: u16, pf_offset: u16 },
}

impl core::fmt::Display for PagingCycleError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidDrxCycle(t) => write!(
                f,
                "DRX cycle T={t} is not one of {VALID_DRX_CYCLES:?} radio frames"
            ),
            Self::InvalidN { t, n } => {
                write!(f, "N={n} must be non-zero, at most T={t}, and divide it")
            }
            Self::InvalidNs(ns) => write!(f, "Ns={ns} is not one of {VALID_NS:?}"),
            Self::InvalidPfOffset { t, pf_offset } => {
                write!(f, "PF_offset={pf_offset} must be less than T={t}")
            }
        }
    }
}

impl std::error::Error for PagingCycleError {}

/// A validated paging-cycle configuration: the `T`, `N`, `Ns` and `PF_offset` of
/// TS 38.304 §7.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagingCycleConfig {
    t: u16,
    n: u16,
    ns: u8,
    pf_offset: u16,
}

impl PagingCycleConfig {
    /// Validate a configuration.
    ///
    /// # Errors
    /// Returns [`PagingCycleError`] when a term is outside what TS 38.331's
    /// `PCCH-Config` can signal. Validated at construction rather than at use so
    /// an impossible configuration cannot produce a plausible-looking occasion.
    pub fn new(t: u16, n: u16, ns: u8, pf_offset: u16) -> Result<Self, PagingCycleError> {
        if !VALID_DRX_CYCLES.contains(&t) {
            return Err(PagingCycleError::InvalidDrxCycle(t));
        }
        if n == 0 || n > t || t % n != 0 {
            return Err(PagingCycleError::InvalidN { t, n });
        }
        if !VALID_NS.contains(&ns) {
            return Err(PagingCycleError::InvalidNs(ns));
        }
        if pf_offset >= t {
            return Err(PagingCycleError::InvalidPfOffset { t, pf_offset });
        }
        Ok(Self {
            t,
            n,
            ns,
            pf_offset,
        })
    }

    /// The configuration this simulator uses when SIB1 signals no `PCCH-Config`:
    /// `N = T` (one paging frame per UE per cycle, spread across the whole
    /// cycle), `Ns = 1` and no offset.
    ///
    /// `N = T` is the maximum spreading `nAndPagingFrameOffset` allows, which is
    /// what a simulator wants: two UEs share a paging frame only when their
    /// `UE_ID`s are congruent modulo `T`.
    ///
    /// # Errors
    /// Returns [`PagingCycleError`] when `t` is not a valid DRX cycle.
    pub fn with_default_spreading(t: u16) -> Result<Self, PagingCycleError> {
        Self::new(t, t, 1, 0)
    }

    /// DRX cycle in radio frames.
    pub fn t(&self) -> u16 {
        self.t
    }

    /// Number of paging frames in the cycle.
    pub fn n(&self) -> u16 {
        self.n
    }

    /// Number of paging occasions per paging frame.
    pub fn ns(&self) -> u8 {
        self.ns
    }

    /// Paging-frame offset.
    pub fn pf_offset(&self) -> u16 {
        self.pf_offset
    }
}

/// A UE's paging occasion (TS 38.304 §7.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagingOccasion {
    /// Every SFN congruent to this modulo `T` is one of this UE's paging frames.
    ///
    /// The spec states the PF as a congruence rather than a single frame, and so
    /// does this: over the 1024-frame SFN cycle a UE has `1024 / T` paging
    /// frames, and collapsing them to one would make paging work in a tenth of
    /// the cycle and silently fail in the rest.
    pf_residue: u16,
    /// The DRX cycle the residue is modulo.
    t: u16,
    /// Index of the paging occasion within the frame (`i_s`).
    ///
    /// Carried, not enforced: selecting a PO within a frame needs sub-frame
    /// timing this simulator does not model. See the module documentation.
    i_s: u8,
}

impl PagingOccasion {
    /// The paging-frame residue: an SFN is a paging frame when
    /// `sfn mod T == pf_residue`.
    pub fn pf_residue(&self) -> u16 {
        self.pf_residue
    }

    /// The `i_s` index of TS 38.304 §7.1.
    pub fn i_s(&self) -> u8 {
        self.i_s
    }

    /// The DRX cycle in radio frames.
    pub fn t(&self) -> u16 {
        self.t
    }

    /// Whether `sfn` is one of this UE's paging frames.
    pub fn is_paging_frame(&self, sfn: u16) -> bool {
        sfn % self.t == self.pf_residue
    }

    /// The first paging frame at or after `sfn`, as an SFN in `0..1024`.
    ///
    /// "At or after" and not "after": a caller already inside the occasion should
    /// transmit now rather than wait a whole cycle.
    pub fn next_paging_frame_at_or_after(&self, sfn: u16) -> u16 {
        let cycle = crate::procedures::paging_occasion::SFN_CYCLE;
        let sfn = sfn % cycle;
        let current_residue = sfn % self.t;
        let frames_ahead = (self.pf_residue + self.t - current_residue) % self.t;
        (sfn + frames_ahead) % cycle
    }

    /// Whether `sfn` is within `tolerance_frames` of one of this UE's paging
    /// frames.
    ///
    /// A tolerance exists because the PDU crosses a UDP transport between the two
    /// clocks: the gNB transmits *in* the frame and the UE reads it a few
    /// milliseconds later. Zero tolerance would drop conformant paging whenever
    /// the send straddled a frame boundary; a tolerance of `T` would accept
    /// everything and make the check decorative.
    pub fn is_within_occasion(&self, sfn: u16, tolerance_frames: u16) -> bool {
        if tolerance_frames >= self.t {
            // Refuse to pretend: a tolerance at or beyond the cycle admits every
            // frame, so the caller has asked for no check at all.
            return true;
        }
        (0..=tolerance_frames).any(|back| {
            let candidate = (sfn + crate::procedures::paging_occasion::SFN_CYCLE - back)
                % crate::procedures::paging_occasion::SFN_CYCLE;
            self.is_paging_frame(candidate)
        })
    }
}

/// The SFN cycle length, duplicated from `nextgsim_common::frame_clock` so this
/// crate's pure derivation does not depend on the clock it is evaluated against.
pub const SFN_CYCLE: u16 = 1024;

/// `UE_ID` for paging: the 5G-S-TMSI modulo 1024 (TS 38.304 §7.1, non-eDRX).
///
/// The 6 octets are the TS 23.003 §2.10.1 form (AMF Set ID + AMF Pointer, then
/// the 32-bit 5G-TMSI) read as a big-endian 48-bit integer, which is the order
/// they travel in.
pub fn ue_id_from_s_tmsi(s_tmsi: &[u8; 6]) -> u16 {
    let value = s_tmsi
        .iter()
        .fold(0u64, |acc, &octet| (acc << 8) | u64::from(octet));
    (value % u64::from(UE_ID_MODULUS)) as u16
}

/// Derive a UE's paging occasion (TS 38.304 §7.1).
///
/// Rearranged from the spec's congruence rather than solved numerically: the
/// formula fixes `(SFN + PF_offset) mod T`, so the SFN residue is that value
/// minus `PF_offset`, taken modulo `T`.
pub fn paging_occasion(ue_id: u16, config: &PagingCycleConfig) -> PagingOccasion {
    let t = u32::from(config.t());
    let n = u32::from(config.n());
    let ue_id_u32 = u32::from(ue_id);

    // (T div N) * (UE_ID mod N) -- the right-hand side of the PF congruence.
    let target = (t / n) * (ue_id_u32 % n);
    // (SFN + PF_offset) mod T = target  =>  SFN mod T = (target - PF_offset) mod T
    let pf_residue = ((target + t - (u32::from(config.pf_offset()) % t)) % t) as u16;

    // i_s = floor(UE_ID / N) mod Ns
    let i_s = ((ue_id_u32 / n) % u32::from(config.ns())) as u8;

    PagingOccasion {
        pf_residue,
        t: config.t(),
        i_s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The spec's own relation, asserted directly rather than against the
    /// implementation's output: for every derived paging frame,
    /// `(SFN + PF_offset) mod T` must equal `(T div N) * (UE_ID mod N)`.
    fn assert_spec_pf_relation(ue_id: u16, config: &PagingCycleConfig) {
        let occasion = paging_occasion(ue_id, config);
        let expected = (config.t() / config.n()) * (ue_id % config.n());
        // Check every paging frame in the SFN cycle, not just the first: the
        // congruence has to hold across the wrap too.
        let mut checked = 0;
        for sfn in 0..SFN_CYCLE {
            if occasion.is_paging_frame(sfn) {
                assert_eq!(
                    (sfn + config.pf_offset()) % config.t(),
                    expected,
                    "TS 38.304 §7.1 PF relation failed at SFN {sfn} for UE_ID {ue_id}"
                );
                checked += 1;
            }
        }
        assert_eq!(
            checked,
            SFN_CYCLE / config.t(),
            "a UE must have exactly 1024/T paging frames per SFN cycle"
        );
    }

    #[test]
    fn the_spec_pf_relation_holds_for_every_valid_configuration() {
        for t in VALID_DRX_CYCLES {
            for divisor in [1u16, 2, 4, 8, 16] {
                let n = t / divisor;
                for ns in VALID_NS {
                    let config = PagingCycleConfig::new(t, n, ns, 0).expect("valid config");
                    // A spread of identities, including the boundaries.
                    for ue_id in [0u16, 1, 7, 255, 512, 1000, 1023] {
                        assert_spec_pf_relation(ue_id, &config);
                    }
                }
            }
        }
    }

    #[test]
    fn the_spec_pf_relation_holds_with_a_paging_frame_offset() {
        for pf_offset in [0u16, 1, 7, 31] {
            let config = PagingCycleConfig::new(32, 32, 1, pf_offset).expect("valid config");
            for ue_id in [0u16, 5, 31, 100, 1023] {
                assert_spec_pf_relation(ue_id, &config);
            }
        }
    }

    #[test]
    fn the_spec_i_s_relation_holds() {
        for t in VALID_DRX_CYCLES {
            for ns in VALID_NS {
                let config = PagingCycleConfig::new(t, t / 2, ns, 0).expect("valid config");
                for ue_id in [0u16, 3, 64, 511, 1023] {
                    let occasion = paging_occasion(ue_id, &config);
                    let expected = ((ue_id / config.n()) % u16::from(config.ns())) as u8;
                    assert_eq!(
                        occasion.i_s(),
                        expected,
                        "TS 38.304 §7.1 i_s relation failed for UE_ID {ue_id}"
                    );
                }
            }
        }
    }

    #[test]
    fn different_identities_land_on_different_frames_under_full_spreading() {
        // The point of N = T: paging load spreads across the cycle instead of
        // every UE waking in the same frame.
        let config = PagingCycleConfig::with_default_spreading(32).expect("valid");
        let residues: std::collections::HashSet<u16> = (0..32u16)
            .map(|ue_id| paging_occasion(ue_id, &config).pf_residue())
            .collect();
        assert_eq!(
            residues.len(),
            32,
            "32 consecutive identities must occupy 32 distinct paging frames"
        );
    }

    #[test]
    fn ue_id_is_the_low_ten_bits_of_the_s_tmsi() {
        // 5G-S-TMSI read big-endian, modulo 1024 -- i.e. the low 10 bits.
        assert_eq!(ue_id_from_s_tmsi(&[0, 0, 0, 0, 0, 0]), 0);
        assert_eq!(ue_id_from_s_tmsi(&[0, 0, 0, 0, 0, 1]), 1);
        assert_eq!(ue_id_from_s_tmsi(&[0, 0, 0, 0, 0x03, 0xFF]), 1023);
        // Bit 10 and above are discarded by the modulus.
        assert_eq!(ue_id_from_s_tmsi(&[0, 0, 0, 0, 0x04, 0x00]), 0);
        assert_eq!(
            ue_id_from_s_tmsi(&[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]),
            1023
        );
        // The high octets must not be ignored -- reading only the last two would
        // make these two identities equal.
        assert_eq!(
            ue_id_from_s_tmsi(&[0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC]),
            (0x123456789ABCu64 % 1024) as u16
        );
    }

    #[test]
    fn the_next_paging_frame_includes_the_current_one() {
        let config = PagingCycleConfig::with_default_spreading(32).expect("valid");
        let occasion = paging_occasion(5, &config);
        let residue = occasion.pf_residue();

        // Already at the occasion: now, not a cycle from now.
        assert_eq!(occasion.next_paging_frame_at_or_after(residue), residue);
        // One frame past it: the next one is T frames on from this residue.
        assert_eq!(
            occasion.next_paging_frame_at_or_after(residue + 1),
            residue + config.t()
        );
    }

    #[test]
    fn the_next_paging_frame_wraps_the_sfn_cycle() {
        let config = PagingCycleConfig::with_default_spreading(32).expect("valid");
        let occasion = paging_occasion(1, &config);
        // From late in the cycle, the answer must be a valid SFN and a real
        // paging frame -- a wrap computed with a plain addition would exceed 1023.
        let next = occasion.next_paging_frame_at_or_after(1020);
        assert!(next < SFN_CYCLE, "SFN {next} is outside 0..1024");
        assert!(occasion.is_paging_frame(next));
    }

    #[test]
    fn the_occasion_tolerance_admits_a_late_pdu_but_not_a_wrong_frame() {
        let config = PagingCycleConfig::with_default_spreading(32).expect("valid");
        let occasion = paging_occasion(5, &config);
        let pf = occasion.pf_residue();

        assert!(occasion.is_within_occasion(pf, 0), "the frame itself");
        assert!(occasion.is_within_occasion(pf + 2, 2), "two frames late");
        assert!(
            !occasion.is_within_occasion(pf + 3, 2),
            "three frames late must fail a two-frame tolerance"
        );
        // EARLY is not tolerated: the gNB transmits in the frame, so a PDU that
        // arrives before it did not come from this occasion.
        assert!(!occasion.is_within_occasion(pf.wrapping_sub(1), 2));
    }

    #[test]
    fn the_tolerance_check_works_across_the_sfn_wrap() {
        let config = PagingCycleConfig::with_default_spreading(32).expect("valid");
        // UE_ID 0 with full spreading has residue 0, so SFN 0 is a paging frame
        // and SFN 1023 is one frame early -- i.e. two frames before SFN 1 wraps.
        let occasion = paging_occasion(0, &config);
        assert!(occasion.is_paging_frame(0));
        assert!(
            occasion.is_within_occasion(1, 2),
            "one frame after SFN 0 must still be inside a two-frame tolerance"
        );
    }

    #[test]
    fn an_impossible_configuration_is_rejected_rather_than_producing_an_occasion() {
        // T outside defaultPagingCycle.
        assert_eq!(
            PagingCycleConfig::new(100, 100, 1, 0),
            Err(PagingCycleError::InvalidDrxCycle(100))
        );
        // N not dividing T: T div N truncates and the two ends would disagree.
        assert_eq!(
            PagingCycleConfig::new(32, 7, 1, 0),
            Err(PagingCycleError::InvalidN { t: 32, n: 7 })
        );
        // N zero, and N above T.
        assert!(PagingCycleConfig::new(32, 0, 1, 0).is_err());
        assert!(PagingCycleConfig::new(32, 64, 1, 0).is_err());
        // Ns outside {1,2,4}.
        assert_eq!(
            PagingCycleConfig::new(32, 32, 3, 0),
            Err(PagingCycleError::InvalidNs(3))
        );
        // An offset that is not inside the cycle.
        assert_eq!(
            PagingCycleConfig::new(32, 32, 1, 32),
            Err(PagingCycleError::InvalidPfOffset {
                t: 32,
                pf_offset: 32
            })
        );
    }

    #[test]
    fn a_tolerance_at_or_beyond_the_cycle_is_reported_as_no_check() {
        let config = PagingCycleConfig::with_default_spreading(32).expect("valid");
        let occasion = paging_occasion(5, &config);
        // Every frame passes, which is what the caller asked for -- and the
        // function says so by construction rather than by looping 32 times.
        for sfn in 0..64u16 {
            assert!(occasion.is_within_occasion(sfn, 32));
        }
    }
}
