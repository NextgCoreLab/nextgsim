//! RRC UE Context Management
//!
//! This module manages UE contexts within the RRC task. Each UE has an associated
//! context that tracks:
//! - UE identity (initial ID, S-TMSI)
//! - RRC establishment cause
//! - RRC connection state
//!
//! # Reference
//!
//! Based on UERANSIM's `RrcUeContext` from `src/gnb/types.hpp` and
//! UE management functions from `src/gnb/rrc/ues.cpp`.

use std::collections::HashMap;

use super::redcap::RedCapProcessor;
use super::transaction::RrcTransactionAllocator;
use crate::tasks::GutiMobileIdentity;
use nextgsim_pdcp::srb_security::SrbSecurity;
use nextgsim_rrc::procedures::suspend_config::{short_i_rnti_of, RanNotificationArea};

/// RRC connection state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RrcState {
    /// Initial state - no RRC connection
    #[default]
    Idle,
    /// RRC Setup Request received, waiting for setup
    SetupRequest,
    /// RRC Setup sent, waiting for completion
    SetupSent,
    /// RRC connection established
    Connected,
    /// RRC connection being released
    Releasing,
}

impl std::fmt::Display for RrcState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RrcState::Idle => write!(f, "Idle"),
            RrcState::SetupRequest => write!(f, "SetupRequest"),
            RrcState::SetupSent => write!(f, "SetupSent"),
            RrcState::Connected => write!(f, "Connected"),
            RrcState::Releasing => write!(f, "Releasing"),
        }
    }
}

/// What the RRC task needs of the AS security context to verify a
/// re-establishment (TS 38.331 §5.3.7.2).
///
/// The full AS context lives on the NGAP task, which derives it from the
/// `SecurityKey` of the InitialContextSetupRequest. The RRC task is handed this
/// subset over `RrcMessage::AsSecurityForReestablishment` because verifying a
/// `shortMAC-I` needs `K_RRCint`, and answering with an `RRCReestablishment`
/// needs the `nextHopChainingCount` — neither of which the RRC task could reach
/// before.
#[derive(Clone)]
pub struct ReestablishmentSecurity {
    /// K_RRCint (128-bit) of this UE's source PCell — what the `shortMAC-I` and
    /// the SRB PDCP MAC-I are computed with.
    pub k_rrc_int: [u8; 16],
    /// K_RRCenc (128-bit) — SRB PDCP ciphering (issue #31).
    ///
    /// Added alongside `k_rrc_int` because the RRC plane is where SRB1 PDUs are
    /// sent and received, so it is the plane that has to cipher and decipher them.
    /// Before this the RRC task could verify a `shortMAC-I` and could not protect a
    /// single PDU.
    pub k_rrc_enc: [u8; 16],
    /// Selected NR integrity algorithm identity (0 = NIA0 … 3 = NIA3). Must be
    /// the same identity the UE used, or the MAC-I will not match.
    pub integrity_alg_id: u8,
    /// Selected NR ciphering algorithm identity (0 = NEA0 … 3 = NEA3). Must be the
    /// same identity the UE used, or the keystreams diverge and every PDU fails
    /// integrity for the wrong reason.
    pub ciphering_alg_id: u8,
    /// The C-RNTI the UE will present in an `RRCReestablishmentRequest`.
    ///
    /// This simulator has no MAC layer and therefore no C-RNTI allocation, so
    /// both ends use the same well-known constant. The consequence, stated here
    /// because it matters: `(C-RNTI, PCI)` does **not** distinguish two UEs on
    /// one cell, so the lookup returns *candidates* and the `shortMAC-I`
    /// verification is what resolves them — which it can, because each UE has a
    /// different `K_RRCint`.
    pub c_rnti: u16,
    /// Physical cell identity of the PCell this context belongs to.
    pub phys_cell_id: u16,
    /// `nextHopChainingCount` of the current AS security context (0-7).
    ///
    /// TS 33.501 §6.8.2.1.1: the initial `KgNB` established at Initial Context
    /// Setup has NCC 0, and a fresh {NH, NCC} pair arrives later in a Path Switch
    /// Request Acknowledge. That path is not wired (issue #39), so this stays 0
    /// in a live run — which is the spec's own initial value, not a placeholder.
    pub next_hop_chaining_count: u8,
}

impl std::fmt::Debug for ReestablishmentSecurity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never print key material.
        f.debug_struct("ReestablishmentSecurity")
            .field("integrity_alg_id", &self.integrity_alg_id)
            .field("c_rnti", &self.c_rnti)
            .field("phys_cell_id", &self.phys_cell_id)
            .field("next_hop_chaining_count", &self.next_hop_chaining_count)
            .finish_non_exhaustive()
    }
}

/// RRC UE context
///
/// Tracks RRC-specific information for a UE, including identity and connection state.
/// Based on UERANSIM's `RrcUeContext` from `src/gnb/types.hpp`.
#[derive(Debug, Clone)]
pub struct RrcUeContext {
    /// UE ID (internal identifier)
    pub ue_id: i32,
    /// Initial UE identity (39-bit value, or None if not set)
    /// This is either a random value or TMSI-part-1 from 5G-S-TMSI
    pub initial_id: Option<i64>,
    /// Whether the initial ID is from S-TMSI (true) or random (false)
    pub is_initial_id_s_tmsi: bool,
    /// RRC establishment cause (from `RRCSetupRequest`)
    pub establishment_cause: i64,
    /// S-TMSI if available (from `RRCSetupComplete`)
    pub s_tmsi: Option<GutiMobileIdentity>,
    /// Current RRC connection state
    pub state: RrcState,
    /// RedCap processor for this UE (Rel-17)
    pub redcap: RedCapProcessor,
    /// UPER-encoded UE-NR-Capability container from UECapabilityInformation
    pub nr_capability: Option<Vec<u8>>,
    /// Per-UE RRC transaction-identifier allocator (TS 38.331 §6.3.2, Wave-6
    /// C4-final). Cycles 0..3 for THIS UE and records the outstanding tid per
    /// procedure so the gNB can verify the UE's echoed tid — replaces the
    /// former gNB-global counter.
    pub transactions: RrcTransactionAllocator,
    /// AS security material for verifying an `RRCReestablishmentRequest`
    /// (TS 38.331 §5.3.7.2). `None` until AS security is activated, and a
    /// re-establishment then falls back to `RRCSetup` — a context the network
    /// cannot verify is a context it must not restore.
    pub reestablishment_security: Option<ReestablishmentSecurity>,
    /// Whether AS security is ACTIVE for this UE, i.e. the UE has answered the
    /// SecurityModeCommand with a SecurityModeComplete (TS 38.331 §5.3.4.3,
    /// issue #31).
    ///
    /// Distinct from `reestablishment_security.is_some()`, which only means the
    /// keys arrived. The gNB must not protect a PDU before the UE has confirmed it
    /// can verify one, and must not accept a protected PDU before it has sent the
    /// command — so this is the state that gates both directions.
    as_security_active: bool,
    /// The next PDCP COUNT for a DOWNLINK protected SRB1 PDU.
    ///
    /// Starts at 1 because the SecurityModeCommand consumed COUNT 0
    /// (`SMC_PDCP_COUNT`). Both ends count the same PDUs in the same order, which
    /// is what lets them agree without signalling the COUNT — see the ceiling note
    /// in the spec.
    dl_pdcp_count: u32,
    /// The next PDCP COUNT expected on an UPLINK protected SRB1 PDU.
    ///
    /// Starts at 0: the UE's first protected uplink PDU is the
    /// SecurityModeComplete.
    ul_pdcp_count: u32,
}

impl RrcUeContext {
    /// Creates a new RRC UE context with the given UE ID
    pub fn new(ue_id: i32) -> Self {
        Self {
            ue_id,
            initial_id: None,
            is_initial_id_s_tmsi: false,
            establishment_cause: 0,
            s_tmsi: None,
            state: RrcState::Idle,
            redcap: RedCapProcessor::new(),
            nr_capability: None,
            transactions: RrcTransactionAllocator::new(),
            reestablishment_security: None,
            as_security_active: false,
            dl_pdcp_count: 1,
            ul_pdcp_count: 0,
        }
    }

    /// Records the AS security material a re-establishment is verified against,
    /// and which SRB PDCP protection uses (issue #31).
    pub fn set_reestablishment_security(&mut self, security: ReestablishmentSecurity) {
        self.reestablishment_security = Some(security);
    }

    /// Marks AS security ACTIVE, on the UE's SecurityModeComplete
    /// (TS 38.331 §5.3.4.3).
    pub fn on_as_security_activated(&mut self) {
        self.as_security_active = true;
    }

    /// Whether AS security is active for this UE.
    pub fn as_security_active(&self) -> bool {
        self.as_security_active
    }

    /// The shared SRB PDCP security state, once the keys have arrived.
    ///
    /// `None` before the NGAP plane hands over the keys, and `None` for an
    /// algorithm identity the shared layer refuses — fail-closed in both cases, so
    /// a caller cannot protect a PDU with a half-built state.
    pub fn srb_security(&self) -> Option<SrbSecurity> {
        let s = self.reestablishment_security.as_ref()?;
        SrbSecurity::new(
            s.k_rrc_enc,
            s.k_rrc_int,
            s.ciphering_alg_id,
            s.integrity_alg_id,
        )
        .ok()
    }

    /// Takes the next downlink PDCP COUNT.
    pub fn next_dl_pdcp_count(&mut self) -> u32 {
        let c = self.dl_pdcp_count;
        self.dl_pdcp_count = self.dl_pdcp_count.wrapping_add(1);
        c
    }

    /// Takes the next expected uplink PDCP COUNT.
    pub fn next_ul_pdcp_count(&mut self) -> u32 {
        let c = self.ul_pdcp_count;
        self.ul_pdcp_count = self.ul_pdcp_count.wrapping_add(1);
        c
    }

    /// Stores the UE-NR-Capability container received in UECapabilityInformation
    pub fn set_nr_capability(&mut self, container: Vec<u8>) {
        self.nr_capability = Some(container);
    }

    /// Sets the initial UE identity
    ///
    /// # Arguments
    /// * `initial_id` - 39-bit initial UE identity
    /// * `is_s_tmsi` - true if the ID is from S-TMSI, false if random
    pub fn set_initial_id(&mut self, initial_id: i64, is_s_tmsi: bool) {
        self.initial_id = Some(initial_id);
        self.is_initial_id_s_tmsi = is_s_tmsi;
    }

    /// Sets the RRC establishment cause
    pub fn set_establishment_cause(&mut self, cause: i64) {
        self.establishment_cause = cause;
    }

    /// Sets the S-TMSI from `RRCSetupComplete`
    pub fn set_s_tmsi(&mut self, s_tmsi: GutiMobileIdentity) {
        self.s_tmsi = Some(s_tmsi);
    }

    /// Transitions to `SetupRequest` state (`RRCSetupRequest` received)
    pub fn on_setup_request(&mut self) {
        self.state = RrcState::SetupRequest;
    }

    /// Transitions to `SetupSent` state (`RRCSetup` sent)
    pub fn on_setup_sent(&mut self) {
        self.state = RrcState::SetupSent;
    }

    /// Transitions to Connected state (`RRCSetupComplete` received)
    pub fn on_setup_complete(&mut self) {
        self.state = RrcState::Connected;
    }

    /// Transitions to Releasing state (`RRCRelease` being sent)
    pub fn on_release(&mut self) {
        self.state = RrcState::Releasing;
    }

    /// Returns true if the UE has an established RRC connection
    pub fn is_connected(&self) -> bool {
        self.state == RrcState::Connected
    }

    /// Returns true if the UE is in idle state
    pub fn is_idle(&self) -> bool {
        self.state == RrcState::Idle
    }
}

/// The identity an `RRCResumeRequest` presents (TS 38.331 §5.3.13.3).
///
/// Two forms of one identity, not two identities — see
/// [`RrcUeContextManager::find_suspended`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuspendedIdentity {
    /// Short 24-bit I-RNTI, from `RRCResumeRequest` on UL-CCCH.
    Short(u32),
    /// Full 40-bit I-RNTI, from `RRCResumeRequest1` on UL-CCCH1.
    Full(u64),
}

/// A UE context retained across suspension to RRC_INACTIVE (TS 38.331 §5.3.8.3).
///
/// This is what makes RRC_INACTIVE a *state* rather than a message: before issue #38
/// the gNB called `delete_ue` on every release, so an `RRCResumeRequest` had nothing
/// to be resolved against and the resume handler fabricated a fresh context —
/// accepting, unauthenticated, any UE that asked.
///
/// It deliberately carries no `Instant`: nothing here expires on the gNB's clock.
/// The UE's `t380` governs when it comes back, and a gNB-side timeout would discard a
/// context the UE still believes in, turning a resume into a silent setup fallback.
#[derive(Debug, Clone)]
pub struct SuspendedUeContext {
    /// The `ue_id` the UE had while connected. Retained for logging and for
    /// re-keying the restored context; the *lookup* key is the I-RNTI.
    pub previous_ue_id: i32,
    /// Full 40-bit I-RNTI, the key this context is stored under.
    pub full_i_rnti: u64,
    /// The AS security context of the source PCell, which is what lets the gNB
    /// recompute the `resumeMAC-I` the UE will present (TS 38.331 §5.3.13.3).
    ///
    /// Carries `c_rnti` and `phys_cell_id` too, and those are exactly the values the
    /// MAC covers — taking them from the *resuming* cell instead would fail every
    /// verification.
    pub security: ReestablishmentSecurity,
    /// The RAN Notification Area the UE was given, so the gNB can tell whether an
    /// arriving RNAU was one it configured (TS 38.304 §5.5).
    pub ran_notification_area: Option<RanNotificationArea>,
    /// `t380` in minutes, as signalled. Recorded, not enforced — see the type docs.
    pub t380_minutes: Option<u16>,
}

// NOTE on what is deliberately NOT stored: the UE's radio bearer configuration.
// `fresh_rrc_resume_params` gives a resumed UE the SRB1 configuration `RRCSetup`
// builds with `fullConfig` set, which tells it to discard and rebuild rather than
// merge a delta (issue #107 recorded that decision). Storing a configuration here
// would only be worth it to send a delta, and the RRC context does not track DRBs at
// all -- so the honest thing is to keep sending `fullConfig` and say so.

/// RRC UE context manager
///
/// Manages all UE contexts within the RRC task. Provides methods to create,
/// find, and delete UE contexts, plus the I-RNTI-keyed store of UEs suspended to
/// RRC_INACTIVE (issue #38).
///
/// Based on UERANSIM's UE management from `src/gnb/rrc/ues.cpp`.
#[derive(Debug, Default)]
pub struct RrcUeContextManager {
    /// UE contexts indexed by UE ID
    contexts: HashMap<i32, RrcUeContext>,
    /// Suspended UE contexts, keyed by **full** I-RNTI (issue #38).
    ///
    /// A separate map, not a state flag on `contexts`, because a suspended UE has no
    /// `ue_id` the network can use: the transport-level id belonged to the RRC
    /// connection that has just gone away, and the next `RRCResumeRequest` may arrive
    /// under a different one. The I-RNTI is the only identity that survives
    /// suspension (TS 38.331 §5.3.8.3), so it has to be the key.
    suspended: HashMap<u64, SuspendedUeContext>,
}

impl RrcUeContextManager {
    /// Creates a new empty UE context manager
    pub fn new() -> Self {
        Self {
            contexts: HashMap::new(),
            suspended: HashMap::new(),
        }
    }

    /// Creates a new UE context with the given ID
    ///
    /// Returns a mutable reference to the created context.
    /// If a context with the same ID already exists, it will be replaced.
    pub fn create_ue(&mut self, ue_id: i32) -> &mut RrcUeContext {
        let ctx = RrcUeContext::new(ue_id);
        self.contexts.insert(ue_id, ctx);
        self.contexts.get_mut(&ue_id).expect("value expected")
    }

    /// Tries to find a UE context by ID
    ///
    /// Returns `Some(&RrcUeContext)` if found, `None` otherwise.
    pub fn try_find_ue(&self, ue_id: i32) -> Option<&RrcUeContext> {
        self.contexts.get(&ue_id)
    }

    /// Tries to find a mutable UE context by ID
    ///
    /// Returns `Some(&mut RrcUeContext)` if found, `None` otherwise.
    pub fn try_find_ue_mut(&mut self, ue_id: i32) -> Option<&mut RrcUeContext> {
        self.contexts.get_mut(&ue_id)
    }

    /// Finds a UE context by ID, creating it if it doesn't exist
    ///
    /// This is useful when receiving messages from a UE that may or may not
    /// have an existing context.
    pub fn find_or_create_ue(&mut self, ue_id: i32) -> &mut RrcUeContext {
        if !self.contexts.contains_key(&ue_id) {
            self.create_ue(ue_id);
        }
        self.contexts.get_mut(&ue_id).expect("value expected")
    }

    /// Deletes a UE context by ID
    ///
    /// Returns the removed context if it existed.
    pub fn delete_ue(&mut self, ue_id: i32) -> Option<RrcUeContext> {
        self.contexts.remove(&ue_id)
    }

    /// Returns the number of UE contexts
    pub fn count(&self) -> usize {
        self.contexts.len()
    }

    /// Returns true if there are no UE contexts
    pub fn is_empty(&self) -> bool {
        self.contexts.is_empty()
    }

    /// Returns an iterator over all UE contexts
    pub fn iter(&self) -> impl Iterator<Item = (&i32, &RrcUeContext)> {
        self.contexts.iter()
    }

    /// Returns a mutable iterator over all UE contexts
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&i32, &mut RrcUeContext)> {
        self.contexts.iter_mut()
    }

    /// Returns all UE IDs
    pub fn ue_ids(&self) -> Vec<i32> {
        self.contexts.keys().copied().collect()
    }

    /// Returns all connected UE IDs
    pub fn connected_ue_ids(&self) -> Vec<i32> {
        self.contexts
            .iter()
            .filter(|(_, ctx)| ctx.is_connected())
            .map(|(id, _)| *id)
            .collect()
    }

    /// Suspend a connected UE to RRC_INACTIVE: move its context out of the active
    /// map and into the I-RNTI-keyed suspended store (TS 38.331 §5.3.8.3).
    ///
    /// `security` is passed in rather than read from the context here, and that is
    /// deliberate: the caller must already have it to put the `nextHopChainingCount`
    /// in the `suspendConfig`, so reading it again would make **two** places decide
    /// whether a UE may be suspended. A revert round proved the second one could not
    /// fail — breaking it changed nothing, because the caller had already refused —
    /// and a guard that cannot fail is worse than none: it reads as protection.
    ///
    /// Refusing a UE with no AS security context is therefore the *caller's* job, and
    /// [`RrcConnectionManager::initiate_rrc_suspend`](crate::rrc::connection::RrcConnectionManager::initiate_rrc_suspend)
    /// does it — without `K_RRCint` the gNB could never verify the `resumeMAC-I`, so
    /// suspending would create exactly the unauthenticatable resume issue #38 reports.
    ///
    /// Returns `None` when the UE is unknown.
    pub fn suspend_ue(
        &mut self,
        ue_id: i32,
        full_i_rnti: u64,
        security: ReestablishmentSecurity,
        ran_notification_area: Option<RanNotificationArea>,
        t380_minutes: Option<u16>,
    ) -> Option<&SuspendedUeContext> {
        self.contexts.get(&ue_id)?;
        // Removed from the active map, not merely flagged: the UE is no longer
        // reachable by `ue_id`, and leaving it there would let a downlink send path
        // address a UE that is not listening.
        self.contexts.remove(&ue_id);
        self.suspended.insert(
            full_i_rnti,
            SuspendedUeContext {
                previous_ue_id: ue_id,
                full_i_rnti,
                security,
                ran_notification_area,
                t380_minutes,
            },
        );
        self.suspended.get(&full_i_rnti)
    }

    /// Look a suspended context up by the identity an `RRCResumeRequest` presented.
    ///
    /// Both forms resolve to the same context: the short I-RNTI is the low 24 bits of
    /// the full one ([`short_i_rnti_of`]), and which form the UE sends depends on
    /// SIB1's `useFullResumeID`, not on the network's preference. A gNB that only
    /// indexed one form would fail to find its own context half the time.
    pub fn find_suspended(&self, identity: SuspendedIdentity) -> Option<&SuspendedUeContext> {
        match identity {
            SuspendedIdentity::Full(full) => self.suspended.get(&full),
            SuspendedIdentity::Short(short) => self
                .suspended
                .values()
                .find(|ctx| short_i_rnti_of(ctx.full_i_rnti) == short),
        }
    }

    /// Take a suspended context out of the store, as a resume does.
    ///
    /// Removal is the point: an I-RNTI is single-use (TS 38.331 §5.3.13.3 — the
    /// network assigns a new one if it suspends the UE again), so leaving it in place
    /// would let the same `RRCResumeRequest` be replayed for as long as the context
    /// lived.
    pub fn take_suspended(&mut self, identity: SuspendedIdentity) -> Option<SuspendedUeContext> {
        let full = self.find_suspended(identity)?.full_i_rnti;
        self.suspended.remove(&full)
    }

    /// How many UEs are suspended in RRC_INACTIVE.
    pub fn suspended_count(&self) -> usize {
        self.suspended.len()
    }

    /// The full I-RNTIs of every suspended UE, ascending, for status reporting.
    pub fn suspended_i_rntis(&self) -> Vec<u64> {
        let mut ids: Vec<u64> = self.suspended.keys().copied().collect();
        ids.sort_unstable();
        ids
    }

    /// UE contexts whose stored AS security context matches the `(C-RNTI, PCI)`
    /// a UE presents in an `RRCReestablishmentRequest` (TS 38.331 §5.3.7.2).
    ///
    /// Returns **candidates**, not one context, and in ascending `ue_id` order so
    /// the caller's verification is deterministic. Two reasons it can be more than
    /// one: the simulator has no C-RNTI allocation, so every UE presents the same
    /// well-known constant (see [`ReestablishmentSecurity::c_rnti`]). The
    /// `shortMAC-I` verification is what resolves the set — each UE has a
    /// different `K_RRCint`, so at most one candidate can produce the presented
    /// MAC.
    ///
    /// A context with no stored AS security is not a candidate: it cannot be
    /// verified, and §5.3.3.1 says an unverifiable context means `RRCSetup`.
    pub fn candidates_for_reestablishment(&self, c_rnti: u16, phys_cell_id: u16) -> Vec<i32> {
        let mut ids: Vec<i32> = self
            .contexts
            .iter()
            .filter(|(_, ctx)| {
                ctx.reestablishment_security
                    .as_ref()
                    .is_some_and(|s| s.c_rnti == c_rnti && s.phys_cell_id == phys_cell_id)
            })
            .map(|(id, _)| *id)
            .collect();
        ids.sort_unstable();
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nextgsim_common::Plmn;

    fn create_test_s_tmsi() -> GutiMobileIdentity {
        GutiMobileIdentity {
            plmn: Plmn::new(1, 1, false),
            amf_region_id: 1,
            amf_set_id: 1,
            amf_pointer: 1,
            tmsi: 0x12345678,
        }
    }

    #[test]
    fn test_rrc_state_default() {
        let state = RrcState::default();
        assert_eq!(state, RrcState::Idle);
    }

    #[test]
    fn test_rrc_state_display() {
        assert_eq!(format!("{}", RrcState::Idle), "Idle");
        assert_eq!(format!("{}", RrcState::SetupRequest), "SetupRequest");
        assert_eq!(format!("{}", RrcState::SetupSent), "SetupSent");
        assert_eq!(format!("{}", RrcState::Connected), "Connected");
        assert_eq!(format!("{}", RrcState::Releasing), "Releasing");
    }

    #[test]
    fn test_rrc_ue_context_new() {
        let ctx = RrcUeContext::new(1);
        assert_eq!(ctx.ue_id, 1);
        assert!(ctx.initial_id.is_none());
        assert!(!ctx.is_initial_id_s_tmsi);
        assert_eq!(ctx.establishment_cause, 0);
        assert!(ctx.s_tmsi.is_none());
        assert_eq!(ctx.state, RrcState::Idle);
        assert!(ctx.is_idle());
        assert!(!ctx.is_connected());
    }

    #[test]
    fn test_rrc_ue_context_set_initial_id() {
        let mut ctx = RrcUeContext::new(1);

        // Set random initial ID
        ctx.set_initial_id(0x123456789, false);
        assert_eq!(ctx.initial_id, Some(0x123456789));
        assert!(!ctx.is_initial_id_s_tmsi);

        // Set S-TMSI initial ID
        ctx.set_initial_id(0x987654321, true);
        assert_eq!(ctx.initial_id, Some(0x987654321));
        assert!(ctx.is_initial_id_s_tmsi);
    }

    #[test]
    fn test_rrc_ue_context_set_establishment_cause() {
        let mut ctx = RrcUeContext::new(1);
        ctx.set_establishment_cause(3); // e.g., mo-Data
        assert_eq!(ctx.establishment_cause, 3);
    }

    #[test]
    fn test_rrc_ue_context_set_s_tmsi() {
        let mut ctx = RrcUeContext::new(1);
        let s_tmsi = create_test_s_tmsi();
        ctx.set_s_tmsi(s_tmsi.clone());
        assert!(ctx.s_tmsi.is_some());
        assert_eq!(ctx.s_tmsi.as_ref().unwrap().tmsi, 0x12345678);
    }

    #[test]
    fn test_rrc_ue_context_state_transitions() {
        let mut ctx = RrcUeContext::new(1);
        assert!(ctx.is_idle());

        ctx.on_setup_request();
        assert_eq!(ctx.state, RrcState::SetupRequest);
        assert!(!ctx.is_idle());
        assert!(!ctx.is_connected());

        ctx.on_setup_sent();
        assert_eq!(ctx.state, RrcState::SetupSent);

        ctx.on_setup_complete();
        assert_eq!(ctx.state, RrcState::Connected);
        assert!(ctx.is_connected());

        ctx.on_release();
        assert_eq!(ctx.state, RrcState::Releasing);
        assert!(!ctx.is_connected());
    }

    #[test]
    fn test_rrc_ue_context_manager_new() {
        let manager = RrcUeContextManager::new();
        assert_eq!(manager.count(), 0);
        assert!(manager.is_empty());
    }

    #[test]
    fn test_rrc_ue_context_manager_create_ue() {
        let mut manager = RrcUeContextManager::new();

        let ctx = manager.create_ue(1);
        assert_eq!(ctx.ue_id, 1);
        assert_eq!(manager.count(), 1);
        assert!(!manager.is_empty());

        // Creating with same ID replaces
        let ctx = manager.create_ue(1);
        ctx.set_establishment_cause(5);
        assert_eq!(manager.count(), 1);
        assert_eq!(manager.try_find_ue(1).unwrap().establishment_cause, 5);
    }

    #[test]
    fn test_rrc_ue_context_manager_try_find_ue() {
        let mut manager = RrcUeContextManager::new();

        assert!(manager.try_find_ue(1).is_none());

        manager.create_ue(1);
        assert!(manager.try_find_ue(1).is_some());
        assert_eq!(manager.try_find_ue(1).unwrap().ue_id, 1);
    }

    #[test]
    fn test_rrc_ue_context_manager_try_find_ue_mut() {
        let mut manager = RrcUeContextManager::new();
        manager.create_ue(1);

        let ctx = manager.try_find_ue_mut(1).unwrap();
        ctx.set_establishment_cause(7);

        assert_eq!(manager.try_find_ue(1).unwrap().establishment_cause, 7);
    }

    #[test]
    fn test_rrc_ue_context_manager_find_or_create_ue() {
        let mut manager = RrcUeContextManager::new();

        // Creates new context
        let ctx = manager.find_or_create_ue(1);
        ctx.set_establishment_cause(3);
        assert_eq!(manager.count(), 1);

        // Returns existing context
        let ctx = manager.find_or_create_ue(1);
        assert_eq!(ctx.establishment_cause, 3);
        assert_eq!(manager.count(), 1);
    }

    #[test]
    fn test_rrc_ue_context_manager_delete_ue() {
        let mut manager = RrcUeContextManager::new();
        manager.create_ue(1);
        manager.create_ue(2);
        assert_eq!(manager.count(), 2);

        let removed = manager.delete_ue(1);
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().ue_id, 1);
        assert_eq!(manager.count(), 1);

        // Deleting non-existent returns None
        let removed = manager.delete_ue(1);
        assert!(removed.is_none());
    }

    #[test]
    fn test_rrc_ue_context_manager_ue_ids() {
        let mut manager = RrcUeContextManager::new();
        manager.create_ue(1);
        manager.create_ue(2);
        manager.create_ue(3);

        let mut ids = manager.ue_ids();
        ids.sort();
        assert_eq!(ids, vec![1, 2, 3]);
    }

    #[test]
    fn test_rrc_ue_context_manager_connected_ue_ids() {
        let mut manager = RrcUeContextManager::new();

        manager.create_ue(1);
        manager.try_find_ue_mut(1).unwrap().on_setup_complete();

        manager.create_ue(2);
        // UE 2 stays in Idle

        manager.create_ue(3);
        manager.try_find_ue_mut(3).unwrap().on_setup_complete();

        let mut connected = manager.connected_ue_ids();
        connected.sort();
        assert_eq!(connected, vec![1, 3]);
    }

    #[test]
    fn test_rrc_ue_context_manager_iter() {
        let mut manager = RrcUeContextManager::new();
        manager.create_ue(1);
        manager.create_ue(2);

        let count = manager.iter().count();
        assert_eq!(count, 2);
    }
}
