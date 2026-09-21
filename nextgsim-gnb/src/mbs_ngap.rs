//! MBS (Multicast-Broadcast Services) session state at the gNB (Rel-17)
//!
//! This is the state machine the NGAP task drives when an MBS procedure arrives
//! on the NG-C association. The wire codec lives in
//! `nextgsim_ngap::procedures::mbs`; this module holds only what the gNB
//! remembers between messages:
//!
//! - which MBS sessions exist, keyed by the TMGI the AMF sent;
//! - which cells are currently radiating each session;
//! - which UEs have joined a multicast (as opposed to broadcast) session.
//!
//! This is the **only** MBS state machine in `nextgsim-gnb`. A second one
//! (`ngap::mbs_context::MbsSessionManager`) used to sit beside it, keyed by a
//! numeric session id that appears nowhere in TS 38.413's IEs and driven by
//! `NgapMessage::Mbs*` variants that nothing ever constructed. It was deleted in
//! issue #188: it modelled the same concern with a key the wire cannot resolve.
//! See this module's `join_ue`/`leave_ue` for where the per-UE half of that
//! concern actually lives now.
//!
//! # Which procedures reach it
//!
//! Two groups, because 3GPP splits MBS across both halves of NGAP.
//!
//! **Session level — non-UE-associated (TS 38.413 §9.2.9),** driven from
//! `NgapTask::handle_mbs_pdu`:
//!
//! - 71 `id-MulticastSessionActivation` -> [`NgapMbsManager::start_session`],
//!   answered with a `MulticastSessionActivationResponse`;
//! - 72 `id-MulticastSessionDeactivation` -> [`NgapMbsManager::stop_session`],
//!   answered with a `MulticastSessionDeactivationResponse`;
//! - 74 `id-MulticastGroupPaging` -> [`NgapMbsManager::sessions_for_cell`] to
//!   decide whether this cell carries the paged group.
//!
//! **Membership level — UE-associated,** driven from the PDU Session Resource
//! procedures, because TS 23.247 delivers a multicast session to a UE *through*
//! its PDU session, so membership travels with it:
//!
//! - `MBSSessionSetupRequestList` (IE 318) on a Setup Request, and
//!   `MBSSessionSetuporModifyRequestList` (319) on a Modify Request ->
//!   [`NgapMbsManager::join_ue`];
//! - `MBSSessionToReleaseList` (317) on a Modify Request ->
//!   [`NgapMbsManager::leave_ue`];
//! - a UE context going away -> [`NgapMbsManager::remove_ue_everywhere`].
//!
//! Each join is answered per session in the response transfer's `iE-Extensions`
//! (312/310 for setup, 313/311 for modify), so the AMF learns which joins the RAN
//! admitted and why the rest were refused.
//!
//! Procedure 68 (`id-BroadcastSessionSetup`) is deliberately NOT routed here.
//! It is a *broadcast* session setup whose `BroadcastSessionSetupRequest`
//! carries `MBS-ServiceArea` and `MBS-SessionTNLInfo5GC` — an MB-UPF tunnel this
//! state machine has no field for — so accepting it would record a session the
//! gNB cannot actually deliver. See `handle_unroutable_pdu`: it is answered with
//! an Error Indication, which is the conformant response to a procedure the node
//! does not support (TS 38.413 §8.7.5).
//!
//! # Session identity
//!
//! Sessions are keyed by the 6 TMGI octets as they arrived, not by a formatted
//! string, so a lookup cannot miss because two code paths disagreed on hex case
//! or on which half of the TMGI comes first (TS 23.003 §30.2).

use std::collections::{HashMap, HashSet};

/// MBS session state at gNB
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GnbMbsState {
    /// Session setup in progress
    Starting,
    /// Session active, broadcasting/multicasting
    Active,
    /// Session stop in progress
    Stopping,
    /// Session terminated
    Stopped,
    /// Session setup failed
    Failed,
}

/// MBS flow (individual content stream within an MBS session)
#[derive(Debug, Clone)]
pub struct MbsFlow {
    /// MBS flow ID
    pub flow_id: u8,
    /// Data radio bearer ID for this flow
    pub mrb_id: u8,
    /// QoS flow indicator
    pub qfi: u8,
    /// Whether PDCP is active for this flow
    pub pdcp_active: bool,
}

/// GTP-U tunnel for MBS downlink
#[derive(Debug, Clone)]
pub struct MbsTunnel {
    /// TEID at UPF (data source)
    pub upf_teid: u32,
    /// UPF IP address
    pub upf_ip: [u8; 4],
    /// TEID at gNB (local receive)
    pub gnb_teid: u32,
}

/// MBS session context at gNB
#[derive(Debug, Clone)]
pub struct GnbMbsSession {
    /// MBS session ID (from AMF/SMF)
    pub mbs_session_id: String,
    /// TMGI (6 octets)
    pub tmgi: [u8; 6],
    /// Area session ID
    pub area_session_id: Option<u32>,
    /// Session state
    pub state: GnbMbsState,
    /// MBS flows in this session
    pub flows: Vec<MbsFlow>,
    /// GTP-U tunnel for downlink
    pub tunnel: Option<MbsTunnel>,
    /// Cell IDs that are broadcasting this session
    pub active_cells: HashSet<i32>,
    /// UEs that have joined this session (multicast mode), by internal UE id.
    ///
    /// Keyed by `ue_id` — the identity `NgapUeContext` is indexed by and that the
    /// PDU Session procedures resolve from `RAN-UE-NGAP-ID` — and deliberately
    /// **not** by C-RNTI. This simulator gives every UE the same C-RNTI
    /// (`SIMULATED_C_RNTI`), whose own documentation notes that `(C-RNTI, PCI)`
    /// identifies a *set* of contexts rather than one; a C-RNTI-keyed membership
    /// set therefore cannot hold two UEs at once, so the second join would be a
    /// silent no-op and the first leave would evict both.
    pub joined_ues: HashSet<i32>,
    /// Whether this is a broadcast session (no per-UE membership)
    pub is_broadcast: bool,
}

impl GnbMbsSession {
    pub fn new_broadcast(mbs_session_id: String, tmgi: [u8; 6]) -> Self {
        Self {
            mbs_session_id,
            tmgi,
            area_session_id: None,
            state: GnbMbsState::Starting,
            flows: Vec::new(),
            tunnel: None,
            active_cells: HashSet::new(),
            joined_ues: HashSet::new(),
            is_broadcast: true,
        }
    }

    pub fn new_multicast(mbs_session_id: String, tmgi: [u8; 6]) -> Self {
        let mut s = Self::new_broadcast(mbs_session_id, tmgi);
        s.is_broadcast = false;
        s
    }

    /// Activates the session in a cell
    pub fn activate_cell(&mut self, cell_id: i32) {
        self.active_cells.insert(cell_id);
        self.state = GnbMbsState::Active;
    }

    /// Deactivates the session in a cell
    pub fn deactivate_cell(&mut self, cell_id: i32) {
        self.active_cells.remove(&cell_id);
        if self.active_cells.is_empty() {
            self.state = GnbMbsState::Stopping;
        }
    }

    /// Records that a UE joined this multicast session.
    ///
    /// Returns `false` for a broadcast session, where there is no per-UE
    /// membership to record: a broadcast session is radiated to the cell whether
    /// or not any particular UE is interested (TS 23.247 §4.2), so accepting a
    /// join would invent state the session does not have. Also `false` when the
    /// UE had already joined, so a caller can tell a new membership from a
    /// repeated request.
    pub fn ue_join(&mut self, ue_id: i32) -> bool {
        if self.is_broadcast {
            return false;
        }
        self.joined_ues.insert(ue_id)
    }

    /// Records that a UE left this session, reporting whether it was a member.
    pub fn ue_leave(&mut self, ue_id: i32) -> bool {
        self.joined_ues.remove(&ue_id)
    }

    /// Whether a given UE is currently a member of this session.
    pub fn has_ue(&self, ue_id: i32) -> bool {
        self.joined_ues.contains(&ue_id)
    }

    /// Returns TMGI as hex string
    pub fn tmgi_hex(&self) -> String {
        self.tmgi.iter().map(|b| format!("{b:02X}")).collect()
    }

    pub fn is_active(&self) -> bool {
        self.state == GnbMbsState::Active
    }
}

/// NGAP MBS session manager at gNB
///
/// Driven by `NgapTask::handle_ngap_pdu` for procedures 71, 72 and 74; see the
/// module docs.
#[derive(Debug, Default)]
pub struct NgapMbsManager {
    /// Active MBS sessions keyed by the 6 TMGI octets.
    ///
    /// Keyed by the raw octets rather than by `mbs_session_id`'s formatted text
    /// because the TMGI is what the AMF actually puts in `id-MBS-SessionID`
    /// (299) on every MBS message, so it is the only identifier a deactivation
    /// or a group paging is guaranteed to arrive with.
    sessions: HashMap<[u8; 6], GnbMbsSession>,
}

impl NgapMbsManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Records the session a `MulticastSessionActivationRequest` activated
    /// (TS 38.413 §9.2.9.1).
    ///
    /// Re-activating an existing TMGI replaces the session, which is what the
    /// AMF asking again means: the previous context is stale.
    pub fn start_session(&mut self, session: GnbMbsSession) -> &GnbMbsSession {
        let tmgi = session.tmgi;
        self.sessions.insert(tmgi, session);
        &self.sessions[&tmgi]
    }

    /// Removes the session a `MulticastSessionDeactivationRequest` named
    /// (TS 38.413 §9.2.9.3), returning it in the `Stopped` state.
    ///
    /// Returns `None` for a TMGI this gNB never activated, so the caller can
    /// tell "torn down" from "was never here".
    pub fn stop_session(&mut self, tmgi: &[u8; 6]) -> Option<GnbMbsSession> {
        let mut session = self.sessions.remove(tmgi)?;
        session.state = GnbMbsState::Stopped;
        Some(session)
    }

    /// Gets a session mutably by TMGI
    pub fn get_mut(&mut self, tmgi: &[u8; 6]) -> Option<&mut GnbMbsSession> {
        self.sessions.get_mut(tmgi)
    }

    /// Gets a session by TMGI
    pub fn get(&self, tmgi: &[u8; 6]) -> Option<&GnbMbsSession> {
        self.sessions.get(tmgi)
    }

    /// Returns all sessions active on a given cell
    pub fn sessions_for_cell(&self, cell_id: i32) -> Vec<&GnbMbsSession> {
        self.sessions
            .values()
            .filter(|s| s.active_cells.contains(&cell_id))
            .collect()
    }

    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Joins a UE to the session named by `tmgi`, for a
    /// `MBSSessionSetupRequestList` (318) or `MBSSessionSetuporModifyRequestList`
    /// (319) item on a PDU Session procedure (TS 23.247 §7.2.1.3).
    ///
    /// Refuses, rather than creating the session, when this gNB is not already
    /// radiating that TMGI in `cell_id`. Establishing it here would need the
    /// Distribution Setup procedures (69/70) against the MB-UPF, which this node
    /// has no user-plane path for — the same reason procedures 68 and 73 are left
    /// unrouted. Admitting the join anyway would tell the AMF a UE is receiving a
    /// session no traffic can reach it on.
    pub fn join_ue(
        &mut self,
        tmgi: &[u8; 6],
        ue_id: i32,
        cell_id: i32,
    ) -> Result<MbsJoinAccepted, MbsJoinRefusal> {
        let session = self
            .sessions
            .get_mut(tmgi)
            .ok_or(MbsJoinRefusal::UnknownSession)?;

        if !session.active_cells.contains(&cell_id) {
            return Err(MbsJoinRefusal::NotActiveInCell);
        }
        if session.is_broadcast {
            return Err(MbsJoinRefusal::BroadcastSessionHasNoMembership);
        }

        let newly_joined = session.ue_join(ue_id);
        Ok(MbsJoinAccepted {
            newly_joined,
            member_count: session.joined_ues.len(),
        })
    }

    /// Removes a UE from the session named by `tmgi`, for an
    /// `MBSSessionToReleaseList` (317) item.
    ///
    /// Returns whether the UE was a member. A leave for an unknown TMGI or a
    /// non-member is not an error: the AMF's intent — that the UE no longer
    /// receive the session — already holds.
    pub fn leave_ue(&mut self, tmgi: &[u8; 6], ue_id: i32) -> bool {
        self.sessions
            .get_mut(tmgi)
            .is_some_and(|session| session.ue_leave(ue_id))
    }

    /// Drops a UE from every session it had joined, returning the TMGIs it left.
    ///
    /// Called when a UE context goes away, so a released UE does not linger as a
    /// member of a session that is still radiating. Without this a long-lived
    /// session would accumulate the ids of every UE that ever joined it, and
    /// `member_count` would stop meaning anything.
    pub fn remove_ue_everywhere(&mut self, ue_id: i32) -> Vec<[u8; 6]> {
        let mut left = Vec::new();
        for (tmgi, session) in self.sessions.iter_mut() {
            if session.ue_leave(ue_id) {
                left.push(*tmgi);
            }
        }
        left
    }

    /// The sessions a given UE has joined.
    pub fn sessions_for_ue(&self, ue_id: i32) -> Vec<&GnbMbsSession> {
        self.sessions.values().filter(|s| s.has_ue(ue_id)).collect()
    }
}

/// What the gNB recorded when it admitted a UE's MBS join.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MbsJoinAccepted {
    /// `false` when the UE was already a member, so the join was a repeat.
    pub newly_joined: bool,
    /// How many UEs are members after the join.
    pub member_count: usize,
}

/// Why the gNB refused a UE's MBS join.
///
/// Distinct variants rather than one error because they map to different NGAP
/// causes and tell the AMF different things to do about it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MbsJoinRefusal {
    /// No session with that TMGI has been activated at this gNB.
    ///
    /// TS 38.413's `unknown-MBS-Session-ID` names exactly this case.
    UnknownSession,
    /// The session exists but is not being radiated in the UE's serving cell,
    /// which is `indicated-MBS-session-area-information-not-served-by-the-gNB`.
    NotActiveInCell,
    /// The TMGI names a broadcast session, which has no per-UE membership.
    BroadcastSessionHasNoMembership,
}

impl MbsJoinRefusal {
    /// A short reason for logs and for the Error Indication's diagnostics.
    pub fn reason(self) -> &'static str {
        match self {
            Self::UnknownSession => "no MBS session with that TMGI is active at this gNB",
            Self::NotActiveInCell => "the MBS session is not radiating in the UE's serving cell",
            Self::BroadcastSessionHasNoMembership => {
                "the TMGI names a broadcast session, which has no per-UE membership"
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_TMGI: [u8; 6] = [0xAB, 0xCD, 0xEF, 0x01, 0x02, 0x03];

    fn test_session() -> GnbMbsSession {
        GnbMbsSession::new_multicast("mbs-001".into(), TEST_TMGI)
    }

    #[test]
    fn test_mbs_session_activate() {
        let mut s = test_session();
        assert_eq!(s.state, GnbMbsState::Starting);
        s.activate_cell(1);
        assert_eq!(s.state, GnbMbsState::Active);
    }

    #[test]
    fn test_mbs_session_last_cell_deactivate() {
        let mut s = test_session();
        s.activate_cell(1);
        s.deactivate_cell(1);
        assert_eq!(s.state, GnbMbsState::Stopping);
    }

    #[test]
    fn test_multicast_ue_join_leave() {
        let mut s = test_session();
        assert!(s.ue_join(0x0100));
        assert_eq!(s.joined_ues.len(), 1);
        assert!(s.ue_leave(0x0100));
        assert!(s.joined_ues.is_empty());
    }

    #[test]
    fn test_broadcast_ue_join_rejected() {
        let mut s = GnbMbsSession::new_broadcast("bc-001".into(), [0; 6]);
        assert!(!s.ue_join(0x0100));
    }

    /// Two different UEs are two different members.
    ///
    /// This is the guard on the keying fix from issue #188: membership used to be
    /// keyed by C-RNTI, and every UE in this simulator carries the same
    /// `SIMULATED_C_RNTI`, so the second UE's join was silently swallowed and the
    /// first UE's leave evicted both. Keyed by `ue_id` both are held.
    #[test]
    fn two_ues_are_two_distinct_members() {
        let mut s = test_session();
        assert!(s.ue_join(1), "the first UE joins");
        assert!(
            s.ue_join(2),
            "the second UE is a distinct member, not a repeat"
        );
        assert_eq!(s.joined_ues.len(), 2);
        assert!(s.has_ue(1) && s.has_ue(2));

        assert!(s.ue_leave(1), "the first UE leaves");
        assert!(
            s.has_ue(2),
            "one UE leaving must not evict the other: that is the C-RNTI collision \
             this keying exists to avoid"
        );
        assert_eq!(s.joined_ues.len(), 1);
    }

    /// A repeated join is reported as a repeat rather than counted twice.
    #[test]
    fn a_repeated_join_is_not_a_second_membership() {
        let mut s = test_session();
        assert!(s.ue_join(7));
        assert!(!s.ue_join(7), "the same UE joining again is not new");
        assert_eq!(s.joined_ues.len(), 1);
    }

    #[test]
    fn test_manager_start_stop() {
        let mut mgr = NgapMbsManager::new();
        mgr.start_session(test_session());
        assert_eq!(mgr.session_count(), 1);
        let stopped = mgr
            .stop_session(&TEST_TMGI)
            .expect("the session was started");
        assert_eq!(stopped.state, GnbMbsState::Stopped);
        assert_eq!(mgr.session_count(), 0);
    }

    /// Deactivating a TMGI that was never activated reports that, rather than
    /// reporting a successful teardown of nothing.
    #[test]
    fn stopping_an_unknown_tmgi_reports_it() {
        let mut mgr = NgapMbsManager::new();
        mgr.start_session(test_session());
        assert!(mgr.stop_session(&[0x99; 6]).is_none());
        assert_eq!(mgr.session_count(), 1, "the real session must survive");
    }

    /// A session is looked up by the TMGI octets the AMF sends, so the NGAP
    /// dispatch path can find it from `id-MBS-SessionID` (299) alone.
    #[test]
    fn a_session_is_found_by_its_tmgi() {
        let mut mgr = NgapMbsManager::new();
        mgr.start_session(test_session());
        assert_eq!(mgr.get(&TEST_TMGI).map(|s| s.tmgi), Some(TEST_TMGI));
        assert!(mgr.get(&[0x00; 6]).is_none());
    }

    #[test]
    fn test_sessions_for_cell() {
        let mut mgr = NgapMbsManager::new();
        let mut s = test_session();
        s.activate_cell(5);
        mgr.start_session(s);
        assert_eq!(mgr.sessions_for_cell(5).len(), 1);
        assert!(mgr.sessions_for_cell(99).is_empty());
    }

    // ------------------------------------------------------------------
    // Per-UE membership through the manager (issue #188)
    // ------------------------------------------------------------------

    const CELL: i32 = 5;

    fn manager_radiating_in_cell() -> NgapMbsManager {
        let mut mgr = NgapMbsManager::new();
        let mut s = test_session();
        s.activate_cell(CELL);
        mgr.start_session(s);
        mgr
    }

    /// A join against a session this cell radiates is admitted, and the UE's
    /// membership is then observable through the manager.
    #[test]
    fn a_joined_ue_is_observable_through_the_manager() {
        let mut mgr = manager_radiating_in_cell();

        let accepted = mgr
            .join_ue(&TEST_TMGI, 42, CELL)
            .expect("the session is radiating in this cell");
        assert!(accepted.newly_joined);
        assert_eq!(accepted.member_count, 1);

        assert!(
            mgr.get(&TEST_TMGI).expect("the session").has_ue(42),
            "the join must be recorded on the session, not just reported"
        );
        assert_eq!(
            mgr.sessions_for_ue(42)
                .iter()
                .map(|s| s.tmgi)
                .collect::<Vec<_>>(),
            vec![TEST_TMGI],
            "the UE's memberships must be discoverable from the UE side too"
        );
    }

    /// A join naming a TMGI this gNB never activated is refused with the reason
    /// TS 38.413's `unknown-MBS-Session-ID` names — not silently created.
    #[test]
    fn a_join_for_an_unactivated_tmgi_is_refused() {
        let mut mgr = manager_radiating_in_cell();
        assert_eq!(
            mgr.join_ue(&[0x99; 6], 42, CELL),
            Err(MbsJoinRefusal::UnknownSession)
        );
        assert_eq!(
            mgr.session_count(),
            1,
            "a refused join must not conjure a session: establishing one needs the \
             Distribution Setup this node cannot do"
        );
    }

    /// A session that exists but is not radiating in the UE's cell is refused,
    /// because traffic could not reach the UE there.
    #[test]
    fn a_join_in_a_cell_not_radiating_the_session_is_refused() {
        let mut mgr = manager_radiating_in_cell();
        assert_eq!(
            mgr.join_ue(&TEST_TMGI, 42, CELL + 1),
            Err(MbsJoinRefusal::NotActiveInCell)
        );
        assert!(
            !mgr.get(&TEST_TMGI).expect("the session").has_ue(42),
            "a refused join must leave no membership behind"
        );
    }

    /// A broadcast session has no per-UE membership, so a join against one is
    /// refused rather than recorded (TS 23.247 §4.2).
    #[test]
    fn a_join_against_a_broadcast_session_is_refused() {
        let mut mgr = NgapMbsManager::new();
        let mut s = GnbMbsSession::new_broadcast("bc-1".into(), TEST_TMGI);
        s.activate_cell(CELL);
        mgr.start_session(s);

        assert_eq!(
            mgr.join_ue(&TEST_TMGI, 42, CELL),
            Err(MbsJoinRefusal::BroadcastSessionHasNoMembership)
        );
    }

    /// A leave removes exactly the named UE from exactly the named session.
    #[test]
    fn a_leave_removes_only_that_ue_from_only_that_session() {
        let mut mgr = manager_radiating_in_cell();
        let other_tmgi = [0x11; 6];
        let mut other = GnbMbsSession::new_multicast("mbs-002".into(), other_tmgi);
        other.activate_cell(CELL);
        mgr.start_session(other);

        mgr.join_ue(&TEST_TMGI, 1, CELL).expect("join a");
        mgr.join_ue(&TEST_TMGI, 2, CELL).expect("join a");
        mgr.join_ue(&other_tmgi, 1, CELL).expect("join b");

        assert!(mgr.leave_ue(&TEST_TMGI, 1));

        assert!(
            !mgr.get(&TEST_TMGI).unwrap().has_ue(1),
            "UE 1 left session A"
        );
        assert!(
            mgr.get(&TEST_TMGI).unwrap().has_ue(2),
            "UE 2 was not asked to leave"
        );
        assert!(
            mgr.get(&other_tmgi).unwrap().has_ue(1),
            "UE 1's membership of the other session is untouched"
        );
    }

    /// Leaving a session the UE never joined, or an unknown TMGI, reports that
    /// rather than erroring: the AMF's intent already holds.
    #[test]
    fn leaving_without_a_membership_reports_it() {
        let mut mgr = manager_radiating_in_cell();
        assert!(!mgr.leave_ue(&TEST_TMGI, 99), "UE 99 was never a member");
        assert!(!mgr.leave_ue(&[0x99; 6], 1), "that TMGI is not active here");
    }

    /// A released UE is dropped from every session it had joined, so a session's
    /// member count keeps meaning "UEs currently receiving this".
    #[test]
    fn releasing_a_ue_drops_it_from_every_session() {
        let mut mgr = manager_radiating_in_cell();
        let other_tmgi = [0x11; 6];
        let mut other = GnbMbsSession::new_multicast("mbs-002".into(), other_tmgi);
        other.activate_cell(CELL);
        mgr.start_session(other);

        mgr.join_ue(&TEST_TMGI, 1, CELL).expect("join a");
        mgr.join_ue(&other_tmgi, 1, CELL).expect("join b");
        mgr.join_ue(&TEST_TMGI, 2, CELL).expect("join a as UE 2");

        let mut left = mgr.remove_ue_everywhere(1);
        left.sort_unstable();
        let mut expected = vec![TEST_TMGI, other_tmgi];
        expected.sort_unstable();
        assert_eq!(left, expected, "both memberships must be reported as left");

        assert!(mgr.sessions_for_ue(1).is_empty());
        assert!(
            mgr.get(&TEST_TMGI).unwrap().has_ue(2),
            "releasing UE 1 must not evict UE 2"
        );
    }
}
