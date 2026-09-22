//! PC5 end to end: discovery → unicast link → UE-to-UE relay (#141), and remote UE →
//! relay → the network through the SRAP adaptation layer (#190).
//!
//! Runs real `SidelinkTask`s in one process, wired to each other's real message inboxes,
//! and asserts what each UE ends up believing about the others — read out of the tasks' own
//! state, not out of log lines:
//!
//! ```text
//! #141, UE-to-UE:
//!   UE-1 announces (Model A)                                   TS 23.304 §6.3.1.2 step 3a
//!     → UE-2 decodes it, matches its filter, records UE-1              §6.3.1.2 step 4b
//!   UE-2 sends a DIRECT COMMUNICATION REQUEST to UE-1                  §6.4.3.1 step 3
//!     → UE-1 accepts, answering with its own Layer-2 ID                §6.4.3.1 step 5a
//!     → UE-2 applies the accept and its link becomes Active            §6.4.3.1 step 4/5
//!   the relay forwards a payload from UE-1 to UE-2                      §6.4.3.10
//!
//! #190, UE-to-Network:
//!   the relay announces its Relay Service Code                          §6.3.2
//!     → the remote UE discovers it and opens a unicast link             §6.4.3.1
//!   each declares its ue-Type-r17 to the network              TS 38.331 §6.2.2
//!   the gNB signals the relay's SRAP bearer mapping                    §5.3.5.17
//!   the remote UE's payload goes through the relay
//!     → adapted through SRAP, with the gNB-assigned local Remote UE ID
//!                                              TS 38.351, TS 38.300 §16.12.2.1
//!     → and submitted on the relay's OWN Uu connection                 §16.12.2.1
//! ```
//!
//! # What makes these assertions load-bearing
//!
//! Each is on a value **only reachable by having decoded a peer's bytes, or by having gone
//! through the adaptation layer**:
//!
//! * `link_peer_l2_id` holds the peer's Layer-2 ID, which TS 23.304 §6.4.3.1 step 4 says
//!   the initiator "obtains" from what the responder sent. A UE that faked a handshake by
//!   setting its own state would have `None` here.
//! * `link_state(..) == Active` is entered at exactly two sites in
//!   `sidelink::link`, both of which require a decoded peer message.
//! * `relay_forwarded_octets` is incremented only inside `RelayForwarder::forward`,
//!   which consults the live link table — so a non-zero count proves both that the relay
//!   ran and that the link it forwarded over was really established.
//! * the **payload recovered from the SRAP PDU at the relay's Uu boundary**, and the
//!   **local Remote UE ID in its header**: a UE-to-UE relay produces no `RelayedUplink` at
//!   all, and a relay with no network-signalled mapping refuses to invent an identity, so
//!   neither value exists unless the whole #190 path ran.
//!
//! The old facade could satisfy none of them: `EstablishPc5Link` set `Active` on the line
//! after `Establishing` with no peer involved, discovery was a `bool`, and `RelayData`
//! emitted a `debug!` and forwarded nothing.
//!
//! # What this does NOT assert, and why
//!
//! **There is no PC5 radio.** The PC5-S PDUs are real — encoded and decoded by
//! `sidelink::pc5s`, one octet at a time, with the message-type octet and the 24-bit
//! Layer-2 IDs on the wire — but the medium between the UEs is a tokio channel, not a
//! sidelink carrier. So this test proves the *procedures* interoperate, not that any
//! PHY/MAC would carry them. That is the same arrangement the #136 SL-PRS stimulus uses
//! (it models propagation from geometry), and it is stated here rather than implied
//! because the startup wording says the same thing.
//!
//! **The relay's Uu leg is SRB1, not a numbered Uu relay RLC channel.** TS 38.300
//! §16.12.2.1 puts SRAP above RLC on the Uu hop, and this tree has no MAC multiplexing for
//! a numbered relay RLC channel — so the SRAP **header and adaptation are real** while the
//! channel carrying them is the one that exists. See `RrcTask::handle_relayed_uplink`.
//!
//! **No gNB is in this process.** The relay's SRAP mapping is injected as the message the
//! UE's RRC task sends on decoding an `sl-L2RelayUE-Config`; that the gNB *builds* that IE
//! and that it survives UPER are covered by `nextgsim-gnb::rrc::sidelink` and
//! `nextgsim-rrc::procedures::sidelink_relay_config` respectively. What this file covers is
//! the effect of it arriving.
//!
//! **This file only compiles under `--features sidelink`**, like `ranging_report_e2e.rs`,
//! and is run by the `sidelink-pc5` CI job.

#![cfg(feature = "sidelink")]

use nextgsim_common::config::{ProseConfig, UeConfig};
use nextgsim_ue::sidelink::{Pc5UnicastState, ProseL2Id};
use nextgsim_ue::tasks::{
    RrcMessage, SidelinkMessage, SidelinkRelayRole, TaskHandle, TaskMessage, UeTaskBase,
};
use nextgsim_ue::SidelinkTask;
use tokio::sync::mpsc;

/// The three UEs' ProSe Layer-2 IDs. Distinct in more than the low bits, so a truncation
/// or a swap cannot produce another UE's expected identity.
const UE1_L2_ID: u32 = 0x00_0A_00_01 & 0x00FF_FFFF;
const UE2_L2_ID: u32 = 0x00_0B_00_02 & 0x00FF_FFFF;
const RELAY_L2_ID: u32 = 0x00_0C_00_10 & 0x00FF_FFFF;

/// The ProSe Application Code all three share. Both ends of a discovery exchange must
/// agree on it or the monitor's filter never matches — which is itself asserted, by
/// `a_ue_on_a_different_application_code_is_not_discovered`.
const APP_CODE: u32 = 0xCAFE_0001;

/// A different code, used to prove the discovery filter is really applied.
const OTHER_APP_CODE: u32 = 0xBEEF_0002;

/// The Relay Service Code the relay UE serves (TS 23.304 §5.4.2).
const RELAY_SERVICE_CODE: u32 = 0x00_AB_CD;

/// A UE configuration with ProSe enabled.
///
/// `prose_enabled: true` matters: it is the runtime half of the gate issue #141 added, and
/// with it false every handler below returns early. That this test has to set it is what
/// makes `prose_enabled` a read flag rather than the write-only one it was.
fn prose_config(local_l2_id: u32, app_code: u32, relay_service_code: Option<u32>) -> UeConfig {
    UeConfig {
        prose_enabled: true,
        prose_config: Some(ProseConfig {
            local_l2_id,
            prose_app_code: app_code,
            discovery_model: "a".to_string(),
            relay_service_code,
            // UE-to-UE by default; `u2n_relay_config` overrides it (issue #190).
            ue_to_network_relay: false,
            rx_interested_freqs: vec![1],
            peer_timeout_ms: 5_000,
        }),
        ..UeConfig::default()
    }
}

/// One UE: its task, and the inbox other UEs reach it on.
struct SidelinkUe {
    task: SidelinkTask,
    /// This UE's own inbox, so a peer can be given a handle to it.
    inbox: TaskHandle<SidelinkMessage>,
    /// The receiving end, drained by [`Self::drain`].
    rx: mpsc::Receiver<TaskMessage<SidelinkMessage>>,
    /// What this UE sent towards its own RRC task — i.e. up its Uu leg.
    ///
    /// Drained rather than discarded since issue #190: the UE-to-Network relay's whole
    /// point is that a remote UE's traffic leaves on the **relay's Uu connection**, which
    /// is an `RrcMessage::RelayedUplink`. A test that dropped this channel could not tell
    /// a relay that carried the traffic from one that adapted it and threw it away.
    rrc_rx: mpsc::Receiver<TaskMessage<RrcMessage>>,
    /// SRAP PDUs seen on the Uu leg so far, accumulated by [`Self::drain_uu`].
    relayed_uplinks: Vec<(u32, Vec<u8>)>,
    /// The most recent `ue-Type-r17` role declared. `Some(None)` means a request was sent
    /// declaring no role, which is different from no request at all.
    declared_role: Option<Option<SidelinkRelayRole>>,
}

impl SidelinkUe {
    fn new(config: UeConfig) -> Self {
        let (tx, rx) = mpsc::channel::<TaskMessage<SidelinkMessage>>(64);
        let (app_tx, _app_rx) = mpsc::channel(64);
        let (nas_tx, _nas_rx) = mpsc::channel(64);
        // Drained by `take_relayed_uplinks` (issue #190): this is the relay's Uu leg.
        let (rrc_tx, rrc_rx) = mpsc::channel(64);
        let (rls_tx, _rls_rx) = mpsc::channel(64);
        let task_base = UeTaskBase {
            config: std::sync::Arc::new(config),
            app_tx: TaskHandle::new(app_tx),
            nas_tx: TaskHandle::new(nas_tx),
            rrc_tx: TaskHandle::new(rrc_tx),
            rls_tx: TaskHandle::new(rls_tx),
            #[cfg(any(
                feature = "nextgsim-she",
                feature = "nextgsim-nwdaf",
                feature = "nextgsim-isac",
                feature = "nextgsim-fl",
                feature = "nextgsim-semantic",
            ))]
            sixg: None,
            rel18: None,
        };
        Self {
            task: SidelinkTask::new(task_base),
            inbox: TaskHandle::new(tx),
            rx,
            rrc_rx,
            relayed_uplinks: Vec::new(),
            declared_role: None,
        }
    }

    /// Drains everything this UE has sent up its Uu leg into the two buckets the tests read.
    ///
    /// One drain rather than two accessors, because a `try_recv` loop consumes the channel:
    /// two independent drainers would have the first to run steal the other's messages, and
    /// the symptom would be a test failing for a reason that is not the code's.
    fn drain_uu(&mut self) {
        while let Ok(msg) = self.rrc_rx.try_recv() {
            match msg {
                TaskMessage::Message(RrcMessage::RelayedUplink { remote_l2_id, pdu }) => {
                    self.relayed_uplinks.push((remote_l2_id, pdu));
                }
                TaskMessage::Message(RrcMessage::SidelinkInterestChanged {
                    relay_role, ..
                }) => {
                    self.declared_role = Some(relay_role);
                }
                // Anything else is a message this test does not reason about, and absorbing
                // it is right: the sidelink task legitimately sends others.
                _ => {}
            }
        }
    }

    /// Every SRAP PDU this UE has sent up its own Uu connection, as
    /// `(remote_l2_id, pdu)`, and clears the record (issue #190).
    ///
    /// This is the far end of the relay leg. A non-empty result proves the relay adapted a
    /// remote UE's payload AND handed it to its Uu leg — neither of which a UE-to-UE relay
    /// does.
    fn take_relayed_uplinks(&mut self) -> Vec<(u32, Vec<u8>)> {
        self.drain_uu();
        std::mem::take(&mut self.relayed_uplinks)
    }

    /// The `ue-Type-r17` relay role this UE has declared to the network, from the most
    /// recent `SidelinkInterestChanged` it sent (issue #190).
    ///
    /// Read off the real message rather than from the task's internals, because the role
    /// only matters if it reaches RRC — that is what puts it on the wire.
    ///
    /// `None` covers both "no request has been sent" and "one was sent declaring no role":
    /// in each case the gNB is not being asked for relay resources, which is the thing under
    /// test.
    fn declared_relay_role(&mut self) -> Option<SidelinkRelayRole> {
        self.drain_uu();
        self.declared_role.flatten()
    }

    /// Takes every message waiting in this UE's inbox and feeds it to the REAL task
    /// handlers, returning how many were processed.
    ///
    /// Driven explicitly rather than by spawning `task.run()` so the test can assert
    /// between steps: a spawned task would need the state read across a channel, and the
    /// point is to read the task's own belief about its peer.
    ///
    /// Only the PC5 messages are dispatched here. That is deliberate: every message this
    /// test causes to be sent is a `Pc5SReceived` or a `RelayPayload`, so a message of
    /// any other kind arriving would mean the code under test sent something unexpected,
    /// and it is better for that to be visible than absorbed.
    async fn drain(&mut self) -> usize {
        let mut handled = 0;
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                TaskMessage::Message(SidelinkMessage::Pc5SReceived { pdu }) => {
                    self.task.handle_pc5s_received(&pdu).await;
                    handled += 1;
                }
                TaskMessage::Message(SidelinkMessage::RelayPayload { .. }) => {
                    // Counted but not re-dispatched: this UE is the payload's
                    // destination, and re-relaying it is what the relay already did.
                    handled += 1;
                }
                other => panic!("unexpected message in a PC5 inbox: {other:?}"),
            }
        }
        handled
    }
}

/// Model A discovery then a PC5 unicast link, both over real PC5-S bytes.
///
/// The `Active` assertion is the one issue #141 exists for: before it, `Active` was
/// reachable with no peer at all.
#[tokio::test]
async fn two_ues_discover_each_other_and_establish_a_real_pc5_unicast_link() {
    let mut ue1 = SidelinkUe::new(prose_config(UE1_L2_ID, APP_CODE, None));
    let mut ue2 = SidelinkUe::new(prose_config(UE2_L2_ID, APP_CODE, None));

    // Each UE transmits into the other's real inbox.
    ue1.task.set_pc5_transmitter(ue2.inbox.clone());
    ue2.task.set_pc5_transmitter(ue1.inbox.clone());

    let ue1_id = ProseL2Id::new(UE1_L2_ID);
    let ue2_id = ProseL2Id::new(UE2_L2_ID);

    // Precondition: neither UE knows the other, and neither holds a link.
    assert_eq!(ue2.task.discovered_peer_count(), 0);
    assert_eq!(ue2.task.link_state(ue1_id), None);

    // --- Discovery: UE-1 announces (TS 23.304 §6.3.1.2 step 3a) ---
    // An empty PDU first: a malformed PC5-S message must be warned about and dropped,
    // not panic the task. Asserted by the absence of a panic and by UE-2 learning
    // nothing from it below.
    ue1.task.handle_pc5s_received(&[]).await;
    ue1.task
        .handle_message(SidelinkMessage::StartDiscovery)
        .await;
    assert!(ue1.task.discovery_active(), "UE-1 is announcing");

    assert_eq!(ue2.drain().await, 1, "UE-2 received UE-1's announcement");
    assert_eq!(
        ue2.task.discovered_peer_count(),
        1,
        "UE-2 discovered UE-1 from its announcement (§6.3.1.2 step 4b)"
    );

    // --- Unicast link: UE-2 requests, UE-1 accepts (§6.4.3.1) ---
    ue2.task
        .handle_message(SidelinkMessage::EstablishPc5Link {
            peer_ue_id: u64::from(UE1_L2_ID),
        })
        .await;

    // UE-2 is WAITING, not active: the whole correction this issue makes.
    assert_eq!(
        ue2.task.link_state(ue1_id),
        Some(Pc5UnicastState::AwaitingAccept),
        "a request has been sent and not yet answered, so the link must not be usable"
    );
    assert_eq!(
        ue2.task.link_peer_l2_id(ue1_id),
        None,
        "UE-2 has not yet learned UE-1's Layer-2 ID: nothing has come back"
    );
    assert_eq!(ue2.task.active_link_count(), 0);

    // UE-1 receives the request and answers it.
    assert_eq!(ue1.drain().await, 1, "UE-1 received the request");
    assert_eq!(
        ue1.task.link_state(ue2_id),
        Some(Pc5UnicastState::Active),
        "the responder's link is up as soon as it has accepted (§6.4.3.1 step 5)"
    );
    assert_eq!(
        ue1.task.link_peer_l2_id(ue2_id),
        Some(ue2_id),
        "UE-1 learned UE-2's Layer-2 ID from the request's Source User Info"
    );

    // UE-2 receives the accept.
    assert_eq!(ue2.drain().await, 1, "UE-2 received the accept");
    assert_eq!(
        ue2.task.link_state(ue1_id),
        Some(Pc5UnicastState::Active),
        "the initiator's link becomes usable only on the accept (§6.4.3.1 step 5)"
    );
    // THE assertion: UE-2 holds UE-1's Layer-2 ID, and it could only have come from
    // UE-1's own accept bytes (§6.4.3.1 step 4).
    assert_eq!(
        ue2.task.link_peer_l2_id(ue1_id),
        Some(ue1_id),
        "UE-2 must have obtained the peer's Layer-2 ID from the accept it decoded"
    );
    assert_eq!(ue2.task.active_link_count(), 1);
    assert_eq!(ue1.task.active_link_count(), 1);
}

/// The discovery filter is really applied: a UE announcing a different ProSe Application
/// Code is not discovered, so no link can be opened to it.
///
/// This is the guard that the old `discovery_active: bool` could not have — it was on or
/// off, never selective.
#[tokio::test]
async fn a_ue_on_a_different_application_code_is_not_discovered() {
    let mut ue1 = SidelinkUe::new(prose_config(UE1_L2_ID, APP_CODE, None));
    let mut ue2 = SidelinkUe::new(prose_config(UE2_L2_ID, OTHER_APP_CODE, None));
    ue1.task.set_pc5_transmitter(ue2.inbox.clone());
    ue2.task.set_pc5_transmitter(ue1.inbox.clone());

    ue1.task
        .handle_message(SidelinkMessage::StartDiscovery)
        .await;
    assert_eq!(ue2.drain().await, 1, "the announcement did arrive");
    assert_eq!(
        ue2.task.discovered_peer_count(),
        0,
        "UE-2's filter is for a different application code, so it must learn nothing"
    );

    // And with no discovered peer, a link cannot be opened (§6.4.3.1 step 2).
    ue2.task
        .handle_message(SidelinkMessage::EstablishPc5Link {
            peer_ue_id: u64::from(UE1_L2_ID),
        })
        .await;
    assert_eq!(
        ue2.task.link_state(ProseL2Id::new(UE1_L2_ID)),
        None,
        "no link may be opened to an undiscovered peer"
    );
}

/// The full criterion-5 chain: discovery → unicast link → relay forwarding, with the
/// forwarded octets read off the relay.
#[tokio::test]
async fn a_two_ue_relay_forwards_a_payload_over_established_pc5_links() {
    // The relay serves an RSC and therefore starts as an L2 UE-to-UE relay.
    let mut relay = SidelinkUe::new(prose_config(
        RELAY_L2_ID,
        APP_CODE,
        Some(RELAY_SERVICE_CODE),
    ));
    let mut ue2 = SidelinkUe::new(prose_config(UE2_L2_ID, APP_CODE, None));

    relay.task.set_pc5_transmitter(ue2.inbox.clone());
    ue2.task.set_pc5_transmitter(relay.inbox.clone());

    let relay_id = ProseL2Id::new(RELAY_L2_ID);
    let ue2_id = ProseL2Id::new(UE2_L2_ID);

    // --- The relay announces, carrying its Relay Service Code (§6.3.2) ---
    relay
        .task
        .handle_message(SidelinkMessage::StartDiscovery)
        .await;
    assert_eq!(ue2.drain().await, 1);
    assert_eq!(
        ue2.task.discovered_peer_count(),
        1,
        "UE-2 discovered the relay"
    );

    // --- UE-2 opens a unicast link to the relay ON ITS OWN ---
    //
    // No `EstablishPc5Link` is sent by this test. Discovering a relay that serves a
    // Relay Service Code is itself what triggers the link (TS 23.304 §6.3.2 into
    // §6.4.3.1) -- which is `EstablishPc5Link`'s production sender, and the thing issue
    // #141 said it had none of. A test that sent the message itself would pass whether or
    // not that trigger existed.
    assert_eq!(
        ue2.task.link_state(relay_id),
        Some(Pc5UnicastState::AwaitingAccept),
        "discovering a relay must itself open the link, with no external prompt"
    );
    assert_eq!(relay.drain().await, 1, "the relay received the request");
    assert_eq!(ue2.drain().await, 1, "UE-2 received the accept");
    assert_eq!(
        ue2.task.link_state(relay_id),
        Some(Pc5UnicastState::Active),
        "UE-2's link to the relay must be established before anything is relayed"
    );
    assert_eq!(
        relay.task.link_state(ue2_id),
        Some(Pc5UnicastState::Active),
        "and the relay's own end of it"
    );

    // --- The relay forwards a payload from UE-1 to UE-2 (§6.4.3.10) ---
    const PAYLOAD: &[u8] = &[0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03];
    assert_eq!(
        relay.task.relay_forwarded_octets(),
        0,
        "precondition: nothing forwarded yet"
    );

    relay
        .task
        .handle_message(SidelinkMessage::RelayPayload {
            source_l2_id: UE1_L2_ID,
            destination_l2_id: UE2_L2_ID,
            payload: PAYLOAD.to_vec(),
        })
        .await;

    // The positive assertion: the octets were forwarded, counted inside
    // `RelayForwarder::forward` against the live link table.
    assert_eq!(
        relay.task.relay_forwarded_octets(),
        PAYLOAD.len() as u64,
        "the relay must have forwarded the payload over its established link to UE-2"
    );
    // And UE-2 actually received it.
    assert_eq!(
        ue2.drain().await,
        1,
        "the forwarded payload reached UE-2's inbox"
    );
}

/// Releasing the link stops the relaying: the forwarding path consults the LIVE link
/// state, not a peer list recorded once.
///
/// Without this, `a_two_ue_relay_forwards_a_payload_over_established_pc5_links` would
/// pass against a relay that forwarded to anyone it had ever seen.
#[tokio::test]
async fn releasing_the_pc5_link_stops_the_relay_forwarding() {
    let mut relay = SidelinkUe::new(prose_config(
        RELAY_L2_ID,
        APP_CODE,
        Some(RELAY_SERVICE_CODE),
    ));
    let mut ue2 = SidelinkUe::new(prose_config(UE2_L2_ID, APP_CODE, None));
    relay.task.set_pc5_transmitter(ue2.inbox.clone());
    ue2.task.set_pc5_transmitter(relay.inbox.clone());

    // Establish, as above: the relay announces, UE-2 discovers it and opens the link on
    // its own, and the handshake completes.
    relay
        .task
        .handle_message(SidelinkMessage::StartDiscovery)
        .await;
    ue2.drain().await;
    relay.drain().await;
    ue2.drain().await;
    assert_eq!(
        relay.task.link_state(ProseL2Id::new(UE2_L2_ID)),
        Some(Pc5UnicastState::Active),
        "precondition: the link is up"
    );

    // One successful forward, so the counter is known to move at all.
    relay
        .task
        .handle_message(SidelinkMessage::RelayPayload {
            source_l2_id: UE1_L2_ID,
            destination_l2_id: UE2_L2_ID,
            payload: vec![0xAA; 4],
        })
        .await;
    assert_eq!(relay.task.relay_forwarded_octets(), 4);
    ue2.drain().await;

    // --- UE-2 releases its link, and the relay applies the release ---
    ue2.task
        .handle_message(SidelinkMessage::ReleasePc5Link {
            peer_ue_id: u64::from(RELAY_L2_ID),
        })
        .await;
    assert_eq!(relay.drain().await, 1, "the relay received the release");
    assert_eq!(
        relay.task.link_state(ProseL2Id::new(UE2_L2_ID)),
        Some(Pc5UnicastState::Released),
        "the relay must have applied the peer's DIRECT COMMUNICATION RELEASE"
    );

    // Now the same payload is NOT forwarded.
    relay
        .task
        .handle_message(SidelinkMessage::RelayPayload {
            source_l2_id: UE1_L2_ID,
            destination_l2_id: UE2_L2_ID,
            payload: vec![0xAA; 4],
        })
        .await;
    assert_eq!(
        relay.task.relay_forwarded_octets(),
        4,
        "the counter must not have moved: the link to the destination is released"
    );
}

// ── Issue #190: L2 UE-to-Network relay, end to end ───────────────────────────────

/// A UE configuration for a UE-to-Network relay: it serves an RSC **and** sets
/// `ue_to_network_relay`, which is what selects the L2 U2N architecture over UE-to-UE
/// (issue #190).
fn u2n_relay_config(local_l2_id: u32) -> UeConfig {
    let mut config = prose_config(local_l2_id, APP_CODE, Some(RELAY_SERVICE_CODE));
    if let Some(prose) = config.prose_config.as_mut() {
        prose.ue_to_network_relay = true;
    }
    config
}

/// **The criterion-4 chain**: remote UE → relay UE → the relay's Uu leg towards the gNB,
/// with the payload read back at the far end.
///
/// ```text
/// relay announces, carrying its Relay Service Code       TS 23.304 §6.3.2
///   → remote UE discovers it and opens a unicast link    §6.4.3.1
/// the gNB signals the relay's SRAP bearer mapping        TS 38.331 §5.3.5.17
/// the remote UE sends a payload through the relay
///   → the relay adapts it through SRAP                   TS 38.351, §16.12.2.1
///   → and submits it on its OWN Uu connection            §16.12.2.1
/// ```
///
/// # What makes the assertion load-bearing
///
/// The payload is recovered from the SRAP PDU **at the relay's Uu boundary**, and it is
/// checked that the SRAP header carries the local Remote UE ID the network assigned. Both
/// are values only reachable by having gone through the adaptation layer:
///
/// * a UE-to-UE relay would have produced `RelayForwardDecision::Forward` and sent a
///   `RelayPayload` over PC5 — no `RelayedUplink` at all;
/// * a relay with no signalled SRAP mapping refuses to invent a local Remote UE ID, so the
///   header could not exist;
/// * the payload equality proves SRAP carried the end-to-end PDU unaltered, which is what
///   "PDCP is terminated at the remote UE and the gNB" requires.
#[tokio::test]
async fn a_remote_ue_reaches_the_network_through_a_relays_srap_adaptation_layer() {
    use nextgsim_rlc::srap::{SrapHeader, SRAP_HEADER_LEN};

    // The local Remote UE ID the network assigns. Deliberately not 0 or 1, so a default or
    // an off-by-one cannot produce it.
    const LOCAL_REMOTE_UE_ID: u8 = 42;
    const PAYLOAD: &[u8] = &[0xDE, 0xAD, 0xBE, 0xEF, 0x5A, 0xA5, 0x01];

    let mut relay = SidelinkUe::new(u2n_relay_config(RELAY_L2_ID));
    let mut remote = SidelinkUe::new(prose_config(UE2_L2_ID, APP_CODE, None));
    relay.task.set_pc5_transmitter(remote.inbox.clone());
    remote.task.set_pc5_transmitter(relay.inbox.clone());

    let relay_id = ProseL2Id::new(RELAY_L2_ID);
    let remote_id = ProseL2Id::new(UE2_L2_ID);

    // --- The relay announces; the remote UE discovers it and opens the link on its own ---
    relay
        .task
        .handle_message(SidelinkMessage::StartDiscovery)
        .await;
    assert_eq!(remote.drain().await, 1, "the announcement arrived");
    assert_eq!(relay.drain().await, 1, "the relay received the request");
    assert_eq!(remote.drain().await, 1, "the remote UE received the accept");
    assert_eq!(
        relay.task.link_state(remote_id),
        Some(Pc5UnicastState::Active),
        "precondition: the relay holds a live PC5 link to the remote UE"
    );

    // --- The relay declares itself a relay to the network (`ue-Type-r17`) ---
    //
    // Read off the real `SidelinkInterestChanged` the task sent. This is what asks the gNB
    // for relay resources, so if it is absent the whole network-side configuration has no
    // trigger.
    assert_eq!(
        relay.declared_relay_role(),
        Some(SidelinkRelayRole::Relay),
        "the relay must declare ue-Type-r17 = relayUE, or the gNB is never asked for \
         relay resources"
    );

    // --- The network signals the relay's SRAP bearer mapping (TS 38.331 §5.3.5.17) ---
    //
    // Sent as the message the UE's RRC task sends on decoding an `sl-L2RelayUE-Config`.
    // `nextgsim-gnb`'s `relay_configs_for_remote_ue` builds that IE and
    // `nextgsim-rrc`'s round-trip tests prove it survives UPER; what this drives is the
    // effect of it arriving.
    relay
        .task
        .handle_message(SidelinkMessage::SrapMappingConfigured {
            remote_l2_id: UE2_L2_ID,
            local_remote_ue_id: LOCAL_REMOTE_UE_ID,
            is_relay: true,
        })
        .await;
    assert_eq!(
        relay.task.srap_local_id_for(UE2_L2_ID),
        Some(LOCAL_REMOTE_UE_ID),
        "the relay must hold the local Remote UE ID the network assigned"
    );
    assert_eq!(
        relay.task.srap_adapted_payload_octets(),
        0,
        "precondition: nothing adapted yet"
    );

    // --- The remote UE's traffic goes through the relay, towards the network ---
    //
    // The destination is UE-1, a Layer-2 ID the relay holds NO link to — deliberately,
    // because a UE-to-Network relay does not forward to a peer. A UE-to-UE relay would drop
    // this.
    relay
        .task
        .handle_message(SidelinkMessage::RelayPayload {
            source_l2_id: UE2_L2_ID,
            destination_l2_id: UE1_L2_ID,
            payload: PAYLOAD.to_vec(),
        })
        .await;

    // The SRAP layer carried it.
    assert_eq!(
        relay.task.srap_adapted_payload_octets(),
        PAYLOAD.len() as u64,
        "SRAP must have adapted the remote UE's payload"
    );
    assert_eq!(relay.task.srap_submitted_pdus(), 1);

    // --- The far end: the SRAP PDU left on the RELAY's own Uu connection ---
    let uplinks = relay.take_relayed_uplinks();
    assert_eq!(
        uplinks.len(),
        1,
        "exactly one SRAP PDU must have left the relay's Uu leg; a UE-to-UE relay would \
         have sent a PC5 RelayPayload and none of these"
    );
    let (reported_remote, pdu) = &uplinks[0];
    assert_eq!(*reported_remote, UE2_L2_ID);
    assert_eq!(pdu.len(), SRAP_HEADER_LEN + PAYLOAD.len());

    // **THE assertion**: the payload survives the adaptation layer, and the header names
    // the remote UE by the identity the NETWORK assigned (TS 38.300 §16.12.2.1).
    let (header, carried) = SrapHeader::decode(pdu).expect("the relay built a decodable SRAP PDU");
    assert_eq!(
        header.local_remote_ue_id, LOCAL_REMOTE_UE_ID,
        "the SRAP header must carry the gNB-assigned local Remote UE ID, which is what the \
         gNB correlates the traffic by"
    );
    assert_eq!(
        carried, PAYLOAD,
        "the remote UE's end-to-end payload must survive the relay unaltered: PDCP is \
         terminated at the remote UE and the gNB, not at the relay"
    );

    // And the relay forwarded NOTHING over PC5 — the traffic went to the network, not to a
    // peer. Without this the test would pass against a relay that did both.
    assert_eq!(
        remote.drain().await,
        0,
        "nothing may have gone back out over PC5: the destination was the network"
    );
}

/// A relay with **no signalled SRAP mapping** adapts nothing, even with a live PC5 link.
///
/// The guard that proves the mapping really comes from the network: without it the E2E
/// above would pass against a relay that invented a local Remote UE ID, which would give
/// the gNB a header naming a remote UE it has assigned no identity to.
#[tokio::test]
async fn a_relay_with_no_signalled_srap_mapping_adapts_nothing() {
    let mut relay = SidelinkUe::new(u2n_relay_config(RELAY_L2_ID));
    let mut remote = SidelinkUe::new(prose_config(UE2_L2_ID, APP_CODE, None));
    relay.task.set_pc5_transmitter(remote.inbox.clone());
    remote.task.set_pc5_transmitter(relay.inbox.clone());

    // Establish the link, exactly as above.
    relay
        .task
        .handle_message(SidelinkMessage::StartDiscovery)
        .await;
    remote.drain().await;
    relay.drain().await;
    remote.drain().await;
    assert_eq!(
        relay.task.link_state(ProseL2Id::new(UE2_L2_ID)),
        Some(Pc5UnicastState::Active),
        "precondition: the link is up, so the ONLY thing missing is the mapping"
    );
    assert_eq!(relay.task.srap_local_id_for(UE2_L2_ID), None);

    // No `SrapMappingConfigured` is sent.
    relay
        .task
        .handle_message(SidelinkMessage::RelayPayload {
            source_l2_id: UE2_L2_ID,
            destination_l2_id: UE1_L2_ID,
            payload: vec![0xAA; 8],
        })
        .await;

    assert_eq!(
        relay.task.srap_adapted_payload_octets(),
        0,
        "with no network-signalled mapping the relay must adapt nothing rather than \
         inventing a local Remote UE ID"
    );
    assert!(
        relay.take_relayed_uplinks().is_empty(),
        "and nothing may reach the relay's Uu leg"
    );
}

/// Releasing the remote UE's PC5 link stops the adaptation, so the UE-to-Network path
/// consults the LIVE link state.
///
/// Without this, the E2E would pass against a relay that adapted traffic from any UE it had
/// ever held a link to.
#[tokio::test]
async fn releasing_the_remote_ues_link_stops_the_srap_adaptation() {
    let mut relay = SidelinkUe::new(u2n_relay_config(RELAY_L2_ID));
    let mut remote = SidelinkUe::new(prose_config(UE2_L2_ID, APP_CODE, None));
    relay.task.set_pc5_transmitter(remote.inbox.clone());
    remote.task.set_pc5_transmitter(relay.inbox.clone());

    relay
        .task
        .handle_message(SidelinkMessage::StartDiscovery)
        .await;
    remote.drain().await;
    relay.drain().await;
    remote.drain().await;
    relay
        .task
        .handle_message(SidelinkMessage::SrapMappingConfigured {
            remote_l2_id: UE2_L2_ID,
            local_remote_ue_id: 7,
            is_relay: true,
        })
        .await;

    // One successful adaptation, so the counter is known to move at all.
    relay
        .task
        .handle_message(SidelinkMessage::RelayPayload {
            source_l2_id: UE2_L2_ID,
            destination_l2_id: UE1_L2_ID,
            payload: vec![0xAA; 4],
        })
        .await;
    assert_eq!(relay.task.srap_adapted_payload_octets(), 4);
    assert_eq!(relay.take_relayed_uplinks().len(), 1);

    // --- The remote UE releases its link ---
    remote
        .task
        .handle_message(SidelinkMessage::ReleasePc5Link {
            peer_ue_id: u64::from(RELAY_L2_ID),
        })
        .await;
    assert_eq!(relay.drain().await, 1, "the relay received the release");

    relay
        .task
        .handle_message(SidelinkMessage::RelayPayload {
            source_l2_id: UE2_L2_ID,
            destination_l2_id: UE1_L2_ID,
            payload: vec![0xAA; 4],
        })
        .await;
    assert_eq!(
        relay.task.srap_adapted_payload_octets(),
        4,
        "the counter must not have moved: the remote UE's link is released, so it is no \
         longer a served remote UE"
    );
    assert!(
        relay.take_relayed_uplinks().is_empty(),
        "and nothing further may reach the Uu leg"
    );
}

/// A UE that has discovered and linked to a relay declares itself a **remote** UE, which is
/// what makes the gNB assign it a local Remote UE ID (TS 38.300 §16.12.2.1).
///
/// The other half of the `ue-Type-r17` trigger: without it a remote UE would never be given
/// the identity it must write in its own SRAP headers.
#[tokio::test]
async fn a_ue_linked_to_a_relay_declares_itself_a_remote_ue() {
    let mut relay = SidelinkUe::new(u2n_relay_config(RELAY_L2_ID));
    let mut remote = SidelinkUe::new(prose_config(UE2_L2_ID, APP_CODE, None));
    relay.task.set_pc5_transmitter(remote.inbox.clone());
    remote.task.set_pc5_transmitter(relay.inbox.clone());

    // Before discovering anything, the UE claims nothing: it is a plain PC5 UE.
    remote
        .task
        .handle_message(SidelinkMessage::StartDiscovery)
        .await;
    assert_eq!(
        remote.declared_relay_role(),
        None,
        "a UE with no link to a relay must not claim to be a remote UE"
    );
    relay.drain().await;

    // Now the relay announces and the link comes up.
    relay
        .task
        .handle_message(SidelinkMessage::StartDiscovery)
        .await;
    remote.drain().await;
    relay.drain().await;
    remote.drain().await;
    assert_eq!(
        remote.task.link_state(ProseL2Id::new(RELAY_L2_ID)),
        Some(Pc5UnicastState::Active),
        "precondition: the remote UE holds a link to the relay"
    );

    assert_eq!(
        remote.declared_relay_role(),
        Some(SidelinkRelayRole::Remote),
        "a UE holding a live link to a relay must declare ue-Type-r17 = remoteUE, or the \
         gNB never assigns it a local Remote UE ID"
    );
}

/// With `prose_enabled: false` the task compiles in and does nothing: no announcement
/// leaves, no peer is discovered, no link opens.
///
/// This is the runtime half of the gate issue #141 added, and the test that makes
/// `UeConfig::prose_enabled` a read flag rather than the write-only one it had been.
#[tokio::test]
async fn a_ue_with_prose_disabled_runs_no_pc5_procedures() {
    let disabled = UeConfig {
        prose_enabled: false,
        prose_config: Some(ProseConfig {
            local_l2_id: UE1_L2_ID,
            prose_app_code: APP_CODE,
            ..ProseConfig::default()
        }),
        ..UeConfig::default()
    };
    let mut ue1 = SidelinkUe::new(disabled);
    let mut ue2 = SidelinkUe::new(prose_config(UE2_L2_ID, APP_CODE, None));
    ue1.task.set_pc5_transmitter(ue2.inbox.clone());

    ue1.task
        .handle_message(SidelinkMessage::StartDiscovery)
        .await;
    assert!(
        !ue1.task.discovery_active(),
        "prose_enabled is false, so discovery must not start"
    );
    assert_eq!(
        ue2.drain().await,
        0,
        "nothing may be transmitted by a UE with ProSe disabled"
    );
    assert_eq!(ue2.task.discovered_peer_count(), 0);
}
