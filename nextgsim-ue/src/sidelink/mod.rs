//! Sidelink for the UE — **the real PC5 procedures, behind an off-by-default feature**
//!
//! # This header was stale, and that is worth recording
//!
//! Until issue #190 it described "an inert facade" whose handlers had no senders. That was
//! true when issue #54 wrote it and **false from issue #141 onwards** — #141 implemented
//! the procedures and added four of the modules re-exported below, but did not rewrite
//! this paragraph, so a reader was told the opposite of what the code did. Corrected
//! rather than deleted, because understating working code is the same failure mode #141's
//! own honesty criterion warned about.
//!
//! # What runs
//!
//! - **PC5-S** (TS 24.554, [`pc5s`]): `Direct Communication Request`/`Accept`/`Reject`/
//!   `Release` and Model A/B discovery messages, as real octets.
//! - **The unicast link state machine** (TS 23.304 §6.4.3.1, [`link`]): `Active` is
//!   reachable only by decoding a peer's message, and `peer_l2_id` only by reading the
//!   peer's own Layer-2 ID off the wire (step 4).
//! - **Model A and B discovery** (§6.3.1.2, §6.3.1.3, [`discovery`]) with a ProSe
//!   Application Code filter, so a UE on another code is not discovered.
//! - **UE-to-UE relay forwarding** (§6.4.3.10, [`relay`]) decided against the live link
//!   table, so releasing a link stops the forwarding.
//! - **L2 UE-to-Network relay** (TS 38.300 §16.12.2.1, TS 38.351; issue #190): a remote
//!   UE's end-to-end bearers are adapted through the SRAP sublayer
//!   (`nextgsim_rlc::srap`) onto the relay's own Uu relay RLC channels, with the bearer
//!   mapping and the local Remote UE ID signalled by the gNB.
//! - **The RRC half** (TS 38.331 §5.8.3, §5.3.5.17): the UE emits a real UPER
//!   `SidelinkUEInformation` — carrying `ue-Type-r17` when it is a relay or a remote UE —
//!   and applies the `sl-ConfigDedicatedNR`, `sl-L2RelayUE-Config` and
//!   `sl-L2RemoteUE-Config` it gets back.
//!
//! # What does not
//!
//! - **No PC5 radio.** The medium between UEs is an in-process channel, so the PDUs are
//!   real and the propagation is not.
//! - **No L3 relay.** `RelayMode::L3Relay` maps to [`RelayRole::None`] with a `warn!`:
//!   L3 relaying forwards IP packets rather than adapting Layer-2 bearers, and there is no
//!   IP forwarding plane here.
//! - [`Pc5RrcConnection`] is still constructed only by its own unit test;
//!   [`Pc5LinkContext`] superseded it.
//!
//! The module sits behind the `sidelink` Cargo feature, off by default, and is gated again
//! at run time by `UeConfig::prose_enabled`: PC5 is not a capability a default UE should
//! advertise, and the gate keeps this module out of the lean `cargo test --workspace`.
//! Issues #54, #141, #190.

pub mod discovery;
pub mod link;
pub mod pc5;
pub mod pc5s;
pub mod positioning;
pub mod relay;
pub mod task;

pub use discovery::{
    DiscoveredPeer, DiscoveryFilter, DiscoveryModel, DiscoveryOutcome, Pc5DiscoveryEngine,
};
pub use link::{Pc5LinkContext, Pc5LinkError, Pc5LinkTable, Pc5Role, Pc5UnicastState};
pub use pc5::{
    Pc5DiscoveredPeer, Pc5Discovery, Pc5DiscoveryMode, Pc5HarqFeedback, Pc5RadioBearerConfig,
    Pc5ResourceMode, Pc5RrcConnection, Pc5RrcState,
};
pub use pc5s::{
    DirectCommunicationAccept, DirectCommunicationReject, DirectCommunicationRelease,
    DirectCommunicationRequest, DirectDiscoveryMessage, Pc5CastType, Pc5RejectCause, Pc5SError,
    Pc5SMessage, Pc5SMessageType, ProseL2Id, RelayServiceCode,
};
pub use positioning::{
    AnchorUe, AoaMeasurement, AodMeasurement, Position3D, PositionEstimate, RttMeasurement,
    SidelinkPositioningEngine, SlPrsResourceConfig,
};
pub use relay::{RelayForwardDecision, RelayForwarder, RelayRole};
pub use task::{SidelinkTask, SPAWN_LOG, START_LOG};
