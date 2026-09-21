//! Sidelink module for UE — **an inert facade, behind an off-by-default feature**
//!
//! Models the shapes of NR sidelink relay (UE-to-UE relay), sidelink discovery,
//! PC5 link establishment and sidelink positioning (TS 23.304). None of it is
//! driven:
//!
//! - Only `SidelinkMessage::StartDiscovery` / `StopDiscovery` have a sender
//!   anywhere in the tree, so `EstablishPc5Link`, `PeerDiscovered`,
//!   `SetRelayMode`, `RelayData`, `PositioningMeasurement` and
//!   `CooperativePositioning` are unreachable handlers.
//! - `EstablishPc5Link` flips `Establishing` → `Active` with no over-the-air
//!   exchange: no PC5-S `Direct Communication Request`/`Accept`
//!   (TS 23.304 §6.4.3.1) is ever sent or awaited.
//! - There is no RRC `SidelinkUEInformation` and no `sl-Config` in
//!   `RRCReconfiguration` (TS 38.331), so a gNB peer reserves no sidelink
//!   resources.
//! - [`Pc5RrcConnection`] is constructed only by its own unit test.
//!
//! The whole module therefore sits behind the `sidelink` Cargo feature, which is
//! off by default, so a default build neither carries these types nor logs a
//! capability claim about them. Enabling the feature compiles the facade back
//! in; it does not make PC5 work. Issue #54 tracks that, and its increment 2 is
//! the real procedures.

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
