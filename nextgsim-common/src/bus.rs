//! In-process topic pub/sub bus for cross-cutting 6G measurements and events
//!
//! # Why this exists
//!
//! Every task in this simulator talks over a point-to-point, typed `mpsc`
//! channel, which suits protocol tasks: the RRC task has exactly one NGAP task
//! to talk to, and the message enum says what it may say. Cross-cutting 6G
//! integration does not fit that shape. A radio measurement is interesting to
//! NWDAF, to ISAC and to the agent framework at once, and with point-to-point
//! wiring the producer must hold a `TaskHandle` per consumer and a distinct
//! message variant per consumer — so **adding a second subscriber means editing
//! the producer**.
//!
//! This is an *additive* fan-out primitive, not a replacement for the task model.
//! Producers keep their existing channels; the bus is an optional second path.
//!
//! # When to use which
//!
//! - **A direct channel** when the message is a *request* — something the
//!   recipient must act on, where the producer needs to know it was delivered and
//!   where losing it is a defect. All protocol signalling is this.
//! - **The bus** when the message is an *observation* — something several
//!   consumers may find interesting, where a slow consumer missing a sample is
//!   acceptable. Measurements, sensing data and analytics inputs are this.
//!
//! The distinction is load-bearing, because the bus **drops** rather than blocks:
//! see [`EventBus::publish`].
//!
//! # Non-normative
//!
//! 6G has no frozen Rel-20 Stage-3 specification, so nothing here is a
//! conformance item. The design is informed by TR 22.870's network-knowledge use
//! cases and by TS 23.288's event-exposure model as prior art, but implements
//! neither.
//!
//! # Not to be confused with
//!
//! `nextgsim-nkef`'s `EventBus` is a different thing: a synchronous,
//! owner-serialised dispatcher over knowledge-graph mutations
//! (`EntityCreated`, `RelationshipAdded`, …) that needs `&mut self` to drain and
//! is internal to the knowledge graph. It is not a cloneable multi-consumer async
//! handle and carries no measurement payloads.

use std::time::Instant;

use tokio::sync::broadcast;

/// Default channel depth. Sixty-four samples is a few seconds of a
/// per-second measurement cadence, which is long enough that a consumer doing
/// real work between receives does not lag, and short enough that a stalled
/// consumer does not pin unbounded memory.
pub const DEFAULT_BUS_CAPACITY: usize = 64;

/// What an event is about.
///
/// Deliberately narrow: a small closed set plus [`Topic::Custom`] for a
/// producer that has something genuinely new to say. Extend the enum when a
/// topic earns a name, rather than routing everything through `Custom`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Topic {
    /// Radio measurements — RSRP/RSRQ/SINR and friends.
    RadioMeasurement,
    /// Integrated sensing (ISAC) data: ranging, positions, detections.
    SensingData,
    /// Analytics inputs and outputs (NWDAF-shaped).
    Analytics,
    /// Energy-saving and power-consumption observations.
    Energy,
    /// Anything else, named by a `'static` string so a topic cannot be
    /// misspelled at runtime into a silently separate stream.
    Custom(&'static str),
}

impl Topic {
    /// A stable name for logging and for receiver-side filtering.
    pub fn name(&self) -> &'static str {
        match self {
            Topic::RadioMeasurement => "radio-measurement",
            Topic::SensingData => "sensing-data",
            Topic::Analytics => "analytics",
            Topic::Energy => "energy",
            Topic::Custom(name) => name,
        }
    }
}

/// One observation on the bus.
///
/// The payload shape mirrors the `measurement_type: String, measurements:
/// Vec<f32>` pair the ISAC and NWDAF messages already use, so a producer can
/// publish alongside its existing point-to-point send without reshaping anything.
#[derive(Debug, Clone)]
pub struct BusEvent {
    /// What this event is about
    pub topic: Topic,
    /// Who published it — a task or entity name, for attribution in logs and for
    /// a consumer that only cares about one source.
    pub source: String,
    /// What kind of measurement the values are, e.g. `"rsrp"`, `"range"`.
    pub measurement_type: String,
    /// The values themselves.
    pub measurements: Vec<f32>,
    /// When it was published. Monotonic, so it is usable for ordering and
    /// intervals but is not a wall-clock time.
    pub timestamp: Instant,
}

impl BusEvent {
    /// An event on `topic` from `source`, carrying `measurements` of
    /// `measurement_type`.
    pub fn new(
        topic: Topic,
        source: impl Into<String>,
        measurement_type: impl Into<String>,
        measurements: Vec<f32>,
    ) -> Self {
        Self {
            topic,
            source: source.into(),
            measurement_type: measurement_type.into(),
            measurements,
            timestamp: Instant::now(),
        }
    }
}

/// A cloneable handle to an in-process broadcast bus.
///
/// Cloning shares the same bus; every clone can publish, and a subscriber taken
/// from any clone sees every publication from any of them.
#[derive(Debug, Clone)]
pub struct EventBus {
    tx: broadcast::Sender<BusEvent>,
}

impl EventBus {
    /// A bus holding up to `capacity` events per subscriber.
    ///
    /// # Panics
    ///
    /// If `capacity` is zero, which `tokio::sync::broadcast` rejects: a bus that
    /// can hold nothing delivers nothing.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "an EventBus needs a capacity of at least 1");
        let (tx, _rx) = broadcast::channel(capacity);
        Self { tx }
    }

    /// Publish an event to every current subscriber.
    ///
    /// Returns the number of subscribers it was delivered to, which is **0 when
    /// there are none** — not an error. A bus is a place to put observations, and
    /// whether anyone is listening is not the producer's problem; making it one
    /// would have every producer handle a failure that means nothing.
    ///
    /// **Never blocks and never applies backpressure.** A subscriber that falls
    /// more than `capacity` events behind loses the oldest ones and is told so by
    /// a [`broadcast::error::RecvError::Lagged`] on its next receive. That is the
    /// deliberate trade: a stalled analytics consumer must not stall the radio
    /// task that feeds it. Anything that must not be lost belongs on a direct
    /// channel.
    pub fn publish(&self, event: BusEvent) -> usize {
        // `send` errors only when there are no receivers, which is not a failure
        // here. Nothing else can go wrong: a full channel drops the oldest.
        self.tx.send(event).unwrap_or(0)
    }

    /// Subscribe. The receiver sees events published **after** this call.
    ///
    /// Filtering is receiver-side: match on [`BusEvent::topic`]. A per-topic
    /// channel would be the next step if one consumer ever cares about a small
    /// slice of a high-rate topic, and is deliberately not built yet.
    pub fn subscribe(&self) -> broadcast::Receiver<BusEvent> {
        self.tx.subscribe()
    }

    /// How many subscribers are listening. For logging and tests; a producer
    /// should not branch on it, since a subscriber can appear or vanish between
    /// the check and the publish.
    pub fn subscriber_count(&self) -> usize {
        self.tx.receiver_count()
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new(DEFAULT_BUS_CAPACITY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::broadcast::error::{RecvError, TryRecvError};

    fn event(measurement: f32) -> BusEvent {
        BusEvent::new(
            Topic::SensingData,
            "test-producer",
            "range",
            vec![measurement],
        )
    }

    fn runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime")
    }

    /// One publish reaches every subscriber, which is the whole point: with
    /// point-to-point channels the producer would need a handle per consumer.
    #[test]
    fn one_publish_reaches_every_subscriber() {
        let bus = EventBus::new(8);
        let mut first = bus.subscribe();
        let mut second = bus.subscribe();
        let mut third = bus.subscribe();
        assert_eq!(bus.subscriber_count(), 3);

        assert_eq!(bus.publish(event(1.5)), 3, "delivered to all three");

        for (label, rx) in [
            ("first", &mut first),
            ("second", &mut second),
            ("third", &mut third),
        ] {
            let received = rx
                .try_recv()
                .unwrap_or_else(|e| panic!("{label} subscriber must receive the event, got {e:?}"));
            assert_eq!(received.measurements, vec![1.5]);
            assert_eq!(received.topic, Topic::SensingData);
            assert_eq!(received.source, "test-producer");
        }
    }

    /// Publishing with nobody listening is a no-op, not an error: a producer must
    /// not have to care whether a consumer has been wired up.
    #[test]
    fn publishing_with_no_subscribers_is_not_an_error() {
        let bus = EventBus::new(8);
        assert_eq!(bus.subscriber_count(), 0);
        assert_eq!(bus.publish(event(1.0)), 0);
        assert_eq!(bus.publish(event(2.0)), 0, "and again");
    }

    /// A subscriber that goes away leaves the bus usable.
    #[test]
    fn a_dropped_subscriber_does_not_break_the_bus() {
        let bus = EventBus::new(8);
        let rx = bus.subscribe();
        let mut survivor = bus.subscribe();
        drop(rx);

        assert_eq!(bus.publish(event(3.0)), 1);
        assert_eq!(
            survivor.try_recv().expect("delivered").measurements,
            vec![3.0]
        );
    }

    /// The trade-off, asserted: a subscriber that falls behind loses the oldest
    /// events and is **told** so by `Lagged`, then carries on from what is still
    /// held — and the publisher was never blocked.
    #[test]
    fn a_lagging_subscriber_observes_lagged_and_recovers() {
        let bus = EventBus::new(2);
        let mut slow = bus.subscribe();

        // Five publishes into a two-deep channel. Every one returns immediately:
        // the publisher is not blocked by the subscriber that is not reading.
        for i in 0..5 {
            assert_eq!(
                bus.publish(event(i as f32)),
                1,
                "publish {i} is not blocked"
            );
        }

        match slow.try_recv() {
            Err(TryRecvError::Lagged(skipped)) => {
                assert_eq!(skipped, 3, "the three oldest of five were dropped");
            }
            other => panic!("expected Lagged, got {other:?}"),
        }

        // And it recovers: the two still held arrive in order.
        assert_eq!(
            slow.try_recv().expect("recovered").measurements,
            vec![3.0],
            "the oldest retained event"
        );
        assert_eq!(slow.try_recv().expect("recovered").measurements, vec![4.0]);
        assert!(matches!(slow.try_recv(), Err(TryRecvError::Empty)));
    }

    /// A lagging subscriber does not cost a keeping-up one anything.
    #[test]
    fn one_subscriber_lagging_does_not_affect_another() {
        let bus = EventBus::new(2);
        let mut slow = bus.subscribe();
        let mut fast = bus.subscribe();

        bus.publish(event(0.0));
        assert_eq!(
            fast.try_recv().expect("fast keeps up").measurements,
            vec![0.0]
        );

        for i in 1..4 {
            bus.publish(event(i as f32));
            assert_eq!(
                fast.try_recv().expect("fast still keeps up").measurements,
                vec![i as f32]
            );
        }

        assert!(
            matches!(slow.try_recv(), Err(TryRecvError::Lagged(_))),
            "the slow one lagged"
        );
        assert!(
            matches!(fast.try_recv(), Err(TryRecvError::Empty)),
            "the fast one saw everything and lagged on nothing"
        );
    }

    /// A clone shares the bus rather than making a second one.
    #[test]
    fn a_cloned_bus_publishes_to_the_same_subscribers() {
        let bus = EventBus::new(8);
        let mut rx = bus.subscribe();
        let clone = bus.clone();

        assert_eq!(clone.publish(event(9.0)), 1);
        assert_eq!(rx.try_recv().expect("delivered").measurements, vec![9.0]);
    }

    /// Receiver-side filtering: a consumer interested in one topic ignores the
    /// rest, which is how the bus stays a single channel.
    #[test]
    fn a_consumer_can_filter_by_topic() {
        let bus = EventBus::new(8);
        let mut rx = bus.subscribe();

        bus.publish(BusEvent::new(Topic::Analytics, "a", "load", vec![0.5]));
        bus.publish(BusEvent::new(Topic::SensingData, "b", "range", vec![1.0]));
        bus.publish(BusEvent::new(Topic::Energy, "c", "watts", vec![2.0]));

        let mut sensing = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            if ev.topic == Topic::SensingData {
                sensing.push(ev);
            }
        }
        assert_eq!(sensing.len(), 1);
        assert_eq!(sensing[0].source, "b");
    }

    /// `Topic::Custom` is a `&'static str`, so a topic cannot be misspelled into a
    /// silently separate stream at runtime.
    #[test]
    fn a_custom_topic_matches_only_itself() {
        assert_eq!(Topic::Custom("mine").name(), "mine");
        assert_eq!(Topic::Custom("mine"), Topic::Custom("mine"));
        assert_ne!(Topic::Custom("mine"), Topic::Custom("yours"));
        assert_ne!(Topic::Custom("sensing-data"), Topic::SensingData);
    }

    /// The async receive path works too, not only `try_recv`.
    #[test]
    fn an_awaiting_subscriber_receives() {
        let bus = EventBus::new(8);
        let mut rx = bus.subscribe();

        runtime().block_on(async {
            bus.publish(event(7.0));
            let received = rx.recv().await.expect("delivered");
            assert_eq!(received.measurements, vec![7.0]);
        });
    }

    /// When every sender is gone the receiver is closed rather than hanging.
    #[test]
    fn dropping_the_bus_closes_its_subscribers() {
        let bus = EventBus::new(8);
        let mut rx = bus.subscribe();
        drop(bus);

        runtime().block_on(async {
            assert!(matches!(rx.recv().await, Err(RecvError::Closed)));
        });
    }

    #[test]
    #[should_panic(expected = "capacity of at least 1")]
    fn a_zero_capacity_bus_is_rejected() {
        let _ = EventBus::new(0);
    }
}
