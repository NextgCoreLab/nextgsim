//! Event-driven knowledge graph updates
//!
//! Provides an event system for real-time notification and processing of
//! knowledge graph changes. Supports event handler registration with callbacks
//! and an event queue for batch processing.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::fmt;

use crate::temporal::Timestamp;

/// Types of knowledge graph events.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
pub enum KnowledgeEventKind {
    /// A new entity was created in the knowledge graph
    EntityCreated,
    /// An existing entity was updated (properties changed)
    EntityUpdated,
    /// An entity was removed from the knowledge graph
    EntityRemoved,
    /// A new relationship was added
    RelationshipAdded,
    /// A relationship was removed (expired or deleted)
    RelationshipRemoved,
}

impl fmt::Display for KnowledgeEventKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KnowledgeEventKind::EntityCreated => write!(f, "EntityCreated"),
            KnowledgeEventKind::EntityUpdated => write!(f, "EntityUpdated"),
            KnowledgeEventKind::EntityRemoved => write!(f, "EntityRemoved"),
            KnowledgeEventKind::RelationshipAdded => write!(f, "RelationshipAdded"),
            KnowledgeEventKind::RelationshipRemoved => write!(f, "RelationshipRemoved"),
        }
    }
}

/// A knowledge graph event with associated data.
#[derive(Debug, Clone)]
pub struct KnowledgeEvent {
    /// Type of event
    pub kind: KnowledgeEventKind,
    /// Timestamp when the event occurred
    pub timestamp: Timestamp,
    /// Primary entity ID involved in the event
    pub entity_id: String,
    /// Secondary entity ID (for relationship events: the other entity)
    pub related_entity_id: Option<String>,
    /// Relationship type (for relationship events)
    pub relation_type: Option<String>,
    /// Changed properties (for update events: key -> (old_value, new_value))
    pub changed_properties: HashMap<String, (Option<String>, Option<String>)>,
}

impl KnowledgeEvent {
    /// Creates a new `EntityCreated` event.
    pub fn entity_created(entity_id: impl Into<String>, timestamp: Timestamp) -> Self {
        Self {
            kind: KnowledgeEventKind::EntityCreated,
            timestamp,
            entity_id: entity_id.into(),
            related_entity_id: None,
            relation_type: None,
            changed_properties: HashMap::new(),
        }
    }

    /// Creates a new `EntityUpdated` event.
    pub fn entity_updated(
        entity_id: impl Into<String>,
        timestamp: Timestamp,
        changed_properties: HashMap<String, (Option<String>, Option<String>)>,
    ) -> Self {
        Self {
            kind: KnowledgeEventKind::EntityUpdated,
            timestamp,
            entity_id: entity_id.into(),
            related_entity_id: None,
            relation_type: None,
            changed_properties,
        }
    }

    /// Creates a new `EntityRemoved` event.
    pub fn entity_removed(entity_id: impl Into<String>, timestamp: Timestamp) -> Self {
        Self {
            kind: KnowledgeEventKind::EntityRemoved,
            timestamp,
            entity_id: entity_id.into(),
            related_entity_id: None,
            relation_type: None,
            changed_properties: HashMap::new(),
        }
    }

    /// Creates a new `RelationshipAdded` event.
    pub fn relationship_added(
        source_id: impl Into<String>,
        target_id: impl Into<String>,
        relation_type: impl Into<String>,
        timestamp: Timestamp,
    ) -> Self {
        Self {
            kind: KnowledgeEventKind::RelationshipAdded,
            timestamp,
            entity_id: source_id.into(),
            related_entity_id: Some(target_id.into()),
            relation_type: Some(relation_type.into()),
            changed_properties: HashMap::new(),
        }
    }

    /// Creates a new `RelationshipRemoved` event.
    pub fn relationship_removed(
        source_id: impl Into<String>,
        target_id: impl Into<String>,
        relation_type: impl Into<String>,
        timestamp: Timestamp,
    ) -> Self {
        Self {
            kind: KnowledgeEventKind::RelationshipRemoved,
            timestamp,
            entity_id: source_id.into(),
            related_entity_id: Some(target_id.into()),
            relation_type: Some(relation_type.into()),
            changed_properties: HashMap::new(),
        }
    }
}

/// Type alias for event handler callbacks.
///
/// Handlers receive a reference to the event and can perform side effects
/// (logging, metric updates, trigger cascading changes, etc.).
pub type EventHandler = Box<dyn Fn(&KnowledgeEvent) + Send + Sync>;

/// Unique identifier for a registered event handler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HandlerId(u64);

impl fmt::Display for HandlerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HandlerId({})", self.0)
    }
}

/// Event bus for knowledge graph changes.
///
/// Supports registering handlers for specific event kinds and dispatching
/// events to matching handlers. Also maintains an event queue for batch processing.
pub struct EventBus {
    /// Registered handlers, grouped by event kind
    handlers: HashMap<KnowledgeEventKind, Vec<(HandlerId, EventHandler)>>,
    /// Handlers that listen to all event kinds
    global_handlers: Vec<(HandlerId, EventHandler)>,
    /// Event queue for batch processing
    queue: VecDeque<KnowledgeEvent>,
    /// Maximum queue size (oldest events are dropped if exceeded)
    max_queue_size: usize,
    /// Next handler ID
    next_handler_id: u64,
    /// Total events dispatched
    events_dispatched: u64,
}

impl fmt::Debug for EventBus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EventBus")
            .field("handler_count", &self.handler_count())
            .field("queue_len", &self.queue.len())
            .field("max_queue_size", &self.max_queue_size)
            .field("events_dispatched", &self.events_dispatched)
            .finish()
    }
}

impl EventBus {
    /// Creates a new event bus with the given maximum queue size.
    pub fn new(max_queue_size: usize) -> Self {
        Self {
            handlers: HashMap::new(),
            global_handlers: Vec::new(),
            queue: VecDeque::new(),
            max_queue_size,
            next_handler_id: 0,
            events_dispatched: 0,
        }
    }

    /// Registers an event handler for a specific event kind.
    ///
    /// Returns a `HandlerId` that can be used to unregister the handler.
    pub fn on(&mut self, kind: KnowledgeEventKind, handler: EventHandler) -> HandlerId {
        let id = HandlerId(self.next_handler_id);
        self.next_handler_id += 1;
        self.handlers
            .entry(kind)
            .or_default()
            .push((id, handler));
        id
    }

    /// Registers an event handler for all event kinds.
    ///
    /// Returns a `HandlerId` that can be used to unregister the handler.
    pub fn on_all(&mut self, handler: EventHandler) -> HandlerId {
        let id = HandlerId(self.next_handler_id);
        self.next_handler_id += 1;
        self.global_handlers.push((id, handler));
        id
    }

    /// Unregisters an event handler by its ID.
    ///
    /// Returns `true` if the handler was found and removed.
    pub fn remove_handler(&mut self, handler_id: HandlerId) -> bool {
        // Check global handlers
        let initial_global_len = self.global_handlers.len();
        self.global_handlers.retain(|(id, _)| *id != handler_id);
        if self.global_handlers.len() < initial_global_len {
            return true;
        }

        // Check kind-specific handlers
        for handlers in self.handlers.values_mut() {
            let initial_len = handlers.len();
            handlers.retain(|(id, _)| *id != handler_id);
            if handlers.len() < initial_len {
                return true;
            }
        }

        false
    }

    /// Dispatches an event immediately to all matching handlers.
    ///
    /// Calls handlers registered for the event's specific kind, plus all
    /// global handlers. Also enqueues the event for batch processing.
    pub fn dispatch(&mut self, event: KnowledgeEvent) {
        self.events_dispatched += 1;

        // Call kind-specific handlers
        if let Some(handlers) = self.handlers.get(&event.kind) {
            for (_, handler) in handlers {
                handler(&event);
            }
        }

        // Call global handlers
        for (_, handler) in &self.global_handlers {
            handler(&event);
        }

        // Enqueue for batch processing
        self.enqueue(event);
    }

    /// Enqueues an event without dispatching to handlers.
    ///
    /// Useful for collecting events that will be batch-processed later.
    pub fn enqueue(&mut self, event: KnowledgeEvent) {
        if self.queue.len() >= self.max_queue_size {
            self.queue.pop_front(); // Drop oldest
        }
        self.queue.push_back(event);
    }

    /// Drains the event queue, returning all queued events.
    ///
    /// The queue is empty after this call.
    pub fn drain_queue(&mut self) -> Vec<KnowledgeEvent> {
        self.queue.drain(..).collect()
    }

    /// Returns the number of events currently in the queue.
    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }

    /// Returns true if the event queue is empty.
    pub fn queue_is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Returns the total number of registered handlers (kind-specific + global).
    pub fn handler_count(&self) -> usize {
        let kind_count: usize = self.handlers.values().map(std::vec::Vec::len).sum();
        kind_count + self.global_handlers.len()
    }

    /// Returns the total number of events dispatched since creation.
    pub fn events_dispatched(&self) -> u64 {
        self.events_dispatched
    }

    /// Clears all handlers and the event queue.
    pub fn clear(&mut self) {
        self.handlers.clear();
        self.global_handlers.clear();
        self.queue.clear();
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new(10_000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    #[test]
    fn test_knowledge_event_creation() {
        let event = KnowledgeEvent::entity_created("gnb-001", 12345);
        assert_eq!(event.kind, KnowledgeEventKind::EntityCreated);
        assert_eq!(event.entity_id, "gnb-001");
        assert_eq!(event.timestamp, 12345);
    }

    #[test]
    fn test_knowledge_event_updated() {
        let mut changes = HashMap::new();
        changes.insert(
            "status".to_string(),
            (Some("active".to_string()), Some("degraded".to_string())),
        );

        let event = KnowledgeEvent::entity_updated("gnb-001", 12345, changes);
        assert_eq!(event.kind, KnowledgeEventKind::EntityUpdated);
        assert_eq!(event.changed_properties.len(), 1);
    }

    #[test]
    fn test_knowledge_event_relationship() {
        let event = KnowledgeEvent::relationship_added("ue-001", "gnb-001", "connected_to", 100);
        assert_eq!(event.kind, KnowledgeEventKind::RelationshipAdded);
        assert_eq!(event.entity_id, "ue-001");
        assert_eq!(event.related_entity_id, Some("gnb-001".to_string()));
        assert_eq!(event.relation_type, Some("connected_to".to_string()));
    }

    #[test]
    fn test_event_bus_handler_registration() {
        let mut bus = EventBus::new(100);

        let counter = Arc::new(AtomicU64::new(0));
        let counter_clone = counter.clone();

        let handler_id = bus.on(
            KnowledgeEventKind::EntityCreated,
            Box::new(move |_event| {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            }),
        );

        assert_eq!(bus.handler_count(), 1);

        // Dispatch a matching event
        bus.dispatch(KnowledgeEvent::entity_created("gnb-001", 100));
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        // Dispatch a non-matching event
        bus.dispatch(KnowledgeEvent::entity_removed("gnb-001", 200));
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        // Remove handler
        assert!(bus.remove_handler(handler_id));
        assert_eq!(bus.handler_count(), 0);

        // Dispatch again - should not increment
        bus.dispatch(KnowledgeEvent::entity_created("gnb-002", 300));
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_event_bus_global_handler() {
        let mut bus = EventBus::new(100);

        let counter = Arc::new(AtomicU64::new(0));
        let counter_clone = counter.clone();

        bus.on_all(Box::new(move |_event| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        }));

        bus.dispatch(KnowledgeEvent::entity_created("a", 1));
        bus.dispatch(KnowledgeEvent::entity_updated("b", 2, HashMap::new()));
        bus.dispatch(KnowledgeEvent::entity_removed("c", 3));

        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_event_bus_queue() {
        let mut bus = EventBus::new(100);

        bus.dispatch(KnowledgeEvent::entity_created("a", 1));
        bus.dispatch(KnowledgeEvent::entity_created("b", 2));
        bus.dispatch(KnowledgeEvent::entity_created("c", 3));

        assert_eq!(bus.queue_len(), 3);

        let events = bus.drain_queue();
        assert_eq!(events.len(), 3);
        assert!(bus.queue_is_empty());
    }

    #[test]
    fn test_event_bus_queue_overflow() {
        let mut bus = EventBus::new(3);

        bus.enqueue(KnowledgeEvent::entity_created("a", 1));
        bus.enqueue(KnowledgeEvent::entity_created("b", 2));
        bus.enqueue(KnowledgeEvent::entity_created("c", 3));
        bus.enqueue(KnowledgeEvent::entity_created("d", 4));

        assert_eq!(bus.queue_len(), 3);

        let events = bus.drain_queue();
        // "a" should have been dropped (oldest)
        assert_eq!(events[0].entity_id, "b");
        assert_eq!(events[1].entity_id, "c");
        assert_eq!(events[2].entity_id, "d");
    }

    #[test]
    fn test_event_bus_events_dispatched() {
        let mut bus = EventBus::new(100);
        assert_eq!(bus.events_dispatched(), 0);

        bus.dispatch(KnowledgeEvent::entity_created("a", 1));
        bus.dispatch(KnowledgeEvent::entity_created("b", 2));

        assert_eq!(bus.events_dispatched(), 2);
    }

    #[test]
    fn test_event_bus_clear() {
        let mut bus = EventBus::new(100);

        bus.on(
            KnowledgeEventKind::EntityCreated,
            Box::new(|_| {}),
        );
        bus.on_all(Box::new(|_| {}));
        bus.enqueue(KnowledgeEvent::entity_created("a", 1));

        assert_eq!(bus.handler_count(), 2);
        assert_eq!(bus.queue_len(), 1);

        bus.clear();
        assert_eq!(bus.handler_count(), 0);
        assert!(bus.queue_is_empty());
    }

    #[test]
    fn test_event_kind_display() {
        assert_eq!(
            format!("{}", KnowledgeEventKind::EntityCreated),
            "EntityCreated"
        );
        assert_eq!(
            format!("{}", KnowledgeEventKind::RelationshipRemoved),
            "RelationshipRemoved"
        );
    }

    #[test]
    fn test_handler_id_display() {
        let id = HandlerId(42);
        assert_eq!(format!("{id}"), "HandlerId(42)");
    }
}
