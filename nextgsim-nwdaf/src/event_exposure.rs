//! Event Exposure Service for NWDAF DCCF
//!
//! Implements event exposure from NFs (AMF, SMF, UPF) for data collection
//! as defined in 3GPP TS 23.288.
//!
//! # 3GPP Reference
//!
//! - TS 23.288: Architecture enhancements for 5G System to support network data analytics
//! - TS 29.520: Nnwdaf_EventsSubscription Service API

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};

use crate::error::NwdafError;

/// Event type for NF exposure
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventType {
    /// Load level information
    LoadLevel,
    /// Network performance
    NetworkPerformance,
    /// NF load
    NfLoad,
    /// Service experience
    ServiceExperience,
    /// UE mobility
    UeMobility,
    /// UE communication
    UeCommunication,
    /// QoS sustainability
    QosSustainability,
    /// Abnormal behavior
    AbnormalBehavior,
    /// User data congestion
    UserDataCongestion,
    /// DN performance
    DnPerformance,
    /// Dispersion analytics
    Dispersion,
    /// Red traffic congestion
    RedTransCongestion,
    /// WLAN performance
    WlanPerformance,
}

/// Event subscription
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventSubscription {
    /// Subscription ID
    pub subscription_id: String,
    /// Event type
    pub event_type: EventType,
    /// NF type that provides this event
    pub nf_type: NfType,
    /// Target area (optional)
    pub target_area: Option<TargetArea>,
    /// Notification target (URI)
    pub notification_target: String,
    /// Subscription status
    pub status: SubscriptionStatus,
    /// Reporting interval in seconds
    pub reporting_interval_s: Option<u32>,
    /// Creation time
    pub created_ms: u64,
}

/// NF type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NfType {
    /// AMF
    Amf,
    /// SMF
    Smf,
    /// UPF
    Upf,
    /// PCF
    Pcf,
    /// UDM
    Udm,
    /// AUSF
    Ausf,
    /// AF
    Af,
}

/// Subscription status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubscriptionStatus {
    /// Active
    Active,
    /// Suspended
    Suspended,
    /// Terminated
    Terminated,
}

/// Target area for event collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetArea {
    /// List of TAC (Tracking Area Code)
    pub tac_list: Vec<u32>,
    /// List of cell IDs
    pub cell_id_list: Vec<u64>,
}

/// Event notification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventNotification {
    /// Subscription ID
    pub subscription_id: String,
    /// Event type
    pub event_type: EventType,
    /// NF instance ID that generated the event
    pub nf_instance_id: String,
    /// Event data
    pub event_data: EventData,
    /// Timestamp
    pub timestamp_ms: u64,
}

/// Event data payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventData {
    /// Load level event
    LoadLevel {
        /// Load level percentage (0-100)
        load_percentage: u8,
        /// Number of registered UEs
        registered_ues: u32,
    },
    /// Network performance event
    NetworkPerformance {
        /// Average packet delay (ms)
        packet_delay_ms: f64,
        /// Packet loss rate (0.0-1.0)
        packet_loss_rate: f64,
        /// Throughput (Mbps)
        throughput_mbps: f64,
    },
    /// NF load event
    NfLoad {
        /// CPU usage percentage (0-100)
        cpu_usage: u8,
        /// Memory usage percentage (0-100)
        memory_usage: u8,
        /// Active sessions
        active_sessions: u32,
    },
    /// Service experience event
    ServiceExperience {
        /// Service ID
        service_id: u32,
        /// Mean opinion score (1.0-5.0)
        mos: f32,
        /// Number of users
        num_users: u32,
    },
    /// UE mobility event
    UeMobility {
        /// UE identifier
        ue_id: u64,
        /// Current cell ID
        cell_id: u64,
        /// Previous cell ID
        prev_cell_id: Option<u64>,
        /// Mobility state
        mobility_state: MobilityState,
    },
    /// Generic event data
    Generic {
        /// Key-value pairs
        data: HashMap<String, String>,
    },
}

/// UE mobility state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MobilityState {
    /// Stationary
    Stationary,
    /// Normal mobility
    Normal,
    /// High mobility
    High,
}

/// Event exposure manager
pub struct EventExposureManager {
    /// Active subscriptions
    subscriptions: HashMap<String, EventSubscription>,
    /// Received event notifications
    notifications: Vec<EventNotification>,
    /// Next subscription ID counter
    next_subscription_id: u64,
}

impl EventExposureManager {
    /// Creates a new event exposure manager
    pub fn new() -> Self {
        Self {
            subscriptions: HashMap::new(),
            notifications: Vec::new(),
            next_subscription_id: 1,
        }
    }

    /// Creates a new event subscription
    pub fn create_subscription(
        &mut self,
        event_type: EventType,
        nf_type: NfType,
        notification_target: String,
        reporting_interval_s: Option<u32>,
    ) -> EventSubscription {
        let subscription_id = format!("sub-{}", self.next_subscription_id);
        self.next_subscription_id += 1;

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let subscription = EventSubscription {
            subscription_id: subscription_id.clone(),
            event_type,
            nf_type,
            target_area: None,
            notification_target,
            status: SubscriptionStatus::Active,
            reporting_interval_s,
            created_ms: now_ms,
        };

        info!(
            "Created event subscription: id={}, type={:?}, nf_type={:?}",
            subscription_id, event_type, nf_type
        );

        self.subscriptions
            .insert(subscription_id.clone(), subscription.clone());
        subscription
    }

    /// Gets a subscription by ID
    pub fn get_subscription(&self, subscription_id: &str) -> Option<&EventSubscription> {
        self.subscriptions.get(subscription_id)
    }

    /// Updates subscription status
    pub fn update_subscription_status(
        &mut self,
        subscription_id: &str,
        status: SubscriptionStatus,
    ) -> Result<(), NwdafError> {
        let subscription = self.subscriptions.get_mut(subscription_id).ok_or_else(|| {
            crate::error::DataCollectionError::InvalidData {
                reason: format!("Subscription {subscription_id} not found"),
            }
        })?;

        subscription.status = status;
        debug!(
            "Updated subscription {} status to {:?}",
            subscription_id, status
        );

        Ok(())
    }

    /// Deletes a subscription
    pub fn delete_subscription(&mut self, subscription_id: &str) -> Result<(), NwdafError> {
        self.subscriptions
            .remove(subscription_id)
            .ok_or_else(|| crate::error::DataCollectionError::InvalidData {
                reason: format!("Subscription {subscription_id} not found"),
            })?;

        info!("Deleted event subscription: {}", subscription_id);
        Ok(())
    }

    /// Receives an event notification
    pub fn receive_notification(&mut self, notification: EventNotification) {
        debug!(
            "Received event notification: type={:?}, nf={}",
            notification.event_type, notification.nf_instance_id
        );

        self.notifications.push(notification);
    }

    /// Gets all notifications for a subscription
    pub fn get_notifications_for_subscription(
        &self,
        subscription_id: &str,
    ) -> Vec<&EventNotification> {
        self.notifications
            .iter()
            .filter(|n| n.subscription_id == subscription_id)
            .collect()
    }

    /// Gets all active subscriptions
    pub fn list_active_subscriptions(&self) -> Vec<&EventSubscription> {
        self.subscriptions
            .values()
            .filter(|s| s.status == SubscriptionStatus::Active)
            .collect()
    }

    /// Clears old notifications (older than retention period)
    pub fn prune_old_notifications(&mut self, retention_ms: u64) {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let before_count = self.notifications.len();
        self.notifications
            .retain(|n| now_ms - n.timestamp_ms < retention_ms);

        let removed = before_count - self.notifications.len();
        if removed > 0 {
            debug!("Pruned {} old notifications", removed);
        }
    }
}

impl Default for EventExposureManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_subscription() {
        let mut manager = EventExposureManager::new();

        let sub = manager.create_subscription(
            EventType::LoadLevel,
            NfType::Amf,
            "http://nwdaf:8080/notify".to_string(),
            Some(60),
        );

        assert_eq!(sub.event_type, EventType::LoadLevel);
        assert_eq!(sub.nf_type, NfType::Amf);
        assert_eq!(sub.status, SubscriptionStatus::Active);
        assert_eq!(manager.subscriptions.len(), 1);
    }

    #[test]
    fn test_update_subscription_status() {
        let mut manager = EventExposureManager::new();

        let sub = manager.create_subscription(
            EventType::NetworkPerformance,
            NfType::Upf,
            "http://nwdaf:8080/notify".to_string(),
            None,
        );

        manager
            .update_subscription_status(&sub.subscription_id, SubscriptionStatus::Suspended)
            .unwrap();

        let updated = manager.get_subscription(&sub.subscription_id).unwrap();
        assert_eq!(updated.status, SubscriptionStatus::Suspended);
    }

    #[test]
    fn test_delete_subscription() {
        let mut manager = EventExposureManager::new();

        let sub = manager.create_subscription(
            EventType::NfLoad,
            NfType::Smf,
            "http://nwdaf:8080/notify".to_string(),
            Some(30),
        );

        assert!(manager.get_subscription(&sub.subscription_id).is_some());

        manager.delete_subscription(&sub.subscription_id).unwrap();

        assert!(manager.get_subscription(&sub.subscription_id).is_none());
    }

    #[test]
    fn test_receive_notification() {
        let mut manager = EventExposureManager::new();

        let sub = manager.create_subscription(
            EventType::LoadLevel,
            NfType::Amf,
            "http://nwdaf:8080/notify".to_string(),
            Some(60),
        );

        let notification = EventNotification {
            subscription_id: sub.subscription_id.clone(),
            event_type: EventType::LoadLevel,
            nf_instance_id: "amf-001".to_string(),
            event_data: EventData::LoadLevel {
                load_percentage: 75,
                registered_ues: 1000,
            },
            timestamp_ms: 0,
        };

        manager.receive_notification(notification);

        let notifications = manager.get_notifications_for_subscription(&sub.subscription_id);
        assert_eq!(notifications.len(), 1);
    }

    #[test]
    fn test_list_active_subscriptions() {
        let mut manager = EventExposureManager::new();

        let sub1 = manager.create_subscription(
            EventType::LoadLevel,
            NfType::Amf,
            "http://nwdaf:8080/notify".to_string(),
            Some(60),
        );

        let sub2 = manager.create_subscription(
            EventType::NfLoad,
            NfType::Smf,
            "http://nwdaf:8080/notify".to_string(),
            Some(30),
        );

        manager
            .update_subscription_status(&sub2.subscription_id, SubscriptionStatus::Suspended)
            .unwrap();

        let active = manager.list_active_subscriptions();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].subscription_id, sub1.subscription_id);
    }
}
