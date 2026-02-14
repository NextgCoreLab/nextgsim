//! GTP-U QoS Enforcement (3GPP TS 29.281, TS 23.501)
//!
//! Implements QoS Flow Identifier (QFI) to DSCP mapping, 5QI to QFI tables,
//! and per-flow token-bucket rate limiting for GTP-U user plane traffic.

use std::collections::HashMap;
use std::time::Instant;

// ============================================================================
// 5QI / QFI Tables (TS 23.501 Table 5.7.4-1)
// ============================================================================

/// 5QI (5G QoS Identifier) characteristics from TS 23.501.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FiveQiCharacteristics {
    /// 5QI value
    pub five_qi: u16,
    /// Resource type
    pub resource_type: QosResourceType,
    /// Default priority level (1-127, lower = higher priority)
    pub priority: u8,
    /// Packet delay budget in milliseconds
    pub packet_delay_budget_ms: u32,
    /// Packet error rate (as -log10, e.g. 6 means 10^-6)
    pub packet_error_rate_exp: u8,
    /// Default maximum data burst volume (bytes, 0 = not applicable)
    pub max_data_burst_bytes: u32,
    /// Default averaging window (ms, 0 = not applicable for non-GBR)
    pub averaging_window_ms: u32,
}

/// QoS resource type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QosResourceType {
    /// Guaranteed Bit Rate
    Gbr,
    /// Non-GBR
    NonGbr,
    /// Delay-critical GBR
    DelayCriticalGbr,
}

/// Standard 5QI table (TS 23.501 Table 5.7.4-1, selected entries).
pub fn standard_5qi_table() -> Vec<FiveQiCharacteristics> {
    vec![
        // GBR
        FiveQiCharacteristics { five_qi: 1, resource_type: QosResourceType::Gbr, priority: 20, packet_delay_budget_ms: 100, packet_error_rate_exp: 2, max_data_burst_bytes: 0, averaging_window_ms: 2000 },
        FiveQiCharacteristics { five_qi: 2, resource_type: QosResourceType::Gbr, priority: 40, packet_delay_budget_ms: 150, packet_error_rate_exp: 3, max_data_burst_bytes: 0, averaging_window_ms: 2000 },
        FiveQiCharacteristics { five_qi: 3, resource_type: QosResourceType::Gbr, priority: 30, packet_delay_budget_ms: 50, packet_error_rate_exp: 3, max_data_burst_bytes: 0, averaging_window_ms: 2000 },
        FiveQiCharacteristics { five_qi: 4, resource_type: QosResourceType::Gbr, priority: 50, packet_delay_budget_ms: 300, packet_error_rate_exp: 6, max_data_burst_bytes: 0, averaging_window_ms: 2000 },
        FiveQiCharacteristics { five_qi: 65, resource_type: QosResourceType::Gbr, priority: 7, packet_delay_budget_ms: 75, packet_error_rate_exp: 2, max_data_burst_bytes: 0, averaging_window_ms: 2000 },
        FiveQiCharacteristics { five_qi: 66, resource_type: QosResourceType::Gbr, priority: 20, packet_delay_budget_ms: 100, packet_error_rate_exp: 2, max_data_burst_bytes: 0, averaging_window_ms: 2000 },
        FiveQiCharacteristics { five_qi: 67, resource_type: QosResourceType::Gbr, priority: 15, packet_delay_budget_ms: 100, packet_error_rate_exp: 3, max_data_burst_bytes: 0, averaging_window_ms: 2000 },
        // Non-GBR
        FiveQiCharacteristics { five_qi: 5, resource_type: QosResourceType::NonGbr, priority: 10, packet_delay_budget_ms: 100, packet_error_rate_exp: 6, max_data_burst_bytes: 0, averaging_window_ms: 0 },
        FiveQiCharacteristics { five_qi: 6, resource_type: QosResourceType::NonGbr, priority: 60, packet_delay_budget_ms: 300, packet_error_rate_exp: 6, max_data_burst_bytes: 0, averaging_window_ms: 0 },
        FiveQiCharacteristics { five_qi: 7, resource_type: QosResourceType::NonGbr, priority: 70, packet_delay_budget_ms: 100, packet_error_rate_exp: 3, max_data_burst_bytes: 0, averaging_window_ms: 0 },
        FiveQiCharacteristics { five_qi: 8, resource_type: QosResourceType::NonGbr, priority: 80, packet_delay_budget_ms: 300, packet_error_rate_exp: 6, max_data_burst_bytes: 0, averaging_window_ms: 0 },
        FiveQiCharacteristics { five_qi: 9, resource_type: QosResourceType::NonGbr, priority: 90, packet_delay_budget_ms: 300, packet_error_rate_exp: 6, max_data_burst_bytes: 0, averaging_window_ms: 0 },
        // Delay-critical GBR (Rel-16+)
        FiveQiCharacteristics { five_qi: 82, resource_type: QosResourceType::DelayCriticalGbr, priority: 19, packet_delay_budget_ms: 10, packet_error_rate_exp: 4, max_data_burst_bytes: 255, averaging_window_ms: 2000 },
        FiveQiCharacteristics { five_qi: 83, resource_type: QosResourceType::DelayCriticalGbr, priority: 22, packet_delay_budget_ms: 10, packet_error_rate_exp: 4, max_data_burst_bytes: 1354, averaging_window_ms: 2000 },
        FiveQiCharacteristics { five_qi: 84, resource_type: QosResourceType::DelayCriticalGbr, priority: 24, packet_delay_budget_ms: 30, packet_error_rate_exp: 5, max_data_burst_bytes: 1354, averaging_window_ms: 2000 },
        FiveQiCharacteristics { five_qi: 85, resource_type: QosResourceType::DelayCriticalGbr, priority: 21, packet_delay_budget_ms: 5, packet_error_rate_exp: 5, max_data_burst_bytes: 255, averaging_window_ms: 2000 },
    ]
}

/// Lookup 5QI characteristics by value.
pub fn lookup_5qi(five_qi: u16) -> Option<FiveQiCharacteristics> {
    standard_5qi_table().into_iter().find(|c| c.five_qi == five_qi)
}

// ============================================================================
// QFI to DSCP Mapping (TS 29.281)
// ============================================================================

/// DSCP (Differentiated Services Code Point) value.
pub type Dscp = u8;

/// Default QFI-to-DSCP mapping based on 5QI resource types.
///
/// Maps QoS Flow Identifier to IP DSCP values for outer GTP-U header
/// marking per TS 29.281 Section 5.2.2.1.
pub fn default_qfi_to_dscp(_qfi: u8, five_qi: u16) -> Dscp {
    match five_qi {
        // Voice / IMS signalling -> EF (Expedited Forwarding)
        1 | 5 | 65 | 69 => 46,
        // Video / conversational -> AF41
        2 | 66 => 34,
        // Low-latency live streaming -> AF31
        3 | 67 | 75 => 26,
        // Non-conversational video -> AF21
        4 => 18,
        // Real-time gaming -> CS5
        7 => 40,
        // High priority data -> AF11
        6 | 8 => 10,
        // XR / Delay-critical GBR -> CS5 (high priority)
        82..=85 => 40,
        // Default best effort
        _ => 0,
    }
}

/// Configurable QFI-to-DSCP mapping table.
#[derive(Debug, Clone)]
pub struct QfiDscpMapper {
    /// Custom QFI -> DSCP overrides
    overrides: HashMap<u8, Dscp>,
    /// 5QI -> DSCP mapping (used when no QFI override exists)
    five_qi_map: HashMap<u16, Dscp>,
}

impl QfiDscpMapper {
    /// Creates a new mapper with default mappings.
    pub fn new() -> Self {
        let mut five_qi_map = HashMap::new();
        for entry in standard_5qi_table() {
            five_qi_map.insert(entry.five_qi, default_qfi_to_dscp(0, entry.five_qi));
        }
        Self {
            overrides: HashMap::new(),
            five_qi_map,
        }
    }

    /// Set a custom QFI -> DSCP mapping.
    pub fn set_qfi_dscp(&mut self, qfi: u8, dscp: Dscp) {
        self.overrides.insert(qfi, dscp);
    }

    /// Set a custom 5QI -> DSCP mapping.
    pub fn set_5qi_dscp(&mut self, five_qi: u16, dscp: Dscp) {
        self.five_qi_map.insert(five_qi, dscp);
    }

    /// Resolve DSCP for a given QFI and 5QI.
    pub fn resolve(&self, qfi: u8, five_qi: u16) -> Dscp {
        if let Some(&dscp) = self.overrides.get(&qfi) {
            return dscp;
        }
        if let Some(&dscp) = self.five_qi_map.get(&five_qi) {
            return dscp;
        }
        default_qfi_to_dscp(qfi, five_qi)
    }
}

impl Default for QfiDscpMapper {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Per-Flow Token Bucket Rate Limiter
// ============================================================================

/// Token bucket rate limiter for a single QoS flow.
#[derive(Debug, Clone)]
pub struct TokenBucket {
    /// Maximum number of tokens (burst size in bytes)
    pub capacity: u64,
    /// Current number of tokens
    tokens: f64,
    /// Token refill rate (bytes per second)
    pub rate_bps: u64,
    /// Last refill timestamp
    last_refill: Instant,
    /// Total bytes passed
    pub bytes_passed: u64,
    /// Total bytes dropped
    pub bytes_dropped: u64,
    /// Total packets passed
    pub packets_passed: u64,
    /// Total packets dropped
    pub packets_dropped: u64,
}

impl TokenBucket {
    /// Creates a new token bucket with the given rate (bytes/sec) and burst size (bytes).
    pub fn new(rate_bps: u64, burst_bytes: u64) -> Self {
        Self {
            capacity: burst_bytes,
            tokens: burst_bytes as f64,
            rate_bps,
            last_refill: Instant::now(),
            bytes_passed: 0,
            bytes_dropped: 0,
            packets_passed: 0,
            packets_dropped: 0,
        }
    }

    /// Creates a token bucket from kbps rate and burst multiplier.
    pub fn from_kbps(rate_kbps: u64, burst_multiplier: f64) -> Self {
        let rate_bps = rate_kbps * 125; // kbps to bytes/sec: kbps * 1000 / 8
        let burst = (rate_bps as f64 * burst_multiplier) as u64;
        Self::new(rate_bps, burst.max(1500)) // At least 1 MTU burst
    }

    /// Refill tokens based on elapsed time.
    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill);
        let new_tokens = elapsed.as_secs_f64() * self.rate_bps as f64;
        self.tokens = (self.tokens + new_tokens).min(self.capacity as f64);
        self.last_refill = now;
    }

    /// Attempt to consume tokens for a packet. Returns true if allowed.
    pub fn allow(&mut self, packet_size: usize) -> bool {
        self.refill();
        let size = packet_size as f64;
        if self.tokens >= size {
            self.tokens -= size;
            self.bytes_passed += packet_size as u64;
            self.packets_passed += 1;
            true
        } else {
            self.bytes_dropped += packet_size as u64;
            self.packets_dropped += 1;
            false
        }
    }

    /// Returns the current token fill level as a fraction (0.0 - 1.0).
    pub fn fill_level(&mut self) -> f64 {
        self.refill();
        if self.capacity == 0 {
            return 1.0;
        }
        self.tokens / self.capacity as f64
    }
}

// ============================================================================
// QoS Flow Enforcer
// ============================================================================

/// Per-session QoS flow enforcement.
///
/// Manages rate limiting and DSCP marking for all QoS flows in a PDU session.
#[derive(Debug)]
pub struct QosFlowEnforcer {
    /// DSCP mapper
    dscp_mapper: QfiDscpMapper,
    /// Per-flow rate limiters: (QFI) -> TokenBucket
    rate_limiters: HashMap<u8, TokenBucket>,
    /// Per-flow 5QI assignments
    flow_5qi: HashMap<u8, u16>,
}

impl QosFlowEnforcer {
    /// Creates a new QoS flow enforcer.
    pub fn new() -> Self {
        Self {
            dscp_mapper: QfiDscpMapper::new(),
            rate_limiters: HashMap::new(),
            flow_5qi: HashMap::new(),
        }
    }

    /// Configures a QoS flow with rate limiting.
    ///
    /// - `qfi`: QoS Flow Identifier (0-63)
    /// - `five_qi`: 5QI value for this flow
    /// - `mbr_kbps`: Maximum Bit Rate in kbps (0 = unlimited)
    /// - `gbr_kbps`: Guaranteed Bit Rate in kbps (0 = non-GBR)
    pub fn configure_flow(&mut self, qfi: u8, five_qi: u16, mbr_kbps: u64, _gbr_kbps: u64) {
        self.flow_5qi.insert(qfi, five_qi);
        if mbr_kbps > 0 {
            let bucket = TokenBucket::from_kbps(mbr_kbps, 0.1); // 100ms burst
            self.rate_limiters.insert(qfi, bucket);
        }
    }

    /// Remove a QoS flow configuration.
    pub fn remove_flow(&mut self, qfi: u8) {
        self.flow_5qi.remove(&qfi);
        self.rate_limiters.remove(&qfi);
    }

    /// Enforce QoS on a packet. Returns (allowed, dscp) or None if QFI unknown.
    pub fn enforce(&mut self, qfi: u8, packet_size: usize) -> Option<(bool, Dscp)> {
        let five_qi = self.flow_5qi.get(&qfi).copied()?;
        let dscp = self.dscp_mapper.resolve(qfi, five_qi);

        let allowed = match self.rate_limiters.get_mut(&qfi) {
            Some(bucket) => bucket.allow(packet_size),
            None => true, // No rate limiter = unlimited
        };

        Some((allowed, dscp))
    }

    /// Get DSCP for a flow without rate limiting check.
    pub fn get_dscp(&self, qfi: u8) -> Option<Dscp> {
        let five_qi = self.flow_5qi.get(&qfi).copied()?;
        Some(self.dscp_mapper.resolve(qfi, five_qi))
    }

    /// Returns flow statistics for a QFI.
    pub fn flow_stats(&mut self, qfi: u8) -> Option<FlowStats> {
        let bucket = self.rate_limiters.get_mut(&qfi)?;
        Some(FlowStats {
            qfi,
            five_qi: self.flow_5qi.get(&qfi).copied().unwrap_or(0),
            bytes_passed: bucket.bytes_passed,
            bytes_dropped: bucket.bytes_dropped,
            packets_passed: bucket.packets_passed,
            packets_dropped: bucket.packets_dropped,
            fill_level: bucket.fill_level(),
        })
    }

    /// Returns the number of configured flows.
    pub fn flow_count(&self) -> usize {
        self.flow_5qi.len()
    }

    /// Returns a mutable reference to the DSCP mapper for custom configuration.
    pub fn dscp_mapper_mut(&mut self) -> &mut QfiDscpMapper {
        &mut self.dscp_mapper
    }
}

impl Default for QosFlowEnforcer {
    fn default() -> Self {
        Self::new()
    }
}

/// Flow statistics snapshot.
#[derive(Debug, Clone)]
pub struct FlowStats {
    /// QoS Flow Identifier
    pub qfi: u8,
    /// 5QI value
    pub five_qi: u16,
    /// Bytes passed
    pub bytes_passed: u64,
    /// Bytes dropped
    pub bytes_dropped: u64,
    /// Packets passed
    pub packets_passed: u64,
    /// Packets dropped
    pub packets_dropped: u64,
    /// Current bucket fill level (0.0-1.0)
    pub fill_level: f64,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_5qi_lookup() {
        let c = lookup_5qi(1).unwrap();
        assert_eq!(c.resource_type, QosResourceType::Gbr);
        assert_eq!(c.priority, 20);
        assert_eq!(c.packet_delay_budget_ms, 100);

        let c = lookup_5qi(82).unwrap();
        assert_eq!(c.resource_type, QosResourceType::DelayCriticalGbr);
        assert_eq!(c.packet_delay_budget_ms, 10);

        assert!(lookup_5qi(999).is_none());
    }

    #[test]
    fn test_default_dscp_mapping() {
        assert_eq!(default_qfi_to_dscp(0, 1), 46); // Voice -> EF
        assert_eq!(default_qfi_to_dscp(0, 2), 34); // Video -> AF41
        assert_eq!(default_qfi_to_dscp(0, 9), 0);  // Best effort
        assert_eq!(default_qfi_to_dscp(0, 82), 40); // XR -> CS5
    }

    #[test]
    fn test_qfi_dscp_mapper() {
        let mut mapper = QfiDscpMapper::new();
        // Default mapping
        assert_eq!(mapper.resolve(1, 1), 46);
        assert_eq!(mapper.resolve(2, 9), 0);

        // Custom override
        mapper.set_qfi_dscp(1, 32);
        assert_eq!(mapper.resolve(1, 1), 32); // Override wins
        assert_eq!(mapper.resolve(2, 1), 46); // Other QFI uses 5QI mapping
    }

    #[test]
    fn test_token_bucket_basic() {
        let mut bucket = TokenBucket::new(1_000_000, 10_000); // 1MB/s, 10KB burst
        assert!(bucket.allow(1000));
        assert_eq!(bucket.packets_passed, 1);
        assert_eq!(bucket.bytes_passed, 1000);
    }

    #[test]
    fn test_token_bucket_rate_limit() {
        let mut bucket = TokenBucket::new(1000, 2000); // 1KB/s, 2KB burst
        // Consume entire burst
        assert!(bucket.allow(2000));
        // Should be rate-limited now
        assert!(!bucket.allow(1000));
        assert_eq!(bucket.packets_dropped, 1);
        assert_eq!(bucket.bytes_dropped, 1000);
    }

    #[test]
    fn test_token_bucket_from_kbps() {
        let bucket = TokenBucket::from_kbps(1000, 0.1); // 1Mbps, 100ms burst
        assert_eq!(bucket.rate_bps, 125_000); // 1000 * 125
        assert!(bucket.capacity >= 1500); // At least 1 MTU
    }

    #[test]
    fn test_qos_flow_enforcer() {
        let mut enforcer = QosFlowEnforcer::new();

        // Configure a GBR voice flow
        enforcer.configure_flow(1, 1, 128, 64); // 128kbps MBR, 64kbps GBR

        // Configure a non-GBR best-effort flow (no rate limit)
        enforcer.configure_flow(5, 9, 0, 0);

        assert_eq!(enforcer.flow_count(), 2);

        // Enforce on voice flow
        let (allowed, dscp) = enforcer.enforce(1, 200).unwrap();
        assert!(allowed);
        assert_eq!(dscp, 46); // EF

        // Enforce on best-effort flow
        let (allowed, dscp) = enforcer.enforce(5, 1500).unwrap();
        assert!(allowed);
        assert_eq!(dscp, 0); // Best effort

        // Unknown QFI
        assert!(enforcer.enforce(99, 100).is_none());
    }

    #[test]
    fn test_qos_flow_enforcer_remove() {
        let mut enforcer = QosFlowEnforcer::new();
        enforcer.configure_flow(1, 1, 128, 64);
        assert_eq!(enforcer.flow_count(), 1);
        enforcer.remove_flow(1);
        assert_eq!(enforcer.flow_count(), 0);
    }

    #[test]
    fn test_qos_flow_stats() {
        let mut enforcer = QosFlowEnforcer::new();
        enforcer.configure_flow(1, 82, 50_000, 10_000); // XR flow

        enforcer.enforce(1, 1000);
        enforcer.enforce(1, 500);

        let stats = enforcer.flow_stats(1).unwrap();
        assert_eq!(stats.qfi, 1);
        assert_eq!(stats.five_qi, 82);
        assert_eq!(stats.packets_passed, 2);
        assert_eq!(stats.bytes_passed, 1500);
    }

    #[test]
    fn test_5qi_xr_entries() {
        // Verify XR 5QI entries (82-85) are present
        for qi in [82, 83, 84, 85] {
            let c = lookup_5qi(qi).unwrap();
            assert_eq!(c.resource_type, QosResourceType::DelayCriticalGbr);
            assert!(c.packet_delay_budget_ms <= 30);
        }
    }
}
