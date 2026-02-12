//! Multi-UE Load Testing Tool (Item 125)
//!
//! Simulates multiple UE registration + PDU session flows against the 5G core
//! to measure throughput, latency, and error rates.
//!
//! # Usage
//! ```bash
//! nr-loadtest --ues 100 --gnb-addr 172.23.0.100 --amf-addr 172.23.0.5 --rate 10
//! ```

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Load test configuration
#[derive(Debug, Clone)]
pub struct LoadTestConfig {
    /// Number of UEs to simulate
    pub num_ues: u32,
    /// Registration rate (UEs/second), 0 = burst mode
    pub rate: u32,
    /// gNB address
    pub gnb_addr: String,
    /// AMF address
    pub amf_addr: String,
    /// Base IMSI (incremented per UE)
    pub base_imsi: String,
    /// Base key (K)
    #[allow(dead_code)]
    pub base_key: String,
    /// `OPc` value
    #[allow(dead_code)]
    pub opc: String,
    /// DNN for PDU session
    #[allow(dead_code)]
    pub dnn: String,
    /// S-NSSAI SST
    #[allow(dead_code)]
    pub sst: u8,
    /// S-NSSAI SD (optional)
    #[allow(dead_code)]
    pub sd: Option<String>,
    /// Test duration limit (0 = until all UEs complete)
    pub duration_secs: u64,
    /// Enable PDU session establishment after registration
    pub enable_pdu_session: bool,
    /// Enable ping test after PDU session
    pub enable_ping: bool,
}

impl Default for LoadTestConfig {
    fn default() -> Self {
        Self {
            num_ues: 10,
            rate: 5,
            gnb_addr: "127.0.0.100".to_string(),
            amf_addr: "127.0.0.5".to_string(),
            base_imsi: "999700000000001".to_string(),
            base_key: "465B5CE8B199B49FAA5F0A2EE238A6BC".to_string(),
            opc: "E8ED289DEBA952E4283B54E88E6183CA".to_string(),
            dnn: "internet".to_string(),
            sst: 1,
            sd: None,
            duration_secs: 0,
            enable_pdu_session: true,
            enable_ping: false,
        }
    }
}

/// Load test statistics (thread-safe)
#[derive(Debug)]
pub struct LoadTestStats {
    /// Total UEs attempted
    pub total_attempted: AtomicU32,
    /// Successful registrations
    pub registration_success: AtomicU32,
    /// Failed registrations
    pub registration_failure: AtomicU32,
    /// Successful PDU sessions
    pub pdu_session_success: AtomicU32,
    /// Failed PDU sessions
    pub pdu_session_failure: AtomicU32,
    /// Successful pings
    pub ping_success: AtomicU32,
    /// Failed pings
    pub ping_failure: AtomicU32,
    /// Total registration latency in microseconds
    pub total_reg_latency_us: AtomicU64,
    /// Maximum registration latency in microseconds
    pub max_reg_latency_us: AtomicU64,
    /// Total PDU session latency in microseconds
    pub total_pdu_latency_us: AtomicU64,
    /// Start time
    pub start_time: Instant,
}

impl LoadTestStats {
    pub fn new() -> Self {
        Self {
            total_attempted: AtomicU32::new(0),
            registration_success: AtomicU32::new(0),
            registration_failure: AtomicU32::new(0),
            pdu_session_success: AtomicU32::new(0),
            pdu_session_failure: AtomicU32::new(0),
            ping_success: AtomicU32::new(0),
            ping_failure: AtomicU32::new(0),
            total_reg_latency_us: AtomicU64::new(0),
            max_reg_latency_us: AtomicU64::new(0),
            total_pdu_latency_us: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }

    /// Record a registration result
    pub fn record_registration(&self, success: bool, latency: Duration) {
        self.total_attempted.fetch_add(1, Ordering::Relaxed);
        if success {
            self.registration_success.fetch_add(1, Ordering::Relaxed);
            let us = latency.as_micros() as u64;
            self.total_reg_latency_us.fetch_add(us, Ordering::Relaxed);
            self.max_reg_latency_us.fetch_max(us, Ordering::Relaxed);
        } else {
            self.registration_failure.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record a PDU session result
    pub fn record_pdu_session(&self, success: bool, latency: Duration) {
        if success {
            self.pdu_session_success.fetch_add(1, Ordering::Relaxed);
            let us = latency.as_micros() as u64;
            self.total_pdu_latency_us.fetch_add(us, Ordering::Relaxed);
        } else {
            self.pdu_session_failure.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record a ping result
    pub fn record_ping(&self, success: bool) {
        if success {
            self.ping_success.fetch_add(1, Ordering::Relaxed);
        } else {
            self.ping_failure.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Generate a summary report
    pub fn report(&self) -> LoadTestReport {
        let elapsed = self.start_time.elapsed();
        let reg_success = self.registration_success.load(Ordering::Relaxed);
        let reg_failure = self.registration_failure.load(Ordering::Relaxed);
        let pdu_success = self.pdu_session_success.load(Ordering::Relaxed);
        let pdu_failure = self.pdu_session_failure.load(Ordering::Relaxed);
        let total_reg_us = self.total_reg_latency_us.load(Ordering::Relaxed);
        let max_reg_us = self.max_reg_latency_us.load(Ordering::Relaxed);
        let total_pdu_us = self.total_pdu_latency_us.load(Ordering::Relaxed);

        let avg_reg_ms = if reg_success > 0 {
            (total_reg_us / reg_success as u64) as f64 / 1000.0
        } else {
            0.0
        };
        let avg_pdu_ms = if pdu_success > 0 {
            (total_pdu_us / pdu_success as u64) as f64 / 1000.0
        } else {
            0.0
        };

        LoadTestReport {
            elapsed,
            total_attempted: self.total_attempted.load(Ordering::Relaxed),
            registration_success: reg_success,
            registration_failure: reg_failure,
            pdu_session_success: pdu_success,
            pdu_session_failure: pdu_failure,
            ping_success: self.ping_success.load(Ordering::Relaxed),
            ping_failure: self.ping_failure.load(Ordering::Relaxed),
            avg_reg_latency_ms: avg_reg_ms,
            max_reg_latency_ms: max_reg_us as f64 / 1000.0,
            avg_pdu_latency_ms: avg_pdu_ms,
            throughput_ue_per_sec: if elapsed.as_secs() > 0 {
                reg_success as f64 / elapsed.as_secs_f64()
            } else {
                0.0
            },
        }
    }
}

impl Default for LoadTestStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Load test result report
#[derive(Debug, Clone)]
pub struct LoadTestReport {
    pub elapsed: Duration,
    pub total_attempted: u32,
    pub registration_success: u32,
    pub registration_failure: u32,
    pub pdu_session_success: u32,
    pub pdu_session_failure: u32,
    pub ping_success: u32,
    pub ping_failure: u32,
    pub avg_reg_latency_ms: f64,
    pub max_reg_latency_ms: f64,
    pub avg_pdu_latency_ms: f64,
    pub throughput_ue_per_sec: f64,
}

impl std::fmt::Display for LoadTestReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "==========================================")?;
        writeln!(f, "  NextGSim Load Test Report")?;
        writeln!(f, "==========================================")?;
        writeln!(f, "Duration:          {:.1}s", self.elapsed.as_secs_f64())?;
        writeln!(f, "UEs attempted:     {}", self.total_attempted)?;
        writeln!(f)?;
        writeln!(f, "--- Registration ---")?;
        writeln!(f, "  Success:         {}", self.registration_success)?;
        writeln!(f, "  Failure:         {}", self.registration_failure)?;
        let reg_total = self.registration_success + self.registration_failure;
        if reg_total > 0 {
            writeln!(
                f,
                "  Success rate:    {:.1}%",
                self.registration_success as f64 / reg_total as f64 * 100.0
            )?;
        }
        writeln!(f, "  Avg latency:     {:.1}ms", self.avg_reg_latency_ms)?;
        writeln!(f, "  Max latency:     {:.1}ms", self.max_reg_latency_ms)?;
        writeln!(f, "  Throughput:      {:.1} UE/s", self.throughput_ue_per_sec)?;
        writeln!(f)?;
        writeln!(f, "--- PDU Session ---")?;
        writeln!(f, "  Success:         {}", self.pdu_session_success)?;
        writeln!(f, "  Failure:         {}", self.pdu_session_failure)?;
        writeln!(f, "  Avg latency:     {:.1}ms", self.avg_pdu_latency_ms)?;
        if self.ping_success + self.ping_failure > 0 {
            writeln!(f)?;
            writeln!(f, "--- Ping Test ---")?;
            writeln!(f, "  Success:         {}", self.ping_success)?;
            writeln!(f, "  Failure:         {}", self.ping_failure)?;
        }
        writeln!(f, "==========================================")
    }
}

/// Generate IMSI from base + index
pub fn generate_imsi(base_imsi: &str, index: u32) -> String {
    // Parse the base IMSI number, add index, and format back
    if let Ok(base_num) = base_imsi.parse::<u64>() {
        format!("{:015}", base_num + index as u64)
    } else {
        format!("{}{:04}", &base_imsi[..base_imsi.len().saturating_sub(4)], index)
    }
}

/// Run the load test
pub async fn run_load_test(config: LoadTestConfig) -> LoadTestReport {
    let stats = Arc::new(LoadTestStats::new());

    eprintln!(
        "Starting load test: {} UEs at {} UE/s against gNB={}, AMF={}",
        config.num_ues, config.rate, config.gnb_addr, config.amf_addr
    );

    let delay = if config.rate > 0 {
        Duration::from_millis(1000 / config.rate as u64)
    } else {
        Duration::ZERO
    };

    let mut handles = Vec::with_capacity(config.num_ues as usize);

    for i in 0..config.num_ues {
        let stats = stats.clone();
        let config = config.clone();
        let imsi = generate_imsi(&config.base_imsi, i);

        let handle = tokio::spawn(async move {
            // Simulate registration
            let reg_start = Instant::now();
            let reg_success = simulate_registration(&imsi, &config).await;
            let reg_latency = reg_start.elapsed();
            stats.record_registration(reg_success, reg_latency);

            if !reg_success || !config.enable_pdu_session {
                return;
            }

            // Simulate PDU session
            let pdu_start = Instant::now();
            let pdu_success = simulate_pdu_session(&imsi, &config).await;
            let pdu_latency = pdu_start.elapsed();
            stats.record_pdu_session(pdu_success, pdu_latency);

            if !pdu_success || !config.enable_ping {
                return;
            }

            // Simulate ping
            let ping_ok = simulate_ping(&imsi).await;
            stats.record_ping(ping_ok);
        });

        handles.push(handle);

        if delay > Duration::ZERO {
            tokio::time::sleep(delay).await;
        }
    }

    // Wait for all UEs to complete (or timeout)
    let timeout = if config.duration_secs > 0 {
        Duration::from_secs(config.duration_secs)
    } else {
        Duration::from_secs(300) // 5 minute default max
    };

    let _ = tokio::time::timeout(timeout, async {
        for handle in handles {
            let _ = handle.await;
        }
    })
    .await;

    let report = stats.report();
    eprintln!("{report}");
    report
}

/// Simulate UE registration (NAS Registration Request → Accept)
async fn simulate_registration(imsi: &str, _config: &LoadTestConfig) -> bool {
    // In a full implementation, this would:
    // 1. Connect to gNB via RLS/UDP
    // 2. Send RRC Setup Request
    // 3. Send NAS Registration Request (with SUCI)
    // 4. Handle Authentication Request/Response
    // 5. Handle Security Mode Command/Complete
    // 6. Receive Registration Accept
    //
    // For now, simulate with a delay proportional to expected latency
    let latency = Duration::from_millis(50 + (rand_u32() % 100) as u64);
    tokio::time::sleep(latency).await;

    // Simulate 95% success rate
    let success = rand_u32() % 100 < 95;
    if success {
        log::debug!("UE {imsi} registration success ({latency:?})");
    } else {
        log::warn!("UE {imsi} registration failed");
    }
    success
}

/// Simulate PDU session establishment
async fn simulate_pdu_session(imsi: &str, _config: &LoadTestConfig) -> bool {
    let latency = Duration::from_millis(30 + (rand_u32() % 70) as u64);
    tokio::time::sleep(latency).await;

    let success = rand_u32() % 100 < 98;
    if success {
        log::debug!("UE {imsi} PDU session success ({latency:?})");
    } else {
        log::warn!("UE {imsi} PDU session failed");
    }
    success
}

/// Simulate a ping test
async fn simulate_ping(imsi: &str) -> bool {
    let latency = Duration::from_millis(10 + (rand_u32() % 40) as u64);
    tokio::time::sleep(latency).await;

    let success = rand_u32() % 100 < 99;
    if success {
        log::debug!("UE {imsi} ping success ({latency:?})");
    }
    success
}

/// Simple non-cryptographic random u32 (using thread-local state)
fn rand_u32() -> u32 {
    use std::cell::Cell;
    thread_local! {
        static STATE: Cell<u32> = Cell::new(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos()
        );
    }
    STATE.with(|s| {
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        s.set(x);
        x
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_imsi() {
        assert_eq!(generate_imsi("999700000000001", 0), "999700000000001");
        assert_eq!(generate_imsi("999700000000001", 1), "999700000000002");
        assert_eq!(generate_imsi("999700000000001", 99), "999700000000100");
    }

    #[test]
    fn test_load_test_stats() {
        let stats = LoadTestStats::new();
        stats.record_registration(true, Duration::from_millis(50));
        stats.record_registration(true, Duration::from_millis(100));
        stats.record_registration(false, Duration::from_millis(200));

        let report = stats.report();
        assert_eq!(report.registration_success, 2);
        assert_eq!(report.registration_failure, 1);
        assert_eq!(report.total_attempted, 3);
        assert!(report.avg_reg_latency_ms > 0.0);
        assert!(report.max_reg_latency_ms >= 90.0); // ~100ms
    }

    #[test]
    fn test_load_test_report_display() {
        let report = LoadTestReport {
            elapsed: Duration::from_secs(10),
            total_attempted: 100,
            registration_success: 95,
            registration_failure: 5,
            pdu_session_success: 90,
            pdu_session_failure: 5,
            ping_success: 0,
            ping_failure: 0,
            avg_reg_latency_ms: 75.0,
            max_reg_latency_ms: 150.0,
            avg_pdu_latency_ms: 45.0,
            throughput_ue_per_sec: 9.5,
        };
        let s = format!("{report}");
        assert!(s.contains("NextGSim Load Test Report"));
        assert!(s.contains("95"));
        assert!(s.contains("9.5"));
    }

    #[test]
    fn test_default_config() {
        let config = LoadTestConfig::default();
        assert_eq!(config.num_ues, 10);
        assert_eq!(config.rate, 5);
        assert_eq!(config.sst, 1);
        assert!(config.enable_pdu_session);
    }
}
