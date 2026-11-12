//! Demo D441A, the RAN half of the User Data Congestion loop (nextgsim#465,
//! feature `demo-d441a-congestion-loop`, toggle convention #458).
//!
//! While the demo is active (its feature built, and the `demo:` block's
//! `d441a-congestion-loop` entry on, in any mode):
//!
//! 1. **A capacity-C cell.** The RLS task serves every UE's downlink through one
//!    FIFO drained at `congestion_export.cell_capacity_mbps` ([`queue::DlQueue`]),
//!    so one UE's flood delays another's packets. **A simulator construct**: the
//!    gNB has no PHY or scheduler, so C is configured, not derived.
//! 2. **A measured occupancy.** Each `window_ms` the RLS task hands the exporter
//!    the octets the queue actually served ([`WindowSample`]); occupancy is
//!    served bits / (C x window).
//! 3. **An exporter** ([`exporter`]) POSTs a cell report to nwdafd's
//!    `/nnwdaf-randata/v1/cell-reports` with an NRF-issued access token: the
//!    measured occupancy in mode `statistics`, plus the 8-sample OLS prediction
//!    ([`predictor`]) in mode `prediction`, nothing in mode `no-loop`.
//!
//! **Non-normative.** TS 23.288 §6.8.1 has the NWDAF "collect measurements per
//! cell" and Table 6.8.2-1 sources them from OAM (Radio Resource Utilization,
//! TS 28.552), collected per §6.2.3 through OAM management services. This push
//! from the gNB stands in for that OAM PM path; the route is not a 3GPP API.
//!
//! With the demo off (or the feature not built) none of this exists: the
//! downlink path, the RLS loop and the CLI are unchanged.

pub mod exporter;
pub mod predictor;
pub mod queue;

use std::sync::{Mutex, OnceLock};
use std::time::{Duration, SystemTime};

use nextgsim_common::config::GnbConfig;

pub use crate::demo::{D441A_KEY, D441A_MODES};

/// Mode: export the measured occupancy.
pub const MODE_STATISTICS: &str = "statistics";
/// Mode: export the measured occupancy and the OLS prediction.
pub const MODE_PREDICTION: &str = "prediction";
/// Mode: no export (the baseline); the capacity-C queue stays in place.
pub const MODE_NO_LOOP: &str = "no-loop";

/// The current D441A mode, if the demo is active in this process. Read it at
/// each window, so an `nr-cli ... set mode=` applies from the next one.
pub fn current_mode() -> Option<String> {
    nextgsim_demo::active_mode(D441A_KEY)
}

/// Whether the demo is active in this process, in any mode.
pub fn demo_active() -> bool {
    current_mode().is_some()
}

/// Startup check, run after the demo block is installed: with the demo active,
/// the `congestion_export` block must name a usable capacity and an NRF. With it
/// inactive the block is not read at all.
pub fn validate_startup(config: &GnbConfig) -> Result<(), String> {
    if !demo_active() {
        return Ok(());
    }
    validate_config(config)
}

/// The checks of [`validate_startup`], on any config.
pub fn validate_config(config: &GnbConfig) -> Result<(), String> {
    let c = &config.congestion_export;
    match c.cell_capacity_mbps {
        Some(mbps) if mbps.is_finite() && mbps > 0.0 => {}
        Some(mbps) => {
            return Err(format!(
                "congestion_export.cell_capacity_mbps must be a positive number, got {mbps}"
            ))
        }
        None => {
            return Err(format!(
                "demo {D441A_KEY} is active but congestion_export.cell_capacity_mbps is not set"
            ))
        }
    }
    if c.nrf_uri.as_deref().is_none_or(str::is_empty) {
        return Err(format!(
            "demo {D441A_KEY} is active but congestion_export.nrf_uri is not set (the exporter \
             registers there and takes its access tokens from it)"
        ));
    }
    if c.window_ms < 100 {
        return Err(format!(
            "congestion_export.window_ms must be at least 100, got {}",
            c.window_ms
        ));
    }
    if c.queue_limit_bytes < queue::MIN_LIMIT_BYTES {
        return Err(format!(
            "congestion_export.queue_limit_bytes must be at least {}, got {}",
            queue::MIN_LIMIT_BYTES,
            c.queue_limit_bytes
        ));
    }
    if c.nf_instance_id.is_empty() {
        return Err("congestion_export.nf_instance_id must not be empty".into());
    }
    Ok(())
}

/// The RLS task's D441A state: the queue and the open window.
#[derive(Debug)]
pub struct CellState {
    /// The capacity-C downlink FIFO.
    pub queue: queue::DlQueue,
    /// When the open window started.
    pub window_start: std::time::Instant,
    /// The window length.
    pub window: Duration,
    /// Where closed windows go; `None` when the exporter could not start.
    pub exporter: Option<tokio::sync::mpsc::Sender<WindowSample>>,
}

impl CellState {
    /// The state for `config`'s block (already validated), feeding `exporter`.
    pub fn new(
        config: &GnbConfig,
        exporter: Option<tokio::sync::mpsc::Sender<WindowSample>>,
    ) -> Self {
        let c = &config.congestion_export;
        let now = std::time::Instant::now();
        Self {
            queue: queue::DlQueue::new(
                c.cell_capacity_mbps.unwrap_or(0.0),
                c.queue_limit_bytes,
                now,
            ),
            window_start: now,
            window: Duration::from_millis(c.window_ms.max(100)),
            exporter,
        }
    }

    /// When the window is over at `now`, close it into a sample (the queue's
    /// served octets are taken) and open the next one.
    pub fn close_window(
        &mut self,
        now: std::time::Instant,
        connected_ues: u32,
    ) -> Option<WindowSample> {
        let elapsed = now.saturating_duration_since(self.window_start);
        if elapsed < self.window {
            return None;
        }
        self.window_start = now;
        Some(WindowSample {
            end: SystemTime::now(),
            elapsed,
            served_octets: self.queue.take_served(),
            connected_ues,
            queue_depth_bytes: self.queue.depth_octets(),
            queue_drops: self.queue.drops(),
        })
    }
}

/// One measurement window, handed from the RLS task to the exporter.
#[derive(Debug, Clone, PartialEq)]
pub struct WindowSample {
    /// When the window closed.
    pub end: SystemTime,
    /// How long it was open (measured, not assumed).
    pub elapsed: Duration,
    /// Downlink RLC PDU octets the queue served in it.
    pub served_octets: u64,
    /// UEs with a live radio association at the window's end.
    pub connected_ues: u32,
    /// Octets queued at the window's end.
    pub queue_depth_bytes: usize,
    /// PDUs dropped at the queue's tail since the start.
    pub queue_drops: u64,
}

impl WindowSample {
    /// Served throughput over the window, kbps.
    pub fn served_kbps(&self) -> f64 {
        let s = self.elapsed.as_secs_f64();
        if s > 0.0 {
            self.served_octets as f64 * 8.0 / s / 1000.0
        } else {
            0.0
        }
    }

    /// Occupancy in percent: served / C, clamped to 0..100.
    pub fn occupancy_pct(&self, capacity_mbps: f64) -> f64 {
        if capacity_mbps <= 0.0 {
            return 0.0;
        }
        (self.served_kbps() / (capacity_mbps * 1000.0) * 100.0).clamp(0.0, 100.0)
    }
}

/// What `nr-cli <gnb> congestion-status` reports.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct Status {
    /// The mode the last window ran under.
    pub mode: Option<String>,
    /// C, kbps.
    pub capacity_kbps: u64,
    /// The last measured occupancy, percent.
    pub last_occupancy_pct: Option<f64>,
    /// The last prediction: `(occupancy %, confidence, horizon ms)`.
    pub last_prediction: Option<(f64, u8, u64)>,
    /// The last export's outcome: an HTTP status, or why nothing was sent.
    pub last_export: Option<String>,
    /// Octets queued at the last window's end.
    pub queue_depth_bytes: usize,
    /// PDUs dropped at the queue's tail.
    pub queue_drops: u64,
    /// Exports answered 2xx.
    pub exports_accepted: u64,
    /// Exports answered otherwise, or not sent (no token, no NWDAF).
    pub exports_refused: u64,
    /// The NWDAF the exporter posts to.
    pub nwdaf: Option<String>,
}

/// The process's exporter status (one exporter per gNB process).
static STATUS: OnceLock<Mutex<Status>> = OnceLock::new();

/// The process's exporter status cell.
pub fn status_cell() -> &'static Mutex<Status> {
    STATUS.get_or_init(|| Mutex::new(Status::default()))
}

/// Update `cell` under its lock.
pub fn update_status(cell: &Mutex<Status>, f: impl FnOnce(&mut Status)) {
    let mut g = cell
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    f(&mut g);
}

/// The `congestion-status` answer, as JSON. Refused when the demo is not active.
pub fn status_json() -> Result<String, String> {
    if !demo_active() {
        return Err(format!(
            "congestion-status: demo {D441A_KEY} is not active in this gNB"
        ));
    }
    let g = status_cell()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut v = serde_json::to_value(&*g).map_err(|e| e.to_string())?;
    v["schema"] = serde_json::json!("nextgsim.congestion-status/1");
    serde_json::to_string_pretty(&v).map_err(|e| e.to_string())
}

/// `true` when `command` is `congestion-status`.
pub fn is_status_command(command: &str) -> bool {
    command.split_whitespace().next() == Some("congestion-status")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn active_config() -> GnbConfig {
        let mut c = GnbConfig::default();
        c.congestion_export.cell_capacity_mbps = Some(5.0);
        c.congestion_export.nrf_uri = Some("http://nrf:7777".into());
        c
    }

    #[test]
    fn a_complete_block_validates() {
        assert_eq!(validate_config(&active_config()), Ok(()));
    }

    /// An active demo with no capacity or no NRF fails startup by name.
    #[test]
    fn a_missing_capacity_or_nrf_fails() {
        let mut c = active_config();
        c.congestion_export.cell_capacity_mbps = None;
        assert!(validate_config(&c)
            .unwrap_err()
            .contains("cell_capacity_mbps"));
        let mut c = active_config();
        c.congestion_export.cell_capacity_mbps = Some(0.0);
        assert!(validate_config(&c).unwrap_err().contains("positive"));
        let mut c = active_config();
        c.congestion_export.nrf_uri = None;
        assert!(validate_config(&c).unwrap_err().contains("nrf_uri"));
    }

    /// The block rejects a misspelt field, as the other demo blocks do.
    #[test]
    fn the_block_rejects_unknown_fields() {
        let err = serde_yaml::from_str::<nextgsim_common::config::CongestionExportConfig>(
            "cell_capacity_mbs: 5\n",
        )
        .unwrap_err();
        assert!(err.to_string().contains("unknown field"), "{err}");
    }

    /// Occupancy is served bits over C x window: 4.8 Mbps of 5 Mbps is 96 %.
    #[test]
    fn occupancy_is_served_over_capacity() {
        let s = WindowSample {
            end: SystemTime::UNIX_EPOCH,
            elapsed: Duration::from_secs(1),
            served_octets: 600_000,
            connected_ues: 2,
            queue_depth_bytes: 0,
            queue_drops: 0,
        };
        assert!((s.served_kbps() - 4800.0).abs() < 1e-9);
        assert!((s.occupancy_pct(5.0) - 96.0).abs() < 1e-9);
        // Never above 100 % (a window that ran short of its tokens' refill).
        let over = WindowSample {
            served_octets: 2_000_000,
            ..s
        };
        assert_eq!(over.occupancy_pct(5.0), 100.0);
    }
}
