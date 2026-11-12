//! Demo D444A (nextgsim#477): the UE application adapts its UL feature stream
//! to the Session-AMBR the network signals. **Non-normative.**
//!
//! The NAS layer applies a PDU SESSION MODIFICATION COMMAND's Session-AMBR and
//! reports it upward (#454, `SessionParametersChanged`); the App task forwards
//! it here. In mode `semantic` the stream re-plans `keep` from the new uplink
//! rate ([`keep_for_link_budget`]) and logs
//! `[demo:d444a-ambr-adapt] rate target … from session-AMBR (PSI 1), keep 128->40`.
//! In mode `raw` (the control) it ignores the AMBR and sends all 128 raw
//! features, which the UPF's QER then polices.
//!
//! TS 23.501 §5.7.2.6 makes the UE responsible for UL Session-AMBR
//! enforcement. This stream is an application choosing to stay under it, not
//! that enforcement.

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use nextgsim_common::config::SemanticStreamConfig;
use nextgsim_demo::demo_log;
use nextgsim_nas::messages::sm::IeSessionAmbr;
use nextgsim_semantic::rate_adapt::{
    datagram_ip_bytes, keep_for_link_budget, rate_target_bps, D444aModel, EncoderId,
    FeatureDatagram, D444A_KEY, FEATURE_DIM,
};
use tokio::net::UdpSocket;

/// The uplink Session-AMBR in bit/s (TS 24.501 Table 9.11.4.14.1).
///
/// The unit names are decimal: 1 Kbps is 1 000 bit/s, 1 Mbps 1 000 000 (as
/// nextgcore's smfd encodes them). Units 1-5 are 1/4/16/64/256 Kbps, 6-10 the
/// same in Mbps, then Gbps, Tbps and Pbps; values above 25 read as 256 Pbps.
/// Unit 0 is "value is not used": `None`.
pub fn session_ambr_uplink_bps(ambr: &IeSessionAmbr) -> Option<u64> {
    bit_rate(ambr.uplink_unit, ambr.uplink)
}

fn bit_rate(unit: u8, value: u16) -> Option<u64> {
    if unit == 0 {
        return None;
    }
    let unit = u32::from(unit.min(25)) - 1;
    let multiple = 4u64.pow(unit % 5) * 1000u64.pow(unit / 5 + 1);
    Some(u64::from(value).saturating_mul(multiple))
}

/// What `semantic-status` reports.
#[derive(Debug, Clone, Default)]
pub struct StreamStatus {
    /// The mode in force for the current epoch.
    pub mode: String,
    /// Features per datagram now.
    pub keep: usize,
    /// The last uplink Session-AMBR the network signalled for the stream's PSI.
    pub ul_ambr_bps: Option<u64>,
    /// Datagrams sent.
    pub sent: u64,
    /// Datagrams the socket refused.
    pub send_errors: u64,
}

/// The stream's state: encoder selection, `keep`, the sequence number.
pub struct D444aStream {
    config: SemanticStreamConfig,
    model: D444aModel,
    mode: String,
    keep: usize,
    ul_ambr_bps: Option<u64>,
    seq: u64,
    sent: u64,
    send_errors: u64,
}

impl D444aStream {
    /// A stream for `config`, starting in `mode` with all 128 features.
    pub fn new(config: SemanticStreamConfig, mode: &str) -> Self {
        let model = D444aModel::new(config.seed);
        Self {
            config,
            model,
            mode: mode.to_string(),
            keep: FEATURE_DIM,
            ul_ambr_bps: None,
            seq: 0,
            sent: 0,
            send_errors: 0,
        }
    }

    /// The `keep` the current mode and AMBR call for.
    fn planned_keep(&self) -> usize {
        match (self.mode.as_str(), self.ul_ambr_bps) {
            ("semantic", Some(ul)) if self.config.follow_session_ambr => {
                keep_for_link_budget(ul, self.config.rate_hz)
            }
            _ => FEATURE_DIM,
        }
    }

    fn replan(&mut self, why: &str) -> String {
        let old = self.keep;
        self.keep = self.planned_keep();
        match self.ul_ambr_bps {
            Some(ul) if self.mode == "semantic" && self.config.follow_session_ambr => format!(
                "rate target {} bps from session-AMBR (PSI {}), keep {old}->{} (UL Session-AMBR \
                 {ul} bps, {} Hz, {} B per datagram{why})",
                rate_target_bps(ul),
                self.config.psi,
                self.keep,
                self.config.rate_hz,
                datagram_ip_bytes(self.keep)
            ),
            Some(ul) => format!(
                "session-AMBR (PSI {}) UL {ul} bps not followed: {}, keep {}{why}",
                self.config.psi,
                if self.mode == "semantic" {
                    "follow_session_ambr is false".to_string()
                } else {
                    format!("mode {}", self.mode)
                },
                self.keep
            ),
            None => format!("mode {}, keep {}{why}", self.mode, self.keep),
        }
    }

    /// The network changed PDU session `psi`'s Session-AMBR. Returns the demo
    /// log line, or `None` for another session or an unusable value.
    pub fn on_session_ambr(&mut self, psi: u8, ambr: &IeSessionAmbr) -> Option<String> {
        if psi != self.config.psi {
            return None;
        }
        self.ul_ambr_bps = Some(session_ambr_uplink_bps(ambr)?);
        Some(self.replan(""))
    }

    /// Re-read the mode at an epoch boundary (a runtime `nr-cli … demo set`
    /// applies from the next epoch). Returns a log line when it changed.
    pub fn on_epoch(&mut self, mode: &str) -> Option<String> {
        if mode == self.mode {
            return None;
        }
        self.mode = mode.to_string();
        let epoch = self.seq / self.config.epoch_datagrams.max(1);
        Some(self.replan(&format!(", mode switched from epoch {epoch}")))
    }

    /// The next datagram's bytes.
    pub fn next_datagram(&mut self) -> Vec<u8> {
        let (label, x) = self.model.generator().sample(self.seq);
        let encoder = if self.mode == "semantic" {
            EncoderId::Pca
        } else {
            EncoderId::Raw
        };
        let d = FeatureDatagram {
            encoder,
            seq: self.seq as u32,
            label,
            values: self.model.encoder(encoder).encode(&x, self.keep),
        };
        self.seq += 1;
        d.encode()
    }

    /// Whether the next datagram starts an epoch.
    pub fn at_epoch_boundary(&self) -> bool {
        self.seq.is_multiple_of(self.config.epoch_datagrams.max(1))
    }

    /// Features per datagram now.
    pub fn keep(&self) -> usize {
        self.keep
    }

    /// The status `semantic-status` prints.
    pub fn status(&self) -> StreamStatus {
        StreamStatus {
            mode: self.mode.clone(),
            keep: self.keep,
            ul_ambr_bps: self.ul_ambr_bps,
            sent: self.sent,
            send_errors: self.send_errors,
        }
    }
}

/// The running stream, for `semantic-status`. One per UE process: the demo
/// streams from the first UE instance only.
static STREAM: Mutex<Option<Arc<Mutex<D444aStream>>>> = Mutex::new(None);

/// Answer `semantic-status`; `None` for any other command.
pub fn route_cli(command: &str) -> Option<String> {
    let word = command.split_whitespace().next()?;
    if !word.eq_ignore_ascii_case("semantic-status") {
        return None;
    }
    let stream = STREAM.lock().ok().and_then(|s| s.clone());
    Some(match stream {
        None => format!("{} stream not running", nextgsim_demo::prefix(D444A_KEY)),
        Some(s) => {
            let st = s.lock().map(|s| s.status()).unwrap_or_default();
            format!(
                "{} mode={} keep={} ul_session_ambr_bps={} sent={} send_errors={}",
                nextgsim_demo::prefix(D444A_KEY),
                st.mode,
                st.keep,
                st.ul_ambr_bps
                    .map_or_else(|| "unknown".to_string(), |v| v.to_string()),
                st.sent,
                st.send_errors
            )
        }
    })
}

/// Start the stream if demo D444A is active and configured: `None` otherwise,
/// so a UE without the demo is unchanged.
pub fn start(config: Option<&SemanticStreamConfig>) -> Option<Arc<Mutex<D444aStream>>> {
    let mode = nextgsim_demo::active_mode(D444A_KEY)?;
    let Some(config) = config.cloned() else {
        demo_log!(
            warn,
            D444A_KEY,
            "demo enabled but the UE config has no semantic_stream block: nothing is streamed"
        );
        return None;
    };
    let mut slot = STREAM.lock().ok()?;
    if slot.is_some() {
        return None; // another UE instance of this process streams already
    }
    demo_log!(
        info,
        D444A_KEY,
        "UL feature stream to {} via {} at {} Hz, mode {mode}, keep {FEATURE_DIM}, follow_session_ambr={}",
        config.sink,
        config.device.as_deref().unwrap_or("the routing table"),
        config.rate_hz,
        config.follow_session_ambr
    );
    let stream = Arc::new(Mutex::new(D444aStream::new(config, &mode)));
    *slot = Some(stream.clone());
    tokio::spawn(run(stream.clone()));
    Some(stream)
}

async fn open_socket(device: Option<&str>) -> std::io::Result<UdpSocket> {
    let socket = UdpSocket::bind(SocketAddr::from(([0, 0, 0, 0], 0))).await?;
    if let Some(dev) = device {
        socket.bind_device(Some(dev.as_bytes()))?;
    }
    Ok(socket)
}

async fn run(stream: Arc<Mutex<D444aStream>>) {
    let (sink, device, rate_hz) = {
        let Ok(s) = stream.lock() else { return };
        (
            s.config.sink,
            s.config.device.clone(),
            s.config.rate_hz.max(1),
        )
    };
    // Until the PDU session's TUN exists the bind fails; retry each second.
    let mut warned = false;
    let socket = loop {
        match open_socket(device.as_deref()).await {
            Ok(s) => break s,
            Err(e) => {
                if !warned {
                    demo_log!(
                        info,
                        D444A_KEY,
                        "waiting for {}: {e}",
                        device.as_deref().unwrap_or("a socket")
                    );
                    warned = true;
                }
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        }
    };
    let mut tick = tokio::time::interval(Duration::from_secs_f64(1.0 / f64::from(rate_hz)));
    // A late tick is skipped rather than burst: a burst is what a policer drops.
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tick.tick().await;
        let bytes = {
            let Ok(mut s) = stream.lock() else { return };
            if s.at_epoch_boundary() {
                if let Some(mode) = nextgsim_demo::active_mode(D444A_KEY) {
                    if let Some(line) = s.on_epoch(&mode) {
                        demo_log!(info, D444A_KEY, "{line}");
                    }
                }
            }
            s.next_datagram()
        };
        let sent = socket.send_to(&bytes, sink).await;
        if let Ok(mut s) = stream.lock() {
            match sent {
                Ok(_) => s.sent += 1,
                Err(e) => {
                    s.send_errors += 1;
                    if s.send_errors.is_power_of_two() {
                        demo_log!(
                            warn,
                            D444A_KEY,
                            "send to {sink}: {e} ({} so far)",
                            s.send_errors
                        );
                    }
                }
            }
        }
    }
}

/// Apply a Session-AMBR change to the running stream, if any.
pub fn on_session_ambr(stream: Option<&Arc<Mutex<D444aStream>>>, psi: u8, ambr: &IeSessionAmbr) {
    let Some(stream) = stream else { return };
    let line = stream
        .lock()
        .ok()
        .and_then(|mut s| s.on_session_ambr(psi, ambr));
    if let Some(line) = line {
        demo_log!(info, D444A_KEY, "{line}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> SemanticStreamConfig {
        serde_yaml::from_str("sink: 127.0.0.1:5444\ndevice: uesimtun0\n").unwrap()
    }

    /// TS 24.501 Table 9.11.4.14.1, decimal units: what smfd's
    /// `encode_ambr_component` emits for 64 kbps (unit 1, 64) and 2 Mbps (unit 6, 2).
    #[test]
    fn the_uplink_session_ambr_reads_the_unit_table_in_decimal() {
        let bps = |u, v| session_ambr_uplink_bps(&IeSessionAmbr::new(6, 1, u, v));
        assert_eq!(bps(1, 64), Some(64_000));
        assert_eq!(bps(6, 2), Some(2_000_000));
        assert_eq!(bps(2, 3), Some(12_000));
        assert_eq!(bps(5, 1), Some(256_000));
        assert_eq!(bps(7, 1), Some(4_000_000));
        assert_eq!(bps(11, 1), Some(1_000_000_000));
        assert_eq!(bps(21, 1), Some(1_000_000_000_000_000));
        assert_eq!(bps(25, 1), bps(200, 1), "units above 25 read as 256 Pbps");
        assert_eq!(bps(0, 64), None, "unit 0: value is not used");
    }

    /// The success signal: a 2 Mbps -> 64 kbps throttle in mode `semantic`
    /// re-plans keep 128 -> 40, and the next datagram carries 40 PCA features.
    #[test]
    fn semantic_mode_follows_a_throttle_to_keep_40() {
        let mut s = D444aStream::new(config(), "semantic");
        assert_eq!(s.keep(), 128);
        let line = s
            .on_session_ambr(1, &IeSessionAmbr::new(6, 2, 6, 2))
            .unwrap();
        assert_eq!(s.keep(), 128, "2 Mbps carries all 128");
        assert!(line.contains("keep 128->128"), "{line}");
        let line = s
            .on_session_ambr(1, &IeSessionAmbr::new(6, 2, 1, 64))
            .unwrap();
        assert!(
            line.starts_with("rate target 51200 bps from session-AMBR (PSI 1), keep 128->40"),
            "{line}"
        );
        let d = FeatureDatagram::decode(&s.next_datagram()).unwrap();
        assert_eq!((d.encoder, d.values.len()), (EncoderId::Pca, 40));
        // Restore.
        let line = s
            .on_session_ambr(1, &IeSessionAmbr::new(6, 2, 6, 2))
            .unwrap();
        assert!(line.contains("keep 40->128"), "{line}");
        // Another PDU session's change is not this stream's.
        assert_eq!(s.on_session_ambr(2, &IeSessionAmbr::new(6, 2, 1, 64)), None);
        assert_eq!(s.keep(), 128);
    }

    /// The raw control and the `follow_session_ambr: false` revert arm keep 128
    /// and never log a rate target.
    #[test]
    fn raw_mode_and_follow_false_ignore_the_throttle() {
        let mut raw = D444aStream::new(config(), "raw");
        let line = raw
            .on_session_ambr(1, &IeSessionAmbr::new(6, 2, 1, 64))
            .unwrap();
        assert!(!line.contains("rate target"), "{line}");
        assert!(line.contains("not followed: mode raw, keep 128"), "{line}");
        let d = FeatureDatagram::decode(&raw.next_datagram()).unwrap();
        assert_eq!((d.encoder, d.values.len()), (EncoderId::Raw, 128));

        let mut cfg = config();
        cfg.follow_session_ambr = false;
        let mut off = D444aStream::new(cfg, "semantic");
        let line = off
            .on_session_ambr(1, &IeSessionAmbr::new(6, 2, 1, 64))
            .unwrap();
        assert!(
            line.contains("not followed: follow_session_ambr is false"),
            "{line}"
        );
        assert_eq!(off.keep(), 128);
    }

    /// A runtime mode switch applies at the next epoch boundary and re-plans
    /// from the AMBR already known.
    #[test]
    fn a_mode_switch_applies_from_the_next_epoch() {
        let mut cfg = config();
        cfg.epoch_datagrams = 4;
        let mut s = D444aStream::new(cfg, "raw");
        s.on_session_ambr(1, &IeSessionAmbr::new(6, 2, 1, 64));
        assert!(s.at_epoch_boundary());
        assert_eq!(s.on_epoch("raw"), None);
        for _ in 0..4 {
            s.next_datagram();
        }
        assert!(s.at_epoch_boundary());
        let line = s.on_epoch("semantic").unwrap();
        assert!(line.contains("keep 128->40"), "{line}");
        assert!(line.contains("mode switched from epoch 1"), "{line}");
    }
}
