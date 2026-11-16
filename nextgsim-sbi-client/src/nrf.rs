//! Keeping an AF's NF profile registered at the NRF (#556): NFRegister
//! (TS 29.510 §5.2.2.2) once, then the NF Heart-Beat (§5.2.2.3.2) for the life
//! of the process.
//!
//! The NRF suspends a profile whose heart-beat lapses and then refuses its
//! token requests (nextgcore nrfd: SUSPENDED at 2 x heartBeatTimer,
//! deregistered later; `invalid_client` after). A consumer that registers once
//! and never beats works only until its cached tokens need refreshing.
//!
//! Registration is tokenless, as nextgcore's NRF management API is.

use std::time::Duration;

use bytes::Bytes;
use http::{HeaderValue, Method, StatusCode};

use crate::{SbiClient, SbiClientConfig};

/// The `heartBeatTimer` (s) an AF registers with. The NRF may answer another
/// (§5.2.2.2.2), which then rules.
pub const HEARTBEAT_SECS: u64 = 60;

/// The interval to beat at for a `heartbeat_s` heartBeatTimer: half of it, at
/// least 1 s, so one lost beat does not lapse the profile.
pub fn beat_every(heartbeat_s: u64) -> Duration {
    Duration::from_millis((heartbeat_s.saturating_mul(1000) / 2).max(1000))
}

fn instance_uri(nrf: &str, id: &str) -> String {
    format!(
        "{}/nnrf-nfm/v1/nf-instances/{id}",
        nrf.trim_end_matches('/')
    )
}

/// The heartBeatTimer in an NFProfile body, if any (and non-zero).
fn heartbeat_of(body: &[u8]) -> Option<u64> {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()?
        .get("heartBeatTimer")?
        .as_u64()
        .filter(|hb| *hb > 0)
}

/// NFRegister `id` as `nfType: AF` (PUT, §5.2.2.2). Returns the heartBeatTimer
/// the NRF answered (it may change the requested one), or the requested one
/// if it answered none.
pub async fn register_af(nrf: &str, id: &str, base: SbiClientConfig) -> Result<u64, String> {
    let plain = SbiClient::new(base);
    let profile = serde_json::json!({
        "nfInstanceId": id,
        "nfType": "AF",
        "nfStatus": "REGISTERED",
        "heartBeatTimer": HEARTBEAT_SECS,
    });
    let r = plain
        .put_json(&instance_uri(nrf, id), &profile, None)
        .await
        .map_err(|e| format!("NRF register: {e}"))?;
    if !r.status.is_success() {
        return Err(format!("NRF register: status {}", r.status.as_u16()));
    }
    Ok(heartbeat_of(&r.body).unwrap_or(HEARTBEAT_SECS))
}

/// The outcome of one NF Heart-Beat.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Beat {
    /// Accepted; `Some` when the NRF answered a profile (200) carrying a
    /// heartBeatTimer, which then rules (§5.2.2.3.2: a changed interval is
    /// returned on the next heart-beat).
    Alive(Option<u64>),
    /// 404: the NRF no longer holds the profile (§5.2.2.3.2 2b), so it must be
    /// registered again.
    NotRegistered,
}

/// One NF Heart-Beat (§5.2.2.3.2): PATCH the instance with a JSON Patch
/// "replace" of `nfStatus` to `REGISTERED`, no If-Match.
pub async fn heartbeat(nrf: &str, id: &str, base: SbiClientConfig) -> Result<Beat, String> {
    let plain = SbiClient::new(base);
    let patch =
        Bytes::from_static(br#"[{"op":"replace","path":"/nfStatus","value":"REGISTERED"}]"#);
    let r = plain
        .request_typed(
            Method::PATCH,
            &instance_uri(nrf, id),
            Some(patch),
            HeaderValue::from_static("application/json-patch+json"),
            None,
        )
        .await
        .map_err(|e| format!("NRF heart-beat: {e}"))?;
    match r.status {
        StatusCode::NOT_FOUND => Ok(Beat::NotRegistered),
        s if s.is_success() => Ok(Beat::Alive(heartbeat_of(&r.body))),
        s => Err(format!("NRF heart-beat: status {}", s.as_u16())),
    }
}

/// Keep `id` registered for the life of the task: every `every` (then at
/// [`beat_every`] of the heartBeatTimer the NRF last answered), one
/// heart-beat; a profile the NRF no longer holds is registered again. A failed
/// round is logged and retried at the same interval.
pub async fn keep_registered(nrf: String, id: String, base: SbiClientConfig, mut every: Duration) {
    loop {
        tokio::time::sleep(every).await;
        match heartbeat(&nrf, &id, base.clone()).await {
            Ok(Beat::Alive(Some(hb))) => every = beat_every(hb),
            Ok(Beat::Alive(None)) => {}
            Ok(Beat::NotRegistered) => {
                tracing::info!("NRF no longer holds {id}: registering it again");
                match register_af(&nrf, &id, base.clone()).await {
                    Ok(hb) => every = beat_every(hb),
                    Err(e) => tracing::warn!("NRF re-registration of {id}: {e}"),
                }
            }
            Err(e) => tracing::warn!("NRF heart-beat of {id}: {e}"),
        }
    }
}
