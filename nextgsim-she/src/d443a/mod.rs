//! Demo D443A (nextgsim#471): the EES picks the edge server on a UPF-measured
//! communication + compute budget. **Non-normative.** This is the nextgsim
//! side; the ranking is eesd's (nextgcore, `ees.edge_budget`).
//!
//! # What is here
//! - [`eas`]: the `nr-she-eas` EAS sidecar. It registers at the EES over
//!   EDGE-3 (TS 29.558 `Eees_EASRegistration`, `CreateEASRegistration`) with
//!   its `svcKpi` (`maxRespTime`, `avlComp`), checks the registration and
//!   re-creates it if the EES lost it, merge-patches `svcKpi.avlComp` when load
//!   is applied (`ModifyIndEASRegistration`), and deregisters on shutdown.
//! - [`stamp`]: an RFC 8762 STAMP Session-Reflector (stateless,
//!   unauthenticated), so upfd's N6 delay measurement (TS 29.244 §5.33.8) has
//!   a far end at the EAS's address.
//! - [`job`]: the demo's job exchange, a fixed 64-octet UDP frame each way.
//!   The EAS "computes" by sleeping `ceil(reqComp / avlComp)` ms.
//! - [`eel`]: the EDGE-1 / EDGE-3 bodies (TS 24.558 `EECRegistration`,
//!   TS 29.558 `EASRegistration`) and how a selection is read back.
//! - [`eec`]: the UE's EEC client: an `Eees_EECRegistration` with
//!   `easSelReqInd: true` and one AC profile, through the PDU session; then
//!   the job on the first `discoveredEas`, or on-device on a 404.
//!
//! # Honesty labels
//! - The compute term is a sleep stand-in, not inference (that is #474).
//! - `reqComp` is a string in TS 24.558; here it is an integer.
//! - TS 29.558 `EndPoint` has no port member, so the job port is the demo's
//!   ([`job::DEFAULT_PORT`] unless configured); `endPt` carries only
//!   `ipv4Addrs`, which is also the N6 measurement endpoint eesd falls back to.
//! - STAMP is unauthenticated: a departure from TS 33.501 clause T.5. The
//!   reflector and the job listener bind the `dn` (or pod) address only, never
//!   a published port.
//! - The STAMP `Session-Sender TTL` is 255, not the received TTL (as the
//!   nextgcore #1389 responder does).
//! - Tokens come from the NRF (client credentials with a CCA). A real EEC
//!   authenticates through the ECS/EES (TS 33.558); the NRF stands in as the
//!   token issuer.
//! - `achievableLatency` and `budgetSrc` are read from eesd's TS 29.500 §6.6.3
//!   vendor-specific member [`VENDOR_MEMBER`] on `DiscoveredEas`.
//! - The UE's two modes behave identically: the mode is eesd's ranking. The UE
//!   registers both so `mode: bogus` fails it too, and logs the mode with each
//!   job.

pub mod eas;
pub mod eec;
pub mod eel;
pub mod job;
pub mod stamp;
#[cfg(test)]
pub(crate) mod stub_ees;
#[cfg(test)]
mod tests;

use std::time::{Duration, SystemTime};

use nextgsim_common::config::NrfOAuth2Config;
use nextgsim_sbi_client::{CcaKey, OAuth2Config, SbiClient, SbiClientConfig};

/// The demo key (nextgsim#458 convention); eesd and smfd use the same.
pub const KEY: &str = "d443a-edge-budget";
/// `measured-budget` (the demo) and `static` (the baseline), as eesd's.
pub const MODES: &[&str] = &["measured-budget", "static"];

/// The TS 29.500 §6.6.3 vendor-specific member eesd writes on `DiscoveredEas`
/// (PEN 32473, the RFC 5612 documentation PEN).
pub const VENDOR_MEMBER: &str = "vendorSpecific-032473";

/// The token audience (`targetNfType`) of the EES's APIs.
pub const EES_NF_TYPE: &str = "EES";
/// eesd's OAuth2 scope of `Eees_EASRegistration`.
pub const SCOPE_EASREGISTRATION: &str = "eees-easregistration";
/// eesd's OAuth2 scope of `Eees_EECRegistration`.
pub const SCOPE_EECREGISTRATION: &str = "eees-eecregistration";

/// The EAS registrations collection.
pub const EASREG_PATH: &str = "/eees-easregistration/v1/registrations";
/// The EEC registrations collection.
pub const EECREG_PATH: &str = "/eees-eecregistration/v1/registrations";

/// The URL of a `Location` the EES answered at `api_root`: a relative one is
/// resolved against it; an absolute one is followed only on the same origin
/// (it carries the bearer token), as D445A's NEF client does. `None`: not
/// followed.
pub fn resolve_location(api_root: &str, location: &str) -> Option<String> {
    let root = api_root.trim_end_matches('/');
    if location.starts_with('/') {
        return Some(format!("{root}{location}"));
    }
    location
        .strip_prefix(root)
        .filter(|rest| rest.starts_with('/'))
        .map(|_| location.to_string())
}

/// NTP 64-bit timestamp (RFC 5905 §6) of `t`, as upfd stamps its probes.
pub fn ntp64(t: SystemTime) -> u64 {
    const NTP_UNIX_OFFSET: u64 = 2_208_988_800;
    let d = t.duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    let secs = d.as_secs().wrapping_add(NTP_UNIX_OFFSET) & 0xffff_ffff;
    let frac = (u64::from(d.subsec_nanos()) << 32) / 1_000_000_000;
    (secs << 32) | frac
}

/// An SBI client toward the EES: h2c, bound to `device` if given (the token
/// requests too), with NRF OAuth2 if `oauth2` is set. With `oauth2.register`,
/// registers `nfType: AF` at the NRF first (tokenless, as nextgcore's NRF
/// management API is), on the same binding.
pub async fn ees_client(
    oauth2: Option<&NrfOAuth2Config>,
    device: Option<&str>,
    timeout: Duration,
) -> Result<SbiClient, String> {
    let base = SbiClientConfig {
        timeout,
        bind_device: device.map(str::to_string),
        ..SbiClientConfig::default()
    };
    let Some(o) = oauth2 else {
        return Ok(SbiClient::new(base));
    };
    let key = CcaKey::load_or_create(&o.cca_key_file).map_err(|e| e.to_string())?;
    if let Some(dir) = &o.public_jwk_dir {
        key.publish_jwk(dir, &o.nf_instance_id)
            .map_err(|e| e.to_string())?;
    }
    if o.register {
        let hb = register_af(&o.nrf, &o.nf_instance_id, base.clone()).await?;
        // The NRF suspends a profile whose heart-beat lapses, and then refuses
        // this client's next token request: NF Heart-Beat (TS 29.510
        // §5.2.2.3.2) at half the heartBeatTimer the NRF answered.
        tokio::spawn(keep_registered(
            o.nrf.clone(),
            o.nf_instance_id.clone(),
            base.clone(),
            beat_every(hb),
        ));
    }
    Ok(SbiClient::new(SbiClientConfig {
        oauth2: Some(OAuth2Config::new(
            o.nrf.clone(),
            o.nf_instance_id.clone(),
            "AF",
            key,
        )),
        ..base
    }))
}

pub(crate) use nextgsim_sbi_client::nrf::{beat_every, keep_registered, register_af};

/// The `heartBeatTimer` the client registers with (s); it beats every half of
/// the one the NRF answers (shared with the D445A AF, #556).
pub const HEARTBEAT_SECS: u64 = nextgsim_sbi_client::nrf::HEARTBEAT_SECS;

#[cfg(test)]
pub(crate) async fn register_af_for_test(
    nrf: &str,
    id: &str,
    base: SbiClientConfig,
) -> Result<u64, String> {
    register_af(nrf, id, base).await
}
