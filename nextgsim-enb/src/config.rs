//! eNB configuration.
//!
//! Field names and YAML shape follow the gNB's (`plmn`, `tac`, `*_configs`,
//! `*_ip`) so one operator vocabulary covers both. Every value that reaches the
//! wire is validated here against the S1AP type it is encoded into, so a bad
//! config fails at load time rather than as an encode error mid-procedure.

use std::fs;
use std::net::IpAddr;
use std::path::Path;

use nextgsim_common::Plmn;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// The S1AP `ENB-ID` CHOICE alternative the eNB identifies itself with
/// (TS 36.413 §9.2.1.37, Global eNB ID).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum EnbIdType {
    /// `macroENB-ID`, 20 bits: the eNB ID is the leftmost 20 bits of the
    /// 28-bit E-UTRAN Cell Identity.
    #[default]
    Macro,
    /// `homeENB-ID`, 28 bits: a HeNB, whose eNB ID equals the cell identity.
    Home,
    /// `short-macroENB-ID`, 18 bits (extension alternative).
    ShortMacro,
    /// `long-macroENB-ID`, 21 bits (extension alternative).
    LongMacro,
}

impl EnbIdType {
    /// The BIT STRING length of this alternative.
    pub const fn bits(self) -> u32 {
        match self {
            Self::Macro => 20,
            Self::Home => 28,
            Self::ShortMacro => 18,
            Self::LongMacro => 21,
        }
    }
}

/// Default Paging DRX (TS 36.413 §9.2.1.16; TS 36.304 §7.1), in radio frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PagingDrx {
    /// 32 radio frames.
    #[serde(rename = "v32")]
    V32,
    /// 64 radio frames.
    #[serde(rename = "v64")]
    V64,
    /// 128 radio frames.
    #[default]
    #[serde(rename = "v128")]
    V128,
    /// 256 radio frames.
    #[serde(rename = "v256")]
    V256,
}

impl PagingDrx {
    /// The `PagingDRX` ENUMERATED index.
    pub const fn index(self) -> u8 {
        match self {
            Self::V32 => 0,
            Self::V64 => 1,
            Self::V128 => 2,
            Self::V256 => 3,
        }
    }
}

/// One MME the eNB connects to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MmeConfig {
    /// MME S1-MME address.
    pub address: IpAddr,
    /// SCTP destination port; IANA assigns 36412 to S1AP (TS 36.412 §7).
    #[serde(default = "default_mme_port")]
    pub port: u16,
}

fn default_mme_port() -> u16 {
    nextgsim_sctp::S1AP_SCTP_PORT
}

/// One entry of the Supported TAs IE (TS 36.413 §9.1.8.4).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SupportedTa {
    /// Tracking Area Code, 16 bits (TS 36.413 §9.2.3.7, `TAC ::= OCTET STRING (SIZE(2))`).
    pub tac: u16,
    /// PLMNs broadcast in this TA, 1..6 (`BPLMNs`, maxnoofBPLMNs = 6).
    pub broadcast_plmns: Vec<Plmn>,
}

/// `dl-Bandwidth` (TS 36.331 §6.2.2 MasterInformationBlock), in resource blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum DlBandwidthRb {
    /// 6 RBs (1.4 MHz).
    #[serde(rename = "n6")]
    N6,
    /// 15 RBs (3 MHz).
    #[serde(rename = "n15")]
    N15,
    /// 25 RBs (5 MHz).
    #[serde(rename = "n25")]
    N25,
    /// 50 RBs (10 MHz).
    #[default]
    #[serde(rename = "n50")]
    N50,
    /// 75 RBs (15 MHz).
    #[serde(rename = "n75")]
    N75,
    /// 100 RBs (20 MHz).
    #[serde(rename = "n100")]
    N100,
}

/// The E-UTRA cell and its RLS radio endpoint (nextgsim #186, increment 2).
///
/// RLS replaces the physical layer and MAC, exactly as on the NR side, so the
/// radio "interface" is a UDP socket; everything above it -- system
/// information, RRC, SRB PDCP -- is TS 36.331 / TS 36.323.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct RadioConfig {
    /// Address the RLS socket binds to (the UE's search space points here).
    pub link_ip: IpAddr,
    /// RLS UDP port. 4997 is the NR gNB's, so an eNB and a gNB on one host
    /// need different ports or addresses.
    pub rls_port: u16,
    /// The cell's local identity within the eNB: the low `28 - enb_id bits`
    /// bits of the E-UTRAN Cell Identity (TS 36.300 §8.2; TS 36.413 §9.2.1.37:
    /// a macro eNB ID is the leftmost 20 bits of the cell identity). Must be 0
    /// for a home eNB, whose ID *is* the cell identity.
    pub cell_id: u32,
    /// `freqBandIndicator` (SIB1), 1..=64.
    pub freq_band: u8,
    /// `dl-Bandwidth` (MIB).
    pub dl_bandwidth: DlBandwidthRb,
    /// `q-RxLevMin` (SIB1) in dBm: even, -140..=-44 (TS 36.331 §6.3.4).
    pub q_rx_lev_min_dbm: i16,
    /// T300 broadcast in SIB2, ms (TS 36.331 §6.3.6 allows 100, 200, 300, 400,
    /// 600, 1000, 1500, 2000).
    pub t300_ms: u16,
    /// How often MIB, SIB1 and the SIB2 SystemInformation are sent to every
    /// discovered UE, ms. TS 36.331 §5.2.1.2 fixes the MIB at 40 ms and SIB1 at
    /// 80 ms over the air; over RLS each broadcast is a UDP datagram per UE, so
    /// the default (160 ms, the SI periodicity this cell advertises) is the SIB2
    /// period applied to all three.
    pub si_broadcast_period_ms: u64,
}

impl Default for RadioConfig {
    fn default() -> Self {
        Self {
            link_ip: IpAddr::V4(std::net::Ipv4Addr::LOCALHOST),
            rls_port: 4997,
            cell_id: 1,
            freq_band: 7,
            dl_bandwidth: DlBandwidthRb::N50,
            q_rx_lev_min_dbm: -140,
            t300_ms: 1000,
            si_broadcast_period_ms: 160,
        }
    }
}

/// The eNB's AS algorithm preferences (TS 33.401 §7.2.4.2.1: "each eNB shall
/// be configured via network management with lists of algorithms which are
/// allowed for usage", ordered by priority). Identities are TS 33.401 §5.1.3 /
/// §5.1.4: 0 = EEA0/EIA0, 1 = 128-EEA1/EIA1 (SNOW 3G), 2 = 128-EEA2/EIA2
/// (AES), 3 = 128-EEA3/EIA3 (ZUC).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct AsSecurityConfig {
    /// Integrity algorithms, highest priority first. EIA0 is refused: TS 33.401
    /// §5.1.4.2 allows it only for unauthenticated emergency calls, which this
    /// eNB does not serve.
    pub integrity: Vec<u8>,
    /// Ciphering algorithms, highest priority first.
    pub ciphering: Vec<u8>,
}

impl Default for AsSecurityConfig {
    fn default() -> Self {
        Self {
            integrity: vec![2, 1],
            ciphering: vec![2, 1, 0],
        }
    }
}

/// eNB configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnbConfig {
    /// eNB ID, interpreted as `enb_id_type.bits()` bits.
    pub enb_id: u32,
    /// Which `ENB-ID` alternative to send.
    #[serde(default)]
    pub enb_id_type: EnbIdType,
    /// The PLMN of the Global eNB ID.
    pub plmn: Plmn,
    /// Primary Tracking Area Code. Used, broadcasting `plmn`, when
    /// `supported_tas` is empty.
    pub tac: u16,
    /// Explicit Supported TAs. Empty means `[{ tac, [plmn] }]`.
    #[serde(default)]
    pub supported_tas: Vec<SupportedTa>,
    /// Human-readable eNB name sent in S1 SETUP REQUEST (optional IE).
    #[serde(default)]
    pub enb_name: Option<String>,
    /// Default Paging DRX (mandatory IE).
    #[serde(default)]
    pub paging_drx: PagingDrx,
    /// MMEs to connect to.
    pub mme_configs: Vec<MmeConfig>,
    /// Local address the S1-MME SCTP association is bound to.
    pub s1ap_ip: IpAddr,
    /// Seconds to wait before re-initiating S1 Setup after a failure that
    /// carried no Time To Wait, or after a lost association. TS 36.413 §8.7.3.3
    /// only constrains the case where Time To Wait is present, so this is an
    /// implementation choice.
    #[serde(default = "default_s1_setup_retry_secs")]
    pub s1_setup_retry_secs: u64,
    /// Milliseconds to wait for S1 SETUP RESPONSE/FAILURE before treating the
    /// attempt as failed. TS 36.413 defines no timer for S1 Setup; without one a
    /// silent MME would stall the eNB forever.
    #[serde(default = "default_s1_setup_response_timeout_ms")]
    pub s1_setup_response_timeout_ms: u64,
    /// The E-UTRA cell and its RLS endpoint.
    #[serde(default)]
    pub radio: RadioConfig,
    /// AS security algorithm preferences.
    #[serde(default)]
    pub security: AsSecurityConfig,
    /// The eNB's S1-U (GTP-U) address, sent as the E-RAB transport layer
    /// address in INITIAL CONTEXT SETUP RESPONSE (TS 36.413 §9.1.4.2). Defaults
    /// to `s1ap_ip`.
    #[serde(default)]
    pub gtp_ip: Option<IpAddr>,
}

fn default_s1_setup_retry_secs() -> u64 {
    10
}

fn default_s1_setup_response_timeout_ms() -> u64 {
    5000
}

impl EnbConfig {
    /// The 28-bit E-UTRAN Cell Identity: the eNB ID in the leftmost bits, the
    /// cell ID in the rest (TS 36.300 §8.2, TS 36.413 §9.2.1.37).
    pub fn cell_identity(&self) -> u32 {
        let bits = self.enb_id_type.bits();
        if bits >= 28 {
            self.enb_id
        } else {
            (self.enb_id << (28 - bits)) | self.radio.cell_id
        }
    }

    /// The Supported TAs the S1 SETUP REQUEST carries.
    pub fn effective_supported_tas(&self) -> Vec<SupportedTa> {
        if self.supported_tas.is_empty() {
            vec![SupportedTa {
                tac: self.tac,
                broadcast_plmns: vec![self.plmn],
            }]
        } else {
            self.supported_tas.clone()
        }
    }
}

/// Configuration errors.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// The file could not be read.
    #[error("cannot read {path}: {source}")]
    Io {
        /// Path that failed.
        path: String,
        /// Underlying error.
        source: std::io::Error,
    },
    /// The YAML did not deserialize.
    #[error("invalid YAML: {0}")]
    Yaml(#[from] serde_yaml::Error),
    /// A value failed validation.
    #[error("invalid configuration: {0}")]
    Invalid(String),
}

/// Load an eNB configuration file without validating it.
pub fn load_enb_config<P: AsRef<Path>>(path: P) -> Result<EnbConfig, ConfigError> {
    let path = path.as_ref();
    let text = fs::read_to_string(path).map_err(|source| ConfigError::Io {
        path: path.display().to_string(),
        source,
    })?;
    Ok(serde_yaml::from_str(&text)?)
}

/// Load and validate an eNB configuration file.
pub fn load_and_validate_enb_config<P: AsRef<Path>>(path: P) -> Result<EnbConfig, ConfigError> {
    let config = load_enb_config(path)?;
    validate_enb_config(&config)?;
    Ok(config)
}

/// Load an eNB configuration file, apply the `NEXTGSIM_*` environment
/// overrides (see [`apply_env_overrides`]), then validate the result.
///
/// Validation runs after the overrides so a malformed override is refused with
/// the same checks as a malformed file.
pub fn load_with_env_and_validate<P: AsRef<Path>>(path: P) -> Result<EnbConfig, ConfigError> {
    let mut config = load_enb_config(path)?;
    apply_env_overrides(&mut config, |k| std::env::var(k).ok())?;
    validate_enb_config(&config)?;
    Ok(config)
}

/// Override deployment-dependent eNB fields from environment variables, so one
/// baked image config serves any test network (`Dockerfile.enb-local`).
///
/// | Variable | Field(s) |
/// |---|---|
/// | `NEXTGSIM_ENB_MME_ADDR` | `mme_configs` -- comma-separated `ip` or `ip:port` (port 36412 if omitted, TS 36.412 §7) |
/// | `NEXTGSIM_ENB_IP` | `s1ap_ip` and `radio.link_ip` |
/// | `NEXTGSIM_ENB_GTP_IP` | `gtp_ip` (the S1-U address; defaults to `s1ap_ip`) |
/// | `NEXTGSIM_ENB_ID` | `enb_id`, decimal or `0x` hex |
/// | `NEXTGSIM_PLMN` | `plmn`, written `MCC-MNC` |
/// | `NEXTGSIM_TAC` | `tac` |
///
/// `NEXTGSIM_PLMN` and `NEXTGSIM_TAC` only override the top-level PLMN and TAC,
/// so they are refused when the file lists `supported_tas` explicitly: there
/// the advertised TAs would silently disagree with the override. `lookup` is
/// injected so tests need not mutate the process environment.
pub fn apply_env_overrides(
    c: &mut EnbConfig,
    lookup: impl Fn(&str) -> Option<String>,
) -> Result<(), ConfigError> {
    use nextgsim_common::env_override::{parse_endpoints, parse_plmn, parse_u32};
    let bad = |var: &str, e: String| ConfigError::Invalid(format!("{var}: {e}"));
    let ip = |var: &str, v: String| v.trim().parse::<IpAddr>().map_err(|_| bad(var, format!("{v:?} is not an IP address")));

    if let Some(v) = lookup("NEXTGSIM_ENB_MME_ADDR") {
        c.mme_configs = parse_endpoints(&v, 36412)
            .map_err(|e| bad("NEXTGSIM_ENB_MME_ADDR", e))?
            .into_iter()
            .map(|a| MmeConfig { address: a.ip(), port: a.port() })
            .collect();
    }
    if let Some(v) = lookup("NEXTGSIM_ENB_IP") {
        let addr = ip("NEXTGSIM_ENB_IP", v)?;
        c.s1ap_ip = addr;
        c.radio.link_ip = addr;
    }
    if let Some(v) = lookup("NEXTGSIM_ENB_GTP_IP") {
        c.gtp_ip = Some(ip("NEXTGSIM_ENB_GTP_IP", v)?);
    }
    if let Some(v) = lookup("NEXTGSIM_ENB_ID") {
        c.enb_id = parse_u32(v.trim()).map_err(|e| bad("NEXTGSIM_ENB_ID", e))?;
    }
    let plmn = lookup("NEXTGSIM_PLMN");
    let tac = lookup("NEXTGSIM_TAC");
    if (plmn.is_some() || tac.is_some()) && !c.supported_tas.is_empty() {
        return Err(ConfigError::Invalid(
            "NEXTGSIM_PLMN/NEXTGSIM_TAC override the top-level plmn/tac, but the file lists \
             supported_tas explicitly; remove supported_tas or edit it instead"
                .into(),
        ));
    }
    if let Some(v) = plmn {
        c.plmn = parse_plmn(v.trim()).map_err(|e| bad("NEXTGSIM_PLMN", e))?;
    }
    if let Some(v) = tac {
        c.tac = v
            .trim()
            .parse()
            .map_err(|_| bad("NEXTGSIM_TAC", format!("{v:?} is not a 16-bit TAC")))?;
    }
    Ok(())
}

/// maxnoofTACs (TS 36.413 §9.3.6).
const MAX_TACS: usize = 256;
/// maxnoofBPLMNs (TS 36.413 §9.3.6).
const MAX_BPLMNS: usize = 6;

fn validate_plmn(plmn: &Plmn, what: &str) -> Result<(), ConfigError> {
    if plmn.mcc > 999 || plmn.mnc > 999 || (!plmn.long_mnc && plmn.mnc > 99) {
        return Err(ConfigError::Invalid(format!(
            "{what}: PLMN {}-{} is not a valid MCC/MNC (long_mnc={})",
            plmn.mcc, plmn.mnc, plmn.long_mnc
        )));
    }
    Ok(())
}

/// X.680 §41.4 PrintableString alphabet.
fn is_printable_string(s: &str) -> bool {
    s.chars()
        .all(|c| c.is_ascii_alphanumeric() || " '()+,-./:=?".contains(c))
}

/// Validate an eNB configuration against the S1AP types it is encoded into.
pub fn validate_enb_config(c: &EnbConfig) -> Result<(), ConfigError> {
    let bits = c.enb_id_type.bits();
    if u64::from(c.enb_id) >= (1u64 << bits) {
        return Err(ConfigError::Invalid(format!(
            "enb_id {:#x} does not fit the {bits}-bit {:?} eNB ID",
            c.enb_id, c.enb_id_type
        )));
    }
    validate_plmn(&c.plmn, "plmn")?;

    let tas = c.effective_supported_tas();
    if tas.len() > MAX_TACS {
        return Err(ConfigError::Invalid(format!(
            "{} supported TAs exceed maxnoofTACs ({MAX_TACS})",
            tas.len()
        )));
    }
    for ta in &tas {
        if ta.broadcast_plmns.is_empty() || ta.broadcast_plmns.len() > MAX_BPLMNS {
            return Err(ConfigError::Invalid(format!(
                "TAC {}: broadcast PLMN count {} is outside 1..={MAX_BPLMNS}",
                ta.tac,
                ta.broadcast_plmns.len()
            )));
        }
        for plmn in &ta.broadcast_plmns {
            validate_plmn(plmn, &format!("TAC {}", ta.tac))?;
        }
    }

    if let Some(name) = &c.enb_name {
        if name.is_empty() || name.chars().count() > 150 || !is_printable_string(name) {
            return Err(ConfigError::Invalid(format!(
                "enb_name {name:?} must be a 1..150 character PrintableString"
            )));
        }
    }
    if c.mme_configs.is_empty() {
        return Err(ConfigError::Invalid("mme_configs is empty".into()));
    }
    if c.s1_setup_response_timeout_ms == 0 {
        return Err(ConfigError::Invalid(
            "s1_setup_response_timeout_ms must be > 0".into(),
        ));
    }
    validate_radio(c)?;
    validate_security(&c.security)
}

fn validate_security(s: &AsSecurityConfig) -> Result<(), ConfigError> {
    if s.integrity.is_empty() || s.ciphering.is_empty() {
        return Err(ConfigError::Invalid(
            "security.integrity and security.ciphering must not be empty".into(),
        ));
    }
    if let Some(a) = s.integrity.iter().chain(&s.ciphering).find(|a| **a > 3) {
        return Err(ConfigError::Invalid(format!(
            "security: algorithm identity {a} is not one of 0..=3"
        )));
    }
    if s.integrity.contains(&0) {
        return Err(ConfigError::Invalid(
            "security.integrity: EIA0 is only for unauthenticated emergency calls (TS 33.401 §5.1.4.2)"
                .into(),
        ));
    }
    Ok(())
}

fn validate_radio(c: &EnbConfig) -> Result<(), ConfigError> {
    let r = &c.radio;
    let cell_bits = 28u32.saturating_sub(c.enb_id_type.bits());
    if u64::from(r.cell_id) >= (1u64 << cell_bits) {
        return Err(ConfigError::Invalid(format!(
            "radio.cell_id {} does not fit the {cell_bits} cell-identity bits a {:?} eNB ID leaves",
            r.cell_id, c.enb_id_type
        )));
    }
    if !(1..=64).contains(&r.freq_band) {
        return Err(ConfigError::Invalid(format!(
            "radio.freq_band {} is outside 1..=64",
            r.freq_band
        )));
    }
    if r.q_rx_lev_min_dbm % 2 != 0 || !(-140..=-44).contains(&r.q_rx_lev_min_dbm) {
        return Err(ConfigError::Invalid(format!(
            "radio.q_rx_lev_min_dbm {} must be even and within -140..=-44",
            r.q_rx_lev_min_dbm
        )));
    }
    if ![100, 200, 300, 400, 600, 1000, 1500, 2000].contains(&r.t300_ms) {
        return Err(ConfigError::Invalid(format!(
            "radio.t300_ms {} is not a TS 36.331 T300 value",
            r.t300_ms
        )));
    }
    if r.si_broadcast_period_ms == 0 {
        return Err(ConfigError::Invalid(
            "radio.si_broadcast_period_ms must be > 0".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    pub(crate) fn valid() -> EnbConfig {
        EnbConfig {
            enb_id: 0x19B,
            enb_id_type: EnbIdType::Macro,
            plmn: Plmn::new(999, 70, false),
            tac: 1,
            supported_tas: vec![],
            enb_name: Some("nextgsim-enb".into()),
            paging_drx: PagingDrx::V128,
            mme_configs: vec![MmeConfig {
                address: IpAddr::V4(Ipv4Addr::LOCALHOST),
                port: 36412,
            }],
            s1ap_ip: IpAddr::V4(Ipv4Addr::LOCALHOST),
            s1_setup_retry_secs: 10,
            s1_setup_response_timeout_ms: 5000,
            radio: RadioConfig::default(),
            security: AsSecurityConfig::default(),
            gtp_ip: None,
        }
    }

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: std::collections::HashMap<String, String> =
            pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        move |k| map.get(k).cloned()
    }

    #[test]
    fn env_overrides_set_the_deployment_fields() {
        let mut c = valid();
        apply_env_overrides(
            &mut c,
            env(&[
                ("NEXTGSIM_ENB_MME_ADDR", "172.24.0.5"),
                ("NEXTGSIM_ENB_IP", "172.24.0.50"),
                ("NEXTGSIM_ENB_GTP_IP", "172.24.0.51"),
                ("NEXTGSIM_ENB_ID", "0x1A0"),
                ("NEXTGSIM_PLMN", "001-01"),
                ("NEXTGSIM_TAC", "7"),
            ]),
        )
        .unwrap();
        validate_enb_config(&c).unwrap();
        let ip = |s: &str| s.parse::<IpAddr>().unwrap();
        assert_eq!(c.mme_configs, vec![MmeConfig { address: ip("172.24.0.5"), port: 36412 }]);
        assert_eq!(c.s1ap_ip, ip("172.24.0.50"));
        assert_eq!(c.radio.link_ip, ip("172.24.0.50"));
        assert_eq!(c.gtp_ip, Some(ip("172.24.0.51")));
        assert_eq!(c.enb_id, 0x1A0);
        assert_eq!(c.plmn, Plmn::new(1, 1, false));
        assert_eq!(c.tac, 7);
    }

    #[test]
    fn no_env_leaves_the_file_untouched() {
        let mut c = valid();
        let before = format!("{c:?}");
        apply_env_overrides(&mut c, |_| None).unwrap();
        assert_eq!(format!("{c:?}"), before);
    }

    #[test]
    fn malformed_env_is_refused_naming_the_variable() {
        for (var, val) in [
            ("NEXTGSIM_ENB_MME_ADDR", "mme.local"),
            ("NEXTGSIM_ENB_IP", "x"),
            ("NEXTGSIM_ENB_ID", "0xZZ"),
            ("NEXTGSIM_PLMN", "99970"),
            ("NEXTGSIM_TAC", "70000"),
        ] {
            let err = apply_env_overrides(&mut valid(), env(&[(var, val)])).unwrap_err();
            assert!(err.to_string().contains(var), "{var}: {err}");
        }
    }

    #[test]
    fn plmn_override_conflicts_with_explicit_supported_tas() {
        let mut c = valid();
        c.supported_tas = vec![SupportedTa { tac: 1, broadcast_plmns: vec![Plmn::new(999, 70, false)] }];
        assert!(apply_env_overrides(&mut c, env(&[("NEXTGSIM_PLMN", "001-01")])).is_err());
    }

    #[test]
    fn security_lists_are_validated() {
        let mut c = valid();
        c.security.integrity = vec![2, 0];
        assert!(validate_enb_config(&c).is_err(), "EIA0");
        let mut c = valid();
        c.security.ciphering = vec![4];
        assert!(validate_enb_config(&c).is_err());
        let mut c = valid();
        c.security.integrity.clear();
        assert!(validate_enb_config(&c).is_err());
    }

    #[test]
    fn cell_identity_puts_the_enb_id_leftmost() {
        let mut c = valid();
        assert_eq!(c.cell_identity(), (0x19B << 8) | 1);
        c.enb_id_type = EnbIdType::Home;
        c.radio.cell_id = 0;
        assert_eq!(c.cell_identity(), 0x19B);
        validate_enb_config(&c).unwrap();
        c.radio.cell_id = 1;
        assert!(
            validate_enb_config(&c).is_err(),
            "a home eNB has no cell bits"
        );
    }

    #[test]
    fn radio_values_are_validated_against_ts_36_331() {
        let mut c = valid();
        c.radio.cell_id = 256;
        assert!(validate_enb_config(&c).is_err());
        let mut c = valid();
        c.radio.q_rx_lev_min_dbm = -139;
        assert!(validate_enb_config(&c).is_err());
        let mut c = valid();
        c.radio.t300_ms = 250;
        assert!(validate_enb_config(&c).is_err());
        let mut c = valid();
        c.radio.freq_band = 0;
        assert!(validate_enb_config(&c).is_err());
    }

    #[test]
    fn valid_config_passes() {
        validate_enb_config(&valid()).unwrap();
    }

    #[test]
    fn enb_id_must_fit_its_alternative() {
        let mut c = valid();
        c.enb_id = 1 << 20;
        assert!(validate_enb_config(&c).is_err());
        c.enb_id_type = EnbIdType::Home;
        // A home eNB ID is the whole 28-bit cell identity, so no cell bits.
        c.radio.cell_id = 0;
        validate_enb_config(&c).unwrap();
        c.enb_id = (1 << 28) - 1;
        validate_enb_config(&c).unwrap();
        c.enb_id = 1 << 28;
        assert!(validate_enb_config(&c).is_err());
    }

    #[test]
    fn default_supported_ta_is_tac_with_plmn() {
        let tas = valid().effective_supported_tas();
        assert_eq!(
            tas,
            vec![SupportedTa {
                tac: 1,
                broadcast_plmns: vec![Plmn::new(999, 70, false)]
            }]
        );
    }

    #[test]
    fn broadcast_plmn_count_is_bounded() {
        let mut c = valid();
        c.supported_tas = vec![SupportedTa {
            tac: 7,
            broadcast_plmns: vec![],
        }];
        assert!(validate_enb_config(&c).is_err());
        c.supported_tas[0].broadcast_plmns = vec![Plmn::new(1, 1, false); 7];
        assert!(validate_enb_config(&c).is_err());
        c.supported_tas[0].broadcast_plmns.truncate(6);
        validate_enb_config(&c).unwrap();
    }

    #[test]
    fn enb_name_must_be_printable_string() {
        let mut c = valid();
        c.enb_name = Some("enb_1".into()); // '_' is not in the PrintableString alphabet
        assert!(validate_enb_config(&c).is_err());
        c.enb_name = Some(String::new());
        assert!(validate_enb_config(&c).is_err());
        c.enb_name = Some("x".repeat(151));
        assert!(validate_enb_config(&c).is_err());
        c.enb_name = Some("x".repeat(150));
        validate_enb_config(&c).unwrap();
    }

    #[test]
    fn two_digit_mnc_above_99_is_rejected() {
        let mut c = valid();
        c.plmn = Plmn::new(999, 100, false);
        assert!(validate_enb_config(&c).is_err());
        c.plmn.long_mnc = true;
        validate_enb_config(&c).unwrap();
    }

    #[test]
    fn yaml_defaults_apply() {
        let y = r#"
enb_id: 0x19B
plmn: { mcc: 999, mnc: 70, long_mnc: false }
tac: 1
mme_configs:
  - address: 127.0.0.1
s1ap_ip: 127.0.0.1
"#;
        let c: EnbConfig = serde_yaml::from_str(y).unwrap();
        assert_eq!(c.enb_id_type, EnbIdType::Macro);
        assert_eq!(c.paging_drx, PagingDrx::V128);
        assert_eq!(c.mme_configs[0].port, 36412);
        assert_eq!(c.s1_setup_retry_secs, 10);
        assert_eq!(c.s1_setup_response_timeout_ms, 5000);
        validate_enb_config(&c).unwrap();
    }
}
