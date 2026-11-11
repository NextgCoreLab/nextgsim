//! The demo toggle convention (nextgsim#458, nextgsim#461), shared so that no
//! demo hand-rolls it.
//!
//! # Schema
//!
//! One block at the root of a gNB or UE YAML config, identical to nextgcore's
//! `nextgcore-demo` crate (nextgcore#1394), so the two repos read one shape
//! without nextgsim depending on nextgcore:
//!
//! ```yaml
//! demo:
//!   enabled: false        # process-wide kill switch; false overrides every demo below
//!   <key>:                # a demo key this binary registered, e.g. d441a-congestion-loop
//!     enabled: false
//!     mode: <mode>        # one of the modes the binary registered for <key>
//! ```
//!
//! - The block is optional. Absent, `demo: ~` and `demo: {}` all mean "no demo".
//! - `demo.enabled` defaults to `false`; so does each `<key>.enabled`.
//! - `<key>.mode` is **required** when `<key>.enabled` is true: there is no
//!   default mode, because a silent fallback is what the convention forbids.
//! - A key or mode the binary did not register, or a misspelt field, fails
//!   startup with a [`DemoConfigError`]. This holds even when the kill switch is
//!   off, so a typo cannot hide behind `enabled: false`.
//! - Keys and modes are lowercase letters, digits and `-`.
//!
//! # Gates
//!
//! A demo is active only when its binary was built with that demo's cargo
//! feature (the binary registers the [`DemoSpec`] under it), **and** both
//! `demo.enabled` and `demo.<key>.enabled` are true. With no demo active the
//! process is unchanged: no banner, no metric series, no demo log lines.
//!
//! # Visibility
//!
//! - [`install`] logs `WARN DEMO MODE ACTIVE: <key>=<mode>,...` once at startup,
//!   and again after every accepted runtime mode change.
//! - [`DemoRuntime::render_metrics`] writes `nextg_demo_enabled{demo,mode} 1` per
//!   active demo, and nothing at all (not even the header) when none is active.
//!   The gNB and UE have no `/metrics` listener, so the family is read with
//!   `nr-cli <node> demo metrics`; this crate opens no listener.
//! - [`demo_log!`] prefixes a demo's log lines with `[demo:<key>]`.
//!
//! # Switching
//!
//! Unlike nextgcore NFs (config plus restart only), a nextgsim gNB or UE switches
//! an **active** demo's mode at runtime, through `nr-cli`:
//!
//! - `nr-cli <node> demo <key> set mode=<mode>`;
//! - `nr-cli <node> demo <key> show`, `demo show`, `demo metrics`.
//!
//! A demo reads its mode with [`active_mode`] at each decision, so a change
//! applies to the next decision. Every `set`, accepted or not, is audited with a
//! `WARN [demo:<key>] audit:` line. A rejected `set` (unknown key or mode, or a
//! demo the config did not enable) leaves the mode unchanged. Enabling or
//! disabling a demo is config plus restart: `set` changes modes only.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::{OnceLock, RwLock};

use serde_yaml::Value;

#[doc(hidden)]
pub use tracing;

/// The name of the config block at the root of a gNB or UE config.
pub const DEMO_BLOCK: &str = "demo";

/// The metric family exported while a demo is active. It has no `nextgsim_`
/// prefix, so nextgsim and nextgcore export one series name.
pub const METRIC_NAME: &str = "nextg_demo_enabled";

/// One demo a binary knows about: its key, its modes, and the demos it needs
/// active alongside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DemoSpec {
    /// The demo's key, as written under `demo:` (e.g. `d441a-congestion-loop`).
    pub key: &'static str,
    /// The modes (arms, variants) the demo accepts.
    pub modes: &'static [&'static str],
    /// Keys of demos that must also be active when this one is.
    pub requires: &'static [&'static str],
}

/// The demos one binary registered. A binary registers a demo only under that
/// demo's cargo feature, so a default build has an empty registry and rejects
/// every `demo.<key>` block.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DemoRegistry {
    specs: Vec<DemoSpec>,
}

impl DemoRegistry {
    /// A registry of `specs`.
    ///
    /// # Panics
    /// On a malformed key or mode, a duplicate key, an empty mode list, or a
    /// `requires` naming an unregistered key. These are programming errors in the
    /// binary's own table, not configuration errors.
    pub fn new(specs: &[DemoSpec]) -> Self {
        let mut seen = std::collections::BTreeSet::new();
        for s in specs {
            assert!(is_name(s.key), "demo key {:?} is not [a-z0-9-]+", s.key);
            assert!(seen.insert(s.key), "demo key {:?} registered twice", s.key);
            assert!(!s.modes.is_empty(), "demo {:?} registers no mode", s.key);
            for m in s.modes {
                assert!(
                    is_name(m),
                    "demo {:?} mode {:?} is not [a-z0-9-]+",
                    s.key,
                    m
                );
            }
        }
        for s in specs {
            for r in s.requires {
                assert!(
                    specs.iter().any(|o| o.key == *r),
                    "demo {:?} requires unregistered demo {:?}",
                    s.key,
                    r
                );
            }
        }
        Self {
            specs: specs.to_vec(),
        }
    }

    /// The spec registered for `key`.
    pub fn get(&self, key: &str) -> Option<&DemoSpec> {
        self.specs.iter().find(|s| s.key == key)
    }

    /// Every registered key, in registration order.
    pub fn keys(&self) -> Vec<&'static str> {
        self.specs.iter().map(|s| s.key).collect()
    }

    /// No demo registered (a default build).
    pub fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }
}

fn is_name(s: &str) -> bool {
    !s.is_empty()
        && s.bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
}

/// Why a `demo:` block was refused. Every variant fails startup.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DemoConfigError {
    /// `demo.<key>` names a demo this binary did not register (not built with
    /// its feature, or misspelt).
    #[error("demo config: unknown demo key {key:?} (this binary knows {known:?})")]
    UnknownKey {
        /// The key as written.
        key: String,
        /// The keys this binary registered.
        known: Vec<&'static str>,
    },
    /// `demo.<key>.mode` is not one of the modes registered for the key.
    #[error("demo config: unknown mode {mode:?} for demo {demo:?} (known modes {known:?})")]
    UnknownMode {
        /// The demo's key.
        demo: String,
        /// The mode as written.
        mode: String,
        /// The modes registered for the key.
        known: Vec<&'static str>,
    },
    /// An active demo needs another demo that is not active.
    #[error("demo config: demo {demo:?} requires demo {requires:?} to be enabled too")]
    MissingDependency {
        /// The active demo.
        demo: String,
        /// The demo it needs, which is not active.
        requires: String,
    },
    /// The block is not the schema's shape: a wrong type, a misspelt field, or
    /// an enabled demo without a mode.
    #[error("demo config: {detail}")]
    Malformed {
        /// What is wrong, naming the path.
        detail: String,
    },
    /// [`init_from_yaml`] ran a second time in this process: a programming
    /// error, not a config error (the state is write-once).
    #[error("demo config: demo state is already installed for this process")]
    AlreadyInstalled,
}

fn malformed(detail: impl Into<String>) -> DemoConfigError {
    DemoConfigError::Malformed {
        detail: detail.into(),
    }
}

/// One `demo.<key>` entry as written.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DemoEntry {
    /// `demo.<key>.enabled`.
    pub enabled: bool,
    /// `demo.<key>.mode`.
    pub mode: Option<String>,
}

/// The `demo:` block as written, before it is checked against a registry.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DemoConfig {
    /// The process-wide kill switch.
    pub enabled: bool,
    /// The per-demo entries, by key.
    pub demos: BTreeMap<String, DemoEntry>,
}

fn as_bool(v: &Value, path: &str) -> Result<bool, DemoConfigError> {
    match v {
        Value::Bool(b) => Ok(*b),
        Value::Null => Ok(false),
        other => Err(malformed(format!(
            "{path} must be a boolean, got {other:?}"
        ))),
    }
}

impl DemoConfig {
    /// Read the `demo:` block from the root of a YAML config. A root with no
    /// `demo` key (or a non-mapping root) yields the empty, disabled config.
    pub fn from_root(root: &Value) -> Result<Self, DemoConfigError> {
        match root.get(DEMO_BLOCK) {
            None => Ok(Self::default()),
            Some(block) => Self::from_block(block),
        }
    }

    /// Read a `demo:` block's value.
    pub fn from_block(block: &Value) -> Result<Self, DemoConfigError> {
        let map = match block {
            Value::Null => return Ok(Self::default()),
            Value::Mapping(m) => m,
            other => return Err(malformed(format!("demo must be a mapping, got {other:?}"))),
        };
        let mut cfg = Self::default();
        for (k, v) in map {
            let Some(k) = k.as_str() else {
                return Err(malformed(format!("demo has a non-string key {k:?}")));
            };
            if k == "enabled" {
                cfg.enabled = as_bool(v, "demo.enabled")?;
                continue;
            }
            cfg.demos
                .insert(k.to_string(), DemoEntry::from_value(k, v)?);
        }
        Ok(cfg)
    }

    /// Check the block against the binary's registry and return what is active.
    ///
    /// Keys, modes and field names are checked whether or not the kill switch is
    /// on. The kill switch then decides: off, nothing is active. On, every entry
    /// with `enabled: true` is, and each one's `requires` must be active too.
    pub fn validate(&self, registry: &DemoRegistry) -> Result<ActiveDemos, DemoConfigError> {
        let mut active = Vec::new();
        for (key, entry) in &self.demos {
            let Some(spec) = registry.get(key) else {
                return Err(DemoConfigError::UnknownKey {
                    key: key.clone(),
                    known: registry.keys(),
                });
            };
            if let Some(mode) = &entry.mode {
                if !spec.modes.contains(&mode.as_str()) {
                    return Err(DemoConfigError::UnknownMode {
                        demo: key.clone(),
                        mode: mode.clone(),
                        known: spec.modes.to_vec(),
                    });
                }
            }
            if entry.enabled {
                let Some(mode) = &entry.mode else {
                    return Err(malformed(format!(
                        "demo.{key}.mode is required when demo.{key}.enabled is true \
                         (known modes {:?})",
                        spec.modes
                    )));
                };
                if self.enabled {
                    active.push((spec.key, mode.clone()));
                }
            }
        }
        for (key, _) in &active {
            let spec = registry.get(key).expect("active keys are registered");
            for req in spec.requires {
                if !active.iter().any(|(k, _)| k == req) {
                    return Err(DemoConfigError::MissingDependency {
                        demo: key.to_string(),
                        requires: req.to_string(),
                    });
                }
            }
        }
        Ok(ActiveDemos { entries: active })
    }
}

impl DemoEntry {
    fn from_value(key: &str, v: &Value) -> Result<Self, DemoConfigError> {
        let map = match v {
            Value::Null => return Ok(Self::default()),
            Value::Mapping(m) => m,
            other => {
                return Err(malformed(format!(
                    "demo.{key} must be a mapping, got {other:?}"
                )))
            }
        };
        let mut e = Self::default();
        for (f, fv) in map {
            match f.as_str() {
                Some("enabled") => e.enabled = as_bool(fv, &format!("demo.{key}.enabled"))?,
                Some("mode") => match fv {
                    Value::String(s) => e.mode = Some(s.clone()),
                    Value::Null => e.mode = None,
                    other => {
                        return Err(malformed(format!(
                            "demo.{key}.mode must be a string, got {other:?}"
                        )))
                    }
                },
                _ => {
                    let name = f
                        .as_str()
                        .map_or_else(|| format!("{f:?}"), |n| format!("{n:?}"));
                    return Err(malformed(format!(
                        "demo.{key} has unknown field {name} (expected enabled, mode)"
                    )));
                }
            }
        }
        Ok(e)
    }
}

/// The demos active in this process, sorted by key. Empty unless both gates
/// and the kill switch are on.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ActiveDemos {
    entries: Vec<(&'static str, String)>,
}

impl ActiveDemos {
    /// No demo active.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The active `(key, mode)` pairs, sorted by key.
    pub fn entries(&self) -> impl Iterator<Item = (&'static str, &str)> {
        self.entries.iter().map(|(k, m)| (*k, m.as_str()))
    }

    /// The mode of `key`, if that demo is active.
    pub fn mode(&self, key: &str) -> Option<&str> {
        self.entries
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, m)| m.as_str())
    }

    /// Whether `key` is active.
    pub fn is_active(&self, key: &str) -> bool {
        self.mode(key).is_some()
    }

    /// `DEMO MODE ACTIVE: <key>=<mode>,...`, or `None` when no demo is active.
    pub fn banner(&self) -> Option<String> {
        if self.is_empty() {
            return None;
        }
        let list: Vec<String> = self.entries().map(|(k, m)| format!("{k}={m}")).collect();
        Some(format!("DEMO MODE ACTIVE: {}", list.join(",")))
    }

    /// Append the `nextg_demo_enabled` family: one `{demo,mode} 1` series per
    /// active demo, and nothing at all (not even the HELP/TYPE header) when none
    /// is active.
    pub fn render_metrics(&self, out: &mut String) {
        if self.is_empty() {
            return;
        }
        let _ = writeln!(
            out,
            "# HELP {METRIC_NAME} 1 for each demo active in this process (nextgsim#458)"
        );
        let _ = writeln!(out, "# TYPE {METRIC_NAME} gauge");
        // Keys and modes are [a-z0-9-]+ (checked at registration), so they need
        // no label escaping.
        for (k, m) in self.entries() {
            let _ = writeln!(out, "{METRIC_NAME}{{demo=\"{k}\",mode=\"{m}\"}} 1");
        }
    }

    fn set_mode(&mut self, key: &str, mode: &str) {
        if let Some((_, m)) = self.entries.iter_mut().find(|(k, _)| *k == key) {
            *m = mode.to_string();
        }
    }
}

/// Why an `nr-cli ... demo` command was refused. A refused `set` leaves every
/// mode unchanged.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DemoCliError {
    /// The key or mode is not registered: the same errors as the config.
    #[error("{0}")]
    Config(#[from] DemoConfigError),
    /// The demo is registered but the config did not enable it. Enabling a demo
    /// is config plus restart; `set` changes modes only.
    #[error(
        "demo {demo:?} is not enabled in this process (enable it in the config \
         and restart; `set` switches modes only)"
    )]
    NotActive {
        /// The demo's key.
        demo: String,
    },
    /// The command line is not one of the `demo` forms.
    #[error("{0}")]
    Usage(String),
    /// [`install`] has not run in this process.
    #[error("demo state is not installed in this process")]
    NotInstalled,
}

const USAGE: &str =
    "usage: demo show | demo metrics | demo <key> show | demo <key> set mode=<mode>";

/// A process's demo state: the registry it was built with, the kill switch as
/// configured, and the active demos with their current modes.
#[derive(Debug, Default)]
pub struct DemoRuntime {
    registry: DemoRegistry,
    kill_switch: bool,
    active: RwLock<ActiveDemos>,
}

impl DemoRuntime {
    /// State for a validated config. `kill_switch` is `demo.enabled` as written,
    /// reported by `show`.
    pub fn new(registry: DemoRegistry, kill_switch: bool, active: ActiveDemos) -> Self {
        Self {
            registry,
            kill_switch,
            active: RwLock::new(active),
        }
    }

    /// Parse and check a config root's `demo:` block into a runtime. An error
    /// must fail startup.
    pub fn from_root(root: &Value, registry: DemoRegistry) -> Result<Self, DemoConfigError> {
        let cfg = DemoConfig::from_root(root)?;
        let active = cfg.validate(&registry)?;
        Ok(Self::new(registry, cfg.enabled, active))
    }

    /// The registry this process was built with.
    pub fn registry(&self) -> &DemoRegistry {
        &self.registry
    }

    /// A copy of the active demos and their current modes.
    pub fn snapshot(&self) -> ActiveDemos {
        self.read().clone()
    }

    /// The current mode of `key`, if that demo is active. Read it at each
    /// decision, so a runtime switch applies to the next one.
    pub fn mode(&self, key: &str) -> Option<String> {
        self.read().mode(key).map(str::to_string)
    }

    fn read(&self) -> std::sync::RwLockReadGuard<'_, ActiveDemos> {
        self.active
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Append the `nextg_demo_enabled` family for the current modes; nothing
    /// when no demo is active.
    pub fn render_metrics(&self, out: &mut String) {
        self.read().render_metrics(out);
    }

    /// Switch the active demo `key` to `mode`, auditing the attempt whether or
    /// not it is accepted. `source` names who asked (e.g. `nr-cli`). Returns the
    /// previous mode. A refusal leaves every mode unchanged.
    pub fn set_mode(&self, key: &str, mode: &str, source: &str) -> Result<String, DemoCliError> {
        // Check and swap under one write lock, so the reported previous mode is
        // the one this call replaced.
        let outcome = self.check_set(key, mode).and_then(|spec_key| {
            let mut active = self
                .active
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let previous = active.mode(spec_key).map(str::to_string).ok_or_else(|| {
                DemoCliError::NotActive {
                    demo: key.to_string(),
                }
            })?;
            active.set_mode(spec_key, mode);
            Ok((spec_key, previous, active.banner()))
        });
        let (spec_key, previous, banner) = match outcome {
            Ok(v) => v,
            Err(e) => {
                // A key that is not registered is user input, so it is quoted
                // rather than used as a prefix.
                match self.registry.get(key) {
                    Some(spec) => tracing::warn!(
                        "[demo:{}] audit: rejected set mode={mode:?} from {source}: {e}; mode \
                         unchanged ({})",
                        spec.key,
                        self.mode(spec.key).as_deref().unwrap_or("not active")
                    ),
                    None => tracing::warn!(
                        "[demo] audit: rejected set on demo {key:?} mode={mode:?} from \
                         {source}: {e}; nothing changed"
                    ),
                }
                return Err(e);
            }
        };
        if previous == mode {
            tracing::warn!(
                "[demo:{spec_key}] audit: set mode={mode} from {source}: already {previous}, \
                 unchanged"
            );
        } else {
            tracing::warn!(
                "[demo:{spec_key}] audit: mode {previous} -> {mode} from {source}; applies \
                 to the next decision"
            );
            if let Some(b) = banner {
                tracing::warn!("{b}");
            }
        }
        Ok(previous)
    }

    fn check_set(&self, key: &str, mode: &str) -> Result<&'static str, DemoCliError> {
        let spec = self
            .registry
            .get(key)
            .ok_or_else(|| DemoConfigError::UnknownKey {
                key: key.to_string(),
                known: self.registry.keys(),
            })?;
        if !spec.modes.contains(&mode) {
            return Err(DemoConfigError::UnknownMode {
                demo: key.to_string(),
                mode: mode.to_string(),
                known: spec.modes.to_vec(),
            }
            .into());
        }
        Ok(spec.key)
    }

    /// One `show` line for `key`.
    pub fn show(&self, key: &str) -> Result<String, DemoCliError> {
        let spec = self
            .registry
            .get(key)
            .ok_or_else(|| DemoConfigError::UnknownKey {
                key: key.to_string(),
                known: self.registry.keys(),
            })?;
        Ok(self.show_line(spec))
    }

    fn show_line(&self, spec: &DemoSpec) -> String {
        let mode = self.mode(spec.key);
        format!(
            "demo={} active={} mode={} modes={} requires={}",
            spec.key,
            mode.is_some(),
            mode.as_deref().unwrap_or("-"),
            spec.modes.join(","),
            if spec.requires.is_empty() {
                "-".to_string()
            } else {
                spec.requires.join(",")
            }
        )
    }

    /// Every registered demo, one line each, after the kill switch.
    pub fn show_all(&self) -> String {
        if self.registry.is_empty() {
            return format!(
                "demo.enabled={}\nno demo is registered in this binary (each demo registers \
                 under its own cargo feature)",
                self.kill_switch
            );
        }
        let mut out = format!("demo.enabled={}", self.kill_switch);
        for spec in &self.registry.specs {
            out.push('\n');
            out.push_str(&self.show_line(spec));
        }
        out
    }

    /// Run the arguments of an `nr-cli <node> demo ...` command (everything
    /// after `demo`).
    pub fn handle_cli(&self, args: &str, source: &str) -> Result<String, DemoCliError> {
        let tokens: Vec<&str> = args.split_whitespace().collect();
        match tokens.as_slice() {
            ["show"] => Ok(self.show_all()),
            ["metrics"] => {
                let mut out = String::new();
                self.render_metrics(&mut out);
                Ok(out)
            }
            [key, "show"] => self.show(key),
            [key, "set", assignment] => {
                let Some(mode) = assignment.strip_prefix("mode=") else {
                    return Err(DemoCliError::Usage(USAGE.to_string()));
                };
                let previous = self.set_mode(key, mode, source)?;
                Ok(if previous == mode {
                    format!("demo={key} mode={mode} (unchanged)")
                } else {
                    format!("demo={key} mode={mode} (was {previous}; applies to the next decision)")
                })
            }
            _ => Err(DemoCliError::Usage(USAGE.to_string())),
        }
    }
}

/// The process's demo state, set once at startup.
static INSTALLED: OnceLock<DemoRuntime> = OnceLock::new();

/// [`install`] was called twice.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("demo state is already installed for this process")]
pub struct AlreadyInstalled;

/// Install the process's demo state and, if any demo is active, log the
/// `WARN DEMO MODE ACTIVE` banner. Call once, at startup.
pub fn install(runtime: DemoRuntime) -> Result<&'static DemoRuntime, AlreadyInstalled> {
    let mut fresh = false;
    let installed = INSTALLED.get_or_init(|| {
        fresh = true;
        runtime
    });
    if !fresh {
        return Err(AlreadyInstalled);
    }
    if let Some(banner) = installed.read().banner() {
        tracing::warn!("{banner}");
    }
    Ok(installed)
}

/// Parse, check and install the `demo:` block of a YAML config in one step:
/// what a gNB or UE startup calls. An error must fail startup.
pub fn init_from_yaml(
    yaml: &str,
    registry: DemoRegistry,
) -> Result<&'static DemoRuntime, DemoConfigError> {
    let root: Value = serde_yaml::from_str(yaml)
        .map_err(|e| malformed(format!("config is not valid YAML: {e}")))?;
    let runtime = DemoRuntime::from_root(&root, registry)?;
    install(runtime).map_err(|AlreadyInstalled| DemoConfigError::AlreadyInstalled)
}

/// The installed state, or `None` before [`install`].
pub fn installed() -> Option<&'static DemoRuntime> {
    INSTALLED.get()
}

/// The current mode of `key` in the installed state, if that demo is active.
/// Read it at each decision, so a runtime switch applies to the next one.
pub fn active_mode(key: &str) -> Option<String> {
    installed().and_then(|r| r.mode(key))
}

/// Append the installed state's `nextg_demo_enabled` family to a metrics body.
/// Writes nothing before [`install`] or when no demo is active.
pub fn render_installed_metrics(out: &mut String) {
    if let Some(r) = installed() {
        r.render_metrics(out);
    }
}

/// If `command` is an `nr-cli` `demo ...` command, run it against the installed
/// state and return its result; otherwise `None`, so the node parses it as
/// before. `source` is recorded in the audit line.
pub fn route_cli(command: &str, source: &str) -> Option<Result<String, DemoCliError>> {
    let trimmed = command.trim_start();
    let (word, rest) = trimmed
        .split_once(char::is_whitespace)
        .unwrap_or((trimmed, ""));
    if !word.eq_ignore_ascii_case("demo") {
        return None;
    }
    Some(match installed() {
        Some(r) => r.handle_cli(rest, source),
        None => Err(DemoCliError::NotInstalled),
    })
}

/// `[demo:<key>]`, the prefix of every demo log line.
pub fn prefix(key: &str) -> String {
    format!("[demo:{key}]")
}

/// Log a demo line at `level` with the `[demo:<key>]` prefix:
///
/// ```
/// nextgsim_demo::demo_log!(info, "d441a-congestion-loop", "lowered the cap to {}", 3);
/// // => INFO [demo:d441a-congestion-loop] lowered the cap to 3
/// ```
#[macro_export]
macro_rules! demo_log {
    ($level:ident, $key:expr, $($arg:tt)+) => {
        $crate::tracing::$level!("[demo:{}] {}", $key, format_args!($($arg)+))
    };
}

#[cfg(test)]
mod tests;
