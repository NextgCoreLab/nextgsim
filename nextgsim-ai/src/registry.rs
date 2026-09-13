//! Shared, reusable ONNX model registry
//!
//! # Why this exists
//!
//! Every 6G consumer of [`OnnxEngine`] loads its model ad hoc: NWDAF's
//! `OnnxPredictor` owns an engine, `Mtlf` keeps a parallel model-management layer
//! of its own, and `nextgsim-semantic`'s `NeuralEncoder` and `NeuralDecoder` own
//! one engine **each** — so an encoder and a decoder pointing at the same `.onnx`
//! file load it twice and hold two `ort::Session`s.
//!
//! This registry resolves a model **by id** and hands out an
//! `Arc<OnnxEngine>`, so a second consumer of the same id gets the same warmed
//! session rather than a second copy. `OnnxEngine::infer` takes `&self` (the
//! session is behind an interior `Mutex`), which is what makes sharing through an
//! `Arc` work at all.
//!
//! # No model ships with this repo
//!
//! There is no `.onnx` in the tree, so the registry must degrade rather than
//! fail: an id with no registered path, or a registered path whose file is
//! absent, yields [`RegistryError::NotProvisioned`] — a **typed** error, so a
//! caller can distinguish "no model here, use your non-neural fallback" from "the
//! model is there and broken". Every consumer keeps its existing fallback; none of
//! them is rewritten.
//!
//! # Not to be confused with
//!
//! `nextgsim-fl`'s `ModelStore` versions federated-learning **weight vectors**
//! (`AggregatedModel` blobs, with pruning, tags, export/import). That is a
//! different concept from an ONNX inference session, and the name overlap is
//! coincidental. This registry does not touch it.
//!
//! # Non-normative
//!
//! 6G has no frozen Rel-20 Stage-3 spec. The framing is informed by TS 23.288
//! (NWDAF analytics/ML model provisioning) and TR 23.700-80, and implements
//! neither.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::config::ExecutionProvider;
use crate::error::ModelError;
use crate::inference::{InferenceEngine, OnnxEngine};

/// Why a model could not be resolved.
#[derive(Debug, Error)]
pub enum RegistryError {
    /// No model is available for this id — either nothing was registered under
    /// it, or the registered file is not on disk.
    ///
    /// The **expected** outcome in this repo, which ships no `.onnx`. A caller
    /// should fall back to its non-neural path rather than treating it as a
    /// failure.
    #[error("model {id:?}{version} is not provisioned: {reason}")]
    NotProvisioned {
        /// The model id that was asked for
        id: String,
        /// `" version X"` when a specific version was asked for, else empty
        version: String,
        /// What was missing
        reason: String,
    },
    /// The model file exists but the ONNX runtime rejected it.
    #[error("loading model {id:?} from {path} failed: {source}")]
    LoadFailed {
        /// The model id
        id: String,
        /// Where it was loaded from
        path: PathBuf,
        /// The underlying runtime error
        #[source]
        source: ModelError,
    },
    /// An engine could not be constructed for the configured execution provider.
    #[error("creating an inference engine failed: {0}")]
    EngineUnavailable(#[source] ModelError),
    /// The registry's internal lock was poisoned by a panic in another thread.
    #[error("the model registry lock is poisoned")]
    LockPoisoned,
}

impl RegistryError {
    fn not_provisioned(id: &str, version: Option<&str>, reason: impl Into<String>) -> Self {
        Self::NotProvisioned {
            id: id.to_string(),
            version: version.map(|v| format!(" version {v}")).unwrap_or_default(),
            reason: reason.into(),
        }
    }

    /// Whether this means "no model here" rather than "the model is broken".
    ///
    /// The distinction a caller acts on: a not-provisioned model is a cue to use
    /// the non-neural fallback, and anything else is worth reporting.
    pub fn is_not_provisioned(&self) -> bool {
        matches!(self, RegistryError::NotProvisioned { .. })
    }
}

/// One entry of a registry manifest: a model id, its version, and where the file
/// is.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegistryEntry {
    /// Logical model id, e.g. `"nwdaf-trajectory"`. Consumers resolve by this.
    pub id: String,
    /// Version string. Compared numerically per dot-separated component where
    /// every component is numeric, and lexicographically otherwise — see
    /// [`version_is_newer`].
    pub version: String,
    /// Path to the `.onnx` file.
    pub path: PathBuf,
}

/// A registry built from configuration.
///
/// Deserialisable so a deployment can provision models without code changes,
/// which is the point of resolving by id rather than by path.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegistryManifest {
    /// The models this deployment provides.
    #[serde(default)]
    pub entries: Vec<RegistryEntry>,
}

/// Whether `candidate` is a newer version than `current`.
///
/// Semver-lite and deliberately dependency-free: dot-separated components are
/// compared as integers when both are numeric, and as strings otherwise, with a
/// missing component treated as absent rather than as zero — so `1.2` and `1.2.0`
/// order by length, and `1.10` is newer than `1.9` (which a plain string compare
/// gets wrong).
pub fn version_is_newer(candidate: &str, current: &str) -> bool {
    let mut candidate_parts = candidate.split('.');
    let mut current_parts = current.split('.');
    loop {
        match (candidate_parts.next(), current_parts.next()) {
            (None, None) => return false, // equal
            (Some(_), None) => return true,
            (None, Some(_)) => return false,
            (Some(a), Some(b)) => {
                let ordering = match (a.parse::<u64>(), b.parse::<u64>()) {
                    (Ok(a), Ok(b)) => a.cmp(&b),
                    _ => a.cmp(b),
                };
                match ordering {
                    std::cmp::Ordering::Equal => continue,
                    std::cmp::Ordering::Greater => return true,
                    std::cmp::Ordering::Less => return false,
                }
            }
        }
    }
}

/// One registered (id, version) pair and the engine loaded for it, if any.
struct Registered {
    path: PathBuf,
    /// `None` until first resolved. Loading is lazy so registering a model that
    /// is never used costs nothing, and so a missing file is reported at the
    /// point a caller asks for it rather than at start-up.
    engine: Option<Arc<OnnxEngine>>,
}

/// A registry handing out shared, ref-counted inference engines by model id.
///
/// Cloning shares the same registry, so two consumers built independently resolve
/// to the same engine.
#[derive(Clone)]
pub struct SharedModelRegistry {
    inner: Arc<Mutex<RegistryInner>>,
    execution_provider: ExecutionProvider,
}

struct RegistryInner {
    /// `id -> version -> entry`. A `HashMap` of versions rather than one entry per
    /// id, because `get_versioned` has to reach a version that is not the latest.
    models: HashMap<String, HashMap<String, Registered>>,
}

impl std::fmt::Debug for SharedModelRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let loaded = self
            .inner
            .lock()
            .map(|inner| {
                inner
                    .models
                    .values()
                    .flat_map(|versions| versions.values())
                    .filter(|entry| entry.engine.is_some())
                    .count()
            })
            .unwrap_or(0);
        f.debug_struct("SharedModelRegistry")
            .field("execution_provider", &self.execution_provider)
            .field("loaded", &loaded)
            .finish_non_exhaustive()
    }
}

impl SharedModelRegistry {
    /// An empty registry whose engines will use `execution_provider`.
    pub fn new(execution_provider: ExecutionProvider) -> Self {
        Self {
            inner: Arc::new(Mutex::new(RegistryInner {
                models: HashMap::new(),
            })),
            execution_provider,
        }
    }

    /// A registry populated from a manifest. Nothing is loaded yet.
    pub fn from_manifest(
        manifest: &RegistryManifest,
        execution_provider: ExecutionProvider,
    ) -> Result<Self, RegistryError> {
        let registry = Self::new(execution_provider);
        for entry in &manifest.entries {
            registry.register(&entry.id, &entry.version, &entry.path)?;
        }
        Ok(registry)
    }

    /// Register a model file under `id` at `version`.
    ///
    /// Re-registering the same `(id, version)` replaces the path **and drops any
    /// engine already loaded for it**, so a consumer that resolves afterwards gets
    /// the new file rather than the old session. A consumer still holding an `Arc`
    /// keeps the engine it has, since dropping a session out from under a running
    /// inference is not something a registry gets to do.
    pub fn register(
        &self,
        id: &str,
        version: &str,
        path: impl Into<PathBuf>,
    ) -> Result<(), RegistryError> {
        let mut inner = self.inner.lock().map_err(|_| RegistryError::LockPoisoned)?;
        inner.models.entry(id.to_string()).or_default().insert(
            version.to_string(),
            Registered {
                path: path.into(),
                engine: None,
            },
        );
        Ok(())
    }

    /// Register an engine that is **already** constructed, under `id` at
    /// `version`.
    ///
    /// For a caller that built an engine itself and wants the registry to share it
    /// onward — and the only way to exercise the sharing in this repo's tests,
    /// since no `.onnx` ships and nothing can be loaded from disk. It populates
    /// exactly the same cache slot the loading path fills, so
    /// [`Self::load_by_id`] behaves identically afterwards.
    ///
    /// `path` is recorded for reporting; it is not read.
    pub fn register_preloaded(
        &self,
        id: &str,
        version: &str,
        path: impl Into<PathBuf>,
        engine: Arc<OnnxEngine>,
    ) -> Result<(), RegistryError> {
        let mut inner = self.inner.lock().map_err(|_| RegistryError::LockPoisoned)?;
        inner.models.entry(id.to_string()).or_default().insert(
            version.to_string(),
            Registered {
                path: path.into(),
                engine: Some(engine),
            },
        );
        Ok(())
    }

    /// Resolve the **latest** registered version of `id`, loading it once.
    ///
    /// Repeated calls return the same `Arc`, which is the reuse the whole registry
    /// exists for: two consumers of one id share one `ort::Session`.
    pub fn load_by_id(&self, id: &str) -> Result<Arc<OnnxEngine>, RegistryError> {
        let version = self.latest_version(id)?;
        self.get_versioned(id, &version)
    }

    /// Resolve the latest registered version of `id`. Alias of [`Self::load_by_id`]
    /// for callers that read better as `get`.
    pub fn get(&self, id: &str) -> Result<Arc<OnnxEngine>, RegistryError> {
        self.load_by_id(id)
    }

    /// Resolve a specific `version` of `id`, loading it once.
    pub fn get_versioned(&self, id: &str, version: &str) -> Result<Arc<OnnxEngine>, RegistryError> {
        // Fast path: already loaded. Taken under the same lock as the load below,
        // so two threads racing on one id cannot each build a session.
        let mut inner = self.inner.lock().map_err(|_| RegistryError::LockPoisoned)?;
        let entry = inner
            .models
            .get_mut(id)
            .and_then(|versions| versions.get_mut(version))
            .ok_or_else(|| {
                RegistryError::not_provisioned(id, Some(version), "no path is registered for it")
            })?;

        if let Some(ref engine) = entry.engine {
            return Ok(Arc::clone(engine));
        }

        let path = entry.path.clone();
        if !path.exists() {
            return Err(RegistryError::not_provisioned(
                id,
                Some(version),
                format!("{} does not exist", path.display()),
            ));
        }

        let engine = Arc::new(Self::load_engine(
            id,
            &path,
            self.execution_provider.clone(),
        )?);
        entry.engine = Some(Arc::clone(&engine));
        tracing::info!(
            "Model registry: loaded {} version {} from {}",
            id,
            version,
            path.display()
        );
        Ok(engine)
    }

    /// The versions registered for `id`, newest first. Empty when the id is
    /// unknown, which is not an error: asking what is available is not asking for
    /// something.
    pub fn versions(&self, id: &str) -> Vec<String> {
        let Ok(inner) = self.inner.lock() else {
            return Vec::new();
        };
        let Some(versions) = inner.models.get(id) else {
            return Vec::new();
        };
        let mut names: Vec<String> = versions.keys().cloned().collect();
        names.sort_by(|a, b| {
            if version_is_newer(a, b) {
                std::cmp::Ordering::Less
            } else if version_is_newer(b, a) {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });
        names
    }

    /// Every registered model id.
    pub fn ids(&self) -> Vec<String> {
        self.inner
            .lock()
            .map(|inner| inner.models.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// How many engines are loaded (not merely registered).
    pub fn loaded_count(&self) -> usize {
        self.inner
            .lock()
            .map(|inner| {
                inner
                    .models
                    .values()
                    .flat_map(|versions| versions.values())
                    .filter(|entry| entry.engine.is_some())
                    .count()
            })
            .unwrap_or(0)
    }

    /// Warm up every **loaded** engine.
    ///
    /// Returns the number warmed. A no-op when nothing is loaded, which is the
    /// state in this repo unless a deployment provisions a model — so it must not
    /// be an error, and it must not force a load either: warming up a registry
    /// would otherwise load every registered model whether or not anyone wants it.
    pub fn warmup_all(&self) -> Result<usize, RegistryError> {
        let engines: Vec<Arc<OnnxEngine>> = {
            let inner = self.inner.lock().map_err(|_| RegistryError::LockPoisoned)?;
            inner
                .models
                .values()
                .flat_map(|versions| versions.values())
                .filter_map(|entry| entry.engine.as_ref().map(Arc::clone))
                .collect()
        };

        let mut warmed = 0;
        for engine in engines {
            match engine.warmup() {
                Ok(()) => warmed += 1,
                // A warmup failure is reported and skipped rather than aborting
                // the sweep: one unusable model must not stop the others warming.
                Err(e) => tracing::warn!("Model registry: warmup failed: {}", e),
            }
        }
        Ok(warmed)
    }

    fn latest_version(&self, id: &str) -> Result<String, RegistryError> {
        self.versions(id)
            .into_iter()
            .next()
            .ok_or_else(|| RegistryError::not_provisioned(id, None, "no version is registered"))
    }

    fn load_engine(
        id: &str,
        path: &Path,
        execution_provider: ExecutionProvider,
    ) -> Result<OnnxEngine, RegistryError> {
        let mut engine =
            OnnxEngine::new(execution_provider).map_err(RegistryError::EngineUnavailable)?;
        engine
            .load_model(path)
            .map_err(|source| RegistryError::LoadFailed {
                id: id.to_string(),
                path: path.to_path_buf(),
                source,
            })?;
        Ok(engine)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> SharedModelRegistry {
        SharedModelRegistry::new(ExecutionProvider::Cpu)
    }

    /// `OnnxEngine` has no `Debug`, so `expect_err` cannot be used on a
    /// `Result<Arc<OnnxEngine>, _>`. This is the same assertion by hand.
    fn expect_error(result: Result<Arc<OnnxEngine>, RegistryError>, why: &str) -> RegistryError {
        match result {
            Err(e) => e,
            Ok(_) => panic!("expected an error because {why}"),
        }
    }

    /// An id with nothing registered is **not provisioned**, typed so a caller can
    /// tell it apart from a broken model and fall back.
    #[test]
    fn an_unregistered_id_is_not_provisioned() {
        let err = expect_error(registry().load_by_id("absent"), "nothing is registered");
        assert!(err.is_not_provisioned(), "{err}");
        assert!(err.to_string().contains("absent"));
    }

    /// A registered id whose file is missing is also not provisioned, rather than
    /// a panic or an opaque runtime error. This repo ships no `.onnx`, so this is
    /// the normal case.
    #[test]
    fn a_registered_id_with_no_file_is_not_provisioned() {
        let reg = registry();
        reg.register("trajectory", "1.0.0", "/nonexistent/trajectory.onnx")
            .expect("register");

        let err = expect_error(reg.load_by_id("trajectory"), "the file is absent");
        assert!(err.is_not_provisioned(), "{err}");
        assert!(
            err.to_string().contains("does not exist"),
            "the error must say what was missing: {err}"
        );
        assert_eq!(reg.loaded_count(), 0, "nothing was loaded");
    }

    /// A specific version that was never registered is not provisioned, and the
    /// error names the version — otherwise a caller cannot tell a wrong version
    /// from a wrong id.
    #[test]
    fn an_unregistered_version_is_not_provisioned_and_says_so() {
        let reg = registry();
        reg.register("trajectory", "1.0.0", "/nonexistent/a.onnx")
            .expect("register");

        let err = expect_error(
            reg.get_versioned("trajectory", "2.0.0"),
            "version 2 was never registered",
        );
        assert!(err.is_not_provisioned(), "{err}");
        assert!(err.to_string().contains("2.0.0"), "{err}");

        // And it must not silently answer with a *different* version. Version
        // 1.0.0 is made resolvable so that a lookup ignoring the version argument
        // would succeed here rather than erroring -- without this, the assertions
        // above pass against exactly that bug, since an absent file yields
        // NotProvisioned either way.
        let reg = registry();
        reg.register_preloaded(
            "trajectory",
            "1.0.0",
            "/models/t.onnx",
            Arc::new(OnnxEngine::new(ExecutionProvider::Cpu).expect("engine")),
        )
        .expect("register");
        assert!(reg.get_versioned("trajectory", "1.0.0").is_ok());
        let err = expect_error(
            reg.get_versioned("trajectory", "2.0.0"),
            "version 2 was never registered, and 1.0.0 must not stand in for it",
        );
        assert!(err.is_not_provisioned(), "{err}");
    }

    /// Version resolution picks the newest, and `1.10` beats `1.9` — which a plain
    /// string comparison gets backwards.
    #[test]
    fn the_latest_version_is_resolved_numerically() {
        let reg = registry();
        for version in ["1.9.0", "1.10.0", "1.2.0"] {
            reg.register("m", version, format!("/nonexistent/{version}.onnx"))
                .expect("register");
        }

        assert_eq!(
            reg.versions("m"),
            vec!["1.10.0", "1.9.0", "1.2.0"],
            "newest first, compared numerically"
        );

        // load_by_id resolves the latest, so its error names that version.
        let err = expect_error(reg.load_by_id("m"), "there is no file");
        assert!(err.to_string().contains("1.10.0"), "{err}");
    }

    #[test]
    fn version_comparison_handles_the_awkward_cases() {
        assert!(
            version_is_newer("1.10", "1.9"),
            "numeric, not lexicographic"
        );
        assert!(!version_is_newer("1.9", "1.10"));
        assert!(!version_is_newer("1.0", "1.0"), "equal is not newer");
        assert!(version_is_newer("1.0.1", "1.0"), "more components is newer");
        assert!(!version_is_newer("1.0", "1.0.1"));
        assert!(version_is_newer("2", "1.99.99"));
        // Non-numeric components fall back to a string compare rather than
        // parsing to 0, which would make every tag equal.
        assert!(version_is_newer("1.0-beta", "1.0-alpha"));
        assert!(!version_is_newer("1.0-alpha", "1.0-beta"));
    }

    /// The registry is a shared handle: a clone resolves against the same
    /// registrations, which is what lets two independently-built consumers share
    /// one engine.
    #[test]
    fn a_cloned_registry_shares_its_registrations() {
        let reg = registry();
        let clone = reg.clone();
        clone
            .register("shared", "1.0.0", "/nonexistent/shared.onnx")
            .expect("register");

        assert_eq!(reg.ids(), vec!["shared".to_string()]);
        assert_eq!(reg.versions("shared"), vec!["1.0.0"]);
    }

    /// A manifest builds a registry without code changes — the point of resolving
    /// by id.
    #[test]
    fn a_manifest_populates_the_registry() {
        let manifest = RegistryManifest {
            entries: vec![
                RegistryEntry {
                    id: "trajectory".to_string(),
                    version: "1.0.0".to_string(),
                    path: PathBuf::from("/nonexistent/t.onnx"),
                },
                RegistryEntry {
                    id: "semantic".to_string(),
                    version: "2.1.0".to_string(),
                    path: PathBuf::from("/nonexistent/s.onnx"),
                },
            ],
        };

        let reg = SharedModelRegistry::from_manifest(&manifest, ExecutionProvider::Cpu)
            .expect("from_manifest");

        let mut ids = reg.ids();
        ids.sort();
        assert_eq!(ids, vec!["semantic".to_string(), "trajectory".to_string()]);
        assert_eq!(reg.versions("semantic"), vec!["2.1.0"]);
    }

    /// A manifest round-trips through serde, so it can come from a config file.
    #[test]
    fn a_manifest_round_trips_through_json() {
        let manifest = RegistryManifest {
            entries: vec![RegistryEntry {
                id: "m".to_string(),
                version: "1.0.0".to_string(),
                path: PathBuf::from("/models/m.onnx"),
            }],
        };

        let json = serde_json::to_string(&manifest).expect("serialise");
        let back: RegistryManifest = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(back, manifest);
    }

    /// An empty manifest is a registry with nothing in it, not an error — a
    /// deployment that provisions no model is the default.
    #[test]
    fn an_empty_manifest_is_an_empty_registry() {
        let reg = SharedModelRegistry::from_manifest(
            &RegistryManifest::default(),
            ExecutionProvider::Cpu,
        )
        .expect("from_manifest");
        assert!(reg.ids().is_empty());
        assert_eq!(reg.loaded_count(), 0);
    }

    /// `warmup_all` is a clean no-op with nothing loaded: it returns `Ok(0)` and
    /// leaves the registry alone.
    ///
    /// **What this test does not prove**, stated because the docstring on
    /// `warmup_all` claims it: that warming does not *force* a load. That property
    /// is unverifiable in this repo — forcing a load of an unprovisioned model is
    /// itself a no-op, since there is no `.onnx` to load, so an implementation that
    /// tried would leave `loaded_count` at 0 exactly as the correct one does. It
    /// becomes observable only once a model file exists.
    #[test]
    fn warmup_all_no_ops_cleanly_with_nothing_loaded() {
        let reg = registry();
        reg.register("m", "1.0.0", "/nonexistent/m.onnx")
            .expect("register");

        assert_eq!(reg.warmup_all().expect("warmup_all"), 0);
        assert_eq!(
            reg.loaded_count(),
            0,
            "warmup must not load what nobody asked for"
        );
    }

    /// Re-registering replaces the path. Asserted through the error message,
    /// because with no `.onnx` in the tree the path is only observable there.
    #[test]
    fn re_registering_a_version_replaces_its_path() {
        let reg = registry();
        reg.register("m", "1.0.0", "/nonexistent/first.onnx")
            .expect("register");
        assert!(expect_error(reg.load_by_id("m"), "the file is absent")
            .to_string()
            .contains("first.onnx"));

        reg.register("m", "1.0.0", "/nonexistent/second.onnx")
            .expect("re-register");
        let err = expect_error(reg.load_by_id("m"), "the file is absent");
        assert!(err.to_string().contains("second.onnx"), "{err}");
        assert!(!err.to_string().contains("first.onnx"));
    }

    /// **The reuse property**, which is the whole reason the registry exists:
    /// repeated resolution of one id returns the *same* `Arc`, so two consumers
    /// share one `ort::Session` rather than loading the file twice.
    ///
    /// Exercised through `register_preloaded` because no `.onnx` ships with this
    /// repo, so nothing can be loaded from disk. It populates the same cache slot
    /// the loading path fills and `load_by_id` reads it the same way, so what is
    /// proven is the sharing; the disk load itself is unexercisable here.
    #[test]
    fn resolving_one_id_twice_returns_the_same_engine() {
        let reg = registry();
        let engine = Arc::new(OnnxEngine::new(ExecutionProvider::Cpu).expect("engine"));
        reg.register_preloaded(
            "shared",
            "1.0.0",
            "/models/shared.onnx",
            Arc::clone(&engine),
        )
        .expect("register");

        let first = reg.load_by_id("shared").expect("resolves");
        let second = reg.load_by_id("shared").expect("resolves again");

        assert!(
            Arc::ptr_eq(&first, &second),
            "both consumers must hold the same engine"
        );
        assert!(Arc::ptr_eq(&first, &engine));
        assert_eq!(reg.loaded_count(), 1, "one engine, not two");
    }

    /// Two *logical consumers* built independently, resolving the same id from
    /// clones of the registry, get one engine — the observable form of the reuse.
    #[test]
    fn two_consumers_of_one_id_share_a_single_engine() {
        let reg = registry();
        reg.register_preloaded(
            "trajectory",
            "1.0.0",
            "/models/trajectory.onnx",
            Arc::new(OnnxEngine::new(ExecutionProvider::Cpu).expect("engine")),
        )
        .expect("register");

        // Each "consumer" holds its own clone of the registry handle, as an
        // independently-constructed NWDAF predictor and semantic codec would.
        let consumer_a = reg.clone();
        let consumer_b = reg.clone();
        let engine_a = consumer_a.load_by_id("trajectory").expect("a resolves");
        let engine_b = consumer_b.load_by_id("trajectory").expect("b resolves");

        assert!(Arc::ptr_eq(&engine_a, &engine_b));
        assert_eq!(reg.loaded_count(), 1);
    }

    /// Different versions of one id are different engines: sharing is per
    /// (id, version), not per id.
    #[test]
    fn two_versions_of_one_id_are_separate_engines() {
        let reg = registry();
        for version in ["1.0.0", "2.0.0"] {
            reg.register_preloaded(
                "m",
                version,
                format!("/models/m-{version}.onnx"),
                Arc::new(OnnxEngine::new(ExecutionProvider::Cpu).expect("engine")),
            )
            .expect("register");
        }

        let v1 = reg.get_versioned("m", "1.0.0").expect("v1");
        let v2 = reg.get_versioned("m", "2.0.0").expect("v2");
        assert!(!Arc::ptr_eq(&v1, &v2));

        // And `load_by_id` picks the newest of the two.
        let latest = reg.load_by_id("m").expect("latest");
        assert!(Arc::ptr_eq(&latest, &v2), "2.0.0 is the latest");
        assert_eq!(reg.loaded_count(), 2);
    }

    /// `warmup_all` warms what is loaded. With no session behind the engine the
    /// warmup itself fails, which is reported and skipped rather than aborting the
    /// sweep — so one unusable model does not stop the others.
    #[test]
    fn warmup_all_skips_an_engine_it_cannot_warm() {
        let reg = registry();
        reg.register_preloaded(
            "m",
            "1.0.0",
            "/models/m.onnx",
            Arc::new(OnnxEngine::new(ExecutionProvider::Cpu).expect("engine")),
        )
        .expect("register");

        // No session was loaded, so warming it cannot succeed; the call still
        // returns Ok, reporting how many were warmed.
        assert_eq!(
            reg.warmup_all().expect("warmup_all must not abort"),
            0,
            "the engine has no session to warm"
        );
    }

    /// Re-registering a version replaces the engine too, so a consumer resolving
    /// afterwards gets the new one rather than the cached old session.
    #[test]
    fn re_registering_a_version_replaces_its_engine() {
        let reg = registry();
        let first = Arc::new(OnnxEngine::new(ExecutionProvider::Cpu).expect("engine"));
        reg.register_preloaded("m", "1.0.0", "/models/m.onnx", Arc::clone(&first))
            .expect("register");
        assert!(Arc::ptr_eq(&reg.load_by_id("m").expect("resolves"), &first));

        let second = Arc::new(OnnxEngine::new(ExecutionProvider::Cpu).expect("engine"));
        reg.register_preloaded("m", "1.0.0", "/models/m.onnx", Arc::clone(&second))
            .expect("re-register");

        let resolved = reg.load_by_id("m").expect("resolves");
        assert!(Arc::ptr_eq(&resolved, &second));
        assert!(!Arc::ptr_eq(&resolved, &first));
    }

    /// Asking what versions exist for an unknown id is not asking for one, so it
    /// answers empty rather than erroring.
    #[test]
    fn querying_an_unknown_id_is_not_an_error() {
        let reg = registry();
        assert!(reg.versions("nope").is_empty());
        assert!(reg.ids().is_empty());
    }
}
