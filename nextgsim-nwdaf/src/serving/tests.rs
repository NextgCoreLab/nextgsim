//! nextgsim#463 at the NWDAF manager: the logs, the counters and the status,
//! against the shipped d440b reference model (`tools/d440b-model/`).
//!
//! The `ort` module needs ONNX Runtime: `#[ignore]`d, run by the model-verify CI
//! job with `--include-ignored`, and failing (not skipping) without
//! `ORT_DYLIB_PATH`.

use std::io::Write as _;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use nextgsim_ai::ModelManifest;

use super::*;
use crate::{NwdafManager, UeMeasurement, Vector3};

pub(crate) fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tools/d440b-model")
}

pub(crate) fn shipped() -> (ModelManifest, Vec<u8>) {
    let dir = model_dir();
    let text = std::fs::read_to_string(dir.join("nwdaf-trajectory-cv.manifest.json")).unwrap();
    let bytes = std::fs::read(dir.join("nwdaf-trajectory-cv.onnx")).unwrap();
    (ModelManifest::from_json(&text).unwrap(), bytes)
}

/// The shipped model with the last byte of its float 2.0 weight flipped: valid
/// ONNX with the same signature, so only the sha256 refuses it.
pub(crate) fn flipped(bytes: &[u8]) -> Vec<u8> {
    let at = bytes
        .windows(4)
        .position(|w| w == [0x00, 0x00, 0x00, 0x40])
        .expect("the float 2.0")
        + 3;
    let mut b = bytes.to_vec();
    b[at] ^= 0x01;
    b
}

/// `model` and `manifest` written as `<dir>/m.onnx` and `<dir>/m.manifest.json`.
pub(crate) fn drop_model(bytes: &[u8], manifest: &ModelManifest) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("m.onnx");
    std::fs::write(&path, bytes).unwrap();
    std::fs::write(
        manifest_path(&path),
        serde_json::to_string_pretty(manifest).unwrap(),
    )
    .unwrap();
    (dir, path)
}

#[derive(Clone, Default)]
struct Captured(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for Captured {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Run `f` with this thread's logs captured, and return them.
pub(crate) fn capture_logs<T>(f: impl FnOnce() -> T) -> (T, String) {
    let sink = Captured::default();
    let writer = sink.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_writer(move || writer.clone())
        .with_ansi(false)
        .with_max_level(tracing::Level::DEBUG)
        .finish();
    let out = tracing::subscriber::with_default(subscriber, f);
    let mut text = sink.0.lock().unwrap().clone();
    let _ = text.flush();
    (out, String::from_utf8(text).unwrap())
}

/// Three samples of one UE on a straight line, 100 ms apart.
pub(crate) fn feed(manager: &mut NwdafManager, ue_id: i32) {
    for (i, (x, y, z)) in [(0.0, 0.0, 0.0), (1.0, 2.0, 0.0), (3.0, 5.0, 1.0)]
        .into_iter()
        .enumerate()
    {
        manager.record_measurement(UeMeasurement {
            ue_id,
            rsrp: -80.0,
            rsrq: None,
            sinr: None,
            position: Vector3::new(x, y, z),
            velocity: None,
            serving_cell_id: 1,
            timestamp_ms: 1_000 + 100 * i as u64,
        });
    }
}

/// One refusal per reason, each through the manager's own loading path: the
/// log says `NWDAF: model refused (<reason>)` and never `model verified`, no
/// model is active, and every prediction is counted as a fallback.
#[test]
fn each_refusal_reason_logs_refused_and_serves_only_the_fallback() {
    type Edit = fn(&mut ModelManifest, &mut Vec<u8>);
    let cases: [(&str, Edit); 4] = [
        ("sha256 mismatch", |_, b| *b = flipped(b)),
        ("opset mismatch", |m, _| m.opset = 11),
        ("input shape mismatch", |m, _| {
            m.inputs[0].shape = vec![1, 8, 3]
        }),
        ("output shape mismatch", |m, _| {
            m.outputs[0].shape = vec![1, 10, 3]
        }),
    ];
    for (label, edit) in cases {
        let (mut m, mut bytes) = shipped();
        edit(&mut m, &mut bytes);
        let (_dir, path) = drop_model(&bytes, &m);
        let mut manager = NwdafManager::new(100);
        let (result, logs) = capture_logs(|| manager.load_verified_trajectory_model(&path));
        let err = result.expect_err(label);
        assert_eq!(err.reason.label(), label);
        assert!(
            logs.contains(&format!(
                "NWDAF: model refused ({label}) id=nwdaf-trajectory-cv version=1.0.0: "
            )),
            "{label}: {logs}"
        );
        assert!(!logs.contains("model verified"), "{label}: {logs}");
        assert!(
            matches!(manager.model_state(), ModelState::Refused { reason, .. } if reason == label)
        );

        feed(&mut manager, 7);
        let p = manager
            .predict_trajectory(7, 1000)
            .expect("the fallback answers");
        assert_eq!(p.waypoints.len(), 10, "linear extrapolation's 10 waypoints");
        let status = manager.status();
        assert!(!status.verified);
        assert!(status.models.is_empty(), "no inventory for a refused model");
        assert_eq!(status.measurements_recorded, 3);
        assert_eq!((status.onnx_inferences, status.fallback_inferences), (0, 1));
    }
}

#[test]
fn a_model_without_a_manifest_is_refused() {
    let (_, bytes) = shipped();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("m.onnx");
    std::fs::write(&path, bytes).unwrap();
    let mut manager = NwdafManager::new(100);
    let (result, logs) = capture_logs(|| manager.load_verified_trajectory_model(&path));
    assert_eq!(result.unwrap_err().reason.label(), "manifest invalid");
    assert!(
        logs.contains("NWDAF: model refused (manifest invalid)"),
        "{logs}"
    );
    assert!(
        logs.contains("m.manifest.json"),
        "names the manifest it looked for"
    );
}

#[test]
fn the_manifest_sits_beside_the_model() {
    assert_eq!(
        manifest_path(std::path::Path::new(
            "/etc/nextgsim/models/nwdaf-trajectory-cv.onnx"
        )),
        PathBuf::from("/etc/nextgsim/models/nwdaf-trajectory-cv.manifest.json")
    );
}

#[test]
fn with_no_model_every_prediction_is_a_counted_fallback() {
    let mut manager = NwdafManager::new(100);
    feed(&mut manager, 1);
    for _ in 0..4 {
        manager.predict_trajectory(1, 1000).expect("linear");
    }
    // Too little history is no prediction, and is not counted.
    assert!(manager.predict_trajectory(99, 1000).is_none());
    let s = manager.status();
    assert_eq!(s.model, ModelState::NotConfigured);
    assert_eq!((s.onnx_inferences, s.fallback_inferences), (0, 4));
    assert_eq!(s.latency.fallback.count, 4);
    assert!(s.latency.fallback.p50_ms.is_some() && s.latency.onnx.p50_ms.is_none());
}

#[test]
fn the_status_json_carries_the_fields_the_demo_reads() {
    let mut manager = NwdafManager::new(100);
    manager.defer_trajectory_model(std::path::Path::new("/m.onnx"));
    manager.set_force_fallback(true);
    let v: serde_json::Value = serde_json::from_str(&manager.status().to_json()).unwrap();
    assert_eq!(v["schema"], STATUS_SCHEMA);
    assert_eq!(v["verified"], false);
    assert_eq!(v["force_fallback"], true);
    assert_eq!(v["model"]["state"], "deferred");
    assert_eq!(v["model"]["path"], "/m.onnx");
    for k in [
        "measurements_recorded",
        "onnx_inferences",
        "fallback_inferences",
    ] {
        assert_eq!(v[k], 0, "{k}");
    }
    assert!(v["latency"]["onnx"]["count"].is_u64());
    assert_eq!(v["models"], serde_json::json!([]));
}

/// The inventory entry validates against TS 28.105 `MLModel-Single`
/// (`6g_docs/specs/TS28105_AiMlNrm.yaml`, l.785-836, NRM version 19.5.0).
///
/// **Local properties only**: `MLModel-Single` is `allOf` `Top`
/// (`TS28623_GenericNrm.yaml`), which is not vendored, so `Top` contributes only
/// the `id` string this entry sets. The `attributes` properties and their types
/// below are copied from the vendored yaml, and `aIMLInferenceName` is checked
/// against its `NgRanInferenceType` arm (l.186-191), the one this entry uses.
#[test]
fn the_inventory_validates_against_ml_model_single_local_properties() {
    // MLModel-Single.allOf[1].properties.attributes.properties: name -> type.
    const ATTRIBUTES: [(&str, &str); 14] = [
        ("mLModelId", "string"),
        ("aIMLInferenceName", "AIMLInferenceName"),
        ("mLModelVersion", "string"),
        ("expectedRunTimeContext", "MLContext"),
        ("trainingContext", "MLContext"),
        ("runTimeContext", "MLContext"),
        ("supportedPerformanceIndicators", "array"),
        ("mLCapabilitiesInfoList", "array"),
        ("mLTrainingType", "enum"),
        ("inferenceScope", "array"),
        ("retrainingEventsMonitorRef", "Dn"),
        ("sourceTrainedMLModelRef", "DnRo"),
        ("aIMLInferenceReportRefList", "DnListRo"),
        ("usedByFunctionRefList", "DnListRo"),
    ];
    const TRAINING_TYPES: [&str; 4] = [
        "INITIAL_TRAINING",
        "PRE_SPECIALISED_TRAINING",
        "RE_TRAINING",
        "FINE_TUNING",
    ];
    const NG_RAN_INFERENCE_TYPES: [&str; 3] = [
        "NG_RAN_NETWORK_ENERGY_SAVING",
        "NG_RAN_LOAD_BALANCING",
        "NG_RAN_MOBILITY_OPTIMIZATION",
    ];

    let (m, _) = shipped();
    let entry = serde_json::to_value(MlModelInventory::from_manifest(&m)).unwrap();
    let obj = entry.as_object().unwrap();
    assert_eq!(
        obj.keys().collect::<Vec<_>>(),
        ["attributes", "id"],
        "Top's id plus the local attributes, nothing else"
    );
    assert_eq!(entry["id"], "nwdaf-trajectory-cv:1.0.0");
    let attrs = entry["attributes"].as_object().unwrap();
    assert!(!attrs.is_empty());
    for (k, v) in attrs {
        let ty = ATTRIBUTES
            .iter()
            .find(|(name, _)| name == k)
            .unwrap_or_else(|| panic!("{k} is not an MLModel-Single attribute"))
            .1;
        match ty {
            "string" => assert!(v.is_string(), "{k} must be a string"),
            "enum" => assert!(TRAINING_TYPES.contains(&v.as_str().unwrap()), "{k}={v}"),
            "AIMLInferenceName" => assert!(
                NG_RAN_INFERENCE_TYPES.contains(&v.as_str().unwrap()),
                "{k}={v}"
            ),
            other => panic!("{k} ({other}) is set but this test does not check it"),
        }
    }
    assert_eq!(attrs["mLModelId"], "nwdaf-trajectory-cv");
    assert_eq!(attrs["mLModelVersion"], "1.0.0");
}

/// Tests that need ONNX Runtime (the model-verify CI job).
mod ort {
    use super::*;

    fn require_ort() {
        assert!(
            std::env::var_os("ORT_DYLIB_PATH").is_some(),
            "this test needs ONNX Runtime: set ORT_DYLIB_PATH to the pinned \
             libonnxruntime.so (the model-verify CI job does)"
        );
    }

    /// The success signal: `model verified ... sha256=<pinned> ... golden=ok`,
    /// then predictions served by the model (`onnx_inferences > 0`,
    /// `fallback_inferences = 0`) with the model's own answer, and the inventory.
    #[test]
    #[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH); run by the model-verify CI job"]
    fn the_shipped_model_is_verified_and_serves_every_prediction() {
        require_ort();
        let path = model_dir().join("nwdaf-trajectory-cv.onnx");
        let mut manager = NwdafManager::new(100);
        let (result, logs) = capture_logs(|| manager.load_verified_trajectory_model(&path));
        result.expect("the shipped model verifies");
        assert!(
            logs.contains(
                "NWDAF: model verified id=nwdaf-trajectory-cv version=1.0.0 \
                 sha256=90a74d28c8e919ad8fdaa0f487976c8a4e803af3e318c5a30e02afcc3ec1f826 opset=13 "
            ),
            "{logs}"
        );
        assert!(logs.contains("golden=ok (3 vectors"), "{logs}");
        assert!(!logs.contains("model refused"), "{logs}");

        feed(&mut manager, 5);
        for _ in 0..3 {
            let p = manager.predict_trajectory(5, 1000).expect("predicts");
            // The model's single waypoint, 2*last - prev, not linear's ten.
            assert_eq!(p.waypoints.len(), 1);
            let w = p.waypoints[0].0;
            assert_eq!((w.x, w.y, w.z), (5.0, 8.0, 2.0));
        }
        let s = manager.status();
        assert!(s.verified);
        assert_eq!((s.onnx_inferences, s.fallback_inferences), (3, 0));
        assert_eq!(s.latency.onnx.count, 3);
        assert_eq!(s.models.len(), 1);
        assert_eq!(s.models[0].attributes.ml_model_id, "nwdaf-trajectory-cv");

        // Forcing the fallback applies from the next prediction, and back.
        manager.set_force_fallback(true);
        assert_eq!(
            manager.predict_trajectory(5, 1000).unwrap().waypoints.len(),
            10
        );
        manager.set_force_fallback(false);
        assert_eq!(
            manager.predict_trajectory(5, 1000).unwrap().waypoints.len(),
            1
        );
        let s = manager.status();
        assert_eq!((s.onnx_inferences, s.fallback_inferences), (4, 1));
    }

    /// The revert arm: the flipped copy is refused, so the same run serves only
    /// the fallback (`fallback_inferences > 0`, `onnx_inferences = 0`) and a
    /// consumer check for "the model served" fails.
    #[test]
    #[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH); run by the model-verify CI job"]
    fn the_flipped_model_is_refused_and_the_consumer_check_fails() {
        require_ort();
        let (m, bytes) = shipped();
        let (_dir, path) = drop_model(&flipped(&bytes), &m);
        let mut manager = NwdafManager::new(100);
        let (result, logs) = capture_logs(|| manager.load_verified_trajectory_model(&path));
        assert_eq!(result.unwrap_err().reason.label(), "sha256 mismatch");
        assert!(
            logs.contains("NWDAF: model refused (sha256 mismatch)"),
            "{logs}"
        );
        assert!(!logs.contains("model verified"));
        feed(&mut manager, 5);
        manager.predict_trajectory(5, 1000).unwrap();
        let s = manager.status();
        assert_eq!((s.onnx_inferences, s.fallback_inferences), (0, 1));
        let consumer_check = s.verified && s.onnx_inferences > 0 && s.fallback_inferences == 0;
        assert!(
            !consumer_check,
            "a consumer requiring the model must fail here"
        );
    }
}
