//! Manifest checks against the shipped d440b reference model
//! (`tools/d440b-model/`, written by `make_reference_model.py`).
//!
//! Every check except the golden vectors runs without ONNX Runtime, so those
//! tests run in every CI job. The golden tests need ONNX Runtime and live in
//! `registry.rs` (they are `#[ignore]`d and run by the model-verify job, which
//! fails them, rather than skipping, when `ORT_DYLIB_PATH` is unset).

use super::*;
use std::path::PathBuf;

/// The shipped artefact directory.
pub(crate) fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tools/d440b-model")
}

pub(crate) fn shipped() -> (ModelManifest, Vec<u8>) {
    let dir = model_dir();
    let text = std::fs::read_to_string(dir.join("nwdaf-trajectory-cv.manifest.json"))
        .expect("the shipped manifest");
    let bytes = std::fs::read(dir.join("nwdaf-trajectory-cv.onnx")).expect("the shipped model");
    (
        ModelManifest::from_json(&text).expect("the shipped manifest parses"),
        bytes,
    )
}

/// Offset of the last byte of the `two` initializer's float 2.0 (`00 00 00 40`).
/// Flipping it keeps the protobuf, the opset and the shapes valid, so ONLY the
/// sha256 can catch it: with that check patched out the model would reach the
/// runtime (and, with ONNX Runtime, fail the golden vectors instead).
pub(crate) fn flip_offset(bytes: &[u8]) -> usize {
    let at = bytes
        .windows(4)
        .position(|w| w == [0x00, 0x00, 0x00, 0x40])
        .expect("the model holds the float 2.0");
    at + 3
}

/// The shipped model with one byte flipped at [`flip_offset`].
pub(crate) fn flipped(bytes: &[u8]) -> Vec<u8> {
    let mut b = bytes.to_vec();
    b[flip_offset(bytes)] ^= 0x01;
    b
}

#[test]
fn the_shipped_model_passes_every_static_check() {
    let (m, bytes) = shipped();
    let report = check_static(&m, &bytes).expect("the shipped model is what its manifest pins");
    assert_eq!(report.sha256, m.sha256);
    assert_eq!(report.opset, 13);
    assert_eq!(
        sha256_hex(&bytes),
        "90a74d28c8e919ad8fdaa0f487976c8a4e803af3e318c5a30e02afcc3ec1f826",
        "the pinned sha256 (docs, CI and this test agree)"
    );
}

#[test]
fn the_protobuf_reader_reads_the_opset_and_the_graph_signature() {
    let (_, bytes) = shipped();
    let h = onnx_proto::parse_model(&bytes).expect("parses");
    assert_eq!(h.default_opset, Some(13));
    assert_eq!(
        h.inputs,
        vec![TensorSpec {
            name: "positions".into(),
            dtype: "float32".into(),
            shape: vec![1, -1, 3],
        }],
        "the symbolic n is -1"
    );
    assert_eq!(
        h.outputs,
        vec![TensorSpec {
            name: "next_position".into(),
            dtype: "float32".into(),
            shape: vec![1, 1, 3],
        }]
    );
}

/// Refusal reason 1/5: sha256. One flipped byte.
#[test]
fn a_one_byte_flip_is_refused_as_a_sha256_mismatch() {
    let (m, bytes) = shipped();
    let bad = flipped(&bytes);
    // The flip is invisible to every other static check...
    let h = onnx_proto::parse_model(&bad).expect("still a valid protobuf");
    assert_eq!(h.default_opset, Some(m.opset));
    assert_eq!(h.inputs, m.inputs);
    assert_eq!(h.outputs, m.outputs);
    // ...so only the sha256 refuses it.
    let err = check_static(&m, &bad).expect_err("a flipped byte must be refused");
    assert!(
        matches!(&err, RefusalReason::Sha256Mismatch { expected, actual }
            if *expected == m.sha256 && *actual == sha256_hex(&bad)),
        "{err}"
    );
    assert_eq!(err.label(), "sha256 mismatch");
}

/// Refusal reason 2/5: opset.
#[test]
fn a_different_opset_is_refused() {
    let (mut m, bytes) = shipped();
    m.opset = 12;
    let err = check_static(&m, &bytes).expect_err("opset 13 is not 12");
    assert_eq!(
        err,
        RefusalReason::OpsetMismatch {
            expected: 12,
            actual: Some(13)
        }
    );
    assert_eq!(err.label(), "opset mismatch");
}

/// Refusal reason 3/5: input shape.
#[test]
fn a_different_input_shape_is_refused() {
    let (mut m, bytes) = shipped();
    m.inputs[0].shape = vec![1, -1, 4];
    let err = check_static(&m, &bytes).expect_err("[1,n,3] is not [1,n,4]");
    assert!(
        matches!(&err, RefusalReason::InputShapeMismatch { actual, .. }
            if actual[0].shape == vec![1, -1, 3]),
        "{err}"
    );
    assert_eq!(err.label(), "input shape mismatch");
    // A renamed input or another element type is the same refusal.
    let edits: [fn(&mut ModelManifest); 2] = [
        |m| m.inputs[0].name = "pos".into(),
        |m| m.inputs[0].dtype = "float64".into(),
    ];
    for edit in edits {
        let (mut m, bytes) = shipped();
        edit(&mut m);
        assert_eq!(
            check_static(&m, &bytes).unwrap_err().label(),
            "input shape mismatch"
        );
    }
}

/// Refusal reason 4/5: output shape.
#[test]
fn a_different_output_shape_is_refused() {
    let (mut m, bytes) = shipped();
    m.outputs[0].shape = vec![1, 10, 3];
    let err = check_static(&m, &bytes).expect_err("[1,1,3] is not [1,10,3]");
    assert!(
        matches!(&err, RefusalReason::OutputShapeMismatch { actual, .. }
            if actual[0].shape == vec![1, 1, 3]),
        "{err}"
    );
    assert_eq!(err.label(), "output shape mismatch");
}

// Refusal reason 5/5 (golden mismatch) needs ONNX Runtime: see
// `registry::tests::ort::a_golden_mismatch_is_refused`.

#[test]
fn bytes_that_are_not_onnx_are_refused_after_the_sha256() {
    let (mut m, _) = shipped();
    let junk = b"\xff\xff\xff\xff not a model".to_vec();
    m.sha256 = sha256_hex(&junk);
    assert_eq!(
        check_static(&m, &junk).unwrap_err().label(),
        "not an onnx model"
    );
}

#[test]
fn an_inconsistent_manifest_is_refused() {
    let (m, _) = shipped();
    let base = serde_json::to_value(&m).unwrap();
    let upper = m.sha256.to_uppercase();
    type Edit = Box<dyn Fn(&mut serde_json::Value)>;
    let cases: Vec<(&str, Edit)> = vec![
        ("short sha256", Box::new(|v| v["sha256"] = "abc".into())),
        (
            "uppercase sha256",
            Box::new(move |v| v["sha256"] = upper.clone().into()),
        ),
        (
            "no golden",
            Box::new(|v| v["golden"] = serde_json::json!([])),
        ),
        (
            "golden length",
            Box::new(|v| v["golden"][0]["input"] = serde_json::json!([1.0])),
        ),
        (
            "negative tolerance",
            Box::new(|v| v["tolerance"] = (-1.0).into()),
        ),
        ("unknown field", Box::new(|v| v["sha265"] = "x".into())),
        (
            "missing opset",
            Box::new(|v| {
                v.as_object_mut().unwrap().remove("opset");
            }),
        ),
    ];
    for (what, edit) in cases {
        let mut v = base.clone();
        edit(&mut v);
        let err = ModelManifest::from_json(&v.to_string()).expect_err(what);
        assert_eq!(err.label(), "manifest invalid", "{what}: {err}");
    }
}

#[test]
fn the_shipped_manifest_carries_the_inventory_fields() {
    let (m, _) = shipped();
    assert_eq!(m.id, "nwdaf-trajectory-cv");
    assert_eq!(m.version, "1.0.0");
    assert_eq!(
        m.inference_name.as_deref(),
        Some("NG_RAN_MOBILITY_OPTIMIZATION")
    );
    assert_eq!(m.training_type.as_deref(), Some("INITIAL_TRAINING"));
    assert_eq!(m.golden.len(), 3);
}
