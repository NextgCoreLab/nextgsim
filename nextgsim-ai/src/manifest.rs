//! Verified model loading: the model manifest and its checks (nextgsim#463).
//!
//! An offline script stands in for the OAM training role of TS 38.300 §16.20.2
//! ("AI/ML Model Training is located in the OAM and AI/ML Model Inference is
//! located in the NG-RAN node"; a model "needs to be trained, validated and
//! tested before deployment"). It ships `<model>.onnx` beside a
//! `<model>.manifest.json` that pins what the artefact must be:
//!
//! - its `sha256`;
//! - the default-domain `opset` it imports;
//! - the name, element type and shape of every graph input and output
//!   (`-1` for a symbolic dimension);
//! - golden input/output vectors and a `tolerance`.
//!
//! [`check_static`] verifies the first three from the bytes alone, without ONNX
//! Runtime: the sha256 over the file, then the opset and the shapes parsed out of
//! the ONNX `ModelProto` by a minimal protobuf reader ([`onnx_proto`]).
//! [`check_golden`] then runs every golden vector through a loaded engine.
//! `SharedModelRegistry::register_verified` runs them in that order, and the
//! model is activated only when all of them pass. Each failed check has its own
//! [`RefusalReason`].
//!
//! # Non-normative
//!
//! TS 28.105 §6.4.1.1: "The way for loading the ML model is not in scope of the
//! present document." This manifest and its checks are an implementation choice,
//! and the inventory is shaped like `MLModel-Single` (TS 28.105 NRM) without
//! implementing the MnS. Golden vectors are valid only on the ONNX Runtime
//! version CI pins.

use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::inference::{InferenceEngine, OnnxEngine};
use crate::tensor::TensorData;

/// One graph input or output as the manifest pins it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorSpec {
    /// The graph input/output name.
    pub name: String,
    /// The element type, as ONNX names it in lowercase with a width
    /// (`float32`, `int64`, ...).
    pub dtype: String,
    /// The dimensions; `-1` is a symbolic (dynamic) dimension.
    pub shape: Vec<i64>,
}

impl fmt::Display for TensorSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}{:?}", self.name, self.dtype, self.shape)
    }
}

/// One golden vector: an input and the output the model must produce for it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoldenVector {
    /// The input tensor's shape (concrete: no `-1`).
    pub input_shape: Vec<i64>,
    /// The input tensor, row-major.
    pub input: Vec<f32>,
    /// The expected output tensor's shape.
    pub output_shape: Vec<i64>,
    /// The expected output tensor, row-major.
    pub output: Vec<f32>,
}

/// What a model artefact must be before it is activated.
///
/// `PartialEq` only: the golden vectors hold `f32`s.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelManifest {
    /// Logical model id (`mLModelId`).
    pub id: String,
    /// Model version (`mLModelVersion`).
    pub version: String,
    /// Lowercase hex sha256 of the `.onnx` file.
    pub sha256: String,
    /// The default-domain (`ai.onnx`) opset the model imports.
    pub opset: i64,
    /// The graph inputs, in graph order (initializers excluded).
    pub inputs: Vec<TensorSpec>,
    /// The graph outputs, in graph order.
    pub outputs: Vec<TensorSpec>,
    /// Golden vectors; at least one.
    pub golden: Vec<GoldenVector>,
    /// Maximum absolute difference per output element.
    pub tolerance: f32,
    /// `aIMLInferenceName` for the inventory (e.g. `NG_RAN_MOBILITY_OPTIMIZATION`).
    #[serde(default)]
    pub inference_name: Option<String>,
    /// `mLTrainingType` for the inventory (e.g. `INITIAL_TRAINING`).
    #[serde(default)]
    pub training_type: Option<String>,
    /// Who produced the artefact (free text, reported only).
    #[serde(default)]
    pub producer: Option<String>,
}

impl ModelManifest {
    /// Parse and sanity-check a manifest.
    ///
    /// # Errors
    /// [`RefusalReason::ManifestInvalid`] when it is not the schema's shape or is
    /// internally inconsistent (no golden vector, a golden whose length does not
    /// match its shape, a sha256 that is not 64 hex digits, a bad tolerance).
    pub fn from_json(text: &str) -> Result<Self, RefusalReason> {
        let manifest: Self = serde_json::from_str(text)
            .map_err(|e| RefusalReason::ManifestInvalid(format!("not a model manifest: {e}")))?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// Check the manifest is internally consistent.
    ///
    /// # Errors
    /// [`RefusalReason::ManifestInvalid`], naming the field.
    pub fn validate(&self) -> Result<(), RefusalReason> {
        let bad = |m: String| Err(RefusalReason::ManifestInvalid(m));
        if self.id.is_empty() || self.version.is_empty() {
            return bad("id and version must be non-empty".into());
        }
        if self.sha256.len() != 64
            || !self
                .sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return bad(format!(
                "sha256 {:?} is not 64 lowercase hex digits",
                self.sha256
            ));
        }
        if self.inputs.is_empty() || self.outputs.is_empty() {
            return bad("inputs and outputs must be non-empty".into());
        }
        if !(self.tolerance.is_finite() && self.tolerance >= 0.0) {
            return bad(format!(
                "tolerance {} must be finite and >= 0",
                self.tolerance
            ));
        }
        if self.golden.is_empty() {
            return bad("golden must hold at least one vector".into());
        }
        for (i, g) in self.golden.iter().enumerate() {
            for (what, shape, len) in [
                ("input", &g.input_shape, g.input.len()),
                ("output", &g.output_shape, g.output.len()),
            ] {
                let elems = shape.iter().try_fold(1usize, |acc, &d| {
                    usize::try_from(d).ok().and_then(|d| acc.checked_mul(d))
                });
                if elems != Some(len) {
                    return bad(format!(
                        "golden[{i}].{what} has {len} values for shape {shape:?}"
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Why a model was refused. Each check has its own reason, so a refusal names
/// exactly what failed.
#[derive(Debug, Clone, PartialEq)]
pub enum RefusalReason {
    /// The manifest is missing, unreadable or inconsistent.
    ManifestInvalid(String),
    /// The model file could not be read.
    Unreadable(String),
    /// The file's sha256 is not the manifest's.
    Sha256Mismatch {
        /// The manifest's sha256.
        expected: String,
        /// The file's sha256.
        actual: String,
    },
    /// The bytes are not a parseable ONNX `ModelProto`.
    NotOnnx(String),
    /// The default-domain opset is not the manifest's.
    OpsetMismatch {
        /// The manifest's opset.
        expected: i64,
        /// The model's default-domain opset, if it imports one.
        actual: Option<i64>,
    },
    /// The graph inputs are not the manifest's.
    InputShapeMismatch {
        /// The manifest's inputs.
        expected: Vec<TensorSpec>,
        /// The model's inputs.
        actual: Vec<TensorSpec>,
    },
    /// The graph outputs are not the manifest's.
    OutputShapeMismatch {
        /// The manifest's outputs.
        expected: Vec<TensorSpec>,
        /// The model's outputs.
        actual: Vec<TensorSpec>,
    },
    /// ONNX Runtime could not be loaded (no `libonnxruntime`, or too old).
    RuntimeUnavailable(String),
    /// ONNX Runtime rejected the model.
    LoadFailed(String),
    /// A golden vector did not reproduce.
    GoldenMismatch {
        /// Which golden vector (0-based).
        case: usize,
        /// What differed.
        detail: String,
    },
}

impl RefusalReason {
    /// The short reason logged as `model refused (<label>)`.
    pub fn label(&self) -> &'static str {
        match self {
            Self::ManifestInvalid(_) => "manifest invalid",
            Self::Unreadable(_) => "model unreadable",
            Self::Sha256Mismatch { .. } => "sha256 mismatch",
            Self::NotOnnx(_) => "not an onnx model",
            Self::OpsetMismatch { .. } => "opset mismatch",
            Self::InputShapeMismatch { .. } => "input shape mismatch",
            Self::OutputShapeMismatch { .. } => "output shape mismatch",
            Self::RuntimeUnavailable(_) => "runtime unavailable",
            Self::LoadFailed(_) => "load failed",
            Self::GoldenMismatch { .. } => "golden mismatch",
        }
    }
}

fn join(specs: &[TensorSpec]) -> String {
    specs
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

impl RefusalReason {
    /// What failed, without the label.
    pub fn detail(&self) -> String {
        match self {
            Self::ManifestInvalid(d)
            | Self::Unreadable(d)
            | Self::NotOnnx(d)
            | Self::RuntimeUnavailable(d)
            | Self::LoadFailed(d) => d.clone(),
            Self::Sha256Mismatch { expected, actual } => {
                format!("manifest {expected}, file {actual}")
            }
            Self::OpsetMismatch { expected, actual } => match actual {
                Some(a) => format!("manifest {expected}, model {a}"),
                None => format!("manifest {expected}, model imports no default-domain opset"),
            },
            Self::InputShapeMismatch { expected, actual }
            | Self::OutputShapeMismatch { expected, actual } => {
                format!("manifest [{}], model [{}]", join(expected), join(actual))
            }
            Self::GoldenMismatch { case, detail } => format!("golden[{case}] {detail}"),
        }
    }
}

impl fmt::Display for RefusalReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.label(), self.detail())
    }
}

/// What [`check_static`] established.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaticReport {
    /// The file's sha256 (equal to the manifest's).
    pub sha256: String,
    /// The model's default-domain opset (equal to the manifest's).
    pub opset: i64,
}

/// Lowercase hex sha256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .fold(String::with_capacity(64), |mut s, b| {
            use fmt::Write as _;
            let _ = write!(s, "{b:02x}");
            s
        })
}

/// The checks that need no ONNX Runtime, in order: sha256, then (from the parsed
/// `ModelProto`) opset, inputs, outputs.
///
/// The sha256 runs first so that a corrupted file is refused as a sha256
/// mismatch whatever the corruption did to the protobuf.
///
/// # Errors
/// The first check that fails.
pub fn check_static(manifest: &ModelManifest, bytes: &[u8]) -> Result<StaticReport, RefusalReason> {
    let actual = sha256_hex(bytes);
    if actual != manifest.sha256 {
        return Err(RefusalReason::Sha256Mismatch {
            expected: manifest.sha256.clone(),
            actual,
        });
    }
    let header = onnx_proto::parse_model(bytes).map_err(RefusalReason::NotOnnx)?;
    if header.default_opset != Some(manifest.opset) {
        return Err(RefusalReason::OpsetMismatch {
            expected: manifest.opset,
            actual: header.default_opset,
        });
    }
    if header.inputs != manifest.inputs {
        return Err(RefusalReason::InputShapeMismatch {
            expected: manifest.inputs.clone(),
            actual: header.inputs,
        });
    }
    if header.outputs != manifest.outputs {
        return Err(RefusalReason::OutputShapeMismatch {
            expected: manifest.outputs.clone(),
            actual: header.outputs,
        });
    }
    Ok(StaticReport {
        sha256: actual,
        opset: manifest.opset,
    })
}

/// Run every golden vector through `engine` and compare within the manifest's
/// tolerance.
///
/// # Errors
/// [`RefusalReason::GoldenMismatch`] for the first vector that does not
/// reproduce: an inference error, a different output shape or length, a
/// non-finite value, or an element further than `tolerance` from the expected.
pub fn check_golden(engine: &OnnxEngine, manifest: &ModelManifest) -> Result<(), RefusalReason> {
    for (case, g) in manifest.golden.iter().enumerate() {
        let mismatch = |detail: String| RefusalReason::GoldenMismatch { case, detail };
        let input = TensorData::float32(g.input.clone(), g.input_shape.clone());
        let out = engine
            .infer(&input)
            .map_err(|e| mismatch(format!("inference failed: {e}")))?;
        if out.shape().dims() != g.output_shape.as_slice() {
            return Err(mismatch(format!(
                "output shape {:?}, expected {:?}",
                out.shape().dims(),
                g.output_shape
            )));
        }
        let got = out
            .as_f32_slice()
            .ok_or_else(|| mismatch("output is not float32".into()))?;
        if got.len() != g.output.len() {
            return Err(mismatch(format!(
                "{} output values, expected {}",
                got.len(),
                g.output.len()
            )));
        }
        for (i, (&a, &e)) in got.iter().zip(&g.output).enumerate() {
            if !a.is_finite() || (a - e).abs() > manifest.tolerance {
                return Err(mismatch(format!(
                    "output[{i}] = {a}, expected {e} (tolerance {})",
                    manifest.tolerance
                )));
            }
        }
    }
    Ok(())
}

/// Load ONNX Runtime without panicking, so a missing or too-old
/// `libonnxruntime` is a refusal rather than a crash.
///
/// `ort` is built `load-dynamic`: its first use `dlopen`s `ORT_DYLIB_PATH` (or
/// `libonnxruntime.so`) and **panics** if that fails. Calling this first loads it
/// through the fallible path; once it succeeds, later uses find it loaded.
///
/// The first outcome is cached for the life of the process and `ort` is never
/// asked twice. That is not an optimisation: in `ort` 2.0.0-rc.11 a *failed*
/// load leaves its library slot marked initialised (its std `OnceLock` runs the
/// fallible init inside `Once::call_once_force`, which completes even on
/// `Err`), so any later `ort` call reads an uninitialised library handle. A
/// process whose runtime failed to load must therefore not touch `ort` again,
/// and every caller of this function sees the same refusal instead.
///
/// # Errors
/// [`RefusalReason::RuntimeUnavailable`] with `ort`'s reason.
pub fn ensure_runtime() -> Result<(), RefusalReason> {
    static RUNTIME: std::sync::OnceLock<Result<(), String>> = std::sync::OnceLock::new();
    RUNTIME
        .get_or_init(|| {
            let path = std::env::var("ORT_DYLIB_PATH")
                .ok()
                .filter(|p| !p.is_empty())
                .unwrap_or_else(|| "libonnxruntime.so".to_string());
            ort::init_from(&path)
                .map(|_| ())
                .map_err(|e| format!("{path}: {e}"))
        })
        .clone()
        .map_err(RefusalReason::RuntimeUnavailable)
}

/// A minimal, read-only protobuf reader for the parts of an ONNX `ModelProto`
/// the manifest pins (onnx.proto3 field numbers):
///
/// - `ModelProto.opset_import` (8): `OperatorSetIdProto { domain = 1, version = 2 }`;
/// - `ModelProto.graph` (7): `GraphProto { initializer = 5, input = 11, output = 12 }`;
/// - `ValueInfoProto { name = 1, type = 2 }`, `TypeProto { tensor_type = 1 }`,
///   `TypeProto.Tensor { elem_type = 1, shape = 2 }`,
///   `TensorShapeProto { dim = 1 }`, `Dimension { dim_value = 1, dim_param = 2 }`;
/// - `TensorProto { name = 8 }` (to drop initializers listed as graph inputs).
///
/// Every other field is skipped by wire type.
pub mod onnx_proto {
    use super::TensorSpec;

    /// What [`parse_model`] read.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ModelHeader {
        /// The version of the default domain (`""` or `ai.onnx`), if imported.
        pub default_opset: Option<i64>,
        /// The graph inputs that are not initializers.
        pub inputs: Vec<TensorSpec>,
        /// The graph outputs.
        pub outputs: Vec<TensorSpec>,
    }

    enum Field<'a> {
        Varint(u64),
        Bytes(&'a [u8]),
        Skipped,
    }

    struct Reader<'a> {
        buf: &'a [u8],
        pos: usize,
    }

    impl<'a> Reader<'a> {
        fn new(buf: &'a [u8]) -> Self {
            Self { buf, pos: 0 }
        }

        fn varint(&mut self) -> Result<u64, String> {
            let mut v = 0u64;
            for shift in (0..64).step_by(7) {
                let b = *self
                    .buf
                    .get(self.pos)
                    .ok_or_else(|| format!("truncated varint at byte {}", self.pos))?;
                self.pos += 1;
                v |= u64::from(b & 0x7f) << shift;
                if b & 0x80 == 0 {
                    return Ok(v);
                }
            }
            Err(format!("varint too long at byte {}", self.pos))
        }

        fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
            let end = self
                .pos
                .checked_add(n)
                .filter(|&e| e <= self.buf.len())
                .ok_or_else(|| format!("field of {n} bytes overruns at byte {}", self.pos))?;
            let s = &self.buf[self.pos..end];
            self.pos = end;
            Ok(s)
        }

        /// The next `(field number, value)`, or `None` at the end.
        fn next(&mut self) -> Result<Option<(u64, Field<'a>)>, String> {
            if self.pos >= self.buf.len() {
                return Ok(None);
            }
            let key = self.varint()?;
            let field = key >> 3;
            if field == 0 {
                return Err(format!("field number 0 at byte {}", self.pos));
            }
            let value = match key & 7 {
                0 => Field::Varint(self.varint()?),
                1 => {
                    self.take(8)?;
                    Field::Skipped
                }
                2 => {
                    let len = usize::try_from(self.varint()?)
                        .map_err(|_| "length does not fit".to_string())?;
                    Field::Bytes(self.take(len)?)
                }
                5 => {
                    self.take(4)?;
                    Field::Skipped
                }
                w => return Err(format!("unsupported wire type {w} at byte {}", self.pos)),
            };
            Ok(Some((field, value)))
        }
    }

    fn utf8(b: &[u8], what: &str) -> Result<String, String> {
        String::from_utf8(b.to_vec()).map_err(|_| format!("{what} is not UTF-8"))
    }

    /// ONNX `TensorProto.DataType` as the manifest's `dtype` names it.
    fn dtype_name(elem: u64) -> String {
        match elem {
            1 => "float32".into(),
            2 => "uint8".into(),
            3 => "int8".into(),
            5 => "int16".into(),
            6 => "int32".into(),
            7 => "int64".into(),
            9 => "bool".into(),
            10 => "float16".into(),
            11 => "float64".into(),
            other => format!("onnx-type-{other}"),
        }
    }

    fn value_info(b: &[u8]) -> Result<TensorSpec, String> {
        let mut name = None;
        let mut dtype = None;
        let mut shape = Vec::new();
        let mut r = Reader::new(b);
        while let Some((f, v)) = r.next()? {
            match (f, v) {
                (1, Field::Bytes(n)) => name = Some(utf8(n, "value name")?),
                (2, Field::Bytes(t)) => {
                    let mut tr = Reader::new(t);
                    while let Some((tf, tv)) = tr.next()? {
                        if let (1, Field::Bytes(tensor)) = (tf, tv) {
                            let mut te = Reader::new(tensor);
                            while let Some((ef, ev)) = te.next()? {
                                match (ef, ev) {
                                    (1, Field::Varint(e)) => dtype = Some(dtype_name(e)),
                                    (2, Field::Bytes(s)) => shape = dims(s)?,
                                    _ => {}
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        let name = name.ok_or("a graph value has no name")?;
        let dtype = dtype.ok_or_else(|| format!("graph value {name:?} is not a tensor"))?;
        Ok(TensorSpec { name, dtype, shape })
    }

    fn dims(b: &[u8]) -> Result<Vec<i64>, String> {
        let mut out = Vec::new();
        let mut r = Reader::new(b);
        while let Some((f, v)) = r.next()? {
            if let (1, Field::Bytes(d)) = (f, v) {
                // A dimension with neither value nor param is unknown: -1, like
                // a symbolic one.
                let mut dim = -1i64;
                let mut dr = Reader::new(d);
                while let Some((df, dv)) = dr.next()? {
                    if let (1, Field::Varint(n)) = (df, dv) {
                        dim = n as i64;
                    }
                }
                out.push(dim);
            }
        }
        Ok(out)
    }

    /// Read the opset imports and the graph's inputs and outputs.
    ///
    /// # Errors
    /// A description of the first malformed field, or a model with no graph.
    pub fn parse_model(bytes: &[u8]) -> Result<ModelHeader, String> {
        let mut default_opset = None;
        let mut graph = None;
        let mut r = Reader::new(bytes);
        while let Some((f, v)) = r.next()? {
            match (f, v) {
                (7, Field::Bytes(g)) => graph = Some(g),
                (8, Field::Bytes(o)) => {
                    let mut domain = String::new();
                    let mut version = None;
                    let mut or = Reader::new(o);
                    while let Some((of, ov)) = or.next()? {
                        match (of, ov) {
                            (1, Field::Bytes(d)) => domain = utf8(d, "opset domain")?,
                            (2, Field::Varint(n)) => version = Some(n as i64),
                            _ => {}
                        }
                    }
                    if domain.is_empty() || domain == "ai.onnx" {
                        default_opset = version.or(default_opset);
                    }
                }
                _ => {}
            }
        }
        let graph = graph.ok_or("the model has no graph")?;
        let mut initializers = Vec::new();
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut r = Reader::new(graph);
        while let Some((f, v)) = r.next()? {
            match (f, v) {
                (5, Field::Bytes(t)) => {
                    let mut tr = Reader::new(t);
                    while let Some((tf, tv)) = tr.next()? {
                        if let (8, Field::Bytes(n)) = (tf, tv) {
                            initializers.push(utf8(n, "initializer name")?);
                        }
                    }
                }
                (11, Field::Bytes(i)) => inputs.push(value_info(i)?),
                (12, Field::Bytes(o)) => outputs.push(value_info(o)?),
                _ => {}
            }
        }
        inputs.retain(|i| !initializers.contains(&i.name));
        Ok(ModelHeader {
            default_opset,
            inputs,
            outputs,
        })
    }
}

#[cfg(test)]
pub(crate) mod tests;
