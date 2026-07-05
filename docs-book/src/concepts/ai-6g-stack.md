# The AI and 6G Simulation Stack

nextgsim ships eight "AI-native 6G" crates — `nextgsim-ai`, `nextgsim-fl`,
`nextgsim-isac`, `nextgsim-semantic`, `nextgsim-she`, `nextgsim-nwdaf`,
`nextgsim-nkef`, `nextgsim-agent` — plus a set of UE- and gNB-side Rel-17/18
prototype modules (ambient IoT, ProSe/sidelink, UAV, ranging, MINT, energy
saving, MBS, NTN). This chapter is a deep dive into what those crates and modules
*actually do when the code runs*, as opposed to what the acronyms suggest. It is
the honesty-critical chapter of this book: the landing page and feature matrix
already carry per-row honesty badges, and everything here is at least as blunt.

The short version, which the rest of the chapter substantiates against source:
**"AI-native 6G" in nextgsim means a set of research-prototype network functions
that are compiled into the binaries but are switched OFF by default and are not
exercised by the validated end-to-end (E2E) path.** The one genuinely
operational piece of ML infrastructure is an ONNX Runtime inference wrapper
(`nextgsim-ai`), and **no `.onnx` model ships with the repository**, so even when
a 6G task is enabled it typically runs a non-neural fallback (mean-pooling,
linear extrapolation, TF-IDF, a kNN surrogate).

> **Honesty note:** These AI/6G modules are validated only by **per-crate unit
> tests** plus a **channel-level integration test**
> (`tests/src/sixg_task_integration.rs`); they are **not** third-party certified
> and **not** conformance-tested. The project's headline validation — the matched
> simulator driving its own gNB/UE against the NextGCore 5G core over N2
> (NGAP/SCTP) and N3 (GTP-U), with the **UE→UPF GTP-U data plane verified**
> (84/84 docker E2E green as of 2026-07-02) — runs with **every 6G/AI feature
> flag off** (see `config/gnb.yaml`, `config/ue.yaml`,
> `docker/exit-gate/gnb.yaml`), so none of the crates below participate in that
> result. All 6G features are **non-normative research prototypes**: there is no
> frozen 3GPP Rel-20 / 6G Stage-3 wire specification to conform to. 3GPP TS/TR
> numbers appear below only where they are cited in this repo's own source
> comments, phrased "per code comments".

## How these crates are wired into the binaries

Understanding the honesty story requires understanding three independent gates,
all of which default to "off":

1. **Compile gate (UE only).** In `nextgsim-ue/Cargo.toml` (lines 27-33) **seven**
   client crates (`nextgsim-she`, `-nwdaf`, `-nkef`, `-isac`, `-agent`, `-fl`,
   `-semantic`) are declared `optional = true`. There is **no `[features]` section
   and no `default` feature** enabling them, so a normal `cargo build`/`nr-ue`
   binary does not contain them at all — every UE-side 6G task type is behind
   `#[cfg(feature = "nextgsim-…")]` in `nextgsim-ue/src/tasks.rs`. Only **five** of
   the seven (`-she`, `-nwdaf`, `-isac`, `-fl`, `-semantic`) actually gate UE task
   types (`tasks.rs` lines 117-159 etc.); `-nkef` and `-agent` are declared
   optional but are not consumed by any UE task type. On the gNB
   side (`nextgsim-gnb/Cargo.toml`) the six crates are non-optional, so they are
   always compiled in, but that only affects binary size, not behaviour.

2. **Spawn gate (config flag).** Even when compiled, a 6G task loop is only
   `tokio::spawn`-ed if its config flag is set. On the gNB,
   `nextgsim-gnb/src/main.rs` computes `any_6g` from `she_enabled`,
   `nwdaf_enabled`, `nkef_enabled`, `isac_enabled`, `agent_enabled`,
   `federated_learning_enabled` and only then calls `init_6g_tasks`; each task is
   further guarded by its own flag. On the UE, `nextgsim-ue/src/main.rs`
   (`spawn_tasks`) does the same with `cfg!(feature = …) && *_enabled`. Every one
   of these flags defaults to `false` in `nextgsim-common/src/config.rs`
   (gNB: lines ~223-229; UE `UeConfig`: lines ~1168-1172) and is written `false`
   in the shipped YAML.

3. **Drive gate (a message actually arrives).** A spawned task is an idle
   actor until something sends it a message. There are only **four live
   send-sites** across the gNB into these tasks (verified by grep): the RRC task
   routes UE-originated 6G RRC messages to SHE/ISAC/NKEF
   (`nextgsim-gnb/src/rrc/task.rs`, `route_6g_ai_ml` / `route_6g_isac` /
   `route_6g_semantic`), and the ISAC task forwards a fused position to NWDAF
   (`nextgsim-gnb/src/isac/task.rs`). Those RRC messages only exist if a UE with
   the (off-by-default) 6G features emits them.

```
              compile gate            spawn gate              drive gate
 UE 6G crate  cargo feature   →   config *_enabled   →   a 6G RRC msg arrives
 (optional,       OFF             (false in YAML)          (none on E2E path)
  no default)                                                     │
                                                                  ▼
                                                        task actor does work
```

Net effect: the AI/6G stack is **opt-in plumbing**. It is well-structured,
unit-tested Rust, but on the standard registration + PDU-session + GTP-U ping
path it is dormant.

## What is real vs prototype

Depth is graded **full** (operational, does real work), **partial** (real
structure with a documented simplification on the operational path), or
**illustrative stub** (exists to demonstrate a shape/flow; the "smart" part is a
placeholder). "Drives a live path?" is answered for a *default* build/run.

| Component | Crate / module | Depth | Drives a live path? | Validation |
|---|---|---|---|---|
| ONNX inference engine | `nextgsim-ai` (`inference.rs` `OnnxEngine`) | full (as a wrapper) | Only if a task is enabled *and* a model is loaded; **no model ships** | unit tests |
| AI/ML NR air-interface models | `nextgsim-ai/src/nr_models.rs` | illustrative stub (lifecycle mgmt only) | No — library types, not wired to PHY | unit tests |
| Federated learning | `nextgsim-fl` + gNB `fl/task.rs`, UE `fl_participant/` | partial (FedAvg/FedProx/Byzantine/DP real) | No — off by default | unit tests + integration test |
| — "SecAgg" secure aggregation | `nextgsim-fl` `AggregationAlgorithm::MaskedSumDemo` | **illustrative stub** (masking demo, NOT Bonawitz) | No | unit tests |
| Semantic codec | `nextgsim-semantic` + UE `semantic_codec/` | illustrative stub (mean-pool / nearest-neighbour) | No — UE crate optional+off | unit tests (incl. honesty guard) |
| ISAC (sensing) | `nextgsim-isac` + gNB `isac/task.rs`, UE `isac_sensor/` | partial (fusion core) / stub (ML, SaaS) | No — off by default | unit tests + integration test |
| NWDAF analytics | `nextgsim-nwdaf` + gNB `nwdaf/task.rs`, UE `nwdaf_reporter/` | partial (linear extrapolation) | No — off by default | unit tests + integration test |
| NKEF knowledge exposure | `nextgsim-nkef` + gNB `nkef/task.rs` | partial (TF-IDF) / stub (neural embed, LLM) | No — off by default | unit tests |
| AI agent framework | `nextgsim-agent` + gNB `agent/task.rs` | partial (real intent executors, no LLM/RL) | No — off by default | unit tests + integration test |
| Service Hosting Env (SHE) | `nextgsim-she` + gNB `she/task.rs`, UE `she_client/` | partial (placement scheduler, no orchestration) | No — off by default | unit tests |
| UE ambient IoT | `nextgsim-ue/src/ambient_iot/` | illustrative stub (fleet + energy model) | No — sim-internal | unit tests |
| UE ProSe / sidelink | `nextgsim-ue/src/prose.rs`, `src/sidelink/` | prototype (bespoke, not wire-conformant) | Task spawned but idle by default | unit tests |
| UE ranging / MINT / UAV | `nextgsim-ue/src/{ranging,mint,uav}.rs` | prototype (bespoke encodings) | Task spawned but idle by default | unit tests |
| gNB energy saving | `nextgsim-gnb/src/energy/` | prototype (cell sleep model) | No — `energy_saving_enabled=false` | unit tests |
| gNB MBS (NGAP) | `nextgsim-gnb/src/mbs_ngap.rs` | prototype (Rel-17 MBS) | No — `mbs_enabled=false` | unit tests |
| NTN (gNB/RRC) | `nextgsim-gnb/src/rrc/ntn_gnb.rs` | prototype, **inert** | No — parsed but unwired | unit tests |

A note on scope: several Rel-17/18 features that *look* adjacent — **RedCap,
SNPN, MINT, UAV, XR** — are a **different maturity tier**. Those carry real NAS
IEs and are integrated end-to-end with NextGCore (documented in
[Configuration](../configuration.md#rel-1718-feature-configuration-ue)); they are
not part of the "AI-native 6G" research stack this chapter covers. The `mint`,
`uav`, and `sidelink` *UE modules* below are the sim-internal driver/context code
behind some of those features.

## Component deep-dives

### nextgsim-ai — the only operational ML backend

`nextgsim-ai` is the shared inference substrate. Its crate doc
(`nextgsim-ai/src/lib.rs`) states plainly that **`OnnxEngine` (ONNX Runtime) is
the "only operational inference backend"**. `inference.rs` binds the real `ort`
crate (`ort::session::Session`, `session.run(ort::inputs![…])`) with selectable
execution providers (CPU/CUDA/CoreML/DirectML/TensorRT) via `config.rs`
`ExecutionProvider`. `tensor.rs` (`TensorData`) and `metrics.rs` round it out.

The load-bearing caveat is repeated in the same doc comment: **no `.onnx` model
ships with the repository**. So the engine works, but every downstream crate that
"would use a model in production" falls back to a non-neural path when none is
loaded. `nr_models.rs` (cited "per code comments" to TS 38.843 AI/ML for the NR
air interface — CSI prediction, beam management, ML positioning) is a
model-lifecycle bookkeeping module (`NrModelDomain`), not something wired to a
physical layer; nextgsim has no PHY, so these are types without a live consumer.

Depth: **full** as an inference wrapper; the surrounding "AI air interface" is
lifecycle scaffolding.

### nextgsim-fl — federated learning, and the "SecAgg" that isn't

This is the most built-out AI crate (`nextgsim-fl/src/lib.rs` is ~3.3k lines).
The genuinely real parts:

- **FedAvg** (`FederatedAggregator::fedavg_aggregate`) — sample-weighted average.
- **FedProx** (`fedprox_aggregate`) — FedAvg plus a proximal correction toward
  the global model.
- **Byzantine-robust rules** — `krum_aggregate`, `multi_krum_aggregate`,
  `trimmed_mean_aggregate`, `median_aggregate`.
- **Differential privacy** — real Gaussian mechanism via Box-Muller
  (`sample_gaussian`), L2 clip-then-noise (`apply_dp_with_rng`), and an
  authoritative **Rényi-DP moments accountant** (`privacy_tracker_renyi`) that
  actually **enforces** the ε budget (`start_round` refuses to run once
  `would_exceed_after_next_round`).
- **Async FL** with staleness weighting (`AsyncFederatedAggregator`), gradient
  compression (`topk_compress`), and hierarchical tiers (`FLTier`).

The honesty flashpoint is **secure aggregation**. The enum variant is named
`AggregationAlgorithm::MaskedSumDemo`, and both its doc comment and a dedicated
module banner spell out that it is **NOT the Bonawitz et al. 2017 protocol and
provides zero privacy against the aggregator**. Why: `register_participant` mints
each participant's x25519 keypair *on the server*, and `secagg_aggregate_internal`
applies the pairwise masks server-side over plaintext gradients the server
already received. The masks themselves are derived by `demo_mask_from_secret`
using **`splitmix64`, a non-cryptographic PRG** — the comment says so — so even
the mask has no cryptographic strength. It exists only to demonstrate that
antisymmetric pairwise masks cancel in the sum. In other words, an earlier audit's
"SecAgg falls through to FedAvg" is now stale: there *is* real pairwise-masking
code, but it is deliberately labelled a demo, not secure aggregation.

Standards positioning is in the crate doc: aligned "per code comments" with the
FL concepts of TR 23.700-80 and the AI/ML training-management model of TS 28.105
§6.2b.2.15, whose NOTE 2 places FL algorithms / DP / secure-aggregation *outside*
the scope of standardization. The `mns_adapter` module is explicitly "illustrative
only — NOT a conformant YANG/JSON MnS schema".

On a live run the gNB `FlAggregatorTask` (`nextgsim-gnb/src/fl/task.rs`) and UE
`FlParticipantTask` are both **off by default**; the training round is exercised
by `test_fl_task_full_training_round_via_channel` in the integration test.

Depth: **partial-to-full** research engine; the "SecAgg" is an **illustrative
stub** by design.

### nextgsim-semantic — mean-pooling wearing a neural-codec label

The operational path is stated in the first line of
`nextgsim-semantic/src/codec.rs`: "operational path is mean-pooling encoder +
nearest-neighbor decoder." `NeuralEncoder::encode` calls the ONNX model *only* if
`load_model` was called; otherwise it runs `encode_fallback` (stride-based
**mean-pooling** with variance-derived importance weights) and emits a one-time
`warn!` via `warn_fallback_once` ("semantic codec output is degraded (not
neural)"). `NeuralDecoder::decode_fallback` is **nearest-neighbour upsampling**
(repeat each value `stride` times). Because no model ships, this is always the
path. There is even a self-guarding unit test
(`honesty_mean_pooling_fallback_label_present`) that fails if the "mean-pooling
fallback" label is ever removed from the source.

The crate doc (`nextgsim-semantic/src/lib.rs`) is careful about provenance:
"per code comments", TR 22.870 **does not define a dedicated semantic-communication
use case**; the crate prototypes at concept level the AI-service use-case family
it most naturally maps to (TR 22.870 §6.24 distributed AI computing, §6.25 AI/ML
training/inference), and TR 22.870 is a Stage-1 study with no Stage-3 wire spec.
Modules like `jscc.rs`, `rate_distortion.rs`, `knowledge.rs`, and `goal.rs` are
concept prototypes in the same spirit.

The UE `SemanticCodecTask` (`nextgsim-ue/src/semantic_codec/`) is behind the
optional `nextgsim-semantic` feature *and* `semantic_comm_enabled`, both off.
Note a routing subtlety: when a UE does emit a "semantic" RRC message, the gNB has
no semantic crate; `route_6g_semantic` records **metadata only** (source and
`data_len`) into NKEF's knowledge graph — it does not decode a payload.

Depth: **illustrative stub**.

### nextgsim-isac — a research fusion core with stubbed ML

`nextgsim-isac/src/lib.rs` opens with an unusually candid scope note: it models
ISAC concepts from TR 22.837, "a Stage-1 feasibility study — no normative
procedures/encodings exist; the TR foreword states it 'shall not be
implemented'", and is therefore "a research prototype, NOT a conformance
implementation." It then separates real from stub:

- **Research-faithful:** the fusion/positioning/tracking core — trilateration,
  an EKF, Bayesian fusion, and an OFDM-radar signal model (`mapping.rs`,
  `waveform.rs`, `saas.rs`).
- **Illustrative stubs:** the Sensing-as-a-Service exposure API, the SHE workload
  integration, and — importantly — the **ML positioning**. In
  `ml_sensing.rs`, the `MlModelType` neural variants (`DeepPositioning`,
  `TrajectoryLstm`, `TransformerTracking`, `FingerprintCnn`) are labelled
  placeholders: "Inference today is always a **kNN surrogate** over the training
  cache regardless of the selected variant", and `MlPositioningResult::actual_algorithm`
  reports what really ran.
- **Out of scope:** authorization/consent is a default-allow research stub
  (`saas::SensingPolicy`); charging and on-the-wire sensing security are not
  modelled.

Live wiring exists but is dormant by default: the gNB `IsacTask`
(`nextgsim-gnb/src/isac/task.rs`) receives `SensingData`, and on producing a fused
position forwards it to NWDAF as `NwdafMessage::UeMeasurement` — this is the
one AI→AI cross-task edge in the codebase. The UE `IsacSensorTask` feeds it, both
gated off. Exercised by `test_isac_task_sensing_data_recorded_via_channel`.

Depth: **partial** (fusion) / **illustrative stub** (ML, SaaS).

### nextgsim-nwdaf — analytics with linear extrapolation, not ML

`nextgsim-nwdaf` presents a four-layer analytics structure per TS 23.288 "per
code comments" (data collection → real-time z-score anomaly → predictive →
prescriptive → closed-loop), with MTLF/AnLF split (`mtlf.rs`, `anlf.rs`),
analytics IDs (`analytics_id.rs`), event exposure, DCCF, and federation modules.
That is a lot more surface than the acronym-existence level.

The operational reality is in `predictor.rs`: the header says the simulator ships
no ONNX model, so `OnnxPredictor` runs `predict_linear` — "it fits a line through
the recent position/time samples and projects forward." It wraps the
`nextgsim-ai` engine and *will* use a learned model if one is loaded, but since
none is shipped, **constant-velocity linear extrapolation is what executes**, and
`PredictionMethod::LinearExtrapolation` is reported accordingly.

The gNB `NwdafTask` (`nextgsim-gnb/src/nwdaf/task.rs`) can emit a handover
recommendation back to RRC (`NwdafHandoverRecommendation`); the UE
`NwdafReporterTask` reports measurements. Both off by default; exercised by
`test_nwdaf_to_rrc_handover_recommendation_arrives` and
`test_nwdaf_task_predict_trajectory_via_channel`.

Depth: **partial** — real service structure, non-ML predictor.

### nextgsim-nkef — TF-IDF and a hash-mock, not an LLM

`nextgsim-nkef` is the "network knowledge exposure" prototype: a knowledge graph
(`storage.rs`, `ontology.rs`), keyword/vector query (`query.rs`, `vector.rs`),
RAG context assembly (`rag.rs`), and access control (`access.rs`). Its embeddings
are honest about themselves in `embedder.rs`: the **operational** embedder is
`TextEmbedder`, computing **TF-IDF** vectors (no external model); the
`OnnxEmbedder`, with no model loaded (the default), falls back to
`mock_neural_embedding` — "a deterministic hash-based multi-scale mock, NOT a
neural network." It can load a sentence-transformer via `nextgsim-ai` if provided,
but none ships. There is **no LLM connection** anywhere.

The gNB `NkefTask` (`nextgsim-gnb/src/nkef/task.rs`) is off by default and is the
sink for the gNB's "semantic" RRC routing (metadata only, as noted above).

Depth: **partial** (TF-IDF) / **illustrative stub** (neural embedding, LLM/RAG).

### nextgsim-agent — real intent executors, no LLM/RL

Here an earlier audit finding is genuinely out of date. The gap analysis said
"all intents return success without actual execution (placeholder)". The current
`nextgsim-agent/src/execution.rs` header states it "perform[s] **real computation
for each intent type instead of returning placeholder successes**", via an
`IntentExecutor` trait returning a rich `IntentExecutionResult`
(`IntentStatus::{Success, PartialSuccess, Failed, Blocked}`, `affected_resources`,
timing). The crate also has `conflict.rs` (multi-agent conflict resolution),
`safety.rs` (guardrails), `audit.rs` (audit trail), `learning.rs`,
`coordination.rs`, and `nwdaf_bridge.rs` / `nkef_bridge.rs`. Authentication is
OAuth 2.0-style tokens (`lib.rs`). What is still **not** there: any LLM-based
intent understanding, reinforcement learning, or an A2A/MCP protocol.

The gNB `AgentTask` (`nextgsim-gnb/src/agent/task.rs`) is off by default;
exercised by `test_agent_register_then_submit_intent`.

Depth: **partial** — real deterministic executors, no learning/LLM layer.

### nextgsim-she — a placement scheduler, not an orchestrator

`nextgsim-she/src/lib.rs` calls itself "a **research model** for distributed
AI/ML compute placement, inspired by the tiered-edge concepts of TS 23.558
(EDGEAPP); it does **not** implement TS 23.558." It provides a three-tier
(LocalEdge <10 ms / RegionalEdge <20 ms / CoreCloud) placement `scheduler.rs`
with policies (ClosestToEdge / MostAvailable / LowestUtilization / PreferredFirst /
EnergyAware), resource capacity
tracking (`resource.rs`), workload lifecycle (`workload.rs`), plus `autoscale.rs`,
`sla.rs`, and `security.rs`. Tier latencies are configured constants, not
measured; there is no real container/VM orchestration. The gNB `SheTask` can load
ONNX models per tier through `nextgsim-ai` (subject to the no-model-ships caveat).

Depth: **partial** — real scheduling logic, simulated infrastructure.

### UE- and gNB-side prototype modules

These live in the binaries rather than the AI crates, and are sim-internal Rust
logic with **bespoke encodings that are not wire-conformant** (consistent with
[gap-protocols](../gaps/gap-protocols.md) and
[6g-gap-analysis](../gaps/6g-gap-analysis.md)):

- **UE ambient IoT** (`nextgsim-ue/src/ambient_iot/fleet.rs`, "Rel-18, TS 22.369"
  per code comments): energy-harvesting device model + `FleetManager`. No air
  interface; sim-internal.
- **UE ProSe / sidelink** (`prose.rs`, `sidelink/pc5.rs`, `sidelink/positioning.rs`,
  "TS 23.303/23.304/23.586" per code comments): PC5 link management, direct
  discovery, L2/L3 relay, and SL-PRS/RTT/AoA positioning via multilateration.
- **UE ranging** (`ranging/task.rs`, "TS 23.586"): RTT + multi-frequency carrier
  phase with ambiguity resolution.
- **UE MINT** (`mint/task.rs`, "TS 23.761"): multi-IMSI/USIM context management.
- **UE UAV** (`uav.rs`, wrapping `rrc/uav.rs`, "TS 23.256"): UAV authorization,
  C2 link quality thresholds, geofence, Remote-ID.
- **gNB energy saving** (`energy/task.rs`, Rel-18): cell sleep/dormant modes and
  efficiency metrics — spawned only if `energy_saving_enabled`.
- **gNB MBS** (`mbs_ngap.rs`, "Rel-17, TS 38.413 §8.21"): MBS session
  start/stop NGAP signalling — behind `mbs_enabled`.
- **NTN** (`rrc/ntn_gnb.rs`): satellite timing-advance / HARQ / beam models that
  are **parsed but inert** — the gNB stores the `ntn_config` block and logs it but
  does not wire it into live RRC/NGAP signalling (see the NTN row in
  [Configuration](../configuration.md#ntn-research-prototype)).

The Rel-18 UE tasks (Ranging, MINT, Sidelink) are unusual: they are **always
compiled and always spawned** (`nextgsim-ue/src/main.rs`, `spawn_tasks`), unlike
the feature-gated AI tasks. But an always-spawned actor is still just an idle
message loop until a CLI command or config drives it, so on the standard E2E run
they sit waiting.

## How to read the AI-native claims

When you see "AI-native 6G" in nextgsim's docs, banners, or crate names, translate
it as follows, all verifiable against the files above:

- **"Supported" means "a crate/task exists and compiles"**, not "runs on the
  default path." Every AI/6G flag is `false` in the shipped YAML, and the UE
  crates are absent from a default build entirely.
- **"AI/ML" usually means a non-neural fallback.** No `.onnx` model ships, so
  semantic → mean-pooling, NWDAF → linear extrapolation, NKEF → TF-IDF/hash-mock,
  ISAC ML → kNN surrogate. The ONNX engine is real; the models are BYO.
- **"Secure aggregation" is a mask-cancellation demo**, explicitly not the
  Bonawitz protocol and with zero privacy against the aggregator.
- **The crates are self-honest.** Read the crate-root and struct doc comments
  first — they carry the caveats (and one, semantic's codec, has a unit test that
  fails if the caveat label is deleted). The 3GPP TS/TR citations in this chapter
  are quoted from those comments and are frequently qualified in-source (e.g.
  TR 22.837 "shall not be implemented"; TR 22.870 "does not define a dedicated
  semantic use case").
- **Validation is unit-level, not conformance-level.** The channel-level
  `sixg_task_integration.rs` proves the tasks spawn, exchange messages, and change
  state; it does not prove any 3GPP behaviour, and it is not part of the
  UE→UPF GTP-U docker E2E.

To actually *run* any of this, set the corresponding `*_enabled` flag in the gNB
YAML (`config/gnb.yaml`) and — for the UE — build with the matching cargo feature
*and* set the UE flag. See [Configuration](../configuration.md) for the flag list.

## Where to look next

- [6G Gap Analysis](../gaps/6g-gap-analysis.md) — the package-by-package gap
  ledger this chapter is consistent with. Note: its completion percentages
  predate the current, larger and more self-honest crates (e.g. FL's
  `MaskedSumDemo`, agent's real `IntentExecutor`), so trust the source over the
  older figures where they differ.
- [Protocol Crates 6G Gap Analysis](../gaps/gap-protocols.md) — the NAS/NGAP/RRC
  side, including where NTN/sidelink encodings are bespoke and unwired.
- [Configuration](../configuration.md) — every `*_enabled` flag, the NTN
  "logging-only/inert" note, and the Rel-17/18 UE features (RedCap/SNPN/MINT/UAV)
  that *are* integrated E2E and are a separate tier from this stack.
- [gNB Architecture](../architecture/gnb.md) and [UE Architecture](../architecture/ue.md)
  — the actor/task model these 6G tasks plug into, and how spawning is gated.
- [Registration Call Flow](registration-flow.md) and
  [PDU Session & User Plane](pdu-session-userplane.md) — the validated live paths,
  which run with the AI/6G stack switched off.
- [Features & APIs](../reference.md) — the honesty-badged feature matrix that
  summarises implemented / partial / prototype status per row.
