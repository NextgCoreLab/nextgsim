# Gap Analysis: 6G-Advanced Crates vs IMT-2030 Requirements

> Generated: 2026-02-07 | Crates analyzed: `nextgsim-she`, `nextgsim-isac`, `nextgsim-fl`, `nextgsim-semantic`

## Executive Summary

| Crate | Files | LoC (approx) | `todo!`/`unimplemented!` | Completeness | Verdict |
|-------|-------|-------------|--------------------------|-------------|---------|
| `nextgsim-she` | 8 `.rs` | ~1,200 | 0 | **85%** | Solid foundation; missing live migration, auto-scaling, energy model |
| `nextgsim-isac` | 1 `.rs` | ~1,800 | 0 | **80%** | Deep algorithms; missing TDoA, radar waveform, AI-native sensing |
| `nextgsim-fl` | 1 `.rs` | ~1,770 | 0 | **82%** | Strong protocol coverage; missing hierarchical FL, heterogeneity-aware scheduling |
| `nextgsim-semantic` | 6 `.rs` | ~1,700 | 0 | **78%** | Good abstractions; missing learned model training loop, knowledge graph, goal-oriented coding |

**Overall 6G-advanced readiness: ~81%** -- core algorithmic foundations are in place with working implementations and tests. The primary gaps are in advanced orchestration, AI-native features, and simulation-level integration.

---

## 1. nextgsim-she -- Secure Heterogeneous Edge Computing

**Reference**: 3GPP TS 23.558 (Edge Application Architecture), IMT-2030 compute-fabric KPIs

### What is Implemented

| Component | File | Key Types/Functions | Status |
|-----------|------|---------------------|--------|
| 3-tier compute model | `tier.rs` | `ComputeTier` (LocalEdge/RegionalEdge/CoreCloud), `ComputeNode`, `TierManager` | Complete |
| Tier latency constraints | `tier.rs` | `max_latency_ms()` (10/20/MAX) | Complete |
| Capability model | `tier.rs` | `ComputeCapability` (Inference/FineTuning/Training), `supported_on()` | Complete |
| Resource tracking | `resource.rs` | `ResourceCapacity`, `ResourceUsage`, `ResourceUtilization` | Complete |
| Workload lifecycle | `workload.rs` | `Workload`, `WorkloadState` (7 states), `WorkloadRequirements` builder | Complete |
| Scheduler with policies | `scheduler.rs` | `WorkloadScheduler` with 4 policies, `PlacementDecision`, `find_placement()` | Complete |
| Workload migration | `scheduler.rs` | `migrate()` -- release source, allocate target | Complete |
| Async task loop | `task.rs` | `SheTask::run()`, message-driven via `mpsc` channel | Complete |
| ONNX inference integration | `task.rs` | `handle_inference_request()`, `handle_batch_inference()`, model load/unload | Complete |
| Message protocol | `messages.rs` | `SheMessage` (12 variants), `SheResponse` (8 variants), serializable API types | Complete |
| Error hierarchy | `error.rs` | `SheError` with 8 variants, `From<AiError>` | Complete |
| Unit tests | all files | 20+ tests covering placement, migration, policies | Complete |

### What is Missing for 6G

| Gap | IMT-2030 Relevance | Severity |
|-----|---------------------|----------|
| **Live workload migration** with state transfer | IMT-2030 requires seamless compute mobility for sub-1ms handover | High |
| **Auto-scaling / elastic provisioning** | Dynamic node pool expansion per load | Medium |
| **Energy-aware scheduling** | IMT-2030 sustainability target (100x energy efficiency vs 5G) | High |
| **Multi-tenancy / network slicing** per-slice compute | 3GPP TS 23.501 slice-specific SHE | Medium |
| **GPU/NPU resource types** beyond FLOPS+memory | Accelerator heterogeneity (TPU, FPGA) | Medium |
| **Service mesh / inter-tier communication** latency simulation | No actual latency modeling between tiers | Medium |
| **Security context** (attestation, TEE) | IMT-2030 trustworthy AI compute | Low |
| **SLA enforcement / QoS monitoring** | Workload SLA violation detection | Low |

### Completeness: **85%**

The crate has a production-grade architecture: 8 well-structured modules, builder patterns, 4 scheduling policies, async message loop, ONNX integration. The primary gap is advanced orchestration (auto-scaling, energy, slicing) rather than foundational missing pieces.

---

## 2. nextgsim-isac -- Integrated Sensing and Communication

**Reference**: 3GPP TR 22.837 (sensing use cases), 3GPP TR 38.857, IMT-2030 sensing KPIs

### What is Implemented

| Component | Location | Key Functions/Types | Status |
|-----------|----------|---------------------|--------|
| Sensing measurement types | `lib.rs:87-103` | `SensingType` -- ToA, TDoA, AoA, ZoA, Rss, Doppler, Rtt | Complete |
| Sensing data model | `lib.rs:106-143` | `SensingMeasurement`, `SensingData`, `DataSource` | Complete |
| Iterative trilateration (Gauss-Newton) | `lib.rs:338-441` | `trilaterate()` with weighted LS, Levenberg-Marquardt damping | Complete |
| Extended Kalman Filter (6-state) | `lib.rs:528-883` | `ExtendedKalmanFilter` with predict, update_range, update_position, update_aoa | Complete |
| Multi-sensor Bayesian fusion | `lib.rs:909-992` | `SensorEstimate`, `bayesian_fuse()` via information filter | Complete |
| Combined multi-sensor pipeline | `lib.rs:1004-1151` | `fuse_multi_sensor()` -- groups ToA/RSS/AoA, runs separate solvers, fuses | Complete |
| Object detection (TR 22.837 UC1) | `lib.rs:1153-1222` | `detect_object()` -- RSS-change-based with RMS threshold | Complete |
| Velocity estimation (Doppler) | `lib.rs:1224-1355` | `estimate_velocity_doppler()` -- 3D LS velocity from multiple Doppler shifts | Complete |
| Range estimation (RTT) | `lib.rs:1357-1392` | `estimate_range_rtt()` -- c*RTT/2 with processing delay subtraction | Complete |
| Sensing-comm resource manager | `lib.rs:1394-1505` | `SensingCommResourceManager` -- fraction-based split with proportional controller | Complete |
| ISAC manager | `lib.rs:1507-1682` | `IsacManager` with anchor registry, trilateration, Bayesian fusion, EKF, tracking | Complete |
| Tracking state | `lib.rs:163-234` | `TrackingState` with simplified Kalman update and prediction | Complete |
| Unit tests | `lib.rs:1686+` | 15+ tests: trilateration, EKF convergence, Bayesian fusion, Doppler, RTT | Complete |

### What is Missing for 6G

| Gap | IMT-2030 Relevance | Severity |
|-----|---------------------|----------|
| **TDoA processing** | Enum defined but no dedicated solver (only ToA/Rtt trilateration used) | Medium |
| **Radar waveform modeling** (OFDM-radar, FMCW) | IMT-2030 native sensing with shared waveform | High |
| **AI-native sensing** (ML-based positioning/detection) | Replaces/augments geometric solvers with neural models | High |
| **Clutter/multipath modeling** | No NLOS detection or multipath mitigation | Medium |
| **Bistatic/multistatic sensing** geometry | Only monostatic anchor model; no reflected-path geometry | Medium |
| **Sensing-as-a-service API** | 3GPP TR 22.837 exposure interface for applications | Medium |
| **Environment mapping** (SLAM-like) | 6G ambient sensing for spatial awareness | Low |
| **ZoA (elevation) update** in EKF | `update_aoa` handles azimuth only; ZoA listed but unused | Low |
| **Doppler measurement integration** into EKF | EKF supports range/position/AoA updates but not direct Doppler updates | Low |

### Completeness: **80%**

The deepest algorithmic crate: real iterative trilateration with Gauss-Newton, a proper 6-state EKF with Joseph-form covariance update, Bayesian multi-sensor fusion, and 3GPP-aligned use cases. The main gaps are waveform-level simulation and AI-native sensing.

---

## 3. nextgsim-fl -- Federated Learning

**Reference**: 3GPP TR 23.700-80 (FL architecture), IMT-2030 AI-native air interface

### What is Implemented

| Component | Location | Key Functions/Types | Status |
|-----------|----------|---------------------|--------|
| FedAvg aggregation | `lib.rs:813-833` | `fedavg_aggregate()` -- sample-weighted averaging | Complete |
| FedProx aggregation | `lib.rs:851-873` | `fedprox_aggregate()` -- proximal correction with mu parameter | Complete |
| Secure aggregation (SecAgg) | `lib.rs:358-459` | `SecAggParticipant` with x25519 DH, pairwise masking, mask cancellation | Complete |
| Differential privacy (Gaussian) | `lib.rs:693-719` | `apply_dp()` -- L2 gradient clipping + Gaussian noise (Box-Muller) | Complete |
| Privacy budget tracking | `lib.rs:126-183` | `PrivacyBudgetTracker` -- linear composition, per-round epsilon | Complete |
| Gradient compression (top-k) | `lib.rs:311-352` | `topk_compress()` / `topk_decompress()` with partial sort | Complete |
| Synchronous FL rounds | `lib.rs:206-253` | `TrainingRound` with deadline, expected participants | Complete |
| Async FL aggregator | `lib.rs:964-1208` | `AsyncFederatedAggregator` with staleness-weighted averaging (alpha decay) | Complete |
| Participant management | `lib.rs:256-268` | `Participant` struct, registration, active status | Complete |
| Message protocol | `lib.rs:1214-1262` | `FlMessage` (5 variants), `FlResponse` (5 variants) | Complete |
| Unit tests | `lib.rs:1281+` | 18 tests: FedAvg, FedProx, SecAgg mask cancellation, DP clipping/noise, staleness, compression | Complete |

### What is Missing for 6G

| Gap | IMT-2030 Relevance | Severity |
|-----|---------------------|----------|
| **Hierarchical FL** (edge -> regional -> cloud) | 6G multi-tier aggregation matching SHE tiers | High |
| **Client selection / scheduling** (heterogeneity-aware) | Bandwidth/compute-aware participant selection per 3GPP TR 23.700-80 | High |
| **Model versioning / distribution** | Described in doc but not implemented; no model store/fetch | Medium |
| **Advanced privacy** (Renyi/zCDP composition) | Linear composition is conservative; advanced accounting needed | Medium |
| **Gradient quantization** (beyond top-k) | 1-bit, ternary, or stochastic quantization for extreme compression | Medium |
| **Personalization** (local fine-tuning layer) | 6G UE-specific model adaptation | Medium |
| **Robustness** (Byzantine-tolerant aggregation) | Krum, trimmed mean, or similar for adversarial clients | Medium |
| **Convergence metrics / training dashboard** | Loss curves, per-participant contribution tracking | Low |
| **Integration with SHE crate** | FL should schedule on SHE compute tiers | Low |

### Completeness: **82%**

Notably strong in protocol diversity (3 aggregation algorithms + async FL), proper cryptographic SecAgg with x25519, and real differential privacy. The top gap is orchestration: hierarchical multi-tier FL and compute-aware client scheduling.

---

## 4. nextgsim-semantic -- Semantic Communication

**Reference**: IMT-2030 goal-oriented communication, Shannon-Weaver Level B/C

### What is Implemented

| Component | File | Key Types/Functions | Status |
|-----------|------|---------------------|--------|
| Semantic features model | `lib.rs:84-159` | `SemanticFeatures` with importance weighting and pruning | Complete |
| Task-type taxonomy | `lib.rs:162-178` | `SemanticTask` (7 variants incl. Custom) | Complete |
| Channel quality model | `lib.rs:181-237` | `ChannelQuality`, `ChannelCategory`, `recommended_compression()` | Complete |
| Basic encoder (mean-pooling) | `lib.rs:264-358` | `SemanticEncoder` with variance-based importance | Complete |
| Basic decoder (NN upsample) | `lib.rs:367-427` | `SemanticDecoder` with task-aware decoding | Complete |
| ONNX neural encoder | `codec.rs:42-175` | `NeuralEncoder` with model load, fallback to mean-pooling | Complete |
| ONNX neural decoder | `codec.rs:180-289` | `NeuralDecoder` with model load, fallback to NN upsample | Complete |
| Combined neural codec | `codec.rs:294-347` | `NeuralCodec` with round-trip encode/decode | Complete |
| JSCC encoder (channel-adaptive) | `jscc.rs:87-241` | `JsccEncoder` with SNR-adaptive symbol count, power normalization | Complete |
| JSCC decoder | `jscc.rs:244-341` | `JsccDecoder` with NN upsample fallback | Complete |
| JSCC codec + round-trip | `jscc.rs:374-429` | `JsccCodec::round_trip()` | Complete |
| Quality metrics | `metrics.rs` | `cosine_similarity()`, `mse()`, `psnr()`, `top_k_accuracy()`, `evaluate()` | Complete |
| Rate-distortion controller | `rate_distortion.rs` | `RdController` with Gaussian R-D function, Shannon capacity, target modes | Complete |
| Multi-modal traits | `multimodal.rs:22-58` | `SemanticEncode<T>`, `SemanticDecode<T>` generic traits | Complete |
| Vector encoder/decoder | `multimodal.rs:68-125` | `VectorEncoder`, `VectorDecoder` implementing traits | Complete |
| Image/Audio/Video markers | `multimodal.rs:136-239` | `ImageData`, `AudioData`, `VideoData` with `SemanticEncode<ImageData>` impl | Complete |
| Message protocol | `lib.rs:441-481` | `SemanticMessage` (3 variants), `SemanticResponse` (3 variants) | Complete |
| Unit tests | all files | 25+ tests across all modules | Complete |

### What is Missing for 6G

| Gap | IMT-2030 Relevance | Severity |
|-----|---------------------|----------|
| **Learned codec training loop** | No mechanism to train/fine-tune encoder/decoder end-to-end | High |
| **Knowledge graph / shared background** | IMT-2030 semantic communication relies on shared knowledge bases | High |
| **Goal-oriented coding** (effectiveness level) | Beyond fidelity: task success rate as optimization target | High |
| **Real channel simulation** (fading, AWGN injection) | JSCC assumes SNR but does not simulate actual channel corruption | Medium |
| **Attention-based importance** | Current variance-based heuristic; should use learned attention maps | Medium |
| **Video temporal coding** | `VideoData` marker exists but no temporal redundancy exploitation | Medium |
| **Speech/NLP-specific codecs** | Only Vec<f32> and ImageData have implementations | Medium |
| **SSIM / perceptual metrics** | Only MSE/PSNR/cosine; no perceptual quality metrics | Low |
| **Multi-user semantic broadcast** | Shared base + user-specific refinement layers | Low |

### Completeness: **78%**

The richest module architecture (6 source files, generic traits, ONNX integration, JSCC, R-D optimization). The fundamental semantic communication pipeline works end-to-end. The main gap is the "semantic" in semantic communication: knowledge graphs, goal-oriented optimization, and learned codecs beyond inference-only.

---

## Cross-Crate Integration Gaps

| Integration | Status | Gap |
|-------------|--------|-----|
| SHE <-> FL | Not connected | FL should submit training workloads to SHE scheduler |
| SHE <-> ISAC | Not connected | Sensing inference should run on SHE edge nodes |
| SHE <-> Semantic | Not connected | Semantic codec should be placed as SHE workload |
| FL <-> Semantic | Not connected | FL could train semantic codec models distributedly |
| ISAC <-> Semantic | Not connected | Sensing data could use semantic compression |
| All <-> nextgsim-core | Partial | No unified simulation tick integration |

## IMT-2030 KPI Coverage

| IMT-2030 KPI | Covered By | Status |
|--------------|-----------|--------|
| Peak data rate (200 Gbps) | -- | Not addressed (PHY layer) |
| User experienced rate (1 Gbps) | -- | Not addressed (PHY layer) |
| Latency (<1ms) | SHE tier constraints | Modeled but not simulated |
| Reliability (1-10^-7) | -- | Not addressed |
| Connection density (10^7/km2) | -- | Not addressed |
| Positioning accuracy (<10cm) | ISAC trilateration + EKF | Algorithmically complete |
| Sensing resolution | ISAC object detection, Doppler | Partial (no waveform-level) |
| AI inference latency (<5ms) | SHE edge inference | Framework in place |
| Energy efficiency (100x) | -- | Not addressed |
| Sustainability | -- | Not addressed |
| Trustworthiness | FL (SecAgg, DP) | Partial |
| Semantic communication | Semantic crate | Framework in place |

## Recommendations (Priority Order)

1. **Connect SHE <-> FL**: Submit FL training rounds as SHE workloads with tier-aware placement
2. **Add energy-aware scheduling** to SHE: critical for IMT-2030 sustainability
3. **Implement hierarchical FL**: edge -> regional -> cloud aggregation matching SHE tiers
4. **Add OFDM-radar waveform model** to ISAC: enables true joint sensing-communication
5. **Implement knowledge graph backbone** in semantic crate: shared semantic context
6. **Add channel simulation** (AWGN, fading) to JSCC pipeline for realistic evaluation
7. **Integrate EKF Doppler updates** in ISAC for seamless velocity tracking
8. **Add Byzantine-tolerant aggregation** to FL for adversarial robustness
9. **Implement TDoA solver** in ISAC (measurement type exists but unused)
10. **Add simulation-tick integration** across all crates for unified evaluation
