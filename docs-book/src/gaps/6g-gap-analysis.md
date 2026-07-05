# nextgsim 6G Gap Analysis Report

**Date:** 2026-02-07
**Scope:** All 21 workspace packages in the nextgsim Rust workspace
**Reference Standards:** 3GPP Rel-18/19, ITU-R IMT-2030 Framework, 3GPP TS 23.288, TR 22.837, TR 23.700-80

---

## Package-by-Package Analysis

### Summary Table

| Package | Category | Completion | Status | Critical 6G Gaps |
|---------|----------|------------|--------|------------------|
| nextgsim-common | Core | 85% | Functional | Needs 6G config types (RIS, NTN, ISAC params) |
| nextgsim-crypto | Core | 90% | Functional | Missing post-quantum crypto (Kyber, Dilithium) |
| nextgsim-sctp | Core | 75% | Functional | No multi-homing, path MTU, bundling |
| nextgsim-nas | 5G Protocol | 60% | Partial | Missing config update, emergency, EAP-AKA' reauth |
| nextgsim-ngap | 5G Protocol | 70% | Functional | Handover basic only, no AMF load balancing |
| nextgsim-rrc | 5G Protocol | 40% | Basic | No measurement report, handover, re-establishment, CA |
| nextgsim-rls | 5G Protocol | 80% | Functional | No channel modeling, MIMO, beamforming sim |
| nextgsim-gtp | 5G Protocol | 70% | Functional | Basic QoS/extension headers only |
| nextgsim-gnb | Binary | 55% | Working | No NWDAF/ISAC/agent integration |
| nextgsim-ue | Binary | 50% | Working | No FL/semantic integration, RRC ~80% missing |
| nextgsim-cli | Binary | 70% | Functional | No 6G-specific commands |
| nextgsim-ai | 6G AI | 65% | Functional | No training, model lifecycle management |
| nextgsim-she | 6G AI | 60% | Functional | No actual edge node deployment, no K8s integration |
| nextgsim-nwdaf | 6G AI | 35% | Prototype | Linear extrapolation only, no ML models, no TS 23.288 services |
| nextgsim-nkef | 6G AI | 30% | Prototype | Keyword search only, no real vector embeddings/LLM |
| nextgsim-isac | 6G AI | 35% | Prototype | Simplified fusion, no real Kalman/particle filter |
| nextgsim-agent | 6G AI | 40% | Prototype | Simplified intent processing, no real multi-agent coordination |
| nextgsim-fl | 6G AI | 45% | Prototype | FedAvg only, simplified DP noise, no SecAgg crypto |
| nextgsim-semantic | 6G AI | 30% | Prototype | Mean-pooling encoder only, no neural codec, no JSCC |
| tests | Testing | 20% | Basic | No integration tests, no conformance testing |

---

## Detailed Package Analysis

### 1. Core Infrastructure

#### nextgsim-common -- 85% Complete

**Implemented:**
- BitBuffer, BitString, OctetString, OctetView (binary data handling)
- GnbConfig, UeConfig, AmfConfig (configuration structures)
- CLI server with process table and command routing
- UDP transport layer
- Structured logging with hex dump and protocol-specific formatters
- Common types (PLMN, TAI, S-NSSAI, etc.)

**5G Gaps:**
- No RLC-related configuration types
- Some optional NAS IEs not represented in types

**6G Gaps:**
- No RIS configuration types (panel geometry, reflection coefficients, phase control)
- NTN configuration prototypes exist (gNB `ntn_config` YAML block, orbit/delay models in the rrc crate) but are logging-only/unwired
- No sub-THz/THz channel configuration parameters
- No ISAC sensing configuration types
- No network slicing enhancement types for 6G (inter-slice coordination)
- No digital twin configuration
- No zero-energy device parameters

#### nextgsim-crypto -- 90% Complete

**Implemented:**
- Milenage (5G-AKA), full implementation with test vectors
- SNOW3G (NEA1/NIA1)
- ZUC (NEA3/NIA3), using external crate
- AES-based (NEA2/NIA2)
- Key derivation functions (KDF)
- ECIES for SUPI concealment
- NIA (integrity algorithms)
- NEA (ciphering algorithms)

**5G Gaps:**
- Additional test vector validation needed
- EAP-AKA' re-authentication not fully tested

**6G Gaps:**
- **Post-quantum cryptography (PQC):** No Kyber (ML-KEM), Dilithium (ML-DSA), or SPHINCS+ support. 3GPP SA3 is studying PQC migration per TR 33.871. This is critical for 6G security.
- **Homomorphic encryption:** No support for privacy-preserving computation needed for federated learning
- **Zero-knowledge proofs:** Not implemented (useful for privacy-preserving authentication in 6G)
- **Quantum key distribution (QKD):** No QKD integration framework
- **Physical layer security:** No PLS primitives (channel-based key generation)

#### nextgsim-sctp -- 75% Complete

**Implemented:**
- SCTP association management (client mode)
- SCTP server mode (for accepting connections)
- Async tokio-based interface over sctp-proto Sans-IO
- Multi-stream support for NGAP (PPID 60)
- Graceful shutdown

**5G Gaps:**
- No multi-homing support
- No path MTU discovery
- No bundling optimization

**6G Gaps:**
- No QUIC transport alternative (being studied for next-gen signaling)
- No transport layer support for NTN high-latency links

---

### 2. 5G Protocol Stack

#### nextgsim-nas -- 60% Complete

**Implemented:**
- NAS header encoding/decoding (plain MM, plain SM, security protected)
- Registration messages: Request, Accept, Reject, Complete
- Authentication messages: Request, Response, Reject, Failure, Result
- Deregistration messages
- Security Mode messages
- Service Request/Accept/Reject messages
- Identity messages
- 5GMM Status messages
- PDU Session Establishment messages
- PDU Session Modification messages
- EAP framework (EAP-AKA')
- NAS security (ciphering, integrity, MAC computation)
- Type 1 IEs and Type 3 IEs
- Capture tests for real packet validation

**5G Gaps (from TODO.txt):**
- Missing NAS SM: PDU session release messages
- Missing NAS MM: Configuration update handling, emergency registration
- Optional IEs not fully implemented
- Some EAP-AKA' edge cases untested

**6G Gaps:**
- No AI/ML-assisted NAS procedure signaling extensions
- NTN NAS data models (satellite timing advance, ephemeris) exist as bespoke prototypes — not wire-conformant, no sender wired into live UE paths
- No network slicing enhancement IEs for 6G
- No support for zero-energy device simplified NAS procedures

#### nextgsim-ngap (procedures: ~364K bytes across 10 files) -- 70% Complete

**Implemented (with ASN.1 PER encoding/decoding):**
- NG Setup procedure
- Initial UE Message
- Initial Context Setup
- NAS Transport (UL/DL)
- PDU Session Resource
- UE Context Release
- Error Indication
- Handover procedure -- basic implementation
- Paging -- basic implementation

**5G Gaps:**
- Handover: Basic implementation only (no inter-AMF, no path switch)
- Paging: Needs optimization
- No AMF load balancing
- No UE TNLA binding update
- No secondary RAT data usage reporting

**6G Gaps:**
- No NGAP extensions for ISAC sensing data transfer
- No NGAP extensions for AI/ML model distribution
- NTN NGAP support module exists (off-wire, bespoke) — no wire-conformant NTN NGAP procedures
- No RIS control plane signaling via NGAP
- No multi-connectivity (DC/MC) enhanced procedures

#### nextgsim-rrc (procedures: ~126K bytes across 6 files) -- 40% Complete

**Implemented:**
- RRC Setup procedure
- RRC Release procedure
- RRC Reconfiguration
- Security Mode Command/Complete
- System Information
- Information Transfer (UL/DL)
- ASN.1 UPER codec

**In gNB RRC task (nextgsim-gnb/src/rrc/):**
- Connection management
- Handover procedure -- recently added
- UE context management

**In UE RRC task (nextgsim-ue/src/rrc/):**
- Cell selection -- recently added
- Handover -- recently added
- Measurement reporting -- recently added
- RRC state machine
- RRC task processing

**5G Gaps (critical - from TODO.txt):**
- Full ASN.1-based RRC message encoding uses placeholders in gNB
- RRC re-establishment not implemented
- Carrier aggregation not supported
- Proper measurement reporting events (A1-A6, B1-B2) partially implemented
- Cell reselection procedures incomplete

**6G Gaps:**
- No AI/ML-based RRC procedure optimization (predictive handover)
- No RIS beam management via RRC
- NTN RRC prototypes exist (timing advance, link sim, constellation, ISL handover in nextgsim-rrc) but use bespoke encodings and are not wired into live signaling; no satellite cell selection or Doppler pre-compensation
- No sub-THz beam tracking procedures
- No wire-conformant sidelink/D2D RRC support (a UE-side ProSe/ranging prototype now exists, sim-internal only)
- No dual connectivity (EN-DC, NR-DC) RRC procedures
- No conditional handover (CHO) / DAPS handover support

#### nextgsim-rls -- 80% Complete

**Implemented:**
- RLS protocol messages: Heartbeat, HeartbeatAck, PduTransmission, PduTransmissionAck
- Cell search (UE side): cell discovery with signal strength
- Cell tracking (gNB side): UE tracking with timeout management
- RRC and user plane data transport with PDU management
- Binary codec with full encode/decode
- Position-based signal strength simulation (Vector3)

**5G Gaps:**
- Simplified signal propagation model (distance-only)
- No fading/shadowing simulation

**6G Gaps:**
- **Channel modeling:** No 3GPP TR 38.901 channel models, no sub-THz channel models
- **MIMO/Beamforming:** No massive MIMO simulation, no beam management
- **RIS integration:** No reflective surface path modeling
- **NTN propagation:** Prototype link simulation exists (delay/Doppler/link models in nextgsim-rrc) but is not integrated with the RLS channel path
- **Spectrum:** No sub-THz/THz propagation models (molecular absorption, weather effects)
- **Sensing:** No radar-like waveform simulation for ISAC
- **D2D/Sidelink:** No direct UE-to-UE communication simulation

#### nextgsim-gtp -- 70% Complete

**Implemented:**
- GTP-U header encoding/decoding per TS 29.281
- GTP-U message types (G-PDU, Echo Req/Rsp, Error Indication, End Marker)
- Extension header support (PDU Session Container, basic)
- Tunnel management with UE/PSI session tracking
- Sequence number handling

**5G Gaps:**
- Extension headers: Basic support only
- QoS flow handling: Simplified
- End marker handling: Basic

**6G Gaps:**
- No QoS flow identifier (QFI) based traffic differentiation for 6G slicing
- No GTP-U extensions for URLLC/time-sensitive networking
- No compute-aware forwarding extensions (JCC)
- No semantic-aware forwarding
- No network coding support

---

### 3. Binary Applications

#### nextgsim-gnb -- 55% Complete

**Implemented:**
- Task-based actor architecture: App, NGAP, RRC, GTP, RLS, SCTP tasks
- NG Setup with AMF
- Initial UE message handling
- NAS transport (UL/DL) relay
- PDU session resource setup
- GTP-U tunneling with loopback mode
- RLS cell tracking and UE management
- CLI server for runtime commands
- YAML configuration loading
- Status reporting and monitoring

**5G Gaps:**
- Full ASN.1-based RRC encoding (uses placeholders)
- AMF load balancing
- Multiple AMF support
- Broadcast/system information handling incomplete

**6G Integration Gaps:**
- No NWDAF task integration (analytics data collection/consumption)
- No ISAC task integration (sensing data processing)
- No agent framework integration (autonomous decision-making)
- No SHE task integration (edge compute orchestration)
- No RIS control integration
- No NTN-aware scheduling
- No xApp/rApp interface (ORAN)

#### nextgsim-ue -- 50% Complete

**Implemented:**
- Task-based architecture: App, NAS (MM+SM), RRC, RLS, TUN tasks
- NAS MM state machine with registration, authentication, security mode
- NAS SM procedures for PDU session establishment
- Deregistration procedures
- TUN interface management with async read/write split
- Timer framework for NAS/RRC timers
- Cell selection
- Handover support
- Measurement reporting
- CLI handler and status reporting

**5G Gaps (from TODO.txt -- critical):**
- NAS MM procedures ~70% missing: access, auth full flow, config, ecall, radio capability, slice selection
- NAS SM procedures ~60% missing: full PDU establishment flow, release, resource allocation
- RRC ~80% missing: re-establishment, full measurement events, cell reselection
- Multiple PDU sessions: Basic support only

**6G Integration Gaps:**
- No FL participant integration (local training, model update submission)
- No semantic communication integration (task-oriented encoding)
- No ISAC measurement reporting
- No AI/ML-assisted mobility (predictive handover consumption from NWDAF)
- No sidelink/D2D support
- No NTN-aware procedures
- No zero-energy device emulation mode

#### nextgsim-cli -- 70% Complete

**Implemented:**
- CLI client with process table discovery
- Command routing to gNB/UE instances
- Protocol for CLI message exchange
- Interactive command interface

**6G Gaps:**
- No commands for 6G-specific operations (ISAC query, NWDAF analytics, FL status)
- No AI agent interaction commands
- No RIS configuration commands
- No network slice management commands

---

### 4. 6G AI-Native Packages

#### nextgsim-ai -- 65% Complete

**Implemented:**
- ONNX Runtime inference engine with full lifecycle (load, infer, batch_infer, warmup)
- Multi-execution-provider support (CPU, CUDA, CoreML, DirectML, TensorRT)
- TensorData abstraction (Float32, Float16, Int64, Int32, Int8, UInt8)
- Tensor shape management with dynamic dimensions
- Model metadata extraction from ONNX files
- Inference metrics (latency tracking, throughput, error counting)
- Configuration system (InferenceConfig, AiConfig)
- Error types (ModelError, InferenceError)
- GPU acceleration support

**Gaps:**
- **Training:** No on-device training or fine-tuning support (inference only)
- **Model lifecycle:** No model versioning, A/B testing, rollback
- **Model compression:** No quantization, pruning, or knowledge distillation utilities
- **Hardware optimization:** No NPU/FPGA acceleration profiles
- **Streaming inference:** No support for continuous/streaming model inputs
- **Model registry:** No centralized model management/discovery

#### nextgsim-she -- 60% Complete

**Implemented:**
- Three-tier compute model: Local Edge (<10ms), Regional Edge (<20ms), Core Cloud
- Workload scheduler with multiple policies (ClosestToEdge, MostAvailable, LeastUtilized)
- Resource capacity tracking (CPU, memory, GPU, FLOPS)
- Compute node management with capability tracking
- Workload lifecycle management (Pending, Placed, Running, Complete, Failed, Migrating)
- SHE task with inference engine integration (loads ONNX models per tier)
- Message-passing architecture (SheMessage/SheResponse)
- Placement decision engine with reason tracking

**Gaps:**
- **Actual deployment:** No real container/VM orchestration (Kubernetes, Docker)
- **Migration:** Workload migration logic is framework-only, no live migration
- **Autoscaling:** No dynamic scaling based on load
- **Multi-access edge computing (MEC):** No ETSI MEC API compliance
- **Latency simulation:** Tier latency is configured, not dynamically measured
- **Network-compute joint optimization:** No JCC (Joint Communication and Computing)
- **Digital twin integration:** No digital twin state synchronization

#### nextgsim-nwdaf -- 35% Complete

**Implemented:**
- UE measurement data structures (RSRP, RSRQ, SINR, position, velocity)
- Cell load data structures
- Measurement history management with circular buffer
- Trajectory prediction using linear extrapolation (comment: "Production would use LSTM/Transformer model via ONNX")
- Handover recommendation with signal-based and predicted-mobility reasons
- Cell load recording and retrieval
- Automation action types (handover params, cell power, load balancing)
- Message/response types for async communication

**Gaps:**
- **ML Models:** Uses linear extrapolation instead of actual ML models (LSTM, Transformer). Comment in code acknowledges this.
- **TS 23.288 Services:** Missing NWDAF service operations:
  - Nnwdaf_AnalyticsSubscription (subscribe to analytics)
  - Nnwdaf_AnalyticsInfo (on-demand analytics query)
  - Nnwdaf_MLModelProvision (ML model distribution)
  - Nnwdaf_DataManagement (data collection coordination)
- **Analytics IDs:** No support for standardized analytics IDs (UE mobility, NF load, service experience, abnormal behavior, etc.)
- **MTLF/AnLF split:** No separation between Model Training Logical Function and Analytics Logical Function
- **Data collection:** No integration with gNB/UE for real measurement collection
- **Closed-loop automation:** Layer 4 automation is defined as enum but not implemented
- **Anomaly detection:** Listed but no implementation
- **Federated analytics:** No support for cross-domain analytics

#### nextgsim-nkef -- 30% Complete

**Implemented:**
- Knowledge graph with entity/relationship management
- Entity types: Gnb, Ue, Cell, Amf, Upf, Slice, PduSession, Service
- Type-indexed entity lookup
- Keyword-based search (comment: "production would use vector similarity")
- RAG context generation from knowledge graph
- Embedding field on entities (but no actual embedding generation)
- NKEF manager with query and context retrieval

**Gaps:**
- **Vector embeddings:** Field exists but no embedding model integration (no sentence transformers)
- **Semantic search:** Uses keyword matching, not vector similarity search
- **LLM integration:** No actual LLM connection for RAG
- **Ontology:** No formal network ontology (OWL/RDF)
- **Real-time updates:** No event-driven knowledge graph updates from network events
- **3GPP TS 23.288 integration:** Not connected to NWDAF for analytics-informed knowledge
- **Intent translation:** No NLP-based intent parsing from natural language
- **Temporal reasoning:** No time-series aware knowledge graph

#### nextgsim-isac -- 35% Complete

**Implemented:**
- Sensing measurement types: ToA, TDoA, AoA, ZoA, RSS, Doppler, RTT
- Sensing data aggregation from multiple cells
- Data source registration with anchor positions
- Simplified Kalman-like position tracking (update with gain calculation)
- Position prediction based on velocity
- Position fusion using weighted centroid (comment: "production would use proper trilateration")
- Track lifecycle management (create, update, cleanup stale)
- ISAC manager with anchor registration and tracking state

**Gaps:**
- **3GPP TR 22.837 compliance:** Missing sensing use cases:
  - Object detection and tracking beyond positioning
  - Gesture/activity recognition
  - Environmental sensing (weather, rain)
  - Intrusion detection
  - Automotive sensing (V2X)
- **Waveform design:** No joint communication-sensing waveform models (OFDM radar, FMCW)
- **Beamforming integration:** No beam-based sensing
- **Proper algorithms:** Uses centroid-based fusion instead of trilateration/multilateration
- **Extended Kalman Filter:** Simplified linear Kalman, no EKF/UKF for non-linear models
- **Particle filter:** Not implemented (mentioned in doc but absent)
- **Multi-sensor fusion:** Basic aggregation only, no Bayesian/Dempster-Shafer fusion
- **Clutter/interference modeling:** No realistic sensing environment simulation
- **Sensing-communication tradeoff:** No resource sharing optimization

#### nextgsim-agent -- 40% Complete

**Implemented:**
- Agent type classification: Mobility, Resource, QoS, Security, Slicing, Custom
- Agent capabilities system: read_state, modify_config, trigger_actions
- OAuth 2.0-style token authentication with expiration
- Intent framework: Query, OptimizeResources, TriggerHandover, AdjustQos, CreateSlice, ModifySlice
- Intent priority-based processing (1-10 scale)
- Agent coordinator for registration, token management, intent routing
- Capability-based access control (intent validation against agent capabilities)
- Resource limits (requests/second, concurrent ops, data scope)
- Heartbeat-based agent liveness tracking

**Gaps:**
- **Intent processing:** All intents return success without actual execution (placeholder)
- **Multi-agent coordination:** No conflict resolution between competing intents
- **Reinforcement learning:** No RL-based agent decision making
- **LLM integration:** No LLM-based intent understanding or generation
- **A2A protocol:** No agent-to-agent communication protocol
- **MCP integration:** No Model Context Protocol for tool use
- **Safety constraints:** No guardrails or safety bounds on agent actions
- **Explainability:** No decision explanation or audit trail
- **Hierarchical agents:** No multi-level agent hierarchy (cell-level, region-level, network-level)
- **Real-time constraints:** No latency-aware agent scheduling

#### nextgsim-fl -- 45% Complete

**Implemented:**
- Federated Averaging (FedAvg) aggregation with weighted averaging
- Training round management (start, collect, aggregate lifecycle)
- Participant registration and tracking
- Differential privacy: gradient clipping and noise injection
- Model versioning (version tracking across rounds)
- Round status tracking (WaitingForParticipants, Collecting, Aggregating, Complete, Failed)
- Timeout-based round management
- Algorithm enum: FedAvg, FedProx, SecAgg (though FedProx and SecAgg fall through to FedAvg)

**Gaps:**
- **FedProx:** Listed but implementation falls through to FedAvg
- **SecAgg:** Listed but implementation falls through to FedAvg (no actual Secure Aggregation crypto)
- **DP noise:** Uses simplified deterministic noise instead of proper Gaussian sampling
- **Asynchronous FL:** Only synchronous rounds supported
- **Heterogeneous models:** No support for different model architectures per participant
- **Client selection:** No intelligent participant selection (contribution-based, resource-aware)
- **Communication efficiency:** No gradient compression, sparsification, or quantization
- **Model poisoning defense:** No Byzantine fault tolerance
- **Split learning:** Not supported
- **Over-the-air aggregation:** No wireless channel-aware aggregation

#### nextgsim-semantic -- 30% Complete

**Implemented:**
- Semantic feature representation with task ID, features, importance weights, compression ratio
- Feature importance-based pruning (top-k feature selection)
- Semantic task types: ImageClassification, ObjectDetection, SpeechRecognition, TextUnderstanding, SensorFusion, VideoAnalytics
- Channel quality model with SNR/bandwidth/PER and quality categories
- Channel-adaptive compression (recommended compression ratio from channel quality)
- Semantic encoder: mean-pooling with variance-based importance (comment: "would use ONNX model in production")
- Semantic decoder: nearest-neighbor upsampling
- Task-aware decoding interface

**Gaps:**
- **Neural codecs:** Uses mean-pooling instead of actual learned codecs (autoencoder, VAE)
- **JSCC:** No joint source-channel coding (just separate compression and transmission)
- **End-to-end training:** No trainable encoder/decoder pipeline
- **Multi-modal:** Limited to 1D data vectors, no image/video/audio specific pipelines
- **Semantic similarity metrics:** No SSIM, LPIPS, or task-specific metrics
- **Knowledge base integration:** No shared knowledge base between encoder/decoder
- **Generative AI integration:** No diffusion model or GAN-based reconstruction
- **Rate-distortion optimization:** No learned rate control
- **Cross-layer optimization:** No PHY layer integration for semantic transmission

---

## 6G Feature Gap Matrix

| 6G Feature (ITU-R IMT-2030) | Package(s) | Current State | Gap Severity |
|------------------------------|------------|---------------|--------------|
| AI/ML native air interface | ai, nwdaf, she | Inference framework exists; no air interface integration | HIGH |
| Integrated Sensing and Communication (ISAC) | isac | Basic tracking/fusion prototype | HIGH |
| Reconfigurable Intelligent Surface (RIS) | NONE | Not started | CRITICAL |
| Non-Terrestrial Networks (NTN) | rrc, ngap, gnb | Prototype modules present (timing advance, ephemeris, link sim, constellation, ISL handover, NAS/NGAP data models) — bespoke encodings, not wire-conformant, largely unwired from live UE/gNB paths | HIGH |
| Sub-THz/THz communication | NONE | Not started | CRITICAL |
| Semantic communication | semantic | Basic encoder/decoder prototype | HIGH |
| Digital twin network | NONE | Not started | HIGH |
| Zero-energy devices / ambient IoT | ue (ambient_iot) | Prototype exists (energy-harvesting model + fleet simulation), sim-internal only | MEDIUM |
| Joint Communication and Computing (JCC) | NONE | Not started | HIGH |
| Enhanced network slicing (6G) | agent (SliceCreate/Modify intents) | Intent types defined, no implementation | HIGH |
| Federated learning | fl | FedAvg prototype with basic DP | MEDIUM |
| Network data analytics (NWDAF) | nwdaf | Linear prediction, no ML models | HIGH |
| Knowledge exposure (NKEF/LLM) | nkef | Knowledge graph prototype, no LLM | MEDIUM |
| AI agent framework | agent | OAuth + intent framework, no real execution | MEDIUM |
| Post-quantum cryptography | NONE | Not started | HIGH |
| ORAN / xApp/rApp | NONE | Not started | HIGH |

---

## Missing 5G Features That Block 6G

These 5G gaps must be resolved before meaningful 6G simulation is possible:

### Priority 1: RLC Layer (CRITICAL BLOCKER)
- **Status:** Completely missing
- **Impact:** Blocks realistic air interface simulation. Without RLC, there is no proper PDCP/MAC simulation, which means ISAC waveforms, semantic transmission, and AI/ML air interface features cannot be realistically tested.
- **Required:** RLC AM, UM, TM entities; encoder/decoder; ARQ procedures

### Priority 2: UE NAS MM Procedures (~70% missing)
- **Impact:** Blocks advanced mobility scenarios needed for 6G (predictive handover, NTN mobility)
- **Required:** Full authentication flow, configuration update, emergency procedures, slice selection, radio capability handling

### Priority 3: UE NAS SM Procedures (~60% missing)
- **Impact:** Blocks multi-PDU session scenarios, QoS-differentiated flows, network slicing
- **Required:** Full PDU session lifecycle (establish, modify, release), resource allocation

### Priority 4: RRC Layer (~80% missing in UE)
- **Impact:** Blocks mobility simulation, measurement-based handover, carrier aggregation
- **Required:** Full measurement events (A1-A6, B1-B2), RRC re-establishment, cell reselection, conditional handover

### Priority 5: gNB RRC Improvements
- **Impact:** Current ASN.1 encoding uses placeholders, limiting interoperability
- **Required:** Full ASN.1 UPER encoding for all RRC messages, SIB broadcasting

---

## Prioritized 6G Implementation Roadmap

### Phase 1: Foundation (Months 1-3) -- Complete 5G Baseline
1. Implement RLC layer (AM/UM/TM)
2. Complete UE NAS MM procedures (auth, config update, slice selection)
3. Complete UE NAS SM procedures (full PDU session lifecycle)
4. Improve RRC layer (measurement reporting, re-establishment)
5. Integrate NWDAF with gNB/UE for real measurement collection

### Phase 2: AI/ML Integration (Months 3-6) -- Connect 6G Packages
1. Integrate NWDAF task into gNB for analytics collection and consumption
2. Replace linear prediction in NWDAF with ONNX model inference (via nextgsim-ai)
3. Integrate ISAC with RLS for sensing simulation alongside communication
4. Connect agent framework to gNB for autonomous decision-making
5. Implement FL participant in UE (local training, model update)
6. Add NWDAF TS 23.288 service APIs (AnalyticsSubscription, AnalyticsInfo)

### Phase 3: 6G Air Interface (Months 6-9) -- Channel & PHY Simulation
1. Implement channel models in RLS (3GPP TR 38.901, sub-THz extensions)
2. Add massive MIMO / beamforming simulation
3. Implement RIS channel modeling and control
4. Wire the existing NTN link-simulation prototypes (satellite propagation, Doppler, delay) into live paths and make encodings wire-conformant
5. Implement semantic communication neural codecs (ONNX-based encoder/decoder)
6. Add JSCC (Joint Source-Channel Coding) support

### Phase 4: 6G Advanced Features (Months 9-12) -- New Capabilities
1. NTN support: prototype modules exist (timing advance, ephemeris, constellation, ISL handover) — remaining work is satellite cell selection, wire-conformant NAS/RRC encodings, and wiring into live signaling
2. Post-quantum cryptography: Kyber/Dilithium integration
3. Digital twin network framework
4. Zero-energy device / ambient IoT simulation (UE-side prototype exists: energy harvesting + fleet sim; extend beyond sim-internal)
5. JCC (Joint Communication and Computing) framework
6. Enhanced network slicing for 6G (inter-slice coordination, slice SLA)

### Phase 5: Production Hardening (Months 12-18) -- Quality & Scale
1. Conformance testing with Open5GS and Free5GC
2. Performance benchmarking (100+ UEs, latency profiling)
3. ORAN xApp/rApp interface for RAN intelligent controller
4. Kubernetes deployment with Helm charts
5. Prometheus metrics and OpenTelemetry tracing
6. Real FL with SecAgg and proper DP
7. LLM integration for NKEF and agent framework

---

## Open work

Forward-looking work items from this analysis are tracked as
[GitHub issues](https://github.com/NextgCoreLab/nextgsim/issues) rather than as a
roadmap here, so the code and the tracker stay the source of truth. Each issue
records the verified current state, a bounded scope, and acceptance criteria.

---

*Report generated from source code analysis of all 21 workspace members in the nextgsim Rust workspace.*
