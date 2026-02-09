# nextgsim 6G Gap Analysis Report

**Date:** 2026-02-07
**Scope:** All 21 workspace packages in the nextgsim Rust workspace
**Reference Standards:** 3GPP Rel-18/19, ITU-R IMT-2030 Framework, 3GPP TS 23.288, TR 22.837, TR 23.700-80

---

## Executive Summary

nextgsim is a pure Rust 5G UE/gNB simulator converted from UERANSIM, currently at approximately **40-50% parity with UERANSIM** for 5G functionality and approximately **25-30% readiness for 6G** simulation capabilities.

**Strengths:**
- Solid 5G control plane foundation (NAS, NGAP, RRC, SCTP) with working registration and PDU session establishment
- Data plane operational (GTP-U tunneling, TUN interfaces, loopback mode, ping working)
- Full 5G crypto stack (Milenage, SNOW3G, ZUC, AES, ECIES)
- 6G AI-native packages created with well-structured architectures (8 packages totaling ~8,400 LoC)
- ONNX Runtime integration for ML inference with GPU acceleration support

**Critical Gaps:**
- RLC layer completely missing (blocks realistic air interface simulation)
- UE NAS MM/SM procedures ~60-70% incomplete
- UE RRC layer ~80% incomplete (handover, re-establishment, measurements)
- 6G packages have structural frameworks but lack integration with the 5G stack
- No RIS, NTN, sub-THz, zero-energy device, digital twin, or JCC support
- No ORAN fronthaul or xApp/rApp interfaces

**Estimated 6G readiness timeline:** 12-18 months to reach baseline 6G simulation capability with current architecture.

---

## Package-by-Package Analysis

### Summary Table

| Package | Category | LoC | Completion | Status | Critical 6G Gaps |
|---------|----------|-----|------------|--------|------------------|
| nextgsim-common | Core | 5,154 | 85% | Functional | Needs 6G config types (RIS, NTN, ISAC params) |
| nextgsim-crypto | Core | 4,090 | 90% | Functional | Missing post-quantum crypto (Kyber, Dilithium) |
| nextgsim-sctp | Core | 1,330 | 75% | Functional | No multi-homing, path MTU, bundling |
| nextgsim-nas | 5G Protocol | 10,252 | 60% | Partial | Missing config update, emergency, EAP-AKA' reauth |
| nextgsim-ngap | 5G Protocol | ~365K bytes | 70% | Functional | Handover basic only, no AMF load balancing |
| nextgsim-rrc | 5G Protocol | ~140K bytes | 40% | Basic | No measurement report, handover, re-establishment, CA |
| nextgsim-rls | 5G Protocol | 1,962 | 80% | Functional | No channel modeling, MIMO, beamforming sim |
| nextgsim-gtp | 5G Protocol | 1,306 | 70% | Functional | Basic QoS/extension headers only |
| nextgsim-gnb | Binary | 12,051 | 55% | Working | No NWDAF/ISAC/agent integration |
| nextgsim-ue | Binary | 18,390 | 50% | Working | No FL/semantic integration, RRC ~80% missing |
| nextgsim-cli | Binary | 1,020 | 70% | Functional | No 6G-specific commands |
| nextgsim-ai | 6G AI | 2,432 | 65% | Functional | No training, model lifecycle management |
| nextgsim-she | 6G AI | 2,726 | 60% | Functional | No actual edge node deployment, no K8s integration |
| nextgsim-nwdaf | 6G AI | 522 | 35% | Prototype | Linear extrapolation only, no ML models, no TS 23.288 services |
| nextgsim-nkef | 6G AI | 476 | 30% | Prototype | Keyword search only, no real vector embeddings/LLM |
| nextgsim-isac | 6G AI | 487 | 35% | Prototype | Simplified fusion, no real Kalman/particle filter |
| nextgsim-agent | 6G AI | 608 | 40% | Prototype | Simplified intent processing, no real multi-agent coordination |
| nextgsim-fl | 6G AI | 628 | 45% | Prototype | FedAvg only, simplified DP noise, no SecAgg crypto |
| nextgsim-semantic | 6G AI | 527 | 30% | Prototype | Mean-pooling encoder only, no neural codec, no JSCC |
| tests | Testing | - | 20% | Basic | No integration tests, no conformance testing |

---

## Detailed Package Analysis

### 1. Core Infrastructure

#### nextgsim-common (5,154 LoC) -- 85% Complete

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
- No NTN configuration (satellite orbit parameters, propagation delay models)
- No sub-THz/THz channel configuration parameters
- No ISAC sensing configuration types
- No network slicing enhancement types for 6G (inter-slice coordination)
- No digital twin configuration
- No zero-energy device parameters

#### nextgsim-crypto (4,090 LoC) -- 90% Complete

**Implemented:**
- Milenage (5G-AKA) -- 959 LoC, full implementation with test vectors
- SNOW3G (NEA1/NIA1) -- 602 LoC
- ZUC (NEA3/NIA3) -- 360 LoC, using external crate
- AES-based (NEA2/NIA2) -- 315 LoC
- Key derivation functions (KDF) -- 663 LoC
- ECIES for SUPI concealment -- 376 LoC
- NIA (integrity algorithms) -- 496 LoC
- NEA (ciphering algorithms) -- 301 LoC

**5G Gaps:**
- Additional test vector validation needed
- EAP-AKA' re-authentication not fully tested

**6G Gaps:**
- **Post-quantum cryptography (PQC):** No Kyber (ML-KEM), Dilithium (ML-DSA), or SPHINCS+ support. 3GPP SA3 is studying PQC migration per TR 33.871. This is critical for 6G security.
- **Homomorphic encryption:** No support for privacy-preserving computation needed for federated learning
- **Zero-knowledge proofs:** Not implemented (useful for privacy-preserving authentication in 6G)
- **Quantum key distribution (QKD):** No QKD integration framework
- **Physical layer security:** No PLS primitives (channel-based key generation)

#### nextgsim-sctp (1,330 LoC) -- 75% Complete

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

#### nextgsim-nas (10,252 LoC) -- 60% Complete

**Implemented:**
- NAS header encoding/decoding (plain MM, plain SM, security protected)
- Registration messages: Request, Accept, Reject, Complete (1,709 LoC)
- Authentication messages: Request, Response, Reject, Failure, Result (1,226 LoC)
- Deregistration messages (485 LoC)
- Security Mode messages (821 LoC)
- Service Request/Accept/Reject messages (767 LoC)
- Identity messages (362 LoC)
- 5GMM Status messages (146 LoC)
- PDU Session Establishment messages (725 LoC)
- PDU Session Modification messages (1,041 LoC)
- EAP framework (EAP-AKA')
- NAS security (ciphering, integrity, MAC computation) -- 2,046 LoC
- Type 1 IEs (39,045 LoC) and Type 3 IEs (32,936 LoC)
- Capture tests for real packet validation

**5G Gaps (from TODO.txt):**
- Missing NAS SM: PDU session release messages
- Missing NAS MM: Configuration update handling, emergency registration
- Optional IEs not fully implemented
- Some EAP-AKA' edge cases untested

**6G Gaps:**
- No AI/ML-assisted NAS procedure signaling extensions
- No NTN-specific NAS extensions (satellite timing advance, ephemeris data)
- No network slicing enhancement IEs for 6G
- No support for zero-energy device simplified NAS procedures

#### nextgsim-ngap (procedures: ~364K bytes across 10 files) -- 70% Complete

**Implemented (with ASN.1 PER encoding/decoding):**
- NG Setup procedure (32,363 bytes)
- Initial UE Message (28,049 bytes)
- Initial Context Setup (45,949 bytes)
- NAS Transport (UL/DL) (29,979 bytes)
- PDU Session Resource (53,058 bytes)
- UE Context Release (37,677 bytes)
- Error Indication (40,748 bytes)
- Handover procedure (73,652 bytes) -- basic implementation
- Paging (22,395 bytes) -- basic implementation

**5G Gaps:**
- Handover: Basic implementation only (no inter-AMF, no path switch)
- Paging: Needs optimization
- No AMF load balancing
- No UE TNLA binding update
- No secondary RAT data usage reporting

**6G Gaps:**
- No NGAP extensions for ISAC sensing data transfer
- No NGAP extensions for AI/ML model distribution
- No NTN-specific NGAP procedures
- No RIS control plane signaling via NGAP
- No multi-connectivity (DC/MC) enhanced procedures

#### nextgsim-rrc (procedures: ~126K bytes across 6 files) -- 40% Complete

**Implemented:**
- RRC Setup procedure (29,177 bytes)
- RRC Release procedure (16,206 bytes)
- RRC Reconfiguration (17,564 bytes)
- Security Mode Command/Complete (20,633 bytes)
- System Information (26,968 bytes)
- Information Transfer (UL/DL) (15,607 bytes)
- ASN.1 UPER codec

**In gNB RRC task (nextgsim-gnb/src/rrc/):**
- Connection management (472 LoC)
- Handover procedure (531 LoC) -- recently added
- UE context management (440 LoC)

**In UE RRC task (nextgsim-ue/src/rrc/):**
- Cell selection (815 LoC) -- recently added
- Handover (403 LoC) -- recently added
- Measurement reporting (580 LoC) -- recently added
- RRC state machine (640 LoC)
- RRC task processing (825 LoC)

**5G Gaps (critical - from TODO.txt):**
- Full ASN.1-based RRC message encoding uses placeholders in gNB
- RRC re-establishment not implemented
- Carrier aggregation not supported
- Proper measurement reporting events (A1-A6, B1-B2) partially implemented
- Cell reselection procedures incomplete

**6G Gaps:**
- No AI/ML-based RRC procedure optimization (predictive handover)
- No RIS beam management via RRC
- No NTN-specific RRC signaling (satellite cell selection, Doppler pre-compensation)
- No sub-THz beam tracking procedures
- No sidelink/D2D RRC support (V2X, ProSe)
- No dual connectivity (EN-DC, NR-DC) RRC procedures
- No conditional handover (CHO) / DAPS handover support

#### nextgsim-rls (1,962 LoC) -- 80% Complete

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
- **NTN propagation:** No satellite link budget, Doppler shift, delay modeling
- **Spectrum:** No sub-THz/THz propagation models (molecular absorption, weather effects)
- **Sensing:** No radar-like waveform simulation for ISAC
- **D2D/Sidelink:** No direct UE-to-UE communication simulation

#### nextgsim-gtp (1,306 LoC) -- 70% Complete

**Implemented:**
- GTP-U header encoding/decoding (769 LoC) per TS 29.281
- GTP-U message types (G-PDU, Echo Req/Rsp, Error Indication, End Marker)
- Extension header support (PDU Session Container, basic)
- Tunnel management with UE/PSI session tracking (486 LoC)
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

#### nextgsim-gnb (12,051 LoC) -- 55% Complete

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

#### nextgsim-ue (18,390 LoC) -- 50% Complete

**Implemented:**
- Task-based architecture: App, NAS (MM+SM), RRC, RLS, TUN tasks
- NAS MM state machine with registration, authentication, security mode
- NAS SM procedures for PDU session establishment
- Deregistration procedures (788 LoC)
- TUN interface management with async read/write split
- Timer framework (1,135 LoC) for NAS/RRC timers
- Cell selection (815 LoC)
- Handover support (403 LoC)
- Measurement reporting (580 LoC)
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

#### nextgsim-cli (1,020 LoC) -- 70% Complete

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

#### nextgsim-ai (2,432 LoC) -- 65% Complete

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

#### nextgsim-she (2,726 LoC) -- 60% Complete

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

#### nextgsim-nwdaf (522 LoC) -- 35% Complete

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

#### nextgsim-nkef (476 LoC) -- 30% Complete

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

#### nextgsim-isac (487 LoC) -- 35% Complete

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

#### nextgsim-agent (608 LoC) -- 40% Complete

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

#### nextgsim-fl (628 LoC) -- 45% Complete

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

#### nextgsim-semantic (527 LoC) -- 30% Complete

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
| Non-Terrestrial Networks (NTN) | NONE | Not started | CRITICAL |
| Sub-THz/THz communication | NONE | Not started | CRITICAL |
| Semantic communication | semantic | Basic encoder/decoder prototype | HIGH |
| Digital twin network | NONE | Not started | HIGH |
| Zero-energy devices / ambient IoT | NONE | Not started | MEDIUM |
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
- **Estimated effort:** 5,000-8,000 LoC

### Priority 2: UE NAS MM Procedures (~70% missing)
- **Impact:** Blocks advanced mobility scenarios needed for 6G (predictive handover, NTN mobility)
- **Required:** Full authentication flow, configuration update, emergency procedures, slice selection, radio capability handling
- **Estimated effort:** 3,000-5,000 LoC

### Priority 3: UE NAS SM Procedures (~60% missing)
- **Impact:** Blocks multi-PDU session scenarios, QoS-differentiated flows, network slicing
- **Required:** Full PDU session lifecycle (establish, modify, release), resource allocation
- **Estimated effort:** 2,000-3,000 LoC

### Priority 4: RRC Layer (~80% missing in UE)
- **Impact:** Blocks mobility simulation, measurement-based handover, carrier aggregation
- **Required:** Full measurement events (A1-A6, B1-B2), RRC re-establishment, cell reselection, conditional handover
- **Estimated effort:** 4,000-6,000 LoC

### Priority 5: gNB RRC Improvements
- **Impact:** Current ASN.1 encoding uses placeholders, limiting interoperability
- **Required:** Full ASN.1 UPER encoding for all RRC messages, SIB broadcasting
- **Estimated effort:** 2,000-3,000 LoC

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
4. Add NTN link simulation (satellite propagation, Doppler, delay)
5. Implement semantic communication neural codecs (ONNX-based encoder/decoder)
6. Add JSCC (Joint Source-Channel Coding) support

### Phase 4: 6G Advanced Features (Months 9-12) -- New Capabilities
1. NTN support: satellite cell selection, NAS/RRC extensions, timing advance
2. Post-quantum cryptography: Kyber/Dilithium integration
3. Digital twin network framework
4. Zero-energy device / ambient IoT simulation
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

## Recommendations

### Architecture Recommendations

1. **Unified Vector3/Position type:** The `Vector3` type is duplicated across nextgsim-nwdaf, nextgsim-isac, and nextgsim-rls. Create a shared type in nextgsim-common to avoid duplication and ensure consistency.

2. **Event bus for 6G integration:** The current task-based architecture with point-to-point message channels makes 6G integration complex. Consider adding a publish-subscribe event bus for cross-cutting concerns (measurements, analytics, sensing data).

3. **Plugin architecture for 6G modules:** Make 6G packages optional plugins that can be enabled/disabled. This keeps the 5G baseline lightweight while allowing 6G feature testing.

4. **Channel model abstraction:** Replace the simple distance-based signal model in RLS with a pluggable channel model interface. This enables adding 3GPP channel models, sub-THz models, RIS models, and NTN propagation models incrementally.

5. **Model registry service:** Create a centralized ONNX model registry that NWDAF, ISAC, semantic, and FL can share, with versioning and lifecycle management.

### Technical Recommendations

1. **Start with NWDAF integration:** The highest-value 6G feature to integrate first is NWDAF with the gNB, as it unlocks AI-driven handover, load balancing, and network optimization without requiring PHY-layer changes.

2. **Prioritize RLC before 6G PHY:** Without RLC, any 6G air interface simulation will be unrealistic. RLC is the foundation for PDCP, MAC, and ultimately ISAC waveform simulation.

3. **Use existing ONNX ecosystem:** The nextgsim-ai ONNX Runtime integration is well-designed. Leverage it to create pre-trained models for NWDAF (trajectory prediction), ISAC (positioning), and semantic (encoding/decoding) rather than building custom ML frameworks.

4. **Post-quantum crypto urgency:** 3GPP SA3 is actively studying PQC migration. Adding Kyber and Dilithium support early positions the simulator for testing PQC-enabled procedures ahead of standardization.

### Testing Recommendations

1. Add integration tests between 6G packages and the 5G stack
2. Create conformance test suites against Open5GS and Free5GC
3. Build performance benchmarks for concurrent UE scaling
4. Add property-based testing for NAS/NGAP encoding/decoding

---

## Code Metrics Summary

| Category | Packages | Total LoC | % of Codebase |
|----------|----------|-----------|---------------|
| Core Infrastructure | common, crypto, sctp | 10,574 | 17% |
| 5G Protocol Stack | nas, ngap, rrc, rls, gtp | ~18,000+ | 29% |
| Binary Applications | gnb, ue, cli | 31,461 | 50% |
| 6G AI-Native | ai, she, nwdaf, nkef, isac, agent, fl, semantic | 8,406 | 13% |
| **Total** | **19 packages** | **~62,000** | **100%** |

Note: NGAP and RRC LoC counts are estimated from file sizes as they contain large ASN.1 generated/derived code.

---

*Report generated from source code analysis of all 21 workspace members in the nextgsim Rust workspace.*
