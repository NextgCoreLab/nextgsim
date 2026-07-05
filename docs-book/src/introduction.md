# Introduction

**NextGSim** is a 5G UE and gNB simulator written in pure Rust — no C dependencies from SCTP to ASN.1 PER. It implements a 3GPP Rel-15 baseline with selected Rel-17/18 features, plus a set of 6G exploration prototypes (AI-native crates for sensing, learning, and autonomous agents) that are disabled by default.

The workspace contains **20 crates** producing **3 binaries**:

| Binary | Role |
|--------|------|
| `nr-gnb` | gNodeB simulator — NGAP, RRC, GTP-U, cell management, UE contexts, handover |
| `nr-ue` | UE simulator — NAS procedures, RRC state machine, PDU sessions, multi-UE (`--num-ues`) |
| `nr-cli` | Interactive CLI — runtime control, status queries, scenario scripting |

NextGSim pairs with the [NextGCore](../index.html) core network (AMF/SMF/UPF and friends) for end-to-end runs: full RLS → RRC → NGAP → NAS registration, 5G-AKA authentication, PDU session establishment with GTP-U user plane, and Xn handover (per the TS 23.502 procedure flows).

## Crate Layers

The workspace is layered from radio link up to AI intelligence:

**Protocol crates** — the wire stack:

- `nextgsim-nas` — 5G NAS codec (registration, authentication, security mode, PDU session, deregistration)
- `nextgsim-ngap` — NGAP ASN.1 PER encoder/decoder for gNB–AMF signaling over SCTP
- `nextgsim-rrc` — RRC setup, reconfiguration, measurement reports, handover (plus unwired NTN prototypes with bespoke encodings)
- `nextgsim-gtp` — GTP-U user-plane tunneling
- `nextgsim-sctp` — pure Rust SCTP, no libsctp
- `nextgsim-rls` — Radio Link Simulation: UE–gNB air interface over UDP
- `nextgsim-rlc` — RLC TM/UM/AM (TS 38.322) with segmentation and AM ARQ

**Core crates** — `nextgsim-gnb`, `nextgsim-ue`, `nextgsim-cli`, `nextgsim-common` (shared types: SUPI, PLMN, S-NSSAI, QoS, config), and `nextgsim-crypto` (pure Rust MILENAGE, Snow3G, ZUC, AES, HMAC-SHA256).

**6G AI crates (8)** — research prototypes: `nextgsim-ai` (sole ONNX Runtime provider for the workspace), `nextgsim-she` (3-tier compute placement), `nextgsim-nwdaf` (4-layer analytics), `nextgsim-nkef` (knowledge graph / RAG context), `nextgsim-isac` (joint radar-comm sensing, EKF fusion), `nextgsim-agent` (intent-based RL agents), `nextgsim-fl` (FedAvg/FedProx, differential privacy), and `nextgsim-semantic` (JSCC semantic codecs).

6G modules are opt-in twice over: gNB tasks are spawned only when config flags (`isac_enabled`, `agent_enabled`, `fl_enabled`, …) are set, and UE 6G modules are additionally gated behind Cargo features, so `--no-default-features` yields a lean 5G-only UE for large-scale load tests.

## Honesty Notes

Read these before quoting capability claims:

- **6G / Rel-20 crates are non-normative research prototypes.** No frozen 6G Stage-3 specification exists; these crates prototype 3GPP Stage-1 study material (e.g. TR 22.870 AI-agent use cases, TR 22.837 ISAC, TR 23.700-80 FL) and are disabled by default.
- **Matched-sim validated, not third-party certified.** End-to-end behavior is validated against the matched NextGCore simulator and against 3GPP/RFC test vectors — no external certification or interop lab has verified conformance.
- **Some ML paths are illustrative stubs.** No trained models ship: the ISAC ML positioning models and the semantic neural codec require user-supplied ONNX models (the default semantic build uses a non-neural fallback), and FL SecAgg is a pairwise-masking simulation demo, not production secure aggregation.
- **Rel-17/18 items** (RedCap, XR QoS, SNPN, MINT, UAV) are implemented, but some UE-side prototypes (ProSe, ranging/sidelink, ambient IoT) use sim-internal encodings where no wire spec is implemented.

## Quick Start

```bash
git clone https://github.com/NextgCoreLab/nextgsim.git
cd nextgsim
cargo build --release          # builds all 20 crates

./target/release/nr-gnb --config config/gnb.yaml   # terminal 1
./target/release/nr-ue  --config config/ue.yaml    # terminal 2
./target/release/nr-cli                            # terminal 3
```

A `docker-compose.yaml` is provided for a full RAN deployment ready to connect to a NextGCore AMF/UPF stack.

## Where to Go Next

- [Project landing page](../index.html) — hero overview, architecture diagrams, workflows
- [Feature matrix](../features.html) — per-feature implementation status
- [CLI & API reference](../api.html) — `nr-cli` commands and runtime APIs

NextGSim is licensed under AGPL-3.0. Contributions are welcome — see the contributing guide for setup, the zero-warning style gate (`cargo fmt` + `cargo clippy -D warnings`), 3GPP spec-reference rules, and DCO sign-off requirements.
