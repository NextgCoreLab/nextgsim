# Introduction

**NextGSim** is a 5G UE and gNB simulator, with an LTE eNB and LTE UE beside them, written in pure Rust — the default build has no C dependencies from SCTP to ASN.1 PER (the optional `kernel-sctp` feature links libsctp). It implements a 3GPP Rel-15 baseline with selected Rel-16/17/18 features, plus a set of 6G exploration prototypes (AI-native crates for sensing, learning, and autonomous agents) that are disabled by default.

The workspace has **26 crates** (plus the integration-test crate; count the
`members` list in `Cargo.toml`) producing **6 binaries**:

| Binary | Role |
|--------|------|
| `nr-gnb` | gNodeB simulator — NGAP, RRC, GTP-U, cell management, UE contexts, paging, handover |
| `nr-ue` | UE simulator — NAS procedures, RRC state machine, PDU sessions, Service Request, multi-UE (`-n/--num-of-UE`) |
| `nr-cli` | CLI — runtime control and status queries of running gNB/UE nodes |
| `nr-loadtest` | Multi-UE registration load test |
| `lte-enb` | LTE eNB — S1 Setup, UE-associated S1AP, E-UTRA cell over RLS, S1-U GTP-U |
| `lte-ue` | LTE UE — cell selection, E-UTRA RRC, EPS NAS attach, AS security, DRBs, TUN |

NextGSim pairs with the [NextGCore](https://github.com/NextgCoreLab/nextgcore) core network for end-to-end runs: full RLS → RRC → NGAP → NAS registration, 5G-AKA authentication, PDU session establishment with a GTP-U user plane, CM-IDLE paging and Service Request, de-registration and re-registration, and N2 handover between gNBs (per the TS 23.502 procedure flows; there is no Xn-C between gNBs). On the LTE side, `lte-ue` and `lte-enb` attach to NextGCore's EPC and pass a ping through the PDN.

## Crate Layers

The workspace is layered from radio link up to AI intelligence:

**Protocol crates** — the wire stack:

- `nextgsim-nas` — 5G NAS codec (registration, authentication, security mode, service request, PDU session, deregistration)
- `nextgsim-eps-nas` — EPS NAS (TS 24.301) for the LTE UE: attach, EPS AKA, NAS security
- `nextgsim-ngap` — NGAP and NRPPa ASN.1 APER codecs for gNB–AMF signaling
- `nextgsim-s1ap` — S1AP ASN.1 APER codec for eNB–MME signaling
- `nextgsim-rrc` — NR RRC UPER codec: setup, reconfiguration, measurement, handover, resume, SIB19 for NTN
- `nextgsim-lte-rrc` — E-UTRA RRC UPER codec
- `nextgsim-pdcp` — PDCP for SRBs and DRBs, AS integrity and ciphering, the SDAP header
- `nextgsim-gtp` — GTP-U user-plane tunneling
- `nextgsim-sctp` — SCTP: pure-Rust userspace backend over UDP (default) or kernel SCTP via libsctp
- `nextgsim-rls` — Radio Link Simulation: UE–gNB air interface over UDP
- `nextgsim-rlc` — RLC TM/UM/AM (TS 38.322) with segmentation and AM ARQ, plus the TS 38.351 SRAP sublayer

All ASN.1 codecs are generated at build time from the 3GPP schemas in `tools/` by the [NextgCoreLab/asn1-codegen](https://github.com/NextgCoreLab/asn1-codegen) compiler.

**Core crates** — `nextgsim-gnb`, `nextgsim-ue`, `nextgsim-enb`, `nextgsim-lte-ue`, `nextgsim-cli`, `nextgsim-common` (shared types: SUPI, PLMN, S-NSSAI, QoS, config), and `nextgsim-crypto` (pure Rust MILENAGE, TUAK, Snow3G, ZUC, AES, HMAC-SHA256, ECIES).

**6G AI crates (8)** — research prototypes: `nextgsim-ai` (sole ONNX Runtime provider for the workspace), `nextgsim-she` (3-tier compute placement), `nextgsim-nwdaf` (4-layer analytics), `nextgsim-nkef` (knowledge graph / RAG context), `nextgsim-isac` (joint radar-comm sensing, EKF fusion), `nextgsim-agent` (intent-based RL agents), `nextgsim-fl` (FedAvg/FedProx, differential privacy), and `nextgsim-semantic` (JSCC semantic codecs).

6G modules are opt-in twice over: they are compiled only with a Cargo feature (`6g` for the gNB, the crate-named features such as `nextgsim-isac` for the UE), and their tasks are spawned only when a config flag (`isac_enabled`, `agent_enabled`, `federated_learning_enabled`, …) is set. The default build is a lean 5G-only UE and gNB.

## Honesty Notes

Read these before quoting capability claims:

- **6G / Rel-20 crates are non-normative research prototypes.** No frozen 6G Stage-3 specification exists; these crates prototype 3GPP Stage-1 study material (e.g. TR 22.870 AI-agent use cases, TR 22.837 ISAC, TR 23.700-80 FL) and are disabled by default.
- **Matched-sim validated, not third-party certified.** End-to-end behavior is validated against the matched NextGCore simulator and against 3GPP/RFC test vectors — no external certification or interop lab has verified conformance.
- **Some ML paths are illustrative stubs.** No trained models ship: the ISAC ML positioning models and the semantic neural codec require user-supplied ONNX models (the default semantic build uses a non-neural fallback), and FL SecAgg is a pairwise-masking simulation demo, not production secure aggregation.
- **Rel-17/18 items** (RedCap, XR QoS, SNPN, MINT, UAV) are implemented. The PC5 procedures behind the `sidelink` feature are real PC5-S messages over an in-process medium (there is no PC5 radio), and a few prototypes (ambient IoT, the UAV tracking report) use sim-internal encodings where no wire spec exists.

## Quick Start

```bash
git clone https://github.com/NextgCoreLab/nextgsim.git
cd nextgsim
cargo build --release          # builds the whole workspace

./target/release/nr-gnb --config config/gnb.yaml        # terminal 1
sudo ./target/release/nr-ue --config config/ue.yaml     # terminal 2
./target/release/nr-cli ue0 -e status                   # terminal 3
```

`docker/docker-compose.yml` runs the gNB and UE as containers on an existing NextGCore Docker network.

## Where to Go Next

- [Project landing page](../index.html) — hero overview, architecture diagrams, workflows
- [Feature matrix](../features.html) — per-feature implementation status
- [CLI & API reference](../api.html) — `nr-cli` commands and runtime APIs
- [nr-cli Reference](cli-reference.md) — every binary's flags and every `nr-cli` command

NextGSim is licensed under AGPL-3.0. Contributions are welcome — see the contributing guide for setup, the style gate (`cargo fmt` + `cargo clippy`), 3GPP spec-reference rules, and sign-off requirements.
