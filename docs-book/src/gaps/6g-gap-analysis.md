# nextgsim — 6G Readiness and Missing-Component Register

**Standards baseline:** 3GPP Rel-18/19; ITU-R IMT-2030. Relevant study items include TS 23.288 (NWDAF), TR 22.837 (integrated sensing) and TR 33.871 (post-quantum migration).

**Status.** This document reflects the current codebase. nextgsim is a UE + gNB 5G/6G simulator of **20 crates** producing **3 binaries** (`nextgsim-gnb`, `nextgsim-ue`, `nr-cli`). Behaviour is validated by unit tests and a matched-simulator end-to-end run in which the simulator drives its own gNB/UE against the NextGCore 5G core over N2 (NGAP/SCTP) and N3 (GTP-U), with the user-plane data path verified. The 6G components below are **non-normative research prototypes**; several are compiled into the binaries but are disabled by default. Each missing or partial component is itemized with a stable task ID and, where the work is bounded, a tracked implementation issue.

---

## 1. Implemented components

**Core infrastructure:** `nextgsim-common` (types, config, CLI, transport), `nextgsim-crypto` (Milenage/5G-AKA, SNOW3G, ZUC, AES, KDF, ECIES), `nextgsim-sctp` (N2 transport), `nextgsim-rlc` (RLC TM/UM; AM data path present).

**5G protocol stack:** `nextgsim-nas` (5GMM + 5GSM), `nextgsim-ngap` (NGAP with ASN.1 PER), `nextgsim-rrc` (RRC with ASN.1 UPER), `nextgsim-rls` (radio-link simulation), `nextgsim-gtp` (GTP-U).

**Binaries:** `nextgsim-gnb`, `nextgsim-ue`, `nextgsim-cli` (`nr-cli`, plus a load-test binary).

**6G AI-native crates:** `nextgsim-ai` (ONNX Runtime inference), `nextgsim-she` (edge compute placement), `nextgsim-nwdaf`, `nextgsim-nkef`, `nextgsim-isac`, `nextgsim-agent`, `nextgsim-fl`, `nextgsim-semantic`. These are prototypes; see the register below for what each is missing.

Working today: 5G control plane (registration, authentication, NAS/AS security), PDU-session establishment, and the user-plane data path (GTP-U tunnelling and the UE TUN interface) against the matched core.

---

## 2. Missing-component register (6G evolution)

Every row has a stable task ID. **Missing** = not started; **Partial** = a baseline or prototype exists and the linked issue scopes the remainder; **Research — not yet scoped** = dependent on 3GPP/industry work that is not yet frozen, so no bounded implementation task is filed yet.

| Task ID | Component | Missing / partial capability | Reference | Status | Tracking |
|---|---|---|---|---|---|
| NGS-6G-01 | AI/ML network analytics | NWDAF is not fed real gNB data and uses linear extrapolation instead of model inference | TS 23.288 | Missing | [#18](https://github.com/NextgCoreLab/nextgsim/issues/18) |
| NGS-6G-02 | Shared model registry | reusable ONNX model registry (load-by-id, versioning, warmup) across the AI crates | — | Missing | [#17](https://github.com/NextgCoreLab/nextgsim/issues/17) |
| NGS-6G-03 | Cross-cutting event bus | in-process pub/sub for measurements/analytics/sensing, to decouple 6G integration | — | Missing | [#16](https://github.com/NextgCoreLab/nextgsim/issues/16) |
| NGS-6G-04 | Channel modelling | pluggable channel-model abstraction in `nextgsim-rls` (currently distance-only) | TR 38.901 | Missing | [#25](https://github.com/NextgCoreLab/nextgsim/issues/25) |
| NGS-6G-05 | Integrated sensing (ISAC) | positioning uses centroid fusion + linear Kalman; needs multilateration + EKF | TR 22.837 | Partial | [#27](https://github.com/NextgCoreLab/nextgsim/issues/27) |
| NGS-6G-06 | Semantic communication | codec is a mean-pooling stub; needs an ONNX neural-codec path | 6G research | Partial | [#28](https://github.com/NextgCoreLab/nextgsim/issues/28) |
| NGS-6G-07 | Knowledge exposure (NKEF) | keyword search only; embeddings never generated — needs real vector similarity | 6G research | Partial | [#29](https://github.com/NextgCoreLab/nextgsim/issues/29) |
| NGS-6G-08 | Post-quantum crypto | ML-KEM/ML-DSA primitives in `nextgsim-crypto` | TR 33.871 | Missing | [#22](https://github.com/NextgCoreLab/nextgsim/issues/22) |
| NGS-6G-09 | Non-terrestrial networks | timing/ephemeris/link-sim prototypes exist but are bespoke and largely unwired from live paths | Rel-17/18 NTN | Research — not yet scoped | — |
| NGS-6G-10 | Reconfigurable Intelligent Surface | RIS panel geometry, reflection modelling and control signalling | 6G research | Research — not yet scoped | — |
| NGS-6G-11 | Sub-THz / THz | sub-THz channel and propagation models (molecular absorption, weather) | 6G research | Research — not yet scoped | — |
| NGS-6G-12 | Joint Communication & Computing | compute-aware forwarding and network/compute joint optimisation | 6G research | Research — not yet scoped | — |
| NGS-6G-13 | Digital twin | network digital-twin state model and synchronisation | ITU-R IMT-2030 | Research — not yet scoped | — |
| NGS-6G-14 | Zero-energy / ambient IoT | UE-side energy-harvesting + fleet prototype exists; not integrated end to end | Rel-19 AIoT | Partial | — |
| NGS-6G-15 | O-RAN xApp/rApp | RAN Intelligent Controller interfaces (E2, A1) for xApps/rApps | O-RAN | Research — not yet scoped | — |

---

## 3. 5G-completion prerequisites

These bring the 5G baseline to the maturity the 6G work depends on. Each is a bounded, tracked task.

| Task ID | Component | Remaining work | Reference | Status | Tracking |
|---|---|---|---|---|---|
| NGS-5G-01 | RLC Acknowledged Mode | ARQ: STATUS with NACK list, automatic retransmission, timers; make AM selectable in the data path (TM/UM already work) | TS 38.322 | Partial | [#15](https://github.com/NextgCoreLab/nextgsim/issues/15) |
| NGS-5G-02 | UE NAS MM | Configuration Update command handling (received parameters are decoded but not applied), rejected/pending NSSAI state, RACS | TS 24.501 | Partial | [#19](https://github.com/NextgCoreLab/nextgsim/issues/19) |
| NGS-5G-03 | UE RRC | measurement event A6, conditional handover wiring, inter-RAT B1/B2 | TS 38.331 | Partial | [#20](https://github.com/NextgCoreLab/nextgsim/issues/20) |
| NGS-5G-04 | gNB RRC | UPER-encoded MIB/SIB1 broadcast over BCCH and migration of residual simplified RRC PDUs to real ASN.1 | TS 38.331 | Partial | [#21](https://github.com/NextgCoreLab/nextgsim/issues/21) |

The full UE 5GSM PDU-session lifecycle (establish / modify / release, multi-session allocation, back-off timers) is already implemented and tested, and is not listed here.

---

## 4. Engineering and test tasks

| Task ID | Task | Reference | Tracking |
|---|---|---|---|
| NGS-ENG-01 | Feature-gate the gNB 6G/AI crates as optional dependencies (UE side already done) | — | [#24](https://github.com/NextgCoreLab/nextgsim/issues/24) |
| NGS-ENG-02 | Property-based (proptest) round-trip tests for NAS/NGAP/RRC codecs | — | [#23](https://github.com/NextgCoreLab/nextgsim/issues/23) |
| NGS-ENG-03 | Resolve the last `Vector3` name collision (rename the RLS wire-coordinate type) | — | [#26](https://github.com/NextgCoreLab/nextgsim/issues/26) |

Progress is tracked on the [issue tracker](https://github.com/NextgCoreLab/nextgsim/issues); the code and the tracker are the source of truth.

---

## 5. Foundations for 6G evolution

- **Solid 5G control plane** — NAS, NGAP, RRC and SCTP with working registration and PDU-session establishment against a real core.
- **Operational data plane** — GTP-U tunnelling and TUN interfaces, verified end to end.
- **Full 5G crypto stack** — Milenage, SNOW3G, ZUC, AES and ECIES with test vectors.
- **ONNX Runtime integration** — `nextgsim-ai` provides model inference with multi-provider (CPU/GPU) support, the substrate the AI/ML register items build on.
- **Task-based actor architecture** — clean separation between NGAP/RRC/RLS/GTP/NAS tasks that the event bus (NGS-6G-03) will further decouple.
