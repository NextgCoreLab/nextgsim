# nextgsim Protocol Crates -- 6G Gap Analysis

> Generated: 2026-02-07, rows re-checked against the code before 0.2.0 | Scope: nextgsim-nas, nextgsim-ngap, nextgsim-rrc, nextgsim-rls, nextgsim-gtp, nextgsim-sctp. The LTE crates (`nextgsim-s1ap`, `nextgsim-lte-rrc`, `nextgsim-eps-nas`) and `nextgsim-pdcp` postdate this analysis and are not covered.

---

## Executive Summary

| Crate | Role | 6G Readiness |
|-------|------|--------------|
| nextgsim-nas | NAS 5GMM/5GSM encoding/decoding | Low |
| nextgsim-ngap | NGAP procedures (gNB-AMF) | Low |
| nextgsim-rrc | RRC procedures (UE-gNB) | Low |
| nextgsim-rls | Radio Link Simulation over UDP | N/A |
| nextgsim-gtp | GTP-U tunnel management | Low |
| nextgsim-sctp | SCTP transport for NGAP | Low |

**Zero `todo!()` or `unimplemented!()` macros found across all 6 crates.** All gaps are marked with comments.

---

## 1. nextgsim-nas

**Spec**: 3GPP TS 24.501 | **Modules**: codec, eap, enums, header, ies, messages (mm + sm), security

### Implemented

| Component | Key Types | Status |
|-----------|----------|--------|
| Header encoding | `PlainMmHeader`, `PlainSmHeader`, `SecuredHeader` (7 bytes) | Complete |
| 5GMM message types | `MmMessageType` enum | Complete |
| 5GMM messages | Registration (Request/Accept/Reject/Complete), Authentication (Request/Response/Result/Reject), Security Mode (Command/Complete), Deregistration, Service (Request/Accept/Reject), Identity (Request/Response), Status, Configuration Update | Complete |
| 5GSM messages | PDU Session Establishment (Request/Accept/Reject), PDU Session Modification (Request/Command/Complete/Reject) | Complete |
| Type 1 IEs | 30+ IE types: RegistrationType, ServiceType, AccessType, PduSessionType, SscMode, IdentityType, etc. | Complete |
| Type 3 IEs | Multiple variable-length IEs | Complete |
| NAS Security | NIA1/2/3, NEA1/2/3, NAS count, MAC compute/verify, protect/unprotect | Complete |
| EAP | EAP message parsing (Identity, AKA Challenge) | Complete |
| Capture tests | Real pcap-based decode tests | Complete |

### Gaps

| Gap | Details |
|-----|---------|
| 5GMM message coverage | DL/UL NAS Transport (`src/messages/mm/nas_transport.rs`, encode/decode for DlNasTransport + UlNasTransport), Notification (`notification.rs`), and UE policy / UPDP (`ue_policy.rs`, TS 24.501 Annex D + TS 24.526 URSP) are now implemented | Complete |
| 5GSM coverage | PDU Session Release (Request/Command/Complete/Reject) now implemented in `src/messages/sm/pdu_session_release.rs` |
| Missing IEs | UE security capabilities, LADN info, MICO indication encoding |
| No 6G NAS IEs | No AI/ML capability, ISAC, semantic comm, sub-THz parameters |
| NTN extensions unwired | NTN NAS IEs (satellite timing advance, NTN access barring) now exist as bespoke sim-only prototypes; the gap is wire conformance and wiring, not absence |


Strong 5GMM and 5GSM baseline with full security algorithms. Key missing: PDU Session Release messages, some advanced IEs, all 6G extensions.

---

## 2. nextgsim-ngap

**Spec**: 3GPP TS 38.413 | **Modules**: codec, procedures

### Implemented

| Procedure | File | Status |
|-----------|------|--------|
| NG Setup | `ng_setup.rs` | Complete (Request/Response/Failure) |
| Initial UE Message | `initial_ue_message.rs` | Complete |
| NAS Transport | `nas_transport.rs` | Complete (DL/UL) |
| Initial Context Setup | `initial_context_setup.rs` | Complete (Request/Response/Failure) |
| PDU Session Resource | `pdu_session_resource.rs` | Complete (Setup/Modify/Release Request/Response) |
| UE Context Release | `ue_context_release.rs` | Complete (Command/Complete) |
| Handover | `handover.rs` | Partial (Required/Request/Command/Ack/Notify) |
| Paging | `paging.rs` | Complete |
| Error Indication | `error_indication.rs` | Complete |
| APER Codec | `codec.rs` | Complete (ASN.1 APER encode/decode) |

### Gaps

| Gap | Details |
|-----|---------|
| Handover preparation failure | Now implemented: `procedures/handover_preparation_failure.rs` |
| AMF Status Indication | Now implemented: `procedures/amf_status_indication.rs` |
| RAN Configuration Update | Now implemented: `procedures/ran_configuration_update.rs`; `ng_reset.rs` and `path_switch.rs` procedures also added |
| NGAP over multiple SCTP streams | Stream management for parallel procedures not fully modeled |
| No 6G NGAP extensions | No ISAC reporting or AI-native procedures; an off-wire NTN support module now exists (bespoke, sim-only) but no wire-conformant NTN IEs |


Core NGAP procedures (NG Setup, NAS transport, PDU session, handover, paging) are well implemented with proper APER encoding. Missing some secondary procedures.

---

## 3. nextgsim-rrc

**Spec**: 3GPP TS 38.331 | **Modules**: codec, procedures

### Implemented

| Procedure | File | Status |
|-----------|------|--------|
| RRC Setup | `rrc_setup.rs` | Complete (Request/Setup/Complete) |
| RRC Release | `rrc_release.rs` | Complete |
| RRC Reconfiguration | `rrc_reconfiguration.rs` | Complete (with NAS container) |
| Security Mode | `security_mode.rs` | Complete (Command/Complete) |
| System Information | `system_information.rs` | Complete (MIB/SIB1); broadcast by the gNB on BCCH and decoded by the UE |
| Information Transfer | `information_transfer.rs` | Complete (DL/UL) |
| Paging (PCCH) | `paging.rs` | Complete (PCCH-Message/Paging, 5G-S-TMSI and full I-RNTI records); broadcast by the gNB and matched by the UE end to end |
| UPER Codec | `codec.rs` | Complete (ASN.1 UPER encode/decode) |

### Gaps

| Gap | Details |
|-----|---------|
| Measurement Report | Done: the UE sends and the gNB parses a real UPER `MeasurementReport` (#107) |
| Residual hand-rolled RRC PDUs | Done (#146): `RRCReestablishment`, `RRCResume`, the handover command and the measurement report are UPER. Two fallbacks remain in the gNB: a bespoke RRCSetup frame if the ASN.1 encode fails, and a bespoke UL-CCCH parse if the UPER decode fails |
| Paging occasion (PF/PO) | Done (#99): the paging occasion is derived from a shared frame clock per TS 38.304 §7.1 (`paging_default_cycle_frames` on both ends) |
| RRC Reestablishment | Now implemented: `procedures/rrc_reestablishment.rs` |
| RRC Resume | Now implemented: `procedures/rrc_resume.rs` + UE-side `nextgsim-ue/src/rrc/resume.rs` (TS 38.331 5.3.13, T319) |
| Conditional Handover | Now implemented: `procedures/conditional_handover.rs` |
| No 6G RRC extensions | No AI/ML configuration, ISAC measurement config or sub-THz band config. (NTN timing advance is done: SIB19 is broadcast and applied, #56) |


Core RRC procedures implemented with UPER codec. Missing RRC Reestablishment/Resume and all 6G extensions.

---

## 4. nextgsim-rls

**Role**: Radio Link Simulation protocol (UERANSIM-compatible)

### Implemented

| Component | File | Status |
|-----------|------|--------|
| Protocol messages | `protocol.rs` | Complete (Heartbeat, HeartbeatAck, PduTransmission, PduTransmissionAck) |
| Codec | `codec.rs` | Complete (encode/decode with version header) |
| UE Cell Search | `cell_search.rs` | Complete (cell discovery, signal tracking, heartbeat management) |
| gNB Cell Tracker | `cell_search.rs` | Complete (UE registration, signal strength) |
| RRC/Data Transport | `transport.rs` | Complete (RRC message and user plane data transport) |


Fully functional simulation protocol. Not 3GPP-standardized, so no 6G relevance.

---

## 5. nextgsim-gtp

**Spec**: 3GPP TS 29.281

### Implemented

| Component | File | Status |
|-----------|------|--------|
| GTP-U header | `codec.rs` | Complete (version, TEID, sequence number, extension headers, QFI) |
| Message types | `codec.rs` | Complete (G-PDU, Echo Req/Resp, Error Indication, End Marker) |
| Tunnel management | `tunnel.rs` | Complete (TunnelManager, PduSession, TEID allocation, session CRUD) |

### Gaps

| Gap | Details |
|-----|---------|
| GTP-U extension header chaining | Done: `codec.rs` chains extension headers |
| PDU Session Container | Done: QFI and RQI on the PDU Session Container (`PduSessionInfo`) |
| 6G extensions | Research markers only: `TsnMarker` and `InNetworkCompute` extension headers, non-normative |


Strong GTP-U implementation for simulation.

---

## 6. nextgsim-sctp

**Spec**: RFC 4960 | **Backend**: sctp-proto (pure Rust)

### Implemented

| Component | File | Status |
|-----------|------|--------|
| Association | `association.rs` | Complete (connect, send, recv, shutdown, state machine), including `MultihomeSctpAssociation` |
| Server | `server.rs` | Complete (accept connections, multi-association) |
| Kernel backend | `kernel.rs` | Complete behind `kernel-sctp`: real SCTP via libsctp, partial-delivery reassembly |
| Config | `lib.rs` | Complete (streams, buffer sizes, NGAP PPID); the shared constants come from nextgcore's `nextgcore-sctp` |

### Gaps

| Gap | Details |
|-----|---------|
| No PR-SCTP | Partial reliability not supported |
| QUIC not selectable | `quic.rs` exists, but the gNB's SCTP task falls back to SCTP when `quic_enabled` is set |


Async tokio interface. The userspace backend is SCTP over UDP and only interoperates with nextgcore's matching backend; the kernel backend talks to any AMF or MME.

---

## Cross-Crate 6G Gap Summary

### Protocol Stack Readiness

| Layer | 5G Status | 6G Gaps |
|-------|-----------|---------|
| NAS (L5) | 80% | No AI/ML IEs, no ISAC, no semantic comm parameters; NTN IEs exist as bespoke sim-only prototypes (not wire-conformant) |
| NGAP (L4) | 85% | No AI-native procedures, no ISAC reporting; off-wire NTN module exists (bespoke, sim-only) |
| RRC (L3) | 75% | No AI config, no ISAC measurement, no sub-THz, no RRC Resume |
| RLS (sim) | 95% | N/A (simulation protocol) |
| GTP-U (L2) | 90% | No deterministic networking, no in-network computing |
| SCTP (L1) | 90% | No QUIC alternative, no multi-homing |

### Recommendations

1. **Complete NAS 5GSM** -- DONE: PDU Session Release messages implemented (`src/messages/sm/pdu_session_release.rs`)
2. **Add RRC Reestablishment/Resume** -- DONE: both implemented (`rrc_reestablishment.rs`, `rrc_resume.rs`)
3. **Add 6G NAS IEs** -- AI/ML capability negotiation, ISAC parameters
4. **Add NTN support** -- Satellite timing advance in NAS/RRC
5. **Implement QUIC transport** -- Future-proof alternative to SCTP
6. **Add PDU Session Container** -- 5GC-specific GTP-U extension header
