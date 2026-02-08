# nextgsim Protocol Crates -- 6G Gap Analysis

> Generated: 2026-02-07 | Scope: nextgsim-nas, nextgsim-ngap, nextgsim-rrc, nextgsim-rls, nextgsim-gtp, nextgsim-sctp

---

## Executive Summary

| Crate | Source Files | Role | Completeness | 6G Readiness |
|-------|-------------|------|-------------|--------------|
| nextgsim-nas | 22 | NAS 5GMM/5GSM encoding/decoding | **80%** | Low |
| nextgsim-ngap | 13 | NGAP procedures (gNB-AMF) | **85%** | Low |
| nextgsim-rrc | 9 | RRC procedures (UE-gNB) | **75%** | Low |
| nextgsim-rls | 5 | Radio Link Simulation over UDP | **95%** | N/A |
| nextgsim-gtp | 3 | GTP-U tunnel management | **90%** | Low |
| nextgsim-sctp | 3 | SCTP transport for NGAP | **90%** | Low |

**Zero `todo!()` or `unimplemented!()` macros found across all 6 crates.** All gaps are marked with comments.

---

## 1. nextgsim-nas

**Spec**: 3GPP TS 24.501 | **Files**: 22 | **Modules**: codec, eap, enums, header, ies, messages (mm + sm), security

### Implemented

| Component | Key Types | Status |
|-----------|----------|--------|
| Header encoding | `PlainMmHeader`, `PlainSmHeader`, `SecuredHeader` (7 bytes) | Complete |
| 5GMM message types | `MmMessageType` enum (24 variants) | Complete |
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
| Missing 5GMM messages | DL/UL NAS Transport, Notification, 5GMM Status with cause | Partial |
| 5GSM coverage | PDU Session Release (Request/Command/Complete/Reject) not implemented |
| Missing IEs | UE security capabilities, LADN info, MICO indication encoding |
| No 6G NAS IEs | No AI/ML capability, ISAC, semantic comm, sub-THz parameters |
| No NTN extensions | No satellite timing advance, NTN-specific access barring |

### Completeness: **80%**

Strong 5GMM and 5GSM baseline with full security algorithms. Key missing: PDU Session Release messages, some advanced IEs, all 6G extensions.

---

## 2. nextgsim-ngap

**Spec**: 3GPP TS 38.413 | **Files**: 13 | **Modules**: codec, procedures (9 procedure modules)

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
| Handover preparation failure | Missing HandoverPreparationFailure procedure |
| AMF Status Indication | Not implemented |
| RAN Configuration Update | Not implemented |
| NGAP over multiple SCTP streams | Stream management for parallel procedures not fully modeled |
| No 6G NGAP extensions | No ISAC reporting, AI-native procedures, NTN-specific IEs |

### Completeness: **85%**

Core NGAP procedures (NG Setup, NAS transport, PDU session, handover, paging) are well implemented with proper APER encoding. Missing some secondary procedures.

---

## 3. nextgsim-rrc

**Spec**: 3GPP TS 38.331 | **Files**: 9 | **Modules**: codec, procedures (6 procedure modules)

### Implemented

| Procedure | File | Status |
|-----------|------|--------|
| RRC Setup | `rrc_setup.rs` | Complete (Request/Setup/Complete) |
| RRC Release | `rrc_release.rs` | Complete |
| RRC Reconfiguration | `rrc_reconfiguration.rs` | Complete (with NAS container) |
| Security Mode | `security_mode.rs` | Complete (Command/Complete) |
| System Information | `system_information.rs` | Complete (MIB/SIB1) |
| Information Transfer | `information_transfer.rs` | Complete (DL/UL) |
| UPER Codec | `codec.rs` | Complete (ASN.1 UPER encode/decode) |

### Gaps

| Gap | Details |
|-----|---------|
| Measurement Report | Simplified byte encoding in gNB/UE (not in this crate directly) |
| RRC Reestablishment | Not implemented |
| RRC Resume | Not implemented (RRC Inactive state) |
| Conditional Handover | Not implemented |
| No 6G RRC extensions | No AI/ML configuration, ISAC measurement config, NTN timing advance, sub-THz band config |

### Completeness: **75%**

Core RRC procedures implemented with UPER codec. Missing RRC Reestablishment/Resume and all 6G extensions.

---

## 4. nextgsim-rls

**Role**: Radio Link Simulation protocol (UERANSIM-compatible) | **Files**: 5

### Implemented

| Component | File | Status |
|-----------|------|--------|
| Protocol messages | `protocol.rs` | Complete (Heartbeat, HeartbeatAck, PduTransmission, PduTransmissionAck) |
| Codec | `codec.rs` | Complete (encode/decode with version header) |
| UE Cell Search | `cell_search.rs` | Complete (cell discovery, signal tracking, heartbeat management) |
| gNB Cell Tracker | `cell_search.rs` | Complete (UE registration, signal strength) |
| RRC/Data Transport | `transport.rs` | Complete (RRC message and user plane data transport) |

### Completeness: **95%**

Fully functional simulation protocol. Not 3GPP-standardized, so no 6G relevance.

---

## 5. nextgsim-gtp

**Spec**: 3GPP TS 29.281 | **Files**: 3

### Implemented

| Component | File | Status |
|-----------|------|--------|
| GTP-U header | `codec.rs` | Complete (version, TEID, sequence number, extension headers, QFI) |
| Message types | `codec.rs` | Complete (G-PDU, Echo Req/Resp, Error Indication, End Marker) |
| Tunnel management | `tunnel.rs` | Complete (TunnelManager, PduSession, TEID allocation, session CRUD) |

### Gaps

| Gap | Details |
|-----|---------|
| No GTP-U extension header chaining | Single extension header only |
| No PDU Session Container IE | Missing 5GC-specific GTP-U extension |
| No 6G extensions | No deterministic networking, no in-network computing markers |

### Completeness: **90%**

Strong GTP-U implementation for simulation. Missing some 5GC-specific extension headers.

---

## 6. nextgsim-sctp

**Spec**: RFC 4960 | **Files**: 3 | **Backend**: sctp-proto (pure Rust)

### Implemented

| Component | File | Status |
|-----------|------|--------|
| Association | `association.rs` | Complete (connect, send, recv, shutdown, state machine) |
| Server | `server.rs` | Complete (accept connections, multi-association) |
| Config | `lib.rs` | Complete (streams, buffer sizes, NGAP PPID) |

### Gaps

| Gap | Details |
|-----|---------|
| No multi-homing | Single address only |
| No PR-SCTP | Partial reliability not supported |
| No QUIC alternative | 6G may move away from SCTP |

### Completeness: **90%**

Production-quality SCTP with async tokio interface. Wire-compatible with nextgcore.

---

## Cross-Crate 6G Gap Summary

### Protocol Stack Readiness

| Layer | 5G Status | 6G Gaps |
|-------|-----------|---------|
| NAS (L5) | 80% | No AI/ML IEs, no ISAC, no NTN, no semantic comm parameters |
| NGAP (L4) | 85% | No AI-native procedures, no ISAC reporting, no NTN IEs |
| RRC (L3) | 75% | No AI config, no ISAC measurement, no sub-THz, no RRC Resume |
| RLS (sim) | 95% | N/A (simulation protocol) |
| GTP-U (L2) | 90% | No deterministic networking, no in-network computing |
| SCTP (L1) | 90% | No QUIC alternative, no multi-homing |

### Recommendations

1. **Complete NAS 5GSM** -- PDU Session Release messages needed for full session lifecycle
2. **Add RRC Reestablishment/Resume** -- Required for RRC Inactive state (power saving)
3. **Add 6G NAS IEs** -- AI/ML capability negotiation, ISAC parameters
4. **Add NTN support** -- Satellite timing advance in NAS/RRC
5. **Implement QUIC transport** -- Future-proof alternative to SCTP
6. **Add PDU Session Container** -- 5GC-specific GTP-U extension header
