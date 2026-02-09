# nextgsim Node/Infrastructure Crates -- 6G Gap Analysis

> Generated: 2026-02-07 | Scope: nextgsim-gnb, nextgsim-ue, nextgsim-cli, nextgsim-common, nextgsim-crypto

---

## Executive Summary

The five node/infrastructure crates provide a solid 5G NR simulation baseline with fully implemented core protocol stacks (NGAP, RRC, NAS, GTP-U, RLS, SCTP). However, **all 6G AI-native network functions exist only as task ID definitions and message type stubs** -- no task implementations exist. RRC message encoding uses simplified byte formats rather than real ASN.1, and several NGAP procedures remain incomplete.

| Crate | Files | Lines (est.) | Completeness | 6G Readiness |
|---|---|---|---|---|
| nextgsim-gnb | 25 | ~12,000 | **75%** | **5% -- stubs only** |
| nextgsim-ue | 28 | ~11,500 | **75%** | **5% -- stubs only** |
| nextgsim-cli | 4 | ~600 | **95%** | N/A |
| nextgsim-common | 12 | ~3,200 | **95%** | N/A |
| nextgsim-crypto | 9 | ~3,500 | **90%** | **0% -- no PQC** |

---

## 1. nextgsim-gnb

**Binary:** `nr-gnb` | **Dependencies:** nextgsim-common, crypto, ngap, rrc, rls, gtp, sctp

### Implemented (5G baseline)

| Component | File(s) | Key Types/Functions | Status |
|---|---|---|---|
| Task framework | `src/tasks.rs` (1546 lines) | `TaskId`, `GnbTaskBase`, `TaskManager`, typed mpsc channels | Fully implemented for 6 core tasks |
| NGAP task | `src/ngap/task.rs` (1147 lines) | `NgapTask::run()`, NG Setup, NAS routing, PDU session management | Mostly implemented |
| NGAP AMF context | `src/ngap/amf_context.rs` | `AmfContext`, stream allocation, PLMN matching | Fully implemented |
| RRC task | `src/rrc/task.rs` | `RrcTask`, UL/DL message routing, NAS extraction | Fully implemented |
| RRC UE context | `src/rrc/ue_context.rs` | `RrcUeContext`, `RrcUeContextManager`, C-RNTI allocation | Fully implemented |
| RRC connection | `src/rrc/connection.rs` | `RrcConnectionManager`, Setup/Release/Reconfiguration | Implemented (simplified encoding) |
| RRC handover | `src/rrc/handover.rs` | `GnbHandoverManager`, intra-gNB handover | Partial -- intra-gNB only |
| GTP-U task | `src/gtp/task.rs` | `GtpTask`, tunnel management, loopback mode | Fully implemented |
| GTP-U tunnel | `src/gtp/tunnel.rs` | `GtpTunnel`, `GtpTunnelManager`, TEID allocation | Fully implemented |
| RLS task | `src/rls/task.rs` | `RlsTask`, heartbeat, UE registration, RRC/data transport | Fully implemented |
| SCTP transport | `src/sctp/task.rs` | `SctpTask`, association lifecycle | Fully implemented |
| SCTP AMF conn | `src/sctp/amf_connection.rs` | `AmfConnection`, stream management | Fully implemented |
| CLI server | `src/app/task.rs` | `AppTask`, command routing | Fully implemented |
| CLI handler | `src/app/cli_handler.rs` | `CliHandler`, info/status/ue-list/amf-list | Fully implemented |
| Config loader | `src/app/config_loader.rs` | Config validation, AMF config | Fully implemented |
| Status reporting | `src/app/status.rs` | `GnbStatusInfo`, YAML/JSON | Fully implemented |

### Gaps and Incomplete Items

| Gap | Location | Details |
|---|---|---|
| **6G task stubs (SHE, NWDAF, NKEF, ISAC, Agent, FL)** | `src/tasks.rs:62-67` | `TaskId` enum defines 6 AI-native IDs. Message types (`SheMessage`, `NwdafMessage`, `NkefMessage`, `IsacMessage`, `AgentMessage`, `FlAggregatorMessage`) are defined with detailed fields but **no task implementations exist**. `GnbTaskBase` only has handles for 6 core tasks. |
| UE Context Release Request | `src/ngap/task.rs:752-755` | Comment: "encoding not yet implemented" -- local cleanup only, no NGAP message sent |
| Unhandled NGAP messages | `src/ngap/task.rs` | Several NGAP message types have `_ => { tracing::warn!("not yet handled") }` match arms |
| Inter-gNB handover | `src/rrc/handover.rs:48` | `InterGnb` variant exists in `HandoverType` but comment: "not implemented yet" |
| Simplified RRC encoding | `src/rrc/connection.rs:243-267` | `build_rrc_setup()` and `build_rrc_release()` use simplified byte format, not real ASN.1 PER |
| Simplified handover PDU | `src/rrc/handover.rs:367-396` | `build_handover_rrc_reconfiguration()` and `parse_measurement_report()` use simplified byte format |
| Placeholder task routing | `src/tasks.rs:1223` | 6G message routes use `TaskId::App` as placeholder |

### todo!/unimplemented!/TODO Count

**0** `todo!()` or `unimplemented!()` macros found. Gaps are marked with comments like "not yet implemented" and "simplified".

---

## 2. nextgsim-ue

**Binary:** `nr-ue` | **Dependencies:** nextgsim-common, crypto, nas, rrc, rls, gtp

### Implemented (5G baseline)

| Component | File(s) | Key Types/Functions | Status |
|---|---|---|---|
| Task framework | `src/tasks.rs` (1749 lines) | `TaskId` (9 variants), `UeTaskBase`, `TaskManager` | Core tasks implemented; 6G stubs only |
| NAS MM state machine | `src/nas/mm/state.rs` (854 lines) | `MmStateMachine`, `RmState`, `CmState`, `MmState`, `MmSubState`, `UpdateStatus` | Fully implemented per TS 24.501 |
| NAS deregistration | `src/nas/mm/deregistration.rs` (789 lines) | `DeregistrationManager`, UE-initiated, network-initiated, collision handling | Fully implemented per TS 24.501 5.5.2 |
| NAS SM procedures | `src/nas/sm/procedure.rs` (686 lines) | `ProcedureTransactionManager`, PTI 1-254, timer management | Fully implemented |
| NAS timer management | `src/timer.rs` (1136 lines) | `NasTimerManager`, T3346-T3582, GPRS Timer 2/3 support | Fully implemented |
| RRC state machine | `src/rrc/state.rs` (641 lines) | `RrcStateMachine`, Idle/Connected/Inactive, transition validation | Fully implemented |
| RRC task | `src/rrc/task.rs` (826 lines) | `RrcTask`, cell selection, measurement, handover, RRC setup/release | Implemented (simplified encoding) |
| RRC cell selection | `src/rrc/cell_selection.rs` | `CellSelector`, suitable/acceptable categories, MIB/SIB1 | Fully implemented per TS 38.304 |
| RRC measurement | `src/rrc/measurement.rs` | `MeasurementManager`, A1-A5 events, SS-RSRP/RSRQ/SINR | Fully implemented |
| RRC handover | `src/rrc/handover.rs` | `HandoverManager`, T304 timer, state machine | Fully implemented (simplified PDU parsing) |
| RLS task | `src/rls/task.rs` | `RlsTask`, UeCellSearch, heartbeat-based cell discovery | Fully implemented |
| TUN task | `src/tun/task.rs` | `TunTask`, TUN interface lifecycle, IP packet handling | Fully implemented |
| TUN interface | `src/tun/interface.rs` | `TunInterface`, async via tun-rs | Fully implemented |
| App task | `src/app/task.rs` (589 lines) | `AppTask`, CLI server, status, NAS action execution | Fully implemented |
| CLI handler | `src/app/cli_handler.rs` (865 lines) | Command handling: info, status, timers, deregister, ps-establish, ps-release | Fully implemented |
| Config loader | `src/app/config_loader.rs` (502 lines) | HPLMN, gNB search list, subscriber key, protection scheme validation | Fully implemented |
| Status reporting | `src/app/status.rs` (494 lines) | `UeStatusInfo`, `StatusReporter`, YAML/JSON | Fully implemented |

### Gaps and Incomplete Items

| Gap | Location | Details |
|---|---|---|
| **6G task stubs (SheClient, NwdafReporter, IsacSensor, FlParticipant, SemanticCodec)** | `src/tasks.rs:72-76` | 5 AI-native `TaskId` variants defined. Message types (`SheClientMessage`, `NwdafReporterMessage`, `IsacSensorMessage`, `FlParticipantMessage`, `SemanticCodecMessage`) have detailed fields (inference requests, sensing, FL training, semantic encode/decode) but **no task implementations**. `UeTaskBase` only has handles for 4 core tasks. |
| 6G message routing | `src/tasks.rs:1441` | Uses `TaskId::App` as placeholder for 6G message routing |
| ps-release-all incomplete | `src/app/cli_handler.rs:523-524` | Only releases first PDU session; comment: "In a full implementation, this would queue releases for all sessions" |
| Simplified RRC messages | `src/rrc/task.rs:201,471,532,655` | Measurement reports, RRC Setup Complete, RRC Setup Request all use simplified byte encoding |
| Simplified handover PDU | `src/rrc/handover.rs:255-304` | `parse_handover_command()` and `build_reconfiguration_complete()` use simplified format |
| UAC always allows | `src/rrc/task.rs` | UAC (Unified Access Control) check always returns `true` |
| Cell selection heuristic | `src/main.rs:780` | "Select the cell with best signal (or just the first one for now)" |

### todo!/unimplemented!/TODO Count

**0** `todo!()` or `unimplemented!()` macros found. Gaps marked with comments.

---

## 3. nextgsim-cli

**Binary:** `nr-cli` | **Dependencies:** nextgsim-common, clap, tokio

### Implemented

| Component | File | Key Types/Functions | Status |
|---|---|---|---|
| CLI entry point | `src/main.rs` (223 lines) | clap CLI, interactive mode, dump nodes, execute commands | Fully implemented |
| CLI client | `src/client.rs` | `CliClient`, UDP send_command, receive_message with timeout | Fully implemented |
| Protocol codec | `src/protocol.rs` | 5 `MessageType` variants, encode/decode with version header | Fully implemented |
| Process table | `src/proc_table.rs` | `/tmp/nextgsim.proc-table/` discovery, `ProcTableEntry` | Fully implemented |

### Gaps

| Gap | Details |
|---|---|
| No 6G-specific commands | CLI only supports 5G node commands; no hooks for NWDAF analytics, SHE offload, ISAC sensing, FL training, or semantic codec control |

### Completeness: **95%** (fully functional for current feature set)

---

## 4. nextgsim-common

**Dependencies:** serde, thiserror, bytes, bitflags, tracing, tokio

### Implemented

| Component | File | Lines | Key Types | Status |
|---|---|---|---|---|
| BitBuffer | `src/bit_buffer.rs` | 505 | `BitBuffer`, `BitBufferReader`, bit-level read/write | Fully implemented with tests |
| BitString | `src/bit_string.rs` | 429 | `BitString`, variable-length bit sequences | Fully implemented with tests |
| CLI server | `src/cli_server.rs` | 461 | `CliServer`, `CliMessage`, `ProcTableEntry`, UDP command server | Fully implemented with tests |
| Config | `src/config.rs` | ~300 | `GnbConfig`, `UeConfig`, `AmfConfig` | Fully implemented |
| Error types | `src/error.rs` | 39 | `Error` enum (Config, Protocol, Network, ASN.1, Crypto, State, YAML) | Fully implemented |
| Logging | `src/logging.rs` | 367 | `init_logging()`, protocol message logging, hex dump | Fully implemented with tests |
| OctetString | `src/octet_string.rs` | 561 | `OctetString`, append/get multi-byte, hex, XOR | Fully implemented with tests |
| OctetView | `src/octet_view.rs` | 434 | `OctetView`, sequential byte parsing | Fully implemented with tests |
| Transport | `src/transport.rs` | 144 | `UdpTransport`, async bind/send/recv | Fully implemented with tests |
| Types | `src/types.rs` | ~300 | `Plmn`, `Tai`, `SNssai`, `Supi`, core 5G types | Fully implemented |

### Gaps

| Gap | Details |
|---|---|
| No 6G-specific types | Missing types for: ISAC sensing data, semantic communication metadata, FL model parameters, SHE compute descriptors |
| No PQC-related config | No configuration fields for post-quantum cryptography algorithm selection |

### Completeness: **95%** (fully functional for 5G; no 6G extensions)

---

## 5. nextgsim-crypto

**Dependencies:** nextgsim-common, aes, ctr, cmac, sha2, hmac, x25519-dalek, rand, zuc

### Implemented

| Component | File | Lines | Key Functions | Status |
|---|---|---|---|---|
| AES-128 | `src/aes.rs` | 316 | `Aes128Block`, `Aes128Cmac`, `xor_block` | Fully implemented, RFC 4493 test vectors |
| ECIES Profile A | `src/ecies.rs` | 377 | `ecies_encrypt/decrypt`, `generate_suci_profile_a`, X25519 + AES-CTR + HMAC-SHA256 | Fully implemented per TS 33.501 |
| KDF | `src/kdf.rs` | 664 | `derive_kausf/kseaf/kamf/knas_enc/knas_int/kgnb/nh/res_star`, HMAC-SHA256 | Fully implemented per TS 33.501 Annex A |
| Milenage | `src/milenage.rs` | 960 | `Milenage` f1-f5, f1\*/f5\*, `compute_opc` | Fully implemented, 6 3GPP TS 35.207 test sets |
| NEA2 | `src/nea.rs` | 302 | `nea2_encrypt/decrypt`, AES-128-CTR | Fully implemented, 3GPP test vector |
| NIA1/NIA2/NIA3 | `src/nia.rs` | 497 | `nia1_compute_mac` (SNOW3G), `nia2_compute_mac` (AES-CMAC), `nia3_compute_mac` (ZUC) | Fully implemented, 3GPP test vectors |
| SNOW3G | `src/snow3g.rs` | 603 | `Snow3g`, `uea2_f8`, `uia2_f9` | Fully implemented, 3GPP TS 35.222 test vectors |
| ZUC (NEA3) | `src/zuc.rs` | 361 | `nea3_encrypt/decrypt` | Fully implemented, 3GPP TS 35.222 test vectors |

### Gaps

| Gap | Details |
|---|---|
| **No post-quantum cryptography** | Missing CRYSTALS-Kyber/ML-KEM for key exchange, CRYSTALS-Dilithium/ML-DSA for signatures -- critical 6G requirement per ongoing 3GPP SA3 work |
| No NEA1 (SNOW3G ciphering) | Only `nea2_encrypt` and `nea3_encrypt` exposed; NEA1 ciphering wrapper not provided (though `uea2_f8` exists in snow3g.rs) |
| No ECIES Profile B | Only Profile A (X25519) implemented; Profile B (secp256r1) not implemented |
| NFKC normalization missing | `encode_kdf_string()` in `src/kdf.rs:358` notes: "Full NFKC normalization is not implemented" |
| No 256-bit security | All algorithms are 128-bit; 6G may require 256-bit variants (ZUC-256, SNOW5G) |

### Completeness: **90%** for 5G | **0%** for 6G PQC

---

## Cross-Crate 6G Gap Summary

### Defined but Unimplemented 6G Functions

| Function | gNB TaskId | UE TaskId | Message Types Defined | Implementation |
|---|---|---|---|---|
| **SHE** (Sub-network Heterogeneous Edge) | `She` | `SheClient` | Inference request/response, edge node update, offload computation | None |
| **NWDAF** (Network Data Analytics) | `Nwdaf` | `NwdafReporter` | Report measurement, trajectory prediction, handover recommendation | None |
| **NKEF** (Network Key Escrow Function) | `Nkef` | -- | Key request/response, key rotation | None |
| **ISAC** (Integrated Sensing and Communication) | `Isac` | `IsacSensor` | Start/stop sensing, sensing measurement, fused position request | None |
| **Agent Framework** | `Agent` | -- | Agent coordination messages | None |
| **Federated Learning** | `FlAggregator` | `FlParticipant` | Global model receive, training start, sample add, update submit | None |
| **Semantic Communication** | -- | `SemanticCodec` | Encode/decode, encoder/decoder update, adaptive compression | None |

### Architecture Readiness for 6G

The task framework (`TaskManager`, typed mpsc channels, `TaskId` enum) is **well-designed for extension**:
- Adding a 6G task requires: (1) implementing the task struct with `run()`, (2) adding a channel handle to `GnbTaskBase`/`UeTaskBase`, (3) spawning in `TaskManager`
- Message types are already defined with rich field structures
- The gap is pure implementation, not architectural

### Missing Infrastructure for 6G

1. **Post-quantum cryptography** -- No PQC algorithms in nextgsim-crypto
2. **AI/ML runtime integration** -- No model loading/inference in node crates (exists in nextgsim-ai but not wired in)
3. **Real ASN.1 encoding** -- RRC messages use simplified byte formats; needed for standards compliance
4. **Inter-gNB handover** -- Only intra-gNB handover implemented
5. **Full NGAP procedure set** -- Several message types unhandled
6. **6G CLI commands** -- No CLI hooks for AI-native functions
7. **6G config types** -- No configuration structs for ISAC, SHE, FL, semantic codec parameters
