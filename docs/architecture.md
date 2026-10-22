# nextgsim Architecture

This document describes the architecture of nextgsim, a pure-Rust 5G UE and gNB simulator with an LTE eNB and LTE UE beside them. The deeper, code-referenced chapters are in the mdBook (`docs-book/src/architecture/gnb.md`, `ue.md`).

## Overview

nextgsim is organized as a Cargo workspace with multiple crates, each responsible for a specific domain. The architecture follows an actor-based task model with message passing for concurrent operations.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           nextgsim Workspace                            │
├─────────────────────────────────────────────────────────────────────────┤
│  Binaries                                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                     │
│  │ nextgsim-gnb│  │ nextgsim-ue │  │nextgsim-cli │                     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                     │
│         │                │                │                             │
├─────────┼────────────────┼────────────────┼─────────────────────────────┤
│  Protocol Libraries                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │nextgsim-nas │  │nextgsim-ngap│  │ nextgsim-rrc│  │ nextgsim-rls│    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │
│         │                │                │                │            │
├─────────┼────────────────┼────────────────┼────────────────┼────────────┤
│  Transport & Support                                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │nextgsim-sctp│  │ nextgsim-gtp│  │nextgsim-    │  │nextgsim-    │    │
│  │             │  │             │  │crypto       │  │common       │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

The diagram shows the NR side. The LTE side (`lte-enb`, `lte-ue`) uses
`nextgsim-s1ap`, `nextgsim-lte-rrc` and `nextgsim-eps-nas` over the same
`rls`, `rlc`, `pdcp`, `gtp` and `sctp` crates.

---

## Workspace Structure

The workspace (`Cargo.toml` `members`) has 26 crates plus the `tests/`
integration-test crate (`nextgsim-integration-tests`). Counted from the
`members` list: `sed -n '/^members/,/^]/p' Cargo.toml | grep -c nextgsim-`.

```
nextgsim/
├── Cargo.toml                    # Workspace manifest (pins asn1-codegen, sctp-proto)
├── rust-toolchain.toml           # Pinned toolchain
├── nextgsim-gnb/                 # nr-gnb
├── nextgsim-ue/                  # nr-ue
├── nextgsim-enb/                 # lte-enb
├── nextgsim-lte-ue/              # lte-ue
├── nextgsim-cli/                 # nr-cli, nr-loadtest
├── nextgsim-common/              # Types, configuration, CLI server, errors
├── nextgsim-nas/                 # 5GS NAS (TS 24.501)
├── nextgsim-eps-nas/             # EPS NAS (TS 24.301, UE side)
├── nextgsim-ngap/                # NGAP + NRPPa (generated APER)
├── nextgsim-s1ap/                # S1AP (generated APER)
├── nextgsim-rrc/                 # NR RRC (generated UPER)
├── nextgsim-lte-rrc/             # E-UTRA RRC (generated UPER)
├── nextgsim-rlc/                 # RLC TM/UM/AM, SRAP
├── nextgsim-pdcp/                # PDCP SRB/DRB, AS security, SDAP header
├── nextgsim-crypto/              # Milenage, TUAK, SNOW3G, ZUC, AES, ECIES, KDF, PQC
├── nextgsim-rls/                 # Radio Link Simulation
├── nextgsim-gtp/                 # GTP-U
├── nextgsim-sctp/                # SCTP (userspace + kernel backends)
├── nextgsim-ai/ -she/ -nwdaf/ -nkef/ -isac/ -agent/ -fl/ -semantic/   # 6G research crates
├── tools/                        # 3GPP ASN.1 schemas + extract_asn1.py
└── tests/                        # Integration tests
```

---

## Crate Descriptions

### Binary Crates

| Crate | Binary | Description |
|-------|--------|-------------|
| `nextgsim-gnb` | `nr-gnb` | 5G gNodeB: App, NGAP, RRC, GTP, RLS and SCTP tasks |
| `nextgsim-ue` | `nr-ue` | 5G UE: App, NAS, RRC, RLS and TUN tasks |
| `nextgsim-enb` | `lte-enb` | LTE eNB: S1 Setup and UE-associated S1AP, E-UTRA cell over RLS, SRB/DRB PDCP, S1-U GTP-U |
| `nextgsim-lte-ue` | `lte-ue` | LTE UE: cell selection, E-UTRA RRC, EPS NAS attach, AS security, DRBs, TUN |
| `nextgsim-cli` | `nr-cli`, `nr-loadtest` | Control interface for running instances; multi-UE load test |

### Protocol Libraries

| Crate | Description | 3GPP Spec |
|-------|-------------|-----------|
| `nextgsim-nas` | 5GMM/5GSM message encoding/decoding, NAS security | TS 24.501 |
| `nextgsim-eps-nas` | EMM/ESM (UE side), EPS AKA, NAS security | TS 24.301 |
| `nextgsim-ngap` | NGAP and NRPPa APER codecs and NG procedures | TS 38.413, TS 38.455 |
| `nextgsim-s1ap` | S1AP APER codec | TS 36.413 |
| `nextgsim-rrc` | NR RRC UPER codec and procedures | TS 38.331 |
| `nextgsim-lte-rrc` | E-UTRA RRC UPER codec | TS 36.331 |
| `nextgsim-rlc` | RLC TM/UM/AM, LTE RLC UM, SRAP | TS 38.322, TS 36.322, TS 38.351 |
| `nextgsim-pdcp` | PDCP entities for SRBs and DRBs, AS integrity/ciphering, SDAP header | TS 38.323, TS 36.323, TS 37.324 |
| `nextgsim-rls` | Radio Link Simulation protocol for UE-gNB / UE-eNB communication over UDP | Custom |

### Transport & Support

| Crate | Description |
|-------|-------------|
| `nextgsim-sctp` | SCTP: a userspace `sctp-proto` backend over UDP (default) and a kernel backend via libsctp (`kernel-sctp` feature). The shared constants come from nextgcore's `nextgcore-sctp` crate |
| `nextgsim-gtp` | GTP-U header encoding/decoding, tunnel management, path supervision |
| `nextgsim-crypto` | Milenage, TUAK, SNOW3G, ZUC, NEA/NIA algorithms, KDF, ECIES, PQC (`pqc` feature) |
| `nextgsim-common` | Common types (PLMN, TAI, SUPI), configuration, CLI server, errors |

---

## Crate Dependencies

Workspace-internal dependencies, from each crate's `[dependencies]`:

```
nextgsim-gnb     common, crypto, ngap, rrc, rls, rlc, pdcp, gtp, sctp (+ optional 6G crates)
nextgsim-ue      common, crypto, nas, rrc, rls, rlc, pdcp, gtp (+ optional 6G crates)
nextgsim-enb     common, crypto, s1ap, lte-rrc, eps-nas, pdcp, rls, rlc, gtp, sctp
nextgsim-lte-ue  common, crypto, eps-nas, lte-rrc, pdcp, rls, rlc, ue
nextgsim-cli     common
nextgsim-nas     common, crypto
nextgsim-eps-nas common, crypto
nextgsim-ngap    common
nextgsim-rrc     common, crypto, rlc
nextgsim-pdcp    crypto
nextgsim-sctp    nextgcore-sctp (git, pinned by rev)
```

---

## Task Architecture

Both gNB and UE use an actor-based task model with Tokio channels for message passing. Each task runs as an independent async task and communicates via typed message channels.

### Message Passing Pattern

```rust
/// Task message envelope
pub enum TaskMessage<T> {
    Message(T),   // Regular message
    Shutdown,     // Shutdown signal
}

/// Base trait for all tasks
#[async_trait]
pub trait Task: Send + 'static {
    type Message: Send;
    async fn run(&mut self, rx: mpsc::Receiver<TaskMessage<Self::Message>>);
}
```

### gNB Task Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                         gNB Application                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐      │
│  │   App   │◄──►│  NGAP   │◄──►│   RRC   │◄──►│   RLS   │      │
│  │  Task   │    │  Task   │    │  Task   │    │  Task   │      │
│  └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘      │
│       │              │              │              │            │
│       │         ┌────┴────┐    ┌────┴────┐        │            │
│       │         │  SCTP   │    │   GTP   │        │            │
│       │         │  Task   │    │  Task   │        │            │
│       │         └────┬────┘    └────┬────┘        │            │
│       │              │              │              │            │
└───────┼──────────────┼──────────────┼──────────────┼────────────┘
        │              │              │              │
   CLI Server     SCTP/N2         GTP-U/N3      RLS/UDP
```

| Task | Responsibility |
|------|----------------|
| App | Configuration, CLI commands, status reporting |
| NGAP | NG Setup, UE context, NAS routing, PDU sessions |
| RRC | RRC connection management, UE tracking, NAS relay |
| GTP | GTP-U tunnel management, user plane forwarding |
| RLS | UE discovery, RRC/data relay over UDP |
| SCTP | AMF connection management, NGAP PDU transport |

### UE Task Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                          UE Application                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐      │
│  │   App   │◄──►│   NAS   │◄──►│   RRC   │◄──►│   RLS   │      │
│  │  Task   │    │  Task   │    │  Task   │    │  Task   │      │
│  └────┬────┘    └─────────┘    └─────────┘    └────┬────┘      │
│       │                                            │            │
│  ┌────┴────┐                                       │            │
│  │   TUN   │                                       │            │
│  │Interface│                                       │            │
│  └────┬────┘                                       │            │
│       │                                            │            │
└───────┼────────────────────────────────────────────┼────────────┘
        │                                            │
   User Plane                                   RLS/UDP
   (IP packets)                              (to gNB)
```

| Task | Responsibility |
|------|----------------|
| App | Configuration, CLI commands, TUN data, status |
| NAS | MM/SM state machines, registration, PDU sessions |
| RRC | RRC state machine, cell selection, measurements |
| RLS | Cell search, gNB connection, RRC/data transport |

---

## State Machines

### UE Mobility Management (MM) States

The UE MM state machine follows 3GPP TS 24.501:

```
                    ┌──────────────────┐
                    │       NULL       │
                    └────────┬─────────┘
                             │ Power on
                             ▼
┌────────────────────────────────────────────────────────────────┐
│                      DEREGISTERED                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ NormalService│  │ PlmnSearch  │  │ NoSupi      │  ...       │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└────────────────────────────┬───────────────────────────────────┘
                             │ Registration Request
                             ▼
                    ┌──────────────────┐
                    │REGISTERED_INITIATED│
                    └────────┬─────────┘
                             │ Registration Accept
                             ▼
┌────────────────────────────────────────────────────────────────┐
│                       REGISTERED                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ NormalService│  │ UpdateNeeded│  │ PlmnSearch  │  ...       │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└────────────────────────────────────────────────────────────────┘
```

### UE RM/CM States

```rust
pub enum RmState {
    Deregistered,  // Not registered with network
    Registered,    // Registered with network
}

pub enum CmState {
    Idle,          // No NAS signaling connection
    Connected,     // NAS signaling connection active
}

pub enum RrcState {
    Idle,          // No RRC connection
    Connected,     // RRC connection active
    Inactive,      // RRC connection suspended
}
```

### gNB AMF States

```rust
pub enum AmfState {
    NotConnected,    // No SCTP association
    Connected,       // SCTP association up, NG Setup not yet sent
    WaitingNgSetup,  // NG Setup in progress
    Ready,           // NG Setup complete, ready for UE procedures
    Overloaded,      // AMF indicated overload
}
```

---

## Protocol Flows

### UE Registration

```
    UE                    gNB                    AMF
     │                     │                      │
     │──RRC Setup Request─►│                      │
     │◄──RRC Setup────────│                      │
     │──RRC Setup Complete─►│                      │
     │  (NAS: Reg Request) │──Initial UE Message─►│
     │                     │  (NAS: Reg Request)  │
     │                     │◄─Auth Request────────│
     │◄─DL NAS Transport───│                      │
     │  (NAS: Auth Request)│                      │
     │──UL NAS Transport──►│                      │
     │  (NAS: Auth Response)│──UL NAS Transport──►│
     │                     │◄─Security Mode Cmd───│
     │◄─DL NAS Transport───│                      │
     │──UL NAS Transport──►│                      │
     │  (NAS: SMC Complete)│──UL NAS Transport───►│
     │                     │◄─Initial Context Setup│
     │                     │  (NAS: Reg Accept)   │
     │◄─RRC Reconfiguration│                      │
     │  (NAS: Reg Accept)  │                      │
     │──RRC Reconfig Compl─►│                      │
     │──UL NAS Transport──►│                      │
     │  (NAS: Reg Complete)│──UL NAS Transport───►│
     │                     │──Init Ctx Setup Resp─►│
```

### PDU Session Establishment

```
    UE                    gNB                    AMF/SMF
     │                     │                      │
     │──UL NAS Transport──►│                      │
     │  (PDU Sess Est Req) │──UL NAS Transport───►│
     │                     │                      │──SMF interaction──►
     │                     │◄─PDU Sess Resource───│
     │                     │  Setup Request       │
     │                     │  (PDU Sess Est Accept)│
     │◄─RRC Reconfiguration│                      │
     │  (PDU Sess Est Accept)                     │
     │──RRC Reconfig Compl─►│                      │
     │                     │──PDU Sess Resource───►│
     │                     │  Setup Response      │
     │                     │                      │
     │◄═══════════════════►│◄═══════════════════►│
     │    User Plane Data (GTP-U tunnel)          │
```

### LTE attach (lte-ue, lte-enb, nextgcore EPC)

```
    UE                    eNB                    MME
     │──RRCConnectionRequest─►│                   │
     │◄─RRCConnectionSetup───│                   │
     │──RRCConnectionSetupComplete (ATTACH REQUEST + PDN CONNECTIVITY REQUEST)
     │                     │──Initial UE Message─►│
     │◄────── EPS AKA, NAS Security Mode (DL/UL NAS Transport) ──────►│
     │                     │◄─Initial Context Setup Request (ATTACH ACCEPT)
     │◄─SecurityModeCommand (AS)                  │
     │◄─RRCConnectionReconfiguration (SRB2, DRB)  │
     │                     │──Initial Context Setup Response (S1-U F-TEID)
     │──ATTACH COMPLETE───►│──Uplink NAS Transport►│
     │◄═══════ IP over the default bearer (PDCP/RLC UM, S1-U GTP-U) ═══════►
```

---

## ASN.1 Codec Strategy

Every ASN.1 codec is generated at build time. Each crate's `build.rs` runs the
`asn1-compiler` over a 3GPP schema in `tools/`, and the generated types use the
`asn1-codecs` runtime. Both come from
[NextgCoreLab/asn1-codegen](https://github.com/NextgCoreLab/asn1-codegen),
pinned by `rev` in the workspace `Cargo.toml` (`[workspace.dependencies]` and
`[patch.crates-io]`, so every crate sees one copy). asn1-codegen is the shared
home of the compiler fixes both this repo and nextgcore need; nextgsim no
longer vendors the compiler.

| Protocol | Crate | Codec | Schema |
|----------|-------|-------|--------|
| NGAP | `nextgsim-ngap` | APER | `tools/ngap-17.9.asn` |
| NRPPa | `nextgsim-ngap` | APER | `tools/nrppa-19.2.0.asn1` |
| S1AP | `nextgsim-s1ap` | APER | `tools/s1ap-19.2.0.asn` |
| NR RRC | `nextgsim-rrc` | UPER | `tools/rrc-19.3.0.asn1` |
| E-UTRA RRC | `nextgsim-lte-rrc` | UPER | `tools/lte-rrc-19.4.0.asn` |

The S1AP and E-UTRA RRC schemas were extracted from the 3GPP `.docx` by
`tools/extract_asn1.py`; `tools/S1AP-PROVENANCE.md` and
`tools/LTE-RRC-PROVENANCE.md` record the source, the command and both sha256
values. The NR RRC schema is a Release 19 extract of TS 38.331 V19.3.0.

The build scripts call the compiler like this (`nextgsim-ngap/build.rs`):

```rust
let mut compiler = Asn1Compiler::new(
    output_file.to_str().expect("value expected"),
    &Visibility::Public,
    vec![Codec::Aper],
    vec![Derive::Debug, Derive::Clone, Derive::PartialEq],
);
compiler.compile_files(&[schema_str])?;
```

The output lands in `OUT_DIR` and is pulled in with
`include!(concat!(env!("OUT_DIR"), "/ngap.rs"))`. The generated RRC codecs are
large: CI gives the Test job swap headroom for them.

---

## Cryptographic Algorithms

| Algorithm | Implementation | Usage |
|-----------|----------------|-------|
| Milenage | Custom (no crate available) | 5G-AKA authentication |
| SNOW3G | Custom (no crate available) | NEA1/NIA1 |
| ZUC | `zuc` crate | NEA3/NIA3 |
| AES | `aes`, `cmac` crates | NEA2/NIA2 |
| ECIES | `x25519-dalek` crate | SUPI concealment |
| KDF | `hmac`, `sha2` crates | Key derivation |

---

## External Dependencies

The authoritative list is `[workspace.dependencies]` in `Cargo.toml`. The ones
that shape the design:

| Dependency | Use |
|------------|-----|
| `tokio` | Async runtime and the task channels |
| `serde`, `serde_yaml` | Configuration |
| `tracing`, `tracing-subscriber` | Logging |
| `aes`, `cmac`, `sha2`, `hmac`, `x25519-dalek`, `zuc` | Cryptographic primitives |
| `asn1-codecs`, `asn1_codecs_derive`, `asn1-compiler` | ASN.1 runtime and build-time compiler, git dependencies on NextgCoreLab/asn1-codegen |
| `sctp-proto` 0.6 | Userspace SCTP. Patched to carry any PPID verbatim; the patched crate lives in nextgcore's `src/vendor/sctp-proto` and is taken via `[patch.crates-io]` (see [nextgcore-integration.md](nextgcore-integration.md#the-patched-sctp-proto)) |
| `nextgcore-sctp` | Shared SCTP constants, a git dependency on nextgcore pinned by `rev` |
| `tun-rs` | TUN interfaces |
| `clap` | Command-line parsing |
| `ort` (`load-dynamic`) | ONNX Runtime for the 6G research crates |
| `thiserror`, `anyhow` | Errors |

---

## Error Handling

Errors are defined using `thiserror`; the shared one is:

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {   // nextgsim-common/src/error.rs
    #[error("Configuration error: {0}")]
    Config(String),
    
    #[error("Protocol error: {0}")]
    Protocol(String),
    
    #[error("Network error: {0}")]
    Network(#[from] std::io::Error),
    
    #[error("ASN.1 encoding error: {0}")]
    Asn1Encode(String),

    #[error("ASN.1 decoding error: {0}")]
    Asn1Decode(String),

    #[error("Crypto error: {0}")]
    Crypto(String),

    #[error("State machine error: {0}")]
    StateMachine(String),

    #[error("YAML parse error: {0}")]
    YamlParse(#[from] serde_yaml::Error),
}
```

---

## Logging

Structured logging via `tracing`:

```bash
# Debug all crates
RUST_LOG=debug ./nr-ue -c config/ue.yaml

# Trace specific crate
RUST_LOG=nextgsim_nas=trace ./nr-ue -c config/ue.yaml

# Multiple levels
RUST_LOG=nextgsim_nas=trace,nextgsim_rrc=debug ./nr-ue -c config/ue.yaml
```

---

## Testing Strategy

| Level | Location | Description |
|-------|----------|-------------|
| Unit | Each crate's `src/` | Encoding/decoding, state transitions |
| Property | `nextgsim-nas`, `nextgsim-ngap`, `nextgsim-rrc` | `proptest` codec round trips |
| Integration | `tests/` | UE-gNB-AMF message flows against a mock AMF, one `[[test]]` per file |
| Conformance | `nextgsim-crypto` | 3GPP test vectors (TS 35.207 Milenage, TS 35.232 TUAK) and NIST ACVP vectors for PQC |
| Cross-repo E2E | nextgcore `docker/rust/e2e.sh`, nightly | Registration, PDU session, ping, paging, Service Request, de-registration against the real core |

Run tests:
```bash
cargo test --workspace                       # All tests
cargo test -p nextgsim-crypto                # Single crate
cargo test -p nextgsim-integration-tests     # Integration tests only
```

---

## References

- [3GPP TS 24.501](https://www.3gpp.org/DynaReport/24501.htm) - NAS protocol for 5G
- [3GPP TS 38.413](https://www.3gpp.org/DynaReport/38413.htm) - NGAP specification
- [3GPP TS 38.331](https://www.3gpp.org/DynaReport/38331.htm) - RRC specification
- [3GPP TS 36.413](https://www.3gpp.org/DynaReport/36413.htm) - S1AP
- [3GPP TS 24.301](https://www.3gpp.org/DynaReport/24301.htm) - EPS NAS
- [3GPP TS 33.501](https://www.3gpp.org/DynaReport/33501.htm) - 5G Security architecture
- [3GPP TS 35.207](https://www.3gpp.org/DynaReport/35207.htm) - Milenage test vectors
- [UERANSIM](https://github.com/aligungr/UERANSIM) - Original C++ implementation
