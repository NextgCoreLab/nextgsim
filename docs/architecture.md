# nextgsim Architecture

This document describes the architecture of nextgsim, a pure-Rust implementation of a 5G UE and gNB simulator.

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

---

## Workspace Structure

```
rust_src/
├── Cargo.toml                    # Workspace manifest
├── nextgsim-gnb/                 # gNB binary crate
├── nextgsim-ue/                  # UE binary crate
├── nextgsim-cli/                 # CLI binary crate
├── nextgsim-common/              # Common types and utilities
├── nextgsim-nas/                 # NAS protocol library
├── nextgsim-ngap/                # NGAP protocol library
├── nextgsim-rrc/                 # RRC protocol library
├── nextgsim-crypto/              # Cryptographic functions
├── nextgsim-rls/                 # Radio Link Simulation
├── nextgsim-gtp/                 # GTP-U protocol
├── nextgsim-sctp/                # SCTP transport
└── tests/                        # Integration tests
```

---

## Crate Descriptions

### Binary Crates

| Crate | Description |
|-------|-------------|
| `nextgsim-gnb` | 5G gNodeB simulator with NGAP, RRC, GTP, RLS tasks |
| `nextgsim-ue` | 5G User Equipment simulator with NAS, RRC, RLS tasks |
| `nextgsim-cli` | Command-line interface for controlling running instances |

### Protocol Libraries

| Crate | Description | 3GPP Spec |
|-------|-------------|-----------|
| `nextgsim-nas` | NAS message encoding/decoding, 5GMM/5GSM messages | TS 24.501 |
| `nextgsim-ngap` | NGAP ASN.1 PER codec, NG interface procedures | TS 38.413 |
| `nextgsim-rrc` | RRC ASN.1 UPER codec, radio resource control | TS 38.331 |
| `nextgsim-rls` | Radio Link Simulation protocol for UE-gNB communication | Custom |

### Transport & Support

| Crate | Description |
|-------|-------------|
| `nextgsim-sctp` | SCTP transport using `sctp-proto` (pure Rust) |
| `nextgsim-gtp` | GTP-U header encoding/decoding, tunnel management |
| `nextgsim-crypto` | Milenage, SNOW3G, ZUC, NEA/NIA algorithms, KDF, ECIES |
| `nextgsim-common` | Common types (PLMN, TAI, SUPI), configuration, errors |

---

## Crate Dependencies

```
nextgsim-gnb
├── nextgsim-common
├── nextgsim-ngap
├── nextgsim-rrc
├── nextgsim-rls
├── nextgsim-gtp
├── nextgsim-sctp
└── nextgsim-crypto

nextgsim-ue
├── nextgsim-common
├── nextgsim-nas
├── nextgsim-rrc
├── nextgsim-rls
├── nextgsim-crypto
└── nextgsim-gtp

nextgsim-cli
└── nextgsim-common

nextgsim-nas
├── nextgsim-common
└── nextgsim-crypto

nextgsim-ngap
└── nextgsim-common

nextgsim-rrc
└── nextgsim-common
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
    WaitingNgSetup,  // SCTP connected, awaiting NG Setup Response
    Connected,       // NG Setup complete, ready for UE procedures
    Ready,           // Alias for Connected
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

---

## ASN.1 Codec Strategy

nextgsim uses the `asn1-compiler` and `asn1-codecs` crates for ASN.1 encoding/decoding:

| Protocol | Codec | Schema |
|----------|-------|--------|
| NGAP | APER (Aligned PER) | `tools/ngap-17.9.asn` |
| RRC | UPER (Unaligned PER) | `tools/rrc-15.6.0.asn1` |

Code is generated at compile time via `build.rs`:

```rust
// In build.rs
let mut compiler = Asn1Compiler::new();
compiler.compile_file("tools/ngap-17.9.asn")?;
compiler.generate("ngap.rs", Codec::Aper)?;
```

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

```toml
# Async runtime
tokio = { version = "1", features = ["full"] }

# Serialization
serde = { version = "1", features = ["derive"] }
serde_yaml = "0.9"

# Logging
tracing = "0.1"
tracing-subscriber = "0.3"

# Cryptography
aes = "0.8"
cmac = "0.7"
sha2 = "0.10"
hmac = "0.12"
x25519-dalek = "2"
zuc = "0.1"

# ASN.1
asn1-codecs = "0.7"

# Transport
sctp-proto = "0.6"

# CLI
clap = { version = "4", features = ["derive"] }

# Error handling
thiserror = "1"
anyhow = "1"
```

---

## Error Handling

Errors are defined using `thiserror` for each crate:

```rust
#[derive(Debug, thiserror::Error)]
pub enum NextgsimError {
    #[error("Configuration error: {0}")]
    Config(String),
    
    #[error("Protocol error: {0}")]
    Protocol(String),
    
    #[error("Network error: {0}")]
    Network(#[from] std::io::Error),
    
    #[error("ASN.1 encoding error: {0}")]
    Asn1Encode(String),
    
    #[error("Crypto error: {0}")]
    Crypto(String),
}
```

---

## Logging

Structured logging via `tracing`:

```bash
# Debug all crates
RUST_LOG=debug ./nextgsim-ue -c config.yaml

# Trace specific crate
RUST_LOG=nextgsim_nas=trace ./nextgsim-ue -c config.yaml

# Multiple levels
RUST_LOG=nextgsim_nas=trace,nextgsim_rrc=debug ./nextgsim-ue -c config.yaml
```

---

## Testing Strategy

| Level | Location | Description |
|-------|----------|-------------|
| Unit | Each crate's `src/` | Encoding/decoding, state transitions |
| Integration | `rust_src/tests/` | UE-gNB-AMF message flows |
| Conformance | Crypto crates | 3GPP test vectors |

Run tests:
```bash
cd rust_src
cargo test                    # All tests
cargo test -p nextgsim-crypto # Single crate
cargo test --test integration # Integration tests only
```

---

## References

- [3GPP TS 24.501](https://www.3gpp.org/DynaReport/24501.htm) - NAS protocol for 5G
- [3GPP TS 38.413](https://www.3gpp.org/DynaReport/38413.htm) - NGAP specification
- [3GPP TS 38.331](https://www.3gpp.org/DynaReport/38331.htm) - RRC specification
- [3GPP TS 33.501](https://www.3gpp.org/DynaReport/33501.htm) - 5G Security architecture
- [3GPP TS 35.207](https://www.3gpp.org/DynaReport/35207.htm) - Milenage test vectors
- [UERANSIM](https://github.com/aligungr/UERANSIM) - Original C++ implementation
