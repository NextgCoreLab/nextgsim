# nextgsim - Pure Rust 5G/6G UE and gNB Simulator

A pure Rust implementation of a 5G User Equipment (UE) and gNodeB (gNB) simulator with full 6G AI-native capabilities, converted from [UERANSIM](https://github.com/aligungr/UERANSIM). This implementation has **zero C library dependencies**.

## Features

- **Pure Rust**: No C/C++ dependencies, fully memory-safe
- **5G-SA Support**: Complete 5G Standalone network simulation
- **6G AI-Native**: Full 6G AI architecture with ML inference, federated learning, and semantic communication
- **Cryptography**: Milenage, SNOW3G, ZUC, AES-based NEA/NIA algorithms, ECIES
- **Protocol Support**: NAS, NGAP (ASN.1 PER), RRC (ASN.1 UPER), GTP-U, SCTP
- **Radio Simulation**: RLS protocol for UE-gNB communication over UDP
- **ML Inference**: ONNX Runtime integration for production AI workloads
- **Async Runtime**: Built on Tokio for high-performance concurrent operations

## Quickstart

### Option 1: Docker (Recommended)

The fastest way to get started is using Docker with the nextgcore 5G core.

```bash
# 1. Start the 5G Core (from nextgcore repository)
cd ../nextgcore/docker/rust
docker compose -f docker-compose-5gc-optimized.yml up -d

# 2. Build and start the simulators
cd ../../nextgsim
docker compose build
docker compose up -d

# 3. Verify UE registration
docker logs nextgsim-ue 2>&1 | grep -E "(REGISTERED|PDU Session)"
# Expected output:
# UE is now REGISTERED
# PDU Session 1 established with IP: 10.45.0.2
```

### Option 2: Build from Source

```bash
# Prerequisites: Rust 1.83+
rustup update stable

# Build all binaries
cargo build --release

# Run gNB (connects to AMF at 127.0.0.5:38412)
./target/release/nr-gnb -c config/gnb.yaml

# Run UE (in another terminal)
./target/release/nr-ue -c config/ue.yaml
```

### Option 3: Run Tests

```bash
# Run all tests (unit + integration)
cargo test

# Run specific test suites
cargo test --test ai_integration     # 22 AI integration tests
cargo test --test e2e_scenario       # 4 end-to-end scenario tests
cargo test --test ue_registration    # UE registration tests
cargo test --test pdu_session        # PDU session tests
```

## Docker Deployment

### Network Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Docker Network: rust_nextgcore-5gc               │
│                         Subnet: 172.23.0.0/24                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    NGAP    ┌─────────────┐    SBI    ┌─────────┐ │
│  │   nextgsim  │◄──────────►│     AMF     │◄─────────►│   NRF   │ │
│  │     gNB     │  38412     │  172.23.0.5 │           │ et al.  │ │
│  │ 172.23.0.100│            └──────┬──────┘           └─────────┘ │
│  └──────┬──────┘                   │                               │
│         │ RLS                      │ N11                           │
│         │                          ▼                               │
│  ┌──────┴──────┐            ┌─────────────┐                       │
│  │   nextgsim  │            │     SMF     │                       │
│  │     UE      │            │  172.23.0.4 │                       │
│  │ 172.23.0.101│            └──────┬──────┘                       │
│  └──────┬──────┘                   │ N4                           │
│         │                          ▼                               │
│         │ GTP-U             ┌─────────────┐                       │
│         └──────────────────►│     UPF     │──────► Internet       │
│                     2152    │  172.23.0.7 │                       │
│                             └─────────────┘                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Configuration Files

| File | Purpose |
|------|---------|
| `config/gnb.yaml` | gNB configuration (AMF address, PLMN, TAC, NCI) |
| `config/ue.yaml` | UE configuration (SUPI, keys, APN, slices) |
| `docker-compose.yaml` | Simulator container orchestration |

### Environment Variables

```bash
# gNB container
RUST_LOG=info,nextgsim_gnb=debug,nextgsim_ngap=debug

# UE container
RUST_LOG=info,nextgsim_ue=debug,nextgsim_nas=debug
```

## Workspace Structure

```
nextgsim/
├── nextgsim-gnb/       # gNB binary - 5G base station simulator
├── nextgsim-ue/        # UE binary - 5G user equipment simulator
├── nextgsim-cli/       # CLI binary - control interface
├── nextgsim-common/    # Common types (PLMN, TAI, SUPI, configs)
├── nextgsim-nas/       # NAS protocol encoding/decoding
├── nextgsim-ngap/      # NGAP protocol (ASN.1 PER codec)
├── nextgsim-rrc/       # RRC protocol (ASN.1 UPER codec)
├── nextgsim-crypto/    # Cryptographic algorithms
├── nextgsim-rls/       # Radio Link Simulation protocol
├── nextgsim-gtp/       # GTP-U tunneling protocol
├── nextgsim-sctp/      # SCTP transport layer
│
│   # 6G AI-Native Network Functions
├── nextgsim-ai/        # Core AI/ML inference infrastructure
├── nextgsim-she/       # Service Hosting Environment (3-tier edge compute)
├── nextgsim-nwdaf/     # Network Data Analytics Function (3GPP TS 23.288)
├── nextgsim-nkef/      # Network Knowledge Exposure Function
├── nextgsim-isac/      # Integrated Sensing and Communication (3GPP TR 22.837)
├── nextgsim-agent/     # AI Agent Framework with OAuth 2.0
├── nextgsim-fl/        # Federated Learning (3GPP TR 23.700-80)
├── nextgsim-semantic/  # Semantic Communication Protocols
│
└── tests/              # Integration tests
```

## 6G AI Architecture

The simulator implements a comprehensive 6G AI-native architecture following 3GPP specifications:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         6G AI-Native Architecture                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   NWDAF      │    │    NKEF      │    │    Agent     │                  │
│  │  Analytics   │◄──►│  Knowledge   │◄──►│  Framework   │                  │
│  │              │    │   Graphs     │    │   (AAF)      │                  │
│  └──────┬───────┘    └──────────────┘    └──────────────┘                  │
│         │                                                                    │
│         ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    Service Hosting Environment (SHE)                  │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │  │
│  │  │ Local Edge  │    │  Regional   │    │    Core     │              │  │
│  │  │   <10ms     │◄──►│    Edge     │◄──►│   Cloud     │              │  │
│  │  │  Inference  │    │   <20ms     │    │  Training   │              │  │
│  │  └─────────────┘    └─────────────┘    └─────────────┘              │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│         │                                         │                          │
│         ▼                                         ▼                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │    ISAC      │    │   Semantic   │    │  Federated   │                  │
│  │   Sensing    │    │    Codec     │    │   Learning   │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### AI Component Summary

| Component | Description | Specification |
|-----------|-------------|---------------|
| **SHE** | 3-tier edge compute (Local/Regional/Core) with latency-aware placement | 3GPP TS 23.558 |
| **NWDAF** | ML-powered analytics: trajectory prediction, handover optimization | 3GPP TS 23.288 |
| **NKEF** | Knowledge graphs, semantic search, RAG support for LLM integration | - |
| **ISAC** | ToA/AoA/Doppler sensing, multi-source fusion, positioning | 3GPP TR 22.837 |
| **Agent** | OAuth 2.0 authentication, intent-based networking, multi-agent coordination | - |
| **FL** | FedAvg aggregation, differential privacy, secure model distribution | 3GPP TR 23.700-80 |
| **Semantic** | Learned codecs, channel-adaptive compression, task-oriented transmission | - |

## Configuration

### gNB Configuration

```yaml
# config/gnb.yaml
nci: '0x000000010'
gnb_id_length: 32
plmn:
  mcc: 999
  mnc: 70
tac: 1
nssai:
  - sst: 1
amf_configs:
  - address: 172.23.0.5  # nextgcore AMF
    port: 38412
link_ip: 172.23.0.100
ngap_ip: 172.23.0.100
gtp_ip: 172.23.0.100
upf_addr: 172.23.0.7
upf_port: 2152
```

### UE Configuration

```yaml
# config/ue.yaml
supi: 'imsi-999700000000001'
hplmn:
  mcc: 999
  mnc: 70
key: '465B5CE8B199B49FAA5F0A2EE238A6BC'
op: 'E8ED289DEBA952E4283B54E88E6183CA'
op_type: OPC
amf: '8000'
gnb_search_list:
  - 172.23.0.100
sessions:
  - type: IPv4
    apn: internet
    slice:
      sst: 1
```

## Testing

### Test Suites

| Suite | Tests | Description |
|-------|-------|-------------|
| `ai_integration` | 22 | AI component integration tests |
| `e2e_scenario` | 4 | End-to-end protocol scenarios |
| `ue_registration` | 3 | UE registration procedures |
| `pdu_session` | 4 | PDU session establishment |
| `user_plane` | 4 | GTP-U data plane tests |
| `multi_ue` | 7 | Multiple UE scenarios |

### Running Tests

```bash
# All tests
cargo test

# With logging
RUST_LOG=debug cargo test -- --nocapture

# Specific test
cargo test test_nwdaf_handover_recommendation

# Integration tests only
cargo test --test ai_integration
cargo test --test e2e_scenario
```

## Development

### Code Quality

```bash
# Format code
cargo fmt

# Run lints
cargo clippy --all-targets

# Check for security vulnerabilities
cargo audit
```

### Adding New Features

1. Create a new crate in the workspace if needed
2. Add it to `Cargo.toml` workspace members
3. Update `Dockerfile.gnb` and `Dockerfile.ue` to include the crate
4. Write unit tests in the crate
5. Add integration tests in `tests/src/`

## Contributing

We welcome contributions! Here's how to get started:

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/nextgcore/nextgsim.git
cd nextgsim

# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build and test
cargo build
cargo test
```

### Contribution Guidelines

1. **Fork the repository** and create a feature branch
2. **Follow Rust conventions**: Use `cargo fmt` and `cargo clippy`
3. **Write tests**: All new features should include tests
4. **Update documentation**: Keep README and doc comments current
5. **Sign commits**: Use `git commit -s` for DCO sign-off

### Pull Request Process

1. Ensure all tests pass: `cargo test`
2. Run lints: `cargo clippy --all-targets`
3. Format code: `cargo fmt`
4. Update CHANGELOG.md if applicable
5. Submit PR with clear description of changes

### Areas for Contribution

- **Protocol enhancements**: NAS, NGAP, RRC message support
- **AI components**: New ML models, inference optimizations
- **Testing**: More integration tests, fuzzing
- **Documentation**: Tutorials, architecture docs
- **Performance**: Profiling, optimization

### Reporting Issues

Use GitHub Issues for:
- Bug reports (include logs and reproduction steps)
- Feature requests
- Documentation improvements

## Key Dependencies

### Core
- **tokio** - Async runtime
- **serde** / **serde_yaml** - Configuration parsing
- **tracing** - Structured logging
- **aes**, **cmac**, **sha2**, **hmac** - Cryptographic primitives
- **x25519-dalek** - Curve25519 for ECIES
- **zuc** - ZUC cipher implementation
- **sctp-proto** - Pure Rust SCTP
- **tun-rs** - TUN interface for user plane
- **asn1-codecs** - ASN.1 PER/UPER encoding

### AI/ML
- **ort** - ONNX Runtime for ML inference (GPU support via CUDA/CoreML/DirectML)
- **ndarray** - N-dimensional tensor operations
- **half** - Half-precision (FP16) floating point support

## References

### 3GPP Specifications
- [3GPP TS 24.501](https://www.3gpp.org/DynaReport/24501.htm) - NAS protocol for 5G
- [3GPP TS 38.413](https://www.3gpp.org/DynaReport/38413.htm) - NGAP specification
- [3GPP TS 38.331](https://www.3gpp.org/DynaReport/38331.htm) - RRC specification
- [3GPP TS 33.501](https://www.3gpp.org/DynaReport/33501.htm) - 5G Security architecture
- [3GPP TS 23.288](https://www.3gpp.org/DynaReport/23288.htm) - NWDAF
- [3GPP TR 22.837](https://www.3gpp.org/DynaReport/22837.htm) - ISAC
- [3GPP TR 23.700-80](https://www.3gpp.org/DynaReport/23700-80.htm) - AI/ML for 5G

### Related Projects
- [UERANSIM](https://github.com/aligungr/UERANSIM) - Original C++ implementation
- [nextgcore](../nextgcore) - Companion 5G core network

## License

GPL-3.0 - See [LICENSE](../LICENSE) for details.
