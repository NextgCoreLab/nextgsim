# nextgsim - Pure Rust 5G/LTE UE, gNB and eNB Simulator

A pure Rust implementation of a 5G User Equipment (UE) and gNodeB (gNB) simulator, plus an LTE eNB and LTE UE, with 6G research/exploration prototypes. The 5G half started as a conversion of [UERANSIM](https://github.com/aligungr/UERANSIM). The default build has **no C library dependencies**; the optional `kernel-sctp` feature links libsctp. (3GPP Rel-20/6G has no frozen stage-3 spec; the 6G features are research concepts informed by TR 22.870 use cases and are disabled by default.)

## Features

- **Pure Rust**: No C/C++ dependencies in the default build
- **5G-SA**: registration, authentication (5G-AKA, EAP-AKA'), NAS and AS security, PDU sessions over a GTP-U user plane, CM-IDLE paging and Service Request (the UE answers a page and holds the uplink packet that triggers a Service Request until the user plane is back), RRC_INACTIVE resume, N2 and conditional handover (no Xn-C between gNBs), re-registration after a network-initiated de-registration
- **LTE (E-UTRA / EPC)**: an LTE eNB (`lte-enb`: S1 Setup, S1AP UE signalling, S1-U GTP-U) and an LTE UE (`lte-ue`: cell selection, RRC, EPS NAS attach with EPS AKA, AS security, SRB2 and DRBs on PDCP/RLC UM, TUN) that attach to nextgcore's EPC and pass a ping through the PDN
- **Rel-17/18 features**: RedCap, XR 5QI, SNPN (NID), MINT/disaster-roaming, and UAV aerial-UE registration, integrated end-to-end with nextgcore (UE config knobs in `docs/configuration.md` → Rel-17/18 Feature Configuration; example configs in `config/features/`)
- **6G Exploration**: Research-grade AI/ML prototypes (ISAC, federated learning, semantic communication) — not conformant to any frozen 3GPP spec (Rel-20 is in discussion; disabled by default)
- **Cryptography**: Milenage, SNOW3G, ZUC, AES-based NEA/NIA algorithms, ECIES
- **Protocol Support**: 5GS NAS, EPS NAS, NGAP and NRPPa (ASN.1 APER), S1AP (APER), NR RRC and E-UTRA RRC (ASN.1 UPER), RLC, PDCP, GTP-U, SCTP. The ASN.1 codecs are generated at build time from the 3GPP schemas in `tools/` by the [NextgCoreLab/asn1-codegen](https://github.com/NextgCoreLab/asn1-codegen) compiler (pinned by rev in `Cargo.toml`)
- **SCTP backends**: an in-process userspace `sctp-proto` over UDP (default; talks to nextgcore's matching backend) or kernel SCTP via libsctp (`--features kernel-sctp`, `sctp_backend: kernel`) for a real AMF/MME
- **Radio Simulation**: RLS protocol for UE-gNB communication over UDP
- **ML Inference**: ONNX Runtime (`ort`) integration available; no models ship, so the operational AI paths use TF-IDF / mean-pooling / linear-extrapolation fallbacks
- **Async Runtime**: Built on Tokio for high-performance concurrent operations

## Quickstart

### Option 1: Docker (Recommended)

The fastest way to get started is using Docker with the nextgcore 5G core.

```bash
# 1. Run the full matched-sim E2E: our gNB + UE vs the nextgcore 5GC.
#    e2e.sh runs a disk preflight, builds the images, then the E2E suite
#    (registration + PDU session + data-plane ping over the GTP-U tunnel).
cd ../nextgcore/docker/rust
./e2e.sh                          # add --keep to leave the stack running

# Or bring the stack up manually (the gNB + UE ship in the SAME compose file):
docker compose -f docker-compose.yml up -d

# 2. Verify UE registration and the PDU session
docker logs nextgsim-ue 2>&1 | grep -E "Registration Accept: UE is now|is now ACTIVE"
# Expected output (abridged):
# Received Registration Accept: UE is now MM[RM-REGISTERED, CM-CONNECTED, 5GMM-REGISTERED.NORMAL-SERVICE, ...]
# PDU Session 1 is now ACTIVE (IPv4: Some(10.45.0.2))
```

### Option 2: Build from Source

```bash
# Prerequisites: rustup. rust-toolchain.toml pins the toolchain (1.95.0),
# so rustup installs it on the first cargo command.

# Build all binaries: nr-gnb, nr-ue, nr-cli, nr-loadtest, lte-enb, lte-ue
cargo build --release

# Run gNB on the host against the nextgcore Docker AMF (172.23.0.5:38412).
# Set ngap_ip in config/gnb-local.yaml to a host address that reaches it.
./target/release/nr-gnb -c config/gnb-local.yaml

# Run UE (in another terminal; needs CAP_NET_ADMIN for its TUN interface)
sudo ./target/release/nr-ue -c config/ue-local.yaml

# Inspect or drive them
./target/release/nr-cli --dump          # lists the running nodes, e.g. gnb, ue0
./target/release/nr-cli ue0 -e status
```

`config/gnb.yaml` / `config/ue.yaml` use the nextgcore Docker addresses
(172.23.0.0/24) instead.

### Option 3: Run Tests

```bash
# Run all tests (unit + integration)
cargo test

# Run specific test suites
cargo test -p nextgsim-integration-tests --test ai_integration
cargo test -p nextgsim-integration-tests --test e2e_scenario
cargo test -p nextgsim-integration-tests --test paging_mt_service_request
cargo test -p nextgsim-integration-tests --test rrc_inactive_resume
```

## Docker Deployment

### Network Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Docker Network: rust_core                        │
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
| `config/gnb.yaml`, `config/ue.yaml` | gNB and UE for the nextgcore Docker network (172.23.0.0/24) |
| `config/gnb-docker.yaml`, `config/ue-docker.yaml` | What `docker/docker-compose.yml` mounts |
| `config/gnb-local.yaml`, `config/ue-local.yaml` | gNB and UE on the host (RLS over 127.0.0.1) against the Docker AMF |
| `config/gnb-kernel.yaml` | Kernel-SCTP N2 (`sctp_backend: kernel`) |
| `config/enb.yaml`, `config/lte-ue.yaml` | LTE eNB and UE for nextgcore's EPC network (172.24.0.0/24) |
| `config/features/*.yaml` | Rel-17/18 feature scenarios (RedCap, XR, MINT, SNPN, UAV) |
| `docker/docker-compose.yml` | Simulator-only gNB + UE, joined to an already-running nextgcore stack's `rust_core` network (the full 5GC + gNB + UE E2E stack lives in `../nextgcore/docker/rust/docker-compose.yml`) |

### Environment Variables

```bash
# gNB container
RUST_LOG=info,nextgsim_gnb=debug,nextgsim_ngap=debug

# UE container
RUST_LOG=info,nextgsim_ue=debug,nextgsim_nas=debug
```

### LTE eNB and UE containers

`Dockerfile.enb-local` (`lte-enb`, built with `--features kernel-sctp`) and
`Dockerfile.lte-ue-local` (`lte-ue`) bake `config/enb.yaml` and
`config/lte-ue.yaml`, which target nextgcore's EPC compose network
(172.24.0.0/24). A compose file adjusts them through `NEXTGSIM_*` variables,
applied before validation so a bad value is refused like a bad file:

| Variable | Node | Overrides |
|---|---|---|
| `NEXTGSIM_ENB_MME_ADDR` | eNB | `mme_configs` (comma-separated `ip` or `ip:port`, port 36412 by default) |
| `NEXTGSIM_ENB_IP` | eNB | `s1ap_ip` and `radio.link_ip` |
| `NEXTGSIM_ENB_GTP_IP` | eNB | `gtp_ip` (S1-U; defaults to `s1ap_ip`) |
| `NEXTGSIM_ENB_ID` | eNB | `enb_id` (decimal or `0x` hex) |
| `NEXTGSIM_PLMN` | both | `plmn`, written `MCC-MNC` |
| `NEXTGSIM_TAC` | eNB | `tac` |
| `NEXTGSIM_UE_ENB_ADDR` | UE | `enb_search_list` |
| `NEXTGSIM_UE_IMSI` / `_KEY` / `_OPC` / `_APN` | UE | the USIM (`subscriber`); key and OPc as 32 hex digits |
| `NEXTGSIM_UE_TUN_NAME` | UE | `tun.name` |

The UE puts the default bearer's PDN address on a TUN (`lteuetun5`), which
needs `cap_add: [NET_ADMIN]` and `devices: [/dev/net/tun:/dev/net/tun]`.
`docker exec <ue> lte-ue-ping-check <target> [count]` then waits for that
address and asserts a 0%-loss ping through the PDN (exit 0 pass, 1 no bearer,
2 loss). `docker exec <ue> lte-ue-detach [--switch-off]` makes the UE detach
(TS 24.301 §5.5.2.2) and exits 0 once it is EMM-DEREGISTERED.

## Workspace Structure

```
nextgsim/
├── nextgsim-gnb/       # gNB binary (nr-gnb) - 5G base station simulator
├── nextgsim-ue/        # UE binary (nr-ue) - 5G user equipment simulator
├── nextgsim-enb/       # eNB binary (lte-enb) - LTE base station: S1-MME, S1-U GTP-U, E-UTRA cell over RLS
├── nextgsim-lte-ue/    # LTE UE binary (lte-ue) - cell selection, RRC, EPS NAS, DRBs and TUN
├── nextgsim-cli/       # nr-cli (control interface) and nr-loadtest (multi-UE load test)
├── nextgsim-common/    # Common types (PLMN, TAI, SUPI), configs, CLI server
├── nextgsim-nas/       # 5GS NAS (TS 24.501) encoding/decoding
├── nextgsim-eps-nas/   # EPS NAS (TS 24.301 EMM/ESM, UE side)
├── nextgsim-ngap/      # NGAP and NRPPa (generated ASN.1 APER codecs)
├── nextgsim-s1ap/      # S1AP (TS 36.413, generated ASN.1 APER codec)
├── nextgsim-rrc/       # NR RRC (TS 38.331, generated ASN.1 UPER codec)
├── nextgsim-lte-rrc/   # E-UTRA RRC (TS 36.331, generated ASN.1 UPER codec)
├── nextgsim-rlc/       # RLC (TS 38.322): TM/UM/AM, plus the TS 38.351 SRAP sublayer
├── nextgsim-pdcp/      # PDCP (TS 38.323 / 36.323): SRB and DRB entities, AS security, SDAP header
├── nextgsim-crypto/    # Cryptographic algorithms (Milenage, TUAK, SNOW3G, ZUC, AES, ECIES, PQC)
├── nextgsim-rls/       # Radio Link Simulation protocol
├── nextgsim-gtp/       # GTP-U tunneling protocol
├── nextgsim-sctp/      # SCTP transport (userspace and kernel backends)
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

The simulator includes a set of 6G/AI research prototypes (disabled by default and not conformant to any frozen 3GPP spec — Rel-20/6G has no stage-3 wire spec). The 3GPP documents referenced below are design inspiration, not a conformance claim:

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

The full key reference is [docs/configuration.md](docs/configuration.md). Two
format points trip people up: `nci` is a YAML integer (`0x000000010`, not a
quoted string), and `key`, `op` and `amf` are byte arrays, not hex strings.
Both examples below are cut down from the shipped `config/gnb.yaml` and
`config/ue.yaml` and parse as they stand.

### gNB Configuration

```yaml
nci: 0x000000010
gnb_id_length: 32
plmn:
  mcc: 999
  mnc: 70
  long_mnc: false
tac: 1
nssai:
  - sst: 1
    sd: null
amf_configs:
  - address: 172.23.0.5   # nextgcore AMF: an IP or a hostname
    port: 38412
sctp_backend: userspace   # or kernel (needs --features kernel-sctp)
link_ip: 172.23.0.100
ngap_ip: 172.23.0.100
gtp_ip: 172.23.0.100
ignore_stream_ids: false
upf_addr: 172.23.0.7
upf_port: 2152
```

### UE Configuration

```yaml
supi:
  supi_type: Imsi
  value: "999700000000001"
protection_scheme: 0
home_network_public_key_id: 0
home_network_public_key: []
routing_indicator: "0000"
hplmn:
  mcc: 999
  mnc: 70
  long_mnc: false
key: [0x46, 0x5B, 0x5C, 0xE8, 0xB1, 0x99, 0xB4, 0x9F, 0xAA, 0x5F, 0x0A, 0x2E, 0xE2, 0x38, 0xA6, 0xBC]
op: [0xE8, 0xED, 0x28, 0x9D, 0xEB, 0xA9, 0x52, 0xE4, 0x28, 0x3B, 0x54, 0xE8, 0x8E, 0x6D, 0x83, 0x4D]
op_type: Opc
amf: [0x80, 0x00]
imei: "356938035643803"
imei_sv: null
supported_algs: { nia1: true, nia2: true, nia3: true, nea1: true, nea2: true, nea3: true }
gnb_search_list:
  - "172.23.0.100"
sessions:
  - type: Ipv4
    s_nssai: { sst: 1, sd: null }
    apn: "internet"
    is_emergency: false
configured_nssai:
  slices:
    - { sst: 1, sd: null }
tun_name: null
```

## Testing

### Integration test suites

The `tests/` crate (`nextgsim-integration-tests`) declares one `[[test]]` target
per file in `tests/src/`; `tests/Cargo.toml` is the list. They include
registration, PDU sessions, the user plane, multi-UE, paging with a
mobile-terminated Service Request, RRC_INACTIVE resume, the RRC handshake,
conditional handover, NTN SIB19 timing, the gNB CLI, and the AI/6G task
integration. A few need a feature to build: `drb_pdcp_user_plane`
(`drb-pdcp`), `up_security_user_plane` (`up-security`), `sdap_user_plane`
(`sdap-dataplane`), `ranging_report_e2e` and `sidelink_pc5_e2e` (`sidelink`),
and the 6G task tests (`6g`).

### Running Tests

```bash
# All tests (default features)
cargo test --workspace

# With logging
RUST_LOG=debug cargo test -p nextgsim-integration-tests -- --nocapture

# Specific test
cargo test -p nextgsim-integration-tests test_nwdaf_handover_recommendation

# One integration suite
cargo test -p nextgsim-integration-tests --test e2e_scenario

# A feature-gated suite
cargo test -p nextgsim-integration-tests --features sidelink --test sidelink_pc5_e2e
```

CI (`.github/workflows/ci.yml`) also runs each feature-gated surface in its
own job; see [docs/ci-and-release.md](docs/ci-and-release.md).

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
3. Write unit tests in the crate
4. Add integration tests in `tests/src/`, with a `[[test]]` entry in `tests/Cargo.toml`
5. If it is behind a cargo feature, add a CI job for it in `.github/workflows/ci.yml`

The `Dockerfile.*-local` images copy prebuilt binaries from `binaries/`, so a
new library crate needs no Dockerfile change.

## Documentation

- [CHANGELOG.md](CHANGELOG.md) — release notes and version history
- [docs/configuration.md](docs/configuration.md) — every gNB, UE, eNB and LTE UE configuration key
- [docs/cli-usage.md](docs/cli-usage.md) — the binaries, their flags, and `nr-cli`
- [docs/ci-and-release.md](docs/ci-and-release.md) — CI pipeline and release process
- [docs/architecture.md](docs/architecture.md) · [docs/nextgcore-integration.md](docs/nextgcore-integration.md) — design and the matched-sim E2E with nextgcore
- `docs-book/` — the mdBook (`mdbook build docs-book` writes `docs/book/`), published with the rest of `docs/` on GitHub Pages at each release

## Contributing

We welcome contributions! Here's how to get started:

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/NextgCoreLab/nextgsim.git
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
- **sctp-proto** - Pure Rust SCTP (0.6.0, patched to carry any PPID; taken from nextgcore's `src/vendor/sctp-proto` via `[patch.crates-io]`)
- **tun-rs** - TUN interface for user plane
- **asn1-codecs**, **asn1_codecs_derive**, **asn1-compiler** - ASN.1 PER/UPER runtime and build-time compiler, from [NextgCoreLab/asn1-codegen](https://github.com/NextgCoreLab/asn1-codegen)

### AI/ML
- **ort** - ONNX Runtime for ML inference (`load-dynamic`: the ONNX Runtime library is loaded at run time, not linked)
- **ndarray** - N-dimensional tensor operations
- **half** - Half-precision (FP16) floating point support

## References

### 3GPP Specifications
- [3GPP TS 24.501](https://www.3gpp.org/DynaReport/24501.htm) - NAS protocol for 5G
- [3GPP TS 38.413](https://www.3gpp.org/DynaReport/38413.htm) - NGAP specification
- [3GPP TS 38.331](https://www.3gpp.org/DynaReport/38331.htm) - RRC specification
- [3GPP TS 24.301](https://www.3gpp.org/DynaReport/24301.htm) - EPS NAS
- [3GPP TS 36.413](https://www.3gpp.org/DynaReport/36413.htm) - S1AP
- [3GPP TS 36.331](https://www.3gpp.org/DynaReport/36331.htm) - E-UTRA RRC
- [3GPP TS 33.501](https://www.3gpp.org/DynaReport/33501.htm) - 5G Security architecture
- [3GPP TS 23.288](https://www.3gpp.org/DynaReport/23288.htm) - NWDAF
- [3GPP TR 22.837](https://www.3gpp.org/DynaReport/22837.htm) - ISAC
- [3GPP TR 23.700-80](https://www.3gpp.org/DynaReport/23700-80.htm) - AI/ML for 5G

### Related Projects
- [UERANSIM](https://github.com/aligungr/UERANSIM) - Original C++ implementation
- [nextgcore](https://github.com/NextgCoreLab/nextgcore) - Companion 5G/EPC core network

## License

GNU AGPL-3.0 - See [LICENSE](LICENSE) for details.
