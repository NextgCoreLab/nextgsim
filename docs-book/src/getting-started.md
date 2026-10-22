# Getting Started

NextGSim is a pure-Rust 5G UE and gNB simulator, with an LTE eNB and LTE UE, implementing a 3GPP Rel-15 baseline with selected Rel-16/17/18 features. Rel-20 (6G) items are **non-normative research prototypes** — no frozen 3GPP Stage-3 spec exists — and are disabled by default. NextGSim is validated against its companion core, NextGCore, in end-to-end tests; it is **not** third-party certified, and no conformance is claimed beyond what those tests exercise.

## Prerequisites

NextGSim is pure Rust — no C toolchain or external protocol libraries are required for the default build.

| Tool | Version | Purpose |
|------|---------|---------|
| Rust | pinned by `rust-toolchain.toml` (1.95.0) | Build toolchain; rustup installs it on the first `cargo` command |
| Docker + docker compose | ≥ 24 | Container runs and E2E testing with NextGCore |
| libsctp (`libsctp-dev`) | any | Only for `--features kernel-sctp` (Linux) |

```bash
# macOS (Homebrew)
brew install rustup
rustup-init
```

## Build

```bash
git clone https://github.com/NextgCoreLab/nextgsim.git
cd nextgsim

cargo build --workspace            # debug
cargo build --workspace --release  # release

# Resulting binaries
ls target/release/{nr,lte}-*       # nr-gnb, nr-ue, nr-cli, nr-loadtest, lte-enb, lte-ue
```

Build a single component:

```bash
cargo build -p nextgsim-ue    # UE simulator
cargo build -p nextgsim-gnb   # gNB simulator
```

## Run a gNB and UE Against a Core

Sample configurations live in `config/`:

- `gnb.yaml`, `ue.yaml` — defaults, pointed at a NextGCore stack (PLMN 999-70)
- `gnb-local.yaml`, `ue-local.yaml` — gNB and UE on the host (RLS over 127.0.0.1) against the Docker AMF
- `enb.yaml`, `lte-ue.yaml` — LTE eNB and UE for NextGCore's EPC stack
- `gnb-docker.yaml`, `ue-docker.yaml` — used by the Docker Compose deployment
- `gnb-kernel.yaml` — kernel-SCTP variant (see below)
- `config/features/` — per-feature scenario configs (RedCap, XR, MINT, SNPN accept/reject, UAV allow/deny)

```bash
# Terminal 1: gNB (connects to the AMF configured in gnb.yaml)
./target/release/nr-gnb --config config/gnb.yaml

# Terminal 2: UE (creates a TUN interface, so it needs CAP_NET_ADMIN)
sudo ./target/release/nr-ue --config config/ue.yaml

# Terminal 3: CLI
./target/release/nr-cli --dump        # gnb, ue0
./target/release/nr-cli ue0           # interactive
```

Key settings to align with your core (comments in `config/gnb.yaml` / `config/ue.yaml`):

- **PLMN / TAC / NSSAI** in `gnb.yaml` must match the AMF configuration (default PLMN 999-70, TAC 1, SST 1).
- **`amf_configs`** points at the AMF NGAP endpoint (default `172.23.0.5:38412`, the NextGCore AMF container).
- **UE subscriber** (`supi`, `key`, `op`/`op_type`) must be provisioned in the core's subscriber database. Default IMSI is `999700000000001` with well-known test K/OPc values. `key`, `op` and `amf` are YAML byte arrays and `supi` is a `{supi_type, value}` mapping; see the [Configuration Reference](configuration.md#configuration-file-format).
- **SUCI protection** (`protection_scheme`) defaults to the null scheme; Profile A/B ECIES key material (TS 33.501 Annex C test vectors, per the config comments) is included commented-out.

### SCTP backend note

The default NGAP transport is `sctp_backend: userspace` — an in-process SCTP-over-UDP that interoperates with the NextGCore simulator AMF but **not** a real AMF (the wire is UDP). To associate with a real kernel-SCTP AMF, use a Linux build with the gNB's `kernel-sctp` cargo feature and set `sctp_backend: kernel` (see `config/gnb-kernel.yaml`); otherwise the connection fails loudly at runtime.

## Multi-UE Simulation

```bash
# Simulate 10 UEs with auto-incrementing IMSI, 100 ms apart
sudo ./target/release/nr-ue --config config/ue.yaml -n 10 -t 100

# With debug logging
RUST_LOG=debug ./target/release/nr-gnb --config config/gnb.yaml
```

## Docker Compose (Full RAN)

`docker/docker-compose.yml` brings up `nextgsim-gnb` and `nextgsim-ue` containers, mounting `config/gnb-docker.yaml` and `config/ue-docker.yaml`, pre-networked to reach a NextGCore AMF/UPF stack:

```bash
cd docker
docker compose up -d       # start gNB + UE on the existing rust_core network
```

The images (`nextgsim/nr-gnb`, `nextgsim/nr-ue`) are built from
`Dockerfile.gnb-local` / `Dockerfile.ue-local`, which copy prebuilt binaries
from `binaries/`; nextgcore's `docker/rust/build.sh` builds both repos and the
images in one go.

## LTE eNB and UE

```bash
cargo build --release -p nextgsim-enb --features kernel-sctp   # Linux, libsctp
cargo build --release -p nextgsim-lte-ue
./target/release/lte-enb --config config/enb.yaml
sudo ./target/release/lte-ue --config config/lte-ue.yaml
```

They target NextGCore's EPC compose network (172.24.0.0/24). The container
images take `NEXTGSIM_*` environment overrides; see the
[Configuration Reference](configuration.md#environment-overrides-lte-containers).

## Lean UE Builds (Feature-Gated 6G Crates)

The 6G/AI crates are **optional dependencies** of both binaries — compiled in only when explicitly enabled, so the default build is already the lean 5G-only UE and gNB. The gNB has one umbrella feature, `6g`; the UE uses crate-named features:

```bash
# Default: minimal 5G-only UE (no 6G crates compiled)
cargo build --release -p nextgsim-ue

# Opt into specific 6G research prototypes (crate-named feature flags)
cargo build --release -p nextgsim-ue \
  --features "nextgsim-isac,nextgsim-agent,nextgsim-fl"

# gNB with every 6G task available
cargo build --release -p nextgsim-gnb --features 6g
```

These 6G features are research explorations only: enabling them compiles prototype code, not standardized protocol behavior.

## Testing and E2E with NextGCore

```bash
cargo test --workspace          # all unit/integration tests
cargo test -p nextgsim-ue       # UE only
cargo test -- --nocapture       # with log output
```

The end-to-end suite runs from the sibling nextgcore checkout. It builds both
repos and their images, starts the core with this repo's gNB and UE, and
asserts registration, the PDU session and a ping through the GTP-U tunnel:

```bash
cd ../nextgcore/docker/rust
./e2e.sh                   # add --keep to leave the stack running
```

The E2E suite validates the full flow — UE registration through the gNB to the core, PDU session establishment, and GTP-U user-plane traffic — against NextGCore. This is matched-simulator validation between the two projects, not interoperability certification against third-party equipment.

## Contributing Checklist

Before submitting changes:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets
cargo test --workspace
cargo audit    # cargo install cargo-audit
```

Each protocol module carries a doc comment citing its 3GPP spec (e.g. TS 38.331 NR RRC, TS 38.413 NGAP, TS 24.501 5G NAS); cite the relevant TS section in commits and PRs. See `CONTRIBUTING.md` for the full process. License: AGPL-3.0.
