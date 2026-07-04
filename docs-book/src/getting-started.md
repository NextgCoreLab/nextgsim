# Getting Started

NextGSim is a pure-Rust 5G UE and gNB simulator implementing a 3GPP Rel-15 baseline with selected Rel-17/18 features. Rel-20 (6G) items are **non-normative research prototypes** — no frozen 3GPP Stage-3 spec exists — and are disabled by default. NextGSim is validated against its companion core, NextGCore, in end-to-end tests; it is **not** third-party certified, and no conformance is claimed beyond what those tests exercise.

## Prerequisites

NextGSim is pure Rust — no C toolchain or external protocol libraries are required for the default build.

| Tool | Version | Purpose |
|------|---------|---------|
| Rust (stable) | ≥ 1.75 | Build toolchain |
| Docker + docker compose | ≥ 24 | Container runs and E2E testing with NextGCore |

```bash
# macOS (Homebrew)
brew install rustup
rustup toolchain install stable
```

## Build

```bash
git clone https://github.com/NextgCoreLab/nextgsim.git
cd nextgsim

cargo build --workspace            # debug
cargo build --workspace --release  # release

# Resulting binaries
ls target/release/nr-*             # nr-gnb, nr-ue, nr-cli
```

Build a single component:

```bash
cargo build -p nextgsim-ue    # UE simulator
cargo build -p nextgsim-gnb   # gNB simulator
```

## Run a gNB and UE Against a Core

Sample configurations live in `config/`:

- `gnb.yaml`, `ue.yaml` — defaults, pointed at a NextGCore stack (PLMN 999-70)
- `gnb-local.yaml`, `ue-local.yaml` — local (non-Docker) addressing
- `gnb-docker.yaml`, `ue-docker.yaml` — used by the Docker Compose deployment
- `gnb-kernel.yaml` — kernel-SCTP variant (see below)
- `config/features/` — per-feature scenario configs (RedCap, XR, MINT, SNPN accept/reject, UAV allow/deny)

```bash
# Terminal 1: gNB (connects to the AMF configured in gnb.yaml)
./target/release/nr-gnb --config config/gnb.yaml

# Terminal 2: UE
./target/release/nr-ue --config config/ue.yaml

# Terminal 3: interactive CLI
./target/release/nr-cli
```

Key settings to align with your core (comments in `config/gnb.yaml` / `config/ue.yaml`):

- **PLMN / TAC / NSSAI** in `gnb.yaml` must match the AMF configuration (default PLMN 999-70, TAC 1, SST 1).
- **`amf_configs`** points at the AMF NGAP endpoint (default `172.23.0.5:38412`, the NextGCore AMF container).
- **UE subscriber** (`supi`, `key`, `op`/`op_type`) must be provisioned in the core's subscriber database. Default IMSI is `999700000000001` with well-known test K/OPc values.
- **SUCI protection** (`protection_scheme`) defaults to the null scheme; Profile A/B ECIES key material (TS 33.501 Annex C test vectors, per the config comments) is included commented-out.

### SCTP backend note

The default NGAP transport is `sctp_backend: userspace` — an in-process SCTP-over-UDP that interoperates with the NextGCore simulator AMF but **not** a real AMF (the wire is UDP). To associate with a real kernel-SCTP AMF, use a Linux build with the gNB's `kernel-sctp` cargo feature and set `sctp_backend: kernel` (see `config/gnb-kernel.yaml`); otherwise the connection fails loudly at runtime.

## Multi-UE Simulation

```bash
# Simulate 10 UEs with auto-incrementing IMSI
./target/release/nr-ue --config config/ue.yaml --num-ues 10

# With debug logging
RUST_LOG=debug ./target/release/nr-gnb --config config/gnb.yaml
```

## Docker Compose (Full RAN)

`docker/docker-compose.yml` brings up `nextgsim-gnb` and `nextgsim-ue` containers, mounting `config/gnb-docker.yaml` and `config/ue-docker.yaml`, pre-networked to reach a NextGCore AMF/UPF stack:

```bash
docker compose up          # start gNB + UE
docker compose up --build  # rebuild images (e.g. after editing configs)
```

## Lean UE Builds (Feature-Gated 6G Crates)

The UE binary's 6G/AI client crates (`nextgsim-she`, `nextgsim-nwdaf`, `nextgsim-nkef`, `nextgsim-isac`, `nextgsim-agent`, `nextgsim-fl`, `nextgsim-semantic`) are **optional dependencies** — they are compiled in only when explicitly enabled, so the default build is already the lean 5G-only UE:

```bash
# Default: minimal 5G-only UE (no 6G crates compiled)
cargo build --release -p nextgsim-ue

# Opt into specific 6G research prototypes (crate-named feature flags)
cargo build --release -p nextgsim-ue \
  --features "nextgsim-isac,nextgsim-agent,nextgsim-fl"
```

These 6G features are research explorations only: enabling them compiles prototype code, not standardized protocol behavior.

## Testing and E2E with NextGCore

```bash
cargo test --workspace          # all unit/integration tests
cargo test -p nextgsim-ue       # UE only
cargo test -- --nocapture       # with log output
```

End-to-end tests require a running NextGCore stack:

```bash
# 1. Start the core (from the sibling nextgcore checkout)
cd ../nextgcore/docker/rust && docker compose up -d

# 2. Run the E2E suite
cd ../../nextgsim
cargo test -p nextgsim-tests --features e2e
```

The E2E suite validates the full flow — UE registration through the gNB to the core, PDU session establishment, and GTP-U user-plane traffic — against NextGCore. This is matched-simulator validation between the two projects, not interoperability certification against third-party equipment.

## Contributing Checklist

Before submitting changes:

```bash
cargo fmt --all
cargo clippy --workspace -- -D warnings
cargo test --workspace
cargo audit    # cargo install cargo-audit
```

Each protocol module carries a doc comment citing its 3GPP spec (e.g. TS 38.331 NR RRC, TS 38.413 NGAP, TS 24.501 5G NAS); cite the relevant TS section in commits and PRs. See `CONTRIBUTING.md` for the full process. License: AGPL-3.0.
