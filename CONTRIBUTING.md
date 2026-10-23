# Contributing to NextGSim

NextGSim is a 5G UE and gNB simulator in Rust, with an LTE eNB and LTE UE, implementing a 3GPP Rel-15 baseline with selected Rel-16/17/18 features and 6G exploration prototypes. 3GPP Rel-20 (6G) has no frozen stage-3 spec, so the 6G items are research-only (informed by TR 22.870 use cases) and disabled by default.

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Rust | pinned by `rust-toolchain.toml` (1.95.0) | Build toolchain; rustup installs it on the first `cargo` command |
| Docker + docker compose | ≥24 | E2E testing with NextGCore |
| libsctp (`libsctp-dev`) | any | Only for `--features kernel-sctp` (the LTE eNB, or a gNB against a real AMF), Linux only |

```bash
# macOS (Homebrew)
brew install rustup
rustup-init
```

## Building

```bash
cd nextgsim
cargo build --workspace            # debug
cargo build --workspace --release  # release
```

To build a single component:

```bash
cargo build -p nextgsim-ue   # UE simulator (nr-ue)
cargo build -p nextgsim-gnb  # gNB simulator (nr-gnb)
cargo build -p nextgsim-enb --features kernel-sctp   # LTE eNB (lte-enb), Linux
cargo build -p nextgsim-lte-ue                       # LTE UE (lte-ue)
```

## Testing

```bash
cargo test --workspace                 # all tests
cargo test -p nextgsim-ue             # UE only
cargo test -- --nocapture             # with log output
```

The `tests/` crate (`nextgsim-integration-tests`) runs the gNB and UE tasks in
one process against a mock AMF; some of its suites need a feature
(`--features drb-pdcp`, `up-security`, `sdap-dataplane`, `sidelink`, `6g`). The
end-to-end suite against a real core runs from the nextgcore repo:

```bash
cd ../nextgcore/docker/rust
./e2e.sh            # builds both repos' images, then registration + PDU session + ping
```

## Code Style

```bash
cargo fmt --all
cargo clippy --workspace --all-targets
```

Pre-commit checklist:
- [ ] `cargo fmt --all -- --check` — no formatting changes
- [ ] `cargo clippy --workspace --all-targets` — no new warnings in the code you touched
- [ ] `cargo test --workspace` — all tests pass
- [ ] The feature-gated CI job for any feature your change touches (see `docs/ci-and-release.md`)
- [ ] `cargo audit` — no unpatched advisories

## Security Auditing

```bash
cargo install cargo-audit
cargo audit
```

## Project Layout

```
nextgsim/
├── nextgsim-gnb/       # gNB (base station) simulator
│   └── src/
│       ├── ngap/       # NGAP (N2 interface to AMF)
│       ├── rrc/        # NR RRC
│       ├── gtp/        # GTP-U data plane
│       ├── daps.rs     # DAPS handover (Rel-16)
│       ├── mbs_ngap.rs # MBS NGAP (Rel-17)
│       └── ...
├── nextgsim-ue/        # UE simulator
│   └── src/
│       ├── nas/        # 5G NAS (N1 interface)
│       ├── rrc/        # NR RRC
│       ├── sidelink/   # PC5-S, discovery, relay (behind the sidelink feature)
│       ├── uav.rs      # UAV context (Rel-18)
│       ├── daps.rs     # DAPS (Rel-16)
│       └── ...
├── nextgsim-enb/       # LTE eNB (lte-enb)
├── nextgsim-lte-ue/    # LTE UE (lte-ue)
├── nextgsim-nas/       # 5GS NAS codec
├── nextgsim-eps-nas/   # EPS NAS (UE side)
├── nextgsim-ngap/      # NGAP + NRPPa codecs (generated at build time)
├── nextgsim-s1ap/      # S1AP codec (generated)
├── nextgsim-rrc/       # NR RRC codec (generated)
├── nextgsim-lte-rrc/   # E-UTRA RRC codec (generated)
├── nextgsim-common/    # Config, types, utilities
├── tests/              # Integration tests (nextgsim-integration-tests)
└── tools/              # 3GPP ASN.1 schemas and extract_asn1.py
```

## Simulated Releases

| Release | Features |
|---------|---------|
| Rel-15 | 5G-NR baseline, SA registration, PDU sessions, paging and Service Request; LTE attach (E-UTRA RRC, S1AP, EPS NAS) |
| Rel-16 | DAPS handover, URLLC, V2X PC5, RACS (UE radio capability IDs) |
| Rel-17 | RedCap, MBS, ProSe and L2 UE-to-Network relay, NTN (SIB19), SNPN, UAV identity |
| Rel-18 | XR traffic, Ambient IoT, RedCap R18, MINT |
| Rel-20* | ISAC, Federated Learning, Semantic comms, Agent AI |

\* Rel-20 (6G) research exploration only — no frozen 3GPP stage-3 spec exists; these features are non-normative prototypes, disabled in the default config.

## 3GPP Specification References

Each module includes a doc comment with the spec reference. When adding protocol logic, always cite:

- **TS 38.331** — NR RRC
- **TS 38.413** — NGAP
- **TS 24.501** — 5G NAS
- **TS 36.413** — S1AP
- **TS 36.331** — E-UTRA RRC
- **TS 24.301** — EPS NAS
- **TS 23.256** — UAV
- **TS 23.304** — ProSe

## Commit Messages

```
feat(gnb): add MBS NGAP session start procedure (TS 38.413 §8.21)
fix(ue): apply NAS security after SecurityModeComplete
test(prose): add relay selection RSRP unit test
```

- **Never** include `Co-Authored-By: Claude` or any AI attribution
- Sign commits: `Signed-off-by: Your Name <email>`

## Pull Request Process

1. Fork the repository
2. Create a feature branch from `main`
3. Implement your change with tests
4. Ensure all CI checks pass
5. Open a PR referencing the 3GPP TS item

## License

AGPL-3.0. See [LICENSE](LICENSE).
