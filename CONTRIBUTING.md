# Contributing to NextGSim

NextGSim is a 5G UE and gNB simulator in Rust targeting 3GPP Rel-15 through Rel-20 (6G research).

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Rust | stable (≥1.75) | Build toolchain |
| Docker + docker-compose | ≥24 | E2E testing with NextGCore |

```bash
# macOS (Homebrew)
brew install rustup
rustup toolchain install stable
```

## Building

```bash
cd nextgsim
cargo build --workspace            # debug
cargo build --workspace --release  # release
```

To build a single component:

```bash
cargo build -p nextgsim-ue   # UE simulator
cargo build -p nextgsim-gnb  # gNB simulator
```

## Testing

```bash
cargo test --workspace                 # all tests
cargo test -p nextgsim-ue             # UE only
cargo test -- --nocapture             # with log output
```

E2E tests require a running NextGCore stack:

```bash
cd ../nextgcore/docker/rust && docker compose up -d
cd ../../nextgsim
cargo test -p nextgsim-tests --features e2e
```

## Code Style

```bash
cargo fmt --all
cargo clippy --workspace -- -D warnings
```

Pre-commit checklist:
- [ ] `cargo fmt --all` — no formatting changes
- [ ] `cargo clippy --workspace -- -D warnings` — zero warnings
- [ ] `cargo test --workspace` — all tests pass
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
│       ├── prose.rs    # ProSe PC5 (Rel-17)
│       ├── uav.rs      # UAV context (Rel-18)
│       ├── daps.rs     # DAPS (Rel-16)
│       └── ...
├── nextgsim-nas/       # Shared NAS codec
├── nextgsim-ngap/      # NGAP ASN.1 codec (auto-generated)
├── nextgsim-common/    # Config, types, utilities
├── nextgsim-tests/     # Integration + E2E tests
└── tools/              # ASN.1 grammars
```

## Simulated Releases

| Release | Features |
|---------|---------|
| Rel-15 | 5G-NR baseline, SA registration, PDU sessions |
| Rel-16 | DAPS handover, URLLC, RedCap, V2X PC5 |
| Rel-17 | MBS, ProSe, NTN, SNPN, UAV identity |
| Rel-18 | XR traffic, Ambient IoT, RedCap R18, MINT |
| Rel-20 | ISAC, Federated Learning, Semantic comms, Agent AI |

## 3GPP Specification References

Each module includes a doc comment with the spec reference. When adding protocol logic, always cite:

- **TS 38.331** — NR RRC
- **TS 38.413** — NGAP
- **TS 24.501** — 5G NAS
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
2. Create a feature branch from `first_implementation`
3. Implement your change with tests
4. Ensure all CI checks pass
5. Open a PR referencing the 3GPP TS item

## License

Apache-2.0. See [LICENSE](LICENSE).
