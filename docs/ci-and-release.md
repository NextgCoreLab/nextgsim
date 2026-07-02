# CI/CD and Release Process

## GitHub Actions workflows (`.github/workflows/`)

### `ci.yml`

| Job | Command | Gates on |
|-----|---------|----------|
| **Check** | `cargo check --workspace` | compile errors |
| **Format** | `cargo fmt --all -- --check` | rustfmt drift (fails on any diff) |
| **Clippy** | `cargo clippy --workspace` | errors only (warnings shown, non-blocking) |
| **Test** | `cargo test --workspace` | any failing test |
| **Docker Build** | builds the Rust binaries, then the gNB and UE images; verifies images | build failures |

The end-to-end data-plane validation lives in the **nextgcore** repo
(`nextgcore/docker/rust/e2e.sh`), which runs this simulator's gNB + UE against the 22-NF core.

### `pages.yml`

Deploys `docs/` to GitHub Pages.

## Branch triggers

Both workflows run on the default release branch **`main`** and on the development branch
**`first_implementation`** (`branches: [first_implementation, main]`).

## Local pre-push gate (mirror of CI)

```bash
cargo fmt --all -- --check
cargo check --workspace
cargo clippy --workspace
cargo test --workspace
```

Full data-plane E2E (from the nextgcore repo): `cd ../nextgcore/docker/rust && ./e2e.sh`.

## Cutting a release

1. Green CI + matched-sim E2E (via nextgcore `e2e.sh`).
2. Bump versions in the workspace `Cargo.toml`(s). Current: `0.1.0`.
3. Promote `## [Unreleased]` in `CHANGELOG.md` to a dated `## [X.Y.Z]`; add a fresh Unreleased.
4. Commit (`Signed-off-by: Murat Parlakisik <parlakisik@gmail.com>`), tag `vX.Y.Z`, push the tag.
5. `gh release create vX.Y.Z --repo NextgCoreLab/nextgsim --title "NextgSim vX.Y.Z" --notes-file …`
   using the CHANGELOG section as notes.

> Keep the honest framing: matched-sim validated (not real-peer / not certified); 6G crates are
> research prototypes, disabled by default.
