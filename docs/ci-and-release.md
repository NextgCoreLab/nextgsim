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

> **Docker jobs are opt-in.** `docker-build` (and the dependent `docker-e2e`/`docker-e2e-epc`) run **only on manual `workflow_dispatch`** (Actions tab → Run workflow), not on push/PR — they recompile both workspaces in a container and need a cross-repo checkout, so they are too heavy/fragile for every commit. The fast gate (Check/Format/Clippy/Test) protects every push/PR.

The end-to-end data-plane validation lives in the **nextgcore** repo
(`nextgcore/docker/rust/e2e.sh`), which runs this simulator's gNB + UE against the 22-NF core.

### `pages.yml`

Deploys `docs/` to GitHub Pages **only when a release is published** (`on: release: [published]`,
plus `workflow_dispatch` for manual re-deploys) — not on every `docs/**` push.

## Branch triggers

`ci.yml` runs on push + PR to the default release branch **`main`** and the development branch
**`first_implementation`** (`branches: [first_implementation, main]`). `pages.yml` is not
branch-triggered — it deploys the docs site when a release is published.

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
4. Commit (`Signed-off-by: Murat Parlakisik <parlakisik@gmail.com>`), tag `X.Y.Z`, push the tag.
5. `gh release create X.Y.Z --repo NextgCoreLab/nextgsim --title "NextgSim X.Y.Z" --notes-file …`
   using the CHANGELOG section as notes.

> Keep the honest framing: matched-sim validated (not real-peer / not certified); 6G crates are
> research prototypes, disabled by default.
