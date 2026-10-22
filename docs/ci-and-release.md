# CI/CD and Release Process

## GitHub Actions workflows (`.github/workflows/`)

### `ci.yml`

Runs on push and pull request to `main`, nightly at 05:00 UTC, and on manual
dispatch.

**Fast gate** (every push and PR):

| Job | Command | Gates on |
|-----|---------|----------|
| **Check** | `cargo check --workspace` | compile errors |
| **Format** | `cargo fmt --all -- --check` | any rustfmt diff |
| **Clippy** | `cargo clippy --workspace` | errors (warnings are shown, not blocking) |
| **Test** | `cargo test --workspace` | any failing test. The job adds swap first, because the generated RRC codecs need the memory |

**Feature-gated jobs** (every push and PR). A default build compiles none of
these surfaces, so each has its own job that turns the feature on and runs its
tests (and usually clippy):

| Job | Feature(s) | Crates |
|-----|------------|--------|
| Event bus | `event-bus` | `nextgsim-common`, `nextgsim-gnb` (with `6g`), `nextgsim-ue` |
| Model registry | `model-registry` | `nextgsim-ai`, `nextgsim-nwdaf`, `nextgsim-semantic` |
| UUAA-MM | `uuaa-mm` | `nextgsim-ue` |
| 6G crates | `6g` | `nextgsim-gnb` |
| DRB PDCP | `drb-pdcp` | `nextgsim-pdcp`, `nextgsim-gnb`, `nextgsim-ue`, the end-to-end test |
| LTE eNB kernel SCTP | `kernel-sctp` (installs libsctp) | `nextgsim-enb` |
| UP security | `up-security` | `nextgsim-pdcp`, `nextgsim-gnb`, `nextgsim-ue`, the end-to-end test |
| SDAP data plane | `sdap-dataplane` | `nextgsim-pdcp`, `nextgsim-gtp`, `nextgsim-gnb`, `nextgsim-ue` |
| Sidelink facade | `sidelink` | `nextgsim-ue` |
| AS security | config-gated (`as_security_enabled`) | `nextgsim-pdcp`, `nextgsim-ue`, `nextgsim-gnb` |
| Sidelink ranging | `sidelink` | `ranging_report_e2e`, `nextgsim-ue` |
| Sidelink PC5 procedures | `sidelink` | `sidelink_pc5_e2e`, `nextgsim-ue`, `nextgsim-rrc`, `nextgsim-rlc` (SRAP) |
| Post-quantum crypto | `pqc` | `nextgsim-crypto`, including the FIPS 203/204 known-answer tests |

**Heavy jobs** (nightly and manual dispatch only):

| Job | What it does |
|-----|--------------|
| **Docker Build** | Builds `nr-gnb`, `nr-ue`, `lte-enb` (with `kernel-sctp`) and `lte-ue`, then the four images `nextgsim/nr-gnb`, `nextgsim/nr-ue`, `nextgsim/lte-enb`, `nextgsim/lte-ue`, and checks each exists |
| **Interop E2E** | Checks out nextgcore `main`, builds both workspaces with nextgcore's builder, starts the 5GC, and registers a real UE through this repo's gNB over N2 |

The nightly runs an hour after nextgcore's (04:00 UTC), so it tests a
nextgcore commit that its own nightly has just validated. nextgcore's nightly
in turn builds this repo's `main` and runs the deeper cross-repo suite: the
data-plane ping, the UE-flow gate (paging, idle uplink, de-registration,
network-triggered re-registration) and the EPC stage (LTE attach and ping). See
[nextgcore-integration.md](nextgcore-integration.md#nightly-ue-flow-gate-and-the-epc-stage).

### `pages.yml`

Deploys `docs/` to GitHub Pages **only when a release is published**
(`on: release: [published]`, plus `workflow_dispatch` for a manual re-deploy).
The mdBook under `docs-book/` builds into `docs/book/`, which is committed, so
rebuild it (`mdbook build docs-book`) in the same change as any edit under
`docs-book/src/`.

### `release.yml`

Manual dispatch only. It releases this repo as a commit pair that nextgcore's
cross-repo E2E validated:

| Input | Meaning |
|-------|---------|
| `version` | Version to release this repo as, for example `0.2.0`. Independent of nextgcore's |
| `validated_run_id` | The nextgcore CI run whose `validated-pair.json` authorises the release |
| `paired_version` | The nextgcore version released alongside, recorded in the tag message |
| `dry_run` | Validate and print everything without committing, tagging or publishing (default `true`) |

It fetches the manifest from that nextgcore run, refuses a sha that is no
longer on the branch or a tag that already exists, generates the release notes
from `git log <previous tag>..<sha>` grouped by commit prefix, bumps
`[workspace.package] version` in `Cargo.toml` and asserts the bump, then
commits, tags, pushes and runs `gh release create`.

## Local pre-push gate (mirror of the fast gate)

```bash
cargo fmt --all -- --check
cargo check --workspace
cargo clippy --workspace
cargo test --workspace
```

Add the feature jobs that touch your change, for example
`cargo test -p nextgsim-ue --features sidelink`. The full data-plane E2E runs
from the nextgcore repo: `cd ../nextgcore/docker/rust && ./e2e.sh`.

## Cutting a release

1. A green nightly in nextgcore whose `validated-pair.json` names the nextgsim
   commit to release.
2. Promote `## [Unreleased]` in `CHANGELOG.md` to a dated `## [X.Y.Z]` section
   and merge that. `release.yml` does not edit the CHANGELOG.
3. Run `release.yml` with `dry_run: true`, read the output, then again with
   `dry_run: false`. It bumps the version, tags `X.Y.Z` and publishes the GitHub
   release, which triggers `pages.yml`.

> Keep the honest framing: matched-sim validated (not real-peer, not
> certified); the 6G crates are research prototypes, disabled by default.
