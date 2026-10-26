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

Dispatched by hand to cut a release; a pushed version tag only publishes images
(see [Docker Hub images](#docker-hub-images)). It releases this repo as a commit
pair that nextgcore's cross-repo E2E validated:

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

### Docker Hub images

`release.yml` publishes every image in `docker/release-images.json` to Docker
Hub (nextgcore #420):

| Image | Built from | Binary |
|-------|------------|--------|
| `nextgsim-nr-gnb` | `Dockerfile.gnb-local` | `nr-gnb`, with `kernel-sctp` |
| `nextgsim-nr-ue` | `Dockerfile.ue-local` | `nr-ue`, plus `nr-cli` (`nr-ue-deregister` needs it) |
| `nextgsim-lte-enb` | `Dockerfile.enb-local` | `lte-enb`, with `kernel-sctp` |
| `nextgsim-lte-ue` | `Dockerfile.lte-ue-local`, stage `release` | `lte-ue` |

The binaries are built in `rust:1.95.0-bookworm`, matching the images' Debian
bookworm base, with the features nextgcore's `Dockerfile.builder` uses for the
validated E2E. The base images (`debian:bookworm-slim`, and `rust:1.95.0-bookworm` for the
build) and every action in `release.yml` are pinned by digest or commit SHA, with the tag in a
comment. To bump a base image, run `docker buildx imagetools inspect <tag>` and replace the
digest.

No published image carries a subscriber key. The gNB and UE images bake no config. The eNB
image bakes `config/enb.yaml`, which holds no secret. The LTE UE image is the `release` stage of
`Dockerfile.lte-ue-local`, without `config/lte-ue.yaml`, whose K/OPc are public test values.
Mount your own config at `/etc/nextgsim/lte-ue.yaml`, for example
`-v ./lte-ue.yaml:/etc/nextgsim/lte-ue.yaml:ro`. The default target, which local and E2E
builds use, still bakes the test config. The `Release image list` check fails if a published
stage COPYs a config holding `key`, `op` or `opc`. The published names replace the local names `nextgsim/nr-gnb`,
`nextgsim-gnb` and so on. nextgcore publishes the core images the same way.

**Trigger: a version tag only.** Publishing runs for:
- a pushed tag `X.Y.Z` or `X.Y.Z-<pre>`;
- the tag a dispatched `release.yml` run with `dry_run: false` has just
  created. That tag is pushed with `GITHUB_TOKEN`, which starts no workflow, so
  the same run publishes it.

Nothing is pushed from a PR, a branch, the nightly or a dry run. The tagged commit must also be on
`main` (`publish-gate` checks its ancestry and fails otherwise), so only reviewed code is
published. Recommended: a tag ruleset (Settings, Rules, Rulesets) that restricts who may create
or move `[0-9]*` tags, since any writer can otherwise push one.

**Setup.** Repository settings, Secrets and variables, Actions:

| Name | Kind | Value |
|------|------|-------|
| `DOCKERHUB_USERNAME` | secret | the Docker Hub account that pushes |
| `DOCKERHUB_TOKEN` | secret | a Docker Hub access token with Read & Write scope, not the password |
| `DOCKERHUB_NAMESPACE` | variable (optional) | the Docker Hub organisation. Without it, images go under `DOCKERHUB_USERNAME` |

Credentials live only in these secrets, never in a file. If either secret is
missing, the `publish-gate` job prints a "Docker Hub publish skipped" notice and
the publish jobs are skipped. The release itself still succeeds. To publish a
release made before the secrets existed, add them and re-run that run's
`Docker Hub publish gate` job. GitHub re-runs its dependent jobs with it.

**What is pushed.** Each image is `<namespace>/<image>:<version>`, built from
the tagged commit. After every versioned push has succeeded,
`<namespace>/<image>:latest` is moved to it. A pre-release never moves `latest`.

**First publish.** A pushed image name is public and cannot be cleanly
withdrawn, so prove the setup first. Point `DOCKERHUB_NAMESPACE` at a private
namespace and push a pre-release tag such as `0.2.0-rc.1`. Check the images,
then set the variable to the public organisation.

**The list cannot silently go stale.** `ci.yml`'s `Release image list` job runs
`docker/check-release-images.sh` on every push and PR. It fails when any of
these is in neither the manifest's `images` nor its `excluded` list (which gives
a reason for each entry):
- a cargo `[[bin]]` target;
- a tracked `Dockerfile*`;
- a `nextgsim` image a compose file or k8s manifest references.

The Open5GS exit-gate images (`docker/exit-gate/`) and the `nr-loadtest` tool
are excluded, with their reasons.

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
   With the Docker Hub secrets set, the same run then pushes the images
   ([Docker Hub images](#docker-hub-images)).

> Keep the honest framing: matched-sim validated (not real-peer, not
> certified); the 6G crates are research prototypes, disabled by default.
