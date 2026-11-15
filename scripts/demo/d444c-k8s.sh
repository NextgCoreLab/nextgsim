#!/usr/bin/env bash
# scripts/demo/d444c-k8s.sh: run demo D444C (nextgsim#479) as a Kubernetes Job on
# ANY cluster and check its table against the committed golden.
#
#   scripts/demo/d444c-k8s.sh [--namespace NS] [--image REF] [--context CTX]
#                             [--timeout SECS] [--run-id ID]
#
#   --namespace NS  the TEMPORARY namespace to create (default d444c-<run-id>). It
#                   must not exist: the script creates it, and deletes only a
#                   namespace it created.
#   --image REF     the nr-semantic-demo image, with a tag or a digest
#                   (registry/repo:tag or registry/repo@sha256:...). Default: the
#                   base's nextgsim/nr-semantic-demo:local (build it with
#                   docker/demo/Dockerfile.nr-semantic-demo; on kind,
#                   `kind load docker-image nextgsim/nr-semantic-demo:local`).
#   --context CTX   the kubeconfig context (passed to every kubectl call; the
#                   current context is never switched). KUBECONFIG is honoured.
#   --timeout SECS  how long to wait for the Job to finish (default 360; the
#                   Job's own activeDeadlineSeconds is 300).
#   --run-id ID     the artefact directory name (default <UTC time>-<pid>).
#
# Only kubectl is needed (kustomize via `kubectl apply -k`). What it does:
#   1. creates the namespace, labelled Pod Security `restricted` (enforce), so the
#      cluster itself refuses the pod if the Job's hardening ever regresses;
#   2. applies an overlay of deploy/k8s/demo-d444c-task-vs-raw/ (namespace and
#      image set by kustomize; the rendered manifest is kept);
#   3. waits for the Job to finish (Complete or Failed), then captures its log,
#      the Job and pod state and the namespace events to
#      ${D444C_ARTEFACTS_ROOT:-/tmp/eks-val/demos/d444c-task-vs-raw}/<run-id>/;
#   4. checks: the Job Complete; the WARN DEMO MODE ACTIVE banner and the
#      nextg_demo_enabled series in the log; the printed table (the log minus the
#      stderr lines) byte-equal to nextgsim-semantic/src/task_vs_raw/golden_seed444.txt,
#      the SAME file the unit test the_seed_444_table_matches_the_committed_golden_row_by_row
#      pins (one source of truth);
#   5. always deletes the namespace it created (EXIT trap; INT and TERM too).
#
# Exit 0: every check passed. 1: a check failed (each prints
# `DEMO CHECK FAILED: <why>`), or the namespace could not be deleted. 2: usage or
# a precondition (no kubectl, the namespace already exists). 130/143: interrupted
# (the namespace is still deleted).
#
# Tested without a cluster by scripts/demo/d444c-k8s-selftest.sh (a fake kubectl).
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
# D444C_REPO: the tree to read the base and the golden from (the self-test runs
# mutated copies of this script from a scratch dir).
repo="${D444C_REPO:-$(cd "$here/../.." && pwd)}"
base_dir="$repo/deploy/k8s/demo-d444c-task-vs-raw"
golden="$repo/nextgsim-semantic/src/task_vs_raw/golden_seed444.txt"
job=d444c-task-vs-raw
container=task-vs-raw
base_image=nextgsim/nr-semantic-demo

usage() { sed -n '4,17p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
info() { printf '[d444c-k8s] %s\n' "$*"; }
fail_check() {
    printf 'DEMO CHECK FAILED: %s\n' "$*" >&2
    if [ -n "${GITHUB_ACTIONS:-}" ]; then printf '::error::%s\n' "$*" >&2; fi
    failures=$((failures + 1))
}

namespace=""
image=""
context=""
timeout_s=360
run_id=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --namespace) [ "$#" -ge 2 ] || usage; namespace="$2"; shift 2 ;;
        --image) [ "$#" -ge 2 ] || usage; image="$2"; shift 2 ;;
        --context) [ "$#" -ge 2 ] || usage; context="$2"; shift 2 ;;
        --timeout) [ "$#" -ge 2 ] || usage; timeout_s="$2"; shift 2 ;;
        --run-id) [ "$#" -ge 2 ] || usage; run_id="$2"; shift 2 ;;
        -h | --help) usage ;;
        *) printf 'd444c-k8s: unknown argument %s\n' "$1" >&2; usage ;;
    esac
done

case "$timeout_s" in '' | *[!0-9]*) printf 'd444c-k8s: --timeout must be whole seconds\n' >&2; exit 2 ;; esac
if [ -z "$run_id" ]; then
    run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
fi
case "$run_id" in '' | */* | .*) printf 'd444c-k8s: bad --run-id %s\n' "$run_id" >&2; exit 2 ;; esac
if [ -z "$namespace" ]; then
    namespace="d444c-$(printf '%s' "$run_id" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9-' '-')"
    namespace="${namespace:0:63}"
    namespace="${namespace%-}"
fi
if ! [[ "$namespace" =~ ^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$ ]]; then
    printf 'd444c-k8s: namespace %s is not a DNS-1123 label\n' "$namespace" >&2
    exit 2
fi

# The image override, as kustomize `images:` fields. A tag or digest is required:
# a bare name would silently keep the base's `local` tag.
image_fields=()
if [ -n "$image" ]; then
    case "$image" in
        *@sha256:*) image_fields=("newName: ${image%@*}" "digest: ${image#*@}") ;;
        *)
            last="${image##*/}"
            case "$last" in
                *:?*) image_fields=("newName: ${image%:*}" "newTag: ${last##*:}") ;;
                *) printf 'd444c-k8s: --image %s needs a tag or a digest\n' "$image" >&2; exit 2 ;;
            esac
            ;;
    esac
fi

command -v kubectl >/dev/null 2>&1 || { printf 'd444c-k8s: kubectl not found\n' >&2; exit 2; }
[ -f "$golden" ] || { printf 'd444c-k8s: golden %s missing\n' "$golden" >&2; exit 2; }
[ -f "$base_dir/kustomization.yaml" ] || { printf 'd444c-k8s: base %s missing\n' "$base_dir" >&2; exit 2; }

kc_args=()
if [ -n "$context" ]; then kc_args=(--context "$context"); fi
# bash 4.2: "${kc_args[@]}" on an empty array is an unbound-variable error.
kc() { kubectl ${kc_args[@]+"${kc_args[@]}"} "$@"; }

out="${D444C_ARTEFACTS_ROOT:-/tmp/eks-val/demos/d444c-task-vs-raw}/$run_id"
mkdir -p "$out"
work="$(mktemp -d)"
created=0
failures=0

cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    if [ "$created" -eq 1 ]; then
        info "deleting namespace $namespace"
        if kc delete namespace "$namespace" --wait=true --timeout=180s > "$out/namespace-delete.txt" 2>&1; then
            info "namespace $namespace deleted"
            printf 'namespace_deleted=yes\n' >> "$out/run.env"
        else
            printf 'namespace_deleted=NO\n' >> "$out/run.env"
            printf 'DEMO CHECK FAILED: could not delete namespace %s (see %s); delete it by hand\n' \
                "$namespace" "$out/namespace-delete.txt" >&2
            if [ "$rc" -eq 0 ]; then rc=1; fi
        fi
    fi
    find "$work" -mindepth 1 -delete 2>/dev/null || true
    rmdir "$work" 2>/dev/null || true
    exit "$rc"
}
trap cleanup EXIT
# INT and TERM exit through the EXIT trap with 128+signal.
trap 'exit 130' INT
trap 'exit 143' TERM

{
    printf 'run_id=%s\nnamespace=%s\ncontext=%s\nimage=%s\ntimeout=%s\n' \
        "$run_id" "$namespace" "${context:-<current>}" "${image:-$base_image:local (base)}" "$timeout_s"
    printf 'golden=%s\n' "${golden#"$repo"/}"
    printf 'started=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$out/run.env"
kc version > "$out/kubectl-version.txt" 2>&1 || true

# 1. The temporary namespace. `create` fails on an existing one, so an existing
# namespace is never adopted, and never deleted.
if ! kc create namespace "$namespace" > "$out/namespace-create.txt" 2>&1; then
    cat "$out/namespace-create.txt" >&2
    printf 'd444c-k8s: could not create namespace %s (does it exist already?); nothing applied, nothing deleted\n' "$namespace" >&2
    exit 2
fi
created=1
info "created namespace $namespace (run $run_id, artefacts $out)"
kc label namespace "$namespace" \
    pod-security.kubernetes.io/enforce=restricted \
    pod-security.kubernetes.io/enforce-version=latest \
    app.kubernetes.io/part-of=nextgsim-demo >> "$out/namespace-create.txt" 2>&1 \
    || { fail_check "could not label namespace $namespace (see $out/namespace-create.txt)"; exit 1; }

# 2. Overlay: a copy of the base plus namespace and image (kustomize refuses an
# absolute base path, so both live side by side under $work).
mkdir -p "$work/base" "$work/overlay"
cp "$base_dir/kustomization.yaml" "$base_dir/job.yaml" "$work/base/"
{
    printf 'apiVersion: kustomize.config.k8s.io/v1beta1\nkind: Kustomization\n'
    printf 'resources:\n  - ../base\nnamespace: %s\n' "$namespace"
    if [ "${#image_fields[@]}" -gt 0 ]; then
        printf 'images:\n  - name: %s\n' "$base_image"
        printf '    %s\n' "${image_fields[@]}"
    fi
} > "$work/overlay/kustomization.yaml"
cp "$work/overlay/kustomization.yaml" "$out/overlay-kustomization.yaml"
kc kustomize "$work/overlay" > "$out/rendered.yaml" 2> "$out/render.err" \
    || { cat "$out/render.err" >&2; fail_check "kubectl kustomize of the overlay failed"; exit 1; }
kc apply -k "$work/overlay" > "$out/apply.txt" 2>&1 || { cat "$out/apply.txt" >&2; fail_check "kubectl apply -k failed"; exit 1; }
info "applied job/$job"

# 3. Wait for Complete or Failed (a Job also gets SuccessCriteriaMet /
# FailureTarget on newer clusters; only the terminal two count).
state=""
waited=0
while :; do
    conds="$(kc -n "$namespace" get job "$job" \
        -o jsonpath='{range .status.conditions[?(@.status=="True")]}{.type}{"\n"}{end}' 2>/dev/null || true)"
    case "$conds" in
        *Failed*) state=Failed; break ;;
        *Complete*) state=Complete; break ;;
    esac
    if [ "$waited" -ge "$timeout_s" ]; then
        state=Timeout
        break
    fi
    sleep 2
    waited=$((waited + 2))
done
info "job/$job: $state after ${waited}s"
printf 'state=%s\nwaited=%s\nfinished=%s\n' "$state" "$waited" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$out/run.env"

# Capture whatever there is, before any check.
kc -n "$namespace" logs "job/$job" -c "$container" > "$out/job.log" 2> "$out/job-log.err" || true
kc -n "$namespace" get job "$job" -o yaml > "$out/job.yaml" 2>&1 || true
kc -n "$namespace" get pods -o wide > "$out/pods.txt" 2>&1 || true
kc -n "$namespace" get pods -o jsonpath='{range .items[*]}{.metadata.name} {.spec.nodeName} {.status.containerStatuses[*].imageID}{"\n"}{end}' \
    > "$out/pod-image.txt" 2>&1 || true
kc -n "$namespace" describe job "$job" > "$out/job-describe.txt" 2>&1 || true
kc -n "$namespace" get events --sort-by=.lastTimestamp > "$out/events.txt" 2>&1 || true

# The table is stdout; `kubectl logs` interleaves stderr. Drop exactly the
# example's stderr shapes (tracing lines and its own `task_vs_raw:` messages);
# anything else (a panic, a stray line) stays in and fails the diff.
grep -v -E '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.]+Z +(TRACE|DEBUG|INFO|WARN|ERROR) |^task_vs_raw: ' \
    "$out/job.log" > "$out/table.txt" || true
grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.]+Z +(TRACE|DEBUG|INFO|WARN|ERROR) |^task_vs_raw: ' \
    "$out/job.log" > "$out/stderr.txt" || true

# 4. Checks: all run, each failure is named.
check_wait() {
    if [ "$state" = Timeout ]; then
        fail_check "timed out after ${timeout_s}s waiting for job/$job (see $out/pods.txt, $out/events.txt)"
    fi
}
check_job() {
    if [ "$state" != Complete ]; then
        fail_check "job/$job did not complete (state $state; see $out/job-describe.txt)"
    fi
}
check_banner() {
    if ! grep -q -E 'WARN .*DEMO MODE ACTIVE: d444c-task-vs-raw=all' "$out/job.log"; then
        fail_check "no 'WARN DEMO MODE ACTIVE: d444c-task-vs-raw=all' in $out/job.log"
    fi
    if ! grep -q -F 'nextg_demo_enabled{demo="d444c-task-vs-raw",mode="all"} 1' "$out/job.log"; then
        fail_check "no nextg_demo_enabled{demo=\"d444c-task-vs-raw\",mode=\"all\"} 1 in $out/job.log"
    fi
}
check_table() {
    if diff -u "$golden" "$out/table.txt" > "$out/table.diff"; then
        info "table: $(wc -l < "$out/table.txt") lines, byte-identical to ${golden#"$repo"/}"
    else
        fail_check "the printed table differs from ${golden#"$repo"/} ($(grep -c '^[-+][^-+]' "$out/table.diff" || true) changed lines; see $out/table.diff)"
    fi
}
check_wait
check_job
check_banner
check_table
sha256sum "$out/table.txt" > "$out/table.sha256" 2>/dev/null || true

if [ "$failures" -ne 0 ]; then
    printf 'verdict=FAIL (%s)\n' "$failures" >> "$out/run.env"
    info "FAIL: $failures check(s) failed; artefacts in $out"
    exit 1
fi
printf 'verdict=PASS\n' >> "$out/run.env"
info "PASS: job complete, banner present, table equals the golden; artefacts in $out"
