#!/usr/bin/env bash
# scripts/demo/d444c-k8s.sh: run demo D444C (nextgsim#479) as a Kubernetes Job on
# ANY cluster and check its table against the committed golden.
#
#   scripts/demo/d444c-k8s.sh [--namespace NS] [--image REF] [--context CTX]
#                             [--arch ARCH] [--timeout SECS] [--run-id ID]
#
#   --namespace NS  the TEMPORARY namespace to create (default d444c-<run-id>). It
#                   must not exist: the script creates it, and deletes only the
#                   namespace it created (same context, same UID, its run label).
#   --image REF     the nr-semantic-demo image, with a tag or a digest
#                   (registry/repo:tag or registry/repo@sha256:...), pulled
#                   IfNotPresent. Default: the base's nextgsim/nr-semantic-demo:local
#                   with imagePullPolicy Never, so no node ever pulls that name
#                   from a registry (build it with docker/demo/Dockerfile.nr-semantic-demo;
#                   on kind, `kind load docker-image nextgsim/nr-semantic-demo:local`).
#   --context CTX   the kubeconfig context. Default: the current context, read
#                   ONCE at start. Either way it is pinned: every kubectl call
#                   passes --context, so a context switch mid-run cannot redirect
#                   the run or its cleanup. KUBECONFIG is honoured.
#   --arch ARCH     the node architecture the Job is pinned to (kubernetes.io/arch;
#                   default amd64, the arch the image is built for).
#   --timeout SECS  how long to wait for the Job to finish (default 360; the
#                   Job's own activeDeadlineSeconds is 300).
#   --run-id ID     the artefact directory name and the namespace's
#                   nextg.demo/run-id label (default <UTC time>-<pid>; must be new).
#
# Only kubectl is needed (kustomize via `kubectl apply -k`). What it does:
#   1. creates the namespace in one call, labelled Pod Security `restricted`
#      (enforce; the cluster's PodSecurity admission then refuses the pod if the
#      Job's hardening ever regresses) and nextg.demo/run-id=<run-id>, and records
#      its UID;
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
#   5. always deletes the namespace it created (EXIT trap; INT and TERM too), on
#      the pinned context, only while its UID and run label still match.
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

usage() { sed -n '4,25p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
info() { printf '[d444c-k8s] %s\n' "$*"; }
fail_check() {
    printf 'DEMO CHECK FAILED: %s\n' "$*" >&2
    if [ -n "${GITHUB_ACTIONS:-}" ]; then printf '::error::%s\n' "$*" >&2; fi
    failures=$((failures + 1))
}

namespace=""
image=""
context=""
arch=amd64
timeout_s=360
run_id=""
# D444C_POLL_S: the Job poll interval (the self-test sets 0.1).
poll_s="${D444C_POLL_S:-2}"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --namespace) [ "$#" -ge 2 ] || usage; namespace="$2"; shift 2 ;;
        --image) [ "$#" -ge 2 ] || usage; image="$2"; shift 2 ;;
        --context) [ "$#" -ge 2 ] || usage; context="$2"; shift 2 ;;
        --arch) [ "$#" -ge 2 ] || usage; arch="$2"; shift 2 ;;
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
# The run id is also a label value: <= 63 characters, alphanumeric at both ends.
if ! [[ "$run_id" =~ ^[A-Za-z0-9]([A-Za-z0-9_.-]{0,61}[A-Za-z0-9])?$ ]]; then
    printf 'd444c-k8s: bad --run-id %s (a label value: [A-Za-z0-9_.-], <= 63, alphanumeric ends)\n' "$run_id" >&2
    exit 2
fi
[[ "$arch" =~ ^[a-z0-9]+$ ]] || { printf 'd444c-k8s: bad --arch %s\n' "$arch" >&2; exit 2; }
[[ "$poll_s" =~ ^[0-9]+(\.[0-9]+)?$ ]] || { printf 'd444c-k8s: bad D444C_POLL_S %s\n' "$poll_s" >&2; exit 2; }
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

out="${D444C_ARTEFACTS_ROOT:-/tmp/eks-val/demos/d444c-task-vs-raw}/$run_id"
if [ -e "$out" ]; then
    printf 'd444c-k8s: run id %s was used already (%s exists); pick a new --run-id\n' "$run_id" "$out" >&2
    exit 2
fi

# Pin the context ONCE. Without --context, kubectl would re-read the current
# context on every call, so a `kubectl config use-context`, `kind create cluster`
# or `aws eks update-kubeconfig` elsewhere mid-run would send the remaining calls,
# and the cleanup's delete, to another cluster.
context_source=--context
if [ -z "$context" ]; then
    context="$(kubectl config current-context 2>/dev/null || true)"
    context_source=current-context
    [ -n "$context" ] || { printf 'd444c-k8s: no --context given and no current context\n' >&2; exit 2; }
fi
kc() { kubectl --context "$context" "$@"; }

mkdir -p "$out"
work="$(mktemp -d)"
attempted=0
ns_uid=""
failures=0

cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    if [ "$attempted" -eq 1 ]; then
        delete_own_namespace || { if [ "$rc" -eq 0 ]; then rc=1; fi; }
    fi
    find "$work" -mindepth 1 -delete 2>/dev/null || true
    rmdir "$work" 2>/dev/null || true
    exit "$rc"
}

# Delete the namespace only if it is still the one this run created: on the
# pinned context, with the recorded UID (when the create returned one) and this
# run's nextg.demo/run-id label. The delete itself selects by that label and the
# name, so a namespace without the label is never matched.
# Gone means NotFound, not merely "the get failed".
namespace_gone() {
    if kc get namespace "$namespace" > "$out/namespace-after.txt" 2>&1; then return 1; fi
    grep -q 'NotFound' "$out/namespace-after.txt"
}
delete_own_namespace() {
    local live live_uid live_run
    live="$(kc get namespace "$namespace" \
        -o jsonpath='{.metadata.uid} {.metadata.labels.nextg\.demo/run-id}' 2> "$out/namespace-get.err" || true)"
    read -r live_uid live_run <<< "$live" || true
    if [ -z "${live_uid:-}" ]; then
        printf 'namespace_deleted=absent\n' >> "$out/run.env"
        info "namespace $namespace is not on $context (nothing to delete)"
        return 0
    fi
    if [ "${live_run:-}" != "$run_id" ] || { [ -n "$ns_uid" ] && [ "$live_uid" != "$ns_uid" ]; }; then
        printf 'namespace_deleted=NO (not ours: uid %s run-id %s)\n' "$live_uid" "${live_run:-<none>}" >> "$out/run.env"
        printf 'DEMO CHECK FAILED: namespace %s on %s is not the one this run created (uid %s, want %s; run-id %s, want %s): NOT deleting it\n' \
            "$namespace" "$context" "$live_uid" "${ns_uid:-<unknown>}" "${live_run:-<none>}" "$run_id" >&2
        return 1
    fi
    info "deleting namespace $namespace (uid $live_uid) on $context"
    if kc delete namespace -l "nextg.demo/run-id=$run_id" --field-selector "metadata.name=$namespace" \
            --wait=true --timeout=180s > "$out/namespace-delete.txt" 2>&1 \
        && namespace_gone; then
        info "namespace $namespace deleted"
        printf 'namespace_deleted=yes\n' >> "$out/run.env"
        return 0
    fi
    printf 'namespace_deleted=NO\n' >> "$out/run.env"
    printf 'DEMO CHECK FAILED: could not delete namespace %s on %s (see %s); delete it by hand\n' \
        "$namespace" "$context" "$out/namespace-delete.txt" >&2
    return 1
}
trap cleanup EXIT
# INT and TERM exit through the EXIT trap with 128+signal.
trap 'exit 130' INT
trap 'exit 143' TERM

{
    printf 'run_id=%s\nnamespace=%s\ncontext=%s (%s)\nimage=%s\narch=%s\ntimeout=%s\n' \
        "$run_id" "$namespace" "$context" "$context_source" "${image:-$base_image:local (base, pull Never)}" "$arch" "$timeout_s"
    printf 'golden=%s\n' "${golden#"$repo"/}"
    printf 'started=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$out/run.env"
kc version > "$out/kubectl-version.txt" 2>&1 || true

# 1. The temporary namespace, created with its labels in one call. `create`
# fails on an existing one, so an existing namespace is never adopted, and never
# deleted. `attempted` is set first: a create that reached the server but not us
# (a signal, a client timeout) is still cleaned up, by its run label.
attempted=1
if ! ns_uid="$(kc create -f - -o jsonpath='{.metadata.uid}' 2> "$out/namespace-create.err" <<EOF
apiVersion: v1
kind: Namespace
metadata:
  name: ${namespace}
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/enforce-version: latest
    app.kubernetes.io/part-of: nextgsim-demo
    nextg.demo/run-id: "${run_id}"
EOF
)"; then
    cat "$out/namespace-create.err" >&2
    if grep -q 'AlreadyExists' "$out/namespace-create.err"; then
        attempted=0
        printf 'd444c-k8s: namespace %s already exists on %s; nothing applied, nothing deleted\n' "$namespace" "$context" >&2
    else
        printf 'd444c-k8s: could not create namespace %s on %s: %s\n' "$namespace" "$context" "$(head -n 1 "$out/namespace-create.err")" >&2
    fi
    exit 2
fi
printf 'namespace_uid=%s\n' "$ns_uid" >> "$out/run.env"
info "created namespace $namespace (uid $ns_uid) on $context (run $run_id, artefacts $out)"

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
    printf 'patches:\n  - target: {kind: Job, name: %s}\n    patch: |-\n' "$job"
    printf '      - op: replace\n        path: /spec/template/spec/nodeSelector/kubernetes.io~1arch\n        value: %s\n' "$arch"
    if [ "${#image_fields[@]}" -gt 0 ]; then
        # A registry image is pulled; only the base's `local` name is Never.
        printf '      - op: replace\n        path: /spec/template/spec/containers/0/imagePullPolicy\n        value: IfNotPresent\n'
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
start=$SECONDS
while :; do
    conds="$(kc -n "$namespace" get job "$job" \
        -o jsonpath='{range .status.conditions[?(@.status=="True")]}{.type}{"\n"}{end}' 2>/dev/null || true)"
    case "$conds" in
        *Failed*) state=Failed; break ;;
        *Complete*) state=Complete; break ;;
    esac
    waited=$((SECONDS - start))
    if [ "$waited" -ge "$timeout_s" ]; then
        state=Timeout
        break
    fi
    sleep "$poll_s"
done
info "job/$job: $state after ${waited}s"
printf 'state=%s\nwaited=%s\nfinished=%s\n' "$state" "$waited" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$out/run.env"

# Capture whatever there is, before any check.
kc -n "$namespace" logs "job/$job" -c "$container" > "$out/job.log" 2> "$out/job-log.err" || true
kc -n "$namespace" get job "$job" -o yaml > "$out/job.yaml" 2>&1 || true
kc -n "$namespace" get pods -o wide > "$out/pods.txt" 2>&1 || true
kc -n "$namespace" get pods -o jsonpath='{range .items[*]}{.metadata.name} {.spec.nodeName} {.status.containerStatuses[*].imageID}{"\n"}{end}' \
    > "$out/pod-image.txt" 2>&1 || true
kc -n "$namespace" get pods -o jsonpath='{range .items[*]}{.metadata.name} phase={.status.phase}{range .status.containerStatuses[*]} waiting={.state.waiting.reason} terminated={.state.terminated.reason} exit={.state.terminated.exitCode}{end}{"\n"}{end}' \
    > "$out/pod-status.txt" 2>&1 || true
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
    local why
    if [ "$state" != Complete ]; then
        # Name the likely cause: the pod's state (exec format error on the wrong
        # arch, ErrImageNeverPull, ...) or the first create/schedule refusal
        # (PodSecurity, no node of the arch).
        why="$(tr '\n' ';' < "$out/pod-status.txt") $(grep -m1 -E 'FailedCreate|FailedScheduling|BackOff|ErrImage' "$out/events.txt" || true)"
        fail_check "job/$job did not complete (state $state; pods: ${why}; see $out/job-describe.txt)"
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
