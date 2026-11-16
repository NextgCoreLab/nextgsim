#!/usr/bin/env bash
# scripts/demo/d444c-k8s-selftest.sh: self-test of scripts/demo/d444c-k8s.sh
# (demo D444C, nextgsim#479) with a FAKE kubectl: no cluster, no docker.
#
#   scripts/demo/d444c-k8s-selftest.sh             # the cases below
#   scripts/demo/d444c-k8s-selftest.sh --mutants   # also: every check forced
#                                                  # always-true must be caught
#
# The fake keeps one namespace store PER CONTEXT and logs every call with the
# context it would hit (--context, else the current context at that moment), so
# a context switch mid-run is observable.
#
# Cases: a pass (table byte-equal to the golden, overlay namespace, image, pull
# policy and arch, --context on every call, the namespace deleted); a changed
# table row; a Job that Failed after printing a correct table (the pod state is
# named); a Job that never finishes (timeout); no DEMO MODE banner; a stray
# stderr line (a panic) in the log; TERM and INT while waiting (the namespace
# still deleted); a namespace that already exists (refused, never deleted); a
# create that fails for another reason (its real error is shown); a create the
# server did but the client lost (cleaned up by its run label); the current
# context switched mid-run (every call stays on the pinned context; the other
# cluster's same-named namespace survives); the namespace replaced mid-run (not
# deleted, named); a delete that fails, or returns 0 but leaves the namespace
# (exit 1); a reused run id; --image with a digest; a bare --image refused. The
# fake's pass log is built from the golden itself, so the script and the unit
# test read one file.
# `cond && ok ... || not_ok ...` throughout: ok only prints and cannot fail.
# shellcheck disable=SC2015
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
script="${D444C_SCRIPT:-$here/d444c-k8s.sh}"
golden="$repo/nextgsim-semantic/src/task_vs_raw/golden_seed444.txt"
# Per-run watchdog (seconds); the mutant runs lower it, since a mutant that
# disables the timeout hangs until it fires.
watchdog="${D444C_SELFTEST_WATCHDOG:-60}"
work="$(mktemp -d)"
trap 'find "$work" -mindepth 1 -delete 2>/dev/null; rmdir "$work" 2>/dev/null || true' EXIT

pass=0
ok() { pass=$((pass + 1)); printf 'ok %d - %s\n' "$pass" "$*"; }
not_ok() { printf 'not ok - %s\n' "$*" >&2; exit 1; }

real_kubectl="$(command -v kubectl || true)"

# --- the fake kubectl ---------------------------------------------------------
mkdir -p "$work/bin"
cat > "$work/bin/kubectl" <<'FAKE'
#!/usr/bin/env bash
# Fake kubectl. FAKE_MODE picks the Job's (and the namespace's) fate. State in
# $FAKE_STATE: current-context; ns/<ctx>/<name> = "<uid> <run-id label>"; every
# call is appended to calls as "ctx=<ctx it hits> <args>".
set -euo pipefail
st="$FAKE_STATE"
ctx=""
args=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --context) ctx="$2"; shift 2 ;;
        -n | --namespace) shift 2 ;;
        *) args+=("$1"); shift ;;
    esac
done
set -- "${args[@]}"
[ -n "$ctx" ] || ctx="$(cat "$st/current-context")"
printf 'ctx=%s %s\n' "$ctx" "$*" >> "$st/calls"
store="$st/ns/$ctx"
mkdir -p "$store"
stderr_lines() {
    printf '%s\n' \
        '2026-11-15T00:00:00.000001Z  WARN nextgsim_demo: DEMO MODE ACTIVE: d444c-task-vs-raw=all' \
        '2026-11-15T00:00:00.000002Z  INFO task_vs_raw: [demo:d444c-task-vs-raw] nextg_demo_enabled{demo="d444c-task-vs-raw",mode="all"} 1 (one-shot: this process has no /metrics)' \
        '2026-11-15T00:00:00.000003Z  INFO task_vs_raw: [demo:d444c-task-vs-raw] sweep: mode all, noise seed 444, emulated AWGN (non-normative)'
}
case "$1 ${2:-}" in
    "config current-context") cat "$st/current-context" ;;
    "version "*) echo "Client Version: fake" ;;
    "create -f")
        manifest="$(cat)"
        printf '%s\n' "$manifest" > "$st/ns-manifest.yaml"
        name="$(sed -n 's/^  name: //p' <<<"$manifest")"
        run="$(sed -n 's/^    nextg.demo\/run-id: "\(.*\)"$/\1/p' <<<"$manifest")"
        if [ "$FAKE_MODE" = create-error ]; then
            echo "error: context \"$ctx\" does not exist" >&2; exit 1
        fi
        if [ "$FAKE_MODE" = ns-exists ] || [ -e "$store/$name" ]; then
            echo "Error from server (AlreadyExists): namespaces \"$name\" already exists" >&2; exit 1
        fi
        echo "uid-$ctx-$name $run" > "$store/$name"
        if [ "$FAKE_MODE" = create-lost ]; then
            echo "Unable to connect to the server: net/http: request canceled (Client.Timeout exceeded)" >&2; exit 1
        fi
        printf 'uid-%s-%s' "$ctx" "$name" ;;
    "get namespace")
        if [ ! -e "$store/$3" ]; then
            echo "Error from server (NotFound): namespaces \"$3\" not found" >&2; exit 1
        fi
        if [ "${4:-}" = -o ]; then cat "$store/$3"; else echo "$3   Active"; fi ;;
    "delete namespace")
        sel="" name=""
        shift 2
        while [ "$#" -gt 0 ]; do
            case "$1" in
                -l) sel="${2#nextg.demo/run-id=}"; shift 2 ;;
                --field-selector) name="${2#metadata.name=}"; shift 2 ;;
                *) shift ;;
            esac
        done
        [ -n "$sel" ] && [ -n "$name" ] || { echo "fake kubectl: delete without -l and --field-selector" >&2; exit 3; }
        if [ "$FAKE_MODE" = delete-fails ]; then echo "error: timed out waiting for the condition" >&2; exit 1; fi
        if [ "$FAKE_MODE" = delete-noop ]; then echo "No resources found"; exit 0; fi
        if [ -e "$store/$name" ] && [ "$(cut -d' ' -f2 < "$store/$name")" = "$sel" ]; then
            rm "$store/$name"
            touch "$st/deleted-$ctx-$name"
            echo "namespace \"$name\" deleted"
        else
            echo "No resources found"
        fi ;;
    "kustomize "*)
        if [ -n "${REAL_KUBECTL:-}" ]; then "$REAL_KUBECTL" kustomize "$2"; else cat "$2/kustomization.yaml"; fi ;;
    "apply -k")
        cp "$3/kustomization.yaml" "$st/applied-overlay.yaml"
        echo "job.batch/d444c-task-vs-raw created" ;;
    "get job")
        if [ "${4:-}" = "-o" ] && [ "${5:-}" = yaml ]; then echo "kind: Job"; exit 0; fi
        n=$(( $(cat "$st/polls" 2>/dev/null || echo 0) + 1 ))
        echo "$n" > "$st/polls"
        [ "$n" -ge 2 ] || exit 0   # the first poll sees a running Job
        if [ "$n" -eq 2 ]; then
            case "$FAKE_MODE" in
                # Another shell / kind / aws eks update-kubeconfig switches the
                # current context while the Job runs.
                ctx-switch) echo ctx-b > "$st/current-context" ;;
                # Someone deletes the namespace and recreates it, unlabelled.
                ns-replaced) for f in "$st"/ns/*/d444c-st-replaced; do echo "uid-other " > "$f"; done ;;
            esac
        fi
        case "$FAKE_MODE" in
            fail) printf 'FailureTarget\nFailed\n' ;;
            timeout | hang) : ;;
            *) printf 'SuccessCriteriaMet\nComplete\n' ;;
        esac ;;
    "logs job/d444c-task-vs-raw")
        case "$FAKE_MODE" in
            timeout | hang) echo "error: container not started" >&2; exit 1 ;;
            nobanner) stderr_lines | grep -v 'DEMO MODE ACTIVE'; cat "$FAKE_GOLDEN" ;;
            diff) stderr_lines; sed '10s/0\.[0-9]\{4\}/0.9999/' "$FAKE_GOLDEN" ;;
            panic) stderr_lines; cat "$FAKE_GOLDEN"; echo "thread 'main' panicked at src/main.rs:1:1:" ;;
            *) stderr_lines; cat "$FAKE_GOLDEN"
               echo '2026-11-15T00:00:01.000000Z  INFO task_vs_raw: [demo:d444c-task-vs-raw] printed 60 rows' ;;
        esac ;;
    "get pods")
        if [ "$FAKE_MODE" = fail ]; then echo "d444c-task-vs-raw-x phase=Failed terminated=Error exit=1"; else echo "fake pods"; fi ;;
    "describe job" | "get events") echo "fake $1 $2" ;;
    *) echo "fake kubectl: unexpected $*" >&2; exit 3 ;;
esac
FAKE
chmod +x "$work/bin/kubectl"

# setup_case NAME: a fresh fake state (current context ctx-a); sets st and out.
setup_case() {
    st="$work/$1"
    mkdir -p "$st"
    echo ctx-a > "$st/current-context"
    out="$st/artefacts/$1"
}
# run_case NAME MODE [script args...]: runs the script under the watchdog in the
# state setup_case made (made here if not yet); sets rc and err.
run_case() {
    local name="$1" mode="$2"; shift 2
    [ -d "$work/$name" ] || setup_case "$name"
    st="$work/$name"
    out="$st/artefacts/$name"
    rc=0
    PATH="$work/bin:$PATH" FAKE_STATE="$st" FAKE_MODE="$mode" FAKE_GOLDEN="$golden" \
        REAL_KUBECTL="$real_kubectl" D444C_ARTEFACTS_ROOT="$st/artefacts" D444C_REPO="$repo" D444C_POLL_S=0.1 \
        timeout "$watchdog" "$script" --run-id "$name" "$@" > "$st/stdout" 2> "$st/stderr" || rc=$?
    err="$(cat "$st/stderr")"
}
deleted() { [ -e "$st/deleted-${2:-ctx-a}-$1" ]; }
exists_on() { [ -e "$st/ns/$2/$1" ]; }

# 0. (first, so the pinning mutant is caught here) The current context switches mid-run (no --context): the run stays on the
# context it pinned, and the other cluster's same-named namespace survives.
setup_case switch
mkdir -p "$st/ns/ctx-b" && echo "uid-shared " > "$st/ns/ctx-b/d444c-st-shared"
run_case switch ctx-switch --namespace d444c-st-shared
[ "$rc" -eq 0 ] && ok "context switch: exit 0" || not_ok "context switch: rc=$rc $err"
[ "$(cat "$st/current-context")" = ctx-b ] && ok "context switch: the fake did switch mid-run" || not_ok "context switch: no switch happened"
if grep -q '^ctx=ctx-b ' "$st/calls"; then not_ok "context switch: calls reached ctx-b: $(grep '^ctx=ctx-b ' "$st/calls")"; fi
ok "context switch: no call reached the new current context"
deleted d444c-st-shared ctx-a && exists_on d444c-st-shared ctx-b && [ "$(cat "$st/ns/ctx-b/d444c-st-shared")" = "uid-shared " ] \
    && ok "context switch: its own namespace deleted on ctx-a; ctx-b's namespace untouched" || not_ok "context switch: wrong namespace deleted"
grep -q -x 'context=ctx-a (current-context)' "$out/run.env" && ok "context switch: run.env records the pinned context" || not_ok "switch: run.env"

# 1. Pass, with --context and --image.
run_case pass pass --namespace d444c-st-pass --context kind-x --image reg.example:5000/nextg/semantic-demo:demo-d444c-abc1234
[ "$rc" -eq 0 ] && ok "pass: exit 0" || not_ok "pass: rc=$rc: $err"
cmp -s "$golden" "$out/table.txt" && ok "pass: the extracted table is byte-equal to the golden" || not_ok "pass: table != golden"
deleted d444c-st-pass kind-x && ok "pass: the namespace it created is deleted" || not_ok "pass: namespace not deleted"
if grep -v -q '^ctx=kind-x ' "$st/calls"; then not_ok "pass: a kubectl call off --context: $(grep -v '^ctx=kind-x ' "$st/calls")"; fi
ok "pass: every kubectl call hits --context"
grep -q -F '    nextg.demo/run-id: "pass"' "$st/ns-manifest.yaml" \
    && grep -q -F '    pod-security.kubernetes.io/enforce: restricted' "$st/ns-manifest.yaml" \
    && ok "pass: the namespace is created with its PSA and run-id labels in one call" || not_ok "pass: ns manifest: $(cat "$st/ns-manifest.yaml")"
grep -q -x 'namespace: d444c-st-pass' "$st/applied-overlay.yaml" && ok "pass: the overlay sets the namespace" || not_ok "pass: overlay namespace"
grep -q -x '    newName: reg.example:5000/nextg/semantic-demo' "$st/applied-overlay.yaml" \
    && grep -q -x '    newTag: demo-d444c-abc1234' "$st/applied-overlay.yaml" \
    && ok "pass: --image with a registry port splits into newName/newTag" || not_ok "pass: overlay image: $(cat "$st/applied-overlay.yaml")"
if [ -n "$real_kubectl" ]; then
    grep -q -F 'image: reg.example:5000/nextg/semantic-demo:demo-d444c-abc1234' "$out/rendered.yaml" \
        && grep -q -F 'namespace: d444c-st-pass' "$out/rendered.yaml" \
        && grep -q -F 'imagePullPolicy: IfNotPresent' "$out/rendered.yaml" \
        && grep -q -F 'kubernetes.io/arch: amd64' "$out/rendered.yaml" \
        && ok "pass: the rendered Job runs the given image (IfNotPresent) on amd64 in the namespace" || not_ok "pass: render: $(cat "$out/rendered.yaml")"
else
    ok "pass: render check skipped (no real kubectl)"
fi
grep -q -x 'verdict=PASS' "$out/run.env" && grep -q -x 'namespace_deleted=yes' "$out/run.env" \
    && grep -q -x 'namespace_uid=uid-kind-x-d444c-st-pass' "$out/run.env" \
    && ok "pass: run.env records the verdict, the UID and the deletion" || not_ok "pass: run.env: $(cat "$out/run.env")"

# 1b. The base image (no --image) is never pulled; --arch reaches the Job.
if [ -n "$real_kubectl" ]; then
    run_case base pass --namespace d444c-st-base --arch arm64
    [ "$rc" -eq 0 ] && grep -q -F 'image: nextgsim/nr-semantic-demo:local' "$out/rendered.yaml" \
        && grep -q -F 'imagePullPolicy: Never' "$out/rendered.yaml" \
        && grep -q -F 'kubernetes.io/arch: arm64' "$out/rendered.yaml" \
        && ok "no --image: the base's local image with pull Never; --arch arm64 sets the nodeSelector" \
        || not_ok "base render: rc=$rc $(cat "$out/rendered.yaml" 2>/dev/null)"
fi

# 2. One changed table row.
run_case diff diff --namespace d444c-st-diff
[ "$rc" -eq 1 ] && ok "table diff: exit 1" || not_ok "table diff: rc=$rc"
case "$err" in *"DEMO CHECK FAILED: the printed table differs"*) ok "table diff: named" ;; *) not_ok "table diff: $err" ;; esac
grep -q '^+task-encoder.*0.9999' "$out/table.diff" && ok "table diff: the diff shows the row" || not_ok "table diff: $(cat "$out/table.diff")"
deleted d444c-st-diff && ok "table diff: namespace deleted" || not_ok "table diff: namespace kept"

# 3. A Job that Failed (after printing a correct table: only the Job check can see it).
run_case fail fail --namespace d444c-st-fail
[ "$rc" -eq 1 ] && ok "job failure: exit 1" || not_ok "job failure: rc=$rc"
case "$err" in *"DEMO CHECK FAILED: job/d444c-task-vs-raw did not complete (state Failed"*) ok "job failure: named" ;; *) not_ok "job failure: $err" ;; esac
case "$err" in *"terminated=Error exit=1"*) ok "job failure: the pod's state is in the message" ;; *) not_ok "job failure: no pod state: $err" ;; esac
case "$err" in *"table differs"*) not_ok "job failure: the table should match here" ;; esac
deleted d444c-st-fail && ok "job failure: namespace deleted" || not_ok "job failure: namespace kept"

# 4. A Job that never finishes.
run_case timeout timeout --namespace d444c-st-timeout --timeout 1
[ "$rc" -eq 1 ] && ok "timeout: exit 1" || not_ok "timeout: rc=$rc ($err)"
case "$err" in *"DEMO CHECK FAILED: timed out after 1s"*) ok "timeout: named" ;; *) not_ok "timeout: $err" ;; esac
deleted d444c-st-timeout && ok "timeout: namespace deleted" || not_ok "timeout: namespace kept"

# 5. No DEMO MODE banner (the table itself is right).
run_case nobanner nobanner --namespace d444c-st-nobanner
[ "$rc" -eq 1 ] && ok "no banner: exit 1" || not_ok "no banner: rc=$rc"
case "$err" in *"DEMO CHECK FAILED: no 'WARN DEMO MODE ACTIVE"*) ok "no banner: named" ;; *) not_ok "no banner: $err" ;; esac
case "$err" in *"table differs"*) not_ok "no banner: the table should match here" ;; esac

# 6. A stray stderr line is not filtered away: it fails the diff.
run_case panic panic --namespace d444c-st-panic
[ "$rc" -eq 1 ] && case "$err" in *"table differs"*) true ;; *) false ;; esac \
    && ok "stray stderr line: kept in the table and fails the diff" || not_ok "panic: rc=$rc $err"

# 7. TERM and INT while waiting: the namespace is still deleted.
for sig in TERM INT; do
    name="sig-$sig"
    setup_case "$name"
    ns="d444c-st-$(printf '%s' "$sig" | tr '[:upper:]' '[:lower:]')"
    # Background jobs start with INT ignored, and bash cannot trap a signal
    # ignored on entry: perl restores the default disposition before exec.
    PATH="$work/bin:$PATH" FAKE_STATE="$st" FAKE_MODE=hang FAKE_GOLDEN="$golden" \
        REAL_KUBECTL="$real_kubectl" D444C_ARTEFACTS_ROOT="$st/artefacts" D444C_REPO="$repo" D444C_POLL_S=0.1 \
        perl -e '$SIG{INT} = "DEFAULT"; exec @ARGV or die' \
        "$script" --run-id "$name" --namespace "$ns" --timeout 600 \
        > "$st/stdout" 2> "$st/stderr" &
    pid=$!
    for _ in $(seq 1 100); do
        if [ -s "$st/polls" ]; then break; fi
        sleep 0.1
    done
    [ -s "$st/polls" ] || { kill -KILL "$pid" 2>/dev/null || true; not_ok "$sig: the script never reached the wait"; }
    kill -s "$sig" "$pid"
    rc=0
    wait "$pid" || rc=$?
    want=143; [ "$sig" = INT ] && want=130
    [ "$rc" -eq "$want" ] && ok "$sig mid-wait: exit $want" || not_ok "$sig: rc=$rc (want $want): $(cat "$st/stderr")"
    deleted "$ns" && ok "$sig mid-wait: namespace deleted" || not_ok "$sig: namespace kept"
done

# 8. The namespace exists already: refused, nothing applied, never deleted.
setup_case exists
mkdir -p "$st/ns/ctx-a" && echo "uid-theirs " > "$st/ns/ctx-a/d444c-st-exists"
run_case exists pass --namespace d444c-st-exists
[ "$rc" -eq 2 ] && ok "existing namespace: exit 2" || not_ok "existing namespace: rc=$rc"
case "$err" in *"already exists on ctx-a; nothing applied, nothing deleted"*) ok "existing namespace: named" ;; *) not_ok "existing namespace: $err" ;; esac
if deleted d444c-st-exists || grep -q ' delete ' "$st/calls"; then not_ok "existing namespace: a delete was sent"; fi
if grep -q ' apply ' "$st/calls"; then not_ok "existing namespace: something was applied"; fi
exists_on d444c-st-exists ctx-a && ok "existing namespace: not applied, not deleted" || not_ok "existing namespace: gone"

# 9. A create that fails for another reason shows that reason.
run_case createerr create-error --namespace d444c-st-createerr
[ "$rc" -eq 2 ] && case "$err" in *"could not create namespace d444c-st-createerr on ctx-a: error: context"*) true ;; *) false ;; esac \
    && ok "create error: exit 2 with the real error" || not_ok "create error: rc=$rc $err"
case "$err" in *"already exists"*) not_ok "create error: claims the namespace exists" ;; esac

# 10. The server created the namespace but the client lost the reply: the run
# label still finds it, and it is deleted.
run_case createlost create-lost --namespace d444c-st-createlost
[ "$rc" -eq 2 ] && deleted d444c-st-createlost && ! exists_on d444c-st-createlost ctx-a \
    && ok "create lost: the namespace the server made is deleted by its run label" || not_ok "create lost: rc=$rc leaked? $err"

# 12. The namespace is replaced mid-run (other UID, no run label): not deleted.
run_case replaced ns-replaced --namespace d444c-st-replaced
[ "$rc" -eq 1 ] && ok "namespace replaced: exit 1" || not_ok "namespace replaced: rc=$rc"
case "$err" in *"DEMO CHECK FAILED: namespace d444c-st-replaced on ctx-a is not the one this run created"*) ok "namespace replaced: named, not deleted" ;; *) not_ok "replaced: $err" ;; esac
exists_on d444c-st-replaced ctx-a && ! grep -q ' delete ' "$st/calls" \
    && ok "namespace replaced: no delete sent" || not_ok "replaced: a delete was sent"

# 13. The namespace cannot be deleted: a pass turns into a failure, named.
run_case delfail delete-fails --namespace d444c-st-delfail
[ "$rc" -eq 1 ] && ok "delete fails: exit 1 although every check passed" || not_ok "delete fails: rc=$rc"
case "$err" in *"DEMO CHECK FAILED: could not delete namespace d444c-st-delfail"*) ok "delete fails: named" ;; *) not_ok "delete fails: $err" ;; esac
grep -q -x 'namespace_deleted=NO' "$out/run.env" && ok "delete fails: run.env records it" || not_ok "delete fails: run.env"
# ... or returns 0 and leaves it.
run_case delnoop delete-noop --namespace d444c-st-delnoop
[ "$rc" -eq 1 ] && case "$err" in *"could not delete namespace d444c-st-delnoop"*) true ;; *) false ;; esac \
    && ok "delete no-op: a namespace still there after delete fails the run" || not_ok "delete no-op: rc=$rc $err"

# 14. A reused run id is refused before any kubectl call.
setup_case reuse
mkdir -p "$st/artefacts/reuse"
run_case reuse pass --namespace d444c-st-reuse
[ "$rc" -eq 2 ] && [ ! -e "$st/calls" ] && ok "reused run id: refused before any kubectl call" || not_ok "reuse: rc=$rc"

# 15. --image forms.
run_case digest pass --namespace d444c-st-digest --image reg.example/nextg/semantic-demo@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
[ "$rc" -eq 0 ] && grep -q -x '    newName: reg.example/nextg/semantic-demo' "$st/applied-overlay.yaml" \
    && grep -q -x '    digest: sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef' "$st/applied-overlay.yaml" \
    && ok "--image name@sha256:...: newName + digest" || not_ok "digest: rc=$rc $(cat "$st/applied-overlay.yaml" 2>/dev/null)"
run_case bare pass --namespace d444c-st-bare --image reg.example:5000/nextg/semantic-demo
[ "$rc" -eq 2 ] && [ ! -e "$st/calls" ] && ok "--image without a tag: refused before any kubectl call" || not_ok "bare image: rc=$rc"

echo "d444c-k8s selftest: $pass checks passed"

# --- mutants: force each check always-true; the self-test must fail ----------
if [ "${1:-}" = --mutants ]; then
    # shellcheck disable=SC2016 # the patterns are literal script lines
    mutants=(
        'check_wait() { => check_wait() { return 0'
        'check_job() { => check_job() { return 0'
        'check_banner() { => check_banner() { return 0'
        'check_table() { => check_table() { return 0'
        '    if [ "$attempted" -eq 1 ]; then =>     if false; then'
        'attempted=1 => attempted=0'
        '    if [ "${live_run:-}" != "$run_id" ] || { [ -n "$ns_uid" ] && [ "$live_uid" != "$ns_uid" ]; }; then =>     if false; then'
        'kc() { kubectl --context "$context" "$@"; } => kc() { kubectl "$@"; }'
        '    if kc get namespace "$namespace" > "$out/namespace-after.txt" 2>&1; then return 1; fi =>     return 0'
        '    if [ "$waited" -ge "$timeout_s" ]; then =>     if false; then'
        '        *Failed*) state=Failed; break ;; =>         *Failed*) state=Complete; break ;;'
        '        delete_own_namespace || { if [ "$rc" -eq 0 ]; then rc=1; fi; } =>         delete_own_namespace || :'
    )
    i=0
    for m in "${mutants[@]}"; do
        i=$((i + 1))
        from="${m%% => *}"
        to="${m#* => }"
        mut="$work/mutant-$i.sh"
        FROM="$from" TO="$to" perl -pe 's/^\Q$ENV{FROM}\E$/$ENV{TO}/' "$script" > "$mut"
        chmod +x "$mut"
        if cmp -s "$script" "$mut"; then not_ok "mutant $i did not apply: $from"; fi
        if D444C_SCRIPT="$mut" D444C_SELFTEST_WATCHDOG=5 timeout 120 "$0" > "$work/mutant-$i.out" 2>&1; then
            not_ok "mutant $i SURVIVED (the self-test passed): $from -> $to"
        fi
        ok "mutant $i caught: $from -> $to ($(grep -m1 '^not ok' "$work/mutant-$i.out" || echo 'watchdog'))"
    done
fi
