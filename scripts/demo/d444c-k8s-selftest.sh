#!/usr/bin/env bash
# scripts/demo/d444c-k8s-selftest.sh: self-test of scripts/demo/d444c-k8s.sh
# (demo D444C, nextgsim#479) with a FAKE kubectl: no cluster, no docker.
#
#   scripts/demo/d444c-k8s-selftest.sh             # the cases below
#   scripts/demo/d444c-k8s-selftest.sh --mutants   # also: every check forced
#                                                  # always-true must be caught
#
# Cases: a pass (table byte-equal to the golden, overlay namespace and image,
# --context on every call, the namespace deleted); a changed table row; a Job
# that Failed after printing a correct table; a Job that never finishes
# (timeout); no DEMO MODE banner; a stray stderr line (a panic) in the log;
# TERM and INT while waiting (the namespace still deleted); a namespace that
# already exists (refused, never deleted); --image with a digest; a bare --image
# refused; a namespace that cannot be deleted (exit 1). The fake's pass log is built from the golden itself, so the script
# and the unit test read one file.
# `cond && ok ... || not_ok ...` throughout: ok only prints and cannot fail.
# shellcheck disable=SC2015
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
script="${D444C_SCRIPT:-$here/d444c-k8s.sh}"
golden="$repo/nextgsim-semantic/src/task_vs_raw/golden_seed444.txt"
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
# Fake kubectl: FAKE_MODE picks the Job's fate; every call is appended to
# $FAKE_STATE/calls (one line, args space-joined).
set -euo pipefail
st="$FAKE_STATE"
printf '%s\n' "$*" >> "$st/calls"
args=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --context | -n | --namespace) shift 2 ;;
        *) args+=("$1"); shift ;;
    esac
done
set -- "${args[@]}"
stderr_lines() {
    printf '%s\n' \
        '2026-11-15T00:00:00.000001Z  WARN nextgsim_demo: DEMO MODE ACTIVE: d444c-task-vs-raw=all' \
        '2026-11-15T00:00:00.000002Z  INFO task_vs_raw: [demo:d444c-task-vs-raw] nextg_demo_enabled{demo="d444c-task-vs-raw",mode="all"} 1 (one-shot: this process has no /metrics)' \
        '2026-11-15T00:00:00.000003Z  INFO task_vs_raw: [demo:d444c-task-vs-raw] sweep: mode all, noise seed 444, emulated AWGN (non-normative)'
}
case "$1 ${2:-}" in
    "version "*) echo "Client Version: fake" ;;
    "create namespace")
        if [ "$FAKE_MODE" = ns-exists ]; then
            echo "Error from server (AlreadyExists): namespaces \"$3\" already exists" >&2
            exit 1
        fi
        echo "namespace/$3 created" ;;
    "label namespace") echo "namespace/$3 labeled" ;;
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
    "get pods" | "describe job" | "get events") echo "fake $1 $2" ;;
    "delete namespace")
        if [ "$FAKE_MODE" = delete-fails ]; then echo "error: timed out waiting for the condition" >&2; exit 1; fi
        echo "namespace \"$3\" deleted"; touch "$st/deleted-$3" ;;
    *) echo "fake kubectl: unexpected $*" >&2; exit 3 ;;
esac
FAKE
chmod +x "$work/bin/kubectl"

# run_case NAME MODE [script args...]: runs the script under a 60 s watchdog;
# sets rc, st (the fake's state dir), out (the run's artefacts) and err.
run_case() {
    local name="$1" mode="$2"; shift 2
    st="$work/$name"
    mkdir -p "$st"
    out="$st/artefacts/$name"
    rc=0
    PATH="$work/bin:$PATH" FAKE_STATE="$st" FAKE_MODE="$mode" FAKE_GOLDEN="$golden" \
        REAL_KUBECTL="$real_kubectl" D444C_ARTEFACTS_ROOT="$st/artefacts" D444C_REPO="$repo" \
        timeout 60 "$script" --run-id "$name" "$@" > "$st/stdout" 2> "$st/stderr" || rc=$?
    err="$(cat "$st/stderr")"
}
deleted() { [ -e "$st/deleted-$1" ]; }

# 1. Pass, with --context and --image.
run_case pass pass --namespace d444c-st-pass --context kind-x --image reg.example:5000/nextg/semantic-demo:demo-d444c-abc1234
[ "$rc" -eq 0 ] && ok "pass: exit 0" || not_ok "pass: rc=$rc: $err"
cmp -s "$golden" "$out/table.txt" && ok "pass: the extracted table is byte-equal to the golden" || not_ok "pass: table != golden"
deleted d444c-st-pass && ok "pass: the namespace it created is deleted" || not_ok "pass: namespace not deleted"
if grep -v -q -e '--context kind-x' "$st/calls"; then not_ok "pass: a kubectl call without --context: $(grep -v -e '--context kind-x' "$st/calls")"; fi
ok "pass: every kubectl call carries --context"
grep -q -x 'namespace: d444c-st-pass' "$st/applied-overlay.yaml" && ok "pass: the overlay sets the namespace" || not_ok "pass: overlay namespace"
grep -q -x '    newName: reg.example:5000/nextg/semantic-demo' "$st/applied-overlay.yaml" \
    && grep -q -x '    newTag: demo-d444c-abc1234' "$st/applied-overlay.yaml" \
    && ok "pass: --image with a registry port splits into newName/newTag" || not_ok "pass: overlay image: $(cat "$st/applied-overlay.yaml")"
if [ -n "$real_kubectl" ]; then
    grep -q -F 'image: reg.example:5000/nextg/semantic-demo:demo-d444c-abc1234' "$out/rendered.yaml" \
        && grep -q -F 'namespace: d444c-st-pass' "$out/rendered.yaml" \
        && ok "pass: the rendered Job runs the given image in the namespace" || not_ok "pass: render: $(cat "$out/rendered.yaml")"
else
    ok "pass: render check skipped (no real kubectl)"
fi
grep -q -x 'verdict=PASS' "$out/run.env" && grep -q -x 'namespace_deleted=yes' "$out/run.env" \
    && ok "pass: run.env records the verdict and the deletion" || not_ok "pass: run.env: $(cat "$out/run.env")"

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
case "$err" in *"table differs"*) not_ok "job failure: the table should match here" ;; esac
deleted d444c-st-fail && ok "job failure: namespace deleted" || not_ok "job failure: namespace kept"

# 4. A Job that never finishes.
run_case timeout timeout --namespace d444c-st-timeout --timeout 4
[ "$rc" -eq 1 ] && ok "timeout: exit 1" || not_ok "timeout: rc=$rc ($err)"
case "$err" in *"DEMO CHECK FAILED: timed out after 4s"*) ok "timeout: named" ;; *) not_ok "timeout: $err" ;; esac
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
    st="$work/$name"
    mkdir -p "$st"
    # Background jobs start with INT ignored, and bash cannot trap a signal
    # ignored on entry: perl restores the default disposition before exec.
    PATH="$work/bin:$PATH" FAKE_STATE="$st" FAKE_MODE=hang FAKE_GOLDEN="$golden" \
        REAL_KUBECTL="$real_kubectl" D444C_ARTEFACTS_ROOT="$st/artefacts" D444C_REPO="$repo" \
        perl -e '$SIG{INT} = "DEFAULT"; exec @ARGV or die' \
        "$script" --run-id "$name" --namespace "d444c-st-$(printf '%s' "$sig" | tr '[:upper:]' '[:lower:]')" --timeout 600 \
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
    deleted "d444c-st-$(printf '%s' "$sig" | tr '[:upper:]' '[:lower:]')" \
        && ok "$sig mid-wait: namespace deleted" || not_ok "$sig: namespace kept"
done

# 8. The namespace exists already: refused, nothing applied, never deleted.
run_case exists ns-exists --namespace d444c-st-exists
[ "$rc" -eq 2 ] && ok "existing namespace: exit 2" || not_ok "existing namespace: rc=$rc"
if deleted d444c-st-exists; then not_ok "existing namespace: it was DELETED"; fi
if grep -q '^apply' "$st/calls" || grep -q -e ' apply ' "$st/calls"; then not_ok "existing namespace: something was applied"; fi
ok "existing namespace: not applied, not deleted"

# 9. --image forms.
run_case digest pass --namespace d444c-st-digest --image reg.example/nextg/semantic-demo@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
[ "$rc" -eq 0 ] && grep -q -x '    newName: reg.example/nextg/semantic-demo' "$st/applied-overlay.yaml" \
    && grep -q -x '    digest: sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef' "$st/applied-overlay.yaml" \
    && ok "--image name@sha256:...: newName + digest" || not_ok "digest: rc=$rc $(cat "$st/applied-overlay.yaml" 2>/dev/null)"
run_case bare pass --namespace d444c-st-bare --image reg.example:5000/nextg/semantic-demo
[ "$rc" -eq 2 ] && [ ! -e "$st/calls" ] && ok "--image without a tag: refused before any kubectl call" || not_ok "bare image: rc=$rc"

# 10. The namespace cannot be deleted: a pass turns into a failure, named.
run_case delfail delete-fails --namespace d444c-st-delfail
[ "$rc" -eq 1 ] && ok "delete fails: exit 1 although every check passed" || not_ok "delete fails: rc=$rc"
case "$err" in *"DEMO CHECK FAILED: could not delete namespace d444c-st-delfail"*) ok "delete fails: named" ;; *) not_ok "delete fails: $err" ;; esac
grep -q -x 'namespace_deleted=NO' "$out/run.env" && ok "delete fails: run.env records it" || not_ok "delete fails: run.env"

echo "d444c-k8s selftest: $pass checks passed"

# --- mutants: force each check always-true; the self-test must fail ----------
if [ "${1:-}" = --mutants ]; then
    # shellcheck disable=SC2016 # the patterns are literal script lines
    mutants=(
        'check_wait() {|check_wait() { return 0'
        'check_job() {|check_job() { return 0'
        'check_banner() {|check_banner() { return 0'
        'check_table() {|check_table() { return 0'
        '    if [ "$created" -eq 1 ]; then|    if false; then'
        '    if [ "$created" -eq 1 ]; then|    if true; then'
        '    if [ "$waited" -ge "$timeout_s" ]; then|    if false; then'
        '        *Failed*) state=Failed; break ;;|        *Failed*) state=Complete; break ;;'
        '            if [ "$rc" -eq 0 ]; then rc=1; fi|            :'
    )
    i=0
    for m in "${mutants[@]}"; do
        i=$((i + 1))
        from="${m%%|*}"
        to="${m#*|}"
        mut="$work/mutant-$i.sh"
        FROM="$from" TO="$to" perl -pe 's/^\Q$ENV{FROM}\E$/$ENV{TO}/' "$script" > "$mut"
        chmod +x "$mut"
        if cmp -s "$script" "$mut"; then not_ok "mutant $i did not apply: $from"; fi
        if D444C_SCRIPT="$mut" timeout 120 "$0" > "$work/mutant-$i.out" 2>&1; then
            not_ok "mutant $i SURVIVED (the self-test passed): $from -> $to"
        fi
        ok "mutant $i caught: $from -> $to ($(grep -m1 '^not ok' "$work/mutant-$i.out" || echo 'watchdog'))"
    done
fi
