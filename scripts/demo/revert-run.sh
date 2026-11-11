#!/usr/bin/env bash
# scripts/demo/revert-run.sh: run one revert arm of a demo check (#456, #458).
#
#   revert-run.sh [--expect TEXT] PATCH -- CHECK [ARGS...]
#
# Applies PATCH to this git tree, runs CHECK, and passes only if CHECK FAILS
# under the patch: a revert arm proves a check depends on the code or config the
# patch takes away. With --expect, CHECK's output must also contain TEXT, so an
# arm cannot pass on an unrelated failure (a stack that never came up).
#
# The tree is restored by an EXIT/INT/TERM/HUP trap, so a failed, interrupted or
# killed run never leaves it patched. After restoring, the harness checks that
# `git status --porcelain` is exactly what it was before the run and that no
# tracked file differs from HEAD; either mismatch fails the arm.
#
# Exit status:
#   0  CHECK failed under the patch (with TEXT, if given): the arm holds.
#   1  CHECK passed under the patch, or failed without TEXT: the arm is broken.
#   2  usage error, a dirty tree, or a PATCH that does not apply (a stale patch
#      must never count as a passing arm).
#   3  the tree could not be restored to its pre-run state.
set -euo pipefail

usage() { echo "usage: $0 [--expect TEXT] PATCH -- CHECK [ARGS...]" >&2; exit 2; }

expect=""
while [ $# -gt 0 ]; do
    case "$1" in
        --expect) [ $# -ge 2 ] || usage; expect="$2"; shift 2 ;;
        --) usage ;;
        -*) usage ;;
        *) break ;;
    esac
done
[ $# -ge 3 ] || usage
patch="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
shift
[ "$1" = "--" ] || usage
shift

root="$(git rev-parse --show-toplevel)" || { echo "revert-run: not in a git tree" >&2; exit 2; }
cd "$root"
[ -f "$patch" ] || { echo "revert-run: no patch $patch" >&2; exit 2; }

# Only a tree whose tracked files match HEAD: restoring then means "back to HEAD",
# which the harness can verify, and the arm cannot run against stray edits.
if ! git diff --quiet HEAD --; then
    echo "revert-run: tracked files differ from HEAD; commit or stash first" >&2
    git status --short >&2
    exit 2
fi
before="$(git status --porcelain)"

if ! git apply --check "$patch" 2>/tmp/revert-run.$$; then
    echo "revert-run: $patch does not apply to $(git rev-parse --short HEAD); refresh it" >&2
    cat /tmp/revert-run.$$ >&2
    rm -f /tmp/revert-run.$$
    exit 2
fi
rm -f /tmp/revert-run.$$

applied=0
pid=""
out="$(mktemp)"
# shellcheck disable=SC2317,SC2329  # reached through the traps below
restore() {
    local rc=$?
    trap - EXIT INT TERM HUP
    # An interrupted run: stop the check first, so it cannot read a half-restored tree.
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        pkill -TERM -P "$pid" 2>/dev/null || true
        kill -TERM "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
    if [ "$applied" = 1 ]; then
        if ! git apply -R "$patch"; then
            echo "revert-run: could not reverse $patch; restoring the tracked files from HEAD" >&2
            git apply --numstat "$patch" | cut -f3- | xargs -r git checkout HEAD --
        fi
        applied=0
    fi
    rm -f "$out"
    if ! git diff --quiet HEAD -- || [ "$(git status --porcelain)" != "$before" ]; then
        echo "revert-run: TREE NOT RESTORED" >&2
        git status --short >&2
        exit 3
    fi
    echo "revert-run: tree restored (no tracked change, status as before the run)"
    exit "$rc"
}
trap restore EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

git apply "$patch"
applied=1
echo "revert-run: applied $(basename "$patch"); running: $*"

# In the background and waited on: bash runs a trap only between foreground
# commands, so a TERM during a foreground check would wait for it to finish.
# pipefail (set above) makes the subshell's status the check's, not tee's.
set +e
( "$@" 2>&1 | tee "$out" ) &
pid=$!
wait "$pid"
rc=$?
pid=""
set -e

if [ "$rc" -eq 0 ]; then
    echo "revert-run: ARM BROKEN: the check PASSED with $(basename "$patch") applied" >&2
    exit 1
fi
if [ -n "$expect" ] && ! grep -qF -- "$expect" "$out"; then
    echo "revert-run: ARM BROKEN: the check failed (exit $rc), but not with '$expect'" >&2
    exit 1
fi
# shellcheck disable=SC2016  # the quotes are literal text in the message
echo "revert-run: arm holds: the check failed (exit $rc) under $(basename "$patch")${expect:+ with '$expect'}"
exit 0
