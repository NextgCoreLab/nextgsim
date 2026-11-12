#!/usr/bin/env bash
# scripts/demo/test-harness.sh: self-test of the demo check library and the revert
# harness (#456). No docker, no network: runs on every PR (ci.yml `demo-harness`).
#
#   scripts/demo/test-harness.sh
#
# Covers assert_log_order (in order, missing step, out of order, literal
# parentheses), demo_seed, demo_assert_private against a planted publish, every
# revert-run.sh exit path (arm holds, arm broken, --expect mismatch, stale patch,
# dirty tree, a TERM in the middle of the check), and that every committed patch
# under reverts/ still applies to this tree.
# `cond && ok ... || not_ok ...` throughout: ok only prints and cannot fail.
# shellcheck disable=SC2015
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
work="$(mktemp -d)"
trap 'find "$work" -mindepth 1 -delete 2>/dev/null; rmdir "$work" 2>/dev/null || true' EXIT

pass=0
ok() { pass=$((pass + 1)); printf 'ok %d - %s\n' "$pass" "$*"; }
not_ok() { printf 'not ok - %s\n' "$*" >&2; exit 1; }

# --- lib.sh -----------------------------------------------------------------
# shellcheck source=scripts/demo/lib.sh
. "$here/lib.sh"
export DEMO_ARTEFACTS="$work/artefacts"

log="$work/chain.log"
printf '%s\n' 'boot' 'HandoverRequired (amf_ue_ngap_id=1, type=Intra5gs)' 'noise' \
    'HandoverCommand sent to source association 1' 'HandoverNotify from association 2' > "$log"

assert_log_order "$log" t 'HandoverRequired \(amf_ue_ngap_id=1' 'HandoverCommand sent' 'HandoverNotify from' >/dev/null \
    && ok "assert_log_order: a chain in order passes" || not_ok "in-order chain failed"

if err="$(assert_log_order "$log" t 'HandoverRequired' 'HandoverRequest forwarded' 'HandoverNotify' 2>&1 >/dev/null)"; then
    not_ok "a missing step passed"
fi
case "$err" in *"step 2/3 missing"*) ok "assert_log_order: names the missing step" ;; *) not_ok "missing: $err" ;; esac

if err="$(assert_log_order "$log" t 'HandoverNotify' 'HandoverRequired' 2>&1 >/dev/null)"; then
    not_ok "an out-of-order chain passed"
fi
case "$err" in *"step 2/2 out of order"*) ok "assert_log_order: names an out-of-order step" ;; *) not_ok "order: $err" ;; esac

# `\(` must stay a literal parenthesis (awk -v would have unescaped it into a group).
printf '%s\n' 'HandoverRequired amf_ue_ngap_id=1' > "$work/noparen.log"
if assert_log_order "$work/noparen.log" t 'HandoverRequired \(amf_ue_ngap_id' >/dev/null 2>&1; then
    not_ok "an escaped parenthesis matched a line without one"
fi
ok "assert_log_order: escaped parentheses are literal"

demo_seed 4242 >/dev/null
[ "$DEMO_SEED" = 4242 ] && [ "$(cat "$DEMO_ARTEFACTS/seed")" = 4242 ] \
    && ok "demo_seed: exports and records a given seed" || not_ok "demo_seed 4242"
if (demo_seed 'x1' >/dev/null 2>&1); then not_ok "a non-numeric seed was accepted"; fi
ok "demo_seed: rejects a non-numeric seed"
unset DEMO_SEED
demo_seed >/dev/null
case "$DEMO_SEED" in '' | *[!0-9]*) not_ok "fresh seed '$DEMO_SEED'" ;; *) ok "demo_seed: draws a fresh seed" ;; esac

# demo_compose_init merges override files, in order (#462), and refuses a missing one.
: > "$work/base.yml"
: > "$work/over.yml"
demo_compose_init "$work/base.yml" "$work/over.yml"
[ "${DEMO_COMPOSE[*]}" = "docker compose -f $work/base.yml -f $work/over.yml" ] \
    && ok "demo_compose_init: base then overrides, in order" || not_ok "init: ${DEMO_COMPOSE[*]}"
if (demo_compose_init "$work/base.yml" "$work/absent.yml" >/dev/null 2>&1); then
    not_ok "a missing override was accepted"
fi
ok "demo_compose_init: refuses a missing override"

# demo_assert_private reads `docker compose config --format json`; a stub stands in.
fake_config() { cat "$work/model.json"; }
DEMO_COMPOSE=(fake_config)
printf '%s' '{"services":{"amf":{"image":"a"},"upf":{"image":"u","ports":[{"target":2152,"published":"2152","protocol":"udp"}]}}}' > "$work/model.json"
if demo_assert_private >/dev/null 2>&1; then not_ok "a published port passed"; fi
ok "demo_assert_private: fails on a published port"
printf '%s' '{"services":{"gnb":{"image":"g","network_mode":"host"}}}' > "$work/model.json"
if demo_assert_private >/dev/null 2>&1; then not_ok "host networking passed"; fi
ok "demo_assert_private: fails on host networking"
printf '%s' '{"services":{"amf":{"image":"a"},"gnb":{"image":"g","ports":[]}}}' > "$work/model.json"
demo_assert_private >/dev/null && ok "demo_assert_private: passes a private stack" || not_ok "private stack failed"

# --- revert-run.sh ----------------------------------------------------------
rr="$here/revert-run.sh"
t="$work/tree"
mkdir -p "$t"
git -C "$t" init -q
git -C "$t" config user.email t@example.invalid
git -C "$t" config user.name t
printf 'feature: on\n' > "$t/conf.yaml"
printf '#!/bin/sh\necho checking\ngrep -q "feature: on" conf.yaml || { echo "DEMO CHECK FAILED: feature off"; exit 1; }\n' > "$t/check.sh"
chmod +x "$t/check.sh"
git -C "$t" add -A
git -C "$t" commit -qm init
touch "$t/untracked.txt"          # an untracked file must survive, status unchanged
status0="$(git -C "$t" status --porcelain)"

sed -i 's/on/off/' "$t/conf.yaml"
git -C "$t" diff > "$work/breaks.patch"
git -C "$t" checkout -q conf.yaml
printf 'feature: on\nother: 1\n' > "$t/conf.yaml"
git -C "$t" diff > "$work/harmless.patch"
git -C "$t" checkout -q conf.yaml

clean() { # the tree is exactly HEAD plus the untracked file
    git -C "$t" diff --quiet HEAD -- && [ "$(git -C "$t" status --porcelain)" = "$status0" ] \
        && grep -qx 'feature: on' "$t/conf.yaml" && [ "$(wc -l < "$t/conf.yaml")" -eq 1 ]
}
run() { (cd "$t" && "$rr" "$@") >"$work/rr.out" 2>&1; }

run "$work/breaks.patch" -- ./check.sh && clean \
    && ok "revert-run: an arm whose patch breaks the check holds (exit 0), tree clean" \
    || not_ok "breaks.patch: $(cat "$work/rr.out")"

run --expect 'feature off' "$work/breaks.patch" -- ./check.sh && clean \
    && ok "revert-run: --expect matches the named failure" || not_ok "--expect match: $(cat "$work/rr.out")"

rc=0; run --expect 'no such reason' "$work/breaks.patch" -- ./check.sh || rc=$?
[ "$rc" -eq 1 ] && clean && ok "revert-run: a failure without the --expect text breaks the arm (exit 1)" \
    || not_ok "--expect mismatch rc=$rc"

rc=0; run "$work/harmless.patch" -- ./check.sh || rc=$?
[ "$rc" -eq 1 ] && clean && grep -q 'ARM BROKEN' "$work/rr.out" \
    && ok "revert-run: a check that still passes under the patch breaks the arm (exit 1), tree clean" \
    || not_ok "harmless.patch rc=$rc"

printf -- '--- a/conf.yaml\n+++ b/conf.yaml\n@@ -1 +1 @@\n-feature: missing\n+feature: off\n' > "$work/stale.patch"
rc=0; run "$work/stale.patch" -- ./check.sh || rc=$?
[ "$rc" -eq 2 ] && clean && ok "revert-run: a stale patch is an error (exit 2), never a passing arm" \
    || not_ok "stale.patch rc=$rc"

printf 'feature: on\n# edit\n' > "$t/conf.yaml"
rc=0; run "$work/breaks.patch" -- ./check.sh || rc=$?
[ "$rc" -eq 2 ] && grep -qx '# edit' "$t/conf.yaml" && ok "revert-run: refuses a dirty tree (exit 2) and leaves it alone" \
    || not_ok "dirty tree rc=$rc"
git -C "$t" checkout -q conf.yaml

# A TERM while the check runs: the trap restores the tree.
printf '#!/bin/sh\necho started > %s\nsleep 30\n' "$work/started" > "$work/slow.sh"
chmod +x "$work/slow.sh"
(cd "$t" && exec "$rr" "$work/breaks.patch" -- "$work/slow.sh") >"$work/rr.out" 2>&1 &
rpid=$!
for _ in $(seq 1 50); do [ -s "$work/started" ] && break; sleep 0.1; done
[ -s "$work/started" ] || not_ok "the slow check never started"
grep -qx 'feature: off' "$t/conf.yaml" || not_ok "the patch was not applied while the check ran"
kill -TERM "$rpid"
rc=0; wait "$rpid" || rc=$?
[ "$rc" -eq 143 ] && clean && ok "revert-run: a TERM mid-check restores the tree (exit 143)" \
    || not_ok "TERM rc=$rc: $(cat "$work/rr.out")"

# --- the committed revert patches still apply --------------------------------
n=0
for p in "$here"/reverts/*.patch; do
    [ -e "$p" ] || continue
    git -C "$repo" apply --check "$p" || not_ok "$(basename "$p") no longer applies to this tree"
    n=$((n + 1))
done
[ "$n" -gt 0 ] && ok "every patch under reverts/ applies ($n)" || not_ok "no revert patch found"

echo "all ${pass} harness checks passed"
