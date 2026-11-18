#!/usr/bin/env bash
# scripts/demo/test-harness.sh: self-test of the demo check library and the revert
# harness (#456). No docker, no network: runs on every PR (ci.yml `demo-harness`).
#
#   scripts/demo/test-harness.sh
#
# Covers assert_log_order (in order, missing step, out of order, GNU escape, literal
# parentheses), demo_seed, demo_assert_private against a planted publish, every
# revert-run.sh exit path (arm holds, arm broken, --expect mismatch, stale patch,
# dirty tree, a TERM in the middle of the check), D441D's episode scorer and the
# staging of every d441d-check.sh arm (dry run), and that every committed patch
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

# #568: `\b` is a word boundary to grep -E and a backspace to awk; the step must
# be named non-portable, not "out of order" (run 43908256776), and the POSIX
# form must pass.
printf '%s\n' 'Uplink data (RLC): ue_id=0, drb_id=1, psi=1, len=475' 'IntentSubmitted (lb-1)' > "$work/gnu.log"
if err="$(assert_log_order "$work/gnu.log" t 'len=475\b' 'IntentSubmitted' 2>&1 >/dev/null)"; then
    not_ok "a GNU-escape step passed under awk"
fi
case "$err" in *"step 1/2: "*"grep -E but not awk"*) ok "assert_log_order: names a non-portable (GNU escape) step" ;; *) not_ok "gnu escape: $err" ;; esac
assert_log_order "$work/gnu.log" t 'len=475([^0-9]|$)' 'IntentSubmitted \(lb-1\)' >/dev/null \
    && ok "assert_log_order: the POSIX boundary passes" || not_ok "POSIX boundary failed"

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

# --- D441D (#467): the episode scorer and the check's dry run ------------------
python3 -I "$here/test_d441d_score.py" >"$work/d441d-score.out" 2>&1 \
    && ok "d441d_score.py: every outcome and every missing link (test_d441d_score.py)" \
    || { cat "$work/d441d-score.out" >&2; not_ok "test_d441d_score.py"; }
for arm in a3 lsq onnx off invalid conflict runtime-switch; do
    D441D_DRY_RUN=1 NEXTGCORE_DIR="$work" DEMO_ARTEFACTS="$work/d441d" "$here/d441d-check.sh" "$arm" \
        >"$work/d441d-dry.out" 2>&1 || { cat "$work/d441d-dry.out" >&2; not_ok "d441d-check.sh $arm dry run"; }
done
grep -qx '  seed: 24' "$work/d441d/d441d/lsq/seed-24/config/gnb-n2ho-target.yaml" \
    && grep -qx '    mode: a3' "$work/d441d/d441d/a3/seed-1/config/gnb.yaml" \
    && ok "d441d-check.sh: every arm stages (dry run), seeds 1-24 on both gNBs" \
    || not_ok "d441d staging"
# A config that drifted from the harness's expectations fails, never runs unseeded.
mkdir -p "$work/d441d-drift/scripts/demo" "$work/d441d-drift/config"
cp "$here"/d441d-check.sh "$here"/lib.sh "$work/d441d-drift/scripts/demo/"
cp -r "$repo/config/demo-predictive-ho" "$work/d441d-drift/config/"
sed -i 's/^  seed: 1$/  seed: 1\n  seed: 2/' "$work/d441d-drift/config/demo-predictive-ho/gnb-east.yaml"
if D441D_DRY_RUN=1 NEXTGCORE_DIR="$work" DEMO_ARTEFACTS="$work/d441d-drift/out" \
    "$work/d441d-drift/scripts/demo/d441d-check.sh" lsq >"$work/d441d-dry.out" 2>&1; then
    not_ok "a config with two seed lines staged"
fi
grep -q 'matches 2 line(s), expected 1' "$work/d441d-dry.out" \
    && ok "d441d-check.sh: a drifted config fails staging" || not_ok "drift: $(cat "$work/d441d-dry.out")"
# Staging that changes more than the arm's keys fails the dry run's diff check.
cp "$repo/config/demo-predictive-ho/gnb-east.yaml" "$work/d441d-drift/config/demo-predictive-ho/"
# shellcheck disable=SC2016  # the planted line is shell text for the copied script
sed -i 's|^    # The staged mode and seed are what the gNBs read.*|    printf "tac: 2\\n" >> "$dir/config/gnb.yaml"|' \
    "$work/d441d-drift/scripts/demo/d441d-check.sh"
grep -q '"tac: 2' "$work/d441d-drift/scripts/demo/d441d-check.sh" || not_ok "the planted staging change did not land"
if D441D_DRY_RUN=1 NEXTGCORE_DIR="$work" DEMO_ARTEFACTS="$work/d441d-drift/out2" \
    "$work/d441d-drift/scripts/demo/d441d-check.sh" lsq >"$work/d441d-dry.out" 2>&1; then
    not_ok "staging with an extra change passed the dry run"
fi
grep -q "changed beyond the arm's keys" "$work/d441d-dry.out" \
    && ok "d441d-check.sh: staging that changes more than the arm's keys fails the dry run" \
    || not_ok "extra change: $(cat "$work/d441d-dry.out")"

# A TERM mid-episode tears the stack down (`compose down`), against a fake docker.
mkdir -p "$work/fakebin"
cat > "$work/fakebin/docker" <<'FAKE'
#!/bin/sh
echo "$*" >> "$FAKE_DOCKER_LOG"
case "$*" in
    *" config --format json"*) echo '{"services":{"amf":{"image":"a"}}}' ;;
esac
exit 0
FAKE
chmod +x "$work/fakebin/docker"
: > "$work/fake-docker.log"
mkdir -p "$work/core/docker/rust"
: > "$work/core/docker/rust/docker-compose.demo-n2ho.yml"
PATH="$work/fakebin:$PATH" FAKE_DOCKER_LOG="$work/fake-docker.log" NEXTGCORE_DIR="$work/core" \
    DEMO_ARTEFACTS="$work/d441d-term" D441D_SEEDS=1 "$here/d441d-check.sh" a3 >"$work/d441d-term.out" 2>&1 &
cpid=$!
for _ in $(seq 1 100); do grep -q ' up -d' "$work/fake-docker.log" && break; sleep 0.1; done
grep -q ' up -d' "$work/fake-docker.log" || not_ok "the fake stack never came up: $(cat "$work/d441d-term.out")"
kill -TERM "$cpid"
rc=0; wait "$cpid" || rc=$?
[ "$rc" -eq 143 ] && grep -q ' down -v --remove-orphans' "$work/fake-docker.log" \
    && ok "d441d-check.sh: a TERM mid-episode takes the stack down (exit 143)" \
    || not_ok "TERM rc=$rc: $(cat "$work/fake-docker.log")"

# --- the committed revert patches still apply --------------------------------
n=0
for p in "$here"/reverts/*.patch; do
    [ -e "$p" ] || continue
    git -C "$repo" apply --check "$p" || not_ok "$(basename "$p") no longer applies to this tree"
    n=$((n + 1))
done
[ "$n" -gt 0 ] && ok "every patch under reverts/ applies ($n)" || not_ok "no revert patch found"

echo "all ${pass} harness checks passed"
