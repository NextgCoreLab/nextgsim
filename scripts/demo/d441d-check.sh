#!/usr/bin/env bash
# scripts/demo/d441d-check.sh: demo D441D, predictive N2 handover vs A3
# (nextgsim#467), seeded A/B on nextgcore's two-gNB stack.
#
#   NEXTGCORE_DIR=../nextgcore scripts/demo/d441d-check.sh lsq     # predictive arm, K seeds
#   NEXTGCORE_DIR=../nextgcore scripts/demo/d441d-check.sh a3      # the A3 baseline, same seeds
#   NEXTGCORE_DIR=../nextgcore scripts/demo/d441d-check.sh onnx    # a d440b-verified model, K seeds
#   NEXTGCORE_DIR=../nextgcore scripts/demo/d441d-check.sh off     # demo off (kill switch), one seed
#   NEXTGCORE_DIR=../nextgcore scripts/demo/d441d-check.sh invalid # mode: bogus fails gnb-west startup
#   NEXTGCORE_DIR=../nextgcore scripts/demo/d441d-check.sh conflict        # + D446A on one gNB
#   NEXTGCORE_DIR=../nextgcore scripts/demo/d441d-check.sh runtime-switch  # nr-cli set, one seed
#
# The stack is nextgcore docker/rust/docker-compose.demo-n2ho.yml (no published
# port: asserted before every `up`), with a staged config directory as
# NEXTGSIM_DIR, from config/demo-predictive-ho/:
#   gnb-west  gnb-west.yaml (the source, NCI 0x10) as gnb.yaml
#   gnb-east  gnb-east.yaml (the target, NCI 0x20, PCI 32) as gnb-n2ho-target.yaml
#   ue        ue.yaml (the committed arc from gnb-west toward gnb-east) as ue-n2ho.yaml
# Per seed only `shadowing.seed` changes, the same on both gNBs; per arm only
# `demo.d441d-predictive-ho.mode` (both gNBs). Each seed is a fresh stack
# (`down -v`, `up`), so no state carries from one seed to the next.
#
# One seeded episode: the UE registers on gnb-west, sets up a PDU session, and
# starts the route 40 s after it starts. The run waits for amfd's
# HandoverNotify, at most D441D_EPISODE_S after the UE's PDU session (a seed
# without one is a failure: decision Q4), then D441D_SETTLE_S more for the shadow A3, the
# lead time and any hand-back, then scores the seed with d441d_score.py (the
# chain, the lead's range, ping-pongs within D441D_PINGPONG_MS, returns,
# failures, A3 fallbacks, stranded, exhausted). Each seed also asserts the two
# gNBs logged the same `RLS: seeded log-normal shadowing on:` line, with this
# seed (the replica needs both blocks equal; #534 review m4).
#
# Pass (operator decision 2026-11-18, specs/demo-d441d-pr3b-pr4-harness-467.md):
#   lsq, onnx   the chain on >= K-2 of K seeds when K >= 10, else on all K: prediction -> NWDAF-recommended
#               -> Sent Handover Required -> amfd HandoverNotify (and gnb-east's
#               Sent Handover Notify) -> A3 would trigger -> lead time, with
#               predictor=<arm>. Lead time is not asserted positive.
#   a3          the chain on all K: measurement-triggered N2 handover to PCI 32
#               -> Sent Handover Required -> HandoverNotify, no predictor line,
#               and the UE's real report inside the replica's range
#               (+-D441D_A3_SLACK_MS): this validates the replica the lead is
#               measured against.
#   every seed  a HandoverNotify more than D441D_EPISODE_S after the route start,
#               or none, is a failure (decision Q4), counted in `failures`.
#   off         the base A3 handover, no DEMO MODE ACTIVE or [demo: line on
#               either gNB, and `demo d441d-predictive-ho show` active=false.
#   invalid     gnb-west logs UnknownMode for d441d-predictive-ho/bogus and
#               does not stay up.
#   conflict    gnb-west (built with demo-d446a-agent-conflict too) logs
#               HandoverDemoConflict and does not stay up.
#   runtime-switch  starts in a3; before the route starts, `set mode=bogus` is
#               rejected and audited and `show` stays a3; `set mode=lsq` is
#               audited; the seed then scores as a predictive lsq episode.
# The table (Markdown) is written to $DEMO_ARTEFACTS/d441d/<arm>/table.md and
# the per-seed rows to rows.tsv; both go to the job summary under GitHub Actions.
#
# The gNB image must be built with --features
# demo-d441d-predictive-ho,demo-d446a-agent-conflict (one image for every arm;
# the D446A key is set only in `conflict`).
#
# Environment:
#   NEXTGCORE_DIR      nextgcore checkout (default: ../nextgcore beside this repo)
#   DEMO_ARTEFACTS     where logs are captured (default /tmp/demo-artefacts)
#   D441D_SEEDS        seeds as "A-B" or a space list (default 1-24; one-seed arms take the first)
#   D441D_SIGMA_DB     shadowing sigma (default 4, as committed)
#   D441D_MIN_CHAINS   seeds that must hold the chain, 1..K (default: lsq/onnx K-2
#                      when K >= 10, else K; a3 K)
#   D441D_EPISODE_S    the episode window: HandoverNotify within this many
#                      seconds of the UE's route start (default 120, decision Q4)
#   D441D_SETTLE_S     then wait this long before scoring (default 15)
#   D441D_PINGPONG_MS  ping-pong window after HandoverNotify (default 5000)
#   D441D_A3_SLACK_MS  a3: the UE's report may fall this far outside the
#                      replica's range and still count (default 250)
#   D441D_ORT_LIB      libonnxruntime.so.1.23.2 for the onnx arm (required there)
#   NR_CLI             nr-cli to copy into gnb-west (default binaries/nr-cli;
#                      required by off and runtime-switch)
#   D441D_KEEP=1       leave the last stack running
#   D441D_DRY_RUN=1    no docker: stage every seed's configs, check them, print the plan
# Exit status: 0 when the arm passes, 1 otherwise (DEMO CHECK FAILED: ...), 2 usage.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/demo/lib.sh
. "$here/lib.sh"

arm="${1:-}"
case "$arm" in
    a3|lsq|onnx|off|invalid|conflict|runtime-switch) ;;
    *) echo "usage: $0 a3|lsq|onnx|off|invalid|conflict|runtime-switch" >&2; exit 2 ;;
esac

sim="$(cd "$here/../.." && pwd)"
core="$(cd "${NEXTGCORE_DIR:-$sim/../nextgcore}" && pwd)"
demo_artefacts "${DEMO_ARTEFACTS:-/tmp/demo-artefacts}/d441d/$arm"
root_artefacts="$DEMO_ARTEFACTS"
src="$sim/config/demo-predictive-ho"
score_py="$here/d441d_score.py"
nr_cli="${NR_CLI:-$sim/binaries/nr-cli}"
sigma="${D441D_SIGMA_DB:-4}"
episode_s="${D441D_EPISODE_S:-120}"
settle_s="${D441D_SETTLE_S:-15}"
pingpong_ms="${D441D_PINGPONG_MS:-5000}"
a3_slack_ms="${D441D_A3_SLACK_MS:-250}"
for v in "$sigma" "$episode_s" "$settle_s" "$pingpong_ms" "$a3_slack_ms"; do
    case "$v" in '' | *[!0-9]*) echo "numeric knobs must be decimal integers, got '$v'" >&2; exit 2 ;; esac
done

# The seeds: "A-B" or a list. Committed before the first run (decision Q5).
seeds_spec="${D441D_SEEDS:-1-24}"
seeds=()
if [[ "$seeds_spec" =~ ^([0-9]+)-([0-9]+)$ ]]; then
    for ((s = BASH_REMATCH[1]; s <= BASH_REMATCH[2]; s++)); do seeds+=("$s"); done
else
    for s in $seeds_spec; do
        case "$s" in *[!0-9]*) echo "D441D_SEEDS: not a seed '$s'" >&2; exit 2 ;; esac
        seeds+=("$s")
    done
fi
[ "${#seeds[@]}" -gt 0 ] || { echo "D441D_SEEDS names no seed" >&2; exit 2; }
case "$arm" in a3|lsq|onnx) ;; *) seeds=("${seeds[0]}") ;; esac
k="${#seeds[@]}"
# The pass bar (operator decision 2026-11-18; the coordinator's ruling on R14):
# lsq and onnx need K-2 of K for a table of at least 10 seeds, else all K; a3
# always all K. An override must be within 1..K, never a vacuous 0.
case "$arm" in
    lsq|onnx) min_chains="${D441D_MIN_CHAINS:-$(( k >= 10 ? k - 2 : k ))}" ;;
    *) min_chains="${D441D_MIN_CHAINS:-$k}" ;;
esac
case "$min_chains" in '' | *[!0-9]*) echo "D441D_MIN_CHAINS must be a decimal integer" >&2; exit 2 ;; esac
if [ "$min_chains" -lt 1 ] || [ "$min_chains" -gt "$k" ]; then
    echo "D441D_MIN_CHAINS=$min_chains is outside 1..$k" >&2; exit 2
fi
# A dry run stages an empty stand-in for the library and needs no nr-cli.
if [ "${D441D_DRY_RUN:-0}" = 1 ]; then D441D_ORT_LIB="${D441D_ORT_LIB:-/dev/null}"; nr_cli=/bin/true; fi
if [ "$arm" = onnx ]; then
    { [ -f "${D441D_ORT_LIB:-}" ] || [ "${D441D_DRY_RUN:-0}" = 1 ]; } || { echo "the onnx arm needs D441D_ORT_LIB (libonnxruntime.so.1.23.2)" >&2; exit 2; }
fi
case "$arm" in off|runtime-switch)
    [ -x "$nr_cli" ] || { echo "the $arm arm needs nr-cli at $nr_cli (NR_CLI)" >&2; exit 2; } ;;
esac

# --- staging ----------------------------------------------------------------
# set_exactly FILE ERE REPLACEMENT: rewrite the one line matching ERE, or fail:
# a config that drifted from what the harness expects must never run unseeded.
set_exactly() {
    local f="$1" re="$2" to="$3" n
    n="$(grep -cE -- "$re" "$f" || true)"
    [ "$n" = 1 ] || { demo_fail "$(basename "$f"): /$re/ matches $n line(s), expected 1"; return 1; }
    sed -i -E "s|$re|$to|" "$f"
}

# stage_configs DIR SEED MODE: the three configs of one episode under DIR/config.
# MODE is the d441d mode, or "off" (demo.enabled: false on both gNBs).
stage_configs() {
    local dir="$1" seed="$2" mode="$3" f
    mkdir -p "$dir/config"
    cp "$src/gnb-west.yaml" "$dir/config/gnb.yaml"
    cp "$src/gnb-east.yaml" "$dir/config/gnb-n2ho-target.yaml"
    cp "$src/ue.yaml" "$dir/config/ue-n2ho.yaml"
    for f in "$dir/config/gnb.yaml" "$dir/config/gnb-n2ho-target.yaml"; do
        set_exactly "$f" '^  seed: [0-9]+$' "  seed: $seed"
        set_exactly "$f" '^  sigma_db: [0-9.]+$' "  sigma_db: $sigma"
        if [ "$mode" = off ]; then
            set_exactly "$f" '^  enabled: true$' '  enabled: false'
        else
            set_exactly "$f" '^    mode: lsq$' "    mode: $mode"
        fi
    done
    if [ "$mode" = onnx ]; then
        # d441d onnx needs d440b-model-verify in verify on the same gNB, the
        # model (tools/d440b-model, #463) and ONNX Runtime (the d440b precedent).
        mkdir -p "$dir/config/models" "$dir/config/ort"
        cp "$sim/tools/d440b-model/nwdaf-trajectory-cv.onnx" \
           "$sim/tools/d440b-model/nwdaf-trajectory-cv.manifest.json" "$dir/config/models/"
        cp "$D441D_ORT_LIB" "$dir/config/ort/libonnxruntime.so.1.23.2"
        for f in "$dir/config/gnb.yaml" "$dir/config/gnb-n2ho-target.yaml"; do
            printf '  d440b-model-verify:\n    enabled: true\n    mode: verify\nnwdaf_model_path: /etc/nextgsim/models/nwdaf-trajectory-cv.onnx\n' >> "$f"
        done
    fi
    if [ "$arm" = conflict ]; then
        printf '  d446a-agent-conflict:\n    enabled: true\n    mode: load-balancing\n' >> "$dir/config/gnb.yaml"
    fi
    # The staged mode and seed are what the gNBs read (a guard on the sed above).
    for f in "$dir/config/gnb.yaml" "$dir/config/gnb-n2ho-target.yaml"; do
        grep -qx "  seed: $seed" "$f" || { demo_fail "$(basename "$f"): seed $seed not staged"; return 1; }
    done
}

# The onnx arm's override: ONNX Runtime from the staged directory (no port, no
# other change; demo_assert_private checks the merged model).
onnx_override() {
    cat > "$1" <<'YAML'
services:
  gnb-west:
    environment:
      ORT_DYLIB_PATH: /etc/nextgsim/ort/libonnxruntime.so.1.23.2
  gnb-east:
    environment:
      ORT_DYLIB_PATH: /etc/nextgsim/ort/libonnxruntime.so.1.23.2
YAML
}

# --- one stack ----------------------------------------------------------------
stage=""
stack_up=false
cleanup_stack() {
    if [ "$stack_up" = true ] && [ "${D441D_KEEP:-0}" != 1 ]; then
        demo_compose down -v --remove-orphans >/dev/null 2>&1 || true
    fi
    stack_up=false
    if [ -n "$stage" ]; then rm -r "$stage" 2>/dev/null || true; fi
    stage=""
}
teardown() {
    local rc=$?
    if [ "$stack_up" = true ]; then capture "$DEMO_ARTEFACTS" || true; fi
    cleanup_stack
    exit "$rc"
}
# Bash runs the EXIT trap on a TERM too, so a cancelled run never leaves a
# stack up (test-harness.sh sends TERM mid-episode).
trap teardown EXIT

# start_stack SEED MODE: stage, assert private, `up`.
start_stack() {
    local seed="$1" mode="$2" refusal="${3:-}" files
    stage="$(mktemp -d -t d441d-XXXXXX)"
    stage_configs "$stage" "$seed" "$mode"
    rm -rf "$DEMO_ARTEFACTS/config"
    cp -r "$stage/config" "$DEMO_ARTEFACTS/config"
    rm -rf "$DEMO_ARTEFACTS/config/ort"   # the runtime library is not an artefact
    export NEXTGSIM_DIR="$stage"
    files=("$core/docker/rust/docker-compose.demo-n2ho.yml")
    if [ "$mode" = onnx ]; then onnx_override "$stage/onnx.compose.yml"; files+=("$stage/onnx.compose.yml"); fi
    demo_compose_init "${files[@]}"
    demo_assert_private
    stack_up=true
    if [ "$refusal" = refusal ]; then
        # gnb-west is meant to refuse its config, so it never turns healthy and
        # `up -d` fails on the services that wait for it (ue). refused_at_startup
        # is the check; a missing refusal still fails there.
        demo_compose up -d || demo_info "compose up -d failed, as it must when gnb-west refuses its config"
    else
        demo_compose up -d
    fi
}

# Timestamped logs (`docker compose logs -t`): the scorer compares times across
# services, and the gNB's own t= values are Unix ms on the same host clock.
capture() {
    local dir="$1" s
    for s in amf smf upf gnb-west gnb-east ue; do
        demo_compose logs --no-color -t "$s" > "$dir/$s.log" 2>&1 || true
    done
}

cli_west() {
    local out="$1"; shift
    local id node
    id="$(demo_compose ps -q gnb-west)"
    docker cp "$nr_cli" "$id:/tmp/nr-cli" >/dev/null 2>&1 || { echo "(nr-cli copy failed)" > "$out"; return 0; }
    node="$(demo_compose exec -T gnb-west /tmp/nr-cli --dump 2>/dev/null | head -1 | tr -d '\r')"
    demo_compose exec -T gnb-west /tmp/nr-cli "$node" -e "$*" > "$out" 2>&1 || true
}

# The m4 check: both gNBs logged the same shadowing line, with this seed.
shadowing_agrees() {
    local seed="$1" w e
    w="$(grep -oE 'RLS: seeded log-normal shadowing on: [^(]*' "$DEMO_ARTEFACTS/gnb-west.log" | head -1)"
    e="$(grep -oE 'RLS: seeded log-normal shadowing on: [^(]*' "$DEMO_ARTEFACTS/gnb-east.log" | head -1)"
    [ -n "$w" ] || { demo_fail "[gnb-west] no 'RLS: seeded log-normal shadowing on:' line"; return 1; }
    [ "$w" = "$e" ] || { demo_fail "the gNBs' shadowing lines differ: west '$w', east '$e'"; return 1; }
    case "$w" in *" seed=$seed "*) ;; *) demo_fail "the shadowing line names another seed: '$w'"; return 1 ;; esac
}

# A startup refusal: gnb-west logs RE and is not running (restart: on-failure
# retries it, so "not running" is read as "has exited at least once").
refused_at_startup() {
    local re="$1" id restarts running exit_code
    demo_wait_log gnb-west "$re" 90
    id="$(demo_compose ps -a -q gnb-west)"
    restarts="$(docker inspect -f '{{.RestartCount}}' "$id")"
    running="$(docker inspect -f '{{.State.Running}}' "$id")"
    exit_code="$(docker inspect -f '{{.State.ExitCode}}' "$id")"
    if [ "$running" = true ] && [ "$restarts" = 0 ]; then
        demo_fail "[gnb-west] logged /$re/ but is still running (never exited)"
    fi
    demo_info "[gnb-west] refused at startup: restarts=$restarts running=$running exit=$exit_code"
}

# --- dry run -------------------------------------------------------------------
# D441D_DRY_RUN=1: no docker. Stage every seed's configs under the artefacts and
# check each differs from the committed one only in the lines the arm changes
# (the seed, sigma, the mode or the kill switch, and the appended onnx/conflict
# blocks); then print the plan.
if [ "${D441D_DRY_RUN:-0}" = 1 ]; then
    case "$arm" in off) mode=off ;; runtime-switch) mode=a3 ;; invalid) mode=bogus ;; conflict) mode=lsq ;; *) mode="$arm" ;; esac
    for seed in "${seeds[@]}"; do
        d="$root_artefacts/seed-$seed"
        rm -rf "$d"; mkdir -p "$d"
        stage_configs "$d" "$seed" "$mode"
        for pair in "gnb-west.yaml gnb.yaml" "gnb-east.yaml gnb-n2ho-target.yaml" "ue.yaml ue-n2ho.yaml"; do
            read -r from to <<< "$pair"
            changed="$(diff "$src/$from" "$d/config/$to" | grep -E '^[<>]' | sed -E 's/^[<>] //' \
                | grep -vxE '  seed: [0-9]+|  sigma_db: [0-9.]+|    mode: [a-z0-9-]+|  enabled: (true|false)' \
                | grep -vxE '  d440b-model-verify:|  d446a-agent-conflict:|    enabled: true|nwdaf_model_path: .*' || true)"
            [ -z "$changed" ] || { printf '%s\n' "$changed" >&2; demo_fail "$to (seed $seed): changed beyond the arm's keys"; }
        done
        printf '[dry-run] seed %s: %s\n' "$seed" "$(grep -hE '^  seed: |^    mode: |^  enabled: ' "$d/config/gnb.yaml" | tr -s ' ' | tr '\n' ' ')"
    done
    rm -rf "$root_artefacts"/seed-*/config/ort
    demo_info "D441D $arm dry run: ${#seeds[@]} seed(s) staged, pass needs $min_chains chained (no docker)"
    trap - EXIT
    exit 0
fi

# --- the arms -----------------------------------------------------------------
rows="$root_artefacts/rows.tsv"
: > "$rows"

case "$arm" in
invalid|conflict)
    seed="${seeds[0]}"
    if [ "$arm" = invalid ]; then mode=bogus; else mode=lsq; fi
    start_stack "$seed" "$mode" refusal
    if [ "$arm" = invalid ]; then
        refused_at_startup 'unknown mode "bogus" for demo "d441d-predictive-ho"'
    else
        refused_at_startup 'HandoverDemoConflict: demos d441d-predictive-ho and d446a-agent-conflict both trigger handovers'
    fi
    capture "$DEMO_ARTEFACTS"
    if grep -qE 'Sent Handover Required' "$DEMO_ARTEFACTS/gnb-west.log"; then
        demo_fail "[gnb-west] handed a UE over although its config was refused"
    fi
    cleanup_stack
    demo_info "D441D $arm: gnb-west refused the config at startup (logs in $root_artefacts)"
    exit 0
    ;;
esac

for seed in "${seeds[@]}"; do
    # A rerun into the same DEMO_ARTEFACTS starts each seed from an empty directory.
    rm -rf "$root_artefacts/seed-$seed"
    demo_artefacts "$root_artefacts/seed-$seed"
    demo_seed "$seed" >/dev/null
    case "$arm" in
        off) mode=off ;;
        runtime-switch) mode=a3 ;;
        *) mode="$arm" ;;
    esac
    start_stack "$seed" "$mode"
    demo_wait_log gnb-west 'Received NG Setup Response' 120
    demo_wait_log gnb-east 'Received NG Setup Response' 60
    demo_wait_log ue 'Received Registration Accept' 120
    demo_wait_log ue 'is now ACTIVE' 60
    demo_info "seed $seed ($mode): registered on gnb-west with a PDU session"

    if [ "$arm" = runtime-switch ]; then
        # Before the route starts (40 s after the UE): the switch applies from
        # the next NWDAF tick, so this episode is the lsq one.
        cli_west "$DEMO_ARTEFACTS/set-bogus.txt" demo d441d-predictive-ho set mode=bogus
        demo_wait_log gnb-west 'audit: rejected set mode="bogus" from nr-cli' 20
        cli_west "$DEMO_ARTEFACTS/show-after-bogus.txt" demo d441d-predictive-ho show
        grep -q 'demo=d441d-predictive-ho active=true mode=a3 ' "$DEMO_ARTEFACTS/show-after-bogus.txt" \
            || demo_fail "[gnb-west] the mode changed on a rejected set: $(cat "$DEMO_ARTEFACTS/show-after-bogus.txt")"
        cli_west "$DEMO_ARTEFACTS/set-lsq.txt" demo d441d-predictive-ho set mode=lsq
        demo_wait_log gnb-west 'audit: mode a3 -> lsq from nr-cli' 20
        demo_info "runtime switch a3 -> lsq applied; bogus was rejected"
    fi

    # Decision Q4: the window runs from the route start. The full amfd line, so
    # a `HandoverNotify ... rejected` does not end the wait (it scores as a failure).
    demo_wait_log ue 'Simulated motion started' 90
    demo_wait_log amf 'HandoverNotify from association [0-9]+ \(amf_ue_ngap_id=' "$episode_s" 2>/dev/null \
        || demo_info "seed $seed: no HandoverNotify within ${episode_s}s of the route start (scored as a failure)"
    sleep "$settle_s"
    capture "$DEMO_ARTEFACTS"
    if [ -x "$nr_cli" ]; then
        cli_west "$DEMO_ARTEFACTS/nwdaf-status.json" nwdaf-status
        cli_west "$DEMO_ARTEFACTS/demo-show.txt" demo d441d-predictive-ho show
        cli_west "$DEMO_ARTEFACTS/demo-metrics.prom" demo metrics
    fi

    # Per-seed hard checks: these are harness or config faults, not outcomes.
    if [ "$mode" = off ]; then
        grep -q 'demo=d441d-predictive-ho active=false ' "$DEMO_ARTEFACTS/demo-show.txt" \
            || demo_fail "[gnb-west] demo show is not disabled: $(cat "$DEMO_ARTEFACTS/demo-show.txt")"
        if grep -q 'nextg_demo_enabled' "$DEMO_ARTEFACTS/demo-metrics.prom"; then
            demo_fail "[gnb-west] demo metrics exported with the demo off"
        fi
    else
        # The banner lists every active demo, comma-separated (d440b too in onnx).
        grep -qE "DEMO MODE ACTIVE: ([^ ]*,)?d441d-predictive-ho=$mode(,| |\$)" "$DEMO_ARTEFACTS/gnb-west.log" \
            || demo_fail "[gnb-west] no 'DEMO MODE ACTIVE: d441d-predictive-ho=$mode' banner"
    fi
    shadowing_agrees "$seed"
    if [ "$mode" = onnx ]; then
        grep -q 'NWDAF: model verified' "$DEMO_ARTEFACTS/gnb-west.log" \
            || demo_fail "[gnb-west] onnx arm without 'NWDAF: model verified'"
    fi

    score_arm="$arm"
    [ "$arm" = runtime-switch ] && score_arm=lsq
    header=()
    [ -s "$rows" ] || header=(--header)
    python3 -I "$score_py" score --arm "$score_arm" --seed "$seed" "${header[@]}" \
        --west "$DEMO_ARTEFACTS/gnb-west.log" --east "$DEMO_ARTEFACTS/gnb-east.log" \
        --amf "$DEMO_ARTEFACTS/amf.log" --ue "$DEMO_ARTEFACTS/ue.log" --window-ms "$((episode_s * 1000))" \
        --pingpong-ms "$pingpong_ms" --a3-slack-ms "$a3_slack_ms" | tee -a "$rows"
    # The seed's verdict in words (the revert arm expects "chain=prediction").
    awk -F'\t' -v s="$seed" '$1 == s { printf "[demo] seed %s: outcome=%s chain=%s\n", s, $3, $4 }' "$rows"
    cleanup_stack
done

DEMO_ARTEFACTS="$root_artefacts"
score_arm="$arm"
[ "$arm" = runtime-switch ] && score_arm=lsq
rc=0
python3 -I "$score_py" table --arm "$score_arm" --seeds "${seeds[*]}" --min-chains "$min_chains" "$rows" \
    > "$root_artefacts/table.md" || rc=$?
cat "$root_artefacts/table.md"
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    { echo "### D441D arm \`$arm\` (seeds ${seeds[*]}, sigma ${sigma} dB)"; cat "$root_artefacts/table.md"; echo; } >> "$GITHUB_STEP_SUMMARY"
fi
[ "$rc" = 0 ] || demo_fail "[$arm] fewer than $min_chains of $k seeds held the chain (see $root_artefacts/rows.tsv)"
demo_info "D441D $arm: $k seed(s) scored, the arm passes (logs in $root_artefacts)"
