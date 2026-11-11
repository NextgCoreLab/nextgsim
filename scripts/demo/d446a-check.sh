#!/usr/bin/env bash
# scripts/demo/d446a-check.sh: demo D446A (nextgsim#483), load-balancing vs
# energy-saving agents over a booster cell, against nextgcore's two-gNB stack.
#
#   NEXTGCORE_DIR=../nextgcore scripts/demo/d446a-check.sh load-balancing   # main arm
#   NEXTGCORE_DIR=../nextgcore scripts/demo/d446a-check.sh energy-saving    # swap arm
#   NEXTGCORE_DIR=../nextgcore scripts/demo/d446a-check.sh off              # demo off
#   NEXTGCORE_DIR=../nextgcore scripts/demo/d446a-check.sh runtime-switch   # nr-cli set
#
# The stack is nextgcore docker/rust/docker-compose.demo-n2ho.yml (no published
# port: asserted before `up`), with a staged config directory as NEXTGSIM_DIR:
#   gnb-west = gNB-B, the booster (config/demo-agent-energy/gnb-b.yaml), energy-1
#   gnb-east = gNB-A, the coverage cell (config/demo-agent-energy/gnb-a.yaml), lb-1
#   ue       = config/demo-agent-energy/ue.yaml, camped on gNB-A
# Only demo.d446a-agent-conflict.mode differs between the two arms (gNB-B's).
# `off` stages the same files with `demo.enabled: false` (the kill switch).
# `runtime-switch` starts in load-balancing; after the first contest it runs
# `nr-cli <gnb-b> demo d446a-agent-conflict set mode=energy-saving`, then starts
# a second UE (IMSI 999700000000002, a second nr-ue process in the ue container,
# its interface uesimtunb0) on gNB-A for the second contest.
# The gNB image must be built with --features demo-d446a-agent-conflict.
#
# Asserted (the strings are the ones the code emits):
#   load-balancing  amf: HandoverRequired ... source_gnb_id=2
#                   cause=RadioNetwork(ReduceLoadInServingCell) exactly once ->
#                   UE <n> relocated to target gNB -> UE Context Release Command
#                   sent to SOURCE; gnb-west: Handover Request received with that
#                   cause -> the handback hold line, and no HANDOVER REQUIRED of
#                   its own; gnb-west audit: IntentSubmitted (energy-1) ->
#                   InboundIntent (lb-1@gnb-2) -> ConflictDetected ... Cell:16 ->
#                   Resolved ... winner lb-1 -> Deferred (energy-1) -> GuardrailHold
#                   ... ues=1, with one Executed entry.
#   energy-saving   gnb-west: Resolved ... winner energy-1 -> Sent Handover Failure
#                   ... CellNotAvailable -> Agent switch-off: cell 16 off the air ->
#                   off the air: N heartbeat(s) dropped; amf: HandoverFailure ...
#                   gnb_id=1; gnb-east: Handover Preparation Failure -> audit
#                   ActuationFailed (lb-1); ping through gNB-A 20/20.
#   runtime-switch  gnb-west: Resolved ... winner lb-1 (policy load-balancing) ->
#                   the nextgsim-demo set audit (load-balancing -> energy-saving
#                   from nr-cli) -> PolicyChanged from=load-balancing
#                   to=energy-saving by=cli -> Resolved ... winner energy-1
#                   (policy energy-saving): refused -> GuardrailHold ... ues=1
#                   (UE 1 is still on the booster), and no switch-off; amf: one
#                   relocation, then HandoverFailure CellNotAvailable gnb_id=1;
#                   gnb-east: ActuationFailed (lb-1) for the second UE; UE 1
#                   still pings through the booster.
#   off             no DEMO MODE ACTIVE, no [demo: line, no HANDOVER REQUIRED
#                   with ReduceLoadInServingCell, and `demo metrics` empty.
#
# Environment:
#   NEXTGCORE_DIR   nextgcore checkout (default: ../nextgcore beside this repo)
#   DEMO_ARTEFACTS  where logs are captured (default /tmp/demo-artefacts)
#   D446A_TIMEOUT   seconds to wait for the decisive event after the PDU session (300)
#   D446A_KEEP=1    leave the stack running afterwards
#   NR_CLI          an nr-cli binary to copy into gnb-west for `agent-audit --json`
#                   and `demo metrics` (default: binaries/nr-cli, if present)
# Exit status: 0 when every assertion holds, 1 otherwise (DEMO CHECK FAILED: ...).
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/demo/lib.sh
. "$here/lib.sh"

mode="${1:-}"
case "$mode" in
    load-balancing|energy-saving|off|runtime-switch) ;;
    *) echo "usage: $0 load-balancing|energy-saving|off|runtime-switch" >&2; exit 2 ;;
esac

sim="$(cd "$here/../.." && pwd)"
core="$(cd "${NEXTGCORE_DIR:-$sim/../nextgcore}" && pwd)"
demo_artefacts "${DEMO_ARTEFACTS:-/tmp/demo-artefacts}/d446a/$mode"
timeout_s="${D446A_TIMEOUT:-300}"
nr_cli="${NR_CLI:-$sim/binaries/nr-cli}"

# The staged NEXTGSIM_DIR: the compose mounts $NEXTGSIM_DIR/config under the
# file names of the N2 handover stack.
stage="$(mktemp -d -t d446a-XXXXXX)"
mkdir -p "$stage/config"
src="$sim/config/demo-agent-energy"
cp "$src/gnb-a.yaml" "$stage/config/gnb-n2ho-target.yaml"
cp "$src/ue.yaml" "$stage/config/ue-n2ho.yaml"
if [ "$mode" = off ]; then
    sed -E 's/^  enabled: true$/  enabled: false/' "$src/gnb-b.yaml" > "$stage/config/gnb.yaml"
    sed -i -E 's/^  enabled: true$/  enabled: false/' "$stage/config/gnb-n2ho-target.yaml"
elif [ "$mode" = runtime-switch ]; then
    cp "$src/gnb-b.yaml" "$stage/config/gnb.yaml"
else
    sed -E "s/^    mode: load-balancing$/    mode: $mode/" "$src/gnb-b.yaml" > "$stage/config/gnb.yaml"
fi
# The second UE (runtime-switch): its own SUPI and IMEI, and its own TUN prefix
# because it shares the ue container with the first.
sed -E -e 's/^  value: "999700000000001"$/  value: "999700000000002"/' \
    -e 's/^imei: "356938035643803"$/imei: "356938035643811"/' \
    -e 's/^tun_name: "uesimtun"$/tun_name: "uesimtunb"/' \
    "$src/ue.yaml" > "$stage/config/ue-d446a-2.yaml"
for want in '"999700000000002"' '"356938035643811"' '"uesimtunb"'; do
    grep -q "$want" "$stage/config/ue-d446a-2.yaml" || { echo "ue-d446a-2.yaml: $want not staged" >&2; exit 1; }
done
cp -r "$stage/config" "$DEMO_ARTEFACTS/config"
export NEXTGSIM_DIR="$stage"

demo_compose_init "$core/docker/rust/docker-compose.demo-n2ho.yml"
demo_seed "${DEMO_SEED:-}"
demo_assert_private

ue2_log() {
    demo_compose exec -T ue cat /tmp/ue2.log > "$DEMO_ARTEFACTS/ue2.log" 2>/dev/null || true
}

teardown() {
    local rc=$?
    for s in amf smf upf gnb-west gnb-east ue; do demo_logs "$s" "$DEMO_ARTEFACTS/$s.log"; done
    [ "$mode" != runtime-switch ] || ue2_log
    if [ "${D446A_KEEP:-0}" != 1 ]; then demo_compose down -v --remove-orphans >/dev/null 2>&1 || true; fi
    rm -r "$stage" 2>/dev/null || true
    exit "$rc"
}
trap teardown EXIT

# Ask gnb-west's nr-cli, when one is available; output to FILE.
cli_west() {
    local out="$1"; shift
    [ -x "$nr_cli" ] || { echo "(no nr-cli at $nr_cli)" > "$out"; return 0; }
    local id
    id="$(demo_compose ps -q gnb-west)"
    docker cp "$nr_cli" "$id:/tmp/nr-cli" >/dev/null 2>&1 || true
    local node
    node="$(demo_compose exec -T gnb-west /tmp/nr-cli --dump 2>/dev/null | head -1 | tr -d '\r')"
    demo_compose exec -T gnb-west /tmp/nr-cli "$node" -e "$*" > "$out" 2>&1 || true
}

demo_compose up -d
demo_wait_log gnb-west 'Received NG Setup Response' 120
demo_wait_log gnb-east 'Received NG Setup Response' 60
demo_wait_log ue 'Received Registration Accept' 120
demo_wait_log ue 'is now ACTIVE' 60
demo_info "registered on gnb-east (gNB-A) with a PDU session; mode $mode"

case "$mode" in
    load-balancing)
        demo_wait_log gnb-west 'audit #[0-9]+: GuardrailHold' "$timeout_s" 2>/dev/null \
            || demo_info "no GuardrailHold within ${timeout_s}s" ;;
    energy-saving)
        demo_wait_log gnb-west 'off the air: [0-9]+ heartbeat' "$timeout_s" 2>/dev/null \
            || demo_info "no dropped heartbeat within ${timeout_s}s" ;;
    off)
        # Long enough for lb-1 to have fired had the kill switch not held it.
        sleep 45 ;;
    runtime-switch)
        # Contest 1, in load-balancing: UE 1 moves to the booster.
        demo_wait_log gnb-west 'audit #[0-9]+: Resolved \(lb-1@gnb-2\) winner lb-1' "$timeout_s"
        demo_wait_log amf 'UE [0-9]+ relocated to target gNB' 60
        # The switch, at runtime, inside the 90 s the switch-off is deferred, so
        # contest 2 still finds it Pending.
        cli_west "$DEMO_ARTEFACTS/set.txt" demo d446a-agent-conflict set mode=energy-saving
        cat "$DEMO_ARTEFACTS/set.txt"
        demo_wait_log gnb-west 'audit: mode load-balancing -> energy-saving from nr-cli' 30
        # Contest 2: a second UE registers on gNB-A, and its first packets make
        # it a served UE that lb-1 moves.
        demo_compose exec -T -d ue sh -c 'exec nr-ue -c /etc/nextgsim/ue-d446a-2.yaml > /tmp/ue2.log 2>&1'
        for _ in $(seq 120); do
            demo_compose exec -T ue grep -q 'is now ACTIVE' /tmp/ue2.log 2>/dev/null && break
            sleep 1
        done
        ue2_log
        grep -q 'is now ACTIVE' "$DEMO_ARTEFACTS/ue2.log" || demo_fail "[ue2] no PDU session within 120 s"
        demo_info "second UE registered on gnb-east with a PDU session"
        demo_compose exec -T ue ping -I uesimtunb0 -c 5 -i 0.5 -W 2 10.45.0.1 > "$DEMO_ARTEFACTS/ping-ue2.txt" 2>&1 || true
        demo_wait_log gnb-west 'audit #[0-9]+: Resolved \(lb-1@gnb-2\) winner energy-1' 60 2>/dev/null \
            || demo_info "no second contest within 60s"
        demo_wait_log gnb-west 'audit #[0-9]+: GuardrailHold' 30 2>/dev/null \
            || demo_info "no GuardrailHold within 30s" ;;
esac
for s in amf gnb-west gnb-east ue; do demo_logs "$s" "$DEMO_ARTEFACTS/$s.log"; done
cli_west "$DEMO_ARTEFACTS/agent-audit.json" agent-audit --json
cli_west "$DEMO_ARTEFACTS/demo-metrics.prom" demo metrics

amf="$DEMO_ARTEFACTS/amf.log" west="$DEMO_ARTEFACTS/gnb-west.log" east="$DEMO_ARTEFACTS/gnb-east.log"
lb_required='HandoverRequired \(amf_ue_ngap_id=[0-9]+, ran_ue_ngap_id=[0-9]+, type=Intra5gs\) source_gnb_id=2 cause=RadioNetwork\(ReduceLoadInServingCell\)'

ping_via_a() {
    demo_compose exec -T ue ping -I uesimtun0 -c 20 -i 0.5 -W 2 10.45.0.1 > "$DEMO_ARTEFACTS/ping.txt" 2>&1 || true
    cat "$DEMO_ARTEFACTS/ping.txt"
    local received
    received="$(sed -nE 's/.* ([0-9]+) received.*/\1/p' "$DEMO_ARTEFACTS/ping.txt")"
    [ "${received:-0}" -ge "$1" ] || demo_fail "[ue] only ${received:-0}/20 pings crossed the network"
}

case "$mode" in
load-balancing)
    assert_log_order "$amf" amf \
        "$lb_required" \
        'UE [0-9]+ relocated to target gNB' \
        'UE Context Release Command sent to SOURCE association'
    n="$(grep -cE "$lb_required" "$amf" || true)"
    [ "$n" = 1 ] || demo_fail "[amf] $n load-balancing HandoverRequired line(s), expected exactly 1"
    assert_log_order "$west" gnb-west \
        'audit #[0-9]+: IntentSubmitted \(energy-1\)' \
        'Handover Request received \(target gNB\):.*cause=RadioNetwork\(ReduceLoadInServingCell\)' \
        'audit #[0-9]+: InboundIntent \(lb-1@gnb-2\)' \
        'audit #[0-9]+: ConflictDetected \(lb-1@gnb-2\) .*Cell:16' \
        'audit #[0-9]+: Resolved \(lb-1@gnb-2\) winner lb-1' \
        'audit #[0-9]+: Deferred \(energy-1\)' \
        'handback hold: UE\[[0-9]+\] is not handed back to its source cell PCI 32' \
        'audit #[0-9]+: GuardrailHold \(energy-1\) SwitchOffWithConnectedUes: ues=1'
    n="$(grep -cE 'audit #[0-9]+: Executed ' "$west" || true)"
    [ "$n" = 1 ] || demo_fail "[gnb-west] $n Executed audit entries, expected 1"
    if grep -qE 'Sent Handover Required' "$west"; then
        demo_fail "[gnb-west] sent a HANDOVER REQUIRED during the handback hold"
    fi
    if grep -qE 'Agent switch-off' "$west"; then
        demo_fail "[gnb-west] switched off although the guardrail should hold"
    fi
    ping_via_a 18
    ;;
energy-saving)
    assert_log_order "$west" gnb-west \
        'audit #[0-9]+: IntentSubmitted \(energy-1\)' \
        'audit #[0-9]+: InboundIntent \(lb-1@gnb-2\)' \
        'audit #[0-9]+: ConflictDetected \(lb-1@gnb-2\) .*Cell:16' \
        'audit #[0-9]+: Resolved \(lb-1@gnb-2\) winner energy-1' \
        'Agent switch-off: cell 16 off the air' \
        'off the air: [1-9][0-9]* heartbeat\(s\) dropped'
    assert_log_order "$amf" amf \
        "$lb_required" \
        'HandoverFailure from association [0-9]+ \(amf_ue_ngap_id=[0-9]+, cause=RadioNetwork\(CellNotAvailable\)\) gnb_id=1'
    assert_log_order "$east" gnb-east \
        'Sent Handover Required: amf_ue_ngap_id=' \
        'Handover Preparation Failure: .*CellNotAvailable' \
        'audit #[0-9]+: ActuationFailed \(lb-1\)'
    if grep -qE 'UE [0-9]+ relocated to target gNB' "$amf"; then
        demo_fail "[amf] a UE was relocated although the booster refused"
    fi
    ping_via_a 20
    ;;
runtime-switch)
    assert_log_order "$west" gnb-west \
        'audit #[0-9]+: Resolved \(lb-1@gnb-2\) winner lb-1@gnb-2 \(policy load-balancing\)' \
        'handback hold: UE\[[0-9]+\] is not handed back to its source cell PCI 32' \
        'audit: mode load-balancing -> energy-saving from nr-cli' \
        'audit #[0-9]+: InboundIntent \(lb-1@gnb-2\)' \
        'audit #[0-9]+: ConflictDetected \(lb-1@gnb-2\) .*Cell:16' \
        'audit #[0-9]+: PolicyChanged \(d446a-agent-conflict\) from=load-balancing to=energy-saving by=cli' \
        'audit #[0-9]+: Resolved \(lb-1@gnb-2\) winner energy-1 \(policy energy-saving\): refused' \
        'audit #[0-9]+: GuardrailHold \(energy-1\) SwitchOffWithConnectedUes: ues=1'
    n="$(grep -cE 'audit #[0-9]+: Resolved ' "$west" || true)"
    [ "$n" = 2 ] || demo_fail "[gnb-west] $n Resolved audit entries, expected 2 (one per contest)"
    n="$(grep -cE 'audit #[0-9]+: Executed ' "$west" || true)"
    [ "$n" = 1 ] || demo_fail "[gnb-west] $n Executed audit entries, expected 1 (contest 1's admission)"
    if grep -qE 'Agent switch-off' "$west"; then
        demo_fail "[gnb-west] switched off with UE 1 connected"
    fi
    n="$(grep -cE 'UE [0-9]+ relocated to target gNB' "$amf" || true)"
    [ "$n" = 1 ] || demo_fail "[amf] $n relocations, expected 1 (the second contest refused)"
    assert_log_order "$amf" amf \
        "$lb_required" \
        'UE [0-9]+ relocated to target gNB' \
        "$lb_required" \
        'HandoverFailure from association [0-9]+ \(amf_ue_ngap_id=[0-9]+, cause=RadioNetwork\(CellNotAvailable\)\) gnb_id=1'
    assert_log_order "$east" gnb-east \
        'audit #[0-9]+: ActuationSucceeded \(lb-1\)' \
        'Handover Preparation Failure: .*CellNotAvailable' \
        'audit #[0-9]+: ActuationFailed \(lb-1\)'
    # UE 1 is still served by the booster, which stayed on.
    ping_via_a 18
    ;;
off)
    for f in "$west" "$east"; do
        if grep -qE 'DEMO MODE ACTIVE|\[demo:' "$f"; then
            demo_fail "[$(basename "$f" .log)] a demo line with the kill switch off"
        fi
    done
    if grep -qE 'ReduceLoadInServingCell' "$amf"; then
        demo_fail "[amf] a load-balancing handover with the demo off"
    fi
    if grep -q 'nextg_demo_enabled' "$DEMO_ARTEFACTS/demo-metrics.prom"; then
        demo_fail "[gnb-west] demo metrics exported with the kill switch off"
    fi
    ping_via_a 18
    ;;
esac

demo_info "D446A $mode: every assertion holds (logs in $DEMO_ARTEFACTS)"
