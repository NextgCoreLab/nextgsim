#!/usr/bin/env bash
# scripts/demo/d440a-m0-check.sh: the D440A M0 gate (nextgsim#462). Can the base
# stack do what predicted paging is measured on?
#   1. idle-mode cell reselection (TS 38.304 §5.2.4) from gnb-west to gnb-east,
#      by a UE that moves while CM-IDLE (config/ue-d440a-m0.yaml, a sim_route);
#   2. a mobile-terminated Service Request (TS 23.502 §4.2.3.3) through gnb-east,
#      the gNB the UE now camps on, answering the PAGING for downlink data.
#
#   NEXTGCORE_DIR=../nextgcore scripts/demo/d440a-m0-check.sh [base|page-all]
#
# Brings up nextgcore's docker/rust/docker-compose.demo-n2ho.yml (two gNBs, one
# TAC, no published port: asserted before `up`) with this checkout's config/
# mounted, plus scripts/demo/d440a-m0.compose.yml (the UE runs ue-d440a-m0.yaml).
#
# Arms:
#   base      (default) no demo block: the base stack alone. This is the gate.
#   page-all  both gNBs run their config/demo-ai-paging/ configs: the D440A demo
#             on in mode page-all, with a paging grid naming the other gNB. Needs
#             a gNB built with --features demo-d440a-ai-paging. Asserts the
#             banner and the grid on both gNBs, then the same chain as base: the
#             predictor-off arm pages exactly like no demo.
#
# Asserted, in order (strings are the ones the code emits):
#   amf   the UE's Initial UE Message from gnb-west (nci=0x10)
#   ue    Cell selection complete -> RRC connection released -> CM-IDLE
#         -> Simulated motion started (the route leaves sim_position only now)
#         -> Cell reselection to gnb-east's cell -> Paged by that cell in RRC_IDLE
#         -> Sending Service Request (MobileTerminatedServices) -> Service Accept
#   amf   Paging sent to 2/2 gNB(s) ... attempt 1 -> Initial UE Message ... nci=0x20
#         -> Service Request from CM-IDLE ... (page answered, T3513 stopped)
#   gnb-east  PAGING from AMF -> RRC Setup -> Sending Initial UE Message
# and that the downlink pings that triggered the page are answered.
#
# Environment:
#   NEXTGCORE_DIR   nextgcore checkout (default: ../nextgcore beside this repo)
#   NR_CLI          nr-cli to copy into gnb-west for `ue-release` (default:
#                   binaries/nr-cli). Required: the UE is released to CM-IDLE by
#                   the gNB, as in docker/rust/ue-flow-check.sh.
#   DEMO_ARTEFACTS  where logs are captured (default /tmp/demo-artefacts/d440a-m0/<arm>)
#   M0_TIMEOUT      seconds to wait for the idle reselection once the route has
#                   started (default 90)
#
# The route clock runs from UE start-up (start_after_secs in ue-d440a-m0.yaml), and
# the release is the harness's. The order is checked, not assumed: a UE that starts
# moving while still CM-CONNECTED (a slow runner) would be handed over across N2,
# and the check then fails naming that, not as an M0 failure.
#   M0_KEEP=1       leave the stack running afterwards
# Exit status: 0 when every assertion holds, 1 otherwise (DEMO CHECK FAILED: ...).
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/demo/lib.sh
. "$here/lib.sh"

arm="${1:-base}"
case "$arm" in
    base | page-all) ;;
    *) demo_fail "unknown arm '$arm' (base | page-all)"; exit 1 ;;
esac

sim="$(cd "$here/../.." && pwd)"
core="$(cd "${NEXTGCORE_DIR:-$sim/../nextgcore}" && pwd)"
export NEXTGSIM_DIR="$sim"
demo_artefacts "${DEMO_ARTEFACTS:-/tmp/demo-artefacts}/d440a-m0/$arm"
timeout_s="${M0_TIMEOUT:-90}"
nr_cli="${NR_CLI:-$sim/binaries/nr-cli}"
[ -x "$nr_cli" ] || { demo_fail "no nr-cli at $nr_cli (set NR_CLI)"; exit 1; }

overrides=("$here/d440a-m0.compose.yml")
[ "$arm" = page-all ] && overrides+=("$here/d440a-page-all.compose.yml")
demo_compose_init "$core/docker/rust/docker-compose.demo-n2ho.yml" "${overrides[@]}"
demo_assert_private

teardown() {
    local rc=$?
    for s in amf smf upf gnb-west gnb-east ue; do demo_logs "$s" "$DEMO_ARTEFACTS/$s.log"; done
    if [ "${M0_KEEP:-0}" != 1 ]; then demo_compose down -v --remove-orphans >/dev/null 2>&1 || true; fi
    exit "$rc"
}
trap teardown EXIT

demo_compose up -d
demo_wait_log gnb-west 'Received NG Setup Response' 120
demo_wait_log gnb-east 'Received NG Setup Response' 60
demo_wait_log ue 'Received Registration Accept' 120
demo_wait_log ue 'is now ACTIVE' 60
demo_info "registered with a PDU session; releasing the UE to CM-IDLE before its route starts"

# The release: gnb-west's `ue-release` (a gNB-initiated UE Context Release).
west="$(demo_compose ps -q gnb-west)"
docker cp "$nr_cli" "$west:/tmp/nr-cli" >/dev/null
node="$(demo_compose exec -T gnb-west /tmp/nr-cli --dump 2>/dev/null | head -1 | tr -d '\r')"
if ! ue_id="$(demo_compose exec -T gnb-west /tmp/nr-cli "$node" --exec ue-list 2>/dev/null \
    | grep -oE 'ue_id: [0-9]+' | head -1 | grep -oE '[0-9]+')"; then
    demo_logs amf "$DEMO_ARTEFACTS/amf.log"
    if grep -q 'HandoverRequired' "$DEMO_ARTEFACTS/amf.log"; then
        demo_fail "harness timing: the UE was handed over before the release (its route started while CM-CONNECTED; raise start_after_secs in config/ue-d440a-m0.yaml)"
    else
        demo_fail "[gnb-west] holds no UE context to release"
    fi
    exit 1
fi
demo_compose exec -T gnb-west /tmp/nr-cli "$node" --exec "ue-release $ue_id" \
    > "$DEMO_ARTEFACTS/ue-release.txt" 2>&1 || true
# The UE also logs CM-IDLE at start-up, before it registers: the release's
# CM-IDLE is the first one after `RRC connection released`.
demo_wait_log ue 'RRC connection released' 30
demo_wait_log ue 'CM state changed to: CM-IDLE' 30

# The release must precede the route: a UE that started moving while connected
# says so in its log before the release's CM-IDLE line.
demo_logs ue "$DEMO_ARTEFACTS/ue.log"
released_line="$(grep -n 'RRC connection released' "$DEMO_ARTEFACTS/ue.log" | head -1 | cut -d: -f1)"
idle_line="$(grep -n 'CM state changed to: CM-IDLE' "$DEMO_ARTEFACTS/ue.log" | cut -d: -f1 \
    | awk -v r="$released_line" '$1 > r { print; exit }')"
[ -n "$idle_line" ] || { demo_fail "[ue] no CM-IDLE after the release"; exit 1; }
motion_line="$(grep -n 'Simulated motion started' "$DEMO_ARTEFACTS/ue.log" | head -1 | cut -d: -f1 || true)"
if [ -n "$motion_line" ] && [ "$motion_line" -lt "$idle_line" ]; then
    demo_fail "harness timing: released after the route started (UE log line $motion_line moves, line $idle_line idles; raise start_after_secs in config/ue-d440a-m0.yaml)"
    exit 1
fi
demo_info "[ue] CM-IDLE before its route starts"
route_s="$(sed -nE 's/^ *start_after_secs: *([0-9.]+).*/\1/p' "$sim/config/ue-d440a-m0.yaml" | head -1)"
demo_wait_log ue 'Simulated motion started' "$(( ${route_s%.*} + 30 ))" 2>/dev/null \
    || demo_info "the UE's route never started"

# Which RLS cell id is gnb-east's: the cell the UE did NOT select at registration.
demo_logs ue "$DEMO_ARTEFACTS/ue.log"
west_cell="$(sed -nE 's/.*Cell selection complete: cell_id=([0-9]+),.*/\1/p' "$DEMO_ARTEFACTS/ue.log" | head -1)"
[ -n "$west_cell" ] || { demo_fail "[ue] no cell selection before the release"; exit 1; }
demo_info "[ue] camped on cell $west_cell (gnb-west); waiting up to ${timeout_s}s for the idle reselection"
# A reselection to any cell but west's (the UE re-selects west itself right after
# the release). On a timeout fall through: the ordered assertion names the gap.
east_cell=""
deadline=$((SECONDS + timeout_s))
while [ -z "$east_cell" ] && [ "$SECONDS" -lt "$deadline" ]; do
    demo_logs ue "$DEMO_ARTEFACTS/ue.log"
    east_cell="$(sed -nE 's/.*Cell reselection: id=([0-9]+),.*/\1/p' "$DEMO_ARTEFACTS/ue.log" \
        | grep -vx "$west_cell" | head -1 || true)"
    [ -n "$east_cell" ] || sleep 2
done
if [ -n "$east_cell" ]; then
    demo_info "[ue] reselected to cell $east_cell (gnb-east) while CM-IDLE"
else
    demo_info "no idle reselection within ${timeout_s}s"
fi

# Downlink data from the UPF to the idle UE: the SMF's N1N2MessageTransfer makes
# the AMF page it, and the UE must answer through the cell it camps on.
ue_ip="$(sed -nE 's/.*is now ACTIVE \(IPv4: Some\(\[([0-9]+), ([0-9]+), ([0-9]+), ([0-9]+)\]\)\).*/\1.\2.\3.\4/p' \
    "$DEMO_ARTEFACTS/ue.log" | head -1)"
[ -n "$ue_ip" ] || { demo_fail "[ue] no PDU session address in the log"; exit 1; }
demo_compose exec -T upf ping -c 20 -i 0.5 -W 3 "$ue_ip" > "$DEMO_ARTEFACTS/ping.txt" 2>&1 || true
cat "$DEMO_ARTEFACTS/ping.txt"
for s in amf gnb-west gnb-east ue; do demo_logs "$s" "$DEMO_ARTEFACTS/$s.log"; done

assert_log_order "$DEMO_ARTEFACTS/ue.log" ue \
    'Cell selection complete: cell_id=[0-9]+' \
    'RRC connection released' \
    'CM state changed to: CM-IDLE' \
    'Simulated motion started: sim_route from \(35, 0, 0\)' \
    "Cell reselection: id=${east_cell:-NONE}," \
    "Paged by cell ${east_cell:-NONE} in RRC_IDLE" \
    'Sending Service Request \(MobileTerminatedServices\)' \
    "Received RRC Setup from cell ${east_cell:-NONE} " \
    'Service Accept received'
assert_log_order "$DEMO_ARTEFACTS/amf.log" amf \
    'Initial UE Message: ran_ue_ngap_id=[0-9]+, nas_pdu_len=[0-9]+, tac=1, nci=0x10' \
    'Paging sent to 2/2 gNB\(s\) serving the registration area: .*attempt 1' \
    'Initial UE Message: ran_ue_ngap_id=[0-9]+, nas_pdu_len=[0-9]+, tac=1, nci=0x20' \
    'Service Request from CM-IDLE: .*\(page answered, T3513 stopped\)'
assert_log_order "$DEMO_ARTEFACTS/gnb-east.log" gnb-east \
    'PAGING from AMF\[[0-9]+\]: matched 1 served TAI' \
    'RRC Setup for UE\[[0-9]+\]' \
    'Sending Initial UE Message'

received="$(sed -nE 's/.* ([0-9]+) received.*/\1/p' "$DEMO_ARTEFACTS/ping.txt")"
[ "${received:-0}" -ge 19 ] \
    || demo_fail "[upf] only ${received:-0}/20 downlink pings reached the paged UE"

if [ "$arm" = page-all ]; then
    for g in gnb-west gnb-east; do
        assert_log_order "$DEMO_ARTEFACTS/$g.log" "$g" \
            'DEMO MODE ACTIVE: d440a-ai-paging=page-all' \
            '\[demo:d440a-ai-paging\] paging grid: this gNB 0x[0-9a-f]+ at \([-0-9]+, [-0-9]+, [-0-9]+\), 1 neighbour cell\(s\)'
    done
else
    for g in gnb-west gnb-east; do
        ! grep -qE 'DEMO MODE ACTIVE|\[demo:' "$DEMO_ARTEFACTS/$g.log" \
            || demo_fail "[$g] logged demo lines in the base arm"
    done
    demo_info "[gnb-west, gnb-east] no demo line in the base arm"
fi

demo_info "D440A M0 ($arm): idle reselection gnb-west -> gnb-east and a mobile-terminated \
Service Request through gnb-east (logs in $DEMO_ARTEFACTS)"
