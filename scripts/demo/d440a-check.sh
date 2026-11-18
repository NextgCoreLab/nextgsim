#!/usr/bin/env bash
# scripts/demo/d440a-check.sh: demo d440a-ai-paging (nextgsim#462 PR 3c) live, on
# the six-gNB one-TAC grid, for one committed live seed.
#
#   NEXTGCORE_DIR=../nextgcore scripts/demo/d440a-check.sh <mode> <seed> [revert-amf]
#     mode        off | page-all | last-served | ct | lsq
#     seed        a live seed of config/demo-ai-paging/d440a-scenario.yaml
#     revert-amf  (predictor modes only) amfd's paging.strategy set back to
#                 registration-area: the check must then fail at the amf step
#                 that wants the recommended gNBs paged first
#
# Brings up nextgcore's docker/rust/docker-compose.demo-n2ho.yml with
# scripts/demo/d440a.compose.yml: six gNBs (config/demo-ai-paging/gnb-{1..6}.yaml),
# the mode in every gNB's config, amfd's paging.strategy recommended-first, and
# the seed's UE (ue-seed-<seed>.yaml, whose route is the replay's route for that
# seed). No published port: asserted before `up`.
#
# On the route's clock (from the UE's `Simulated motion started` line, and the
# seed's times in config/demo-ai-paging/d440a-live-seeds.yaml):
#   release_at_s  the harness releases the UE on the gNB that serves it then
#                 (`nr-cli ue-release`; with a predictor mode that gNB asks its
#                 NWDAF for a recommendation and puts it in the UE Context
#                 Release Complete, TS 38.413 §9.3.1.100);
#   page_at_s     the UPF pings the idle UE, so amfd pages it (TS 23.502
#                 §4.2.3.3): attempt 1 to the recommended gNBs within the
#                 registration area, and a retransmission to the whole area.
#
# Asserted, in order (the strings the code emits):
#   ue        is now ACTIVE -> Simulated motion started: sim_route ->
#             RRC connection released -> CM-IDLE -> Paged by cell N in RRC_IDLE
#             -> Sending Service Request (MobileTerminatedServices) -> Service Accept
#   amf       Paging sent to K/M gNB(s) <scope>: ... attempt 1 -> Service Request
#             from CM-IDLE: ... (page answered, T3513 stopped), where <scope> is
#             "recommended by the RAN in the registration area" for a predictor
#             mode and "serving the registration area" (6/6) for off and page-all
#   serving   (predictor modes) [demo:d440a-ai-paging] paging recommendation
#             ue=N predictor=<mode> -> UE Context Release Complete sent ...
#             nodes=[0x<serving>(visited)...]
#   every gNB the banner and a paging grid of 5 cells (all modes but off); no
#             paging recommendation in off or page-all; no demo line in off
#   upf       at least one downlink ping answered by the paged UE
#
# Recorded, not asserted (artefacts result.txt): the serving gNB, the
# recommendation, the gNB the UE answered through, whether attempt 1 hit it, the
# PAGING messages sent, and the replay's line for the same seed and mode.
# Whether a mode hits is the measurement, not a pass condition.
#
# Environment:
#   NEXTGCORE_DIR   nextgcore checkout (default ../nextgcore beside this repo)
#   NR_CLI          nr-cli to copy into the gNBs (default binaries/nr-cli)
#   DEMO_ARTEFACTS  where captures go (default /tmp/demo-artefacts/d440a/<mode>-<seed>[+revert-amf])
#   D440A_KEEP=1    leave the stack running afterwards
# Exit status: 0 when every assertion holds, 1 otherwise (DEMO CHECK FAILED: ...).
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/demo/lib.sh
. "$here/lib.sh"

mode="${1:-}"; seed="${2:-}"; variant="${3:-}"
case "$mode" in
    off | page-all | last-served | ct | lsq) ;;
    *) demo_fail "usage: $0 <off|page-all|last-served|ct|lsq> <seed> [revert-amf]"; exit 1 ;;
esac
case "$variant" in
    '') ;;
    revert-amf)
        case "$mode" in
            last-served | ct | lsq) ;;
            *) demo_fail "revert-amf needs a predictor mode, not $mode"; exit 1 ;;
        esac ;;
    *) demo_fail "unknown variant '$variant' (revert-amf)"; exit 1 ;;
esac
sim="$(cd "$here/../.." && pwd)"
seeds_file="$sim/config/demo-ai-paging/d440a-live-seeds.yaml"
# The seed's clock, and the replay's line for this mode, from the generated file.
if ! read -r release_at page_at replay_line < <(python3 - "$seeds_file" "$seed" "$mode" <<'PY'
import sys, yaml
doc = yaml.safe_load(open(sys.argv[1]))
seed, mode = int(sys.argv[2]), sys.argv[3]
for s in doc["seeds"]:
    if s["seed"] == seed:
        r = s["replay"].get(mode)
        line = "none" if r is None else ",".join("%s=%s" % (k, r[k]) for k in r)
        print(s["release_at_s"], s["page_at_s"], line.replace(" ", ""))
        break
else:
    sys.exit(1)
PY
); then
    demo_fail "seed '$seed' is not a live seed of $seeds_file"
    exit 1
fi
nr_cli="${NR_CLI:-$sim/binaries/nr-cli}"
[ -x "$nr_cli" ] || { demo_fail "no nr-cli at $nr_cli (set NR_CLI)"; exit 1; }

core_dir="${NEXTGCORE_DIR:-$sim/../nextgcore}"
[ -d "$core_dir" ] || { demo_fail "no nextgcore checkout at $core_dir (set NEXTGCORE_DIR)"; exit 1; }
core="$(cd "$core_dir" && pwd)"
export NEXTGSIM_DIR="$sim" D440A_MODE="$mode" D440A_SEED="$seed"
D440A_AMF_STRATEGY=recommended-first
[ "$variant" = revert-amf ] && D440A_AMF_STRATEGY=registration-area
export D440A_AMF_STRATEGY
demo_artefacts "${DEMO_ARTEFACTS:-/tmp/demo-artefacts}/d440a/$mode-$seed${variant:++$variant}"
A="$DEMO_ARTEFACTS"
gnbs=(gnb-west gnb-east gnb-3 gnb-4 gnb-5 gnb-6)
demo_compose_init "$core/docker/rust/docker-compose.demo-n2ho.yml" "$here/d440a.compose.yml"
demo_assert_private

# Timestamped logs (`docker logs --timestamps`): the route's clock is wall time.
tlogs() { demo_compose logs --no-color --no-log-prefix --timestamps "$1" > "$2" 2>&1 || true; }
capture() {
    local s
    for s in amf smf upf ue "${gnbs[@]}"; do tlogs "$s" "$A/$s.log"; done
}
teardown() {
    local rc=$?
    capture
    if [ "${D440A_KEEP:-0}" != 1 ]; then demo_compose down -v --remove-orphans >/dev/null 2>&1 || true; fi
    exit "$rc"
}
trap teardown EXIT
sleep_until() { python3 -c "import time; time.sleep(max(0.0, $1 - time.time()))"; }

demo_info "mode $mode, seed $seed: release at +${release_at} s, page at +${page_at} s of the route"
demo_compose up -d
for g in "${gnbs[@]}"; do demo_wait_log "$g" 'Received NG Setup Response' 180; done
demo_wait_log ue 'is now ACTIVE' 120
demo_wait_log ue 'Simulated motion started: sim_route' 120
tlogs ue "$A/ue.log"
active_line="$(grep -n 'is now ACTIVE' "$A/ue.log" | head -1 | cut -d: -f1)"
motion_line="$(grep -n 'Simulated motion started: sim_route' "$A/ue.log" | head -1 | cut -d: -f1)"
if [ "$motion_line" -lt "$active_line" ]; then
    demo_fail "harness timing: the route started before the PDU session was up (raise START_AFTER_S in nwdaf::d440a::replay::live and regenerate)"
    exit 1
fi
t0="$(date -u -d "$(sed -n "${motion_line}p" "$A/ue.log" | cut -d' ' -f1)" +%s.%N)"

# The release, on the gNB that serves the UE at release_at_s.
for g in "${gnbs[@]}"; do docker cp "$nr_cli" "$(demo_compose ps -q "$g"):/tmp/nr-cli" >/dev/null; done
sleep_until "$t0 + $release_at"
serving=""; ue_id=""
for g in "${gnbs[@]}"; do
    node="$(demo_compose exec -T "$g" /tmp/nr-cli --dump 2>/dev/null | head -1 | tr -d '\r')"
    id="$(demo_compose exec -T "$g" /tmp/nr-cli "$node" --exec ue-list 2>/dev/null \
        | grep -oE 'ue_id: [0-9]+' | head -1 | grep -oE '[0-9]+' || true)"
    if [ -n "$id" ]; then serving="$g"; ue_id="$id"; break; fi
done
[ -n "$serving" ] || { demo_fail "no gNB holds the UE's context at the release (+${release_at} s)"; exit 1; }
demo_compose exec -T "$serving" /tmp/nr-cli "$node" --exec "ue-release $ue_id" > "$A/ue-release.txt" 2>&1 || true
demo_info "released ue_id $ue_id on $serving at +$(python3 -c "import time; print('%.1f' % (time.time() - $t0))") s"
demo_wait_log ue 'RRC connection released' 30

# Downlink data at page_at_s: the SMF's N1N2MessageTransfer makes amfd page.
sleep_until "$t0 + $page_at"
tlogs ue "$A/ue.log"
ue_ip="$(sed -nE 's/.*is now ACTIVE \(IPv4: Some\(\[([0-9]+), ([0-9]+), ([0-9]+), ([0-9]+)\]\)\).*/\1.\2.\3.\4/p' "$A/ue.log" | tail -1)"
[ -n "$ue_ip" ] || { demo_fail "[ue] no PDU session address"; exit 1; }
demo_compose exec -T upf ping -c 40 -i 0.5 -W 3 "$ue_ip" > "$A/ping.txt" 2>&1 || true
tail -2 "$A/ping.txt"
capture

if [ "$mode" = off ] || [ "$mode" = page-all ]; then
    scope='Paging sent to 6/6 gNB\(s\) serving the registration area: .*attempt 1'
else
    scope='Paging sent to [1-6]/[1-6] gNB\(s\) recommended by the RAN in the registration area: .*attempt 1'
fi
assert_log_order "$A/ue.log" ue \
    'is now ACTIVE' \
    'Simulated motion started: sim_route' \
    'RRC connection released' \
    'CM state changed to: CM-IDLE' \
    'Paged by cell [0-9]+ in RRC_IDLE' \
    'Sending Service Request \(MobileTerminatedServices\)' \
    'Service Accept received'
assert_log_order "$A/amf.log" amf \
    "$scope" \
    'Service Request from CM-IDLE: .*\(page answered, T3513 stopped\)'

for g in "${gnbs[@]}"; do
    if [ "$mode" = off ]; then
        ! grep -qE 'DEMO MODE ACTIVE|\[demo:' "$A/$g.log" || demo_fail "[$g] logged demo lines in the off arm"
    else
        assert_log_order "$A/$g.log" "$g" \
            "DEMO MODE ACTIVE: d440a-ai-paging=$mode" \
            '\[demo:d440a-ai-paging\] paging grid: this gNB 0x[0-9a-f]+ at \([-0-9]+, [-0-9]+, [-0-9]+\), 5 neighbour cell\(s\)' >/dev/null
    fi
    if [ "$mode" = off ] || [ "$mode" = page-all ]; then
        ! grep -q 'paging recommendation' "$A/$g.log" || demo_fail "[$g] asked for a recommendation in $mode"
    fi
done
serving_id=""
for i in "${!gnbs[@]}"; do [ "${gnbs[$i]}" = "$serving" ] && serving_id="$(printf '%x' $((i + 1)))"; done
if [ "$mode" != off ] && [ "$mode" != page-all ]; then
    assert_log_order "$A/$serving.log" "$serving" \
        "\\[demo:d440a-ai-paging\\] paging recommendation ue=[0-9]+ predictor=$mode " \
        "UE Context Release Complete sent ue_id=[0-9]+ amf_ue_ngap_id=[0-9]+ nodes=\\[0x${serving_id}\\(visited\\)"
fi

received="$(sed -nE 's/.* ([0-9]+) received.*/\1/p' "$A/ping.txt")"
[ "${received:-0}" -ge 1 ] || demo_fail "[upf] no downlink ping reached the paged UE"

# The measurement: what attempt 1 paged, and through which gNB the UE answered.
python3 - "$A" "$mode" "$seed" "$serving" "$serving_id" "$replay_line" <<'PY' | tee "$A/result.txt"
import re, sys
art, mode, seed, serving, serving_id, replay = sys.argv[1:]
amf = open(f"{art}/amf.log").read().splitlines()
pagings, attempts, answered = 0, 0, None
for i, line in enumerate(amf):
    m = re.search(r"Paging sent to (\d+)/\d+ gNB\(s\) .*attempt (\d+)", line)
    if m:
        pagings += int(m.group(1)); attempts = int(m.group(2))
    if "(page answered, T3513 stopped)" in line:
        for prev in reversed(amf[:i]):
            n = re.search(r"Initial UE Message: .* nci=(0x[0-9a-f]+)", prev)
            if n:
                answered = n.group(1); break
        break
nodes = "none"
try:
    for line in open(f"{art}/{serving}.log"):
        m = re.search(r"UE Context Release Complete sent .* nodes=(\[[^]]*\])", line)
        if m:
            nodes = m.group(1)
except OSError:
    pass
print(f"live mode={mode} seed={seed} serving=0x{serving_id} recommended={nodes} "
      f"answered_nci={answered} first_attempt_hit={str(attempts == 1).lower()} "
      f"pagings={pagings}")
print(f"replay mode={mode} seed={seed} {replay}")
PY
demo_info "D440A $mode seed $seed${variant:+ ($variant)}: the chain holds (logs in $A)"
