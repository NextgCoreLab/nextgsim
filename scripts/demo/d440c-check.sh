#!/usr/bin/env bash
# scripts/demo/d440c-check.sh: demo d440c-nwdaf-paging (nextgsim#464) on the
# three-gNB compose stack, the same run and the same verdict as the EKS
# scenario (nextgcore deploy/eks/overlays/demo-d440c-nwdaf-paging/).
#
#   NEXTGCORE_DIR=../nextgcore scripts/demo/d440c-check.sh <mode> [variant]
#     mode     nwdaf-mobility | registration-area | off
#     variant  nwdaf-down | revert   (mode nwdaf-mobility only)
#
# Brings up nextgcore's docker/rust/docker-compose.demo-n2ho.yml with
# scripts/demo/d440c.compose.yml (gnb-c at 172.23.0.103, nwdafd, the lap UE, the
# mode in amfd's and the gNBs' configs; no published port: asserted before
# `up`). The UE starts once nwdafd has subscribed to amfd's LOCATION_REPORT, so
# its registration on A is in the history. Then, on the route's clock (the UE's
# `Simulated motion started` line and config/demo-nwdaf-paging/ue-lap.yaml):
# release at the dwell's far point (gnb-west's `nr-cli ue-release`), wait for the
# route's end, ask nwdafd for the UE's UE_MOBILITY prediction, [nwdaf-down: stop
# nwdafd], ping the UE from the UPF, and hand the logs to nextgcore's
# analyze.py, which gives the verdict (its header lists the checks).
#
# Environment:
#   NEXTGCORE_DIR   nextgcore checkout (default ../nextgcore beside this repo)
#   D440C_TAG       tag of the demo images: nextgcore-rust/{amf,nwdaf}, built with
#                   demo-d440c-nwdaf-paging / amf-mobility, and nextgsim-gnb, built
#                   with demo-d440c-nwdaf-paging (default d440c)
#   NR_CLI          nr-cli to copy into gnb-west (default binaries/nr-cli)
#   DEMO_ARTEFACTS  where captures go (default /tmp/demo-artefacts/d440c/<mode>[+<variant>])
#   D440C_KEEP=1    leave the stack running afterwards
# Exit status: 0 when analyze.py passes, 1 otherwise.
# Not yet run: written for compose and checked statically; it needs
# nextgcore#1496's analyze.py.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/demo/lib.sh
. "$here/lib.sh"

mode="${1:-}"; variant="${2:-}"
case "$mode+$variant" in
    nwdaf-mobility+ | nwdaf-mobility+nwdaf-down | nwdaf-mobility+revert | registration-area+ | off+) ;;
    *) demo_fail "usage: $0 <nwdaf-mobility [nwdaf-down|revert] | registration-area | off>"; exit 1 ;;
esac
sim="$(cd "$here/../.." && pwd)"
core="$(cd "${NEXTGCORE_DIR:-$sim/../nextgcore}" && pwd)"
analyze="$core/deploy/eks/overlays/demo-d440c-nwdaf-paging/analyze.py"
[ -f "$analyze" ] || { demo_fail "no $analyze (nextgcore too old?)"; exit 1; }
nr_cli="${NR_CLI:-$sim/binaries/nr-cli}"
[ -x "$nr_cli" ] || { demo_fail "no nr-cli at $nr_cli (set NR_CLI)"; exit 1; }
export NEXTGSIM_DIR="$sim" D440C_MODE="$mode"
D440C_NWDAF_UE_MOBILITY=true
[ "$variant" = revert ] && D440C_NWDAF_UE_MOBILITY=false
export D440C_NWDAF_UE_MOBILITY
art="${DEMO_ARTEFACTS:-/tmp/demo-artefacts}/d440c/$mode${variant:++$variant}"
demo_artefacts "$art"
S="$art/scenario"; mkdir -p "$S"
demo_compose_init "$core/docker/rust/docker-compose.demo-n2ho.yml" "$here/d440c.compose.yml"
demo_assert_private

# Logs as `docker logs --timestamps` writes them: the kubelet format analyze.py reads.
tlogs() { demo_compose logs --no-color --no-log-prefix --timestamps "$1" > "$2" 2>&1 || true; }
capture() {
    tlogs amf "$S/amf.log"; tlogs smf "$S/smf.log"; tlogs ue "$S/ue.log"
    tlogs gnb-west "$S/gnb-a.log"; tlogs gnb-east "$S/gnb-b.log"; tlogs gnb-c "$S/gnb-c.log"
    [ "$variant" = nwdaf-down ] || tlogs nwdaf "$S/nwdaf.log"
    demo_compose exec -T amf curl -s --max-time 5 http://127.0.0.1:9090/metrics > "$S/amf.prom" 2>/dev/null || true
}
teardown() {
    local rc=$?
    capture
    if [ "${D440C_KEEP:-0}" != 1 ]; then demo_compose down -v --remove-orphans >/dev/null 2>&1 || true; fi
    exit "$rc"
}
trap teardown EXIT

demo_compose up -d --scale ue=0
for g in gnb-west gnb-east gnb-c; do demo_wait_log "$g" 'Received NG Setup Response' 120; done
if [ "$variant" = revert ]; then
    demo_wait_log nwdaf 'NWDAF registered with NRF' 120
else
    demo_wait_log nwdaf 'subscribed to AMF LOCATION_REPORT id=' 180
fi
demo_compose up -d ue
demo_wait_log ue 'is now ACTIVE' 120
demo_wait_log ue 'Simulated motion started: sim_route' 150

# The route's clock: the release at waypoint 10 (the dwell's far point), the end.
read -r far end < <(python3 - "$sim/config/demo-nwdaf-paging/ue-lap.yaml" <<'PY'
import math, sys, yaml
u = yaml.safe_load(open(sys.argv[1]))
r = u["sim_route"]; p = u["sim_position"]
pts = [(p["x"], p["y"])] + [(w["x"], w["y"]) for w in r["waypoints"]]
cum = [0.0]
for a, b in zip(pts, pts[1:]):
    cum.append(cum[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
print("%.1f %.1f" % (cum[10] / r["speed_mps"], cum[-1] / r["speed_mps"]))
PY
)
tlogs ue "$S/ue.log"
motion="$(grep -m1 'Simulated motion started: sim_route' "$S/ue.log" | cut -d' ' -f1)"
t0="$(date -u -d "$motion" +%s.%N)"
demo_info "route started $motion; release at +${far} s, end at +${end} s"
python3 -c "import time; time.sleep(max(0.0, $t0 + $far - time.time()))"
west="$(demo_compose ps -q gnb-west)"
docker cp "$nr_cli" "$west:/tmp/nr-cli" >/dev/null
node="$(demo_compose exec -T gnb-west /tmp/nr-cli --dump 2>/dev/null | head -1 | tr -d '\r')"
ue_id="$(demo_compose exec -T gnb-west /tmp/nr-cli "$node" --exec ue-list 2>/dev/null \
    | grep -oE 'ue_id: [0-9]+' | head -1 | grep -oE '[0-9]+' || true)"
[ -n "$ue_id" ] || { demo_fail "gnb-west holds no UE context at the dwell (the UE is not on A)"; exit 1; }
demo_compose exec -T gnb-west /tmp/nr-cli "$node" --exec "ue-release $ue_id" > "$S/ue-release.txt" 2>&1 || true
demo_wait_log ue 'Received RRC Release from cell' 30
python3 -c "import time; time.sleep(max(0.0, $t0 + $end + 3 - time.time()))"

now="$(date -u +%FT%TZ)"; later="$(date -u -d '+30 seconds' +%FT%TZ)"
q="$(python3 -c '
import json, sys, urllib.parse as u
q = lambda v: u.quote(json.dumps(v, separators=(",", ":")), safe="")
print("event-id=UE_MOBILITY&tgt-ue=%s&ana-req=%s" % (q({"supis": [sys.argv[1]]}), q({"startTs": sys.argv[2], "endTs": sys.argv[3]})))' \
    imsi-999700000000001 "$now" "$later")"
demo_compose exec -T nwdaf sh -c \
    "curl -s --http2-prior-knowledge --max-time 10 -o /tmp/body -w '%{http_code}\n' 'http://172.23.0.30:7777/nnwdaf-analyticsinfo/v1/analytics?$q'; cat /tmp/body 2>/dev/null" \
    > "$S/ue-mobility.http" 2>&1 || true
demo_info "UE_MOBILITY GET: $(head -1 "$S/ue-mobility.http")"
if [ "$variant" = nwdaf-down ]; then
    tlogs nwdaf "$S/nwdaf.log"
    demo_compose stop nwdaf >/dev/null
    echo "nwdaf pods before the downlink: $(demo_compose ps -q --status running nwdaf | grep -c . || true)" > "$S/nwdaf-down.txt"
fi

ue_ip="$(sed -nE 's/.*is now ACTIVE \(IPv4: Some\(\[([0-9]+), ([0-9]+), ([0-9]+), ([0-9]+)\]\)\).*/\1.\2.\3.\4/p' "$S/ue.log" | tail -1)"
[ -n "$ue_ip" ] || { demo_fail "[ue] no PDU session address"; exit 1; }
demo_compose exec -T upf ping -c 20 -i 0.5 -W 3 "$ue_ip" > "$S/ping.txt" 2>&1 || true
tail -2 "$S/ping.txt"
sleep 5
capture
# Not `analyze.py | tee` under set -e: a failing pipeline would exit before the
# message below.
if ! python3 "$analyze" "$S" "$mode" "${variant:-none}" > "$S/analyze.txt"; then
    cat "$S/analyze.txt"
    demo_fail "analyze.py: the $mode${variant:++$variant} arm does not hold (see $S/analyze.txt)"
    exit 1
fi
cat "$S/analyze.txt"
