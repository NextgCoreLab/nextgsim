#!/usr/bin/env bash
# scripts/demo/nkef-check.sh: demo d447a-nkef-live (nextgsim#488), one seeded
# acceptance run of one mode against real nextgcore binaries.
#
#   NEXTGCORE_DIR=../nextgcore NR_CLI=target/release/nr-cli scripts/demo/nkef-check.sh [on|baseline]
#
# Brings up nextgcore's standalone docker/rust/docker-compose.demo-n2ho.yml with
# config/demo-nkef/compose.override.yml (one gNB with NKEF and the demo block, a
# stationary UE; no published port, asserted on the rendered model) from a fresh
# core (`down -v`). The gNB image must be built with `--features 6g,nkef-sba`;
# nr-cli is copied into the gNB container to ask it.
#
# Mode `on` asserts, in order:
#   - gnb: `NKEF task started at=`, NG Setup, then one internal bundle
#     `domains=ran,n2,sba consistent=true path=ue>cell>gnb>amf>nf-instance
#     cm=CONNECTED stale=false`, with the AMF's ueLocationTimestamp later than the
#     task start;
#   - amf: `Event subscription created` then `Event subscription removed`;
#   - the bundle's AMF nfInstanceId equals the one a separate NRF discovery returns;
#   - external replies: `congestion=Low` at idle, `Medium|High` during a ping
#     burst, and no `imsi-`, `ncgi`, `nrCellId`, `tac`, `nci`, `ngap_id`,
#     `172.23.` or UUID in either;
#   - a runtime `set mode=bogus` is refused and audited, the mode stays `on`;
#   - expiry: BSF has no `expired_at`; after a SIGKILL of bsf (no graceful
#     NFDeregister) the NRF logs `missed heartbeat, marking SUSPENDED` for it and
#     its fact shows `expired_at` within 2 x heartBeatTimer + the 5 s poll;
#   - restart: with the UE stopped and the gNB restarted, the bundle says
#     `stale=true`.
# Mode `baseline` switches the mode with nr-cli before any query and asserts the
# bundle carries no live domain and amfd logs no `Event subscription created`.
#
# Environment:
#   NEXTGCORE_DIR        nextgcore checkout (default ../nextgcore beside this repo)
#   NR_CLI               an nr-cli binary for the gNB container (default target/release/nr-cli)
#   DEMO_ARTEFACTS       where logs go (default /tmp/demo-artefacts/nkef/<mode>)
#   NKEF_KEEP=1          leave the stack running
# Exit status: 0 when every assertion holds, 1 otherwise (DEMO CHECK FAILED: ...).
# `cond && ok || demo_fail ...` throughout: demo_fail is the else arm.
# shellcheck disable=SC2015
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/demo/lib.sh
. "$here/lib.sh"

mode="${1:-on}"
case "$mode" in on | baseline) ;; *) echo "usage: $0 [on|baseline]" >&2; exit 2 ;; esac
sim="$(cd "$here/../.." && pwd)"
core="$(cd "${NEXTGCORE_DIR:-$sim/../nextgcore}" && pwd)"
export NEXTGSIM_DIR="$sim"
demo_artefacts "${DEMO_ARTEFACTS:-/tmp/demo-artefacts}/nkef/$mode"
nr_cli="${NR_CLI:-$sim/target/release/nr-cli}"
[ -x "$nr_cli" ] || { demo_fail "no nr-cli at $nr_cli (set NR_CLI)"; exit 1; }
key=d447a-nkef-live
question='is my route congested and which core functions serve me?'

demo_compose_init "$core/docker/rust/docker-compose.demo-n2ho.yml"
DEMO_COMPOSE+=(-f "$sim/config/demo-nkef/compose.override.yml")
demo_seed "${DEMO_SEED:-}"
demo_assert_private

teardown() {
    local rc=$?
    for s in amf nrf gnb-west ue; do demo_logs "$s" "$DEMO_ARTEFACTS/$s.log"; done
    if [ "${NKEF_KEEP:-0}" != 1 ]; then demo_compose down -v --remove-orphans >/dev/null 2>&1 || true; fi
    exit "$rc"
}
trap teardown EXIT

gnb_cli() { # <command...>: run one nr-cli command on the gNB, output on stdout
    local node
    node="$(demo_compose exec -T gnb-west /tmp/nr-cli --dump | grep -m1 -i gnb || true)"
    [ -n "$node" ] || { demo_fail "nr-cli --dump lists no gNB"; return 1; }
    demo_compose exec -T gnb-west /tmp/nr-cli "$node" -e "$*"
}
install_cli() {
    docker cp "$nr_cli" "$(demo_compose ps -q gnb-west)":/tmp/nr-cli
}
ask() { # <file> <args...>: one nkef-query, saved to <file>
    local out="$DEMO_ARTEFACTS/$1"
    shift
    gnb_cli nkef-query "$@" > "$out" 2>&1 || { cat "$out" >&2; demo_fail "nkef-query $* failed"; return 1; }
    cat "$out"
}
leak_check() { # <file>
    if grep -nE 'imsi-|ncgi|nrCellId|tac|nci|ngap_id|172\.23\.|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' "$1"; then
        demo_fail "[external] $1 leaks an identity or location (lines above)"
        return 1
    fi
}

# A fresh core: no UE context or NF profile survives from an earlier run.
demo_compose down -v --remove-orphans >/dev/null 2>&1 || true
demo_compose up -d
demo_wait_log gnb-west 'NKEF task started at=' 120
demo_wait_log gnb-west 'Received NG Setup Response' 120
demo_wait_log ue 'is now ACTIVE' 180
install_cli
demo_logs gnb-west "$DEMO_ARTEFACTS/gnb-west.log"
grep -E 'WARN.*DEMO MODE ACTIVE: d447a-nkef-live=on' "$DEMO_ARTEFACTS/gnb-west.log" \
    || { demo_fail "[gnb] no DEMO MODE ACTIVE banner"; exit 1; }
gnb_cli demo metrics > "$DEMO_ARTEFACTS/metrics.txt"
grep -F "nextg_demo_enabled{demo=\"$key\",mode=\"on\"} 1" "$DEMO_ARTEFACTS/metrics.txt" \
    || { demo_fail "[gnb] demo metrics: $(cat "$DEMO_ARTEFACTS/metrics.txt")"; exit 1; }

if [ "$mode" = baseline ]; then
    gnb_cli demo "$key" set mode=baseline | tee "$DEMO_ARTEFACTS/set-baseline.txt"
    sleep 7 # one NRF poll and several RLS windows, which baseline must ignore
    ask internal.txt "$question"
    grep -qE '^NKEF bundle consumer=internal domains=none ' "$DEMO_ARTEFACTS/internal.txt" \
        || demo_fail "[baseline] the bundle carries live facts"
    demo_logs amf "$DEMO_ARTEFACTS/amf.log"
    if grep -n 'Event subscription created' "$DEMO_ARTEFACTS/amf.log"; then
        demo_fail "[baseline] NKEF subscribed at amfd"
    fi
    demo_info "baseline: no live domain, no amfd subscription (logs in $DEMO_ARTEFACTS)"
    exit 0
fi

sleep 7 # at least one NRF poll and RLS window
started="$(grep -oE 'NKEF task started at=[0-9TZ:-]+' "$DEMO_ARTEFACTS/gnb-west.log" | head -1 | cut -d= -f2)"
ask internal.txt "$question"
header="$(head -1 "$DEMO_ARTEFACTS/internal.txt")"
printf '%s\n' "$header" | grep -qE '^NKEF bundle consumer=internal domains=ran,n2,sba consistent=true path=ue>cell>gnb>amf>nf-instance cm=CONNECTED stale=false ' \
    || demo_fail "[internal] header: $header"
loc_ts="$(grep -oE 'ueLocationTimestamp=[0-9TZ:-]+' "$DEMO_ARTEFACTS/internal.txt" | head -1 | cut -d= -f2)"
[ -n "$loc_ts" ] && [[ "$loc_ts" > "$started" || "$loc_ts" == "$started" ]] \
    || demo_fail "[internal] ueLocationTimestamp '$loc_ts' is not after NKEF task started at=$started"
demo_info "[internal] ueLocationTimestamp=$loc_ts >= started at=$started"

demo_logs amf "$DEMO_ARTEFACTS/amf.log"
assert_log_order "$DEMO_ARTEFACTS/amf.log" amf \
    'Event subscription created: id=[^,]+, events=.*LOCATION_REPORT' \
    'Event subscription removed: '
demo_logs gnb-west "$DEMO_ARTEFACTS/gnb-west.log"
assert_log_order "$DEMO_ARTEFACTS/gnb-west.log" gnb-west \
    'NKEF task started at=' \
    'Received NG Setup Response' \
    '\[demo:d447a-nkef-live\] NKEF: amfd subscribe status=201 reports=2' \
    '\[demo:d447a-nkef-live\] NKEF: amfd unsubscribe status=204' \
    '\[demo:d447a-nkef-live\] NKEF bundle consumer=internal domains=ran,n2,sba consistent=true'

# The AMF instance in the bundle is the one the NRF itself names.
bundle_nf="$(printf '%s\n' "$header" | grep -oE 'amf_nf_instance=[0-9a-f-]+' | cut -d= -f2)"
net="$(docker inspect -f '{{range $k, $v := .NetworkSettings.Networks}}{{$k}} {{end}}' "$(demo_compose ps -q nrf)" | awk '{print $1}')"
docker run --rm --network "$net" curlimages/curl:8.11.1 -s --http2-prior-knowledge \
    'http://172.23.0.10:7777/nnrf-disc/v1/nf-instances?target-nf-type=AMF&requester-nf-type=NEF' \
    > "$DEMO_ARTEFACTS/nrf-amf.json"
nrf_nf="$(jq -r '.nfInstances[0].nfInstanceId' "$DEMO_ARTEFACTS/nrf-amf.json")"
[ -n "$bundle_nf" ] && [ "$bundle_nf" = "$nrf_nf" ] \
    || demo_fail "[nrf] bundle AMF nfInstanceId '$bundle_nf' != NRF '$nrf_nf'"
demo_info "[nrf] AMF nfInstanceId $nrf_nf in the bundle and at the NRF"

# External: congestion only, Low at idle, higher under a ping burst.
ask external-idle.txt --as external "$question"
grep -qE '^area-1 congestion=Low ' "$DEMO_ARTEFACTS/external-idle.txt" || demo_fail "[external] idle is not Low"
leak_check "$DEMO_ARTEFACTS/external-idle.txt"
demo_compose exec -T ue ping -I uesimtun0 -i 0.01 -s 1200 -c 800 -q 10.45.0.1 > "$DEMO_ARTEFACTS/ping-burst.txt" 2>&1 &
burst=$!
sleep 3
ask external-burst.txt --as external "$question"
wait "$burst" || true
grep -qE '^area-1 congestion=(Medium|High) ' "$DEMO_ARTEFACTS/external-burst.txt" \
    || demo_fail "[external] the ping burst did not raise the level"
leak_check "$DEMO_ARTEFACTS/external-burst.txt"

# Kill-switch semantics at runtime: a bogus mode is refused, audited, unchanged.
if gnb_cli demo "$key" set mode=bogus > "$DEMO_ARTEFACTS/set-bogus.txt" 2>&1; then
    demo_fail "[demo] set mode=bogus was accepted"
fi
gnb_cli demo "$key" show > "$DEMO_ARTEFACTS/show.txt"
grep -q 'mode=on\|mode: on\|"on"\| on' "$DEMO_ARTEFACTS/show.txt" || demo_fail "[demo] mode changed: $(cat "$DEMO_ARTEFACTS/show.txt")"
demo_logs gnb-west "$DEMO_ARTEFACTS/gnb-west.log"
grep -E '\[demo:d447a-nkef-live\] audit:.*bogus' "$DEMO_ARTEFACTS/gnb-west.log" || demo_fail "[demo] the refused set was not audited"

# Expiry: the BSF before and after it stops heartbeating. SIGKILL, not `stop`:
# a graceful bsfd shutdown NFDeregisters (bsfd lib.rs), which would prove
# deregistration instead of the NRF's no-heartbeat suspension.
ask before-stop.txt bsf instance
bsf_line="$(grep -E 'nfType=BSF' "$DEMO_ARTEFACTS/before-stop.txt" | head -1)"
[ -n "$bsf_line" ] || demo_fail "[expiry] no BSF fact before the stop"
printf '%s\n' "$bsf_line" | grep -q 'expired_at' && demo_fail "[expiry] BSF expired before the stop"
bsf_id="$(printf '%s\n' "$bsf_line" | grep -oE 'nfInstanceId=[0-9a-f-]+' | cut -d= -f2)"
docker run --rm --network "$net" curlimages/curl:8.11.1 -s --http2-prior-knowledge \
    "http://172.23.0.10:7777/nnrf-nfm/v1/nf-instances/$bsf_id" > "$DEMO_ARTEFACTS/nrf-bsf.json"
hb="$(jq -r '.heartBeatTimer // empty' "$DEMO_ARTEFACTS/nrf-bsf.json")"
[ -n "$hb" ] || demo_fail "[expiry] the NRF profile of $bsf_id has no heartBeatTimer"
poll=5
bound=$((2 * hb + poll))
t0="$(date +%s)"
docker kill --signal KILL "$(demo_compose ps -q bsf)" >/dev/null
demo_wait_log nrf "NF instance $bsf_id missed heartbeat, marking SUSPENDED" $((bound + 30))
demo_wait_log gnb-west "\\[demo:d447a-nkef-live\\] NKEF fact expired subject=nf-$bsf_id " $((bound + 30))
t1="$(date +%s)"
ask after-stop.txt bsf instance
grep -E "nfInstanceId=$bsf_id" "$DEMO_ARTEFACTS/after-stop.txt" | grep -qE 'expired_at=[0-9]{4}-' \
    || demo_fail "[expiry] the BSF fact has no expired_at after the kill"
printf 'elapsed=%s heartBeatTimer=%s poll=%s bound=%s\n' "$((t1 - t0))" "$hb" "$poll" "$bound" > "$DEMO_ARTEFACTS/expiry.txt"
[ "$((t1 - t0))" -le "$bound" ] || demo_fail "[expiry] $((t1 - t0))s > 2 x heartBeatTimer ($hb) + poll ($poll)"
demo_info "[expiry] BSF $bsf_id suspended by the NRF and expired $((t1 - t0))s after the kill (bound ${bound}s)"

# Restart without the UE: the AMF's location predates this run.
demo_compose stop ue
demo_compose restart gnb-west
for _ in $(seq 30); do
    demo_logs gnb-west "$DEMO_ARTEFACTS/gnb-west-restart.log"
    [ "$(grep -c 'NKEF task started at=' "$DEMO_ARTEFACTS/gnb-west-restart.log")" -ge 2 ] && break
    sleep 2
done
demo_wait_log gnb-west 'Received NG Setup Response' 60
demo_logs gnb-west "$DEMO_ARTEFACTS/gnb-west-restart.log"
install_cli
sleep 5
ask restart.txt "$question"
grep -qE ' stale=true ' "$DEMO_ARTEFACTS/restart.txt" || demo_fail "[restart] the bundle is not stale=true: $(head -1 "$DEMO_ARTEFACTS/restart.txt")"

demo_info "d447a-nkef-live on: every assertion holds (logs in $DEMO_ARTEFACTS)"
