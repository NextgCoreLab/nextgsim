#!/usr/bin/env bash
# lte-ue-ping-check: assert a 0%-loss ping through the LTE UE's default bearer
# (nextgsim #186 increment 5; nextgcore #303's EPC stage runs it).
#
# Run INSIDE the lte-ue container (docker exec), after lte-ue has attached:
#   lte-ue-ping-check [TARGET] [COUNT]
# TARGET defaults to $LTE_UE_PING_TARGET, else the PGW's SGi-side gateway the
# test network routes to; COUNT defaults to 20.
#
# Privileges: the lte-ue container must have been started with
#   --cap-add NET_ADMIN --device /dev/net/tun
# (docker-compose: cap_add: [NET_ADMIN], devices: [/dev/net/tun:/dev/net/tun]),
# because lte-ue creates the TUN and assigns the PDN address to it. The ping
# itself needs nothing more: iputils-ping is in the image and runs as root there.
#
# Exit status: 0 on 0% loss; 1 if the TUN never got an IPv4 address within
# LTE_UE_TUN_WAIT_SECS (default 60); 2 on any packet loss.
set -euo pipefail

target="${1:-${LTE_UE_PING_TARGET:-}}"
count="${2:-20}"
tun="${LTE_UE_TUN:-lteuetun5}"
wait_secs="${LTE_UE_TUN_WAIT_SECS:-60}"

if [ -z "$target" ]; then
  echo "lte-ue-ping-check: no target (pass one, or set LTE_UE_PING_TARGET)" >&2
  exit 64
fi

addr=""
for _ in $(seq 1 "$wait_secs"); do
  addr="$(ip -4 -o addr show dev "$tun" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 || true)"
  [ -n "$addr" ] && break
  sleep 1
done
if [ -z "$addr" ]; then
  echo "lte-ue-ping-check: $tun has no IPv4 address after ${wait_secs}s -- the default bearer is not up" >&2
  ip -o link show 2>/dev/null || true
  exit 1
fi
echo "lte-ue-ping-check: $tun is $addr; pinging $target x$count through the PDN"

out="$(ping -I "$tun" -c "$count" -i 0.2 -W 2 "$target" || true)"
echo "$out"
if echo "$out" | grep -qE '(^| )0% packet loss'; then
  echo "lte-ue-ping-check: PASS (0% loss)"
  exit 0
fi
echo "lte-ue-ping-check: FAIL (loss, or no reply)" >&2
exit 2
