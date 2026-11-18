#!/usr/bin/env bash
# scripts/demo/d441d-reverts.sh: the D441D revert arm, docker-free (nextgsim#467).
#
# scripts/demo/reverts/d441d-tick-off.patch patches the NWDAF tick out (the
# ticket's revert arm). Through revert-run.sh it must make the named unit test
# FAIL, and the tree is restored after:
#
#   d441d-tick-off  -> a_predictive_tick_recommends_the_neighbour_to_rrc
#
# The same patch is the LIVE revert arm of ci.yml's demo-d441d job: the gNB
# rebuilt under it, `d441d-check.sh lsq` on one seed must fail at its first
# link (`chain=prediction`). The A3 tests stay green under it: they do not
# need the tick (mode a3 and the demo off take the #162 path).
#
# Exit 0 when the arm fails its named test as it must.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here/../.."

name=a_predictive_tick_recommends_the_neighbour_to_rrc
echo "[demo] revert arm d441d-tick-off: $name must fail"
"$here/revert-run.sh" --expect "$name ... FAILED" "$here/reverts/d441d-tick-off.patch" -- \
    cargo test -p nextgsim-gnb --features demo-d441d-predictive-ho --lib "$name"
echo "[demo] the D441D revert arm failed its named test"
