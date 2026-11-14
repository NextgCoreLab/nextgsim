#!/usr/bin/env bash
# scripts/demo/d444c-reverts.sh: the D444C revert arms (nextgsim#479). The demo
# is crate-level, so every arm is a nextgsim-semantic unit test.
#
# Each arm applies scripts/demo/reverts/<arm>.patch through revert-run.sh and
# passes only if the named test then FAILS (and the tree is restored after):
#
#   479-thread-rng    AwgnChannel::transmit_with_rng ignores its seeded RNG and
#                     draws from thread_rng()
#                     -> two_runs_with_the_same_seed_print_byte_identical_tables
#   479-top-k-scorer  the sweep scores with metrics::top_k_accuracy (feature
#                     indices) instead of NearestCentroid (the label)
#                     -> shuffled_labels_score_at_chance
#
# Docker-free: each arm rebuilds and runs one test. ARMS="a b" runs a subset.
# Exit 0 when every arm fails its named test as it must.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here/../.."
declare -A test_of=(
    [479-thread-rng]="-p nextgsim-semantic --features demo-d444c-task-vs-raw --lib two_runs_with_the_same_seed_print_byte_identical_tables"
    [479-top-k-scorer]="-p nextgsim-semantic --features demo-d444c-task-vs-raw --lib shuffled_labels_score_at_chance"
)
arms="${ARMS:-479-thread-rng 479-top-k-scorer}"
for arm in $arms; do
    args="${test_of[$arm]:?unknown arm $arm}"
    name="${args##* }"
    echo "[demo] revert arm $arm: $name must fail"
    # shellcheck disable=SC2086
    "$here/revert-run.sh" --expect "$name ... FAILED" "$here/reverts/$arm.patch" -- \
        cargo test $args
done
echo "[demo] every D444C revert arm failed its named test"
