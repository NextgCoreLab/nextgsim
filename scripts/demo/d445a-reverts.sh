#!/usr/bin/env bash
# scripts/demo/d445a-reverts.sh: the D445A code revert arms nextgsim can run on
# its own (nextgsim#480).
#
# Each arm applies scripts/demo/reverts/<arm>.patch through revert-run.sh and
# passes only if the named test then FAILS (and the tree is restored after):
#
#   480-t8-ignored     the AF logs a T8 LoC report but does not exclude
#                      -> t8_exclusion_excludes_ue3_and_readmits_it
#   480-samples-trust  the AF weighs an update by its client-supplied samples
#                      -> a_huge_samples_claim_is_capped
#   480-no-provenance  the AF admits a peer the NEF cannot identify
#                      -> a_peer_the_nef_cannot_identify_is_refused
#   480-poison-off     UE 005 never poisons in poisoned-ue
#                      -> the_poisoned_arm_separates_the_rules
#   480-norm-unchecked the AF aggregates an update whatever its norm
#                      -> a_huge_finite_delta_is_refused_and_the_af_survives
#   480-identity-clash the AF lets one GPSI hold two labels, or a label change GPSI
#                      -> an_identity_clash_is_refused_before_subscribing
#   480-subscribe-first the AF subscribes T8 for a clashing identity, then refuses
#                      -> an_identity_clash_is_refused_before_subscribing
#   480-t8-no-evict    the T8 listener never evicts, so a source at its cap is
#                      refused -> idle_connections_do_not_starve_a_notification
#   480-t8-evict-any-source  eviction takes any source's idle connection, even
#                      one still opening
#                      -> a_churning_source_cannot_evict_another_sources_notification
#   480-t8-no-fair-share  at the total cap, the oldest idle connection of any source
#                      is evicted (the round-1 policy)
#                      -> eviction_takes_from_the_largest_source_and_never_the_nefs_one
#
# The other revert arms are knobs, not patches, and have their own tests in
# revert-hooks builds: --no-t8 (without_t8_the_round_waits_for_the_deadline),
# --freeze-global (a_frozen_global_fails_the_scorecard) and fl.leak_raw
# (a_leaky_update_is_refused_and_its_canary_counted). The "nefd at udm" and
# "client on eth0" arms are deployment arms (nextgcore's demo-check.sh).
#
# Docker-free: each arm rebuilds and runs one test. ARMS="a b" runs a subset.
# Exit 0 when every arm fails its named test as it must.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here/../.."
fl="-p nextgsim-fl --features demo-d445a-fl-t8 --lib"
declare -A test_of=(
    [480-t8-ignored]="$fl t8_exclusion_excludes_ue3_and_readmits_it"
    [480-samples-trust]="$fl a_huge_samples_claim_is_capped"
    [480-no-provenance]="$fl a_peer_the_nef_cannot_identify_is_refused"
    [480-poison-off]="$fl the_poisoned_arm_separates_the_rules"
    [480-norm-unchecked]="$fl a_huge_finite_delta_is_refused_and_the_af_survives"
    [480-identity-clash]="$fl an_identity_clash_is_refused_before_subscribing"
    [480-subscribe-first]="$fl an_identity_clash_is_refused_before_subscribing"
    [480-t8-no-evict]="$fl idle_connections_do_not_starve_a_notification"
    [480-t8-evict-any-source]="$fl a_churning_source_cannot_evict_another_sources_notification"
    [480-t8-no-fair-share]="$fl eviction_takes_from_the_largest_source_and_never_the_nefs_one"
)
arms="${ARMS:-480-t8-ignored 480-samples-trust 480-no-provenance 480-poison-off \
480-norm-unchecked 480-identity-clash 480-subscribe-first 480-t8-no-evict \
480-t8-evict-any-source 480-t8-no-fair-share}"
for arm in $arms; do
    args="${test_of[$arm]:?unknown arm $arm}"
    name="${args##* }"
    echo "[demo] revert arm $arm: $name must fail"
    # shellcheck disable=SC2086
    "$here/revert-run.sh" --expect "$name ... FAILED" "$here/reverts/$arm.patch" -- \
        cargo test $args
done
echo "[demo] every D445A revert arm failed its named test"
