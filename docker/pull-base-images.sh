#!/usr/bin/env bash
# Pre-pull every base image the given Dockerfiles build FROM, each pull retried
# with backoff (#448).
#
#   docker/pull-base-images.sh [--list] DOCKERFILE...
#
# WHY. A nightly failed on `failed to resolve source metadata for
# docker.io/library/debian:bookworm-slim@sha256:...: TLS handshake timeout`: one
# Docker Hub blip and the image build had no second chance. A pull is safe to
# repeat, so it is the step that gets the retry; the `docker build` after it
# runs ONCE, and a genuine build error fails it on the first attempt.
#
# Once pulled, a DIGEST-PINNED base resolves from the local image store, so the
# build needs no registry round trip for it (the classic builder and BuildKit's
# default resolve mode both prefer a local image). A base pinned only by tag is
# still re-resolved by the build, so the retry covers it less; this repo pins by
# digest (#420).
#
# What is pulled: the image of every `FROM` line, verbatim (tag and digest
# kept), once each. Skipped: `scratch`, a FROM naming an earlier stage, and a
# FROM that uses a build ARG (it cannot be resolved here; a warning says so).
# `--platform=...` flags are honoured. --list prints the refs without pulling.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
list_only=false
if [ "${1:-}" = --list ]; then list_only=true; shift; fi
[ $# -gt 0 ] || { echo "usage: pull-base-images.sh [--list] DOCKERFILE..." >&2; exit 2; }

# Emits "<ref>\t<platform>" per external base, in file order, deduplicated.
# The ref comes FIRST: tab is IFS whitespace, so `read` would drop an empty
# leading platform field and shift the ref into it.
refs=$(awk '
  FNR == 1 { delete stages }
  toupper($1) == "FROM" {
    platform = ""; i = 2
    while (i <= NF && $i ~ /^--/) {
      if ($i ~ /^--platform=/) { platform = substr($i, 12) }
      i++
    }
    ref = $i
    if (i + 2 <= NF && toupper($(i + 1)) == "AS") { stages[tolower($(i + 2))] = 1 }
    if (ref == "" || tolower(ref) == "scratch" || (tolower(ref) in stages)) { next }
    if (ref ~ /\$/ || platform ~ /\$/) {
      printf "::warning::%s:%d: FROM %s uses a build ARG; not pre-pulled\n", FILENAME, FNR, ref > "/dev/stderr"
      next
    }
    key = ref "\t" platform
    if (!(key in seen)) { seen[key] = 1; print key }
  }
' "$@")

if [ -z "$refs" ]; then
  echo "pull-base-images.sh: no external base image in: $*"
  exit 0
fi

while IFS=$'\t' read -r ref platform; do
  if $list_only; then
    printf '%s%s\n' "${platform:+--platform=${platform} }" "$ref"
    continue
  fi
  echo "pre-pulling ${ref}${platform:+ (${platform})}"
  if [ -n "$platform" ]; then
    "${here}/retry.sh" docker pull --platform "$platform" "$ref"
  else
    "${here}/retry.sh" docker pull "$ref"
  fi
done <<<"$refs"
