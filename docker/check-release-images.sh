#!/usr/bin/env bash
# Fail when an image this repo defines is not in the release publish list
# (nextgcore #420).
#
# docker/release-images.json is the ONE list release.yml builds and pushes to
# Docker Hub. A list kept by hand drifts: a new simulator binary gets a
# Dockerfile and a compose service, and the release silently ships without it.
# This check makes that drift a red CI run instead. Everything below must be
# either an entry in `images` or in `excluded` (with its reason):
#   - every cargo [[bin]] target in the workspace;
#   - every tracked Dockerfile (Dockerfile, Dockerfile.*; not *.dockerignore);
#   - every nextgsim image a compose file or k8s manifest references, by its
#     published name or one of its `local_names`.
# Run from the repo root: docker/check-release-images.sh
set -euo pipefail

manifest="${1:-docker/release-images.json}"
fail=0
err() { echo "::error::$*"; fail=1; }

command -v jq >/dev/null || { echo "::error::jq is required"; exit 1; }
jq -e '.images | type == "array" and length > 0' "$manifest" >/dev/null \
  || { echo "::error::${manifest}: no .images array"; exit 1; }

images=$(jq -r '.images[].image' "$manifest")
binaries=$(jq -r '.images[] | .binary, (.extra_binaries // [])[].binary' "$manifest")
dockerfiles=$(jq -r '.images[].dockerfile' "$manifest" | sort -u)
excluded_paths=$(jq -r '(.excluded // [])[] | select(.path) | .path' "$manifest")
excluded_bins=$(jq -r '(.excluded // [])[] | select(.bin) | .bin' "$manifest")
known_names=$(jq -r '(.images[] | .image, (.local_names // [])[]), ((.excluded // [])[] | (.local_names // [])[])' "$manifest")

in_list() { grep -qxF -- "$1" <<<"$2"; }

# The manifest itself.
dups=$(sort <<<"$images" | uniq -d)
[ -z "$dups" ] || err "${manifest}: duplicate images: ${dups//$'\n'/ }"
while IFS=$'\t' read -r image dockerfile context binary args; do
  [[ "$image" =~ ^[a-z0-9]+(-[a-z0-9]+)*$ ]] || err "${image}: not a valid Docker Hub repository name"
  [ -f "$dockerfile" ] || err "${image}: dockerfile ${dockerfile} does not exist"
  [ -d "$context" ] || err "${image}: context ${context} does not exist"
  [[ " ${args} " == *" --bin ${binary} "* ]] || err "${image}: cargo_args do not build --bin ${binary}"
  grep -qE "^COPY +binaries/${binary}( |$)" "$dockerfile" \
    || err "${image}: ${dockerfile} does not COPY binaries/${binary}"
done < <(jq -r '.images[] | [.image, .dockerfile, .context, .binary, .cargo_args] | @tsv' "$manifest")
# An extra binary lands in binaries/ beside an image's own (no Dockerfile of its own).
while IFS=$'\t' read -r image binary args; do
  [ -n "$image" ] || continue
  [[ " ${args} " == *" --bin ${binary} "* ]] || err "${image}: extra binary ${binary}: cargo_args do not build --bin ${binary}"
done < <(jq -r '.images[] | .image as $i | (.extra_binaries // [])[] | [$i, .binary, .cargo_args] | @tsv' "$manifest")
# No published image may bake a subscriber key (K, OP/OPc). The published
# stage is the manifest's `target`, else the Dockerfile's last stage (and, for
# a target, the lines from its FROM to the next FROM).
while IFS=$'\t' read -r image dockerfile target; do
  [ -f "$dockerfile" ] || continue
  if [ -n "$target" ]; then
    stage=$(awk -v t="$target" 'toupper($1)=="FROM" { on = (tolower($NF) == tolower(t)) } on' "$dockerfile")
    [ -n "$stage" ] || err "${image}: ${dockerfile} has no stage named ${target}"
  else
    stage=$(awk 'toupper($1)=="FROM" { buf="" } { buf = buf $0 "\n" } END { printf "%s", buf }' "$dockerfile")
  fi
  for f in $(grep -E '^COPY ' <<<"$stage" | grep -oE 'config/[^ ]+' || true); do
    if [ -f "$f" ] && grep -qiE '^[[:space:]]*(key|op|opc)[[:space:]]*:' "$f"; then
      err "${image}: the published stage bakes ${f}, which holds a subscriber key (K/OP/OPc); mount it at run time instead"
    fi
  done
done < <(jq -r '.images[] | [.image, .dockerfile, (.target // "")] | @tsv' "$manifest")
while IFS= read -r p; do
  [ -z "$p" ] || [ -e "$p" ] || err "${manifest}: excluded path ${p} does not exist"
done <<<"$excluded_paths"

# Every [[bin]] target.
bins=$(cargo metadata --no-deps --format-version 1 \
  | jq -r '.packages[].targets[] | select(.kind | index("bin")) | .name')
for b in $bins; do
  in_list "$b" "$binaries" || in_list "$b" "$excluded_bins" \
    || err "binary ${b} is not in ${manifest}: add its image to .images (or the bin to .excluded with a reason)"
done
for b in $binaries; do
  in_list "$b" "$bins" || err "${manifest}: binary ${b} is not a [[bin]] target of this workspace"
done

# Every tracked Dockerfile.
while IFS= read -r f; do
  case "$f" in *.dockerignore) continue ;; esac
  in_list "$f" "$dockerfiles" || in_list "$f" "$excluded_paths" \
    || err "${f} is not in ${manifest}: add its image to .images (or the file to .excluded with a reason)"
done < <(git ls-files | grep -E '(^|/)Dockerfile(\.[^/]*)?$' || true)

# Every nextgsim image a compose file or k8s manifest references.
while IFS= read -r ref; do
  [ -n "$ref" ] || continue
  in_list "$ref" "$known_names" \
    || err "image ${ref} is referenced by a compose file or k8s manifest but is not in ${manifest}"
done < <(git ls-files | grep -E '\.ya?ml$' | grep -v '^\.github/' \
  | xargs grep -ohE 'image: *"?nextgsim[-/][a-z0-9-]+' | sed -E 's/^image: *"?//' | sort -u || true)

if [ "$fail" -ne 0 ]; then
  echo "The release publish list is incomplete; see the errors above."
  exit 1
fi
echo "All $(wc -l <<<"$images") images in ${manifest} cover every binary, Dockerfile and referenced image."
