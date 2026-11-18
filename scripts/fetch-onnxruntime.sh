#!/usr/bin/env bash
# scripts/fetch-onnxruntime.sh: fetch the pinned ONNX Runtime C library (#463, #467).
#
#   lib="$(scripts/fetch-onnxruntime.sh /tmp/ort)"
#
# From the PyPI wheel onnxruntime 1.23.2 (cp311, manylinux_2_27 x86_64), the
# same artefact and pins as nextgcore's D440B/D441D EKS image builds: the
# wheel's sha256 and the sha256 of the libonnxruntime.so.1.23.2 it carries are
# both checked, and a mismatch fails before anything uses the library. Only the
# C library is extracted. It replaced the GitHub release tarball, whose asset
# returned 404 on 2026-11-18 (nextgsim#583).
#
# Prints the library's path on stdout. Needs python3 with pip.
set -euo pipefail
out="${1:?usage: $0 OUTDIR}"
WHEEL_SHA256=da44b99206e77734c5819aa2142c69e64f3b46edc3bd314f6a45a932defc0b3e
LIB_SHA256=718c3fb0785d5ea06061b92c32ce4ad35be1bd4c71ae6192efe06f1ab21f0f8a
LIB=libonnxruntime.so.1.23.2
mkdir -p "${out}/wheel"
python3 -m pip download -q onnxruntime==1.23.2 --no-deps --only-binary=:all: \
    --platform manylinux_2_27_x86_64 --python-version 3.11 -d "${out}/wheel" >&2
wheels=("${out}"/wheel/onnxruntime-1.23.2-*.whl)
if [ "${#wheels[@]}" != 1 ] || [ ! -f "${wheels[0]}" ]; then
    echo "expected one onnxruntime 1.23.2 wheel in ${out}/wheel" >&2; exit 1
fi
echo "${WHEEL_SHA256}  ${wheels[0]}" | sha256sum -c - >&2
python3 -I - "${wheels[0]}" "${out}/${LIB}" <<'PY'
import sys, zipfile
open(sys.argv[2], "wb").write(zipfile.ZipFile(sys.argv[1]).read("onnxruntime/capi/libonnxruntime.so.1.23.2"))
PY
echo "${LIB_SHA256}  ${out}/${LIB}" | sha256sum -c - >&2
echo "${out}/${LIB}"
