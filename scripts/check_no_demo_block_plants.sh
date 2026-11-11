#!/usr/bin/env bash
# Plants one demo block per shape into the REAL tree and expects
# check_no_demo_block.py to fail on each; a trap restores every file, so a
# failed run never leaves the tree planted (nextgsim#461, #458).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
guard=scripts/check_no_demo_block.py
files=(k8s/configmap.yaml config/gnb.yaml config/ue.yaml k8s/kustomization.yaml docker/docker-compose.yml)
trap 'git checkout -- "${files[@]}"' EXIT

expect_fail() { # name, then the guard's extra args
  local name=$1; shift
  if python3 "$guard" "$@" >/tmp/demo-guard.out 2>&1; then
    echo "::error::the guard missed: $name"; cat /tmp/demo-guard.out; exit 1
  fi
  echo "caught: $name"
  git checkout -- "${files[@]}"
}

# 1. A block-style key inside the ConfigMap's embedded gNB config.
sed -i '0,/gnb.yaml: |/s//gnb.yaml: |\n    demo:\n      enabled: true/' k8s/configmap.yaml
expect_fail "block-style demo: in k8s/configmap.yaml"

# 2. A flow-style key.
printf 'extra: {nci: 16, demo: {enabled: true}}\n' >> config/gnb.yaml
expect_fail "flow-style {demo: ...} in config/gnb.yaml"

# 3. One-line JSON.
printf '{"supi":"x","demo":{"enabled":true}}\n' >> config/ue.yaml
expect_fail 'one-line JSON "demo": in config/ue.yaml'

# 4. A shipped profile referencing a demo-only path.
printf 'configMapGenerator:\n  - name: x\n    files:\n      - ../config/demo-x/gnb.yaml\n' >> k8s/kustomization.yaml
expect_fail "k8s/kustomization.yaml naming config/demo-x"
printf '      - ../k8s/overlays/demo-x/gnb.yaml:/etc/x\n' >> docker/docker-compose.yml
expect_fail "docker/docker-compose.yml naming k8s/overlays/demo-x"

# 5. Render pass: a root demo key in a rendered ConfigMap, whatever put it there.
render=$(mktemp)
sed -i '0,/gnb.yaml: |/s//gnb.yaml: |\n    demo:\n      enabled: true/' k8s/configmap.yaml
kubectl kustomize k8s > "$render"
git checkout -- k8s/configmap.yaml
expect_fail "a rendered ConfigMap with a root demo key" --path k8s/overlays --render "$render"
kubectl kustomize k8s > "$render"
python3 "$guard" --path k8s/overlays --render "$render" >/dev/null \
  || { echo "::error::the render pass flags the clean kustomize render"; exit 1; }
echo "clean: kubectl kustomize k8s"
