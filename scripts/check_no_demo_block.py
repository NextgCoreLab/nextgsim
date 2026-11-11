#!/usr/bin/env python3
"""Fail on a `demo:` block in anything nextgsim ships (nextgsim#458, #461).

nextgcore runs the same check over its own tree (nextgcore#1394,
deploy/check_no_demo_block.py); the rules are identical and only the paths
differ. The demo toggle convention keeps demos out of production profiles: the
shipped k8s manifests, the docker compose files and the configs they mount, and
the config/ samples carry no `demo:` block. A demo run uses its own overlay,
which is the only place one may appear:

  * k8s/overlays/demo*/
  * docker/docker-compose.demo*.yml
  * config/demo*/

The scan is textual, line by line, over every git-tracked file under the
scanned paths (default: k8s docker config), so it also catches a `demo:` key
inside a ConfigMap's embedded gNB or UE config, or a JSON `"demo":` member.
Comment lines and prose (.md, .txt) are ignored.

Two more rules close the gaps a text scan alone leaves:

  * reference: a file outside the demo paths that NAMES one (overlays/demo,
    configs/demo, docker-compose.demo) fails. A shipped profile could otherwise
    pull a demo file in through a configMapGenerator, a kustomize resource or a
    compose mount. Only the demo profile and runner may name them.
  * render (--render FILE): every ConfigMap data value in a rendered stream
    (kubectl kustomize / helm template) that is a config with a root `demo` key
    fails, however it got there.

Usage: check_no_demo_block.py [--path PATH]... [--render FILE]...
       check_no_demo_block.py --self-test
Exit 0 when clean; 1 listing every hit otherwise.
"""
import argparse
import fnmatch
import os
import re
import subprocess
import sys
import tempfile

DEFAULT_PATHS = ['k8s', 'docker', 'config']

# Where a demo block belongs. Kept narrow on purpose: everything else is a
# shipped profile.
DEMO_ALLOW = [
    'k8s/overlays/demo*/*',
    'docker/docker-compose.demo*.yml',
    'config/demo*/*',
]

# Files that may NAME the demo paths without being demo content: a demo
# runner. nextgsim has none yet (nextgcore's are deploy/eks/profiles/demo/ and
# k8s/demo-check.sh); a demo PR that adds one lists it here. Everything else
# that names one is a shipped profile pulling a demo in, which the text rule
# cannot see (a configMapGenerator file, a kustomize resource, a compose mount).
SELF = 'scripts/check_no_demo_block.py'
REF_ALLOW = []

# `demo:` as a YAML mapping key (optionally quoted) at the start of a line, or
# inside a flow mapping / single-line JSON object after `{` or `,`.
DEMO_KEY = re.compile(r'''(^\s*(-\s+)?|[{,]\s*)["']?demo["']?\s*:(\s|$|[{"'\[])''')
# A reference to a demo-only path from outside it.
# `configs?/`: nextgsim's tree is config/, nextgcore's configs/.
DEMO_REF = re.compile(r'(overlays/demo|configs?/demo|docker-compose\.demo)')
PROSE = ('.md', '.txt')


def tracked(paths, root):
    out = subprocess.run(
        ['git', '-C', root, 'ls-files', '--', *paths],
        check=True, capture_output=True, text=True,
    ).stdout
    return [p for p in out.splitlines() if p]


def allowed(path, globs=None):
    return any(fnmatch.fnmatch(path, g) for g in (DEMO_ALLOW if globs is None else globs))


def scan(files, root):
    hits = []
    for rel in files:
        # This checker's own self-test fixtures plant every shape it catches.
        if rel.endswith(PROSE) or allowed(rel) or rel == SELF:
            continue
        full = os.path.join(root, rel)
        try:
            with open(full, encoding='utf-8', errors='replace') as f:
                for n, line in enumerate(f, 1):
                    if line.lstrip().startswith('#'):
                        continue
                    if DEMO_KEY.search(line):
                        hits.append(f'{rel}:{n}: {line.rstrip()}')
                    elif DEMO_REF.search(line) and not allowed(rel, REF_ALLOW):
                        hits.append(f'{rel}:{n}: references a demo-only path: {line.rstrip()}')
        except (IsADirectoryError, FileNotFoundError):
            continue
    return hits


def config_has_demo(text):
    """A ConfigMap value that parses as a YAML/JSON mapping with a root `demo`."""
    import yaml
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError:
        return False
    return isinstance(doc, dict) and 'demo' in doc


def scan_render(path):
    """Structural pass over a rendered manifest stream (kubectl kustomize / helm
    template output): every ConfigMap data value that is a config with a root
    `demo` key. Catches what the text pass cannot: a demo file pulled in by a
    generator or a resource from an allowlisted path."""
    import yaml
    hits = []
    with open(path) as f:
        docs = list(yaml.safe_load_all(f))
    for d in docs:
        items = d.get('items', []) if isinstance(d, dict) and d.get('kind') == 'List' else [d]
        for o in items:
            if not isinstance(o, dict) or o.get('kind') != 'ConfigMap':
                continue
            for k, v in (o.get('data') or {}).items():
                if isinstance(v, str) and config_has_demo(v):
                    hits.append(f'{path}: ConfigMap {o.get("metadata", {}).get("name")}/{k} '
                                f'has a root demo: block')
    return hits


def self_test():
    """The scan must catch a planted block in the shapes it claims to cover,
    and must leave the allowed overlay paths, comments and prose alone."""
    with tempfile.TemporaryDirectory() as d:
        def put(rel, text):
            p = os.path.join(d, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, 'w') as f:
                f.write(text)
        subprocess.run(['git', 'init', '-q', d], check=True)
        put('k8s/configmap.yaml', 'kind: ConfigMap\ndata:\n  gnb.yaml: |\n    demo:\n      enabled: true\n')
        put('docker/exit-gate/gnb.yaml', 'nci: 16\ndemo:\n  enabled: false\n')
        put('docker/settings.json', '{\n  "demo": {"enabled": true}\n}\n')
        put('config/ue.yaml', '"demo" : {}\n')
        put('config/flow.yaml', 'gnb: {nci: 16, extra: {demo: {enabled: true}}}\n')
        put('config/oneline.json', '{"nci":16,"demo":{"enabled":true}}\n')
        put('k8s/kustomization.yaml',
            'configMapGenerator:\n  - name: x\n    files:\n      - ../config/demo-x/gnb.yaml\n')
        put('k8s/profile/kustomization.yaml', 'resources:\n  - ../overlays/demo-x\n')
        put('docker/docker-compose.yml',
            'services:\n  gnb:\n    volumes:\n      - ../config/demo-x/gnb.yaml:/etc/gnb.yaml\n')
        bad = ['k8s/configmap.yaml', 'docker/exit-gate/gnb.yaml', 'docker/settings.json',
               'config/ue.yaml', 'config/flow.yaml', 'config/oneline.json',
               'k8s/kustomization.yaml', 'k8s/profile/kustomization.yaml',
               'docker/docker-compose.yml']
        put('k8s/lookalike.yaml', 'name: {demolition: 1, nodemo: 2}\n')
        put('k8s/overlays/demo-x/patch.yaml', 'demo:\n  enabled: true\n')
        put('k8s/overlays/demo-x/kustomization.yaml', 'resources:\n  - ../../../config/demo-x/gnb.yaml\n')
        put('docker/docker-compose.demo-x.yml', 'demo:\n')
        put('config/demo-x/gnb.yaml', 'demo:\n  enabled: true\n')
        put('k8s/comment.yaml', '# demo:\n  # demo: true\ndemolition: 1\nnodemo: x\n')
        put('docker/exit-gate/README.md', 'demo:\n')
        subprocess.run(['git', '-C', d, 'add', '-A'], check=True)
        hits = scan(tracked(DEFAULT_PATHS, d), d)
        got = sorted({h.split(':', 1)[0] for h in hits})
        if got != sorted(bad):
            print(f'self-test FAILED: flagged {got}, expected {sorted(bad)}')
            for h in hits:
                print(f'  {h}')
            return 1
        # Render pass: a ConfigMap whose config has a root demo key, in YAML
        # and in a List, is caught; one without, or with demo below the root,
        # is not.
        r = os.path.join(d, 'render.yaml')
        with open(r, 'w') as f:
            f.write('kind: ConfigMap\nmetadata: {name: a}\ndata:\n'
                    '  ue.yaml: "supi: x\\ndemo: {enabled: true}\\n"\n'
                    '  ok.yaml: "gnb: {demo: 1}\\n"\n'
                    '---\nkind: List\nitems:\n- kind: ConfigMap\n  metadata: {name: b}\n'
                    '  data: {gnb.yaml: \'{"demo": {}}\'}\n'
                    '---\nkind: ConfigMap\nmetadata: {name: c}\ndata: {x.yaml: "plain"}\n')
        rh = scan_render(r)
        want = ['ConfigMap a/ue.yaml', 'ConfigMap b/gnb.yaml']
        if len(rh) != 2 or not all(any(w in h for h in rh) for w in want):
            print(f'self-test FAILED (render pass): {rh}')
            return 1
    print('self-test ok: planted demo blocks and references are caught (text and render), '
          'demo overlays/profile/comments/prose/lookalikes are not')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--path', action='append', help='tracked path to scan (repeatable)')
    ap.add_argument('--root', default='.', help='repository root (default .)')
    ap.add_argument('--render', action='append', default=[],
                    help='rendered manifest stream to check structurally (repeatable)')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    hits = scan(tracked(a.path or DEFAULT_PATHS, a.root), a.root)
    for r in a.render:
        hits += scan_render(r)
    if hits:
        print('A `demo:` block is in a shipped profile (nextgsim#458: demos run only from '
              'their own overlay, e.g. k8s/overlays/demo-<key>/):')
        for h in hits:
            print(f'  {h}')
        return 1
    print('no demo: block in shipped manifests, compose files or configs')
    return 0


if __name__ == '__main__':
    sys.exit(main())
