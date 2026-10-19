#!/usr/bin/env python3
"""Extract the ASN.1 modules from a 3GPP specification .docx.

3GPP marks every normative ASN.1 block with a ``-- ASN1START`` paragraph and
closes it with ``-- ASN1STOP``. This script reads the document's paragraphs
directly from ``word/document.xml`` -- one paragraph per output line, with tabs
and line breaks kept -- and writes the text between each START/STOP pair.

Reading the .docx rather than a pandoc ``.txt`` matters: pandoc's plain-text
writer wraps long paragraphs, and a wrapped ``--`` comment turns its tail into
schema text. Stdlib only, so it runs anywhere Python 3 does.

Usage:
    extract_asn1.py SPEC.docx OUT.asn [--blocks N] [--skip-module NAME ...]

``--blocks N`` keeps only the first N blocks; ``--skip-module`` drops any block
whose module header names NAME (e.g. an annex-defined module that the main
protocol does not import).
"""
import argparse
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def paragraphs(docx):
    with zipfile.ZipFile(docx) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    for p in root.find(W + "body").iter(W + "p"):
        parts = []
        for el in p.iter():
            if el.tag == W + "t":
                parts.append(el.text or "")
            elif el.tag == W + "tab":
                parts.append("\t")
            elif el.tag in (W + "br", W + "cr"):
                parts.append("\n")
        # Non-breaking and other exotic spaces are not ASN.1 whitespace.
        yield "".join(parts).replace(" ", " ").replace("‑", "-")


def blocks(lines):
    cur = None
    for line in lines:
        s = line.strip()
        if s == "-- ASN1START":
            cur = []
        elif s == "-- ASN1STOP":
            if cur is not None:
                yield cur
            cur = None
        elif cur is not None:
            cur.append(line)


MODULE_RE = re.compile(r"^([A-Z][A-Za-z0-9-]*)\s*(\{|DEFINITIONS|$)")


def module_name(block):
    for line in block:
        s = line.strip()
        if not s or s.startswith("--"):
            continue
        m = MODULE_RE.match(s)
        return m.group(1) if m else None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("docx")
    ap.add_argument("out")
    ap.add_argument("--blocks", type=int)
    ap.add_argument("--skip-module", action="append", default=[])
    a = ap.parse_args()
    out = []
    kept = []
    for i, b in enumerate(blocks(paragraphs(a.docx))):
        if a.blocks is not None and i >= a.blocks:
            break
        name = module_name(b)
        if name in a.skip_module:
            continue
        kept.append(name)
        out.append("\n".join(b).rstrip() + "\n")
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print("modules:", ", ".join(str(k) for k in kept), file=sys.stderr)


if __name__ == "__main__":
    main()
