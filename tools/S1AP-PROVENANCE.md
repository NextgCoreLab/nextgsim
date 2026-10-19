# Provenance of `tools/s1ap-19.2.0.asn`

| | |
|---|---|
| Source | 3GPP TS 36.413 V19.2.0 (2026-06), `36413-j20.docx` from `https://www.3gpp.org/ftp/Specs/archive/36_series/36.413/` |
| Source sha256 | `a4f845161803b1a77659175795c635cf09b6711dd77f11f7f68702b551c07ae7` |
| Output sha256 | `c3d60594ecb7dbc730efe740b66f05f91dfd1fd69344353dea3372b1a9d2ac62` |
| Modules | S1AP-PDU-Descriptions, S1AP-PDU-Contents, S1AP-IEs, S1AP-CommonDataTypes, S1AP-Constants, S1AP-Containers (§9.3.2–§9.3.7) |
| Excluded | `SonTransfer-IEs` (Annex B), see below |

## Method (reproducible)

```bash
python3 tools/extract_asn1.py 36413-j20.docx tools/s1ap-19.2.0.asn \
    --skip-module SonTransfer-IEs
sha256sum tools/s1ap-19.2.0.asn   # must match the table above
```

`extract_asn1.py` reads the document's paragraphs straight from
`word/document.xml`, one per line, and keeps the text between each
`-- ASN1START` / `-- ASN1STOP` pair, which is how 3GPP delimits normative ASN.1.
No other edits are made.

It reads the `.docx`, not a pandoc `.txt`, on purpose. pandoc's plain-text
writer wraps long paragraphs (the copy in `6g_docs/specs/36413-j20.txt` splits
`s1ap-PDU-Descriptions (0)}` over two lines), and a wrapped `--` comment turns
its tail into schema text. The same script will extract TS 36.331 for the LTE
RRC increment.

## Why `SonTransfer-IEs` is excluded

Annex B defines it as the IE module of the SON Transfer *application*. S1AP
carries those IEs inside transparent containers, and no S1AP module imports
from it, so compiling it into the S1AP codec would only add unrelated types.
It can be generated into a module of its own if the SON transfer procedures are
ever implemented, as NRPPa is beside NGAP.

## Upgrading

Download the newer `36413-xNN.docx`, rerun the command above into
`tools/s1ap-<version>.asn`, point `nextgsim-s1ap/build.rs` at it, and update
this table, including both sha256 values.
