# Provenance of `tools/lte-rrc-19.4.0.asn`

| | |
|---|---|
| Source | 3GPP TS 36.331 V19.4.0, `36331-j40.docx` from `https://www.3gpp.org/ftp/Specs/archive/36_series/36.331/` |
| Source sha256 | `26407c6c0ae31fd5da09eb81f7e9cd8cf49c0c51606811b13ce15a2e326fe058` |
| Output sha256 | `d3a80171a15fb8ebfe99e73883021dd63f483e5c2058194a7d0d8326065e9084` |
| Module | `EUTRA-RRC-Definitions` (§6.2–§6.4), 17,808 lines; it has no `IMPORTS` |
| Excluded | `PC5-RRC-Definitions`, `NBIOT-RRC-Definitions`, `EUTRA-UE-Variables`, `NBIOT-UE-Variables`, `EUTRA-Sidelink-Preconf`, `EUTRA-InterNodeDefinitions`, `NBIOT-InterNodeDefinitions` |

## Method (reproducible)

```bash
python3 tools/extract_asn1.py 36331-j40.docx tools/lte-rrc-19.4.0.asn \
    --only-module EUTRA-RRC-Definitions --space-glued-comments
sha256sum tools/lte-rrc-19.4.0.asn   # must match the table above
```

TS 36.331 does not give each module one ASN.1 block the way TS 36.413 does:
it opens `EUTRA-RRC-Definitions` in one `-- ASN1START`/`-- ASN1STOP` block and
then delimits every message and IE separately -- about 560 blocks before the
module's `END`. `--only-module` therefore assigns each block to the module whose
header block last preceded it. The same change makes `--skip-module` drop a
whole module rather than only its header block; the S1AP extraction is
unaffected (it still reproduces its recorded sha256).

### The one edit: `--space-glued-comments`

Two lines of the source glue a comment to the keyword before it:

```
	simultaneousAckNackAndCQI-Format4-Format5-r13		ENUMERATED {setup}		OPTIONAL-- Need OR
	[[condReconfigurationTriggerNR-r17	CondReconfigurationTriggerNR-r17	OPTIONAL-- Need ON
```

The flag inserts one space before a `--` that directly follows a letter or
digit (`OPTIONAL -- Need OR`). X.680 §12.6.2 starts a comment at `--` wherever it
appears and §12.3 forbids `--` inside an identifier, so the edit cannot change
the schema; it is needed because the asn1-compiler tokenizer (NextgCoreLab/asn1-codegen) reads
`OPTIONAL--` as a single identifier and stops with
`Tokenize Error (16) at Line: 6354`. Every other character is the source's.

## Why only `EUTRA-RRC-Definitions`

It carries every message the eNB and UE exchange over the air (BCCH, CCCH,
DCCH, PCCH) and is self-contained. The UE variables are internal state, not a
wire format; the inter-node module (HandoverPreparationInformation,
UECapabilityInformation containers) is for later increments and can be
generated as its own module when needed; the NB-IoT, PC5 and sidelink modules
are other radio access types.

## Upgrading

Download the newer `36331-xNN.docx`, rerun the command above into
`tools/lte-rrc-<version>.asn`, point `nextgsim-lte-rrc/build.rs` at it, and
update this table, including both sha256 values.
