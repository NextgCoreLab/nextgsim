# Features & APIs

## Supported-Features Matrix

The [feature matrix](../features.html) lists the gNB and UE feature surfaces actually wired into
live task code — NGAP/RRC/NAS procedures, user plane, and Rel-17/18 + 6G prototype rows — each
with its governing 3GPP spec and an honest status badge (**implemented** / **partial** /
**prototype**).

## CLI & API

The [CLI & API page](../api.html) documents the runtime control surface. See also the
[nr-cli Reference](cli-reference.md) chapter in this book.

## Validation status

E2E validation runs against the matched NextGCore 5G core (Docker recipe in
[Getting Started](getting-started.md)). This proves internal consistency of the pair — it is
**not** third-party conformance certification. The 6G/AI crates are non-normative research
prototypes of 3GPP Stage-1 studies; several Rel-17/18 features use sim-internal encodings where
no wire spec is implemented (see the feature matrix for per-row status).
