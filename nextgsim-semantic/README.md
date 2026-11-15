# nextgsim-semantic

Semantic-communication research prototypes for nextgsim: feature codecs, JSCC,
channel models, metrics, and the codec halves of demos D444A (nextgsim#477) and
D444C (nextgsim#479).

**Non-normative.** TR 22.870 (§9.13, §6.59.1) is a Stage-1 study with no Stage-3
wire spec; nothing here is a conformant 3GPP implementation.

## Demo D444C: `task_vs_raw` (feature `demo-d444c-task-vs-raw`)

- **Crate-level.** No gNB, UE, NF or network; one in-process sweep and one table.
- **Emulated channel.** `AwgnChannel` adds seeded Gaussian noise to the
  dequantised symbols in the receiver. A real PDU session would deliver the
  bytes bit-exact or drop them.
- **Synthetic data.** D444A's pinned generator; the gap between the task encoder
  (an unsupervised linear PCA), top-variance raw and mean-pool at equal bytes is
  set by that generator, not learned.

```sh
cargo run -p nextgsim-semantic --features demo-d444c-task-vs-raw --example task_vs_raw -- \
    --config config/demo-task-vs-raw/task_vs_raw.yaml --seed 444
```

See `docs-book/src/demos/d444c-task-vs-raw.md`.
