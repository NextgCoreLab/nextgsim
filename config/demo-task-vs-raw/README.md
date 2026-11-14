# Demo D444C: task-oriented classification vs raw under an emulated AWGN sweep (nextgsim#479)

**Crate-level and non-normative.** No gNB, UE, NF or network is involved; the
noise is an emulated AWGN channel added in-process, and the data comes from a
pinned synthetic generator. See `docs-book/src/demos/d444c-task-vs-raw.md`.

| File | What |
|---|---|
| `task_vs_raw.yaml` | the `demo:` block of the `task_vs_raw` example, mode `all` |

```sh
cargo run -p nextgsim-semantic --features demo-d444c-task-vs-raw --example task_vs_raw -- \
    --config config/demo-task-vs-raw/task_vs_raw.yaml --seed 444
```
