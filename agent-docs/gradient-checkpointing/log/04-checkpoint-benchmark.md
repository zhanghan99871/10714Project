# Log 04 — Checkpoint benchmark (order of work, step 4)

- Date: 2026-11-14. Branch: `gradient-checkpointing`. Status: **implemented, run, committed** in `28fc334` after the user's review (log [05](05-review-and-commit.md)).
- Decision: [11-benchmark.md](../11-benchmark.md) (updated); decision-07 measurement ([07](../07-cast-of-segment-inputs.md), updated).

## What was done

[apps/benchmark.py](../../../apps/benchmark.py):
- `--checkpoint`: `LanguageModel(..., checkpoint=True)`, one segment per `TransformerLayer` (log [03](03-checkpoint.md)).
- `--release-inputs`: runs under `autocast(release_inputs=True)`; `epoch_general_ptb` opens autocast itself, so the
  benchmark replaces `simple_ml._amp_context` (no change to `simple_ml`).
- After the timed block of a checkpoint run, one extra forward counts the segment inputs (non-leaf) that a cast
  released, through a wrapper around `nn_transformer.checkpoint`.
- `--suite amp` (the 12 configurations of log [02](02-amp-benchmark.md), replaces `--all`) and `--suite checkpoint`
  (14 configurations).
- New size `long` (sequence length 512, otherwise as `activations`), to test a size that does not fit without
  checkpoints.

Command (GPU 1):
`CUDA_VISIBLE_DEVICES=1 python apps/benchmark.py --suite checkpoint --steps 10 --warmup 2 --out agent-workspace/benchmark/checkpoint.json`
Raw results: [agent-workspace/benchmark/checkpoint.json](../../../agent-workspace/benchmark/checkpoint.json),
[agent-workspace/benchmark/checkpoint.log](../../../agent-workspace/benchmark/checkpoint.log).

## Results

Transformer LM, 4 layers, embedding 256, hidden 1024, batch 32, vocabulary 1000. Ratios are against float32 without
checkpoints of the same size; speeds use the counter-off column (log 02, finding 6).

| size (seq_len) | amp | checkpoint | release_inputs | peak MB | peak ratio | ms/step (counter on) | ms/step (counter off) | speed ratio | loss | released segment inputs |
|---|---|---|---|---|---|---|---|---|---|---|
| activations (256) | float32 | off | off | 10543 | 1.00 | 3431 | 3435 | 1.00 | 14.9774 | – |
| activations (256) | float32 | **on** | off | **3046** | **0.29** | 3948 | 3956 | 0.87 | 14.9774 | 0 |
| activations (256) | float16 | off | off | 10618 | 1.01 | 2520 | 2496 | 1.38 | 14.9775 | – |
| activations (256) | float16 | **on** | off | **2841** | **0.27** | 2820 | 2793 | 1.23 | 14.9775 | 0 |
| activations (256) | bfloat16 | off | off | 10618 | 1.01 | 2500 | 2466 | 1.39 | 14.9776 | – |
| activations (256) | bfloat16 | **on** | off | **2841** | **0.27** | 2812 | 2798 | 1.23 | 14.9776 | 0 |
| activations (256) | float16 | off | on | 9858 | 0.94 | 2630 | 2506 | 1.37 | 14.9775 | – |
| activations (256) | float16 | **on** | on | **2817** | **0.27** | 2797 | 2790 | 1.23 | 14.9775 | 0 |
| long (512) | float32 | off | off | out of memory | | | | | | |
| long (512) | float32 | **on** | off | **9012** | | 9614 | 9607 | | 14.5496 | 0 |
| long (512) | float16 | off | off | out of memory | | | | | | |
| long (512) | float16 | **on** | off | **7940** | | 7144 | 7162 | | 14.5494 | 0 |
| long (512) | bfloat16 | off | off | out of memory | | | | | | |
| long (512) | bfloat16 | **on** | off | **7940** | | 7170 | 7149 | | 14.5496 | 0 |

## Findings

1. **Checkpoints cut the peak by 71–73%** (10 543 → 3046 MB in float32, 10 618 → 2841 MB under AMP).
2. **They cost 12–15% more time per step** (float32 3435 → 3956 ms; float16 2496 → 2793; bfloat16 2466 → 2798).
   Backward recomputes each layer once.
3. **The loss is the same with and without checkpoints** for every dtype (to the printed precision; the tests show
   bit-identical gradients).
4. **A sequence length of 512 fits only with checkpoints:** without them all three dtypes run out of memory on the 16 GB
   GPU; with them the peak is 9012 MB (float32) and 7940 MB (float16 / bfloat16).
5. **AMP and checkpoints combine.** With checkpoints, AMP lowers the peak a little more (3046 → 2841 MB, −7%) and
   keeps most of its speed advantage (3956 → 2793 ms, 1.42×).
6. **Decision 07: no cast released a segment input** in any run, also under `autocast(release_inputs=True)` (0 in every
   checkpoint run). `release_inputs=True` lowers the peak by 7% without checkpoints (10 618 → 9858 MB) and by 1% with
   them (2841 → 2817 MB); the time differences are within the run-to-run variation.
7. **Memory counter:** counter-on and counter-off times differ by less than 5% in every run (log 02, finding 6).

## Decisions during this step (made by Claude, as allowed by the user)

| # | Decision | Reason | Recorded in |
|---|---|---|---|
| 1 | `--suite amp` / `--suite checkpoint` replace `--all`; the AMP suite keeps the 12 configurations of log 02 | one script for both benchmarks | [11](../11-benchmark.md) (updated) |
| 2 | New size `long` (seq_len 512) in the checkpoint suite | shows a size that fits only with checkpoints | [11](../11-benchmark.md) (updated) |
| 3 | `release_inputs` through a replacement of `simple_ml._amp_context` in the benchmark | `epoch_general_ptb` has no argument for it; no change to `simple_ml` | this log |
| 4 | Released segment inputs counted in one extra forward, with a wrapper around `nn_transformer.checkpoint` | `epoch_general_ptb` gives no access to the graph between forward and backward | this log |
| 5 | No input protection (decision 07 follow-up) | measured: no cast released a segment input in the transformer | [07](../07-cast-of-segment-inputs.md) (updated) |
| 6 | Steps 3 and 4 stay uncommitted | the user asked to commit only the AMP benchmark; **later** committed as `28fc334` after the user's review (log [05](05-review-and-commit.md)) | this log |

## Also changed in this step

The `checkpoint()` docstring said that the segment inputs "keep their data". A cast in `fn` can release an input
(decision 07), so it now says that `checkpoint()` does not release the inputs or the outputs, and that backward also
releases the adjoint of a recomputed activation. Full suite after the change: `904 passed in 35.92s`
([agent-workspace/pytest_after_checkpoint_final.log](../../../agent-workspace/pytest_after_checkpoint_final.log));
`tests/checkpoint` after the final docstring reflow: `60 passed`.
