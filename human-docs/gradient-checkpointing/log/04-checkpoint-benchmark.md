# Log 04: The checkpoint benchmark (step 4)

Date: 2026-11-14. Branch: `gradient-checkpointing`. Status: done, run and committed (`28fc334`).
The full version is in [agent-docs/gradient-checkpointing/log/04-checkpoint-benchmark.md](../../../agent-docs/gradient-checkpointing/log/04-checkpoint-benchmark.md).

## What we did

- `apps/benchmark.py` has the new options `--checkpoint` and `--release-inputs`. `--suite checkpoint` runs 14 configurations on GPU 1.
- A new size, `long`, has a sequence length of 512.

## Results (transformer, 4 layers, batch 32)

| Sequence length | dtype | Peak without checkpoints | Peak with checkpoints | Time with checkpoints compared with no checkpoints |
|---|---|---|---|---|
| 256 | float32 | 10 543 MB | 3046 MB | 15% more |
| 256 | float16 | 10 618 MB | 2841 MB | 12% more |
| 256 | bfloat16 | 10 618 MB | 2841 MB | 13% more |
| 512 | float32 | out of memory | 9012 MB | – |
| 512 | float16 | out of memory | 7940 MB | – |
| 512 | bfloat16 | out of memory | 7940 MB | – |

## Findings

1. Checkpoints decrease the peak memory by 71–73%.
2. Each step takes 12–15% more time, because backward computes each layer again.
3. The loss is the same with and without checkpoints.
4. A sequence length of 512 fits on the GPU only with checkpoints.
5. AMP and checkpoints work together. With checkpoints, float16 is 1.42 times faster than float32.
6. Decision 07: no cast released a segment input, also with `autocast(release_inputs=True)`. Thus, we do not add protection.

## Decisions in this step (made by Claude)

1. `--suite amp` and `--suite checkpoint` replace `--all`.
2. The new size `long` shows a size that fits only with checkpoints.
3. The benchmark sets `release_inputs` itself, so `simple_ml` does not change.
4. One more forward pass counts the segment inputs that a cast released.
5. `checkpoint()` gets no input protection.
6. Steps 3 and 4 are not committed. Later, after your review, they are committed as `28fc334` (log 05).

## Also changed

The docstring of `checkpoint()` now says that it does not release the inputs or the outputs. Before, it said that the inputs keep their data, but a cast in `fn` can release an input.
