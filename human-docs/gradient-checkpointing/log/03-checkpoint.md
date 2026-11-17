# Log 03: `checkpoint()` (step 3)

Date: 2026-11-14. Branch: `gradient-checkpointing`. Status: done, tested and committed (`28fc334`).
The full version is in [agent-docs/gradient-checkpointing/log/03-checkpoint.md](../../../agent-docs/gradient-checkpointing/log/03-checkpoint.md).

## What we did

- `autograd.py` has `checkpoint(fn, *args)` now, which the package exports as `ndl.checkpoint`. It follows decisions 01 to 08:
  - It records each node that an op makes during the segment.
  - It releases the interior nodes when `fn` returns, and keeps the outputs.
  - Backward recomputes a released node when a gradient needs it, and releases the node again after it processes it.
- We removed the guard "Cannot release an uninitialized tensor." from `release_cached_data()` (decision 06).
- `nn.Transformer` and `LanguageModel` have the option `checkpoint=False`. When it is on, each layer is one segment.
- `tests/checkpoint/test_checkpoint.py` has 13 tests, which give 60 cases.

## Results

- The 60 checkpoint cases pass. All 904 tests pass.
- In a small test on the CPU, the memory after forward fell from 93 MB to 17 MB, and the peak fell from 194 MB to 51 MB.
- On the GPU, the peak of the transformer benchmark fell from 10 543 MB to 3046 MB. Log 04 has all the results.

## Decisions in this step (made by Claude)

1. Backward also drops the gradients of interior nodes. Without this, the peak in the CPU test was 118 MB, not 51 MB. Decision 05 shows this update.
2. If `fn` raises an error, `checkpoint()` still stops recording.
3. The docstring tells the limits: dicts, nested segments, `LAZY_MODE`, and a read after the optimizer step.
4. `Transformer.forward` calls `checkpoint(layer, x)` for each layer.
5. The tests changed a little from decision 09. Decision 09 shows the changes.
