# Log 03 — `checkpoint()` (order of work, step 3)

- Date: 2026-11-14. Branch: `gradient-checkpointing`. Status: **implemented, tested, committed** in `28fc334` after the
  user's review (log [05](05-review-and-commit.md)). Line numbers refer to `28fc334`.
- Decisions: [01](../01-api-form.md)–[08](../08-special-modes.md); tests [09](../09-tests.md).

## What was done

| File | Change |
|---|---|
| [python/needle/autograd.py](../../../python/needle/autograd.py) | `_CheckpointState` / `_CHECKPOINT` next to the grad-mode state (L219-231); in `Value._init` the `_checkpoint_interior` flag and the record hook (L397-401); the `_shape is None` guard removed from `release_cached_data()` (decision 06); `checkpoint()` with its docstring (L687-721) and `_segment_outputs()` (L724-728); in `_compute_gradient_of_variables` the release of interior nodes and their adjoints (L803-808) |
| [python/needle/\_\_init\_\_.py](../../../python/needle/__init__.py) | `from .autograd import checkpoint` (L5) → `ndl.checkpoint` |
| [python/needle/nn/nn_transformer.py](../../../python/needle/nn/nn_transformer.py) | `Transformer(..., checkpoint=False)` (L335, L343); `forward` runs one segment per layer when on (L389-393) |
| [apps/models.py](../../../apps/models.py) | `LanguageModel(..., checkpoint=False)` passed to `nn.Transformer` (L65, L75, L86) |
| [tests/checkpoint/test_checkpoint.py](../../../tests/checkpoint/test_checkpoint.py) | 13 test functions, 60 cases (decision 09 update) |

Diff: 4 library files, +83 / −10 lines in `28fc334`.

Process: the AMP benchmark was running and re-imports `python/needle` and `apps/` in every configuration process, so
the code was developed in a scratch copy and applied to the repository with
[agent-workspace/apply_checkpoint.py](../../../agent-workspace/apply_checkpoint.py) after the AMP commit. The repository
files are byte-identical to the scratch copy (checked with `cmp`).

## Results

- `tests/checkpoint`: `60 passed in 1.90s`.
- Full suite: `904 passed in 36.65s` (844 before + 60 new).
  Log: [agent-workspace/pytest_after_checkpoint.log](../../../agent-workspace/pytest_after_checkpoint.log).
- Memory, cpu, small transformer LM (4 layers, batch 8, seq 64), one step
  ([probe_checkpoint_memory.py](../../../agent-workspace/probe_checkpoint_memory.py)):
  ```
  no checkpoint: after forward   93.4 MB | peak  194.1 MB | after backward  167.2 MB
  checkpoint   : after forward   17.2 MB | peak   50.5 MB | after backward   20.2 MB
  ```
- First GPU run of the benchmark (transformer, activations size, 1 step): float32 peak 10 543 MB without checkpoints,
  3046 MB with them; 3347 vs 3889 ms per step; same loss. Full results: log [04](04-checkpoint-benchmark.md).

## Decisions during this step (made by Claude, as allowed by the user)

| # | Decision | Reason | Recorded in |
|---|---|---|---|
| 1 | Backward also drops the `.grad` and the partial adjoints of an interior node when it releases the node | the adjoints of interior nodes were the largest part of the peak: 117.9 → 50.5 MB in the probe | [05](../05-backward-release-scope.md) (updated) |
| 2 | `checkpoint()` resets the record in a `finally` block | if `fn` raises, ops made later must not be recorded; `set_grad_enabled` also restores its state on an exception ([autograd.py:199-201](../../../python/needle/autograd.py#L199-L201)) | this log |
| 3 | The docstring lists the limits of decisions 03 (dicts), 04 (nesting), 05 (read after step) and 08 (`LAZY_MODE`); no note on casts | these four were decided as docstring notes; decision 07 did not include a note | this log |
| 4 | `Transformer.forward` loops over `self.layers.modules` and calls `checkpoint(layer, x)`; `nn_transformer` imports `checkpoint` from `needle.autograd` | decision 11 integration; the `checkpoint` constructor argument shadows the function only inside `__init__` | this log |
| 5 | Tests: two chained segments (MLP, `TransformerLayer`), one nested-result segment (`LSTM`); interior `.grad` check; mixed list/tuple result; a decision-07 test | working rule: tests may change without asking | [09](../09-tests.md) (updated) |

## Next

Step 4, the checkpoint benchmark — log [04](04-checkpoint-benchmark.md).
