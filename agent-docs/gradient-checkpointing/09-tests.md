# Decision 09 — Correctness tests

- Date: 2026-11-14. Status: **accepted** by the user.
- Plan: [00-plan.md](00-plan.md) §6.
- **Working rule (user):** tests may be added, edited or removed without asking. The list below is the starting point.

## Decision

New file `tests/checkpoint/test_checkpoint.py` (one folder per feature, like `tests/fp16test/` and `tests/mma/`),
parametrized over the numpy, cpu and cuda devices like the existing tests
([test_cast_release.py:14-19](../../tests/fp16test/test_cast_release.py#L14-L19)).

| No. | What the test checks | Decision |
|---|---|---|
| 1 | Gradients with and without `checkpoint()` are bit-identical for an MLP, a `TransformerLayer` and an `LSTM` | plan |
| 2 | Test 1 also under float16 and bfloat16 autocast | plan |
| 3 | `Dropout` with a fixed seed: identical gradients with and without `checkpoint()` | plan |
| 4 | `BatchNorm`: running statistics change once per forward | plan |
| 5 | After forward, interior buffers are freed (weakref) and the segment outputs keep their data | [02](02-segment-membership.md), [03](03-segment-outputs.md) |
| 6 | After backward, recomputed interior nodes are released again and outputs keep their data | [05](05-backward-release-scope.md) |
| 7 | A counting op runs once in forward and at most once more in backward | plan |
| 8 | In an `LSTM` segment, `Split` and `MakeTensorTuple` nodes are released | [06](06-tensortuple.md) |
| 9 | A nested return value `output, (h, c)` keeps all three tensors | [03](03-segment-outputs.md) |
| 10 | Under `no_grad`, `checkpoint()` returns the same result as `fn` and releases nothing | [08](08-special-modes.md) |
| 11 | An interior tensor read after `opt.step()` has the new-weight value (records the documented limit) | [05](05-backward-release-scope.md) |
| 12 | A second backward through the same graph gives the same gradients | [05](05-backward-release-scope.md) |

Not in this decision: measurement tests (decision [07](07-cast-of-segment-inputs.md) measurement, VRAM saving). They
need a memory counter (open decision: memory measurement).

## Update 2026-11-14: as implemented (log [03](log/03-checkpoint.md))

[tests/checkpoint/test_checkpoint.py](../../tests/checkpoint/test_checkpoint.py): 13 test functions, 60 cases on numpy,
cpu and cuda, all passing. Changes to the list above (Claude, allowed by the working rule):
- Tests 1–2 use two chained segments for the MLP and the `TransformerLayer`, one segment with a nested result for the
  `LSTM`.
- Test 6 also checks that the `.grad` of interior nodes is `None` after backward (decision 05 update).
- Test 9 also checks a mixed list/tuple result.
- New test `test_cast_releases_segment_input` (decision [07](07-cast-of-segment-inputs.md)): an explicit cast in the
  second segment releases the first segment's output; the gradients stay identical.

## Why bit-identical gradients are expected (tests 1–3)

- The recompute runs the same ops on the same inputs.
- The graph does not change, so backward sums the adjoints in the same order.
- Recompute is already tested to be bit-exact on all three devices
  ([test_cast_release.py:158-164](../../tests/fp16test/test_cast_release.py#L158-L164)).

## Comparison with PyTorch

PyTorch gets identical Dropout masks in the recompute by saving and restoring the RNG state
(`preserve_rng_state`, `torch/utils/checkpoint.py` L1576-1585). needle has no RNG step in the recompute because the
Dropout mask is a leaf that is never released (plan §3.2), so test 3 checks that the mask is reused.
