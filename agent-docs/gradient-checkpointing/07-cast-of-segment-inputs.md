# Decision 07 — Casts that release a segment input

- Date: 2026-11-14. Status: **accepted** by the user.
- Plan: [00-plan.md](00-plan.md) §5 ("Explicit casts inside a segment release their input").

## Decision

In the **first version, `checkpoint()` does not protect its inputs** from a cast that releases them. Test cases will
measure how often such a release makes backward recompute the previous segment, and how much VRAM it costs or saves.
Protection is added later if the tests show a need. (No docstring note was decided.)

## Context

- `ops.Cast` releases a non-leaf input by default (`release_input=True`,
  [ops_mathematic.py:50-53](../../python/needle/ops/ops_mathematic.py#L50-L53), [ops_mathematic.py:64-72](../../python/needle/ops/ops_mathematic.py#L64-L72));
  `.astype()`, `.half()`, `.float()`, `.bfloat16()` do the same ([autograd.py:655-670](../../python/needle/autograd.py#L655-L670)).
- Casts that autocast adds go through `_autocast_cast()` ([autograd.py:108-123](../../python/needle/autograd.py#L108-L123)):
  parameters with `release_input=False` (line 120); other tensors with `_AUTOCAST.release_inputs` (line 123), default
  `False` ([amp.py:81](../../python/needle/amp.py#L81); reason in [amp.py:75-78](../../python/needle/amp.py#L75-L78)).
  Probe [agent-workspace/probe_implicit_cast_release.py](../../agent-workspace/probe_implicit_cast_release.py):
  ```
  autocast(release_inputs=False): y float16, x released: False
  autocast(release_inputs=True ): y float16, x released: True
  ```
- All other casts in the library pass `release_input=False`: backward seed
  ([autograd.py:576](../../python/needle/autograd.py#L576)), adjoint sums ([autograd.py:699](../../python/needle/autograd.py#L699),
  [autograd.py:703](../../python/needle/autograd.py#L703)), dtype fix-up ([autograd.py:739](../../python/needle/autograd.py#L739)),
  `Cast.gradient` ([ops_mathematic.py:61](../../python/needle/ops/ops_mathematic.py#L61)), `GradScaler.scale`
  ([amp.py:190](../../python/needle/amp.py#L190)).
- So a segment input is released by an explicit cast in `fn`, or by an autocast cast under
  `autocast(release_inputs=True)`. Then, in backward, the recompute of the segment needs the input and recomputes it
  from the released interior nodes of the previous segment. Gradients stay correct; backward does extra work, and two
  segments are in memory for a short time. Between forward and backward, the release also frees the input's buffer —
  hence "costs or saves".
- No needle module or app casts explicitly (the only `astype` in `nn/` is a numpy cast of a mask constant,
  [nn_transformer.py:64](../../python/needle/nn/nn_transformer.py#L64)).

## Options considered

- Protect the segment inputs: `checkpoint()` marks them and `_release_cast_input()` skips marked tensors (~3 lines).
- Do not protect, with a docstring note.
- **Do not protect in the first version; measure first (chosen).**

## Comparison with PyTorch

- A PyTorch cast does not keep its input: `ToCopyBackward0` saves no tensor, so the input is freed when nothing else
  holds it, and kept when another op saved it. Probe
  [agent-workspace/probe_torch_cast_memory.py](../../agent-workspace/probe_torch_cast_memory.py) (GPU, PyTorch 2.10.0):
  ```
  cast grad_fn: ToCopyBackward0 | saved tensors: []
  case A  after cast: 6 MB | after `del h`: 2 MB  (float32 input freed: nothing holds it)
  case B  after cast + `del h`: 6 MB  (float32 input kept: ExpBackward0 saved it)
  ```
- needle needs the explicit release because every node keeps its `inputs` ([autograd.py:380](../../python/needle/autograd.py#L380));
  the `Cast` docstring says so ([ops_mathematic.py:28-32](../../python/needle/ops/ops_mathematic.py#L28-L32)). In case A
  needle's release matches PyTorch; in case B it frees more than PyTorch and recomputes later.
- PyTorch `checkpoint()` keeps its inputs explicitly: `ctx.save_for_backward(*tensor_inputs)` (reentrant,
  `torch/utils/checkpoint.py` L264) and `_NoopSaveInputs` (non-reentrant, L1600). A PyTorch cast never frees them.

## Justification

1. **User:** measure first, then decide whether protection is worth its code.
2. No needle code triggers it with default settings, so the first version loses nothing in the planned use cases.

## Follow-up

- Added to the test scope ([00-plan.md](00-plan.md) §6): measure how often a released segment input causes a recompute
  of the previous segment, and its VRAM effect, for an explicit cast and for `autocast(release_inputs=True)`.

## Update 2026-11-14: measured (log [04](log/04-checkpoint-benchmark.md))

- Transformer LM, one segment per layer: **0 segment inputs released by a cast** in every checkpoint run, also under
  `autocast(release_inputs=True)`.
- `release_inputs=True` lowers the peak by 7% without checkpoints (10 618 → 9858 MB) and by 1% with them
  (2841 → 2817 MB); the time differences are within the run-to-run variation.
- **Decision (Claude, allowed by the user): no protection.** Nothing in the planned use case triggers it. The test
  `test_cast_releases_segment_input` keeps the behaviour of an explicit cast correct.
