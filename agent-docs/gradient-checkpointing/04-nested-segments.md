# Decision 04 — Nested segments

- Date: 2026-11-14. Status: **accepted** by the user.
- Plan: [00-plan.md](00-plan.md).

## Decision

Nested checkpoints (a `checkpoint()` call inside another `checkpoint()` call) are **not supported**. The `checkpoint()`
docstring says so. `checkpoint()` does not check for nesting and does not raise an error; the behaviour of a nested
call is not defined and not tested.

## Context

The main use case checkpoints each `TransformerLayer` once and needs no nesting. A nested call would come from, for
example, checkpointing a whole model and also each of its layers.

## Options considered

- **(a) Outer releases the inner outputs** (like PyTorch Rules 1 and 2): least memory after forward, but the backward
  peak is the whole outer segment (see below).
- **(b) Each call is its own segment**; the outer keeps the inner outputs. Same result as a chain of flat segments.
- **(c) Not supported, documented in the docstring (chosen).**

## Comparison with PyTorch

Reference: PyTorch 2.10.0, `torch/utils/checkpoint.py` L621-650 (env `vda`). PyTorch non-reentrant supports nesting:

> Rule 1. Saved tensors are managed by inner-most checkpoint only and hidden from any outer layers of checkpoint.
>
> Rule 2. The inputs of inner checkpoints are treated as tensors saved to its parent checkpoint.
>
> Rule 3. We should start recomputation as if there are no checkpoints currently active. Checkpoints encountered
> during recomputation are still respected.

Rule 3 keeps the backward peak low: while the outer segment is recomputed, the inner checkpoints still free their
interior tensors. needle cannot follow Rule 3 without extra work: `realize_cached_data()` keeps every node that it
recomputes (`self.cached_data = data`, [autograd.py:328](../../python/needle/autograd.py#L328)), so a recompute that
passes through an inner segment keeps that segment's interior nodes too.

## Justification

1. **User:** this is a course project with no outside users; keep it simple.
2. No planned use case needs nesting.
3. In needle, nesting would not lower the backward peak below that of one flat outer segment (Rule 3 above), so
   supporting it would add semantics without a memory benefit.
4. Documenting instead of raising keeps the code minimal.
