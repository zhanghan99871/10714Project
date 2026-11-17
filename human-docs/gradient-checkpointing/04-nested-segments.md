# Decision 04: Nested segments

Date: 2026-11-14. Status: accepted.
The full version is in [agent-docs/gradient-checkpointing/04-nested-segments.md](../../agent-docs/gradient-checkpointing/04-nested-segments.md).

## Decision

Needle does not support a `checkpoint()` call inside another `checkpoint()` call. The docstring of `checkpoint()` tells this. `checkpoint()` does not raise an error. We do not define or test what a nested call does.

## Options

- **The outer segment releases the inner outputs.** This is like PyTorch. It uses the least memory after forward, but it does not decrease the memory during backward in needle.
- **Each call makes its own segment.** The result is the same as a chain of separate segments.
- **No support, with a note in the docstring (chosen).**

## Comparison with PyTorch

- PyTorch supports nested checkpoints.
- When PyTorch recomputes an outer segment, the inner checkpoints still free their interior tensors. Thus, its memory during backward stays low.
- Needle keeps each node that it recomputes. Thus, in needle, nesting would not decrease the memory during backward.

## Justification

1. This is a course project with no outside users. Thus, we keep it simple.
2. No planned use case needs nesting.
3. In needle, nesting would give no memory benefit during backward.
4. A note in the docstring needs no code.
