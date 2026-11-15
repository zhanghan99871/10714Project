# Decision 08: `no_grad`, no tensor that requires grad, and `LAZY_MODE`

Date: 2026-11-14. Status: accepted.
The full version is in [agent-docs/gradient-checkpointing/08-special-modes.md](../../agent-docs/gradient-checkpointing/08-special-modes.md).

## Decision

- **Under `no_grad`, or when no tensor in `fn` requires grad:** no code. `checkpoint()` records nothing, releases nothing, and only gives back the result of `fn`.
- **`LAZY_MODE`:** `checkpoint()` does not support it. One line in the docstring tells this.

## Why the first two cases need no code

- In both cases, no op result requires grad. Thus, each op gives a plain leaf, and no graph node.
- The record hook records only nodes that come from an op. Thus, it records nothing.
- A test shows this. In normal training, the parameters require grad, so the ops make graph nodes, and `checkpoint()` works normally.

## Comparison with PyTorch

- Under `no_grad`, the non-reentrant mode of PyTorch only runs `fn`. Needle does the same.
- The reentrant mode needs an input that requires grad. If no output requires grad, it raises an error.
- PyTorch has no lazy mode.

## Justification

1. The current code already gives the correct behavior.
2. Only three tests use `LAZY_MODE`. A line in the docstring is the simplest choice.
