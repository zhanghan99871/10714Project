# Decision 07: Casts that release a segment input

Date: 2026-11-14. Status: accepted.
The full version is in [agent-docs/gradient-checkpointing/07-cast-of-segment-inputs.md](../../agent-docs/gradient-checkpointing/07-cast-of-segment-inputs.md).

## Decision

In the first version, `checkpoint()` does not protect its inputs from a cast that releases them. Tests will measure how often this makes backward recompute the previous segment, and how much VRAM it costs or saves. If the tests show a need, we add protection later.

## The problem

- These casts release a segment input:
  - an explicit cast, such as `ops.cast()` or `.half()`, with the default `release_input=True`
  - a cast that autocast adds, under `autocast(release_inputs=True)`
- With the default settings, the casts that autocast adds keep their input. A test shows this.
- If a cast releases a segment input, backward must recompute that input from the previous segment. The gradients stay correct, but backward does more work, and two segments are in memory for a short time. Between forward and backward, the release frees memory.
- No needle module or app casts a tensor explicitly.

## Comparison with PyTorch

- A cast in PyTorch does not keep its input. PyTorch frees the input when nothing else holds it.
- Each needle node keeps its inputs. Thus, needle needs an explicit release to free the input.
- `checkpoint()` in PyTorch keeps its inputs explicitly, and a cast never frees them.

## Justification

1. We measure first. Then we decide whether protection is worth the code.
2. With the default settings, no needle code has this problem.

## Update: measured (log 04)

- In the transformer benchmark, no cast released a segment input, also with `autocast(release_inputs=True)`.
- `release_inputs=True` decreases the peak by 7% without checkpoints and by 1% with checkpoints.
- **Decision (Claude): no protection.** Nothing in our use case needs it. A test checks that an explicit cast still gives correct gradients.
