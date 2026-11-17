# Decision 03 — How `checkpoint()` finds segment outputs

- Date: 2026-11-14. Status: **accepted** by the user.
- Plan: [00-plan.md](00-plan.md), step 2; decision [02](02-segment-membership.md) ("release the recorded non-leaf
  tensors, except the segment outputs").

## Decision

`checkpoint()` **searches the return value of `fn` through nested tuples and lists and keeps every `Value` (`Tensor` or
`TensorTuple`, decision [06](06-tensortuple.md)) that it finds** (these are the segment outputs; they are not released). The return value itself is returned unchanged.

**Dicts are not supported yet.** The `checkpoint()` docstring says so, with the reason: `TransformerLayer`, `RNN` and
`LSTM` do not return dicts.

## Context

Decision 02 releases every recorded non-leaf tensor except the outputs, so `checkpoint()` must find the outputs. If it
released an output, the caller's value would be lost (needle has one buffer per node) and the next segment would
recompute it.

Return forms of needle modules:

| Module | Return | Reference |
|---|---|---|
| `TransformerLayer` | `x` (`Tensor`) | [nn_transformer.py:316](../../python/needle/nn/nn_transformer.py#L316) |
| `MultiHeadAttention` | `result, probs` (tuple) | [nn_transformer.py:142](../../python/needle/nn/nn_transformer.py#L142) |
| `RNN` | `output, h_n` (tuple) | [nn_sequence.py:199](../../python/needle/nn/nn_sequence.py#L199) |
| `LSTMCell` | `new_h, new_c` (tuple) | [nn_sequence.py:316](../../python/needle/nn/nn_sequence.py#L316) |
| `LSTM` | `output, (h_n, c_n)` (nested tuple) | [nn_sequence.py:393](../../python/needle/nn/nn_sequence.py#L393) |
| `Transformer` | `x, init.zeros_like(x)` (tuple) | [nn_transformer.py:393](../../python/needle/nn/nn_transformer.py#L393) |

## Options considered

- `Tensor` only — fails for every tuple-returning module.
- `Tensor` and flat tuple/list — fails for `LSTM`.
- **Nested tuples and lists (chosen)** — covers every form in the table.
- Also dicts — no needle module returns one.

## Comparison with PyTorch

Reference: PyTorch 2.10.0, `torch/utils/checkpoint.py` (env `vda`).

- PyTorch does not need to find the outputs. The non-reentrant pack hook replaces only the graph's reference to a saved
  tensor with a `_Holder` (L1144-1152); the output tensor that the caller holds keeps its storage.
- Nested structures (docstring L417-419): *"The reentrant version does not consider tensors in nested structures (e.g.,
  custom objects, lists, dicts, etc) as participating in autograd, while the non-reentrant version does."*
- needle sits between the two: nested tuples and lists are supported, dicts and custom objects are not.

## Justification

1. Covers every return form that needle modules use, including the nested tuple of `LSTM`.
2. Minimal code: one short recursive function, like `_unpack_params`
   ([nn_basic.py:14-30](../../python/needle/nn/nn_basic.py#L14-L30)).
3. Dict support is not needed by any current module; the docstring states the limit so a user is not surprised.

## Consequences

- A tensor inside an unsupported container (dict, custom object) is not found, so `checkpoint()` treats it as an
  interior node: released at the end of the segment, recomputed when read, and subject to the read-after-step limit
  (decision [05](05-backward-release-scope.md), "Consequences and limits").
- A `TensorTuple` in the return value counts as an output (decision [06](06-tensortuple.md)).
