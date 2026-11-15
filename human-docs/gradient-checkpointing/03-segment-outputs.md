# Decision 03: How `checkpoint()` finds the outputs of a segment

Date: 2026-11-14. Status: accepted.
The full version is in [agent-docs/gradient-checkpointing/03-segment-outputs.md](../../agent-docs/gradient-checkpointing/03-segment-outputs.md).

## Decision

`checkpoint()` searches the return value of `fn` through nested tuples and lists. It keeps each tensor that it finds. These tensors are the segment outputs, and `checkpoint()` does not release them.

`checkpoint()` does not support dicts yet. Its docstring tells this, and gives the reason: `TransformerLayer`, `RNN` and `LSTM` do not return dicts.

## Why `checkpoint()` must find the outputs

Decision 02 releases each recorded tensor except the outputs. Needle keeps only one buffer per node. If `checkpoint()` released an output, your value would be lost.

## The return values of needle modules

- `TransformerLayer` returns a tensor.
- `MultiHeadAttention`, `RNN`, `LSTMCell` and `Transformer` return a tuple.
- `LSTM` returns a nested tuple: `output, (h_n, c_n)`.

## Comparison with PyTorch

- PyTorch does not need to find the outputs. It removes only the graph's reference to a saved tensor. The output that your code holds keeps its value.
- The reentrant mode of PyTorch does not support tensors in nested structures. The non-reentrant mode supports them.
- Needle supports nested tuples and lists, but not dicts.

## Justification

1. It covers all the return values of the needle modules, including the nested tuple of `LSTM`.
2. It needs only one short recursive function.
3. No needle module returns a dict. The docstring tells the user about this limit.

## Results

- `checkpoint()` does not find a tensor in a dict. It releases that tensor as an interior node.
- A `TensorTuple` in the return value is also an output (decision 06).
