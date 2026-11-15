# Decision 06: `TensorTuple` nodes

Date: 2026-11-14. Status: accepted.
The full version is in [agent-docs/gradient-checkpointing/06-tensortuple.md](../../agent-docs/gradient-checkpointing/06-tensortuple.md).

## Decision

`checkpoint()` releases `TensorTuple` nodes as it releases `Tensor` nodes. This needs three changes:

1. We remove the guard "Cannot release an uninitialized tensor." from `release_cached_data()`.
2. The record hook goes in `Value._init`. Both kinds of node go through `_init`, so one hook records both.
3. The search for segment outputs (decision 03) keeps each `Tensor` and each `TensorTuple` that it finds.

## Why

- `ops.split` (in `RNN` and `LSTM`) and `stack` make `TensorTuple` nodes.
- If `checkpoint()` skips these nodes, 12–13% of the segment memory of `RNN` and `LSTM` stays alive. `TransformerLayer` has no `TensorTuple` nodes.
- A test released all the nodes of an `LSTM`, including 24 `TensorTuple` nodes. The gradients were identical.

## The guard

- Han added the guard in commit `9806253`. The commit does not give a reason.
- For a `Tensor`, the guard fires only if the tensor was never computed. Then the release would change nothing.
- For a `TensorTuple`, the guard always fires, because a `TensorTuple` never has metadata.
- No code or test needs the guard. All 828 tests pass without it.

## Comparison with PyTorch

PyTorch has no tuple node. `torch.split` gives a normal Python tuple of tensors. The checkpoint hook of PyTorch sees each saved tensor, so an op with more than one output needs no special code. In needle, the tuple is a node of its own, so `checkpoint()` must record and release it.

## Justification

1. `RNN` and `LSTM` keep all of their memory saving.
2. The code is almost the same size as when we skip the nodes.
3. The guard has no purpose that code or tests need.

## Results

- The guard is Han's code. You may want to tell Han about the change.
