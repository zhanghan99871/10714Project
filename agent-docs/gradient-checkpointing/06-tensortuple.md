# Decision 06 — `TensorTuple` nodes

- Date: 2026-11-14. Status: **accepted** by the user.
- Plan: [00-plan.md](00-plan.md) §5. Changes decisions [02](02-segment-membership.md) (hook location) and
  [03](03-segment-outputs.md) (output search).

## Decision

`checkpoint()` releases `TensorTuple` nodes like `Tensor` nodes. Three changes:

1. **Remove the guard** `if self._shape is None: raise RuntimeError("Cannot release an uninitialized tensor.")` from
   `Value.release_cached_data()` ([autograd.py:341-344](../../python/needle/autograd.py#L341-L344)).
2. **The record hook goes in `Value._init`**, not in `Tensor.make_from_op`. Both `Value.make_from_op` (used by
   `TensorTuple`, [autograd.py:405](../../python/needle/autograd.py#L405)) and `Tensor.make_from_op`
   ([autograd.py:499](../../python/needle/autograd.py#L499)) call `_init`, so one hook records both kinds of node.
3. **The output search of decision 03 keeps every `Value`** (`Tensor` or `TensorTuple`) that it finds.

## Context

- `TensorTuple` nodes come from `Split` (`ops.split`, used by `RNN` and `LSTM`,
  [nn_sequence.py:180](../../python/needle/nn/nn_sequence.py#L180), [nn_sequence.py:312](../../python/needle/nn/nn_sequence.py#L312))
  and `MakeTensorTuple` (used by `stack`, [ops_mathematic.py:592-594](../../python/needle/ops/ops_mathematic.py#L592-L594)).
  `FusedAddScalars` is a third `TensorTupleOp` that no code uses.
- A `MakeTensorTuple` node holds the buffers of its inputs, so if it is not released, those buffers stay alive even when
  their `Tensor` nodes are released.
- Memory held by `TensorTuple` nodes after forward. Probe
  [agent-workspace/probe_tensortuple_memory.py](../../agent-workspace/probe_tensortuple_memory.py):
  ```
  RNN : non-leaf buffers 62464 bytes, held by TensorTuple nodes 8192 bytes (13%), ops ['MakeTensorTuple', 'Split']
  LSTM: non-leaf buffers 409088 bytes, held by TensorTuple nodes 50176 bytes (12%), ops ['MakeTensorTuple', 'Split']
  TransformerLayer: non-leaf buffers 335744 bytes, held by TensorTuple nodes 0 bytes (0%), ops []
  ```

## The guard

- Added by Han in commit `9806253` ("meta", 2026-11-03) together with
  [tests/design/test_metadata.py](../../tests/design/test_metadata.py). The commit message gives no reason.
- What it does: it refuses to release a node without cached metadata. For a `Tensor`, `_shape` is `None` only when the
  tensor was never computed (`LAZY_MODE`): both places that set data on a non-leaf call `_update_metadata()`
  ([autograd.py:328-331](../../python/needle/autograd.py#L328-L331), [autograd.py:382-389](../../python/needle/autograd.py#L382-L389)),
  and then `cached_data` is already `None`. For a `TensorTuple`, `_update_metadata()` never runs
  (`isinstance(self, Tensor)` at lines 330 and 388) and `TensorTuple` has no `shape`/`dtype`/`device`
  ([autograd.py:537-556](../../python/needle/autograd.py#L537-L556) are `Tensor` properties), so the guard always fires.
- Nothing depends on it: no test checks its message; the only library caller, `_release_cast_input()`, releases only a
  `Tensor` whose `cached_data` is not `None` ([ops_mathematic.py:64-72](../../python/needle/ops/ops_mathematic.py#L64-L72)).
  The full suite passes without it (copy of the repository, guard removed): `828 passed in 33.77s`.
- After removal, releasing a never-computed `Tensor` is a no-op.

## Options considered

- **(a) Release `TensorTuple` nodes (chosen).**
- (b) Skip them: no code, but `RNN`/`LSTM` keep 12–13% of the segment memory alive.

## Evidence that the release works

Probe [agent-workspace/probe_tensortuple_release.py](../../agent-workspace/probe_tensortuple_release.py) releases
every non-leaf node of a 2-layer `LSTM` (guard bypassed with `cached_data = None`) and runs backward:
```
TensorTuples released: 24 | grads identical: True
```

## Comparison with PyTorch

PyTorch has no tuple node type: `torch.split` returns a Python tuple of ordinary tensors that share one backward
node (`tuple Tensor SplitBackward0`). Its checkpoint pack hook sees each saved tensor
(`torch/utils/checkpoint.py` L1144-1152), so ops with several outputs need no special case. In needle the tuple is a
graph node of its own and must be recorded and released like any other node.

## Justification

1. `RNN`/`LSTM` segments lose none of their saving (12–13% otherwise kept).
2. About the same code as skipping: the guard goes away, the hook moves to `_init`, and the output search tests for
   `Value` instead of `Tensor`.
3. The guard has no purpose that any code or test relies on.

## Consequences

- The guard is Han's code; the user may tell Han about its removal.
- `LAZY_MODE` behaviour of `checkpoint()` is still an open decision ([00-plan.md](00-plan.md) §6).
