# AMP side note 03: `Stack` with mixed dtypes, and `float16` overflow

Date: 2026-11-15. Branch: `gradient-checkpointing`, commit `33bdf5f`. The line numbers refer to this commit. The AMP code is the same as on `main` (`83ae951`), but some lines in `autograd.py` moved because of the checkpoint code.

## Summary

- This file is a side note. It describes no defect.
- **`Stack` with mixed dtypes:** `stack()` (the helper function) accepts inputs with mixed dtypes under autocast. `Stack(...)` (the op class) raises a `TypeError` for them, because autocast does not cast the tensors in a `TensorTuple`. At this commit, only `stack()` calls `Stack(`.
- **`float16` overflow:** a `float16` result larger than 65504 becomes inf. This is a property of `float16`. It is not a defect in the code. PyTorch has the same behavior. `bfloat16` does not have this problem.

## Finding 1: `Stack(...)` and `stack()` with mixed dtypes

### Behavior

- `stack()` accepts inputs with mixed dtypes. Under autocast, it casts them to the widest dtype.
- `Stack(...)` raises a `TypeError` if its inputs have mixed dtypes.

### Why

Autocast `promote` looks only at the inputs that are a `Tensor` ([autograd.py:126-145](../../python/needle/autograd.py#L126-L145)):

```python
dtypes = {x.dtype for x in inputs if isinstance(x, Tensor)}
if not dtypes:
    return inputs
```

The input of `Stack` is one `TensorTuple`, not a `Tensor`. Thus, autocast does not cast the tensors in it. The `stack()` helper casts the tensors before it makes the tuple ([ops_mathematic.py:592-594](../../python/needle/ops/ops_mathematic.py#L592-L594)):

```python
def stack(args, axis):
    # under autocast, tensors of mixed dtypes are promoted to the widest one
    return Stack(axis)(make_tuple(*autocast_promote(args)))
```

If code calls `Stack(axis)(make_tuple(a16, b32))` directly, `Stack.compute` raises the error ([ops_mathematic.py:559-564](../../python/needle/ops/ops_mathematic.py#L559-L564)):

```python
dtypes = {str(a.dtype) for a in args}
if len(dtypes) > 1:
    raise TypeError(
        f"stack needs tensors of a single dtype, got {sorted(dtypes)}; "
        "use .astype() to convert"
    )
```

`autocast_promote` does nothing outside autocast. Thus, outside autocast, `stack()` also raises this error for mixed dtypes.

### Callers

At this commit, the only call to `Stack(` in `python/`, `apps/` and `tests/` is the call in `stack()` ([ops_mathematic.py:594](../../python/needle/ops/ops_mathematic.py#L594)).

## Finding 2: `float16` overflow

**This is a property of `float16`. It is not a defect in the code.**

### Behavior

The largest `float16` value is 65504. A larger result becomes inf.

Ops such as `MulScalar`, `DivScalar` and `EWiseDiv` use the `promote` policy. If all their inputs are `float16`, `promote` does not cast, so the op runs in `float16`. Thus, a result larger than 65504 becomes inf.

### Effect on training

- **Backward pass.** `GradScaler` finds inf or NaN in the gradients. It then skips the optimizer step and halves the scale.
- **Forward pass.** If the forward pass overflows, a smaller scale does not help, because the forward result does not use the scale. `GradScaler` then skips every step. PyTorch has the same limit.
- **`bfloat16`.** `bfloat16` has the same exponent range as `float32`. Thus, this overflow does not occur with `bfloat16`.

### Comparison with PyTorch

PyTorch has the same behavior. `mul` and `div` are not in the PyTorch `float32` list (`AT_FORALL_FP32` in [`aten/src/ATen/autocast_mode.h`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/autocast_mode.h)). That list contains the ops that overflow easily, for example `exp`, `pow` and `logsumexp`. Needle also runs `Exp`, `EWisePow`, `PowerScalar` and `LogSumExp` in `float32`.

## Evidence

| Source | What it shows |
|---|---|
| needle [autograd.py:126-145](../../python/needle/autograd.py#L126-L145) | autocast looks only at `Tensor` inputs |
| needle [ops_mathematic.py:592-594](../../python/needle/ops/ops_mathematic.py#L592-L594) | `stack()` promotes before it makes the tuple |
| needle [ops_mathematic.py:559-564](../../python/needle/ops/ops_mathematic.py#L559-L564) | `Stack.compute` rejects mixed dtypes |
| PyTorch [`aten/src/ATen/autocast_mode.h`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/autocast_mode.h) | the `float32` list of autocast (`AT_FORALL_FP32`) |
