# AMP review 02: `_sum_adjoints` adds half precision gradients in float32

Date: 2026-11-15. Branch: `gradient-checkpointing`, commit `33bdf5f`. The line numbers in needle refer to this commit. The AMP code is the same as on `main` (`83ae951`), but some lines in `autograd.py` moved because of the checkpoint code. The PyTorch source is at commit [`3886d18`](https://github.com/pytorch/pytorch/tree/3886d1829cbb149d10ee3bd38781fe012c57fabb) (main).

## Summary

- When a `float16` or `bfloat16` node has more than one adjoint, needle adds the adjoints in `float32`. It rounds the total one time.
- PyTorch does not do this. PyTorch adds the adjoints in their own dtype.
- Thus, this behavior is an addition in needle. It is not a copy of PyTorch behavior.
- It costs memory and time. For a node with `n` adjoints, it makes `n` `float32` copies and runs `n + 1` cast kernels.

## What needle does

`_sum_adjoints` ([autograd.py:747-763](../../python/needle/autograd.py#L747-L763)):

```python
def _sum_adjoints(grads: List["Value"]) -> "Value":
    if (
        len(grads) > 1
        and isinstance(grads[0], Tensor)
        and grads[0].dtype in _HALF_TYPES
    ):
        total = sum_node_list(
            [needle.ops.cast(g, "float32", release_input=False) for g in grads]
        )
        return needle.ops.cast(total, grads[0].dtype, release_input=False)
    return sum_node_list(grads)
```

The backward loop calls this function one time for each node ([autograd.py:782](../../python/needle/autograd.py#L782)):

```python
each.grad = _sum_adjoints(node_to_output_grads_list[each])
```

For a half precision node with `n` adjoints, the function does these steps:

1. It casts each of the `n` adjoints to `float32`. This makes `n` new `float32` tensors.
2. It adds the `float32` tensors with `sum_node_list` (`reduce(add, ...)`, [autograd.py:850-855](../../python/needle/autograd.py#L850-L855)). This makes `n - 1` `float32` sums.
3. It casts the total back to the half dtype. The result is rounded one time.

Without this function, `sum_node_list` adds the half precision adjoints directly. The CUDA add kernel loads each value as `float`, adds, and stores the result in the half dtype. Thus, each partial sum is rounded to the half dtype.

## When the function has an effect

The function has an effect only on a half precision node that has two or more adjoints. Two examples:

- **A cached weight copy.** Autocast casts a parameter one time in each region and keeps the copy in a cache. If the model uses the copy many times (for example, an RNN weight at each time step), the `float16` copy node gets one adjoint for each use.
- **A `float16` activation that two ops use**, for example in a residual connection.

The function has no effect on the `float32` master weights. Their adjoints come through `Cast.gradient`, which gives `float32` ([ops_mathematic.py:61](../../python/needle/ops/ops_mathematic.py#L61)). Thus, they are already added in `float32`.

## Why the code does this

The docstring gives the reason: a half precision running sum loses precision quickly ([autograd.py:748-750](../../python/needle/autograd.py#L748-L750)). `bfloat16` has 8 significand bits. Thus, `256 + 1` gives `256` in `bfloat16`. For a sum of many small adjoints, the small values can stop having an effect on the total.

## Costs

- **Memory.** Step 1 makes `n` `float32` copies, and the list keeps all of them until the sum is complete. The original half precision adjoints also stay alive, because `node_to_output_grads_list` keeps them until `backward()` returns. Each `float32` copy is two times the size of the half precision adjoint.
- **Time.** The function adds `n + 1` cast kernels for each node that has this behavior.

## Comparison with PyTorch

**PyTorch adds adjoints in their own dtype.** `InputBuffer::accumulate` adds each new adjoint to the running sum with a normal add. It does not cast to `float32` ([input_buffer.cpp:157-171](https://github.com/pytorch/pytorch/blob/3886d1829cbb149d10ee3bd38781fe012c57fabb/torch/csrc/autograd/input_buffer.cpp#L157-L171)):

```cpp
if (at::GradMode::is_enabled()) {
  buffer[pos] = old_var + var;
} ...
  buffer[pos] = old_var.add_(var);
} else {
  buffer[pos] = old_var + var;
}
```

Thus, in PyTorch, each partial sum of `float16` adjoints is stored in `float16`. PyTorch also caches the weight copies in an autocast region. Thus, a `float16` weight copy that an RNN uses at each time step gets its adjoints added in `float16`.

**PyTorch casts each adjoint to the dtype of its tensor.** `validate_outputs` casts a gradient that has a different dtype ([engine.cpp:1052-1055](https://github.com/pytorch/pytorch/blob/3886d1829cbb149d10ee3bd38781fe012c57fabb/torch/csrc/autograd/engine.cpp#L1052-L1055)):

```cpp
if (metadata.grad_dtype().has_value()) {
  if (grad.scalar_type() != metadata.grad_dtype().value()) {
    grad = grad.to(metadata.grad_dtype().value());
```

The expected dtype is the dtype of the tensor, unless the user sets a different dtype on a leaf ([variable.cpp:968-977](https://github.com/pytorch/pytorch/blob/3886d1829cbb149d10ee3bd38781fe012c57fabb/torch/csrc/autograd/variable.cpp#L968-L977)):

```cpp
} else {
  return std::optional<at::ScalarType>(self.scalar_type());
}
```

| Item | needle | PyTorch |
|---|---|---|
| dtype of the sum of half precision adjoints | `float32`, rounded one time | the half dtype, rounded after each add |
| Extra memory for the sum | `n` `float32` copies | none |
| Adjoint dtype equals the tensor dtype | yes (cast in the backward loop) | yes (cast in `validate_outputs`) |

## Evidence

| Source | What it shows |
|---|---|
| needle [autograd.py:747-763](../../python/needle/autograd.py#L747-L763) | the sum in `float32` |
| needle [autograd.py:782](../../python/needle/autograd.py#L782) | the call for each node in backward |
| PyTorch [input_buffer.cpp:157-171](https://github.com/pytorch/pytorch/blob/3886d1829cbb149d10ee3bd38781fe012c57fabb/torch/csrc/autograd/input_buffer.cpp#L157-L171) | the sum in the dtype of the adjoints |
| PyTorch [engine.cpp:1052-1055](https://github.com/pytorch/pytorch/blob/3886d1829cbb149d10ee3bd38781fe012c57fabb/torch/csrc/autograd/engine.cpp#L1052-L1055) and [variable.cpp:968-977](https://github.com/pytorch/pytorch/blob/3886d1829cbb149d10ee3bd38781fe012c57fabb/torch/csrc/autograd/variable.cpp#L968-L977) | the cast of each adjoint to the dtype of its tensor |
