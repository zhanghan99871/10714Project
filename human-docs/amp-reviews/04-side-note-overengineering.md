# AMP side note 04: code for cases that do not occur

Date: 2026-11-15. Branch: `gradient-checkpointing`, commit `33bdf5f`. The line numbers refer to this commit. The AMP code is the same as on `main` (`83ae951`), but some lines in `autograd.py` moved because of the checkpoint code.

## Summary

- This file is a side note. It describes no defect: all four parts work.
- Four parts of the AMP code handle cases that the normal code paths do not cause. Each part adds code but has no effect in normal training.

| # | Code | Case that it handles | Occurs in normal use |
|---|---|---|---|
| 1 | `canonical_dtype` | a dtype that is not a string | no |
| 2 | adjoint dtype cast in the backward loop | a custom op with a wrong gradient | no |
| 3 | `astype` fallback | a backend without `cast` | no (all backends have `cast`) |
| 4 | `_AutocastDisabled` | `backward()` inside an autocast region | no |

## Finding 1: `canonical_dtype` accepts dtypes that are not strings

[autograd.py:24-36](../../python/needle/autograd.py#L24-L36):

```python
def canonical_dtype(dtype) -> Optional[str]:
    if dtype is None:
        return None
    if isinstance(dtype, str) and dtype == "bfloat16":
        return "bfloat16"
    return numpy.dtype(dtype).name
```

The function accepts strings, numpy dtypes (`numpy.dtype("float16")`) and numpy scalar types (`numpy.float16`). It changes each of them to one name, for example `"float16"`.

**Why it is not necessary.** The needle code uses dtype strings. The simplest design accepts only strings and refuses all other dtypes.

**Status.** It works. We decided to keep it.

## Finding 2: the backward loop casts adjoints that have the wrong dtype

[autograd.py:789-798](../../python/needle/autograd.py#L789-L798):

```python
# Invariant for mixed precision: a node's adjoint has the node's dtype
# (so float32 master weights always get float32 grads and partial
# adjoints of one node can be summed). Ops' gradients normally ensure
# this already; cast any stray contribution instead of failing later.
if (
    isinstance(prior, Tensor)
    and isinstance(prior_grad, Tensor)
    and prior_grad.dtype != prior.dtype
):
    prior_grad = needle.ops.cast(prior_grad, prior.dtype, release_input=False)
```

### Why it is not necessary

No built-in gradient gives an adjoint with the wrong dtype:

- A gradient op gets its dtype from `out_grad` or from `node.inputs`. For all ops except `Cast`, these tensors have the same dtype, because `_check_input_dtypes` refuses mixed inputs and `promote` casts them first.
- `Cast.gradient` casts the adjoint back to the input dtype ([ops_mathematic.py:61](../../python/needle/ops/ops_mathematic.py#L61)).
- In the `gradient` bodies of `ops_mathematic.py` and `ops_logarithmic.py`, the only new tensor is the ReLU mask. The mask uses the dtype of the input (`dtype=a.dtype`).

The only test for this code uses a custom op whose gradient is wrong on purpose ([test_cast_autograd.py:138-150](../../tests/fp16test/test_cast_autograd.py#L138-L150)):

```python
class Leaky(ndl.autograd.TensorOp):  # gradient wrongly returned in float32
    ...
    def gradient(self, out_grad, node):
        return ops.cast(out_grad, "float32") * 2.0
```

A wrong gradient is a bug in that op. If the loop does not cast, the next add of adjoints with mixed dtypes fails, and the error shows the bug.

### The commit message is not correct

Commit `aeff073` ("Add a differentiable Cast op and dtype-aware autograd") gives this reason: "so float16 and float32 contributions to a shared parameter can be summed." The test for that case, `test_shared_parameter_used_in_both_precisions`, does not need this code ([test_cast_autograd.py:119](../../tests/fp16test/test_cast_autograd.py#L119)):

```python
y = ops.summation(W * 3.0) + ops.summation(W.half() * 2.0).float()
```

The `W * 3.0` path gives a `float32` adjoint to `W`. The `W.half()` path goes through `Cast.gradient`, which also gives a `float32` adjoint. Thus, both adjoints are `float32` before this code runs.

### Comparison with PyTorch

PyTorch has a similar cast in `validate_outputs` (see [02-fp32-adjoint-sum.md](02-fp32-adjoint-sum.md#comparison-with-pytorch)).

## Finding 3: `NDArray.astype` has a fallback for a backend without `cast`

[ndarray.py:342-344](../../python/needle/backend_ndarray/ndarray.py#L342-L344):

```python
if not hasattr(self.device.mod, "cast"):
    # backend without a cast kernel (numpy backend): go through numpy
    return NDArray(self.numpy(), device=self.device, dtype=dtype)
```

**Why it is not necessary.** The comment says that the numpy backend has no cast kernel. This is not true. All three backends have `cast`:

| Backend | Location |
|---|---|
| numpy | [ndarray_backend_numpy.py:104](../../python/needle/backend_ndarray/ndarray_backend_numpy.py#L104) |
| CPU | [ndarray_backend_cpu.cc:658](../../src/ndarray_backend_cpu.cc#L658) |
| CUDA | [ndarray_backend_cuda.cu:825](../../src/ndarray_backend_cuda.cu#L825) |

Thus, this branch never runs. It is dead code.

## Finding 4: `_AutocastDisabled` turns off autocast in the backward pass

[autograd.py:234-243](../../python/needle/autograd.py#L234-L243):

```python
class _AutocastDisabled:
    """Internal: suspend autocast (backward always runs in the forward dtypes)."""

    def __enter__(self):
        self.prev = _AUTOCAST.enabled
        _AUTOCAST.enabled = False

    def __exit__(self, *exc):
        _AUTOCAST.enabled = self.prev
        return False
```

The backward pass uses it ([autograd.py:743](../../python/needle/autograd.py#L743)):

```python
with _AutocastDisabled(), set_grad_enabled(create_graph):
```

### Why it is not necessary

1. **In normal training, it does nothing.** The training loop calls `backward()` after the autocast region closes ([simple_ml.py:230-234](../../apps/simple_ml.py#L230-L234)):

   ```python
   with _grad_context(opt), _amp_context(amp):
       output = model(X)
       loss = loss_fn(output, y)
   if opt is not None:
       _optimizer_step(loss, opt, scaler)   # backward() is here
   ```

   The PTB loop has the same structure ([simple_ml.py:357-361](../../apps/simple_ml.py#L357-L361)). Thus, autocast is already off when the backward pass starts.
2. **It has an effect only when `backward()` runs inside an autocast region.** Training code does not do this.
3. **A public tool already does the same thing.** `amp.autocast(enabled=False)` also turns off autocast and restores the old state on exit ([amp.py:85-112](../../python/needle/amp.py#L85-L112)).

### Gradient checkpointing does not need it

The checkpoint recompute calls `op.compute` directly on the recorded graph ([autograd.py:339-341](../../python/needle/autograd.py#L339-L341)):

```python
data = self.op.compute(
    *[x.realize_cached_data() for x in self.inputs]
)
```

The autocast hook is only in `Tensor.make_from_op` ([autograd.py:510-511](../../python/needle/autograd.py#L510-L511)). Thus, the recompute does not use the autocast state.

## Evidence

| Source | What it shows |
|---|---|
| needle [autograd.py:24-36](../../python/needle/autograd.py#L24-L36) | finding 1: `canonical_dtype` |
| needle [autograd.py:789-798](../../python/needle/autograd.py#L789-L798), [test_cast_autograd.py:119](../../tests/fp16test/test_cast_autograd.py#L119), [test_cast_autograd.py:138-150](../../tests/fp16test/test_cast_autograd.py#L138-L150), commit `aeff073` | finding 2: the cast, the two tests and the commit message |
| needle [ndarray.py:342-344](../../python/needle/backend_ndarray/ndarray.py#L342-L344) and the three backends | finding 3: the fallback and the `cast` kernels |
| needle [autograd.py:234-243](../../python/needle/autograd.py#L234-L243), [autograd.py:743](../../python/needle/autograd.py#L743), [simple_ml.py:230-234](../../apps/simple_ml.py#L230-L234), [simple_ml.py:357-361](../../apps/simple_ml.py#L357-L361) | finding 4: where autocast is on and off |
