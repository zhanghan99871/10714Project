"""Reference NDArray backend implemented with numpy.

Supports the same element types as the native backends (``__dtypes__``) with the
same semantics: float16 and bfloat16 are *storage* types. Every operation widens
its inputs to float32, computes in float32 and rounds the result once (numpy's
float32 -> float16 and ml_dtypes' float32 -> bfloat16 conversions round to
nearest even, like the native backends). Matmul and sum reductions accumulate in
float32. bfloat16 arrays are ml_dtypes.bfloat16 arrays; ml_dtypes is imported
only when a bfloat16 array is created.

Like the native backends, floating point exceptions follow IEEE semantics silently:
overflow gives inf, 0/0 gives NaN, and so on, without numpy RuntimeWarnings. Under
mixed precision, float16 overflow is expected and is detected by loss scaling.
"""
import functools

import numpy as np


__device_name__ = "numpy"
__dtypes__ = ("float32", "float16", "bfloat16")

_COMPUTE = np.float32  # dtype all arithmetic is carried out in


def _np_dtype(name):
    """numpy dtype object for a dtype name (bfloat16 via ml_dtypes)."""
    if name == "bfloat16":
        try:
            import ml_dtypes
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise ImportError(
                "bfloat16 arrays need the ml_dtypes package: pip install ml_dtypes"
            ) from e
        return np.dtype(ml_dtypes.bfloat16)
    return np.dtype(name)


class Array:
    def __init__(self, size, dtype="float32"):
        if dtype not in __dtypes__:
            raise ValueError(f"unsupported dtype: {dtype}")
        self.array = np.empty(size, dtype=_np_dtype(dtype))

    @property
    def size(self):
        return self.array.size

    @property
    def dtype(self):
        return self.array.dtype.name

    @property
    def itemsize(self):
        return self.array.itemsize


def _ieee(fn):
    """Run a backend function with numpy floating point warnings silenced."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with np.errstate(all="ignore"):
            return fn(*args, **kwargs)

    return wrapper


def _f32(a):
    """Contents of an Array as float32 (no copy when already float32)."""
    return a.array.astype(_COMPUTE, copy=False)


def _check_same_dtype(*arrays):
    dtypes = {a.dtype for a in arrays}
    if len(dtypes) > 1:
        raise ValueError(f"dtype mismatch: {' vs '.join(sorted(dtypes))}")


def to_numpy(a, shape, strides, offset):
    # a strided *view* of the buffer: compact()/ewise_setitem rely on this
    return np.lib.stride_tricks.as_strided(
        a.array[offset:], shape, tuple([s * a.itemsize for s in strides])
    )


def from_numpy(a, out):
    if out.dtype == "bfloat16" and a.dtype == np.uint16:
        a = a.view(out.array.dtype)  # bfloat16 bit patterns, as the native backends take them
    if a.dtype != out.array.dtype:
        raise ValueError(f"from_numpy: expected {out.dtype} array, got {a.dtype}")
    if a.size != out.size:
        raise ValueError("from_numpy: size mismatch")
    out.array[:] = a.flatten()


@_ieee
def fill(out, val):
    # go through float32 like the native backends (their scalar argument is a C float)
    out.array.fill(np.float32(val))


@_ieee
def cast(a, out):
    if a.size != out.size:
        raise ValueError("cast: size mismatch")
    # through float32 (exact for every source dtype), then one rounding to the target,
    # exactly like the native backends' Load / Store
    out.array[:] = _f32(a)


def compact(a, out, shape, strides, offset):
    _check_same_dtype(a, out)
    out.array[:] = to_numpy(a, shape, strides, offset).flatten()


def ewise_setitem(a, out, shape, strides, offset):
    _check_same_dtype(a, out)
    to_numpy(out, shape, strides, offset)[:] = a.array.reshape(shape)


@_ieee
def scalar_setitem(size, val, out, shape, strides, offset):
    to_numpy(out, shape, strides, offset)[:] = np.float32(val)


@_ieee
def _ewise(fn, a, b, out):
    _check_same_dtype(a, b, out)
    out.array[:] = fn(_f32(a), _f32(b))


@_ieee
def _scalar(fn, a, val, out):
    _check_same_dtype(a, out)
    # scalars are float32, like the float argument of the native kernels
    out.array[:] = fn(_f32(a), np.float32(val))


@_ieee
def _unary(fn, a, out):
    _check_same_dtype(a, out)
    out.array[:] = fn(_f32(a))


def ewise_add(a, b, out):
    _ewise(np.add, a, b, out)


def scalar_add(a, val, out):
    _scalar(np.add, a, val, out)


def ewise_mul(a, b, out):
    _ewise(np.multiply, a, b, out)


def scalar_mul(a, val, out):
    _scalar(np.multiply, a, val, out)


def ewise_div(a, b, out):
    _ewise(np.divide, a, b, out)


def scalar_div(a, val, out):
    _scalar(np.divide, a, val, out)


def ewise_power(a, b, out):
    _ewise(np.power, a, b, out)


def scalar_power(a, val, out):
    _scalar(np.power, a, val, out)


def ewise_maximum(a, b, out):
    _ewise(np.maximum, a, b, out)


def scalar_maximum(a, val, out):
    _scalar(np.maximum, a, val, out)


def ewise_eq(a, b, out):
    _ewise(np.equal, a, b, out)


def scalar_eq(a, val, out):
    _scalar(np.equal, a, val, out)


def ewise_ge(a, b, out):
    _ewise(np.greater_equal, a, b, out)


def scalar_ge(a, val, out):
    _scalar(np.greater_equal, a, val, out)


def ewise_log(a, out):
    _unary(np.log, a, out)


def ewise_exp(a, out):
    _unary(np.exp, a, out)


def ewise_tanh(a, out):
    _unary(np.tanh, a, out)


@_ieee
def matmul(a, b, out, m, n, p):
    _check_same_dtype(a, b, out)
    out.array[:] = (_f32(a).reshape(m, n) @ _f32(b).reshape(n, p)).reshape(-1)


@_ieee
def matmul_batched(a, b, out, batch, m, n, p, a_batch_stride, b_batch_stride):
    """out[z] = a[z] @ b[z]; a batch stride of 0 means one matrix shared by all z."""
    _check_same_dtype(a, b, out)
    A = _f32(a).reshape(-1 if a_batch_stride else 1, m, n)
    B = _f32(b).reshape(-1 if b_batch_stride else 1, n, p)
    out.array[:] = np.broadcast_to(np.matmul(A, B), (batch, m, p)).reshape(-1)


@_ieee
def reduce_max(a, out, reduce_size):
    _check_same_dtype(a, out)
    out.array[:] = _f32(a).reshape(-1, reduce_size).max(axis=1)


@_ieee
def reduce_sum(a, out, reduce_size):
    _check_same_dtype(a, out)
    out.array[:] = a.array[:].reshape(-1, reduce_size).sum(axis=1, dtype=_COMPUTE)
