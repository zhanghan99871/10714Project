"""Automatic mixed precision.

    with ndl.autocast():                 # float16 by default
        logits = model(x)
        loss = loss_fn(logits, y)
    loss.backward()                      # inside or outside the region: same result

Inside an autocast region every op casts its tensor inputs according to its
``autocast_policy`` (see needle.autograd):

* ``"lower"``   -- MatMul, Conv: run in the autocast dtype (float16). These are the
  ops that dominate time and activation memory.
* ``"float32"`` -- Summation, LogSumExp, LogSoftmax, Exp, Log, EWisePow,
  PowerScalar: overflow-prone or precision-sensitive, run in float32. Layers built
  from them (softmax, LayerNorm/BatchNorm statistics, SoftmaxLoss) therefore
  compute in float32 without changes to the modules.
* ``"promote"`` -- every other op: inputs of different dtypes are cast to the
  widest one (e.g. a float16 matmul result plus a float32 bias gives float32);
  inputs of a single dtype are left alone.

Parameters stay float32 ("master weights"); their float16 copies are created by
differentiable casts, so gradients arrive in float32. Within one region each
parameter is cast once and reused (``cache_enabled``); the cache is dropped when
the outermost region exits.

Autocast is ignored in LAZY_MODE, and never applies while gradients are
computed: backward ops run in the dtypes chosen by the forward pass.
"""
import contextlib

from . import autograd
from .autograd import canonical_dtype

__all__ = ["autocast", "is_autocast_enabled", "get_autocast_dtype"]

_SUPPORTED = ("float16", "float32")


class autocast(contextlib.ContextDecorator):
    """Context manager / decorator enabling (or disabling) automatic mixed precision.

    Args:
        enabled: False suspends autocast inside an enclosing region.
        dtype: the low precision dtype used by "lower" ops ("float16").
        cache_enabled: reuse one cast per parameter within the region.
        release_inputs: let casts inserted by autocast release their (non-leaf)
            inputs, as an explicit ops.cast does by default. Off by default: an
            activation feeding a float16 matmul is often used again (e.g. by a
            residual connection), and a released input would then be recomputed.
    """

    def __init__(self, enabled=True, dtype="float16", cache_enabled=True, release_inputs=False):
        dtype = canonical_dtype(dtype)
        if dtype not in _SUPPORTED:
            raise TypeError(f"autocast dtype must be one of {_SUPPORTED}, got {dtype}")
        self.enabled = bool(enabled)
        self.dtype = dtype
        self.cache_enabled = bool(cache_enabled)
        self.release_inputs = bool(release_inputs)
        self._saved = []  # stack, so one instance can be re-entered (decorator, recursion)

    def __enter__(self):
        state = autograd._AUTOCAST
        self._saved.append(
            (state.enabled, state.dtype, state.cache_enabled, state.release_inputs, state.cache)
        )
        if self.enabled and not state.enabled:
            state.cache = {}  # entering an autocast region from outside: fresh cache
        state.enabled = self.enabled
        state.dtype = self.dtype
        state.cache_enabled = self.cache_enabled
        state.release_inputs = self.release_inputs
        return self

    def __exit__(self, *exc):
        state = autograd._AUTOCAST
        enabled, dtype, cache_enabled, release_inputs, cache = self._saved.pop()
        state.enabled = enabled
        state.dtype = dtype
        state.cache_enabled = cache_enabled
        state.release_inputs = release_inputs
        # leaving the outermost region drops the cached parameter casts
        state.cache = cache if enabled else {}
        return False


def is_autocast_enabled() -> bool:
    return autograd._AUTOCAST.enabled


def get_autocast_dtype() -> str:
    return autograd._AUTOCAST.dtype
