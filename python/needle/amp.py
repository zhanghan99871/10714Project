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

Loss scaling (float16 only)
---------------------------
float16 gradients smaller than ~6e-8 flush to zero and lose precision below
~6e-5. GradScaler multiplies the loss by a large factor before backward so the
float16 gradients stay in range, divides the (float32) parameter gradients by
the same factor before the optimizer step, skips steps whose gradients
overflowed, and adapts the factor over time:

    scaler = ndl.amp.GradScaler()
    for x, y in data:
        opt.reset_grad()
        with ndl.autocast():
            loss = loss_fn(model(x), y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)          # optional: only needed before clipping
        opt.clip_grad_norm(1.0)
        scaler.step(opt)              # skipped if a gradient is inf / NaN
        scaler.update()
"""
import contextlib

import numpy

from . import autograd
from .autograd import canonical_dtype

__all__ = ["autocast", "is_autocast_enabled", "get_autocast_dtype", "GradScaler"]

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


def _all_finite(t) -> bool:
    """No inf / NaN in tensor t (one scalar transferred to the host)."""
    data = t.realize_cached_data()
    if hasattr(data, "all_finite"):  # needle NDArray
        return data.all_finite()
    return bool(numpy.isfinite(data).all())  # numpy array backend


class GradScaler:
    """Dynamic loss scaling for float16 mixed precision training.

    Args:
        init_scale: initial scale factor (2**16).
        growth_factor: multiply the scale by this after `growth_interval`
            consecutive steps without inf / NaN gradients (2.0).
        backoff_factor: multiply the scale by this when a step is skipped (0.5).
        growth_interval: number of clean steps between scale increases (2000).
        enabled: False makes every method a pass-through (scale() returns the
            loss, step() just calls optimizer.step()), so the same training loop
            serves float32 and mixed precision runs.

    Powers of two (the defaults) keep scaling and unscaling exact in float32.
    Not needed for bfloat16, whose exponent range equals float32's.

    Attributes:
        skipped_steps: total number of optimizer steps skipped because of
            inf / NaN gradients.
    """

    def __init__(
        self,
        init_scale: float = 2.0 ** 16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
    ):
        if not init_scale > 0:
            raise ValueError("init_scale must be positive")
        if not growth_factor > 1.0:
            raise ValueError("growth_factor must be > 1")
        if not 0.0 < backoff_factor < 1.0:
            raise ValueError("backoff_factor must be in (0, 1)")
        if int(growth_interval) < 1:
            raise ValueError("growth_interval must be >= 1")
        self._enabled = bool(enabled)
        self._scale = float(init_scale)
        self._growth_factor = float(growth_factor)
        self._backoff_factor = float(backoff_factor)
        self._growth_interval = int(growth_interval)
        self._growth_tracker = 0  # consecutive clean steps
        self._per_optimizer = {}  # id(opt) -> {"unscaled": bool, "found_inf": bool}
        self.skipped_steps = 0

    def is_enabled(self) -> bool:
        return self._enabled

    def get_scale(self) -> float:
        return self._scale if self._enabled else 1.0

    def scale(self, loss):
        """Return loss * scale (computed in float32) for backward()."""
        if not self._enabled:
            return loss
        if loss.dtype != "float32":
            # keep the user's loss tensor intact: it is often printed afterwards
            loss = loss.astype("float32", release_input=False)
        return loss * self._scale

    def _state(self, optimizer):
        return self._per_optimizer.setdefault(
            id(optimizer), {"unscaled": False, "found_inf": False}
        )

    def unscale_(self, optimizer) -> None:
        """Divide the optimizer's gradients by the scale (in place) and record
        whether any of them is inf / NaN. Call it before inspecting or clipping
        gradients; step() calls it otherwise. At most once per update()."""
        if not self._enabled:
            return
        state = self._state(optimizer)
        if state["unscaled"]:
            raise RuntimeError(
                "unscale_() has already been called on this optimizer since the last update()"
            )
        inv_scale = 1.0 / self._scale
        found_inf = False
        for p in optimizer.params:
            if p.grad is None:
                continue
            # detach: drop the backward graph the gradient may still reference
            p.grad = (p.grad * inv_scale).detach()
            if not found_inf and not _all_finite(p.grad):
                found_inf = True
        state["unscaled"] = True
        state["found_inf"] = found_inf

    def step(self, optimizer) -> bool:
        """Unscale (if not done yet) and call optimizer.step(), unless a gradient
        is inf / NaN, in which case the step is skipped. Returns whether the
        optimizer stepped."""
        if not self._enabled:
            optimizer.step()
            return True
        if not self._state(optimizer)["unscaled"]:
            self.unscale_(optimizer)
        if self._state(optimizer)["found_inf"]:
            self.skipped_steps += 1
            return False
        optimizer.step()
        return True

    def update(self, new_scale: float = None) -> None:
        """Adjust the scale for the next iteration: back off if any optimizer
        saw inf / NaN gradients since the last update, grow after
        `growth_interval` clean iterations. `new_scale` sets it explicitly."""
        if not self._enabled:
            return
        if new_scale is not None:
            if not new_scale > 0:
                raise ValueError("new_scale must be positive")
            self._scale = float(new_scale)
            self._growth_tracker = 0
        elif self._per_optimizer:
            if any(s["found_inf"] for s in self._per_optimizer.values()):
                self._scale *= self._backoff_factor
                self._growth_tracker = 0
            else:
                self._growth_tracker += 1
                if self._growth_tracker == self._growth_interval:
                    self._scale *= self._growth_factor
                    self._growth_tracker = 0
        self._per_optimizer = {}

    def state_dict(self) -> dict:
        return {
            "scale": self._scale,
            "growth_factor": self._growth_factor,
            "backoff_factor": self._backoff_factor,
            "growth_interval": self._growth_interval,
            "growth_tracker": self._growth_tracker,
            "skipped_steps": self.skipped_steps,
        }

    def load_state_dict(self, state: dict) -> None:
        self._scale = float(state["scale"])
        self._growth_factor = float(state["growth_factor"])
        self._backoff_factor = float(state["backoff_factor"])
        self._growth_interval = int(state["growth_interval"])
        self._growth_tracker = int(state["growth_tracker"])
        self.skipped_steps = int(state.get("skipped_steps", 0))
