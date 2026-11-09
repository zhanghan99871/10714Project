"""ops.Cast releases its (non-leaf) input by default, so a cast does not keep the
full-precision copy of an activation alive next to the low-precision one."""
import gc
import sys
import weakref

sys.path.append("./python")
import numpy as np
import pytest

import needle as ndl
from needle import autograd, ops

_DEVICES = [
    ndl.cpu_numpy(),
    ndl.cpu(),
    pytest.param(ndl.cuda(), marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU")),
]
_IDS = ["numpy", "cpu", "cuda"]
f16, f32 = np.float16, np.float32


def T(x, device, requires_grad=True, dtype=None):
    return ndl.Tensor(x, device=device, requires_grad=requires_grad, dtype=dtype)


class Counting(autograd.TensorOp):
    """y = 2 * x that counts how often its forward runs (to observe recomputation)."""

    calls = 0

    def compute(self, a):
        Counting.calls += 1
        return a * 2.0

    def gradient(self, out_grad, node):
        return out_grad * 2.0


class CountingTanh(autograd.TensorOp):
    """tanh whose gradient reads its own output (like ops.Tanh), counting forwards."""

    calls = 0

    def compute(self, a):
        CountingTanh.calls += 1
        return a.tanh()

    def gradient(self, out_grad, node):
        return out_grad * (-(node * node) + 1)


def buffer_ref(t):
    """Weak reference to a tensor's data buffer: dead once nothing holds the memory."""
    return weakref.ref(t.cached_data)


# --------------------------------------------------------------------------- what is released


@pytest.mark.parametrize("cast_fn", ["ops.cast", "astype", "half", "Cast op"])
@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_non_leaf_input_released_by_default(cast_fn, device):
    x = T(np.random.randn(64, 32).astype(f32), device)
    y = x * 2.0  # float32 graph intermediate
    ref = buffer_ref(y)
    expected = y.numpy().copy()
    h = {
        "ops.cast": lambda: ops.cast(y, "float16"),
        "astype": lambda: y.astype("float16"),
        "half": lambda: y.half(),
        "Cast op": lambda: ops.Cast("float16")(y),
    }[cast_fn]()
    assert h.dtype == "float16" and h.cached_data is not None
    assert y.cached_data is None
    gc.collect()
    assert ref() is None, "float32 buffer should be freed"
    # metadata survives, and the value can still be recomputed exactly
    assert y.shape == (64, 32) and y.dtype == "float32" and y.device == device
    assert y.cached_data is None
    np.testing.assert_array_equal(y.numpy(), expected)


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_release_input_false_keeps_data(device):
    x = T(np.ones((4, 4), f32), device)
    for fn in (
        lambda y: ops.cast(y, "float16", release_input=False),
        lambda y: y.astype("float16", release_input=False),
        lambda y: y.half(release_input=False),
        lambda y: ops.Cast("float16", release_input=False)(y),
    ):
        y = x * 2.0
        fn(y)
        assert y.cached_data is not None


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_leaves_are_never_released(device):
    w = T(np.ones((3, 3), f32), device)  # parameter / master weight
    p = ndl.nn.Parameter(np.ones((3, 3), f32), device=device)
    for leaf in (w, p):
        leaf.half()
        assert leaf.cached_data is not None
    # inference: everything is detached (a leaf), so nothing is released either
    a = T(np.ones((3, 3), f32), device, requires_grad=False)
    y = a * 2.0
    assert y.is_leaf()
    y.half()
    assert y.cached_data is not None
    # a cast to the same dtype is the identity and touches nothing
    z = w * 1.0
    assert z.astype("float32") is z and z.cached_data is not None


def test_lazy_mode_does_not_release():
    autograd.LAZY_MODE = True
    try:
        x = T(np.full(3, 1.5, f32), ndl.cpu())
        y = x * 2.0
        h = y.half()
        assert h.cached_data is None  # nothing computed yet, nothing released
        np.testing.assert_array_equal(h.numpy(), np.full(3, 3.0, f16))
        assert y.cached_data is not None
    finally:
        autograd.LAZY_MODE = False


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_float16_to_float32_releases_too(device):
    x = T(np.ones((8, 8), f16), device)
    y = x * 3.0  # float16 intermediate
    ref = buffer_ref(y)
    out = y.float()
    gc.collect()
    assert out.dtype == "float32" and ref() is None and y.cached_data is None


# --------------------------------------------------------------------------- correctness


def _model(device, release):
    """fp32 -> fp16 compute -> fp32 loss, with several cast points and a tanh whose
    gradient needs its released output."""
    np.random.seed(0)
    X = T(np.random.randn(6, 5).astype(f32), device, requires_grad=False)
    W1 = T((np.random.randn(5, 4) * 0.5).astype(f32), device)
    W2 = T((np.random.randn(4, 3) * 0.5).astype(f32), device)
    a = X @ W1  # float32 intermediate
    h = ops.tanh(a.half(release_input=release))  # float16, tanh output released below
    t = h.float(release_input=release)
    u = (t.half(release_input=release) @ W2.half()).float(release_input=release)
    loss = ops.summation(ops.relu(u) * t.sum(axes=(1,)).reshape((6, 1)).broadcast_to((6, 3)))
    loss.backward()
    return loss.numpy(), W1.grad.numpy(), W2.grad.numpy()


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_gradients_identical_with_and_without_release(device):
    kept = _model(device, release=False)
    released = _model(device, release=True)
    for k, r in zip(kept, released):
        assert k.dtype == r.dtype
        np.testing.assert_array_equal(k, r)  # recomputation is bit-exact


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_cast_backward_needs_no_recompute(device):
    """Cast's gradient only reads the input's dtype: the released input is not rebuilt."""
    x = T(np.ones((4, 4), f32), device)
    Counting.calls = 0
    y = Counting()(x)
    h = y.half()
    assert Counting.calls == 1 and y.cached_data is None
    ops.summation(h.float() * 3.0).backward()
    assert Counting.calls == 1, "backward through the cast should not recompute its input"
    assert y.cached_data is None
    np.testing.assert_array_equal(x.grad.numpy(), np.full((4, 4), 6.0, f32))


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_recompute_when_a_gradient_needs_the_value(device):
    """tanh's gradient reads its output: it is recomputed once, and the result is exact."""
    np.random.seed(1)
    x = np.random.randn(5, 3).astype(f32)
    grads = {}
    for release in (False, True):
        X = T(x, device)
        CountingTanh.calls = 0
        y = CountingTanh()(X)
        ops.summation(y.half(release_input=release).float() * 2.0).backward()
        grads[release] = (X.grad.numpy(), CountingTanh.calls)
    assert grads[False][1] == 1 and grads[True][1] == 2  # one extra forward when released
    np.testing.assert_array_equal(grads[False][0], grads[True][0])


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_input_reused_after_cast_is_recomputed(device):
    x = T(np.arange(6, dtype=f32).reshape(2, 3), device)
    Counting.calls = 0
    y = Counting()(x)
    h = y.half()  # releases y
    z = y + 1.0  # y used again in forward: recomputed on demand
    assert Counting.calls == 2
    np.testing.assert_array_equal(z.numpy(), np.arange(6, dtype=f32).reshape(2, 3) * 2 + 1)
    (ops.summation(z) + ops.summation(h.float())).backward()
    np.testing.assert_array_equal(x.grad.numpy(), np.full((2, 3), 4.0, f32))


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_gradients_are_not_released(device):
    """Casts inside backward (Cast.gradient, dtype fix-ups) never release gradients."""
    W = T(np.full((3, 2), 0.5, f32), device)
    # X requires grad, so the adjoints built from it are graph intermediates
    # (non-leaves) that a release would actually free
    X = T(np.ones((4, 3), f16), device)
    W16 = W.half()
    out = X @ W16
    loss = ops.summation(out.float())
    loss.backward()
    assert W.grad.dtype == "float32" and W.grad.cached_data is not None
    # the float16 adjoints that Cast.gradient casts back are kept as well
    assert W16.grad.dtype == "float16" and W16.grad.cached_data is not None
    assert out.grad.cached_data is not None
    np.testing.assert_array_equal(W.grad.numpy(), np.full((3, 2), 4.0, f32))
    # a float32 seed for a float16 output is cast in backward(); it stays intact too
    a = T(np.ones(3, f32), device)
    y = a.half() * 2.0
    seed = T(np.ones(3, f32), device)
    y.backward(seed)
    assert seed.cached_data is not None and a.grad.cached_data is not None


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_view_input_frees_nothing_but_stays_correct(device):
    """A view (here: reshape of a compact tensor) shares its buffer, so releasing it
    frees no memory until the base is gone, but everything stays correct."""
    x = T(np.ones((4, 4), f32), device)
    base = x * 2.0
    view = ops.reshape(base, (16,))
    assert view.cached_data._handle is base.cached_data._handle  # same memory
    h = view.half()
    assert view.cached_data is None and base.cached_data is not None
    ops.summation(h.float()).backward()
    np.testing.assert_array_equal(x.grad.numpy(), np.full((4, 4), 2.0, f32))


def test_memory_returned_for_a_chain_of_casts():
    """fp32 -> fp16 -> fp32 -> fp16: every intermediate buffer except the last is freed."""
    dev = ndl.cpu()
    x = T(np.random.randn(256, 256).astype(f32), dev)
    a = x * 1.0
    refs = [buffer_ref(a)]
    b = a.half()
    refs.append(buffer_ref(b))
    c = b.float()
    refs.append(buffer_ref(c))
    d = c.half()
    gc.collect()
    assert [r() is None for r in refs] == [True, True, True]
    assert d.cached_data is not None
    ops.summation(d.float()).backward()
    np.testing.assert_array_equal(x.grad.numpy(), np.ones((256, 256), f32))
