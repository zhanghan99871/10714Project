"""bfloat16 in the frontend: Tensor creation, Cast / astype, gradients, dtype rules."""
import gc
import sys
import weakref

sys.path.append("./python")
import numpy as np
import pytest

ml_dtypes = pytest.importorskip("ml_dtypes")
import needle as ndl
import needle.nn as nn
from needle import autograd, ops

bf16 = ml_dtypes.bfloat16
f16, f32 = np.float16, np.float32
_DEVICES = [
    ndl.cpu_numpy(),
    ndl.cpu(),
    pytest.param(ndl.cuda(), marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU")),
]
_IDS = ["numpy", "cpu", "cuda"]


def T(x, device, dtype=None, requires_grad=True):
    return ndl.Tensor(x, device=device, dtype=dtype, requires_grad=requires_grad)


def bits(x):
    return np.ascontiguousarray(x).view(np.uint16 if x.dtype.itemsize == 2 else np.uint32)


# --------------------------------------------------------------------------- creation


def test_canonical_dtype():
    for d in ("bfloat16", bf16, np.dtype(bf16)):
        assert autograd.canonical_dtype(d) == "bfloat16"
    assert autograd.canonical_dtype(np.float16) == "float16"
    assert autograd.numpy_dtype("bfloat16") == np.dtype(bf16)


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_tensor_creation(device):
    x = np.random.randn(3, 4).astype(f32)
    for t in (T(x, device, dtype="bfloat16"), T(x, device, dtype=bf16), T(x.astype(bf16), device)):
        assert t.dtype == "bfloat16" and t.numpy().dtype == np.dtype(bf16)
        np.testing.assert_array_equal(bits(t.numpy()), bits(x.astype(bf16)))
    t32 = T(x, device)
    tb = ndl.Tensor(t32, dtype="bfloat16")  # same device: converted on the device
    assert tb.dtype == "bfloat16" and tb.is_leaf()
    other = ndl.cpu() if device != ndl.cpu() else ndl.cpu_numpy()
    assert ndl.Tensor(tb, device=other).dtype == "bfloat16"
    for fn in (ndl.init.zeros, ndl.init.ones, ndl.init.rand, ndl.init.randn):
        assert fn(2, 3, device=device, dtype="bfloat16").dtype == "bfloat16"
    assert ndl.init.zeros_like(tb).dtype == "bfloat16"
    oh = ndl.init.one_hot(5, T(np.array([0, 4]), device), device=device, dtype="bfloat16")
    assert oh.dtype == "bfloat16"
    assert nn.Linear(4, 3, device=device, dtype="bfloat16").weight.dtype == "bfloat16"


# --------------------------------------------------------------------------- casts


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_cast_forward(device):
    x = np.array([1 / 3, 1e-30, 3e38, 70000.0, -0.1, 1.0 + 2 ** -8], f32)
    a = T(x, device)
    b = a.bfloat16()
    assert b.dtype == "bfloat16"
    np.testing.assert_array_equal(bits(b.numpy()), bits(x.astype(bf16)))
    assert np.isfinite(b.numpy().astype(f32)).all()  # float32 range: nothing overflows
    np.testing.assert_array_equal(b.float().numpy(), x.astype(bf16).astype(f32))
    # bfloat16 <-> float16 goes through float32 exactly once
    h = b.half()
    with np.errstate(over="ignore"):
        np.testing.assert_array_equal(bits(h.numpy()), bits(x.astype(bf16).astype(f32).astype(f16)))
    assert b.bfloat16() is b and b.astype(np.dtype(bf16)) is b


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_cast_releases_input_and_backprops_in_input_dtype(device):
    x = T(np.random.randn(64, 32).astype(f32), device)
    y = x * 2.0
    ref = weakref.ref(y.cached_data)
    b = y.bfloat16()
    gc.collect()
    assert ref() is None and y.cached_data is None  # released by default
    w = np.random.randn(64, 32).astype(bf16)
    ops.summation(b * T(w, device, requires_grad=False)).float().backward()
    assert x.grad.dtype == "float32"
    np.testing.assert_array_equal(x.grad.numpy(), 2 * w.astype(f32))
    z = T(np.ones(3, bf16), device)
    ops.summation(z.float() * 3.0).backward()
    assert z.grad.dtype == "bfloat16"
    np.testing.assert_array_equal(z.grad.numpy().astype(f32), np.full(3, 3.0, f32))


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_master_weight_pattern(device):
    np.random.seed(1)
    x = np.random.randn(8, 5).astype(bf16)
    w = (np.random.randn(5, 3) * 0.5).astype(f32)
    X, W = T(x, device, requires_grad=False), T(w, device)
    h = X @ W.bfloat16()
    assert h.dtype == "bfloat16"
    loss = ops.summation(ops.relu(h).float() ** 2) / 8
    loss.backward()
    assert W.grad.dtype == "float32"
    hh = x.astype(np.float64) @ w.astype(bf16).astype(np.float64)
    g = 2 * np.maximum(hh, 0) / 8
    np.testing.assert_allclose(W.grad.numpy(), x.astype(np.float64).T @ g, rtol=5e-2, atol=5e-2)


# --------------------------------------------------------------------------- gradients


def test_bf16_adjoints_accumulate_in_float32():
    """A bfloat16 node used 300 times: a bfloat16 running sum of its unit adjoints
    stops at 256 (256 + 1 rounds back to 256)."""
    W = T(np.ones((1,), f32), ndl.cpu())
    b = W.bfloat16()
    total = ops.summation(ops.stack([b * 1.0 for _ in range(300)], 0))
    total.backward()
    assert b.grad.dtype == "bfloat16" and b.grad.numpy().astype(f32)[0] == 300.0
    assert W.grad.numpy()[0] == 300.0


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_ops_preserve_bfloat16(device):
    np.random.seed(2)
    x = T(np.random.randn(2, 4, 4, 3).astype(bf16), device)
    w = T((np.random.randn(3, 3, 3, 2) * 0.3).astype(bf16), device)
    outs = {
        "stack": ops.stack([x, x * 2.0], 1),
        "split": ops.split(x, 3)[1],
        "dilate": ops.dilate(x, (1, 2), 1),
        "conv_stride2": ops.conv(x, w, stride=2, padding=1),
        "batched_matmul": ops.reshape(x, (2, 4, 12)) @ ops.reshape(x, (2, 12, 4)),
        "logsumexp": ops.logsumexp(x, axes=(3,)),
        "logsoftmax": ops.logsoftmax(ops.reshape(x, (8, 12))),
        "relu_tanh_exp": ops.exp(ops.tanh(ops.relu(x))),
        "rdiv_rsub": 1.0 / (ops.exp(x) + 1.0) - 2.0,
        "bcast": ops.broadcast_to(ops.summation(x, (0,)), (5, 4, 4, 3)),
        "pad_crop": ops.crop(ops.pad(x, ((0, 0), (1, 1), (0, 2), (0, 0))), ((0, 0), (1, 0), (0, 1), (0, 0))),
        "flip": ops.flip(x, (1, 2)),
    }
    for name, y in outs.items():
        assert y.dtype == "bfloat16", name
        x.grad = w.grad = None
        ops.summation(y).backward()
        assert x.grad.dtype == "bfloat16", name
        if w.grad is not None:
            assert w.grad.dtype == "bfloat16", name


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_bf16_conv_matches_float32(device):
    np.random.seed(3)
    x = np.random.randn(2, 6, 6, 3).astype(bf16)
    w = (np.random.randn(3, 3, 3, 4) * 0.3).astype(bf16)
    res = {}
    for dt in ("bfloat16", "float32"):
        X, W = T(x, device, dtype=dt), T(w, device, dtype=dt)
        y = ops.conv(X, W, stride=2, padding=1)
        ops.summation(y * y).backward()
        res[dt] = [a.numpy().astype(f32) for a in (y, X.grad, W.grad)]
    for got, want in zip(res["bfloat16"], res["float32"]):
        np.testing.assert_allclose(got, want, rtol=5e-2, atol=2e-1)


# --------------------------------------------------------------------------- dtype rules


def test_widest_dtype():
    w = autograd._widest_dtype
    assert w({"float16", "bfloat16"}) == "float32"
    assert w({"bfloat16", "float32"}) == "float32"
    assert w({"float16", "float32"}) == "float32"
    assert w({"float16", "bfloat16", "float32"}) == "float32"
    assert w({"bfloat16"}) == "bfloat16"
    assert w({"float16", "float64"}) == "float64"


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_mixed_dtypes_outside_autocast_raise(device):
    b = T(np.ones((2, 2), bf16), device)
    for other in (T(np.ones((2, 2), f16), device), T(np.ones((2, 2), f32), device)):
        for fn in (lambda: b + other, lambda: b @ other, lambda: ops.stack([b, other], 0)):
            with pytest.raises(TypeError, match="dtype"):
                fn()
    assert (b * 2.5 + 1).dtype == "bfloat16" and (2 - b).dtype == "bfloat16"


# --------------------------------------------------------------------------- interplay


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_no_grad_and_optimizers(device):
    np.random.seed(4)
    layer = nn.Linear(4, 3, device=device)
    x = T(np.random.randn(5, 4).astype(bf16), device, requires_grad=False)
    with ndl.no_grad():
        y = x @ layer.weight.bfloat16()
    assert not y.requires_grad and y.dtype == "bfloat16"
    for opt_cls in (ndl.optim.SGD, ndl.optim.Adam):
        opt = opt_cls(layer.parameters(), lr=0.1)
        before = layer.weight.numpy().copy()
        opt.reset_grad()
        h = x @ layer.weight.bfloat16() + ops.broadcast_to(ops.reshape(layer.bias.bfloat16(), (1, 3)), (5, 3))
        ops.summation(h.float() ** 2).backward()
        assert layer.weight.grad.dtype == "float32"
        opt.step()
        assert layer.weight.dtype == "float32" and not np.array_equal(layer.weight.numpy(), before)
