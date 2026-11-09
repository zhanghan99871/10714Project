"""ops.Cast / Tensor.astype and dtype handling in autograd (mixed precision, step 2)."""
import sys

sys.path.append("./python")
sys.path.append("./apps")
import numpy as np
import pytest

import needle as ndl
import needle.nn as nn
from needle import ops

_DEVICES = [
    ndl.cpu_numpy(),
    ndl.cpu(),
    pytest.param(ndl.cuda(), marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU")),
]
_IDS = ["numpy", "cpu", "cuda"]
f16, f32 = np.float16, np.float32


def T(x, device, dtype=None, requires_grad=True):
    return ndl.Tensor(x, device=device, dtype=dtype, requires_grad=requires_grad)


def bits(x):
    x = np.ascontiguousarray(x)
    return x.view(np.uint16 if x.dtype == f16 else np.uint32)


# --------------------------------------------------------------------------- forward


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_cast_forward(device):
    x = np.array([[1 / 3, -2.5e-8, 70000.0], [65504.0, -0.1, 1e-5]], f32)
    a = T(x, device)
    h = ops.cast(a, "float16")
    assert h.dtype == "float16" and h.device == device and h.shape == a.shape
    with np.errstate(over="ignore"):
        expected = x.astype(f16)
    np.testing.assert_array_equal(bits(h.numpy()), bits(expected))
    back = h.astype(np.float32)  # numpy dtype objects are accepted
    assert back.dtype == "float32"
    np.testing.assert_array_equal(back.numpy(), expected.astype(f32))
    assert a.half().dtype == "float16" and h.float().dtype == "float32"


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_cast_to_same_dtype_is_identity(device):
    a = T(np.ones((2, 2), f32), device)
    assert ops.cast(a, "float32") is a
    assert a.astype(np.dtype("float32")) is a
    assert a.float() is a
    h = a.half()
    assert h.half() is h


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_cast_does_not_alias(device):
    a = T(np.ones(3, f32), device)
    b = ops.Cast("float32")(a)  # explicit op on the same dtype makes a copy
    a.data = T(np.zeros(3, f32), device)
    np.testing.assert_array_equal(b.numpy(), np.ones(3, f32))


# --------------------------------------------------------------------------- backward


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_cast_gradient_is_identity_in_input_dtype(device):
    np.random.seed(0)
    x = np.random.randn(3, 4).astype(f32)
    w = np.random.randn(3, 4).astype(f16)
    a = T(x, device)
    y = ops.cast(a, "float16") * T(w, device, requires_grad=False)
    ops.summation(y).backward()
    assert a.grad.dtype == "float32"
    # d(cast(a) * w)/da = w (in float16), cast back to float32 exactly
    np.testing.assert_array_equal(a.grad.numpy(), w.astype(f32))

    b = T(x.astype(f16), device)
    ops.summation(ops.cast(b, "float32") ** 2).backward()
    assert b.grad.dtype == "float16"
    np.testing.assert_array_equal(
        bits(b.grad.numpy()), bits((2 * x.astype(f16).astype(f32)).astype(f16))
    )


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_master_weight_pattern(device):
    """float32 weights, float16 compute, float32 loss: grads land in float32."""
    np.random.seed(1)
    x = np.random.randn(8, 5).astype(f16)
    w = (np.random.randn(5, 3) * 0.5).astype(f32)
    b = np.random.randn(3).astype(f32)
    X = T(x, device, requires_grad=False)
    W, B = T(w, device), T(b, device)
    h = X @ W.half() + ops.broadcast_to(ops.reshape(B.half(), (1, 3)), (8, 3))
    assert h.dtype == "float16"
    loss = ops.summation(ops.relu(h).float() ** 2) / 8
    assert loss.dtype == "float32"
    loss.backward()
    assert W.grad.dtype == "float32" and B.grad.dtype == "float32"

    # reference in float64 from the same float16-rounded inputs
    w16, b16 = w.astype(f16).astype(np.float64), b.astype(f16).astype(np.float64)
    hh = x.astype(np.float64) @ w16 + b16
    g = 2 * np.maximum(hh, 0) * (hh > 0) / 8
    np.testing.assert_allclose(W.grad.numpy(), x.astype(np.float64).T @ g, rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(B.grad.numpy(), g.sum(0), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_shared_parameter_used_in_both_precisions(device):
    """Partial adjoints from a float32 and a float16 use are summed in float32."""
    w = np.array([1.5, -2.0, 0.25], f32)
    W = T(w, device)
    y = ops.summation(W * 3.0) + ops.summation(W.half() * 2.0).float()
    y.backward()
    assert W.grad.dtype == "float32"
    np.testing.assert_array_equal(W.grad.numpy(), np.full(3, 5.0, f32))


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_backward_out_grad_dtype(device):
    a = T(np.ones((2, 3), f32), device)
    y = a.half() * 2.0
    y.backward(T(np.full((2, 3), 0.5, f32), device))  # float32 seed for a float16 output
    assert a.grad.dtype == "float32"
    np.testing.assert_array_equal(a.grad.numpy(), np.full((2, 3), 1.0, f32))
    b = T(np.ones(2, f16), device)
    (b * 3.0).sum().backward()  # default seed takes the output dtype
    assert b.grad.dtype == "float16"
    np.testing.assert_array_equal(b.grad.numpy(), np.full(2, 3.0, f16))


def test_stray_gradient_dtype_is_corrected():
    """compute_gradient_of_variables casts an adjoint that has the wrong dtype."""

    class Leaky(ndl.autograd.TensorOp):  # gradient wrongly returned in float32
        def compute(self, a):
            return a * 2.0

        def gradient(self, out_grad, node):
            return ops.cast(out_grad, "float32") * 2.0

    a = T(np.ones(3, f16), ndl.cpu())
    Leaky()(a).sum().backward()
    assert a.grad.dtype == "float16"
    np.testing.assert_array_equal(a.grad.numpy(), np.full(3, 2.0, f16))


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_numerical_gradient_through_casts(device):
    """Composite function with a float16 middle section vs finite differences in float64."""
    np.random.seed(2)
    x = np.random.uniform(0.5, 1.5, (4, 3)).astype(f32)
    A = T(x, device)
    out = ops.summation(ops.exp(ops.tanh(A.half()) * 0.5).float() * A)
    out.backward()

    def f(z):
        z16 = z.astype(f16).astype(np.float64)
        return np.sum(np.exp(np.tanh(z16) * 0.5) * z)

    eps = 1e-3
    num = np.zeros_like(x, dtype=np.float64)
    for idx in np.ndindex(*x.shape):
        e = np.zeros_like(x, dtype=np.float64)
        e[idx] = eps
        # straight-through: differentiate the smooth function, rounding only the values
        z = x.astype(np.float64)
        num[idx] = (np.sum(np.exp(np.tanh(z + e) * 0.5) * (z + e))
                    - np.sum(np.exp(np.tanh(z - e) * 0.5) * (z - e))) / (2 * eps)
    np.testing.assert_allclose(A.grad.numpy(), num, rtol=1e-2, atol=1e-2)
    assert abs(out.numpy().item() - f(x)) < 1e-2


# --------------------------------------------------------------------------- dtype rules


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_mixed_dtype_ops_raise_clear_error(device):
    a = T(np.ones((2, 2), f16), device)
    b = T(np.ones((2, 2), f32), device)
    for fn in (lambda: a + b, lambda: a * b, lambda: a / b, lambda: a @ b, lambda: a ** b,
               lambda: ops.stack([a, b], 0)):
        with pytest.raises(TypeError, match="dtype"):
            fn()
    # fine after an explicit cast, and scalars never change the dtype
    assert (a + b.half()).dtype == "float16"
    assert (a.float() @ b).dtype == "float32"
    assert (a * 3.5 + 1).dtype == "float16" and (a - 2).dtype == "float16"


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_ops_preserve_float16(device):
    """Every op keeps float16 in forward and backward (Stack/Dilate used to give float32)."""
    np.random.seed(3)
    x = T(np.random.randn(2, 4, 4, 3).astype(f16), device)
    w = T(np.random.randn(3, 3, 3, 2).astype(f16) * 0.3, device)
    outs = {
        "stack": ops.stack([x, x * 2.0], 1),
        "split": ops.split(x, 3)[1],
        "dilate": ops.dilate(x, (1, 2), 1),
        "conv_stride2": ops.conv(x, w, stride=2, padding=1),
        "logsumexp": ops.logsumexp(x, axes=(3,)),
        "logsoftmax": ops.logsoftmax(ops.reshape(x, (8, 12))),
        "relu_tanh_exp": ops.exp(ops.tanh(ops.relu(x))),
        "transpose_bcast": ops.broadcast_to(ops.summation(x, (0,)), (5, 4, 4, 3)),
        "pad_crop": ops.crop(ops.pad(x, ((0, 0), (1, 1), (0, 2), (0, 0))), ((0, 0), (1, 0), (0, 1), (0, 0))),
        "flip": ops.flip(x, (1, 2)),
    }
    for name, y in outs.items():
        assert y.dtype == "float16", name
        x.grad = w.grad = None
        ops.summation(y).backward()
        assert x.grad.dtype == "float16", name
        if w.grad is not None:
            assert w.grad.dtype == "float16", name


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_float16_conv_matches_float32(device):
    np.random.seed(4)
    x = np.random.randn(2, 6, 6, 3).astype(f16)
    w = (np.random.randn(3, 3, 3, 4) * 0.3).astype(f16)
    res = {}
    for dt in ("float16", "float32"):
        X, W = T(x, device, dtype=dt), T(w, device, dtype=dt)
        y = ops.conv(X, W, stride=2, padding=1)
        ops.summation(y * y).backward()
        res[dt] = (y.numpy().astype(f32), X.grad.numpy().astype(f32), W.grad.numpy().astype(f32))
    for got, want in zip(res["float16"], res["float32"]):
        np.testing.assert_allclose(got, want, rtol=1e-2, atol=5e-2)


# --------------------------------------------------------------------------- Tensor construction / graph


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_tensor_constructor_dtypes(device):
    a = T(np.ones(3, f32), device)
    h = ndl.Tensor(a, dtype=np.float16)  # same device: converted on the device
    assert h.dtype == "float16" and h.device == device and h.is_leaf()
    assert ndl.Tensor(h).dtype == "float16"
    assert ndl.Tensor(np.ones(3, f16), device=device).dtype == "float16"
    assert ndl.Tensor(np.ones(3), device=device).dtype == "float32"  # float64 -> float32
    assert ndl.Tensor([1.0, 2.0], device=device, dtype="float16").dtype == "float16"
    other = ndl.cpu() if device != ndl.cpu() else ndl.cpu_numpy()
    moved = ndl.Tensor(h, device=other)
    assert moved.dtype == "float16" and moved.device == other
    assert h.detach().dtype == "float16" and h.data.dtype == "float16"
    for init_fn in (ndl.init.zeros, ndl.init.ones, ndl.init.rand, ndl.init.randn):
        assert init_fn(2, 3, device=device, dtype="float16").dtype == "float16"
    assert ndl.init.zeros_like(h).dtype == "float16"


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_cast_node_survives_release_cached_data(device):
    a = T(np.array([0.1, 0.2, 0.3], f32), device)
    h = a.half()
    y = ops.summation(h * 2.0)
    expected = h.numpy().copy()
    h.release_cached_data()
    assert h.cached_data is None and h.dtype == "float16" and h.shape == (3,)
    assert h.cached_data is None  # metadata did not trigger recomputation
    np.testing.assert_array_equal(h.numpy(), expected)
    y.backward()
    assert a.grad.dtype == "float32"
    np.testing.assert_array_equal(a.grad.numpy(), np.full(3, 2.0, f32))


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_parameter_dtype_and_optimizer_step(device):
    """SGD/Adam update float32 parameters from grads that flowed through float16."""
    np.random.seed(5)
    for opt_cls in (ndl.optim.SGD, ndl.optim.Adam):
        layer = nn.Linear(4, 3, device=device)
        opt = opt_cls(layer.parameters(), lr=0.1)
        before = [p.numpy().copy() for p in layer.parameters()]
        x = T(np.random.randn(5, 4).astype(f16), device, requires_grad=False)
        h = x @ layer.weight.half() + ops.broadcast_to(ops.reshape(layer.bias.half(), (1, 3)), (5, 3))
        loss = ops.summation(h.float() ** 2)
        opt.reset_grad()
        loss.backward()
        assert all(p.grad.dtype == "float32" for p in layer.parameters())
        opt.step()
        for p, old in zip(layer.parameters(), before):
            assert p.dtype == "float32" and not np.allclose(p.numpy(), old)


def test_float32_training_unchanged():
    """The float32-only path gives the same numbers as before the dtype changes."""
    from models import ResNet9

    np.random.seed(0)
    dev = ndl.cpu()
    model = ResNet9(device=dev)
    x = ndl.Tensor(np.random.rand(2, 3, 32, 32).astype(f32), device=dev)
    y = ndl.Tensor(np.array([1, 2]), device=dev)
    loss = nn.SoftmaxLoss()(model(x), y)
    loss.backward()
    assert loss.dtype == "float32"
    np.testing.assert_allclose(loss.numpy(), 4.283184, rtol=1e-6)
    assert all(p.grad.dtype == "float32" for p in model.parameters())
