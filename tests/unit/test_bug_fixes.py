"""Regression tests for bugs fixed on the fix_bugs branch."""
import sys

sys.path.append("./python")
import numpy as np
import pytest

import needle as ndl
import needle.nn as nn
from needle.backend_ndarray import ndarray as nd


_DEVICES = [
    ndl.cpu(),
    pytest.param(
        ndl.cuda(), marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU")
    ),
]


def numerical_grad(f, arrays, eps=1e-2):
    """Central finite differences of scalar f(*arrays) w.r.t. each array (float64)."""
    grads = []
    for a in arrays:
        g = np.zeros_like(a)
        it = np.nditer(a, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            orig = a[idx]
            a[idx] = orig + eps
            fp = f(*arrays)
            a[idx] = orig - eps
            fm = f(*arrays)
            a[idx] = orig
            g[idx] = (fp - fm) / (2 * eps)
        grads.append(g)
    return grads


# --- 1. gradients must not read cached_data of released inputs ----------------


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_backward_after_release_cached_data(device):
    x = ndl.Tensor(np.random.randn(2, 3).astype(np.float32), device=device)
    h = x * 2.0
    r = ndl.ops.reshape(h, (3, 2))
    b = ndl.ops.broadcast_to(ndl.ops.reshape(r, (1, 3, 2)), (4, 3, 2))
    lse = ndl.ops.logsumexp(b, axes=(2,))
    ls = ndl.ops.logsoftmax(ndl.ops.reshape(lse, (4, 3)))
    loss = ndl.ops.summation(ls)
    for t in (h, r, b, lse):
        t.release_cached_data()
    loss.backward()
    assert x.grad.shape == x.shape


# --- 2. device=None modules must not mix backends ---------------------------


def test_rnn_lstm_transformer_default_device():
    x = ndl.Tensor(np.ones((3, 2, 4), dtype=np.float32), device=ndl.cpu())
    out, _ = nn.RNN(4, 5)(x)
    assert out.shape == (3, 2, 5)
    out, _ = nn.LSTM(4, 5)(x)
    assert out.shape == (3, 2, 5)
    x = ndl.Tensor(np.ones((5, 2, 16), dtype=np.float32), device=ndl.cpu())
    out, _ = nn.Transformer(16, 32, 1, num_head=2, dim_head=8)(x)
    assert out.shape == (5, 2, 16)


# --- 3 & 4. conv gradient: uneven stride, non-square kernels ------------------


def _np_conv(A, B, stride, padding):
    A = np.pad(A, ((0, 0), (padding, padding), (padding, padding), (0, 0)))
    N, H, W, _ = A.shape
    Kh, Kw, _, Cout = B.shape
    Ho = (H - Kh) // stride + 1
    Wo = (W - Kw) // stride + 1
    out = np.zeros((N, Ho, Wo, Cout))
    for i in range(Kh):
        for j in range(Kw):
            patch = A[:, i : i + Ho * stride : stride, j : j + Wo * stride : stride, :]
            out += patch @ B[i, j]
    return out


@pytest.mark.parametrize(
    "H,Kh,Kw,stride,padding",
    [
        (6, 3, 3, 1, 1),
        (6, 3, 3, 2, 1),
        (5, 3, 3, 2, 1),  # (H + 2p - K) not divisible by stride
        (7, 3, 3, 3, 0),
        (5, 3, 1, 1, 1),  # non-square kernel
        (6, 2, 3, 2, 1),  # non-square kernel + stride
        (4, 2, 2, 1, 2),  # padding >= kernel size
    ],
)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_conv_gradient(H, Kh, Kw, stride, padding, device):
    np.random.seed(0)
    A = np.random.randn(2, H, H, 2)
    B = np.random.randn(Kh, Kw, 2, 3)
    Ho = (H + 2 * padding - Kh) // stride + 1
    Wo = (H + 2 * padding - Kw) // stride + 1
    W_out = np.random.randn(2, Ho, Wo, 3)

    At = ndl.Tensor(A.astype(np.float32), device=device)
    Bt = ndl.Tensor(B.astype(np.float32), device=device)
    out = ndl.ops.conv(At, Bt, stride=stride, padding=padding)
    np.testing.assert_allclose(out.numpy(), _np_conv(A, B, stride, padding), rtol=1e-4, atol=1e-4)
    (out * ndl.Tensor(W_out.astype(np.float32), device=device)).sum().backward()

    gA, gB = numerical_grad(lambda a, b: np.sum(_np_conv(a, b, stride, padding) * W_out), [A, B])
    np.testing.assert_allclose(At.grad.numpy(), gA, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(Bt.grad.numpy(), gB, rtol=1e-3, atol=1e-3)


# --- 5. negative slice starts ------------------------------------------------


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_negative_slice_start(device):
    a = np.arange(24, dtype=np.float32).reshape(4, 6)
    A = nd.array(a, device=device)
    np.testing.assert_array_equal(A[-2:, -3:-1].numpy(), a[-2:, -3:-1])
    np.testing.assert_array_equal(A[-3:-1, 1:].numpy(), a[-3:-1, 1:])


# --- 6. broadcast_to adding leading dimensions -------------------------------


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_broadcast_to_new_dims(device):
    a = np.random.randn(3, 1).astype(np.float32)
    A = nd.array(a, device=device)
    np.testing.assert_array_equal(
        A.broadcast_to((2, 3, 4)).numpy(), np.broadcast_to(a, (2, 3, 4))
    )
    x = ndl.Tensor(a, device=device)
    y = ndl.ops.broadcast_to(x, (2, 3, 4))
    y.sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), np.full((3, 1), 8.0))
    with pytest.raises(AssertionError):
        A.broadcast_to((2, 4, 4))


# --- 7. EWisePow with negative bases ----------------------------------------


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_ewise_pow_negative_base(device):
    a = np.array([-2.0, -1.5, 0.0, 3.0], dtype=np.float32)
    b = np.array([2.0, 3.0, 2.0, 0.5], dtype=np.float32)
    A = ndl.Tensor(a, device=device)
    B = ndl.Tensor(b, device=device)
    out = A ** B
    np.testing.assert_allclose(out.numpy(), a ** b, rtol=1e-5)
    out.sum().backward()
    np.testing.assert_allclose(A.grad.numpy(), b * a ** (b - 1), rtol=1e-5)


# --- 8. CIFAR10 transforms with batched indices ------------------------------


def test_cifar10_transforms_batched_index():
    ds = ndl.data.CIFAR10Dataset.__new__(ndl.data.CIFAR10Dataset)
    ds.X = np.random.rand(4, 3, 32, 32).astype(np.float32)
    ds.y = np.arange(4)
    ds.transforms = [ndl.data.RandomFlipHorizontal(p=1.0)]
    X, y = next(iter(ndl.data.DataLoader(ds, batch_size=2)))
    assert X.shape == (2, 3, 32, 32)
    np.testing.assert_allclose(X.numpy(), ds.X[:2][..., ::-1])
    img, _ = ds[0]
    np.testing.assert_allclose(img, ds.X[0][..., ::-1])
