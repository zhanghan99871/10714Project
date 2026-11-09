"""Batched matrix multiplication: NDArray backends, the MatMul op and attention."""
import sys

sys.path.append("./python")
import numpy as np
import pytest

import needle as ndl
import needle.nn as nn
from needle import autograd, ops
from needle.backend_ndarray import ndarray as nd

_HAS_GPU = ndl.cuda().enabled()
_DEVICES = [
    ndl.cpu_numpy(),
    ndl.cpu(),
    pytest.param(ndl.cuda(), marks=pytest.mark.skipif(not _HAS_GPU, reason="No GPU")),
]
_IDS = ["numpy", "cpu", "cuda"]
_NATIVE = [ndl.cpu(), pytest.param(ndl.cuda(), marks=pytest.mark.skipif(not _HAS_GPU, reason="No GPU"))]
_NATIVE_IDS = ["cpu", "cuda"]
f16, f32 = np.float16, np.float32

SHAPES = [
    ((3, 4, 5), (3, 5, 2)),  # plain batch
    ((2, 3, 4, 5), (2, 3, 5, 6)),  # two batch dims
    ((2, 3, 4, 5), (5, 6)),  # shared right matrix (e.g. a weight)
    ((4, 5), (2, 3, 5, 6)),  # shared left matrix
    ((2, 1, 4, 5), (1, 3, 5, 2)),  # partial broadcast on both sides
    ((3, 4, 5), (1, 5, 2)),  # size-1 batch dim
    ((2, 3, 16, 24), (2, 3, 24, 8)),  # tiled sizes
    ((1, 1, 1), (1, 1, 1)),
]


def T(x, device, dtype=None, requires_grad=True):
    return ndl.Tensor(x, device=device, dtype=dtype, requires_grad=requires_grad)


def ref_grads(a, b, g):
    """Analytic gradients of sum(g * (a @ b)) in float64, reduced to the input shapes."""

    def reduce(grad, shape):
        extra = grad.ndim - len(shape)
        grad = grad.sum(axis=tuple(range(extra))) if extra else grad
        axes = tuple(i for i, d in enumerate(shape) if d == 1 and grad.shape[i] != 1)
        return grad.sum(axis=axes, keepdims=True) if axes else grad

    ga = g @ np.swapaxes(b, -1, -2)
    gb = np.swapaxes(a, -1, -2) @ g
    return reduce(ga, a.shape), reduce(gb, b.shape)


# --------------------------------------------------------------------------- NDArray level


@pytest.mark.parametrize("sa,sb", SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_ndarray_batched_matmul_matches_numpy(sa, sb, device):
    np.random.seed(0)
    a, b = np.random.randn(*sa).astype(f32), np.random.randn(*sb).astype(f32)
    out = nd.array(a, device=device) @ nd.array(b, device=device)
    expected = a.astype(np.float64) @ b.astype(np.float64)
    assert out.shape == expected.shape and out.dtype == "float32"
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_non_compact_inputs(device):
    """Permuted and sliced (non-compact) views as both operands."""
    np.random.seed(1)
    a = np.random.randn(2, 6, 5, 4).astype(f32)
    b = np.random.randn(2, 3, 7, 6).astype(f32)
    A = nd.array(a, device=device).permute((0, 1, 3, 2))[:, 1:4, :, :]  # (2, 3, 4, 5)
    B = nd.array(b, device=device).permute((0, 1, 3, 2))[:, :, 1:, :]  # (2, 3, 5, 7)
    assert not A.is_compact() and not B.is_compact()
    expected = np.swapaxes(a, 2, 3)[:, 1:4] @ np.swapaxes(b, 2, 3)[:, :, 1:]
    np.testing.assert_allclose((A @ B).numpy(), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("m,n,p", [(7, 9, 3), (16, 24, 8)], ids=["naive", "tiled"])
@pytest.mark.parametrize("device", _NATIVE, ids=_NATIVE_IDS)
def test_batched_equals_per_slice_2d_matmul(m, n, p, device):
    """Every batch entry is bit-identical to the 2D matmul of that slice."""
    np.random.seed(2)
    a, b = np.random.randn(5, m, n).astype(f32), np.random.randn(5, n, p).astype(f32)
    batched = (nd.array(a, device=device) @ nd.array(b, device=device)).numpy()
    for z in range(5):
        single = (nd.array(a[z], device=device) @ nd.array(b[z], device=device)).numpy()
        np.testing.assert_array_equal(batched[z], single)


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_shared_operand_is_not_copied(device):
    W = nd.array(np.random.randn(5, 6).astype(f32), device=device)
    buf, stride = W._batch_operand((2, 3))
    assert stride == 0 and buf._handle is W._handle
    X = nd.array(np.random.randn(2, 3, 4, 5).astype(f32), device=device)
    buf, stride = X._batch_operand((2, 3))
    assert stride == 20 and buf._handle is X._handle
    P = nd.array(np.random.randn(2, 1, 4, 5).astype(f32), device=device)
    buf, stride = P._batch_operand((2, 3))  # partial broadcast is materialized
    assert stride == 20 and buf.shape == (2, 3, 4, 5)


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_float16_accumulates_in_float32(device):
    np.random.seed(3)
    a = np.random.uniform(-1, 1, (3, 8, 512)).astype(f16)
    b = np.random.uniform(-1, 1, (512, 4)).astype(f16)
    out = (nd.array(a, device=device) @ nd.array(b, device=device)).numpy()
    assert out.dtype == f16
    expected = (a.astype(np.float64) @ b.astype(np.float64)).astype(f16)
    ulp = np.abs(out.view(np.int16).astype(np.int32) - expected.view(np.int16).astype(np.int32))
    assert ulp.max() <= 1


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_errors(device):
    A = nd.array(np.ones((2, 3, 4), f32), device=device)
    with pytest.raises(ValueError, match="at least 2"):
        A @ nd.array(np.ones(4, f32), device=device)
    with pytest.raises(ValueError, match="not aligned"):
        A @ nd.array(np.ones((2, 5, 2), f32), device=device)
    with pytest.raises(ValueError, match="broadcast"):
        A @ nd.array(np.ones((3, 4, 2), f32), device=device)
    with pytest.raises(AssertionError, match="dtype"):
        A @ nd.array(np.ones((2, 4, 2), f16), device=device)


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_degenerate_sizes(device):
    out = nd.array(np.ones((3, 2, 0), f32), device=device) @ nd.array(np.ones((3, 0, 4), f32), device=device)
    np.testing.assert_array_equal(out.numpy(), np.zeros((3, 2, 4), f32))
    out = nd.array(np.ones((0, 2, 3), f32), device=device) @ nd.array(np.ones((3, 4), f32), device=device)
    assert out.shape == (0, 2, 4)


def test_fallback_without_kernel(monkeypatch):
    """Backends built before matmul_batched existed loop over 2D matmuls."""
    np.random.seed(4)
    for sa, sb in SHAPES[:6]:
        a, b = np.random.randn(*sa).astype(f32), np.random.randn(*sb).astype(f32)
        with_kernel = (nd.array(a, device=nd.cpu()) @ nd.array(b, device=nd.cpu())).numpy()
        with monkeypatch.context() as mp:
            mp.delattr(nd.ndarray_backend_cpu, "matmul_batched")
            fallback = (nd.array(a, device=nd.cpu()) @ nd.array(b, device=nd.cpu())).numpy()
        np.testing.assert_array_equal(fallback, with_kernel)


def test_broadcast_shapes():
    assert nd.broadcast_shapes((2, 1), (3,)) == (2, 3)
    assert nd.broadcast_shapes((), (4, 5)) == (4, 5)
    assert nd.broadcast_shapes((2, 1, 4), (3, 1)) == (2, 3, 4)
    with pytest.raises(ValueError):
        nd.broadcast_shapes((2, 3), (4, 3))


# --------------------------------------------------------------------------- MatMul op


@pytest.mark.parametrize("sa,sb", SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_matmul_op_gradients(sa, sb, device):
    np.random.seed(5)
    a, b = np.random.randn(*sa), np.random.randn(*sb)
    A, B = T(a, device), T(b, device)
    out = A @ B
    g = np.random.randn(*out.shape)
    ops.summation(out * T(g, device, requires_grad=False)).backward()
    ga, gb = ref_grads(a, b, g)
    assert A.grad.shape == A.shape and B.grad.shape == B.shape
    np.testing.assert_allclose(A.grad.numpy(), ga, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(B.grad.numpy(), gb, rtol=1e-4, atol=1e-4)


def test_matmul_op_finite_differences():
    np.random.seed(6)
    a, b = np.random.randn(2, 1, 3, 4), np.random.randn(3, 4, 2)
    A, B = T(a, ndl.cpu()), T(b, ndl.cpu())
    ops.summation((A @ B) ** 2).backward()
    f = lambda x, y: np.sum((x @ y) ** 2)
    eps = 1e-3
    for arr, grad in ((a, A.grad.numpy()), (b, B.grad.numpy())):
        num = np.zeros_like(arr)
        for idx in np.ndindex(*arr.shape):
            old = arr[idx]
            arr[idx] = old + eps
            hi = f(a, b)
            arr[idx] = old - eps
            lo = f(a, b)
            arr[idx] = old
            num[idx] = (hi - lo) / (2 * eps)
        np.testing.assert_allclose(grad, num, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_2d_matmul_gradient_unchanged(device):
    np.random.seed(7)
    a, b = np.random.randn(4, 5).astype(f32), np.random.randn(5, 3).astype(f32)
    A, B = T(a, device), T(b, device)
    out = A @ B
    out.backward()
    ones = T(np.ones((4, 3), f32), device, requires_grad=False)
    np.testing.assert_array_equal(A.grad.numpy(), (ones @ ops.transpose(T(b, device))).numpy())
    np.testing.assert_array_equal(B.grad.numpy(), (ops.transpose(T(a, device)) @ ones).numpy())


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_linear_on_3d_input(device):
    """Linear now accepts (batch, seq, features) directly (shared weight, stride 0)."""
    np.random.seed(8)
    layer = nn.Linear(6, 4, device=device)
    x = np.random.randn(2, 5, 6).astype(f32)
    X3 = T(x, device)
    y3 = layer(X3)
    ops.summation(y3 * y3).backward()
    g3 = [p.grad.numpy() for p in layer.parameters()]
    X2 = T(x.reshape(10, 6), device)
    y2 = layer(X2)
    ops.summation(y2 * y2).backward()
    assert y3.shape == (2, 5, 4)
    np.testing.assert_allclose(y3.numpy().reshape(10, 4), y2.numpy(), rtol=1e-5, atol=1e-5)
    for a, b in zip(g3, [p.grad.numpy() for p in layer.parameters()]):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(X3.grad.numpy().reshape(10, 6), X2.grad.numpy(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_batched_matmul_under_autocast(device):
    np.random.seed(9)
    W = T(np.random.randn(5, 3).astype(f32), device)
    X = T(np.random.randn(2, 4, 5).astype(f32), device, requires_grad=False)
    with ndl.autocast():
        y = X @ W
    assert y.dtype == "float16"
    ops.summation(y.float()).backward()
    assert W.grad.dtype == "float32"
    expected = X.numpy().astype(f16).astype(np.float64).reshape(8, 5).sum(0)[:, None].repeat(3, 1)
    np.testing.assert_allclose(W.grad.numpy(), expected, rtol=1e-2, atol=1e-2)


# --------------------------------------------------------------------------- attention


def old_attention_matmul(a, b_transpose):
    """The previous MultiHeadAttention.matmul: broadcast, multiply, sum over d."""
    a_shape = (*a.shape[:-1], 1, *a.shape[-1:])
    a = a.reshape(a_shape)
    bt_shape = (*b_transpose.shape[:-2], 1, *b_transpose.shape[-2:])
    b_transpose = b_transpose.reshape(bt_shape)
    s = list(a_shape)
    s[-2] = bt_shape[-2]
    a = a.broadcast_to(s)
    s = list(bt_shape)
    s[-3] = a_shape[-3]
    b_transpose = b_transpose.broadcast_to(s)
    return (a * b_transpose).sum(len(a.shape) - 1)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_attention_matches_previous_implementation(causal, device):
    rs = np.random.RandomState(0)
    q, k, v = (rs.randn(2, 3, 7, 8).astype(f32) for _ in range(3))
    layer = nn.MultiHeadAttention(causal=causal, device=device)
    Q, K, V = T(q, device), T(k, device), T(v, device)
    out, probs = layer(Q, K, V)
    ops.summation(out).backward()

    Q2, K2, V2 = T(q, device), T(k, device), T(v, device)
    logits = old_attention_matmul(Q2, K2) * (1 / np.sqrt(8))
    if causal:
        mask = T(layer.create_causal_mask(7, 7, device), device, requires_grad=False)
        logits = logits + ops.broadcast_to(mask, logits.shape)
    probs2 = layer.softmax(logits)
    out2 = old_attention_matmul(probs2, ops.transpose(V2, (2, 3)))
    ops.summation(out2).backward()

    np.testing.assert_allclose(out.numpy(), out2.numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(probs.numpy(), probs2.numpy(), rtol=1e-5, atol=1e-6)
    for a, b in ((Q.grad, Q2.grad), (K.grad, K2.grad), (V.grad, V2.grad)):
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_attention_no_longer_materializes_5d_tensors(device):
    B, H, Q, D = 2, 3, 6, 16
    layer = nn.MultiHeadAttention(causal=True, device=device)
    x = [T(np.random.randn(B, H, Q, D).astype(f32), device) for _ in range(3)]
    out, _ = layer(*x)
    nodes = autograd.find_topo_sort([out])
    assert max(len(n.shape) for n in nodes) == 4
    assert max(int(np.prod(n.shape)) for n in nodes) <= B * H * Q * max(Q, D)


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_attention_under_autocast(device):
    rs = np.random.RandomState(1)
    q, k, v = (rs.randn(2, 2, 9, 8).astype(f32) for _ in range(3))
    res = {}
    for amp in (False, True):
        layer = nn.MultiHeadAttention(causal=True, device=device)
        X = [T(a, device) for a in (q, k, v)]
        with ndl.autocast(enabled=amp):
            out, probs = layer(*X)
        ops.summation(out.float()).backward()
        assert all(np.isfinite(t.grad.numpy()).all() for t in X)
        res[amp] = out.numpy().astype(f32)
    assert out.dtype == "float16"  # probs @ v runs as a float16 batched matmul
    np.testing.assert_allclose(res[True], res[False], rtol=2e-2, atol=2e-2)
