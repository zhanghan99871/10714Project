"""bfloat16 storage in the NDArray backends (numpy, cpu, cuda): operations.

Every op widens bfloat16 to float32, computes in float32 and rounds once, so the
reference for an element-wise op is the float32 result rounded by ml_dtypes.
Matmul and reductions accumulate in float32.
"""
import sys

sys.path.append("./python")
import numpy as np
import pytest

ml_dtypes = pytest.importorskip("ml_dtypes")
from needle.backend_ndarray import ndarray as nd

bf16 = ml_dtypes.bfloat16
f16, f32 = np.float16, np.float32
_HAS_GPU = nd.cuda().enabled()
_DEVICES = [
    nd.cpu_numpy(),
    nd.cpu(),
    pytest.param(nd.cuda(), marks=pytest.mark.skipif(not _HAS_GPU, reason="No GPU")),
]
_IDS = ["numpy", "cpu", "cuda"]
_NATIVE = [nd.cpu(), pytest.param(nd.cuda(), marks=pytest.mark.skipif(not _HAS_GPU, reason="No GPU"))]
_NATIVE_IDS = ["cpu", "cuda"]


def rbf(*shape, low=-2.0, high=2.0):
    return np.random.uniform(low, high, shape).astype(bf16)


def ref(fn, *arrays):
    """fn in float32 on bfloat16 inputs, rounded once to bfloat16."""
    with np.errstate(all="ignore"):
        return np.asarray(fn(*[a.astype(f32) for a in arrays])).astype(f32).astype(bf16)


def ulp_diff(a, b):
    """Max distance in bfloat16 ulps (NaNs must coincide)."""
    a32, b32 = a.astype(f32), b.astype(f32)
    np.testing.assert_array_equal(np.isnan(a32), np.isnan(b32))
    ok = ~np.isnan(a32)
    ia = np.ascontiguousarray(a[ok]).view(np.int16).astype(np.int32)
    ib = np.ascontiguousarray(b[ok]).view(np.int16).astype(np.int32)
    ia = np.where(ia < 0, -32768 - ia, ia)
    ib = np.where(ib < 0, -32768 - ib, ib)
    return int(np.max(np.abs(ia - ib), initial=0))


# --------------------------------------------------------------------------- element-wise


EXACT = {
    "add": (lambda a, b: a + b, lambda a, b: a + b),
    "sub": (lambda a, b: a - b, lambda a, b: a - b),
    "mul": (lambda a, b: a * b, lambda a, b: a * b),
    "div": (lambda a, b: a / b, lambda a, b: a / b),
    "maximum": (lambda a, b: a.maximum(b), np.maximum),
    "eq": (lambda a, b: a == b, lambda a, b: (a == b).astype(f32)),
    "ge": (lambda a, b: a >= b, lambda a, b: (a >= b).astype(f32)),
    "scalar_mul": (lambda a, b: a * 1.3, lambda a, b: a * f32(1.3)),
    "scalar_add": (lambda a, b: a + 0.1, lambda a, b: a + f32(0.1)),
    "scalar_div": (lambda a, b: a / 3.0, lambda a, b: a / f32(3.0)),
    "rsub": (lambda a, b: 1.0 - a, lambda a, b: f32(1.0) - a),
    "neg": (lambda a, b: -a, lambda a, b: -a),
}


@pytest.mark.parametrize("name", list(EXACT))
@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_elementwise_ops_exact(name, device):
    np.random.seed(0)
    op, np_op = EXACT[name]
    a, b = rbf(6, 7), rbf(6, 7, low=0.5, high=3.0)
    b[0, :3] = a[0, :3]
    out = op(nd.array(a, device=device), nd.array(b, device=device))
    assert out.dtype == "bfloat16"
    assert ulp_diff(out.numpy(), ref(np_op, a, b)) == 0


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_transcendental_and_power_ops(device):
    np.random.seed(1)
    a, b = rbf(6, 9, low=0.05, high=3.0), rbf(6, 9)
    A, B = nd.array(a, device=device), nd.array(b, device=device)
    for out, expected in [
        (A.log(), ref(np.log, a)), (A.exp(), ref(np.exp, a)), (B.tanh(), ref(np.tanh, b)),
        (A ** 3, ref(lambda x: x ** 3, a)), (A ** B, ref(np.power, a, b)),
        (1.0 / A, ref(lambda x: 1 / x, a)),
    ]:
        assert out.dtype == "bfloat16"
        assert ulp_diff(out.numpy(), expected) <= 1  # libm may differ by one rounding


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_float32_range_no_overflow(device):
    """Products that overflow float16 are fine in bfloat16 (float32's exponent)."""
    a = np.array([60000.0, 1e15, 1e-15], f32)  # squares: 3.6e9, 1e30, 1e-30
    A = nd.array(a, device=device, dtype="bfloat16")
    out = (A * A).numpy().astype(f32)
    np.testing.assert_allclose(out, (a.astype(bf16).astype(f32)) ** 2, rtol=2 ** -7)
    assert np.isfinite(out).all()


# --------------------------------------------------------------------------- views and indexing


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_views_and_setitem(device):
    np.random.seed(2)
    x = rbf(4, 6)
    A = nd.array(x, device=device)
    for view, expected in [
        (A.reshape((6, 4)), x.reshape(6, 4)),
        (A.permute((1, 0)), x.T),
        (A[1:3, ::2], x[1:3, ::2]),
        (A[-2:, -3:], x[-2:, -3:]),
        (A[:, 0:1].broadcast_to((2, 4, 1)), np.broadcast_to(x[:, 0:1], (2, 4, 1))),
        (A.flip((0, 1)), x[::-1, ::-1]),
        (A.pad(((1, 0), (0, 2))), np.pad(x, ((1, 0), (0, 2)))),
    ]:
        assert view.dtype == "bfloat16"
        np.testing.assert_array_equal(view.compact().numpy().view(np.uint16),
                                      np.ascontiguousarray(expected).view(np.uint16))
    B = nd.array(x, device=device)
    B[0, :] = 7.25
    B[1:3, 1:4] = nd.array(np.ones((2, 3), f32), device=device)  # float32 source is cast
    x2 = x.copy()
    x2[0, :] = 7.25
    x2[1:3, 1:4] = 1
    np.testing.assert_array_equal(B.numpy().view(np.uint16), x2.view(np.uint16))


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_mixed_dtypes_rejected(device):
    A = nd.array(rbf(3, 3), device=device)
    for other in (np.ones((3, 3), f32), np.ones((3, 3), f16)):
        B = nd.array(other, device=device)
        for op in (lambda: A + B, lambda: A * B, lambda: A @ B, lambda: A.maximum(B)):
            with pytest.raises(AssertionError, match="dtype"):
                op()


# --------------------------------------------------------------------------- matmul / reductions


@pytest.mark.parametrize(
    "sa,sb",
    [((16, 16), (16, 16)), ((5, 7), (7, 3)), ((64, 1024), (1024, 8)),
     ((3, 8, 24), (24, 16)), ((2, 1, 5, 7), (1, 3, 7, 2))],
    ids=["tiled", "naive", "long-k", "batched-shared", "batched-broadcast"],
)
@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_matmul_accumulates_in_float32(sa, sb, device):
    np.random.seed(3)
    a, b = rbf(*sa, low=-1, high=1), rbf(*sb, low=-1, high=1)
    out = nd.array(a, device=device) @ nd.array(b, device=device)
    assert out.dtype == "bfloat16"
    exact = (a.astype(np.float64) @ b.astype(np.float64)).astype(f32).astype(bf16)
    assert ulp_diff(out.numpy(), exact) <= 1  # float32 accumulation + one rounding


@pytest.mark.parametrize("axis", [None, 0, 1])
@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_reductions(axis, device):
    np.random.seed(4)
    a = rbf(4, 400, low=0, high=1)
    A = nd.array(a, device=device)
    S, M = A.sum(axis=axis), A.max(axis=axis)
    assert S.dtype == "bfloat16" and M.dtype == "bfloat16"
    exp_max = a.astype(f32).max(axis=axis)
    np.testing.assert_array_equal(M.numpy().astype(f32).reshape(np.shape(exp_max)), exp_max)
    exact = np.asarray(a.astype(np.float64).sum(axis=axis)).astype(f32).astype(bf16)
    assert ulp_diff(S.numpy().reshape(exact.shape), exact) <= 1


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_sum_does_not_accumulate_in_bf16(device):
    """With only 8 significant bits, a bfloat16 running sum of ones stops at 256."""
    out = nd.array(np.ones(5000, f32), device=device, dtype="bfloat16").sum().numpy()
    assert out[0] == f32(5000).astype(bf16)  # 4992: 5000 rounded once, not stuck at 256
    total = nd.array(np.ones(5000, f32), device=device, dtype="bfloat16").sum(dtype="float32")
    assert total.dtype == "float32" and total.numpy()[0] == 5000.0


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_isfinite_all_finite(device):
    x = np.array([0.0, -1.5, 3e38, np.inf, -np.inf, np.nan], f32)
    A = nd.array(x, device=device, dtype="bfloat16")
    np.testing.assert_array_equal(A.isfinite().numpy().astype(f32), np.isfinite(x.astype(bf16).astype(f32)))
    assert not A.all_finite() and A[0:3].all_finite()


# --------------------------------------------------------------------------- cross-backend agreement


CROSS = {
    "add": lambda a, b: a + b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "scalar": lambda a, b: a * 0.7 + 0.1,
    "permute_compact": lambda a, b: a.permute((1, 0)).compact(),
    "broadcast": lambda a, b: a[:, 0:1].broadcast_to((16, 24)).compact(),
    "max": lambda a, b: a.max(axis=1),
    "cast_f16_roundtrip": lambda a, b: a.astype("float16").astype("bfloat16"),
}
CROSS_APPROX = {
    "exp": lambda a, b: a.exp(),
    "tanh": lambda a, b: a.tanh(),
    "matmul": lambda a, b: a @ b.permute((1, 0)),
    "sum": lambda a, b: a.sum(axis=0),
}


@pytest.mark.parametrize("op", list(CROSS) + list(CROSS_APPROX))
@pytest.mark.parametrize("device", _NATIVE, ids=_NATIVE_IDS)
def test_native_backends_match_numpy_backend(op, device):
    np.random.seed(5)
    a, b = rbf(16, 24), rbf(16, 24, low=0.5, high=2)
    fn = CROSS.get(op) or CROSS_APPROX[op]
    ref_out = fn(nd.array(a, device=nd.cpu_numpy()), nd.array(b, device=nd.cpu_numpy())).numpy()
    out = fn(nd.array(a, device=device), nd.array(b, device=device)).numpy()
    assert out.dtype == np.dtype(bf16)
    assert ulp_diff(out, ref_out) <= (0 if op in CROSS else 1)
