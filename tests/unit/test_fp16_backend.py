"""float16 storage support in the native (cpu / cuda) NDArray backends.

float16 arrays are stored in 16 bits and computed in float32, so the reference for an
element-wise op is numpy computed in float32 and rounded to float16 once.
"""
import sys

sys.path.append("./python")
import numpy as np
import pytest

from needle.backend_ndarray import ndarray as nd

_DEVICES = [
    nd.cpu(),
    pytest.param(nd.cuda(), marks=pytest.mark.skipif(not nd.cuda().enabled(), reason="No GPU")),
]
_IDS = ["cpu", "cuda"]
f16, f32 = np.float16, np.float32


def rand16(*shape, low=-2.0, high=2.0):
    return np.random.uniform(low, high, shape).astype(f16)


def ref(fn, *arrays):
    """fn evaluated in float32 on float16 inputs, rounded once to float16."""
    return np.asarray(fn(*[a.astype(f32) for a in arrays])).astype(f16)


def assert_ulp(actual, desired, max_ulp=1):
    """float16 arrays equal up to `max_ulp` units in the last place (NaNs must match)."""
    assert actual.dtype == f16 and desired.dtype == f16
    assert actual.shape == desired.shape
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(desired))
    ok = ~np.isnan(desired)
    a = actual[ok].view(np.int16).astype(np.int32)
    d = desired[ok].view(np.int16).astype(np.int32)
    # map sign-magnitude bit patterns onto a monotonic integer line
    a = np.where(a < 0, -32768 - a, a)
    d = np.where(d < 0, -32768 - d, d)
    assert np.max(np.abs(a - d), initial=0) <= max_ulp, np.max(np.abs(a - d))


# --------------------------------------------------------------------------- dtype plumbing


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_dtype_roundtrip(device):
    x = rand16(3, 4)
    A = nd.array(x, device=device)  # float16 numpy input keeps its dtype
    assert A.dtype == "float16" and A.itemsize == 2
    out = A.numpy()
    assert out.dtype == f16
    np.testing.assert_array_equal(out, x)

    B = nd.array(x.astype(np.float64), device=device)  # other inputs -> float32 (unchanged)
    assert B.dtype == "float32"
    C = nd.array(x.astype(f32), device=device, dtype="float16")
    assert C.dtype == "float16"
    np.testing.assert_array_equal(C.numpy(), x)
    assert nd.array(A, device=device).dtype == "float16"  # copying keeps dtype
    assert nd.empty((2, 3), dtype="float16", device=device).dtype == "float16"
    F = nd.full((2, 3), 1.5, dtype=np.float16, device=device)
    np.testing.assert_array_equal(F.numpy(), np.full((2, 3), 1.5, f16))


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_views_keep_dtype(device):
    x = rand16(4, 6)
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
        assert view.dtype == "float16"
        np.testing.assert_array_equal(view.numpy(), expected)
        np.testing.assert_array_equal(view.compact().numpy(), expected)


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_setitem(device):
    x = rand16(4, 5)
    A = nd.array(x, device=device)
    A[1:3, 1:4] = nd.array(np.ones((2, 3), f16), device=device)
    A[0, :] = 7.25
    A[3, 0:2] = nd.array(np.array([[0.1, 0.2]], f32), device=device)  # float32 source is cast
    x[1:3, 1:4] = 1
    x[0, :] = 7.25
    x[3, 0:2] = np.array([0.1, 0.2], f32).astype(f16)
    np.testing.assert_array_equal(A.numpy(), x)


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_mixed_dtype_rejected(device):
    A = nd.array(rand16(3, 3), device=device)
    B = nd.array(rand16(3, 3).astype(f32), device=device)
    for op in (lambda: A + B, lambda: A * B, lambda: A @ B, lambda: A ** B, lambda: A.maximum(B)):
        with pytest.raises(AssertionError, match="dtype"):
            op()
    with pytest.raises(TypeError):
        nd.empty((2,), dtype="int32", device=device)


def test_numpy_backend_rejects_float16():
    with pytest.raises(TypeError, match="does not support"):
        nd.array(rand16(2, 2), device=nd.cpu_numpy())
    # float32 is unaffected
    np.testing.assert_array_equal(
        nd.array(np.ones(3, f32), device=nd.cpu_numpy()).numpy(), np.ones(3, f32)
    )


# --------------------------------------------------------------------------- casting


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_cast_float16_to_float32_exhaustive(device):
    """Every one of the 65536 float16 bit patterns widens exactly like numpy."""
    x = np.arange(65536, dtype=np.uint32).astype(np.uint16).view(f16)
    out = nd.array(x, device=device).astype("float32").numpy()
    assert out.dtype == f32
    expected = x.astype(f32)
    np.testing.assert_array_equal(np.isnan(out), np.isnan(expected))
    ok = ~np.isnan(expected)
    np.testing.assert_array_equal(out[ok].view(np.uint32), expected[ok].view(np.uint32))


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_cast_float32_to_float16_matches_numpy(device):
    """float32 -> float16 rounds to nearest even, including subnormals and overflow."""
    np.random.seed(0)
    halves = np.arange(65536, dtype=np.uint32).astype(np.uint16).view(f16).astype(f32)
    halves = halves[np.isfinite(halves)]
    nxt = np.nextafter(halves, np.float32(np.inf))
    mid = (halves[:-1].astype(np.float64) + np.sort(halves)[1:].astype(np.float64)) / 2
    special = np.array(
        [0.0, -0.0, np.inf, -np.inf, 65504, 65519.99, 65520, 65536, 1e6, -1e6,
         2.0 ** -24, 2.0 ** -25, 2.0 ** -25 * 1.0001, 2.0 ** -26, 6.1e-5, 1e-8, -1e-8, 1e-40],
        dtype=np.float64,
    )
    x = np.concatenate([
        halves, nxt, np.sort(mid).astype(f32), special.astype(f32),
        (np.random.randn(200000) * 10.0 ** np.random.uniform(-9, 6, 200000)).astype(f32),
    ]).astype(f32)
    out = nd.array(x, device=device).astype("float16").numpy()
    assert out.dtype == f16
    with np.errstate(over="ignore"):
        expected = x.astype(f16)
    np.testing.assert_array_equal(out.view(np.uint16), expected.view(np.uint16))
    nan_out = nd.array(np.array([np.nan], f32), device=device).astype("float16").numpy()
    assert np.isnan(nan_out).all()


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_astype_same_dtype(device):
    A = nd.array(rand16(2, 3), device=device)
    assert A.astype("float16", copy=False) is A
    B = A.astype("float16")
    assert B is not A
    np.testing.assert_array_equal(B.numpy(), A.numpy())
    C = nd.array(rand16(4, 6), device=device)[1:3, ::2]  # non-compact source
    np.testing.assert_array_equal(C.astype("float32").numpy(), C.numpy().astype(f32))


# --------------------------------------------------------------------------- arithmetic


EXACT_BINARY = {
    "add": (lambda a, b: a + b, lambda a, b: a + b),
    "mul": (lambda a, b: a * b, lambda a, b: a * b),
    "div": (lambda a, b: a / b, lambda a, b: a / b),
    "maximum": (lambda a, b: a.maximum(b), np.maximum),
    "eq": (lambda a, b: a == b, lambda a, b: (a == b).astype(f32)),
    "ge": (lambda a, b: a >= b, lambda a, b: (a >= b).astype(f32)),
}


@pytest.mark.parametrize("name", list(EXACT_BINARY))
@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_ewise_and_scalar_ops(name, device):
    np.random.seed(1)
    op, np_op = EXACT_BINARY[name]
    a, b = rand16(5, 7), rand16(5, 7)
    b[0, :3] = a[0, :3]  # some equal entries for eq / ge
    A, B = nd.array(a, device=device), nd.array(b, device=device)
    C = op(A, B)
    assert C.dtype == "float16"
    assert_ulp(C.numpy(), ref(np_op, a, b), max_ulp=0)
    s = f16(0.75)
    C = op(A, float(s))
    assert C.dtype == "float16"
    assert_ulp(C.numpy(), ref(lambda x: np_op(x, f32(s)), a), max_ulp=0)


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_unary_and_power_ops(device):
    np.random.seed(2)
    a = rand16(6, 9, low=0.05, high=3.0)
    b = rand16(6, 9, low=-2.0, high=2.0)
    A, B = nd.array(a, device=device), nd.array(b, device=device)
    for C, expected in [
        (A.log(), ref(np.log, a)),
        (A.exp(), ref(np.exp, a)),
        (B.tanh(), ref(np.tanh, b)),
        (A ** 3, ref(lambda x: x ** 3, a)),
        (A ** B, ref(np.power, a, b)),
        (-B, -b),
        (1 - B, ref(lambda x: 1 - x, b)),
    ]:
        assert C.dtype == "float16"
        assert_ulp(C.numpy(), expected, max_ulp=1)  # libm may differ by one rounding


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_overflow_saturates_to_inf(device):
    A = nd.array(np.array([60000.0, -60000.0, 300.0], f16), device=device)
    out = (A * 2.0).numpy()
    assert out[0] == np.inf and out[1] == -np.inf and out[2] == 600
    assert (A * A).numpy()[2] == np.inf  # 90000 > float16 max


# --------------------------------------------------------------------------- matmul / reductions


@pytest.mark.parametrize(
    "m,n,p",
    [(16, 16, 16), (8, 64, 24), (5, 7, 3), (31, 33, 17), (64, 1024, 8)],
    ids=["tiled", "tiled-rect", "naive-small", "naive-odd", "long-k"],
)
@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_matmul_accumulates_in_float32(m, n, p, device):
    np.random.seed(3)
    a, b = rand16(m, n, low=-1, high=1), rand16(n, p, low=-1, high=1)
    C = nd.array(a, device=device) @ nd.array(b, device=device)
    assert C.dtype == "float16" and C.shape == (m, p)
    expected = (a.astype(np.float64) @ b.astype(np.float64)).astype(f16)
    # float32 accumulation + a single final rounding: within 1 float16 ulp of exact
    assert_ulp(C.numpy(), expected, max_ulp=1)


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_matmul_on_views(device):
    np.random.seed(4)
    a = rand16(16, 24)
    A = nd.array(a, device=device)
    C = A[:, :8].permute((1, 0)) @ A[:, 8:]  # (8, 16) @ (16, 16), both non-compact
    expected = (a[:, :8].T.astype(np.float64) @ a[:, 8:].astype(np.float64)).astype(f16)
    assert_ulp(C.numpy(), expected, max_ulp=1)


@pytest.mark.parametrize("axis", [None, 0, 1, 2])
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_reductions(axis, keepdims, device):
    np.random.seed(5)
    a = rand16(4, 300, 5, low=0, high=1)
    A = nd.array(a, device=device)
    S = A.sum(axis=axis, keepdims=keepdims)
    M = A.max(axis=axis, keepdims=keepdims)
    assert S.dtype == "float16" and M.dtype == "float16"
    exp_sum = a.astype(np.float64).sum(axis=axis, keepdims=keepdims and axis is not None)
    exp_max = a.max(axis=axis, keepdims=keepdims and axis is not None)
    np.testing.assert_array_equal(M.numpy().reshape(np.shape(exp_max)), exp_max)
    # summing 300 values: a float16 accumulator would be off by many ulps here
    assert_ulp(S.numpy().reshape(np.shape(exp_sum)), np.asarray(exp_sum).astype(f16), max_ulp=1)


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_sum_beats_float16_accumulation(device):
    """2048 ones: a float16 running sum would stop at 2048 only by luck; 4096 + 1s stall."""
    a = np.ones(5000, f16)
    out = nd.array(a, device=device).sum().numpy()
    assert out.dtype == f16 and out[0] == f16(5000)  # float16 accumulation would give 2048
