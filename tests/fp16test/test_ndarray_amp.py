"""NDArray-level mixed precision support: dtype handling across devices, the AMP
helpers (astype / isfinite / all_finite / sum(dtype=)), and agreement between the
numpy reference backend and the native backends."""
import sys
import types

sys.path.append("./python")
import numpy as np
import pytest

from needle.backend_ndarray import ndarray as nd

_HAS_GPU = nd.cuda().enabled()
_DEVICES = [
    nd.cpu_numpy(),
    nd.cpu(),
    pytest.param(nd.cuda(), marks=pytest.mark.skipif(not _HAS_GPU, reason="No GPU")),
]
_IDS = ["numpy", "cpu", "cuda"]
# backends compared against the numpy reference backend
_NATIVE = [
    nd.cpu(),
    pytest.param(nd.cuda(), marks=pytest.mark.skipif(not _HAS_GPU, reason="No GPU")),
]
_NATIVE_IDS = ["cpu", "cuda"]
_DTYPES = ["float32", "float16"]
f16, f32 = np.float16, np.float32


def bits(x):
    """Bit patterns, so that results can be compared exactly (and NaN == NaN)."""
    x = np.ascontiguousarray(x)
    return x.view(np.uint16 if x.dtype == f16 else np.uint32)


def ulp_diff(a, b):
    """Max distance in float16 ulps between two float16 arrays (NaNs must coincide)."""
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    ok = ~np.isnan(a)
    ia = a[ok].view(np.int16).astype(np.int32)
    ib = b[ok].view(np.int16).astype(np.int32)
    ia = np.where(ia < 0, -32768 - ia, ia)
    ib = np.where(ib < 0, -32768 - ib, ib)
    return int(np.max(np.abs(ia - ib), initial=0))


# --------------------------------------------------------------------------- numpy backend


def test_numpy_backend_float32_unchanged():
    """The float32 path of the numpy backend gives exactly what plain numpy float32 gives."""
    np.random.seed(0)
    a = np.random.randn(6, 8).astype(f32)
    b = np.random.rand(6, 8).astype(f32) + 0.5
    dev = nd.cpu_numpy()
    A, B = nd.array(a, device=dev), nd.array(b, device=dev)
    for got, want in [
        (A + B, a + b), (A * B, a * b), (A / B, a / b), (A + 0.1, a + f32(0.1)),
        (A * 3, a * f32(3)), (B ** 2.5, b ** f32(2.5)), (B ** A, b ** a),
        (A.maximum(B), np.maximum(a, b)), (A.maximum(0), np.maximum(a, f32(0))),
        ((A >= B), (a >= b).astype(f32)), ((A == A), np.ones_like(a)),
        (B.log(), np.log(b)), (A.exp(), np.exp(a)), (A.tanh(), np.tanh(a)),
        (A @ B.permute((1, 0)), a @ b.T), (A.sum(axis=1), a.sum(axis=1)),
        (A.max(axis=0), a.max(axis=0)),
    ]:
        assert got.dtype == "float32"
        np.testing.assert_array_equal(bits(got.numpy()), bits(want))


def test_numpy_backend_ieee_without_warnings():
    dev = nd.cpu_numpy()
    with np.errstate(all="raise"):  # any numpy warning inside the backend would raise
        A = nd.array(np.array([0.0, 1.0, -1.0, 60000.0], f32), device=dev)
        out = (A / 0.0).numpy()
        assert np.isnan(out[0]) and out[1] == np.inf and out[2] == -np.inf
        assert A.log().numpy()[0] == -np.inf
        h = A.astype("float16")
        assert ((h * 2.0).numpy()[3]) == np.inf
        assert nd.array(np.array([1e6], f32), device=dev).astype("float16").numpy()[0] == np.inf


def test_numpy_backend_rejects_mismatched_buffers():
    mod = nd.ndarray_backend_numpy
    a16, a32 = mod.Array(4, "float16"), mod.Array(4)
    with pytest.raises(ValueError, match="dtype mismatch"):
        mod.ewise_add(a16, a32, a32)
    with pytest.raises(ValueError, match="expected float16"):
        mod.from_numpy(np.zeros(4, f32), a16)
    with pytest.raises(ValueError, match="unsupported dtype"):
        mod.Array(4, "int8")


def test_float32_only_backend():
    """A backend without __dtypes__ (e.g. an old build) is float32-only, with clear errors."""
    fake = types.ModuleType("fake_backend")
    for name in ("from_numpy", "to_numpy", "fill", "compact", "scalar_add"):
        setattr(fake, name, getattr(nd.ndarray_backend_numpy, name))
    fake.Array = lambda size: nd.ndarray_backend_numpy.Array(size)  # no dtype argument
    dev = nd.BackendDevice("fake", fake)
    assert dev.dtypes == ("float32",) and not dev.supports_dtype("float16")
    A = nd.array(np.arange(3, dtype=f32), device=dev)
    assert A.dtype == "float32"
    np.testing.assert_array_equal(A.astype("float32").numpy(), np.arange(3, dtype=f32))
    with pytest.raises(TypeError, match="does not support dtype float16"):
        nd.array(np.ones(3, f16), device=dev)
    with pytest.raises(TypeError, match="does not support dtype float16"):
        A.astype("float16")


# --------------------------------------------------------------------------- cross-backend agreement


def _same_float16_results(device, fn, *inputs):
    """fn on the numpy backend and on `device` must give identical float16 bits."""
    ref = fn(*[nd.array(x, device=nd.cpu_numpy()) for x in inputs])
    got = fn(*[nd.array(x, device=device) for x in inputs])
    assert ref.dtype == got.dtype == "float16"
    return ref.numpy(), got.numpy()


EXACT_OPS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "max": lambda a, b: a.maximum(b),
    "ge": lambda a, b: a >= b,
    "gt": lambda a, b: a > b,
    "scalar_mul": lambda a, b: a * 1.3,
    "scalar_div": lambda a, b: a / 3.0,
    "scalar_add": lambda a, b: a + 0.1,
    "rsub": lambda a, b: 1 - a,
    "square": lambda a, b: a ** 2,
    "permute_compact": lambda a, b: a.permute((1, 0)).compact(),
    "slice_compact": lambda a, b: a[1:5, ::3].compact(),
    "broadcast": lambda a, b: a[:, 0:1].broadcast_to((6, 9)).compact(),
    "flip": lambda a, b: a.flip((0, 1)),
    "pad": lambda a, b: a.pad(((1, 2), (0, 1))),
    "reduce_max": lambda a, b: a.max(axis=1),
    "cast_roundtrip": lambda a, b: (a.astype("float32") * 1.001).astype("float16"),
}


@pytest.mark.parametrize("op", list(EXACT_OPS))
@pytest.mark.parametrize("device", _NATIVE, ids=_NATIVE_IDS)
def test_backends_bit_identical(op, device):
    np.random.seed(1)
    a = np.random.uniform(-3, 3, (6, 9)).astype(f16)
    b = np.random.uniform(0.5, 3, (6, 9)).astype(f16)
    ref, got = _same_float16_results(device, EXACT_OPS[op], a, b)
    np.testing.assert_array_equal(bits(got), bits(ref))


APPROX_OPS = {
    "exp": lambda a, b: a.exp(),
    "log": lambda a, b: b.log(),
    "tanh": lambda a, b: a.tanh(),
    "pow": lambda a, b: b ** a,
    "matmul_tiled": lambda a, b: a @ b.permute((1, 0)),  # (16, 24) @ (24, 16): tiled on cpu
    "matmul_naive": lambda a, b: a[:, :7] @ b.permute((1, 0))[:7, :5],
    "sum": lambda a, b: a.sum(axis=0),
    "sum_all": lambda a, b: a.sum(),
}


@pytest.mark.parametrize("op", list(APPROX_OPS))
@pytest.mark.parametrize("device", _NATIVE, ids=_NATIVE_IDS)
def test_backends_agree_within_one_ulp(op, device):
    """Transcendentals (different libm) and float32 accumulation order may differ by one
    float16 rounding, never more."""
    np.random.seed(2)
    a = np.random.uniform(-2, 2, (16, 24)).astype(f16)
    b = np.random.uniform(0.5, 2, (16, 24)).astype(f16)
    ref, got = _same_float16_results(device, APPROX_OPS[op], a, b)
    assert ulp_diff(got, ref) <= 1


# --------------------------------------------------------------------------- dtype across devices


@pytest.mark.parametrize("src", _DEVICES, ids=_IDS)
@pytest.mark.parametrize("dst", _DEVICES, ids=_IDS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_to_device_keeps_dtype(src, dst, dtype):
    x = np.random.randn(3, 5).astype(dtype)
    A = nd.array(x, device=src)
    B = A.to(dst)
    assert B.device == dst and B.dtype == dtype
    np.testing.assert_array_equal(B.numpy(), x)
    C = nd.array(A, device=dst, dtype="float32")
    assert C.dtype == "float32"
    np.testing.assert_array_equal(C.numpy(), x.astype(f32))


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_creation_helpers_respect_dtype(device, dtype):
    np.random.seed(3)
    for A in (
        device.rand(3, 4, dtype=dtype),
        device.randn(3, 4, dtype=dtype),
        device.one_hot(5, np.array([0, 3, 4]), dtype=dtype),
        nd.empty((2, 2), dtype=dtype, device=device),
        nd.full((2, 2), 0.1, dtype=dtype, device=device),
        nd.array([[1.0, 2.0]], dtype=dtype, device=device),
    ):
        assert A.dtype == dtype and A.numpy().dtype == dtype
    np.testing.assert_array_equal(
        nd.full((2,), 0.1, dtype=dtype, device=device).numpy(), np.full(2, f32(0.1)).astype(dtype)
    )


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_module_level_helpers(device):
    x = np.array([[1.5, -2.0], [np.inf, 3.0]], f32)
    A = nd.array(x, device=device)
    H = nd.astype(A, "float16")
    assert H.dtype == "float16" and nd.astype(H, "float16", copy=False) is H
    np.testing.assert_array_equal(nd.isfinite(H).numpy(), np.isfinite(x).astype(f16))
    assert not nd.all_finite(H) and nd.all_finite(H[0:1, :])
    s = nd.sum(nd.astype(A[0:1, :], "float16"), axis=1, dtype="float32")
    assert s.dtype == "float32" and s.numpy()[0] == -0.5


# --------------------------------------------------------------------------- AMP helpers


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_isfinite_and_all_finite(device, dtype):
    x = np.array([0.0, -1.5, 65504.0, -65504.0, np.inf, -np.inf, np.nan, 6e-8], dtype)
    A = nd.array(x, device=device)
    mask = A.isfinite()
    assert mask.dtype == dtype
    np.testing.assert_array_equal(mask.numpy(), np.isfinite(x).astype(dtype))
    assert not A.all_finite()
    assert A[0:4].all_finite()  # largest float16 values are still finite
    assert A[0:3].all_finite() and not A[4:5].all_finite()
    assert not A[5:6].all_finite() and not A[6:7].all_finite()
    # non-compact views and multi-dimensional arrays
    B = nd.array(np.ones((4, 6), dtype), device=device)
    assert B.permute((1, 0))[::2, 1:].all_finite()
    B[2, 3] = float("nan")
    assert not B.all_finite() and B[0:2, :].all_finite()
    assert B.permute((1, 0))[3:4, 2:3].isfinite().numpy().item() == 0
    assert nd.empty((0,), dtype=dtype, device=device).all_finite()


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_float16_overflow_detected(device):
    """The loss scaling pattern: scaling float16 gradients can overflow to inf."""
    g = nd.array(np.array([1.0, 300.0, -2.0], f16), device=device)
    assert g.all_finite()
    assert (g * 100.0).all_finite()  # 30000 < 65504
    assert not (g * 1000.0).all_finite()  # 300000 overflows to inf
    assert not (g * 0.0 / 0.0).all_finite()  # NaN


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_sum_output_dtype(device):
    """float16 input, float32 output: the total is not squeezed back into float16 range."""
    x = np.full((4, 1000), 100.0, f16)
    A = nd.array(x, device=device)
    s16 = A.sum(axis=1)
    assert s16.dtype == "float16" and np.all(np.isinf(s16.numpy()))  # 100000 > 65504
    s32 = A.sum(axis=1, dtype="float32")
    assert s32.dtype == "float32"
    np.testing.assert_array_equal(s32.numpy(), np.full(4, 100000.0, f32))
    total = A.sum(dtype=np.float32)
    assert total.dtype == "float32" and total.numpy().item() == 400000.0
    k = A.sum(axis=0, keepdims=True, dtype="float32")
    assert k.shape == (1, 1000) and k.dtype == "float32"
    # same dtype requested: identical to the plain call
    np.testing.assert_array_equal(A.sum(axis=1, dtype="float16").numpy(), s16.numpy())
    assert A.dtype == "float16"  # input untouched


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_astype_precision(device):
    """float32 -> float16 -> float32 loses precision the same way numpy does."""
    x = np.array([1 / 3, 1e-5, 1e-8, 70000.0, -0.1], f32)
    A = nd.array(x, device=device)
    with np.errstate(over="ignore"):
        expected = x.astype(f16)
    H = A.astype("float16")
    np.testing.assert_array_equal(bits(H.numpy()), bits(expected))
    np.testing.assert_array_equal(H.astype("float32").numpy(), expected.astype(f32))
