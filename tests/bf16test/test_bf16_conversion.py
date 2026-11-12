"""bfloat16 storage in the NDArray backends (numpy, cpu, cuda): conversions.

bfloat16 is the upper half of a float32 (8-bit exponent, 7-bit mantissa). The
reference for every conversion is ml_dtypes, the bfloat16 implementation numpy
code uses (and what NDArray.numpy() returns for bfloat16 arrays).
"""
import sys

sys.path.append("./python")
import numpy as np
import pytest

ml_dtypes = pytest.importorskip("ml_dtypes")
from needle.backend_ndarray import ndarray as nd

bf16 = ml_dtypes.bfloat16
f16, f32 = np.float16, np.float32
_DEVICES = [
    nd.cpu_numpy(),
    nd.cpu(),
    pytest.param(nd.cuda(), marks=pytest.mark.skipif(not nd.cuda().enabled(), reason="No GPU")),
]
_IDS = ["numpy", "cpu", "cuda"]


def bits(x):
    return np.ascontiguousarray(x).view(np.uint16 if x.dtype.itemsize == 2 else np.uint32)


def all_bf16():
    """Every one of the 65536 bfloat16 bit patterns."""
    return np.arange(65536, dtype=np.uint32).astype(np.uint16).view(bf16)


# --------------------------------------------------------------------------- dtype plumbing


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_backend_advertises_bfloat16(device):
    assert "bfloat16" in device.dtypes and device.supports_dtype("bfloat16")
    assert device.supports_dtype(bf16)


def test_normalize_dtype():
    for d in ("bfloat16", bf16, np.dtype(bf16)):
        assert nd.normalize_dtype(d) == "bfloat16"
    assert nd.numpy_dtype("bfloat16") == np.dtype(bf16)
    assert nd.numpy_dtype("float16") == np.dtype(f16)
    with pytest.raises(TypeError):
        nd.normalize_dtype("bfloat8")


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_round_trip_and_dtype_inference(device):
    x = np.random.randn(3, 5).astype(bf16)
    A = nd.array(x, device=device)  # bfloat16 numpy input keeps its dtype
    assert A.dtype == "bfloat16" and A.itemsize == 2
    out = A.numpy()
    assert out.dtype == np.dtype(bf16)
    np.testing.assert_array_equal(bits(out), bits(x))
    assert nd.array(A, device=device).dtype == "bfloat16"  # copies keep it
    assert nd.array(x.astype(f32), device=device).dtype == "float32"  # unchanged default
    C = nd.array(x.astype(f32), device=device, dtype="bfloat16")
    np.testing.assert_array_equal(bits(C.numpy()), bits(x))
    D = nd.array(x.astype(f32), device=device, dtype=bf16)  # ml_dtypes type as dtype
    assert D.dtype == "bfloat16"


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_creation_helpers(device):
    np.random.seed(0)
    for A in (
        nd.empty((2, 3), dtype="bfloat16", device=device),
        nd.full((2, 3), 0.1, dtype="bfloat16", device=device),
        device.rand(2, 3, dtype="bfloat16"),
        device.randn(2, 3, dtype="bfloat16"),
        device.one_hot(4, np.array([0, 3]), dtype="bfloat16"),
        nd.array([[1.0, 2.5]], dtype="bfloat16", device=device),
    ):
        assert A.dtype == "bfloat16" and A.numpy().dtype == np.dtype(bf16)
    full = nd.full((3,), 0.1, dtype="bfloat16", device=device).numpy()
    np.testing.assert_array_equal(bits(full), bits(np.full(3, f32(0.1)).astype(bf16)))


# --------------------------------------------------------------------------- conversions


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_bf16_to_float32_exhaustive(device):
    """Widening is exact for every one of the 65536 bit patterns."""
    x = all_bf16()
    out = nd.array(x, device=device).astype("float32").numpy()
    expected = x.astype(f32)
    np.testing.assert_array_equal(np.isnan(out), np.isnan(expected))
    ok = ~np.isnan(expected)
    np.testing.assert_array_equal(bits(out[ok]), bits(expected[ok]))


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_float32_to_bf16_rounding(device):
    """Round to nearest even, ties, overflow to inf, subnormals, signed zero."""
    np.random.seed(1)
    grid = all_bf16().astype(f32)
    grid = np.sort(grid[np.isfinite(grid)])
    mids = ((grid[:-1].astype(np.float64) + grid[1:]) / 2).astype(f32)  # exact ties
    special = np.array(
        [0.0, -0.0, np.inf, -np.inf, 3.3895314e38, 3.39e38, f32(3.4028235e38), -3.4e38,
         1e-38, 1e-40, 1e-45, -1e-45, 1.0 + 2 ** -8, 1.0 + 3 * 2 ** -9, 65504.0, 1e-5],
        dtype=f32,
    )
    rand = (np.random.randn(300000) * 10.0 ** np.random.uniform(-40, 38, 300000)).astype(f32)
    x = np.concatenate([grid, np.nextafter(grid, f32(np.inf)), np.nextafter(grid, f32(-np.inf)),
                        mids, special, rand]).astype(f32)
    out = nd.array(x, device=device).astype("bfloat16").numpy()
    with np.errstate(over="ignore"):
        expected = x.astype(bf16)
    np.testing.assert_array_equal(np.isnan(out.astype(f32)), np.isnan(expected.astype(f32)))
    np.testing.assert_array_equal(bits(out), bits(expected))


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_nan_stays_nan(device):
    """NaNs whose payload sits only in the dropped low bits must not round to inf."""
    payloads = np.array([0x7F800001, 0xFF800001, 0x7FC00000, 0x7F80FFFF, 0x7FFFFFFF],
                        dtype=np.uint32).view(f32)
    out = nd.array(payloads, device=device).astype("bfloat16").numpy().astype(f32)
    assert np.isnan(out).all()


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_float16_bf16_casts_go_through_float32(device):
    """float16 <-> bfloat16 equals widening to float32 and rounding once."""
    h = np.arange(65536, dtype=np.uint32).astype(np.uint16).view(f16)
    h = h[np.isfinite(h)]
    out = nd.array(h, device=device).astype("bfloat16").numpy()
    np.testing.assert_array_equal(bits(out), bits(h.astype(f32).astype(bf16)))
    b = all_bf16()
    b = b[np.isfinite(b.astype(f32))]
    with np.errstate(over="ignore"):
        expected = b.astype(f32).astype(f16)
    out = nd.array(b, device=device).astype("float16").numpy()
    np.testing.assert_array_equal(bits(out), bits(expected))


def test_range_is_float32s():
    """bfloat16 keeps float32's exponent range: values that over/underflow float16 survive."""
    x = np.array([1e5, 3e38, 1e-8, 1e-30], f32)
    A = nd.array(x, device=nd.cpu())
    np.testing.assert_allclose(A.astype("bfloat16").numpy().astype(f32), x, rtol=2 ** -8)
    with np.errstate(over="ignore", under="ignore"):
        h = A.astype("float16").numpy()
    assert np.isinf(h[0]) and np.isinf(h[1]) and h[3] == 0  # float16 cannot hold them


# --------------------------------------------------------------------------- transfer details


def test_native_transfer_uses_uint16_bit_patterns():
    """C++ exchanges bfloat16 as uint16 (numpy has no built-in bfloat16); NDArray
    attaches the ml_dtypes type."""
    A = nd.array(np.array([1.0, -2.5], f32), device=nd.cpu(), dtype="bfloat16")
    raw = A.device.to_numpy(A._handle, A.shape, A.strides, A._offset)
    assert raw.dtype == np.uint16
    np.testing.assert_array_equal(raw, np.array([1.0, -2.5], f32).astype(bf16).view(np.uint16))
    h = nd.cpu().Array(2, "bfloat16")
    nd.cpu().from_numpy(np.array([3.0, 4.0], bf16), h)  # an ml_dtypes array is accepted too
    with pytest.raises(ValueError, match="expected bfloat16"):
        nd.cpu().from_numpy(np.array([3.0, 4.0], f16), h)


def test_missing_ml_dtypes_gives_clear_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ml_dtypes":
            raise ImportError("No module named 'ml_dtypes'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="pip install ml_dtypes"):
        nd.bfloat16_type()
    with pytest.raises(ImportError, match="pip install ml_dtypes"):
        nd.ndarray_backend_numpy.Array(4, "bfloat16")
    # the string name still normalizes, and other dtypes are unaffected
    assert nd.normalize_dtype("bfloat16") == "bfloat16"
    A = nd.array(np.ones(3, f16), device=nd.cpu())
    np.testing.assert_array_equal((A * 2.0).numpy(), np.full(3, 2.0, f16))
