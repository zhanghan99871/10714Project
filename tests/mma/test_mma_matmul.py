"""Tensor core (MMA) matmul for float16 / bfloat16 on CUDA.

Needs a GPU: float16 MMA needs compute capability 7.0+, bfloat16 8.0+ (tests for a
dtype the GPU cannot run on the tensor cores are skipped). The scalar kernel is
the reference and is selected with set_mma_enabled(False).
"""
import os
import subprocess
import sys
import time

sys.path.append("./python")
sys.path.append("./apps")
import numpy as np
import pytest

from needle.backend_ndarray import ndarray as nd

CUDA = nd.cuda()
pytestmark = pytest.mark.skipif(not CUDA.enabled(), reason="No GPU")
f16, f32 = np.float16, np.float32


def _bf16():
    ml_dtypes = pytest.importorskip("ml_dtypes")
    return ml_dtypes.bfloat16


def np_type(dtype):
    return f16 if dtype == "float16" else _bf16()


def require_mma(dtype):
    if not CUDA.mma_available(dtype):
        pytest.skip(f"no tensor core matmul for {dtype} on this GPU / build")


@pytest.fixture
def mma_switch():
    """Set the MMA switch for a test and restore it afterwards."""
    before = CUDA.mma_enabled()
    yield CUDA.set_mma_enabled
    CUDA.set_mma_enabled(before)


# precision (significant bits) and minimum normal exponent of the half types
_FORMAT = {"float16": (11, -14), "bfloat16": (8, -126)}


def ulp(x, dtype):
    """Unit in the last place of the half precision type at |x| (subnormals included)."""
    p, emin = _FORMAT[dtype]
    x = np.abs(np.asarray(x, np.float64))
    e = np.maximum(np.floor(np.log2(np.where(x > 0, x, 1.0))), emin)
    return np.exp2(e - (p - 1))


def accumulation_bound(a, b, k, kernel):
    """Worst-case float32 accumulation error of a @ b, as a multiple of sum |a_ik b_kj|.

    tensor cores: products are exact, each 16-wide step aligns the partial sum to its
      largest term and truncates (it does not round to nearest), ~2^-23 relative per
      step; 2^-21 per step keeps a 4x safety margin.
    scalar kernel: one rounded fused multiply-add per term, <= k * 2^-24 relative.
    Bug-sized errors (wrong elements, missing tiles) are O(sum |a b|), orders above this.
    """
    magnitude = np.abs(a.astype(np.float64)) @ np.abs(b.astype(np.float64))
    steps = -(-k // 16)
    rel = {"mma": steps * 2.0 ** -21, "scalar": k * 2.0 ** -24}[kernel]
    return rel * magnitude


def check_within(out, ref, bound, dtype, what):
    """|out - ref| <= bound everywhere; on failure report the worst element."""
    err = np.abs(out.astype(np.float64) - ref.astype(np.float64))
    excess = err / bound
    worst = np.unravel_index(np.argmax(excess), excess.shape)
    assert excess.max() <= 1.0, (
        f"{what}: worst element {worst}: got {out[worst]}, expected {ref[worst]}, "
        f"|err| = {err[worst]:.3g} = {err[worst] / ulp(ref[worst], dtype):.1f} ulp, "
        f"bound {bound[worst]:.3g} ({excess.max():.2f}x the bound)"
    )
    return float(excess.max())


def run(a, b, dtype, mma, switch):
    switch(mma)
    out = nd.array(a, device=CUDA, dtype=dtype) @ nd.array(b, device=CUDA, dtype=dtype)
    assert out.dtype == dtype
    return out.numpy()


def exact(a, b, dtype):
    """float64 product rounded once to the half precision type."""
    return (a.astype(np.float64) @ b.astype(np.float64)).astype(f32).astype(np_type(dtype))


SHAPES = [
    ((1, 1), (1, 1)),
    ((5, 7), (7, 3)),
    ((16, 16), (16, 16)),
    ((15, 17), (17, 15)),
    ((64, 64), (64, 64)),
    ((65, 63), (63, 65)),
    ((257, 130), (130, 300)),
    ((32, 2048), (2048, 16)),  # long inner dimension
    ((3, 40, 24), (3, 24, 33)),  # batched
    ((2, 3, 20, 24), (24, 18)),  # shared right operand (stride 0)
    ((24, 18), (2, 3, 18, 20)),  # shared left operand
    ((2, 1, 9, 12), (1, 3, 12, 7)),  # partial broadcast
]


# --------------------------------------------------------------------------- switch / availability


def test_switch_and_availability(mma_switch):
    assert isinstance(CUDA.mma_available("float16"), bool)
    assert isinstance(CUDA.mma_available("bfloat16"), bool)
    assert CUDA.mma_available("float32") is False  # float32 never uses the tensor cores
    mma_switch(False)
    assert CUDA.mma_enabled() is False
    mma_switch(True)
    assert CUDA.mma_enabled() is True


def test_environment_variable_disables_mma():
    code = ("import sys; sys.path.append('./python'); "
            "from needle.backend_ndarray import ndarray as nd; print(nd.cuda().mma_enabled())")
    for value, expected in (("0", "False"), ("1", "True")):
        env = dict(os.environ, NEEDLE_CUDA_MMA=value)
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
        assert out.stdout.strip().splitlines()[-1] == expected, out.stderr


# --------------------------------------------------------------------------- correctness


@pytest.mark.parametrize("sa,sb", SHAPES)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_mma_exact_on_integer_data(sa, sb, dtype, mma_switch):
    """Small integers: every product and partial sum is an exactly representable
    integer, so ANY accumulation order or truncation gives the exact result.
    A mismatch here is an indexing / tiling bug, not rounding."""
    require_mma(dtype)
    rng = np.random.default_rng(0)
    k = sa[-1]
    hi = 3 if dtype == "float16" else 1  # |sum| <= hi^2 * k fits the type's integers
    assert hi * hi * k <= (2048 if dtype == "float16" else 256)
    a = rng.integers(-hi, hi + 1, sa).astype(f32).astype(np_type(dtype))
    b = rng.integers(-hi, hi + 1, sb).astype(f32).astype(np_type(dtype))
    out = run(a, b, dtype, True, mma_switch)
    ref = exact(a, b, dtype)
    mismatch = np.argwhere(out.astype(f32) != ref.astype(f32))
    assert mismatch.size == 0, f"{len(mismatch)} wrong elements, first at {tuple(mismatch[0])}"


@pytest.mark.parametrize("sa,sb", SHAPES)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_mma_matches_exact_product(sa, sb, dtype, mma_switch):
    """Random data: error within one rounding of the output plus the tensor cores'
    float32 accumulation error (they truncate while accumulating, so near-zero
    results of large cancelling sums can be many ulps off -- see accumulation_bound)."""
    require_mma(dtype)
    np.random.seed(0)
    a = np.random.uniform(-1, 1, sa).astype(f32).astype(np_type(dtype))
    b = np.random.uniform(-1, 1, sb).astype(f32).astype(np_type(dtype))
    out = run(a, b, dtype, True, mma_switch)
    ref = exact(a, b, dtype)
    bound = ulp(ref.astype(f32), dtype) + accumulation_bound(a, b, sa[-1], "mma")
    check_within(out, ref, bound, dtype, "tensor cores vs exact")


@pytest.mark.parametrize("sa,sb", SHAPES)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_mma_agrees_with_scalar_kernel(sa, sb, dtype, mma_switch):
    require_mma(dtype)
    np.random.seed(1)
    a = np.random.uniform(-1, 1, sa).astype(f32).astype(np_type(dtype))
    b = np.random.uniform(-1, 1, sb).astype(f32).astype(np_type(dtype))
    on, off = run(a, b, dtype, True, mma_switch), run(a, b, dtype, False, mma_switch)
    k = sa[-1]
    bound = (2 * ulp(off.astype(f32), dtype)
             + accumulation_bound(a, b, k, "mma") + accumulation_bound(a, b, k, "scalar"))
    check_within(on, off, bound, dtype, "tensor cores vs scalar kernel")


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_batched_equals_per_slice_and_deterministic(dtype, mma_switch):
    """2D and batched products run the same kernel: bit-identical slices."""
    require_mma(dtype)
    mma_switch(True)
    np.random.seed(2)
    t = np_type(dtype)
    a = np.random.randn(4, 70, 40).astype(f32).astype(t)
    b = np.random.randn(4, 40, 90).astype(f32).astype(t)
    batched = (nd.array(a, device=CUDA) @ nd.array(b, device=CUDA)).numpy()
    again = (nd.array(a, device=CUDA) @ nd.array(b, device=CUDA)).numpy()
    np.testing.assert_array_equal(batched.view(np.uint16), again.view(np.uint16))
    for z in range(4):
        single = (nd.array(a[z], device=CUDA) @ nd.array(b[z], device=CUDA)).numpy()
        np.testing.assert_array_equal(batched[z].view(np.uint16), single.view(np.uint16))


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_large_batch_and_views(dtype, mma_switch):
    """More batch entries than the grid's z limit (65535), and non-compact inputs."""
    require_mma(dtype)
    mma_switch(True)
    np.random.seed(3)
    t = np_type(dtype)
    a = np.random.randn(70000, 2, 3).astype(f32).astype(t)
    b = np.random.randn(3, 2).astype(f32).astype(t)
    out = (nd.array(a, device=CUDA) @ nd.array(b, device=CUDA)).numpy()
    ref = exact(a, b, dtype)
    check_within(out, ref, ulp(ref.astype(f32), dtype) + accumulation_bound(a, b, 3, "mma"),
                 dtype, "large batch")
    x = np.random.randn(2, 40, 30).astype(f32).astype(t)
    X = nd.array(x, device=CUDA)
    V = X.permute((0, 2, 1))[:, 5:25, :]  # (2, 20, 40), non-compact
    W = X[:, :, 3:23]  # (2, 40, 20), non-compact
    va, wb = np.swapaxes(x, 1, 2)[:, 5:25, :], x[:, :, 3:23]
    ref = (va.astype(np.float64) @ wb.astype(np.float64)).astype(f32).astype(t)
    check_within((V @ W).numpy(), ref, ulp(ref.astype(f32), dtype) + accumulation_bound(va, wb, 40, "mma"),
                 dtype, "views")


def test_float32_does_not_use_tensor_cores(mma_switch):
    np.random.seed(4)
    a, b = np.random.randn(65, 70).astype(f32), np.random.randn(70, 33).astype(f32)
    on = run(a, b, "float32", True, mma_switch)
    off = run(a, b, "float32", False, mma_switch)
    np.testing.assert_array_equal(on, off)


# --------------------------------------------------------------------------- end to end


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_transformer_training_step_with_mma(dtype, mma_switch):
    require_mma(dtype)
    import needle as ndl
    import needle.nn as nn

    res = {}
    for mma in (False, True):
        mma_switch(mma)
        np.random.seed(0)
        model = nn.Transformer(32, 64, 2, num_head=4, dim_head=8, causal=True,
                               device=ndl.cuda(), batch_first=True)
        x = ndl.Tensor(np.random.randn(4, 16, 32).astype(f32), device=ndl.cuda(), requires_grad=False)
        with ndl.autocast(dtype=dtype):
            out, _ = model(x)
            loss = ndl.ops.summation(out * out) / (4 * 16 * 32)
        loss.backward()
        grads = np.concatenate([p.grad.numpy().ravel() for p in model.parameters()])
        assert np.isfinite(grads).all()
        res[mma] = (loss.numpy(), grads)
    np.testing.assert_allclose(res[True][0], res[False][0], rtol=2e-2)
    g_on, g_off = res[True][1], res[False][1]
    assert np.linalg.norm(g_on - g_off) / np.linalg.norm(g_off) < 5e-2


# --------------------------------------------------------------------------- speed


def _time_matmul(A, B, repeats=5):
    (A @ B)[0:1, 0:1].numpy()  # warm-up; reading back synchronizes
    start = time.perf_counter()
    for _ in range(repeats):
        out = A @ B
    out[0:1, 0:1].numpy()
    return (time.perf_counter() - start) / repeats


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_mma_is_faster(dtype, mma_switch):
    require_mma(dtype)
    np.random.seed(5)
    t = np_type(dtype)
    A = nd.array(np.random.randn(1024, 1024).astype(f32).astype(t), device=CUDA)
    B = nd.array(np.random.randn(1024, 1024).astype(f32).astype(t), device=CUDA)
    mma_switch(False)
    scalar = _time_matmul(A, B)
    mma_switch(True)
    mma = _time_matmul(A, B)
    print(f"\n{dtype} 1024^3 matmul: scalar {scalar * 1e3:.2f} ms, tensor cores {mma * 1e3:.2f} ms "
          f"({scalar / mma:.1f}x)")
    assert mma < scalar
