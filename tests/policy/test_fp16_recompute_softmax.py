"""The "fp16_none" segment policy (float16 + recompute), the fused ops.softmax and
the half-precision attention softmax option (Transformer.half_softmax)."""
import gc
import sys

sys.path.append("./python")
sys.path.append("./apps")
sys.path.append("./tests/policy")
import numpy as np
import pytest

import needle as ndl
import needle.nn as nn
from needle import ops
from needle.backend_ndarray import ndarray as nd
from needle.nn.nn_transformer import MultiHeadAttention
from needle.policy import measure_step
from test_segment_policies import LAYERS, assert_same, batch, loss_and_grads, make_lm, manual_forward

_HAS_GPU = ndl.cuda().enabled()
_DEVICES = [
    ndl.cpu_numpy(),
    ndl.cpu(),
    pytest.param(ndl.cuda(), marks=pytest.mark.skipif(not _HAS_GPU, reason="No GPU")),
]
_IDS = ["numpy", "cpu", "cuda"]
f32 = np.float32


def np_softmax(x, axis):
    e = np.exp(x - x.max(axis, keepdims=True))
    return e / e.sum(axis, keepdims=True)


def np_softmax_grad(p, g, axis):
    return p * (g - (g * p).sum(axis, keepdims=True))


def as_f32(t):
    return np.asarray(t.numpy(), dtype=f32)


# --------------------------------------------------------------------------- ops.softmax


@pytest.mark.parametrize("axis", [-1, 1, 3])
@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_softmax_matches_numpy(device, axis):
    rng = np.random.RandomState(0)
    x = (3 * rng.randn(2, 3, 5, 7)).astype(f32)
    g = rng.randn(2, 3, 5, 7).astype(f32)
    t = ndl.Tensor(x, device=device, requires_grad=True)
    p = ops.softmax(t, axis=axis)
    (p * ndl.Tensor(g, device=device)).sum().backward()
    ref = np_softmax(x, axis)
    assert p.dtype == "float32"
    np.testing.assert_allclose(p.numpy(), ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(t.grad.numpy(), np_softmax_grad(ref, g, axis), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype,tol", [("float16", 2e-3), ("bfloat16", 2e-2)])
@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_softmax_half_keeps_dtype(device, dtype, tol):
    rng = np.random.RandomState(1)
    x = (3 * rng.randn(4, 6, 9)).astype(f32)
    g = rng.randn(4, 6, 9).astype(f32)
    leaf = ndl.Tensor(x, device=device, requires_grad=True)
    h = leaf.astype(dtype, release_input=False)
    p = ops.softmax(h, axis=2)
    assert p.dtype == dtype
    (p * ndl.Tensor(g, device=device, dtype=dtype)).sum().backward()
    xh = as_f32(h)  # the rounded input
    ref = np_softmax(xh, 2)
    np.testing.assert_allclose(as_f32(p), ref, atol=tol)
    np.testing.assert_allclose(leaf.grad.numpy(), np_softmax_grad(ref, g, 2), atol=tol)


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_softmax_half_is_rounded_once(device):
    """float16 in, float16 out, float32 in between: every probability is the exact
    softmax of the float16 logits rounded once (within half a float16 ulp), not
    the result of rounding after each step."""
    x = (8 * np.random.RandomState(5).randn(64, 128)).astype(np.float16)
    x64 = x.astype(np.float64)
    e = np.exp(x64 - x64.max(1, keepdims=True))
    ref = e / e.sum(1, keepdims=True)
    p = ops.softmax(ndl.Tensor(x, device=device, dtype="float16"), axis=1).numpy().astype(np.float64)
    normal = ref > 1e-3
    assert np.max(np.abs(p - ref)[normal] / ref[normal]) <= 2.0 ** -11 * 1.01


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_softmax_fp16_gradient_of_long_flat_rows(device):
    """Rows of 512 nearly equal logits sum to ~512 after exp: the op-by-op softmax's
    division gradient would square that (inf in float16); the fused gradient is
    as accurate as float16 storage allows."""
    rng = np.random.RandomState(2)
    x = (0.01 * rng.randn(3, 512)).astype(f32)
    g = rng.randn(3, 512).astype(f32)
    leaf = ndl.Tensor(x, device=device, requires_grad=True)
    p = ops.softmax(leaf.astype("float16", release_input=False), axis=1)
    (p * ndl.Tensor(g, device=device, dtype="float16")).sum().backward()
    ref = np_softmax_grad(np_softmax(x, 1), g, 1)
    grad = leaf.grad.numpy()
    assert np.all(np.isfinite(grad))
    assert np.abs(grad - ref).max() <= 2e-2 * np.abs(ref).max()


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_softmax_large_half_logits(device):
    x = np.array([[60000.0, 59990.0, -60000.0], [-65000.0, -65000.0, -65000.0]], dtype=f32)
    p = ops.softmax(ndl.Tensor(x, device=device, dtype="float16"), axis=1)
    np.testing.assert_allclose(as_f32(p), np_softmax(x.astype(np.float16).astype(f32), 1), atol=1e-3)


def test_softmax_under_autocast_keeps_input_dtype():
    dev = ndl.cpu()
    x = np.random.RandomState(3).randn(2, 5).astype(f32)
    with ndl.autocast(dtype="float16"):
        assert ops.softmax(ndl.Tensor(x, device=dev, dtype="float16")).dtype == "float16"
        assert ops.softmax(ndl.Tensor(x, device=dev)).dtype == "float32"
        # the op-by-op building blocks are float32 ops under autocast
        assert ops.exp(ndl.Tensor(x, device=dev, dtype="float16")).dtype == "float32"


# --------------------------------------------------------------------------- "fp16_none"


def test_fp16_none_api():
    m = make_lm(ndl.cpu(), policies="fp32")
    tr = m.sequence_model
    tr.set_policies(["fp16_none", "fp32", "none"])
    assert tr.policies == ["fp16_none", "fp32", "none"] and tr.needs_loss_scaling
    tr.set_policies(["none", "fp32", "none"])
    assert not tr.needs_loss_scaling


@pytest.mark.parametrize("dropout", [0.0, 0.2])
@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_fp16_none_gradients_identical_to_fp16(device, dropout):
    """Recomputing replays the recorded float16 ops (and reuses the dropout masks):
    loss and gradients are bit-identical to keeping the float16 activations."""
    x, y = batch(device)
    for fp16_pol, none_pol in (("fp16", "fp16_none"),
                               (["fp16", "fp32", "fp16"], ["fp16_none", "fp32", "fp16_none"]),
                               (["fp16", "none", "fp16"], ["fp16", "none", "fp16_none"])):
        a = loss_and_grads(make_lm(device, policies=fp16_pol, dropout=dropout), x, y)
        b = loss_and_grads(make_lm(device, policies=none_pol, dropout=dropout), x, y)
        assert_same(a, b)


@pytest.mark.parametrize("release", [True, False])
@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_fp16_none_matches_manual_reference(device, release):
    x, y = batch(device)
    pol = ["fp16_none", "fp32", "fp16_none"]
    m = make_lm(device, policies=pol, fp16_release_inputs=release)
    ref = loss_and_grads(make_lm(device, fp16_release_inputs=release), x, y,
                         forward=lambda mm, xx: manual_forward(mm, xx, pol, release_inputs=release))
    assert_same(loss_and_grads(m, x, y), ref)


def test_fp16_none_runs_layer_once_and_recomputes_in_backward():
    dev = ndl.cpu()
    x, y = batch(dev)
    m = make_lm(dev, policies="fp16_none")
    calls = []
    for layer in m.sequence_model.layers.modules:
        fwd = layer.forward
        layer.forward = lambda h, _f=fwd: (calls.append(1), _f(h))[1]
    loss = nn.SoftmaxLoss()(m(x)[0], y)
    assert len(calls) == LAYERS  # Python code runs once; backward replays recorded ops
    loss.backward()
    assert len(calls) == LAYERS
    assert all(np.all(np.isfinite(p.grad.numpy())) for p in m.parameters())


def _held_after_forward(dev, x, policy, half_softmax=False):
    m = make_lm(dev, policies=policy, size=(64, 100, 256))
    m.sequence_model.half_softmax = half_softmax
    gc.collect()
    base = nd.memory_allocated(dev)
    out = m(x)[0]
    held = nd.memory_allocated(dev) - base
    del out, m
    gc.collect()
    return held


def test_fp16_none_holds_only_the_boundaries():
    dev = ndl.cpu()
    x, _ = batch(dev, vocab=100, shape=(32, 8))
    held = {p: _held_after_forward(dev, x, p) for p in ("fp32", "fp16", "none", "fp16_none")}
    # both keep only the boundaries and the dropout masks (leaves, reused by the
    # recompute), which are float16 in "fp16_none"
    assert held["fp16_none"] <= held["none"] < held["fp16"] < held["fp32"]


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_training_with_fp16_none_matches_fp16(device):
    """GradScaler training: steps under "fp16_none" give exactly the parameters
    and loss scales of the same steps under "fp16"."""
    results = []
    for schedule in (["fp16", "fp16", "fp16"], ["fp16_none", ["fp16", "fp16_none", "fp16"], "fp16_none"]):
        m = make_lm(device, policies="fp16", dropout=0.1)
        opt = ndl.optim.Adam(m.parameters(), lr=1e-2)
        sc = ndl.amp.GradScaler(init_scale=2.0 ** 8)
        scales = []
        for step, pol in enumerate(schedule):
            x, y = batch(device, seed=20 + step)
            opt.reset_grad()
            np.random.seed(step)
            m.set_policies(pol)
            sc.scale(nn.SoftmaxLoss()(m(x)[0], y)).backward()
            sc.step(opt)
            sc.update()
            scales.append(sc.get_scale())
        results.append(([p.numpy() for p in m.parameters()], scales))
    for a, b in zip(results[0][0], results[1][0]):
        np.testing.assert_array_equal(a, b)
    assert results[0][1] == results[1][1]


# --------------------------------------------------------------------------- half_softmax


def test_half_softmax_option_reaches_every_layer():
    from models import LanguageModel

    np.random.seed(0)
    m = LanguageModel(16, 30, 32, num_layers=LAYERS, seq_model="transformer", device=ndl.cpu(),
                      policies="fp16", half_softmax=True)
    tr = m.sequence_model
    attns = [layer.attn_layer.attn for layer in tr.layers.modules]
    assert tr.half_softmax and all(a.half_softmax for a in attns)
    tr.half_softmax = False
    assert not tr.half_softmax and not any(a.half_softmax for a in attns)
    assert not MultiHeadAttention().half_softmax  # off by default


def _record_probs_dtypes(m):
    seen = []
    for layer in m.sequence_model.layers.modules:
        attn = layer.attn_layer.attn
        fwd = attn.forward

        def spy(q, k, v, _fwd=fwd):
            result, probs = _fwd(q, k, v)
            seen.append(probs.dtype)
            return result, probs

        attn.forward = spy
    return seen


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_half_softmax_dtype_per_policy(device):
    x, _ = batch(device)
    pol = ["fp32", "fp16", "fp16_none"]
    for flag, expected in ((False, ["float32"] * 3), (True, ["float32", "float16", "float16"])):
        m = make_lm(device, policies=pol)
        m.sequence_model.half_softmax = flag
        seen = _record_probs_dtypes(m)
        m(x)
        assert seen == expected


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_half_softmax_leaves_float32_segments_unchanged(device):
    x, y = batch(device)
    for pol in ("fp32", "none", ["none", "fp32", "none"]):
        res = []
        for flag in (False, True):
            m = make_lm(device, policies=pol, dropout=0.1)
            m.sequence_model.half_softmax = flag
            res.append(loss_and_grads(m, x, y))
        assert_same(*res)


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_half_softmax_gradients(device):
    """With the half softmax, fp16 segments stay as close to float32 as without it,
    and "fp16_none" is still bit-identical to "fp16"."""
    x, y = batch(device)
    ref = loss_and_grads(make_lm(device, policies="fp32"), x, y)
    res = {}
    for flag in (False, True):
        for pol in ("fp16", "fp16_none"):
            m = make_lm(device, policies=pol)
            m.sequence_model.half_softmax = flag
            res[flag, pol] = loss_and_grads(m, x, y)
        assert_same(res[flag, "fp16"], res[flag, "fp16_none"])
    err = {}
    for flag in (False, True):
        loss, grads = res[flag, "fp16"]
        assert abs(float(loss) - float(ref[0])) < 1e-2
        assert all(np.all(np.isfinite(g)) for g in grads)
        # relative L2 error over all gradients (some have near-zero float32 values)
        err[flag] = np.sqrt(sum(((g - r) ** 2).sum() for g, r in zip(grads, ref[1]))
                            / sum((r ** 2).sum() for r in ref[1]))
    assert err[True] < 0.1 and err[True] <= 1.5 * err[False] + 0.01, err
    assert not np.array_equal(res[False, "fp16"][1][0], res[True, "fp16"][1][0])


def test_half_softmax_reduces_fp16_memory():
    dev = ndl.cpu()
    x, y = batch(dev, vocab=100, shape=(64, 4))
    held = {flag: _held_after_forward(dev, x, "fp16", flag) for flag in (False, True)}
    # the float32 softmax output is released by the cast to float16 for the next
    # matmul (and recomputed in backward), so after forward the gain is small; at
    # the peak of the step it is large
    assert held[True] < held[False]
    m = make_lm(dev, policies="fp16", size=(64, 100, 256))
    loss_fn = lambda: nn.SoftmaxLoss()(m(x)[0], y)
    peak = {}
    for flag in (False, True):
        m.sequence_model.half_softmax = flag
        peak[flag] = measure_step(m.sequence_model, loss_fn, "fp16", params=m.parameters(),
                                  warmup=0, repeats=1)["peak"]
    # 11% here (sequence length 64); the share grows with the attention matrices
    # (T^2): 26% at length 256
    assert peak[True] < 0.95 * peak[False]
    assert _held_after_forward(dev, x, "fp32", True) == _held_after_forward(dev, x, "fp32", False)
