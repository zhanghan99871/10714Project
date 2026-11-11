"""Causal attention mask that is safe in float16."""
import sys

sys.path.append("./python")
import numpy as np
import pytest

import needle as ndl
import needle.nn as nn
from needle.backend_ndarray import ndarray as nd

_DEVICES = [
    ndl.cpu_numpy(),
    ndl.cpu(),
    pytest.param(ndl.cuda(), marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU")),
]
_IDS = ["numpy", "cpu", "cuda"]
f16, f32 = np.float16, np.float32


def T(x, device, dtype=None, requires_grad=True):
    return ndl.Tensor(x, device=device, dtype=dtype, requires_grad=requires_grad)


def old_create_causal_mask(self, i, j, device, dtype="float32"):
    """The previous mask (-finfo(float32).max), converted to the logits' dtype."""
    with np.errstate(over="ignore"):
        mask = (-np.finfo(np.float32).max * np.triu(np.ones((1, 1, i, j), f32), j - i + 1)).astype(dtype)
    return nd.array(mask, device=device, dtype=dtype)


def attention(device, dtype, q_len, kv_len, mask_fn=None, seed=0, dropout=0.0, scale=1.0):
    layer = nn.MultiHeadAttention(causal=True, dropout=dropout, device=device)
    if mask_fn is not None:
        layer.create_causal_mask = mask_fn.__get__(layer)
    rs = np.random.RandomState(seed)
    Q = T(rs.randn(2, 3, q_len, 8) * scale, device, dtype=dtype)
    K = T(rs.randn(2, 3, kv_len, 8) * scale, device, dtype=dtype)
    V = T(rs.randn(2, 3, kv_len, 8), device, dtype=dtype)
    out, probs = layer(Q, K, V)
    out.sum().backward()
    return out, probs, (Q.grad, K.grad, V.grad)


# --------------------------------------------------------------------------- the mask itself


@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_mask_values(dtype, device):
    layer = nn.MultiHeadAttention(causal=True, device=device)
    for i, j in [(4, 4), (3, 5), (5, 3)]:
        m = layer.create_causal_mask(i, j, device, dtype=dtype)
        assert m.dtype == dtype and m.shape == (1, 1, i, j)
        v = m.numpy()
        assert np.isfinite(v).all()
        value = nn.MultiHeadAttention.causal_mask_value(dtype)
        expected = np.triu(np.ones((1, 1, i, j)), j - i + 1) * value
        np.testing.assert_array_equal(v, expected.astype(dtype))
        # headroom: adding a large negative logit does not overflow to -inf
        assert np.isfinite(np.array(value, dtype) + np.array(-1000.0, dtype))


def test_mask_default_dtype_is_float32():
    layer = nn.MultiHeadAttention(causal=True, device=ndl.cpu())
    assert layer.create_causal_mask(3, 3, ndl.cpu()).dtype == "float32"


# --------------------------------------------------------------------------- behaviour


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_float32_results_bit_identical_to_previous_mask(device):
    new = attention(device, "float32", 6, 6)
    old = attention(device, "float32", 6, 6, mask_fn=old_create_causal_mask)
    for a, b in zip([new[0], new[1], *new[2]], [old[0], old[1], *old[2]]):
        np.testing.assert_array_equal(a.numpy(), b.numpy())


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_float16_fully_masked_rows_stay_finite(device):
    """More queries than keys: the first rows see no key at all. The old mask is
    -inf in float16 and turns those rows (and every gradient) into NaN."""
    out, probs, grads = attention(device, "float16", 8, 6)
    assert out.dtype == "float16"
    assert np.isfinite(out.numpy()).all() and all(np.isfinite(g.numpy()).all() for g in grads)
    np.testing.assert_allclose(probs.numpy()[:, :, :2].astype(f32), 1 / 6, rtol=1e-3)  # uniform
    old_out, _, old_grads = attention(device, "float16", 8, 6, mask_fn=old_create_causal_mask)
    assert np.isnan(old_out.numpy()).any() and np.isnan(old_grads[0].numpy()).any()


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_float16_attention_matches_float32(device):
    o16, p16, g16 = attention(device, "float16", 7, 7)
    o32, p32, g32 = attention(device, "float32", 7, 7)
    np.testing.assert_allclose(o16.numpy().astype(f32), o32.numpy(), rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(p16.numpy().astype(f32), p32.numpy(), atol=5e-3)
    assert np.all(p16.numpy()[np.triu(np.ones((7, 7), bool), 1)[None, None].repeat(2, 0).repeat(3, 1)] == 0)
    for a, b in zip(g16, g32):
        np.testing.assert_allclose(a.numpy().astype(f32), b.numpy(), rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_float16_inputs_into_float32_module(device):
    """The mask follows the logits' dtype: float16 activations through a module
    built with dtype="float32" no longer hit a mixed-dtype error."""
    out, _, _ = attention(device, "float16", 5, 5)
    assert out.dtype == "float16"


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_large_negative_logits_with_mask(device):
    """Logits around -100 plus the mask stay finite in float16 (no -inf)."""
    layer = nn.MultiHeadAttention(causal=True, device=device)
    q = T(np.full((1, 1, 4, 4), 10.0), device, dtype="float16")
    k = T(np.full((1, 1, 4, 4), -10.0), device, dtype="float16")  # q.k / 2 = -200
    out, probs = layer(q, k, k)
    assert np.isfinite(out.numpy()).all() and np.isfinite(probs.numpy()).all()
    np.testing.assert_allclose(probs.numpy()[0, 0, -1].astype(f32), 0.25, rtol=1e-3)


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_causal_transformer_under_autocast_and_in_float16(device):
    np.random.seed(0)
    x = np.random.randn(2, 9, 16)
    results = {}
    for mode in ("float32", "autocast", "float16"):
        np.random.seed(1)
        dt = "float16" if mode == "float16" else "float32"
        model = nn.Transformer(16, 32, 2, num_head=2, dim_head=8, causal=True,
                               device=device, dtype=dt, batch_first=True)
        X = T(x, device, dtype=dt)
        with ndl.autocast(enabled=mode == "autocast"):
            out, _ = model(X)
        out.sum().backward()
        assert np.isfinite(out.numpy()).all()
        assert all(np.isfinite(p.grad.numpy()).all() for p in model.parameters())
        results[mode] = out.numpy().astype(f32)
    np.testing.assert_allclose(results["autocast"], results["float32"], rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(results["float16"], results["float32"], rtol=5e-2, atol=5e-2)
